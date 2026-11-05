"""Provider-agnostic tool-calling LLM client with a disk cache and a mock mode.

The agent speaks one neutral message format; adapters translate it per provider:

    {"role": "user", "text": "..."}
    {"role": "assistant", "text": "...", "tool_calls": [{"id", "name", "args", "signature"?}]}
    {"role": "tool", "results": [{"id", "name", "result": {...}}]}

Provider comes from LLM_PROVIDER in .env (gemini | anthropic | openai).
Each real reply is stored in cache/llm/<sha256>.json, keyed on everything that
affects it, so re-running the same conversation costs nothing.
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

from .mock_llm import mock_reply

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

# Cheapest sensible default per provider (override with LLM_MODEL).
DEFAULT_MODELS = {
    "gemini": "gemini-3.5-flash-lite",
    "anthropic": "claude-haiku-5-5",
    "openai": "gpt-5-mini",
}
KEY_NAMES = {"gemini": "GEMINI_API_KEY", "anthropic": "ANTHROPIC_API_KEY", "openai": "OPENAI_API_KEY"}
MAX_TOKENS = 2000
RETRIES = 2  # transient errors only (rate limit, overload)


class LLMError(RuntimeError):
    """The provider could not be reached or rejected the request."""


@dataclass
class LLMReply:
    text: str
    tool_calls: list[dict] = field(default_factory=list)
    cached: bool = False  # True if served from disk cache or mock: no API cost


class LLMClient:
    def __init__(self, provider: str | None = None, model: str | None = None,
                 mock: bool = False, cache_dir: Path | None = None):
        self.mock = mock
        self.provider = "mock" if mock else (provider or os.getenv("LLM_PROVIDER") or "gemini").lower()
        if not mock and self.provider not in DEFAULT_MODELS:
            raise ValueError(f"Unknown provider '{self.provider}'. Use gemini, anthropic or openai.")
        self.model = "mock" if mock else (model or os.getenv("LLM_MODEL") or DEFAULT_MODELS[self.provider])
        self.cache_dir = cache_dir or ROOT / "cache" / "llm"
        self.api_calls = 0  # real (paid) calls made by this client

    def chat(self, system: str, messages: list[dict], tools: list[dict]) -> LLMReply:
        """Return the model's next turn: final text, tool calls, or both."""
        if self.mock:
            return mock_reply(messages)

        payload = json.dumps([self.provider, self.model, system, messages, tools], sort_keys=True)
        path = self.cache_dir / f"{hashlib.sha256(payload.encode()).hexdigest()}.json"
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            return LLMReply(data["text"], data["tool_calls"], cached=True)

        reply = self._call_with_retry(system, messages, tools)
        self.api_calls += 1
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"provider": self.provider, "model": self.model, "text": reply.text,
                                    "tool_calls": reply.tool_calls}, indent=2), encoding="utf-8")
        return reply

    # ----------------------------------------------------------------- private
    def _call_with_retry(self, system: str, messages: list[dict], tools: list[dict]) -> LLMReply:
        if not os.getenv(KEY_NAMES[self.provider]):
            raise LLMError(f"{KEY_NAMES[self.provider]} is not set. Add it to .env or use mock mode.")
        adapter = {"gemini": _call_gemini, "anthropic": _call_anthropic, "openai": _call_openai}[self.provider]
        for attempt in range(RETRIES + 1):
            try:
                return adapter(self.model, system, messages, tools)
            except Exception as exc:
                transient = any(code in str(exc) for code in ("429", "500", "503", "overloaded"))
                if not transient or attempt == RETRIES:
                    raise LLMError(f"{self.provider} call failed: {exc}") from exc
                time.sleep(2 ** (attempt + 1))
        raise LLMError("unreachable")


# -------------------------------------------------------------------- Gemini
def _call_gemini(model: str, system: str, messages: list[dict], tools: list[dict]) -> LLMReply:
    from google import genai
    from google.genai import types

    contents = []
    for m in messages:
        if m["role"] == "user":
            contents.append(types.Content(role="user", parts=[types.Part(text=m["text"])]))
        elif m["role"] == "assistant":
            parts = [types.Part(text=m["text"])] if m.get("text") else []
            for call in m.get("tool_calls", []):
                # Gemini's thinking models require their signature to be echoed back on each call.
                signature = base64.b64decode(call["signature"]) if call.get("signature") else None
                parts.append(types.Part(function_call=types.FunctionCall(name=call["name"], args=call["args"]),
                                        thought_signature=signature))
            contents.append(types.Content(role="model", parts=parts))
        else:
            contents.append(types.Content(role="user", parts=[
                types.Part(function_response=types.FunctionResponse(name=r["name"], response=r["result"]))
                for r in m["results"]]))

    declarations = [types.FunctionDeclaration(name=t["name"], description=t["description"],
                                              parameters_json_schema=t["parameters"]) for t in tools]
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    response = client.models.generate_content(
        model=model, contents=contents,
        config=types.GenerateContentConfig(
            system_instruction=system, temperature=0.0, max_output_tokens=MAX_TOKENS,
            tools=[types.Tool(function_declarations=declarations)],
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True)))

    candidates = response.candidates or []
    parts = (candidates[0].content.parts if candidates and candidates[0].content else None) or []
    text, calls = [], []
    for part in parts:
        if part.function_call:
            call = {"id": f"call_{len(calls) + 1}", "name": part.function_call.name,
                    "args": dict(part.function_call.args or {})}
            if part.thought_signature:
                call["signature"] = base64.b64encode(part.thought_signature).decode()
            calls.append(call)
        elif part.text and not part.thought:
            text.append(part.text)
    return LLMReply("".join(text).strip(), calls)


# ----------------------------------------------------------------- Anthropic
def _call_anthropic(model: str, system: str, messages: list[dict], tools: list[dict]) -> LLMReply:
    import anthropic

    converted = []
    for m in messages:
        if m["role"] == "user":
            converted.append({"role": "user", "content": m["text"]})
        elif m["role"] == "assistant":
            blocks = [{"type": "text", "text": m["text"]}] if m.get("text") else []
            blocks += [{"type": "tool_use", "id": c["id"], "name": c["name"], "input": c["args"]}
                       for c in m.get("tool_calls", [])]
            converted.append({"role": "assistant", "content": blocks})
        else:
            converted.append({"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": r["id"], "content": json.dumps(r["result"])}
                for r in m["results"]]})

    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    response = client.messages.create(
        model=model, max_tokens=MAX_TOKENS, system=system, messages=converted,
        tools=[{"name": t["name"], "description": t["description"], "input_schema": t["parameters"]}
               for t in tools])
    text = "".join(b.text for b in response.content if b.type == "text").strip()
    calls = [{"id": b.id, "name": b.name, "args": dict(b.input)}
             for b in response.content if b.type == "tool_use"]
    return LLMReply(text, calls)


# -------------------------------------------------------------------- OpenAI
def _call_openai(model: str, system: str, messages: list[dict], tools: list[dict]) -> LLMReply:
    from openai import OpenAI

    converted = [{"role": "system", "content": system}]
    for m in messages:
        if m["role"] == "user":
            converted.append({"role": "user", "content": m["text"]})
        elif m["role"] == "assistant":
            message = {"role": "assistant", "content": m.get("text") or None}
            if m.get("tool_calls"):
                message["tool_calls"] = [
                    {"id": c["id"], "type": "function",
                     "function": {"name": c["name"], "arguments": json.dumps(c["args"])}}
                    for c in m["tool_calls"]]
            converted.append(message)
        else:
            converted += [{"role": "tool", "tool_call_id": r["id"], "content": json.dumps(r["result"])}
                          for r in m["results"]]

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    response = client.chat.completions.create(
        model=model, messages=converted,
        tools=[{"type": "function", "function": t} for t in tools])
    message = response.choices[0].message
    calls = []
    for c in message.tool_calls or []:
        try:
            args = json.loads(c.function.arguments or "{}")
        except json.JSONDecodeError:
            args = {}  # the tool layer then reports the missing arguments back to the model
        calls.append({"id": c.id, "name": c.function.name, "args": args})
    return LLMReply((message.content or "").strip(), calls)
