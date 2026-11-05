"""The agent loop: ask the model, run the tools it picks, repeat until it answers.

    user request
      -> model replies with tool calls -> tools run -> results go back to the model
      -> ... (at most MAX_STEPS model turns)
      -> model replies with plain text: the final explanation
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from .llm import LLMClient
from .mock_llm import build_user_message
from .tools import TOOLS, ToolOutcome, execute
from .workspace import Workspace

MAX_STEPS = 8

SYSTEM_PROMPT = """You are a signal-processing assistant. You work on signals in a workspace,
only through the tools provided. Each user message lists the signals available.

Rules:
1. Every number you state must come from a tool result. Never estimate or invent values.
2. Use the tools instead of answering from memory, and use only as many as the request needs.
3. Check the sample rate first: filter cutoffs must be below Nyquist.
4. To remove one frequency (such as 50 or 60 Hz mains hum) use a notch filter.
5. After filtering, verify the effect: call compute_fft on the new signal with compare_with set to the
   original and probe_freqs_hz set to the frequencies you changed.
6. To detect events in a noisy waveform, filter first. For ECG R-peaks and heart rate, band-pass
   about 5 to 30 Hz, then use detect_peaks with min_distance_s about 0.3.
7. Filtering creates a new signal. Use the name returned in new_signal for later steps.
8. If a tool returns an error, fix the arguments and try again. If it returns a warning, act on it
   or tell the user.
9. Report what was applied, not what was designed: apply_filter's applied_gain_db_at_cutoff is the real
   gain. With zero_phase it is double the single-pass design gain.
10. Plots are shown to the user automatically. Do not try to display them yourself.
11. When finished, reply in plain text with no tool call: briefly say what you did, why you chose those
    settings, and what the key numbers mean. Keep it under 150 words."""


@dataclass
class AgentResult:
    text: str
    trace: list[ToolOutcome] = field(default_factory=list)
    llm_turns: int = 0
    api_calls: int = 0        # paid calls; cached and mock turns are not counted
    finished: bool = True     # False if the step limit stopped the run

    def tools_used(self) -> list[str]:
        return [t.name for t in self.trace]


def _for_llm(result: dict) -> dict:
    """What the model sees. File names are hidden: it cannot open them, and leaving them
    out keeps the cache key the same from one session to the next."""
    view = dict(result)
    if view.pop("figure", None):
        view["plot_shown_to_user"] = True
    return view


class Agent:
    def __init__(self, llm: LLMClient, workspace: Workspace, max_steps: int = MAX_STEPS):
        self.llm, self.ws, self.max_steps = llm, workspace, max_steps
        self.messages: list[dict] = []  # kept across turns, so follow-up requests have context

    def run(self, request: str, on_tool: Callable[[ToolOutcome], None] | None = None) -> AgentResult:
        """Handle one user request. `on_tool` is called after each tool, for live display."""
        self.messages.append({"role": "user", "text": build_user_message(self.ws.describe(), request)})
        result = AgentResult(text="")
        calls_before = self.llm.api_calls

        for _ in range(self.max_steps):
            reply = self.llm.chat(SYSTEM_PROMPT, self.messages, TOOLS)
            result.llm_turns += 1
            self.messages.append({"role": "assistant", "text": reply.text, "tool_calls": reply.tool_calls})

            if not reply.tool_calls:
                result.text = reply.text
                break

            outputs = []
            for call in reply.tool_calls:
                outcome = execute(self.ws, call["name"], call["args"])
                result.trace.append(outcome)
                outputs.append({"id": call["id"], "name": call["name"], "result": _for_llm(outcome.result)})
                if on_tool:
                    on_tool(outcome)
            self.messages.append({"role": "tool", "results": outputs})
        else:
            result.finished = False
            result.text = (f"I stopped after {self.max_steps} steps without finishing. "
                           "The tool calls made so far are listed below.")
            # Close the turn so the history stays valid for a follow-up request.
            self.messages.append({"role": "assistant", "text": result.text, "tool_calls": []})

        result.api_calls = self.llm.api_calls - calls_before
        return result
