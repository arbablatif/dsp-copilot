"""A rule-based stand-in for the LLM, used by --mock.

It reads the latest request with keyword rules, builds a plan of tool calls, and
emits them one turn at a time in the same format a real provider returns. The
agent loop, tools, plots and evaluation therefore all run for real with no API.

It is deliberately simple: it proves the plumbing, not language understanding.
Live-model behaviour is measured separately (see README).
"""
from __future__ import annotations

import json
import re
from typing import Callable

CONTEXT_MARKER = "Signals in workspace: "
REQUEST_MARKER = "Request: "

# A step takes the tool results so far and returns (tool_name, args).
Step = Callable[[list[dict]], tuple[str, dict]]


def build_user_message(signals: list[dict], request: str) -> str:
    """The text sent to any model (real or mock) for one user turn."""
    return f"{CONTEXT_MARKER}{json.dumps(signals)}\n\n{REQUEST_MARKER}{request}"


def _parse_user_message(text: str) -> tuple[list[dict], str]:
    head, _, request = text.partition(f"\n\n{REQUEST_MARKER}")
    return json.loads(head.removeprefix(CONTEXT_MARKER)), request


def _frequencies(text: str) -> list[float]:
    """All frequencies in the text: '1 kHz' -> 1000.0, '50 Hz' -> 50.0."""
    found = re.findall(r"(\d+(?:\.\d+)?)\s*(k?)hz", text)
    return [float(v) * (1000 if k else 1) for v, k in found]


def _order(text: str) -> int | None:
    match = re.search(r"(\d+)(?:st|nd|rd|th)[- ]order|order[- ](\d+)", text)
    return int(match.group(1) or match.group(2)) if match else None


def _pick_signal(request: str, signals: list[dict]) -> str:
    """The signal whose name shares the most words with the request; else the most recent."""
    words = set(re.findall(r"[a-z0-9]+", request))
    best, best_score = signals[-1]["name"], 0
    for s in signals:
        score = len(words & set(s["name"].split("_")))
        if score > best_score:
            best, best_score = s["name"], score
    return best


def _last_new_signal(results: list[dict], default: str) -> str:
    for r in reversed(results):
        if "new_signal" in r["result"]:
            return r["result"]["new_signal"]
    return default


def _last_filter(results: list[dict]) -> str:
    for r in reversed(results):
        if "filter_id" in r["result"]:
            return r["result"]["filter_id"]
    return "filter_1"


def _plan(request: str, signals: list[dict]) -> list[Step]:
    text = request.lower()
    target = _pick_signal(text, signals)
    freqs = _frequencies(text)
    steps: list[Step] = []

    def add_filter(method: str, response: str, cutoff: list[float], order: int | None = None) -> None:
        args = {"signal": target, "method": method, "response": response, "cutoff_hz": cutoff}
        if order:
            args["order"] = order
        steps.append(lambda r: ("design_filter", args))
        steps.append(lambda r: ("apply_filter", {"signal": target, "filter_id": _last_filter(r)}))

    method = "fir" if re.search(r"\bfir\b", text) else "chebyshev1" if "chebyshev" in text else "butterworth"
    wants_heart_rate = bool(re.search(r"heart ?rate|r-?peak|\bbpm\b", text))
    probes: list[float] = []

    if re.search(r"\bhum\b|mains|notch", text):
        hum = freqs[0] if freqs else 50.0
        add_filter("notch", "notch", [hum])
        probes = [hum]
    elif re.search(r"low-?pass", text) and freqs:
        add_filter(method, "lowpass", freqs[:1], _order(text))
        probes = freqs[:1]
    elif re.search(r"high-?pass", text) and freqs:
        add_filter(method, "highpass", freqs[:1], _order(text))
    elif re.search(r"band-?pass", text) and len(freqs) >= 2:
        add_filter(method, "bandpass", freqs[:2], _order(text))
    filtered = bool(steps)

    if filtered:
        steps.append(lambda r: ("compute_fft", {"signal": _last_new_signal(r, target), "compare_with": target,
                                                "probe_freqs_hz": probes or None}))
    if wants_heart_rate:
        add_filter("butterworth", "bandpass", [5.0, 30.0], 2)  # QRS energy sits in this band
        steps.append(lambda r: ("detect_peaks", {"signal": _last_new_signal(r, target), "domain": "time",
                                                 "min_distance_s": 0.3}))
    if re.search(r"\bsnr\b|signal.to.noise", text):
        clean = next((s["name"] for s in signals if "clean" in s["name"] and s["name"] != target), None)
        if clean:
            steps.append(lambda r: ("compute_snr", {"signal": target, "method": "reference",
                                                    "reference_signal": clean}))
        else:
            steps.append(lambda r: ("compute_snr", {"signal": target, "method": "band"}))
    if "spectrogram" in text:
        steps.append(lambda r: ("compute_spectrogram", {"signal": target}))
    if not filtered and re.search(r"spectrum|fft|dominant|frequenc", text):
        steps.append(lambda r: ("compute_fft", {"signal": target}))
    if not steps:  # nothing recognised: a safe overview
        steps.append(lambda r: ("describe_signal", {"signal": target}))
        steps.append(lambda r: ("compute_fft", {"signal": target}))
    return steps


def _summary(results: list[dict]) -> str:
    """Final answer, built only from the tool results."""
    lines = ["(Mock mode: tools were chosen by keyword rules, not an LLM.)"]
    for item in results:
        name, r = item["name"], item["result"]
        if "error" in r:
            lines.append(f"- {name} failed: {r['error']}")
        elif name == "design_filter":
            lines.append(f"- Designed a {r['method']} {r['response']} filter at {r['cutoff_hz']} Hz.")
        elif name == "apply_filter":
            lines.append(f"- Applied it, creating {r['new_signal']} (RMS {r['rms_before']} to {r['rms_after']}). "
                         f"Gain at the cutoff: {r['applied_gain_db_at_cutoff'][0]} dB.")
        elif name == "compute_fft":
            peaks = ", ".join(f"{p['freq_hz']:g} Hz" for p in r["peaks"])
            lines.append(f"- Strongest spectral peaks: {peaks or 'none above the noise floor'}.")
            for p in r.get("probes", []):
                if "difference_db" in p:
                    lines.append(f"- Level at {p['freq_hz']:g} Hz changed by {p['difference_db']} dB.")
        elif name == "compute_spectrogram":
            lines.append(f"- Spectrogram: dominant frequency moves from about {r['dominant_freq_start_hz']} Hz "
                         f"to {r['dominant_freq_end_hz']} Hz.")
        elif name == "detect_peaks":
            lines.append(f"- Found {r.get('count')} peaks, a rate of {r.get('rate_per_min')} per minute.")
        elif name == "compute_snr":
            lines.append(f"- SNR is {r['snr_db']} dB ({r['method']} method).")
        elif name == "describe_signal":
            lines.append(f"- {r['name']}: {r['duration_s']} s at {r['fs_hz']} Hz.")
    return "\n".join(lines)


def mock_reply(messages: list[dict]):
    """Next turn of the plan. Stateless: progress is read from the message history."""
    from .llm import LLMReply  # imported here because llm.py imports this module

    last_user = max(i for i, m in enumerate(messages) if m["role"] == "user")
    signals, request = _parse_user_message(messages[last_user]["text"])
    results = [r for m in messages[last_user:] if m["role"] == "tool" for r in m["results"]]

    plan = _plan(request, signals)
    if len(results) >= len(plan):
        return LLMReply(_summary(results), [], cached=True)
    name, args = plan[len(results)](results)
    args = {k: v for k, v in args.items() if v is not None}
    return LLMReply("", [{"id": f"mock_{len(results) + 1}", "name": name, "args": args}], cached=True)
