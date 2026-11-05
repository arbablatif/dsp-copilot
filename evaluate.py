"""Run the 6 example requests and check each answer against known ground truth.

    python evaluate.py --mock     # no API calls: rule-based planner, real tools
    python evaluate.py            # live model (about 20-30 calls, cached afterwards)

Checks measure the signals the agent actually produced, not what it says it did.
Results are written to results/eval_<mode>.json and .md, with plots in results/figures/<mode>/.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from dsp_copilot.agent import Agent, AgentResult
from dsp_copilot.dsp import band_power
from dsp_copilot.llm import LLMClient, LLMError
from dsp_copilot.signals import Signal, load_signal
from dsp_copilot.workspace import Workspace

ROOT = Path(__file__).resolve().parent
SAMPLES = ROOT / "samples"
TRUTH = json.loads((SAMPLES / "ground_truth.json").read_text(encoding="utf-8"))

Check = Callable[[Workspace, AgentResult], tuple[bool, str]]


def db_change(before: Signal, after: Signal, f0: float, half_band: float = 1.0) -> float:
    """Change in power around f0, in dB (negative = removed)."""
    p0 = band_power(before.data, before.fs, f0 - half_band, f0 + half_band)
    p1 = band_power(after.data, after.fs, f0 - half_band, f0 + half_band)
    return 10 * np.log10(max(p1, 1e-30) / max(p0, 1e-30))


def filtered_signal(ws: Workspace) -> Signal | None:
    """The most recent signal created by apply_filter."""
    derived = [s for s in ws.signals.values() if "parent" in s.meta]
    return derived[-1] if derived else None


def tool_results(result: AgentResult, name: str) -> list[dict]:
    return [t.result for t in result.trace if t.name == name and t.ok]


def check_hum(ws: Workspace, result: AgentResult) -> tuple[bool, str]:
    out = filtered_signal(ws)
    if out is None:
        return False, "no filtered signal was produced"
    original = ws.signals["ecg_50hz_hum"]
    hum, kept = db_change(original, out, 50.0), db_change(original, out, 10.0)
    spectrum = bool(tool_results(result, "compute_fft"))
    return (hum <= -20 and abs(kept) <= 1 and spectrum,
            f"50 Hz changed {hum:.1f} dB (need <= -20), 10 Hz changed {kept:.2f} dB, spectrum shown: {spectrum}")


def check_tones(ws: Workspace, result: AgentResult) -> tuple[bool, str]:
    found = [p["freq_hz"] for r in tool_results(result, "compute_fft") + tool_results(result, "detect_peaks")
             for p in r.get("peaks", [])]
    hits = {f: any(abs(f - p) <= 2 for p in found) for f in TRUTH["three_tones"]["tones_hz"]}
    in_answer = "440" in result.text
    return all(hits.values()) and in_answer, f"tones found: {hits}, answer mentions 440: {in_answer}"


def check_spectrogram(ws: Workspace, result: AgentResult) -> tuple[bool, str]:
    runs = [t for t in result.trace if t.name == "compute_spectrogram" and t.ok]
    if not runs or not runs[0].figures[0].exists():
        return False, "no spectrogram figure"
    r, truth = runs[0].result, TRUTH["chirp_noisy"]
    start, end = r["dominant_freq_start_hz"], r["dominant_freq_end_hz"]
    ok = abs(start - truth["f_start_hz"]) <= 100 and abs(end - truth["f_end_hz"]) <= 100
    return ok, (f"sweep measured {start} -> {end} Hz, "
                f"truth {truth['f_start_hz']} -> {truth['f_end_hz']} Hz (+/-100)")


def check_lowpass(ws: Workspace, result: AgentResult) -> tuple[bool, str]:
    out = filtered_signal(ws)
    if out is None:
        return False, "no filtered signal was produced"
    spec = ws.filters[out.meta["filter_id"]]
    original = ws.signals["three_tones"]
    stop, keep = db_change(original, out, 3000.0), db_change(original, out, 440.0)
    design = spec.method == "butterworth" and spec.order == 4 and spec.cutoff_hz == [1000.0]
    # The answer must state the gain really applied: -6 dB at the cutoff with zero-phase filtering.
    # (Mentioning the -3 dB single-pass figure as well is fine, so that is not what is tested.)
    expected = r"(?<![\d.])6(\.0\d*)?\s*dB" if out.meta["zero_phase"] else r"(?<![\d.])3(\.0\d*)?\s*dB"
    states_gain = bool(re.search(expected, result.text))
    return (stop <= -30 and abs(keep) <= 1 and design and states_gain,
            f"3 kHz changed {stop:.1f} dB (need <= -30), 440 Hz changed {keep:.2f} dB, "
            f"design {spec.method} order {spec.order} at {spec.cutoff_hz} Hz, "
            f"answer states the applied cutoff gain: {states_gain}")


def check_snr(ws: Workspace, result: AgentResult) -> tuple[bool, str]:
    runs = tool_results(result, "compute_snr")
    if not runs:
        return False, "compute_snr was not called"
    truth, got = TRUTH["tone_noisy"]["snr_db"], runs[-1]["snr_db"]
    return abs(got - truth) <= 1, f"SNR {got} dB by {runs[-1]['method']} method, truth {truth} dB (+/-1)"


def check_heart_rate(ws: Workspace, result: AgentResult) -> tuple[bool, str]:
    runs = [r for r in tool_results(result, "detect_peaks") if r.get("domain") == "time"]
    if not runs:
        return False, "detect_peaks was not called in the time domain"
    truth, got = TRUTH["ecg_50hz_hum"]["bpm"], runs[-1].get("rate_per_min")
    if got is None:
        return False, "no rate was reported"
    return abs(got - truth) <= 3, f"{got} bpm from {runs[-1]['count']} peaks, truth {truth} bpm (+/-3)"


@dataclass
class Case:
    name: str
    files: list[str]
    request: str
    check: Check


CASES = [
    Case("hum_removal", ["ecg_50hz_hum.csv"], "Remove the 50 Hz hum and show the spectrum", check_hum),
    Case("dominant_tones", ["three_tones.wav"], "What are the dominant frequencies in this signal?", check_tones),
    Case("spectrogram", ["chirp_noisy.wav"], "Show me a spectrogram of the chirp", check_spectrogram),
    Case("lowpass", ["three_tones.wav"], "Low-pass at 1 kHz with a 4th-order Butterworth and compare before and after",
         check_lowpass),
    Case("snr", ["tone_noisy.wav", "tone_clean.wav"], "What is the SNR of the noisy tone?", check_snr),
    Case("heart_rate", ["ecg_50hz_hum.csv"], "Detect the R-peaks and estimate the heart rate", check_heart_rate),
]


def run_case(case: Case, llm: LLMClient, out_dir: Path) -> dict:
    case_dir = out_dir / case.name
    shutil.rmtree(case_dir, ignore_errors=True)
    ws = Workspace(case_dir)
    for file in case.files:
        ws.add_signal(load_signal(SAMPLES / file))
    result = Agent(llm, ws).run(case.request)
    passed, detail = case.check(ws, result)
    return {
        "name": case.name, "request": case.request, "passed": bool(passed), "detail": detail,
        "tools": result.tools_used(), "tool_errors": [t.result["error"] for t in result.trace if not t.ok],
        "llm_turns": result.llm_turns, "api_calls": result.api_calls, "finished": result.finished,
        "tool_calls": [{"name": t.name, "args": t.args} for t in result.trace],
        "answer": result.text,
    }


def write_markdown(path: Path, rows: list[dict], label: str) -> None:
    lines = [f"# Evaluation: {label}", "", "| Request | Result | Tools | Measured |", "|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['request']} | {'PASS' if r['passed'] else 'FAIL'} | "
                     f"{', '.join(r['tools'])} | {r['detail']} |")
    passed = sum(r["passed"] for r in rows)
    lines += ["", f"**{passed}/{len(rows)} passed.**", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mock", action="store_true", help="Run with no API calls.")
    args = parser.parse_args()

    llm = LLMClient(mock=args.mock)
    mode = "mock" if args.mock else "live"
    label = f"{llm.provider} / {llm.model}"
    results_dir = ROOT / "results"
    results_dir.mkdir(exist_ok=True)

    rows = []
    for case in CASES:
        try:
            row = run_case(case, llm, results_dir / "figures" / mode)
        except LLMError as exc:  # stop at once: never keep retrying a failing API in a loop
            print(f"Stopped at '{case.name}': {exc}")
            raise SystemExit(1)
        rows.append(row)
        print(f"{'PASS' if row['passed'] else 'FAIL'}  {row['name']:<15} {row['detail']}")
        print(f"      tools: {' -> '.join(row['tools'])}")

    passed = sum(r["passed"] for r in rows)
    calls = sum(r["api_calls"] for r in rows)
    print(f"\n{passed}/{len(rows)} passed | {label} | paid API calls this run: {calls}")
    summary = {"label": label, "mode": mode, "passed": passed, "total": len(rows), "cases": rows}
    (results_dir / f"eval_{mode}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_markdown(results_dir / f"eval_{mode}.md", rows, label)


if __name__ == "__main__":
    main()
