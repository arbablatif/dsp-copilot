"""Command line entry point.

    python cli.py samples/ecg_50hz_hum.csv "remove the 50 Hz hum and show the spectrum" --mock
    python cli.py samples/tone_noisy.wav samples/tone_clean.wav "what is the SNR of tone_noisy?"
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from dsp_copilot.agent import Agent
from dsp_copilot.llm import LLMClient
from dsp_copilot.signals import load_signal
from dsp_copilot.workspace import Workspace


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask for signal processing in plain English.")
    parser.add_argument("files", nargs="+", type=Path, help="One or more CSV or WAV files.")
    parser.add_argument("request", help="What you want done, in plain English.")
    parser.add_argument("--fs", type=float, help="Sample rate, for CSV files with no time column.")
    parser.add_argument("--mock", action="store_true", help="Run with no API calls.")
    parser.add_argument("--out", type=Path, default=Path("out/cli"), help="Folder for plots.")
    args = parser.parse_args()

    workspace = Workspace(args.out)
    for file in args.files:
        workspace.add_signal(load_signal(file, fs=args.fs))

    llm = LLMClient(mock=args.mock)
    result = Agent(llm, workspace).run(args.request)

    for i, step in enumerate(result.trace, 1):
        print(f"[{i}] {step.name}({json.dumps(step.args)})")
        print(f"    -> {json.dumps(step.result)}")
    print(f"\n{result.text}\n")
    print(f"Model: {llm.provider}/{llm.model} | turns: {result.llm_turns} | paid API calls: {result.api_calls}")
    print(f"Plots saved in {args.out}")


if __name__ == "__main__":
    main()
