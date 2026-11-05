"""Streamlit UI: upload signals, ask in plain English, inspect every tool call.

Run:  streamlit run app.py
"""
from __future__ import annotations

import json
import uuid
from pathlib import Path

import streamlit as st

from dsp_copilot.agent import Agent
from dsp_copilot.llm import LLMClient, LLMError
from dsp_copilot.signals import SignalLoadError, load_signal
from dsp_copilot.tools import ToolOutcome
from dsp_copilot.workspace import Workspace

ROOT = Path(__file__).resolve().parent
SAMPLES = ROOT / "samples"
EXAMPLES = [
    "Remove the 50 Hz hum and show the spectrum",
    "Detect the R-peaks and estimate the heart rate",
    "What are the dominant frequencies in this signal?",
    "Show me a spectrogram",
    "Low-pass at 1 kHz with a 4th-order Butterworth and compare before and after",
    "What is the SNR of the noisy tone?",
]

st.set_page_config(page_title="DSP Copilot", layout="wide")


def new_session() -> None:
    """Reset everything that belongs to one conversation."""
    session_id = uuid.uuid4().hex[:8]
    st.session_state.workspace = Workspace(ROOT / "out" / f"session_{session_id}")
    st.session_state.history = []     # [{"request", "text", "trace", "stats"}]
    st.session_state.agent = None     # rebuilt when the mode changes
    st.session_state.agent_mode = None
    st.session_state.loaded = set()


if "workspace" not in st.session_state:
    new_session()


def show_step(number: int, step: ToolOutcome) -> None:
    """One tool call: name, arguments, result and any plots."""
    status = "OK" if step.ok else "ERROR"
    with st.expander(f"{number}. {step.name} ({status}, {step.seconds:.2f} s)", expanded=bool(step.figures)):
        left, right = st.columns(2)
        left.caption("Arguments")
        left.json(step.args)
        right.caption("Result")
        right.json(step.result)
        for figure in step.figures:
            st.image(str(figure), use_container_width=True)


def show_turn(turn: dict) -> None:
    with st.chat_message("user"):
        st.write(turn["request"])
    with st.chat_message("assistant"):
        st.write(turn["text"])
        st.caption(turn["stats"])
        for i, step in enumerate(turn["trace"], 1):
            show_step(i, step)


# ---------------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Signals")
    uploads = st.file_uploader("Upload CSV or WAV", type=["csv", "wav"], accept_multiple_files=True)
    csv_fs = st.number_input("Sample rate for CSV files with no time column (Hz)", min_value=0.0, value=0.0)

    sample_files = sorted(p.name for p in SAMPLES.glob("*") if p.suffix in (".csv", ".wav"))
    chosen = st.multiselect("Or use synthetic samples", sample_files, default=sample_files[:1])

    ws: Workspace = st.session_state.workspace
    for name in chosen:
        if name not in st.session_state.loaded:
            ws.add_signal(load_signal(SAMPLES / name))
            st.session_state.loaded.add(name)
    for upload in uploads or []:
        if upload.name not in st.session_state.loaded:
            try:
                ws.add_signal(load_signal(upload.getvalue(), filename=upload.name, fs=csv_fs or None))
                st.session_state.loaded.add(upload.name)
            except SignalLoadError as exc:
                st.error(str(exc))

    if ws.signals:
        st.caption("In the workspace")
        st.dataframe([{k: s[k] for k in ("name", "fs_hz", "duration_s")} for s in ws.describe()],
                      hide_index=True, use_container_width=True)

    st.header("Model")
    mock = st.toggle("Mock mode (no API calls)", value=True)
    if st.button("New conversation"):
        new_session()
        st.rerun()

# ------------------------------------------------------------------------- main
st.title("DSP Copilot")
st.caption("Ask for signal processing in plain English. Every tool call the agent makes is listed with its "
           "arguments, results and plots.")

if st.session_state.agent is None or st.session_state.agent_mode != mock:
    st.session_state.agent = Agent(LLMClient(mock=mock), st.session_state.workspace)
    st.session_state.agent_mode = mock
agent: Agent = st.session_state.agent
st.sidebar.caption(f"Using {agent.llm.provider} / {agent.llm.model}")

for turn in st.session_state.history:
    show_turn(turn)

with st.expander("Example requests"):
    st.markdown("\n".join(f"- {example}" for example in EXAMPLES))

request = st.chat_input("e.g. remove the 50 Hz hum and show the spectrum")
if request:
    if not ws.signals:
        st.warning("Add a signal first, using the sidebar.")
    else:
        with st.chat_message("user"):
            st.write(request)
        with st.chat_message("assistant"):
            live = st.status("Working...", expanded=True)
            try:
                result = agent.run(request, on_tool=lambda step: live.write(f"Ran `{step.name}`"))
            except LLMError as exc:
                live.update(label="Failed", state="error")
                st.error(str(exc))
                st.stop()
            live.update(label=f"Done: {len(result.trace)} tool calls", state="complete", expanded=False)
        stats = (f"{agent.llm.provider} / {agent.llm.model} | {result.llm_turns} model turns | "
                 f"{result.api_calls} paid API calls")
        st.session_state.history.append({"request": request, "text": result.text,
                                         "trace": result.trace, "stats": stats})
        st.rerun()
