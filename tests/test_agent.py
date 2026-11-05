"""Loading, tool validation and the agent loop, all with no API calls."""
import io
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.io import wavfile
from scipy.signal import chirp

from dsp_copilot import llm as llm_module
from dsp_copilot import tools
from dsp_copilot.agent import Agent
from dsp_copilot.llm import LLMClient, LLMReply
from dsp_copilot.signals import Signal, SignalLoadError, load_signal
from dsp_copilot.workspace import Workspace
from evaluate import CASES, run_case

FS = 1000.0


@pytest.fixture
def ws(tmp_path):
    workspace = Workspace(tmp_path)
    t = np.arange(2000) / FS
    workspace.add_signal(Signal("sig", np.sin(2 * np.pi * 100 * t), FS))
    return workspace


class ScriptedLLM:
    """Replays fixed replies, so the loop can be tested without any provider."""
    provider, model, api_calls = "scripted", "scripted", 0

    def __init__(self, replies):
        self.replies, self.seen = list(replies), []

    def chat(self, system, messages, tool_schemas):
        self.seen.append(json.loads(json.dumps(messages)))
        return self.replies.pop(0)


# ---------------------------------------------------------------------- loading
def test_csv_sample_rate_comes_from_the_time_column():
    s = load_signal(b"time_s,v\n0.000,1\n0.002,2\n0.004,3\n" + b"0.006,1\n" * 1 + b"".join(
        f"{0.008 + 0.002 * i:.3f},0\n".encode() for i in range(20)), filename="My File.csv")
    assert s.fs == pytest.approx(500)
    assert s.name == "my_file"


def test_csv_sample_rate_is_not_left_with_floating_point_error():
    rows = "".join(f"{i * 0.002:.3f},0\n" for i in range(200))
    s = load_signal(f"time_s,v\n{rows}".encode(), filename="a.csv")
    assert s.fs == 500.0


def test_csv_without_time_needs_fs():
    raw = "\n".join(str(i) for i in range(100)).encode()
    with pytest.raises(SignalLoadError, match="sample rate"):
        load_signal(raw, filename="x.csv")
    assert load_signal(raw, filename="x.csv", fs=200).fs == 200


def test_stereo_int16_wav_is_scaled_and_mixed_to_mono():
    buffer = io.BytesIO()
    wavfile.write(buffer, 8000, np.full((100, 2), 32767, dtype=np.int16))
    s = load_signal(buffer.getvalue(), filename="a.wav")
    assert s.data.shape == (100,) and s.data.max() == pytest.approx(1.0)


def test_unsupported_file_type_is_rejected():
    with pytest.raises(SignalLoadError, match="Unsupported"):
        load_signal(b"abc", filename="a.mp3")


# ------------------------------------------------------------------- validation
def test_every_tool_schema_has_a_handler():
    assert {t["name"] for t in tools.TOOLS} == set(tools._DISPATCH)


@pytest.mark.parametrize("name, args, message", [
    ("compute_fft", {"signal": "nope"}, "Available signals"),
    ("compute_fft", {}, "Missing required"),
    ("compute_fft", {"signal": "sig", "colour": "red"}, "Unknown argument"),
    ("compute_fft", {"signal": "sig", "window": "kaiser"}, "must be one of"),
    ("compute_fft", {"signal": "sig", "n_peaks": "many"}, "must be of type"),
    ("design_filter", {"signal": "sig", "method": "butterworth", "response": "lowpass", "cutoff_hz": [900]},
     "Nyquist"),
    ("apply_filter", {"signal": "sig", "filter_id": "filter_9"}, "Unknown filter"),
    ("run_python", {"code": "print(1)"}, "Unknown tool"),
])
def test_bad_calls_become_readable_errors(ws, name, args, message):
    outcome = tools.execute(ws, name, args)
    assert not outcome.ok and message in outcome.result["error"]


def test_harmless_type_slips_are_coerced(ws):
    args = {"signal": "sig", "method": "butterworth", "response": "lowpass", "cutoff_hz": 200, "order": 4.0}
    outcome = tools.execute(ws, "design_filter", args)
    assert outcome.ok and outcome.result["order"] == 4


def test_apply_filter_reports_the_gain_actually_applied(ws):
    args = {"signal": "sig", "method": "butterworth", "response": "lowpass", "cutoff_hz": [200]}
    designed = tools.execute(ws, "design_filter", args).result
    assert designed["single_pass_gain_db_at_cutoff"][0] == pytest.approx(-3.01, abs=0.01)
    both_ways = tools.execute(ws, "apply_filter", {"signal": "sig", "filter_id": "filter_1"}).result
    assert both_ways["applied_gain_db_at_cutoff"][0] == pytest.approx(-6.02, abs=0.01)
    one_way = tools.execute(ws, "apply_filter", {"signal": "sig", "filter_id": "filter_1", "zero_phase": False})
    assert one_way.result["applied_gain_db_at_cutoff"][0] == pytest.approx(-3.01, abs=0.01)


def test_notch_depth_is_not_reported_as_a_meaningless_number(ws):
    args = {"signal": "sig", "method": "notch", "response": "notch", "cutoff_hz": [100]}
    designed = tools.execute(ws, "design_filter", args).result
    assert designed["single_pass_gain_db_at_cutoff"] == ["below -120"]
    applied = tools.execute(ws, "apply_filter", {"signal": "sig", "filter_id": "filter_1"}).result
    assert applied["applied_gain_db_at_cutoff"] == ["below -120"]


def test_spectrogram_reports_the_true_start_and_end_of_a_sweep(tmp_path):
    fs = 8000.0
    t = np.arange(int(fs * 4)) / fs
    workspace = Workspace(tmp_path)
    workspace.add_signal(Signal("sweep", chirp(t, 200, t[-1], 3000), fs))
    result = tools.execute(workspace, "compute_spectrogram", {"signal": "sweep"}).result
    assert result["dominant_freq_start_hz"] == pytest.approx(200, abs=100)
    assert result["dominant_freq_end_hz"] == pytest.approx(3000, abs=100)


def test_tool_results_are_json_serialisable(ws):
    outcome = tools.execute(ws, "compute_snr", {"signal": "sig", "method": "band"})
    assert outcome.ok
    json.dumps(outcome.result)


# ---------------------------------------------------------------------- the loop
def test_tool_error_goes_back_to_the_model_which_recovers(ws):
    bad = {"id": "1", "name": "compute_fft", "args": {"signal": "typo"}}
    good = {"id": "2", "name": "compute_fft", "args": {"signal": "sig"}}
    llm = ScriptedLLM([LLMReply("", [bad]), LLMReply("", [good]), LLMReply("done")])
    result = Agent(llm, ws).run("spectrum please")
    assert result.text == "done" and result.tools_used() == ["compute_fft", "compute_fft"]
    assert "Available signals" in llm.seen[1][-1]["results"][0]["result"]["error"]


def test_step_limit_stops_a_model_that_never_finishes(ws):
    call = {"id": "1", "name": "describe_signal", "args": {"signal": "sig"}}
    result = Agent(ScriptedLLM([LLMReply("", [call])] * 3), ws, max_steps=3).run("loop")
    assert not result.finished and len(result.trace) == 3


def test_figure_paths_are_hidden_from_the_model(ws):
    call = {"id": "1", "name": "compute_fft", "args": {"signal": "sig"}}
    llm = ScriptedLLM([LLMReply("", [call]), LLMReply("done")])
    result = Agent(llm, ws).run("spectrum")
    sent = llm.seen[1][-1]["results"][0]["result"]
    assert "figure" not in sent and sent["plot_shown_to_user"] is True
    assert Path(result.trace[0].figures[0]).exists()


def test_follow_up_request_can_use_the_earlier_filtered_signal(ws):
    agent = Agent(LLMClient(mock=True), ws)
    agent.run("low-pass at 200 Hz")
    assert "sig_filtered" in ws.signals
    assert agent.run("show the spectrum of sig_filtered").tools_used() == ["compute_fft"]


# ------------------------------------------------------------------------- cache
def test_cache_means_the_same_conversation_is_only_paid_for_once(tmp_path, monkeypatch):
    calls = []

    def fake_adapter(model, system, messages, tool_schemas):
        calls.append(1)
        return LLMReply("hello", [{"id": "a", "name": "describe_signal", "args": {"signal": "x"}}])

    monkeypatch.setattr(llm_module, "_call_gemini", fake_adapter)
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    client = LLMClient(provider="gemini", model="test-model", cache_dir=tmp_path)
    messages = [{"role": "user", "text": "hi"}]

    first = client.chat("system", messages, [])
    second = client.chat("system", messages, [])
    assert len(calls) == 1 and client.api_calls == 1
    assert not first.cached and second.cached
    assert second.text == first.text and second.tool_calls == first.tool_calls


def test_mock_mode_never_touches_a_provider(monkeypatch):
    monkeypatch.setattr(llm_module, "_call_gemini", lambda *a: pytest.fail("provider was called"))
    assert LLMClient(mock=True).api_calls == 0


# -------------------------------------------------------------------- end to end
@pytest.mark.parametrize("case", CASES, ids=lambda c: c.name)
def test_example_requests_end_to_end_in_mock_mode(case, tmp_path):
    client = LLMClient(mock=True)
    row = run_case(case, client, tmp_path)
    assert row["passed"], row["detail"]
    assert row["tool_errors"] == [] and client.api_calls == 0
