"""Drives the real Streamlit script in mock mode (the toggle's default), with no browser."""
from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parents[1] / "app.py")


def test_request_shows_each_tool_call_and_supports_a_follow_up():
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception

    at.chat_input[0].set_value("Remove the 50 Hz hum and show the spectrum").run()
    assert not at.exception
    labels = [e.label for e in at.expander]
    for tool in ("design_filter", "apply_filter", "compute_fft"):
        assert any(tool in label for label in labels), labels
    assert any("0 paid API calls" in c.value for c in at.caption)

    at.chat_input[0].set_value("what is the snr of ecg_50hz_hum_filtered").run()
    assert not at.exception
    assert any("compute_snr" in e.label for e in at.expander)
