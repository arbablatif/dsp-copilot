"""Numerical tests: each DSP function is checked against a known answer."""
import numpy as np
import pytest

from dsp_copilot import dsp

FS = 1000.0
T = np.arange(int(FS * 2)) / FS


def tone(freq, amp=1.0):
    return amp * np.sin(2 * np.pi * freq * T)


def test_spectrum_reads_true_amplitude_and_frequency():
    f, amp = dsp.amplitude_spectrum(tone(123.4, 0.7), FS)
    peak = dsp.spectral_peaks(f, amp, 1)[0]
    assert peak["freq_hz"] == pytest.approx(123.4, abs=0.1)
    assert peak["amplitude"] == pytest.approx(0.7, rel=0.2)  # between bins, the Hann window reads low


def test_noise_bumps_are_not_reported_as_peaks():
    x = tone(100) + 0.01 * np.random.default_rng(0).standard_normal(len(T))
    f, amp = dsp.amplitude_spectrum(x, FS)
    assert [round(p["freq_hz"]) for p in dsp.spectral_peaks(f, amp, 5)] == [100]


@pytest.mark.parametrize("method", ["fir", "butterworth", "chebyshev1"])
def test_lowpass_keeps_low_and_removes_high(method):
    spec = dsp.design_filter(FS, method, "lowpass", [100], order=101 if method == "fir" else 6)
    y = dsp.apply_filter(spec, tone(20) + tone(300))
    assert dsp.band_power(y, FS, 19, 21) == pytest.approx(0.5, rel=0.3)
    assert dsp.band_power(y, FS, 299, 301) < 1e-4


def test_notch_removes_only_its_frequency():
    spec = dsp.design_filter(FS, "notch", "notch", [50])
    y = dsp.apply_filter(spec, tone(50) + tone(120))
    # At least 30 dB down from the tone's 0.5 power. The limit is the notch's edge transient
    # (about Q / (pi * f0) = 0.2 s), which matters on a signal only 2 s long.
    assert dsp.band_power(y, FS, 49, 51) < 0.5 * 10 ** (-30 / 10)
    assert dsp.band_power(y, FS, 119, 121) == pytest.approx(0.5, rel=0.05)


def test_zero_phase_does_not_shift_but_causal_does():
    pulse = np.exp(-((T - 1.0) ** 2) / (2 * 0.02 ** 2))
    spec = dsp.design_filter(FS, "butterworth", "lowpass", [40])
    assert abs(np.argmax(dsp.apply_filter(spec, pulse, zero_phase=True)) - 1000) <= 1
    assert np.argmax(dsp.apply_filter(spec, pulse, zero_phase=False)) > 1005


@pytest.mark.parametrize("kwargs, message", [
    (dict(method="butterworth", response="lowpass", cutoff_hz=[600]), "Nyquist"),
    (dict(method="butterworth", response="bandpass", cutoff_hz=[100]), "exactly 2"),
    (dict(method="butterworth", response="lowpass", cutoff_hz=[100], order=40), "order"),
    (dict(method="wavelet", response="lowpass", cutoff_hz=[100]), "method"),
])
def test_bad_filter_requests_are_rejected(kwargs, message):
    with pytest.raises(dsp.FilterDesignError, match=message):
        dsp.design_filter(FS, **kwargs)


def test_time_peaks_and_rate():
    x = np.zeros(5000)
    x[250::500] = 1.0  # one spike every 0.5 s at 1 kHz = 120 per minute
    idx = dsp.time_peaks(x, FS, min_distance_s=0.2)
    assert len(idx) == 10
    assert dsp.rate_from_peaks(idx, FS)["rate_per_min"] == pytest.approx(120)


def test_irregular_peaks_raise_a_warning():
    assert "warning" in dsp.rate_from_peaks(np.array([0, 100, 150, 900, 950]), FS)
    assert "warning" not in dsp.rate_from_peaks(np.array([0, 500, 1000, 1500]), FS)


def test_both_snr_methods_agree_with_the_constructed_value():
    rng = np.random.default_rng(1)
    clean = tone(100, 1.0)
    noise = rng.standard_normal(len(T)) * np.sqrt(0.5 / 10)  # 10 dB below the tone's power of 0.5
    x = clean + noise
    truth = 10 * np.log10(np.mean(clean ** 2) / np.mean(noise ** 2))
    assert dsp.snr_reference(x, clean)["snr_db"] == pytest.approx(truth, abs=0.01)
    assert dsp.snr_band(x, FS, 100)["snr_db"] == pytest.approx(truth, abs=0.5)
