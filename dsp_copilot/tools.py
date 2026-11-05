"""The tools the LLM may call: JSON schemas, argument checks and dispatch.

This file is the agent's only route to the data. The LLM sends a tool name and
JSON arguments; `execute` validates them against the schema, runs the matching
function, and returns a compact JSON result. No model-written code ever runs.

Errors are returned to the LLM as {"error": ...} so it can correct itself.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import dsp, plots
from .signals import Signal
from .workspace import ToolError, Workspace

MAX_LISTED = 20       # cap on list lengths sent back to the LLM (keeps tokens low)
MAX_VIEW_SAMPLES = 4000  # before/after plots show this many samples so waveforms stay readable
GAIN_FLOOR_DB = -120.0   # a notch's theoretical null is infinitely deep; deeper values are rounding noise


def _schema(properties: dict, required: list[str]) -> dict:
    return {"type": "object", "properties": properties, "required": required}


SIGNAL_ARG = {"type": "string", "description": "Name of a signal in the workspace."}

TOOLS: list[dict] = [
    {
        "name": "describe_signal",
        "description": "Basic facts about a signal: sample rate, Nyquist, duration, RMS, peak, mean.",
        "parameters": _schema({"signal": SIGNAL_ARG}, ["signal"]),
    },
    {
        "name": "plot_signal",
        "description": "Plot a signal in the time domain, optionally only between start_s and end_s.",
        "parameters": _schema({
            "signal": SIGNAL_ARG,
            "start_s": {"type": "number", "description": "Start of the view in seconds."},
            "end_s": {"type": "number", "description": "End of the view in seconds."},
        }, ["signal"]),
    },
    {
        "name": "compute_fft",
        "description": ("Compute and plot the amplitude spectrum, and list the strongest spectral peaks. "
                        "Use compare_with to overlay another signal (e.g. before vs after filtering) and "
                        "probe_freqs_hz to read exact levels at frequencies of interest in both signals."),
        "parameters": _schema({
            "signal": SIGNAL_ARG,
            "compare_with": {"type": "string", "description": "Optional second signal to overlay."},
            "window": {"type": "string", "enum": ["hann", "hamming", "blackman", "boxcar"],
                       "description": "Window function. Default hann."},
            "n_peaks": {"type": "integer", "description": "How many peaks to list. Default 5."},
            "min_freq_hz": {"type": "number", "description": "Ignore peaks below this frequency."},
            "max_freq_hz": {"type": "number", "description": "Upper limit of the plot and peak search."},
            "probe_freqs_hz": {"type": "array", "items": {"type": "number"},
                               "description": "Frequencies at which to report the exact level."},
        }, ["signal"]),
    },
    {
        "name": "compute_spectrogram",
        "description": "Compute and plot a spectrogram (how frequency content changes over time).",
        "parameters": _schema({
            "signal": SIGNAL_ARG,
            "nperseg": {"type": "integer", "description": "Samples per window. Larger = finer frequency, "
                                                          "coarser time. Default about 32 ms."},
            "overlap": {"type": "number", "description": "Window overlap fraction, 0 to 0.95. Default 0.75."},
            "max_freq_hz": {"type": "number", "description": "Upper limit of the frequency axis."},
        }, ["signal"]),
    },
    {
        "name": "design_filter",
        "description": ("Design a digital filter for a signal's sample rate and plot its frequency response. "
                        "Returns a filter_id for apply_filter. Use method=notch for removing a single "
                        "frequency such as mains hum. method=fir gives linear phase; butterworth is maximally "
                        "flat; chebyshev1 has a steeper roll-off with passband ripple."),
        "parameters": _schema({
            "signal": {"type": "string", "description": "Signal whose sample rate the filter is designed for."},
            "method": {"type": "string", "enum": list(dsp.FILTER_METHODS)},
            "response": {"type": "string", "enum": list(dsp.FILTER_RESPONSES)},
            "cutoff_hz": {"type": "array", "items": {"type": "number"},
                          "description": "One frequency for lowpass/highpass/notch; two for bandpass/bandstop."},
            "order": {"type": "integer", "description": "IIR order (default 4) or FIR tap count (default 101)."},
            "ripple_db": {"type": "number", "description": "Chebyshev passband ripple in dB. Default 1."},
            "q": {"type": "number", "description": "Notch quality factor. Default 30; higher = narrower."},
        }, ["signal", "method", "response", "cutoff_hz"]),
    },
    {
        "name": "apply_filter",
        "description": ("Apply a designed filter, creating a NEW signal (the original is kept) and plotting "
                        "before vs after. zero_phase=true (default) avoids any time shift but doubles the "
                        "attenuation in dB; use false to simulate a real-time filter."),
        "parameters": _schema({
            "signal": SIGNAL_ARG,
            "filter_id": {"type": "string", "description": "ID returned by design_filter."},
            "zero_phase": {"type": "boolean", "description": "Forward-backward filtering. Default true."},
            "output_name": {"type": "string", "description": "Optional name for the new signal."},
        }, ["signal", "filter_id"]),
    },
    {
        "name": "detect_peaks",
        "description": ("Detect peaks. domain=time finds events in the waveform (e.g. ECG R-peaks) and reports "
                        "their rate per minute; domain=frequency finds spectral peaks. For noisy signals, "
                        "filter first. Plots the result."),
        "parameters": _schema({
            "signal": SIGNAL_ARG,
            "domain": {"type": "string", "enum": ["time", "frequency"], "description": "Default time."},
            "min_distance_s": {"type": "number", "description": "Time domain: minimum spacing between peaks."},
            "prominence_fraction": {"type": "number",
                                    "description": "Time domain: required prominence as a fraction of the "
                                                   "signal's range, 0 to 1. Default 0.4."},
            "height_fraction": {"type": "number", "description": "Time domain: minimum height as a fraction of "
                                                                 "the largest value, 0 to 1."},
            "polarity": {"type": "string", "enum": ["positive", "negative"], "description": "Default positive."},
            "n_peaks": {"type": "integer", "description": "Frequency domain: how many peaks. Default 5."},
        }, ["signal"]),
    },
    {
        "name": "compute_snr",
        "description": ("Compute signal-to-noise ratio in dB. method=reference compares with a clean reference "
                        "signal (most accurate; needs reference_signal). method=band treats a narrow band around "
                        "signal_freq_hz as signal and everything else as noise; if signal_freq_hz is omitted, "
                        "the strongest spectral peak is used."),
        "parameters": _schema({
            "signal": SIGNAL_ARG,
            "method": {"type": "string", "enum": ["band", "reference"]},
            "reference_signal": {"type": "string", "description": "Clean signal, for method=reference."},
            "signal_freq_hz": {"type": "number", "description": "Frequency of the wanted tone, for method=band."},
            "bandwidth_hz": {"type": "number", "description": "Width of the signal band, for method=band."},
        }, ["signal", "method"]),
    },
]

_SCHEMAS = {t["name"]: t["parameters"] for t in TOOLS}


@dataclass
class ToolOutcome:
    name: str
    args: dict
    result: dict                                   # what the LLM sees
    figures: list[Path] = field(default_factory=list)
    seconds: float = 0.0

    @property
    def ok(self) -> bool:
        return "error" not in self.result


# ------------------------------------------------------------------ validation
def validate_args(name: str, args: dict) -> dict:
    """Check arguments against the tool's schema and coerce harmless type slips (4.0 -> 4)."""
    schema = _SCHEMAS[name]
    props = schema["properties"]
    unknown = sorted(set(args) - set(props))
    if unknown:
        raise ToolError(f"Unknown argument(s) {unknown} for {name}. Allowed: {sorted(props)}")
    missing = [k for k in schema["required"] if args.get(k) is None]
    if missing:
        raise ToolError(f"Missing required argument(s) {missing} for {name}.")

    clean = {}
    for key, value in args.items():
        if value is None:
            continue  # some models send null for optional arguments
        spec = props[key]
        clean[key] = _coerce(key, value, spec)
        if "enum" in spec and clean[key] not in spec["enum"]:
            raise ToolError(f"{key} must be one of {spec['enum']}, got '{value}'.")
    return clean


def _coerce(key: str, value, spec: dict):
    kind = spec["type"]
    try:
        if kind == "string":
            if not isinstance(value, str):
                raise TypeError
            return value
        if kind == "boolean":
            if not isinstance(value, bool):
                raise TypeError
            return value
        if isinstance(value, bool):
            raise TypeError  # bool is an int subclass in Python; reject it for numbers
        if kind == "integer":
            if float(value) != int(float(value)):
                raise TypeError
            return int(float(value))
        if kind == "number":
            number = float(value)
            if not np.isfinite(number):
                raise TypeError
            return number
        if kind == "array":
            items = value if isinstance(value, (list, tuple)) else [value]  # allow a bare number
            numbers = [float(v) for v in items]
            if not numbers or not np.isfinite(numbers).all():
                raise TypeError
            return numbers
    except (TypeError, ValueError):
        raise ToolError(f"Argument '{key}' must be of type {kind}, got {value!r}.") from None
    raise ToolError(f"Unsupported schema type {kind}.")


# ------------------------------------------------------------------ tool bodies
def _gains(gains_db) -> list:
    """Gains for the LLM: real numbers, or 'below -120' where the value has no physical meaning."""
    return [round(float(g), 2) if g > GAIN_FLOOR_DB else f"below {GAIN_FLOOR_DB:.0f}" for g in gains_db]


def _describe_signal(ws: Workspace, signal: str) -> tuple[dict, list[Path]]:
    return ws.get_signal(signal).summary(), []


def _plot_signal(ws: Workspace, signal: str, start_s: float = 0.0, end_s: float | None = None):
    s = ws.get_signal(signal)
    end_s = s.duration if end_s is None else end_s
    if not 0 <= start_s < end_s:
        raise ToolError(f"Need 0 <= start_s < end_s; signal lasts {s.duration:.3f} s.")
    i0, i1 = int(start_s * s.fs), min(len(s.data), int(np.ceil(end_s * s.fs)))
    if i1 - i0 < 2:
        raise ToolError("The requested view contains fewer than 2 samples.")
    path = ws.new_figure_path(f"time_{s.name}")
    plots.time_plot(path, f"{s.name}: waveform", s.t[i0:i1], s.data[i0:i1])
    return {"figure": path.name, "view_s": [round(start_s, 4), round(i1 / s.fs, 4)]}, [path]


def _level_at(freqs: np.ndarray, amp: np.ndarray, target: float) -> float:
    """Highest level within +/-1 bin of target, so a tone between bins is not under-read."""
    i = int(np.argmin(np.abs(freqs - target)))
    return round(float(dsp.to_db(amp[max(0, i - 1):i + 2].max())), 1)


def _compute_fft(ws: Workspace, signal: str, compare_with: str | None = None, window: str = "hann",
                 n_peaks: int = 5, min_freq_hz: float = 0.0, max_freq_hz: float | None = None,
                 probe_freqs_hz: list[float] | None = None):
    s = ws.get_signal(signal)
    other = ws.get_signal(compare_with) if compare_with else None
    if other is not None and other.fs != s.fs:
        raise ToolError(f"Cannot overlay: {s.name} is {s.fs} Hz but {other.name} is {other.fs} Hz.")
    if max_freq_hz is not None and not 0 < max_freq_hz <= s.fs / 2:
        raise ToolError(f"max_freq_hz must be between 0 and Nyquist ({s.fs / 2} Hz).")
    n_peaks = int(np.clip(n_peaks, 1, MAX_LISTED))

    f, amp = dsp.amplitude_spectrum(s.data, s.fs, window)
    peaks = dsp.spectral_peaks(f, amp, n_peaks, min_freq_hz, max_freq_hz)
    result = {"peaks": peaks, "bin_resolution_hz": round(float(f[1] - f[0]), 4), "window": window}

    series = [(s.name, f, dsp.to_db(amp))]
    if other is not None:
        f2, amp2 = dsp.amplitude_spectrum(other.data, other.fs, window)
        series.append((other.name, f2, dsp.to_db(amp2)))
        result["compare_peaks"] = dsp.spectral_peaks(f2, amp2, n_peaks, min_freq_hz, max_freq_hz)

    if probe_freqs_hz:
        probes = []
        for target in probe_freqs_hz[:MAX_LISTED]:
            if not 0 <= target <= s.fs / 2:
                raise ToolError(f"Probe {target} Hz is outside 0 to Nyquist ({s.fs / 2} Hz).")
            row = {"freq_hz": target, "level_db": _level_at(f, amp, target)}
            if other is not None:
                row["compare_level_db"] = _level_at(f2, amp2, target)
                row["difference_db"] = round(row["level_db"] - row["compare_level_db"], 1)
            probes.append(row)
        result["probes"] = probes

    path = ws.new_figure_path(f"fft_{s.name}")
    title = f"Spectrum: {s.name}" + (f" vs {other.name}" if other is not None else "")
    plots.spectrum_plot(path, title, series, peaks, max_freq_hz)
    result["figure"] = path.name
    return result, [path]


def _compute_spectrogram(ws: Workspace, signal: str, nperseg: int | None = None, overlap: float = 0.75,
                         max_freq_hz: float | None = None):
    s = ws.get_signal(signal)
    if max_freq_hz is not None and not 0 < max_freq_hz <= s.fs / 2:
        raise ToolError(f"max_freq_hz must be between 0 and Nyquist ({s.fs / 2} Hz).")
    f, t, sxx, used = dsp.spectrogram_db(s.data, s.fs, nperseg, overlap)
    ridge = f[np.argmax(sxx, axis=0)]  # strongest frequency in each time slice
    edge = min(5, len(ridge))          # a few slices at each end: robust, but still the true start and end
    path = ws.new_figure_path(f"spectrogram_{s.name}")
    plots.spectrogram_plot(path, f"Spectrogram: {s.name}", f, t, sxx, max_freq_hz)
    return {"figure": path.name, "nperseg": used,
            "time_resolution_s": round(used / s.fs, 5), "freq_resolution_hz": round(s.fs / used, 3),
            "dominant_freq_start_hz": round(float(np.median(ridge[:edge])), 1),
            "dominant_freq_end_hz": round(float(np.median(ridge[-edge:])), 1)}, [path]


def _design_filter(ws: Workspace, signal: str, method: str, response: str, cutoff_hz: list[float],
                   order: int | None = None, ripple_db: float = 1.0, q: float = 30.0):
    s = ws.get_signal(signal)
    spec = dsp.design_filter(s.fs, method, response, cutoff_hz, order, ripple_db, q)
    if not spec.is_stable():
        raise ToolError("The designed filter is unstable; lower the order.")
    filter_id = ws.add_filter(spec)
    path = ws.new_figure_path(f"response_{filter_id}")
    label = f"{spec.method} {spec.response}, order {spec.order}"
    plots.filter_response_plot(path, f"Frequency response: {label}", spec)
    gains = spec.gain_db(spec.cutoff_hz)
    return {"filter_id": filter_id, "method": spec.method, "response": spec.response,
            "cutoff_hz": spec.cutoff_hz, "order": spec.order, "stable": True, **spec.params,
            "single_pass_gain_db_at_cutoff": _gains(gains),
            "figure": path.name}, [path]


def _apply_filter(ws: Workspace, signal: str, filter_id: str, zero_phase: bool = True,
                  output_name: str | None = None):
    s, spec = ws.get_signal(signal), ws.get_filter(filter_id)
    if abs(spec.fs - s.fs) > 1e-9:
        raise ToolError(f"{filter_id} was designed for {spec.fs} Hz but {s.name} is {s.fs} Hz. "
                        "Design a new filter for this signal.")
    y = dsp.apply_filter(spec, s.data, zero_phase)
    name = ws.add_signal(Signal(output_name or f"{s.name}_filtered", y, s.fs,
                                source=f"{filter_id} applied to {s.name}",
                                meta={"parent": s.name, "filter_id": filter_id, "zero_phase": zero_phase}))
    n = min(len(y), MAX_VIEW_SAMPLES)
    path = ws.new_figure_path(f"compare_{name}")
    plots.compare_time_plot(path, f"Before vs after: {filter_id}", s.t[:n], s.data[:n], y[:n], s.name, name)
    rms_in, rms_out = (float(np.sqrt(np.mean(v ** 2))) for v in (s.data, y))
    passes = 2 if zero_phase else 1  # forward-backward filtering applies the response twice
    applied = spec.gain_db(spec.cutoff_hz) * passes
    return {"new_signal": name, "zero_phase": zero_phase, "filter_passes": passes,
            "applied_gain_db_at_cutoff": _gains(applied),
            "rms_before": float(f"{rms_in:.4g}"), "rms_after": float(f"{rms_out:.4g}"),
            "figure": path.name}, [path]


def _detect_peaks(ws: Workspace, signal: str, domain: str = "time", min_distance_s: float | None = None,
                  prominence_fraction: float = 0.4, height_fraction: float | None = None,
                  polarity: str = "positive", n_peaks: int = 5):
    s = ws.get_signal(signal)
    if domain == "frequency":
        f, amp = dsp.amplitude_spectrum(s.data, s.fs)
        peaks = dsp.spectral_peaks(f, amp, int(np.clip(n_peaks, 1, MAX_LISTED)))
        path = ws.new_figure_path(f"spectral_peaks_{s.name}")
        plots.spectrum_plot(path, f"Spectral peaks: {s.name}", [(s.name, f, dsp.to_db(amp))], peaks)
        return {"domain": "frequency", "peaks": peaks, "figure": path.name}, [path]

    if not 0 < prominence_fraction <= 1:
        raise ToolError("prominence_fraction must be between 0 and 1.")
    idx = dsp.time_peaks(s.data, s.fs, min_distance_s, prominence_fraction, height_fraction, polarity)
    path = ws.new_figure_path(f"peaks_{s.name}")
    plots.time_plot(path, f"Detected peaks: {s.name}", s.t, s.data, idx)
    return {"domain": "time", "count": int(len(idx)), **dsp.rate_from_peaks(idx, s.fs),
            "first_peak_times_s": [round(float(i / s.fs), 4) for i in idx[:MAX_LISTED]],
            "figure": path.name}, [path]


def _compute_snr(ws: Workspace, signal: str, method: str, reference_signal: str | None = None,
                 signal_freq_hz: float | None = None, bandwidth_hz: float | None = None):
    s = ws.get_signal(signal)
    if method == "reference":
        if not reference_signal:
            raise ToolError("method=reference needs reference_signal.")
        ref = ws.get_signal(reference_signal)
        if ref.fs != s.fs:
            raise ToolError(f"Sample rates differ: {s.fs} Hz vs {ref.fs} Hz.")
        return {"method": "reference", "reference": ref.name, **dsp.snr_reference(s.data, ref.data)}, []

    if signal_freq_hz is None:
        f, amp = dsp.amplitude_spectrum(s.data, s.fs)
        peaks = dsp.spectral_peaks(f, amp, 1)
        if not peaks:
            raise ToolError("No clear spectral peak found; provide signal_freq_hz.")
        signal_freq_hz = peaks[0]["freq_hz"]
    if not 0 < signal_freq_hz < s.fs / 2:
        raise ToolError(f"signal_freq_hz must be between 0 and Nyquist ({s.fs / 2} Hz).")
    return {"method": "band", "signal_freq_hz": signal_freq_hz,
            **dsp.snr_band(s.data, s.fs, signal_freq_hz, bandwidth_hz)}, []


_DISPATCH = {
    "describe_signal": _describe_signal,
    "plot_signal": _plot_signal,
    "compute_fft": _compute_fft,
    "compute_spectrogram": _compute_spectrogram,
    "design_filter": _design_filter,
    "apply_filter": _apply_filter,
    "detect_peaks": _detect_peaks,
    "compute_snr": _compute_snr,
}


def execute(ws: Workspace, name: str, args: dict | None) -> ToolOutcome:
    """Run one tool call. Never raises: failures become {"error": ...} for the LLM to read."""
    args = dict(args or {})
    start = time.perf_counter()
    figures: list[Path] = []
    try:
        if name not in _DISPATCH:
            raise ToolError(f"Unknown tool '{name}'. Available tools: {sorted(_DISPATCH)}")
        result, figures = _DISPATCH[name](ws, **validate_args(name, args))
    except (ToolError, dsp.FilterDesignError) as exc:
        result = {"error": str(exc)}
    except Exception as exc:  # a numeric failure must not end the conversation
        result = {"error": f"{type(exc).__name__}: {exc}"}
    return ToolOutcome(name, args, result, figures, round(time.perf_counter() - start, 4))
