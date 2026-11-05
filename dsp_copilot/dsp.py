"""Pure signal-processing functions (NumPy / SciPy only, no LLM, no plotting).

Everything here is deterministic and unit-tested, so the numbers the agent
reports come from well-understood code, not from the language model.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import signal as sps

EPS = 1e-20  # floor that keeps log10 finite


# --------------------------------------------------------------------- spectra
def amplitude_spectrum(x: np.ndarray, fs: float, window: str = "hann") -> tuple[np.ndarray, np.ndarray]:
    """Single-sided amplitude spectrum, scaled so a sine of amplitude A reads A.

    Returns (freqs_hz, amplitude). The window's coherent gain (sum of weights)
    is divided out so different windows give comparable tone amplitudes.
    """
    x = np.asarray(x, dtype=np.float64)
    w = sps.get_window(window, len(x))
    X = np.fft.rfft((x - x.mean()) * w)
    amp = np.abs(X) / w.sum()
    amp[1:] *= 2.0  # fold negative frequencies into positive ones (not DC)
    if len(x) % 2 == 0:
        amp[-1] /= 2.0  # the Nyquist bin has no mirror image
    return np.fft.rfftfreq(len(x), 1 / fs), amp


def to_db(values: np.ndarray, power: bool = False) -> np.ndarray:
    """Convert amplitude (20*log10) or power (10*log10) to decibels."""
    return (10 if power else 20) * np.log10(np.maximum(values, EPS))


def spectral_peaks(freqs: np.ndarray, amp: np.ndarray, n_peaks: int = 5,
                   min_freq: float = 0.0, max_freq: float | None = None,
                   prominence_db: float = 10.0, above_floor_db: float = 15.0) -> list[dict]:
    """Strongest local maxima of an amplitude spectrum.

    A peak must stand `above_floor_db` above the median level (a robust estimate
    of the noise floor), so random noise bumps are not reported as components.

    Peak frequencies are refined with parabolic interpolation on the dB curve,
    which estimates a tone's frequency to a fraction of one FFT bin.
    """
    db = to_db(amp)
    mask = (freqs > max(min_freq, 0.0)) & (freqs <= (max_freq if max_freq else freqs[-1]))
    idx, _ = sps.find_peaks(np.where(mask, db, db.min()), prominence=prominence_db)
    idx = idx[mask[idx] & (db[idx] >= np.median(db[mask]) + above_floor_db)]
    idx = idx[np.argsort(db[idx])[::-1]][:n_peaks]
    df = freqs[1] - freqs[0]
    peaks = []
    for i in idx:
        f = freqs[i]
        if 0 < i < len(db) - 1:  # parabolic interpolation around the bin
            a, b, c = db[i - 1], db[i], db[i + 1]
            denom = a - 2 * b + c
            if denom != 0:
                f = freqs[i] + 0.5 * (a - c) / denom * df
        peaks.append({"freq_hz": round(float(f), 2), "amplitude": float(f"{amp[i]:.4g}"),
                      "level_db": round(float(db[i]), 1)})
    return peaks


def band_power(x: np.ndarray, fs: float, f_lo: float, f_hi: float) -> float:
    """Mean power of x inside [f_lo, f_hi] Hz (integrated Hann periodogram)."""
    f, pxx = sps.periodogram(x, fs, window="hann", detrend="constant", scaling="density")
    df = f[1] - f[0]
    m = (f >= f_lo) & (f <= f_hi)
    return float(pxx[m].sum() * df)


def spectrogram_db(x: np.ndarray, fs: float, nperseg: int | None = None,
                   overlap: float = 0.75) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Short-time Fourier power in dB. Returns (freqs, times, Sxx_db, nperseg)."""
    if nperseg is None:
        # ~32 ms windows: a common time/frequency trade-off for audio-like signals
        nperseg = int(2 ** np.ceil(np.log2(max(fs * 0.032, 16))))
    nperseg = int(np.clip(nperseg, 16, max(16, len(x) // 4)))
    noverlap = int(np.clip(overlap, 0.0, 0.95) * nperseg)
    f, t, sxx = sps.spectrogram(x, fs, window="hann", nperseg=nperseg, noverlap=noverlap,
                                scaling="density", detrend="constant")
    return f, t, to_db(sxx, power=True), nperseg


# --------------------------------------------------------------------- filters
FILTER_METHODS = ("fir", "butterworth", "chebyshev1", "notch")
FILTER_RESPONSES = ("lowpass", "highpass", "bandpass", "bandstop", "notch")


@dataclass
class FilterSpec:
    """A designed filter. IIR filters are stored as second-order sections (numerically safer)."""
    method: str
    response: str
    cutoff_hz: list[float]
    fs: float
    order: int
    sos: np.ndarray | None = None      # IIR form
    b: np.ndarray | None = None        # FIR taps (a = [1])
    params: dict = field(default_factory=dict)

    @property
    def is_fir(self) -> bool:
        return self.b is not None

    def gain_db(self, freqs) -> np.ndarray:
        """Magnitude response in dB at the given frequencies (Hz)."""
        freqs = np.atleast_1d(np.asarray(freqs, dtype=np.float64))
        if self.is_fir:
            _, h = sps.freqz(self.b, [1.0], worN=freqs, fs=self.fs)
        else:
            _, h = sps.sosfreqz(self.sos, worN=freqs, fs=self.fs)
        return to_db(np.abs(h))

    def is_stable(self) -> bool:
        if self.is_fir:
            return True
        _, poles, _ = sps.sos2zpk(self.sos)
        return bool(np.all(np.abs(poles) < 1.0))


class FilterDesignError(ValueError):
    """Invalid filter request (e.g. cutoff above Nyquist)."""


def design_filter(fs: float, method: str, response: str, cutoff_hz: list[float],
                  order: int | None = None, ripple_db: float = 1.0, q: float = 30.0) -> FilterSpec:
    """Design a FIR or IIR filter, validating every parameter first."""
    method, response = method.lower(), response.lower()
    if method not in FILTER_METHODS:
        raise FilterDesignError(f"method must be one of {FILTER_METHODS}")
    if response not in FILTER_RESPONSES:
        raise FilterDesignError(f"response must be one of {FILTER_RESPONSES}")
    if method == "notch" or response == "notch":
        method, response = "notch", "notch"

    cutoff = sorted(float(c) for c in np.atleast_1d(cutoff_hz))
    nyq = fs / 2
    for c in cutoff:
        if not 0 < c < nyq:
            raise FilterDesignError(f"cutoff {c} Hz must be between 0 and Nyquist ({nyq} Hz)")
    need = 2 if response in ("bandpass", "bandstop") else 1
    if len(cutoff) != need:
        raise FilterDesignError(f"{response} needs exactly {need} cutoff frequenc{'ies' if need == 2 else 'y'}")
    if need == 2 and cutoff[0] == cutoff[1]:
        raise FilterDesignError("band edges must differ")
    wn = cutoff if need == 2 else cutoff[0]

    if method == "notch":
        if not 0.5 <= q <= 1000:
            raise FilterDesignError("notch quality factor q must be between 0.5 and 1000")
        b, a = sps.iirnotch(cutoff[0], q, fs=fs)
        return FilterSpec("notch", "notch", cutoff, fs, 2, sos=sps.tf2sos(b, a),
                          params={"q": q, "notch_width_hz": round(cutoff[0] / q, 3)})

    if method == "fir":
        numtaps = int(order or 101)
        if not 3 <= numtaps <= 4001:
            raise FilterDesignError("FIR order (number of taps) must be between 3 and 4001")
        if response in ("highpass", "bandstop") and numtaps % 2 == 0:
            numtaps += 1  # these responses need a type-I (odd length) FIR
        b = sps.firwin(numtaps, wn, pass_zero=response in ("lowpass", "bandstop"), fs=fs, window="hamming")
        return FilterSpec("fir", response, cutoff, fs, numtaps, b=b, params={"window": "hamming"})

    n = int(order or 4)
    if not 1 <= n <= 12:
        raise FilterDesignError("IIR order must be between 1 and 12")
    if method == "butterworth":
        sos = sps.butter(n, wn, btype=response, fs=fs, output="sos")
        params = {}
    else:
        if not 0.01 <= ripple_db <= 6:
            raise FilterDesignError("ripple_db must be between 0.01 and 6 dB")
        sos = sps.cheby1(n, ripple_db, wn, btype=response, fs=fs, output="sos")
        params = {"ripple_db": ripple_db}
    return FilterSpec(method, response, cutoff, fs, n, sos=sos, params=params)


def apply_filter(spec: FilterSpec, x: np.ndarray, zero_phase: bool = True) -> np.ndarray:
    """Filter x. Zero-phase (forward-backward) avoids time shifts but doubles the attenuation."""
    if spec.is_fir:
        if zero_phase:
            if len(x) <= 3 * len(spec.b):
                raise FilterDesignError(f"signal too short ({len(x)} samples) for a zero-phase "
                                        f"{len(spec.b)}-tap FIR; use fewer taps")
            return sps.filtfilt(spec.b, [1.0], x)
        return sps.lfilter(spec.b, [1.0], x)
    if zero_phase:
        return sps.sosfiltfilt(spec.sos, x)
    return sps.sosfilt(spec.sos, x)


# ----------------------------------------------------------------------- peaks
def time_peaks(x: np.ndarray, fs: float, min_distance_s: float | None = None,
               prominence_fraction: float = 0.4, height_fraction: float | None = None,
               polarity: str = "positive") -> np.ndarray:
    """Indices of peaks in a time series.

    Thresholds are fractions of the signal's range so the same call works for
    millivolt ECG and full-scale audio alike.
    """
    y = -x if polarity == "negative" else np.asarray(x, dtype=np.float64)
    y = y - np.median(y)
    span = float(y.max() - y.min()) or 1.0
    kwargs = {"prominence": prominence_fraction * span}
    if min_distance_s:
        kwargs["distance"] = max(1, int(round(min_distance_s * fs)))
    if height_fraction is not None:
        kwargs["height"] = height_fraction * float(y.max())
    idx, _ = sps.find_peaks(y, **kwargs)
    return idx


def rate_from_peaks(idx: np.ndarray, fs: float) -> dict:
    """Mean interval and rate (per minute) of a peak train, e.g. heart rate from R-peaks."""
    if len(idx) < 2:
        return {"mean_interval_s": None, "rate_per_min": None, "interval_std_s": None}
    intervals = np.diff(idx) / fs
    stats = {"mean_interval_s": round(float(intervals.mean()), 4),
             "rate_per_min": round(60.0 / float(intervals.mean()), 2),
             "interval_std_s": round(float(intervals.std()), 4)}
    if intervals.std() > 0.25 * intervals.mean():
        # Very uneven spacing usually means false detections (noise, T-waves), not a real rhythm.
        stats["warning"] = ("Peak spacing is very irregular, so this rate is unreliable. Likely false "
                            "detections: filter the signal first or raise the thresholds.")
    return stats


# ------------------------------------------------------------------------- SNR
def snr_band(x: np.ndarray, fs: float, signal_freq_hz: float, bandwidth_hz: float | None = None) -> dict:
    """SNR of a narrowband component: power inside a band around it vs everything else."""
    f, pxx = sps.periodogram(x, fs, window="hann", detrend="constant", scaling="density")
    df = f[1] - f[0]
    if bandwidth_hz is None:
        bandwidth_hz = 8 * df  # +/- 4 bins covers the Hann main lobe with margin
    bandwidth_hz = max(bandwidth_hz, 4 * df)
    in_band = np.abs(f - signal_freq_hz) <= bandwidth_hz / 2
    noise_band = ~in_band & (f > 0)
    p_sig, p_noise = pxx[in_band].sum() * df, pxx[noise_band].sum() * df
    return {"snr_db": round(float(10 * np.log10(max(p_sig, EPS) / max(p_noise, EPS))), 2),
            "signal_power": float(f"{p_sig:.4g}"), "noise_power": float(f"{p_noise:.4g}"),
            "band_hz": [round(float(signal_freq_hz - bandwidth_hz / 2), 2),
                        round(float(signal_freq_hz + bandwidth_hz / 2), 2)]}


def snr_reference(x: np.ndarray, reference: np.ndarray) -> dict:
    """SNR when a clean reference is known: noise = x - reference."""
    n = min(len(x), len(reference))
    ref, noise = reference[:n], x[:n] - reference[:n]
    p_sig, p_noise = float(np.mean(ref ** 2)), float(np.mean(noise ** 2))
    return {"snr_db": round(float(10 * np.log10(max(p_sig, EPS) / max(p_noise, EPS))), 2),
            "signal_power": float(f"{p_sig:.4g}"), "noise_power": float(f"{p_noise:.4g}"),
            "samples_compared": n}
