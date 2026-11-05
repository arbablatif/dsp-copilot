"""Signal container and CSV / WAV loading.

Every signal in the app is a 1-D float64 NumPy array plus a sample rate.
Stereo WAV files are mixed down to mono; integer WAV samples are scaled to [-1, 1].

CSV rules:
  * If a column is named like time (time, t, time_s, seconds, ...), the sample rate
    is derived from the median spacing of that column.
  * Otherwise the caller must provide `fs`.
  * The signal is the first numeric column that is not the time column
    (or the column named by `column`).
"""
from __future__ import annotations

import io
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import wavfile

TIME_COLUMN = re.compile(r"^(t|time|times|time_s|time_sec|seconds|sec|s|timestamp)$", re.IGNORECASE)
MAX_SAMPLES = 5_000_000  # guard against accidentally huge uploads


class SignalLoadError(ValueError):
    """Raised when a file cannot be turned into a signal."""


@dataclass
class Signal:
    name: str
    data: np.ndarray          # 1-D float64 samples
    fs: float                 # sample rate in Hz
    source: str = ""          # file name or the operation that produced it
    meta: dict = field(default_factory=dict)

    @property
    def duration(self) -> float:
        return len(self.data) / self.fs

    @property
    def t(self) -> np.ndarray:
        return np.arange(len(self.data)) / self.fs

    def summary(self) -> dict:
        """Compact description that is safe to send to the LLM."""
        x = self.data
        return {
            "name": self.name,
            "fs_hz": round(self.fs, 3),
            "nyquist_hz": round(self.fs / 2, 3),
            "samples": int(len(x)),
            "duration_s": round(self.duration, 4),
            "rms": float(f"{np.sqrt(np.mean(x ** 2)):.4g}"),
            "peak_abs": float(f"{np.max(np.abs(x)):.4g}"),
            "mean": float(f"{np.mean(x):.4g}"),
            "source": self.source,
        }


def _clean_name(filename: str) -> str:
    """'My Recording (1).wav' -> 'my_recording_1'."""
    stem = Path(filename).stem.lower()
    return re.sub(r"[^a-z0-9]+", "_", stem).strip("_") or "signal"


def load_wav(raw: bytes, filename: str) -> Signal:
    try:
        fs, x = wavfile.read(io.BytesIO(raw))
    except Exception as exc:  # scipy raises several types for malformed files
        raise SignalLoadError(f"Could not read WAV '{filename}': {exc}") from exc
    x = np.asarray(x)
    if np.issubdtype(x.dtype, np.integer):
        # Scale integer PCM to [-1, 1]; uint8 WAV is offset-binary around 128.
        if x.dtype == np.uint8:
            x = (x.astype(np.float64) - 128.0) / 128.0
        else:
            x = x.astype(np.float64) / float(np.iinfo(x.dtype).max)
    x = x.astype(np.float64)
    channels = 1 if x.ndim == 1 else x.shape[1]
    if x.ndim == 2:
        x = x.mean(axis=1)  # mono mixdown
    return _finish(x, float(fs), filename, {"channels": channels, "format": "wav"})


def load_csv(raw: bytes, filename: str, fs: float | None = None, column: str | None = None) -> Signal:
    try:
        df = pd.read_csv(io.BytesIO(raw))
    except Exception as exc:
        raise SignalLoadError(f"Could not parse CSV '{filename}': {exc}") from exc

    # A headerless numeric CSV gets numbers as column names; re-read without a header.
    if all(_is_number(c) for c in df.columns):
        df = pd.read_csv(io.BytesIO(raw), header=None)
        df.columns = [f"col{i}" for i in range(df.shape[1])]

    numeric = df.select_dtypes(include="number")
    time_cols = [c for c in numeric.columns if TIME_COLUMN.match(str(c).strip())]
    time_col = time_cols[0] if time_cols else None

    if column is not None:
        if column not in numeric.columns:
            raise SignalLoadError(f"Column '{column}' not found. Numeric columns: {list(numeric.columns)}")
        value_col = column
    else:
        candidates = [c for c in numeric.columns if c != time_col]
        if not candidates:
            raise SignalLoadError(f"No numeric signal column in '{filename}'.")
        value_col = candidates[0]

    if time_col is not None:
        t = numeric[time_col].to_numpy(dtype=np.float64)
        dt = np.diff(t)
        if len(dt) == 0 or np.median(dt) <= 0:
            raise SignalLoadError(f"Time column '{time_col}' is not increasing.")
        # Rounded because 1 / 0.002 in floating point gives 499.99999999999994, not 500.
        derived_fs = float(f"{1.0 / float(np.median(dt)):.9g}")
        fs = fs or derived_fs
    elif fs is None:
        raise SignalLoadError(f"'{filename}' has no time column; please provide the sample rate (fs).")

    x = numeric[value_col].to_numpy(dtype=np.float64)
    if np.isnan(x).any():
        x = pd.Series(x).interpolate(limit_direction="both").to_numpy()
    return _finish(x, float(fs), filename, {"column": str(value_col), "format": "csv",
                                            "time_column": str(time_col) if time_col else None})


def load_signal(path_or_bytes: str | Path | bytes, filename: str | None = None,
                fs: float | None = None, column: str | None = None) -> Signal:
    """Load a CSV or WAV from a path or raw bytes (as Streamlit uploads provide)."""
    if isinstance(path_or_bytes, (str, Path)):
        path = Path(path_or_bytes)
        raw, filename = path.read_bytes(), filename or path.name
    else:
        raw = path_or_bytes
        if not filename:
            raise SignalLoadError("A filename is required when loading from bytes.")
    ext = Path(filename).suffix.lower()
    if ext == ".wav":
        return load_wav(raw, filename)
    if ext in (".csv", ".txt"):
        return load_csv(raw, filename, fs=fs, column=column)
    raise SignalLoadError(f"Unsupported file type '{ext}'. Use .csv or .wav.")


def _finish(x: np.ndarray, fs: float, filename: str, meta: dict) -> Signal:
    if x.ndim != 1 or len(x) < 16:
        raise SignalLoadError(f"'{filename}' must contain at least 16 samples.")
    if len(x) > MAX_SAMPLES:
        raise SignalLoadError(f"'{filename}' has {len(x)} samples; the limit is {MAX_SAMPLES}.")
    if not np.isfinite(x).all():
        raise SignalLoadError(f"'{filename}' contains NaN or infinite values.")
    if not (0 < fs < 1e7):
        raise SignalLoadError(f"Sample rate {fs} Hz is not plausible.")
    return Signal(name=_clean_name(filename), data=x, fs=fs, source=filename, meta=meta)


def _is_number(value) -> bool:
    try:
        float(value)
        return True
    except (TypeError, ValueError):
        return False
