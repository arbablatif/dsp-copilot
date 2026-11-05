"""Matplotlib figures for the tools. Each function saves one PNG and returns nothing.

Style: one shared look (recessive grid, thin lines, fixed series colours).
Series colour follows the role: blue = the signal being asked about,
orange = the comparison / original. Spectrograms use a single-hue sequential map.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: no GUI backend, safe inside Streamlit and tests
import matplotlib.pyplot as plt
import numpy as np

from .dsp import FilterSpec

SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE = "#2a78d6", "#eb6834"
MAX_POINTS = 20_000  # above this, plot a min/max envelope so spikes stay visible


def _figure(height: float = 3.2):
    fig, ax = plt.subplots(figsize=(8.5, height), dpi=120, facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8, length=0)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    return fig, ax


def _finish(fig, ax, path: Path, title: str, xlabel: str, ylabel: str, legend: bool = False) -> None:
    ax.set_title(title, loc="left", fontsize=10, color=INK)
    ax.set_xlabel(xlabel, fontsize=8.5, color=MUTED)
    ax.set_ylabel(ylabel, fontsize=8.5, color=MUTED)
    if legend:
        # Above the plot area, so it can never sit on top of the data.
        ax.legend(frameon=False, fontsize=8, labelcolor=MUTED, loc="lower right",
                  bbox_to_anchor=(1.0, 1.0), ncol=2, borderaxespad=0.2)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def _envelope(t: np.ndarray, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Reduce a long trace to per-bucket min/max pairs, which keeps peaks that striding would drop."""
    if len(x) <= MAX_POINTS:
        return t, x
    size = int(np.ceil(len(x) / (MAX_POINTS // 2)))
    n = len(x) // size * size
    blocks = x[:n].reshape(-1, size)
    times = np.repeat(t[:n].reshape(-1, size)[:, 0], 2)
    return times, np.column_stack([blocks.min(axis=1), blocks.max(axis=1)]).ravel()


def time_plot(path: Path, title: str, t: np.ndarray, x: np.ndarray, peaks: np.ndarray | None = None) -> None:
    fig, ax = _figure()
    ax.plot(*_envelope(t, x), color=BLUE, linewidth=1.0)
    if peaks is not None and len(peaks):
        ax.plot(t[peaks], x[peaks], "o", color=ORANGE, markersize=5, markeredgecolor=SURFACE,
                markeredgewidth=1, label=f"{len(peaks)} peaks")
    _finish(fig, ax, path, title, "Time (s)", "Amplitude", legend=peaks is not None and len(peaks) > 0)


def compare_time_plot(path: Path, title: str, t: np.ndarray, before: np.ndarray, after: np.ndarray,
                      before_label: str, after_label: str) -> None:
    fig, ax = _figure()
    ax.plot(*_envelope(t, before), color=ORANGE, linewidth=1.0, label=before_label)
    ax.plot(*_envelope(t, after), color=BLUE, linewidth=1.0, label=after_label)
    _finish(fig, ax, path, title, "Time (s)", "Amplitude", legend=True)


def spectrum_plot(path: Path, title: str, series: list[tuple[str, np.ndarray, np.ndarray]],
                  peaks: list[dict] | None = None, max_freq: float | None = None) -> None:
    """series = [(label, freqs, level_db), ...]; the first entry is the main signal."""
    fig, ax = _figure(3.6)
    colors = [BLUE, ORANGE]
    for i, (label, f, db) in reversed(list(enumerate(series))):  # main signal drawn on top
        ax.plot(f, db, color=colors[i], linewidth=0.9, label=label)
    top = max(float(db.max()) for _, _, db in series)
    floor = float(np.percentile(series[0][2], 1))
    ax.set_ylim(max(floor - 5, top - 120), top + 10)
    x_max = max_freq or series[0][1][-1]
    ax.set_xlim(0, x_max)
    labelled: list[float] = []
    for p in (peaks or [])[:5]:  # strongest first; skip any label that would collide with one already placed
        if any(abs(p["freq_hz"] - other) < 0.07 * x_max for other in labelled):
            continue
        labelled.append(p["freq_hz"])
        ax.annotate(f"{p['freq_hz']:g} Hz", (p["freq_hz"], p["level_db"]), xytext=(0, 5),
                    textcoords="offset points", ha="center", fontsize=7.5, color=INK)
    _finish(fig, ax, path, title, "Frequency (Hz)", "Level (dB re 1)", legend=len(series) > 1)


def spectrogram_plot(path: Path, title: str, f: np.ndarray, t: np.ndarray, sxx_db: np.ndarray,
                     max_freq: float | None = None, dynamic_range_db: float = 70.0) -> None:
    fig, ax = _figure(3.8)
    ax.grid(False)
    top = float(sxx_db.max())
    mesh = ax.pcolormesh(t, f, sxx_db, shading="auto", cmap="Blues", vmin=top - dynamic_range_db, vmax=top,
                         rasterized=True)
    if max_freq:
        ax.set_ylim(0, max_freq)
    bar = fig.colorbar(mesh, ax=ax, pad=0.01)
    bar.set_label("Power (dB/Hz)", fontsize=8, color=MUTED)
    bar.ax.tick_params(colors=MUTED, labelsize=7, length=0)
    bar.outline.set_visible(False)
    _finish(fig, ax, path, title, "Time (s)", "Frequency (Hz)")


def filter_response_plot(path: Path, title: str, spec: FilterSpec) -> None:
    nyq = spec.fs / 2
    log_axis = max(spec.cutoff_hz) < nyq / 20  # low cutoffs vanish on a linear axis
    grid = np.geomspace(nyq / 1e4, nyq * 0.999, 4096) if log_axis else np.linspace(nyq / 4096, nyq * 0.999, 4096)
    grid = np.unique(np.concatenate([grid, spec.cutoff_hz]))  # guarantee the notch itself is sampled
    fig, ax = _figure()
    ax.plot(grid, np.maximum(spec.gain_db(grid), -120), color=BLUE, linewidth=1.4)
    for c in spec.cutoff_hz:
        ax.axvline(c, color=MUTED, linewidth=0.7, linestyle=":")
    if log_axis:
        ax.set_xscale("log")
    _finish(fig, ax, path, title, "Frequency (Hz)", "Gain (dB)")
