"""Generate the synthetic demo signals and their ground truth.

All signals are built from known parameters with a fixed seed, so the evaluation
can check the agent's answers against exact values. Nothing here is real data.

Run:  python scripts/make_samples.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.io import wavfile
from scipy.signal import chirp

ROOT = Path(__file__).resolve().parents[1]
SAMPLES = ROOT / "samples"
SEED = 7


def write_wav(path: Path, x: np.ndarray, fs: int) -> np.ndarray:
    """Save as 16-bit PCM and return the quantised samples actually stored."""
    pcm = np.round(np.clip(x, -1, 1) * 32767).astype(np.int16)
    wavfile.write(path, fs, pcm)
    return pcm.astype(np.float64) / 32767


def ecg_beat(t: np.ndarray) -> np.ndarray:
    """One PQRST complex as a sum of Gaussians; t is seconds relative to the R-peak."""
    waves = [  # (amplitude mV, centre s, width s)
        (0.12, -0.18, 0.025),   # P
        (-0.14, -0.04, 0.010),  # Q
        (1.00, 0.00, 0.012),    # R
        (-0.22, 0.04, 0.012),   # S
        (0.28, 0.24, 0.045),    # T
    ]
    return sum(a * np.exp(-((t - c) ** 2) / (2 * w ** 2)) for a, c, w in waves)


def make_ecg(rng, fs=500, seconds=20, bpm=72, hum_hz=50.0) -> tuple[np.ndarray, np.ndarray, dict]:
    t = np.arange(int(fs * seconds)) / fs
    clean = np.zeros_like(t)
    r_times = np.arange(0.4, seconds - 0.4, 60.0 / bpm)
    for r in r_times:
        clean += ecg_beat(t - r)
    hum = 0.25 * np.sin(2 * np.pi * hum_hz * t)
    wander = 0.20 * np.sin(2 * np.pi * 0.3 * t)
    noise = 0.01 * rng.standard_normal(len(t))
    truth = {"fs": fs, "bpm": bpm, "hum_hz": hum_hz, "beats": int(len(r_times))}
    return t, clean + hum + wander + noise, truth


def make_vibration(rng, fs=5000, seconds=4) -> tuple[np.ndarray, np.ndarray, dict]:
    """Rotating-machine style signal: shaft harmonics plus decaying impacts at a fault rate."""
    t = np.arange(int(fs * seconds)) / fs
    shaft, fault_rate, resonance = 29.5, 107.0, 1800.0
    x = sum(a * np.sin(2 * np.pi * shaft * k * t) for k, a in ((1, 0.5), (2, 0.2), (3, 0.1)))
    for t0 in np.arange(0.01, seconds, 1 / fault_rate):
        tail = t - t0
        on = tail >= 0
        x[on] += 0.4 * np.exp(-tail[on] * 400) * np.sin(2 * np.pi * resonance * tail[on])
    x = x + 0.05 * rng.standard_normal(len(t))
    return t, x, {"fs": fs, "shaft_hz": shaft, "fault_rate_hz": fault_rate, "resonance_hz": resonance}


def main() -> None:
    SAMPLES.mkdir(exist_ok=True)
    rng = np.random.default_rng(SEED)
    truth: dict = {}

    # ECG with mains hum and baseline wander (CSV with a time column)
    t, ecg, truth["ecg_50hz_hum"] = make_ecg(rng)
    np.savetxt(SAMPLES / "ecg_50hz_hum.csv", np.column_stack([t, ecg]), delimiter=",",
               header="time_s,ecg_mv", comments="", fmt=["%.4f", "%.6f"])

    # Three tones in white noise
    fs = 16000
    t = np.arange(fs * 3) / fs
    tones = [440.0, 1000.0, 3000.0]
    x = sum(0.2 * np.sin(2 * np.pi * f * t) for f in tones) + 0.02 * rng.standard_normal(len(t))
    write_wav(SAMPLES / "three_tones.wav", x, fs)
    truth["three_tones"] = {"fs": fs, "tones_hz": tones}

    # Linear chirp in noise
    t = np.arange(fs * 4) / fs
    x = 0.5 * chirp(t, f0=100, t1=t[-1], f1=4000, method="linear") + 0.03 * rng.standard_normal(len(t))
    write_wav(SAMPLES / "chirp_noisy.wav", x, fs)
    truth["chirp_noisy"] = {"fs": fs, "f_start_hz": 100, "f_end_hz": 4000}

    # Vibration record (CSV with a time column)
    t, x, truth["vibration"] = make_vibration(rng)
    np.savetxt(SAMPLES / "vibration.csv", np.column_stack([t, x]), delimiter=",",
               header="time_s,accel_g", comments="", fmt=["%.5f", "%.6f"])

    # Clean / noisy tone pair with a known SNR, measured on the stored 16-bit samples
    t = np.arange(fs * 2) / fs
    clean = 0.5 * np.sin(2 * np.pi * 1000 * t)
    target_snr_db = 10.0
    noise_sigma = np.sqrt(np.mean(clean ** 2) / 10 ** (target_snr_db / 10))
    noisy = clean + noise_sigma * rng.standard_normal(len(t))
    stored_clean = write_wav(SAMPLES / "tone_clean.wav", clean, fs)
    stored_noisy = write_wav(SAMPLES / "tone_noisy.wav", noisy, fs)
    actual = 10 * np.log10(np.mean(stored_clean ** 2) / np.mean((stored_noisy - stored_clean) ** 2))
    truth["tone_noisy"] = {"fs": fs, "tone_hz": 1000.0, "snr_db": round(float(actual), 3)}

    (SAMPLES / "ground_truth.json").write_text(json.dumps(truth, indent=2), encoding="utf-8")
    print(f"Wrote {len(list(SAMPLES.iterdir()))} files to {SAMPLES.name}/")


if __name__ == "__main__":
    main()
