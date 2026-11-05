# Evaluation: gemini / gemini-3.5-flash-lite

| Request | Result | Tools | Measured |
|---|---|---|---|
| Remove the 50 Hz hum and show the spectrum | PASS | design_filter, apply_filter, compute_fft | 50 Hz changed -54.0 dB (need <= -20), 10 Hz changed -0.00 dB, spectrum shown: True |
| What are the dominant frequencies in this signal? | PASS | compute_fft | tones found: {440.0: True, 1000.0: True, 3000.0: True}, answer mentions 440: True |
| Show me a spectrogram of the chirp | PASS | compute_spectrogram | sweep measured 125.0 -> 3968.8 Hz, truth 100 -> 4000 Hz (+/-100) |
| Low-pass at 1 kHz with a 4th-order Butterworth and compare before and after | PASS | design_filter, apply_filter, compute_fft | 3 kHz changed -84.2 dB (need <= -30), 440 Hz changed -0.01 dB, design butterworth order 4 at [1000.0] Hz, answer states the applied cutoff gain: True |
| What is the SNR of the noisy tone? | PASS | compute_snr | SNR 10.0 dB by reference method, truth 10.002 dB (+/-1) |
| Detect the R-peaks and estimate the heart rate | PASS | compute_fft, design_filter, apply_filter, detect_peaks | 72.0 bpm from 24 peaks, truth 72 bpm (+/-3) |

**6/6 passed.**
