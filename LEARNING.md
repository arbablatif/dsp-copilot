# Learning notes: DSP Copilot

Plain-language notes on how this project works and why it was built this way, so you can explain it to a client without reading the code.

## The one-minute explanation

"You describe what you want done to a signal, in English. An AI model decides which standard signal-processing steps to run, in what order, and with what settings. The steps themselves are ordinary, tested engineering code, so the numbers are real. The model then explains the result, and you can see every step it took."

## How the agent works

### What is tool calling?

A language model only produces text. Tool calling gives it a menu of functions, each with a description and a list of arguments. Instead of answering, the model can reply with a structured request such as:

```json
{"name": "design_filter", "args": {"signal": "ecg", "method": "notch", "response": "notch", "cutoff_hz": [50]}}
```

Our code runs that function and returns the result as text. The model reads it and either calls another tool or gives its final answer. That cycle is the **agent loop** (`dsp_copilot/agent.py`).

For "remove the hum and show the spectrum" the live model made three calls: design a notch, apply it, then compare the spectra. Nobody scripted that sequence.

### Why tools, instead of letting the model write code?

- **Safety.** Model-written code could do anything on the machine. Here, only 8 known functions can run.
- **Accuracy.** Language models are unreliable at arithmetic. SciPy isn't.
- **Auditability.** Each step is a named action with arguments that a person can read.

The cost is flexibility: the agent can't do anything outside its 8 tools.

### Why do errors go back to the model?

If the model asks for a 9 kHz cutoff on a signal whose Nyquist limit is 8 kHz, the tool returns the reason as text instead of crashing. The model can then fix the request. This happened in a live run: it asked for 250 Hz, was refused because of a bug of ours, and retried with 249 Hz by itself.

### Why is there a step limit?

A model could keep calling tools forever, costing money. The loop stops after 8 model turns.

### What are the cache and mock mode for?

- **Cache:** each reply is saved to disk under a fingerprint of the whole conversation, so an identical conversation is free. Change anything the model sees, even one number, and the fingerprint changes. That's why a small tool change forced some paid calls again.
- **Mock mode:** a simple keyword planner replaces the model, so everything else runs for real with no API key. It's useful for tests and demos, but it proves the plumbing, not the intelligence.

### Why can the provider be switched?

Each provider formats tool calls differently. The agent uses one neutral format, and a small adapter translates it for Gemini, Anthropic or OpenAI (`dsp_copilot/llm.py`). One detail: Gemini's thinking models return a "thought signature" with each tool call, and it must be sent back on the next turn, so we store it.

Only the Gemini adapter has been tested against a real API so far.

## The signal-processing concepts

### Sample rate and Nyquist

The sample rate is how many measurements are taken per second. The **Nyquist frequency** is half of that, and it's the highest frequency the data can represent. The ECG is sampled at 500 Hz, so nothing above 250 Hz can be seen or filtered. That's why cutoffs are checked against it.

### FFT and spectrum

The FFT rewrites a signal as a sum of sine waves and shows how strong each frequency is. A hum that's hidden in the waveform shows up as one sharp spike at 50 Hz.

### Windowing

The FFT assumes the signal repeats perfectly. Real recordings don't, and the mismatch smears energy into nearby frequencies ("leakage"). A window, such as the Hann window used here, fades the ends of the signal to reduce this. In exchange, each peak gets slightly wider.

### Decibels

Decibels compare levels on a logarithmic scale. For amplitude, −6 dB is about half, −20 dB is a tenth, and −60 dB is a thousandth. For power, −3 dB is half.

### Spectrogram

An FFT says which frequencies exist, but not when. A spectrogram runs many short FFTs along the signal to show frequency over time, so a chirp appears as a rising line. There's a built-in trade-off: longer windows give sharper frequency detail but blurrier timing.

### FIR and IIR filters

| | FIR | IIR (Butterworth, Chebyshev, notch) |
|---|---|---|
| How it works | Weighted average of recent inputs | Also feeds back earlier outputs |
| Stability | Always stable | Must be checked (the tool does) |
| Phase | Can be perfectly linear, so the shape is preserved | Non-linear |
| Cost | Needs many coefficients | Sharp with very few |

Butterworth is as flat as possible in the passband. Chebyshev I rolls off faster but allows some ripple. A **notch** removes one narrow frequency, which suits mains hum. Its **Q** sets the width: at 50 Hz with Q=30, the notch is about 1.7 Hz wide.

### Zero-phase filtering

Normal filters delay a signal, which would shift the timing of ECG peaks. Zero-phase filtering runs the filter forward, then backward, which cancels the delay.

The catch is that the filter is **applied twice**, which doubles the attenuation in dB. A 4th-order Butterworth is −3 dB at its cutoff, but −6 dB when run zero-phase. It also needs the whole recording, so it can't run in real time. This detail caused a real mistake in this project (see below).

### Edge transients

Filters need time to settle. A narrow notch rings for about Q / (π × f₀), or roughly 0.2 s here, so the start and end of a filtered signal are less clean than the middle.

### Peak detection

A peak is a local maximum, but most of them are noise. Two settings do the real work: **prominence** (how far it stands above its surroundings) and **minimum distance** (how close two events can be).

Filtering first matters. QRS energy sits at roughly 5 to 30 Hz, so a band-pass there removes baseline drift, T-waves and hum. On the raw ECG the detector found 42 peaks (about 128 bpm). On the filtered ECG it found the correct 24 (72 bpm).

### SNR

Signal-to-noise ratio compares wanted power with unwanted power, in dB. There are two methods:

- **Reference:** if you have the clean signal, noise = noisy − clean. This is exact, but you rarely have one.
- **Band:** treat a narrow band around a tone as signal and everything else as noise. It works without a reference, but only for narrowband signals.

On the test tone they gave 10.0 and 10.1 dB, against a true 10.0 dB.

## Design decisions worth mentioning

- **Synthetic signals with ground truth.** Because we built each signal, we know the correct answer exactly. That makes automatic checks possible. It also means the results don't show performance on messy real data.
- **Check the data, not the claim.** The evaluation measures the agent's output signals directly. It doesn't trust the agent's statements.
- **Plots stay out of the model's context.** It's told a plot was shown, but it can't see images here, and leaving file names out keeps the cache reusable.
- **The original signal is never changed.** Filtering creates a new signal, so before and after can always be compared.

## The most useful lesson: a passing score isn't the whole story

The first live run scored 6 out of 6, yet reading the answers showed two were wrong. One said −3 dB where −6 dB had been applied. Another said a chirp started at 312.5 Hz when it really started at 100 Hz.

Both times the model had faithfully repeated a misleading tool output. One tool reported the design gain, not the gain applied. The other used a loose definition of "start". The checks missed both, because they looked at the signals, not the explanations.

Then a tightened check wrongly failed a correct answer, because it banned "−3 dB" and the answer mentioned that figure while explaining the correct −6 dB. So checks need reviewing too.

For a client, the takeaway is: **a model is only as accurate as the information its tools give it, so name and report tool outputs precisely, and have a person read real outputs as well as the score.**

## Questions a client might ask

**Can it get things wrong?** Yes. The numbers come from tested code, but the model can choose poor settings or word an explanation badly. That's why every step is displayed. Treat the tool trace as the record.

**Does our data go to the AI provider?** The samples don't. The model receives file names, summary statistics and tool results. In mock mode nothing leaves the machine.

**Can it handle our real data?** It reads CSV and WAV, but it has only been evaluated on synthetic signals. Real data would need its own evaluation set.

**Can we add our own analysis?** Yes. Add a function, a schema entry and a dispatch line in `dsp_copilot/tools.py`, and the model can use it.

**What does it cost to run?** A typical request took 2 to 5 model turns on the cheapest Gemini Flash model, and repeated requests are free from the cache. Check current provider pricing for an exact figure.
