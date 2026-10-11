# AI audio pipeline

Run from the repository root:

```powershell
python -m pip install -r requirements.txt
python -m demo3.app
```

The local Gradio page accepts an uploaded audio file or a microphone recording. It classifies the mixture with an explainable scikit-compatible signal-feature triage, estimates a stationary noise floor from the opening segment, applies a conservative spectral gate, and returns a denoised WAV plus a JSON report.

Each run displays only the useful review evidence: one before/after waveform, one before/after spectrum, the audio output and a short report with detected mixture, level change and ANC limitation.

This is the correct working AI path for arbitrary audio today: classification and denoising can operate without a known noise recording. It is not falsely labeled as physical ANC. Spectral attenuation can reduce or distort wanted content. An unknown upload does not provide the reference needed to generate correctly timed anti-noise. The Personalization notebook demonstrates reference-based FxLMS separately.

For the faculty presentation, show the same clip before/after, the detected class, the estimated level change, and the limitation text. Then explain the research bridge: source separation or a reference microphone would feed a selected component into FxLMS; the upload denoiser is the fallback when no correlated reference exists.
