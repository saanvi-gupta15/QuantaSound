"""Regenerate the presentation notebook (execution is a separate explicit step)."""
from pathlib import Path
import nbformat as nbf


def main():
    md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
    cells = [
        md("""# QuantaSound · Review 1
## Quantum-AI Personalized Noise Cancellation
**Working foundations:** sound analysis → ordinary filtering → simulated FxLMS ANC.

**Research teasers:** isolated-clip classification and offline QAOA. These are separate experiments, not an integrated personalized headphone product.

Run all cells after installing `requirements.txt`. First run downloads recordings; later runs use the cache. Keep playback quiet and avoid normalizing before/after independently."""),
        code("""from pathlib import Path
import sys, json
root = Path.cwd()
if not (root / 'demo1').exists():
    root = root.parent
assert (root / 'demo1').exists(), 'Open the notebook from the repo root or demo1/'
sys.path.insert(0, str(root))
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display, Audio
from demo1.plots import demo_a, demo_b, demo_c, RESULTS
from demo1.data import demo_clips
plt.rcParams.update({'font.size': 10})
%matplotlib inline"""),
        md("""## A · What noise looks like
Real ESC-50 engine, siren and keyboard clips, plus LibriSpeech narration via librosa (ESC-50 has no speech class). Waveform = amplitude over time; mel spectrogram = time-varying spectral energy on a perceptual frequency axis. The note above each plot reports the measured PSD peak and 10–90% cumulative-energy band. A peak alone does not describe the whole sound. Spectrogram colors are relative **within each clip**, not calibrated loudness between clips."""),
        code("""figures, spectral_notes = demo_a()
for figure in figures:
    display(figure)
    plt.close(figure)
spectral_notes"""),
        code("""# Optional listening: real recordings, first five seconds, no autoplay.
import librosa
for label, path in demo_clips():
    samples, sr = librosa.load(path, sr=16000, duration=5)
    print(label)
    playback = samples * (0.9 / max(1.0, float(np.max(np.abs(samples)))))
    display(Audio(playback, rate=sr, normalize=False))
# Playback gain only prevents resampling overshoot; spectral plots use the original samples."""),
        md("""## B · Ordinary digital filtering — not ANC
Known target: 1 kHz sine. Interference: 200 Hz hum plus weak white noise. A third-order Butterworth band-stop removes 150–250 Hz using causal `lfilter`.

**This is filtering, not ANC, because it removes content instead of generating acoustic anti-noise.** Any wanted sound in the rejected band would also be removed.

SNR = 10 log₁₀(target power / noise power). Output component SNR filters the known target and noise separately, so it does not hide which component is measured. Clean-reference SDR additionally penalizes phase/amplitude distortion relative to the original target. Both exclude the initial 0.25 s; no alignment or gain fitting is used."""),
        code("""figure, filtering_metrics = demo_b()
display(figure); plt.close(figure)
print(json.dumps(filtering_metrics, indent=2))"""),
        md("""## C · Adaptive ANC with FxLMS (simulation)
`reference x → primary path P → disturbance d at ear`

`reference x → adaptive FIR W → y → secondary path S → cancellation contribution`

`error e = d − S*y`; update `w ← w + μ e x_filtered`, where `x_filtered = S_hat*x`.

The physical speaker command is **−y** with this sign convention. The secondary path stores past controller samples; it is not an instantaneous multiply. We assume `S_hat = S`, a stationary linear acoustic path and an ideal external reference. Primary delay (6 samples) exceeds secondary delay (2 samples), allowing a causal controller. These favorable assumptions must be stated to judges. An independent Gaussian disturbance (standard deviation 0.003, seed 107) is added at the ear in both cases; it cannot be predicted from the reference and creates an explicit residual noise floor.

Report noise reduction = 10 log₁₀(mean(d²)/mean(e²)) on the **same final 4 seconds**; whole-run reduction also includes adaptation. These numbers describe this synthetic experiment, not headphone attenuation specifications."""),
        code("""anc_metrics = {}
for kind in ('tonal', 'broadband'):
    figure, anc_metrics[kind] = demo_c(kind)
    display(figure); plt.close(figure)
    print(kind, json.dumps(anc_metrics[kind], indent=2))"""),
        code("""# Exported pairs use the SAME gain: preserve the actual level difference.
for kind in ('tonal', 'broadband'):
    for stage in ('before', 'after'):
        print(kind, stage)
        display(Audio(filename=str(RESULTS / f'c_{kind}_{stage}.wav')))
# These sounds were generated in simulation; no microphone/speaker feedback loop runs here."""),
        md("""### Explain the physics and limits
Destructive interference reduces pressure at the modeled error microphone. Lower frequencies allow more timing tolerance; broadband cancellation also needs reference preview and an accurate secondary-path model. Our broadband input is low-pass noise below approximately 900 Hz, not unrestricted white noise. Without the explicitly added independent ear disturbance, a perfectly predictable tone can approach numerical precision. Real hardware adds further sensor noise, latency, clipping, changing fit and spatial effects.

If divergence occurs, reduce μ and check the subtraction/update sign. If adaptation does nothing, inspect the secondary-path history and filtered reference. This demo does not yet include RLS, live audio I/O, sound separation or online user learning."""),
        md("""## Preliminary teaser · Five-class classifier
20 MFCC means + 20 standard deviations → 200-tree random forest. Classes: engine, siren, keyboard, rain, dog. Use all 200 clips in these classes: official folds 1–4 train; fold 5 tests. Source recording IDs are checked for leakage. No test-fold tuning. Chance accuracy is 20% on this balanced subset.

This is a small **isolated single-label** classification baseline. It does not identify simultaneous overlapping sources or classify speech. A single 40-clip test fold is preliminary, not a general performance claim."""),
        code("""from demo1.classifier import run_classifier
figure, classifier_metrics = run_classifier()
display(figure); plt.close(figure)
print(json.dumps(classifier_metrics, indent=2))"""),
        md("""## Preliminary teaser · User preferences and offline QAOA
Four binary decisions `z_i`: request cancellation of engine, typing, speech, siren. Positive preference rewards blocking; negative preference penalizes it. Edit the values to illustrate personalization **inputs**, not learned preferences.

Toy QUBO: `E(z) = −Σ reward_i z_i + 0.8 z_engine z_typing + 2(Σ z_i − 2)²`, with `reward = [3,2,4,5] × preference`. This includes an artificial interaction and a soft exactly-two budget. It is a decision illustration, not an acoustic transfer-function optimizer; preserving an overlapping source requires separation/control methods beyond these four bits.

The NumPy simulator prepares a uniform four-qubit state, applies the diagonal cost phase and X mixers, then optimizes p=1 QAOA angles with four classical restarts. Exhaustive evaluation of all 16 choices supplies the exact classical baseline. Compare expected cost, gap and optimum probability. Simulator timing is not quantum hardware timing. No speedup or scaling advantage is established."""),
        code("""from demo1.quantum import benchmark
preferences = [1.0, 0.6, -1.0, -1.0]  # engine, typing, speech, siren
states, costs, probabilities, quantum_metrics = benchmark(preferences)
print(json.dumps(quantum_metrics, indent=2))
fig, ax = plt.subplots(figsize=(10, 4), layout='constrained')
ax.bar(range(16), probabilities)
ax.set_xticks(range(16), [''.join(str(int(bit)) for bit in row) for row in states], rotation=45)
ax.set(xlabel='Block bits: engine / typing / speech / siren', ylabel='Probability', title='PRELIMINARY p=1 QAOA simulation')
plt.show()"""),
        md("""## Review 1 conclusion and next experiments
We have demonstrated real sound analysis, measured digital filtering, and converging acoustic-path ANC simulation. A preliminary classifier and a toy preference-dependent quantum optimization problem are reproducible offline. Continuous user feedback, overlapping-source separation, an integrated policy/controller and physical ANC validation remain future work.

For Review 2: test secondary-path mismatch and changing environments; define safe preserve/block behavior; add source-aware preference learning; measure hardware latency. For quantum research, derive a physically justified QUBO and compare quality/runtime against strong classical solvers across instance sizes and multiple seeds.

### Sources
- [ESC-50 dataset and prescribed folds](https://github.com/karolpiczak/ESC-50); see downloaded `data/ESC50_LICENSE.txt` for per-recording attribution. Dataset: CC BY-NC 3.0.
- [librosa speech examples](https://librosa.org/doc/0.11.0/recordings.html); `libri1`, narration by Garth Comira; see `DATA_SOURCES.md`.
- [FxLMS review](https://www.mdpi.com/2076-3417/8/11/2313).
- [Farhi et al., QAOA](https://arxiv.org/abs/1411.4028).

See `REVIEW1_GUIDE.md` for the presentation script and `results/metrics.json` for the recorded full run.""")
    ]
    for i, cell in enumerate(cells):
        cell.id = f"review1-{i:02d}"
    notebook = nbf.v4.new_notebook(cells=cells, metadata={
        "kernelspec":{"display_name":"Python 3", "language":"python", "name":"python3"},
        "language_info":{"name":"python", "version":"3.12"}})
    nbf.write(notebook, Path(__file__).with_name("Review1.ipynb"))


if __name__ == "__main__":
    main()
