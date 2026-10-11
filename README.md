# QuantaSound
### Quantum-AI Personalized Noise Cancellation · Review 1

A reproducible Python demonstration of the signal-processing foundations for personalized noise cancellation, with preliminary AI and quantum experiments.

**New personalization milestone:** [interactive Python notebook](demo2/Personalization.ipynb) · [feature guide](demo2/README.md) · [four-scenario results](demo2/results/summary.json). Select sounds, hear before/after simulated ANC, measure residual noise and speech effects, and save named preference profiles. Qiskit now implements the offline QAOA circuit alongside the independently checked NumPy simulator.

**Current AI dashboard:** [trained recognition + source separation + simulated ANC](demo4/README.md), launched with `python -m demo4.app` after installing `requirements-ai.txt`. AudioSet-trained AST suggests sounds, the user chooses a target, pretrained AudioSep estimates its waveform, and a control filter produces simulated anti-noise and ear output. This is offline simulation, not live headphone cancellation.

**Legacy Demo 3:** the earlier upload demo uses heuristic labels and a spectral gate, not a trained sound classifier. It is retained for history and is superseded by Demo 4. Do not present it as the full AI/ANC pipeline.

```bash
python -m pip install -r requirements.txt
python -m jupyterlab demo2/Personalization.ipynb
```

**Start here:** [executed review notebook](demo1/Review1.ipynb) · [presentation guide](demo1/REVIEW1_GUIDE.md) · [measured results](demo1/results/metrics.json).

For a kernel-free rehearsal, download [Review1.html](demo1/Review1.html) and open it in a browser. Figures and audio are embedded; rerunning or changing preferences requires the notebook.

| Demo | What works now | Scope |
|---|---|---|
| A | Real engine, siren, keyboard and speech waveforms/mel spectrograms; measured frequency notes | Offline audio analysis |
| B | Butterworth band-stop, before/after spectra, SNR and distortion metrics | Digital filtering, not ANC |
| C | Stateful FxLMS, explicit primary/secondary FIR paths, convergence and noise reduction | Idealized acoustic simulation |
| AI teaser | MFCC + random forest; five ESC-50 classes; held-out official fold | Preliminary isolated-clip baseline |
| Quantum teaser | Four-bit preference QUBO, p=1 QAOA statevector simulation, exact classical baseline | Preliminary offline proof of concept |

This is **not yet a real-time personalized headphone system**. Demo 2 uses known references; Demo 4 integrates a pretrained source estimator. Learned user feedback, continuous low-latency headphone control and hardware validation remain future work. Modern ANC systems can already adapt to fit/environment; personalization is the project direction, not a claim that every existing headphone treats every listener identically. Quantum advantage is a research question.

## Recorded Review 1 results

| Measurement | Result | Interpretation |
|---|---:|---|
| Filtering component SNR improvement | 23.17 dB | Known 1 kHz target and 200 Hz hum + white noise |
| FxLMS tonal noise reduction | 40.99 dB | Final 4–8 s; ideal known paths plus explicit independent ear disturbance |
| FxLMS low-frequency broadband reduction | 24.25 dB | Same measurement window and path assumptions |
| Classifier held-out accuracy | 77.5% (31/40) | Five classes; official fold 5; no source overlap |
| QAOA probability of exact optimum | 26.99% | Four-qubit p=1 toy problem; no quantum advantage |

The complete notebook was executed successfully and **9 tests passed**. Full parameters, component SNR/SDR, whole-run ANC metrics, confusion matrix and QAOA cost gap are in `demo1/results/metrics.json`. Results are educational simulations/baselines, not product specifications.

## Setup (Python 3.12 recommended)

```bash
git clone https://github.com/saanvi-gupta15/QuantaSound.git
cd QuantaSound
python -m venv .venv
```

Activate on Windows PowerShell with `.venv\Scripts\Activate.ps1`, or macOS/Linux with `source .venv/bin/activate`. If PowerShell blocks activation, run `.venv\Scripts\python.exe` directly in place of `python`.

```bash
python -m pip install -r requirements.txt
python -m ipykernel install --user --name quantasound --display-name "Python (QuantaSound)"
python -m jupyterlab demo1/Review1.ipynb
```

Select the **Python (QuantaSound)** kernel, then Run All. The committed notebook already includes outputs for review without rerunning.

## Reproduce without Jupyter

```bash
python -m demo1.run --all
python -m pytest -q
```

First full run downloads 200 five-second ESC-50 WAVs (about 90 MB), metadata/licenses and a short speech example. It creates figures, paired synthetic ANC WAVs and `demo1/results/metrics.json`. Allow a few minutes for first-use downloads and librosa compilation. Downloads need internet; cached runs work offline. Audio datasets are not committed. Do a complete rehearsal before the review.

```bash
python -m demo1.run              # offline B/C + QAOA, writes metrics_core.json
python -m demo1.run --audio      # also real-audio Demo A
python -m demo1.run --classifier # also the five-class classifier
```

`requirements-tested.txt` records the exact resolved environment used for the committed results; `requirements.txt` gives portable version ranges. CI runs deterministic simulations and tests without downloading the dataset.

## Repository map

- `demo1/Review1.ipynb`: review-ready narrative, code, figures, audio players and metrics.
- `demo1/dsp.py`: Butterworth example and sample-by-sample FxLMS.
- `demo1/data.py`: pinned ESC-50 downloads, provenance manifests, speech cache.
- `demo1/classifier.py`: MFCC features and leak-checked train/test split.
- `demo1/quantum.py`: explicit QUBO and small QAOA statevector simulator.
- `demo1/results/`: measured run, presentation PNGs and synthetic WAV pairs.
- `tests/`: acoustic path, adaptation, metrics and quantum-state checks.
- `demo2/`: connected personalized simulation, Python widget controls, local profiles, four-scenario results and Qiskit cross-checks.

## Collaboration

Use one branch per person, e.g. `p1/setup`, `p2/fxlms`, `p3/quantum`, `p4/audio-analysis`, `p5/review-figures`. Pull before starting, commit small coherent changes, and review diffs before merging. Avoid simultaneous notebook edits; put algorithm changes in the Python modules. `python -m demo1.build_notebook` rebuilds a clean notebook and clears saved outputs, so rerun it before committing regenerated presentation material.

Suggested ownership: P1 setup/rehearsal, P2 filtering/FxLMS, P3 algorithm review/quantum teaser, P4 sound analysis/physics explanation, P5 plots and presentation polish. See the [review guide](demo1/REVIEW1_GUIDE.md).

Dataset licensing and attribution are documented in [DATA_SOURCES.md](demo1/DATA_SOURCES.md). No hardware or quantum performance guarantee is implied by these simulations.
