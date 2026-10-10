# Personalized ANC: four connected features

Open [Personalization.ipynb](Personalization.ipynb) in JupyterLab for working controls. The [HTML copy](Personalization.html) includes fixed example results and audio; controls require the notebook's Python kernel.

```bash
python -m pip install -r requirements.txt
python -m jupyterlab demo2/Personalization.ipynb
```

Select the environment where requirements were installed, then Run All. In the interface, choose engine and/or keyboard to reduce; speech remains designated for preservation. Click Run cancellation, compare playback and read the metrics. Enter a profile name and Save preferences. Choose a saved name and Load profile to restore it. Reopening automatically loads a saved `default` profile if one exists.

### If before and after sound similar

There is **no siren** in these mixtures. The original engine example is very low bass: the removed signal peaks around 37 Hz and about 93% of its energy is below 100 Hz. The full mixed recording changes by only about 1.87 dB; wanted speech and keyboard clicks remain. This can make the difference difficult to hear on small speakers. The embedded audio was checked against the exported WAVs; the before and after files are distinct and correctly embedded.

The notebook now starts with a separately labeled **audible-hum demonstration**: a generated 200 Hz tone with 400/600 Hz harmonics (RMS 0.16), speech (RMS 0.12) and quieter keyboard clicks (RMS 0.025). It is an easier synthetic teaching case, not improved performance on the real-engine benchmark. The scene selector defaults to this example; choose the original recording to reproduce the previous results. Sound preferences persist, while scene choice is an experiment setting and is not saved in the preference profile.

**Hear each source** provides solo auditions, individually scaled for identification only. The new **quick A/B** player plays the final five seconds before, half a second of silence, then the same five seconds after, using exactly the existing pair's shared gain. This makes comparison easier without increasing the claimed reduction. Keep the player volume unchanged; avoid playing before and after simultaneously.

## What changed

1. **Selection:** choose engine, keyboard, both or neither for a mixture of actual recordings.
2. **Playback:** before/after simulated ear signals share one gain; exported PCM WAVs are playable/downloadable from the Jupyter file browser.
3. **Measurements:** total noise reduction, selected-target residual reduction, speech-reference SDR and speech counterfactual distortion/gain, plus convergence and spectral plots.
4. **Persistence:** versioned local JSON profiles with validation and atomic replacement; saved choices survive kernel and app restarts. They are explicit settings, not inferred preferences.

`demo2/local/` contains personal profiles and interactive exports and is ignored by Git. Run cancellation does not automatically save preferences. Saving an existing profile name updates it. The latest interactive export is overwritten on the next run.

## Scientific scope

We mix the source tracks ourselves. The separately known selected tracks form an early noise reference. The reference is not inferred from the mixture and labels are known from source metadata. scikit-learn's classifier remains a separately evaluated baseline; it cannot by itself separate overlapping waveforms.

The controller adapts on the full simulated error-microphone signal including speech and unselected sources. Its anti-noise is propagated through a stateful secondary FIR; the implementation does not simply subtract the known target from the output. The primary path is shared among all sources, stationary and known. The speaker model is low-pass, so high-frequency keyboard clicks are difficult. Source repetition, perfect reference preview and exact secondary-path knowledge make this a favorable simulation.

### Recorded results (final 5–10 seconds)

| Selection | Total noise reduction | Selected-target residual reduction | Speech counterfactual error RMS |
|---|---:|---:|---:|
| Preserve all | 0.00 dB | Not applicable | 0.00% |
| Reduce engine | 3.73 dB | 12.45 dB | 15.55% |
| Reduce keyboard | 0.89 dB | 2.49 dB | 4.55% |
| Reduce both | 6.35 dB | 6.35 dB | 16.44% |

These results show partial reduction, not perfect removal. Speech is designated for preservation but the adaptive controller still perturbs its counterfactual contribution. The engine case's stronger target reduction comes with greater speech-related adaptation error in this experiment. No tuning against these clips is claimed as generalization to unseen sounds.

The paired speech-present/absent rerun measures how speech changes adaptation. It is a simulation diagnostic, not a perceptual score. Total-noise reduction includes unselected sounds, so selecting only one source may give modest total reduction even if the target is reduced. Negative values indicate amplification and are not hidden. No device attenuation claims are made.

## Python stack

- NumPy/SciPy: mixed-source signal paths, FxLMS, metrics and QAOA's classical optimizer.
- librosa/soundfile: audio loading, resampling and WAV export.
- Matplotlib/ipywidgets: plots and interactive notebook controls.
- scikit-learn: existing MFCC/random-forest baseline in `demo1/classifier.py` (not fake automatic labeling of this mixture).
- **Qiskit:** real `QuantumCircuit`, cost `DiagonalGate`, X-mixer rotations and `Statevector` simulation for the offline QAOA research module. Tests compare its probabilities with the independent NumPy simulator. No account or quantum cloud hardware is used.

The toy QUBO does not drive the cancellation controls. Its exactly-two soft budget is educational and must be replaced by a justified acoustic optimization model before integration.

## Reproduce and test

```bash
python -m demo2.run
python -m pytest -q
```

The first command generates all four scenarios, WAV pairs, plots, full metrics and a Qiskit benchmark under `demo2/results/`. Initial audio download requires internet; subsequent runs use the cache from Review 1. `demo2/results/summary.json` records source gains, repetition and dependency versions. The notebook separately executes engine and keyboard examples to keep its review length manageable.

To rebuild and execute the notebook:

```bash
python -m demo2.build_notebook
python -m demo1.execute_notebook demo2/Personalization.ipynb
python -m jupyter nbconvert --to html --embed-images demo2/Personalization.ipynb
```

## Audio attribution

Uses the same engine/keyboard ESC-50 recordings and LibriSpeech `libri1` speech file as Review 1. See [source attribution and retained licenses](../demo1/DATA_SOURCES.md). The five-second environmental recordings repeat to ten seconds; the first ten seconds of narration are used. Tracks are centered and RMS-scaled. The notebook, HTML and exported before/after WAVs contain adapted mixes/excerpts subject to those source licenses, including ESC-50's noncommercial restriction. Raw full datasets remain excluded from Git.
