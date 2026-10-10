"""Build the Python interactive personalization notebook."""
from pathlib import Path
import nbformat as nbf


def main():
    md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
    cells = [
        md("""# QuantaSound · Personalized cancellation
## Choose sounds → simulate anti-noise → listen → measure → save preferences

**New:** sound-selection controls, paired playback, performance measurements and persistent profiles.

This experiment mixes real speech, engine and keyboard recordings. Separate source tracks provide **ideal known references**; we are not separating sounds from one microphone mixture. FxLMS generates a control signal, propagates it through the simulated speaker-to-ear path and adds the opposing sound at the ear. This is an offline simulation, not physical headphone ANC.

Use Python 3.12, install `requirements.txt`, open this notebook in JupyterLab and Run All. The controls need a running Python kernel; GitHub and exported HTML only show the saved examples."""),
        code("""from pathlib import Path
import sys, json
root = Path.cwd()
if not (root/'demo2').exists():
    root = root.parent
assert (root/'demo2').exists(), 'Open from the repo root or demo2 folder'
sys.path.insert(0, str(root))
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display, Audio, Image
from demo2.simulation import load_sources, simulate
from demo2.presentation import show_result, metric_rows
%matplotlib inline"""),
        md("""## 1 · Your sound preferences
Select engine and/or keyboard to reduce; speech is always designated for preservation. Leave both unchecked to hear the no-cancellation baseline. Click **Run cancellation** to compute the result. Click **Save preferences** to store a named profile. The saved-profile dropdown selects a name; click **Load profile** to apply it. A profile named `default` is restored when the controls reopen.

Profiles are plain local JSON under `demo2/local/profiles/`, excluded from Git. They store deliberate choices; no preference learning is claimed. The current result goes to `demo2/local/latest/`, replacing the previous local run."""),
        code("""from demo2.ui import show_controls
controls = show_controls()"""),
        md("""## 2 · Reproducible example: reduce engine, preserve speech and keyboard
The following cells run independently of the controls so the notebook has reviewable saved outputs. We repeat the five-second environmental recordings to ten seconds. Speech uses its first ten seconds. All tracks are resampled to 16 kHz, centered and scaled to documented RMS levels; repeated material is an easier stationary teaching case.

Assumptions: one shared primary FIR with 32-sample delay, a known secondary FIR with a low-pass speaker model (1.5 kHz cutoff), 64 controller taps and μ=0.004. The reference arrives early enough for a causal controller. An independent ear disturbance sets a finite noise floor. There is no actuator clipping or changing acoustic path."""),
        code("""sources, provenance = load_sources()
engine_result = simulate(sources, ['engine'])
show_result(engine_result, root/'demo2'/'local'/'notebook_engine')
print(json.dumps(provenance, indent=2))"""),
        md("""## 3 · Experimental case: reduce keyboard, preserve speech and engine
Keyboard impacts span a broad frequency range. The modeled low-pass secondary path limits what can be canceled. Separately known early references are favorable assumptions; this result is not a real-world keyboard-removal guarantee. Negative reduction means the controller made the measured residual worse, and is retained honestly."""),
        code("""keyboard_result = simulate(sources, ['keyboard'])
show_result(keyboard_result, root/'demo2'/'local'/'notebook_keyboard')"""),
        md("""## 4 · Reading the measurements
- **Total residual-noise reduction:** 10 log₁₀(power(before − clean speech) / power(after − clean speech)). This includes unselected noise and any controller artifacts; it need not be large when preserving another noisy source.
- **Selected-target residual reduction:** compare selected-source-plus-floor power with `after − wanted`. Wanted includes speech and unselected sounds. This is a reference-based residual metric, not an isolated-source measurement from a real microphone.
- **Speech-reference SDR:** speech power divided by error power relative to clean speech; includes remaining interference. No delay or gain fitting is used.
- **Speech counterfactual distortion:** repeat adaptation with speech absent, holding reference, noise floor and paths fixed. Subtract that output from the speech-present output. Compare the difference with clean speech at the ear; report error RMS as a percentage of speech RMS. This captures speech-driven adaptation effects, but is not a perceptual intelligibility score or a source-separation result.
- **Speech counterfactual gain:** projection of that difference onto clean speech; near 1 is desirable. Read it alongside distortion, not alone.

Metrics use the final five seconds; whole-run noise reduction is also saved. All outputs are simulated amplitudes, not calibrated sound-pressure levels. Both listening files use the same gain. Keep player/system volume unchanged between before and after."""),
        md("""## 5 · Save and reload preferences in Python
The interface uses these same functions. This isolated temporary-directory example demonstrates persistence without overwriting your real saved profiles."""),
        code("""import tempfile
from demo2.profiles import save_profile, load_profile
with tempfile.TemporaryDirectory() as example_directory:
    saved = save_profile('student', ['keyboard'], example_directory)
    restored = load_profile('student', example_directory)
    assert restored == saved
    print('Reloaded from disk:', restored)"""),
        md("""## 6 · Qiskit offline research module
Qiskit executes a four-qubit p=1 QAOA circuit using its statevector simulator. We cross-check its output probabilities against the existing NumPy implementation and compare against exhaustive classical search. SciPy optimizes the circuit angles; no IBM account, quantum device or cloud service is required.

This separate toy QUBO has an artificial exactly-two soft budget and four sound choices. Its results **do not override the sound-selection controls or drive FxLMS**. It is not yet a physically calibrated acoustic optimizer. scikit-learn continues to implement the five-class MFCC/random-forest classifier in `demo1/classifier.py`; classification is separate from obtaining cancellation waveforms."""),
        code("""from demo1.quantum import benchmark, problem, statevector, qiskit_statevector, qiskit_circuit
_, costs = problem()
angles = [.3, .7]
np.testing.assert_allclose(np.abs(statevector(angles, costs))**2,
                           np.abs(qiskit_statevector(angles, costs))**2, atol=1e-12)
states, costs, probabilities, qiskit_metrics = benchmark(backend='qiskit')
print(json.dumps(qiskit_metrics, indent=2))
print(qiskit_circuit(qiskit_metrics['angles'], costs).draw(output='text'))
fig, ax = plt.subplots(figsize=(10, 3), layout='constrained')
ax.bar(range(16), probabilities)
ax.set_xticks(range(16), [''.join(str(int(v)) for v in row) for row in states], rotation=45)
ax.set(xlabel='Bits: engine / typing / speech / siren', ylabel='Probability', title='Qiskit QAOA — preliminary offline toy problem')
plt.show()"""),
        md("""## What this milestone proves
The user's saved choices now affect a connected ANC simulation with real-source mixtures, generated anti-noise, listening outputs and quantitative evaluation. It does not yet prove microphone-only source separation, live low-latency cancellation, learned preferences or quantum advantage.

Run `python -m demo2.run` to export all four selections to `demo2/results/`; `python -m pytest -q` checks DSP, profile storage, widget callbacks and Qiskit/NumPy agreement. Source attribution is in `../demo1/DATA_SOURCES.md`, with the ten-second speech excerpt/repeated environmental clips documented in `README.md` here.

References: [Qiskit Statevector](https://docs.quantum.ibm.com/api/qiskit/2.0/qiskit.quantum_info.Statevector), [QAOA paper](https://arxiv.org/abs/1411.4028), [ESC-50](https://github.com/karolpiczak/ESC-50).""")
    ]
    for i, cell in enumerate(cells):
        cell.id = f"personalization-{i:02d}"
    notebook = nbf.v4.new_notebook(cells=cells, metadata={
        "kernelspec":{"display_name":"Python 3", "language":"python", "name":"python3"},
        "language_info":{"name":"python", "version":"3.12"}})
    nbf.write(notebook, Path(__file__).with_name("Personalization.ipynb"))


if __name__ == "__main__":
    main()
