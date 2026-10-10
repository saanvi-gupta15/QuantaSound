# Review 1 presentation guide

## Five-minute walkthrough

1. **0:00–0:30 — Problem and scope.** “We are building toward user-aware noise control. Today we demonstrate the signal-processing foundation and two preliminary research modules.” Avoid claiming every existing ANC product behaves identically.
2. **0:30–1:30 — Demo A.** Show engine, siren, keyboard and speech. Point to a sustained band, a changing pitch and a transient. Read the measured dominant-frequency note. Explain that speech and noise can overlap in frequency.
3. **1:30–2:15 — Demo B.** Explain the known 1 kHz target/200 Hz noise, notch band and measured SNR. Say: “This is filtering, not ANC: we removed signal content, not generated acoustic anti-noise.” SNR alone does not prove clean target preservation; show SDR too.
4. **2:15–3:45 — Demo C.** Draw P, W and S. Explain why the reference must be early and why S filters both the control output and the adaptation reference. Show error convergence and final-window noise reduction for tone and low-frequency broadband noise. State perfect secondary-path knowledge and ideal linear simulation.
5. **3:45–4:30 — Teasers.** Show the classifier confusion matrix and its held-out test fold. Change a preference in the QUBO to illustrate the idea. Explain that QAOA is offline, simulated and compared to the exact optimum, not evidence of speedup.
6. **4:30–5:00 — Next review.** User-feedback learning, source separation, secondary-path estimation, latency/hardware experiments and a fair scaling benchmark.

## Questions judges may ask

**Why not just a notch filter?** It removes wanted content in the same frequency band. ANC generates a secondary sound field to reduce the disturbance at a specific location; personalization also needs a way to identify/preserve desired sources.

**Why filtered-x?** The speaker-to-ear path changes the control signal. Filtering the reference through the estimated secondary path approximates the gradient at the error microphone.

**Does this cancel sound throughout the room?** No. The model targets one error microphone. Cancellation is spatially dependent.

**Does the broadband demo prove real-world broadband ANC?** No. It uses low-pass noise, causal preview, known FIR paths and no hardware noise or clipping. These are deliberately favorable teaching conditions.

**How was SNR calculated?** Demo B uses separate known target/noise components, filtered with the same linear filter. Demo C reports disturbance-to-residual power reduction, not speech SNR. Measurement windows are recorded in the JSON.

**Where is the quantum computer?** A four-qubit pure-state simulator implements a single QAOA layer. Its classical optimizer searches circuit angles. Exhaustive classical search is trivial at this size. More frequencies alone do not establish a hard combinatorial task: we must justify discrete choices, interactions and constraints.

**Does it learn my preferences today?** No. The teaser accepts preference weights explicitly. Continuous user feedback and a controller integration are planned work.

## Rehearsal checklist

- Install dependencies; execute `python -m demo1.run --all` once while online.
- Run `python -m pytest -q` and open `Review1.ipynb` with its outputs.
- Keep plots and `results/metrics.json` available if the live notebook kernel fails.
- Listen at modest volume. Never independently normalize before/after clips when demonstrating reduction.
- Explain assumptions alongside every metric; do not describe results as hardware attenuation or quantum advantage.

## One-week ownership plan

| Day | Work | Lead / review |
|---|---|---|
| 1 | Environment, clone, branch workflow, run committed demos | P1 / everyone |
| 2 | Explain spectrograms and measured frequency bands | P4 + P5 / P2 |
| 3 | Filtering and SNR/SDR definitions | P2 / P3 |
| 4 | FxLMS paths, sign convention, convergence | P2 / P3 + P4 |
| 5 | Classifier split and quantum teaser explanation | P1 + P3 |
| 6 | Rehearse, review claims and limitations | Everyone |
| 7 | Freeze demo environment, capture backup plots | P1 + P5 |

The committed code is a starting deliverable. Each presenter should be able to explain and modify their assigned experiment.
