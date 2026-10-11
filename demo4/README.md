# QuantaSound · trained AI and simulated selective ANC

The intended product is an app that recognizes background sounds, lets the user
choose what to suppress, and controls compatible ANC headphones. **This review
prototype is offline software; it does not cancel the room through ordinary earphones.**

## Run

Use Python 3.12 in an activated virtual environment, from the repository root:

```powershell
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-ai.txt
python -m demo4.download_models
python -m demo4.download_models --separator
python -m demo4.evaluate --prepare-only
python -m demo4.app
```

Open http://127.0.0.1:7860. If an older demo is running there, stop that terminal
with Ctrl+C first, or use `python -m demo4.app --port 7861`.
After setup on Windows, double-click `Start-QuantaSound.cmd` in the repository
folder to launch with its virtual environment. Keep the terminal window open.
Models download once (about 1.3 GB combined, plus small tokenizer files) into
ignored `demo4/local/`. Allow several GB of RAM/disk space. CPU inference can take
minutes. Do not wait until the presentation to download and rehearse.

1. Upload/record up to 30 seconds, or select a prepared five-second example.
2. Click **Identify sounds**. Suggestions come from a trained 527-label model.
3. Choose one sound to reduce, or type a short description of a sound you hear.
4. Click **Estimate selected source and simulate ANC**.
5. Compare before/after at unchanged volume. Inspect the extracted source if the
   result is poor. The app shows two compact graphs and exports a technical report.

The microphone control records a clip; it is not continuous listening or a live
headphone loop. Inference results are cached by audio, prompt and model revision.

## What actually runs

| Stage | Implementation | Limitation |
|---|---|---|
| Recognition | MIT AST, pretrained/fine-tuned on AudioSet; independent sigmoid scores | 527 labels, imperfect detection; not every sound |
| Selection | User selects a suggestion or supplies a text description | AI does not decide what the user considers unwanted |
| Source estimate | Pretrained AudioSep, text-conditioned neural separation; bounded target-band fallback if memory is insufficient | Leakage, missing sounds and artifacts are possible; the report names the method |
| Anti-noise | Calibrated FIR or adaptive FxLMS; negative speaker command through S(z) | Assumed stationary, known primary/secondary paths |
| Ear output | Original sound through P(z) plus anti-noise through S(z) | Offline simulation ignores neural processing delay |

There is no spectral-gate denoising or whole-mixture volume reduction in this
pipeline. Neural separation estimates a waveform; a control filter uses that
estimate. Separation itself modifies spectral content and is **not** physical ANC.
The offline resulting waveform alone does not demonstrate live acoustic cancellation.

The primary path has a 32-sample delay and FIR `[.8, .15, -.05]`; the secondary
path is `[0, 0, .7, .2, -.05]`, at 16 kHz. The calibrated controller has 64 taps.
This ideal broadband speaker is deliberately optimistic. Uploaded recordings
are treated as source-plane mixtures, although real uploads already contain
their own unknown room acoustics. No real speaker/headphone was calibrated.

Before/after and anti-noise WAVs use one shared export gain. Graphs use the raw
simulation amplitudes on common scales. Overall level reduction is **not** a
source-removal accuracy score. Tracking the estimated reference is also not proof
that the AI extracted the right sound. Clean stems are required for that check.

## Validation

```powershell
python -m pytest -q
python -m demo4.evaluate
```

The integration evaluation builds speech+engine and speech+keyboard recordings
from the repository's credited ESC-50/LibriSpeech sources, passes only the mixture
and text prompt to the AI, and evaluates against held-back component waveforms.
It reports residual improvement including leakage/artifacts. These are two
constructed examples, not a general benchmark or a claim the pretrained model
never saw these source clips. Results are written to `validation.json`.

## Route to the intended live product

The app is the preference interface. A suitable headset must provide reference
microphones, error microphones near the ears, controllable speakers, measured
acoustic paths and a low-latency DSP loop. The fast ANC loop operates sample by
sample; the slower AI updates source choices/control policy. The offline
separator cannot simply be put into that fast loop unchanged.

Start with measured low-frequency tonal cancellation on a controlled hardware
rig. Measure end-to-end latency, causality, stability and ear-position residual
before attempting overlapping sources or abrupt keyboard clicks. Live selective
ANC is a research goal, not a guaranteed later software switch. Playing music
can mask noise; that is a separate effect from destructive interference.

## Model credits

- [AudioSet](https://research.google.com/audioset/): Google's labeled sound dataset.
- [MIT AST](https://huggingface.co/MIT/ast-finetuned-audioset-10-10-0.4593), revision
  `f826b80d28226b62986cc218e5cec390b1096902` (model card: BSD-3-Clause).
- [AudioSep](https://github.com/Audio-AGI/AudioSep), MIT source; vendored license and
  provenance are in `vendor/audiosep/`. The
  [recommended checkpoint](https://huggingface.co/nielsr/audiosep-demo) is pinned
  to `9188caa202a0d31845367e4dad8bb5549eddb4ba` (model card: Apache-2.0).
- RoBERTa tokenizer at `FacebookAI/roberta-base`, revision
  `e2da8e2f811d1448a5b465c236feacd80ffbac7b`.

QuantaSound integrates pretrained models. It does **not** claim to have trained
these foundation models from scratch. Binary downloads are checksum verified;
PyTorch loads tensor weights with `weights_only=True`. No remote model code is run.
Qiskit/QAOA remains a separate offline optimization experiment in Demo 1/2.
