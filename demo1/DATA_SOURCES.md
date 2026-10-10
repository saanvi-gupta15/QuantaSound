# Data sources and attribution

## ESC-50

- Author: Karol J. Piczak. *ESC: Dataset for Environmental Sound Classification*, ACM Multimedia 2015.
- Upstream: https://github.com/karolpiczak/ESC-50
- Pinned revision: `33c8ce9eb2cf0b1c2f8bcf322eb349b6be34dbb6`.
- Dataset license: Creative Commons Attribution-NonCommercial 3.0. Individual recording credits/licenses are retained in `data/ESC50_LICENSE.txt`; do not assume every source clip has the same license.
- Demo A takes the first metadata entry for engine, siren and keyboard typing. The classifier uses all 40 recordings from each of engine, siren, keyboard typing, rain and dog (200 total).
- `data/demo_a_manifest.json` and `data/classifier_manifest.json` identify filenames, original source recording IDs, pinned download URLs and SHA-256 hashes. Metadata includes source folds; all variants from a recording stay on the same side of the split.
- Raw audio is downloaded on demand and excluded from Git. Generated plots are derived analyses; attribution remains here and in the notebook.

## Speech (separate from ESC-50)

Demo A uses librosa's `libri1` example from LibriSpeech: *The Ashiel Mystery: A Detective Story*, by Marion Bryce (Mrs. Charles Bryce), narrated by Garth Comira. The downloaded example metadata identifies chapter XVI and the CC BY 4.0 license; the catalog description differs, so the retained per-file metadata is authoritative here. LibriSpeech's source recordings are from LibriVox. The downloaded attribution/license text is available via `librosa.util.example_info('libri1')`; a copy is saved as `data/SPEECH_LICENSE.txt` in this review package. `data/speech_manifest.json` records its source and SHA-256 hash.

- Example catalog: https://librosa.org/doc/0.11.0/recordings.html
- Data repository: https://github.com/librosa/data
- LibriSpeech: https://www.openslr.org/12/

Only the first five seconds are analyzed. No speech samples are used to train the ESC-50 classifier. The speech download is cached under `data/speech/` and excluded from Git.

## Synthetic signals

Demo B uses generated sinusoids and seeded Gaussian noise. Demo C uses a generated 200 Hz tone or seeded low-pass broadband noise. Exported ANC WAVs are synthetic; both members of a before/after pair have identical gain. They are demonstrations of a simulated error-microphone signal, not recorded hardware results.
