"""Source-aware FxLMS with explicit oracle references and reproducible metrics."""
from dataclasses import dataclass
from pathlib import Path
import json
import numpy as np
from scipy import signal
import soundfile as sf
from demo1.dsp import fxlms, power_db_ratio
from .profiles import validate_selection

SR = 16000


def load_sources(seconds=10):
    """Repeat five-second environmental clips; speech uses its first ten seconds."""
    import librosa
    from demo1.data import demo_clips
    paths = dict(demo_clips())
    sources, provenance = {}, {}
    for name, label, rms in [("engine", "engine", 0.12), ("keyboard", "keyboard_typing", 0.10),
                            ("speech", "speech (LibriSpeech, separate from ESC-50)", 0.12)]:
        x, _ = librosa.load(paths[label], sr=SR, mono=True)
        n = int(seconds * SR)
        repeated = len(x) < n
        x = np.tile(x, int(np.ceil(n/len(x))))[:n].astype(float)
        x -= np.mean(x)
        gain = rms / max(np.sqrt(np.mean(x*x)), 1e-12)
        sources[name] = x * gain
        provenance[name] = {"file":paths[label].name, "repeated_to_length":repeated,
                            "rms_after_scaling":rms, "input_gain":float(gain)}
    return sources, provenance


@dataclass
class PersonalizedResult:
    before: np.ndarray
    after: np.ndarray
    speech: np.ndarray
    desired: np.ndarray
    selected_at_ear: np.ndarray
    anti_noise_at_ear: np.ndarray
    speech_counterfactual_delta: np.ndarray
    metrics: dict
    sr: int


def simulate(sources, reduce=("engine",), sr=SR):
    """Controller adapts on full ear error, including wanted speech.

    Reference = sum of separately known selected tracks; NOT separation from
    the mixture. One scalar controller, shared source primary path, known FIR
    secondary path, ideal preview. No direct target subtraction at the output.
    """
    selection = validate_selection(reduce)
    if set(sources) != {"engine", "keyboard", "speech"}:
        raise ValueError("Provide engine, keyboard and speech source arrays")
    arrays = {key:np.asarray(value, dtype=float) for key, value in sources.items()}
    n = len(arrays["speech"])
    if sr != SR or n < sr:
        raise ValueError("Use 16 kHz sources lasting at least one second")
    if any(x.ndim != 1 or len(x) != n or not np.all(np.isfinite(x)) for x in arrays.values()):
        raise ValueError("Sources must be finite equal-length mono arrays")
    p = np.r_[np.zeros(32), 0.8, 0.15, -0.05]
    # Low-pass speaker model limits high-frequency click cancellation.
    s = np.r_[np.zeros(2), signal.firwin(17, 1500, fs=sr)]
    ear = {key:signal.lfilter(p, [1], value) for key, value in arrays.items()}
    floor = np.random.default_rng(107).normal(0, 0.001, n)
    selected = sum((arrays[key] for key in selection), np.zeros(n))
    target = sum((ear[key] for key in selection), np.zeros(n))
    wanted = sum((ear[key] for key in arrays if key not in selection), np.zeros(n))
    before = sum(ear.values(), np.zeros(n)) + floor
    if selection:
        result = fxlms(selected, p, s, length=64, mu=0.004, additive_disturbance=wanted+floor)
        after = result.error
        anti_noise = -result.secondary_output
        # Re-run the SAME controller experiment without speech. Difference
        # includes speech-driven adaptation changes, not just an untouched stem.
        no_speech = fxlms(selected, p, s, length=64, mu=0.004,
                         additive_disturbance=wanted-ear["speech"]+floor)
        speech_delta = after-no_speech.error
    else:
        after = before.copy()
        anti_noise = np.zeros(n)
        speech_delta = ear["speech"].copy()
    tail = slice(n//2, None)
    residual_before = before-ear["speech"]
    residual_after = after-ear["speech"]
    speech_error = speech_delta-ear["speech"]
    speech_energy = float(np.mean(ear["speech"][tail]**2))
    speech_error_energy = float(np.mean(speech_error[tail]**2))
    metrics = {
        "selected_for_reduction":selection, "preserve_speech":True,
        "reference_mode":"oracle: separately known source tracks, no source separation",
        "sample_rate_hz":sr, "seconds":n/sr, "measurement_window_s":[n/(2*sr), n/sr],
        "total_noise_reduction_db":power_db_ratio(residual_before[tail], residual_after[tail]),
        "selected_residual_reduction_db":(
            power_db_ratio((target+floor)[tail], (after-wanted)[tail]) if selection else None),
        "speech_reference_sdr_before_db":power_db_ratio(ear["speech"][tail], residual_before[tail]),
        "speech_reference_sdr_after_db":power_db_ratio(ear["speech"][tail], residual_after[tail]),
        "speech_counterfactual_distortion_rms_percent":float(100*np.sqrt(speech_error_energy/max(speech_energy, 1e-20))),
        "speech_counterfactual_gain":float(np.dot(speech_delta[tail], ear["speech"][tail]) /
                                           max(np.dot(ear["speech"][tail], ear["speech"][tail]), 1e-20)),
        "whole_run_noise_reduction_db":power_db_ratio(residual_before, residual_after),
        "mu":0.004, "controller_taps":64, "primary_fir":p.tolist(), "secondary_fir":s.tolist(),
        "uncorrelated_ear_disturbance_std":0.001, "seed":107,
        "limits":"Simulation only; exact stationary paths, early noise references, no actuator clipping. "
                 "Speech metrics are reference-based proxies, not intelligibility scores."
    }
    return PersonalizedResult(before, after, ear["speech"], wanted, target, anti_noise, speech_delta, metrics, sr)


def export_result(result, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    # One gain applied to both listening files; headroom prevents clipping.
    gain = 0.9/max(np.max(np.abs(result.before)), np.max(np.abs(result.after)), 1e-12)
    for name, x in [("before", result.before), ("after", result.after)]:
        sf.write(directory/f"{name}.wav", gain*x, result.sr, subtype="PCM_16")
    metrics = {**result.metrics, "shared_playback_gain":float(gain), "audio_format":"PCM_16"}
    (directory/"metrics.json").write_text(json.dumps(metrics, indent=2)+"\n", encoding="utf-8")
    return directory/"before.wav", directory/"after.wav", directory/"metrics.json"
