"""Source-aware FxLMS with explicit oracle references and reproducible metrics."""
from dataclasses import dataclass
from pathlib import Path
import json
import numpy as np
from scipy import signal, linalg
import soundfile as sf
from demo1.dsp import fxlms, power_db_ratio
from .profiles import validate_selection

SR = 16000


def load_sources(seconds=10, scene="recorded"):
    """Repeat five-second environmental clips; speech uses its first ten seconds."""
    if scene not in ("recorded", "audible_hum"):
        raise ValueError("scene must be recorded or audible_hum")
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
        if scene == "audible_hum" and name == "engine":
            t = np.arange(n)/SR
            x = np.sin(2*np.pi*200*t)+0.3*np.sin(2*np.pi*400*t)+0.1*np.sin(2*np.pi*600*t)
            rms = 0.16
        if scene == "audible_hum" and name == "keyboard":
            rms = 0.025
        x -= np.mean(x)
        gain = rms / max(np.sqrt(np.mean(x*x)), 1e-12)
        sources[name] = x * gain
        provenance[name] = {"file":paths[label].name, "repeated_to_length":repeated,
                            "rms_after_scaling":rms, "input_gain":float(gain)}
        if scene == "audible_hum" and name == "engine":
            provenance[name].update(file=None, repeated_to_length=False,
                                    generator="Synthetic 200 Hz hum plus 400/600 Hz harmonics; not a real engine recording")
        provenance[name]["scene"] = scene
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


def calibrated_fir(primary, secondary, length=64, regularization=0.001):
    """Fit S*W to P using paths only, never evaluation audio; penalize effort."""
    matrix = linalg.convolution_matrix(secondary, length)
    if len(primary) > len(matrix) or regularization <= 0:
        raise ValueError("Invalid calibration length or regularization")
    target = np.pad(primary, (0, len(matrix)-len(primary)))
    return np.linalg.lstsq(np.vstack([matrix, np.sqrt(regularization)*np.eye(length)]),
                           np.r_[target, np.zeros(length)], rcond=None)[0]


def simulate(sources, reduce=("engine",), sr=SR, controller="adaptive_lowpass"):
    """Adaptive baseline or explicitly calibrated fixed-controller comparison.

    Reference = sum of separately known selected tracks; NOT separation from
    the mixture. One scalar controller, shared source primary path, known FIR
    secondary path, ideal preview. The adaptive baseline uses full ear error;
    calibration uses paths only and freezes the controller. No direct source
    subtraction at the output.
    """
    if controller not in ("adaptive_lowpass", "calibrated_lowpass", "calibrated_broadband"):
        raise ValueError("Unknown controller scenario")
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
    if controller == "calibrated_broadband":
        s = np.array([0., 0., 0.7, 0.2, -0.05])
    ear = {key:signal.lfilter(p, [1], value) for key, value in arrays.items()}
    floor = np.random.default_rng(107).normal(0, 0.001, n)
    selected = sum((arrays[key] for key in selection), np.zeros(n))
    target = sum((ear[key] for key in selection), np.zeros(n))
    wanted = sum((ear[key] for key in arrays if key not in selection), np.zeros(n))
    before = sum(ear.values(), np.zeros(n)) + floor
    control = np.zeros(n)
    if selection and controller != "adaptive_lowpass":
        # Frozen offline path calibration: no speech is used to update weights.
        # The known noise reference drives an actual control FIR and secondary FIR.
        weights = calibrated_fir(p, s)
        control = signal.lfilter(weights, [1], selected)
        anti_noise = -signal.lfilter(s, [1], control)
        after = before+anti_noise
        speech_delta = ear["speech"].copy()  # fixed reference-only controller: by construction
    elif selection:
        result = fxlms(selected, p, s, length=64, mu=0.004, additive_disturbance=wanted+floor)
        after = result.error
        control = result.control
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
        "controller_scenario":controller,
        "controller_description":("Online FxLMS, low-pass secondary path" if controller == "adaptive_lowpass"
            else "Offline calibrated fixed FIR; " + ("same low-pass path" if controller == "calibrated_lowpass"
                 else "DIFFERENT ideal broadband secondary path; not original hardware")),
        "speech_metric_interpretation":("Paired adaptive reruns" if controller == "adaptive_lowpass"
            else "Zero speech adaptation error is by construction: frozen controller and exact isolated noise reference"),
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
        "mu":0.004 if controller == "adaptive_lowpass" else None,
        "calibration_regularization":None if controller == "adaptive_lowpass" else 0.001,
        "control_rms":float(np.sqrt(np.mean(control**2))), "control_peak":float(np.max(np.abs(control))),
        "controller_taps":64, "primary_fir":p.tolist(), "secondary_fir":s.tolist(),
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
