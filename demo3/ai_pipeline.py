"""Generic uploaded-audio pipeline.

This module can denoise arbitrary mono audio with a conservative spectral gate.
It does not claim physical ANC: no anti-noise is generated unless a correlated
reference and calibrated secondary path are supplied to the separate FxLMS demo.
"""
from dataclasses import dataclass
from pathlib import Path
import json
import numpy as np
import soundfile as sf
from scipy import signal


@dataclass
class Analysis:
    input_path: str
    output_path: str
    sample_rate_hz: int
    duration_s: float
    estimated_noise_floor_db: float
    estimated_noise_reduction_db: float
    processing_mode: str
    limitations: str


def _to_mono(x):
    x = np.asarray(x, dtype=float)
    return x.mean(axis=1) if x.ndim == 2 else x


def classify_audio(x, sr):
    """Lightweight explainable classifier for arbitrary uploads.

    This is a triage label, not a source separator. It uses spectral centroid,
    zero crossing rate and low-frequency energy to decide what the denoiser
    should explain to the presenter.
    """
    x = _to_mono(x)
    f, p = signal.welch(x, sr, nperseg=min(2048, len(x)))
    total = np.sum(p) + 1e-20
    centroid = float(np.sum(f*p)/total)
    low_ratio = float(np.sum(p[f < 300])/total)
    zcr = float(np.mean(np.abs(np.diff(np.signbit(x)))))
    if low_ratio > .55:
        label, reason = "low-frequency engine/fan-like", "most spectral power is below 300 Hz"
    elif centroid > 2500 and zcr > .12:
        label, reason = "transient/high-frequency keyboard-like", "high centroid and rapid sign changes"
    elif centroid < 1200:
        label, reason = "speech/music-like", "energy is concentrated in lower harmonics"
    else:
        label, reason = "mixed/unknown", "no single heuristic dominates"
    return {"label":label, "reason":reason, "spectral_centroid_hz":centroid,
            "low_frequency_power_ratio":low_ratio, "zero_crossing_rate":zcr,
            "warning":"Classification describes the mixture; it does not recover a separate source waveform."}


def spectral_gate(x, sr, noise_seconds=0.25, reduction_db=12.0):
    """Suppress stationary spectral components estimated from the first noise window."""
    x = _to_mono(x)
    n_fft, hop = 2048, 512
    _, _, z = signal.stft(x, sr, nperseg=n_fft, noverlap=n_fft-hop, boundary="zeros")
    mag, phase = np.abs(z), np.angle(z)
    noise_frames = max(1, int(noise_seconds*sr/hop))
    noise = np.median(mag[:, :noise_frames], axis=1, keepdims=True)
    # Soft mask: preserve components above estimated floor and attenuate below.
    threshold = noise * 1.8
    ratio = np.clip((mag-threshold)/(mag+1e-12), 0, 1)
    floor = 10**(-reduction_db/20)
    mask = floor + (1-floor)*ratio
    _, y = signal.istft(mask*mag*np.exp(1j*phase), sr, nperseg=n_fft,
                        noverlap=n_fft-hop, input_onesided=True, boundary=True)
    return y[:len(x)], {"noise_seconds":noise_seconds, "attenuation_floor_db":reduction_db,
                        "noise_floor_db":float(20*np.log10(np.median(noise)+1e-12)),
                        "frames":int(mag.shape[1])}


def process_file(input_path, output_path=None, noise_seconds=0.25, reduction_db=12.0):
    input_path = Path(input_path)
    x, sr = sf.read(input_path)
    x = _to_mono(x)
    if len(x) < max(1024, int(noise_seconds*sr)):
        raise ValueError("Audio is too short for a noise-profile estimate")
    y, gate = spectral_gate(x, sr, noise_seconds, reduction_db)
    if output_path is None:
        output_path = input_path.with_name(input_path.stem+"_denoised.wav")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    gain = .95/max(np.max(np.abs(y)), 1e-12)
    sf.write(output_path, gain*y, sr, subtype="PCM_16")
    before_tail, after_tail = x[len(x)//2:], y[len(y)//2:]
    before_power, after_power = np.mean(before_tail**2), np.mean(after_tail**2)
    result = {"input":str(input_path), "output":str(output_path), "classification":classify_audio(x,sr),
              "noise_profile":gate, "sample_rate_hz":int(sr), "duration_s":len(x)/sr,
              "estimated_level_change_db":float(10*np.log10((before_power+1e-20)/(after_power+1e-20))),
              "processing_mode":"AI-assisted spectral denoising (not physical ANC)",
              "anti_noise_generated":False,
              "limitations":"Arbitrary uploads are denoised by attenuation in the time-frequency domain. "
                           "A correlated reference microphone and secondary-path model are required for physical ANC anti-noise.",
              "shared_gain_for_output":float(gain)}
    output_path.with_suffix(".json").write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")
    return result
