import json
import numpy as np
import soundfile as sf
from demo3.ai_pipeline import classify_audio, process_file, spectral_gate


def test_spectral_gate_reduces_stationary_tone_without_changing_length():
    sr = 16000
    t = np.arange(sr*2)/sr
    rng = np.random.default_rng(4)
    x = .1*np.sin(2*np.pi*200*t) + .015*rng.normal(size=len(t))
    y, metadata = spectral_gate(x, sr, noise_seconds=.25, reduction_db=12)
    assert len(y) == len(x)
    assert np.all(np.isfinite(y))
    assert metadata["frames"] > 1
    assert np.mean(y[int(.5*sr):]**2) < np.mean(x[int(.5*sr):]**2)


def test_unknown_upload_returns_audio_and_explainable_report(tmp_path):
    sr = 16000
    t = np.arange(sr)/sr
    source = tmp_path/"unknown.wav"
    sf.write(source, .1*np.sin(2*np.pi*200*t), sr)
    destination = tmp_path/"clean.wav"
    result = process_file(source, destination)
    assert destination.exists()
    report = json.loads(destination.with_suffix(".json").read_text())
    assert report["anti_noise_generated"] is False
    assert "reference" in report["limitations"]
    assert result["sample_rate_hz"] == sr


def test_classifier_explains_mixture_features():
    sr = 16000
    t = np.arange(sr)/sr
    result = classify_audio(.1*np.sin(2*np.pi*200*t), sr)
    assert result["label"] == "low-frequency engine/fan-like"
    assert result["low_frequency_power_ratio"] > .5
