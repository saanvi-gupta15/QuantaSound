import json
import numpy as np
import pytest
from demo2.profiles import save_profile, load_profile, list_profiles
from demo2.simulation import simulate, export_result
from demo1.quantum import problem, statevector, qiskit_statevector


@pytest.fixture
def sources():
    t = np.arange(32000)/16000
    return {"engine":0.2*np.sin(2*np.pi*200*t),
            "keyboard":0.03*np.sin(2*np.pi*3500*t),
            "speech":0.1*np.sin(2*np.pi*700*t)}


def test_no_selection_is_identity(sources):
    result = simulate(sources, [])
    np.testing.assert_array_equal(result.before, result.after)
    assert result.metrics["total_noise_reduction_db"] == 0
    assert result.metrics["selected_residual_reduction_db"] is None
    assert result.metrics["speech_counterfactual_distortion_rms_percent"] == 0


def test_selection_cancels_reference_not_direct_subtraction(sources):
    result = simulate(sources, ["engine"])
    np.testing.assert_allclose(result.after, result.before+result.anti_noise_at_ear, atol=1e-14)
    assert result.metrics["selected_residual_reduction_db"] > 10
    # Secondary-path delay prevents an instantaneous cancelling output. The
    # independent ear floor can then drive tiny updates before the primary arrives.
    np.testing.assert_allclose(result.anti_noise_at_ear[:3], 0, atol=1e-14)
    assert not np.array_equal(result.after, result.desired)


def test_profile_persists_and_rejects_invalid_input(tmp_path):
    save_profile("student", ["keyboard"], tmp_path)
    assert load_profile("student", tmp_path)["reduce"] == ["keyboard"]
    assert list_profiles(tmp_path) == ["student"]
    save_profile("student", [], tmp_path)
    assert load_profile("student", tmp_path)["reduce"] == []
    with pytest.raises(ValueError):
        save_profile("../escape", ["engine"], tmp_path)
    with pytest.raises(ValueError):
        save_profile("student", ["speech"], tmp_path)
    (tmp_path/"broken.json").write_text("{}")
    with pytest.raises(ValueError):
        load_profile("broken", tmp_path)


def test_export_keeps_one_gain_for_both_audio_files(sources, tmp_path):
    import soundfile as sf
    result = simulate(sources, ["engine"])
    before, after, metrics = export_result(result, tmp_path)
    gain = json.loads(metrics.read_text())["shared_playback_gain"]
    np.testing.assert_allclose(sf.read(before)[0], result.before*gain, atol=1/32768)
    np.testing.assert_allclose(sf.read(after)[0], result.after*gain, atol=1/32768)


def test_qiskit_matches_numpy_probabilities():
    _, costs = problem()
    for angles in [[0,0], [.3,.7], [1.2,2.3]]:
        np.testing.assert_allclose(np.abs(qiskit_statevector(angles,costs))**2,
                                   np.abs(statevector(angles,costs))**2, atol=1e-12)


def test_calibrated_broadband_on_unseen_transients(sources):
    from scipy.signal import lfilter
    from demo2.simulation import calibrated_fir
    rng = np.random.default_rng(451)
    sources['keyboard'] = .08*rng.normal(size=32000)
    result = simulate(sources, ['keyboard'], controller='calibrated_broadband')
    p, s = result.metrics['primary_fir'], result.metrics['secondary_fir']
    w = calibrated_fir(p, s)
    np.testing.assert_allclose(result.anti_noise_at_ear,
                               -lfilter(s, [1], lfilter(w, [1], sources['keyboard'])), atol=1e-14)
    assert result.metrics['selected_residual_reduction_db'] > 25
    assert result.metrics['speech_counterfactual_distortion_rms_percent'] == 0
    lowpass = simulate(sources, ['keyboard'], controller='calibrated_lowpass')
    assert lowpass.metrics['selected_residual_reduction_db'] < result.metrics['selected_residual_reduction_db']


def test_python_widget_save_load_and_run_callbacks(tmp_path, sources, monkeypatch):
    import demo2.ui as ui
    seen = []
    monkeypatch.setattr(ui, "load_sources", lambda **kwargs: (sources, {}))
    monkeypatch.setattr(ui, "show_result", lambda result, directory: seen.append(result))
    _, controls = ui.build_controls(tmp_path, tmp_path/"audio")
    controls["engine"].value = False
    controls["keyboard"].value = True
    controls["save"].click()
    # A newly constructed interface restores default from disk.
    _, restored = ui.build_controls(tmp_path, tmp_path/"audio")
    assert restored["keyboard"].value and not restored["engine"].value
    restored["run"].click()
    assert seen[0].metrics["selected_for_reduction"] == ["keyboard"]
    assert not restored["run"].disabled


def test_audible_scene_is_explicit_and_does_not_replace_recorded_source(tmp_path, monkeypatch):
    import soundfile as sf
    import demo1.data
    from demo2.simulation import load_sources
    t = np.arange(16000)/16000
    path = tmp_path/"original.wav"
    sf.write(path, .2*np.sin(2*np.pi*40*t), 16000)
    labels = ["engine", "keyboard_typing", "speech (LibriSpeech, separate from ESC-50)"]
    monkeypatch.setattr(demo1.data, "demo_clips", lambda: [(name,path) for name in labels])
    original, _ = load_sources(seconds=1)
    audible, info = load_sources(seconds=1, scene="audible_hum")
    assert np.argmax(np.abs(np.fft.rfft(original["engine"]))) == 40
    assert np.argmax(np.abs(np.fft.rfft(audible["engine"]))) == 200
    assert info["engine"]["file"] is None
    assert "Synthetic" in info["engine"]["generator"]
    assert np.sqrt(np.mean(audible["keyboard"]**2)) == pytest.approx(.025)
