import json
import numpy as np
import pytest
import soundfile as sf
from scipy.signal import lfilter
from demo4.control import cancel, export, PRIMARY, SECONDARY
from demo4.recognition import read_audio, SR


def signals():
    t = np.arange(2*SR)/SR
    target = .12*np.sin(2*np.pi*200*t)
    wanted = .08*np.sin(2*np.pi*1100*t)
    return target+wanted,target,wanted


def test_antinoise_passes_through_secondary_and_preserves_unselected_tone():
    mix,target,wanted = signals()
    before,after,command,anti,*rest = cancel(mix,target,estimated=False)
    np.testing.assert_allclose(anti,lfilter(SECONDARY,[1],command))
    np.testing.assert_allclose(after,before+anti)
    desired = lfilter(PRIMARY,[1],wanted)
    assert np.mean((after[SR:]-desired[SR:])**2) < .001*np.mean(target**2)


def test_zero_strength_is_identity_and_leakage_is_not_hidden():
    mix,target,_ = signals()
    zero = cancel(mix,target,0)
    np.testing.assert_array_equal(zero[0],zero[1])
    bad = cancel(mix,mix)  # If the separator leaks everything, the controller cancels everything.
    assert np.mean(bad[1]**2) < .01*np.mean(bad[0]**2)
    assert 'leaked' in bad[-1]['wanted_content_note']
    assert 'target_residual_reduction_db' not in bad[-1]


def test_equal_gain_export(tmp_path,monkeypatch):
    import demo4.control as module
    monkeypatch.setattr(module,'LOCAL',tmp_path)
    mix,target,_ = signals()
    result = cancel(mix,target)
    files = export(result)
    gain = json.loads(open(files['report']).read())['shared_playback_gain']
    for key,x in zip(('before','after','speaker_command'),result[:3]):
        np.testing.assert_allclose(sf.read(files[key])[0],gain*x,atol=1/32768)


@pytest.mark.parametrize('count',[3200,96000,96001,200000])
def test_separator_chunks_keep_short_clips_and_last_samples(count):
    torch = pytest.importorskip('torch')
    from demo4.separation import chunked_inference
    x = np.random.default_rng(19).normal(size=count).astype('float32')
    class Identity:
        def __call__(self,values): return {'waveform':values['mixture']}
    actual = chunked_inference(x,torch.zeros(1,512),Identity())
    np.testing.assert_array_equal(actual,x)


def test_rejects_oversize_audio_before_inference(tmp_path):
    path = tmp_path/'long.wav'
    sf.write(path,np.zeros(31*8000),8000)
    with pytest.raises(ValueError,match='30 seconds'): read_audio(path)


def test_wrong_shape_or_silent_reference_is_not_cancelled():
    mix,target,_ = signals()
    with pytest.raises(ValueError): cancel(mix,target[:10])
    with pytest.raises(ValueError): cancel(mix,np.zeros_like(target))
