"""Reference-driven anti-noise generation. No masking or spectral denoising."""
from pathlib import Path
import json
import uuid
import numpy as np
import soundfile as sf
from scipy import signal
from demo1.dsp import fxlms, power_db_ratio
from demo2.simulation import calibrated_fir
from .recognition import SR, LOCAL

PRIMARY = np.r_[np.zeros(32), .8, .15, -.05]
SECONDARY = np.array([0., 0., .7, .2, -.05])


def cancel(source_mixture, target_reference, strength=1.0, method='Calibrated FIR', estimated=True):
    """Source-plane mixture/reference -> simulated acoustic paths -> ear pressure.

    Both inputs are assumed synchronized BEFORE propagation through P. This does
    not reproduce headphone fit or real microphone latency. An arbitrary file
    of the same sound type is not a valid coherent cancellation reference.
    """
    mix = np.asarray(source_mixture, dtype=float)
    reference = np.asarray(target_reference, dtype=float)
    if mix.ndim != 1 or reference.shape != mix.shape or len(mix) < SR//10:
        raise ValueError('Mixture and reference must be synchronized, equal-length mono arrays.')
    if not np.all(np.isfinite(mix)) or not np.all(np.isfinite(reference)):
        raise ValueError('Audio contains non-finite values.')
    if not 0 <= strength <= 1:
        raise ValueError('Reduction strength must be between zero and one.')
    if method not in ('Calibrated FIR','Adaptive FxLMS'):
        raise ValueError('Unknown controller.')
    if np.mean(reference**2) < 1e-12:
        raise ValueError('Selected reference is silent; choose an audible source.')
    p, s = PRIMARY, SECONDARY
    before = signal.lfilter(p,[1],mix)
    target = signal.lfilter(p,[1],reference)
    wanted = before-target
    if method == 'Calibrated FIR':
        weights = calibrated_fir(p,s)
        command = -strength*signal.lfilter(weights,[1],reference)
        # Fixed control path uses only the selected reference, so wanted signal
        # preservation here follows by construction from the oracle references.
        speech_note = 'Any wanted sound leaked into the estimated reference can also be cancelled.'
    else:
        result = fxlms(reference,p,s,length=64,mu=.001,additive_disturbance=wanted)
        command = -strength*result.control
        speech_note = 'Online adaptation sees full error, so wanted content may affect the controller.'
    anti_at_ear = signal.lfilter(s,[1],command)
    after = before+anti_at_ear
    if not np.all(np.isfinite(after)) or not np.all(np.isfinite(command)):
        raise ValueError('Controller became unstable; no output was exported. Try the calibrated FIR.')
    start = len(mix)//2
    residual = after-wanted
    metrics = {'method':method,'strength':strength,'reference_required':True,
        'processing':'Control FIR -> speaker command -> secondary path -> acoustic sum at simulated ear',
        'spectral_gate_used':False,'neural_separation_used':estimated,
        'measurement_window_s':[start/SR,len(mix)/SR],
        'estimated_reference_tracking_db':power_db_ratio(target[start:],residual[start:]),
        'measurement_warning':'Tracking the estimated reference is not proof of true target removal or preservation of speech. Ground-truth stems are needed for that evaluation.',
        'mixture_level_change_db':power_db_ratio(before[start:],after[start:]),
        'control_rms':float(np.sqrt(np.mean(command**2))), 'control_peak':float(np.max(np.abs(command))),
        'secondary_fir':s.tolist(),'primary_fir':p.tolist(),'sample_rate_hz':SR,
        'simulation_assumptions':'Offline aligned reference; ideal broadband stationary speaker, known P/S, no actuator clipping. Neural processing latency is NOT included. Not live headphone ANC.',
        'wanted_content_note':speech_note}
    return before, after, command, anti_at_ear, target, wanted, metrics


def export(result, metadata=None):
    before,after,command,anti,target,wanted,metrics = result
    directory = LOCAL/'runs'/uuid.uuid4().hex
    directory.mkdir(parents=True)
    gain = .9/max(np.max(np.abs(before)), np.max(np.abs(after)), np.max(np.abs(command)),1e-12)
    files = {}
    for name,x in [('before',before),('after',after),('speaker_command',command)]:
        path = directory/f'{name}.wav'
        sf.write(path,gain*x,SR,subtype='PCM_16'); files[name] = str(path)
    pair = np.r_[before[len(before)//2:],np.zeros(SR//2),after[len(after)//2:]]
    path = directory/'comparison.wav'; sf.write(path,gain*pair,SR,subtype='PCM_16')
    files['comparison'] = str(path)
    report = {**metrics,'shared_playback_gain':gain,**(metadata or {})}
    path = directory/'report.json'; path.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    files['report'] = str(path)
    return files
