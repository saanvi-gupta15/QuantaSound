"""Reproducible small integration check with known stems; not a dataset benchmark.

python -m demo4.evaluate --prepare-only
python -m demo4.evaluate
"""
import argparse
import json
from pathlib import Path
import numpy as np
import soundfile as sf
from scipy.signal import lfilter
from demo1.data import demo_clips
from demo1.dsp import power_db_ratio
from .recognition import read_audio, analyze, LOCAL, SR
from .control import cancel, export, PRIMARY


def prepare():
    clips = dict(demo_clips())
    speech = read_audio(clips['speech (LibriSpeech, separate from ESC-50)'])[:5*SR]
    def level(x,rms): return x*rms/max(np.sqrt(np.mean(x*x)),1e-8)
    speech = level(speech,.07)
    directory = LOCAL/'examples'; directory.mkdir(parents=True,exist_ok=True)
    rows = []
    for category,query in [('engine','Engine'),('keyboard_typing','Computer keyboard')]:
        noise = level(read_audio(clips[category])[:len(speech)],.12)
        mix = speech+noise
        path = directory/f'speech_and_{category}.wav'
        sf.write(path,mix,SR,subtype='FLOAT')
        np.savez_compressed(directory/f'{category}_ground_truth.npz',wanted=speech,target=noise)
        rows.append((path,speech,noise,query))
    return rows


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--prepare-only',action='store_true')
    args = parser.parse_args()
    rows = prepare()
    if args.prepare_only:
        for path,*_ in rows: print(path)
        return
    from .separation import separate
    reports=[]
    for path,wanted,target,query in rows:
        mix = read_audio(path)
        recognition = analyze(mix)
        print(path.name,list(recognition['scores'].items())[:5],flush=True)
        reference,metadata = separate(mix,query)
        result = cancel(mix,reference)
        desired = lfilter(PRIMARY,[1],wanted)
        disturbance = lfilter(PRIMARY,[1],target)
        start = len(mix)//2
        # Residual includes missed target AND wanted-content errors/artifacts.
        improvement = power_db_ratio(disturbance[start:],(result[1]-desired)[start:])
        record={'example':path.name,'query':query,'top_predictions':list(recognition['scores'].items())[:5],
                'ground_truth_residual_improvement_db':improvement,
                'interpretation':'Final-half unwanted residual relative to true clean wanted stem; includes leakage/artifacts. Negative means worse.',
                'separation':metadata,'evaluation_limit':'Two constructed mixtures; not unseen AudioSep training data or a general accuracy benchmark.'}
        files = export(result,record)
        record['report_file']=str(Path(files['report']).relative_to(Path(__file__).parent))
        reports.append(record)
        print(json.dumps(record,indent=2),flush=True)
    (Path(__file__).parent/'validation.json').write_text(json.dumps(reports,indent=2)+'\n')


if __name__ == '__main__': main()
