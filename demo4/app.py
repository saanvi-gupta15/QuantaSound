"""Python dashboard: AudioSet recognition -> AudioSep -> simulated ANC."""
import argparse
import gc
import hashlib
import time
import gradio as gr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from .recognition import SR, LOCAL, analyze, candidates, read_audio, load_model
from .control import cancel, export


def fingerprint(x):
    return hashlib.sha256(x.tobytes()).hexdigest()


def classify(path, progress=gr.Progress()):
    if not path:
        return None,gr.Dropdown(choices=[],value=None),None,'Upload or record a short clip first.'
    progress(.1, desc='AudioSet-trained model is identifying candidate sounds…')
    x = read_audio(path)
    result = analyze(x)
    suggested = candidates(result)
    options = [f'{label}  ({score:.0%})' for label, score in suggested]
    labels_by_display = {f'{label}  ({score:.0%})': label for label, score in suggested}
    top = list(result['scores'].items())[:8]
    fig, ax = plt.subplots(figsize=(8,3.2), layout='constrained')
    ax.barh([v[0] for v in top][::-1], [v[1] for v in top][::-1], color='#168b87')
    ax.set(xlim=(0,1), xlabel='Model score · not calibrated confidence', title='Likely sounds in this recording')
    directory = LOCAL/'charts'; directory.mkdir(parents=True,exist_ok=True)
    chart = directory/f'{fingerprint(x)}.png'
    fig.savefig(chart,dpi=130); plt.close(fig)
    state = {'fingerprint':fingerprint(x),'recognition':result,'labels_by_display':labels_by_display}
    note = ('Choose one sound to reduce. Other sounds may overlap with it.' if options else
            'No candidate exceeded 0.10. The model may not recognize this clip; enter a target description only if you hear it.')
    load_model.cache_clear(); gc.collect()
    return state, gr.CheckboxGroup(choices=options,value=None,label='Detected sounds · select ONE target'), str(chart), note


def comparison_plot(result, directory):
    before, after = result[:2]
    fig, axes = plt.subplots(2,1,figsize=(9,5.4),layout='constrained')
    # Frame RMS makes the level trend readable without a dense waveform overlay.
    step = SR//10
    for audio,label,color in [(before,'Before','#69788d'),(after,'Simulated ear output','#168b87')]:
        starts = np.arange(0,len(audio),step)
        rms = [np.sqrt(np.mean(audio[i:i+step]**2)) for i in starts]
        axes[0].plot(starts/SR,20*np.log10(np.maximum(rms,1e-8)),label=label,color=color)
        f,p = signal.welch(audio,SR,nperseg=min(2048,len(audio)))
        axes[1].plot(f,10*np.log10(p+1e-20),label=label,color=color,lw=1)
    axes[0].set(xlabel='Time (s)',ylabel='RMS level (dB)',title='Overall level · lower does not prove selective cancellation')
    axes[1].set(xlabel='Frequency (Hz)',ylabel='PSD (dB/Hz)',xlim=(0,8000),title='Spectrum · identical amplitude scale')
    for ax in axes: ax.legend(loc='best'); ax.grid(alpha=.15)
    destination = directory/'comparison.png'
    fig.savefig(destination,dpi=130); plt.close(fig)
    return str(destination)


def process(path, state, target, strength, method, progress=gr.Progress()):
    from pathlib import Path
    from .separation import separate
    if not path or not state:
        raise gr.Error('Identify sounds in the current recording first.')
    x = read_audio(path)
    if fingerprint(x) != state['fingerprint']:
        raise gr.Error('The recording changed. Click Identify sounds again.')
    if not target or len(target) != 1:
        raise gr.Error('Choose exactly ONE detected sound. The controller must know which source to estimate.')
    selected_display = target[0]
    target = state.get('labels_by_display',{}).get(selected_display, selected_display)
    start = time.perf_counter()
    progress(.15,desc='Estimating the selected source (AudioSep when memory allows)…')
    reference, separation_info = separate(x,target.strip())
    progress(.8,desc='Generating anti-noise through the simulated speaker path…')
    result = cancel(x,reference,strength,method)
    metadata = {'target_description':target,'recognition':state['recognition'],
                'separation':separation_info,'elapsed_s':time.perf_counter()-start,
                'live_anc':False}
    files = export(result,metadata)
    directory = Path(files['report']).parent
    graph = comparison_plot(result,directory)
    # Reference preview uses the same gain as before/after, never peak normalized independently.
    import json
    import soundfile as sf
    gain = json.loads(Path(files['report']).read_text())['shared_playback_gain']
    preview = directory/'estimated_source.wav'
    sf.write(preview,reference*gain,SR,subtype='PCM_16')
    change = result[-1]['mixture_level_change_db']
    note = (f'**Target:** {target} · **Strength:** {strength:.0%} · '
            f'**Overall level decrease:** {change:.2f} dB (final half)\n\n'
            f'Processing: {metadata["elapsed_s"]:.1f} s for {len(x)/SR:.1f} s of audio. '
            'Listen to the estimated source: if it contains speech or another wanted sound, that content can also be cancelled. '
            '**True target reduction is unknown for an upload without clean ground truth.** '
            'These players share one playback gain. This is an offline acoustic simulation, not room cancellation.')
    return files['before'], files['after'], str(preview), files['comparison'], graph, note, [files['speaker_command'],files['report']]


def build():
    with gr.Blocks(title='QuantaSound · Select what you hear') as app:
        gr.Markdown('''# QuantaSound
### Select what you hear
**Review 1 · trained sound recognition + source separation + simulated ANC**

Record or upload **up to 30 seconds**. Identify sounds, choose one to reduce, then compare.
The microphone records a clip; this version does not run live headphone cancellation.''')
        state = gr.State(None)
        with gr.Row():
            with gr.Column():
                audio = gr.Audio(sources=['upload','microphone'],type='filepath',label='1 · Background recording',format='wav')
                examples = sorted((LOCAL/'examples').glob('speech_and_*.wav'))
                if examples:
                    gr.Examples([[str(path)] for path in examples],inputs=[audio],label='Prepared review recordings · processed by the same AI')
                identify = gr.Button('Identify sounds',variant='primary')
                status = gr.Markdown('First use downloads pretrained weights. Use a 5–10 second clip for the review.')
            with gr.Column():
                detections = gr.Image(label='AudioSet model · 527 possible labels',type='filepath')
                target = gr.CheckboxGroup(choices=[],label='2 · Which sound should be reduced?',info='The AI can detect several overlapping sounds. Select exactly one target per run.')
                strength = gr.Slider(0,1,value=1,step=.1,label='Reduction strength')
        with gr.Accordion('Controller choice',open=False):
            method = gr.Radio(['Calibrated FIR','Adaptive FxLMS'],value='Calibrated FIR',label='Simulated controller',info='Calibrated FIR fits the assumed acoustic paths. FxLMS learns online from the simulated error microphone and can affect wanted sounds.')
        run = gr.Button('3 · Estimate selected source and simulate ANC',variant='primary')
        with gr.Row():
            before = gr.Audio(label='Before · simulated ear',type='filepath')
            after = gr.Audio(label='After · simulated ear',type='filepath')
        pair = gr.Audio(label='Quick A/B · before, pause, after · same volume',type='filepath')
        report = gr.Markdown()
        graph = gr.Image(label='Measured comparison',type='filepath')
        with gr.Accordion('Check what the AI extracted',open=False):
            reference = gr.Audio(label='Estimated selected source · listen for unwanted leakage',type='filepath')
            downloads = gr.Files(label='Engineering outputs · anti-noise command and report')
            gr.Markdown('The anti-noise command alone is not cleaned audio. Playing it through ordinary earphones will not reproduce this simulated cancellation. If AudioSep cannot fit in available memory, the report identifies the bounded source-estimation fallback.')
        with gr.Accordion('How it works · and the route to headphones',open=False):
            gr.Markdown('''**AudioSet-trained AST → your choice → AudioSep source estimate → control filter → simulated speaker → simulated ear.**

The classifier recognizes likely events. The separator estimates their waveform. The controller generates a negative speaker command; we add its simulated acoustic contribution to the original sound at the ear. No opening noise profile or spectral gate is used.

**What is assumed:** known fixed acoustic paths, an ideal broadband speaker, aligned offline audio, no hardware clipping. Source estimation can miss the target or leak wanted sound. Recognition covers 527 labels, not every sound.

**Future product:** the app sets preferences; a low-latency headphone controller uses reference/error microphones and a measured speaker path. AI updates the choice on a slower loop. Live selective cancellation must be validated on hardware, especially for abrupt clicks. Music is optional playback, not the cancellation mechanism.

Models are pretrained by their original authors; QuantaSound integrates them. Quantum QAOA remains the separate offline experiment in Demo 1/2, not part of this audio controller.

[AudioSet](https://research.google.com/audioset/) · [AST model](https://huggingface.co/MIT/ast-finetuned-audioset-10-10-0.4593) · [AudioSep](https://github.com/Audio-AGI/AudioSep)''')
        identify.click(classify,[audio],[state,target,detections,status],concurrency_id='models',concurrency_limit=1)
        run.click(process,[audio,state,target,strength,method],
                  [before,after,reference,pair,graph,report,downloads],concurrency_id='models',concurrency_limit=1)
        # Do not leave results from a previous recording presented as the new one.
        audio.change(lambda: (None,gr.CheckboxGroup(choices=[],value=None),None,None,None,None,None,None,'',[]),
                     outputs=[state,target,detections,before,after,reference,pair,graph,report,downloads])
        for setting in (target,strength,method):
            setting.change(lambda:(None,None,None,None,None,'',[]),
                           outputs=[before,after,reference,pair,graph,report,downloads])
    return app


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port',type=int,default=7860)
    args = parser.parse_args()
    build().queue().launch(server_name='127.0.0.1',server_port=args.port,share=False)
