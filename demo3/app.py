"""Launch a local Python Gradio interface: python -m demo3.app"""
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
from scipy import signal
import gradio as gr
from .ai_pipeline import process_file


def plot_comparison(input_path, output_path):
    before, sr = sf.read(input_path)
    after, _ = sf.read(output_path)
    if before.ndim == 2: before = before.mean(axis=1)
    if after.ndim == 2: after = after.mean(axis=1)
    n = min(len(before), len(after), sr*10)
    t = np.arange(n)/sr
    fig, axes = plt.subplots(2, 1, figsize=(9, 5.5), layout="constrained")
    axes[0].plot(t, before[:n], lw=.45, alpha=.65, label="Before")
    axes[0].plot(t, after[:n], lw=.45, alpha=.8, label="After")
    axes[0].set(xlabel="Time (s)", ylabel="Amplitude", title="Waveform")
    axes[0].legend()
    for x, label in [(before[n//2:n], "Before"), (after[n//2:n], "After")]:
        f, p = signal.welch(x, sr, nperseg=min(2048, len(x)))
        axes[1].plot(f, 10*np.log10(p+1e-20), lw=1, label=label)
    axes[1].set(xlabel="Frequency (Hz)", ylabel="PSD (dB)", title="Spectrum · final half")
    axes[1].set_xlim(0, min(sr/2, 8000)); axes[1].legend()
    for ax in axes: ax.grid(alpha=.2)
    fig.suptitle("QuantaSound · arbitrary-audio denoising")
    figure_path = Path(output_path).with_suffix(".png")
    fig.savefig(figure_path, dpi=130)
    plt.close(fig)
    return figure_path


def run(upload, noise_seconds, reduction_db):
    if upload is None:
        return None, None, "Upload or record an audio clip first."
    try:
        result = process_file(upload, Path(upload).with_name(Path(upload).stem+"_quanta_denoised.wav"),
                              noise_seconds, reduction_db)
        figure = plot_comparison(upload, result["output"])
        c = result["classification"]
        text = (f"Detected mixture: {c['label']}\nReason: {c['reason']}\n"
                f"Estimated level change: {result['estimated_level_change_db']:.2f} dB\n"
                f"Output: {result['output']}\n\n"
                "This output uses AI-assisted spectral denoising. It attenuates time-frequency bins; "
                "it does not claim physical anti-noise ANC for an unknown upload. "
                "For true ANC, provide a correlated reference microphone and calibrated speaker-to-ear path.")
        return result["output"], figure, text
    except Exception as error:
        return None, None, f"Could not process audio: {error}"


def main():
    with gr.Blocks(title="QuantaSound AI Audio Lab") as demo:
        gr.Markdown("""# QuantaSound AI Audio Lab
Upload or record any audio clip. The Python pipeline classifies the mixture, estimates a noise profile from the opening segment and produces a denoised WAV.

**Scientific boundary:** this upload mode is denoising. True physical ANC generates anti-noise and requires a correlated reference plus a calibrated secondary path; it is demonstrated separately in the Personalization notebook.""")
        audio = gr.Audio(sources=["upload", "microphone"], type="filepath", label="Noisy audio")
        noise = gr.Slider(.05, 2, value=.25, step=.05, label="Opening noise-profile duration (seconds)")
        reduction = gr.Slider(3, 30, value=12, step=1, label="Spectral attenuation floor (dB)")
        button = gr.Button("Analyze and denoise", variant="primary")
        output = gr.Audio(label="Denoised output", type="filepath")
        graph = gr.Image(label="Before / after graphs", type="filepath")
        report = gr.Textbox(label="Explainable report", lines=8)
        button.click(run, [audio, noise, reduction], [output, graph, report])
    demo.launch()


if __name__ == "__main__":
    main()
