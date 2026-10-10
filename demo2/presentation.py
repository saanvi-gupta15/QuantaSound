"""Metrics and plots for the personalized experiment."""
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal


def metric_rows(result):
    m = result.metrics
    selected = m["selected_residual_reduction_db"]
    return [
        ("Total residual-noise reduction", f"{m['total_noise_reduction_db']:.2f} dB", "Higher is better; unselected noise is still present"),
        ("Selected-target residual reduction", "N/A" if selected is None else f"{selected:.2f} dB", "Includes added artifacts; can be negative"),
        ("Speech-reference SDR before → after", f"{m['speech_reference_sdr_before_db']:.2f} → {m['speech_reference_sdr_after_db']:.2f} dB", "Remaining interference and distortion relative to clean speech"),
        ("Speech counterfactual distortion", f"{m['speech_counterfactual_distortion_rms_percent']:.2f}% RMS", "Lower is better; paired speech-present/absent runs"),
        ("Speech counterfactual gain", f"{m['speech_counterfactual_gain']:.4f}", "Near 1 is desirable; not a perceptual speech-quality score"),
    ]


def plot_result(result):
    t = np.arange(len(result.before))/result.sr
    fig, axes = plt.subplots(3, 1, figsize=(11, 9), layout="constrained")
    axes[0].plot(t, result.before, label="Before", alpha=0.55, lw=0.5)
    axes[0].plot(t, result.after, label="After", alpha=0.8, lw=0.5)
    axes[0].set(xlabel="Time (s)", ylabel="Ear signal amplitude")
    axes[0].legend(loc="upper right")
    block = result.sr//10
    for x, label in [(result.before-result.speech, "Residual relative to speech: before"),
                     (result.after-result.speech, "Residual relative to speech: after")]:
        power = np.mean(x[:len(x)//block*block].reshape(-1,block)**2, axis=1)
        axes[1].plot((np.arange(len(power))+.5)*block/result.sr, 10*np.log10(power+1e-20), label=label)
    start, end = result.metrics["measurement_window_s"]
    axes[1].axvspan(start, end, color="green", alpha=.1)
    axes[1].set(xlabel="Time (s)", ylabel="100 ms power (dB)")
    axes[1].legend(loc="upper right")
    for x, label in [(result.before-result.speech, "Residual before"), (result.after-result.speech, "Residual after")]:
        f, power = signal.welch(x[len(x)//2:], result.sr, nperseg=2048)
        axes[2].plot(f, 10*np.log10(power+1e-20), label=label)
    axes[2].set(xlabel="Frequency (Hz)", ylabel="PSD (dB re 1 amplitude²/Hz)")
    axes[2].legend()
    for ax in axes:
        ax.grid(alpha=.15)
    selected = ", ".join(result.metrics["selected_for_reduction"]) or "none"
    fig.suptitle(f"Known-reference ANC simulation · reduce {selected} · preserve speech\n"
                 f"Total noise reduction: {result.metrics['total_noise_reduction_db']:.2f} dB; measured {start:g}–{end:g} s")
    return fig


def show_result(result, directory):
    from IPython.display import Audio, HTML, display
    from .simulation import export_result
    before, after, metrics = export_result(result, directory)
    display(HTML("<table><tr><th>Measurement</th><th>Result</th><th>Meaning</th></tr>" +
                 "".join("<tr>"+"".join(f"<td>{value}</td>" for value in row)+"</tr>" for row in metric_rows(result))+"</table>"))
    print("Before — same playback gain as after")
    display(Audio(filename=str(before)))
    print("After — do not adjust player volume between comparisons")
    display(Audio(filename=str(after)))
    # Sequential playback uses the exact exported samples and a single player
    # volume. It does not normalize each half or change the measured signals.
    import soundfile as sf
    a, sr = sf.read(before)
    b, _ = sf.read(after)
    split = len(a)//2
    pair = np.r_[a[split:], np.zeros(sr//2), b[split:]]
    pair_path = metrics.parent/"before_then_after.wav"
    sf.write(pair_path, pair, sr, subtype="PCM_16")
    duration = (len(a)-split)/sr
    print(f"Quick A/B: BEFORE (final {duration:g} s), 0.5 s silence, AFTER (same final {duration:g} s). One player, unchanged gain.")
    display(Audio(filename=str(pair_path)))
    figure = plot_result(result)
    display(figure)
    plt.close(figure)
    print(f"Saved audio and full metrics in {metrics.parent}")


def show_sources(sources, sr=16000):
    """Identification-only auditions, explicitly independently scaled."""
    from IPython.display import Audio, display
    print("Sources present: engine/hum, keyboard and speech. There is NO siren in this mixture.")
    print("Each solo below is independently scaled for identification, NOT for judging cancellation or mixture balance.")
    for name in ("engine", "keyboard", "speech"):
        x = sources[name]
        print(name)
        peak = max(float(np.max(np.abs(x))), 1e-12)
        display(Audio(0.7*x/peak, rate=sr, normalize=False))
