"""Plots shared by the notebook and the command-line review runner."""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import librosa
import librosa.display
from .data import demo_clips
from .dsp import filtering_example, anc_example

RESULTS = Path(__file__).resolve().parent / "results"


def save(fig, name):
    RESULTS.mkdir(parents=True, exist_ok=True)
    fig.savefig(RESULTS / name, dpi=140, bbox_inches="tight")
    return fig


def spectrum(ax, x, sr, label):
    f, p = signal.welch(x, sr, nperseg=min(2048, len(x)))
    ax.plot(f, 10*np.log10(p+1e-20), label=label)
    ax.set(xlabel="Frequency (Hz)", ylabel="PSD (dB re 1 amplitude²/Hz)")
    ax.grid(alpha=0.2)
    ax.legend()


def demo_a():
    rows = []
    figures = []
    notes = {"engine":"Look for sustained low-frequency bands and harmonics.",
             "siren":"Look for pitched bands that move with time.",
             "keyboard_typing":"Short impacts spread energy across many frequencies."}
    for index, (label, path) in enumerate(demo_clips()):
        x, sr = librosa.load(path, sr=16000, mono=True, duration=5)
        f, p = signal.welch(x, sr, nperseg=2048)
        cdf = np.cumsum(p)/np.sum(p)
        low, high = f[np.searchsorted(cdf, [0.1, 0.9])]
        peak = float(f[np.argmax(p)])
        mel = librosa.feature.melspectrogram(y=x, sr=sr, n_fft=1024, hop_length=256, n_mels=64)
        fig, axes = plt.subplots(2, 1, figsize=(10, 6), layout="constrained")
        axes[0].plot(np.arange(len(x))/sr, x, lw=0.7, color="#146C94")
        axes[0].set(title=label.replace("_", " "), xlabel="Time (s)", ylabel="Amplitude")
        display = librosa.display.specshow(librosa.power_to_db(mel, ref=np.max), sr=sr,
                       hop_length=256, x_axis="time", y_axis="mel", ax=axes[1])
        fig.colorbar(display, ax=axes[1], format="%+2.0f dB", label="Relative to clip maximum")
        note = notes.get(label, "Speech has changing harmonics and broadband consonants.")
        fig.suptitle(f"Demo A · {note}\nMeasured PSD peak {peak:.0f} Hz; central 80% spectral energy {low:.0f}–{high:.0f} Hz", fontsize=10)
        figures.append(save(fig, f"a_{index+1}.png"))
        rows.append({"label":label,"file":path.name,"peak_hz":peak,
                     "energy_10_percent_hz":float(low),"energy_90_percent_hz":float(high)})
    return figures, rows


def demo_b():
    t, clean, noise, output, metrics = filtering_example()
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), layout="constrained")
    window = (t >= 1) & (t < 1.025)
    axes[0].plot(t[window], (clean+noise)[window], label="Noisy input", alpha=0.7)
    axes[0].plot(t[window], output[window], label="Filtered output")
    axes[0].plot(t[window], clean[window], label="Original clean target", ls="--", alpha=0.7)
    axes[0].set(xlabel="Time (s)", ylabel="Amplitude"); axes[0].legend()
    spectrum(axes[1], (clean+noise)[2000:], 8000, "Before")
    spectrum(axes[1], output[2000:], 8000, "After")
    fig.suptitle(f"Demo B · Digital filtering, NOT ANC\nComponent SNR improvement: {metrics['snr_improvement_db']:.2f} dB")
    return save(fig, "b_filtering.png"), metrics


def demo_c(kind):
    t, result, metrics = anc_example(kind)
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), layout="constrained")
    axes[0].plot(t, result.error, lw=0.4, color="#146C94")
    axes[0].set(xlabel="Time (s)", ylabel="Residual amplitude", title="Error microphone e = d − S*y")
    block = 800
    for x, label in [(result.disturbance, "Uncontrolled d"), (result.error, "Residual e")]:
        power = np.mean(x[:len(x)//block*block].reshape(-1, block)**2, axis=1)
        axes[1].plot((np.arange(len(power))+.5)*block/8000, 10*np.log10(power+1e-20), label=label)
    axes[1].axvspan(4, 8, alpha=0.12, color="green", label="Reported measurement window")
    axes[1].set(xlabel="Time (s)", ylabel="100 ms mean power (dB)"); axes[1].legend()
    spectrum(axes[2], result.disturbance[len(t)//2:], 8000, "Before: d")
    spectrum(axes[2], result.error[len(t)//2:], 8000, "After: e")
    axes[2].set_xlim(0, 1500)
    fig.suptitle(f"Demo C · Simulated FxLMS · {kind}\nNoise reduction over 4–8 s: {metrics['noise_reduction_db']:.2f} dB")
    save(fig, f"c_{kind}.png")
    # Same gain on both WAVs preserves the measured level difference.
    import soundfile as sf
    gain = 0.9/max(np.max(np.abs(result.disturbance)), np.max(np.abs(result.error)), 1e-12)
    sf.write(RESULTS/f"c_{kind}_before.wav", gain*result.disturbance, 8000, subtype="PCM_16")
    sf.write(RESULTS/f"c_{kind}_after.wav", gain*result.error, 8000, subtype="PCM_16")
    return fig, metrics
