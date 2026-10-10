"""Classical filtering and sample-by-sample filtered-x LMS."""
from dataclasses import dataclass
import numpy as np
from scipy import signal


def power_db_ratio(before, after):
    """10 log10(mean(before**2) / mean(after**2)), with a numerical floor."""
    return float(10 * np.log10(max(np.mean(np.asarray(before)**2), 1e-20)
                              / max(np.mean(np.asarray(after)**2), 1e-20)))


def filtering_example(sr=8000, seconds=4, seed=7):
    """Known 1 kHz target plus 200 Hz hum; a causal Butterworth band-stop."""
    t = np.arange(int(sr * seconds)) / sr
    clean = 0.4 * np.sin(2 * np.pi * 1000 * t)
    noise = 0.5 * np.sin(2 * np.pi * 200 * t) + 0.025 * np.random.default_rng(seed).normal(size=t.size)
    b, a = signal.butter(3, [150, 250], btype="bandstop", fs=sr)
    filtered_clean = signal.lfilter(b, a, clean)
    filtered_noise = signal.lfilter(b, a, noise)
    output = signal.lfilter(b, a, clean + noise)
    # Exclude the first 0.25 s equally; no delay alignment or gain fitting.
    region = slice(int(0.25 * sr), None)
    before = power_db_ratio(clean[region], noise[region])
    after = power_db_ratio(filtered_clean[region], filtered_noise[region])
    metrics = {"input_snr_db": before, "output_component_snr_db": after,
               "snr_improvement_db": after - before,
               "clean_reference_sdr_db": power_db_ratio(clean[region], output[region] - clean[region]),
               "measurement_start_s": 0.25}
    return t, clean, noise, output, metrics


@dataclass
class ANCResult:
    reference: np.ndarray
    disturbance: np.ndarray
    control: np.ndarray
    secondary_output: np.ndarray
    error: np.ndarray
    weights: np.ndarray


def fxlms(x, primary, secondary, secondary_estimate=None, length=32, mu=0.003):
    """Simulate e=d-S*y, w <- w+mu*e*x_filtered.

    Physical loudspeaker command is -y under this sign convention. Both the
    real secondary path and its estimate retain independent FIR history.
    The external reference arrives before the disturbance at the ear.
    """
    x = np.asarray(x, dtype=float)
    p, s = np.asarray(primary, dtype=float), np.asarray(secondary, dtype=float)
    sh = s.copy() if secondary_estimate is None else np.asarray(secondary_estimate, dtype=float)
    if x.ndim != 1 or not x.size or not np.all(np.isfinite(x)):
        raise ValueError("Reference must be a finite nonempty mono signal")
    if length < 1 or mu < 0 or not np.isfinite(mu):
        raise ValueError("length must be positive and mu finite and nonnegative")
    if any(h.ndim != 1 or h.size == 0 or not np.all(np.isfinite(h)) for h in (p, s, sh)):
        raise ValueError("Paths must be finite nonempty FIR arrays")
    d = signal.lfilter(p, [1.0], x)
    xf = signal.lfilter(sh, [1.0], x)
    w = np.zeros(length)
    xb, xfb, yb = np.zeros(length), np.zeros(length), np.zeros(len(s))
    y, sy, e = np.zeros_like(x), np.zeros_like(x), np.zeros_like(x)
    for n in range(len(x)):
        xb[1:] = xb[:-1]; xb[0] = x[n]
        y[n] = w @ xb
        yb[1:] = yb[:-1]; yb[0] = y[n]
        sy[n] = s @ yb  # stateful propagation of controller output to the ear
        e[n] = d[n] - sy[n]
        xfb[1:] = xfb[:-1]; xfb[0] = xf[n]
        w += mu * e[n] * xfb
        if not np.all(np.isfinite(w)) or np.linalg.norm(w) > 1e6:
            raise RuntimeError("FxLMS diverged; reduce mu and check path/sign assumptions")
    return ANCResult(x, d, y, sy, e, w)


def anc_example(kind="tonal", sr=8000, seconds=8, seed=7):
    t = np.arange(int(sr * seconds)) / sr
    rng = np.random.default_rng(seed)
    if kind == "tonal":
        x = 0.5 * np.sin(2 * np.pi * 200 * t)
    elif kind == "broadband":
        # Low-frequency broadband noise, explicitly not full-band white noise.
        x = signal.sosfilt(signal.butter(3, 900, fs=sr, output="sos"), rng.normal(size=t.size))
        x *= 0.3 / np.sqrt(np.mean(x*x))
    else:
        raise ValueError("kind must be tonal or broadband")
    # Primary delay 6 samples > secondary delay 2: causal reference preview.
    p = np.r_[np.zeros(6), 0.8, 0.25, -0.1]
    s = np.array([0., 0., 0.7, 0.2, -0.05])
    result = fxlms(x, p, s)
    tail = slice(len(x)//2, None)
    metrics = {"noise_reduction_db": power_db_ratio(result.disturbance[tail], result.error[tail]),
               "whole_run_reduction_db": power_db_ratio(result.disturbance, result.error),
               "measurement_start_s": seconds/2, "measurement_end_s": seconds,
               "sample_rate_hz": sr, "mu": 0.003, "controller_taps": 32,
               "primary_fir": p.tolist(), "secondary_fir": s.tolist(),
               "known_secondary_path": True, "noise_kind": kind}
    return t, result, metrics
