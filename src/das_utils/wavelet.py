"""Continuous Morlet wavelet power for DAS strain data.

A ship's signature is a genuinely time-varying (transient, non-stationary)
signal, not just an elevated broadband level - a plain bandpass-then-RMS
reduction, or a static per-channel z-score, throws away exactly the
time-frequency structure that distinguishes a passing vessel from steady
background noise. The continuous wavelet transform (CWT) keeps that
structure: it's a standard, well-understood time-frequency decomposition
(each scale is a band-limited, time-localized filter), used here in its
analytic Morlet form for a good balance of time and frequency resolution.

Implemented directly via FFT (Torrence & Compo's formulation of the Morlet
wavelet in the frequency domain) rather than pulled in from an external
wavelet library, since only a handful of scales are needed and this keeps
the dependency footprint at plain numpy/scipy.
"""

import os
import time

import numpy as np
import pandas as pd

MORLET_W0 = 6.0  # standard nondimensional Morlet frequency (good time/frequency tradeoff)


def morlet_scale_for_freq(freq: np.ndarray | float, w0: float = MORLET_W0) -> np.ndarray | float:
    """Wavelet scale whose dominant frequency is ``freq`` Hz, for the Morlet wavelet."""
    return (w0 + np.sqrt(2 + w0**2)) / (4 * np.pi * np.asarray(freq))


def morlet_cwt_power(
    data: np.ndarray,
    fs: float,
    freqs: np.ndarray,
    w0: float = MORLET_W0,
) -> np.ndarray:
    """Time-averaged Morlet wavelet power at each of ``freqs``, per channel, for one file.

    ``data`` is ``(n_samples, n_channels)``. Computed via FFT: one forward
    FFT of the whole signal, then for each requested frequency the Morlet
    wavelet's known closed-form Fourier-domain shape (Torrence & Compo 1998)
    is multiplied in and inverse-FFT'd back, giving the complex wavelet
    coefficients at that scale for every channel and time sample at once.

    Returns ``(len(freqs), n_channels)``: mean squared wavelet magnitude
    over time (collapsing the time axis) - a single "how much wavelet power
    did each frequency see" value per channel per file, the same shape a
    reducer in ``image.py`` would produce, for one frequency at a time.
    """
    n_samples, n_channels = data.shape
    x = np.asarray(data, dtype=np.float64)
    x = x - x.mean(axis=0, keepdims=True)

    fft_freqs = np.fft.fftfreq(n_samples, d=1.0 / fs) * 2 * np.pi  # angular frequency, rad/s
    xf = np.fft.fft(x, axis=0)

    scales = morlet_scale_for_freq(np.asarray(freqs, dtype=np.float64), w0)
    power = np.empty((len(freqs), n_channels), dtype=np.float64)
    for i, scale in enumerate(scales):
        # Analytic Morlet wavelet, frequency domain (Torrence & Compo 1998, eq. 6),
        # normalized so its FFT has unit energy; zero for negative frequencies (analytic).
        norm = (np.pi**-0.25) * np.sqrt(2 * np.pi * scale * fs)
        wavelet_f = norm * np.exp(-0.5 * (scale * fft_freqs - w0) ** 2) * (fft_freqs > 0)
        coef = np.fft.ifft(xf * wavelet_f[:, None], axis=0)
        power[i] = np.mean(np.abs(coef) ** 2, axis=0)
    return power


def morlet_cwt_power_series(
    data: np.ndarray,
    fs: float,
    freq: float,
    cols: np.ndarray | None = None,
    w0: float = MORLET_W0,
) -> np.ndarray:
    """Time-resolved Morlet wavelet power at one frequency, for selected channels.

    Same computation as ``morlet_cwt_power`` but returns the full time
    series of instantaneous power (``|coefficient|^2``) instead of
    collapsing it - for feeding a genuinely temporal detector (e.g. an
    STA/LTA ratio) rather than a single per-file summary value.
    ``cols`` restricts the (more expensive) inverse FFT to just the channels
    needed, e.g. the stepped subset - pass ``None`` for every channel.

    Returns ``(n_samples, len(cols))``.
    """
    n_samples = data.shape[0]
    cols = np.arange(data.shape[1]) if cols is None else np.asarray(cols)
    x = np.asarray(data[:, cols], dtype=np.float64)
    x = x - x.mean(axis=0, keepdims=True)

    fft_freqs = np.fft.fftfreq(n_samples, d=1.0 / fs) * 2 * np.pi
    xf = np.fft.fft(x, axis=0)

    scale = morlet_scale_for_freq(float(freq), w0)
    norm = (np.pi**-0.25) * np.sqrt(2 * np.pi * scale * fs)
    wavelet_f = norm * np.exp(-0.5 * (scale * fft_freqs - w0) ** 2) * (fft_freqs > 0)
    coef = np.fft.ifft(xf * wavelet_f[:, None], axis=0)
    return np.abs(coef) ** 2


def wavelet_image(
    indexed_files: list[tuple[pd.Timestamp, str]],
    fs: float,
    freq: float,
    cache_path: str | None = None,
    progress_every: int = 200,
) -> tuple[np.ndarray, pd.DatetimeIndex]:
    """Per-channel Morlet wavelet power at a single frequency, one row per file.

    Same shape and role as ``image.build_image``'s output - ``(n_files,
    n_channels)`` - but the per-file, per-channel value is
    ``morlet_cwt_power`` at ``freq`` instead of a bandpass-then-RMS
    reduction, so it can be dropped into anything that consumes an "image"
    (``correlate.estimate_translation_offset``, ``quality``, etc.) to compare
    a specific frequency's sensitivity against the broadband default.
    """
    tag = f"wavelet|{freq}"
    if cache_path and os.path.exists(cache_path):
        cached = np.load(cache_path, allow_pickle=False)
        if (
            "tag" in cached
            and str(cached["tag"]) == tag
            and len(cached["timestamps"]) == len(indexed_files)
        ):
            return cached["image"], pd.DatetimeIndex(cached["timestamps"])

    rows = []
    t0 = time.time()
    for i, (_ts, f) in enumerate(indexed_files):
        arr = np.load(f).astype(np.float64)
        rows.append(morlet_cwt_power(arr, fs, np.array([freq]))[0])
        if progress_every and i % progress_every == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed if elapsed else 0.0
            eta = (len(indexed_files) - i - 1) / rate if rate else float("nan")
            n = len(indexed_files)
            print(f"wavelet_image[{freq}Hz]: {i + 1}/{n} elapsed={elapsed:.0f}s eta={eta:.0f}s")

    image = np.stack(rows, axis=0).astype(np.float32)
    bin_times = pd.DatetimeIndex([ts for ts, _ in indexed_files])

    if cache_path:
        os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
        np.savez(cache_path, image=image, timestamps=bin_times.values, tag=np.array(tag))
    return image, bin_times
