"""Frequency-domain augmentations for DAS waveform data.

The existing ``augment.py`` covers spatial/temporal array ops. DAS data is
inherently spectral - strain signals are broadband with characteristic
frequency bands for different physical sources (vessel engines ~1-20 Hz,
seismic ~0.1-5 Hz, environmental/tidal < 0.1 Hz). Frequency-domain
augmentations exploit this structure to make a model robust to spectral
variations that genuinely occur between deployments, seasons, or nearby
noise sources.

All functions here operate on **raw waveform** data, shape
``(n_time, n_channels)``, with a known sampling rate ``fs`` (Hz). The time
axis (axis 0) is the signal axis; each channel is filtered independently.
They return float64 arrays of the same shape (unless a resampling op
changes the time dimension).

Two categories:

* **Spectral filters** (deterministic, not seeded): ``bandpass``,
  ``bandstop``, ``lowpass``, ``highpass``, ``notch`` - shape the frequency
  content. Useful as preprocessing or as fixed augmentation.
* **Spectral augmentations** (random, seeded): ``spectral_mask``,
  ``frequency_shift``, ``phase_shift``, ``spectral_warp``, ``mixup``,
  ``channel_shuffle``, ``gain_randomization``, ``time_warp``,
  ``frequency_jitter`` - perturb the spectrum to simulate deployment
  variability.

Convention: all augmentations accept a ``seed`` for reproducibility, and
return a new array (never modify in place).
"""

import numpy as np
from scipy.fft import irfft, rfft, rfftfreq
from scipy.signal import butter, sosfiltfilt

# ---------------------------------------------------------------------------
# Spectral filters (deterministic)
# ---------------------------------------------------------------------------


def bandpass(
    image: np.ndarray, fs: float, low: float, high: float, order: int = 4
) -> np.ndarray:
    """Butterworth bandpass ``[low, high]`` Hz along time (axis 0), per channel."""
    sos = butter(order, [low, high], btype="bandpass", fs=fs, output="sos")
    return sosfiltfilt(sos, np.asarray(image, dtype=np.float64), axis=0).astype(np.float64)


def bandstop(
    image: np.ndarray, fs: float, low: float, high: float, order: int = 4
) -> np.ndarray:
    """Butterworth band-stop (notch band) ``[low, high]`` Hz along time, per channel."""
    sos = butter(order, [low, high], btype="bandstop", fs=fs, output="sos")
    return sosfiltfilt(sos, np.asarray(image, dtype=np.float64), axis=0).astype(np.float64)


def lowpass(
    image: np.ndarray, fs: float, cutoff: float, order: int = 4
) -> np.ndarray:
    """Butterworth lowpass below ``cutoff`` Hz along time, per channel."""
    sos = butter(order, cutoff, btype="lowpass", fs=fs, output="sos")
    return sosfiltfilt(sos, np.asarray(image, dtype=np.float64), axis=0).astype(np.float64)


def highpass(
    image: np.ndarray, fs: float, cutoff: float, order: int = 4
) -> np.ndarray:
    """Butterworth highpass above ``cutoff`` Hz along time, per channel."""
    sos = butter(order, cutoff, btype="highpass", fs=fs, output="sos")
    return sosfiltfilt(sos, np.asarray(image, dtype=np.float64), axis=0).astype(np.float64)


def notch(
    image: np.ndarray, fs: float, freq: float, quality: float = 30.0
) -> np.ndarray:
    """Notch (band-stop) filter at a single ``freq`` Hz with quality factor Q.

    Removes a narrow band - useful for removing a known tonal interferer
    (e.g. 50/60 Hz mains hum, a fixed-frequency mechanical resonance).
    ``quality`` is the IIR notch Q: higher = narrower stop band.
    """
    half = freq / (2 * quality)
    sos = butter(4, [max(freq - half, 1e-4), freq + half], btype="bandstop", fs=fs, output="sos")
    return sosfiltfilt(sos, np.asarray(image, dtype=np.float64), axis=0).astype(np.float64)


# ---------------------------------------------------------------------------
# Spectral augmentations (random, seeded)
# ---------------------------------------------------------------------------


def spectral_mask(
    image: np.ndarray,
    fs: float,
    n_bands: int = 1,
    max_width_hz: float = 5.0,
    seed: int | None = None,
) -> np.ndarray:
    """Zero out ``n_bands`` random frequency bands in the rFFT of each channel.

    Similar to SpecAugust time masking but in the frequency domain: picks
    ``n_bands`` random contiguous frequency ranges, each up to
    ``max_width_hz`` wide, and zeros their rFFT coefficients before
    inverse-transforming. Forces the model to not rely on any narrow
    frequency band.

    Each channel gets independent mask positions (so different channels
    lose different bands - stronger than masking all channels identically).
    """
    img = np.asarray(image, dtype=np.float64)
    n_time, n_chan = img.shape
    freqs = rfftfreq(n_time, d=1.0 / fs)
    rng = np.random.default_rng(seed)

    spec = rfft(img, axis=0)
    for c in range(n_chan):
        for _ in range(n_bands):
            width = rng.uniform(0, max_width_hz)
            f0 = rng.uniform(freqs[1], freqs[-1] - width)
            mask = (freqs >= f0) & (freqs < f0 + width)
            spec[mask, c] = 0.0
    return irfft(spec, n=n_time, axis=0)


def frequency_shift(
    image: np.ndarray, fs: float, max_shift_hz: float = 2.0, seed: int | None = None
) -> np.ndarray:
    """Shift the spectrum of each channel by a random amount up to ``max_shift_hz``.

    Multiplicative shift in the frequency domain (circular): moves all
    spectral content up or down by a random offset in [-max, max] Hz. This
    simulates Doppler-like effects or slight variations in the
    interrogator's effective sampling rate between deployments.

    Each channel gets an independent shift. The shift is an integer number
    of FFT bins (no fractional interpolation).
    """
    img = np.asarray(image, dtype=np.float64)
    n_time, n_chan = img.shape
    rng = np.random.default_rng(seed)
    spec = rfft(img, axis=0)
    max_bins = int(max_shift_hz / fs * n_time)
    out = np.empty_like(spec)
    for c in range(n_chan):
        shift = int(rng.integers(-max_bins, max_bins + 1)) if max_bins > 0 else 0
        out[:, c] = np.roll(spec[:, c], shift, axis=0)
    return irfft(out, n=n_time, axis=0)


def phase_shift(
    image: np.ndarray, seed: int | None = None
) -> np.ndarray:
    """Randomize the phase of each channel's spectrum, keeping magnitudes.

    Replaces the phase of every rFFT coefficient with uniform random noise
    while preserving the power spectrum. This destroys temporal structure
    (the waveform looks like noise) but keeps the frequency content
    identical - a strong test of whether a model relies on phase vs.
    amplitude information.

    Each channel gets independent phase.
    """
    img = np.asarray(image, dtype=np.float64)
    n_time, n_chan = img.shape
    rng = np.random.default_rng(seed)
    spec = rfft(img, axis=0)
    mag = np.abs(spec)
    phase = np.exp(1j * rng.uniform(0, 2 * np.pi, spec.shape))
    spec = mag * phase
    # DC and (for even-length) Nyquist bins must be real
    spec[0, :] = np.abs(spec[0, :])
    if n_time % 2 == 0:
        spec[-1, :] = np.abs(spec[-1, :])
    return irfft(spec, n=n_time, axis=0)


def spectral_warp(
    image: np.ndarray,
    fs: float,
    max_warp: float = 0.1,
    seed: int | None = None,
) -> np.ndarray:
    """Non-linearly warp the frequency axis of each channel's spectrum.

    Resamples the magnitude spectrum with a random piecewise-linear warp,
    stretching some frequency bands and compressing others, then
    re-applies the original phase. ``max_warp`` is the maximum fractional
    displacement of any frequency bin (0.1 = bins can move by 10% of the
    spectrum length). This simulates non-uniform frequency scaling (e.g.
    dispersion effects, varying propagation conditions).

    Only the magnitude is warped; phase is preserved from the original.
    """
    img = np.asarray(image, dtype=np.float64)
    n_time, n_chan = img.shape
    n_freq = n_time // 2 + 1
    rng = np.random.default_rng(seed)
    spec = rfft(img, axis=0)
    mag = np.abs(spec)
    phase = np.angle(spec)

    # random warp function: piecewise linear displacement
    orig_bins = np.arange(n_freq, dtype=np.float64)
    max_disp = max_warp * n_freq
    out_mag = np.empty_like(mag)
    for c in range(n_chan):
        displ = rng.uniform(-max_disp, max_disp, size=max(n_freq // 8, 4))
        # interpolate displacement to full resolution
        ctrl_pts = np.linspace(0, n_freq - 1, len(displ))
        full_disp = np.interp(orig_bins, ctrl_pts, displ)
        warped_bins = np.clip(orig_bins + full_disp, 0, n_freq - 1)
        out_mag[:, c] = np.interp(warped_bins, orig_bins, mag[:, c])

    warped_spec = out_mag * np.exp(1j * phase)
    warped_spec[0, :] = np.abs(warped_spec[0, :])
    if n_time % 2 == 0:
        warped_spec[-1, :] = np.abs(warped_spec[-1, :])
    return irfft(warped_spec, n=n_time, axis=0)


def mixup(
    image: np.ndarray, alpha: float = 0.2, seed: int | None = None
) -> np.ndarray:
    """Linearly combine each channel with a randomly chosen other channel.

    For each channel ``c``, picks a random channel ``c'`` and replaces
    ``c`` with ``lambda * c + (1 - lambda) * c'`` where ``lambda ~ Beta(alpha,
    alpha)``. This is the classic MixUp augmentation adapted per-channel
    rather than per-sample, encouraging the model to learn smooth
    transitions between neighbouring spatial positions.

    ``alpha`` controls the mixing strength: 0 = no mixing, larger = more
    aggressive (0.2 is a mild default).
    """
    img = np.asarray(image, dtype=np.float64)
    n_time, n_chan = img.shape
    rng = np.random.default_rng(seed)
    out = img.copy()
    partners = rng.integers(0, n_chan, size=n_chan)
    lambdas = rng.beta(alpha, alpha, size=n_chan)
    for c in range(n_chan):
        out[:, c] = lambdas[c] * img[:, c] + (1 - lambdas[c]) * img[:, partners[c]]
    return out


def channel_shuffle(
    image: np.ndarray, max_swaps: int = 5, seed: int | None = None
) -> np.ndarray:
    """Randomly swap ``max_swaps`` pairs of channels.

    Exchanges the data of two channels entirely. Unlike ``channel_dropout``
    (which zeros), this preserves all information but misplaces it spatially
    - a strong regularizer against over-reliance on channel-position
    priors. Only nearby channels within a small window are swapped by
    default to keep the perturbation physically plausible.
    """
    img = np.asarray(image, dtype=np.float64)
    n_chan = img.shape[1]
    rng = np.random.default_rng(seed)
    out = img.copy()
    for _ in range(max_swaps):
        i, j = int(rng.integers(0, n_chan)), int(rng.integers(0, n_chan))
        out[:, [i, j]] = out[:, [j, i]]
    return out


def gain_randomization(
    image: np.ndarray,
    low: float = 0.8,
    high: float = 1.2,
    seed: int | None = None,
) -> np.ndarray:
    """Multiply each channel by a random gain factor in ``[low, high]``.

    Simulates per-channel gain variations between deployments (different
    interrogator settings, fiber attenuation changes, connector
    differences). Each channel gets an independent scalar multiplier.
    """
    img = np.asarray(image, dtype=np.float64)
    n_chan = img.shape[1]
    rng = np.random.default_rng(seed)
    gains = rng.uniform(low, high, size=n_chan)
    return img * gains[np.newaxis, :]


def time_warp(
    image: np.ndarray, fs: float, max_stretch: float = 0.1, seed: int | None = None
) -> np.ndarray:
    """Non-linearly warp the time axis of each channel by up to ``max_stretch``.

    Resamples each channel's time series with a random piecewise-linear
    time warp, stretching some segments and compressing others. This
    simulates variable propagation speed or non-uniform strain rates along
    the fiber. ``max_stretch`` is the maximum fractional time displacement
    (0.1 = up to 10% of the time axis).

    Uses ``scipy.signal.resample_poly`` for anti-aliased resampling.
    """
    img = np.asarray(image, dtype=np.float64)
    n_time, n_chan = img.shape
    rng = np.random.default_rng(seed)
    orig_t = np.arange(n_time, dtype=np.float64)
    out = np.empty_like(img)
    max_disp = max_stretch * n_time
    for c in range(n_chan):
        displ = rng.uniform(-max_disp, max_disp, size=max(n_time // 16, 4))
        ctrl = np.linspace(0, n_time - 1, len(displ))
        full_disp = np.interp(orig_t, ctrl, displ)
        warped_t = np.clip(orig_t + full_disp, 0, n_time - 1)
        out[:, c] = np.interp(warped_t, orig_t, img[:, c])
    return out


def frequency_jitter(
    image: np.ndarray, fs: float, sigma_hz: float = 0.5, seed: int | None = None
) -> np.ndarray:
    """Add independent Gaussian jitter to each channel's frequency bins.

    Perturbs the magnitude of each rFFT coefficient by multiplicative
    Gaussian noise in the log-spectrum: ``log|H| += N(0, sigma_hz)``. This
    is a gentle, smooth spectral perturbation (unlike ``spectral_mask``,
    which zeros bands) that simulates small spectral variability between
    recording sessions.

    ``sigma_hz`` controls the perturbation strength (0.5 = ~50% multiplicative
    variation per bin on average).
    """
    img = np.asarray(image, dtype=np.float64)
    n_time, n_chan = img.shape
    rng = np.random.default_rng(seed)
    spec = rfft(img, axis=0)
    log_mag = np.log(np.abs(spec) + 1e-12)
    jitter = rng.normal(0, sigma_hz, spec.shape)
    mag = np.exp(log_mag + jitter)
    phase = np.angle(spec)
    out_spec = mag * np.exp(1j * phase)
    out_spec[0, :] = np.abs(out_spec[0, :])
    if n_time % 2 == 0:
        out_spec[-1, :] = np.abs(out_spec[-1, :])
    return irfft(out_spec, n=n_time, axis=0)


# ---------------------------------------------------------------------------
# Composition helper
# ---------------------------------------------------------------------------


def random_augment(
    image: np.ndarray,
    fs: float,
    p: float = 0.5,
    seed: int | None = None,
) -> np.ndarray:
    """Apply a random subset of spectral augmentations, each with probability ``p``.

    A convenience composition: each augmentation in the palette is applied
    independently with probability ``p``. At least one is always applied
    (if p > 0 and the image is non-trivial). Returns the augmented image.
    Use this as a one-call training-time augmentation pipeline.

    The palette: ``spectral_mask``, ``frequency_shift``, ``gain_randomization``,
    ``frequency_jitter``, ``mixup``, ``channel_shuffle``, ``time_warp``.
    Phase/warp are excluded from the default palette as they're more
    destructive - add them manually if desired.
    """
    img = np.asarray(image, dtype=np.float64)
    rng = np.random.default_rng(seed)
    augmentations = [
        lambda x: spectral_mask(x, fs, seed=int(rng.integers(0, 2**31))),
        lambda x: frequency_shift(x, fs, max_shift_hz=2.0, seed=int(rng.integers(0, 2**31))),
        lambda x: gain_randomization(x, seed=int(rng.integers(0, 2**31))),
        lambda x: frequency_jitter(x, fs, sigma_hz=0.3, seed=int(rng.integers(0, 2**31))),
        lambda x: mixup(x, alpha=0.2, seed=int(rng.integers(0, 2**31))),
        lambda x: channel_shuffle(x, max_swaps=3, seed=int(rng.integers(0, 2**31))),
        lambda x: time_warp(x, fs, max_stretch=0.05, seed=int(rng.integers(0, 2**31))),
    ]
    out = img
    applied = 0
    for aug in augmentations:
        if rng.random() < p:
            out = aug(out)
            applied += 1
    if applied == 0 and p > 0:
        out = augmentations[0](out)
    return out
