"""Spatio-temporal transforms and augmentations for DAS images.

Operates on any ``(n_time, n_channels)`` array. Two kinds of function live
here:

* **Preprocessing transforms** (``normalize``, ``detrend``, ``taper``,
  ``smooth_spatial``/``smooth_temporal``, ``decimate``, ``crop``) - clean,
  deterministic reshaping of a real image before analysis or modelling.
* **Augmentations** (``add_noise``, ``time_shift``, ``channel_dropout``,
  ``time_mask``) - generate perturbed copies for robustness/training. These
  are seeded for reproducibility.

Everything is axis-explicit: axis 0 is time, axis 1 is channels (the DAS
convention used throughout this package). None of these touch the raw-file
format - they're pure array ops, so they compose and test cleanly.
"""

import numpy as np
from scipy.signal.windows import tukey


def normalize(
    image: np.ndarray, how: str = "zscore", axis: int = 0
) -> np.ndarray:
    """Per-``axis`` normalization: ``zscore`` (mean/std), ``robust``
    (median/MAD), or ``minmax`` (min/max to [0, 1])."""
    img = np.asarray(image, dtype=np.float64)
    if how == "zscore":
        loc = img.mean(axis=axis, keepdims=True)
        scale = img.std(axis=axis, keepdims=True)
    elif how == "robust":
        loc = np.median(img, axis=axis, keepdims=True)
        scale = np.median(np.abs(img - loc), axis=axis, keepdims=True) * 1.4826
    elif how == "minmax":
        scale = img.max(axis=axis, keepdims=True) - img.min(axis=axis, keepdims=True)
        return (img - img.min(axis=axis, keepdims=True)) / np.where(scale == 0, 1, scale)
    else:
        raise ValueError(f"unknown normalization {how!r}")
    scale = np.where(scale == 0, 1.0, scale)
    return (img - loc) / scale


def detrend(image: np.ndarray, axis: int = 0) -> np.ndarray:
    """Subtract a per-channel (axis 1) linear least-squares fit along time (axis 0).

    Removes slow gain drift or tidal/thermal baselines without touching
    higher-frequency content. ``axis`` is the fit axis (time by default).
    """
    img = np.asarray(image, dtype=np.float64)
    n = img.shape[axis]
    t = np.arange(n, dtype=np.float64)
    # fit per column of the non-fit axis
    a = np.stack([np.ones_like(t), t - t.mean()], axis=0)  # (2, n)
    # solve along `axis`: reshape so fit axis is first
    moved = np.moveaxis(img, axis, 0)  # (n, ...channels)
    flat = moved.reshape(n, -1)
    coef, *_ = np.linalg.lstsq(a.T, flat, rcond=None)
    trend = (a.T @ coef).reshape(moved.shape)
    return np.moveaxis(moved - trend, 0, axis)


def taper(image: np.ndarray, fraction: float = 0.1, axis: int = 0) -> np.ndarray:
    """Apply a Tukey (tapered cosine) window along ``axis`` to soften edge effects."""
    img = np.asarray(image, dtype=np.float64).copy()
    n = img.shape[axis]
    window = tukey(n, alpha=fraction)
    return img * window.reshape([-1 if i == axis else 1 for i in range(img.ndim)])


def smooth_spatial(image: np.ndarray, k: int = 3) -> np.ndarray:
    """Moving-average smoothing along channels (axis 1) with window ``2k+1``.

    Mirrors ``smooth_temporal`` but across the fiber - useful to suppress
    per-channel noise while preserving broad spatial features. Edges are
    handled by truncating the averaging window (no padding).
    """
    img = np.asarray(image, dtype=np.float64)
    if k < 1:
        return img
    n = img.shape[1]
    out = np.empty_like(img)
    for c in range(n):
        lo, hi = max(0, c - k), min(n, c + k + 1)
        out[:, c] = img[:, lo:hi].mean(axis=1)
    return out


def smooth_temporal(image: np.ndarray, k: int = 3) -> np.ndarray:
    """Moving-average smoothing along time (axis 0) with window ``2k+1``."""
    img = np.asarray(image, dtype=np.float64)
    if k < 1:
        return img
    n = img.shape[0]
    out = np.empty_like(img)
    for t in range(n):
        lo, hi = max(0, t - k), min(n, t + k + 1)
        out[t] = img[lo:hi].mean(axis=0)
    return out


def decimate(image: np.ndarray, factor: int, axis: int = 0) -> np.ndarray:
    """Mean-pool (block-average) along ``axis`` by integer ``factor``.

    Reduces a dense image to a coarser time grid without aliasing (as a
    simple downsample would). The last partial block is dropped.
    """
    img = np.asarray(image, dtype=np.float64)
    if factor < 2:
        return img
    n = (img.shape[axis] // factor) * factor
    sl = [slice(None)] * img.ndim
    sl[axis] = slice(0, n)
    trimmed = img[tuple(sl)]
    return _mean_block(trimmed, factor, axis)


def _mean_block(arr: np.ndarray, factor: int, axis: int) -> np.ndarray:
    new_shape = list(arr.shape)
    new_shape[axis] = arr.shape[axis] // factor
    new_shape.insert(axis + 1, factor)
    return arr.reshape(new_shape).mean(axis=axis + 1)


def crop(
    image: np.ndarray, t0: int = 0, t1: int | None = None, c0: int = 0, c1: int | None = None
) -> np.ndarray:
    """Sub-range of the image: rows ``[t0:t1]``, columns ``[c0:c1]``."""
    img = np.asarray(image)
    return img[t0:t1, c0:c1]


def add_noise(image: np.ndarray, sigma: float, seed: int | None = None) -> np.ndarray:
    """Add Gaussian noise of std ``sigma`` (in the image's own units) to every sample."""
    rng = np.random.default_rng(seed)
    return np.asarray(image, dtype=np.float64) + rng.normal(0.0, sigma, image.shape)


def time_shift(image: np.ndarray, shift: int, axis: int = 0) -> np.ndarray:
    """Roll along ``axis`` by ``shift`` rows; the vacated edge is zero-filled (not wrapped)."""
    img = np.asarray(image, dtype=np.float64).copy()
    n = img.shape[axis]
    s = int(shift) % n if n else 0
    if s == 0:
        return img
    moved = np.moveaxis(img, axis, 0)
    out = np.zeros_like(moved)
    if s > 0:
        out[s:] = moved[:-s]
    else:
        out[:s] = moved[-s:]
    return np.moveaxis(out, 0, axis)


def channel_dropout(image: np.ndarray, p: float, seed: int | None = None) -> np.ndarray:
    """Zero out a random fraction ``p`` of channels (columns), keeping the rest intact."""
    rng = np.random.default_rng(seed)
    img = np.asarray(image, dtype=np.float64).copy()
    n = img.shape[1]
    drop = rng.random(n) < p
    img[:, drop] = 0.0
    return img


def time_mask(image: np.ndarray, width: int, seed: int | None = None) -> np.ndarray:
    """Zero out a single contiguous block of ``width`` time rows at a random position."""
    rng = np.random.default_rng(seed)
    img = np.asarray(image, dtype=np.float64).copy()
    n = img.shape[0]
    if width >= n:
        return np.zeros_like(img)
    start = int(rng.integers(0, n - width + 1))
    img[start : start + width] = 0.0
    return img
