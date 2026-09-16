"""Stream raw DAS strain files into a compact, time-resolved per-channel image.

Raw DAS data is too large to hold for long stretches (hours of data across
thousands of channels is tens to hundreds of GB). This generalizes the
reduction in ``energy.py``: stream one file at a time, optionally bandpass,
and reduce each file to a single per-channel row with a pluggable reducer -
producing a ``(n_files, n_channels)`` array small enough to keep in memory,
with an optional on-disk cache.

The downstream modules (``stats``, ``quality``, ``channels``, ``augment``)
operate on such an image (any ``(n_time, n_channels)`` array), so they stay
decoupled from the raw-file format and are testable with synthetic data.
"""

import os
import time

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt

# Reducers: name -> function(axis=0) returning a 1-D per-channel row.
_REDUCERS = {
    "rms": lambda a: np.sqrt(np.mean(a**2, axis=0)),
    "std": lambda a: a.std(axis=0),
    "mean": lambda a: a.mean(axis=0),
    "abs_mean": lambda a: np.mean(np.abs(a), axis=0),
    "max": lambda a: np.max(np.abs(a), axis=0),
}


def build_image(
    indexed_files: list[tuple[pd.Timestamp, str]],
    reducer: str = "rms",
    fs: float | None = None,
    bandpass_hz: tuple[float, float] | None = None,
    cache_path: str | None = None,
    progress_every: int = 200,
) -> tuple[np.ndarray, pd.DatetimeIndex]:
    """Per-channel reduced image, one row per file, ``(n_files, n_channels)`` float32.

    ``indexed_files`` is ``[(timestamp, path), ...]`` as returned by
    ``io.index_das_files`` (already time-sorted). ``reducer`` selects the
    per-file, per-channel summary statistic (``rms`` default - a
    bandpass-free noise/energy proxy; use ``std`` for a mean-removed noise
    estimate). If ``bandpass_hz`` is given, a Butterworth bandpass is applied
    along axis 0 before reducing (requires ``fs``).

    Returns ``(image, bin_times)``. If ``cache_path`` exists with a matching
    row count (and matching reducer/bandpass), it is loaded instead of
    recomputing - a coarse cache key (file *count* plus reducer/bandpass
    tag), so bust the cache manually if files were swapped without changing
    the count.
    """
    if reducer not in _REDUCERS:
        raise ValueError(f"unknown reducer {reducer!r}; choose from {sorted(_REDUCERS)}")
    tag = f"{reducer}|{bandpass_hz}"
    if cache_path and os.path.exists(cache_path):
        cached = np.load(cache_path, allow_pickle=False)
        if (
            "tag" in cached
            and str(cached["tag"]) == tag
            and len(cached["timestamps"]) == len(indexed_files)
        ):
            return cached["image"], pd.DatetimeIndex(cached["timestamps"])

    reduce_fn = _REDUCERS[reducer]
    sos = butter(4, bandpass_hz, btype="bandpass", fs=fs, output="sos") if bandpass_hz else None

    rows = []
    t0 = time.time()
    for i, (_ts, f) in enumerate(indexed_files):
        arr = np.load(f).astype(np.float32)
        if sos is not None:
            arr = sosfiltfilt(sos, arr, axis=0)
        rows.append(reduce_fn(arr))
        if progress_every and i % progress_every == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed if elapsed else 0.0
            eta = (len(indexed_files) - i - 1) / rate if rate else float("nan")
            n = len(indexed_files)
            print(f"build_image[{reducer}]: {i + 1}/{n} elapsed={elapsed:.0f}s eta={eta:.0f}s")

    image = np.stack(rows, axis=0).astype(np.float32)
    bin_times = pd.DatetimeIndex([ts for ts, _ in indexed_files])

    if cache_path:
        os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
        np.savez(cache_path, image=image, timestamps=bin_times.values, tag=np.array(tag))
    return image, bin_times
