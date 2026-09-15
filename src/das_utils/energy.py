"""Reduce raw DAS strain files to a compressed, time-resolved per-channel energy image.

Raw DAS data is too large to hold for long stretches (hours of data across
thousands of channels is tens to hundreds of GB). This streams one file at a
time, bandpass-filters it, and reduces each file to a single per-channel RMS
value - producing a ``(n_files, n_channels)`` array that's small enough to
keep in memory and query freely, with an optional on-disk cache so repeat
analysis runs don't re-filter the raw data.
"""

import os
import time

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt

DEFAULT_BANDPASS_HZ = (1.0, 20.0)  # reasonable default for vessel-scale broadband signatures


def bandpass_energy_image(
    indexed_files: list[tuple[pd.Timestamp, str]],
    fs: float,
    bandpass_hz: tuple[float, float] = DEFAULT_BANDPASS_HZ,
    cache_path: str | None = None,
    progress_every: int = 200,
) -> tuple[np.ndarray, pd.DatetimeIndex]:
    """Bandpassed per-channel RMS energy for every file in ``indexed_files``, one row each.

    ``indexed_files`` is ``[(timestamp, path), ...]`` as returned by
    ``io.index_das_files`` - already time-sorted.

    Returns ``(energies, bin_times)``: ``energies`` is ``(n_files, n_channels)``
    float32, ``bin_times`` the matching per-row timestamps.

    If ``cache_path`` is given and exists with a matching file count, loads
    from cache instead of recomputing. This is a coarse cache key (file
    *count*, not content) - good enough when the underlying file set is
    static, but bust the cache manually (delete the file) if files were
    swapped without changing the count.
    """
    if cache_path and os.path.exists(cache_path):
        cached = np.load(cache_path)
        if len(cached["timestamps"]) == len(indexed_files):
            return cached["energies"], pd.DatetimeIndex(cached["timestamps"])

    sos = butter(4, bandpass_hz, btype="bandpass", fs=fs, output="sos")
    rows = []
    t0 = time.time()
    for i, (_ts, f) in enumerate(indexed_files):
        arr = np.load(f).astype(np.float32)
        filtered = sosfiltfilt(sos, arr, axis=0)
        rows.append(np.mean(filtered**2, axis=0))
        if progress_every and i % progress_every == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed if elapsed else 0
            eta = (len(indexed_files) - i - 1) / rate if rate else float("nan")
            n = len(indexed_files)
            print(f"bandpass_energy_image: {i + 1}/{n} elapsed={elapsed:.0f}s eta={eta:.0f}s")

    energies = np.stack(rows, axis=0)
    bin_times = pd.DatetimeIndex([ts for ts, _ in indexed_files])

    if cache_path:
        os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
        np.savez(cache_path, energies=energies, timestamps=bin_times.values)
    return energies, bin_times
