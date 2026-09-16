"""Inter-channel spatial correlation for DAS.

This is distinct from ``correlate.py`` (which correlates DAS against an
*external* reference like AIS). Here we look at how DAS channels relate to
*each other* along the fiber.

The user's hypothesis, which this module is built to test and exploit:
nearby channels are expected to be **strongly correlated** - they share the
same local strain field, so their per-window energy tracks together. When
that correlation drops, something genuinely *local* is happening (a source
acting on just one channel, or a boundary between two regimes), and the
channels are carrying independent information rather than redundancy.

Two scales of result:

* **Lag profile** (``neighbor_correlation``): mean correlation between
  channel *i* and channel *i+lag*, averaged over the whole recording and
  over all channel pairs. Should decay with lag - sharply, if the spatial
  coherence length is short.
* **Sliding window** (``sliding_neighbor_correlation``): the same
  neighbour correlation, but computed per time-window, giving a time series
  of "how correlated are adjacent channels right now." Low stretches are
  the intervals we want (see ``low_correlation_intervals``).

All correlations are Pearson correlations of the per-window reduced image
(axis 0 = time, axis 1 = channels), so this is correlation of *energy
envelopes* across channels, not raw waveform phase.
"""

import numpy as np
import pandas as pd


def _zscore(image: np.ndarray) -> np.ndarray:
    """Per-channel (axis 0) z-score; channels with zero std stay zero."""
    img = np.asarray(image, dtype=np.float64)
    mean = img.mean(axis=0, keepdims=True)
    std = img.std(axis=0, keepdims=True)
    std[std == 0] = 1.0
    return (img - mean) / std


def neighbor_correlation(image: np.ndarray, max_lag: int = 10) -> np.ndarray:
    """Mean Pearson correlation between channel *i* and channel *i+lag*, per lag.

    Returns an array of length ``max_lag``; entry ``k`` is the mean over all
    channel pairs and all time of corr(channel i, channel i+k+1). A
    physically coherent fiber decays with k; a flat ~0 profile means the
    channels are essentially independent (or the reduction wiped out the
    coherence); a profile stuck near 1 means the array is dominated by a
    common-mode signal (gain drift, a regional event) rather than local
    strain.
    """
    img = np.asarray(image, dtype=np.float64)
    z = _zscore(img)
    n_chan = z.shape[1]
    out = np.empty(max_lag, dtype=np.float64)
    for k in range(1, max_lag + 1):
        if k >= n_chan:
            out[k - 1] = np.nan
            continue
        out[k - 1] = np.mean(z[:, :-k] * z[:, k:])
    return out


def correlation_matrix(image: np.ndarray, channels: np.ndarray | None = None) -> np.ndarray:
    """ ``(m, m)`` Pearson correlation matrix between the given channel columns.

    ``channels`` selects a subset of column indices; default is all. Useful
    to inspect coherence structure over a chosen span (e.g. the persistent
    channels from ``quality.select_persistent_channels``) rather than a
    full 3917x3917 matrix.
    """
    img = np.asarray(image, dtype=np.float64)
    cols = img if channels is None else img[:, channels]
    z = _zscore(cols)
    n = z.shape[1]
    # z is z-scored with population std (ddof=0), so (z.T @ z) / n is the
    # Pearson correlation matrix with an exact unit diagonal.
    return (z.T @ z) / z.shape[0] if n > 0 else np.zeros((0, 0))


def measure_correlation(image: np.ndarray) -> np.ndarray:
    """Full ``(n_channels, n_channels)`` Pearson correlation matrix.

    The standalone, self-contained measure of inter-channel redundancy.
    Entry ``[i, j]`` is the Pearson correlation between channel ``i`` and
    channel ``j`` over the whole time axis. A ``TargetSelector`` takes
    this matrix as input to enforce decorrelation between selected
    targets.

    For a single pair without building the full matrix, use
    ``pair_correlation``.
    """
    return correlation_matrix(image)


def pair_correlation(image: np.ndarray, i: int, j: int) -> float:
    """Pearson correlation between two channels (column indices ``i`` and ``j``).

    Cheaper than ``measure_correlation`` when you only need one pair: it
    z-scores the two columns and dots them, never building the full
    ``(n, n)`` matrix. Returns 0.0 for a constant channel (zero std).
    """
    img = np.asarray(image, dtype=np.float64)
    a, b = img[:, i], img[:, j]
    sa, sb = a.std(), b.std()
    if sa == 0 or sb == 0:
        return 0.0
    return float(((a - a.mean()) / sa) @ ((b - b.mean()) / sb) / len(a))


def sliding_neighbor_correlation(
    image: np.ndarray, window: int, lag: int = 1, step: int | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Per-time-window mean neighbour correlation, sliding along time.

    For each window of ``window`` consecutive rows, computes the mean
    correlation between channel *i* and channel *i+lag* over all channel
    pairs, exactly as ``neighbor_correlation`` does for the whole recording.
    Returns ``(corr, centers)``: ``corr`` is the correlation value per
    window and ``centers`` the index of each window's centre row (so the
    caller can map to timestamps via ``bin_times[centers]``).

    ``step`` defaults to ``window`` (non-overlapping); set smaller for an
    overlapping, smoother time series. Windows shorter than ~20 rows give
    noisy correlation estimates - prefer longer windows.
    """
    img = np.asarray(image, dtype=np.float64)
    n_time = img.shape[0]
    if window < 2:
        raise ValueError("window must be >= 2")
    step = window if step is None else step
    corrs, centers = [], []
    for start in range(0, n_time - window + 1, step):
        block = img[start : start + window]
        z = _zscore(block)
        if lag < z.shape[1]:
            corrs.append(np.mean(z[:, :-lag] * z[:, lag:]))
        else:
            corrs.append(np.nan)
        centers.append(start + window // 2)
    return np.asarray(corrs, dtype=np.float64), np.asarray(centers, dtype=np.int64)


def low_correlation_intervals(
    corr: np.ndarray,
    centers: np.ndarray,
    max_corr: float,
    min_length: int = 1,
) -> pd.DataFrame:
    """Contiguous runs where sliding neighbour correlation stays <= ``max_corr``.

    These are the intervals where nearby channels are *not* too correlated -
    i.e. where the per-channel signal is locally informative rather than a
    shared common-mode response. ``corr``/``centers`` come from
    ``sliding_neighbor_correlation``. Returns a frame with ``start_idx``,
    ``end_idx`` (inclusive centre indices), ``length`` (in windows), and
    ``mean_corr``. Runs shorter than ``min_length`` windows are dropped.

    ``centers`` are kept as plain indices; the caller maps them to real
    timestamps with the same ``bin_times`` used to build the image.
    """
    below = corr <= max_corr
    runs = []
    i = 0
    n = len(corr)
    while i < n:
        if below[i]:
            j = i
            while j < n and below[j]:
                j += 1
            if j - i >= min_length:
                runs.append(
                    {
                        "start_idx": int(centers[i]),
                        "end_idx": int(centers[j - 1]),
                        "length": int(j - i),
                        "mean_corr": float(np.mean(corr[i:j])),
                    }
                )
            i = j
        else:
            i += 1
    return pd.DataFrame(runs)
