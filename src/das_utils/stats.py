"""Per-channel descriptive statistics for a DAS image.

Operates on any ``(n_time, n_channels)`` array (e.g. the output of
``image.build_image``). Everything here is computed per channel (along axis
0), so the result is one row per channel. These are the building blocks
``quality.py`` uses to decide which channels are *persistently* good and
which are dead, saturated, or intermittently noisy.

Key notions:

* **Noise floor** is the robust background level - the *median* of the
  per-window reduced value, not the mean, so a handful of transient events
  (a passing ship, a knock) don't inflate it. A "good noise level" channel
  has a healthy, non-zero, non-extreme floor.
* **Dropout** is a near-silent channel (floor below a global floor). A
  **saturated** channel pegs near the ADC ceiling. Both are unusable and
  easy to flag from the extremes of the per-time distribution.
* **MAD** (median absolute deviation) is a robust spread estimate, used
  instead of std where transient spikes would otherwise dominate.
"""

import numpy as np
import pandas as pd


def _mad(a: np.ndarray, axis: int = 0) -> np.ndarray:
    """Median absolute deviation (scaled to be a robust std estimate)."""
    return np.median(np.abs(a - np.median(a, axis=axis, keepdims=True)), axis=axis) * 1.4826


def channel_stats(
    image: np.ndarray, int16_bits: int | None = None
) -> pd.DataFrame:
    """Per-channel descriptive statistics, one row per channel.

    Columns:
    - ``median``, ``mean``, ``std``, ``mad`` - central tendency and spread
      of the per-window reduced value over time.
    - ``p01``, ``p99`` - robust tail bounds (1st/99th percentile over time).
    - ``dynamic_range`` - ``p99 / max(p01, eps)``: how wide the channel's
      range is; near-constant channels (dead or pegged) score ~1.
    - ``dropout_frac`` - fraction of windows below ``dropout_floor``
      (defaults to 1% of the global median across channels).
    - ``saturation_frac`` - fraction of windows above ``saturation_ceil``
      (defaults to 99% of the int16 full-scale if ``int16_bits`` given, else
      the global p99.9).

    ``int16_bits`` (e.g. 15 for signed int16 full-scale 32767) lets you flag
    ADC saturation when the image was reduced from raw int16 strain.
    """
    img = np.asarray(image, dtype=np.float64)
    # median of per-channel medians, not median of all values - robust to a
    # few extreme (saturated) channels that would inflate the global floor.
    g_median = float(np.median(np.median(img, axis=0)))
    dropout_floor = 0.01 * g_median if g_median > 0 else 0.0
    if int16_bits is not None:
        ceil = float((2 ** (int16_bits - 1)) - 1)
    else:
        ceil = float(np.percentile(img, 99.9))

    p01 = np.percentile(img, 1, axis=0)
    p99 = np.percentile(img, 99, axis=0)
    return pd.DataFrame(
        {
            "median": np.median(img, axis=0),
            "mean": img.mean(axis=0),
            "std": img.std(axis=0),
            "mad": _mad(img, axis=0),
            "p01": p01,
            "p99": p99,
            "dynamic_range": p99 / np.maximum(p01, np.finfo(np.float64).eps),
            "dropout_frac": np.mean(img < dropout_floor, axis=0),
            "saturation_frac": np.mean(img > ceil, axis=0),
        }
    )


def temporal_profile(image: np.ndarray) -> np.ndarray:
    """Per-window mean across all channels - the dataset's global energy in time.

    Useful to spot recording-wide changes (gain drift, an outage, a strong
    regional event) before interpreting per-channel or correlation results.
    """
    return np.asarray(image, dtype=np.float64).mean(axis=1)
