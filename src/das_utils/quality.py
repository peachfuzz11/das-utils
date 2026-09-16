"""Channel quality: which DAS channels are *persistently* good and which are broken.

A DAS array routinely has bad channels: dead (dropouts), pegged (saturated),
or intermittently noisy. The goal here is the opposite of finding loud
events - it's finding channels whose **background** behaviour is stable and
healthy over the whole recording, so they can be trusted as a clean
reference set (and so genuinely localised signals can be told apart from
array-wide artefacts).

Two things together define a "good" channel:

* **A healthy noise floor** - non-zero (not dead) and not extreme (not
  saturated/pegged), sitting within a reasonable band of the array's typical
  background. We use the *median* per-window reduced value as the floor so
  transient events don't inflate it.
* **Persistence** - the floor stays put over time. A channel whose energy
  wanders or blinks on and off is unreliable even if its median looks fine.
  We score persistence as the fraction of time-windows whose value falls
  within a factor-of-two band of the channel's own median (high = stable).

``select_persistent_channels`` combines both: keep channels whose floor is
in a healthy band *and* whose persistence score clears a threshold, then
rank them. Defaults are deliberately permissive - tune to your deployment.
"""

import numpy as np
import pandas as pd

from das_utils.stats import channel_stats


def persistence_score(image: np.ndarray, band: float = 2.0) -> np.ndarray:
    """Fraction of time-windows within ``band`` x of each channel's own median.

    A channel that holds a stable background scores ~1; one that blinks,
    drifts, or spikes sporadically scores lower. ``band=2`` means a window
    counts as "in-band" when its value is within [0.5x, 2x] of the channel
    median - a generous tolerance that still rejects large excursions.
    """
    img = np.asarray(image, dtype=np.float64)
    med = np.median(img, axis=0, keepdims=True)
    lo = med / band
    hi = med * band
    return np.mean((img >= lo) & (img <= hi), axis=0)


def channel_quality(
    image: np.ndarray, int16_bits: int | None = None, persistence_band: float = 2.0
) -> pd.DataFrame:
    """One row per channel with noise-floor and persistence diagnostics.

    Adds to ``channel_stats``:
    - ``persistence`` - the persistence score (see ``persistence_score``).
    - ``noise_floor`` - the per-channel median (robust background level).
    - ``good`` - boolean: healthy floor band AND persistence >= threshold AND
      not dropped/saturated (the default selection; relax by ignoring it).
    """
    stats = channel_stats(image, int16_bits=int16_bits).copy()
    stats["persistence"] = persistence_score(image, band=persistence_band)
    stats["noise_floor"] = stats["median"]
    stats["good"] = (
        (stats["dropout_frac"] <= 0.0)
        & (stats["saturation_frac"] <= 0.0)
        & (stats["persistence"] >= 0.9)
    )
    return stats


def measure_noise(image: np.ndarray, int16_bits: int | None = None) -> np.ndarray:
    """Per-channel noise quality score in [0, 1]. Higher = better.

    A single, self-contained measure of how trustworthy a channel's
    background noise is. Combines:

    * **Persistence** (temporal stability of the noise floor - see
      ``persistence_score``): a channel whose energy wanders or blinks
      is unreliable even if its median level looks fine.
    * **Health** (``(1 - dropout) * (1 - saturation)``): dead channels
      (floor near zero) and pegged channels (floor at the ADC ceiling)
      are penalized toward zero.

    The product naturally zeroes out broken channels and down-weights
    intermittents, giving one number per channel that the
    ``TargetSelector`` can rank by. Pass ``int16_bits=16`` when the image
    was reduced from raw int16 strain, so saturation is flagged against
    the correct full-scale.
    """
    q = channel_quality(image, int16_bits=int16_bits)
    health = (1.0 - q["dropout_frac"].to_numpy()) * (1.0 - q["saturation_frac"].to_numpy())
    return q["persistence"].to_numpy() * health


def select_persistent_channels(
    image: np.ndarray,
    floor_quantiles: tuple[float, float] = (0.05, 0.95),
    min_persistence: float = 0.9,
    max_dropout: float = 0.0,
    max_saturation: float = 0.0,
    int16_bits: int | None = None,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Indices of channels with a healthy, persistent noise floor, ranked by persistence.

    A channel is kept when its noise floor (per-channel median) lies within
    ``floor_quantiles`` of the array-wide floor distribution, its persistence
    score is >= ``min_persistence``, and it is neither dropped out nor
    saturated. Returns ``(indices, quality_df)`` where ``indices`` are the
    surviving channel column indices sorted by persistence (best first);
    ``quality_df`` is the full per-channel table for inspection.

    The floor-quantile band rejects dead channels (floor near zero, below the
    low quantile) and hot/saturated ones (floor above the high quantile),
    while the persistence cut rejects intermittents that happen to have a
    plausible median. Both are needed - a channel can pass one and fail the
    other.
    """
    q = channel_quality(image, int16_bits=int16_bits)
    lo_q, hi_q = floor_quantiles
    floor_lo, floor_hi = np.quantile(q["noise_floor"], lo_q), np.quantile(q["noise_floor"], hi_q)

    keep = (
        (q["noise_floor"] >= floor_lo)
        & (q["noise_floor"] <= floor_hi)
        & (q["persistence"] >= min_persistence)
        & (q["dropout_frac"] <= max_dropout)
        & (q["saturation_frac"] <= max_saturation)
    )
    q["keep"] = keep
    idx = np.where(keep)[0]
    idx = idx[np.argsort(q["persistence"].to_numpy()[idx])[::-1]]
    return idx, q
