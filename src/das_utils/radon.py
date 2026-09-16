"""Linear Radon transform (slant-stack) detection of a moving source crossing the fiber.

A single ship crossing the cable is a *moving* source: its signature arrives
at nearby channels at systematically shifted times, tracing a straight line
in the (time, channel) plane at the apparent velocity the geometry of its
pass implies. Neither a per-channel energy threshold nor a fixed-band
STA/LTA ratio uses that shape at all - they would score a real, coherent,
moving wavefront exactly the same as an equal-energy but temporally
scattered coincidence. The linear Radon transform (here as a slant-stack:
sum energy along candidate lines of varying slope and intercept) picks out
exactly that shape, and is what "the ship sends a single wave that crosses
several channels in sequence" corresponds to as a hypothesis to score
directly.

It also gives essentially the same protections ``detect.py`` builds in
separately, for free, from the model itself:

* A fiber-wide, simultaneous artifact (gain step, seismic transient) is a
  *horizontal* line (zero slope, i.e. infinite apparent velocity) - outside
  any realistic ship-crossing slope, so restricting the searched slopes to a
  physically plausible range of apparent velocities already excludes it
  (no separate common-mode-suppression pass needed).
* A single-channel noise spike has no coherent slope at all - stacking
  along any candidate line simply averages it down towards the noise floor,
  the same way summing incoherent noise always underperforms summing a
  genuinely correlated signal.
"""

import numpy as np
import pandas as pd

from das_utils.detect import step_channels


def linear_radon_stack(image: np.ndarray, slownesses: np.ndarray) -> np.ndarray:
    """Slant-stack energy for every candidate line through a local (time, channel) patch.

    ``image`` is ``(n_time, n_channels)`` - a local patch, channel index 0
    assumed to be the leftmost offset (the caller centers this on its target
    channel). ``slownesses`` are trial slopes in time-bins per channel
    (signed - positive and negative cover both directions of travel along
    the fiber; can be fractional, rounded to the nearest bin). For candidate
    slowness ``s`` and intercept row ``t0``, the stacked value is the mean of
    ``image[t0 + round(k*s), k]`` over every channel offset ``k`` whose
    shifted row falls inside the patch (channels that would fall outside are
    simply excluded from that line's average, not zero-padded, so lines near
    the patch edge aren't penalized for it).

    Returns ``(len(slownesses), n_time)``: stacked energy per (slowness,
    intercept) pair, ``-inf`` wherever the line would run off either end of
    the patch for that (slowness, intercept) - a partial line covering fewer
    channels is a *noisier* (higher-variance) statistic, not a fair
    comparison to a fully-supported one, so it's excluded rather than
    quietly averaged over whatever channels happened to fit (which biases
    the max search toward the patch's edges for no physical reason). The
    strongest finite entry is the best-fit, fully-supported line's score and
    location.
    """
    n_time, n_channels = image.shape
    offsets = np.arange(n_channels)
    stack = np.full((len(slownesses), n_time), -np.inf)
    for si, s in enumerate(slownesses):
        shifts = np.round(offsets * s).astype(int)
        for t0 in range(n_time):
            idx = t0 + shifts
            if idx[0] < 0 or idx[-1] >= n_time or idx.min() < 0 or idx.max() >= n_time:
                continue
            stack[si, t0] = image[idx, offsets].mean()
    return stack


def radon_score(image: np.ndarray, slownesses: np.ndarray) -> tuple[float, float, int]:
    """Best-fit line's stacked energy, slowness, and intercept for one patch.

    Returns ``(best_score, best_slowness, best_t0)`` - the strongest
    coherent, moving-source line found in ``image`` (see
    ``linear_radon_stack``).
    """
    stack = linear_radon_stack(image, slownesses)
    si, t0 = np.unravel_index(np.argmax(stack), stack.shape)
    return float(stack[si, t0]), float(slownesses[si]), int(t0)


def _robust_finite_median_mad(values: np.ndarray) -> tuple[float, float]:
    """Median and MAD*1.4826 of the finite entries of ``values``."""
    finite = values[np.isfinite(values)]
    median = float(np.median(finite))
    mad = float(np.median(np.abs(finite - median)) * 1.4826)
    return median, mad


def radon_detect_stream(
    indexed_files: list[tuple[pd.Timestamp, str]],
    fs: float,
    step: int,
    patch_radius: int,
    slownesses: np.ndarray,
    bin_s: float = 0.25,
    bandpass_hz: tuple[float, float] = (1.0, 20.0),
    score_ratio_threshold: float = 6.0,
    progress_every: int = 50,
) -> pd.DataFrame:
    """Radon slant-stack detection over a stream of files, one row per (file, stepped channel).

    For every stepped channel, builds a local ``(n_time_bins, 2*patch_radius+1)``
    energy patch (bandpass power binned to ``bin_s`` resolution) centered on
    it, and slant-stacks it (``linear_radon_stack``). The winning line's
    score is turned into a robust z-score against the *stack matrix's own*
    median/MAD (not the raw patch's - stacked values already average over
    the patch width and so have a tighter spread; scoring against the wrong
    distribution makes pure noise look artificially significant) -
    ``triggered`` is set when that z-score clears ``score_ratio_threshold``.

    ``slownesses`` should span the range of apparent velocities a real
    crossing could plausibly produce for this cable's channel spacing and
    typical vessel speeds - see the module docstring for why deliberately
    excluding near-zero slopes (fiber-wide artifacts) is part of the
    detector's design, not just a parameter choice.

    Returns one row per (file, stepped channel): ``file``, ``timestamp``,
    ``time_s`` (0.0 placeholder, one aggregate row per file - see
    ``detect.common_mode_suppression``), ``channel``, ``score`` (the
    normalized ratio), ``triggered``.
    """
    from das_utils.detect import bandpass_filtered_trimmed

    rows = []
    for i, (ts, f) in enumerate(indexed_files):
        data = np.load(f)
        n_channels = data.shape[1]
        cols = step_channels(n_channels, step)

        filtered, _trim = bandpass_filtered_trimmed(data.astype(np.float64), fs, bandpass_hz)
        n_samples = filtered.shape[0]
        bin_len = max(1, round(bin_s * fs))
        n_bins = n_samples // bin_len
        power = (filtered[: n_bins * bin_len] ** 2).reshape(n_bins, bin_len, n_channels).mean(axis=1)

        for c in cols:
            lo, hi = max(0, c - patch_radius), min(n_channels, c + patch_radius + 1)
            patch = power[:, lo:hi]
            if patch.shape[1] < 3:
                continue
            stack = linear_radon_stack(patch, slownesses)
            if not np.isfinite(stack).any():
                continue
            si, t0 = np.unravel_index(np.nanargmax(np.where(np.isfinite(stack), stack, -np.inf)), stack.shape)
            best_score = float(stack[si, t0])
            # Robust z-score of the best line against the *stack's own* distribution (not the
            # raw patch's), since stacked values are already averages over the patch width and
            # so have a tighter spread than individual samples - comparing against the wrong
            # scale is what makes noise alone look artificially significant (see module tests).
            stack_median, stack_mad = _robust_finite_median_mad(stack)
            ratio = (best_score - stack_median) / max(stack_mad, 1e-12)
            rows.append(
                {
                    "file": f,
                    "timestamp": ts,
                    "time_s": 0.0,
                    "channel": int(c),
                    "score": float(ratio),
                    "triggered": bool(ratio >= score_ratio_threshold),
                }
            )
        if progress_every and i % progress_every == 0:
            print(f"radon_detect_stream: {i + 1}/{len(indexed_files)}")

    return pd.DataFrame(rows)
