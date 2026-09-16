"""Select which DAS channels are worth processing, and their context windows.

The downstream task: process a *target* channel together with a window of
neighbouring channels for spatial context. Processing every channel on the
fiber is wasteful when nearby channels are strongly correlated - most carry
redundant copies of the same local strain signal. This module picks a sparse
set of **targets** that each carry unique information, so you process far
fewer targets while still giving each one a full context window.

Three pieces, cleanly separated:

* **Noise** (``quality.measure_noise``): a per-channel quality score. Higher
  = more persistent, healthier noise. This is the *ranking* - which channels
  are worth processing at all.
* **Correlation** (``channels.measure_correlation``): a full ``(n, n)``
  Pearson matrix, or ``channels.pair_correlation`` for a single pair. This is
  the *redundancy* measure - how much two channels overlap.
* **``TargetSelector``**: the class that combines the two. It greedily picks
  targets that are high-noise, spaced (``min_sep`` <= gap <= ``max_sep``),
  and decorrelated (``|corr|`` <= ``max_corr``), then gives each target its
  context window.

The separation matters: the noise and correlation functions are independent,
testable, and reusable on their own. The selector is the only thing that
knows how to combine them.

Usage::

    noise = measure_noise(image)
    corr = measure_correlation(image)
    selector = TargetSelector(noise, corr)
    for target, cols in selector.plan(radius=50, min_sep=15, max_sep=100, max_corr=0.7):
        block = image[:, cols]
        process(block, center=np.where(cols == target)[0][0])
"""

import numpy as np


def greedy_decorrelated(
    corr: np.ndarray,
    candidates: np.ndarray,
    max_corr: float = 0.9,
    min_sep: int = 0,
) -> np.ndarray:
    """Greedy selection of decorrelated, spaced candidates from a correlation matrix.

    Iterates ``candidates`` in order (best quality first). A candidate is
    kept only if it is at least ``min_sep`` channel indices from every
    already-kept target **and** has ``|corr|`` <= ``max_corr`` with each of
    them. Returns the kept candidates' global channel indices, in selection
    order.

    ``corr`` is the ``(m, m)`` Pearson matrix among the ``m`` candidates
    (row ``i`` corresponds to ``candidates[i]``), e.g. from
    ``channels.correlation_matrix(image, channels=candidates)``.
    """
    selected: list[int] = []
    sel_pos: list[int] = []
    for ci, c in enumerate(int(x) for x in candidates):
        if any(abs(c - s) < min_sep for s in selected):
            continue
        if any(abs(corr[ci, sp]) > max_corr for sp in sel_pos):
            continue
        selected.append(c)
        sel_pos.append(ci)
    return np.asarray(selected, dtype=np.int64)


def context_window(
    target: int,
    radius: int,
    n_channels: int,
    persistent_idx: np.ndarray | None = None,
) -> np.ndarray:
    """Channels in ``[target-radius, target+radius]``, optionally filtered to ``persistent_idx``.

    The full spatial context for processing ``target`` - every channel
    within ``radius`` (inclusive), in ascending index order. If
    ``persistent_idx`` is given, channels not in that set are dropped
    (except the target itself, which is always retained). For the
    noise-score-based equivalent, use ``TargetSelector.context_window``.
    """
    lo, hi = max(0, target - radius), min(n_channels, target + radius + 1)
    cols = np.arange(lo, hi, dtype=np.int64)
    if persistent_idx is not None:
        keep = np.isin(cols, persistent_idx) | (cols == target)
        cols = cols[keep]
    return cols


class TargetSelector:
    """Combines noise quality and inter-channel correlation to select processing targets.

    Takes two precomputed inputs - a per-channel noise score (higher =
    better, from ``quality.measure_noise``) and a full ``(n, n)``
    correlation matrix (from ``channels.measure_correlation``) - and
    greedily selects targets that are:

    * **High-noise** - candidates are ranked by noise score, best first.
      A ``min_noise`` floor drops broken/erratic channels entirely.
    * **Spaced** - consecutive targets are at least ``min_sep`` channels
      apart (redundancy floor) and at most ``max_sep`` apart (coverage
      guarantee - no fiber span left unprocessed).
    * **Decorrelated** - ``|corr|`` between any two targets <= ``max_corr``
      (adapts to regions where the coherence length exceeds ``min_sep``).

    The two spatial guards compose: ``min_sep`` is a cheap floor;
    ``max_corr`` catches residual redundancy ``min_sep`` can't. ``max_sep``
    fills any gaps left by the decorrelation filter so no part of the
    fiber goes uncovered.
    """

    def __init__(
        self,
        noise: np.ndarray,
        correlation: np.ndarray,
        candidates: np.ndarray | None = None,
    ):
        """``noise``: ``(n_channels,)`` quality scores, higher = better.

        ``correlation``: ``(n_channels, n_channels)`` Pearson matrix, e.g.
        from ``channels.measure_correlation``.

        ``candidates``: column indices to consider (default: all). When
        given, only these channels can become targets; the rest are still
        available for context windows.
        """
        self.noise = np.asarray(noise, dtype=np.float64)
        self.corr = np.asarray(correlation, dtype=np.float64)
        self.n = len(self.noise)
        if candidates is None:
            candidates = np.arange(self.n)
        self.candidates = np.asarray(candidates, dtype=np.int64)
        # rank candidates by noise score, best first
        self.ranked = self.candidates[np.argsort(-self.noise[self.candidates])]

    def noise_of(self, i: int) -> float:
        """Noise quality score of one channel."""
        return float(self.noise[i])

    def correlation_between(self, i: int, j: int) -> float:
        """Pearson correlation between two channels (column indices)."""
        return float(self.corr[i, j])

    def select(
        self,
        min_sep: int = 0,
        max_sep: int | None = None,
        max_corr: float = 0.9,
        max_count: int | None = None,
        min_noise: float = 0.0,
    ) -> np.ndarray:
        """Indices of selected target channels, sorted ascending.

        Greedy decorrelation pass (ranked by noise), then gap-filling if
        ``max_sep`` is given. ``min_noise`` drops candidates below that
        score before selection. ``max_count`` caps the count, keeping the
        highest-noise targets.
        """
        if max_sep is not None and max_sep < min_sep:
            raise ValueError(f"max_sep ({max_sep}) must be >= min_sep ({min_sep})")

        valid = self.ranked[self.noise[self.ranked] >= min_noise]
        if len(valid) == 0:
            return np.array([], dtype=np.int64)

        corr_sub = self.corr[np.ix_(valid, valid)]
        selected = greedy_decorrelated(corr_sub, valid, max_corr=max_corr, min_sep=min_sep)
        selected = np.sort(selected)

        if max_sep is not None:
            selected = self._fill_gaps(selected, max_sep, min_noise, min_sep)

        if max_count is not None and len(selected) > max_count:
            order = np.argsort(-self.noise[selected])
            selected = np.sort(selected[order[:max_count]])
        return selected

    def _fill_gaps(
        self, selected: np.ndarray, max_sep: int, min_noise: float, min_sep: int
    ) -> np.ndarray:
        """Insert the highest-noise channel near the midpoint of any gap > max_sep.

        Repeats until every consecutive gap <= max_sep, so no fiber span
        is left uncovered. Gap-fill channels respect ``min_sep`` from their
        neighbours but bypass the ``max_corr`` filter - coverage matters
        more than decorrelation there.
        """
        if len(selected) < 2:
            return selected
        selected = sorted(int(x) for x in selected)
        i = 0
        while i < len(selected) - 1:
            gap = selected[i + 1] - selected[i]
            if gap > max_sep:
                lo = selected[i] + min_sep
                hi = selected[i + 1] - min_sep
                if lo > hi:
                    i += 1
                    continue
                cands = np.arange(lo, hi + 1, dtype=np.int64)
                cands = cands[self.noise[cands] >= min_noise]
                if len(cands) > 0:
                    mid = (selected[i] + selected[i + 1]) // 2
                    # highest noise first, closest to midpoint as tiebreaker
                    order = np.lexsort((np.abs(cands - mid), -self.noise[cands]))
                    best = int(cands[order[0]])
                    selected.insert(i + 1, best)
                    continue  # recheck the lower sub-gap
            i += 1
        return np.array(selected, dtype=np.int64)

    def context_window(self, target: int, radius: int, min_noise: float = 0.0) -> np.ndarray:
        """Channels within +/- radius of ``target``, filtered to usable channels.

        Every channel within ``radius`` (inclusive) whose noise score >=
        ``min_noise``, plus the target itself (always retained). In
        ascending index order.
        """
        lo, hi = max(0, target - radius), min(self.n, target + radius + 1)
        cols = np.arange(lo, hi, dtype=np.int64)
        if min_noise > 0:
            cols = cols[(self.noise[cols] >= min_noise) | (cols == target)]
        return cols

    def plan(
        self,
        radius: int,
        min_sep: int = 0,
        max_sep: int | None = None,
        max_corr: float = 0.9,
        max_count: int | None = None,
        min_noise: float = 0.0,
    ) -> list[tuple[int, np.ndarray]]:
        """Full processing plan: ``(target, context_columns)`` for each target.

        One call gives everything to iterate over: the selected targets
        and, for each, its quality-filtered context window::

            for target, cols in selector.plan(radius=50, min_sep=15, max_sep=100, max_corr=0.7):
                block = image[:, cols]
                process(block, center=np.where(cols == target)[0][0])
        """
        targets = self.select(min_sep, max_sep, max_corr, max_count, min_noise)
        return [(int(t), self.context_window(int(t), radius, min_noise)) for t in targets]
