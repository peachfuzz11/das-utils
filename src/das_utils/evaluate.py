"""Match detector output against AIS-derived ground truth and score it.

Ground-truth "crossings" are built the same way ``correlate.py`` picks
reference events: one closest-approach point per vessel per pass near the
route. Detections (e.g. clustered output of ``detect.cluster_detections``)
are matched to crossings by nearest neighbour in space (channel position)
and time, within a configurable ``max_range_m``/``max_time_s`` - swept
across several ranges to report the expected easier-at-larger-range
tradeoff.
"""

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from das_utils.correlate import to_epoch_seconds


def ground_truth_crossings(
    ais_df: pd.DataFrame,
    ids: np.ndarray,
    lats: np.ndarray,
    lons: np.ndarray,
    mmsi_col: str = "MMSI",
    lat_col: str = "Latitude",
    lon_col: str = "Longitude",
    time_col: str = "Timestamp",
    dist_col: str = "DistToRoute_m",
) -> pd.DataFrame:
    """One row per vessel's closest approach to the route (its ground-truth crossing).

    ``ais_df`` should already be filtered to genuinely transiting vessels
    near the route (e.g. ``ais.filter_near_positions`` output). For each
    ``MMSI``, keeps the row with the smallest ``dist_col`` and assigns it the
    nearest route channel (``channel``). One crossing per vessel per input
    frame - pass one frame per time-block (e.g. per train/test split) to get
    per-block crossings.

    ``channel`` is the 0-based *position* in the ``(ids, lats, lons)`` arrays
    (i.e. ``lats[channel]``/``lons[channel]``) - the same "array index as
    channel id" convention ``detect.py`` uses for its own ``channel`` column
    (a raw data-array column index), not the literal value in ``ids``. This
    keeps ``match_detections`` comparing like with like even when ``ids``
    holds arbitrary, non-zero-based channel ids (as the real cable position
    table does - see AGENTS.md on the position/recorded-channel-count
    mismatch). ``ids`` itself is unused here for that reason; it stays a
    parameter only for symmetry with the array triple callers already have.
    """
    from das_utils.geo import build_nearest_tree, nearest_index

    del ids  # kept for signature symmetry with (ids, lats, lons); see docstring
    tree, ref_lat, ref_lon = build_nearest_tree(lats, lons)
    idx_closest = ais_df.groupby(mmsi_col)[dist_col].idxmin()
    crossings = ais_df.loc[idx_closest].reset_index(drop=True)

    _, nearest = nearest_index(
        tree, ref_lat, ref_lon, crossings[lat_col].to_numpy(), crossings[lon_col].to_numpy()
    )
    crossings = crossings.copy()
    crossings["channel"] = nearest
    crossings["timestamp"] = crossings[time_col]
    return crossings[[mmsi_col, "timestamp", "channel", lat_col, lon_col, dist_col]]


def match_detections(
    detections: pd.DataFrame,
    ground_truth: pd.DataFrame,
    positions_lats: np.ndarray,
    positions_lons: np.ndarray,
    positions_ids: np.ndarray,
    max_range_m: float,
    max_time_s: float = 300.0,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Match ``detections`` to ``ground_truth`` crossings within range/time tolerance.

    Both inputs need a ``channel`` and ``timestamp`` column, where
    ``channel`` is a 0-based index into ``(positions_lats, positions_lons)``
    (the convention ``detect.py`` and ``ground_truth_crossings`` both use -
    see their docstrings) - ``positions_ids`` is accepted only for signature
    symmetry with callers that have the full ``(ids, lats, lons)`` triple and
    is not used to look up positions. Matching is greedy nearest-neighbour in
    physical distance (via each channel's lat/lon) among candidate pairs
    already within ``max_time_s``, one-to-one (each ground-truth crossing
    claims at most one detection and vice versa).

    Returns ``(matched, false_positives, false_negatives)``: ``matched`` has
    one row per matched pair with a ``dist_m`` column; ``false_positives`` is
    the unmatched detections; ``false_negatives`` is the unmatched
    ground-truth crossings.
    """
    del positions_ids  # kept for signature symmetry; see docstring

    det = detections.reset_index(drop=True).copy()
    gt = ground_truth.reset_index(drop=True).copy()
    det_t = to_epoch_seconds(pd.DatetimeIndex(det["timestamp"]))
    gt_t = to_epoch_seconds(pd.DatetimeIndex(gt["timestamp"]))

    from das_utils.geo import local_xy

    ref_lat = float(np.mean(positions_lats))
    ref_lon = float(np.mean(positions_lons))
    det_channels = det["channel"].to_numpy(dtype=np.int64)
    gt_channels = gt["channel"].to_numpy(dtype=np.int64)
    det_lats = positions_lats[det_channels]
    det_lons = positions_lons[det_channels]
    gt_lats = positions_lats[gt_channels]
    gt_lons = positions_lons[gt_channels]
    det_xy = np.column_stack(local_xy(det_lats, det_lons, ref_lat, ref_lon)) if len(det) else np.empty((0, 2))
    gt_xy = np.column_stack(local_xy(gt_lats, gt_lons, ref_lat, ref_lon)) if len(gt) else np.empty((0, 2))

    pairs = []
    if len(det_xy) and len(gt_xy):
        tree = cKDTree(gt_xy)
        dists, gt_idx = tree.query(det_xy, k=1, distance_upper_bound=max_range_m)
        for di, (d, gi) in enumerate(zip(dists, gt_idx, strict=True)):
            if not np.isfinite(d) or gi >= len(gt_xy):
                continue
            if abs(det_t[di] - gt_t[gi]) > max_time_s:
                continue
            pairs.append((di, int(gi), float(d)))

    pairs.sort(key=lambda p: p[2])
    used_det, used_gt = set(), set()
    matched_rows = []
    for di, gi, d in pairs:
        if di in used_det or gi in used_gt:
            continue
        used_det.add(di)
        used_gt.add(gi)
        row = {**det.iloc[di].to_dict(), "gt_channel": gt["channel"].iloc[gi], "dist_m": d}
        matched_rows.append(row)

    matched = pd.DataFrame(matched_rows)
    false_positives = det.drop(index=list(used_det))
    false_negatives = gt.drop(index=list(used_gt))
    return matched, false_positives, false_negatives


def compute_metrics(
    matched: pd.DataFrame, false_positives: pd.DataFrame, false_negatives: pd.DataFrame
) -> dict:
    """Accuracy/precision/recall/F1 from matched/FP/FN detection-vs-ground-truth tables.

    There are no true negatives in this event-detection setting (a "correct
    absence" isn't an enumerable unit), so ``accuracy`` here is
    ``tp / (tp + fp + fn)`` - the fraction of all involved events (detected
    or true) that were correctly matched - rather than the classic
    ``(tp+tn)/(tp+tn+fp+fn)``, which is undefined without negatives.
    """
    tp, fp, fn = len(matched), len(false_positives), len(false_negatives)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    accuracy = tp / (tp + fp + fn) if (tp + fp + fn) else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def evaluate_at_ranges(
    detections: pd.DataFrame,
    ground_truth: pd.DataFrame,
    positions_lats: np.ndarray,
    positions_lons: np.ndarray,
    positions_ids: np.ndarray,
    ranges_m: tuple[float, ...] = (250.0, 500.0, 1000.0),
    max_time_s: float = 300.0,
) -> pd.DataFrame:
    """Sweep ``match_detections`` + ``compute_metrics`` across ``ranges_m``.

    Returns one row per range with ``range_m`` plus every ``compute_metrics``
    field - larger ranges should show non-decreasing recall/precision/F1,
    since the matching tolerance only loosens.
    """
    rows = []
    for r in ranges_m:
        matched, fp, fn = match_detections(
            detections, ground_truth, positions_lats, positions_lons, positions_ids, r, max_time_s
        )
        rows.append({"range_m": r, **compute_metrics(matched, fp, fn)})
    return pd.DataFrame(rows)
