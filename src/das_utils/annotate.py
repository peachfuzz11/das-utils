"""Build a per-file, per-channel "distance to nearest ship" ground-truth annotation.

This curates the dataset once, independently of any detector: for every DAS
file, at the file's *center* time, find every AIS vessel reporting nearby and
record each channel's distance to the closest one. Thresholding that
distance at a chosen range gives a boolean ground-truth label
(``distance_m <= range_m``) per (file, channel) - which is what
``labels_for_range``/``benchmark_detections`` in this module use to score a
detector's own per-(file, channel) predictions across several range
thresholds in one pass over the annotation.

Doing this once and caching it to disk (``save_annotations``/
``load_annotations``) means the (potentially slow, AIS-file-scanning) ground
truth only has to be built once; any number of detectors or range sweeps
after that just read the cached JSON.
"""

import json

import numpy as np
import pandas as pd

from das_utils.geo import build_nearest_tree, local_xy

NO_SHIP_SENTINEL_M = 999_999.0  # "no AIS vessel reporting nearby" - always beyond any real range_m


def nearest_ship_distance_m(
    channel_lats: np.ndarray,
    channel_lons: np.ndarray,
    ship_lats: np.ndarray,
    ship_lons: np.ndarray,
) -> np.ndarray:
    """Per-channel distance (m) to the closest of one or more ship positions.

    Returns an array the length of ``channel_lats``, filled with
    ``NO_SHIP_SENTINEL_M`` if no ship positions are given.
    """
    n_channels = len(channel_lats)
    if len(ship_lats) == 0:
        return np.full(n_channels, NO_SHIP_SENTINEL_M)

    tree, ref_lat, ref_lon = build_nearest_tree(np.asarray(ship_lats), np.asarray(ship_lons))
    x, y = local_xy(channel_lats, channel_lons, ref_lat, ref_lon)
    dist, _ = tree.query(np.column_stack([x, y]))
    return dist


def _ship_positions_at(
    ais_near: pd.DataFrame,
    center_time: pd.Timestamp,
    time_tol_s: float,
    mmsi_col: str,
    lat_col: str,
    lon_col: str,
    time_col: str,
) -> tuple[np.ndarray, np.ndarray]:
    """One (lat, lon) per vessel reporting within ``time_tol_s`` of ``center_time``.

    Multiple reports from the same vessel in the window collapse to its
    report closest in time to ``center_time`` - the best available position
    estimate for that instant.
    """
    lo, hi = center_time - pd.Timedelta(seconds=time_tol_s), center_time + pd.Timedelta(seconds=time_tol_s)
    window = ais_near[ais_near[time_col].between(lo, hi)]
    if window.empty:
        return np.array([]), np.array([])

    window = window.copy()
    window["_dt"] = (window[time_col] - center_time).abs()
    closest = window.loc[window.groupby(mmsi_col)["_dt"].idxmin()]
    return closest[lat_col].to_numpy(), closest[lon_col].to_numpy()


def annotate_file(
    center_time: pd.Timestamp,
    ais_near: pd.DataFrame,
    channel_lats: np.ndarray,
    channel_lons: np.ndarray,
    time_tol_s: float = 30.0,
    mmsi_col: str = "MMSI",
    lat_col: str = "Latitude",
    lon_col: str = "Longitude",
    time_col: str = "Timestamp",
) -> np.ndarray:
    """Per-channel distance (m) to the nearest ship at ``center_time``, for one file.

    ``ais_near`` should already be filtered to vessels near the route (e.g.
    ``ais.filter_near_positions`` output) covering at least
    ``[center_time - time_tol_s, center_time + time_tol_s]``.
    """
    ship_lats, ship_lons = _ship_positions_at(
        ais_near, center_time, time_tol_s, mmsi_col, lat_col, lon_col, time_col
    )
    return nearest_ship_distance_m(channel_lats, channel_lons, ship_lats, ship_lons)


def build_annotations(
    indexed_files: list[tuple[pd.Timestamp, str]],
    ais_near: pd.DataFrame,
    ids: np.ndarray,
    lats: np.ndarray,
    lons: np.ndarray,
    file_duration_s: float = 10.0,
    time_tol_s: float = 30.0,
    progress_every: int = 200,
) -> dict:
    """Per-channel nearest-ship distance for every file's center time.

    ``ais_near`` is one dataframe already filtered near the route (e.g. via
    ``ais.filter_near_positions``) spanning the full time range of
    ``indexed_files`` - pull it once for the whole batch rather than
    per-file, since ``annotate_file`` only needs a small time slice of it
    each call.

    Returns ``{"channel_ids": [...], "files": {path: {"timestamp": iso,
    "distances_m": [...]}}}`` - one distance per entry of ``ids``, in the
    same order.
    """
    files_out = {}
    for i, (ts, path) in enumerate(indexed_files):
        center_time = ts + pd.Timedelta(seconds=file_duration_s / 2)
        distances = annotate_file(center_time, ais_near, lats, lons, time_tol_s=time_tol_s)
        files_out[path] = {
            "timestamp": center_time.isoformat(),
            "distances_m": np.round(distances, 1).tolist(),
        }
        if progress_every and i % progress_every == 0:
            print(f"build_annotations: {i + 1}/{len(indexed_files)}")

    return {"channel_ids": np.asarray(ids).tolist(), "files": files_out}


def save_annotations(annotations: dict, path: str) -> None:
    """Write annotations (from ``build_annotations``) to a JSON file."""
    with open(path, "w") as f:
        json.dump(annotations, f)


def load_annotations(path: str) -> dict:
    """Read annotations back from a JSON file written by ``save_annotations``."""
    with open(path) as f:
        return json.load(f)


def labels_for_range(annotations: dict, range_m: float) -> dict[str, np.ndarray]:
    """Boolean "ship within range_m" label per channel, for every file.

    Returns ``{file_path: bool_array}``, one array per file (length
    ``len(annotations["channel_ids"])``), thresholding each file's stored
    ``distances_m``.
    """
    return {
        path: np.asarray(entry["distances_m"]) <= range_m
        for path, entry in annotations["files"].items()
    }


def benchmark_detections(
    detections: pd.DataFrame,
    annotations: dict,
    ranges_m: tuple[float, ...] = (250.0, 500.0, 1000.0),
) -> pd.DataFrame:
    """Per-channel classification accuracy/precision/recall/F1, swept across ``ranges_m``.

    ``detections`` must have one row per (file, channel) *actually scored* by
    the detector (e.g. ``detect.detect_stream`` output, or a dense per-channel
    baseline) with columns ``file``, ``channel``, ``triggered``. Only
    channels the detector scored are counted - this measures the detector's
    accuracy on the channels it examined, not on the full array. Channels are
    matched to ``annotations["channel_ids"]`` by position (index into that
    list), the same convention ``detect.py``/``evaluate.py`` use throughout
    this package.

    For each range, aggregates a confusion matrix over every scored
    (file, channel) pair across every file with both a detection and an
    annotation, then returns one row of metrics per range - true negatives
    are well-defined here (unlike ``evaluate.compute_metrics``), so
    ``accuracy`` is the standard ``(tp+tn)/(tp+tn+fp+fn)``.
    """
    rows = []
    for range_m in ranges_m:
        labels = labels_for_range(annotations, range_m)
        tp = fp = tn = fn = 0
        for path, group in detections.groupby("file"):
            if path not in labels:
                continue
            truth = labels[path]
            channels = group["channel"].to_numpy()
            pred = group["triggered"].to_numpy(dtype=bool)
            actual = truth[channels]
            tp += int(np.sum(pred & actual))
            fp += int(np.sum(pred & ~actual))
            fn += int(np.sum(~pred & actual))
            tn += int(np.sum(~pred & ~actual))

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) else 0.0
        rows.append(
            {
                "range_m": range_m,
                "tp": tp,
                "fp": fp,
                "tn": tn,
                "fn": fn,
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )
    return pd.DataFrame(rows)
