"""End-to-end ship-detection evaluation: train/test split, offset correction, metrics by range.

Usage: uv run python scripts/run_ship_detection_eval.py

Train split: das_08_12/ (used to derive the cable's geo-registration offset
via the AIS-DAS correlation recipe). Test split: das_12_16/ + das_16_22/
(held out; final reported metrics come only from here). Both splits are
scored with a fixed-threshold RMS baseline and the improved, contextual
stepped detector (das_utils.detect), and metrics are reported across several
match-range thresholds so the "easier at larger range" tradeoff is visible.
"""

import numpy as np
import pandas as pd

from das_utils import (
    ais,
    correlate,
    detect,
    energy,
    evaluate,
    geo,
    image,
    io,
    quality,
)

DATA_DIR = "/home/ph/data/das"
DATE = "2025-11-22"
TRAIN_DIRS = [f"{DATA_DIR}/das_08_12"]
TEST_DIRS = [f"{DATA_DIR}/das_12_16", f"{DATA_DIR}/das_16_22"]
CABLE_JSON = f"{DATA_DIR}/das_cable.json"
AIS_CSV = f"{DATA_DIR}/aisdk-2025-11-22.csv"
RANGES_M = (250.0, 500.0, 1000.0)
STEP = 50
MAX_TIME_S = 300.0
CACHE_DIR = "/tmp/das_eval_cache"
# Cap on files per test directory: the full test split is ~3600 files (~10 hours) of raw
# strain data; running the full amount is possible with this same script (raise/remove this
# cap) but costs proportionally more wall-clock time in a single session.
MAX_TEST_FILES_PER_DIR = 400


def build_ground_truth(files, ids, lats, lons, dx_m):
    start, end = files[0][0], files[-1][0] + pd.Timedelta(seconds=10)
    raw_ais = ais.extract_time_window(AIS_CSV, start, end)
    near = ais.filter_near_positions(raw_ais, ids, lats, lons, max_dist_m=2000.0)
    near = near[near["SOG"] > 1.0]  # genuinely transiting, not loitering (AGENTS.md)
    return evaluate.ground_truth_crossings(near, ids, lats, lons)


def baseline_detect(files, fs, cache_path=None):
    """Simple fixed-threshold RMS energy baseline: flag any channel > k std above its own mean."""
    img, bin_times = image.build_image(
        files, reducer="rms", fs=fs, bandpass_hz=(1.0, 20.0), cache_path=cache_path
    )
    mean, std = img.mean(axis=0, keepdims=True), img.std(axis=0, keepdims=True)
    std[std == 0] = 1.0
    z = (img - mean) / std
    rows = []
    for wi in range(z.shape[0]):
        hits = np.where(z[wi] > 4.0)[0]
        for c in hits:
            rows.append({"timestamp": bin_times[wi], "channel": int(c), "score": float(z[wi, c])})
    return pd.DataFrame(rows)


def main():
    import os

    os.makedirs(CACHE_DIR, exist_ok=True)
    cable = io.load_channel_positions(CABLE_JSON)
    ids, lats, lons = io.positions_arrays(cable)
    fs = cable["metadata"]["fs"]
    dx_m = cable["metadata"]["dx"]

    train_files = io.index_das_files(TRAIN_DIRS, DATE)
    test_files = io.index_das_files(TEST_DIRS, DATE)
    if MAX_TEST_FILES_PER_DIR is not None:
        by_dir: dict[str, list] = {}
        for ts, f in test_files:
            by_dir.setdefault(os.path.dirname(f), []).append((ts, f))
        test_files = sorted(
            (row for files in by_dir.values() for row in files[:MAX_TEST_FILES_PER_DIR]),
            key=lambda p: p[0],
        )
    print(f"train files: {len(train_files)}, test files: {len(test_files)}")

    # 1. Derive cable offset correction from the train split.
    train_energies, train_bin_times = energy.bandpass_energy_image(
        train_files, fs, cache_path=f"{CACHE_DIR}/train_energy.npz"
    )
    train_start, train_end = train_files[0][0], train_files[-1][0] + pd.Timedelta(seconds=10)
    train_ais_raw = ais.extract_time_window(AIS_CSV, train_start, train_end)
    train_ais_near = ais.filter_near_positions(train_ais_raw, ids, lats, lons, max_dist_m=2000.0)
    train_ais_near = train_ais_near[train_ais_near["SOG"] > 1.0]
    candidates = train_ais_near.loc[train_ais_near.groupby("MMSI")["DistToRoute_m"].idxmin()]
    dlat_m, dlon_m, per_candidate = correlate.estimate_translation_offset(
        candidates, train_energies, train_bin_times, lats, lons, channel_spacing_m=dx_m
    )
    print(f"derived offset: dlat_m={dlat_m:.1f} dlon_m={dlon_m:.1f} (n={len(per_candidate)})")
    sensitivity = correlate.margin_sensitivity(
        candidates, train_energies, train_bin_times, lats, lons,
        channel_spacing_m=dx_m, margins_m=[300, 700, 1200, 1800],
    )
    print("margin sensitivity:\n", sensitivity)

    corrected = geo.apply_translation(cable["positions"], dlat_m, dlon_m)
    corrected_ids, corrected_lats, corrected_lons = io.positions_arrays(
        {"positions": corrected, "metadata": cable["metadata"]}
    )

    # 2. Ground truth (test split).
    test_gt = build_ground_truth(test_files, corrected_ids, corrected_lats, corrected_lons, dx_m)
    print(f"test ground-truth crossings: {len(test_gt)}")

    # 3. Persistent channels (diagnostic) and per-channel temporal baseline (train split),
    # reused for the improved detector. train_energies is already bandpass(1-20Hz)+RMS per
    # file/channel - the same reduction image.build_image(reducer="rms", bandpass_hz=(1,20))
    # would produce - so reuse it instead of recomputing.
    persistent_idx, _ = quality.select_persistent_channels(train_energies)
    print(f"persistent channels: {len(persistent_idx)}/{train_energies.shape[1]}")
    channel_floor, channel_scale = detect.channel_baseline(train_energies)

    # 4. Run both detectors over the test split.
    baseline_dets = baseline_detect(test_files, fs, cache_path=f"{CACHE_DIR}/test_baseline_image.npz")
    print(f"baseline raw detections: {len(baseline_dets)}")
    baseline_events = detect.cluster_detections(
        baseline_dets.assign(triggered=True), time_tol_s=10.0, channel_tol=int(500 / dx_m)
    )
    print(f"baseline clustered events: {len(baseline_events)}")

    improved_raw = detect.detect_stream(
        test_files, fs, step=STEP, channel_floor=channel_floor, channel_scale=channel_scale
    )
    improved_raw = detect.common_mode_suppression(improved_raw, max_frac_triggered=0.05)
    improved_raw = detect.spatial_corroboration(improved_raw, channel_gap=STEP, time_tol_s=5.0)
    improved_events = detect.cluster_detections(improved_raw, time_tol_s=10.0, channel_tol=int(500 / dx_m))
    print(f"improved clustered events: {len(improved_events)}")

    # 5. Metrics across ranges.
    baseline_metrics = evaluate.evaluate_at_ranges(
        baseline_events, test_gt, corrected_lats, corrected_lons, corrected_ids, RANGES_M, MAX_TIME_S
    )
    improved_metrics = evaluate.evaluate_at_ranges(
        improved_events, test_gt, corrected_lats, corrected_lons, corrected_ids, RANGES_M, MAX_TIME_S
    )
    print("\n=== baseline (fixed-threshold RMS) ===")
    print(baseline_metrics.to_string(index=False))
    print("\n=== improved (stepped, contextual) ===")
    print(improved_metrics.to_string(index=False))


if __name__ == "__main__":
    main()
