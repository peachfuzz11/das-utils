"""Curate a per-file, per-channel "distance to nearest ship" annotation, then benchmark detectors.

Usage: uv run python scripts/curate_and_benchmark.py

Two stages:

1. **Curate**: for every DAS file in scope, at the file's center time, record
   each channel's distance to the nearest AIS-reporting vessel
   (``das_utils.annotate``). This is a one-time, detector-independent ground
   truth, cached to disk as JSON.
2. **Benchmark**: run a fixed-threshold RMS baseline and the improved,
   temporal-baseline stepped detector (``das_utils.detect``) over the same
   files, then threshold the annotation at several range thresholds and
   score both detectors' per-(file, channel) predictions against it
   (``das_utils.annotate.benchmark_detections``) - this is a true
   accuracy/precision/recall/F1 with well-defined true negatives, unlike the
   event-matching approach in ``run_ship_detection_eval.py``.

Channel-index note: the cable position table has 3919 entries (ids 1..3919)
but the recorded data only has 3917 channel columns - a known, previously
unresolved mismatch (see AGENTS.md). Absent a verified mapping, this script
follows the same convention already used elsewhere in this package
(``correlate.closest_approach_offset`` clips to the data's channel count):
position array index *i* (0-based, after sorting by id) is treated as data
column *i*. This is an assumption, not a verified fact - flagged here rather
than silently relied on.

Geo-registration note: ``das_cable.json`` is *uncorrected* - AGENTS.md and a
prior investigation both document a real cable geo-registration offset on
the order of ~1000m. Skipping this correction doesn't just shift the
detector's apparent position error - it corrupts the curated ground truth
itself (a real close crossing gets annotated as "far" if the channel table
is off by hundreds of meters), which silently caps recall regardless of how
good the detector is. So this script derives and applies that correction
(``correlate.estimate_translation_offset``) before building the annotation,
the same way ``run_ship_detection_eval.py`` does.
"""

import os

import numpy as np
import pandas as pd

from das_utils import ais, annotate, correlate, detect, geo, image, io, quality

DATA_DIR = "/home/ph/data/das"
DATE = "2025-11-22"
ALL_DIRS = [f"{DATA_DIR}/das_08_12", f"{DATA_DIR}/das_12_16", f"{DATA_DIR}/das_16_22"]
CABLE_JSON = f"{DATA_DIR}/das_cable.json"
AIS_CSV = f"{DATA_DIR}/aisdk-2025-11-22.csv"
RANGES_M = (100.0, 250.0, 500.0, 1000.0, 2000.0)
STEP = 50
Z_THRESHOLD = 6.0
SUSTAIN_DURATION = 0.9
COMMON_MODE_MAX_FRAC = 0.05
CACHE_DIR = "/tmp/das_eval_cache_v2"
ANNOTATIONS_PATH = f"{CACHE_DIR}/annotations.json"
# None = every file across ALL_DIRS ("loop over all the files"); set an int to scope a
# faster run - the curation and benchmark logic is identical either way.
MAX_FILES_PER_DIR = 150


def load_files():
    files = io.index_das_files(ALL_DIRS, DATE)
    if MAX_FILES_PER_DIR is None:
        return files
    by_dir: dict[str, list] = {}
    for ts, f in files:
        by_dir.setdefault(os.path.dirname(f), []).append((ts, f))
    return sorted(
        (row for group in by_dir.values() for row in group[:MAX_FILES_PER_DIR]), key=lambda p: p[0]
    )


def dense_baseline(files, fs, cache_path):
    """Fixed-threshold RMS baseline, densely: one triggered/score row per (file, channel)."""
    img, bin_times = image.build_image(
        files, reducer="rms", fs=fs, bandpass_hz=(1.0, 20.0), cache_path=cache_path
    )
    mean, std = img.mean(axis=0, keepdims=True), img.std(axis=0, keepdims=True)
    std[std == 0] = 1.0
    z = (img - mean) / std
    n_files, n_channels = z.shape
    file_paths = [f for _, f in files]
    rows = {
        "file": np.repeat(file_paths, n_channels),
        "channel": np.tile(np.arange(n_channels), n_files),
        "score": z.ravel(),
        "triggered": (z > 4.0).ravel(),
    }
    return pd.DataFrame(rows)


def main():
    os.makedirs(CACHE_DIR, exist_ok=True)
    cable = io.load_channel_positions(CABLE_JSON)
    ids, lats, lons = io.positions_arrays(cable)
    fs = cable["metadata"]["fs"]
    dx_m = cable["metadata"]["dx"]

    files = load_files()
    print(f"files in scope: {len(files)}")

    n_channels = np.load(files[0][1], mmap_mode="r").shape[1]
    if len(ids) != n_channels:
        print(f"NOTE: {len(ids)} position entries vs {n_channels} data channels - "
              f"truncating positions to the first {n_channels} by sorted id (see script docstring).")
    ids, lats, lons = ids[:n_channels], lats[:n_channels], lons[:n_channels]

    # 0. Reference image (from the same file scope): used both to derive the geo-registration
    # offset below and, later, as the improved detector's per-channel temporal baseline.
    baseline_cache = f"{CACHE_DIR}/baseline_image_{len(files)}.npz"
    img, bin_times = image.build_image(
        files, reducer="rms", fs=fs, bandpass_hz=(1.0, 20.0), cache_path=baseline_cache
    )
    persistent_idx, _ = quality.select_persistent_channels(img)
    print(f"persistent channels: {len(persistent_idx)}/{img.shape[1]}")
    channel_floor, channel_scale = detect.channel_baseline(img)

    # 1. Derive and apply the cable's geo-registration correction (das_cable.json is
    # uncorrected - see script docstring) before building the ground-truth annotation.
    start, end = files[0][0] - pd.Timedelta(seconds=60), files[-1][0] + pd.Timedelta(seconds=70)
    raw_ais = ais.extract_time_window(AIS_CSV, start, end)
    ais_near = ais.filter_near_positions(raw_ais, ids, lats, lons, max_dist_m=max(RANGES_M) * 1.5)
    print(f"AIS reports near route: {len(ais_near)}")
    transiting = ais_near[ais_near["SOG"] > 1.0]
    candidates = transiting.loc[transiting.groupby("MMSI")["DistToRoute_m"].idxmin()]
    if len(candidates) >= 3:
        dlat_m, dlon_m, _ = correlate.estimate_translation_offset(
            candidates, img, bin_times, lats, lons, channel_spacing_m=dx_m
        )
        print(f"derived offset: dlat_m={dlat_m:.1f} dlon_m={dlon_m:.1f} (n={len(candidates)})")
    else:
        dlat_m, dlon_m = 0.0, 0.0
        print(f"only {len(candidates)} transiting candidates - skipping offset correction")
    corrected = geo.apply_translation(cable["positions"], dlat_m, dlon_m)
    ids, lats, lons = io.positions_arrays({"positions": corrected, "metadata": cable["metadata"]})
    ids, lats, lons = ids[:n_channels], lats[:n_channels], lons[:n_channels]

    # 2. Curate: build (or load cached) per-file, per-channel ship-distance annotation,
    # using the corrected positions.
    if os.path.exists(ANNOTATIONS_PATH):
        print(f"loading cached annotations from {ANNOTATIONS_PATH}")
        annotations = annotate.load_annotations(ANNOTATIONS_PATH)
    else:
        ais_near = ais.filter_near_positions(raw_ais, ids, lats, lons, max_dist_m=max(RANGES_M) * 1.5)
        annotations = annotate.build_annotations(files, ais_near, ids, lats, lons, progress_every=200)
        annotate.save_annotations(annotations, ANNOTATIONS_PATH)
        print(f"saved annotations to {ANNOTATIONS_PATH}")

    any_ship = sum(
        1 for entry in annotations["files"].values() if min(entry["distances_m"]) < max(RANGES_M)
    )
    print(f"files with a ship within {max(RANGES_M):.0f}m: {any_ship}/{len(annotations['files'])}")

    # 3. Run both detectors, densely, over the same files.
    baseline_dets = dense_baseline(files, fs, baseline_cache)
    print(f"baseline dense rows: {len(baseline_dets)}, triggered: {int(baseline_dets['triggered'].sum())}")

    improved_dets = detect.detect_stream(
        files, fs, step=STEP, channel_floor=channel_floor, channel_scale=channel_scale,
        z_threshold=Z_THRESHOLD, sustain_duration=SUSTAIN_DURATION,
    )
    print(
        f"improved dense rows: {len(improved_dets)}, "
        f"sustained-only triggered: {int(improved_dets['triggered'].sum())}"
    )
    improved_dets = detect.common_mode_suppression(improved_dets, max_frac_triggered=COMMON_MODE_MAX_FRAC)
    print(f"improved triggered after common-mode suppression: {int(improved_dets['triggered'].sum())}")
    improved_dets = detect.spatial_corroboration(improved_dets, channel_gap=STEP, time_tol_s=5.0)
    print(f"improved triggered after spatial corroboration: {int(improved_dets['triggered'].sum())}")

    # 4. Benchmark both against the curated annotation, across range thresholds.
    baseline_metrics = annotate.benchmark_detections(baseline_dets, annotations, RANGES_M)
    improved_metrics = annotate.benchmark_detections(improved_dets, annotations, RANGES_M)
    print("\n=== baseline (fixed-threshold RMS), per-channel classification ===")
    print(baseline_metrics.to_string(index=False))
    print("\n=== improved (stepped, contextual), per-channel classification ===")
    print(improved_metrics.to_string(index=False))


if __name__ == "__main__":
    main()
