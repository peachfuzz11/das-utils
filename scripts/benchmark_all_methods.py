"""Benchmark every detection methodology on one shared dataset and ground truth.

Usage: uv run python scripts/benchmark_all_methods.py

Treats the whole DAS dataset (all three time-of-day folders) as a single
dataset: one cable geo-registration offset (from ``find_cable_offset.py``,
run once and reused here as a fixed constant - not re-derived), one curated
per-file per-channel ship-distance annotation (``das_utils.annotate``), and
every method benchmarked against exactly the same files and ground truth so
results are directly comparable.

Methods compared:

- **baseline**: fixed-threshold RMS energy z-score (naive strawman).
- **temporal_zscore**: per-channel z-score against a broad temporal baseline
  (``detect.channel_baseline``/``detect_file``), the first redesign this
  session, + common-mode suppression + spatial corroboration.
- **sta_lta_bandpass**: classic causal STA/LTA (``detect.sta_lta_stream``) on
  bandpass-filtered power, + common-mode suppression + spatial corroboration.
- **sta_lta_wavelet_<f>hz**: same STA/LTA, but the energy source is Morlet
  wavelet power at a single frequency (``wavelet.morlet_cwt_power_series``)
  instead of a bandpass - one per frequency in ``WAVELET_FREQS_TO_TRY``.
- **radon**: linear Radon (slant-stack) detection of a moving source
  crossing the fiber (``das_utils.radon``) - scores the coherent, sequentially
  time-shifted line shape a passing vessel's wavefront traces across nearby
  channels, rather than a per-channel energy level.

Each method's full per-range accuracy/precision/recall/F1 table
(``annotate.benchmark_detections``) is printed and the whole comparison is
written to ``RESULTS_CSV`` for later reference.
"""

import json
import os

import numpy as np
import pandas as pd

from das_utils import ais, annotate, detect, geo, image, io, quality, radon

DATA_DIR = "/home/ph/data/das"
DATE = "2025-11-22"
ALL_DIRS = [f"{DATA_DIR}/das_08_12", f"{DATA_DIR}/das_12_16", f"{DATA_DIR}/das_16_22"]
CABLE_JSON = f"{DATA_DIR}/das_cable.json"
AIS_CSV = f"{DATA_DIR}/aisdk-2025-11-22.csv"
CACHE_DIR = "/tmp/das_eval_cache"
FIXED_OFFSET_PATH = f"{CACHE_DIR}/fixed_offset.json"
ANNOTATIONS_PATH = f"{CACHE_DIR}/annotations_fixed_offset.json"
RESULTS_CSV = f"{CACHE_DIR}/all_methods_results.csv"

RANGES_M = (100.0, 250.0, 500.0, 1000.0, 2000.0)
STEP = 50
COMMON_MODE_MAX_FRAC = 0.05
CORROBORATION_TIME_TOL_S = 5.0

# Same, single dataset for every method - "treat the dataset as one single training dataset".
# None = every file; an evenly-spaced stride keeps full-day coverage while keeping the whole
# sweep (baseline + several STA/LTA variants) tractable in one run.
MAX_FILES_PER_DIR = 300

Z_THRESHOLD = 6.0
Z_SUSTAIN_DURATION = 0.9

STA_LTA_BIN_S = 0.25
STA_LTA_SHORT_S = 1.0
STA_LTA_LONG_S = 20.0
STA_LTA_RATIO_THRESHOLD = 3.0
STA_LTA_SUSTAIN_DURATION = 0.3
WAVELET_FREQS_TO_TRY = [5.0, 20.0, 80.0]

RADON_PATCH_RADIUS = 15
RADON_BIN_S = 0.1
# bins-per-channel; excludes |slowness| < 0.1 deliberately - near-zero is a fiber-wide,
# simultaneous artifact, not a moving source (see das_utils.radon module docstring).
RADON_SLOWNESSES = np.concatenate([np.linspace(-2.0, -0.1, 8), np.linspace(0.1, 2.0, 8)])
RADON_SCORE_THRESHOLD = 6.0


def load_files():
    files = io.index_das_files(ALL_DIRS, DATE)
    if MAX_FILES_PER_DIR is None:
        return files
    by_dir: dict[str, list] = {}
    for ts, f in files:
        by_dir.setdefault(os.path.dirname(f), []).append((ts, f))
    sampled = []
    for group in by_dir.values():
        stride = max(1, len(group) // MAX_FILES_PER_DIR)
        sampled.extend(group[::stride][:MAX_FILES_PER_DIR])
    return sorted(sampled, key=lambda p: p[0])


def dense_baseline(files, fs, cache_path):
    img, bin_times = image.build_image(
        files, reducer="rms", fs=fs, bandpass_hz=(1.0, 20.0), cache_path=cache_path
    )
    mean, std = img.mean(axis=0, keepdims=True), img.std(axis=0, keepdims=True)
    std[std == 0] = 1.0
    z = (img - mean) / std
    n_files, n_channels = z.shape
    file_paths = [f for _, f in files]
    return pd.DataFrame(
        {
            "file": np.repeat(file_paths, n_channels),
            "channel": np.tile(np.arange(n_channels), n_files),
            "score": z.ravel(),
            "triggered": (z > 4.0).ravel(),
        }
    )


def postprocess(dets, dx_m):
    dets = detect.common_mode_suppression(dets, max_frac_triggered=COMMON_MODE_MAX_FRAC)
    dets = detect.spatial_corroboration(dets, channel_gap=STEP, time_tol_s=CORROBORATION_TIME_TOL_S)
    return dets


def main():
    os.makedirs(CACHE_DIR, exist_ok=True)
    cable = io.load_channel_positions(CABLE_JSON)
    ids, lats, lons = io.positions_arrays(cable)
    fs = cable["metadata"]["fs"]
    dx_m = cable["metadata"]["dx"]

    files = load_files()
    print(f"files in scope: {len(files)}")
    n_channels = np.load(files[0][1], mmap_mode="r").shape[1]
    ids, lats, lons = ids[:n_channels], lats[:n_channels], lons[:n_channels]

    # Fixed geo-registration offset, derived once by find_cable_offset.py - not re-derived here.
    with open(FIXED_OFFSET_PATH) as f:
        offset = json.load(f)
    print(f"using fixed offset: {offset}")
    corrected = geo.apply_translation(cable["positions"], offset["dlat_m"], offset["dlon_m"])
    ids, lats, lons = io.positions_arrays({"positions": corrected, "metadata": cable["metadata"]})
    ids, lats, lons = ids[:n_channels], lats[:n_channels], lons[:n_channels]

    # Curated ground truth, built once with the corrected positions, reused by every method.
    if os.path.exists(ANNOTATIONS_PATH):
        print(f"loading cached annotations from {ANNOTATIONS_PATH}")
        annotations = annotate.load_annotations(ANNOTATIONS_PATH)
    else:
        start, end = files[0][0] - pd.Timedelta(seconds=60), files[-1][0] + pd.Timedelta(seconds=70)
        raw_ais = ais.extract_time_window(AIS_CSV, start, end)
        ais_near = ais.filter_near_positions(raw_ais, ids, lats, lons, max_dist_m=max(RANGES_M) * 1.5)
        print(f"AIS reports near route: {len(ais_near)}")
        annotations = annotate.build_annotations(files, ais_near, ids, lats, lons, progress_every=200)
        annotate.save_annotations(annotations, ANNOTATIONS_PATH)
        print(f"saved annotations to {ANNOTATIONS_PATH}")
    any_ship = sum(
        1 for entry in annotations["files"].values() if min(entry["distances_m"]) < max(RANGES_M)
    )
    print(f"files with a ship within {max(RANGES_M):.0f}m: {any_ship}/{len(annotations['files'])}")

    # Reference image, used by both the baseline z-score comparator and the temporal_zscore
    # method's baseline.
    ref_cache = f"{CACHE_DIR}/all_methods_ref_image_{len(files)}.npz"
    img, _ = image.build_image(files, reducer="rms", fs=fs, bandpass_hz=(1.0, 20.0), cache_path=ref_cache)
    persistent_idx, _ = quality.select_persistent_channels(img)
    print(f"persistent channels: {len(persistent_idx)}/{img.shape[1]}")
    channel_floor, channel_scale = detect.channel_baseline(img)

    all_results = []

    def run_method(name, dets):
        metrics = annotate.benchmark_detections(dets, annotations, RANGES_M)
        metrics.insert(0, "method", name)
        print(f"\n=== {name} ===")
        print(metrics.to_string(index=False))
        all_results.append(metrics)

    print("\n--- baseline: fixed-threshold RMS z-score ---")
    run_method("baseline_fixed_threshold", dense_baseline(files, fs, ref_cache))

    print("\n--- temporal_zscore: channel_baseline + common-mode + corroboration ---")
    tz_dets = detect.detect_stream(
        files, fs, step=STEP, channel_floor=channel_floor, channel_scale=channel_scale,
        z_threshold=Z_THRESHOLD, sustain_duration=Z_SUSTAIN_DURATION, progress_every=100,
    )
    run_method("temporal_zscore", postprocess(tz_dets, dx_m))

    print("\n--- sta_lta_bandpass (1-20Hz) ---")
    sl_bp = detect.sta_lta_stream(
        files, fs, step=STEP, bandpass_hz=(1.0, 20.0), bin_s=STA_LTA_BIN_S,
        short_s=STA_LTA_SHORT_S, long_s=STA_LTA_LONG_S, ratio_threshold=STA_LTA_RATIO_THRESHOLD,
        sustain_duration=STA_LTA_SUSTAIN_DURATION, progress_every=100,
    )
    run_method("sta_lta_bandpass_1_20hz", postprocess(sl_bp, dx_m))

    for freq in WAVELET_FREQS_TO_TRY:
        print(f"\n--- sta_lta_wavelet_{freq:.0f}hz ---")
        sl_wv = detect.sta_lta_stream(
            files, fs, step=STEP, bandpass_hz=None, wavelet_freq=freq, bin_s=STA_LTA_BIN_S,
            short_s=STA_LTA_SHORT_S, long_s=STA_LTA_LONG_S, ratio_threshold=STA_LTA_RATIO_THRESHOLD,
            sustain_duration=STA_LTA_SUSTAIN_DURATION, progress_every=100,
        )
        run_method(f"sta_lta_wavelet_{freq:.0f}hz", postprocess(sl_wv, dx_m))

    print("\n--- radon: linear slant-stack moving-source detection ---")
    radon_dets = radon.radon_detect_stream(
        files, fs, step=STEP, patch_radius=RADON_PATCH_RADIUS, slownesses=RADON_SLOWNESSES,
        bin_s=RADON_BIN_S, score_ratio_threshold=RADON_SCORE_THRESHOLD, progress_every=50,
    )
    run_method("radon", postprocess(radon_dets, dx_m))

    final = pd.concat(all_results, ignore_index=True)
    final.to_csv(RESULTS_CSV, index=False)
    print(f"\nsaved combined results to {RESULTS_CSV}")
    print("\n=== FINAL SUMMARY (all methods x all ranges) ===")
    print(final.to_string(index=False))


if __name__ == "__main__":
    main()
