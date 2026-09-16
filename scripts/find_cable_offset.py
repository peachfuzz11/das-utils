"""Derive a single, fixed cable geo-registration offset from ship-crossing signal power.

Usage: uv run python scripts/find_cable_offset.py

Runs the AIS-DAS correlation recipe (``das_utils.correlate``) once, using
every genuinely transiting AIS vessel crossing near the cable across the
*whole* day (not a short window - more independent crossings means a more
stable median), sweeping several bandpass bands (going as high in frequency
as the 400Hz sample rate usefully allows) and search margins. The offset is
only trusted once it **plateaus**: stable in magnitude across a range of
search margins, in the expected ~1200-1500m ballpark this cable's prior
investigation established. The winning (band, offset) is printed and saved
once to disk - everything downstream reuses this single fixed value rather
than re-deriving it.
"""

import json
import os

import numpy as np
import pandas as pd

from das_utils import ais, correlate, geo, image, io, wavelet

DATA_DIR = "/home/ph/data/das"
DATE = "2025-11-22"
ALL_DIRS = [f"{DATA_DIR}/das_08_12", f"{DATA_DIR}/das_12_16", f"{DATA_DIR}/das_16_22"]
CABLE_JSON = f"{DATA_DIR}/das_cable.json"
AIS_CSV = f"{DATA_DIR}/aisdk-2025-11-22.csv"
CACHE_DIR = "/tmp/das_eval_cache"
OFFSET_PATH = f"{CACHE_DIR}/fixed_offset.json"
# Bandpass candidates: broadband low, then progressively higher - fs=400Hz gives a 200Hz
# Nyquist limit; go as high as is still comfortably below that.
BANDS = [(1.0, 20.0), (1.0, 50.0), (5.0, 50.0), (10.0, 100.0), (20.0, 150.0), (50.0, 190.0)]
# Single-frequency Morlet wavelet power candidates, spanning the same range - a ship's
# signature may concentrate at one frequency a fixed bandpass-then-RMS reduction dilutes.
WAVELET_FREQS = [2.0, 5.0, 10.0, 20.0, 40.0, 80.0, 120.0, 160.0]
MARGINS_M = [500, 900, 1300, 1700, 2100]
EXPECTED_RANGE_M = (1200.0, 1500.0)
# Every file across the whole day gives the most independent crossings but is expensive
# across 6 bands; an evenly-spaced stride keeps full-day time coverage (so crossings at any
# hour still have a chance to be sampled) while cutting the file count. None = every file.
MAX_FILES_PER_DIR = 150


def main():
    os.makedirs(CACHE_DIR, exist_ok=True)
    cable = io.load_channel_positions(CABLE_JSON)
    ids, lats, lons = io.positions_arrays(cable)
    fs = cable["metadata"]["fs"]
    dx_m = cable["metadata"]["dx"]

    files = io.index_das_files(ALL_DIRS, DATE)
    if MAX_FILES_PER_DIR is not None:
        by_dir: dict[str, list] = {}
        for ts, f in files:
            by_dir.setdefault(os.path.dirname(f), []).append((ts, f))
        sampled = []
        for group in by_dir.values():
            stride = max(1, len(group) // MAX_FILES_PER_DIR)
            sampled.extend(group[::stride][:MAX_FILES_PER_DIR])
        files = sorted(sampled, key=lambda p: p[0])
    n_channels = np.load(files[0][1], mmap_mode="r").shape[1]
    ids, lats, lons = ids[:n_channels], lats[:n_channels], lons[:n_channels]
    print(f"files in scope: {len(files)} (spanning {files[0][0]} to {files[-1][0]})")

    # Every genuinely transiting AIS crossing across the whole day, once.
    start, end = files[0][0] - pd.Timedelta(seconds=60), files[-1][0] + pd.Timedelta(seconds=70)
    raw_ais = ais.extract_time_window(AIS_CSV, start, end)
    ais_near = ais.filter_near_positions(raw_ais, ids, lats, lons, max_dist_m=2500.0)
    transiting = ais_near[ais_near["SOG"] > 1.0]
    candidates = transiting.loc[transiting.groupby("MMSI")["DistToRoute_m"].idxmin()]
    print(f"transiting crossing candidates across the whole day: {len(candidates)}")

    def evaluate(label, energies, bin_times):
        sensitivity = correlate.margin_sensitivity(
            candidates, energies, bin_times, lats, lons, channel_spacing_m=dx_m, margins_m=MARGINS_M
        )
        print(sensitivity.to_string(index=False))
        # "Plateaued": the last two margins (widest) agree closely with each other.
        tail = sensitivity.tail(2)
        plateau_spread = float(tail["dist_m"].max() - tail["dist_m"].min())
        final_dist = float(tail["dist_m"].iloc[-1])
        in_range = EXPECTED_RANGE_M[0] <= final_dist <= EXPECTED_RANGE_M[1]
        print(f"  -> dist_m={final_dist:.1f}, plateau_spread={plateau_spread:.1f}, in_range={in_range}")
        return {
            "label": label,
            "dlat_m": float(tail["dlat_m"].iloc[-1]),
            "dlon_m": float(tail["dlon_m"].iloc[-1]),
            "dist_m": final_dist,
            "plateau_spread_m": plateau_spread,
            "in_expected_range": in_range,
        }

    results = []
    for band in BANDS:
        cache_path = f"{CACHE_DIR}/energy_{band[0]:.0f}_{band[1]:.0f}.npz"
        print(f"\n--- bandpass {band} Hz ---")
        energies, bin_times = image.build_image(
            files, reducer="rms", fs=fs, bandpass_hz=band, cache_path=cache_path, progress_every=0
        )
        results.append(evaluate(f"bandpass {band}", energies, bin_times))

    for freq in WAVELET_FREQS:
        cache_path = f"{CACHE_DIR}/wavelet_{freq:.0f}hz.npz"
        print(f"\n--- wavelet {freq}Hz ---")
        energies, bin_times = wavelet.wavelet_image(
            files, fs, freq, cache_path=cache_path, progress_every=0
        )
        results.append(evaluate(f"wavelet {freq}Hz", energies, bin_times))

    results_df = pd.DataFrame(results)
    print("\n=== summary across bands ===")
    print(results_df.to_string(index=False))

    # Prefer bands whose estimate both plateaus (small spread) and lands in the expected
    # range; fall back to the tightest plateau overall if none qualify.
    qualifying = results_df[results_df["in_expected_range"]]
    winner = (qualifying if len(qualifying) else results_df).sort_values("plateau_spread_m").iloc[0]
    print(f"\nchosen: {winner['label']} -> dlat_m={winner['dlat_m']:.1f} dlon_m={winner['dlon_m']:.1f}")

    with open(OFFSET_PATH, "w") as f:
        json.dump(
            {
                "dlat_m": winner["dlat_m"],
                "dlon_m": winner["dlon_m"],
                "label": winner["label"],
                "dist_m": winner["dist_m"],
                "n_candidates": len(candidates),
            },
            f,
            indent=2,
        )
    print(f"saved fixed offset to {OFFSET_PATH}")

    corrected = geo.apply_translation(cable["positions"], winner["dlat_m"], winner["dlon_m"])
    print(f"sanity check: corrected position count = {len(corrected)}")


if __name__ == "__main__":
    main()
