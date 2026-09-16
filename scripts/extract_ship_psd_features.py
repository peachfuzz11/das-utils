"""Extract PSD-based spectral features for confirmed ship crossings vs background noise.

Usage: uv run python scripts/extract_ship_psd_features.py

Follows the approach in Pedersen et al., "A Feasibility Study of Automated
Detection and Classification of Signals in Distributed Acoustic Sensing"
(Sensors 2025, 25(17), 5445): characterize each signal by its Welch power
spectral density and dominant frequency, then see whether that
frequency-domain representation separates ships from background noise.

Reuses the curated ground-truth annotation already built with the fixed,
validated cable offset (``annotations_fixed_offset.json`` from
``benchmark_all_methods.py`` / ``find_cable_offset.py``) - "the files that
have a ship" means every file in that annotation whose nearest AIS-derived
distance is inside ``SHIP_MAX_DIST_M``. For each such file, the channel
nearest the ship (from the annotation's own distance array) is treated as
the "ship" sample; for every file with *no* AIS vessel within
``NOISE_MIN_DIST_M`` of the whole cable, a handful of random channels are
"noise" samples. Both get the same PSD/dominant-frequency/band-power
features, written to ``OUTPUT_CSV`` for the analysis notebook to load and
plot (kept as a separate script rather than notebook cells so the slow,
raw-file-reading part is cached and reusable).
"""

import os

import numpy as np
import pandas as pd

from das_utils import annotate, psd

DATA_DIR = "/home/ph/data/das"
CACHE_DIR = "/tmp/das_eval_cache"
ANNOTATIONS_PATH = f"{CACHE_DIR}/annotations_fixed_offset.json"
OUTPUT_CSV = f"{CACHE_DIR}/ship_psd_features.csv"

FS = 400.0
SHIP_MAX_DIST_M = 300.0
NOISE_MIN_DIST_M = 2000.0
NOISE_SAMPLES_PER_FILE = 3
NPERSEG = 2048
BAND_EDGES = np.array([0, 1, 2, 5, 10, 20, 30, 40, 60, 80, 120, 160, 200], dtype=float)


def extract_features(path: str, channel: int) -> dict:
    data = np.load(path, mmap_mode="r")
    signal = np.asarray(data[:, channel], dtype=np.float64)
    freqs, p = psd.channel_psd(signal, FS, nperseg=NPERSEG)
    dom_freq = psd.dominant_frequency(freqs, p, fmin=0.5)
    powers = psd.band_powers(freqs, p, BAND_EDGES)
    total = powers.sum()
    powers_norm = powers / total if total > 0 else powers
    row = {"dominant_freq_hz": dom_freq, "total_power": float(total)}
    for i in range(len(BAND_EDGES) - 1):
        row[f"band_{BAND_EDGES[i]:.0f}_{BAND_EDGES[i + 1]:.0f}hz"] = float(powers_norm[i])
    return row


def main():
    annotations = annotate.load_annotations(ANNOTATIONS_PATH)
    rows = []
    rng = np.random.default_rng(0)

    n_ship = n_noise = 0
    for i, (path, entry) in enumerate(annotations["files"].items()):
        distances = np.asarray(entry["distances_m"])
        min_dist = float(distances.min())

        if min_dist <= SHIP_MAX_DIST_M:
            channel = int(np.argmin(distances))
            row = extract_features(path, channel)
            row.update(
                {"file": path, "channel": channel, "distance_m": min_dist, "label": "ship"}
            )
            rows.append(row)
            n_ship += 1
        elif min_dist >= NOISE_MIN_DIST_M:
            n_channels = distances.shape[0]
            channels = rng.choice(n_channels, size=NOISE_SAMPLES_PER_FILE, replace=False)
            for c in channels:
                row = extract_features(path, int(c))
                row.update(
                    {"file": path, "channel": int(c), "distance_m": min_dist, "label": "noise"}
                )
                rows.append(row)
                n_noise += 1

        if i % 100 == 0:
            print(f"{i + 1}/{len(annotations['files'])}: ship={n_ship} noise={n_noise}")

    df = pd.DataFrame(rows)
    os.makedirs(CACHE_DIR, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nsaved {len(df)} samples ({n_ship} ship, {n_noise} noise) to {OUTPUT_CSV}")
    print(df.groupby("label")["dominant_freq_hz"].describe())


if __name__ == "__main__":
    main()
