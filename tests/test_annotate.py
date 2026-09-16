"""Tests for building and using the per-file, per-channel ship-distance annotation."""

import numpy as np
import pandas as pd
import pytest

from das_utils.annotate import (
    NO_SHIP_SENTINEL_M,
    annotate_file,
    benchmark_detections,
    build_annotations,
    labels_for_range,
    load_annotations,
    nearest_ship_distance_m,
    save_annotations,
)
from das_utils.geo import METERS_PER_DEGREE_LAT

BASE_LAT = 55.30


def make_route(n_channels=100, dx_m=2.0, base_lat=BASE_LAT, base_lon=11.0):
    lon_step_deg = dx_m / (METERS_PER_DEGREE_LAT * np.cos(np.radians(base_lat)))
    lons = base_lon + np.arange(n_channels) * lon_step_deg
    lats = np.full(n_channels, base_lat)
    ids = np.arange(n_channels)
    return ids, lats, lons


class TestNearestShipDistanceM:
    def test_no_ships_returns_sentinel(self):
        ids, lats, lons = make_route(10)
        dist = nearest_ship_distance_m(lats, lons, np.array([]), np.array([]))
        assert np.all(dist == NO_SHIP_SENTINEL_M)

    def test_distance_zero_at_ship_channel(self):
        ids, lats, lons = make_route(10)
        dist = nearest_ship_distance_m(lats, lons, np.array([lats[5]]), np.array([lons[5]]))
        assert dist[5] == pytest.approx(0.0, abs=1e-6)
        assert dist[0] > dist[5]


class TestAnnotateFile:
    def test_picks_closest_report_in_time_window(self):
        ids, lats, lons = make_route(50)
        ais_near = pd.DataFrame(
            {
                "MMSI": [1, 1],
                "Latitude": [lats[20], lats[20]],
                "Longitude": [lons[20], lons[10]],
                "Timestamp": pd.to_datetime(["2025-01-01 00:00:00", "2025-01-01 00:00:05"]),
            }
        )
        center = pd.Timestamp("2025-01-01 00:00:01")
        dist = annotate_file(center, ais_near, lats, lons, time_tol_s=30.0)
        assert dist[20] == pytest.approx(0.0, abs=1e-6)

    def test_empty_window_gives_sentinel(self):
        ids, lats, lons = make_route(10)
        ais_near = pd.DataFrame(
            {"MMSI": [1], "Latitude": [lats[5]], "Longitude": [lons[5]],
             "Timestamp": pd.to_datetime(["2025-01-01 05:00:00"])}
        )
        center = pd.Timestamp("2025-01-01 00:00:00")
        dist = annotate_file(center, ais_near, lats, lons, time_tol_s=30.0)
        assert np.all(dist == NO_SHIP_SENTINEL_M)


class TestBuildSaveLoadAnnotations:
    def test_roundtrip(self, tmp_path):
        ids, lats, lons = make_route(20)
        files = [
            (pd.Timestamp("2025-01-01 00:00:00"), "a.npy"),
            (pd.Timestamp("2025-01-01 00:00:10"), "b.npy"),
        ]
        ais_near = pd.DataFrame(
            {"MMSI": [1], "Latitude": [lats[10]], "Longitude": [lons[10]],
             "Timestamp": pd.to_datetime(["2025-01-01 00:00:05"])}
        )
        ann = build_annotations(
            files, ais_near, ids, lats, lons, file_duration_s=10.0, time_tol_s=30.0, progress_every=0
        )
        path = str(tmp_path / "ann.json")
        save_annotations(ann, path)
        loaded = load_annotations(path)
        assert loaded["channel_ids"] == list(range(20))
        assert set(loaded["files"]) == {"a.npy", "b.npy"}
        assert loaded["files"]["a.npy"]["distances_m"][10] == pytest.approx(0.0, abs=1.0)


class TestLabelsForRange:
    def test_thresholds_distances(self):
        ann = {
            "channel_ids": [0, 1, 2],
            "files": {"f.npy": {"timestamp": "t", "distances_m": [10.0, 600.0, 2000.0]}},
        }
        labels = labels_for_range(ann, 500.0)
        assert list(labels["f.npy"]) == [True, False, False]
        labels = labels_for_range(ann, 1000.0)
        assert list(labels["f.npy"]) == [True, True, False]


class TestBenchmarkDetections:
    def test_perfect_detector(self):
        ann = {
            "channel_ids": [0, 1, 2, 3],
            "files": {"f.npy": {"timestamp": "t", "distances_m": [10.0, 600.0, 2000.0, 5.0]}},
        }
        detections = pd.DataFrame(
            {
                "file": ["f.npy"] * 4,
                "channel": [0, 1, 2, 3],
                "triggered": [True, False, False, True],
            }
        )
        result = benchmark_detections(detections, ann, ranges_m=(500.0,))
        row = result.iloc[0]
        assert row["tp"] == 2
        assert row["fp"] == 0
        assert row["fn"] == 0
        assert row["tn"] == 2
        assert row["accuracy"] == pytest.approx(1.0)
        assert row["f1"] == pytest.approx(1.0)

    def test_looser_range_can_turn_fp_into_tp(self):
        ann = {
            "channel_ids": [0, 1],
            "files": {"f.npy": {"timestamp": "t", "distances_m": [10.0, 800.0]}},
        }
        detections = pd.DataFrame({"file": ["f.npy"] * 2, "channel": [0, 1], "triggered": [True, True]})
        result = benchmark_detections(detections, ann, ranges_m=(500.0, 1000.0))
        tight = result[result["range_m"] == 500.0].iloc[0]
        loose = result[result["range_m"] == 1000.0].iloc[0]
        assert tight["fp"] == 1
        assert loose["fp"] == 0
        assert loose["precision"] >= tight["precision"]

    def test_ignores_files_without_annotation(self):
        ann = {"channel_ids": [0], "files": {}}
        detections = pd.DataFrame({"file": ["missing.npy"], "channel": [0], "triggered": [True]})
        result = benchmark_detections(detections, ann, ranges_m=(500.0,))
        assert result.iloc[0][["tp", "fp", "tn", "fn"]].sum() == 0
