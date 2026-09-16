"""Tests for matching detections to AIS-derived ground truth and scoring them."""

import numpy as np
import pandas as pd
import pytest

from das_utils.evaluate import (
    compute_metrics,
    evaluate_at_ranges,
    ground_truth_crossings,
    match_detections,
)
from das_utils.geo import METERS_PER_DEGREE_LAT

N_CHANNELS = 200
DX_M = 2.0
BASE_LAT = 55.30


def make_route(n_channels=N_CHANNELS, dx_m=DX_M, base_lat=BASE_LAT, base_lon=11.0):
    lon_step_deg = dx_m / (METERS_PER_DEGREE_LAT * np.cos(np.radians(base_lat)))
    lons = base_lon + np.arange(n_channels) * lon_step_deg
    lats = np.full(n_channels, base_lat)
    ids = np.arange(n_channels)
    return ids, lats, lons


class TestGroundTruthCrossings:
    def test_picks_closest_approach_per_vessel(self):
        ids, lats, lons = make_route()
        ais = pd.DataFrame(
            {
                "MMSI": [111, 111, 222],
                "Latitude": [lats[50], lats[50], lats[150]],
                "Longitude": [lons[50] + 0.01, lons[50], lons[150]],
                "Timestamp": pd.to_datetime(
                    ["2025-01-01 00:00:00", "2025-01-01 00:00:10", "2025-01-01 00:01:00"]
                ),
                "DistToRoute_m": [500.0, 10.0, 5.0],
            }
        )
        crossings = ground_truth_crossings(ais, ids, lats, lons)
        assert len(crossings) == 2
        row111 = crossings[crossings["MMSI"] == 111].iloc[0]
        assert row111["DistToRoute_m"] == pytest.approx(10.0)
        assert row111["channel"] == 50


class TestMatchDetections:
    def test_matches_within_range_and_time(self):
        ids, lats, lons = make_route()
        gt = pd.DataFrame(
            {"channel": [50], "timestamp": pd.to_datetime(["2025-01-01 00:00:00"])}
        )
        det = pd.DataFrame(
            {"channel": [52], "timestamp": pd.to_datetime(["2025-01-01 00:00:05"])}
        )
        matched, fp, fn = match_detections(det, gt, lats, lons, ids, max_range_m=50.0, max_time_s=60.0)
        assert len(matched) == 1
        assert len(fp) == 0
        assert len(fn) == 0

    def test_no_match_outside_range(self):
        ids, lats, lons = make_route()
        gt = pd.DataFrame({"channel": [50], "timestamp": pd.to_datetime(["2025-01-01 00:00:00"])})
        det = pd.DataFrame({"channel": [150], "timestamp": pd.to_datetime(["2025-01-01 00:00:05"])})
        matched, fp, fn = match_detections(det, gt, lats, lons, ids, max_range_m=50.0, max_time_s=60.0)
        assert len(matched) == 0
        assert len(fp) == 1
        assert len(fn) == 1

    def test_no_match_outside_time_tolerance(self):
        ids, lats, lons = make_route()
        gt = pd.DataFrame({"channel": [50], "timestamp": pd.to_datetime(["2025-01-01 00:00:00"])})
        det = pd.DataFrame({"channel": [50], "timestamp": pd.to_datetime(["2025-01-01 01:00:00"])})
        matched, fp, fn = match_detections(det, gt, lats, lons, ids, max_range_m=1000.0, max_time_s=60.0)
        assert len(matched) == 0
        assert len(fp) == 1
        assert len(fn) == 1


class TestComputeMetrics:
    def test_perfect_match(self):
        matched = pd.DataFrame({"a": [1, 2, 3]})
        empty = pd.DataFrame()
        m = compute_metrics(matched, empty, empty)
        assert m["precision"] == 1.0
        assert m["recall"] == 1.0
        assert m["f1"] == 1.0
        assert m["accuracy"] == 1.0

    def test_all_false_positive(self):
        matched = pd.DataFrame()
        fp = pd.DataFrame({"a": [1, 2]})
        fn = pd.DataFrame()
        m = compute_metrics(matched, fp, fn)
        assert m["precision"] == 0.0
        assert m["recall"] == 0.0
        assert m["f1"] == 0.0


class TestEvaluateAtRanges:
    def test_recall_nondecreasing_with_range(self):
        ids, lats, lons = make_route()
        gt = pd.DataFrame(
            {"channel": [50, 150], "timestamp": pd.to_datetime(["2025-01-01 00:00:00"] * 2)}
        )
        det = pd.DataFrame(
            {"channel": [55, 150], "timestamp": pd.to_datetime(["2025-01-01 00:00:00"] * 2)}
        )
        result = evaluate_at_ranges(
            det, gt, lats, lons, ids, ranges_m=(1.0, 20.0, 200.0), max_time_s=60.0
        )
        recalls = result.sort_values("range_m")["recall"].to_numpy()
        assert np.all(np.diff(recalls) >= 0)
        assert recalls[-1] == pytest.approx(1.0)
