import numpy as np
import pandas as pd
import pytest

from das_utils.ais import extract_time_window, filter_near_positions


class TestExtractTimeWindow:
    def test_strips_header_hash_and_filters_to_window(self, tmp_path):
        csv = tmp_path / "ais.csv"
        csv.write_text(
            "# Timestamp,Type of mobile,MMSI,Latitude,Longitude,SOG\n"
            "22/11/2025 07:59:59,Class A,111,55.34,11.00,10.0\n"
            "22/11/2025 08:00:05,Class A,222,55.35,11.01,12.0\n"
            "22/11/2025 08:00:15,Class A,333,55.36,11.02,9.0\n"
            "22/11/2025 08:01:00,Class A,444,55.37,11.03,11.0\n"
        )
        start = pd.Timestamp("2025-11-22 08:00:00")
        end = pd.Timestamp("2025-11-22 08:01:00")

        df = extract_time_window(str(csv), start, end)

        assert list(df.columns[:2]) == ["Timestamp", "Type of mobile"]  # "# " stripped
        assert set(df["MMSI"]) == {222, 333}  # before start and >= end both excluded


class TestFilterNearPositions:
    def test_filters_by_class_distance_and_duplicates(self):
        ids = np.array([1, 2, 3])
        lats = np.array([55.34, 55.34, 55.34])
        lons = np.array([11.00, 11.01, 11.02])

        df = pd.DataFrame(
            {
                "Timestamp": pd.to_datetime(["2025-01-01 00:00:00"] * 4),
                "Type of mobile": ["Class A", "Class A", "Base Station", "Class A"],
                "MMSI": [1, 1, 2, 3],
                "Latitude": [55.34, 55.34, 55.34, 60.0],  # last one is far away
                "Longitude": [11.00, 11.00, 11.01, 20.0],
            }
        )

        result = filter_near_positions(df, ids, lats, lons, max_dist_m=500.0)

        assert len(result) == 1  # dup MMSI=1 row collapsed, base station and far point dropped
        assert result["MMSI"].iloc[0] == 1
        assert "DistToRoute_m" in result.columns
        assert result["DistToRoute_m"].iloc[0] == pytest.approx(0.0, abs=1.0)
