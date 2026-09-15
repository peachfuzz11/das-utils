import math

import numpy as np
import pytest

from das_utils.geo import (
    apply_translation,
    build_nearest_tree,
    local_xy,
    nearest_index,
    offset_m,
    positions_to_geojson_linestring,
)


class TestLocalXy:
    def test_ref_point_is_origin(self):
        x, y = local_xy(55.34, 11.00, 55.34, 11.00)
        assert x == pytest.approx(0.0, abs=1e-6)
        assert y == pytest.approx(0.0, abs=1e-6)

    def test_one_degree_latitude_is_about_111km(self):
        _, y = local_xy(56.34, 11.00, 55.34, 11.00)
        assert y == pytest.approx(111320.0, rel=1e-3)


class TestOffsetM:
    def test_matches_local_xy_magnitude(self):
        dlat_m, dlon_m = offset_m(55.35, 11.02, 55.34, 11.00)
        x, y = local_xy(55.35, 11.02, 55.34, 11.00)
        assert math.hypot(dlat_m, dlon_m) == pytest.approx(math.hypot(x, y), rel=1e-6)


class TestNearestIndex:
    def test_finds_closest_of_several_points(self):
        lats = np.array([55.30, 55.34, 55.40])
        lons = np.array([11.00, 11.00, 11.00])
        tree, ref_lat, ref_lon = build_nearest_tree(lats, lons)
        _, idx = nearest_index(tree, ref_lat, ref_lon, 55.341, 11.00)
        assert idx[0] == 1


class TestApplyTranslation:
    def test_shifts_every_point_by_the_same_meters(self):
        positions = {
            1: {"lat": 55.34, "lon": 11.00, "depth": 10.0},
            2: {"lat": 55.35, "lon": 11.02, "depth": 12.0},
        }
        corrected = apply_translation(positions, dlat_m=100.0, dlon_m=-200.0)
        for cid in positions:
            c, orig = corrected[cid], positions[cid]
            dlat, dlon = offset_m(c["lat"], c["lon"], orig["lat"], orig["lon"])
            assert dlat == pytest.approx(100.0, abs=0.5)
            assert dlon == pytest.approx(-200.0, abs=0.5)
            assert corrected[cid]["depth"] == positions[cid]["depth"]  # untouched


class TestPositionsToGeojsonLinestring:
    def test_produces_ordered_coordinates(self):
        positions = {2: {"lat": 55.35, "lon": 11.02}, 1: {"lat": 55.34, "lon": 11.00}}
        gj = positions_to_geojson_linestring(positions, name="test-route")
        coords = gj["features"][0]["geometry"]["coordinates"]
        assert coords == [[11.00, 55.34], [11.02, 55.35]]  # ordered by channel id, not insertion order
        assert gj["features"][0]["properties"]["name"] == "test-route"
