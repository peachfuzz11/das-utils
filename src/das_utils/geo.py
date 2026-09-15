"""Local-projection geometry helpers for DAS channel positions.

A flat-earth (equirectangular) projection is accurate enough at typical DAS
deployment scales (a few to a few tens of km) and much cheaper than a real
geodesy library - don't reach for one here unless a deployment genuinely
spans hundreds of km.
"""

import math

import numpy as np
from scipy.spatial import cKDTree

# A single meters-per-degree-of-latitude constant, used everywhere in this
# module. Deliberately *not* derived from a separate "earth radius" - two
# independently-chosen approximations (a mean radius for one calculation, a
# WGS84-ish constant for another) can disagree by ~0.1%, which is small but
# real, and was enough to break exact round-tripping between local_xy and
# offset_m before this was unified. One constant, one source of truth.
METERS_PER_DEGREE_LAT = 111320.0


def local_xy(lat, lon, ref_lat: float, ref_lon: float):
    """Project (lat, lon) to local meters east/north of (ref_lat, ref_lon).

    Inputs may be scalars or arrays. Good for building a KD-tree over a
    single route or comparing points within a few tens of km of ref.
    """
    x = (lon - ref_lon) * METERS_PER_DEGREE_LAT * np.cos(np.radians(ref_lat))
    y = (lat - ref_lat) * METERS_PER_DEGREE_LAT
    return x, y


def offset_m(lat, lon, ref_lat, ref_lon) -> tuple[float, float]:
    """(delta_lat_m, delta_lon_m) from (ref_lat, ref_lon) to (lat, lon).

    This is the building block for both "how far apart are these two
    points" (via hypot of the result) and "what translation moves ref onto
    lat/lon" (the result itself, added back to ref as degrees). Same
    projection as ``local_xy``, with (x, y) relabeled as (dlon_m, dlat_m) -
    kept as a separate name because the two calls read differently at the
    point of use (projecting many points for a KD-tree, vs. computing one
    offset vector), not because they compute anything different.
    """
    x, y = local_xy(lat, lon, ref_lat, ref_lon)
    return y, x  # (dlat_m, dlon_m)


def build_nearest_tree(lats: np.ndarray, lons: np.ndarray) -> tuple[cKDTree, float, float]:
    """A KD-tree over a set of positions (e.g. every channel on a cable), for repeated
    nearest-point queries. Returns (tree, ref_lat, ref_lon) - reuse the same ref for
    any later `local_xy` calls against points you'll query this tree with.
    """
    ref_lat, ref_lon = float(lats.mean()), float(lons.mean())
    x, y = local_xy(lats, lons, ref_lat, ref_lon)
    tree = cKDTree(np.column_stack([x, y]))
    return tree, ref_lat, ref_lon


def nearest_index(tree: cKDTree, ref_lat: float, ref_lon: float, lat, lon):
    """Index (or array of indices) into the arrays a tree was built from, nearest to (lat, lon)."""
    x, y = local_xy(np.atleast_1d(lat), np.atleast_1d(lon), ref_lat, ref_lon)
    dist, idx = tree.query(np.column_stack([x, y]))
    return dist, idx


def apply_translation(positions: dict, dlat_m: float, dlon_m: float) -> dict:
    """Shift every {id: {"lat", "lon", ...}} entry by a constant translation.

    Converts the (dlat_m, dlon_m) translation to per-point degree deltas
    using each point's own latitude for the longitude scale factor, so it
    stays accurate across a route that spans a non-trivial latitude range.
    """
    dlat_deg = dlat_m / METERS_PER_DEGREE_LAT
    corrected = {}
    for cid, p in positions.items():
        dlon_deg = dlon_m / (METERS_PER_DEGREE_LAT * math.cos(math.radians(p["lat"])))
        corrected[cid] = {**p, "lat": p["lat"] + dlat_deg, "lon": p["lon"] + dlon_deg}
    return corrected


def positions_to_geojson_linestring(
    positions: dict, name: str = "cable", properties: dict | None = None
) -> dict:
    """A {id: {"lat","lon",...}} channel-position table as a standard GeoJSON LineString.

    `positions` keys are sorted (numerically if all-int, else lexically) to
    determine route order - pass an already-ordered dict of str keys if
    your channel ids don't sort into the right physical order numerically.
    """
    ids = sorted(positions)
    coords = [[positions[i]["lon"], positions[i]["lat"]] for i in ids]
    return {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {"name": name, **(properties or {})},
                "geometry": {"type": "LineString", "coordinates": coords},
            }
        ],
    }
