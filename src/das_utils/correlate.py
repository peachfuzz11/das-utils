"""Correlate independently-positioned reference events (e.g. AIS ship crossings)
against DAS channel energy to find a route's geographic mis-registration.

The core idea: if a route's channel-to-lat/lon mapping is correct, "the
channel nearest a reference event's true position" and "the channel with
peak DAS energy when that event happens" are the same channel. When they're
not, the discrepancy - aggregated over several independent, well-chosen
events - is the correction.

Two things matter a lot for getting a reliable answer, both covered in
AGENTS.md at more length:

* **Only use genuinely moving/transiting reference events.** A near-fixed
  reference makes "find the best-matching channel" degenerate into "find
  the single loudest channel in the whole dataset," which has nothing to
  do with that specific event. Filter candidates *before* calling anything
  here (e.g. exclude AIS contacts below a cruising-speed threshold) -
  these functions don't second-guess your candidate selection.
* **Bound the search window, and check the result is stable across it.**
  Too wide, and the search can walk off to a distant, unrelated, more
  powerful feature and "win" spuriously even for a moving reference. Use
  ``margin_sensitivity`` below before trusting a single margin's answer.
"""

import numpy as np
import pandas as pd

from das_utils.geo import build_nearest_tree, nearest_index, offset_m


def to_epoch_seconds(times: pd.DatetimeIndex) -> np.ndarray:
    """Seconds-since-epoch for a DatetimeIndex, independent of its internal time unit.

    ``DatetimeIndex.astype("int64")`` returns raw integers in whatever unit
    the index happens to be stored in (ns, us, ms, or s - pandas has stored
    non-nanosecond resolution since 2.x, and which one you get can depend on
    how the index was constructed). Naively dividing by ``1e9`` silently
    assumes nanoseconds and produces wrong values - not an error, just a
    scaled timestamp that still looks plausible - if that assumption is
    wrong. Forcing ``datetime64[ns]`` first makes the unit explicit.
    """
    return times.values.astype("datetime64[ns]").astype("int64") / 1e9


def closest_approach_offset(
    lat: float,
    lon: float,
    timestamp: pd.Timestamp,
    energies: np.ndarray,
    bin_times: pd.DatetimeIndex,
    route_tree,
    route_ref_lat: float,
    route_ref_lon: float,
    route_lats: np.ndarray,
    route_lons: np.ndarray,
    channel_spacing_m: float,
    search_margin_m: float = 1500.0,
) -> tuple[float, float]:
    """Offset implied by one reference event at (lat, lon, timestamp).

    Finds the DAS time-bin covering ``timestamp``, then the peak-energy
    channel within +/- ``search_margin_m`` of the channel naively nearest
    (lat, lon). Returns (dlat_m, dlon_m): the event's true position minus
    that detected channel's stated position - this is what you'd add to
    the route's stated positions to correct them (see
    ``geo.apply_translation``).

    ``route_tree``/``route_ref_lat``/``route_ref_lon`` come from
    ``geo.build_nearest_tree(route_lats, route_lons)`` - build it once and
    reuse across many calls rather than rebuilding per event.
    """
    n_channels = energies.shape[1]
    bin_epoch = to_epoch_seconds(bin_times)
    bi = int(np.argmin(np.abs(bin_epoch - timestamp.value / 1e9)))

    _, nearest = nearest_index(route_tree, route_ref_lat, route_ref_lon, lat, lon)
    nearest_col = int(np.clip(nearest[0], 0, n_channels - 1))

    margin_ch = round(search_margin_m / channel_spacing_m)
    lo, hi = max(0, nearest_col - margin_ch), min(n_channels, nearest_col + margin_ch)
    detected_col = lo + int(np.argmax(energies[bi, lo:hi]))

    return offset_m(lat, lon, route_lats[detected_col], route_lons[detected_col])


def estimate_translation_offset(
    candidates: pd.DataFrame,
    energies: np.ndarray,
    bin_times: pd.DatetimeIndex,
    route_lats: np.ndarray,
    route_lons: np.ndarray,
    channel_spacing_m: float,
    search_margin_m: float = 1500.0,
    lat_col: str = "Latitude",
    lon_col: str = "Longitude",
    time_col: str = "Timestamp",
) -> tuple[float, float, pd.DataFrame]:
    """Per-candidate offsets plus the aggregate (median) translation.

    ``candidates`` should already be filtered to the events you trust as
    reference points - one row per event (e.g. each vessel's own
    closest-approach point to the route). Returns
    ``(dlat_m, dlon_m, per_candidate)`` where ``per_candidate`` has one row
    per input candidate with its own ``dlat_m``/``dlon_m``/``dist_m``, for
    inspecting agreement before trusting the aggregate.
    """
    route_tree, ref_lat, ref_lon = build_nearest_tree(route_lats, route_lons)

    rows = []
    for _, row in candidates.iterrows():
        dlat_m, dlon_m = closest_approach_offset(
            row[lat_col], row[lon_col], row[time_col],
            energies, bin_times, route_tree, ref_lat, ref_lon, route_lats, route_lons,
            channel_spacing_m, search_margin_m,
        )
        rows.append({"dlat_m": dlat_m, "dlon_m": dlon_m, "dist_m": float(np.hypot(dlat_m, dlon_m))})
    per_candidate = pd.concat([candidates.reset_index(drop=True), pd.DataFrame(rows)], axis=1)

    return float(per_candidate["dlat_m"].median()), float(per_candidate["dlon_m"].median()), per_candidate


def margin_sensitivity(
    candidates: pd.DataFrame,
    energies: np.ndarray,
    bin_times: pd.DatetimeIndex,
    route_lats: np.ndarray,
    route_lons: np.ndarray,
    channel_spacing_m: float,
    margins_m: list[float],
    lat_col: str = "Latitude",
    lon_col: str = "Longitude",
    time_col: str = "Timestamp",
) -> pd.DataFrame:
    """Median offset magnitude across a range of search margins.

    Always run this before trusting a single margin's result. A real
    detection produces a plateau - stable across a healthy range - and only
    degrades well outside it; if the answer keeps changing as the margin
    changes, it isn't a real detection yet.
    """
    results = []
    for margin_m in margins_m:
        dlat_m, dlon_m, _ = estimate_translation_offset(
            candidates, energies, bin_times, route_lats, route_lons,
            channel_spacing_m, margin_m, lat_col, lon_col, time_col,
        )
        dist_m = float(np.hypot(dlat_m, dlon_m))
        results.append({"margin_m": margin_m, "dlat_m": dlat_m, "dlon_m": dlon_m, "dist_m": dist_m})
    return pd.DataFrame(results)
