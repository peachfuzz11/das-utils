"""Synthetic recovery tests for the AIS-DAS correlation recipe.

Nothing in this domain is easy to unit-test against real data, so these
inject a known translation offset into a synthetic cable + energy image and
check it's recovered - the same discipline recommended in AGENTS.md.

The synthetic route is made long (2000 channels) and events kept well away
from either end, so that "true position = stated position + offset" never
lands outside the route itself - that would make the nearest-neighbor
lookup clamp to an endpoint and silently break the scenario rather than
testing what's intended.
"""

import math

import numpy as np
import pandas as pd
import pytest

from das_utils.correlate import estimate_translation_offset, margin_sensitivity
from das_utils.geo import METERS_PER_DEGREE_LAT

N_CHANNELS = 2000
DX_M = 2.0


def make_synthetic_route(
    n_channels: int = N_CHANNELS, dx_m: float = DX_M, base_lat: float = 55.34, base_lon: float = 11.00
):
    """A straight, west-to-east synthetic cable's *stated* (uncorrected) positions."""
    lon_step_deg = dx_m / (METERS_PER_DEGREE_LAT * np.cos(np.radians(base_lat)))
    lons = base_lon + np.arange(n_channels) * lon_step_deg
    lats = np.full(n_channels, base_lat)
    return lats, lons


def inject_offset_scenario(true_dlat_m: float, true_dlon_m: float, event_channels: list[int]):
    """Build synthetic (candidates, energies, bin_times, route_lats, route_lons) with a known offset.

    For each channel index in `event_channels`, one timestamp/energy-bin has
    a sharp spike there, and the corresponding "AIS" candidate is placed at
    that channel's *true* (offset-shifted) position - exactly the
    relationship closest_approach_offset is meant to recover.
    """
    route_lats, route_lons = make_synthetic_route()

    n_bins = len(event_channels)
    energies = np.random.default_rng(0).normal(1.0, 0.1, size=(n_bins, N_CHANNELS)) ** 2
    start = pd.Timestamp("2025-01-01 00:00:00")
    bin_times = pd.DatetimeIndex(start + pd.to_timedelta(np.arange(n_bins) * 10, "s"))

    dlat_deg = true_dlat_m / METERS_PER_DEGREE_LAT
    rows = []
    for i, ch in enumerate(event_channels):
        energies[i, ch] = 1000.0  # unmistakable spike
        true_lat = route_lats[ch] + dlat_deg
        dlon_deg = true_dlon_m / (METERS_PER_DEGREE_LAT * np.cos(np.radians(route_lats[ch])))
        true_lon = route_lons[ch] + dlon_deg
        rows.append({"Latitude": true_lat, "Longitude": true_lon, "Timestamp": bin_times[i]})

    candidates = pd.DataFrame(rows)
    return candidates, energies, bin_times, route_lats, route_lons


class TestEstimateTranslationOffset:
    def test_recovers_known_eastward_offset(self):
        # +800m = +400 channels; events centered so true_ch + offset + margin stays in bounds
        candidates, energies, bin_times, route_lats, route_lons = inject_offset_scenario(
            true_dlat_m=50.0, true_dlon_m=800.0, event_channels=[700, 800, 900, 1000]
        )
        dlat_m, dlon_m, per_candidate = estimate_translation_offset(
            candidates, energies, bin_times, route_lats, route_lons,
            channel_spacing_m=DX_M, search_margin_m=900.0,
        )
        assert dlat_m == pytest.approx(50.0, abs=5.0)
        assert dlon_m == pytest.approx(800.0, abs=5.0)
        assert len(per_candidate) == 4
        # dist_m is each candidate's own recovered offset magnitude - all 4
        # should agree tightly with each other (and with hypot(50, 800))
        assert per_candidate["dist_m"].std() < 5.0
        assert per_candidate["dist_m"].mean() == pytest.approx(math.hypot(50.0, 800.0), abs=5.0)

    def test_recovers_known_westward_offset(self):
        # -500m = -250 channels
        candidates, energies, bin_times, route_lats, route_lons = inject_offset_scenario(
            true_dlat_m=-30.0, true_dlon_m=-500.0, event_channels=[900, 1000, 1100]
        )
        dlat_m, dlon_m, _ = estimate_translation_offset(
            candidates, energies, bin_times, route_lats, route_lons,
            channel_spacing_m=DX_M, search_margin_m=600.0,
        )
        assert dlat_m == pytest.approx(-30.0, abs=5.0)
        assert dlon_m == pytest.approx(-500.0, abs=5.0)

    def test_too_narrow_search_margin_misses_the_true_channel(self):
        # offset is 800m east == 400 channels; a 50m margin (25 channels) can't
        # reach back that far from the naive (offset-biased) guess, so the
        # detected channel stays near the naive guess instead of the true one
        candidates, energies, bin_times, route_lats, route_lons = inject_offset_scenario(
            true_dlat_m=0.0, true_dlon_m=800.0, event_channels=[1000]
        )
        dlat_m, dlon_m, _ = estimate_translation_offset(
            candidates, energies, bin_times, route_lats, route_lons,
            channel_spacing_m=DX_M, search_margin_m=50.0,
        )
        assert dlon_m != pytest.approx(800.0, abs=5.0)


class TestMarginSensitivity:
    def test_plateaus_once_margin_covers_the_true_offset(self):
        # offset is 600m == 300 channels; margin must be >= that to recover it
        candidates, energies, bin_times, route_lats, route_lons = inject_offset_scenario(
            true_dlat_m=0.0, true_dlon_m=600.0, event_channels=[700, 900, 1100, 1300]
        )
        result = margin_sensitivity(
            candidates, energies, bin_times, route_lats, route_lons,
            channel_spacing_m=DX_M, margins_m=[100, 300, 700, 1000],
        )
        wide = result[result["margin_m"] >= 700]
        assert wide["dist_m"].std() < 5.0
        assert wide["dist_m"].iloc[-1] == pytest.approx(600.0, abs=5.0)
