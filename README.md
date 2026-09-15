# das-utils

Reusable algorithms and helpers for working with Distributed Acoustic Sensing (DAS)
data: loading raw strain files and channel-position tables, reducing raw data to a
compressed per-channel energy image, and correlating DAS against an independent
positioning signal (AIS ship traffic, so far) to find and correct geographic
mis-registration in a route's channel-position mapping.

Extracted from a real investigation in `peachfuzz11/nn` (branch `das-cable-offset`,
PR #3) that found and corrected a ~1.2km mis-registration in a DAS cable's geojson.
**Read `AGENTS.md` first** - it has the domain background, the method, and (most
usefully) the pitfalls that cost real time to find: near-stationary reference events
producing confidently-wrong answers, unbounded search windows walking off to unrelated
features, and a pandas datetime-resolution gotcha that silently scales timestamps by
the wrong factor.

## Install

```bash
uv sync
```

## Layout

```
src/das_utils/
  io.py          # load_channel_positions, index_das_files, positions_arrays
  geo.py         # local_xy, offset_m, nearest-channel KD-tree lookup, apply_translation
  energy.py      # bandpass_energy_image - streamed + cached raw-file reduction
  ais.py         # fast AIS extraction/filtering near a route
  correlate.py   # the AIS-DAS offset-correlation recipe
tests/           # synthetic recovery tests (inject a known offset, verify it's found)
```

## Quick example

```python
from das_utils import (
    load_channel_positions, positions_arrays, index_das_files,
    bandpass_energy_image, extract_time_window, filter_near_positions,
    estimate_translation_offset, margin_sensitivity, apply_translation,
    positions_to_geojson_linestring,
)

cable = load_channel_positions("route.json")
ids, lats, lons = positions_arrays(cable)
dx = cable["metadata"]["dx"]

files = index_das_files(["das_08_12", "das_12_16"], date="2025-11-22")
energies, bin_times = bandpass_energy_image(
    files, fs=cable["metadata"]["fs"], cache_path="energy_cache.npz"
)

ais = extract_time_window("aisdk-2025-11-22.csv", bin_times[0], bin_times[-1])
near = filter_near_positions(ais, ids, lats, lons, max_dist_m=2000.0)

# check result stability across search margins before trusting one
print(margin_sensitivity(near, energies, bin_times, lats, lons, dx, margins_m=[500, 1000, 1500, 2000]))

dlat_m, dlon_m, per_candidate = estimate_translation_offset(
    near, energies, bin_times, lats, lons, dx, search_margin_m=1500.0
)

corrected = apply_translation(cable["positions"], dlat_m, dlon_m)
geojson = positions_to_geojson_linestring(corrected, name="route")
```

## Development

```bash
uv run pytest
uv run ruff check .
```
