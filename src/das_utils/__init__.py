from das_utils.ais import extract_time_window, filter_near_positions
from das_utils.correlate import (
    closest_approach_offset,
    estimate_translation_offset,
    margin_sensitivity,
    to_epoch_seconds,
)
from das_utils.energy import bandpass_energy_image
from das_utils.geo import (
    apply_translation,
    build_nearest_tree,
    local_xy,
    nearest_index,
    offset_m,
    positions_to_geojson_linestring,
)
from das_utils.io import index_das_files, load_channel_positions, positions_arrays

__all__ = [
    "apply_translation",
    "bandpass_energy_image",
    "build_nearest_tree",
    "closest_approach_offset",
    "estimate_translation_offset",
    "extract_time_window",
    "filter_near_positions",
    "index_das_files",
    "load_channel_positions",
    "local_xy",
    "margin_sensitivity",
    "nearest_index",
    "offset_m",
    "positions_arrays",
    "positions_to_geojson_linestring",
    "to_epoch_seconds",
]


def main() -> None:
    print("das-utils: import the package for its functions - see AGENTS.md for the recipes.")
