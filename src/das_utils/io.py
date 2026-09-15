"""Loading DAS channel-position tables and indexing raw strain files.

Two format assumptions baked in here, both common but not universal - adjust
if your deployment differs:
- Channel positions come as JSON: ``{"positions": {"<channel_id>": {"lat",
  "lon", "depth"}, ...}, "metadata": {"fs", "dx", "gl", ...}}``.
- Raw files are named ``HHMMSS.<ext>`` (no date - you supply it), one file
  per short time window, shape ``(n_samples, n_channels)``.
"""

import glob
import json
import os

import numpy as np
import pandas as pd


def load_channel_positions(path: str) -> dict:
    """Channel positions/metadata, keyed by int channel id.

    Returns ``{"positions": {id: {"lat","lon","depth"}}, "metadata": {...}}``.
    """
    raw = json.load(open(path))
    positions = {int(k): v for k, v in raw["positions"].items()}
    return {"positions": positions, "metadata": raw["metadata"]}


def positions_arrays(cable: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A cable's positions dict as parallel (ids, lats, lons) arrays, sorted by id.

    Note the position table and the raw recorded channel count can differ
    (some channels at either end may be unrecorded) - don't assume
    ``len(ids) == n_recorded_channels`` without checking; see the module
    docstring in ``energy.py`` and AGENTS.md for why this matters.
    """
    ids = np.array(sorted(cable["positions"]))
    lats = np.array([cable["positions"][i]["lat"] for i in ids])
    lons = np.array([cable["positions"][i]["lon"] for i in ids])
    return ids, lats, lons


def index_das_files(
    directories: list[str], date: str, pattern: str = "*.npy"
) -> list[tuple[pd.Timestamp, str]]:
    """(timestamp, path) for every *complete* file matching ``pattern`` under ``directories``, sorted.

    Filenames are expected to encode only ``HHMMSS`` (no date, hence the
    ``date`` argument - construct real timestamps early so every later join
    is on actual ``Timestamp`` values, not string/positional matching that
    silently breaks on gaps).

    A file still being written, or truncated by a transfer error, loads
    with the wrong element count. Rather than trying to partially read such
    a file, this drops any whose size doesn't match the majority (modal)
    size across the whole set, and reports how many were dropped.
    """
    files = []
    for d in directories:
        files.extend(glob.glob(os.path.join(d, pattern)))
    if not files:
        raise FileNotFoundError(f"no files matching {pattern!r} under {directories}")

    sizes = [os.path.getsize(f) for f in files]
    expected_size = max(set(sizes), key=sizes.count)
    complete = [f for f in files if os.path.getsize(f) == expected_size]
    dropped = len(files) - len(complete)
    if dropped:
        print(f"index_das_files: dropping {dropped} truncated/incomplete file(s)")

    indexed = []
    for f in complete:
        hhmmss = os.path.splitext(os.path.basename(f))[0]
        ts = pd.Timestamp(f"{date} {hhmmss[:2]}:{hhmmss[2:4]}:{hhmmss[4:6]}")
        indexed.append((ts, f))
    indexed.sort(key=lambda p: p[0])
    return indexed
