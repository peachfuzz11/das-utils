"""Fast extraction and filtering of AIS traffic near a DAS route.

Built against the Danish Maritime Authority's public ``aisdk-YYYY-MM-DD.csv``
format (``# Timestamp`` in ``DD/MM/YYYY HH:MM:SS``, ``Type of mobile``,
``MMSI``, ``Latitude``/``Longitude``, ``SOG`` in knots, ``Name``), but the
filtering step works on any dataframe with those column names - swap in your
own loader if your AIS source differs and reuse ``filter_near_positions``.
"""

import io
import subprocess

import numpy as np
import pandas as pd

from das_utils.geo import build_nearest_tree, nearest_index

VESSEL_CLASSES = ("Class A", "Class B")  # real ships; excludes base stations/AtoN/SAR aircraft etc.


def extract_time_window(
    ais_csv_path: str, start: pd.Timestamp, end: pd.Timestamp, date_format: str = "%d/%m/%Y %H:%M:%S"
) -> pd.DataFrame:
    """Rows of a large, time-sorted daily AIS dump within [start, end).

    These dumps are commonly multi-GB but sorted by time, so a cheap
    line-prefix ``grep`` for just the hour(s) needed is dramatically faster
    than paging the whole file through pandas - a few seconds instead of
    minutes, and avoids ever holding the full file in memory.

    Assumes every row's date matches ``start``'s date (fine for a window
    that doesn't cross midnight, which covers most DAS recording sessions;
    extend the hour-prefix loop over both dates if yours does).
    """
    prefix_date = start.strftime("%d/%m/%Y")
    hours = sorted({t.strftime("%H") for t in pd.date_range(start, end, freq="h")})

    with open(ais_csv_path) as src:
        header = src.readline()
    chunks = [header]
    for hour in hours:
        result = subprocess.run(
            ["grep", f"^{prefix_date} {hour}:", ais_csv_path], stdout=subprocess.PIPE, check=False
        )
        chunks.append(result.stdout.decode())

    df = pd.read_csv(io.StringIO("".join(chunks)), low_memory=False)
    df.columns = [c.lstrip("# ") for c in df.columns]
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], format=date_format)
    return df[df["Timestamp"].between(start, end, inclusive="left")].reset_index(drop=True)


def filter_near_positions(
    df: pd.DataFrame,
    ids: np.ndarray,
    lats: np.ndarray,
    lons: np.ndarray,
    max_dist_m: float = 2000.0,
    vessel_classes: tuple[str, ...] = VESSEL_CLASSES,
) -> pd.DataFrame:
    """AIS rows within ``max_dist_m`` of any of the given (ids, lats, lons) points.

    Adds ``DistToRoute_m`` (distance to the nearest given point) and
    ``NearestId`` (that point's id) columns. Also restricts to
    ``vessel_classes`` and drops duplicate (MMSI, Timestamp) rows, which are
    common in these dumps (the same report received by multiple base
    stations).
    """
    df = df[df["Type of mobile"].isin(vessel_classes)]
    df = df.drop_duplicates(subset=["MMSI", "Timestamp"]).reset_index(drop=True)

    tree, ref_lat, ref_lon = build_nearest_tree(lats, lons)
    dist, idx = nearest_index(tree, ref_lat, ref_lon, df["Latitude"].to_numpy(), df["Longitude"].to_numpy())
    df = df.copy()
    df["DistToRoute_m"] = dist
    df["NearestId"] = ids[idx]

    return df[df["DistToRoute_m"] <= max_dist_m].copy()
