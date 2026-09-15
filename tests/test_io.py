import json

import numpy as np
import pytest

from das_utils.io import index_das_files, load_channel_positions, positions_arrays


class TestLoadChannelPositions:
    def test_keys_become_ints_and_metadata_preserved(self, tmp_path):
        raw = {
            "positions": {
                "1": {"lat": 55.34, "lon": 11.00, "depth": 10.0},
                "2": {"lat": 55.35, "lon": 11.01, "depth": 11.0},
            },
            "metadata": {"fs": 400.0, "dx": 2.04, "gl": 4.08},
        }
        path = tmp_path / "cable.json"
        path.write_text(json.dumps(raw))

        cable = load_channel_positions(str(path))
        assert set(cable["positions"]) == {1, 2}
        assert cable["metadata"]["fs"] == 400.0

    def test_positions_arrays_sorted_by_id(self, tmp_path):
        raw = {
            "positions": {"3": {"lat": 55.36, "lon": 11.02}, "1": {"lat": 55.34, "lon": 11.00}},
            "metadata": {},
        }
        path = tmp_path / "cable.json"
        path.write_text(json.dumps(raw))
        cable = load_channel_positions(str(path))
        ids, lats, lons = positions_arrays(cable)
        assert list(ids) == [1, 3]
        assert lats[0] == pytest.approx(55.34)


class TestIndexDasFiles:
    def test_sorts_by_time_and_drops_truncated_files(self, tmp_path):
        d = tmp_path / "das"
        d.mkdir()
        full = np.zeros((10, 5), dtype=np.int16)
        np.save(d / "080010.npy", full)
        np.save(d / "080000.npy", full)
        np.save(d / "080020.npy", full[:3])  # truncated / smaller file

        indexed = index_das_files([str(d)], date="2025-01-01")

        assert [ts.strftime("%H%M%S") for ts, _ in indexed] == ["080000", "080010"]

    def test_raises_if_no_files_match(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            index_das_files([str(tmp_path)], date="2025-01-01")
