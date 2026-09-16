import numpy as np
import pandas as pd
import pytest

from das_utils.image import build_image


def _write_files(tmp_path, n_files=3, n_samples=200, n_channels=5, seed=0):
    rng = np.random.default_rng(seed)
    d = tmp_path / "das"
    d.mkdir()
    indexed = []
    for i in range(n_files):
        arr = rng.standard_normal((n_samples, n_channels)).astype(np.float32) * 10
        p = d / f"{8:02d}{0:02d}{i * 10:02d}.npy"
        np.save(p, arr.astype(np.int16))
        ts = pd.Timestamp(f"2025-01-01 08:00:{i * 10:02d}")
        indexed.append((ts, str(p)))
    return indexed


class TestBuildImage:
    def test_rms_shape_and_cache(self, tmp_path):
        indexed = _write_files(tmp_path)
        cache = tmp_path / "img.npz"
        img, times = build_image(indexed, reducer="rms", cache_path=str(cache))
        assert img.shape == (3, 5)
        assert len(times) == 3
        assert cache.exists()
        # cache hit (same count + tag) returns without recompute
        img2, _ = build_image(indexed, reducer="rms", cache_path=str(cache))
        assert np.allclose(img, img2)

    def test_reducer_changes_values(self, tmp_path):
        indexed = _write_files(tmp_path)
        rms, _ = build_image(indexed, reducer="rms")
        std, _ = build_image(indexed, reducer="std")
        # for zero-mean noise, rms and std differ only by sqrt factor for large n; both positive
        assert np.all(rms > 0) and np.all(std > 0)
        assert not np.allclose(rms, std)

    def test_bad_reducer_raises(self, tmp_path):
        indexed = _write_files(tmp_path)
        with pytest.raises(ValueError):
            build_image(indexed, reducer="nope")
