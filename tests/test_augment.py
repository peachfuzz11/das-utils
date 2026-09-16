import numpy as np
import pytest

from das_utils.augment import (
    add_noise,
    channel_dropout,
    crop,
    decimate,
    detrend,
    normalize,
    smooth_spatial,
    smooth_temporal,
    taper,
    time_mask,
    time_shift,
)


class TestNormalize:
    def test_zscore_zero_mean_unit_std(self):
        rng = np.random.default_rng(0)
        img = rng.standard_normal((100, 3)) * 5 + 7
        z = normalize(img, how="zscore")
        assert np.allclose(z.mean(axis=0), 0, atol=1e-9)
        assert np.allclose(z.std(axis=0), 1, atol=1e-9)

    def test_minmax_in_unit_interval(self):
        img = np.array([[0.0, 5.0], [10.0, 15.0]])
        m = normalize(img, how="minmax", axis=0)
        assert m.min() >= 0 and m.max() <= 1

    def test_constant_channel_does_not_explode(self):
        img = np.full((10, 2), 4.0)
        z = normalize(img, how="zscore")
        assert np.all(np.isfinite(z))


class TestDetrend:
    def test_removes_linear_trend(self):
        t = np.arange(200)
        img = (3.0 * t + 1.0).reshape(-1, 1)
        d = detrend(img, axis=0)
        assert np.allclose(d, 0, atol=1e-8)


class TestTaper:
    def test_edges_attenuated(self):
        img = np.ones((100, 3))
        out = taper(img, fraction=0.2, axis=0)
        assert out[0, 0] < 0.5 and out[-1, 0] < 0.5
        assert out[50, 0] == pytest.approx(1.0)


class TestSmoothing:
    def test_smooth_spatial_reduces_variance(self):
        rng = np.random.default_rng(0)
        img = rng.standard_normal((10, 50))
        s = smooth_spatial(img, k=3)
        assert s.std() < img.std()

    def test_smooth_temporal_reduces_variance(self):
        rng = np.random.default_rng(1)
        img = rng.standard_normal((50, 10))
        s = smooth_temporal(img, k=3)
        assert s.std() < img.std()


class TestDecimate:
    def test_mean_pools_by_factor(self):
        img = np.arange(12, dtype=float).reshape(6, 2)
        d = decimate(img, factor=2, axis=0)
        assert d.shape == (3, 2)
        assert d[0, 0] == pytest.approx(1.0)  # mean(0,1)
        assert d[2, 0] == pytest.approx(9.0)  # mean(8,9)

    def test_factor_one_returns_input(self):
        img = np.ones((4, 4))
        assert np.array_equal(decimate(img, factor=1), img)


class TestCrop:
    def test_slices_rows_and_cols(self):
        img = np.arange(20).reshape(4, 5)
        assert crop(img, 1, 3, 1, 4).tolist() == [[6, 7, 8], [11, 12, 13]]


class TestAugmentations:
    def test_add_noise_is_seeded(self):
        img = np.zeros((50, 3))
        a = add_noise(img, 1.0, seed=0)
        b = add_noise(img, 1.0, seed=0)
        assert np.array_equal(a, b)
        assert a.std() == pytest.approx(1.0, abs=0.05)

    def test_time_shift_zero_fills(self):
        img = np.ones((10, 2))
        s = time_shift(img, 3, axis=0)
        assert np.all(s[:3] == 0)
        assert np.all(s[3:] == 1)
        # zero shift is identity
        assert np.array_equal(time_shift(img, 0), img)

    def test_channel_dropout_zeros_fraction(self):
        img = np.ones((20, 100))
        out = channel_dropout(img, p=0.2, seed=0)
        zeroed = (out == 0).all(axis=0).sum()
        assert 10 <= zeroed <= 30  # roughly 20 of 100

    def test_time_mask_blanks_one_block(self):
        img = np.ones((30, 4))
        out = time_mask(img, width=5, seed=0)
        zero_rows = (out == 0).all(axis=1).sum()
        assert zero_rows == 5
