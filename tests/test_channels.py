import numpy as np
import pytest

from das_utils.channels import (
    correlation_matrix,
    low_correlation_intervals,
    neighbor_correlation,
    sliding_neighbor_correlation,
)


def _correlated_pair(n=500, rho=0.8, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, 2))
    x[:, 1] = rho * x[:, 0] + np.sqrt(1 - rho**2) * x[:, 1]
    return x


class TestNeighborCorrelation:
    def test_known_pair_correlation_recovered(self):
        img = _correlated_pair(2000, rho=0.7, seed=1)
        c = neighbor_correlation(img, max_lag=1)
        assert c[0] == pytest.approx(0.7, abs=0.05)

    def test_independent_channels_near_zero(self):
        rng = np.random.default_rng(2)
        img = rng.standard_normal((2000, 2))
        c = neighbor_correlation(img, max_lag=1)
        assert abs(c[0]) < 0.06

    def test_decays_with_lag(self):
        # AR(1) chain: channel k = 0.6 * channel k-1 + noise -> corr decays ~0.6^lag
        rng = np.random.default_rng(3)
        n = 3000
        img = np.empty((n, 10))
        img[:, 0] = rng.standard_normal(n)
        for k in range(1, 10):
            img[:, k] = 0.6 * img[:, k - 1] + rng.standard_normal(n)
        c = neighbor_correlation(img, max_lag=5)
        assert c[0] > c[4]
        assert c[0] > 0.4 and c[4] < 0.2


class TestCorrelationMatrix:
    def test_diagonal_is_one(self):
        rng = np.random.default_rng(0)
        img = rng.standard_normal((1000, 4))
        m = correlation_matrix(img)
        assert m.shape == (4, 4)
        assert np.allclose(np.diag(m), 1.0, atol=1e-9)
        assert np.allclose(m, m.T)


class TestSlidingCorrelation:
    def test_detects_low_correlation_interval(self):
        # first half correlated, second half independent
        rng = np.random.default_rng(4)
        hi = _correlated_pair(200, rho=0.9, seed=10)
        lo = rng.standard_normal((200, 2))
        img = np.vstack([hi, lo])
        corr, centers = sliding_neighbor_correlation(img, window=100, step=50)
        assert len(corr) >= 2
        # earliest window (correlated half) should beat latest (independent half)
        assert corr[0] > corr[-1]

    def test_window_too_small_raises(self):
        with pytest.raises(ValueError):
            sliding_neighbor_correlation(np.zeros((10, 3)), window=1)


class TestLowCorrelationIntervals:
    def test_finds_contiguous_low_run(self):
        corr = np.array([0.9, 0.9, 0.2, 0.1, 0.2, 0.8, 0.1])
        centers = np.arange(len(corr)) * 10
        runs = low_correlation_intervals(corr, centers, max_corr=0.3, min_length=2)
        assert len(runs) == 1
        assert runs["length"].iloc[0] == 3
        assert runs["start_idx"].iloc[0] == 20
        assert runs["end_idx"].iloc[0] == 40

    def test_min_length_filters_short_runs(self):
        corr = np.array([0.2, 0.8, 0.2, 0.1])
        centers = np.arange(len(corr))
        runs = low_correlation_intervals(corr, centers, max_corr=0.3, min_length=2)
        # only one run of length 2 (indices 2,3)
        assert len(runs) == 1
        assert runs["length"].iloc[0] == 2
