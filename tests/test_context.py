import numpy as np
import pytest

from das_utils.channels import measure_correlation, pair_correlation
from das_utils.context import TargetSelector, context_window, greedy_decorrelated
from das_utils.quality import measure_noise


class TestGreedyDecorrelated:
    def test_redundant_pair_dropped(self):
        corr = np.array(
            [
                [1.0, 0.95, 0.0, 0.0],
                [0.95, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.1],
                [0.0, 0.0, 0.1, 1.0],
            ]
        )
        sel = greedy_decorrelated(corr, np.array([0, 1, 2, 3]), max_corr=0.5)
        assert sel.tolist() == [0, 2, 3]

    def test_min_sep_enforced(self):
        corr = np.eye(4)
        sel = greedy_decorrelated(corr, np.array([0, 1, 2, 10]), max_corr=0.9, min_sep=5)
        assert sel.tolist() == [0, 10]

    def test_max_corr_adapts_beyond_min_sep(self):
        corr = np.array([[1.0, 0.95], [0.95, 1.0]])
        sel = greedy_decorrelated(corr, np.array([0, 5]), max_corr=0.5, min_sep=3)
        assert sel.tolist() == [0]


class TestMeasureNoise:
    def test_dead_channel_scores_zero(self):
        rng = np.random.default_rng(0)
        img = rng.standard_normal((200, 3)) * 0.1 + 10
        img[:, 1] = 0.0  # dead
        score = measure_noise(img)
        assert score[1] < 0.1
        assert score[0] > 0.5

    def test_saturated_channel_scores_zero(self):
        rng = np.random.default_rng(0)
        img = rng.standard_normal((200, 3)) * 0.1 + 10
        img[:, 2] = 40000.0
        score = measure_noise(img, int16_bits=16)
        assert score[2] < 0.1
        assert score[0] > 0.5

    def test_stable_scores_higher_than_blinking(self):
        rng = np.random.default_rng(0)
        stable = rng.standard_normal((100, 1)) * 0.1 + 5.0
        blinking = np.where(np.arange(100) % 2 == 0, 0.0, 102.0).reshape(100, 1)
        img = np.hstack([stable, blinking])
        score = measure_noise(img)
        assert score[0] > score[1]


class TestMeasureCorrelation:
    def test_matrix_matches_pair(self):
        rng = np.random.default_rng(0)
        img = rng.standard_normal((2000, 4))
        full = measure_correlation(img)
        rho = pair_correlation(img, 0, 1)
        assert rho == pytest.approx(full[0, 1], abs=1e-9)
        assert full.shape == (4, 4)
        assert np.allclose(np.diag(full), 1.0)

    def test_pair_correlation_known_rho(self):
        rng = np.random.default_rng(1)
        x = rng.standard_normal(3000)
        y = 0.7 * x + np.sqrt(1 - 0.49) * rng.standard_normal(3000)
        img = np.stack([x, y], axis=1)
        assert pair_correlation(img, 0, 1) == pytest.approx(0.7, abs=0.03)

    def test_constant_channel_returns_zero(self):
        img = np.ones((100, 2))
        img[:, 1] = np.arange(100, dtype=float)
        assert pair_correlation(img, 0, 1) == 0.0


class TestTargetSelector:
    def _chain(self, n=4000, m=60, rho=0.6, seed=0):
        rng = np.random.default_rng(seed)
        img = np.empty((n, m))
        img[:, 0] = rng.standard_normal(n)
        for k in range(1, m):
            img[:, k] = rho * img[:, k - 1] + rng.standard_normal(n)
        return img

    def test_select_min_sep_and_decorrelated(self):
        img = self._chain()
        noise = measure_noise(img)
        corr = measure_correlation(img)
        sel = TargetSelector(noise, corr).select(min_sep=10, max_corr=0.5)
        assert np.all(np.diff(sel) >= 10)
        assert len(sel) > 1

    def test_select_prefers_high_noise(self):
        # channel 5 has artificially high persistence
        img = self._chain(m=20, seed=3)
        img[:, 5] = 10.0 + np.random.default_rng(9).standard_normal(4000) * 0.001
        noise = measure_noise(img)
        corr = measure_correlation(img)
        sel = TargetSelector(noise, corr).select(min_sep=3, max_corr=0.5)
        assert 5 in sel.tolist()

    def test_max_sep_fills_gaps(self):
        img = self._chain(n=2000, m=100)
        noise = measure_noise(img)
        corr = measure_correlation(img)
        # without max_sep: gaps can be large
        sel_no = TargetSelector(noise, corr).select(min_sep=20, max_corr=0.5)
        # with max_sep=30: all gaps <= 30
        sel_gap = TargetSelector(noise, corr).select(min_sep=20, max_sep=30, max_corr=0.5)
        assert len(sel_gap) >= len(sel_no)
        diffs = np.diff(sel_gap)
        assert np.all(diffs <= 30)

    def test_max_sep_fills_large_gap(self):
        # 200 channels, noise is high only at 0 and 199
        img = self._chain(n=2000, m=200, seed=5)
        noise = measure_noise(img)
        noise[:198] = 0.0
        noise[0] = 1.0
        noise[199] = 1.0
        corr = measure_correlation(img)
        sel = TargetSelector(noise, corr).select(min_sep=0, max_sep=50, max_corr=0.99)
        # gap was 199 > 50, so gap-filling must add channels in between
        diffs = np.diff(sel)
        assert np.all(diffs <= 50)

    def test_max_sep_below_min_sep_raises(self):
        noise = np.ones(10)
        corr = np.eye(10)
        with pytest.raises(ValueError):
            TargetSelector(noise, corr).select(min_sep=10, max_sep=5)

    def test_min_noise_filters_bad_channels(self):
        img = self._chain(n=1000, m=20)
        noise = measure_noise(img)
        noise[5] = 0.0  # mark channel 5 as bad
        corr = measure_correlation(img)
        sel = TargetSelector(noise, corr).select(min_sep=3, max_corr=0.5, min_noise=0.1)
        assert 5 not in sel.tolist()

    def test_max_count_keeps_highest_noise(self):
        img = self._chain(n=2000, m=40)
        noise = measure_noise(img)
        corr = measure_correlation(img)
        sel = TargetSelector(noise, corr).select(min_sep=5, max_corr=0.5, max_count=3)
        assert len(sel) <= 3

    def test_correlation_between(self):
        img = self._chain(n=2000, m=10)
        corr = measure_correlation(img)
        sel = TargetSelector(np.ones(10), corr)
        assert sel.correlation_between(0, 1) == pytest.approx(corr[0, 1])

    def test_noise_of(self):
        noise = np.array([0.1, 0.5, 0.9])
        sel = TargetSelector(noise, np.eye(3))
        assert sel.noise_of(2) == 0.9

    def test_context_window_filters_by_noise(self):
        noise = np.zeros(100)
        noise[48:52] = 0.8
        noise[55] = 0.8
        sel = TargetSelector(noise, np.eye(100))
        cols = sel.context_window(target=50, radius=10, min_noise=0.1)
        assert 50 in cols.tolist()
        assert 40 not in cols.tolist()  # noise[40]=0 < 0.1
        assert 55 in cols.tolist()

    def test_context_window_no_filter(self):
        sel = TargetSelector(np.ones(100), np.eye(100))
        cols = sel.context_window(target=50, radius=5)
        assert cols.tolist() == list(range(45, 56))

    def test_plan_returns_pairs(self):
        img = self._chain(n=1000, m=30)
        noise = measure_noise(img)
        corr = measure_correlation(img)
        plan = TargetSelector(noise, corr).plan(radius=4, min_sep=10, max_corr=0.5)
        for target, cols in plan:
            assert target in cols.tolist()
            assert cols.min() >= max(0, target - 4)
            assert cols.max() <= min(29, target + 4)
        targets = [t for t, _ in plan]
        assert np.all(np.diff(targets) >= 10)


class TestContextWindowStandalone:
    def test_full_window(self):
        cols = context_window(target=50, radius=10, n_channels=100)
        assert cols.tolist() == list(range(40, 61))

    def test_filters_to_persistent(self):
        persistent = np.array([48, 49, 50, 51, 55])
        cols = context_window(target=50, radius=5, n_channels=100, persistent_idx=persistent)
        assert cols.tolist() == [48, 49, 50, 51, 55]
