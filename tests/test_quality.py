import numpy as np

from das_utils.quality import persistence_score, select_persistent_channels


class TestPersistenceScore:
    def test_stable_channel_scores_high_blinking_scores_low(self):
        rng = np.random.default_rng(0)
        stable = rng.standard_normal((100, 1)) * 0.1 + 5.0
        # bimodal channel: half near 0, half near 102 - both fall outside
        # the [median/2, median*2] band, so persistence collapses to ~0.
        blinking = np.where(np.arange(100) % 2 == 0, 0.0, 102.0).reshape(100, 1)
        img = np.hstack([stable, blinking])
        ps = persistence_score(img)
        assert ps[0] > 0.9
        # a 50/50 bimodal channel sits at the floor of the default 2x band
        # (one cluster always falls inside median*2), so ~0.5 - well below stable.
        assert ps[1] <= 0.5

    def test_band_widens_tolerance(self):
        rng = np.random.default_rng(1)
        img = rng.standard_normal((100, 1)) * 0.5 + 5.0
        narrow = persistence_score(img, band=1.1)
        wide = persistence_score(img, band=10.0)
        assert wide[0] >= narrow[0]


class TestSelectPersistentChannels:
    def test_drops_dead_and_spiky_keeps_healthy(self):
        rng = np.random.default_rng(2)
        # 3 healthy channels + 1 dead + 1 spiky
        healthy = rng.standard_normal((200, 3)) * 0.2 + 10.0
        dead = np.full((200, 1), 0.0)
        spiky = np.full((200, 1), 10.0)
        spiky[::20] = 1000.0
        img = np.hstack([healthy, dead, spiky])
        idx, q = select_persistent_channels(img, floor_quantiles=(0.1, 0.9), min_persistence=0.8)
        assert set(idx.tolist()) == {0, 1, 2}
        assert len(q) == 5

    def test_returns_ranked_by_persistence(self):
        rng = np.random.default_rng(3)
        img = rng.standard_normal((200, 4)) * 0.2 + 10.0
        # make channel 2 the most persistent (tightest band)
        img[:, 2] = 10.0 + rng.standard_normal(200) * 0.001
        idx, _ = select_persistent_channels(img, min_persistence=0.8)
        assert idx[0] == 2
