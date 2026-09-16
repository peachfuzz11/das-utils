import numpy as np
import pytest

from das_utils.stats import channel_stats, temporal_profile


class TestChannelStats:
    def test_dropout_and_saturation_flagged(self):
        # channel 0: dead (all tiny), channel 1: normal, channel 2: pegged at ADC ceiling
        rng = np.random.default_rng(0)
        img = rng.standard_normal((100, 3)) + 10
        img[:, 0] = 1e-6  # dead
        img[:, 2] = 40000.0  # above int16 full-scale (32767) -> saturated
        s = channel_stats(img, int16_bits=16)
        assert s["dropout_frac"][0] > 0.9
        assert s["saturation_frac"][2] > 0.9
        assert 0.0 <= s["saturation_frac"][1] < 0.1

    def test_dynamic_range_near_one_for_constant_channel(self):
        img = np.full((50, 2), 7.0)
        s = channel_stats(img)
        assert s["dynamic_range"][0] == pytest.approx(1.0, rel=1e-6)

    def test_temporal_profile_shape(self):
        img = np.ones((40, 5))
        prof = temporal_profile(img)
        assert prof.shape == (40,)
        assert np.allclose(prof, 1.0)
