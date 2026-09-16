"""Tests for the Morlet continuous-wavelet power feature."""

import numpy as np
import pytest

from das_utils.wavelet import morlet_cwt_power, morlet_cwt_power_series

FS = 400.0
N_SAMPLES = 4000


def tone(freq, amplitude=5.0, n_samples=N_SAMPLES, fs=FS, seed=0, noise=0.1):
    t = np.arange(n_samples) / fs
    rng = np.random.default_rng(seed)
    return amplitude * np.sin(2 * np.pi * freq * t) + rng.normal(0, noise, n_samples)


class TestMorletCwtPower:
    def test_isolates_true_tone_frequency(self):
        data = np.column_stack([tone(10.0, seed=0), tone(80.0, seed=1)])
        freqs = np.array([5.0, 10.0, 20.0, 40.0, 80.0, 150.0])
        power = morlet_cwt_power(data, FS, freqs)
        assert np.argmax(power[:, 0]) == 1  # 10Hz row
        assert np.argmax(power[:, 1]) == 4  # 80Hz row

    def test_noise_only_channel_has_low_power_everywhere(self):
        rng = np.random.default_rng(2)
        data = rng.normal(0, 0.1, size=(N_SAMPLES, 1))
        freqs = np.array([5.0, 20.0, 80.0])
        power = morlet_cwt_power(data, FS, freqs)
        assert np.all(power < 1.0)

    def test_output_shape(self):
        data = np.zeros((N_SAMPLES, 4))
        freqs = np.array([5.0, 10.0, 20.0])
        power = morlet_cwt_power(data, FS, freqs)
        assert power.shape == (3, 4)

    def test_stronger_tone_gives_more_power(self):
        weak = tone(10.0, amplitude=1.0, seed=3)
        strong = tone(10.0, amplitude=10.0, seed=3)
        freqs = np.array([10.0])
        p_weak = morlet_cwt_power(weak[:, None], FS, freqs)
        p_strong = morlet_cwt_power(strong[:, None], FS, freqs)
        assert p_strong[0, 0] > p_weak[0, 0]
        assert p_strong[0, 0] == pytest.approx(p_weak[0, 0] * 100, rel=0.2)  # power ~ amplitude^2


class TestMorletCwtPowerSeries:
    def test_matches_time_average_of_scalar_version(self):
        data = np.column_stack([tone(10.0, seed=0), tone(80.0, seed=1)])
        series = morlet_cwt_power_series(data, FS, 10.0)
        scalar = morlet_cwt_power(data, FS, np.array([10.0]))[0]
        assert series.mean(axis=0) == pytest.approx(scalar, rel=0.05)

    def test_selects_only_requested_columns(self):
        data = np.column_stack([tone(10.0, seed=0), tone(80.0, seed=1), tone(10.0, seed=2)])
        series = morlet_cwt_power_series(data, FS, 10.0, cols=np.array([0, 2]))
        assert series.shape == (N_SAMPLES, 2)

    def test_power_rises_when_tone_is_present(self):
        n = N_SAMPLES
        t = np.arange(n) / FS
        data = np.zeros((n, 1))
        data[n // 3 : 2 * n // 3, 0] = 5 * np.sin(2 * np.pi * 10 * t[: n // 3])
        series = morlet_cwt_power_series(data, FS, 10.0)
        quiet_power = series[: n // 4, 0].mean()
        loud_power = series[n // 2 - 50 : n // 2 + 50, 0].mean()
        assert loud_power > quiet_power * 10
