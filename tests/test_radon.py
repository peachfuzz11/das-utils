"""Tests for the linear Radon (slant-stack) moving-source detector."""

import numpy as np
import pandas as pd
import pytest

from das_utils.radon import linear_radon_stack, radon_detect_stream, radon_score

FS = 400.0
N_CHANNELS = 200
N_SAMPLES = 4000  # 10s at 400Hz


def make_moving_source_patch(n_time=30, n_channels=21, slowness=0.5, t0=10, amplitude=20.0, seed=0):
    rng = np.random.default_rng(seed)
    img = rng.normal(1.0, 0.2, size=(n_time, n_channels)) ** 2
    for k in range(n_channels):
        t = t0 + round(k * slowness)
        if 0 <= t < n_time:
            img[t, k] += amplitude
    return img


class TestLinearRadonStack:
    def test_shape(self):
        img = np.zeros((10, 5))
        stack = linear_radon_stack(img, np.array([0.0, 0.5, 1.0]))
        assert stack.shape == (3, 10)

    def test_flat_line_scores_highest_at_zero_slowness(self):
        img = np.zeros((10, 5))
        img[4, :] = 10.0  # a simultaneous, fiber-wide event: a horizontal line
        stack = linear_radon_stack(img, np.array([-1.0, 0.0, 1.0]))
        si, t0 = np.unravel_index(np.argmax(stack), stack.shape)
        assert si == 1  # zero slowness wins for a purely horizontal feature


class TestRadonScore:
    def test_recovers_known_slowness_and_intercept(self):
        img = make_moving_source_patch(slowness=0.5, t0=10)
        slownesses = np.linspace(-1.0, 1.0, 21)
        score, s, t0 = radon_score(img, slownesses)
        assert s == pytest.approx(0.5)
        assert t0 == 10
        assert score > 5.0

    def test_recovers_negative_slowness(self):
        img = make_moving_source_patch(slowness=-0.7, t0=15)
        slownesses = np.linspace(-1.0, 1.0, 21)
        _, s, _ = radon_score(img, slownesses)
        assert s == pytest.approx(-0.7, abs=0.15)

    def test_pure_noise_gives_low_score(self):
        rng = np.random.default_rng(1)
        img = rng.normal(1.0, 0.2, size=(30, 21)) ** 2
        slownesses = np.linspace(-1.0, 1.0, 21)
        score, _, _ = radon_score(img, slownesses)
        assert score < 3.0


class TestRadonDetectStream:
    def test_flags_channel_with_moving_source(self, tmp_path):
        rng = np.random.default_rng(2)
        data = rng.normal(0, 1.0, size=(N_SAMPLES, N_CHANNELS)).astype(np.int16)
        # Inject a coherent, sequentially-delayed pulse across a span of channels -
        # a crude discrete stand-in for a moving source's wavefront.
        target = 100
        data = data.astype(np.float64)
        for k in range(-10, 11):
            c = target + k
            delay_samples = int(round(k * 15))  # small time skew per channel = slant
            center = N_SAMPLES // 2 + delay_samples
            if 0 <= c < N_CHANNELS and 0 <= center < N_SAMPLES:
                t = np.arange(N_SAMPLES)
                data[:, c] += 30 * np.exp(-((t - center) ** 2) / (2 * 20**2))
        data = data.astype(np.int16)
        f0 = tmp_path / "000000.npy"
        np.save(f0, data)
        indexed = [(pd.Timestamp("2025-01-01 00:00:00"), str(f0))]

        slownesses = np.linspace(-2.0, 2.0, 11)
        df = radon_detect_stream(
            indexed, fs=FS, step=10, patch_radius=15, slownesses=slownesses,
            bin_s=0.1, score_ratio_threshold=3.0, progress_every=0,
        )
        triggered_channels = set(df.loc[df["triggered"], "channel"])
        assert any(abs(c - target) <= 15 for c in triggered_channels)

    def test_quiet_data_rarely_triggers(self, tmp_path):
        rng = np.random.default_rng(3)
        f0 = tmp_path / "000000.npy"
        np.save(f0, rng.normal(0, 1.0, size=(N_SAMPLES, N_CHANNELS)).astype(np.int16))
        indexed = [(pd.Timestamp("2025-01-01 00:00:00"), str(f0))]
        slownesses = np.linspace(-2.0, 2.0, 11)
        df = radon_detect_stream(
            indexed, fs=FS, step=10, patch_radius=15, slownesses=slownesses,
            bin_s=0.1, progress_every=0,
        )
        assert df["triggered"].mean() < 0.1

    def test_output_has_expected_columns(self, tmp_path):
        rng = np.random.default_rng(4)
        f0 = tmp_path / "000000.npy"
        np.save(f0, rng.normal(0, 1.0, size=(N_SAMPLES, N_CHANNELS)).astype(np.int16))
        indexed = [(pd.Timestamp("2025-01-01 00:00:00"), str(f0))]
        df = radon_detect_stream(
            indexed, fs=FS, step=10, patch_radius=15, slownesses=np.linspace(-1, 1, 5),
            bin_s=0.1, progress_every=0,
        )
        assert set(df.columns) == {"file", "timestamp", "time_s", "channel", "score", "triggered"}
