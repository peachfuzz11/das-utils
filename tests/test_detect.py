"""Synthetic recovery tests for the stepped, temporal-baseline ship detector."""

import numpy as np
import pandas as pd
import pytest

from das_utils.detect import (
    channel_baseline,
    cluster_detections,
    common_mode_suppression,
    detect_file,
    detect_stream,
    spatial_corroboration,
    sta_lta_ratio,
    sta_lta_stream,
    step_channels,
)

FS = 400.0
N_CHANNELS = 800
N_SAMPLES = 4000  # 10s at 400Hz


def make_quiet_data(n_samples=N_SAMPLES, n_channels=N_CHANNELS, seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(0.0, 1.0, size=(n_samples, n_channels)).astype(np.int16)


def flat_baseline(n_channels=N_CHANNELS, floor=0.0, scale=1.0):
    return np.full(n_channels, floor), np.full(n_channels, scale)


def inject_ship(data, channel, amplitude=40.0, start_frac=0.05, end_frac=0.95, width=0):
    """Add a sustained mid-band tone on a small span of channels around `channel`."""
    n_samples = data.shape[0]
    t = np.arange(n_samples) / FS
    tone = amplitude * np.sin(2 * np.pi * 5.0 * t)
    lo, hi = int(start_frac * n_samples), int(end_frac * n_samples)
    mask = np.zeros(n_samples)
    mask[lo:hi] = 1.0
    out = data.astype(np.float64)
    for c in range(max(0, channel - width), min(data.shape[1], channel + width + 1)):
        out[:, c] += tone * mask
    return out.astype(np.int16)


class TestStepChannels:
    def test_includes_last_channel(self):
        cols = step_channels(800, 50)
        assert cols[-1] == 799
        assert cols[0] == 0

    def test_step_one_is_every_channel(self):
        cols = step_channels(10, 1)
        assert list(cols) == list(range(10))

    def test_rejects_nonpositive_step(self):
        with pytest.raises(ValueError):
            step_channels(10, 0)


class TestChannelBaseline:
    def test_floor_and_scale_match_reference_stats(self):
        rng = np.random.default_rng(0)
        img = rng.normal(5.0, 1.0, size=(200, 10))
        floor, scale = channel_baseline(img)
        assert floor == pytest.approx(np.median(img, axis=0))
        assert np.all(scale > 0)

    def test_constant_channel_gets_nonzero_scale(self):
        img = np.ones((50, 3)) * 7.0
        floor, scale = channel_baseline(img)
        assert np.all(floor == 7.0)
        assert np.all(scale > 0)


class TestDetectFile:
    def test_flags_channel_with_injected_ship(self):
        data = make_quiet_data()
        target = 400
        data = inject_ship(data, target)
        floor, scale = flat_baseline()
        df = detect_file(data, fs=FS, step=25, channel_floor=floor, channel_scale=scale, sub_windows=10)
        triggered_channels = set(df.loc[df["triggered"], "channel"])
        assert target in triggered_channels

    def test_quiet_data_rarely_triggers(self):
        data = make_quiet_data(seed=1)
        floor, scale = flat_baseline()
        df = detect_file(data, fs=FS, step=25, channel_floor=floor, channel_scale=scale, sub_windows=10)
        assert df["triggered"].mean() < 0.05

    def test_produces_n_responses_per_file(self):
        data = make_quiet_data()
        floor, scale = flat_baseline()
        df = detect_file(
            data, fs=FS, step=100, channel_floor=floor, channel_scale=scale, sub_windows=5
        )
        n_cols = len(step_channels(N_CHANNELS, 100))
        assert len(df) == 5 * n_cols

    def test_elevated_but_not_sustained_does_not_trigger(self):
        # a real ship holds a channel elevated for nearly the whole file; a
        # short blip covering under sustain_duration's worth of windows should not.
        data = make_quiet_data(seed=4)
        data = inject_ship(data, 400, start_frac=0.0, end_frac=0.2)  # only ~2/10 windows
        floor, scale = flat_baseline()
        df = detect_file(
            data, fs=FS, step=25, channel_floor=floor, channel_scale=scale,
            sustain_duration=0.9, sub_windows=10,
        )
        assert not df.loc[df["channel"] == 400, "triggered"].any()


class TestDetectStream:
    def test_merges_across_files_with_absolute_timestamps(self, tmp_path):
        data0 = inject_ship(make_quiet_data(seed=2), 300)
        data1 = make_quiet_data(seed=3)
        f0, f1 = tmp_path / "000000.npy", tmp_path / "000010.npy"
        np.save(f0, data0)
        np.save(f1, data1)
        indexed = [
            (pd.Timestamp("2025-01-01 00:00:00"), str(f0)),
            (pd.Timestamp("2025-01-01 00:00:10"), str(f1)),
        ]
        floor, scale = flat_baseline()
        df = detect_stream(
            indexed, fs=FS, step=25, channel_floor=floor, channel_scale=scale,
            sub_windows=5, progress_every=0,
        )
        assert set(df["file"]) == {str(f0), str(f1)}
        assert df["timestamp"].min() >= pd.Timestamp("2025-01-01 00:00:00")
        assert df["timestamp"].max() < pd.Timestamp("2025-01-01 00:00:20")


class TestCommonModeSuppression:
    def test_suppresses_fiber_wide_trigger(self):
        df = pd.DataFrame(
            {
                "file": ["f.npy"] * 4,
                "time_s": [0.5] * 4,
                "channel": [0, 25, 50, 75],
                "triggered": [True, True, True, True],
            }
        )
        out = common_mode_suppression(df, max_frac_triggered=0.5)
        assert not out["triggered"].any()

    def test_keeps_localized_trigger(self):
        df = pd.DataFrame(
            {
                "file": ["f.npy"] * 4,
                "time_s": [0.5] * 4,
                "channel": [0, 25, 50, 75],
                "triggered": [True, False, False, False],
            }
        )
        out = common_mode_suppression(df, max_frac_triggered=0.5)
        assert out["triggered"].tolist() == [True, False, False, False]

    def test_windows_are_independent_per_file(self):
        df = pd.DataFrame(
            {
                "file": ["f0.npy", "f0.npy", "f1.npy", "f1.npy", "f1.npy", "f1.npy"],
                "time_s": [0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
                "channel": [0, 25, 0, 25, 50, 75],
                "triggered": [True, True, True, False, False, False],
            }
        )
        out = common_mode_suppression(df, max_frac_triggered=0.5)
        assert out.loc[(out["file"] == "f1.npy") & (out["channel"] == 0), "triggered"].all()
        assert not out.loc[out["file"] == "f0.npy", "triggered"].any()


class TestSpatialCorroboration:
    def test_keeps_triggers_with_a_nearby_neighbor(self):
        df = pd.DataFrame(
            {
                "file": ["f.npy"] * 3,
                "timestamp": pd.to_datetime(
                    ["2025-01-01 00:00:00", "2025-01-01 00:00:01", "2025-01-01 00:00:00"]
                ),
                "channel": [400, 425, 700],
                "triggered": [True, True, True],
            }
        )
        out = spatial_corroboration(df, channel_gap=50, time_tol_s=5.0)
        assert list(out["triggered"]) == [True, True, False]

    def test_drops_isolated_single_channel_blip(self):
        df = pd.DataFrame(
            {
                "file": ["f.npy"],
                "timestamp": pd.to_datetime(["2025-01-01 00:00:00"]),
                "channel": [400],
                "triggered": [True],
            }
        )
        out = spatial_corroboration(df, channel_gap=50, time_tol_s=5.0)
        assert not out["triggered"].any()

    def test_neighbor_must_be_within_time_tolerance(self):
        df = pd.DataFrame(
            {
                "file": ["f.npy"] * 2,
                "timestamp": pd.to_datetime(["2025-01-01 00:00:00", "2025-01-01 00:10:00"]),
                "channel": [400, 425],
                "triggered": [True, True],
            }
        )
        out = spatial_corroboration(df, channel_gap=50, time_tol_s=5.0)
        assert not out["triggered"].any()

    def test_neighbors_across_files_do_not_corroborate(self):
        df = pd.DataFrame(
            {
                "file": ["f0.npy", "f1.npy"],
                "timestamp": pd.to_datetime(["2025-01-01 00:00:00", "2025-01-01 00:00:00"]),
                "channel": [400, 425],
                "triggered": [True, True],
            }
        )
        out = spatial_corroboration(df, channel_gap=50, time_tol_s=5.0)
        assert not out["triggered"].any()


class TestStaLtaRatio:
    def test_flat_signal_gives_ratio_near_one(self):
        energy = np.full((200, 3), 5.0)
        ratio = sta_lta_ratio(energy, short_bins=4, long_bins=40)
        valid = ratio[~np.isnan(ratio).any(axis=1)]
        assert valid == pytest.approx(1.0, abs=1e-6)

    def test_step_increase_spikes_ratio(self):
        energy = np.concatenate([np.full(100, 1.0), np.full(20, 50.0)])[:, None]
        ratio = sta_lta_ratio(energy, short_bins=4, long_bins=40)
        assert np.nanmax(ratio[100:120]) > 10.0

    def test_early_bins_without_history_are_nan(self):
        energy = np.ones((10, 1))
        ratio = sta_lta_ratio(energy, short_bins=4, long_bins=40)
        assert np.isnan(ratio[0, 0])


class TestStaLtaStream:
    def test_flags_channel_with_step_change(self, tmp_path):
        fs = FS
        n = N_SAMPLES
        rng = np.random.default_rng(5)
        data = rng.normal(0, 1.0, size=(n, N_CHANNELS)).astype(np.int16)
        target = 400
        t = np.arange(n) / fs
        tone_sig = 40 * np.sin(2 * np.pi * 5.0 * t)
        data = data.astype(np.float64)
        data[n // 4 :, target] += tone_sig[n // 4 :]
        data = data.astype(np.int16)
        f0 = tmp_path / "000000.npy"
        np.save(f0, data)
        indexed = [(pd.Timestamp("2025-01-01 00:00:00"), str(f0))]
        df = sta_lta_stream(
            indexed, fs=fs, step=25, bin_s=0.1, short_s=0.5, long_s=2.0,
            ratio_threshold=3.0, sustain_duration=0.1, progress_every=0,
        )
        assert bool(df.loc[df["channel"] == target, "triggered"].iloc[0])

    def test_quiet_stream_rarely_triggers(self, tmp_path):
        rng = np.random.default_rng(6)
        f0 = tmp_path / "000000.npy"
        np.save(f0, rng.normal(0, 1.0, size=(N_SAMPLES, N_CHANNELS)).astype(np.int16))
        indexed = [(pd.Timestamp("2025-01-01 00:00:00"), str(f0))]
        df = sta_lta_stream(indexed, fs=FS, step=25, bin_s=0.25, progress_every=0)
        assert df["triggered"].mean() < 0.1

    def test_requires_exactly_one_energy_source(self, tmp_path):
        rng = np.random.default_rng(7)
        f0 = tmp_path / "000000.npy"
        np.save(f0, rng.normal(0, 1.0, size=(1000, 100)).astype(np.int16))
        indexed = [(pd.Timestamp("2025-01-01 00:00:00"), str(f0))]
        with pytest.raises(ValueError):
            sta_lta_stream(indexed, fs=FS, step=25, bandpass_hz=(1.0, 20.0), wavelet_freq=10.0)
        with pytest.raises(ValueError):
            sta_lta_stream(indexed, fs=FS, step=25, bandpass_hz=None, wavelet_freq=None)


class TestClusterDetections:
    def test_merges_nearby_triggers_into_one_event(self):
        df = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(
                    ["2025-01-01 00:00:00", "2025-01-01 00:00:02", "2025-01-01 00:05:00"]
                ),
                "channel": [400, 410, 400],
                "score": [0.9, 0.95, 0.8],
                "triggered": [True, True, True],
            }
        )
        events = cluster_detections(df, time_tol_s=5.0, channel_tol=20)
        assert len(events) == 2
        seed = events[events["n_members"] == 2].iloc[0]
        assert seed["channel"] == 410  # highest-score member is the seed
