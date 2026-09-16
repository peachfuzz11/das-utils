"""Tests for PSD-based spectral feature extraction."""

import numpy as np
import pytest

from das_utils.psd import band_powers, channel_psd, dominant_frequency

FS = 400.0
N = 8000


def tone(freq, amplitude=5.0, seed=0, noise=0.1):
    t = np.arange(N) / FS
    rng = np.random.default_rng(seed)
    return amplitude * np.sin(2 * np.pi * freq * t) + rng.normal(0, noise, N)


class TestChannelPsd:
    def test_returns_freqs_and_psd_same_length(self):
        freqs, psd = channel_psd(tone(30.0), FS, nperseg=1024)
        assert len(freqs) == len(psd)
        assert freqs[0] == 0.0
        assert freqs[-1] == pytest.approx(FS / 2, rel=0.01)


class TestDominantFrequency:
    def test_recovers_tone_frequency(self):
        freqs, psd = channel_psd(tone(30.0), FS, nperseg=1024)
        assert dominant_frequency(freqs, psd) == pytest.approx(30.0, abs=1.0)

    def test_respects_fmin_fmax(self):
        # two tones; restrict search to exclude the stronger low one
        sig = tone(5.0, amplitude=20.0, seed=1) + tone(60.0, amplitude=2.0, seed=2)
        freqs, psd = channel_psd(sig, FS, nperseg=1024)
        assert dominant_frequency(freqs, psd, fmin=20.0) == pytest.approx(60.0, abs=2.0)

    def test_raises_for_empty_range(self):
        freqs, psd = channel_psd(tone(30.0), FS, nperseg=1024)
        with pytest.raises(ValueError):
            dominant_frequency(freqs, psd, fmin=1000.0, fmax=2000.0)


class TestBandPowers:
    def test_shape_matches_bands(self):
        freqs, psd = channel_psd(tone(30.0), FS, nperseg=1024)
        powers = band_powers(freqs, psd, np.array([0, 10, 20, 30, 40, 50]))
        assert powers.shape == (5,)

    def test_power_concentrated_in_tone_band(self):
        freqs, psd = channel_psd(tone(30.0, amplitude=20.0), FS, nperseg=1024)
        powers = band_powers(freqs, psd, np.array([0, 10, 20, 30, 40, 50]))
        assert np.argmax(powers) == 3  # the [30,40) band
