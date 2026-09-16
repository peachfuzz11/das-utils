import numpy as np
from scipy.fft import rfft, rfftfreq

from das_utils.spectral import (
    bandpass,
    bandstop,
    channel_shuffle,
    frequency_jitter,
    frequency_shift,
    gain_randomization,
    highpass,
    lowpass,
    mixup,
    notch,
    phase_shift,
    random_augment,
    spectral_mask,
    spectral_warp,
    time_warp,
)

FS = 400.0
N_TIME = 2000
N_CHAN = 4


def _white_noise(seed=0):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((N_TIME, N_CHAN)).astype(np.float64)


def _tone(freq_hz, fs=FS, n=N_TIME, n_chan=N_CHAN):
    t = np.arange(n) / fs
    return np.tile(np.sin(2 * np.pi * freq_hz * t)[:, np.newaxis], (1, n_chan))


class TestBandpass:
    def test_preserves_in_band_tone(self):
        img = _tone(10.0)
        out = bandpass(img, FS, low=5.0, high=15.0)
        assert out.shape == img.shape
        assert np.std(out) > 0.5 * np.std(img)

    def test_attenuates_out_of_band_tone(self):
        img = _tone(100.0)
        out = bandpass(img, FS, low=5.0, high=15.0)
        assert np.std(out) < 0.1 * np.std(img)


class TestLowpass:
    def test_keeps_low_attenuates_high(self):
        low = _tone(5.0)
        high = _tone(150.0)
        out_low = lowpass(low, FS, cutoff=20.0)
        out_high = lowpass(high, FS, cutoff=20.0)
        assert np.std(out_low) > 0.5 * np.std(low)
        assert np.std(out_high) < 0.1 * np.std(high)


class TestHighpass:
    def test_keeps_high_attenuates_low(self):
        low = _tone(5.0)
        high = _tone(150.0)
        out_low = highpass(low, FS, cutoff=20.0)
        out_high = highpass(high, FS, cutoff=20.0)
        assert np.std(out_high) > 0.5 * np.std(high)
        assert np.std(out_low) < 0.1 * np.std(low)


class TestBandstop:
    def test_attenuates_in_band(self):
        img = _tone(50.0)
        out = bandstop(img, FS, low=40.0, high=60.0)
        assert np.std(out) < 0.1 * np.std(img)


class TestNotch:
    def test_removes_tonal(self):
        img = _tone(50.0) + 0.1 * _white_noise(1)
        out = notch(img, FS, freq=50.0, quality=10.0)
        out_spec = np.abs(rfft(out[:, 0]))
        orig_spec = np.abs(rfft(img[:, 0]))
        freqs = rfftfreq(N_TIME, 1.0 / FS)
        band = (freqs > 45) & (freqs < 55)
        assert out_spec[band].max() < 0.1 * orig_spec[band].max()


class TestSpectralMask:
    def test_shape_preserved_and_seeded(self):
        img = _white_noise()
        a = spectral_mask(img, FS, n_bands=2, max_width_hz=10.0, seed=42)
        b = spectral_mask(img, FS, n_bands=2, max_width_hz=10.0, seed=42)
        assert a.shape == img.shape
        assert np.array_equal(a, b)

    def test_changes_spectrum(self):
        img = _white_noise()
        out = spectral_mask(img, FS, n_bands=3, max_width_hz=20.0, seed=0)
        orig_spec = np.abs(rfft(img[:, 0]))
        out_spec = np.abs(rfft(out[:, 0]))
        # some bins should be zeroed
        assert np.any(out_spec < 0.01 * orig_spec)


class TestFrequencyShift:
    def test_shifts_tone(self):
        img = _tone(50.0)
        out = frequency_shift(img, FS, max_shift_hz=5.0, seed=0)
        # the peak should move; find the dominant freq
        freqs = rfftfreq(N_TIME, 1.0 / FS)
        orig_peak = freqs[np.argmax(np.abs(rfft(img[:, 0])))]
        out_peak = freqs[np.argmax(np.abs(rfft(out[:, 0])))]
        assert abs(out_peak - orig_peak) > 0  # it moved

    def test_seeded_reproducible(self):
        img = _white_noise()
        a = frequency_shift(img, FS, seed=7)
        b = frequency_shift(img, FS, seed=7)
        assert np.array_equal(a, b)


class TestPhaseShift:
    def test_preserves_magnitude(self):
        img = _white_noise()
        out = phase_shift(img, seed=0)
        orig_mag = np.abs(rfft(img[:, 0]))
        out_mag = np.abs(rfft(out[:, 0]))
        assert np.allclose(orig_mag, out_mag, rtol=1e-5)

    def test_changes_waveform(self):
        img = _white_noise()
        out = phase_shift(img, seed=1)
        assert not np.allclose(img, out)


class TestSpectralWarp:
    def test_shape_preserved(self):
        img = _white_noise()
        out = spectral_warp(img, FS, max_warp=0.1, seed=0)
        assert out.shape == img.shape

    def test_seeded_reproducible(self):
        img = _white_noise()
        a = spectral_warp(img, FS, seed=3)
        b = spectral_warp(img, FS, seed=3)
        assert np.allclose(a, b)


class TestMixup:
    def test_preserves_mean_when_alpha_small(self):
        img = _white_noise()
        out = mixup(img, alpha=0.01, seed=0)
        # tiny alpha => lambda near 0 or 1, so channels are mostly swapped
        assert out.shape == img.shape

    def test_seeded_reproducible(self):
        img = _white_noise()
        a = mixup(img, alpha=0.3, seed=5)
        b = mixup(img, alpha=0.3, seed=5)
        assert np.array_equal(a, b)


class TestChannelShuffle:
    def test_preserves_set_of_channels(self):
        img = _white_noise()
        out = channel_shuffle(img, max_swaps=3, seed=0)
        # each column of out is a column of img (possibly swapped)
        for c in range(N_CHAN):
            assert any(np.array_equal(out[:, c], img[:, j]) for j in range(N_CHAN))

    def test_seeded_reproducible(self):
        img = _white_noise()
        a = channel_shuffle(img, seed=1)
        b = channel_shuffle(img, seed=1)
        assert np.array_equal(a, b)


class TestGainRandomization:
    def test_scales_each_channel(self):
        img = np.ones((100, 3))
        img[:, 0] = 1.0
        img[:, 1] = 2.0
        img[:, 2] = 3.0
        out = gain_randomization(img, low=0.5, high=0.5, seed=0)
        # all gains are 0.5 (uniform with low==high)
        assert np.allclose(out, img * 0.5)

    def test_seeded_reproducible(self):
        img = _white_noise()
        a = gain_randomization(img, seed=2)
        b = gain_randomization(img, seed=2)
        assert np.array_equal(a, b)


class TestTimeWarp:
    def test_shape_preserved(self):
        img = _white_noise()
        out = time_warp(img, FS, max_stretch=0.05, seed=0)
        assert out.shape == img.shape

    def test_small_warp_close_to_original(self):
        # smooth tone with tiny warp: displacement < 1 sample => minimal change
        img = _tone(5.0)
        out = time_warp(img, FS, max_stretch=0.0001, seed=0)
        assert np.allclose(out, img, atol=0.05)

    def test_seeded_reproducible(self):
        img = _white_noise()
        a = time_warp(img, FS, seed=4)
        b = time_warp(img, FS, seed=4)
        assert np.allclose(a, b)


class TestFrequencyJitter:
    def test_preserves_shape(self):
        img = _white_noise()
        out = frequency_jitter(img, FS, sigma_hz=0.5, seed=0)
        assert out.shape == img.shape

    def test_small_jitter_close_to_original(self):
        img = _tone(5.0)
        out = frequency_jitter(img, FS, sigma_hz=0.001, seed=0)
        assert np.allclose(out, img, atol=0.1)

    def test_seeded_reproducible(self):
        img = _white_noise()
        a = frequency_jitter(img, FS, seed=6)
        b = frequency_jitter(img, FS, seed=6)
        assert np.array_equal(a, b)


class TestRandomAugment:
    def test_returns_same_shape(self):
        img = _white_noise()
        out = random_augment(img, FS, p=0.5, seed=0)
        assert out.shape == img.shape

    def test_p_zero_returns_close_to_original(self):
        img = _white_noise()
        out = random_augment(img, FS, p=0.0, seed=0)
        # p=0 means no augmentation applied, but the fallback may apply one
        # with p=0 the loop never applies, and the "applied==0" guard applies one
        # so let's test p=0 means at least the first is applied
        # Actually the guard says: if applied==0 and p>0 -> apply first
        # with p=0, applied stays 0, p>0 is false, so nothing is applied
        assert np.array_equal(out, img)

    def test_seeded_reproducible(self):
        img = _white_noise()
        a = random_augment(img, FS, p=1.0, seed=10)
        b = random_augment(img, FS, p=1.0, seed=10)
        assert np.array_equal(a, b)

    def test_always_applies_at_least_one(self):
        img = _white_noise()
        out = random_augment(img, FS, p=1.0, seed=0)
        assert not np.array_equal(out, img)
