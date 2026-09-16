"""Power spectral density features for a single DAS channel's waveform.

Where ``wavelet.py`` targets *one* candidate frequency at a time (useful
inside a detector's inner loop), this module characterizes a signal's
*whole* spectral shape at once - the feature representation used to compare
confirmed ship crossings against background noise (Welch PSD, dominant
frequency, per-band power), following the approach in Pedersen et al.,
"A Feasibility Study of Automated Detection and Classification of Signals
in Distributed Acoustic Sensing" (Sensors 2025, 25(17), 5445): compute each
signal's PSD, read off its dominant frequency component, and use that
frequency-domain representation as the feature vector clustering/PCA is run
on.
"""

import numpy as np
from scipy.signal import welch


def channel_psd(
    signal: np.ndarray, fs: float, nperseg: int = 1024
) -> tuple[np.ndarray, np.ndarray]:
    """Welch power spectral density of a single-channel waveform.

    ``signal`` is 1-D, ``(n_samples,)``. Returns ``(freqs, psd)`` - Welch's
    method (averaged, overlapping periodograms) rather than a single raw
    FFT, since it trades a little frequency resolution for a much less
    noisy power estimate, which matters when comparing shapes across many
    short (few-second) signals.
    """
    nperseg = min(nperseg, len(signal))
    freqs, psd = welch(np.asarray(signal, dtype=np.float64), fs=fs, nperseg=nperseg)
    return freqs, psd


def dominant_frequency(
    freqs: np.ndarray, psd: np.ndarray, fmin: float = 0.0, fmax: float | None = None
) -> float:
    """Frequency of the PSD's peak within ``[fmin, fmax]``.

    ``fmax=None`` means no upper bound. Restricting the search range matters
    when very-low-frequency drift or a DC-adjacent bin would otherwise
    dominate the raw PSD regardless of any genuine higher-frequency
    signature.
    """
    fmax = float(freqs[-1]) if fmax is None else fmax
    mask = (freqs >= fmin) & (freqs <= fmax)
    if not mask.any():
        raise ValueError(f"no PSD bins in [{fmin}, {fmax}] Hz")
    sub_freqs, sub_psd = freqs[mask], psd[mask]
    return float(sub_freqs[np.argmax(sub_psd)])


def band_powers(freqs: np.ndarray, psd: np.ndarray, band_edges: np.ndarray) -> np.ndarray:
    """Total PSD power in each ``[band_edges[i], band_edges[i+1])`` bin.

    Returns an array of length ``len(band_edges) - 1`` - a coarse,
    fixed-width feature vector summarizing the spectral shape, suitable as
    input to PCA/clustering without depending on the exact FFT bin grid
    (which can differ slightly file to file).
    """
    powers = np.zeros(len(band_edges) - 1)
    for i in range(len(band_edges) - 1):
        mask = (freqs >= band_edges[i]) & (freqs < band_edges[i + 1])
        powers[i] = psd[mask].sum() if mask.any() else 0.0
    return powers
