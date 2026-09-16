"""Per-file, stepped ship-detection scoring over a DAS channel image.

The detector is applied to one file (or one time-chunk) at a time and steps
across channels every ``step`` columns, producing one response per step
position - not one response per raw channel.

Each stepped channel's bandpass RMS energy is compared against its own
**temporal** baseline (``channel_baseline``: that channel's robust median and
MAD over a broad reference period, e.g. the whole scan) rather than its
*spatial* neighbours. A first version of this module used a spatial
percentile-rank score instead, but validation against real AIS-derived
ground truth showed it fails specifically at close range: a real vessel's
signature routinely spans a wider stretch of fiber than the context window
used for the "background" reference, so the reference itself gets
contaminated by the same signal and the contrast collapses. A channel's own
long-run temporal baseline doesn't have that problem - it isn't measured at
the same instant as a candidate event.

Ship crossings are a *rare* event relative to the number of (file, channel,
time-window) samples a stepped scan produces, so a per-sample false-positive
rate that looks small in isolation still produces overwhelming absolute false
positives at scale and wrecks precision. Three independent, AND-ed filters
guard against this rather than relying on a single threshold:

1. **Sustained in time** (``detect_file``) - a real crossing elevates a
   channel for nearly its *entire* dwell time in the file, not a brief
   instant; requiring elevation for ``sustain_duration`` (nearly all)
   sub-windows rejects transient spikes.
2. **Not a common-mode artifact** (``common_mode_suppression``) - a
   fiber-wide event (gain step, seismic transient, interrogator hiccup)
   elevates *every* channel at once, which a purely per-channel threshold
   cannot tell apart from many independent ship detections. If more than a
   small fraction of the stepped channels trigger in the same sub-window,
   none of them are trusted - a real, localized vessel could never do this.
3. **Corroborated in space** (``spatial_corroboration``) - conversely, a real
   vessel lights up more than one nearby channel at once (its signature
   spans some real physical distance), so a genuine detection should never
   be spatially isolated *either*. A trigger with no other triggered channel
   nearby, but not so many as to look common-mode, is what's meant to
   survive.
"""

import numpy as np
import pandas as pd

EDGE_TRIM_S = 1.0  # sosfiltfilt settling time near a 1Hz low cutoff needs about this long


def bandpass_filtered_trimmed(
    data: np.ndarray, fs: float, bandpass_hz: tuple[float, float], edge_s: float = EDGE_TRIM_S
) -> tuple[np.ndarray, int]:
    """Zero-phase bandpass filter ``data``, discarding the filter's settling transient.

    ``sosfiltfilt`` output is measurably contaminated for roughly
    ``edge_s`` seconds at the start *and* end of a short signal - the filter
    hasn't "settled" yet there. For this package's default ``(1, 20)`` Hz
    band on a 10s file, that's 2-3x the steady-state power, at *every* file's
    edges, regardless of whether a real signal is present - left in, it's
    indistinguishable from (and roughly the same magnitude as) a genuine
    detection, and was silently inflating every bandpass-based feature in
    this module before this was diagnosed. Trim it rather than treat it as
    data.

    Returns ``(trimmed, trim_samples)`` - ``trim_samples`` is how many
    samples were cut from the start (needed to convert an index into the
    trimmed array back to a time offset from the original ``data``).
    """
    from scipy.signal import butter, sosfiltfilt

    sos = butter(4, bandpass_hz, btype="bandpass", fs=fs, output="sos")
    filtered = sosfiltfilt(sos, data, axis=0)
    trim = int(round(edge_s * fs))
    if trim > 0 and filtered.shape[0] > 2 * trim:
        return filtered[trim:-trim], trim
    return filtered, 0


def step_channels(n_channels: int, step: int) -> np.ndarray:
    """Channel indices to score: every ``step``-th channel, always including the last.

    ``step=1`` scores every channel; larger ``step`` trades spatial
    resolution for speed.
    """
    if step < 1:
        raise ValueError("step must be >= 1")
    idx = np.arange(0, n_channels, step, dtype=np.int64)
    if idx[-1] != n_channels - 1:
        idx = np.append(idx, n_channels - 1)
    return idx


def channel_baseline(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-channel robust temporal baseline: ``(floor, scale)`` from a reference energy image.

    ``image`` is ``(n_time, n_channels)`` (e.g. bandpass RMS per file, from
    ``image.build_image``), ideally spanning a broad enough period that any
    one transient event doesn't dominate it. ``floor`` is the per-channel
    median; ``scale`` is the median absolute deviation (scaled to be a
    robust std estimate), floored to a small positive number so a
    channel that is perfectly constant in the reference period never
    produces a divide-by-zero later.
    """
    img = np.asarray(image, dtype=np.float64)
    floor = np.median(img, axis=0)
    scale = np.median(np.abs(img - floor), axis=0) * 1.4826
    scale = np.maximum(scale, 1e-6 * np.maximum(floor, 1.0))
    return floor, scale


def detect_file(
    data: np.ndarray,
    fs: float,
    step: int,
    channel_floor: np.ndarray,
    channel_scale: np.ndarray,
    z_threshold: float = 6.0,
    sustain_duration: float = 0.9,
    bandpass_hz: tuple[float, float] = (1.0, 20.0),
    sub_windows: int = 10,
) -> pd.DataFrame:
    """Detection scores for one file: one row per stepped channel per time-window.

    ``data`` is ``(n_samples, n_channels)`` raw strain for one file (or
    chunk). The file's time axis is split into ``sub_windows`` equal
    sub-windows (default 10, giving 10 responses per file for a 10s file -
    i.e. one per second); within each sub-window, per-channel bandpass RMS
    energy is computed for every stepped channel (``step_channels``) and
    turned into a z-score against that channel's own ``channel_floor``/
    ``channel_scale`` (from ``channel_baseline``, computed once up front over
    a broad reference period - not from this file alone).

    A channel is flagged (``triggered=True``) when its z-score exceeds
    ``z_threshold`` in at least ``sustain_duration`` (default 0.9, i.e.
    nearly all) of the sub-windows - sustained for essentially the whole
    file, not a brief spike. Ship crossings are rare relative to the volume
    of (file, channel, window) samples scanned, so both need to be strict: a
    low ``z_threshold`` or a loose ``sustain_duration`` triggers on noise
    alone at a rate that overwhelms the true positives - only ``triggered``
    rows this strict are meant to survive on their own;
    ``common_mode_suppression`` and ``spatial_corroboration`` filter further
    before treating a trigger as a real detection.

    Returns a DataFrame with one row per ``(sub_window, stepped channel)``:
    columns ``time_s`` (sub-window center, seconds from file start),
    ``channel``, ``score`` (the z-score), ``triggered``.
    """
    n_channels = data.shape[1]
    filtered, trim = bandpass_filtered_trimmed(data.astype(np.float32), fs, bandpass_hz)
    n_samples = filtered.shape[0]
    if n_samples % sub_windows:
        n_samples = (n_samples // sub_windows) * sub_windows
        filtered = filtered[:n_samples]

    chunk = n_samples // sub_windows
    cols = step_channels(n_channels, step)
    floor, scale = channel_floor[cols], channel_scale[cols]

    rows = []
    for w in range(sub_windows):
        block = filtered[w * chunk : (w + 1) * chunk, cols]
        energy = np.sqrt(np.mean(block**2, axis=0))
        time_s = (w + 0.5) * chunk / fs + trim / fs
        z = (energy - floor) / scale
        for c, zi in zip(cols, z, strict=True):
            rows.append({"time_s": time_s, "channel": int(c), "score": float(zi)})

    df = pd.DataFrame(rows)
    if df.empty:
        return df.assign(triggered=pd.Series(dtype=bool))

    trigger_frac = df.groupby("channel")["score"].transform(lambda s: (s > z_threshold).mean())
    df["triggered"] = trigger_frac >= sustain_duration
    return df


def detect_stream(
    indexed_files: list[tuple[pd.Timestamp, str]],
    fs: float,
    step: int,
    channel_floor: np.ndarray,
    channel_scale: np.ndarray,
    z_threshold: float = 6.0,
    sustain_duration: float = 0.9,
    bandpass_hz: tuple[float, float] = (1.0, 20.0),
    sub_windows: int = 10,
    progress_every: int = 50,
) -> pd.DataFrame:
    """Apply ``detect_file`` to every file in ``indexed_files``, merging into one table.

    Each file contributes its own ``sub_windows`` x ``len(step_channels(...))``
    rows (N responses per file), with ``time_s`` converted to an absolute
    ``timestamp`` using each file's own start time. This is the main
    entry point for scoring a whole recording session - pipe the result
    through ``common_mode_suppression`` then ``spatial_corroboration`` before
    ``cluster_detections`` to reject fiber-wide artifacts and spatially
    isolated (almost certainly noise) triggers respectively.
    """
    frames = []
    for i, (ts, f) in enumerate(indexed_files):
        data = np.load(f)
        df = detect_file(
            data, fs, step, channel_floor, channel_scale, z_threshold, sustain_duration,
            bandpass_hz, sub_windows,
        )
        if not df.empty:
            df = df.copy()
            df["timestamp"] = ts + pd.to_timedelta(df["time_s"], unit="s")
            df["file"] = f
            frames.append(df)
        if progress_every and i % progress_every == 0:
            print(f"detect_stream: {i + 1}/{len(indexed_files)}")

    if not frames:
        return pd.DataFrame(columns=["time_s", "channel", "score", "triggered", "timestamp", "file"])
    return pd.concat(frames, ignore_index=True)


def common_mode_suppression(df: pd.DataFrame, max_frac_triggered: float = 0.05) -> pd.DataFrame:
    """Distrust every trigger in a (file, time-window) where too many channels fired at once.

    ``df`` must have ``file``, ``time_s``, ``triggered`` columns (e.g.
    ``detect_stream`` output). A real, localized vessel can never
    simultaneously elevate more than a small fraction of the stepped
    channels across the *whole* fiber; if more than ``max_frac_triggered``
    of them trigger in the same (file, time-window), that is a fiber-wide
    artifact (a gain step, a seismic transient, an interrogator hiccup), not
    several independent ships - every trigger in that window is reset to
    ``triggered=False``.
    """
    df = df.copy()
    frac = df.groupby(["file", "time_s"])["triggered"].transform("mean")
    df["triggered"] = df["triggered"] & (frac <= max_frac_triggered)
    return df


def spatial_corroboration(
    df: pd.DataFrame,
    channel_gap: int,
    time_tol_s: float = 5.0,
) -> pd.DataFrame:
    """Drop triggers with no other triggered channel nearby - keep only corroborated ones.

    ``df`` must have ``file``, ``timestamp``, ``channel``, ``triggered``
    columns (e.g. ``detect_stream`` output, ideally after
    ``common_mode_suppression``). A real vessel lights up more than one
    nearby channel at once, so a genuine detection should have at least one
    other triggered stepped channel within ``channel_gap`` indices and
    ``time_tol_s`` seconds, in the *same* file. Keep ``channel_gap`` tight
    (e.g. one ``step`` - just the adjacent stepped position) so this checks
    for a real, spatially-compact signal rather than accepting anything
    fiber-wide (that's what ``common_mode_suppression`` is for). A trigger
    with no such neighbour is reset to ``triggered=False``.
    """
    df = df.copy()
    still_triggered = pd.Series(False, index=df.index)
    for _, group in df[df["triggered"]].groupby("file"):
        idx = group.index.to_numpy()
        channels = group["channel"].to_numpy()
        times = group["timestamp"].astype("datetime64[ns]").astype("int64").to_numpy() / 1e9
        n = len(idx)
        for i in range(n):
            near = (
                (np.abs(channels - channels[i]) <= channel_gap)
                & (np.abs(times - times[i]) <= time_tol_s)
                & (np.arange(n) != i)
            )
            if near.any():
                still_triggered.loc[idx[i]] = True
    df["triggered"] = still_triggered
    return df


def cluster_detections(
    df: pd.DataFrame,
    time_tol_s: float = 5.0,
    channel_tol: int = 50,
) -> pd.DataFrame:
    """Collapse triggered detections into events, deduping nearby (time, channel) hits.

    ``df`` must have ``timestamp``, ``channel``, ``score``, ``triggered``
    columns (e.g. from ``detect_stream``). Only ``triggered`` rows are
    clustered. Greedily groups rows within ``time_tol_s`` seconds *and*
    ``channel_tol`` channels of a cluster's seed (its highest-score member),
    then repeats on the remainder - so one passing vessel that lights up many
    nearby step-channels/time-windows becomes one event.

    Returns one row per event: ``timestamp`` (seed time), ``channel`` (seed
    channel), ``score`` (seed's, the cluster's peak), ``n_members``.
    """
    hits = df[df["triggered"]].sort_values("score", ascending=False).reset_index(drop=True)
    if hits.empty:
        return pd.DataFrame(columns=["timestamp", "channel", "score", "n_members"])

    times = hits["timestamp"].astype("datetime64[ns]").astype("int64") / 1e9
    channels = hits["channel"].to_numpy()
    scores = hits["score"].to_numpy()

    assigned = np.zeros(len(hits), dtype=bool)
    events = []
    for i in range(len(hits)):
        if assigned[i]:
            continue
        dt = np.abs(times - times[i])
        dc = np.abs(channels - channels[i])
        members = (~assigned) & (dt <= time_tol_s) & (dc <= channel_tol)
        assigned |= members
        events.append(
            {
                "timestamp": hits["timestamp"].iloc[i],
                "channel": int(channels[i]),
                "score": float(scores[i]),
                "n_members": int(members.sum()),
            }
        )
    return pd.DataFrame(events)


def sta_lta_ratio(
    energy_bins: np.ndarray,
    short_bins: int,
    long_bins: int,
) -> np.ndarray:
    """Causal short-term/long-term average ratio, per column, at every bin.

    ``energy_bins`` is ``(n_bins, n_channels)``. At bin *i*, STA is the mean
    of the ``short_bins`` most recent bins (ending at *i*); LTA is the mean
    of the ``long_bins`` immediately *before* that (so STA and LTA windows
    never overlap and LTA is never contaminated by the event STA is trying
    to detect). This is the classic seismology STA/LTA detector: unlike a
    baseline fixed once over a broad reference period, both windows slide
    forward with time, so the "background" a bin is judged against is always
    the recent past, not a single global statistic - appropriate for a
    genuinely non-stationary temporal signal.

    Returns an array the same shape as ``energy_bins``, ``nan`` wherever
    there isn't yet ``short_bins + long_bins`` of history (the first bins of
    a stream).
    """
    n_bins, n_channels = energy_bins.shape
    csum = np.concatenate([np.zeros((1, n_channels)), np.cumsum(energy_bins, axis=0)], axis=0)
    idx = np.arange(n_bins)

    sta_lo = np.maximum(idx - short_bins + 1, 0)
    sta = (csum[idx + 1] - csum[sta_lo]) / (idx + 1 - sta_lo)[:, None]

    lta_hi = sta_lo
    lta_lo = np.maximum(idx - short_bins - long_bins + 1, 0)
    has_history = lta_hi > lta_lo
    denom = np.maximum(lta_hi - lta_lo, 1)
    lta = (csum[lta_hi] - csum[lta_lo]) / denom[:, None]

    ratio = sta / np.maximum(lta, 1e-12)
    return np.where(has_history[:, None], ratio, np.nan)


def sta_lta_stream(
    indexed_files: list[tuple[pd.Timestamp, str]],
    fs: float,
    step: int,
    bandpass_hz: tuple[float, float] | None = (1.0, 20.0),
    wavelet_freq: float | None = None,
    bin_s: float = 0.25,
    short_s: float = 1.0,
    long_s: float = 20.0,
    ratio_threshold: float = 3.0,
    sustain_duration: float = 0.5,
    progress_every: int = 50,
) -> pd.DataFrame:
    """STA/LTA ship detection over a stream of files, one row per (file, stepped channel).

    A causal alternative to the ``detect_file``/``channel_baseline`` z-score
    approach: at ``bin_s`` resolution, each stepped channel's energy feeds
    ``sta_lta_ratio`` (see its docstring) with a rolling history buffer
    carried *across* file boundaries, so the long-term average has real
    history even near the start of a file, not just within it.

    The energy source is pluggable: bandpass-filtered power (default, pass
    ``bandpass_hz``) or Morlet wavelet power at a single frequency (pass
    ``wavelet_freq`` instead - see ``wavelet.morlet_cwt_power_series``). A
    channel is flagged (``triggered=True``) for a file when its ratio
    exceeds ``ratio_threshold`` in at least ``sustain_duration`` of that
    file's bins - looser than ``detect_file``'s default because STA/LTA's
    ratio is already a much sharper, self-normalizing signal than a raw
    z-score against a fixed baseline.

    Returns one row per (file, stepped channel): ``file``, ``timestamp``
    (file start), ``time_s`` (always 0.0 - a placeholder so this is a single
    aggregate row per file, compatible with ``common_mode_suppression``'s
    ``(file, time_s)`` grouping), ``channel``, ``score`` (max ratio seen in
    the file), ``triggered``.
    """
    if (bandpass_hz is None) == (wavelet_freq is None):
        raise ValueError("pass exactly one of bandpass_hz or wavelet_freq")

    from scipy.signal import butter, sosfiltfilt

    sos = butter(4, bandpass_hz, btype="bandpass", fs=fs, output="sos") if bandpass_hz else None
    bin_len = max(1, round(bin_s * fs))
    short_bins = max(1, round(short_s / bin_s))
    long_bins = max(1, round(long_s / bin_s))
    history: np.ndarray | None = None  # (n_hist_bins, n_cols), carried across files

    rows = []
    for i, (ts, f) in enumerate(indexed_files):
        data = np.load(f)
        n_samples, n_channels = data.shape
        cols = step_channels(n_channels, step)

        if wavelet_freq is not None:
            from das_utils.wavelet import morlet_cwt_power_series

            energy = morlet_cwt_power_series(data, fs, wavelet_freq, cols)
        else:
            filtered = sosfiltfilt(sos, data[:, cols].astype(np.float64), axis=0)
            energy = filtered**2

        n_bins = n_samples // bin_len
        energy_bins = energy[: n_bins * bin_len].reshape(n_bins, bin_len, len(cols)).mean(axis=1)
        # The bandpass filter's edge transient (see bandpass_filtered_trimmed) inflates the
        # start/end of each file's *own* energy the same way regardless of a real signal - mask
        # those bins out of this file's trigger decision (not out of the rolling history/LTA
        # buffer, which only needs them as approximately-representative background, not exact).
        edge_bins = round(EDGE_TRIM_S / bin_s) if bandpass_hz else 0

        combined = energy_bins if history is None else np.concatenate([history, energy_bins], axis=0)
        offset = 0 if history is None else history.shape[0]
        ratio = sta_lta_ratio(combined, short_bins, long_bins)[offset:]
        valid = np.ones(len(ratio), dtype=bool)
        if edge_bins > 0 and len(valid) > 2 * edge_bins:
            valid[:edge_bins] = False
            valid[-edge_bins:] = False
        ratio_valid = ratio[valid] if valid.any() else ratio

        triggered_frac = np.mean(ratio_valid > ratio_threshold, axis=0)
        max_ratio = np.nanmax(ratio_valid, axis=0) if len(ratio_valid) else np.zeros(len(cols))
        for c, score, frac in zip(cols, max_ratio, triggered_frac, strict=True):
            rows.append(
                {
                    "file": f,
                    "timestamp": ts,
                    "time_s": 0.0,  # one aggregate row per file - see common_mode_suppression
                    "channel": int(c),
                    "score": float(score),
                    "triggered": bool(frac >= sustain_duration),
                }
            )

        history = combined[-(short_bins + long_bins):]
        if progress_every and i % progress_every == 0:
            print(f"sta_lta_stream: {i + 1}/{len(indexed_files)}")

    return pd.DataFrame(rows)
