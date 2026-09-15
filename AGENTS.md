# AGENTS.md — DAS (Distributed Acoustic Sensing) domain knowledge

Everything below was learned the hard way working through one real task: finding and
correcting a geographic mis-registration in a DAS cable's channel-position geojson,
using the raw strain data correlated against AIS ship traffic. That investigation is
in `peachfuzz11/nn`, branch `das-cable-offset` (PR #3) — `estimate_cable_offset.py`,
`README.md`, `visualize.ipynb`. Read that PR's commit history if you want the full
blow-by-blow.

The reusable primitives from that task are already extracted into this repo's
`src/das_utils/` package (`geo.py`, `io.py`, `energy.py`, `ais.py`, `correlate.py`) with
a synthetic-recovery test suite in `tests/` - read those modules' docstrings first, they
cover the same ground as this file but as working code. This file stays as the prose
version: the *why*, the dead ends, and anything not (yet) captured as code.

## What DAS actually is, in the terms this document uses

A DAS interrogator sends laser pulses down a fiber-optic cable and measures phase/strain
changes at every point along it via backscatter. The output is a 2D array: **time ×
channel**, where each "channel" is a fixed point along the fiber (spaced every `dx`
meters — a hardware/processing parameter, not a physical sensor; the real spatial
resolution is closer to the *gauge length* `gl`, typically `gl ≈ 2×dx`). Anything that
strains the ground/water near the fiber — a passing ship, a bridge, machinery, seismic
activity, tides — shows up as elevated broadband or tonal energy at the channels nearest
to it, roughly proportional to the log of proximity.

Critically: **the DAS system has no innate idea where in the world each channel is.**
Geolocation comes from a separate survey (GPS-tracked fiber-laying, or route
digitization) that maps channel index → lat/lon/depth. That mapping can be — and in the
task this file is drawn from, was — wrong by an amount worth correcting, and DAS data
itself is one of the best tools for finding out how wrong, if you have an independent
positioning signal (AIS, in this case) to correlate against.

## Data formats you'll encounter

**Raw DAS files**: commonly one file per short time window (`HHMMSS.npy` was the
convention here — 10-second files), shape `(n_time_samples, n_channels)`, `int16`,
sampled at `fs` Hz (400 Hz here). Concatenate along axis 0 to get continuous time. At
scale (hours of data, thousands of channels) this is tens to hundreds of GB raw — you
almost always want to reduce to a compressed per-channel time series (energy, RMS, an
STA/LTA statistic) rather than holding raw strain in memory for long stretches. Stream
file-by-file, cache the reduced representation.

**Watch for truncated files.** A file still being written, or cut short by a copy
error, loads with the wrong element count and throws on `np.load`. Check file size
against the modal/expected size for that file set before trusting it; drop mismatches
rather than trying to partial-read them (a `filtfilt` on a torn signal is worse than one
fewer data point).

**Cable position metadata**: typically a JSON with `positions: {channel_id (str) ->
{lat, lon, depth}}` and `metadata: {fs, dx, gl}`. Two gotchas seen in practice:
- The position table and the raw array can have **different channel counts** (3919
  positions vs. 3917 recorded channels, in this case) — some channels at one or both
  ends may be un-recorded or outside a maintained span. **Don't assume which end is
  missing, or that array-column-order matches position-key order, without checking.**
  If you have any independent landmark with known real-world coordinates near the
  route (a bridge, a platform, a shore crossing), use it to verify the mapping
  empirically rather than assuming "channel 1 = array column 0" is safe. See the
  pitfalls section — this exact assumption cost significant time before being
  (mostly) resolved by a different fix.
- `dx` lets you convert channel-index deltas to along-fiber meters. This is *not* the
  same as straight-line lat/lon distance unless the fiber runs dead straight between
  the two points — fine for short spans, not for long ones.

**AIS data**: if it's the Danish Maritime Authority's public `aisdk-YYYY-MM-DD.csv`
dumps (or similar), expect: `# Timestamp` (note the leading `# ` — strip it), DD/MM/YYYY
HH:MM:SS format, `Type of mobile` (filter to `Class A`/`Class B` for real vessels — Base
Station/AtoN/SAR Airborne rows pollute the file), `MMSI`, `Latitude`/`Longitude`, `SOG`
(knots), `COG`, `Name`. These daily dumps are usually multi-GB but **time-sorted** — a
`grep '^DD/MM/YYYY HH:'` line-prefix filter for just the hour(s) you need is dramatically
faster than loading the whole file into pandas, and was the difference between a 5-second
extraction and paging through 2.7GB.

## Core reusable primitives

These are implemented and tested in `src/das_utils/` - read the module docstrings for
full detail. Quick map of what's where:

- **`geo.py`** - `local_xy`/`offset_m` (flat-earth projection - good enough for any DAS
  deployment span under a few tens of km, don't reach for real geodesy), `build_nearest_tree`/
  `nearest_index` (KD-tree nearest-channel lookup), `apply_translation` (shift a whole
  position table by a constant offset), `positions_to_geojson_linestring`.
- **`io.py`** - `load_channel_positions`, `positions_arrays`, `index_das_files` (sorts
  by time, drops truncated files by majority-size vote).
- **`energy.py`** - `bandpass_energy_image`: streams raw files one at a time, bandpass +
  RMS-reduces each to one row, optionally cached to disk. A 1-20Hz bandpass is a
  reasonable default for vessel-scale signatures (engine/hull/propeller broadband and
  low tonals); adjust per deployment and target event type.
- **`ais.py`** - `extract_time_window` (fast grep-based extraction from a large sorted
  AIS dump), `filter_near_positions` (class/distance filtering + nearest-point columns).
- **`correlate.py`** - `closest_approach_offset`, `estimate_translation_offset`,
  `margin_sensitivity`, `to_epoch_seconds` - the recipe below, as reusable functions.

## The AIS-DAS correlation recipe (for cable geo-registration, or similar tasks)

The general problem: you have DAS's own internal channel index (ground truth about
*where along the fiber* something happened) and AIS's own reported lat/lon (ground
truth about *where on Earth* a vessel was). If the cable's stated geo-registration
were correct, "nearest channel to a vessel's AIS position" and "channel with peak DAS
energy when that vessel is nearby" would be the same channel. When they're not, the
size and direction of that discrepancy — aggregated over many independent vessels — is
your correction.

**What worked, in the end (after two false starts — see pitfalls):**

1. Filter AIS to contacts within some generous radius (2km worked here) of the
   *uncorrected* cable, in the DAS recording's time window.
2. **Keep only vessels genuinely transiting at cruising speed** (a few knots and up —
   tune to the strait/channel; large commercial vessels rarely dawdle in a shipping
   lane). Discard anything loitering, moored, or doing dynamic positioning nearby, no
   matter how much data it offers you.
3. For each qualifying vessel, take its **single closest-approach point** to the
   (uncorrected) cable, and the one raw DAS time-bin covering that instant.
4. Around the *naively* nearest channel (via the KD-tree lookup above), search a
   **bounded** window for the channel with peak energy at that instant. Bound size
   should be checked for stability (see below), not assumed.
5. The offset for that vessel = its true AIS position minus the detected channel's
   stated position. Aggregate (median, across vessels) for the final correction —
   should agree in both magnitude and direction across independent vessels if it's a
   real, simple-translation mis-registration.
6. **Always plot offset-vs-search-margin before trusting a number.** A real signal
   produces a plateau: the result is stable across a healthy range of margins and only
   degrades well outside it. If the "best" answer keeps changing as you widen or
   narrow the search, you don't have a real detection yet, you have an artifact.

This generalizes beyond cable-offset-correction: any task that needs "which DAS channel
corresponds to a known-position event" (locating a leak, verifying an as-built route,
correlating a construction/maintenance vessel's activity to a specific span) is the same
recipe — a reliable, independently-positioned, *moving* reference event, a bounded local
search, and a stability check.

## Hard-won pitfalls (read this before you re-derive them the slow way)

**A near-stationary reference is worse than useless — it's actively misleading.** If a
"ground truth" vessel barely moves, its predicted channel position is essentially
constant, so "find the channel that best matches its track" degenerates into "find the
single loudest channel on the *entire* dataset" — regardless of whether that channel has
anything to do with the vessel. Worse, a long, clean, high-confidence-looking match from
such a vessel (lots of samples, low noise) can dominate a naive confidence-weighted
average over many short, noisier, but *genuinely diagnostic* moving-vessel matches, and
you won't notice unless you specifically check whether your top-confidence match came
from something that was actually moving. **Exclude non-transiting references outright;
don't just downweight them.**

**An unbounded (or too-wide) search window has the same failure mode from a different
angle.** Even with a genuinely moving reference, if the search range is wide enough to
reach some other loud, unrelated, fixed feature elsewhere in the data, a sum- or
mean-based alignment score can "walk off" to it and score better there than at the true
(weaker, but real) local match — especially if that vessel's track only sweeps a
short physical distance. A percentile-based score (require *sustained* elevation across
most of the window, not just a few strong samples) helps but is not sufficient on its
own; bounding the search to a physically-plausible range and checking result stability
across that bound (see recipe step 6) is what actually catches this.

**Scoring metric choice matters more than it looks like it should.** `sum`/`mean`
energy along a candidate alignment is gameable by a handful of coincidentally strong
samples. A percentile (e.g. 25th) forces *most* of the window to be elevated, which is
a much better proxy for "this is genuinely the same physical source the whole time."
Still combine with the bounded-search discipline above — belt and suspenders.

**Don't trust a single strong "landmark" coincidence to validate your channel-index
mapping.** In this task, independently-known coordinates for a real structure near the
route (fetched from OpenStreetMap) happened to line up naturally with the single
loudest, most persistent channel in the data — strong-looking evidence that a
particular indexing convention (array-column order reversed relative to the position
table's channel-id order) was correct. It turned out to be a coincidence (or at least,
not the actual fix) once the analysis was redone properly with a genuinely diagnostic
method (bounded search, moving-vessel-only references) — the *original* indexing
convention was right all along; the earlier analysis's error was entirely in the
search/scoring methodology, not the channel mapping. One matching landmark is a hint
worth chasing, not proof — resolve indexing/mapping questions with the same discipline
(multiple independent checks, stability under parameter changes) you'd apply to the
main result.

**Verify time alignment explicitly, don't assume filenames are self-describing.** DAS
filenames often encode only `HHMMSS`, no date — get the date from context (a companion
AIS file's name, deployment metadata) and construct real timestamps early, so every
later join (AIS timestamp ↔ DAS file ↔ energy-image row) is done on real `Timestamp`
values, not string/positional matching that silently breaks on gaps or reordering.

**Check for time gaps in a DAS file sequence before treating rows as evenly spaced.** A
missing chunk of files (an outage, a transfer gap) means file index ≠ uniform time step;
always derive bin timestamps from the filenames/metadata, never from `arange(n) * dt`.

**`pandas.DatetimeIndex.astype("int64")` is not reliably nanoseconds.** Since pandas 2.x,
datetime64 arrays can be stored at second/millisecond/microsecond/nanosecond resolution,
and *which* one you get from a given construction path isn't obvious - building a
`DatetimeIndex` from `pd.Timestamp(...) + pd.to_timedelta(...)` produced microsecond
resolution in the pandas version this repo was built against. Naively doing
`idx.astype("int64") / 1e9` to get epoch seconds silently assumes nanoseconds; get it
wrong and every timestamp is off by a constant factor (1e3, 1e6...) - not a crash, a
plausible-looking wrong number that then breaks every time-bin lookup downstream. This
is exactly the kind of bug a synthetic recovery test catches immediately (the recovered
offset comes back wildly wrong or completely absent) and real-data testing might not
(if your real data happens to load through a code path that produces `datetime64[ns]`
in your particular pandas version). Use `correlate.to_epoch_seconds`, which forces
`datetime64[ns]` before converting, rather than reimplementing the conversion inline.
`pd.Timestamp.value` (the scalar accessor) is *not* affected - it's documented and
verified to always return nanoseconds regardless of the Timestamp's own stored unit -
so a mismatch between a scalar timestamp (ns) and an array-derived epoch (some other
unit) is an easy way to introduce exactly this bug without either side looking wrong in
isolation.

## `das-utils` repo layout (as built)

```
src/das_utils/
  io.py          # load_channel_positions, index_das_files, positions_arrays
  geo.py         # local_xy, offset_m, build_nearest_tree/nearest_index, apply_translation,
                 # positions_to_geojson_linestring
  energy.py      # bandpass_energy_image - streaming + optional disk cache
  ais.py         # extract_time_window (fast grep extraction), filter_near_positions
  correlate.py   # to_epoch_seconds, closest_approach_offset, estimate_translation_offset,
                 # margin_sensitivity
tests/
  # synthetic fixtures with a known injected offset, checking it's recovered -
  # nothing in this domain is easy to unit-test against real data, so lean on this
  # pattern for anything new: inject known ground truth, verify recovery
```

Not yet built, worth doing before the second real use case: a `cli/` of thin wrappers
per task (offset correction, event localization, ...), and making `correlate.py`'s
notion of "what counts as a valid reference event" more generic (currently the caller
pre-filters candidates themselves, e.g. by an AIS SOG threshold, which works but a
proper pluggable predicate would make the intent more explicit and reusable across
reference-event types beyond AIS - an ROV, a known controlled source, etc).

## Open questions / not fully resolved

- The exact physical identity of the single loudest, most persistent channel in the
  reference dataset was never conclusively determined (candidates considered: a nearby
  bridge's structural noise, an active cable-work vessel in near-contact with the
  fiber). It didn't end up mattering for the final method, but if a future task needs
  to *explain* every strong feature rather than just avoid being misled by them, that's
  unfinished business worth a cleaner approach (e.g. spectral/tonal fingerprinting
  rather than broadband energy alone).
- Only a translation (constant offset) correction was modeled. A real mis-registration
  could in principle include rotation or scale error too; this wasn't needed here (the
  translation model fit all reference vessels tightly), but a more general repo should
  probably support fitting a full similarity/affine transform when a translation-only
  fit doesn't converge as cleanly as it did in this task.
