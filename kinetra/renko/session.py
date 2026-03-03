"""
Renko Session Module
====================

Broker fingerprinting and session-break detection for M1 OHLCV data.

Every M1 file must be fingerprinted before DSP / brick construction to
ensure that daily session gaps (e.g. XAUUSD 21:00–21:59 UTC on MetaAPI)
do not generate spurious brick bursts when the reference price resets.

Design decisions (from RENKO_KINETRA_DESIGN_SPEC.md §29.2–29.3):
  - Session break rule: gaps >= 30 min in M1 data reset the Renko reference
    price; no bricks are emitted across the gap.
  - The dominant missing-minute-of-day range becomes the canonical
    ``session_break_utc_hour`` / ``session_break_utc_minute`` stored in
    ``spread_profile.json`` and passed to ``build_renko()``.
  - QC metrics (coverage, gap histogram, spike count, OHLC integrity) are
    computed once and persisted alongside the SessionProfile so the
    qualification pipeline can gate on data quality.

Canonical usage::

    from kinetra.renko.session import detect_session_break, SessionProfile

    profile = detect_session_break(m1_df, symbol="XAUUSD")
    # Pass to build_renko:
    bricks = build_renko(closes, brick_size=...,
                         session_break_minutes=profile.session_break_minutes)

See Also:
    - ``kinetra/renko/brick_engine.py`` — uses session_break_minutes param
    - ``kinetra/renko/qualify.py`` — calls detect_session_break() in pipeline
    - ``docs/MANUAL.md §29.2–29.3`` — spec
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────

# Gap threshold used throughout the pipeline (minutes)
DEFAULT_SESSION_BREAK_MINUTES: float = 30.0

# Maximum consecutive-missing-minutes window considered a session break
# (gaps longer than this are likely weekend or holiday closures, not a
# recurring daily break that should be profiled as session_break_utc)
MAX_SESSION_BREAK_MINUTES: float = 240.0

# Minimum fraction of trading hours in a day that must be missing to call it
# a "dominant session break" (avoids flagging random data gaps)
MIN_DOMINANT_HOUR_FRACTION: float = 0.50

# Spike detection: flag a bar if (high - low) > SPIKE_IQR_MULT * IQR(ranges)
SPIKE_IQR_MULT: float = 10.0

# Spike detection: minimum absolute move (in price units) to also count as spike
SPIKE_MIN_PRICE_MOVE: float = 0.0  # 0 = rely entirely on IQR gate

# Minimum bars required to compute a meaningful session profile
MIN_BARS_FOR_PROFILE: int = 500


# ══════════════════════════════════════════════════════════════════════════════
# Data containers
# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class GapRecord:
    """One gap event in the M1 series.

    Attributes
    ----------
    start_time : str
        ISO-8601 UTC timestamp of the last bar before the gap.
    end_time : str
        ISO-8601 UTC timestamp of the first bar after the gap.
    gap_minutes : float
        Duration of the gap in minutes.
    """

    start_time: str
    end_time: str
    gap_minutes: float


@dataclass
class QCMetrics:
    """Data-quality metrics computed from a raw M1 series.

    Attributes
    ----------
    total_bars : int
        Number of rows in the M1 file.
    expected_minutes : int
        Expected number of minutes from first to last bar
        (calendar minutes, ignoring weekends/holidays).
    coverage_ratio : float
        ``total_bars / expected_minutes``.  Values > 1.0 indicate
        duplicate rows; values < 0.5 indicate heavy data loss.
    n_gaps : int
        Number of gap events (consecutive missing-minute runs).
    top_gaps : list[GapRecord]
        Up to 10 largest gaps, sorted descending by duration.
    spike_count : int
        Number of bars where the high–low range is an outlier (>
        ``SPIKE_IQR_MULT`` × IQR of all ranges).
    spike_fraction : float
        ``spike_count / total_bars``.
    ohlc_integrity_ok : bool
        True if every row satisfies ``low <= open,close <= high``.
    ohlc_violations : int
        Count of rows violating OHLC integrity.
    has_volume : bool
        True if a non-zero volume column was detected.
    """

    total_bars: int
    expected_minutes: int
    coverage_ratio: float
    n_gaps: int
    top_gaps: List[GapRecord]
    spike_count: int
    spike_fraction: float
    ohlc_integrity_ok: bool
    ohlc_violations: int
    has_volume: bool

    def to_dict(self) -> Dict:
        d = asdict(self)
        d["top_gaps"] = [asdict(g) for g in self.top_gaps]
        return d


@dataclass
class SessionProfile:
    """Broker fingerprint and session-break parameters for one M1 instrument.

    Produced by :func:`detect_session_break` and persisted alongside
    ``spread_profile.json`` so that the Renko pipeline can consume it
    without re-scanning the raw data.

    Attributes
    ----------
    symbol : str
        Instrument symbol (e.g. "XAUUSD").
    broker_source : str
        Broker identifier (e.g. "metaapi", "ctrader", "mt5").
    session_break_minutes : float
        Gap threshold (minutes) to use in ``build_renko()``.
        Equals ``dominant_gap_minutes`` when a daily pattern is detected;
        otherwise ``DEFAULT_SESSION_BREAK_MINUTES``.
    dominant_gap_minutes : float
        Duration (minutes) of the dominant recurring daily gap, or 0.0 if
        no dominant pattern was found.
    session_break_utc_hour : Optional[int]
        UTC hour at which the daily session break typically starts, or None.
    session_break_utc_minute : int
        UTC minute of the break start within the hour (default 0).
    session_break_duration_minutes : float
        Observed typical duration of the dominant daily break.
    weekend_bar_count : int
        Number of Saturday/Sunday bars detected in the data.
    data_start : str
        ISO-8601 UTC timestamp of the first bar.
    data_end : str
        ISO-8601 UTC timestamp of the last bar.
    n_bars : int
        Total bar count used to compute this profile.
    qc : QCMetrics
        Full data-quality metrics.
    """

    symbol: str
    broker_source: str
    session_break_minutes: float
    dominant_gap_minutes: float
    session_break_utc_hour: Optional[int]
    session_break_utc_minute: int
    session_break_duration_minutes: float
    weekend_bar_count: int
    data_start: str
    data_end: str
    n_bars: int
    qc: QCMetrics

    def to_dict(self) -> Dict:
        d = asdict(self)
        d["qc"] = self.qc.to_dict()
        return d

    def save(self, path: Path) -> None:
        """Persist the profile to a JSON file (atomic write)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.to_dict(), indent=2))
        tmp.replace(path)

    @classmethod
    def load(cls, path: Path) -> "SessionProfile":
        """Load a profile previously saved with :meth:`save`."""
        raw = json.loads(Path(path).read_text())
        qc_raw = raw.pop("qc")
        top_gaps = [GapRecord(**g) for g in qc_raw.pop("top_gaps", [])]
        qc = QCMetrics(**qc_raw, top_gaps=top_gaps)
        return cls(**raw, qc=qc)


# ══════════════════════════════════════════════════════════════════════════════
# Internal helpers
# ══════════════════════════════════════════════════════════════════════════════


def _ensure_utc_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of ``df`` with a UTC DatetimeIndex.

    Handles both 'time' column and pre-set DatetimeIndex inputs.
    """
    df = df.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        time_col = next(
            (c for c in df.columns if c.lower() in ("time", "datetime", "date", "timestamp")),
            None,
        )
        if time_col is None:
            raise ValueError(
                "Cannot find a time column in DataFrame; "
                "expected one of: time, datetime, date, timestamp"
            )
        df.index = pd.to_datetime(df[time_col], utc=True, errors="coerce")
    else:
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        else:
            df.index = df.index.tz_convert("UTC")
    df = df.sort_index()
    return df


def _compute_gaps(
    times: pd.DatetimeIndex,
) -> Tuple[np.ndarray, List[GapRecord]]:
    """Compute per-consecutive-pair minute gaps.

    Returns
    -------
    gap_minutes : np.ndarray  (length N-1)
    top_gaps : List[GapRecord]  (up to 10, sorted descending)
    """
    if len(times) < 2:
        return np.array([], dtype=np.float64), []

    # pandas DatetimeIndex may be stored as microseconds (us) or nanoseconds (ns)
    # depending on Python / pandas version.  Convert via timedelta to stay correct.
    td_deltas = times[1:] - times[:-1]  # TimedeltaIndex
    deltas = td_deltas.total_seconds().values / 60.0  # minutes, float64
    gap_mask = deltas > 1.5  # actual gaps (> 1 expected minute)
    gap_indices = np.where(gap_mask)[0]

    records: List[GapRecord] = []
    for idx in gap_indices:
        records.append(
            GapRecord(
                start_time=times[idx].isoformat(),
                end_time=times[idx + 1].isoformat(),
                gap_minutes=float(deltas[idx]),
            )
        )

    records.sort(key=lambda r: r.gap_minutes, reverse=True)
    return deltas, records[:10]


def _detect_dominant_daily_gap(
    times: pd.DatetimeIndex,
    gap_minutes_arr: np.ndarray,
) -> Tuple[Optional[int], int, float, float]:
    """Find the UTC hour of a dominant recurring daily session break.

    Strategy: for each gap > DEFAULT_SESSION_BREAK_MINUTES and <
    MAX_SESSION_BREAK_MINUTES, record the UTC hour of the gap start.
    The hour that appears in at least ``MIN_DOMINANT_HOUR_FRACTION`` of
    trading days is the dominant session break hour.

    Returns
    -------
    utc_hour : Optional[int]
        Dominant break start hour, or None if not found.
    utc_minute : int
        Minute of break start (mode within the dominant hour).
    gap_duration_minutes : float
        Typical gap duration (median within the dominant hour).
    session_break_minutes : float
        Recommended ``session_break_minutes`` for ``build_renko()``.
    """
    if len(times) < 2 or len(gap_minutes_arr) == 0:
        return None, 0, 0.0, DEFAULT_SESSION_BREAK_MINUTES

    mask = (gap_minutes_arr > DEFAULT_SESSION_BREAK_MINUTES) & (
        gap_minutes_arr < MAX_SESSION_BREAK_MINUTES
    )
    gap_idx = np.where(mask)[0]

    if len(gap_idx) == 0:
        return None, 0, 0.0, DEFAULT_SESSION_BREAK_MINUTES

    # UTC hour of the bar just before each qualifying gap
    gap_start_times = times[gap_idx]
    hours = gap_start_times.hour.values
    gap_durations = gap_minutes_arr[gap_idx]

    # Count unique calendar days in the series (proxy for trading days)
    n_calendar_days = max(len(np.unique(times.normalize())), 1)

    # Find most frequent hour
    hour_vals, hour_counts = np.unique(hours, return_counts=True)
    best_idx = int(np.argmax(hour_counts))
    best_hour = int(hour_vals[best_idx])
    best_count = int(hour_counts[best_idx])

    fraction = best_count / n_calendar_days
    if fraction < MIN_DOMINANT_HOUR_FRACTION:
        # Not recurrent enough — no dominant pattern
        return None, 0, 0.0, DEFAULT_SESSION_BREAK_MINUTES

    # Mode minute within the dominant hour
    hour_mask = hours == best_hour
    minutes_in_hour = gap_start_times[hour_mask].minute.values
    if len(minutes_in_hour) > 0:
        min_vals, min_counts = np.unique(minutes_in_hour, return_counts=True)
        best_minute = int(min_vals[np.argmax(min_counts)])
    else:
        best_minute = 0

    # Median gap duration for this hour
    typical_duration = float(np.median(gap_durations[hour_mask]))

    # Recommended break threshold: half the typical gap length, clamped
    recommended = float(np.clip(typical_duration / 2.0, 15.0, MAX_SESSION_BREAK_MINUTES))

    return best_hour, best_minute, typical_duration, recommended


def _count_spikes(df: pd.DataFrame) -> Tuple[int, float]:
    """Count high-low range outliers (proxy for data spikes).

    A bar is a spike if its range exceeds ``SPIKE_IQR_MULT`` times the
    inter-quartile range (Q75 - Q25) of all ranges, AND exceeds
    ``SPIKE_MIN_PRICE_MOVE`` in absolute terms.

    Returns
    -------
    spike_count : int
    spike_fraction : float
    """
    if df.empty or "high" not in df.columns or "low" not in df.columns:
        return 0, 0.0

    ranges = (df["high"] - df["low"]).values.astype(np.float64)
    ranges = ranges[np.isfinite(ranges)]
    if len(ranges) < 4:
        return 0, 0.0

    q25, q75 = np.percentile(ranges, [25, 75])
    iqr = q75 - q25
    if iqr <= 0:
        return 0, 0.0

    threshold = SPIKE_IQR_MULT * iqr
    spike_mask = ranges > threshold
    if SPIKE_MIN_PRICE_MOVE > 0:
        spike_mask = spike_mask & (ranges > SPIKE_MIN_PRICE_MOVE)

    spike_count = int(np.sum(spike_mask))
    spike_fraction = spike_count / len(ranges)
    return spike_count, spike_fraction


def _check_ohlc_integrity(df: pd.DataFrame) -> Tuple[bool, int]:
    """Verify ``low <= open,close <= high`` for each bar.

    Returns
    -------
    integrity_ok : bool
    violation_count : int
    """
    needed = {"open", "high", "low", "close"}
    if not needed.issubset(set(df.columns)):
        return True, 0  # can't check, assume ok

    o = df["open"].values.astype(np.float64)
    h = df["high"].values.astype(np.float64)
    lo = df["low"].values.astype(np.float64)
    c = df["close"].values.astype(np.float64)

    violations = (lo > h) | (o > h) | (c > h) | (lo > o) | (lo > c)
    # Ignore rows with NaN values
    valid = np.isfinite(o) & np.isfinite(h) & np.isfinite(lo) & np.isfinite(c)
    violation_count = int(np.sum(violations & valid))
    return violation_count == 0, violation_count


def _count_weekend_bars(times: pd.DatetimeIndex) -> int:
    """Count bars that fall on Saturday (5) or Sunday (6) by ISO weekday."""
    if len(times) == 0:
        return 0
    weekdays = times.weekday  # Monday=0, Sunday=6
    return int(np.sum((weekdays == 5) | (weekdays == 6)))


def _expected_minutes(times: pd.DatetimeIndex) -> int:
    """Rough expected-minutes count: calendar minutes from first to last bar."""
    if len(times) < 2:
        return len(times)
    total_seconds = (times[-1] - times[0]).total_seconds()
    return max(int(total_seconds / 60), 1)


# ══════════════════════════════════════════════════════════════════════════════
# Public API
# ══════════════════════════════════════════════════════════════════════════════


def detect_session_break(
    df: pd.DataFrame,
    symbol: str = "",
    broker_source: str = "unknown",
) -> SessionProfile:
    """Fingerprint an M1 OHLCV DataFrame and detect the dominant session break.

    This is the canonical entry point for broker-fingerprinting.  Call this
    BEFORE ``build_renko()`` for any new instrument or any new data file.
    The returned :class:`SessionProfile` should be saved to
    ``data/renko_qualified/<symbol>/session_profile.json`` and its
    ``session_break_minutes`` field passed to ``build_renko()``.

    Parameters
    ----------
    df : pd.DataFrame
        Raw M1 OHLCV data.  Must contain at minimum a time column (or a
        DatetimeIndex) and a ``close`` column.  Optionally ``open``,
        ``high``, ``low``, ``volume`` for full QC.
    symbol : str
        Instrument symbol for labelling (e.g. "XAUUSD").
    broker_source : str
        Broker identifier for traceability (e.g. "metaapi", "ctrader").

    Returns
    -------
    SessionProfile
        Full broker fingerprint including QC metrics and recommended
        ``session_break_minutes`` for ``build_renko()``.

    Raises
    ------
    ValueError
        If ``df`` is empty or has fewer than ``MIN_BARS_FOR_PROFILE`` bars.

    Notes
    -----
    For instruments where no dominant daily gap is found (e.g. FX majors on
    some brokers that fill gaps with synthetic bars) the returned
    ``session_break_minutes`` defaults to ``DEFAULT_SESSION_BREAK_MINUTES``
    (30 min), which is the safe conservative value.

    Example
    -------
    >>> profile = detect_session_break(m1_df, symbol="XAUUSD",
    ...                                broker_source="metaapi")
    >>> profile.session_break_utc_hour
    21
    >>> bricks = build_renko(m1_df["close"], brick_size=0.50,
    ...                      session_break_minutes=profile.session_break_minutes)
    """
    if df is None or df.empty:
        raise ValueError("df is empty — cannot detect session break")

    df_utc = _ensure_utc_datetime_index(df)
    n_bars = len(df_utc)

    if n_bars < MIN_BARS_FOR_PROFILE:
        raise ValueError(
            f"Only {n_bars} bars available; need >= {MIN_BARS_FOR_PROFILE} "
            "for a meaningful session profile"
        )

    times: pd.DatetimeIndex = df_utc.index  # type: ignore[assignment]

    # ── Gap analysis ────────────────────────────────────────────────────
    gap_minutes_arr, top_gaps = _compute_gaps(times)
    n_gaps = int(np.sum(gap_minutes_arr > 1.5)) if len(gap_minutes_arr) > 0 else 0

    # ── Dominant daily session break ────────────────────────────────────
    utc_hour, utc_minute, gap_duration, session_break_minutes = _detect_dominant_daily_gap(
        times, gap_minutes_arr
    )

    # ── Spike detection ─────────────────────────────────────────────────
    spike_count, spike_fraction = _count_spikes(df_utc)

    # ── OHLC integrity ──────────────────────────────────────────────────
    ohlc_ok, ohlc_violations = _check_ohlc_integrity(df_utc)

    # ── Coverage ────────────────────────────────────────────────────────
    expected_min = _expected_minutes(times)
    coverage_ratio = round(n_bars / max(expected_min, 1), 4)

    # ── Weekend bars ────────────────────────────────────────────────────
    weekend_bars = _count_weekend_bars(times)

    # ── Volume presence ─────────────────────────────────────────────────
    has_volume = (
        "volume" in df_utc.columns
        and df_utc["volume"].notna().any()
        and float(df_utc["volume"].sum()) > 0
    )

    qc = QCMetrics(
        total_bars=n_bars,
        expected_minutes=expected_min,
        coverage_ratio=coverage_ratio,
        n_gaps=n_gaps,
        top_gaps=top_gaps,
        spike_count=spike_count,
        spike_fraction=round(spike_fraction, 6),
        ohlc_integrity_ok=ohlc_ok,
        ohlc_violations=ohlc_violations,
        has_volume=has_volume,
    )

    data_start = times[0].isoformat()
    data_end = times[-1].isoformat()

    profile = SessionProfile(
        symbol=symbol,
        broker_source=broker_source,
        session_break_minutes=session_break_minutes,
        dominant_gap_minutes=gap_duration,
        session_break_utc_hour=utc_hour,
        session_break_utc_minute=utc_minute,
        session_break_duration_minutes=gap_duration,
        weekend_bar_count=weekend_bars,
        data_start=data_start,
        data_end=data_end,
        n_bars=n_bars,
        qc=qc,
    )

    if utc_hour is not None:
        logger.info(
            "%s: dominant session break at %02d:%02d UTC (~%.0f min); session_break_minutes=%.1f",
            symbol or "?",
            utc_hour,
            utc_minute,
            gap_duration,
            session_break_minutes,
        )
    else:
        logger.info(
            "%s: no dominant daily session break detected; "
            "using default session_break_minutes=%.1f",
            symbol or "?",
            session_break_minutes,
        )

    return profile


def clamp_spikes(
    df: pd.DataFrame,
    *,
    iqr_mult: float = SPIKE_IQR_MULT,
    min_price_move: float = 0.0,
    method: str = "clamp",
) -> pd.DataFrame:
    """
    Remove or clamp artefactual spike bars from an M1 OHLCV DataFrame.

    A bar is identified as a spike when its high-low range exceeds
    ``iqr_mult`` times the inter-quartile range (Q75 - Q25) of all bar
    ranges, AND (optionally) exceeds ``min_price_move`` in absolute terms.

    This is a **data-hygiene pre-processing step** to be called *before*
    :func:`kinetra.renko.brick_engine.build_renko` when the session
    profile reports ``qc.spike_count > 0``.  It prevents artefactual
    bricks from large resume-of-session jumps corrupting FlipRate, Markov,
    and VPIN statistics.

    .. note::
        Renko is inherently noise-filtering (small wiggles never form bricks).
        This function targets *data artefacts* — bars with ranges 10× the
        IQR that represent broker quote errors or resume-of-session jumps —
        not normal volatility.

    Parameters
    ----------
    df : pd.DataFrame
        M1 OHLCV DataFrame.  Must contain at minimum a ``close`` column.
        If ``high`` / ``low`` are present they are used for detection;
        otherwise detection falls back to close-to-close moves.
    iqr_mult : float, default 10.0
        Bars with range > ``iqr_mult * IQR(ranges)`` are classified as
        spikes.  Increase to be more lenient, decrease to be stricter.
    min_price_move : float, default 0.0
        Additional absolute threshold.  When > 0, a bar must *also* exceed
        this range to be classified as a spike.  Set to a small multiple of
        typical ATR to avoid clamping genuine large-move sessions.
    method : {"clamp", "drop"}
        How to handle spike bars:

        - ``"clamp"`` *(default)*: replace the bar's ``high`` and ``low``
          with ``max(open, close)`` and ``min(open, close)`` respectively,
          and leave ``open`` / ``close`` unchanged.  This keeps the bar in
          the series (preserving continuity for session-break detection)
          but prevents it from generating artefactual bricks.
        - ``"drop"``: remove the spike bars entirely from the DataFrame.
          Use only when the downstream pipeline can tolerate gaps (e.g.
          after session-break detection has already been run).

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame with spike bars clamped or removed.
        The index and all columns other than ``high`` / ``low`` are
        preserved (``clamp`` mode) or the rows are dropped (``drop`` mode).
        Returns the original DataFrame unchanged if no spikes are found or
        if the required columns are missing.

    Raises
    ------
    ValueError
        If ``method`` is not ``"clamp"`` or ``"drop"``.

    Examples
    --------
    >>> profile = detect_session_break(m1_df, symbol="XAUUSD")
    >>> if profile.qc.spike_count > 0:
    ...     m1_df = clamp_spikes(m1_df)
    >>> bricks = build_renko(m1_df["close"],
    ...                      brick_size=0.5,
    ...                      session_break_minutes=profile.session_break_minutes)
    """
    if method not in ("clamp", "drop"):
        raise ValueError(f"method must be 'clamp' or 'drop', got {method!r}")

    if df.empty:
        return df

    # ── Detect spike bars ────────────────────────────────────────────────────
    have_hl = "high" in df.columns and "low" in df.columns
    have_close = "close" in df.columns

    if have_hl:
        ranges = (df["high"] - df["low"]).values.astype(np.float64)
    elif have_close:
        # Fall back to abs(close diff) as a proxy range
        closes = df["close"].values.astype(np.float64)
        ranges = np.abs(np.diff(closes, prepend=closes[0]))
    else:
        # Cannot detect — return unchanged
        logger.debug("clamp_spikes: no high/low or close columns found; returning unchanged")
        return df

    ranges_finite = ranges[np.isfinite(ranges)]
    if len(ranges_finite) < 4:
        return df

    q25, q75 = np.percentile(ranges_finite, [25, 75])
    iqr = q75 - q25
    if iqr <= 0:
        return df

    threshold = iqr_mult * iqr
    spike_mask = np.isfinite(ranges) & (ranges > threshold)
    if min_price_move > 0:
        spike_mask = spike_mask & (ranges > min_price_move)

    n_spikes = int(np.sum(spike_mask))
    if n_spikes == 0:
        return df

    logger.info(
        "clamp_spikes: %d spike bar(s) detected (threshold=%.6f, method=%s)",
        n_spikes,
        threshold,
        method,
    )

    # ── Apply clamping or dropping ────────────────────────────────────────────
    result = df.copy()

    if method == "drop":
        return result.loc[~spike_mask].reset_index(drop=False)

    # method == "clamp"
    if not have_hl:
        # Nothing to clamp without high/low columns
        logger.debug("clamp_spikes: no high/low columns; cannot clamp, returning unchanged")
        return df

    spike_idx = np.where(spike_mask)[0]

    have_open = "open" in result.columns

    for i in spike_idx:
        row_loc = result.index[i]
        close_val = float(result.at[row_loc, "close"]) if have_close else 0.0
        open_val = float(result.at[row_loc, "open"]) if have_open else close_val

        clamped_high = max(open_val, close_val)
        clamped_low = min(open_val, close_val)

        result.at[row_loc, "high"] = clamped_high
        result.at[row_loc, "low"] = clamped_low

    return result


def session_break_minutes_for(
    profile_path: Path,
    fallback: float = DEFAULT_SESSION_BREAK_MINUTES,
) -> float:
    """Load ``session_break_minutes`` from a persisted :class:`SessionProfile`.

    Convenience helper for use in ``build_renko()`` call sites that have a
    path to a saved profile but don't need the full object.

    Parameters
    ----------
    profile_path : Path
        Path to ``session_profile.json``.
    fallback : float
        Value to return if the file does not exist or cannot be parsed.

    Returns
    -------
    float
        ``session_break_minutes`` from the profile, or ``fallback``.
    """
    try:
        p = SessionProfile.load(Path(profile_path))
        return p.session_break_minutes
    except Exception:
        logger.debug(
            "Could not load session profile from %s; using fallback %.1f", profile_path, fallback
        )
        return fallback
