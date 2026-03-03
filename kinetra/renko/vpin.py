"""
Renko VPIN Module — Volume-Synchronized Probability of Informed Trading
========================================================================

Implements the **Bulk Volume Classification (BVC)** approach to VPIN
estimation as specified in ``RENKO_KINETRA_DESIGN_SPEC.md §5.5``.

Algorithm
---------
1. Divide M1 price/volume series into **equal-volume buckets** (not time).
2. For each bucket, classify volume as buy/sell using BVC:
   ``buy_frac = Φ((close − open) / σ(close − open))``
   ``buy_vol  = bucket_vol × buy_frac``
   ``sell_vol = bucket_vol × (1 − buy_frac)``
3. Order imbalance per bucket:
   ``OI = |buy_vol − sell_vol| / bucket_vol``
4. VPIN = rolling mean of OI over the last *n_buckets*.

Usage
-----
::

    from kinetra.renko.vpin import compute_vpin, vpin_timeseries, vpin_baseline

    # Single scalar VPIN from M1 data
    vpin = compute_vpin(df_m1, bucket_size=50_000, n_buckets=50)

    # Full time-series (one VPIN value per bucket boundary)
    ts = vpin_timeseries(df_m1, bucket_size=50_000, n_buckets=50)

    # Baseline statistics for normalisation
    baseline = vpin_baseline(df_m1, bucket_size=50_000, n_buckets=50)

Design Decisions
----------------
- **Volume buckets, not time buckets:** VPIN intentionally de-synchronises
  from clock time.  High-activity periods produce more buckets per hour,
  giving faster signal updates when it matters most.
- **BVC over tick-rule:** BVC uses bar-level (open, close) information
  and a Gaussian CDF.  It avoids the need for tick-level trade data that
  retail brokers rarely provide.
- **Adaptive bucket sizing:** ``auto_bucket_size()`` derives bucket size
  from median daily volume — no magic numbers.
- **NaN shield:** If σ(close − open) ≈ 0 the BVC fraction is set to 0.5
  (equal buy/sell — no information).

This module is the canonical VPIN implementation for Renko Kinetra.
All RL environments and circuit breakers should import from here.

See Also
--------
- ``kinetra.monitoring.circuit_breakers`` — VPIN > extreme → flatten
- ``kinetra.rl.risk_env`` — VPIN observations in Layer 3 risk env
- ``kinetra.renko.dsp`` — DSP instrument filtering (Phase 2)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import norm  # type: ignore[import-untyped]

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# Constants
# ══════════════════════════════════════════════════════════════════════════════

# Minimum standard deviation for BVC — below this we treat price change
# as zero-information (buy_frac = 0.5).
_MIN_PRICE_STD = 1e-12

# Minimum bars required to produce meaningful VPIN
_MIN_BARS = 100

# Default number of buckets for the rolling VPIN window
_DEFAULT_N_BUCKETS = 50


# ══════════════════════════════════════════════════════════════════════════════
# Data containers
# ══════════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True, slots=True)
class VPINBucket:
    """
    A single equal-volume bucket with classified buy/sell volume.

    Attributes
    ----------
    bucket_idx : int
        Sequential bucket index.
    start_bar : int
        Index of the first M1 bar contributing to this bucket.
    end_bar : int
        Index of the last M1 bar contributing to this bucket (inclusive).
    total_volume : float
        Total volume accumulated in this bucket.
    buy_volume : float
        Volume classified as buy (via BVC).
    sell_volume : float
        Volume classified as sell (via BVC).
    order_imbalance : float
        |buy − sell| / total.  Range [0, 1].
    start_time : Optional[pd.Timestamp]
        Timestamp of the first bar (for time-mapping).
    end_time : Optional[pd.Timestamp]
        Timestamp of the last bar (for time-mapping).
    """

    bucket_idx: int
    start_bar: int
    end_bar: int
    total_volume: float
    buy_volume: float
    sell_volume: float
    order_imbalance: float
    start_time: Optional[pd.Timestamp] = None
    end_time: Optional[pd.Timestamp] = None


@dataclass(frozen=True, slots=True)
class VPINBaseline:
    """
    Baseline VPIN statistics for an instrument.

    Used to normalise live VPIN observations into [0, 1] range for the
    Layer 3 risk environment.

    Attributes
    ----------
    mean : float
        Mean VPIN over the entire sample.
    std : float
        Standard deviation of VPIN.
    p50 : float
        Median VPIN.
    p75 : float
        75th percentile VPIN.
    p90 : float
        90th percentile VPIN.
    p95 : float
        95th percentile VPIN.
    p99 : float
        99th percentile VPIN (extreme threshold candidate).
    n_buckets_total : int
        Total number of buckets in the sample.
    bucket_size : int
        Volume bucket size used.
    rolling_window : int
        Number of buckets used for the rolling mean.
    """

    mean: float
    std: float
    p50: float
    p75: float
    p90: float
    p95: float
    p99: float
    n_buckets_total: int
    bucket_size: int
    rolling_window: int


@dataclass(slots=True)
class VPINTimeSeries:
    """
    Full VPIN time series mapped back to clock time.

    Attributes
    ----------
    bucket_times : np.ndarray
        End-time of each bucket (datetime64).
    bucket_oi : np.ndarray
        Order imbalance per bucket [0, 1].
    vpin_values : np.ndarray
        Rolling VPIN (one per bucket, NaN for first ``n_buckets - 1``).
    n_buckets_window : int
        Rolling window size used.
    bucket_size : int
        Volume per bucket.
    buckets : list[VPINBucket]
        Raw bucket records.
    """

    bucket_times: np.ndarray
    bucket_oi: np.ndarray
    vpin_values: np.ndarray
    n_buckets_window: int
    bucket_size: int
    buckets: List[VPINBucket] = field(default_factory=list)


# ══════════════════════════════════════════════════════════════════════════════
# Column resolution helpers
# ══════════════════════════════════════════════════════════════════════════════


def _resolve_col(df: pd.DataFrame, candidates: List[str]) -> str:
    """Return the first column name from *candidates* that exists in *df*."""
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(f"None of {candidates} found in DataFrame columns: {list(df.columns)}")


def _resolve_ohlcv(
    df: pd.DataFrame,
) -> Tuple[str, str, str, str, str]:
    """Resolve canonical open/high/low/close/volume column names."""
    open_col = _resolve_col(df, ["open", "Open", "OPEN"])
    high_col = _resolve_col(df, ["high", "High", "HIGH"])
    low_col = _resolve_col(df, ["low", "Low", "LOW"])
    close_col = _resolve_col(df, ["close", "Close", "CLOSE"])
    vol_col = _resolve_col(df, ["volume", "Volume", "VOLUME", "tick_volume", "tickVolume"])
    return open_col, high_col, low_col, close_col, vol_col


def _resolve_time(df: pd.DataFrame) -> Optional[str]:
    """Resolve the time column, returning None if absent."""
    for c in ["time", "Time", "TIME", "datetime", "Datetime", "timestamp", "date"]:
        if c in df.columns:
            return c
    # Fall back to index if it's a DatetimeIndex
    if isinstance(df.index, pd.DatetimeIndex):
        return None  # signal to use df.index
    return None


# ══════════════════════════════════════════════════════════════════════════════
# Core: BVC classification
# ══════════════════════════════════════════════════════════════════════════════


def _bulk_volume_classify(
    opens: np.ndarray,
    closes: np.ndarray,
    volumes: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Classify bar volume as buy/sell using Bulk Volume Classification.

    For each bar:
        Δ = close − open
        σ = std(Δ) over the full array (rolling would add complexity
            with minimal accuracy gain on M1 data)
        buy_fraction = Φ(Δ / σ)         [Φ = standard normal CDF]
        buy_vol  = volume × buy_fraction
        sell_vol = volume × (1 − buy_fraction)

    Parameters
    ----------
    opens : np.ndarray
        Open prices.
    closes : np.ndarray
        Close prices.
    volumes : np.ndarray
        Volumes (tick or real).

    Returns
    -------
    buy_volumes : np.ndarray
        Classified buy volume per bar.
    sell_volumes : np.ndarray
        Classified sell volume per bar.
    """
    delta = closes - opens

    # Robust σ: use the full-sample std of price changes
    sigma = np.std(delta)
    if sigma < _MIN_PRICE_STD:
        # No price movement → 50/50 classification
        buy_frac = np.full_like(delta, 0.5)
    else:
        z_scores = delta / sigma
        buy_frac = norm.cdf(z_scores)

    # Clamp to [0, 1] for numerical safety
    buy_frac = np.clip(buy_frac, 0.0, 1.0)

    buy_vol = volumes * buy_frac
    sell_vol = volumes * (1.0 - buy_frac)

    return buy_vol, sell_vol


# ══════════════════════════════════════════════════════════════════════════════
# Core: volume bucketing
# ══════════════════════════════════════════════════════════════════════════════


def _create_volume_buckets(
    buy_volumes: np.ndarray,
    sell_volumes: np.ndarray,
    volumes: np.ndarray,
    bucket_size: float,
    times: Optional[np.ndarray] = None,
) -> List[VPINBucket]:
    """
    Partition the bar-level classified volumes into equal-volume buckets.

    Each bucket accumulates bars until ``bucket_size`` volume is reached.
    Partial final buckets are discarded (insufficient information).

    Parameters
    ----------
    buy_volumes : np.ndarray
        Per-bar buy volume (from BVC).
    sell_volumes : np.ndarray
        Per-bar sell volume (from BVC).
    volumes : np.ndarray
        Per-bar total volume.
    bucket_size : float
        Target volume per bucket.
    times : np.ndarray or None
        Per-bar timestamps for time mapping.

    Returns
    -------
    list[VPINBucket]
        Completed buckets.  The last (incomplete) bucket is discarded.
    """
    n = len(volumes)
    if n == 0 or bucket_size <= 0:
        return []

    buckets: List[VPINBucket] = []
    bucket_idx = 0
    start_bar = 0
    accum_vol = 0.0
    accum_buy = 0.0
    accum_sell = 0.0

    for i in range(n):
        bar_vol = volumes[i]
        if bar_vol <= 0:
            continue

        remaining_to_fill = bucket_size - accum_vol

        if bar_vol < remaining_to_fill:
            # This bar doesn't fill the bucket — accumulate
            accum_vol += bar_vol
            accum_buy += buy_volumes[i]
            accum_sell += sell_volumes[i]
        else:
            # This bar fills (and possibly overflows) the bucket
            # Proportional allocation of the bar that fills the bucket
            fill_frac = remaining_to_fill / bar_vol if bar_vol > 0 else 1.0
            fill_frac = min(fill_frac, 1.0)

            accum_buy += buy_volumes[i] * fill_frac
            accum_sell += sell_volumes[i] * fill_frac
            accum_vol = bucket_size  # exactly filled

            # Compute order imbalance for this bucket
            oi = abs(accum_buy - accum_sell) / accum_vol if accum_vol > 0 else 0.0

            start_time = None
            end_time = None
            if times is not None:
                start_time = pd.Timestamp(times[start_bar])
                end_time = pd.Timestamp(times[i])

            buckets.append(
                VPINBucket(
                    bucket_idx=bucket_idx,
                    start_bar=start_bar,
                    end_bar=i,
                    total_volume=accum_vol,
                    buy_volume=accum_buy,
                    sell_volume=accum_sell,
                    order_imbalance=float(np.clip(oi, 0.0, 1.0)),
                    start_time=start_time,
                    end_time=end_time,
                )
            )

            # Carry the overflow into the next bucket
            overflow_vol = bar_vol - remaining_to_fill
            overflow_buy = buy_volumes[i] * (1.0 - fill_frac)
            overflow_sell = sell_volumes[i] * (1.0 - fill_frac)

            bucket_idx += 1
            start_bar = i
            accum_vol = overflow_vol
            accum_buy = overflow_buy
            accum_sell = overflow_sell

            # Handle the case where one bar overflows multiple buckets
            while accum_vol >= bucket_size:
                sub_frac = bucket_size / accum_vol if accum_vol > 0 else 1.0
                sub_buy = accum_buy * sub_frac
                sub_sell = accum_sell * sub_frac
                sub_oi = abs(sub_buy - sub_sell) / bucket_size if bucket_size > 0 else 0.0

                buckets.append(
                    VPINBucket(
                        bucket_idx=bucket_idx,
                        start_bar=i,
                        end_bar=i,
                        total_volume=bucket_size,
                        buy_volume=sub_buy,
                        sell_volume=sub_sell,
                        order_imbalance=float(np.clip(sub_oi, 0.0, 1.0)),
                        start_time=pd.Timestamp(times[i]) if times is not None else None,
                        end_time=pd.Timestamp(times[i]) if times is not None else None,
                    )
                )
                accum_vol -= bucket_size
                accum_buy -= sub_buy
                accum_sell -= sub_sell
                bucket_idx += 1

    # Discard incomplete final bucket (insufficient volume = unreliable)
    return buckets


# ══════════════════════════════════════════════════════════════════════════════
# Core: VPIN computation
# ══════════════════════════════════════════════════════════════════════════════


def compute_vpin(
    df: pd.DataFrame,
    bucket_size: Optional[int] = None,
    n_buckets: int = _DEFAULT_N_BUCKETS,
) -> float:
    """
    Compute the current VPIN value from M1 OHLCV data.

    This is the primary entry point for a **single scalar** VPIN reading.
    Returns the most recent rolling-mean order imbalance.

    Parameters
    ----------
    df : pd.DataFrame
        M1 OHLCV data with columns: open, close, volume (minimum).
        Additional columns are ignored.
    bucket_size : int or None
        Volume per bucket.  If ``None``, auto-derived from median daily
        volume via ``auto_bucket_size()``.
    n_buckets : int
        Number of buckets for the rolling VPIN window (default 50).

    Returns
    -------
    float
        VPIN value in [0, 1].  Returns NaN if insufficient data.

    Raises
    ------
    ValueError
        If ``bucket_size`` is non-positive or DataFrame is too small.

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({
    ...     "open":   [1.0, 1.1, 1.0, 1.2] * 100,
    ...     "close":  [1.1, 1.0, 1.2, 1.1] * 100,
    ...     "volume": [1000] * 400,
    ... })
    >>> vpin = compute_vpin(df, bucket_size=5000, n_buckets=10)
    >>> 0.0 <= vpin <= 1.0 or np.isnan(vpin)
    True
    """
    ts = vpin_timeseries(df, bucket_size=bucket_size, n_buckets=n_buckets)

    # Return the last valid VPIN value
    valid = ts.vpin_values[np.isfinite(ts.vpin_values)]
    if len(valid) == 0:
        return float("nan")

    return float(valid[-1])


def vpin_timeseries(
    df: pd.DataFrame,
    bucket_size: Optional[int] = None,
    n_buckets: int = _DEFAULT_N_BUCKETS,
) -> VPINTimeSeries:
    """
    Compute the full VPIN time series from M1 OHLCV data.

    Returns one VPIN value per bucket (rolling mean of order imbalance
    over the last ``n_buckets`` buckets).  The first ``n_buckets - 1``
    values are NaN (insufficient history).

    Parameters
    ----------
    df : pd.DataFrame
        M1 OHLCV data.
    bucket_size : int or None
        Volume per bucket.  Auto-derived if None.
    n_buckets : int
        Rolling VPIN window (default 50).

    Returns
    -------
    VPINTimeSeries
        Full VPIN time series with bucket details.

    Raises
    ------
    ValueError
        If data is too small or bucket_size is invalid.
    """
    if len(df) < _MIN_BARS:
        raise ValueError(f"Need at least {_MIN_BARS} bars for VPIN computation, got {len(df)}")

    # ── Resolve columns ──────────────────────────────────────────────
    open_col, _h, _l, close_col, vol_col = _resolve_ohlcv(df)
    time_col = _resolve_time(df)

    opens = df[open_col].to_numpy(dtype=np.float64)
    closes = df[close_col].to_numpy(dtype=np.float64)
    volumes = df[vol_col].to_numpy(dtype=np.float64)

    # Replace zero/negative volumes with a tiny epsilon so bucketing
    # doesn't stall (zero-volume bars are uninformative anyway)
    volumes = np.maximum(volumes, 0.0)

    # ── Resolve timestamps ───────────────────────────────────────────
    times: Optional[np.ndarray] = None
    if time_col is not None:
        times = pd.to_datetime(df[time_col]).to_numpy()
    elif isinstance(df.index, pd.DatetimeIndex):
        times = df.index.to_numpy()

    # ── Auto bucket size ─────────────────────────────────────────────
    if bucket_size is None:
        bucket_size = auto_bucket_size(volumes, times)
        logger.debug("Auto bucket_size: %d", bucket_size)

    if bucket_size <= 0:
        raise ValueError(f"bucket_size must be positive, got {bucket_size}")

    if n_buckets < 1:
        raise ValueError(f"n_buckets must be >= 1, got {n_buckets}")

    # ── BVC classification ───────────────────────────────────────────
    buy_vol, sell_vol = _bulk_volume_classify(opens, closes, volumes)

    # ── Volume bucketing ─────────────────────────────────────────────
    buckets = _create_volume_buckets(buy_vol, sell_vol, volumes, bucket_size, times)

    if len(buckets) == 0:
        logger.warning(
            "No complete volume buckets formed (bucket_size=%d, total_vol=%.0f)",
            bucket_size,
            volumes.sum(),
        )
        return VPINTimeSeries(
            bucket_times=np.array([], dtype="datetime64[ns]"),
            bucket_oi=np.array([], dtype=np.float64),
            vpin_values=np.array([], dtype=np.float64),
            n_buckets_window=n_buckets,
            bucket_size=int(bucket_size),
            buckets=[],
        )

    # ── Extract OI array ─────────────────────────────────────────────
    oi_arr = np.array([b.order_imbalance for b in buckets], dtype=np.float64)

    # ── Bucket end times ─────────────────────────────────────────────
    if buckets[0].end_time is not None:
        bucket_times = np.array([b.end_time for b in buckets], dtype="datetime64[ns]")
    else:
        bucket_times = np.arange(len(buckets), dtype="datetime64[ns]")

    # ── Rolling VPIN ─────────────────────────────────────────────────
    vpin_vals = _rolling_mean(oi_arr, n_buckets)

    return VPINTimeSeries(
        bucket_times=bucket_times,
        bucket_oi=oi_arr,
        vpin_values=vpin_vals,
        n_buckets_window=n_buckets,
        bucket_size=int(bucket_size),
        buckets=buckets,
    )


def vpin_baseline(
    df: pd.DataFrame,
    bucket_size: Optional[int] = None,
    n_buckets: int = _DEFAULT_N_BUCKETS,
) -> VPINBaseline:
    """
    Compute baseline VPIN statistics for normalisation.

    Used to establish the "normal" VPIN range for an instrument so that
    live VPIN can be normalised into [0, 1] for the risk env observation.

    Parameters
    ----------
    df : pd.DataFrame
        Historical M1 OHLCV data (at least several weeks recommended).
    bucket_size : int or None
        Volume per bucket.  Auto-derived if None.
    n_buckets : int
        Rolling VPIN window (default 50).

    Returns
    -------
    VPINBaseline
        Baseline statistics.

    Raises
    ------
    ValueError
        If insufficient data to compute meaningful baseline.
    """
    ts = vpin_timeseries(df, bucket_size=bucket_size, n_buckets=n_buckets)

    valid = ts.vpin_values[np.isfinite(ts.vpin_values)]
    if len(valid) < 2:
        raise ValueError(
            f"Insufficient VPIN data for baseline: {len(valid)} valid readings "
            f"(need at least 2).  Try reducing bucket_size."
        )

    return VPINBaseline(
        mean=float(np.mean(valid)),
        std=float(np.std(valid)),
        p50=float(np.percentile(valid, 50)),
        p75=float(np.percentile(valid, 75)),
        p90=float(np.percentile(valid, 90)),
        p95=float(np.percentile(valid, 95)),
        p99=float(np.percentile(valid, 99)),
        n_buckets_total=len(ts.bucket_oi),
        bucket_size=ts.bucket_size,
        rolling_window=n_buckets,
    )


# ══════════════════════════════════════════════════════════════════════════════
# Normalisation helpers
# ══════════════════════════════════════════════════════════════════════════════


def normalise_vpin(
    vpin_value: float,
    baseline: VPINBaseline,
    clip: bool = True,
) -> float:
    """
    Normalise a raw VPIN value to [0, 1] using baseline statistics.

    Uses a percentile-based mapping:
      ``normalised = (vpin − baseline.p50) / (baseline.p99 − baseline.p50)``
    Clipped to [0, 1] by default.

    Parameters
    ----------
    vpin_value : float
        Raw VPIN reading.
    baseline : VPINBaseline
        Baseline statistics from ``vpin_baseline()``.
    clip : bool
        If True, clip output to [0, 1].

    Returns
    -------
    float
        Normalised VPIN.  0 = typical, 1 = extreme.
    """
    if not np.isfinite(vpin_value):
        return 0.0

    denom = baseline.p99 - baseline.p50
    if denom < 1e-12:
        # Degenerate baseline — all VPIN values are the same
        return 0.5

    result = (vpin_value - baseline.p50) / denom

    if clip:
        result = float(np.clip(result, 0.0, 1.0))

    return float(result)


def normalise_vpin_zscore(
    vpin_value: float,
    baseline: VPINBaseline,
    clip_sigma: float = 4.0,
) -> float:
    """
    Normalise VPIN via z-score, then sigmoid-map to [0, 1].

    More responsive to outliers than percentile normalisation.
    Uses ``z = (vpin − mean) / std`` followed by a sigmoid.

    Parameters
    ----------
    vpin_value : float
        Raw VPIN reading.
    baseline : VPINBaseline
        Baseline statistics.
    clip_sigma : float
        Number of standard deviations mapped to the [0, 1] range.

    Returns
    -------
    float
        Normalised VPIN in [0, 1].
    """
    if not np.isfinite(vpin_value) or baseline.std < 1e-12:
        return 0.5

    z = (vpin_value - baseline.mean) / baseline.std
    z_clipped = np.clip(z, -clip_sigma, clip_sigma)

    # Linear map from [-clip_sigma, clip_sigma] → [0, 1]
    return float((z_clipped + clip_sigma) / (2.0 * clip_sigma))


# ══════════════════════════════════════════════════════════════════════════════
# Adaptive bucket sizing
# ══════════════════════════════════════════════════════════════════════════════


def auto_bucket_size(
    volumes: np.ndarray,
    times: Optional[np.ndarray] = None,
    buckets_per_day: int = 50,
) -> int:
    """
    Derive a volume bucket size from median daily volume.

    The goal is approximately ``buckets_per_day`` buckets per trading day,
    so the VPIN updates ~50 times per day with default settings.

    Parameters
    ----------
    volumes : np.ndarray
        Per-bar volumes (M1 expected: ~1440 bars/day).
    times : np.ndarray or None
        Per-bar timestamps.  If provided, daily volume is computed
        precisely.  If None, assumes M1 (1440 bars/day).
    buckets_per_day : int
        Target number of buckets per trading day (default 50).

    Returns
    -------
    int
        Recommended volume bucket size (always ≥ 1).
    """
    if len(volumes) == 0:
        return 1

    pos_vol = volumes[volumes > 0]
    if len(pos_vol) == 0:
        return 1

    if times is not None and len(times) == len(volumes):
        # Compute actual daily volume from timestamps
        try:
            ts = pd.to_datetime(times)
            daily = pd.Series(volumes, index=ts).resample("D").sum()
            daily_positive = daily[daily > 0]
            if len(daily_positive) > 0:
                median_daily = float(daily_positive.median())
                size = int(median_daily / max(buckets_per_day, 1))
                return max(size, 1)
        except Exception:
            pass

    # Fallback: assume M1 data (~1440 bars per day)
    bars_per_day = 1440
    total_vol = float(pos_vol.sum())
    n_days = max(len(volumes) / bars_per_day, 1.0)
    avg_daily = total_vol / n_days

    size = int(avg_daily / max(buckets_per_day, 1))
    return max(size, 1)


# ══════════════════════════════════════════════════════════════════════════════
# Multi-instrument aggregation
# ══════════════════════════════════════════════════════════════════════════════


def compute_vpin_multi(
    instruments: dict[str, pd.DataFrame],
    bucket_size: Optional[int] = None,
    n_buckets: int = _DEFAULT_N_BUCKETS,
) -> Tuple[dict[str, float], float, float]:
    """
    Compute VPIN for multiple instruments and return portfolio-level stats.

    This is used by the Layer 3 risk env to populate ``vpin_mean`` and
    ``vpin_max`` observation features.

    Parameters
    ----------
    instruments : dict[str, pd.DataFrame]
        Map of symbol → M1 OHLCV DataFrame.
    bucket_size : int or None
        Volume per bucket.  If None, auto-derived per instrument.
    n_buckets : int
        Rolling VPIN window.

    Returns
    -------
    per_instrument : dict[str, float]
        VPIN per instrument (NaN if computation failed).
    mean_vpin : float
        Mean VPIN across instruments (excluding NaN).
    max_vpin : float
        Max VPIN across instruments (excluding NaN).
    """
    per_instrument: dict[str, float] = {}

    for symbol, df in instruments.items():
        try:
            if len(df) < _MIN_BARS:
                per_instrument[symbol] = float("nan")
                continue
            v = compute_vpin(df, bucket_size=bucket_size, n_buckets=n_buckets)
            per_instrument[symbol] = v
        except Exception as exc:
            logger.warning("VPIN failed for %s: %s", symbol, exc)
            per_instrument[symbol] = float("nan")

    valid_vpins = [v for v in per_instrument.values() if np.isfinite(v)]

    if valid_vpins:
        mean_vpin = float(np.mean(valid_vpins))
        max_vpin = float(np.max(valid_vpins))
    else:
        mean_vpin = 0.0
        max_vpin = 0.0

    return per_instrument, mean_vpin, max_vpin


# ══════════════════════════════════════════════════════════════════════════════
# VPIN regime detection
# ══════════════════════════════════════════════════════════════════════════════


def classify_vpin_regime(
    vpin_value: float,
    baseline: VPINBaseline,
) -> str:
    """
    Classify current VPIN into a regime label.

    Regimes (based on baseline percentiles):
      - ``"normal"``   : VPIN ≤ p75
      - ``"elevated"`` : p75 < VPIN ≤ p90
      - ``"high"``     : p90 < VPIN ≤ p95
      - ``"extreme"``  : VPIN > p95

    Parameters
    ----------
    vpin_value : float
        Current raw VPIN reading.
    baseline : VPINBaseline
        Baseline statistics.

    Returns
    -------
    str
        One of: ``"normal"``, ``"elevated"``, ``"high"``, ``"extreme"``.
    """
    if not np.isfinite(vpin_value):
        return "normal"

    if vpin_value > baseline.p95:
        return "extreme"
    elif vpin_value > baseline.p90:
        return "high"
    elif vpin_value > baseline.p75:
        return "elevated"
    else:
        return "normal"


def is_vpin_extreme(
    vpin_value: float,
    baseline: VPINBaseline,
    threshold_percentile: float = 95.0,
) -> bool:
    """
    Check if VPIN exceeds the extreme threshold.

    Used by circuit breakers as a binary gate.

    Parameters
    ----------
    vpin_value : float
        Current raw VPIN.
    baseline : VPINBaseline
        Baseline statistics.
    threshold_percentile : float
        Percentile threshold (default 95).  The corresponding value
        is looked up from the baseline.

    Returns
    -------
    bool
        True if VPIN exceeds the threshold.
    """
    if not np.isfinite(vpin_value):
        return False

    if threshold_percentile >= 99.0:
        threshold = baseline.p99
    elif threshold_percentile >= 95.0:
        threshold = baseline.p95
    elif threshold_percentile >= 90.0:
        threshold = baseline.p90
    elif threshold_percentile >= 75.0:
        threshold = baseline.p75
    elif threshold_percentile >= 50.0:
        threshold = baseline.p50
    else:
        threshold = baseline.mean

    return vpin_value > threshold


def vpin_excess_kurtosis(values: np.ndarray | list[float], min_samples: int = 20) -> float:
    """
    Compute excess kurtosis of a VPIN value stream.

    Returns NaN when there are insufficient finite samples.
    Uses Fisher definition (Gaussian ~= 0, fat-tail > 0).
    """
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size < max(4, int(min_samples)):
        return float("nan")
    return float(pd.Series(arr).kurt())


# ══════════════════════════════════════════════════════════════════════════════
# Internal utilities
# ══════════════════════════════════════════════════════════════════════════════


def _rolling_mean(arr: np.ndarray, window: int) -> np.ndarray:
    """
    Compute rolling mean with NaN for the warm-up period.

    Uses cumulative sum for O(n) performance.

    Parameters
    ----------
    arr : np.ndarray
        Input array.
    window : int
        Rolling window size.

    Returns
    -------
    np.ndarray
        Rolling mean, with first ``window - 1`` elements as NaN.
    """
    n = len(arr)
    if n == 0:
        return np.array([], dtype=np.float64)

    result = np.full(n, np.nan, dtype=np.float64)

    if window > n:
        # Not enough data for even one complete window
        return result

    if window <= 0:
        return result

    # Cumulative sum approach — O(n)
    cumsum = np.cumsum(arr)
    result[window - 1] = cumsum[window - 1] / window

    if window < n:
        result[window:] = (cumsum[window:] - cumsum[:-window]) / window

    return result


# ══════════════════════════════════════════════════════════════════════════════
# Module-level exports
# ══════════════════════════════════════════════════════════════════════════════

__all__ = [
    # Core functions
    "compute_vpin",
    "vpin_timeseries",
    "vpin_baseline",
    # Multi-instrument
    "compute_vpin_multi",
    # Normalisation
    "normalise_vpin",
    "normalise_vpin_zscore",
    # Regime detection
    "classify_vpin_regime",
    "is_vpin_extreme",
    "vpin_excess_kurtosis",
    # Adaptive sizing
    "auto_bucket_size",
    # Data containers
    "VPINBucket",
    "VPINBaseline",
    "VPINTimeSeries",
]
