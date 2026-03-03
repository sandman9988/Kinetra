"""
Renko DSP (Digital Signal Processing) Module
=============================================

Variance Ratio profiling, brick sizing, friction floor computation,
and regime classification for Renko instrument filtering.

Design decisions (from RENKO_KINETRA_DESIGN_SPEC.md §5):
  - VR (Variance Ratio) on M30 is the primary DSP tool to detect
    trend-persistent scales and derive DSP brick sizes.
  - VR > 1 at scale T → price is persistent/trending at T-bar horizon.
  - Friction floor is law: no brick size below
    max(Y × spread_p50_price, Z × spread_p95_price).
  - Regime classification: TRENDING / WEAK_TREND / RANDOM_WALK / MEAN_REVERTING.

This module is the canonical home for Renko DSP analysis.  All scripts
should import from here (DRY compliance).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import numpy as np

if TYPE_CHECKING:
    from kinetra.renko.backtest import FilterParams

# ══════════════════════════════════════════════════════════════════════════════
# Constants
# ══════════════════════════════════════════════════════════════════════════════

# Default VR scale grid for M30 bars:
# 2 bars = 1 hour  →  288 bars ≈ 6 trading days
M30_VR_SCALES: List[int] = [2, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 288]

# Default VR scale grid for H4 bars (legacy / alternate):
# 2 bars = 8 hours  →  64 bars ≈ ~10 trading days
H4_VR_SCALES: List[int] = [2, 4, 6, 8, 12, 16, 24, 32, 48, 64]

# Regime classification thresholds (§5.1)
_TRENDING_THRESHOLD: float = 1.10
_WEAK_TREND_THRESHOLD: float = 1.02
_RANDOM_WALK_THRESHOLD: float = 0.98


# Minimum VR for portfolio candidacy (§5.1)
DEFAULT_MIN_VR: float = 1.05

# Default friction floor multipliers (§5.3)
DEFAULT_Y_MULT: float = 5.0  # multiplier for spread_p50
DEFAULT_Z_MULT: float = 2.0  # multiplier for spread_p95
DEFAULT_ROLLING_WINDOW_BARS: int = 1440  # M1: ~1 day
DEFAULT_ROLLING_MIN_OBS: int = 180

# Maximum friction ratio (§5.4) — reject if friction_ratio > this
DEFAULT_MAX_FRICTION_RATIO: float = 0.25

# Default friction multiplier for commission component (§29.2)
DEFAULT_FRICTION_MULT: float = 4.0  # 25% friction ratio

# Minimum absolute brick as fraction of median price (§5.2)
_MIN_BRICK_PRICE_FRACTION: float = 0.0001  # 0.01% of price

# Regime labels
REGIME_TRENDING: str = "TRENDING"
REGIME_WEAK_TREND: str = "WEAK_TREND"
REGIME_RANDOM_WALK: str = "RANDOM_WALK"
REGIME_MEAN_REVERTING: str = "MEAN_REVERTING"
REGIME_INSUFFICIENT_DATA: str = "INSUFFICIENT_DATA"


# ══════════════════════════════════════════════════════════════════════════════
# Dataclasses
# ══════════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True, slots=True)
class SpreadProfile:
    """Spread characteristics for one instrument, in both points and price.

    All spread values are measured from CSV data at multiple percentiles
    to support friction floor computation and regime-aware trading.

    Attributes
    ----------
    symbol : str
        Instrument symbol (e.g. "XAUUSD+").
    tick_size : float
        Price per point (minimum price increment).
    spread_p50_pts : float
        Median spread in points (broker units).
    spread_p75_pts : float
        75th percentile spread in points.
    spread_p95_pts : float
        95th percentile spread in points.
    spread_p50_price : float
        Median spread in price units (= p50_pts × tick_size).
    spread_p75_price : float
        75th percentile spread in price units.
    spread_p95_price : float
        95th percentile spread in price units.
    spread_source : str
        Origin of spread data: "csv_m30", "csv_h4", "csv_m1",
        "spec_typical", or "spec_default".
    n_bars : int
        Number of bars used for spread measurement.
    """

    symbol: str
    tick_size: float
    spread_p50_pts: float
    spread_p75_pts: float
    spread_p95_pts: float
    spread_p50_price: float
    spread_p75_price: float
    spread_p95_price: float
    spread_source: str
    n_bars: int

    @classmethod
    def from_points(
        cls,
        symbol: str,
        tick_size: float,
        p50_pts: float,
        p75_pts: float,
        p95_pts: float,
        source: str = "csv_m30",
        n_bars: int = 0,
    ) -> "SpreadProfile":
        """Construct from point-denominated spread percentiles.

        Parameters
        ----------
        symbol : str
            Instrument symbol.
        tick_size : float
            Price per point.  Must be > 0.
        p50_pts, p75_pts, p95_pts : float
            Spread percentiles in points.
        source : str
            Source label for the measurement.
        n_bars : int
            Number of bars used in measurement.

        Returns
        -------
        SpreadProfile
        """
        ts = max(tick_size, 1e-10)
        return cls(
            symbol=symbol,
            tick_size=ts,
            spread_p50_pts=p50_pts,
            spread_p75_pts=p75_pts,
            spread_p95_pts=p95_pts,
            spread_p50_price=p50_pts * ts,
            spread_p75_price=p75_pts * ts,
            spread_p95_price=p95_pts * ts,
            spread_source=source,
            n_bars=n_bars,
        )

    @classmethod
    def from_price(
        cls,
        symbol: str,
        tick_size: float,
        p50_price: float,
        p75_price: float,
        p95_price: float,
        source: str = "direct",
        n_bars: int = 0,
    ) -> "SpreadProfile":
        """Construct from price-denominated spread percentiles.

        Parameters
        ----------
        symbol : str
            Instrument symbol.
        tick_size : float
            Price per point.  Must be > 0.
        p50_price, p75_price, p95_price : float
            Spread percentiles in price units.
        source : str
            Source label for the measurement.
        n_bars : int
            Number of bars in measurement.

        Returns
        -------
        SpreadProfile
        """
        ts = max(tick_size, 1e-10)
        return cls(
            symbol=symbol,
            tick_size=ts,
            spread_p50_pts=p50_price / ts,
            spread_p75_pts=p75_price / ts,
            spread_p95_pts=p95_price / ts,
            spread_p50_price=p50_price,
            spread_p75_price=p75_price,
            spread_p95_price=p95_price,
            spread_source=source,
            n_bars=n_bars,
        )


@dataclass(frozen=True, slots=True)
class FrictionFloor:
    """Computed friction floor for one instrument.

    The friction floor is the minimum acceptable brick size, derived
    from spread characteristics and commission costs. No brick below
    this floor should ever be used — it is physically impossible to
    trade profitably at sub-floor granularity because friction eats the edge.

    The friction floor formula (§5.3, §29.2):
        floor = max(Y × spread_p50_price, Z × spread_p95_price,
                    friction_mult × friction_per_unit)

    Where friction_per_unit = (commission_rt / contract_size) + spread_p50_price

    For ECN brokers:
        - commission_rt = $7.00 (round-trip)
        - spread_p50_price ≈ $0.01-$0.10 (tight spreads)
        - friction_per_unit ≈ $0.08-$0.17
        - floor ≈ $0.32-$0.68 (with friction_mult=4)

    Attributes
    ----------
    symbol : str
        Instrument symbol.
    floor_from_median : float
        Y × spread_p50_price component.
    floor_from_p95 : float
        Z × spread_p95_price component.
    floor_from_commission : float
        Friction multiplier × (commission + spread) per unit.
    floor_price : float
        max(floor_from_median, floor_from_p95, floor_from_commission) — the binding floor.
    dsp_brick : float
        DSP-suggested brick size (before floor enforcement).
    dsp_brick_above_floor : bool
        True if the DSP brick is ≥ the floor.
    effective_min_mult : float
        floor / dsp_brick — the minimum brick multiplier that satisfies
        the floor.  Values > 1.0 mean DSP brick is below floor.
    y_mult : float
        Y multiplier used for floor computation (default 5.0).
    z_mult : float
        Z multiplier used for floor computation (default 2.0).
    friction_mult : float
        Friction multiplier for commission+spread component (default 4.0).
    commission_rt : float
        Round-trip commission in USD per lot.
    contract_size : float
        Contract size (units per lot, e.g., 100 oz for XAUUSD).
    """

    symbol: str
    floor_from_median: float
    floor_from_p95: float
    floor_from_commission: float
    floor_price: float
    dsp_brick: float
    dsp_brick_above_floor: bool
    effective_min_mult: float
    y_mult: float
    z_mult: float
    friction_mult: float = 4.0
    commission_rt: float = 0.0
    contract_size: float = 100.0


@dataclass(frozen=True, slots=True)
class DSPResult:
    """Complete VR-based DSP analysis result for one instrument.

    Attributes
    ----------
    symbol : str
        Instrument symbol.
    n_bars : int
        Number of bars used for VR analysis.
    years : float
        Approximate years of data (based on bar count and timeframe).
    vr_profile : Dict[int, float]
        Full VR profile: scale (bars) → VR value.
    vr_peak_scale : int
        Scale (in bars) with the highest VR — the natural trend horizon.
    vr_peak_scale_hours : float
        Peak scale converted to hours (depends on source timeframe).
    vr_peak_value : float
        VR value at the peak scale.
    regime : str
        Regime classification: TRENDING / WEAK_TREND / RANDOM_WALK /
        MEAN_REVERTING / INSUFFICIENT_DATA.
    dsp_brick_size : float
        Median |displacement| over peak scale — the natural brick size.
    tma_period : int
        TMA period derived from peak scale, clamped to [5, 80].
        Note: TMA is NOT used for flip entry filtering (empirically
        rejected) but is retained for adaptive stop sizing.
    bars_per_hour : float
        Bars per hour for the source timeframe (e.g. 2.0 for M30).
    """

    symbol: str
    n_bars: int
    years: float
    vr_profile: Dict[int, float]
    vr_peak_scale: int
    vr_peak_scale_hours: float
    vr_peak_value: float
    regime: str
    dsp_brick_size: float
    tma_period: int
    bars_per_hour: float = 2.0  # default: M30


# ══════════════════════════════════════════════════════════════════════════════
# Core VR Functions
# ══════════════════════════════════════════════════════════════════════════════


def vr_profile(
    closes: np.ndarray,
    scales: Optional[List[int]] = None,
    min_windows: int = 30,
) -> Dict[int, float]:
    """
    Compute the Variance Ratio at each scale.

    VR(T) = Var(T-bar returns) / (T × Var(1-bar returns))

    VR > 1 at scale T → price is persistent/trending at T-bar horizon.
    VR < 1 → price is mean-reverting at that horizon.
    VR ≈ 1 → random walk (no exploitable structure).

    Parameters
    ----------
    closes : np.ndarray
        Close price array.  Must have at least 3 elements.
    scales : list of int or None
        Scale grid (in bars) to evaluate.  If None, uses
        :data:`M30_VR_SCALES`.
    min_windows : int
        Minimum number of non-overlapping windows required at each
        scale for the estimate to be considered reliable.  Scales
        with fewer windows are excluded from the result.

    Returns
    -------
    Dict[int, float]
        Mapping of scale → VR value.  Empty dict if no scale has
        enough data.

    Raises
    ------
    ValueError
        If ``closes`` has fewer than 3 elements or contains
        non-finite values.

    Notes
    -----
    Uses non-overlapping windows (not rolling/overlapping) for unbiased
    variance estimates.  This is more conservative than overlapping
    estimators but avoids autocorrelation in the estimates.
    """
    if scales is None:
        scales = M30_VR_SCALES

    c = np.asarray(closes, dtype=np.float64).ravel()

    if len(c) < 3:
        raise ValueError(f"closes must have ≥ 3 elements, got {len(c)}")
    if not np.all(np.isfinite(c)):
        raise ValueError("closes contains non-finite values (NaN/Inf)")

    # Guard against zero/negative prices (e.g. crypto edge cases)
    c = np.maximum(c, 1e-10)

    log_ret = np.diff(np.log(c))
    var1 = float(np.var(log_ret))

    if var1 < 1e-20:
        # Degenerate series (constant price) — VR is 1.0 everywhere
        return {s: 1.0 for s in scales if len(log_ret) // s >= min_windows}

    profile: Dict[int, float] = {}
    for s in scales:
        n_windows = len(log_ret) // s
        if n_windows < min_windows:
            continue
        # Non-overlapping T-bar returns
        n_ret = np.add.reduceat(log_ret, np.arange(0, n_windows * s, s))[:n_windows]
        vr = float(np.var(n_ret) / (s * var1))
        profile[s] = round(vr, 4)

    return profile


def vr_peak(
    profile: Dict[int, float],
) -> Tuple[int, float]:
    """
    Find the scale with the highest VR in a profile.

    Parameters
    ----------
    profile : Dict[int, float]
        VR profile from :func:`vr_profile`.

    Returns
    -------
    peak_scale : int
        Scale (bars) with highest VR.
    peak_vr : float
        VR value at peak scale.

    Raises
    ------
    ValueError
        If the profile is empty.
    """
    if not profile:
        raise ValueError("VR profile is empty — cannot determine peak")
    peak_scale = max(profile, key=lambda s: profile[s])
    return peak_scale, profile[peak_scale]


# ══════════════════════════════════════════════════════════════════════════════
# Regime Classification
# ══════════════════════════════════════════════════════════════════════════════


def classify_regime(peak_vr: float) -> str:
    """
    Classify the market regime based on peak VR value.

    Parameters
    ----------
    peak_vr : float
        The peak Variance Ratio value from VR profiling.

    Returns
    -------
    str
        One of: ``"TRENDING"``, ``"WEAK_TREND"``, ``"RANDOM_WALK"``,
        ``"MEAN_REVERTING"``.

    Notes
    -----
    Thresholds (from design spec §5.1):
      - VR ≥ 1.10 → TRENDING
      - VR ≥ 1.02 → WEAK_TREND
      - VR ≥ 0.98 → RANDOM_WALK
      - VR < 0.98 → MEAN_REVERTING
    """
    if peak_vr >= _TRENDING_THRESHOLD:
        return REGIME_TRENDING
    if peak_vr >= _WEAK_TREND_THRESHOLD:
        return REGIME_WEAK_TREND
    if peak_vr >= _RANDOM_WALK_THRESHOLD:
        return REGIME_RANDOM_WALK
    return REGIME_MEAN_REVERTING


def is_tradeable_regime(
    peak_vr: float,
    min_vr: float = DEFAULT_MIN_VR,
) -> bool:
    """
    Check whether an instrument's VR indicates a tradeable regime.

    Parameters
    ----------
    peak_vr : float
        Peak VR value.
    min_vr : float
        Minimum VR for portfolio candidacy.

    Returns
    -------
    bool
        True if peak_vr ≥ min_vr.
    """
    return peak_vr >= min_vr


# ══════════════════════════════════════════════════════════════════════════════
# Brick Sizing
# ══════════════════════════════════════════════════════════════════════════════


def brick_from_scale(
    closes: np.ndarray,
    scale_bars: int,
) -> float:
    """
    Compute the natural Renko brick size for a given VR peak scale.

    The DSP-suggested brick size is the **median absolute price
    displacement** over ``scale_bars`` — the natural amplitude of
    trend moves at the instrument's peak persistence horizon.

    Parameters
    ----------
    closes : np.ndarray
        Close price array.
    scale_bars : int
        Number of bars corresponding to the VR peak scale.
        Must be ≥ 1.

    Returns
    -------
    float
        Brick size in price units.  Always ≥ 0.01% of median price.

    Raises
    ------
    ValueError
        If ``scale_bars < 1``.
    """
    if scale_bars < 1:
        raise ValueError(f"scale_bars must be ≥ 1, got {scale_bars}")

    c = np.asarray(closes, dtype=np.float64).ravel()
    c = np.maximum(c, 1e-10)

    if len(c) <= scale_bars:
        # Not enough data for full displacement — use scaled single-bar
        brick = float(np.median(np.abs(np.diff(c)))) * scale_bars
    else:
        displacements = np.abs(c[scale_bars:] - c[:-scale_bars])
        brick = float(np.median(displacements))

    # Floor: 0.01% of median price
    min_brick = float(np.median(c)) * _MIN_BRICK_PRICE_FRACTION
    return max(brick, min_brick)


# ══════════════════════════════════════════════════════════════════════════════
# Friction Floor
# ══════════════════════════════════════════════════════════════════════════════


def compute_friction_floor(
    spread: SpreadProfile,
    dsp_brick: float,
    y_mult: float = DEFAULT_Y_MULT,
    z_mult: float = DEFAULT_Z_MULT,
    commission_rt: float = 0.0,
    contract_size: float = 100.0,
    friction_mult: float = 4.0,
) -> FrictionFloor:
    """
    Compute the adaptive friction floor for an instrument.

    The friction floor is the minimum brick size below which trading
    is physically unprofitable because friction (spread + commission)
    dominates the edge.

    Formula (§5.3, §29.2):
        friction_per_unit = (commission_rt / contract_size) + spread_p50_price
        floor = max(Y × spread_p50_price, Z × spread_p95_price,
                    friction_mult × friction_per_unit)

    For XAUUSD ECN example:
        - commission_rt = $7.00 (round-trip)
        - contract_size = 100 oz
        - spread_p50_price = $0.01-$0.10
        - friction_per_unit = $7/100 + $0.01 = $0.08
        - floor = max(5 × $0.01, 2 × $0.05, 4 × $0.08) = $0.32

    Parameters
    ----------
    spread : SpreadProfile
        Spread characteristics for the instrument.
    dsp_brick : float
        DSP-suggested brick size (from :func:`brick_from_scale`).
    y_mult : float
        Multiplier for median spread.  Default: 5.0.
    z_mult : float
        Multiplier for p95 spread.  Default: 2.0.
    commission_rt : float
        Round-trip commission in USD per lot.  For ECN brokers, typically $7.00.
        Default: 0.0 (commission-free).
    contract_size : float
        Units per lot (e.g., 100 oz for XAUUSD).  Default: 100.0.
    friction_mult : float
        Multiplier for friction_per_unit.  Default: 4.0 (25% friction ratio).

    Returns
    -------
    FrictionFloor
        Complete floor analysis including whether DSP brick passes.
    """
    # Spread-based floor components
    floor_median = y_mult * spread.spread_p50_price
    floor_p95 = z_mult * spread.spread_p95_price

    # Commission-based floor component
    # friction_per_unit is the cost per $1 of brick per lot
    # = (commission / contract_size) + spread_p50
    friction_per_unit = (commission_rt / contract_size) + spread.spread_p50_price
    floor_commission = friction_mult * friction_per_unit

    # Binding floor is the maximum of all components
    floor = max(floor_median, floor_p95, floor_commission)

    above = dsp_brick >= floor
    min_mult = floor / dsp_brick if dsp_brick > 0 else 999.0

    return FrictionFloor(
        symbol=spread.symbol,
        floor_from_median=floor_median,
        floor_from_p95=floor_p95,
        floor_from_commission=floor_commission,
        floor_price=floor,
        dsp_brick=dsp_brick,
        dsp_brick_above_floor=above,
        effective_min_mult=min_mult,
        y_mult=y_mult,
        z_mult=z_mult,
        friction_mult=friction_mult,
        commission_rt=commission_rt,
        contract_size=contract_size,
    )


def compute_full_friction_floor(
    spread: SpreadProfile,
    dsp_brick: float,
    commission_rt: float,
    contract_size: float,
    y_mult: float = DEFAULT_Y_MULT,
    z_mult: float = DEFAULT_Z_MULT,
    friction_mult: float = 4.0,
) -> FrictionFloor:
    """
    Compute the complete friction floor including commission.

    This is a convenience wrapper that enforces commission is provided.
    Use this function for ECN brokers where commission is significant.

    Parameters
    ----------
    spread : SpreadProfile
        Spread characteristics for the instrument.
    dsp_brick : float
        DSP-suggested brick size (from :func:`brick_from_scale`).
    commission_rt : float
        Round-trip commission in USD per lot.  Required.
        For typical ECN brokers: $7.00 (3.50/side).
    contract_size : float
        Units per lot (e.g., 100 oz for XAUUSD).  Required.
    y_mult : float
        Multiplier for median spread.  Default: 5.0.
    z_mult : float
        Multiplier for p95 spread.  Default: 2.0.
    friction_mult : float
        Multiplier for friction_per_unit.  Default: 4.0 (25% friction ratio).

    Returns
    -------
    FrictionFloor
        Complete floor analysis including commission component.

    Examples
    --------
    >>> # XAUUSD ECN broker
    >>> spread = SpreadProfile.from_price("XAUUSD", 0.01, 0.01, 0.05, 0.1, "live")
    >>> floor = compute_full_friction_floor(
    ...     spread=spread,
    ...     dsp_brick=1.0,  # $1.00 brick
    ...     commission_rt=7.0,  # $7 RT
    ...     contract_size=100.0,  # 100 oz
    ... )
    >>> floor.floor_from_commission  # 4 × ($7/100 + $0.01) = $0.32
    0.32
    >>> floor.floor_price  # max(5×$0.01, 2×$0.05, $0.32) = $0.32
    0.32
    >>> floor.dsp_brick_above_floor  # $1.00 >= $0.32
    True
    """
    return compute_friction_floor(
        spread=spread,
        dsp_brick=dsp_brick,
        y_mult=y_mult,
        z_mult=z_mult,
        commission_rt=commission_rt,
        contract_size=contract_size,
        friction_mult=friction_mult,
    )


def build_rolling_spread_profile(
    symbol: str,
    spread_values: np.ndarray,
    tick_size: float,
    window_bars: int = DEFAULT_ROLLING_WINDOW_BARS,
    min_obs: int = DEFAULT_ROLLING_MIN_OBS,
    source: str = "csv_m1_rolling",
) -> SpreadProfile:
    """Build a spread profile from causal rolling-window typical spread.

    This is explicitly **no-lookahead**:
      - for each bar ``t``, the window uses ``[t-window+1, ..., t]`` only
      - no future spreads are included in the estimate for that bar

    Parameters
    ----------
    symbol : str
        Instrument symbol.
    spread_values : np.ndarray
        Raw spread values in points, typically from M1 ``spread`` column.
    tick_size : float
        Price per point.
    window_bars : int
        Trailing window size in bars.
    min_obs : int
        Minimum count of positive spread observations in a window.
    source : str
        Source label attached to output profile.

    Returns
    -------
    SpreadProfile
        Percentiles of rolling-window typical spread (points + price).

    Raises
    ------
    ValueError
        If not enough valid observations are available.
    """
    if window_bars < 2:
        raise ValueError(f"window_bars must be >= 2, got {window_bars}")
    if min_obs < 1:
        raise ValueError(f"min_obs must be >= 1, got {min_obs}")

    sv = np.asarray(spread_values, dtype=np.float64).ravel()
    if len(sv) < min_obs:
        raise ValueError(f"Need >= {min_obs} spread bars, got {len(sv)}")

    positive = sv > 0
    if int(np.count_nonzero(positive)) < min_obs:
        raise ValueError("Not enough positive spread values for rolling profile")

    x = np.where(positive, sv, 0.0)
    csum = np.concatenate(([0.0], np.cumsum(x)))
    ccount = np.concatenate(([0], np.cumsum(positive.astype(np.int64))))

    # Trailing causal windows [i-window+1, i]
    idx = np.arange(len(sv)) + 1
    start = np.maximum(0, idx - window_bars)
    win_sum = csum[idx] - csum[start]
    win_cnt = ccount[idx] - ccount[start]

    means = np.divide(
        win_sum,
        np.maximum(win_cnt, 1),
        out=np.zeros_like(win_sum, dtype=np.float64),
        where=win_cnt > 0,
    )
    means = means[win_cnt >= min_obs]

    if len(means) < 10:
        raise ValueError(
            f"Need >= 10 rolling windows with >= {min_obs} valid obs, got {len(means)}"
        )

    return build_spread_profile(
        symbol=symbol,
        spread_values=means,
        tick_size=tick_size,
        source=source,
    )


def compute_rolling_friction_floor(
    symbol: str,
    spread_values: np.ndarray,
    tick_size: float,
    dsp_brick: float,
    x_mult: float = DEFAULT_Y_MULT,
    window_bars: int = DEFAULT_ROLLING_WINDOW_BARS,
    min_obs: int = DEFAULT_ROLLING_MIN_OBS,
    tail_mult: Optional[float] = DEFAULT_Z_MULT,
) -> FrictionFloor:
    """Compute friction floor from rolling-window spread statistics.

    Primary rule requested:
      ``brick_min >= x_mult * typical_rolling_spread``

    Where ``typical_rolling_spread`` is the p50 of trailing-window mean spread.
    Optional tail protection keeps the p95 component:
      ``brick_min = max(x_mult*p50_roll, tail_mult*p95_roll)``.
    """
    spread_roll = build_rolling_spread_profile(
        symbol=symbol,
        spread_values=spread_values,
        tick_size=tick_size,
        window_bars=window_bars,
        min_obs=min_obs,
        source="csv_m1_rolling",
    )
    floor_median = x_mult * spread_roll.spread_p50_price
    if tail_mult is not None and tail_mult > 0:
        floor_p95 = tail_mult * spread_roll.spread_p95_price
        floor = max(floor_median, floor_p95)
    else:
        floor_p95 = 0.0
        floor = floor_median

    above = dsp_brick >= floor
    min_mult = floor / dsp_brick if dsp_brick > 0 else 999.0
    return FrictionFloor(
        symbol=symbol,
        floor_from_median=floor_median,
        floor_from_p95=floor_p95,
        floor_price=floor,
        dsp_brick=dsp_brick,
        dsp_brick_above_floor=above,
        effective_min_mult=min_mult,
        y_mult=x_mult,
        z_mult=float(tail_mult or 0.0),
    )


def friction_ratio(
    spread_p75_price: float,
    brick_size: float,
) -> float:
    """
    Compute the friction ratio for a candidate brick size.

    friction_ratio = spread_p75_price / brick_size

    A high friction ratio means spread eats a large fraction of each
    brick, making profitable trading unlikely.

    Parameters
    ----------
    spread_p75_price : float
        75th percentile spread in price units.
    brick_size : float
        Candidate brick size in price units.

    Returns
    -------
    float
        Friction ratio.  Returns ``inf`` if brick_size ≤ 0.

    Notes
    -----
    Hard cap from design spec (§5.4): reject if > 0.25 (25%).
    RL can eventually learn a tighter per-regime threshold.
    """
    if brick_size <= 0:
        return float("inf")
    return spread_p75_price / brick_size


def passes_friction_gate(
    spread_p75_price: float,
    brick_size: float,
    max_ratio: float = DEFAULT_MAX_FRICTION_RATIO,
) -> bool:
    """
    Check whether a candidate brick passes the friction gate.

    Parameters
    ----------
    spread_p75_price : float
        75th percentile spread in price units.
    brick_size : float
        Candidate brick size in price units.
    max_ratio : float
        Maximum allowed friction ratio.  Default: 0.25.

    Returns
    -------
    bool
        True if friction_ratio ≤ max_ratio.
    """
    return friction_ratio(spread_p75_price, brick_size) <= max_ratio


# ══════════════════════════════════════════════════════════════════════════════
# Brick Sweep
# ══════════════════════════════════════════════════════════════════════════════

# Default multiplier grid for brick sweep around DSP brick
DEFAULT_SWEEP_MULTIPLIERS: List[float] = [0.5, 0.7, 1.0, 1.3, 1.5, 2.0, 2.5, 3.0]


@dataclass(frozen=True, slots=True)
class SweepCandidate:
    """One candidate brick size from a brick sweep.

    Attributes
    ----------
    multiplier : float
        Multiplier relative to DSP brick.
    brick_size : float
        Candidate brick size in price units.
    friction_ratio : float
        spread_p75_price / brick_size.
    passed_floor : bool
        Whether brick ≥ friction floor.
    passed_friction : bool
        Whether friction_ratio ≤ max_friction_ratio.
    """

    multiplier: float
    brick_size: float
    friction_ratio: float
    passed_floor: bool
    passed_friction: bool

    @property
    def passed_all(self) -> bool:
        """True if both floor and friction gates pass."""
        return self.passed_floor and self.passed_friction


def sweep_brick_sizes(
    dsp_brick: float,
    floor: FrictionFloor,
    spread: SpreadProfile,
    multipliers: Optional[List[float]] = None,
    max_friction: float = DEFAULT_MAX_FRICTION_RATIO,
) -> List[SweepCandidate]:
    """
    Generate candidate brick sizes and pre-filter on friction gates.

    This produces the candidate grid for empirical backtesting.
    Only candidates that pass both floor and friction gates should
    proceed to backtest evaluation.

    Parameters
    ----------
    dsp_brick : float
        DSP-suggested brick size.
    floor : FrictionFloor
        Friction floor for the instrument.
    spread : SpreadProfile
        Spread characteristics.
    multipliers : list of float or None
        Multiplier grid.  If None, uses :data:`DEFAULT_SWEEP_MULTIPLIERS`.
    max_friction : float
        Maximum friction ratio.

    Returns
    -------
    List[SweepCandidate]
        One candidate per multiplier, with gate results.
    """
    if multipliers is None:
        multipliers = DEFAULT_SWEEP_MULTIPLIERS

    candidates: List[SweepCandidate] = []
    for mult in multipliers:
        brick = dsp_brick * mult
        if brick <= 0:
            continue

        fr = friction_ratio(spread.spread_p75_price, brick)
        pf = brick >= floor.floor_price
        pfr = fr <= max_friction

        candidates.append(
            SweepCandidate(
                multiplier=mult,
                brick_size=brick,
                friction_ratio=fr,
                passed_floor=pf,
                passed_friction=pfr,
            )
        )

    return candidates


# ══════════════════════════════════════════════════════════════════════════════
# High-Level DSP Runner
# ══════════════════════════════════════════════════════════════════════════════


def run_dsp(
    closes: np.ndarray,
    symbol: str = "",
    scales: Optional[List[int]] = None,
    min_windows: int = 30,
    bars_per_hour: float = 2.0,
    bars_per_year: Optional[float] = None,
) -> DSPResult:
    """
    Run complete VR-based DSP analysis on a close price series.

    Computes the VR profile, identifies the peak persistent scale,
    derives the natural brick size, classifies the regime, and
    determines the TMA period.

    Parameters
    ----------
    closes : np.ndarray
        Close price array (e.g. M30 close prices).
    symbol : str
        Instrument symbol (for labelling).
    scales : list of int or None
        VR scale grid.  If None, uses :data:`M30_VR_SCALES`.
    min_windows : int
        Minimum non-overlapping windows per scale.
    bars_per_hour : float
        How many bars per hour in the source data.
        M30 = 2.0, H4 = 0.25, M1 = 60.0.
    bars_per_year : float or None
        Approximate bars per trading year.  If None, computed as
        ``bars_per_hour × 24 × 252``.

    Returns
    -------
    DSPResult
        Complete analysis result.

    Raises
    ------
    ValueError
        If ``closes`` is too short (< 3 elements) or contains
        non-finite values.
    """
    if scales is None:
        scales = M30_VR_SCALES

    c = np.asarray(closes, dtype=np.float64).ravel()

    if len(c) < 3:
        raise ValueError(f"closes must have ≥ 3 elements, got {len(c)}")

    if bars_per_year is None:
        bars_per_year = bars_per_hour * 24.0 * 252.0

    n_bars = len(c)
    years = n_bars / bars_per_year if bars_per_year > 0 else 0.0

    # ── VR profile ──────────────────────────────────────────────────────
    prof = vr_profile(c, scales=scales, min_windows=min_windows)

    if not prof:
        # Insufficient data for any scale — return fallback
        c_safe = np.maximum(c, 1e-10)
        mid = float(np.median(c_safe))
        brick = mid * 0.005  # fallback: 0.5% of price
        return DSPResult(
            symbol=symbol,
            n_bars=n_bars,
            years=years,
            vr_profile={},
            vr_peak_scale=1,
            vr_peak_scale_hours=1.0 / bars_per_hour,
            vr_peak_value=1.0,
            regime=REGIME_INSUFFICIENT_DATA,
            dsp_brick_size=brick,
            tma_period=10,
            bars_per_hour=bars_per_hour,
        )

    # ── Peak scale ──────────────────────────────────────────────────────
    peak_scale, peak_vr = vr_peak(prof)
    peak_hours = peak_scale / bars_per_hour

    # ── Brick size ──────────────────────────────────────────────────────
    brick = brick_from_scale(c, peak_scale)

    # ── TMA period ──────────────────────────────────────────────────────
    tma_period = max(5, min(peak_scale, 80))

    # ── Regime ──────────────────────────────────────────────────────────
    regime = classify_regime(peak_vr)

    return DSPResult(
        symbol=symbol,
        n_bars=n_bars,
        years=years,
        vr_profile=prof,
        vr_peak_scale=peak_scale,
        vr_peak_scale_hours=peak_hours,
        vr_peak_value=peak_vr,
        regime=regime,
        dsp_brick_size=brick,
        tma_period=tma_period,
        bars_per_hour=bars_per_hour,
    )


# ══════════════════════════════════════════════════════════════════════════════
# Spread Measurement Utilities
# ══════════════════════════════════════════════════════════════════════════════


def measure_spread_percentiles(
    spread_values: np.ndarray,
) -> Tuple[float, float, float]:
    """
    Compute spread percentiles (p50, p75, p95) from raw spread data.

    Parameters
    ----------
    spread_values : np.ndarray
        Array of positive spread values (in points).
        Zero/negative values are filtered out.

    Returns
    -------
    p50, p75, p95 : float
        Spread percentiles.

    Raises
    ------
    ValueError
        If fewer than 10 positive spread values remain after filtering.
    """
    sv = np.asarray(spread_values, dtype=np.float64).ravel()
    sv = sv[sv > 0]

    if len(sv) < 10:
        raise ValueError(
            f"Need ≥ 10 positive spread values for percentile estimation, got {len(sv)}"
        )

    p50 = float(np.percentile(sv, 50))
    p75 = float(np.percentile(sv, 75))
    p95 = float(np.percentile(sv, 95))
    return p50, p75, p95


def build_spread_profile(
    symbol: str,
    spread_values: np.ndarray,
    tick_size: float,
    source: str = "csv_m30",
) -> SpreadProfile:
    """
    Build a SpreadProfile from raw spread data.

    Convenience function that measures percentiles and constructs
    the dataclass in one step.

    Parameters
    ----------
    symbol : str
        Instrument symbol.
    spread_values : np.ndarray
        Raw spread values in points (from CSV ``spread`` column).
    tick_size : float
        Price per point.
    source : str
        Source label.

    Returns
    -------
    SpreadProfile
    """
    sv = np.asarray(spread_values, dtype=np.float64).ravel()
    sv = sv[sv > 0]
    p50, p75, p95 = measure_spread_percentiles(sv)
    return SpreadProfile.from_points(
        symbol=symbol,
        tick_size=tick_size,
        p50_pts=p50,
        p75_pts=p75,
        p95_pts=p95,
        source=source,
        n_bars=len(sv),
    )


def scaled_filter_params(dsp_result: DSPResult, bricks_per_day: float) -> FilterParams:
    """
    Derive adaptive FilterParams from DSP analysis result.

    Scales filter window sizes based on the VR peak scale to match
    the instrument's natural trend horizon. Thresholds are fixed
    empirical defaults — validated through backtesting.

    Parameters
    ----------
    dsp_result : DSPResult
        Complete DSP analysis result.
    bricks_per_day : float
        Average bricks per day (from brick_summary).

    Returns
    -------
    FilterParams
        Adaptive filter configuration.
    """
    # Local import avoids module-cycle issues while ensuring runtime availability.
    from kinetra.renko.backtest import FilterParams

    # Scale window sizes based on VR peak scale (trend horizon)
    fliprate_window = max(10, min(100, dsp_result.vr_peak_scale))
    markov_window = fliprate_window

    # Fixed thresholds — empirically validated
    return FilterParams(
        fliprate_window=fliprate_window,
        fliprate_threshold=0.35,
        markov_window=markov_window,
        markov_threshold=0.55,
    )
