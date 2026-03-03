"""
Spread-Gated Renko Backtest Module
==================================

Implements adaptive Renko backtesting with spread-based gating.

This module provides an alternative to the fixed-brick qualification framework.
It uses:
- Adaptive brick sizes based on volatility (Garman-Klass + roofing filter)
- Spread gating (percentile/ratio/both modes) to filter high-spread periods
- Grid search optimization over gate parameters (W, Q, θ)
- Hysteresis and rate limiting to prevent excessive brick size variation

Key insight: Many instruments rejected by the qualification framework due to
strict fixed-brick requirements actually have massive profit potential when
spread gating and adaptive brick sizing are applied.

Usage::

    from kinetra.renko.spread_gated_backtest import (
        backtest_strict,
        run_grid_search,
        AdaptiveBacktestResult,
        qualify_instrument_adaptive,
    )

    result = qualify_instrument_adaptive(
        symbol="XAUUSD",
        m1_df=df,
        spread_pts=1.5,
        tick_size=0.01,
        output_dir=Path("results/adaptive"),
    )

    if result.qualified:
        print(f"Qualified: score={result.score:.2f}, omega={result.omega:.2f}")

See Also:
    - ``kinetra/renko/backtest.py`` — Fixed-brick backtest with FilterParams
    - ``kinetra/renko/dsp.py`` — DSP analysis for brick sizing
    - ``docs/MANUAL.md §29`` — Renko architecture
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from numpy.typing import NDArray


logger = logging.getLogger(__name__)


# ==============================================================================
# CONFIGURATION CONSTANTS
# ==============================================================================

# Contract defaults (XAUUSD-based, override per instrument)
DEFAULT_TICK = 0.01
DEFAULT_CONTRACT_MULT = 100.0
DEFAULT_SLIP_PTS_ONEWAY = 1.0
DEFAULT_STOP_BRICKS = 1.0

# Position sizing defaults
DEFAULT_LOT_STEP = 0.01
DEFAULT_MIN_LOT = 0.01
DEFAULT_MAX_LOT = 50.0
DEFAULT_START_BAL = 1000.0
DEFAULT_LOTS_PER_1000 = 0.01

# Volatility window for adaptive brick sizing
DEFAULT_SIGMA_WINDOW = 240  # 4 hours on M1

# Roofing filter defaults (John Ehlers)
DEFAULT_HP_PERIOD = 120
DEFAULT_SS_PERIOD = 30

# Brick stability parameters
DEFAULT_BAND = (0.9, 1.1)  # Clamp brick size to [0.9x, 1.1x] of median
DEFAULT_HYSTERESIS = 0.07  # Ignore changes < 7%
DEFAULT_RATE_LIMIT = 0.05  # Max 5% change per step

# Friction floor multiplier (brick >= k4 * spread)
DEFAULT_FLOOR_K4 = 4.0

# Grid search parameters
DEFAULT_Q_VALS = [0.70, 0.80, 0.90, 0.95, 0.98]
DEFAULT_W_VALS = [60, 240, 720]  # M1 bars
DEFAULT_THETA_VALS = [0.20, 0.30, 0.40, 0.50, 0.70, 1.00]

# Quality gate thresholds
MIN_OMEGA = 1.2
MIN_Z_FACTOR = 1.8
MIN_TRADES = 20
MAX_DRAWDOWN_PCT = 15.0


# ==============================================================================
# ENUMS
# ==============================================================================


class GateType(str, Enum):
    """Spread gate type."""

    NONE = "none"
    PERCENTILE = "percentile"
    RATIO = "ratio"
    BOTH = "both"


class AdaptiveBacktestStatus(str, Enum):
    """Status of adaptive backtest qualification."""

    QUALIFIED = "qualified"
    INSUFFICIENT_DATA = "insufficient_data"
    INSUFFICIENT_TRADES = "insufficient_trades"
    POOR_PERFORMANCE = "poor_performance"
    EXCESSIVE_DRAWDOWN = "excessive_drawdown"
    FAILED_GATES = "failed_gates"
    ERROR = "error"


# ==============================================================================
# DATA CLASSES
# ==============================================================================


@dataclass(frozen=True, slots=True)
class SpreadGateConfig:
    """Configuration for spread gating."""

    gate_type: GateType = GateType.NONE
    # Percentile gate params
    window: int = 240
    quantile: float = 0.90
    # Ratio gate params
    max_spread_brick_ratio: float = 0.50  # spread <= theta * brick
    # Floor params
    floor_k4: float = DEFAULT_FLOOR_K4  # brick >= k4 * spread

    def __post_init__(self):
        if not 0 < self.quantile <= 1.0:
            raise ValueError(f"quantile must be in (0, 1], got {self.quantile}")
        if self.window <= 0:
            raise ValueError(f"window must be > 0, got {self.window}")
        if self.max_spread_brick_ratio <= 0:
            raise ValueError(
                f"max_spread_brick_ratio must be > 0, got {self.max_spread_brick_ratio}"
            )
        if self.floor_k4 <= 0:
            raise ValueError(f"floor_k4 must be > 0, got {self.floor_k4}")


@dataclass(frozen=True, slots=True)
class BrickStabilityConfig:
    """Configuration for brick size stability."""

    band_low: float = DEFAULT_BAND[0]
    band_high: float = DEFAULT_BAND[1]
    hysteresis: float = DEFAULT_HYSTERESIS
    rate_limit: float = DEFAULT_RATE_LIMIT

    def __post_init__(self):
        if not 0 < self.band_low < self.band_high:
            raise ValueError(f"band_low ({self.band_low}) must be < band_high ({self.band_high})")
        if not 0 < self.hysteresis < 1.0:
            raise ValueError(f"hysteresis must be in (0, 1), got {self.hysteresis}")
        if not 0 < self.rate_limit < 1.0:
            raise ValueError(f"rate_limit must be in (0, 1), got {self.rate_limit}")


@dataclass(frozen=True, slots=True)
class AdaptiveBacktestConfig:
    """Complete configuration for adaptive Renko backtest."""

    # Contract params
    tick: float = DEFAULT_TICK
    contract_mult: float = DEFAULT_CONTRACT_MULT
    slip_pts_oneway: float = DEFAULT_SLIP_PTS_ONEWAY
    stop_bricks: float = DEFAULT_STOP_BRICKS

    # Position sizing
    lot_step: float = DEFAULT_LOT_STEP
    min_lot: float = DEFAULT_MIN_LOT
    max_lot: float = DEFAULT_MAX_LOT
    start_bal: float = DEFAULT_START_BAL
    lots_per_1000: float = DEFAULT_LOTS_PER_1000

    # Volatility/brick sizing
    sigma_window: int = DEFAULT_SIGMA_WINDOW
    hp_period: int = DEFAULT_HP_PERIOD
    ss_period: int = DEFAULT_SS_PERIOD

    # Brick stability
    brick_stability: BrickStabilityConfig = field(default_factory=BrickStabilityConfig)

    # Spread gating
    spread_gate: SpreadGateConfig = field(default_factory=SpreadGateConfig)

    # Data filtering
    session_break_minutes: float = 30.0
    min_bars: int = 1000

    # Quality gates
    min_omega: float = MIN_OMEGA
    min_z_factor: float = MIN_Z_FACTOR
    min_trades: float = MIN_TRADES
    max_drawdown_pct: float = MAX_DRAWDOWN_PCT


@dataclass(slots=True)
class AdaptiveBacktestResult:
    """Result of a single adaptive backtest configuration."""

    # Config that produced this result
    gate_type: str
    window: Optional[int] = None
    quantile: Optional[float] = None
    max_spread_brick_ratio: Optional[float] = None
    floor_k4: float = DEFAULT_FLOOR_K4
    brick_type: str = "base"  # "base" or "floor"

    # Performance metrics
    end_balance: float = 0.0
    net_pnl: float = 0.0
    trades: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_pct: float = 0.0

    # Risk metrics
    omega: float = 0.0
    sortino: float = 0.0
    cvar95: float = 0.0
    volatility: float = 0.0
    z_factor: float = 0.0

    # Execution metrics
    gated_entries: int = 0
    gate_bars_open: int = 0  # bricks where spread gate was OPEN
    total_cost_usd: float = 0.0
    pnl_per_cost: float = 0.0

    # Brick statistics
    brick_count: int = 0
    avg_brick_size: float = 0.0
    std_brick_size: float = 0.0

    # Score (for ranking)
    score: float = 0.0  # balance / (1 + |dd|)

    # Timestamp
    run_time: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    # Status flags
    passed_gates: bool = False
    fail_reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


@dataclass(slots=True)
class AdaptiveQualificationResult:
    """Result of full adaptive qualification pipeline."""

    symbol: str
    instrument_id: str
    status: AdaptiveBacktestStatus

    # Best configuration
    best_config: AdaptiveBacktestResult
    # Pareto frontier configurations
    pareto_configs: List[AdaptiveBacktestResult] = field(default_factory=list)
    # Top N configurations
    top_configs: List[AdaptiveBacktestResult] = field(default_factory=list)

    # All configurations tested
    all_configs: List[AdaptiveBacktestResult] = field(default_factory=list)

    # Data summary
    total_bars: int = 0
    total_bricks: int = 0
    data_start: str = ""
    data_end: str = ""

    # Gate coverage metrics (Sprint 6A)
    gate_bar_fraction: float = 0.0   # fraction of bricks where spread gate was open
    gate_trade_fraction: float = 0.0  # fraction of entry attempts allowed by gate

    # Timestamp
    qualified_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    @property
    def qualified(self) -> bool:
        """True if instrument passed qualification."""
        return self.status == AdaptiveBacktestStatus.QUALIFIED

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        payload = asdict(self)
        payload["best_config"] = self.best_config.to_dict()
        payload["pareto_configs"] = [c.to_dict() for c in self.pareto_configs]
        payload["top_configs"] = [c.to_dict() for c in self.top_configs]
        payload["all_configs"] = [c.to_dict() for c in self.all_configs]
        return payload


# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================


def lots_from_capital(cap: float, config: AdaptiveBacktestConfig) -> float:
    """Calculate lot size from capital using fixed leverage."""
    raw = config.lots_per_1000 * (cap / 1000.0)
    lots = math.floor(raw / config.lot_step) * config.lot_step
    return max(config.min_lot, min(config.max_lot, lots))


def garman_klass_sigma(
    o: NDArray[np.float64],
    h: NDArray[np.float64],
    l: NDArray[np.float64],
    c: NDArray[np.float64],
    win: int,
) -> NDArray[np.float64]:
    """
    Garman-Klass volatility estimator.

    Uses high-low and close-open range to estimate realized volatility.
    More efficient than close-only estimators.

    Args:
        o, h, l, c: Open, high, low, close arrays
        win: Rolling window size

    Returns:
        Array of volatility estimates
    """
    log_hl = np.log(h / l)
    log_co = np.log(c / o)
    rs = 0.5 * log_hl**2 - (2 * np.log(2) - 1) * log_co**2

    # Rolling mean with minimum periods
    result = np.full(len(rs), np.nan, dtype=np.float64)
    for i in range(win - 1, len(rs)):
        result[i] = np.mean(rs[i - win + 1 : i + 1])

    return np.sqrt(np.maximum(result, 0.0))


def supersmoother(x: NDArray[np.float64], period: int) -> NDArray[np.float64]:
    """
    John Ehlers' SuperSmoother filter.

    A low-pass filter with minimal phase lag and excellent
    time-domain properties.

    Args:
        x: Input signal
        period: Filter period

    Returns:
        Filtered signal
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.zeros_like(x)

    a1 = math.exp(-1.414 * math.pi / period)
    b1 = 2 * a1 * math.cos(1.414 * math.pi / period)
    c2 = b1
    c3 = -a1 * a1
    c1 = 1 - c2 - c3

    # Initialize with first values
    y[:2] = x[:2]

    for i in range(2, len(x)):
        y[i] = c1 * (x[i] + x[i - 1]) / 2 + c2 * y[i - 1] + c3 * y[i - 2]

    return y


def roofing_filter(
    x: NDArray[np.float64],
    hp_period: int = DEFAULT_HP_PERIOD,
    ss_period: int = DEFAULT_SS_PERIOD,
) -> NDArray[np.float64]:
    """
    John Ehlers' Roofing Filter.

    High-pass filter followed by SuperSmoother to isolate cyclic
    components from price data.

    Args:
        x: Input signal (volatility)
        hp_period: High-pass period
        ss_period: SuperSmoother period

    Returns:
        Filtered signal (absolute value)
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.zeros_like(x)

    a1 = math.exp(-1.414 * math.pi / hp_period)
    b1 = 2 * a1 * math.cos(1.414 * math.pi / hp_period)
    c2 = b1
    c3 = -a1 * a1
    c1 = (1 + c2 - c3) / 4

    y[:2] = 0.0

    for i in range(2, len(x)):
        y[i] = c1 * (x[i] - 2 * x[i - 1] + x[i - 2]) + c2 * y[i - 1] + c3 * y[i - 2]

    z = supersmoother(y, ss_period)
    return np.abs(z)


def apply_stability(
    brick_raw: NDArray[np.float64],
    clamp_lo: float,
    clamp_hi: float,
    hyster: float,
    rate: float,
) -> NDArray[np.float64]:
    """
    Apply clamping, hysteresis, and rate limiting to brick sizes.

    Prevents excessive variation in adaptive brick sizes.

    Args:
        brick_raw: Raw brick size series
        clamp_lo: Minimum allowed brick size
        clamp_hi: Maximum allowed brick size
        hyster: Hysteresis threshold (fractional)
        rate: Max rate of change per step (fractional)

    Returns:
        Stabilized brick size series
    """
    s = np.copy(brick_raw)
    s[~np.isfinite(s)] = np.nanmedian(s[np.isfinite(s)])
    s = np.nan_to_num(s, nan=np.nanmedian(s))

    out = np.zeros_like(s)
    out[0] = np.clip(s[0], clamp_lo, clamp_hi)

    for i in range(1, len(s)):
        target = np.clip(s[i], clamp_lo, clamp_hi)
        prev = out[i - 1]

        # Hysteresis: ignore small changes
        if prev > 1e-3 and abs(target - prev) / prev < hyster:
            target = prev

        # Rate limiting: limit change per step
        if prev > 0:
            ratio = target / prev
            ratio = np.clip(ratio, 1 - rate, 1 + rate)
            out[i] = prev * ratio
        else:
            out[i] = target

    return out


def build_variable_renko(
    times_ns: NDArray[np.int64],
    price: NDArray[np.float64],
    brick_series: NDArray[np.float64],
) -> Tuple[
    NDArray[np.int64], NDArray[np.int8], NDArray[np.float64], NDArray[np.float64], NDArray[np.int32]
]:
    """
    Build Renko bricks with time-varying brick sizes.

    Args:
        times_ns: Unix timestamps in nanoseconds
        price: Price array (M1 close)
        brick_series: Brick size array (same length as price)

    Returns:
        tuple: (brick_times, directions, close_prices, brick_sizes, bar_indices)
    """
    last = float(price[0])
    rt, rd, rc, rb, bi = [], [], [], [], []

    for i in range(1, len(price)):
        b = float(brick_series[i])
        if b <= 0 or not np.isfinite(b):
            continue

        p = float(price[i])
        ts = int(times_ns[i])

        # Calculate how many bricks up or down
        up = int((p - last) // b) if p > last else 0
        dn = int((last - p) // b) if p < last else 0

        if up:
            for _ in range(up):
                last += b
                rt.append(ts)
                rd.append(1)
                rc.append(last)
                rb.append(b)
                bi.append(i)
        elif dn:
            for _ in range(dn):
                last -= b
                rt.append(ts)
                rd.append(-1)
                rc.append(last)
                rb.append(b)
                bi.append(i)

    if not rt:
        return (
            np.array([], dtype=np.int64),
            np.array([], dtype=np.int8),
            np.array([], dtype=np.float64),
            np.array([], dtype=np.float64),
            np.array([], dtype=np.int32),
        )

    return (
        np.array(rt, dtype=np.int64),
        np.array(rd, dtype=np.int8),
        np.array(rc, dtype=np.float64),
        np.array(rb, dtype=np.float64),
        np.array(bi, dtype=np.int32),
    )


def rolling_quantile(
    arr: NDArray[np.float64],
    W: int,
    q: float,
    min_periods: int = 10,
) -> NDArray[np.float64]:
    """
    Calculate rolling quantile with minimum period requirement.

    Args:
        arr: Input array
        W: Window size
        q: Quantile (0-1)
        min_periods: Minimum periods for valid output

    Returns:
        Rolling quantile array
    """
    s = pd.Series(arr)
    result = s.rolling(W, min_periods=min_periods).quantile(q)
    return result.bfill().to_numpy(dtype=np.float64)


def risk_metrics(eq: NDArray[np.float64]) -> Dict[str, float]:
    """
    Calculate risk metrics from equity curve.

    Args:
        eq: Equity array

    Returns:
        dict: sortino, cvar95 (95% expected shortfall), volatility, z_factor
    """
    if len(eq) < 2:
        return dict(sortino=np.nan, cvar95=np.nan, vol=np.nan, z_factor=np.nan)

    rets = np.diff(eq) / np.maximum(eq[:-1], 1e-9)
    neg = rets[rets < 0]
    dd = neg.std(ddof=0) if len(neg) > 0 else 0.0
    mean = rets.mean()

    # Sortino: mean / downside deviation
    sortino = mean / dd if dd > 1e-12 else (np.inf if mean > 0 else np.nan)

    # 95% CVaR (Expected Shortfall)
    q = np.quantile(rets, 0.05)
    tail = rets[rets <= q]
    cvar = tail.mean() if len(tail) > 0 else np.nan

    # Volatility
    vol = rets.std(ddof=0)

    # Z-factor: mean / (std / sqrt(n))
    z_factor = mean / (vol / np.sqrt(len(rets))) if vol > 1e-12 else np.nan

    return dict(sortino=sortino, cvar95=cvar, vol=vol, z_factor=z_factor)


def calculate_omega(
    equity: NDArray[np.float64],
    risk_free_rate: float = 0.0,
) -> float:
    """
    Calculate Omega ratio from an equity curve.

    Converts equity to period returns then delegates to the canonical
    ``kinetra.backtesting.metrics.omega_ratio`` (DRY-08).

    Args:
        equity: Equity curve array
        risk_free_rate: Risk-free rate per period (subtracted from each return)

    Returns:
        Omega ratio. Returns np.nan when fewer than 2 equity points are
        provided. Returns 10.0 (finite sentinel) when there are no losing
        periods — avoids inf polluting downstream calculations.
    """
    if len(equity) < 2:
        return np.nan

    from kinetra.backtesting.metrics import omega_ratio as _canonical_omega

    rets = np.diff(equity) / np.maximum(equity[:-1], 1e-9) - risk_free_rate
    return _canonical_omega(rets)


# ==============================================================================
# CORE BACKTEST ENGINE
# ==============================================================================


def backtest_strict(
    times_ns: NDArray[np.int64],
    price: NDArray[np.float64],
    spread_pts: NDArray[np.float64],
    brick_series: NDArray[np.float64],
    config: AdaptiveBacktestConfig,
) -> AdaptiveBacktestResult:
    """
    Renko backtest with strict spread gating.

    This is the core engine for adaptive Renko backtesting. Key features:
    - Variable brick sizes per bar
    - Spread gating (percentile/ratio/both modes)
    - Proper position tracking with flip handling
    - Accurate drawdown measurement

    Args:
        times_ns: Unix timestamps in nanoseconds
        price: Close price array (M1)
        spread_pts: Spread in points array (same length as price)
        brick_series: Brick size array (same length as price)
        config: Backtest configuration

    Returns:
        AdaptiveBacktestResult with all metrics
    """
    # Build Renko bricks
    rt, rd, rc, rb, bi = build_variable_renko(times_ns, price, brick_series)

    if len(rt) == 0:
        result = AdaptiveBacktestResult(
            gate_type=config.spread_gate.gate_type.value,
            floor_k4=config.spread_gate.floor_k4,
        )
        result.fail_reason = "No bricks generated"
        return result

    # Pre-compute friction costs for each brick
    idx = np.searchsorted(times_ns, rt, side="right") - 1
    idx[idx < 0] = 0
    spread_r = spread_pts[idx]
    friction_cost_pts = 2.0 * ((spread_r + config.slip_pts_oneway) * config.tick)
    friction_cost_usd = friction_cost_pts * config.contract_mult

    # Spread threshold array (for percentile gate)
    thr = None
    if config.spread_gate.gate_type in (GateType.PERCENTILE, GateType.BOTH):
        thr = rolling_quantile(
            spread_pts,
            W=config.spread_gate.window,
            q=config.spread_gate.quantile,
            min_periods=max(10, config.spread_gate.window // 3),
        )

    # State variables
    bal = config.start_bal
    pos = 0  # +1 long, -1 short, 0 flat
    entry = 0.0
    entry_brick = 0.0
    lots = 0.0

    # Metrics
    peak = config.start_bal
    mdd = 0.0
    trades = 0
    wins = 0
    gp = 0.0
    gl = 0.0
    gated = 0
    total_cost = 0.0

    equity_curve = []
    trade_pnls = []

    def gate_ok(i: int) -> bool:
        """Check if spread gate allows entry."""
        gt = config.spread_gate.gate_type
        if gt == GateType.NONE:
            return True

        bar_i = int(bi[i])
        sp = float(spread_pts[bar_i])
        b = float(rb[i])

        # Ratio gate: spread <= theta * brick
        ok_ratio = (sp * config.tick) / max(b, 1e-9) <= config.spread_gate.max_spread_brick_ratio

        if gt == GateType.PERCENTILE:
            return sp <= float(thr[bar_i])
        if gt == GateType.RATIO:
            return ok_ratio
        if gt == GateType.BOTH:
            return (sp <= float(thr[bar_i])) and ok_ratio

        raise ValueError(f"Unknown gate type: {gt}")

    # Main backtest loop
    for i in range(len(rc)):
        px = float(rc[i])
        d = int(rd[i])
        Fi_usd = float(friction_cost_usd[i])

        # Update equity and drawdown
        if pos != 0:
            equity = bal + (px - entry) * pos * config.contract_mult * lots
        else:
            equity = bal

        peak = max(peak, equity)
        mdd = min(mdd, equity - peak)

        # Position management
        if pos != 0:
            # Check stop loss
            stop_price = entry - pos * config.stop_bricks * entry_brick
            stop_hit = (pos == 1 and px <= stop_price) or (pos == -1 and px >= stop_price)

            # Check flip (direction change)
            flip = d != pos

            if stop_hit or flip:
                # Close position
                pnl_price = (px - entry) * pos - friction_cost_pts[i]
                pnl_usd = pnl_price * config.contract_mult * lots
                bal += pnl_usd
                trades += 1
                total_cost += Fi_usd

                if pnl_usd > 0:
                    wins += 1
                    gp += pnl_usd
                else:
                    gl += pnl_usd

                trade_pnls.append(pnl_usd)

                pos = 0
                lots = 0.0

                # FLIP LOGIC: Enter in opposite direction on same brick
                if flip:
                    if gate_ok(i):
                        pos = d
                        entry = px
                        entry_brick = float(rb[i])
                        lots = lots_from_capital(bal, config)

                        # Update equity and DD after new entry
                        equity = bal + (px - entry) * pos * config.contract_mult * lots
                        peak = max(peak, equity)
                        mdd = min(mdd, equity - peak)
                    else:
                        gated += 1

        # NEW ENTRY LOGIC: Only if not already in position
        elif pos == 0 and i > 0 and rd[i] != rd[i - 1]:
            if gate_ok(i):
                pos = d
                entry = px
                entry_brick = float(rb[i])
                lots = lots_from_capital(bal, config)

                # Update equity and DD after new entry
                equity = bal + (px - entry) * pos * config.contract_mult * lots
                peak = max(peak, equity)
                mdd = min(mdd, equity - peak)
            else:
                gated += 1

        equity_curve.append(equity)

    # Count bricks where the spread gate was open (all brick indices)
    gate_bars_open = sum(1 for i in range(len(rc)) if gate_ok(i))

    # Calculate final metrics
    wr = wins / trades if trades > 0 else 0.0
    pf = (gp / (-gl)) if gl < 0 else float("inf")
    pnl_per_cost = (bal - config.start_bal) / total_cost if total_cost > 0 else np.nan

    # Risk metrics
    eq_array = np.array(equity_curve, dtype=np.float64)
    rm = risk_metrics(eq_array)
    omega = calculate_omega(eq_array)

    max_dd_abs = abs(mdd)
    max_dd_pct = (max_dd_abs / peak) * 100 if peak > 0 else 0.0

    # Score: balance / (1 + |dd|)
    score = bal / (1.0 + max_dd_abs)

    # Brick statistics
    avg_brick = float(np.mean(rb)) if len(rb) > 0 else 0.0
    std_brick = float(np.std(rb)) if len(rb) > 0 else 0.0

    # Determine brick type
    brick_type = "base"
    if config.spread_gate.floor_k4 > 0:
        brick_type = f"floor_k{config.spread_gate.floor_k4}"

    # Build result
    result = AdaptiveBacktestResult(
        gate_type=config.spread_gate.gate_type.value,
        window=config.spread_gate.window
        if config.spread_gate.gate_type in (GateType.PERCENTILE, GateType.BOTH)
        else None,
        quantile=config.spread_gate.quantile
        if config.spread_gate.gate_type in (GateType.PERCENTILE, GateType.BOTH)
        else None,
        max_spread_brick_ratio=config.spread_gate.max_spread_brick_ratio
        if config.spread_gate.gate_type in (GateType.RATIO, GateType.BOTH)
        else None,
        floor_k4=config.spread_gate.floor_k4,
        brick_type=brick_type,
        end_balance=bal,
        net_pnl=bal - config.start_bal,
        trades=trades,
        win_rate=wr,
        profit_factor=pf,
        max_drawdown=mdd,
        max_drawdown_pct=max_dd_pct,
        omega=omega,
        sortino=rm["sortino"],
        cvar95=rm["cvar95"],
        volatility=rm["vol"],
        z_factor=rm["z_factor"] if "z_factor" in rm else 0.0,
        gated_entries=gated,
        gate_bars_open=gate_bars_open,
        total_cost_usd=total_cost,
        pnl_per_cost=pnl_per_cost,
        brick_count=len(rb),
        avg_brick_size=avg_brick,
        std_brick_size=std_brick,
        score=score,
    )

    # Check quality gates
    passed_gates = True
    fail_reason = ""

    if trades < config.min_trades:
        passed_gates = False
        fail_reason = f"Insufficient trades: {trades} < {config.min_trades}"

    if omega < config.min_omega:
        passed_gates = False
        if fail_reason:
            fail_reason += "; "
        fail_reason += f"Omega too low: {omega:.2f} < {config.min_omega}"

    rm_z = rm.get("z_factor", 0.0)
    if rm_z < config.min_z_factor:
        passed_gates = False
        if fail_reason:
            fail_reason += "; "
        fail_reason += f"Z-factor too low: {rm_z:.2f} < {config.min_z_factor}"

    if max_dd_pct > config.max_drawdown_pct:
        passed_gates = False
        if fail_reason:
            fail_reason += "; "
        fail_reason += f"Drawdown too high: {max_dd_pct:.1f}% > {config.max_drawdown_pct}%"

    result.passed_gates = passed_gates
    result.fail_reason = fail_reason

    return result


# ==============================================================================
# GRID SEARCH
# ==============================================================================


def run_grid_search(
    times_ns: NDArray[np.int64],
    price: NDArray[np.float64],
    spread_pts: NDArray[np.float64],
    brick_base: NDArray[np.float64],
    brick_floor: NDArray[np.float64],
    config: AdaptiveBacktestConfig,
    q_vals: Optional[List[float]] = None,
    w_vals: Optional[List[int]] = None,
    theta_vals: Optional[List[float]] = None,
    verbose: bool = False,
) -> List[AdaptiveBacktestResult]:
    """
    Run grid search over spread gate parameters.

    Tests combinations of:
    - No gate (baseline)
    - Percentile gate (W, Q)
    - Ratio gate (θ)
    - Both gates (W, Q, θ)

    Args:
        times_ns: Unix timestamps
        price: Close prices
        spread_pts: Spread points
        brick_base: Base brick sizes
        brick_floor: Floor brick sizes (with k4 multiplier)
        config: Base configuration
        q_vals: Quantile values to test
        w_vals: Window sizes to test
        theta_vals: Ratio thresholds to test
        verbose: Log progress

    Returns:
        List of AdaptiveBacktestResult
    """
    if q_vals is None:
        q_vals = DEFAULT_Q_VALS
    if w_vals is None:
        w_vals = DEFAULT_W_VALS
    if theta_vals is None:
        theta_vals = DEFAULT_THETA_VALS

    results: List[AdaptiveBacktestResult] = []

    # Build bricks for both types
    rt_base, rd_base, rc_base, rb_base, bi_base = build_variable_renko(times_ns, price, brick_base)
    rt_floor, rd_floor, rc_floor, rb_floor, bi_floor = build_variable_renko(
        times_ns, price, brick_floor
    )

    # Skip if no bricks
    if len(rt_base) == 0 and len(rt_floor) == 0:
        return results

    # Helper to run backtest with specific brick type
    def run_brick_type(rt, rd, rc, rb, bi, brick_label: str) -> List[AdaptiveBacktestResult]:
        local_results: List[AdaptiveBacktestResult] = []

        # Re-index arrays to original M1 bars
        idx = np.searchsorted(times_ns, rt, side="right") - 1
        idx[idx < 0] = 0

        # No gate baseline
        cfg_none = AdaptiveBacktestConfig(
            **{k: v for k, v in asdict(config).items() if k != "spread_gate"},
            spread_gate=SpreadGateConfig(gate_type=GateType.NONE),
        )
        res = backtest_strict(
            times_ns,
            price,
            spread_pts,
            brick_base if brick_label == "base" else brick_floor,
            cfg_none,
        )
        res.brick_type = brick_label
        local_results.append(res)

        # Percentile gate
        for W in w_vals:
            for Q in q_vals:
                cfg_pct = AdaptiveBacktestConfig(
                    **{k: v for k, v in asdict(config).items() if k != "spread_gate"},
                    spread_gate=SpreadGateConfig(
                        gate_type=GateType.PERCENTILE,
                        window=W,
                        quantile=Q,
                        floor_k4=config.spread_gate.floor_k4,
                    ),
                )
                res = backtest_strict(
                    times_ns,
                    price,
                    spread_pts,
                    brick_base if brick_label == "base" else brick_floor,
                    cfg_pct,
                )
                res.brick_type = brick_label
                local_results.append(res)

        # Ratio gate
        for th in theta_vals:
            cfg_ratio = AdaptiveBacktestConfig(
                **{k: v for k, v in asdict(config).items() if k != "spread_gate"},
                spread_gate=SpreadGateConfig(
                    gate_type=GateType.RATIO,
                    max_spread_brick_ratio=th,
                    floor_k4=config.spread_gate.floor_k4,
                ),
            )
            res = backtest_strict(
                times_ns,
                price,
                spread_pts,
                brick_base if brick_label == "base" else brick_floor,
                cfg_ratio,
            )
            res.brick_type = brick_label
            local_results.append(res)

        # Both gates
        for W in w_vals:
            for Q in q_vals:
                for th in theta_vals:
                    cfg_both = AdaptiveBacktestConfig(
                        **{k: v for k, v in asdict(config).items() if k != "spread_gate"},
                        spread_gate=SpreadGateConfig(
                            gate_type=GateType.BOTH,
                            window=W,
                            quantile=Q,
                            max_spread_brick_ratio=th,
                            floor_k4=config.spread_gate.floor_k4,
                        ),
                    )
                    res = backtest_strict(
                        times_ns,
                        price,
                        spread_pts,
                        brick_base if brick_label == "base" else brick_floor,
                        cfg_both,
                    )
                    res.brick_type = brick_label
                    local_results.append(res)

        return local_results

    # Run for base brick type
    if len(rt_base) > 0:
        if verbose:
            logger.info("Running grid search for base brick type...")
        results.extend(run_brick_type(rt_base, rd_base, rc_base, rb_base, bi_base, "base"))

    # Run for floor brick type
    if len(rt_floor) > 0:
        if verbose:
            logger.info("Running grid search for floor brick type...")
        results.extend(
            run_brick_type(
                rt_floor,
                rd_floor,
                rc_floor,
                rb_floor,
                bi_floor,
                f"floor_k{config.spread_gate.floor_k4}",
            )
        )

    return results


def find_pareto_frontier(
    results: List[AdaptiveBacktestResult],
) -> List[AdaptiveBacktestResult]:
    """
    Extract Pareto frontier (non-dominated points in return vs drawdown).

    A result dominates another if it has higher (or equal) return AND
    lower (or equal) drawdown, with at least one strict improvement.

    Args:
        results: List of backtest results

    Returns:
        List of non-dominated results
    """
    if not results:
        return []

    pts = np.array([[r.end_balance, abs(r.max_drawdown)] for r in results])
    keep = []

    for i, (eb, dd) in enumerate(pts):
        dominated = False
        for j, (eb2, dd2) in enumerate(pts):
            if i == j:
                continue
            if (eb2 >= eb and dd2 <= dd) and (eb2 > eb or dd2 < dd):
                dominated = True
                break
        if not dominated:
            keep.append(i)

    return [results[i] for i in keep]


# ==============================================================================
# QUALIFICATION PIPELINE
# ==============================================================================


def qualify_instrument_adaptive(
    symbol: str,
    m1_df: pd.DataFrame,
    spread_pts: float | NDArray[np.float64],
    tick_size: float = DEFAULT_TICK,
    config: Optional[AdaptiveBacktestConfig] = None,
    output_dir: Optional[Path] = None,
    verbose: bool = False,
) -> AdaptiveQualificationResult:
    """
    Run full adaptive qualification pipeline for a single instrument.

    This is the main entry point for spread-gated qualification. It:
    1. Computes adaptive brick sizes from volatility
    2. Runs grid search over spread gate parameters
    3. Selects best configuration based on score
    4. Evaluates quality gates
    5. Returns complete qualification result

    Args:
        symbol: Instrument symbol
        m1_df: M1 OHLCV DataFrame (must have columns: time, open, high, low, close, spread)
        spread_pts: Spread in points (scalar or array)
        tick_size: Tick size in price units
        config: Backtest configuration (uses defaults if None)
        output_dir: Directory to save results (None = no save)
        verbose: Log progress

    Returns:
        AdaptiveQualificationResult with all configurations and best selection
    """
    if config is None:
        config = AdaptiveBacktestConfig()

    # Create instrument ID
    instrument_id = symbol

    # Validate data
    required_cols = ["time", "open", "high", "low", "close"]
    for col in required_cols:
        if col not in m1_df.columns:
            raise ValueError(f"Missing required column: {col}")

    # Prepare data
    df = m1_df.copy()
    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values("time").reset_index(drop=True)

    if len(df) < config.min_bars:
        return AdaptiveQualificationResult(
            symbol=symbol,
            instrument_id=instrument_id,
            status=AdaptiveBacktestStatus.INSUFFICIENT_DATA,
            best_config=AdaptiveBacktestResult(gate_type="none"),
            total_bars=len(df),
        )

    # Extract arrays
    times_ns = df["time"].astype("int64").to_numpy()
    o = df["open"].to_numpy(dtype=np.float64)
    h = df["high"].to_numpy(dtype=np.float64)
    l = df["low"].to_numpy(dtype=np.float64)
    c = df["close"].to_numpy(dtype=np.float64)

    # Handle spread
    if isinstance(spread_pts, (int, float)):
        spread_arr = np.full(len(df), float(spread_pts), dtype=np.float64)
    else:
        spread_arr = np.asarray(spread_pts, dtype=np.float64)
        if len(spread_arr) != len(df):
            raise ValueError(
                f"spread_pts length ({len(spread_arr)}) must match df length ({len(df)})"
            )

    # If no spread column in df, use provided spread_pts
    if "spread" not in df.columns or df["spread"].isna().all():
        pass  # Use spread_arr
    else:
        # Override with df spread if present and valid
        df_spread = df["spread"].to_numpy(dtype=np.float64)
        valid_mask = np.isfinite(df_spread) & (df_spread > 0)
        spread_arr[valid_mask] = df_spread[valid_mask]

    # Compute adaptive brick sizes
    if verbose:
        logger.info("Computing adaptive brick sizes for %s...", symbol)

    sig = garman_klass_sigma(o, h, l, c, config.sigma_window)
    sig = np.nan_to_num(sig, nan=0.0)

    v = roofing_filter(sig, hp_period=config.hp_period, ss_period=config.ss_period)
    med = np.nanmedian(v)
    if med <= 0:
        med = np.nanmedian(v[v > 0]) if np.any(v > 0) else 1.0

    braw = v * (1.0 / med if med > 0 else 1.0)
    braw = np.nan_to_num(braw, nan=1.0)

    # Apply stability
    brick_base = apply_stability(
        braw,
        config.brick_stability.band_low,
        config.brick_stability.band_high,
        config.brick_stability.hysteresis,
        config.brick_stability.rate_limit,
    )

    # Apply friction floor
    spread_price = spread_arr * tick_size
    brick_floor = apply_stability(
        np.maximum(brick_base, config.spread_gate.floor_k4 * spread_price),
        config.brick_stability.band_low,
        config.brick_stability.band_high,
        config.brick_stability.hysteresis,
        config.brick_stability.rate_limit,
    )

    # Run grid search
    if verbose:
        logger.info("Running grid search for %s...", symbol)

    all_results = run_grid_search(
        times_ns=times_ns,
        price=c,
        spread_pts=spread_arr,
        brick_base=brick_base,
        brick_floor=brick_floor,
        config=config,
        verbose=verbose,
    )

    if not all_results:
        return AdaptiveQualificationResult(
            symbol=symbol,
            instrument_id=instrument_id,
            status=AdaptiveBacktestStatus.INSUFFICIENT_DATA,
            best_config=AdaptiveBacktestResult(gate_type="none"),
            total_bars=len(df),
            fail_reason="No valid configurations (no bricks generated)",
        )

    # Sort by score and find best
    all_results.sort(key=lambda r: r.score, reverse=True)
    best = all_results[0]

    # Find Pareto frontier
    pareto = find_pareto_frontier(all_results)

    # Top 10 configs
    top_10 = all_results[:10]

    # Determine qualification status
    if best.passed_gates:
        status = AdaptiveBacktestStatus.QUALIFIED
    else:
        if best.trades < config.min_trades:
            status = AdaptiveBacktestStatus.INSUFFICIENT_TRADES
        elif best.max_drawdown_pct > config.max_drawdown_pct:
            status = AdaptiveBacktestStatus.EXCESSIVE_DRAWDOWN
        else:
            status = AdaptiveBacktestStatus.POOR_PERFORMANCE

    # Compute gate coverage fractions from the best configuration
    _total_entry_attempts = best.trades + best.gated_entries
    _gate_bar_fraction = best.gate_bars_open / max(best.brick_count, 1)
    _gate_trade_fraction = best.trades / max(_total_entry_attempts, 1)

    # Build result
    result = AdaptiveQualificationResult(
        symbol=symbol,
        instrument_id=instrument_id,
        status=status,
        best_config=best,
        pareto_configs=pareto,
        top_configs=top_10,
        all_configs=all_results,
        total_bars=len(df),
        total_bricks=best.brick_count,
        data_start=df["time"].iloc[0].isoformat(),
        data_end=df["time"].iloc[-1].isoformat(),
        gate_bar_fraction=_gate_bar_fraction,
        gate_trade_fraction=_gate_trade_fraction,
    )

    # Save if output_dir provided
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save individual result
        result_file = output_dir / f"{symbol}_adaptive_qualification.json"
        result_file.write_text(
            json.dumps(result.to_dict(), indent=2, default=str), encoding="utf-8"
        )

        if verbose:
            logger.info("Saved qualification result to %s", result_file)

    return result


# ==============================================================================
# CLI UTILITIES
# ==============================================================================


def print_qualification_summary(result: AdaptiveQualificationResult) -> None:
    """Print a human-readable summary of qualification result."""
    print("\n" + "=" * 70)
    print(f"ADAPTIVE QUALIFICATION: {result.symbol}")
    print("=" * 70)
    print(f"Status        : {result.status.value}")
    print(f"Data range    : {result.data_start} to {result.data_end}")
    print(f"Total bars    : {result.total_bars:,}")
    print(f"Total bricks  : {result.total_bricks:,}")
    print()

    if result.qualified:
        print("✅ QUALIFIED")
        print("-" * 70)
        print(f"Best config  : {result.best_config.gate_type}")
        if result.best_config.window:
            print(f"  Window     : {result.best_config.window} bars")
        if result.best_config.quantile:
            print(f"  Quantile   : {result.best_config.quantile}")
        if result.best_config.max_spread_brick_ratio:
            print(f"  Ratio θ    : {result.best_config.max_spread_brick_ratio}")
        print(f"  Brick type : {result.best_config.brick_type}")
        print()
        print("Performance:")
        print(f"  End balance : ${result.best_config.end_balance:,.2f}")
        print(f"  Net P&L     : ${result.best_config.net_pnl:,.2f}")
        print(f"  Omega       : {result.best_config.omega:.2f}")
        print(f"  Z-factor    : {result.best_config.z_factor:.2f}")
        print(f"  Win rate    : {result.best_config.win_rate:.1%}")
        print(f"  Profit fac  : {result.best_config.profit_factor:.3f}")
        print(
            f"  Max DD      : ${abs(result.best_config.max_drawdown):,.2f} ({result.best_config.max_drawdown_pct:.1f}%)"
        )
        print(f"  Sortino     : {result.best_config.sortino:.2f}")
        print()
        print("Execution:")
        print(f"  Trades      : {result.best_config.trades:,}")
        print(
            f"  Gated       : {result.best_config.gated_entries} ({result.best_config.gated_entries / result.best_config.trades * 100:.1f}% of flips)"
        )
        print(f"  Total cost  : ${result.best_config.total_cost_usd:,.2f}")
        print(f"  PnL / Cost  : {result.best_config.pnl_per_cost:.2f}")
        print()
        print("Bricks:")
        print(f"  Avg size    : {result.best_config.avg_brick_size:.4f}")
        print(f"  Std size    : {result.best_config.std_brick_size:.4f}")
        print()
        print(f"Score        : {result.best_config.score:.2f}")
    else:
        print("❌ NOT QUALIFIED")
        print("-" * 70)
        print(f"Reason: {result.best_config.fail_reason}")
        print()
        print("Best attempted config:")
        print(f"  Gate type   : {result.best_config.gate_type}")
        print(f"  Brick type  : {result.best_config.brick_type}")
        print(f"  End balance : ${result.best_config.end_balance:,.2f}")
        print(f"  Omega       : {result.best_config.omega:.2f}")
        print(f"  Trades      : {result.best_config.trades}")
        print(f"  Max DD      : {result.best_config.max_drawdown_pct:.1f}%")

    print("=" * 70)
