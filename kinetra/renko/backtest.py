"""
Renko Backtester
================

Two-mode Renko backtester for the Kinetra pipeline:

  **Mode 1 — Instrument Backtest:**
    Single-instrument Renko backtest with friction, filter evaluation,
    stop-loss, colour-change exit, and full trade tracking.

  **Mode 2 — Portfolio Backtest:**
    Portfolio-level backtest with time-ordered trade merging, cluster-capped
    equal-risk sizing, and composite equity curve construction.

Design decisions (from RENKO_KINETRA_DESIGN_SPEC.md §9):
  - Backtests operate on Renko bricks built from M30 close prices.
  - Entry: colour flip + filter pass (FlipRate + Markov).
  - Stop: 1 brick in backtest mode, 0.5 brick in live mode.
  - Exit: colour change (first opposite-direction brick).
  - Friction: real spread + commission + swap via ``FrictionCalculator``.
  - Metrics: Omega, Z-factor, PF, win rate, max DD, friction total.

This module is the canonical Renko backtester for Kinetra.  All scripts
should import from here rather than maintaining inline backtesting logic.

Usage::

    from kinetra.renko.backtest import (
        backtest_instrument,
        backtest_portfolio,
        FilterParams,
        RenkoTrade,
        InstrumentBacktestResult,
        PortfolioBacktestResult,
    )

See Also:
    - ``kinetra/renko/brick_engine.py`` — Renko brick construction
    - ``kinetra/renko/filters.py`` — FlipRate, Markov signal filters
    - ``kinetra/renko/dsp.py`` — DSP analysis (VR, brick sizing, friction)
    - ``kinetra/renko/portfolio.py`` — portfolio construction, clustering, sizing
    - ``docs/MANUAL.md §9`` — backtesting specification
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from kinetra.renko.brick_engine import build_renko
from kinetra.renko.filters import (
    evaluate_entry,
    flip_rate,
    markov_stickiness,
)
from kinetra.renko.vol_sizer import quantize_and_clamp_lot

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# Data containers
# ══════════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True, slots=True)
class FilterParams:
    """
    Configuration for Renko signal filters.

    All thresholds are per-instrument and should be derived empirically
    or via DSP scaling (see ``kinetra/renko/dsp.py``).  Fixed defaults
    are provided as sensible starting points but MUST be validated.

    Attributes
    ----------
    fliprate_window : int
        Rolling window size for FlipRate computation (bricks).
    fliprate_threshold : float
        Maximum FlipRate to allow entry (lower = stricter).  Must be in (0, 1).
    markov_window : int
        Rolling window size for Markov stickiness computation (bricks).
    markov_threshold : float
        Minimum Markov stickiness to allow entry (higher = stricter).  Must be in (0, 1).
    """

    fliprate_window: int = 50
    fliprate_threshold: float = 0.35
    markov_window: int = 50
    markov_threshold: float = 0.55

    def __post_init__(self) -> None:
        if self.fliprate_window < 2:
            raise ValueError(f"fliprate_window must be >= 2, got {self.fliprate_window}")
        if not (0.0 < self.fliprate_threshold < 1.0):
            raise ValueError(f"fliprate_threshold must be in (0, 1), got {self.fliprate_threshold}")
        if self.markov_window < 2:
            raise ValueError(f"markov_window must be >= 2, got {self.markov_window}")
        if not (0.0 < self.markov_threshold < 1.0):
            raise ValueError(f"markov_threshold must be in (0, 1), got {self.markov_threshold}")


@dataclass(frozen=True, slots=True)
class StopParams:
    """
    Stop-loss and exit configuration.

    Attributes
    ----------
    stop_bricks : float
        Stop distance as a multiple of brick_size.
        Default 1.0 for backtest, 0.5 for live.
    exit_on_colour_change : bool
        If True, exit on first opposite-direction brick.
    allow_short : bool
        If True, allow short entries on down-flips.
    """

    stop_bricks: float = 1.0
    exit_on_colour_change: bool = True
    allow_short: bool = True


@dataclass(frozen=True, slots=True)
class RenkoEmpiricalCore:
    """
    Locked empirical strategy core for XAUUSD Renko flip trading.

    This profile encodes the fixed assumptions from the discovery phase:
    - instrument: XAUUSD
    - brick_size: 1.0 price units
    - entry: colour flip gated by Markov persistence
    - exit: opposite colour flip (FlipExit)
    - stop: 1 brick worst-case
    """

    instrument: str
    brick_size: float
    filter_params: FilterParams
    stop_params: StopParams


def xauusd_empirical_core() -> RenkoEmpiricalCore:
    """
    Return the fixed empirical Renko core discovered on XAUUSD (2023-2026).

    The intent is to keep signal logic stable and avoid reintroducing
    indicator/parameter churn after discovery has converged.
    """
    return RenkoEmpiricalCore(
        instrument="XAUUSD",
        brick_size=1.0,
        filter_params=FilterParams(
            # Keep FlipRate defaults; signal gate is Markov persistence.
            fliprate_window=50,
            fliprate_threshold=0.35,
            markov_window=50,
            markov_threshold=0.55,
        ),
        stop_params=StopParams(
            stop_bricks=1.0,
            exit_on_colour_change=True,
            allow_short=True,
        ),
    )


@dataclass(frozen=True, slots=True)
class RiskParams:
    """
    Risk management parameters for loss-cluster breaker and drawdown controls.

    Attributes
    ----------
    loss_cluster_window : int
        Number of recent trades to evaluate for loss clustering.
    loss_cluster_threshold : float
        Fraction of losses in the window to trigger throttle (0-1).
    loss_cluster_cooldown : int
        Number of trades to skip entries after throttle trigger.
    dd_throttle_pct : float
        Drawdown threshold to trigger reduced exposure (0-1).
        For single instrument, this skips entries.
    dd_halt_pct : float
        Drawdown threshold to halt trading entirely (0-1).
    """

    loss_cluster_window: int = 10
    loss_cluster_threshold: float = 0.7
    loss_cluster_cooldown: int = 5
    dd_throttle_pct: float = 0.5
    dd_halt_pct: float = 0.2


class SizingMode(str, Enum):
    """
    Position sizing mode for backtesting.

    Attributes
    ----------
    FIXED_LOT
        Every trade uses exactly ``VolSizingParams.fixed_lot`` lots.
        This is the simplest baseline — unit P&L, independent of equity.
        Produces equity curves in USD-per-trade units.

    COMPOUNDING
        Classic fixed-lot-per-equity compounding:
        ``lots = equity / compounding_capital_per_lot``.
        Replicates the research baseline (0.01 lot / $1,000 equity).
        Produces exponential equity curves that are difficult to compare
        across instruments or portfolio configurations.

    VOL_TARGET
        Volatility-targeted sizing (the Next Experiment from the research
        chronology):
        ``lots = (target_vol_pct × equity) / (brick × usd_per_point × vol̂ × 100k)``
        Where ``vol̂`` is a causal rolling std of per-brick P&L fractions.
        Produces equity curves that are directly comparable across
        instruments and portfolio configurations.
    """

    FIXED_LOT = "fixed_lot"
    COMPOUNDING = "compounding"
    VOL_TARGET = "vol_target"


@dataclass(frozen=True, slots=True)
class VolSizingParams:
    """
    Parameters for vol-targeted sizing inside :func:`backtest_instrument`.

    Attributes
    ----------
    target_vol_pct : float
        Fraction of equity to risk per 1σ brick-vol move.
        Default ``0.01`` (1 %).
    vol_window : int
        Causal rolling window for brick P&L std (bricks).
        Default ``50``.
    min_vol_window : int
        Minimum bricks before vol is estimated from data.
        Default ``20``.
    vol_floor : float
        Minimum vol estimate (fraction of brick_size).
        Default ``0.003``.
    vol_ceil : float
        Maximum vol estimate (fraction of brick_size).
        Default ``0.20``.
    initial_vol_fallback : float
        Conservative vol used during the warmup period.
        Default ``0.02``.
    usd_per_point : float
        USD value of 1 price unit per 1 standard lot.
        Required for VOL_TARGET mode.  Default ``1.0``.
    initial_equity : float
        Starting equity for the equity-proportional sizing.
        Default ``100_000.0``.
    lot_step : float
        Broker minimum lot increment.
        Default ``0.01``.
    min_lots : float
        Minimum lot size; below this the trade is skipped.
        Default ``0.01``.
    lot_ceiling : float
        Hard upper cap on lot size.
        Default ``100.0``.
    fixed_lot : float
        Lot size used for ``FIXED_LOT`` mode, and also used as the
        compounding lot unit for ``COMPOUNDING`` mode.
        Default ``0.01``.
    compounding_capital_per_lot : float
        Equity per 1 lot increment in ``COMPOUNDING`` mode.
        Replicates the research baseline when set to ``1_000.0``
        (0.01 lot / $1,000).
        Default ``1_000.0``.
    """

    target_vol_pct: float = 0.01
    vol_window: int = 50
    min_vol_window: int = 20
    vol_floor: float = 0.003
    vol_ceil: float = 0.20
    initial_vol_fallback: float = 0.02
    usd_per_point: float = 1.0
    initial_equity: float = 100_000.0
    lot_step: float = 0.01
    min_lots: float = 0.01
    lot_ceiling: float = 100.0
    fixed_lot: float = 0.01
    compounding_capital_per_lot: float = 1_000.0


@dataclass(slots=True)
class RenkoTrade:
    """
    A single completed Renko round-trip trade.

    Lightweight trade record optimised for Renko backtesting.
    Retains all information needed for portfolio equity curve construction,
    friction analysis, and performance reporting.
    """

    trade_id: str  # Sequential trade identifier (e.g., "T-000001")
    symbol: str
    direction: int  # +1 long, -1 short
    entry_price: float
    exit_price: float
    entry_time: datetime
    exit_time: datetime
    gross_pts: float  # (exit - entry) × direction  [price points]
    gross_usd: float  # USD P&L before friction
    friction_usd: float  # total friction (spread + commission + swap + slippage)
    net_usd: float  # gross_usd − friction_usd
    exit_reason: str  # "colour_change" | "stop" | "end_of_data"
    brick_size: float  # brick size used for this trade
    n_bricks_held: int = 0  # bricks held during the trade
    lots: float = 1.0  # lot size for this trade (sizing-mode aware)
    sizing_mode: str = SizingMode.FIXED_LOT.value  # which sizing mode produced this trade

    @property
    def is_winner(self) -> bool:
        """Return True if the trade was net profitable."""
        return self.net_usd > 0.0

    @property
    def holding_seconds(self) -> float:
        """Wall-clock holding duration in seconds."""
        return (self.exit_time - self.entry_time).total_seconds()

    @property
    def holding_hours(self) -> float:
        """Wall-clock holding duration in hours."""
        return self.holding_seconds / 3600.0

    @property
    def friction_ratio(self) -> float:
        """Friction as a fraction of gross |P&L|.  Inf if gross is zero."""
        absgrss = abs(self.gross_usd)
        if absgrss < 1e-12:
            return float("inf")
        return self.friction_usd / absgrss

    def to_dict(self) -> dict:
        """Serialize trade to dictionary for JSON export."""
        return {
            "trade_id": self.trade_id,
            "symbol": self.symbol,
            "direction": self.direction,
            "entry_price": self.entry_price,
            "exit_price": self.exit_price,
            "entry_time": self.entry_time.isoformat() if self.entry_time else None,
            "exit_time": self.exit_time.isoformat() if self.exit_time else None,
            "gross_pts": self.gross_pts,
            "gross_usd": self.gross_usd,
            "friction_usd": self.friction_usd,
            "net_usd": self.net_usd,
            "exit_reason": self.exit_reason,
            "brick_size": self.brick_size,
            "n_bricks_held": self.n_bricks_held,
            "lots": self.lots,
            "sizing_mode": self.sizing_mode,
        }


@dataclass(slots=True)
class InstrumentBacktestResult:
    """
    Complete backtest result for a single instrument.

    Attributes
    ----------
    symbol : str
        Instrument symbol.
    brick_size : float
        Brick size used.
    filter_params : FilterParams
        Filter configuration used.
    stop_params : StopParams
        Stop/exit configuration used.
    trades : list[RenkoTrade]
        All completed trades.
    equity_curve : list[float]
        Cumulative net P&L after each trade (starts at 0.0).
    n_source_bars : int
        Number of M30 source bars.
    n_bricks : int
        Number of Renko bricks produced.
    years : float
        Duration of the data in years.
    omega : float
        Omega ratio of net returns.
    z_factor : float
        Z-factor of net returns.
    profit_factor : float
        Sum of wins / sum of losses.
    win_rate : float
        Fraction of trades that are net profitable.
    max_dd_usd : float
        Maximum drawdown in USD.
    total_friction_usd : float
        Total friction costs across all trades.
    avg_net_per_trade : float
        Average net P&L per trade.
    trades_per_year : float
        Annualised trade frequency.
    """

    symbol: str
    brick_size: float
    filter_params: FilterParams
    stop_params: StopParams
    trades: List[RenkoTrade]
    equity_curve: List[float]
    n_source_bars: int
    n_bricks: int
    years: float
    omega: float
    z_factor: float
    profit_factor: float
    win_rate: float
    max_dd_usd: float
    total_friction_usd: float
    avg_net_per_trade: float
    trades_per_year: float


@dataclass(slots=True)
class PortfolioBacktestResult:
    """
    Portfolio-level backtest result with time-ordered equity curve.

    Attributes
    ----------
    instruments : list[str]
        Symbols included in the portfolio.
    instrument_results : dict[str, InstrumentBacktestResult]
        Per-instrument results.
    allocation_weights : dict[str, float]
        Allocation weight per instrument (from sizing / RL).
    equity_curve : list[float]
        Portfolio cumulative equity curve (time-ordered trade merge).
    total_trades : int
        Total trades across all instruments.
    omega : float
        Portfolio-level Omega ratio.
    z_factor : float
        Portfolio-level Z-factor.
    profit_factor : float
        Portfolio-level profit factor.
    win_rate : float
        Portfolio-level win rate.
    net_pnl_usd : float
        Total net P&L.
    max_dd_usd : float
        Portfolio maximum drawdown.
    calmar_ratio : float
        Annualised return / |max drawdown|.
    years : float
        Duration of the portfolio (max across instruments).
    trades_per_year : float
        Annualised trade count.
    per_instrument_contribution : dict[str, float]
        Net P&L contribution per instrument.
    cluster_contribution : dict[str, float]
        Net P&L contribution per cluster.
    """

    instruments: List[str]
    instrument_results: Dict[str, InstrumentBacktestResult]
    allocation_weights: Dict[str, float]
    equity_curve: List[float]
    total_trades: int
    omega: float
    z_factor: float
    profit_factor: float
    win_rate: float
    net_pnl_usd: float
    max_dd_usd: float
    calmar_ratio: float
    years: float
    trades_per_year: float
    per_instrument_contribution: Dict[str, float] = field(default_factory=dict)
    cluster_contribution: Dict[str, float] = field(default_factory=dict)


# ══════════════════════════════════════════════════════════════════════════════
# Internal helpers
# ══════════════════════════════════════════════════════════════════════════════


def _ensure_utc(ts: pd.Timestamp) -> datetime:
    """Coerce a pandas Timestamp to a timezone-aware datetime (UTC)."""
    if ts.tzinfo is None:
        return ts.to_pydatetime().replace(tzinfo=timezone.utc)
    return ts.to_pydatetime()


def _compute_metrics(returns: np.ndarray) -> Tuple[float, float, float, float]:
    """
    Compute (omega, z_factor, profit_factor, win_rate) from net returns.

    Imports from canonical metrics module — never reimplement these.
    """
    from kinetra.backtesting.metrics import calculate_z_factor, omega_ratio

    if len(returns) == 0:
        return 0.0, 0.0, 0.0, 0.0

    omega = float(omega_ratio(returns))
    z = float(calculate_z_factor(returns))

    gains = float(returns[returns > 0].sum()) if np.any(returns > 0) else 0.0
    losses = float(-returns[returns <= 0].sum()) if np.any(returns <= 0) else 0.0
    pf = (gains / losses) if losses > 0 else (10.0 if gains > 0 else 0.0)
    wr = float(np.mean(returns > 0))

    return omega, z, pf, wr


def _max_drawdown(equity: Sequence[float]) -> float:
    """Compute max drawdown from an equity curve.  Returns a negative value."""
    eq = np.asarray(equity, dtype=np.float64)
    if len(eq) < 2:
        return 0.0
    peak = np.maximum.accumulate(eq)
    dd = eq - peak
    return float(dd.min())


def _apply_friction_to_trade(
    calc,  # FrictionCalculator — imported lazily
    is_long: bool,
    entry_price: float,
    exit_price: float,
    entry_time: datetime,
    exit_time: datetime,
) -> Tuple[float, float, float]:
    """
    Apply friction via ``FrictionCalculator``.

    Returns (gross_usd, friction_usd, net_usd).
    """
    try:
        rt = calc.round_trip(
            is_long=is_long,
            lots=1.0,
            entry_price=entry_price,
            exit_price=exit_price,
            entry_dt=entry_time,
            exit_dt=exit_time,
        )
        gross = rt.gross_pnl_usd
        friction = rt.friction.total_usd
        net = rt.net_pnl_usd
    except Exception:
        # Fallback: simple estimate
        direction = 1 if is_long else -1
        pts = (exit_price - entry_price) * direction
        gross = pts  # approximate USD = points (instrument-dependent)
        try:
            spread = calc.spec.spread_usd_per_lot
        except Exception:
            spread = 0.0
        friction = abs(spread)
        net = gross - friction

    return float(gross), float(friction), float(net)


# ══════════════════════════════════════════════════════════════════════════════
# Mode 1 — Instrument Backtest
# ══════════════════════════════════════════════════════════════════════════════


def backtest_instrument(
    symbol: str,
    closes: pd.Series,
    brick_size: float,
    filter_params: Optional[FilterParams] = None,
    stop_params: Optional[StopParams] = None,
    risk_params: Optional[RiskParams] = None,
    calc=None,
    *,
    date_start: Optional[datetime] = None,
    date_end: Optional[datetime] = None,
    warmup_override: Optional[int] = None,
    session_break_minutes: float = 30.0,
    sizing_mode: SizingMode = SizingMode.FIXED_LOT,
    vol_sizing_params: Optional[VolSizingParams] = None,
) -> InstrumentBacktestResult:
    """
    Run a single-instrument Renko backtest.

    Parameters
    ----------
    symbol : str
        Instrument symbol (for labelling trades).
    closes : pd.Series
        Close prices indexed by datetime.  Typically M30 close prices.
    brick_size : float
        Fixed brick size in price units.
    filter_params : FilterParams or None
        Filter configuration.  Defaults to ``FilterParams()`` if None.
    stop_params : StopParams or None
        Stop/exit configuration.  Defaults to ``StopParams()`` if None.
    risk_params : RiskParams or None
        Risk management configuration.  Defaults to ``RiskParams()`` if None.
    calc : FrictionCalculator or None
        Friction calculator for the instrument.  If None, friction is
        estimated as zero (gross = net).
    date_start : datetime or None
        If given, only use data from this time onward.
    date_end : datetime or None
        If given, only use data up to this time.
    warmup_override : int or None
        Override the default warmup period (max filter windows + buffer).
    sizing_mode : SizingMode, default SizingMode.FIXED_LOT
        Position sizing mode.  See :class:`SizingMode` for options.

        - ``FIXED_LOT`` — every trade uses ``vol_sizing_params.fixed_lot``
          lots.  P&L is in USD-per-lot-per-point units.  Default.
        - ``COMPOUNDING`` — classic 0.01-lot-per-$1,000-equity compounding.
          Replicates the research baseline.
        - ``VOL_TARGET`` — volatility-targeted sizing.  Requires
          ``vol_sizing_params.usd_per_point`` to be set correctly.
    vol_sizing_params : VolSizingParams or None
        Parameters for ``COMPOUNDING`` and ``VOL_TARGET`` modes.
        Ignored when ``sizing_mode == FIXED_LOT``.
        Defaults to ``VolSizingParams()`` if None.

    session_break_minutes : float, default 30.0
        Minimum gap in minutes to trigger a session break in
        :func:`~kinetra.renko.brick_engine.build_renko`.  Gaps above this
        threshold reset the Renko reference price so that no bricks are
        emitted across session boundaries (e.g. the daily rollover gap on
        XAUUSD).  Obtain the correct value from
        :func:`~kinetra.renko.session.detect_session_break` or
        :func:`~kinetra.renko.session.session_break_minutes_for`.

    Returns
    -------
    InstrumentBacktestResult
        Complete backtest results with trades, equity curve, and metrics.

    Raises
    ------
    ValueError
        If ``brick_size <= 0`` or ``closes`` is empty.
    """
    if brick_size <= 0:
        raise ValueError(f"brick_size must be > 0, got {brick_size}")

    vol_params_provided = vol_sizing_params is not None

    if filter_params is None:
        filter_params = FilterParams()
    if stop_params is None:
        stop_params = StopParams()
    if risk_params is None:
        risk_params = RiskParams()
    if vol_sizing_params is None:
        vol_sizing_params = VolSizingParams()

    # ── Instrument-spec economics (pip value / lot constraints / friction) ───
    # If caller did not provide explicit friction calculator and sizing params,
    # derive them from canonical broker specs.
    base_symbol = symbol.split("__", 1)[0]
    if calc is None:
        try:
            from kinetra.friction_cost import get_calculator, get_calculator_with_data

            # Prefer data-derived spread/costs (CSV-measured) for realistic backtests.
            # Fall back to spec-only calculator when spread measurement data is absent.
            try:
                calc = get_calculator_with_data(base_symbol, timeframe="M1")
            except Exception:
                calc = get_calculator(base_symbol)
        except Exception as exc:
            logger.debug("%s: could not auto-load friction calculator (%s)", base_symbol, exc)
            calc = None

    if (not vol_params_provided) and (calc is not None) and hasattr(calc, "spec"):
        try:
            spec = calc.spec
            tick = max(float(getattr(spec, "tick_size", 0.0)), 1e-12)
            tick_value_usd = float(getattr(spec, "tick_value_usd", 0.0))
            # Backtester expects USD value of 1 full price unit per 1 lot.
            usd_per_price_unit = tick_value_usd / tick if tick_value_usd > 0 else 1.0

            lot_step = max(float(getattr(spec, "volume_step", vol_sizing_params.lot_step)), 1e-6)
            min_lots = max(float(getattr(spec, "volume_min", vol_sizing_params.min_lots)), 0.0)
            lot_ceiling = max(
                float(getattr(spec, "volume_max", vol_sizing_params.lot_ceiling)), min_lots
            )

            vol_sizing_params = VolSizingParams(
                target_vol_pct=vol_sizing_params.target_vol_pct,
                vol_window=vol_sizing_params.vol_window,
                min_vol_window=vol_sizing_params.min_vol_window,
                vol_floor=vol_sizing_params.vol_floor,
                vol_ceil=vol_sizing_params.vol_ceil,
                initial_vol_fallback=vol_sizing_params.initial_vol_fallback,
                usd_per_point=usd_per_price_unit,
                initial_equity=vol_sizing_params.initial_equity,
                lot_step=lot_step,
                min_lots=min_lots,
                lot_ceiling=lot_ceiling,
                fixed_lot=max(min_lots, vol_sizing_params.fixed_lot),
                compounding_capital_per_lot=vol_sizing_params.compounding_capital_per_lot,
            )
        except Exception as exc:
            logger.debug("%s: could not auto-derive sizing from spec (%s)", base_symbol, exc)

    # ── Date filtering ──────────────────────────────────────────────────
    if date_start is not None:
        ts_start = pd.Timestamp(date_start)
        if ts_start.tzinfo is None:
            ts_start = ts_start.tz_localize("UTC")
        mask = closes.index >= ts_start
        closes = closes.loc[mask]
    if date_end is not None:
        ts_end = pd.Timestamp(date_end)
        if ts_end.tzinfo is None:
            ts_end = ts_end.tz_localize("UTC")
        mask = closes.index <= ts_end
        closes = closes.loc[mask]

    n_source = len(closes)
    if n_source == 0:
        return _empty_instrument_result(symbol, brick_size, filter_params, stop_params)

    # Estimate years from data span
    if n_source >= 2:
        span = (closes.index[-1] - closes.index[0]).total_seconds()
        years = max(span / (365.25 * 86400), 0.001)
    else:
        years = 0.001

    # ── Build Renko bricks ──────────────────────────────────────────────
    bricks = build_renko(closes, brick_size, session_break_minutes=session_break_minutes)
    n_bricks = len(bricks)

    if n_bricks == 0:
        return _empty_instrument_result(
            symbol,
            brick_size,
            filter_params,
            stop_params,
            n_source_bars=n_source,
            years=years,
        )

    # ── Compute filter signals ──────────────────────────────────────────
    directions = bricks["direction"].values.astype(np.int8)
    brick_closes = bricks["brick_close"].values.astype(np.float64)
    brick_times = bricks["time"].values

    fr_vals = flip_rate(directions, filter_params.fliprate_window)
    pUU_vals, pDD_vals = markov_stickiness(directions, filter_params.markov_window)

    # ── Warmup period ───────────────────────────────────────────────────
    filter_warmup = max(filter_params.fliprate_window, filter_params.markov_window)
    warmup = warmup_override if warmup_override is not None else (filter_warmup + 5)
    warmup = min(warmup, n_bricks - 1)  # can't exceed available bricks

    if n_bricks <= warmup + 2:
        return _empty_instrument_result(
            symbol,
            brick_size,
            filter_params,
            stop_params,
            n_source_bars=n_source,
            n_bricks=n_bricks,
            years=years,
        )

    # ── Vol-targeted sizing: precompute causal rolling vol array ─────────
    # This is done once before the loop (fully vectorised, no lookahead).
    # We use the signed brick P&L (brick_size × direction) as the input.
    _vol_lots_array: Optional[np.ndarray] = None
    if sizing_mode == SizingMode.VOL_TARGET:
        from kinetra.renko.vol_sizer import (
            VolSizingConfig,
            compute_vol_targeted_lots_batch,
        )

        _brick_pts = directions.astype(np.float64) * brick_size
        _equity_curve_array = np.full(n_bricks, vol_sizing_params.initial_equity, dtype=np.float64)
        _vs_cfg = VolSizingConfig(
            target_vol_pct=vol_sizing_params.target_vol_pct,
            vol_window=vol_sizing_params.vol_window,
            min_vol_window=vol_sizing_params.min_vol_window,
            vol_floor=vol_sizing_params.vol_floor,
            vol_ceil=vol_sizing_params.vol_ceil,
            initial_vol_fallback=vol_sizing_params.initial_vol_fallback,
        )
        # Compute lots at every brick position; we use equity=initial_equity
        # for the precomputed array — equity is then scaled in-loop as
        # cumulative equity changes.  We store the base lots-per-equity-unit
        # here and multiply by live equity inside the loop.
        # Actually, pass all-ones equity so we get lots per $1 equity, then
        # multiply in-loop by actual equity.
        _equity_ones = np.ones(n_bricks, dtype=np.float64)
        _vol_lots_array = compute_vol_targeted_lots_batch(
            brick_pts=_brick_pts,
            brick_size=brick_size,
            equity_curve=_equity_ones,
            usd_per_point=vol_sizing_params.usd_per_point,
            config=_vs_cfg,
            lot_step=vol_sizing_params.lot_step,
            min_lots=0.0,  # will apply min_lots in-loop after equity scaling
            lot_ceiling=vol_sizing_params.lot_ceiling,
        )

    # ── Core backtest loop ──────────────────────────────────────────────
    stop_distance = stop_params.stop_bricks * brick_size
    trades: List[RenkoTrade] = []
    equity: List[float] = [0.0]
    cumulative = 0.0
    _trade_counter = 0  # Sequential trade ID counter

    # Running equity for sizing (starts at initial_equity for VOL_TARGET /
    # COMPOUNDING; unused for FIXED_LOT)
    live_equity = vol_sizing_params.initial_equity

    in_pos = False
    pos_direction = 0
    entry_price = 0.0
    entry_time: Optional[datetime] = None
    entry_brick_idx = 0
    entry_lots: float = 1.0  # lots opened for the current position

    # Risk management state
    peak_equity = 0.0
    recent_wins: List[bool] = []
    throttle_counter = 0
    halted = False

    for i in range(warmup, n_bricks):
        price = float(brick_closes[i])
        bd = int(directions[i])
        bt = _ensure_utc(pd.Timestamp(brick_times[i]))

        fr_val = float(fr_vals[i]) if np.isfinite(fr_vals[i]) else float("nan")
        pUU_val = float(pUU_vals[i]) if np.isfinite(pUU_vals[i]) else float("nan")
        pDD_val = float(pDD_vals[i]) if np.isfinite(pDD_vals[i]) else float("nan")

        # ── Check exits ─────────────────────────────────────────────
        if in_pos:
            # Real-time halt: enforce the drawdown circuit-breaker against
            # *unrealized* equity so that a runaway open position cannot
            # exceed the halt threshold by an arbitrary amount before it
            # closes.  `cumulative` only reflects realized P&L; unrealized
            # P&L is approximated as (price Δ) × lots × usd_per_point.
            if not halted and peak_equity > 0:
                unreal_pts = (price - entry_price) * pos_direction
                unreal_usd = unreal_pts * entry_lots * vol_sizing_params.usd_per_point
                unreal_cumulative = cumulative + unreal_usd
                if (peak_equity - unreal_cumulative) / peak_equity > risk_params.dd_halt_pct:
                    halted = True

            stop_hit = (pos_direction == 1 and price <= entry_price - stop_distance) or (
                pos_direction == -1 and price >= entry_price + stop_distance
            )
            colour_change = stop_params.exit_on_colour_change and bd != pos_direction

            if stop_hit or colour_change or halted:
                reason = (
                    "stop"
                    if stop_hit
                    else ("halt" if halted and not colour_change else "colour_change")
                )
                n_held = i - entry_brick_idx

                gross_1lot, fric_1lot, net_1lot = _apply_friction_safe(
                    calc,
                    pos_direction == 1,
                    entry_price,
                    price,
                    entry_time,
                    bt,
                )
                # Scale P&L by actual lot size
                gross = gross_1lot * entry_lots
                fric = fric_1lot * entry_lots
                net = net_1lot * entry_lots

                cumulative += net
                live_equity = vol_sizing_params.initial_equity + cumulative
                _trade_counter += 1
                trades.append(
                    RenkoTrade(
                        trade_id=f"BT-{_trade_counter:06d}",
                        symbol=symbol,
                        direction=pos_direction,
                        entry_price=entry_price,
                        exit_price=price,
                        entry_time=entry_time,
                        exit_time=bt,
                        gross_pts=(price - entry_price) * pos_direction,
                        gross_usd=gross,
                        friction_usd=fric,
                        net_usd=net,
                        exit_reason=reason,
                        brick_size=brick_size,
                        n_bricks_held=n_held,
                        lots=entry_lots,
                        sizing_mode=sizing_mode.value,
                    )
                )
                equity.append(cumulative)
                in_pos = False

                # Update risk state after trade
                peak_equity = max(peak_equity, cumulative)
                dd = (peak_equity - cumulative) / peak_equity if peak_equity > 0 else 0.0
                if dd > risk_params.dd_halt_pct:
                    halted = True

                recent_wins.append(net > 0)
                if len(recent_wins) > risk_params.loss_cluster_window:
                    recent_wins = recent_wins[-risk_params.loss_cluster_window :]
                loss_frac = 1.0 - (sum(recent_wins) / len(recent_wins)) if recent_wins else 0.0
                if loss_frac > risk_params.loss_cluster_threshold:
                    throttle_counter = risk_params.loss_cluster_cooldown

        # ── Check entries ───────────────────────────────────────────
        if not in_pos and not halted:
            # Risk throttle: skip entries if in cooldown
            if throttle_counter > 0:
                throttle_counter -= 1
                continue

            # Only enter on a colour flip.
            # NOTE: on the same brick where an exit occurred, both the exit
            # and entry use identical brick_close prices.  This means friction
            # is charged twice at the same price (once for the close, once for
            # the re-open).  This is intentional: the flip strategy models
            # real-world round-turn costs on every reversal, not netting them.
            is_flip = i > 0 and directions[i] != directions[i - 1]
            if not is_flip:
                continue

            # Evaluate filter gates
            entry_ok = evaluate_entry(
                direction=bd,
                flip_rate_val=fr_val,
                pUU=pUU_val,
                pDD=pDD_val,
                fliprate_threshold=filter_params.fliprate_threshold,
                markov_threshold=filter_params.markov_threshold,
            )

            if entry_ok:
                # ── Compute lot size based on sizing mode ────────────
                trade_lots = _compute_trade_lots(
                    sizing_mode=sizing_mode,
                    vol_sizing_params=vol_sizing_params,
                    vol_lots_array=_vol_lots_array,
                    brick_idx=i,
                    live_equity=live_equity,
                    brick_size=brick_size,
                )
                if trade_lots <= 0:
                    # Skip entry — vol target sizer returned 0 (e.g. equity too low)
                    continue

                if bd == 1:
                    in_pos = True
                    pos_direction = 1
                    entry_price = price
                    entry_time = bt
                    entry_brick_idx = i
                    entry_lots = trade_lots
                elif bd == -1 and stop_params.allow_short:
                    in_pos = True
                    pos_direction = -1
                    entry_price = price
                    entry_time = bt
                    entry_brick_idx = i
                    entry_lots = trade_lots

    # ── Force-close open position at end of data ────────────────────────
    if in_pos:
        price = float(brick_closes[-1])
        bt = _ensure_utc(pd.Timestamp(brick_times[-1]))
        n_held = (n_bricks - 1) - entry_brick_idx

        gross_1lot, fric_1lot, net_1lot = _apply_friction_safe(
            calc,
            pos_direction == 1,
            entry_price,
            price,
            entry_time,
            bt,
        )
        gross = gross_1lot * entry_lots
        fric = fric_1lot * entry_lots
        net = net_1lot * entry_lots

        cumulative += net
        _trade_counter += 1
        trades.append(
            RenkoTrade(
                trade_id=f"BT-{_trade_counter:06d}",
                symbol=symbol,
                direction=pos_direction,
                entry_price=entry_price,
                exit_price=price,
                entry_time=entry_time,
                exit_time=bt,
                gross_pts=(price - entry_price) * pos_direction,
                gross_usd=gross,
                friction_usd=fric,
                net_usd=net,
                exit_reason="end_of_data",
                brick_size=brick_size,
                n_bricks_held=n_held,
                lots=entry_lots,
                sizing_mode=sizing_mode.value,
            )
        )
        equity.append(cumulative)

        # Update risk state for final trade
        peak_equity = max(peak_equity, cumulative)
        recent_wins.append(net > 0)
        if len(recent_wins) > risk_params.loss_cluster_window:
            recent_wins = recent_wins[-risk_params.loss_cluster_window :]

    # ── Compute metrics ─────────────────────────────────────────────────
    if trades:
        ret = np.array([t.net_usd for t in trades], dtype=np.float64)
        omega, z, pf, wr = _compute_metrics(ret)
        max_dd = _max_drawdown(equity)
        total_friction = float(sum(t.friction_usd for t in trades))
        avg_net = float(ret.mean())
        tpy = len(trades) / max(years, 0.001)
    else:
        omega, z, pf, wr = 0.0, 0.0, 0.0, 0.0
        max_dd = 0.0
        total_friction = 0.0
        avg_net = 0.0
        tpy = 0.0

    return InstrumentBacktestResult(
        symbol=symbol,
        brick_size=brick_size,
        filter_params=filter_params,
        stop_params=stop_params,
        trades=trades,
        equity_curve=equity,
        n_source_bars=n_source,
        n_bricks=n_bricks,
        years=years,
        omega=omega,
        z_factor=z,
        profit_factor=pf,
        win_rate=wr,
        max_dd_usd=max_dd,
        total_friction_usd=total_friction,
        avg_net_per_trade=avg_net,
        trades_per_year=tpy,
    )


def _compute_trade_lots(
    sizing_mode: SizingMode,
    vol_sizing_params: VolSizingParams,
    vol_lots_array: Optional[np.ndarray],
    brick_idx: int,
    live_equity: float,
    brick_size: float,
) -> float:
    """
    Return the lot size for a trade entry at *brick_idx*.

    Parameters
    ----------
    sizing_mode : SizingMode
        Active sizing mode.
    vol_sizing_params : VolSizingParams
        Sizing configuration (used for COMPOUNDING and VOL_TARGET).
    vol_lots_array : np.ndarray or None
        Precomputed per-brick lots-per-$1-equity from the vectorised vol
        calculation.  Only used in VOL_TARGET mode.
    brick_idx : int
        Current brick index.
    live_equity : float
        Current running equity (initial_equity + cumulative P&L).
    brick_size : float
        Brick size (price units).

    Returns
    -------
    float
        Lot size to use.  Returns 0.0 to skip the trade.
    """
    if sizing_mode == SizingMode.FIXED_LOT:
        return vol_sizing_params.fixed_lot

    if sizing_mode == SizingMode.COMPOUNDING:
        cap = max(vol_sizing_params.compounding_capital_per_lot, 1.0)
        lots = (live_equity / cap) * vol_sizing_params.fixed_lot
        return quantize_and_clamp_lot(
            lots,
            lot_step=vol_sizing_params.lot_step,
            min_lots=vol_sizing_params.min_lots,
            ceilings=(vol_sizing_params.lot_ceiling,),
        )

    if sizing_mode == SizingMode.VOL_TARGET:
        if vol_lots_array is None or brick_idx >= len(vol_lots_array):
            return 0.0
        # vol_lots_array contains lots-per-$1-equity; scale by live equity
        lots_per_dollar = float(vol_lots_array[brick_idx])
        scaled = lots_per_dollar * max(live_equity, 0.0)
        return quantize_and_clamp_lot(
            scaled,
            lot_step=vol_sizing_params.lot_step,
            min_lots=vol_sizing_params.min_lots,
            ceilings=(vol_sizing_params.lot_ceiling,),
        )

    return vol_sizing_params.fixed_lot  # fallback


def _apply_friction_safe(
    calc,
    is_long: bool,
    entry_price: float,
    exit_price: float,
    entry_time: Optional[datetime],
    exit_time: datetime,
) -> Tuple[float, float, float]:
    """Apply friction calculator with fallback for None calc."""
    if calc is None:
        direction = 1 if is_long else -1
        pts = (exit_price - entry_price) * direction
        return float(pts), 0.0, float(pts)

    et = entry_time if entry_time is not None else exit_time
    return _apply_friction_to_trade(calc, is_long, entry_price, exit_price, et, exit_time)


def _empty_instrument_result(
    symbol: str,
    brick_size: float,
    filter_params: FilterParams,
    stop_params: StopParams,
    *,
    n_source_bars: int = 0,
    n_bricks: int = 0,
    years: float = 0.0,
) -> InstrumentBacktestResult:
    """Return an empty result when no trades can be generated."""
    return InstrumentBacktestResult(
        symbol=symbol,
        brick_size=brick_size,
        filter_params=filter_params,
        stop_params=stop_params,
        trades=[],
        equity_curve=[0.0],
        n_source_bars=n_source_bars,
        n_bricks=n_bricks,
        years=years,
        omega=0.0,
        z_factor=0.0,
        profit_factor=0.0,
        win_rate=0.0,
        max_dd_usd=0.0,
        total_friction_usd=0.0,
        avg_net_per_trade=0.0,
        trades_per_year=0.0,
    )


# ══════════════════════════════════════════════════════════════════════════════
# Mode 2 — Portfolio Backtest
# ══════════════════════════════════════════════════════════════════════════════


def backtest_portfolio(
    instrument_results: Dict[str, InstrumentBacktestResult],
    allocation_weights: Optional[Dict[str, float]] = None,
    *,
    cluster_map: Optional[Dict[str, str]] = None,
) -> PortfolioBacktestResult:
    """
    Build a portfolio-level backtest from per-instrument results.

    Merges all trades across instruments in time order, applies allocation
    weights, and computes portfolio-level metrics and equity curve.

    Parameters
    ----------
    instrument_results : dict[str, InstrumentBacktestResult]
        Per-instrument backtest results (keyed by symbol).
    allocation_weights : dict[str, float] or None
        Allocation weight per instrument.  If None, equal weight (1.0) is
        used for all instruments.
    cluster_map : dict[str, str] or None
        Mapping of symbol → cluster name (for cluster contribution).
        If None, all instruments are assigned to "default".

    Returns
    -------
    PortfolioBacktestResult
        Portfolio-level results with equity curve and metrics.
    """
    if not instrument_results:
        return _empty_portfolio_result()

    if allocation_weights is None:
        allocation_weights = {sym: 1.0 for sym in instrument_results}

    if cluster_map is None:
        cluster_map = {sym: "default" for sym in instrument_results}

    # ── Merge all trades (time-ordered) ─────────────────────────────────
    all_trades: List[Tuple[datetime, float, str, str]] = []
    # (exit_time, scaled_net_usd, symbol, cluster)

    for sym, result in instrument_results.items():
        weight = allocation_weights.get(sym, 1.0)
        cluster = cluster_map.get(sym, "default")
        for t in result.trades:
            scaled_net = t.net_usd * weight
            all_trades.append((t.exit_time, scaled_net, sym, cluster))

    all_trades.sort(key=lambda x: x[0])

    # ── Build portfolio equity curve ────────────────────────────────────
    equity: List[float] = [0.0]
    cumulative = 0.0
    net_returns: List[float] = []
    per_instrument_pnl: Dict[str, float] = {sym: 0.0 for sym in instrument_results}
    cluster_pnl: Dict[str, float] = {}

    for exit_time, scaled_net, sym, cluster in all_trades:
        cumulative += scaled_net
        equity.append(cumulative)
        net_returns.append(scaled_net)
        per_instrument_pnl[sym] += scaled_net
        cluster_pnl[cluster] = cluster_pnl.get(cluster, 0.0) + scaled_net

    # ── Compute portfolio metrics ───────────────────────────────────────
    ret = np.array(net_returns, dtype=np.float64) if net_returns else np.array([0.0])
    n_trades = len(net_returns)

    omega, z, pf, wr = _compute_metrics(ret) if n_trades > 0 else (0.0, 0.0, 0.0, 0.0)
    max_dd = _max_drawdown(equity)

    # Years = max across instruments
    max_years = max(
        (r.years for r in instrument_results.values()),
        default=0.001,
    )
    max_years = max(max_years, 0.001)

    tpy = n_trades / max_years

    # Calmar ratio
    calmar = 0.0
    if max_dd < 0 and max_years > 0:
        annual_pnl = float(ret.sum()) / max_years
        calmar = annual_pnl / abs(max_dd)

    return PortfolioBacktestResult(
        instruments=list(instrument_results.keys()),
        instrument_results=instrument_results,
        allocation_weights=allocation_weights,
        equity_curve=equity,
        total_trades=n_trades,
        omega=omega,
        z_factor=z,
        profit_factor=pf,
        win_rate=wr,
        net_pnl_usd=float(ret.sum()),
        max_dd_usd=max_dd,
        calmar_ratio=calmar,
        years=max_years,
        trades_per_year=tpy,
        per_instrument_contribution=per_instrument_pnl,
        cluster_contribution=cluster_pnl,
    )


def _empty_portfolio_result() -> PortfolioBacktestResult:
    """Return an empty portfolio result."""
    return PortfolioBacktestResult(
        instruments=[],
        instrument_results={},
        allocation_weights={},
        equity_curve=[0.0],
        total_trades=0,
        omega=0.0,
        z_factor=0.0,
        profit_factor=0.0,
        win_rate=0.0,
        net_pnl_usd=0.0,
        max_dd_usd=0.0,
        calmar_ratio=0.0,
        years=0.0,
        trades_per_year=0.0,
    )


# ══════════════════════════════════════════════════════════════════════════════
# Walk-Forward Validation
# ══════════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True, slots=True)
class WalkForwardResult:
    """In-sample vs out-of-sample comparison for a single instrument."""

    symbol: str
    is_trades: int
    is_omega: float
    is_z: float
    oos_trades: int
    oos_omega: float
    oos_z: float
    oos_degradation_omega: float  # (oos - is) / is
    oos_degradation_z: float  # (oos - is) / is
    oos_passed: bool  # meets minimum OOS criteria


#: Default OOS pass criteria
OOS_MIN_OMEGA: float = 1.5
OOS_MIN_Z: float = 1.0
OOS_MIN_TRADES: int = 5

#: Default IS/OOS split ratio
IS_FRACTION: float = 0.70


def walk_forward_instrument(
    symbol: str,
    closes: pd.Series,
    brick_size: float,
    filter_params: Optional[FilterParams] = None,
    stop_params: Optional[StopParams] = None,
    calc=None,
    *,
    is_fraction: float = IS_FRACTION,
    oos_min_omega: float = OOS_MIN_OMEGA,
    oos_min_z: float = OOS_MIN_Z,
    oos_min_trades: int = OOS_MIN_TRADES,
    session_break_minutes: float = 30.0,
) -> Optional[WalkForwardResult]:
    """
    Run IS/OOS walk-forward validation for one instrument.

    Splits data at ``is_fraction`` (default 70%), runs identical backtests
    on both halves, and compares degradation.

    Parameters
    ----------
    symbol : str
        Instrument symbol.
    closes : pd.Series
        Close prices indexed by datetime.
    brick_size : float
        Brick size in price units.
    filter_params : FilterParams or None
        Filter configuration.
    stop_params : StopParams or None
        Stop/exit configuration.
    calc : FrictionCalculator or None
        Friction calculator.
    is_fraction : float
        Fraction of data for in-sample (0–1).
    oos_min_omega : float
        Minimum OOS Omega to pass.
    oos_min_z : float
        Minimum OOS Z-factor to pass.
    oos_min_trades : int
        Minimum OOS trade count to pass.

    Returns
    -------
    WalkForwardResult or None
        None if insufficient data for a meaningful split.
    """
    if len(closes) < 100:
        return None

    split_idx = int(len(closes) * is_fraction)
    if split_idx < 50 or split_idx >= len(closes) - 50:
        return None

    is_closes = closes.iloc[:split_idx]
    oos_closes = closes.iloc[split_idx:]

    # Run IS backtest
    is_result = backtest_instrument(
        symbol=symbol,
        closes=is_closes,
        brick_size=brick_size,
        filter_params=filter_params,
        stop_params=stop_params,
        calc=calc,
        session_break_minutes=session_break_minutes,
    )

    # Run OOS backtest with SAME parameters (no re-optimisation)
    oos_result = backtest_instrument(
        symbol=symbol,
        closes=oos_closes,
        brick_size=brick_size,
        filter_params=filter_params,
        stop_params=stop_params,
        calc=calc,
        session_break_minutes=session_break_minutes,
    )

    # Compute degradation
    deg_omega = 0.0
    if is_result.omega > 0:
        deg_omega = (oos_result.omega - is_result.omega) / is_result.omega

    deg_z = 0.0
    if is_result.z_factor > 0:
        deg_z = (oos_result.z_factor - is_result.z_factor) / is_result.z_factor

    oos_passed = (
        oos_result.omega >= oos_min_omega
        and oos_result.z_factor >= oos_min_z
        and len(oos_result.trades) >= oos_min_trades
    )

    return WalkForwardResult(
        symbol=symbol,
        is_trades=len(is_result.trades),
        is_omega=is_result.omega,
        is_z=is_result.z_factor,
        oos_trades=len(oos_result.trades),
        oos_omega=oos_result.omega,
        oos_z=oos_result.z_factor,
        oos_degradation_omega=deg_omega,
        oos_degradation_z=deg_z,
        oos_passed=oos_passed,
    )


# ══════════════════════════════════════════════════════════════════════════════
# Monte Carlo Validation
# ══════════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True, slots=True)
class MonteCarloResult:
    """Result of Monte Carlo parameter perturbation test.

    Attributes
    ----------
    n_runs : int
        Number of MC runs performed.
    omegas : np.ndarray
        Omega ratio from each run.
    z_factors : np.ndarray
        Z-factor from each run.
    max_dds : np.ndarray
        Max drawdown from each run.
    trade_counts : np.ndarray
        Number of trades from each run.
    median_omega : float
        Median Omega across runs.
    p5_omega : float
        5th percentile Omega (worst-case).
    median_z : float
        Median Z-factor across runs.
    p5_z : float
        5th percentile Z-factor.
    passed : bool
        True if median_omega > 2.0 and p5_omega > 1.0.
    """

    n_runs: int
    omegas: np.ndarray
    z_factors: np.ndarray
    max_dds: np.ndarray
    trade_counts: np.ndarray
    median_omega: float
    p5_omega: float
    median_z: float
    p5_z: float
    passed: bool


# ── Default parallelism ─────────────────────────────────────────────────
# Cap at physical cores to avoid memory pressure from large pd.Series
# copies across workers.  Set KINETRA_BACKTEST_JOBS env var to override.
_DEFAULT_N_JOBS: int = min(os.cpu_count() or 1, 8)


def _mc_single_run(
    run_seed: int,
    symbol: str,
    closes: pd.Series,
    brick_size: float,
    filter_params: FilterParams,
    stop_params: StopParams,
    calc,
    brick_jitter_pct: float,
    filter_jitter_pct: float,
    session_break_minutes: float,
) -> Tuple[float, float, float, int]:
    """Execute one Monte Carlo run — top-level function for joblib pickling."""
    rng = np.random.default_rng(run_seed)

    brick_mult = 1.0 + rng.uniform(-brick_jitter_pct, brick_jitter_pct)
    jittered_brick = max(brick_size * brick_mult, 1e-10)

    fr_t_mult = 1.0 + rng.uniform(-filter_jitter_pct, filter_jitter_pct)
    mk_t_mult = 1.0 + rng.uniform(-filter_jitter_pct, filter_jitter_pct)

    jittered_fp = FilterParams(
        fliprate_window=filter_params.fliprate_window,
        fliprate_threshold=max(0.01, filter_params.fliprate_threshold * fr_t_mult),
        markov_window=filter_params.markov_window,
        markov_threshold=max(0.01, min(0.99, filter_params.markov_threshold * mk_t_mult)),
    )

    result = backtest_instrument(
        symbol=symbol,
        closes=closes,
        brick_size=jittered_brick,
        filter_params=jittered_fp,
        stop_params=stop_params,
        calc=calc,
        session_break_minutes=session_break_minutes,
    )

    return result.omega, result.z_factor, result.max_dd_usd, len(result.trades)


def monte_carlo_instrument(
    symbol: str,
    closes: pd.Series,
    brick_size: float,
    filter_params: Optional[FilterParams] = None,
    stop_params: Optional[StopParams] = None,
    calc=None,
    *,
    n_runs: int = 100,
    brick_jitter_pct: float = 0.10,
    filter_jitter_pct: float = 0.15,
    seed: Optional[int] = None,
    session_break_minutes: float = 30.0,
    n_jobs: Optional[int] = None,
) -> MonteCarloResult:
    """
    Monte Carlo parameter perturbation test for one instrument.

    For each run, jitters brick size and filter thresholds by random
    amounts within the specified ranges, then runs a full instrument
    backtest.  Runs are **embarrassingly parallel** and are dispatched
    across CPU cores via ``joblib`` for ~N× speedup where N = core count.

    Parameters
    ----------
    symbol : str
        Instrument symbol.
    closes : pd.Series
        Close prices indexed by datetime.
    brick_size : float
        Nominal brick size.
    filter_params : FilterParams or None
        Base filter configuration (will be perturbed).
    stop_params : StopParams or None
        Stop/exit configuration (NOT perturbed).
    calc : FrictionCalculator or None
        Friction calculator.
    n_runs : int
        Number of Monte Carlo runs.
    brick_jitter_pct : float
        Max ± perturbation on brick_size as a fraction.
    filter_jitter_pct : float
        Max ± perturbation on filter thresholds as a fraction.
    seed : int or None
        Random seed for reproducibility.
    session_break_minutes : float
        Session break threshold passed to ``build_renko``.
    n_jobs : int or None
        Number of parallel workers.  ``None`` uses ``_DEFAULT_N_JOBS``
        (capped at 8 or ``os.cpu_count()``).  ``1`` forces serial
        execution.  Can also be set via ``KINETRA_BACKTEST_JOBS`` env var.

    Returns
    -------
    MonteCarloResult
        Summary statistics across all runs.
    """
    if filter_params is None:
        filter_params = FilterParams()
    if stop_params is None:
        stop_params = StopParams()

    # Resolve worker count
    if n_jobs is None:
        n_jobs = int(os.environ.get("KINETRA_BACKTEST_JOBS", str(_DEFAULT_N_JOBS)))
    n_jobs = max(1, n_jobs)

    # Pre-generate per-run seeds from the master seed so results are
    # reproducible regardless of parallelism.
    master_rng = np.random.default_rng(seed)
    run_seeds = master_rng.integers(0, 2**31, size=n_runs).tolist()

    # ── Parallel execution ──────────────────────────────────────────────
    if n_jobs > 1 and n_runs > 1:
        results = Parallel(n_jobs=n_jobs, backend="loky")(
            delayed(_mc_single_run)(
                run_seed=run_seeds[run],
                symbol=symbol,
                closes=closes,
                brick_size=brick_size,
                filter_params=filter_params,
                stop_params=stop_params,
                calc=calc,
                brick_jitter_pct=brick_jitter_pct,
                filter_jitter_pct=filter_jitter_pct,
                session_break_minutes=session_break_minutes,
            )
            for run in range(n_runs)
        )
    else:
        # Serial fallback (n_jobs=1 or single run)
        results = [
            _mc_single_run(
                run_seed=run_seeds[run],
                symbol=symbol,
                closes=closes,
                brick_size=brick_size,
                filter_params=filter_params,
                stop_params=stop_params,
                calc=calc,
                brick_jitter_pct=brick_jitter_pct,
                filter_jitter_pct=filter_jitter_pct,
                session_break_minutes=session_break_minutes,
            )
            for run in range(n_runs)
        ]

    # ── Collect results ─────────────────────────────────────────────────
    omegas = np.array([r[0] for r in results], dtype=np.float64)
    z_factors = np.array([r[1] for r in results], dtype=np.float64)
    max_dds = np.array([r[2] for r in results], dtype=np.float64)
    trade_counts = np.array([r[3] for r in results], dtype=np.int32)

    med_omega = float(np.median(omegas))
    p5_omega = float(np.percentile(omegas, 5))
    med_z = float(np.median(z_factors))
    p5_z = float(np.percentile(z_factors, 5))
    passed = med_omega > 2.0 and p5_omega > 1.0

    return MonteCarloResult(
        n_runs=n_runs,
        omegas=omegas,
        z_factors=z_factors,
        max_dds=max_dds,
        trade_counts=trade_counts,
        median_omega=med_omega,
        p5_omega=p5_omega,
        median_z=med_z,
        p5_z=p5_z,
        passed=passed,
    )


# ══════════════════════════════════════════════════════════════════════════════
# Stress Testing
# ══════════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True, slots=True)
class FrictionStressResult:
    """Result of scaling friction costs by a multiplier."""

    symbol: str
    friction_multiplier: float
    original_omega: float
    stressed_omega: float
    original_net_pnl: float
    stressed_net_pnl: float
    still_profitable: bool
    still_omega_above_one: bool


def stress_test_friction(
    result: InstrumentBacktestResult,
    multiplier: float = 1.5,
) -> FrictionStressResult:
    """
    Test sensitivity of results to friction cost scaling.

    Scales all friction costs by ``multiplier`` and recomputes metrics.
    Does NOT re-run the backtest — just adjusts the trade P&L.

    Parameters
    ----------
    result : InstrumentBacktestResult
        Original backtest result.
    multiplier : float
        Factor to multiply friction costs by (e.g. 1.5 = 50% higher costs).

    Returns
    -------
    FrictionStressResult
        Comparison of original vs stressed metrics.
    """
    from kinetra.backtesting.metrics import omega_ratio

    if not result.trades:
        return FrictionStressResult(
            symbol=result.symbol,
            friction_multiplier=multiplier,
            original_omega=result.omega,
            stressed_omega=0.0,
            original_net_pnl=0.0,
            stressed_net_pnl=0.0,
            still_profitable=False,
            still_omega_above_one=False,
        )

    # Recompute net P&L with scaled friction
    stressed_nets = []
    for t in result.trades:
        extra_friction = t.friction_usd * (multiplier - 1.0)
        stressed_net = t.net_usd - extra_friction
        stressed_nets.append(stressed_net)

    stressed_arr = np.array(stressed_nets, dtype=np.float64)
    stressed_omega = float(omega_ratio(stressed_arr))
    stressed_pnl = float(stressed_arr.sum())

    original_pnl = float(sum(t.net_usd for t in result.trades))

    return FrictionStressResult(
        symbol=result.symbol,
        friction_multiplier=multiplier,
        original_omega=result.omega,
        stressed_omega=stressed_omega,
        original_net_pnl=original_pnl,
        stressed_net_pnl=stressed_pnl,
        still_profitable=stressed_pnl > 0,
        still_omega_above_one=stressed_omega > 1.0,
    )


# ══════════════════════════════════════════════════════════════════════════════
# Brick Sweep
# ══════════════════════════════════════════════════════════════════════════════


@dataclass(slots=True)
class SweepPoint:
    """Result of backtesting at one brick-size multiplier."""

    multiplier: float
    brick_size: float
    friction_ratio: float
    n_trades: int
    trades_per_year: float
    omega: float
    z_factor: float
    profit_factor: float
    win_rate: float
    net_pnl_usd: float
    max_dd_usd: float
    total_friction_usd: float
    avg_net_per_trade: float
    passed_floor: bool
    passed_quality: bool


@dataclass(slots=True)
class BrickSweepResult:
    """Complete sweep across multiple brick sizes for one instrument."""

    symbol: str
    dsp_brick_size: float
    sweep_points: List[SweepPoint]
    best_point: Optional[SweepPoint]
    best_result: Optional[InstrumentBacktestResult]
    qualified: bool  # at least one point passed all quality gates


#: Default sweep multipliers around the DSP-suggested brick size
DEFAULT_SWEEP_MULTIPLIERS: Tuple[float, ...] = (0.5, 0.7, 1.0, 1.3, 1.5, 2.0, 2.5, 3.0)

#: Default quality gate thresholds
DEFAULT_MIN_TRADES: int = 15
DEFAULT_MIN_Z: float = 2.0
DEFAULT_MAX_FRICTION_RATIO: float = 0.25


def _sweep_single_multiplier(
    mult: float,
    symbol: str,
    closes: pd.Series,
    dsp_brick_size: float,
    spread_price: float,
    friction_floor: float,
    filter_params: Optional[FilterParams],
    stop_params: Optional[StopParams],
    calc,
    min_trades: int,
    min_z: float,
    max_friction_ratio: float,
    session_break_minutes: float,
) -> Tuple[SweepPoint, Optional[InstrumentBacktestResult]]:
    """Evaluate one brick-size multiplier — top-level for joblib pickling."""
    candidate_brick = dsp_brick_size * mult

    passed_floor = candidate_brick >= friction_floor
    fr_ratio = (spread_price / candidate_brick) if candidate_brick > 0 else 999.0

    result = backtest_instrument(
        symbol=symbol,
        closes=closes,
        brick_size=candidate_brick,
        filter_params=filter_params,
        stop_params=stop_params,
        calc=calc,
        session_break_minutes=session_break_minutes,
    )

    n = len(result.trades)
    passed_quality = (
        passed_floor
        and n >= min_trades
        and result.z_factor >= min_z
        and fr_ratio <= max_friction_ratio
        and result.omega > 1.0
    )

    sp = SweepPoint(
        multiplier=mult,
        brick_size=candidate_brick,
        friction_ratio=fr_ratio,
        n_trades=n,
        trades_per_year=result.trades_per_year,
        omega=result.omega,
        z_factor=result.z_factor,
        profit_factor=result.profit_factor,
        win_rate=result.win_rate,
        net_pnl_usd=sum(t.net_usd for t in result.trades),
        max_dd_usd=result.max_dd_usd,
        total_friction_usd=result.total_friction_usd,
        avg_net_per_trade=result.avg_net_per_trade,
        passed_floor=passed_floor,
        passed_quality=passed_quality,
    )

    return sp, result if passed_quality else None


def sweep_brick_sizes(
    symbol: str,
    closes: pd.Series,
    dsp_brick_size: float,
    spread_price: float,
    friction_floor: float,
    *,
    multipliers: Sequence[float] = DEFAULT_SWEEP_MULTIPLIERS,
    filter_params: Optional[FilterParams] = None,
    stop_params: Optional[StopParams] = None,
    calc=None,
    min_trades: int = DEFAULT_MIN_TRADES,
    min_z: float = DEFAULT_MIN_Z,
    max_friction_ratio: float = DEFAULT_MAX_FRICTION_RATIO,
    session_break_minutes: float = 30.0,
    n_jobs: Optional[int] = None,
) -> BrickSweepResult:
    """
    Sweep multiple brick sizes for one instrument and select the best.

    For each candidate brick (DSP suggestion × multiplier), runs a full
    instrument backtest and evaluates quality gates:
      - Brick ≥ friction floor
      - Z-factor ≥ min_z
      - Trades ≥ min_trades
      - Friction ratio ≤ max_friction_ratio
      - Omega > 1.0

    Among qualifying points, selects the one with highest Omega.

    Backtests for each multiplier are independent and are dispatched in
    parallel via ``joblib`` for ~N× speedup where N = core count.

    Parameters
    ----------
    symbol : str
        Instrument symbol.
    closes : pd.Series
        Close prices indexed by datetime.
    dsp_brick_size : float
        DSP-suggested brick size (from VR peak scale).
    spread_price : float
        Spread in price units (for friction ratio computation).
    friction_floor : float
        Minimum brick size from friction floor computation.
    multipliers : sequence of float
        Multipliers to apply to ``dsp_brick_size``.
    filter_params : FilterParams or None
        Filter configuration.
    stop_params : StopParams or None
        Stop/exit configuration.
    calc : FrictionCalculator or None
        Friction calculator.
    min_trades : int
        Minimum trade count for quality gate.
    min_z : float
        Minimum Z-factor for quality gate.
    max_friction_ratio : float
        Maximum friction ratio (spread/brick) for quality gate.
    session_break_minutes : float
        Session break threshold passed to ``build_renko``.
    n_jobs : int or None
        Number of parallel workers.  ``None`` uses ``_DEFAULT_N_JOBS``.
        ``1`` forces serial execution.

    Returns
    -------
    BrickSweepResult
        Sweep results with best qualifying point.
    """
    # Filter out non-positive candidates early
    valid_mults = [m for m in multipliers if dsp_brick_size * m > 0]
    if not valid_mults:
        return BrickSweepResult(
            symbol=symbol,
            dsp_brick_size=dsp_brick_size,
            sweep_points=[],
            best_point=None,
            best_result=None,
            qualified=False,
        )

    # Resolve worker count
    if n_jobs is None:
        n_jobs = int(os.environ.get("KINETRA_BACKTEST_JOBS", str(_DEFAULT_N_JOBS)))
    n_jobs = max(1, n_jobs)

    # ── Parallel execution ──────────────────────────────────────────────
    if n_jobs > 1 and len(valid_mults) > 1:
        raw_results = Parallel(n_jobs=n_jobs, backend="loky")(
            delayed(_sweep_single_multiplier)(
                mult=mult,
                symbol=symbol,
                closes=closes,
                dsp_brick_size=dsp_brick_size,
                spread_price=spread_price,
                friction_floor=friction_floor,
                filter_params=filter_params,
                stop_params=stop_params,
                calc=calc,
                min_trades=min_trades,
                min_z=min_z,
                max_friction_ratio=max_friction_ratio,
                session_break_minutes=session_break_minutes,
            )
            for mult in valid_mults
        )
    else:
        raw_results = [
            _sweep_single_multiplier(
                mult=mult,
                symbol=symbol,
                closes=closes,
                dsp_brick_size=dsp_brick_size,
                spread_price=spread_price,
                friction_floor=friction_floor,
                filter_params=filter_params,
                stop_params=stop_params,
                calc=calc,
                min_trades=min_trades,
                min_z=min_z,
                max_friction_ratio=max_friction_ratio,
                session_break_minutes=session_break_minutes,
            )
            for mult in valid_mults
        ]

    # ── Collect and select best ─────────────────────────────────────────
    sweep_points: List[SweepPoint] = []
    best_point: Optional[SweepPoint] = None
    best_result: Optional[InstrumentBacktestResult] = None

    for sp, result in raw_results:
        sweep_points.append(sp)
        if sp.passed_quality:
            if best_point is None or sp.omega > best_point.omega:
                best_point = sp
                best_result = result

    # If nothing qualified, pick the best-Z that passed floor as a fallback
    if best_point is None and sweep_points:
        floor_passed = [p for p in sweep_points if p.passed_floor and p.n_trades > 0]
        if floor_passed:
            best_point = max(floor_passed, key=lambda p: p.z_factor)

    qualified = best_point is not None and best_point.passed_quality

    return BrickSweepResult(
        symbol=symbol,
        dsp_brick_size=dsp_brick_size,
        sweep_points=sweep_points,
        best_point=best_point,
        best_result=best_result,
        qualified=qualified,
    )
