"""
Renko Live Trader
=================

Sprint 6 — Renko-native paper trading and live execution controller.

Wires the full Renko three-layer runtime into an incremental bar-by-bar
execution loop:

  M1 bar feed
    └─► incremental brick construction   (build_renko_incremental)
          └─► Layer 1: flip + filter     (evaluate_entry / FilterParams)
                └─► Layer 2: allocation  (AllocationAgent.act)
                      └─► Layer 3: risk  (RiskAgent.act)
                            └─► sizing   (RenkoSizer)
                                  └─► OrderDispatcher (paper / live)

Design decisions
----------------
- **Incremental bricks** — new M1 bars are appended to a rolling price
  buffer; bricks are emitted only when the brick threshold is crossed.
  No full rebuild on each tick.
- **Session-break awareness** — session_break_minutes is read from the
  per-instrument SessionProfile (via QualificationRegistry) so no bricks
  are emitted across the daily rollover gap.
- **Capital gates (PER)** — three progressive live gates:
    Gate 0 (paper)   — all logic active, no real orders
    Gate 1 (micro)   — 0.01 lots, max DD 3 %, max 2 instruments
    Gate 2 (small)   — 0.1  lots, max DD 5 %, max 5 instruments
    Gate 3 (full)    — config-specified lots, max DD 10 %, no cap
  Advancement from paper → Gate 1 requires a minimum paper PER score.
- **Execution-aware sizing** — lot size = (target_risk_usd / (brick_size
  × usd_per_point)) scaled by Layer 2 allocation weight and Layer 3
  exposure scalar, clamped by the active PER gate's lot ceiling.
- **Broker neutrality** — OrderDispatcher is an ABC; PaperDispatcher and
  a broker-specific concrete class (MetaAPIDispatcher, cTraderDispatcher)
  implement it without any logic leaking into this module.

Hard rules (§29 AGENT_RULES_MASTER.md)
---------------------------------------
- ❌ Never call build_renko() on the full history on every tick —
  use RenkoBrickBuffer for O(1) incremental construction.
- ❌ Never re-read qualification.json on every bar — load once at startup.
- ❌ Never use RL to calibrate brick size or filter thresholds — DSP only.
- ✅ Always pass session_break_minutes from SessionProfile to build_renko.
- ✅ Always run CircuitBreakerManager.evaluate() before executing orders.
- ✅ Always record all decisions and outcomes to the trade log.

Usage::

    from kinetra.renko.live_trader import RenkoLiveTrader, LiveTraderConfig

    config = LiveTraderConfig(
        symbols=["XAUUSD", "NAS100"],
        gate=PERGate.SIMULATED,
        target_risk_usd=50.0,
    )
    trader = RenkoLiveTrader(config)
    trader.start()        # blocking loop — Ctrl-C to stop
    trader.stop()

See Also:
    - ``kinetra/renko/qualify.py``          — QualificationRegistry, SessionProfile
    - ``kinetra/renko/brick_engine.py``     — build_renko
    - ``kinetra/renko/filters.py``          — flip_rate, markov_stickiness, evaluate_entry
    - ``kinetra/monitoring/circuit_breakers.py`` — CircuitBreakerManager
    - ``docs/MANUAL.md §7`` — runtime architecture
"""

from __future__ import annotations

import json
import logging
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from kinetra.config import PROJECT_ROOT
from kinetra.monitoring import emit_event, emit_health
from kinetra.renko.backtest import SizingMode
from kinetra.renko.brick_engine import build_renko
from kinetra.renko.filters import evaluate_entry, flip_rate, markov_stickiness
from kinetra.renko.vpin import vpin_excess_kurtosis

if TYPE_CHECKING:
    from kinetra.monitoring.circuit_breakers import CircuitBreakerManager
    from kinetra.renko.qualify import QualificationResult

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════════

# Directory where live trade logs are written
LIVE_LOG_DIR: Path = PROJECT_ROOT / "results" / "renko" / "live"

# Directory where the qualification registry lives
DEFAULT_QUAL_DIR: Path = PROJECT_ROOT / "data" / "renko_qualified"

# Minimum paper-trade performance score to advance through a PER gate
PER_MIN_SCORE_GATE1: float = 1.5  # paper Omega >= 1.5 to unlock micro lots
PER_MIN_SCORE_GATE2: float = 2.0  # micro Omega >= 2.0 to unlock small lots
PER_MIN_SCORE_GATE3: float = 2.5  # small Omega >= 2.5 to unlock full lots

# PER gate lot ceilings
PER_LOT_CEILING: Dict[str, float] = {
    "paper": 0.0,  # no real orders
    "micro": 0.01,
    "small": 0.10,
    "full": 999.0,  # only broker margin limits apply
}

# PER gate max drawdown limits (fraction of equity)
PER_MAX_DD: Dict[str, float] = {
    "paper": 1.0,  # no real money, no hard limit
    "micro": 0.03,
    "small": 0.05,
    "full": 0.10,
}

# Maximum instruments tradeable per gate
PER_MAX_INSTRUMENTS: Dict[str, int] = {
    "paper": 999,
    "micro": 2,
    "small": 5,
    "full": 999,
}

# Minimum paper trades required before PER evaluation is meaningful
PER_MIN_PAPER_TRADES: int = 30

# Brick buffer: keep this many M1 bars in the rolling window
# ~2 trading weeks of M1 = 14 × 24 × 60 ≈ 20 160 bars
BRICK_BUFFER_BARS: int = 21_000
HEARTBEAT_INTERVAL_S: float = 60.0
NUMERIC_EPS: float = 1e-12


def _is_finite_number(x: Any) -> bool:
    try:
        return bool(np.isfinite(float(x)))
    except Exception:
        return False


def _safe_float(x: Any, default: float) -> float:
    try:
        v = float(x)
    except Exception:
        return float(default)
    return v if np.isfinite(v) else float(default)


def _safe_positive(x: Any, default: float, floor: float = NUMERIC_EPS) -> float:
    v = _safe_float(x, default)
    if v <= floor:
        return max(float(default), floor)
    return v


def _safe_div(numer: float, denom: float, default: float = 0.0) -> float:
    n = _safe_float(numer, 0.0)
    d = _safe_float(denom, 0.0)
    if abs(d) <= NUMERIC_EPS:
        return float(default)
    out = n / d
    return out if np.isfinite(out) else float(default)


def _infer_price_digits_from_tick_size(tick_size: float, fallback: int = 5) -> int:
    t = _safe_positive(tick_size, 0.00001)
    if t >= 1.0:
        return 0
    for d in range(1, 10):
        scaled = t * (10**d)
        if abs(scaled - round(scaled)) <= 1e-9:
            return d
    return fallback


def _quantize_price(price: float, tick_size: float, digits: int) -> float:
    p = _safe_float(price, 0.0)
    t = _safe_positive(tick_size, 0.00001)
    q = round(p / t) * t
    return round(q, max(int(digits), 0))


def _quantize_lots(lots: float, lot_step: float, min_lots: float) -> float:
    l = max(0.0, _safe_float(lots, 0.0))
    s = _safe_positive(lot_step, 0.01)
    m = max(0.0, _safe_float(min_lots, 0.01))
    q = round(l / s) * s
    q = round(q, 8)
    if q < m:
        return 0.0
    return q


def _compute_streak_stats(pnls: List[float]) -> Dict[str, Any]:
    """Compute consecutive win/loss streak metrics from trade PnL series."""
    max_win = 0
    max_loss = 0
    cur_win = 0
    cur_loss = 0
    current_type = "flat"
    current_len = 0

    for p in pnls:
        v = _safe_float(p, 0.0)
        if v > 0:
            cur_win += 1
            cur_loss = 0
            max_win = max(max_win, cur_win)
            current_type = "win"
            current_len = cur_win
        elif v < 0:
            cur_loss += 1
            cur_win = 0
            max_loss = max(max_loss, cur_loss)
            current_type = "loss"
            current_len = cur_loss
        else:
            cur_win = 0
            cur_loss = 0
            current_type = "flat"
            current_len = 0

    return {
        "max_consecutive_wins": int(max_win),
        "max_consecutive_losses": int(max_loss),
        "current_streak_type": current_type,
        "current_streak_length": int(current_len),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Enums & simple value types
# ═══════════════════════════════════════════════════════════════════════════════


class PERGate(str, Enum):
    """Progressive Exposure Ramp gate — capital deployment level.

    Stages:
        SIMULATED: Backtesting / dry-run (no real money)
        MICRO:     Live trading with micro lots
        SMALL:     Live trading with reduced size
        FULL:      Full production size
    """

    SIMULATED = "simulated"
    MICRO = "micro"
    SMALL = "small"
    FULL = "full"

    @property
    def lot_ceiling(self) -> float:
        return PER_LOT_CEILING[self.value]

    @property
    def max_dd(self) -> float:
        return PER_MAX_DD[self.value]

    @property
    def max_instruments(self) -> int:
        return PER_MAX_INSTRUMENTS[self.value]

    @property
    def next_gate(self) -> Optional["PERGate"]:
        order = [PERGate.SIMULATED, PERGate.MICRO, PERGate.SMALL, PERGate.FULL]
        idx = order.index(self)
        return order[idx + 1] if idx < len(order) - 1 else None

    @property
    def required_score(self) -> float:
        return {
            PERGate.SIMULATED: 0.0,
            PERGate.MICRO: PER_MIN_SCORE_GATE1,
            PERGate.SMALL: PER_MIN_SCORE_GATE2,
            PERGate.FULL: PER_MIN_SCORE_GATE3,
        }[self]


class TradeDirection(int, Enum):
    LONG = 1
    SHORT = -1


# ═══════════════════════════════════════════════════════════════════════════════
# Data containers
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class LiveTrade:
    """A live or paper trade record."""

    trade_id: str
    symbol: str
    direction: TradeDirection
    entry_price: float
    entry_time: datetime
    brick_size: float
    lots: float
    target_risk_usd: float
    gate: PERGate
    signal_id: str = ""
    broker_ticket: str = ""
    layer2_weight: float = 1.0  # allocation agent weight [0, 1]
    layer3_exposure: float = 1.0  # risk agent scalar [0, 1]
    exit_price: Optional[float] = None
    exit_time: Optional[datetime] = None
    exit_reason: Optional[str] = None  # "colour_change" | "stop" | "circuit_breaker" | "manual"
    gross_pts: float = 0.0
    gross_usd: float = 0.0
    # Friction cost breakdown
    spread_usd: float = 0.0
    commission_usd: float = 0.0
    swap_usd: float = 0.0
    friction_usd: float = 0.0  # Total friction = spread + commission + swap
    net_usd: float = 0.0
    is_open: bool = True

    def close(
        self,
        exit_price: float,
        exit_time: datetime,
        exit_reason: str,
        friction_usd: float,
        usd_per_point: float,
        spread_usd: float = 0.0,
        commission_usd: float = 0.0,
        swap_usd: float = 0.0,
    ) -> None:
        """Record trade exit."""
        self.exit_price = exit_price
        self.exit_time = exit_time
        self.exit_reason = exit_reason
        self.spread_usd = max(0.0, _safe_float(spread_usd, 0.0))
        self.commission_usd = max(0.0, _safe_float(commission_usd, 0.0))
        self.swap_usd = max(0.0, _safe_float(swap_usd, 0.0))
        self.friction_usd = self.spread_usd + self.commission_usd + self.swap_usd
        # If legacy friction_usd is passed without breakdown, use it
        if spread_usd == 0.0 and commission_usd == 0.0 and swap_usd == 0.0:
            self.friction_usd = max(0.0, _safe_float(friction_usd, 0.0))
        self.is_open = False
        self.gross_pts = (exit_price - self.entry_price) * self.direction.value
        upp = _safe_float(usd_per_point, 0.0)
        lots = _safe_float(self.lots, 0.0)
        self.gross_usd = self.gross_pts * upp * lots
        self.net_usd = self.gross_usd - self.friction_usd
        if not _is_finite_number(self.gross_usd):
            self.gross_usd = 0.0
        if not _is_finite_number(self.net_usd):
            self.net_usd = -self.friction_usd

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trade_id": self.trade_id,
            "signal_id": self.signal_id,
            "broker_ticket": self.broker_ticket,
            "symbol": self.symbol,
            "direction": self.direction.value,
            "entry_price": self.entry_price,
            "entry_time": self.entry_time.isoformat() if self.entry_time else None,
            "exit_price": self.exit_price,
            "exit_time": self.exit_time.isoformat() if self.exit_time else None,
            "exit_reason": self.exit_reason,
            "brick_size": self.brick_size,
            "lots": self.lots,
            "target_risk_usd": self.target_risk_usd,
            "gate": self.gate.value,
            "layer2_weight": self.layer2_weight,
            "layer3_exposure": self.layer3_exposure,
            "gross_pts": self.gross_pts,
            "gross_usd": self.gross_usd,
            "spread_usd": self.spread_usd,
            "commission_usd": self.commission_usd,
            "swap_usd": self.swap_usd,
            "friction_usd": self.friction_usd,
            "net_usd": self.net_usd,
            "is_open": self.is_open,
        }


@dataclass
class InstrumentLiveState:
    """Per-instrument live runtime state."""

    symbol: str
    brick_size: float
    filter_fliprate_threshold: float
    filter_markov_threshold: float
    filter_fliprate_window: int
    filter_markov_window: int
    stop_bricks: float
    session_break_minutes: float
    usd_per_point: float
    tick_size: float
    price_digits: int
    cluster: str

    # Rolling M1 price buffer — grows up to BRICK_BUFFER_BARS
    _price_buffer: List[float] = field(default_factory=list, repr=False)
    _time_buffer: List[datetime] = field(default_factory=list, repr=False)

    # Current brick sequence (rebuilt when buffer is flushed)
    _bricks: Optional[pd.DataFrame] = field(default=None, repr=False)

    # Lock protecting _price_buffer, _time_buffer, and _bricks from concurrent
    # access when multiple threads call append_bar / get_bricks simultaneously
    # (e.g. the bar provider pushing bars while the entry loop reads bricks).
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    # Current open position (None = flat)
    open_trade: Optional[LiveTrade] = field(default=None)

    # Completed trades this session
    session_trades: List[LiveTrade] = field(default_factory=list)

    # Allocation weight from last Layer 2 decision [0, 1]
    layer2_weight: float = 1.0

    # Exposure scalar from last Layer 3 decision [0, 1]
    layer3_exposure: float = 1.0

    # Peak equity for this instrument (for DD tracking)
    peak_equity: float = 0.0
    cumulative_pnl: float = 0.0

    # Startup warmup: skip first N colour flips after startup/reconnect.
    startup_flips_seen: int = 0
    startup_warmup_done: bool = False
    consecutive_losses: int = 0
    pause_until_utc: Optional[datetime] = None
    active_stop_price: Optional[float] = None
    entry_brick_index: Optional[int] = None

    # Track the last brick index we checked for entry (prevents double-counting blockers per brick)
    last_checked_brick_idx: int = -1

    def append_bar(self, close: float, bar_time: datetime) -> None:
        """Append one M1 close to the rolling buffer.

        Thread-safe: acquires ``_lock`` so that concurrent bar callbacks
        from a push-based BarProvider cannot interleave with get_bricks().
        """
        with self._lock:
            self._price_buffer.append(close)
            self._time_buffer.append(bar_time)
            # Trim to window
            if len(self._price_buffer) > BRICK_BUFFER_BARS:
                self._price_buffer = self._price_buffer[-BRICK_BUFFER_BARS:]
                self._time_buffer = self._time_buffer[-BRICK_BUFFER_BARS:]
            # Invalidate cached bricks
            self._bricks = None

    def get_bricks(self) -> pd.DataFrame:
        """Build (or return cached) bricks from the current price buffer.

        Thread-safe: acquires ``_lock`` so that a concurrent append_bar()
        call cannot mutate the buffers while we are building the Series.
        """
        with self._lock:
            if self._bricks is not None:
                return self._bricks
            if len(self._price_buffer) < 2:
                return pd.DataFrame(
                    columns=["brick_open", "brick_close", "direction", "time", "session_break"]
                )
            # Take a snapshot of the buffers so build_renko() runs outside the lock
            prices_snap = list(self._price_buffer)
            times_snap = list(self._time_buffer)

        closes = pd.Series(
            prices_snap,
            index=pd.DatetimeIndex(times_snap, tz="UTC"),
        )
        bricks = build_renko(
            closes,
            brick_size=self.brick_size,
            session_break_minutes=self.session_break_minutes,
        )
        with self._lock:
            # Only cache if the buffer has not been updated while we were building
            if self._bricks is None:
                self._bricks = bricks
        return bricks

    @property
    def current_drawdown(self) -> float:
        """Current drawdown as a negative fraction (0 = no drawdown)."""
        if self.peak_equity <= 0:
            return 0.0
        return (self.cumulative_pnl - self.peak_equity) / self.peak_equity

    def record_closed_trade(self, trade: LiveTrade) -> None:
        """Record a closed trade and update equity tracking."""
        self.session_trades.append(trade)
        self.cumulative_pnl += trade.net_usd
        self.peak_equity = max(self.peak_equity, self.cumulative_pnl)


# ═══════════════════════════════════════════════════════════════════════════════
# Sizing
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True, slots=True)
class RenkoSizer:
    """
    Compute lot size for one Renko trade.

    lot_size = (target_risk_usd / (brick_size × usd_per_point))
               × layer2_weight
               × layer3_exposure

    The result is then rounded to the nearest valid lot step and clamped to
    the PER gate's ceiling.  A result of 0.0 means "skip this entry" (e.g.
    when Layer 2 or 3 sets weight/exposure to zero for this instrument).

    This sizer implements ``SizingMode.FIXED_LOT``-style 1R risk targeting.
    For volatility-targeted sizing, see :class:`VolTargetSizerLive`.
    """

    target_risk_usd: float = 100.0
    lot_step: float = 0.01
    min_lots: float = 0.01

    def compute(
        self,
        brick_size: float,
        usd_per_point: float,
        layer2_weight: float,
        layer3_exposure: float,
        gate: PERGate,
    ) -> float:
        """
        Return the lot size to trade, or 0.0 to skip.

        Parameters
        ----------
        brick_size : float
            Brick size in price units (1R stop distance).
        usd_per_point : float
            USD per 1 price unit per lot (contract value).
        layer2_weight : float
            Allocation weight from Layer 2 RL agent [0, 1].
        layer3_exposure : float
            Exposure scalar from Layer 3 RL agent [0, 1].
        gate : PERGate
            Active PER gate (determines lot ceiling).
        """
        if brick_size <= 0 or usd_per_point <= 0:
            return 0.0
        if layer2_weight < 1e-6 or layer3_exposure < 1e-6:
            return 0.0
        if gate == PERGate.SIMULATED:
            return 0.0  # paper mode — no real lots

        # 1R = stop_bricks (default 0.5 live) × brick_size × usd_per_point × lots
        # We size for target_risk_usd at 1 brick stop
        value_per_lot = brick_size * usd_per_point
        if value_per_lot <= 0:
            return 0.0

        base_lots = self.target_risk_usd / value_per_lot
        scaled = base_lots * layer2_weight * layer3_exposure

        # Round to lot step
        stepped = max(0.0, round(scaled / self.lot_step) * self.lot_step)

        # Clamp to gate ceiling
        ceiling = gate.lot_ceiling
        clamped = min(stepped, ceiling)

        if clamped < self.min_lots:
            return 0.0
        return clamped


class VolTargetSizerLive:
    """
    Thin wrapper around :class:`~kinetra.renko.vol_sizer.VolTargetSizer`
    for use inside :class:`RenkoLiveTrader`.

    Adapts the vol sizer interface to the live trader's equity tracking
    and gate-ceiling conventions so that ``RenkoLiveTrader`` can swap
    between ``SizingMode.FIXED_LOT`` / ``FIXED_RISK`` and
    ``SizingMode.VOL_TARGET`` without changing any call sites.

    Parameters
    ----------
    config : VolSizingConfig
        Configuration forwarded to the underlying
        :class:`~kinetra.renko.vol_sizer.VolTargetSizer`.
    lot_step : float
        Broker minimum lot increment.
    min_lots : float
        Broker minimum lot size.
    """

    def __init__(
        self,
        config: Any,
        lot_step: float = 0.01,
        min_lots: float = 0.01,
    ) -> None:
        from kinetra.renko.vol_sizer import VolTargetSizer

        self._sizer = VolTargetSizer(config)
        self._lot_step = lot_step
        self._min_lots = min_lots

    def update(self, symbol: str, brick_pts: float) -> None:
        """Register one completed brick P&L (price points, signed)."""
        self._sizer.update(symbol, brick_pts)

    def compute(
        self,
        symbol: str,
        equity_usd: float,
        brick_size: float,
        usd_per_point: float,
        layer2_weight: float,
        layer3_exposure: float,
        gate: PERGate,
    ) -> float:
        """
        Return the vol-targeted lot size, or 0.0 to skip.

        Parameters
        ----------
        symbol : str
            Instrument symbol.
        equity_usd : float
            Current session equity (initial + cumulative P&L).
        brick_size : float
            Instrument brick size (price units).
        usd_per_point : float
            USD per 1 price unit per lot.
        layer2_weight : float
            Layer 2 allocation weight [0, 1].
        layer3_exposure : float
            Layer 3 exposure scalar [0, 1].
        gate : PERGate
            Active PER gate (determines lot ceiling).
        """
        if gate == PERGate.SIMULATED:
            return 0.0
        return self._sizer.compute(
            symbol=symbol,
            equity_usd=equity_usd,
            brick_size=brick_size,
            usd_per_point=usd_per_point,
            layer2_weight=layer2_weight,
            layer3_exposure=layer3_exposure,
            lot_step=self._lot_step,
            min_lots=self._min_lots,
            gate_lot_ceiling=gate.lot_ceiling,
        )

    def vol_estimate(self, symbol: str, brick_size: float) -> float:
        """Current vol estimate for *symbol* (fraction of brick_size)."""
        return self._sizer.vol_estimate(symbol, brick_size)

    def is_warmed_up(self, symbol: str) -> bool:
        """True once enough bricks have been observed for *symbol*."""
        return self._sizer.is_warmed_up(symbol)

    def n_observations(self, symbol: str) -> int:
        """Number of bricks in the rolling buffer for *symbol*."""
        return self._sizer.n_observations(symbol)


# ═══════════════════════════════════════════════════════════════════════════════
# Order dispatcher (broker-neutral ABC)
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class OrderResult:
    """Result of a submitted order."""

    success: bool
    order_id: Optional[str] = None
    filled_price: Optional[float] = None
    filled_lots: Optional[float] = None
    error: Optional[str] = None
    raw: Optional[Dict[str, Any]] = None


class OrderDispatcher(ABC):
    """
    Broker-neutral interface for submitting and closing orders.

    Implementations
    ---------------
    - :class:`PaperDispatcher`     — no-op, tracks positions in memory
    - ``MetaAPIDispatcher``        — submits to MetaAPI cloud broker
    - ``cTraderDispatcher``        — submits via cTrader Open API

    All dispatchers must honour the PER gate lot ceiling enforced by
    :class:`RenkoSizer`.  No extra sizing logic belongs here.
    """

    @abstractmethod
    def open_position(
        self,
        symbol: str,
        direction: TradeDirection,
        lots: float,
        price: float,
        stop_price: float,
        comment: str = "",
    ) -> OrderResult:
        """Submit a market order with a hard stop-loss."""

    @abstractmethod
    def close_position(
        self,
        symbol: str,
        order_id: str,
        price: float,
        lots: float,
        comment: str = "",
    ) -> OrderResult:
        """Close an existing position at market."""

    @abstractmethod
    def get_spread_pts(self, symbol: str) -> float:
        """Return current spread in broker points (for friction estimation)."""

    @abstractmethod
    def get_equity(self) -> Optional[float]:
        """Poll current account equity from the broker.

        Returns the broker's reported balance (float) when available, or
        ``None`` when the value cannot be obtained (paper mode, network
        error, or not yet authenticated).  Callers must treat ``None`` as
        "use internal simulated equity" — it must never raise.
        """

    def get_order_book_imbalance(self, symbol: str, levels: int = 5) -> Optional[float]:
        """Optional L2 imbalance in [-1, 1]. Default: unavailable."""
        _ = (symbol, levels)
        return None


class PaperDispatcher(OrderDispatcher):
    """
    Paper-trading dispatcher — logs decisions, makes no real broker calls.

    Fills are simulated at the price passed in (next-bar-open proxy).
    Friction is estimated from the ``spread_pts`` provided at construction.
    """

    def __init__(self, spread_pts: Dict[str, float] | None = None) -> None:
        self._spread_pts: Dict[str, float] = spread_pts or {}
        self._order_counter: int = 0

    def open_position(
        self,
        symbol: str,
        direction: TradeDirection,
        lots: float,
        price: float,
        stop_price: float,
        comment: str = "",
    ) -> OrderResult:
        self._order_counter += 1
        oid = f"PAPER-{self._order_counter:06d}"
        logger.info(
            "[PAPER] OPEN %s %s %.5f @ %.5f stop=%.5f %s",
            symbol,
            direction.name,
            lots,
            price,
            stop_price,
            comment,
        )
        return OrderResult(
            success=True,
            order_id=oid,
            filled_price=price,
            filled_lots=lots,
        )

    def close_position(
        self,
        symbol: str,
        order_id: str,
        price: float,
        lots: float,
        comment: str = "",
    ) -> OrderResult:
        logger.info(
            "[PAPER] CLOSE %s %s @ %.5f (%s)",
            symbol,
            order_id,
            price,
            comment,
        )
        return OrderResult(
            success=True,
            order_id=order_id,
            filled_price=price,
            filled_lots=lots,
        )

    def get_spread_pts(self, symbol: str) -> float:
        return max(0.0, _safe_float(self._spread_pts.get(symbol, 1.0), 1.0))

    def get_equity(self) -> Optional[float]:
        """Paper mode: return None — engine tracks simulated equity internally."""
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# RL agent interface (thin ABC — real agents are torch/numpy policies)
# ═══════════════════════════════════════════════════════════════════════════════


class AllocationAgent(ABC):
    """
    Layer 2 allocation agent interface.

    Returns a weight vector (one per instrument) given the current
    portfolio observation.  Used by :class:`RenkoLiveTrader` to scale
    lot sizes per instrument.
    """

    @abstractmethod
    def act(self, observation: np.ndarray) -> np.ndarray:
        """
        Return allocation weights.

        Parameters
        ----------
        observation : np.ndarray
            Feature vector built by :class:`RenkoLiveTrader._build_l2_obs`.

        Returns
        -------
        np.ndarray, shape (n_instruments,)
            Per-instrument weights in [0, 1].  May not sum to 1 — each
            weight is applied independently to its instrument's sizing.
        """


class RiskAgent(ABC):
    """
    Layer 3 risk overlay agent interface.

    Returns a single scalar in [0, 1] that scales the ENTIRE portfolio
    exposure.  0 = flat, 1 = full exposure.
    """

    @abstractmethod
    def act(self, observation: np.ndarray) -> float:
        """
        Return portfolio exposure scalar.

        Parameters
        ----------
        observation : np.ndarray
            Feature vector built by :class:`RenkoLiveTrader._build_l3_obs`.

        Returns
        -------
        float
            Exposure scalar ∈ [0, 1].
        """


class UniformAllocationAgent(AllocationAgent):
    """Baseline Layer 2 agent — equal weight 1.0 for all instruments."""

    def act(self, observation: np.ndarray) -> np.ndarray:
        n = max(1, len(observation))
        return np.ones(n, dtype=np.float32)


class FullExposureRiskAgent(RiskAgent):
    """Baseline Layer 3 agent — always full exposure (scalar = 1.0)."""

    def act(self, observation: np.ndarray) -> float:
        return 1.0


# ═══════════════════════════════════════════════════════════════════════════════
# PER gate evaluator
# ═══════════════════════════════════════════════════════════════════════════════


def evaluate_per_gate(
    completed_trades: List[LiveTrade],
    current_gate: PERGate,
) -> Tuple[bool, str]:
    """
    Evaluate whether the system qualifies to advance to the next PER gate.

    Parameters
    ----------
    completed_trades : list[LiveTrade]
        All completed trades at the current gate level.
    current_gate : PERGate
        The gate currently active.

    Returns
    -------
    tuple[bool, str]
        ``(can_advance, reason_string)``
    """
    next_gate = current_gate.next_gate
    if next_gate is None:
        return False, "already at FULL gate"

    if len(completed_trades) < PER_MIN_PAPER_TRADES:
        return (
            False,
            f"need >= {PER_MIN_PAPER_TRADES} trades (have {len(completed_trades)})",
        )

    # Compute Omega ratio from net P&L array
    returns = np.array([t.net_usd for t in completed_trades], dtype=np.float64)
    gains = returns[returns > 0]
    losses = returns[returns < 0]
    if gains.size == 0:
        return False, "no winning trades — Omega = 0"
    if losses.size == 0:
        omega = float("inf")
    else:
        omega = float(gains.sum() / abs(losses.sum()))

    required = next_gate.required_score
    if omega < required:
        return (
            False,
            f"Omega {omega:.2f} < {required:.2f} required for {next_gate.value}",
        )

    return True, f"Omega {omega:.2f} >= {required:.2f} — ready for {next_gate.value}"


# ═══════════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class LiveTraderConfig:
    """
    Configuration for :class:`RenkoLiveTrader`.

    Attributes
    ----------
    symbols : list[str]
        Instruments to trade.  Must have ``qualification.json`` in the
        registry.  At most ``gate.max_instruments`` are traded simultaneously.
    gate : PERGate
        Starting PER gate.
    target_risk_usd : float
        1R risk target per trade in USD.  Used by :class:`RenkoSizer`
        (``SizingMode.FIXED_RISK``).  Ignored when
        ``sizing_mode == SizingMode.VOL_TARGET``.
    stop_bricks : float
        Stop distance as multiple of brick_size (0.5 = live canonical).
    qual_dir : Path
        Root directory of the qualification registry.
    log_dir : Path
        Directory where live trade logs (JSONL) are written.
    broker_source : str
        Broker identifier used to select the correct qualification files
        and to stamp trade log entries.
    poll_interval_seconds : float
        How often (seconds) to check for new M1 bars from the bar provider.
        Ignored when ``bar_provider`` is push-based.
    allow_short : bool
        Whether to allow short entries on down-flips.
    auto_advance_gate : bool
        If True, automatically advance the PER gate when ``evaluate_per_gate``
        returns True.  If False, gate advancement requires an explicit call to
        :meth:`RenkoLiveTrader.advance_gate`.
    lot_step : float
        Minimum lot increment for the broker.
    min_lots : float
        Minimum lot size for the broker.
    vpin_extreme_threshold : float
        VPIN value above which the circuit breaker fires (passed to
        CircuitBreakerManager config).  Default 0.85.
    vpin_kurtosis_window : int
        Rolling history length used to compute excess kurtosis of VPIN.
    drawdown_halt_pct : float
        Portfolio drawdown fraction at which all trading is halted.
    sizing_mode : SizingMode
        Position sizing mode.

        - ``SizingMode.FIXED_RISK`` (default) — targets ``target_risk_usd``
          per trade using the 1R brick-stop formula.
        - ``SizingMode.VOL_TARGET`` — volatility-targeted sizing: lots are
          sized so that the position contributes a fixed fraction of equity
          in P&L volatility per brick move, normalised by the rolling brick
          P&L standard deviation.  Requires ``vol_target_pct`` and
          ``vol_usd_per_point`` to be set.
    vol_target_pct : float
        Fraction of equity to risk per 1σ brick-vol move.  Only used when
        ``sizing_mode == SizingMode.VOL_TARGET``.  Default ``0.01`` (1 %).
    vol_window : int
        Rolling window (bricks) for the vol estimate.
        Only used when ``sizing_mode == SizingMode.VOL_TARGET``.
        Default ``50``.
    vol_floor : float
        Minimum vol estimate (fraction of brick_size) to prevent oversizing.
        Default ``0.003``.
    vol_ceil : float
        Maximum vol estimate (fraction of brick_size) to cap sizing during
        crisis / spike periods.  Default ``0.20``.
    initial_equity_usd : float
        Starting equity for vol-targeted sizing.  The live equity is tracked
        as ``initial_equity_usd + session_pnl``.  Default ``100_000.0``.
    """

    symbols: List[str] = field(default_factory=list)
    gate: PERGate = PERGate.SIMULATED
    target_risk_usd: float = 100.0
    stop_bricks: float = 0.5  # live canonical (1.0 for backtest)
    qual_dir: Path = field(default_factory=lambda: DEFAULT_QUAL_DIR)
    log_dir: Path = field(default_factory=lambda: LIVE_LOG_DIR)
    broker_source: str = "unknown"
    poll_interval_seconds: float = 60.0
    allow_short: bool = True
    auto_advance_gate: bool = False
    lot_step: float = 0.01
    min_lots: float = 0.01
    vpin_extreme_threshold: float = 0.85
    vpin_kurtosis_window: int = 50
    drawdown_halt_pct: float = 0.10
    sizing_mode: SizingMode = SizingMode.FIXED_LOT
    vol_target_pct: float = 0.01
    vol_window: int = 50
    vol_floor: float = 0.003
    vol_ceil: float = 0.20
    initial_equity_usd: float = 100_000.0
    paper_lots: float = 0.01
    startup_skip_flips: int = 2
    monday_open_utc: str = "00:00"
    friday_close_utc: str = "23:59"
    loss_brake_after_consecutive_losses: int = 8
    loss_flat_after_consecutive_losses: int = 12
    loss_pause_minutes: float = 120.0
    trailing_mae_enabled: bool = False
    trailing_mae_after_bricks: int = 1
    trailing_mae_fraction: float = 0.5
    break_even_enabled: bool = False
    break_even_after_bricks: int = 1
    break_even_trigger_bricks: float = 1.0
    break_even_buffer_ticks: int = 0
    skip_qualification: bool = False  # If True, trade even if not qualified


# ═══════════════════════════════════════════════════════════════════════════════
# Bar provider interface
# ═══════════════════════════════════════════════════════════════════════════════


class BarProvider(ABC):
    """
    Source of M1 OHLCV bars for the live trader.

    Implementations
    ---------------
    - :class:`HistoricalBarProvider` — replay from a CSV (for paper trading)
    - ``MetaAPIBarProvider``         — streams from MetaAPI
    - ``cTraderBarProvider``         — streams via cTrader Open API

    The provider feeds bars one at a time by calling the registered
    callback.  Pull-based providers are polled at
    ``LiveTraderConfig.poll_interval_seconds``; push-based providers call
    the callback directly from their own thread.
    """

    @abstractmethod
    def subscribe(
        self,
        symbol: str,
        callback: Callable[[str, float, datetime], None],
    ) -> None:
        """
        Register a callback to receive bars.

        Parameters
        ----------
        symbol : str
            Instrument symbol.
        callback : callable
            ``callback(symbol, close_price, bar_time)`` — called once
            per completed M1 bar.
        """

    @abstractmethod
    def start(self) -> None:
        """Begin delivering bars to subscribers."""

    @abstractmethod
    def stop(self) -> None:
        """Stop delivering bars and release resources."""


class VPINProvider(ABC):
    """
    Optional provider for live VPIN snapshots.

    Implementations may compute VPIN from tick/M1 feeds or read from an
    external risk process. Values should be raw VPIN in [0, 1].
    """

    @abstractmethod
    def start(self) -> None:
        """Start the VPIN provider."""

    @abstractmethod
    def stop(self) -> None:
        """Stop the VPIN provider."""

    @abstractmethod
    def get_latest(self) -> Dict[str, float]:
        """Return latest symbol->vpin snapshot."""


class NullVPINProvider(VPINProvider):
    """No-op provider used when no VPIN feed is configured."""

    def start(self) -> None:
        return

    def stop(self) -> None:
        return

    def get_latest(self) -> Dict[str, float]:
        return {}


class HistoricalBarProvider(BarProvider):
    """
    Replay-mode bar provider — feeds pre-loaded M1 data bar by bar.

    Useful for paper-trading backreplay and integration testing.

    Parameters
    ----------
    data : dict[str, pd.DataFrame]
        M1 DataFrames keyed by symbol.  Each DataFrame must have a ``time``
        (or datetime index) column and a ``close`` column.
    speed_multiplier : float
        Wall-clock speed-up factor.  ``float("inf")`` = as fast as possible.
    """

    def __init__(
        self,
        data: Dict[str, pd.DataFrame],
        speed_multiplier: float = float("inf"),
    ) -> None:
        self._data = data
        self._speed = speed_multiplier
        self._callbacks: Dict[str, List[Callable]] = {}
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

    def subscribe(
        self,
        symbol: str,
        callback: Callable[[str, float, datetime], None],
    ) -> None:
        if symbol not in self._callbacks:
            self._callbacks[symbol] = []
        self._callbacks[symbol].append(callback)

    def start(self) -> None:
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._replay, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)

    def _replay(self) -> None:
        """Interleave bars across symbols in chronological order."""
        # Build a sorted event queue: (bar_time, symbol, close)
        events: List[Tuple[pd.Timestamp, str, float]] = []
        for sym, df in self._data.items():
            if sym not in self._callbacks:
                continue
            _df = df.reset_index()
            # Resolve time column
            time_col = next(
                (c for c in _df.columns if c.lower() in ("time", "datetime", "date")),
                None,
            )
            if time_col is None and isinstance(df.index, pd.DatetimeIndex):
                _df["_time"] = df.index
                time_col = "_time"
            if time_col is None:
                logger.warning("HistoricalBarProvider: no time column for %s — skipping", sym)
                continue
            close_col = next((c for c in _df.columns if c.lower() == "close"), None)
            if close_col is None:
                logger.warning("HistoricalBarProvider: no close column for %s — skipping", sym)
                continue
            for _, row in _df.iterrows():
                t = pd.Timestamp(row[time_col])
                if t.tzinfo is None:
                    t = t.tz_localize("UTC")
                events.append((t, sym, float(row[close_col])))

        events.sort(key=lambda x: x[0])

        prev_ts: Optional[pd.Timestamp] = None
        for ts, sym, close in events:
            if self._stop_event.is_set():
                break
            if prev_ts is not None and self._speed < float("inf"):
                dt = (ts - prev_ts).total_seconds()
                sleep_s = dt / self._speed
                if sleep_s > 0:
                    self._stop_event.wait(timeout=sleep_s)
            bar_time = ts.to_pydatetime()
            for cb in self._callbacks.get(sym, []):
                try:
                    cb(sym, close, bar_time)
                except Exception:
                    logger.exception("BarProvider callback raised for %s", sym)
            prev_ts = ts


# ═══════════════════════════════════════════════════════════════════════════════
# RenkoLiveTrader
# ═══════════════════════════════════════════════════════════════════════════════


class RenkoLiveTrader:
    """
    Renko-native live/paper trader.

    Lifecycle
    ---------
    1. ``__init__`` — load qualification registry, initialise per-instrument
       state, wire RL agents, set up circuit breakers.
    2. ``start`` — subscribe to bar provider, begin processing.
    3. (bar arrives) → ``_on_bar`` → bricks → Layer 1 → Layer 2/3 → sizing
       → dispatcher.
    4. ``stop`` — graceful shutdown, flush trade log.

    Parameters
    ----------
    config : LiveTraderConfig
        Trader configuration.
    bar_provider : BarProvider
        Source of M1 bars.  Defaults to ``HistoricalBarProvider({})`` if
        None (useful for testing — produces no bars).
    dispatcher : OrderDispatcher
        Order execution backend.  Defaults to :class:`PaperDispatcher`.
    allocation_agent : AllocationAgent or None
        Layer 2 RL agent.  Defaults to :class:`UniformAllocationAgent`.
    risk_agent : RiskAgent or None
        Layer 3 RL agent.  Defaults to :class:`FullExposureRiskAgent`.
    circuit_breaker_manager : CircuitBreakerManager or None
        If provided, evaluated before every entry.  If None, circuit
        breakers are disabled (paper trading / testing).
    """

    def __init__(
        self,
        config: LiveTraderConfig,
        bar_provider: Optional[BarProvider] = None,
        vpin_provider: Optional[VPINProvider] = None,
        dispatcher: Optional[OrderDispatcher] = None,
        allocation_agent: Optional[AllocationAgent] = None,
        risk_agent: Optional[RiskAgent] = None,
        circuit_breaker_manager: Optional["CircuitBreakerManager"] = None,
    ) -> None:
        self._config = config
        self._provider = bar_provider or HistoricalBarProvider({})
        self._vpin_provider = vpin_provider or NullVPINProvider()
        self._dispatcher = dispatcher or PaperDispatcher()
        self._l2_agent: AllocationAgent = allocation_agent or UniformAllocationAgent()
        self._l3_agent: RiskAgent = risk_agent or FullExposureRiskAgent()
        self._cb_manager = circuit_breaker_manager

        self._gate = config.gate
        self._monday_open_minutes = self._parse_hhmm_utc(config.monday_open_utc, "monday_open_utc")
        self._friday_close_minutes = self._parse_hhmm_utc(
            config.friday_close_utc, "friday_close_utc"
        )

        # ── Sizer selection ───────────────────────────────────────────
        self._sizing_mode = config.sizing_mode
        if config.sizing_mode == SizingMode.VOL_TARGET:
            from kinetra.renko.vol_sizer import VolSizingConfig

            _vs_cfg = VolSizingConfig(
                target_vol_pct=config.vol_target_pct,
                vol_window=config.vol_window,
                vol_floor=config.vol_floor,
                vol_ceil=config.vol_ceil,
            )
            self._vol_sizer: Optional[VolTargetSizerLive] = VolTargetSizerLive(
                config=_vs_cfg,
                lot_step=config.lot_step,
                min_lots=config.min_lots,
            )
        else:
            self._vol_sizer = None

        self._sizer = RenkoSizer(
            target_risk_usd=config.target_risk_usd,
            lot_step=config.lot_step,
            min_lots=config.min_lots,
        )

        # Per-instrument live state — populated in _load_qualification_registry
        self._instrument_states: Dict[str, InstrumentLiveState] = {}

        # All completed trades this session (all instruments combined)
        self._all_trades: List[LiveTrade] = []
        self._trade_lock = threading.Lock()

        # Session equity tracking (portfolio-level)
        self._session_pnl: float = 0.0
        self._session_peak: float = float(config.initial_equity_usd)

        # Running trade ID counter
        self._trade_counter: int = 0

        # Control flags
        self._running = False
        self._halted = False  # set by circuit breaker or DD limit

        # Live VPIN feed cache (updated externally via update_vpin()).
        self._vpin_latest: Dict[str, float] = {}
        self._vpin_history: Dict[str, List[float]] = {}
        self._bar_counter: int = 0
        self._last_heartbeat_ts: datetime = datetime.now(tz=timezone.utc)
        self._entry_block_counts: Dict[str, int] = {}
        self._entry_signals_seen: int = 0
        self._entries_opened: int = 0

        # Log file handle
        config.log_dir.mkdir(parents=True, exist_ok=True)
        _ts = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self._log_path = config.log_dir / f"live_trades_{_ts}.jsonl"
        self._log_handle = open(self._log_path, "a")  # noqa: SIM115

        logger.info(
            "RenkoLiveTrader initialised: gate=%s, symbols=%s, log=%s",
            self._gate.value,
            config.symbols,
            self._log_path,
        )
        emit_event(
            stream="paper_trading" if self._gate == PERGate.SIMULATED else "live_trading",
            component="renko_live_trader",
            event_type="init",
            status="info",
            payload={
                "gate": self._gate.value,
                "symbols": list(config.symbols),
                "log_path": str(self._log_path),
                "sizing_mode": self._sizing_mode.value,
                "startup_skip_flips": int(self._config.startup_skip_flips),
                "monday_open_utc": str(self._config.monday_open_utc),
                "friday_close_utc": str(self._config.friday_close_utc),
                "loss_brake_after_consecutive_losses": int(
                    self._config.loss_brake_after_consecutive_losses
                ),
                "loss_flat_after_consecutive_losses": int(
                    self._config.loss_flat_after_consecutive_losses
                ),
                "loss_pause_minutes": float(self._config.loss_pause_minutes),
                "trailing_mae_enabled": bool(self._config.trailing_mae_enabled),
                "trailing_mae_after_bricks": int(self._config.trailing_mae_after_bricks),
                "trailing_mae_fraction": float(self._config.trailing_mae_fraction),
                "break_even_enabled": bool(self._config.break_even_enabled),
                "break_even_after_bricks": int(self._config.break_even_after_bricks),
                "break_even_trigger_bricks": float(self._config.break_even_trigger_bricks),
                "break_even_buffer_ticks": int(self._config.break_even_buffer_ticks),
            },
        )

        # Load qualification data for all requested symbols
        self._load_qualification_registry()
        for sym in self._instrument_states:
            self._vpin_latest.setdefault(sym, float("nan"))
            self._vpin_history.setdefault(sym, [])

    # ──────────────────────────────────────────────────────────────────────────
    # Qualification registry loading
    # ──────────────────────────────────────────────────────────────────────────

    def _load_qualification_registry(self) -> None:
        """
        Load per-instrument qualification data from disk.

        Only instruments that are:
          - listed in config.symbols
          - have a qualification.json with qualified=True
          - have broker_source matching config.broker_source (or "unknown")
        are added to the active instrument set.

        Instruments that exceed the PER gate's max_instruments cap (ranked
        by Omega descending) are excluded with a warning.
        """
        qual_dir = Path(self._config.qual_dir)
        if not qual_dir.exists():
            logger.warning(
                "Qualification directory not found: %s — no instruments loaded",
                qual_dir,
            )
            return

        qualified: List[Tuple[str, "QualificationResult"]] = []

        for sym in self._config.symbols:
            q_path = qual_dir / sym / "qualification.json"
            if not q_path.exists():
                logger.warning("%s: no qualification.json found in %s", sym, qual_dir)
                continue
            try:
                raw = json.loads(q_path.read_text())
            except Exception as exc:
                logger.error("%s: failed to load qualification.json: %s", sym, exc)
                continue

            if not raw.get("qualified", False):
                if not self._config.skip_qualification:
                    logger.info("%s: not qualified — skipping", sym)
                    continue
                logger.warning(
                    "%s: not qualified but skip_qualification=True — including anyway", sym
                )

            if raw.get("recalibration_due", False):
                logger.warning(
                    "%s: recalibration_due=True — parameters may be stale. "
                    "Run recalibration before live trading.",
                    sym,
                )

            bs = raw.get("broker_source", "unknown")
            if (
                self._config.broker_source != "unknown"
                and bs != "unknown"
                and bs != self._config.broker_source
            ):
                logger.warning(
                    "%s: broker_source mismatch (file=%s, config=%s) — skipping",
                    sym,
                    bs,
                    self._config.broker_source,
                )
                continue

            qualified.append((sym, raw))

        # Sort by Omega descending so gate cap keeps the best instruments
        qualified.sort(key=lambda x: x[1].get("omega", 0.0), reverse=True)

        cap = self._gate.max_instruments
        if len(qualified) > cap:
            dropped = [s for s, _ in qualified[cap:]]
            logger.warning(
                "PER gate %s: max_instruments=%d — dropping %s",
                self._gate.value,
                cap,
                dropped,
            )
            qualified = qualified[:cap]

        for sym, raw in qualified:
            fp = raw.get("filter_params", {})
            # risk_params reserved for Sprint 5A RiskParams integration
            _ = raw.get("risk_params", {})

            # Session break minutes — load from session_profile.json if available
            sbm = _load_session_break_minutes(qual_dir / sym, default=30.0)

            tick_size = _safe_positive(raw.get("tick_size", 0.0001), 0.0001)
            state = InstrumentLiveState(
                symbol=sym,
                brick_size=_safe_positive(raw.get("brick_size", 1.0), 1.0),
                filter_fliprate_threshold=float(
                    np.clip(_safe_float(fp.get("fliprate_threshold", 0.35), 0.35), 0.0, 1.0)
                ),
                filter_markov_threshold=float(
                    np.clip(_safe_float(fp.get("markov_threshold", 0.55), 0.55), 0.0, 1.0)
                ),
                filter_fliprate_window=max(2, int(_safe_float(fp.get("fliprate_window", 50), 50))),
                filter_markov_window=max(2, int(_safe_float(fp.get("markov_window", 50), 50))),
                stop_bricks=_safe_positive(self._config.stop_bricks, 1.0),
                session_break_minutes=_safe_positive(sbm, 30.0),
                usd_per_point=_safe_positive(raw.get("usd_per_point", 1.0), 1.0),
                tick_size=tick_size,
                price_digits=max(
                    0,
                    int(
                        _safe_float(
                            raw.get("digits", _infer_price_digits_from_tick_size(tick_size)),
                            _infer_price_digits_from_tick_size(tick_size),
                        )
                    ),
                ),
                cluster=raw.get("cluster", "unknown"),
                startup_flips_seen=0,
                startup_warmup_done=(int(self._config.startup_skip_flips) <= 0),
            )
            self._instrument_states[sym] = state
            logger.info(
                "%s: loaded qualification (brick=%.5f, fr_thr=%.3f, mk_thr=%.3f, sbm=%.0f, usd_per_point=%.3f, tick_size=%.5f, digits=%d)",
                sym,
                state.brick_size,
                state.filter_fliprate_threshold,
                state.filter_markov_threshold,
                state.session_break_minutes,
                float(state.usd_per_point),
                float(state.tick_size),
                int(state.price_digits),
            )

        logger.info(
            "Loaded %d qualified instruments: %s",
            len(self._instrument_states),
            list(self._instrument_states.keys()),
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Lifecycle
    # ──────────────────────────────────────────────────────────────────────────

    def start(self) -> None:
        """Start the live trader — subscribe to bar provider and begin loop."""
        if not self._instrument_states:
            logger.warning("No qualified instruments loaded — nothing to trade.")
            emit_health(
                component="renko_live_trader",
                status="warn",
                checks={"qualified_instruments": "empty"},
            )
            return

        self._running = True
        for sym in self._instrument_states:
            self._provider.subscribe(sym, self._on_bar)

        logger.info("RenkoLiveTrader starting: gate=%s", self._gate.value)
        emit_health(
            component="renko_live_trader",
            status="ok",
            checks={"start": "pass"},
            details={"gate": self._gate.value, "n_symbols": len(self._instrument_states)},
        )
        try:
            self._vpin_provider.start()
        except Exception:
            logger.exception("VPIN provider start failed; continuing without live VPIN feed")
        self._provider.start()

    def stop(self) -> None:
        """Gracefully stop — close open positions, flush log."""
        self._running = False
        self._provider.stop()
        try:
            self._vpin_provider.stop()
        except Exception:
            logger.exception("VPIN provider stop failed")
        self._force_close_all("manual_stop")
        self._flush_log()
        try:
            self._log_handle.close()
        except Exception:
            pass
        logger.info(
            "RenkoLiveTrader stopped. Session P&L: %.2f USD, trades: %d",
            self._session_pnl,
            len(self._all_trades),
        )
        emit_health(
            component="renko_live_trader",
            status="ok",
            checks={"stop": "pass"},
            metrics={
                "session_pnl_usd": float(self._session_pnl),
                "n_completed_trades": int(len(self._all_trades)),
            },
            details={"gate": self._gate.value},
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Bar processing
    # ──────────────────────────────────────────────────────────────────────────

    def _on_bar(self, symbol: str, close: float, bar_time: datetime) -> None:
        """
        Called once per completed M1 bar.

        Steps
        -----
        1. Append bar to rolling buffer → rebuild bricks.
        2. Feed new bricks to the vol sizer (if VOL_TARGET mode active).
        3. Check if the last brick triggers an exit on open position.
        4. Check if the last brick triggers an entry.
        5. If entry signal: run Layer 2/3, size, submit order.
        """
        if not self._running or self._halted:
            return

        state = self._instrument_states.get(symbol)
        if state is None:
            return
        if not _is_finite_number(close):
            return
        if bar_time.tzinfo is None:
            bar_time = bar_time.replace(tzinfo=timezone.utc)
        if not self._is_in_weekly_window(bar_time):
            if self._is_friday_after_close(bar_time):
                self._force_close_all("weekly_close")
            return
        self._bar_counter += 1
        self._emit_heartbeat_if_due(bar_time)

        # 1. Update buffer
        prev_n_bricks = len(state.get_bricks())
        state.append_bar(close, bar_time)

        # 2. Get bricks
        bricks = state.get_bricks()
        if bricks.empty or len(bricks) < 2:
            return

        # 2a. Refresh live VPIN snapshot (best-effort).
        self._refresh_vpin_feed()

        # 2b. Feed any new bricks into the vol sizer so the rolling vol
        #     estimate stays current between entries.
        if self._vol_sizer is not None:
            new_n = len(bricks)
            if new_n > prev_n_bricks:
                new_bricks = bricks.iloc[prev_n_bricks:]
                for _, row in new_bricks.iterrows():
                    d = int(row["direction"])
                    self._vol_sizer.update(symbol, float(state.brick_size) * d)

        directions = bricks["direction"].values.astype(np.int8)
        brick_closes = bricks["brick_close"].values.astype(np.float64)
        brick_times = bricks["time"].values
        n = len(bricks)
        last_idx = n - 1

        last_price = _quantize_price(
            float(brick_closes[last_idx]),
            state.tick_size,
            state.price_digits,
        )
        last_dir = int(directions[last_idx])
        last_time_raw = brick_times[last_idx]
        last_time = pd.Timestamp(last_time_raw)
        if last_time.tzinfo is None:
            last_time = last_time.tz_localize("UTC")
        last_time_dt = last_time.to_pydatetime()

        # 3. Check exit for open position
        if state.open_trade is not None:
            self._check_exit(
                state=state,
                price=last_price,
                direction=last_dir,
                bar_time=last_time_dt,
                all_directions=directions,
                brick_closes=brick_closes,
                last_idx=last_idx,
            )

        # 4. Check entry — only when a new brick formed AND it's a colour flip.
        #    Entry signals require a colour flip, which can only happen on brick
        #    formation.  Track last_idx to avoid counting the same brick twice
        #    when multiple bars arrive before the next brick forms.
        is_new_brick = len(bricks) > prev_n_bricks
        if (
            is_new_brick
            and state.open_trade is None
            and not self._halted
            and last_idx > 0
            and last_idx != state.last_checked_brick_idx  # Skip if same brick as before
            and directions[last_idx] != directions[last_idx - 1]  # Colour flip!
        ):
            state.last_checked_brick_idx = last_idx
            self._check_entry(
                state,
                last_idx,
                directions,
                brick_closes,
                brick_times,
                last_price,
                last_dir,
                last_time_dt,
                is_new_brick,
            )

    def _emit_heartbeat_if_due(self, now: datetime) -> None:
        """Emit lightweight periodic system health snapshot."""
        if now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)
        elapsed = (now - self._last_heartbeat_ts).total_seconds()
        if elapsed < HEARTBEAT_INTERVAL_S:
            return
        self._last_heartbeat_ts = now
        open_positions = sum(
            1 for s in self._instrument_states.values() if s.open_trade is not None
        )
        emit_health(
            component="renko_live_trader",
            status="warn" if self._halted else "ok",
            checks={"heartbeat": "pass"},
            metrics={
                "bars_seen": int(self._bar_counter),
                "open_positions": int(open_positions),
                "session_pnl_usd": float(self._session_pnl),
                "portfolio_dd": float(self._portfolio_drawdown()),
            },
            details={"gate": self._gate.value},
        )

    @staticmethod
    def _parse_hhmm_utc(raw: Any, field_name: str) -> int:
        s = str(raw).strip()
        parts = s.split(":", 1)
        if len(parts) != 2:
            raise ValueError(f"{field_name} must be HH:MM (UTC), got {raw!r}")
        try:
            hh = int(parts[0])
            mm = int(parts[1])
        except Exception as exc:
            raise ValueError(f"{field_name} must be HH:MM (UTC), got {raw!r}") from exc
        if hh < 0 or hh > 23 or mm < 0 or mm > 59:
            raise ValueError(f"{field_name} must be HH:MM (UTC), got {raw!r}")
        return hh * 60 + mm

    def _is_in_weekly_window(self, ts: datetime) -> bool:
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        wd = ts.weekday()  # Mon=0 .. Sun=6
        mins = ts.hour * 60 + ts.minute
        if wd >= 5:
            return False
        if wd == 0 and mins < self._monday_open_minutes:
            return False
        if wd == 4 and mins >= self._friday_close_minutes:
            return False
        return True

    def _is_friday_after_close(self, ts: datetime) -> bool:
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        if ts.weekday() != 4:
            return False
        mins = ts.hour * 60 + ts.minute
        return mins >= self._friday_close_minutes

    def _check_exit(
        self,
        state: InstrumentLiveState,
        price: float,
        direction: int,
        bar_time: datetime,
        all_directions: np.ndarray,
        brick_closes: np.ndarray,
        last_idx: int,
    ) -> None:
        """Evaluate stop and colour-change exits for the open position."""
        trade = state.open_trade
        if trade is None:
            LOG.warning("Exit evaluation called with no open trade, skipping")
            return

        if state.active_stop_price is None:
            stop_distance = state.stop_bricks * state.brick_size
            state.active_stop_price = (
                trade.entry_price - stop_distance
                if trade.direction == TradeDirection.LONG
                else trade.entry_price + stop_distance
            )
        self._maybe_apply_break_even_stop(state, trade, brick_closes, last_idx)
        self._maybe_update_trailing_stop(state, trade, brick_closes, last_idx)
        stop_ref = float(state.active_stop_price)
        stop_hit = (trade.direction == TradeDirection.LONG and price <= stop_ref) or (
            trade.direction == TradeDirection.SHORT and price >= stop_ref
        )
        colour_change = direction != trade.direction.value

        if stop_hit or colour_change:
            reason = "stop" if stop_hit else "colour_change"
            spread_pts = max(0.0, _safe_float(self._dispatcher.get_spread_pts(state.symbol), 0.0))
            friction = (
                spread_pts * state.tick_size * state.usd_per_point * _safe_float(trade.lots, 0.0)
            )

            result = self._dispatcher.close_position(
                symbol=state.symbol,
                order_id=trade.trade_id,
                price=price,
                lots=trade.lots,
                comment=reason,
            )
            exit_price = result.filled_price if result.filled_price is not None else price
            exit_price = _quantize_price(exit_price, state.tick_size, state.price_digits)
            trade.close(exit_price, bar_time, reason, friction, state.usd_per_point)
            state.open_trade = None
            state.active_stop_price = None
            state.entry_brick_index = None
            state.record_closed_trade(trade)
            self._record_trade_closed(trade)
            if trade.net_usd < 0:
                state.consecutive_losses += 1
                brake_after = max(
                    int(_safe_float(self._config.loss_brake_after_consecutive_losses, 8)),
                    1,
                )
                flat_after = max(
                    int(_safe_float(self._config.loss_flat_after_consecutive_losses, 12)),
                    brake_after + 1,
                )
                if state.consecutive_losses >= flat_after:
                    logger.error(
                        "%s: consecutive loss halt triggered (%d >= %d)",
                        state.symbol,
                        state.consecutive_losses,
                        flat_after,
                    )
                    self._halted = True
                    emit_health(
                        component="renko_live_trader",
                        status="critical",
                        checks={"loss_streak_halt": "triggered"},
                        details={
                            "symbol": state.symbol,
                            "consecutive_losses": int(state.consecutive_losses),
                            "threshold": int(flat_after),
                        },
                    )
                elif state.consecutive_losses >= brake_after:
                    pause_mins = max(_safe_float(self._config.loss_pause_minutes, 120.0), 1.0)
                    state.pause_until_utc = bar_time + timedelta(minutes=pause_mins)
                    logger.warning(
                        "%s: consecutive loss brake triggered (%d >= %d), paused until %s",
                        state.symbol,
                        state.consecutive_losses,
                        brake_after,
                        state.pause_until_utc.isoformat(),
                    )
                    emit_event(
                        stream="paper_trading"
                        if trade.gate == PERGate.SIMULATED
                        else "live_trading",
                        component="renko_live_trader",
                        event_type="loss_brake",
                        status="warn",
                        payload={
                            "symbol": state.symbol,
                            "consecutive_losses": int(state.consecutive_losses),
                            "brake_after": int(brake_after),
                            "pause_until_utc": state.pause_until_utc.isoformat(),
                        },
                    )
            else:
                state.consecutive_losses = 0
                state.pause_until_utc = None
            emit_event(
                stream="paper_trading" if trade.gate == PERGate.SIMULATED else "live_trading",
                component="renko_live_trader",
                event_type="trade_closed",
                status="ok",
                payload={
                    "symbol": state.symbol,
                    "trade_id": trade.trade_id,
                    "direction": trade.direction.name,
                    "entry_price": float(trade.entry_price),
                    "exit_price": float(exit_price),
                    "reason": reason,
                    "lots": float(trade.lots),
                    "net_usd": float(trade.net_usd),
                    "friction_usd": float(trade.friction_usd),
                    "session_pnl_usd": float(self._session_pnl),
                },
            )

            logger.info(
                "[%s] %s %s @ %.5f → %.5f (%s) net=%.2f USD",
                self._gate.value.upper(),
                state.symbol,
                trade.direction.name,
                trade.entry_price,
                exit_price,
                reason,
                trade.net_usd,
            )

            # Portfolio-level DD check
            self._check_portfolio_dd()

    def _check_entry(
        self,
        state: InstrumentLiveState,
        last_idx: int,
        directions: np.ndarray,
        brick_closes: np.ndarray,
        brick_times: np.ndarray,
        price: float,
        direction: int,
        bar_time: datetime,
        is_new_brick: bool,
    ) -> None:
        """Evaluate Layer 1 filter → Layer 2/3 → size → open position.

        Called only when a colour flip is verified by the caller.
        """
        # Colour flip confirmed by caller — this is a valid entry signal
        self._entry_signals_seen += 1
        if state.pause_until_utc is not None and bar_time < state.pause_until_utc:
            if is_new_brick:
                self._note_entry_block("loss_pause")
            return
        if state.pause_until_utc is not None and bar_time >= state.pause_until_utc:
            state.pause_until_utc = None

        # Startup warmup gate: skip first N flip signals after startup/reconnect.
        if not state.startup_warmup_done:
            state.startup_flips_seen += 1
            if state.startup_flips_seen <= int(self._config.startup_skip_flips):
                if is_new_brick:
                    self._note_entry_block("startup_skip")
                logger.info(
                    "%s: startup warmup skip flip %d/%d",
                    state.symbol,
                    state.startup_flips_seen,
                    int(self._config.startup_skip_flips),
                )
                return
            state.startup_warmup_done = True
            logger.info(
                "%s: startup warmup complete after %d skipped flips",
                state.symbol,
                int(self._config.startup_skip_flips),
            )

        # Compute filter signals
        fr_vals = flip_rate(directions, state.filter_fliprate_window)
        pUU_vals, pDD_vals = markov_stickiness(directions, state.filter_markov_window)

        if last_idx >= len(fr_vals) or not np.isfinite(fr_vals[last_idx]):
            if is_new_brick:
                self._note_entry_block("fliprate_unready")
            return
        if last_idx >= len(pUU_vals) or not np.isfinite(pUU_vals[last_idx]):
            if is_new_brick:
                self._note_entry_block("markov_unready")
            return

        fr_val = float(fr_vals[last_idx])
        pUU_val = float(pUU_vals[last_idx])
        pDD_val = float(pDD_vals[last_idx])

        # Layer 1 filter gate
        entry_ok = evaluate_entry(
            direction=direction,
            flip_rate_val=fr_val,
            pUU=pUU_val,
            pDD=pDD_val,
            fliprate_threshold=state.filter_fliprate_threshold,
            markov_threshold=state.filter_markov_threshold,
        )
        if not entry_ok:
            if is_new_brick:
                self._note_entry_block("filter_reject")
            return

        # Direction check
        if direction == -1 and not self._config.allow_short:
            if is_new_brick:
                self._note_entry_block("short_disabled")
            return

        # Circuit breakers (non-negotiable)
        if self._cb_manager is not None:
            cb_result = self._evaluate_circuit_breakers()
            if cb_result:
                if is_new_brick:
                    self._note_entry_block(f"circuit_breaker:{cb_result}")
                logger.info(
                    "%s: circuit breaker fired (%s) — skipping entry",
                    state.symbol,
                    cb_result,
                )
                return

        # Layer 2: allocation weight for this instrument
        l2_obs = self._build_l2_obs(state)
        l2_weights = self._l2_agent.act(l2_obs)
        sym_idx = list(self._instrument_states.keys()).index(state.symbol)
        if sym_idx >= len(l2_weights):
            if is_new_brick:
                self._note_entry_block("alloc_index_miss")
            return
        l2_weight = float(np.clip(_safe_float(l2_weights[sym_idx], 0.0), 0.0, 1.0))

        # Layer 3: portfolio exposure scalar
        l3_obs = self._build_l3_obs()
        l3_exposure = float(np.clip(_safe_float(self._l3_agent.act(l3_obs), 0.0), 0.0, 1.0))

        # Update state
        state.layer2_weight = l2_weight
        state.layer3_exposure = l3_exposure

        # Size — use vol-targeted sizer when configured, else fixed-risk sizer
        if self._sizing_mode == SizingMode.VOL_TARGET and self._vol_sizer is not None:
            live_equity = self._config.initial_equity_usd + self._session_pnl
            lots = self._vol_sizer.compute(
                symbol=state.symbol,
                equity_usd=live_equity,
                brick_size=state.brick_size,
                usd_per_point=state.usd_per_point,
                layer2_weight=l2_weight,
                layer3_exposure=l3_exposure,
                gate=self._gate,
            )
        else:
            lots = self._sizer.compute(
                brick_size=state.brick_size,
                usd_per_point=state.usd_per_point,
                layer2_weight=l2_weight,
                layer3_exposure=l3_exposure,
                gate=self._gate,
            )

        # Paper mode: use configured synthetic lot size for realistic dry-run P&L.
        effective_lots = (
            lots
            if lots > 0
            else (float(self._config.paper_lots) if self._gate == PERGate.SIMULATED else 0.0)
        )
        effective_lots = _quantize_lots(
            effective_lots,
            self._config.lot_step,
            self._config.min_lots,
        )
        # Debug logging: show computed sizing details so we can diagnose zero-lot skips
        logger.debug(
            "%s: sizing computed raw_lots=%.6f quantized_lots=%.6f gate=%s layer2=%.3f layer3=%.3f brick=%.5f usd_per_point=%.6f",
            state.symbol,
            float(lots),
            float(effective_lots),
            self._gate.value,
            float(l2_weight),
            float(l3_exposure),
            float(state.brick_size),
            float(state.usd_per_point),
        )
        if effective_lots <= 0 and self._gate != PERGate.SIMULATED:
            # Log detailed reason before skipping
            logger.info(
                "%s: zero_lots — skipping entry (raw_lots=%.6f, quantized=%.6f, gate=%s)",
                state.symbol,
                float(lots),
                float(effective_lots),
                self._gate.value,
            )
            if is_new_brick:
                self._note_entry_block("zero_lots")
            return

        stop_price = (
            price - state.stop_bricks * state.brick_size
            if direction == 1
            else price + state.stop_bricks * state.brick_size
        )
        stop_price = _quantize_price(stop_price, state.tick_size, state.price_digits)

        # Dispatch
        result = self._dispatcher.open_position(
            symbol=state.symbol,
            direction=TradeDirection(direction),
            lots=effective_lots,
            price=price,
            stop_price=stop_price,
            comment=f"renko_{self._gate.value}",
        )

        if result.success:
            filled_price = result.filled_price if result.filled_price is not None else price
            filled_price = _quantize_price(filled_price, state.tick_size, state.price_digits)
            self._trade_counter += 1
            trade = LiveTrade(
                trade_id=result.order_id or f"T{self._trade_counter:06d}",
                symbol=state.symbol,
                direction=TradeDirection(direction),
                entry_price=filled_price,
                entry_time=bar_time,
                brick_size=state.brick_size,
                lots=effective_lots,
                target_risk_usd=self._config.target_risk_usd,
                gate=self._gate,
                layer2_weight=l2_weight,
                layer3_exposure=l3_exposure,
            )
            state.open_trade = trade
            self._entries_opened += 1
            state.active_stop_price = stop_price
            state.entry_brick_index = int(last_idx)
            logger.info(
                "[%s] %s OPEN %s @ %.5f lots=%.2f stop=%.5f w2=%.2f w3=%.2f",
                self._gate.value.upper(),
                state.symbol,
                TradeDirection(direction).name,
                filled_price,
                effective_lots,
                stop_price,
                l2_weight,
                l3_exposure,
            )
            emit_event(
                stream="paper_trading" if self._gate == PERGate.SIMULATED else "live_trading",
                component="renko_live_trader",
                event_type="trade_opened",
                status="ok",
                payload={
                    "symbol": state.symbol,
                    "trade_id": trade.trade_id,
                    "direction": TradeDirection(direction).name,
                    "entry_price": float(filled_price),
                    "stop_price": float(stop_price),
                    "active_stop_price": float(stop_price),
                    "lots": float(effective_lots),
                    "layer2_weight": float(l2_weight),
                    "layer3_exposure": float(l3_exposure),
                    "gate": self._gate.value,
                },
            )
        else:
            if is_new_brick:
                self._note_entry_block("open_reject")

    def _note_entry_block(self, reason: str) -> None:
        key = str(reason).strip() or "unknown"
        self._entry_block_counts[key] = int(self._entry_block_counts.get(key, 0)) + 1

    def _maybe_update_trailing_stop(
        self,
        state: InstrumentLiveState,
        trade: LiveTrade,
        brick_closes: np.ndarray,
        last_idx: int,
    ) -> None:
        """Optional MAE-based trailing stop with one-way ratchet."""
        if not bool(self._config.trailing_mae_enabled):
            return
        entry_idx = state.entry_brick_index
        if entry_idx is None or last_idx <= entry_idx:
            return
        bricks_open = int(last_idx - entry_idx)
        if bricks_open < max(int(self._config.trailing_mae_after_bricks), 1):
            return
        frac = float(np.clip(_safe_float(self._config.trailing_mae_fraction, 0.5), 0.0, 1.0))
        window = brick_closes[int(entry_idx) : int(last_idx) + 1]
        if window.size <= 0:
            return
        entry = float(trade.entry_price)
        if trade.direction == TradeDirection.LONG:
            adverse = max(0.0, entry - float(np.min(window)))
            candidate = entry - (adverse * frac)
            current = float(
                state.active_stop_price if state.active_stop_price is not None else candidate
            )
            state.active_stop_price = max(current, candidate)
        else:
            adverse = max(0.0, float(np.max(window)) - entry)
            candidate = entry + (adverse * frac)
            current = float(
                state.active_stop_price if state.active_stop_price is not None else candidate
            )
            state.active_stop_price = min(current, candidate)
        state.active_stop_price = _quantize_price(
            float(state.active_stop_price),
            state.tick_size,
            state.price_digits,
        )

    def _maybe_apply_break_even_stop(
        self,
        state: InstrumentLiveState,
        trade: LiveTrade,
        brick_closes: np.ndarray,
        last_idx: int,
    ) -> None:
        """Optional break-even stop activation after favorable move."""
        if not bool(self._config.break_even_enabled):
            return
        entry_idx = state.entry_brick_index
        if entry_idx is None or last_idx <= entry_idx:
            return
        bricks_open = int(last_idx - entry_idx)
        if bricks_open < max(int(self._config.break_even_after_bricks), 1):
            return
        window = brick_closes[int(entry_idx) : int(last_idx) + 1]
        if window.size <= 0:
            return

        trigger = _safe_positive(
            float(self._config.break_even_trigger_bricks) * float(state.brick_size),
            float(state.brick_size),
        )
        entry = float(trade.entry_price)
        buffer_ticks = max(int(self._config.break_even_buffer_ticks), 0)
        buffer_price = float(buffer_ticks) * float(state.tick_size)

        if trade.direction == TradeDirection.LONG:
            mfe = max(0.0, float(np.max(window)) - entry)
            if mfe + NUMERIC_EPS < trigger:
                return
            candidate = entry + buffer_price
            current = float(
                state.active_stop_price if state.active_stop_price is not None else candidate
            )
            state.active_stop_price = max(current, candidate)
        else:
            mfe = max(0.0, entry - float(np.min(window)))
            if mfe + NUMERIC_EPS < trigger:
                return
            candidate = entry - buffer_price
            current = float(
                state.active_stop_price if state.active_stop_price is not None else candidate
            )
            state.active_stop_price = min(current, candidate)

        state.active_stop_price = _quantize_price(
            float(state.active_stop_price),
            state.tick_size,
            state.price_digits,
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Circuit breakers
    # ──────────────────────────────────────────────────────────────────────────

    def _refresh_vpin_feed(self) -> None:
        """Pull latest VPIN snapshot from provider and update caches."""
        try:
            snap = self._vpin_provider.get_latest()
            if not snap:
                return
            for sym, val in snap.items():
                self.update_vpin(sym, float(val))
        except Exception:
            logger.exception("VPIN provider snapshot failed")

    def update_vpin(self, symbol: str, vpin_value: float) -> None:
        """
        Update rolling VPIN state for one symbol.

        This method is intended to be called by an external VPIN feed process.
        """
        if symbol not in self._instrument_states:
            return
        if not np.isfinite(vpin_value):
            return
        self._vpin_latest[symbol] = float(vpin_value)
        hist = self._vpin_history.setdefault(symbol, [])
        hist.append(float(vpin_value))
        w = max(int(self._config.vpin_kurtosis_window), 4)
        if len(hist) > w:
            del hist[:-w]

    def _vpin_snapshot(self) -> tuple[float, float, float]:
        """Return (vpin_mean, vpin_max, vpin_kurtosis)."""
        vals = [
            v
            for s, v in self._vpin_latest.items()
            if s in self._instrument_states and np.isfinite(v)
        ]
        if not vals:
            return 0.0, 0.0, 0.0
        v_mean = float(np.mean(vals))
        v_max = float(np.max(vals))

        # Conservative tail-risk aggregation: use max kurtosis across symbols.
        k_vals: List[float] = []
        for sym in self._instrument_states:
            h = self._vpin_history.get(sym, [])
            k = vpin_excess_kurtosis(h, min_samples=max(8, self._config.vpin_kurtosis_window // 3))
            if np.isfinite(k):
                k_vals.append(float(k))
        v_kurt = float(np.max(k_vals)) if k_vals else 0.0
        return v_mean, v_max, v_kurt

    def _evaluate_circuit_breakers(self) -> Optional[str]:
        """
        Run the CircuitBreakerManager if configured.

        Returns a non-empty string describing the fired breaker, or None
        if all breakers are clear.
        """
        if self._cb_manager is None:
            return None
        try:
            from kinetra.monitoring.circuit_breakers import PortfolioSnapshot

            vpin_mean, vpin_max, vpin_kurtosis = self._vpin_snapshot()
            n_open = sum(1 for s in self._instrument_states.values() if s.open_trade is not None)

            snap = PortfolioSnapshot(
                portfolio_dd=self._portfolio_drawdown(),
                vpin_mean=vpin_mean,
                vpin_max=vpin_max,
                vpin_kurtosis=vpin_kurtosis,
                spread_ratio_max=1.0,
                entry_concurrence=0.0,
                n_active_positions=n_open,
                n_instruments=len(self._instrument_states),
            )
            eval_result = self._cb_manager.evaluate(snap)
            if eval_result.is_triggered:
                fired_names = [r.breaker_type.value for r in eval_result.triggered_breakers]
                emit_event(
                    stream="live_trading",
                    component="renko_live_trader",
                    event_type="circuit_breaker_triggered",
                    status="warn",
                    payload={
                        "fired": fired_names,
                        "portfolio_dd": float(snap.portfolio_dd),
                        "vpin_mean": float(vpin_mean),
                        "vpin_max": float(vpin_max),
                        "vpin_kurtosis": float(vpin_kurtosis),
                        "n_active_positions": int(n_open),
                    },
                )
                return ", ".join(fired_names)
        except Exception:
            logger.exception("Circuit breaker evaluation failed")
        return None

    def _check_portfolio_dd(self) -> None:
        """Halt trading if portfolio drawdown exceeds the gate limit."""
        dd = self._portfolio_drawdown()
        if dd < -self._config.drawdown_halt_pct:
            logger.warning(
                "Portfolio drawdown %.2f%% exceeds halt threshold %.2f%% — halting",
                dd * 100,
                self._config.drawdown_halt_pct * 100,
            )
            self._halted = True
            emit_health(
                component="renko_live_trader",
                status="critical",
                checks={"drawdown_halt": "triggered"},
                metrics={
                    "portfolio_dd": float(dd),
                    "threshold": float(-self._config.drawdown_halt_pct),
                },
                details={"gate": self._gate.value},
            )
            self._force_close_all("dd_halt")

    def _portfolio_drawdown(self) -> float:
        """Current portfolio-level drawdown as a negative fraction."""
        peak_equity = max(
            _safe_float(self._session_peak, self._config.initial_equity_usd),
            NUMERIC_EPS,
        )
        current_equity = _safe_float(
            self._config.initial_equity_usd + self._session_pnl,
            self._config.initial_equity_usd,
        )
        if peak_equity <= NUMERIC_EPS:
            return 0.0
        dd = _safe_div(current_equity - peak_equity, peak_equity, default=0.0)
        return float(np.clip(dd, -1.0, 0.0))

    # ──────────────────────────────────────────────────────────────────────────
    # RL observation builders
    # ──────────────────────────────────────────────────────────────────────────

    def _build_l2_obs(self, state: InstrumentLiveState) -> np.ndarray:
        """
        Build the Layer 2 observation vector.

        Features (per instrument, fixed-length):
          [vr_proxy, friction_ratio, recent_win_rate, session_pnl_norm,
           layer3_exposure, n_instruments_norm]

        This is intentionally compact — the RL agent trained offline with
        the same feature set.  Extend only with a matching retrain.
        """
        bricks = state.get_bricks()
        n_bricks = len(bricks)

        # VR proxy: use win rate of last 20 bricks as a stickiness proxy
        vr_proxy = 0.5
        if n_bricks >= 20:
            dirs = bricks["direction"].values[-20:].astype(np.int8)
            runs = int(np.sum(dirs[1:] == dirs[:-1]))
            vr_proxy = float(runs / max(len(dirs) - 1, 1))

        # Friction ratio from qualification (static per instrument)
        bricks_df = bricks
        friction_ratio = 0.1
        if not bricks_df.empty:
            # Approximate as spread (0 for paper) / brick_size
            spread_pts = self._dispatcher.get_spread_pts(state.symbol)
            friction_ratio = float(np.clip(spread_pts / max(state.brick_size, 1e-9), 0.0, 1.0))

        # Recent win rate from session trades
        recent = state.session_trades[-20:]
        win_rate = float(sum(1 for t in recent if t.net_usd > 0) / max(len(recent), 1))

        # Session P&L normalised by target risk
        pnl_norm = float(
            np.clip(state.cumulative_pnl / max(self._config.target_risk_usd, 1.0), -5.0, 5.0)
        )

        # Current Layer 3 exposure
        l3_exp = float(state.layer3_exposure)

        # Number of instruments (normalised)
        n_instr_norm = float(len(self._instrument_states) / max(self._gate.max_instruments, 1))

        obs = np.array(
            [vr_proxy, friction_ratio, win_rate, pnl_norm, l3_exp, n_instr_norm],
            dtype=np.float32,
        )
        # Replicate for each instrument position (uniform agent ignores this)
        n = len(self._instrument_states)
        return np.tile(obs, n)

    def _build_l3_obs(self) -> np.ndarray:
        """
        Build the Layer 3 risk observation vector.

        Features:
          [portfolio_dd, portfolio_return_5d, n_open_positions_norm,
           vpin_mean, vpin_max, spread_regime, corr_regime,
           vol_regime, vr_drift, recalibration_pending]

        Matches N_RISK_OBS_FEATURES = 10 from risk_env.py.
        """
        n_open = sum(1 for s in self._instrument_states.values() if s.open_trade is not None)
        n_total = max(len(self._instrument_states), 1)

        dd = float(np.clip(self._portfolio_drawdown(), -1.0, 0.0))
        n_open_norm = float(n_open / n_total)

        # Recent portfolio return (last 5 closed trades)
        recent_5 = self._all_trades[-5:]
        ret_5 = float(sum(t.net_usd for t in recent_5) / max(self._config.target_risk_usd, 1.0))
        ret_5 = float(np.clip(ret_5, -5.0, 5.0))
        vpin_mean, vpin_max, _ = self._vpin_snapshot()

        obs = np.array(
            [
                dd,  # portfolio_dd
                ret_5,  # portfolio_return_5d
                n_open_norm,  # n_open_positions_norm
                vpin_mean,  # vpin_mean
                vpin_max,  # vpin_max
                0.0,  # spread_regime
                0.0,  # corr_regime
                0.0,  # vol_regime
                0.0,  # vr_drift
                0.0,  # recalibration_pending
            ],
            dtype=np.float32,
        )
        return obs

    # ──────────────────────────────────────────────────────────────────────────
    # Trade recording
    # ──────────────────────────────────────────────────────────────────────────

    def _record_trade_closed(self, trade: LiveTrade) -> None:
        """Thread-safe recording of a closed trade."""
        with self._trade_lock:
            self._all_trades.append(trade)
            self._session_pnl += _safe_float(trade.net_usd, 0.0)
            current_equity = self._config.initial_equity_usd + self._session_pnl
            self._session_peak = max(self._session_peak, current_equity)
        self._append_trade_log(trade)
        self._maybe_advance_gate()

    def _append_trade_log(self, trade: LiveTrade) -> None:
        """Append one trade to the JSONL log file."""
        try:
            self._log_handle.write(json.dumps(trade.to_dict()) + "\n")
            self._log_handle.flush()
        except Exception:
            logger.exception("Failed to write trade log")

    def _flush_log(self) -> None:
        """Flush the log file (called on shutdown)."""
        try:
            self._log_handle.flush()
        except Exception:
            pass

    # ──────────────────────────────────────────────────────────────────────────
    # PER gate management
    # ──────────────────────────────────────────────────────────────────────────

    def _maybe_advance_gate(self) -> None:
        """Check PER advancement after each trade (if auto_advance enabled)."""
        if not self._config.auto_advance_gate:
            return
        gate_trades = [t for t in self._all_trades if t.gate == self._gate]
        can_advance, reason = evaluate_per_gate(gate_trades, self._gate)
        if can_advance:
            self.advance_gate(reason)

    def advance_gate(self, reason: str = "") -> bool:
        """
        Attempt to advance to the next PER gate.

        Returns True if advanced, False if already at FULL or prerequisites
        not met.
        """
        next_gate = self._gate.next_gate
        if next_gate is None:
            logger.info("PER: already at FULL gate — no advancement")
            return False

        gate_trades = [t for t in self._all_trades if t.gate == self._gate]
        can_advance, eval_reason = evaluate_per_gate(gate_trades, self._gate)
        if not can_advance:
            logger.info("PER: cannot advance (%s)", eval_reason)
            return False

        old_gate = self._gate
        self._gate = next_gate
        logger.info(
            "PER GATE ADVANCED: %s → %s (%s) %s",
            old_gate.value,
            self._gate.value,
            eval_reason,
            reason,
        )
        # Log gate advancement event
        event = {
            "event": "gate_advanced",
            "from_gate": old_gate.value,
            "to_gate": self._gate.value,
            "reason": eval_reason,
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
            "n_trades_at_gate": len(gate_trades),
        }
        try:
            self._log_handle.write(json.dumps(event) + "\n")
            self._log_handle.flush()
        except Exception:
            pass
        return True

    # ──────────────────────────────────────────────────────────────────────────
    # Force-close
    # ──────────────────────────────────────────────────────────────────────────

    def _force_close_all(self, reason: str) -> None:
        """Close all open positions (used on stop / circuit breaker halt)."""
        now = datetime.now(tz=timezone.utc)
        for state in self._instrument_states.values():
            if state.open_trade is not None:
                trade = state.open_trade
                last_price = state._price_buffer[-1] if state._price_buffer else trade.entry_price
                spread_pts = max(
                    0.0, _safe_float(self._dispatcher.get_spread_pts(state.symbol), 0.0)
                )
                friction = (
                    spread_pts
                    * state.tick_size
                    * state.usd_per_point
                    * _safe_float(trade.lots, 0.0)
                )
                result = self._dispatcher.close_position(
                    symbol=state.symbol,
                    order_id=trade.trade_id,
                    price=last_price,
                    lots=trade.lots,
                    comment=reason,
                )
                exit_price = result.filled_price if result.filled_price is not None else last_price
                exit_price = _quantize_price(exit_price, state.tick_size, state.price_digits)
                trade.close(exit_price, now, reason, friction, state.usd_per_point)
                state.open_trade = None
                state.record_closed_trade(trade)
                self._record_trade_closed(trade)

    # ──────────────────────────────────────────────────────────────────────────
    # Status / reporting
    # ──────────────────────────────────────────────────────────────────────────

    @property
    def gate(self) -> PERGate:
        """Active PER gate."""
        return self._gate

    @property
    def is_halted(self) -> bool:
        """True if the trader has been halted by a circuit breaker or DD limit."""
        return self._halted

    @property
    def session_pnl(self) -> float:
        """Cumulative session P&L (USD)."""
        return self._session_pnl

    @property
    def n_completed_trades(self) -> int:
        """Total completed trades this session."""
        return len(self._all_trades)

    def vol_sizer_report(self) -> Optional[Dict[str, dict]]:
        """
        Return a diagnostic snapshot of the vol sizer state, or None if
        ``sizing_mode != VOL_TARGET``.

        Keyed by symbol; each value contains:
        ``vol_estimate``, ``n_observations``, ``is_warmed_up``.
        """
        if self._vol_sizer is None:
            return None
        report = {}
        for sym, state in self._instrument_states.items():
            report[sym] = {
                "vol_estimate": self._vol_sizer.vol_estimate(sym, state.brick_size),
                "n_observations": self._vol_sizer.n_observations(sym),
                "is_warmed_up": self._vol_sizer.is_warmed_up(sym),
            }
        return report

    def session_summary(self) -> Dict[str, Any]:
        """
        Return a summary dict of the current session.

        Suitable for logging, menu display, or JSON serialisation.
        """
        returns = np.array([t.net_usd for t in self._all_trades], dtype=np.float64)
        gains = returns[returns > 0]
        losses = returns[returns < 0]
        gross_profit = float(gains.sum()) if gains.size > 0 else 0.0
        gross_loss_abs = float(abs(losses.sum())) if losses.size > 0 else 0.0
        omega = (
            float(gross_profit / gross_loss_abs)
            if losses.size > 0 and gains.size > 0
            else (float("inf") if gains.size > 0 else 0.0)
        )
        win_rate = float(gains.size / max(len(returns), 1))
        profit_factor = (
            float(gross_profit / gross_loss_abs)
            if gross_loss_abs > NUMERIC_EPS
            else (float("inf") if gross_profit > 0 else 0.0)
        )
        streaks = _compute_streak_stats([float(t.net_usd) for t in self._all_trades])

        per_instrument = {}
        for sym, state in self._instrument_states.items():
            st = state.session_trades
            per_instrument[sym] = {
                "n_trades": len(st),
                "net_usd": float(sum(t.net_usd for t in st)),
                "win_rate": float(sum(1 for t in st if t.net_usd > 0) / max(len(st), 1)),
                "open_position": state.open_trade is not None,
                "consecutive_losses": int(state.consecutive_losses),
                "paused_until_utc": (
                    state.pause_until_utc.isoformat() if state.pause_until_utc else None
                ),
            }

        can_advance, advance_reason = evaluate_per_gate(
            [t for t in self._all_trades if t.gate == self._gate],
            self._gate,
        )

        summary: Dict[str, Any] = {
            "gate": self._gate.value,
            "sizing_mode": self._sizing_mode.value,
            "is_halted": self._halted,
            "session_pnl_usd": self._session_pnl,
            "n_completed_trades": len(self._all_trades),
            "omega": omega,
            "profit_factor": profit_factor,
            "gross_profit_usd": gross_profit,
            "gross_loss_abs_usd": gross_loss_abs,
            "win_rate": win_rate,
            "portfolio_drawdown": self._portfolio_drawdown(),
            "max_consecutive_wins": streaks["max_consecutive_wins"],
            "max_consecutive_losses": streaks["max_consecutive_losses"],
            "current_streak_type": streaks["current_streak_type"],
            "current_streak_length": streaks["current_streak_length"],
            "per_instrument": per_instrument,
            "can_advance_gate": can_advance,
            "advance_reason": advance_reason,
            "log_path": str(self._log_path),
            "entry_signals_seen": int(self._entry_signals_seen),
            "entries_opened": int(self._entries_opened),
            "entry_block_counts": dict(
                sorted(self._entry_block_counts.items(), key=lambda kv: (-kv[1], kv[0]))
            ),
        }
        vol_report = self.vol_sizer_report()
        if vol_report is not None:
            summary["vol_sizer"] = vol_report
        return summary

    def __repr__(self) -> str:
        return (
            f"RenkoLiveTrader("
            f"gate={self._gate.value}, "
            f"symbols={list(self._instrument_states.keys())}, "
            f"trades={len(self._all_trades)}, "
            f"pnl={self._session_pnl:.2f}"
            f")"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Module-level helpers
# ═══════════════════════════════════════════════════════════════════════════════


def _load_session_break_minutes(sym_qual_dir: Path, default: float = 30.0) -> float:
    """
    Load session_break_minutes from session_profile.json in *sym_qual_dir*.

    Returns *default* if the file is absent or malformed.
    """
    sp_path = sym_qual_dir / "session_profile.json"
    if not sp_path.exists():
        return default
    try:
        raw = json.loads(sp_path.read_text())
        sbm = raw.get("session_break_minutes")
        if sbm is not None:
            return float(sbm)
    except Exception as exc:
        logger.debug("Could not load session_profile.json from %s: %s", sym_qual_dir, exc)
    return default


def load_agent_from_file(
    agent_type: str,
    path: Path,
) -> "AllocationAgent | RiskAgent":
    """
    Load a trained RL agent from a checkpoint file.

    Parameters
    ----------
    agent_type : str
        "allocation" for Layer 2 (:class:`AllocationAgent`) or
        "risk" for Layer 3 (:class:`RiskAgent`).
    path : Path
        Path to the checkpoint (NumPy ``.npz`` or PyTorch ``.pt``).

    Returns
    -------
    AllocationAgent or RiskAgent
        Loaded policy.  Falls back to the uniform/full-exposure baseline
        if loading fails.
    """
    path = Path(path)
    if not path.exists():
        logger.warning("Agent checkpoint not found: %s — using baseline agent", path)
        return UniformAllocationAgent() if agent_type == "allocation" else FullExposureRiskAgent()

    try:
        if path.suffix == ".npz":
            return _load_numpy_agent(agent_type, path)
        if path.suffix in (".pt", ".pth"):
            return _load_torch_agent(agent_type, path)
        logger.warning("Unrecognised agent file extension: %s — using baseline", path.suffix)
    except Exception as exc:
        logger.error("Failed to load agent from %s: %s", path, exc)

    return UniformAllocationAgent() if agent_type == "allocation" else FullExposureRiskAgent()


def _load_numpy_agent(agent_type: str, path: Path) -> "AllocationAgent | RiskAgent":
    """Load a LinearPolicy agent from a NumPy checkpoint."""
    data = np.load(path, allow_pickle=False)

    class _LinearAllocationAgent(AllocationAgent):
        def __init__(self, w: np.ndarray, b: np.ndarray) -> None:
            self._w = w
            self._b = b

        def act(self, observation: np.ndarray) -> np.ndarray:
            raw = self._w @ observation.astype(np.float32) + self._b
            return np.clip(raw, 0.0, 1.0).astype(np.float32)

    class _LinearRiskAgent(RiskAgent):
        def __init__(self, w: np.ndarray, b: np.ndarray) -> None:
            self._w = w
            self._b = b

        def act(self, observation: np.ndarray) -> float:
            raw = (self._w @ observation.astype(np.float32) + self._b).item()
            return float(np.clip(raw, 0.0, 1.0))

    w = data["weights"]
    b = data.get("bias", np.zeros(w.shape[0]))
    if agent_type == "allocation":
        return _LinearAllocationAgent(w, b)
    return _LinearRiskAgent(w, b)


def _load_torch_agent(agent_type: str, path: Path) -> "AllocationAgent | RiskAgent":
    """Load a PyTorch MLP agent from a checkpoint."""
    try:
        import torch  # noqa: F401
        import torch.nn as nn

        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        state_dict = checkpoint.get("model_state_dict") or checkpoint

        # Infer architecture from state_dict
        layer_keys = [k for k in state_dict if "weight" in k]
        if not layer_keys:
            raise ValueError("No weight tensors found in checkpoint")

        layers: List[nn.Module] = []
        for i, key in enumerate(layer_keys):
            w = state_dict[key]
            out_features, in_features = w.shape
            layers.append(nn.Linear(in_features, out_features))
            if i < len(layer_keys) - 1:
                layers.append(nn.ReLU())

        model = nn.Sequential(*layers)
        model.load_state_dict(
            {k: v for k, v in state_dict.items() if "weight" in k or "bias" in k},
            strict=False,
        )
        model.eval()

        class _TorchAllocationAgent(AllocationAgent):
            def __init__(self, m: nn.Module) -> None:
                self._model = m

            def act(self, observation: np.ndarray) -> np.ndarray:
                with torch.no_grad():
                    x = torch.tensor(observation, dtype=torch.float32).unsqueeze(0)
                    raw = self._model(x).squeeze(0).numpy()
                return np.clip(raw, 0.0, 1.0).astype(np.float32)

        class _TorchRiskAgent(RiskAgent):
            def __init__(self, m: nn.Module) -> None:
                self._model = m

            def act(self, observation: np.ndarray) -> float:
                with torch.no_grad():
                    x = torch.tensor(observation, dtype=torch.float32).unsqueeze(0)
                    raw = float(self._model(x).squeeze(0).item())
                return float(np.clip(raw, 0.0, 1.0))

        if agent_type == "allocation":
            return _TorchAllocationAgent(model)
        return _TorchRiskAgent(model)

    except ImportError:
        raise ImportError(
            "PyTorch is required to load .pt agent checkpoints. Install it with: pip install torch"
        )
