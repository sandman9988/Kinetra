"""
Renko Trading Engine
====================

Single engine used for every mode:

  ``backtest(closes)``
      Batch historical simulation.  Bricks and filter arrays are
      precomputed for efficiency on large datasets.

  ``process_bar(price, time)``
      Feed one M1 bar in streaming mode.  Maintains incremental brick
      and filter state between calls.

  ``run(bar_provider, dispatcher, stop_event)``
      Full streaming loop — subscribes to a bar provider and dispatches
      trade signals.  Used for both paper trading and live trading; the
      *only* difference between the two is the ``Dispatcher`` passed in.

Sizing modes (``EngineConfig.sizing_mode``):
  ``static``       — fixed ``min_lots`` every trade.
  ``compounding``  — ``compounding_per_k`` lots per $1 000 equity,
                     grows as account grows.
  ``risk_based``   — constant-risk: ``target_risk_usd / (brick × usd_per_point)``.

Usage::

    from kinetra.renko.trading_engine import (
        EngineConfig, RenkoEngine, PaperDispatcher,
    )

    cfg = EngineConfig(symbol="XAUUSD", brick_size=10.0)
    engine = RenkoEngine(cfg)

    # batch backtest
    results = engine.backtest(m1_closes)

    # paper trading (historical replay)
    from kinetra.renko.live_trader import HistoricalBarProvider
    provider = HistoricalBarProvider({"XAUUSD": m1_df})
    results = engine.run(provider, PaperDispatcher())

    # live micro-lots
    import threading
    stop = threading.Event()
    results = engine.run(live_provider, broker_dispatcher, stop_event=stop)
"""

from __future__ import annotations

import json
import logging
import math
import queue
import threading
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from kinetra.renko.brick_engine import IncrementalRenkoBuilder, build_renko
from kinetra.renko.filters import (
    evaluate_entry,
    flip_rate,
    markov_stickiness,
    permutation_entropy,
)
from kinetra.renko.live_trader import (
    LiveTrade,
    OrderDispatcher,
    PaperDispatcher,
    PERGate,
    TradeDirection,
)
from kinetra.renko.trade_analytics import analyze_trades

if TYPE_CHECKING:
    from kinetra.friction_cost import InstrumentSpec

LOG = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════════
# Persistence Types
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class EngineState:
    """Serializable engine state for crash recovery."""

    version: int = 1
    timestamp: str = ""  # ISO format UTC
    symbol: str = ""

    # Position state
    in_pos: bool = False
    pos_dir: int = 0
    entry_price: float = 0.0
    entry_time: Optional[str] = None  # ISO format
    entry_lots: float = 0.0
    open_order_id: Optional[str] = None
    open_signal_id: Optional[str] = None

    # Engine state
    brick_count: int = 0
    cumulative_pnl: float = 0.0
    live_equity: float = 0.0

    # Brick builder state (for resume)
    last_ref_price: Optional[float] = None
    last_bar_time: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "timestamp": self.timestamp,
            "symbol": self.symbol,
            "in_pos": self.in_pos,
            "pos_dir": self.pos_dir,
            "entry_price": self.entry_price,
            "entry_time": self.entry_time,
            "entry_lots": self.entry_lots,
            "open_order_id": self.open_order_id,
            "open_signal_id": self.open_signal_id,
            "brick_count": self.brick_count,
            "cumulative_pnl": self.cumulative_pnl,
            "live_equity": self.live_equity,
            "last_ref_price": self.last_ref_price,
            "last_bar_time": self.last_bar_time,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "EngineState":
        return cls(
            version=d.get("version", 1),
            timestamp=d.get("timestamp", ""),
            symbol=d.get("symbol", ""),
            in_pos=d.get("in_pos", False),
            pos_dir=d.get("pos_dir", 0),
            entry_price=d.get("entry_price", 0.0),
            entry_time=d.get("entry_time"),
            entry_lots=d.get("entry_lots", 0.0),
            open_order_id=d.get("open_order_id"),
            open_signal_id=d.get("open_signal_id"),
            brick_count=d.get("brick_count", 0),
            cumulative_pnl=d.get("cumulative_pnl", 0.0),
            live_equity=d.get("live_equity", 0.0),
            last_ref_price=d.get("last_ref_price"),
            last_bar_time=d.get("last_bar_time"),
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Atomic Persistence Helpers
# ═══════════════════════════════════════════════════════════════════════════════


def _atomic_write_json(path: Path, data: Dict[str, Any]) -> None:
    """Write JSON atomically using temp file + rename."""
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    try:
        tmp_path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
        tmp_path.replace(path)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
        raise


def _load_json_if_exists(path: Path) -> Optional[Dict[str, Any]]:
    """Load JSON if file exists, return None otherwise."""
    try:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        LOG.warning("Failed to load state from %s: %s", path, exc)
    return None


# ═══════════════════════════════════════════════════════════════════════════════
# Async Order Types
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class OrderRequest:
    """Signal to open a position — queued for async execution."""

    signal_id: str
    direction: int  # 1 or -1
    price: float
    stop_price: float
    lots: float
    bar_time: pd.Timestamp
    brick_count: int


@dataclass
class OrderFill:
    """Fill confirmation from broker."""

    signal_id: str
    order_id: str
    client_order_id: str
    filled_price: float
    filled_lots: float
    success: bool
    error: Optional[str] = None


# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class EngineConfig:
    """Unified config for all RenkoEngine modes."""

    symbol: str
    brick_size: float

    # Contract spec (XAUUSD defaults)
    usd_per_tick: float = 1.0  # $1 per tick per lot  (100 oz × $0.01)
    tick_size: float = 0.01  # minimum price increment
    pip_size: float = 0.1  # 1 pip = 0.1 price (10 ticks)

    # Strategy params
    stop_bricks: float = 0.5
    trailing_mfe_fraction: float = 0.0  # lock this fraction of MFE (0 disables)
    trailing_mfe_after_bricks: int = 0  # activate trailing after N held bricks
    min_warmup_bricks: int = 2
    startup_skip_flips: int = 2
    fliprate_window: int = 50
    markov_window: int = 50
    fliprate_threshold: float = 0.35
    markov_threshold: float = 0.55
    allow_short: bool = True

    # Regime filters
    use_time_decay: bool = True
    time_decay_window: int = 50
    stale_brick_factor: float = 2.0
    markov_stale_penalty: float = 0.15
    use_permutation_entropy: bool = True
    pe_window: int = 50
    pe_order: int = 3
    pe_entry_threshold: float = 0.70
    pe_exit_threshold: float = 0.85
    pe_low_entropy: float = 0.30
    pe_markov_relax: float = 0.03
    pe_markov_tighten: float = 0.05
    pe_vol_adjust: bool = True
    pe_vol_sensitivity: float = 0.50

    # Execution microstructure
    use_obi_slippage_buffer: bool = False
    obi_levels: int = 5
    obi_max_buffer_bricks: float = 0.25
    obi_ema_alpha: float = 0.30

    # VWAP order slicing (optional, disabled by default — fully backward compatible)
    # When enabled the engine splits each entry into up to ``vwap_max_slices``
    # child orders, each capped at ``vwap_participation_rate`` of the most
    # recently observed bar volume.  The dispatcher's ``open_position`` is still
    # called per slice; aggregation is the dispatcher's responsibility.
    use_vwap_slicing: bool = False
    vwap_participation_rate: float = 0.10  # target fraction of bar volume per slice
    vwap_min_slice_lots: float = 0.01      # minimum lots per slice (floor)
    vwap_max_slices: int = 5               # hard cap on number of slices

    # Sizing
    sizing_mode: str = "compounding"  # "static" | "compounding" | "risk_based"
    target_risk_usd: float = 100.0  # risk_based: USD at risk per brick
    initial_equity: float = 1_000.0  # starting account equity (default $1,000)
    compounding_per_k: float = 0.01  # compounding: lots per $1 000 equity
    lot_step: float = 0.01
    min_lots: float = 0.01
    gate_lot_ceiling: Optional[float] = None  # None = use broker volume_max from spec

    # Friction
    spread_ticks: float = 89.0  # ~8.9 pips for XAUUSD
    commission_per_lot: float = 7.0  # round-trip
    spread_sides: float = 1.0  # 1.0=single-spread RT cost, 2.0=entry+exit spread
    conservative_fills: bool = False  # apply adverse fill offsets (backtest hardening)
    entry_slip_bricks: float = 0.0
    exit_slip_bricks: float = 0.0
    stop_worst_case_bricks: float = 0.0
    flip_trade_through_bricks: float = 0.0

    # Broker feasibility / margin model
    # margin_rate = required_margin / notional (e.g., 0.01 => ~1:100 leverage)
    margin_rate: float = 0.01
    margin_call_level_pct: float = 100.0
    stop_out_level_pct: float = 50.0
    enforce_margin_checks: bool = True

    # Swap (overnight carry)
    # swap_long/short_usd_per_day: USD per lot per effective swap day.
    # For mode 0 (pips): pre-computed as swap_points × tick_value_usd.
    # For mode 1 (% p.a.): pre-computed at build time using the spec's poll-time price.
    # triple_swap_day: weekday that carries 3× the daily rate (1=Mon … 7=Sun).
    swap_long_usd_per_day: float = 0.0
    swap_short_usd_per_day: float = 0.0
    triple_swap_day: int = 3  # Wednesday default (metals/forex)

    # Persistence
    persist_dir: Optional[str] = None  # State directory (None = ~/.kinetra/state/<symbol>)


# ──────────────────────────────────────────────────────────────────────────────
# Swap helpers
# ──────────────────────────────────────────────────────────────────────────────


def _count_effective_swap_days(
    entry_time: "pd.Timestamp",
    exit_time: "pd.Timestamp",
    triple_swap_day: int,
) -> float:
    """Count effective swap days between entry and exit.

    Rules (matching the Pepperstone cTrader screenshots):
    - A swap is charged for each midnight UTC the position spans.
    - Saturday and Sunday midnights are skipped (Weekend swaps: Disabled).
    - The triple_swap_day midnight charges 3× instead of 1×.
      triple_swap_day uses 1=Mon … 7=Sun convention (matching spec).

    Returns the total effective day-count (a float to support the 3× multiplier).
    """
    if entry_time is None or exit_time is None or entry_time >= exit_time:
        return 0.0

    # First midnight strictly after entry
    first_midnight = entry_time.floor("D") + pd.Timedelta(days=1)
    total = 0.0
    current = first_midnight
    while current <= exit_time:
        # pandas weekday: 0=Mon … 6=Sun; spec convention: 1=Mon … 7=Sun
        wd_pandas = current.weekday()
        if wd_pandas < 5:  # Mon–Fri only (skip Sat=5, Sun=6)
            wd_spec = wd_pandas + 1  # convert to 1-based
            total += 3.0 if wd_spec == triple_swap_day else 1.0
        current += pd.Timedelta(days=1)
    return total


# ──────────────────────────────────────────────────────────────────────────────
# Engine
# ──────────────────────────────────────────────────────────────────────────────


class RenkoEngine:
    """
    Unified Renko trading engine for backtesting, paper, and live trading.

    All modes use identical strategy logic.  The only difference between
    modes is *where bars come from* and *what happens when a signal fires*:

    +--------------+---------------------+------------------------+
    | Mode         | Bar source          | Dispatcher             |
    +==============+=====================+========================+
    | Backtest     | ``pd.Series``       | None (P&L only)        |
    +--------------+---------------------+------------------------+
    | Paper        | ``HistoricalBarProvider`` | ``PaperDispatcher``|
    +--------------+---------------------+------------------------+
    | Live micro   | live ``BarProvider``| broker + lot_ceiling=0.01|
    +--------------+---------------------+------------------------+
    """

    def __init__(
        self,
        config: EngineConfig,
        spec: Optional["InstrumentSpec"] = None,
        quiet_mode: bool = False,
    ) -> None:
        """Initialize the Renko engine.

        Parameters
        ----------
        quiet_mode : bool
            If True, suppress brick-by-brick logs (for live trading)


        Parameters
        ----------
        config : EngineConfig
            Trading configuration
        spec : InstrumentSpec, optional
            Broker instrument spec. If provided and gate_lot_ceiling is None,
            uses spec.volume_max as the lot ceiling.
        """
        self.cfg = config
        self._spec = spec
        self._quiet_mode = quiet_mode
        # Resolve gate_lot_ceiling: use broker volume_max if not explicitly set
        self._lot_ceiling = (
            config.gate_lot_ceiling
            if config.gate_lot_ceiling is not None
            else (getattr(spec, "volume_max", 10.0) if spec is not None else 10.0)
        )
        self._usd_per_point = config.usd_per_tick / config.tick_size
        self._reset_state()
        self._init_persistence(config.persist_dir)

    # ── Public API ────────────────────────────────────────────────────────────

    def reset(self) -> None:
        """Clear all mutable state. Called automatically at start of each run."""
        self._reset_state()

    def backtest(self, closes: pd.Series) -> Dict[str, Any]:
        """
        Batch historical simulation on M1 close prices.

        Filter arrays are precomputed once for the full brick sequence —
        efficient on large datasets (years of M1 data).

        Parameters
        ----------
        closes : pd.Series
            M1 close prices with a DatetimeIndex.

        Returns
        -------
        dict
            ``{"trades": [...], "summary": {...}}``
        """
        self._reset_state()
        bricks = build_renko(closes, brick_size=self.cfg.brick_size)
        if bricks.empty:
            return {"error": "no_bricks"}

        directions = bricks["direction"].values.astype(np.int8)
        brick_closes = bricks["brick_close"].values.astype(np.float64)
        times = bricks["time"].values
        n = len(bricks)

        # Precompute filter arrays over the full sequence (batch path only)
        fr_vals = flip_rate(directions, self.cfg.fliprate_window)
        pUU_vals, pDD_vals = markov_stickiness(directions, self.cfg.markov_window)

        for i in range(n):
            b_close = float(brick_closes[i])
            direction = int(directions[i])
            bar_time = pd.Timestamp(times[i])
            fr_val = float(fr_vals[i]) if np.isfinite(fr_vals[i]) else float("nan")
            pUU_val = float(pUU_vals[i]) if np.isfinite(pUU_vals[i]) else float("nan")
            pDD_val = float(pDD_vals[i]) if np.isfinite(pDD_vals[i]) else float("nan")
            self._process_brick(b_close, direction, bar_time, fr_val, pUU_val, pDD_val)

        # Force-close any open position at end of data
        if self._in_pos and n > 0:
            self._force_close(float(brick_closes[-1]), pd.Timestamp(times[-1]))

        return self._make_results()

    def process_bar(self, price: float, time: datetime) -> None:
        """
        Feed one M1 bar in streaming mode.

        Maintains incremental brick and filter state between calls.
        Call this in your own bar loop, or use :meth:`run` which wires
        everything to a :class:`~kinetra.renko.live_trader.BarProvider`.
        """
        ts = _ensure_utc(pd.Timestamp(time))
        new_bricks = self._builder.update(price, ts)
        for _b_open, b_close, direction in new_bricks:
            self._dir_deque.append(direction)
            fr_val, pUU_val, pDD_val = self._streaming_filter_vals()
            self._process_brick(
                b_close, direction, ts, fr_val, pUU_val, pDD_val, self._active_dispatcher
            )

    def run(
        self,
        bar_provider: Any,
        dispatcher: Optional[OrderDispatcher] = None,
        stop_event: Optional[threading.Event] = None,
    ) -> Dict[str, Any]:
        """
        Streaming trading loop — paper or live.

        ASYNC ARCHITECTURE:
        - Bar Processor Thread: Processes price bars, generates signals (NEVER blocks)
        - Order Executor Thread: Handles blocking broker calls (blocking OK)
        - Fill Callback: Updates position state when fills arrive

        This prevents bar drops during order placement delays.

        Parameters
        ----------
        bar_provider
            Any object with ``subscribe(symbol, callback)`` / ``start()`` /
            ``stop()`` — matches the
            :class:`~kinetra.renko.live_trader.BarProvider` interface.
            Callbacks may be called with extra keyword args (e.g.
            ``open_``, ``high``, ``low``, ``volume``) — the engine
            ignores these and only consumes ``close`` and ``timestamp``.
        dispatcher
            Order execution.  Defaults to :class:`~kinetra.renko.live_trader.PaperDispatcher`
            (no real orders).
        stop_event
            Set this ``threading.Event`` to stop cleanly.  If ``None``
            the loop runs until ``KeyboardInterrupt``.

        Returns
        -------
        dict
            Same format as :meth:`backtest`.
        """
        if dispatcher is None:
            dispatcher = PaperDispatcher()

        self._reset_state()
        self._active_dispatcher = dispatcher

        # ── Crash-recovery: restore persisted state if present ────────────────
        prev_state = self._load_state()
        if prev_state is not None:
            self._restore_state(prev_state)

        # ── Thread-safe queues ────────────────────────────────────────────────
        # Bar queue: Twisted reactor → Bar processor
        _bar_queue: "queue.Queue[Optional[tuple]]" = queue.Queue(maxsize=1024)
        # Order queue: Bar processor → Order executor
        _order_queue: "queue.Queue[Optional[OrderRequest]]" = queue.Queue(maxsize=256)
        # Fill queue: Order executor → Bar processor (for state updates)
        _fill_queue: "queue.Queue[OrderFill]" = queue.Queue(maxsize=256)

        last_bar_ts: Optional[pd.Timestamp] = None
        _shutdown = threading.Event()

        # ── Thread 1: Bar Processor (NEVER blocks on orders) ───────────────────
        def _bar_processor() -> None:
            """Process bars, generate signals, queue orders (never blocks)."""
            nonlocal last_bar_ts
            try:
                while not _shutdown.is_set():
                    try:
                        item = _bar_queue.get(timeout=0.1)
                    except queue.Empty:
                        # Check for pending fills
                        self._process_pending_fills(_fill_queue, dispatcher)
                        continue

                    if item is None:  # Shutdown sentinel
                        break

                    close, timestamp = item
                    ts = _ensure_utc(pd.Timestamp(timestamp))
                    self._stream_bars_seen += 1

                    # In M1-bar mode, deduplicate bars by timestamp (same minute
                    # can only produce one bar).  In tick mode the dedup is
                    # skipped — multiple ticks within the same second are valid
                    # and must all reach the Renko builder.
                    if not _tick_mode:
                        if last_bar_ts is not None and ts <= last_bar_ts:
                            self._stream_duplicate_bars_dropped += 1
                            continue
                        last_bar_ts = ts

                    # Process bar/tick and generate signals (non-blocking)
                    try:
                        self._process_bar_async(close, ts, _order_queue, dispatcher)
                    except Exception as exc:
                        LOG.error("[BAR-PROCESSOR] Error processing bar: %s", exc, exc_info=True)
            except Exception as exc:
                LOG.error("[BAR-PROCESSOR] Thread crashed: %s", exc, exc_info=True)
                with _thread_health_lock:
                    _thread_health["bar_processor"] = False
                _shutdown.set()

        # ── Thread 2: Order Executor (handles blocking calls) ─────────────────
        def _order_executor() -> None:
            """Execute orders and report fills (blocking OK here)."""
            try:
                while not _shutdown.is_set():
                    try:
                        req = _order_queue.get(timeout=0.5)
                    except queue.Empty:
                        continue

                    if req is None:  # Shutdown sentinel
                        break

                    try:
                        # This blocks, but that's OK — dedicated order thread
                        result = dispatcher.open_position(
                            symbol=self.cfg.symbol,
                            direction=TradeDirection(req.direction),
                            lots=req.lots,
                            price=req.price,
                            stop_price=req.stop_price,
                            comment=req.signal_id,
                        )

                        # Report fill back to bar processor (non-blocking)
                        fill = OrderFill(
                            signal_id=req.signal_id,
                            order_id=result.order_id or "",
                            client_order_id=getattr(result, "raw", {}).get("client_order_id", ""),
                            filled_price=result.filled_price or req.price,
                            filled_lots=result.filled_lots or req.lots,
                            success=result.success,
                            error=result.error,
                        )
                        try:
                            _fill_queue.put_nowait(fill)
                        except queue.Full:
                            LOG.error("[ORDER-EXECUTOR] Fill queue full - dropping fill report!")
                    except Exception as exc:
                        LOG.error("[ORDER-EXECUTOR] Order failed: %s", exc, exc_info=True)
                        # Report failure so bar processor can reset state (non-blocking)
                        fill = OrderFill(
                            signal_id=req.signal_id,
                            order_id="",
                            client_order_id="",
                            filled_price=req.price,
                            filled_lots=req.lots,
                            success=False,
                            error=str(exc),
                        )
                        try:
                            _fill_queue.put_nowait(fill)
                        except queue.Full:
                            LOG.error("[ORDER-EXECUTOR] Fill queue full - dropping failure report!")
            except Exception as exc:
                LOG.error("[ORDER-EXECUTOR] Thread crashed: %s", exc, exc_info=True)
                with _thread_health_lock:
                    _thread_health["order_executor"] = False
                _shutdown.set()

        # ── Reactor callbacks (Twisted thread — must never block) ─────────────
        # Prefer the tick path (subscribe_ticks) when available: builds Renko
        # bricks directly from spot prices instead of waiting up to 60 s for
        # an M1 bar boundary.  Falls back to M1 bars for non-cTrader providers.
        _tick_mode: bool = hasattr(bar_provider, "subscribe_ticks")

        def _on_tick(symbol: str, price: float, timestamp: datetime, **_: Any) -> None:
            """Reactor-thread tick callback — must not block."""
            try:
                _bar_queue.put_nowait((price, timestamp))
            except queue.Full:
                LOG.warning("[engine] Tick queue full — dropping tick for %s", symbol)

        def _on_bar(symbol: str, close: float, timestamp: datetime, **_: Any) -> None:
            """Reactor-thread M1 bar callback — must not block."""
            try:
                _bar_queue.put_nowait((close, timestamp))
            except queue.Full:
                LOG.warning(
                    "[engine] Bar queue full — dropping bar for %s at %s", symbol, timestamp
                )

        import time as _time

        # Thread health monitoring — must be initialised BEFORE threads start so that
        # the crash-handler inside each thread can safely acquire the lock.
        _thread_health_lock = threading.Lock()
        _thread_health = {"bar_processor": True, "order_executor": True}

        # Start threads
        bar_thread = threading.Thread(target=_bar_processor, daemon=True, name="bar_processor")
        order_thread = threading.Thread(target=_order_executor, daemon=True, name="order_executor")
        bar_thread.start()
        order_thread.start()

        def _monitor_threads():
            """Monitor thread health and log alerts."""
            while not _shutdown.is_set():
                _time.sleep(5.0)
                with _thread_health_lock:
                    if not _thread_health["bar_processor"]:
                        LOG.critical("[HEALTH] Bar processor thread has died!")
                    if not _thread_health["order_executor"]:
                        LOG.critical("[HEALTH] Order executor thread has died!")

        monitor_thread = threading.Thread(
            target=_monitor_threads, daemon=True, name="thread_monitor"
        )
        monitor_thread.start()

        if _tick_mode:
            bar_provider.subscribe_ticks(self.cfg.symbol, _on_tick)
            LOG.info("[engine] Tick mode: Renko built from raw spot ticks (no M1 delay)")
        else:
            bar_provider.subscribe(self.cfg.symbol, _on_bar)
            LOG.info("[engine] M1 mode: Renko built from M1 bar closes")
        bar_provider.start()

        try:
            if stop_event is not None:
                stop_event.wait()
            else:
                while True:
                    _time.sleep(1)
                    # Check if threads are still alive
                    if not bar_thread.is_alive():
                        with _thread_health_lock:
                            _thread_health["bar_processor"] = False
                        LOG.critical("[HEALTH] Bar processor thread has stopped!")
                    if not order_thread.is_alive():
                        with _thread_health_lock:
                            _thread_health["order_executor"] = False
                        LOG.critical("[HEALTH] Order executor thread has stopped!")
        except KeyboardInterrupt:
            pass
        finally:
            bar_provider.stop()
            _shutdown.set()
            # Signal threads to exit
            _bar_queue.put(None)
            _order_queue.put(None)
            bar_thread.join(timeout=5.0)
            order_thread.join(timeout=5.0)
            self._active_dispatcher = None
            # Final state save before returning
            self._save_state()

        return self._make_results()

    def _process_pending_fills(
        self, fill_queue: "queue.Queue[OrderFill]", dispatcher: OrderDispatcher
    ) -> None:
        """Process fill confirmations from order executor."""
        state_changed = False
        try:
            while True:
                fill = fill_queue.get_nowait()
                with self._state_lock:
                    if fill.success:
                        self._open_order_id = fill.order_id
                        self._open_client_order_id = fill.client_order_id
                        self._waiting_for_fill = False
                        # Clear pending_close if this was a close confirmation
                        if self._pending_close:
                            self._pending_close = False
                        state_changed = True
                        LOG.info(
                            "[FILL-CONFIRMED] signal_id=%s order_id=%s",
                            fill.signal_id,
                            fill.order_id,
                        )
                    else:
                        # Fill failed — reset position state
                        self._in_pos = False
                        self._pos_dir = 0
                        self._waiting_for_fill = False
                        self._pending_close = False
                        state_changed = True
                        LOG.error("[FILL-FAILED] signal_id=%s error=%s", fill.signal_id, fill.error)
        except queue.Empty:
            pass

        # Persist state changes atomically
        if state_changed:
            if not self._save_state():
                LOG.error("[CRITICAL] Failed to persist state after fill update")

    def _process_bar_async(
        self,
        close: float,
        bar_time: pd.Timestamp,
        order_queue: "queue.Queue[Optional[OrderRequest]]",
        dispatcher: OrderDispatcher,
    ) -> None:
        """Process one bar — async version (non-blocking)."""
        if not self._running or self._halted:
            return

        if not np.isfinite(close):
            return

        # 1. Update brick builder (thread-safe, local state only)
        new_bricks = self._builder.update(close, bar_time)
        if not new_bricks:
            return

        # 2. Process each new brick
        for _b_open, b_close, direction in new_bricks:
            self._dir_deque.append(direction)
            self._update_timing_metrics(bar_time)
            self._current_pe_value()

            with self._state_lock:
                self._brick_count += 1
                in_pos = self._in_pos
                waiting = self._waiting_for_fill or self._pending_close

            fr_val, pUU_val, pDD_val = self._streaming_filter_vals()

            # Handle exits (non-blocking check) — only if position exists and no pending ops
            if in_pos and not waiting:
                self._check_exit_async(b_close, direction, bar_time, dispatcher)

            # Handle entries (queue order request, don't block) — only if flat and no pending ops
            with self._state_lock:
                can_enter = not self._in_pos and not self._waiting_for_fill

            if can_enter:
                self._check_entry_async(
                    b_close, direction, bar_time, fr_val, pUU_val, pDD_val, order_queue
                )

    def replay_simulate_from_csv(self, csv_path: str) -> Dict[str, Any]:
        """Convenience: load M1 CSV and run backtest."""
        df = pd.read_csv(csv_path, parse_dates=["time"])
        closes = pd.Series(df["close"].values, index=pd.DatetimeIndex(df["time"], tz="UTC"))
        return self.backtest(closes)

    def _check_exit_async(
        self,
        b_close: float,
        direction: int,
        bar_time: pd.Timestamp,
        dispatcher: OrderDispatcher,
    ) -> None:
        """Check for exit signals — async version."""
        pe_val = self._current_pe_value()
        entropy_exit = bool(
            bool(getattr(self.cfg, "use_permutation_entropy", True))
            and np.isfinite(pe_val)
            and pe_val >= float(getattr(self.cfg, "pe_exit_threshold", 0.85))
        )
        with self._state_lock:
            colour_change = direction != self._pos_dir
            if not colour_change and not entropy_exit:
                return

            # Mark close as pending — prevent duplicate exits
            self._pending_close = True
            pos_dir = self._pos_dir
            entry_price = self._entry_price
            entry_time = self._entry_time
            entry_lots = self._entry_lots
            open_order_id = self._open_order_id
            open_signal_id = self._open_signal_id

        stop_dist = self.cfg.stop_bricks * self.cfg.brick_size
        reason = "entropy_exit" if entropy_exit else "colour_change"

        # Fire-and-forget close (don't wait for confirmation)
        if dispatcher is not None and open_order_id is not None:
            try:
                close_result = dispatcher.close_position(
                    symbol=self.cfg.symbol,
                    order_id=open_order_id,
                    price=b_close,
                    lots=entry_lots,
                    comment=reason,
                )
                if not getattr(close_result, "success", False):
                    # Close failed - position likely still open
                    LOG.error(
                        "[EXIT-FAILED] Close rejected: %s",
                        getattr(close_result, "error", "unknown"),
                    )
                    with self._state_lock:
                        self._pending_close = False
                    return
                LOG.info("[EXIT-SENT] order_id=%s price=%.2f", open_order_id, b_close)
            except Exception as exc:
                LOG.error("[EXIT-FAILED] Close dispatch raised: %s", exc)
                # Reset pending flag so we can retry
                with self._state_lock:
                    self._pending_close = False
                return

        # Record trade completion
        gross_usd = (b_close - entry_price) * pos_dir * self._usd_per_point * entry_lots
        trade = LiveTrade(
            trade_id=open_order_id or f"T-{len(self._completed) + 1:06d}",
            symbol=self.cfg.symbol,
            direction=TradeDirection(pos_dir),
            entry_price=entry_price,
            entry_time=entry_time,
            brick_size=self.cfg.brick_size,
            lots=entry_lots,
            target_risk_usd=self.cfg.target_risk_usd,
            gate=PERGate.SIMULATED,
        )
        trade.close(
            exit_price=b_close,
            exit_time=bar_time,
            exit_reason=reason,
            friction_usd=0.0,  # Simplified
            usd_per_point=self._usd_per_point,
        )

        # Update completed trades list (thread-safe append)
        with self._state_lock:
            self._completed.append(trade)
            self._cumulative_pnl += trade.net_usd

            # Reset position state
            self._in_pos = False
            self._pos_dir = 0
            self._open_order_id = None
            self._open_client_order_id = None
            self._open_signal_id = None
            self._pending_close = False

        # Persist trade and updated state atomically
        persist_ok = self._persist_trade(trade)
        state_ok = self._save_state()
        if not persist_ok or not state_ok:
            LOG.error("[CRITICAL] Persistence failure - trade=%s state=%s", persist_ok, state_ok)

        LOG.info(
            "[EXIT] %s @ %.2f | %s | net=$%.2f",
            "LONG" if pos_dir == 1 else "SHORT",
            b_close,
            reason,
            trade.net_usd,
        )

    def _check_entry_async(
        self,
        b_close: float,
        direction: int,
        bar_time: pd.Timestamp,
        fr_val: float,
        pUU_val: float,
        pDD_val: float,
        order_queue: "queue.Queue[Optional[OrderRequest]]",
    ) -> None:
        """Check for entry signals — queues order request (non-blocking)."""
        with self._state_lock:
            is_flip = self._prev_dir is not None and direction != self._prev_dir
            self._prev_dir = direction

            if not is_flip:
                return
            if self._startup_flips_seen < int(self.cfg.startup_skip_flips):
                self._startup_flips_seen += 1
                return

            # Check filters
            if not np.isfinite(fr_val) or not np.isfinite(pUU_val) or not np.isfinite(pDD_val):
                return

        entry_ok = evaluate_entry(
            direction=direction,
            flip_rate_val=fr_val,
            pUU=max(0.0, float(pUU_val) - float(self._last_stale_penalty)),
            pDD=max(0.0, float(pDD_val) - float(self._last_stale_penalty)),
            fliprate_threshold=self.cfg.fliprate_threshold,
            markov_threshold=self._effective_markov_threshold(self._last_pe_value),
        )

        if entry_ok and bool(getattr(self.cfg, "use_permutation_entropy", True)):
            pe_thr = self._effective_pe_entry_threshold()
            if np.isfinite(self._last_pe_value) and self._last_pe_value > pe_thr:
                entry_ok = False

        if not entry_ok:
            return
        if direction == -1 and not self.cfg.allow_short:
            return

        # Calculate position size (uses self._live_equity, lock not needed for read)
        lots = self._compute_lots()
        if lots <= 0:
            return

        stop_dist = self.cfg.stop_bricks * self.cfg.brick_size
        req_price, obi, obi_buf_px = self._obi_adjusted_entry_price(
            b_close, direction, self._active_dispatcher
        )
        stop_price = req_price - stop_dist if direction == 1 else req_price + stop_dist

        with self._state_lock:
            self._signal_counter += 1
            signal_id = f"S-{self._brick_count:08d}-{self._signal_counter:06d}"

            # Mark that we're waiting for fill (prevents duplicate entries)
            self._waiting_for_fill = True

        # Queue order request (non-blocking)
        req = OrderRequest(
            signal_id=signal_id,
            direction=direction,
            price=req_price,
            stop_price=stop_price,
            lots=lots,
            bar_time=bar_time,
            brick_count=self._brick_count,
        )

        try:
            order_queue.put_nowait(req)
            LOG.info(
                "[ENTRY-QUEUED] %s %s %.3f lots @ %.2f signal_id=%s",
                "BUY" if direction == 1 else "SELL",
                self.cfg.symbol,
                lots,
                req_price,
                signal_id,
            )
            if np.isfinite(obi):
                LOG.info(
                    "[ENTRY-OBI] signal_id=%s imbalance=%.3f buffer_px=%.5f",
                    signal_id,
                    obi,
                    obi_buf_px,
                )
        except queue.Full:
            LOG.error("[ENTRY-FAILED] Order queue full — dropping signal %s", signal_id)
            with self._state_lock:
                self._waiting_for_fill = False

    # ── Internal state ────────────────────────────────────────────────────────

    def _reset_state(self) -> None:
        max_w = max(self.cfg.fliprate_window, self.cfg.markov_window, self.cfg.pe_window) + 2
        self._state_lock = threading.Lock()  # Protects all position state
        self._running = True
        self._halted = False
        self._builder = IncrementalRenkoBuilder(self.cfg.brick_size)
        self._dir_deque: deque = deque(maxlen=max_w)
        self._brick_ttf_minutes: deque = deque(maxlen=max(5, int(self.cfg.time_decay_window)))
        self._brick_count = 0
        self._stream_bars_seen = 0
        self._stream_duplicate_bars_dropped = 0
        self._flip_count = 0
        self._startup_flips_seen = 0
        self._filter_ready_count = 0
        self._last_brick_time: Optional[pd.Timestamp] = None
        self._last_ttf_minutes: float = float("nan")
        self._last_ttf_avg_minutes: float = float("nan")
        self._last_stale_penalty: float = 0.0
        self._last_activity_ratio: float = float("nan")
        self._last_pe_value: float = float("nan")
        self._last_obi_imbalance: float = float("nan")
        self._obi_imbalance_ema: float = float("nan")
        self._last_obi_buffer_px: float = 0.0
        self._current_run_dir: Optional[int] = None
        self._current_run_len: int = 0
        self._in_pos = False
        self._pos_dir = 0
        self._entry_price = 0.0
        self._entry_time: Optional[pd.Timestamp] = None
        self._entry_brick_index: Optional[int] = None
        self._entry_lots = 0.0
        self._best_price_since_entry: Optional[float] = None
        self._open_order_id: Optional[str] = None  # broker order_id for the current position
        self._open_client_order_id: Optional[str] = None
        self._open_signal_id: Optional[str] = None
        self._waiting_for_fill: bool = (
            False  # True between open_position() call and fill confirmation
        )
        self._pending_close: bool = False  # True after close sent, before confirmation
        self._prev_dir: Optional[int] = None
        self._cumulative_pnl = 0.0
        self._live_equity = self.cfg.initial_equity
        self._equity_mtm = self.cfg.initial_equity
        self._mtm_peak_equity = self.cfg.initial_equity
        self._max_drawdown_mtm_usd = 0.0
        self._max_drawdown_mtm_pct = 0.0
        self._completed: List[LiveTrade] = []
        self._signal_counter: int = 0
        self._signal_events: List[Dict[str, Any]] = []
        self._active_dispatcher: Optional[OrderDispatcher] = None
        self._spread_points_series: Optional[pd.Series] = None
        self._used_margin_usd: float = 0.0
        self._max_used_margin_usd: float = 0.0
        self._min_margin_level_pct: float = float("inf")
        self._margin_reject_count: int = 0
        self._last_eval: Dict[str, Any] = {
            "direction": "NA",
            "signal_id": "",
            "is_flip": False,
            "fr": float("nan"),
            "markov": float("nan"),
            "obi": float("nan"),
            "obi_buffer_px": 0.0,
            "entry_ok": False,
            "lots": 0.0,
            "warmup_ready": False,
            "warmup_remaining": max(int(self.cfg.min_warmup_bricks), 1),
            "startup_skip_remaining": max(int(self.cfg.startup_skip_flips), 0),
            "reason": "awaiting_bricks",
        }

    # ── Persistence ────────────────────────────────────────────────────────────

    def _init_persistence(self, persist_dir: Optional[str] = None) -> None:
        """Initialize persistence directory and load previous state if available.

        Parameters
        ----------
        persist_dir : str, optional
            Directory for state files. Defaults to .kinetra/state/<symbol>/
        """
        if persist_dir:
            self._state_dir = Path(persist_dir)
        else:
            self._state_dir = Path.home() / ".kinetra" / "state" / self.cfg.symbol

        self._state_dir.mkdir(parents=True, exist_ok=True)
        self._state_file = self._state_dir / "engine_state.json"
        self._trades_file = self._state_dir / "trades.jsonl"

        # Attempt to restore state (only for live trading resume)
        LOG.info("[PERSIST] State dir: %s", self._state_dir)

    def _save_state(self) -> bool:
        """Atomically save current engine state to disk.

        Returns
        -------
        bool
            True if save succeeded, False otherwise
        """
        if not hasattr(self, "_state_file"):
            return True  # No persistence configured, consider success

        try:
            with self._state_lock:
                state = EngineState(
                    timestamp=datetime.utcnow().isoformat() + "Z",
                    symbol=self.cfg.symbol,
                    in_pos=self._in_pos,
                    pos_dir=self._pos_dir,
                    entry_price=self._entry_price,
                    entry_time=self._entry_time.isoformat() if self._entry_time else None,
                    entry_lots=self._entry_lots,
                    open_order_id=self._open_order_id,
                    open_signal_id=self._open_signal_id,
                    brick_count=self._brick_count,
                    cumulative_pnl=self._cumulative_pnl,
                    live_equity=self._live_equity,
                    last_ref_price=getattr(self._builder, "_last_ref_price", None),
                    last_bar_time=self._last_brick_time.isoformat()
                    if self._last_brick_time
                    else None,
                )
            _atomic_write_json(self._state_file, state.to_dict())
            return True
        except Exception as exc:
            LOG.error("[PERSIST-CRITICAL] Failed to save state: %s", exc)
            return False

    def _load_state(self) -> Optional[EngineState]:
        """Load engine state from disk if available."""
        if not hasattr(self, "_state_file"):
            return None

        data = _load_json_if_exists(self._state_file)
        if data:
            LOG.info("[PERSIST] Loaded previous state from %s", self._state_file)
            return EngineState.from_dict(data)
        return None

    def _restore_state(self, state: EngineState) -> None:
        """Apply a previously persisted EngineState to the live engine fields.

        Called by :meth:`run` after :meth:`_reset_state` when a state file is
        found on disk (crash-recovery path).  The position is considered fully
        filled (``_waiting_for_fill = False``) because the fill confirmation
        was already processed before the state was last saved.
        """
        if state.symbol and state.symbol != self.cfg.symbol:
            LOG.warning(
                "[RECOVER] State symbol mismatch: file=%s engine=%s — skipping restore",
                state.symbol,
                self.cfg.symbol,
            )
            return

        with self._state_lock:
            self._in_pos = state.in_pos
            self._pos_dir = state.pos_dir
            self._brick_count = state.brick_count
            self._cumulative_pnl = state.cumulative_pnl
            self._live_equity = state.live_equity if state.live_equity > 0.0 else self._live_equity
            # A recovered position is already filled — don't block exits.
            self._waiting_for_fill = False
            self._pending_close = False

            if state.in_pos:
                # Only restore position fields when there is actually an open position.
                self._entry_price = state.entry_price
                self._entry_time = (
                    _ensure_utc(pd.Timestamp(state.entry_time)) if state.entry_time else None
                )
                self._entry_lots = state.entry_lots
                self._open_order_id = state.open_order_id
                self._open_signal_id = state.open_signal_id
            else:
                # Flat state: guarantee all position fields are cleared.
                # (Avoids ghost order_ids from a state file saved mid-close.)
                self._entry_price = 0.0
                self._entry_time = None
                self._entry_lots = 0.0
                self._open_order_id = None
                self._open_signal_id = None

        # Seed the incremental brick builder with the last known reference price
        # so it doesn't emit phantom bricks from zero before the first real bar.
        if state.last_ref_price is not None:
            try:
                self._builder._last_ref_price = float(state.last_ref_price)
            except Exception:
                pass

        LOG.warning(
            "[RECOVER] State restored: in_pos=%s pos_dir=%d equity=%.2f bricks=%d"
            " open_order_id=%s",
            state.in_pos,
            state.pos_dir,
            self._live_equity,
            self._brick_count,
            state.open_order_id or "-",
        )

    def _persist_trade(self, trade: LiveTrade) -> bool:
        """Append a completed trade to the trades log (atomic append).

        Returns
        -------
        bool
            True if persistence succeeded, False otherwise
        """
        if not hasattr(self, "_trades_file"):
            return True  # No persistence configured, consider success

        try:
            # Convert LiveTrade to dict and append as JSON line.
            # Attribute names match the LiveTrade dataclass fields exactly.
            trade_dict = {
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "entry_time": trade.entry_time.isoformat() if trade.entry_time else None,
                "exit_time": trade.exit_time.isoformat() if trade.exit_time else None,
                "direction": trade.direction.value
                if hasattr(trade.direction, "value")
                else str(trade.direction),
                "entry_price": trade.entry_price,
                "exit_price": trade.exit_price,
                "lots": trade.lots,
                "gross_usd": trade.gross_usd,
                "spread_usd": trade.spread_usd,
                "commission_usd": trade.commission_usd,
                "swap_usd": trade.swap_usd,
                "net_usd": trade.net_usd,
                "signal_id": trade.signal_id,
                "broker_ticket": trade.broker_ticket,
            }
            line = json.dumps(trade_dict, default=str) + "\n"
            # POSIX O_APPEND guarantees atomicity for writes <= PIPE_BUF on a
            # single-writer process; no staging temp file needed for append logs.
            with open(self._trades_file, "a", encoding="utf-8") as f:
                f.write(line)
            return True
        except Exception as exc:
            LOG.error("[PERSIST-CRITICAL] Failed to persist trade: %s", exc)
            # Also log trade details so it can be manually recovered
            LOG.error(
                "[PERSIST-RECOVERY] Trade data: id=%s symbol=%s entry=%s exit=%s net_pnl=%.2f",
                trade.order_id,
                trade.symbol,
                trade.entry_time,
                trade.exit_time,
                trade.net_usd,
            )
            return False

    def _clear_persisted_state(self) -> None:
        """Clear persisted state (e.g., after graceful shutdown)."""
        if hasattr(self, "_state_file") and self._state_file.exists():
            try:
                self._state_file.unlink()
                LOG.info("[PERSIST] Cleared state file")
            except Exception as exc:
                LOG.warning("[PERSIST] Failed to clear state: %s", exc)

    def set_dynamic_spread_series(self, spread_points: Optional[pd.Series]) -> None:
        """Set optional timestamp-indexed spread-points series for dynamic trade costing."""
        if spread_points is None:
            self._spread_points_series = None
            return
        if not isinstance(spread_points.index, pd.DatetimeIndex):
            raise ValueError("spread_points index must be DatetimeIndex")
        s = pd.to_numeric(spread_points, errors="coerce").dropna()
        if s.empty:
            self._spread_points_series = None
            return
        if s.index.tz is None:
            s.index = s.index.tz_localize("UTC")
        else:
            s.index = s.index.tz_convert("UTC")
        s = s.sort_index()
        s = s[~s.index.duplicated(keep="last")]
        self._spread_points_series = s

    @property
    def bricks_processed(self) -> int:
        """Total number of Renko bricks processed since last reset."""
        return int(self._brick_count)

    @property
    def duplicate_bars_dropped(self) -> int:
        """Number of out-of-order/duplicate bars dropped in streaming mode."""
        return int(self._stream_duplicate_bars_dropped)

    @property
    def stream_bars_seen(self) -> int:
        """Total streaming bars delivered to the engine callback."""
        return int(self._stream_bars_seen)

    @property
    def flips_seen(self) -> int:
        """Total Renko colour flips seen in this run."""
        return int(self._flip_count)

    @property
    def filter_ready_bricks(self) -> int:
        """Bricks where both FR and Markov values were finite."""
        return int(self._filter_ready_count)

    @property
    def last_brick_time(self) -> Optional[pd.Timestamp]:
        """UTC timestamp for the most recently processed Renko brick."""
        return self._last_brick_time

    # ── Sizing ────────────────────────────────────────────────────────────────

    def _compute_lots(self) -> float:
        cfg = self.cfg
        if cfg.sizing_mode == "static":
            raw = cfg.min_lots
        elif cfg.sizing_mode == "compounding":
            # Grow proportionally with equity: compounding_per_k lots per $1 000
            raw = max(self._live_equity, 0.0) / 1_000.0 * cfg.compounding_per_k
        else:  # risk_based
            value_per_lot = cfg.brick_size * self._usd_per_point
            if value_per_lot <= 0:
                return 0.0
            raw = cfg.target_risk_usd / value_per_lot

        # Broker-style deterministic stepping: floor to step (never exceed raw).
        # Small epsilon avoids float boundary artifacts near exact step edges.
        stepped = math.floor((raw + 1e-12) / cfg.lot_step) * cfg.lot_step
        return float(max(cfg.min_lots, min(stepped, self._lot_ceiling)))

    # ── P&L ───────────────────────────────────────────────────────────────────

    def _simulate_pnl(
        self,
        entry: float,
        exit_price: float,
        lots: float,
        direction: int,
        entry_time: Optional["pd.Timestamp"] = None,
        exit_time: Optional["pd.Timestamp"] = None,
    ) -> Tuple[float, float, float, float, float]:
        """Return (gross_usd, spread_usd, commission_usd, swap_usd, net_usd).

        Friction costs are broken out separately:
        - spread_usd: bid-ask spread cost (full round-trip)
        - commission_usd: broker commission (round-trip)
        - swap_usd: overnight carry cost, based on midnights spanned
        """
        gross_usd = (exit_price - entry) * direction * self._usd_per_point * lots
        spread_points = float(self.cfg.spread_ticks)
        if self._spread_points_series is not None:

            def _lookup(ts: Optional["pd.Timestamp"]) -> float:
                if ts is None or self._spread_points_series is None:
                    return spread_points
                t = _ensure_utc(pd.Timestamp(ts))
                v = self._spread_points_series.asof(t)
                if pd.isna(v):
                    return spread_points
                return max(float(v), 0.0)

            entry_sp = _lookup(entry_time)
            exit_sp = _lookup(exit_time)
            spread_points = max(0.0, (entry_sp + exit_sp) * 0.5)

        spread_usd = (
            spread_points
            * self.cfg.usd_per_tick
            * lots
            * max(float(getattr(self.cfg, "spread_sides", 1.0)), 0.0)
        )
        commission_usd = self.cfg.commission_per_lot * lots

        # Swap: charged per midnight spanned; triple-swap day counts as 3×.
        # Live trades carry actual broker-applied swap so only apply here in
        # backtest / paper mode (when times are available).
        eff_days = _count_effective_swap_days(entry_time, exit_time, self.cfg.triple_swap_day)
        if eff_days > 0:
            daily_rate = (
                self.cfg.swap_long_usd_per_day
                if direction == 1
                else self.cfg.swap_short_usd_per_day
            )
            swap_usd = daily_rate * lots * eff_days
        else:
            swap_usd = 0.0

        total_friction = spread_usd + commission_usd + swap_usd
        return gross_usd, spread_usd, commission_usd, swap_usd, gross_usd - total_friction

    def _required_margin_usd(self, lots: float, price: float) -> float:
        """Required margin in USD for a position at given price."""
        notional_usd = (
            max(float(lots), 0.0) * max(float(price), 0.0) * max(float(self._usd_per_point), 0.0)
        )
        mr = max(float(getattr(self.cfg, "margin_rate", 0.0)), 0.0)
        return notional_usd * mr

    def _refresh_margin_state(self, mark_price: float) -> None:
        """Recompute used margin and margin level at current mark."""
        if self._in_pos and self._entry_lots > 0.0:
            self._used_margin_usd = self._required_margin_usd(self._entry_lots, float(mark_price))
        else:
            self._used_margin_usd = 0.0
        if self._used_margin_usd > self._max_used_margin_usd:
            self._max_used_margin_usd = self._used_margin_usd
        if self._used_margin_usd > 1e-12:
            ml = (float(self._equity_mtm) / float(self._used_margin_usd)) * 100.0
            self._min_margin_level_pct = min(float(self._min_margin_level_pct), float(ml))

    def _update_mtm_drawdown(
        self,
        mark_price: float,
        mark_time: Optional["pd.Timestamp"] = None,
    ) -> None:
        """Update mark-to-market equity and drawdown using current mark price."""
        closed_equity = float(self.cfg.initial_equity + self._cumulative_pnl)
        mtm_equity = closed_equity
        if self._in_pos and self._entry_lots > 0.0:
            _, _, _, _, unrealized_net = self._simulate_pnl(
                self._entry_price,
                float(mark_price),
                float(self._entry_lots),
                int(self._pos_dir),
                entry_time=self._entry_time,
                exit_time=mark_time,
            )
            mtm_equity = closed_equity + float(unrealized_net)

        self._equity_mtm = float(mtm_equity)
        if self._equity_mtm > self._mtm_peak_equity:
            self._mtm_peak_equity = self._equity_mtm
        self._refresh_margin_state(float(mark_price))

        drawdown_usd = max(0.0, float(self._mtm_peak_equity - self._equity_mtm))
        if drawdown_usd > self._max_drawdown_mtm_usd:
            self._max_drawdown_mtm_usd = drawdown_usd
            self._max_drawdown_mtm_pct = (
                (drawdown_usd / self._mtm_peak_equity) * 100.0
                if self._mtm_peak_equity > 0.0
                else 0.0
            )

    def _apply_entry_fill_price(self, signal_price: float, direction: int) -> float:
        """Apply adverse entry fill offset in price units when enabled."""
        if not bool(getattr(self.cfg, "conservative_fills", False)):
            return float(signal_price)
        slip_bricks = max(float(getattr(self.cfg, "entry_slip_bricks", 0.0)), 0.0)
        slip_px = slip_bricks * float(self.cfg.brick_size)
        return float(signal_price) + int(direction) * slip_px

    def _apply_exit_fill_price(self, signal_price: float, reason: str, pos_dir: int) -> float:
        """Apply adverse exit fill offset (slip + event-specific penalties)."""
        if not bool(getattr(self.cfg, "conservative_fills", False)):
            return float(signal_price)
        slip_bricks = max(float(getattr(self.cfg, "exit_slip_bricks", 0.0)), 0.0)
        if reason in ("stop", "trailing_stop"):
            slip_bricks += max(float(getattr(self.cfg, "stop_worst_case_bricks", 0.0)), 0.0)
        elif reason == "colour_change":
            slip_bricks += max(float(getattr(self.cfg, "flip_trade_through_bricks", 0.0)), 0.0)
        slip_px = slip_bricks * float(self.cfg.brick_size)
        # Long exits worsen lower; short exits worsen higher.
        return float(signal_price) - int(pos_dir) * slip_px

    # ── Filters (streaming path) ──────────────────────────────────────────────

    def _streaming_filter_vals(self) -> Tuple[float, float, float]:
        """Compute filter values from the rolling direction deque."""
        arr = np.array(self._dir_deque, dtype=np.int8)
        n = len(arr)
        fw, mw = self.cfg.fliprate_window, self.cfg.markov_window
        nan = float("nan")

        if n < fw + 1:
            return nan, nan, nan

        try:
            fr_arr = flip_rate(arr, fw, min_periods=fw)
            fr_val = float(fr_arr[-1]) if np.isfinite(fr_arr[-1]) else nan
        except (ValueError, IndexError):
            fr_val = nan

        if n < mw + 1:
            return fr_val, nan, nan

        try:
            pUU_arr, pDD_arr = markov_stickiness(arr, mw, min_periods=mw)
            pUU = float(pUU_arr[-1]) if np.isfinite(pUU_arr[-1]) else nan
            pDD = float(pDD_arr[-1]) if np.isfinite(pDD_arr[-1]) else nan
        except (ValueError, IndexError):
            pUU, pDD = nan, nan

        return fr_val, pUU, pDD

    def _update_timing_metrics(self, bar_time: pd.Timestamp) -> Tuple[float, float, float, float]:
        """Update brick formation timing metrics.

        Returns: (ttf_minutes, ttf_avg_minutes, stale_penalty, activity_ratio)
        """
        ttf_min = float("nan")
        ttf_avg = float("nan")
        stale_penalty = 0.0
        activity_ratio = float("nan")

        if self._last_brick_time is not None:
            dt_min = (bar_time - self._last_brick_time).total_seconds() / 60.0
            if np.isfinite(dt_min) and dt_min > 0:
                ttf_min = float(dt_min)
                self._brick_ttf_minutes.append(ttf_min)

        self._last_brick_time = bar_time

        if self._brick_ttf_minutes:
            ttf_avg = float(np.mean(self._brick_ttf_minutes))
            if np.isfinite(ttf_min) and ttf_min > 0 and ttf_avg > 0:
                activity_ratio = float(ttf_avg / ttf_min)

        if (
            bool(getattr(self.cfg, "use_time_decay", True))
            and np.isfinite(ttf_min)
            and np.isfinite(ttf_avg)
            and ttf_avg > 0
        ):
            stale_factor = max(float(getattr(self.cfg, "stale_brick_factor", 2.0)), 1.0)
            if ttf_min > stale_factor * ttf_avg:
                ratio = float(ttf_min / (ttf_avg * stale_factor))
                excess = max(0.0, ratio - 1.0)
                max_pen = max(float(getattr(self.cfg, "markov_stale_penalty", 0.15)), 0.0)
                stale_penalty = float(min(max_pen, max_pen * excess))

        self._last_ttf_minutes = float(ttf_min)
        self._last_ttf_avg_minutes = float(ttf_avg)
        self._last_stale_penalty = float(stale_penalty)
        self._last_activity_ratio = float(activity_ratio)
        return float(ttf_min), float(ttf_avg), float(stale_penalty), float(activity_ratio)

    def _current_pe_value(self) -> float:
        """Current permutation entropy over the latest direction window."""
        if not bool(getattr(self.cfg, "use_permutation_entropy", True)):
            return float("nan")
        arr = np.array(self._dir_deque, dtype=np.int8)
        w = max(int(getattr(self.cfg, "pe_window", 50)), 3)
        order = max(int(getattr(self.cfg, "pe_order", 3)), 2)
        if len(arr) < w:
            return float("nan")
        pe = permutation_entropy(arr[-w:], order=order)
        self._last_pe_value = float(pe) if np.isfinite(pe) else float("nan")
        return self._last_pe_value

    def _effective_markov_threshold(self, pe_val: float) -> float:
        thr = float(self.cfg.markov_threshold)
        if not bool(getattr(self.cfg, "use_permutation_entropy", True)) or not np.isfinite(pe_val):
            return thr
        low_pe = float(getattr(self.cfg, "pe_low_entropy", 0.30))
        if pe_val <= low_pe:
            thr -= float(getattr(self.cfg, "pe_markov_relax", 0.03))
        elif pe_val >= float(getattr(self.cfg, "pe_entry_threshold", 0.70)):
            thr += float(getattr(self.cfg, "pe_markov_tighten", 0.05))
        return float(np.clip(thr, 0.01, 0.99))

    def _effective_pe_entry_threshold(self) -> float:
        base = float(getattr(self.cfg, "pe_entry_threshold", 0.70))
        if not bool(getattr(self.cfg, "pe_vol_adjust", True)) or not np.isfinite(
            self._last_activity_ratio
        ):
            return float(np.clip(base, 0.05, 0.99))
        sens = float(np.clip(getattr(self.cfg, "pe_vol_sensitivity", 0.50), 0.0, 2.0))
        # Slow regime (activity_ratio < 1) -> lower threshold (stricter).
        dyn = base + (self._last_activity_ratio - 1.0) * 0.20 * sens
        return float(np.clip(dyn, 0.05, 0.99))

    def _extract_imbalance_from_depth(self, depth: Any, levels: int) -> float:
        """Extract normalized imbalance in [-1, 1] from flexible depth payloads."""
        if depth is None:
            return float("nan")
        if isinstance(depth, (int, float)) and np.isfinite(depth):
            return float(np.clip(float(depth), -1.0, 1.0))
        if isinstance(depth, dict):
            bid = depth.get("total_bid_volume")
            ask = depth.get("total_ask_volume")
            if bid is None or ask is None:
                bids = depth.get("bids")
                asks = depth.get("asks")
                if isinstance(bids, list) and isinstance(asks, list):
                    use_n = max(int(levels), 1)
                    bid = sum(
                        float((x or {}).get("volume", (x or {}).get("volume_real", 0.0)))
                        for x in bids[:use_n]
                    )
                    ask = sum(
                        float((x or {}).get("volume", (x or {}).get("volume_real", 0.0)))
                        for x in asks[:use_n]
                    )
            try:
                b = float(bid)
                a = float(ask)
                denom = b + a
                if denom > 0 and np.isfinite(denom):
                    return float(np.clip((b - a) / denom, -1.0, 1.0))
            except Exception:
                return float("nan")
        return float("nan")

    def _fetch_order_book_imbalance(
        self, dispatcher: Optional[OrderDispatcher], symbol: str, levels: int
    ) -> float:
        """Best-effort order book imbalance fetch with graceful fallback."""
        if dispatcher is None:
            return float("nan")
        try:
            fn = getattr(dispatcher, "get_order_book_imbalance", None)
            if callable(fn):
                return self._extract_imbalance_from_depth(fn(symbol, levels=levels), levels)
        except Exception:
            pass
        try:
            connector = getattr(dispatcher, "_connector", None)
            if connector is None:
                return float("nan")
            fn = getattr(connector, "get_order_book_imbalance", None)
            if callable(fn):
                return self._extract_imbalance_from_depth(fn(symbol, levels=levels), levels)
            fn = getattr(connector, "get_market_depth", None)
            if callable(fn):
                return self._extract_imbalance_from_depth(fn(symbol), levels)
        except Exception:
            return float("nan")
        return float("nan")

    def _obi_adjusted_entry_price(
        self,
        price: float,
        direction: int,
        dispatcher: Optional[OrderDispatcher],
    ) -> Tuple[float, float, float]:
        """Apply OBI-driven slippage buffer to entry price.

        Returns (adjusted_price, imbalance_used, buffer_px).
        """
        base = float(price)
        if not bool(getattr(self.cfg, "use_obi_slippage_buffer", False)):
            return base, float("nan"), 0.0
        levels = max(int(getattr(self.cfg, "obi_levels", 5)), 1)
        imb = self._fetch_order_book_imbalance(dispatcher, self.cfg.symbol, levels)
        if np.isfinite(imb):
            alpha = float(np.clip(getattr(self.cfg, "obi_ema_alpha", 0.30), 0.0, 1.0))
            if np.isfinite(self._obi_imbalance_ema):
                self._obi_imbalance_ema = (
                    alpha * float(imb) + (1.0 - alpha) * self._obi_imbalance_ema
                )
            else:
                self._obi_imbalance_ema = float(imb)
        eff_imb = self._obi_imbalance_ema if np.isfinite(self._obi_imbalance_ema) else float("nan")
        if not np.isfinite(eff_imb):
            self._last_obi_imbalance = float("nan")
            self._last_obi_buffer_px = 0.0
            return base, float("nan"), 0.0
        max_buf_px = max(float(getattr(self.cfg, "obi_max_buffer_bricks", 0.25)), 0.0) * float(
            self.cfg.brick_size
        )
        directional = max(0.0, eff_imb) if direction == 1 else max(0.0, -eff_imb)
        buf_px = float(np.clip(directional * max_buf_px, 0.0, max_buf_px))
        adj = base + (float(direction) * buf_px)
        self._last_obi_imbalance = float(eff_imb)
        self._last_obi_buffer_px = buf_px
        return float(adj), float(eff_imb), float(buf_px)

    # ── Core brick logic ──────────────────────────────────────────────────────

    def _process_brick(
        self,
        b_close: float,
        direction: int,
        bar_time: pd.Timestamp,
        fr_val: float,
        pUU_val: float,
        pDD_val: float,
        dispatcher: Optional[OrderDispatcher] = None,
    ) -> None:
        """Core strategy logic for one brick. Mutates engine state."""
        self._dir_deque.append(direction)
        self._update_timing_metrics(bar_time)
        pe_val = self._current_pe_value()
        self._brick_count += 1
        stop_dist = self.cfg.stop_bricks * self.cfg.brick_size
        warmup_needed = max(int(self.cfg.min_warmup_bricks), 1)
        warmup_remaining = max(0, warmup_needed - self._brick_count)
        warmup_ready = warmup_remaining == 0
        markov_raw = pUU_val if direction == 1 else pDD_val
        markov_adj = (
            max(0.0, float(markov_raw) - float(self._last_stale_penalty))
            if np.isfinite(markov_raw)
            else float("nan")
        )
        markov_thr_eff = self._effective_markov_threshold(pe_val)
        pe_thr_eff = self._effective_pe_entry_threshold()
        self._last_eval = {
            "direction": "BUY" if direction == 1 else "SELL",
            "signal_id": "",
            "is_flip": bool(self._prev_dir is not None and direction != self._prev_dir),
            "fr": fr_val,
            "markov": markov_adj,
            "pe": pe_val,
            "obi": self._last_obi_imbalance,
            "obi_buffer_px": self._last_obi_buffer_px,
            "markov_threshold_eff": markov_thr_eff,
            "pe_threshold_eff": pe_thr_eff,
            "ttf_min": self._last_ttf_minutes,
            "ttf_avg_min": self._last_ttf_avg_minutes,
            "stale_penalty": self._last_stale_penalty,
            "entry_ok": False,
            "lots": 0.0,
            "warmup_ready": warmup_ready,
            "warmup_remaining": warmup_remaining,
            "startup_skip_remaining": max(
                0, int(self.cfg.startup_skip_flips) - int(self._startup_flips_seen)
            ),
            "reason": "in_position" if self._in_pos else "no_flip_or_filters_not_ready",
        }
        if bool(self._last_eval.get("is_flip", False)):
            self._flip_count += 1
        if np.isfinite(fr_val) and np.isfinite(pUU_val) and np.isfinite(pDD_val):
            self._filter_ready_count += 1
        ended_run_len = 0
        if self._current_run_dir is None:
            self._current_run_dir = direction
            self._current_run_len = 1
        elif direction == self._current_run_dir:
            self._current_run_len += 1
        else:
            ended_run_len = self._current_run_len
            self._current_run_dir = direction
            self._current_run_len = 1

        # ── Brick logging ─────────────────────────────────────────────────────
        dir_str = "UP" if direction == 1 else "DN"
        pos_str = (
            f" [IN {('LONG' if self._pos_dir == 1 else 'SHORT') if self._in_pos else 'FLAT'}]"
            if self._in_pos
            else ""
        )
        if not self._quiet_mode:
            LOG.info(
                "Brick: %s %s @ %.2f | FR=%.3f pUU=%.3f pDD=%.3f%s",
                bar_time.strftime("%H:%M:%S"),
                dir_str,
                b_close,
                fr_val if np.isfinite(fr_val) else float("nan"),
                pUU_val if np.isfinite(pUU_val) else float("nan"),
                pDD_val if np.isfinite(pDD_val) else float("nan"),
                pos_str,
            )

        # ── Exits ────────────────────────────────────────────────────────────
        if self._in_pos:
            # Track best favorable excursion while position is open.
            if self._best_price_since_entry is None:
                self._best_price_since_entry = b_close
            elif self._pos_dir == 1:
                self._best_price_since_entry = max(
                    float(self._best_price_since_entry), float(b_close)
                )
            else:
                self._best_price_since_entry = min(
                    float(self._best_price_since_entry), float(b_close)
                )

            held_bricks = (
                self._brick_count - self._entry_brick_index
                if self._entry_brick_index is not None
                else 0
            )
            trail_stop_price: Optional[float] = None
            trail_after = max(int(getattr(self.cfg, "trailing_mfe_after_bricks", 0)), 0)
            trail_frac = float(
                np.clip(float(getattr(self.cfg, "trailing_mfe_fraction", 0.0)), 0.0, 1.0)
            )
            trailing_enabled = trail_after > 0 and trail_frac > 0.0
            if (
                trailing_enabled
                and held_bricks >= trail_after
                and self._best_price_since_entry is not None
            ):
                if self._pos_dir == 1:
                    mfe_pts = max(
                        0.0, float(self._best_price_since_entry) - float(self._entry_price)
                    )
                    trail_stop_price = float(self._entry_price) + trail_frac * mfe_pts
                else:
                    mfe_pts = max(
                        0.0, float(self._entry_price) - float(self._best_price_since_entry)
                    )
                    trail_stop_price = float(self._entry_price) - trail_frac * mfe_pts

            if (
                dispatcher is not None
                and self._open_order_id is not None
                and not self._waiting_for_fill
            ):
                broker_open = self._is_broker_position_open(dispatcher, self._open_order_id)
                if broker_open is False:
                    LOG.warning(
                        "[SYNC] Broker already flat for order_id=%s signal_id=%s; "
                        "closing local state as broker_flat_sync at %.2f",
                        self._open_order_id,
                        self._open_signal_id or "-",
                        b_close,
                    )
                    self._force_close(
                        b_close,
                        bar_time,
                        reason="broker_flat_sync",
                        dispatcher=dispatcher,
                    )
                    self._update_mtm_drawdown(b_close, bar_time)
                    self._last_eval["reason"] = "broker_flat_sync"
                    self._prev_dir = direction
                    return

            if (
                bool(getattr(self.cfg, "enforce_margin_checks", False))
                and self._used_margin_usd > 1e-12
            ):
                margin_level = (float(self._equity_mtm) / float(self._used_margin_usd)) * 100.0
                if margin_level <= float(getattr(self.cfg, "stop_out_level_pct", 50.0)):
                    LOG.warning(
                        "[MARGIN STOP-OUT] signal_id=%s margin_level=%.2f%% stop_out=%.2f%% used_margin=$%.2f equity_mtm=$%.2f",
                        self._open_signal_id or "-",
                        margin_level,
                        float(getattr(self.cfg, "stop_out_level_pct", 50.0)),
                        self._used_margin_usd,
                        self._equity_mtm,
                    )
                    self._force_close(
                        b_close,
                        bar_time,
                        reason="stop_out",
                        dispatcher=dispatcher,
                    )
                    self._update_mtm_drawdown(b_close, bar_time)
                    self._last_eval["reason"] = "stop_out"
                    self._prev_dir = direction
                    return
            static_stop = (
                self._entry_price - stop_dist
                if self._pos_dir == 1
                else self._entry_price + stop_dist
            )
            effective_stop = static_stop
            if trail_stop_price is not None:
                if self._pos_dir == 1:
                    effective_stop = max(static_stop, trail_stop_price)
                else:
                    effective_stop = min(static_stop, trail_stop_price)

            stop_hit = (self._pos_dir == 1 and b_close <= effective_stop) or (
                self._pos_dir == -1 and b_close >= effective_stop
            )
            entropy_exit = bool(
                bool(getattr(self.cfg, "use_permutation_entropy", True))
                and np.isfinite(pe_val)
                and pe_val >= float(getattr(self.cfg, "pe_exit_threshold", 0.85))
            )
            trail_hit = trail_stop_price is not None and (
                (self._pos_dir == 1 and b_close <= trail_stop_price)
                or (self._pos_dir == -1 and b_close >= trail_stop_price)
            )
            colour_change = direction != self._pos_dir
            colour_change_exit = colour_change and not self._waiting_for_fill

            if colour_change and self._waiting_for_fill:
                LOG.info(
                    "[EXIT-GUARD] colour_change ignored while awaiting fill | signal_id=%s order_id=%s client_order_id=%s brick=%d time=%s pos_dir=%s brick_dir=%s entry=%.2f now=%.2f",
                    self._open_signal_id or "-",
                    self._open_order_id or "-",
                    self._open_client_order_id or "-",
                    self._brick_count,
                    bar_time.isoformat(),
                    "LONG" if self._pos_dir == 1 else "SHORT",
                    "UP" if direction == 1 else "DN",
                    self._entry_price,
                    b_close,
                )

            if stop_hit or colour_change_exit or entropy_exit:
                reason = (
                    "entropy_exit"
                    if entropy_exit
                    else (
                        "trailing_stop" if trail_hit else ("stop" if stop_hit else "colour_change")
                    )
                )
                exit_price = self._apply_exit_fill_price(b_close, reason, self._pos_dir)
                held_bricks = (
                    self._brick_count - self._entry_brick_index
                    if self._entry_brick_index is not None
                    else -1
                )
                LOG.info(
                    "[EXIT-TRIGGER] reason=%s signal_id=%s order_id=%s client_order_id=%s held_bricks=%d brick=%d time=%s pos_dir=%s brick_dir=%s entry=%.2f stop=%.2f now=%.2f waiting_for_fill=%s",
                    reason,
                    self._open_signal_id or "-",
                    self._open_order_id or "-",
                    self._open_client_order_id or "-",
                    held_bricks,
                    self._brick_count,
                    bar_time.isoformat(),
                    "LONG" if self._pos_dir == 1 else "SHORT",
                    "UP" if direction == 1 else "DN",
                    self._entry_price,
                    effective_stop,
                    b_close,
                    self._waiting_for_fill,
                )

                # Live mode: only finalize local close after broker close succeeds.
                if dispatcher is not None and self._open_order_id is not None:
                    try:
                        close_result = dispatcher.close_position(
                            symbol=self.cfg.symbol,
                            order_id=self._open_order_id,
                            price=b_close,
                            lots=self._entry_lots,
                            comment=reason,
                        )
                    except Exception as exc:
                        LOG.error(
                            "Close dispatch raised for order_id=%s: %s",
                            self._open_order_id,
                            exc,
                        )
                        self._last_eval["reason"] = "close_dispatch_error"
                        return

                    if not close_result or not getattr(close_result, "success", False):
                        LOG.error(
                            "Close dispatch failed for order_id=%s client_order_id=%s: %s",
                            self._open_order_id,
                            (getattr(close_result, "raw", {}) or {}).get("client_order_id")
                            or self._open_client_order_id
                            or "-",
                            getattr(close_result, "error", "unknown_error"),
                        )
                        self._last_eval["reason"] = "close_dispatch_failed"
                        return

                    filled = getattr(close_result, "filled_price", None)
                    if filled is not None:
                        exit_price = float(filled)
                    LOG.info(
                        "[EXIT-DISPATCH] success signal_id=%s order_id=%s requested=%.2f filled=%s raw=%s",
                        self._open_signal_id or "-",
                        self._open_order_id or "-",
                        b_close,
                        f"{exit_price:.2f}",
                        getattr(close_result, "raw", None),
                    )

                gross_usd, spread_usd, commission_usd, swap_usd, net_usd = self._simulate_pnl(
                    self._entry_price,
                    exit_price,
                    self._entry_lots,
                    self._pos_dir,
                    entry_time=self._entry_time,
                    exit_time=bar_time,
                )
                # Use broker order_id when available — ties record to execution log
                trade_id = self._open_order_id or f"T-{len(self._completed) + 1:06d}"
                trade = LiveTrade(
                    trade_id=trade_id,
                    symbol=self.cfg.symbol,
                    direction=TradeDirection(self._pos_dir),
                    entry_price=self._entry_price,
                    entry_time=self._entry_time,
                    brick_size=self.cfg.brick_size,
                    lots=self._entry_lots,
                    target_risk_usd=self.cfg.target_risk_usd,
                    gate=PERGate.SIMULATED,
                    signal_id=self._open_signal_id or "",
                    broker_ticket=self._open_order_id or "",
                )
                trade.close(
                    exit_price=exit_price,
                    exit_time=bar_time,
                    exit_reason=reason,
                    friction_usd=float(spread_usd + commission_usd + swap_usd),
                    usd_per_point=self._usd_per_point,
                    spread_usd=float(spread_usd),
                    commission_usd=float(commission_usd),
                    swap_usd=float(swap_usd),
                )
                if reason == "colour_change" and ended_run_len > 0:
                    trade.trend_run_bricks = int(ended_run_len)
                    try:
                        trade.run_capture = min(
                            max(float(trade.n_bricks_held) / float(ended_run_len), 0.0), 1.0
                        )
                    except Exception:
                        trade.run_capture = 0.0
                self._completed.append(trade)
                self._cumulative_pnl += trade.net_usd
                self._live_equity = self.cfg.initial_equity + self._cumulative_pnl

                # Log position close
                close_dir = "LONG" if self._pos_dir == 1 else "SHORT"
                LOG.info(
                    "Close: %s @ %.2f | %s | held_bricks=%d trend_run=%s capture=%s signal_id=%s order_id=%s client_order_id=%s | gross=$%.2f spread=$%.2f comm=$%.2f net=$%.2f | equity=$%.2f",
                    close_dir,
                    exit_price,
                    reason,
                    held_bricks,
                    (
                        str(getattr(trade, "trend_run_bricks", "-"))
                        if reason == "colour_change"
                        else "-"
                    ),
                    (
                        f"{100.0 * float(getattr(trade, 'run_capture', 0.0)):.1f}%"
                        if reason == "colour_change"
                        else "-"
                    ),
                    self._open_signal_id or "-",
                    self._open_order_id or "-",
                    self._open_client_order_id or "-",
                    gross_usd,
                    spread_usd,
                    commission_usd,
                    net_usd,
                    self._live_equity,
                )

                # Sync to real broker equity when available (live mode).
                # PaperDispatcher.get_equity() returns None — keeps simulated value.
                if dispatcher is not None:
                    broker_eq = dispatcher.get_equity()
                    if broker_eq is not None:
                        self._live_equity = broker_eq
                self._in_pos = False
                self._open_order_id = None
                self._open_client_order_id = None
                self._open_signal_id = None
                self._entry_brick_index = None
                self._waiting_for_fill = False

        # ── Entries ──────────────────────────────────────────────────────────
        if not self._in_pos:
            is_flip = self._prev_dir is not None and direction != self._prev_dir
            if not is_flip:
                self._last_eval["reason"] = "no_flip"
            if is_flip and self._startup_flips_seen < int(self.cfg.startup_skip_flips):
                self._startup_flips_seen += 1
                self._last_eval["startup_skip_remaining"] = max(
                    0, int(self.cfg.startup_skip_flips) - int(self._startup_flips_seen)
                )
                self._last_eval["reason"] = "startup_skip"
            elif is_flip and not warmup_ready:
                self._last_eval["reason"] = "warmup"
            elif is_flip and np.isfinite(fr_val) and np.isfinite(pUU_val) and np.isfinite(pDD_val):
                self._signal_counter += 1
                signal_id = f"S-{self._brick_count:08d}-{self._signal_counter:06d}"
                self._last_eval["signal_id"] = signal_id
                entry_ok = evaluate_entry(
                    direction=direction,
                    flip_rate_val=fr_val,
                    pUU=max(0.0, float(pUU_val) - float(self._last_stale_penalty)),
                    pDD=max(0.0, float(pDD_val) - float(self._last_stale_penalty)),
                    fliprate_threshold=self.cfg.fliprate_threshold,
                    markov_threshold=markov_thr_eff,
                )
                if (
                    entry_ok
                    and bool(getattr(self.cfg, "use_permutation_entropy", True))
                    and np.isfinite(pe_val)
                    and pe_val > pe_thr_eff
                ):
                    entry_ok = False

                # Entry evaluation logging
                entry_dir = "BUY" if direction == 1 else "SELL"
                status = "✓ PASS" if entry_ok else "✗ FAIL"
                LOG.info(
                    "Entry eval [%s]: FR=%.3f M=%.3f PE=%.3f (M_thr=%.3f PE_thr=%.3f stale_pen=%.3f) → %s  (price=%.2f)",
                    entry_dir,
                    fr_val,
                    max(
                        0.0,
                        float(pUU_val if direction == 1 else pDD_val)
                        - float(self._last_stale_penalty),
                    ),
                    pe_val if np.isfinite(pe_val) else float("nan"),
                    markov_thr_eff,
                    pe_thr_eff,
                    float(self._last_stale_penalty),
                    status,
                    b_close,
                )
                self._last_eval["entry_ok"] = bool(entry_ok)
                self._last_eval["reason"] = "entry_pass" if entry_ok else "gate_reject"
                signal_event: Dict[str, Any] = {
                    "signal_id": signal_id,
                    "symbol": self.cfg.symbol,
                    "time": bar_time.isoformat(),
                    "brick_index": int(self._brick_count),
                    "direction": int(direction),
                    "price": float(b_close),
                    "flip_rate": float(fr_val) if np.isfinite(fr_val) else None,
                    "markov": (
                        float(pUU_val if direction == 1 else pDD_val)
                        if np.isfinite(pUU_val if direction == 1 else pDD_val)
                        else None
                    ),
                    "entry_ok": bool(entry_ok),
                    "reason": str(self._last_eval["reason"]),
                    "requested_lots": 0.0,
                    "order_success": False,
                    "order_id": None,
                    "client_order_id": None,
                    "order_error": None,
                    "order_raw": None,
                    "obi_imbalance": None,
                    "obi_buffer_px": 0.0,
                }

                if entry_ok and (direction == 1 or self.cfg.allow_short):
                    lots = self._compute_lots()
                    self._last_eval["lots"] = float(lots)
                    signal_event["requested_lots"] = float(lots)
                    req_margin = self._required_margin_usd(lots, b_close)
                    free_margin = float(self._equity_mtm - self._used_margin_usd)
                    signal_event["required_margin_usd"] = float(req_margin)
                    signal_event["free_margin_usd"] = float(free_margin)
                    if bool(getattr(self.cfg, "enforce_margin_checks", False)) and req_margin > (
                        free_margin + 1e-9
                    ):
                        self._margin_reject_count += 1
                        self._last_eval["reason"] = "insufficient_margin"
                        signal_event["entry_ok"] = False
                        signal_event["reason"] = "insufficient_margin"
                        self._signal_events.append(signal_event)
                        self._prev_dir = direction
                        self._update_mtm_drawdown(b_close, bar_time)
                        return
                    if lots > 0:
                        requested_price, obi_used, obi_buf_px = self._obi_adjusted_entry_price(
                            b_close, direction, dispatcher
                        )
                        signal_event["obi_imbalance"] = (
                            float(obi_used) if np.isfinite(obi_used) else None
                        )
                        signal_event["obi_buffer_px"] = float(obi_buf_px)
                        # Mark that we're waiting for fill — skip color_change exits until confirmed
                        self._waiting_for_fill = True
                        self._in_pos = True
                        self._pos_dir = direction
                        self._entry_price = self._apply_entry_fill_price(requested_price, direction)
                        self._entry_time = bar_time
                        self._entry_brick_index = self._brick_count
                        self._entry_lots = lots
                        self._best_price_since_entry = b_close
                        LOG.warning(
                            "[ENTRY] %s %s %.3f lots @ $%.2f (SL: $%.2f) signal_id=%s brick=%d time=%s obi=%s obi_buf_px=%.5f",
                            entry_dir,
                            self.cfg.symbol,
                            lots,
                            self._entry_price,
                            self._entry_price - stop_dist,
                            signal_id,
                            self._brick_count,
                            bar_time.isoformat(),
                            (f"{float(obi_used):.3f}" if np.isfinite(obi_used) else "n/a"),
                            float(obi_buf_px),
                        )
                        if dispatcher is not None:
                            stop_price = (
                                requested_price - stop_dist
                                if direction == 1
                                else requested_price + stop_dist
                            )
                            result = dispatcher.open_position(
                                symbol=self.cfg.symbol,
                                direction=TradeDirection(direction),
                                lots=lots,
                                price=requested_price,
                                stop_price=stop_price,
                            )
                            # Store broker order_id for close call + audit trail
                            if result and result.success:
                                self._open_order_id = result.order_id
                                self._open_client_order_id = (getattr(result, "raw", {}) or {}).get(
                                    "client_order_id"
                                )
                                self._open_signal_id = signal_id
                                self._waiting_for_fill = False
                                self._last_eval["reason"] = "entered"
                                LOG.info(
                                    "[ENTRY-FILLED] signal_id=%s order_id=%s client_order_id=%s dir=%s lots=%.3f req=%.2f fill=%s",
                                    signal_id,
                                    result.order_id,
                                    self._open_client_order_id or "-",
                                    entry_dir,
                                    lots,
                                    b_close,
                                    (
                                        f"{float(result.filled_price):.2f}"
                                        if getattr(result, "filled_price", None) is not None
                                        else "n/a"
                                    ),
                                )
                                signal_event["order_success"] = True
                                signal_event["order_id"] = result.order_id
                                signal_event["client_order_id"] = self._open_client_order_id
                                signal_event["order_raw"] = getattr(result, "raw", None)
                            else:
                                self._in_pos = False
                                self._pos_dir = 0
                                self._entry_price = 0.0
                                self._entry_time = None
                                self._entry_brick_index = None
                                self._entry_lots = 0.0
                                self._best_price_since_entry = None
                                self._open_order_id = None
                                self._open_client_order_id = None
                                self._open_signal_id = None
                                self._waiting_for_fill = False
                                self._last_eval["reason"] = "order_submit_failed"
                                LOG.error(
                                    "[ENTRY-FAILED] signal_id=%s dir=%s lots=%.3f error=%s",
                                    signal_id,
                                    entry_dir,
                                    lots,
                                    getattr(result, "error", None) if result else "no_result",
                                )
                                signal_event["order_error"] = (
                                    getattr(result, "error", None) if result else "no_result"
                                )
                                signal_event["order_raw"] = (
                                    getattr(result, "raw", None) if result else None
                                )
                        else:
                            self._open_signal_id = signal_id
                            self._waiting_for_fill = False
                            LOG.info(
                                "[ENTRY-SIM] signal_id=%s dir=%s lots=%.3f price=%.2f",
                                signal_id,
                                entry_dir,
                                lots,
                                b_close,
                            )
                            signal_event["order_success"] = True
                            signal_event["order_id"] = "BACKTEST"
                            signal_event["client_order_id"] = "SIMULATED"
                            signal_event["order_raw"] = {"mode": "backtest"}
                    else:
                        self._last_eval["reason"] = "sizing_zero"
                        signal_event["reason"] = "sizing_zero"
                self._signal_events.append(signal_event)
            elif is_flip:
                self._last_eval["reason"] = "filters_not_ready"

        self._prev_dir = direction
        self._update_mtm_drawdown(b_close, bar_time)

    def _force_close(
        self,
        price: float,
        time: pd.Timestamp,
        *,
        reason: str = "end_of_data",
        dispatcher: Optional[OrderDispatcher] = None,
    ) -> None:
        """Close the open position at end-of-data."""
        gross_usd, spread_usd, commission_usd, swap_usd, net_usd = self._simulate_pnl(
            self._entry_price,
            price,
            self._entry_lots,
            self._pos_dir,
            entry_time=self._entry_time,
            exit_time=time,
        )
        trade = LiveTrade(
            trade_id=self._open_order_id or f"T-{len(self._completed) + 1:06d}",
            symbol=self.cfg.symbol,
            direction=TradeDirection(self._pos_dir),
            entry_price=self._entry_price,
            entry_time=self._entry_time,
            brick_size=self.cfg.brick_size,
            lots=self._entry_lots,
            target_risk_usd=self.cfg.target_risk_usd,
            gate=PERGate.SIMULATED,
            signal_id=self._open_signal_id or "",
            broker_ticket=self._open_order_id or "",
        )
        trade.close(
            exit_price=price,
            exit_time=time,
            exit_reason=reason,
            friction_usd=float(spread_usd + commission_usd + swap_usd),
            usd_per_point=self._usd_per_point,
            spread_usd=float(spread_usd),
            commission_usd=float(commission_usd),
            swap_usd=float(swap_usd),
        )
        self._completed.append(trade)
        self._cumulative_pnl += trade.net_usd
        self._live_equity = self.cfg.initial_equity + self._cumulative_pnl
        self._update_mtm_drawdown(price, time)
        if dispatcher is not None:
            broker_eq = dispatcher.get_equity()
            if broker_eq is not None:
                self._live_equity = broker_eq
        self._in_pos = False
        self._open_order_id = None
        self._open_client_order_id = None
        self._open_signal_id = None
        self._entry_brick_index = None
        self._best_price_since_entry = None
        self._waiting_for_fill = False

    def _is_broker_position_open(
        self,
        dispatcher: OrderDispatcher,
        order_id: str,
    ) -> Optional[bool]:
        """Best-effort broker state check for local position reconciliation."""
        checker = getattr(dispatcher, "is_position_open", None)
        if checker is None:
            return None
        try:
            return checker(order_id)
        except Exception as exc:
            LOG.warning(
                "Broker position check raised for order_id=%s: %s",
                order_id,
                exc,
            )
            return None

    def _make_results(self) -> Dict[str, Any]:
        """Build the standard results dict from completed trades."""
        analytics = analyze_trades(self._completed, initial_equity=self.cfg.initial_equity)
        return {
            "trades": [t.to_dict() for t in self._completed],
            "signals": list(self._signal_events),
            "summary": {
                # Trade counts
                "n_trades": analytics.n_trades,
                "n_winners": analytics.n_winners,
                "n_losers": analytics.n_losers,
                "win_rate": analytics.win_rate,
                # P&L
                "net_usd": analytics.net_pnl,
                "gross_profit": analytics.gross_profit,
                "gross_loss": analytics.gross_loss,
                "avg_trade": analytics.avg_trade,
                "avg_winner": analytics.avg_winner,
                "avg_loser": analytics.avg_loser,
                # Friction cost breakdown
                "total_spread_usd": analytics.total_spread_usd,
                "total_commission_usd": analytics.total_commission_usd,
                "total_swap_usd": analytics.total_swap_usd,
                "total_friction_usd": analytics.total_friction_usd,
                "avg_spread_per_trade": analytics.avg_spread_per_trade,
                "avg_commission_per_trade": analytics.avg_commission_per_trade,
                "avg_swap_per_trade": analytics.avg_swap_per_trade,
                "friction_pct_of_gross": analytics.friction_pct_of_gross,
                # Risk-adjusted metrics
                "profit_factor": analytics.profit_factor,
                "omega": analytics.omega,
                "expectancy": analytics.expectancy,
                "z_factor": analytics.z_factor,
                "sharpe_ratio": analytics.sharpe_ratio,
                "calmar_ratio": analytics.calmar_ratio,
                # Streaks
                "max_win_streak": analytics.max_win_streak,
                "max_loss_streak": analytics.max_loss_streak,
                # MAE/MFE
                "avg_mae_usd": analytics.avg_mae_usd,
                "avg_mfe_usd": analytics.avg_mfe_usd,
                "mfe_mae_ratio": analytics.mfe_mae_ratio,
                # Holding times
                "avg_holding_hours": analytics.avg_holding_hours,
                "avg_winner_hours": analytics.avg_winner_hours,
                "avg_loser_hours": analytics.avg_loser_hours,
                "avg_run_capture": analytics.avg_run_capture,
                "median_run_capture": analytics.median_run_capture,
                "run_capture_samples": analytics.run_capture_samples,
                # Drawdown
                "max_drawdown_pct": analytics.max_drawdown_pct,
                "max_drawdown_usd": analytics.max_drawdown_usd,
                "max_drawdown_mtm_pct": float(self._max_drawdown_mtm_pct),
                "max_drawdown_mtm_usd": float(self._max_drawdown_mtm_usd),
                "max_used_margin_usd": float(self._max_used_margin_usd),
                "min_margin_level_pct": (
                    float(self._min_margin_level_pct)
                    if np.isfinite(self._min_margin_level_pct)
                    else None
                ),
                "margin_rejects": int(self._margin_reject_count),
                # Equity
                "final_equity": self._live_equity,
                "final_equity_mtm": float(self._equity_mtm),
                "total_return_pct": analytics.total_return_pct,
            },
        }


def _ensure_utc(ts: pd.Timestamp) -> pd.Timestamp:
    return ts if ts.tzinfo is not None else ts.tz_localize("UTC")
