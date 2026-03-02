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

import threading
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from kinetra.renko.brick_engine import IncrementalRenkoBuilder, build_renko
from kinetra.renko.filters import evaluate_entry, flip_rate, markov_stickiness
from kinetra.renko.live_trader import (
    LiveTrade,
    OrderDispatcher,
    PaperDispatcher,
    PERGate,
    TradeDirection,
)
from kinetra.renko.trade_analytics import analyze_trades

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
    fliprate_window: int = 50
    markov_window: int = 50
    fliprate_threshold: float = 0.35
    markov_threshold: float = 0.55
    allow_short: bool = True

    # Sizing
    sizing_mode: str = "compounding"  # "static" | "compounding" | "risk_based"
    target_risk_usd: float = 100.0  # risk_based: USD at risk per brick
    initial_equity: float = 1_000.0  # starting account equity (default $1,000)
    compounding_per_k: float = 0.01  # compounding: lots per $1 000 equity
    lot_step: float = 0.01
    min_lots: float = 0.01
    gate_lot_ceiling: float = 10.0  # hard cap; set 0.01 for micro-live

    # Friction
    spread_ticks: float = 89.0  # ~8.9 pips for XAUUSD
    commission_per_lot: float = 7.0  # round-trip


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

    def __init__(self, config: EngineConfig) -> None:
        self.cfg = config
        self._usd_per_point = config.usd_per_tick / config.tick_size
        self._reset_state()

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

        # Accept extra kwargs from CTraderBarProvider (open_, high, low, volume…)
        def _on_bar(symbol: str, close: float, timestamp: datetime, **_: Any) -> None:
            self.process_bar(close, timestamp)

        bar_provider.subscribe(self.cfg.symbol, _on_bar)
        bar_provider.start()

        import time as _time

        try:
            if stop_event is not None:
                stop_event.wait()
            else:
                while True:
                    _time.sleep(1)
        except KeyboardInterrupt:
            pass
        finally:
            bar_provider.stop()
            self._active_dispatcher = None

        return self._make_results()

    def replay_simulate_from_csv(self, csv_path: str) -> Dict[str, Any]:
        """Convenience: load M1 CSV and run backtest."""
        df = pd.read_csv(csv_path, parse_dates=["time"])
        closes = pd.Series(df["close"].values, index=pd.DatetimeIndex(df["time"], tz="UTC"))
        return self.backtest(closes)

    # ── Internal state ────────────────────────────────────────────────────────

    def _reset_state(self) -> None:
        max_w = max(self.cfg.fliprate_window, self.cfg.markov_window) + 2
        self._builder = IncrementalRenkoBuilder(self.cfg.brick_size)
        self._dir_deque: deque = deque(maxlen=max_w)
        self._in_pos = False
        self._pos_dir = 0
        self._entry_price = 0.0
        self._entry_time: Optional[pd.Timestamp] = None
        self._entry_lots = 0.0
        self._open_order_id: Optional[str] = None  # broker order_id for the current position
        self._prev_dir: Optional[int] = None
        self._cumulative_pnl = 0.0
        self._live_equity = self.cfg.initial_equity
        self._completed: List[LiveTrade] = []
        self._active_dispatcher: Optional[OrderDispatcher] = None

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

        stepped = round(raw / cfg.lot_step) * cfg.lot_step
        return float(max(cfg.min_lots, min(stepped, cfg.gate_lot_ceiling)))

    # ── P&L ───────────────────────────────────────────────────────────────────

    def _simulate_pnl(
        self,
        entry: float,
        exit_price: float,
        lots: float,
        direction: int,
    ) -> Tuple[float, float, float]:
        """Return (gross_usd, friction_usd, net_usd)."""
        gross_usd = (exit_price - entry) * direction * self._usd_per_point * lots
        friction_usd = (
            self.cfg.spread_ticks * self.cfg.usd_per_tick * lots
            + self.cfg.commission_per_lot * lots
        )
        return gross_usd, friction_usd, gross_usd - friction_usd

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
        stop_dist = self.cfg.stop_bricks * self.cfg.brick_size

        # ── Exits ────────────────────────────────────────────────────────────
        if self._in_pos:
            stop_hit = (self._pos_dir == 1 and b_close <= self._entry_price - stop_dist) or (
                self._pos_dir == -1 and b_close >= self._entry_price + stop_dist
            )
            colour_change = direction != self._pos_dir

            if stop_hit or colour_change:
                reason = "stop" if stop_hit else "colour_change"
                gross_usd, friction_usd, net_usd = self._simulate_pnl(
                    self._entry_price, b_close, self._entry_lots, self._pos_dir
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
                )
                trade.close(
                    exit_price=b_close,
                    exit_time=bar_time,
                    exit_reason=reason,
                    friction_usd=float(friction_usd),
                    usd_per_point=self._usd_per_point,
                )
                self._completed.append(trade)
                self._cumulative_pnl += trade.net_usd
                self._live_equity = self.cfg.initial_equity + self._cumulative_pnl

                # Sync to real broker equity when available (live mode).
                # PaperDispatcher.get_equity() returns None — keeps simulated value.
                if dispatcher is not None:
                    broker_eq = dispatcher.get_equity()
                    if broker_eq is not None:
                        self._live_equity = broker_eq

                if dispatcher is not None and self._open_order_id is not None:
                    dispatcher.close_position(
                        symbol=self.cfg.symbol,
                        order_id=self._open_order_id,
                        price=b_close,
                        lots=self._entry_lots,
                        comment=reason,
                    )
                self._in_pos = False
                self._open_order_id = None

        # ── Entries ──────────────────────────────────────────────────────────
        if not self._in_pos:
            is_flip = self._prev_dir is not None and direction != self._prev_dir
            if is_flip and np.isfinite(fr_val) and np.isfinite(pUU_val) and np.isfinite(pDD_val):
                entry_ok = evaluate_entry(
                    direction=direction,
                    flip_rate_val=fr_val,
                    pUU=pUU_val,
                    pDD=pDD_val,
                    fliprate_threshold=self.cfg.fliprate_threshold,
                    markov_threshold=self.cfg.markov_threshold,
                )

                # Live feedback: show entry evaluation
                import logging

                LOG = logging.getLogger("kinetra.renko")
                dir_str = "BUY" if direction == 1 else "SELL"
                status = "✓ PASS" if entry_ok else "✗ FAIL"
                LOG.info(
                    "Entry eval [%s]: FR=%.3f M=%.3f → %s  (price=%.2f)",
                    dir_str,
                    fr_val,
                    pUU_val if direction == 1 else pDD_val,
                    status,
                    b_close,
                )

                if entry_ok and (direction == 1 or self.cfg.allow_short):
                    lots = self._compute_lots()
                    if lots > 0:
                        self._in_pos = True
                        self._pos_dir = direction
                        self._entry_price = b_close
                        self._entry_time = bar_time
                        self._entry_lots = lots
                        LOG.info("  → Opening %s %.3f lots @ %.2f", dir_str, lots, b_close)
                        if dispatcher is not None:
                            stop_price = (
                                b_close - stop_dist if direction == 1 else b_close + stop_dist
                            )
                            result = dispatcher.open_position(
                                symbol=self.cfg.symbol,
                                direction=TradeDirection(direction),
                                lots=lots,
                                price=b_close,
                                stop_price=stop_price,
                            )
                            # Store broker order_id for close call + audit trail
                            if result and result.success:
                                self._open_order_id = result.order_id

        self._prev_dir = direction

    def _force_close(self, price: float, time: pd.Timestamp) -> None:
        """Close the open position at end-of-data."""
        gross_usd, friction_usd, net_usd = self._simulate_pnl(
            self._entry_price, price, self._entry_lots, self._pos_dir
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
        )
        trade.close(
            exit_price=price,
            exit_time=time,
            exit_reason="end_of_data",
            friction_usd=float(friction_usd),
            usd_per_point=self._usd_per_point,
        )
        self._completed.append(trade)
        self._cumulative_pnl += trade.net_usd
        self._live_equity = self.cfg.initial_equity + self._cumulative_pnl
        self._in_pos = False

    def _make_results(self) -> Dict[str, Any]:
        """Build the standard results dict from completed trades."""
        analytics = analyze_trades(self._completed, initial_equity=self.cfg.initial_equity)
        return {
            "trades": [t.to_dict() for t in self._completed],
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
                # Drawdown
                "max_drawdown_pct": analytics.max_drawdown_pct,
                "max_drawdown_usd": analytics.max_drawdown_usd,
                # Equity
                "final_equity": self._live_equity,
                "total_return_pct": analytics.total_return_pct,
            },
        }


def _ensure_utc(ts: pd.Timestamp) -> pd.Timestamp:
    return ts if ts.tzinfo is not None else ts.tz_localize("UTC")
