#!/usr/bin/env python3
"""
Kinetra Renko Brick Trade Engine - Unified Runner
==================================================

CANONICAL REFERENCE: XAUUSD
  - All engine changes validated on XAUUSD first
  - XAUUSD is the empirical test bed
  - Other symbols only added after XAUUSD validation passes

SINGLE ENGINE, SEQUENTIAL VALIDATION:
  1. Download historical M1 data
  2. DSP analysis → find brick size > friction costs
  3. Backtest last 3 months (quick validation)
  4. If good → backtest 3 years with rolling OOS
  5. If pass → paper trading (simulated)
  6. If pass → micro lots live
  7. If pass → scaled lots live

NO NEW CODE AT LATER STAGES - same engine throughout.
All code changes go through full validation chain on XAUUSD first.

Usage:
    python scripts/renko_engine.py XAUUSD --stage all
    python scripts/renko_engine.py XAUUSD --stage backtest --months 3
    python scripts/renko_engine.py XAUUSD --stage paper
    python scripts/renko_engine.py XAUUSD --stage live --size micro
"""

import json
import logging
import os
import sys
import threading
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import click
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

if TYPE_CHECKING:
    from kinetra.renko.trading_engine import RenkoEngine

from rich import box
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table

from kinetra.renko.brick_engine import bricks_per_day, build_renko
from kinetra.renko.trading_engine import EngineConfig, RenkoEngine

KR = PROJECT_ROOT
console = Console()
LOG = logging.getLogger("kinetra.renko")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    handlers=[
        logging.FileHandler(
            KR
            / "outputs"
            / "logs"
            / f"renko_engine_{datetime.now(timezone.utc).strftime('%Y%m%d')}.log"
        ),
        logging.StreamHandler(),
    ],
)


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────


def load_m1_data(symbol: str) -> Optional[pd.Series]:
    """Load M1 data from canonical location."""
    from kinetra.data_utils import load_mt5_csv

    data_dir = KR / "data" / "master_standardized" / "ctrader" / "pepperstone" / "metals" / symbol
    m1_file = list(data_dir.glob("*_M1_*.csv"))

    if not m1_file:
        return None

    df = load_mt5_csv(str(m1_file[0]))

    # load_mt5_csv returns Title case columns: Close, and sets datetime as index
    close_col = "Close" if "Close" in df.columns else "close"

    # Index should already be DatetimeIndex from load_mt5_csv
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError(f"Index is not DatetimeIndex: {type(df.index)}")

    # Create Series with proper DatetimeIndex (ensure UTC)
    series = pd.Series(df[close_col].values, index=df.index)

    # Ensure UTC timezone
    if series.index.tz is None:
        series.index = series.index.tz_localize("UTC")
    else:
        series.index = series.index.tz_convert("UTC")

    # Sort by index and remove duplicates
    series = series.sort_index()
    series = series[~series.index.duplicated(keep="first")]

    return series


def get_data_path(symbol: str) -> Path:
    """Get canonical data directory for symbol."""
    return KR / "data" / "master_standardized" / "ctrader" / "pepperstone" / "metals" / symbol


def _format_number(value: float) -> str:
    """Format large numbers with K/M suffixes for compact display."""
    abs_val = abs(value)
    if abs_val >= 1_000_000:
        return f"{value / 1_000_000:.2f}M"
    elif abs_val >= 1_000:
        return f"{value / 1_000:.2f}K"
    else:
        return f"{value:.2f}"


def _adaptive_panel_width(min_width: int = 88, max_width: int = 160, pad: int = 2) -> int:
    """Choose a panel width based on current terminal width."""
    try:
        usable = max(int(console.size.width) - int(pad), 40)
    except Exception:
        usable = 120
    if usable < min_width:
        return usable
    return min(usable, max_width)


def _run_preflight_checks(connector, symbol: str) -> bool:
    """Run minimal live-trading safety checks before enabling REAL orders."""
    checks = []

    # 1) Connector/session health
    try:
        connected = bool(connector.is_connected())
        checks.append(("connection", connected, "connector authenticated"))
    except Exception as e:
        checks.append(("connection", False, f"is_connected failed: {e}"))

    # 2) Account snapshot
    snapshot = None
    try:
        snapshot = connector.get_account_snapshot(timeout_s=10.0)
        balance = float(snapshot.get("balance", 0.0))
        checks.append(("account_snapshot", balance > 0.0, f"balance=${balance:,.2f}"))
    except Exception as e:
        checks.append(("account_snapshot", False, f"snapshot failed: {e}"))

    # 3) Symbol resolution
    symbol_id = None
    try:
        symbol_id = connector.find_symbol_id(symbol, timeout_s=10.0)
        checks.append(("symbol_resolution", symbol_id is not None, f"{symbol} -> {symbol_id}"))
    except Exception as e:
        checks.append(("symbol_resolution", False, f"resolve failed: {e}"))

    # 4) Symbol metadata sanity (digits)
    try:
        if symbol_id is None:
            raise RuntimeError("symbol id unavailable")
        digits = int(connector.get_digits(symbol_id, timeout_s=10.0))
        checks.append(("symbol_digits", 0 <= digits <= 10, f"digits={digits}"))
    except Exception as e:
        checks.append(("symbol_digits", False, f"digits failed: {e}"))

    click.echo()
    click.secho("Preflight results:", fg="cyan", bold=True)
    all_ok = True
    for name, ok, detail in checks:
        mark = "PASS" if ok else "FAIL"
        color = "green" if ok else "red"
        click.secho(f"  [{mark}] {name:16s} {detail}", fg=color)
        all_ok = all_ok and ok

    # Print account context for operator confirmation.
    if snapshot:
        click.echo(
            f"  Account: {snapshot.get('account_id')}  "
            f"Broker: {snapshot.get('broker_name', 'unknown')}  "
            f"Balance: ${float(snapshot.get('balance', 0.0)):,.2f}"
        )
    return all_ok


def _load_dsp_profile(symbol: str) -> tuple[Optional[dict], Path]:
    """Load per-symbol DSP profile used by backtest/paper/live modes."""
    dsp_file = get_data_path(symbol) / "dsp_profile.json"
    if not dsp_file.exists():
        return None, dsp_file
    with open(dsp_file) as f:
        return json.load(f), dsp_file


def _build_stream_runtime(symbol: str, dsp: dict, lot_ceiling: float, live_orders: bool):
    """Build shared runtime components for paper/dry-run/live execution."""
    from kinetra.renko.ctrader_dispatcher import CTraderBarProvider, CTraderOrderDispatcher
    from kinetra.renko.live_trader import PaperDispatcher

    cfg, spec = _build_engine_config(symbol, dsp, sizing_mode="static", lot_ceiling=lot_ceiling)
    engine = RenkoEngine(cfg, spec=spec if live_orders else None, quiet_mode=live_orders)
    bar_provider = CTraderBarProvider
    dispatcher_cls = CTraderOrderDispatcher if live_orders else PaperDispatcher
    return cfg, spec, engine, bar_provider, dispatcher_cls


def _apply_strategy_overrides(
    cfg: EngineConfig,
    *,
    stop_bricks: Optional[float] = None,
    target_risk: Optional[float] = None,
    brick_size: Optional[float] = None,
    fliprate_window: Optional[int] = None,
    markov_window: Optional[int] = None,
    fliprate_threshold: Optional[float] = None,
    markov_threshold: Optional[float] = None,
) -> None:
    """Apply CLI optimization overrides consistently across all execution modes."""
    if stop_bricks is not None:
        cfg.stop_bricks = float(stop_bricks)
    if target_risk is not None:
        cfg.target_risk_usd = float(target_risk)
    if brick_size is not None:
        cfg.brick_size = float(brick_size)
    if fliprate_window is not None:
        cfg.fliprate_window = int(fliprate_window)
    if markov_window is not None:
        cfg.markov_window = int(markov_window)
    if fliprate_threshold is not None:
        cfg.fliprate_threshold = float(fliprate_threshold)
    if markov_threshold is not None:
        cfg.markov_threshold = float(markov_threshold)


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


class DriftAdaptationController:
    """Bounded online adaptation of safe live parameters under drift."""

    def __init__(self, engine: "RenkoEngine", opts: dict) -> None:
        self.engine = engine
        self.opts = opts
        self.enabled = bool(opts.get("enabled", False))
        self.last_bar = -1
        self.baseline = {
            "target_risk_usd": float(engine.cfg.target_risk_usd),
            "stop_bricks": float(engine.cfg.stop_bricks),
            "fliprate_threshold": float(engine.cfg.fliprate_threshold),
            "markov_threshold": float(engine.cfg.markov_threshold),
        }
        self.adapt_count = 0
        self.last_action = "none"

    def _set_param(self, key: str, new_val: float, reason: str) -> None:
        old_val = float(getattr(self.engine.cfg, key))
        if abs(new_val - old_val) < 1e-12:
            return
        setattr(self.engine.cfg, key, float(new_val))
        self.adapt_count += 1
        self.last_action = f"{key}:{old_val:.4f}->{new_val:.4f} ({reason})"
        LOG.warning("[ADAPT] %s", self.last_action)

    def _rollback(self, reason: str) -> None:
        for key, val in self.baseline.items():
            setattr(self.engine.cfg, key, float(val))
        self.last_action = f"rollback ({reason})"
        LOG.warning("[ADAPT] rollback to baseline: %s", reason)

    def maybe_adapt(self, bars: int) -> None:
        if not self.enabled:
            return
        step = int(self.opts.get("every_bars", 100))
        if bars <= 0 or bars - self.last_bar < step:
            return
        self.last_bar = bars

        summary = self.engine._make_results().get("summary", {})
        n_trades = int(summary.get("n_trades", 0))
        if n_trades < int(self.opts.get("min_trades", 20)):
            return

        omega = float(summary.get("omega", 0.0))
        win_rate = float(summary.get("win_rate", 0.0))
        if omega < float(self.opts.get("rollback_omega", 0.8)) and self.adapt_count > 0:
            self._rollback(f"omega={omega:.3f} below rollback")
            return

        low_omega = float(self.opts.get("low_omega", 1.0))
        high_omega = float(self.opts.get("high_omega", 1.8))
        low_wr = float(self.opts.get("low_wr", 0.45))
        high_wr = float(self.opts.get("high_wr", 0.60))
        step_risk = float(self.opts.get("step_risk", 0.10))
        step_thr = float(self.opts.get("step_thr", 0.01))
        step_stop = float(self.opts.get("step_stop", 0.05))

        if omega < low_omega or win_rate < low_wr:
            # Defensive mode under drift.
            self._set_param(
                "target_risk_usd",
                _clamp(
                    self.engine.cfg.target_risk_usd * (1.0 - step_risk),
                    float(self.opts.get("risk_min", 25.0)),
                    float(self.opts.get("risk_max", 200.0)),
                ),
                f"defensive omega={omega:.3f} wr={win_rate:.3f}",
            )
            self._set_param(
                "stop_bricks",
                _clamp(
                    self.engine.cfg.stop_bricks * (1.0 + step_stop),
                    float(self.opts.get("stop_min", 0.3)),
                    float(self.opts.get("stop_max", 1.5)),
                ),
                "defensive widen stop",
            )
            self._set_param(
                "fliprate_threshold",
                _clamp(
                    self.engine.cfg.fliprate_threshold + step_thr,
                    float(self.opts.get("flip_min", 0.25)),
                    float(self.opts.get("flip_max", 0.45)),
                ),
                "defensive stricter flip gate",
            )
            self._set_param(
                "markov_threshold",
                _clamp(
                    self.engine.cfg.markov_threshold + step_thr,
                    float(self.opts.get("markov_min", 0.50)),
                    float(self.opts.get("markov_max", 0.70)),
                ),
                "defensive stricter markov gate",
            )
            return

        if omega > high_omega and win_rate > high_wr:
            # Opportunity mode in stable favorable regime.
            self._set_param(
                "target_risk_usd",
                _clamp(
                    self.engine.cfg.target_risk_usd * (1.0 + step_risk),
                    float(self.opts.get("risk_min", 25.0)),
                    float(self.opts.get("risk_max", 200.0)),
                ),
                f"offensive omega={omega:.3f} wr={win_rate:.3f}",
            )
            self._set_param(
                "stop_bricks",
                _clamp(
                    self.engine.cfg.stop_bricks * (1.0 - step_stop),
                    float(self.opts.get("stop_min", 0.3)),
                    float(self.opts.get("stop_max", 1.5)),
                ),
                "offensive tighten stop",
            )
            self._set_param(
                "fliprate_threshold",
                _clamp(
                    self.engine.cfg.fliprate_threshold - step_thr,
                    float(self.opts.get("flip_min", 0.25)),
                    float(self.opts.get("flip_max", 0.45)),
                ),
                "offensive looser flip gate",
            )
            self._set_param(
                "markov_threshold",
                _clamp(
                    self.engine.cfg.markov_threshold - step_thr,
                    float(self.opts.get("markov_min", 0.50)),
                    float(self.opts.get("markov_max", 0.70)),
                ),
                "offensive looser markov gate",
            )


def _run_streaming_engine(
    engine, bar_provider, dispatcher, stop_event: threading.Event, stop_msg: str
):
    """Run the shared streaming loop for paper/dry-run/live modes."""
    try:
        return engine.run(
            bar_provider=bar_provider,
            dispatcher=dispatcher,
            stop_event=stop_event,
        )
    except KeyboardInterrupt:
        click.echo()
        click.secho(stop_msg, fg="yellow")
        stop_event.set()
        return engine._make_results()


def _run_execution_preflight_trade(
    connector,
    symbol: str,
    cfg: EngineConfig,
    bar_provider_cls,
    dispatcher_cls,
    lots: float,
    hold_seconds: float = 1.5,
    timeout_s: float = 30.0,
) -> bool:
    """Place and close a tiny live order to verify execution path."""
    from kinetra.renko.live_trader import TradeDirection

    click.echo()
    click.secho("Running execution preflight (tiny live order)...", fg="yellow", bold=True)

    bar_provider = bar_provider_cls(connector)
    dispatcher = dispatcher_cls(connector, bar_provider)
    latest = {"close": None}
    bar_ready = threading.Event()

    def _on_bar(symbol: str, close: float, timestamp, **_) -> None:
        latest["close"] = float(close)
        bar_ready.set()

    bar_provider.subscribe(symbol, _on_bar)
    try:
        bar_provider.start()
    except Exception as e:
        click.secho(f"❌ Execution preflight failed: bar provider start error ({e})", fg="red")
        return False

    try:
        if not bar_ready.wait(timeout=timeout_s):
            click.secho("❌ Execution preflight failed: no live bar received", fg="red")
            return False

        entry_price = float(latest["close"])
        stop_dist = max(cfg.stop_bricks * cfg.brick_size, cfg.tick_size)
        open_result = dispatcher.open_position(
            symbol=symbol,
            direction=TradeDirection.LONG,
            lots=lots,
            price=entry_price,
            stop_price=entry_price - stop_dist,
            comment="preflight_open",
        )
        if not getattr(open_result, "success", False):
            click.secho(
                f"❌ Execution preflight failed: open rejected ({getattr(open_result, 'error', 'unknown')})",
                fg="red",
            )
            return False

        order_id = getattr(open_result, "order_id", "")
        click.secho(f"  Opened tiny preflight position id={order_id}", fg="cyan")
        time.sleep(max(0.0, hold_seconds))

        close_result = dispatcher.close_position(
            symbol=symbol,
            order_id=order_id,
            price=entry_price,
            lots=lots,
            comment="preflight_close",
        )
        if not getattr(close_result, "success", False):
            click.secho(
                f"❌ Execution preflight failed: close rejected ({getattr(close_result, 'error', 'unknown')})",
                fg="red",
            )
            return False

        click.secho("✅ Execution preflight passed (open + close confirmed)", fg="green")
        return True
    finally:
        try:
            bar_provider.stop()
        except Exception:
            pass


def _stats_panel(
    summary: dict,
    symbol: str,
    mode: str,
    engine: Optional["RenkoEngine"] = None,
    trades: list = None,
) -> Panel:
    """Build a rich Panel from engine state (single source of truth) with detailed stats."""
    # ── EXTRACT METRICS ────────────────────────────────────────────────────────
    # Trade counts
    n = summary.get("n_trades", 0)
    w = summary.get("n_winners", 0)
    l = summary.get("n_losers", 0)
    wr = summary.get("win_rate", 0.0)

    # P&L
    net = summary.get("net_usd", 0.0)
    eq = summary.get("final_equity", 0.0)
    gross_profit = summary.get("gross_profit", 0.0)
    gross_loss = summary.get("gross_loss", 0.0)
    avg_trade = summary.get("avg_trade", 0.0)
    avg_w = summary.get("avg_winner", 0.0)
    avg_l = summary.get("avg_loser", 0.0)
    expect = summary.get("expectancy", 0.0)
    pf = summary.get("profit_factor", 0.0)

    # Risk-adjusted metrics
    om = summary.get("omega", 0.0)
    z_factor = summary.get("z_factor", 0.0)
    sharpe = summary.get("sharpe_ratio", 0.0)
    calmar = summary.get("calmar_ratio", 0.0)

    # Streaks
    max_ws = summary.get("max_win_streak", 0)
    max_ls = summary.get("max_loss_streak", 0)

    # Drawdown
    dd = summary.get("max_drawdown_pct", 0.0)
    dd_usd = summary.get("max_drawdown_usd", 0.0)
    total_return = summary.get("total_return_pct", 0.0)

    # MAE/MFE (prefer summary, fall back to calculating from trades)
    mae_usd = summary.get("avg_mae_usd", 0.0)
    mfe_usd = summary.get("avg_mfe_usd", 0.0)
    mfe_ratio = summary.get("mfe_mae_ratio", 0.0)
    if not mae_usd and trades:
        maes = [
            abs(float(getattr(t, "max_adverse_excursion", 0)))
            for t in trades
            if hasattr(t, "max_adverse_excursion")
        ]
        mfes = [
            abs(float(getattr(t, "max_favorable_excursion", 0)))
            for t in trades
            if hasattr(t, "max_favorable_excursion")
        ]
        if maes:
            mae_usd = sum(maes) / len(maes)
        if mfes:
            mfe_usd = sum(mfes) / len(mfes)
        if maes and mfes and mae_usd > 0:
            mfe_ratio = mfe_usd / mae_usd

    # Holding times
    avg_hold = summary.get("avg_holding_hours", 0.0)
    avg_win_hold = summary.get("avg_winner_hours", 0.0)
    avg_lose_hold = summary.get("avg_loser_hours", 0.0)

    # ── BUILD TABLE ────────────────────────────────────────────────────────────
    t = Table(box=box.SIMPLE, show_header=False, padding=(0, 1))
    t.add_column("", style="dim", width=24)
    t.add_column("", justify="right", width=20)
    t.add_column("", style="dim", width=24)
    t.add_column("", justify="right", width=20)

    # Colour coding
    wr_col = "green" if wr >= 0.5 else "red"
    net_col = "green" if net >= 0 else "red"
    om_col = "green" if om >= 1.5 else ("yellow" if om >= 1.0 else "red")
    z_col = "green" if z_factor >= 2.5 else ("yellow" if z_factor >= 1.5 else "red")
    mfe_col = "green" if mfe_ratio >= 1.5 else ("yellow" if mfe_ratio >= 1.0 else "red")

    # ── LIVE POSITION (if available) ───────────────────────────────────────────
    if engine is not None:
        pos_str = "FLAT"
        if engine._in_pos:
            dir_str = "LONG" if engine._pos_dir == 1 else "SHORT"
            pos_str = (
                f"[bold yellow]{dir_str}[/] {engine._entry_lots:.3f} @ ${engine._entry_price:.2f}"
            )

        t.add_row(
            "Position",
            pos_str,
            "Bricks",
            f"{int(getattr(engine, 'bricks_processed', 0))}",
        )

    # ── SECTION: PERFORMANCE ──────────────────────────────────────────────────
    t.add_row(
        "[dim]── PERFORMANCE ──────────[/]",
        "",
        "[dim]── PERFORMANCE ──────────[/]",
        "",
    )
    t.add_row(
        "Trades",
        f"{n}  ({w}W / {l}L)",
        "Win rate",
        f"[{wr_col}]{wr:.1%}[/]",
    )
    t.add_row(
        "Net P&L",
        f"[{net_col}]${_format_number(net)}[/]",
        "Total return",
        f"{total_return:.1f}%",
    )

    # ── SECTION: RISK-ADJUSTED ────────────────────────────────────────────────
    t.add_row(
        "[dim]── RISK-ADJUSTED ────────[/]",
        "",
        "[dim]── RISK-ADJUSTED ────────[/]",
        "",
    )
    t.add_row(
        "Omega",
        f"[{om_col}]{om:.3f}[/]",
        "Z-Factor",
        f"[{z_col}]{z_factor:.2f}[/]",
    )
    t.add_row(
        "Sharpe",
        f"{sharpe:.2f}",
        "Calmar",
        f"{calmar:.1f}",
    )
    t.add_row(
        "Profit factor",
        f"{pf:.2f}",
        "Return/DD",
        f"{abs(total_return / dd):.1f}" if dd != 0 else "N/A",
    )

    # ── SECTION: P&L DETAIL ───────────────────────────────────────────────────
    t.add_row(
        "[dim]── P&L DETAIL ───────────[/]",
        "",
        "[dim]── P&L DETAIL ───────────[/]",
        "",
    )
    t.add_row(
        "Gross profit",
        f"[green]${_format_number(gross_profit)}[/]",
        "Gross loss",
        f"[red]-${_format_number(gross_loss)}[/]",
    )
    t.add_row(
        "Avg winner",
        f"[green]${_format_number(avg_w)}[/]",
        "Avg loser",
        f"[red]-${_format_number(avg_l)}[/]",
    )
    t.add_row(
        "Avg trade",
        f"${_format_number(avg_trade)}",
        "Expectancy",
        f"${_format_number(expect)}",
    )

    # ── SECTION: RISK ─────────────────────────────────────────────────────────
    t.add_row(
        "[dim]── RISK ─────────────────[/]",
        "",
        "[dim]── RISK ─────────────────[/]",
        "",
    )
    t.add_row(
        "Max drawdown",
        f"{dd:.2f}%",
        "Max DD $",
        f"${_format_number(dd_usd)}",
    )
    t.add_row(
        "Max win streak",
        f"{max_ws}",
        "Max loss streak",
        f"{max_ls}",
    )

    # ── SECTION: EXECUTION (if data available) ────────────────────────────────
    if mae_usd > 0 or mfe_usd > 0 or avg_hold > 0:
        t.add_row(
            "[dim]── EXECUTION ───────────[/]",
            "",
            "[dim]── EXECUTION ───────────[/]",
            "",
        )
        if mae_usd > 0 or mfe_usd > 0:
            t.add_row(
                "Avg MAE",
                f"[red]${_format_number(mae_usd)}[/]",
                "Avg MFE",
                f"[green]${_format_number(mfe_usd)}[/]",
            )
            t.add_row(
                "MFE/MAE ratio",
                f"[{mfe_col}]{mfe_ratio:.2f}[/]",
                "",
                "",
            )
        if avg_hold > 0:
            t.add_row(
                "Avg hold time",
                f"{avg_hold:.1f}h",
                "Win hold",
                f"{avg_win_hold:.1f}h" if avg_win_hold > 0 else "-",
            )
            t.add_row(
                "Loser hold",
                f"{avg_lose_hold:.1f}h" if avg_lose_hold > 0 else "-",
                "",
                "",
            )

    # ── SECTION: EQUITY ───────────────────────────────────────────────────────
    t.add_row(
        "[dim]── EQUITY ───────────────[/]",
        "",
        "[dim]── EQUITY ───────────────[/]",
        "",
    )
    t.add_row(
        "Final equity",
        f"[bold]${_format_number(eq)}[/]",
        "",
        "",
    )

    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    border = "green" if mode.startswith("live") or mode == "paper" else "cyan"
    return Panel(
        t,
        title=f"[bold]{symbol}[/]  [{mode.upper()}]  {ts}",
        border_style=border,
        width=_adaptive_panel_width(min_width=96, max_width=170),
    )


def _print_stats(summary: dict, symbol: str, mode: str, trades: list = None) -> None:
    """Render stats panel to terminal and log a one-liner to file."""
    console.print(_stats_panel(summary, symbol, mode, trades=trades))
    LOG.info(
        "[%s %s] trades=%d net=%.2f omega=%.3f dd=%.2f%% equity=%.2f",
        symbol,
        mode,
        summary.get("n_trades", 0),
        summary.get("net_usd", 0.0),
        summary.get("omega", 0.0),
        summary.get("max_drawdown_pct", 0.0),
        summary.get("final_equity", 0.0),
    )


def _dashboard_panel(
    engine: "RenkoEngine",
    symbol: str,
    mode: str,
    bricks: int,
    acct_info: dict,
) -> Panel:
    """Build compact live snapshot panel for operator visibility."""
    summary = engine._make_results().get("summary", {})
    last_eval = getattr(engine, "_last_eval", {}) or {}

    def _fmt_eval_num(v) -> str:
        try:
            f = float(v)
            if pd.isna(f):
                return "n/a"
            return f"{f:.3f}"
        except Exception:
            return "n/a"

    table = Table(box=box.SIMPLE, show_header=False, expand=True)
    table.add_column(style="cyan")
    table.add_column()
    account_type_raw = acct_info.get("account_type", "-")

    def _format_account_type(v) -> str:
        try:
            code = int(v)
        except Exception:
            return str(v)
        labels = {
            0: "Unknown",
            1: "Hedged",
            2: "Netting",
        }
        return f"{labels.get(code, 'Type')} ({code})"

    table.add_row("[dim]Trade Performance[/]", "")
    table.add_row("Mode", mode.upper())
    table.add_row("Symbol", symbol)
    table.add_row("Renko Bricks", str(int(bricks)))
    table.add_row("Dupes dropped", str(int(getattr(engine, "duplicate_bars_dropped", 0))))
    table.add_row("Trades", str(int(summary.get("n_trades", 0))))
    table.add_row("Net P&L", f"${float(summary.get('net_usd', 0.0)):,.2f}")
    table.add_row("Omega", f"{float(summary.get('omega', 0.0)):.3f}")

    table.add_row("[dim]Trade Eval / Decision[/]", "")
    table.add_row(
        "Eval",
        f"{last_eval.get('direction', 'NA')} | flip={last_eval.get('is_flip', False)} | pass={last_eval.get('entry_ok', False)}",
    )
    table.add_row(
        "Warmup",
        f"ready={last_eval.get('warmup_ready', False)} remaining={int(last_eval.get('warmup_remaining', 0) or 0)}",
    )
    table.add_row(
        "Eval metrics",
        f"FR={_fmt_eval_num(last_eval.get('fr'))} M={_fmt_eval_num(last_eval.get('markov'))} lots={float(last_eval.get('lots', 0.0)):.3f}",
    )
    table.add_row("Decision", str(last_eval.get("reason", "n/a")))
    last_brick_time = getattr(engine, "last_brick_time", None)
    last_brick_s = (
        pd.Timestamp(last_brick_time).strftime("%H:%M:%S") if last_brick_time is not None else "n/a"
    )
    table.add_row("[dim]Live Filter Activity[/]", "")
    table.add_row("Bars seen", str(int(getattr(engine, "stream_bars_seen", 0))))
    table.add_row("Flips seen", str(int(getattr(engine, "flips_seen", 0))))
    table.add_row("Filter-ready", str(int(getattr(engine, "filter_ready_bricks", 0))))
    table.add_row("Last brick UTC", last_brick_s)

    table.add_row("[dim]Connection[/]", "")
    status = str(acct_info.get("connection_status", "-")).upper()
    hb_on = bool(acct_info.get("heartbeat_on", False))
    if status == "UP":
        status_fmt = "[green]UP[/]"
        hb_dot = "[green]●[/]" if hb_on else "[green]○[/]"
    elif status == "DEGRADED":
        status_fmt = "[yellow]DEGRADED[/]"
        hb_dot = "[yellow]●[/]" if hb_on else "[yellow]○[/]"
    elif status == "DOWN":
        status_fmt = "[red]DOWN[/]"
        hb_dot = "[red]●[/]" if hb_on else "[red]○[/]"
    else:
        status_fmt = status
        hb_dot = "[yellow]○[/]" if hb_on else "[dim]○[/]"
    table.add_row("Status", status_fmt)
    table.add_row("Heartbeat", hb_dot)
    table.add_row("Endpoint", str(acct_info.get("endpoint", "-")))
    table.add_row("Failovers", str(acct_info.get("failover_count", "-")))
    table.add_row("Generation", str(acct_info.get("failover_generation", "-")))
    table.add_row("Last failover", str(acct_info.get("last_failover_utc", "-")))
    table.add_row("Req timeouts", str(acct_info.get("request_timeouts", "-")))
    table.add_row("Snapshot source", str(acct_info.get("snapshot_source", "-")))

    table.add_row("[dim]Account / Broker Info[/]", "")
    table.add_row("Balance", f"${float(acct_info.get('balance', 0.0)):,.2f}")
    table.add_row("Account", str(acct_info.get("account_id", "-")))
    table.add_row("Broker", str(acct_info.get("broker", "-")))
    table.add_row("Environment", str(acct_info.get("environment", "-")))
    table.add_row("Account type", _format_account_type(account_type_raw))

    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    return Panel(
        table,
        title=f"[bold]{symbol}[/] LIVE SNAPSHOT  {ts}",
        border_style="red",
        width=_adaptive_panel_width(min_width=96, max_width=170),
    )


def _dashboard_errors_panel(symbol: str, acct_info: dict) -> Panel:
    """Build separate recent-errors panel for live dashboard."""
    recent_errors = list(acct_info.get("recent_errors", []) or [])
    table = Table(box=box.SIMPLE, show_header=False, expand=True)
    table.add_column(style="cyan")
    table.add_column()
    if recent_errors:
        for idx, msg in enumerate(recent_errors, start=1):
            table.add_row(f"E{idx}", str(msg))
    else:
        table.add_row("E1", "none")
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    return Panel(
        table,
        title=f"[bold]{symbol}[/] LIVE ERRORS  {ts}",
        border_style="red",
        width=_adaptive_panel_width(min_width=96, max_width=170),
    )


def _dashboard_renderable(
    engine: "RenkoEngine",
    symbol: str,
    mode: str,
    bricks: int,
    acct_info: dict,
):
    """Render snapshot + recent-errors as stacked panels."""
    return Group(
        _dashboard_panel(engine, symbol, mode, bricks, acct_info),
        _dashboard_errors_panel(symbol, acct_info),
    )


def _print_dashboard(
    engine: "RenkoEngine",
    symbol: str,
    mode: str,
    bricks: int,
    acct_info: dict,
) -> None:
    """Print one snapshot panel (non-live mode)."""
    console.print(_dashboard_renderable(engine, symbol, mode, bricks, acct_info))


def _print_system_spec(cfg: EngineConfig, spec, dsp: dict) -> None:
    """Print complete trading system specification before performance analysis."""
    console.print()

    # Header
    spec_table = Table(box=box.ROUNDED, title="[bold cyan]TRADING SYSTEM SPECIFICATION[/]")
    spec_table.add_column("Parameter", style="cyan", width=30)
    spec_table.add_column("Value", justify="right", width=25)
    spec_table.add_column("Notes", style="dim", width=40)

    # Instrument & Broker
    spec_table.add_row("Symbol", cfg.symbol, "cTrader Pepperstone ECN")
    spec_table.add_row("Contract Size", f"{spec.contract_size:.0f} oz", "Standard gold lot")
    spec_table.add_row(
        "Tick Size", f"${spec.tick_size:.2f}", f"Value per tick: ${spec.tick_value_usd:.2f}"
    )

    # Brick & Filters
    spec_table.add_row("Brick Size", f"${cfg.brick_size:.2f}", "Price movement threshold")
    brick_window = cfg.fliprate_window
    spec_table.add_row(
        "Brick Window",
        f"{brick_window} bricks",
        f"Filter lookback period (~{brick_window / 50:.1f} days)",
    )

    # Entry & Exit
    spec_table.add_row("Entry Signal", "Colour flip + filters", "2-brick direction change required")
    spec_table.add_row(
        "FlipRate Gate", f"< {cfg.fliprate_threshold:.0%}", "Reject choppy markets (% flips)"
    )
    spec_table.add_row(
        "Markov Gate", f"> {cfg.markov_threshold:.0%}", "Require direction persistence"
    )
    spec_table.add_row(
        "Stop Loss (SL)",
        f"{cfg.stop_bricks:.1f} brick",
        f"${cfg.stop_bricks * cfg.brick_size:.2f} fixed distance",
    )
    spec_table.add_row("Exit Signal", "Colour change (opposite)", "Exit on first reversal brick")
    spec_table.add_row("Trailing Stop", "Off", "Standard 1-brick fixed stop used")

    # Friction Costs
    spec_table.add_row(
        "Spread",
        f"{cfg.spread_ticks:.1f} ticks",
        f"${cfg.spread_ticks * cfg.usd_per_tick:.2f} per round-trip",
    )
    spec_table.add_row(
        "Commission", f"${cfg.commission_per_lot:.2f}/lot", "ECN round-trip per standard lot"
    )

    # Swap rates (from spec if available)
    if hasattr(spec, "swap_long") and spec.swap_long:
        spec_table.add_row(
            "Swap Long", f"${spec.swap_long:.3f}/day", "Cost to hold long positions overnight"
        )
        spec_table.add_row(
            "Swap Short", f"${spec.swap_short:+.3f}/day", "Earn on short positions (positive carry)"
        )
        if hasattr(spec, "triple_swap_day"):
            spec_table.add_row(
                "Triple Swap", f"{spec.triple_swap_day}", "3× swap charged on Wednesdays"
            )

    # Position Sizing
    spec_table.add_row("Initial Equity", f"${cfg.initial_equity:,.2f}", "Starting account balance")
    spec_table.add_row(
        "Risk per Trade", f"${cfg.target_risk_usd:.2f}", "Target USD at risk per position"
    )
    spec_table.add_row(
        "Lot Ceiling", f"{cfg.gate_lot_ceiling:.2f} lots", "Maximum position size cap"
    )
    sizing_note = (
        "Dual scenario (static + compound)" if cfg.symbol.upper() == "XAUUSD" else cfg.sizing_mode
    )
    spec_table.add_row(
        "Sizing Mode (Backtest)",
        sizing_note,
        "Static (0.01) + Compounding (max 10.0)" if cfg.symbol.upper() == "XAUUSD" else "",
    )

    # Trading Hours
    spec_table.add_row("Trading Hours", "24/5 (Mon-Fri)", "No weekend trading (forex session)")
    spec_table.add_row("Week Start", "Monday 00:00 UTC", "New trading week begins")
    spec_table.add_row("Week Close", "Friday 24:00 UTC", "End of week, swap settlement")

    # DSP-derived
    vr_peak = dsp.get("vr_peak_scale", "N/A")
    regime = dsp.get("regime", "UNKNOWN")
    spec_table.add_row("VR Peak Scale", f"{vr_peak} M30 bars", "Trend persistence peak from DSP")
    spec_table.add_row("Regime", str(regime), "Market classification from DSP")

    # Validation gates
    spec_table.add_row("Min Omega", "≥ 1.5", "Statistical significance gate")
    spec_table.add_row("Min Trades", "≥ 30", "Sample size sufficiency")
    spec_table.add_row("Target Win Rate", "> 50%", "More winners than losers")
    spec_table.add_row("Target MFE/MAE", "> 1.5", "Execution quality (capture favorable moves)")

    console.print(spec_table)


# ──────────────────────────────────────────────────────────────────────────────
# STAGE 2: DSP ANALYSIS
# ──────────────────────────────────────────────────────────────────────────────


def stage_dsp(symbol: str) -> bool:
    """Run DSP analysis to find optimal brick size."""
    click.echo(f"\n{'=' * 60}")
    click.secho("STAGE 2: DSP ANALYSIS", fg="cyan", bold=True)
    click.echo(f"{'=' * 60}")

    # Load M1 data
    closes = load_m1_data(symbol)
    if closes is None:
        click.secho(f"❌ No M1 data found for {symbol}", fg="red")
        return False

    click.echo(f"Loaded {len(closes):,} M1 bars")

    # Run DSP analysis
    from kinetra.renko.dsp import run_dsp

    click.echo("Running DSP analysis...")
    result = run_dsp(closes, symbol=symbol)

    if "error" in result:
        click.secho(f"❌ DSP analysis failed: {result['error']}", fg="red")
        return False

    # Save DSP profile
    dsp_dir = KR / "outputs" / "dsp"
    dsp_dir.mkdir(parents=True, exist_ok=True)
    dsp_file = dsp_dir / f"{symbol}_dsp.json"

    with open(dsp_file, "w") as f:
        json.dump(result, f, indent=2, default=str)

    click.secho(f"✅ DSP profile saved to {dsp_file}", fg="green")

    # Print summary
    click.echo()
    click.echo(f"Brick Size: ${result.get('brick_size', 0):.2f}")
    click.echo(f"VR Peak Scale: {result.get('vr_peak_scale', 0)} M30 bars")
    click.echo(f"Regime: {result.get('regime', 'UNKNOWN')}")
    click.echo(f"Bricks/Day: {result.get('bricks_per_day', 0):.1f}")

    return True


def _build_engine_config(
    symbol: str,
    dsp: dict,
    sizing_mode: str = "compounding",
    lot_ceiling: float = 999.0,
) -> tuple:
    """Build a canonical EngineConfig from a DSP profile dict.

    Returns:
        (EngineConfig, InstrumentSpec) tuple

    Calibrates against InstrumentSpec (contract_spec.json) for:
    - tick_size, usd_per_tick (from broker)
    - spread_ticks (median from CSV or spec)
    - commission_per_lot (ECN rate)

    Converts vr_peak_scale (in M30 bars from DSP) to brick-based filter window
    by building sample bricks and measuring empirical frequency.
    """
    from kinetra.friction_cost import load_spec

    # Load instrument spec from contract_spec.json (broker calibration)
    spec = load_spec(symbol)

    LOG.info(
        "Loaded %s spec: tick_size=%.5f, contract_size=%.0f, spread=%.1f ticks, commission=$%.2f/lot",
        symbol,
        spec.tick_size,
        spec.contract_size,
        spec.spread_points,
        spec.commission_per_lot,
    )

    stop_bricks = 1.0 if symbol.upper() == "XAUUSD" else 0.5
    brick_size = float(dsp.get("brick_size", 1.0))

    # Load M1 data to estimate brick frequency
    closes = load_m1_data(symbol)
    if closes is not None and len(closes) > 1000:
        # Build sample bricks to measure frequency
        bricks = build_renko(closes.tail(min(10000, len(closes))), brick_size)
        if len(bricks) > 10:
            bpd = bricks_per_day(bricks)
            # vr_peak_scale is in M30 bars (from M30_VR_SCALES in dsp.py)
            # M30: 48 bars per trading day (2 bars/hour × 24 hours)
            vr_peak_scale_m30 = int(dsp.get("vr_peak_scale", 50))
            m30_bars_per_day = 2 * 24  # M30: 48 bars/day
            days_in_peak = vr_peak_scale_m30 / m30_bars_per_day
            window = max(10, int(bpd * days_in_peak))
        else:
            window = 50  # Fallback
    else:
        window = 50  # Fallback

    return (
        EngineConfig(
            symbol=symbol,
            brick_size=brick_size,
            usd_per_tick=spec.tick_value_usd,  # From spec (calculated: tick_size × contract_size)
            tick_size=spec.tick_size,  # From spec (broker)
            stop_bricks=stop_bricks,
            fliprate_window=window,
            markov_window=window,
            fliprate_threshold=0.35,
            markov_threshold=0.55,
            spread_ticks=spec.spread_points,  # From spec (median of CSV or broker snapshot)
            commission_per_lot=spec.commission_per_lot,  # From spec ($7.00 ECN standard)
            sizing_mode=sizing_mode,
            gate_lot_ceiling=lot_ceiling,
        ),
        spec,
    )


# ──────────────────────────────────────────────────────────────────────────────
# STAGE 3: BACKTEST
# ──────────────────────────────────────────────────────────────────────────────


def stage_backtest(
    symbol: str,
    months: int = 3,
    min_omega: float = 1.5,
    min_trades: int = 30,
    stop_bricks_override: Optional[float] = None,
    target_risk_override: Optional[float] = None,
    brick_size_override: Optional[float] = None,
    fliprate_window_override: Optional[int] = None,
    markov_window_override: Optional[int] = None,
    fliprate_threshold_override: Optional[float] = None,
    markov_threshold_override: Optional[float] = None,
) -> bool:
    """Backtest the last N months of data."""
    click.echo(f"\n{'=' * 60}")
    click.secho(f"STAGE 3: BACKTEST ({months} months)", fg="cyan", bold=True)
    click.echo(f"{'=' * 60}")

    # Load all M1 data
    closes = load_m1_data(symbol)
    if closes is None:
        click.secho(f"❌ No M1 data found for {symbol}", fg="red")
        return False

    # Filter to last N months
    cutoff = closes.index[-1] - pd.DateOffset(months=months)
    test_closes = closes[closes.index >= cutoff]

    click.echo(f"Testing {len(test_closes):,} bars ({months} months)")
    click.echo(f"Range: {test_closes.index[0]} to {test_closes.index[-1]}")

    # Load DSP profile
    dsp, dsp_file = _load_dsp_profile(symbol)
    if dsp is None:
        click.secho(f"❌ No DSP profile found at {dsp_file}", fg="red")
        return False

    # Build config for both scenarios
    cfg_sample, spec = _build_engine_config(symbol, dsp, sizing_mode="static")
    _apply_strategy_overrides(
        cfg_sample,
        stop_bricks=stop_bricks_override,
        target_risk=target_risk_override,
        brick_size=brick_size_override,
        fliprate_window=fliprate_window_override,
        markov_window=markov_window_override,
        fliprate_threshold=fliprate_threshold_override,
        markov_threshold=markov_threshold_override,
    )

    # Print system specification BEFORE backtest
    click.echo()
    _print_system_spec(cfg_sample, spec, dsp)

    # Suppress detailed log messages during backtest
    original_log_level = LOG.level
    LOG.setLevel(logging.WARNING)

    # Run backtest with both sizing scenarios for XAUUSD
    scenarios = ["static", "compounding"] if symbol.upper() == "XAUUSD" else ["risk_based"]
    all_pass = True

    for scenario in scenarios:
        click.echo(f"--- {scenario.upper()} SIZING ---")

        # For compounding: use realistic lot ceiling (10.0)
        # For static: use min_lots (0.01)
        lot_ceiling = 10.0 if scenario == "compounding" else 0.01

        cfg, _ = _build_engine_config(symbol, dsp, sizing_mode=scenario, lot_ceiling=lot_ceiling)
        _apply_strategy_overrides(
            cfg,
            stop_bricks=stop_bricks_override,
            target_risk=target_risk_override,
            brick_size=brick_size_override,
            fliprate_window=fliprate_window_override,
            markov_window=markov_window_override,
            fliprate_threshold=fliprate_threshold_override,
            markov_threshold=markov_threshold_override,
        )
        engine = RenkoEngine(cfg)
        results = engine.backtest(test_closes)

        if "error" in results:
            click.secho(f"❌ Error: {results['error']}", fg="red")
            all_pass = False
            continue

        summary = results.get("summary", {})
        trades = results.get("trades", [])

        _print_stats(summary, symbol, f"backtest-{scenario}", trades=trades)

        n_trades = summary.get("n_trades", 0)
        omega = summary.get("omega", 0.0)
        passes = n_trades >= min_trades and omega >= min_omega

        if passes:
            click.secho("✅ PASS", fg="green")
        else:
            click.secho(
                f"❌ FAIL - trades={n_trades} (need {min_trades}), omega={omega:.2f} (need {min_omega})",
                fg="red",
            )
            all_pass = False

    # Restore original log level
    LOG.setLevel(original_log_level)

    return all_pass


# ──────────────────────────────────────────────────────────────────────────────
# STAGE 4: PAPER TRADING (Live Broker Data, No Real Orders)
# ──────────────────────────────────────────────────────────────────────────────


def stage_paper(
    symbol: str,
    months: int = 3,
    min_omega: float = 1.5,
    min_trades: int = 30,
    stop_bricks_override: Optional[float] = None,
    target_risk_override: Optional[float] = None,
    brick_size_override: Optional[float] = None,
    fliprate_window_override: Optional[int] = None,
    markov_window_override: Optional[int] = None,
    fliprate_threshold_override: Optional[float] = None,
    markov_threshold_override: Optional[float] = None,
    drift_adapt: bool = False,
    drift_opts: Optional[dict] = None,
) -> bool:
    """Paper trading with live broker data - NO REAL ORDERS.

    Connects to cTrader Open API to stream live M1 bars, runs the same
    RenkoEngine strategy logic as backtesting, and displays real-time stats.
    Uses PaperDispatcher to simulate fills without placing real orders.
    """
    click.echo(f"\n{'=' * 60}")
    click.secho("STAGE 4: PAPER TRADING (Live Data)", fg="cyan", bold=True)
    click.echo(f"{'=' * 60}")
    click.secho("⚠️  Paper trading - NO REAL ORDERS will be placed", fg="yellow")

    # Try to import cTrader connector
    try:
        from kinetra.connectors.ctrader_connector import build_connector
    except ImportError as e:
        click.secho(f"❌ cTrader connector not available: {e}", fg="red")
        click.echo("Install with: pip install ctrader-open-api")
        return False

    # Load DSP profile
    dsp, dsp_file = _load_dsp_profile(symbol)
    if dsp is None:
        click.secho(f"❌ No DSP profile found at {dsp_file}", fg="red")
        click.echo("Run --stage dsp first")
        return False

    # Shared runtime build (same engine path as live, paper dispatcher)
    cfg, spec, engine, bar_provider_cls, dispatcher_cls = _build_stream_runtime(
        symbol, dsp, lot_ceiling=0.01, live_orders=False
    )
    _apply_strategy_overrides(
        cfg,
        stop_bricks=stop_bricks_override,
        target_risk=target_risk_override,
        brick_size=brick_size_override,
        fliprate_window=fliprate_window_override,
        markov_window=markov_window_override,
        fliprate_threshold=fliprate_threshold_override,
        markov_threshold=markov_threshold_override,
    )
    adaptor = DriftAdaptationController(
        engine,
        {"enabled": drift_adapt, **(drift_opts or {})},
    )

    # Print system specification
    click.echo()
    _print_system_spec(cfg, spec, dsp)

    click.echo()
    click.secho("Connecting to cTrader...", fg="yellow")

    # Connect to cTrader
    try:
        connector = build_connector(timeout_s=30.0)
        click.secho("✅ Connected to cTrader", fg="green")
    except Exception as e:
        click.secho(f"❌ Failed to connect to cTrader: {e}", fg="red")
        click.echo("Check your .env.openapi credentials")
        return False

    # Create bar provider and paper dispatcher
    bar_provider = bar_provider_cls(connector)
    paper_dispatcher = dispatcher_cls(spread_pts={symbol: cfg.spread_ticks})

    # Stats tracking
    stats_lock = threading.Lock()
    last_stats_time = [time.time()]
    stats_interval = 60.0  # Print stats every 60 seconds
    trade_count = [0]
    stop_event = threading.Event()

    def print_periodic_stats():
        """Print stats periodically."""
        while not stop_event.is_set():
            time.sleep(stats_interval)
            with stats_lock:
                results = engine._make_results()
                summary = results.get("summary", {})
                trades = results.get("trades", [])
                n_trades = len(trades)
                if n_trades > trade_count[0]:
                    trade_count[0] = n_trades
                    adaptor.maybe_adapt(int(getattr(engine, "bricks_processed", 0)))
                    console.print()
                    _print_stats(summary, symbol, "paper", trades=trades)
                    last_stats_time[0] = time.time()

    # Start periodic stats thread
    stats_thread = threading.Thread(target=print_periodic_stats, daemon=True)
    stats_thread.start()

    click.echo()
    click.secho("Starting paper trading loop...", fg="cyan")
    click.echo("Streaming live M1 bars from cTrader")
    click.echo("Press Ctrl+C to stop and see final results")
    click.echo()

    try:
        results = _run_streaming_engine(
            engine,
            bar_provider,
            paper_dispatcher,
            stop_event=stop_event,
            stop_msg="Stopping paper trading...",
        )
    finally:
        stop_event.set()
        try:
            bar_provider.stop()
        except Exception:
            pass
        try:
            connector.stop()
        except Exception:
            pass

        # Print final stats
        summary = results.get("summary", {})
        trades = results.get("trades", [])

        click.echo()
        click.secho("=" * 60, fg="cyan")
        click.secho("PAPER TRADING RESULTS", fg="cyan", bold=True)
        click.secho("=" * 60, fg="cyan")
        _print_stats(summary, symbol, "paper", trades=trades)

        # Summary
        n_trades = summary.get("n_trades", 0)
        omega = summary.get("omega", 0.0)
        if n_trades >= min_trades and omega >= min_omega:
            click.secho(f"\n✅ PASS - {n_trades} trades, Omega={omega:.3f}", fg="green")
            return True
        else:
            click.secho(
                f"\n⚠️  Insufficient trades or Omega - {n_trades} trades (need {min_trades}), Omega={omega:.3f} (need {min_omega})",
                fg="yellow",
            )
            return False


# ──────────────────────────────────────────────────────────────────────────────
# MAIN CLI
# ──────────────────────────────────────────────────────────────────────────────


def stage_live_real(
    symbol: str = "XAUUSD",
    live_size: str = "micro",
    preflight_test_order: bool = False,
    ack_live: str = "",
    preflight_lots: float = 0.01,
    stop_bricks_override: Optional[float] = None,
    target_risk_override: Optional[float] = None,
    brick_size_override: Optional[float] = None,
    fliprate_window_override: Optional[int] = None,
    markov_window_override: Optional[int] = None,
    fliprate_threshold_override: Optional[float] = None,
    markov_threshold_override: Optional[float] = None,
    drift_adapt: bool = False,
    drift_opts: Optional[dict] = None,
) -> bool:
    """REAL live trading with REAL orders via cTrader."""
    import threading
    import time

    from kinetra.connectors.ctrader_connector import build_connector

    click.echo(f"\n{'=' * 60}")
    click.secho("STAGE 5: LIVE TRADING (REAL ORDERS)", fg="red", bold=True)
    click.echo(f"{'=' * 60}")

    LOT_CEILINGS = {"micro": 0.01, "small": 0.10, "full": 50.0}
    lot_ceiling = LOT_CEILINGS.get(live_size, 0.01)

    dsp, dsp_file = _load_dsp_profile(symbol)
    if dsp is None:
        click.secho("❌ No DSP profile found", fg="red")
        return False
    cfg, spec, engine, bar_provider_cls, dispatcher_cls = _build_stream_runtime(
        symbol, dsp, lot_ceiling=lot_ceiling, live_orders=True
    )
    _apply_strategy_overrides(
        cfg,
        stop_bricks=stop_bricks_override,
        target_risk=target_risk_override,
        brick_size=brick_size_override,
        fliprate_window=fliprate_window_override,
        markov_window=markov_window_override,
        fliprate_threshold=fliprate_threshold_override,
        markov_threshold=markov_threshold_override,
    )
    adaptor = DriftAdaptationController(
        engine,
        {"enabled": drift_adapt, **(drift_opts or {})},
    )

    click.echo()
    _print_system_spec(cfg, spec, dsp)

    click.echo()
    click.secho("=" * 60, fg="red")
    click.secho("  LIVE TRADING CONFIGURATION", fg="red", bold=True)
    click.secho("=" * 60, fg="red")
    click.echo(f"  Symbol:        {symbol}")
    click.echo(f"  Lot Ceiling:   {lot_ceiling:.2f} lots ({live_size})")
    click.echo()

    click.secho("Connecting to cTrader...", fg="yellow")
    try:
        connector = build_connector(timeout_s=30.0)
        click.secho("✅ Connected", fg="green")
    except Exception as e:
        click.secho(f"❌ Failed: {e}", fg="red")
        return False

    click.echo()
    click.secho("Running preflight checks...", fg="yellow", bold=True)
    if not _run_preflight_checks(connector, symbol):
        click.secho("❌ Preflight failed - aborting", fg="red")
        try:
            connector.stop()
        except Exception:
            pass
        return False
    click.secho("✅ Preflight passed", fg="green")
    click.echo()

    if preflight_test_order:
        if ack_live != "I_UNDERSTAND_LIVE_RISK":
            click.secho(
                '❌ Execution preflight blocked: pass --ack-live "I_UNDERSTAND_LIVE_RISK"',
                fg="red",
            )
            try:
                connector.stop()
            except Exception:
                pass
            return False
        if preflight_lots <= 0:
            click.secho("❌ Execution preflight blocked: --preflight-lots must be > 0", fg="red")
            try:
                connector.stop()
            except Exception:
                pass
            return False
        if not _run_execution_preflight_trade(
            connector=connector,
            symbol=symbol,
            cfg=cfg,
            bar_provider_cls=bar_provider_cls,
            dispatcher_cls=dispatcher_cls,
            lots=preflight_lots,
        ):
            try:
                connector.stop()
            except Exception:
                pass
            return False

    import logging

    logging.getLogger("kinetra.connectors").setLevel(logging.WARNING)
    logging.getLogger("kinetra.dns_hardening").setLevel(logging.WARNING)
    logging.getLogger("kinetra.renko.ctrader_dispatcher").setLevel(logging.WARNING)

    recent_errors = deque(maxlen=5)
    errors_lock = threading.Lock()

    class _DashboardErrorHandler(logging.Handler):
        """Collect recent ERROR log lines for the live dashboard."""

        def emit(self, record: logging.LogRecord) -> None:
            try:
                if record.levelno < logging.ERROR:
                    return
                ts = datetime.fromtimestamp(record.created, timezone.utc).strftime("%H:%M:%S")
                msg = f"{ts} {record.name}: {record.getMessage()}"
                if len(msg) > 120:
                    msg = msg[:117] + "..."
                with errors_lock:
                    recent_errors.appendleft(msg)
            except Exception:
                pass

    dashboard_error_handler = _DashboardErrorHandler()
    dashboard_error_handler.setLevel(logging.ERROR)
    # Attach once at the top of the hierarchy to avoid duplicate records
    # from child loggers propagating upward.
    dashboard_error_loggers = [logging.getLogger("kinetra")]
    for _lg in dashboard_error_loggers:
        _lg.addHandler(dashboard_error_handler)

    bar_provider = bar_provider_cls(connector)
    live_dispatcher = dispatcher_cls(connector, bar_provider)

    stats_lock = threading.Lock()
    bar_count = [0]
    stop_event = threading.Event()
    replace_panel = os.getenv("KINETRA_LIVE_DASHBOARD_REPLACE", "0").strip().lower() not in {
        "0",
        "false",
        "no",
        "off",
    }
    use_live_replace = bool(replace_panel and console.is_terminal and sys.stdout.isatty())
    if replace_panel and not use_live_replace:
        LOG.info(
            "Live dashboard replace mode disabled (stdout is not an interactive TTY); using periodic snapshots"
        )

    def _conn_status() -> str:
        hs = str(getattr(connector, "health_status", "") or "").upper()
        if hs in {"UP", "DEGRADED", "DOWN"}:
            return hs
        return "UP" if connector.is_connected() else "DOWN"

    last_acct_info = {
        "broker": "Pepperstone",
        "account_id": connector.credentials.account_id,
        "account_type": "ECN",
        "environment": connector.credentials.environment,
        "balance": engine._live_equity,
        "connection_status": _conn_status(),
        "endpoint": getattr(connector, "selected_endpoint", "-") or "-",
        "failover_count": int(getattr(connector, "failover_count", 0)),
        "failover_generation": int(getattr(connector, "failover_generation", 0)),
        "last_failover_utc": getattr(connector, "last_failover_utc", "-") or "-",
        "request_timeouts": int(getattr(connector, "request_timeout_count", 0)),
        "snapshot_source": "cached",
        "recent_errors": [],
    }
    try:
        init_snapshot = connector.get_account_snapshot(timeout_s=10.0)
        broker_balance = float(init_snapshot.get("balance", engine._live_equity))
        engine._live_equity = broker_balance
        last_acct_info = {
            "broker": init_snapshot.get("broker_name", "Pepperstone"),
            "account_id": init_snapshot.get("account_id", connector.credentials.account_id),
            "account_type": init_snapshot.get("account_type", "ECN"),
            "environment": connector.credentials.environment,
            "balance": broker_balance,
            "connection_status": _conn_status(),
            "endpoint": getattr(connector, "selected_endpoint", "-") or "-",
            "failover_count": int(getattr(connector, "failover_count", 0)),
            "failover_generation": int(getattr(connector, "failover_generation", 0)),
            "last_failover_utc": getattr(connector, "last_failover_utc", "-") or "-",
            "request_timeouts": int(getattr(connector, "request_timeout_count", 0)),
            "snapshot_source": "fresh",
            "recent_errors": [],
        }
    except Exception as e:
        LOG.warning("Initial account snapshot unavailable, using fallback equity: %s", e)

    def update_dashboard():
        last_bricks = 0
        last_print_ts = 0.0
        last_signature = None
        last_emit_ts = 0.0
        min_interval_default = "2" if use_live_replace else "30"
        min_interval_s = max(
            float(os.getenv("KINETRA_LIVE_DASHBOARD_MIN_INTERVAL_S", min_interval_default)), 2.0
        )
        min_emit_gap_s = max(
            float(os.getenv("KINETRA_LIVE_DASHBOARD_MIN_EMIT_GAP_S", "1.5")),
            0.5,
        )
        heartbeat_interval_s = max(
            float(os.getenv("KINETRA_LIVE_HEARTBEAT_INTERVAL_S", "2.0")), 1.0
        )
        heartbeat_on = False
        last_heartbeat_flip = time.time()
        while not stop_event.is_set():
            time.sleep(2)
            with stats_lock:
                current_bricks = int(getattr(engine, "bricks_processed", 0))
                now = time.time()
                if (now - last_heartbeat_flip) >= heartbeat_interval_s:
                    heartbeat_on = not heartbeat_on
                    last_heartbeat_flip = now
                bricks_advanced = current_bricks > last_bricks
                if bricks_advanced:
                    bar_count[0] = current_bricks
                    last_bricks = current_bricks
                    adaptor.maybe_adapt(current_bricks)
                if bricks_advanced:
                    try:
                        snapshot = connector.get_account_snapshot()
                        acct_info = {
                            "broker": snapshot.get("broker_name", "Pepperstone"),
                            "account_id": snapshot.get(
                                "account_id", connector.credentials.account_id
                            ),
                            "account_type": snapshot.get("account_type", "ECN"),
                            "environment": connector.credentials.environment,
                            "balance": snapshot.get("balance", engine._live_equity),
                            "connection_status": _conn_status(),
                            "endpoint": getattr(connector, "selected_endpoint", "-") or "-",
                            "failover_count": int(getattr(connector, "failover_count", 0)),
                            "failover_generation": int(
                                getattr(connector, "failover_generation", 0)
                            ),
                            "last_failover_utc": getattr(connector, "last_failover_utc", "-")
                            or "-",
                            "request_timeouts": int(getattr(connector, "request_timeout_count", 0)),
                            "snapshot_source": "fresh",
                            "heartbeat_on": heartbeat_on,
                        }
                        last_acct_info.update(acct_info)
                    except Exception:
                        acct_info = dict(last_acct_info)
                        acct_info["connection_status"] = _conn_status()
                        acct_info["endpoint"] = getattr(connector, "selected_endpoint", "-") or "-"
                        acct_info["failover_count"] = int(getattr(connector, "failover_count", 0))
                        acct_info["failover_generation"] = int(
                            getattr(connector, "failover_generation", 0)
                        )
                        acct_info["last_failover_utc"] = (
                            getattr(connector, "last_failover_utc", "-") or "-"
                        )
                        acct_info["request_timeouts"] = int(
                            getattr(connector, "request_timeout_count", 0)
                        )
                        acct_info["snapshot_source"] = "cached"
                        acct_info["heartbeat_on"] = heartbeat_on
                else:
                    acct_info = dict(last_acct_info)
                    acct_info["connection_status"] = _conn_status()
                    acct_info["endpoint"] = getattr(connector, "selected_endpoint", "-") or "-"
                    acct_info["failover_count"] = int(getattr(connector, "failover_count", 0))
                    acct_info["failover_generation"] = int(
                        getattr(connector, "failover_generation", 0)
                    )
                    acct_info["last_failover_utc"] = (
                        getattr(connector, "last_failover_utc", "-") or "-"
                    )
                    acct_info["request_timeouts"] = int(
                        getattr(connector, "request_timeout_count", 0)
                    )
                    acct_info["snapshot_source"] = "cached"
                    acct_info["heartbeat_on"] = heartbeat_on
                try:
                    with errors_lock:
                        err_lines = list(recent_errors)
                    summary = engine._make_results().get("summary", {})
                    last_eval = getattr(engine, "_last_eval", {}) or {}
                    acct_info["recent_errors"] = err_lines
                    signature = (
                        int(summary.get("n_trades", 0)),
                        str(last_eval.get("reason", "n/a")),
                        bool(last_eval.get("is_flip", False)),
                        bool(acct_info.get("heartbeat_on", False)) if use_live_replace else False,
                        str(acct_info.get("connection_status", "-")),
                        err_lines[0] if err_lines else "",
                    )
                    should_print = (
                        last_signature is None
                        or signature != last_signature
                        or (now - last_print_ts) >= min_interval_s
                    )
                    if should_print and (now - last_emit_ts) >= min_emit_gap_s:
                        # Set dedupe state before rendering to prevent duplicate
                        # emissions in case stdout writes interleave.
                        last_print_ts = now
                        last_signature = signature
                        last_emit_ts = now
                        panel = _dashboard_renderable(
                            engine, symbol, "LIVE", current_bricks, acct_info
                        )
                        if live_ctx is not None:
                            live_ctx.update(panel, refresh=True)
                        else:
                            _print_dashboard(engine, symbol, "LIVE", current_bricks, acct_info)
                except Exception as e:
                    LOG.warning("Dashboard update failed: %s", e)

    live_ctx = None
    dashboard_thread = None
    if use_live_replace:
        live_ctx = Live(console=console, refresh_per_second=4, transient=False)
        live_ctx.start()
    dashboard_thread = threading.Thread(target=update_dashboard, daemon=True)
    dashboard_thread.start()

    click.secho("Starting LIVE trading...", fg="red", bold=True)
    click.echo("REAL orders will be submitted")
    click.echo("Press Ctrl+C to stop")
    click.echo()

    try:
        results = _run_streaming_engine(
            engine,
            bar_provider,
            live_dispatcher,
            stop_event=stop_event,
            stop_msg="Stopping live trading...",
        )
    finally:
        stop_event.set()
        if live_ctx is not None:
            try:
                live_ctx.stop()
            except Exception:
                pass
        click.echo()
        click.secho("Closing connections...", fg="yellow")
        try:
            bar_provider.stop()
            connector.stop()
        except Exception as e:
            click.secho(f"  Warning: {e}", fg="yellow")
        for _lg in dashboard_error_loggers:
            try:
                _lg.removeHandler(dashboard_error_handler)
            except Exception:
                pass

        summary = results.get("summary", {})
        trades = results.get("trades", [])

        click.echo()
        click.secho("=" * 60, fg="cyan")
        click.secho("  LIVE TRADING RESULTS", fg="cyan", bold=True)
        click.secho("=" * 60, fg="cyan")
        _print_stats(summary, symbol, "LIVE", trades=trades)

        n_trades = summary.get("n_trades", 0)
        omega = summary.get("omega", 0.0)
        net_pnl = summary.get("net_usd", 0.0)

        click.echo()
        if n_trades > 0:
            click.secho(f"  Trades:  {n_trades}", fg="cyan")
            click.secho(f"  Net P&L: ${net_pnl:,.2f}", fg="green" if net_pnl > 0 else "red")
            click.secho(f"  Omega:   {omega:.3f}", fg="green" if omega >= 1.5 else "yellow")
        else:
            click.secho("  No trades executed", fg="yellow")

        click.echo()
        click.secho("⚠️  Verify positions in cTrader terminal", fg="yellow")

    return True


def stage_live_dryrun(
    symbol: str,
    live_size: str = "micro",
    stop_bricks_override: Optional[float] = None,
    target_risk_override: Optional[float] = None,
    brick_size_override: Optional[float] = None,
    fliprate_window_override: Optional[int] = None,
    markov_window_override: Optional[int] = None,
    fliprate_threshold_override: Optional[float] = None,
    markov_threshold_override: Optional[float] = None,
    drift_adapt: bool = False,
    drift_opts: Optional[dict] = None,
) -> bool:
    """Dry-run mode: live bars with paper orders via the shared engine path."""
    click.echo(f"\n{'=' * 60}")
    click.secho("STAGE 5: LIVE DRY-RUN (Shadow Mode)", fg="cyan", bold=True)
    click.echo(f"{'=' * 60}")
    click.secho("Using live data with paper fills (no real orders)", fg="yellow")
    # Keep gate thresholds aligned with paper stage defaults.
    return stage_paper(
        symbol=symbol,
        months=3,
        min_omega=1.5,
        min_trades=30,
        stop_bricks_override=stop_bricks_override,
        target_risk_override=target_risk_override,
        brick_size_override=brick_size_override,
        fliprate_window_override=fliprate_window_override,
        markov_window_override=markov_window_override,
        fliprate_threshold_override=fliprate_threshold_override,
        markov_threshold_override=markov_threshold_override,
        drift_adapt=drift_adapt,
        drift_opts=drift_opts,
    )


@click.command()
@click.argument("symbol", default="XAUUSD")
@click.option("--dry-run", is_flag=True, help="Paper trading with live data")
@click.option(
    "--live-size",
    type=click.Choice(["micro", "small", "full"]),
    default="micro",
    help="Lot size for live trading",
)
@click.option(
    "--stage",
    type=click.Choice(["dsp", "backtest", "paper", "live", "all"]),
    default="backtest",
    help="Which stage to run",
)
@click.option("--months", type=int, default=3, help="Months of data for backtest")
@click.option("--min-omega", type=float, default=1.5, help="Minimum Omega ratio to pass")
@click.option("--min-trades", type=int, default=30, help="Minimum trades to pass")
@click.option("--stop-bricks", type=float, default=None, help="Override stop distance in bricks")
@click.option("--target-risk", type=float, default=None, help="Override target USD risk per trade")
@click.option(
    "--brick-size", type=float, default=None, help="Override Renko brick size (price units)"
)
@click.option(
    "--fliprate-window", type=int, default=None, help="Override flip-rate window (bricks)"
)
@click.option("--markov-window", type=int, default=None, help="Override Markov window (bricks)")
@click.option(
    "--fliprate-threshold", type=float, default=None, help="Override flip-rate gate threshold"
)
@click.option("--markov-threshold", type=float, default=None, help="Override Markov gate threshold")
@click.option("--monday-start", type=str, default="", help="Legacy launcher option (reserved)")
@click.option("--friday-end", type=str, default="", help="Legacy launcher option (reserved)")
@click.option("--auto-download", is_flag=True, help="Legacy launcher option (reserved)")
@click.option(
    "--preflight-test-order/--no-preflight-test-order",
    default=False,
    help="Place/close tiny real order before live loop",
)
@click.option(
    "--preflight-lots",
    type=float,
    default=0.01,
    help="Lots for execution preflight test order",
)
@click.option(
    "--ack-live",
    type=str,
    default="",
    help='Required when --preflight-test-order. Must equal "I_UNDERSTAND_LIVE_RISK".',
)
@click.option(
    "--drift-adapt/--no-drift-adapt", default=False, help="Enable bounded online drift adaptation"
)
@click.option(
    "--adapt-every-bars", type=int, default=100, help="Adaptation evaluation cadence in bricks"
)
@click.option(
    "--adapt-min-trades", type=int, default=20, help="Minimum trades before adaptation is active"
)
@click.option(
    "--adapt-low-omega", type=float, default=1.0, help="Defensive adaptation trigger omega"
)
@click.option(
    "--adapt-high-omega", type=float, default=1.8, help="Offensive adaptation trigger omega"
)
@click.option(
    "--adapt-low-win-rate", type=float, default=0.45, help="Defensive adaptation trigger win rate"
)
@click.option(
    "--adapt-high-win-rate", type=float, default=0.60, help="Offensive adaptation trigger win rate"
)
@click.option(
    "--adapt-step-risk", type=float, default=0.10, help="Fractional risk step per adaptation"
)
@click.option(
    "--adapt-step-threshold", type=float, default=0.01, help="Threshold step per adaptation"
)
@click.option(
    "--adapt-step-stop", type=float, default=0.05, help="Stop-bricks fractional step per adaptation"
)
@click.option("--adapt-risk-min", type=float, default=25.0, help="Minimum target risk cap")
@click.option("--adapt-risk-max", type=float, default=200.0, help="Maximum target risk cap")
@click.option("--adapt-stop-min", type=float, default=0.30, help="Minimum stop_bricks cap")
@click.option("--adapt-stop-max", type=float, default=1.50, help="Maximum stop_bricks cap")
@click.option(
    "--adapt-fliprate-min", type=float, default=0.25, help="Minimum fliprate threshold cap"
)
@click.option(
    "--adapt-fliprate-max", type=float, default=0.45, help="Maximum fliprate threshold cap"
)
@click.option("--adapt-markov-min", type=float, default=0.50, help="Minimum markov threshold cap")
@click.option("--adapt-markov-max", type=float, default=0.70, help="Maximum markov threshold cap")
@click.option("--adapt-rollback-omega", type=float, default=0.80, help="Rollback trigger omega")
def main(
    symbol: str = "XAUUSD",
    stage: str = "backtest",
    months: int = 3,
    min_omega: float = 1.5,
    min_trades: int = 30,
    dry_run: bool = False,
    live_size: str = "micro",
    stop_bricks: Optional[float] = None,
    target_risk: Optional[float] = None,
    brick_size: Optional[float] = None,
    fliprate_window: Optional[int] = None,
    markov_window: Optional[int] = None,
    fliprate_threshold: Optional[float] = None,
    markov_threshold: Optional[float] = None,
    monday_start: str = "",
    friday_end: str = "",
    auto_download: bool = False,
    preflight_test_order: bool = False,
    preflight_lots: float = 0.01,
    ack_live: str = "",
    drift_adapt: bool = False,
    adapt_every_bars: int = 100,
    adapt_min_trades: int = 20,
    adapt_low_omega: float = 1.0,
    adapt_high_omega: float = 1.8,
    adapt_low_win_rate: float = 0.45,
    adapt_high_win_rate: float = 0.60,
    adapt_step_risk: float = 0.10,
    adapt_step_threshold: float = 0.01,
    adapt_step_stop: float = 0.05,
    adapt_risk_min: float = 25.0,
    adapt_risk_max: float = 200.0,
    adapt_stop_min: float = 0.30,
    adapt_stop_max: float = 1.50,
    adapt_fliprate_min: float = 0.25,
    adapt_fliprate_max: float = 0.45,
    adapt_markov_min: float = 0.50,
    adapt_markov_max: float = 0.70,
    adapt_rollback_omega: float = 0.80,
):
    """Kinetra Renko Brick Trade Engine.

    Validate a symbol through the full pipeline:
    DSP → Backtest → Paper → Live
    """
    click.echo(f"{'=' * 70}")
    click.secho(f"Kinetra Renko Engine: {symbol}", fg="cyan", bold=True)
    click.echo(f"{'=' * 70}")
    drift_opts = {
        "every_bars": adapt_every_bars,
        "min_trades": adapt_min_trades,
        "low_omega": adapt_low_omega,
        "high_omega": adapt_high_omega,
        "low_wr": adapt_low_win_rate,
        "high_wr": adapt_high_win_rate,
        "step_risk": adapt_step_risk,
        "step_thr": adapt_step_threshold,
        "step_stop": adapt_step_stop,
        "risk_min": adapt_risk_min,
        "risk_max": adapt_risk_max,
        "stop_min": adapt_stop_min,
        "stop_max": adapt_stop_max,
        "flip_min": adapt_fliprate_min,
        "flip_max": adapt_fliprate_max,
        "markov_min": adapt_markov_min,
        "markov_max": adapt_markov_max,
        "rollback_omega": adapt_rollback_omega,
    }
    if monday_start or friday_end or auto_download:
        LOG.info(
            "Accepted legacy launcher options: monday_start=%s friday_end=%s auto_download=%s",
            monday_start or "-",
            friday_end or "-",
            auto_download,
        )

    if stage in ("dsp", "all"):
        if not stage_dsp(symbol):
            click.secho("❌ DSP stage failed", fg="red")
            raise SystemExit(1)

    if stage in ("backtest", "all"):
        if not stage_backtest(
            symbol,
            months=months,
            min_omega=min_omega,
            min_trades=min_trades,
            stop_bricks_override=stop_bricks,
            target_risk_override=target_risk,
            brick_size_override=brick_size,
            fliprate_window_override=fliprate_window,
            markov_window_override=markov_window,
            fliprate_threshold_override=fliprate_threshold,
            markov_threshold_override=markov_threshold,
        ):
            click.secho("❌ Backtest stage failed", fg="red")
            raise SystemExit(1)

    if stage in ("paper", "all"):
        if not stage_paper(
            symbol,
            months=months,
            min_omega=min_omega,
            min_trades=min_trades,
            stop_bricks_override=stop_bricks,
            target_risk_override=target_risk,
            brick_size_override=brick_size,
            fliprate_window_override=fliprate_window,
            markov_window_override=markov_window,
            fliprate_threshold_override=fliprate_threshold,
            markov_threshold_override=markov_threshold,
            drift_adapt=drift_adapt,
            drift_opts=drift_opts,
        ):
            click.secho("❌ Paper stage failed", fg="red")
            raise SystemExit(1)

    if stage == "live":
        if dry_run:
            # Dry-run: live data, paper orders
            if not stage_live_dryrun(
                symbol,
                live_size=live_size,
                stop_bricks_override=stop_bricks,
                target_risk_override=target_risk,
                brick_size_override=brick_size,
                fliprate_window_override=fliprate_window,
                markov_window_override=markov_window,
                fliprate_threshold_override=fliprate_threshold,
                markov_threshold_override=markov_threshold,
                drift_adapt=drift_adapt,
                drift_opts=drift_opts,
            ):
                raise SystemExit(1)
        else:
            # REAL live trading
            if not stage_live_real(
                symbol,
                live_size=live_size,
                preflight_test_order=preflight_test_order,
                preflight_lots=preflight_lots,
                ack_live=ack_live,
                stop_bricks_override=stop_bricks,
                target_risk_override=target_risk,
                brick_size_override=brick_size,
                fliprate_window_override=fliprate_window,
                markov_window_override=markov_window,
                fliprate_threshold_override=fliprate_threshold,
                markov_threshold_override=markov_threshold,
                drift_adapt=drift_adapt,
                drift_opts=drift_opts,
            ):
                raise SystemExit(1)

    click.secho("✅ All stages passed!", fg="green")


if __name__ == "__main__":
    main()
