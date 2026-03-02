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
import sys
import threading
import time
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
from rich.console import Console
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
            f"{len(engine._dir_deque)}",
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
    return Panel(t, title=f"[bold]{symbol}[/]  [{mode.upper()}]  {ts}", border_style=border)


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
    symbol: str, months: int = 3, min_omega: float = 1.5, min_trades: int = 30
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
    dsp_dir = get_data_path(symbol)
    dsp_file = dsp_dir / "dsp_profile.json"
    if not dsp_file.exists():
        click.secho(f"❌ No DSP profile found at {dsp_file}", fg="red")
        return False

    with open(dsp_file) as f:
        dsp = json.load(f)

    # Build config for both scenarios
    cfg_sample, spec = _build_engine_config(symbol, dsp, sizing_mode="static")

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
        from kinetra.renko.ctrader_dispatcher import CTraderBarProvider
        from kinetra.renko.live_trader import PaperDispatcher
    except ImportError as e:
        click.secho(f"❌ cTrader connector not available: {e}", fg="red")
        click.echo("Install with: pip install ctrader-open-api")
        return False

    # Load DSP profile
    dsp_dir = get_data_path(symbol)
    dsp_file = dsp_dir / "dsp_profile.json"
    if not dsp_file.exists():
        click.secho(f"❌ No DSP profile found at {dsp_file}", fg="red")
        click.echo("Run --stage dsp first")
        return False

    with open(dsp_file) as f:
        dsp = json.load(f)

    # Build engine config (use static sizing for paper trading)
    cfg, spec = _build_engine_config(symbol, dsp, sizing_mode="static", lot_ceiling=0.01)

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
    bar_provider = CTraderBarProvider(connector)
    paper_dispatcher = PaperDispatcher(spread_pts={symbol: cfg.spread_ticks})

    # Create engine
    engine = RenkoEngine(cfg)

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
        # Run the engine with paper dispatcher
        results = engine.run(
            bar_provider=bar_provider,
            dispatcher=paper_dispatcher,
            stop_event=stop_event,
        )

    except KeyboardInterrupt:
        click.echo()
        click.secho("Stopping paper trading...", fg="yellow")
        stop_event.set()

        # Get final results
        results = engine._make_results()

    finally:
        # Cleanup
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


@click.command()
@click.argument("symbol", default="XAUUSD")
@click.option(
    "--stage",
    type=click.Choice(["dsp", "backtest", "paper", "live", "all"]),
    default="backtest",
    help="Which stage to run",
)
@click.option("--months", type=int, default=3, help="Months of data for backtest")
@click.option("--min-omega", type=float, default=1.5, help="Minimum Omega ratio to pass")
@click.option("--min-trades", type=int, default=30, help="Minimum trades to pass")
def main(symbol: str, stage: str, months: int, min_omega: float, min_trades: int):
    """Kinetra Renko Brick Trade Engine.

    Validate a symbol through the full pipeline:
    DSP → Backtest → Paper → Live
    """
    click.echo(f"{'=' * 70}")
    click.secho(f"Kinetra Renko Engine: {symbol}", fg="cyan", bold=True)
    click.echo(f"{'=' * 70}")

    if stage in ("dsp", "all"):
        if not stage_dsp(symbol):
            click.secho("❌ DSP stage failed", fg="red")
            raise SystemExit(1)

    if stage in ("backtest", "all"):
        if not stage_backtest(symbol, months=months, min_omega=min_omega, min_trades=min_trades):
            click.secho("❌ Backtest stage failed", fg="red")
            raise SystemExit(1)

    if stage in ("paper", "all"):
        if not stage_paper(symbol, months=months, min_omega=min_omega, min_trades=min_trades):
            click.secho("❌ Paper stage failed", fg="red")
            raise SystemExit(1)

    if stage == "live":
        click.secho("⚠️  LIVE stage requires broker connection", fg="yellow")
        click.echo("Use --stage paper for simulation mode")
        raise SystemExit(1)

    click.secho("✅ All stages passed!", fg="green")


if __name__ == "__main__":
    main()
