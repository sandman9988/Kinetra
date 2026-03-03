#!/usr/bin/env python3
"""
Kinetra Pipeline v3 - Simplified Qualification & Trading Flow
==============================================================

New pipeline (your spec):
  1. Download M1 data for a symbol
  2. 3-month DSP backtest (quick screen)
     → If fail: disqualify, move to next
     → If pass: continue to step 3
  3. Download 1-3 years M1 data
  4. Multi-window walk-forward backtest
     → If fail: disqualify
     → If pass: qualify & enable trading
  5. Paper trading gate
     → If pass (Omega ≥ 1.5): unlock micro
  6. Micro lots gate (0.01)
     → If pass (Omega ≥ 2.0): unlock small
  7. Small lots gate (0.1)
     → If pass (Omega ≥ 2.5): unlock full

Entry point: python scripts/pipeline_v3.py [--symbol XAUUSD] [--mode paper|live]
"""

import json
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import click
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from kinetra.config import PROJECT_ROOT as KR
from kinetra.renko.live_trader import LiveTraderConfig, PERGate, RenkoLiveTrader
from kinetra.renko.qualify import QualificationRegistry

# ═══════════════════════════════════════════════════════════════════════════════
# Stage 1: Quick 3-Month DSP Screen
# ═══════════════════════════════════════════════════════════════════════════════


def stage_1_dsp_screen(symbol: str) -> bool:
    """
    Quick backtest on 3 months of data to verify brick size viability.

    Returns True if passes (move to Stage 2), False if disqualify.
    """
    click.echo(f"\n{'=' * 70}")
    click.secho(f"STAGE 1: DSP BRICK SIZE SCREEN (3 months)", fg="cyan", bold=True)
    click.echo(f"{'=' * 70}\n")

    click.echo(f"Symbol: {symbol}")
    click.echo(f"Timeframe: M1 (last 3 months)")
    click.echo(f"Purpose: Verify brick size viability via DSP analysis")

    # TODO: Implement DSP screen
    # - Load 3 months M1 data
    # - Run DSP analysis (vr_profile, brick_from_scale)
    # - Quick 100-trade backtest
    # - Check: Omega > 0.5 OR Z-score > 1.0

    click.secho("\n⚠️  Stage 1 (DSP screen) not yet implemented", fg="yellow")
    should_skip = click.confirm("Skip to Stage 3 (full backtest)?")

    return should_skip  # Return True to proceed to Stage 3


# ═══════════════════════════════════════════════════════════════════════════════
# Stage 3: Full Multi-Window Backtest
# ═══════════════════════════════════════════════════════════════════════════════


def stage_3_multi_window_backtest(symbol: str) -> bool:
    """
    Walk-forward backtest on 1-3 years of data with multiple rolling windows.

    Windows:
      - IS: 70% (learning)
      - OOS: 30% (testing)
      - Roll every 3 months

    Returns True if passes, False if disqualify.
    """
    click.echo(f"\n{'=' * 70}")
    click.secho(f"STAGE 3: MULTI-WINDOW WALK-FORWARD BACKTEST (1-3 years)", fg="cyan", bold=True)
    click.echo(f"{'=' * 70}\n")

    click.echo(f"Symbol: {symbol}")
    click.echo(f"Timeframe: M1 (1-3 years)")
    click.echo(f"Windows: 70/30 IS/OOS, rolling 3-month step")
    click.echo(f"Pass criteria: Omega ≥ 1.5 in both IS and OOS, survival ≥ 80%")

    # TODO: Implement multi-window backtest
    # - Load 1-3 years M1 data
    # - Split into rolling 70/30 windows
    # - For each window:
    #   - Train on IS
    #   - Test on OOS
    #   - Check Omega IS >= 1.5 AND OOS >= 1.2 AND survival >= 80%
    # - If all windows pass: QUALIFY

    click.secho("\n⚠️  Stage 3 (multi-window backtest) not yet implemented", fg="yellow")
    return click.confirm("Mark as qualified anyway (for testing)?")


# ═══════════════════════════════════════════════════════════════════════════════
# Qualification Status
# ═══════════════════════════════════════════════════════════════════════════════


def get_qual_status(symbol: str) -> dict:
    """Load qualification status."""
    qual_dir = KR / "data" / "renko_qualified" / symbol
    qual_file = qual_dir / "qualification.json"

    if not qual_file.exists():
        return {
            "qualified": False,
            "gate": "paper",
            "reason": "not-yet-tested",
            "omega": 0.0,
            "n_trades": 0,
        }

    try:
        with open(qual_file) as f:
            qual = json.load(f)
        return {
            "qualified": qual.get("qualified", False),
            "gate": "paper" if qual.get("qualified") else "none",
            "reason": qual.get("disqualification_reason", "unknown"),
            "omega": qual.get("omega", 0.0),
            "n_trades": qual.get("n_trades", 0),
        }
    except Exception:
        return {
            "qualified": False,
            "gate": "none",
            "reason": "corrupt-qualification-json",
            "omega": 0.0,
            "n_trades": 0,
        }


def discover_symbols() -> list[str]:
    """Find all downloaded symbols."""
    data_root = KR / "data" / "master_standardized" / "ctrader"
    if not data_root.exists():
        return []

    symbols = set()
    for account_dir in data_root.iterdir():
        if account_dir.is_dir():
            for sym_dir in account_dir.iterdir():
                if sym_dir.is_dir():
                    symbols.add(sym_dir.name)

    return sorted(symbols)


# ═══════════════════════════════════════════════════════════════════════════════
# Main CLI
# ═══════════════════════════════════════════════════════════════════════════════


@click.group()
def cli():
    """Kinetra Pipeline v3 - Simplified qualification & trading."""
    pass


@cli.command()
@click.option("--symbol", "-s", default=None, help="Symbol to screen")
def screen(symbol: Optional[str]):
    """Run qualification screens (Stage 1 & 3)."""

    symbols = discover_symbols()
    if not symbols:
        click.secho("❌ No symbols found", fg="red")
        return

    # Select symbol
    if symbol is None:
        click.echo("\n📊 Available symbols:")
        for i, sym in enumerate(symbols, 1):
            qual = get_qual_status(sym)
            status = "✅ QUAL" if qual["qualified"] else "⏳ SCREEN"
            click.echo(f"  {i}. {sym:12s} {status:12s} (Ω={qual['omega']:.2f})")

        idx = click.prompt("\nSelect symbol", type=click.IntRange(1, len(symbols))) - 1
        symbol = symbols[idx]

    qual = get_qual_status(symbol)

    click.echo(f"\n{'=' * 70}")
    click.secho(f"QUALIFICATION STATUS: {symbol}", fg="cyan", bold=True)
    click.echo(f"{'=' * 70}")
    click.echo(f"Qualified: {'✅ YES' if qual['qualified'] else '❌ NO'}")
    click.echo(f"Gate: {qual['gate']}")
    click.echo(f"Omega: {qual['omega']:.2f}")
    click.echo(f"Trades: {qual['n_trades']}")
    click.echo(f"Reason: {qual['reason']}")

    if qual["qualified"]:
        click.secho("\n✅ Already qualified - ready for trading", fg="green")
        return

    # Run screening pipeline
    should_proceed = click.confirm("\nStart qualification pipeline?")
    if not should_proceed:
        return

    # Stage 1: DSP screen
    if not stage_1_dsp_screen(symbol):
        click.secho(f"\n❌ {symbol} disqualified at Stage 1 (DSP screen)", fg="red")
        return

    # Stage 3: Multi-window backtest
    if not stage_3_multi_window_backtest(symbol):
        click.secho(f"\n❌ {symbol} disqualified at Stage 3 (multi-window)", fg="red")
        return

    click.secho(f"\n✅ {symbol} QUALIFIED - ready for paper trading", fg="green")


@cli.command()
@click.option("--symbol", "-s", required=True, help="Symbol to trade")
@click.option(
    "--gate",
    "-g",
    default="paper",
    type=click.Choice(["paper", "micro", "small", "full"]),
    help="PER gate",
)
@click.option(
    "--mode",
    "-m",
    default="dry_run",
    type=click.Choice(["paper", "dry_run", "live"]),
    help="Trading mode",
)
def trade(symbol: str, gate: str, mode: str):
    """Start live trading."""

    qual = get_qual_status(symbol)

    click.echo(f"\n{'=' * 70}")
    click.secho(f"LIVE TRADING: {symbol}", fg="cyan", bold=True)
    click.echo(f"{'=' * 70}")
    click.echo(f"Gate: {gate}")
    click.echo(f"Mode: {mode}")
    click.echo(f"Qualified: {'✅ YES' if qual['qualified'] else '❌ NO (trading unqualified)'}")
    click.echo(f"Current Omega: {qual['omega']:.2f}")

    if not qual["qualified"]:
        proceed = click.confirm("\n⚠️  Symbol not qualified. Proceed anyway?")
        if not proceed:
            return

    should_start = click.confirm("\nStart trading?")
    if not should_start:
        return

    # Launch trader
    click.secho(f"\n▶️  Launching trader...", fg="green")

    config = LiveTraderConfig(
        symbols=[symbol],
        gate=PERGate(gate),
        target_risk_usd=100.0,
        startup_skip_flips=2,
        allow_short=True,
        skip_qualification=not qual["qualified"],  # Allow unqualified if confirmed
    )

    trader = RenkoLiveTrader(config)
    try:
        trader.start()
    except KeyboardInterrupt:
        click.echo("\n\n⏹️  Stopping...")
        trader.stop()
        click.secho("✅ Shutdown complete", fg="green")


@cli.command()
def status():
    """Show system status."""

    symbols = discover_symbols()
    qualified = [s for s in symbols if get_qual_status(s)["qualified"]]

    click.echo(f"\n{'=' * 70}")
    click.secho(f"SYSTEM STATUS", fg="cyan", bold=True)
    click.echo(f"{'=' * 70}")
    click.echo(f"Total symbols: {len(symbols)}")
    click.echo(f"Qualified: {len(qualified)}")
    click.echo(f"Screening: {len(symbols) - len(qualified)}")

    if symbols:
        click.echo(f"\nSymbols:")
        for sym in symbols:
            qual = get_qual_status(sym)
            status = "✅" if qual["qualified"] else "⏳"
            click.echo(f"  {status} {sym:12s} Ω={qual['omega']:6.2f} {qual['reason']}")


if __name__ == "__main__":
    cli()
