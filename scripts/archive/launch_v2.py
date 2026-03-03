#!/usr/bin/env python3
"""
Kinetra Live Trading Launcher v2
==================================

Simplified pipeline:
  1. Download M1 data
  2. 3-month DSP screen (quick pass/fail)
  3. Full backtest (1-3 years, multi-window)
  4. PER gates: paper → micro → small → full

Usage:
    python scripts/launch_v2.py

Then select:
  - Symbol (discover from data/)
  - Gate (paper/micro/small/full)
  - Mode (dry-run / live)
"""

import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

import click

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_ROOT = PROJECT_ROOT / "data" / "master_standardized" / "ctrader" / "pepperstone_demo_45841299"
QUAL_ROOT = PROJECT_ROOT / "data" / "renko_qualified"


def discover_symbols() -> list[str]:
    """Find all downloaded symbols."""
    if not DATA_ROOT.exists():
        return []
    symbols = []
    for account_dir in DATA_ROOT.iterdir():
        if account_dir.is_dir():
            for sym_dir in account_dir.iterdir():
                if sym_dir.is_dir() and sym_dir.name not in symbols:
                    symbols.append(sym_dir.name)
    return sorted(symbols)


def get_qual_status(symbol: str) -> dict:
    """Load qualification status for symbol."""
    qual_file = QUAL_ROOT / symbol / "qualification.json"
    if not qual_file.exists():
        return {"qualified": False, "reason": "not-yet-tested"}
    with open(qual_file) as f:
        qual = json.load(f)

    return {
        "qualified": qual.get("qualified", False),
        "reason": qual.get("disqualification_reason", "unknown"),
        "omega": qual.get("omega", 0.0),
        "n_trades": qual.get("n_trades", 0),
    }


@click.command()
@click.option("--symbol", "-s", default=None, help="Symbol (auto-select if not specified)")
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
def main(symbol: str, gate: str, mode: str):
    """Launch Renko live trader with simplified qualification flow."""

    # Discover symbols
    symbols = discover_symbols()
    if not symbols:
        click.secho("❌ No symbols found in data/", fg="red")
        sys.exit(1)

    # Select symbol if not specified
    if symbol is None:
        click.echo("\n📊 Available symbols:")
        for i, sym in enumerate(symbols, 1):
            qual = get_qual_status(sym)
            status = "✅ QUAL" if qual["qualified"] else "⏳ SCREEN"
            click.echo(f"  {i}. {sym:12s} {status:12s} (Ω={qual['omega']:.2f})")

        idx = click.prompt("Select symbol", type=click.IntRange(1, len(symbols))) - 1
        symbol = symbols[idx]

    # Check qualification
    qual = get_qual_status(symbol)
    if not qual["qualified"]:
        click.secho(
            f"\n⚠️  {symbol} not qualified: {qual['reason']}\n"
            f"   Ω = {qual['omega']:.2f}, n_trades = {qual['n_trades']}\n"
            f"   Run: python scripts/renko/qualify_instruments.py --symbol {symbol}",
            fg="yellow",
        )
        should_proceed = click.confirm("Proceed with trading anyway?")
        if not should_proceed:
            click.echo("Cancelled.")
            sys.exit(0)

    # Confirm mode
    click.echo()
    click.secho(f"Launching {mode.upper()} mode for {symbol} @ {gate} gate", fg="green", bold=True)

    if not click.confirm("Proceed?"):
        click.echo("Cancelled.")
        sys.exit(0)

    # Launch trader
    from kinetra.renko.live_trader import LiveTraderConfig, PERGate, RenkoLiveTrader

    config = LiveTraderConfig(
        symbols=[symbol],
        gate=PERGate(gate),
        target_risk_usd=100.0,
        startup_skip_flips=2,
        allow_short=True,
    )

    trader = RenkoLiveTrader(config)
    try:
        click.echo(f"\n▶️  Trader started. Press Ctrl+C to stop.\n")
        trader.start()
    except KeyboardInterrupt:
        click.echo("\n\n⏹️  Stopping...")
        trader.stop()
        click.secho("✅ Shutdown complete", fg="green")


if __name__ == "__main__":
    main()
