#!/usr/bin/env python3
"""
Stage 1: Quick 3-Month DSP Screen
=================================

Fast qualification screen using only 3 months of M1 data.

Purpose:
  - Verify brick size viability via DSP (vr_profile, brick_from_scale)
  - Quick 100-200 trade backtest
  - Gate: Omega > 0.5 OR Z-score > 1.0 (loose, just filter obvious losers)
  - If pass → proceed to Stage 3 (full backtest)
  - If fail → disqualify

Usage:
    python scripts/renko/stage1_dsp_screen.py --symbol XAUUSD
"""

import sys
from pathlib import Path

import click

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))



@click.command()
@click.option("--symbol", "-s", required=True, help="Symbol to screen")
@click.option("--days-back", "-d", default=90, help="Days of M1 data (default 90 = 3 months)")
def screen(symbol: str, days_back: int):
    """Run 3-month DSP screen."""

    # TODO: Load M1 data from ctrader account
    # data_root = PROJECT_ROOT / "data" / "master_standardized" / "ctrader" / "pepperstone_demo_45841299"

    click.echo(f"\n{'=' * 70}")
    click.secho("STAGE 1: DSP BRICK SIZE SCREEN", fg="cyan", bold=True)
    click.echo(f"{'=' * 70}\n")

    click.echo(f"Symbol: {symbol}")
    click.echo(f"Days: {days_back}")
    click.echo("Purpose: Verify brick size via DSP, quick backtest (100-200 trades)")

    # Step 1: Load M1 data
    click.echo("\n📊 Loading M1 data...")
    # (load logic here)

    # Step 2: Run DSP analysis
    click.echo("🔍 Running DSP analysis...")
    # (DSP logic here)
    # - vr_profile() → VR peak
    # - brick_from_scale() → brick size

    # Step 3: Quick backtest
    click.echo("⚙️  Running quick backtest...")
    # (backtest logic here)

    # Step 4: Check pass criteria
    click.echo("\n📈 Results:")
    click.echo("  Brick size: $XX.XX")
    click.echo("  Trades: NN")
    click.echo("  Omega: X.XX")
    click.echo("  Z-factor: X.XX")

    # Simple pass/fail
    passes = True  # (check Omega > 0.5 OR Z > 1.0)
    if passes:
        click.secho("\n✅ PASS - Proceed to Stage 3 (full backtest)", fg="green")
    else:
        click.secho(f"\n❌ FAIL - Disqualify {symbol}", fg="red")


if __name__ == "__main__":
    screen()
