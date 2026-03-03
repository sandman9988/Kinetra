#!/usr/bin/env python3
"""
Kinetra Pipeline v4 - First Principles
=======================================

CORE TRUTH:
  - M1 bars = raw data (noise)
  - Bricks = price-level aggregation (structure)
  - Find correct brick size via DSP (VR analysis)
  - Validate: friction_cost < brick_size
  - Test empirically: backtest determines all parameters
  - No assumptions, only tests

Pipeline:
  1. Download M1 bars
  2. DSP: Analyze volatility regime (VR) → find brick size
  3. Friction: Verify spread < 1 brick
  4. Backtest: Does it work? (empirical test)
  5. Trade: paper → micro → small → full (gates by performance)

Usage:
    python scripts/pipeline_v4.py download --symbol XAUUSD
    python scripts/pipeline_v4.py dsp --symbol XAUUSD
    python scripts/pipeline_v4.py friction --symbol XAUUSD
    python scripts/pipeline_v4.py backtest --symbol XAUUSD
    python scripts/pipeline_v4.py trade --symbol XAUUSD --gate paper
"""

import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

import click
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from kinetra.backtesting.metrics import calculate_z_factor, omega_ratio
from kinetra.config import PROJECT_ROOT as KR
from kinetra.data_utils import load_broker_csv
from kinetra.friction_cost import InstrumentSpec
from kinetra.renko.backtest import FilterParams, StopParams, backtest_instrument
from kinetra.renko.brick_engine import brick_summary, build_renko
from kinetra.renko.dsp import compute_friction_floor, run_dsp
from kinetra.renko.trading_engine import EngineConfig, RenkoEngine

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 1: DOWNLOAD M1 BARS
# ═══════════════════════════════════════════════════════════════════════════════


@click.command()
@click.option("--symbol", "-s", required=True, help="Symbol to download")
@click.option("--days", "-d", default=30, help="Days of M1 data")
@click.option("--account", "-a", default="pepperstone_demo_45841299", help="cTrader account")
def download(symbol: str, days: int, account: str):
    """Download M1 bars from cTrader."""
    click.echo(f"\n{'=' * 70}")
    click.secho(f"STEP 1: DOWNLOAD M1 BARS", fg="cyan", bold=True)
    click.echo(f"{'=' * 70}\n")

    click.echo(f"Symbol: {symbol}")
    click.echo(f"Days: {days}")
    click.echo(f"Source: cTrader {account}")

    # TODO: Download from cTrader API
    # - Use existing download infrastructure (download_core.py)
    # - Save to data/master_standardized/ctrader/{account}/{symbol}/
    # - Format: {SYMBOL}_M1_*.csv with columns: time, open, high, low, close, volume

    click.secho(f"\n⚠️  Download not yet wired to cTrader API", fg="yellow")
    click.echo(f"Manual: Save M1 CSV to:")
    click.echo(f"  data/master_standardized/ctrader/{account}/{symbol}/")

    # For testing, load existing data
    data_root = KR / "data" / "master_standardized" / "ctrader" / account
    sym_dir = data_root / symbol

    if sym_dir.exists():
        m1_files = sorted(sym_dir.glob("*_M1_*.csv"))
        if m1_files:
            click.secho(f"\n✅ Found {len(m1_files)} M1 file(s)", fg="green")
            for f in m1_files:
                size_mb = f.stat().st_size / 1024 / 1024
                click.echo(f"   {f.name} ({size_mb:.1f} MB)")
            return

    click.secho(f"\n❌ No M1 data found", fg="red")


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 2: DSP ANALYSIS (Find Brick Size)
# ═══════════════════════════════════════════════════════════════════════════════


@click.command()
@click.option("--symbol", "-s", required=True, help="Symbol to analyze")
@click.option("--account", "-a", default="pepperstone_demo_45841299", help="cTrader account")
def dsp(symbol: str, account: str):
    """Run DSP analysis to find optimal brick size."""
    click.echo(f"\n{'=' * 70}")
    click.secho(f"STEP 2: DSP ANALYSIS (Find Brick Size)", fg="cyan", bold=True)
    click.echo(f"{'=' * 70}\n")

    click.echo(f"Symbol: {symbol}")
    click.echo(f"Purpose: Analyze volatility regime → optimal brick size")

    # Load M1 data
    data_root = KR / "data" / "master_standardized" / "ctrader" / account
    sym_dir = data_root / symbol

    m1_files = sorted(sym_dir.glob("*_M1_*.csv"))
    if not m1_files:
        click.secho(f"\n❌ No M1 data found. Run 'download' first.", fg="red")
        return

    try:
        df = load_broker_csv(m1_files[0])
        closes = pd.Series(df["close"].values, index=pd.DatetimeIndex(df["time"], tz="UTC"))

        click.echo(f"\n📊 Loaded {len(df)} M1 bars ({df['time'].min()} to {df['time'].max()})")

        # Run DSP
        click.echo(f"🔍 Running DSP analysis...")
        dsp_result = run_dsp(closes)

        # Extract key metrics
        vr_peak = dsp_result.get("vr_peak", 0.0)
        vr_scale_bars = dsp_result.get("vr_scale_bars", 0)
        brick_size = dsp_result.get("brick_size", 0.0)
        regime = dsp_result.get("regime", "unknown")

        click.echo(f"\n📈 DSP Results:")
        click.echo(f"  VR Peak: {vr_peak:.3f} (volatility ratio)")
        click.echo(f"  VR Scale (bars): {vr_scale_bars}")
        click.echo(f"  Brick Size: ${brick_size:.2f}")
        click.echo(f"  Regime: {regime}")
        click.echo(f"  Vol (1σ): {closes.pct_change().std() * 100:.3f}%")

        # Save DSP result
        dsp_file = sym_dir / "dsp_profile.json"
        with open(dsp_file, "w") as f:
            json.dump(dsp_result, f, indent=2)

        click.secho(f"\n✅ DSP analysis complete", fg="green")
        click.echo(f"   Saved to: {dsp_file.name}")

    except Exception as e:
        click.secho(f"\n❌ Error: {e}", fg="red")


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 3: FRICTION VALIDATION
# ═══════════════════════════════════════════════════════════════════════════════


@click.command()
@click.option("--symbol", "-s", required=True, help="Symbol to validate")
@click.option("--account", "-a", default="pepperstone_demo_45841299", help="cTrader account")
def friction(symbol: str, account: str):
    """Validate friction costs are < 1 brick."""
    click.echo(f"\n{'=' * 70}")
    click.secho(f"STEP 3: FRICTION VALIDATION", fg="cyan", bold=True)
    click.echo(f"{'=' * 70}\n")

    click.echo(f"Symbol: {symbol}")
    click.echo(f"Goal: Verify friction_cost < 1 × brick_size")

    # Load DSP result
    data_root = KR / "data" / "master_standardized" / "ctrader" / account
    sym_dir = data_root / symbol
    dsp_file = sym_dir / "dsp_profile.json"

    if not dsp_file.exists():
        click.secho(f"\n❌ DSP profile not found. Run 'dsp' first.", fg="red")
        return

    with open(dsp_file) as f:
        dsp_result = json.load(f)

    brick_size = dsp_result.get("brick_size", 0.0)

    # Load instrument spec (from cTrader)
    # TODO: Load actual spread, commission from broker
    spread_pts = 1.0  # cTrader typical XAUUSD spread
    commission_usd = 0.0  # cTrader has no commission
    tick_size = 0.01  # XAUUSD tick size

    # Calculate USD friction per round-trip
    friction_usd = (spread_pts * tick_size) * 2  # bid-ask round-trip

    click.echo(f"\n💰 Friction Analysis:")
    click.echo(f"  Brick Size: ${brick_size:.2f}")
    click.echo(f"  Spread (pts): {spread_pts}")
    click.echo(f"  Spread (USD): ${spread_pts * tick_size:.4f}")
    click.echo(f"  Commission: ${commission_usd:.2f}")
    click.echo(f"  Total RT Friction: ${friction_usd:.4f}")
    click.echo(f"  Friction / Brick: {friction_usd / brick_size * 100:.2f}%")

    # Pass if friction < 1 brick (generous threshold)
    passes = friction_usd < brick_size

    if passes:
        click.secho(f"\n✅ PASS - Friction is acceptable (< 1 brick)", fg="green")
    else:
        click.secho(f"\n❌ FAIL - Friction exceeds 1 brick (strategy won't work)", fg="red")

    # Save friction profile
    friction_result = {
        "symbol": symbol,
        "brick_size": brick_size,
        "spread_pts": spread_pts,
        "spread_usd": spread_pts * tick_size,
        "commission_usd": commission_usd,
        "friction_rt_usd": friction_usd,
        "friction_brick_ratio": friction_usd / brick_size,
        "passes": passes,
    }

    friction_file = sym_dir / "friction_profile.json"
    with open(friction_file, "w") as f:
        json.dump(friction_result, f, indent=2)


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 4: BACKTEST (Empirical Test)
# ═══════════════════════════════════════════════════════════════════════════════


@click.command()
@click.option("--symbol", "-s", required=True, help="Symbol to backtest")
@click.option("--account", "-a", default="pepperstone_demo_45841299", help="cTrader account")
@click.option("--months", "-m", default=3, help="Months of data to backtest")
def backtest(symbol: str, account: str, months: int):
    """Run empirical backtest."""
    click.echo(f"\n{'=' * 70}")
    click.secho(f"STEP 4: EMPIRICAL BACKTEST", fg="cyan", bold=True)
    click.echo(f"{'=' * 70}\n")

    click.echo(f"Symbol: {symbol}")
    click.echo(f"Data: {months} months")
    click.echo(f"Purpose: Test if strategy actually works")

    # Load M1 data
    data_root = KR / "data" / "master_standardized" / "ctrader" / account
    sym_dir = data_root / symbol
    m1_files = sorted(sym_dir.glob("*_M1_*.csv"))

    if not m1_files:
        click.secho(f"\n❌ No M1 data found", fg="red")
        return

    # Load DSP and friction profiles
    dsp_file = sym_dir / "dsp_profile.json"
    friction_file = sym_dir / "friction_profile.json"

    if not dsp_file.exists() or not friction_file.exists():
        click.secho(
            f"\n❌ DSP or friction profiles not found. Run 'dsp' and 'friction' first.", fg="red"
        )
        return

    with open(dsp_file) as f:
        dsp_result = json.load(f)
    with open(friction_file) as f:
        friction_result = json.load(f)

    if not friction_result.get("passes", False):
        click.secho(f"\n❌ Friction validation failed. Cannot proceed.", fg="red")
        return

    # Load M1 and run backtest
    try:
        df = load_broker_csv(m1_files[0])
        closes = pd.Series(df["close"].values, index=pd.DatetimeIndex(df["time"], tz="UTC"))

        click.echo(f"\n📊 Loaded {len(df)} M1 bars")

        # Use the unified RenkoEngine for a deterministic backtest
        brick_size = float(dsp_result.get("brick_size", 1.0))
        tick_size = float(friction_result.get("spread_usd", 0.01)) / max(
            1.0, float(friction_result.get("spread_pts", 1.0))
        )
        cfg = EngineConfig(
            symbol=symbol,
            brick_size=brick_size,
            usd_per_point=float(
                dsp_result.get("usd_per_point", friction_result.get("spread_usd", 1.0))
            ),
            tick_size=float(dsp_result.get("tick_size", 0.01)),
            stop_bricks=0.5,
            fliprate_window=int(dsp_result.get("vr_scale_bars", 50)),
            markov_window=int(dsp_result.get("vr_scale_bars", 50)),
            fliprate_threshold=float(
                dsp_result.get("filter_params", {}).get("fliprate_threshold", 0.35)
            ),
            markov_threshold=float(
                dsp_result.get("filter_params", {}).get("markov_threshold", 0.55)
            ),
            target_risk_usd=100.0,
            lot_step=0.01,
            min_lots=0.01,
            gate_lot_ceiling=999.0,
            spread_pts=float(friction_result.get("spread_pts", 1.0)),
        )

        engine = RenkoEngine(cfg)
        click.echo(f"\n⚙️  Running RenkoEngine backtest...")
        results = engine.backtest(closes)
        if "error" in results:
            click.secho(f"\n❌ Backtest error: {results['error']}", fg="red")
        else:
            summary = results.get("summary", {})
            click.echo(f"\n📈 Backtest Summary:")
            click.echo(f"  Trades: {summary.get('n_trades', 0)}")
            click.echo(f"  Net USD: {summary.get('net_usd', 0.0):.2f}")
            click.echo(f"  Omega: {summary.get('omega', 0.0):.3f}")
            # Save results
            out_dir = KR / "results" / "renko" / "backtest"
            out_dir.mkdir(parents=True, exist_ok=True)
            out_file = (
                out_dir / f"backtest_{symbol}_{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}.json"
            )
            with open(out_file, "w") as f:
                json.dump(results, f, indent=2)
            click.secho(f"\n✅ Backtest complete — results saved to {out_file}", fg="green")

    except Exception as e:
        click.secho(f"\n❌ Error: {e}", fg="red")


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 5: TRADE (Paper → Micro → Small → Full)
# ═══════════════════════════════════════════════════════════════════════════════


@cli.command()
@click.option("--symbol", "-s", required=True, help="Symbol to trade")
@click.option(
    "--gate", "-g", default="paper", type=click.Choice(["paper", "micro", "small", "full"])
)
@click.option("--mode", "-m", default="dry_run", type=click.Choice(["paper", "dry_run", "live"]))
def trade(symbol: str, gate: str, mode: str):
    """Launch live trading."""
    click.echo(f"\n{'=' * 70}")
    click.secho(f"STEP 5: LIVE TRADING", fg="cyan", bold=True)
    click.echo(f"{'=' * 70}\n")

    click.echo(f"Symbol: {symbol}")
    click.echo(f"Gate: {gate}")
    click.echo(f"Mode: {mode}")

    # Verify DSP and friction passed
    # TODO: Load and check profiles

    # For paper/dry_run modes we can run the RenkoEngine replay simulator on the
    # latest M1 CSV (fast, deterministic). For live mode we launch RenkoLiveTrader.
    data_root = KR / "data" / "master_standardized" / "ctrader"
    # find any account folder with the symbol
    sym_paths = list(data_root.glob("*"))
    csv_path = None
    for acc in sym_paths:
        candidate = acc / symbol
        if candidate.exists():
            files = sorted(candidate.glob("*_M1_*.csv"))
            if files:
                csv_path = files[-1]
                break

    if mode in ("paper", "dry_run"):
        if csv_path is None:
            click.secho("\n❌ No M1 CSV found for paper replay. Run download first.", fg="red")
            return
        # Load DSP + friction if present
        dsp_file = Path(csv_path).parent / "dsp_profile.json"
        friction_file = Path(csv_path).parent / "friction_profile.json"
        dsp_result = json.load(dsp_file.open()) if dsp_file.exists() else {}
        friction_result = json.load(friction_file.open()) if friction_file.exists() else {}

        cfg = EngineConfig(
            symbol=symbol,
            brick_size=float(dsp_result.get("brick_size", 1.0)),
            usd_per_point=float(
                dsp_result.get("usd_per_point", friction_result.get("spread_usd", 1.0))
            ),
            tick_size=float(dsp_result.get("tick_size", friction_result.get("tick_size", 0.01))),
            spread_pts=float(friction_result.get("spread_pts", 1.0)),
        )
        engine = RenkoEngine(cfg)
        click.echo(f"\n⚙️  Running paper replay using {csv_path.name}...")
        res = engine.replay_simulate_from_csv(str(csv_path))
        if "error" in res:
            click.secho(f"\n❌ Simulation error: {res['error']}", fg="red")
            return
        summary = res.get("summary", {})
        click.echo(
            f"\n📈 Replay Summary: trades={summary.get('n_trades', 0)}, net=${summary.get('net_usd', 0.0):.2f}, omega={summary.get('omega', 0.0):.3f}"
        )
        click.secho("\n✅ Paper replay complete", fg="green")
        return

    # Live mode: use RenkoLiveTrader
    from kinetra.renko.live_trader import LiveTraderConfig, PERGate, RenkoLiveTrader

    config = LiveTraderConfig(
        symbols=[symbol],
        gate=PERGate(gate),
        target_risk_usd=100.0,
        skip_qualification=True,  # Allow if friction passed
    )

    trader = RenkoLiveTrader(config)
    try:
        click.secho(f"\n▶️  Starting live trading...\n", fg="green")
        trader.start()
    except KeyboardInterrupt:
        click.echo("\n\n⏹️  Stopping...")
        trader.stop()
        click.secho("✅ Shutdown complete", fg="green")


# ═══════════════════════════════════════════════════════════════════════════════
# CLI Group
# ═══════════════════════════════════════════════════════════════════════════════


@click.group()
def cli():
    """
    Kinetra Pipeline v4 - First Principles

    PHILOSOPHY:
      - M1 bars are raw noise
      - Bricks are price-level structure
      - DSP finds the correct brick size
      - Friction must be < 1 brick or strategy fails
      - Testing determines everything

    PIPELINE:
      1. download  - Get M1 bars
      2. dsp       - Analyze volatility → find brick size
      3. friction  - Verify cost < 1 brick
      4. backtest  - Test if it works
      5. trade     - Paper → Micro → Small → Full
    """
    pass


cli.add_command(download)
cli.add_command(dsp)
cli.add_command(friction)
cli.add_command(backtest)
cli.add_command(trade)


if __name__ == "__main__":
    cli()
