#!/usr/bin/env python3
"""
Debug script: Analyze Renko flip blockers in live trading.

Reads the most recent live trades file and checks:
1. Are bricks actually being generated?
2. What are the direction sequences?
3. Why are flips blocked?
4. What's the brick size vs market volatility?
"""

import json
import sys
from pathlib import Path

import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from kinetra.config import PROJECT_ROOT
from kinetra.renko.brick_engine import build_renko
from kinetra.renko.filters import flip_rate, markov_stickiness


def analyze_live_session(symbol: str = "XAUUSD") -> None:
    """Analyze the most recent live trading session."""
    # Find the most recent live trades file
    live_dir = PROJECT_ROOT / "results" / "renko" / "live"
    live_files = sorted(live_dir.glob("*.jsonl"))

    if not live_files:
        print("❌ No live trade files found")
        return

    most_recent = live_files[-1]
    print(f"📖 Analyzing {most_recent.name}")

    # Read trades
    trades = []
    with open(most_recent) as f:
        for line in f:
            if line.strip():
                trades.append(json.loads(line))

    if not trades:
        print("❌ No trades in file")
        return

    print(f"📊 Found {len(trades)} trades")

    # Get M1 data for the symbol
    symbol_dir = PROJECT_ROOT / "data" / "master_standardized" / "commodity" / symbol
    if not symbol_dir.exists():
        print(f"❌ Symbol dir not found: {symbol_dir}")
        return

    m1_files = sorted(symbol_dir.glob("*_M1_*.csv"))
    if not m1_files:
        print(f"❌ No M1 files found in {symbol_dir}")
        return

    latest_m1 = m1_files[-1]
    print(f"📊 Reading M1 data from {latest_m1.name}")

    # Load M1 data
    df = pd.read_csv(latest_m1, parse_dates=["time"])
    print(f"📈 Loaded {len(df)} M1 bars from {df['time'].min()} to {df['time'].max()}")

    # Get brick size from config or qualification
    qual_dir = PROJECT_ROOT / "data" / "renko_qualified" / symbol
    qual_json = qual_dir / "qualification.json"

    brick_size = 1.0
    if qual_json.exists():
        with open(qual_json) as f:
            qual = json.load(f)
            brick_size = float(qual.get("brick_size", 1.0))
        print(f"🧱 Brick size: {brick_size} (from qualification)")
    else:
        print(f"🧱 Brick size: {brick_size} (default)")

    # Build bricks
    closes = df.set_index("time")["close"]
    bricks = build_renko(closes, brick_size=brick_size)

    if bricks.empty:
        print("❌ No bricks generated")
        return

    print(f"🧱 Generated {len(bricks)} bricks")

    # Analyze directions
    directions = bricks["direction"].values
    print("\n📊 Direction sequence (last 20):")
    for i, d in enumerate(directions[-20:]):
        print(f"   [{i - 20:+3d}] direction={d:+2d}")

    # Count flips
    n_flips = 0
    for i in range(1, len(directions)):
        if directions[i] != directions[i - 1]:
            n_flips += 1

    print(
        f"\n✅ Colour flips: {n_flips} / {len(directions) - 1} = {n_flips / (len(directions) - 1) * 100:.1f}%"
    )

    # Compute filter metrics
    fr_vals = flip_rate(directions, window=20)
    pUU, pDD = markov_stickiness(directions, window=20)

    print("\n🔍 Filter metrics (last 5):")
    for i in range(max(0, len(fr_vals) - 5), len(fr_vals)):
        print(f"   [{i:3d}] FR={fr_vals[i]:6.3f} pUU={pUU[i]:6.3f} pDD={pDD[i]:6.3f}")

    # Check the last flip situation
    print("\n🔍 Last brick analysis:")
    last_brick = bricks.iloc[-1]
    if len(bricks) >= 2:
        prev_brick = bricks.iloc[-2]
        print(
            f"   Previous: close={prev_brick['brick_close']:.5f} dir={prev_brick['direction']:+2d}"
        )
        print(
            f"   Last:     close={last_brick['brick_close']:.5f} dir={last_brick['direction']:+2d}"
        )
        print(f"   → Same direction? {last_brick['direction'] == prev_brick['direction']}")

    # Volatility analysis
    recent_closes = closes.iloc[-100:]
    volatility_pct = recent_closes.pct_change().std() * 100
    atr_simple = (df.iloc[-100:]["high"].values - df.iloc[-100:]["low"].values).mean()
    print("\n📊 Volatility:")
    print(f"   Vol (%)  : {volatility_pct:.3f}%")
    print(f"   ATR (avg): {atr_simple:.5f}")
    print(f"   Brick sz : {brick_size:.2f}")
    print(f"   Bricks/100bars: {len(bricks) / (len(df) / 100):.1f}")


if __name__ == "__main__":
    analyze_live_session()
