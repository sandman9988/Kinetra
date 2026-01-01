#!/usr/bin/env python3
"""
Fat Candle Analysis

Berserker's prey: FAT CANDLES (big energy releases)
Not MR trader - catching big directional moves.

Questions:
1. What predicts FAT CANDLE occurring?
2. What predicts FAT CANDLE direction?
3. What is the optimal context for each direction?
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from kinetra.mt5_connector import load_csv_data
from kinetra.physics_engine import PhysicsEngine


def analyze_fat_candles(df: pd.DataFrame, berserker_mask: pd.Series):
    """
    Analyze fat candle occurrence and direction.
    """
    df_all = df.copy()

    # Define "fat candle" - significantly larger than average
    df_all['candle_size'] = abs(df_all['close'] - df_all['open']) / df_all['close'] * 100

    # Fat candle threshold: top 25% of moves
    fat_threshold = df_all['candle_size'].quantile(0.75)
    very_fat_threshold = df_all['candle_size'].quantile(0.90)

    df_all['is_fat'] = df_all['candle_size'] > fat_threshold
    df_all['is_very_fat'] = df_all['candle_size'] > very_fat_threshold
    df_all['fat_direction'] = np.sign(df_all['close'] - df_all['open'])

    print("\n" + "=" * 80)
    print("FAT CANDLE ANALYSIS")
    print("=" * 80)

    print(f"\n  Fat candle threshold (75th pct): {fat_threshold:.3f}%")
    print(f"  Very fat candle threshold (90th pct): {very_fat_threshold:.3f}%")

    # Look at NEXT bar after berserker
    df_berk = df_all[berserker_mask].copy()

    # Shift fat candle indicators to see next bar
    df_berk['next_is_fat'] = df_all['is_fat'].shift(-1).loc[df_berk.index]
    df_berk['next_is_very_fat'] = df_all['is_very_fat'].shift(-1).loc[df_berk.index]
    df_berk['next_fat_direction'] = df_all['fat_direction'].shift(-1).loc[df_berk.index]
    df_berk['next_candle_size'] = df_all['candle_size'].shift(-1).loc[df_berk.index]

    # How often does fat candle follow berserker?
    fat_rate = df_berk['next_is_fat'].mean() * 100
    very_fat_rate = df_berk['next_is_very_fat'].mean() * 100

    baseline_fat_rate = df_all['is_fat'].mean() * 100
    baseline_very_fat_rate = df_all['is_very_fat'].mean() * 100

    print(f"\n  FAT CANDLE OCCURRENCE:")
    print(f"    Baseline fat candle rate: {baseline_fat_rate:.1f}%")
    print(f"    After berserker: {fat_rate:.1f}% (lift: {fat_rate/baseline_fat_rate:.2f}x)")
    print(f"\n    Baseline very fat rate: {baseline_very_fat_rate:.1f}%")
    print(f"    After berserker: {very_fat_rate:.1f}% (lift: {very_fat_rate/baseline_very_fat_rate:.2f}x)")

    return df_berk, df_all


def analyze_fat_candle_direction(df_berk: pd.DataFrame, df_all: pd.DataFrame):
    """
    What predicts the direction of fat candles after berserker?
    """
    print("\n" + "=" * 80)
    print("FAT CANDLE DIRECTION PREDICTION")
    print("=" * 80)

    # Only look at berserker bars followed by fat candles
    fat_followers = df_berk[df_berk['next_is_fat'] == True].copy()

    print(f"\n  Berserker bars followed by fat candle: {len(fat_followers)}")

    if len(fat_followers) < 50:
        print("  Not enough samples for analysis")
        return

    # Direction breakdown
    up_fat = (fat_followers['next_fat_direction'] > 0).mean() * 100
    down_fat = (fat_followers['next_fat_direction'] < 0).mean() * 100

    print(f"  Up fat candles: {up_fat:.1f}%")
    print(f"  Down fat candles: {down_fat:.1f}%")

    # Add context features
    lookback = 20
    rolling_high = df_all['high'].rolling(lookback).max()
    rolling_low = df_all['low'].rolling(lookback).min()
    range_size = rolling_high - rolling_low

    fat_followers['range_position'] = (df_all.loc[fat_followers.index, 'close'] - rolling_low.loc[fat_followers.index]) / range_size.loc[fat_followers.index].clip(lower=1e-10)
    fat_followers['momentum'] = df_all['close'].pct_change(lookback).loc[fat_followers.index]

    bar_range = df_all['high'] - df_all['low']
    buying_pressure = (df_all['close'] - df_all['low']) / bar_range.clip(lower=1e-10)
    fat_followers['buying_pressure'] = buying_pressure.rolling(5).mean().loc[fat_followers.index]

    # Test context signals for fat candle direction
    print("\n  CONTEXT → FAT CANDLE DIRECTION:")
    print(f"\n  {'Context':>40} │ {'Count':>6} │ {'Up%':>7} │ {'Down%':>7} │ {'Bias':>8}")
    print("  " + "─" * 75)

    contexts = [
        ('Near resistance (>0.8)', fat_followers['range_position'] > 0.8),
        ('Near support (<0.2)', fat_followers['range_position'] < 0.2),
        ('Mid-range (0.3-0.7)', (fat_followers['range_position'] > 0.3) & (fat_followers['range_position'] < 0.7)),
        ('Bullish momentum', fat_followers['momentum'] > 0),
        ('Bearish momentum', fat_followers['momentum'] < 0),
        ('High buying pressure (>0.6)', fat_followers['buying_pressure'] > 0.6),
        ('Low buying pressure (<0.4)', fat_followers['buying_pressure'] < 0.4),
    ]

    for name, mask in contexts:
        subset = fat_followers[mask]
        if len(subset) < 20:
            print(f"  {name:>40} │ {'<20':>6} │ {'-':>7} │ {'-':>7} │ {'-':>8}")
            continue

        up_pct = (subset['next_fat_direction'] > 0).mean() * 100
        down_pct = (subset['next_fat_direction'] < 0).mean() * 100

        if up_pct > down_pct + 5:
            bias = f"+{up_pct - 50:.0f}% UP"
        elif down_pct > up_pct + 5:
            bias = f"+{down_pct - 50:.0f}% DN"
        else:
            bias = "neutral"

        print(f"  {name:>40} │ {len(subset):>6} │ {up_pct:>6.1f}% │ {down_pct:>6.1f}% │ {bias:>8}")


def analyze_optimal_fat_candle_setup(df_berk: pd.DataFrame, df_all: pd.DataFrame):
    """
    Find the optimal setup for catching fat candle direction.
    """
    print("\n" + "=" * 80)
    print("OPTIMAL FAT CANDLE SETUPS")
    print("=" * 80)

    fat_followers = df_berk[df_berk['next_is_fat'] == True].copy()

    if len(fat_followers) < 50:
        return

    # Add all context features
    lookback = 20
    rolling_high = df_all['high'].rolling(lookback).max()
    rolling_low = df_all['low'].rolling(lookback).min()
    range_size = rolling_high - rolling_low

    fat_followers['range_position'] = (df_all.loc[fat_followers.index, 'close'] - rolling_low.loc[fat_followers.index]) / range_size.loc[fat_followers.index].clip(lower=1e-10)
    fat_followers['momentum'] = df_all['close'].pct_change(lookback).loc[fat_followers.index]

    bar_range = df_all['high'] - df_all['low']
    buying_pressure = (df_all['close'] - df_all['low']) / bar_range.clip(lower=1e-10)
    fat_followers['buying_pressure'] = buying_pressure.rolling(5).mean().loc[fat_followers.index]

    # Volume trend
    fat_followers['volume_trend'] = df_all['volume'].pct_change(5).loc[fat_followers.index]

    # Test combined setups
    setups = [
        # UP fat candle setups
        ('LONG setup: support + bearish mom + low buy press',
         (fat_followers['range_position'] < 0.3) &
         (fat_followers['momentum'] < 0) &
         (fat_followers['buying_pressure'] < 0.45),
         'up'),

        ('LONG setup: support + volume surge',
         (fat_followers['range_position'] < 0.3) &
         (fat_followers['volume_trend'] > 0.2),
         'up'),

        # DOWN fat candle setups
        ('SHORT setup: resistance + bullish mom + high buy press',
         (fat_followers['range_position'] > 0.7) &
         (fat_followers['momentum'] > 0) &
         (fat_followers['buying_pressure'] > 0.55),
         'down'),

        ('SHORT setup: resistance + volume surge',
         (fat_followers['range_position'] > 0.7) &
         (fat_followers['volume_trend'] > 0.2),
         'down'),

        # Counter-momentum setups (exhaustion)
        ('EXHAUSTION UP → fat candle DOWN',
         (fat_followers['range_position'] > 0.75) &
         (fat_followers['momentum'] > 0),
         'down'),

        ('EXHAUSTION DOWN → fat candle UP',
         (fat_followers['range_position'] < 0.25) &
         (fat_followers['momentum'] < 0),
         'up'),
    ]

    print(f"\n  {'Setup':>50} │ {'Count':>6} │ {'Correct%':>10} │ {'Edge':>8}")
    print("  " + "─" * 85)

    for name, mask, expected in setups:
        subset = fat_followers[mask]
        if len(subset) < 15:
            print(f"  {name:>50} │ {'<15':>6} │ {'-':>10} │ {'-':>8}")
            continue

        if expected == 'up':
            correct = (subset['next_fat_direction'] > 0).mean() * 100
        else:
            correct = (subset['next_fat_direction'] < 0).mean() * 100

        edge = correct - 50
        print(f"  {name:>50} │ {len(subset):>6} │ {correct:>9.1f}% │ {edge:>+7.1f}%")


def analyze_candle_size_vs_context(df_berk: pd.DataFrame, df_all: pd.DataFrame):
    """
    Does context affect size of the following candle?
    """
    print("\n" + "=" * 80)
    print("CANDLE SIZE BY CONTEXT")
    print("=" * 80)

    # Add context features
    lookback = 20
    rolling_high = df_all['high'].rolling(lookback).max()
    rolling_low = df_all['low'].rolling(lookback).min()
    range_size = rolling_high - rolling_low

    df_berk = df_berk.copy()
    df_berk['range_position'] = (df_all.loc[df_berk.index, 'close'] - rolling_low.loc[df_berk.index]) / range_size.loc[df_berk.index].clip(lower=1e-10)
    df_berk['momentum'] = df_all['close'].pct_change(lookback).loc[df_berk.index]

    baseline = df_berk['next_candle_size'].mean()

    print(f"\n  Baseline next candle size after berserker: {baseline:.4f}%")

    contexts = [
        ('Near resistance (>0.8)', df_berk['range_position'] > 0.8),
        ('Near support (<0.2)', df_berk['range_position'] < 0.2),
        ('Mid-range', (df_berk['range_position'] > 0.3) & (df_berk['range_position'] < 0.7)),
        ('Extreme bullish mom', df_berk['momentum'] > df_berk['momentum'].quantile(0.9)),
        ('Extreme bearish mom', df_berk['momentum'] < df_berk['momentum'].quantile(0.1)),
    ]

    print(f"\n  {'Context':>30} │ {'Count':>6} │ {'Avg Size':>10} │ {'Lift':>8}")
    print("  " + "─" * 65)

    for name, mask in contexts:
        subset = df_berk[mask]
        if len(subset) < 30:
            continue

        avg_size = subset['next_candle_size'].mean()
        lift = avg_size / baseline

        print(f"  {name:>30} │ {len(subset):>6} │ {avg_size:>9.4f}% │ {lift:>7.2f}x")


def main():
    # Load data
    project_root = Path(__file__).parent.parent
    csv_files = list(project_root.glob("*BTCUSD*.csv"))
    if not csv_files:
        print("No BTCUSD CSV file found")
        return

    data = load_csv_data(str(csv_files[0]))
    print(f"Loaded {len(data)} bars")

    # Compute physics
    engine = PhysicsEngine(lookback=20)
    physics = engine.compute_physics_state(data['close'], data['volume'], include_percentiles=True)

    df = data.copy()
    df['energy_pct'] = physics['energy_pct']
    df['damping_pct'] = physics['damping_pct']

    # Flow consistency
    df['return_sign'] = np.sign(df['close'].pct_change())
    df['flow_consistency'] = df['return_sign'].rolling(5).apply(
        lambda x: (x == x.iloc[-1]).mean() if len(x) > 0 else 0.5, raw=False
    ).fillna(0.5)

    # Volume percentile
    vol_window = min(500, len(df))
    df['volume_pct'] = df['volume'].rolling(vol_window).apply(
        lambda x: (x.iloc[-1] > x.iloc[:-1]).mean() if len(x) > 1 else 0.5, raw=False
    ).fillna(0.5)

    df = df.dropna()

    # Signals
    berserker = (df['energy_pct'] > 0.75) & (df['damping_pct'] < 0.25)
    berserker_plus = berserker & (df['flow_consistency'] > 0.7) & (df['volume_pct'] > 0.6)

    print(f"\nBerserker signals: {berserker.sum()}")
    print(f"Berserker+ signals: {berserker_plus.sum()}")

    # Run analysis
    df_berk, df_all = analyze_fat_candles(df, berserker)
    analyze_fat_candle_direction(df_berk, df_all)
    analyze_optimal_fat_candle_setup(df_berk, df_all)
    analyze_candle_size_vs_context(df_berk, df_all)

    # Summary
    print("\n" + "=" * 80)
    print("FAT CANDLE HUNTING STRATEGY")
    print("=" * 80)
    print("""
  BERSERKER = FAT CANDLE DETECTOR (not MR trader)

  1. MAGNITUDE: Berserker predicts BIG MOVES
     - Fat candle rate after berserker: ~2x baseline

  2. DIRECTION: Context determines fat candle direction
     - EXHAUSTION at extreme → reversal fat candle
     - BREAKOUT with volume → continuation fat candle

  3. BEST SETUPS:
     - LONG fat: Support + bearish mom + low buying pressure
     - SHORT fat: Resistance + bullish mom + high buying pressure

  The prey is the FAT CANDLE, not the direction itself.
  Direction is just how we capture it.
""")


if __name__ == "__main__":
    main()
