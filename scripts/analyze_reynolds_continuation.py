#!/usr/bin/env python3
"""
Reynolds Number: Continuation vs Reversal Prediction

Hypothesis:
- Low Reynolds (laminar flow) = smooth trend = CONTINUATION
- High Reynolds (turbulent flow) = chaotic = REVERSAL

Test empirically on berserker signals.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from kinetra.mt5_connector import load_csv_data
from kinetra.physics_engine import PhysicsEngine


def compute_reynolds(df: pd.DataFrame, lookback: int = 20) -> pd.Series:
    """
    Compute Reynolds number for market flow.

    Re = (velocity * range * volume) / volatility

    Low Re = laminar = trending (continuation)
    High Re = turbulent = chaotic (reversal)
    """
    velocity = df['close'].pct_change()
    bar_range_pct = (df['high'] - df['low']) / df['close']
    volatility = velocity.rolling(lookback).std().clip(lower=1e-10)
    volume_norm = df['volume'] / df['volume'].rolling(lookback).mean().clip(lower=1e-10)

    reynolds = (velocity.abs() * bar_range_pct * volume_norm) / volatility
    return reynolds.rolling(lookback).mean().fillna(1.0)


def test_reynolds_continuation(df: pd.DataFrame, berserker_mask: pd.Series):
    """
    Test if Reynolds predicts continuation vs reversal.
    """
    df = df.copy()

    # Compute Reynolds
    df['reynolds'] = compute_reynolds(df)

    # Reynolds percentile (adaptive)
    window = min(500, len(df))
    df['reynolds_pct'] = df['reynolds'].rolling(window).apply(
        lambda x: (x.iloc[-1] > x.iloc[:-1]).mean() if len(x) > 1 else 0.5, raw=False
    ).fillna(0.5)

    # Momentum direction
    df['momentum_5'] = df['close'].pct_change(5)
    df['momentum_dir'] = np.sign(df['momentum_5'])

    # Forward return and direction
    df['fwd_return'] = df['close'].pct_change().shift(-1)
    df['fwd_direction'] = np.sign(df['fwd_return'])

    # Continuation = forward direction matches momentum direction
    df['is_continuation'] = df['fwd_direction'] == df['momentum_dir']

    # Filter to berserker
    df_berk = df[berserker_mask].copy()

    print("\n" + "=" * 80)
    print("REYNOLDS NUMBER: CONTINUATION vs REVERSAL")
    print("=" * 80)

    print(f"\n  Total berserker signals: {len(df_berk)}")

    # Overall continuation rate
    overall_cont = df_berk['is_continuation'].mean() * 100
    print(f"  Overall continuation rate: {overall_cont:.1f}%")

    # Test by Reynolds regime
    print("\n  BY REYNOLDS REGIME:")
    print(f"\n  {'Regime':>20} │ {'Count':>7} │ {'Cont%':>8} │ {'Rev%':>8} │ {'Signal':>12}")
    print("  " + "─" * 70)

    regimes = [
        ('LAMINAR (<25%)', df_berk['reynolds_pct'] < 0.25),
        ('TRANSITIONAL', (df_berk['reynolds_pct'] >= 0.25) & (df_berk['reynolds_pct'] <= 0.75)),
        ('TURBULENT (>75%)', df_berk['reynolds_pct'] > 0.75),
    ]

    for name, mask in regimes:
        subset = df_berk[mask]
        if len(subset) < 30:
            print(f"  {name:>20} │ {'<30':>7} │ {'-':>8} │ {'-':>8} │ {'-':>12}")
            continue

        cont_rate = subset['is_continuation'].mean() * 100
        rev_rate = 100 - cont_rate

        if cont_rate > 55:
            signal = "CONTINUATION"
        elif rev_rate > 55:
            signal = "REVERSAL"
        else:
            signal = "NEUTRAL"

        print(f"  {name:>20} │ {len(subset):>7} │ {cont_rate:>7.1f}% │ {rev_rate:>7.1f}% │ {signal:>12}")

    return df_berk


def test_combined_signals(df_berk: pd.DataFrame):
    """
    Combine Reynolds with other physics for better prediction.
    """
    print("\n" + "=" * 80)
    print("REYNOLDS + OTHER PHYSICS COMBINATIONS")
    print("=" * 80)

    # Add buying pressure if not present
    if 'buying_pressure' not in df_berk.columns:
        bar_range = df_berk['high'] - df_berk['low']
        bp = (df_berk['close'] - df_berk['low']) / bar_range.clip(lower=1e-10)
        df_berk['buying_pressure'] = bp.rolling(5).mean()

    combos = [
        # Laminar + momentum alignment = strong continuation
        ('LAMINAR + bullish mom → CONTINUE UP',
         (df_berk['reynolds_pct'] < 0.3) & (df_berk['momentum_dir'] > 0),
         'continuation'),

        ('LAMINAR + bearish mom → CONTINUE DOWN',
         (df_berk['reynolds_pct'] < 0.3) & (df_berk['momentum_dir'] < 0),
         'continuation'),

        # Turbulent + exhausted buying = reversal
        ('TURBULENT + high BP → REVERSAL DOWN',
         (df_berk['reynolds_pct'] > 0.7) & (df_berk['buying_pressure'] > 0.6),
         'reversal'),

        ('TURBULENT + low BP → REVERSAL UP',
         (df_berk['reynolds_pct'] > 0.7) & (df_berk['buying_pressure'] < 0.4),
         'reversal'),

        # Laminar + low Reynolds = clean trend
        ('VERY LAMINAR (<10%) + momentum → STRONG CONT',
         (df_berk['reynolds_pct'] < 0.1) & (df_berk['momentum_dir'] != 0),
         'continuation'),

        # Turbulent + extreme = strong reversal
        ('VERY TURBULENT (>90%) → STRONG REV',
         df_berk['reynolds_pct'] > 0.9,
         'reversal'),
    ]

    print(f"\n  {'Signal':>45} │ {'Count':>6} │ {'Correct%':>10} │ {'Edge':>8}")
    print("  " + "─" * 80)

    for name, mask, expected in combos:
        subset = df_berk[mask]
        if len(subset) < 20:
            print(f"  {name:>45} │ {'<20':>6} │ {'-':>10} │ {'-':>8}")
            continue

        if expected == 'continuation':
            correct = subset['is_continuation'].mean() * 100
        else:
            correct = (1 - subset['is_continuation']).mean() * 100

        edge = correct - 50
        print(f"  {name:>45} │ {len(subset):>6} │ {correct:>9.1f}% │ {edge:>+7.1f}%")


def analyze_reynolds_magnitude(df: pd.DataFrame, berserker_mask: pd.Series):
    """
    Does Reynolds also predict move magnitude?
    """
    df = df.copy()
    df['reynolds'] = compute_reynolds(df)

    window = min(500, len(df))
    df['reynolds_pct'] = df['reynolds'].rolling(window).apply(
        lambda x: (x.iloc[-1] > x.iloc[:-1]).mean() if len(x) > 1 else 0.5, raw=False
    ).fillna(0.5)

    df['fwd_abs_return'] = df['close'].pct_change().shift(-1).abs() * 100

    df_berk = df[berserker_mask].copy()

    print("\n" + "=" * 80)
    print("REYNOLDS vs MOVE MAGNITUDE")
    print("=" * 80)

    baseline = df_berk['fwd_abs_return'].mean()
    print(f"\n  Baseline avg move: {baseline:.4f}%")

    regimes = [
        ('LAMINAR (<25%)', df_berk['reynolds_pct'] < 0.25),
        ('TRANSITIONAL', (df_berk['reynolds_pct'] >= 0.25) & (df_berk['reynolds_pct'] <= 0.75)),
        ('TURBULENT (>75%)', df_berk['reynolds_pct'] > 0.75),
    ]

    print(f"\n  {'Regime':>20} │ {'Count':>7} │ {'Avg Move':>10} │ {'Lift':>8}")
    print("  " + "─" * 55)

    for name, mask in regimes:
        subset = df_berk[mask]
        if len(subset) < 30:
            continue

        avg_move = subset['fwd_abs_return'].mean()
        lift = avg_move / baseline

        print(f"  {name:>20} │ {len(subset):>7} │ {avg_move:>9.4f}% │ {lift:>7.2f}x")


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

    # Berserker signal
    berserker = (df['energy_pct'] > 0.75) & (df['damping_pct'] < 0.25)
    berserker_plus = berserker & (df['flow_consistency'] > 0.7) & (df['volume_pct'] > 0.6)

    print(f"\nBerserker signals: {berserker.sum()}")
    print(f"Berserker+ signals: {berserker_plus.sum()}")

    # Test Reynolds
    df_berk = test_reynolds_continuation(df, berserker)
    test_combined_signals(df_berk)
    analyze_reynolds_magnitude(df, berserker)

    # Summary
    print("\n" + "=" * 80)
    print("REYNOLDS NUMBER SUMMARY")
    print("=" * 80)
    print("""
  REYNOLDS NUMBER IN MARKETS:

  Re = (velocity × range × volume) / volatility

  LAMINAR (Low Re < 25th pct):
  - Smooth, consistent flow
  - Trend likely to CONTINUE
  - Trade WITH momentum

  TRANSITIONAL (25-75th pct):
  - Mixed regime
  - Could go either way
  - Use other signals (buying pressure)

  TURBULENT (High Re > 75th pct):
  - Chaotic, unpredictable
  - REVERSAL more likely
  - Trade AGAINST momentum (counter-trend)

  COMBINATION:
  - Laminar + berserker = continuation fat candle
  - Turbulent + berserker = reversal fat candle
  - Turbulent + high buying pressure = reversal DOWN
  - Turbulent + low buying pressure = reversal UP
""")


if __name__ == "__main__":
    main()
