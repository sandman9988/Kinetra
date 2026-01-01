#!/usr/bin/env python3
"""
Berserker Context Analysis

Key insight: Extreme readings can signal REVERSAL or ACCELERATION.
Context determines which.

Context Categories:
1. EXHAUSTION (expect reversal):
   - At range extreme (support/resistance)
   - Momentum extreme (overextended)
   - Flow exhaustion (buying/selling pressure fading)

2. BREAKOUT (expect acceleration):
   - Breaking range boundary
   - Fresh momentum (not overextended)
   - Flow supporting direction

This determines how to interpret berserker signal.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from kinetra.mt5_connector import load_csv_data
from kinetra.physics_engine import PhysicsEngine


def classify_berserker_context(df: pd.DataFrame, lookback: int = 20):
    """
    Classify berserker bars into EXHAUSTION vs BREAKOUT context.
    """
    result = df.copy()

    # === RANGE CONTEXT ===
    rolling_high = df['high'].rolling(lookback).max()
    rolling_low = df['low'].rolling(lookback).min()
    range_size = rolling_high - rolling_low

    # Position in range
    result['range_position'] = (df['close'] - rolling_low) / range_size.clip(lower=1e-10)

    # Is it breaking out of range?
    result['breaking_high'] = df['high'] > rolling_high.shift(1)
    result['breaking_low'] = df['low'] < rolling_low.shift(1)
    result['is_breakout'] = result['breaking_high'] | result['breaking_low']

    # === MOMENTUM CONTEXT ===
    momentum = df['close'].pct_change(lookback)
    result['momentum'] = momentum

    # Momentum percentile (how extreme?)
    result['momentum_pct'] = momentum.rolling(lookback * 5).apply(
        lambda x: (x.iloc[-1] > x.iloc[:-1]).mean() if len(x) > 1 else 0.5, raw=False
    ).fillna(0.5)

    # Is momentum overextended?
    result['momentum_extreme'] = (result['momentum_pct'] > 0.9) | (result['momentum_pct'] < 0.1)

    # === FLOW CONTEXT ===
    bar_range = df['high'] - df['low']
    buying_pressure = (df['close'] - df['low']) / bar_range.clip(lower=1e-10)
    result['buying_pressure'] = buying_pressure.rolling(5).mean()

    # Volume trend
    result['volume_trend'] = df['volume'].pct_change(5)

    # Flow fading? (direction continues but volume decreasing)
    price_direction = np.sign(df['close'].pct_change(5))
    result['flow_fading'] = (result['volume_trend'] < -0.2)

    # === CONTEXT CLASSIFICATION ===
    # EXHAUSTION: At extreme + momentum extreme + flow fading
    result['exhaustion_up'] = (
        (result['range_position'] > 0.8) &  # Near resistance
        (result['momentum'] > 0) &  # Bullish momentum
        ((result['momentum_pct'] > 0.8) | result['flow_fading'])  # Overextended or fading
    )

    result['exhaustion_down'] = (
        (result['range_position'] < 0.2) &  # Near support
        (result['momentum'] < 0) &  # Bearish momentum
        ((result['momentum_pct'] < 0.2) | result['flow_fading'])  # Overextended or fading
    )

    # BREAKOUT: Breaking range + fresh momentum + strong volume
    result['breakout_up'] = (
        result['breaking_high'] &
        (result['volume_trend'] > 0) &  # Volume increasing
        (result['buying_pressure'] > 0.6)  # Strong buying
    )

    result['breakout_down'] = (
        result['breaking_low'] &
        (result['volume_trend'] > 0) &  # Volume increasing
        (result['buying_pressure'] < 0.4)  # Strong selling
    )

    # Forward return
    result['fwd_return'] = df['close'].pct_change().shift(-1)
    result['fwd_direction'] = np.sign(result['fwd_return'])

    return result.dropna()


def test_context_signals(df: pd.DataFrame, berserker_mask: pd.Series):
    """
    Test how context affects berserker signal interpretation.
    """
    df_berk = df[berserker_mask].copy()

    print("\n" + "=" * 80)
    print("BERSERKER CONTEXT: EXHAUSTION vs BREAKOUT")
    print("=" * 80)

    # Base stats
    up_pct = (df_berk['fwd_direction'] > 0).mean() * 100
    down_pct = (df_berk['fwd_direction'] < 0).mean() * 100
    print(f"\n  All berserker bars ({len(df_berk)}): Up {up_pct:.1f}%, Down {down_pct:.1f}%")

    contexts = [
        # Exhaustion contexts - expect REVERSAL
        ('EXHAUSTION UP (→ expect DOWN)', df_berk['exhaustion_up'], 'down'),
        ('EXHAUSTION DOWN (→ expect UP)', df_berk['exhaustion_down'], 'up'),

        # Breakout contexts - expect ACCELERATION
        ('BREAKOUT UP (→ expect MORE UP)', df_berk['breakout_up'], 'up'),
        ('BREAKOUT DOWN (→ expect MORE DOWN)', df_berk['breakout_down'], 'down'),
    ]

    print(f"\n{'Context':>40} │ {'Count':>7} │ {'Correct%':>10} │ {'Edge':>8}")
    print("─" * 75)

    for name, mask, expected in contexts:
        subset = df_berk[mask]
        if len(subset) < 20:
            print(f"{name:>40} │ {'<20':>7} │ {'-':>10} │ {'-':>8}")
            continue

        if expected == 'up':
            correct = (subset['fwd_direction'] > 0).mean() * 100
        else:
            correct = (subset['fwd_direction'] < 0).mean() * 100

        edge = correct - 50
        print(f"{name:>40} │ {len(subset):>7} │ {correct:>9.1f}% │ {edge:>+7.1f}%")


def test_refined_contexts(df: pd.DataFrame, berserker_mask: pd.Series):
    """
    Test refined context combinations.
    """
    df_berk = df[berserker_mask].copy()

    print("\n" + "=" * 80)
    print("REFINED CONTEXT ANALYSIS")
    print("=" * 80)

    # Add momentum direction for counter-trend
    df_berk['momentum_dir'] = np.sign(df_berk['momentum'])
    df_berk['counter_dir'] = -df_berk['momentum_dir']

    contexts = [
        # EXHAUSTION + REVERSAL
        ('Bullish exhaustion → SHORT',
         (df_berk['range_position'] > 0.75) &
         (df_berk['momentum'] > 0) &
         (df_berk['buying_pressure'] > 0.5),
         'down'),

        ('Bearish exhaustion → LONG',
         (df_berk['range_position'] < 0.25) &
         (df_berk['momentum'] < 0) &
         (df_berk['buying_pressure'] < 0.5),
         'up'),

        # BREAKOUT + CONTINUATION
        ('Bullish breakout → LONG',
         df_berk['breakout_up'],
         'up'),

        ('Bearish breakout → SHORT',
         df_berk['breakout_down'],
         'down'),

        # MID-RANGE (unclear context)
        ('Mid-range bullish mom → counter (SHORT)',
         (df_berk['range_position'] > 0.35) &
         (df_berk['range_position'] < 0.65) &
         (df_berk['momentum'] > 0),
         'down'),

        ('Mid-range bearish mom → counter (LONG)',
         (df_berk['range_position'] > 0.35) &
         (df_berk['range_position'] < 0.65) &
         (df_berk['momentum'] < 0),
         'up'),

        # FLOW FADING (exhaustion signal)
        ('Flow fading + bullish → SHORT',
         (df_berk['flow_fading']) &
         (df_berk['momentum'] > 0),
         'down'),

        ('Flow fading + bearish → LONG',
         (df_berk['flow_fading']) &
         (df_berk['momentum'] < 0),
         'up'),
    ]

    print(f"\n{'Context':>45} │ {'Count':>7} │ {'Correct%':>10} │ {'Edge':>8}")
    print("─" * 80)

    for name, mask, expected in contexts:
        subset = df_berk[mask]
        if len(subset) < 30:
            print(f"{name:>45} │ {'<30':>7} │ {'-':>10} │ {'-':>8}")
            continue

        if expected == 'up':
            correct = (subset['fwd_direction'] > 0).mean() * 100
        else:
            correct = (subset['fwd_direction'] < 0).mean() * 100

        edge = correct - 50
        print(f"{name:>45} │ {len(subset):>7} │ {correct:>9.1f}% │ {edge:>+7.1f}%")


def create_context_based_signal(df: pd.DataFrame, berserker_mask: pd.Series):
    """
    Create a unified direction signal based on context.
    """
    df_berk = df[berserker_mask].copy()

    print("\n" + "=" * 80)
    print("CONTEXT-BASED DIRECTION SIGNAL")
    print("=" * 80)

    # Rule-based direction assignment
    # 1. Exhaustion at extreme → Reversal
    # 2. Breakout → Continuation
    # 3. Mid-range → Counter-trend (default berserker behavior)

    conditions = [
        # Exhaustion → Reversal
        (df_berk['exhaustion_up'], -1),  # Short
        (df_berk['exhaustion_down'], 1),  # Long

        # Breakout → Continuation
        (df_berk['breakout_up'], 1),  # Long
        (df_berk['breakout_down'], -1),  # Short

        # Flow fading → Reversal
        ((df_berk['flow_fading']) & (df_berk['momentum'] > 0), -1),
        ((df_berk['flow_fading']) & (df_berk['momentum'] < 0), 1),
    ]

    # Default: counter-trend
    df_berk['signal_direction'] = -np.sign(df_berk['momentum'])

    # Override with context rules (priority order)
    for condition, direction in conditions:
        df_berk.loc[condition, 'signal_direction'] = direction

    # Skip neutral signals
    df_berk = df_berk[df_berk['signal_direction'] != 0]

    # Evaluate
    correct = (df_berk['signal_direction'] == df_berk['fwd_direction']).mean() * 100

    print(f"\n  Context-aware signal accuracy: {correct:.1f}% (edge: {correct-50:+.1f}%)")
    print(f"  Total signals: {len(df_berk)}")

    # Breakdown by context type
    print("\n  Breakdown by applied rule:")

    # Which rule was applied?
    rule_applied = []
    for idx in df_berk.index:
        if df.loc[idx, 'exhaustion_up']:
            rule_applied.append('exhaustion_up')
        elif df.loc[idx, 'exhaustion_down']:
            rule_applied.append('exhaustion_down')
        elif df.loc[idx, 'breakout_up']:
            rule_applied.append('breakout_up')
        elif df.loc[idx, 'breakout_down']:
            rule_applied.append('breakout_down')
        elif df.loc[idx, 'flow_fading']:
            rule_applied.append('flow_fading')
        else:
            rule_applied.append('counter_trend')

    df_berk['rule'] = rule_applied

    print(f"\n  {'Rule':>20} │ {'Count':>7} │ {'Accuracy':>10}")
    print("  " + "─" * 45)

    for rule in df_berk['rule'].unique():
        subset = df_berk[df_berk['rule'] == rule]
        acc = (subset['signal_direction'] == subset['fwd_direction']).mean() * 100
        print(f"  {rule:>20} │ {len(subset):>7} │ {acc:>9.1f}%")


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

    # Compute context
    print("Computing context metrics...")
    df = classify_berserker_context(data, lookback=20)

    # Add physics state
    df['energy_pct'] = physics['energy_pct'].loc[df.index]
    df['damping_pct'] = physics['damping_pct'].loc[df.index]

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

    # Run tests
    test_context_signals(df, berserker)
    test_refined_contexts(df, berserker)
    create_context_based_signal(df, berserker)

    # Summary
    print("\n" + "=" * 80)
    print("CONTEXT RULES FOR BERSERKER")
    print("=" * 80)
    print("""
  CONTEXT DETERMINES SIGNAL TYPE:

  ┌─ EXHAUSTION CONTEXT (→ REVERSAL) ────────────────────────┐
  │  - At range extreme (>0.8 or <0.2)                       │
  │  - Momentum in same direction (bulls at top, bears at    │
  │    bottom)                                               │
  │  - Flow fading (volume decreasing)                       │
  │  → Trade AGAINST momentum                                │
  └──────────────────────────────────────────────────────────┘

  ┌─ BREAKOUT CONTEXT (→ CONTINUATION) ──────────────────────┐
  │  - Breaking range boundary (new high/low)                │
  │  - Volume increasing                                     │
  │  - Flow supporting direction                             │
  │  → Trade WITH momentum                                   │
  └──────────────────────────────────────────────────────────┘

  ┌─ MID-RANGE CONTEXT (→ COUNTER-TREND) ────────────────────┐
  │  - In middle of range (0.35-0.65)                        │
  │  - No clear exhaustion or breakout                       │
  │  → Default: counter-trend (berserker = mean reversion)   │
  └──────────────────────────────────────────────────────────┘
""")


if __name__ == "__main__":
    main()
