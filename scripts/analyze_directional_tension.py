#!/usr/bin/env python3
"""
Directional Tension Analysis

Physics insight: Energy tells us WHEN, but we need ASYMMETRY to tell us WHERE.

Key concepts:
1. Tension = stored energy waiting to release (berserker detects this)
2. Direction = which side has more tension (asymmetric pressure)

Measures to test:
1. Order flow pressure (approximated from OHLC)
2. Directional energy asymmetry (up-moves vs down-moves energy)
3. Resistance vs support tension (price compression)
4. Volume-weighted directional pressure
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from kinetra.mt5_connector import load_csv_data
from kinetra.physics_engine import PhysicsEngine


def compute_directional_tension(df: pd.DataFrame, lookback: int = 20) -> pd.DataFrame:
    """
    Compute directional tension metrics.

    These measure WHICH SIDE has more stored energy.
    """
    result = df.copy()

    # === 1. ORDER FLOW PRESSURE (from OHLC) ===
    # Close position within bar: (C - L) / (H - L) = buying pressure
    # 1.0 = closed at high (buyers won), 0.0 = closed at low (sellers won)
    bar_range = df['high'] - df['low']
    result['buying_pressure'] = (df['close'] - df['low']) / bar_range.clip(lower=1e-10)

    # Cumulative buying pressure over lookback
    result['cum_buying_pressure'] = result['buying_pressure'].rolling(lookback).mean()

    # === 2. DIRECTIONAL ENERGY (up-moves vs down-moves) ===
    returns = df['close'].pct_change()

    # Energy in up-moves vs down-moves
    up_returns = returns.clip(lower=0)
    down_returns = returns.clip(upper=0).abs()

    # Energy = 0.5 * v^2 (velocity squared)
    up_energy = (up_returns ** 2).rolling(lookback).sum()
    down_energy = (down_returns ** 2).rolling(lookback).sum()

    # Asymmetry: positive = more up energy, negative = more down energy
    total_energy = up_energy + down_energy + 1e-10
    result['energy_asymmetry'] = (up_energy - down_energy) / total_energy

    # === 3. SUPPORT/RESISTANCE TENSION ===
    # How close is price to recent highs vs lows?
    rolling_high = df['high'].rolling(lookback).max()
    rolling_low = df['low'].rolling(lookback).min()
    price_range = rolling_high - rolling_low + 1e-10

    # Position in range: 1.0 = at resistance, 0.0 = at support
    result['range_position'] = (df['close'] - rolling_low) / price_range

    # Tension at extremes (higher at boundaries)
    result['boundary_tension'] = 2 * abs(result['range_position'] - 0.5)

    # === 4. VOLUME-WEIGHTED DIRECTION ===
    # Up-volume vs down-volume
    bar_direction = np.sign(df['close'] - df['open'])
    volume_direction = bar_direction * df['volume']
    result['volume_flow'] = volume_direction.rolling(lookback).sum() / df['volume'].rolling(lookback).sum()

    # === 5. MOMENTUM EXHAUSTION ===
    # Price momentum vs volume momentum
    price_momentum = df['close'].pct_change(lookback)
    volume_momentum = df['volume'].pct_change(lookback)

    # Divergence: if price rising but volume falling = bulls exhausting
    result['momentum_volume_ratio'] = price_momentum / (volume_momentum.abs() + 0.01)

    # === 6. CANDLE BODY RATIO (directional conviction) ===
    body = abs(df['close'] - df['open'])
    result['body_ratio'] = body / bar_range.clip(lower=1e-10)

    # Directional body
    result['directional_body'] = np.sign(df['close'] - df['open']) * result['body_ratio']
    result['cum_directional_body'] = result['directional_body'].rolling(lookback).mean()

    # === 7. WICKS ASYMMETRY (rejection pressure) ===
    upper_wick = df['high'] - df[['close', 'open']].max(axis=1)
    lower_wick = df[['close', 'open']].min(axis=1) - df['low']

    total_wicks = upper_wick + lower_wick + 1e-10
    # Positive = more upper wick (sellers rejecting), Negative = more lower wick (buyers rejecting)
    result['wick_asymmetry'] = (upper_wick - lower_wick) / total_wicks
    result['cum_wick_asymmetry'] = result['wick_asymmetry'].rolling(lookback).mean()

    # Forward return for testing
    result['fwd_return'] = returns.shift(-1)
    result['fwd_direction'] = np.sign(result['fwd_return'])

    return result.dropna()


def test_directional_signals(df: pd.DataFrame, berserker_mask: pd.Series):
    """
    Test which directional tension measures predict direction on berserker bars.
    """
    df_berk = df[berserker_mask].copy()

    print("\n" + "=" * 80)
    print("DIRECTIONAL TENSION SIGNALS ON BERSERKER BARS")
    print("=" * 80)

    signals = {
        # Buying pressure: high = expect continuation up, low = expect continuation down
        'High buying pressure (>0.6)': df_berk['cum_buying_pressure'] > 0.6,
        'Low buying pressure (<0.4)': df_berk['cum_buying_pressure'] < 0.4,

        # Energy asymmetry: positive = more up energy
        'Positive energy asymmetry': df_berk['energy_asymmetry'] > 0.1,
        'Negative energy asymmetry': df_berk['energy_asymmetry'] < -0.1,

        # Range position: at resistance or support
        'At resistance (>0.8)': df_berk['range_position'] > 0.8,
        'At support (<0.2)': df_berk['range_position'] < 0.2,

        # Volume flow
        'Positive volume flow': df_berk['volume_flow'] > 0.2,
        'Negative volume flow': df_berk['volume_flow'] < -0.2,

        # Wick asymmetry: upper wicks = seller rejection
        'Upper wick heavy (sellers)': df_berk['cum_wick_asymmetry'] > 0.1,
        'Lower wick heavy (buyers)': df_berk['cum_wick_asymmetry'] < -0.1,

        # Directional body
        'Bullish bodies': df_berk['cum_directional_body'] > 0.1,
        'Bearish bodies': df_berk['cum_directional_body'] < -0.1,
    }

    print(f"\n{'Signal':>35} │ {'Count':>7} │ {'Up%':>7} │ {'Down%':>7} │ {'Edge':>8}")
    print("─" * 75)

    results = []

    for name, mask in signals.items():
        subset = df_berk[mask]
        if len(subset) < 30:
            continue

        up_pct = (subset['fwd_direction'] > 0).mean() * 100
        down_pct = (subset['fwd_direction'] < 0).mean() * 100

        # Determine expected direction and edge
        if 'resistance' in name.lower() or 'upper wick' in name.lower() or 'bearish' in name.lower():
            expected = 'down'
            edge = down_pct - 50
        elif 'support' in name.lower() or 'lower wick' in name.lower() or 'bullish' in name.lower():
            expected = 'up'
            edge = up_pct - 50
        elif 'high buying' in name.lower() or 'positive' in name.lower():
            # High buying pressure at berserker = exhaustion = expect reversal DOWN
            expected = 'reversal'
            edge = down_pct - 50
        elif 'low buying' in name.lower() or 'negative' in name.lower():
            expected = 'reversal'
            edge = up_pct - 50
        else:
            expected = 'unknown'
            edge = max(up_pct, down_pct) - 50

        print(f"{name:>35} │ {len(subset):>7} │ {up_pct:>6.1f}% │ {down_pct:>6.1f}% │ {edge:>+7.1f}%")

        results.append({
            'signal': name,
            'count': len(subset),
            'up_pct': up_pct,
            'down_pct': down_pct,
            'edge': edge,
        })

    return results


def test_combined_signals(df: pd.DataFrame, berserker_mask: pd.Series):
    """
    Test combinations of directional signals.
    """
    df_berk = df[berserker_mask].copy()

    print("\n" + "=" * 80)
    print("COMBINED DIRECTIONAL SIGNALS (TENSION ASYMMETRY)")
    print("=" * 80)

    # Hypothesis: At resistance + upper wicks + exhausted buying = SHORT
    #             At support + lower wicks + exhausted selling = LONG

    combos = [
        ('BULLS EXHAUSTED (short)',
         (df_berk['range_position'] > 0.7) &
         (df_berk['cum_wick_asymmetry'] > 0) &
         (df_berk['cum_buying_pressure'] > 0.5),
         'down'),

        ('BEARS EXHAUSTED (long)',
         (df_berk['range_position'] < 0.3) &
         (df_berk['cum_wick_asymmetry'] < 0) &
         (df_berk['cum_buying_pressure'] < 0.5),
         'up'),

        ('UPWARD TENSION (long)',
         (df_berk['energy_asymmetry'] < -0.1) &  # More down energy = coiled for up
         (df_berk['volume_flow'] < 0) &  # Selling exhausted
         (df_berk['range_position'] < 0.4),  # Near support
         'up'),

        ('DOWNWARD TENSION (short)',
         (df_berk['energy_asymmetry'] > 0.1) &  # More up energy = coiled for down
         (df_berk['volume_flow'] > 0) &  # Buying exhausted
         (df_berk['range_position'] > 0.6),  # Near resistance
         'down'),

        ('WICK REJECTION UP',
         (df_berk['cum_wick_asymmetry'] < -0.15) &  # Strong lower wicks
         (df_berk['range_position'] < 0.5),
         'up'),

        ('WICK REJECTION DOWN',
         (df_berk['cum_wick_asymmetry'] > 0.15) &  # Strong upper wicks
         (df_berk['range_position'] > 0.5),
         'down'),
    ]

    print(f"\n{'Signal':>35} │ {'Count':>7} │ {'Correct%':>10} │ {'Edge':>8}")
    print("─" * 70)

    for name, mask, expected_dir in combos:
        subset = df_berk[mask]
        if len(subset) < 20:
            print(f"{name:>35} │ {'<20':>7} │ {'-':>10} │ {'-':>8}")
            continue

        if expected_dir == 'up':
            correct_pct = (subset['fwd_direction'] > 0).mean() * 100
        else:
            correct_pct = (subset['fwd_direction'] < 0).mean() * 100

        edge = correct_pct - 50

        print(f"{name:>35} │ {len(subset):>7} │ {correct_pct:>9.1f}% │ {edge:>+7.1f}%")


def test_counter_trend_with_tension(df: pd.DataFrame, berserker_mask: pd.Series):
    """
    Enhance counter-trend signal with directional tension.
    """
    df_berk = df[berserker_mask].copy()
    df_berk['momentum_5'] = df['close'].pct_change(5).loc[df_berk.index]
    df_berk['counter_direction'] = -np.sign(df_berk['momentum_5'])

    print("\n" + "=" * 80)
    print("COUNTER-TREND + TENSION CONFIRMATION")
    print("=" * 80)

    # Base counter-trend
    base_correct = (df_berk['counter_direction'] == df_berk['fwd_direction']).mean() * 100

    print(f"\n  Base counter-trend: {base_correct:.1f}% (edge: {base_correct-50:+.1f}%)")

    # Add tension confirmation
    # If counter-trend is UP (momentum was down), we want:
    # - Lower wicks (buyers rejecting)
    # - Near support
    # - Negative volume flow (sellers exhausted)

    counter_up = df_berk['counter_direction'] > 0
    counter_down = df_berk['counter_direction'] < 0

    enhancements = [
        ('+ Wick confirmation',
         (counter_up & (df_berk['cum_wick_asymmetry'] < 0)) |
         (counter_down & (df_berk['cum_wick_asymmetry'] > 0))),

        ('+ Range confirmation',
         (counter_up & (df_berk['range_position'] < 0.5)) |
         (counter_down & (df_berk['range_position'] > 0.5))),

        ('+ Volume flow confirmation',
         (counter_up & (df_berk['volume_flow'] < 0)) |
         (counter_down & (df_berk['volume_flow'] > 0))),

        ('+ Energy asymmetry confirmation',
         (counter_up & (df_berk['energy_asymmetry'] < 0)) |
         (counter_down & (df_berk['energy_asymmetry'] > 0))),

        ('ALL confirmations',
         ((counter_up & (df_berk['cum_wick_asymmetry'] < 0) & (df_berk['range_position'] < 0.5)) |
          (counter_down & (df_berk['cum_wick_asymmetry'] > 0) & (df_berk['range_position'] > 0.5)))),
    ]

    print(f"\n{'Enhancement':>35} │ {'Count':>7} │ {'Accuracy':>10} │ {'Edge':>8}")
    print("─" * 70)

    for name, mask in enhancements:
        subset = df_berk[mask]
        if len(subset) < 30:
            print(f"{name:>35} │ {'<30':>7} │ {'-':>10} │ {'-':>8}")
            continue

        correct = (subset['counter_direction'] == subset['fwd_direction']).mean() * 100
        edge = correct - 50

        print(f"{name:>35} │ {len(subset):>7} │ {correct:>9.1f}% │ {edge:>+7.1f}%")


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

    # Compute directional tension
    print("Computing directional tension metrics...")
    df = compute_directional_tension(data, lookback=20)

    # Add physics state
    df['energy_pct'] = physics['energy_pct'].loc[df.index]
    df['damping_pct'] = physics['damping_pct'].loc[df.index]

    # Flow consistency (from earlier analysis)
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
    test_directional_signals(df, berserker)
    test_combined_signals(df, berserker)
    test_counter_trend_with_tension(df, berserker_plus)

    # Summary
    print("\n" + "=" * 80)
    print("DIRECTIONAL TENSION SUMMARY")
    print("=" * 80)
    print("""
  PHYSICS OF DIRECTION:

  1. TENSION tells us WHEN (berserker = high energy, low damping)
  2. ASYMMETRY tells us WHERE (which side has more stored energy)

  Key Directional Signals:
  - Wick asymmetry: Upper wicks = sellers rejecting = bearish
                    Lower wicks = buyers rejecting = bullish
  - Range position: At resistance + berserker = short
                    At support + berserker = long
  - Volume flow: Exhausted buyers (positive flow + berserker) = short
                 Exhausted sellers (negative flow + berserker) = long
  - Energy asymmetry: More up-energy = exhausted bulls = expect down
                      More down-energy = exhausted bears = expect up

  COMBINE with counter-trend for enhanced direction signal.
""")


if __name__ == "__main__":
    main()
