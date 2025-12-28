#!/usr/bin/env python3
"""
Extended Physics Features Analysis

Testing additional physics concepts for fat candle prediction:
1. Liquidity (volume depth, spread proxy)
2. ROC (Rate of Change - normalized velocity)
3. Acceleration (momentum of momentum - second derivative)
4. Angular Momentum (rotational energy in price cycles)
5. Inertia (resistance to direction change)
6. Potential Energy (stored energy in compression/ranges)
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from kinetra.mt5_connector import load_csv_data
from kinetra.physics_engine import PhysicsEngine


def compute_extended_physics(df: pd.DataFrame, lookback: int = 20) -> pd.DataFrame:
    """
    Compute extended physics features.
    """
    result = df.copy()

    # === 1. LIQUIDITY (approximated from OHLC) ===
    # Spread proxy: (High - Low) / Volume = how much price moves per unit volume
    # Lower = more liquid (big volume, small move)
    bar_range = df['high'] - df['low']
    result['liquidity_proxy'] = df['volume'] / (bar_range.clip(lower=1e-10) * df['close'] / 100)
    result['liquidity_pct'] = result['liquidity_proxy'].rolling(lookback * 5).apply(
        lambda x: (x.iloc[-1] > x.iloc[:-1]).mean() if len(x) > 1 else 0.5, raw=False
    ).fillna(0.5)

    # Volume depth: how consistent is volume? (stable = deep liquidity)
    result['volume_stability'] = 1 - (df['volume'].rolling(lookback).std() / df['volume'].rolling(lookback).mean().clip(lower=1))

    # === 2. ROC (Rate of Change - normalized) ===
    # ROC = (P_t - P_{t-n}) / P_{t-n} * 100
    for period in [5, 10, 20]:
        result[f'roc_{period}'] = df['close'].pct_change(period) * 100

    # Normalized ROC (by volatility)
    volatility = df['close'].pct_change().rolling(lookback).std()
    result['roc_normalized'] = result['roc_20'] / (volatility * 100).clip(lower=0.1)

    # === 3. ACCELERATION (second derivative) ===
    # Velocity = first derivative (momentum)
    velocity = df['close'].pct_change()
    # Acceleration = change in velocity
    acceleration = velocity.diff()
    result['acceleration'] = acceleration

    # Smoothed acceleration
    result['acceleration_smooth'] = acceleration.rolling(5).mean()

    # Acceleration percentile
    result['acceleration_pct'] = acceleration.rolling(lookback * 5).apply(
        lambda x: (x.iloc[-1] > x.iloc[:-1]).mean() if len(x) > 1 else 0.5, raw=False
    ).fillna(0.5)

    # Jerk (change in acceleration - third derivative)
    result['jerk'] = acceleration.diff()

    # === 4. ANGULAR MOMENTUM (rotational/cyclical) ===
    # Price oscillation around mean (like rotation)
    price_mean = df['close'].rolling(lookback).mean()
    price_deviation = df['close'] - price_mean

    # Angular position (normalized deviation)
    result['angular_position'] = price_deviation / df['close'].rolling(lookback).std().clip(lower=1e-10)

    # Angular velocity (rate of rotation)
    result['angular_velocity'] = result['angular_position'].diff()

    # Angular momentum = position * velocity (cross product analog)
    result['angular_momentum'] = result['angular_position'] * result['angular_velocity']

    # === 5. INERTIA (resistance to direction change) ===
    # Trend persistence = how long has current direction continued?
    direction = np.sign(df['close'].pct_change())

    # Count consecutive same-direction bars
    def count_consecutive(series):
        counts = []
        count = 1
        for i in range(len(series)):
            if i == 0:
                counts.append(1)
            elif series.iloc[i] == series.iloc[i-1] and series.iloc[i] != 0:
                count += 1
                counts.append(count)
            else:
                count = 1
                counts.append(count)
        return pd.Series(counts, index=series.index)

    result['bars_same_direction'] = count_consecutive(direction)

    # Inertia = momentum * time in direction
    result['inertia'] = abs(velocity) * result['bars_same_direction']

    # === 6. POTENTIAL ENERGY (stored in compression) ===
    # Range compression = current range vs average range
    avg_range = bar_range.rolling(lookback).mean()
    result['range_compression'] = 1 - (bar_range / avg_range.clip(lower=1e-10))

    # Bollinger Band width (compression indicator)
    bb_std = df['close'].rolling(lookback).std()
    bb_mean = df['close'].rolling(lookback).mean()
    result['bb_width'] = (2 * bb_std) / bb_mean.clip(lower=1e-10)

    # Potential energy = compression * volatility (coiled spring)
    result['potential_energy'] = result['range_compression'].clip(lower=0) * volatility

    # === 7. IMPULSE (force over time) ===
    # Impulse = momentum change over time window
    momentum = df['close'].pct_change(lookback)
    result['impulse'] = momentum.diff(5)  # Change in momentum over 5 bars

    # Forward return for testing
    result['fwd_return'] = df['close'].pct_change().shift(-1)
    result['fwd_direction'] = np.sign(result['fwd_return'])
    result['fwd_abs_return'] = result['fwd_return'].abs()

    return result.dropna()


def test_physics_features(df: pd.DataFrame, berserker_mask: pd.Series):
    """
    Test which physics features predict fat candles and direction.
    """
    df_berk = df[berserker_mask].copy()

    # Define fat candle
    fat_threshold = df['fwd_abs_return'].quantile(0.75)
    df_berk['is_fat'] = df_berk['fwd_abs_return'] > fat_threshold

    print("\n" + "=" * 80)
    print("EXTENDED PHYSICS FEATURES - FAT CANDLE PREDICTION")
    print("=" * 80)

    # Test each feature for fat candle prediction
    features = [
        # Liquidity
        ('High liquidity', df_berk['liquidity_pct'] > 0.7),
        ('Low liquidity', df_berk['liquidity_pct'] < 0.3),
        ('Stable volume', df_berk['volume_stability'] > 0.5),

        # ROC
        ('High ROC (>2%)', df_berk['roc_20'] > 2),
        ('Low ROC (<-2%)', df_berk['roc_20'] < -2),
        ('Extreme ROC (normalized)', df_berk['roc_normalized'].abs() > 2),

        # Acceleration
        ('Positive acceleration', df_berk['acceleration_smooth'] > 0),
        ('Negative acceleration', df_berk['acceleration_smooth'] < 0),
        ('Extreme acceleration', df_berk['acceleration_pct'].apply(lambda x: x > 0.9 or x < 0.1)),
        ('High jerk (abrupt change)', df_berk['jerk'].abs() > df_berk['jerk'].abs().quantile(0.9)),

        # Angular momentum
        ('High angular momentum', df_berk['angular_momentum'].abs() > df_berk['angular_momentum'].abs().quantile(0.75)),
        ('Positive angular mom', df_berk['angular_momentum'] > 0),
        ('Negative angular mom', df_berk['angular_momentum'] < 0),

        # Inertia
        ('High inertia (>5 bars)', df_berk['bars_same_direction'] > 5),
        ('Low inertia (<3 bars)', df_berk['bars_same_direction'] < 3),

        # Potential energy
        ('High compression', df_berk['range_compression'] > 0.3),
        ('Narrow BB', df_berk['bb_width'] < df_berk['bb_width'].quantile(0.25)),
        ('High potential energy', df_berk['potential_energy'] > df_berk['potential_energy'].quantile(0.75)),

        # Impulse
        ('Positive impulse', df_berk['impulse'] > 0),
        ('Negative impulse', df_berk['impulse'] < 0),
        ('Strong impulse', df_berk['impulse'].abs() > df_berk['impulse'].abs().quantile(0.75)),
    ]

    # Baseline
    baseline_fat_rate = df_berk['is_fat'].mean() * 100

    print(f"\n  Baseline fat candle rate: {baseline_fat_rate:.1f}%")
    print(f"\n  {'Feature':>35} │ {'Count':>6} │ {'Fat%':>7} │ {'Lift':>6}")
    print("  " + "─" * 65)

    results = []
    for name, mask in features:
        subset = df_berk[mask]
        if len(subset) < 30:
            continue

        fat_rate = subset['is_fat'].mean() * 100
        lift = fat_rate / baseline_fat_rate

        results.append({'name': name, 'count': len(subset), 'fat_rate': fat_rate, 'lift': lift})
        print(f"  {name:>35} │ {len(subset):>6} │ {fat_rate:>6.1f}% │ {lift:>5.2f}x")

    # Sort by lift
    results = sorted(results, key=lambda x: x['lift'], reverse=True)

    print("\n  TOP 5 FAT CANDLE PREDICTORS:")
    for r in results[:5]:
        print(f"    {r['name']}: {r['fat_rate']:.1f}% fat rate ({r['lift']:.2f}x lift)")

    return results


def test_direction_features(df: pd.DataFrame, berserker_mask: pd.Series):
    """
    Test which physics features predict direction.
    """
    df_berk = df[berserker_mask].copy()

    print("\n" + "=" * 80)
    print("EXTENDED PHYSICS FEATURES - DIRECTION PREDICTION")
    print("=" * 80)

    features = [
        # Acceleration - decelerating trend = reversal?
        ('Positive accel + up mom', (df_berk['acceleration_smooth'] > 0) & (df_berk['roc_20'] > 0), 'up'),
        ('Negative accel + up mom (slowing)', (df_berk['acceleration_smooth'] < 0) & (df_berk['roc_20'] > 0), 'down'),
        ('Positive accel + down mom', (df_berk['acceleration_smooth'] > 0) & (df_berk['roc_20'] < 0), 'up'),
        ('Negative accel + down mom (slowing)', (df_berk['acceleration_smooth'] < 0) & (df_berk['roc_20'] < 0), 'down'),

        # Angular momentum
        ('Positive angular mom', df_berk['angular_momentum'] > df_berk['angular_momentum'].quantile(0.75), 'up'),
        ('Negative angular mom', df_berk['angular_momentum'] < df_berk['angular_momentum'].quantile(0.25), 'down'),

        # Inertia + reversal
        ('High inertia bullish (exhausted?)', (df_berk['bars_same_direction'] > 5) & (df_berk['roc_20'] > 0), 'down'),
        ('High inertia bearish (exhausted?)', (df_berk['bars_same_direction'] > 5) & (df_berk['roc_20'] < 0), 'up'),

        # Potential energy release
        ('Compressed + bullish', (df_berk['range_compression'] > 0.3) & (df_berk['roc_5'] > 0), 'up'),
        ('Compressed + bearish', (df_berk['range_compression'] > 0.3) & (df_berk['roc_5'] < 0), 'down'),

        # Impulse direction
        ('Strong positive impulse', df_berk['impulse'] > df_berk['impulse'].quantile(0.8), 'up'),
        ('Strong negative impulse', df_berk['impulse'] < df_berk['impulse'].quantile(0.2), 'down'),

        # Liquidity + direction
        ('Low liquidity + bullish', (df_berk['liquidity_pct'] < 0.3) & (df_berk['roc_5'] > 0), 'up'),
        ('Low liquidity + bearish', (df_berk['liquidity_pct'] < 0.3) & (df_berk['roc_5'] < 0), 'down'),
    ]

    print(f"\n  {'Feature':>45} │ {'Count':>6} │ {'Correct%':>10} │ {'Edge':>8}")
    print("  " + "─" * 80)

    results = []
    for name, mask, expected in features:
        subset = df_berk[mask]
        if len(subset) < 30:
            continue

        if expected == 'up':
            correct = (subset['fwd_direction'] > 0).mean() * 100
        else:
            correct = (subset['fwd_direction'] < 0).mean() * 100

        edge = correct - 50
        results.append({'name': name, 'count': len(subset), 'correct': correct, 'edge': edge, 'expected': expected})
        print(f"  {name:>45} │ {len(subset):>6} │ {correct:>9.1f}% │ {edge:>+7.1f}%")

    # Sort by edge
    results = sorted(results, key=lambda x: x['edge'], reverse=True)

    print("\n  TOP 5 DIRECTION PREDICTORS:")
    for r in results[:5]:
        print(f"    {r['name']}: {r['correct']:.1f}% correct ({r['expected']}, {r['edge']:+.1f}% edge)")

    return results


def test_combined_physics(df: pd.DataFrame, berserker_mask: pd.Series):
    """
    Test combinations of physics features.
    """
    df_berk = df[berserker_mask].copy()

    # Add buying pressure from earlier
    bar_range = df['high'] - df['low']
    buying_pressure = (df['close'] - df['low']) / bar_range.clip(lower=1e-10)
    df_berk['buying_pressure'] = buying_pressure.rolling(5).mean().loc[df_berk.index]

    print("\n" + "=" * 80)
    print("COMBINED PHYSICS SIGNALS")
    print("=" * 80)

    # Combine best features from each category
    combos = [
        # Exhaustion combo: high inertia + deceleration + extreme buying pressure
        ('BULL EXHAUSTION: inertia + decel + high BP',
         (df_berk['bars_same_direction'] > 4) &
         (df_berk['acceleration_smooth'] < 0) &
         (df_berk['roc_20'] > 0) &
         (df_berk['buying_pressure'] > 0.6),
         'down'),

        ('BEAR EXHAUSTION: inertia + decel + low BP',
         (df_berk['bars_same_direction'] > 4) &
         (df_berk['acceleration_smooth'] > 0) &
         (df_berk['roc_20'] < 0) &
         (df_berk['buying_pressure'] < 0.4),
         'up'),

        # Compression breakout: potential energy + impulse
        ('BREAKOUT UP: compression + positive impulse',
         (df_berk['range_compression'] > 0.2) &
         (df_berk['impulse'] > 0) &
         (df_berk['acceleration_smooth'] > 0),
         'up'),

        ('BREAKOUT DOWN: compression + negative impulse',
         (df_berk['range_compression'] > 0.2) &
         (df_berk['impulse'] < 0) &
         (df_berk['acceleration_smooth'] < 0),
         'down'),

        # Low liquidity moves (thin market = big moves)
        ('THIN MARKET UP: low liquidity + bullish',
         (df_berk['liquidity_pct'] < 0.3) &
         (df_berk['roc_5'] > 0) &
         (df_berk['acceleration_smooth'] > 0),
         'up'),

        ('THIN MARKET DOWN: low liquidity + bearish',
         (df_berk['liquidity_pct'] < 0.3) &
         (df_berk['roc_5'] < 0) &
         (df_berk['acceleration_smooth'] < 0),
         'down'),

        # Angular momentum reversal
        ('ROTATION REVERSAL UP: negative ang mom extremes',
         (df_berk['angular_momentum'] < df_berk['angular_momentum'].quantile(0.1)) &
         (df_berk['angular_velocity'] > 0),
         'up'),

        ('ROTATION REVERSAL DOWN: positive ang mom extremes',
         (df_berk['angular_momentum'] > df_berk['angular_momentum'].quantile(0.9)) &
         (df_berk['angular_velocity'] < 0),
         'down'),
    ]

    print(f"\n  {'Combo':>50} │ {'Count':>6} │ {'Correct%':>10} │ {'Edge':>8}")
    print("  " + "─" * 85)

    for name, mask, expected in combos:
        subset = df_berk[mask]
        if len(subset) < 20:
            print(f"  {name:>50} │ {'<20':>6} │ {'-':>10} │ {'-':>8}")
            continue

        if expected == 'up':
            correct = (subset['fwd_direction'] > 0).mean() * 100
        else:
            correct = (subset['fwd_direction'] < 0).mean() * 100

        edge = correct - 50
        print(f"  {name:>50} │ {len(subset):>6} │ {correct:>9.1f}% │ {edge:>+7.1f}%")


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

    # Compute extended physics
    print("Computing extended physics features...")
    df = compute_extended_physics(data, lookback=20)

    # Add original physics
    df['energy_pct'] = physics['energy_pct'].loc[df.index]
    df['damping_pct'] = physics['damping_pct'].loc[df.index]

    # Flow consistency
    df['return_sign'] = np.sign(df['close'].pct_change())
    df['flow_consistency'] = df['return_sign'].rolling(5).apply(
        lambda x: (x == x.iloc[-1]).mean() if len(x) > 0 else 0.5, raw=False
    ).fillna(0.5)

    df = df.dropna()

    # Signals
    berserker = (df['energy_pct'] > 0.75) & (df['damping_pct'] < 0.25)

    print(f"\nBerserker signals: {berserker.sum()}")

    # Run tests
    test_physics_features(df, berserker)
    test_direction_features(df, berserker)
    test_combined_physics(df, berserker)

    # Summary
    print("\n" + "=" * 80)
    print("EXTENDED PHYSICS SUMMARY")
    print("=" * 80)
    print("""
  NEW PHYSICS FEATURES TESTED:

  1. LIQUIDITY (volume/range):
     - Low liquidity = thin market = bigger moves
     - High liquidity = deep market = absorbed moves

  2. ROC (Rate of Change):
     - Normalized velocity measure
     - Extreme ROC = overextension

  3. ACCELERATION (d²P/dt²):
     - Positive accel = momentum building
     - Negative accel = momentum fading (reversal signal)
     - KEY: deceleration + trend = exhaustion

  4. ANGULAR MOMENTUM:
     - Rotation around mean (cyclical behavior)
     - Extreme angular mom = cycle turning point

  5. INERTIA:
     - How long trend has continued
     - High inertia + deceleration = exhausted

  6. POTENTIAL ENERGY:
     - Stored in compression (tight ranges)
     - Releases as breakout

  7. IMPULSE:
     - Momentum change over time
     - Strong impulse = directional bias

  BEST COMBINATIONS:
  - Exhaustion: inertia + deceleration + extreme buying pressure
  - Breakout: compression + impulse + acceleration
  - Thin market: low liquidity + directional momentum
""")


if __name__ == "__main__":
    main()
