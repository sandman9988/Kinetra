#!/usr/bin/env python3
"""
Test marginal/incremental gains of stacking direction filters.

Question: What is the cumulative edge when we add each filter?
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from kinetra.mt5_connector import load_csv_data
from kinetra.physics_engine import PhysicsEngine


def main():
    # Load data
    project_root = Path(__file__).parent.parent
    csv_files = list(project_root.glob("*BTCUSD*.csv"))
    if not csv_files:
        print("No BTCUSD CSV file found")
        return

    data = load_csv_data(str(csv_files[0]))
    print(f"Loaded {len(data)} bars")

    # Compute features
    engine = PhysicsEngine(lookback=20)
    physics = engine.compute_physics_state(data['close'], data['volume'], include_percentiles=True)

    df = data.copy()
    df['energy_pct'] = physics['energy_pct']
    df['damping_pct'] = physics['damping_pct']
    df['entropy_pct'] = physics['entropy_pct']
    df['returns'] = df['close'].pct_change()
    df['momentum_5'] = df['close'].pct_change(5)

    # Forward returns
    df['fwd_return'] = df['returns'].shift(-1)
    df['fwd_direction'] = np.sign(df['fwd_return'])

    # Flow consistency
    df['return_sign'] = np.sign(df['returns'])
    df['flow_consistency'] = df['return_sign'].rolling(5).apply(
        lambda x: (x == x.iloc[-1]).mean() if len(x) > 0 else 0.5, raw=False
    ).fillna(0.5)

    # Laminar score
    df['trend_5'] = df['close'].pct_change(5)
    df['vol_5'] = df['returns'].rolling(5).std()
    df['smoothness'] = df['trend_5'].abs() / (df['vol_5'] + 1e-10)
    df['laminar_score'] = df['flow_consistency'] * (1 - df['entropy_pct']) * df['smoothness'].clip(0, 5) / 5

    window = min(200, len(df))
    df['laminar_score_pct'] = df['laminar_score'].rolling(window, min_periods=20).apply(
        lambda x: (x.iloc[-1] > x.iloc[:-1]).mean() if len(x) > 1 else 0.5, raw=False
    ).fillna(0.5)

    df = df.dropna()

    # Base berserker condition
    berserker = (df['energy_pct'] > 0.75) & (df['damping_pct'] < 0.25)
    df_berk = df[berserker].copy()

    print(f"\nBerserker bars: {len(df_berk)}")

    # Counter-trend direction
    df_berk['counter_direction'] = -np.sign(df_berk['momentum_5'])
    df_berk['counter_correct'] = df_berk['counter_direction'] == df_berk['fwd_direction']

    print("\n" + "=" * 70)
    print("MARGINAL GAINS: STACKING DIRECTION FILTERS")
    print("=" * 70)

    # Test filters incrementally
    filters = [
        ('Base: Counter-trend on berserker', pd.Series(True, index=df_berk.index)),
        ('+ Low entropy (< 0.4)', df_berk['entropy_pct'] < 0.4),
        ('+ High flow consistency (> 0.7)', df_berk['flow_consistency'] > 0.7),
        ('+ Laminar flow (> 0.6)', df_berk['laminar_score_pct'] > 0.6),
        ('+ High volume (> 0.6)', df_berk['volume_pct'] > 0.6 if 'volume_pct' in df_berk.columns else pd.Series(True, index=df_berk.index)),
    ]

    # Compute volume percentile if not present
    if 'volume_pct' not in df_berk.columns:
        vol_window = min(500, len(df))
        df_berk['volume_pct'] = df['volume'].rolling(vol_window).apply(
            lambda x: (x.iloc[-1] > x.iloc[:-1]).mean() if len(x) > 1 else 0.5, raw=False
        ).reindex(df_berk.index).fillna(0.5)

    print("\n--- Individual Filter Impact ---\n")

    for name, mask in filters:
        subset = df_berk[mask]
        if len(subset) < 30:
            print(f"  {name}: <30 signals, skipping")
            continue

        accuracy = subset['counter_correct'].mean() * 100
        edge = accuracy - 50

        print(f"  {name}")
        print(f"    Signals: {len(subset)}, Accuracy: {accuracy:.1f}%, Edge: {edge:+.1f}%")

    # === CUMULATIVE STACKING ===
    print("\n--- Cumulative Filter Stacking ---\n")

    cumulative_mask = pd.Series(True, index=df_berk.index)
    cumulative_filters = [
        ('Base: Berserker', pd.Series(True, index=df_berk.index)),
        ('+ Low entropy', df_berk['entropy_pct'] < 0.4),
        ('+ Laminar flow', df_berk['laminar_score_pct'] > 0.6),
        ('+ High flow', df_berk['flow_consistency'] > 0.7),
    ]

    prev_accuracy = 50
    for name, mask in cumulative_filters:
        cumulative_mask = cumulative_mask & mask
        subset = df_berk[cumulative_mask]

        if len(subset) < 20:
            print(f"  {name}: <20 signals after stacking")
            continue

        accuracy = subset['counter_correct'].mean() * 100
        edge = accuracy - 50
        marginal = accuracy - prev_accuracy

        print(f"  {name}")
        print(f"    Signals: {len(subset)}, Accuracy: {accuracy:.1f}%, "
              f"Edge: {edge:+.1f}%, Marginal: {marginal:+.1f}%")

        prev_accuracy = accuracy

    # === BEST COMBINATION ===
    print("\n" + "=" * 70)
    print("OPTIMAL COMBINATION SEARCH")
    print("=" * 70)

    # Try all combinations
    from itertools import combinations

    features = {
        'low_entropy': df_berk['entropy_pct'] < 0.4,
        'high_flow': df_berk['flow_consistency'] > 0.7,
        'laminar': df_berk['laminar_score_pct'] > 0.6,
        'high_volume': df_berk['volume_pct'] > 0.6,
    }

    best_combo = None
    best_edge = 0
    best_signals = 0

    all_combos = []
    for r in range(1, len(features) + 1):
        for combo in combinations(features.keys(), r):
            mask = pd.Series(True, index=df_berk.index)
            for f in combo:
                mask = mask & features[f]

            subset = df_berk[mask]
            if len(subset) >= 30:
                accuracy = subset['counter_correct'].mean() * 100
                edge = accuracy - 50
                all_combos.append({
                    'combo': '+'.join(combo),
                    'signals': len(subset),
                    'accuracy': accuracy,
                    'edge': edge,
                })

                if edge > best_edge:
                    best_edge = edge
                    best_combo = combo
                    best_signals = len(subset)

    # Sort by edge
    all_combos = sorted(all_combos, key=lambda x: x['edge'], reverse=True)

    print("\nTop 5 combinations by edge:\n")
    for i, c in enumerate(all_combos[:5], 1):
        print(f"  {i}. {c['combo']}")
        print(f"     Signals: {c['signals']}, Accuracy: {c['accuracy']:.1f}%, Edge: {c['edge']:+.1f}%")

    print("\n" + "=" * 70)
    print("FINAL DIRECTION SIGNAL FORMULA")
    print("=" * 70)

    if best_combo:
        print(f"\n  Optimal filters: {' + '.join(best_combo)}")
        print(f"  Signals: {best_signals}")
        print(f"  Direction edge: {best_edge:+.1f}%")
        print(f"\n  Signal: BERSERKER + {' + '.join(best_combo)} → COUNTER-TREND")


if __name__ == "__main__":
    main()
