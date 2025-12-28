#!/usr/bin/env python3
"""
Theorem Validation Framework

Systematically tests physics hypotheses and explores feature combinations
to find high-probability trigger conditions.

Theorems to validate:
1. Underdamped regimes have more energy release potential
2. High damping precedes energy release
3. Energy builds before release (positive velocity)
4. Regime transitions precede releases

Exploration approach:
- Test all single-feature conditions
- Test all pairwise combinations
- Test triple combinations with best pairs
- Rank by lift and hit rate

Usage:
    python scripts/validate_theorems.py --symbol BTCUSD
"""

import argparse
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from itertools import combinations
from typing import List, Dict, Tuple
from dataclasses import dataclass

sys.path.insert(0, str(Path(__file__).parent.parent))

from kinetra import PhysicsEngine, load_csv_data


@dataclass
class TheoremResult:
    """Result of testing a theorem/hypothesis."""
    name: str
    hypothesis: str
    condition_count: int
    target_count: int
    overlap_count: int
    hit_rate: float
    base_rate: float
    lift: float
    p_value: float = None  # Optional statistical significance


def compute_features(data: pd.DataFrame, lookback: int = 20) -> pd.DataFrame:
    """Compute all physics features for analysis."""
    engine = PhysicsEngine(lookback=lookback)
    physics = engine.compute_physics_state(data['close'])

    df = data.copy()
    df['energy'] = physics['energy']
    df['damping'] = physics['damping']
    df['entropy'] = physics['entropy']
    df['regime'] = physics['regime']

    # Derived features
    df['energy_pct'] = df['energy'].rank(pct=True)
    df['damping_pct'] = df['damping'].rank(pct=True)
    df['entropy_pct'] = df['entropy'].rank(pct=True)

    # Dynamics
    df['energy_velocity'] = df['energy'].diff()
    df['energy_accel'] = df['energy_velocity'].diff()
    df['damping_velocity'] = df['damping'].diff()

    # Rolling stats
    df['energy_ma5'] = df['energy'].rolling(5).mean()
    df['energy_ma20'] = df['energy'].rolling(20).mean()
    df['energy_std5'] = df['energy'].rolling(5).std()

    # Regime encoding
    df['is_underdamped'] = df['regime'] == 'underdamped'
    df['is_critical'] = df['regime'] == 'critical'
    df['is_overdamped'] = df['regime'] == 'overdamped'

    # Lagged features
    df['regime_lag1'] = df['regime'].shift(1)
    df['regime_lag2'] = df['regime'].shift(2)
    df['energy_lag1'] = df['energy'].shift(1)

    # Target: next bar is high energy (top 20%)
    threshold_80 = df['energy'].quantile(0.80)
    df['is_high_energy'] = df['energy'] >= threshold_80
    df['next_is_high'] = df['is_high_energy'].shift(-1).fillna(False)

    return df.dropna()


def test_theorem(
    df: pd.DataFrame,
    name: str,
    hypothesis: str,
    condition: pd.Series
) -> TheoremResult:
    """Test a single theorem/hypothesis."""
    condition_count = condition.sum()
    target_count = df['next_is_high'].sum()
    overlap_count = (condition & df['next_is_high']).sum()

    hit_rate = overlap_count / condition_count if condition_count > 0 else 0
    base_rate = df['next_is_high'].mean()
    lift = hit_rate / base_rate if base_rate > 0 else 0

    return TheoremResult(
        name=name,
        hypothesis=hypothesis,
        condition_count=int(condition_count),
        target_count=int(target_count),
        overlap_count=int(overlap_count),
        hit_rate=float(hit_rate),
        base_rate=float(base_rate),
        lift=float(lift),
    )


def validate_core_theorems(df: pd.DataFrame) -> List[TheoremResult]:
    """Validate core physics theorems."""
    results = []

    # Theorem 1: Underdamped has more energy release
    results.append(test_theorem(
        df,
        "T1: Underdamped Release",
        "Underdamped regime has higher energy release probability",
        df['is_underdamped']
    ))

    # Theorem 2: Critical regime is transition zone
    results.append(test_theorem(
        df,
        "T2: Critical Transition",
        "Critical regime precedes energy release",
        df['is_critical']
    ))

    # Theorem 3: Overdamped stores energy
    results.append(test_theorem(
        df,
        "T3: Overdamped Storage",
        "Overdamped regime precedes release (energy stored)",
        df['is_overdamped']
    ))

    # Theorem 4: High damping precedes release
    results.append(test_theorem(
        df,
        "T4: High Damping",
        "High damping (>70th pct) precedes release",
        df['damping_pct'] > 0.70
    ))

    # Theorem 5: Rising energy precedes release
    results.append(test_theorem(
        df,
        "T5: Rising Energy",
        "Positive energy velocity precedes release",
        df['energy_velocity'] > 0
    ))

    # Theorem 6: Energy acceleration
    results.append(test_theorem(
        df,
        "T6: Energy Accelerating",
        "Positive energy acceleration precedes release",
        df['energy_accel'] > 0
    ))

    # Theorem 7: Energy above MA
    results.append(test_theorem(
        df,
        "T7: Energy > MA20",
        "Energy above 20-bar MA precedes release",
        df['energy'] > df['energy_ma20']
    ))

    # Theorem 8: Regime transition from overdamped
    results.append(test_theorem(
        df,
        "T8: From Overdamped",
        "Transition from overdamped precedes release",
        (df['regime_lag1'] == 'overdamped') & (df['regime'] != 'overdamped')
    ))

    # Theorem 9: Regime transition to underdamped
    results.append(test_theorem(
        df,
        "T9: To Underdamped",
        "Transition to underdamped precedes release",
        (df['regime_lag1'] != 'underdamped') & (df['regime'] == 'underdamped')
    ))

    # Theorem 10: High entropy (disorder/uncertainty)
    results.append(test_theorem(
        df,
        "T10: High Entropy",
        "High entropy (>70th pct) precedes release",
        df['entropy_pct'] > 0.70
    ))

    return results


def explore_feature_combinations(df: pd.DataFrame, top_n: int = 20) -> List[Dict]:
    """
    Systematically explore feature combinations.
    Tests single features, pairs, and triples.
    """
    # Define condition generators
    conditions = {
        # Regime conditions
        'underdamped': df['is_underdamped'],
        'critical': df['is_critical'],
        'overdamped': df['is_overdamped'],

        # Percentile conditions
        'energy>p50': df['energy_pct'] > 0.50,
        'energy>p70': df['energy_pct'] > 0.70,
        'energy>p90': df['energy_pct'] > 0.90,
        'damping>p50': df['damping_pct'] > 0.50,
        'damping>p70': df['damping_pct'] > 0.70,
        'entropy>p50': df['entropy_pct'] > 0.50,
        'entropy>p70': df['entropy_pct'] > 0.70,

        # Dynamics
        'energy_rising': df['energy_velocity'] > 0,
        'energy_falling': df['energy_velocity'] < 0,
        'energy_accel+': df['energy_accel'] > 0,
        'energy_accel-': df['energy_accel'] < 0,
        'damping_rising': df['damping_velocity'] > 0,

        # MA comparisons
        'energy>ma5': df['energy'] > df['energy_ma5'],
        'energy>ma20': df['energy'] > df['energy_ma20'],
        'ma5>ma20': df['energy_ma5'] > df['energy_ma20'],

        # Regime transitions
        'was_overdamped': df['regime_lag1'] == 'overdamped',
        'was_critical': df['regime_lag1'] == 'critical',
        'was_underdamped': df['regime_lag1'] == 'underdamped',
    }

    results = []
    base_rate = df['next_is_high'].mean()

    # Test all single conditions
    for name, cond in conditions.items():
        count = cond.sum()
        if count < 10:  # Skip if too few samples
            continue

        hits = (cond & df['next_is_high']).sum()
        hit_rate = hits / count if count > 0 else 0
        lift = hit_rate / base_rate if base_rate > 0 else 0

        results.append({
            'conditions': name,
            'n_conditions': 1,
            'signals': count,
            'hits': hits,
            'hit_rate': hit_rate * 100,
            'lift': lift,
        })

    # Test all pairs
    condition_names = list(conditions.keys())
    for c1, c2 in combinations(condition_names, 2):
        combined = conditions[c1] & conditions[c2]
        count = combined.sum()
        if count < 10:
            continue

        hits = (combined & df['next_is_high']).sum()
        hit_rate = hits / count if count > 0 else 0
        lift = hit_rate / base_rate if base_rate > 0 else 0

        results.append({
            'conditions': f"{c1} + {c2}",
            'n_conditions': 2,
            'signals': count,
            'hits': hits,
            'hit_rate': hit_rate * 100,
            'lift': lift,
        })

    # Test triples with top pairs
    results_df = pd.DataFrame(results)
    top_pairs = results_df[results_df['n_conditions'] == 2].nlargest(10, 'lift')

    for _, row in top_pairs.iterrows():
        pair_conds = row['conditions'].split(' + ')
        combined_pair = conditions[pair_conds[0]] & conditions[pair_conds[1]]

        for c3 in condition_names:
            if c3 in pair_conds:
                continue

            triple = combined_pair & conditions[c3]
            count = triple.sum()
            if count < 10:
                continue

            hits = (triple & df['next_is_high']).sum()
            hit_rate = hits / count if count > 0 else 0
            lift = hit_rate / base_rate if base_rate > 0 else 0

            results.append({
                'conditions': f"{row['conditions']} + {c3}",
                'n_conditions': 3,
                'signals': count,
                'hits': hits,
                'hit_rate': hit_rate * 100,
                'lift': lift,
            })

    # Sort by lift and return top N
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('lift', ascending=False).head(top_n)

    return results_df.to_dict('records')


def compute_regime_energy_stats(df: pd.DataFrame) -> Dict:
    """Compute energy statistics by regime."""
    stats = {}

    for regime in ['underdamped', 'critical', 'overdamped']:
        mask = df['regime'] == regime
        regime_df = df[mask]

        if len(regime_df) == 0:
            continue

        # Energy stats
        energy_mean = regime_df['energy'].mean()
        energy_std = regime_df['energy'].std()
        energy_max = regime_df['energy'].max()

        # Release probability
        release_prob = regime_df['next_is_high'].mean()

        # Velocity stats
        velocity_mean = regime_df['energy_velocity'].mean()

        stats[regime] = {
            'count': len(regime_df),
            'pct_of_total': len(regime_df) / len(df) * 100,
            'energy_mean': energy_mean,
            'energy_std': energy_std,
            'energy_max': energy_max,
            'release_prob': release_prob * 100,
            'velocity_mean': velocity_mean,
        }

    return stats


def main():
    parser = argparse.ArgumentParser(description='Validate physics theorems and explore combinations')
    parser.add_argument('--data', type=str, help='Path to CSV data file')
    parser.add_argument('--symbol', type=str, default='BTCUSD', help='Symbol name')
    parser.add_argument('--lookback', type=int, default=20, help='Physics lookback')
    parser.add_argument('--top-n', type=int, default=30, help='Top N combinations to show')

    args = parser.parse_args()

    # Find data
    if args.data:
        data_path = Path(args.data)
    else:
        project_root = Path(__file__).parent.parent
        csv_files = list(project_root.glob(f"*{args.symbol}*.csv"))
        if not csv_files:
            print(f"No CSV file found for {args.symbol}")
            return
        data_path = csv_files[0]
        print(f"Using: {data_path.name}")

    # Load and compute
    print(f"\nLoading data...")
    data = load_csv_data(str(data_path))
    print(f"Loaded {len(data)} bars")

    print(f"\nComputing physics features (lookback={args.lookback})...")
    df = compute_features(data, args.lookback)

    # Validate core theorems
    print("\n" + "=" * 70)
    print("THEOREM VALIDATION")
    print("=" * 70)

    theorems = validate_core_theorems(df)
    base_rate = df['next_is_high'].mean()

    print(f"\nBase rate (random): {base_rate*100:.1f}% chance of high-energy next bar")
    print(f"\n{'Theorem':<25} {'Signals':>8} {'Hit%':>7} {'Lift':>6} {'Status':>12}")
    print("-" * 70)

    for t in sorted(theorems, key=lambda x: -x.lift):
        status = "CONFIRMED" if t.lift > 1.1 else "WEAK" if t.lift > 1.0 else "REJECTED"
        print(f"{t.name:<25} {t.condition_count:>8,} {t.hit_rate*100:>6.1f}% {t.lift:>5.2f}x {status:>12}")

    # Regime energy statistics
    print("\n" + "=" * 70)
    print("REGIME ENERGY ANALYSIS")
    print("=" * 70)

    regime_stats = compute_regime_energy_stats(df)
    overall_energy = df['energy'].mean()

    print(f"\n{'Regime':<12} {'Count':>8} {'%Total':>7} {'Avg Energy':>12} {'vs Avg':>8} {'Release%':>9}")
    print("-" * 70)

    for regime, stats in sorted(regime_stats.items(), key=lambda x: -x[1]['energy_mean']):
        ratio = stats['energy_mean'] / overall_energy
        print(f"{regime:<12} {stats['count']:>8,} {stats['pct_of_total']:>6.1f}% "
              f"{stats['energy_mean']:>12,.0f} {ratio:>7.2f}x {stats['release_prob']:>8.1f}%")

    # Systematic exploration
    print("\n" + "=" * 70)
    print(f"SYSTEMATIC EXPLORATION (Top {args.top_n} Combinations)")
    print("=" * 70)

    combinations_results = explore_feature_combinations(df, args.top_n)

    print(f"\n{'Conditions':<50} {'Signals':>8} {'Hit%':>7} {'Lift':>6}")
    print("-" * 75)

    for r in combinations_results:
        cond_str = r['conditions'][:48] + '..' if len(r['conditions']) > 50 else r['conditions']
        print(f"{cond_str:<50} {r['signals']:>8,} {r['hit_rate']:>6.1f}% {r['lift']:>5.2f}x")

    # Best trigger summary
    if combinations_results:
        best = combinations_results[0]
        print("\n" + "=" * 70)
        print("BEST DISCOVERED TRIGGER")
        print("=" * 70)
        print(f"\n  Conditions: {best['conditions']}")
        print(f"  Signals:    {best['signals']:,}")
        print(f"  Hit Rate:   {best['hit_rate']:.1f}%")
        print(f"  Lift:       {best['lift']:.2f}x vs random")
        print(f"  Base Rate:  {base_rate*100:.1f}%")

    print("\n" + "=" * 70)
    print("CONCLUSION: Use RL to learn adaptive composite of these conditions")
    print("=" * 70)


if __name__ == "__main__":
    main()
