#!/usr/bin/env python3
"""
Trigger Analysis Script

Analyzes what happens BEFORE high-energy bars to find reliable entry signals.
Goal: Identify the point of release with high probability.

Looks at:
- Energy buildup patterns (lagged energy)
- Regime transitions before releases
- Damping/entropy patterns
- Momentum divergence

Usage:
    python scripts/analyze_triggers.py --symbol BTCUSD --lookback 20
"""

import argparse
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
from collections import Counter

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from kinetra import PhysicsEngine, load_csv_data


def compute_physics_features(data: pd.DataFrame, lookback: int = 20) -> pd.DataFrame:
    """Compute all physics features for analysis."""
    engine = PhysicsEngine(lookback=lookback)
    physics = engine.compute_physics_state(data['close'])

    df = data.copy()
    df['energy'] = physics['energy']
    df['damping'] = physics['damping']
    df['entropy'] = physics['entropy']
    df['regime'] = physics['regime']

    # Momentum
    df['momentum'] = data['close'].diff(lookback)
    df['momentum_pct'] = data['close'].pct_change(lookback) * 100

    # Direction-specific energy
    df['energy_long'] = np.where(df['momentum'] > 0, df['energy'], 0)
    df['energy_short'] = np.where(df['momentum'] < 0, df['energy'], 0)

    # Lagged features (what happened before)
    for lag in [1, 2, 3, 5, 10]:
        df[f'energy_lag{lag}'] = df['energy'].shift(lag)
        df[f'damping_lag{lag}'] = df['damping'].shift(lag)
        df[f'regime_lag{lag}'] = df['regime'].shift(lag)

    # Energy change (buildup detection)
    df['energy_change_1'] = df['energy'] - df['energy'].shift(1)
    df['energy_change_3'] = df['energy'] - df['energy'].shift(3)
    df['energy_change_5'] = df['energy'] - df['energy'].shift(5)

    # Rolling energy stats (buildup patterns)
    df['energy_ma5'] = df['energy'].rolling(5).mean()
    df['energy_ma10'] = df['energy'].rolling(10).mean()
    df['energy_std5'] = df['energy'].rolling(5).std()

    # Energy acceleration
    df['energy_accel'] = df['energy_change_1'] - df['energy_change_1'].shift(1)

    return df.dropna()


def identify_high_energy_bars(df: pd.DataFrame, percentile: float = 80) -> pd.DataFrame:
    """Identify top energy bars (the targets we want to predict)."""
    threshold = np.percentile(df['energy'], percentile)
    high_energy = df[df['energy'] >= threshold].copy()
    high_energy['is_high_energy'] = True
    return high_energy


def analyze_pre_release_patterns(
    df: pd.DataFrame,
    high_energy_indices: pd.Index,
    lookback_bars: int = 5
) -> dict:
    """
    Analyze what happens in the N bars BEFORE high-energy releases.

    Returns patterns that could serve as triggers.
    """
    results = {
        'regime_before': [],
        'regime_sequence': [],
        'energy_buildup': [],
        'damping_before': [],
        'entropy_before': [],
        'energy_change_before': [],
    }

    for idx in high_energy_indices:
        # Get position in dataframe
        try:
            pos = df.index.get_loc(idx)
        except KeyError:
            continue

        if pos < lookback_bars:
            continue

        # Get the bars before this high-energy bar
        pre_bars = df.iloc[pos - lookback_bars:pos]

        # Regime before release
        results['regime_before'].append(pre_bars['regime'].iloc[-1])  # Immediately before
        results['regime_sequence'].append(tuple(pre_bars['regime'].tolist()))

        # Energy buildup pattern
        energy_trend = pre_bars['energy'].tolist()
        results['energy_buildup'].append(energy_trend)

        # Average damping before
        results['damping_before'].append(pre_bars['damping'].mean())

        # Average entropy before
        results['entropy_before'].append(pre_bars['entropy'].mean())

        # Energy change leading up
        if 'energy_change_3' in pre_bars.columns:
            results['energy_change_before'].append(pre_bars['energy_change_3'].iloc[-1])

    return results


def compute_trigger_statistics(patterns: dict, df: pd.DataFrame) -> dict:
    """Compute statistics about pre-release patterns to find triggers."""

    stats = {}

    # Regime distribution before high-energy bars
    if patterns['regime_before']:
        regime_counts = Counter(patterns['regime_before'])
        total = sum(regime_counts.values())
        stats['regime_before_distribution'] = {
            k: v / total * 100 for k, v in regime_counts.items()
        }

        # Compare to overall regime distribution
        overall_regime = df['regime'].value_counts(normalize=True) * 100
        stats['regime_overall'] = overall_regime.to_dict()

    # Regime sequence patterns (most common sequences before release)
    if patterns['regime_sequence']:
        seq_counts = Counter(patterns['regime_sequence'])
        top_sequences = seq_counts.most_common(10)
        stats['top_regime_sequences'] = [
            {'sequence': list(seq), 'count': count, 'pct': count / len(patterns['regime_sequence']) * 100}
            for seq, count in top_sequences
        ]

    # Energy buildup patterns
    if patterns['energy_buildup']:
        buildups = np.array(patterns['energy_buildup'])

        # Average energy trajectory before release
        avg_trajectory = np.mean(buildups, axis=0)
        stats['avg_energy_trajectory'] = avg_trajectory.tolist()

        # Was energy rising or falling before release?
        rising_count = sum(1 for b in buildups if b[-1] > b[0])
        stats['energy_rising_before_pct'] = rising_count / len(buildups) * 100

        # Energy acceleration pattern
        accelerating = sum(1 for b in buildups if len(b) >= 3 and (b[-1] - b[-2]) > (b[-2] - b[-3]))
        stats['energy_accelerating_pct'] = accelerating / len(buildups) * 100

    # Damping before release
    if patterns['damping_before']:
        damping_arr = np.array(patterns['damping_before'])
        overall_damping = df['damping'].mean()
        stats['damping_before_mean'] = float(np.mean(damping_arr))
        stats['damping_overall_mean'] = float(overall_damping)
        stats['damping_elevated_before'] = stats['damping_before_mean'] > overall_damping

    # Entropy before release
    if patterns['entropy_before']:
        entropy_arr = np.array(patterns['entropy_before'])
        overall_entropy = df['entropy'].mean()
        stats['entropy_before_mean'] = float(np.mean(entropy_arr))
        stats['entropy_overall_mean'] = float(overall_entropy)
        stats['entropy_elevated_before'] = stats['entropy_before_mean'] > overall_entropy

    # Energy change before release
    if patterns['energy_change_before']:
        changes = np.array([c for c in patterns['energy_change_before'] if not np.isnan(c)])
        if len(changes) > 0:
            stats['energy_change_before_mean'] = float(np.mean(changes))
            stats['energy_change_positive_pct'] = float(np.sum(changes > 0) / len(changes) * 100)

    return stats


def find_trigger_conditions(df: pd.DataFrame, stats: dict) -> list:
    """
    Based on patterns, define specific trigger conditions.

    Returns list of trigger rules with their hit rates.
    """
    triggers = []

    # Mark high energy bars
    threshold_80 = np.percentile(df['energy'], 80)
    df['is_high_energy'] = df['energy'] >= threshold_80
    df['next_is_high'] = df['is_high_energy'].shift(-1)  # What we want to predict

    # Trigger 1: Regime-based
    # If overdamped regime often precedes releases, test it
    if 'regime_before_distribution' in stats:
        best_regime = max(stats['regime_before_distribution'], key=stats['regime_before_distribution'].get)
        regime_mask = df['regime'] == best_regime
        hit_rate = df.loc[regime_mask, 'next_is_high'].mean() * 100 if regime_mask.sum() > 0 else 0
        base_rate = df['is_high_energy'].mean() * 100

        triggers.append({
            'name': f'Regime = {best_regime}',
            'condition': f"regime == '{best_regime}'",
            'signals': int(regime_mask.sum()),
            'hits': int(df.loc[regime_mask, 'next_is_high'].sum()),
            'hit_rate': hit_rate,
            'base_rate': base_rate,
            'lift': hit_rate / base_rate if base_rate > 0 else 0,
        })

    # Trigger 2: Energy rising
    energy_rising = df['energy_change_3'] > 0
    if energy_rising.sum() > 0:
        hit_rate = df.loc[energy_rising, 'next_is_high'].mean() * 100
        base_rate = df['is_high_energy'].mean() * 100
        triggers.append({
            'name': 'Energy Rising (3-bar)',
            'condition': 'energy_change_3 > 0',
            'signals': int(energy_rising.sum()),
            'hits': int(df.loc[energy_rising, 'next_is_high'].sum()),
            'hit_rate': hit_rate,
            'base_rate': base_rate,
            'lift': hit_rate / base_rate if base_rate > 0 else 0,
        })

    # Trigger 3: Energy above MA
    energy_above_ma = df['energy'] > df['energy_ma10']
    if energy_above_ma.sum() > 0:
        hit_rate = df.loc[energy_above_ma, 'next_is_high'].mean() * 100
        base_rate = df['is_high_energy'].mean() * 100
        triggers.append({
            'name': 'Energy > MA10',
            'condition': 'energy > energy_ma10',
            'signals': int(energy_above_ma.sum()),
            'hits': int(df.loc[energy_above_ma, 'next_is_high'].sum()),
            'hit_rate': hit_rate,
            'base_rate': base_rate,
            'lift': hit_rate / base_rate if base_rate > 0 else 0,
        })

    # Trigger 4: Energy accelerating
    energy_accel = df['energy_accel'] > 0
    if energy_accel.sum() > 0:
        hit_rate = df.loc[energy_accel, 'next_is_high'].mean() * 100
        base_rate = df['is_high_energy'].mean() * 100
        triggers.append({
            'name': 'Energy Accelerating',
            'condition': 'energy_accel > 0',
            'signals': int(energy_accel.sum()),
            'hits': int(df.loc[energy_accel, 'next_is_high'].sum()),
            'hit_rate': hit_rate,
            'base_rate': base_rate,
            'lift': hit_rate / base_rate if base_rate > 0 else 0,
        })

    # Trigger 5: Damping high (energy stored)
    damping_high = df['damping'] > df['damping'].quantile(0.7)
    if damping_high.sum() > 0:
        hit_rate = df.loc[damping_high, 'next_is_high'].mean() * 100
        base_rate = df['is_high_energy'].mean() * 100
        triggers.append({
            'name': 'Damping > 70th percentile',
            'condition': 'damping > Q70',
            'signals': int(damping_high.sum()),
            'hits': int(df.loc[damping_high, 'next_is_high'].sum()),
            'hit_rate': hit_rate,
            'base_rate': base_rate,
            'lift': hit_rate / base_rate if base_rate > 0 else 0,
        })

    # Trigger 6: Combined - Rising energy + high damping
    combined = energy_rising & damping_high
    if combined.sum() > 0:
        hit_rate = df.loc[combined, 'next_is_high'].mean() * 100
        base_rate = df['is_high_energy'].mean() * 100
        triggers.append({
            'name': 'Rising Energy + High Damping',
            'condition': 'energy_change_3 > 0 AND damping > Q70',
            'signals': int(combined.sum()),
            'hits': int(df.loc[combined, 'next_is_high'].sum()),
            'hit_rate': hit_rate,
            'base_rate': base_rate,
            'lift': hit_rate / base_rate if base_rate > 0 else 0,
        })

    # Trigger 7: Energy spike detection (std deviation breakout)
    energy_spike = df['energy'] > (df['energy_ma5'] + 2 * df['energy_std5'])
    if energy_spike.sum() > 0:
        hit_rate = df.loc[energy_spike, 'next_is_high'].mean() * 100
        base_rate = df['is_high_energy'].mean() * 100
        triggers.append({
            'name': 'Energy Spike (>2 std)',
            'condition': 'energy > ma5 + 2*std5',
            'signals': int(energy_spike.sum()),
            'hits': int(df.loc[energy_spike, 'next_is_high'].sum()),
            'hit_rate': hit_rate,
            'base_rate': base_rate,
            'lift': hit_rate / base_rate if base_rate > 0 else 0,
        })

    # Trigger 8: Regime transition from critical
    regime_was_critical = df['regime_lag1'] == 'critical'
    if regime_was_critical.sum() > 0:
        hit_rate = df.loc[regime_was_critical, 'next_is_high'].mean() * 100
        base_rate = df['is_high_energy'].mean() * 100
        triggers.append({
            'name': 'Was Critical (lag 1)',
            'condition': 'regime_lag1 == critical',
            'signals': int(regime_was_critical.sum()),
            'hits': int(df.loc[regime_was_critical, 'next_is_high'].sum()),
            'hit_rate': hit_rate,
            'base_rate': base_rate,
            'lift': hit_rate / base_rate if base_rate > 0 else 0,
        })

    # Trigger 9: Triple combo - Rising + High Damping + Above MA
    triple = energy_rising & damping_high & energy_above_ma
    if triple.sum() > 0:
        hit_rate = df.loc[triple, 'next_is_high'].mean() * 100
        base_rate = df['is_high_energy'].mean() * 100
        triggers.append({
            'name': 'Triple: Rising+Damping+MA',
            'condition': 'rising AND damping>Q70 AND energy>MA10',
            'signals': int(triple.sum()),
            'hits': int(df.loc[triple, 'next_is_high'].sum()),
            'hit_rate': hit_rate,
            'base_rate': base_rate,
            'lift': hit_rate / base_rate if base_rate > 0 else 0,
        })

    # Trigger 10: Energy in top 10% of recent (rolling)
    energy_top10_rolling = df['energy'] > df['energy'].rolling(50).quantile(0.9)
    if energy_top10_rolling.sum() > 0:
        hit_rate = df.loc[energy_top10_rolling, 'next_is_high'].mean() * 100
        base_rate = df['is_high_energy'].mean() * 100
        triggers.append({
            'name': 'Energy Top 10% (rolling 50)',
            'condition': 'energy > rolling_Q90',
            'signals': int(energy_top10_rolling.sum()),
            'hits': int(df.loc[energy_top10_rolling, 'next_is_high'].sum()),
            'hit_rate': hit_rate,
            'base_rate': base_rate,
            'lift': hit_rate / base_rate if base_rate > 0 else 0,
        })

    # Trigger 11: Energy spike + was critical
    spike_from_critical = energy_spike & regime_was_critical
    if spike_from_critical.sum() > 0:
        hit_rate = df.loc[spike_from_critical, 'next_is_high'].mean() * 100
        base_rate = df['is_high_energy'].mean() * 100
        triggers.append({
            'name': 'Spike from Critical',
            'condition': 'energy_spike AND was_critical',
            'signals': int(spike_from_critical.sum()),
            'hits': int(df.loc[spike_from_critical, 'next_is_high'].sum()),
            'hit_rate': hit_rate,
            'base_rate': base_rate,
            'lift': hit_rate / base_rate if base_rate > 0 else 0,
        })

    # Trigger 12: Very high energy (top 5%)
    energy_top5 = df['energy'] > df['energy'].quantile(0.95)
    if energy_top5.sum() > 0:
        hit_rate = df.loc[energy_top5, 'next_is_high'].mean() * 100
        base_rate = df['is_high_energy'].mean() * 100
        triggers.append({
            'name': 'Energy > 95th percentile',
            'condition': 'energy > Q95',
            'signals': int(energy_top5.sum()),
            'hits': int(df.loc[energy_top5, 'next_is_high'].sum()),
            'hit_rate': hit_rate,
            'base_rate': base_rate,
            'lift': hit_rate / base_rate if base_rate > 0 else 0,
        })

    # Trigger 13: Underdamped regime (rare but volatile)
    underdamped = df['regime'] == 'underdamped'
    if underdamped.sum() > 0:
        hit_rate = df.loc[underdamped, 'next_is_high'].mean() * 100
        base_rate = df['is_high_energy'].mean() * 100
        triggers.append({
            'name': 'Underdamped Regime',
            'condition': 'regime == underdamped',
            'signals': int(underdamped.sum()),
            'hits': int(df.loc[underdamped, 'next_is_high'].sum()),
            'hit_rate': hit_rate,
            'base_rate': base_rate,
            'lift': hit_rate / base_rate if base_rate > 0 else 0,
        })

    # Trigger 14: Consecutive energy rises
    consecutive_rise = (df['energy_change_1'] > 0) & (df['energy_change_1'].shift(1) > 0) & (df['energy_change_1'].shift(2) > 0)
    if consecutive_rise.sum() > 0:
        hit_rate = df.loc[consecutive_rise, 'next_is_high'].mean() * 100
        base_rate = df['is_high_energy'].mean() * 100
        triggers.append({
            'name': '3 Consecutive Energy Rises',
            'condition': '3 bars of rising energy',
            'signals': int(consecutive_rise.sum()),
            'hits': int(df.loc[consecutive_rise, 'next_is_high'].sum()),
            'hit_rate': hit_rate,
            'base_rate': base_rate,
            'lift': hit_rate / base_rate if base_rate > 0 else 0,
        })

    # Trigger 15: Quad combo - best conditions
    quad = energy_rising & damping_high & energy_above_ma & (df['energy_accel'] > 0)
    if quad.sum() > 0:
        hit_rate = df.loc[quad, 'next_is_high'].mean() * 100
        base_rate = df['is_high_energy'].mean() * 100
        triggers.append({
            'name': 'Quad: Rise+Damp+MA+Accel',
            'condition': 'all four conditions',
            'signals': int(quad.sum()),
            'hits': int(df.loc[quad, 'next_is_high'].sum()),
            'hit_rate': hit_rate,
            'base_rate': base_rate,
            'lift': hit_rate / base_rate if base_rate > 0 else 0,
        })

    return sorted(triggers, key=lambda x: x['lift'], reverse=True)


def main():
    parser = argparse.ArgumentParser(description='Analyze trigger patterns before energy releases')
    parser.add_argument('--data', type=str, help='Path to CSV data file')
    parser.add_argument('--symbol', type=str, default='BTCUSD', help='Symbol name')
    parser.add_argument('--lookback', type=int, default=20, help='Lookback for energy calculation')
    parser.add_argument('--pre-bars', type=int, default=5, help='Bars to analyze before release')
    parser.add_argument('--percentile', type=float, default=80, help='Percentile threshold for high energy')

    args = parser.parse_args()

    # Find data file
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

    # Load and process data
    print(f"\nLoading data from {data_path}...")
    data = load_csv_data(str(data_path))
    print(f"Loaded {len(data)} bars")

    print(f"\nComputing physics features (lookback={args.lookback})...")
    df = compute_physics_features(data, lookback=args.lookback)

    # Identify high energy bars
    print(f"\nIdentifying top {100 - args.percentile:.0f}% energy bars...")
    high_energy = identify_high_energy_bars(df, args.percentile)
    print(f"Found {len(high_energy)} high-energy bars ({len(high_energy)/len(df)*100:.1f}% of total)")

    # Analyze pre-release patterns
    print(f"\nAnalyzing {args.pre_bars} bars before each release...")
    patterns = analyze_pre_release_patterns(df, high_energy.index, args.pre_bars)

    # Compute statistics
    stats = compute_trigger_statistics(patterns, df)

    # Find trigger conditions
    triggers = find_trigger_conditions(df, stats)

    # Print results
    print("\n" + "=" * 70)
    print("TRIGGER ANALYSIS: Pre-Release Patterns")
    print("=" * 70)

    print(f"\nRegime distribution BEFORE high-energy releases:")
    if 'regime_before_distribution' in stats:
        for regime, pct in sorted(stats['regime_before_distribution'].items(), key=lambda x: -x[1]):
            overall = stats['regime_overall'].get(regime, 0)
            diff = pct - overall
            print(f"  {regime:12s}: {pct:5.1f}% (overall: {overall:5.1f}%, diff: {diff:+5.1f}%)")

    print(f"\nEnergy buildup patterns:")
    if 'energy_rising_before_pct' in stats:
        print(f"  Energy rising before release: {stats['energy_rising_before_pct']:.1f}%")
    if 'energy_accelerating_pct' in stats:
        print(f"  Energy accelerating before:   {stats['energy_accelerating_pct']:.1f}%")
    if 'energy_change_positive_pct' in stats:
        print(f"  Positive energy change:       {stats['energy_change_positive_pct']:.1f}%")

    print(f"\nDamping/Entropy before release:")
    if 'damping_before_mean' in stats:
        elevated = "ELEVATED" if stats['damping_elevated_before'] else "NORMAL"
        print(f"  Damping:  {stats['damping_before_mean']:.4f} vs {stats['damping_overall_mean']:.4f} overall ({elevated})")
    if 'entropy_before_mean' in stats:
        elevated = "ELEVATED" if stats['entropy_elevated_before'] else "NORMAL"
        print(f"  Entropy:  {stats['entropy_before_mean']:.4f} vs {stats['entropy_overall_mean']:.4f} overall ({elevated})")

    print("\n" + "=" * 70)
    print("TRIGGER CONDITIONS (sorted by lift)")
    print("=" * 70)
    print(f"\n{'Trigger':<35} {'Signals':>8} {'Hits':>6} {'Hit%':>7} {'Base%':>7} {'Lift':>6}")
    print("-" * 70)

    for t in triggers:
        print(f"{t['name']:<35} {t['signals']:>8,} {t['hits']:>6,} {t['hit_rate']:>6.1f}% {t['base_rate']:>6.1f}% {t['lift']:>5.2f}x")

    # Find best trigger
    if triggers:
        best = triggers[0]
        print(f"\n{'='*70}")
        print(f"BEST TRIGGER: {best['name']}")
        print(f"  Condition: {best['condition']}")
        print(f"  Hit Rate:  {best['hit_rate']:.1f}% (vs {best['base_rate']:.1f}% base)")
        print(f"  Lift:      {best['lift']:.2f}x better than random")
        print(f"  Signals:   {best['signals']:,} times in dataset")
        print(f"{'='*70}")

    # Show top regime sequences
    if 'top_regime_sequences' in stats:
        print(f"\nTop regime sequences before release (last {args.pre_bars} bars):")
        for i, seq in enumerate(stats['top_regime_sequences'][:5], 1):
            seq_str = ' -> '.join(seq['sequence'])
            print(f"  {i}. {seq_str} ({seq['pct']:.1f}%)")


if __name__ == "__main__":
    main()
