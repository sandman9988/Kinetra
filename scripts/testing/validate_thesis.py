#!/usr/bin/env python3
"""
Empirical Validation of Energy-Transfer Trading Thesis

Tests each claim one at a time against actual BTCUSD data.
Uses the actual PhysicsEngine for consistency.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from kinetra.mt5_connector import load_csv_data
from kinetra.physics_engine import PhysicsEngine


def compute_thesis_metrics(df: pd.DataFrame, lookback: int = 20) -> pd.DataFrame:
    """
    Compute metrics using the actual PhysicsEngine.
    """
    result = df.copy()

    # Use actual PhysicsEngine
    engine = PhysicsEngine(lookback=lookback)
    physics_state = engine.compute_physics_state(
        df['close'],
        volume=df['volume'],
        include_percentiles=True
    )

    # Merge physics state
    result['energy'] = physics_state['energy']
    result['damping'] = physics_state['damping']
    result['entropy'] = physics_state['entropy']
    result['energy_pct'] = physics_state['energy_pct']
    result['damping_pct'] = physics_state['damping_pct']
    result['entropy_pct'] = physics_state['entropy_pct']
    result['regime'] = physics_state['regime']

    # Body Ratio (conviction metric)
    epsilon = 1e-8
    result['body_ratio'] = abs(df['close'] - df['open']) / (df['high'] - df['low'] + epsilon)

    # Forward returns (what we're trying to predict)
    result['fwd_return_1'] = df['close'].pct_change().shift(-1)
    result['fwd_return_5'] = df['close'].pct_change(5).shift(-5)
    result['fwd_abs_return_1'] = result['fwd_return_1'].abs()
    result['fwd_abs_return_5'] = result['fwd_return_5'].abs()

    return result.dropna()


def test_energy_correlation(df: pd.DataFrame) -> dict:
    """
    Test: Does high energy correlate with high forward moves?
    """
    # Correlation between energy and forward absolute returns
    corr_1 = df['energy'].corr(df['fwd_abs_return_1'])
    corr_5 = df['energy'].corr(df['fwd_abs_return_5'])

    # Compare high energy vs low energy regimes
    high_energy = df['energy_pct'] > 0.75
    low_energy = df['energy_pct'] < 0.25

    avg_move_high = df.loc[high_energy, 'fwd_abs_return_1'].mean() * 100
    avg_move_low = df.loc[low_energy, 'fwd_abs_return_1'].mean() * 100

    return {
        'claim': 'High energy leads to larger price moves',
        'correlation_1bar': corr_1,
        'correlation_5bar': corr_5,
        'avg_move_high_energy': avg_move_high,
        'avg_move_low_energy': avg_move_low,
        'ratio': avg_move_high / avg_move_low if avg_move_low > 0 else 0,
        'verdict': 'SUPPORTED' if avg_move_high > avg_move_low * 1.1 else 'NOT SUPPORTED',
    }


def test_damping_regimes(df: pd.DataFrame) -> dict:
    """
    Test: Do different damping regimes have different move characteristics?

    Physics interpretation:
    - High damping = high friction = overdamped = ranges/consolidation
    - Low damping = low friction = underdamped = trending
    """
    underdamped = df['damping_pct'] < 0.25  # Low damping = underdamped
    overdamped = df['damping_pct'] > 0.75   # High damping = overdamped

    avg_move_under = df.loc[underdamped, 'fwd_abs_return_1'].mean() * 100
    avg_move_over = df.loc[overdamped, 'fwd_abs_return_1'].mean() * 100

    # Also check regime labels from physics engine
    underdamped_regime = df['regime'] == 'underdamped'
    overdamped_regime = df['regime'] == 'overdamped'

    regime_move_under = df.loc[underdamped_regime, 'fwd_abs_return_1'].mean() * 100
    regime_move_over = df.loc[overdamped_regime, 'fwd_abs_return_1'].mean() * 100

    return {
        'claim': 'Underdamped regimes have larger price moves',
        'avg_move_underdamped_pct': avg_move_under,
        'avg_move_overdamped_pct': avg_move_over,
        'pct_ratio': avg_move_under / avg_move_over if avg_move_over > 0 else 0,
        'regime_move_underdamped': regime_move_under,
        'regime_move_overdamped': regime_move_over,
        'regime_ratio': regime_move_under / regime_move_over if regime_move_over > 0 else 0,
        'verdict': 'SUPPORTED' if avg_move_under > avg_move_over * 1.1 else 'NOT SUPPORTED',
    }


def test_berserker_variations(df: pd.DataFrame) -> dict:
    """
    Test: Find the BEST berserker condition combination.

    Try multiple threshold combinations to find what actually works.
    """
    base_move = df['fwd_abs_return_1'].mean() * 100

    results = []

    # Test various energy thresholds
    for energy_thresh in [0.5, 0.6, 0.7, 0.8, 0.9]:
        for damping_thresh in [0.1, 0.2, 0.3, 0.4, 0.5]:
            condition = (df['energy_pct'] > energy_thresh) & (df['damping_pct'] < damping_thresh)
            count = condition.sum()
            if count > 50:  # Need sufficient samples
                move = df.loc[condition, 'fwd_abs_return_1'].mean() * 100
                lift = move / base_move
                results.append({
                    'energy_thresh': energy_thresh,
                    'damping_thresh': damping_thresh,
                    'signals': count,
                    'avg_move': move,
                    'lift': lift,
                })

    # Find best combination
    if results:
        best = max(results, key=lambda x: x['lift'])
    else:
        best = {'energy_thresh': 0, 'damping_thresh': 0, 'signals': 0, 'avg_move': 0, 'lift': 0}

    # Test original berserker condition
    original = (df['energy_pct'] > 0.75) & (df['damping_pct'] < 0.25)
    original_count = original.sum()
    original_move = df.loc[original, 'fwd_abs_return_1'].mean() * 100 if original_count > 0 else 0

    return {
        'claim': 'Berserker condition (high E, low damping) = explosive moves',
        'baseline_move': base_move,
        'original_signals': original_count,
        'original_move': original_move,
        'original_lift': original_move / base_move if base_move > 0 else 0,
        'best_energy_thresh': best['energy_thresh'],
        'best_damping_thresh': best['damping_thresh'],
        'best_signals': best['signals'],
        'best_move': best['avg_move'],
        'best_lift': best['lift'],
        'verdict': 'SUPPORTED' if best['lift'] > 1.3 else 'NEEDS REFINEMENT',
    }


def test_energy_release_patterns(df: pd.DataFrame) -> dict:
    """
    Test: Is there a pattern BEFORE energy release?

    Look for: energy building (rising) then releasing.
    """
    # Energy velocity (is energy increasing or decreasing?)
    df_test = df.copy()
    df_test['energy_vel'] = df_test['energy'].diff()
    df_test['energy_rising'] = df_test['energy_vel'] > 0
    df_test['energy_falling'] = df_test['energy_vel'] < 0

    # Was energy building in previous bars?
    df_test['energy_was_rising'] = df_test['energy_rising'].shift(1)

    # Energy release = high current energy + was rising + now starts falling
    potential_release = (
        (df_test['energy_pct'] > 0.7) &
        df_test['energy_was_rising'].fillna(False)
    )

    count = potential_release.sum()
    base_move = df_test['fwd_abs_return_1'].mean() * 100
    release_move = df_test.loc[potential_release, 'fwd_abs_return_1'].mean() * 100 if count > 0 else 0

    # Also test: energy just peaked (was rising, now falling)
    peaked = (
        (df_test['energy_pct'] > 0.7) &
        df_test['energy_was_rising'].fillna(False) &
        df_test['energy_falling']
    )
    peaked_count = peaked.sum()
    peaked_move = df_test.loc[peaked, 'fwd_abs_return_1'].mean() * 100 if peaked_count > 0 else 0

    return {
        'claim': 'Energy builds before release (actionable pattern)',
        'baseline_move': base_move,
        'energy_building_signals': count,
        'energy_building_move': release_move,
        'building_lift': release_move / base_move if base_move > 0 else 0,
        'energy_peaked_signals': peaked_count,
        'energy_peaked_move': peaked_move,
        'peaked_lift': peaked_move / base_move if base_move > 0 else 0,
        'verdict': 'SUPPORTED' if release_move > base_move * 1.2 else 'NOT SUPPORTED',
    }


def test_volume_energy_interaction(df: pd.DataFrame) -> dict:
    """
    Test: Does volume amplify energy signals?
    """
    # Volume percentile
    window = min(500, len(df))
    df_test = df.copy()
    df_test['volume_pct'] = df_test['volume'].rolling(window).apply(
        lambda x: (x.iloc[-1] > x.iloc[:-1]).mean() if len(x) > 1 else 0.5,
        raw=False
    ).fillna(0.5)

    # High energy alone
    high_energy = df_test['energy_pct'] > 0.75
    he_count = high_energy.sum()
    he_move = df_test.loc[high_energy, 'fwd_abs_return_1'].mean() * 100 if he_count > 0 else 0

    # High energy + high volume
    high_both = (df_test['energy_pct'] > 0.75) & (df_test['volume_pct'] > 0.75)
    hb_count = high_both.sum()
    hb_move = df_test.loc[high_both, 'fwd_abs_return_1'].mean() * 100 if hb_count > 0 else 0

    # High energy + low volume
    high_energy_low_vol = (df_test['energy_pct'] > 0.75) & (df_test['volume_pct'] < 0.25)
    helv_count = high_energy_low_vol.sum()
    helv_move = df_test.loc[high_energy_low_vol, 'fwd_abs_return_1'].mean() * 100 if helv_count > 0 else 0

    base_move = df_test['fwd_abs_return_1'].mean() * 100

    return {
        'claim': 'High volume amplifies energy signals',
        'baseline_move': base_move,
        'high_energy_only_signals': he_count,
        'high_energy_only_move': he_move,
        'high_energy_high_vol_signals': hb_count,
        'high_energy_high_vol_move': hb_move,
        'high_energy_low_vol_signals': helv_count,
        'high_energy_low_vol_move': helv_move,
        'volume_amplification': hb_move / he_move if he_move > 0 else 0,
        'verdict': 'SUPPORTED' if hb_move > he_move * 1.1 else 'NOT SUPPORTED',
    }


def test_entropy_filtering(df: pd.DataFrame) -> dict:
    """
    Test: Can low entropy improve berserker signals?
    """
    base_move = df['fwd_abs_return_1'].mean() * 100

    # Standard berserker
    berserker = (df['energy_pct'] > 0.75) & (df['damping_pct'] < 0.25)
    berk_count = berserker.sum()
    berk_move = df.loc[berserker, 'fwd_abs_return_1'].mean() * 100 if berk_count > 0 else 0

    # Berserker + low entropy (clean signal)
    berk_clean = berserker & (df['entropy_pct'] < 0.5)
    bc_count = berk_clean.sum()
    bc_move = df.loc[berk_clean, 'fwd_abs_return_1'].mean() * 100 if bc_count > 0 else 0

    # Berserker + high entropy (noisy signal)
    berk_noisy = berserker & (df['entropy_pct'] > 0.5)
    bn_count = berk_noisy.sum()
    bn_move = df.loc[berk_noisy, 'fwd_abs_return_1'].mean() * 100 if bn_count > 0 else 0

    return {
        'claim': 'Low entropy improves berserker signal quality',
        'baseline_move': base_move,
        'berserker_signals': berk_count,
        'berserker_move': berk_move,
        'clean_signals': bc_count,
        'clean_move': bc_move,
        'noisy_signals': bn_count,
        'noisy_move': bn_move,
        'clean_vs_noisy': bc_move / bn_move if bn_move > 0 else 0,
        'verdict': 'SUPPORTED' if bc_move > bn_move else 'NOT SUPPORTED',
    }


def test_body_ratio_trigger(df: pd.DataFrame) -> dict:
    """
    Test: Does body ratio add signal value?
    """
    high_body = df['body_ratio'] > df['body_ratio'].quantile(0.75)
    low_body = df['body_ratio'] < df['body_ratio'].quantile(0.25)

    # High body + high energy
    combo = high_body & (df['energy_pct'] > 0.75)
    combo_count = combo.sum()
    combo_move = df.loc[combo, 'fwd_abs_return_1'].mean() * 100 if combo_count > 0 else 0

    # High energy alone for comparison
    high_energy = df['energy_pct'] > 0.75
    he_move = df.loc[high_energy, 'fwd_abs_return_1'].mean() * 100

    base_move = df['fwd_abs_return_1'].mean() * 100

    return {
        'claim': 'Body ratio (conviction) adds signal value',
        'baseline_move': base_move,
        'high_energy_move': he_move,
        'high_body_high_energy_signals': combo_count,
        'high_body_high_energy_move': combo_move,
        'body_ratio_lift': combo_move / he_move if he_move > 0 else 0,
        'verdict': 'SUPPORTED' if combo_move > he_move * 1.1 else 'NOT SUPPORTED',
    }


def main():
    # Find data
    project_root = Path(__file__).parent.parent
    csv_files = list(project_root.glob("*BTCUSD*.csv"))
    if not csv_files:
        print("No BTCUSD CSV file found")
        return
    data_path = csv_files[0]
    print(f"Using: {data_path.name}")

    # Load and compute
    print("\nLoading data...")
    data = load_csv_data(str(data_path))
    print(f"Loaded {len(data)} bars")

    print("\nComputing thesis metrics (using PhysicsEngine)...")
    df = compute_thesis_metrics(data)
    print(f"Computed {len(df)} bars with all metrics")

    # Run tests
    print("\n" + "=" * 70)
    print("EMPIRICAL VALIDATION OF THESIS CLAIMS")
    print("=" * 70)

    tests = [
        test_energy_correlation,
        test_damping_regimes,
        test_berserker_variations,
        test_energy_release_patterns,
        test_volume_energy_interaction,
        test_entropy_filtering,
        test_body_ratio_trigger,
    ]

    results = []
    for test_func in tests:
        result = test_func(df)
        results.append(result)

        print(f"\n--- {result['claim']} ---")
        for key, value in result.items():
            if key not in ['claim', 'verdict']:
                if isinstance(value, float):
                    print(f"  {key}: {value:.4f}")
                else:
                    print(f"  {key}: {value}")
        print(f"  VERDICT: {result['verdict']}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    supported = sum(1 for r in results if r['verdict'] == 'SUPPORTED')
    needs_work = sum(1 for r in results if 'REFINEMENT' in r['verdict'])

    print(f"\n  Supported: {supported}/{len(results)}")
    print(f"  Needs Refinement: {needs_work}/{len(results)}")
    print(f"  Not Supported: {len(results) - supported - needs_work}/{len(results)}")

    # Actionable insights
    print("\n" + "=" * 70)
    print("ACTIONABLE INSIGHTS FOR BERSERKER ENTRY")
    print("=" * 70)

    # Find best berserker variation
    berk_result = next(r for r in results if 'berserker' in r['claim'].lower())
    print(f"\n  Best berserker thresholds:")
    print(f"    Energy > {berk_result.get('best_energy_thresh', 0.75)*100:.0f}th percentile")
    print(f"    Damping < {berk_result.get('best_damping_thresh', 0.25)*100:.0f}th percentile")
    print(f"    Expected lift: {berk_result.get('best_lift', 0):.2f}x")


if __name__ == "__main__":
    main()
