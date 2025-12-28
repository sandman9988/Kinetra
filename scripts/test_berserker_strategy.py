#!/usr/bin/env python3
"""
Test Berserker Strategy: Fat Candle Hunter with Energy Recovery Exit

Tests:
1. Fat candle prediction accuracy (magnitude)
2. Direction: Laminar=continuation, Turbulent=reversal
3. Energy recovery exit with dynamic MAE/MFE thresholds
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from kinetra.mt5_connector import load_csv_data
from kinetra.berserker_strategy import (
    PhysicsFeatures,
    CompositePredictor,
    EnergyRecoveryExit,
    HistoricalStats,
    HistoricalStatsTracker,
    FlowRegime,
    Direction,
    backtest_strategy,
)


def test_fat_candle_prediction(df: pd.DataFrame, feature_df: pd.DataFrame):
    """Test fat candle magnitude prediction."""
    predictor = CompositePredictor()

    predictions = []
    for i in range(100, len(feature_df) - 3):
        row = feature_df.iloc[i]

        pred = predictor.predict(
            energy_pct=row['energy_pct'],
            damping_pct=row['damping_pct'],
            entropy_pct=row['entropy_pct'],
            jerk_pct=row['jerk_pct'],
            impulse_pct=row['impulse_pct'],
            liquidity_pct=row['liquidity_pct'],
            reynolds_pct=row['reynolds_pct'],
            buying_pressure=row['buying_pressure'],
            momentum=row['momentum_5'],
            flow_consistency=row['flow_consistency'],
            inertia=int(row['inertia']),
        )

        # Forward absolute return (fat candle = big move either direction)
        fwd_abs_return = abs((df.iloc[i + 1]['close'] - df.iloc[i]['close']) / df.iloc[i]['close'])

        predictions.append({
            'bar': i,
            'probability': pred.probability,
            'magnitude_lift': pred.magnitude_lift,
            'confidence': pred.confidence,
            'flow_regime': pred.flow_regime.value,
            'fwd_abs_return': fwd_abs_return,
        })

    pred_df = pd.DataFrame(predictions)

    # Define fat candle (top 25%)
    fat_threshold = pred_df['fwd_abs_return'].quantile(0.75)
    pred_df['is_fat'] = pred_df['fwd_abs_return'] > fat_threshold

    print("\n" + "=" * 70)
    print("FAT CANDLE PREDICTION (Magnitude)")
    print("=" * 70)

    baseline_fat_rate = pred_df['is_fat'].mean() * 100
    baseline_move = pred_df['fwd_abs_return'].mean() * 100
    print(f"\n  Baseline fat candle rate: {baseline_fat_rate:.1f}%")
    print(f"  Baseline avg move: {baseline_move:.4f}%")

    print("\n  BY CONFIDENCE LEVEL:")
    print(f"\n  {'Confidence':<12} │ {'Count':>7} │ {'Fat%':>8} │ {'Lift':>7} │ {'Avg Move':>10}")
    print("  " + "─" * 55)

    for conf in ['LOW', 'MEDIUM', 'HIGH', 'BERSERKER']:
        subset = pred_df[pred_df['confidence'] == conf]
        if len(subset) < 20:
            continue
        fat_rate = subset['is_fat'].mean() * 100
        lift = fat_rate / baseline_fat_rate
        avg_move = subset['fwd_abs_return'].mean() * 100
        print(f"  {conf:<12} │ {len(subset):>7} │ {fat_rate:>7.1f}% │ {lift:>6.2f}x │ {avg_move:>9.4f}%")

    return pred_df


def test_direction_by_flow(df: pd.DataFrame, feature_df: pd.DataFrame):
    """Test direction prediction by flow regime."""
    predictor = CompositePredictor()

    predictions = []
    for i in range(100, len(feature_df) - 3):
        row = feature_df.iloc[i]

        pred = predictor.predict(
            energy_pct=row['energy_pct'],
            damping_pct=row['damping_pct'],
            entropy_pct=row['entropy_pct'],
            jerk_pct=row['jerk_pct'],
            impulse_pct=row['impulse_pct'],
            liquidity_pct=row['liquidity_pct'],
            reynolds_pct=row['reynolds_pct'],
            buying_pressure=row['buying_pressure'],
            momentum=row['momentum_5'],
            flow_consistency=row['flow_consistency'],
            inertia=int(row['inertia']),
        )

        if pred.confidence not in ['HIGH', 'BERSERKER']:
            continue
        if pred.direction == Direction.NEUTRAL:
            continue

        fwd_return = (df.iloc[i + 1]['close'] - df.iloc[i]['close']) / df.iloc[i]['close']

        predictions.append({
            'bar': i,
            'flow_regime': pred.flow_regime.value,
            'direction': pred.direction.value,
            'momentum': row['momentum_5'],
            'fwd_return': fwd_return,
            'correct': (pred.direction == Direction.BUY and fwd_return > 0) or
                       (pred.direction == Direction.SELL and fwd_return < 0),
        })

    pred_df = pd.DataFrame(predictions)

    print("\n" + "=" * 70)
    print("DIRECTION BY FLOW REGIME")
    print("(Laminar=Continuation, Turbulent=Reversal)")
    print("=" * 70)

    print(f"\n  {'Flow Regime':<15} │ {'Count':>7} │ {'Accuracy':>10} │ {'Edge':>8}")
    print("  " + "─" * 50)

    for regime in ['laminar', 'transitional', 'turbulent']:
        subset = pred_df[pred_df['flow_regime'] == regime]
        if len(subset) < 20:
            continue
        accuracy = subset['correct'].mean() * 100
        edge = accuracy - 50
        print(f"  {regime:<15} │ {len(subset):>7} │ {accuracy:>9.1f}% │ {edge:>+7.1f}%")

    # Test continuation vs reversal explicitly
    print("\n  CONTINUATION VS REVERSAL:")

    # Laminar should be continuation (same direction as momentum)
    laminar = pred_df[pred_df['flow_regime'] == 'laminar']
    if len(laminar) > 10:
        cont_correct = ((laminar['direction'] == 'BUY') & (laminar['fwd_return'] > 0) |
                       (laminar['direction'] == 'SELL') & (laminar['fwd_return'] < 0)).mean() * 100
        print(f"    Laminar (continuation): {cont_correct:.1f}% accuracy")

    # Turbulent should be reversal (opposite direction to momentum)
    turb = pred_df[pred_df['flow_regime'] == 'turbulent']
    if len(turb) > 10:
        rev_correct = turb['correct'].mean() * 100
        print(f"    Turbulent (reversal): {rev_correct:.1f}% accuracy")


def test_energy_recovery_exit(df: pd.DataFrame, feature_df: pd.DataFrame):
    """Test energy recovery exit effectiveness."""
    print("\n" + "=" * 70)
    print("ENERGY RECOVERY EXIT")
    print("(Dynamic MAE/MFE thresholds from rolling history)")
    print("=" * 70)

    # Run backtest with different confidence levels
    for min_conf in ['HIGH', 'BERSERKER']:
        print(f"\n  {min_conf} signals:")
        results = backtest_strategy(df, min_confidence=min_conf)

        print(f"    Trades: {results.get('trades', 0)}")
        if results.get('trades', 0) > 0:
            print(f"    Win Rate: {results['win_rate']:.1f}%")
            print(f"    Total P&L: {results['total_pnl']:.2f}%")
            print(f"    Avg P&L: {results['avg_pnl']:.4f}%")
            print(f"    Profit Factor: {results['profit_factor']:.2f}")
            print(f"    Avg MFE: {results['avg_mfe']:.3f}%")
            print(f"    Avg MAE: {results['avg_mae']:.3f}%")
            print(f"    MFE/MAE Ratio: {results['avg_mfe_mae_ratio']:.2f}")
            print(f"    MFE Captured: {results['avg_mfe_captured']:.1f}%")
            print(f"    Avg Bars Held: {results['avg_bars_held']:.1f}")
            print(f"    Exit Reasons: {results['by_exit_reason']}")


def analyze_flow_regime_performance(df: pd.DataFrame):
    """Detailed flow regime analysis."""
    print("\n" + "=" * 70)
    print("FLOW REGIME DETAILED ANALYSIS")
    print("=" * 70)

    results = backtest_strategy(df, min_confidence='HIGH')

    if results.get('trades', 0) > 0:
        regime_stats = results.get('by_flow_regime', {})
        print(f"\n  {'Regime':<15} │ {'Trades':>7} │ {'Avg P&L':>10} │ {'Total P&L':>12}")
        print("  " + "─" * 55)

        for regime in ['laminar', 'transitional', 'turbulent']:
            count = regime_stats.get('count', {}).get(regime, 0)
            mean_pnl = regime_stats.get('mean', {}).get(regime, 0)
            total_pnl = regime_stats.get('sum', {}).get(regime, 0)
            if count > 0:
                print(f"  {regime:<15} │ {count:>7} │ {mean_pnl:>+9.4f}% │ {total_pnl:>+11.2f}%")


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
    print("Computing physics features...")
    features = PhysicsFeatures()
    feature_df = features.compute(data)

    # Tests
    test_fat_candle_prediction(data, feature_df)
    test_direction_by_flow(data, feature_df)
    test_energy_recovery_exit(data, feature_df)
    analyze_flow_regime_performance(data)

    # Summary
    print("\n" + "=" * 70)
    print("BERSERKER STRATEGY SUMMARY")
    print("=" * 70)
    print("""
  BERSERKER = FAT CANDLE HUNTER

  PHASE 1: FAT CANDLE DETECTION
  - High energy + low damping = berserker state
  - Boosted by: jerk (1.37x), impulse (1.30x), liquidity (1.34x)
  - Predicts WHEN fat candle occurs (not direction)

  PHASE 2: DIRECTION
  - LAMINAR flow → CONTINUATION (follow momentum)
  - TURBULENT flow → REVERSAL (fade momentum)
  - Buying pressure confirms direction

  PHASE 3: EXIT
  - NO fixed stops
  - Exit when energy depletes (berserker exhausted)
  - Dynamic MAE/MFE thresholds from rolling history
  - Protect gains when MFE exceeds historical p75
""")


if __name__ == "__main__":
    main()
