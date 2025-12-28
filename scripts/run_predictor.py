#!/usr/bin/env python3
"""
Trigger Predictor Demo

Shows the probability predictor in action on historical data.
Identifies BERSERKER opportunities with high-probability energy release.

Usage:
    python scripts/run_predictor.py --symbol BTCUSD
"""

import argparse
import sys
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from kinetra import PhysicsEngine, load_csv_data, TriggerPredictor


def main():
    parser = argparse.ArgumentParser(description='Run trigger predictor demo')
    parser.add_argument('--data', type=str, help='Path to CSV data file')
    parser.add_argument('--symbol', type=str, default='BTCUSD', help='Symbol name')
    parser.add_argument('--lookback', type=int, default=20, help='Physics lookback')
    parser.add_argument('--bars', type=int, default=100, help='Bars to analyze')

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

    # Load data
    print("\nLoading data...")
    data = load_csv_data(str(data_path))
    print(f"Loaded {len(data)} bars")

    # Compute physics
    print(f"\nComputing physics features (lookback={args.lookback})...")
    engine = PhysicsEngine(lookback=args.lookback)
    physics = engine.compute_physics_state(data['close'])

    df = data.copy()
    df['energy'] = physics['energy']
    df['damping'] = physics['damping']
    df['entropy'] = physics['entropy']
    df['regime'] = physics['regime']
    df['energy_velocity'] = df['energy'].diff()
    df['energy_accel'] = df['energy_velocity'].diff()
    df['energy_ma20'] = df['energy'].rolling(20).mean()
    df['momentum_5'] = df['close'].pct_change(5)
    df['returns'] = df['close'].pct_change()
    df = df.fillna(0)

    # Mark actual high-energy bars (for validation)
    threshold_80 = df['energy'].quantile(0.80)
    df['is_high_energy'] = df['energy'] >= threshold_80
    df['next_is_high'] = df['is_high_energy'].shift(-1).fillna(False)

    # Initialize predictor
    predictor = TriggerPredictor(lookback=args.lookback, horizon=2)

    # Warm up history
    warmup_bars = min(200, len(df) - args.bars - 10)
    for i in range(warmup_bars):
        predictor.update_history(df.iloc[i]['energy'], df.iloc[i]['damping'])

    # Run predictions
    print("\n" + "=" * 80)
    print("TRIGGER PREDICTOR - BERSERKER SIGNALS")
    print("=" * 80)

    start_bar = len(df) - args.bars
    predictions = []
    berserker_signals = []

    for i in range(start_bar, len(df) - 2):  # -2 for horizon validation
        pred = predictor.predict_from_dataframe(df, i)
        predictions.append({
            'bar': i,
            'time': df.iloc[i].get('time', i),
            'price': df.iloc[i]['close'],
            'energy': df.iloc[i]['energy'],
            'regime': df.iloc[i]['regime'],
            'probability': pred.probability,
            'direction': pred.direction.value,
            'confidence': pred.confidence,
            'conditions': ', '.join(pred.conditions_met[:4]),
            'actual_high': df.iloc[i]['next_is_high'],
        })

        if pred.confidence == 'BERSERKER':
            berserker_signals.append(predictions[-1])

    pred_df = pd.DataFrame(predictions)

    # Show berserker signals
    print(f"\nFound {len(berserker_signals)} BERSERKER signals in last {args.bars} bars\n")

    if berserker_signals:
        print(f"{'Bar':>6} {'Price':>12} {'Prob':>6} {'Dir':>6} {'Actual':>8} {'Conditions':<40}")
        print("-" * 80)

        correct = 0
        for sig in berserker_signals[-20:]:  # Last 20
            actual = "✓ HIT" if sig['actual_high'] else "✗ miss"
            if sig['actual_high']:
                correct += 1
            print(f"{sig['bar']:>6} {sig['price']:>12,.2f} {sig['probability']*100:>5.0f}% "
                  f"{sig['direction']:>6} {actual:>8} {sig['conditions'][:38]:<40}")

    # Validation statistics
    print("\n" + "=" * 80)
    print("BERSERKER SIGNAL VALIDATION")
    print("=" * 80)

    if len(berserker_signals) > 0:
        berserker_df = pd.DataFrame(berserker_signals)
        hit_rate = berserker_df['actual_high'].mean() * 100
        base_rate = pred_df['actual_high'].mean() * 100
        lift = hit_rate / base_rate if base_rate > 0 else 0

        print(f"\n  Total signals:    {len(berserker_signals)}")
        print(f"  Hit rate:         {hit_rate:.1f}%")
        print(f"  Base rate:        {base_rate:.1f}%")
        print(f"  Lift:             {lift:.2f}x")
        print(f"\n  BERSERKER achieves {lift:.1f}x better than random!")

    # Show confidence distribution
    print("\n" + "=" * 80)
    print("CONFIDENCE DISTRIBUTION")
    print("=" * 80)

    confidence_stats = pred_df.groupby('confidence').agg({
        'bar': 'count',
        'actual_high': 'mean',
    }).rename(columns={'bar': 'count', 'actual_high': 'hit_rate'})

    print(f"\n{'Confidence':<12} {'Signals':>10} {'Hit Rate':>10}")
    print("-" * 35)
    for conf in ['BERSERKER', 'HIGH', 'MEDIUM', 'LOW']:
        if conf in confidence_stats.index:
            row = confidence_stats.loc[conf]
            print(f"{conf:<12} {int(row['count']):>10} {row['hit_rate']*100:>9.1f}%")

    # Example output message
    print("\n" + "=" * 80)
    print("SAMPLE OUTPUT MESSAGE")
    print("=" * 80)

    if berserker_signals:
        last_sig = berserker_signals[-1]
        prob = int(last_sig['probability'] * 100)
        direction = last_sig['direction']
        print(f"""
  There is an {prob}% chance that {direction} energy will be released
  in the next 2 bars that fits our quantile profile for BERSERKER standby.

  Conditions: {last_sig['conditions']}
  Current regime: {last_sig['regime']}
  Current price: {last_sig['price']:,.2f}
""")


if __name__ == "__main__":
    main()
