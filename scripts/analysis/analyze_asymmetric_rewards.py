#!/usr/bin/env python3
"""
Asymmetric Reward Shaping: The Journey Counts

Key principle: Punish MAE (adverse excursion), reward MFE capture efficiency.

Metrics:
- Edge Ratio: MFE / MAE (opportunity vs risk)
- Journey Quality: How clean was the path to profit?
- Capture Efficiency: P&L / MFE (how much of the opportunity we got)
- Path Score: Penalize "oscillating" trades vs "clean" trades

A trade that goes +1% then -0.5% then ends +0.5% is WORSE than
a trade that goes +0.5% directly, even if same final P&L.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from kinetra.mt5_connector import load_csv_data
from kinetra.physics_engine import PhysicsEngine


def analyze_trade_journey(df: pd.DataFrame, entry_idx: int, direction: int, max_bars: int = 10):
    """
    Analyze the full journey of a trade, bar by bar.

    Returns per-bar metrics showing the path quality.
    """
    entry_price = df.iloc[entry_idx]['close']
    journey = []

    running_mfe = 0
    running_mae = 0

    for i in range(1, max_bars + 1):
        if entry_idx + i >= len(df):
            break

        high = df.iloc[entry_idx + i]['high']
        low = df.iloc[entry_idx + i]['low']
        close = df.iloc[entry_idx + i]['close']

        if direction == 1:  # Long
            pnl = (close - entry_price) / entry_price * 100
            bar_mfe = (high - entry_price) / entry_price * 100
            bar_mae = (entry_price - low) / entry_price * 100
        else:  # Short
            pnl = (entry_price - close) / entry_price * 100
            bar_mfe = (entry_price - low) / entry_price * 100
            bar_mae = (high - entry_price) / entry_price * 100

        running_mfe = max(running_mfe, bar_mfe)
        running_mae = max(running_mae, bar_mae)

        # Journey quality: how much of MFE are we keeping?
        # If MFE = 0.5% but PnL = 0.1%, we gave back 80%
        mfe_leak = running_mfe - pnl if running_mfe > 0 else 0

        # Asymmetric score: reward capture, punish MAE
        # Score = (PnL - MAE_penalty) / MFE
        mae_penalty = running_mae * 1.5  # MAE costs 1.5x its value
        asymmetric_score = (pnl - mae_penalty) / running_mfe if running_mfe > 0 else 0

        journey.append({
            'bar': i,
            'pnl': pnl,
            'mfe': running_mfe,
            'mae': running_mae,
            'mfe_leak': mfe_leak,
            'edge_ratio': running_mfe / running_mae if running_mae > 0 else float('inf'),
            'capture_eff': pnl / running_mfe * 100 if running_mfe > 0 else 0,
            'asymmetric_score': asymmetric_score,
        })

    return journey


def compute_journey_quality(journey):
    """
    Compute overall journey quality metrics.

    Good journey: Quick to profit, minimal drawdown, steady capture.
    Bad journey: Oscillates, deep drawdown before profit, gives back gains.
    """
    if not journey:
        return None

    # Time to first profit
    first_profit_bar = next((j['bar'] for j in journey if j['pnl'] > 0), None)

    # Maximum MFE leak during journey
    max_leak = max(j['mfe_leak'] for j in journey)

    # Final metrics
    final = journey[-1]

    # Path efficiency: integral of PnL / integral of MFE opportunity
    pnl_area = sum(j['pnl'] for j in journey)
    mfe_area = sum(j['mfe'] for j in journey)
    path_efficiency = pnl_area / mfe_area if mfe_area > 0 else 0

    # Monotonicity: how often did PnL increase vs decrease?
    pnl_changes = [journey[i]['pnl'] - journey[i-1]['pnl'] for i in range(1, len(journey))]
    monotonic_score = sum(1 for c in pnl_changes if c > 0) / len(pnl_changes) if pnl_changes else 0

    return {
        'first_profit_bar': first_profit_bar,
        'max_mfe_leak': max_leak,
        'final_pnl': final['pnl'],
        'final_mfe': final['mfe'],
        'final_mae': final['mae'],
        'edge_ratio': final['edge_ratio'],
        'capture_eff': final['capture_eff'],
        'path_efficiency': path_efficiency * 100,
        'monotonic_score': monotonic_score * 100,
        'asymmetric_score': final['asymmetric_score'],
    }


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
    df['momentum_5'] = df['close'].pct_change(5)
    df['counter_direction'] = -np.sign(df['momentum_5'])

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

    print("\n" + "=" * 80)
    print("ASYMMETRIC REWARD ANALYSIS: THE JOURNEY COUNTS")
    print("=" * 80)

    # Analyze all trades
    all_journeys = []

    for bar in df[berserker_plus].index:
        idx = df.index.get_loc(bar)
        direction = int(df.loc[bar, 'counter_direction'])
        if direction == 0 or idx + 10 >= len(df):
            continue

        journey = analyze_trade_journey(df, idx, direction, max_bars=10)
        quality = compute_journey_quality(journey)
        if quality:
            all_journeys.append(quality)

    print(f"\nAnalyzed {len(all_journeys)} Berserker+ trades")

    # === JOURNEY QUALITY METRICS ===
    print("\n" + "─" * 80)
    print("JOURNEY QUALITY DISTRIBUTION")
    print("─" * 80)

    print(f"\n{'Metric':>20} │ {'Mean':>10} │ {'Median':>10} │ {'Std':>10}")
    print("─" * 60)

    for metric in ['final_pnl', 'final_mfe', 'final_mae', 'edge_ratio',
                   'capture_eff', 'path_efficiency', 'monotonic_score']:
        values = [j[metric] for j in all_journeys if j[metric] != float('inf')]
        print(f"{metric:>20} │ {np.mean(values):>10.3f} │ {np.median(values):>10.3f} │ {np.std(values):>10.3f}")

    # === MAE/MFE INEFFICIENCY ===
    print("\n" + "─" * 80)
    print("MAE/MFE INEFFICIENCY ANALYSIS")
    print("─" * 80)

    # Bucket by edge ratio (MFE/MAE)
    good_edge = [j for j in all_journeys if j['edge_ratio'] > 1.5]  # More MFE than MAE
    poor_edge = [j for j in all_journeys if j['edge_ratio'] < 0.8]  # More MAE than MFE
    neutral = [j for j in all_journeys if 0.8 <= j['edge_ratio'] <= 1.5]

    print(f"\n  Edge Ratio = MFE / MAE (how much opportunity vs how much pain)")
    print(f"\n  {'Category':>20} │ {'Count':>8} │ {'Avg PnL':>10} │ {'Avg MFE':>10} │ {'Avg MAE':>10}")
    print("  " + "─" * 70)

    for name, bucket in [('GOOD (>1.5)', good_edge), ('NEUTRAL (0.8-1.5)', neutral), ('POOR (<0.8)', poor_edge)]:
        if bucket:
            print(f"  {name:>20} │ {len(bucket):>8} │ {np.mean([j['final_pnl'] for j in bucket]):>+9.4f}% │ "
                  f"{np.mean([j['final_mfe'] for j in bucket]):>9.4f}% │ {np.mean([j['final_mae'] for j in bucket]):>9.4f}%")

    # === PATH QUALITY ===
    print("\n" + "─" * 80)
    print("PATH QUALITY: CLEAN vs CHOPPY TRADES")
    print("─" * 80)

    # Bucket by monotonicity (how often PnL increased)
    clean = [j for j in all_journeys if j['monotonic_score'] > 60]  # Mostly increasing
    choppy = [j for j in all_journeys if j['monotonic_score'] < 40]  # Oscillating

    print(f"\n  Monotonic Score = % of bars where PnL increased")
    print(f"\n  {'Path Type':>20} │ {'Count':>8} │ {'Avg PnL':>10} │ {'Capture%':>10} │ {'MFE Leak':>10}")
    print("  " + "─" * 70)

    for name, bucket in [('CLEAN (>60%)', clean), ('CHOPPY (<40%)', choppy)]:
        if bucket:
            print(f"  {name:>20} │ {len(bucket):>8} │ {np.mean([j['final_pnl'] for j in bucket]):>+9.4f}% │ "
                  f"{np.mean([j['capture_eff'] for j in bucket]):>9.1f}% │ {np.mean([j['max_mfe_leak'] for j in bucket]):>9.4f}%")

    # === ASYMMETRIC SCORING ===
    print("\n" + "=" * 80)
    print("ASYMMETRIC REWARD SCORING")
    print("=" * 80)

    print("""
  Formula: Asymmetric Score = (PnL - 1.5 * MAE) / MFE

  Interpretation:
  - Positive = Good trade (profit exceeds penalized MAE)
  - Negative = Bad trade (MAE penalty overwhelms profit)
  - Higher = Better risk-adjusted journey
""")

    # Distribution of asymmetric scores
    scores = [j['asymmetric_score'] for j in all_journeys]
    positive = sum(1 for s in scores if s > 0)
    negative = sum(1 for s in scores if s <= 0)

    print(f"  Score Distribution:")
    print(f"    Positive (good): {positive} ({positive/len(scores)*100:.1f}%)")
    print(f"    Negative (bad):  {negative} ({negative/len(scores)*100:.1f}%)")
    print(f"    Mean score: {np.mean(scores):+.4f}")
    print(f"    Median score: {np.median(scores):+.4f}")

    # Best vs worst trades by asymmetric score
    sorted_journeys = sorted(all_journeys, key=lambda x: x['asymmetric_score'], reverse=True)

    print(f"\n  Top 5 trades by journey quality:")
    for i, j in enumerate(sorted_journeys[:5], 1):
        print(f"    {i}. Score: {j['asymmetric_score']:+.3f} │ PnL: {j['final_pnl']:+.3f}% │ "
              f"MFE: {j['final_mfe']:.3f}% │ MAE: {j['final_mae']:.3f}%")

    print(f"\n  Bottom 5 trades by journey quality:")
    for i, j in enumerate(sorted_journeys[-5:], 1):
        print(f"    {i}. Score: {j['asymmetric_score']:+.3f} │ PnL: {j['final_pnl']:+.3f}% │ "
              f"MFE: {j['final_mfe']:.3f}% │ MAE: {j['final_mae']:.3f}%")

    # === RECOMMENDATIONS ===
    print("\n" + "=" * 80)
    print("ASYMMETRIC EXIT RULES")
    print("=" * 80)

    # Find when edge ratio is best (MFE growing faster than MAE)
    avg_edge_by_bar = []
    for bar_n in range(1, 11):
        edges = []
        for bar in df[berserker_plus].index:
            idx = df.index.get_loc(bar)
            direction = int(df.loc[bar, 'counter_direction'])
            if direction == 0 or idx + bar_n >= len(df):
                continue

            journey = analyze_trade_journey(df, idx, direction, max_bars=bar_n)
            if journey:
                final = journey[-1]
                if final['edge_ratio'] != float('inf'):
                    edges.append(final['edge_ratio'])

        if edges:
            avg_edge_by_bar.append({'bar': bar_n, 'avg_edge': np.mean(edges)})

    print("\n  Edge Ratio (MFE/MAE) by Exit Bar:")
    print(f"\n  {'Bar':>4} │ {'Edge Ratio':>12} │ {'Interpretation':>30}")
    print("  " + "─" * 55)

    for e in avg_edge_by_bar:
        interp = "MFE > MAE (good)" if e['avg_edge'] > 1 else "MAE > MFE (bad)"
        print(f"  +{e['bar']:>3} │ {e['avg_edge']:>11.3f}x │ {interp:>30}")

    # Find optimal exit
    best_edge_bar = max(avg_edge_by_bar, key=lambda x: x['avg_edge'])

    print(f"\n  RECOMMENDATION:")
    print(f"  ─────────────────────────────────────────────────────────")
    print(f"  Best edge ratio at bar +{best_edge_bar['bar']}: {best_edge_bar['avg_edge']:.2f}x")
    print(f"")
    print(f"  Rules to punish MAE/MFE inefficiency:")
    print(f"  1. Exit if MAE > 0.5% before any MFE (path went wrong)")
    print(f"  2. Use trailing stop of 0.2-0.3% to lock in MFE")
    print(f"  3. Take profit at bar +3 if edge ratio dropping")
    print(f"  4. Never hold past bar +5 without trailing stop lock")


if __name__ == "__main__":
    main()
