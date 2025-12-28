#!/usr/bin/env python3
"""
Analyze if berserker bars are REVERSAL or CONTINUATION signals.

Key finding from direction analysis:
- Momentum has NEGATIVE correlation with forward returns on berserker bars
- This suggests mean reversion, not continuation

Test hypothesis: Berserker bars mark exhaustion/reversal points.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from kinetra.mt5_connector import load_csv_data
from kinetra.physics_engine import PhysicsEngine


def analyze_reversal_patterns(data_path: str = None):
    # Find data
    if data_path is None:
        project_root = Path(__file__).parent.parent
        csv_files = list(project_root.glob("*BTCUSD*.csv"))
        if not csv_files:
            print("No BTCUSD CSV file found")
            return
        data_path = str(csv_files[0])

    print(f"Using: {Path(data_path).name}")
    data = load_csv_data(data_path)
    print(f"Loaded {len(data)} bars")

    # Compute physics
    engine = PhysicsEngine(lookback=20)
    physics = engine.compute_physics_state(data['close'], data['volume'], include_percentiles=True)

    df = data.copy()
    df['energy_pct'] = physics['energy_pct']
    df['damping_pct'] = physics['damping_pct']

    # Price dynamics
    df['returns'] = df['close'].pct_change()
    df['bar_direction'] = np.sign(df['close'] - df['open'])

    # Momentum
    for w in [3, 5, 10]:
        df[f'momentum_{w}'] = df['close'].pct_change(w)

    # Forward returns
    df['fwd_return_1'] = df['returns'].shift(-1)
    df['fwd_return_2'] = df['close'].pct_change(2).shift(-2)
    df['fwd_return_3'] = df['close'].pct_change(3).shift(-3)
    df['fwd_direction_1'] = np.sign(df['fwd_return_1'])

    df = df.dropna()

    # Berserker condition
    berserker = (df['energy_pct'] > 0.75) & (df['damping_pct'] < 0.25)

    print("\n" + "=" * 70)
    print("REVERSAL vs CONTINUATION ANALYSIS")
    print("=" * 70)

    print(f"\nBerserker bars: {berserker.sum()}")

    # === TEST 1: CURRENT BAR DIRECTION vs FORWARD DIRECTION ===
    print("\n--- Current Bar Direction vs Forward Direction ---")

    df_berk = df[berserker].copy()

    # If current bar is UP, does next bar go UP or DOWN?
    up_bars = df_berk[df_berk['bar_direction'] > 0]
    down_bars = df_berk[df_berk['bar_direction'] < 0]

    up_then_up = (up_bars['fwd_direction_1'] > 0).sum()
    up_then_down = (up_bars['fwd_direction_1'] < 0).sum()
    up_continuation = up_then_up / len(up_bars) * 100 if len(up_bars) > 0 else 0
    up_reversal = up_then_down / len(up_bars) * 100 if len(up_bars) > 0 else 0

    down_then_down = (down_bars['fwd_direction_1'] < 0).sum()
    down_then_up = (down_bars['fwd_direction_1'] > 0).sum()
    down_continuation = down_then_down / len(down_bars) * 100 if len(down_bars) > 0 else 0
    down_reversal = down_then_up / len(down_bars) * 100 if len(down_bars) > 0 else 0

    print(f"\n  After UP berserker bar ({len(up_bars)} bars):")
    print(f"    Continues UP:   {up_continuation:.1f}%")
    print(f"    Reverses DOWN:  {up_reversal:.1f}%")

    print(f"\n  After DOWN berserker bar ({len(down_bars)} bars):")
    print(f"    Continues DOWN: {down_continuation:.1f}%")
    print(f"    Reverses UP:    {down_reversal:.1f}%")

    avg_continuation = (up_continuation + down_continuation) / 2
    avg_reversal = (up_reversal + down_reversal) / 2
    print(f"\n  Average continuation: {avg_continuation:.1f}%")
    print(f"  Average reversal: {avg_reversal:.1f}%")

    if avg_reversal > avg_continuation:
        print("  --> REVERSAL BIAS DETECTED")
    else:
        print("  --> CONTINUATION BIAS")

    # === TEST 2: MOMENTUM EXHAUSTION ===
    print("\n--- Momentum Exhaustion Test ---")
    print("  (Does extreme momentum on berserker bars lead to reversal?)")

    # Split by momentum quintiles
    df_berk['mom_quintile'] = pd.qcut(df_berk['momentum_5'], 5, labels=[1, 2, 3, 4, 5])

    print("\n  Momentum quintile -> Next bar direction:")
    for q in [1, 2, 3, 4, 5]:
        subset = df_berk[df_berk['mom_quintile'] == q]
        next_up = (subset['fwd_direction_1'] > 0).mean() * 100
        next_down = (subset['fwd_direction_1'] < 0).mean() * 100
        mean_mom = subset['momentum_5'].mean() * 100
        print(f"    Q{q} (mom={mean_mom:+.2f}%): UP={next_up:.1f}%, DOWN={next_down:.1f}%")

    # Extreme momentum
    extreme_up = df_berk['momentum_5'] > df_berk['momentum_5'].quantile(0.8)
    extreme_down = df_berk['momentum_5'] < df_berk['momentum_5'].quantile(0.2)

    print(f"\n  After EXTREME UP momentum berserker ({extreme_up.sum()} bars):")
    next_up = (df_berk.loc[extreme_up, 'fwd_direction_1'] > 0).mean() * 100
    print(f"    Next bar UP: {next_up:.1f}% (reversal = {100-next_up:.1f}%)")

    print(f"\n  After EXTREME DOWN momentum berserker ({extreme_down.sum()} bars):")
    next_up = (df_berk.loc[extreme_down, 'fwd_direction_1'] > 0).mean() * 100
    print(f"    Next bar UP: {next_up:.1f}% (reversal = {100-next_up:.1f}%)")

    # === TEST 3: COUNTER-TREND STRATEGY ===
    print("\n" + "=" * 70)
    print("COUNTER-TREND STRATEGY TEST")
    print("(Fade the momentum on berserker bars)")
    print("=" * 70)

    # Strategy: On berserker bar, trade AGAINST the current momentum
    df_berk['strategy_direction'] = -np.sign(df_berk['momentum_5'])  # Counter-trend
    df_berk['strategy_correct'] = df_berk['strategy_direction'] == df_berk['fwd_direction_1']

    counter_accuracy = df_berk['strategy_correct'].mean() * 100

    # Compare to trend-following
    df_berk['trend_direction'] = np.sign(df_berk['momentum_5'])
    df_berk['trend_correct'] = df_berk['trend_direction'] == df_berk['fwd_direction_1']
    trend_accuracy = df_berk['trend_correct'].mean() * 100

    print(f"\n  Counter-trend accuracy: {counter_accuracy:.1f}%")
    print(f"  Trend-following accuracy: {trend_accuracy:.1f}%")
    print(f"  Edge for counter-trend: {counter_accuracy - trend_accuracy:+.1f}%")

    # Counter-trend with stronger filter
    print("\n  With momentum filter (only extreme momentum):")

    extreme = (df_berk['momentum_5'].abs() > df_berk['momentum_5'].abs().quantile(0.7))
    df_extreme = df_berk[extreme]

    if len(df_extreme) > 50:
        counter_acc_extreme = df_extreme['strategy_correct'].mean() * 100
        trend_acc_extreme = df_extreme['trend_correct'].mean() * 100
        print(f"    Counter-trend: {counter_acc_extreme:.1f}% ({len(df_extreme)} signals)")
        print(f"    Trend-follow:  {trend_acc_extreme:.1f}%")
        print(f"    Edge: {counter_acc_extreme - trend_acc_extreme:+.1f}%")

    # === TEST 4: PROFITABILITY ===
    print("\n" + "=" * 70)
    print("PROFITABILITY TEST")
    print("=" * 70)

    # Counter-trend P&L
    df_berk['counter_pnl'] = -np.sign(df_berk['momentum_5']) * df_berk['fwd_return_1']
    df_berk['trend_pnl'] = np.sign(df_berk['momentum_5']) * df_berk['fwd_return_1']

    counter_total = df_berk['counter_pnl'].sum() * 100
    trend_total = df_berk['trend_pnl'].sum() * 100

    print(f"\n  Counter-trend cumulative return: {counter_total:+.2f}%")
    print(f"  Trend-following cumulative return: {trend_total:+.2f}%")

    # Per-trade stats
    counter_avg = df_berk['counter_pnl'].mean() * 100
    trend_avg = df_berk['trend_pnl'].mean() * 100
    print(f"\n  Counter-trend avg per signal: {counter_avg:+.4f}%")
    print(f"  Trend-following avg per signal: {trend_avg:+.4f}%")

    # Win rate
    counter_wins = (df_berk['counter_pnl'] > 0).mean() * 100
    trend_wins = (df_berk['trend_pnl'] > 0).mean() * 100
    print(f"\n  Counter-trend win rate: {counter_wins:.1f}%")
    print(f"  Trend-following win rate: {trend_wins:.1f}%")

    # === SUMMARY ===
    print("\n" + "=" * 70)
    print("SUMMARY: BERSERKER DIRECTION SIGNAL")
    print("=" * 70)

    if counter_accuracy > trend_accuracy:
        print("\n  Berserker bars show MEAN REVERSION tendency")
        print("  --> Trade AGAINST momentum for direction")
        print(f"  --> Counter-trend edge: {counter_accuracy - 50:+.1f}% over random")
    else:
        print("\n  Berserker bars show CONTINUATION tendency")
        print("  --> Trade WITH momentum for direction")
        print(f"  --> Trend-follow edge: {trend_accuracy - 50:+.1f}% over random")


if __name__ == "__main__":
    analyze_reversal_patterns()
