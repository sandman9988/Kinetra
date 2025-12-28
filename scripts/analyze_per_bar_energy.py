#!/usr/bin/env python3
"""
Per-Bar Energy Release Analysis

Measure how energy dissipates bar-by-bar after berserker trigger.
Not cumulative - but the actual per-bar contribution.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from kinetra.mt5_connector import load_csv_data
from kinetra.physics_engine import PhysicsEngine


def analyze_per_bar_energy(df: pd.DataFrame, signals: pd.Index, direction_col: str, max_bars: int = 10):
    """
    Analyze energy release bar-by-bar.

    Returns the average move on each bar AFTER the signal.
    """
    per_bar_moves = {i: [] for i in range(1, max_bars + 1)}
    per_bar_abs_moves = {i: [] for i in range(1, max_bars + 1)}
    per_bar_correct = {i: [] for i in range(1, max_bars + 1)}

    for bar in signals:
        idx = df.index.get_loc(bar)
        direction = int(df.loc[bar, direction_col])
        if direction == 0:
            continue

        entry_price = df.iloc[idx]['close']

        for i in range(1, max_bars + 1):
            if idx + i >= len(df):
                break

            # Individual bar's contribution
            prev_close = df.iloc[idx + i - 1]['close']
            curr_close = df.iloc[idx + i]['close']
            bar_return = (curr_close - prev_close) / prev_close * 100

            # Directional return (positive if in our direction)
            dir_return = bar_return * direction

            per_bar_moves[i].append(dir_return)
            per_bar_abs_moves[i].append(abs(bar_return))
            per_bar_correct[i].append(1 if dir_return > 0 else 0)

    results = []
    for i in range(1, max_bars + 1):
        if per_bar_moves[i]:
            results.append({
                'bar': i,
                'avg_dir_return': np.mean(per_bar_moves[i]),
                'avg_abs_return': np.mean(per_bar_abs_moves[i]),
                'hit_rate': np.mean(per_bar_correct[i]) * 100,
                'samples': len(per_bar_moves[i]),
            })

    return results


def analyze_cumulative_energy(df: pd.DataFrame, signals: pd.Index, direction_col: str, max_bars: int = 10):
    """
    Analyze cumulative energy capture from entry to bar N.
    """
    cumulative_returns = {i: [] for i in range(1, max_bars + 1)}
    mfe_at_bar = {i: [] for i in range(1, max_bars + 1)}

    for bar in signals:
        idx = df.index.get_loc(bar)
        direction = int(df.loc[bar, direction_col])
        if direction == 0:
            continue

        entry_price = df.iloc[idx]['close']
        peak = entry_price

        for i in range(1, max_bars + 1):
            if idx + i >= len(df):
                break

            high = df.iloc[idx + i]['high']
            low = df.iloc[idx + i]['low']
            close = df.iloc[idx + i]['close']

            # Cumulative return from entry
            if direction == 1:
                cum_return = (close - entry_price) / entry_price * 100
                peak = max(peak, high)
                mfe = (peak - entry_price) / entry_price * 100
            else:
                cum_return = (entry_price - close) / entry_price * 100
                peak = min(peak, low)
                mfe = (entry_price - peak) / entry_price * 100

            cumulative_returns[i].append(cum_return)
            mfe_at_bar[i].append(mfe)

    results = []
    for i in range(1, max_bars + 1):
        if cumulative_returns[i]:
            results.append({
                'bar': i,
                'cum_return': np.mean(cumulative_returns[i]),
                'mfe': np.mean(mfe_at_bar[i]),
                'win_rate': sum(1 for r in cumulative_returns[i] if r > 0) / len(cumulative_returns[i]) * 100,
            })

    return results


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

    # Baseline
    baseline_abs_return = df['close'].pct_change().abs().mean() * 100

    print("\n" + "=" * 70)
    print("PER-BAR ENERGY RELEASE ANALYSIS")
    print("(How much energy is released on each bar after trigger?)")
    print("=" * 70)

    print(f"\n  Baseline avg absolute move: {baseline_abs_return:.3f}% per bar")

    # === BASE BERSERKER ===
    print(f"\n--- BASE BERSERKER ({berserker.sum()} signals) ---")
    print(f"  Direction: COUNTER-TREND (fade momentum)\n")

    per_bar = analyze_per_bar_energy(df, df[berserker].index, 'counter_direction')
    cumulative = analyze_cumulative_energy(df, df[berserker].index, 'counter_direction')

    print(f"  {'Bar':>4} {'DirReturn':>10} {'AbsReturn':>10} {'HitRate':>8} | {'CumReturn':>10} {'MFE':>8}")
    print("  " + "-" * 60)

    for pb, cum in zip(per_bar, cumulative):
        lift = pb['avg_abs_return'] / baseline_abs_return
        print(f"  +{pb['bar']:>3} {pb['avg_dir_return']:>+9.4f}% {pb['avg_abs_return']:>9.4f}% "
              f"({lift:.2f}x) {pb['hit_rate']:>6.1f}% | "
              f"{cum['cum_return']:>+9.4f}% {cum['mfe']:>7.4f}%")

    # === BERSERKER+ ===
    print(f"\n--- BERSERKER+ HIGH FLOW ({berserker_plus.sum()} signals) ---\n")

    per_bar_plus = analyze_per_bar_energy(df, df[berserker_plus].index, 'counter_direction')
    cumulative_plus = analyze_cumulative_energy(df, df[berserker_plus].index, 'counter_direction')

    print(f"  {'Bar':>4} {'DirReturn':>10} {'AbsReturn':>10} {'HitRate':>8} | {'CumReturn':>10} {'MFE':>8}")
    print("  " + "-" * 60)

    for pb, cum in zip(per_bar_plus, cumulative_plus):
        lift = pb['avg_abs_return'] / baseline_abs_return
        print(f"  +{pb['bar']:>3} {pb['avg_dir_return']:>+9.4f}% {pb['avg_abs_return']:>9.4f}% "
              f"({lift:.2f}x) {pb['hit_rate']:>6.1f}% | "
              f"{cum['cum_return']:>+9.4f}% {cum['mfe']:>7.4f}%")

    # === ENERGY DECAY ANALYSIS ===
    print("\n" + "=" * 70)
    print("ENERGY DECAY PATTERN")
    print("=" * 70)

    if per_bar:
        print("\n  Base Berserker - Per-bar directional edge:")
        total_edge = 0
        for pb in per_bar:
            edge = pb['avg_dir_return']
            total_edge += edge
            bar_status = "CAPTURING" if edge > 0 else "LOSING"
            print(f"    Bar +{pb['bar']}: {edge:+.4f}% ({bar_status})")
        print(f"    Total edge over {len(per_bar)} bars: {total_edge:+.4f}%")

    if per_bar_plus:
        print("\n  Berserker+ - Per-bar directional edge:")
        total_edge = 0
        for pb in per_bar_plus:
            edge = pb['avg_dir_return']
            total_edge += edge
            bar_status = "CAPTURING" if edge > 0 else "LOSING"
            print(f"    Bar +{pb['bar']}: {edge:+.4f}% ({bar_status})")
        print(f"    Total edge over {len(per_bar_plus)} bars: {total_edge:+.4f}%")

    # === OPTIMAL EXIT BAR ===
    print("\n" + "=" * 70)
    print("OPTIMAL EXIT TIMING")
    print("=" * 70)

    if cumulative:
        best_bar_base = max(cumulative, key=lambda x: x['cum_return'])
        print(f"\n  Base Berserker: Best exit at bar +{best_bar_base['bar']}")
        print(f"    Cumulative return: {best_bar_base['cum_return']:+.4f}%")
        print(f"    MFE at that point: {best_bar_base['mfe']:.4f}%")

    if cumulative_plus:
        best_bar_plus = max(cumulative_plus, key=lambda x: x['cum_return'])
        print(f"\n  Berserker+: Best exit at bar +{best_bar_plus['bar']}")
        print(f"    Cumulative return: {best_bar_plus['cum_return']:+.4f}%")
        print(f"    MFE at that point: {best_bar_plus['mfe']:.4f}%")

    # === BAR 1 FOCUS ===
    print("\n" + "=" * 70)
    print("IMMEDIATE ENERGY RELEASE (Bar +1)")
    print("=" * 70)

    if per_bar:
        bar1 = per_bar[0]
        print(f"\n  Base Berserker bar +1:")
        print(f"    Avg directional return: {bar1['avg_dir_return']:+.4f}%")
        print(f"    Avg absolute return: {bar1['avg_abs_return']:.4f}%")
        print(f"    Lift vs baseline: {bar1['avg_abs_return']/baseline_abs_return:.2f}x")
        print(f"    Hit rate: {bar1['hit_rate']:.1f}%")

    if per_bar_plus:
        bar1_plus = per_bar_plus[0]
        print(f"\n  Berserker+ bar +1:")
        print(f"    Avg directional return: {bar1_plus['avg_dir_return']:+.4f}%")
        print(f"    Avg absolute return: {bar1_plus['avg_abs_return']:.4f}%")
        print(f"    Lift vs baseline: {bar1_plus['avg_abs_return']/baseline_abs_return:.2f}x")
        print(f"    Hit rate: {bar1_plus['hit_rate']:.1f}%")


if __name__ == "__main__":
    main()
