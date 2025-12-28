#!/usr/bin/env python3
"""
Unified Energy Analysis: Per-Bar vs Per-Trend

Shows both perspectives:
1. PER-BAR: Individual bar contribution (how energy dissipates each bar)
2. PER-TREND: Cumulative capture (how much of total move we get)

Key metric: CAPTURE EFFICIENCY = Cumulative P&L / MFE
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from kinetra.mt5_connector import load_csv_data
from kinetra.physics_engine import PhysicsEngine


def analyze_unified(df: pd.DataFrame, signals: pd.Index, direction_col: str, max_bars: int = 10):
    """
    Unified analysis showing both per-bar and cumulative metrics.
    """
    results = []

    for bar_n in range(1, max_bars + 1):
        per_bar_dir = []
        per_bar_abs = []
        per_bar_hit = []
        cumulative_pnl = []
        mfe_values = []
        mae_values = []

        for bar in signals:
            idx = df.index.get_loc(bar)
            direction = int(df.loc[bar, direction_col])
            if direction == 0 or idx + bar_n >= len(df):
                continue

            entry_price = df.iloc[idx]['close']

            # Per-bar calculation
            prev_close = df.iloc[idx + bar_n - 1]['close']
            curr_close = df.iloc[idx + bar_n]['close']
            bar_return = (curr_close - prev_close) / prev_close * 100
            dir_return = bar_return * direction

            per_bar_dir.append(dir_return)
            per_bar_abs.append(abs(bar_return))
            per_bar_hit.append(1 if dir_return > 0 else 0)

            # Cumulative calculation
            close_n = df.iloc[idx + bar_n]['close']
            if direction == 1:
                cum_pnl = (close_n - entry_price) / entry_price * 100
                # MFE: max high seen from entry to bar_n
                mfe = max((df.iloc[idx + j]['high'] - entry_price) / entry_price * 100
                         for j in range(1, bar_n + 1))
                # MAE: max adverse (lowest low)
                mae = max((entry_price - df.iloc[idx + j]['low']) / entry_price * 100
                         for j in range(1, bar_n + 1))
            else:
                cum_pnl = (entry_price - close_n) / entry_price * 100
                mfe = max((entry_price - df.iloc[idx + j]['low']) / entry_price * 100
                         for j in range(1, bar_n + 1))
                mae = max((df.iloc[idx + j]['high'] - entry_price) / entry_price * 100
                         for j in range(1, bar_n + 1))

            cumulative_pnl.append(cum_pnl)
            mfe_values.append(mfe)
            mae_values.append(mae)

        if not per_bar_dir:
            continue

        avg_cum_pnl = np.mean(cumulative_pnl)
        avg_mfe = np.mean(mfe_values)
        capture_eff = avg_cum_pnl / avg_mfe * 100 if avg_mfe > 0 else 0

        results.append({
            'bar': bar_n,
            # Per-bar metrics
            'per_bar_dir': np.mean(per_bar_dir),
            'per_bar_abs': np.mean(per_bar_abs),
            'per_bar_hit': np.mean(per_bar_hit) * 100,
            # Cumulative metrics
            'cum_pnl': avg_cum_pnl,
            'mfe': avg_mfe,
            'mae': np.mean(mae_values),
            'capture_eff': capture_eff,
            'win_rate': sum(1 for p in cumulative_pnl if p > 0) / len(cumulative_pnl) * 100,
            'samples': len(per_bar_dir),
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

    # Baseline
    baseline_abs = df['close'].pct_change().abs().mean() * 100

    # Signals
    berserker = (df['energy_pct'] > 0.75) & (df['damping_pct'] < 0.25)
    berserker_plus = berserker & (df['flow_consistency'] > 0.7) & (df['volume_pct'] > 0.6)

    print("\n" + "=" * 80)
    print("UNIFIED ENERGY ANALYSIS: PER-BAR vs PER-TREND")
    print("=" * 80)
    print(f"\nBaseline move: {baseline_abs:.3f}% per bar")

    # === BERSERKER+ (best signal) ===
    print(f"\n{'─' * 80}")
    print(f"BERSERKER+ ({berserker_plus.sum()} signals) - Counter-trend direction")
    print(f"{'─' * 80}")

    results = analyze_unified(df, df[berserker_plus].index, 'counter_direction')

    print(f"\n{'Bar':>4} │ {'─── PER-BAR ───':^28} │ {'─── PER-TREND (Cumulative) ───':^35}")
    print(f"{'':>4} │ {'Dir%':>8} {'Abs%':>8} {'Hit%':>8} │ {'Cum%':>8} {'MFE%':>8} {'MAE%':>8} {'Capt%':>8}")
    print("─" * 85)

    for r in results:
        lift = r['per_bar_abs'] / baseline_abs
        print(f"+{r['bar']:>3} │ {r['per_bar_dir']:>+7.3f}% {r['per_bar_abs']:>7.3f}% {r['per_bar_hit']:>6.1f}% │ "
              f"{r['cum_pnl']:>+7.3f}% {r['mfe']:>7.3f}% {r['mae']:>7.3f}% {r['capture_eff']:>6.1f}%")

    # === ANALYSIS ===
    print("\n" + "=" * 80)
    print("INTERPRETATION")
    print("=" * 80)

    if results:
        # Per-bar decay
        print("\n┌─ PER-BAR PERSPECTIVE ─────────────────────────────────────┐")
        print("│ How energy releases bar-by-bar:                           │")

        total_edge = sum(r['per_bar_dir'] for r in results)
        bar1 = results[0]
        print(f"│  Bar +1 captures: {bar1['per_bar_dir']:+.3f}% (lift: {bar1['per_bar_abs']/baseline_abs:.2f}x)           │")
        print(f"│  Direction accuracy: {bar1['per_bar_hit']:.1f}%                                │")
        print(f"│  Total edge over 10 bars: {total_edge:+.4f}%                       │")

        # Find where per-bar edge turns negative consistently
        edge_bars = [r for r in results if r['per_bar_dir'] > 0]
        print(f"│  Positive-edge bars: {len(edge_bars)}/10                                  │")
        print("└───────────────────────────────────────────────────────────┘")

        # Per-trend analysis
        print("\n┌─ PER-TREND PERSPECTIVE ───────────────────────────────────┐")
        print("│ Cumulative capture over full move:                        │")

        best_cum = max(results, key=lambda x: x['cum_pnl'])
        best_eff = max(results, key=lambda x: x['capture_eff'])

        print(f"│  Best cumulative P&L: bar +{best_cum['bar']} ({best_cum['cum_pnl']:+.3f}%)            │")
        print(f"│  MFE at that point: {best_cum['mfe']:.3f}%                             │")
        print(f"│  Capture efficiency: {best_cum['capture_eff']:.1f}%                              │")
        print(f"│                                                           │")
        print(f"│  Best capture efficiency: bar +{best_eff['bar']} ({best_eff['capture_eff']:.1f}%)           │")
        print("└───────────────────────────────────────────────────────────┘")

        # Key insight
        print("\n┌─ KEY INSIGHT ─────────────────────────────────────────────┐")
        bar10 = results[-1]
        mfe_leak = bar10['mfe'] - bar10['cum_pnl']
        print(f"│  MFE opportunity: {bar10['mfe']:.3f}%                                 │")
        print(f"│  Actual capture:  {bar10['cum_pnl']:+.3f}%                                │")
        print(f"│  LEAKED: {mfe_leak:.3f}% ({(mfe_leak/bar10['mfe']*100):.0f}% of opportunity)                    │")
        print(f"│                                                           │")
        print("│  This is why trailing stops matter!                       │")
        print("│  The edge is real but market takes it back.               │")
        print("└───────────────────────────────────────────────────────────┘")

        # Optimal strategy
        print("\n" + "=" * 80)
        print("OPTIMAL STRATEGY BASED ON BOTH PERSPECTIVES")
        print("=" * 80)

        print("""
  PER-BAR insight:
  - Most directional edge is on BAR +1 (immediate reaction)
  - Volatility stays elevated but direction becomes random after bar 3

  PER-TREND insight:
  - MFE keeps growing = opportunity exists
  - But cumulative P&L doesn't grow as fast = giving back gains

  SOLUTION:
  1. Enter on Berserker+ signal (counter-trend)
  2. Tight trailing stop (0.2-0.3%) to lock in immediate gains
  3. Let it run if trend continues, but don't give back the edge

  The energy is FRONT-LOADED in direction, SUSTAINED in volatility.
  Capture the direction early, ride the volatility with trailing stop.
""")

    # === COMPARE BASE VS PLUS ===
    print("\n" + "=" * 80)
    print("BASE BERSERKER vs BERSERKER+ COMPARISON")
    print("=" * 80)

    results_base = analyze_unified(df, df[berserker].index, 'counter_direction')

    if results_base and results:
        print(f"\n{'':>12} │ {'Base Berserker':>20} │ {'Berserker+':>20}")
        print(f"{'Metric':>12} │ {'('+str(berserker.sum())+' signals)':>20} │ {'('+str(berserker_plus.sum())+' signals)':>20}")
        print("─" * 60)

        b1_base = results_base[0]
        b1_plus = results[0]
        print(f"{'Bar+1 Dir%':>12} │ {b1_base['per_bar_dir']:>+19.4f}% │ {b1_plus['per_bar_dir']:>+19.4f}%")
        print(f"{'Bar+1 Hit%':>12} │ {b1_base['per_bar_hit']:>19.1f}% │ {b1_plus['per_bar_hit']:>19.1f}%")
        print(f"{'Bar+1 Lift':>12} │ {b1_base['per_bar_abs']/baseline_abs:>19.2f}x │ {b1_plus['per_bar_abs']/baseline_abs:>19.2f}x")

        total_base = sum(r['per_bar_dir'] for r in results_base)
        total_plus = sum(r['per_bar_dir'] for r in results)
        print(f"{'10bar Edge%':>12} │ {total_base:>+19.4f}% │ {total_plus:>+19.4f}%")

        b10_base = results_base[-1]
        b10_plus = results[-1]
        print(f"{'Capture Eff':>12} │ {b10_base['capture_eff']:>19.1f}% │ {b10_plus['capture_eff']:>19.1f}%")


if __name__ == "__main__":
    main()
