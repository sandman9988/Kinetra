#!/usr/bin/env python3
"""
Energy Capture Analysis - How much of the move can we capture?

Clean analysis of trade management for berserker entries.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import List, Any

from numpy import floating

sys.path.insert(0, str(Path(__file__).parent.parent))

from kinetra.mt5_connector import load_csv_data
from kinetra.physics_engine import PhysicsEngine


@dataclass
class Trade:
    entry_bar: int
    exit_bar: int
    direction: int
    entry_price: float
    exit_price: float
    pnl_pct: float
    mae_pct: float
    mfe_pct: float
    exit_reason: str


def simulate_with_trailing_stop(
    df: pd.DataFrame,
    entry_idx: int,
    direction: int,
    trailing_pct: float,
    stop_loss_pct: float = None,
    max_bars: int = 20,
) -> Trade:
    """Simulate trade with trailing stop."""
    entry_price = df.iloc[entry_idx]['close']

    peak_price = entry_price
    mae = 0
    mfe = 0
    exit_idx = entry_idx
    exit_price = entry_price
    exit_reason = 'max_bars'

    for i in range(1, max_bars + 1):
        idx = entry_idx + i
        if idx >= len(df):
            break

        high = df.iloc[idx]['high']
        low = df.iloc[idx]['low']
        close = df.iloc[idx]['close']

        if direction == 1:  # Long
            # Update MFE/MAE
            mfe = max(mfe, (high - entry_price) / entry_price * 100)
            mae = max(mae, (entry_price - low) / entry_price * 100)

            # Update peak for trailing
            peak_price = max(peak_price, high)
            trail_stop = peak_price * (1 - trailing_pct / 100)

            # Check stop loss first
            if stop_loss_pct and low <= entry_price * (1 - stop_loss_pct / 100):
                exit_price = entry_price * (1 - stop_loss_pct / 100)
                exit_idx = idx
                exit_reason = 'stop_loss'
                break

            # Check trailing stop
            if low <= trail_stop:
                exit_price = max(trail_stop, low)
                exit_idx = idx
                exit_reason = 'trailing_stop'
                break

        else:  # Short
            mfe = max(mfe, (entry_price - low) / entry_price * 100)
            mae = max(mae, (high - entry_price) / entry_price * 100)

            peak_price = min(peak_price, low)
            trail_stop = peak_price * (1 + trailing_pct / 100)

            if stop_loss_pct and high >= entry_price * (1 + stop_loss_pct / 100):
                exit_price = entry_price * (1 + stop_loss_pct / 100)
                exit_idx = idx
                exit_reason = 'stop_loss'
                break

            if high >= trail_stop:
                exit_price = min(trail_stop, high)
                exit_idx = idx
                exit_reason = 'trailing_stop'
                break

        exit_idx = idx
        exit_price = close

    # Calculate P&L
    if direction == 1:
        pnl_pct = (exit_price - entry_price) / entry_price * 100
    else:
        pnl_pct = (entry_price - exit_price) / entry_price * 100

    return Trade(
        entry_bar=entry_idx,
        exit_bar=exit_idx,
        direction=direction,
        entry_price=entry_price,
        exit_price=exit_price,
        pnl_pct=pnl_pct,
        mae_pct=mae,
        mfe_pct=mfe,
        exit_reason=exit_reason,
    )


def calculate_omega(pnls: List[float]) -> float:
    """Omega ratio: gains / |losses|"""
    gains = sum(p for p in pnls if p > 0)
    losses = abs(sum(p for p in pnls if p < 0))
    return gains / losses if losses > 0 else float('inf')


def test_strategy(df: pd.DataFrame, signals: pd.Index, ts_pct: float, sl_pct: float = None) -> dict[str, int | float |
                                                                                                         floating[
                                                                                                             Any]] | None:
    """Test a trailing stop strategy."""
    trades = []

    for bar in signals:
        idx = df.index.get_loc(bar)
        if idx + 20 >= len(df):
            continue

        direction = int(df.loc[bar, 'counter_direction'])
        if direction == 0:
            continue

        trade = simulate_with_trailing_stop(
            df, idx, direction,
            trailing_pct=ts_pct,
            stop_loss_pct=sl_pct,
            max_bars=15,
        )
        trades.append(trade)

    if not trades:
        return None

    pnls = [t.pnl_pct for t in trades]
    mfes = [t.mfe_pct for t in trades]
    maes = [t.mae_pct for t in trades]

    wins = [t for t in trades if t.pnl_pct > 0]
    losses = [t for t in trades if t.pnl_pct <= 0]

    # Energy capture: average profit / average MFE opportunity
    avg_mfe = np.mean(mfes)
    avg_pnl = np.mean(pnls)
    energy_capture = avg_pnl / avg_mfe * 100 if avg_mfe > 0 else 0

    # For winning trades only
    if wins:
        win_pnls = [t.pnl_pct for t in wins]
        win_mfes = [t.mfe_pct for t in wins]
        win_capture = np.mean(win_pnls) / np.mean(win_mfes) * 100 if np.mean(win_mfes) > 0 else 0
    else:
        win_capture = 0

    return {
        'trades': len(trades),
        'win_rate': len(wins) / len(trades) * 100,
        'avg_pnl': avg_pnl,
        'total_pnl': sum(pnls),
        'avg_mfe': avg_mfe,
        'avg_mae': np.mean(maes),
        'omega': calculate_omega(pnls),
        'energy_capture': energy_capture,  # % of available energy captured
        'win_capture': win_capture,  # % captured on winning trades
        'profit_factor': sum(p for p in pnls if p > 0) / abs(sum(p for p in pnls if p < 0)) if sum(p for p in pnls if p < 0) != 0 else float('inf'),
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

    # Flow consistency for enhanced signal
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

    # Define signal types
    berserker = (df['energy_pct'] > 0.75) & (df['damping_pct'] < 0.25)
    berserker_plus = berserker & (df['flow_consistency'] > 0.7) & (df['volume_pct'] > 0.6)

    print(f"\nBerserker signals: {berserker.sum()}")
    print(f"Berserker+ (high flow + volume): {berserker_plus.sum()}")

    # === TRAILING STOP OPTIMIZATION ===
    print("\n" + "=" * 70)
    print("TRAILING STOP OPTIMIZATION")
    print("=" * 70)

    ts_values = [0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6, 0.75, 1.0]

    print("\n--- Base Berserker Signal ---\n")
    print(f"  {'TS%':>5} {'Trades':>7} {'Win%':>7} {'AvgPnL':>8} {'TotalPnL':>10} {'Omega':>7} {'Capture':>8}")
    print("  " + "-" * 60)

    results = []
    for ts in ts_values:
        r = test_strategy(df, df[berserker].index, ts_pct=ts)
        if r:
            results.append({'ts': ts, **r})
            omega_str = f"{r['omega']:.2f}" if r['omega'] != float('inf') else "inf"
            print(f"  {ts:>4.1f}% {r['trades']:>7} {r['win_rate']:>6.1f}% "
                  f"{r['avg_pnl']:>+7.3f}% {r['total_pnl']:>+9.1f}% "
                  f"{omega_str:>7} {r['energy_capture']:>7.1f}%")

    # Best by Omega
    if results:
        best = max(results, key=lambda x: x['omega'] if x['omega'] != float('inf') else 0)
        print(f"\n  Best: TS {best['ts']}% - Omega {best['omega']:.2f}, Capture {best['energy_capture']:.1f}%")

    # === WITH STOP LOSS ===
    print("\n--- Trailing Stop + Stop Loss ---\n")
    print(f"  {'TS%':>5} {'SL%':>5} {'Win%':>7} {'AvgPnL':>8} {'TotalPnL':>10} {'Omega':>7} {'Capture':>8}")
    print("  " + "-" * 65)

    sl_ts_combos = [
        (0.3, 0.5), (0.3, 0.75), (0.3, 1.0),
        (0.4, 0.5), (0.4, 0.75), (0.4, 1.0),
        (0.5, 0.75), (0.5, 1.0),
    ]

    combo_results = []
    for ts, sl in sl_ts_combos:
        r = test_strategy(df, df[berserker].index, ts_pct=ts, sl_pct=sl)
        if r:
            combo_results.append({'ts': ts, 'sl': sl, **r})
            omega_str = f"{r['omega']:.2f}" if r['omega'] != float('inf') else "inf"
            print(f"  {ts:>4.1f}% {sl:>4.1f}% {r['win_rate']:>6.1f}% "
                  f"{r['avg_pnl']:>+7.3f}% {r['total_pnl']:>+9.1f}% "
                  f"{omega_str:>7} {r['energy_capture']:>7.1f}%")

    # === BERSERKER+ SIGNAL ===
    print("\n" + "=" * 70)
    print("BERSERKER+ (High Flow + Volume) - Optimal Direction")
    print("=" * 70)

    print("\n--- Trailing Stop on Berserker+ ---\n")
    print(f"  {'TS%':>5} {'Trades':>7} {'Win%':>7} {'AvgPnL':>8} {'TotalPnL':>10} {'Omega':>7} {'Capture':>8}")
    print("  " + "-" * 60)

    plus_results = []
    for ts in ts_values:
        r = test_strategy(df, df[berserker_plus].index, ts_pct=ts)
        if r and r['trades'] >= 30:
            plus_results.append({'ts': ts, **r})
            omega_str = f"{r['omega']:.2f}" if r['omega'] != float('inf') else "inf"
            print(f"  {ts:>4.1f}% {r['trades']:>7} {r['win_rate']:>6.1f}% "
                  f"{r['avg_pnl']:>+7.3f}% {r['total_pnl']:>+9.1f}% "
                  f"{omega_str:>7} {r['energy_capture']:>7.1f}%")

    # === SUMMARY ===
    print("\n" + "=" * 70)
    print("ENERGY CAPTURE SUMMARY")
    print("=" * 70)

    if results:
        best_base = max(results, key=lambda x: x['total_pnl'])
        print(f"\n  BASE BERSERKER (best by total P&L):")
        print(f"    Strategy: TS {best_base['ts']}%")
        print(f"    Total P&L: {best_base['total_pnl']:+.1f}%")
        print(f"    Win rate: {best_base['win_rate']:.1f}%")
        print(f"    Omega: {best_base['omega']:.2f}")
        print(f"    Avg MFE opportunity: {best_base['avg_mfe']:.3f}%")
        print(f"    Energy capture rate: {best_base['energy_capture']:.1f}%")

    best_plus = None  # Initialize before conditional block
    if plus_results:
        best_plus = max(plus_results, key=lambda x: x['total_pnl'])
        print(f"\n  BERSERKER+ (High Flow + Volume):")
        print(f"    Strategy: TS {best_plus['ts']}%")
        print(f"    Total P&L: {best_plus['total_pnl']:+.1f}%")
        print(f"    Win rate: {best_plus['win_rate']:.1f}%")
        print(f"    Omega: {best_plus['omega']:.2f}")
        print(f"    Avg MFE opportunity: {best_plus['avg_mfe']:.3f}%")
        print(f"    Energy capture rate: {best_plus['energy_capture']:.1f}%")

    # Final recommendation
    print("\n  " + "-" * 50)
    print("  RECOMMENDED SETUP:")
    print("  " + "-" * 50)
    if plus_results and best_plus:
        print(f"    Signal: BERSERKER + High Flow + High Volume")
        print(f"    Direction: COUNTER-TREND (fade momentum)")
        print(f"    Exit: Trailing Stop {best_plus['ts']}%")
        print(f"    Expected: {best_plus['avg_pnl']:+.3f}% per trade, {best_plus['omega']:.2f} Omega")


if __name__ == "__main__":
    main()
