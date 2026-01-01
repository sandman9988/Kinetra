#!/usr/bin/env python3
"""
Trade Management Analysis for Berserker Entries

Analyze how much of the energy release we can capture:
- MAE (Maximum Adverse Excursion) - worst drawdown during trade
- MFE (Maximum Favorable Excursion) - best profit during trade
- Trailing Stop optimization
- Energy capture efficiency
- Omega ratio for risk-adjusted performance
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass

sys.path.insert(0, str(Path(__file__).parent.parent))

from kinetra.mt5_connector import load_csv_data
from kinetra.physics_engine import PhysicsEngine


@dataclass
class TradeResult:
    """Result of a single trade."""
    entry_bar: int
    exit_bar: int
    direction: int  # 1 = long, -1 = short
    entry_price: float
    exit_price: float
    pnl: float
    pnl_pct: float
    mae: float  # Maximum Adverse Excursion (worst drawdown)
    mfe: float  # Maximum Favorable Excursion (best profit)
    mae_pct: float
    mfe_pct: float
    bars_held: int
    capture_efficiency: float  # pnl / mfe (how much of the move we captured)
    energy_at_entry: float


def compute_mae_mfe(
    df: pd.DataFrame,
    entry_bar: int,
    direction: int,
    hold_bars: int = 10,
) -> Tuple[float, float, float, float]:
    """
    Compute MAE and MFE for a trade.

    Args:
        df: DataFrame with OHLC
        entry_bar: Entry bar index
        direction: 1 for long, -1 for short
        hold_bars: How many bars to analyze

    Returns:
        (mae, mfe, mae_pct, mfe_pct)
    """
    entry_price = df.iloc[entry_bar]['close']
    end_bar = min(entry_bar + hold_bars, len(df) - 1)

    # Get price path
    highs = df.iloc[entry_bar:end_bar + 1]['high'].values
    lows = df.iloc[entry_bar:end_bar + 1]['low'].values
    closes = df.iloc[entry_bar:end_bar + 1]['close'].values

    if direction == 1:  # Long
        # MAE = worst low relative to entry
        mae = entry_price - lows.min()
        # MFE = best high relative to entry
        mfe = highs.max() - entry_price
    else:  # Short
        # MAE = worst high relative to entry
        mae = highs.max() - entry_price
        # MFE = best low relative to entry
        mfe = entry_price - lows.min()

    mae_pct = mae / entry_price * 100
    mfe_pct = mfe / entry_price * 100

    return mae, mfe, mae_pct, mfe_pct


def simulate_trade_with_stops(
    df: pd.DataFrame,
    entry_bar: int,
    direction: int,
    stop_loss_pct: float = None,
    take_profit_pct: float = None,
    trailing_stop_pct: float = None,
    max_bars: int = 20,
) -> TradeResult:
    """
    Simulate a trade with stop loss, take profit, and/or trailing stop.

    Args:
        df: DataFrame with OHLC and energy
        entry_bar: Entry bar index
        direction: 1 for long, -1 for short
        stop_loss_pct: Fixed stop loss percentage
        take_profit_pct: Fixed take profit percentage
        trailing_stop_pct: Trailing stop percentage (from peak)
        max_bars: Maximum bars to hold

    Returns:
        TradeResult
    """
    entry_price = df.iloc[entry_bar]['close']
    energy_at_entry = df.iloc[entry_bar].get('energy_pct', 0.5)

    best_price = entry_price
    worst_price = entry_price
    exit_bar = entry_bar
    exit_price = entry_price
    exit_reason = 'max_bars'

    # Track MAE/MFE
    mae = 0
    mfe = 0

    for i in range(1, max_bars + 1):
        bar_idx = entry_bar + i
        if bar_idx >= len(df):
            break

        high = df.iloc[bar_idx]['high']
        low = df.iloc[bar_idx]['low']
        close = df.iloc[bar_idx]['close']

        # Update best/worst
        if direction == 1:  # Long
            best_price = max(best_price, high)
            worst_price = min(worst_price, low)
            current_pnl_pct = (close - entry_price) / entry_price * 100
            mfe = max(mfe, (high - entry_price) / entry_price * 100)
            mae = max(mae, (entry_price - low) / entry_price * 100)
        else:  # Short
            best_price = min(best_price, low)
            worst_price = max(worst_price, high)
            current_pnl_pct = (entry_price - close) / entry_price * 100
            mfe = max(mfe, (entry_price - low) / entry_price * 100)
            mae = max(mae, (high - entry_price) / entry_price * 100)

        # Check stop loss
        if stop_loss_pct is not None:
            if direction == 1 and low <= entry_price * (1 - stop_loss_pct / 100):
                exit_price = entry_price * (1 - stop_loss_pct / 100)
                exit_bar = bar_idx
                exit_reason = 'stop_loss'
                break
            elif direction == -1 and high >= entry_price * (1 + stop_loss_pct / 100):
                exit_price = entry_price * (1 + stop_loss_pct / 100)
                exit_bar = bar_idx
                exit_reason = 'stop_loss'
                break

        # Check take profit
        if take_profit_pct is not None:
            if direction == 1 and high >= entry_price * (1 + take_profit_pct / 100):
                exit_price = entry_price * (1 + take_profit_pct / 100)
                exit_bar = bar_idx
                exit_reason = 'take_profit'
                break
            elif direction == -1 and low <= entry_price * (1 - take_profit_pct / 100):
                exit_price = entry_price * (1 - take_profit_pct / 100)
                exit_bar = bar_idx
                exit_reason = 'take_profit'
                break

        # Check trailing stop
        if trailing_stop_pct is not None:
            if direction == 1:
                trail_stop = best_price * (1 - trailing_stop_pct / 100)
                if low <= trail_stop:
                    exit_price = trail_stop
                    exit_bar = bar_idx
                    exit_reason = 'trailing_stop'
                    break
            else:
                trail_stop = best_price * (1 + trailing_stop_pct / 100)
                if high >= trail_stop:
                    exit_price = trail_stop
                    exit_bar = bar_idx
                    exit_reason = 'trailing_stop'
                    break

        exit_bar = bar_idx
        exit_price = close

    # Calculate P&L
    if direction == 1:
        pnl = exit_price - entry_price
    else:
        pnl = entry_price - exit_price

    pnl_pct = pnl / entry_price * 100

    # Capture efficiency: how much of the possible move we captured
    capture_efficiency = pnl_pct / mfe if mfe > 0 else 0

    return TradeResult(
        entry_bar=entry_bar,
        exit_bar=exit_bar,
        direction=direction,
        entry_price=entry_price,
        exit_price=exit_price,
        pnl=pnl,
        pnl_pct=pnl_pct,
        mae=mae * direction,  # Keep sign for analysis
        mfe=mfe,
        mae_pct=mae,
        mfe_pct=mfe,
        bars_held=exit_bar - entry_bar,
        capture_efficiency=capture_efficiency,
        energy_at_entry=energy_at_entry,
    )


def calculate_omega_ratio(returns: List[float], threshold: float = 0) -> float:
    """
    Calculate Omega ratio.

    Omega = sum(returns > threshold) / |sum(returns < threshold)|

    Higher is better. > 1 means more gains than losses.
    """
    gains = sum(r for r in returns if r > threshold)
    losses = abs(sum(r for r in returns if r < threshold))

    if losses == 0:
        return float('inf') if gains > 0 else 0

    return gains / losses


def analyze_berserker_trades(df: pd.DataFrame) -> Dict:
    """Analyze MAE/MFE profile of berserker trades."""
    # Identify berserker bars
    berserker = (df['energy_pct'] > 0.75) & (df['damping_pct'] < 0.25)

    # Get momentum for direction
    df['momentum_5'] = df['close'].pct_change(5)
    df['counter_direction'] = -np.sign(df['momentum_5'])

    berserker_bars = df[berserker].index.tolist()
    print(f"Analyzing {len(berserker_bars)} berserker trades...")

    # Analyze each trade
    trades = []
    for bar in berserker_bars:
        bar_idx = df.index.get_loc(bar)
        if bar_idx + 20 >= len(df):
            continue

        direction = int(df.loc[bar, 'counter_direction'])
        if direction == 0:
            continue

        mae, mfe, mae_pct, mfe_pct = compute_mae_mfe(
            df, bar_idx, direction, hold_bars=10
        )

        trades.append({
            'bar': bar,
            'direction': direction,
            'mae_pct': mae_pct,
            'mfe_pct': mfe_pct,
            'mfe_mae_ratio': mfe_pct / mae_pct if mae_pct > 0 else float('inf'),
        })

    trades_df = pd.DataFrame(trades)

    return {
        'n_trades': len(trades_df),
        'avg_mae_pct': trades_df['mae_pct'].mean(),
        'avg_mfe_pct': trades_df['mfe_pct'].mean(),
        'median_mae_pct': trades_df['mae_pct'].median(),
        'median_mfe_pct': trades_df['mfe_pct'].median(),
        'avg_mfe_mae_ratio': trades_df['mfe_mae_ratio'].replace([np.inf, -np.inf], np.nan).mean(),
        'p25_mae': trades_df['mae_pct'].quantile(0.25),
        'p75_mae': trades_df['mae_pct'].quantile(0.75),
        'p25_mfe': trades_df['mfe_pct'].quantile(0.25),
        'p75_mfe': trades_df['mfe_pct'].quantile(0.75),
    }


def test_stop_strategies(df: pd.DataFrame) -> list[Any]:
    """Test different stop loss / trailing stop strategies."""
    berserker = (df['energy_pct'] > 0.75) & (df['damping_pct'] < 0.25)
    df['momentum_5'] = df['close'].pct_change(5)
    df['counter_direction'] = -np.sign(df['momentum_5'])

    berserker_bars = df[berserker].index.tolist()

    strategies = [
        {'name': 'No stops (10 bars)', 'sl': None, 'tp': None, 'ts': None},
        {'name': 'SL 0.5%', 'sl': 0.5, 'tp': None, 'ts': None},
        {'name': 'SL 1.0%', 'sl': 1.0, 'tp': None, 'ts': None},
        {'name': 'TP 0.5%', 'sl': None, 'tp': 0.5, 'ts': None},
        {'name': 'TP 1.0%', 'sl': None, 'tp': 1.0, 'ts': None},
        {'name': 'TS 0.3%', 'sl': None, 'tp': None, 'ts': 0.3},
        {'name': 'TS 0.5%', 'sl': None, 'tp': None, 'ts': 0.5},
        {'name': 'TS 0.75%', 'sl': None, 'tp': None, 'ts': 0.75},
        {'name': 'SL 0.5% + TS 0.5%', 'sl': 0.5, 'tp': None, 'ts': 0.5},
        {'name': 'SL 0.75% + TS 0.5%', 'sl': 0.75, 'tp': None, 'ts': 0.5},
        {'name': 'SL 1% + TS 0.5%', 'sl': 1.0, 'tp': None, 'ts': 0.5},
        {'name': 'SL 0.5% + TP 1%', 'sl': 0.5, 'tp': 1.0, 'ts': None},
        {'name': 'SL 0.75% + TP 1.5%', 'sl': 0.75, 'tp': 1.5, 'ts': None},
    ]

    results = []
    for strat in strategies:
        trades = []
        for bar in berserker_bars:
            bar_idx = df.index.get_loc(bar)
            if bar_idx + 20 >= len(df):
                continue

            direction = int(df.loc[bar, 'counter_direction'])
            if direction == 0:
                continue

            trade = simulate_trade_with_stops(
                df, bar_idx, direction,
                stop_loss_pct=strat['sl'],
                take_profit_pct=strat['tp'],
                trailing_stop_pct=strat['ts'],
                max_bars=10,
            )
            trades.append(trade)

        if not trades:
            continue

        pnls = [t.pnl_pct for t in trades]

        # Efficiency: for winning trades, how much of MFE did we capture?
        winning_trades = [t for t in trades if t.pnl_pct > 0 and t.mfe_pct > 0]
        efficiencies = [t.pnl_pct / t.mfe_pct * 100 for t in winning_trades]

        win_rate = sum(1 for p in pnls if p > 0) / len(pnls) * 100
        avg_pnl = np.mean(pnls)
        total_pnl = sum(pnls)
        omega = calculate_omega_ratio(pnls)
        avg_efficiency = np.mean(efficiencies) if efficiencies else 0
        avg_bars = np.mean([t.bars_held for t in trades])

        # Profit factor
        wins = sum(p for p in pnls if p > 0)
        losses = abs(sum(p for p in pnls if p < 0))
        profit_factor = wins / losses if losses > 0 else float('inf')

        results.append({
            'strategy': strat['name'],
            'trades': len(trades),
            'win_rate': win_rate,
            'avg_pnl': avg_pnl,
            'total_pnl': total_pnl,
            'omega': omega,
            'profit_factor': profit_factor,
            'capture_efficiency': avg_efficiency * 100,
            'avg_bars': avg_bars,
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
    df = df.dropna()

    # === MAE/MFE ANALYSIS ===
    print("\n" + "=" * 70)
    print("MAE/MFE ANALYSIS (Counter-trend Berserker Trades)")
    print("=" * 70)

    stats = analyze_berserker_trades(df)

    print(f"\n  Trades analyzed: {stats['n_trades']}")
    print(f"\n  Maximum Adverse Excursion (MAE) - worst drawdown:")
    print(f"    Average: {stats['avg_mae_pct']:.3f}%")
    print(f"    Median:  {stats['median_mae_pct']:.3f}%")
    print(f"    P25-P75: {stats['p25_mae']:.3f}% - {stats['p75_mae']:.3f}%")
    print(f"\n  Maximum Favorable Excursion (MFE) - best profit:")
    print(f"    Average: {stats['avg_mfe_pct']:.3f}%")
    print(f"    Median:  {stats['median_mfe_pct']:.3f}%")
    print(f"    P25-P75: {stats['p25_mfe']:.3f}% - {stats['p75_mfe']:.3f}%")
    print(f"\n  MFE/MAE Ratio: {stats['avg_mfe_mae_ratio']:.2f}x")
    print(f"    (Higher = better reward/risk profile)")

    # === STOP STRATEGY COMPARISON ===
    print("\n" + "=" * 70)
    print("STOP STRATEGY COMPARISON")
    print("(Testing different SL/TP/TS combinations)")
    print("=" * 70)

    results = test_stop_strategies(df)

    # Sort by Omega ratio
    results = sorted(results, key=lambda x: x['omega'], reverse=True)

    print("\n  Strategy Comparison (sorted by Omega ratio):\n")
    print(f"  {'Strategy':<25} {'Trades':>7} {'Win%':>7} {'AvgPnL':>8} {'Total':>8} {'Omega':>7} {'PF':>6} {'Eff%':>6}")
    print("  " + "-" * 85)

    for r in results:
        omega_str = f"{r['omega']:.2f}" if r['omega'] != float('inf') else "inf"
        pf_str = f"{r['profit_factor']:.2f}" if r['profit_factor'] != float('inf') else "inf"
        print(f"  {r['strategy']:<25} {r['trades']:>7} {r['win_rate']:>6.1f}% "
              f"{r['avg_pnl']:>+7.3f}% {r['total_pnl']:>+7.1f}% "
              f"{omega_str:>7} {pf_str:>6} {r['capture_efficiency']:>5.1f}%")

    # === OPTIMAL STRATEGY ===
    print("\n" + "=" * 70)
    print("OPTIMAL STRATEGY ANALYSIS")
    print("=" * 70)

    # Find best by different criteria
    best_omega = max(results, key=lambda x: x['omega'] if x['omega'] != float('inf') else 0)
    best_total = max(results, key=lambda x: x['total_pnl'])
    best_efficiency = max(results, key=lambda x: x['capture_efficiency'])

    print(f"\n  Best Omega ratio: {best_omega['strategy']}")
    print(f"    Omega: {best_omega['omega']:.2f}, Win rate: {best_omega['win_rate']:.1f}%")

    print(f"\n  Best total P&L: {best_total['strategy']}")
    print(f"    Total: {best_total['total_pnl']:+.1f}%, Avg: {best_total['avg_pnl']:+.3f}%")

    print(f"\n  Best capture efficiency: {best_efficiency['strategy']}")
    print(f"    Efficiency: {best_efficiency['capture_efficiency']:.1f}% of MFE captured")

    # === ENERGY RECOVERY SUMMARY ===
    print("\n" + "=" * 70)
    print("ENERGY RECOVERY SUMMARY")
    print("=" * 70)

    print(f"""
  Physics-based berserker entries offer:

  1. Average MFE (max profit opportunity): {stats['avg_mfe_pct']:.3f}%
  2. Average MAE (max drawdown risk):      {stats['avg_mae_pct']:.3f}%
  3. Reward/Risk ratio:                    {stats['avg_mfe_mae_ratio']:.2f}x

  Best capture strategy: {best_total['strategy']}
  - Captures {best_total['capture_efficiency']:.1f}% of available move
  - Win rate: {best_total['win_rate']:.1f}%
  - Omega ratio: {best_total['omega']:.2f}
    """)


if __name__ == "__main__":
    main()
