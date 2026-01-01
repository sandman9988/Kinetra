#!/usr/bin/env python3
"""
Train RL Agent on Physics Features

Cross-validation across:
1. Multiple instruments (if available)
2. Multiple timeframes
3. Walk-forward out-of-sample testing

Avoid curve fitting by:
- Training on one period, testing on another
- Training on one instrument, testing on others
- Using only percentile features (adaptive, no fixed thresholds)
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

from kinetra.mt5_connector import load_csv_data
from kinetra.rl_physics_env import (
    PhysicsTradingEnv, SimpleRLAgent, train_rl_agent, evaluate_agent
)


def load_all_data() -> Dict[str, pd.DataFrame]:
    """Load all available CSV data files."""
    project_root = Path(__file__).parent.parent
    csv_files = list(project_root.glob("*.csv"))

    data = {}
    for f in csv_files:
        name = f.stem
        try:
            df = load_csv_data(str(f))
            if len(df) > 500:  # Minimum required
                data[name] = df
                print(f"  Loaded {name}: {len(df)} bars")
        except Exception as e:
            print(f"  Failed to load {name}: {e}")

    return data


def split_data(df: pd.DataFrame, train_ratio: float = 0.7) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split data into train/test chronologically."""
    split_idx = int(len(df) * train_ratio)
    return df.iloc[:split_idx].copy(), df.iloc[split_idx:].copy()


def walk_forward_validation(
    df: pd.DataFrame,
    n_folds: int = 5,
    train_episodes: int = 50,
    verbose: bool = True
) -> List[Dict]:
    """
    Walk-forward validation: train on fold N, test on fold N+1.

    This simulates real trading where we train on past, trade on future.
    """
    fold_size = len(df) // (n_folds + 1)
    results = []

    for fold in range(n_folds):
        train_start = fold * fold_size
        train_end = (fold + 1) * fold_size
        test_start = train_end
        test_end = (fold + 2) * fold_size

        train_df = df.iloc[train_start:train_end].copy().reset_index(drop=True)
        test_df = df.iloc[test_start:test_end].copy().reset_index(drop=True)

        if len(train_df) < 200 or len(test_df) < 100:
            continue

        if verbose:
            print(f"\n  Fold {fold + 1}/{n_folds}: Train {len(train_df)} bars, Test {len(test_df)} bars")

        # Train
        agent, _, train_stats = train_rl_agent(train_df, n_episodes=train_episodes, verbose=False)

        # Get final training stats
        final_train = train_stats[-1] if train_stats else {}

        # Test (out of sample)
        test_stats, test_trades = evaluate_agent(agent, test_df)

        results.append({
            'fold': fold + 1,
            'train_bars': len(train_df),
            'test_bars': len(test_df),
            'train_trades': final_train.get('n_trades', 0),
            'train_win_rate': final_train.get('win_rate', 0),
            'train_pnl': final_train.get('total_pnl', 0),
            'test_trades': test_stats.get('n_trades', 0),
            'test_win_rate': test_stats.get('win_rate', 0),
            'test_pnl': test_stats.get('total_pnl', 0),
            'test_reward': test_stats.get('total_reward', 0),
            'test_pf': test_stats.get('profit_factor', 0),
        })

        if verbose:
            print(f"    Train: {final_train.get('n_trades', 0)} trades, "
                  f"{final_train.get('win_rate', 0):.1%} WR, "
                  f"{final_train.get('total_pnl', 0):.2f}% PnL")
            print(f"    Test:  {test_stats.get('n_trades', 0)} trades, "
                  f"{test_stats.get('win_rate', 0):.1%} WR, "
                  f"{test_stats.get('total_pnl', 0):.2f}% PnL")

    return results


def cross_instrument_validation(
    data: Dict[str, pd.DataFrame],
    train_episodes: int = 50,
    verbose: bool = True
) -> List[Dict]:
    """
    Cross-instrument validation: train on one instrument, test on others.

    This tests if the physics features generalize across markets.
    """
    instruments = list(data.keys())
    if len(instruments) < 2:
        print("  Need at least 2 instruments for cross-validation")
        return []

    results = []

    for train_instr in instruments:
        train_df = data[train_instr].copy().reset_index(drop=True)

        if verbose:
            print(f"\n  Training on {train_instr} ({len(train_df)} bars)")

        # Train
        agent, _, train_stats = train_rl_agent(train_df, n_episodes=train_episodes, verbose=False)
        final_train = train_stats[-1] if train_stats else {}

        # Test on other instruments
        for test_instr in instruments:
            if test_instr == train_instr:
                continue

            test_df = data[test_instr].copy().reset_index(drop=True)
            test_stats, _ = evaluate_agent(agent, test_df)

            results.append({
                'train_instrument': train_instr,
                'test_instrument': test_instr,
                'train_bars': len(train_df),
                'test_bars': len(test_df),
                'train_trades': final_train.get('n_trades', 0),
                'train_win_rate': final_train.get('win_rate', 0),
                'train_pnl': final_train.get('total_pnl', 0),
                'test_trades': test_stats.get('n_trades', 0),
                'test_win_rate': test_stats.get('win_rate', 0),
                'test_pnl': test_stats.get('total_pnl', 0),
                'test_pf': test_stats.get('profit_factor', 0),
            })

            if verbose:
                print(f"    Test on {test_instr}: "
                      f"{test_stats.get('n_trades', 0)} trades, "
                      f"{test_stats.get('win_rate', 0):.1%} WR, "
                      f"{test_stats.get('total_pnl', 0):.2f}% PnL")

    return results


def analyze_feature_importance(
    df: pd.DataFrame,
    n_episodes: int = 100
) -> pd.DataFrame:
    """Train agent and analyze which features it learned to use."""
    print("\n  Training agent to analyze feature importance...")

    agent, env, _ = train_rl_agent(df, n_episodes=n_episodes, verbose=False)

    # Get feature names
    feature_names = [c.replace('_pct', '') for c in env.feature_cols] + ['pos_dir', 'bars_held', 'pnl']

    return agent.get_feature_importance(feature_names)


def main():
    print("=" * 80)
    print("RL PHYSICS TRAINING - CROSS VALIDATION")
    print("=" * 80)

    # Load all data
    print("\nLoading data...")
    data = load_all_data()

    if not data:
        print("No data files found!")
        return

    # Single instrument walk-forward (time-based)
    print("\n" + "=" * 80)
    print("WALK-FORWARD VALIDATION (Time-based)")
    print("Trains on past, tests on future - simulates real trading")
    print("=" * 80)

    for name, df in data.items():
        print(f"\n{name}:")
        results = walk_forward_validation(df, n_folds=5, train_episodes=50)

        if results:
            # Aggregate results
            avg_train_wr = np.mean([r['train_win_rate'] for r in results])
            avg_test_wr = np.mean([r['test_win_rate'] for r in results])
            avg_train_pnl = np.mean([r['train_pnl'] for r in results])
            avg_test_pnl = np.mean([r['test_pnl'] for r in results])

            print(f"\n  SUMMARY:")
            print(f"    Avg Train WR: {avg_train_wr:.1%}")
            print(f"    Avg Test WR:  {avg_test_wr:.1%}")
            print(f"    Avg Train PnL: {avg_train_pnl:.2f}%")
            print(f"    Avg Test PnL:  {avg_test_pnl:.2f}%")

            # Check for overfitting
            wr_decay = avg_train_wr - avg_test_wr
            pnl_decay = avg_train_pnl - avg_test_pnl

            if wr_decay > 0.1 or pnl_decay > 5:
                print(f"    ⚠️  OVERFITTING DETECTED (WR decay: {wr_decay:.1%}, PnL decay: {pnl_decay:.1f}%)")
            else:
                print(f"    ✓ Generalizes well (WR decay: {wr_decay:.1%}, PnL decay: {pnl_decay:.1f}%)")

    # Cross-instrument validation
    if len(data) >= 2:
        print("\n" + "=" * 80)
        print("CROSS-INSTRUMENT VALIDATION")
        print("Trains on one instrument, tests on others - tests feature generalization")
        print("=" * 80)

        results = cross_instrument_validation(data, train_episodes=50)

        if results:
            print("\n  CROSS-INSTRUMENT MATRIX:")
            print(f"  {'Train→Test':>30} │ {'Test WR':>10} │ {'Test PnL':>10}")
            print("  " + "─" * 60)

            for r in results:
                pair = f"{r['train_instrument']} → {r['test_instrument']}"
                print(f"  {pair:>30} │ {r['test_win_rate']:>9.1%} │ {r['test_pnl']:>+9.2f}%")

            # Average cross-instrument performance
            avg_cross_wr = np.mean([r['test_win_rate'] for r in results])
            avg_cross_pnl = np.mean([r['test_pnl'] for r in results])

            print(f"\n  AVERAGE CROSS-INSTRUMENT:")
            print(f"    Win Rate: {avg_cross_wr:.1%}")
            print(f"    P&L: {avg_cross_pnl:+.2f}%")

    # Feature importance
    print("\n" + "=" * 80)
    print("FEATURE IMPORTANCE (What did RL learn?)")
    print("=" * 80)

    # Use largest dataset
    largest = max(data.items(), key=lambda x: len(x[1]))
    importance = analyze_feature_importance(largest[1], n_episodes=100)

    print(f"\n  Trained on {largest[0]}:")
    print(f"\n  {'Feature':>20} │ {'Importance':>12}")
    print("  " + "─" * 40)

    for _, row in importance.head(10).iterrows():
        print(f"  {row['feature']:>20} │ {row['importance']:>11.1%}")

    print("\n  TOP FEATURES (RL learned formula):")
    top_features = importance.head(5)['feature'].tolist()
    for i, f in enumerate(top_features, 1):
        print(f"    {i}. {f}")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("""
  CROSS-VALIDATION APPROACH:

  1. WALK-FORWARD (Time):
     - Train on past periods, test on future
     - Simulates real trading conditions
     - Checks if strategy decays over time

  2. CROSS-INSTRUMENT:
     - Train on one market, test on another
     - Tests if physics features generalize
     - Avoids market-specific curve fitting

  3. FEATURE IMPORTANCE:
     - Shows what RL learned to prioritize
     - Reveals emergent trading formula
     - Physics features weighted by predictive power

  If both validations pass (low decay), the physics
  features genuinely capture market dynamics.
""")


if __name__ == "__main__":
    main()
