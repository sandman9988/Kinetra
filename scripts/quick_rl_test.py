#!/usr/bin/env python3
"""Quick RL test with all physics features."""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from kinetra.mt5_connector import load_csv_data
from kinetra.rl_physics_env import PhysicsTradingEnv, train_rl_agent

# Load data
project_root = Path(__file__).parent.parent
csv_files = list(project_root.glob("*BTCUSD*.csv"))
if not csv_files:
    print("No BTCUSD CSV file found")
    sys.exit(1)

data = load_csv_data(str(csv_files[0]))
print(f"Loaded {len(data)} bars")

# Split 70/30
split = int(len(data) * 0.7)
train_df = data.iloc[:split].copy().reset_index(drop=True)
test_df = data.iloc[split:].copy().reset_index(drop=True)

print(f"Train: {len(train_df)} bars, Test: {len(test_df)} bars")

# Train
print("\nTraining RL agent on physics features...")
agent, env, stats = train_rl_agent(train_df, n_episodes=30, verbose=True)

# Get feature importance
feature_names = [c.replace('_pct', '') for c in env.feature_cols] + ['pos_dir', 'bars_held', 'pnl']
importance_df = agent.get_feature_importance(feature_names)

print("\n" + "=" * 60)
print("FEATURE IMPORTANCE (What RL learned)")
print("=" * 60)

for _, row in importance_df.head(10).iterrows():
    print(f"  {row['feature']:>20}: {row['importance']:.1%}")

# Test on unseen data
print("\n" + "=" * 60)
print("OUT-OF-SAMPLE TEST")
print("=" * 60)

test_env = PhysicsTradingEnv(test_df)
state = test_env.reset()

while not test_env.done:
    action = agent.select_action(state, training=False)
    state, reward, done, info = test_env.step(action)

test_stats = test_env.get_episode_stats()
print(f"\n  Trades: {test_stats['n_trades']}")
print(f"  Win Rate: {test_stats['win_rate']:.1%}")
print(f"  Total P&L: {test_stats['total_pnl']:.2f}%")
print(f"  Profit Factor: {test_stats['profit_factor']:.2f}")
print(f"  Avg MFE: {test_stats['avg_mfe']:.3f}%")
print(f"  Avg MAE: {test_stats['avg_mae']:.3f}%")
