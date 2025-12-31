#!/usr/bin/env python3
"""Quick SAC test on physics data."""

import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

# Check dependencies
try:
    from stable_baselines3 import SAC
    import gymnasium
    print("[OK] stable-baselines3 and gymnasium available")
except ImportError as e:
    print(f"[ERROR] Missing dependency: {e}")
    print("Install with: pip install stable-baselines3 gymnasium")
    sys.exit(1)

from rl_exploration_framework import (
    create_sb3_agent,
    PhysicsGymEnv,
    SB3_AVAILABLE,
    TrainingProgressCallback,
)

# Load test data
DATA_DIR = Path("/home/user/Kinetra/data/runs/berserker_run3/data/")

def load_first_dataset():
    """Load first available dataset."""
    csv_files = list(DATA_DIR.glob("*.csv"))
    if not csv_files:
        print(f"[ERROR] No CSV files in {DATA_DIR}")
        sys.exit(1)

    # Pick a medium-sized file
    csv_file = sorted(csv_files, key=lambda x: x.stat().st_size)[len(csv_files)//2]
    print(f"[LOAD] {csv_file.name}")

    df = pd.read_csv(csv_file)
    if 'time' in df.columns:
        df['time'] = pd.to_datetime(df['time'])

    print(f"  Bars: {len(df):,}")
    return df

def compute_physics_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute 64 physics features."""
    physics = pd.DataFrame(index=df.index)

    close = df['close'].values
    high = df['high'].values if 'high' in df else close
    low = df['low'].values if 'low' in df else close
    volume = df['volume'].values if 'volume' in df else np.ones(len(close))

    # Returns
    returns = np.diff(close, prepend=close[0]) / (close + 1e-10)
    log_returns = np.log(close / np.roll(close, 1).clip(1e-10))
    log_returns[0] = 0

    # Volatility measures
    for w in [5, 10, 20, 50]:
        physics[f'vol_{w}'] = pd.Series(returns).rolling(w, min_periods=1).std().values
        physics[f'ret_{w}'] = pd.Series(returns).rolling(w, min_periods=1).mean().values

    # Momentum
    for w in [5, 10, 20, 50]:
        physics[f'mom_{w}'] = (close - np.roll(close, w)) / (np.roll(close, w) + 1e-10)

    # RSI-like
    gains = np.maximum(returns, 0)
    losses = np.maximum(-returns, 0)
    for w in [5, 14, 20]:
        avg_gain = pd.Series(gains).rolling(w, min_periods=1).mean().values
        avg_loss = pd.Series(losses).rolling(w, min_periods=1).mean().values
        rs = avg_gain / (avg_loss + 1e-10)
        physics[f'rsi_{w}'] = 100 - 100 / (1 + rs)

    # Bollinger-like
    for w in [10, 20]:
        ma = pd.Series(close).rolling(w, min_periods=1).mean().values
        std = pd.Series(close).rolling(w, min_periods=1).std().values
        physics[f'bb_pct_{w}'] = (close - ma) / (2 * std + 1e-10)

    # Volume features
    vol_ma = pd.Series(volume).rolling(20, min_periods=1).mean().values
    physics['vol_ratio'] = volume / (vol_ma + 1e-10)
    physics['vol_z'] = (volume - vol_ma) / (pd.Series(volume).rolling(20).std().values + 1e-10)

    # Range features
    atr = pd.Series(high - low).rolling(14, min_periods=1).mean().values
    physics['atr_pct'] = atr / (close + 1e-10)
    physics['range_pct'] = (high - low) / (close + 1e-10)

    # Trend features
    for w in [5, 10, 20, 50]:
        ma = pd.Series(close).rolling(w, min_periods=1).mean().values
        physics[f'trend_{w}'] = (close - ma) / (ma + 1e-10)

    # Acceleration (2nd derivative)
    physics['accel'] = np.gradient(np.gradient(close))
    physics['accel_z'] = (physics['accel'] - physics['accel'].mean()) / (physics['accel'].std() + 1e-10)

    # Energy proxy
    physics['energy'] = returns**2
    physics['energy_ma'] = pd.Series(physics['energy']).rolling(20, min_periods=1).mean().values

    # Regime indicators
    physics['regime_bull'] = (returns > 0).astype(float)
    physics['regime_bear'] = (returns < 0).astype(float)
    physics['regime_streak'] = pd.Series(physics['regime_bull']).rolling(5, min_periods=1).sum().values

    # Fill to 64 dimensions
    while len(physics.columns) < 64:
        physics[f'pad_{len(physics.columns)}'] = 0.0

    # Take first 64
    physics = physics.iloc[:, :64]

    # Clean NaN
    physics = physics.fillna(0)

    return physics

def extract_features(physics_df: pd.DataFrame, bar: int) -> np.ndarray:
    """Extract 64-dim feature vector at bar."""
    if bar >= len(physics_df):
        bar = len(physics_df) - 1

    features = physics_df.iloc[bar].values.astype(np.float32)

    # Normalize
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    features = np.clip(features, -10, 10)

    return features

def main():
    print("=" * 60)
    print("SAC TEST ON PHYSICS DATA")
    print("=" * 60)

    # Load data
    df = load_first_dataset()

    # Compute physics
    print("\n[COMPUTE] Physics features...")
    physics_df = compute_physics_features(df)
    print(f"  Features: {len(physics_df.columns)}")

    # Create prices DataFrame
    prices_df = df[['close']].copy()
    if 'high' in df.columns:
        prices_df['high'] = df['high']
    if 'low' in df.columns:
        prices_df['low'] = df['low']

    # Create SAC agent
    print("\n[CREATE] SAC agent...")
    agent, env = create_sb3_agent(
        agent_type="SAC",
        physics_state=physics_df,
        prices=prices_df,
        feature_extractor=lambda ps, bar: extract_features(ps, bar),
        max_steps=500,
        risk_penalty=0.1,
        transaction_cost=0.0001,
    )
    print("  [OK] Agent initialized")

    # Train
    print("\n[TRAIN] SAC for 10,000 steps...")
    callback = TrainingProgressCallback(print_every=2000)
    agent.train(total_timesteps=10000, callback=callback)

    # Evaluate
    print("\n[EVAL] Running 5 test episodes...")
    test_rewards = []
    test_pnls = []

    for ep in range(5):
        obs, _ = env.reset()
        episode_reward = 0

        for _ in range(env.max_steps):
            action = agent.get_continuous_action(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(np.array([action]))
            episode_reward += reward

            if terminated or truncated:
                break

        test_rewards.append(episode_reward)
        test_pnls.append(info.get('total_pnl', 0))
        print(f"  Episode {ep+1}: Reward={episode_reward:+.2f}, PnL=${info.get('total_pnl', 0):+,.0f}")

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Avg Reward: {np.mean(test_rewards):+.2f} ± {np.std(test_rewards):.2f}")
    print(f"Avg PnL:    ${np.mean(test_pnls):+,.0f}")
    print("=" * 60)

if __name__ == "__main__":
    main()
