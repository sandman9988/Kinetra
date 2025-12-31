#!/usr/bin/env python3
"""
Exploration Batch Runner - Risk-Averse Config
Uses best discovered config: MAE_w=2.5, LR=0.05-0.08, 35 episodes
"""
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

# Add parent and tests to path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "tests"))

from rl_exploration_framework import (
    MultiInstrumentLoader,
    MultiInstrumentEnv,
    RewardShaper,
    LinearQAgent,
    FeatureTracker,
)
from test_physics_pipeline import get_rl_state_features, get_rl_feature_names


def run_batch(
    data_dir: str = "data/master",
    config_name: str = "RiskAverse_Optimal",
    mae_w: float = 2.5,
    lr: float = 0.05,
    gamma: float = 0.9,
    episodes: int = 35,
):
    """Run batch exploration with specified config."""
    print("=" * 70)
    print(f"EXPLORATION BATCH: {config_name}")
    print("=" * 70)
    print(f"  MAE_w={mae_w}, LR={lr}, Gamma={gamma}, Episodes={episodes}")

    # Load instruments
    loader = MultiInstrumentLoader(data_dir=data_dir, verbose=True)
    loader.load_all()

    if not loader.instruments:
        print("[ERROR] No instruments found!")
        return None

    print(f"\nLoaded {len(loader.instruments)} instruments")

    # Create reward shaper with optimal risk-averse config
    reward_shaper = RewardShaper(
        pnl_weight=1.0,
        edge_ratio_weight=0.3,
        mae_penalty_weight=mae_w,  # KEY: Penalize adverse excursion
        regime_bonus_weight=0.2,
        entropy_alignment_weight=0.1,
    )

    # Create multi-instrument env
    env = MultiInstrumentEnv(
        loader=loader,
        feature_extractor=get_rl_state_features,
        reward_shaper=reward_shaper,
        sampling_mode="round_robin",
    )

    # Create agent with optimal config
    agent = LinearQAgent(
        state_dim=env.state_dim,
        n_actions=env.n_actions,
        learning_rate=lr,
        gamma=gamma,
    )

    feature_names = get_rl_feature_names()
    tracker = FeatureTracker(feature_names)

    # Training loop
    all_rewards = []
    all_trades = []
    all_pnl = []
    per_instrument = {key: [] for key in loader.instruments}
    epsilon = 1.0

    for episode in range(episodes):
        state = env.reset()
        total_reward = 0

        for step in range(500):
            action = agent.select_action(state, epsilon)
            tracker.record(state, action)
            next_state, reward, done, info = env.step(action)
            agent.update(state, action, reward, next_state, done)
            total_reward += reward
            state = next_state
            if done:
                break

        epsilon = max(0.1, epsilon * 0.95)

        stats = env.get_episode_stats()
        all_rewards.append(total_reward)
        all_trades.append(stats["trades"])
        all_pnl.append(stats.get("total_pnl", 0))
        per_instrument[stats["instrument"]].append({
            "episode": episode,
            "reward": total_reward,
            "trades": stats["trades"],
            "pnl": stats.get("total_pnl", 0),
        })

        if (episode + 1) % 5 == 0:
            recent_r = np.mean(all_rewards[-5:])
            recent_pnl = np.mean(all_pnl[-5:])
            print(f"  Ep {episode+1:3d}/{episodes} | ε={epsilon:.3f} | R={recent_r:+8.2f} | PnL=${recent_pnl:+10,.0f}")

    # Compile results
    top_features = tracker.get_top_features_per_action(top_k=5)
    weights = agent.get_feature_weights()

    # Per-instrument summary
    inst_summary = {}
    for key, eps in per_instrument.items():
        if eps:
            inst_summary[key] = {
                "episodes": len(eps),
                "avg_reward": float(np.mean([e["reward"] for e in eps])),
                "avg_pnl": float(np.mean([e["pnl"] for e in eps])),
            }

    results = {
        "config": config_name,
        "mae_w": mae_w,
        "lr": lr,
        "gamma": gamma,
        "episodes": episodes,
        "instruments": list(loader.instruments.keys()),
        "total_reward": float(sum(all_rewards)),
        "avg_reward": float(np.mean(all_rewards)),
        "total_pnl": float(sum(all_pnl)),
        "avg_pnl": float(np.mean(all_pnl)),
        "per_instrument": inst_summary,
        "top_features": top_features,
    }

    # Print summary (PnL is now in PERCENTAGE terms for cross-instrument comparison)
    print("\n" + "=" * 70)
    print(f"RESULTS: {config_name} (PnL in PERCENTAGE terms)")
    print("=" * 70)
    print(f"  Total Reward: {results['total_reward']:+.1f}")
    print(f"  Avg Reward:   {results['avg_reward']:+.2f}")
    print(f"  Total PnL:    {results['total_pnl']:+.2f}%")
    print(f"  Avg PnL:      {results['avg_pnl']:+.2f}%")
    print("\nPer Instrument:")
    for key, stats in inst_summary.items():
        print(f"  {key}: R={stats['avg_reward']:+.2f}, PnL={stats['avg_pnl']:+.2f}%")

    print("\nTop Features Learned:")
    for action, features in top_features.items():
        top_3 = features[:3]
        top_str = ", ".join([f"{name}({corr:+.3f})" for name, corr in top_3])
        print(f"  {action}: {top_str}")

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    results_file = results_dir / f"batch_{config_name}_{timestamp}.json"

    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n[SAVED] {results_file}")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/master")
    parser.add_argument("--config", default="RiskAverse_Optimal")
    parser.add_argument("--mae-w", type=float, default=2.5)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--gamma", type=float, default=0.9)
    parser.add_argument("--episodes", type=int, default=35)
    args = parser.parse_args()

    run_batch(
        data_dir=args.data_dir,
        config_name=args.config,
        mae_w=args.mae_w,
        lr=args.lr,
        gamma=args.gamma,
        episodes=args.episodes,
    )
