#!/usr/bin/env python3
"""
Interactive Exploration Training
=================================

One unified process:
1. Check available data
2. Show what we have
3. Confirm to proceed
4. Run exploration training
5. Show results

Usage:
    python scripts/explore_interactive.py
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
import json

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from rl_exploration_framework import (
    MultiInstrumentLoader,
    MultiInstrumentEnv,
    RewardShaper,
    LinearQAgent,
    FeatureTracker,
)


def get_rl_state_features(physics_state: dict) -> np.ndarray:
    """Extract RL state features from physics state."""
    features = [
        physics_state.get('energy', 0),
        physics_state.get('entropy', 0),
        physics_state.get('damping', 0),
        physics_state.get('energy_percentile', 0.5),
        physics_state.get('entropy_percentile', 0.5),
        physics_state.get('regime_confidence', 0),
    ]
    return np.array(features, dtype=np.float32)


def get_rl_feature_names() -> list:
    """Get feature names for tracking."""
    return ['energy', 'entropy', 'damping', 'energy_pct', 'entropy_pct', 'regime_conf']


def print_header(text):
    """Print section header."""
    print("\n" + "="*80)
    print(f"  {text}")
    print("="*80)


def print_step(step_num, text):
    """Print step header."""
    print(f"\n[STEP {step_num}] {text}")
    print("-" * 80)


def main():
    """Run interactive exploration training."""

    print_header("KINETRA INTERACTIVE EXPLORATION")

    data_dir = Path("data/master")

    # ========================================================================
    # STEP 1: CHECK AVAILABLE DATA
    # ========================================================================
    print_step(1, "Checking available data...")

    if not data_dir.exists():
        print(f"\n❌ Data directory not found: {data_dir}")
        print("   Run download script first to get market data")
        return

    csv_files = list(data_dir.glob("*.csv"))

    if not csv_files:
        print(f"\n❌ No CSV files found in {data_dir}")
        print("   Run download script to get market data")
        return

    print(f"\n✅ Found {len(csv_files)} data files")

    # Group by symbol and timeframe
    symbols = {}
    for f in csv_files:
        parts = f.stem.split('_')
        if len(parts) >= 2:
            symbol = parts[0]
            timeframe = parts[1]
            if symbol not in symbols:
                symbols[symbol] = []
            symbols[symbol].append(timeframe)

    print(f"\nAvailable symbols ({len(symbols)}):")
    for symbol, timeframes in sorted(symbols.items())[:20]:  # Show first 20
        tf_str = ', '.join(sorted(set(timeframes)))
        print(f"  {symbol:15s} : {tf_str}")

    if len(symbols) > 20:
        print(f"  ... and {len(symbols) - 20} more symbols")

    # ========================================================================
    # STEP 2: CONFIRM TO PROCEED
    # ========================================================================
    print_step(2, "Ready to train exploration agent")

    print("\nTraining configuration:")
    print("  • Multi-instrument exploration")
    print("  • Physics-based features (energy, entropy, damping)")
    print("  • Linear Q-learning agent")
    print("  • Risk-averse reward shaping (MAE penalty)")
    print("  • 35 episodes (default)")

    response = input("\nProceed with training? [Y/n]: ").strip().lower()
    if response and response not in ['y', 'yes']:
        print("\n⚠️  Training cancelled by user")
        return

    # ========================================================================
    # STEP 3: LOAD DATA
    # ========================================================================
    print_step(3, "Loading instruments...")

    loader = MultiInstrumentLoader(data_dir=str(data_dir), verbose=True)
    loader.load_all()

    if not loader.instruments:
        print("\n❌ Failed to load any instruments")
        return

    print(f"\n✅ Loaded {len(loader.instruments)} instruments")
    for key, data in list(loader.instruments.items())[:10]:
        print(f"  {key}: {len(data)} bars")

    if len(loader.instruments) > 10:
        print(f"  ... and {len(loader.instruments) - 10} more instruments")

    # ========================================================================
    # STEP 4: SETUP TRAINING
    # ========================================================================
    print_step(4, "Setting up exploration training...")

    # Reward shaper with risk-averse config
    reward_shaper = RewardShaper(
        pnl_weight=1.0,
        edge_ratio_weight=0.3,
        mae_penalty_weight=2.5,  # Penalize adverse excursion
        regime_bonus_weight=0.2,
        entropy_alignment_weight=0.1,
    )

    # Multi-instrument environment
    env = MultiInstrumentEnv(
        loader=loader,
        feature_extractor=get_rl_state_features,
        reward_shaper=reward_shaper,
        sampling_mode="round_robin",
    )

    # Linear Q agent
    agent = LinearQAgent(
        state_dim=env.state_dim,
        n_actions=env.n_actions,
        learning_rate=0.05,
        gamma=0.9,
    )

    # Feature tracker
    feature_names = get_rl_feature_names()
    tracker = FeatureTracker(feature_names)

    print(f"\n✅ Training setup complete")
    print(f"  State dimension: {env.state_dim}")
    print(f"  Actions: {env.n_actions} (HOLD, LONG, SHORT, CLOSE)")
    print(f"  Instruments: {len(loader.instruments)}")

    # ========================================================================
    # STEP 5: RUN TRAINING
    # ========================================================================
    print_step(5, "Running exploration training...")

    episodes = 35
    all_rewards = []
    all_pnl = []
    per_instrument = {key: [] for key in loader.instruments}
    epsilon = 1.0

    print(f"\nTraining for {episodes} episodes...")
    print(f"{'Episode':>8s} {'ε':>8s} {'Reward':>12s} {'PnL':>15s} {'Instrument':>15s}")
    print("-" * 80)

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
        all_pnl.append(stats.get("total_pnl", 0))

        instrument = stats.get("instrument", "unknown")
        per_instrument[instrument].append({
            "episode": episode,
            "reward": total_reward,
            "pnl": stats.get("total_pnl", 0),
        })

        # Print every episode
        if (episode + 1) % 5 == 0 or episode < 5:
            recent_r = np.mean(all_rewards[-5:]) if len(all_rewards) >= 5 else np.mean(all_rewards)
            recent_pnl = np.mean(all_pnl[-5:]) if len(all_pnl) >= 5 else np.mean(all_pnl)
            print(f"{episode+1:8d} {epsilon:8.3f} {recent_r:+12.2f} ${recent_pnl:+14,.0f} {instrument:>15s}")

    # ========================================================================
    # STEP 6: SHOW RESULTS
    # ========================================================================
    print_step(6, "Training complete! Analyzing results...")

    top_features = tracker.get_top_features_per_action(top_k=5)

    # Per-instrument summary
    inst_summary = {}
    for key, eps in per_instrument.items():
        if eps:
            inst_summary[key] = {
                "episodes": len(eps),
                "avg_reward": float(np.mean([e["reward"] for e in eps])),
                "avg_pnl": float(np.mean([e["pnl"] for e in eps])),
            }

    # Print results
    print_header("EXPLORATION RESULTS")

    print(f"\n📊 Overall Performance:")
    print(f"  Total Episodes:  {episodes}")
    print(f"  Total Reward:    {sum(all_rewards):+.1f}")
    print(f"  Avg Reward:      {np.mean(all_rewards):+.2f}")
    print(f"  Total P&L:       ${sum(all_pnl):+,.0f}")
    print(f"  Avg P&L:         ${np.mean(all_pnl):+,.0f}")

    print(f"\n📈 Per Instrument (top 10 by episodes):")
    sorted_inst = sorted(inst_summary.items(), key=lambda x: x[1]['episodes'], reverse=True)
    for key, stats in sorted_inst[:10]:
        print(f"  {key:20s}: {stats['episodes']:3d} eps | R={stats['avg_reward']:+8.2f} | PnL=${stats['avg_pnl']:+10,.0f}")

    print(f"\n🔍 Top Features Discovered:")
    action_names = ['HOLD', 'LONG', 'SHORT', 'CLOSE']
    for action_idx, features in top_features.items():
        action_name = action_names[action_idx] if action_idx < len(action_names) else f"Action{action_idx}"
        top_3 = features[:3]
        top_str = ", ".join([f"{name}({corr:+.3f})" for name, corr in top_3])
        print(f"  {action_name:6s}: {top_str}")

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    results_file = results_dir / f"exploration_{timestamp}.json"

    results = {
        "timestamp": timestamp,
        "episodes": episodes,
        "instruments": list(loader.instruments.keys()),
        "total_reward": float(sum(all_rewards)),
        "avg_reward": float(np.mean(all_rewards)),
        "total_pnl": float(sum(all_pnl)),
        "avg_pnl": float(np.mean(all_pnl)),
        "per_instrument": inst_summary,
        "top_features": top_features,
    }

    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n💾 Results saved: {results_file}")

    print_header("EXPLORATION COMPLETE!")

    print("\n✅ Next steps:")
    print("  • Review results in results/ directory")
    print("  • Analyze feature importance patterns")
    print("  • Try different reward configurations")
    print("  • Train for more episodes for convergence")
    print("  • Run with specific instruments or timeframes")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
