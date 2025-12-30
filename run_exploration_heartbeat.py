#!/usr/bin/env python3
"""
Exploration Runner with Live Heartbeat Stats
Shows real-time progress per instrument, per run, cumulative, and per portfolio.
"""
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from rl_exploration_framework import (
    MultiInstrumentLoader,
    MultiInstrumentEnv,
    RewardShaper,
    LinearQAgent,
    FeatureTracker,
)
from test_physics_pipeline import get_rl_state_features, get_rl_feature_names


# Asset class mapping
ASSET_CLASS = {
    "AUDJPY+": "Forex", "AUDUSD+": "Forex", "EURJPY+": "Forex",
    "GBPJPY+": "Forex", "GBPUSD+": "Forex",
    "BTCJPY": "Crypto", "BTCUSD": "Crypto", "ETHEUR": "Crypto", "XRPJPY": "Crypto",
    "DJ30ft": "Index", "NAS100": "Index", "Nikkei225": "Index",
    "EU50": "Index", "GER40": "Index", "SA40": "Index", "US2000": "Index",
    "COPPER-C": "Commodity", "UKOUSD": "Commodity",
    "XAGUSD": "Metals", "XAUAUD+": "Metals", "XAUUSD+": "Metals", "XPTUSD": "Metals",
}


def get_asset_class(instrument_key):
    """Get asset class from instrument key."""
    for prefix, cls in ASSET_CLASS.items():
        if instrument_key.startswith(prefix):
            return cls
    return "Other"


def print_heartbeat(
    episode: int,
    total_episodes: int,
    instrument: str,
    reward: float,
    pnl: float,
    trades: int,
    cumulative: dict,
    portfolio: dict,
):
    """Print formatted heartbeat stats. PnL is now in PERCENTAGE terms."""
    asset_class = get_asset_class(instrument)

    # Per-episode stats (PnL is now percentage, not absolute dollars)
    print(f"\n{'─'*80}")
    print(f"│ EP {episode:3d}/{total_episodes} │ {instrument:<20} │ {asset_class:<10} │")
    print(f"├{'─'*80}")
    print(f"│ Reward: {reward:+10.2f} │ PnL: {pnl:+8.2f}% │ Trades: {trades:3d} │")

    # Cumulative stats
    print(f"├{'─'*80}")
    print(f"│ CUMULATIVE: Total R={cumulative['total_reward']:+.1f} │ "
          f"Total PnL={cumulative['total_pnl']:+.2f}% │ "
          f"Avg R={cumulative['avg_reward']:+.2f} │")

    # Portfolio breakdown
    print(f"├{'─'*80}")
    print(f"│ PORTFOLIO BY ASSET CLASS:")
    for cls, stats in sorted(portfolio.items()):
        if stats['episodes'] > 0:
            avg_r = stats['total_reward'] / stats['episodes']
            avg_pnl = stats['total_pnl'] / stats['episodes']
            print(f"│   {cls:<12}: Eps={stats['episodes']:3d} │ "
                  f"Avg R={avg_r:+8.2f} │ Avg PnL={avg_pnl:+8.2f}% │")
    print(f"└{'─'*80}")


def run_with_heartbeat(
    data_dir: str = "data/master",
    config_name: str = "Heartbeat_Run",
    mae_w: float = 2.5,
    lr: float = 0.05,
    gamma: float = 0.9,
    episodes: int = 50,
):
    """Run exploration with live heartbeat output."""
    print("=" * 80)
    print(f"  EXPLORATION WITH HEARTBEAT: {config_name}")
    print(f"  MAE_w={mae_w}, LR={lr}, Gamma={gamma}, Episodes={episodes}")
    print("=" * 80)

    # Load instruments
    loader = MultiInstrumentLoader(data_dir=data_dir, verbose=True)
    loader.load_all()

    if not loader.instruments:
        print("[ERROR] No instruments found!")
        return None

    n_instruments = len(loader.instruments)
    print(f"\n[READY] {n_instruments} instruments loaded")
    print("=" * 80)

    # Create reward shaper
    reward_shaper = RewardShaper(
        pnl_weight=1.0,
        edge_ratio_weight=0.3,
        mae_penalty_weight=mae_w,
        regime_bonus_weight=0.2,
        entropy_alignment_weight=0.1,
    )

    # Create env
    env = MultiInstrumentEnv(
        loader=loader,
        feature_extractor=get_rl_state_features,
        reward_shaper=reward_shaper,
        sampling_mode="round_robin",
    )

    # Create agent
    agent = LinearQAgent(
        state_dim=env.state_dim,
        n_actions=env.n_actions,
        learning_rate=lr,
        gamma=gamma,
    )

    feature_names = get_rl_feature_names()
    tracker = FeatureTracker(feature_names)

    # Tracking structures
    cumulative = {
        'total_reward': 0.0,
        'total_pnl': 0.0,
        'total_trades': 0,
        'episodes': 0,
        'avg_reward': 0.0,
    }

    portfolio = defaultdict(lambda: {
        'total_reward': 0.0,
        'total_pnl': 0.0,
        'total_trades': 0,
        'episodes': 0,
    })

    per_instrument = defaultdict(lambda: {
        'rewards': [],
        'pnls': [],
        'trades': [],
    })

    epsilon = 1.0

    # Training loop with heartbeat
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

        # Get episode stats
        stats = env.get_episode_stats()
        instrument = stats["instrument"]
        pnl = stats.get("total_pnl", 0)
        trades = stats["trades"]
        asset_class = get_asset_class(instrument)

        # Update cumulative
        cumulative['total_reward'] += total_reward
        cumulative['total_pnl'] += pnl
        cumulative['total_trades'] += trades
        cumulative['episodes'] += 1
        cumulative['avg_reward'] = cumulative['total_reward'] / cumulative['episodes']

        # Update portfolio
        portfolio[asset_class]['total_reward'] += total_reward
        portfolio[asset_class]['total_pnl'] += pnl
        portfolio[asset_class]['total_trades'] += trades
        portfolio[asset_class]['episodes'] += 1

        # Update per-instrument
        per_instrument[instrument]['rewards'].append(total_reward)
        per_instrument[instrument]['pnls'].append(pnl)
        per_instrument[instrument]['trades'].append(trades)

        # Print heartbeat
        print_heartbeat(
            episode=episode + 1,
            total_episodes=episodes,
            instrument=instrument,
            reward=total_reward,
            pnl=pnl,
            trades=trades,
            cumulative=cumulative,
            portfolio=dict(portfolio),
        )

    # Final summary
    print("\n" + "=" * 80)
    print("  FINAL RESULTS (PnL in PERCENTAGE terms for cross-instrument comparison)")
    print("=" * 80)

    print(f"\n[CUMULATIVE]")
    print(f"  Total Episodes: {cumulative['episodes']}")
    print(f"  Total Reward:   {cumulative['total_reward']:+.1f}")
    print(f"  Total PnL:      {cumulative['total_pnl']:+.2f}%")
    print(f"  Avg PnL/Ep:     {cumulative['total_pnl'] / cumulative['episodes'] if cumulative['episodes'] > 0 else 0:+.2f}%")
    print(f"  Avg Reward:     {cumulative['avg_reward']:+.2f}")

    print(f"\n[PORTFOLIO BY ASSET CLASS]")
    print(f"  {'Class':<12} {'Episodes':>8} {'Avg Reward':>12} {'Avg PnL%':>12} {'Total PnL%':>12}")
    print("  " + "-" * 60)
    for cls in sorted(portfolio.keys()):
        stats = portfolio[cls]
        if stats['episodes'] > 0:
            avg_r = stats['total_reward'] / stats['episodes']
            avg_pnl = stats['total_pnl'] / stats['episodes']
            print(f"  {cls:<12} {stats['episodes']:>8} {avg_r:>+12.2f} {avg_pnl:>+12.2f}% {stats['total_pnl']:>+11.2f}%")

    print(f"\n[TOP 10 INSTRUMENTS BY AVG REWARD]")
    inst_avg = []
    for inst, data in per_instrument.items():
        if len(data['rewards']) > 0:
            avg_r = np.mean(data['rewards'])
            avg_pnl = np.mean(data['pnls'])
            inst_avg.append((inst, avg_r, avg_pnl, len(data['rewards'])))

    inst_avg.sort(key=lambda x: x[1], reverse=True)
    print(f"  {'Instrument':<25} {'Eps':>5} {'Avg Reward':>12} {'Avg PnL%':>12}")
    print("  " + "-" * 56)
    for inst, avg_r, avg_pnl, eps in inst_avg[:10]:
        print(f"  {inst:<25} {eps:>5} {avg_r:>+12.2f} {avg_pnl:>+12.2f}%")

    print(f"\n[BOTTOM 5 INSTRUMENTS]")
    for inst, avg_r, avg_pnl, eps in inst_avg[-5:]:
        print(f"  {inst:<25} {eps:>5} {avg_r:>+12.2f} {avg_pnl:>+12.2f}%")

    # Top features
    print(f"\n[TOP FEATURES LEARNED]")
    top_features = tracker.get_top_features_per_action(top_k=5)
    for action, features in top_features.items():
        top_3 = features[:3]
        top_str = ", ".join([f"{name}({corr:+.3f})" for name, corr in top_3])
        print(f"  {action}: {top_str}")

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results = {
        "config": config_name,
        "cumulative": cumulative,
        "portfolio": {k: dict(v) for k, v in portfolio.items()},
        "per_instrument": {k: {
            "avg_reward": float(np.mean(v['rewards'])) if v['rewards'] else 0,
            "avg_pnl": float(np.mean(v['pnls'])) if v['pnls'] else 0,
            "episodes": len(v['rewards']),
        } for k, v in per_instrument.items()},
        "top_features": top_features,
    }

    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    results_file = results_dir / f"heartbeat_{config_name}_{timestamp}.json"

    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n[SAVED] {results_file}")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/master")
    parser.add_argument("--config", default="Heartbeat_FullDataset")
    parser.add_argument("--mae-w", type=float, default=2.5)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--gamma", type=float, default=0.9)
    parser.add_argument("--episodes", type=int, default=50)
    args = parser.parse_args()

    run_with_heartbeat(
        data_dir=args.data_dir,
        config_name=args.config,
        mae_w=args.mae_w,
        lr=args.lr,
        gamma=args.gamma,
        episodes=args.episodes,
    )
