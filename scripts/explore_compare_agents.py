#!/usr/bin/env python3
"""
Agent Comparison - Exploration Step 2
======================================

Compare 4 agent types to discover which works best where:
1. LinearQ - Simple baseline
2. PPO - On-policy actor-critic
3. SAC - Off-policy with entropy
4. TD3 - Deterministic twin critics

Measure performance across:
- Asset class (forex, crypto, indices, metals, commodities)
- Regime (overdamped, underdamped, laminar, breakout)
- Timeframe (M15, M30, H1, H4)
- Volatility (low, medium, high)

THE MARKET TELLS US, WE DON'T ASSUME!

Usage:
    python scripts/explore_compare_agents.py
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
import json
from collections import defaultdict

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from rl_exploration_framework import (
    MultiInstrumentLoader,
    MultiInstrumentEnv,
    RewardShaper,
    LinearQAgent,
    FeatureTracker,
)

try:
    from rl_exploration_framework_agents import PPOAgent, SACAgent, TD3Agent
    DEEP_AGENTS_AVAILABLE = True
except ImportError:
    DEEP_AGENTS_AVAILABLE = False
    print("⚠️  Deep RL agents not available. Install PyTorch: pip install torch")


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


def classify_symbol(symbol: str) -> str:
    """Classify symbol into asset class."""
    symbol_upper = symbol.upper().replace('+', '').replace('-', '')

    if 'BTC' in symbol_upper or 'ETH' in symbol_upper or 'XRP' in symbol_upper:
        return 'crypto'
    elif len(symbol_upper) == 6 and symbol_upper.isalpha():
        return 'forex'
    elif any(x in symbol_upper for x in ['XAU', 'XAG', 'XPT', 'SILVER', 'GOLD']):
        return 'metals'
    elif any(x in symbol_upper for x in ['SPX', 'NAS', 'DOW', 'DJ', 'DAX', 'FTSE', 'NIKKEI', 'US', 'GER', 'UK', 'SA']):
        return 'indices'
    elif any(x in symbol_upper for x in ['OIL', 'WTI', 'BRENT', 'GAS', 'COPPER']):
        return 'commodities'
    else:
        return 'unknown'


def classify_volatility(stats: dict) -> str:
    """Classify episode volatility."""
    pnl_std = stats.get('pnl_std', 0)

    if pnl_std < 100:
        return 'low'
    elif pnl_std < 500:
        return 'medium'
    else:
        return 'high'


def print_header(text: str):
    """Print section header."""
    print("\n" + "=" * 80)
    print(f"  {text}")
    print("=" * 80)


def train_agent(agent_name: str, agent, env, episodes: int) -> dict:
    """Train a single agent and return performance breakdown."""
    print(f"\n🏃 Training {agent_name}...")

    epsilon = 1.0
    epsilon_decay = 0.95
    epsilon_min = 0.1

    # Performance tracking
    performance_breakdown = defaultdict(lambda: {
        'episodes': 0,
        'total_reward': 0,
        'total_pnl': 0,
        'rewards': [],
        'pnls': []
    })

    for episode in range(episodes):
        state = env.reset()
        total_reward = 0

        for step in range(500):
            action = agent.select_action(state, epsilon)
            next_state, reward, done, info = env.step(action)
            agent.update(state, action, reward, next_state, done)
            total_reward += reward
            state = next_state
            if done:
                break

        epsilon = max(epsilon_min, epsilon * epsilon_decay)

        # Get episode stats
        stats = env.get_episode_stats()
        pnl = stats.get("total_pnl", 0)
        instrument = stats.get("instrument", "unknown")
        regime = stats.get("regime", "unknown")

        # Extract breakdown dimensions
        symbol = instrument.split('_')[0]
        timeframe = instrument.split('_')[1] if '_' in instrument else 'unknown'
        asset_class = classify_symbol(symbol)
        volatility = classify_volatility(stats)

        # Record to breakdown
        breakdown_keys = [
            f"asset_class:{asset_class}",
            f"regime:{regime}",
            f"timeframe:{timeframe}",
            f"volatility:{volatility}",
            f"{asset_class}:{regime}",
            f"{asset_class}:{timeframe}",
            f"{regime}:{timeframe}",
            f"all:all"
        ]

        for key in breakdown_keys:
            perf = performance_breakdown[key]
            perf['episodes'] += 1
            perf['total_reward'] += total_reward
            perf['total_pnl'] += pnl
            perf['rewards'].append(total_reward)
            perf['pnls'].append(pnl)

        # Print progress
        if (episode + 1) % 10 == 0 or episode < 5:
            overall = performance_breakdown['all:all']
            avg_r = overall['total_reward'] / overall['episodes']
            avg_pnl = overall['total_pnl'] / overall['episodes']
            print(f"  Ep {episode+1:3d}: ε={epsilon:.3f} | R_avg={avg_r:+8.2f} | PnL_avg=${avg_pnl:+10,.0f} | {instrument[:20]:20s}")

    overall = performance_breakdown['all:all']
    print(f"✅ {agent_name} complete: R={overall['total_reward'] / overall['episodes']:+.2f}, PnL=${overall['total_pnl'] / overall['episodes']:+,.0f}")

    return performance_breakdown


def compare_agents_on_dimension(results: dict, dimension: str):
    """Compare agents on a specific dimension (e.g., asset_class, regime)."""
    print(f"\n📊 By {dimension.replace('_', ' ').title()}:")

    # Get all unique values for this dimension
    all_keys = set()
    for agent_results in results.values():
        all_keys.update([k for k in agent_results.keys() if k.startswith(f"{dimension}:")])

    # Print comparison for each value
    for key in sorted(all_keys):
        value = key.split(':')[1]
        print(f"\n  {value.upper()}:")

        for agent_name, agent_results in results.items():
            if key in agent_results:
                perf = agent_results[key]
                if perf['episodes'] > 0:
                    avg_r = perf['total_reward'] / perf['episodes']
                    avg_pnl = perf['total_pnl'] / perf['episodes']
                    print(f"    {agent_name:10s}: {perf['episodes']:3d} eps | R={avg_r:+8.2f} | PnL=${avg_pnl:+10,.0f}")


def print_agent_comparison_table(results: dict):
    """Print comparison table across all agents."""
    print_header("AGENT COMPARISON RESULTS")

    # Overall performance
    print(f"\n📊 Overall Performance:")
    print(f"{'Agent':15s} {'Episodes':>10s} {'Avg Reward':>12s} {'Avg PnL':>15s}")
    print("-" * 55)

    for agent_name, agent_results in results.items():
        overall = agent_results['all:all']
        if overall['episodes'] > 0:
            avg_r = overall['total_reward'] / overall['episodes']
            avg_pnl = overall['total_pnl'] / overall['episodes']
            print(f"{agent_name:15s} {overall['episodes']:10d} {avg_r:+12.2f} ${avg_pnl:+15,.0f}")

    # Breakdown by dimensions
    compare_agents_on_dimension(results, 'asset_class')
    compare_agents_on_dimension(results, 'regime')
    compare_agents_on_dimension(results, 'timeframe')
    compare_agents_on_dimension(results, 'volatility')


def identify_best_agent_per_category(results: dict) -> dict:
    """Identify which agent performs best in each category."""
    best_agents = {}

    # Get all breakdown keys from first agent
    first_agent_results = list(results.values())[0]
    all_keys = [k for k in first_agent_results.keys() if k != 'all:all']

    for key in all_keys:
        best_agent = None
        best_pnl = -np.inf

        for agent_name, agent_results in results.items():
            if key in agent_results:
                perf = agent_results[key]
                if perf['episodes'] > 0:
                    avg_pnl = perf['total_pnl'] / perf['episodes']
                    if avg_pnl > best_pnl:
                        best_pnl = avg_pnl
                        best_agent = agent_name

        if best_agent:
            best_agents[key] = {
                'agent': best_agent,
                'avg_pnl': best_pnl
            }

    return best_agents


def print_recommendations(best_agents: dict):
    """Print recommendations based on results."""
    print_header("RECOMMENDATIONS")

    print("\n🎯 Best Agent by Category:")

    # Group by dimension
    by_dimension = defaultdict(list)
    for key, info in best_agents.items():
        if ':' in key:
            dimension = key.split(':')[0]
            value = key.split(':')[1]
            by_dimension[dimension].append((value, info['agent'], info['avg_pnl']))

    for dimension, items in sorted(by_dimension.items()):
        print(f"\n  {dimension.replace('_', ' ').title()}:")
        for value, agent, pnl in sorted(items):
            print(f"    {value:15s}: {agent:10s} (${pnl:+,.0f})")

    print("\n💡 Key Insights:")
    print("  • Check if one agent dominates across all categories → use it universally")
    print("  • Check if different agents excel in different areas → specialize")
    print("  • Low performance across all agents → measurement/feature issue")
    print("\n  THE MARKET HAS TOLD US - NOW WE KNOW!")


def main():
    """Run agent comparison exploration."""
    print_header("AGENT COMPARISON - EXPLORATION")

    print("\n📋 Testing 4 agents on identical data:")
    print("  1. LinearQ  - Simple baseline")
    print("  2. PPO      - On-policy actor-critic")
    print("  3. SAC      - Off-policy with entropy")
    print("  4. TD3      - Deterministic twin critics")

    if not DEEP_AGENTS_AVAILABLE:
        print("\n❌ Deep RL agents (PPO, SAC, TD3) not available")
        print("   Install PyTorch: pip install torch")
        print("\n   Proceeding with LinearQ only...")

    # Load training data
    train_dir = Path("data/prepared/train")

    if not train_dir.exists():
        print(f"\n❌ Training data not found: {train_dir}")
        print(f"   Run: python scripts/prepare_data.py")
        return

    print(f"\n📥 Loading training data from {train_dir}...")

    loader = MultiInstrumentLoader(data_dir=str(train_dir), verbose=False)
    loader.load_all()

    if not loader.instruments:
        print("\n❌ No instruments loaded")
        return

    print(f"✅ Loaded {len(loader.instruments)} instruments")

    # Setup environment (shared across all agents)
    reward_shaper = RewardShaper(
        pnl_weight=1.0,
        edge_ratio_weight=0.3,
        mae_penalty_weight=2.5,
        regime_bonus_weight=0.2,
        entropy_alignment_weight=0.1,
    )

    env = MultiInstrumentEnv(
        loader=loader,
        feature_extractor=get_rl_state_features,
        reward_shaper=reward_shaper,
        sampling_mode="round_robin",
    )

    state_dim = env.state_dim
    n_actions = env.n_actions
    episodes = 30  # Episodes per agent

    print(f"\n⚙️  Environment: state_dim={state_dim}, actions={n_actions}")
    print(f"📊 Training each agent for {episodes} episodes...")

    # Initialize agents
    agents = {
        'LinearQ': LinearQAgent(
            state_dim=state_dim,
            n_actions=n_actions,
            learning_rate=0.05,
            gamma=0.9,
        )
    }

    if DEEP_AGENTS_AVAILABLE:
        agents['PPO'] = PPOAgent(
            state_dim=state_dim,
            n_actions=n_actions,
            learning_rate=3e-4,
            gamma=0.99,
        )
        agents['SAC'] = SACAgent(
            state_dim=state_dim,
            n_actions=n_actions,
            learning_rate=3e-4,
            gamma=0.99,
        )
        agents['TD3'] = TD3Agent(
            state_dim=state_dim,
            n_actions=n_actions,
            learning_rate=3e-4,
            gamma=0.99,
        )

    # Train each agent
    results = {}

    for agent_name, agent in agents.items():
        performance = train_agent(agent_name, agent, env, episodes)
        results[agent_name] = performance

    # Analysis
    print_agent_comparison_table(results)

    # Identify best agents per category
    best_agents = identify_best_agent_per_category(results)
    print_recommendations(best_agents)

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path("results/exploration")
    results_dir.mkdir(parents=True, exist_ok=True)
    results_file = results_dir / f"agent_comparison_{timestamp}.json"

    # Convert to serializable format
    results_serializable = {}
    for agent_name, agent_results in results.items():
        agent_data = {}
        for key, perf in agent_results.items():
            agent_data[key] = {
                'episodes': perf['episodes'],
                'avg_reward': float(perf['total_reward'] / perf['episodes']) if perf['episodes'] > 0 else 0,
                'avg_pnl': float(perf['total_pnl'] / perf['episodes']) if perf['episodes'] > 0 else 0,
            }
        results_serializable[agent_name] = agent_data

    # Add best agents summary
    best_agents_serializable = {}
    for key, info in best_agents.items():
        best_agents_serializable[key] = {
            'agent': info['agent'],
            'avg_pnl': float(info['avg_pnl'])
        }

    output = {
        'timestamp': timestamp,
        'agents_tested': list(agents.keys()),
        'episodes_per_agent': episodes,
        'instruments': list(loader.instruments.keys()),
        'results': results_serializable,
        'best_agents': best_agents_serializable,
    }

    with open(results_file, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\n💾 Results saved: {results_file}")

    print_header("AGENT COMPARISON COMPLETE")

    print(f"\n🔬 Next Steps:")
    print(f"  1. Review which agent(s) performed best")
    print(f"  2. If one dominates → use it universally")
    print(f"  3. If different agents excel → explore specialization")
    print(f"  4. Test measurement impact per winning agent")
    print(f"\n  Run: python scripts/explore_measurements.py")


if __name__ == '__main__':
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
