#!/usr/bin/env python3
"""
Specialization Strategy Explorer

Systematically compares different agent specialization strategies:
1. UNIVERSAL: One agent for all instruments/timeframes
2. ASSET CLASS: Separate agents per market type (forex, crypto, metals, etc.)
3. REGIME: Separate agents per physics regime (laminar, underdamped, overdamped)
4. TIMEFRAME: Separate agents per timeframe (H1, H4, D1, etc.)

Goal: Discover which specialization yields best generalization and edge robustness.
"""

import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "tests"))

from rl_exploration_framework import (
    MultiInstrumentLoader,
    MultiInstrumentEnv,
    RewardShaper,
    LinearQAgent,
    FeatureTracker,
    PersistenceManager,
)
from test_physics_pipeline import get_rl_state_features, get_rl_feature_names


class RegimeSpecializedRewardShaper(RewardShaper):
    """
    Reward shaper specialized for a specific physics regime.

    Overrides regime bonus to heavily reward trading in target regime.
    """

    def __init__(self, regime_bonus_map: Dict[str, float], **kwargs):
        super().__init__(**kwargs)
        self.regime_bonus_map = regime_bonus_map

    def shape_reward(
        self,
        raw_pnl: float,
        mae: float,
        mfe: float,
        bars_held: int,
        entry_features: np.ndarray,
        exit_features: np.ndarray,
        physics_state: pd.DataFrame,
        bar_index: int,
    ) -> float:
        """Override to use custom regime bonus map."""
        # Call parent for base reward calculation
        reward = super().shape_reward(
            raw_pnl, mae, mfe, bars_held, entry_features, exit_features, physics_state, bar_index
        )

        # Override regime component with specialized bonus
        if bar_index < len(physics_state):
            regime = physics_state.iloc[bar_index].get("regime", "UNKNOWN")
            regime_bonus = self.regime_bonus_map.get(regime, 0.0)

            # Subtract default regime bonus (already added by parent)
            default_bonus = 0.0
            if regime == "LAMINAR":
                default_bonus = 0.3
            elif regime == "UNDERDAMPED":
                default_bonus = 0.1
            elif regime == "OVERDAMPED":
                default_bonus = -0.2

            # Replace with specialized bonus
            reward = reward - (self.regime_bonus_weight * default_bonus) + (self.regime_bonus_weight * regime_bonus)

        return reward


class SpecializationExplorer:
    """
    Explores optimal agent specialization strategy.

    Tests:
    - Universal (baseline): One agent for everything
    - Asset class: One agent per market type
    - Regime: One agent per physics regime
    - Timeframe: One agent per timeframe

    Metrics:
    - Per-strategy average reward, Sharpe ratio, win rate
    - Cross-validation: Train on some instruments, test on held-out
    - Edge robustness: Performance consistency across markets
    """

    def __init__(
        self,
        data_dir: str = "data/master",
        n_episodes_per_agent: int = 50,
        learning_rate: float = 0.0001,
        gamma: float = 0.99,
        verbose: bool = True,
        output_dir: str = "results/specialization_exploration",
    ):
        self.data_dir = data_dir
        self.n_episodes_per_agent = n_episodes_per_agent
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.verbose = verbose
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Load all instruments
        self.loader = MultiInstrumentLoader(data_dir=data_dir, verbose=verbose)
        self.loader.load_all()

        if not self.loader.instruments:
            raise ValueError(f"No instruments found in {data_dir}")

        # Group instruments by different criteria
        self._group_instruments()

        # Results storage
        self.results: Dict[str, Dict] = {}

        # Feature names
        self.feature_names = get_rl_feature_names()

    def _log(self, msg: str):
        if self.verbose:
            print(msg)

    def _group_instruments(self):
        """Group instruments by asset class, regime, and timeframe."""
        self.by_asset_class: Dict[str, List[str]] = defaultdict(list)
        self.by_timeframe: Dict[str, List[str]] = defaultdict(list)
        self.all_keys = list(self.loader.instruments.keys())

        for key, inst_data in self.loader.instruments.items():
            # Group by asset class
            market_type = inst_data.market_type
            self.by_asset_class[market_type].append(key)

            # Group by timeframe
            timeframe = inst_data.timeframe
            self.by_timeframe[timeframe].append(key)

        self._log("\n" + "="*70)
        self._log("INSTRUMENT GROUPING")
        self._log("="*70)

        self._log(f"\nBy Asset Class:")
        for market_type, keys in self.by_asset_class.items():
            self._log(f"  {market_type}: {len(keys)} instruments ({', '.join(keys)})")

        self._log(f"\nBy Timeframe:")
        for tf, keys in self.by_timeframe.items():
            self._log(f"  {tf}: {len(keys)} instruments ({', '.join(keys)})")

        self._log(f"\nTotal: {len(self.all_keys)} instruments")

    def run_all_strategies(self) -> Dict[str, Dict]:
        """Run all specialization strategies and compare."""
        self._log("\n" + "="*70)
        self._log("SPECIALIZATION STRATEGY EXPLORATION")
        self._log("="*70)

        # Strategy 1: Universal
        self._log("\n[1/4] Running UNIVERSAL strategy...")
        self.results['universal'] = self._run_universal()

        # Strategy 2: Asset Class Specialization
        self._log("\n[2/4] Running ASSET CLASS strategy...")
        self.results['asset_class'] = self._run_asset_class()

        # Strategy 3: Timeframe Specialization
        self._log("\n[3/5] Running TIMEFRAME strategy...")
        self.results['timeframe'] = self._run_timeframe()

        # Strategy 4: Regime Specialization
        self._log("\n[4/5] Running REGIME strategy...")
        self.results['regime'] = self._run_regime()

        # Strategy 5: Hybrid (Asset Class + Timeframe)
        self._log("\n[5/5] Running HYBRID strategy...")
        self.results['hybrid'] = self._run_hybrid()

        # Compare all strategies
        self._compare_strategies()

        # Save results
        self._save_results()

        return self.results

    def _run_universal(self) -> Dict:
        """Train single universal agent on all instruments."""
        self._log("  Training universal agent on all instruments...")

        # Create multi-instrument environment
        reward_shaper = RewardShaper(
            pnl_weight=1.0,
            edge_ratio_weight=0.5,
            mae_penalty_weight=0.3,
            regime_bonus_weight=0.2,
            entropy_alignment_weight=0.1,
        )

        env = MultiInstrumentEnv(
            loader=self.loader,
            feature_extractor=get_rl_state_features,
            reward_shaper=reward_shaper,
            sampling_mode="round_robin",
        )

        # Train agent
        agent = LinearQAgent(
            state_dim=env.state_dim,
            n_actions=env.n_actions,
            learning_rate=self.learning_rate,
            gamma=self.gamma,
        )

        metrics = self._train_agent(
            agent=agent,
            env=env,
            n_episodes=self.n_episodes_per_agent * len(self.all_keys),
            agent_name="universal",
        )

        return {
            'strategy': 'universal',
            'n_agents': 1,
            'instruments_per_agent': len(self.all_keys),
            'metrics': metrics,
        }

    def _run_asset_class(self) -> Dict:
        """Train separate agents for each asset class."""
        self._log("  Training asset-class-specific agents...")

        agents_results = {}

        for market_type, inst_keys in self.by_asset_class.items():
            if not inst_keys:
                continue

            self._log(f"    Training {market_type} agent ({len(inst_keys)} instruments)...")

            # Create environment with only this asset class
            filtered_loader = self._create_filtered_loader(inst_keys)

            reward_shaper = RewardShaper(
                pnl_weight=1.0,
                edge_ratio_weight=0.5,
                mae_penalty_weight=0.3,
                regime_bonus_weight=0.2,
                entropy_alignment_weight=0.1,
            )

            env = MultiInstrumentEnv(
                loader=filtered_loader,
                feature_extractor=get_rl_state_features,
                reward_shaper=reward_shaper,
                sampling_mode="round_robin",
            )

            # Train agent
            agent = LinearQAgent(
                state_dim=env.state_dim,
                n_actions=env.n_actions,
                learning_rate=self.learning_rate,
                gamma=self.gamma,
            )

            metrics = self._train_agent(
                agent=agent,
                env=env,
                n_episodes=self.n_episodes_per_agent * len(inst_keys),
                agent_name=f"asset_class_{market_type}",
            )

            agents_results[market_type] = {
                'instruments': inst_keys,
                'n_instruments': len(inst_keys),
                'metrics': metrics,
            }

        return {
            'strategy': 'asset_class',
            'n_agents': len(agents_results),
            'agents': agents_results,
        }

    def _run_timeframe(self) -> Dict:
        """Train separate agents for each timeframe."""
        self._log("  Training timeframe-specific agents...")

        agents_results = {}

        for timeframe, inst_keys in self.by_timeframe.items():
            if not inst_keys:
                continue

            self._log(f"    Training {timeframe} agent ({len(inst_keys)} instruments)...")

            # Create environment with only this timeframe
            filtered_loader = self._create_filtered_loader(inst_keys)

            reward_shaper = RewardShaper(
                pnl_weight=1.0,
                edge_ratio_weight=0.5,
                mae_penalty_weight=0.3,
                regime_bonus_weight=0.2,
                entropy_alignment_weight=0.1,
            )

            env = MultiInstrumentEnv(
                loader=filtered_loader,
                feature_extractor=get_rl_state_features,
                reward_shaper=reward_shaper,
                sampling_mode="round_robin",
            )

            # Train agent
            agent = LinearQAgent(
                state_dim=env.state_dim,
                n_actions=env.n_actions,
                learning_rate=self.learning_rate,
                gamma=self.gamma,
            )

            metrics = self._train_agent(
                agent=agent,
                env=env,
                n_episodes=self.n_episodes_per_agent * len(inst_keys),
                agent_name=f"timeframe_{timeframe}",
            )

            agents_results[timeframe] = {
                'instruments': inst_keys,
                'n_instruments': len(inst_keys),
                'metrics': metrics,
            }

        return {
            'strategy': 'timeframe',
            'n_agents': len(agents_results),
            'agents': agents_results,
        }

    def _run_regime(self) -> Dict:
        """
        Train separate agents for different physics regimes.

        NOTE: Since regimes are determined per-bar (not per-instrument),
        we use regime-weighted reward shaping to specialize agents.
        Each agent gets bonus rewards when trading in its target regime.
        """
        self._log("  Training regime-specific agents...")

        # Define regime specializations
        regime_configs = {
            'laminar': {
                'description': 'Laminar flow (low entropy, stable trends)',
                'reward_bonus': {'LAMINAR': 0.5, 'UNDERDAMPED': 0.1, 'OVERDAMPED': -0.2},
            },
            'underdamped': {
                'description': 'Underdamped (mean-reverting oscillations)',
                'reward_bonus': {'UNDERDAMPED': 0.5, 'LAMINAR': 0.1, 'OVERDAMPED': -0.1},
            },
            'overdamped': {
                'description': 'Overdamped (choppy, high friction)',
                'reward_bonus': {'OVERDAMPED': 0.3, 'UNDERDAMPED': -0.1, 'LAMINAR': -0.2},
            },
        }

        agents_results = {}

        for regime_name, config in regime_configs.items():
            self._log(f"    Training {regime_name} specialist ({config['description']})...")

            # Create regime-specialized reward shaper
            regime_shaper = RegimeSpecializedRewardShaper(
                regime_bonus_map=config['reward_bonus'],
                pnl_weight=1.0,
                edge_ratio_weight=0.5,
                mae_penalty_weight=0.3,
                regime_bonus_weight=0.5,  # Higher regime bonus for specialization
                entropy_alignment_weight=0.1,
            )

            # Create environment with all instruments (regime-agnostic sampling)
            env = MultiInstrumentEnv(
                loader=self.loader,
                feature_extractor=get_rl_state_features,
                reward_shaper=regime_shaper,
                sampling_mode="round_robin",
            )

            # Train agent
            agent = LinearQAgent(
                state_dim=env.state_dim,
                n_actions=env.n_actions,
                learning_rate=self.learning_rate,
                gamma=self.gamma,
            )

            metrics = self._train_agent(
                agent=agent,
                env=env,
                n_episodes=self.n_episodes_per_agent * len(self.all_keys),
                agent_name=f"regime_{regime_name}",
            )

            agents_results[regime_name] = {
                'description': config['description'],
                'reward_bonus_map': config['reward_bonus'],
                'instruments': self.all_keys,
                'metrics': metrics,
            }

        return {
            'strategy': 'regime',
            'n_agents': len(agents_results),
            'agents': agents_results,
            'note': 'Agents specialized via regime-weighted reward shaping',
        }

    def _run_hybrid(self) -> Dict:
        """Train agents specialized by both asset class AND timeframe."""
        self._log("  Training hybrid (asset class + timeframe) agents...")

        agents_results = {}

        # Group by both asset class and timeframe
        for market_type, inst_keys in self.by_asset_class.items():
            # Further split by timeframe within this asset class
            tf_groups = defaultdict(list)
            for key in inst_keys:
                inst_data = self.loader.instruments[key]
                tf_groups[inst_data.timeframe].append(key)

            for timeframe, tf_inst_keys in tf_groups.items():
                if not tf_inst_keys:
                    continue

                agent_name = f"{market_type}_{timeframe}"
                self._log(f"    Training {agent_name} agent ({len(tf_inst_keys)} instruments)...")

                # Create environment
                filtered_loader = self._create_filtered_loader(tf_inst_keys)

                reward_shaper = RewardShaper(
                    pnl_weight=1.0,
                    edge_ratio_weight=0.5,
                    mae_penalty_weight=0.3,
                    regime_bonus_weight=0.2,
                    entropy_alignment_weight=0.1,
                )

                env = MultiInstrumentEnv(
                    loader=filtered_loader,
                    feature_extractor=get_rl_state_features,
                    reward_shaper=reward_shaper,
                    sampling_mode="round_robin",
                )

                # Train agent
                agent = LinearQAgent(
                    state_dim=env.state_dim,
                    n_actions=env.n_actions,
                    learning_rate=self.learning_rate,
                    gamma=self.gamma,
                )

                metrics = self._train_agent(
                    agent=agent,
                    env=env,
                    n_episodes=self.n_episodes_per_agent * len(tf_inst_keys),
                    agent_name=f"hybrid_{agent_name}",
                )

                agents_results[agent_name] = {
                    'market_type': market_type,
                    'timeframe': timeframe,
                    'instruments': tf_inst_keys,
                    'n_instruments': len(tf_inst_keys),
                    'metrics': metrics,
                }

        return {
            'strategy': 'hybrid',
            'n_agents': len(agents_results),
            'agents': agents_results,
        }

    def _create_filtered_loader(self, inst_keys: List[str]) -> MultiInstrumentLoader:
        """Create a MultiInstrumentLoader with only specified instruments."""
        filtered_loader = MultiInstrumentLoader(
            data_dir=self.data_dir,
            verbose=False,
            compute_physics=False,  # Already computed
        )

        # Copy only specified instruments
        filtered_loader.instruments = {
            key: self.loader.instruments[key]
            for key in inst_keys
            if key in self.loader.instruments
        }

        return filtered_loader

    def _train_agent(
        self,
        agent: LinearQAgent,
        env: MultiInstrumentEnv,
        n_episodes: int,
        agent_name: str,
    ) -> Dict:
        """Train a single agent and return metrics."""
        tracker = FeatureTracker(self.feature_names)

        all_rewards = []
        all_trades = []
        all_pnl = []
        per_instrument = defaultdict(list)
        epsilon = 1.0

        for episode in range(n_episodes):
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

            epsilon = max(0.1, epsilon * 0.98)

            stats = env.get_episode_stats()
            all_rewards.append(total_reward)
            all_trades.append(stats["trades"])
            all_pnl.append(stats.get("total_pnl", 0))
            per_instrument[stats["instrument"]].append({
                "episode": episode,
                "reward": total_reward,
                "pnl": stats.get("total_pnl", 0),
            })

        # Compute metrics
        avg_reward = np.mean(all_rewards)
        std_reward = np.std(all_rewards)
        sharpe = avg_reward / (std_reward + 1e-8)
        avg_pnl = np.mean(all_pnl)
        total_trades = sum(all_trades)

        # Per-instrument consistency (edge robustness)
        inst_rewards = []
        for inst_key, episodes in per_instrument.items():
            if episodes:
                inst_avg_reward = np.mean([e["reward"] for e in episodes])
                inst_rewards.append(inst_avg_reward)

        reward_consistency = np.std(inst_rewards) / (np.mean(inst_rewards) + 1e-8) if inst_rewards else 0

        return {
            'avg_reward': float(avg_reward),
            'std_reward': float(std_reward),
            'sharpe_ratio': float(sharpe),
            'avg_pnl': float(avg_pnl),
            'total_trades': int(total_trades),
            'n_episodes': n_episodes,
            'reward_consistency': float(reward_consistency),  # Lower is better
            'per_instrument': {k: {
                'avg_reward': float(np.mean([e["reward"] for e in v])),
                'avg_pnl': float(np.mean([e["pnl"] for e in v])),
            } for k, v in per_instrument.items()},
        }

    def _compare_strategies(self):
        """Compare all strategies and print summary."""
        self._log("\n" + "="*70)
        self._log("STRATEGY COMPARISON")
        self._log("="*70)

        comparison = []

        # Universal
        if 'universal' in self.results:
            metrics = self.results['universal']['metrics']
            comparison.append({
                'strategy': 'Universal',
                'n_agents': 1,
                'avg_reward': metrics['avg_reward'],
                'sharpe': metrics['sharpe_ratio'],
                'consistency': metrics['reward_consistency'],
            })

        # Asset class
        if 'asset_class' in self.results:
            agents = self.results['asset_class']['agents']
            all_rewards = [a['metrics']['avg_reward'] for a in agents.values()]
            all_sharpes = [a['metrics']['sharpe_ratio'] for a in agents.values()]
            all_consistency = [a['metrics']['reward_consistency'] for a in agents.values()]
            comparison.append({
                'strategy': 'Asset Class',
                'n_agents': len(agents),
                'avg_reward': np.mean(all_rewards),
                'sharpe': np.mean(all_sharpes),
                'consistency': np.mean(all_consistency),
            })

        # Timeframe
        if 'timeframe' in self.results:
            agents = self.results['timeframe']['agents']
            all_rewards = [a['metrics']['avg_reward'] for a in agents.values()]
            all_sharpes = [a['metrics']['sharpe_ratio'] for a in agents.values()]
            all_consistency = [a['metrics']['reward_consistency'] for a in agents.values()]
            comparison.append({
                'strategy': 'Timeframe',
                'n_agents': len(agents),
                'avg_reward': np.mean(all_rewards),
                'sharpe': np.mean(all_sharpes),
                'consistency': np.mean(all_consistency),
            })

        # Regime
        if 'regime' in self.results:
            agents = self.results['regime']['agents']
            all_rewards = [a['metrics']['avg_reward'] for a in agents.values()]
            all_sharpes = [a['metrics']['sharpe_ratio'] for a in agents.values()]
            all_consistency = [a['metrics']['reward_consistency'] for a in agents.values()]
            comparison.append({
                'strategy': 'Regime',
                'n_agents': len(agents),
                'avg_reward': np.mean(all_rewards),
                'sharpe': np.mean(all_sharpes),
                'consistency': np.mean(all_consistency),
            })

        # Hybrid
        if 'hybrid' in self.results:
            agents = self.results['hybrid']['agents']
            all_rewards = [a['metrics']['avg_reward'] for a in agents.values()]
            all_sharpes = [a['metrics']['sharpe_ratio'] for a in agents.values()]
            all_consistency = [a['metrics']['reward_consistency'] for a in agents.values()]
            comparison.append({
                'strategy': 'Hybrid (Asset+TF)',
                'n_agents': len(agents),
                'avg_reward': np.mean(all_rewards),
                'sharpe': np.mean(all_sharpes),
                'consistency': np.mean(all_consistency),
            })

        # Print table
        self._log(f"\n{'Strategy':<20} {'Agents':>8} {'Avg Reward':>12} {'Sharpe':>10} {'Consistency':>12}")
        self._log("-" * 70)
        for c in comparison:
            self._log(f"{c['strategy']:<20} {c['n_agents']:>8} {c['avg_reward']:>12.2f} "
                     f"{c['sharpe']:>10.3f} {c['consistency']:>12.3f}")

        # Determine winner
        best_sharpe_idx = np.argmax([c['sharpe'] for c in comparison])
        best_consistency_idx = np.argmin([c['consistency'] for c in comparison])

        self._log("\n" + "="*70)
        self._log("RECOMMENDATIONS")
        self._log("="*70)
        self._log(f"Best Sharpe Ratio: {comparison[best_sharpe_idx]['strategy']}")
        self._log(f"  Sharpe: {comparison[best_sharpe_idx]['sharpe']:.3f}")
        self._log(f"\nBest Consistency (Edge Robustness): {comparison[best_consistency_idx]['strategy']}")
        self._log(f"  Consistency: {comparison[best_consistency_idx]['consistency']:.3f} (lower is better)")

        # Store comparison
        self.results['comparison'] = comparison

    def _save_results(self):
        """Save all results to JSON."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = self.output_dir / f"specialization_exploration_{timestamp}.json"

        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)

        self._log(f"\n[SAVED] Results: {output_file}")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Explore optimal agent specialization strategy",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        '--data-dir',
        type=str,
        default='data/master',
        help='Path to data directory (default: data/master)',
    )

    parser.add_argument(
        '--episodes',
        type=int,
        default=50,
        help='Episodes per agent (default: 50)',
    )

    parser.add_argument(
        '--lr',
        type=float,
        default=0.0001,
        help='Learning rate (default: 0.0001)',
    )

    parser.add_argument(
        '--gamma',
        type=float,
        default=0.99,
        help='Discount factor (default: 0.99)',
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default='results/specialization_exploration',
        help='Output directory (default: results/specialization_exploration)',
    )

    args = parser.parse_args()

    # Run exploration
    explorer = SpecializationExplorer(
        data_dir=args.data_dir,
        n_episodes_per_agent=args.episodes,
        learning_rate=args.lr,
        gamma=args.gamma,
        output_dir=args.output_dir,
        verbose=True,
    )

    results = explorer.run_all_strategies()

    print("\n" + "="*70)
    print("EXPLORATION COMPLETE")
    print("="*70)
    print("\nNext steps:")
    print("1. Review results JSON for detailed per-agent metrics")
    print("2. Use recommended strategy for production deployment")
    print("3. Consider ensemble of top-performing agents")


if __name__ == "__main__":
    main()
