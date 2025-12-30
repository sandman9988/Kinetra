#!/usr/bin/env python3
"""
Exploration-Based ML/RL Strategy Testing

FIRST PRINCIPLES APPROACH - No Gating, Pure Exploration

Two paradigms:
1. Fat Candle Hunter: "Find me an edge by focusing on fat candles"
2. Laminar Flow Trader: "Find the best trend opportunities via rolling distributions"

NO magic numbers - physics/kinematics based with adaptive features.
NO pre-filtering or gating - let the agent discover what matters.
We don't know what we don't know.

Usage:
    python scripts/test_exploration_strategies.py --symbol BTCUSD --mode fat_candle
    python scripts/test_exploration_strategies.py --symbol EURUSD --mode laminar --episodes 100
"""

import argparse
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from kinetra.backtest_engine import BacktestEngine, BacktestResult
from kinetra.data_quality import validate_data
from kinetra.physics_engine import PhysicsEngine
from kinetra.symbol_spec import SymbolSpec, get_symbol_spec

# === EXPLORATION REWARD SHAPERS ===


class FatCandleRewardShaper:
    """
    Reward shaper for fat candle exploration.

    FIRST PRINCIPLES - No gating, no conditional logic based on predictions.
    Simple P&L scaled by bar magnitude captured.
    Let the agent figure out what predicts fat candles.
    """

    def __init__(self):
        self.name = "FatCandleExplorer"
        self.description = "Pure P&L weighted by magnitude - agent discovers patterns"

    def __call__(
        self,
        pnl: float,
        bar_magnitude: float,
        atr: float,
        physics_state: Dict[str, Any],
    ) -> float:
        """
        Calculate reward.

        Args:
            pnl: Trade P&L
            bar_magnitude: |close - open| of the bar
            atr: Average True Range
            physics_state: Current physics features

        Returns:
            Shaped reward
        """
        # Pure P&L weighted by magnitude captured
        # No conditional logic - agent discovers patterns
        magnitude_ratio = bar_magnitude / atr if atr > 0 else 1.0
        return pnl * magnitude_ratio


class LaminarFlowRewardShaper:
    """
    Reward shaper for laminar flow (trend) exploration.

    FIRST PRINCIPLES - No regime gating.
    Simple P&L weighted by trade efficiency (MFE capture).
    Agent learns when flow is laminar without explicit rules.
    """

    def __init__(self):
        self.name = "LaminarFlowTrader"
        self.description = "Pure efficiency-weighted P&L - no regime gating"

    def __call__(
        self,
        pnl: float,
        bars_held: int,
        mfe: float,
        mae: float,
        physics_state: Dict[str, Any],
    ) -> float:
        """
        Calculate reward.

        Args:
            pnl: Trade P&L
            bars_held: Number of bars position was held
            mfe: Maximum Favorable Excursion
            mae: Maximum Adverse Excursion
            physics_state: Current physics features

        Returns:
            Shaped reward
        """
        # Pure efficiency metric - no regime gating
        efficiency = mfe / (mfe + mae) if (mfe + mae) > 0 else 0.5
        return pnl * (0.5 + efficiency)


# === SIMPLE EXPLORATION AGENT ===


class SimpleExplorationAgent:
    """
    Simple exploration agent for testing.

    Uses epsilon-greedy exploration with physics features.
    This is a simplified version - real training uses KinetraAgent.
    """

    def __init__(
        self,
        feature_names: List[str],
        epsilon: float = 0.3,
        learning_rate: float = 0.01,
    ):
        self.feature_names = feature_names
        self.epsilon = epsilon
        self.learning_rate = learning_rate

        # Simple linear weights for each action
        n_features = len(feature_names)
        self.weights = {
            "long": np.random.randn(n_features) * 0.01,
            "short": np.random.randn(n_features) * 0.01,
            "hold": np.zeros(n_features),
        }

        self.action_history = []
        self.reward_history = []

    def get_features(self, physics_state: Dict[str, Any]) -> np.ndarray:
        """Extract feature vector from physics state."""
        features = []
        for name in self.feature_names:
            value = physics_state.get(name, 0.0)
            if isinstance(value, (int, float)):
                features.append(value)
            else:
                features.append(0.0)
        return np.array(features)

    def select_action(self, physics_state: Dict[str, Any]) -> str:
        """Select action using epsilon-greedy."""
        if np.random.random() < self.epsilon:
            # Explore
            return np.random.choice(["long", "short", "hold"])

        # Exploit
        features = self.get_features(physics_state)

        q_values = {action: np.dot(features, weights) for action, weights in self.weights.items()}

        return max(q_values, key=q_values.get)

    def update(self, features: np.ndarray, action: str, reward: float):
        """Update weights based on reward."""
        # Simple gradient update
        self.weights[action] += self.learning_rate * reward * features

        self.action_history.append(action)
        self.reward_history.append(reward)

    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance based on weight magnitudes."""
        importance = {}

        for i, name in enumerate(self.feature_names):
            total_weight = sum(abs(self.weights[action][i]) for action in self.weights)
            importance[name] = total_weight

        # Normalize
        max_imp = max(importance.values()) if importance else 1.0
        if max_imp > 0:
            importance = {k: v / max_imp for k, v in importance.items()}

        return dict(sorted(importance.items(), key=lambda x: -x[1]))


# === EXPLORATION RUNNER ===


@dataclass
class ExplorationResult:
    """Results from exploration run."""

    mode: str
    total_episodes: int
    total_trades: int
    total_pnl: float
    avg_reward: float
    win_rate: float
    feature_importance: Dict[str, float]
    action_distribution: Dict[str, int]
    learning_curve: List[float]


def run_exploration(
    data: pd.DataFrame,
    symbol_spec: SymbolSpec,
    mode: str = "fat_candle",
    episodes: int = 10,
    verbose: bool = True,
) -> ExplorationResult:
    """
    Run exploration-based strategy testing.

    FIRST PRINCIPLES: No gating, no filtering, pure exploration.
    Let the agent discover patterns in the physics feature space.

    Args:
        data: OHLCV DataFrame
        symbol_spec: Instrument specification
        mode: "fat_candle" or "laminar"
        episodes: Number of exploration episodes
        verbose: Print progress

    Returns:
        ExplorationResult with findings
    """
    if verbose:
        print(f"\n{'=' * 60}")
        print(f"EXPLORATION MODE: {mode.upper()}")
        print(f"{'=' * 60}")
        print(f"Symbol: {symbol_spec.symbol}")
        print(f"Episodes: {episodes}")
        print(f"Data bars: {len(data):,}")
        print()

    # Select reward shaper
    if mode == "fat_candle":
        reward_shaper = FatCandleRewardShaper()
    else:
        reward_shaper = LaminarFlowRewardShaper()

    if verbose:
        print(f"Reward Shaper: {reward_shaper.name}")
        print(f"Description: {reward_shaper.description}")
        print()

    # Compute physics features
    physics = PhysicsEngine()
    physics_state = physics.compute_physics_state(data["close"])

    # Add OHLCV to physics state
    physics_state["open"] = data["open"].values
    physics_state["high"] = data["high"].values
    physics_state["low"] = data["low"].values
    physics_state["close"] = data["close"].values
    if "volume" in data.columns:
        physics_state["volume"] = data["volume"].values

    # ALL features available - no pre-selection (FIRST PRINCIPLES)
    feature_names = [
        "energy",
        "energy_pct",
        "damping",
        "damping_pct",
        "entropy",
        "entropy_pct",
        # Add any additional physics features that exist
    ]

    # Check which features actually exist
    available_features = [f for f in feature_names if f in physics_state.columns]

    if verbose:
        print(f"Available features: {len(available_features)}")
        print(f"  {', '.join(available_features[:10])}...")
        print()

    # Create exploration agent
    agent = SimpleExplorationAgent(
        feature_names=available_features,
        epsilon=0.3,  # 30% exploration
        learning_rate=0.01,
    )

    # Calculate ATR for reward shaping
    high = data["high"].values
    low = data["low"].values
    close = data["close"].values
    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]

    tr = np.maximum(high - low, np.maximum(np.abs(high - prev_close), np.abs(low - prev_close)))
    atr = pd.Series(tr).rolling(20).mean().fillna(tr).values

    # Run exploration episodes
    all_trades = []
    all_rewards = []
    learning_curve = []

    lookback = 20

    for episode in range(episodes):
        episode_rewards = []
        position = None  # None, "long", or "short"
        entry_price = 0.0
        entry_bar = 0
        mfe = 0.0
        mae = 0.0

        for i in range(lookback, len(data) - 1):
            # Get current state
            state = {f: physics_state[f].iloc[i] for f in available_features}

            current_close = data["close"].iloc[i]
            current_open = data["open"].iloc[i]
            current_high = data["high"].iloc[i]
            current_low = data["low"].iloc[i]

            # Select action
            action = agent.select_action(state)

            # Execute action
            reward = 0.0

            if position is None:
                # No position - can enter
                if action == "long":
                    position = "long"
                    entry_price = current_close
                    entry_bar = i
                    mfe = 0.0
                    mae = 0.0
                elif action == "short":
                    position = "short"
                    entry_price = current_close
                    entry_bar = i
                    mfe = 0.0
                    mae = 0.0
            else:
                # Update MFE/MAE
                if position == "long":
                    mfe = max(mfe, current_high - entry_price)
                    mae = max(mae, entry_price - current_low)
                else:
                    mfe = max(mfe, entry_price - current_low)
                    mae = max(mae, current_high - entry_price)

                # Check for exit
                should_exit = (
                    (position == "long" and action == "short")
                    or (position == "short" and action == "long")
                    or action == "hold"  # Exit on hold
                )

                if should_exit:
                    # Calculate P&L
                    if position == "long":
                        pnl = (
                            (current_close - entry_price)
                            / symbol_spec.tick_size
                            * symbol_spec.tick_value
                        )
                    else:
                        pnl = (
                            (entry_price - current_close)
                            / symbol_spec.tick_size
                            * symbol_spec.tick_value
                        )

                    bars_held = i - entry_bar
                    bar_magnitude = abs(current_close - current_open)
                    current_atr = atr[i]

                    # Calculate reward based on mode
                    if mode == "fat_candle":
                        reward = reward_shaper(pnl, bar_magnitude, current_atr, state)
                    else:
                        reward = reward_shaper(pnl, bars_held, mfe, mae, state)

                    all_trades.append(
                        {
                            "entry_bar": entry_bar,
                            "exit_bar": i,
                            "position": position,
                            "pnl": pnl,
                            "reward": reward,
                            "mfe": mfe,
                            "mae": mae,
                        }
                    )

                    position = None

            # Update agent
            features = agent.get_features(state)
            agent.update(features, action, reward)

            episode_rewards.append(reward)
            all_rewards.append(reward)

        # Track learning curve
        avg_reward = np.mean(episode_rewards) if episode_rewards else 0.0
        learning_curve.append(avg_reward)

        if verbose and (episode + 1) % max(1, episodes // 10) == 0:
            print(
                f"Episode {episode + 1}/{episodes}: "
                f"Avg Reward = {avg_reward:.4f}, "
                f"Trades = {len(all_trades)}"
            )

    # Calculate results
    total_pnl = sum(t["pnl"] for t in all_trades)
    win_rate = len([t for t in all_trades if t["pnl"] > 0]) / len(all_trades) if all_trades else 0.0

    action_counts = {}
    for action in agent.action_history:
        action_counts[action] = action_counts.get(action, 0) + 1

    result = ExplorationResult(
        mode=mode,
        total_episodes=episodes,
        total_trades=len(all_trades),
        total_pnl=total_pnl,
        avg_reward=np.mean(all_rewards) if all_rewards else 0.0,
        win_rate=win_rate,
        feature_importance=agent.get_feature_importance(),
        action_distribution=action_counts,
        learning_curve=learning_curve,
    )

    if verbose:
        print()
        print("=" * 60)
        print("EXPLORATION RESULTS")
        print("=" * 60)
        print(f"Total Trades: {result.total_trades}")
        print(f"Total P&L: ${result.total_pnl:,.2f}")
        print(f"Win Rate: {result.win_rate:.1%}")
        print(f"Avg Reward: {result.avg_reward:.4f}")
        print()
        print("Feature Importance (discovered by agent):")
        for feat, imp in list(result.feature_importance.items())[:10]:
            print(f"  {feat}: {imp:.3f}")
        print()
        print("Action Distribution:")
        total_actions = sum(action_counts.values())
        for action, count in sorted(action_counts.items()):
            pct = count / total_actions * 100 if total_actions > 0 else 0
            print(f"  {action}: {count} ({pct:.1f}%)")

    return result


def load_data(filepath: str) -> pd.DataFrame:
    """Load OHLCV data from CSV."""
    df = pd.read_csv(filepath)

    # Standardize column names
    column_map = {
        "<DATE>": "date",
        "<TIME>": "time",
        "<OPEN>": "open",
        "<HIGH>": "high",
        "<LOW>": "low",
        "<CLOSE>": "close",
        "<TICKVOL>": "volume",
        "<VOL>": "volume",
        "<SPREAD>": "spread",
    }

    df.columns = [column_map.get(c, c.lower()) for c in df.columns]

    # Combine date and time if separate
    if "date" in df.columns and "time" in df.columns:
        df["datetime"] = pd.to_datetime(df["date"] + " " + df["time"])
        df.set_index("datetime", inplace=True)
    elif "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"])
        df.set_index("time", inplace=True)

    return df


def main():
    parser = argparse.ArgumentParser(description="Exploration-based strategy testing")
    parser.add_argument("--symbol", type=str, default="BTCUSD", help="Symbol to test")
    parser.add_argument("--data", type=str, help="Path to CSV data file")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["fat_candle", "laminar"],
        default="fat_candle",
        help="Exploration mode",
    )
    parser.add_argument("--episodes", type=int, default=10, help="Number of episodes")

    args = parser.parse_args()

    # Load data
    if args.data:
        data_path = args.data
    else:
        # Try to find data file
        data_dir = Path(__file__).parent.parent / "data" / "master"
        candidates = list(data_dir.glob(f"{args.symbol}*.csv"))
        if candidates:
            data_path = str(candidates[0])
        else:
            print(f"No data file found for {args.symbol}")
            print(f"Please specify --data or place CSV in {data_dir}")
            sys.exit(1)

    print(f"Loading data from {data_path}...")
    data = load_data(data_path)
    print(f"Loaded {len(data):,} bars")

    # Get symbol spec
    try:
        symbol_spec = get_symbol_spec(args.symbol)
    except KeyError:
        print(f"Symbol spec not found for {args.symbol}, using defaults")
        symbol_spec = SymbolSpec(symbol=args.symbol)

    # Run exploration
    result = run_exploration(
        data=data,
        symbol_spec=symbol_spec,
        mode=args.mode,
        episodes=args.episodes,
        verbose=True,
    )

    print("\n" + "=" * 60)
    print("KEY INSIGHT: FIRST PRINCIPLES")
    print("=" * 60)
    print("The agent discovered feature importance WITHOUT any gating or filtering.")
    print("Features ranked by learned importance reflect actual predictive value")
    print("for this specific instrument and timeframe.")
    print()
    print("This is the foundation for exploration-based ML/RL strategies:")
    print("- No magic numbers")
    print("- No predetermined thresholds")
    print("- Pure pattern discovery from physics features")
    print("=" * 60)


if __name__ == "__main__":
    main()
