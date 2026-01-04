#!/usr/bin/env python3
"""
Train Dueling DQN on Denoised Market Data
==========================================

Physics-first DRL training framework.
Non-prescriptive: discovers optimal triggers via shaped rewards only.

Reward Shaping:
- Base: Net PnL (gross - friction)
- Efficiency: MAE/MFE ratio (Pythagorean path efficiency)
- Risk: Drawdown penalty

NO hand-crafted indicators, NO static rules.

Usage:
    # Train on denoised BTCUSD
    python scripts/train_dqn.py --input data/prepared/denoised/BTCUSD_H1_denoised.csv

    # Train with custom parameters
    python scripts/train_dqn.py --input data.csv --episodes 200 --lr 1e-4

    # Continue training from checkpoint
    python scripts/train_dqn.py --input data.csv --load-model models/dqn_checkpoint.pt

__version__ = "1.0.0"
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import torch
from tqdm import tqdm  # type: ignore[import-untyped]

# Project root
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from kinetra.drl_dueling_dqn import DQNAgent, TradingEnvironment  # noqa: E402

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_data(filepath: Path, use_denoised: bool = True) -> tuple:
    """
    Load and prepare data for training.

    Args:
        filepath: CSV file path
        use_denoised: Use denoised columns if available

    Returns:
        (prices, prices_denoised)
    """
    df = pd.read_csv(filepath)

    # Normalize column names
    df.columns = df.columns.str.lower().str.strip()

    # Get prices
    if "close" not in df.columns:
        raise ValueError("Missing 'close' column")

    prices = df["close"].values

    # Use denoised if available
    if use_denoised and "close_denoised" in df.columns:
        prices_denoised = df["close_denoised"].values
        logger.info("✅ Using denoised prices for state features")
    else:
        prices_denoised = prices
        logger.warning("⚠️  No denoised data found, using raw prices")

    return prices, prices_denoised


def train_dqn(
    prices: np.ndarray,
    prices_denoised: np.ndarray,
    num_episodes: int = 100,
    window_size: int = 50,
    lr: float = 1e-4,
    target_update_freq: int = 10,
    save_dir: Optional[Path] = None,
    checkpoint_path: Optional[Path] = None,
):
    """
    Train Dueling DQN agent.

    Args:
        prices: Original prices (for PnL calculation)
        prices_denoised: Denoised prices (for state features)
        num_episodes: Training episodes
        window_size: State window size
        lr: Learning rate
        target_update_freq: Target network update frequency
        save_dir: Directory to save models/plots
        checkpoint_path: Load checkpoint to continue training
    """
    # Create environment
    env = TradingEnvironment(
        prices=prices,
        prices_denoised=prices_denoised,
        window_size=window_size,
    )

    # Create agent
    state_size = window_size + 1  # Returns + position + unrealized PnL
    action_size = 3  # 0=flat, 1=hold, 2=long

    agent = DQNAgent(
        state_size=state_size,
        action_size=action_size,
        lr=lr,
    )

    # Load checkpoint if provided
    if checkpoint_path and checkpoint_path.exists():
        agent.load(checkpoint_path)
        logger.info(f"Loaded checkpoint from {checkpoint_path}")

    # Training metrics
    rewards_history = []
    epsilon_history = []

    logger.info("\n🚀 Starting DQN Training")
    logger.info(f"   Device: {agent.device}")
    logger.info(f"   Episodes: {num_episodes}")
    logger.info(f"   State size: {state_size}")
    logger.info(f"   Action size: {action_size}")
    logger.info(f"   Data bars: {len(prices)}\n")

    # Training loop
    for episode in tqdm(range(num_episodes), desc="Training"):
        state = env.reset()
        total_reward = 0.0
        steps = 0

        while True:
            # Select and execute action
            action = agent.select_action(state, training=True)
            next_state, reward, done, _ = env.step(action)

            # Store transition
            agent.buffer.push(state, action, reward, next_state, done)

            # Train
            agent.train_step()

            state = next_state
            total_reward += reward
            steps += 1

            if done:
                break

        # Update target network periodically
        if (episode + 1) % target_update_freq == 0:
            agent.update_target_network()

        # Track metrics
        rewards_history.append(total_reward)
        epsilon = agent.epsilon_end + (agent.epsilon_start - agent.epsilon_end) * np.exp(
            -agent.steps_done / agent.epsilon_decay
        )
        epsilon_history.append(epsilon)

        # Log progress
        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(rewards_history[-10:])
            logger.info(
                f"Episode {episode+1}/{num_episodes} | "
                f"Avg Reward (last 10): {avg_reward:.2f} | "
                f"Epsilon: {epsilon:.3f} | "
                f"Trades: {len(env.trades)}"
            )

    # Save final model
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)
        model_path = save_dir / "dqn_final.pt"
        agent.save(model_path)
        logger.info(f"\n✅ Model saved to {model_path}")

        # Plot training curves
        plot_training_curves(rewards_history, epsilon_history, save_dir)

    return agent, env, rewards_history


def plot_training_curves(rewards, epsilons, save_dir):
    """Plot and save training curves."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    # Rewards
    ax1.plot(rewards, alpha=0.6, label="Episode Reward")
    ax1.plot(pd.Series(rewards).rolling(10).mean(), linewidth=2, label="MA(10)")
    ax1.set_title("Training Rewards per Episode")
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Total Reward")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Epsilon decay
    ax2.plot(epsilons)
    ax2.set_title("Epsilon Decay (Exploration Rate)")
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Epsilon")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = save_dir / "training_curves.png"
    plt.savefig(plot_path, dpi=150)
    logger.info(f"📊 Training curves saved to {plot_path}")
    plt.close()


def backtest_agent(agent, env):
    """Backtest trained agent on full dataset."""
    state = env.reset()
    equity_curve = [env.initial_cash]
    positions = [0]
    actions_taken = []

    logger.info("\n📈 Running backtest with trained policy...")

    while True:
        action = agent.select_action(state, training=False)  # Greedy
        state, _, done, _ = env.step(action)

        # Track equity
        current_price = env.prices[env.step_idx - 1] if env.step_idx > 0 else env.prices[0]
        equity = env.cash + current_price * env.position * 1000  # Approx
        equity_curve.append(equity)
        positions.append(env.position)
        actions_taken.append(action)

        if done:
            break

    # Calculate metrics
    total_return = (equity_curve[-1] - equity_curve[0]) / equity_curve[0]
    max_equity = max(equity_curve)
    max_drawdown = (max_equity - min(equity_curve)) / max_equity if max_equity > 0 else 0

    logger.info("\n✅ Backtest Results:")
    logger.info(f"   Total Return: {total_return*100:.2f}%")
    logger.info(f"   Max Drawdown: {max_drawdown*100:.2f}%")
    logger.info(f"   Total Trades: {len(env.trades)}")

    # MAE/MFE analysis
    if env.trades:
        avg_mfe = np.mean([t.mfe for t in env.trades])
        avg_mae = np.mean([abs(t.mae) for t in env.trades])
        avg_pnl = np.mean([t.pnl for t in env.trades])

        logger.info(f"   Avg MFE: {avg_mfe*100:.2f}%")
        logger.info(f"   Avg MAE: {avg_mae*100:.2f}%")
        logger.info(f"   Avg PnL per trade: ${avg_pnl:.2f}")

    return equity_curve, positions


def main():
    parser = argparse.ArgumentParser(description="Train Dueling DQN on denoised market data")
    parser.add_argument(
        "--input", type=str, required=True, help="Input CSV file (with denoised columns)"
    )
    parser.add_argument(
        "--episodes", type=int, default=100, help="Number of training episodes (default: 100)"
    )
    parser.add_argument(
        "--window", type=int, default=50, help="State window size (default: 50)"
    )
    parser.add_argument(
        "--lr", type=float, default=1e-4, help="Learning rate (default: 1e-4)"
    )
    parser.add_argument(
        "--target-update", type=int, default=10, help="Target network update frequency (default: 10)"
    )
    parser.add_argument(
        "--output-dir", type=str, default="models/dqn", help="Output directory (default: models/dqn)"
    )
    parser.add_argument(
        "--load-model", type=str, help="Load checkpoint to continue training"
    )
    parser.add_argument(
        "--no-backtest", action="store_true", help="Skip backtest after training"
    )

    args = parser.parse_args()

    # Check GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"🖥️  Device: {device}")
    if device.type == "cuda":
        logger.info(f"   GPU: {torch.cuda.get_device_name(0)}")

    # Load data
    filepath = Path(args.input)
    if not filepath.exists():
        logger.error(f"File not found: {filepath}")
        return 1

    prices, prices_denoised = load_data(filepath)
    logger.info(f"📊 Loaded {len(prices)} bars from {filepath.name}")

    # Train
    save_dir = Path(args.output_dir)
    checkpoint_path = Path(args.load_model) if args.load_model else None

    agent, env, rewards = train_dqn(
        prices=prices,
        prices_denoised=prices_denoised,
        num_episodes=args.episodes,
        window_size=args.window,
        lr=args.lr,
        target_update_freq=args.target_update,
        save_dir=save_dir,
        checkpoint_path=checkpoint_path,
    )

    # Backtest
    if not args.no_backtest:
        equity, positions = backtest_agent(agent, env)

    return 0


if __name__ == "__main__":
    sys.exit(main())
