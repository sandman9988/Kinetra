#!/usr/bin/env python3
"""
Berserker Entry Training - GPU Accelerated

Features:
- GPU acceleration (AMD ROCm / NVIDIA CUDA)
- Atomic model saving (no corruption on interrupt)
- Run-based data management
- Comprehensive logging
- Prometheus metrics

Usage:
    python scripts/train_berserker.py --run berserker_run1
    python scripts/train_berserker.py --new-run  # Creates new run
"""

import sys
import os
import json
import tempfile
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
import argparse
import logging

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from kinetra.mt5_connector import load_csv_data
from kinetra.metrics_server import start_metrics_server, RLMetrics
from kinetra.rl_gpu_trainer import DQN, ReplayBuffer, TrainingConfig, TradingEnv, PhysicsFeatureComputer
from kinetra.data_manager import DataManager

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class AtomicSaver:
    """Atomic model saving to prevent corruption."""

    def __init__(self, models_dir: Path):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)

    def save_checkpoint(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        episode: int,
        metrics: Dict,
        filename: str = "checkpoint.pt"
    ) -> Path:
        """Atomically save model checkpoint.

        Uses temp file + rename pattern to prevent corruption.
        """
        checkpoint = {
            'episode': episode,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics,
            'timestamp': datetime.now().isoformat(),
        }

        target_path = self.models_dir / filename
        temp_path = self.models_dir / f".{filename}.tmp"

        # Write to temp file first
        torch.save(checkpoint, temp_path)

        # Atomic rename
        shutil.move(str(temp_path), str(target_path))

        return target_path

    def save_best(self, model: nn.Module, metrics: Dict) -> Path:
        """Save best model checkpoint."""
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'metrics': metrics,
            'timestamp': datetime.now().isoformat(),
        }

        target_path = self.models_dir / "best_model.pt"
        temp_path = self.models_dir / ".best_model.pt.tmp"

        torch.save(checkpoint, temp_path)
        shutil.move(str(temp_path), str(target_path))

        return target_path

    def load_checkpoint(self, filename: str = "checkpoint.pt") -> Optional[Dict]:
        """Load checkpoint if it exists."""
        path = self.models_dir / filename
        if path.exists():
            return torch.load(path)
        return None


class RunLogger:
    """Logs training results to file."""

    def __init__(self, logs_dir: Path):
        self.logs_dir = Path(logs_dir)
        self.logs_dir.mkdir(parents=True, exist_ok=True)

        # Create log files
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results_file = self.logs_dir / f"results_{timestamp}.jsonl"
        self.summary_file = self.logs_dir / "summary.json"

    def log_episode(self, episode: int, metrics: Dict):
        """Log single episode results."""
        entry = {
            'episode': episode,
            'timestamp': datetime.now().isoformat(),
            **metrics
        }

        # Append to JSONL file (one JSON object per line)
        with open(self.results_file, 'a') as f:
            f.write(json.dumps(entry) + '\n')

    def log_summary(self, summary: Dict):
        """Log final summary."""
        summary['completed_at'] = datetime.now().isoformat()
        with open(self.summary_file, 'w') as f:
            json.dump(summary, f, indent=2)

    def get_results(self) -> List[Dict]:
        """Read all logged results."""
        results = []
        for log_file in sorted(self.logs_dir.glob("results_*.jsonl")):
            with open(log_file) as f:
                for line in f:
                    if line.strip():
                        results.append(json.loads(line))
        return results


def detect_device() -> torch.device:
    """Detect best available device (GPU or CPU)."""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        gpu_name = torch.cuda.get_device_name(0)
        logger.info(f"Using GPU: {gpu_name}")
        return device
    else:
        logger.warning("No GPU detected. Using CPU (training will be slow)")
        return torch.device('cpu')


def train_berserker(
    run_dir: Path,
    n_episodes: int = 100,
    resume: bool = True,
    metrics_port: int = 8001,
):
    """Train Berserker entry strategy.

    Args:
        run_dir: Path to run folder
        n_episodes: Number of training episodes
        resume: Resume from checkpoint if available
        metrics_port: Port for Prometheus metrics
    """

    logger.info("=" * 70)
    logger.info("KINETRA BERSERKER TRAINING")
    logger.info("=" * 70)

    # Setup paths
    data_dir = run_dir / "data"
    models_dir = run_dir / "models"
    logs_dir = run_dir / "logs"

    # Initialize components
    device = detect_device()
    saver = AtomicSaver(models_dir)
    run_logger = RunLogger(logs_dir)
    metrics = start_metrics_server(metrics_port)

    # Load data
    data_files = sorted(data_dir.glob("*.csv"))
    if not data_files:
        logger.error(f"No data files found in {data_dir}")
        return

    logger.info(f"Loading {len(data_files)} data files...")
    instruments = []
    for path in data_files:
        name = path.stem
        logger.info(f"  Loading {name}...")
        data = load_csv_data(str(path))
        fc = PhysicsFeatureComputer()
        features = fc.compute(data)
        instruments.append((name, data, features))

    # Create environments
    envs = []
    for name, data, features in instruments:
        env = TradingEnv(data, features)
        envs.append((name, env))
        logger.info(f"  {name}: {len(data)} bars, {env.state_dim} state dims")

    state_dim = envs[0][1].state_dim
    action_dim = envs[0][1].action_dim

    # Training config
    config = TrainingConfig(
        learning_rate=3e-4,
        batch_size=256,           # Larger batches for GPU
        gamma=0.95,
        epsilon_start=0.8,
        epsilon_end=0.05,
        epsilon_decay=0.995,
        buffer_size=50000,
        target_update_freq=100,
        n_episodes=n_episodes,
    )

    # Initialize networks
    q_net = DQN(state_dim, action_dim).to(device)
    target_net = DQN(state_dim, action_dim).to(device)
    target_net.load_state_dict(q_net.state_dict())

    optimizer = torch.optim.Adam(q_net.parameters(), lr=config.learning_rate)
    buffer = ReplayBuffer(config.buffer_size, device)

    # Resume from checkpoint
    start_episode = 0
    best_pnl = float('-inf')

    if resume:
        checkpoint = saver.load_checkpoint()
        if checkpoint:
            q_net.load_state_dict(checkpoint['model_state_dict'])
            target_net.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_episode = checkpoint['episode']
            best_pnl = checkpoint['metrics'].get('best_pnl', float('-inf'))
            logger.info(f"Resumed from episode {start_episode}")

    epsilon = config.epsilon_start
    total_steps = 0

    logger.info(f"\nTraining for {n_episodes} episodes across {len(envs)} instruments...")
    logger.info(f"Monitor: http://localhost:{metrics_port}/metrics")
    logger.info("-" * 70)

    try:
        for episode in range(start_episode, n_episodes):
            episode_stats = {'trades': 0, 'wins': 0, 'pnl': 0, 'mfe': [], 'mae': []}
            losses = []

            for inst_name, env in envs:
                state = env.reset()
                done = False

                while not done:
                    # Action selection
                    if np.random.random() < epsilon:
                        action = np.random.randint(0, action_dim)
                    else:
                        with torch.no_grad():
                            state_t = torch.FloatTensor(state).unsqueeze(0).to(device)
                            q = q_net(state_t)
                            action = q.argmax().item()

                    next_state, reward, done, info = env.step(action)
                    buffer.push(state, action, reward, next_state, done)

                    # Track trades
                    if info.get('trade_pnl') is not None:
                        pnl = info['trade_pnl']
                        episode_stats['trades'] += 1
                        episode_stats['pnl'] += pnl
                        if pnl > 0:
                            episode_stats['wins'] += 1
                        episode_stats['mfe'].append(info.get('mfe', 0))
                        episode_stats['mae'].append(info.get('mae', 0))
                        metrics.record_trade(pnl)

                    # Training step
                    if len(buffer) >= config.batch_size:
                        batch = buffer.sample(config.batch_size)
                        states_t, actions_t, rewards_t, next_states_t, dones_t = batch

                        # Handle if already tensors (from GPU buffer)
                        if isinstance(states_t, torch.Tensor):
                            # NaN handling on GPU tensors
                            states_t = torch.nan_to_num(states_t, nan=0.5, posinf=1.0, neginf=0.0)
                            next_states_t = torch.nan_to_num(next_states_t, nan=0.5, posinf=1.0, neginf=0.0)
                            states_t = torch.clamp(states_t, -10, 10)
                            next_states_t = torch.clamp(next_states_t, -10, 10)
                            rewards_t = torch.clamp(rewards_t, -10, 10)
                        else:
                            # Convert numpy arrays
                            states_np = np.nan_to_num(np.array(states_t), nan=0.5, posinf=1.0, neginf=0.0)
                            next_states_np = np.nan_to_num(np.array(next_states_t), nan=0.5, posinf=1.0, neginf=0.0)
                            states_t = torch.FloatTensor(np.clip(states_np, -10, 10)).to(device)
                            actions_t = torch.LongTensor(actions_t).to(device)
                            rewards_t = torch.FloatTensor(np.clip(rewards_t, -10, 10)).to(device)
                            next_states_t = torch.FloatTensor(np.clip(next_states_np, -10, 10)).to(device)
                            dones_t = torch.FloatTensor(dones_t).to(device)

                        with torch.no_grad():
                            next_q = target_net(next_states_t).max(1)[0]
                            targets = rewards_t + config.gamma * next_q * (1 - dones_t)
                            targets = torch.clamp(targets, -100, 100)

                        current_q = q_net(states_t).gather(1, actions_t.unsqueeze(1)).squeeze()
                        loss = F.huber_loss(current_q, targets)

                        optimizer.zero_grad()
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(q_net.parameters(), max_norm=1.0)
                        optimizer.step()
                        losses.append(loss.item())

                    if total_steps % config.target_update_freq == 0:
                        target_net.load_state_dict(q_net.state_dict())

                    state = next_state
                    total_steps += 1

            # Epsilon decay
            epsilon = max(config.epsilon_end, epsilon * config.epsilon_decay)

            # Episode metrics
            n_trades = episode_stats['trades']
            win_rate = (episode_stats['wins'] / n_trades * 100) if n_trades > 0 else 0
            total_pnl = episode_stats['pnl']
            avg_loss = np.mean(losses) if losses else 0
            avg_mfe = np.mean(episode_stats['mfe']) if episode_stats['mfe'] else 0
            avg_mae = np.mean(episode_stats['mae']) if episode_stats['mae'] else 0
            mfe_mae_ratio = avg_mfe / avg_mae if avg_mae > 0 else 0

            episode_metrics = {
                'trades': n_trades,
                'win_rate': win_rate,
                'total_pnl': total_pnl,
                'avg_loss': avg_loss,
                'avg_mfe': avg_mfe,
                'avg_mae': avg_mae,
                'mfe_mae_ratio': mfe_mae_ratio,
                'epsilon': epsilon,
            }

            # Log to file
            run_logger.log_episode(episode + 1, episode_metrics)

            # Update Prometheus
            rl_metrics = RLMetrics(
                episode=episode + 1,
                total_trades=n_trades,
                win_rate=win_rate,
                total_pnl=total_pnl,
                avg_pnl=total_pnl / n_trades if n_trades > 0 else 0,
                avg_mfe=avg_mfe,
                avg_mae=avg_mae,
                mfe_mae_ratio=mfe_mae_ratio,
                mfe_captured=0,
                epsilon=epsilon,
                loss=avg_loss,
                reward=0,
            )
            metrics.update_rl_metrics(rl_metrics)
            metrics.complete_episode()

            # Save checkpoint every 10 episodes
            if (episode + 1) % 10 == 0:
                saver.save_checkpoint(
                    q_net, optimizer, episode + 1,
                    {'best_pnl': best_pnl, **episode_metrics}
                )
                logger.info(f"Checkpoint saved at episode {episode + 1}")

            # Save best model
            if total_pnl > best_pnl:
                best_pnl = total_pnl
                saver.save_best(q_net, episode_metrics)

            # Print progress
            logger.info(
                f"Ep {episode+1:3d}/{n_episodes} | "
                f"Trades: {n_trades:3d} | "
                f"WR: {win_rate:5.1f}% | "
                f"PnL: {total_pnl:+8.2f}% | "
                f"Loss: {avg_loss:.4f} | "
                f"ε: {epsilon:.3f}"
            )

    except KeyboardInterrupt:
        logger.info("\nTraining interrupted. Saving checkpoint...")
        saver.save_checkpoint(
            q_net, optimizer, episode,
            {'best_pnl': best_pnl}
        )

    # Final save
    saver.save_checkpoint(
        q_net, optimizer, n_episodes,
        {'best_pnl': best_pnl}
    )

    # Log summary
    run_logger.log_summary({
        'total_episodes': n_episodes,
        'best_pnl': best_pnl,
        'final_epsilon': epsilon,
        'total_steps': total_steps,
        'instruments': [name for name, _ in envs],
    })

    logger.info("-" * 70)
    logger.info(f"Training complete! Best P&L: {best_pnl:.2f}%")
    logger.info(f"Models saved to: {models_dir}")
    logger.info(f"Logs saved to: {logs_dir}")


def main():
    parser = argparse.ArgumentParser(description="Berserker Entry Training")
    parser.add_argument("--run", type=str, help="Run name (e.g., berserker_run1)")
    parser.add_argument("--new-run", action="store_true", help="Create new run")
    parser.add_argument("--episodes", type=int, default=100, help="Number of episodes")
    parser.add_argument("--no-resume", action="store_true", help="Don't resume from checkpoint")
    parser.add_argument("--port", type=int, default=8001, help="Metrics port")
    args = parser.parse_args()

    dm = DataManager()

    # Get or create run
    if args.new_run:
        run_dir = dm.create_run(strategy="berserker")
    elif args.run:
        run_dir = dm.get_run(args.run)
        if run_dir is None:
            logger.error(f"Run not found: {args.run}")
            logger.info("Available runs:")
            for run in dm.list_runs():
                logger.info(f"  - {run['name']}")
            return
    else:
        # Use most recent run or create new one
        runs = dm.list_runs()
        if runs:
            run_name = runs[-1]['name']
            run_dir = dm.get_run(run_name)
            logger.info(f"Using existing run: {run_name}")
        else:
            run_dir = dm.create_run(strategy="berserker")

    train_berserker(
        run_dir=run_dir,
        n_episodes=args.episodes,
        resume=not args.no_resume,
        metrics_port=args.port,
    )


if __name__ == "__main__":
    main()
