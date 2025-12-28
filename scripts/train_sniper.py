#!/usr/bin/env python3
"""
Sniper Strategy RL Training

Learns to identify stable, laminar trending conditions:
- Low Reynolds number (smooth flow)
- Low entropy (orderly)
- High flow consistency (sustained direction)
- Low damping (momentum persists)
- Strong trend detection

Opposite of Berserker (which seeks turbulence/explosions).
Sniper waits for clean setups and rides trends.

Usage:
    python3 scripts/train_sniper.py --run sniper_run1 --episodes 200
"""

import sys
import os
import json
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from kinetra.rl_gpu_trainer import DQN, ReplayBuffer, PhysicsFeatureComputer
from kinetra.metrics_server import start_metrics_server, RLMetrics
from kinetra.data_manager import DataManager

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


@dataclass
class SniperConfig:
    """Configuration for Sniper training."""
    # Network
    hidden_sizes: Tuple[int, ...] = (128, 64, 32)
    learning_rate: float = 1e-4  # Lower LR for stability
    gamma: float = 0.98  # Higher gamma - we care about longer trends

    # Exploration
    epsilon_start: float = 0.9
    epsilon_end: float = 0.05
    epsilon_decay: float = 0.997  # Slower decay - more exploration

    # Training
    batch_size: int = 256
    buffer_size: int = 100000
    target_update_freq: int = 200
    n_episodes: int = 200

    # Sniper-specific
    min_trend_bars: int = 5  # Minimum bars to consider a trend
    laminar_threshold: float = 0.3  # Reynolds < this = laminar


class SniperFeatureComputer(PhysicsFeatureComputer):
    """Extended features for Sniper strategy."""

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        # Get base physics features
        result = super().compute(df)

        # === SNIPER-SPECIFIC FEATURES ===

        # Trend strength (ADX-like)
        high_low = df['high'] - df['low']
        high_close = abs(df['high'] - df['close'].shift(1))
        low_close = abs(df['low'] - df['close'].shift(1))
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = tr.rolling(14).mean()

        up_move = df['high'] - df['high'].shift(1)
        down_move = df['low'].shift(1) - df['low']

        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)

        plus_di = 100 * pd.Series(plus_dm).rolling(14).mean() / atr
        minus_di = 100 * pd.Series(minus_dm).rolling(14).mean() / atr

        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
        result['trend_strength'] = dx.rolling(14).mean() / 100  # Normalize to 0-1

        # Laminar score (inverse of Reynolds, clamped)
        result['laminar_score'] = 1 - result['reynolds_pct'].clip(0, 1)

        # Stability (inverse of entropy)
        result['stability'] = 1 - result['entropy_pct'].clip(0, 1)

        # Momentum persistence (autocorrelation of returns)
        returns = df['close'].pct_change()
        result['momentum_persistence'] = returns.rolling(20).apply(
            lambda x: x.autocorr(lag=1) if len(x) > 1 else 0, raw=False
        ).fillna(0).clip(-1, 1)

        # Trend direction (-1 to +1)
        sma_fast = df['close'].rolling(10).mean()
        sma_slow = df['close'].rolling(50).mean()
        result['trend_direction'] = ((sma_fast - sma_slow) / (sma_slow + 1e-10)).clip(-0.1, 0.1) * 10

        # Consecutive direction bars
        direction = np.sign(returns)
        def count_consecutive(x):
            if len(x) == 0:
                return 0
            last_dir = x.iloc[-1]
            count = 0
            for v in reversed(x.values):
                if v == last_dir:
                    count += 1
                else:
                    break
            return count * last_dir  # Positive for up, negative for down

        result['consecutive_bars'] = direction.rolling(10).apply(count_consecutive, raw=False).fillna(0) / 10

        # Volume trend (rising volume in trend = strong)
        vol_ma = df['volume'].rolling(20).mean()
        result['volume_trend'] = (df['volume'] / vol_ma.clip(lower=1)).clip(0, 3) / 3

        # Price position in channel (0=bottom, 1=top)
        rolling_high = df['high'].rolling(50).max()
        rolling_low = df['low'].rolling(50).min()
        result['channel_position'] = (df['close'] - rolling_low) / (rolling_high - rolling_low + 1e-10)

        # Convert new features to percentiles
        new_features = ['trend_strength', 'laminar_score', 'stability',
                       'momentum_persistence', 'volume_trend']
        window = min(500, len(df))

        for col in new_features:
            result[f'{col}_pct'] = result[col].rolling(window, min_periods=20).apply(
                lambda x: (x.iloc[-1] > x.iloc[:-1]).mean() if len(x) > 1 else 0.5, raw=False
            ).fillna(0.5)

        # Keep raw values for some
        result['trend_direction_raw'] = result['trend_direction']
        result['consecutive_bars_raw'] = result['consecutive_bars']
        result['channel_position_raw'] = result['channel_position']

        return result.fillna(0.5)


class SniperEnv:
    """Trading environment optimized for Sniper strategy."""

    HOLD = 0
    LONG = 1
    SHORT = 2
    EXIT = 3

    def __init__(self, df: pd.DataFrame, feature_df: pd.DataFrame, config: SniperConfig):
        self.df = df
        self.feature_df = feature_df
        self.config = config

        # Feature columns for state
        self.feature_cols = [
            # Core physics (percentiles)
            'energy_pct', 'damping_pct', 'entropy_pct', 'reynolds_pct', 'viscosity_pct',
            # Flow
            'buying_pressure_pct', 'flow_consistency_pct', 'range_position_pct',
            # Sniper-specific (percentiles)
            'trend_strength_pct', 'laminar_score_pct', 'stability_pct',
            'momentum_persistence_pct', 'volume_trend_pct',
            # Raw directional
            'trend_direction_raw', 'consecutive_bars_raw', 'channel_position_raw',
            'roc_pct', 'momentum_dir_pct'
        ]

        # Filter to existing columns
        self.feature_cols = [c for c in self.feature_cols if c in feature_df.columns]

        self.state_dim = len(self.feature_cols) + 5  # + position info
        self.action_dim = 4

        self.reset()

    def reset(self) -> np.ndarray:
        self.idx = 100
        self.position = 0
        self.entry_price = 0.0
        self.entry_bar = 0
        self.entry_laminar = 0.0
        self.max_profit = 0.0
        self.max_loss = 0.0
        self.trades = []
        return self._get_state()

    def _get_state(self) -> np.ndarray:
        if self.idx >= len(self.feature_df):
            return np.zeros(self.state_dim)

        features = self.feature_df.iloc[self.idx][self.feature_cols].values.astype(float)

        # Position info
        pos_dir = float(self.position)
        bars_held = float(self.idx - self.entry_bar) / 20 if self.position != 0 else 0

        if self.position != 0:
            current_price = self.df.iloc[self.idx]['close']
            pnl = (current_price - self.entry_price) / self.entry_price * self.position * 100
        else:
            pnl = 0.0

        # Profit/loss tracking
        profit_ratio = self.max_profit / (self.max_loss + 0.001) if self.max_loss > 0 else 0
        laminar_change = self.feature_df.iloc[self.idx].get('laminar_score_pct', 0.5) - self.entry_laminar

        return np.concatenate([features, [pos_dir, bars_held, pnl, profit_ratio, laminar_change]])

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        reward = 0.0
        info = {}

        current = self.df.iloc[self.idx]
        high, low, close = current['high'], current['low'], current['close']
        features = self.feature_df.iloc[self.idx]

        laminar = features.get('laminar_score_pct', 0.5)
        trend_strength = features.get('trend_strength_pct', 0.5)
        stability = features.get('stability_pct', 0.5)
        trend_dir = features.get('trend_direction_raw', 0)

        # Update max profit/loss if in position
        if self.position != 0:
            if self.position == 1:
                current_profit = (high - self.entry_price) / self.entry_price * 100
                current_loss = (self.entry_price - low) / self.entry_price * 100
            else:
                current_profit = (self.entry_price - low) / self.entry_price * 100
                current_loss = (high - self.entry_price) / self.entry_price * 100

            self.max_profit = max(self.max_profit, current_profit)
            self.max_loss = max(self.max_loss, current_loss)

        # === ACTION EXECUTION ===

        if action == self.LONG and self.position == 0:
            self.position = 1
            self.entry_price = close
            self.entry_bar = self.idx
            self.entry_laminar = laminar
            self.max_profit = 0.0
            self.max_loss = 0.0

            # Reward for entering during laminar + trend aligned conditions
            entry_quality = laminar * stability * (1 if trend_dir > 0 else 0.5)
            reward = 0.1 * entry_quality  # Small reward for good entry

        elif action == self.SHORT and self.position == 0:
            self.position = -1
            self.entry_price = close
            self.entry_bar = self.idx
            self.entry_laminar = laminar
            self.max_profit = 0.0
            self.max_loss = 0.0

            entry_quality = laminar * stability * (1 if trend_dir < 0 else 0.5)
            reward = 0.1 * entry_quality

        elif action == self.EXIT and self.position != 0:
            pnl = (close - self.entry_price) / self.entry_price * 100 * self.position
            bars_held = self.idx - self.entry_bar

            # === SNIPER REWARD FUNCTION ===
            # Rewards riding trends, penalizes choppy trades

            # Base P&L reward
            reward = pnl

            # Bonus for riding trends (more bars = better if profitable)
            if pnl > 0 and bars_held >= self.config.min_trend_bars:
                trend_bonus = min(1.0, bars_held / 20) * pnl * 0.5
                reward += trend_bonus

            # Penalty for quick exits (sniper should be patient)
            if bars_held < self.config.min_trend_bars:
                reward -= 0.5

            # Bonus for capturing most of the move
            if self.max_profit > 0:
                capture_ratio = pnl / self.max_profit
                if capture_ratio > 0.7:
                    reward += 0.5 * capture_ratio

            # Penalty for letting winners become losers
            if self.max_profit > 1.0 and pnl < 0:
                reward -= self.max_profit * 0.5

            info['trade_pnl'] = pnl
            info['bars_held'] = bars_held
            info['mfe'] = self.max_profit
            info['mae'] = self.max_loss

            self.trades.append({
                'pnl': pnl,
                'bars': bars_held,
                'mfe': self.max_profit,
                'mae': self.max_loss,
                'entry_laminar': self.entry_laminar,
            })

            self.position = 0

        elif action == self.HOLD and self.position != 0:
            # Small penalty for holding - encourages action
            # But less penalty during laminar conditions (patience is good)
            hold_penalty = 0.001 * (1 - laminar)
            reward = -hold_penalty

        # Move to next bar
        self.idx += 1
        done = self.idx >= len(self.df) - 1

        # Force close at end
        if done and self.position != 0:
            pnl = (self.df.iloc[self.idx]['close'] - self.entry_price) / self.entry_price * 100 * self.position
            reward = pnl
            self.trades.append({'pnl': pnl, 'bars': self.idx - self.entry_bar,
                               'mfe': self.max_profit, 'mae': self.max_loss})
            self.position = 0

        return self._get_state(), reward, done, info

    def get_stats(self) -> Dict:
        if not self.trades:
            return {'trades': 0}

        pnls = [t['pnl'] for t in self.trades]
        bars = [t['bars'] for t in self.trades]

        return {
            'trades': len(self.trades),
            'win_rate': sum(1 for p in pnls if p > 0) / len(pnls) * 100,
            'total_pnl': sum(pnls),
            'avg_pnl': np.mean(pnls),
            'avg_bars': np.mean(bars),
            'avg_mfe': np.mean([t['mfe'] for t in self.trades]),
            'avg_mae': np.mean([t['mae'] for t in self.trades]),
        }


def train_sniper(
    run_dir: Path,
    n_episodes: int = 200,
    metrics_port: int = 8002,  # Different port from Berserker
):
    """Train Sniper strategy."""

    logger.info("=" * 70)
    logger.info("KINETRA SNIPER TRAINING")
    logger.info("=" * 70)
    logger.info("Strategy: Laminar flow, stable trends, patient entries")

    # Setup
    data_dir = run_dir / "data"
    models_dir = run_dir / "models"
    logs_dir = run_dir / "logs"
    models_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

    # Start metrics server
    metrics = start_metrics_server(metrics_port)
    logger.info(f"Metrics: http://localhost:{metrics_port}/metrics")

    # Config
    config = SniperConfig(n_episodes=n_episodes)

    # Load data
    data_files = sorted(data_dir.glob("*.csv"))
    if not data_files:
        logger.error(f"No data in {data_dir}")
        return

    logger.info(f"Loading {len(data_files)} data files...")
    envs = []

    for path in data_files:
        try:
            df = pd.read_csv(path, parse_dates=['time'] if 'time' in pd.read_csv(path, nrows=1).columns else None)
            if 'time' in df.columns:
                df.set_index('time', inplace=True)

            # Compute sniper features
            fc = SniperFeatureComputer()
            features = fc.compute(df)

            env = SniperEnv(df, features, config)
            envs.append((path.stem, env))
            logger.info(f"  {path.stem}: {len(df)} bars, {env.state_dim} state dims")
        except Exception as e:
            logger.warning(f"  {path.stem}: Failed - {e}")

    if not envs:
        logger.error("No valid environments")
        return

    state_dim = envs[0][1].state_dim
    action_dim = envs[0][1].action_dim

    # Networks
    q_net = DQN(state_dim, action_dim, config.hidden_sizes).to(device)
    target_net = DQN(state_dim, action_dim, config.hidden_sizes).to(device)
    target_net.load_state_dict(q_net.state_dict())

    optimizer = torch.optim.Adam(q_net.parameters(), lr=config.learning_rate)
    buffer = ReplayBuffer(config.buffer_size, device)

    epsilon = config.epsilon_start
    total_steps = 0
    best_pnl = float('-inf')

    logger.info(f"\nTraining for {n_episodes} episodes...")
    logger.info("-" * 70)

    for episode in range(n_episodes):
        episode_stats = {'trades': 0, 'wins': 0, 'pnl': 0, 'bars_held': []}
        losses = []

        for inst_name, env in envs:
            state = env.reset()
            done = False

            while not done:
                # Epsilon-greedy action
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
                if 'trade_pnl' in info:
                    episode_stats['trades'] += 1
                    episode_stats['pnl'] += info['trade_pnl']
                    episode_stats['bars_held'].append(info['bars_held'])
                    if info['trade_pnl'] > 0:
                        episode_stats['wins'] += 1

                # Training step
                if len(buffer) >= config.batch_size:
                    batch = buffer.sample(config.batch_size)
                    states_t, actions_t, rewards_t, next_states_t, dones_t = batch

                    if isinstance(states_t, torch.Tensor):
                        states_t = torch.nan_to_num(states_t, nan=0.5)
                        next_states_t = torch.nan_to_num(next_states_t, nan=0.5)
                        states_t = torch.clamp(states_t, -10, 10)
                        next_states_t = torch.clamp(next_states_t, -10, 10)
                        rewards_t = torch.clamp(rewards_t, -10, 10)

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

        # Decay epsilon
        epsilon = max(config.epsilon_end, epsilon * config.epsilon_decay)

        # Episode metrics
        n_trades = episode_stats['trades']
        win_rate = (episode_stats['wins'] / n_trades * 100) if n_trades > 0 else 0
        total_pnl = episode_stats['pnl']
        avg_loss = np.mean(losses) if losses else 0
        avg_bars = np.mean(episode_stats['bars_held']) if episode_stats['bars_held'] else 0

        # Update Prometheus
        rl_metrics = RLMetrics(
            episode=episode + 1,
            total_trades=n_trades,
            win_rate=win_rate,
            total_pnl=total_pnl,
            avg_pnl=total_pnl / n_trades if n_trades > 0 else 0,
            avg_mfe=0, avg_mae=0, mfe_mae_ratio=0, mfe_captured=0,
            epsilon=epsilon,
            loss=avg_loss,
            reward=0,
        )
        metrics.update_rl_metrics(rl_metrics)
        metrics.complete_episode()

        # Save best
        if total_pnl > best_pnl:
            best_pnl = total_pnl
            torch.save(q_net.state_dict(), models_dir / "best_sniper.pt")

        # Save checkpoint
        if (episode + 1) % 10 == 0:
            torch.save({
                'episode': episode + 1,
                'model_state_dict': q_net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_pnl': best_pnl,
            }, models_dir / "checkpoint.pt")

        logger.info(
            f"Ep {episode+1:3d}/{n_episodes} | "
            f"Trades: {n_trades:3d} | "
            f"WR: {win_rate:5.1f}% | "
            f"PnL: {total_pnl:+8.2f}% | "
            f"AvgBars: {avg_bars:5.1f} | "
            f"Loss: {avg_loss:.4f} | "
            f"ε: {epsilon:.3f}"
        )

    logger.info("-" * 70)
    logger.info(f"Training complete! Best P&L: {best_pnl:.2f}%")
    logger.info(f"Models: {models_dir}")


def main():
    parser = argparse.ArgumentParser(description="Sniper Strategy Training")
    parser.add_argument("--run", type=str, help="Run name")
    parser.add_argument("--new-run", action="store_true", help="Create new run")
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--port", type=int, default=8002)
    args = parser.parse_args()

    dm = DataManager()

    if args.new_run:
        run_dir = dm.create_run(strategy="sniper")
    elif args.run:
        run_dir = dm.get_run(args.run)
        if run_dir is None:
            # Create it
            run_dir = dm.create_run(name=args.run, strategy="sniper")
    else:
        run_dir = dm.create_run(strategy="sniper")

    train_sniper(
        run_dir=run_dir,
        n_episodes=args.episodes,
        metrics_port=args.port,
    )


if __name__ == "__main__":
    main()
