"""
GPU-Accelerated RL Trainer for Physics-Based Trading

Uses PyTorch with ROCm (AMD GPU) or CUDA (NVIDIA).
Trains across multiple timeframes and instruments.

Let RL discover:
- What predicts fat candles (magnitude)
- What predicts continuation vs reversal (direction)
- Optimal exit timing (energy recovery)

No hardcoded rules - pure feature learning.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import time

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch not available. Install with: pip install torch")


@dataclass
class TrainingConfig:
    """Configuration for RL training."""
    # Network
    hidden_sizes: Tuple[int, ...] = (128, 64, 32)
    learning_rate: float = 0.001
    gamma: float = 0.99  # Discount factor

    # Exploration
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay: float = 0.995

    # Training
    batch_size: int = 64
    buffer_size: int = 50000
    target_update_freq: int = 100
    n_episodes: int = 200

    # Device
    device: str = 'auto'  # 'auto', 'cuda', 'cpu'


class PhysicsFeatureComputer:
    """Compute ALL physics features for RL - no hardcoded rules."""

    def __init__(self, lookback: int = 20):
        self.lookback = lookback

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute all physics features as percentiles."""
        result = df.copy()
        window = min(500, len(df))

        # === CORE PHYSICS ===
        velocity = df['close'].pct_change()
        result['energy'] = 0.5 * (velocity ** 2)

        vol = velocity.rolling(self.lookback).std()
        mean_abs = velocity.abs().rolling(self.lookback).mean()
        result['damping'] = vol / (mean_abs + 1e-10)

        # Entropy (return dispersion)
        result['entropy'] = velocity.rolling(self.lookback).std() / (velocity.rolling(self.lookback).mean().abs() + 1e-10)

        # === DERIVATIVES ===
        result['acceleration'] = velocity.diff()
        result['jerk'] = result['acceleration'].diff()

        momentum = df['close'].pct_change(self.lookback)
        result['impulse'] = momentum.diff(5)

        # === ORDER FLOW ===
        bar_range = (df['high'] - df['low']).clip(lower=1e-10)
        result['liquidity'] = df['volume'] / (bar_range * df['close'] / 100)

        bp = (df['close'] - df['low']) / bar_range
        result['buying_pressure'] = bp.rolling(5).mean()

        # === FLOW DYNAMICS ===
        bar_range_pct = bar_range / df['close']
        volatility = velocity.rolling(self.lookback).std().clip(lower=1e-10)
        volume_norm = df['volume'] / df['volume'].rolling(self.lookback).mean().clip(lower=1e-10)

        # Reynolds number
        result['reynolds'] = ((velocity.abs() * bar_range_pct * volume_norm) / volatility).rolling(self.lookback).mean()

        # Viscosity
        result['viscosity'] = (bar_range_pct / volume_norm.clip(lower=1e-10)).rolling(self.lookback).mean()

        # === ROTATIONAL/CYCLICAL ===
        price_mean = df['close'].rolling(self.lookback).mean()
        price_std = df['close'].rolling(self.lookback).std().clip(lower=1e-10)
        angular_pos = (df['close'] - price_mean) / price_std
        angular_vel = angular_pos.diff()
        result['angular_momentum'] = angular_pos * angular_vel

        # === POTENTIAL ENERGY ===
        avg_range = bar_range.rolling(self.lookback).mean()
        range_compression = (1 - bar_range / avg_range.clip(lower=1e-10)).clip(lower=0)
        result['potential_energy'] = range_compression * volatility

        # === MARKET TORQUE (bull/bear tension creating rotational force) ===
        # From physics: torque = imbalance × acceleration
        # High imbalance + high acceleration = price trajectory "bends" sharply
        imbalance = (result['buying_pressure'] - 0.5) * 2  # Map 0-1 to -1 to +1
        result['torque'] = imbalance * result['acceleration'] * 1000  # Scale for visibility

        # === MARKET REYNOLDS NUMBER (alternative formulation) ===
        # Re = momentum / (viscosity × friction)
        # Low Re = laminar (trend), High Re = turbulent (chaos)
        result['market_reynolds'] = result['energy'] / (result['viscosity'] * result['damping'] + 1e-10)

        # === CONTEXT ===
        rolling_high = df['high'].rolling(self.lookback).max()
        rolling_low = df['low'].rolling(self.lookback).min()
        result['range_position'] = (df['close'] - rolling_low) / (rolling_high - rolling_low + 1e-10)

        return_sign = np.sign(velocity)
        result['flow_consistency'] = return_sign.rolling(5).apply(
            lambda x: (x == x.iloc[-1]).mean() if len(x) > 0 else 0.5, raw=False
        )

        result['roc'] = df['close'].pct_change(self.lookback)

        # Momentum direction (let RL learn what to do with it)
        result['momentum_dir'] = np.sign(df['close'].pct_change(5))

        # === CONVERT TO PERCENTILES ===
        feature_cols = [
            'energy', 'damping', 'entropy', 'acceleration', 'jerk', 'impulse',
            'liquidity', 'buying_pressure', 'reynolds', 'viscosity',
            'angular_momentum', 'potential_energy', 'torque', 'market_reynolds',
            'range_position', 'flow_consistency', 'roc'
        ]

        for col in feature_cols:
            result[f'{col}_pct'] = result[col].rolling(window, min_periods=self.lookback).apply(
                lambda x: (x.iloc[-1] > x.iloc[:-1]).mean() if len(x) > 1 else 0.5, raw=False
            ).fillna(0.5)

        # Keep momentum_dir as-is (not percentile)
        result['momentum_dir_pct'] = (result['momentum_dir'] + 1) / 2  # Map -1,0,1 to 0,0.5,1

        return result.fillna(0.5)


if TORCH_AVAILABLE:

    class DQN(nn.Module):
        """Deep Q-Network for physics-based trading."""

        def __init__(self, state_dim: int, action_dim: int, hidden_sizes: Tuple[int, ...] = (128, 64, 32)):
            super().__init__()

            layers = []
            prev_size = state_dim

            for size in hidden_sizes:
                layers.append(nn.Linear(prev_size, size))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(0.1))
                prev_size = size

            layers.append(nn.Linear(prev_size, action_dim))

            self.network = nn.Sequential(*layers)

        def forward(self, x):
            return self.network(x)


    class ReplayBuffer:
        """Experience replay buffer."""

        def __init__(self, capacity: int, device: torch.device):
            self.capacity = capacity
            self.device = device
            self.buffer = []
            self.position = 0

        def push(self, state, action, reward, next_state, done):
            if len(self.buffer) < self.capacity:
                self.buffer.append(None)
            self.buffer[self.position] = (state, action, reward, next_state, done)
            self.position = (self.position + 1) % self.capacity

        def sample(self, batch_size: int):
            indices = np.random.choice(len(self.buffer), batch_size, replace=False)
            batch = [self.buffer[i] for i in indices]

            states = torch.FloatTensor(np.array([b[0] for b in batch])).to(self.device)
            actions = torch.LongTensor([b[1] for b in batch]).to(self.device)
            rewards = torch.FloatTensor([b[2] for b in batch]).to(self.device)
            next_states = torch.FloatTensor(np.array([b[3] for b in batch])).to(self.device)
            dones = torch.FloatTensor([b[4] for b in batch]).to(self.device)

            return states, actions, rewards, next_states, dones

        def __len__(self):
            return len(self.buffer)


    class TradingEnv:
        """Trading environment with all physics features."""

        # Actions: HOLD, LONG, SHORT, EXIT
        HOLD = 0
        LONG = 1
        SHORT = 2
        EXIT = 3

        def __init__(self, df: pd.DataFrame, feature_df: pd.DataFrame):
            self.df = df
            self.feature_df = feature_df
            self.feature_cols = [c for c in feature_df.columns if c.endswith('_pct')]

            self.state_dim = len(self.feature_cols) + 4  # + position info
            self.action_dim = 4

            self.reset()

        def reset(self) -> np.ndarray:
            self.idx = 100
            self.position = 0  # -1 short, 0 flat, 1 long
            self.entry_price = 0.0
            self.entry_bar = 0
            self.mfe = 0.0
            self.mae = 0.0
            self.trades = []
            return self._get_state()

        def _get_state(self) -> np.ndarray:
            if self.idx >= len(self.feature_df):
                return np.zeros(self.state_dim)

            features = self.feature_df.iloc[self.idx][self.feature_cols].values.astype(float)

            # Position info
            pos_dir = float(self.position)
            bars_held = float(self.idx - self.entry_bar) / 10 if self.position != 0 else 0

            if self.position != 0:
                current_price = self.df.iloc[self.idx]['close']
                pnl = (current_price - self.entry_price) / self.entry_price * self.position
            else:
                pnl = 0.0

            mfe_ratio = self.mfe / (self.mae + 1e-6) if self.mae > 0 else 0

            return np.concatenate([features, [pos_dir, bars_held, pnl, mfe_ratio]])

        def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
            reward = 0.0
            info = {}

            current = self.df.iloc[self.idx]
            high, low, close = current['high'], current['low'], current['close']

            # Update MFE/MAE if in position
            if self.position != 0:
                if self.position == 1:
                    self.mfe = max(self.mfe, (high - self.entry_price) / self.entry_price * 100)
                    self.mae = max(self.mae, (self.entry_price - low) / self.entry_price * 100)
                else:
                    self.mfe = max(self.mfe, (self.entry_price - low) / self.entry_price * 100)
                    self.mae = max(self.mae, (high - self.entry_price) / self.entry_price * 100)

            # Execute action
            if action == self.LONG and self.position == 0:
                self.position = 1
                self.entry_price = close
                self.entry_bar = self.idx
                self.mfe = 0.0
                self.mae = 0.0

            elif action == self.SHORT and self.position == 0:
                self.position = -1
                self.entry_price = close
                self.entry_bar = self.idx
                self.mfe = 0.0
                self.mae = 0.0

            elif action == self.EXIT and self.position != 0:
                pnl = (close - self.entry_price) / self.entry_price * 100 * self.position

                # Reward based on P&L, MFE capture, MAE
                mfe_capture = pnl / self.mfe if self.mfe > 0.01 else 0
                efficiency = self.mfe / (self.mae + 0.01)

                # Composite reward: P&L + bonus for efficiency
                reward = pnl + 0.5 * mfe_capture - 0.3 * self.mae

                self.trades.append({
                    'pnl': pnl,
                    'mfe': self.mfe,
                    'mae': self.mae,
                    'mfe_capture': mfe_capture * 100,
                    'efficiency': efficiency,
                    'bars': self.idx - self.entry_bar,
                })

                self.position = 0

            elif action == self.HOLD and self.position != 0:
                # Small time penalty
                reward = -0.001

            # Move to next bar
            self.idx += 1
            done = self.idx >= len(self.df) - 1

            # Force close at end
            if done and self.position != 0:
                pnl = (self.df.iloc[self.idx]['close'] - self.entry_price) / self.entry_price * 100 * self.position
                reward = pnl
                self.trades.append({'pnl': pnl, 'mfe': self.mfe, 'mae': self.mae})
                self.position = 0

            return self._get_state(), reward, done, info

        def get_stats(self) -> Dict:
            if not self.trades:
                return {'trades': 0}

            pnls = [t['pnl'] for t in self.trades]
            mfes = [t['mfe'] for t in self.trades]
            maes = [t['mae'] for t in self.trades]

            return {
                'trades': len(self.trades),
                'win_rate': sum(1 for p in pnls if p > 0) / len(pnls) * 100,
                'total_pnl': sum(pnls),
                'avg_pnl': np.mean(pnls),
                'avg_mfe': np.mean(mfes),
                'avg_mae': np.mean(maes),
                'mfe_mae_ratio': np.mean(mfes) / (np.mean(maes) + 0.01),
            }


    class GPUTrainer:
        """GPU-accelerated DQN trainer."""

        def __init__(self, config: TrainingConfig = None):
            self.config = config or TrainingConfig()

            # Detect device
            if self.config.device == 'auto':
                if torch.cuda.is_available():
                    self.device = torch.device('cuda')
                    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
                else:
                    self.device = torch.device('cpu')
                    print("Using CPU")
            else:
                self.device = torch.device(self.config.device)

            self.policy_net = None
            self.target_net = None
            self.optimizer = None
            self.buffer = None
            self.epsilon = self.config.epsilon_start

        def init_networks(self, state_dim: int, action_dim: int):
            """Initialize networks for given dimensions."""
            self.policy_net = DQN(state_dim, action_dim, self.config.hidden_sizes).to(self.device)
            self.target_net = DQN(state_dim, action_dim, self.config.hidden_sizes).to(self.device)
            self.target_net.load_state_dict(self.policy_net.state_dict())

            self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.config.learning_rate)
            self.buffer = ReplayBuffer(self.config.buffer_size, self.device)

        def select_action(self, state: np.ndarray, training: bool = True) -> int:
            if training and np.random.random() < self.epsilon:
                return np.random.randint(4)

            with torch.no_grad():
                state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.policy_net(state_t)
                return q_values.argmax().item()

        def train_step(self) -> float:
            if len(self.buffer) < self.config.batch_size:
                return 0.0

            states, actions, rewards, next_states, dones = self.buffer.sample(self.config.batch_size)

            # Current Q values
            q_values = self.policy_net(states).gather(1, actions.unsqueeze(1))

            # Target Q values
            with torch.no_grad():
                next_q = self.target_net(next_states).max(1)[0]
                target_q = rewards + self.config.gamma * next_q * (1 - dones)

            # Loss
            loss = F.mse_loss(q_values.squeeze(), target_q)

            # Optimize
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
            self.optimizer.step()

            return loss.item()

        def update_target(self):
            self.target_net.load_state_dict(self.policy_net.state_dict())

        def decay_epsilon(self):
            self.epsilon = max(self.config.epsilon_end, self.epsilon * self.config.epsilon_decay)

        def train(self, envs: List[TradingEnv], n_episodes: int = None) -> List[Dict]:
            """Train on multiple environments (instruments/timeframes)."""
            n_episodes = n_episodes or self.config.n_episodes

            # Initialize networks from first env
            self.init_networks(envs[0].state_dim, envs[0].action_dim)

            stats = []
            step_count = 0

            for episode in range(n_episodes):
                episode_reward = 0
                episode_loss = 0
                n_updates = 0

                # Train on each environment
                for env in envs:
                    state = env.reset()

                    while True:
                        action = self.select_action(state)
                        next_state, reward, done, _ = env.step(action)

                        self.buffer.push(state, action, reward, next_state, done)

                        loss = self.train_step()
                        if loss > 0:
                            episode_loss += loss
                            n_updates += 1

                        step_count += 1
                        if step_count % self.config.target_update_freq == 0:
                            self.update_target()

                        state = next_state
                        episode_reward += reward

                        if done:
                            break

                self.decay_epsilon()

                # Aggregate stats across envs
                avg_stats = {
                    'episode': episode,
                    'total_reward': episode_reward,
                    'avg_loss': episode_loss / n_updates if n_updates > 0 else 0,
                    'epsilon': self.epsilon,
                    'trades': sum(e.get_stats().get('trades', 0) for e in envs),
                    'avg_pnl': np.mean([e.get_stats().get('avg_pnl', 0) for e in envs if e.get_stats().get('trades', 0) > 0]),
                    'avg_win_rate': np.mean([e.get_stats().get('win_rate', 50) for e in envs if e.get_stats().get('trades', 0) > 0]),
                }
                stats.append(avg_stats)

                if (episode + 1) % 20 == 0:
                    print(f"Episode {episode + 1}/{n_episodes}: "
                          f"Trades={avg_stats['trades']}, "
                          f"WinRate={avg_stats['avg_win_rate']:.1f}%, "
                          f"PnL={avg_stats['avg_pnl']:.4f}%, "
                          f"ε={self.epsilon:.3f}")

            return stats

        def get_feature_importance(self, feature_names: List[str]) -> pd.DataFrame:
            """Get feature importance from first layer weights."""
            if self.policy_net is None:
                return pd.DataFrame()

            with torch.no_grad():
                weights = self.policy_net.network[0].weight.cpu().numpy()

            importance = np.abs(weights).mean(axis=0)
            importance = importance / importance.sum()

            return pd.DataFrame({
                'feature': feature_names[:len(importance)],
                'importance': importance
            }).sort_values('importance', ascending=False)


def train_across_instruments(
    data_paths: List[str],
    config: TrainingConfig = None,
) -> Tuple['GPUTrainer', List[Dict]]:
    """
    Train RL across multiple instruments/timeframes.

    Let RL discover:
    - Fat candle probability
    - Continuation vs reversal
    - Optimal exit timing
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch required. Install with: pip install torch")

    config = config or TrainingConfig()
    feature_computer = PhysicsFeatureComputer()

    # Load and prepare all environments
    envs = []
    for path in data_paths:
        print(f"Loading: {Path(path).name}")

        # Load data (assuming CSV format)
        df = pd.read_csv(path)
        if 'time' in df.columns:
            df['time'] = pd.to_datetime(df['time'])
            df.set_index('time', inplace=True)

        # Compute features
        feature_df = feature_computer.compute(df)

        # Create environment
        env = TradingEnv(df, feature_df)
        envs.append(env)

    print(f"\nTraining on {len(envs)} instruments/timeframes")
    print(f"State dimension: {envs[0].state_dim}")
    print(f"Action dimension: {envs[0].action_dim}")

    # Train
    trainer = GPUTrainer(config)
    stats = trainer.train(envs, config.n_episodes)

    # Feature importance
    feature_names = [c.replace('_pct', '') for c in envs[0].feature_cols] + ['pos_dir', 'bars_held', 'pnl', 'mfe_ratio']
    importance = trainer.get_feature_importance(feature_names)

    print("\n" + "=" * 60)
    print("FEATURE IMPORTANCE (What RL learned)")
    print("=" * 60)
    for _, row in importance.head(15).iterrows():
        print(f"  {row['feature']:>20}: {row['importance']:.1%}")

    return trainer, stats


if __name__ == "__main__":
    # Test with BTCUSD data
    project_root = Path(__file__).parent.parent
    csv_files = list(project_root.glob("*BTCUSD*.csv"))

    if csv_files:
        trainer, stats = train_across_instruments(
            [str(f) for f in csv_files],
            TrainingConfig(n_episodes=50)
        )
    else:
        print("No CSV files found")
