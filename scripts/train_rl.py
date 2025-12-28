#!/usr/bin/env python3
"""
Automated RL Training Loop

Continuous training cycle:
1. Collect experiences via backtest (VIRTUAL mode)
2. Train policy network on replay buffer
3. Checkpoint periodically
4. Evaluate and log metrics

Supports:
- Synthetic data (for testing)
- Real MT5 data via bridge
- Atomic checkpointing for crash recovery

Usage:
    python scripts/train_rl.py --episodes 100
    python scripts/train_rl.py --live  # Use MT5 bridge for live data
"""

import sys
import os
import argparse
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from collections import deque
import random

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kinetra import (
    # Persistence
    AtomicCheckpointer,
    CheckpointType,
    # Friction
    FrictionModel,
    TradingMode,
    get_symbol_spec,
    # MT5 Bridge
    MT5Bridge,
)

from kinetra.physics_v7 import (
    compute_oscillator_state,
    compute_fractal_dimension_katz,
    compute_sample_entropy,
    compute_ftle_fast,
)

# Try to import PyTorch
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("PyTorch not installed. Using numpy-based simple policy.")


# =============================================================================
# REPLAY BUFFER
# =============================================================================

class ReplayBuffer:
    """Experience replay buffer for off-policy RL."""

    def __init__(self, capacity: int = 100000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> Tuple:
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(dones),
        )

    def __len__(self):
        return len(self.buffer)


# =============================================================================
# NEURAL NETWORK POLICY (PyTorch)
# =============================================================================

if HAS_TORCH:
    class DQN(nn.Module):
        """Deep Q-Network for trading actions."""

        def __init__(self, state_dim: int = 20, action_dim: int = 4, hidden_dim: int = 128):
            super().__init__()
            self.fc1 = nn.Linear(state_dim, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, hidden_dim)
            self.fc3 = nn.Linear(hidden_dim, hidden_dim // 2)
            self.out = nn.Linear(hidden_dim // 2, action_dim)

        def forward(self, x):
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = F.relu(self.fc3(x))
            return self.out(x)


# =============================================================================
# SIMPLE NUMPY POLICY (fallback)
# =============================================================================

class SimplePolicy:
    """Simple linear policy using numpy (fallback if no PyTorch)."""

    def __init__(self, state_dim: int = 20, action_dim: int = 4):
        self.state_dim = state_dim
        self.action_dim = action_dim
        # Simple linear weights
        self.weights = np.random.randn(state_dim, action_dim) * 0.1
        self.bias = np.zeros(action_dim)
        self.learning_rate = 0.001

    def predict(self, state: np.ndarray) -> np.ndarray:
        """Get Q-values for state."""
        return np.dot(state, self.weights) + self.bias

    def act(self, state: np.ndarray, epsilon: float = 0.1) -> int:
        """Epsilon-greedy action selection."""
        if np.random.random() < epsilon:
            return np.random.randint(self.action_dim)
        q_values = self.predict(state)
        return int(np.argmax(q_values))

    def update(self, states, actions, targets):
        """Simple gradient update."""
        for state, action, target in zip(states, actions, targets):
            q_values = self.predict(state)
            error = target - q_values[action]
            # Update weights for the taken action
            self.weights[:, action] += self.learning_rate * error * state
            self.bias[action] += self.learning_rate * error

    def state_dict(self):
        return {'weights': self.weights, 'bias': self.bias}

    def load_state_dict(self, state):
        self.weights = state['weights']
        self.bias = state['bias']


# =============================================================================
# DATA GENERATION
# =============================================================================

def generate_synthetic_episode(
    n_bars: int = 500,
    seed: Optional[int] = None
) -> pd.DataFrame:
    """Generate a single episode of synthetic data."""
    if seed is not None:
        np.random.seed(seed)

    base_price = 1.1000
    returns = np.random.randn(n_bars) * 0.001  # 0.1% volatility

    # Add some trend
    trend = np.cumsum(np.random.randn(n_bars) * 0.0001)
    returns += np.diff(np.concatenate([[0], trend]))

    close = base_price * np.exp(np.cumsum(returns))
    high = close * (1 + np.abs(np.random.randn(n_bars) * 0.0003))
    low = close * (1 - np.abs(np.random.randn(n_bars) * 0.0003))
    open_ = np.roll(close, 1)
    open_[0] = close[0]

    high = np.maximum(high, np.maximum(open_, close))
    low = np.minimum(low, np.minimum(open_, close))

    volume = 1000 * (1 + np.random.exponential(0.3, n_bars))

    dates = pd.date_range(start='2024-01-01', periods=n_bars, freq='h')

    return pd.DataFrame({
        'Open': open_,
        'High': high,
        'Low': low,
        'Close': close,
        'Volume': volume,
    }, index=dates)


def compute_state(df: pd.DataFrame, idx: int, lookback: int = 50) -> np.ndarray:
    """Compute state vector for a given bar."""
    start = max(0, idx - lookback + 1)
    window = df.iloc[start:idx+1]

    if len(window) < 10:
        return np.zeros(20)

    close = window['Close'].values
    high = window['High'].values
    low = window['Low'].values
    volume = window['Volume'].values

    # Oscillator state
    osc = compute_oscillator_state(high, low, close, volume, lookback=min(20, len(window)))

    # Features
    mass = osc['mass'][-1] if len(osc['mass']) > 0 else 0
    force = osc['force'][-1] if len(osc['force']) > 0 else 0
    accel = osc['acceleration'][-1] if len(osc['acceleration']) > 0 else 0
    velocity = osc['velocity'][-1] if len(osc['velocity']) > 0 else 0
    displacement = osc['displacement'][-1] if len(osc['displacement']) > 0 else 0
    symc = osc['symc'][-1] if len(osc['symc']) > 0 else 1.0

    fd = compute_fractal_dimension_katz(close)[-1] if len(close) >= 10 else 1.5
    se = compute_sample_entropy(close, m=2)[-1] if len(close) >= 20 else 0.5
    ftle = compute_ftle_fast(close, window=min(20, len(close)))[-1] if len(close) >= 20 else 0

    returns = np.diff(close) / close[:-1] if len(close) > 1 else [0]
    ret_mean = np.mean(returns)
    ret_std = np.std(returns) + 1e-10
    vol_ratio = volume[-1] / np.mean(volume) if np.mean(volume) > 0 else 1

    state = np.array([
        np.clip(mass / 1e6, -5, 5),
        np.clip(force / 1e3, -5, 5),
        np.clip(accel * 100, -5, 5),
        np.clip(velocity * 100, -5, 5),
        np.clip(displacement * 10, -5, 5),
        np.clip(symc, 0, 5),
        np.clip(fd - 1.5, -1, 1),
        np.clip(se, 0, 3),
        np.clip(ftle * 10, -2, 2),
        np.clip(ret_mean * 1000, -5, 5),
        np.clip(ret_std * 100, 0, 5),
        np.clip(vol_ratio - 1, -2, 2),
        1.0 if symc < 0.8 else 0.0,
        1.0 if 0.8 <= symc <= 1.2 else 0.0,
        1.0 if symc > 1.2 else 0.0,
        np.clip(np.sum(returns[-5:]) * 100 if len(returns) >= 5 else 0, -5, 5),
        np.clip(np.sum(returns[-10:]) * 100 if len(returns) >= 10 else 0, -5, 5),
        np.clip((np.mean(high - low) / close[-1]) * 100, 0, 5),
        np.clip((close[-1] - np.min(low)) / (np.max(high) - np.min(low) + 1e-10), 0, 1),
        0.0,  # Placeholder
    ])

    return state


# =============================================================================
# TRAINING LOOP
# =============================================================================

class RLTrainer:
    """Automated RL training with checkpointing."""

    def __init__(
        self,
        checkpoint_dir: str = "./checkpoints",
        state_dim: int = 20,
        action_dim: int = 4,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay: int = 10000,
        batch_size: int = 64,
        target_update: int = 100,
        learning_rate: float = 0.001,
    ):
        self.checkpoint_dir = checkpoint_dir
        self.checkpointer = AtomicCheckpointer(checkpoint_dir)

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update = target_update

        self.replay_buffer = ReplayBuffer(capacity=100000)
        self.step = 0
        self.episode = 0

        # Initialize policy
        if HAS_TORCH:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.policy = DQN(state_dim, action_dim).to(self.device)
            self.target = DQN(state_dim, action_dim).to(self.device)
            self.target.load_state_dict(self.policy.state_dict())
            self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)
        else:
            self.policy = SimplePolicy(state_dim, action_dim)
            self.target = None
            self.optimizer = None

        # Metrics
        self.episode_rewards = []
        self.episode_lengths = []

        # Try to resume from checkpoint
        self._try_resume()

    def _try_resume(self):
        """Try to resume from checkpoint."""
        state = self.checkpointer.load_latest_rl_model()
        if state is not None:
            print(f"Resuming from step {state['step']}, episode {state['episode']}")
            self.step = state['step']
            self.episode = state['episode']

            if HAS_TORCH:
                self.policy.load_state_dict(state['model'])
                if state['optimizer']:
                    self.optimizer.load_state_dict(state['optimizer'])
            else:
                self.policy.load_state_dict(state['model'])

            # Load replay buffer
            buffer_data = self.checkpointer.load_latest_replay_buffer()
            if buffer_data and 'experiences' in buffer_data:
                for exp in buffer_data['experiences']:
                    state_arr = np.array(exp['state'])
                    next_state_arr = np.array(exp['next_state'])
                    self.replay_buffer.push(
                        state_arr, exp['action'], exp['reward'],
                        next_state_arr, exp['done']
                    )
                print(f"Loaded {len(self.replay_buffer)} experiences")

    def get_epsilon(self) -> float:
        """Get current epsilon for exploration."""
        return self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
            np.exp(-self.step / self.epsilon_decay)

    def select_action(self, state: np.ndarray) -> int:
        """Select action using epsilon-greedy."""
        epsilon = self.get_epsilon()

        if np.random.random() < epsilon:
            return np.random.randint(self.action_dim)

        if HAS_TORCH:
            with torch.no_grad():
                state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.policy(state_t)
                return int(q_values.argmax(dim=1).item())
        else:
            return self.policy.act(state, epsilon=0)

    def train_step(self) -> Optional[float]:
        """Perform one training step."""
        if len(self.replay_buffer) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        if HAS_TORCH:
            states_t = torch.FloatTensor(states).to(self.device)
            actions_t = torch.LongTensor(actions).to(self.device)
            rewards_t = torch.FloatTensor(rewards).to(self.device)
            next_states_t = torch.FloatTensor(next_states).to(self.device)
            dones_t = torch.FloatTensor(dones).to(self.device)

            # Current Q values
            q_values = self.policy(states_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)

            # Target Q values
            with torch.no_grad():
                next_q_values = self.target(next_states_t).max(dim=1)[0]
                target_q_values = rewards_t + self.gamma * next_q_values * (1 - dones_t)

            # Loss
            loss = F.mse_loss(q_values, target_q_values)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            return loss.item()
        else:
            # Simple numpy update
            targets = rewards + self.gamma * np.max(
                [self.policy.predict(ns) for ns in next_states], axis=1
            ) * (1 - dones)
            self.policy.update(states, actions, targets)
            return 0.0

    def run_episode(self, df: pd.DataFrame, symbol: str = "EURUSD") -> Dict:
        """Run one training episode."""
        spec = get_symbol_spec(symbol)
        friction_model = FrictionModel(spec, mode=TradingMode.VIRTUAL)

        warmup = 100
        position = 0
        entry_price = 0
        entry_bar = 0
        episode_reward = 0
        n_trades = 0

        for i in range(warmup, len(df) - 1):
            state = compute_state(df, i)
            action = self.select_action(state)

            close = df['Close'].iloc[i]
            next_close = df['Close'].iloc[i + 1]
            reward = 0
            done = False

            # Execute action
            if action == 1 and position == 0:  # Buy
                position = 1
                entry_price = close
                entry_bar = i
                reward = -0.01  # Small entry cost

            elif action == 2 and position == 0:  # Sell
                position = -1
                entry_price = close
                entry_bar = i
                reward = -0.01

            elif action == 3 and position != 0:  # Close
                pnl_pct = ((close / entry_price) - 1) * 100 * position
                reward = pnl_pct - 0.02  # Subtract friction
                n_trades += 1
                position = 0

            elif action == 0 and position != 0:  # Hold
                unrealized = ((next_close / close) - 1) * 100 * position
                reward = unrealized * 0.1

            # Next state
            next_state = compute_state(df, i + 1)

            # Store experience
            self.replay_buffer.push(state, action, reward, next_state, done)

            # Train
            if self.step % 4 == 0:
                self.train_step()

            # Update target network
            if HAS_TORCH and self.step % self.target_update == 0:
                self.target.load_state_dict(self.policy.state_dict())

            episode_reward += reward
            self.step += 1

        self.episode += 1

        return {
            'episode': self.episode,
            'reward': episode_reward,
            'trades': n_trades,
            'epsilon': self.get_epsilon(),
            'buffer_size': len(self.replay_buffer),
        }

    def save_checkpoint(self):
        """Save checkpoint atomically."""
        if HAS_TORCH:
            model_state = self.policy.state_dict()
            opt_state = self.optimizer.state_dict()
        else:
            model_state = self.policy.state_dict()
            opt_state = None

        self.checkpointer.save_rl_model(
            model=model_state,
            optimizer=opt_state,
            step=self.step,
            episode=self.episode,
            metrics={'epsilon': self.get_epsilon()}
        )

        # Save replay buffer
        experiences = []
        for state, action, reward, next_state, done in list(self.replay_buffer.buffer)[-10000:]:
            experiences.append({
                'state': state.tolist(),
                'action': int(action),
                'reward': float(reward),
                'next_state': next_state.tolist(),
                'done': bool(done),
            })

        self.checkpointer.save(
            CheckpointType.REPLAY_BUFFER,
            {'experiences': experiences, 'step': self.step}
        )

    def train(self, n_episodes: int = 100, checkpoint_every: int = 10):
        """Main training loop."""
        print(f"\nStarting training from episode {self.episode}")
        print(f"Using {'PyTorch' if HAS_TORCH else 'NumPy'} backend")
        print(f"Epsilon: {self.get_epsilon():.3f}")
        print("-" * 60)

        for ep in range(n_episodes):
            # Generate episode data
            df = generate_synthetic_episode(n_bars=500, seed=self.episode + ep)

            # Run episode
            metrics = self.run_episode(df)

            self.episode_rewards.append(metrics['reward'])

            # Log
            avg_reward = np.mean(self.episode_rewards[-100:])
            print(f"Ep {metrics['episode']:4d} | "
                  f"Reward: {metrics['reward']:+7.2f} | "
                  f"Avg: {avg_reward:+7.2f} | "
                  f"Trades: {metrics['trades']:3d} | "
                  f"ε: {metrics['epsilon']:.3f} | "
                  f"Buffer: {metrics['buffer_size']}")

            # Checkpoint
            if (ep + 1) % checkpoint_every == 0:
                self.save_checkpoint()
                print(f"  [Checkpoint saved]")

        # Final checkpoint
        self.save_checkpoint()
        print("\nTraining complete!")


def main():
    parser = argparse.ArgumentParser(description='Automated RL Training')
    parser.add_argument('--episodes', type=int, default=100, help='Number of episodes')
    parser.add_argument('--checkpoint-dir', default='./checkpoints', help='Checkpoint directory')
    parser.add_argument('--checkpoint-every', type=int, default=10, help='Checkpoint frequency')
    parser.add_argument('--live', action='store_true', help='Use MT5 bridge for live data')
    args = parser.parse_args()

    print("=" * 60)
    print("KINETRA RL TRAINER")
    print("VIRTUAL MODE - Exploring freely")
    print("=" * 60)

    if args.live:
        # Test MT5 bridge
        print("\nTesting MT5 bridge connection...")
        bridge = MT5Bridge(mode="auto")
        if bridge.connect():
            print(f"  Mode: {bridge.mode}")
            if bridge.mode in ["direct", "bridge"]:
                print("  Live data available!")
            else:
                print("  Offline mode - using synthetic data")
        else:
            print("  Bridge not available - using synthetic data")

    # Create trainer
    trainer = RLTrainer(checkpoint_dir=args.checkpoint_dir)

    # Train
    trainer.train(n_episodes=args.episodes, checkpoint_every=args.checkpoint_every)


if __name__ == "__main__":
    main()
