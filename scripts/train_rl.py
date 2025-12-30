#!/usr/bin/env python3
"""
Automated RL Training Loop - Physics Only

Continuous training cycle:
1. Collect experiences via backtest (VIRTUAL mode)
2. Train policy network on replay buffer
3. Checkpoint periodically
4. Evaluate and log metrics

PHYSICS ONLY: No RSI, MACD, Bollinger, etc.
Uses: Kinematics, Energy, Flow, Entropy, Field Theory

Supports:
- Real MT5 data from data/master/
- Synthetic data (fallback for testing)
- Atomic checkpointing for crash recovery

Usage:
    python scripts/train_rl.py --episodes 100
    python scripts/train_rl.py --episodes 100 --data-dir data/master --timeframe H1
    python scripts/train_rl.py --sync  # Git pull before training
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
import glob
import re

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

# Physics-only measurements (NO traditional indicators)
from kinetra.measurements import (
    MeasurementEngine,
    KinematicsMeasures,
    EnergyMeasures,
    FlowMeasures,
    ThermodynamicsMeasures,
    FieldMeasures,
    MicrostructureMeasures,
    PercentileNormalizer,
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


# =============================================================================
# REAL DATA LOADING (MT5 Format)
# =============================================================================

class RealDataLoader:
    """
    Load real market data from data/master/ directory.

    MT5 format: tab-separated with columns:
    <DATE>, <TIME>, <OPEN>, <HIGH>, <LOW>, <CLOSE>, <TICKVOL>, <VOL>, <SPREAD>
    """

    def __init__(self, data_dir: str = "data/master", timeframe: str = "H1"):
        self.data_dir = Path(data_dir)
        self.timeframe = timeframe
        self.files: List[Path] = []
        self.current_file_idx = 0
        self._scan_files()

    def _scan_files(self):
        """Scan for matching data files."""
        pattern = f"*_{self.timeframe}_*.csv"
        self.files = sorted(self.data_dir.glob(pattern))
        if self.files:
            print(f"Found {len(self.files)} {self.timeframe} data files")
        else:
            print(f"WARNING: No {self.timeframe} files in {self.data_dir}")

    def _parse_symbol(self, filepath: Path) -> str:
        """Extract symbol name from filename."""
        # Format: SYMBOL_TIMEFRAME_START_END.csv
        name = filepath.stem
        parts = name.split('_')
        if parts:
            return parts[0].replace('+', '')  # Remove + suffix
        return "UNKNOWN"

    def load_file(self, filepath: Path) -> Optional[pd.DataFrame]:
        """Load a single MT5 CSV file into standardized DataFrame."""
        try:
            # Read tab-separated data
            df = pd.read_csv(filepath, sep='\t')

            # Normalize column names
            df.columns = [c.lower().replace('<', '').replace('>', '') for c in df.columns]

            # Combine date + time into datetime
            if 'date' in df.columns and 'time' in df.columns:
                # Handle MT5 date format: 2024.01.02 -> 2024-01-02
                date_str = df['date'].astype(str).str.replace('.', '-', regex=False)
                df['datetime'] = pd.to_datetime(date_str + ' ' + df['time'].astype(str))
                df = df.set_index('datetime')

            # Rename to standard columns
            column_map = {
                'open': 'Open',
                'high': 'High',
                'low': 'Low',
                'close': 'Close',
                'tickvol': 'Volume',
                'vol': 'RealVolume',
                'spread': 'Spread',
            }

            df = df.rename(columns=column_map)

            # Ensure we have required columns
            required = ['Open', 'High', 'Low', 'Close', 'Volume']
            for col in required:
                if col not in df.columns:
                    if col == 'Volume' and 'RealVolume' in df.columns:
                        df['Volume'] = df['RealVolume']
                    else:
                        print(f"  Missing column {col} in {filepath.name}")
                        return None

            # Drop date/time text columns
            df = df.drop(columns=['date', 'time'], errors='ignore')

            return df[['Open', 'High', 'Low', 'Close', 'Volume'] +
                      [c for c in ['Spread', 'RealVolume'] if c in df.columns]]

        except Exception as e:
            print(f"  Error loading {filepath.name}: {e}")
            return None

    def get_random_episode(self, min_bars: int = 500) -> Tuple[pd.DataFrame, str]:
        """
        Get a random episode from available data.

        Returns (dataframe, symbol) tuple.
        """
        if not self.files:
            # Fallback to synthetic
            return generate_synthetic_episode(min_bars), "SYNTHETIC"

        # Shuffle files for variety
        available = list(self.files)
        random.shuffle(available)

        for filepath in available:
            df = self.load_file(filepath)
            if df is not None and len(df) >= min_bars:
                symbol = self._parse_symbol(filepath)
                # Take a random slice if file is large
                if len(df) > min_bars * 2:
                    max_start = len(df) - min_bars
                    start_idx = random.randint(0, max_start)
                    df = df.iloc[start_idx:start_idx + min_bars].reset_index(drop=True)
                return df, symbol

        # Fallback
        print("  No valid data files, using synthetic")
        return generate_synthetic_episode(min_bars), "SYNTHETIC"

    def get_all_symbols(self) -> List[str]:
        """Get list of all available symbols."""
        symbols = set()
        for f in self.files:
            symbols.add(self._parse_symbol(f))
        return sorted(symbols)


def git_sync():
    """Pull latest changes from git."""
    import subprocess
    print("\n[GIT SYNC] Pulling latest changes...")
    try:
        result = subprocess.run(
            ['git', 'pull', '--rebase'],
            capture_output=True,
            text=True,
            timeout=60
        )
        if result.returncode == 0:
            print(f"  {result.stdout.strip()}")
            return True
        else:
            print(f"  Error: {result.stderr}")
            return False
    except Exception as e:
        print(f"  Git sync failed: {e}")
        return False


# =============================================================================
# PHYSICS STATE COMPUTATION
# =============================================================================

# Physics state dimension: 30 core features (percentile normalized)
PHYSICS_STATE_DIM = 30

# Global measurement cache per episode
_measurement_cache = {}

def compute_physics_measurements_for_episode(df: pd.DataFrame) -> Dict[str, np.ndarray]:
    """
    Compute ALL physics measurements for the entire episode upfront.
    Returns dict of measurement_name -> array (length = len(df)).
    """
    engine = MeasurementEngine(percentile_window=100)

    open_ = df['Open'].values
    high = df['High'].values
    low = df['Low'].values
    close = df['Close'].values
    volume = df['Volume'].values

    # Spread approximation (high - low as proxy if not available)
    spread = (high - low) * 0.001  # Approximate spread

    # Compute all raw measurements
    raw = engine.compute_all(open_, high, low, close, volume, spread)

    # Normalize to percentiles (instrument-agnostic)
    normalized = engine.normalize_to_percentiles(raw)

    return normalized


def compute_state(df: pd.DataFrame, idx: int, lookback: int = 50) -> np.ndarray:
    """
    Compute physics-only state vector for a given bar.

    Uses ONLY:
    - Kinematics (velocity, acceleration, jerk, snap, crackle, pop)
    - Energy (kinetic, potential_compression, potential_displacement, efficiency)
    - Flow dynamics (reynolds, damping, viscosity, liquidity)
    - Thermodynamics (entropy, entropy_rate, phase_compression)
    - Field theory (gradient, divergence, buying_pressure, body_ratio)
    - Microstructure (volume_surge, volume_trend)

    NO traditional indicators (RSI, MACD, etc.)
    """
    global _measurement_cache

    # Get or compute measurements for this episode
    episode_id = id(df)
    if episode_id not in _measurement_cache:
        _measurement_cache[episode_id] = compute_physics_measurements_for_episode(df)

    m = _measurement_cache[episode_id]

    if idx < 50:
        return np.zeros(PHYSICS_STATE_DIM)

    # Extract percentile features at current index (all normalized 0-1)
    try:
        state = np.array([
            # === KINEMATICS (6 derivatives) ===
            m['velocity_pct'][idx],           # 0: velocity percentile
            m['acceleration_pct'][idx],       # 1: acceleration percentile
            m['jerk_pct'][idx],               # 2: jerk percentile (fat candle predictor)
            m['snap_pct'][idx],               # 3: snap percentile
            m['crackle_pct'][idx],            # 4: crackle percentile
            m['pop_pct'][idx],                # 5: pop percentile

            # === MOMENTUM & IMPULSE ===
            m['momentum_pct'][idx],           # 6: momentum percentile
            m['impulse_pct'][idx],            # 7: impulse percentile

            # === ENERGY ===
            m['kinetic_energy_pct'][idx],             # 8: kinetic energy
            m['potential_energy_compression_pct'][idx],  # 9: compression PE
            m['potential_energy_displacement_pct'][idx], # 10: displacement PE
            m['energy_efficiency_pct'][idx],          # 11: KE/PE ratio
            m['energy_release_rate_pct'][idx],        # 12: rate of energy change

            # === FLOW DYNAMICS ===
            m['reynolds_pct'][idx],           # 13: Reynolds number (laminar vs turbulent)
            m['damping_pct'][idx],            # 14: damping coefficient
            m['viscosity_pct'][idx],          # 15: market viscosity
            m['liquidity_pct'][idx],          # 16: liquidity measure
            m['reynolds_momentum_corr'][idx], # 17: Re-momentum relationship

            # === THERMODYNAMICS ===
            m['entropy_pct'][idx],            # 18: Shannon entropy
            m['entropy_rate_pct'][idx],       # 19: rate of entropy change
            m['phase_compression_pct'][idx],  # 20: phase space compression

            # === FIELD MEASURES ===
            m['price_gradient_pct'][idx],     # 21: price gradient
            m['gradient_magnitude_pct'][idx], # 22: |gradient|
            m['divergence_pct'][idx],         # 23: flow divergence
            m['buying_pressure'][idx],        # 24: buying pressure (already 0-1)
            m['body_ratio'][idx],             # 25: body ratio (already 0-1)

            # === MICROSTRUCTURE ===
            m['volume_surge_pct'][idx],       # 26: volume surge
            m['volume_trend_pct'][idx],       # 27: volume trend

            # === CROSS-INTERACTIONS ===
            m['jerk_energy_pct'][idx],        # 28: jerk energy
            m['release_potential_pct'][idx],  # 29: release potential
        ])

        # Replace any NaN/inf with 0.5 (neutral percentile)
        state = np.nan_to_num(state, nan=0.5, posinf=1.0, neginf=0.0)

        return state

    except (KeyError, IndexError):
        return np.full(PHYSICS_STATE_DIM, 0.5)


# =============================================================================
# TRAINING LOOP
# =============================================================================

class RLTrainer:
    """Automated RL training with checkpointing."""

    def __init__(
        self,
        checkpoint_dir: str = "./checkpoints",
        state_dim: int = PHYSICS_STATE_DIM,  # 30 physics features
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

        # Clear measurement cache to free memory
        global _measurement_cache
        episode_id = id(df)
        if episode_id in _measurement_cache:
            del _measurement_cache[episode_id]

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

    def train(self, n_episodes: int = 100, checkpoint_every: int = 10,
              data_loader: Optional['RealDataLoader'] = None):
        """Main training loop."""
        print(f"\nStarting training from episode {self.episode}")
        print(f"Using {'PyTorch' if HAS_TORCH else 'NumPy'} backend")
        print(f"Data source: {'Real data' if data_loader else 'Synthetic'}")
        print(f"Epsilon: {self.get_epsilon():.3f}")
        print("-" * 60)

        for ep in range(n_episodes):
            # Get episode data
            if data_loader:
                df, symbol = data_loader.get_random_episode(min_bars=500)
            else:
                df = generate_synthetic_episode(n_bars=500, seed=self.episode + ep)
                symbol = "SYNTHETIC"

            # Run episode
            metrics = self.run_episode(df, symbol=symbol)
            metrics['symbol'] = symbol

            self.episode_rewards.append(metrics['reward'])

            # Log
            avg_reward = np.mean(self.episode_rewards[-100:])
            print(f"Ep {metrics['episode']:4d} | "
                  f"{symbol:12} | "
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
    parser = argparse.ArgumentParser(
        description='Kinetra RL Training - Physics Only',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train with real data (recommended)
  python scripts/train_rl.py --episodes 100 --data-dir data/master

  # Train with specific timeframe
  python scripts/train_rl.py --episodes 200 --timeframe H4

  # Sync git and train
  python scripts/train_rl.py --sync --episodes 100

  # Train with synthetic data (testing)
  python scripts/train_rl.py --episodes 50 --synthetic
        """
    )
    parser.add_argument('--episodes', type=int, default=100, help='Number of episodes')
    parser.add_argument('--checkpoint-dir', default='./checkpoints', help='Checkpoint directory')
    parser.add_argument('--checkpoint-every', type=int, default=10, help='Checkpoint frequency')
    parser.add_argument('--data-dir', default='data/master', help='Directory with MT5 CSV data')
    parser.add_argument('--timeframe', default='H1', help='Timeframe to use (M15, M30, H1, H4)')
    parser.add_argument('--synthetic', action='store_true', help='Use synthetic data (for testing)')
    parser.add_argument('--sync', action='store_true', help='Git pull before training')
    parser.add_argument('--fresh', action='store_true', help='Start fresh (ignore checkpoints)')
    args = parser.parse_args()

    print("=" * 60)
    print("KINETRA RL TRAINER - PHYSICS ONLY")
    print("NO RSI/MACD/Bollinger - Pure kinematics, energy, flow, entropy")
    print(f"State dimension: {PHYSICS_STATE_DIM} physics features")
    print("=" * 60)

    # Git sync if requested
    if args.sync:
        git_sync()

    # Fresh start if requested
    if args.fresh:
        import shutil
        if os.path.exists(args.checkpoint_dir):
            backup = f"{args.checkpoint_dir}_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            shutil.move(args.checkpoint_dir, backup)
            print(f"\n[FRESH] Moved old checkpoints to {backup}")
        os.makedirs(args.checkpoint_dir, exist_ok=True)

    # Initialize data loader
    data_loader = None
    if not args.synthetic:
        data_loader = RealDataLoader(data_dir=args.data_dir, timeframe=args.timeframe)
        if data_loader.files:
            symbols = data_loader.get_all_symbols()
            print(f"\nData: {len(symbols)} symbols, {len(data_loader.files)} files")
            print(f"Symbols: {', '.join(symbols[:10])}{'...' if len(symbols) > 10 else ''}")
        else:
            print("\nNo real data found, falling back to synthetic")
            data_loader = None

    # Create trainer
    trainer = RLTrainer(checkpoint_dir=args.checkpoint_dir)

    # Train
    trainer.train(
        n_episodes=args.episodes,
        checkpoint_every=args.checkpoint_every,
        data_loader=data_loader
    )


if __name__ == "__main__":
    main()
