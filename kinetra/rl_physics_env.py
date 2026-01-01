"""
RL Environment for Physics-Based Trading

Let RL discover the optimal formula combining:
- Energy, Damping, Entropy (core physics)
- Acceleration, Jerk, Impulse (derivatives)
- Liquidity, Buying Pressure (order flow)
- Range Position, Flow Consistency (context)
- ROC, Inertia (momentum)

Actions: HOLD, ENTER_LONG, ENTER_SHORT, EXIT
Reward: ARS-style (PnL + α*MFE - β*MAE - γ*Time)
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, Dict, List
from dataclasses import dataclass
from enum import IntEnum


class Action(IntEnum):
    HOLD = 0
    ENTER_LONG = 1
    ENTER_SHORT = 2
    EXIT = 3


@dataclass
class Position:
    direction: int  # 1 for long, -1 for short, 0 for flat
    entry_price: float
    entry_bar: int
    mfe: float = 0.0  # Maximum favorable excursion
    mae: float = 0.0  # Maximum adverse excursion


class PhysicsFeatureComputer:
    """Compute all physics features for RL state."""

    def __init__(self, lookback: int = 20):
        self.lookback = lookback

    def compute_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute all physics features as percentile ranks [0, 1].

        All features are adaptive (percentile-based, no fixed thresholds).
        """
        result = df.copy()
        window = min(500, len(df))

        # === CORE PHYSICS ===
        # Velocity and Energy
        velocity = df['close'].pct_change()
        energy = 0.5 * (velocity ** 2)
        result['energy'] = energy.rolling(self.lookback).mean()

        # Damping (volatility / mean abs return)
        vol = velocity.rolling(self.lookback).std()
        mean_abs = velocity.abs().rolling(self.lookback).mean()
        result['damping'] = vol / (mean_abs + 1e-10)

        # Entropy (simplified - use return dispersion)
        result['entropy'] = velocity.rolling(self.lookback).apply(
            lambda x: -np.sum(np.histogram(x, bins=10, density=True)[0] *
                             np.log(np.histogram(x, bins=10, density=True)[0] + 1e-10)) / 10,
            raw=True
        )

        # === DERIVATIVES ===
        # Acceleration (d²P/dt²)
        result['acceleration'] = velocity.diff()

        # Jerk (d³P/dt³) - best fat candle predictor
        result['jerk'] = result['acceleration'].diff()

        # Impulse (momentum change)
        momentum = df['close'].pct_change(self.lookback)
        result['impulse'] = momentum.diff(5)

        # === ORDER FLOW ===
        # Liquidity
        bar_range = df['high'] - df['low']
        result['liquidity'] = df['volume'] / (bar_range.clip(lower=1e-10) * df['close'] / 100)

        # Buying pressure
        bp = (df['close'] - df['low']) / bar_range.clip(lower=1e-10)
        result['buying_pressure'] = bp.rolling(5).mean()

        # === CONTEXT ===
        # Range position
        rolling_high = df['high'].rolling(self.lookback).max()
        rolling_low = df['low'].rolling(self.lookback).min()
        result['range_position'] = (df['close'] - rolling_low) / (rolling_high - rolling_low + 1e-10)

        # Flow consistency (laminar flow)
        return_sign = np.sign(velocity)
        result['flow_consistency'] = return_sign.rolling(5).apply(
            lambda x: (x == x.iloc[-1]).mean() if len(x) > 0 else 0.5, raw=False
        )

        # === MOMENTUM ===
        # ROC (normalized)
        result['roc'] = df['close'].pct_change(self.lookback)

        # Inertia (bars in same direction)
        direction = np.sign(velocity)
        counts = []
        count = 1
        for i in range(len(direction)):
            if i == 0:
                counts.append(1)
            elif direction.iloc[i] == direction.iloc[i-1] and direction.iloc[i] != 0:
                count += 1
                counts.append(count)
            else:
                count = 1
                counts.append(count)
        result['inertia'] = pd.Series(counts, index=df.index)

        # Volume percentile
        result['volume_pct'] = df['volume'].rolling(window).apply(
            lambda x: (x.iloc[-1] > x.iloc[:-1]).mean() if len(x) > 1 else 0.5, raw=False
        )

        # === REYNOLDS NUMBER (Turbulent vs Laminar) ===
        bar_range_pct = (df['high'] - df['low']) / df['close']
        volatility_re = velocity.rolling(self.lookback).std().clip(lower=1e-10)
        volume_norm = df['volume'] / df['volume'].rolling(self.lookback).mean().clip(lower=1e-10)
        reynolds = (velocity.abs() * bar_range_pct * volume_norm) / volatility_re
        result['reynolds'] = reynolds.rolling(self.lookback).mean()

        # === VISCOSITY (resistance to flow) ===
        avg_volume = df['volume'].rolling(self.lookback).mean().clip(lower=1e-10)
        volume_norm_v = df['volume'] / avg_volume
        viscosity = bar_range_pct / volume_norm_v.clip(lower=1e-10)
        result['viscosity'] = viscosity.rolling(self.lookback).mean()

        # === ANGULAR MOMENTUM (rotational/cyclical) ===
        price_mean = df['close'].rolling(self.lookback).mean()
        price_deviation = df['close'] - price_mean
        angular_position = price_deviation / df['close'].rolling(self.lookback).std().clip(lower=1e-10)
        angular_velocity = angular_position.diff()
        result['angular_momentum'] = angular_position * angular_velocity

        # === POTENTIAL ENERGY (stored in compression) ===
        bar_range = df['high'] - df['low']
        avg_range = bar_range.rolling(self.lookback).mean()
        range_compression = 1 - (bar_range / avg_range.clip(lower=1e-10))
        result['potential_energy'] = range_compression.clip(lower=0) * volatility_re

        # === MOMENTUM DIRECTION (for RL to learn continuation vs reversal) ===
        result['momentum_direction'] = np.sign(df['close'].pct_change(5))

        # === CONVERT TO PERCENTILES (Adaptive, no fixed thresholds) ===
        feature_cols = ['energy', 'damping', 'entropy', 'acceleration', 'jerk',
                       'impulse', 'liquidity', 'buying_pressure', 'range_position',
                       'flow_consistency', 'roc', 'inertia', 'volume_pct', 'reynolds',
                       'viscosity', 'angular_momentum', 'potential_energy']

        for col in feature_cols:
            result[f'{col}_pct'] = result[col].rolling(window, min_periods=self.lookback).apply(
                lambda x: (x.iloc[-1] > x.iloc[:-1]).mean() if len(x) > 1 else 0.5, raw=False
            ).fillna(0.5)

        return result.fillna(0.5)


class PhysicsTradingEnv:
    """
    RL Environment for physics-based trading.

    State: All physics features as percentiles [0, 1]
    Actions: HOLD, ENTER_LONG, ENTER_SHORT, EXIT
    Reward: ARS-style asymmetric reward
    """

    def __init__(
        self,
        df: pd.DataFrame,
        alpha: float = 0.5,   # MFE reward weight
        beta: float = 1.0,    # MAE penalty weight
        gamma: float = 0.01,  # Time decay per bar
        max_bars_held: int = 10,
    ):
        self.feature_computer = PhysicsFeatureComputer()
        self.features_df = self.feature_computer.compute_features(df)
        self.df = df
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.max_bars_held = max_bars_held

        # Feature columns for state
        self.feature_cols = [c for c in self.features_df.columns if c.endswith('_pct')]

        # Environment state
        self.current_idx = 0
        self.position: Optional[Position] = None
        self.done = False

        # Episode tracking
        self.trades: List[Dict] = []

    @property
    def state_dim(self) -> int:
        """Dimension of state vector."""
        # Features + position info (direction, bars_held, unrealized_pnl)
        return len(self.feature_cols) + 3

    @property
    def action_dim(self) -> int:
        """Number of actions."""
        return 4

    def reset(self, start_idx: Optional[int] = None) -> np.ndarray:
        """Reset environment to starting state."""
        # Skip warmup period
        warmup = 100
        if start_idx is None:
            self.current_idx = warmup
        else:
            self.current_idx = max(warmup, start_idx)

        self.position = None
        self.done = False
        self.trades = []

        return self._get_state()

    def _get_state(self) -> np.ndarray:
        """Get current state vector."""
        if self.current_idx >= len(self.features_df):
            return np.zeros(self.state_dim)

        # Physics features (all as percentiles)
        features = self.features_df.iloc[self.current_idx][self.feature_cols].values.astype(float)

        # Position info
        if self.position is None:
            pos_dir = 0.0
            bars_held = 0.0
            unrealized_pnl = 0.0
        else:
            pos_dir = float(self.position.direction)
            bars_held = float(self.current_idx - self.position.entry_bar) / self.max_bars_held
            current_price = self.df.iloc[self.current_idx]['close']
            unrealized_pnl = (current_price - self.position.entry_price) / self.position.entry_price
            unrealized_pnl *= self.position.direction

        position_info = np.array([pos_dir, bars_held, unrealized_pnl])

        return np.concatenate([features, position_info])

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Take action and return (next_state, reward, done, info).
        """
        reward = 0.0
        info = {}

        current_price = self.df.iloc[self.current_idx]['close']
        current_high = self.df.iloc[self.current_idx]['high']
        current_low = self.df.iloc[self.current_idx]['low']

        # Handle action
        if action == Action.HOLD:
            # Update MFE/MAE if in position
            if self.position is not None:
                self._update_excursions(current_high, current_low)
                # Small time penalty for holding
                reward = -self.gamma

        elif action == Action.ENTER_LONG:
            if self.position is None:
                self.position = Position(
                    direction=1,
                    entry_price=current_price,
                    entry_bar=self.current_idx
                )
                info['action'] = 'enter_long'

        elif action == Action.ENTER_SHORT:
            if self.position is None:
                self.position = Position(
                    direction=-1,
                    entry_price=current_price,
                    entry_bar=self.current_idx
                )
                info['action'] = 'enter_short'

        elif action == Action.EXIT:
            if self.position is not None:
                reward = self._close_position(current_price)
                info['action'] = 'exit'

        # Move to next bar
        self.current_idx += 1

        # Check if done
        if self.current_idx >= len(self.df) - 1:
            self.done = True
            # Force close any open position
            if self.position is not None:
                final_price = self.df.iloc[self.current_idx]['close']
                reward += self._close_position(final_price)

        # Auto-exit if held too long
        if self.position is not None:
            bars_held = self.current_idx - self.position.entry_bar
            if bars_held >= self.max_bars_held:
                reward += self._close_position(current_price)
                info['action'] = 'auto_exit'

        next_state = self._get_state()

        return next_state, reward, self.done, info

    def _update_excursions(self, high: float, low: float):
        """Update MFE/MAE for current position."""
        if self.position is None:
            return

        entry = self.position.entry_price

        if self.position.direction == 1:  # Long
            favorable = (high - entry) / entry * 100
            adverse = (entry - low) / entry * 100
        else:  # Short
            favorable = (entry - low) / entry * 100
            adverse = (high - entry) / entry * 100

        self.position.mfe = max(self.position.mfe, favorable)
        self.position.mae = max(self.position.mae, adverse)

    def _close_position(self, exit_price: float) -> float:
        """
        Close position and compute ARS reward.

        R = (PnL / energy) + α*MFE - β*MAE - γ*bars_held
        """
        if self.position is None:
            return 0.0

        entry = self.position.entry_price
        direction = self.position.direction
        bars_held = self.current_idx - self.position.entry_bar

        # P&L
        pnl = (exit_price - entry) / entry * 100 * direction

        # Get entry bar energy for normalization
        entry_energy = self.features_df.iloc[self.position.entry_bar]['energy']
        energy_norm = max(entry_energy, 1e-6)

        # ARS reward: asymmetric - punish MAE more than reward MFE
        reward = (
            pnl / (energy_norm * 1000 + 1) +  # Energy-normalized P&L
            self.alpha * self.position.mfe -   # Reward capturing MFE
            self.beta * self.position.mae -    # Punish MAE
            self.gamma * bars_held             # Time decay
        )

        # Record trade
        self.trades.append({
            'entry_bar': self.position.entry_bar,
            'exit_bar': self.current_idx,
            'direction': direction,
            'pnl': pnl,
            'mfe': self.position.mfe,
            'mae': self.position.mae,
            'bars_held': bars_held,
            'reward': reward,
        })

        self.position = None

        return reward

    def get_episode_stats(self) -> Dict:
        """Get statistics for current episode."""
        if not self.trades:
            return {'n_trades': 0}

        pnls = [t['pnl'] for t in self.trades]
        rewards = [t['reward'] for t in self.trades]
        mfes = [t['mfe'] for t in self.trades]
        maes = [t['mae'] for t in self.trades]

        wins = sum(1 for p in pnls if p > 0)
        total = len(pnls)

        return {
            'n_trades': total,
            'win_rate': wins / total if total > 0 else 0,
            'total_pnl': sum(pnls),
            'avg_pnl': np.mean(pnls),
            'total_reward': sum(rewards),
            'avg_reward': np.mean(rewards),
            'avg_mfe': np.mean(mfes),
            'avg_mae': np.mean(maes),
            'profit_factor': sum(p for p in pnls if p > 0) / abs(sum(p for p in pnls if p < 0) or 1),
        }


class SimpleRLAgent:
    """
    Simple Q-learning agent for physics trading.

    Uses linear function approximation with physics features.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        learning_rate: float = 0.01,
        gamma: float = 0.99,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.995,
        epsilon_min: float = 0.01,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lr = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        # Linear Q-function weights: Q(s,a) = s · W[:, a]
        self.weights = np.random.randn(state_dim, action_dim) * 0.01

    def get_q_values(self, state: np.ndarray) -> np.ndarray:
        """Get Q-values for all actions."""
        return state @ self.weights

    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """Select action using epsilon-greedy policy."""
        if training and np.random.random() < self.epsilon:
            return np.random.randint(self.action_dim)
        return int(np.argmax(self.get_q_values(state)))

    def update(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ):
        """Update Q-function with TD learning."""
        # Current Q-value
        q_current = self.get_q_values(state)[action]

        # Target Q-value
        if done:
            q_target = reward
        else:
            q_target = reward + self.gamma * np.max(self.get_q_values(next_state))

        # TD error
        td_error = q_target - q_current

        # Update weights for this action
        self.weights[:, action] += self.lr * td_error * state

        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def get_feature_importance(self, feature_names: List[str]) -> pd.DataFrame:
        """Get feature importance from learned weights."""
        # Aggregate weights across actions (absolute mean)
        importance = np.abs(self.weights).mean(axis=1)

        # Normalize
        importance = importance / importance.sum()

        return pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)


def train_rl_agent(
    df: pd.DataFrame,
    n_episodes: int = 100,
    verbose: bool = True
) -> Tuple[SimpleRLAgent, PhysicsTradingEnv, List[Dict]]:
    """
    Train RL agent on physics features.

    Returns trained agent, environment, and episode stats.
    """
    env = PhysicsTradingEnv(df)
    agent = SimpleRLAgent(env.state_dim, env.action_dim)

    episode_stats = []

    for episode in range(n_episodes):
        state = env.reset()
        total_reward = 0
        steps = 0

        while not env.done:
            action = agent.select_action(state)
            next_state, reward, done, info = env.step(action)

            agent.update(state, action, reward, next_state, done)

            state = next_state
            total_reward += reward
            steps += 1

        stats = env.get_episode_stats()
        stats['episode'] = episode
        stats['total_reward'] = total_reward
        stats['steps'] = steps
        stats['epsilon'] = agent.epsilon
        episode_stats.append(stats)

        if verbose and (episode + 1) % 10 == 0:
            print(f"Episode {episode + 1}/{n_episodes}: "
                  f"Trades={stats['n_trades']}, "
                  f"WinRate={stats['win_rate']:.1%}, "
                  f"PnL={stats['total_pnl']:.2f}%, "
                  f"Reward={total_reward:.2f}, "
                  f"ε={agent.epsilon:.3f}")

    return agent, env, episode_stats


def evaluate_agent(
    agent: SimpleRLAgent,
    df: pd.DataFrame,
    start_idx: Optional[int] = None
) -> tuple[dict, list[dict]]:
    """Evaluate trained agent on data."""
    env = PhysicsTradingEnv(df)
    state = env.reset(start_idx)
    total_reward = 0

    while not env.done:
        action = agent.select_action(state, training=False)
        state, reward, done, info = env.step(action)
        total_reward += reward

    stats = env.get_episode_stats()
    stats['total_reward'] = total_reward

    return stats, env.trades
