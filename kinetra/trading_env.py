"""
Trading Environment for RL

Gym-compatible environment where the agent learns to trigger entries
based on physics state. No static thresholds - learns through exploration.

State: Physics features (energy, damping, entropy, regime) + rolling stats
Actions: 0=hold, 1=long, 2=short, 3=close
Reward: Physics-aligned with MFE/MAE
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass
from enum import IntEnum

from .physics_engine import PhysicsEngine
from .reward_shaping import AdaptiveRewardShaper


class Action(IntEnum):
    HOLD = 0
    LONG = 1
    SHORT = 2
    CLOSE = 3


@dataclass
class Position:
    """Current position state."""
    direction: int  # 0=flat, 1=long, -1=short
    entry_price: float = 0.0
    entry_bar: int = 0
    entry_energy: float = 0.0
    mfe: float = 0.0  # Max favorable excursion
    mae: float = 0.0  # Max adverse excursion


class TradingEnv:
    """
    Physics-based trading environment for RL.

    The agent learns WHEN to enter based on physics state.
    No static thresholds - discovery through exploration.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        lookback: int = 20,
        initial_capital: float = 10000.0,
        max_position: float = 1.0,
        spread_pips: float = 10.0,
        commission_pct: float = 0.001,
        state_history: int = 5,
    ):
        """
        Initialize trading environment.

        Args:
            data: OHLCV DataFrame
            lookback: Physics engine lookback
            initial_capital: Starting capital
            max_position: Max position size
            spread_pips: Spread cost in pips
            commission_pct: Commission as % of trade
            state_history: How many bars of history in state
        """
        self.data = data.reset_index(drop=True)
        self.lookback = lookback
        self.initial_capital = initial_capital
        self.max_position = max_position
        self.spread_pips = spread_pips
        self.commission_pct = commission_pct
        self.state_history = state_history

        # Compute physics features
        self._compute_features()

        # Initialize reward shaper
        self.reward_shaper = AdaptiveRewardShaper()

        # State/action dimensions
        self.state_dim = self._get_state_dim()
        self.action_dim = 4  # hold, long, short, close

        # Episode state
        self.reset()

    def _compute_features(self):
        """Compute all physics features upfront."""
        engine = PhysicsEngine(lookback=self.lookback)
        physics = engine.compute_physics_state(self.data['close'])

        self.features = pd.DataFrame({
            'open': self.data['open'],
            'high': self.data['high'],
            'low': self.data['low'],
            'close': self.data['close'],
            'energy': physics['energy'],
            'damping': physics['damping'],
            'entropy': physics['entropy'],
            'regime': physics['regime'],
        })

        # Encode regime as numeric
        regime_map = {'underdamped': 0, 'critical': 1, 'overdamped': 2}
        self.features['regime_code'] = self.features['regime'].map(regime_map).fillna(1)

        # Momentum features
        self.features['returns'] = self.data['close'].pct_change()
        self.features['momentum_5'] = self.data['close'].pct_change(5)
        self.features['momentum_20'] = self.data['close'].pct_change(20)

        # Energy dynamics
        self.features['energy_change'] = self.features['energy'].pct_change()
        self.features['energy_ma5'] = self.features['energy'].rolling(5).mean()
        self.features['energy_ma20'] = self.features['energy'].rolling(20).mean()
        self.features['energy_std5'] = self.features['energy'].rolling(5).std()

        # Volatility (for reward normalization)
        self.features['atr'] = self._compute_atr(14)

        # Fill NaN
        self.features = self.features.fillna(0)

        # Normalize features for neural network
        self._normalize_features()

    def _compute_atr(self, period: int = 14) -> pd.Series:
        """Compute Average True Range."""
        high = self.data['high']
        low = self.data['low']
        close = self.data['close']

        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())

        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(period).mean()

    def _normalize_features(self):
        """Normalize features to [0, 1] or [-1, 1] range."""
        # Features to normalize
        to_normalize = [
            'energy', 'damping', 'entropy', 'returns',
            'momentum_5', 'momentum_20', 'energy_change',
            'energy_ma5', 'energy_ma20', 'energy_std5', 'atr'
        ]

        self.feature_stats = {}
        for col in to_normalize:
            if col in self.features.columns:
                mean = self.features[col].mean()
                std = self.features[col].std() + 1e-8
                self.feature_stats[col] = {'mean': mean, 'std': std}
                self.features[f'{col}_norm'] = (self.features[col] - mean) / std

    def _get_state_dim(self) -> int:
        """Calculate state dimension."""
        # Core features (normalized)
        core_features = 11  # energy, damping, entropy, regime, returns, momentum*2, energy_change, energy_ma*2, energy_std
        # Position state
        position_features = 4  # direction, unrealized_pnl_norm, bars_in_trade, mfe_mae_ratio
        # History (optional)
        history_features = self.state_history * 3  # energy, damping, entropy history

        return core_features + position_features + history_features

    def reset(self) -> np.ndarray:
        """Reset environment to start of episode."""
        self.current_bar = self.lookback + self.state_history
        self.capital = self.initial_capital
        self.position = Position(direction=0)
        self.trades = []
        self.equity_curve = [self.initial_capital]

        return self._get_state()

    def _get_state(self) -> np.ndarray:
        """Get current state vector."""
        bar = self.current_bar
        f = self.features

        # Core normalized features
        state = [
            f.loc[bar, 'energy_norm'] if 'energy_norm' in f.columns else 0,
            f.loc[bar, 'damping_norm'] if 'damping_norm' in f.columns else 0,
            f.loc[bar, 'entropy_norm'] if 'entropy_norm' in f.columns else 0,
            f.loc[bar, 'regime_code'] / 2.0,  # 0, 0.5, 1
            f.loc[bar, 'returns_norm'] if 'returns_norm' in f.columns else 0,
            f.loc[bar, 'momentum_5_norm'] if 'momentum_5_norm' in f.columns else 0,
            f.loc[bar, 'momentum_20_norm'] if 'momentum_20_norm' in f.columns else 0,
            f.loc[bar, 'energy_change_norm'] if 'energy_change_norm' in f.columns else 0,
            f.loc[bar, 'energy_ma5_norm'] if 'energy_ma5_norm' in f.columns else 0,
            f.loc[bar, 'energy_ma20_norm'] if 'energy_ma20_norm' in f.columns else 0,
            f.loc[bar, 'energy_std5_norm'] if 'energy_std5_norm' in f.columns else 0,
        ]

        # Position state
        if self.position.direction != 0:
            current_price = f.loc[bar, 'close']
            unrealized = (current_price - self.position.entry_price) * self.position.direction
            unrealized_pct = unrealized / self.position.entry_price
            bars_in_trade = bar - self.position.entry_bar
            mfe_mae_ratio = self.position.mfe / (self.position.mae + 1e-8) if self.position.mae > 0 else 1.0

            state.extend([
                self.position.direction,  # -1, 0, 1
                np.clip(unrealized_pct * 10, -1, 1),  # Normalized unrealized P&L
                np.clip(bars_in_trade / 100, 0, 1),  # Normalized time in trade
                np.clip(mfe_mae_ratio, 0, 2) / 2,  # MFE/MAE ratio
            ])
        else:
            state.extend([0, 0, 0, 0.5])

        # History
        for i in range(self.state_history):
            hist_bar = bar - i - 1
            if hist_bar >= 0:
                state.extend([
                    f.loc[hist_bar, 'energy_norm'] if 'energy_norm' in f.columns else 0,
                    f.loc[hist_bar, 'damping_norm'] if 'damping_norm' in f.columns else 0,
                    f.loc[hist_bar, 'entropy_norm'] if 'entropy_norm' in f.columns else 0,
                ])
            else:
                state.extend([0, 0, 0])

        return np.array(state, dtype=np.float32)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Execute action and return next state.

        Args:
            action: 0=hold, 1=long, 2=short, 3=close

        Returns:
            next_state, reward, done, info
        """
        bar = self.current_bar
        f = self.features
        current_price = f.loc[bar, 'close']
        reward = 0.0
        info = {'action': action, 'bar': bar}

        # Execute action
        if action == Action.HOLD:
            # Just update position if holding
            if self.position.direction != 0:
                self._update_position_stats(current_price)

        elif action == Action.LONG:
            if self.position.direction == 0:
                # Open long
                self._open_position(1, current_price, bar)
                info['opened'] = 'long'
            elif self.position.direction == -1:
                # Close short and open long
                reward = self._close_position(current_price, bar)
                self._open_position(1, current_price, bar)
                info['closed'] = 'short'
                info['opened'] = 'long'

        elif action == Action.SHORT:
            if self.position.direction == 0:
                # Open short
                self._open_position(-1, current_price, bar)
                info['opened'] = 'short'
            elif self.position.direction == 1:
                # Close long and open short
                reward = self._close_position(current_price, bar)
                self._open_position(-1, current_price, bar)
                info['closed'] = 'long'
                info['opened'] = 'short'

        elif action == Action.CLOSE:
            if self.position.direction != 0:
                reward = self._close_position(current_price, bar)
                info['closed'] = 'position'

        # Update equity
        self._update_equity(current_price)

        # Move to next bar
        self.current_bar += 1
        done = self.current_bar >= len(self.features) - 1

        # Final reward adjustment if done with open position
        if done and self.position.direction != 0:
            final_price = self.features.loc[self.current_bar, 'close']
            reward += self._close_position(final_price, self.current_bar)

        next_state = self._get_state() if not done else np.zeros(self.state_dim)

        return next_state, reward, done, info

    def _open_position(self, direction: int, price: float, bar: int):
        """Open new position."""
        # Apply spread cost
        if direction == 1:  # Long - buy at ask
            entry_price = price * (1 + self.spread_pips / 10000)
        else:  # Short - sell at bid
            entry_price = price * (1 - self.spread_pips / 10000)

        self.position = Position(
            direction=direction,
            entry_price=entry_price,
            entry_bar=bar,
            entry_energy=self.features.loc[bar, 'energy'],
            mfe=0.0,
            mae=0.0,
        )

    def _close_position(self, price: float, bar: int) -> float:
        """Close position and return reward."""
        if self.position.direction == 0:
            return 0.0

        # Calculate P&L
        if self.position.direction == 1:  # Long
            exit_price = price * (1 - self.spread_pips / 10000)  # Sell at bid
        else:  # Short
            exit_price = price * (1 + self.spread_pips / 10000)  # Buy at ask

        pnl = (exit_price - self.position.entry_price) * self.position.direction
        pnl_pct = pnl / self.position.entry_price

        # Commission
        commission = abs(pnl) * self.commission_pct
        net_pnl = pnl - commission

        # Calculate reward using physics-aligned shaper
        atr = self.features.loc[bar, 'atr']
        time_in_trade = bar - self.position.entry_bar

        reward = self.reward_shaper.calculate_reward(
            pnl=net_pnl,
            energy=self.position.entry_energy,
            mfe=self.position.mfe,
            mae=self.position.mae,
            atr=atr,
            time_in_trade=time_in_trade,
        )

        # Record trade
        self.trades.append({
            'entry_bar': self.position.entry_bar,
            'exit_bar': bar,
            'direction': self.position.direction,
            'entry_price': self.position.entry_price,
            'exit_price': exit_price,
            'pnl': net_pnl,
            'pnl_pct': pnl_pct,
            'mfe': self.position.mfe,
            'mae': self.position.mae,
            'energy_at_entry': self.position.entry_energy,
        })

        # Update capital
        self.capital += net_pnl * self.initial_capital

        # Reset position
        self.position = Position(direction=0)

        return reward

    def _update_position_stats(self, current_price: float):
        """Update MFE/MAE for current position."""
        if self.position.direction == 0:
            return

        excursion = (current_price - self.position.entry_price) * self.position.direction

        if excursion > self.position.mfe:
            self.position.mfe = excursion
        if excursion < -self.position.mae:
            self.position.mae = -excursion

    def _update_equity(self, current_price: float):
        """Update equity curve."""
        unrealized = 0.0
        if self.position.direction != 0:
            unrealized = (current_price - self.position.entry_price) * self.position.direction
            unrealized *= self.initial_capital / self.position.entry_price

        self.equity_curve.append(self.capital + unrealized)

    def get_episode_stats(self) -> Dict:
        """Get statistics for completed episode."""
        trades_df = pd.DataFrame(self.trades) if self.trades else pd.DataFrame()

        stats = {
            'total_trades': len(self.trades),
            'final_capital': self.capital,
            'total_return': (self.capital - self.initial_capital) / self.initial_capital,
            'max_drawdown': self._compute_max_drawdown(),
        }

        if len(trades_df) > 0:
            stats['win_rate'] = (trades_df['pnl'] > 0).mean()
            stats['avg_pnl'] = trades_df['pnl'].mean()
            stats['avg_mfe'] = trades_df['mfe'].mean()
            stats['avg_mae'] = trades_df['mae'].mean()
            stats['avg_energy_at_entry'] = trades_df['energy_at_entry'].mean()

        return stats

    def _compute_max_drawdown(self) -> float:
        """Compute maximum drawdown from equity curve."""
        equity = np.array(self.equity_curve)
        peak = np.maximum.accumulate(equity)
        drawdown = (peak - equity) / peak
        return float(np.max(drawdown)) if len(drawdown) > 0 else 0.0
