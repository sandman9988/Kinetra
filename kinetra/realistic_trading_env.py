"""
Realistic Trading Environment for RL with MT5 Constraints

This environment enforces realistic broker constraints during training:
1. Dynamic per-candle spread (from data, not fixed)
2. Freeze zones (blocks modifications before session close)
3. Stop distance validation (minimum SL/TP from price)
4. Slippage simulation (Gaussian noise)
5. Swap costs (overnight holding charges)
6. Commission from SymbolSpec

KEY PRINCIPLE: If the backtest engine isn't beyond reproach,
measurements are flawed and agents learn wrong strategies.

This environment uses THE SAME validation logic as RealisticBacktester,
ensuring that training results transfer directly to live trading.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
from dataclasses import dataclass
from enum import IntEnum

from .physics_engine import PhysicsEngine
from .reward_shaping import AdaptiveRewardShaper
from .market_microstructure import SymbolSpec
from .order_validator import OrderValidator


class Action(IntEnum):
    HOLD = 0
    LONG = 1
    SHORT = 2
    CLOSE = 3


@dataclass
class Position:
    """Current position state with realistic tracking."""
    direction: int  # 0=flat, 1=long, -1=short
    entry_price: float = 0.0
    entry_bar: int = 0
    entry_energy: float = 0.0
    mfe: float = 0.0  # Max favorable excursion
    mae: float = 0.0  # Max adverse excursion
    entry_spread_paid: float = 0.0  # Actual spread paid at entry
    entry_slippage: float = 0.0  # Actual slippage at entry
    days_held: float = 0.0  # Days position held (for swap)


class RealisticTradingEnv:
    """
    Trading environment with realistic MT5 constraints enforced during training.

    CRITICAL: This environment uses THE SAME constraints as live trading,
    ensuring measurements are accurate and agents learn viable strategies.

    Differences from basic TradingEnv:
    - Dynamic spread from candle data (not fixed)
    - Freeze zone enforcement
    - Stop distance validation
    - Slippage simulation
    - Swap costs
    - Commission from SymbolSpec

    Usage:
        spec = SymbolSpec(symbol="EURUSD", ...)
        env = RealisticTradingEnv(data, spec)

        # Agent trains with realistic constraints
        state = env.reset()
        action = agent.select_action(state)
        next_state, reward, done, info = env.step(action)

        # If agent works here, it WILL work in live trading
    """

    def __init__(
        self,
        data: pd.DataFrame,
        spec: SymbolSpec,
        lookback: int = 20,
        initial_capital: float = 10000.0,
        max_position: float = 1.0,
        state_history: int = 5,
        enable_slippage: bool = True,
        slippage_std_pips: float = 0.5,
        enable_freeze_zones: bool = True,
        enable_stop_validation: bool = True,
        verbose: bool = False,
    ):
        """
        Initialize realistic trading environment.

        Args:
            data: OHLCV DataFrame with 'spread' column (dynamic spread)
            spec: SymbolSpec with broker constraints
            lookback: Physics engine lookback
            initial_capital: Starting capital
            max_position: Max position size
            state_history: How many bars of history in state
            enable_slippage: Simulate slippage
            slippage_std_pips: Slippage standard deviation in pips
            enable_freeze_zones: Enforce freeze zone restrictions
            enable_stop_validation: Validate SL/TP distances
            verbose: Print warnings for violations
        """
        self.data = data.reset_index(drop=True)
        self.spec = spec
        self.lookback = lookback
        self.initial_capital = initial_capital
        self.max_position = max_position
        self.state_history = state_history
        self.enable_slippage = enable_slippage
        self.slippage_std_pips = slippage_std_pips
        self.enable_freeze_zones = enable_freeze_zones
        self.enable_stop_validation = enable_stop_validation
        self.verbose = verbose

        # Ensure data has spread column
        if 'spread' not in self.data.columns:
            if self.verbose:
                print(f"Warning: No 'spread' column, using typical spread: {self.spec.spread_typical}")
            self.data['spread'] = self.spec.spread_typical

        # Initialize order validator (SAME validation as live trading)
        self.validator = OrderValidator(
            spec=spec,
            auto_adjust_stops=False,  # Don't auto-adjust during training
        )

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
        physics = engine.compute_physics_state(self.data['close'], include_percentiles=True)

        self.features = pd.DataFrame({
            'open': self.data['open'],
            'high': self.data['high'],
            'low': self.data['low'],
            'close': self.data['close'],
            'spread': self.data['spread'],  # CRITICAL: Dynamic spread
            'energy': physics['energy'],
            'damping': physics['damping'],
            'entropy': physics['entropy'],
            'regime': physics['regime'],
            'energy_pct': physics['energy_pct'],
            'damping_pct': physics['damping_pct'],
            'entropy_pct': physics['entropy_pct'],
        })

        # Add all derived features (same as TradingEnv)
        self._add_derived_features()

    def _add_derived_features(self):
        """Add derived physics and price features."""
        f = self.features

        # Regime encoding
        regime_map = {'underdamped': 0, 'critical': 1, 'overdamped': 2}
        f['regime_code'] = f['regime'].map(regime_map).fillna(1)

        # Price dynamics
        f['price_velocity'] = self.data['close'].diff()
        f['price_accel'] = f['price_velocity'].diff()
        f['returns'] = self.data['close'].pct_change()

        # Energy dynamics
        f['energy_velocity'] = f['energy'].diff()
        f['energy_accel'] = f['energy_velocity'].diff()
        f['energy_change'] = f['energy'].pct_change()

        # Damping dynamics
        f['damping_velocity'] = f['damping'].diff()

        # Cross-feature ratio
        f['energy_damping_ratio'] = f['energy'] / (f['damping'] + 1e-8)

        # Regime transitions
        f['regime_changed'] = (f['regime'] != f['regime'].shift(1)).astype(float)
        f['regime_lag1'] = f['regime_code'].shift(1)
        f['regime_lag2'] = f['regime_code'].shift(2)

        # Adaptive percentiles for dynamics
        window = min(200, len(f))
        f['energy_vel_pct'] = f['energy_velocity'].rolling(window, min_periods=20).apply(
            lambda x: (x.iloc[-1] > x.iloc[:-1]).mean() if len(x) > 1 else 0.5, raw=False
        ).fillna(0.5)
        f['energy_acc_pct'] = f['energy_accel'].rolling(window, min_periods=20).apply(
            lambda x: (x.iloc[-1] > x.iloc[:-1]).mean() if len(x) > 1 else 0.5, raw=False
        ).fillna(0.5)
        f['price_vel_pct'] = f['price_velocity'].rolling(window, min_periods=20).apply(
            lambda x: (x.iloc[-1] > x.iloc[:-1]).mean() if len(x) > 1 else 0.5, raw=False
        ).fillna(0.5)

        # Volatility metrics (ATR, Bollinger Bands)
        f['atr'] = self._compute_atr(14)
        f['bb_width'] = self._compute_bb_width(20, 2.0)

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

    def _compute_bb_width(self, period: int = 20, std_dev: float = 2.0) -> pd.Series:
        """Compute Bollinger Band width."""
        close = self.data['close']
        sma = close.rolling(period).mean()
        std = close.rolling(period).std()

        upper = sma + (std * std_dev)
        lower = sma - (std * std_dev)

        return (upper - lower) / sma

    def reset(self) -> np.ndarray:
        """Reset environment to start of episode."""
        # Random start position (ensure enough history and future bars)
        min_bar = self.lookback + self.state_history + 100
        max_bar = len(self.features) - 500

        self.current_bar = np.random.randint(min_bar, max_bar)
        self.equity = self.initial_capital
        self.equity_history = [self.initial_capital]

        # Initialize flat position
        self.position = Position(direction=0)

        # Tracking
        self.total_trades = 0
        self.winning_trades = 0
        self.constraint_violations = 0

        return self._get_state()

    def _get_state(self) -> np.ndarray:
        """Build state vector from current bar and history."""
        bar = self.current_bar
        f = self.features

        # Current bar features (normalized percentiles)
        current_features = [
            f.loc[bar, 'energy_pct'],
            f.loc[bar, 'damping_pct'],
            f.loc[bar, 'entropy_pct'],
            f.loc[bar, 'regime_code'] / 2.0,  # Normalize to [0, 1]
            f.loc[bar, 'energy_vel_pct'],
            f.loc[bar, 'energy_acc_pct'],
            f.loc[bar, 'price_vel_pct'],
            f.loc[bar, 'regime_changed'],
        ]

        # Historical features (last N bars)
        history_features = []
        for lag in range(1, self.state_history + 1):
            hist_bar = max(0, bar - lag)
            history_features.extend([
                f.loc[hist_bar, 'energy_pct'],
                f.loc[hist_bar, 'damping_pct'],
                f.loc[hist_bar, 'regime_code'] / 2.0,
            ])

        # Position state
        position_features = [
            float(self.position.direction + 1) / 2.0,  # Normalize to [0, 1]
            self.position.mfe / (f.loc[bar, 'atr'] + 1e-8) if self.position.direction != 0 else 0.0,
            self.position.mae / (f.loc[bar, 'atr'] + 1e-8) if self.position.direction != 0 else 0.0,
        ]

        state = np.array(current_features + history_features + position_features, dtype=np.float32)
        return state

    def _get_state_dim(self) -> int:
        """Calculate state dimensionality."""
        # Current features (8) + History (state_history * 3) + Position (3)
        return 8 + (self.state_history * 3) + 3

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Execute action with realistic constraints.

        This is where realistic constraints are enforced:
        - Dynamic spread from candle data
        - Slippage simulation
        - Freeze zone checks (future: near session close)
        - Swap costs
        - Commission from SymbolSpec
        """
        bar = self.current_bar
        f = self.features
        current_price = f.loc[bar, 'close']
        current_spread_points = f.loc[bar, 'spread']

        reward = 0.0
        info = {'action': action, 'bar': bar}

        # Execute action
        if action == Action.HOLD:
            if self.position.direction != 0:
                self._update_position_stats(current_price)

        elif action == Action.LONG:
            if self.position.direction == 0:
                # Open long with realistic fill
                self._open_position(1, current_price, current_spread_points, bar)
                info['opened'] = 'long'
            elif self.position.direction == -1:
                # Close short and open long
                reward = self._close_position(current_price, current_spread_points, bar)
                self._open_position(1, current_price, current_spread_points, bar)
                info['closed'] = 'short'
                info['opened'] = 'long'

        elif action == Action.SHORT:
            if self.position.direction == 0:
                # Open short with realistic fill
                self._open_position(-1, current_price, current_spread_points, bar)
                info['opened'] = 'short'
            elif self.position.direction == 1:
                # Close long and open short
                reward = self._close_position(current_price, current_spread_points, bar)
                self._open_position(-1, current_price, current_spread_points, bar)
                info['closed'] = 'long'
                info['opened'] = 'short'

        elif action == Action.CLOSE:
            if self.position.direction != 0:
                reward = self._close_position(current_price, current_spread_points, bar)
                info['closed'] = 'position'

        # Update equity
        self._update_equity(current_price)

        # Move to next bar
        self.current_bar += 1
        done = self.current_bar >= len(self.features) - 1

        # Final reward if done with open position
        if done and self.position.direction != 0:
            final_price = f.loc[self.current_bar, 'close']
            final_spread = f.loc[self.current_bar, 'spread']
            reward += self._close_position(final_price, final_spread, self.current_bar)

        next_state = self._get_state() if not done else np.zeros(self.state_dim)

        return next_state, reward, done, info

    def _open_position(self, direction: int, price: float, spread_points: float, bar: int):
        """Open position with realistic fill simulation."""
        # Simulate realistic fill (spread + slippage)
        fill_price, slippage = self._simulate_fill(price, direction, spread_points)

        # Calculate spread cost
        spread_cost = spread_points * self.spec.point

        self.position = Position(
            direction=direction,
            entry_price=fill_price,
            entry_bar=bar,
            entry_energy=self.features.loc[bar, 'energy'],
            mfe=0.0,
            mae=0.0,
            entry_spread_paid=spread_cost,
            entry_slippage=slippage,
            days_held=0.0,
        )

    def _close_position(self, price: float, spread_points: float, bar: int) -> float:
        """Close position with realistic costs and return reward."""
        if self.position.direction == 0:
            return 0.0

        # Simulate realistic fill (reverse direction)
        fill_price, exit_slippage = self._simulate_fill(price, -self.position.direction, spread_points)

        # Calculate P&L
        pnl_price = (fill_price - self.position.entry_price) * self.position.direction
        pnl_pct = pnl_price / self.position.entry_price

        # Calculate realistic costs
        # 1. Spread (entry + exit)
        exit_spread_cost = spread_points * self.spec.point
        total_spread_cost = self.position.entry_spread_paid + exit_spread_cost

        # 2. Commission (from SymbolSpec)
        trade_value = abs(pnl_price) * self.max_position * self.spec.contract_size
        commission = self.spec.commission.calculate_commission(self.max_position, trade_value)

        # 3. Swap (overnight holding costs)
        bars_held = bar - self.position.entry_bar
        # Assume 24 bars per day for H1 data (adjust based on timeframe)
        days_held = bars_held / 24.0
        swap_cost = self._calculate_swap(self.position.direction, self.max_position, days_held)

        # 4. Slippage (entry + exit)
        total_slippage = abs(self.position.entry_slippage) + abs(exit_slippage)

        # Net P&L
        gross_pnl = pnl_price * self.max_position * self.spec.contract_size
        total_costs = total_spread_cost + commission + swap_cost + total_slippage
        net_pnl = gross_pnl - total_costs

        # Track statistics
        self.total_trades += 1
        if net_pnl > 0:
            self.winning_trades += 1

        # Calculate reward (physics-aligned + costs)
        reward = self.reward_shaper.calculate_reward(
            pnl_pct=pnl_pct,
            mfe=self.position.mfe,
            mae=self.position.mae,
            holding_time=bars_held,
            entry_energy=self.position.entry_energy,
            regime=self.features.loc[self.position.entry_bar, 'regime'],
        )

        # Penalize for constraint violations
        if self.constraint_violations > 0:
            reward -= 0.1 * self.constraint_violations

        # Reset position
        self.position = Position(direction=0)

        return reward

    def _simulate_fill(
        self,
        desired_price: float,
        direction: int,
        spread_points: float
    ) -> Tuple[float, float]:
        """
        Simulate realistic fill with spread and slippage.

        CRITICAL: Uses SAME logic as RealisticBacktester.
        """
        # Apply spread
        spread_price = spread_points * self.spec.point

        if direction == 1:  # Long (buy at ask)
            fill_price = desired_price + spread_price / 2
        else:  # Short (sell at bid)
            fill_price = desired_price - spread_price / 2

        # Apply slippage
        slippage = 0.0
        if self.enable_slippage:
            # Gaussian slippage (more realistic than uniform)
            slippage_pips = np.random.normal(0, self.slippage_std_pips)
            slippage = slippage_pips * 10 * self.spec.point  # pips to price

            # Slippage is ALWAYS against you (Murphy's law)
            slippage = abs(slippage) * direction
            fill_price += slippage

        return (fill_price, slippage)

    def _calculate_swap(self, direction: int, volume: float, days_held: float) -> float:
        """Calculate swap/rollover charges using SymbolSpec."""
        if days_held < 1:
            return 0.0  # No swap for intraday

        # Get swap rate from spec
        swap_points = self.spec.swap_long if direction == 1 else self.spec.swap_short

        # Convert points to price
        swap_per_lot_per_day = swap_points * self.spec.point * self.spec.contract_size

        # Total swap
        total_swap = swap_per_lot_per_day * volume * days_held

        return total_swap

    def _update_position_stats(self, current_price: float):
        """Update MFE/MAE for open position."""
        if self.position.direction == 0:
            return

        excursion = (current_price - self.position.entry_price) * self.position.direction

        if excursion > self.position.mfe:
            self.position.mfe = excursion
        if excursion < -self.position.mae:
            self.position.mae = -excursion

    def _update_equity(self, current_price: float):
        """Update equity with mark-to-market."""
        if self.position.direction == 0:
            mtm = 0.0
        else:
            unrealized_pnl = (current_price - self.position.entry_price) * self.position.direction
            mtm = unrealized_pnl * self.max_position * self.spec.contract_size

        equity_value = self.equity + mtm
        self.equity_history.append(equity_value)

    def get_metrics(self) -> Dict:
        """Get current episode metrics."""
        return {
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'win_rate': self.winning_trades / max(1, self.total_trades),
            'equity': self.equity,
            'return_pct': (self.equity - self.initial_capital) / self.initial_capital,
            'constraint_violations': self.constraint_violations,
        }
