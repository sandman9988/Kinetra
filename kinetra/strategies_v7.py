"""
Trading Strategies based on Energy-Transfer Theorem v7.0

Implements Berserker and Sniper agents with exact physics formulations
and energy-weighted dynamic exits.
"""

import numpy as np
import pandas as pd
from backtesting import Backtest, Strategy

from .physics_v7 import (
    compute_body_ratio_indicator,
    compute_energy_v7,
    compute_damping_v7,
    compute_entropy_v7,
    compute_agent_signal,
    EnergyWeightedExitManager,
)


# =============================================================================
# HELPER INDICATORS
# =============================================================================

def compute_velocity_v7(close):
    """Price velocity for direction detection."""
    close = pd.Series(close)
    return close.diff().fillna(0.0).values


def compute_momentum_v7(close, lookback: int = 10):
    """Momentum: current / previous."""
    close = pd.Series(close)
    result = close / close.shift(lookback)
    return result.fillna(1.0).values


def compute_trend_strength(close, lookback: int = 20):
    """Trend strength: |SMA direction| / volatility."""
    close = pd.Series(close)
    sma = close.rolling(lookback).mean()
    sma_direction = sma.diff()
    volatility = close.rolling(lookback).std() + 1e-10
    strength = np.abs(sma_direction) / volatility
    return strength.fillna(0.0).values


# =============================================================================
# BASE V7 STRATEGY
# =============================================================================

class BaseV7Strategy(Strategy):
    """
    Base class for v7.0 physics strategies.

    Provides:
    - Physics indicators (body_ratio, energy, damping, entropy)
    - Agent signal computation
    - Position sizing
    """
    # Parameters
    lookback = 20
    vol_ewma_span = 10
    min_history = 50
    risk_per_trade = 0.02

    def init(self):
        """Initialize physics indicators."""
        # Body ratio: |C-O| / (H-L+ε)
        self.body_ratio = self.I(
            compute_body_ratio_indicator,
            self.data.Open, self.data.High, self.data.Low, self.data.Close
        )

        # Energy: body_ratio² × vol_ewma (normalized)
        self.energy = self.I(
            compute_energy_v7,
            self.data.Open, self.data.High, self.data.Low, self.data.Close,
            self.data.Volume, self.vol_ewma_span
        )

        # Damping: range_t / range_{t-1}
        self.damping = self.I(
            compute_damping_v7,
            self.data.High, self.data.Low
        )

        # Entropy: CoV of volume
        self.entropy = self.I(
            compute_entropy_v7,
            self.data.Volume, self.lookback
        )

        # Velocity for direction
        self.velocity = self.I(compute_velocity_v7, self.data.Close)

        # Agent signal: 0=None, 1=Sniper, 2=Berserker
        self.agent_signal = self.I(
            compute_agent_signal,
            self.energy, self.damping, self.min_history
        )

        # Trend strength for confirmation
        self.trend_strength = self.I(
            compute_trend_strength,
            self.data.Close, self.lookback
        )

    def get_position_size(self) -> float:
        """Calculate position size based on risk parameters."""
        return self.risk_per_trade


# =============================================================================
# BERSERKER STRATEGY
# =============================================================================

class BerserkerStrategy(BaseV7Strategy):
    """
    Berserker Agent Strategy.

    Activation: energy > Q75 AND damping < Q25 (Underdamped, High Energy)

    Behavior:
    - Aggressive trend following
    - Enters on strong momentum with conviction
    - Larger position sizes
    - Trails aggressively

    Physics Principle: Underdamped systems oscillate with increasing amplitude.
    In markets, this means momentum builds and trends extend.
    """
    # Berserker-specific parameters
    entry_energy_threshold = 75  # Percentile
    entry_damping_threshold = 25  # Percentile
    position_multiplier = 1.5     # Larger positions

    def init(self):
        super().init()
        # Track entry state for exit calculations
        self.entry_energy_value = 0.0
        self.entry_price = 0.0
        self.cumulative_score = 0.0
        self.max_score = 0.0
        self.bars_in_trade = 0

    def next(self):
        if len(self.data.Close) < self.min_history:
            return

        # Current agent signal
        agent = self.agent_signal[-1]

        # Check for Berserker activation (signal = 2)
        if agent == 2 and not self.position:
            # Use price change for direction instead of velocity indicator
            price_change = self.data.Close[-1] - self.data.Close[-2]

            # Enter in direction of momentum
            if price_change > 0:
                self.buy()
                self.entry_energy_value = self.energy[-1]
                self.entry_price = self.data.Close[-1]
                self.cumulative_score = 0.0
                self.max_score = 0.0
                self.bars_in_trade = 0

            elif price_change < 0:
                self.sell()
                self.entry_energy_value = self.energy[-1]
                self.entry_price = self.data.Close[-1]
                self.cumulative_score = 0.0
                self.max_score = 0.0
                self.bars_in_trade = 0

        # Manage open position with energy-weighted exit
        elif self.position:
            self.bars_in_trade += 1

            # Update energy-weighted score
            if len(self.data.Close) >= 2:
                delta_c = abs(self.data.Close[-1] - self.data.Close[-2])
                bar_score = delta_c * self.energy[-1] / (self.entry_energy_value + 1e-10)

                # Direction check using tracked entry price
                if self.position.is_long:
                    pnl_dir = 1 if self.data.Close[-1] > self.entry_price else -1
                else:
                    pnl_dir = 1 if self.data.Close[-1] < self.entry_price else -1

                if pnl_dir > 0:
                    self.cumulative_score += bar_score
                else:
                    self.cumulative_score -= bar_score * 0.5

                self.cumulative_score = max(0, self.cumulative_score)
                self.max_score = max(self.max_score, self.cumulative_score)

            # Exit condition: score declined below 85% of max
            if self.max_score > 0 and self.bars_in_trade > 3:
                score_ratio = self.cumulative_score / self.max_score
                if score_ratio < 0.85:
                    self.position.close()

            # Also exit if agent signal changes (regime shift)
            if agent != 2 and self.bars_in_trade > 5:
                self.position.close()


# =============================================================================
# SNIPER STRATEGY
# =============================================================================

class SniperStrategy(BaseV7Strategy):
    """
    Sniper Agent Strategy.

    Activation: Q25 < damping < Q75 AND energy > Q60 (Critical, Moderate-High Energy)

    Behavior:
    - Precision entries at optimal points
    - Waits for confirmation
    - Smaller position sizes, tighter risk
    - More selective

    Physics Principle: Critical damping reaches equilibrium fastest.
    In markets, this means efficient mean reversion or trend continuation.
    """
    # Sniper-specific parameters
    entry_energy_threshold = 60   # Percentile
    position_multiplier = 1.0     # Standard positions
    confirmation_bars = 2         # Wait for confirmation

    def init(self):
        super().init()
        self.signal_count = 0
        self.entry_energy_value = 0.0
        self.entry_price = 0.0
        self.cumulative_score = 0.0
        self.max_score = 0.0
        self.bars_in_trade = 0

    def next(self):
        if len(self.data.Close) < self.min_history:
            return

        agent = self.agent_signal[-1]

        # Track consecutive Sniper signals for confirmation
        if agent == 1:
            self.signal_count += 1
        else:
            self.signal_count = 0

        # Enter only after confirmation
        if agent == 1 and self.signal_count >= self.confirmation_bars and not self.position:
            # Use price change for direction
            price_change = self.data.Close[-1] - self.data.Close[-2]

            if price_change > 0:
                self.buy()
                self.entry_energy_value = self.energy[-1]
                self.entry_price = self.data.Close[-1]
                self.cumulative_score = 0.0
                self.max_score = 0.0
                self.bars_in_trade = 0

            elif price_change < 0:
                self.sell()
                self.entry_energy_value = self.energy[-1]
                self.entry_price = self.data.Close[-1]
                self.cumulative_score = 0.0
                self.max_score = 0.0
                self.bars_in_trade = 0

        # Manage open position
        elif self.position:
            self.bars_in_trade += 1

            # Energy-weighted score update
            if len(self.data.Close) >= 2:
                delta_c = abs(self.data.Close[-1] - self.data.Close[-2])
                bar_score = delta_c * self.energy[-1] / (self.entry_energy_value + 1e-10)

                if self.position.is_long:
                    pnl_dir = 1 if self.data.Close[-1] > self.entry_price else -1
                else:
                    pnl_dir = 1 if self.data.Close[-1] < self.entry_price else -1

                if pnl_dir > 0:
                    self.cumulative_score += bar_score
                else:
                    self.cumulative_score -= bar_score * 0.5

                self.cumulative_score = max(0, self.cumulative_score)
                self.max_score = max(self.max_score, self.cumulative_score)

            # Exit: score peaked (Sniper is more aggressive on exits)
            if self.max_score > 0 and self.bars_in_trade > 2:
                score_ratio = self.cumulative_score / self.max_score
                if score_ratio < 0.80:  # Tighter exit for Sniper
                    self.position.close()

            # Exit if signal changes
            if agent != 1 and self.bars_in_trade > 3:
                self.position.close()


# =============================================================================
# COMBINED MULTI-AGENT STRATEGY
# =============================================================================

class MultiAgentV7Strategy(BaseV7Strategy):
    """
    Combined Multi-Agent Strategy (v7.0).

    Dynamically switches between Berserker and Sniper based on physics state.
    Each agent has its own entry/exit logic while sharing physics calculations.

    Agent Selection:
    - Berserker: Underdamped (energy > Q75, damping < Q25)
    - Sniper: Critical (Q25 < damping < Q75, energy > Q60)
    - None: All other states (stay flat)
    """
    berserker_position_mult = 1.5
    sniper_position_mult = 1.0
    confirmation_bars = 2

    def init(self):
        super().init()
        self.current_agent = 0  # 0=None, 1=Sniper, 2=Berserker
        self.signal_count = 0
        self.entry_energy_value = 0.0
        self.entry_price = 0.0
        self.cumulative_score = 0.0
        self.max_score = 0.0
        self.bars_in_trade = 0

    def next(self):
        if len(self.data.Close) < self.min_history:
            return

        agent = self.agent_signal[-1]

        # Track signals
        if agent == self.current_agent:
            self.signal_count += 1
        else:
            self.signal_count = 0
            self.current_agent = int(agent)

        # No position - look for entries
        if not self.position:
            price_change = self.data.Close[-1] - self.data.Close[-2]

            # Berserker activation (immediate)
            if agent == 2:
                if price_change > 0:
                    self.buy()
                    self._record_entry()
                elif price_change < 0:
                    self.sell()
                    self._record_entry()

            # Sniper activation (requires confirmation)
            elif agent == 1 and self.signal_count >= self.confirmation_bars:
                if price_change > 0:
                    self.buy()
                    self._record_entry()
                elif price_change < 0:
                    self.sell()
                    self._record_entry()

        # Manage position
        else:
            self.bars_in_trade += 1
            self._update_score()

            # Dynamic exit based on which agent entered
            exit_threshold = 0.85 if self.current_agent == 2 else 0.80

            if self.max_score > 0 and self.bars_in_trade > 3:
                score_ratio = self.cumulative_score / self.max_score
                if score_ratio < exit_threshold:
                    self.position.close()
                    return

            # Exit on agent change (regime shift)
            if agent == 0 and self.bars_in_trade > 5:
                self.position.close()

    def _record_entry(self):
        """Record entry metrics."""
        self.entry_energy_value = self.energy[-1]
        self.entry_price = self.data.Close[-1]
        self.cumulative_score = 0.0
        self.max_score = 0.0
        self.bars_in_trade = 0

    def _update_score(self):
        """Update energy-weighted score."""
        if len(self.data.Close) < 2:
            return

        delta_c = abs(self.data.Close[-1] - self.data.Close[-2])
        bar_score = delta_c * self.energy[-1] / (self.entry_energy_value + 1e-10)

        if self.position.is_long:
            pnl_dir = 1 if self.data.Close[-1] > self.entry_price else -1
        else:
            pnl_dir = 1 if self.data.Close[-1] < self.entry_price else -1

        if pnl_dir > 0:
            self.cumulative_score += bar_score
        else:
            self.cumulative_score -= bar_score * 0.5

        self.cumulative_score = max(0, self.cumulative_score)
        self.max_score = max(self.max_score, self.cumulative_score)


# =============================================================================
# STRATEGY REGISTRY
# =============================================================================

STRATEGY_V7_REGISTRY = {
    "berserker": BerserkerStrategy,
    "sniper": SniperStrategy,
    "multi_agent_v7": MultiAgentV7Strategy,
}


def list_v7_strategies():
    """List available v7.0 strategies."""
    return list(STRATEGY_V7_REGISTRY.keys())


def get_v7_strategy(name: str):
    """Get v7.0 strategy by name."""
    if name not in STRATEGY_V7_REGISTRY:
        raise ValueError(f"Unknown strategy: {name}. Available: {list(STRATEGY_V7_REGISTRY.keys())}")
    return STRATEGY_V7_REGISTRY[name]
