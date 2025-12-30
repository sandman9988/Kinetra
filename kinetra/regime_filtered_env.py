"""
Regime-Filtered Trading Environment

Extends TradingEnv with 3-regime filtering:
1. Physics Regime: laminar, underdamped, overdamped
2. Volatility Regime: low_vol, medium_vol, high_vol
3. Momentum Regime: uptrend, ranging, downtrend

Enables focused training on specific market conditions for regime-specialized agents.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, List, Set
from dataclasses import dataclass
from enum import Enum

from .trading_env import TradingEnv, Position, Action


class PhysicsRegime(Enum):
    """Physics-based regime classification."""
    LAMINAR = "laminar"          # Low entropy, stable trends
    UNDERDAMPED = "underdamped"  # Mean-reverting oscillations
    OVERDAMPED = "overdamped"    # Choppy, high friction
    CRITICAL = "critical"        # Transition state


class VolatilityRegime(Enum):
    """Volatility-based regime classification."""
    LOW_VOL = "low_vol"       # Low volatility (< 33rd percentile)
    MEDIUM_VOL = "medium_vol" # Medium volatility (33-67th percentile)
    HIGH_VOL = "high_vol"     # High volatility (> 67th percentile)


class MomentumRegime(Enum):
    """Momentum-based regime classification."""
    UPTREND = "uptrend"     # Strong upward momentum
    RANGING = "ranging"     # Sideways/choppy
    DOWNTREND = "downtrend" # Strong downward momentum


@dataclass
class RegimeFilter:
    """
    Configuration for regime filtering.

    Set any regime to None to allow all values (no filtering).
    Set to specific values to filter to only those regimes.
    """
    physics_regimes: Optional[Set[PhysicsRegime]] = None
    volatility_regimes: Optional[Set[VolatilityRegime]] = None
    momentum_regimes: Optional[Set[MomentumRegime]] = None

    def matches(
        self,
        physics: PhysicsRegime,
        volatility: VolatilityRegime,
        momentum: MomentumRegime,
    ) -> bool:
        """Check if a bar matches the filter criteria."""
        # If filter is None, accept all
        if self.physics_regimes is not None and physics not in self.physics_regimes:
            return False

        if self.volatility_regimes is not None and volatility not in self.volatility_regimes:
            return False

        if self.momentum_regimes is not None and momentum not in self.momentum_regimes:
            return False

        return True

    def __str__(self) -> str:
        """Human-readable filter description."""
        parts = []
        if self.physics_regimes:
            parts.append(f"Physics: {', '.join(r.value for r in self.physics_regimes)}")
        if self.volatility_regimes:
            parts.append(f"Vol: {', '.join(r.value for r in self.volatility_regimes)}")
        if self.momentum_regimes:
            parts.append(f"Momentum: {', '.join(r.value for r in self.momentum_regimes)}")

        return " | ".join(parts) if parts else "No filter (all regimes)"


class RegimeFilteredTradingEnv(TradingEnv):
    """
    Trading environment with 3-regime filtering.

    Filters training episodes to only include bars matching specific regime combinations.
    Useful for training regime-specialized agents.

    Example:
        # Train agent only in low-vol laminar uptrends
        regime_filter = RegimeFilter(
            physics_regimes={PhysicsRegime.LAMINAR},
            volatility_regimes={VolatilityRegime.LOW_VOL},
            momentum_regimes={MomentumRegime.UPTREND},
        )
        env = RegimeFilteredTradingEnv(data, regime_filter=regime_filter)
    """

    def __init__(
        self,
        data: pd.DataFrame,
        regime_filter: Optional[RegimeFilter] = None,
        min_filtered_bars: int = 100,
        **kwargs
    ):
        """
        Initialize regime-filtered environment.

        Args:
            data: OHLCV DataFrame
            regime_filter: RegimeFilter specifying which regimes to include
            min_filtered_bars: Minimum bars that must match filter (safety check)
            **kwargs: Passed to parent TradingEnv.__init__()
        """
        # Initialize parent first (computes physics features)
        super().__init__(data, **kwargs)

        # Store filter
        self.regime_filter = regime_filter or RegimeFilter()  # No filter = allow all

        # Compute additional regime classifications
        self._compute_regime_classifications()

        # Find valid bars matching filter
        self._compute_valid_bars()

        # Validate
        if len(self.valid_bars) < min_filtered_bars:
            raise ValueError(
                f"Only {len(self.valid_bars)} bars match filter (min: {min_filtered_bars}). "
                f"Filter: {self.regime_filter}"
            )

        print(f"[RegimeFilteredEnv] Filter: {self.regime_filter}")
        print(f"[RegimeFilteredEnv] Valid bars: {len(self.valid_bars)}/{len(self.features)} "
              f"({100*len(self.valid_bars)/len(self.features):.1f}%)")

    def _compute_regime_classifications(self):
        """Compute volatility and momentum regime classifications."""
        # Volatility regime (using ATR percentiles)
        atr = self.features['atr']
        vol_percentiles = atr.rolling(200, min_periods=20).apply(
            lambda x: (x.iloc[-1] >= x.iloc[:-1]).mean() if len(x) > 1 else 0.5,
            raw=False
        ).fillna(0.5)

        self.features['vol_regime'] = pd.cut(
            vol_percentiles,
            bins=[0, 0.33, 0.67, 1.0],
            labels=['low_vol', 'medium_vol', 'high_vol'],
            include_lowest=True
        ).astype(str)

        # Momentum regime (using rolling returns)
        momentum_window = 20
        momentum = self.features['close'].pct_change(momentum_window)

        # Classify momentum
        momentum_percentiles = momentum.rolling(200, min_periods=20).apply(
            lambda x: (x.iloc[-1] >= x.iloc[:-1]).mean() if len(x) > 1 else 0.5,
            raw=False
        ).fillna(0.5)

        def classify_momentum(pct):
            if pct > 0.6:
                return 'uptrend'
            elif pct < 0.4:
                return 'downtrend'
            else:
                return 'ranging'

        self.features['momentum_regime'] = momentum_percentiles.apply(classify_momentum)

        # Map physics regime (already computed by parent)
        # Convert from string to PhysicsRegime enum
        regime_map = {
            'underdamped': PhysicsRegime.UNDERDAMPED,
            'critical': PhysicsRegime.CRITICAL,
            'overdamped': PhysicsRegime.OVERDAMPED,
            'laminar': PhysicsRegime.LAMINAR,
        }

        self.features['physics_regime_enum'] = self.features['regime'].map(
            lambda x: regime_map.get(x.lower(), PhysicsRegime.CRITICAL)
        )

        self.features['vol_regime_enum'] = self.features['vol_regime'].map({
            'low_vol': VolatilityRegime.LOW_VOL,
            'medium_vol': VolatilityRegime.MEDIUM_VOL,
            'high_vol': VolatilityRegime.HIGH_VOL,
        })

        self.features['momentum_regime_enum'] = self.features['momentum_regime'].map({
            'uptrend': MomentumRegime.UPTREND,
            'ranging': MomentumRegime.RANGING,
            'downtrend': MomentumRegime.DOWNTREND,
        })

    def _compute_valid_bars(self):
        """Find bars that match the regime filter."""
        self.valid_bars = []

        for idx in range(len(self.features)):
            physics = self.features.iloc[idx]['physics_regime_enum']
            volatility = self.features.iloc[idx]['vol_regime_enum']
            momentum = self.features.iloc[idx]['momentum_regime_enum']

            # Skip if any regime is None (data initialization)
            if pd.isna(physics) or pd.isna(volatility) or pd.isna(momentum):
                continue

            # Check filter
            if self.regime_filter.matches(physics, volatility, momentum):
                self.valid_bars.append(idx)

        self.valid_bars = np.array(self.valid_bars)

    def reset(self) -> np.ndarray:
        """
        Reset environment to start of episode.

        Overrides parent to ensure current_bar starts in valid_bars.
        """
        # Call parent reset first
        state = super().reset()

        # If valid_bars not yet initialized (during __init__), return parent's state
        # This happens when parent's __init__ calls reset() before we've finished setup
        if not hasattr(self, 'valid_bars'):
            return state

        # Now override current_bar to start at a random valid bar
        min_bar = self.lookback + self.state_history + 100  # Need history
        max_bar = len(self.features) - 500  # Need future bars

        valid_start_bars = self.valid_bars[
            (self.valid_bars >= min_bar) & (self.valid_bars <= max_bar)
        ]

        if len(valid_start_bars) == 0:
            raise ValueError("No valid start bars found in filtered set")

        # Override current_bar to random valid bar
        self.current_bar = int(np.random.choice(valid_start_bars))

        # Return fresh state from the new bar
        return self._get_state()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Execute action and return next state.

        Overrides parent to skip to next valid bar if current bar is filtered out.
        """
        # Execute action normally
        next_state, reward, done, info = super().step(action)

        # If not done and current bar is not in valid set, skip to next valid bar
        if not done and self.current_bar not in self.valid_bars:
            # Find next valid bar
            next_valid = self.valid_bars[self.valid_bars > self.current_bar]

            if len(next_valid) == 0:
                # No more valid bars, episode done
                done = True
                info['early_termination'] = 'no_more_valid_bars'
            else:
                # Jump to next valid bar
                self.current_bar = next_valid[0]
                next_state = self._get_state()
                info['skipped_bars'] = next_valid[0] - (self.current_bar - 1)

        return next_state, reward, done, info

    def get_regime_distribution(self) -> Dict[str, Dict[str, int]]:
        """Get distribution of regimes in the dataset."""
        distribution = {
            'physics': {},
            'volatility': {},
            'momentum': {},
        }

        for regime in PhysicsRegime:
            count = (self.features['physics_regime_enum'] == regime).sum()
            distribution['physics'][regime.value] = int(count)

        for regime in VolatilityRegime:
            count = (self.features['vol_regime_enum'] == regime).sum()
            distribution['volatility'][regime.value] = int(count)

        for regime in MomentumRegime:
            count = (self.features['momentum_regime_enum'] == regime).sum()
            distribution['momentum'][regime.value] = int(count)

        return distribution


def create_regime_specialists(data: pd.DataFrame, **env_kwargs) -> Dict[str, RegimeFilteredTradingEnv]:
    """
    Create a set of regime-specialized environments for training.

    Returns dict of environments, one for each important regime combination.
    """
    specialists = {}

    # 1. Laminar trend followers (low/medium vol + uptrend/downtrend)
    specialists['laminar_uptrend'] = RegimeFilteredTradingEnv(
        data,
        regime_filter=RegimeFilter(
            physics_regimes={PhysicsRegime.LAMINAR},
            momentum_regimes={MomentumRegime.UPTREND},
        ),
        **env_kwargs
    )

    specialists['laminar_downtrend'] = RegimeFilteredTradingEnv(
        data,
        regime_filter=RegimeFilter(
            physics_regimes={PhysicsRegime.LAMINAR},
            momentum_regimes={MomentumRegime.DOWNTREND},
        ),
        **env_kwargs
    )

    # 2. Mean-reversion specialists (underdamped + ranging)
    specialists['underdamped_ranging'] = RegimeFilteredTradingEnv(
        data,
        regime_filter=RegimeFilter(
            physics_regimes={PhysicsRegime.UNDERDAMPED},
            momentum_regimes={MomentumRegime.RANGING},
        ),
        **env_kwargs
    )

    # 3. High volatility breakout specialist
    specialists['high_vol_breakout'] = RegimeFilteredTradingEnv(
        data,
        regime_filter=RegimeFilter(
            volatility_regimes={VolatilityRegime.HIGH_VOL},
            physics_regimes={PhysicsRegime.LAMINAR, PhysicsRegime.UNDERDAMPED},
        ),
        **env_kwargs
    )

    # 4. Low volatility grinder (overdamped + low vol = avoid)
    # This specialist learns to stay out of choppy low-edge conditions
    specialists['avoid_choppy'] = RegimeFilteredTradingEnv(
        data,
        regime_filter=RegimeFilter(
            physics_regimes={PhysicsRegime.OVERDAMPED},
            volatility_regimes={VolatilityRegime.LOW_VOL, VolatilityRegime.MEDIUM_VOL},
        ),
        **env_kwargs
    )

    return specialists
