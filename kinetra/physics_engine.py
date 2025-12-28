"""
Physics Engine for Market Modeling

Models markets as kinetic energy systems with:
- Energy: Market momentum (kinetic energy from price changes)
- Damping: Market friction (resistance to movement)
- Entropy: Market disorder/uncertainty
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional
from enum import Enum


class RegimeType(Enum):
    """Market regime classifications based on physics state."""
    UNDERDAMPED = "underdamped"  # High energy, low friction -> trending
    CRITICAL = "critical"          # Balanced -> transitional
    OVERDAMPED = "overdamped"      # Low energy, high friction -> ranging


class PhysicsEngine:
    """
    Physics-based market state calculator.
    
    Uses first principles to compute:
    - Kinetic Energy: E_t = 0.5 * m * (ΔP_t / Δt)²
    - Damping Coefficient: ζ = friction / (2 * √(spring_constant * mass))
    - Entropy: H = -Σ p_i * log(p_i) for price distribution
    """
    
    def __init__(self, mass: float = 1.0, lookback: int = 20):
        """
        Initialize physics engine.
        
        Args:
            mass: Virtual mass for kinetic energy calculation (default: 1.0)
            lookback: Window for rolling calculations (default: 20 bars)
        """
        self.mass = mass
        self.lookback = lookback
    
    def calculate_energy(self, prices: pd.Series) -> pd.Series:
        """
        Calculate kinetic energy from price momentum.
        
        E_t = 0.5 * m * (ΔP_t / Δt)²
        
        Args:
            prices: Time series of prices
            
        Returns:
            Series of kinetic energy values (always >= 0)
        """
        # Calculate velocity (price change rate)
        velocity = prices.diff() / 1.0  # Δt = 1 bar
        
        # Kinetic energy formula
        energy = 0.5 * self.mass * velocity ** 2
        
        # NaN shield: first value is NaN due to diff()
        energy = energy.fillna(0.0)
        
        # Ensure non-negative (physics constraint)
        assert (energy >= 0).all(), "Energy cannot be negative (physics violation)"
        
        return energy
    
    def calculate_damping(self, prices: pd.Series, volume: Optional[pd.Series] = None) -> pd.Series:
        """
        Calculate damping coefficient (friction).
        
        Uses volatility as proxy for friction:
        ζ = rolling_std(returns) / rolling_mean(|returns|)
        
        Args:
            prices: Time series of prices
            volume: Optional volume data (for liquidity-based friction)
            
        Returns:
            Series of damping coefficients (always >= 0)
        """
        returns = prices.pct_change()
        
        # Rolling statistics
        volatility = returns.rolling(self.lookback).std()
        mean_abs_return = returns.abs().rolling(self.lookback).mean()
        
        # Damping = noise-to-signal ratio
        damping = volatility / (mean_abs_return + 1e-10)  # Avoid division by zero
        
        # Optional: incorporate volume (low volume -> higher friction)
        if volume is not None:
            volume_factor = 1.0 / (volume.rolling(self.lookback).mean() + 1e-10)
            damping = damping * volume_factor
        
        # NaN shield
        damping = damping.fillna(damping.mean())
        
        # Ensure non-negative
        damping = damping.clip(lower=0.0)
        
        return damping
    
    def calculate_entropy(self, prices: pd.Series, bins: int = 10) -> pd.Series:
        """
        Calculate Shannon entropy of price distribution.
        
        H = -Σ p_i * log(p_i)
        
        Higher entropy = more disorder/uncertainty
        
        Args:
            prices: Time series of prices
            bins: Number of bins for discretization
            
        Returns:
            Series of entropy values
        """
        returns = prices.pct_change().dropna()
        
        def rolling_entropy(window_returns):
            if len(window_returns) < 5:
                return 0.0
            
            # Create histogram
            counts, _ = np.histogram(window_returns, bins=bins)
            
            # Convert to probabilities
            total = counts.sum()
            if total == 0:
                return 0.0
            
            probabilities = counts / total
            
            # Shannon entropy (with NaN shield for log(0))
            probabilities = probabilities[probabilities > 0]
            if len(probabilities) == 0:
                return 0.0
            
            entropy = -np.sum(probabilities * np.log(probabilities))
            
            # Ensure non-negative (numerical stability)
            entropy = max(0.0, entropy)
            
            return entropy
        
        # Apply rolling window
        entropy_series = returns.rolling(self.lookback).apply(rolling_entropy, raw=False)
        
        # NaN shield
        entropy_series = entropy_series.fillna(0.0)
        
        # Ensure all values are non-negative
        entropy_series = entropy_series.clip(lower=0.0)
        
        return entropy_series
    
    def classify_regime(
        self, 
        energy: float, 
        damping: float, 
        history_energy: pd.Series,
        history_damping: pd.Series
    ) -> RegimeType:
        """
        Classify market regime using rolling percentiles (NO FIXED THRESHOLDS).
        
        Args:
            energy: Current kinetic energy
            damping: Current damping coefficient
            history_energy: Historical energy values for percentile calculation
            history_damping: Historical damping values for percentile calculation
            
        Returns:
            RegimeType classification
        """
        # Calculate dynamic thresholds from history
        energy_75pct = np.percentile(history_energy.dropna(), 75)
        damping_25pct = np.percentile(history_damping.dropna(), 25)
        damping_75pct = np.percentile(history_damping.dropna(), 75)
        
        # Physics-based classification
        if energy > energy_75pct and damping < damping_25pct:
            return RegimeType.UNDERDAMPED  # High energy, low friction -> trending
        elif damping_25pct <= damping <= damping_75pct:
            return RegimeType.CRITICAL  # Balanced -> transitional
        else:
            return RegimeType.OVERDAMPED  # High friction -> ranging
    
    def compute_physics_state(
        self,
        prices: pd.Series,
        volume: Optional[pd.Series] = None,
        include_percentiles: bool = True
    ) -> pd.DataFrame:
        """
        Compute complete physics state for market data.

        Args:
            prices: Time series of prices
            volume: Optional volume data
            include_percentiles: Include rolling percentile ranks (adaptive per instrument)

        Returns:
            DataFrame with columns: energy, damping, entropy, regime, and optionally
            energy_pct, damping_pct, entropy_pct (rolling percentile ranks 0-1)
        """
        energy = self.calculate_energy(prices)
        damping = self.calculate_damping(prices, volume)
        entropy = self.calculate_entropy(prices)

        # Ensure all series have same index
        energy = energy.reindex(prices.index, fill_value=0.0)
        damping = damping.reindex(prices.index, fill_value=0.0)
        entropy = entropy.reindex(prices.index, fill_value=0.0)

        # Create state DataFrame
        state = pd.DataFrame({
            'energy': energy,
            'damping': damping,
            'entropy': entropy
        }, index=prices.index)

        # Force non-negative values (numerical stability)
        state['energy'] = state['energy'].clip(lower=0.0)
        state['damping'] = state['damping'].clip(lower=0.0)
        state['entropy'] = state['entropy'].clip(lower=0.0)

        # Add rolling percentile ranks (ADAPTIVE PER INSTRUMENT)
        # These are the key features for RL - no fixed thresholds
        if include_percentiles:
            window = min(500, len(state))  # Rolling window for percentiles
            state['energy_pct'] = state['energy'].rolling(window, min_periods=self.lookback).apply(
                lambda x: (x.iloc[-1] > x.iloc[:-1]).mean() if len(x) > 1 else 0.5,
                raw=False
            ).fillna(0.5)
            state['damping_pct'] = state['damping'].rolling(window, min_periods=self.lookback).apply(
                lambda x: (x.iloc[-1] > x.iloc[:-1]).mean() if len(x) > 1 else 0.5,
                raw=False
            ).fillna(0.5)
            state['entropy_pct'] = state['entropy'].rolling(window, min_periods=self.lookback).apply(
                lambda x: (x.iloc[-1] > x.iloc[:-1]).mean() if len(x) > 1 else 0.5,
                raw=False
            ).fillna(0.5)

        # Classify regime for each timestep (human-readable label, not for trading)
        regimes = []
        for i in range(len(state)):
            if i < self.lookback:
                regimes.append(RegimeType.CRITICAL)  # Not enough history
            else:
                history_energy = state['energy'].iloc[:i]
                history_damping = state['damping'].iloc[:i]
                regime = self.classify_regime(
                    state['energy'].iloc[i],
                    state['damping'].iloc[i],
                    history_energy,
                    history_damping
                )
                regimes.append(regime)

        state['regime'] = [r.value for r in regimes]

        # Validate physics constraints
        assert (state['energy'] >= 0).all(), "Energy must be non-negative"
        assert (state['damping'] >= 0).all(), "Damping must be non-negative"
        assert (state['entropy'] >= 0).all(), "Entropy must be non-negative"

        return state


# Standalone functions for convenience
def calculate_energy(prices: pd.Series, mass: float = 1.0) -> pd.Series:
    """Calculate kinetic energy from prices."""
    engine = PhysicsEngine(mass=mass)
    return engine.calculate_energy(prices)


def calculate_damping(prices: pd.Series, lookback: int = 20) -> pd.Series:
    """Calculate damping coefficient from prices."""
    engine = PhysicsEngine(lookback=lookback)
    return engine.calculate_damping(prices)


def calculate_entropy(prices: pd.Series, lookback: int = 20, bins: int = 10) -> pd.Series:
    """Calculate Shannon entropy from prices."""
    engine = PhysicsEngine(lookback=lookback)
    return engine.calculate_entropy(prices, bins=bins)
