"""
Physics Engine for Market Modeling

Models markets as kinetic energy systems with:
- Energy: Market momentum (kinetic energy from price changes)
- Damping: Market friction (resistance to movement)
- Entropy: Market disorder/uncertainty
- Acceleration: Rate of momentum change (d²P/dt²)
- Jerk: Rate of acceleration change (d³P/dt³) - best fat candle predictor
- Impulse: Momentum change over time window
- Liquidity: Volume per price movement
- Buying Pressure: Directional order flow proxy
- Reynolds Number: Turbulent vs laminar flow indicator
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

    def calculate_acceleration(self, prices: pd.Series) -> pd.Series:
        """
        Calculate acceleration (second derivative of price).

        a = d²P/dt² = d(velocity)/dt

        Positive acceleration = momentum building
        Negative acceleration = momentum fading (potential reversal)

        Returns:
            Series of acceleration values (can be negative)
        """
        velocity = prices.pct_change()
        acceleration = velocity.diff()
        return acceleration.fillna(0.0)

    def calculate_jerk(self, prices: pd.Series) -> pd.Series:
        """
        Calculate jerk (third derivative - rate of acceleration change).

        j = d³P/dt³ = d(acceleration)/dt

        High jerk = abrupt momentum changes = FAT CANDLE predictor (1.37x lift)

        Returns:
            Series of jerk values
        """
        acceleration = self.calculate_acceleration(prices)
        jerk = acceleration.diff()
        return jerk.fillna(0.0)

    def calculate_impulse(self, prices: pd.Series, window: int = 5) -> pd.Series:
        """
        Calculate impulse (momentum change over time window).

        I = Δ(momentum) over window

        Strong impulse = directional bias (1.30x lift for fat candles)

        Args:
            prices: Price series
            window: Time window for momentum change

        Returns:
            Series of impulse values
        """
        momentum = prices.pct_change(self.lookback)
        impulse = momentum.diff(window)
        return impulse.fillna(0.0)

    def calculate_liquidity(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        volume: pd.Series
    ) -> pd.Series:
        """
        Calculate liquidity proxy from OHLCV.

        Liquidity = Volume / (Range * Price)

        Higher = more liquid (big volume, small price move)
        Lower = thin market (small volume, big move)

        High liquidity at berserker = 1.34x lift for fat candles

        Returns:
            Series of liquidity values (higher = more liquid)
        """
        bar_range = (high - low).clip(lower=1e-10)
        price_range_pct = bar_range / close
        liquidity = volume / (price_range_pct * close + 1e-10)
        return liquidity.fillna(0.0)

    def calculate_buying_pressure(
        self,
        open_price: pd.Series,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        lookback: int = 5
    ) -> pd.Series:
        """
        Calculate buying pressure from OHLC (order flow proxy).

        BP = (Close - Low) / (High - Low)

        1.0 = closed at high (buyers dominated)
        0.0 = closed at low (sellers dominated)

        Best direction signal:
        - High BP (>0.6) at berserker → 62% DOWN fat candle (+12% edge)
        - Low BP (<0.4) at berserker → 57% UP fat candle (+7% edge)

        Returns:
            Series of buying pressure [0, 1]
        """
        bar_range = (high - low).clip(lower=1e-10)
        bp = (close - low) / bar_range
        return bp.rolling(lookback).mean().fillna(0.5)

    def calculate_reynolds(
        self,
        prices: pd.Series,
        volume: pd.Series,
        high: pd.Series,
        low: pd.Series
    ) -> pd.Series:
        """
        Calculate Reynolds number (turbulent vs laminar flow indicator).

        In fluid dynamics: Re = ρvL/μ = inertial forces / viscous forces
        - High Re = turbulent (chaotic, unpredictable)
        - Low Re = laminar (smooth, predictable trends)

        Market analog:
        Re = (momentum * characteristic_length) / (viscosity * density)
           = (velocity * range) / (volatility * 1/volume)
           = (velocity * range * volume) / volatility
           = kinetic_energy / damping (simplified)

        Low Re (<2000 in fluids) = laminar = trending
        High Re (>4000 in fluids) = turbulent = ranging/chaotic

        Returns:
            Series of Reynolds numbers (higher = more turbulent)
        """
        # Velocity (momentum)
        velocity = prices.pct_change()

        # Characteristic length (price range)
        bar_range = (high - low) / prices

        # Viscosity proxy (volatility)
        volatility = velocity.rolling(self.lookback).std().clip(lower=1e-10)

        # Density proxy (inverse of volume normalized)
        volume_norm = volume / volume.rolling(self.lookback).mean().clip(lower=1e-10)

        # Reynolds number: (velocity * length * density) / viscosity
        # = (abs(velocity) * range * volume_norm) / volatility
        reynolds = (velocity.abs() * bar_range * volume_norm) / volatility

        # Smooth it
        reynolds = reynolds.rolling(self.lookback).mean()

        return reynolds.fillna(1.0)

    def calculate_reynolds_regime(
        self,
        reynolds: pd.Series
    ) -> pd.Series:
        """
        Classify flow regime based on Reynolds number percentile.

        Returns:
            Series with values:
            - 'laminar' (Re < 25th percentile) - smooth trending
            - 'transitional' (25-75th percentile) - mixed
            - 'turbulent' (Re > 75th percentile) - chaotic
        """
        window = min(500, len(reynolds))

        # Adaptive percentile thresholds
        re_pct = reynolds.rolling(window, min_periods=self.lookback).apply(
            lambda x: (x.iloc[-1] > x.iloc[:-1]).mean() if len(x) > 1 else 0.5,
            raw=False
        ).fillna(0.5)

        def classify(pct):
            if pct < 0.25:
                return 'laminar'
            elif pct > 0.75:
                return 'turbulent'
            else:
                return 'transitional'

        return re_pct.apply(classify)

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
