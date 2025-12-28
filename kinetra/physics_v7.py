"""
Energy-Transfer Trading Theorem v7.0 Implementation

Physics-based trading using the exact formulation from the theoretical framework:
- Body Ratio: |C-O| / (H-L+ε)
- Energy: body_ratio² × vol_ewma
- Damping: (H_t-L_t) / (H_{t-1}-L_{t-1}+ε)
- Entropy: σ(V) / μ(V) over lookback window
- Friction: spread_bps + commission_bps + funding_bps

Agent Activation:
- Berserker: energy > Q75 AND damping < Q25 (Underdamped, High Energy)
- Sniper: Q25 < damping < Q75 AND energy > Q60 (Critical, Moderate-High Energy)

Dynamic Exit: Energy-weighted cumulative PnL score S_τ
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, Tuple, List
from enum import Enum
from dataclasses import dataclass


class AgentType(Enum):
    """Agent activation types based on physics state."""
    NONE = "none"           # No agent active - stay flat
    BERSERKER = "berserker" # Underdamped, high energy - aggressive trend following
    SNIPER = "sniper"       # Critical damping, moderate energy - precision entries


class RegimeState(Enum):
    """Market regime based on damping classification."""
    UNDERDAMPED = "underdamped"   # Low friction, trending
    CRITICAL = "critical"         # Balanced friction, transitional
    OVERDAMPED = "overdamped"     # High friction, ranging


@dataclass
class PhysicsState:
    """Complete physics state at a given bar."""
    body_ratio: float
    energy: float
    damping: float
    entropy: float
    friction: float
    regime: RegimeState
    active_agent: AgentType

    def to_dict(self) -> Dict:
        return {
            'body_ratio': self.body_ratio,
            'energy': self.energy,
            'damping': self.damping,
            'entropy': self.entropy,
            'friction': self.friction,
            'regime': self.regime.value,
            'active_agent': self.active_agent.value
        }


@dataclass
class TradeMetrics:
    """Metrics for an open trade."""
    entry_price: float
    entry_bar: int
    entry_energy: float
    direction: int  # 1 for long, -1 for short
    cumulative_score: float
    max_score: float
    bars_held: int


class PhysicsEngineV7:
    """
    Energy-Transfer Trading Theorem v7.0 Physics Engine.

    Implements exact mathematical formulations from the theoretical framework.
    """

    def __init__(
        self,
        lookback: int = 20,
        vol_ewma_span: int = 10,
        epsilon: float = 1e-10,
        spread_bps: float = 1.0,
        commission_bps: float = 0.5,
        funding_bps: float = 0.0
    ):
        """
        Initialize physics engine.

        Args:
            lookback: Window for entropy calculation (w)
            vol_ewma_span: Span for volume EWMA
            epsilon: Smoothing constant (ε)
            spread_bps: Spread in basis points
            commission_bps: Commission in basis points
            funding_bps: Funding rate in basis points
        """
        self.lookback = lookback
        self.vol_ewma_span = vol_ewma_span
        self.epsilon = epsilon
        self.spread_bps = spread_bps
        self.commission_bps = commission_bps
        self.funding_bps = funding_bps

    def calculate_body_ratio(
        self,
        open_: pd.Series,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series
    ) -> pd.Series:
        """
        Calculate body ratio: |C_t - O_t| / (H_t - L_t + ε)

        Measures the proportion of the bar's range covered by the body.
        Higher values = stronger directional conviction.
        """
        body = np.abs(close - open_)
        range_ = high - low + self.epsilon
        body_ratio = body / range_

        # Ensure bounded [0, 1]
        return body_ratio.clip(0.0, 1.0)

    def calculate_energy(
        self,
        body_ratio: pd.Series,
        volume: pd.Series
    ) -> pd.Series:
        """
        Calculate energy: (body_ratio_t)² × vol_ewma_t

        Energy represents market momentum weighted by participation (volume).
        Higher energy = stronger directional move with conviction.
        """
        # Volume EWMA
        vol_ewma = volume.ewm(span=self.vol_ewma_span).mean()

        # Normalize volume to prevent scale issues
        vol_normalized = vol_ewma / (vol_ewma.rolling(100).mean() + self.epsilon)

        # Energy = body_ratio² × normalized volume
        energy = (body_ratio ** 2) * vol_normalized

        return energy.fillna(0.0).clip(lower=0.0)

    def calculate_damping(
        self,
        high: pd.Series,
        low: pd.Series
    ) -> pd.Series:
        """
        Calculate damping: (H_t - L_t) / (H_{t-1} - L_{t-1} + ε)

        Damping represents range expansion/contraction.
        - damping > 1: Range expanding (momentum building)
        - damping < 1: Range contracting (momentum fading)
        - damping ≈ 1: Stable range
        """
        current_range = high - low
        prev_range = current_range.shift(1) + self.epsilon

        damping = current_range / prev_range

        # Clip to reasonable range to avoid extreme outliers
        return damping.fillna(1.0).clip(lower=0.01, upper=10.0)

    def calculate_entropy(
        self,
        volume: pd.Series
    ) -> pd.Series:
        """
        Calculate entropy: σ(V) / μ(V) over lookback window

        This is the Coefficient of Variation (CoV) of volume.
        Higher entropy = more disorder/uncertainty in participation.
        """
        vol_std = volume.rolling(self.lookback).std()
        vol_mean = volume.rolling(self.lookback).mean() + self.epsilon

        entropy = vol_std / vol_mean

        return entropy.fillna(0.0).clip(lower=0.0)

    def calculate_friction(self) -> float:
        """
        Calculate friction: spread_bps + commission_bps + funding_bps

        Total transaction costs in basis points.
        """
        return self.spread_bps + self.commission_bps + self.funding_bps

    def classify_regime(
        self,
        damping: float,
        damping_history: pd.Series
    ) -> RegimeState:
        """
        Classify market regime based on damping percentiles.

        - Underdamped: damping < Q25 (low friction, trending)
        - Critical: Q25 <= damping <= Q75 (balanced)
        - Overdamped: damping > Q75 (high friction, ranging)
        """
        q25 = damping_history.quantile(0.25)
        q75 = damping_history.quantile(0.75)

        if damping < q25:
            return RegimeState.UNDERDAMPED
        elif damping > q75:
            return RegimeState.OVERDAMPED
        else:
            return RegimeState.CRITICAL

    def determine_active_agent(
        self,
        energy: float,
        damping: float,
        energy_history: pd.Series,
        damping_history: pd.Series
    ) -> AgentType:
        """
        Determine which agent should be active based on physics state.

        Berserker Agent:
            energy > Q75(energy) AND damping < Q25(damping)
            → Underdamped, High Energy → Aggressive trend following

        Sniper Agent:
            Q25(damping) < damping < Q75(damping) AND energy > Q60(energy)
            → Critical Damping, Moderate-High Energy → Precision entries

        None: All other conditions → Stay flat
        """
        energy_q60 = energy_history.quantile(0.60)
        energy_q75 = energy_history.quantile(0.75)
        damping_q25 = damping_history.quantile(0.25)
        damping_q75 = damping_history.quantile(0.75)

        # Berserker: High energy + Low damping (Underdamped)
        if energy > energy_q75 and damping < damping_q25:
            return AgentType.BERSERKER

        # Sniper: Moderate-high energy + Critical damping
        if damping_q25 < damping < damping_q75 and energy > energy_q60:
            return AgentType.SNIPER

        return AgentType.NONE

    def compute_physics_state(
        self,
        df: pd.DataFrame,
        min_history: int = 50
    ) -> pd.DataFrame:
        """
        Compute complete physics state for OHLCV DataFrame.

        Args:
            df: DataFrame with Open, High, Low, Close, Volume columns
            min_history: Minimum bars needed for percentile calculations

        Returns:
            DataFrame with physics metrics and agent activations
        """
        # Extract OHLCV
        open_ = df['Open']
        high = df['High']
        low = df['Low']
        close = df['Close']
        volume = df['Volume'] if 'Volume' in df.columns else pd.Series(1.0, index=df.index)

        # Calculate base metrics
        body_ratio = self.calculate_body_ratio(open_, high, low, close)
        energy = self.calculate_energy(body_ratio, volume)
        damping = self.calculate_damping(high, low)
        entropy = self.calculate_entropy(volume)
        friction = self.calculate_friction()

        # Create result DataFrame
        result = pd.DataFrame({
            'body_ratio': body_ratio,
            'energy': energy,
            'damping': damping,
            'entropy': entropy,
            'friction': friction
        }, index=df.index)

        # Classify regime and determine active agent for each bar
        regimes = []
        agents = []

        for i in range(len(result)):
            if i < min_history:
                regimes.append(RegimeState.CRITICAL.value)
                agents.append(AgentType.NONE.value)
            else:
                # Use rolling history for percentile calculations
                energy_hist = result['energy'].iloc[:i]
                damping_hist = result['damping'].iloc[:i]

                current_damping = result['damping'].iloc[i]
                current_energy = result['energy'].iloc[i]

                regime = self.classify_regime(current_damping, damping_hist)
                agent = self.determine_active_agent(
                    current_energy, current_damping,
                    energy_hist, damping_hist
                )

                regimes.append(regime.value)
                agents.append(agent.value)

        result['regime'] = regimes
        result['active_agent'] = agents

        return result


class EnergyWeightedExitManager:
    """
    Dynamic exit based on energy-weighted cumulative PnL score.

    S_τ = Σ(|ΔC_i| × energy_i / Ē_entry)

    Exit when S_τ reaches local maximum AND S_τ > 0.85 × max(S)
    """

    def __init__(self, exit_threshold: float = 0.85):
        """
        Initialize exit manager.

        Args:
            exit_threshold: Fraction of max score to trigger exit (default: 0.85)
        """
        self.exit_threshold = exit_threshold
        self.active_trades: Dict[str, TradeMetrics] = {}

    def open_trade(
        self,
        trade_id: str,
        entry_price: float,
        entry_bar: int,
        entry_energy: float,
        direction: int
    ):
        """Record a new trade opening."""
        self.active_trades[trade_id] = TradeMetrics(
            entry_price=entry_price,
            entry_bar=entry_bar,
            entry_energy=max(entry_energy, 1e-10),  # Prevent division by zero
            direction=direction,
            cumulative_score=0.0,
            max_score=0.0,
            bars_held=0
        )

    def update_score(
        self,
        trade_id: str,
        current_close: float,
        prev_close: float,
        current_energy: float
    ) -> Tuple[bool, float]:
        """
        Update the energy-weighted score and check for exit signal.

        Args:
            trade_id: Trade identifier
            current_close: Current close price
            prev_close: Previous close price
            current_energy: Current bar's energy

        Returns:
            Tuple of (should_exit, current_score)
        """
        if trade_id not in self.active_trades:
            return False, 0.0

        trade = self.active_trades[trade_id]

        # Calculate bar contribution to score
        delta_c = abs(current_close - prev_close)
        bar_score = delta_c * current_energy / trade.entry_energy

        # Direction-adjusted score
        pnl_direction = np.sign(current_close - trade.entry_price) * trade.direction
        if pnl_direction >= 0:
            trade.cumulative_score += bar_score
        else:
            trade.cumulative_score -= bar_score * 0.5  # Penalize adverse moves less

        trade.cumulative_score = max(0, trade.cumulative_score)  # Floor at zero
        trade.max_score = max(trade.max_score, trade.cumulative_score)
        trade.bars_held += 1

        # Exit condition: score peaked and now declining past threshold
        if trade.max_score > 0:
            score_ratio = trade.cumulative_score / trade.max_score

            # Local maximum detection + threshold check
            if score_ratio < self.exit_threshold and trade.bars_held > 3:
                return True, trade.cumulative_score

        return False, trade.cumulative_score

    def close_trade(self, trade_id: str) -> Optional[TradeMetrics]:
        """Remove trade and return final metrics."""
        return self.active_trades.pop(trade_id, None)

    def get_trade_metrics(self, trade_id: str) -> Optional[TradeMetrics]:
        """Get current metrics for a trade."""
        return self.active_trades.get(trade_id)


# =============================================================================
# DAMPED HARMONIC OSCILLATOR MODEL - MICROSTRUCTURE REGIME
# =============================================================================
#
# Market as a Damped Harmonic Oscillator:
#   m*x'' + c*x' + k*x = F(t)
#
# Where:
#   m = "Mass" = Liquidity Depth (resistance to price change)
#   c = "Damping" = Friction/Viscosity (energy dissipation rate)
#   k = "Spring constant" = Mean reversion strength
#   F = "Force" = Order Flow (buy/sell pressure)
#
# Regimes:
#   - Thin market (Low Mass): Little force → big acceleration → BREAKOUT
#   - Thick market (High Mass): Force absorbed → MEAN REVERSION
#
# Level 2 (DOM) tells us what RESISTS future moves.
# Volume tells us what already HAPPENED.
# =============================================================================


def compute_liquidity_mass(high, low, close, volume, lookback: int = 20):
    """
    Compute "Mass" from liquidity depth proxy.

    Without DOM data, estimate liquidity from:
    - Price impact = |ΔPrice| / Volume (Kyle's Lambda)
    - Higher lambda = thinner market = lower mass

    Mass = inverse of price impact (thick = high mass, thin = low mass)

    Returns:
        Array of mass values (higher = more liquidity = harder to move)
    """
    high = pd.Series(high)
    low = pd.Series(low)
    close = pd.Series(close)
    volume = pd.Series(volume)

    # Price change magnitude
    price_change = close.diff().abs()

    # Kyle's Lambda: price impact per unit volume
    # Lower volume with same price move = thinner market
    volume_safe = volume.replace(0, np.nan).fillna(volume.mean())
    lambda_kyle = price_change / (volume_safe + 1e-10)

    # Smooth with rolling window
    lambda_smooth = lambda_kyle.rolling(lookback, min_periods=1).mean()

    # Mass = inverse of lambda (thick market = high mass)
    # Normalize by rolling mean to make adaptive
    lambda_mean = lambda_smooth.rolling(lookback * 2, min_periods=1).mean()
    mass = lambda_mean / (lambda_smooth + 1e-10)

    return mass.fillna(1.0).clip(0.1, 10.0).values


def compute_order_flow_force(open_, high, low, close, volume):
    """
    Compute "Force" from order flow imbalance.

    Without tick data, estimate from:
    - Buying pressure = (Close - Low) / (High - Low) * Volume
    - Selling pressure = (High - Close) / (High - Low) * Volume
    - Net Force = Buy pressure - Sell pressure

    Returns:
        Array of force values (positive = buy pressure, negative = sell pressure)
    """
    high = pd.Series(high)
    low = pd.Series(low)
    close = pd.Series(close)
    volume = pd.Series(volume)

    # Range
    range_ = high - low + 1e-10

    # Buy/Sell pressure (money flow approximation)
    buy_pressure = ((close - low) / range_) * volume
    sell_pressure = ((high - close) / range_) * volume

    # Net force
    force = buy_pressure - sell_pressure

    return force.fillna(0.0).values


def compute_acceleration(force, mass):
    """
    Newton's Second Law: F = ma → a = F/m

    Low mass + high force = high acceleration (breakout potential)
    High mass + same force = low acceleration (absorption/range)

    Returns:
        Array of acceleration values
    """
    force = pd.Series(force)
    mass = pd.Series(mass)

    acceleration = force / (mass + 1e-10)

    return acceleration.fillna(0.0).values


def classify_regime_adaptive(
    symc_series: pd.Series,
    underdamped_pct: float = 33.0,
    overdamped_pct: float = 67.0,
    lookback: int = 100
) -> np.ndarray:
    """
    ADAPTIVE regime classification - NO FIXED THRESHOLDS.

    Uses rolling percentiles of the SymC distribution to classify regimes.
    Each third of the distribution gets equal representation.

    This is critical because:
    - Fixed thresholds (0.8, 1.2) cause 52%+ to fall in "critical"
    - Different instruments have different SymC distributions
    - Market conditions shift the distribution over time

    VIRTUAL MODE: This is for logging/analysis only, NOT for gating entries.
    The system explores freely - regime is used for reward shaping.

    Args:
        symc_series: SymC ratio series
        underdamped_pct: Percentile below which = underdamped (default 33rd)
        overdamped_pct: Percentile above which = overdamped (default 67th)
        lookback: Rolling window for percentile calculation

    Returns:
        Array of regime labels
    """
    symc = symc_series.values
    n = len(symc)
    regimes = np.full(n, 'critical', dtype=object)

    for i in range(n):
        # Use available history up to lookback
        start = max(0, i - lookback + 1)
        window = symc[start:i+1]

        if len(window) < 5:
            # Not enough data - use current value relative to 1.0
            if symc[i] < 0.9:
                regimes[i] = 'underdamped'
            elif symc[i] > 1.1:
                regimes[i] = 'overdamped'
            continue

        # Adaptive thresholds from data distribution
        low_thresh = np.percentile(window, underdamped_pct)
        high_thresh = np.percentile(window, overdamped_pct)

        # Classify based on where current value falls
        if symc[i] < low_thresh:
            regimes[i] = 'underdamped'
        elif symc[i] > high_thresh:
            regimes[i] = 'overdamped'
        # else: 'critical' (middle third)

    return regimes


def compute_symc_ratio(high, low, close, volume, lookback: int = 20):
    """
    Compute SymC Ratio (χ) - The governing metric for market stability.

    χ = γ / (2ω) ≈ Liquidity Replenishment Rate / Order Arrival Frequency

    Regimes (ADAPTIVE, not fixed):
    - Underdamped (bottom 33%): Volatile, overshooting - MOMENTUM works
    - Overdamped (top 33%): Sluggish, mean-reverting - SNIPER works
    - Critical (middle 33%): Transitional - both can work

    Proxy calculation:
    - γ (damping) ≈ volume absorption rate (how fast liquidity absorbs orders)
    - ω (natural frequency) ≈ volatility (how fast price oscillates)
    """
    high = pd.Series(high)
    low = pd.Series(low)
    close = pd.Series(close)
    volume = pd.Series(volume)

    # Natural frequency proxy: volatility (price oscillation rate)
    returns = close.pct_change()
    omega = returns.rolling(lookback, min_periods=1).std().abs() + 1e-10

    # Damping proxy: volume absorption (volume / price_change)
    price_change = close.diff().abs() + 1e-10
    gamma = volume / price_change
    gamma_norm = gamma / gamma.rolling(lookback * 2, min_periods=1).mean()

    # SymC ratio: γ / (2ω) - normalized
    symc = gamma_norm / (2 * omega * 100)  # Scaled for interpretability

    # Clip to reasonable range
    symc = symc.fillna(1.0).clip(0.1, 5.0)

    return symc.values


def compute_oscillator_state(high, low, close, volume, lookback: int = 20):
    """
    Compute full Damped Harmonic Oscillator state with SymC regime.

    PHYSICS CORRECTION: Mass ≠ Volume!
    - Mass = Liquidity Depth (resistance to price change) via Kyle's Lambda
    - Force = Order Flow (buy/sell pressure)
    - Volume is the RESULT of force, not the mass

    Returns dict with:
    - mass: Liquidity-based market inertia (NOT volume)
    - force: Order flow pressure
    - acceleration: F/m (breakout potential)
    - velocity: Rate of price change
    - displacement: Distance from equilibrium
    - symc: SymC ratio for regime classification
    - regime: 'underdamped' (momentum), 'overdamped' (mean-revert), 'critical'
    """
    high = pd.Series(high)
    low = pd.Series(low)
    close = pd.Series(close)
    volume = pd.Series(volume)

    # Core components - CORRECT PHYSICS
    mass = compute_liquidity_mass(high, low, close, volume, lookback)
    force = compute_order_flow_force(high.values, high.values, low.values, close.values, volume.values)
    acceleration = compute_acceleration(force, mass)

    # Velocity = rate of price change
    velocity = close.pct_change().fillna(0.0).values

    # Displacement = distance from equilibrium (rolling mean)
    equilibrium = close.rolling(lookback, min_periods=1).mean()
    displacement = ((close - equilibrium) / equilibrium).fillna(0.0).values

    # SymC ratio for proper regime classification
    symc = compute_symc_ratio(high.values, low.values, close.values, volume.values, lookback)
    symc_series = pd.Series(symc)

    # ADAPTIVE regime classification - NO FIXED THRESHOLDS
    # Uses percentiles of the actual data distribution
    # This adapts to different instruments and market conditions
    regime = classify_regime_adaptive(symc_series)

    return {
        'mass': mass,
        'force': force,
        'acceleration': acceleration,
        'velocity': velocity,
        'displacement': displacement,
        'symc': symc,
        'regime': regime
    }


# =============================================================================
# BACKTESTING.PY INTEGRATION
# =============================================================================

def compute_body_ratio_indicator(open_, high, low, close, epsilon: float = 1e-10):
    """Body ratio indicator for backtesting.py."""
    open_ = pd.Series(open_)
    high = pd.Series(high)
    low = pd.Series(low)
    close = pd.Series(close)

    body = np.abs(close - open_)
    range_ = high - low + epsilon
    return (body / range_).clip(0.0, 1.0).values


def compute_energy_v7(open_, high, low, close, volume, span: int = 10, epsilon: float = 1e-10):
    """Energy indicator (v7.0 formulation) for backtesting.py."""
    body_ratio = compute_body_ratio_indicator(open_, high, low, close, epsilon)
    body_ratio = pd.Series(body_ratio)
    volume = pd.Series(volume)

    vol_ewma = volume.ewm(span=span).mean()
    vol_normalized = vol_ewma / (vol_ewma.rolling(100).mean() + epsilon)

    energy = (body_ratio ** 2) * vol_normalized
    return energy.fillna(0.0).clip(lower=0.0).values


def compute_damping_v7(high, low, epsilon: float = 1e-10):
    """Damping indicator (v7.0 formulation) for backtesting.py."""
    high = pd.Series(high)
    low = pd.Series(low)

    current_range = high - low
    prev_range = current_range.shift(1) + epsilon

    damping = current_range / prev_range
    # Clip to reasonable range [0.01, 10] to avoid extreme outliers
    return damping.fillna(1.0).clip(lower=0.01, upper=10.0).values


def compute_entropy_v7(volume, lookback: int = 20, epsilon: float = 1e-10):
    """Entropy indicator (v7.0 CoV formulation) for backtesting.py."""
    volume = pd.Series(volume)

    vol_std = volume.rolling(lookback).std()
    vol_mean = volume.rolling(lookback).mean() + epsilon

    entropy = vol_std / vol_mean
    return entropy.fillna(0.0).clip(lower=0.0).values


def compute_dominant_period_single(returns_window):
    """
    Compute dominant period for a single window using autocorrelation.
    Helper function for rolling application.
    """
    n = len(returns_window)
    if n < 10:
        return max(5, n // 4)

    # Search range: 2 bars to half the window length
    max_lag = min(n // 2, n - 2)
    if max_lag < 3:
        return max(5, n // 4)

    # Find lag with highest absolute autocorrelation
    best_lag = 2
    best_corr = 0

    returns_arr = np.array(returns_window)
    for lag in range(2, max_lag):
        if lag >= len(returns_arr):
            break
        # Manual autocorrelation calculation
        x1 = returns_arr[:-lag]
        x2 = returns_arr[lag:]
        if len(x1) < 3 or np.std(x1) == 0 or np.std(x2) == 0:
            continue
        corr = np.corrcoef(x1, x2)[0, 1]
        if not np.isnan(corr) and abs(corr) > best_corr:
            best_corr = abs(corr)
            best_lag = lag

    return best_lag


def compute_dominant_period(close):
    """
    Compute dominant period using autocorrelation - fully adaptive.
    Returns a single dominant period for the entire series.
    Used for initial lookback window sizing.
    """
    close = pd.Series(close)
    returns = close.pct_change().dropna()
    n = len(returns)

    if n < 20:
        return max(5, n // 4)

    # Search range: 2 bars to half the data length
    max_lag = n // 2

    # Find lag with highest absolute autocorrelation
    best_lag = 2
    best_corr = 0

    for lag in range(2, max_lag):
        corr = returns.autocorr(lag=lag)
        if not np.isnan(corr) and abs(corr) > best_corr:
            best_corr = abs(corr)
            best_lag = lag

    return best_lag


def compute_rolling_dominant_period(close, base_window: int = None):
    """
    Compute ROLLING dominant period - adapts per bar.

    Computes local dominant period at each point in time using
    a rolling window of returns. This adapts to:
    - Different instruments
    - Different timeframes
    - Changing market regimes

    Returns array of dominant periods (one per bar).
    """
    close = pd.Series(close)
    returns = close.pct_change()
    n = len(close)

    # Use initial dominant period estimate for base window if not provided
    if base_window is None:
        base_window = compute_dominant_period(close)
        base_window = max(20, min(base_window * 2, n // 4))  # Scale up for stability

    periods = np.full(n, base_window, dtype=float)

    # Compute rolling dominant period
    for i in range(base_window, n):
        window_returns = returns.iloc[max(0, i - base_window):i].dropna()
        if len(window_returns) >= 10:
            periods[i] = compute_dominant_period_single(window_returns.values)

    return periods


def compute_direction_probability(close, lookback: int):
    """
    Compute probability of direction continuation - fully adaptive.

    Uses the consistency of price movement direction over the rolling period.
    P(direction) = (bars moving in current direction) / lookback

    Returns values [0, 1] where:
    - 1.0 = all bars moved in same direction (strong trend)
    - 0.5 = random walk
    - 0.0 = all bars moved opposite (reversal pattern)
    """
    close = pd.Series(close)
    price_change = close.diff()

    # Current direction: +1 if last move was up, -1 if down
    current_dir = np.sign(price_change)

    # Rolling count of bars moving in same direction as current
    same_dir = (current_dir == current_dir.shift(1)).astype(float)

    # Direction consistency probability
    p_direction = same_dir.rolling(lookback, min_periods=lookback).mean()

    return p_direction.fillna(0.5).values


def compute_energy_probability(energy, lookback: int):
    """
    Compute probability of significant energy release - fully adaptive.

    Uses percentile rank of current energy in rolling distribution.
    P(energy) = percentile_rank(current_energy) in rolling window

    Returns values [0, 1] where:
    - 1.0 = energy at maximum of rolling period
    - 0.5 = energy at median
    - 0.0 = energy at minimum
    """
    energy = pd.Series(energy)

    # Rolling percentile rank
    def percentile_rank(x):
        if len(x) < 2:
            return 0.5
        rank = (x.iloc[:-1] < x.iloc[-1]).sum()
        return rank / (len(x) - 1)

    p_energy = energy.rolling(lookback, min_periods=lookback).apply(
        percentile_rank, raw=False
    )

    return p_energy.fillna(0.5).values


def compute_composite_probability(close, energy, damping):
    """
    Compute composite probability of direction AND potential energy release.

    Composite_P = P(direction) × P(energy)

    This is fully adaptive - uses dominant period computed from price data.
    No fixed values anywhere.

    Returns:
        Tuple of (composite_prob, direction_prob, energy_prob)
        All values in [0, 1]
    """
    close = pd.Series(close)
    energy = pd.Series(energy)

    # Compute dominant period from price (fully adaptive)
    dominant_period = compute_dominant_period(close)

    # Direction probability over dominant period
    p_direction = compute_direction_probability(close, dominant_period)

    # Energy probability over dominant period
    p_energy = compute_energy_probability(energy, dominant_period)

    # Composite probability (joint probability)
    composite = p_direction * p_energy

    return composite, p_direction, p_energy


def compute_agent_signal(energy, damping, close, high=None, low=None, volume=None):
    """
    Compute agent activation signal using MICROSTRUCTURE REGIME.

    DAMPED HARMONIC OSCILLATOR MODEL:
    - Mass = Liquidity Depth (resistance to price change)
    - Force = Order Flow (buy/sell pressure)
    - Acceleration = F/m (breakout potential)

    BERSERKER = BREAKOUT REGIME (Low Mass):
    - Thin market (low liquidity) → easy to move
    - High energy + high acceleration
    - Explosive momentum capture

    SNIPER = RANGE REGIME (High Mass):
    - Thick market (high liquidity) → absorbs force
    - High direction consistency (laminar flow)
    - Precision mean-reversion or continuation

    Returns:
        Array with values:
        - 2: Berserker (breakout regime, high acceleration)
        - 1: Sniper (range regime, laminar flow)
        - 0: No agent (stay flat)
    """
    energy = pd.Series(energy).reset_index(drop=True)
    damping = pd.Series(damping).reset_index(drop=True)
    close = pd.Series(close).reset_index(drop=True)
    n = len(energy)

    # Compute oscillator state if OHLCV provided (microstructure regime)
    use_oscillator = high is not None and low is not None and volume is not None
    if use_oscillator:
        high = pd.Series(high).reset_index(drop=True)
        low = pd.Series(low).reset_index(drop=True)
        volume = pd.Series(volume).reset_index(drop=True)
        oscillator = compute_oscillator_state(high.values, low.values, close.values, volume.values)
        mass = pd.Series(oscillator['mass'])
        acceleration = pd.Series(oscillator['acceleration'])

    # Compute ROLLING dominant period - adapts per bar
    rolling_periods = compute_rolling_dominant_period(close)

    # Pre-compute direction scores for all bars
    direction_scores = np.zeros(n)
    for i in range(1, n):
        T = int(rolling_periods[i])
        if T < 5 or i < T:
            continue

        start_idx = max(0, i - T)
        local_close = close.iloc[start_idx:i + 1]
        price_changes = local_close.diff().dropna()

        if len(price_changes) < 2:
            continue

        curr_dir = np.sign(price_changes.iloc[-1])
        if curr_dir == 0:
            continue

        # Laminar flow = ratio of bars moving in current direction
        same_dir_ratio = (np.sign(price_changes) == curr_dir).sum() / len(price_changes)

        # Consecutive momentum
        consecutive = 0
        for j in range(len(price_changes) - 1, -1, -1):
            if np.sign(price_changes.iloc[j]) == curr_dir:
                consecutive += 1
            else:
                break

        # Direction score combines flow ratio + consecutive momentum
        direction_scores[i] = (same_dir_ratio + consecutive / T) / 2.0

    direction_scores = pd.Series(direction_scores)

    # Initialize signals
    signals = np.zeros(n)

    # Compute signals bar-by-bar with MEDIAN-BASED thresholds
    for i in range(n):
        T = int(rolling_periods[i])
        if T < 5 or i < T:
            continue

        start_idx = max(0, i - T)

        local_energy = energy.iloc[start_idx:i + 1]
        local_damping = damping.iloc[start_idx:i + 1]
        local_direction = direction_scores.iloc[start_idx:i + 1]

        if len(local_energy) < 5:
            continue

        # Current values
        curr_energy = energy.iloc[i]
        curr_damping = damping.iloc[i]
        curr_direction = direction_scores.iloc[i]

        # GAUSSIAN DISTRIBUTION THRESHOLDS - mean + k*std
        # k derived from local kurtosis (no fixed values)
        energy_mean = local_energy.mean()
        energy_std = local_energy.std()
        damping_mean = local_damping.mean()
        damping_std = local_damping.std()
        direction_mean = local_direction.mean()
        direction_std = local_direction.std()

        # Adaptive k based on local kurtosis (heavier tails = lower k)
        # This makes thresholds adapt to the distribution shape
        energy_kurtosis = max(0.1, local_energy.kurtosis() if len(local_energy) > 3 else 0)
        k_energy = 1.0 / (1.0 + energy_kurtosis / 3.0)  # Normalized by mesokurtic baseline

        damping_kurtosis = max(0.1, local_damping.kurtosis() if len(local_damping) > 3 else 0)
        k_damping = 1.0 / (1.0 + damping_kurtosis / 3.0)

        direction_kurtosis = max(0.1, local_direction.kurtosis() if len(local_direction) > 3 else 0)
        k_direction = 1.0 / (1.0 + direction_kurtosis / 3.0)

        # Berserker thresholds: mean + k*std for upper, mean - k*std for lower
        energy_upper = energy_mean + k_energy * energy_std
        damping_lower = damping_mean - k_damping * damping_std
        direction_upper = direction_mean + k_direction * direction_std
        damping_upper = damping_mean + k_damping * damping_std

        # MICROSTRUCTURE REGIME CONDITIONS (if oscillator available)
        if use_oscillator:
            local_mass = mass.iloc[start_idx:i + 1]
            local_accel = acceleration.iloc[start_idx:i + 1]
            curr_mass = mass.iloc[i]
            curr_accel = acceleration.iloc[i]

            mass_mean = local_mass.mean()
            accel_mean = local_accel.mean()
            accel_std = local_accel.std()

            # Low mass = breakout regime, High mass = range regime
            is_breakout_regime = curr_mass < mass_mean
            is_range_regime = curr_mass >= mass_mean

            # High acceleration = strong force in thin market
            high_acceleration = curr_accel > accel_mean + accel_std

            # Berserker: BREAKOUT regime + high energy + high acceleration
            berserker_cond = (
                is_breakout_regime and
                curr_energy > energy_upper and
                high_acceleration
            )

            # Sniper: RANGE regime + laminar flow (high direction consistency)
            sniper_cond = (
                is_range_regime and
                curr_direction > direction_upper and
                curr_energy > energy_mean
            )
        else:
            # Fallback: original Gaussian-based conditions
            berserker_cond = (
                curr_energy > energy_upper and
                curr_direction > direction_upper and
                curr_damping < damping_lower
            )

            sniper_cond = (
                curr_energy > energy_mean and
                curr_direction > direction_mean and
                damping_lower < curr_damping < damping_upper
            )

        if berserker_cond:
            signals[i] = 2
        elif sniper_cond:
            signals[i] = 1

    return signals


# =============================================================================
# ASYMMETRICAL REWARD FUNCTION - PHYSICS-BASED
# =============================================================================
#
# R_t = (PnL_net × α) - (σ_price × λ) - (Duration × δ)
#
# Where:
#   - PnL_net: Includes Spread + Commission + Slippage (true cost)
#   - α (Asymmetry): Losses punished 3x harder than gains
#   - λ (Volatility penalty): High σ = high friction = lower reward
#   - δ (Time decay): Stagnating trades penalized
# =============================================================================


def compute_asymmetric_reward(
    pnl_gross: float,
    spread_cost: float,
    commission: float,
    slippage: float,
    volatility: float,
    duration_bars: int,
    alpha_loss: float = 3.0,
    alpha_gain: float = 1.0,
    lambda_vol: float = 0.1,
    delta_time: float = 0.01
) -> float:
    """
    Asymmetrical Reward Function for RL training.

    Forces agent to seek "High Quality, Low Drag" trades.
    Cures the Commission Death Spiral.

    R = (PnL_net × α) - (σ × λ) - (Duration × δ)

    Args:
        pnl_gross: Raw PnL before costs
        spread_cost: Spread cost (entry + exit)
        commission: Commission cost
        slippage: Estimated slippage
        volatility: Current price volatility (σ)
        duration_bars: How long trade was held
        alpha_loss: Asymmetry multiplier for losses (default 3.0)
        alpha_gain: Asymmetry multiplier for gains (default 1.0)
        lambda_vol: Volatility penalty weight
        delta_time: Time decay penalty weight

    Returns:
        Asymmetric reward value
    """
    # Net PnL = Gross - ALL friction costs
    pnl_net = pnl_gross - spread_cost - commission - slippage

    # Asymmetry: punish losses 3x harder than rewarding gains
    if pnl_net < 0:
        alpha = alpha_loss
    else:
        alpha = alpha_gain

    # Base reward (asymmetric)
    reward = pnl_net * alpha

    # Volatility penalty (high friction environment)
    vol_penalty = volatility * lambda_vol

    # Time decay (stagnating trades penalized)
    time_penalty = duration_bars * delta_time

    # Final reward
    return reward - vol_penalty - time_penalty


def detect_frozen_market(symc_history: pd.Series, threshold: float = 2.0, lookback: int = 10) -> bool:
    """
    Detect "Frozen" market condition - extreme overdamping before shock.

    When χ >> 1.2 (extreme overdamping), the market is "Silent before the Storm."
    This precedes volatility explosions and regime shifts.

    Returns True if market is in "frozen" danger zone.
    """
    if len(symc_history) < lookback:
        return False

    recent = symc_history.iloc[-lookback:]
    avg_symc = recent.mean()
    increasing = (recent.diff().dropna() > 0).sum() / len(recent.diff().dropna())

    # Frozen = high SymC AND increasing (liquidity flooding in, volatility suppressed)
    return avg_symc > threshold and increasing > 0.6


def compute_trade_quality_score(
    pnl_net: float,
    spread_pips: float,
    safety_margin: float = 2.0
) -> float:
    """
    Quality score for Shadow Trading validation.

    A Shadow trade only counts as a "Win" if it clears the spread
    by a safety margin (e.g., 2.0 pips).

    Returns:
        Quality score [0, 1] where 1 = high quality trade
    """
    if pnl_net <= 0:
        return 0.0

    # How many pips above the safety threshold?
    excess = pnl_net - (spread_pips * safety_margin)

    if excess <= 0:
        return 0.0  # Didn't clear safety margin

    # Score based on excess over threshold
    return min(1.0, excess / (spread_pips * 5))  # Max score at 5x safety margin


# =============================================================================
# PERFORMANCE METRICS - PHYSICS-BASED
# =============================================================================

def calculate_omega_ratio(returns: pd.Series, threshold: float = 0.0) -> float:
    """
    OMEGA RATIO - The "Thermodynamic/State-Space" Choice.

    Partitions the state space of returns:
    - Area above threshold = Positive Energy/Work
    - Area below threshold = Negative Energy/Waste

    Captures all higher moments (skew, kurtosis) without assuming distribution.
    Ideal for RL reward functions - produces stable agents.

    Formula: Ω(L) = ∫[L,∞](1-F(x))dx / ∫[-∞,L]F(x)dx
    """
    returns = pd.Series(returns).dropna()
    if len(returns) < 2:
        return 1.0

    gains = returns[returns > threshold].sum()
    losses = abs(returns[returns < threshold].sum())

    if losses == 0:
        return float('inf') if gains > 0 else 1.0

    return gains / losses


def calculate_stutzer_index(returns: pd.Series, benchmark: float = 0.0, max_iter: int = 100) -> float:
    """
    STUTZER INDEX - The "Statistical Mechanics" Choice.

    Based on Large Deviation Theory from statistical physics.
    Treats equity curve as a particle trajectory.
    Calculates exponential decay rate of probability of underperformance.

    Maximizes "probability current" away from ruin state.
    Naturally penalizes skewness and kurtosis (fat tails).

    Formula: I_p = max_θ>0 (-log(1/T * Σ exp(-θ(R_p - R_b))))
    """
    returns = pd.Series(returns).dropna()
    if len(returns) < 2:
        return 0.0

    excess = returns - benchmark
    T = len(excess)

    # Optimize θ to maximize the decay rate
    best_stutzer = 0.0
    best_theta = 0.0

    # Grid search for optimal theta (Lagrange multiplier)
    for theta in np.linspace(0.01, 10.0, max_iter):
        try:
            # Compute moment generating function
            mgf = np.mean(np.exp(-theta * excess))
            if mgf > 0:
                stutzer = -np.log(mgf)
                if stutzer > best_stutzer:
                    best_stutzer = stutzer
                    best_theta = theta
        except (OverflowError, RuntimeWarning):
            continue

    return best_stutzer


def calculate_rachev_ratio(returns: pd.Series, alpha: float = 0.05, beta: float = 0.05) -> float:
    """
    RACHEV RATIO - The "Econophysics/Power Law" Choice.

    Designed for Lévy Stable / Power Law distributions (fat tails).
    Abandons standard deviation (meaningless in power-law distributions).
    Measures ratio of Right Tail (extreme gains) to Left Tail (extreme losses).

    Handles Black Swan events that Kelly ignores.
    Ideal for breakout trading, volatility shocks, Hurst Exponent systems.

    Formula: CVaR_α(Gains) / CVaR_β(Losses)
    Where CVaR = Conditional Value at Risk (Expected Tail Loss/Gain)
    """
    returns = pd.Series(returns).dropna()
    if len(returns) < 10:
        return 1.0

    # CVaR for gains (right tail) - expected value of top α percentile
    gain_threshold = returns.quantile(1 - alpha)
    gains_tail = returns[returns >= gain_threshold]
    cvar_gains = gains_tail.mean() if len(gains_tail) > 0 else 0

    # CVaR for losses (left tail) - expected value of bottom β percentile
    loss_threshold = returns.quantile(beta)
    losses_tail = returns[returns <= loss_threshold]
    cvar_losses = abs(losses_tail.mean()) if len(losses_tail) > 0 else 0

    if cvar_losses == 0:
        return float('inf') if cvar_gains > 0 else 1.0

    return cvar_gains / cvar_losses


def calculate_all_physics_metrics(returns: pd.Series) -> dict:
    """
    Calculate all physics-based performance metrics.

    Returns dict with:
    - omega: Thermodynamic state-space ratio
    - stutzer: Statistical mechanics path stability
    - rachev: Power law tail ratio
    """
    returns = pd.Series(returns).dropna()

    return {
        'omega': calculate_omega_ratio(returns),
        'stutzer': calculate_stutzer_index(returns),
        'rachev': calculate_rachev_ratio(returns),
    }


def calculate_z_factor(returns: pd.Series, benchmark: float = 0.0) -> float:
    """
    Calculate Z-factor: (mean - benchmark) / std * sqrt(n)

    Measures statistical significance of edge.
    Target: > 2.5
    """
    n = len(returns)
    if n < 2:
        return 0.0

    mean_ret = returns.mean()
    std_ret = returns.std()

    if std_ret == 0:
        return float('inf') if mean_ret > benchmark else 0.0

    return (mean_ret - benchmark) / std_ret * np.sqrt(n)


def calculate_energy_captured(
    trade_returns: pd.Series,
    energy_at_entry: pd.Series,
    total_energy: float
) -> float:
    """
    Calculate percentage of energy captured by profitable trades.

    Target: > 65%
    """
    if total_energy == 0:
        return 0.0

    profitable_mask = trade_returns > 0
    energy_captured = energy_at_entry[profitable_mask].sum()

    return (energy_captured / total_energy) * 100


def calculate_mfe_captured(
    actual_pnl: pd.Series,
    mfe: pd.Series
) -> float:
    """
    Calculate percentage of Maximum Favorable Excursion captured.

    MFE Captured = actual_pnl / mfe for each trade
    Target: > 60%
    """
    valid_mask = mfe > 0
    if valid_mask.sum() == 0:
        return 0.0

    capture_ratios = actual_pnl[valid_mask] / mfe[valid_mask]
    return capture_ratios.clip(0, 1).mean() * 100


def validate_theorem_targets(
    omega: float,
    z_factor: float,
    energy_pct: float,
    mfe_pct: float
) -> Dict[str, bool]:
    """
    Validate performance against theorem targets.

    Targets:
    - Omega Ratio > 2.7
    - Z-Factor > 2.5
    - Energy Captured > 65%
    - MFE Captured > 60%
    """
    return {
        'omega_passed': omega > 2.7,
        'z_factor_passed': z_factor > 2.5,
        'energy_captured_passed': energy_pct > 65.0,
        'mfe_captured_passed': mfe_pct > 60.0,
        'all_passed': omega > 2.7 and z_factor > 2.5 and energy_pct > 65.0 and mfe_pct > 60.0
    }


# =============================================================================
# PRACTICAL PHYSICS MEASURES - REPLACING HURST EXPONENT
# =============================================================================
#
# Hurst Exponent requires 500+ bars for stability → impractical for trading.
# These measures work on 30-200 bars and provide actionable regime info:
#
# 1. FRACTAL DIMENSION (FD): The "Roughness" Gauge
#    - FD < 1.25 = Trend (smooth path, momentum works)
#    - FD > 1.50 = Chop (rough path, mean-reversion works)
#    - FD ≈ 1.50 = Random Walk
#
# 2. SAMPLE ENTROPY: The "Disorder" Monitor
#    - Falling = Market organizing → Breakout coming
#    - Spiking = Shock in progress
#    - Stable = Status quo
#
# 3. CENTER OF MASS: The "Balance" Point
#    - Volume-weighted price = TRUE conviction level
#    - Fake moves have CoM lagging price
#    - Genuine moves have CoM leading price
#
# 4. FTLE (Finite Time Lyapunov Exponent): The "Chaos" Detector
#    - High FTLE = Chaos/Bifurcation (regime about to flip)
#    - Low FTLE = Stable dynamics
#    - Works on 50-100 bar windows
# =============================================================================


def compute_fractal_dimension_katz(close, window: int = 50):
    """
    Katz Fractal Dimension - measures path complexity.

    FD = log10(n) / (log10(n) + log10(d/L))

    Where:
    - n = number of points
    - L = total path length (sum of |price changes|)
    - d = diameter (max distance from first point)

    Interpretation:
    - FD < 1.25: Trending (smooth path) → MOMENTUM works
    - FD > 1.50: Choppy (rough path) → MEAN REVERSION works
    - FD ≈ 1.50: Random walk

    Works on 30-100 bars (much better than Hurst's 500+).
    """
    close = pd.Series(close)
    n = len(close)

    if n < window:
        return np.full(n, 1.5)  # Neutral

    fd = np.full(n, 1.5)

    for i in range(window, n):
        local = close.iloc[i - window:i].values
        num_points = len(local)

        # Total path length L = sum of absolute price changes
        path_length = np.sum(np.abs(np.diff(local)))

        if path_length == 0:
            fd[i] = 1.5
            continue

        # Diameter d = max distance from first point
        diameter = np.max(np.abs(local - local[0]))

        if diameter == 0:
            fd[i] = 1.5
            continue

        # Katz FD formula
        log_n = np.log10(num_points - 1)
        log_d_L = np.log10(diameter / path_length)

        if log_n + log_d_L != 0:
            fd[i] = log_n / (log_n + log_d_L)
        else:
            fd[i] = 1.5

        # Clip to valid range [1.0, 2.0]
        fd[i] = np.clip(fd[i], 1.0, 2.0)

    return fd


def compute_fractal_dimension_higuchi(close, k_max: int = 10, window: int = 100):
    """
    Higuchi Fractal Dimension - more robust than Katz.

    Measures self-similarity at different scales.
    Uses the slope of log(L(k)) vs log(1/k) where L(k) is curve length at scale k.

    Interpretation:
    - FD < 1.25: Strong trend (low complexity)
    - FD > 1.50: Choppy/mean-reverting (high complexity)
    - FD ≈ 1.50: Random walk

    Works on 50-200 bars. More stable than Katz.
    """
    close = pd.Series(close)
    n = len(close)

    if n < window:
        return np.full(n, 1.5)

    fd = np.full(n, 1.5)

    for i in range(window, n):
        local = close.iloc[i - window:i].values
        N = len(local)

        # Ensure k_max is valid
        k_max_valid = min(k_max, N // 4)
        if k_max_valid < 2:
            fd[i] = 1.5
            continue

        # Compute L(k) for different k values
        lk = []
        ks = []

        for k in range(1, k_max_valid + 1):
            Lm_values = []

            for m in range(1, k + 1):
                # Indices for this subsequence
                indices = np.arange(m - 1, N, k)
                if len(indices) < 2:
                    continue

                # Subsequence values
                subseq = local[indices]

                # Length of this subseries
                a = len(indices) - 1
                if a == 0:
                    continue

                norm_factor = (N - 1) / (a * k)
                Lm = np.sum(np.abs(np.diff(subseq))) * norm_factor
                Lm_values.append(Lm)

            if len(Lm_values) > 0:
                lk.append(np.mean(Lm_values))
                ks.append(k)

        if len(ks) < 3:
            fd[i] = 1.5
            continue

        # Linear regression of log(L(k)) vs log(1/k)
        log_k = np.log(1.0 / np.array(ks))
        log_lk = np.log(np.array(lk) + 1e-10)

        # Slope = Fractal Dimension
        try:
            slope, _ = np.polyfit(log_k, log_lk, 1)
            fd[i] = np.clip(slope, 1.0, 2.0)
        except:
            fd[i] = 1.5

    return fd


def compute_sample_entropy(close, m: int = 2, r_mult: float = 0.2, window: int = 100):
    """
    Sample Entropy - measures predictability/regularity.

    SampEn = -log(A/B)

    Where:
    - B = number of template matches for patterns of length m
    - A = number of template matches for patterns of length m+1
    - r = tolerance (typically 0.2 * std)

    Interpretation:
    - FALLING entropy = Market organizing → Breakout imminent
    - SPIKING entropy = Shock in progress → Chaos
    - STABLE entropy = Status quo → Range-bound

    Works on 50-200 bars. Great for detecting pre-breakout compression.
    """
    close = pd.Series(close)
    n = len(close)

    if n < window:
        return np.full(n, 0.0)

    sampen = np.full(n, 0.0)

    for i in range(window, n):
        local = close.iloc[i - window:i].values
        N = len(local)

        # Tolerance based on local std
        std = np.std(local)
        if std == 0:
            sampen[i] = 0.0
            continue

        r = r_mult * std

        # Count template matches
        def count_matches(template_len):
            count = 0
            for j in range(N - template_len):
                for k in range(j + 1, N - template_len):
                    # Check if templates match within tolerance
                    template_j = local[j:j + template_len]
                    template_k = local[k:k + template_len]
                    if np.max(np.abs(template_j - template_k)) < r:
                        count += 1
            return count

        B = count_matches(m)
        A = count_matches(m + 1)

        if B == 0 or A == 0:
            sampen[i] = 0.0
        else:
            sampen[i] = -np.log(A / B)

    return sampen


def compute_sample_entropy_fast(close, m: int = 2, r_mult: float = 0.2, window: int = 100):
    """
    Fast Sample Entropy - vectorized approximation.

    Uses correlation integral approximation for speed.
    Suitable for real-time trading applications.
    """
    close = pd.Series(close)
    n = len(close)

    if n < window:
        return np.full(n, 0.0)

    sampen = np.full(n, 0.0)

    for i in range(window, n):
        local = close.iloc[i - window:i].values
        N = len(local)

        std = np.std(local)
        if std == 0:
            continue

        r = r_mult * std

        # Create embedded vectors
        def embed(x, m):
            return np.array([x[j:j + m] for j in range(len(x) - m + 1)])

        # Embedded sequences
        X_m = embed(local, m)
        X_m1 = embed(local, m + 1)

        # Count matches using vectorized distance
        def count_matches_fast(X):
            n_templates = len(X)
            count = 0
            for j in range(n_templates):
                # Chebyshev distance to all other templates
                distances = np.max(np.abs(X - X[j]), axis=1)
                # Count matches (excluding self)
                matches = np.sum(distances < r) - 1
                count += matches
            return count / 2  # Divide by 2 to avoid double counting

        B = count_matches_fast(X_m)
        A = count_matches_fast(X_m1)

        if B > 0 and A > 0:
            sampen[i] = -np.log(A / B)
        elif B > 0:
            sampen[i] = np.log(B)  # Maximum entropy estimate
        else:
            sampen[i] = 0.0

    return sampen


def compute_center_of_mass(close, volume, window: int = 20):
    """
    Volume-Weighted Center of Mass - true conviction level.

    CoM = Σ(price × volume) / Σ(volume)

    This is the volume-weighted average price (VWAP-like) over a window.
    Reveals where the REAL money is positioned.

    Interpretation:
    - Price > CoM + threshold: Overextended UP (potential reversal)
    - Price < CoM - threshold: Overextended DOWN (potential reversal)
    - Price ≈ CoM: Fair value, genuine move in progress

    Works on 10-50 bars. Essential for detecting fake breakouts.
    """
    close = pd.Series(close)
    volume = pd.Series(volume)
    n = len(close)

    # Handle zero volume
    volume = volume.replace(0, np.nan).fillna(volume.mean())

    # Rolling center of mass
    pv = close * volume
    com = pv.rolling(window, min_periods=1).sum() / volume.rolling(window, min_periods=1).sum()

    return com.fillna(close).values


def compute_com_divergence(close, volume, window: int = 20):
    """
    Center of Mass Divergence - distance from conviction level.

    Measures how far price has deviated from volume-weighted center.
    Normalized by rolling volatility for comparability.

    Interpretation:
    - Positive: Price above CoM (bullish conviction or overextended)
    - Negative: Price below CoM (bearish conviction or oversold)
    - Near zero: Price at fair value

    Use with trend direction to detect genuine vs fake moves.
    """
    close = pd.Series(close)
    volume = pd.Series(volume)

    com = compute_center_of_mass(close.values, volume.values, window)
    com = pd.Series(com, index=close.index)

    # Raw divergence
    divergence = close - com

    # Normalize by rolling volatility
    volatility = close.rolling(window, min_periods=1).std() + 1e-10
    normalized = divergence / volatility

    return normalized.fillna(0.0).values


def compute_ftle(close, window: int = 50, tau: int = 1):
    """
    Finite Time Lyapunov Exponent - chaos/stability detector.

    FTLE measures exponential divergence of nearby trajectories.
    High FTLE = chaotic dynamics, regime about to flip.
    Low FTLE = stable dynamics, current regime persists.

    Uses return space reconstruction for practical calculation.

    Interpretation:
    - FTLE > 0.1: Chaos/Bifurcation → Be cautious, regime shift likely
    - FTLE < 0.05: Stable → Current trend/range likely to continue
    - FTLE spiking: Transition in progress

    Works on 50-100 bars. Superior to Hurst for detecting regime changes.
    """
    close = pd.Series(close)
    n = len(close)

    if n < window + tau:
        return np.full(n, 0.0)

    ftle = np.full(n, 0.0)

    for i in range(window, n):
        local = close.iloc[i - window:i].values

        # Create phase space embedding
        returns = np.diff(local)
        if len(returns) < 2:
            continue

        # Find nearest neighbors and compute divergence
        divergences = []

        for j in range(len(returns) - tau):
            # Current point
            x_j = returns[j]

            # Find nearest neighbor (excluding self and immediate neighbors)
            distances = np.abs(returns - x_j)
            distances[max(0, j - 2):min(len(returns), j + 3)] = np.inf

            if np.all(np.isinf(distances)):
                continue

            k = np.argmin(distances)
            d0 = distances[k]

            if d0 == 0:
                continue

            # Compute divergence after tau steps
            if j + tau < len(returns) and k + tau < len(returns):
                dt = np.abs(returns[j + tau] - returns[k + tau])
                if dt > 0 and d0 > 0:
                    # Lyapunov exponent = (1/tau) * log(dt/d0)
                    divergences.append(np.log(dt / d0) / tau)

        if len(divergences) > 0:
            ftle[i] = np.mean(divergences)
            # Clip to reasonable range
            ftle[i] = np.clip(ftle[i], -1.0, 1.0)

    return ftle


def compute_ftle_fast(close, window: int = 50):
    """
    Fast FTLE approximation using variance ratios.

    Uses the ratio of short-term to long-term variance as a proxy.
    Much faster than full FTLE calculation.
    """
    close = pd.Series(close)
    n = len(close)

    if n < window:
        return np.full(n, 0.0)

    returns = close.pct_change().fillna(0.0)

    # Short-term variance
    short_var = returns.rolling(window // 4, min_periods=1).var()

    # Long-term variance
    long_var = returns.rolling(window, min_periods=1).var()

    # FTLE proxy = log ratio of variances
    # High ratio = short-term volatility spiking = regime change
    ratio = short_var / (long_var + 1e-10)
    ftle_proxy = np.log(ratio + 1e-10)

    return ftle_proxy.fillna(0.0).clip(-2.0, 2.0).values


def compute_market_state_features(high, low, close, volume, window: int = 50):
    """
    Compute all practical physics measures as a feature vector.

    Returns dict with:
    - fractal_dim: Fractal Dimension (trend vs chop)
    - sample_entropy: Sample Entropy (order vs chaos)
    - center_of_mass: Volume-weighted price level
    - com_divergence: Distance from CoM (normalized)
    - ftle: Chaos/stability indicator

    These replace Hurst Exponent for short-timeframe trading.
    """
    close = pd.Series(close)
    volume = pd.Series(volume)

    return {
        'fractal_dim': compute_fractal_dimension_katz(close.values, window),
        'sample_entropy': compute_sample_entropy_fast(close.values, window=window),
        'center_of_mass': compute_center_of_mass(close.values, volume.values, window // 2),
        'com_divergence': compute_com_divergence(close.values, volume.values, window // 2),
        'ftle': compute_ftle_fast(close.values, window),
    }


def classify_market_regime_physics(fractal_dim, sample_entropy, ftle, symc):
    """
    Classify market regime using physics measures.

    Combines:
    - Fractal Dimension (trend purity)
    - Sample Entropy (predictability)
    - FTLE (stability)
    - SymC Ratio (damping)

    Returns regime label and confidence score.
    """
    fd = pd.Series(fractal_dim)
    se = pd.Series(sample_entropy)
    ftle_s = pd.Series(ftle)
    symc_s = pd.Series(symc)
    n = len(fd)

    regimes = []
    confidences = []

    for i in range(n):
        fd_val = fd.iloc[i]
        se_val = se.iloc[i]
        ftle_val = ftle_s.iloc[i]
        symc_val = symc_s.iloc[i]

        # Trend regime: Low FD + Low FTLE + Underdamped
        trend_score = (1.5 - fd_val) + (0.1 - abs(ftle_val)) + (1.0 - symc_val)

        # Range regime: High FD + Low FTLE + Overdamped
        range_score = (fd_val - 1.5) + (0.1 - abs(ftle_val)) + (symc_val - 1.0)

        # Chaos regime: High FTLE + High Entropy
        chaos_score = abs(ftle_val) + se_val

        scores = {'trend': trend_score, 'range': range_score, 'chaos': chaos_score}
        best = max(scores, key=scores.get)
        confidence = scores[best] / (sum(abs(v) for v in scores.values()) + 1e-10)

        regimes.append(best)
        confidences.append(min(1.0, max(0.0, confidence)))

    return regimes, confidences


# =============================================================================
# RL FEATURE EXTRACTION - REGIME-AWARE STATE SPACE
# =============================================================================
#
# For Distributional RL (QR-DQN) or PPO:
# - Context Vector: HMM regime states (one-hot)
# - Risk Vector: Kurtosis, VPIN proxy, Entropy
# - Momentum Vector: Fractal Dimension, FTLE
# - Flow Vector: Order flow, CoM divergence
# =============================================================================


def compute_kurtosis_rolling(returns, window: int = 50):
    """
    Rolling excess kurtosis for tail risk measurement.

    High kurtosis = fat tails = standard risk measures fail.
    Feed this to QR-DQN to learn distribution-aware policies.
    """
    returns = pd.Series(returns)
    kurt = returns.rolling(window, min_periods=window // 2).apply(
        lambda x: pd.Series(x).kurtosis(), raw=False
    )
    return kurt.fillna(0.0).clip(-10, 10).values


def compute_vpin_proxy(close, volume, window: int = 50):
    """
    VPIN (Volume-Synchronized Probability of Informed Trading) proxy.

    Without tick data, estimate toxicity from:
    - Volume imbalance in directional buckets
    - Higher VPIN = more informed trading = crash risk

    Returns [0, 1] where 1 = maximum toxicity.
    """
    close = pd.Series(close)
    volume = pd.Series(volume)

    # Classify volume as buy/sell
    direction = np.sign(close.diff())

    # Rolling buy/sell volume
    buy_vol = (volume * (direction > 0).astype(float)).rolling(window, min_periods=1).sum()
    sell_vol = (volume * (direction < 0).astype(float)).rolling(window, min_periods=1).sum()
    total_vol = buy_vol + sell_vol + 1e-10

    # VPIN proxy = |buy - sell| / total
    vpin = np.abs(buy_vol - sell_vol) / total_vol

    return vpin.fillna(0.0).clip(0, 1).values


def compute_rl_state_vector(
    high, low, close, volume,
    window: int = 50
) -> Dict[str, np.ndarray]:
    """
    Compute complete RL state vector for Regime-Conditioned Network.

    Returns dict with feature groups:
    - context: HMM-like regime indicators
    - risk: Kurtosis, VPIN, Entropy (for QR-DQN distribution learning)
    - momentum: Fractal Dimension, FTLE (for trend detection)
    - flow: Order flow, CoM divergence (for conviction)
    - oscillator: Mass, Force, Acceleration, SymC (for microstructure)

    Architecture should use multi-head design:
    - Head A (Range Logic): Active when FD > 1.5
    - Head B (Trend Logic): Active when FD < 1.25
    - Head C (Crisis Logic): Active when FTLE > 0.1 or VPIN > 0.7
    """
    high = pd.Series(high)
    low = pd.Series(low)
    close = pd.Series(close)
    volume = pd.Series(volume)

    returns = close.pct_change().fillna(0.0)

    # Physics measures
    fd = compute_fractal_dimension_katz(close.values, window)
    se = compute_sample_entropy_fast(close.values, window=window)
    com = compute_center_of_mass(close.values, volume.values, window // 2)
    com_div = compute_com_divergence(close.values, volume.values, window // 2)
    ftle = compute_ftle_fast(close.values, window)

    # Oscillator state
    oscillator = compute_oscillator_state(
        high.values, low.values, close.values, volume.values, window // 2
    )

    # Risk measures
    kurtosis = compute_kurtosis_rolling(returns.values, window)
    vpin = compute_vpin_proxy(close.values, volume.values, window)

    # SymC for regime
    symc = oscillator['symc']

    # Regime classification (one-hot encoding)
    regimes, confidences = classify_market_regime_physics(fd, se, ftle, symc)

    regime_onehot = np.zeros((len(close), 3))
    for i, regime in enumerate(regimes):
        if regime == 'trend':
            regime_onehot[i, 0] = 1
        elif regime == 'range':
            regime_onehot[i, 1] = 1
        else:  # chaos
            regime_onehot[i, 2] = 1

    return {
        # Context (for gating)
        'regime_trend': regime_onehot[:, 0],
        'regime_range': regime_onehot[:, 1],
        'regime_chaos': regime_onehot[:, 2],
        'regime_confidence': np.array(confidences),

        # Risk (for distribution learning)
        'kurtosis': kurtosis,
        'vpin': vpin,
        'entropy': se,

        # Momentum (for trend detection)
        'fractal_dim': fd,
        'ftle': ftle,

        # Flow (for conviction)
        'order_flow': oscillator['force'],
        'com_divergence': com_div,

        # Oscillator (for microstructure)
        'mass': oscillator['mass'],
        'acceleration': oscillator['acceleration'],
        'velocity': oscillator['velocity'],
        'symc': symc,
    }


def compute_action_mask(
    state_vector: Dict[str, np.ndarray],
    idx: int,
    mode: str = "virtual"
) -> Dict[str, bool]:
    """
    Compute action mask based on regime state.

    VIRTUAL MODE: All actions allowed - explore freely to find alpha!
    LIVE MODE: Dynamic conditional locks based on physics.

    The mask is NEVER used to gate entries during exploration.
    In live trading, it protects capital during extreme conditions.

    Args:
        state_vector: Physics state vectors
        idx: Current bar index
        mode: "virtual" (no gates) or "live" (gates enforced)

    Returns dict of allowed actions.
    """
    # Default: all actions allowed
    mask = {
        'buy': True,
        'sell': True,
        'hold': True,
        'close': True,
    }

    # VIRTUAL MODE: No gates! Explore freely.
    if mode == "virtual":
        return mask

    # LIVE MODE: Dynamic conditional locks
    # These use ADAPTIVE thresholds based on recent distribution

    regime_chaos = state_vector['regime_chaos'][idx]
    vpin = state_vector['vpin'][idx]
    ftle = state_vector['ftle'][idx]
    symc = state_vector['symc'][idx]

    # Get adaptive thresholds from recent data
    lookback = min(idx + 1, 100)
    if lookback > 10:
        ftle_recent = state_vector['ftle'][max(0, idx-lookback):idx+1]
        vpin_recent = state_vector['vpin'][max(0, idx-lookback):idx+1]
        symc_recent = state_vector['symc'][max(0, idx-lookback):idx+1]

        # Adaptive: extreme = top 5% of recent distribution
        ftle_extreme = np.percentile(np.abs(ftle_recent), 95)
        vpin_extreme = np.percentile(vpin_recent, 95)
        symc_extreme = np.percentile(symc_recent, 95)
    else:
        # Fallback for insufficient data
        ftle_extreme = 0.1
        vpin_extreme = 0.7
        symc_extreme = 2.0

    # Crisis/Chaos: No new positions (only in extreme conditions)
    if regime_chaos > 0.7 or abs(ftle) > ftle_extreme:
        mask['buy'] = False
        mask['sell'] = False

    # High toxicity: No new positions
    if vpin > vpin_extreme:
        mask['buy'] = False
        mask['sell'] = False

    # Frozen market: Force close existing positions
    if symc > symc_extreme:
        mask['buy'] = False
        mask['sell'] = False

    return mask


# =============================================================================
# WELFORD'S ONLINE ALGORITHM - INCREMENTAL STATISTICS
# =============================================================================
#
# For real-time trading, recalculating statistics from scratch is too slow.
# Welford's algorithm computes mean, variance, and higher moments incrementally.
#
# Live metrics:
# - Sortino Ratio: Tracks running sum of squared negative returns
# - Ulcer Index: Tracks running sum of squared drawdowns from peak
# - Standard Deviation: Welford's classic algorithm
# - Downside Deviation: Welford's applied to negative returns only
# =============================================================================


@dataclass
class WelfordState:
    """State for Welford's online algorithm."""
    count: int = 0
    mean: float = 0.0
    M2: float = 0.0  # Sum of squared deviations
    min_val: float = float('inf')
    max_val: float = float('-inf')

    def update(self, x: float):
        """Update statistics with new value using Welford's algorithm."""
        self.count += 1
        delta = x - self.mean
        self.mean += delta / self.count
        delta2 = x - self.mean
        self.M2 += delta * delta2
        self.min_val = min(self.min_val, x)
        self.max_val = max(self.max_val, x)

    @property
    def variance(self) -> float:
        """Population variance."""
        if self.count < 2:
            return 0.0
        return self.M2 / self.count

    @property
    def sample_variance(self) -> float:
        """Sample variance (Bessel's correction)."""
        if self.count < 2:
            return 0.0
        return self.M2 / (self.count - 1)

    @property
    def std(self) -> float:
        """Population standard deviation."""
        return np.sqrt(self.variance)

    @property
    def sample_std(self) -> float:
        """Sample standard deviation."""
        return np.sqrt(self.sample_variance)


@dataclass
class OnlineDownsideState:
    """State for online downside deviation calculation."""
    count: int = 0
    negative_count: int = 0
    sum_sq_negative: float = 0.0
    threshold: float = 0.0

    def update(self, ret: float):
        """Update with new return value."""
        self.count += 1
        if ret < self.threshold:
            self.negative_count += 1
            self.sum_sq_negative += (ret - self.threshold) ** 2

    @property
    def downside_deviation(self) -> float:
        """Downside deviation (target = threshold)."""
        if self.count < 2:
            return 0.0
        return np.sqrt(self.sum_sq_negative / self.count)


@dataclass
class OnlineUlcerState:
    """State for online Ulcer Index calculation."""
    count: int = 0
    peak_equity: float = 0.0
    sum_sq_drawdown: float = 0.0

    def update(self, equity: float):
        """Update with new equity value."""
        self.count += 1
        self.peak_equity = max(self.peak_equity, equity)

        if self.peak_equity > 0:
            drawdown_pct = (self.peak_equity - equity) / self.peak_equity * 100
            self.sum_sq_drawdown += drawdown_pct ** 2

    @property
    def ulcer_index(self) -> float:
        """Ulcer Index = RMS of percentage drawdowns."""
        if self.count < 2:
            return 0.0
        return np.sqrt(self.sum_sq_drawdown / self.count)


@dataclass
class OnlineSortinoState:
    """State for online Sortino Ratio calculation."""
    return_state: WelfordState = None
    downside_state: OnlineDownsideState = None
    risk_free_rate: float = 0.0

    def __post_init__(self):
        if self.return_state is None:
            self.return_state = WelfordState()
        if self.downside_state is None:
            self.downside_state = OnlineDownsideState(threshold=self.risk_free_rate)

    def update(self, ret: float):
        """Update with new return value."""
        self.return_state.update(ret)
        self.downside_state.update(ret)

    @property
    def sortino_ratio(self) -> float:
        """Sortino Ratio = (mean - rf) / downside_deviation."""
        dd = self.downside_state.downside_deviation
        if dd == 0:
            return 0.0 if self.return_state.mean <= self.risk_free_rate else float('inf')
        return (self.return_state.mean - self.risk_free_rate) / dd


@dataclass
class OnlineOmegaState:
    """State for online Omega Ratio calculation."""
    sum_gains: float = 0.0
    sum_losses: float = 0.0
    count: int = 0
    threshold: float = 0.0

    def update(self, ret: float):
        """Update with new return value."""
        self.count += 1
        if ret > self.threshold:
            self.sum_gains += ret - self.threshold
        elif ret < self.threshold:
            self.sum_losses += self.threshold - ret

    @property
    def omega_ratio(self) -> float:
        """Omega Ratio = sum(gains) / sum(losses)."""
        if self.sum_losses == 0:
            return float('inf') if self.sum_gains > 0 else 1.0
        return self.sum_gains / self.sum_losses


class OnlineMetricsTracker:
    """
    Complete online metrics tracker for live trading.

    Tracks all performance metrics incrementally using Welford's algorithm.
    No need to recalculate from scratch on every bar.
    """

    def __init__(self, risk_free_rate: float = 0.0, initial_equity: float = 100000.0):
        """
        Initialize online metrics tracker.

        Args:
            risk_free_rate: Risk-free rate for Sortino/Sharpe
            initial_equity: Starting equity value
        """
        self.initial_equity = initial_equity
        self.current_equity = initial_equity
        self.risk_free_rate = risk_free_rate

        # Online states
        self.return_state = WelfordState()
        self.sortino_state = OnlineSortinoState(risk_free_rate=risk_free_rate)
        self.ulcer_state = OnlineUlcerState()
        self.omega_state = OnlineOmegaState()

        # Trade tracking
        self.trade_count = 0
        self.winning_trades = 0
        self.total_pnl = 0.0
        self.max_drawdown = 0.0
        self.peak_equity = initial_equity

    def update_return(self, ret: float):
        """Update with a new return value (percentage or decimal)."""
        self.return_state.update(ret)
        self.sortino_state.update(ret)
        self.omega_state.update(ret)

    def update_equity(self, equity: float):
        """Update with new equity value."""
        self.current_equity = equity
        self.ulcer_state.update(equity)

        # Update peak and drawdown
        self.peak_equity = max(self.peak_equity, equity)
        if self.peak_equity > 0:
            current_dd = (self.peak_equity - equity) / self.peak_equity
            self.max_drawdown = max(self.max_drawdown, current_dd)

    def record_trade(self, pnl: float):
        """Record a completed trade."""
        self.trade_count += 1
        self.total_pnl += pnl
        if pnl > 0:
            self.winning_trades += 1

        # Update equity
        self.current_equity += pnl
        self.update_equity(self.current_equity)

        # Update return (as percentage of equity before trade)
        if self.current_equity - pnl != 0:
            ret = pnl / (self.current_equity - pnl)
            self.update_return(ret)

    @property
    def sharpe_ratio(self) -> float:
        """Sharpe Ratio = (mean - rf) / std."""
        std = self.return_state.sample_std
        if std == 0:
            return 0.0 if self.return_state.mean <= self.risk_free_rate else float('inf')
        return (self.return_state.mean - self.risk_free_rate) / std

    @property
    def sortino_ratio(self) -> float:
        """Sortino Ratio from online state."""
        return self.sortino_state.sortino_ratio

    @property
    def omega_ratio(self) -> float:
        """Omega Ratio from online state."""
        return self.omega_state.omega_ratio

    @property
    def ulcer_index(self) -> float:
        """Ulcer Index from online state."""
        return self.ulcer_state.ulcer_index

    @property
    def ulcer_performance_index(self) -> float:
        """UPI = (total_return - rf) / ulcer_index."""
        ui = self.ulcer_index
        if ui == 0:
            return 0.0

        total_return = (self.current_equity / self.initial_equity - 1) * 100
        return (total_return - self.risk_free_rate * 100) / ui

    @property
    def win_rate(self) -> float:
        """Win rate as percentage."""
        if self.trade_count == 0:
            return 0.0
        return self.winning_trades / self.trade_count * 100

    @property
    def max_drawdown_pct(self) -> float:
        """Maximum drawdown as percentage."""
        return self.max_drawdown * 100

    @property
    def total_return_pct(self) -> float:
        """Total return as percentage."""
        return (self.current_equity / self.initial_equity - 1) * 100

    def get_metrics(self) -> Dict[str, float]:
        """Get all current metrics as a dict."""
        return {
            'total_return_pct': self.total_return_pct,
            'sharpe_ratio': self.sharpe_ratio,
            'sortino_ratio': self.sortino_ratio,
            'omega_ratio': self.omega_ratio,
            'ulcer_index': self.ulcer_index,
            'ulcer_performance_index': self.ulcer_performance_index,
            'max_drawdown_pct': self.max_drawdown_pct,
            'win_rate': self.win_rate,
            'trade_count': self.trade_count,
            'total_pnl': self.total_pnl,
            'current_equity': self.current_equity,
            'mean_return': self.return_state.mean,
            'return_std': self.return_state.sample_std,
        }

    def reset(self, initial_equity: float = None):
        """Reset all states."""
        if initial_equity is not None:
            self.initial_equity = initial_equity

        self.current_equity = self.initial_equity
        self.return_state = WelfordState()
        self.sortino_state = OnlineSortinoState(risk_free_rate=self.risk_free_rate)
        self.ulcer_state = OnlineUlcerState()
        self.omega_state = OnlineOmegaState()
        self.trade_count = 0
        self.winning_trades = 0
        self.total_pnl = 0.0
        self.max_drawdown = 0.0
        self.peak_equity = self.initial_equity


def compute_online_sortino(returns: List[float], risk_free_rate: float = 0.0) -> float:
    """
    Compute Sortino Ratio using online algorithm.

    This is a convenience function that processes a list of returns
    using the online algorithm. For true real-time use, use OnlineSortinoState.
    """
    state = OnlineSortinoState(risk_free_rate=risk_free_rate)
    for ret in returns:
        state.update(ret)
    return state.sortino_ratio


def compute_online_ulcer(equity_curve: List[float]) -> float:
    """
    Compute Ulcer Index using online algorithm.

    This is a convenience function that processes an equity curve
    using the online algorithm. For true real-time use, use OnlineUlcerState.
    """
    state = OnlineUlcerState()
    for equity in equity_curve:
        state.update(equity)
    return state.ulcer_index


def compute_online_omega(returns: List[float], threshold: float = 0.0) -> float:
    """
    Compute Omega Ratio using online algorithm.

    This is a convenience function that processes a list of returns
    using the online algorithm. For true real-time use, use OnlineOmegaState.
    """
    state = OnlineOmegaState(threshold=threshold)
    for ret in returns:
        state.update(ret)
    return state.omega_ratio


# =============================================================================
# WASSERSTEIN DISTANCE - OPTIMAL TRANSPORT FOR DISTRIBUTION COMPARISON
# =============================================================================
#
# The "Earth Mover's Distance" measures how much "work" is needed to transform
# one distribution into another. Superior to KL-divergence because:
#
# 1. Works with non-overlapping supports (KL = infinity in that case)
# 2. Captures geometry of distributions, not just overlap
# 3. Stable gradients for RL training (used in QR-DQN, WGAN)
# 4. Natural metric for regime shift detection
#
# Uses:
# - Regime Shift Detection: W(current_returns, historical_returns)
# - QR-DQN Training: Compare predicted vs actual return distributions
# - Risk Measurement: Distribution distance from benchmark
# =============================================================================


def wasserstein_distance_1d(p: np.ndarray, q: np.ndarray) -> float:
    """
    1D Wasserstein Distance (Earth Mover's Distance) between two samples.

    For 1D distributions, W_1 = integral of |F_p(x) - F_q(x)| dx
    which equals the L1 distance between sorted samples.

    This is the "work" needed to transform distribution p into q.

    Args:
        p: First sample array
        q: Second sample array

    Returns:
        Wasserstein-1 distance (always >= 0)
    """
    p = np.asarray(p).flatten()
    q = np.asarray(q).flatten()

    if len(p) == 0 or len(q) == 0:
        return 0.0

    # Sort both samples
    p_sorted = np.sort(p)
    q_sorted = np.sort(q)

    # If different lengths, interpolate to common grid
    n = max(len(p), len(q))

    if len(p) != n:
        p_sorted = np.interp(
            np.linspace(0, 1, n),
            np.linspace(0, 1, len(p)),
            p_sorted
        )
    if len(q) != n:
        q_sorted = np.interp(
            np.linspace(0, 1, n),
            np.linspace(0, 1, len(q)),
            q_sorted
        )

    # W_1 distance = mean of absolute differences of sorted values
    return np.mean(np.abs(p_sorted - q_sorted))


def wasserstein_distance_cdf(p: np.ndarray, q: np.ndarray, n_points: int = 100) -> float:
    """
    Wasserstein Distance via CDF comparison.

    More accurate for continuous distributions.
    Computes W_1 = integral |F_p(x) - F_q(x)| dx over common support.

    Args:
        p: First sample array
        q: Second sample array
        n_points: Number of points for CDF evaluation

    Returns:
        Wasserstein-1 distance
    """
    p = np.asarray(p).flatten()
    q = np.asarray(q).flatten()

    if len(p) == 0 or len(q) == 0:
        return 0.0

    # Common support
    x_min = min(p.min(), q.min())
    x_max = max(p.max(), q.max())

    if x_max == x_min:
        return 0.0

    x = np.linspace(x_min, x_max, n_points)

    # Empirical CDFs
    cdf_p = np.searchsorted(np.sort(p), x, side='right') / len(p)
    cdf_q = np.searchsorted(np.sort(q), x, side='right') / len(q)

    # Integrate |F_p - F_q|
    dx = (x_max - x_min) / (n_points - 1)
    return np.sum(np.abs(cdf_p - cdf_q)) * dx


def compute_regime_shift_wasserstein(
    returns: np.ndarray,
    window: int = 50,
    reference_window: int = 200
) -> np.ndarray:
    """
    Detect regime shifts using Wasserstein Distance.

    Compares recent return distribution (window) to historical (reference_window).
    High distance = regime has changed significantly.

    Interpretation:
    - W < 0.001: Same regime, stable
    - W in [0.001, 0.005]: Minor shift, monitor
    - W > 0.005: Major regime change, adapt strategy

    Args:
        returns: Array of returns
        window: Recent window for current distribution
        reference_window: Historical window for reference distribution

    Returns:
        Array of Wasserstein distances (one per bar after warmup)
    """
    returns = np.asarray(returns).flatten()
    n = len(returns)

    if n < window + reference_window:
        return np.zeros(n)

    distances = np.zeros(n)

    for i in range(window + reference_window, n):
        # Recent distribution
        recent = returns[i - window:i]

        # Historical distribution (before recent)
        historical = returns[i - window - reference_window:i - window]

        # Wasserstein distance
        distances[i] = wasserstein_distance_1d(recent, historical)

    return distances


def compute_distribution_stability(
    returns: np.ndarray,
    window: int = 50,
    lag: int = 10
) -> np.ndarray:
    """
    Measure distribution stability over time.

    Compares distribution at time t vs t-lag.
    Low stability = distribution is changing rapidly (regime transition).

    Returns stability score [0, 1] where 1 = perfectly stable.
    """
    returns = np.asarray(returns).flatten()
    n = len(returns)

    if n < window + lag:
        return np.ones(n)  # Assume stable if not enough data

    stability = np.ones(n)

    for i in range(window + lag, n):
        current = returns[i - window:i]
        lagged = returns[i - window - lag:i - lag]

        w_dist = wasserstein_distance_1d(current, lagged)

        # Convert to stability score (inverse, normalized)
        # Typical W for returns is 0.001-0.01
        stability[i] = 1.0 / (1.0 + w_dist * 100)

    return stability


def quantile_wasserstein(
    predicted_quantiles: np.ndarray,
    actual: float,
    tau: np.ndarray = None
) -> float:
    """
    Wasserstein Distance for QR-DQN evaluation.

    Compares predicted quantile distribution to actual return.
    Used to evaluate how well the quantile network predicted the distribution.

    Args:
        predicted_quantiles: Array of predicted quantile values
        actual: Actual realized return
        tau: Quantile levels (default: uniform from 0 to 1)

    Returns:
        Wasserstein distance from predicted distribution to actual
    """
    predicted_quantiles = np.asarray(predicted_quantiles).flatten()
    n = len(predicted_quantiles)

    if n == 0:
        return 0.0

    if tau is None:
        tau = np.linspace(0.5 / n, 1 - 0.5 / n, n)

    # Predicted CDF is step function at predicted quantiles
    # Actual is a point mass at 'actual'

    # Distance is weighted sum of |predicted_q - actual|
    # Weight by quantile spacing
    weights = np.diff(np.concatenate([[0], tau, [1]]))
    weights = (weights[:-1] + weights[1:]) / 2

    return np.sum(weights * np.abs(predicted_quantiles - actual))


def compute_tail_wasserstein(
    p: np.ndarray,
    q: np.ndarray,
    tail_percentile: float = 0.05
) -> Tuple[float, float]:
    """
    Wasserstein Distance focused on distribution tails.

    For fat-tailed distributions, the tails matter most.
    This computes W distance separately for left and right tails.

    Args:
        p: First sample
        q: Second sample
        tail_percentile: Percentile defining tail (default 5%)

    Returns:
        Tuple of (left_tail_W, right_tail_W)
    """
    p = np.asarray(p).flatten()
    q = np.asarray(q).flatten()

    if len(p) < 20 or len(q) < 20:
        return 0.0, 0.0

    # Left tail (worst returns)
    p_left_threshold = np.percentile(p, tail_percentile * 100)
    q_left_threshold = np.percentile(q, tail_percentile * 100)

    p_left = p[p <= p_left_threshold]
    q_left = q[q <= q_left_threshold]

    left_W = wasserstein_distance_1d(p_left, q_left) if len(p_left) > 0 and len(q_left) > 0 else 0.0

    # Right tail (best returns)
    p_right_threshold = np.percentile(p, (1 - tail_percentile) * 100)
    q_right_threshold = np.percentile(q, (1 - tail_percentile) * 100)

    p_right = p[p >= p_right_threshold]
    q_right = q[q >= q_right_threshold]

    right_W = wasserstein_distance_1d(p_right, q_right) if len(p_right) > 0 and len(q_right) > 0 else 0.0

    return left_W, right_W


class OnlineWassersteinTracker:
    """
    Online Wasserstein Distance tracker using reservoir sampling.

    Maintains two reservoirs (current and historical) and computes
    Wasserstein distance between them incrementally.

    Useful for real-time regime shift detection.
    """

    def __init__(
        self,
        current_size: int = 50,
        historical_size: int = 200,
        update_freq: int = 10
    ):
        """
        Initialize tracker.

        Args:
            current_size: Size of current distribution reservoir
            historical_size: Size of historical distribution reservoir
            update_freq: How often to update W distance
        """
        self.current_size = current_size
        self.historical_size = historical_size
        self.update_freq = update_freq

        self.current_reservoir: List[float] = []
        self.historical_reservoir: List[float] = []
        self.count = 0
        self.last_distance = 0.0

    def update(self, value: float) -> float:
        """
        Add new value and return current Wasserstein distance.

        Args:
            value: New return value

        Returns:
            Current Wasserstein distance (updated every update_freq)
        """
        self.count += 1

        # Update current reservoir (sliding window)
        self.current_reservoir.append(value)
        if len(self.current_reservoir) > self.current_size:
            # Move oldest to historical
            oldest = self.current_reservoir.pop(0)
            self._add_to_historical(oldest)

        # Compute distance periodically
        if self.count % self.update_freq == 0:
            if len(self.current_reservoir) >= self.current_size // 2 and \
               len(self.historical_reservoir) >= self.historical_size // 2:
                self.last_distance = wasserstein_distance_1d(
                    np.array(self.current_reservoir),
                    np.array(self.historical_reservoir)
                )

        return self.last_distance

    def _add_to_historical(self, value: float):
        """Add value to historical reservoir using reservoir sampling."""
        if len(self.historical_reservoir) < self.historical_size:
            self.historical_reservoir.append(value)
        else:
            # Reservoir sampling: replace random element with decreasing probability
            j = np.random.randint(0, self.count)
            if j < self.historical_size:
                self.historical_reservoir[j] = value

    @property
    def distance(self) -> float:
        """Current Wasserstein distance."""
        return self.last_distance

    @property
    def is_regime_shift(self) -> bool:
        """Check if current distance indicates regime shift."""
        return self.last_distance > 0.005  # Threshold for significant shift

    def reset(self):
        """Reset tracker."""
        self.current_reservoir = []
        self.historical_reservoir = []
        self.count = 0
        self.last_distance = 0.0


def compute_rl_distribution_features(
    returns: np.ndarray,
    window: int = 50,
    n_quantiles: int = 10
) -> Dict[str, np.ndarray]:
    """
    Compute distribution-aware features for QR-DQN.

    Returns features that capture the full return distribution,
    not just mean/variance. Essential for Distributional RL.

    Features:
    - quantiles: Return quantiles (for QR-DQN targets)
    - wasserstein: Regime shift indicator
    - tail_left_w: Left tail change (crash risk)
    - tail_right_w: Right tail change (opportunity)
    - stability: Distribution stability score
    """
    returns = np.asarray(returns).flatten()
    n = len(returns)

    # Initialize outputs
    quantiles = np.zeros((n, n_quantiles))
    wasserstein = np.zeros(n)
    tail_left = np.zeros(n)
    tail_right = np.zeros(n)
    stability = np.ones(n)

    tau = np.linspace(0.5 / n_quantiles, 1 - 0.5 / n_quantiles, n_quantiles)

    for i in range(window, n):
        local = returns[i - window:i]

        # Quantiles of current window
        quantiles[i] = np.percentile(local, tau * 100)

        # Wasserstein vs historical
        if i >= 2 * window:
            historical = returns[i - 2 * window:i - window]
            wasserstein[i] = wasserstein_distance_1d(local, historical)
            tail_left[i], tail_right[i] = compute_tail_wasserstein(local, historical)

        # Stability vs lagged
        if i >= window + 10:
            lagged = returns[i - window - 10:i - 10]
            w = wasserstein_distance_1d(local, lagged)
            stability[i] = 1.0 / (1.0 + w * 100)

    return {
        'quantiles': quantiles,
        'wasserstein': wasserstein,
        'tail_left_w': tail_left,
        'tail_right_w': tail_right,
        'stability': stability,
    }
