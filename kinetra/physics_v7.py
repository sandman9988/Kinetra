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


def compute_agent_signal(energy, damping, lookback: int = 50):
    """
    Compute agent activation signal using rolling quantiles.

    Returns:
        Array with values:
        - 2: Berserker active
        - 1: Sniper active
        - 0: No agent
    """
    energy = pd.Series(energy)
    damping = pd.Series(damping)

    # Use rolling quantiles for efficiency (O(n) instead of O(n²))
    e_q60 = energy.rolling(lookback, min_periods=lookback).quantile(0.60)
    e_q75 = energy.rolling(lookback, min_periods=lookback).quantile(0.75)
    d_q25 = damping.rolling(lookback, min_periods=lookback).quantile(0.25)
    d_q75 = damping.rolling(lookback, min_periods=lookback).quantile(0.75)

    # Berserker: energy > Q75 AND damping < Q25 (Underdamped, High Energy)
    berserker = (energy > e_q75) & (damping < d_q25)

    # Sniper: Q25 < damping < Q75 AND energy > Q60 (Critical, Moderate-High Energy)
    sniper = (damping > d_q25) & (damping < d_q75) & (energy > e_q60)

    # Combine signals (Berserker takes priority)
    signals = np.zeros(len(energy))
    signals[sniper] = 1
    signals[berserker] = 2

    return signals


# =============================================================================
# PERFORMANCE METRICS
# =============================================================================

def calculate_omega_ratio(returns: pd.Series, threshold: float = 0.0) -> float:
    """
    Calculate Omega ratio: Σ(gains above threshold) / Σ(losses below threshold)

    Target: > 2.7
    """
    gains = returns[returns > threshold].sum()
    losses = abs(returns[returns < threshold].sum())

    if losses == 0:
        return float('inf') if gains > 0 else 0.0

    return gains / losses


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
