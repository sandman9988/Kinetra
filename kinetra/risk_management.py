"""
Risk Management Module

Implements:
- Non-Linear Risk-of-Ruin (RoR) probability
- Composite Health Score (CHS) across multiple dimensions
- Dynamic position sizing based on risk gates
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd


@dataclass
class RiskMetrics:
    """Container for risk metrics."""

    risk_of_ruin: float
    position_size: float
    health_score: float
    kelly_fraction: float
    max_drawdown: float


class RiskManager:
    """
    Physics-aware risk management system.

    Key Features:
    - Non-linear Risk-of-Ruin calculation
    - Composite Health Score (CHS) across agents, risk, and market classes
    - Dynamic position sizing with circuit breakers
    """

    def __init__(
        self,
        max_risk_of_ruin: float = 0.10,
        min_health_score: float = 0.55,
        base_position_size: float = 0.02,
        lookback: int = 100,
    ):
        """
        Initialize risk manager.

        Args:
            max_risk_of_ruin: Maximum acceptable RoR (default: 10%)
            min_health_score: Minimum CHS to allow trading (default: 0.55)
            base_position_size: Base position size as fraction of equity (default: 2%)
            lookback: Rolling window for risk calculations (default: 100 bars)
        """
        self.max_risk_of_ruin = max_risk_of_ruin
        self.min_health_score = min_health_score
        self.base_position_size = base_position_size
        self.lookback = lookback

    def calculate_risk_of_ruin(
        self, current_equity: float, ruin_level: float, returns: pd.Series
    ) -> float:
        """
        Calculate non-linear risk-of-ruin probability.

        Formula: P(ruin) = exp(-2μ(X_t - L_t) / σ²)

        Where:
        - μ = expected return per trade
        - X_t = current equity
        - L_t = ruin level (e.g., 50% of starting capital)
        - σ² = variance of returns

        Args:
            current_equity: Current account equity
            ruin_level: Equity level that constitutes "ruin"
            returns: Historical returns series

        Returns:
            Probability of ruin (0 to 1)
        """
        # Calculate statistical parameters
        mu = returns.mean()
        sigma_squared = returns.var()

        # Distance to ruin
        distance = current_equity - ruin_level

        # Non-linear RoR formula
        if sigma_squared < 1e-10:
            # No volatility -> no risk if mu > 0, full risk if mu <= 0
            return 0.0 if mu > 0 else 1.0

        exponent = -2.0 * mu * distance / (sigma_squared + 1e-10)

        # Clamp exponent to prevent overflow
        exponent = np.clip(exponent, -50, 50)

        ror = np.exp(exponent)

        # Ensure probability is in [0, 1]
        ror = np.clip(ror, 0.0, 1.0)

        return float(ror)

    def calculate_agent_health_score(
        self, win_rate: float, avg_win_loss_ratio: float, omega_ratio: Optional[float] = None
    ) -> float:
        """
        Calculate health score for RL agents.

        CHS_agents = 0.4 * WinRate + 0.3 * (AvgWin/AvgLoss) + 0.3 * Omega

        Args:
            win_rate: Percentage of winning trades (0-1)
            avg_win_loss_ratio: Average win / average loss
            omega_ratio: Omega ratio (upside/downside)

        Returns:
            Agent health score (0-1)
        """
        # Normalize components to [0, 1]
        win_rate_norm = np.clip(win_rate, 0.0, 1.0)

        # Win/Loss ratio typically 1-3, normalize to [0, 1]
        wl_norm = np.clip(avg_win_loss_ratio / 3.0, 0.0, 1.0)

        # Omega ratio typically 1-5, normalize to [0, 1]
        if omega_ratio is not None:
            omega_norm = np.clip(omega_ratio / 5.0, 0.0, 1.0)
        else:
            omega_norm = 0.5  # Neutral if not provided

        # Weighted combination
        chs_agents = 0.4 * win_rate_norm + 0.3 * wl_norm + 0.3 * omega_norm

        return float(chs_agents)

    def calculate_risk_health_score(
        self,
        risk_of_ruin: float,
        max_drawdown: float,
        volatility: float,
        max_dd_threshold: float = 0.20,
        max_vol_threshold: float = 0.50,
    ) -> float:
        """
        Calculate health score for risk metrics.

        CHS_risk = 0.5 * (1 - RoR) + 0.3 * (1 - DD/DD_max) + 0.2 * (1 - Vol/Vol_max)

        Args:
            risk_of_ruin: Current RoR probability
            max_drawdown: Current drawdown (e.g., 0.15 = 15% DD)
            volatility: Current return volatility
            max_dd_threshold: Maximum acceptable drawdown
            max_vol_threshold: Maximum acceptable volatility

        Returns:
            Risk health score (0-1)
        """
        # Invert RoR (lower is better)
        ror_component = 1.0 - np.clip(risk_of_ruin, 0.0, 1.0)

        # Drawdown component (lower is better)
        dd_component = 1.0 - np.clip(max_drawdown / max_dd_threshold, 0.0, 1.0)

        # Volatility component (lower is better)
        vol_component = 1.0 - np.clip(volatility / max_vol_threshold, 0.0, 1.0)

        # Weighted combination
        chs_risk = 0.5 * ror_component + 0.3 * dd_component + 0.2 * vol_component

        return float(chs_risk)

    def calculate_class_health_score(
        self, regime_stability: float, energy_capture_pct: float, false_activation_rate: float
    ) -> float:
        """
        Calculate health score for market classification quality.

        CHS_class = 0.4 * RegimeStability + 0.4 * EnergyCaptured - 0.2 * FalseActivationRate

        Args:
            regime_stability: How often regime predictions are correct (0-1)
            energy_capture_pct: Percentage of available energy captured (0-1)
            false_activation_rate: Rate of trades in wrong regime (0-1)

        Returns:
            Classification health score (0-1)
        """
        stability_norm = np.clip(regime_stability, 0.0, 1.0)
        energy_norm = np.clip(energy_capture_pct, 0.0, 1.0)
        false_act_norm = np.clip(false_activation_rate, 0.0, 1.0)

        chs_class = 0.4 * stability_norm + 0.4 * energy_norm - 0.2 * false_act_norm

        # Ensure result is in [0, 1]
        chs_class = np.clip(chs_class, 0.0, 1.0)

        return float(chs_class)

    def composite_health_score(self, chs_agents: float, chs_risk: float, chs_class: float) -> float:
        """
        Calculate overall Composite Health Score.

        CHS = 0.4 * CHS_agents + 0.3 * CHS_risk + 0.3 * CHS_class

        Args:
            chs_agents: Agent health score
            chs_risk: Risk health score
            chs_class: Classification health score

        Returns:
            Overall composite health score (0-1)
        """
        chs = 0.4 * chs_agents + 0.3 * chs_risk + 0.3 * chs_class

        # Ensure result is in [0, 1]
        chs = np.clip(chs, 0.0, 1.0)

        return float(chs)

    def calculate_position_size(
        self,
        equity: float,
        risk_of_ruin: float,
        health_score: float,
        kelly_fraction: Optional[float] = None,
    ) -> float:
        """
        Calculate dynamic position size with risk gates.

        Args:
            equity: Current account equity
            risk_of_ruin: Current RoR probability
            health_score: Current CHS
            kelly_fraction: Optional Kelly criterion fraction

        Returns:
            Position size in currency units (0 if gates fail)
        """
        # Gate 1: Risk-of-Ruin check
        if risk_of_ruin > self.max_risk_of_ruin:
            return 0.0  # Circuit breaker

        # Gate 2: Health Score check
        if health_score < self.min_health_score:
            return 0.0  # Circuit breaker

        # Base position size
        position = equity * self.base_position_size

        # Scale down based on risk
        risk_scalar = 1.0 - (risk_of_ruin / self.max_risk_of_ruin)
        position *= risk_scalar

        # Scale by health score
        health_scalar = (health_score - self.min_health_score) / (1.0 - self.min_health_score)
        position *= health_scalar

        # Optional: Kelly criterion adjustment
        if kelly_fraction is not None:
            kelly_fraction = np.clip(kelly_fraction, 0.0, 0.25)  # Max 25% Kelly
            position *= kelly_fraction

        return float(position)

    def check_risk_gates(
        self, current_equity: float, ruin_level: float, returns: pd.Series, health_score: float
    ) -> Tuple[bool, str]:
        """
        Check if all risk gates pass.

        Args:
            current_equity: Current account equity
            ruin_level: Ruin threshold
            returns: Historical returns
            health_score: Current CHS

        Returns:
            (passed: bool, message: str)
        """
        # Calculate RoR
        ror = self.calculate_risk_of_ruin(current_equity, ruin_level, returns)

        # Check RoR gate
        if ror > self.max_risk_of_ruin:
            return False, f"Risk-of-Ruin too high: {ror:.2%} > {self.max_risk_of_ruin:.2%}"

        # Check health gate
        if health_score < self.min_health_score:
            return False, f"Health Score too low: {health_score:.2f} < {self.min_health_score:.2f}"

        return True, "All risk gates passed"


# Standalone functions for convenience
def calculate_risk_of_ruin(current_equity: float, ruin_level: float, returns: pd.Series) -> float:
    """Calculate non-linear risk-of-ruin probability."""
    manager = RiskManager()
    return manager.calculate_risk_of_ruin(current_equity, ruin_level, returns)


def composite_health_score(chs_agents: float, chs_risk: float, chs_class: float) -> float:
    """Calculate overall composite health score."""
    manager = RiskManager()
    return manager.composite_health_score(chs_agents, chs_risk, chs_class)


def compute_chs(
    energy_capture: Union[float, np.ndarray],
    omega: Union[float, np.ndarray],
    stability: Union[float, np.ndarray],
    alpha: float = 0.4,
    beta: float = 0.4,
    gamma: float = 0.2,
) -> Union[float, np.ndarray]:
    """
    Composite Health Score (CHS): \\( \\alpha \\cdot E + \\beta \\cdot \\Omega + \\gamma \\cdot S \\).
    Theorem: Halt if <0.90. Vectorized for batches; adaptive weights via percentiles if std >0.1.
    NaN shield: np.clip and np.isfinite check.
    """
    # Vectorized NaN/Inf shield
    energy_capture = np.nan_to_num(energy_capture, nan=0.0, posinf=1.0, neginf=0.0)
    omega = np.nan_to_num(omega, nan=1.0, posinf=5.0, neginf=1.0)
    stability = np.nan_to_num(stability, nan=0.5, posinf=1.0, neginf=0.0)

    if isinstance(energy_capture, (pd.Series, list)):
        scores = pd.DataFrame({"energy": energy_capture, "omega": omega, "stability": stability})
        if scores.std().max() > 0.1:
            total_std = scores.std().sum()
            alpha = scores["energy"].std() / total_std
            beta = scores["omega"].std() / total_std
            gamma = scores["stability"].std() / total_std
        weights = np.array([alpha, beta, gamma])
        chs = np.dot(scores.values, weights)
    else:
        scores = np.array([energy_capture, omega, stability])
        weights = np.array([alpha, beta, gamma])
        chs = np.dot(weights, scores)

    chs = np.clip(chs, 0, 1)  # Final shield
    assert np.isfinite(chs).all(), "Non-finite CHS values detected"
    # If p-value provided (extendable), assert p < 0.01
    # Example: if 'p_value' in locals(): assert p_value < 0.01, "Statistical significance failed"
    return chs


def compute_ror(
    mu: float, sigma: Union[float, np.ndarray], X: float, L: float = 0
) -> Union[float, np.ndarray]:
    """
    Non-Linear Risk-of-Ruin: \\( P(\\mathrm{ruin}) = \\exp\\left(-2\\mu(X-L)/\\sigma^2\\right) \\). Halt if >0.05. Vectorized; sigma clip to avoid div0.
    NaN shield: np.clip and np.isfinite.
    """
    # Vectorized NaN/Inf shield
    sigma = np.nan_to_num(sigma, nan=1e-8, posinf=np.inf, neginf=1e-8)
    sigma = np.clip(sigma, 1e-8, np.inf)

    exponent = -2 * mu * (X - L) / (sigma**2)
    exponent = np.clip(exponent, -50, 50)  # Prevent overflow
    ror = np.exp(exponent)
    ror = np.clip(ror, 0, 1)  # Finite, non-neg
    assert np.isfinite(ror).all(), "Non-finite RoR values detected"
    # If p-value provided, assert p < 0.01 (extendable)
    return ror
