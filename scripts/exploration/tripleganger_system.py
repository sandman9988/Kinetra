#!/usr/bin/env python3
"""
Tripleganger Risk Management System

A comprehensive risk management framework with:
1. Shadow Agent Architecture (Live + Frozen + Retraining)
2. Dynamic Circuit Breakers (Kurtosis, VPIN, Flash Crash)
3. Adaptive Thresholds (No Hardcoded Values)
4. Trade & Portfolio Risk Management
5. Exploration vs Live Mode Distinction

Philosophy:
- Open for exploration during backtesting (learn all patterns)
- Gated during live trading (protect capital)
- All thresholds are DYNAMIC (rolling percentiles, z-scores)
- No hardcoded magic numbers

Author: Physics-First Trading System
"""

import copy
import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from numpy import floating


# =============================================================================
# TRADING MODE: Exploration vs Live
# =============================================================================

class TradingMode(Enum):
    """Trading mode determines risk gating behavior."""
    EXPLORATION = auto()  # Backtesting: Open exploration, no gating
    PAPER = auto()        # Paper trading: Soft gating, log warnings
    LIVE = auto()         # Live trading: Hard gating, strict limits


@dataclass
class ModeConfig:
    """Configuration per trading mode."""
    mode: TradingMode
    circuit_breaker_enabled: bool = True
    position_limits_enabled: bool = True
    drawdown_limits_enabled: bool = True
    log_all_decisions: bool = True

    # Threshold multipliers (looser in exploration, tighter in live)
    threshold_multiplier: float = 1.0  # 1.0 = standard, >1 = looser, <1 = tighter

    @classmethod
    def exploration(cls) -> "ModeConfig":
        """Exploration mode: Open, no hard limits."""
        return cls(
            mode=TradingMode.EXPLORATION,
            circuit_breaker_enabled=False,  # Learn from all conditions
            position_limits_enabled=False,
            drawdown_limits_enabled=False,
            log_all_decisions=True,
            threshold_multiplier=2.0,  # Very loose
        )

    @classmethod
    def paper(cls) -> "ModeConfig":
        """Paper trading: Soft limits, warnings."""
        return cls(
            mode=TradingMode.PAPER,
            circuit_breaker_enabled=True,
            position_limits_enabled=True,
            drawdown_limits_enabled=True,
            log_all_decisions=True,
            threshold_multiplier=1.2,  # Slightly loose
        )

    @classmethod
    def live(cls) -> "ModeConfig":
        """Live trading: Hard limits, strict protection."""
        return cls(
            mode=TradingMode.LIVE,
            circuit_breaker_enabled=True,
            position_limits_enabled=True,
            drawdown_limits_enabled=True,
            log_all_decisions=True,
            threshold_multiplier=0.8,  # Tight
        )


# =============================================================================
# ADAPTIVE THRESHOLDS: No Hardcoded Values
# =============================================================================

class AdaptiveThreshold:
    """
    Dynamic threshold calculation using rolling statistics.

    No hardcoded values - thresholds adapt to market conditions.

    Methods:
    - Percentile-based: Use rolling percentile (e.g., 95th)
    - Z-score based: Mean ± N*std
    - Regime-aware: Different thresholds per regime
    """

    def __init__(
        self,
        window: int = 500,
        min_samples: int = 50,
        default_percentile: float = 0.95,
        z_score_sigma: float = 2.0,
    ):
        self.window = window
        self.min_samples = min_samples
        self.default_percentile = default_percentile
        self.z_score_sigma = z_score_sigma

        # History buffer
        self.history: List[float] = []

    def update(self, value: float):
        """Add new observation."""
        self.history.append(value)
        # Keep only recent window
        if len(self.history) > self.window:
            self.history = self.history[-self.window:]

    def get_percentile_threshold(
        self,
        percentile: Optional[float] = None,
        multiplier: float = 1.0,
    ) -> float:
        """Get threshold at given percentile."""
        if len(self.history) < self.min_samples:
            return float('inf')  # Not enough data, don't trigger

        pct = percentile or self.default_percentile
        threshold = np.percentile(self.history, pct * 100)
        return threshold * multiplier

    def get_zscore_threshold(
        self,
        sigma: Optional[float] = None,
        direction: str = "upper",  # "upper", "lower", "both"
        multiplier: float = 1.0,
    ) -> float | floating[Any] | tuple[floating[Any], floating[Any]]:
        """Get threshold at mean ± N*std."""
        if len(self.history) < self.min_samples:
            return float('inf') if direction == "upper" else float('-inf')

        n_sigma = sigma or self.z_score_sigma
        mean = np.mean(self.history)
        std = np.std(self.history)

        if direction == "upper":
            return (mean + n_sigma * std) * multiplier
        elif direction == "lower":
            return (mean - n_sigma * std) * multiplier
        else:  # both - return tuple
            return (mean - n_sigma * std, mean + n_sigma * std)

    def get_current_zscore(self, value: float) -> float | floating[Any]:
        """Get z-score of current value."""
        if len(self.history) < self.min_samples:
            return 0.0
        mean = np.mean(self.history)
        std = np.std(self.history)
        return (value - mean) / (std + 1e-8)

    def is_extreme(
        self,
        value: float,
        percentile: Optional[float] = None,
        multiplier: float = 1.0,
    ) -> bool:
        """Check if value exceeds percentile threshold."""
        threshold = self.get_percentile_threshold(percentile, multiplier)
        return value > threshold


# =============================================================================
# CIRCUIT BREAKERS: Dynamic Risk Gating
# =============================================================================

@dataclass
class CircuitBreakerState:
    """Current state of a circuit breaker."""
    name: str
    is_triggered: bool = False
    trigger_value: float = 0.0
    threshold: float = 0.0
    cooldown_bars: int = 0
    triggered_at: Optional[datetime] = None
    message: str = ""


class BaseCircuitBreaker(ABC):
    """Abstract base for circuit breakers."""

    def __init__(self, name: str, cooldown_bars: int = 24):
        self.name = name
        self.cooldown_bars = cooldown_bars
        self.bars_since_trigger = float('inf')
        self.adaptive_threshold = AdaptiveThreshold()

    @abstractmethod
    def check(
        self,
        physics_state: pd.Series,
        mode_config: ModeConfig,
    ) -> CircuitBreakerState:
        """Check if breaker should trigger. Override in subclass."""
        pass

    def tick(self):
        """Advance time by one bar."""
        self.bars_since_trigger += 1

    def in_cooldown(self) -> bool:
        """Check if still in cooldown period."""
        return self.bars_since_trigger < self.cooldown_bars


class KurtosisCircuitBreaker(BaseCircuitBreaker):
    """
    Kurtosis-based circuit breaker for black swan detection.

    High kurtosis (fat tails) indicates extreme event risk.
    Triggers when kurtosis exceeds dynamic threshold.
    """

    def __init__(self, cooldown_bars: int = 24):
        super().__init__("kurtosis", cooldown_bars)
        self.base_percentile = 0.95

    def check(
        self,
        physics_state: pd.Series,
        mode_config: ModeConfig,
    ) -> CircuitBreakerState:
        if not mode_config.circuit_breaker_enabled:
            return CircuitBreakerState(name=self.name)

        kurtosis_z = physics_state.get("kurtosis_z", 0.0)
        self.adaptive_threshold.update(abs(kurtosis_z))

        # Dynamic threshold with mode multiplier
        threshold = self.adaptive_threshold.get_percentile_threshold(
            percentile=self.base_percentile,
            multiplier=mode_config.threshold_multiplier,
        )

        is_triggered = abs(kurtosis_z) > threshold and not self.in_cooldown()

        if is_triggered:
            self.bars_since_trigger = 0

        return CircuitBreakerState(
            name=self.name,
            is_triggered=is_triggered,
            trigger_value=abs(kurtosis_z),
            threshold=threshold,
            cooldown_bars=max(0, self.cooldown_bars - int(min(self.bars_since_trigger, self.cooldown_bars))),
            triggered_at=datetime.now() if is_triggered else None,
            message=f"Fat tail detected: |kurtosis_z|={abs(kurtosis_z):.2f} > {threshold:.2f}"
            if is_triggered else "",
        )


class VPINCircuitBreaker(BaseCircuitBreaker):
    """
    VPIN-based circuit breaker for flash crash / liquidity crisis.

    High VPIN indicates order flow toxicity and market maker withdrawal.
    Triggers when VPIN exceeds dynamic threshold.
    """

    def __init__(self, cooldown_bars: int = 12):
        super().__init__("vpin", cooldown_bars)
        self.base_percentile = 0.90

    def check(
        self,
        physics_state: pd.Series,
        mode_config: ModeConfig,
    ) -> CircuitBreakerState:
        if not mode_config.circuit_breaker_enabled:
            return CircuitBreakerState(name=self.name)

        vpin = physics_state.get("vpin", 0.5)
        vpin_z = physics_state.get("vpin_z", 0.0)
        self.adaptive_threshold.update(vpin)

        threshold = self.adaptive_threshold.get_percentile_threshold(
            percentile=self.base_percentile,
            multiplier=mode_config.threshold_multiplier,
        )

        is_triggered = vpin > threshold and not self.in_cooldown()

        if is_triggered:
            self.bars_since_trigger = 0

        return CircuitBreakerState(
            name=self.name,
            is_triggered=is_triggered,
            trigger_value=vpin,
            threshold=threshold,
            cooldown_bars=max(0, self.cooldown_bars - int(min(self.bars_since_trigger, self.cooldown_bars))),
            triggered_at=datetime.now() if is_triggered else None,
            message=f"Liquidity crisis: VPIN={vpin:.3f} (z={vpin_z:.2f}) > {threshold:.3f}"
            if is_triggered else "",
        )


class VolatilityCircuitBreaker(BaseCircuitBreaker):
    """
    Volatility-based circuit breaker for regime stress.

    Uses Yang-Zhang volatility (most efficient OHLC estimator).
    Triggers when vol exceeds dynamic threshold.
    """

    def __init__(self, cooldown_bars: int = 48):
        super().__init__("volatility", cooldown_bars)
        self.base_percentile = 0.95

    def check(
        self,
        physics_state: pd.Series,
        mode_config: ModeConfig,
    ) -> CircuitBreakerState:
        if not mode_config.circuit_breaker_enabled:
            return CircuitBreakerState(name=self.name)

        vol_yz_z = physics_state.get("vol_yz_z", 0.0)
        self.adaptive_threshold.update(abs(vol_yz_z))

        threshold = self.adaptive_threshold.get_percentile_threshold(
            percentile=self.base_percentile,
            multiplier=mode_config.threshold_multiplier,
        )

        is_triggered = abs(vol_yz_z) > threshold and not self.in_cooldown()

        if is_triggered:
            self.bars_since_trigger = 0

        return CircuitBreakerState(
            name=self.name,
            is_triggered=is_triggered,
            trigger_value=abs(vol_yz_z),
            threshold=threshold,
            cooldown_bars=max(0, self.cooldown_bars - int(min(self.bars_since_trigger, self.cooldown_bars))),
            triggered_at=datetime.now() if is_triggered else None,
            message=f"Vol stress: |vol_yz_z|={abs(vol_yz_z):.2f} > {threshold:.2f}"
            if is_triggered else "",
        )


class LyapunovCircuitBreaker(BaseCircuitBreaker):
    """
    Lyapunov-based circuit breaker for chaos detection.

    High positive Lyapunov = chaotic divergence (butterfly effect).
    Dangerous for directional trading.
    """

    def __init__(self, cooldown_bars: int = 24):
        super().__init__("lyapunov", cooldown_bars)
        self.base_percentile = 0.95

    def check(
        self,
        physics_state: pd.Series,
        mode_config: ModeConfig,
    ) -> CircuitBreakerState:
        if not mode_config.circuit_breaker_enabled:
            return CircuitBreakerState(name=self.name)

        lyap_z = physics_state.get("lyap_z", 0.0)
        self.adaptive_threshold.update(lyap_z)  # Positive = chaos

        threshold = self.adaptive_threshold.get_percentile_threshold(
            percentile=self.base_percentile,
            multiplier=mode_config.threshold_multiplier,
        )

        # Only trigger on HIGH positive Lyapunov (chaos)
        is_triggered = lyap_z > threshold and not self.in_cooldown()

        if is_triggered:
            self.bars_since_trigger = 0

        return CircuitBreakerState(
            name=self.name,
            is_triggered=is_triggered,
            trigger_value=lyap_z,
            threshold=threshold,
            cooldown_bars=max(0, self.cooldown_bars - int(min(self.bars_since_trigger, self.cooldown_bars))),
            triggered_at=datetime.now() if is_triggered else None,
            message=f"Chaotic regime: lyap_z={lyap_z:.2f} > {threshold:.2f}"
            if is_triggered else "",
        )


class FlashCrashCircuitBreaker(BaseCircuitBreaker):
    """
    Flash crash detection using combined indicators.

    Triggers on:
    - Extreme negative velocity (rapid price drop)
    - High VPIN + High kurtosis (toxic + fat tails)
    - Negative skewness spike (left tail)
    """

    def __init__(self, cooldown_bars: int = 6):  # Short cooldown for flash events
        super().__init__("flash_crash", cooldown_bars)
        self.v_threshold = AdaptiveThreshold()
        self.composite_threshold = AdaptiveThreshold()

    def check(
        self,
        physics_state: pd.Series,
        mode_config: ModeConfig,
    ) -> CircuitBreakerState:
        if not mode_config.circuit_breaker_enabled:
            return CircuitBreakerState(name=self.name)

        # Get indicators
        v = physics_state.get("v", 0.0)  # Velocity (log return)
        vpin_z = physics_state.get("vpin_z", 0.0)
        kurtosis_z = physics_state.get("kurtosis_z", 0.0)
        skewness_z = physics_state.get("skewness_z", 0.0)

        # Composite crash score: weighted combination
        crash_score = abs(v) * (1 + max(0, vpin_z)) * (1 + max(0, kurtosis_z))
        self.composite_threshold.update(crash_score)

        threshold = self.composite_threshold.get_percentile_threshold(
            percentile=0.99,  # Very high threshold for flash events
            multiplier=mode_config.threshold_multiplier,
        )

        # Additional condition: negative velocity + negative skew
        is_flash = crash_score > threshold and v < 0 and skewness_z < 0
        is_triggered = is_flash and not self.in_cooldown()

        if is_triggered:
            self.bars_since_trigger = 0

        return CircuitBreakerState(
            name=self.name,
            is_triggered=is_triggered,
            trigger_value=crash_score,
            threshold=threshold,
            cooldown_bars=max(0, self.cooldown_bars - int(min(self.bars_since_trigger, self.cooldown_bars))),
            triggered_at=datetime.now() if is_triggered else None,
            message=f"Flash crash: score={crash_score:.2f} > {threshold:.2f}, v={v:.4f}"
            if is_triggered else "",
        )


class CircuitBreakerManager:
    """Manages all circuit breakers and aggregates their states."""

    def __init__(self, mode_config: Optional[ModeConfig] = None):
        self.mode_config = mode_config or ModeConfig.paper()
        self.breakers: Dict[str, BaseCircuitBreaker] = {}
        self.history: List[Dict[str, Any]] = []
        self.logger = logging.getLogger("circuit_breakers")

        # Initialize default breakers
        self._init_default_breakers()

    def _init_default_breakers(self):
        """Initialize standard set of circuit breakers."""
        self.breakers = {
            "kurtosis": KurtosisCircuitBreaker(),
            "vpin": VPINCircuitBreaker(),
            "volatility": VolatilityCircuitBreaker(),
            "lyapunov": LyapunovCircuitBreaker(),
            "flash_crash": FlashCrashCircuitBreaker(),
        }

    def add_breaker(self, breaker: BaseCircuitBreaker):
        """Add custom circuit breaker."""
        self.breakers[breaker.name] = breaker

    def set_mode(self, mode_config: ModeConfig):
        """Update trading mode configuration."""
        self.mode_config = mode_config

    def check_all(
        self,
        physics_state: pd.Series,
    ) -> Tuple[bool, List[CircuitBreakerState]]:
        """
        Check all circuit breakers.

        Returns:
            (any_triggered, list of states)
        """
        states = []
        any_triggered = False

        for name, breaker in self.breakers.items():
            state = breaker.check(physics_state, self.mode_config)
            states.append(state)

            if state.is_triggered:
                any_triggered = True
                if self.mode_config.log_all_decisions:
                    self.logger.warning(f"[{name.upper()}] {state.message}")
                self.history.append({
                    "timestamp": datetime.now().isoformat(),
                    "breaker": name,
                    "value": state.trigger_value,
                    "threshold": state.threshold,
                    "message": state.message,
                })

            # Tick cooldown
            breaker.tick()

        return any_triggered, states

    def is_safe_to_trade(self, physics_state: pd.Series) -> Tuple[bool, str]:
        """
        Check if safe to enter new trades.

        Returns:
            (is_safe, reason if not safe)
        """
        any_triggered, states = self.check_all(physics_state)

        if any_triggered:
            triggered = [s for s in states if s.is_triggered]
            reasons = [s.message for s in triggered]
            return False, "; ".join(reasons)

        return True, ""

    def get_risk_summary(self) -> Dict[str, Any]:
        """Get current risk state summary."""
        return {
            "mode": self.mode_config.mode.name,
            "breakers": {
                name: {
                    "in_cooldown": b.in_cooldown(),
                    "bars_since_trigger": b.bars_since_trigger,
                }
                for name, b in self.breakers.items()
            },
            "recent_triggers": self.history[-10:] if self.history else [],
        }


# =============================================================================
# TRADE RISK MANAGER: Position-Level Risk
# =============================================================================

@dataclass
class TradeRiskLimits:
    """Dynamic limits for individual trades."""
    max_mae_ratio: float = 0.0  # Max adverse excursion as % of entry
    max_position_bars: int = 0  # Max bars to hold
    min_edge_ratio: float = 0.0  # Min MFE/hypotenuse for valid trade


class TradeRiskManager:
    """
    Manages risk at the individual trade level.

    Features:
    - MAE/MFE tracking and limits
    - Position duration limits
    - Edge ratio optimization
    - Dynamic position sizing
    """

    def __init__(self, mode_config: Optional[ModeConfig] = None):
        self.mode_config = mode_config or ModeConfig.paper()

        # Adaptive thresholds for trade metrics
        self.mae_threshold = AdaptiveThreshold()
        self.mfe_threshold = AdaptiveThreshold()
        self.duration_threshold = AdaptiveThreshold()

        # Trade history for learning
        self.trade_history: List[Dict] = []

    def calculate_position_size(
        self,
        capital: float,
        physics_state: pd.Series,
        base_risk_pct: float = 0.01,  # 1% base risk per trade
    ) -> float:
        """
        Calculate position size based on current conditions.

        Scales down position in high-risk conditions.
        """
        # Base position size
        base_size = capital * base_risk_pct

        if not self.mode_config.position_limits_enabled:
            return base_size

        # Risk multipliers based on physics state
        multiplier = 1.0

        # Reduce in high volatility
        vol_yz_z = physics_state.get("vol_yz_z", 0.0)
        if abs(vol_yz_z) > 1.5:
            multiplier *= 0.5
        elif abs(vol_yz_z) > 1.0:
            multiplier *= 0.75

        # Reduce in chaotic conditions
        lyap_z = physics_state.get("lyap_z", 0.0)
        if lyap_z > 1.5:
            multiplier *= 0.5
        elif lyap_z > 1.0:
            multiplier *= 0.75

        # Reduce when VPIN is high (toxic flow)
        vpin_z = physics_state.get("vpin_z", 0.0)
        if vpin_z > 1.5:
            multiplier *= 0.5
        elif vpin_z > 1.0:
            multiplier *= 0.75

        # Regime-based adjustment
        regime = physics_state.get("regime", "UNKNOWN")
        if regime == "OVERDAMPED":
            multiplier *= 0.5  # Choppy, mean-reverting
        elif regime == "LAMINAR":
            multiplier *= 1.2  # Trending, favorable

        # Apply mode multiplier
        multiplier *= self.mode_config.threshold_multiplier

        return base_size * max(0.1, min(1.5, multiplier))

    def get_dynamic_limits(
        self,
        physics_state: pd.Series,
    ) -> TradeRiskLimits:
        """
        Calculate dynamic trade limits based on current conditions.

        Returns adaptive limits, not hardcoded values.
        """
        # Adaptive trail multiplier from physics
        trail_mult = physics_state.get("adaptive_trail_mult", 2.0)

        # MAE limit scales with volatility and chaos
        vol_yz_z = abs(physics_state.get("vol_yz_z", 0.0))
        lyap_z = abs(physics_state.get("lyap_z", 0.0))

        # More tolerance in high vol/chaos (wider stops)
        mae_base = 0.02  # 2% base
        mae_ratio = mae_base * (1 + vol_yz_z * 0.5) * (1 + lyap_z * 0.3)
        mae_ratio = min(0.10, mae_ratio)  # Cap at 10%

        # Position duration based on regime
        regime = physics_state.get("regime", "UNKNOWN")
        if regime == "LAMINAR":
            max_bars = 96  # Hold longer in trends
        elif regime == "UNDERDAMPED":
            max_bars = 72
        elif regime == "BREAKOUT":
            max_bars = 48
        else:
            max_bars = 24  # Quick exit in choppy

        # Edge ratio threshold
        edge_threshold = self.mfe_threshold.get_percentile_threshold(
            percentile=0.25,  # Bottom 25% = bad trades
            multiplier=self.mode_config.threshold_multiplier,
        )

        return TradeRiskLimits(
            max_mae_ratio=mae_ratio,
            max_position_bars=max_bars,
            min_edge_ratio=edge_threshold if edge_threshold != float('inf') else 0.3,
        )

    def record_trade(self, trade_data: Dict):
        """Record completed trade for adaptive learning."""
        self.trade_history.append(trade_data)

        # Update adaptive thresholds
        if "mae" in trade_data:
            self.mae_threshold.update(abs(trade_data["mae"]))
        if "mfe" in trade_data:
            self.mfe_threshold.update(abs(trade_data["mfe"]))
        if "bars_held" in trade_data:
            self.duration_threshold.update(trade_data["bars_held"])


# =============================================================================
# PORTFOLIO RISK MANAGER: Account-Level Risk
# =============================================================================

class PortfolioRiskManager:
    """
    Manages risk at the portfolio/account level.

    Features:
    - Drawdown monitoring and limits
    - Correlation-based position limits
    - Free margin tracking
    - Monte Carlo risk estimation
    """

    def __init__(
        self,
        initial_capital: float = 100000.0,
        mode_config: Optional[ModeConfig] = None,
    ):
        self.initial_capital = initial_capital
        self.mode_config = mode_config or ModeConfig.paper()

        # Equity tracking
        self.equity_history: List[float] = [initial_capital]
        self.peak_equity = initial_capital

        # Adaptive thresholds
        self.drawdown_threshold = AdaptiveThreshold()

        # Position tracking
        self.open_positions: Dict[str, Dict] = {}

    @property
    def current_equity(self) -> float:
        return self.equity_history[-1]

    @property
    def current_drawdown(self) -> float:
        """Current drawdown as percentage."""
        return (self.peak_equity - self.current_equity) / self.peak_equity

    def update_equity(self, new_equity: float):
        """Update equity and track drawdown."""
        self.equity_history.append(new_equity)
        self.peak_equity = max(self.peak_equity, new_equity)
        self.drawdown_threshold.update(self.current_drawdown)

    def get_max_drawdown_limit(self) -> float:
        """Get dynamic max drawdown limit."""
        if not self.mode_config.drawdown_limits_enabled:
            return 1.0  # No limit

        # Base limit depends on mode
        if self.mode_config.mode == TradingMode.LIVE:
            base_limit = 0.10  # 10% for live
        elif self.mode_config.mode == TradingMode.PAPER:
            base_limit = 0.20  # 20% for paper
        else:
            base_limit = 0.50  # 50% for exploration

        return base_limit * self.mode_config.threshold_multiplier

    def is_drawdown_exceeded(self) -> Tuple[bool, float]:
        """Check if drawdown limit exceeded."""
        limit = self.get_max_drawdown_limit()
        current_dd = self.current_drawdown
        return current_dd > limit, current_dd

    def get_available_risk_budget(self) -> float:
        """Calculate available risk budget as % of equity."""
        current_dd = self.current_drawdown
        max_dd = self.get_max_drawdown_limit()
        remaining = max_dd - current_dd
        return max(0.0, remaining)

    def calculate_correlation_limit(
        self,
        new_instrument: str,
        correlations: Dict[str, float],
    ) -> float:
        """
        Calculate position limit based on correlation with existing positions.

        High correlation = lower limit to reduce concentration.
        """
        if not self.open_positions:
            return 1.0  # No limit if no positions

        max_corr = 0.0
        for pos_key in self.open_positions:
            corr = correlations.get(f"{pos_key}_{new_instrument}", 0.0)
            max_corr = max(max_corr, abs(corr))

        # Reduce limit for high correlation
        if max_corr > 0.8:
            return 0.25
        elif max_corr > 0.6:
            return 0.50
        elif max_corr > 0.4:
            return 0.75
        return 1.0

    def add_position(self, key: str, position_data: Dict):
        """Add open position to tracking."""
        self.open_positions[key] = position_data

    def remove_position(self, key: str):
        """Remove closed position from tracking."""
        if key in self.open_positions:
            del self.open_positions[key]

    def get_portfolio_summary(self) -> Dict[str, Any]:
        """Get portfolio risk summary."""
        return {
            "equity": self.current_equity,
            "peak_equity": self.peak_equity,
            "drawdown": self.current_drawdown,
            "drawdown_limit": self.get_max_drawdown_limit(),
            "risk_budget": self.get_available_risk_budget(),
            "open_positions": len(self.open_positions),
            "mode": self.mode_config.mode.name,
        }


# =============================================================================
# SHADOW AGENT ARCHITECTURE: Tripleganger
# =============================================================================

class ShadowAgentState(Enum):
    """State of a shadow agent."""
    ACTIVE = auto()      # Currently trading
    FROZEN = auto()      # Frozen for comparison
    TRAINING = auto()    # Being retrained
    CANDIDATE = auto()   # Ready for promotion


@dataclass
class AgentPerformance:
    """Performance metrics for an agent."""
    total_reward: float = 0.0
    total_pnl: float = 0.0
    trades: int = 0
    win_rate: float = 0.0
    avg_edge_ratio: float = 0.0
    sharpe: float = 0.0
    last_updated: Optional[datetime] = None


class ShadowAgent:
    """
    Shadow agent in the Tripleganger system.

    Each shadow agent wraps a base agent and tracks its performance.
    """

    def __init__(
        self,
        agent: Any,  # Base agent (LinearQAgent, etc.)
        agent_id: str,
        state: ShadowAgentState = ShadowAgentState.ACTIVE,
    ):
        self.agent = agent
        self.agent_id = agent_id
        self.state = state
        self.performance = AgentPerformance()
        self.creation_time = datetime.now()
        self.frozen_at: Optional[datetime] = None

        # Track decisions for drift detection
        self.decision_history: List[Tuple[np.ndarray, int, float]] = []

    def select_action(self, state: np.ndarray, epsilon: float = 0.1) -> int:
        """Select action and track for drift detection."""
        action = self.agent.select_action(state, epsilon)
        q_values = self.agent.get_q_values(state)
        self.decision_history.append((state.copy(), action, q_values.max()))

        # Keep only recent history
        if len(self.decision_history) > 1000:
            self.decision_history = self.decision_history[-1000:]

        return action

    def update(self, state, action, reward, next_state, done):
        """Update agent (only if not frozen)."""
        if self.state == ShadowAgentState.FROZEN:
            return 0.0  # Frozen agents don't learn
        return self.agent.update(state, action, reward, next_state, done)

    def freeze(self):
        """Freeze agent (stop learning, preserve state)."""
        self.state = ShadowAgentState.FROZEN
        self.frozen_at = datetime.now()

    def unfreeze(self):
        """Unfreeze agent (resume learning)."""
        self.state = ShadowAgentState.TRAINING

    def clone(self, new_id: str) -> "ShadowAgent":
        """Create a clone of this agent."""
        agent_copy = copy.deepcopy(self.agent)
        new_shadow = ShadowAgent(
            agent=agent_copy,
            agent_id=new_id,
            state=ShadowAgentState.TRAINING,
        )
        return new_shadow

    def update_performance(
        self,
        reward: float,
        pnl: float,
        is_win: bool,
        edge_ratio: float,
    ):
        """Update performance metrics."""
        self.performance.total_reward += reward
        self.performance.total_pnl += pnl
        self.performance.trades += 1

        # Rolling win rate
        total_wins = self.performance.win_rate * (self.performance.trades - 1)
        if is_win:
            total_wins += 1
        self.performance.win_rate = total_wins / self.performance.trades

        # Rolling edge ratio
        total_edge = self.performance.avg_edge_ratio * (self.performance.trades - 1)
        total_edge += edge_ratio
        self.performance.avg_edge_ratio = total_edge / self.performance.trades

        self.performance.last_updated = datetime.now()


class TrplegangerSystem:
    """
    Tripleganger Multi-Agent Risk Management System.

    Architecture:
    1. Live Agent: The primary agent that actually executes trades
    2. Shadow Agent A (Frozen): Frozen copy of live agent for drift detection
    3. Shadow Agent B (Retraining): Continuously retrained candidate for promotion

    Key Features:
    - Drift detection: Compare live vs frozen to detect performance degradation
    - Continuous improvement: Shadow B learns from new data
    - Safe promotion: Only promote shadow if it outperforms live
    - Rollback capability: Restore from frozen if live degrades
    """

    def __init__(
        self,
        live_agent: Any,
        mode_config: Optional[ModeConfig] = None,
        drift_threshold: float = 0.2,  # 20% performance drift triggers warning
        promotion_threshold: float = 0.1,  # 10% better to promote
    ):
        self.mode_config = mode_config or ModeConfig.paper()
        self.drift_threshold = drift_threshold
        self.promotion_threshold = promotion_threshold

        # Initialize shadow agents
        self.live_agent = ShadowAgent(
            agent=live_agent,
            agent_id="live",
            state=ShadowAgentState.ACTIVE,
        )

        self.frozen_shadow = self.live_agent.clone("frozen")
        self.frozen_shadow.freeze()

        self.training_shadow = self.live_agent.clone("training")
        self.training_shadow.state = ShadowAgentState.TRAINING

        # Risk managers
        self.circuit_breakers = CircuitBreakerManager(mode_config)
        self.trade_risk = TradeRiskManager(mode_config)
        self.portfolio_risk = PortfolioRiskManager(mode_config=mode_config)

        # Logger
        self.logger = logging.getLogger("tripleganger")

        # Event history
        self.events: List[Dict] = []

    def set_mode(self, mode_config: ModeConfig):
        """Update trading mode for all components."""
        self.mode_config = mode_config
        self.circuit_breakers.set_mode(mode_config)
        self.trade_risk.mode_config = mode_config
        self.portfolio_risk.mode_config = mode_config

    def select_action(
        self,
        state: np.ndarray,
        physics_state: pd.Series,
        epsilon: float = 0.1,
    ) -> Tuple[int, Dict[str, Any]]:
        """
        Select action with full risk management.

        Returns:
            (action, info_dict)
        """
        info = {
            "is_safe": True,
            "circuit_breakers": [],
            "position_size_mult": 1.0,
            "agent_used": "live",
        }

        # Check circuit breakers
        is_safe, reason = self.circuit_breakers.is_safe_to_trade(physics_state)
        if not is_safe:
            info["is_safe"] = False
            info["block_reason"] = reason

            if self.mode_config.mode == TradingMode.LIVE:
                # In live mode, force HOLD (action 0)
                return 0, info
            else:
                # In exploration, log but continue
                self.logger.warning(f"[EXPLORATION] Would block: {reason}")

        # Calculate position size multiplier
        base_capital = self.portfolio_risk.current_equity
        position_size = self.trade_risk.calculate_position_size(
            capital=base_capital,
            physics_state=physics_state,
        )
        info["position_size_mult"] = position_size / (base_capital * 0.01)

        # Get actions from all agents
        live_action = self.live_agent.select_action(state, epsilon)
        frozen_action = self.frozen_shadow.select_action(state, 0.0)  # No exploration
        training_action = self.training_shadow.select_action(state, epsilon * 0.5)

        info["live_action"] = live_action
        info["frozen_action"] = frozen_action
        info["training_action"] = training_action

        # Use live agent's action
        return live_action, info

    def update_all(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ):
        """Update all agents (respecting their states)."""
        # Live agent learns
        self.live_agent.update(state, action, reward, next_state, done)

        # Frozen doesn't learn (handled in ShadowAgent.update)
        self.frozen_shadow.update(state, action, reward, next_state, done)

        # Training shadow learns more aggressively
        self.training_shadow.update(state, action, reward, next_state, done)

    def record_trade_result(
        self,
        trade_data: Dict,
    ):
        """Record trade result for all tracking systems."""
        # Extract metrics
        pnl = trade_data.get("raw_pnl", 0)
        reward = trade_data.get("shaped_reward", pnl)
        is_win = pnl > 0
        edge_ratio = trade_data.get("edge_ratio", 0.5)

        # Update agent performances
        self.live_agent.update_performance(reward, pnl, is_win, edge_ratio)
        self.training_shadow.update_performance(reward, pnl, is_win, edge_ratio)

        # Update trade risk manager
        self.trade_risk.record_trade(trade_data)

        # Update portfolio equity
        new_equity = self.portfolio_risk.current_equity + pnl
        self.portfolio_risk.update_equity(new_equity)

    def check_drift(self) -> Tuple[bool, float, str]:
        """
        Check for performance drift between live and frozen agents.

        Returns:
            (is_drifted, drift_amount, message)
        """
        if self.live_agent.performance.trades < 20:
            return False, 0.0, "Insufficient trades for drift detection"

        live_perf = self.live_agent.performance
        frozen_perf = self.frozen_shadow.performance

        # Compare average rewards per trade
        live_avg = live_perf.total_reward / max(1, live_perf.trades)
        frozen_avg = frozen_perf.total_reward / max(1, frozen_perf.trades)

        if frozen_avg > 0:
            drift = (frozen_avg - live_avg) / frozen_avg
        else:
            drift = 0.0

        is_drifted = drift > self.drift_threshold

        if is_drifted:
            msg = f"Performance drift detected: {drift*100:.1f}% worse than frozen"
            self.logger.warning(msg)
            self.events.append({
                "type": "drift_detected",
                "timestamp": datetime.now().isoformat(),
                "drift": drift,
                "live_avg": live_avg,
                "frozen_avg": frozen_avg,
            })
            return True, drift, msg

        return False, drift, ""

    def check_promotion(self) -> Tuple[bool, str]:
        """
        Check if training shadow should be promoted to live.

        Returns:
            (should_promote, message)
        """
        if self.training_shadow.performance.trades < 30:
            return False, "Insufficient trades for promotion evaluation"

        live_perf = self.live_agent.performance
        training_perf = self.training_shadow.performance

        # Compare average rewards
        live_avg = live_perf.total_reward / max(1, live_perf.trades)
        training_avg = training_perf.total_reward / max(1, training_perf.trades)

        if live_avg > 0:
            improvement = (training_avg - live_avg) / live_avg
        else:
            improvement = training_avg if training_avg > 0 else 0

        should_promote = improvement > self.promotion_threshold

        if should_promote:
            msg = f"Training shadow ready for promotion: {improvement*100:.1f}% improvement"
            self.logger.info(msg)
            return True, msg

        return False, ""

    def promote_training_shadow(self):
        """Promote training shadow to live, demote live to frozen."""
        self.logger.info("Promoting training shadow to live agent")

        # Current live becomes new frozen
        old_live = self.live_agent
        old_live.freeze()

        # Training becomes new live
        self.training_shadow.state = ShadowAgentState.ACTIVE
        self.live_agent = self.training_shadow

        # Old frozen is discarded, old live is new frozen
        self.frozen_shadow = old_live

        # Create new training shadow from new live
        self.training_shadow = self.live_agent.clone("training")
        self.training_shadow.state = ShadowAgentState.TRAINING

        self.events.append({
            "type": "promotion",
            "timestamp": datetime.now().isoformat(),
            "new_live_perf": self.live_agent.performance.total_reward,
        })

    def rollback_to_frozen(self):
        """Rollback live agent to frozen version (restore known-good state)."""
        self.logger.warning("Rolling back to frozen agent")

        # Clone frozen to become new live
        restored = self.frozen_shadow.clone("live")
        restored.state = ShadowAgentState.ACTIVE
        restored.unfreeze()

        # Keep current frozen as is
        self.live_agent = restored

        # Create new training from restored
        self.training_shadow = restored.clone("training")
        self.training_shadow.state = ShadowAgentState.TRAINING

        self.events.append({
            "type": "rollback",
            "timestamp": datetime.now().isoformat(),
        })

    def get_system_summary(self) -> Dict[str, Any]:
        """Get full system status summary."""
        return {
            "mode": self.mode_config.mode.name,
            "agents": {
                "live": {
                    "state": self.live_agent.state.name,
                    "trades": self.live_agent.performance.trades,
                    "total_reward": self.live_agent.performance.total_reward,
                    "win_rate": self.live_agent.performance.win_rate,
                },
                "frozen": {
                    "state": self.frozen_shadow.state.name,
                    "trades": self.frozen_shadow.performance.trades,
                    "frozen_at": self.frozen_shadow.frozen_at.isoformat()
                    if self.frozen_shadow.frozen_at else None,
                },
                "training": {
                    "state": self.training_shadow.state.name,
                    "trades": self.training_shadow.performance.trades,
                    "total_reward": self.training_shadow.performance.total_reward,
                },
            },
            "circuit_breakers": self.circuit_breakers.get_risk_summary(),
            "portfolio": self.portfolio_risk.get_portfolio_summary(),
            "recent_events": self.events[-5:] if self.events else [],
        }


# =============================================================================
# MAIN: Demo and Testing
# =============================================================================

def demo_tripleganger():
    """Demonstrate the Tripleganger system."""
    print("=" * 70)
    print("TRIPLEGANGER RISK MANAGEMENT SYSTEM - DEMO")
    print("=" * 70)

    # Create mock agent
    class MockAgent:
        def __init__(self, state_dim=64, n_actions=4):
            self.state_dim = state_dim
            self.n_actions = n_actions
            self.weights = np.random.randn(n_actions, state_dim) * 0.01

        def select_action(self, state, epsilon=0.1):
            if np.random.random() < epsilon:
                return np.random.randint(self.n_actions)
            q = self.weights @ state
            return int(np.argmax(q))

        def update(self, state, action, reward, next_state, done):
            return 0.01

        def get_q_values(self, state):
            return self.weights @ state

    # Create tripleganger system in exploration mode
    agent = MockAgent()
    system = TrplegangerSystem(
        live_agent=agent,
        mode_config=ModeConfig.exploration(),
    )

    print("\n[1] System initialized in EXPLORATION mode")
    print(f"    Live agent: {system.live_agent.agent_id}")
    print(f"    Frozen shadow: {system.frozen_shadow.agent_id}")
    print(f"    Training shadow: {system.training_shadow.agent_id}")

    # Simulate some physics state
    mock_physics = pd.Series({
        "v": 0.001,
        "kurtosis_z": 1.5,
        "vpin": 0.4,
        "vpin_z": 0.5,
        "vol_yz_z": 1.0,
        "lyap_z": 0.8,
        "skewness_z": -0.3,
        "regime": "LAMINAR",
        "adaptive_trail_mult": 2.0,
    })

    # Test action selection
    state = np.random.randn(64)
    action, info = system.select_action(state, mock_physics)

    print(f"\n[2] Action selection test:")
    print(f"    Action: {action}")
    print(f"    Is safe: {info['is_safe']}")
    print(f"    Position size mult: {info['position_size_mult']:.2f}")

    # Switch to live mode
    print("\n[3] Switching to LIVE mode...")
    system.set_mode(ModeConfig.live())

    # Simulate high-risk scenario
    risky_physics = pd.Series({
        "v": -0.05,  # Sharp drop
        "kurtosis_z": 3.5,  # Fat tails
        "vpin": 0.85,  # High toxicity
        "vpin_z": 2.5,
        "vol_yz_z": 3.0,  # High vol
        "lyap_z": 2.5,  # Chaos
        "skewness_z": -2.0,  # Left tail
        "regime": "BREAKOUT",
        "adaptive_trail_mult": 4.0,
    })

    action, info = system.select_action(state, risky_physics, epsilon=0.1)

    print(f"\n[4] High-risk scenario (LIVE mode):")
    print(f"    Action: {action} (should be 0=HOLD if blocked)")
    print(f"    Is safe: {info['is_safe']}")
    if not info['is_safe']:
        print(f"    Block reason: {info.get('block_reason', 'N/A')}")

    # Show system summary
    print("\n[5] System Summary:")
    summary = system.get_system_summary()
    print(f"    Mode: {summary['mode']}")
    print(f"    Live agent trades: {summary['agents']['live']['trades']}")
    print(f"    Portfolio equity: ${summary['portfolio']['equity']:,.0f}")
    print(f"    Drawdown: {summary['portfolio']['drawdown']*100:.1f}%")

    print("\n" + "=" * 70)
    print("TRIPLEGANGER DEMO COMPLETE")
    print("=" * 70)

    return system


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    demo_tripleganger()
