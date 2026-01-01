#!/usr/bin/env python3
"""
Specialist Agents with Doppelgänger Integration & Self-Healing Portfolio

Architecture:
- Asset Class Specialists: Forex, Crypto, Index, Commodity, Metals
- Each specialist has: Live Agent + Shadow A (Frozen) + Shadow B (Online)
- Virtual PnL tracking with full risk metrics
- Self-Healing Portfolio Health Score

Risk Metrics:
- Sharpe, Sortino, Omega (risk-adjusted return)
- Calmar, Burke, Sterling (drawdown-based)
- Edge Validation (profit factor, win rate)
- Portfolio Health Score (composite 0-100)

Author: Physics-First Trading System
"""

import copy
import json
import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from numpy import floating


# =============================================================================
# ASSET CLASS DEFINITIONS
# =============================================================================

class AssetClass(Enum):
    FOREX = "Forex"
    CRYPTO = "Crypto"
    INDEX = "Index"
    COMMODITY = "Commodity"
    METALS = "Metals"
    OTHER = "Other"


INSTRUMENT_MAPPING = {
    # Forex
    "AUDJPY+": AssetClass.FOREX, "AUDUSD+": AssetClass.FOREX,
    "EURJPY+": AssetClass.FOREX, "GBPJPY+": AssetClass.FOREX,
    "GBPUSD+": AssetClass.FOREX,
    # Crypto
    "BTCJPY": AssetClass.CRYPTO, "BTCUSD": AssetClass.CRYPTO,
    "ETHEUR": AssetClass.CRYPTO, "XRPJPY": AssetClass.CRYPTO,
    # Index
    "DJ30ft": AssetClass.INDEX, "NAS100": AssetClass.INDEX,
    "Nikkei225": AssetClass.INDEX, "EU50": AssetClass.INDEX,
    "GER40": AssetClass.INDEX, "SA40": AssetClass.INDEX,
    "US2000": AssetClass.INDEX,
    # Commodity
    "COPPER-C": AssetClass.COMMODITY, "UKOUSD": AssetClass.COMMODITY,
    # Metals
    "XAGUSD": AssetClass.METALS, "XAUAUD+": AssetClass.METALS,
    "XAUUSD+": AssetClass.METALS, "XPTUSD": AssetClass.METALS,
}


def get_asset_class(instrument_key: str) -> AssetClass:
    """Get asset class from instrument key."""
    for prefix, cls in INSTRUMENT_MAPPING.items():
        if instrument_key.startswith(prefix.replace("+", "")):
            return cls
    return AssetClass.OTHER


# =============================================================================
# RISK METRICS CALCULATOR
# =============================================================================

class RiskMetrics:
    """
    Comprehensive risk metrics calculator.

    Includes:
    - Sharpe Ratio (risk-adjusted return vs total volatility)
    - Sortino Ratio (risk-adjusted return vs downside volatility)
    - Omega Ratio (probability-weighted gains/losses)
    - Calmar Ratio (return / max drawdown)
    - Burke Ratio (return / sqrt(sum of squared drawdowns))
    - Sterling Ratio (return / avg of top N drawdowns)
    """

    def __init__(
        self,
        trading_days_per_year: int = 252,
        risk_free_rate: float = 0.0,
        omega_threshold: float = 0.0,
    ):
        self.trading_days_per_year = trading_days_per_year
        self.risk_free_rate = risk_free_rate
        self.omega_threshold = omega_threshold

    def sharpe_ratio(self, returns: np.ndarray) -> float:
        """Annualized Sharpe Ratio."""
        if len(returns) < 30 or np.std(returns) < 1e-8:
            return 0.0
        excess_return = np.mean(returns) - self.risk_free_rate / self.trading_days_per_year
        return excess_return / np.std(returns) * np.sqrt(self.trading_days_per_year)

    def sortino_ratio(self, returns: np.ndarray) -> float:
        """Annualized Sortino Ratio (downside deviation only)."""
        if len(returns) < 30:
            return 0.0
        downside_returns = returns[returns < self.omega_threshold]
        if len(downside_returns) < 5:
            return float('inf') if np.mean(returns) > 0 else 0.0
        downside_dev = np.std(downside_returns)
        if downside_dev < 1e-8:
            return float('inf') if np.mean(returns) > 0 else 0.0
        excess_return = np.mean(returns) - self.risk_free_rate / self.trading_days_per_year
        return excess_return / downside_dev * np.sqrt(self.trading_days_per_year)

    def omega_ratio(self, returns: np.ndarray, threshold: float = None) -> float:
        """Omega Ratio (probability-weighted gains/losses)."""
        if len(returns) < 30:
            return 1.0
        L = threshold if threshold is not None else self.omega_threshold
        gains = np.sum(np.maximum(returns - L, 0))
        losses = np.sum(np.maximum(L - returns, 0))
        return (gains + 1e-8) / (losses + 1e-8)

    def calmar_ratio(self, equity_curve: np.ndarray, years: float = None) -> float:
        """Calmar Ratio (CAGR / Max Drawdown)."""
        if len(equity_curve) < 30:
            return 0.0

        if years is None:
            years = len(equity_curve) / self.trading_days_per_year
        if years < 0.1:
            return 0.0

        cagr = (equity_curve[-1] / equity_curve[0]) ** (1 / years) - 1

        peak = np.maximum.accumulate(equity_curve)
        drawdown = (peak - equity_curve) / (peak + 1e-8)
        max_dd = np.max(drawdown)

        if max_dd < 1e-8:
            return float('inf') if cagr > 0 else 0.0
        return cagr / max_dd

    def burke_ratio(
        self,
        equity_curve: np.ndarray,
        n_drawdowns: int = 5,
        years: float = None,
    ) -> float:
        """Burke Ratio (CAGR / sqrt(sum of squared drawdowns))."""
        if len(equity_curve) < 30:
            return 0.0

        if years is None:
            years = len(equity_curve) / self.trading_days_per_year
        if years < 0.1:
            return 0.0

        cagr = (equity_curve[-1] / equity_curve[0]) ** (1 / years) - 1

        # Find individual drawdowns
        drawdowns = self._find_drawdowns(equity_curve)
        if not drawdowns:
            return float('inf') if cagr > 0 else 0.0

        # Top N drawdowns
        top_dd = sorted(drawdowns, reverse=True)[:n_drawdowns]
        sum_sq = sum(dd**2 for dd in top_dd)

        if sum_sq < 1e-8:
            return float('inf') if cagr > 0 else 0.0
        return cagr / np.sqrt(sum_sq)

    def sterling_ratio(
        self,
        equity_curve: np.ndarray,
        n_drawdowns: int = 3,
        adjustment: float = 0.10,
        years: float = None,
    ) -> float:
        """Sterling Ratio (CAGR / (avg of top N drawdowns + adjustment))."""
        if len(equity_curve) < 30:
            return 0.0

        if years is None:
            years = len(equity_curve) / self.trading_days_per_year
        if years < 0.1:
            return 0.0

        cagr = (equity_curve[-1] / equity_curve[0]) ** (1 / years) - 1

        drawdowns = self._find_drawdowns(equity_curve)
        if not drawdowns:
            return float('inf') if cagr > 0 else 0.0

        top_dd = sorted(drawdowns, reverse=True)[:n_drawdowns]
        avg_dd = np.mean(top_dd) + adjustment

        if avg_dd < 1e-8:
            return float('inf') if cagr > 0 else 0.0
        return cagr / avg_dd

    def _find_drawdowns(self, equity_curve: np.ndarray) -> List[float]:
        """Find individual drawdown events."""
        peak = equity_curve[0]
        drawdowns = []
        current_dd = 0.0

        for equity in equity_curve:
            if equity > peak:
                if current_dd > 0:
                    drawdowns.append(current_dd)
                peak = equity
                current_dd = 0.0
            else:
                dd = (peak - equity) / peak
                current_dd = max(current_dd, dd)

        if current_dd > 0:
            drawdowns.append(current_dd)

        return drawdowns

    def ulcer_index(self, equity_curve: np.ndarray) -> float:
        """Ulcer Index (sqrt of mean squared drawdown)."""
        if len(equity_curve) < 30:
            return 0.0
        peak = np.maximum.accumulate(equity_curve)
        drawdown = (peak - equity_curve) / (peak + 1e-8)
        return np.sqrt(np.mean(drawdown**2))

    def calculate_all(
        self,
        returns: np.ndarray,
        equity_curve: np.ndarray,
    ) -> Dict[str, float]:
        """Calculate all risk metrics."""
        return {
            "sharpe": self.sharpe_ratio(returns),
            "sortino": self.sortino_ratio(returns),
            "omega": self.omega_ratio(returns),
            "calmar": self.calmar_ratio(equity_curve),
            "burke": self.burke_ratio(equity_curve),
            "sterling": self.sterling_ratio(equity_curve),
            "ulcer_index": self.ulcer_index(equity_curve),
        }


# =============================================================================
# VIRTUAL PORTFOLIO (Per-Agent Tracking)
# =============================================================================

class VirtualPortfolio:
    """
    Virtual portfolio for tracking agent performance.

    Tracks:
    - Equity curve and returns
    - Drawdown (current, max)
    - All risk metrics (Sharpe, Sortino, Omega, Calmar, Burke, Sterling)
    - Trade statistics (count, win rate, profit factor)
    """

    def __init__(
        self,
        initial_capital: float = 100000.0,
        trading_days_per_year: int = 252,
    ):
        self.initial_capital = initial_capital
        self.trading_days_per_year = trading_days_per_year

        # State
        self.capital = initial_capital
        self.position = 0  # 0 = flat, 1 = long, -1 = short
        self.entry_price = 0.0

        # Tracking
        self.equity_curve = deque([initial_capital], maxlen=10000)
        self.returns = deque(maxlen=10000)

        # Trade stats
        self.trade_count = 0
        self.winning_trades = 0
        self.gross_profit = 0.0
        self.gross_loss = 0.0
        self.realized_pnl = 0.0
        self.unrealized_pnl = 0.0

        # Drawdown tracking
        self.peak_equity = initial_capital
        self.current_drawdown = 0.0
        self.max_drawdown = 0.0

        # Risk calculator
        self.risk_calc = RiskMetrics(trading_days_per_year)

    def apply_action(self, action: int, current_price: float, position_size: float = 1.0):
        """
        Apply trading action.

        Actions: 0=HOLD, 1=LONG, 2=SHORT, 3=CLOSE
        """
        pnl = 0.0

        if action == 3 or (action in [1, 2] and self.position != 0):
            # Close existing position
            if self.position == 1:  # Was long
                pnl = (current_price - self.entry_price) * position_size
            elif self.position == -1:  # Was short
                pnl = (self.entry_price - current_price) * position_size

            if pnl > 0:
                self.winning_trades += 1
                self.gross_profit += pnl
            else:
                self.gross_loss += abs(pnl)

            self.trade_count += 1
            self.realized_pnl += pnl
            self.position = 0
            self.entry_price = 0.0

        if action == 1 and self.position == 0:  # Enter long
            self.position = 1
            self.entry_price = current_price
        elif action == 2 and self.position == 0:  # Enter short
            self.position = -1
            self.entry_price = current_price

        # Update unrealized
        self.update_unrealized(current_price, position_size)

        return pnl

    def update_unrealized(self, current_price: float, position_size: float = 1.0):
        """Update unrealized PnL."""
        if self.position == 1:
            self.unrealized_pnl = (current_price - self.entry_price) * position_size
        elif self.position == -1:
            self.unrealized_pnl = (self.entry_price - current_price) * position_size
        else:
            self.unrealized_pnl = 0.0

        # Update equity
        current_equity = self.initial_capital + self.realized_pnl + self.unrealized_pnl
        self._update_equity(current_equity)

    def _update_equity(self, current_equity: float):
        """Update equity curve and drawdown."""
        prev_equity = self.equity_curve[-1] if self.equity_curve else self.initial_capital

        self.equity_curve.append(current_equity)

        if prev_equity > 0:
            ret = (current_equity - prev_equity) / prev_equity
            self.returns.append(ret)

        # Drawdown
        if current_equity > self.peak_equity:
            self.peak_equity = current_equity

        self.current_drawdown = (self.peak_equity - current_equity) / (self.peak_equity + 1e-8)
        self.max_drawdown = max(self.max_drawdown, self.current_drawdown)

    def total_equity(self) -> float:
        """Current total equity."""
        return self.initial_capital + self.realized_pnl + self.unrealized_pnl

    def win_rate(self) -> float:
        """Win rate percentage."""
        if self.trade_count == 0:
            return 0.0
        return self.winning_trades / self.trade_count

    def profit_factor(self) -> float:
        """Profit factor (gross profit / gross loss)."""
        if self.gross_loss < 1e-8:
            return float('inf') if self.gross_profit > 0 else 0.0
        return self.gross_profit / self.gross_loss

    def get_metrics(self) -> Dict[str, float]:
        """Get all performance metrics."""
        equity_arr = np.array(self.equity_curve)
        returns_arr = np.array(self.returns)

        metrics = self.risk_calc.calculate_all(returns_arr, equity_arr)

        metrics.update({
            "total_equity": self.total_equity(),
            "realized_pnl": self.realized_pnl,
            "unrealized_pnl": self.unrealized_pnl,
            "current_drawdown": self.current_drawdown,
            "max_drawdown": self.max_drawdown,
            "trade_count": self.trade_count,
            "win_rate": self.win_rate(),
            "profit_factor": self.profit_factor(),
        })

        return metrics

    def reset(self):
        """Reset portfolio to initial state."""
        self.capital = self.initial_capital
        self.position = 0
        self.entry_price = 0.0
        self.equity_curve = deque([self.initial_capital], maxlen=10000)
        self.returns = deque(maxlen=10000)
        self.trade_count = 0
        self.winning_trades = 0
        self.gross_profit = 0.0
        self.gross_loss = 0.0
        self.realized_pnl = 0.0
        self.unrealized_pnl = 0.0
        self.peak_equity = self.initial_capital
        self.current_drawdown = 0.0
        self.max_drawdown = 0.0


# =============================================================================
# EDGE VALIDATOR
# =============================================================================

class EdgeValidator:
    """
    Validates if an agent maintains trading edge.

    Uses rolling window of trades to assess:
    - Profit factor (>1.3 required)
    - Win rate (>52% required)
    - Sharpe/Sortino ratios
    - Max drawdown limits
    """

    def __init__(
        self,
        window: int = 200,
        min_profit_factor: float = 1.3,
        min_win_rate: float = 0.52,
        min_sharpe: float = 0.8,
        max_drawdown: float = 0.20,
    ):
        self.window = window
        self.min_profit_factor = min_profit_factor
        self.min_win_rate = min_win_rate
        self.min_sharpe = min_sharpe
        self.max_drawdown = max_drawdown

        self.trade_history = deque(maxlen=window)

    def record_trade(self, pnl: float, metrics: Dict[str, float]):
        """Record a completed trade."""
        self.trade_history.append({
            "pnl": pnl,
            "win": pnl > 0,
            "sharpe": metrics.get("sharpe", 0),
            "sortino": metrics.get("sortino", 0),
            "max_dd": metrics.get("max_drawdown", 0),
        })

    def has_edge(self) -> bool:
        """Check if agent maintains edge."""
        if len(self.trade_history) < 50:
            return True  # Insufficient data, assume edge

        recent = list(self.trade_history)[-50:]

        # Profit factor
        gross_profit = sum(t["pnl"] for t in recent if t["pnl"] > 0)
        gross_loss = abs(sum(t["pnl"] for t in recent if t["pnl"] < 0))
        pf = gross_profit / (gross_loss + 1e-8)

        # Win rate
        wins = sum(1 for t in recent if t["win"])
        wr = wins / len(recent)

        # Average Sharpe
        avg_sharpe = np.mean([t["sharpe"] for t in recent])

        # Max drawdown
        max_dd = max(t["max_dd"] for t in recent)

        return (
            pf > self.min_profit_factor and
            wr > self.min_win_rate and
            avg_sharpe > self.min_sharpe and
            max_dd < self.max_drawdown
        )

    def get_summary(self) -> Dict[str, float]:
        """Get edge validation summary."""
        if len(self.trade_history) < 10:
            return {"has_edge": True, "confidence": 0.0}

        recent = list(self.trade_history)

        gross_profit = sum(t["pnl"] for t in recent if t["pnl"] > 0)
        gross_loss = abs(sum(t["pnl"] for t in recent if t["pnl"] < 0))

        return {
            "has_edge": self.has_edge(),
            "profit_factor": gross_profit / (gross_loss + 1e-8),
            "win_rate": sum(1 for t in recent if t["win"]) / len(recent),
            "avg_sharpe": np.mean([t["sharpe"] for t in recent]),
            "max_dd": max(t["max_dd"] for t in recent),
            "trade_count": len(recent),
        }


# =============================================================================
# DOPPELGÄNGER SPECIALIST
# =============================================================================

class DoppelgangerSpecialist:
    """
    Asset Class Specialist with Doppelgänger Shadow Agents.

    Architecture:
    - Live Agent: Primary trading agent
    - Shadow A (Frozen): Frozen copy for drift detection
    - Shadow B (Online): Continuously learning for potential promotion

    Each agent has its own VirtualPortfolio for independent PnL tracking.
    """

    def __init__(
        self,
        asset_class: AssetClass,
        agent: Any,  # Base agent (LinearQAgent, etc.)
        initial_capital: float = 100000.0,
        drift_threshold: float = 0.15,  # 15% drift triggers warning
        promotion_threshold: float = 0.10,  # 10% improvement for promotion
    ):
        self.asset_class = asset_class
        self.drift_threshold = drift_threshold
        self.promotion_threshold = promotion_threshold

        # Create shadow agents
        self.live_agent = agent
        self.shadow_a_agent = copy.deepcopy(agent)  # Frozen
        self.shadow_b_agent = copy.deepcopy(agent)  # Online learning

        # Virtual portfolios
        self.live_port = VirtualPortfolio(initial_capital)
        self.shadow_a_port = VirtualPortfolio(initial_capital)
        self.shadow_b_port = VirtualPortfolio(initial_capital)

        # Edge validators
        self.live_validator = EdgeValidator()
        self.shadow_a_validator = EdgeValidator()
        self.shadow_b_validator = EdgeValidator()

        # Tracking
        self.bars_processed = 0
        self.promotion_count = 0
        self.rollback_count = 0
        self.frozen_at_bar = 0

        # Logging
        self.logger = logging.getLogger(f"specialist_{asset_class.value}")

    def select_action(self, state: np.ndarray, epsilon: float = 0.1) -> Tuple[int, Dict]:
        """
        Select action from live agent, track all shadows.

        Returns:
            (action, info_dict with all agent actions)
        """
        live_action = self.live_agent.select_action(state, epsilon)
        shadow_a_action = self.shadow_a_agent.select_action(state, 0.0)  # No exploration
        shadow_b_action = self.shadow_b_agent.select_action(state, epsilon * 0.5)

        return live_action, {
            "live_action": live_action,
            "shadow_a_action": shadow_a_action,
            "shadow_b_action": shadow_b_action,
        }

    def update(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        current_price: float,
    ):
        """Update all agents and portfolios."""
        self.bars_processed += 1

        # Update agents (Shadow A is frozen, doesn't learn)
        self.live_agent.update(state, action, reward, next_state, done)
        # shadow_a_agent: NO UPDATE (frozen)
        self.shadow_b_agent.update(state, action, reward, next_state, done)

        # Apply actions to virtual portfolios
        live_action, info = self.select_action(state, epsilon=0.0)

        live_pnl = self.live_port.apply_action(info["live_action"], current_price)
        shadow_a_pnl = self.shadow_a_port.apply_action(info["shadow_a_action"], current_price)
        shadow_b_pnl = self.shadow_b_port.apply_action(info["shadow_b_action"], current_price)

        # Record trades if closed
        if live_pnl != 0:
            self.live_validator.record_trade(live_pnl, self.live_port.get_metrics())
        if shadow_a_pnl != 0:
            self.shadow_a_validator.record_trade(shadow_a_pnl, self.shadow_a_port.get_metrics())
        if shadow_b_pnl != 0:
            self.shadow_b_validator.record_trade(shadow_b_pnl, self.shadow_b_port.get_metrics())

    def check_drift(self) -> Tuple[bool, float, str]:
        """
        Check for performance drift between live and frozen shadow.

        Returns:
            (is_drifted, drift_amount, message)
        """
        if self.bars_processed < 100:
            return False, 0.0, "Insufficient data"

        live_metrics = self.live_port.get_metrics()
        shadow_a_metrics = self.shadow_a_port.get_metrics()

        # Compare Sharpe ratios
        live_sharpe = live_metrics["sharpe"]
        frozen_sharpe = shadow_a_metrics["sharpe"]

        if frozen_sharpe > 0:
            drift = (frozen_sharpe - live_sharpe) / frozen_sharpe
        else:
            drift = 0.0

        is_drifted = drift > self.drift_threshold

        if is_drifted:
            msg = f"{self.asset_class.value}: Drift {drift*100:.1f}% (Live Sharpe={live_sharpe:.2f}, Frozen={frozen_sharpe:.2f})"
            self.logger.warning(msg)
            return True, drift, msg

        return False, drift, ""

    def check_promotion(self) -> Tuple[bool, str]:
        """
        Check if Shadow B should be promoted to Live.

        Returns:
            (should_promote, message)
        """
        if self.bars_processed < 200:
            return False, "Insufficient data"

        live_metrics = self.live_port.get_metrics()
        shadow_b_metrics = self.shadow_b_port.get_metrics()

        # Compare multiple metrics
        live_score = (
            live_metrics["sharpe"] * 0.3 +
            live_metrics["sortino"] * 0.2 +
            live_metrics["calmar"] * 0.3 +
            live_metrics["profit_factor"] * 0.2
        )

        shadow_b_score = (
            shadow_b_metrics["sharpe"] * 0.3 +
            shadow_b_metrics["sortino"] * 0.2 +
            shadow_b_metrics["calmar"] * 0.3 +
            shadow_b_metrics["profit_factor"] * 0.2
        )

        if live_score > 0:
            improvement = (shadow_b_score - live_score) / live_score
        else:
            improvement = shadow_b_score if shadow_b_score > 0 else 0

        should_promote = (
            improvement > self.promotion_threshold and
            self.shadow_b_validator.has_edge()
        )

        if should_promote:
            msg = f"{self.asset_class.value}: Shadow B promotion ready ({improvement*100:.1f}% improvement)"
            self.logger.info(msg)
            return True, msg

        return False, ""

    def promote_shadow_b(self):
        """Promote Shadow B to Live."""
        self.logger.info(f"{self.asset_class.value}: Promoting Shadow B to Live")

        # Current live becomes new frozen (Shadow A)
        self.shadow_a_agent = copy.deepcopy(self.live_agent)
        self.shadow_a_port = VirtualPortfolio(self.live_port.initial_capital)
        self.frozen_at_bar = self.bars_processed

        # Shadow B becomes live
        self.live_agent = copy.deepcopy(self.shadow_b_agent)

        # Create new Shadow B from new live
        self.shadow_b_agent = copy.deepcopy(self.live_agent)
        self.shadow_b_port.reset()

        self.promotion_count += 1

    def rollback_to_frozen(self):
        """Rollback Live to frozen Shadow A."""
        self.logger.warning(f"{self.asset_class.value}: Rolling back to frozen agent")

        # Restore from frozen
        self.live_agent = copy.deepcopy(self.shadow_a_agent)
        self.live_port.reset()

        # Create new Shadow B from restored
        self.shadow_b_agent = copy.deepcopy(self.live_agent)
        self.shadow_b_port.reset()

        self.rollback_count += 1

    def get_summary(self) -> Dict[str, Any]:
        """Get specialist summary."""
        return {
            "asset_class": self.asset_class.value,
            "bars_processed": self.bars_processed,
            "promotions": self.promotion_count,
            "rollbacks": self.rollback_count,
            "live": self.live_port.get_metrics(),
            "shadow_a": self.shadow_a_port.get_metrics(),
            "shadow_b": self.shadow_b_port.get_metrics(),
            "live_has_edge": self.live_validator.has_edge(),
            "shadow_b_has_edge": self.shadow_b_validator.has_edge(),
        }

    def generate_chart_data(self) -> Dict[str, Any]:
        """Generate data for visualization charts."""
        return {
            "asset_class": self.asset_class.value,
            "timestamps": list(range(len(self.live_port.equity_curve))),
            "live_equity": list(self.live_port.equity_curve),
            "shadow_a_equity": list(self.shadow_a_port.equity_curve),
            "shadow_b_equity": list(self.shadow_b_port.equity_curve),
        }


# =============================================================================
# PORTFOLIO HEALTH SCORE
# =============================================================================

class PortfolioHealthScore:
    """
    Self-Healing Portfolio Health Score Calculator.

    Composite score (0-100) based on four pillars:
    1. Return & Efficiency (Sharpe, Sortino, Omega, Calmar)
    2. Downside Risk & Resilience (Drawdown, Ulcer Index)
    3. Structural Stability (Correlation, Diversification)
    4. Behavioral Health (Edge decay, Agent adaptation)
    """

    def __init__(
        self,
        return_weight: float = 0.25,
        risk_weight: float = 0.30,
        structure_weight: float = 0.25,
        behavior_weight: float = 0.20,
        critical_penalty: float = 0.7,
    ):
        self.weights = {
            "return": return_weight,
            "risk": risk_weight,
            "structure": structure_weight,
            "behavior": behavior_weight,
        }
        self.critical_penalty = critical_penalty

        # Thresholds
        self.max_acceptable_dd = 0.25
        self.risk_calc = RiskMetrics()

    def calculate(
        self,
        specialists: List[DoppelgangerSpecialist],
        correlation_matrix: np.ndarray = None,
    ) -> tuple[float, dict[str, str]] | tuple[floating[Any], dict[str | Any, floating[Any] | str | float | int | Any]]:
        """
        Calculate portfolio health score from all specialists.

        Returns:
            (health_score 0-100, breakdown dict)
        """
        if not specialists:
            return 50.0, {"status": "No specialists"}

        # Aggregate metrics across specialists
        all_metrics = [s.live_port.get_metrics() for s in specialists]

        # Return score
        avg_sharpe = np.mean([m["sharpe"] for m in all_metrics])
        avg_sortino = np.mean([m["sortino"] for m in all_metrics])
        avg_calmar = np.mean([m["calmar"] for m in all_metrics])
        avg_omega = np.mean([m["omega"] for m in all_metrics])

        sharpe_score = min(max(avg_sharpe, 0) / 2.0, 1.0)
        sortino_score = min(max(avg_sortino, 0) / 3.0, 1.0)
        calmar_score = min(max(avg_calmar, 0) / 3.0, 1.0)
        omega_score = min(avg_omega / 3.0, 1.0)

        return_score = np.mean([sharpe_score, sortino_score, calmar_score, omega_score])

        # Risk score
        max_dd_across = max(m["max_drawdown"] for m in all_metrics)
        avg_ulcer = np.mean([m["ulcer_index"] for m in all_metrics])

        dd_score = max(0, 1 - max_dd_across / self.max_acceptable_dd)
        ulcer_score = max(0, 1 - avg_ulcer)

        risk_score = np.mean([dd_score, ulcer_score])

        # Structure score
        if correlation_matrix is not None and correlation_matrix.size > 0:
            triu_idx = np.triu_indices_from(correlation_matrix, k=1)
            avg_corr = np.mean(np.abs(correlation_matrix[triu_idx]))
            structure_score = 1 - avg_corr
        else:
            structure_score = 0.5  # Neutral

        # Behavior score
        live_edges = [s.live_validator.has_edge() for s in specialists]
        shadow_edges = [s.shadow_b_validator.has_edge() for s in specialists]

        edge_pct = sum(live_edges) / len(live_edges)
        adaptation_pct = sum(shadow_edges) / len(shadow_edges)

        behavior_score = np.mean([edge_pct, adaptation_pct])

        # Composite
        composite = (
            self.weights["return"] * return_score +
            self.weights["risk"] * risk_score +
            self.weights["structure"] * structure_score +
            self.weights["behavior"] * behavior_score
        )

        # Critical penalty
        if risk_score < 0.4 or structure_score < 0.5:
            composite *= self.critical_penalty

        health_score = np.clip(composite * 100, 0, 100)

        # Status
        if health_score > 80:
            status = "Healthy"
        elif health_score > 60:
            status = "Degrading"
        elif health_score > 40:
            status = "Critical"
        else:
            status = "Emergency"

        breakdown = {
            "health_score": health_score,
            "status": status,
            "return_score": return_score * 100,
            "risk_score": risk_score * 100,
            "structure_score": structure_score * 100,
            "behavior_score": behavior_score * 100,
            "avg_sharpe": avg_sharpe,
            "avg_calmar": avg_calmar,
            "max_drawdown": max_dd_across,
            "specialists_with_edge": sum(live_edges),
            "total_specialists": len(specialists),
        }

        return health_score, breakdown

    def get_healing_actions(self, health_score: float) -> List[str]:
        """Get recommended self-healing actions based on health score."""
        if health_score > 80:
            return ["Normal operation", "Optional micro-rebalancing"]
        elif health_score > 60:
            return [
                "Reduce overall risk budget by 20-50%",
                "Increase monitoring frequency",
                "Consider partial specialist retraining",
            ]
        elif health_score > 40:
            return [
                "Move 50% to defensive allocation",
                "Force Shadow B promotions if edge detected",
                "Mandatory specialist retraining",
                "Exit correlated positions",
            ]
        else:
            return [
                "EMERGENCY: Full defensive mode",
                "Pause all alpha strategies",
                "Emergency retraining on recent regime",
                "Alert human oversight",
            ]


# =============================================================================
# DEMO
# =============================================================================

def demo_specialist_system():
    """Demonstrate the specialist system."""
    print("=" * 80)
    print("  SPECIALIST AGENT + DOPPELGÄNGER SYSTEM DEMO")
    print("=" * 80)

    # Mock agent
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
            pass

    # Create specialists for each asset class
    specialists = []
    for asset_class in [AssetClass.FOREX, AssetClass.CRYPTO, AssetClass.INDEX]:
        agent = MockAgent()
        specialist = DoppelgangerSpecialist(
            asset_class=asset_class,
            agent=agent,
            initial_capital=100000.0,
        )
        specialists.append(specialist)

    print(f"\n[CREATED] {len(specialists)} specialists")

    # Simulate trading
    print("\n[SIMULATING] 500 bars of trading...")
    for bar in range(500):
        state = np.random.randn(64)
        current_price = 100 + np.random.randn() * 2

        for specialist in specialists:
            action, info = specialist.select_action(state, epsilon=0.1)
            reward = np.random.randn() * 10
            next_state = np.random.randn(64)

            specialist.update(
                state=state,
                action=action,
                reward=reward,
                next_state=next_state,
                done=(bar == 499),
                current_price=current_price,
            )

    # Check drift and promotion
    print("\n[DRIFT DETECTION]")
    for specialist in specialists:
        is_drifted, drift, msg = specialist.check_drift()
        print(f"  {specialist.asset_class.value}: Drift={drift*100:.1f}% {'⚠️' if is_drifted else '✓'}")

    print("\n[PROMOTION CHECK]")
    for specialist in specialists:
        should_promote, msg = specialist.check_promotion()
        print(f"  {specialist.asset_class.value}: {'Ready for promotion' if should_promote else 'Not ready'}")

    # Calculate health score
    health_calc = PortfolioHealthScore()
    health_score, breakdown = health_calc.calculate(specialists)

    print(f"\n[PORTFOLIO HEALTH SCORE]")
    print(f"  Score: {health_score:.1f}/100 - {breakdown['status']}")
    print(f"  Return Score: {breakdown['return_score']:.1f}")
    print(f"  Risk Score: {breakdown['risk_score']:.1f}")
    print(f"  Structure Score: {breakdown['structure_score']:.1f}")
    print(f"  Behavior Score: {breakdown['behavior_score']:.1f}")

    # Healing actions
    actions = health_calc.get_healing_actions(health_score)
    print(f"\n[RECOMMENDED ACTIONS]")
    for action in actions:
        print(f"  • {action}")

    # Per-specialist summary
    print(f"\n[SPECIALIST SUMMARIES]")
    for specialist in specialists:
        summary = specialist.get_summary()
        live = summary["live"]
        print(f"\n  {summary['asset_class']}:")
        print(f"    Sharpe: {live['sharpe']:.2f} | Calmar: {live['calmar']:.2f} | MaxDD: {live['max_drawdown']*100:.1f}%")
        print(f"    Trades: {live['trade_count']} | Win Rate: {live['win_rate']*100:.1f}%")
        print(f"    Has Edge: {'✓' if summary['live_has_edge'] else '✗'}")

    print("\n" + "=" * 80)
    print("  DEMO COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    demo_specialist_system()
