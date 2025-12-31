"""
Backtest Engine Compatibility Module

This module provides compatibility classes for code that still imports from backtest_engine.
The main backtesting functionality has been moved to realistic_backtester.py.

These classes are maintained for backward compatibility with existing code.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .symbol_spec import SymbolSpec


class TradeDirection(Enum):
    """Trade direction enumeration."""

    LONG = "long"
    SHORT = "short"


@dataclass
class Trade:
    """Individual trade record with comprehensive tracking."""

    trade_id: int
    symbol: str
    direction: TradeDirection
    lots: float
    entry_time: datetime
    entry_price: float
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None

    # Costs
    spread_cost: float = 0.0
    commission: float = 0.0
    slippage: float = 0.0
    swap_cost: float = 0.0

    # Results
    gross_pnl: float = 0.0
    net_pnl: float = 0.0

    # Physics state at entry
    energy_at_entry: float = 0.0
    regime_at_entry: str = ""

    # Trade quality metrics
    mfe: float = 0.0  # Maximum Favorable Excursion
    mae: float = 0.0  # Maximum Adverse Excursion

    @property
    def is_closed(self) -> bool:
        """Check if trade is closed."""
        return self.exit_price is not None

    @property
    def total_cost(self) -> float:
        """Calculate total transaction costs."""
        return self.spread_cost + self.commission + self.slippage + abs(self.swap_cost)

    @property
    def holding_time(self) -> Optional[timedelta]:
        """Calculate holding time for closed trades."""
        if self.exit_time and self.entry_time:
            return self.exit_time - self.entry_time
        return None

    @property
    def price_captured(self) -> float:
        """Calculate price difference captured."""
        if not self.is_closed:
            return 0.0
        if self.direction == TradeDirection.LONG:
            return self.exit_price - self.entry_price
        else:
            return self.entry_price - self.exit_price

    @property
    def mfe_efficiency(self) -> float:
        """MFE efficiency: how much of MFE was captured as profit (0-1)."""
        if self.mfe > 0:
            return max(0, min(1.0, self.price_captured / self.mfe))
        return 0.0

    @property
    def mae_efficiency(self) -> float:
        """MAE efficiency: how well adverse excursion was limited (0-1)."""
        if self.mfe > 0:
            return max(0, 1 - self.mae / self.mfe)
        return 0.0


@dataclass
class BacktestResult:
    """Complete backtest results with comprehensive metrics."""

    # Trade list
    trades: List[Trade]

    # Summary metrics
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0

    # P&L
    gross_profit: float = 0.0
    gross_loss: float = 0.0
    total_gross_pnl: float = 0.0
    total_costs: float = 0.0
    total_net_pnl: float = 0.0

    # Cost breakdown
    total_spread_cost: float = 0.0
    total_commission: float = 0.0
    total_slippage: float = 0.0
    total_swap_cost: float = 0.0

    # Risk metrics
    max_drawdown: float = 0.0
    max_drawdown_pct: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    min_margin_level: float = float("inf")

    # CVaR metrics
    cvar_95: float = 0.0
    cvar_99: float = 0.0

    # Kinetra-specific metrics
    omega_ratio: float = 0.0
    z_factor: float = 0.0
    energy_captured_pct: float = 0.0
    mfe_capture_pct: float = 0.0

    # Equity curve
    equity_curve: Optional[pd.Series] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "win_rate": self.win_rate,
            "gross_profit": self.gross_profit,
            "gross_loss": self.gross_loss,
            "total_gross_pnl": self.total_gross_pnl,
            "total_costs": self.total_costs,
            "total_net_pnl": self.total_net_pnl,
            "cost_breakdown": {
                "spread": self.total_spread_cost,
                "commission": self.total_commission,
                "slippage": self.total_slippage,
                "swap": self.total_swap_cost,
            },
            "max_drawdown": self.max_drawdown,
            "max_drawdown_pct": self.max_drawdown_pct,
            "sharpe_ratio": self.sharpe_ratio,
            "omega_ratio": self.omega_ratio,
            "z_factor": self.z_factor,
            "energy_captured_pct": self.energy_captured_pct,
        }


@dataclass
class OpenPosition:
    """Represents an open trading position."""

    trade: Trade
    symbol_spec: SymbolSpec
    current_price: float
    
    def update(self, price: float):
        """Update current price for the position."""
        self.current_price = price


@dataclass
class PortfolioState:
    """Current state of the portfolio."""

    equity: float = 0.0
    balance: float = 0.0
    used_margin: float = 0.0
    free_margin: float = 0.0
    margin_level: float = 0.0
    open_positions_count: int = 0
    
    def update(self, balance: float, open_positions: List[OpenPosition]):
        """Update portfolio state based on current balance and positions."""
        self.balance = balance
        self.open_positions_count = len(open_positions)
        
        # Calculate used margin
        self.used_margin = 0.0
        unrealized_pnl = 0.0
        
        for position in open_positions:
            trade = position.trade
            spec = position.symbol_spec
            
            # Calculate margin requirement
            position_value = spec.contract_size * trade.lots * position.current_price
            margin = position_value * spec.margin_initial
            self.used_margin += margin
            
            # Calculate unrealized P&L
            if trade.direction == TradeDirection.LONG:
                pnl = (position.current_price - trade.entry_price) * spec.contract_size * trade.lots
            else:
                pnl = (trade.entry_price - position.current_price) * spec.contract_size * trade.lots
            unrealized_pnl += pnl
        
        # Calculate equity and margin level
        self.equity = self.balance + unrealized_pnl
        self.free_margin = self.equity - self.used_margin
        
        if self.used_margin > 0:
            self.margin_level = (self.equity / self.used_margin) * 100
        else:
            self.margin_level = float('inf')


class BacktestEngine:
    """
    Legacy BacktestEngine class for backward compatibility.
    
    NOTE: This class is deprecated. Use RealisticBacktester instead.
    The realistic backtester provides MT5-accurate constraints and better friction modeling.
    """
    
    def __init__(self, *args, **kwargs):
        raise DeprecationWarning(
            "BacktestEngine is deprecated. Please use RealisticBacktester instead. "
            "See kinetra/realistic_backtester.py for the new implementation."
        )
