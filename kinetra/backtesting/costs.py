"""
Cost Models
===========

Trading cost models (spread, commission, slippage, swap).
Consolidated from multiple implementations.

Features:
- Fixed spread model
- Dynamic spread model (per-bar spreads)
- Commission calculation
- Slippage modeling
- Swap/rollover costs
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import pandas as pd


@dataclass
class TradingCosts:
    """Trading costs for a single trade."""
    spread_cost: float = 0.0
    commission: float = 0.0
    slippage: float = 0.0
    swap: float = 0.0
    
    @property
    def total(self) -> float:
        """Total cost."""
        return self.spread_cost + self.commission + self.slippage + self.swap


class CostModel(ABC):
    """Base cost model."""
    
    @abstractmethod
    def calculate_entry_cost(
        self,
        symbol: str,
        volume: float,
        price: float,
        **kwargs
    ) -> TradingCosts:
        """Calculate entry costs."""
        pass
        
    @abstractmethod
    def calculate_exit_cost(
        self,
        symbol: str,
        volume: float,
        price: float,
        **kwargs
    ) -> TradingCosts:
        """Calculate exit costs."""
        pass


class FixedCostModel(CostModel):
    """
    Fixed spread cost model.
    
    From: backtest_engine.py, physics_backtester.py
    """
    
    def __init__(
        self,
        spread_pips: float = 2.0,
        commission_per_lot: float = 0.0,
        slippage_pips: float = 0.5
    ):
        """
        Initialize fixed cost model.
        
        Args:
            spread_pips: Fixed spread in pips
            commission_per_lot: Commission per lot
            slippage_pips: Expected slippage in pips
        """
        self.spread_pips = spread_pips
        self.commission_per_lot = commission_per_lot
        self.slippage_pips = slippage_pips
        
    def calculate_entry_cost(
        self,
        symbol: str,
        volume: float,
        price: float,
        pip_value: float = 0.0001
    ) -> TradingCosts:
        """Calculate entry costs."""
        # Spread cost (half spread on entry)
        spread_cost = (self.spread_pips / 2) * pip_value * volume
        
        # Commission
        commission = self.commission_per_lot * volume
        
        # Slippage
        slippage = self.slippage_pips * pip_value * volume
        
        return TradingCosts(
            spread_cost=spread_cost,
            commission=commission,
            slippage=slippage
        )
        
    def calculate_exit_cost(
        self,
        symbol: str,
        volume: float,
        price: float,
        pip_value: float = 0.0001
    ) -> TradingCosts:
        """Calculate exit costs."""
        # Same as entry for fixed model
        return self.calculate_entry_cost(symbol, volume, price, pip_value)


class DynamicCostModel(CostModel):
    """
    Dynamic spread cost model (per-bar spreads).
    
    From: realistic_backtester.py
    
    Uses actual spread data from OHLCV if available.
    """
    
    def __init__(
        self,
        commission_per_lot: float = 0.0,
        slippage_multiplier: float = 1.5  # Slippage as multiple of spread
    ):
        """
        Initialize dynamic cost model.
        
        Args:
            commission_per_lot: Commission per lot
            slippage_multiplier: Slippage as multiple of current spread
        """
        self.commission_per_lot = commission_per_lot
        self.slippage_multiplier = slippage_multiplier
        
    def calculate_entry_cost(
        self,
        symbol: str,
        volume: float,
        price: float,
        spread: Optional[float] = None,
        pip_value: float = 0.0001
    ) -> TradingCosts:
        """
        Calculate entry costs using actual spread.
        
        Args:
            spread: Actual spread at entry (in pips)
            If None, uses a default
        """
        if spread is None:
            spread = 2.0  # Default fallback
            
        # Spread cost (half spread on entry)
        spread_cost = (spread / 2) * pip_value * volume
        
        # Commission
        commission = self.commission_per_lot * volume
        
        # Slippage (proportional to spread)
        slippage = spread * self.slippage_multiplier * pip_value * volume
        
        return TradingCosts(
            spread_cost=spread_cost,
            commission=commission,
            slippage=slippage
        )
        
    def calculate_exit_cost(
        self,
        symbol: str,
        volume: float,
        price: float,
        spread: Optional[float] = None,
        pip_value: float = 0.0001
    ) -> TradingCosts:
        """Calculate exit costs."""
        return self.calculate_entry_cost(symbol, volume, price, spread, pip_value)
