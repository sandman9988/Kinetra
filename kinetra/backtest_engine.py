"""
Backtest Engine - Legacy Classes for Backward Compatibility

This module provides backward compatibility for code that still imports
from backtest_engine. The main backtesting functionality has been moved
to realistic_backtester.py.

For new code, import from realistic_backtester instead.
"""

from dataclasses import dataclass
from typing import List

# Re-export classes from realistic_backtester for backward compatibility
from .realistic_backtester import Trade, TradeDirection, BacktestResult

# Legacy classes that are still used by portfolio_backtest.py
from .symbol_spec import SymbolSpec


@dataclass
class OpenPosition:
    """
    Legacy class representing an open position in a portfolio.
    
    This is a minimal implementation for backward compatibility with
    portfolio_backtest.py.
    """
    trade: Trade
    symbol_spec: SymbolSpec
    current_price: float
    
    def update(self, price: float):
        """Update the current price of the position."""
        self.current_price = price


@dataclass
class PortfolioState:
    """
    Legacy class representing portfolio state.
    
    This is a minimal implementation for backward compatibility with
    portfolio_backtest.py.
    """
    balance: float = 0.0
    equity: float = 0.0
    used_margin: float = 0.0
    free_margin: float = 0.0
    margin_level: float = 0.0
    
    def update(self, balance: float, open_positions: List[OpenPosition]):
        """Update portfolio state based on balance and open positions."""
        self.balance = balance
        self.used_margin = 0.0
        self.equity = balance
        
        # Calculate used margin and floating P&L from open positions
        for position in open_positions:
            # Simple margin calculation (actual margin depends on leverage)
            # This is a placeholder - real calculation would use symbol_spec
            trade = position.trade
            if hasattr(trade, 'lots'):
                # Estimate margin (100:1 leverage as default)
                self.used_margin += trade.lots * position.current_price * 100000 / 100
            
            # Calculate floating P&L
            if hasattr(trade, 'direction') and hasattr(trade, 'entry_price'):
                if trade.direction == 1:  # Long
                    pnl = (position.current_price - trade.entry_price) * (trade.lots if hasattr(trade, 'lots') else 1.0) * 100000
                else:  # Short  
                    pnl = (trade.entry_price - position.current_price) * (trade.lots if hasattr(trade, 'lots') else 1.0) * 100000
                self.equity += pnl
        
        self.free_margin = self.equity - self.used_margin
        
        # Calculate margin level (percentage)
        if self.used_margin > 0:
            self.margin_level = (self.equity / self.used_margin) * 100
        else:
            self.margin_level = float('inf')


__all__ = [
    'Trade',
    'TradeDirection',
    'BacktestResult',
    'OpenPosition',
    'PortfolioState',
]
