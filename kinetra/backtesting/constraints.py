"""
MT5 Constraints
===============

MetaTrader 5 execution constraints for realistic backtesting.
Extracted from realistic_backtester.py.

Features:
- Freeze zones (no SL/TP modification near market price)
- Stops level validation
- MT5 error code simulation
- Market hours enforcement
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class MT5ErrorCode(Enum):
    """MT5 error codes."""
    SUCCESS = 0
    INVALID_STOPS = 130  # Invalid stops
    FREEZE_LEVEL = 132   # Market is changing (freeze zone)
    TRADE_DISABLED = 133 # Trading disabled
    NOT_ENOUGH_MONEY = 134
    

@dataclass
class MT5Constraints:
    """
    MT5 trading constraints.
    
    From realistic_backtester.py
    """
    freeze_distance_pips: float = 5.0  # Minimum distance from market for modifications
    stops_level_pips: float = 10.0     # Minimum distance for SL/TP
    max_slippage_pips: float = 3.0     # Maximum allowed slippage
    enforce_market_hours: bool = True  # Only trade during market hours
    
    def validate_stop_loss(
        self,
        market_price: float,
        stop_loss: float,
        direction: int,  # 1 for buy, -1 for sell
        pip_value: float = 0.0001
    ) -> tuple[bool, Optional[MT5ErrorCode]]:
        """
        Validate stop loss placement.
        
        Args:
            market_price: Current market price
            stop_loss: Proposed stop loss
            direction: Trade direction (1=buy, -1=sell)
            pip_value: Pip value
            
        Returns:
            (is_valid, error_code)
        """
        distance_pips = abs(market_price - stop_loss) / pip_value
        
        # Check minimum distance
        if distance_pips < self.stops_level_pips:
            return False, MT5ErrorCode.INVALID_STOPS
            
        # Check freeze zone
        if distance_pips < self.freeze_distance_pips:
            return False, MT5ErrorCode.FREEZE_LEVEL
            
        return True, MT5ErrorCode.SUCCESS
        
    def validate_take_profit(
        self,
        market_price: float,
        take_profit: float,
        direction: int,
        pip_value: float = 0.0001
    ) -> tuple[bool, Optional[MT5ErrorCode]]:
        """Validate take profit placement."""
        # Same logic as stop loss
        return self.validate_stop_loss(market_price, take_profit, direction, pip_value)
        
    def can_modify_position(
        self,
        market_price: float,
        current_sl: float,
        new_sl: float,
        direction: int,
        pip_value: float = 0.0001
    ) -> tuple[bool, Optional[MT5ErrorCode]]:
        """
        Check if position can be modified (freeze zone).
        
        Args:
            market_price: Current price
            current_sl: Current stop loss
            new_sl: New stop loss
            direction: Trade direction
            pip_value: Pip value
            
        Returns:
            (can_modify, error_code)
        """
        # Check if new SL is in freeze zone
        distance_pips = abs(market_price - new_sl) / pip_value
        
        if distance_pips < self.freeze_distance_pips:
            return False, MT5ErrorCode.FREEZE_LEVEL
            
        # Check if new SL meets minimum distance
        if distance_pips < self.stops_level_pips:
            return False, MT5ErrorCode.INVALID_STOPS
            
        return True, MT5ErrorCode.SUCCESS
