"""
Canonical backtesting trade types.

This module restores the lightweight ``Trade`` / ``TradeDirection`` API
expected by ``kinetra.backtesting.__init__`` and legacy imports.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Optional


class TradeDirection(Enum):
    """Trade direction enumeration."""

    LONG = "long"
    SHORT = "short"


@dataclass
class Trade:
    """Minimal canonical trade record used across backtesting adapters."""

    trade_id: int
    symbol: str
    direction: TradeDirection
    lots: float
    entry_time: datetime
    entry_price: float
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    spread_cost: float = 0.0
    commission: float = 0.0
    slippage: float = 0.0
    swap_cost: float = 0.0
    gross_pnl: float = 0.0
    net_pnl: float = 0.0

    @property
    def is_closed(self) -> bool:
        return self.exit_price is not None

    @property
    def total_cost(self) -> float:
        return self.spread_cost + self.commission + self.slippage + abs(self.swap_cost)

    @property
    def holding_time(self) -> Optional[timedelta]:
        if self.exit_time is None:
            return None
        return self.exit_time - self.entry_time
