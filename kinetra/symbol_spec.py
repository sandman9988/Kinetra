"""
Symbol Specification and Trading Costs

Complete instrument cost modeling for accurate backtesting:
- Spread (bid/ask)
- Commission (per lot or per trade)
- Swap rates (long/short)
- Swap days (triple swap handling)
- Contract specifications (lot size, tick value, margin)
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Optional, List, Tuple
from enum import Enum
from datetime import datetime, time
import json


class SwapType(Enum):
    """Swap calculation type."""
    POINTS = "points"           # Swap in points
    MONEY = "money"             # Swap in account currency
    INTEREST = "interest"       # Swap as annual interest rate
    MARGIN_CURRENCY = "margin"  # Swap in margin currency


class CommissionType(Enum):
    """Commission calculation type."""
    PER_LOT = "per_lot"         # Fixed $ per lot
    PER_DEAL = "per_deal"       # Fixed $ per trade
    PERCENTAGE = "percentage"   # % of trade value


@dataclass
class SwapSpec:
    """Swap/rollover specification."""
    long_rate: float = 0.0      # Swap for long positions
    short_rate: float = 0.0     # Swap for short positions
    swap_type: SwapType = SwapType.POINTS
    triple_swap_day: int = 3    # Wednesday = 3 (Mon=1, Sun=7)

    def calculate_swap(
        self,
        position_type: str,  # "long" or "short"
        lots: float,
        contract_size: float,
        tick_value: float,
        days_held: int = 1,
        day_of_week: int = 1
    ) -> float:
        """
        Calculate swap cost for holding position overnight.

        Returns negative value for cost, positive for credit.
        """
        rate = self.long_rate if position_type == "long" else self.short_rate

        # Apply triple swap on triple swap day
        multiplier = days_held
        if day_of_week == self.triple_swap_day:
            multiplier = 3  # Weekend rollover

        if self.swap_type == SwapType.POINTS:
            # Swap in points * tick value * lots
            return rate * tick_value * lots * multiplier
        elif self.swap_type == SwapType.MONEY:
            # Direct money value per lot
            return rate * lots * multiplier
        elif self.swap_type == SwapType.INTEREST:
            # Annual interest rate converted to daily
            # This is simplified - real calculation depends on notional
            daily_rate = rate / 365
            return daily_rate * lots * contract_size * multiplier
        else:
            return rate * lots * multiplier


@dataclass
class CommissionSpec:
    """Commission specification."""
    rate: float = 0.0
    commission_type: CommissionType = CommissionType.PER_LOT
    minimum: float = 0.0        # Minimum commission per trade

    def calculate_commission(
        self,
        lots: float,
        trade_value: float,
        is_round_trip: bool = False
    ) -> float:
        """Calculate commission for a trade."""
        multiplier = 2 if is_round_trip else 1

        if self.commission_type == CommissionType.PER_LOT:
            commission = self.rate * lots * multiplier
        elif self.commission_type == CommissionType.PER_DEAL:
            commission = self.rate * multiplier
        elif self.commission_type == CommissionType.PERCENTAGE:
            commission = trade_value * (self.rate / 100) * multiplier
        else:
            commission = 0.0

        return max(commission, self.minimum * multiplier)


@dataclass
class SymbolSpec:
    """
    Complete symbol specification for accurate cost modeling.

    All costs that affect P&L:
    - Spread (entry/exit slippage)
    - Commission (broker fees)
    - Swap (overnight financing)
    - Slippage (execution quality)
    """
    # Basic info
    symbol: str
    description: str = ""
    base_currency: str = ""
    quote_currency: str = ""

    # Contract specifications
    contract_size: float = 100000.0    # Units per lot (forex = 100k)
    tick_size: float = 0.00001         # Minimum price movement
    tick_value: float = 1.0            # Value per tick per lot in account currency
    digits: int = 5                     # Price decimal places

    # Margin requirements
    margin_initial: float = 0.01       # Initial margin as % (1% = 100:1 leverage)
    margin_maintenance: float = 0.005  # Maintenance margin %
    margin_hedged: float = 0.5         # Margin for hedged positions (50% = half)

    # Lot constraints
    volume_min: float = 0.01           # Minimum lot size
    volume_max: float = 100.0          # Maximum lot size
    volume_step: float = 0.01          # Lot size increment

    # Spread
    spread_points: float = 10.0        # Typical spread in points
    spread_variable: bool = True       # Is spread variable?
    spread_min: float = 5.0            # Minimum spread
    spread_max: float = 50.0           # Maximum spread (news/low liquidity)

    # Commission
    commission: CommissionSpec = field(default_factory=CommissionSpec)

    # Swap
    swap: SwapSpec = field(default_factory=SwapSpec)

    # Execution
    slippage_avg: float = 0.5          # Average slippage in points
    slippage_max: float = 5.0          # Max slippage in points

    # Trading hours (simplified)
    trading_hours_start: time = time(0, 0)
    trading_hours_end: time = time(23, 59)

    def spread_cost(self, lots: float, price: float) -> float:
        """Calculate spread cost for entry."""
        spread_value = self.spread_points * self.tick_size
        return spread_value * lots * self.contract_size

    def total_entry_cost(self, lots: float, price: float) -> float:
        """Total cost to enter a position."""
        spread = self.spread_cost(lots, price)
        commission = self.commission.calculate_commission(
            lots,
            lots * self.contract_size * price
        )
        slippage = self.slippage_avg * self.tick_value * lots
        return spread + commission + slippage

    def total_exit_cost(self, lots: float, price: float) -> float:
        """Total cost to exit a position."""
        # No spread on exit (already paid on entry via bid/ask)
        commission = self.commission.calculate_commission(
            lots,
            lots * self.contract_size * price
        )
        slippage = self.slippage_avg * self.tick_value * lots
        return commission + slippage

    def holding_cost(
        self,
        position_type: str,
        lots: float,
        days: int,
        day_of_week: int = 1
    ) -> float:
        """Calculate overnight holding cost."""
        return self.swap.calculate_swap(
            position_type,
            lots,
            self.contract_size,
            self.tick_value,
            days,
            day_of_week
        )

    def round_trip_cost(
        self,
        lots: float,
        entry_price: float,
        exit_price: float,
        position_type: str = "long",
        holding_days: int = 0
    ) -> Dict[str, float]:
        """
        Calculate complete round-trip costs.

        Returns breakdown of all costs.
        """
        entry_cost = self.total_entry_cost(lots, entry_price)
        exit_cost = self.total_exit_cost(lots, exit_price)

        # Holding costs
        swap_cost = 0.0
        if holding_days > 0:
            for day in range(holding_days):
                swap_cost += self.holding_cost(
                    position_type,
                    lots,
                    1,
                    (day % 7) + 1  # Simplified day rotation
                )

        total = entry_cost + exit_cost + abs(swap_cost)

        return {
            "spread_cost": self.spread_cost(lots, entry_price),
            "commission_entry": self.commission.calculate_commission(lots, lots * self.contract_size * entry_price),
            "commission_exit": self.commission.calculate_commission(lots, lots * self.contract_size * exit_price),
            "slippage": self.slippage_avg * self.tick_value * lots * 2,
            "swap_cost": swap_cost,
            "total_cost": total,
            "cost_in_points": total / (self.tick_value * lots) if lots > 0 else 0
        }

    def to_dict(self) -> Dict:
        """Serialize to dictionary."""
        return {
            "symbol": self.symbol,
            "description": self.description,
            "contract_size": self.contract_size,
            "tick_size": self.tick_size,
            "tick_value": self.tick_value,
            "digits": self.digits,
            "spread_points": self.spread_points,
            "commission_rate": self.commission.rate,
            "commission_type": self.commission.commission_type.value,
            "swap_long": self.swap.long_rate,
            "swap_short": self.swap.short_rate,
            "swap_type": self.swap.swap_type.value,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'SymbolSpec':
        """Deserialize from dictionary."""
        commission = CommissionSpec(
            rate=data.get("commission_rate", 0),
            commission_type=CommissionType(data.get("commission_type", "per_lot"))
        )
        swap = SwapSpec(
            long_rate=data.get("swap_long", 0),
            short_rate=data.get("swap_short", 0),
            swap_type=SwapType(data.get("swap_type", "points"))
        )
        return cls(
            symbol=data["symbol"],
            description=data.get("description", ""),
            contract_size=data.get("contract_size", 100000),
            tick_size=data.get("tick_size", 0.00001),
            tick_value=data.get("tick_value", 1.0),
            digits=data.get("digits", 5),
            spread_points=data.get("spread_points", 10),
            commission=commission,
            swap=swap,
        )


# ============================================
# Pre-defined Symbol Specifications
# ============================================

# Forex majors (typical ECN broker)
BTCUSD_SPEC = SymbolSpec(
    symbol="BTCUSD",
    description="Bitcoin vs US Dollar",
    base_currency="BTC",
    quote_currency="USD",
    contract_size=1.0,              # 1 BTC per lot
    tick_size=0.01,                 # $0.01 minimum move
    tick_value=0.01,                # $0.01 per tick per lot
    digits=2,
    spread_points=5000,             # $50 typical spread
    spread_min=2000,
    spread_max=20000,
    commission=CommissionSpec(rate=0, commission_type=CommissionType.PER_LOT),
    swap=SwapSpec(long_rate=-15.0, short_rate=-15.0, swap_type=SwapType.POINTS),
    slippage_avg=100,               # $1 avg slippage
)

EURUSD_SPEC = SymbolSpec(
    symbol="EURUSD",
    description="Euro vs US Dollar",
    base_currency="EUR",
    quote_currency="USD",
    contract_size=100000,
    tick_size=0.00001,
    tick_value=1.0,                 # $1 per pip for 1 lot
    digits=5,
    spread_points=10,               # 1 pip typical
    spread_min=5,
    spread_max=30,
    commission=CommissionSpec(rate=3.5, commission_type=CommissionType.PER_LOT),
    swap=SwapSpec(long_rate=-6.5, short_rate=1.2, swap_type=SwapType.POINTS),
)

XAUUSD_SPEC = SymbolSpec(
    symbol="XAUUSD",
    description="Gold vs US Dollar",
    base_currency="XAU",
    quote_currency="USD",
    contract_size=100,              # 100 oz per lot
    tick_size=0.01,
    tick_value=1.0,                 # $1 per tick for 1 lot
    digits=2,
    spread_points=30,               # 30 cents typical
    spread_min=15,
    spread_max=100,
    commission=CommissionSpec(rate=0, commission_type=CommissionType.PER_LOT),
    swap=SwapSpec(long_rate=-25.0, short_rate=5.0, swap_type=SwapType.POINTS),
)

COPPER_SPEC = SymbolSpec(
    symbol="COPPER-C",
    description="Copper CFD",
    base_currency="COPPER",
    quote_currency="USD",
    contract_size=25000,            # 25000 lbs per lot
    tick_size=0.0001,
    tick_value=2.5,                 # $2.50 per tick for 1 lot
    digits=4,
    spread_points=30,
    spread_min=15,
    spread_max=80,
    commission=CommissionSpec(rate=0, commission_type=CommissionType.PER_LOT),
    swap=SwapSpec(long_rate=-8.0, short_rate=-4.0, swap_type=SwapType.POINTS),
)

# Registry of default specs
DEFAULT_SPECS: Dict[str, SymbolSpec] = {
    "BTCUSD": BTCUSD_SPEC,
    "EURUSD": EURUSD_SPEC,
    "XAUUSD": XAUUSD_SPEC,
    "COPPER-C": COPPER_SPEC,
}


def get_symbol_spec(symbol: str) -> Optional[SymbolSpec]:
    """Get symbol specification from registry."""
    return DEFAULT_SPECS.get(symbol.upper())


def fetch_mt5_symbol_spec(symbol: str) -> Optional[SymbolSpec]:
    """
    Fetch symbol specification from MT5 terminal.

    Requires MT5 to be running and connected.
    """
    try:
        import MetaTrader5 as mt5

        if not mt5.initialize():
            return None

        info = mt5.symbol_info(symbol)
        if info is None:
            return None

        # Get swap info
        swap = SwapSpec(
            long_rate=info.swap_long,
            short_rate=info.swap_short,
            swap_type=SwapType.POINTS,  # MT5 typically uses points
            triple_swap_day=info.swap_rollover3days + 1  # MT5 uses 0-indexed
        )

        # Commission (if available)
        # Note: MT5 doesn't always expose commission in symbol_info
        commission = CommissionSpec(rate=0, commission_type=CommissionType.PER_LOT)

        spec = SymbolSpec(
            symbol=info.name,
            description=info.description,
            base_currency=info.currency_base,
            quote_currency=info.currency_profit,
            contract_size=info.trade_contract_size,
            tick_size=info.trade_tick_size,
            tick_value=info.trade_tick_value,
            digits=info.digits,
            spread_points=info.spread,
            volume_min=info.volume_min,
            volume_max=info.volume_max,
            volume_step=info.volume_step,
            margin_initial=info.margin_initial,
            swap=swap,
            commission=commission,
        )

        return spec

    except ImportError:
        return None
    except Exception as e:
        print(f"Error fetching MT5 symbol info: {e}")
        return None


# Example usage and testing
if __name__ == "__main__":
    # Test BTCUSD costs
    btc = BTCUSD_SPEC

    print(f"=== {btc.symbol} Cost Analysis ===")
    print(f"Contract size: {btc.contract_size} BTC")
    print(f"Typical spread: ${btc.spread_points * btc.tick_size:.2f}")

    # Calculate round trip for 0.1 lot, held 3 days
    costs = btc.round_trip_cost(
        lots=0.1,
        entry_price=95000,
        exit_price=96000,
        position_type="long",
        holding_days=3
    )

    print(f"\nRound-trip costs (0.1 lot, 3 days):")
    for k, v in costs.items():
        print(f"  {k}: ${v:.2f}")

    # Gross P&L
    gross_pnl = (96000 - 95000) * 0.1  # $100 profit
    net_pnl = gross_pnl - costs["total_cost"]
    print(f"\nGross P&L: ${gross_pnl:.2f}")
    print(f"Net P&L: ${net_pnl:.2f}")
    print(f"Cost as % of gross: {(costs['total_cost']/gross_pnl)*100:.1f}%")
