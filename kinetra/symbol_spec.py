"""
Symbol Specification and Trading Costs

Complete instrument cost modeling for accurate backtesting:
- Spread (bid/ask)
- Commission (per lot or per trade)
- Swap rates (long/short)
- Swap days (triple swap handling)
- Contract specifications (lot size, tick value, margin)
"""

import json
from dataclasses import dataclass, field
from datetime import datetime, time
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


class SwapType(Enum):
    """Swap calculation type."""

    POINTS = "points"  # Swap in points
    MONEY = "money"  # Swap in account currency
    INTEREST = "interest"  # Swap as annual interest rate
    MARGIN_CURRENCY = "margin"  # Swap in margin currency


class CommissionType(Enum):
    """Commission calculation type."""

    PER_LOT = "per_lot"  # Fixed $ per lot
    PER_DEAL = "per_deal"  # Fixed $ per trade
    PERCENTAGE = "percentage"  # % of trade value


@dataclass
class SwapSpec:
    """Swap/rollover specification."""

    long_rate: float = 0.0  # Swap for long positions
    short_rate: float = 0.0  # Swap for short positions
    swap_type: SwapType = SwapType.POINTS
    triple_swap_day: int = 3  # Wednesday = 3 (Mon=1, Sun=7)

    def calculate_swap(
        self,
        position_type: str,  # "long" or "short"
        lots: float,
        contract_size: float,
        tick_value: float,
        days_held: int = 1,
        day_of_week: int = 1,
        price: float = 0.0,
    ) -> float:
        """
        Calculate swap cost for holding position overnight.

        MT5 swap calculation modes:
        - POINTS: Swap = rate * tick_value * lots * days
        - MONEY: Swap = rate * lots * days (direct currency value)
        - INTEREST: Swap = (rate / 100 / 360) * lots * contract_size * price * days
        - MARGIN_CURRENCY: Swap = rate * lots * days (in margin currency)

        Args:
            position_type: "long" or "short"
            lots: Position size in lots
            contract_size: Contract size per lot
            tick_value: Value of one tick per lot
            days_held: Number of overnight rollovers (1 = single night)
            day_of_week: Day when position was opened (1=Mon, 7=Sun)
            price: Current price (required for INTEREST swap type)

        Returns:
            Swap cost (negative = cost, positive = credit)
        """
        rate = self.long_rate if position_type == "long" else self.short_rate

        # Calculate effective swap days including triple swap
        swap_days = 0
        for d in range(days_held):
            current_day = ((day_of_week - 1 + d) % 7) + 1  # Cycle through days
            if current_day == self.triple_swap_day:
                swap_days += 3  # Weekend rollover
            else:
                swap_days += 1

        if self.swap_type == SwapType.POINTS:
            # Swap in points: rate * tick_value * lots
            return rate * tick_value * lots * swap_days

        elif self.swap_type == SwapType.MONEY:
            # Direct money value per lot per day
            return rate * lots * swap_days

        elif self.swap_type == SwapType.INTEREST:
            # Annual interest rate as percentage
            # Swap = (rate% / 100 / 360) * notional_value * days
            # notional = lots * contract_size * price
            if price <= 0:
                return 0.0
            notional = lots * contract_size * price
            daily_rate = rate / 100 / 360  # Rate is in percentage, 360-day convention
            return notional * daily_rate * swap_days

        elif self.swap_type == SwapType.MARGIN_CURRENCY:
            # Swap in margin currency per lot
            return rate * lots * swap_days

        else:
            return rate * lots * swap_days


@dataclass
class CommissionSpec:
    """Commission specification."""

    rate: float = 0.0
    commission_type: CommissionType = CommissionType.PER_LOT
    minimum: float = 0.0  # Minimum commission per trade

    def calculate_commission(
        self, lots: float, trade_value: float, is_round_trip: bool = False
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
    - Swap (overnight financing at rollover)
    - Slippage (execution quality)
    """

    # Basic info
    symbol: str
    description: str = ""
    base_currency: str = ""
    quote_currency: str = ""

    # Contract specifications
    contract_size: float = 100000.0  # Units per lot (forex = 100k)
    tick_size: float = 0.00001  # Minimum price movement
    tick_value: float = 1.0  # Value per tick per lot in account currency
    digits: int = 5  # Price decimal places

    # Margin requirements
    margin_initial: float = 0.01  # Initial margin as % (1% = 100:1 leverage)
    margin_maintenance: float = 0.005  # Maintenance margin %
    margin_hedged: float = 0.5  # Margin for hedged positions (50% = half)
    margin_rate_long: float = 1.0  # Margin rate multiplier for long positions
    margin_rate_short: float = 1.0  # Margin rate multiplier for short positions
    stop_out_level: float = 0.5  # Stop out at 50% margin level

    # Lot constraints
    volume_min: float = 0.01  # Minimum lot size
    volume_max: float = 100.0  # Maximum lot size
    volume_step: float = 0.01  # Lot size increment

    # Spread
    spread_points: float = 10.0  # Typical spread in points

    # Server time settings (for swap rollover calculation)
    server_gmt_offset: int = 2  # Server GMT offset in hours (e.g., 2 for GMT+2)
    rollover_hour: int = 0  # Hour when swap is charged (server time, typically 0)
    spread_variable: bool = True  # Is spread variable?
    spread_min: float = 5.0  # Minimum spread
    spread_max: float = 50.0  # Maximum spread (news/low liquidity)

    # Commission
    commission: CommissionSpec = field(default_factory=CommissionSpec)

    # Swap
    swap: SwapSpec = field(default_factory=SwapSpec)

    # Execution
    slippage_avg: float = 0.5  # Average slippage in points
    slippage_max: float = 5.0  # Max slippage in points

    # Trading hours (simplified)
    trading_hours_start: time = time(0, 0)
    trading_hours_end: time = time(23, 59)

    def calculate_margin(self, lots: float, price: float, position_type: str = "long") -> float:
        """
        Calculate required margin for a position.

        Margin = Lots * ContractSize * Price * MarginRate * MarginInitial

        Args:
            lots: Position size in lots
            price: Current price
            position_type: "long" or "short"

        Returns:
            Required margin in account currency
        """
        margin_rate = self.margin_rate_long if position_type == "long" else self.margin_rate_short
        notional = lots * self.contract_size * price
        return notional * self.margin_initial * margin_rate

    def calculate_margin_level(self, equity: float, used_margin: float) -> float:
        """
        Calculate margin level percentage.

        Margin Level = (Equity / Used Margin) * 100

        Args:
            equity: Current account equity
            used_margin: Total margin used by open positions

        Returns:
            Margin level as percentage (e.g., 150.0 = 150%)
        """
        if used_margin <= 0:
            return float("inf")
        return (equity / used_margin) * 100

    def is_stop_out(self, equity: float, used_margin: float) -> bool:
        """Check if margin level triggers stop-out."""
        margin_level = self.calculate_margin_level(equity, used_margin)
        return margin_level < (self.stop_out_level * 100)

    def max_lots_for_margin(
        self, free_margin: float, price: float, position_type: str = "long"
    ) -> float:
        """
        Calculate maximum lot size given available margin.

        Args:
            free_margin: Available margin
            price: Current price
            position_type: "long" or "short"

        Returns:
            Maximum lots (rounded to volume_step)
        """
        margin_rate = self.margin_rate_long if position_type == "long" else self.margin_rate_short
        margin_per_lot = self.contract_size * price * self.margin_initial * margin_rate

        if margin_per_lot <= 0:
            return 0.0

        max_lots = free_margin / margin_per_lot
        max_lots = min(max_lots, self.volume_max)
        max_lots = max(0, max_lots)

        # Round down to volume_step
        max_lots = (max_lots // self.volume_step) * self.volume_step

        return max_lots

    def spread_cost(self, lots: float, price: float) -> float:
        """Calculate spread cost for entry."""
        spread_value = self.spread_points * self.tick_size
        return spread_value * lots * self.contract_size

    def total_entry_cost(self, lots: float, price: float) -> float:
        """Total cost to enter a position."""
        spread = self.spread_cost(lots, price)
        commission = self.commission.calculate_commission(lots, lots * self.contract_size * price)
        slippage = self.slippage_avg * self.tick_value * lots
        return spread + commission + slippage

    def total_exit_cost(self, lots: float, price: float) -> float:
        """Total cost to exit a position."""
        # No spread on exit (already paid on entry via bid/ask)
        commission = self.commission.calculate_commission(lots, lots * self.contract_size * price)
        slippage = self.slippage_avg * self.tick_value * lots
        return commission + slippage

    def count_rollovers(self, entry_time: datetime, exit_time: datetime) -> Tuple[int, List[int]]:
        """
        Count number of rollover events between entry and exit.

        Rollover occurs at rollover_hour in server time (adjusted by server_gmt_offset).

        Args:
            entry_time: Position entry time (UTC or local - should be consistent)
            exit_time: Position exit time

        Returns:
            Tuple of (rollover_count, list of day_of_week for each rollover)
        """
        from datetime import timedelta

        # Convert rollover hour to UTC
        rollover_hour_utc = (self.rollover_hour - self.server_gmt_offset) % 24

        rollovers = []
        current = entry_time.replace(hour=rollover_hour_utc, minute=0, second=0, microsecond=0)

        # If entry is after rollover time, start from next day's rollover
        if entry_time.hour >= rollover_hour_utc or (
            entry_time.hour == rollover_hour_utc and entry_time.minute > 0
        ):
            current += timedelta(days=1)

        # Count each rollover crossed
        while current < exit_time:
            rollovers.append(current.isoweekday())  # 1=Mon, 7=Sun
            current += timedelta(days=1)

        return len(rollovers), rollovers

    def holding_cost(
        self,
        position_type: str,
        lots: float,
        days: int,
        day_of_week: int = 1,
        price: float = 0.0,
    ) -> float:
        """Calculate overnight holding cost (legacy method for simple calculation)."""
        return self.swap.calculate_swap(
            position_type,
            lots,
            self.contract_size,
            self.tick_value,
            days,
            day_of_week,
            price,
        )

    def calculate_swap_cost(
        self,
        position_type: str,
        lots: float,
        entry_time: datetime,
        exit_time: datetime,
        price: float = 0.0,
    ) -> float:
        """
        Calculate swap cost based on actual rollover events.

        Args:
            position_type: "long" or "short"
            lots: Position size
            entry_time: When position was opened
            exit_time: When position was closed
            price: Position price (required for INTEREST swap type)

        Returns:
            Total swap cost (negative = cost, positive = credit)
        """
        rollover_count, rollover_days = self.count_rollovers(entry_time, exit_time)

        if rollover_count == 0:
            return 0.0

        # Calculate swap for each rollover, applying triple swap on the right day
        total_swap = 0.0
        for dow in rollover_days:
            if dow == self.swap.triple_swap_day:
                multiplier = 3  # Triple swap (covers weekend)
            else:
                multiplier = 1
            total_swap += self.swap.calculate_swap(
                position_type,
                lots,
                self.contract_size,
                self.tick_value,
                days_held=multiplier,
                day_of_week=dow,
                price=price,
            )

        return total_swap

    def round_trip_cost(
        self,
        lots: float,
        entry_price: float,
        exit_price: float,
        position_type: str = "long",
        holding_days: int = 0,
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
                    (day % 7) + 1,  # Simplified day rotation
                )

        total = entry_cost + exit_cost + abs(swap_cost)

        return {
            "spread_cost": self.spread_cost(lots, entry_price),
            "commission_entry": self.commission.calculate_commission(
                lots, lots * self.contract_size * entry_price
            ),
            "commission_exit": self.commission.calculate_commission(
                lots, lots * self.contract_size * exit_price
            ),
            "slippage": self.slippage_avg * self.tick_value * lots * 2,
            "swap_cost": swap_cost,
            "total_cost": total,
            "cost_in_points": total / (self.tick_value * lots) if lots > 0 else 0,
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
    def from_dict(cls, data: Dict) -> "SymbolSpec":
        """Deserialize from dictionary."""
        commission = CommissionSpec(
            rate=data.get("commission_rate", 0),
            commission_type=CommissionType(data.get("commission_type", "per_lot")),
        )
        swap = SwapSpec(
            long_rate=data.get("swap_long", 0),
            short_rate=data.get("swap_short", 0),
            swap_type=SwapType(data.get("swap_type", "points")),
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
    contract_size=1.0,  # 1 BTC per lot
    tick_size=0.01,  # $0.01 minimum move
    tick_value=0.01,  # $0.01 per tick per lot
    digits=2,
    spread_points=5000,  # $50 typical spread
    spread_min=2000,
    spread_max=20000,
    commission=CommissionSpec(rate=0, commission_type=CommissionType.PER_LOT),
    swap=SwapSpec(long_rate=-15.0, short_rate=-15.0, swap_type=SwapType.POINTS),
    slippage_avg=100,  # $1 avg slippage
)

EURUSD_SPEC = SymbolSpec(
    symbol="EURUSD",
    description="Euro vs US Dollar",
    base_currency="EUR",
    quote_currency="USD",
    contract_size=100000,
    tick_size=0.00001,
    tick_value=1.0,  # $1 per pip for 1 lot
    digits=5,
    spread_points=10,  # 1 pip typical
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
    contract_size=100,  # 100 oz per lot
    tick_size=0.01,
    tick_value=1.0,  # $1 per tick for 1 lot
    digits=2,
    spread_points=30,  # 30 cents typical
    spread_min=15,
    spread_max=100,
    commission=CommissionSpec(rate=0, commission_type=CommissionType.PER_LOT),
    swap=SwapSpec(long_rate=-68.09, short_rate=47.91, swap_type=SwapType.POINTS),
)

COPPER_SPEC = SymbolSpec(
    symbol="COPPER-C",
    description="Copper CFD",
    base_currency="COPPER",
    quote_currency="USD",
    contract_size=25000,  # 25000 lbs per lot
    tick_size=0.0001,
    tick_value=2.5,  # $2.50 per tick for 1 lot
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
            triple_swap_day=info.swap_rollover3days + 1,  # MT5 uses 0-indexed
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
        lots=0.1, entry_price=95000, exit_price=96000, position_type="long", holding_days=3
    )

    print(f"\nRound-trip costs (0.1 lot, 3 days):")
    for k, v in costs.items():
        print(f"  {k}: ${v:.2f}")

    # Gross P&L
    gross_pnl = (96000 - 95000) * 0.1  # $100 profit
    net_pnl = gross_pnl - costs["total_cost"]
    print(f"\nGross P&L: ${gross_pnl:.2f}")
    print(f"Net P&L: ${net_pnl:.2f}")
    print(f"Cost as % of gross: {(costs['total_cost'] / gross_pnl) * 100:.1f}%")
