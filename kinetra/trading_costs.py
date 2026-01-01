"""
Trading Costs Module
====================

Comprehensive trading cost calculation including:
- Spread costs
- Commission (per lot, per trade, percentage)
- Swap fees (overnight interest)
- Triple swap days (typically Wednesday for forex)
- Slippage estimation
- Market impact costs
- Financing costs

All calculations follow broker-standard conventions with
high precision for financial accuracy.
"""

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta, date
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)


class CommissionType(Enum):
    """Commission calculation type."""
    NONE = auto()
    PER_LOT = auto()           # Fixed amount per lot
    PER_TRADE = auto()         # Fixed amount per trade
    PERCENTAGE = auto()        # Percentage of trade value
    PER_MILLION = auto()       # Amount per million USD traded
    TIERED = auto()            # Volume-based tiers


class SwapType(Enum):
    """Swap calculation type."""
    POINTS = auto()            # Swap in points
    PERCENTAGE = auto()        # Annual percentage rate
    MONEY = auto()             # Fixed money amount
    INTEREST_DIFF = auto()     # Interest rate differential


class SlippageModel(Enum):
    """Slippage estimation model."""
    FIXED = auto()             # Fixed points
    PROPORTIONAL = auto()      # Proportional to spread
    VOLUME_BASED = auto()      # Based on order size
    VOLATILITY_BASED = auto()  # Based on market volatility


@dataclass
class SwapSpec:
    """Swap (overnight interest) specification."""
    swap_type: SwapType = SwapType.POINTS
    swap_long: float = 0.0     # Swap for long positions
    swap_short: float = 0.0    # Swap for short positions
    triple_swap_day: int = 3   # Day of week (0=Mon, 3=Wed)
    swap_rollover_time: str = "00:00"  # Server time for rollover
    
    # For percentage-based swaps
    annual_rate_long: float = 0.0
    annual_rate_short: float = 0.0
    
    def get_swap_for_position(
        self,
        is_long: bool,
        lot_size: float,
        contract_size: float,
        current_price: float,
        point_value: float,
    ) -> float:
        """
        Calculate swap amount for position.
        
        Args:
            is_long: True for long position
            lot_size: Position size in lots
            contract_size: Contract size per lot
            current_price: Current market price
            point_value: Point value in account currency
            
        Returns:
            Swap amount in account currency
        """
        if self.swap_type == SwapType.POINTS:
            swap_points = self.swap_long if is_long else self.swap_short
            return swap_points * lot_size * point_value
            
        elif self.swap_type == SwapType.PERCENTAGE:
            rate = self.annual_rate_long if is_long else self.annual_rate_short
            position_value = lot_size * contract_size * current_price
            daily_rate = rate / 365
            return position_value * daily_rate
            
        elif self.swap_type == SwapType.MONEY:
            swap_amount = self.swap_long if is_long else self.swap_short
            return swap_amount * lot_size
            
        return 0.0


@dataclass
class CommissionSpec:
    """Commission specification."""
    commission_type: CommissionType = CommissionType.NONE
    commission_value: float = 0.0      # Base commission value
    commission_min: float = 0.0        # Minimum commission
    commission_max: float = float('inf')  # Maximum commission
    
    # For tiered commission
    volume_tiers: List[Tuple[float, float]] = field(default_factory=list)
    # [(volume_threshold, commission_rate), ...]
    
    def calculate(
        self,
        lot_size: float,
        trade_value: float,
        monthly_volume: float = 0.0,
    ) -> float:
        """
        Calculate commission for trade.
        
        Args:
            lot_size: Trade size in lots
            trade_value: Trade notional value
            monthly_volume: Monthly trading volume for tiered rates
            
        Returns:
            Commission amount in account currency
        """
        if self.commission_type == CommissionType.NONE:
            return 0.0
            
        elif self.commission_type == CommissionType.PER_LOT:
            commission = self.commission_value * lot_size
            
        elif self.commission_type == CommissionType.PER_TRADE:
            commission = self.commission_value
            
        elif self.commission_type == CommissionType.PERCENTAGE:
            commission = trade_value * (self.commission_value / 100)
            
        elif self.commission_type == CommissionType.PER_MILLION:
            commission = (trade_value / 1_000_000) * self.commission_value
            
        elif self.commission_type == CommissionType.TIERED:
            rate = self.commission_value  # Default rate
            for threshold, tier_rate in sorted(self.volume_tiers, reverse=True):
                if monthly_volume >= threshold:
                    rate = tier_rate
                    break
            commission = trade_value * (rate / 100)
            
        else:
            commission = 0.0
        
        # Apply min/max
        commission = max(self.commission_min, min(self.commission_max, commission))
        
        return commission


@dataclass
class SlippageSpec:
    """Slippage specification."""
    model: SlippageModel = SlippageModel.PROPORTIONAL
    base_slippage_points: float = 1.0
    spread_multiplier: float = 0.5
    volume_impact_factor: float = 0.1
    volatility_factor: float = 0.5
    max_slippage_points: float = 10.0


@dataclass
class TradingCostSpec:
    """Complete trading cost specification."""
    symbol: str
    
    # Basic specs
    point: float = 0.0001       # Point size
    tick_size: float = 0.0001   # Minimum price movement
    tick_value: float = 10.0    # Tick value in account currency
    contract_size: float = 100000  # Contract size per lot
    
    # Spread
    spread_points: float = 1.0
    spread_variable: bool = True
    
    # Commission
    commission: CommissionSpec = field(default_factory=CommissionSpec)
    
    # Swap
    swap: SwapSpec = field(default_factory=SwapSpec)
    
    # Slippage
    slippage: SlippageSpec = field(default_factory=SlippageSpec)
    
    # Additional costs
    exchange_fee: float = 0.0      # Exchange fees per lot
    clearing_fee: float = 0.0     # Clearing fees per lot
    regulatory_fee: float = 0.0   # Regulatory fees per lot


@dataclass
class TradeCosts:
    """Calculated costs for a trade."""
    spread_cost: float = 0.0
    commission_open: float = 0.0
    commission_close: float = 0.0
    slippage_open: float = 0.0
    slippage_close: float = 0.0
    swap_cost: float = 0.0
    exchange_fees: float = 0.0
    other_fees: float = 0.0
    
    @property
    def total_entry_cost(self) -> float:
        """Total cost at entry."""
        return self.spread_cost + self.commission_open + self.slippage_open
    
    @property
    def total_exit_cost(self) -> float:
        """Total cost at exit."""
        return self.commission_close + self.slippage_close
    
    @property
    def total_holding_cost(self) -> float:
        """Total cost from holding (swaps)."""
        return self.swap_cost
    
    @property
    def total_fees(self) -> float:
        """Total fees."""
        return self.exchange_fees + self.other_fees
    
    @property
    def total_cost(self) -> float:
        """Total all-in cost."""
        return (
            self.spread_cost +
            self.commission_open +
            self.commission_close +
            self.slippage_open +
            self.slippage_close +
            self.swap_cost +
            self.exchange_fees +
            self.other_fees
        )
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'spread_cost': self.spread_cost,
            'commission_open': self.commission_open,
            'commission_close': self.commission_close,
            'slippage_open': self.slippage_open,
            'slippage_close': self.slippage_close,
            'swap_cost': self.swap_cost,
            'exchange_fees': self.exchange_fees,
            'other_fees': self.other_fees,
            'total_entry_cost': self.total_entry_cost,
            'total_exit_cost': self.total_exit_cost,
            'total_holding_cost': self.total_holding_cost,
            'total_cost': self.total_cost,
        }


class SwapCalendar:
    """
    Calendar for swap calculations including holidays and triple swap days.
    """
    
    # Default forex holidays (major markets closed)
    DEFAULT_HOLIDAYS = [
        # 2024
        date(2024, 1, 1),   # New Year
        date(2024, 1, 15),  # MLK Day
        date(2024, 2, 19),  # Presidents Day
        date(2024, 3, 29),  # Good Friday
        date(2024, 5, 27),  # Memorial Day
        date(2024, 7, 4),   # Independence Day
        date(2024, 9, 2),   # Labor Day
        date(2024, 11, 28), # Thanksgiving
        date(2024, 12, 25), # Christmas
        # 2025
        date(2025, 1, 1),
        date(2025, 1, 20),
        date(2025, 2, 17),
        date(2025, 4, 18),
        date(2025, 5, 26),
        date(2025, 7, 4),
        date(2025, 9, 1),
        date(2025, 11, 27),
        date(2025, 12, 25),
    ]
    
    def __init__(
        self,
        triple_swap_day: int = 2,  # Wednesday (0=Monday)
        holidays: List[date] = None,
    ):
        """
        Initialize swap calendar.
        
        Args:
            triple_swap_day: Day of week for triple swap (0-6)
            holidays: List of holiday dates
        """
        self.triple_swap_day = triple_swap_day
        self.holidays = set(holidays or self.DEFAULT_HOLIDAYS)
    
    def get_swap_multiplier(self, for_date: date) -> int:
        """
        Get swap multiplier for a date.
        
        Triple swap typically applies on Wednesday for forex to account
        for weekend days. Holidays may add additional multipliers.
        
        Args:
            for_date: Date to check
            
        Returns:
            Swap multiplier (1, 2, 3, or more)
        """
        day_of_week = for_date.weekday()
        
        # Weekend - no swap (market closed)
        if day_of_week >= 5:
            return 0
        
        multiplier = 1
        
        # Triple swap day (typically Wednesday for forex)
        if day_of_week == self.triple_swap_day:
            multiplier = 3  # Covers Saturday and Sunday
        
        # Check for upcoming holidays
        next_day = for_date + timedelta(days=1)
        
        # If tomorrow is a holiday, add extra swap
        while next_day in self.holidays or next_day.weekday() >= 5:
            if next_day in self.holidays:
                multiplier += 1
            next_day += timedelta(days=1)
            
            # Safety limit
            if (next_day - for_date).days > 7:
                break
        
        return multiplier
    
    def is_trading_day(self, check_date: date) -> bool:
        """Check if date is a trading day."""
        if check_date.weekday() >= 5:
            return False
        return check_date not in self.holidays
    
    def get_next_trading_day(self, from_date: date) -> date:
        """Get next trading day."""
        next_day = from_date + timedelta(days=1)
        while not self.is_trading_day(next_day):
            next_day += timedelta(days=1)
        return next_day


class TradingCostCalculator:
    """
    Calculate all trading costs with high precision.
    
    Features:
    - Spread cost calculation
    - Commission calculation (multiple types)
    - Swap fee calculation with calendar awareness
    - Slippage estimation
    - Market impact estimation
    - Total cost analysis
    """
    
    def __init__(
        self,
        spec: TradingCostSpec,
        swap_calendar: SwapCalendar = None,
        use_decimal_precision: bool = True,
    ):
        """
        Initialize cost calculator.
        
        Args:
            spec: Trading cost specification
            swap_calendar: Swap calendar (auto-creates if None)
            use_decimal_precision: Use Decimal for high precision
        """
        self.spec = spec
        self.calendar = swap_calendar or SwapCalendar(spec.swap.triple_swap_day)
        self.use_decimal = use_decimal_precision
    
    def _to_decimal(self, value: float) -> Decimal | float:
        """Convert to Decimal for precision."""
        if self.use_decimal:
            return Decimal(str(value))
        return value
    
    def _from_decimal(self, value: Union[Decimal, float]) -> float:
        """Convert from Decimal."""
        if isinstance(value, Decimal):
            return float(value.quantize(Decimal('0.00000001'), rounding=ROUND_HALF_UP))
        return value
    
    def calculate_spread_cost(
        self,
        lot_size: float,
        spread_points: float = None,
    ) -> float:
        """
        Calculate spread cost.
        
        Spread cost is: spread_in_price * contract_size * lot_size
        
        For forex (EURUSD with 5 decimal):
        - spread_points = 10 means 0.00010 price spread (1 pip)
        - 1 lot = 100,000 units
        - cost = 0.00010 * 100,000 * 1 = $10
        
        Args:
            lot_size: Position size in lots
            spread_points: Current spread in points (uses spec default if None)
            
        Returns:
            Spread cost in account currency
        """
        spread = spread_points if spread_points is not None else self.spec.spread_points
        
        # Convert spread from points to price units
        spread_in_price = spread * self.spec.point
        
        # Calculate cost: spread * contract_size * lots
        # This gives the cost in quote currency (USD for EURUSD)
        cost = spread_in_price * self.spec.contract_size * lot_size
        
        return self._from_decimal(self._to_decimal(cost))
    
    def calculate_commission(
        self,
        lot_size: float,
        price: float,
        is_entry: bool = True,
        monthly_volume: float = 0.0,
    ) -> float:
        """
        Calculate commission.
        
        Args:
            lot_size: Position size in lots
            price: Trade price
            is_entry: True for entry, False for exit
            monthly_volume: Monthly trading volume
            
        Returns:
            Commission in account currency
        """
        trade_value = lot_size * self.spec.contract_size * price
        
        return self.spec.commission.calculate(
            lot_size=lot_size,
            trade_value=trade_value,
            monthly_volume=monthly_volume,
        )
    
    def calculate_slippage(
        self,
        lot_size: float,
        spread_points: float = None,
        volatility: float = None,
        is_market_order: bool = True,
    ) -> float:
        """
        Estimate slippage cost.
        
        Args:
            lot_size: Position size in lots
            spread_points: Current spread
            volatility: Current volatility (e.g., ATR)
            is_market_order: True for market order
            
        Returns:
            Estimated slippage in points
        """
        if not is_market_order:
            return 0.0
        
        slippage_spec = self.spec.slippage
        spread = spread_points or self.spec.spread_points
        
        if slippage_spec.model == SlippageModel.FIXED:
            slippage = slippage_spec.base_slippage_points
            
        elif slippage_spec.model == SlippageModel.PROPORTIONAL:
            slippage = spread * slippage_spec.spread_multiplier
            
        elif slippage_spec.model == SlippageModel.VOLUME_BASED:
            # Larger orders have more slippage
            slippage = (
                slippage_spec.base_slippage_points +
                lot_size * slippage_spec.volume_impact_factor
            )
            
        elif slippage_spec.model == SlippageModel.VOLATILITY_BASED:
            vol = volatility or spread * 10  # Default estimate
            slippage = (
                slippage_spec.base_slippage_points +
                vol * slippage_spec.volatility_factor
            )
        else:
            slippage = slippage_spec.base_slippage_points
        
        # Apply maximum
        slippage = min(slippage, slippage_spec.max_slippage_points)
        
        # Convert to account currency
        point_value = self.spec.tick_value * (self.spec.point / self.spec.tick_size)
        return slippage * point_value * lot_size
    
    def calculate_swap(
        self,
        is_long: bool,
        lot_size: float,
        current_price: float,
        holding_days: int = 1,
        start_date: date = None,
    ) -> Tuple[float, int]:
        """
        Calculate swap fees for holding period.
        
        Args:
            is_long: True for long position
            lot_size: Position size in lots
            current_price: Current price
            holding_days: Number of days held
            start_date: Start date for calendar calculation
            
        Returns:
            (total_swap_cost, total_swap_days)
        """
        start_date = start_date or date.today()
        
        point_value = self.spec.tick_value * (self.spec.point / self.spec.tick_size)
        
        total_swap = 0.0
        total_days = 0
        current_date = start_date
        
        for _ in range(holding_days):
            # Get swap multiplier for this date
            multiplier = self.calendar.get_swap_multiplier(current_date)
            
            if multiplier > 0:
                daily_swap = self.spec.swap.get_swap_for_position(
                    is_long=is_long,
                    lot_size=lot_size,
                    contract_size=self.spec.contract_size,
                    current_price=current_price,
                    point_value=point_value,
                )
                
                total_swap += daily_swap * multiplier
                total_days += multiplier
            
            current_date += timedelta(days=1)
        
        return total_swap, total_days
    
    def calculate_exchange_fees(self, lot_size: float) -> float:
        """Calculate exchange and regulatory fees."""
        fees = (
            self.spec.exchange_fee +
            self.spec.clearing_fee +
            self.spec.regulatory_fee
        ) * lot_size
        
        return fees
    
    def calculate_total_cost(
        self,
        lot_size: float,
        entry_price: float,
        exit_price: float = None,
        is_long: bool = True,
        holding_days: int = 0,
        spread_at_entry: float = None,
        spread_at_exit: float = None,
        start_date: date = None,
        monthly_volume: float = 0.0,
    ) -> TradeCosts:
        """
        Calculate all trading costs for a complete trade.
        
        Args:
            lot_size: Position size in lots
            entry_price: Entry price
            exit_price: Exit price (None for entry-only calculation)
            is_long: True for long position
            holding_days: Days position held
            spread_at_entry: Spread at entry
            spread_at_exit: Spread at exit
            start_date: Trade start date
            monthly_volume: Monthly trading volume
            
        Returns:
            Complete cost breakdown
        """
        costs = TradeCosts()
        
        # Spread cost (paid at entry for market order)
        costs.spread_cost = self.calculate_spread_cost(
            lot_size=lot_size,
            spread_points=spread_at_entry,
        )
        
        # Commission at entry
        costs.commission_open = self.calculate_commission(
            lot_size=lot_size,
            price=entry_price,
            is_entry=True,
            monthly_volume=monthly_volume,
        )
        
        # Slippage at entry
        costs.slippage_open = self.calculate_slippage(
            lot_size=lot_size,
            spread_points=spread_at_entry,
        )
        
        # Exit costs if exit_price provided
        if exit_price is not None:
            costs.commission_close = self.calculate_commission(
                lot_size=lot_size,
                price=exit_price,
                is_entry=False,
                monthly_volume=monthly_volume,
            )
            
            costs.slippage_close = self.calculate_slippage(
                lot_size=lot_size,
                spread_points=spread_at_exit or spread_at_entry,
            )
        
        # Swap costs
        if holding_days > 0:
            swap_cost, _ = self.calculate_swap(
                is_long=is_long,
                lot_size=lot_size,
                current_price=(entry_price + (exit_price or entry_price)) / 2,
                holding_days=holding_days,
                start_date=start_date,
            )
            costs.swap_cost = swap_cost
        
        # Exchange fees
        costs.exchange_fees = self.calculate_exchange_fees(lot_size)
        
        return costs
    
    def calculate_breakeven_pips(
        self,
        lot_size: float,
        entry_price: float,
        is_long: bool = True,
        holding_days: int = 1,
    ) -> float:
        """
        Calculate breakeven distance in pips.
        
        Args:
            lot_size: Position size in lots
            entry_price: Entry price
            is_long: True for long position
            holding_days: Expected holding period
            
        Returns:
            Breakeven distance in pips
        """
        costs = self.calculate_total_cost(
            lot_size=lot_size,
            entry_price=entry_price,
            exit_price=entry_price,  # At entry price for breakeven
            is_long=is_long,
            holding_days=holding_days,
        )
        
        # Convert cost to pips
        point_value = self.spec.tick_value * (self.spec.point / self.spec.tick_size)
        pip_value = point_value * lot_size * 10  # 1 pip = 10 points typically
        
        if pip_value > 0:
            return costs.total_cost / pip_value
        return 0.0


class CostAnalyzer:
    """
    Analyze trading costs across different scenarios.
    """
    
    def __init__(self, calculator: TradingCostCalculator):
        """
        Initialize cost analyzer.
        
        Args:
            calculator: Trading cost calculator
        """
        self.calculator = calculator
    
    def analyze_holding_period(
        self,
        lot_size: float,
        entry_price: float,
        is_long: bool,
        max_days: int = 30,
    ) -> Dict[int, TradeCosts]:
        """
        Analyze costs across different holding periods.
        
        Args:
            lot_size: Position size
            entry_price: Entry price
            is_long: True for long
            max_days: Maximum days to analyze
            
        Returns:
            Dict of days -> costs
        """
        results = {}
        
        for days in range(max_days + 1):
            costs = self.calculator.calculate_total_cost(
                lot_size=lot_size,
                entry_price=entry_price,
                exit_price=entry_price,
                is_long=is_long,
                holding_days=days,
            )
            results[days] = costs
        
        return results
    
    def analyze_lot_sizes(
        self,
        lot_sizes: List[float],
        entry_price: float,
        is_long: bool = True,
        holding_days: int = 1,
    ) -> Dict[float, TradeCosts]:
        """
        Analyze costs across different lot sizes.
        
        Args:
            lot_sizes: List of lot sizes to analyze
            entry_price: Entry price
            is_long: True for long
            holding_days: Holding period
            
        Returns:
            Dict of lot_size -> costs
        """
        results = {}
        
        for lot_size in lot_sizes:
            costs = self.calculator.calculate_total_cost(
                lot_size=lot_size,
                entry_price=entry_price,
                exit_price=entry_price,
                is_long=is_long,
                holding_days=holding_days,
            )
            results[lot_size] = costs
        
        return results
    
    def calculate_effective_spread(
        self,
        lot_size: float,
        entry_price: float,
        holding_days: int = 1,
    ) -> float:
        """
        Calculate effective spread including all costs.
        
        This represents the total cost as an equivalent spread.
        
        Args:
            lot_size: Position size
            entry_price: Entry price
            holding_days: Holding period
            
        Returns:
            Effective spread in points
        """
        costs = self.calculator.calculate_total_cost(
            lot_size=lot_size,
            entry_price=entry_price,
            exit_price=entry_price,
            is_long=True,
            holding_days=holding_days,
        )
        
        # Convert total cost to points
        point_value = (
            self.calculator.spec.tick_value *
            (self.calculator.spec.point / self.calculator.spec.tick_size)
        )
        
        if point_value > 0 and lot_size > 0:
            return costs.total_cost / (point_value * lot_size)
        return 0.0
    
    def get_cost_breakdown_pct(
        self,
        lot_size: float,
        entry_price: float,
        exit_price: float,
        is_long: bool,
        holding_days: int,
    ) -> Dict[str, float]:
        """
        Get cost breakdown as percentage of trade value.
        
        Args:
            lot_size: Position size
            entry_price: Entry price
            exit_price: Exit price
            is_long: True for long
            holding_days: Holding period
            
        Returns:
            Cost breakdown as percentages
        """
        costs = self.calculator.calculate_total_cost(
            lot_size=lot_size,
            entry_price=entry_price,
            exit_price=exit_price,
            is_long=is_long,
            holding_days=holding_days,
        )
        
        trade_value = lot_size * self.calculator.spec.contract_size * entry_price
        
        if trade_value == 0:
            return {}
        
        return {
            'spread_pct': (costs.spread_cost / trade_value) * 100,
            'commission_pct': ((costs.commission_open + costs.commission_close) / trade_value) * 100,
            'slippage_pct': ((costs.slippage_open + costs.slippage_close) / trade_value) * 100,
            'swap_pct': (costs.swap_cost / trade_value) * 100,
            'fees_pct': ((costs.exchange_fees + costs.other_fees) / trade_value) * 100,
            'total_pct': (costs.total_cost / trade_value) * 100,
        }


# Pre-configured cost specs for common instruments
def get_forex_major_spec(symbol: str) -> TradingCostSpec:
    """Get cost spec for forex majors."""
    return TradingCostSpec(
        symbol=symbol,
        point=0.00001,
        tick_size=0.00001,
        tick_value=10.0,
        contract_size=100000,
        spread_points=10,  # 1 pip
        commission=CommissionSpec(
            commission_type=CommissionType.PER_LOT,
            commission_value=3.5,  # $3.50 per lot per side
        ),
        swap=SwapSpec(
            swap_type=SwapType.POINTS,
            swap_long=-8.5,
            swap_short=2.3,
            triple_swap_day=2,  # Wednesday
        ),
        slippage=SlippageSpec(
            model=SlippageModel.PROPORTIONAL,
            spread_multiplier=0.3,
        ),
    )


def get_index_cfd_spec(symbol: str) -> TradingCostSpec:
    """Get cost spec for index CFDs."""
    return TradingCostSpec(
        symbol=symbol,
        point=0.1,
        tick_size=0.1,
        tick_value=0.1,
        contract_size=1,
        spread_points=10,  # 1 point
        commission=CommissionSpec(
            commission_type=CommissionType.PERCENTAGE,
            commission_value=0.02,  # 0.02%
        ),
        swap=SwapSpec(
            swap_type=SwapType.PERCENTAGE,
            annual_rate_long=-0.05,  # 5% annual
            annual_rate_short=-0.03,  # 3% annual (short rebate)
            triple_swap_day=4,  # Friday for indices
        ),
        slippage=SlippageSpec(
            model=SlippageModel.FIXED,
            base_slippage_points=2.0,
        ),
    )


def get_crypto_spec(symbol: str) -> TradingCostSpec:
    """Get cost spec for crypto."""
    return TradingCostSpec(
        symbol=symbol,
        point=0.01,
        tick_size=0.01,
        tick_value=0.01,
        contract_size=1,
        spread_points=50,  # Wide spread
        commission=CommissionSpec(
            commission_type=CommissionType.PERCENTAGE,
            commission_value=0.1,  # 0.1%
        ),
        swap=SwapSpec(
            swap_type=SwapType.PERCENTAGE,
            annual_rate_long=-0.10,  # 10% annual
            annual_rate_short=-0.10,
            triple_swap_day=2,
        ),
        slippage=SlippageSpec(
            model=SlippageModel.VOLATILITY_BASED,
            base_slippage_points=5.0,
            volatility_factor=0.5,
        ),
    )


# Export all components
__all__ = [
    'CommissionType',
    'SwapType',
    'SlippageModel',
    'SwapSpec',
    'CommissionSpec',
    'SlippageSpec',
    'TradingCostSpec',
    'TradeCosts',
    'SwapCalendar',
    'TradingCostCalculator',
    'CostAnalyzer',
    'get_forex_major_spec',
    'get_index_cfd_spec',
    'get_crypto_spec',
]
