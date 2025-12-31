"""
Unified Symbol Information Module
=================================

Comprehensive symbol specifications with accurate:
- Contract sizes per asset class
- Point/pip values that differ by instrument
- Tick values for P&L calculation
- Margin requirements
- Trading costs

Asset Class Specifics:
----------------------
FOREX (5 digit):
  - Contract: 100,000 base currency
  - Point: 0.00001 (5th decimal)
  - Pip: 0.0001 (4th decimal) = 10 points
  - Pip value: ~$10 per pip per lot (for USD pairs)
  
FOREX (3 digit - JPY pairs):
  - Contract: 100,000 base currency  
  - Point: 0.001 (3rd decimal)
  - Pip: 0.01 (2nd decimal) = 10 points
  - Pip value: ~$10 per pip per lot (converted)

GOLD (XAUUSD):
  - Contract: 100 troy ounces
  - Point: 0.01 ($0.01 move)
  - Tick value: $1 per 0.01 move per lot
  
INDICES (US30, NAS100, etc):
  - Contract: 1 unit (CFD)
  - Point: varies (0.1 or 1.0)
  - Tick value: $1 per point typically

CRYPTO (BTCUSD):
  - Contract: 1 unit
  - Point: 0.01 or 1.0
  - Tick value: contract_size * point
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, time
from enum import Enum, auto
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)


class AssetClass(Enum):
    """Asset class categorization."""
    FOREX_MAJOR = auto()      # EUR, GBP, USD, JPY, CHF, CAD, AUD, NZD
    FOREX_MINOR = auto()      # Cross pairs
    FOREX_EXOTIC = auto()     # Emerging market currencies
    CRYPTO = auto()           # Bitcoin, Ethereum, etc.
    COMMODITY_METAL = auto()  # Gold, Silver, Platinum
    COMMODITY_ENERGY = auto() # Oil, Natural Gas
    COMMODITY_AGRI = auto()   # Wheat, Corn, Coffee
    INDEX_US = auto()         # S&P 500, Dow, Nasdaq
    INDEX_EU = auto()         # DAX, FTSE, CAC
    INDEX_ASIA = auto()       # Nikkei, Hang Seng
    STOCK_US = auto()         # US Stocks
    STOCK_EU = auto()         # European Stocks
    BOND = auto()             # Treasury bonds
    ETF = auto()              # Exchange traded funds


class CalcMode(Enum):
    """Profit calculation mode (MT5 standard)."""
    FOREX = auto()            # Profit = (close - open) * contract * lots
    FOREX_NO_LEVERAGE = auto()
    FUTURES = auto()          # Profit = (close - open) * tick_value / tick_size * lots
    CFD = auto()              # Same as FOREX
    CFD_INDEX = auto()        # Profit = (close - open) * contract * lots
    CFD_LEVERAGE = auto()
    EXCHANGE_STOCKS = auto()  # Profit = (close - open) * contract * lots
    EXCHANGE_FUTURES = auto()
    EXCHANGE_OPTIONS = auto()


class MarginMode(Enum):
    """Margin calculation mode."""
    RETAIL_NETTING = auto()
    EXCHANGE = auto()
    RETAIL_HEDGING = auto()


@dataclass
class TradingSession:
    """Trading session times."""
    start: time
    end: time
    name: str = ""
    
    def is_open(self, current_time: time) -> bool:
        """Check if session is open."""
        if self.start <= self.end:
            return self.start <= current_time <= self.end
        else:  # Overnight session
            return current_time >= self.start or current_time <= self.end


@dataclass 
class SymbolInfo:
    """
    Complete symbol information with accurate specifications.
    
    Designed to match MT5 symbol_info() structure while being
    broker-agnostic and supporting multiple calculation modes.
    """
    
    # ═══════════════════════════════════════════════════════════════
    # IDENTITY
    # ═══════════════════════════════════════════════════════════════
    symbol: str                           # Symbol name (e.g., "EURUSD")
    name: str = ""                        # Full name
    description: str = ""                 # Description
    path: str = ""                        # Symbol path in Market Watch
    
    # Asset classification
    asset_class: AssetClass = AssetClass.FOREX_MAJOR
    base_currency: str = ""               # Base currency (EUR in EURUSD)
    quote_currency: str = ""              # Quote/profit currency (USD in EURUSD)  
    margin_currency: str = ""             # Currency for margin calculation
    
    # ═══════════════════════════════════════════════════════════════
    # PRICE PRECISION
    # ═══════════════════════════════════════════════════════════════
    digits: int = 5                       # Price decimal places
    point: float = 0.00001                # Minimum price change (1 point)
    
    # For display/common usage
    pip_digits: int = 4                   # Pip decimal places (forex convention)
    pip_size: float = 0.0001              # Size of 1 pip (10 points for 5-digit)
    
    # ═══════════════════════════════════════════════════════════════
    # CONTRACT SPECIFICATIONS
    # ═══════════════════════════════════════════════════════════════
    contract_size: float = 100000.0       # Units per 1.0 lot
    
    # Tick value: monetary value of 1 tick (point) movement per lot
    # CRITICAL: This varies significantly by instrument!
    #
    # FOREX 5-digit (EURUSD): tick_value = ~$1 per point per lot
    #   1 pip (10 points) = ~$10 per lot
    #
    # GOLD (XAUUSD): tick_value = $1 per $0.01 move per lot (100oz)
    #   $1 move = $100 per lot
    #
    # INDICES (US500): tick_value = $1 per 0.1 point per lot
    #
    tick_size: float = 0.00001            # Minimum price increment
    tick_value: float = 1.0               # Value of 1 tick in profit currency
    tick_value_profit: float = 1.0        # Tick value in profit currency
    tick_value_loss: float = 1.0          # Tick value for losses
    
    # Calculation mode
    calc_mode: CalcMode = CalcMode.FOREX
    
    # ═══════════════════════════════════════════════════════════════
    # VOLUME CONSTRAINTS  
    # ═══════════════════════════════════════════════════════════════
    volume_min: float = 0.01              # Minimum lot size
    volume_max: float = 100.0             # Maximum lot size
    volume_step: float = 0.01             # Lot size increment
    volume_limit: float = 0.0             # Max total volume (0 = unlimited)
    
    # ═══════════════════════════════════════════════════════════════
    # MARGIN REQUIREMENTS
    # ═══════════════════════════════════════════════════════════════
    margin_mode: MarginMode = MarginMode.RETAIL_NETTING
    margin_initial: float = 0.0           # Initial margin per lot (0 = use leverage)
    margin_maintenance: float = 0.0       # Maintenance margin per lot
    margin_long: float = 1.0              # Long margin rate multiplier
    margin_short: float = 1.0             # Short margin rate multiplier
    margin_limit: float = 0.0             # Margin limit
    margin_stop: float = 0.0              # Stop out margin level
    margin_stop_limit: float = 0.0        # Limit stop out level
    margin_hedged: float = 0.5            # Hedged margin (50% = half margin)
    
    # ═══════════════════════════════════════════════════════════════
    # SPREAD
    # ═══════════════════════════════════════════════════════════════
    spread: int = 0                       # Current spread in points
    spread_float: bool = True             # Floating spread
    spread_balance: int = 0               # Spread balance
    
    # Typical values for backtesting
    spread_typical: float = 10.0          # Typical spread in points
    spread_min: float = 5.0               # Minimum spread
    spread_max: float = 50.0              # Maximum spread (news/low liquidity)
    
    # ═══════════════════════════════════════════════════════════════
    # SWAP (OVERNIGHT INTEREST)
    # ═══════════════════════════════════════════════════════════════
    swap_mode: int = 1                    # 0=disabled, 1=points, 2=currency, 3=%
    swap_long: float = 0.0                # Swap for long positions
    swap_short: float = 0.0               # Swap for short positions
    swap_rollover3days: int = 3           # Triple swap day (3=Wednesday)
    swap_time: time = time(0, 0)          # Swap charge time (server)
    
    # ═══════════════════════════════════════════════════════════════
    # STOPS & LIMITS
    # ═══════════════════════════════════════════════════════════════
    stops_level: int = 0                  # Min distance for stops (points)
    freeze_level: int = 0                 # Freeze distance (points)
    
    # ═══════════════════════════════════════════════════════════════
    # TRADING
    # ═══════════════════════════════════════════════════════════════
    trade_mode: int = 4                   # 0=disabled, 4=full
    trade_allowed: bool = True
    trade_close_by_allowed: bool = True
    filling_mode: int = 1                 # Order fill mode
    expiration_mode: int = 15             # Order expiration modes allowed
    
    # Trading sessions
    sessions: List[TradingSession] = field(default_factory=list)
    
    # ═══════════════════════════════════════════════════════════════
    # SERVER/TIME
    # ═══════════════════════════════════════════════════════════════
    server_timezone: str = "UTC+2"        # Server timezone
    
    # ═══════════════════════════════════════════════════════════════
    # COMPUTED PROPERTIES
    # ═══════════════════════════════════════════════════════════════
    
    @property
    def pip_value(self) -> float:
        """
        Value of 1 pip per lot in profit currency.
        
        For 5-digit forex: 1 pip = 10 points
        For 3-digit forex (JPY): 1 pip = 10 points
        """
        points_per_pip = self.pip_size / self.point
        return self.tick_value * points_per_pip
    
    @property
    def point_value(self) -> float:
        """Value of 1 point per lot in profit currency."""
        return self.tick_value
    
    def calculate_pip_value(self, lots: float, price: float = 0.0) -> float:
        """
        Calculate pip value for given lot size.
        
        For forex, pip value in USD depends on:
        - Account currency
        - Quote currency
        - Current exchange rate (for non-USD quote currencies)
        
        Args:
            lots: Position size in lots
            price: Current price (needed for cross-rate calculation)
            
        Returns:
            Pip value in profit currency
        """
        return self.pip_value * lots
    
    def calculate_point_value(self, lots: float) -> float:
        """Calculate point value for given lot size."""
        return self.tick_value * lots
    
    def calculate_profit(
        self,
        direction: int,  # 1 = long, -1 = short
        lots: float,
        open_price: float,
        close_price: float,
    ) -> float:
        """
        Calculate profit/loss for a trade.
        
        Uses the appropriate calculation mode for the instrument.
        
        Args:
            direction: 1 for long, -1 for short
            lots: Position size in lots
            open_price: Entry price
            close_price: Exit price
            
        Returns:
            Profit/loss in profit currency
        """
        price_diff = (close_price - open_price) * direction
        
        if self.calc_mode in (CalcMode.FOREX, CalcMode.CFD, CalcMode.CFD_INDEX):
            # P&L = price_diff * contract_size * lots
            return price_diff * self.contract_size * lots
            
        elif self.calc_mode == CalcMode.FUTURES:
            # P&L = price_diff * (tick_value / tick_size) * lots
            if self.tick_size > 0:
                return price_diff * (self.tick_value / self.tick_size) * lots
            return 0.0
            
        elif self.calc_mode == CalcMode.EXCHANGE_STOCKS:
            # P&L = price_diff * lots (contract_size = 1 share)
            return price_diff * self.contract_size * lots
            
        else:
            # Default forex-style
            return price_diff * self.contract_size * lots
    
    def calculate_spread_cost(self, lots: float) -> float:
        """
        Calculate spread cost for entry.
        
        Spread cost = spread_in_price * contract_size * lots
        
        Args:
            lots: Position size in lots
            
        Returns:
            Spread cost in profit currency
        """
        spread_in_price = self.spread_typical * self.point
        return spread_in_price * self.contract_size * lots
    
    def calculate_margin(
        self,
        lots: float,
        price: float,
        leverage: float = 100.0,
        is_long: bool = True,
    ) -> float:
        """
        Calculate required margin.
        
        Args:
            lots: Position size
            price: Current price
            leverage: Account leverage
            is_long: True for long position
            
        Returns:
            Required margin in margin currency
        """
        if self.margin_initial > 0:
            # Fixed margin per lot
            return self.margin_initial * lots
        
        # Calculate from leverage
        notional = lots * self.contract_size * price
        margin_rate = self.margin_long if is_long else self.margin_short
        
        return (notional / leverage) * margin_rate
    
    def calculate_swap(
        self,
        is_long: bool,
        lots: float,
        price: float,
        days: int = 1,
    ) -> float:
        """
        Calculate swap for holding position.
        
        Args:
            is_long: True for long position
            lots: Position size
            price: Current price
            days: Number of days held
            
        Returns:
            Swap amount (positive = credit, negative = cost)
        """
        swap_rate = self.swap_long if is_long else self.swap_short
        
        if self.swap_mode == 1:  # Points
            return swap_rate * self.tick_value * lots * days
        elif self.swap_mode == 2:  # Currency
            return swap_rate * lots * days
        elif self.swap_mode == 3:  # Percentage
            notional = lots * self.contract_size * price
            return (swap_rate / 100 / 365) * notional * days
        
        return 0.0
    
    def normalize_volume(self, volume: float) -> float:
        """Normalize volume to valid lot size."""
        if volume < self.volume_min:
            return 0.0
        if volume > self.volume_max:
            volume = self.volume_max
        
        # Round to volume_step
        steps = round(volume / self.volume_step)
        return steps * self.volume_step
    
    def normalize_price(self, price: float) -> float:
        """Normalize price to tick size."""
        if self.tick_size <= 0:
            return price
        ticks = round(price / self.tick_size)
        return ticks * self.tick_size
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'symbol': self.symbol,
            'name': self.name,
            'description': self.description,
            'asset_class': self.asset_class.name,
            'base_currency': self.base_currency,
            'quote_currency': self.quote_currency,
            'digits': self.digits,
            'point': self.point,
            'pip_size': self.pip_size,
            'contract_size': self.contract_size,
            'tick_size': self.tick_size,
            'tick_value': self.tick_value,
            'calc_mode': self.calc_mode.name,
            'volume_min': self.volume_min,
            'volume_max': self.volume_max,
            'volume_step': self.volume_step,
            'spread_typical': self.spread_typical,
            'swap_long': self.swap_long,
            'swap_short': self.swap_short,
            'swap_rollover3days': self.swap_rollover3days,
            'margin_initial': self.margin_initial,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SymbolInfo':
        """Create from dictionary."""
        # Handle enums
        if 'asset_class' in data and isinstance(data['asset_class'], str):
            data['asset_class'] = AssetClass[data['asset_class']]
        if 'calc_mode' in data and isinstance(data['calc_mode'], str):
            data['calc_mode'] = CalcMode[data['calc_mode']]
        
        # Filter to known fields
        known_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in known_fields}
        
        return cls(**filtered)


# ═══════════════════════════════════════════════════════════════════════════
# PRE-DEFINED SYMBOL SPECIFICATIONS
# ═══════════════════════════════════════════════════════════════════════════

# Helper function to create forex symbol
def _forex_symbol(
    symbol: str,
    description: str,
    base: str,
    quote: str,
    digits: int = 5,
    spread: float = 10.0,
    swap_long: float = 0.0,
    swap_short: float = 0.0,
) -> SymbolInfo:
    """Create forex symbol with standard specs."""
    is_jpy = quote == "JPY" or base == "JPY"
    
    return SymbolInfo(
        symbol=symbol,
        name=symbol,
        description=description,
        asset_class=AssetClass.FOREX_MAJOR,
        base_currency=base,
        quote_currency=quote,
        margin_currency=base,
        digits=digits,
        point=0.001 if is_jpy else 0.00001,
        pip_digits=digits - 1,
        pip_size=0.01 if is_jpy else 0.0001,
        contract_size=100000,
        tick_size=0.001 if is_jpy else 0.00001,
        tick_value=1.0 if quote == "USD" else 1.0,  # Approximately $1 per point
        calc_mode=CalcMode.FOREX,
        volume_min=0.01,
        volume_max=100.0,
        volume_step=0.01,
        spread_typical=spread,
        spread_min=spread / 2,
        spread_max=spread * 5,
        swap_long=swap_long,
        swap_short=swap_short,
        swap_rollover3days=3,  # Wednesday
    )


# ═══════════════════════════════════════════════════════════════════════════
# FOREX MAJORS
# ═══════════════════════════════════════════════════════════════════════════

EURUSD = SymbolInfo(
    symbol="EURUSD",
    name="EURUSD",
    description="Euro vs US Dollar",
    asset_class=AssetClass.FOREX_MAJOR,
    base_currency="EUR",
    quote_currency="USD",
    margin_currency="EUR",
    digits=5,
    point=0.00001,
    pip_digits=4,
    pip_size=0.0001,
    contract_size=100000,  # 100,000 EUR
    tick_size=0.00001,
    tick_value=1.0,  # $1 per point per lot (100,000 * 0.00001 = $1)
    calc_mode=CalcMode.FOREX,
    volume_min=0.01,
    volume_max=100.0,
    volume_step=0.01,
    spread_typical=10.0,  # 1 pip typical
    spread_min=5.0,
    spread_max=30.0,
    swap_long=-6.5,
    swap_short=1.2,
    swap_rollover3days=3,
)

GBPUSD = SymbolInfo(
    symbol="GBPUSD",
    name="GBPUSD",
    description="British Pound vs US Dollar",
    asset_class=AssetClass.FOREX_MAJOR,
    base_currency="GBP",
    quote_currency="USD",
    margin_currency="GBP",
    digits=5,
    point=0.00001,
    pip_digits=4,
    pip_size=0.0001,
    contract_size=100000,
    tick_size=0.00001,
    tick_value=1.0,
    calc_mode=CalcMode.FOREX,
    volume_min=0.01,
    volume_max=100.0,
    volume_step=0.01,
    spread_typical=15.0,
    spread_min=8.0,
    spread_max=40.0,
    swap_long=-4.2,
    swap_short=-1.8,
    swap_rollover3days=3,
)

USDJPY = SymbolInfo(
    symbol="USDJPY",
    name="USDJPY",
    description="US Dollar vs Japanese Yen",
    asset_class=AssetClass.FOREX_MAJOR,
    base_currency="USD",
    quote_currency="JPY",
    margin_currency="USD",
    digits=3,
    point=0.001,
    pip_digits=2,
    pip_size=0.01,
    contract_size=100000,
    tick_size=0.001,
    tick_value=0.67,  # ~$0.67 per point at 150 JPY/USD (100000 * 0.001 / 150)
    calc_mode=CalcMode.FOREX,
    volume_min=0.01,
    volume_max=100.0,
    volume_step=0.01,
    spread_typical=12.0,
    spread_min=6.0,
    spread_max=35.0,
    swap_long=7.26,
    swap_short=-16.66,
    swap_rollover3days=3,
)

AUDUSD = SymbolInfo(
    symbol="AUDUSD",
    name="AUDUSD",
    description="Australian Dollar vs US Dollar",
    asset_class=AssetClass.FOREX_MAJOR,
    base_currency="AUD",
    quote_currency="USD",
    margin_currency="AUD",
    digits=5,
    point=0.00001,
    pip_digits=4,
    pip_size=0.0001,
    contract_size=100000,
    tick_size=0.00001,
    tick_value=1.0,
    calc_mode=CalcMode.FOREX,
    volume_min=0.01,
    volume_max=100.0,
    volume_step=0.01,
    spread_typical=12.0,
    spread_min=6.0,
    spread_max=35.0,
    swap_long=-2.5,
    swap_short=-1.2,
    swap_rollover3days=3,
)

# ═══════════════════════════════════════════════════════════════════════════
# METALS
# ═══════════════════════════════════════════════════════════════════════════

XAUUSD = SymbolInfo(
    symbol="XAUUSD",
    name="XAUUSD",
    description="Gold vs US Dollar",
    asset_class=AssetClass.COMMODITY_METAL,
    base_currency="XAU",
    quote_currency="USD",
    margin_currency="USD",
    digits=2,
    point=0.01,
    pip_digits=1,
    pip_size=0.1,  # $0.10 move is common "pip" for gold
    contract_size=100,  # 100 troy ounces
    tick_size=0.01,
    tick_value=1.0,  # $1 per $0.01 move per lot (100oz * $0.01 = $1)
    calc_mode=CalcMode.CFD,
    volume_min=0.01,
    volume_max=50.0,
    volume_step=0.01,
    spread_typical=30.0,  # $0.30 typical
    spread_min=15.0,
    spread_max=100.0,
    swap_long=-68.09,
    swap_short=47.91,
    swap_rollover3days=3,
)

XAGUSD = SymbolInfo(
    symbol="XAGUSD",
    name="XAGUSD",
    description="Silver vs US Dollar",
    asset_class=AssetClass.COMMODITY_METAL,
    base_currency="XAG",
    quote_currency="USD",
    margin_currency="USD",
    digits=3,
    point=0.001,
    pip_digits=2,
    pip_size=0.01,
    contract_size=5000,  # 5000 troy ounces
    tick_size=0.001,
    tick_value=5.0,  # $5 per $0.001 move per lot (5000oz * $0.001 = $5)
    calc_mode=CalcMode.CFD,
    volume_min=0.01,
    volume_max=50.0,
    volume_step=0.01,
    spread_typical=30.0,
    spread_min=15.0,
    spread_max=80.0,
    swap_long=-10.0,
    swap_short=2.0,
    swap_rollover3days=3,
)

# ═══════════════════════════════════════════════════════════════════════════
# INDICES
# ═══════════════════════════════════════════════════════════════════════════

US500 = SymbolInfo(
    symbol="US500",
    name="US500",
    description="S&P 500 Index CFD",
    asset_class=AssetClass.INDEX_US,
    base_currency="USD",
    quote_currency="USD",
    margin_currency="USD",
    digits=1,
    point=0.1,
    pip_digits=0,
    pip_size=1.0,  # 1 point
    contract_size=1,  # 1 unit per lot (CFD)
    tick_size=0.1,
    tick_value=0.1,  # $0.10 per 0.1 point move per lot
    calc_mode=CalcMode.CFD_INDEX,
    volume_min=0.1,
    volume_max=100.0,
    volume_step=0.1,
    spread_typical=5.0,  # 0.5 points
    spread_min=3.0,
    spread_max=20.0,
    swap_long=-8.0,
    swap_short=-2.0,
    swap_rollover3days=5,  # Friday for indices
)

US30 = SymbolInfo(
    symbol="US30",
    name="US30",
    description="Dow Jones 30 Index CFD",
    asset_class=AssetClass.INDEX_US,
    base_currency="USD",
    quote_currency="USD",
    margin_currency="USD",
    digits=1,
    point=0.1,
    pip_digits=0,
    pip_size=1.0,
    contract_size=1,
    tick_size=0.1,
    tick_value=0.1,
    calc_mode=CalcMode.CFD_INDEX,
    volume_min=0.1,
    volume_max=100.0,
    volume_step=0.1,
    spread_typical=20.0,  # 2 points
    spread_min=10.0,
    spread_max=50.0,
    swap_long=-10.0,
    swap_short=-3.0,
    swap_rollover3days=5,
)

NAS100 = SymbolInfo(
    symbol="NAS100",
    name="NAS100",
    description="Nasdaq 100 Index CFD",
    asset_class=AssetClass.INDEX_US,
    base_currency="USD",
    quote_currency="USD",
    margin_currency="USD",
    digits=1,
    point=0.1,
    pip_digits=0,
    pip_size=1.0,
    contract_size=1,
    tick_size=0.1,
    tick_value=0.1,
    calc_mode=CalcMode.CFD_INDEX,
    volume_min=0.1,
    volume_max=100.0,
    volume_step=0.1,
    spread_typical=15.0,
    spread_min=8.0,
    spread_max=40.0,
    swap_long=-12.0,
    swap_short=-4.0,
    swap_rollover3days=5,
)

NIKKEI225 = SymbolInfo(
    symbol="Nikkei225",
    name="Nikkei225",
    description="Nikkei 225 Index CFD",
    asset_class=AssetClass.INDEX_ASIA,
    base_currency="JPY",
    quote_currency="JPY",
    margin_currency="JPY",
    digits=0,
    point=1.0,
    pip_digits=0,
    pip_size=1.0,
    contract_size=1,
    tick_size=1.0,
    tick_value=1.0,  # 1 JPY per point
    calc_mode=CalcMode.CFD_INDEX,
    volume_min=0.1,
    volume_max=100.0,
    volume_step=0.1,
    spread_typical=10.0,
    spread_min=5.0,
    spread_max=30.0,
    swap_long=-5.0,
    swap_short=-5.0,
    swap_rollover3days=5,
)

DJ30FT = SymbolInfo(
    symbol="DJ30ft",
    name="DJ30ft",
    description="Dow Jones 30 Futures",
    asset_class=AssetClass.INDEX_US,
    base_currency="USD",
    quote_currency="USD",
    margin_currency="USD",
    digits=0,
    point=1.0,
    pip_digits=0,
    pip_size=1.0,
    contract_size=1,
    tick_size=1.0,
    tick_value=1.0,
    calc_mode=CalcMode.CFD_INDEX,
    volume_min=0.1,
    volume_max=100.0,
    volume_step=0.1,
    spread_typical=30.0,
    spread_min=15.0,
    spread_max=80.0,
    swap_long=-8.0,
    swap_short=-3.0,
    swap_rollover3days=5,
)

# ═══════════════════════════════════════════════════════════════════════════
# CRYPTO
# ═══════════════════════════════════════════════════════════════════════════

BTCUSD = SymbolInfo(
    symbol="BTCUSD",
    name="BTCUSD",
    description="Bitcoin vs US Dollar",
    asset_class=AssetClass.CRYPTO,
    base_currency="BTC",
    quote_currency="USD",
    margin_currency="USD",
    digits=2,
    point=0.01,
    pip_digits=0,
    pip_size=1.0,  # $1 is common "pip" for BTC
    contract_size=1,  # 1 BTC per lot
    tick_size=0.01,
    tick_value=0.01,  # $0.01 per $0.01 move per lot (1 BTC)
    calc_mode=CalcMode.CFD,
    volume_min=0.01,
    volume_max=10.0,
    volume_step=0.01,
    spread_typical=5000.0,  # $50 typical
    spread_min=2000.0,
    spread_max=20000.0,
    swap_long=-25.0,
    swap_short=-25.0,
    swap_rollover3days=3,
)

ETHUSD = SymbolInfo(
    symbol="ETHUSD",
    name="ETHUSD",
    description="Ethereum vs US Dollar",
    asset_class=AssetClass.CRYPTO,
    base_currency="ETH",
    quote_currency="USD",
    margin_currency="USD",
    digits=2,
    point=0.01,
    pip_digits=0,
    pip_size=1.0,
    contract_size=1,  # 1 ETH per lot
    tick_size=0.01,
    tick_value=0.01,
    calc_mode=CalcMode.CFD,
    volume_min=0.01,
    volume_max=50.0,
    volume_step=0.01,
    spread_typical=300.0,  # $3 typical
    spread_min=100.0,
    spread_max=1000.0,
    swap_long=-20.0,
    swap_short=-20.0,
    swap_rollover3days=3,
)

# ═══════════════════════════════════════════════════════════════════════════
# COMMODITIES
# ═══════════════════════════════════════════════════════════════════════════

USOIL = SymbolInfo(
    symbol="USOIL",
    name="USOIL",
    description="US Crude Oil WTI",
    asset_class=AssetClass.COMMODITY_ENERGY,
    base_currency="OIL",
    quote_currency="USD",
    margin_currency="USD",
    digits=2,
    point=0.01,
    pip_digits=1,
    pip_size=0.1,
    contract_size=1000,  # 1000 barrels per lot
    tick_size=0.01,
    tick_value=10.0,  # $10 per $0.01 move per lot (1000 * $0.01 = $10)
    calc_mode=CalcMode.CFD,
    volume_min=0.01,
    volume_max=50.0,
    volume_step=0.01,
    spread_typical=5.0,  # $0.05 typical
    spread_min=3.0,
    spread_max=20.0,
    swap_long=-10.0,
    swap_short=-10.0,
    swap_rollover3days=3,
)

UKOUSD = SymbolInfo(
    symbol="UKOUSD",
    name="UKOUSD",
    description="Brent Crude Oil",
    asset_class=AssetClass.COMMODITY_ENERGY,
    base_currency="OIL",
    quote_currency="USD",
    margin_currency="USD",
    digits=2,
    point=0.01,
    pip_digits=1,
    pip_size=0.1,
    contract_size=1000,
    tick_size=0.01,
    tick_value=10.0,
    calc_mode=CalcMode.CFD,
    volume_min=0.01,
    volume_max=50.0,
    volume_step=0.01,
    spread_typical=6.0,
    spread_min=3.0,
    spread_max=25.0,
    swap_long=-8.0,
    swap_short=-8.0,
    swap_rollover3days=3,
)

COPPER = SymbolInfo(
    symbol="COPPER-C",
    name="COPPER-C",
    description="Copper Futures CFD",
    asset_class=AssetClass.COMMODITY_METAL,
    base_currency="COPPER",
    quote_currency="USD",
    margin_currency="USD",
    digits=4,
    point=0.0001,
    pip_digits=3,
    pip_size=0.001,
    contract_size=25000,  # 25,000 lbs per lot
    tick_size=0.0001,
    tick_value=2.5,  # $2.50 per 0.0001 move per lot (25000 * 0.0001 = $2.50)
    calc_mode=CalcMode.CFD,
    volume_min=0.1,
    volume_max=50.0,
    volume_step=0.1,
    spread_typical=15.0,
    spread_min=8.0,
    spread_max=40.0,
    swap_long=-8.0,
    swap_short=-4.0,
    swap_rollover3days=3,
)


# ═══════════════════════════════════════════════════════════════════════════
# SYMBOL REGISTRY
# ═══════════════════════════════════════════════════════════════════════════

SYMBOL_REGISTRY: Dict[str, SymbolInfo] = {
    # Forex
    "EURUSD": EURUSD,
    "GBPUSD": GBPUSD,
    "USDJPY": USDJPY,
    "AUDUSD": AUDUSD,
    "GBPUSD+": GBPUSD,  # Alias
    
    # Metals
    "XAUUSD": XAUUSD,
    "XAUUSD+": XAUUSD,  # Alias
    "XAGUSD": XAGUSD,
    
    # Indices
    "US500": US500,
    "US30": US30,
    "NAS100": NAS100,
    "Nikkei225": NIKKEI225,
    "DJ30ft": DJ30FT,
    
    # Crypto
    "BTCUSD": BTCUSD,
    "ETHUSD": ETHUSD,
    
    # Commodities
    "USOIL": USOIL,
    "UKOUSD": UKOUSD,
    "COPPER-C": COPPER,
}


def get_symbol_info(symbol: str) -> SymbolInfo:
    """
    Get symbol info from registry.
    
    Args:
        symbol: Symbol name
        
    Returns:
        SymbolInfo (default if not found)
    """
    # Exact match
    if symbol in SYMBOL_REGISTRY:
        return SYMBOL_REGISTRY[symbol]
    
    # Try uppercase
    upper = symbol.upper()
    if upper in SYMBOL_REGISTRY:
        return SYMBOL_REGISTRY[upper]
    
    # Try removing suffix
    base = symbol.split('.')[0].split('_')[0].replace('+', '')
    if base in SYMBOL_REGISTRY:
        return SYMBOL_REGISTRY[base]
    if base.upper() in SYMBOL_REGISTRY:
        return SYMBOL_REGISTRY[base.upper()]
    
    # Return default
    logger.warning(f"Symbol {symbol} not found in registry, returning default")
    return SymbolInfo(symbol=symbol)


def register_symbol(info: SymbolInfo) -> None:
    """Register a symbol in the registry."""
    SYMBOL_REGISTRY[info.symbol] = info


def list_symbols() -> List[str]:
    """List all registered symbols."""
    return sorted(SYMBOL_REGISTRY.keys())


# ═══════════════════════════════════════════════════════════════════════════
# UTILITY FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

def calculate_position_value(
    symbol: str,
    lots: float,
    price: float,
) -> float:
    """Calculate position notional value."""
    info = get_symbol_info(symbol)
    return lots * info.contract_size * price


def calculate_pip_value(
    symbol: str,
    lots: float,
) -> float:
    """Calculate pip value for symbol and lot size."""
    info = get_symbol_info(symbol)
    return info.pip_value * lots


def calculate_profit_loss(
    symbol: str,
    direction: int,
    lots: float,
    open_price: float,
    close_price: float,
) -> float:
    """Calculate profit/loss for a trade."""
    info = get_symbol_info(symbol)
    return info.calculate_profit(direction, lots, open_price, close_price)


# Export
__all__ = [
    'AssetClass',
    'CalcMode',
    'MarginMode',
    'TradingSession',
    'SymbolInfo',
    'SYMBOL_REGISTRY',
    'get_symbol_info',
    'register_symbol',
    'list_symbols',
    'calculate_position_value',
    'calculate_pip_value',
    'calculate_profit_loss',
    # Pre-defined symbols
    'EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD',
    'XAUUSD', 'XAGUSD',
    'US500', 'US30', 'NAS100', 'NIKKEI225', 'DJ30FT',
    'BTCUSD', 'ETHUSD',
    'USOIL', 'UKOUSD', 'COPPER',
]
