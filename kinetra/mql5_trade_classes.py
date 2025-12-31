"""
MQL5 Trade Classes - Python Implementation
==========================================

Python equivalents of MQL5 Standard Library trade classes:
- CAccountInfo: Trade account properties
- CSymbolInfo: Trade instrument properties  
- COrderInfo: Pending order properties
- CHistoryOrderInfo: History order properties
- CPositionInfo: Open position properties
- CDealInfo: History deal properties
- CTrade: Trade operations execution
- CTerminalInfo: Terminal environment properties

These classes mirror the MQL5 API structure for seamless integration
between Python trading systems and MetaTrader 5.

Reference: https://www.mql5.com/en/docs/standardlibrary/tradeclasses
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import IntEnum, auto
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# MQL5 ENUMERATIONS
# ═══════════════════════════════════════════════════════════════════════════

class ENUM_ACCOUNT_TRADE_MODE(IntEnum):
    """Account trade mode."""
    ACCOUNT_TRADE_MODE_DEMO = 0
    ACCOUNT_TRADE_MODE_CONTEST = 1
    ACCOUNT_TRADE_MODE_REAL = 2


class ENUM_ACCOUNT_STOPOUT_MODE(IntEnum):
    """Stop out calculation mode."""
    ACCOUNT_STOPOUT_MODE_PERCENT = 0
    ACCOUNT_STOPOUT_MODE_MONEY = 1


class ENUM_ACCOUNT_MARGIN_MODE(IntEnum):
    """Margin calculation mode."""
    ACCOUNT_MARGIN_MODE_RETAIL_NETTING = 0
    ACCOUNT_MARGIN_MODE_EXCHANGE = 1
    ACCOUNT_MARGIN_MODE_RETAIL_HEDGING = 2


class ENUM_SYMBOL_CALC_MODE(IntEnum):
    """Profit calculation mode."""
    SYMBOL_CALC_MODE_FOREX = 0
    SYMBOL_CALC_MODE_FOREX_NO_LEVERAGE = 1
    SYMBOL_CALC_MODE_FUTURES = 2
    SYMBOL_CALC_MODE_CFD = 3
    SYMBOL_CALC_MODE_CFDINDEX = 4
    SYMBOL_CALC_MODE_CFDLEVERAGE = 5
    SYMBOL_CALC_MODE_EXCH_STOCKS = 32
    SYMBOL_CALC_MODE_EXCH_FUTURES = 33
    SYMBOL_CALC_MODE_EXCH_OPTIONS = 34


class ENUM_SYMBOL_TRADE_MODE(IntEnum):
    """Symbol trade mode."""
    SYMBOL_TRADE_MODE_DISABLED = 0
    SYMBOL_TRADE_MODE_LONGONLY = 1
    SYMBOL_TRADE_MODE_SHORTONLY = 2
    SYMBOL_TRADE_MODE_CLOSEONLY = 3
    SYMBOL_TRADE_MODE_FULL = 4


class ENUM_SYMBOL_TRADE_EXECUTION(IntEnum):
    """Trade execution mode."""
    SYMBOL_TRADE_EXECUTION_REQUEST = 0
    SYMBOL_TRADE_EXECUTION_INSTANT = 1
    SYMBOL_TRADE_EXECUTION_MARKET = 2
    SYMBOL_TRADE_EXECUTION_EXCHANGE = 3


class ENUM_SYMBOL_SWAP_MODE(IntEnum):
    """Swap calculation mode."""
    SYMBOL_SWAP_MODE_DISABLED = 0
    SYMBOL_SWAP_MODE_POINTS = 1
    SYMBOL_SWAP_MODE_CURRENCY_SYMBOL = 2
    SYMBOL_SWAP_MODE_CURRENCY_MARGIN = 3
    SYMBOL_SWAP_MODE_CURRENCY_DEPOSIT = 4
    SYMBOL_SWAP_MODE_INTEREST_CURRENT = 5
    SYMBOL_SWAP_MODE_INTEREST_OPEN = 6
    SYMBOL_SWAP_MODE_REOPEN_CURRENT = 7
    SYMBOL_SWAP_MODE_REOPEN_BID = 8


class ENUM_DAY_OF_WEEK(IntEnum):
    """Day of week."""
    SUNDAY = 0
    MONDAY = 1
    TUESDAY = 2
    WEDNESDAY = 3
    THURSDAY = 4
    FRIDAY = 5
    SATURDAY = 6


class ENUM_ORDER_TYPE(IntEnum):
    """Order type."""
    ORDER_TYPE_BUY = 0
    ORDER_TYPE_SELL = 1
    ORDER_TYPE_BUY_LIMIT = 2
    ORDER_TYPE_SELL_LIMIT = 3
    ORDER_TYPE_BUY_STOP = 4
    ORDER_TYPE_SELL_STOP = 5
    ORDER_TYPE_BUY_STOP_LIMIT = 6
    ORDER_TYPE_SELL_STOP_LIMIT = 7
    ORDER_TYPE_CLOSE_BY = 8


class ENUM_ORDER_STATE(IntEnum):
    """Order state."""
    ORDER_STATE_STARTED = 0
    ORDER_STATE_PLACED = 1
    ORDER_STATE_CANCELED = 2
    ORDER_STATE_PARTIAL = 3
    ORDER_STATE_FILLED = 4
    ORDER_STATE_REJECTED = 5
    ORDER_STATE_EXPIRED = 6
    ORDER_STATE_REQUEST_ADD = 7
    ORDER_STATE_REQUEST_MODIFY = 8
    ORDER_STATE_REQUEST_CANCEL = 9


class ENUM_ORDER_TYPE_FILLING(IntEnum):
    """Order filling mode."""
    ORDER_FILLING_FOK = 0  # Fill or Kill
    ORDER_FILLING_IOC = 1  # Immediate or Cancel
    ORDER_FILLING_RETURN = 2  # Return


class ENUM_ORDER_TYPE_TIME(IntEnum):
    """Order expiration type."""
    ORDER_TIME_GTC = 0  # Good Till Cancelled
    ORDER_TIME_DAY = 1  # Day order
    ORDER_TIME_SPECIFIED = 2  # Specified expiration
    ORDER_TIME_SPECIFIED_DAY = 3  # Specified day


class ENUM_POSITION_TYPE(IntEnum):
    """Position type."""
    POSITION_TYPE_BUY = 0
    POSITION_TYPE_SELL = 1


class ENUM_POSITION_REASON(IntEnum):
    """Position open reason."""
    POSITION_REASON_CLIENT = 0
    POSITION_REASON_MOBILE = 1
    POSITION_REASON_WEB = 2
    POSITION_REASON_EXPERT = 3


class ENUM_DEAL_TYPE(IntEnum):
    """Deal type."""
    DEAL_TYPE_BUY = 0
    DEAL_TYPE_SELL = 1
    DEAL_TYPE_BALANCE = 2
    DEAL_TYPE_CREDIT = 3
    DEAL_TYPE_CHARGE = 4
    DEAL_TYPE_CORRECTION = 5
    DEAL_TYPE_BONUS = 6
    DEAL_TYPE_COMMISSION = 7
    DEAL_TYPE_COMMISSION_DAILY = 8
    DEAL_TYPE_COMMISSION_MONTHLY = 9
    DEAL_TYPE_COMMISSION_AGENT_DAILY = 10
    DEAL_TYPE_COMMISSION_AGENT_MONTHLY = 11
    DEAL_TYPE_INTEREST = 12
    DEAL_TYPE_BUY_CANCELED = 13
    DEAL_TYPE_SELL_CANCELED = 14


class ENUM_DEAL_ENTRY(IntEnum):
    """Deal entry type."""
    DEAL_ENTRY_IN = 0
    DEAL_ENTRY_OUT = 1
    DEAL_ENTRY_INOUT = 2
    DEAL_ENTRY_OUT_BY = 3


class ENUM_DEAL_REASON(IntEnum):
    """Deal reason."""
    DEAL_REASON_CLIENT = 0
    DEAL_REASON_MOBILE = 1
    DEAL_REASON_WEB = 2
    DEAL_REASON_EXPERT = 3
    DEAL_REASON_SL = 4
    DEAL_REASON_TP = 5
    DEAL_REASON_SO = 6
    DEAL_REASON_ROLLOVER = 7
    DEAL_REASON_VMARGIN = 8
    DEAL_REASON_SPLIT = 9


class ENUM_TRADE_REQUEST_ACTIONS(IntEnum):
    """Trade request action type."""
    TRADE_ACTION_DEAL = 1      # Market order
    TRADE_ACTION_PENDING = 5   # Pending order
    TRADE_ACTION_SLTP = 6      # Modify SL/TP
    TRADE_ACTION_MODIFY = 7    # Modify pending order
    TRADE_ACTION_REMOVE = 8    # Delete pending order
    TRADE_ACTION_CLOSE_BY = 10 # Close by opposite position


class ENUM_TRADE_RETCODE(IntEnum):
    """Trade server return codes."""
    TRADE_RETCODE_REQUOTE = 10004
    TRADE_RETCODE_REJECT = 10006
    TRADE_RETCODE_CANCEL = 10007
    TRADE_RETCODE_PLACED = 10008
    TRADE_RETCODE_DONE = 10009
    TRADE_RETCODE_DONE_PARTIAL = 10010
    TRADE_RETCODE_ERROR = 10011
    TRADE_RETCODE_TIMEOUT = 10012
    TRADE_RETCODE_INVALID = 10013
    TRADE_RETCODE_INVALID_VOLUME = 10014
    TRADE_RETCODE_INVALID_PRICE = 10015
    TRADE_RETCODE_INVALID_STOPS = 10016
    TRADE_RETCODE_TRADE_DISABLED = 10017
    TRADE_RETCODE_MARKET_CLOSED = 10018
    TRADE_RETCODE_NO_MONEY = 10019
    TRADE_RETCODE_PRICE_CHANGED = 10020
    TRADE_RETCODE_PRICE_OFF = 10021
    TRADE_RETCODE_INVALID_EXPIRATION = 10022
    TRADE_RETCODE_ORDER_CHANGED = 10023
    TRADE_RETCODE_TOO_MANY_REQUESTS = 10024
    TRADE_RETCODE_NO_CHANGES = 10025
    TRADE_RETCODE_SERVER_DISABLES_AT = 10026
    TRADE_RETCODE_CLIENT_DISABLES_AT = 10027
    TRADE_RETCODE_LOCKED = 10028
    TRADE_RETCODE_FROZEN = 10029
    TRADE_RETCODE_INVALID_FILL = 10030
    TRADE_RETCODE_CONNECTION = 10031
    TRADE_RETCODE_ONLY_REAL = 10032
    TRADE_RETCODE_LIMIT_ORDERS = 10033
    TRADE_RETCODE_LIMIT_VOLUME = 10034
    TRADE_RETCODE_INVALID_ORDER = 10035
    TRADE_RETCODE_POSITION_CLOSED = 10036
    TRADE_RETCODE_INVALID_CLOSE_VOLUME = 10038
    TRADE_RETCODE_CLOSE_ORDER_EXIST = 10039
    TRADE_RETCODE_LIMIT_POSITIONS = 10040
    TRADE_RETCODE_REJECT_CANCEL = 10041
    TRADE_RETCODE_LONG_ONLY = 10042
    TRADE_RETCODE_SHORT_ONLY = 10043
    TRADE_RETCODE_CLOSE_ONLY = 10044
    TRADE_RETCODE_FIFO_CLOSE = 10045


# ═══════════════════════════════════════════════════════════════════════════
# CAccountInfo - Account Properties
# ═══════════════════════════════════════════════════════════════════════════

class CAccountInfo:
    """
    Class for working with trade account properties.
    
    Provides access to account information including:
    - Balance, equity, margin
    - Leverage and trade permissions
    - Account type (demo/real)
    """
    
    def __init__(self):
        """Initialize account info."""
        self._login: int = 0
        self._trade_mode: ENUM_ACCOUNT_TRADE_MODE = ENUM_ACCOUNT_TRADE_MODE.ACCOUNT_TRADE_MODE_DEMO
        self._leverage: int = 100
        self._limit_orders: int = 200
        self._margin_so_mode: ENUM_ACCOUNT_STOPOUT_MODE = ENUM_ACCOUNT_STOPOUT_MODE.ACCOUNT_STOPOUT_MODE_PERCENT
        self._trade_allowed: bool = True
        self._trade_expert: bool = True
        self._margin_mode: ENUM_ACCOUNT_MARGIN_MODE = ENUM_ACCOUNT_MARGIN_MODE.ACCOUNT_MARGIN_MODE_RETAIL_HEDGING
        
        # Double properties
        self._balance: float = 10000.0
        self._credit: float = 0.0
        self._profit: float = 0.0
        self._equity: float = 10000.0
        self._margin: float = 0.0
        self._margin_free: float = 10000.0
        self._margin_level: float = 0.0
        self._margin_so_call: float = 100.0
        self._margin_so_so: float = 50.0
        self._margin_initial: float = 0.0
        self._margin_maintenance: float = 0.0
        self._assets: float = 0.0
        self._liabilities: float = 0.0
        self._commission_blocked: float = 0.0
        
        # String properties
        self._name: str = ""
        self._server: str = ""
        self._currency: str = "USD"
        self._company: str = ""
    
    # ─────────────────────────────────────────────────────────────────
    # Integer Properties
    # ─────────────────────────────────────────────────────────────────
    
    def Login(self) -> int:
        """Get account number."""
        return self._login
    
    def TradeMode(self) -> ENUM_ACCOUNT_TRADE_MODE:
        """Get account trade mode (demo/contest/real)."""
        return self._trade_mode
    
    def TradeModeDescription(self) -> str:
        """Get trade mode description."""
        modes = {
            ENUM_ACCOUNT_TRADE_MODE.ACCOUNT_TRADE_MODE_DEMO: "Demo",
            ENUM_ACCOUNT_TRADE_MODE.ACCOUNT_TRADE_MODE_CONTEST: "Contest",
            ENUM_ACCOUNT_TRADE_MODE.ACCOUNT_TRADE_MODE_REAL: "Real",
        }
        return modes.get(self._trade_mode, "Unknown")
    
    def Leverage(self) -> int:
        """Get account leverage."""
        return self._leverage
    
    def LimitOrders(self) -> int:
        """Get maximum allowed pending orders."""
        return self._limit_orders
    
    def MarginSOMode(self) -> ENUM_ACCOUNT_STOPOUT_MODE:
        """Get stop out mode."""
        return self._margin_so_mode
    
    def TradeAllowed(self) -> bool:
        """Check if trading is allowed."""
        return self._trade_allowed
    
    def TradeExpert(self) -> bool:
        """Check if expert trading is allowed."""
        return self._trade_expert
    
    def MarginMode(self) -> ENUM_ACCOUNT_MARGIN_MODE:
        """Get margin calculation mode."""
        return self._margin_mode
    
    def MarginModeDescription(self) -> str:
        """Get margin mode description."""
        modes = {
            ENUM_ACCOUNT_MARGIN_MODE.ACCOUNT_MARGIN_MODE_RETAIL_NETTING: "Retail Netting",
            ENUM_ACCOUNT_MARGIN_MODE.ACCOUNT_MARGIN_MODE_EXCHANGE: "Exchange",
            ENUM_ACCOUNT_MARGIN_MODE.ACCOUNT_MARGIN_MODE_RETAIL_HEDGING: "Retail Hedging",
        }
        return modes.get(self._margin_mode, "Unknown")
    
    # ─────────────────────────────────────────────────────────────────
    # Double Properties
    # ─────────────────────────────────────────────────────────────────
    
    def Balance(self) -> float:
        """Get account balance."""
        return self._balance
    
    def Credit(self) -> float:
        """Get account credit."""
        return self._credit
    
    def Profit(self) -> float:
        """Get current profit."""
        return self._profit
    
    def Equity(self) -> float:
        """Get account equity."""
        return self._equity
    
    def Margin(self) -> float:
        """Get used margin."""
        return self._margin
    
    def FreeMargin(self) -> float:
        """Get free margin."""
        return self._margin_free
    
    def MarginLevel(self) -> float:
        """Get margin level percentage."""
        return self._margin_level
    
    def MarginCall(self) -> float:
        """Get margin call level."""
        return self._margin_so_call
    
    def MarginStopOut(self) -> float:
        """Get stop out level."""
        return self._margin_so_so
    
    def MarginInitial(self) -> float:
        """Get initial margin."""
        return self._margin_initial
    
    def MarginMaintenance(self) -> float:
        """Get maintenance margin."""
        return self._margin_maintenance
    
    def Assets(self) -> float:
        """Get account assets."""
        return self._assets
    
    def Liabilities(self) -> float:
        """Get account liabilities."""
        return self._liabilities
    
    def CommissionBlocked(self) -> float:
        """Get blocked commission."""
        return self._commission_blocked
    
    # ─────────────────────────────────────────────────────────────────
    # String Properties
    # ─────────────────────────────────────────────────────────────────
    
    def Name(self) -> str:
        """Get account holder name."""
        return self._name
    
    def Server(self) -> str:
        """Get trade server name."""
        return self._server
    
    def Currency(self) -> str:
        """Get account currency."""
        return self._currency
    
    def Company(self) -> str:
        """Get broker company name."""
        return self._company
    
    # ─────────────────────────────────────────────────────────────────
    # Utility Methods
    # ─────────────────────────────────────────────────────────────────
    
    def InfoInteger(self, prop_id: int) -> int:
        """Get integer property by ID."""
        # Implement based on MT5 property IDs
        return 0
    
    def InfoDouble(self, prop_id: int) -> float:
        """Get double property by ID."""
        return 0.0
    
    def InfoString(self, prop_id: int) -> str:
        """Get string property by ID."""
        return ""
    
    def OrderProfitCheck(
        self,
        symbol: str,
        order_type: ENUM_ORDER_TYPE,
        volume: float,
        price_open: float,
        price_close: float,
    ) -> float:
        """
        Calculate potential profit for order.
        
        Args:
            symbol: Symbol name
            order_type: Order type
            volume: Order volume
            price_open: Open price
            price_close: Close price
            
        Returns:
            Potential profit
        """
        # This would need symbol info for accurate calculation
        direction = 1 if order_type in (ENUM_ORDER_TYPE.ORDER_TYPE_BUY,) else -1
        return (price_close - price_open) * direction * volume * 100000  # Simplified
    
    def MarginCheck(
        self,
        symbol: str,
        order_type: ENUM_ORDER_TYPE,
        volume: float,
        price: float,
    ) -> float:
        """
        Check margin required for order.
        
        Args:
            symbol: Symbol name
            order_type: Order type
            volume: Order volume
            price: Order price
            
        Returns:
            Required margin (negative if insufficient)
        """
        # Simplified margin calculation
        notional = volume * 100000 * price  # Assuming forex
        required = notional / self._leverage
        
        if required > self._margin_free:
            return -required  # Negative indicates insufficient
        return required
    
    def FreeMarginCheck(
        self,
        symbol: str,
        order_type: ENUM_ORDER_TYPE,
        volume: float,
        price: float,
    ) -> float:
        """
        Check free margin after order.
        
        Returns:
            Free margin after order
        """
        margin_required = abs(self.MarginCheck(symbol, order_type, volume, price))
        return self._margin_free - margin_required
    
    def MaxLotCheck(
        self,
        symbol: str,
        order_type: ENUM_ORDER_TYPE,
        price: float,
    ) -> float:
        """
        Calculate maximum lot size for order.
        
        Returns:
            Maximum lot size
        """
        if price <= 0:
            return 0.0
        
        # Simplified: margin_per_lot = notional / leverage
        margin_per_lot = (100000 * price) / self._leverage
        
        if margin_per_lot <= 0:
            return 0.0
        
        max_lots = self._margin_free / margin_per_lot
        return max(0, min(max_lots, 100.0))  # Cap at 100 lots
    
    def Update(
        self,
        balance: float = None,
        equity: float = None,
        margin: float = None,
        profit: float = None,
        leverage: int = None,
    ) -> None:
        """Update account properties."""
        if balance is not None:
            self._balance = balance
            if equity is None:
                self._equity = balance  # Sync equity with balance if not specified
        if equity is not None:
            self._equity = equity
        if margin is not None:
            self._margin = margin
            self._margin_free = self._equity - margin
            if margin > 0:
                self._margin_level = (self._equity / margin) * 100
        if profit is not None:
            self._profit = profit
        if leverage is not None:
            self._leverage = leverage
    
    def SetLeverage(self, leverage: int) -> None:
        """Set account leverage."""
        self._leverage = leverage
    
    def SetBalance(self, balance: float, sync_equity: bool = True) -> None:
        """Set account balance (and optionally sync equity)."""
        self._balance = balance
        if sync_equity:
            self._equity = balance
            self._margin_free = balance - self._margin


# ═══════════════════════════════════════════════════════════════════════════
# CSymbolInfo - Symbol Properties
# ═══════════════════════════════════════════════════════════════════════════

class CSymbolInfo:
    """
    Class for easy access to the symbol properties.
    
    CSymbolInfo class provides access to the symbol properties.
    Mirrors MQL5 Standard Library: #include <Trade\\SymbolInfo.mqh>
    
    Reference: https://www.mql5.com/en/docs/standardlibrary/tradeclasses/csymbolinfo
    """
    
    def __init__(self, symbol: str = ""):
        """Initialize symbol info."""
        self._name: str = symbol
        self._select: bool = False
        
        # ═══════════════════════════════════════════════════════════════
        # Volumes
        # ═══════════════════════════════════════════════════════════════
        self._volume: int = 0              # Volume of last deal
        self._volumehigh: int = 0          # Maximal volume for a day
        self._volumelow: int = 0           # Minimal volume for a day
        self._volume_real: float = 0.0     # Real volume
        
        # ═══════════════════════════════════════════════════════════════
        # Miscellaneous
        # ═══════════════════════════════════════════════════════════════
        self._time: datetime = datetime.now()  # Time of last quote
        self._spread: int = 0              # Spread in points
        self._spread_float: bool = True    # Floating spread flag
        self._ticks_bookdepth: int = 10    # Depth of ticks saving
        
        # ═══════════════════════════════════════════════════════════════
        # Levels
        # ═══════════════════════════════════════════════════════════════
        self._trade_stops_level: int = 0   # Minimal indent for stops (points)
        self._trade_freeze_level: int = 0  # Freeze distance (points)
        
        # ═══════════════════════════════════════════════════════════════
        # Bid prices
        # ═══════════════════════════════════════════════════════════════
        self._bid: float = 0.0             # Current Bid price
        self._bidhigh: float = 0.0         # Maximal Bid for a day
        self._bidlow: float = 0.0          # Minimal Bid for a day
        
        # ═══════════════════════════════════════════════════════════════
        # Ask prices
        # ═══════════════════════════════════════════════════════════════
        self._ask: float = 0.0             # Current Ask price
        self._askhigh: float = 0.0         # Maximal Ask for a day
        self._asklow: float = 0.0          # Minimal Ask for a day
        
        # ═══════════════════════════════════════════════════════════════
        # Last prices
        # ═══════════════════════════════════════════════════════════════
        self._last: float = 0.0            # Current Last price
        self._lasthigh: float = 0.0        # Maximal Last for a day
        self._lastlow: float = 0.0         # Minimal Last for a day
        
        # ═══════════════════════════════════════════════════════════════
        # Trade modes
        # ═══════════════════════════════════════════════════════════════
        self._trade_calc_mode: ENUM_SYMBOL_CALC_MODE = ENUM_SYMBOL_CALC_MODE.SYMBOL_CALC_MODE_FOREX
        self._trade_mode: ENUM_SYMBOL_TRADE_MODE = ENUM_SYMBOL_TRADE_MODE.SYMBOL_TRADE_MODE_FULL
        self._trade_exe_mode: ENUM_SYMBOL_TRADE_EXECUTION = ENUM_SYMBOL_TRADE_EXECUTION.SYMBOL_TRADE_EXECUTION_INSTANT
        
        # ═══════════════════════════════════════════════════════════════
        # Swaps
        # ═══════════════════════════════════════════════════════════════
        self._swap_mode: ENUM_SYMBOL_SWAP_MODE = ENUM_SYMBOL_SWAP_MODE.SYMBOL_SWAP_MODE_POINTS
        self._swap_long: float = 0.0       # Swap for long positions
        self._swap_short: float = 0.0      # Swap for short positions
        self._swap_rollover3days: ENUM_DAY_OF_WEEK = ENUM_DAY_OF_WEEK.WEDNESDAY
        
        # ═══════════════════════════════════════════════════════════════
        # Margins and flags
        # ═══════════════════════════════════════════════════════════════
        self._margin_initial: float = 0.0       # Initial margin
        self._margin_maintenance: float = 0.0   # Maintenance margin
        self._margin_long: float = 1.0          # Long margin rate
        self._margin_short: float = 1.0         # Short margin rate
        self._margin_limit: float = 0.0         # Limit order margin rate
        self._margin_stop: float = 0.0          # Stop order margin rate
        self._margin_stop_limit: float = 0.0    # StopLimit order margin rate
        self._margin_hedged: float = 0.5        # Hedged margin rate
        self._trade_time_flags: int = 0         # Expiration mode flags
        self._trade_fill_flags: int = 0         # Filling mode flags
        
        # ═══════════════════════════════════════════════════════════════
        # Quantization
        # ═══════════════════════════════════════════════════════════════
        self._digits: int = 5              # Digits after period
        self._point: float = 0.00001       # Value of one point
        self._tick_value: float = 1.0      # Tick value (minimal price change)
        self._tick_value_profit: float = 1.0   # Tick price for profit
        self._tick_value_loss: float = 1.0     # Tick price for loss
        self._tick_size: float = 0.00001   # Minimal price change
        
        # ═══════════════════════════════════════════════════════════════
        # Contract sizes
        # ═══════════════════════════════════════════════════════════════
        self._contract_size: float = 100000.0  # Trade contract size
        self._volume_min: float = 0.01     # Minimal volume
        self._volume_max: float = 100.0    # Maximal volume
        self._volume_step: float = 0.01    # Volume step
        self._volume_limit: float = 0.0    # Volume limit
        
        # ═══════════════════════════════════════════════════════════════
        # Text properties
        # ═══════════════════════════════════════════════════════════════
        self._currency_base: str = ""      # Base currency
        self._currency_profit: str = ""    # Profit currency
        self._currency_margin: str = ""    # Margin currency
        self._bank: str = ""               # Current quote source
        self._description: str = ""        # Symbol description
        self._path: str = ""               # Path in symbols tree
        
        # ═══════════════════════════════════════════════════════════════
        # Session properties
        # ═══════════════════════════════════════════════════════════════
        self._session_deals: int = 0           # Deals in current session
        self._session_buy_orders: int = 0      # Buy orders at moment
        self._session_sell_orders: int = 0     # Sell orders at moment
        self._session_turnover: float = 0.0    # Turnover
        self._session_interest: float = 0.0    # Open interest
        self._session_buy_orders_volume: float = 0.0   # Buy orders volume
        self._session_sell_orders_volume: float = 0.0  # Sell orders volume
        self._session_open: float = 0.0        # Session open price
        self._session_close: float = 0.0       # Session close price
        self._session_aw: float = 0.0          # Average weighted price
        self._session_price_settlement: float = 0.0    # Settlement price
        self._session_price_limit_min: float = 0.0     # Min price limit
        self._session_price_limit_max: float = 0.0     # Max price limit
        
        # ═══════════════════════════════════════════════════════════════
        # Order modes
        # ═══════════════════════════════════════════════════════════════
        self._order_mode: int = 127
        self._filling_mode: int = 3
        self._expiration_mode: int = 15
    
    # ═══════════════════════════════════════════════════════════════════
    # Controlling
    # ═══════════════════════════════════════════════════════════════════
    
    def Refresh(self) -> bool:
        """Refresh symbol data."""
        # In real implementation, would fetch from MT5
        return True
    
    def RefreshRates(self) -> bool:
        """Refresh symbol quotes."""
        return True
    
    # ═══════════════════════════════════════════════════════════════════
    # Properties
    # ═══════════════════════════════════════════════════════════════════
    
    def Name(self, name: str = None) -> str:
        """Get/set symbol name."""
        if name is not None:
            self._name = name
        return self._name
    
    def Select(self, select: bool = None) -> bool:
        """Get/set the 'Market Watch' symbol flag."""
        if select is not None:
            self._select = select
        return self._select
    
    def IsSynchronized(self) -> bool:
        """Check symbol synchronization with server."""
        return True
    
    # ═══════════════════════════════════════════════════════════════════
    # Volumes
    # ═══════════════════════════════════════════════════════════════════
    
    def Volume(self) -> int:
        """Get volume of last deal."""
        return self._volume
    
    def VolumeHigh(self) -> int:
        """Get maximal volume for a day."""
        return self._volumehigh
    
    def VolumeLow(self) -> int:
        """Get minimal volume for a day."""
        return self._volumelow
    
    # ═══════════════════════════════════════════════════════════════════
    # Miscellaneous
    # ═══════════════════════════════════════════════════════════════════
    
    def Time(self) -> datetime:
        """Get time of last quote."""
        return self._time
    
    def Spread(self) -> int:
        """Get amount of spread (in points)."""
        return self._spread
    
    def SpreadFloat(self) -> bool:
        """Get flag of floating spread."""
        return self._spread_float
    
    def TicksBookDepth(self) -> int:
        """Get depth of ticks saving."""
        return self._ticks_bookdepth
    
    # ═══════════════════════════════════════════════════════════════════
    # Levels
    # ═══════════════════════════════════════════════════════════════════
    
    def StopsLevel(self) -> int:
        """Get minimal indent for orders (in points)."""
        return self._trade_stops_level
    
    def FreezeLevel(self) -> int:
        """Get distance of freezing trade operations (in points)."""
        return self._trade_freeze_level
    
    # ═══════════════════════════════════════════════════════════════════
    # Bid prices
    # ═══════════════════════════════════════════════════════════════════
    
    def Bid(self) -> float:
        """Get current Bid price."""
        return self._bid
    
    def BidHigh(self) -> float:
        """Get maximal Bid price for a day."""
        return self._bidhigh
    
    def BidLow(self) -> float:
        """Get minimal Bid price for a day."""
        return self._bidlow
    
    # ═══════════════════════════════════════════════════════════════════
    # Ask prices
    # ═══════════════════════════════════════════════════════════════════
    
    def Ask(self) -> float:
        """Get current Ask price."""
        return self._ask
    
    def AskHigh(self) -> float:
        """Get maximal Ask price for a day."""
        return self._askhigh
    
    def AskLow(self) -> float:
        """Get minimal Ask price for a day."""
        return self._asklow
    
    # ═══════════════════════════════════════════════════════════════════
    # Last prices
    # ═══════════════════════════════════════════════════════════════════
    
    def Last(self) -> float:
        """Get current Last price."""
        return self._last
    
    def LastHigh(self) -> float:
        """Get maximal Last price for a day."""
        return self._lasthigh
    
    def LastLow(self) -> float:
        """Get minimal Last price for a day."""
        return self._lastlow
    
    # ═══════════════════════════════════════════════════════════════════
    # Trade modes
    # ═══════════════════════════════════════════════════════════════════
    
    def TradeCalcMode(self) -> ENUM_SYMBOL_CALC_MODE:
        """Get mode of contract cost calculation."""
        return self._trade_calc_mode
    
    def TradeCalcModeDescription(self) -> str:
        """Get mode of contract cost calculation as string."""
        modes = {
            ENUM_SYMBOL_CALC_MODE.SYMBOL_CALC_MODE_FOREX: "Forex",
            ENUM_SYMBOL_CALC_MODE.SYMBOL_CALC_MODE_FOREX_NO_LEVERAGE: "Forex No Leverage",
            ENUM_SYMBOL_CALC_MODE.SYMBOL_CALC_MODE_FUTURES: "Futures",
            ENUM_SYMBOL_CALC_MODE.SYMBOL_CALC_MODE_CFD: "CFD",
            ENUM_SYMBOL_CALC_MODE.SYMBOL_CALC_MODE_CFDINDEX: "CFD Index",
            ENUM_SYMBOL_CALC_MODE.SYMBOL_CALC_MODE_CFDLEVERAGE: "CFD Leverage",
            ENUM_SYMBOL_CALC_MODE.SYMBOL_CALC_MODE_EXCH_STOCKS: "Exchange Stocks",
            ENUM_SYMBOL_CALC_MODE.SYMBOL_CALC_MODE_EXCH_FUTURES: "Exchange Futures",
            ENUM_SYMBOL_CALC_MODE.SYMBOL_CALC_MODE_EXCH_OPTIONS: "Exchange Options",
        }
        return modes.get(self._trade_calc_mode, "Unknown")
    
    def TradeMode(self) -> ENUM_SYMBOL_TRADE_MODE:
        """Get type of order execution."""
        return self._trade_mode
    
    def TradeModeDescription(self) -> str:
        """Get type of order execution as string."""
        modes = {
            ENUM_SYMBOL_TRADE_MODE.SYMBOL_TRADE_MODE_DISABLED: "Trade Disabled",
            ENUM_SYMBOL_TRADE_MODE.SYMBOL_TRADE_MODE_LONGONLY: "Long Only",
            ENUM_SYMBOL_TRADE_MODE.SYMBOL_TRADE_MODE_SHORTONLY: "Short Only",
            ENUM_SYMBOL_TRADE_MODE.SYMBOL_TRADE_MODE_CLOSEONLY: "Close Only",
            ENUM_SYMBOL_TRADE_MODE.SYMBOL_TRADE_MODE_FULL: "Full Access",
        }
        return modes.get(self._trade_mode, "Unknown")
    
    def TradeExecution(self) -> ENUM_SYMBOL_TRADE_EXECUTION:
        """Get trade execution mode."""
        return self._trade_exe_mode
    
    def TradeExecutionDescription(self) -> str:
        """Get execution mode as string."""
        modes = {
            ENUM_SYMBOL_TRADE_EXECUTION.SYMBOL_TRADE_EXECUTION_REQUEST: "Request Execution",
            ENUM_SYMBOL_TRADE_EXECUTION.SYMBOL_TRADE_EXECUTION_INSTANT: "Instant Execution",
            ENUM_SYMBOL_TRADE_EXECUTION.SYMBOL_TRADE_EXECUTION_MARKET: "Market Execution",
            ENUM_SYMBOL_TRADE_EXECUTION.SYMBOL_TRADE_EXECUTION_EXCHANGE: "Exchange Execution",
        }
        return modes.get(self._trade_exe_mode, "Unknown")
    
    # ═══════════════════════════════════════════════════════════════════
    # Swaps
    # ═══════════════════════════════════════════════════════════════════
    
    def SwapMode(self) -> ENUM_SYMBOL_SWAP_MODE:
        """Get swap calculation mode."""
        return self._swap_mode
    
    def SwapModeDescription(self) -> str:
        """Get swap calculation mode as string."""
        modes = {
            ENUM_SYMBOL_SWAP_MODE.SYMBOL_SWAP_MODE_DISABLED: "Disabled",
            ENUM_SYMBOL_SWAP_MODE.SYMBOL_SWAP_MODE_POINTS: "Points",
            ENUM_SYMBOL_SWAP_MODE.SYMBOL_SWAP_MODE_CURRENCY_SYMBOL: "Symbol Currency",
            ENUM_SYMBOL_SWAP_MODE.SYMBOL_SWAP_MODE_CURRENCY_MARGIN: "Margin Currency",
            ENUM_SYMBOL_SWAP_MODE.SYMBOL_SWAP_MODE_CURRENCY_DEPOSIT: "Deposit Currency",
            ENUM_SYMBOL_SWAP_MODE.SYMBOL_SWAP_MODE_INTEREST_CURRENT: "Interest Current",
            ENUM_SYMBOL_SWAP_MODE.SYMBOL_SWAP_MODE_INTEREST_OPEN: "Interest Open",
            ENUM_SYMBOL_SWAP_MODE.SYMBOL_SWAP_MODE_REOPEN_CURRENT: "Reopen Current",
            ENUM_SYMBOL_SWAP_MODE.SYMBOL_SWAP_MODE_REOPEN_BID: "Reopen Bid",
        }
        return modes.get(self._swap_mode, "Unknown")
    
    def SwapRollover3days(self) -> ENUM_DAY_OF_WEEK:
        """Get day of triple swap charge."""
        return self._swap_rollover3days
    
    def SwapRollover3daysDescription(self) -> str:
        """Get day of triple swap charge as string."""
        days = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
        return days[self._swap_rollover3days]
    
    # ═══════════════════════════════════════════════════════════════════
    # Margins and flags
    # ═══════════════════════════════════════════════════════════════════
    
    def MarginInitial(self) -> float:
        """Get value of initial margin."""
        return self._margin_initial
    
    def MarginMaintenance(self) -> float:
        """Get value of maintenance margin."""
        return self._margin_maintenance
    
    def MarginLong(self) -> float:
        """Get rate of margin charging for long positions."""
        return self._margin_long
    
    def MarginShort(self) -> float:
        """Get rate of margin charging for short positions."""
        return self._margin_short
    
    def MarginLimit(self) -> float:
        """Get rate of margin charging for Limit orders."""
        return self._margin_limit
    
    def MarginStop(self) -> float:
        """Get rate of margin charging for Stop orders."""
        return self._margin_stop
    
    def MarginStopLimit(self) -> float:
        """Get rate of margin charging for StopLimit orders."""
        return self._margin_stop_limit
    
    def MarginHedged(self) -> float:
        """Get hedged margin rate."""
        return self._margin_hedged
    
    def TradeTimeFlags(self) -> int:
        """Get flags of allowed expiration modes."""
        return self._trade_time_flags
    
    def TradeFillFlags(self) -> int:
        """Get flags of allowed filling modes."""
        return self._trade_fill_flags
    
    # ═══════════════════════════════════════════════════════════════════
    # Quantization
    # ═══════════════════════════════════════════════════════════════════
    
    def Digits(self) -> int:
        """Get number of digits after period."""
        return self._digits
    
    def Point(self) -> float:
        """Get value of one point."""
        return self._point
    
    def TickValue(self) -> float:
        """Get tick value (minimal change of price)."""
        return self._tick_value
    
    def TickValueProfit(self) -> float:
        """Get calculated tick price for profitable position."""
        return self._tick_value_profit
    
    def TickValueLoss(self) -> float:
        """Get calculated tick price for losing position."""
        return self._tick_value_loss
    
    def TickSize(self) -> float:
        """Get minimal change of price."""
        return self._tick_size
    
    # ═══════════════════════════════════════════════════════════════════
    # Contract sizes
    # ═══════════════════════════════════════════════════════════════════
    
    def ContractSize(self) -> float:
        """Get amount of trade contract."""
        return self._contract_size
    
    def LotsMin(self) -> float:
        """Get minimal volume to close a deal."""
        return self._volume_min
    
    def LotsMax(self) -> float:
        """Get maximal volume to close a deal."""
        return self._volume_max
    
    def LotsStep(self) -> float:
        """Get minimal step of volume change."""
        return self._volume_step
    
    def LotsLimit(self) -> float:
        """Get maximal allowed volume of position and orders."""
        return self._volume_limit
    
    # ═══════════════════════════════════════════════════════════════════
    # Swap sizes
    # ═══════════════════════════════════════════════════════════════════
    
    def SwapLong(self) -> float:
        """Get value of long position swap."""
        return self._swap_long
    
    def SwapShort(self) -> float:
        """Get value of short position swap."""
        return self._swap_short
    
    # ═══════════════════════════════════════════════════════════════════
    # Text properties
    # ═══════════════════════════════════════════════════════════════════
    
    def CurrencyBase(self) -> str:
        """Get name of symbol base currency."""
        return self._currency_base
    
    def CurrencyProfit(self) -> str:
        """Get profit currency name."""
        return self._currency_profit
    
    def CurrencyMargin(self) -> str:
        """Get margin currency name."""
        return self._currency_margin
    
    def Bank(self) -> str:
        """Get name of current quote source."""
        return self._bank
    
    def Description(self) -> str:
        """Get string description of symbol."""
        return self._description
    
    def Path(self) -> str:
        """Get path in symbols tree."""
        return self._path
    
    # ═══════════════════════════════════════════════════════════════════
    # Symbol session properties
    # ═══════════════════════════════════════════════════════════════════
    
    def SessionDeals(self) -> int:
        """Get number of deals in current session."""
        return self._session_deals
    
    def SessionBuyOrders(self) -> int:
        """Get number of Buy orders at the moment."""
        return self._session_buy_orders
    
    def SessionSellOrders(self) -> int:
        """Get number of Sell orders at the moment."""
        return self._session_sell_orders
    
    def SessionTurnover(self) -> float:
        """Get summary turnover of current session."""
        return self._session_turnover
    
    def SessionInterest(self) -> float:
        """Get summary open interest of current session."""
        return self._session_interest
    
    def SessionBuyOrdersVolume(self) -> float:
        """Get current volume of Buy orders."""
        return self._session_buy_orders_volume
    
    def SessionSellOrdersVolume(self) -> float:
        """Get current volume of Sell orders."""
        return self._session_sell_orders_volume
    
    def SessionOpen(self) -> float:
        """Get open price of current session."""
        return self._session_open
    
    def SessionClose(self) -> float:
        """Get close price of current session."""
        return self._session_close
    
    def SessionAW(self) -> float:
        """Get average weighted price of current session."""
        return self._session_aw
    
    def SessionPriceSettlement(self) -> float:
        """Get settlement price of current session."""
        return self._session_price_settlement
    
    def SessionPriceLimitMin(self) -> float:
        """Get minimal price of current session."""
        return self._session_price_limit_min
    
    def SessionPriceLimitMax(self) -> float:
        """Get maximal price of current session."""
        return self._session_price_limit_max
    
    # ═══════════════════════════════════════════════════════════════════
    # Access to MQL5 API functions
    # ═══════════════════════════════════════════════════════════════════
    
    def InfoInteger(self, prop_id: int) -> int:
        """Get value of specified integer type property."""
        # Map property IDs to internal values
        return 0
    
    def InfoDouble(self, prop_id: int) -> float:
        """Get value of specified double type property."""
        return 0.0
    
    def InfoString(self, prop_id: int) -> str:
        """Get value of specified string type property."""
        return ""
    
    # ═══════════════════════════════════════════════════════════════════
    # Service functions
    # ═══════════════════════════════════════════════════════════════════
    
    def NormalizePrice(self, price: float) -> float:
        """Normalize price to symbol tick size."""
        if self._tick_size <= 0:
            return round(price, self._digits)
        return round(price / self._tick_size) * self._tick_size
    
    def NormalizeLots(self, lots: float) -> float:
        """Normalize lots to symbol step."""
        if lots < self._volume_min:
            return 0.0
        lots = min(lots, self._volume_max)
        return round(lots / self._volume_step) * self._volume_step
    
    # ═══════════════════════════════════════════════════════════════════
    # Utility Methods (extensions)
    # ═══════════════════════════════════════════════════════════════════
    
    def CheckMargin(
        self,
        order_type: ENUM_ORDER_TYPE,
        volume: float,
        price: float,
    ) -> float:
        """Calculate margin required for order."""
        if self._margin_initial > 0:
            return self._margin_initial * volume
        
        # Calculate from contract size
        return volume * self._contract_size * price * self._margin_long
    
    def CheckVolume(self, volume: float) -> bool:
        """Check if volume is valid."""
        if volume < self._volume_min:
            return False
        if volume > self._volume_max:
            return False
        
        # Check step
        steps = round(volume / self._volume_step)
        normalized = steps * self._volume_step
        return abs(volume - normalized) < 1e-10
    
    def Update(
        self,
        bid: float = None,
        ask: float = None,
        last: float = None,
        volume: int = None,
        spread: int = None,
        time: datetime = None,
    ) -> None:
        """Update price data."""
        if bid is not None:
            self._bid = bid
        if ask is not None:
            self._ask = ask
        if last is not None:
            self._last = last
        if volume is not None:
            self._volume = volume
        if spread is not None:
            self._spread = spread
        elif bid is not None and ask is not None and self._point > 0:
            self._spread = int((ask - bid) / self._point)
        if time is not None:
            self._time = time
    
    def SetContractSpec(
        self,
        digits: int = None,
        point: float = None,
        tick_size: float = None,
        tick_value: float = None,
        contract_size: float = None,
        volume_min: float = None,
        volume_max: float = None,
        volume_step: float = None,
    ) -> None:
        """Set contract specifications."""
        if digits is not None:
            self._digits = digits
        if point is not None:
            self._point = point
        if tick_size is not None:
            self._tick_size = tick_size
        if tick_value is not None:
            self._tick_value = tick_value
            self._tick_value_profit = tick_value
            self._tick_value_loss = tick_value
        if contract_size is not None:
            self._contract_size = contract_size
        if volume_min is not None:
            self._volume_min = volume_min
        if volume_max is not None:
            self._volume_max = volume_max
        if volume_step is not None:
            self._volume_step = volume_step
    
    def SetSwapSpec(
        self,
        swap_mode: ENUM_SYMBOL_SWAP_MODE = None,
        swap_long: float = None,
        swap_short: float = None,
        swap_rollover3days: ENUM_DAY_OF_WEEK = None,
    ) -> None:
        """Set swap specifications."""
        if swap_mode is not None:
            self._swap_mode = swap_mode
        if swap_long is not None:
            self._swap_long = swap_long
        if swap_short is not None:
            self._swap_short = swap_short
        if swap_rollover3days is not None:
            self._swap_rollover3days = swap_rollover3days
    
    def SetCurrency(
        self,
        base: str = None,
        profit: str = None,
        margin: str = None,
    ) -> None:
        """Set currency information."""
        if base is not None:
            self._currency_base = base
        if profit is not None:
            self._currency_profit = profit
        if margin is not None:
            self._currency_margin = margin
    
    def SetDescription(self, description: str, path: str = None) -> None:
        """Set symbol description and path."""
        self._description = description
        if path is not None:
            self._path = path


# ═══════════════════════════════════════════════════════════════════════════
# COrderInfo - Pending Order Properties
# ═══════════════════════════════════════════════════════════════════════════

class COrderInfo:
    """
    Class for working with pending order properties.
    
    Provides access to all pending order information.
    """
    
    def __init__(self):
        """Initialize order info."""
        self._ticket: int = 0
        self._time_setup: datetime = datetime.now()
        self._time_setup_msc: int = 0
        self._time_done: datetime = datetime.now()
        self._time_done_msc: int = 0
        self._time_expiration: datetime = datetime.now()
        self._type: ENUM_ORDER_TYPE = ENUM_ORDER_TYPE.ORDER_TYPE_BUY
        self._type_filling: ENUM_ORDER_TYPE_FILLING = ENUM_ORDER_TYPE_FILLING.ORDER_FILLING_FOK
        self._type_time: ENUM_ORDER_TYPE_TIME = ENUM_ORDER_TYPE_TIME.ORDER_TIME_GTC
        self._state: ENUM_ORDER_STATE = ENUM_ORDER_STATE.ORDER_STATE_STARTED
        self._magic: int = 0
        self._position_id: int = 0
        self._position_by_id: int = 0
        
        self._volume_initial: float = 0.0
        self._volume_current: float = 0.0
        self._price_open: float = 0.0
        self._sl: float = 0.0
        self._tp: float = 0.0
        self._price_current: float = 0.0
        self._price_stoplimit: float = 0.0
        
        self._symbol: str = ""
        self._comment: str = ""
        self._external_id: str = ""
    
    def Select(self, ticket: int) -> bool:
        """Select order by ticket."""
        self._ticket = ticket
        return True
    
    def SelectByIndex(self, index: int) -> bool:
        """Select order by index."""
        return True
    
    # Integer properties
    def Ticket(self) -> int:
        return self._ticket
    
    def TimeSetup(self) -> datetime:
        return self._time_setup
    
    def TimeSetupMsc(self) -> int:
        return self._time_setup_msc
    
    def TimeDone(self) -> datetime:
        return self._time_done
    
    def TimeDoneMsc(self) -> int:
        return self._time_done_msc
    
    def TimeExpiration(self) -> datetime:
        return self._time_expiration
    
    def OrderType(self) -> ENUM_ORDER_TYPE:
        return self._type
    
    def TypeDescription(self) -> str:
        types = {
            ENUM_ORDER_TYPE.ORDER_TYPE_BUY: "Buy",
            ENUM_ORDER_TYPE.ORDER_TYPE_SELL: "Sell",
            ENUM_ORDER_TYPE.ORDER_TYPE_BUY_LIMIT: "Buy Limit",
            ENUM_ORDER_TYPE.ORDER_TYPE_SELL_LIMIT: "Sell Limit",
            ENUM_ORDER_TYPE.ORDER_TYPE_BUY_STOP: "Buy Stop",
            ENUM_ORDER_TYPE.ORDER_TYPE_SELL_STOP: "Sell Stop",
            ENUM_ORDER_TYPE.ORDER_TYPE_BUY_STOP_LIMIT: "Buy Stop Limit",
            ENUM_ORDER_TYPE.ORDER_TYPE_SELL_STOP_LIMIT: "Sell Stop Limit",
        }
        return types.get(self._type, "Unknown")
    
    def TypeFilling(self) -> ENUM_ORDER_TYPE_FILLING:
        return self._type_filling
    
    def TypeFillingDescription(self) -> str:
        modes = {
            ENUM_ORDER_TYPE_FILLING.ORDER_FILLING_FOK: "Fill or Kill",
            ENUM_ORDER_TYPE_FILLING.ORDER_FILLING_IOC: "Immediate or Cancel",
            ENUM_ORDER_TYPE_FILLING.ORDER_FILLING_RETURN: "Return",
        }
        return modes.get(self._type_filling, "Unknown")
    
    def TypeTime(self) -> ENUM_ORDER_TYPE_TIME:
        return self._type_time
    
    def TypeTimeDescription(self) -> str:
        modes = {
            ENUM_ORDER_TYPE_TIME.ORDER_TIME_GTC: "Good Till Cancelled",
            ENUM_ORDER_TYPE_TIME.ORDER_TIME_DAY: "Day Order",
            ENUM_ORDER_TYPE_TIME.ORDER_TIME_SPECIFIED: "Specified Time",
            ENUM_ORDER_TYPE_TIME.ORDER_TIME_SPECIFIED_DAY: "Specified Day",
        }
        return modes.get(self._type_time, "Unknown")
    
    def State(self) -> ENUM_ORDER_STATE:
        return self._state
    
    def StateDescription(self) -> str:
        states = {
            ENUM_ORDER_STATE.ORDER_STATE_STARTED: "Started",
            ENUM_ORDER_STATE.ORDER_STATE_PLACED: "Placed",
            ENUM_ORDER_STATE.ORDER_STATE_CANCELED: "Canceled",
            ENUM_ORDER_STATE.ORDER_STATE_PARTIAL: "Partially Filled",
            ENUM_ORDER_STATE.ORDER_STATE_FILLED: "Filled",
            ENUM_ORDER_STATE.ORDER_STATE_REJECTED: "Rejected",
            ENUM_ORDER_STATE.ORDER_STATE_EXPIRED: "Expired",
        }
        return states.get(self._state, "Unknown")
    
    def Magic(self) -> int:
        return self._magic
    
    def PositionId(self) -> int:
        return self._position_id
    
    def PositionById(self) -> int:
        return self._position_by_id
    
    # Double properties
    def VolumeInitial(self) -> float:
        return self._volume_initial
    
    def VolumeCurrent(self) -> float:
        return self._volume_current
    
    def PriceOpen(self) -> float:
        return self._price_open
    
    def StopLoss(self) -> float:
        return self._sl
    
    def TakeProfit(self) -> float:
        return self._tp
    
    def PriceCurrent(self) -> float:
        return self._price_current
    
    def PriceStopLimit(self) -> float:
        return self._price_stoplimit
    
    # String properties
    def Symbol(self) -> str:
        return self._symbol
    
    def Comment(self) -> str:
        return self._comment
    
    def ExternalId(self) -> str:
        return self._external_id


# ═══════════════════════════════════════════════════════════════════════════
# CHistoryOrderInfo - History Order Properties
# ═══════════════════════════════════════════════════════════════════════════

class CHistoryOrderInfo(COrderInfo):
    """
    Class for working with history order properties.
    
    Extends COrderInfo with history-specific properties.
    """
    
    def __init__(self):
        """Initialize history order info."""
        super().__init__()
        self._reason: int = 0
    
    def Reason(self) -> int:
        """Get order close reason."""
        return self._reason
    
    def ReasonDescription(self) -> str:
        """Get reason description."""
        return "Expert"  # Simplified


# ═══════════════════════════════════════════════════════════════════════════
# CPositionInfo - Open Position Properties
# ═══════════════════════════════════════════════════════════════════════════

class CPositionInfo:
    """
    Class for working with open position properties.
    
    Provides comprehensive position information including:
    - Entry details
    - Current P&L
    - Stop levels
    """
    
    def __init__(self):
        """Initialize position info."""
        self._ticket: int = 0
        self._time: datetime = datetime.now()
        self._time_msc: int = 0
        self._time_update: datetime = datetime.now()
        self._time_update_msc: int = 0
        self._type: ENUM_POSITION_TYPE = ENUM_POSITION_TYPE.POSITION_TYPE_BUY
        self._magic: int = 0
        self._identifier: int = 0
        self._reason: ENUM_POSITION_REASON = ENUM_POSITION_REASON.POSITION_REASON_EXPERT
        
        self._volume: float = 0.0
        self._price_open: float = 0.0
        self._sl: float = 0.0
        self._tp: float = 0.0
        self._price_current: float = 0.0
        self._swap: float = 0.0
        self._profit: float = 0.0
        self._commission: float = 0.0
        
        self._symbol: str = ""
        self._comment: str = ""
        self._external_id: str = ""
    
    def Select(self, symbol: str) -> bool:
        """Select position by symbol."""
        self._symbol = symbol
        return True
    
    def SelectByIndex(self, index: int) -> bool:
        """Select position by index."""
        return True
    
    def SelectByTicket(self, ticket: int) -> bool:
        """Select position by ticket."""
        self._ticket = ticket
        return True
    
    def SelectByMagic(self, symbol: str, magic: int) -> bool:
        """Select position by symbol and magic number."""
        self._symbol = symbol
        self._magic = magic
        return True
    
    # Integer properties
    def Ticket(self) -> int:
        return self._ticket
    
    def Time(self) -> datetime:
        return self._time
    
    def TimeMsc(self) -> int:
        return self._time_msc
    
    def TimeUpdate(self) -> datetime:
        return self._time_update
    
    def TimeUpdateMsc(self) -> int:
        return self._time_update_msc
    
    def PositionType(self) -> ENUM_POSITION_TYPE:
        return self._type
    
    def TypeDescription(self) -> str:
        types = {
            ENUM_POSITION_TYPE.POSITION_TYPE_BUY: "Buy",
            ENUM_POSITION_TYPE.POSITION_TYPE_SELL: "Sell",
        }
        return types.get(self._type, "Unknown")
    
    def Magic(self) -> int:
        return self._magic
    
    def Identifier(self) -> int:
        return self._identifier
    
    def Reason(self) -> ENUM_POSITION_REASON:
        return self._reason
    
    def ReasonDescription(self) -> str:
        reasons = {
            ENUM_POSITION_REASON.POSITION_REASON_CLIENT: "Client",
            ENUM_POSITION_REASON.POSITION_REASON_MOBILE: "Mobile",
            ENUM_POSITION_REASON.POSITION_REASON_WEB: "Web",
            ENUM_POSITION_REASON.POSITION_REASON_EXPERT: "Expert",
        }
        return reasons.get(self._reason, "Unknown")
    
    # Double properties
    def Volume(self) -> float:
        return self._volume
    
    def PriceOpen(self) -> float:
        return self._price_open
    
    def StopLoss(self) -> float:
        return self._sl
    
    def TakeProfit(self) -> float:
        return self._tp
    
    def PriceCurrent(self) -> float:
        return self._price_current
    
    def Swap(self) -> float:
        return self._swap
    
    def Profit(self) -> float:
        return self._profit
    
    def Commission(self) -> float:
        return self._commission
    
    # String properties
    def Symbol(self) -> str:
        return self._symbol
    
    def Comment(self) -> str:
        return self._comment
    
    def ExternalId(self) -> str:
        return self._external_id
    
    # Utility methods
    def StoreState(self) -> None:
        """Store current position state."""
        pass
    
    def CheckState(self) -> bool:
        """Check if position state changed."""
        return False


# ═══════════════════════════════════════════════════════════════════════════
# CDealInfo - History Deal Properties
# ═══════════════════════════════════════════════════════════════════════════

class CDealInfo:
    """
    Class for working with history deal properties.
    
    Deals are the result of trade operations (order execution).
    """
    
    def __init__(self):
        """Initialize deal info."""
        self._ticket: int = 0
        self._order: int = 0
        self._time: datetime = datetime.now()
        self._time_msc: int = 0
        self._type: ENUM_DEAL_TYPE = ENUM_DEAL_TYPE.DEAL_TYPE_BUY
        self._entry: ENUM_DEAL_ENTRY = ENUM_DEAL_ENTRY.DEAL_ENTRY_IN
        self._magic: int = 0
        self._reason: ENUM_DEAL_REASON = ENUM_DEAL_REASON.DEAL_REASON_EXPERT
        self._position_id: int = 0
        
        self._volume: float = 0.0
        self._price: float = 0.0
        self._commission: float = 0.0
        self._swap: float = 0.0
        self._profit: float = 0.0
        self._fee: float = 0.0
        self._sl: float = 0.0
        self._tp: float = 0.0
        
        self._symbol: str = ""
        self._comment: str = ""
        self._external_id: str = ""
    
    def Select(self, ticket: int) -> bool:
        """Select deal by ticket."""
        self._ticket = ticket
        return True
    
    def SelectByIndex(self, index: int) -> bool:
        """Select deal by index."""
        return True
    
    # Integer properties
    def Ticket(self) -> int:
        return self._ticket
    
    def Order(self) -> int:
        return self._order
    
    def Time(self) -> datetime:
        return self._time
    
    def TimeMsc(self) -> int:
        return self._time_msc
    
    def DealType(self) -> ENUM_DEAL_TYPE:
        return self._type
    
    def TypeDescription(self) -> str:
        types = {
            ENUM_DEAL_TYPE.DEAL_TYPE_BUY: "Buy",
            ENUM_DEAL_TYPE.DEAL_TYPE_SELL: "Sell",
            ENUM_DEAL_TYPE.DEAL_TYPE_BALANCE: "Balance",
            ENUM_DEAL_TYPE.DEAL_TYPE_CREDIT: "Credit",
            ENUM_DEAL_TYPE.DEAL_TYPE_COMMISSION: "Commission",
        }
        return types.get(self._type, "Unknown")
    
    def Entry(self) -> ENUM_DEAL_ENTRY:
        return self._entry
    
    def EntryDescription(self) -> str:
        entries = {
            ENUM_DEAL_ENTRY.DEAL_ENTRY_IN: "In",
            ENUM_DEAL_ENTRY.DEAL_ENTRY_OUT: "Out",
            ENUM_DEAL_ENTRY.DEAL_ENTRY_INOUT: "In/Out",
            ENUM_DEAL_ENTRY.DEAL_ENTRY_OUT_BY: "Out By",
        }
        return entries.get(self._entry, "Unknown")
    
    def Magic(self) -> int:
        return self._magic
    
    def Reason(self) -> ENUM_DEAL_REASON:
        return self._reason
    
    def ReasonDescription(self) -> str:
        reasons = {
            ENUM_DEAL_REASON.DEAL_REASON_CLIENT: "Client",
            ENUM_DEAL_REASON.DEAL_REASON_EXPERT: "Expert",
            ENUM_DEAL_REASON.DEAL_REASON_SL: "Stop Loss",
            ENUM_DEAL_REASON.DEAL_REASON_TP: "Take Profit",
            ENUM_DEAL_REASON.DEAL_REASON_SO: "Stop Out",
        }
        return reasons.get(self._reason, "Unknown")
    
    def PositionId(self) -> int:
        return self._position_id
    
    # Double properties
    def Volume(self) -> float:
        return self._volume
    
    def Price(self) -> float:
        return self._price
    
    def Commission(self) -> float:
        return self._commission
    
    def Swap(self) -> float:
        return self._swap
    
    def Profit(self) -> float:
        return self._profit
    
    def Fee(self) -> float:
        return self._fee
    
    def StopLoss(self) -> float:
        return self._sl
    
    def TakeProfit(self) -> float:
        return self._tp
    
    # String properties
    def Symbol(self) -> str:
        return self._symbol
    
    def Comment(self) -> str:
        return self._comment
    
    def ExternalId(self) -> str:
        return self._external_id


# ═══════════════════════════════════════════════════════════════════════════
# CTrade - Trade Operations
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class MqlTradeRequest:
    """Trade request structure."""
    action: ENUM_TRADE_REQUEST_ACTIONS = ENUM_TRADE_REQUEST_ACTIONS.TRADE_ACTION_DEAL
    magic: int = 0
    order: int = 0
    symbol: str = ""
    volume: float = 0.0
    price: float = 0.0
    stoplimit: float = 0.0
    sl: float = 0.0
    tp: float = 0.0
    deviation: int = 10
    type: ENUM_ORDER_TYPE = ENUM_ORDER_TYPE.ORDER_TYPE_BUY
    type_filling: ENUM_ORDER_TYPE_FILLING = ENUM_ORDER_TYPE_FILLING.ORDER_FILLING_FOK
    type_time: ENUM_ORDER_TYPE_TIME = ENUM_ORDER_TYPE_TIME.ORDER_TIME_GTC
    expiration: datetime = None
    comment: str = ""
    position: int = 0
    position_by: int = 0


@dataclass
class MqlTradeResult:
    """Trade result structure."""
    retcode: ENUM_TRADE_RETCODE = ENUM_TRADE_RETCODE.TRADE_RETCODE_DONE
    deal: int = 0
    order: int = 0
    volume: float = 0.0
    price: float = 0.0
    bid: float = 0.0
    ask: float = 0.0
    comment: str = ""
    request_id: int = 0
    retcode_external: int = 0


class CTrade:
    """
    Class for trade operations execution.
    
    Provides methods for:
    - Opening/closing positions
    - Placing/modifying/deleting pending orders
    - Modifying position stops
    """
    
    def __init__(self):
        """Initialize trade class."""
        self._request = MqlTradeRequest()
        self._result = MqlTradeResult()
        
        self._magic: int = 0
        self._deviation: int = 10
        self._type_filling: ENUM_ORDER_TYPE_FILLING = ENUM_ORDER_TYPE_FILLING.ORDER_FILLING_FOK
        self._type_filling_async: ENUM_ORDER_TYPE_FILLING = ENUM_ORDER_TYPE_FILLING.ORDER_FILLING_FOK
        self._log_level: int = 1
        self._async_mode: bool = False
        
        # Trade executor callback
        self._executor: Optional[Callable[[MqlTradeRequest], MqlTradeResult]] = None
    
    def SetExecutor(self, executor: Callable[[MqlTradeRequest], MqlTradeResult]) -> None:
        """Set trade executor callback."""
        self._executor = executor
    
    # ─────────────────────────────────────────────────────────────────
    # Settings
    # ─────────────────────────────────────────────────────────────────
    
    def SetExpertMagicNumber(self, magic: int) -> None:
        """Set expert magic number."""
        self._magic = magic
    
    def SetDeviationInPoints(self, deviation: int) -> None:
        """Set price deviation in points."""
        self._deviation = deviation
    
    def SetTypeFilling(self, type_filling: ENUM_ORDER_TYPE_FILLING) -> None:
        """Set order fill type."""
        self._type_filling = type_filling
    
    def SetTypeFillingBySymbol(self, symbol: str) -> None:
        """Set fill type based on symbol."""
        # Would query symbol info in real implementation
        pass
    
    def SetAsyncMode(self, async_mode: bool) -> None:
        """Set async mode."""
        self._async_mode = async_mode
    
    def SetLogLevel(self, level: int) -> None:
        """Set logging level."""
        self._log_level = level
    
    def SetMarginMode(self) -> bool:
        """Set margin mode."""
        return True
    
    def LogLevel(self) -> int:
        """Get log level."""
        return self._log_level
    
    def RequestMagic(self) -> int:
        """Get request magic number."""
        return self._request.magic
    
    def RequestSymbol(self) -> str:
        """Get request symbol."""
        return self._request.symbol
    
    def RequestVolume(self) -> float:
        """Get request volume."""
        return self._request.volume
    
    def RequestPrice(self) -> float:
        """Get request price."""
        return self._request.price
    
    def RequestStopLimit(self) -> float:
        """Get request stop limit price."""
        return self._request.stoplimit
    
    def RequestSL(self) -> float:
        """Get request stop loss."""
        return self._request.sl
    
    def RequestTP(self) -> float:
        """Get request take profit."""
        return self._request.tp
    
    def RequestDeviation(self) -> int:
        """Get request deviation."""
        return self._request.deviation
    
    def RequestType(self) -> ENUM_ORDER_TYPE:
        """Get request order type."""
        return self._request.type
    
    def RequestTypeFilling(self) -> ENUM_ORDER_TYPE_FILLING:
        """Get request fill type."""
        return self._request.type_filling
    
    def RequestTypeTime(self) -> ENUM_ORDER_TYPE_TIME:
        """Get request time type."""
        return self._request.type_time
    
    def RequestExpiration(self) -> datetime:
        """Get request expiration."""
        return self._request.expiration
    
    def RequestComment(self) -> str:
        """Get request comment."""
        return self._request.comment
    
    def RequestAction(self) -> ENUM_TRADE_REQUEST_ACTIONS:
        """Get request action."""
        return self._request.action
    
    def RequestActionDescription(self) -> str:
        """Get request action description."""
        actions = {
            ENUM_TRADE_REQUEST_ACTIONS.TRADE_ACTION_DEAL: "Market Order",
            ENUM_TRADE_REQUEST_ACTIONS.TRADE_ACTION_PENDING: "Pending Order",
            ENUM_TRADE_REQUEST_ACTIONS.TRADE_ACTION_SLTP: "Modify SL/TP",
            ENUM_TRADE_REQUEST_ACTIONS.TRADE_ACTION_MODIFY: "Modify Order",
            ENUM_TRADE_REQUEST_ACTIONS.TRADE_ACTION_REMOVE: "Remove Order",
            ENUM_TRADE_REQUEST_ACTIONS.TRADE_ACTION_CLOSE_BY: "Close By",
        }
        return actions.get(self._request.action, "Unknown")
    
    def RequestPosition(self) -> int:
        """Get request position ticket."""
        return self._request.position
    
    def RequestPositionBy(self) -> int:
        """Get request close by position ticket."""
        return self._request.position_by
    
    # ─────────────────────────────────────────────────────────────────
    # Result Access
    # ─────────────────────────────────────────────────────────────────
    
    def Result(self) -> MqlTradeResult:
        """Get trade result."""
        return self._result
    
    def ResultRetcode(self) -> ENUM_TRADE_RETCODE:
        """Get result return code."""
        return self._result.retcode
    
    def ResultRetcodeDescription(self) -> str:
        """Get return code description."""
        codes = {
            ENUM_TRADE_RETCODE.TRADE_RETCODE_DONE: "Done",
            ENUM_TRADE_RETCODE.TRADE_RETCODE_DONE_PARTIAL: "Partial Fill",
            ENUM_TRADE_RETCODE.TRADE_RETCODE_REQUOTE: "Requote",
            ENUM_TRADE_RETCODE.TRADE_RETCODE_REJECT: "Rejected",
            ENUM_TRADE_RETCODE.TRADE_RETCODE_CANCEL: "Cancelled",
            ENUM_TRADE_RETCODE.TRADE_RETCODE_ERROR: "Error",
            ENUM_TRADE_RETCODE.TRADE_RETCODE_TIMEOUT: "Timeout",
            ENUM_TRADE_RETCODE.TRADE_RETCODE_INVALID: "Invalid",
            ENUM_TRADE_RETCODE.TRADE_RETCODE_INVALID_VOLUME: "Invalid Volume",
            ENUM_TRADE_RETCODE.TRADE_RETCODE_INVALID_PRICE: "Invalid Price",
            ENUM_TRADE_RETCODE.TRADE_RETCODE_INVALID_STOPS: "Invalid Stops",
            ENUM_TRADE_RETCODE.TRADE_RETCODE_NO_MONEY: "No Money",
        }
        return codes.get(self._result.retcode, f"Code: {self._result.retcode}")
    
    def ResultDeal(self) -> int:
        """Get result deal ticket."""
        return self._result.deal
    
    def ResultOrder(self) -> int:
        """Get result order ticket."""
        return self._result.order
    
    def ResultVolume(self) -> float:
        """Get result volume."""
        return self._result.volume
    
    def ResultPrice(self) -> float:
        """Get result price."""
        return self._result.price
    
    def ResultBid(self) -> float:
        """Get result bid."""
        return self._result.bid
    
    def ResultAsk(self) -> float:
        """Get result ask."""
        return self._result.ask
    
    def ResultComment(self) -> str:
        """Get result comment."""
        return self._result.comment
    
    # ─────────────────────────────────────────────────────────────────
    # Trade Operations
    # ─────────────────────────────────────────────────────────────────
    
    def _send_request(self) -> bool:
        """Send trade request."""
        if self._executor:
            self._result = self._executor(self._request)
        else:
            # Simulate success
            self._result = MqlTradeResult(
                retcode=ENUM_TRADE_RETCODE.TRADE_RETCODE_DONE,
                deal=1,
                order=1,
                volume=self._request.volume,
                price=self._request.price,
            )
        
        success = self._result.retcode == ENUM_TRADE_RETCODE.TRADE_RETCODE_DONE
        
        if self._log_level > 0:
            logger.info(
                f"Trade {self.RequestActionDescription()}: "
                f"{self._request.symbol} {self._request.volume} lots @ {self._request.price} "
                f"-> {self.ResultRetcodeDescription()}"
            )
        
        return success
    
    def Buy(
        self,
        volume: float,
        symbol: str = None,
        price: float = 0.0,
        sl: float = 0.0,
        tp: float = 0.0,
        comment: str = "",
    ) -> bool:
        """
        Open buy position.
        
        Args:
            volume: Position volume
            symbol: Symbol name
            price: Open price (0 = market)
            sl: Stop loss
            tp: Take profit
            comment: Order comment
            
        Returns:
            True if successful
        """
        self._request = MqlTradeRequest(
            action=ENUM_TRADE_REQUEST_ACTIONS.TRADE_ACTION_DEAL,
            magic=self._magic,
            symbol=symbol or "",
            volume=volume,
            price=price,
            sl=sl,
            tp=tp,
            deviation=self._deviation,
            type=ENUM_ORDER_TYPE.ORDER_TYPE_BUY,
            type_filling=self._type_filling,
            comment=comment,
        )
        return self._send_request()
    
    def Sell(
        self,
        volume: float,
        symbol: str = None,
        price: float = 0.0,
        sl: float = 0.0,
        tp: float = 0.0,
        comment: str = "",
    ) -> bool:
        """Open sell position."""
        self._request = MqlTradeRequest(
            action=ENUM_TRADE_REQUEST_ACTIONS.TRADE_ACTION_DEAL,
            magic=self._magic,
            symbol=symbol or "",
            volume=volume,
            price=price,
            sl=sl,
            tp=tp,
            deviation=self._deviation,
            type=ENUM_ORDER_TYPE.ORDER_TYPE_SELL,
            type_filling=self._type_filling,
            comment=comment,
        )
        return self._send_request()
    
    def BuyLimit(
        self,
        volume: float,
        price: float,
        symbol: str = None,
        sl: float = 0.0,
        tp: float = 0.0,
        expiration: datetime = None,
        comment: str = "",
    ) -> bool:
        """Place buy limit order."""
        self._request = MqlTradeRequest(
            action=ENUM_TRADE_REQUEST_ACTIONS.TRADE_ACTION_PENDING,
            magic=self._magic,
            symbol=symbol or "",
            volume=volume,
            price=price,
            sl=sl,
            tp=tp,
            type=ENUM_ORDER_TYPE.ORDER_TYPE_BUY_LIMIT,
            type_filling=self._type_filling,
            type_time=ENUM_ORDER_TYPE_TIME.ORDER_TIME_SPECIFIED if expiration else ENUM_ORDER_TYPE_TIME.ORDER_TIME_GTC,
            expiration=expiration,
            comment=comment,
        )
        return self._send_request()
    
    def SellLimit(
        self,
        volume: float,
        price: float,
        symbol: str = None,
        sl: float = 0.0,
        tp: float = 0.0,
        expiration: datetime = None,
        comment: str = "",
    ) -> bool:
        """Place sell limit order."""
        self._request = MqlTradeRequest(
            action=ENUM_TRADE_REQUEST_ACTIONS.TRADE_ACTION_PENDING,
            magic=self._magic,
            symbol=symbol or "",
            volume=volume,
            price=price,
            sl=sl,
            tp=tp,
            type=ENUM_ORDER_TYPE.ORDER_TYPE_SELL_LIMIT,
            type_filling=self._type_filling,
            type_time=ENUM_ORDER_TYPE_TIME.ORDER_TIME_SPECIFIED if expiration else ENUM_ORDER_TYPE_TIME.ORDER_TIME_GTC,
            expiration=expiration,
            comment=comment,
        )
        return self._send_request()
    
    def BuyStop(
        self,
        volume: float,
        price: float,
        symbol: str = None,
        sl: float = 0.0,
        tp: float = 0.0,
        expiration: datetime = None,
        comment: str = "",
    ) -> bool:
        """Place buy stop order."""
        self._request = MqlTradeRequest(
            action=ENUM_TRADE_REQUEST_ACTIONS.TRADE_ACTION_PENDING,
            magic=self._magic,
            symbol=symbol or "",
            volume=volume,
            price=price,
            sl=sl,
            tp=tp,
            type=ENUM_ORDER_TYPE.ORDER_TYPE_BUY_STOP,
            type_filling=self._type_filling,
            type_time=ENUM_ORDER_TYPE_TIME.ORDER_TIME_SPECIFIED if expiration else ENUM_ORDER_TYPE_TIME.ORDER_TIME_GTC,
            expiration=expiration,
            comment=comment,
        )
        return self._send_request()
    
    def SellStop(
        self,
        volume: float,
        price: float,
        symbol: str = None,
        sl: float = 0.0,
        tp: float = 0.0,
        expiration: datetime = None,
        comment: str = "",
    ) -> bool:
        """Place sell stop order."""
        self._request = MqlTradeRequest(
            action=ENUM_TRADE_REQUEST_ACTIONS.TRADE_ACTION_PENDING,
            magic=self._magic,
            symbol=symbol or "",
            volume=volume,
            price=price,
            sl=sl,
            tp=tp,
            type=ENUM_ORDER_TYPE.ORDER_TYPE_SELL_STOP,
            type_filling=self._type_filling,
            type_time=ENUM_ORDER_TYPE_TIME.ORDER_TIME_SPECIFIED if expiration else ENUM_ORDER_TYPE_TIME.ORDER_TIME_GTC,
            expiration=expiration,
            comment=comment,
        )
        return self._send_request()
    
    def PositionClose(
        self,
        ticket: int,
        deviation: int = None,
    ) -> bool:
        """
        Close position by ticket.
        
        Args:
            ticket: Position ticket
            deviation: Price deviation
            
        Returns:
            True if successful
        """
        self._request = MqlTradeRequest(
            action=ENUM_TRADE_REQUEST_ACTIONS.TRADE_ACTION_DEAL,
            magic=self._magic,
            position=ticket,
            deviation=deviation if deviation is not None else self._deviation,
        )
        return self._send_request()
    
    def PositionCloseBy(
        self,
        ticket: int,
        ticket_by: int,
    ) -> bool:
        """Close position by opposite position."""
        self._request = MqlTradeRequest(
            action=ENUM_TRADE_REQUEST_ACTIONS.TRADE_ACTION_CLOSE_BY,
            magic=self._magic,
            position=ticket,
            position_by=ticket_by,
        )
        return self._send_request()
    
    def PositionClosePartial(
        self,
        ticket: int,
        volume: float,
        deviation: int = None,
    ) -> bool:
        """Partially close position."""
        self._request = MqlTradeRequest(
            action=ENUM_TRADE_REQUEST_ACTIONS.TRADE_ACTION_DEAL,
            magic=self._magic,
            position=ticket,
            volume=volume,
            deviation=deviation if deviation is not None else self._deviation,
        )
        return self._send_request()
    
    def PositionModify(
        self,
        ticket: int,
        sl: float,
        tp: float,
    ) -> bool:
        """Modify position stops."""
        self._request = MqlTradeRequest(
            action=ENUM_TRADE_REQUEST_ACTIONS.TRADE_ACTION_SLTP,
            magic=self._magic,
            position=ticket,
            sl=sl,
            tp=tp,
        )
        return self._send_request()
    
    def OrderDelete(self, ticket: int) -> bool:
        """Delete pending order."""
        self._request = MqlTradeRequest(
            action=ENUM_TRADE_REQUEST_ACTIONS.TRADE_ACTION_REMOVE,
            order=ticket,
        )
        return self._send_request()
    
    def OrderModify(
        self,
        ticket: int,
        price: float,
        sl: float,
        tp: float,
        expiration: datetime = None,
        stoplimit: float = 0.0,
    ) -> bool:
        """Modify pending order."""
        self._request = MqlTradeRequest(
            action=ENUM_TRADE_REQUEST_ACTIONS.TRADE_ACTION_MODIFY,
            order=ticket,
            price=price,
            sl=sl,
            tp=tp,
            stoplimit=stoplimit,
            type_time=ENUM_ORDER_TYPE_TIME.ORDER_TIME_SPECIFIED if expiration else ENUM_ORDER_TYPE_TIME.ORDER_TIME_GTC,
            expiration=expiration,
        )
        return self._send_request()


# ═══════════════════════════════════════════════════════════════════════════
# CTerminalInfo - Terminal Properties
# ═══════════════════════════════════════════════════════════════════════════

class CTerminalInfo:
    """
    Class for getting terminal environment properties.
    """
    
    def __init__(self):
        """Initialize terminal info."""
        self._build: int = 0
        self._community_account: bool = False
        self._community_connection: bool = False
        self._connected: bool = True
        self._dlls_allowed: bool = True
        self._trade_allowed: bool = True
        self._email_enabled: bool = False
        self._ftp_enabled: bool = False
        self._notifications_enabled: bool = False
        self._mqid: bool = False
        self._max_bars: int = 100000
        self._codepage: int = 0
        self._cpu_cores: int = 4
        self._memory_physical: int = 16384
        self._memory_total: int = 16384
        self._memory_available: int = 8192
        self._memory_used: int = 8192
        self._opencl_support: int = 0
        self._screen_dpi: int = 96
        self._screen_left: int = 0
        self._screen_top: int = 0
        self._screen_width: int = 1920
        self._screen_height: int = 1080
        self._vps: bool = False
        self._keystate_left: int = 0
        self._keystate_right: int = 0
        self._keystate_control: int = 0
        self._keystate_shift: int = 0
        
        self._language: str = "English"
        self._company: str = ""
        self._name: str = "MetaTrader 5"
        self._path: str = ""
        self._data_path: str = ""
        self._commondata_path: str = ""
    
    def Build(self) -> int:
        return self._build
    
    def IsConnected(self) -> bool:
        return self._connected
    
    def IsDLLsAllowed(self) -> bool:
        return self._dlls_allowed
    
    def IsTradeAllowed(self) -> bool:
        return self._trade_allowed
    
    def MaxBars(self) -> int:
        return self._max_bars
    
    def CPUCores(self) -> int:
        return self._cpu_cores
    
    def MemoryPhysical(self) -> int:
        return self._memory_physical
    
    def MemoryTotal(self) -> int:
        return self._memory_total
    
    def MemoryAvailable(self) -> int:
        return self._memory_available
    
    def MemoryUsed(self) -> int:
        return self._memory_used
    
    def IsVPS(self) -> bool:
        return self._vps
    
    def Language(self) -> str:
        return self._language
    
    def Company(self) -> str:
        return self._company
    
    def Name(self) -> str:
        return self._name
    
    def Path(self) -> str:
        return self._path
    
    def DataPath(self) -> str:
        return self._data_path
    
    def CommonDataPath(self) -> str:
        return self._commondata_path


# ═══════════════════════════════════════════════════════════════════════════
# EXPORT
# ═══════════════════════════════════════════════════════════════════════════

__all__ = [
    # Enumerations
    'ENUM_ACCOUNT_TRADE_MODE',
    'ENUM_ACCOUNT_STOPOUT_MODE',
    'ENUM_ACCOUNT_MARGIN_MODE',
    'ENUM_SYMBOL_CALC_MODE',
    'ENUM_SYMBOL_TRADE_MODE',
    'ENUM_SYMBOL_TRADE_EXECUTION',
    'ENUM_SYMBOL_SWAP_MODE',
    'ENUM_DAY_OF_WEEK',
    'ENUM_ORDER_TYPE',
    'ENUM_ORDER_STATE',
    'ENUM_ORDER_TYPE_FILLING',
    'ENUM_ORDER_TYPE_TIME',
    'ENUM_POSITION_TYPE',
    'ENUM_POSITION_REASON',
    'ENUM_DEAL_TYPE',
    'ENUM_DEAL_ENTRY',
    'ENUM_DEAL_REASON',
    'ENUM_TRADE_REQUEST_ACTIONS',
    'ENUM_TRADE_RETCODE',
    # Classes
    'CAccountInfo',
    'CSymbolInfo',
    'COrderInfo',
    'CHistoryOrderInfo',
    'CPositionInfo',
    'CDealInfo',
    'CTrade',
    'CTerminalInfo',
    # Structures
    'MqlTradeRequest',
    'MqlTradeResult',
]
