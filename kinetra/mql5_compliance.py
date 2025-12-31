"""
MQL5/MetaAPI Compliance Module
==============================

Ensures compliance with MetaTrader 5 and MetaAPI standards:
- ENUM constants matching MT5 specification
- Order types and trade operations
- Symbol properties (SYMBOL_*)
- Account properties (ACCOUNT_*)
- Position/Order properties
- Error codes (TRADE_RETCODE_*)
- Time handling (server time, GMT offsets)

Reference: https://www.mql5.com/en/docs/constants
"""

from dataclasses import dataclass
from datetime import datetime, time
from enum import IntEnum
from typing import Any, Dict, List, Tuple

# ============================================================================
# MQL5 ENUM Constants
# ============================================================================

class ENUM_TIMEFRAMES(IntEnum):
    """MQL5 Timeframe constants."""
    PERIOD_CURRENT = 0
    PERIOD_M1 = 1
    PERIOD_M2 = 2
    PERIOD_M3 = 3
    PERIOD_M4 = 4
    PERIOD_M5 = 5
    PERIOD_M6 = 6
    PERIOD_M10 = 10
    PERIOD_M12 = 12
    PERIOD_M15 = 15
    PERIOD_M20 = 20
    PERIOD_M30 = 30
    PERIOD_H1 = 16385  # 0x4001
    PERIOD_H2 = 16386
    PERIOD_H3 = 16387
    PERIOD_H4 = 16388
    PERIOD_H6 = 16390
    PERIOD_H8 = 16392
    PERIOD_H12 = 16396
    PERIOD_D1 = 16408  # 0x4018
    PERIOD_W1 = 32769  # 0x8001
    PERIOD_MN1 = 49153  # 0xC001


class ENUM_ORDER_TYPE(IntEnum):
    """MQL5 Order type constants."""
    ORDER_TYPE_BUY = 0
    ORDER_TYPE_SELL = 1
    ORDER_TYPE_BUY_LIMIT = 2
    ORDER_TYPE_SELL_LIMIT = 3
    ORDER_TYPE_BUY_STOP = 4
    ORDER_TYPE_SELL_STOP = 5
    ORDER_TYPE_BUY_STOP_LIMIT = 6
    ORDER_TYPE_SELL_STOP_LIMIT = 7
    ORDER_TYPE_CLOSE_BY = 8


class ENUM_ORDER_TYPE_FILLING(IntEnum):
    """MQL5 Order filling modes."""
    ORDER_FILLING_FOK = 0  # Fill or Kill
    ORDER_FILLING_IOC = 1  # Immediate or Cancel
    ORDER_FILLING_BOC = 2  # Book or Cancel (passive only)
    ORDER_FILLING_RETURN = 3  # Return (partial fill allowed)


class ENUM_ORDER_TYPE_TIME(IntEnum):
    """MQL5 Order time in force."""
    ORDER_TIME_GTC = 0  # Good Till Cancelled
    ORDER_TIME_DAY = 1  # Day order
    ORDER_TIME_SPECIFIED = 2  # Until specified time
    ORDER_TIME_SPECIFIED_DAY = 3  # Until specified day


class ENUM_TRADE_REQUEST_ACTIONS(IntEnum):
    """MQL5 Trade request action types."""
    TRADE_ACTION_DEAL = 1  # Market order
    TRADE_ACTION_PENDING = 5  # Pending order
    TRADE_ACTION_SLTP = 6  # Modify SL/TP
    TRADE_ACTION_MODIFY = 7  # Modify pending order
    TRADE_ACTION_REMOVE = 8  # Delete pending order
    TRADE_ACTION_CLOSE_BY = 10  # Close position by opposite


class ENUM_TRADE_RETCODE(IntEnum):
    """MQL5 Trade return codes."""
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


class ENUM_SYMBOL_INFO_INTEGER(IntEnum):
    """MQL5 Symbol integer properties."""
    SYMBOL_SELECT = 0
    SYMBOL_VISIBLE = 76
    SYMBOL_SESSION_DEALS = 56
    SYMBOL_SESSION_BUY_ORDERS = 60
    SYMBOL_SESSION_SELL_ORDERS = 62
    SYMBOL_VOLUME = 10
    SYMBOL_VOLUMEHIGH = 11
    SYMBOL_VOLUMELOW = 12
    SYMBOL_TIME = 15
    SYMBOL_DIGITS = 17
    SYMBOL_SPREAD_FLOAT = 41
    SYMBOL_SPREAD = 18
    SYMBOL_TRADE_CALC_MODE = 29
    SYMBOL_TRADE_MODE = 30
    SYMBOL_START_TIME = 51
    SYMBOL_EXPIRATION_TIME = 52
    SYMBOL_TRADE_STOPS_LEVEL = 31
    SYMBOL_TRADE_FREEZE_LEVEL = 32
    SYMBOL_TRADE_EXEMODE = 33
    SYMBOL_SWAP_MODE = 37
    SYMBOL_SWAP_ROLLOVER3DAYS = 40
    SYMBOL_MARGIN_HEDGED_USE_LEG = 44
    SYMBOL_EXPIRATION_MODE = 49
    SYMBOL_FILLING_MODE = 50
    SYMBOL_ORDER_MODE = 71
    SYMBOL_ORDER_GTC_MODE = 77
    SYMBOL_OPTION_MODE = 78
    SYMBOL_OPTION_RIGHT = 79


class ENUM_SYMBOL_INFO_DOUBLE(IntEnum):
    """MQL5 Symbol double properties."""
    SYMBOL_BID = 1
    SYMBOL_BIDHIGH = 2
    SYMBOL_BIDLOW = 3
    SYMBOL_ASK = 4
    SYMBOL_ASKHIGH = 5
    SYMBOL_ASKLOW = 6
    SYMBOL_LAST = 7
    SYMBOL_LASTHIGH = 8
    SYMBOL_LASTLOW = 9
    SYMBOL_VOLUME_REAL = 86
    SYMBOL_VOLUMEHIGH_REAL = 87
    SYMBOL_VOLUMELOW_REAL = 88
    SYMBOL_OPTION_STRIKE = 89
    SYMBOL_POINT = 16
    SYMBOL_TRADE_TICK_VALUE = 26
    SYMBOL_TRADE_TICK_VALUE_PROFIT = 53
    SYMBOL_TRADE_TICK_VALUE_LOSS = 54
    SYMBOL_TRADE_TICK_SIZE = 27
    SYMBOL_TRADE_CONTRACT_SIZE = 28
    SYMBOL_TRADE_ACCRUED_INTEREST = 55
    SYMBOL_TRADE_FACE_VALUE = 57
    SYMBOL_TRADE_LIQUIDITY_RATE = 58
    SYMBOL_VOLUME_MIN = 34
    SYMBOL_VOLUME_MAX = 35
    SYMBOL_VOLUME_STEP = 36
    SYMBOL_VOLUME_LIMIT = 59
    SYMBOL_SWAP_LONG = 38
    SYMBOL_SWAP_SHORT = 39
    SYMBOL_MARGIN_INITIAL = 42
    SYMBOL_MARGIN_MAINTENANCE = 43
    SYMBOL_SESSION_VOLUME = 57
    SYMBOL_SESSION_TURNOVER = 58
    SYMBOL_SESSION_INTEREST = 59
    SYMBOL_SESSION_BUY_ORDERS_VOLUME = 61
    SYMBOL_SESSION_SELL_ORDERS_VOLUME = 63
    SYMBOL_SESSION_OPEN = 64
    SYMBOL_SESSION_CLOSE = 65
    SYMBOL_SESSION_AW = 66
    SYMBOL_SESSION_PRICE_SETTLEMENT = 67
    SYMBOL_SESSION_PRICE_LIMIT_MIN = 68
    SYMBOL_SESSION_PRICE_LIMIT_MAX = 69
    SYMBOL_MARGIN_HEDGED = 70
    SYMBOL_PRICE_CHANGE = 82
    SYMBOL_PRICE_VOLATILITY = 83
    SYMBOL_PRICE_THEORETICAL = 84
    SYMBOL_PRICE_GREEKS_DELTA = 85
    SYMBOL_PRICE_GREEKS_THETA = 86
    SYMBOL_PRICE_GREEKS_GAMMA = 87
    SYMBOL_PRICE_GREEKS_VEGA = 88
    SYMBOL_PRICE_GREEKS_RHO = 89
    SYMBOL_PRICE_GREEKS_OMEGA = 90
    SYMBOL_PRICE_SENSITIVITY = 91


class ENUM_SYMBOL_TRADE_MODE(IntEnum):
    """MQL5 Symbol trade modes."""
    SYMBOL_TRADE_MODE_DISABLED = 0
    SYMBOL_TRADE_MODE_LONGONLY = 1
    SYMBOL_TRADE_MODE_SHORTONLY = 2
    SYMBOL_TRADE_MODE_CLOSEONLY = 3
    SYMBOL_TRADE_MODE_FULL = 4


class ENUM_SYMBOL_TRADE_EXECUTION(IntEnum):
    """MQL5 Trade execution modes."""
    SYMBOL_TRADE_EXECUTION_REQUEST = 0
    SYMBOL_TRADE_EXECUTION_INSTANT = 1
    SYMBOL_TRADE_EXECUTION_MARKET = 2
    SYMBOL_TRADE_EXECUTION_EXCHANGE = 3


class ENUM_SYMBOL_SWAP_MODE(IntEnum):
    """MQL5 Swap calculation modes."""
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
    """MQL5 Day of week constants."""
    SUNDAY = 0
    MONDAY = 1
    TUESDAY = 2
    WEDNESDAY = 3
    THURSDAY = 4
    FRIDAY = 5
    SATURDAY = 6


class ENUM_ACCOUNT_INFO_INTEGER(IntEnum):
    """MQL5 Account integer properties."""
    ACCOUNT_LOGIN = 0
    ACCOUNT_TRADE_MODE = 32
    ACCOUNT_LEVERAGE = 35
    ACCOUNT_LIMIT_ORDERS = 47
    ACCOUNT_MARGIN_SO_MODE = 44
    ACCOUNT_TRADE_ALLOWED = 33
    ACCOUNT_TRADE_EXPERT = 34
    ACCOUNT_MARGIN_MODE = 45
    ACCOUNT_CURRENCY_DIGITS = 46
    ACCOUNT_FIFO_CLOSE = 48


class ENUM_ACCOUNT_INFO_DOUBLE(IntEnum):
    """MQL5 Account double properties."""
    ACCOUNT_BALANCE = 37
    ACCOUNT_CREDIT = 38
    ACCOUNT_PROFIT = 39
    ACCOUNT_EQUITY = 40
    ACCOUNT_MARGIN = 41
    ACCOUNT_MARGIN_FREE = 42
    ACCOUNT_MARGIN_LEVEL = 43
    ACCOUNT_MARGIN_SO_CALL = 45
    ACCOUNT_MARGIN_SO_SO = 46
    ACCOUNT_MARGIN_INITIAL = 50
    ACCOUNT_MARGIN_MAINTENANCE = 51
    ACCOUNT_ASSETS = 52
    ACCOUNT_LIABILITIES = 53
    ACCOUNT_COMMISSION_BLOCKED = 54


class ENUM_POSITION_PROPERTY_INTEGER(IntEnum):
    """MQL5 Position integer properties."""
    POSITION_TICKET = 1
    POSITION_TIME = 2
    POSITION_TIME_MSC = 26
    POSITION_TIME_UPDATE = 27
    POSITION_TIME_UPDATE_MSC = 28
    POSITION_TYPE = 3
    POSITION_MAGIC = 4
    POSITION_IDENTIFIER = 5


class ENUM_POSITION_PROPERTY_DOUBLE(IntEnum):
    """MQL5 Position double properties."""
    POSITION_VOLUME = 6
    POSITION_PRICE_OPEN = 7
    POSITION_SL = 8
    POSITION_TP = 9
    POSITION_PRICE_CURRENT = 10
    POSITION_COMMISSION = 11
    POSITION_SWAP = 12
    POSITION_PROFIT = 13


# ============================================================================
# MQL5 Data Structures
# ============================================================================

@dataclass
class MqlTradeRequest:
    """MQL5 Trade request structure."""
    action: ENUM_TRADE_REQUEST_ACTIONS
    magic: int = 0
    order: int = 0
    symbol: str = ""
    volume: float = 0.0
    price: float = 0.0
    stoplimit: float = 0.0
    sl: float = 0.0
    tp: float = 0.0
    deviation: int = 0
    type: ENUM_ORDER_TYPE = ENUM_ORDER_TYPE.ORDER_TYPE_BUY
    type_filling: ENUM_ORDER_TYPE_FILLING = ENUM_ORDER_TYPE_FILLING.ORDER_FILLING_FOK
    type_time: ENUM_ORDER_TYPE_TIME = ENUM_ORDER_TYPE_TIME.ORDER_TIME_GTC
    expiration: datetime = None
    comment: str = ""
    position: int = 0
    position_by: int = 0


@dataclass
class MqlTradeResult:
    """MQL5 Trade result structure."""
    retcode: ENUM_TRADE_RETCODE = ENUM_TRADE_RETCODE.TRADE_RETCODE_ERROR
    deal: int = 0
    order: int = 0
    volume: float = 0.0
    price: float = 0.0
    bid: float = 0.0
    ask: float = 0.0
    comment: str = ""
    request_id: int = 0
    retcode_external: int = 0


@dataclass
class MqlRates:
    """MQL5 OHLCV bar structure."""
    time: datetime
    open: float
    high: float
    low: float
    close: float
    tick_volume: int
    spread: int
    real_volume: int = 0


@dataclass
class MqlTick:
    """MQL5 Tick structure."""
    time: datetime
    bid: float
    ask: float
    last: float
    volume: int
    time_msc: int = 0
    flags: int = 0
    volume_real: float = 0.0


@dataclass
class MqlBookInfo:
    """MQL5 Market depth entry."""
    type: int  # BOOK_TYPE_SELL or BOOK_TYPE_BUY
    price: float
    volume: int
    volume_real: float = 0.0


# ============================================================================
# Compliance Helpers
# ============================================================================

def timeframe_to_string(tf: ENUM_TIMEFRAMES) -> str:
    """Convert MQL5 timeframe to string (like MQL5 TimeframeToString)."""
    mapping = {
        ENUM_TIMEFRAMES.PERIOD_M1: "M1",
        ENUM_TIMEFRAMES.PERIOD_M5: "M5",
        ENUM_TIMEFRAMES.PERIOD_M15: "M15",
        ENUM_TIMEFRAMES.PERIOD_M30: "M30",
        ENUM_TIMEFRAMES.PERIOD_H1: "H1",
        ENUM_TIMEFRAMES.PERIOD_H4: "H4",
        ENUM_TIMEFRAMES.PERIOD_D1: "D1",
        ENUM_TIMEFRAMES.PERIOD_W1: "W1",
        ENUM_TIMEFRAMES.PERIOD_MN1: "MN1",
    }
    return mapping.get(tf, "UNKNOWN")


def string_to_timeframe(s: str) -> ENUM_TIMEFRAMES:
    """Convert string to MQL5 timeframe."""
    mapping = {
        "M1": ENUM_TIMEFRAMES.PERIOD_M1,
        "M5": ENUM_TIMEFRAMES.PERIOD_M5,
        "M15": ENUM_TIMEFRAMES.PERIOD_M15,
        "M30": ENUM_TIMEFRAMES.PERIOD_M30,
        "H1": ENUM_TIMEFRAMES.PERIOD_H1,
        "H4": ENUM_TIMEFRAMES.PERIOD_H4,
        "D1": ENUM_TIMEFRAMES.PERIOD_D1,
        "W1": ENUM_TIMEFRAMES.PERIOD_W1,
        "MN1": ENUM_TIMEFRAMES.PERIOD_MN1,
    }
    return mapping.get(s.upper(), ENUM_TIMEFRAMES.PERIOD_H1)


def timeframe_to_minutes(tf: ENUM_TIMEFRAMES) -> int:
    """Convert timeframe to minutes."""
    mapping = {
        ENUM_TIMEFRAMES.PERIOD_M1: 1,
        ENUM_TIMEFRAMES.PERIOD_M5: 5,
        ENUM_TIMEFRAMES.PERIOD_M15: 15,
        ENUM_TIMEFRAMES.PERIOD_M30: 30,
        ENUM_TIMEFRAMES.PERIOD_H1: 60,
        ENUM_TIMEFRAMES.PERIOD_H4: 240,
        ENUM_TIMEFRAMES.PERIOD_D1: 1440,
        ENUM_TIMEFRAMES.PERIOD_W1: 10080,
        ENUM_TIMEFRAMES.PERIOD_MN1: 43200,
    }
    return mapping.get(tf, 60)


def retcode_to_string(retcode: ENUM_TRADE_RETCODE) -> str:
    """Get human-readable description of trade return code."""
    descriptions = {
        ENUM_TRADE_RETCODE.TRADE_RETCODE_DONE: "Request completed",
        ENUM_TRADE_RETCODE.TRADE_RETCODE_DONE_PARTIAL: "Request completed partially",
        ENUM_TRADE_RETCODE.TRADE_RETCODE_ERROR: "Request processing error",
        ENUM_TRADE_RETCODE.TRADE_RETCODE_TIMEOUT: "Request timed out",
        ENUM_TRADE_RETCODE.TRADE_RETCODE_INVALID: "Invalid request",
        ENUM_TRADE_RETCODE.TRADE_RETCODE_INVALID_VOLUME: "Invalid volume",
        ENUM_TRADE_RETCODE.TRADE_RETCODE_INVALID_PRICE: "Invalid price",
        ENUM_TRADE_RETCODE.TRADE_RETCODE_INVALID_STOPS: "Invalid stops",
        ENUM_TRADE_RETCODE.TRADE_RETCODE_TRADE_DISABLED: "Trade is disabled",
        ENUM_TRADE_RETCODE.TRADE_RETCODE_MARKET_CLOSED: "Market is closed",
        ENUM_TRADE_RETCODE.TRADE_RETCODE_NO_MONEY: "Not enough money",
        ENUM_TRADE_RETCODE.TRADE_RETCODE_PRICE_CHANGED: "Price changed",
        ENUM_TRADE_RETCODE.TRADE_RETCODE_FROZEN: "Trade frozen",
        ENUM_TRADE_RETCODE.TRADE_RETCODE_INVALID_FILL: "Invalid order filling type",
        ENUM_TRADE_RETCODE.TRADE_RETCODE_LONG_ONLY: "Only long positions allowed",
        ENUM_TRADE_RETCODE.TRADE_RETCODE_SHORT_ONLY: "Only short positions allowed",
        ENUM_TRADE_RETCODE.TRADE_RETCODE_CLOSE_ONLY: "Only position closing allowed",
    }
    return descriptions.get(retcode, f"Unknown error ({retcode})")


def validate_volume(
    volume: float,
    volume_min: float,
    volume_max: float,
    volume_step: float
) -> Tuple[bool, str, float]:
    """
    Validate and normalize volume according to MT5 rules.

    Returns:
        (is_valid, error_message, normalized_volume)
    """
    if volume < volume_min:
        return False, f"Volume {volume} below minimum {volume_min}", volume_min

    if volume > volume_max:
        return False, f"Volume {volume} above maximum {volume_max}", volume_max

    # Normalize to volume_step
    if volume_step > 0:
        normalized = round(volume / volume_step) * volume_step
        normalized = round(normalized, 8)  # Avoid floating point issues
    else:
        normalized = volume

    return True, "", normalized


def validate_stops(
    order_type: ENUM_ORDER_TYPE,
    price: float,
    sl: float,
    tp: float,
    stops_level: int,
    point: float
) -> Tuple[bool, ENUM_TRADE_RETCODE, str]:
    """
    Validate SL/TP placement according to MT5 rules.

    Args:
        order_type: Buy or Sell
        price: Entry/current price
        sl: Stop Loss price
        tp: Take Profit price
        stops_level: Minimum distance in points (from SYMBOL_TRADE_STOPS_LEVEL)
        point: Symbol point value

    Returns:
        (is_valid, retcode, error_message)
    """
    min_distance = stops_level * point

    if order_type in [ENUM_ORDER_TYPE.ORDER_TYPE_BUY, ENUM_ORDER_TYPE.ORDER_TYPE_BUY_LIMIT,
                      ENUM_ORDER_TYPE.ORDER_TYPE_BUY_STOP]:
        # For buy orders: SL below price, TP above price
        if sl > 0 and price - sl < min_distance:
            return False, ENUM_TRADE_RETCODE.TRADE_RETCODE_INVALID_STOPS, \
                   f"SL too close: {(price - sl) / point:.1f} points, minimum {stops_level}"

        if tp > 0 and tp - price < min_distance:
            return False, ENUM_TRADE_RETCODE.TRADE_RETCODE_INVALID_STOPS, \
                   f"TP too close: {(tp - price) / point:.1f} points, minimum {stops_level}"
    else:
        # For sell orders: SL above price, TP below price
        if sl > 0 and sl - price < min_distance:
            return False, ENUM_TRADE_RETCODE.TRADE_RETCODE_INVALID_STOPS, \
                   f"SL too close: {(sl - price) / point:.1f} points, minimum {stops_level}"

        if tp > 0 and price - tp < min_distance:
            return False, ENUM_TRADE_RETCODE.TRADE_RETCODE_INVALID_STOPS, \
                   f"TP too close: {(price - tp) / point:.1f} points, minimum {stops_level}"

    return True, ENUM_TRADE_RETCODE.TRADE_RETCODE_DONE, ""


def calculate_swap(
    swap_mode: ENUM_SYMBOL_SWAP_MODE,
    swap_long: float,
    swap_short: float,
    position_type: ENUM_ORDER_TYPE,
    lots: float,
    contract_size: float,
    point: float,
    tick_value: float,
    price: float,
    days: int = 1,
    swap_rollover3days: ENUM_DAY_OF_WEEK = ENUM_DAY_OF_WEEK.WEDNESDAY,
    day_of_week: ENUM_DAY_OF_WEEK = ENUM_DAY_OF_WEEK.MONDAY
) -> float:
    """
    Calculate swap according to MT5 swap modes.

    MT5 swap calculation modes:
    - POINTS: Swap = rate * point * contract_size * lots
    - CURRENCY_SYMBOL: Swap = rate * lots (directly in symbol currency)
    - CURRENCY_MARGIN: Swap = rate * lots (in margin currency)
    - CURRENCY_DEPOSIT: Swap = rate * lots (in deposit currency)
    - INTEREST_CURRENT: Swap = rate% * position_value / 360
    - INTEREST_OPEN: Swap = rate% * (lots * contract_size * open_price) / 360
    - REOPEN_CURRENT: Position reopened at current price
    - REOPEN_BID: Position reopened at bid price
    """
    is_long = position_type in [ENUM_ORDER_TYPE.ORDER_TYPE_BUY]
    swap_rate = swap_long if is_long else swap_short

    # Account for triple swap day (weekend rollover)
    effective_days = days
    if day_of_week == swap_rollover3days:
        effective_days = days * 3

    if swap_mode == ENUM_SYMBOL_SWAP_MODE.SYMBOL_SWAP_MODE_DISABLED:
        return 0.0

    elif swap_mode == ENUM_SYMBOL_SWAP_MODE.SYMBOL_SWAP_MODE_POINTS:
        # Swap in points
        return swap_rate * point * contract_size * lots * effective_days

    elif swap_mode in [ENUM_SYMBOL_SWAP_MODE.SYMBOL_SWAP_MODE_CURRENCY_SYMBOL,
                       ENUM_SYMBOL_SWAP_MODE.SYMBOL_SWAP_MODE_CURRENCY_MARGIN,
                       ENUM_SYMBOL_SWAP_MODE.SYMBOL_SWAP_MODE_CURRENCY_DEPOSIT]:
        # Direct currency value per lot
        return swap_rate * lots * effective_days

    elif swap_mode in [ENUM_SYMBOL_SWAP_MODE.SYMBOL_SWAP_MODE_INTEREST_CURRENT,
                       ENUM_SYMBOL_SWAP_MODE.SYMBOL_SWAP_MODE_INTEREST_OPEN]:
        # Annual interest rate as percentage
        position_value = lots * contract_size * price
        daily_rate = swap_rate / 100 / 360  # Annual rate to daily
        return position_value * daily_rate * effective_days

    else:
        # Default: treat as points
        return swap_rate * tick_value * lots * effective_days


def normalize_price(price: float, digits: int, tick_size: float = 0) -> float:
    """Normalize price to symbol precision."""
    if tick_size > 0:
        return round(price / tick_size) * tick_size
    return round(price, digits)


def is_market_open(
    current_time: datetime,
    trading_hours: List[Tuple[time, time]],
    day_of_week: int = None
) -> bool:
    """
    Check if market is open based on trading hours.

    Args:
        current_time: Current server time
        trading_hours: List of (start_time, end_time) tuples
        day_of_week: Optional day filter (0=Sunday, 6=Saturday)

    Returns:
        True if market is open
    """
    if day_of_week is None:
        day_of_week = current_time.weekday()

    # Most forex markets closed on weekends
    if day_of_week in [5, 6]:  # Saturday, Sunday
        return False

    current_t = current_time.time()

    for start, end in trading_hours:
        if start <= end:
            # Normal session (e.g., 09:00 - 17:00)
            if start <= current_t <= end:
                return True
        else:
            # Overnight session (e.g., 22:00 - 06:00)
            if current_t >= start or current_t <= end:
                return True

    return False


# ============================================================================
# MetaAPI Compliance
# ============================================================================

class MetaAPICompliance:
    """
    Ensures compliance with MetaAPI SDK standards.

    MetaAPI differences from direct MT5:
    - Async by default
    - Different authentication flow
    - Rate limiting
    - Cloud-specific error codes
    """

    # MetaAPI-specific error codes
    METAAPI_ERROR_CODES = {
        "E_AUTH": "Authentication failed",
        "E_SERVER": "Server error",
        "E_TIMEOUT": "Request timed out",
        "E_RATE_LIMIT": "Rate limit exceeded",
        "E_NOT_FOUND": "Resource not found",
        "E_VALIDATION": "Validation error",
        "E_FORBIDDEN": "Access forbidden",
    }

    # Rate limits (requests per second)
    RATE_LIMITS = {
        "rpc": 60,  # RPC requests
        "streaming": 100,  # Streaming subscriptions
        "history": 10,  # Historical data requests
    }

    @staticmethod
    def validate_account_id(account_id: str) -> bool:
        """Validate MetaAPI account ID format (UUID)."""
        import re
        uuid_pattern = r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$'
        return bool(re.match(uuid_pattern, account_id.lower()))

    @staticmethod
    def validate_token(token: str) -> bool:
        """Basic validation of MetaAPI token format."""
        # MetaAPI tokens are JWT format
        parts = token.split('.')
        return len(parts) == 3 and all(len(p) > 10 for p in parts)

    @staticmethod
    def map_mt5_retcode_to_metaapi(retcode: ENUM_TRADE_RETCODE) -> Dict[str, Any]:
        """Map MT5 return code to MetaAPI error format."""
        return {
            "error": retcode.name,
            "numericCode": int(retcode),
            "message": retcode_to_string(retcode),
            "details": {}
        }


# ============================================================================
# Export all for easy importing
# ============================================================================

__all__ = [
    # Enums
    "ENUM_TIMEFRAMES",
    "ENUM_ORDER_TYPE",
    "ENUM_ORDER_TYPE_FILLING",
    "ENUM_ORDER_TYPE_TIME",
    "ENUM_TRADE_REQUEST_ACTIONS",
    "ENUM_TRADE_RETCODE",
    "ENUM_SYMBOL_INFO_INTEGER",
    "ENUM_SYMBOL_INFO_DOUBLE",
    "ENUM_SYMBOL_TRADE_MODE",
    "ENUM_SYMBOL_TRADE_EXECUTION",
    "ENUM_SYMBOL_SWAP_MODE",
    "ENUM_DAY_OF_WEEK",
    "ENUM_ACCOUNT_INFO_INTEGER",
    "ENUM_ACCOUNT_INFO_DOUBLE",
    "ENUM_POSITION_PROPERTY_INTEGER",
    "ENUM_POSITION_PROPERTY_DOUBLE",
    # Data structures
    "MqlTradeRequest",
    "MqlTradeResult",
    "MqlRates",
    "MqlTick",
    "MqlBookInfo",
    # Helper functions
    "timeframe_to_string",
    "string_to_timeframe",
    "timeframe_to_minutes",
    "retcode_to_string",
    "validate_volume",
    "validate_stops",
    "calculate_swap",
    "normalize_price",
    "is_market_open",
    # MetaAPI
    "MetaAPICompliance",
]
