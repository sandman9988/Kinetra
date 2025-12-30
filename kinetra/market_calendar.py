"""
Market Calendar Module

Handles trading hours, holidays, and session scheduling per instrument/exchange.
Enables accurate gap detection by knowing when markets are expected to be open.
"""

import json
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple


class Exchange(Enum):
    """Supported exchanges."""

    FOREX = "forex"  # 24/5 Sun 5pm - Fri 5pm ET
    NYSE = "nyse"
    NASDAQ = "nasdaq"
    CME = "cme"  # Futures
    CRYPTO = "crypto"  # 24/7
    LSE = "lse"  # London
    TSE = "tse"  # Tokyo
    CUSTOM = "custom"


@dataclass
class TradingSession:
    """A single trading session within a day."""

    open_time: time
    close_time: time
    session_name: str = ""  # e.g., "Asian", "European", "US"

    def contains(self, t: time) -> bool:
        """Check if time is within this session."""
        if self.open_time <= self.close_time:
            return self.open_time <= t <= self.close_time
        else:
            # Overnight session (e.g., 22:00 - 06:00)
            return t >= self.open_time or t <= self.close_time

    def duration_minutes(self) -> int:
        """Get session duration in minutes."""
        open_mins = self.open_time.hour * 60 + self.open_time.minute
        close_mins = self.close_time.hour * 60 + self.close_time.minute
        if close_mins >= open_mins:
            return close_mins - open_mins
        else:
            return (24 * 60 - open_mins) + close_mins


@dataclass
class DaySchedule:
    """Trading schedule for a single day of the week."""

    sessions: List[TradingSession] = field(default_factory=list)
    is_trading_day: bool = True

    def is_trading_time(self, t: time) -> bool:
        """Check if time is within any trading session."""
        if not self.is_trading_day:
            return False
        return any(session.contains(t) for session in self.sessions)

    def total_trading_minutes(self) -> int:
        """Total trading minutes in this day."""
        if not self.is_trading_day:
            return 0
        return sum(s.duration_minutes() for s in self.sessions)


class MarketCalendar:
    """
    Per-instrument trading calendar.

    Tracks:
    - Trading hours per day of week
    - Market holidays
    - Half days / early closes
    - Session breaks (lunch breaks, etc.)

    Usage:
        calendar = MarketCalendar.forex()
        if calendar.is_trading_time(datetime.now()):
            # Market is open

        expected = calendar.expected_bars(start, end, "H1")
    """

    def __init__(
        self,
        symbol: str = "",
        exchange: Exchange = Exchange.CUSTOM,
        timezone_offset: int = 0,  # Hours from UTC
    ):
        self.symbol = symbol
        self.exchange = exchange
        self.timezone_offset = timezone_offset

        # Day of week -> schedule (0 = Monday, 6 = Sunday)
        self.weekly_schedule: Dict[int, DaySchedule] = {}

        # Specific dates with no trading
        self.holidays: Set[date] = set()

        # Early close dates -> close time
        self.half_days: Dict[date, time] = {}

        # Late open dates -> open time
        self.late_opens: Dict[date, time] = {}

        # Initialize default schedule (all days trading, 24h)
        for day in range(7):
            self.weekly_schedule[day] = DaySchedule(
                sessions=[TradingSession(time(0, 0), time(23, 59))],
                is_trading_day=True,
            )

    def set_weekly_schedule(self, schedule: Dict[int, DaySchedule]):
        """Set the weekly trading schedule."""
        self.weekly_schedule = schedule

    def add_holiday(self, holiday_date: date):
        """Add a market holiday."""
        self.holidays.add(holiday_date)

    def add_holidays(self, dates: List[date]):
        """Add multiple holidays."""
        self.holidays.update(dates)

    def add_half_day(self, half_date: date, close_time: time):
        """Add an early close day."""
        self.half_days[half_date] = close_time

    def is_holiday(self, dt: datetime) -> bool:
        """Check if date is a holiday."""
        return dt.date() in self.holidays

    def is_trading_time(self, dt: datetime) -> bool:
        """
        Check if the given datetime is within trading hours.

        Accounts for:
        - Weekly schedule
        - Holidays
        - Half days
        """
        if self.is_holiday(dt):
            return False

        day_of_week = dt.weekday()
        schedule = self.weekly_schedule.get(day_of_week)

        if schedule is None or not schedule.is_trading_day:
            return False

        current_time = dt.time()

        # Check for half day
        if dt.date() in self.half_days:
            early_close = self.half_days[dt.date()]
            if current_time > early_close:
                return False

        # Check for late open
        if dt.date() in self.late_opens:
            late_open = self.late_opens[dt.date()]
            if current_time < late_open:
                return False

        return schedule.is_trading_time(current_time)

    def next_trading_time(self, dt: datetime) -> datetime:
        """
        Find the next trading time from given datetime.

        Useful for calculating expected bar times.
        """
        if self.is_trading_time(dt):
            return dt

        # Search forward up to 7 days
        current = dt
        for _ in range(7 * 24 * 60):  # Max 7 days in minutes
            current += timedelta(minutes=1)
            if self.is_trading_time(current):
                return current

        return current  # Fallback

    def expected_bars(
        self,
        start: datetime,
        end: datetime,
        timeframe: str,
    ) -> int:
        """
        Calculate expected number of bars between two times.

        Args:
            start: Start datetime
            end: End datetime
            timeframe: Bar timeframe (M1, M5, M15, M30, H1, H4, D1)

        Returns:
            Expected number of complete bars
        """
        # Parse timeframe to minutes
        tf_minutes = self._parse_timeframe_minutes(timeframe)
        if tf_minutes == 0:
            return 0

        # Count trading minutes
        trading_minutes = 0
        current = start

        while current < end:
            if self.is_trading_time(current):
                trading_minutes += 1
            current += timedelta(minutes=1)

        return trading_minutes // tf_minutes

    def get_session_breaks(self, dt: date) -> List[Tuple[time, time]]:
        """
        Get session breaks for a specific date.

        Returns list of (break_start, break_end) tuples.
        """
        if dt in self.holidays:
            return [(time(0, 0), time(23, 59))]  # Whole day is break

        day_of_week = dt.weekday()
        schedule = self.weekly_schedule.get(day_of_week)

        if schedule is None or not schedule.is_trading_day:
            return [(time(0, 0), time(23, 59))]

        # Find gaps between sessions
        breaks = []
        sorted_sessions = sorted(schedule.sessions, key=lambda s: s.open_time)

        for i in range(len(sorted_sessions) - 1):
            current_close = sorted_sessions[i].close_time
            next_open = sorted_sessions[i + 1].open_time
            if current_close < next_open:
                breaks.append((current_close, next_open))

        return breaks

    def get_trading_minutes_per_day(self) -> Dict[int, int]:
        """Get trading minutes for each day of week."""
        return {
            day: schedule.total_trading_minutes() for day, schedule in self.weekly_schedule.items()
        }

    def _parse_timeframe_minutes(self, timeframe: str) -> int:
        """Parse timeframe string to minutes."""
        tf = timeframe.upper()

        if tf.startswith("M"):
            return int(tf[1:])
        elif tf.startswith("H"):
            return int(tf[1:]) * 60
        elif tf == "D1" or tf == "D":
            return 24 * 60
        elif tf == "W1" or tf == "W":
            return 7 * 24 * 60
        elif tf == "MN1" or tf == "MN":
            return 30 * 24 * 60  # Approximate
        else:
            return 0

    def to_dict(self) -> Dict:
        """Serialize to dictionary."""
        return {
            "symbol": self.symbol,
            "exchange": self.exchange.value,
            "timezone_offset": self.timezone_offset,
            "weekly_schedule": {
                day: {
                    "sessions": [
                        {
                            "open_time": s.open_time.isoformat(),
                            "close_time": s.close_time.isoformat(),
                            "session_name": s.session_name,
                        }
                        for s in schedule.sessions
                    ],
                    "is_trading_day": schedule.is_trading_day,
                }
                for day, schedule in self.weekly_schedule.items()
            },
            "holidays": [d.isoformat() for d in self.holidays],
            "half_days": {d.isoformat(): t.isoformat() for d, t in self.half_days.items()},
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "MarketCalendar":
        """Deserialize from dictionary."""
        calendar = cls(
            symbol=data.get("symbol", ""),
            exchange=Exchange(data.get("exchange", "custom")),
            timezone_offset=data.get("timezone_offset", 0),
        )

        # Parse weekly schedule
        for day_str, schedule_data in data.get("weekly_schedule", {}).items():
            day = int(day_str)
            sessions = [
                TradingSession(
                    open_time=time.fromisoformat(s["open_time"]),
                    close_time=time.fromisoformat(s["close_time"]),
                    session_name=s.get("session_name", ""),
                )
                for s in schedule_data.get("sessions", [])
            ]
            calendar.weekly_schedule[day] = DaySchedule(
                sessions=sessions,
                is_trading_day=schedule_data.get("is_trading_day", True),
            )

        # Parse holidays
        for h in data.get("holidays", []):
            calendar.holidays.add(date.fromisoformat(h))

        # Parse half days
        for d, t in data.get("half_days", {}).items():
            calendar.half_days[date.fromisoformat(d)] = time.fromisoformat(t)

        return calendar

    def save_json(self, path: str):
        """Save calendar to JSON file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load_json(cls, path: str) -> "MarketCalendar":
        """Load calendar from JSON file."""
        with open(path, "r") as f:
            data = json.load(f)
        return cls.from_dict(data)

    # === Factory Methods for Common Calendars ===

    @classmethod
    def forex(cls, symbol: str = "") -> "MarketCalendar":
        """
        Standard forex calendar (24/5).

        Sunday 5pm ET (22:00 UTC) to Friday 5pm ET (22:00 UTC)
        """
        calendar = cls(symbol=symbol, exchange=Exchange.FOREX, timezone_offset=0)

        # 24-hour sessions Monday-Thursday
        full_day = DaySchedule(
            sessions=[TradingSession(time(0, 0), time(23, 59))],
            is_trading_day=True,
        )

        # Sunday: Opens at 22:00 UTC
        sunday = DaySchedule(
            sessions=[TradingSession(time(22, 0), time(23, 59))],
            is_trading_day=True,
        )

        # Friday: Closes at 22:00 UTC
        friday = DaySchedule(
            sessions=[TradingSession(time(0, 0), time(22, 0))],
            is_trading_day=True,
        )

        # Saturday: Closed
        saturday = DaySchedule(sessions=[], is_trading_day=False)

        calendar.weekly_schedule = {
            0: full_day,  # Monday
            1: full_day,  # Tuesday
            2: full_day,  # Wednesday
            3: full_day,  # Thursday
            4: friday,  # Friday
            5: saturday,  # Saturday
            6: sunday,  # Sunday
        }

        return calendar

    @classmethod
    def crypto(cls, symbol: str = "") -> "MarketCalendar":
        """
        Crypto calendar (24/7).

        Always trading, no holidays.
        """
        calendar = cls(symbol=symbol, exchange=Exchange.CRYPTO, timezone_offset=0)

        full_day = DaySchedule(
            sessions=[TradingSession(time(0, 0), time(23, 59))],
            is_trading_day=True,
        )

        calendar.weekly_schedule = {i: full_day for i in range(7)}

        return calendar

    @classmethod
    def nyse(cls, symbol: str = "") -> "MarketCalendar":
        """
        NYSE calendar.

        9:30 AM - 4:00 PM ET, Monday-Friday
        """
        calendar = cls(symbol=symbol, exchange=Exchange.NYSE, timezone_offset=-5)

        # Regular session
        trading_day = DaySchedule(
            sessions=[TradingSession(time(9, 30), time(16, 0), "Regular")],
            is_trading_day=True,
        )

        # Weekend closed
        closed = DaySchedule(sessions=[], is_trading_day=False)

        calendar.weekly_schedule = {
            0: trading_day,  # Monday
            1: trading_day,  # Tuesday
            2: trading_day,  # Wednesday
            3: trading_day,  # Thursday
            4: trading_day,  # Friday
            5: closed,  # Saturday
            6: closed,  # Sunday
        }

        # Add major US holidays
        calendar._add_us_holidays()

        return calendar

    @classmethod
    def cme_futures(cls, symbol: str = "") -> "MarketCalendar":
        """
        CME Futures calendar.

        Near 24-hour trading Sunday evening through Friday afternoon.
        With daily maintenance break.
        """
        calendar = cls(symbol=symbol, exchange=Exchange.CME, timezone_offset=-6)

        # Two sessions with break (4:00 PM - 5:00 PM CT maintenance)
        trading_day = DaySchedule(
            sessions=[
                TradingSession(time(17, 0), time(16, 0)),  # Overnight session
            ],
            is_trading_day=True,
        )

        # Friday closes at 4:00 PM CT
        friday = DaySchedule(
            sessions=[TradingSession(time(0, 0), time(16, 0))],
            is_trading_day=True,
        )

        # Saturday closed
        saturday = DaySchedule(sessions=[], is_trading_day=False)

        # Sunday opens at 5:00 PM CT
        sunday = DaySchedule(
            sessions=[TradingSession(time(17, 0), time(23, 59))],
            is_trading_day=True,
        )

        calendar.weekly_schedule = {
            0: trading_day,  # Monday
            1: trading_day,  # Tuesday
            2: trading_day,  # Wednesday
            3: trading_day,  # Thursday
            4: friday,  # Friday
            5: saturday,  # Saturday
            6: sunday,  # Sunday
        }

        calendar._add_us_holidays()

        return calendar

    def _add_us_holidays(self):
        """Add major US market holidays for recent years."""
        # 2024 holidays
        self.add_holidays(
            [
                date(2024, 1, 1),  # New Year's Day
                date(2024, 1, 15),  # MLK Day
                date(2024, 2, 19),  # Presidents Day
                date(2024, 3, 29),  # Good Friday
                date(2024, 5, 27),  # Memorial Day
                date(2024, 6, 19),  # Juneteenth
                date(2024, 7, 4),  # Independence Day
                date(2024, 9, 2),  # Labor Day
                date(2024, 11, 28),  # Thanksgiving
                date(2024, 12, 25),  # Christmas
            ]
        )

        # 2025 holidays
        self.add_holidays(
            [
                date(2025, 1, 1),  # New Year's Day
                date(2025, 1, 20),  # MLK Day
                date(2025, 2, 17),  # Presidents Day
                date(2025, 4, 18),  # Good Friday
                date(2025, 5, 26),  # Memorial Day
                date(2025, 6, 19),  # Juneteenth
                date(2025, 7, 4),  # Independence Day
                date(2025, 9, 1),  # Labor Day
                date(2025, 11, 27),  # Thanksgiving
                date(2025, 12, 25),  # Christmas
            ]
        )

        # Half days (early closes at 1:00 PM ET)
        early_close = time(13, 0)
        self.add_half_day(date(2024, 7, 3), early_close)  # Day before July 4th
        self.add_half_day(date(2024, 11, 29), early_close)  # Day after Thanksgiving
        self.add_half_day(date(2024, 12, 24), early_close)  # Christmas Eve
        self.add_half_day(date(2025, 7, 3), early_close)
        self.add_half_day(date(2025, 11, 28), early_close)
        self.add_half_day(date(2025, 12, 24), early_close)


# === Pre-built Calendar Instances ===

FOREX_24H = MarketCalendar.forex()
CRYPTO_24_7 = MarketCalendar.crypto()
NYSE_CALENDAR = MarketCalendar.nyse()
CME_FUTURES = MarketCalendar.cme_futures()


def get_calendar_for_symbol(symbol: str) -> MarketCalendar:
    """
    Get appropriate calendar based on symbol.

    Heuristic based on symbol naming conventions.
    """
    symbol_upper = symbol.upper()

    # Crypto detection
    if any(c in symbol_upper for c in ["BTC", "ETH", "XRP", "LTC", "DOGE", "SOL"]):
        return MarketCalendar.crypto(symbol)

    # Index futures
    if any(idx in symbol_upper for idx in ["NAS", "DJ30", "SPX", "ES", "NQ", "YM"]):
        return MarketCalendar.cme_futures(symbol)

    # Commodity futures
    if any(
        commodity in symbol_upper for commodity in ["GOLD", "SILVER", "COPPER", "OIL", "CL", "GC"]
    ):
        return MarketCalendar.cme_futures(symbol)

    # Stock symbols (simple heuristic - no currency pair pattern)
    if len(symbol_upper) <= 5 and not any(
        pair in symbol_upper for pair in ["USD", "EUR", "GBP", "JPY", "CHF", "AUD", "CAD", "NZD"]
    ):
        return MarketCalendar.nyse(symbol)

    # Default to forex
    return MarketCalendar.forex(symbol)
