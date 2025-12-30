"""
Market Events Module

Historical database of significant market events:
- Flash crashes
- Black swan events
- Circuit breakers
- Liquidity crises

Used for:
- Stress testing strategies
- Applying realistic costs during historical events
- Understanding strategy behavior in extreme conditions
"""

import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Set


class EventType(Enum):
    """Types of market events."""

    FLASH_CRASH = "flash_crash"
    BLACK_SWAN = "black_swan"
    CIRCUIT_BREAKER = "circuit_breaker"
    LIQUIDITY_CRISIS = "liquidity_crisis"
    CENTRAL_BANK = "central_bank"  # Major central bank announcements
    GEOPOLITICAL = "geopolitical"
    NATURAL_DISASTER = "natural_disaster"
    EARNINGS_SHOCK = "earnings_shock"
    REGULATORY = "regulatory"


@dataclass
class MarketEvent:
    """
    A significant market event that affected trading conditions.

    Used to apply realistic costs and slippage during backtesting
    when trades overlap with historical events.
    """

    name: str
    date: datetime
    event_type: EventType
    affected_symbols: List[str]  # ["*"] for all, or specific symbols
    severity: float  # 1-10 scale (10 = most severe)
    description: str = ""

    # Impact metrics
    price_impact_pct: float = 0.0  # Typical max drawdown during event
    volatility_multiplier: float = 1.0  # How much volatility increased
    spread_multiplier: float = 1.0  # How much spreads widened
    slippage_multiplier: float = 1.0  # Expected slippage increase

    # Timing
    duration_hours: float = 1.0  # How long the acute phase lasted
    recovery_hours: float = 24.0  # How long until normal conditions

    # Additional context
    tags: List[str] = field(default_factory=list)

    def affects_symbol(self, symbol: str) -> bool:
        """Check if this event affects a given symbol."""
        if "*" in self.affected_symbols:
            return True

        symbol_upper = symbol.upper()
        for affected in self.affected_symbols:
            if affected.upper() in symbol_upper or symbol_upper in affected.upper():
                return True
        return False

    def is_active_at(self, dt: datetime) -> bool:
        """Check if event was active at given datetime."""
        event_end = self.date + timedelta(hours=self.duration_hours)
        return self.date <= dt <= event_end

    def get_cost_multipliers(self, dt: datetime) -> Dict[str, float]:
        """
        Get cost multipliers for a given time relative to event.

        Returns factors that decay over the recovery period.
        """
        if dt < self.date:
            return {"spread": 1.0, "slippage": 1.0, "volatility": 1.0}

        hours_since = (dt - self.date).total_seconds() / 3600

        if hours_since <= self.duration_hours:
            # Peak impact during acute phase
            return {
                "spread": self.spread_multiplier,
                "slippage": self.slippage_multiplier,
                "volatility": self.volatility_multiplier,
            }
        elif hours_since <= self.recovery_hours:
            # Decaying impact during recovery
            recovery_pct = (hours_since - self.duration_hours) / (
                self.recovery_hours - self.duration_hours
            )
            decay = 1 - recovery_pct

            return {
                "spread": 1.0 + (self.spread_multiplier - 1) * decay,
                "slippage": 1.0 + (self.slippage_multiplier - 1) * decay,
                "volatility": 1.0 + (self.volatility_multiplier - 1) * decay,
            }
        else:
            # Fully recovered
            return {"spread": 1.0, "slippage": 1.0, "volatility": 1.0}

    def to_dict(self) -> Dict:
        """Serialize to dictionary."""
        return {
            "name": self.name,
            "date": self.date.isoformat(),
            "event_type": self.event_type.value,
            "affected_symbols": self.affected_symbols,
            "severity": self.severity,
            "description": self.description,
            "price_impact_pct": self.price_impact_pct,
            "volatility_multiplier": self.volatility_multiplier,
            "spread_multiplier": self.spread_multiplier,
            "slippage_multiplier": self.slippage_multiplier,
            "duration_hours": self.duration_hours,
            "recovery_hours": self.recovery_hours,
            "tags": self.tags,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "MarketEvent":
        """Deserialize from dictionary."""
        return cls(
            name=data["name"],
            date=datetime.fromisoformat(data["date"]),
            event_type=EventType(data["event_type"]),
            affected_symbols=data["affected_symbols"],
            severity=data["severity"],
            description=data.get("description", ""),
            price_impact_pct=data.get("price_impact_pct", 0.0),
            volatility_multiplier=data.get("volatility_multiplier", 1.0),
            spread_multiplier=data.get("spread_multiplier", 1.0),
            slippage_multiplier=data.get("slippage_multiplier", 1.0),
            duration_hours=data.get("duration_hours", 1.0),
            recovery_hours=data.get("recovery_hours", 24.0),
            tags=data.get("tags", []),
        )


class MarketEventCalendar:
    """
    Historical database of significant market events.

    Pre-loaded with major historical events across all asset classes.
    Can be extended with custom events.

    Usage:
        calendar = MarketEventCalendar()

        # Check if trade overlaps an event
        event = calendar.overlaps_event(entry_time, exit_time, "EURUSD")
        if event:
            # Apply event costs
            multipliers = event.get_cost_multipliers(current_time)
    """

    def __init__(self, load_historical: bool = True):
        self.events: List[MarketEvent] = []

        if load_historical:
            self._load_historical_events()

    def _load_historical_events(self):
        """Load pre-defined historical events."""

        # === FLASH CRASHES ===

        self.events.append(
            MarketEvent(
                name="Flash Crash 2010",
                date=datetime(2010, 5, 6, 14, 30),  # 2:30 PM ET
                event_type=EventType.FLASH_CRASH,
                affected_symbols=["*"],  # All US equities/indices
                severity=9.0,
                description="Dow Jones dropped ~1000 points in minutes, HFT-triggered",
                price_impact_pct=9.2,
                volatility_multiplier=10.0,
                spread_multiplier=20.0,
                slippage_multiplier=15.0,
                duration_hours=0.5,
                recovery_hours=4.0,
                tags=["hft", "equities", "indices"],
            )
        )

        self.events.append(
            MarketEvent(
                name="Sterling Flash Crash 2016",
                date=datetime(2016, 10, 7, 0, 7),  # Asian session
                event_type=EventType.FLASH_CRASH,
                affected_symbols=["GBPUSD", "GBPJPY", "EURGBP", "GBP"],
                severity=8.0,
                description="GBP dropped 6% in 2 minutes during Asian session",
                price_impact_pct=6.1,
                volatility_multiplier=8.0,
                spread_multiplier=50.0,
                slippage_multiplier=20.0,
                duration_hours=0.1,
                recovery_hours=6.0,
                tags=["forex", "gbp", "asian_session"],
            )
        )

        self.events.append(
            MarketEvent(
                name="JPY Flash Crash 2019",
                date=datetime(2019, 1, 3, 9, 30),  # Tokyo time
                event_type=EventType.FLASH_CRASH,
                affected_symbols=["USDJPY", "AUDJPY", "EURJPY", "JPY"],
                severity=7.5,
                description="Yen surged during thin New Year liquidity",
                price_impact_pct=4.0,
                volatility_multiplier=6.0,
                spread_multiplier=30.0,
                slippage_multiplier=15.0,
                duration_hours=0.25,
                recovery_hours=8.0,
                tags=["forex", "jpy", "new_year", "thin_liquidity"],
            )
        )

        self.events.append(
            MarketEvent(
                name="Crypto Flash Crash May 2021",
                date=datetime(2021, 5, 19, 12, 0),
                event_type=EventType.FLASH_CRASH,
                affected_symbols=["BTCUSD", "ETHUSD", "BTC", "ETH", "CRYPTO"],
                severity=8.5,
                description="Bitcoin dropped 30% on China crypto crackdown fears",
                price_impact_pct=30.0,
                volatility_multiplier=5.0,
                spread_multiplier=10.0,
                slippage_multiplier=8.0,
                duration_hours=4.0,
                recovery_hours=48.0,
                tags=["crypto", "btc", "eth", "china"],
            )
        )

        self.events.append(
            MarketEvent(
                name="Crypto Flash Crash Dec 2021",
                date=datetime(2021, 12, 4, 2, 0),
                event_type=EventType.FLASH_CRASH,
                affected_symbols=["BTCUSD", "ETHUSD", "BTC", "ETH", "CRYPTO"],
                severity=7.0,
                description="Weekend flash crash, BTC dropped 20%",
                price_impact_pct=22.0,
                volatility_multiplier=4.0,
                spread_multiplier=8.0,
                slippage_multiplier=6.0,
                duration_hours=2.0,
                recovery_hours=24.0,
                tags=["crypto", "weekend"],
            )
        )

        # === BLACK SWAN EVENTS ===

        self.events.append(
            MarketEvent(
                name="CHF Black Swan 2015",
                date=datetime(2015, 1, 15, 9, 30),
                event_type=EventType.BLACK_SWAN,
                affected_symbols=["EURCHF", "USDCHF", "GBPCHF", "CHF"],
                severity=10.0,
                description="SNB removed EUR/CHF floor, CHF surged 30%",
                price_impact_pct=30.0,
                volatility_multiplier=50.0,
                spread_multiplier=100.0,
                slippage_multiplier=50.0,
                duration_hours=2.0,
                recovery_hours=72.0,
                tags=["forex", "chf", "snb", "central_bank"],
            )
        )

        self.events.append(
            MarketEvent(
                name="Brexit Vote 2016",
                date=datetime(2016, 6, 24, 4, 0),  # Results came in overnight
                event_type=EventType.BLACK_SWAN,
                affected_symbols=["GBPUSD", "EURGBP", "GBPJPY", "GBP", "UK100"],
                severity=8.5,
                description="UK voted to leave EU, GBP crashed 10%",
                price_impact_pct=11.0,
                volatility_multiplier=8.0,
                spread_multiplier=15.0,
                slippage_multiplier=10.0,
                duration_hours=6.0,
                recovery_hours=168.0,  # Weeks of elevated volatility
                tags=["forex", "gbp", "political", "referendum"],
            )
        )

        self.events.append(
            MarketEvent(
                name="COVID Crash March 2020",
                date=datetime(2020, 3, 12, 9, 30),
                event_type=EventType.BLACK_SWAN,
                affected_symbols=["*"],
                severity=10.0,
                description="Global market crash as COVID pandemic declared",
                price_impact_pct=12.0,  # Single day S&P drop
                volatility_multiplier=6.0,
                spread_multiplier=10.0,
                slippage_multiplier=8.0,
                duration_hours=8.0,
                recovery_hours=720.0,  # Month of chaos
                tags=["pandemic", "global", "circuit_breaker"],
            )
        )

        self.events.append(
            MarketEvent(
                name="COVID Oil Crash 2020",
                date=datetime(2020, 4, 20, 14, 0),
                event_type=EventType.BLACK_SWAN,
                affected_symbols=["CL", "OIL", "USOIL", "WTI", "UKOUSD"],
                severity=10.0,
                description="WTI crude went negative for first time in history",
                price_impact_pct=300.0,  # Went from $17 to -$37
                volatility_multiplier=20.0,
                spread_multiplier=50.0,
                slippage_multiplier=30.0,
                duration_hours=8.0,
                recovery_hours=72.0,
                tags=["commodities", "oil", "covid", "storage"],
            )
        )

        self.events.append(
            MarketEvent(
                name="Trump Election 2016",
                date=datetime(2016, 11, 9, 2, 0),
                event_type=EventType.GEOPOLITICAL,
                affected_symbols=["*"],
                severity=7.0,
                description="Unexpected Trump victory caused overnight volatility",
                price_impact_pct=5.0,
                volatility_multiplier=4.0,
                spread_multiplier=8.0,
                slippage_multiplier=5.0,
                duration_hours=4.0,
                recovery_hours=24.0,
                tags=["political", "us_election"],
            )
        )

        self.events.append(
            MarketEvent(
                name="Russia-Ukraine War Start 2022",
                date=datetime(2022, 2, 24, 4, 0),
                event_type=EventType.GEOPOLITICAL,
                affected_symbols=["*", "XAUUSD", "OIL", "EURUSD", "RUB"],
                severity=9.0,
                description="Russia invaded Ukraine, global risk-off",
                price_impact_pct=4.0,
                volatility_multiplier=5.0,
                spread_multiplier=8.0,
                slippage_multiplier=6.0,
                duration_hours=12.0,
                recovery_hours=168.0,
                tags=["geopolitical", "war", "russia", "ukraine"],
            )
        )

        self.events.append(
            MarketEvent(
                name="FTX Collapse 2022",
                date=datetime(2022, 11, 8, 12, 0),
                event_type=EventType.BLACK_SWAN,
                affected_symbols=["BTCUSD", "ETHUSD", "BTC", "ETH", "CRYPTO", "SOL"],
                severity=9.0,
                description="FTX exchange collapsed, crypto contagion",
                price_impact_pct=25.0,
                volatility_multiplier=6.0,
                spread_multiplier=15.0,
                slippage_multiplier=10.0,
                duration_hours=48.0,
                recovery_hours=336.0,  # 2 weeks
                tags=["crypto", "exchange", "fraud"],
            )
        )

        self.events.append(
            MarketEvent(
                name="SVB Bank Failure 2023",
                date=datetime(2023, 3, 10, 12, 0),
                event_type=EventType.LIQUIDITY_CRISIS,
                affected_symbols=["*", "USDJPY", "XAUUSD"],
                severity=7.5,
                description="Silicon Valley Bank collapsed, banking contagion fears",
                price_impact_pct=5.0,
                volatility_multiplier=3.0,
                spread_multiplier=5.0,
                slippage_multiplier=4.0,
                duration_hours=24.0,
                recovery_hours=168.0,
                tags=["banking", "us", "rates"],
            )
        )

        # === CIRCUIT BREAKERS ===

        self.events.append(
            MarketEvent(
                name="China Circuit Breaker Jan 2016",
                date=datetime(2016, 1, 4, 1, 30),
                event_type=EventType.CIRCUIT_BREAKER,
                affected_symbols=["CHINA50", "HSI", "AUDUSD", "AUD"],
                severity=7.0,
                description="Chinese markets halted twice in first week of 2016",
                price_impact_pct=7.0,
                volatility_multiplier=4.0,
                spread_multiplier=10.0,
                slippage_multiplier=8.0,
                duration_hours=1.0,
                recovery_hours=48.0,
                tags=["china", "circuit_breaker", "asian"],
            )
        )

        self.events.append(
            MarketEvent(
                name="COVID Circuit Breakers March 2020",
                date=datetime(2020, 3, 9, 9, 34),
                event_type=EventType.CIRCUIT_BREAKER,
                affected_symbols=["*"],
                severity=9.5,
                description="Multiple circuit breaker halts during COVID crash",
                price_impact_pct=8.0,
                volatility_multiplier=8.0,
                spread_multiplier=20.0,
                slippage_multiplier=15.0,
                duration_hours=0.25,
                recovery_hours=8.0,
                tags=["circuit_breaker", "covid", "us"],
            )
        )

        # === CENTRAL BANK EVENTS ===

        self.events.append(
            MarketEvent(
                name="ECB QE Announcement 2015",
                date=datetime(2015, 1, 22, 13, 45),
                event_type=EventType.CENTRAL_BANK,
                affected_symbols=["EURUSD", "EURGBP", "EURJPY", "EUR"],
                severity=6.0,
                description="ECB announced massive QE program",
                price_impact_pct=2.5,
                volatility_multiplier=3.0,
                spread_multiplier=5.0,
                slippage_multiplier=3.0,
                duration_hours=2.0,
                recovery_hours=24.0,
                tags=["ecb", "qe", "central_bank"],
            )
        )

        self.events.append(
            MarketEvent(
                name="Fed Emergency Rate Cut 2020",
                date=datetime(2020, 3, 15, 17, 0),  # Sunday evening
                event_type=EventType.CENTRAL_BANK,
                affected_symbols=["*"],
                severity=8.0,
                description="Fed cut rates to zero, emergency action",
                price_impact_pct=5.0,
                volatility_multiplier=5.0,
                spread_multiplier=8.0,
                slippage_multiplier=6.0,
                duration_hours=12.0,
                recovery_hours=72.0,
                tags=["fed", "rates", "covid", "emergency"],
            )
        )

        self.events.append(
            MarketEvent(
                name="BOJ Negative Rates 2016",
                date=datetime(2016, 1, 29, 3, 30),
                event_type=EventType.CENTRAL_BANK,
                affected_symbols=["USDJPY", "EURJPY", "GBPJPY", "JPY"],
                severity=7.0,
                description="Bank of Japan surprised with negative rates",
                price_impact_pct=3.0,
                volatility_multiplier=4.0,
                spread_multiplier=8.0,
                slippage_multiplier=5.0,
                duration_hours=4.0,
                recovery_hours=48.0,
                tags=["boj", "negative_rates", "surprise"],
            )
        )

    def add_event(self, event: MarketEvent):
        """Add a custom event."""
        self.events.append(event)

    def get_events_in_range(
        self,
        start: datetime,
        end: datetime,
    ) -> List[MarketEvent]:
        """Get all events within a date range."""
        return [
            e
            for e in self.events
            if start <= e.date <= end
            or (e.date <= start and e.date + timedelta(hours=e.recovery_hours) >= start)
        ]

    def get_events_for_symbol(self, symbol: str) -> List[MarketEvent]:
        """Get all events affecting a specific symbol."""
        return [e for e in self.events if e.affects_symbol(symbol)]

    def overlaps_event(
        self,
        entry_time: datetime,
        exit_time: datetime,
        symbol: str = "*",
    ) -> Optional[MarketEvent]:
        """
        Check if a trade period overlaps with any market event.

        Returns the most severe overlapping event.
        """
        overlapping = []

        for event in self.events:
            if not event.affects_symbol(symbol):
                continue

            event_end = event.date + timedelta(hours=event.recovery_hours)

            # Check overlap
            if entry_time <= event_end and exit_time >= event.date:
                overlapping.append(event)

        if not overlapping:
            return None

        # Return most severe
        return max(overlapping, key=lambda e: e.severity)

    def get_active_event(
        self,
        dt: datetime,
        symbol: str = "*",
    ) -> Optional[MarketEvent]:
        """Get the active event at a specific time."""
        for event in self.events:
            if not event.affects_symbol(symbol):
                continue

            event_end = event.date + timedelta(hours=event.recovery_hours)
            if event.date <= dt <= event_end:
                return event

        return None

    def get_cost_multipliers(
        self,
        dt: datetime,
        symbol: str = "*",
    ) -> Dict[str, float]:
        """
        Get aggregate cost multipliers for a time and symbol.

        Combines effects if multiple events overlap.
        """
        base = {"spread": 1.0, "slippage": 1.0, "volatility": 1.0}

        for event in self.events:
            if not event.affects_symbol(symbol):
                continue

            event_end = event.date + timedelta(hours=event.recovery_hours)
            if event.date <= dt <= event_end:
                multipliers = event.get_cost_multipliers(dt)
                # Use max of overlapping events
                base["spread"] = max(base["spread"], multipliers["spread"])
                base["slippage"] = max(base["slippage"], multipliers["slippage"])
                base["volatility"] = max(base["volatility"], multipliers["volatility"])

        return base

    def get_events_by_type(self, event_type: EventType) -> List[MarketEvent]:
        """Get all events of a specific type."""
        return [e for e in self.events if e.event_type == event_type]

    def get_events_by_severity(self, min_severity: float = 7.0) -> List[MarketEvent]:
        """Get events above a severity threshold."""
        return [e for e in self.events if e.severity >= min_severity]

    def to_json(self, path: str):
        """Save events to JSON file."""
        data = [e.to_dict() for e in self.events]
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def from_json(cls, path: str) -> "MarketEventCalendar":
        """Load events from JSON file."""
        calendar = cls(load_historical=False)
        with open(path, "r") as f:
            data = json.load(f)
        calendar.events = [MarketEvent.from_dict(e) for e in data]
        return calendar

    def summary(self) -> str:
        """Get summary of loaded events."""
        by_type = {}
        for event in self.events:
            by_type[event.event_type] = by_type.get(event.event_type, 0) + 1

        lines = [
            f"Market Event Calendar",
            f"=" * 40,
            f"Total Events: {len(self.events)}",
            f"",
            f"By Type:",
        ]

        for etype, count in sorted(by_type.items(), key=lambda x: -x[1]):
            lines.append(f"  {etype.value}: {count}")

        lines.extend(
            [
                f"",
                f"Highest Severity Events:",
            ]
        )

        for event in sorted(self.events, key=lambda e: -e.severity)[:5]:
            lines.append(f"  [{event.severity:.1f}] {event.name} ({event.date.date()})")

        return "\n".join(lines)


# Singleton instance
MARKET_EVENT_CALENDAR = MarketEventCalendar()
