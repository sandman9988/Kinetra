"""
Data Quality Module

Comprehensive data validation, gap detection, and quality reporting.
Ensures backtest accuracy by detecting missing data, price anomalies,
and market structure events.
"""

import warnings
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .market_calendar import MarketCalendar, get_calendar_for_symbol
from .symbol_spec import SymbolSpec


class GapType(Enum):
    """Types of data gaps."""

    WEEKEND = "weekend"
    HOLIDAY = "holiday"
    SESSION_BREAK = "session_break"  # Lunch breaks, maintenance
    UNEXPECTED = "unexpected"  # Missing data during trading hours
    ROLLOVER = "rollover"  # Futures contract rollover
    HALT = "halt"  # Trading halt / circuit breaker


@dataclass
class GapInfo:
    """Information about a detected gap in data."""

    start_time: datetime
    end_time: datetime
    expected_bars: int
    gap_type: GapType
    duration_minutes: int = 0

    def __post_init__(self):
        if self.duration_minutes == 0:
            self.duration_minutes = int((self.end_time - self.start_time).total_seconds() / 60)


@dataclass
class OpeningGap:
    """Price gap at session/day open."""

    time: datetime
    prev_close: float
    gap_open: float
    gap_high: float
    gap_low: float
    gap_pct: float  # (open - prev_close) / prev_close
    gap_atr_multiple: float  # Gap size in ATR units
    direction: str  # "up" or "down"

    @property
    def gap_size(self) -> float:
        """Absolute gap size in price."""
        return abs(self.gap_open - self.prev_close)


@dataclass
class RolloverEvent:
    """Futures/CFD rollover event."""

    time: datetime
    old_price: float
    new_price: float
    price_adjustment: float
    symbol: str = ""


@dataclass
class DataQualityReport:
    """Comprehensive data quality report."""

    # Basic stats
    total_bars: int = 0
    first_bar: Optional[datetime] = None
    last_bar: Optional[datetime] = None

    # Gap analysis
    missing_bars: int = 0
    gap_count: int = 0
    gaps: List[GapInfo] = field(default_factory=list)
    gaps_by_type: Dict[GapType, int] = field(default_factory=dict)

    # Opening gaps (price jumps)
    opening_gaps: List[OpeningGap] = field(default_factory=list)
    avg_opening_gap_pct: float = 0.0
    max_opening_gap_pct: float = 0.0

    # OHLC validation
    ohlc_violations: List[int] = field(default_factory=list)
    ohlc_violation_pct: float = 0.0

    # Timestamp issues
    duplicate_timestamps: List[datetime] = field(default_factory=list)
    out_of_order: List[int] = field(default_factory=list)

    # Rollover events
    rollover_events: List[RolloverEvent] = field(default_factory=list)

    # Summary metrics
    completeness_pct: float = 0.0
    quality_score: float = 0.0  # 0-100 score

    # Warnings and recommendations
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "total_bars": self.total_bars,
            "first_bar": self.first_bar.isoformat() if self.first_bar else None,
            "last_bar": self.last_bar.isoformat() if self.last_bar else None,
            "missing_bars": self.missing_bars,
            "gap_count": self.gap_count,
            "gaps_by_type": {k.value: v for k, v in self.gaps_by_type.items()},
            "opening_gap_count": len(self.opening_gaps),
            "avg_opening_gap_pct": self.avg_opening_gap_pct,
            "max_opening_gap_pct": self.max_opening_gap_pct,
            "ohlc_violations": len(self.ohlc_violations),
            "ohlc_violation_pct": self.ohlc_violation_pct,
            "duplicate_timestamps": len(self.duplicate_timestamps),
            "out_of_order_bars": len(self.out_of_order),
            "rollover_events": len(self.rollover_events),
            "completeness_pct": self.completeness_pct,
            "quality_score": self.quality_score,
            "warnings": self.warnings,
        }

    def summary(self) -> str:
        """Human-readable summary."""
        lines = [
            f"Data Quality Report",
            f"=" * 40,
            f"Bars: {self.total_bars:,} ({self.first_bar} to {self.last_bar})",
            f"Completeness: {self.completeness_pct:.1%}",
            f"Quality Score: {self.quality_score:.1f}/100",
            f"",
            f"Gaps: {self.gap_count} ({self.missing_bars} missing bars)",
        ]

        for gap_type, count in self.gaps_by_type.items():
            lines.append(f"  - {gap_type.value}: {count}")

        lines.extend(
            [
                f"",
                f"Opening Gaps: {len(self.opening_gaps)}",
                f"  Avg: {self.avg_opening_gap_pct:.2%}, Max: {self.max_opening_gap_pct:.2%}",
                f"",
                f"OHLC Violations: {len(self.ohlc_violations)} ({self.ohlc_violation_pct:.2%})",
                f"Duplicate Timestamps: {len(self.duplicate_timestamps)}",
                f"Out of Order: {len(self.out_of_order)}",
                f"Rollover Events: {len(self.rollover_events)}",
            ]
        )

        if self.warnings:
            lines.append("")
            lines.append("Warnings:")
            for w in self.warnings:
                lines.append(f"  ! {w}")

        return "\n".join(lines)


class DataQualityChecker:
    """
    Comprehensive data quality checker.

    Validates OHLCV data for:
    - Missing bars / gaps
    - Price discontinuities (opening gaps)
    - OHLC relationship validity
    - Timestamp issues
    - Rollover events

    Usage:
        checker = DataQualityChecker(symbol_spec, calendar)
        report = checker.generate_report(data)

        if report.completeness_pct < 0.95:
            warnings.warn(f"Data incomplete: {report.completeness_pct:.1%}")
    """

    def __init__(
        self,
        symbol_spec: Optional[SymbolSpec] = None,
        calendar: Optional[MarketCalendar] = None,
        timeframe: str = "H1",
    ):
        self.symbol_spec = symbol_spec
        self.calendar = calendar or (
            get_calendar_for_symbol(symbol_spec.symbol) if symbol_spec else None
        )
        self.timeframe = timeframe
        self.timeframe_minutes = self._parse_timeframe(timeframe)

    def _parse_timeframe(self, tf: str) -> int:
        """Parse timeframe to minutes."""
        tf = tf.upper()
        if tf.startswith("M"):
            return int(tf[1:])
        elif tf.startswith("H"):
            return int(tf[1:]) * 60
        elif tf in ("D1", "D"):
            return 24 * 60
        return 60  # Default to H1

    def check_gaps(self, data: pd.DataFrame) -> List[GapInfo]:
        """
        Detect gaps in data based on expected bar frequency.

        Classifies gaps as:
        - Weekend: Expected weekend closure
        - Holiday: Known holiday
        - Session break: Expected intraday break
        - Unexpected: Missing data during expected trading
        """
        gaps = []

        if len(data) < 2:
            return gaps

        # Get timestamps
        if isinstance(data.index, pd.DatetimeIndex):
            timestamps = data.index.to_pydatetime()
        elif "time" in data.columns:
            timestamps = pd.to_datetime(data["time"]).to_pydatetime()
        else:
            return gaps

        expected_delta = timedelta(minutes=self.timeframe_minutes)
        max_gap = expected_delta * 1.5  # Allow 50% tolerance

        for i in range(1, len(timestamps)):
            actual_delta = timestamps[i] - timestamps[i - 1]

            if actual_delta > max_gap:
                # Determine gap type
                gap_type = self._classify_gap(
                    timestamps[i - 1],
                    timestamps[i],
                )

                expected_bars = int(actual_delta.total_seconds() / 60 / self.timeframe_minutes)

                gaps.append(
                    GapInfo(
                        start_time=timestamps[i - 1],
                        end_time=timestamps[i],
                        expected_bars=expected_bars,
                        gap_type=gap_type,
                    )
                )

        return gaps

    def _classify_gap(self, start: datetime, end: datetime) -> GapType:
        """Classify a gap based on calendar and timing."""

        # Check for weekend
        if start.weekday() == 4 and end.weekday() == 0:  # Friday to Monday
            return GapType.WEEKEND
        if start.weekday() >= 5 or end.weekday() == 0:
            return GapType.WEEKEND

        # Check calendar for holidays
        if self.calendar:
            if self.calendar.is_holiday(start) or self.calendar.is_holiday(end):
                return GapType.HOLIDAY

            # Check for session breaks
            breaks = self.calendar.get_session_breaks(start.date())
            for break_start, break_end in breaks:
                if start.time() <= break_start and end.time() >= break_end:
                    return GapType.SESSION_BREAK

        # Default to unexpected
        return GapType.UNEXPECTED

    def check_opening_gaps(
        self,
        data: pd.DataFrame,
        threshold_atr: float = 0.5,
        lookback: int = 20,
    ) -> List[OpeningGap]:
        """
        Detect significant price gaps at bar opens.

        These are price discontinuities that can affect stop-loss execution.

        Args:
            data: OHLCV DataFrame
            threshold_atr: Minimum gap size in ATR units
            lookback: ATR calculation lookback
        """
        opening_gaps = []

        if len(data) < lookback + 1:
            return opening_gaps

        # Calculate ATR
        high = data["high"].values
        low = data["low"].values
        close = data["close"].values
        prev_close = np.roll(close, 1)
        prev_close[0] = close[0]

        tr = np.maximum(high - low, np.maximum(np.abs(high - prev_close), np.abs(low - prev_close)))
        atr = pd.Series(tr).rolling(lookback).mean().values

        # Check each bar for opening gap
        open_prices = data["open"].values

        for i in range(1, len(data)):
            prev_c = close[i - 1]
            curr_open = open_prices[i]
            curr_high = high[i]
            curr_low = low[i]

            gap = curr_open - prev_c
            gap_atr = abs(gap) / atr[i] if atr[i] > 0 else 0

            if gap_atr >= threshold_atr:
                # Get timestamp
                if isinstance(data.index, pd.DatetimeIndex):
                    bar_time = data.index[i].to_pydatetime()
                elif "time" in data.columns:
                    bar_time = pd.to_datetime(data["time"].iloc[i])
                else:
                    bar_time = datetime.now()

                opening_gaps.append(
                    OpeningGap(
                        time=bar_time,
                        prev_close=prev_c,
                        gap_open=curr_open,
                        gap_high=curr_high,
                        gap_low=curr_low,
                        gap_pct=gap / prev_c if prev_c > 0 else 0,
                        gap_atr_multiple=gap_atr,
                        direction="up" if gap > 0 else "down",
                    )
                )

        return opening_gaps

    def check_ohlc_validity(self, data: pd.DataFrame) -> List[int]:
        """
        Validate OHLC relationships.

        Checks:
        - high >= max(open, close)
        - low <= min(open, close)
        - high >= low
        - All prices > 0
        """
        violations = []

        for i in range(len(data)):
            o = data["open"].iloc[i]
            h = data["high"].iloc[i]
            l = data["low"].iloc[i]
            c = data["close"].iloc[i]

            # Check positive prices
            if o <= 0 or h <= 0 or l <= 0 or c <= 0:
                violations.append(i)
                continue

            # High must be highest
            if h < o or h < c:
                violations.append(i)
                continue

            # Low must be lowest
            if l > o or l > c:
                violations.append(i)
                continue

            # High >= Low
            if h < l:
                violations.append(i)
                continue

        return violations

    def check_timestamps(
        self,
        data: pd.DataFrame,
    ) -> Tuple[List[datetime], List[int]]:
        """
        Check for timestamp issues.

        Returns:
            (duplicate_timestamps, out_of_order_indices)
        """
        duplicates = []
        out_of_order = []

        # Get timestamps
        if isinstance(data.index, pd.DatetimeIndex):
            timestamps = data.index.to_pydatetime()
        elif "time" in data.columns:
            timestamps = pd.to_datetime(data["time"]).to_pydatetime()
        else:
            return duplicates, out_of_order

        seen = set()
        prev_ts = None

        for i, ts in enumerate(timestamps):
            # Check duplicates
            if ts in seen:
                duplicates.append(ts)
            seen.add(ts)

            # Check order
            if prev_ts is not None and ts < prev_ts:
                out_of_order.append(i)
            prev_ts = ts

        return duplicates, out_of_order

    def check_rollover_gaps(
        self,
        data: pd.DataFrame,
        threshold_pct: float = 0.01,  # 1% price jump
    ) -> List[RolloverEvent]:
        """
        Detect potential futures/CFD rollover events.

        Large price jumps that are typical of contract rollovers.
        """
        rollovers = []

        if len(data) < 2:
            return rollovers

        close = data["close"].values
        open_prices = data["open"].values

        for i in range(1, len(data)):
            prev_c = close[i - 1]
            curr_o = open_prices[i]

            pct_change = abs(curr_o - prev_c) / prev_c if prev_c > 0 else 0

            # Check if this looks like a rollover (large gap, typically monthly)
            if pct_change >= threshold_pct:
                # Get timestamp
                if isinstance(data.index, pd.DatetimeIndex):
                    bar_time = data.index[i].to_pydatetime()
                elif "time" in data.columns:
                    bar_time = pd.to_datetime(data["time"].iloc[i])
                else:
                    bar_time = datetime.now()

                # Check if this is near month end (typical rollover timing)
                if bar_time.day <= 5 or bar_time.day >= 25:
                    rollovers.append(
                        RolloverEvent(
                            time=bar_time,
                            old_price=prev_c,
                            new_price=curr_o,
                            price_adjustment=curr_o - prev_c,
                            symbol=self.symbol_spec.symbol if self.symbol_spec else "",
                        )
                    )

        return rollovers

    def generate_report(self, data: pd.DataFrame) -> DataQualityReport:
        """
        Generate comprehensive data quality report.

        Args:
            data: OHLCV DataFrame with time index or column

        Returns:
            DataQualityReport with all analysis results
        """
        report = DataQualityReport()

        if len(data) == 0:
            report.warnings.append("Empty dataset")
            return report

        # Basic stats
        report.total_bars = len(data)

        # Get timestamps
        if isinstance(data.index, pd.DatetimeIndex):
            report.first_bar = data.index[0].to_pydatetime()
            report.last_bar = data.index[-1].to_pydatetime()
        elif "time" in data.columns:
            report.first_bar = pd.to_datetime(data["time"].iloc[0])
            report.last_bar = pd.to_datetime(data["time"].iloc[-1])

        # Gap analysis
        report.gaps = self.check_gaps(data)
        report.gap_count = len(report.gaps)
        report.missing_bars = sum(g.expected_bars for g in report.gaps)

        # Count by type
        report.gaps_by_type = {}
        for gap in report.gaps:
            report.gaps_by_type[gap.gap_type] = report.gaps_by_type.get(gap.gap_type, 0) + 1

        # Opening gaps
        report.opening_gaps = self.check_opening_gaps(data)
        if report.opening_gaps:
            gap_pcts = [abs(g.gap_pct) for g in report.opening_gaps]
            report.avg_opening_gap_pct = np.mean(gap_pcts)
            report.max_opening_gap_pct = max(gap_pcts)

        # OHLC validation
        report.ohlc_violations = self.check_ohlc_validity(data)
        report.ohlc_violation_pct = len(report.ohlc_violations) / report.total_bars

        # Timestamp checks
        report.duplicate_timestamps, report.out_of_order = self.check_timestamps(data)

        # Rollover events
        report.rollover_events = self.check_rollover_gaps(data)

        # Calculate completeness
        if self.calendar and report.first_bar and report.last_bar:
            expected = self.calendar.expected_bars(
                report.first_bar,
                report.last_bar,
                self.timeframe,
            )
            if expected > 0:
                report.completeness_pct = report.total_bars / expected
            else:
                report.completeness_pct = 1.0
        else:
            # Estimate based on gaps
            total_expected = report.total_bars + report.missing_bars
            report.completeness_pct = (
                report.total_bars / total_expected if total_expected > 0 else 1.0
            )

        # Calculate quality score (0-100)
        score = 100.0

        # Penalize for issues
        unexpected_gaps = report.gaps_by_type.get(GapType.UNEXPECTED, 0)
        score -= unexpected_gaps * 5  # -5 per unexpected gap

        score -= report.ohlc_violation_pct * 50  # Up to -50 for violations

        score -= len(report.duplicate_timestamps) * 2  # -2 per duplicate
        score -= len(report.out_of_order) * 5  # -5 per out of order

        # Large opening gaps are a warning but not necessarily bad data
        if report.max_opening_gap_pct > 0.05:  # > 5% gap
            score -= 5

        report.quality_score = max(0, min(100, score))

        # Generate warnings
        if report.completeness_pct < 0.95:
            report.warnings.append(f"Data completeness below 95%: {report.completeness_pct:.1%}")

        if unexpected_gaps > 0:
            report.warnings.append(
                f"{unexpected_gaps} unexpected gaps detected during trading hours"
            )

        if report.ohlc_violations:
            report.warnings.append(
                f"{len(report.ohlc_violations)} bars with invalid OHLC relationships"
            )

        if report.duplicate_timestamps:
            report.warnings.append(f"{len(report.duplicate_timestamps)} duplicate timestamps")

        if report.out_of_order:
            report.warnings.append(f"{len(report.out_of_order)} bars out of chronological order")

        if report.max_opening_gap_pct > 0.10:  # > 10% gap
            report.warnings.append(f"Large opening gap detected: {report.max_opening_gap_pct:.1%}")

        return report

    def validate_for_backtest(
        self,
        data: pd.DataFrame,
        min_completeness: float = 0.90,
        allow_ohlc_violations: bool = False,
    ) -> Tuple[bool, DataQualityReport]:
        """
        Validate data is suitable for backtesting.

        Args:
            data: OHLCV DataFrame
            min_completeness: Minimum required completeness
            allow_ohlc_violations: Whether to allow OHLC violations

        Returns:
            (is_valid, report)
        """
        report = self.generate_report(data)

        is_valid = True

        if report.completeness_pct < min_completeness:
            is_valid = False

        if not allow_ohlc_violations and report.ohlc_violations:
            is_valid = False

        if report.duplicate_timestamps:
            is_valid = False

        if report.out_of_order:
            is_valid = False

        return is_valid, report


def validate_data(
    data: pd.DataFrame,
    symbol: str = "",
    timeframe: str = "H1",
    raise_on_error: bool = False,
) -> DataQualityReport:
    """
    Quick validation function.

    Args:
        data: OHLCV DataFrame
        symbol: Symbol for calendar detection
        timeframe: Bar timeframe
        raise_on_error: Raise exception on critical issues

    Returns:
        DataQualityReport
    """
    calendar = get_calendar_for_symbol(symbol) if symbol else None
    checker = DataQualityChecker(calendar=calendar, timeframe=timeframe)
    report = checker.generate_report(data)

    if raise_on_error and report.quality_score < 50:
        raise ValueError(f"Data quality too low: {report.quality_score}/100")

    if report.warnings:
        for w in report.warnings:
            warnings.warn(w)

    return report
