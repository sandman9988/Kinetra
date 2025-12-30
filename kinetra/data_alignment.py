"""
Point-in-Time Data Alignment for Multi-Timeframe Backtesting

Ensures no look-ahead bias when using multiple timeframes:
- At any given timestamp, only completed bars are visible
- Higher timeframe bars only visible after their close time
- Proper alignment of features across timeframes
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


class TimeframeSpec:
    """Specification for a timeframe."""

    # Standard timeframe durations in minutes
    DURATIONS = {
        "M1": 1,
        "M5": 5,
        "M15": 15,
        "M30": 30,
        "H1": 60,
        "H4": 240,
        "D1": 1440,
        "W1": 10080,
        "MN1": 43200,  # Approximate
    }

    def __init__(self, name: str):
        """
        Initialize timeframe spec.

        Args:
            name: Timeframe name (M1, M5, M15, M30, H1, H4, D1, W1, MN1)
        """
        self.name = name.upper()
        if self.name not in self.DURATIONS:
            raise ValueError(f"Unknown timeframe: {name}")
        self.minutes = self.DURATIONS[self.name]

    def bar_close_time(self, bar_open: datetime) -> datetime:
        """Get the close time for a bar that opened at given time."""
        return bar_open + timedelta(minutes=self.minutes)

    def get_completed_bar_time(self, current_time: datetime) -> datetime:
        """
        Get the close time of the most recently completed bar.

        Args:
            current_time: Current timestamp

        Returns:
            Close time of the last completed bar
        """
        # Round down to the start of the current bar
        if self.name == "MN1":
            # Monthly - last completed month
            if current_time.day == 1 and current_time.hour == 0 and current_time.minute == 0:
                # Exactly at month start, previous month just closed
                completed = current_time
            else:
                # Current month is incomplete, get previous month end
                first_of_month = current_time.replace(
                    day=1, hour=0, minute=0, second=0, microsecond=0
                )
                completed = first_of_month
        elif self.name == "W1":
            # Weekly - assumes week starts Monday
            days_since_monday = current_time.weekday()
            week_start = current_time - timedelta(days=days_since_monday)
            week_start = week_start.replace(hour=0, minute=0, second=0, microsecond=0)
            if current_time >= week_start + timedelta(weeks=1):
                completed = week_start + timedelta(weeks=1)
            else:
                completed = week_start
        elif self.name == "D1":
            # Daily
            day_start = current_time.replace(hour=0, minute=0, second=0, microsecond=0)
            if current_time >= day_start + timedelta(days=1):
                completed = day_start + timedelta(days=1)
            else:
                completed = day_start
        else:
            # Intraday timeframes
            total_minutes = current_time.hour * 60 + current_time.minute
            bars_completed = total_minutes // self.minutes
            day_start = current_time.replace(hour=0, minute=0, second=0, microsecond=0)
            completed = day_start + timedelta(minutes=bars_completed * self.minutes)

        return completed

    def __repr__(self) -> str:
        return f"TimeframeSpec({self.name})"


class PointInTimeAligner:
    """
    Aligns multi-timeframe data to ensure point-in-time correctness.

    Usage:
        aligner = PointInTimeAligner()
        aligner.add_data("EURUSD", "M15", df_m15)
        aligner.add_data("EURUSD", "H1", df_h1)
        aligner.add_data("EURUSD", "H4", df_h4)

        # Get aligned data at a specific timestamp
        aligned = aligner.get_aligned_data("EURUSD", current_timestamp)
        # Returns dict with only completed bars visible at that time
    """

    def __init__(self):
        """Initialize the aligner."""
        self.data: Dict[str, Dict[str, pd.DataFrame]] = {}  # symbol -> timeframe -> data
        self.timeframes: Dict[str, TimeframeSpec] = {}

    def add_data(
        self,
        symbol: str,
        timeframe: str,
        data: pd.DataFrame,
        time_column: Optional[str] = None,
    ) -> None:
        """
        Add OHLCV data for a symbol/timeframe combination.

        Args:
            symbol: Instrument symbol
            timeframe: Timeframe string (M15, H1, etc.)
            data: DataFrame with OHLCV data
            time_column: Name of time column (if not index)
        """
        tf_spec = TimeframeSpec(timeframe)
        self.timeframes[timeframe] = tf_spec

        # Ensure datetime index
        df = data.copy()
        if time_column:
            df = df.set_index(time_column)
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)

        # Sort by time
        df = df.sort_index()

        # Store
        if symbol not in self.data:
            self.data[symbol] = {}
        self.data[symbol][timeframe] = df

    def get_last_completed_bar(
        self,
        symbol: str,
        timeframe: str,
        current_time: datetime,
    ) -> Optional[pd.Series]:
        """
        Get the most recently completed bar for a symbol/timeframe.

        Args:
            symbol: Instrument symbol
            timeframe: Timeframe string
            current_time: Current timestamp

        Returns:
            Series with bar data, or None if no completed bar available
        """
        if symbol not in self.data or timeframe not in self.data[symbol]:
            return None

        df = self.data[symbol][timeframe]
        tf_spec = self.timeframes[timeframe]

        # Find the last bar that closed before or at current_time
        # Bar close time = bar index + timeframe duration
        bar_durations = timedelta(minutes=tf_spec.minutes)

        # Bars that have closed by current_time
        closed_mask = df.index + bar_durations <= current_time
        closed_bars = df[closed_mask]

        if len(closed_bars) == 0:
            return None

        return closed_bars.iloc[-1]

    def get_completed_bars(
        self,
        symbol: str,
        timeframe: str,
        current_time: datetime,
        lookback: int = 100,
    ) -> pd.DataFrame:
        """
        Get the last N completed bars for a symbol/timeframe.

        Args:
            symbol: Instrument symbol
            timeframe: Timeframe string
            current_time: Current timestamp
            lookback: Number of bars to return

        Returns:
            DataFrame with completed bars only
        """
        if symbol not in self.data or timeframe not in self.data[symbol]:
            return pd.DataFrame()

        df = self.data[symbol][timeframe]
        tf_spec = self.timeframes[timeframe]

        # Bars that have closed by current_time
        bar_duration = timedelta(minutes=tf_spec.minutes)
        closed_mask = df.index + bar_duration <= current_time
        closed_bars = df[closed_mask]

        return closed_bars.tail(lookback)

    def get_aligned_data(
        self,
        symbol: str,
        current_time: datetime,
        lookback: Dict[str, int] = None,
    ) -> Dict[str, pd.DataFrame]:
        """
        Get point-in-time aligned data across all timeframes for a symbol.

        Args:
            symbol: Instrument symbol
            current_time: Current timestamp
            lookback: Dict of timeframe -> lookback bars (default 100)

        Returns:
            Dict of timeframe -> DataFrame with only completed bars
        """
        if symbol not in self.data:
            return {}

        if lookback is None:
            lookback = {tf: 100 for tf in self.data[symbol].keys()}

        result = {}
        for tf in self.data[symbol].keys():
            lb = lookback.get(tf, 100)
            result[tf] = self.get_completed_bars(symbol, tf, current_time, lb)

        return result

    def get_aligned_features(
        self,
        symbol: str,
        base_timeframe: str,
        feature_timeframes: List[str],
        current_time: datetime,
    ) -> Dict[str, Optional[pd.Series]]:
        """
        Get the last completed bar from each timeframe, aligned to current time.

        Useful for building feature vectors from multiple timeframes.

        Args:
            symbol: Instrument symbol
            base_timeframe: Primary timeframe for trading
            feature_timeframes: List of higher timeframes for features
            current_time: Current timestamp

        Returns:
            Dict of timeframe -> last completed bar Series
        """
        result = {}

        # Base timeframe
        result[base_timeframe] = self.get_last_completed_bar(symbol, base_timeframe, current_time)

        # Feature timeframes
        for tf in feature_timeframes:
            result[tf] = self.get_last_completed_bar(symbol, tf, current_time)

        return result


def resample_to_timeframe(
    data: pd.DataFrame,
    target_timeframe: str,
    source_timeframe: str = "M1",
) -> pd.DataFrame:
    """
    Resample OHLCV data to a higher timeframe.

    Args:
        data: Source DataFrame with OHLCV columns
        target_timeframe: Target timeframe (H1, H4, etc.)
        source_timeframe: Source timeframe (default M1)

    Returns:
        Resampled DataFrame
    """
    tf_spec = TimeframeSpec(target_timeframe)
    rule = f"{tf_spec.minutes}T"  # e.g., "60T" for H1

    # Standard OHLCV resampling
    resampled = data.resample(rule).agg(
        {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum" if "volume" in data.columns else "first",
        }
    )

    # Drop incomplete bars (NaN open means no data in that period)
    resampled = resampled.dropna(subset=["open"])

    return resampled


def align_dataframes_by_time(
    base_df: pd.DataFrame,
    other_df: pd.DataFrame,
    other_tf_minutes: int,
    suffix: str = "",
) -> pd.DataFrame:
    """
    Align a higher timeframe DataFrame to a base timeframe.

    For each row in base_df, finds the last completed bar from other_df.

    Args:
        base_df: Base timeframe DataFrame (e.g., M15)
        other_df: Higher timeframe DataFrame (e.g., H1)
        other_tf_minutes: Duration of other timeframe in minutes
        suffix: Suffix to add to other_df columns

    Returns:
        base_df with aligned columns from other_df
    """
    result = base_df.copy()
    other_duration = timedelta(minutes=other_tf_minutes)

    # For each column in other_df, create aligned version
    for col in other_df.columns:
        new_col = f"{col}{suffix}" if suffix else f"{col}_htf"
        aligned_values = []

        for base_time in base_df.index:
            # Find last completed bar in other_df
            closed_mask = other_df.index + other_duration <= base_time
            closed_bars = other_df[closed_mask]

            if len(closed_bars) > 0:
                aligned_values.append(closed_bars[col].iloc[-1])
            else:
                aligned_values.append(np.nan)

        result[new_col] = aligned_values

    return result
