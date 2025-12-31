"""
DataPackage: Standardized data container for backtesting and exploration.

Bridges the gap between raw CSV data and backtest/RL engines.
Ensures consistent data format across all trading systems.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Tuple, Dict, Any
from enum import Enum

import pandas as pd

from .market_microstructure import AssetClass, SymbolSpec


class DataFormat(Enum):
    """Output format for different engines."""
    BACKTEST_ENGINE = "backtest_engine"      # kinetra/backtest_engine.py (lowercase)
    PHYSICS_BACKTESTER = "physics_backtester"  # Backtesting.py format (Capitalized)
    RL_ENVIRONMENT = "rl_environment"        # RL exploration (physics_state + prices)
    RAW = "raw"                              # Original OHLCV only


@dataclass
class DataPackage:
    """
    Standardized data package for backtesting and exploration.

    Contains all necessary components for trading system execution:
    - Price data (OHLCV)
    - Physics state (computed features)
    - Symbol specifications
    - Market metadata
    - Quality reports

    Provides conversion methods for different backtest engine formats.
    """

    # ========== Core Data ==========
    prices: pd.DataFrame
    """OHLCV price data with DatetimeIndex.
    Columns: open, high, low, close, volume (lowercase)"""

    physics_state: Optional[pd.DataFrame] = None
    """Physics engine features (64-dim state vector).
    Computed from prices if not provided."""

    # ========== Metadata ==========
    symbol: str = ""
    """Trading symbol (e.g., BTCUSD, EURUSD)"""

    timeframe: str = ""
    """Data timeframe (e.g., H1, M30, D1)"""

    market_type: AssetClass = AssetClass.FOREX
    """Market classification (forex/crypto/shares/indices/metals/energy/etfs)"""

    # ========== Specifications ==========
    symbol_spec: Optional[SymbolSpec] = None
    """Symbol contract specification (spreads, swaps, margins)"""

    # ========== Quality & Validation ==========
    quality_report: Optional[Dict[str, Any]] = None
    """Data quality validation results"""

    is_validated: bool = False
    """Whether data has passed quality checks"""

    validation_warnings: list = field(default_factory=list)
    """List of validation warnings (non-fatal)"""

    validation_errors: list = field(default_factory=list)
    """List of validation errors (fatal)"""

    # ========== Time Range ==========
    start_date: Optional[datetime] = None
    """First timestamp in dataset"""

    end_date: Optional[datetime] = None
    """Last timestamp in dataset"""

    total_bars: int = 0
    """Total number of price bars"""

    # ========== Processing Metadata ==========
    source_file: Optional[str] = None
    """Original CSV file path"""

    preprocessing_applied: list = field(default_factory=list)
    """List of preprocessing steps applied"""

    cache_path: Optional[str] = None
    """Path to cached physics state (if cached)"""

    def __post_init__(self):
        """Auto-compute metadata from prices if not provided."""
        if len(self.prices) > 0:
            if self.start_date is None:
                self.start_date = self.prices.index[0]
            if self.end_date is None:
                self.end_date = self.prices.index[-1]
            if self.total_bars == 0:
                self.total_bars = len(self.prices)

    # ========== Format Conversion Methods ==========

    def to_backtest_engine_format(self) -> pd.DataFrame:
        """
        Convert to backtest_engine.py format (lowercase OHLCV).

        Returns:
            DataFrame with columns: open, high, low, close, volume
        """
        df = self.prices.copy()

        # Ensure lowercase column names
        df.columns = df.columns.str.lower()

        # Ensure required columns exist
        required = ['open', 'high', 'low', 'close']
        for col in required:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")

        # Add volume if missing (use zeros)
        if 'volume' not in df.columns:
            df['volume'] = 0

        return df[['open', 'high', 'low', 'close', 'volume']]

    def to_physics_backtester_format(self) -> pd.DataFrame:
        """
        Convert to Backtesting.py format (Capitalized OHLCV).

        Returns:
            DataFrame with columns: Open, High, Low, Close, Volume
        """
        df = self.to_backtest_engine_format()

        # Capitalize column names
        df.columns = df.columns.str.capitalize()

        return df

    def to_rl_environment_format(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Convert to RL exploration format.

        Returns:
            (physics_state, prices) tuple
            - physics_state: 64-dim feature vector per timestep
            - prices: OHLCV data (lowercase)
        """
        if self.physics_state is None:
            raise ValueError(
                "Physics state not computed. Call compute_physics_state() first "
                "or use UnifiedDataLoader with compute_physics=True"
            )

        return self.physics_state, self.to_backtest_engine_format()

    def to_raw_format(self) -> pd.DataFrame:
        """
        Return raw OHLCV data as-is.

        Returns:
            Original prices DataFrame
        """
        return self.prices.copy()

    def to_format(self, format_type: DataFormat) -> pd.DataFrame | Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Convert to specified format.

        Args:
            format_type: Target format (from DataFormat enum)

        Returns:
            Converted data in requested format
        """
        if format_type == DataFormat.BACKTEST_ENGINE:
            return self.to_backtest_engine_format()
        elif format_type == DataFormat.PHYSICS_BACKTESTER:
            return self.to_physics_backtester_format()
        elif format_type == DataFormat.RL_ENVIRONMENT:
            return self.to_rl_environment_format()
        elif format_type == DataFormat.RAW:
            return self.to_raw_format()
        else:
            raise ValueError(f"Unknown format: {format_type}")

    # ========== Validation Methods ==========

    def validate(self) -> bool:
        """
        Run validation checks on the data.

        Returns:
            True if validation passed, False otherwise
        """
        self.validation_warnings.clear()
        self.validation_errors.clear()

        # Check for empty data
        if len(self.prices) == 0:
            self.validation_errors.append("Empty price data")
            self.is_validated = False
            return False

        # Check for required columns
        required = ['open', 'high', 'low', 'close']
        df_cols = [c.lower() for c in self.prices.columns]
        missing = [col for col in required if col not in df_cols]
        if missing:
            self.validation_errors.append(f"Missing columns: {missing}")
            self.is_validated = False
            return False

        # Check OHLC relationships
        df = self.prices.copy()
        df.columns = df.columns.str.lower()

        invalid_high = (df['high'] < df['open']) | (df['high'] < df['close'])
        invalid_low = (df['low'] > df['open']) | (df['low'] > df['close'])

        if invalid_high.any():
            count = invalid_high.sum()
            self.validation_errors.append(f"{count} bars with high < open/close")

        if invalid_low.any():
            count = invalid_low.sum()
            self.validation_errors.append(f"{count} bars with low > open/close")

        # Check for NaN values
        if df[required].isna().any().any():
            nan_counts = df[required].isna().sum()
            self.validation_warnings.append(f"NaN values found: {nan_counts.to_dict()}")

        # Check for zero prices
        zero_prices = (df[required] == 0).any(axis=1)
        if zero_prices.any():
            count = zero_prices.sum()
            self.validation_errors.append(f"{count} bars with zero prices")

        # Check time index
        if not isinstance(df.index, pd.DatetimeIndex):
            self.validation_errors.append("Index is not DatetimeIndex")

        # Check for duplicates
        if df.index.duplicated().any():
            count = df.index.duplicated().sum()
            self.validation_warnings.append(f"{count} duplicate timestamps")

        # Check for out-of-order timestamps
        if not df.index.is_monotonic_increasing:
            self.validation_errors.append("Timestamps not in chronological order")

        # Set validation status
        self.is_validated = len(self.validation_errors) == 0

        return self.is_validated

    # ========== Utility Methods ==========

    def summary(self) -> str:
        """
        Generate summary report of the data package.

        Returns:
            Multi-line string summary
        """
        lines = [
            f"DataPackage Summary",
            f"{'='*50}",
            f"Symbol:        {self.symbol}",
            f"Timeframe:     {self.timeframe}",
            f"Market Type:   {self.market_type.value if self.market_type else 'Unknown'}",
            f"",
            f"Time Range:    {self.start_date} → {self.end_date}",
            f"Total Bars:    {self.total_bars:,}",
            f"",
            f"Data Shape:    {self.prices.shape}",
            f"Columns:       {list(self.prices.columns)}",
            f"",
            f"Validation:    {'✓ PASSED' if self.is_validated else '✗ FAILED'}",
        ]

        if self.validation_warnings:
            lines.append(f"Warnings:      {len(self.validation_warnings)}")
            for w in self.validation_warnings:
                lines.append(f"  - {w}")

        if self.validation_errors:
            lines.append(f"Errors:        {len(self.validation_errors)}")
            for e in self.validation_errors:
                lines.append(f"  - {e}")

        if self.physics_state is not None:
            lines.append(f"")
            lines.append(f"Physics State: {self.physics_state.shape}")

        if self.symbol_spec:
            lines.append(f"")
            lines.append(f"Symbol Spec:   ✓ Loaded")
            lines.append(f"  Spread:      {self.symbol_spec.spread_typical} points")
            lines.append(f"  Commission:  ${self.symbol_spec.commission_per_lot}/lot")

        if self.preprocessing_applied:
            lines.append(f"")
            lines.append(f"Preprocessing: {len(self.preprocessing_applied)} steps")
            for step in self.preprocessing_applied:
                lines.append(f"  - {step}")

        return "\n".join(lines)

    def __repr__(self) -> str:
        """String representation."""
        status = "✓" if self.is_validated else "✗"
        return (
            f"DataPackage({self.symbol} {self.timeframe}, "
            f"{self.total_bars:,} bars, "
            f"{self.market_type.value if self.market_type else 'unknown'}, "
            f"{status})"
        )
