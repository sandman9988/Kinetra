"""
Data Utilities for Physics Backtesting

Loads and validates real MT5 market data for empirical testing.
NO synthetic or simulated data - real market data only.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Tuple, List
from datetime import datetime


def load_mt5_csv(
    filepath: str,
    date_column: str = 'time',
    date_format: str = '%Y.%m.%d %H:%M:%S'
) -> pd.DataFrame:
    """
    Load OHLCV data from MT5 CSV export.

    MT5 exports data in format:
    time,open,high,low,close,tick_volume,spread,real_volume

    Args:
        filepath: Path to MT5 exported CSV file
        date_column: Name of datetime column (default: 'time')
        date_format: Datetime format string (MT5 default: '%Y.%m.%d %H:%M:%S')

    Returns:
        Validated OHLCV DataFrame
    """
    df = pd.read_csv(filepath, sep='\t')

    # Handle different MT5 export formats
    if date_column in df.columns:
        df[date_column] = pd.to_datetime(df[date_column], format=date_format)
        df = df.set_index(date_column)

    # Standardize column names to Backtesting.py format
    column_map = {
        'open': 'Open',
        'high': 'High',
        'low': 'Low',
        'close': 'Close',
        'tick_volume': 'Volume',
        'real_volume': 'Volume',
        'volume': 'Volume',
        '<open>': 'Open',
        '<high>': 'High',
        '<low>': 'Low',
        '<close>': 'Close',
        '<tickvol>': 'Volume',
        '<vol>': 'Volume',
        '<date>': 'Date',
        '<time>': 'Time',
    }

    df.columns = [column_map.get(c.lower().strip(), c) for c in df.columns]

    # Ensure required columns exist
    required = ['Open', 'High', 'Low', 'Close']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Add Volume if not present
    if 'Volume' not in df.columns:
        df['Volume'] = 0

    # Select only needed columns
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()

    # Validate data integrity
    is_valid, message = validate_ohlcv(df)
    if not is_valid:
        raise ValueError(f"Invalid OHLCV data: {message}")

    return df


def load_mt5_history(
    filepath: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
) -> pd.DataFrame:
    """
    Load MT5 history file with optional date filtering.

    Args:
        filepath: Path to MT5 CSV export
        start_date: Filter start date (YYYY-MM-DD)
        end_date: Filter end date (YYYY-MM-DD)

    Returns:
        Filtered OHLCV DataFrame
    """
    df = load_mt5_csv(filepath)

    if start_date:
        df = df[df.index >= start_date]
    if end_date:
        df = df[df.index <= end_date]

    if len(df) == 0:
        raise ValueError("No data in specified date range")

    return df


def load_multiple_mt5_files(
    filepaths: List[str],
    sort_by_date: bool = True
) -> pd.DataFrame:
    """
    Load and concatenate multiple MT5 CSV files.

    Useful for combining different time periods.

    Args:
        filepaths: List of file paths
        sort_by_date: Whether to sort by date after concatenation

    Returns:
        Combined OHLCV DataFrame
    """
    dfs = []
    for filepath in filepaths:
        df = load_mt5_csv(filepath)
        dfs.append(df)

    combined = pd.concat(dfs)

    if sort_by_date:
        combined = combined.sort_index()

    # Remove duplicates
    combined = combined[~combined.index.duplicated(keep='first')]

    return combined


def validate_ohlcv(df: pd.DataFrame) -> Tuple[bool, str]:
    """
    Validate OHLCV data integrity.

    Checks:
    - Required columns present
    - No null values
    - High >= max(Open, Close)
    - Low <= min(Open, Close)
    - All prices positive

    Args:
        df: DataFrame to validate

    Returns:
        Tuple of (is_valid, error_message)
    """
    required_columns = ['Open', 'High', 'Low', 'Close']

    # Check columns exist
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        return False, f"Missing columns: {missing}"

    # Check for nulls
    null_counts = df[required_columns].isnull().sum()
    if null_counts.sum() > 0:
        return False, f"Null values found: {null_counts.to_dict()}"

    # Check OHLC relationships
    invalid_high = (df['High'] < df[['Open', 'Close']].max(axis=1)).sum()
    if invalid_high > 0:
        return False, f"Invalid High values: {invalid_high} bars where High < max(Open, Close)"

    invalid_low = (df['Low'] > df[['Open', 'Close']].min(axis=1)).sum()
    if invalid_low > 0:
        return False, f"Invalid Low values: {invalid_low} bars where Low > min(Open, Close)"

    # Check for non-positive prices
    non_positive = (df[required_columns] <= 0).sum().sum()
    if non_positive > 0:
        return False, f"Non-positive prices found: {non_positive}"

    return True, "Data validation passed"


def preprocess_mt5_data(
    df: pd.DataFrame,
    fill_gaps: bool = True,
    remove_weekends: bool = True
) -> pd.DataFrame:
    """
    Preprocess MT5 data for backtesting.

    Args:
        df: Raw MT5 OHLCV DataFrame
        fill_gaps: Forward-fill small gaps in data
        remove_weekends: Remove weekend bars (for forex)

    Returns:
        Preprocessed DataFrame
    """
    df = df.copy()

    # Remove weekend data if requested (forex markets)
    if remove_weekends and isinstance(df.index, pd.DatetimeIndex):
        df = df[df.index.dayofweek < 5]

    # Fill small gaps (up to 4 bars)
    if fill_gaps:
        df = df.ffill(limit=4)

    # Drop any remaining NaN rows
    df = df.dropna()

    return df


def get_data_summary(df: pd.DataFrame) -> dict:
    """
    Get summary statistics of the loaded data.

    Args:
        df: OHLCV DataFrame

    Returns:
        Dict with data summary
    """
    returns = df['Close'].pct_change().dropna()

    summary = {
        'start_date': str(df.index[0]) if hasattr(df.index[0], 'strftime') else str(df.index[0]),
        'end_date': str(df.index[-1]) if hasattr(df.index[-1], 'strftime') else str(df.index[-1]),
        'total_bars': len(df),
        'price_start': float(df['Close'].iloc[0]),
        'price_end': float(df['Close'].iloc[-1]),
        'price_high': float(df['High'].max()),
        'price_low': float(df['Low'].min()),
        'total_return_pct': float((df['Close'].iloc[-1] / df['Close'].iloc[0] - 1) * 100),
        'annualized_volatility': float(returns.std() * np.sqrt(252 * 24)),  # Assuming hourly data
        'mean_volume': float(df['Volume'].mean()) if df['Volume'].sum() > 0 else 0,
    }

    return summary


def split_data(
    df: pd.DataFrame,
    train_ratio: float = 0.7,
    validation_ratio: float = 0.15
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split data into train/validation/test sets chronologically.

    NO random shuffling - maintains temporal order.

    Args:
        df: OHLCV DataFrame
        train_ratio: Proportion for training (default: 70%)
        validation_ratio: Proportion for validation (default: 15%)

    Returns:
        Tuple of (train_df, validation_df, test_df)
    """
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + validation_ratio))

    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()

    return train_df, val_df, test_df


def create_walk_forward_windows(
    df: pd.DataFrame,
    train_bars: int = 500,
    test_bars: int = 100,
    step_bars: int = 50
) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Create walk-forward analysis windows.

    Each window consists of:
    - Training period: train_bars of historical data
    - Testing period: test_bars of out-of-sample data

    Args:
        df: OHLCV DataFrame
        train_bars: Number of bars for training
        test_bars: Number of bars for testing
        step_bars: Number of bars to step forward each window

    Returns:
        List of (train_df, test_df) tuples
    """
    windows = []
    start = 0

    while start + train_bars + test_bars <= len(df):
        train_df = df.iloc[start:start + train_bars].copy()
        test_df = df.iloc[start + train_bars:start + train_bars + test_bars].copy()
        windows.append((train_df, test_df))
        start += step_bars

    return windows


def export_backtest_data(
    df: pd.DataFrame,
    filepath: str,
    format: str = 'csv'
) -> None:
    """
    Export processed data for external use.

    Args:
        df: OHLCV DataFrame
        filepath: Output file path
        format: Output format ('csv' or 'parquet')
    """
    if format == 'csv':
        df.to_csv(filepath)
    elif format == 'parquet':
        df.to_parquet(filepath)
    else:
        raise ValueError(f"Unknown format: {format}")


def find_mt5_data_files(directory: str, pattern: str = '*.csv') -> List[str]:
    """
    Find MT5 data files in a directory.

    Args:
        directory: Directory to search
        pattern: File pattern (default: *.csv)

    Returns:
        List of file paths
    """
    path = Path(directory)
    files = list(path.glob(pattern))
    return sorted([str(f) for f in files])
