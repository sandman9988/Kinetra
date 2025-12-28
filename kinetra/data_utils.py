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
    date_column: str = None,
    date_format: str = None
) -> pd.DataFrame:
    """
    Load OHLCV data from MT5 CSV export.

    Auto-detects:
    - Separator (tab, comma, semicolon)
    - Column names (various MT5 export formats)
    - Date format

    Args:
        filepath: Path to MT5 exported CSV file
        date_column: Name of datetime column (auto-detected if None)
        date_format: Datetime format string (auto-detected if None)

    Returns:
        Validated OHLCV DataFrame
    """
    # Try different separators
    df = None
    for sep in ['\t', ',', ';']:
        try:
            df = pd.read_csv(filepath, sep=sep)
            if len(df.columns) >= 4:  # At least OHLC
                break
        except Exception:
            continue

    if df is None or len(df.columns) < 4:
        raise ValueError(f"Could not parse CSV file: {filepath}")

    # Clean column names - remove angle brackets and lowercase
    def clean_col(c):
        c = str(c).strip().lower()
        # Remove angle brackets: <date> -> date
        if c.startswith('<') and c.endswith('>'):
            c = c[1:-1]
        return c

    df.columns = [clean_col(c) for c in df.columns]

    # Standardize column names to Backtesting.py format
    column_map = {
        'open': 'Open',
        'high': 'High',
        'low': 'Low',
        'close': 'Close',
        'tick_volume': 'Volume',
        'tickvol': 'Volume',
        'real_volume': 'Volume',
        'volume': 'Volume',
        'vol': 'Volume',
        'date': 'Date',
        'time': 'Time',
        'datetime': 'DateTime',
    }

    df.columns = [column_map.get(c, c) for c in df.columns]

    # Try to find and parse datetime index
    datetime_cols = ['DateTime', 'Date', 'Time']
    date_col = None
    time_col = None

    for col in datetime_cols:
        if col in df.columns:
            if col == 'DateTime':
                date_col = col
                break
            elif col == 'Date':
                date_col = col
            elif col == 'Time' and date_col:
                time_col = col

    # Combine Date and Time if both exist
    if date_col and time_col and time_col in df.columns:
        df['DateTime'] = df[date_col].astype(str) + ' ' + df[time_col].astype(str)
        date_col = 'DateTime'

    # Parse datetime with multiple format attempts
    if date_col and date_col in df.columns:
        date_formats = [
            '%Y.%m.%d %H:%M:%S',
            '%Y.%m.%d %H:%M',
            '%Y-%m-%d %H:%M:%S',
            '%Y-%m-%d %H:%M',
            '%d.%m.%Y %H:%M:%S',
            '%d.%m.%Y %H:%M',
            '%Y/%m/%d %H:%M:%S',
            '%Y/%m/%d %H:%M',
            None,  # Let pandas infer
        ]

        for fmt in date_formats:
            try:
                if fmt:
                    df[date_col] = pd.to_datetime(df[date_col], format=fmt)
                else:
                    df[date_col] = pd.to_datetime(df[date_col])
                df = df.set_index(date_col)
                break
            except Exception:
                continue

    # If no datetime index, use numeric index
    if not isinstance(df.index, pd.DatetimeIndex):
        df = df.reset_index(drop=True)

    # Ensure required columns exist
    required = ['Open', 'High', 'Low', 'Close']
    missing = [c for c in required if c not in df.columns]
    if missing:
        # Try to find columns by position if names don't match
        # Standard order: Date, Time, Open, High, Low, Close, Volume
        if len(df.columns) >= 6:
            # Assume positional columns after date/time
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if len(numeric_cols) >= 4:
                df = df.rename(columns={
                    numeric_cols[0]: 'Open',
                    numeric_cols[1]: 'High',
                    numeric_cols[2]: 'Low',
                    numeric_cols[3]: 'Close',
                })
                if len(numeric_cols) >= 5:
                    df = df.rename(columns={numeric_cols[4]: 'Volume'})

        # Check again
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}. Found: {list(df.columns)}")

    # Add Volume if not present
    if 'Volume' not in df.columns:
        df['Volume'] = 0

    # Select only needed columns
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()

    # Convert to numeric (in case of string data)
    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Drop rows with NaN
    df = df.dropna()

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
