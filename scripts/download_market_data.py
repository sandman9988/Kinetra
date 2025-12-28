#!/usr/bin/env python3
"""
MT5 Market Data Downloader

Downloads all active MarketWatch symbols with multiple timeframes.
Calculates friction costs from symbol_info (spread, swap, margin).
Normalizes and validates data for RL training.

Usage:
    python3 scripts/download_market_data.py
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import json
import numpy as np
import pandas as pd

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    MT5_AVAILABLE = False
    print("MetaTrader5 not installed. Run: pip install MetaTrader5")


# Timeframes to download
TIMEFRAMES = {
    'M15': mt5.TIMEFRAME_M15 if MT5_AVAILABLE else 15,
    'M30': mt5.TIMEFRAME_M30 if MT5_AVAILABLE else 30,
    'H1': mt5.TIMEFRAME_H1 if MT5_AVAILABLE else 60,
    'H4': mt5.TIMEFRAME_H4 if MT5_AVAILABLE else 240,
}

# Minimum bars required for training
MIN_BARS = 1000


class SymbolInfo:
    """Extracted symbol information for friction calculations."""

    def __init__(self, info):
        self.name = info.name
        self.description = info.description
        self.path = info.path

        # Price info
        self.digits = info.digits
        self.point = info.point
        self.bid = info.bid
        self.ask = info.ask

        # Spread
        self.spread = info.spread  # In points
        self.spread_float = info.spread_float

        # Contract
        self.trade_contract_size = info.trade_contract_size
        self.volume_min = info.volume_min
        self.volume_max = info.volume_max
        self.volume_step = info.volume_step

        # Margin
        self.margin_initial = info.margin_initial
        self.margin_maintenance = info.margin_maintenance
        self.margin_currency = getattr(info, 'currency_margin', 'USD')

        # Swap (overnight costs)
        self.swap_long = info.swap_long
        self.swap_short = info.swap_short
        self.swap_mode = info.swap_mode

        # Trading
        self.trade_mode = info.trade_mode
        self.visible = info.visible

    def spread_pct(self) -> float:
        """Spread as percentage of price."""
        mid_price = (self.bid + self.ask) / 2
        if mid_price > 0:
            return (self.spread * self.point / mid_price) * 100
        return 0

    def friction_score(self) -> float:
        """
        Calculate friction score (0-1) based on trading costs.
        Higher = more friction = more expensive to trade.
        """
        # Spread friction (main cost)
        spread_pct = self.spread_pct()
        spread_friction = min(1.0, spread_pct / 0.1)  # 0.1% = high friction

        # Swap friction (overnight cost magnitude)
        max_swap = max(abs(self.swap_long), abs(self.swap_short))
        swap_friction = min(1.0, max_swap / 50)  # 50 points = high swap

        # Volume friction (tight constraints = high friction)
        volume_ratio = self.volume_min / max(self.volume_max, 1)
        volume_friction = min(1.0, volume_ratio * 100)

        # Dimensionless aggregate: root-mean-square of component frictions
        friction_squared_mean = (
            spread_friction ** 2 +
            swap_friction ** 2 +
            volume_friction ** 2
        ) / 3.0
        friction = min(1.0, friction_squared_mean ** 0.5)

        return round(friction, 4)

    def to_dict(self) -> Dict:
        """Convert to dictionary for storage."""
        return {
            'name': self.name,
            'description': self.description,
            'digits': self.digits,
            'point': self.point,
            'spread': self.spread,
            'spread_pct': self.spread_pct(),
            'contract_size': self.trade_contract_size,
            'volume_min': self.volume_min,
            'volume_max': self.volume_max,
            'swap_long': self.swap_long,
            'swap_short': self.swap_short,
            'friction_score': self.friction_score(),
        }


def connect_mt5() -> bool:
    """Initialize MT5 connection."""
    if not MT5_AVAILABLE:
        print("MT5 not available")
        return False

    if not mt5.initialize():
        print(f"MT5 initialization failed: {mt5.last_error()}")
        return False

    print(f"MT5 connected: {mt5.terminal_info().name}")
    print(f"Account: {mt5.account_info().login}")
    return True


def get_marketwatch_symbols() -> List[str]:
    """Get all symbols visible in MarketWatch."""
    symbols = mt5.symbols_get()
    if symbols is None:
        return []

    # Filter for visible (in MarketWatch) and tradeable symbols
    active = [
        s.name for s in symbols
        if s.visible and s.trade_mode != 0
    ]

    print(f"Found {len(active)} active symbols in MarketWatch")
    return sorted(active)


def get_symbol_info(symbol: str) -> Optional[SymbolInfo]:
    """Get detailed symbol information."""
    info = mt5.symbol_info(symbol)
    if info is None:
        return None
    return SymbolInfo(info)


def download_symbol_data(
    symbol: str,
    timeframe_name: str,
    timeframe: int,
    days: int = 365
) -> Optional[pd.DataFrame]:
    """Download OHLCV data for a symbol."""

    # Calculate date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)

    # Download rates
    rates = mt5.copy_rates_range(symbol, timeframe, start_date, end_date)

    if rates is None or len(rates) < MIN_BARS:
        return None

    # Convert to DataFrame
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.set_index('time', inplace=True)

    # Rename columns to standard format
    df = df.rename(columns={
        'open': 'open',
        'high': 'high',
        'low': 'low',
        'close': 'close',
        'tick_volume': 'volume',
        'spread': 'spread',
        'real_volume': 'real_volume'
    })

    # Keep only needed columns
    cols = ['open', 'high', 'low', 'close', 'volume']
    if 'spread' in df.columns:
        cols.append('spread')
    df = df[cols]

    return df


def validate_data(df: pd.DataFrame, symbol: str) -> Tuple[bool, List[str]]:
    """
    Validate data quality for training.
    Returns (is_valid, list of issues).
    """
    issues = []

    # Check for NaN values
    nan_count = df.isna().sum().sum()
    if nan_count > 0:
        nan_pct = nan_count / df.size * 100
        if nan_pct > 5:
            issues.append(f"High NaN rate: {nan_pct:.1f}%")

    # Check for zero prices
    zero_prices = (df[['open', 'high', 'low', 'close']] == 0).sum().sum()
    if zero_prices > 0:
        issues.append(f"Zero prices found: {zero_prices}")

    # Check for negative prices
    neg_prices = (df[['open', 'high', 'low', 'close']] < 0).sum().sum()
    if neg_prices > 0:
        issues.append(f"Negative prices found: {neg_prices}")

    # Check OHLC consistency (high >= low, etc.)
    invalid_hlc = (df['high'] < df['low']).sum()
    if invalid_hlc > 0:
        issues.append(f"Invalid H/L: {invalid_hlc}")

    # Check for gaps (more than 10x normal gap)
    if len(df) > 1:
        time_diffs = df.index.to_series().diff().dropna()
        median_gap = time_diffs.median()
        large_gaps = (time_diffs > median_gap * 10).sum()
        if large_gaps > len(df) * 0.01:  # More than 1% large gaps
            issues.append(f"Large gaps: {large_gaps}")

    # Check minimum bars
    if len(df) < MIN_BARS:
        issues.append(f"Insufficient bars: {len(df)} < {MIN_BARS}")

    is_valid = len([i for i in issues if 'Insufficient' in i or 'Zero' in i or 'Negative' in i]) == 0

    return is_valid, issues


def normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize OHLCV data for training.
    - Fill NaN values
    - Ensure proper dtypes
    - Add normalized columns
    """
    df = df.copy()

    # Forward fill then backward fill NaN
    df = df.ffill().bfill()

    # Ensure float64
    for col in ['open', 'high', 'low', 'close']:
        df[col] = df[col].astype(np.float64)

    # Ensure volume is positive
    df['volume'] = df['volume'].clip(lower=0).astype(np.float64)

    # Add returns (normalized price change)
    df['returns'] = df['close'].pct_change()

    # Add normalized range
    df['range_pct'] = (df['high'] - df['low']) / df['close'] * 100

    # Add body ratio (how much of range is body)
    body = abs(df['close'] - df['open'])
    wick = df['high'] - df['low']
    df['body_ratio'] = (body / wick.clip(lower=1e-10)).clip(0, 1)

    # Fill any remaining NaN in new columns
    df = df.ffill().bfill()

    return df


def add_friction_columns(df: pd.DataFrame, symbol_info: SymbolInfo) -> pd.DataFrame:
    """Add friction-related columns from symbol info."""
    df = df.copy()

    # Static friction from symbol info
    df['friction_score'] = symbol_info.friction_score()
    df['spread_pct'] = symbol_info.spread_pct()
    df['swap_long'] = symbol_info.swap_long
    df['swap_short'] = symbol_info.swap_short
    df['contract_size'] = symbol_info.trade_contract_size

    # Dynamic friction (if spread column exists)
    if 'spread' in df.columns:
        # Convert spread points to percentage
        df['spread_dynamic_pct'] = (df['spread'] * symbol_info.point / df['close']) * 100
        # Use dynamic spread for friction if available
        df['friction_dynamic'] = df['spread_dynamic_pct'] / 0.1  # Normalize
        df['friction_dynamic'] = df['friction_dynamic'].clip(0, 1)

    return df


def download_all_data(
    output_dir: Path,
    timeframes: Dict[str, int] = TIMEFRAMES,
    days: int = 365,
    max_symbols: int = None
) -> Dict:
    """
    Download data for all MarketWatch symbols.

    Returns summary of downloaded data.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Connect to MT5
    if not connect_mt5():
        return {'error': 'Failed to connect to MT5'}

    # Get symbols
    symbols = get_marketwatch_symbols()
    if max_symbols:
        symbols = symbols[:max_symbols]

    print(f"\nDownloading {len(symbols)} symbols x {len(timeframes)} timeframes...")
    print(f"Output: {output_dir}")
    print("-" * 60)

    summary = {
        'timestamp': datetime.now().isoformat(),
        'symbols_total': len(symbols),
        'timeframes': list(timeframes.keys()),
        'days': days,
        'downloaded': [],
        'skipped': [],
        'symbol_info': {},
    }

    for i, symbol in enumerate(symbols):
        print(f"\n[{i+1}/{len(symbols)}] {symbol}")

        # Get symbol info
        info = get_symbol_info(symbol)
        if info is None:
            print(f"  ! Could not get symbol info")
            summary['skipped'].append({'symbol': symbol, 'reason': 'No info'})
            continue

        summary['symbol_info'][symbol] = info.to_dict()

        for tf_name, tf_value in timeframes.items():
            # Download data
            df = download_symbol_data(symbol, tf_name, tf_value, days)

            if df is None:
                print(f"  {tf_name}: No data")
                continue

            # Validate
            is_valid, issues = validate_data(df, symbol)
            if not is_valid:
                print(f"  {tf_name}: Invalid - {', '.join(issues)}")
                summary['skipped'].append({
                    'symbol': symbol,
                    'timeframe': tf_name,
                    'reason': issues
                })
                continue

            # Normalize
            df = normalize_ohlcv(df)

            # Add friction
            df = add_friction_columns(df, info)

            # Generate filename
            start_str = df.index[0].strftime('%Y%m%d%H%M')
            end_str = df.index[-1].strftime('%Y%m%d%H%M')
            filename = f"{symbol}_{tf_name}_{start_str}_{end_str}.csv"
            filepath = output_dir / filename

            # Save
            df.to_csv(filepath)

            print(f"  {tf_name}: {len(df)} bars -> {filename}")
            summary['downloaded'].append({
                'symbol': symbol,
                'timeframe': tf_name,
                'bars': len(df),
                'file': filename,
                'friction': info.friction_score(),
            })

    # Save summary
    summary_file = output_dir / 'download_summary.json'
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    # Disconnect
    mt5.shutdown()

    # Print summary
    print("\n" + "=" * 60)
    print("DOWNLOAD COMPLETE")
    print("=" * 60)
    print(f"Downloaded: {len(summary['downloaded'])} files")
    print(f"Skipped: {len(summary['skipped'])} files")
    print(f"Summary: {summary_file}")

    return summary


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Download MT5 market data")
    parser.add_argument('--output', type=str, default='data/master',
                       help='Output directory')
    parser.add_argument('--days', type=int, default=365,
                       help='Days of history to download')
    parser.add_argument('--max-symbols', type=int, default=None,
                       help='Limit number of symbols (for testing)')
    parser.add_argument('--clean', action='store_true',
                       help='Remove existing data before download')
    args = parser.parse_args()

    output_dir = Path(args.output)

    if args.clean and output_dir.exists():
        import shutil
        print(f"Cleaning {output_dir}...")
        for f in output_dir.glob("*.csv"):
            f.unlink()

    summary = download_all_data(
        output_dir=output_dir,
        days=args.days,
        max_symbols=args.max_symbols,
    )

    if 'error' in summary:
        print(f"Error: {summary['error']}")
        sys.exit(1)


if __name__ == '__main__':
    main()
