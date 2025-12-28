#!/usr/bin/env python3
"""
MetaAPI Market Data Downloader

Downloads market data via MetaAPI cloud service.
Works on Linux/Mac/Windows - no need for MT5 Python package.

Setup:
1. Create account at https://app.metaapi.cloud/
2. Add your MT5 account (broker credentials)
3. Get your API token
4. Set environment variables:
   export METAAPI_TOKEN="your-token"
   export METAAPI_ACCOUNT_ID="your-account-id"

Usage:
    python3 scripts/download_metaapi.py
"""

import os
import sys
import asyncio
import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import pandas as pd
import numpy as np

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from metaapi_cloud_sdk import MetaApi
    METAAPI_AVAILABLE = True
except ImportError:
    METAAPI_AVAILABLE = False
    print("MetaAPI not installed. Run: pip install metaapi-cloud-sdk")


# Configuration
TIMEFRAMES = ['15m', '30m', '1h', '4h']
MIN_BARS = 500
DAYS_TO_DOWNLOAD = 365


class MetaAPIDownloader:
    """Downloads market data via MetaAPI."""

    def __init__(self, token: str, account_id: str):
        self.token = token
        self.account_id = account_id
        self.api = None
        self.account = None
        self.connection = None

    async def connect(self) -> bool:
        """Connect to MetaAPI."""
        try:
            self.api = MetaApi(self.token)
            self.account = await self.api.metatrader_account_api.get_account(self.account_id)

            # Wait for deployment
            if self.account.state != 'DEPLOYED':
                print("Deploying account...")
                await self.account.deploy()

            # Wait for connection
            if self.account.connection_status != 'CONNECTED':
                print("Waiting for connection...")
                await self.account.wait_connected()

            # Get RPC connection for history
            self.connection = self.account.get_rpc_connection()
            await self.connection.connect()
            await self.connection.wait_synchronized()

            print(f"Connected to account: {self.account.name}")
            return True

        except Exception as e:
            print(f"Connection failed: {e}")
            return False

    async def get_symbols(self) -> List[str]:
        """Get available symbols."""
        try:
            symbols = await self.connection.get_symbols()
            # Filter for forex, metals, indices, crypto
            tradeable = [s['symbol'] for s in symbols if s.get('tradeMode') != 'DISABLED']
            print(f"Found {len(tradeable)} tradeable symbols")
            return sorted(tradeable)
        except Exception as e:
            print(f"Failed to get symbols: {e}")
            return []

    async def get_symbol_info(self, symbol: str) -> Optional[Dict]:
        """Get symbol specification."""
        try:
            spec = await self.connection.get_symbol_specification(symbol)
            price = await self.connection.get_symbol_price(symbol)

            # Calculate friction metrics
            spread = spec.get('spread', 0)
            point = spec.get('point', 0.00001)
            bid = price.get('bid', 1)

            spread_pct = (spread * point / bid) * 100 if bid > 0 else 0
            swap_long = spec.get('swapLong', 0)
            swap_short = spec.get('swapShort', 0)

            # Friction score
            spread_friction = min(1.0, spread_pct / 0.1)
            swap_friction = min(1.0, max(abs(swap_long), abs(swap_short)) / 50)
            friction_score = 0.6 * spread_friction + 0.4 * swap_friction

            return {
                'symbol': symbol,
                'description': spec.get('description', ''),
                'digits': spec.get('digits', 5),
                'point': point,
                'spread': spread,
                'spread_pct': round(spread_pct, 4),
                'contract_size': spec.get('contractSize', 100000),
                'swap_long': swap_long,
                'swap_short': swap_short,
                'volume_min': spec.get('minVolume', 0.01),
                'volume_max': spec.get('maxVolume', 100),
                'friction_score': round(friction_score, 4),
            }
        except Exception as e:
            print(f"Failed to get info for {symbol}: {e}")
            return None

    async def download_candles(
        self,
        symbol: str,
        timeframe: str,
        days: int = DAYS_TO_DOWNLOAD
    ) -> Optional[pd.DataFrame]:
        """Download historical candles."""
        try:
            end_time = datetime.utcnow()
            start_time = end_time - timedelta(days=days)

            candles = await self.connection.get_historical_candles(
                symbol=symbol,
                timeframe=timeframe,
                start_time=start_time,
                limit=50000  # Max allowed
            )

            if not candles or len(candles) < MIN_BARS:
                return None

            # Convert to DataFrame
            df = pd.DataFrame(candles)
            df['time'] = pd.to_datetime(df['time'])
            df.set_index('time', inplace=True)

            # Rename columns
            df = df.rename(columns={
                'open': 'open',
                'high': 'high',
                'low': 'low',
                'close': 'close',
                'tickVolume': 'volume',
                'spread': 'spread'
            })

            # Keep only needed columns
            cols = ['open', 'high', 'low', 'close', 'volume']
            if 'spread' in df.columns:
                cols.append('spread')
            df = df[[c for c in cols if c in df.columns]]

            return df

        except Exception as e:
            print(f"Failed to download {symbol} {timeframe}: {e}")
            return None

    async def close(self):
        """Close connection."""
        if self.connection:
            await self.connection.close()


def validate_and_normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and normalize data."""
    df = df.copy()

    # Fill NaN
    df = df.ffill().bfill()

    # Ensure positive prices
    for col in ['open', 'high', 'low', 'close']:
        df[col] = df[col].clip(lower=0.00001)

    # Ensure volume >= 0
    df['volume'] = df['volume'].clip(lower=0)

    # Add computed features
    df['returns'] = df['close'].pct_change()
    df['range_pct'] = (df['high'] - df['low']) / df['close'] * 100

    # Fill any remaining NaN
    df = df.ffill().bfill()

    return df


async def main():
    import argparse

    parser = argparse.ArgumentParser(description="Download data via MetaAPI")
    parser.add_argument('--output', type=str, default='data/master',
                       help='Output directory')
    parser.add_argument('--days', type=int, default=365,
                       help='Days of history')
    parser.add_argument('--symbols', type=str, default=None,
                       help='Comma-separated symbols (default: all)')
    parser.add_argument('--clean', action='store_true',
                       help='Remove existing data')
    args = parser.parse_args()

    # Get credentials
    token = os.environ.get('METAAPI_TOKEN')
    account_id = os.environ.get('METAAPI_ACCOUNT_ID')

    if not token or not account_id:
        print("""
MetaAPI credentials not found!

Setup:
1. Go to https://app.metaapi.cloud/
2. Create account and add your MT5 broker credentials
3. Get your API token and account ID
4. Set environment variables:

   export METAAPI_TOKEN="your-token-here"
   export METAAPI_ACCOUNT_ID="your-account-id-here"

Then run this script again.
""")
        return

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.clean:
        for f in output_dir.glob("*.csv"):
            f.unlink()

    # Connect
    downloader = MetaAPIDownloader(token, account_id)
    if not await downloader.connect():
        return

    # Get symbols
    if args.symbols:
        symbols = [s.strip() for s in args.symbols.split(',')]
    else:
        symbols = await downloader.get_symbols()
        # Limit to common forex/metals/indices for now
        priority = ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCAD', 'USDCHF',
                   'XAUUSD', 'XAGUSD', 'BTCUSD', 'ETHUSD',
                   'US500', 'US30', 'GER40', 'UK100']
        symbols = [s for s in priority if s in symbols] + \
                  [s for s in symbols if s not in priority]

    print(f"\nDownloading {len(symbols)} symbols x {len(TIMEFRAMES)} timeframes...")
    print("-" * 60)

    # Download symbol info
    symbol_info = []

    for i, symbol in enumerate(symbols[:50]):  # Limit to 50 for free tier
        print(f"\n[{i+1}/{min(len(symbols), 50)}] {symbol}")

        info = await downloader.get_symbol_info(symbol)
        if info:
            symbol_info.append(info)

        for tf in TIMEFRAMES:
            df = await downloader.download_candles(symbol, tf, args.days)

            if df is None or len(df) < MIN_BARS:
                print(f"  {tf}: No data")
                continue

            # Validate and normalize
            df = validate_and_normalize(df)

            # Add friction if we have symbol info
            if info:
                df['friction_score'] = info['friction_score']
                df['spread_pct'] = info['spread_pct']

            # Save
            tf_map = {'15m': 'M15', '30m': 'M30', '1h': 'H1', '4h': 'H4'}
            tf_str = tf_map.get(tf, tf)
            filename = f"{symbol}_{tf_str}.csv"
            df.to_csv(output_dir / filename)
            print(f"  {tf}: {len(df)} bars -> {filename}")

    # Save symbol info
    if symbol_info:
        info_df = pd.DataFrame(symbol_info)
        info_df.to_csv(output_dir / 'symbol_info.csv', index=False)
        print(f"\nSymbol info saved: symbol_info.csv")

    await downloader.close()

    print("\n" + "=" * 60)
    print("DOWNLOAD COMPLETE")
    print("=" * 60)
    print(f"Output: {output_dir}")
    print(f"Files: {len(list(output_dir.glob('*.csv')))}")


if __name__ == '__main__':
    if not METAAPI_AVAILABLE:
        sys.exit(1)
    asyncio.run(main())
