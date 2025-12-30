#!/usr/bin/env python3
"""
MetaAPI MT5 Data Manager - Download & Sync Historical Market Data

Features:
- Initial download (2+ years of OHLC candles)
- Incremental sync (extend existing database daily/hourly)
- Metadata tracking (last sync timestamp)
- Partial candle handling (refreshes latest 2 candles)
- Retry logic with exponential backoff
- Pure Python (no web server needed)

Setup:
1. Sign up at https://app.metaapi.cloud/ (from residential IP)
2. Add your MT5 account (demo or live)
3. Get API token and account ID from dashboard
4. Set environment variables:
   export METAAPI_TOKEN="your-token-here"
   export METAAPI_ACCOUNT_ID="your-account-id-here"

Usage:
    # Initial download (2 years)
    python3 scripts/mt5_metaapi_sync.py --init --symbol EURUSD --timeframe H1 --years 2

    # Daily sync (extend with new data)
    python3 scripts/mt5_metaapi_sync.py --sync --symbol EURUSD --timeframe H1

    # Sync all symbols
    python3 scripts/mt5_metaapi_sync.py --sync-all

Automation (cron):
    # Daily at 2 AM UTC
    0 2 * * * cd /path/to/Kinetra && python3 scripts/mt5_metaapi_sync.py --sync-all
"""

import os
import sys
import asyncio
import json
import time
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import argparse

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np

try:
    from metaapi_cloud_sdk import MetaApi
    METAAPI_AVAILABLE = True
except ImportError:
    METAAPI_AVAILABLE = False
    print("❌ MetaAPI not installed. Run: pip install metaapi-cloud-sdk")
    sys.exit(1)


# Configuration
DEFAULT_OUTPUT_DIR = Path("data/metaapi")
DEFAULT_TIMEFRAME = "H1"
DEFAULT_SYMBOL = "EURUSD"
DEFAULT_YEARS = 2
MIN_BARS = 500
MAX_RETRIES = 4
RETRY_DELAYS = [2, 4, 8, 16]  # Exponential backoff in seconds

# Timeframe conversions
TIMEFRAME_MAP = {
    'M1': 1, 'M5': 5, 'M15': 15, 'M30': 30,
    'H1': 60, 'H4': 240, 'D1': 1440
}


def get_timeframe_minutes(tf: str) -> int:
    """Convert timeframe string to minutes."""
    return TIMEFRAME_MAP.get(tf, 60)


class MetaAPIDataManager:
    """Manages MetaAPI data download and synchronization."""

    def __init__(self, token: str, account_id: str, output_dir: Path = DEFAULT_OUTPUT_DIR):
        self.token = token
        self.account_id = account_id
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.api = None
        self.account = None
        self.connection = None

    async def connect(self) -> bool:
        """Connect to MetaAPI with retry logic."""
        for attempt in range(MAX_RETRIES):
            try:
                print(f"🔌 Connecting to MetaAPI (attempt {attempt + 1}/{MAX_RETRIES})...")

                self.api = MetaApi(self.token)
                self.account = await self.api.metatrader_account_api.get_account(self.account_id)

                # Deploy if needed
                if self.account.state != 'DEPLOYED':
                    print("📦 Deploying account...")
                    await self.account.deploy()
                    await asyncio.sleep(5)  # Wait for deployment

                # Wait for connection
                if self.account.connection_status != 'CONNECTED':
                    print("⏳ Waiting for connection...")
                    await self.account.wait_connected()

                # Get RPC connection
                self.connection = self.account.get_rpc_connection()
                await self.connection.connect()
                await self.connection.wait_synchronized()

                print(f"✅ Connected to: {self.account.name}")
                return True

            except Exception as e:
                print(f"❌ Connection failed: {e}")
                if attempt < MAX_RETRIES - 1:
                    delay = RETRY_DELAYS[attempt]
                    print(f"⏳ Retrying in {delay}s...")
                    await asyncio.sleep(delay)
                else:
                    print("❌ Max retries reached. Connection failed.")
                    return False

    async def download_candles_chunked(
        self,
        symbol: str,
        timeframe: str,
        start_time: datetime,
        end_time: datetime,
        max_candles_per_request: int = 10000
    ) -> List[Dict]:
        """
        Download candles in chunks with retry logic.

        MetaAPI limits: ~10,000-50,000 candles per request depending on broker.
        For 2 years of M1 data (~1M candles), we need ~20-100 requests.
        """
        all_candles = []
        current_start = start_time

        print(f"📊 Downloading {symbol} {timeframe} from {start_time.date()} to {end_time.date()}")

        while current_start < end_time:
            # Calculate chunk end time
            tf_minutes = get_timeframe_minutes(timeframe)
            chunk_duration = timedelta(minutes=tf_minutes * max_candles_per_request)
            current_end = min(current_start + chunk_duration, end_time)

            # Download chunk with retry
            for attempt in range(MAX_RETRIES):
                try:
                    candles = await self.connection.get_historical_candles(
                        symbol=symbol,
                        timeframe=timeframe,
                        start_time=current_start,
                        limit=max_candles_per_request
                    )

                    if candles:
                        # Filter candles within time range
                        chunk = [
                            c for c in candles
                            if current_start <= datetime.fromisoformat(c['time'].replace('Z', '+00:00')) < current_end
                        ]
                        all_candles.extend(chunk)
                        print(f"  ✓ Got {len(chunk)} candles ({current_start.date()} to {current_end.date()})")

                    break  # Success, move to next chunk

                except Exception as e:
                    print(f"  ❌ Chunk download failed: {e}")
                    if attempt < MAX_RETRIES - 1:
                        delay = RETRY_DELAYS[attempt]
                        print(f"  ⏳ Retrying in {delay}s...")
                        await asyncio.sleep(delay)
                    else:
                        print(f"  ❌ Failed to download chunk after {MAX_RETRIES} attempts")
                        # Continue to next chunk even if this one failed

            # Move to next chunk
            current_start = current_end
            await asyncio.sleep(0.2)  # Rate limiting (5 req/sec max)

        print(f"✅ Total candles downloaded: {len(all_candles)}")
        return all_candles

    def candles_to_dataframe(self, candles: List[Dict]) -> pd.DataFrame:
        """Convert candles to pandas DataFrame."""
        if not candles:
            return pd.DataFrame()

        df = pd.DataFrame(candles)

        # Parse time
        df['time'] = pd.to_datetime(df['time'], utc=True)
        df = df.set_index('time')

        # Rename columns
        df = df.rename(columns={
            'brokerTime': 'broker_time',
            'tickVolume': 'tick_volume'
        })

        # Keep relevant columns
        columns = ['open', 'high', 'low', 'close', 'volume', 'tick_volume']
        df = df[[c for c in columns if c in df.columns]]

        # Sort by time
        df = df.sort_index()

        # Remove duplicates (keep last)
        df = df[~df.index.duplicated(keep='last')]

        return df

    def validate_and_clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate and clean OHLCV data."""
        if df.empty:
            return df

        df = df.copy()

        # Fill NaN
        df = df.ffill().bfill()

        # Ensure positive prices
        for col in ['open', 'high', 'low', 'close']:
            if col in df.columns:
                df[col] = df[col].clip(lower=0.00001)

        # Ensure volume >= 0
        if 'volume' in df.columns:
            df['volume'] = df['volume'].clip(lower=0)

        # Add features
        df['returns'] = df['close'].pct_change()
        df['range_pct'] = (df['high'] - df['low']) / df['close'] * 100
        df['body_pct'] = abs(df['close'] - df['open']) / df['close'] * 100

        # Fill any remaining NaN
        df = df.ffill().bfill()

        return df

    def get_metadata_path(self, symbol: str, timeframe: str) -> Path:
        """Get metadata file path for symbol/timeframe."""
        return self.output_dir / f"{symbol}_{timeframe}_metadata.json"

    def get_data_path(self, symbol: str, timeframe: str) -> Path:
        """Get data file path for symbol/timeframe."""
        return self.output_dir / f"{symbol}_{timeframe}_history.csv"

    def save_metadata(self, symbol: str, timeframe: str, last_time: datetime, bars: int):
        """Save metadata (last sync timestamp)."""
        metadata = {
            'symbol': symbol,
            'timeframe': timeframe,
            'last_sync_time': last_time.isoformat(),
            'last_sync_timestamp': last_time.timestamp(),
            'total_bars': bars,
            'updated_at': datetime.utcnow().isoformat(),
        }

        metadata_path = self.get_metadata_path(symbol, timeframe)
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"💾 Metadata saved: {metadata_path.name}")

    def load_metadata(self, symbol: str, timeframe: str) -> Optional[Dict]:
        """Load metadata if exists."""
        metadata_path = self.get_metadata_path(symbol, timeframe)

        if not metadata_path.exists():
            return None

        try:
            with open(metadata_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"⚠️  Failed to load metadata: {e}")
            return None

    async def download_initial_history(
        self,
        symbol: str,
        timeframe: str,
        years: int = DEFAULT_YEARS
    ) -> bool:
        """Download initial historical data (e.g., 2 years)."""
        print(f"\n{'='*60}")
        print(f"INITIAL DOWNLOAD: {symbol} {timeframe} ({years} years)")
        print(f"{'='*60}\n")

        # Calculate date range
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(days=years * 365)

        # Download candles in chunks
        candles = await self.download_candles_chunked(
            symbol, timeframe, start_time, end_time
        )

        if not candles:
            print(f"❌ No data received for {symbol} {timeframe}")
            return False

        if len(candles) < MIN_BARS:
            print(f"⚠️  Insufficient data: {len(candles)} < {MIN_BARS} bars")
            return False

        # Convert to DataFrame
        df = self.candles_to_dataframe(candles)
        df = self.validate_and_clean(df)

        # Save data
        data_path = self.get_data_path(symbol, timeframe)
        df.to_csv(data_path)
        print(f"💾 Saved {len(df)} bars to: {data_path.name}")

        # Save metadata
        last_time = df.index[-1].to_pydatetime()
        self.save_metadata(symbol, timeframe, last_time, len(df))

        print(f"\n✅ Initial download complete!")
        print(f"   Bars: {len(df)}")
        print(f"   Period: {df.index[0].date()} to {df.index[-1].date()}")

        return True

    async def extend_history(
        self,
        symbol: str,
        timeframe: str
    ) -> bool:
        """Extend existing history with new data (incremental sync)."""
        print(f"\n{'='*60}")
        print(f"SYNC: {symbol} {timeframe}")
        print(f"{'='*60}\n")

        # Load existing data
        data_path = self.get_data_path(symbol, timeframe)
        if not data_path.exists():
            print(f"❌ No existing data found. Run --init first.")
            return False

        # Load metadata
        metadata = self.load_metadata(symbol, timeframe)
        if not metadata:
            print(f"⚠️  No metadata found. Using data file timestamp.")
            df_existing = pd.read_csv(data_path, index_col='time', parse_dates=True)
            last_time = df_existing.index[-1].to_pydatetime()
        else:
            last_time = datetime.fromisoformat(metadata['last_sync_time'])

        print(f"📅 Last sync: {last_time}")

        # CRITICAL: Refresh last 2 candles to handle partial candles
        tf_minutes = get_timeframe_minutes(timeframe)
        refresh_window = timedelta(minutes=tf_minutes * 2)
        safe_start = last_time - refresh_window

        end_time = datetime.utcnow()

        if end_time <= safe_start:
            print(f"⏭️  No new data to sync yet.")
            return True

        # Download new candles
        candles = await self.download_candles_chunked(
            symbol, timeframe, safe_start, end_time, max_candles_per_request=5000
        )

        if not candles:
            print(f"⚠️  No new candles received")
            return False

        # Convert to DataFrame
        df_new = self.candles_to_dataframe(candles)
        df_new = self.validate_and_clean(df_new)

        # Load existing data
        df_existing = pd.read_csv(data_path, index_col='time', parse_dates=True)

        # Merge (new data overwrites overlapping timestamps)
        df_combined = pd.concat([df_existing, df_new])
        df_combined = df_combined[~df_combined.index.duplicated(keep='last')]
        df_combined = df_combined.sort_index()

        new_bars = len(df_combined) - len(df_existing)

        # Save updated data
        df_combined.to_csv(data_path)
        print(f"💾 Updated: {data_path.name}")
        print(f"   Added: {new_bars} new bars")
        print(f"   Total: {len(df_combined)} bars")

        # Update metadata
        last_time_new = df_combined.index[-1].to_pydatetime()
        self.save_metadata(symbol, timeframe, last_time_new, len(df_combined))

        print(f"\n✅ Sync complete!")
        print(f"   Period: {df_combined.index[0].date()} to {df_combined.index[-1].date()}")

        return True

    async def sync_all(self, symbols: List[str], timeframes: List[str]) -> Dict:
        """Sync all configured symbols and timeframes."""
        print(f"\n{'='*60}")
        print(f"SYNC ALL: {len(symbols)} symbols × {len(timeframes)} timeframes")
        print(f"{'='*60}\n")

        results = {'success': [], 'failed': []}

        for symbol in symbols:
            for timeframe in timeframes:
                print(f"\n--- {symbol} {timeframe} ---")
                try:
                    success = await self.extend_history(symbol, timeframe)
                    if success:
                        results['success'].append(f"{symbol}_{timeframe}")
                    else:
                        results['failed'].append(f"{symbol}_{timeframe}")
                except Exception as e:
                    print(f"❌ Error: {e}")
                    results['failed'].append(f"{symbol}_{timeframe}")

                await asyncio.sleep(0.5)  # Rate limiting

        print(f"\n{'='*60}")
        print("SYNC ALL COMPLETE")
        print(f"{'='*60}")
        print(f"Success: {len(results['success'])}")
        print(f"Failed: {len(results['failed'])}")

        return results

    async def close(self):
        """Close MetaAPI connection."""
        if self.connection:
            try:
                await self.connection.close()
                print("\n🔌 Disconnected from MetaAPI")
            except:
                pass


async def main():
    parser = argparse.ArgumentParser(
        description="MetaAPI MT5 Data Manager - Download & Sync Historical Data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Initial download (2 years of EURUSD H1)
  %(prog)s --init --symbol EURUSD --timeframe H1 --years 2

  # Daily sync (extend with new data)
  %(prog)s --sync --symbol EURUSD --timeframe H1

  # Sync all configured symbols
  %(prog)s --sync-all --symbols EURUSD,GBPUSD --timeframes H1,H4

  # Use custom output directory
  %(prog)s --init --symbol BTCUSD --timeframe H1 --output data/crypto
        """
    )

    # Mode
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument('--init', action='store_true',
                           help='Initial download (e.g., 2 years)')
    mode_group.add_argument('--sync', action='store_true',
                           help='Extend existing data with new candles')
    mode_group.add_argument('--sync-all', action='store_true',
                           help='Sync all configured symbols/timeframes')

    # Data parameters
    parser.add_argument('--symbol', type=str, default=DEFAULT_SYMBOL,
                       help=f'Symbol to download (default: {DEFAULT_SYMBOL})')
    parser.add_argument('--timeframe', type=str, default=DEFAULT_TIMEFRAME,
                       choices=list(TIMEFRAME_MAP.keys()),
                       help=f'Timeframe (default: {DEFAULT_TIMEFRAME})')
    parser.add_argument('--years', type=int, default=DEFAULT_YEARS,
                       help=f'Years of history for --init (default: {DEFAULT_YEARS})')

    # Sync-all parameters
    parser.add_argument('--symbols', type=str,
                       help='Comma-separated symbols for --sync-all (e.g., EURUSD,GBPUSD)')
    parser.add_argument('--timeframes', type=str,
                       help='Comma-separated timeframes for --sync-all (e.g., H1,H4)')

    # Output
    parser.add_argument('--output', type=str, default=str(DEFAULT_OUTPUT_DIR),
                       help=f'Output directory (default: {DEFAULT_OUTPUT_DIR})')

    args = parser.parse_args()

    # Get MetaAPI credentials from environment
    token = os.environ.get('METAAPI_TOKEN')
    account_id = os.environ.get('METAAPI_ACCOUNT_ID')

    if not token or not account_id:
        print("""
❌ MetaAPI credentials not found!

Setup Instructions:
1. Sign up at https://app.metaapi.cloud/ (from residential/mobile IP)
2. Add your MT5 account (demo or live)
3. Get your API token and account ID from dashboard
4. Set environment variables:

   export METAAPI_TOKEN="your-token-here"
   export METAAPI_ACCOUNT_ID="your-account-id-here"

Then run this script again.
        """)
        sys.exit(1)

    # Initialize manager
    manager = MetaAPIDataManager(token, account_id, Path(args.output))

    # Connect to MetaAPI
    if not await manager.connect():
        sys.exit(1)

    try:
        # Execute requested mode
        if args.init:
            success = await manager.download_initial_history(
                symbol=args.symbol,
                timeframe=args.timeframe,
                years=args.years
            )
            sys.exit(0 if success else 1)

        elif args.sync:
            success = await manager.extend_history(
                symbol=args.symbol,
                timeframe=args.timeframe
            )
            sys.exit(0 if success else 1)

        elif args.sync_all:
            symbols = args.symbols.split(',') if args.symbols else [DEFAULT_SYMBOL]
            timeframes = args.timeframes.split(',') if args.timeframes else [DEFAULT_TIMEFRAME]

            results = await manager.sync_all(symbols, timeframes)
            sys.exit(0 if results['failed'] == [] else 1)

    finally:
        await manager.close()


if __name__ == '__main__':
    if not METAAPI_AVAILABLE:
        sys.exit(1)

    asyncio.run(main())
