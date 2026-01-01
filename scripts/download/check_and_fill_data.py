#!/usr/bin/env python3
"""
Check and Fill Missing Data
============================

Continuously improve database by:
1. Checking for missing timeframes (e.g., has M15/H1/H4 but missing M30)
2. Checking for data gaps in existing files
3. Checking for new symbols from broker
4. Offering to download missing pieces

Usage:
    python scripts/check_and_fill_data.py
"""

import os
import sys
import asyncio
import getpass
from pathlib import Path
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Set, Tuple
import pandas as pd
from collections import defaultdict

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from metaapi_cloud_sdk import MetaApi
    METAAPI_AVAILABLE = True
except ImportError:
    METAAPI_AVAILABLE = False
    print("❌ MetaAPI not installed. Run: pip install metaapi-cloud-sdk")
    sys.exit(1)


TIMEFRAME_MAP = {
    'M15': '15m',
    'M30': '30m',
    'H1': '1h',
    'H4': '4h',
    'D1': '1d'
}

ALL_TIMEFRAMES = list(TIMEFRAME_MAP.keys())


def print_header(text: str):
    """Print section header."""
    print("\n" + "=" * 80)
    print(f"  {text}")
    print("=" * 80)


def print_step(step_num: int, text: str):
    """Print step header."""
    print(f"\n[STEP {step_num}] {text}")
    print("-" * 80)


def analyze_existing_data(data_dir: Path) -> Dict:
    """Analyze what data we currently have."""
    csv_files = list(data_dir.glob("*.csv"))

    if not csv_files:
        return {
            'total_files': 0,
            'symbols': {},
            'by_timeframe': {},
            'files': []
        }

    # Parse filenames
    by_symbol = defaultdict(lambda: {'timeframes': set(), 'files': []})
    by_timeframe = defaultdict(list)

    for filepath in csv_files:
        parts = filepath.stem.split('_')
        if len(parts) < 2:
            continue

        symbol = parts[0]
        timeframe = parts[1]

        by_symbol[symbol]['timeframes'].add(timeframe)
        by_symbol[symbol]['files'].append(filepath)
        by_timeframe[timeframe].append(symbol)

    return {
        'total_files': len(csv_files),
        'symbols': dict(by_symbol),
        'by_timeframe': dict(by_timeframe),
        'files': csv_files
    }


def find_missing_timeframes(existing_data: Dict) -> List[Tuple[str, Set[str]]]:
    """Find symbols with incomplete timeframe coverage."""
    missing = []

    for symbol, info in existing_data['symbols'].items():
        has_timeframes = info['timeframes']
        missing_tfs = set(ALL_TIMEFRAMES) - has_timeframes

        if missing_tfs:
            missing.append((symbol, missing_tfs))

    return missing


def find_data_gaps(filepath: Path, timeframe: str) -> List[Dict]:
    """Find gaps in a data file."""
    try:
        df = pd.read_csv(filepath)
        df['time'] = pd.to_datetime(df['time'])
        df = df.sort_values('time')

        # Expected intervals (in minutes)
        intervals = {
            'M15': 15,
            'M30': 30,
            'H1': 60,
            'H4': 240,
            'D1': 1440
        }

        expected_interval = intervals.get(timeframe, 60)

        # Calculate time differences
        time_diffs = df['time'].diff()

        # Find large gaps (>3x expected interval, accounting for weekends)
        threshold = timedelta(minutes=expected_interval * 3)
        large_gaps = []

        for idx in time_diffs.index[1:]:
            gap = time_diffs.loc[idx]
            if gap > threshold:
                prev_time = df.loc[idx - 1, 'time']
                curr_time = df.loc[idx, 'time']

                # Check if gap is just a weekend (Friday to Monday)
                is_weekend = (prev_time.weekday() == 4 and  # Friday
                             curr_time.weekday() == 0 and    # Monday
                             gap <= timedelta(days=3))

                if not is_weekend:
                    large_gaps.append({
                        'start': prev_time,
                        'end': curr_time,
                        'gap_hours': gap.total_seconds() / 3600
                    })

        return large_gaps

    except Exception as e:
        return []


def check_for_gaps(existing_data: Dict) -> Dict:
    """Check all files for data gaps."""
    files_with_gaps = {}

    print("\n🔍 Scanning for data gaps...")

    for i, filepath in enumerate(existing_data['files'], 1):
        if i % 10 == 0:
            print(f"  Progress: {i}/{len(existing_data['files'])}", end='\r')

        parts = filepath.stem.split('_')
        if len(parts) < 2:
            continue

        symbol = parts[0]
        timeframe = parts[1]

        gaps = find_data_gaps(filepath, timeframe)

        if gaps:
            files_with_gaps[filepath.name] = {
                'symbol': symbol,
                'timeframe': timeframe,
                'gaps': gaps
            }

    print(f"  Completed: {len(existing_data['files'])}/{len(existing_data['files'])}")

    return files_with_gaps


class DataFiller:
    """Fill missing data using MetaAPI."""

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

            if self.account.state != 'DEPLOYED':
                print("  Deploying account...")
                await self.account.deploy()

            if self.account.connection_status != 'CONNECTED':
                print("  Waiting for connection...")
                await self.account.wait_connected()

            self.connection = self.account.get_rpc_connection()
            await self.connection.connect()
            await self.connection.wait_synchronized()

            print(f"✅ Connected to: {self.account.name}")
            return True

        except Exception as e:
            print(f"❌ Connection failed: {e}")
            return False

    async def download_candles(self, symbol: str, timeframe: str, days: int = 365) -> pd.DataFrame:
        """Download candles for symbol/timeframe."""
        try:
            start_time = datetime.now(timezone.utc) - timedelta(days=days)

            # Use account object for historical candles (not RPC connection)
            candles = await self.account.get_historical_candles(
                symbol=symbol,
                timeframe=TIMEFRAME_MAP[timeframe],
                start_time=start_time,
                limit=50000
            )

            if not candles or len(candles) < 100:
                return None

            # Convert to DataFrame
            df = pd.DataFrame(candles)
            df['time'] = pd.to_datetime(df['time'])

            # Standardize columns
            df = df.rename(columns={
                'tickVolume': 'volume',
                'spread': 'spread'
            })

            # Keep essential columns
            cols = ['time', 'open', 'high', 'low', 'close', 'volume']
            if 'spread' in df.columns:
                cols.append('spread')

            df = df[[c for c in cols if c in df.columns]]

            return df

        except Exception as e:
            return None

    async def fill_missing_timeframes(self, missing: List[Tuple[str, Set[str]]], output_dir: Path):
        """Download missing timeframes."""
        total = sum(len(tfs) for _, tfs in missing)
        downloaded = 0
        failed = 0

        print(f"\n📥 Downloading {total} missing timeframes...")

        for symbol, timeframes in missing:
            print(f"\n{symbol}:")

            for tf in sorted(timeframes):
                df = await self.download_candles(symbol, tf, days=365)

                if df is None:
                    print(f"  {tf}: ❌ No data")
                    failed += 1
                    continue

                # Save
                start_date = df['time'].min().strftime('%Y%m%d%H%M')
                end_date = df['time'].max().strftime('%Y%m%d%H%M')
                output_file = output_dir / f"{symbol}_{tf}_{start_date}_{end_date}.csv"

                df.to_csv(output_file, index=False)
                print(f"  {tf}: ✅ {len(df)} bars → {output_file.name}")
                downloaded += 1

        return downloaded, failed

    async def fill_gaps(self, files_with_gaps: Dict, output_dir: Path):
        """Re-download files with gaps to fill them."""
        print(f"\n🔧 Re-downloading {len(files_with_gaps)} files with gaps...")

        downloaded = 0
        failed = 0

        for filename, info in files_with_gaps.items():
            symbol = info['symbol']
            timeframe = info['timeframe']

            print(f"\n{symbol} {timeframe} ({len(info['gaps'])} gaps):")

            # Re-download
            df = await self.download_candles(symbol, timeframe, days=365)

            if df is None:
                print(f"  ❌ Failed to re-download")
                failed += 1
                continue

            # Save (will replace old file)
            start_date = df['time'].min().strftime('%Y%m%d%H%M')
            end_date = df['time'].max().strftime('%Y%m%d%H%M')
            output_file = output_dir / f"{symbol}_{timeframe}_{start_date}_{end_date}.csv"

            # Remove old file(s)
            old_files = list(output_dir.glob(f"{symbol}_{timeframe}_*.csv"))
            for old_file in old_files:
                old_file.unlink()

            df.to_csv(output_file, index=False)
            print(f"  ✅ {len(df)} bars → {output_file.name}")
            downloaded += 1

        return downloaded, failed

    async def close(self):
        """Close connection."""
        if self.connection:
            await self.connection.close()


async def main():
    """Run data check and fill."""
    print_header("CHECK AND FILL MISSING DATA")

    data_dir = Path("data/master")

    if not data_dir.exists():
        print(f"\n❌ Data directory not found: {data_dir}")
        print(f"   Run: python scripts/download_interactive.py")
        return

    # Step 1: Analyze existing data
    print_step(1, "Analyzing existing data...")

    existing_data = analyze_existing_data(data_dir)

    if existing_data['total_files'] == 0:
        print("\n❌ No data files found")
        print(f"   Run: python scripts/download_interactive.py")
        return

    print(f"\n✅ Found {existing_data['total_files']} files")
    print(f"   Symbols: {len(existing_data['symbols'])}")
    print(f"   Timeframes: {', '.join(existing_data['by_timeframe'].keys())}")

    # Step 2: Check for missing timeframes
    print_step(2, "Checking for missing timeframes...")

    missing_timeframes = find_missing_timeframes(existing_data)

    if missing_timeframes:
        print(f"\n⚠️  Found {len(missing_timeframes)} symbols with incomplete timeframe coverage:")

        for symbol, missing_tfs in missing_timeframes[:10]:
            has_tfs = existing_data['symbols'][symbol]['timeframes']
            print(f"  {symbol:15s}: has {', '.join(sorted(has_tfs)):20s} | missing {', '.join(sorted(missing_tfs))}")

        if len(missing_timeframes) > 10:
            print(f"  ... and {len(missing_timeframes) - 10} more symbols")
    else:
        print("\n✅ All symbols have complete timeframe coverage")

    # Step 3: Check for data gaps
    print_step(3, "Checking for data gaps...")

    files_with_gaps = check_for_gaps(existing_data)

    if files_with_gaps:
        print(f"\n⚠️  Found {len(files_with_gaps)} files with data gaps:")

        for filename, info in list(files_with_gaps.items())[:10]:
            gap_summary = f"{len(info['gaps'])} gaps"
            if info['gaps']:
                largest_gap = max(info['gaps'], key=lambda g: g['gap_hours'])
                gap_summary += f" (largest: {largest_gap['gap_hours']:.1f}h)"
            print(f"  {filename:50s}: {gap_summary}")

        if len(files_with_gaps) > 10:
            print(f"  ... and {len(files_with_gaps) - 10} more files")
    else:
        print("\n✅ No significant data gaps found")

    # Summary
    print_header("SUMMARY")

    total_issues = len(missing_timeframes) + len(files_with_gaps)

    if total_issues == 0:
        print("\n🎉 Database is complete and healthy!")
        print("\n✅ Next step: Prepare data")
        print("   python scripts/prepare_data.py")
        return

    print(f"\n📊 Issues found:")
    print(f"  Missing timeframes: {len(missing_timeframes)} symbols")
    print(f"  Data gaps:          {len(files_with_gaps)} files")
    print(f"  Total issues:       {total_issues}")

    # Offer to fix
    print("\n💡 Actions available:")
    print("  1. Download missing timeframes")
    print("  2. Re-download files with gaps")
    print("  3. Do both (recommended)")
    print("  4. Skip (continue with existing data)")

    choice = input("\nSelect action [1-4]: ").strip()

    if choice == '4':
        print("\n⚠️  Skipping fixes")
        return

    # Try loading from .env file first
    env_file = Path.cwd() / '.env'
    if env_file.exists():
        with open(env_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    if key == 'METAAPI_TOKEN' and key not in os.environ:
                        os.environ[key] = value
                    elif key == 'METAAPI_ACCOUNT_ID' and key not in os.environ:
                        os.environ[key] = value

    # Get credentials
    token = os.environ.get('METAAPI_TOKEN')
    account_id = os.environ.get('METAAPI_ACCOUNT_ID')

    # Check for placeholder values
    placeholder_patterns = ['your-token-here', 'your-account-id-here', 'placeholder', 'example']

    # Check token
    if token and any(placeholder in token.lower() for placeholder in placeholder_patterns):
        print(f"\n⚠️  Found placeholder METAAPI_TOKEN (ignoring it)")
        token = None

    # Prompt for token if not set
    if not token:
        print("\n📋 MetaAPI Token Required")
        print("Get your token from: https://app.metaapi.cloud/")
        token = getpass.getpass("\nEnter your MetaAPI token (hidden): ").strip()

        if not token:
            print("\n❌ No token provided")
            return

    # Check account ID
    if account_id and any(placeholder in account_id.lower() for placeholder in placeholder_patterns):
        print(f"\n⚠️  Found placeholder METAAPI_ACCOUNT_ID (ignoring it)")
        account_id = None

    # Prompt for account ID if not set
    if not account_id:
        print("\n📋 MetaAPI Account ID Required")
        print("Get this from: https://app.metaapi.cloud/")
        print("(UUID format: e8f8c21a-32b5-40b0-9bf7-672e8ffab91f)")
        account_id = getpass.getpass("\nEnter your MetaAPI account ID (hidden): ").strip()

        if not account_id:
            print("\n❌ No account ID provided")
            return

    # Connect and fix
    filler = DataFiller(token, account_id)

    print("\n🔌 Connecting to MetaAPI...")
    if not await filler.connect():
        return

    try:
        downloaded_count = 0
        failed_count = 0

        if choice in ['1', '3'] and missing_timeframes:
            d, f = await filler.fill_missing_timeframes(missing_timeframes, data_dir)
            downloaded_count += d
            failed_count += f

        if choice in ['2', '3'] and files_with_gaps:
            d, f = await filler.fill_gaps(files_with_gaps, data_dir)
            downloaded_count += d
            failed_count += f

        # Final summary
        print_header("FILL COMPLETE")

        print(f"\n📊 Results:")
        print(f"  ✅ Downloaded: {downloaded_count}")
        print(f"  ❌ Failed:     {failed_count}")

        # Re-analyze
        new_data = analyze_existing_data(data_dir)
        new_missing = find_missing_timeframes(new_data)

        print(f"\n📈 Database status:")
        print(f"  Files:              {new_data['total_files']}")
        print(f"  Symbols:            {len(new_data['symbols'])}")
        print(f"  Incomplete symbols: {len(new_missing)}")

        if len(new_missing) == 0 and failed_count == 0:
            print("\n🎉 Database is now complete!")

        print("\n✅ Next step: Check integrity and prepare")
        print("   python scripts/check_data_integrity.py")
        print("   python scripts/prepare_data.py")

    finally:
        await filler.close()


if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
