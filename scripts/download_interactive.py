#!/usr/bin/env python3
"""
Interactive MetaAPI Data Downloader
====================================

Step-by-step workflow:
1. Select MetaAPI account
2. Select asset class(es) to download
3. Select specific symbols and timeframes
4. Download efficiently with progress tracking

Usage:
    python scripts/download_interactive.py
"""

import os
import sys
import asyncio
import json
import getpass
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set
import pandas as pd

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from metaapi_cloud_sdk import MetaApi
    METAAPI_AVAILABLE = True
except ImportError:
    METAAPI_AVAILABLE = False
    print("❌ MetaAPI not installed. Run: pip install metaapi-cloud-sdk")
    sys.exit(1)


# Market classifications
ASSET_CLASSES = {
    'forex': {
        'name': 'Forex (Currency Pairs)',
        'patterns': ['USD', 'EUR', 'GBP', 'JPY', 'AUD', 'CAD', 'CHF', 'NZD'],
        'examples': ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD']
    },
    'crypto': {
        'name': 'Cryptocurrency',
        'patterns': ['BTC', 'ETH', 'XRP', 'LTC', 'ADA', 'DOT'],
        'examples': ['BTCUSD', 'ETHUSD', 'BTCJPY', 'ETHEUR']
    },
    'indices': {
        'name': 'Stock Indices',
        'patterns': ['SPX', 'NAS', 'DOW', 'DJ', 'DAX', 'FTSE', 'NIKKEI', 'US30', 'US500', 'GER40', 'UK100'],
        'examples': ['US500', 'NAS100', 'GER40', 'DJ30ft']
    },
    'metals': {
        'name': 'Precious Metals',
        'patterns': ['XAU', 'XAG', 'GOLD', 'SILVER', 'XPT', 'XPD'],
        'examples': ['XAUUSD', 'XAGUSD', 'XPTUSD']
    },
    'commodities': {
        'name': 'Commodities',
        'patterns': ['OIL', 'WTI', 'BRENT', 'GAS', 'COPPER'],
        'examples': ['UKOUSD', 'COPPER-C']
    }
}

TIMEFRAME_MAP = {
    'M15': '15m',
    'M30': '30m',
    'H1': '1h',
    'H4': '4h',
    'D1': '1d'
}


def classify_symbol(symbol: str) -> str:
    """Classify symbol into asset class."""
    symbol_upper = symbol.upper().replace('+', '').replace('-', '')

    # Check each asset class
    for class_id, info in ASSET_CLASSES.items():
        for pattern in info['patterns']:
            if pattern in symbol_upper:
                return class_id

    return 'unknown'


def print_header(text: str):
    """Print section header."""
    print("\n" + "=" * 80)
    print(f"  {text}")
    print("=" * 80)


def print_step(step_num: int, text: str):
    """Print step header."""
    print(f"\n[STEP {step_num}] {text}")
    print("-" * 80)


def save_credentials_to_env(token: str, account_id: str = None):
    """Save credentials to .env file for persistent storage."""
    env_file = Path.cwd() / '.env'

    # Read existing .env if it exists
    env_lines = {}
    if env_file.exists():
        with open(env_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    env_lines[key] = value

    # Update credentials
    if token:
        env_lines['METAAPI_TOKEN'] = token
    if account_id:
        env_lines['METAAPI_ACCOUNT_ID'] = account_id

    # Write back
    with open(env_file, 'w') as f:
        f.write("# Kinetra MetaAPI Credentials\n")
        f.write("# Auto-generated - do not commit to git\n\n")
        for key, value in env_lines.items():
            f.write(f"{key}={value}\n")

    print(f"\n✅ Credentials saved to {env_file}")

    # Add to .gitignore if not already there
    gitignore = Path.cwd() / '.gitignore'
    if gitignore.exists():
        content = gitignore.read_text()
        if '.env' not in content:
            with open(gitignore, 'a') as f:
                f.write("\n# Environment variables\n.env\n")


class InteractiveDownloader:
    """Interactive downloader with step-by-step workflow."""

    def __init__(self):
        self.api = None
        self.account = None
        self.connection = None
        self.account_id = None
        self.token = None

    async def step1_select_account(self) -> bool:
        """Step 1: Select MetaAPI account."""
        print_step(1, "Select MetaAPI Account")

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

        # Check for token in environment
        self.token = os.environ.get('METAAPI_TOKEN')

        # Check for placeholder values
        placeholder_patterns = ['your-token-here', 'your-account-id-here', 'placeholder', 'example']

        should_save = False  # Track if we should save credentials

        if self.token and any(placeholder in self.token.lower() for placeholder in placeholder_patterns):
            print(f"\n⚠️  Found placeholder token in environment (ignoring it)")
            self.token = None

        if not self.token:
            print("\n📋 MetaAPI Token Required")
            print("Get your token from: https://app.metaapi.cloud/")
            print("(Sign up if you don't have an account)")

            # Use getpass for hidden input
            self.token = getpass.getpass("\nEnter your MetaAPI token (hidden): ").strip()

            if not self.token:
                print("\n❌ No token provided")
                return False

            # Ask if they want to save it
            save = input("\n💾 Save credentials to .env file? [1=Yes, 2=No]: ").strip()
            should_save = (save == '1')

        print(f"\n✅ Using API token: {self.token[:8]}***")

        # Check for account ID in environment
        env_account_id = os.environ.get('METAAPI_ACCOUNT_ID')

        if env_account_id:
            # Check if it's a placeholder
            if any(placeholder in env_account_id.lower() for placeholder in placeholder_patterns):
                print(f"\n⚠️  Found placeholder account ID (ignoring it)")
                env_account_id = None
            else:
                print(f"\n✅ Found account ID: {env_account_id[:8]}***")
                response = input(f"\nUse this account? [1=Yes, 2=List all accounts]: ").strip()

                if response == '1':
                    self.account_id = env_account_id
                    return True

        # List available accounts
        try:
            self.api = MetaApi(self.token)
            accounts = await self.api.metatrader_account_api.get_accounts()

            if not accounts:
                print("\n❌ No MetaAPI accounts found")
                return False

            print(f"\n📋 Available Accounts ({len(accounts)}):")
            for i, acc in enumerate(accounts, 1):
                print(f"  {i}. {acc.name} ({acc.login}) - {acc.type}")

            choice = input(f"\nSelect account [1-{len(accounts)}]: ").strip()

            try:
                idx = int(choice) - 1
                if 0 <= idx < len(accounts):
                    self.account_id = accounts[idx].id
                    print(f"\n✅ Selected: {accounts[idx].name}")

                    # Save credentials if requested
                    if should_save:
                        save_credentials_to_env(self.token, self.account_id)
                        os.environ['METAAPI_TOKEN'] = self.token
                        os.environ['METAAPI_ACCOUNT_ID'] = self.account_id

                    return True
                else:
                    print(f"\n❌ Invalid choice")
                    return False
            except ValueError:
                print(f"\n❌ Invalid input")
                return False

        except Exception as e:
            print(f"\n❌ Failed to list accounts: {e}")
            return False

    async def connect(self) -> bool:
        """Connect to MetaAPI account."""
        try:
            print(f"\n🔌 Connecting to MetaAPI...")

            if not self.api:
                self.api = MetaApi(self.token)

            self.account = await self.api.metatrader_account_api.get_account(self.account_id)

            # Deploy if needed
            if self.account.state != 'DEPLOYED':
                print("  Deploying account...")
                await self.account.deploy()

            # Wait for connection
            if self.account.connection_status != 'CONNECTED':
                print("  Waiting for connection...")
                await self.account.wait_connected()

            # Get RPC connection
            self.connection = self.account.get_rpc_connection()
            await self.connection.connect()
            await self.connection.wait_synchronized()

            print(f"✅ Connected to: {self.account.name}")
            return True

        except Exception as e:
            print(f"❌ Connection failed: {e}")
            return False

    async def step2_select_asset_classes(self) -> List[str]:
        """Step 2: Select asset classes to download."""
        print_step(2, "Select Asset Classes")

        print("\nAvailable asset classes:")
        class_ids = list(ASSET_CLASSES.keys())
        for i, class_id in enumerate(class_ids, 1):
            info = ASSET_CLASSES[class_id]
            examples = ', '.join(info['examples'][:3])
            print(f"  {i}. {info['name']:25s} (e.g., {examples})")

        print(f"  {len(class_ids) + 1}. All classes")

        choice = input(f"\nSelect classes [1-{len(class_ids) + 1}, or comma-separated like 1,3,4]: ").strip()

        # Parse selection
        if choice == str(len(class_ids) + 1):
            selected = class_ids
        else:
            try:
                indices = [int(x.strip()) - 1 for x in choice.split(',')]
                selected = [class_ids[i] for i in indices if 0 <= i < len(class_ids)]
            except (ValueError, IndexError):
                print("\n❌ Invalid selection, using all classes")
                selected = class_ids

        print(f"\n✅ Selected: {', '.join([ASSET_CLASSES[c]['name'] for c in selected])}")
        return selected

    async def step3_select_symbols(self, asset_classes: List[str]) -> List[str]:
        """Step 3: Select symbols from chosen asset classes."""
        print_step(3, "Select Symbols")

        # Get all available symbols
        print("\n🔍 Fetching available symbols from broker...")
        try:
            all_symbols = await self.connection.get_symbols()
            tradeable = [s['symbol'] for s in all_symbols if s.get('tradeMode') != 'DISABLED']
            print(f"✅ Found {len(tradeable)} tradeable symbols")
        except Exception as e:
            print(f"❌ Failed to fetch symbols: {e}")
            return []

        # Classify symbols
        by_class = {c: [] for c in asset_classes}
        for symbol in tradeable:
            class_id = classify_symbol(symbol)
            if class_id in asset_classes:
                by_class[class_id].append(symbol)

        # Show breakdown
        print("\nSymbols by asset class:")
        for class_id in asset_classes:
            symbols = sorted(by_class[class_id])
            print(f"\n  {ASSET_CLASSES[class_id]['name']} ({len(symbols)} symbols):")

            # Show first 20
            for symbol in symbols[:20]:
                print(f"    {symbol}")
            if len(symbols) > 20:
                print(f"    ... and {len(symbols) - 20} more")

        # Selection options
        print("\nWhat would you like to download?")
        print("  1. All symbols from selected classes")
        print("  2. Specific symbols (comma-separated)")
        print("  3. Top N symbols from each class")

        choice = input("\nEnter choice [1-3]: ").strip()

        if choice == '1':
            # All symbols from selected classes
            selected = []
            for class_id in asset_classes:
                selected.extend(by_class[class_id])

        elif choice == '2':
            # Specific symbols
            symbol_input = input("\nEnter symbols (comma-separated): ").strip()
            selected = [s.strip().upper() for s in symbol_input.split(',')]
            # Validate
            selected = [s for s in selected if s in tradeable]

        elif choice == '3':
            # Top N from each class
            n = input("\nHow many from each class? [default: 5]: ").strip()
            n = int(n) if n.isdigit() else 5
            selected = []
            for class_id in asset_classes:
                selected.extend(sorted(by_class[class_id])[:n])

        else:
            print("❌ Invalid choice, using all symbols")
            selected = []
            for class_id in asset_classes:
                selected.extend(by_class[class_id])

        print(f"\n✅ Selected {len(selected)} symbols")
        return selected

    async def step4_select_timeframes(self) -> List[str]:
        """Step 4: Select timeframes to download."""
        print_step(4, "Select Timeframes")

        timeframes = list(TIMEFRAME_MAP.keys())
        print("\nAvailable timeframes:")
        for i, tf in enumerate(timeframes, 1):
            print(f"  {i}. {tf:5s} ({TIMEFRAME_MAP[tf]})")
        print(f"  {len(timeframes) + 1}. All timeframes")

        choice = input(f"\nSelect timeframes [1-{len(timeframes) + 1}, or comma-separated]: ").strip()

        if choice == str(len(timeframes) + 1):
            selected = timeframes
        else:
            try:
                indices = [int(x.strip()) - 1 for x in choice.split(',')]
                selected = [timeframes[i] for i in indices if 0 <= i < len(timeframes)]
            except (ValueError, IndexError):
                print("❌ Invalid selection, using all timeframes")
                selected = timeframes

        print(f"\n✅ Selected: {', '.join(selected)}")
        return selected

    async def download_candles(self, symbol: str, timeframe: str, days: int = 365) -> Optional[pd.DataFrame]:
        """Download candle data for symbol/timeframe."""
        try:
            start_time = datetime.utcnow() - timedelta(days=days)

            candles = await self.connection.get_historical_candles(
                symbol=symbol,
                timeframe=TIMEFRAME_MAP[timeframe],
                start_time=start_time,
                limit=50000
            )

            if not candles or len(candles) < 500:
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
            print(f"    ❌ {timeframe}: {e}")
            return None

    async def step5_download(self, symbols: List[str], timeframes: List[str], output_dir: Path):
        """Step 5: Download data efficiently."""
        print_step(5, "Download Data")

        total = len(symbols) * len(timeframes)
        print(f"\n📥 Downloading {len(symbols)} symbols × {len(timeframes)} timeframes = {total} datasets")
        print(f"📁 Output: {output_dir}")

        output_dir.mkdir(parents=True, exist_ok=True)

        downloaded = 0
        failed = 0
        skipped = 0

        for i, symbol in enumerate(symbols, 1):
            print(f"\n[{i}/{len(symbols)}] {symbol}")

            for tf in timeframes:
                # Check if already exists
                filename = f"{symbol}_{tf}_*.csv"
                existing = list(output_dir.glob(filename))

                if existing:
                    print(f"  {tf}: ⏭️  Already exists ({existing[0].name})")
                    skipped += 1
                    continue

                # Download
                df = await self.download_candles(symbol, tf, days=365)

                if df is None:
                    print(f"  {tf}: ❌ No data")
                    failed += 1
                    continue

                # Save with date range in filename
                start_date = df['time'].min().strftime('%Y%m%d%H%M')
                end_date = df['time'].max().strftime('%Y%m%d%H%M')
                output_file = output_dir / f"{symbol}_{tf}_{start_date}_{end_date}.csv"

                df.to_csv(output_file, index=False)
                print(f"  {tf}: ✅ {len(df)} bars → {output_file.name}")
                downloaded += 1

        # Summary
        print_header("DOWNLOAD COMPLETE")
        print(f"\n✅ Downloaded: {downloaded}")
        print(f"⏭️  Skipped: {skipped} (already exist)")
        print(f"❌ Failed: {failed}")
        print(f"\n📁 Data saved to: {output_dir}")

    async def close(self):
        """Close connection."""
        if self.connection:
            await self.connection.close()


async def main():
    """Run interactive download workflow."""
    print_header("KINETRA INTERACTIVE DATA DOWNLOADER")

    downloader = InteractiveDownloader()

    # Step 1: Select account
    if not await downloader.step1_select_account():
        return

    # Connect to account
    if not await downloader.connect():
        return

    try:
        # Step 2: Select asset classes
        asset_classes = await downloader.step2_select_asset_classes()
        if not asset_classes:
            print("❌ No asset classes selected")
            return

        # Step 3: Select symbols
        symbols = await downloader.step3_select_symbols(asset_classes)
        if not symbols:
            print("❌ No symbols selected")
            return

        # Step 4: Select timeframes
        timeframes = await downloader.step4_select_timeframes()
        if not timeframes:
            print("❌ No timeframes selected")
            return

        # Confirm before download
        print_header("DOWNLOAD CONFIRMATION")
        print(f"\n📊 Summary:")
        print(f"  Symbols:    {len(symbols)}")
        print(f"  Timeframes: {len(timeframes)}")
        print(f"  Total:      {len(symbols) * len(timeframes)} datasets")

        confirm = input(f"\nProceed with download? [1=Yes, 2=No]: ").strip()
        if confirm != '1':
            print("\n⚠️  Download cancelled")
            return

        # Step 5: Download
        output_dir = Path("data/master")
        await downloader.step5_download(symbols, timeframes, output_dir)

    finally:
        await downloader.close()


if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n⚠️  Download interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
