#!/usr/bin/env python3
"""
MetaAPI Cloud Data Sync

Automatically downloads and categorizes instrument data from MetaTrader via MetaAPI.

Features:
- Download all available symbols and specifications
- Categorize instruments by asset class (forex, crypto, indices, commodities, etc.)
- Download historical OHLCV data for all timeframes
- Weekly auto-update capability (Saturday night)

Usage:
    # List accounts
    python scripts/metaapi_sync.py --list-accounts

    # Download symbol specifications
    python scripts/metaapi_sync.py --symbols

    # Download historical data for all symbols
    python scripts/metaapi_sync.py --history --timeframe H1 --days 365

    # Full sync (symbols + history)
    python scripts/metaapi_sync.py --full-sync

    # Weekly update mode (run via cron on Saturday)
    python scripts/metaapi_sync.py --weekly-update
"""

import os
import sys
import json
import asyncio
import argparse
import tempfile
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# MetaAPI SDK
from metaapi_cloud_sdk import MetaApi

import pandas as pd
import numpy as np


# =============================================================================
# ATOMIC FILE OPERATIONS
# =============================================================================

def atomic_write_text(filepath: Path, content: str, mode: str = 'w'):
    """
    Write content to file atomically.

    Writes to a temp file first, then atomically renames to target.
    Prevents corrupt/partial files on interrupted writes.
    """
    filepath = Path(filepath)

    # Create temp file in same directory (required for atomic rename)
    fd, tmp_path = tempfile.mkstemp(
        suffix='.tmp',
        prefix=f'.{filepath.name}.',
        dir=filepath.parent
    )

    try:
        with os.fdopen(fd, mode) as f:
            f.write(content)

        # Atomic rename (POSIX guarantees atomicity on same filesystem)
        os.replace(tmp_path, filepath)

    except Exception:
        # Clean up temp file on failure
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def atomic_write_json(filepath: Path, data: dict, indent: int = 2):
    """Write JSON data to file atomically."""
    content = json.dumps(data, indent=indent)
    atomic_write_text(filepath, content)


def atomic_write_csv(filepath: Path, df: pd.DataFrame, **kwargs):
    """
    Write DataFrame to CSV atomically.

    Args:
        filepath: Target file path
        df: DataFrame to save
        **kwargs: Additional arguments for df.to_csv()
    """
    filepath = Path(filepath)

    # Create temp file in same directory
    fd, tmp_path = tempfile.mkstemp(
        suffix='.tmp',
        prefix=f'.{filepath.name}.',
        dir=filepath.parent
    )
    os.close(fd)  # Close fd, pandas will open by path

    try:
        df.to_csv(tmp_path, **kwargs)

        # Atomic rename
        os.replace(tmp_path, filepath)

    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


# =============================================================================
# ASSET CLASS CATEGORIZATION
# =============================================================================

class AssetClass(Enum):
    """Asset class categories for instrument organization."""
    FOREX_MAJOR = "forex_major"
    FOREX_MINOR = "forex_minor"
    FOREX_EXOTIC = "forex_exotic"
    CRYPTO = "crypto"
    INDEX = "index"
    COMMODITY_METAL = "commodity_metal"
    COMMODITY_ENERGY = "commodity_energy"
    COMMODITY_SOFT = "commodity_soft"
    STOCK = "stock"
    ETF = "etf"
    BOND = "bond"
    UNKNOWN = "unknown"


# Forex major pairs (USD crosses with major currencies)
FOREX_MAJORS = {
    'EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'AUDUSD', 'USDCAD', 'NZDUSD'
}

# Forex minor (cross pairs without USD)
FOREX_MINORS = {
    'EURGBP', 'EURJPY', 'EURCHF', 'EURAUD', 'EURCAD', 'EURNZD',
    'GBPJPY', 'GBPCHF', 'GBPAUD', 'GBPCAD', 'GBPNZD',
    'AUDJPY', 'AUDCHF', 'AUDCAD', 'AUDNZD',
    'NZDJPY', 'NZDCHF', 'NZDCAD',
    'CADJPY', 'CADCHF', 'CHFJPY'
}

# Crypto symbols
CRYPTO_KEYWORDS = ['BTC', 'ETH', 'LTC', 'XRP', 'BCH', 'ADA', 'DOT', 'LINK',
                   'SOL', 'DOGE', 'AVAX', 'MATIC', 'UNI', 'ATOM']

# Index symbols
INDEX_KEYWORDS = ['US500', 'US30', 'US100', 'NAS100', 'DJ30', 'SP500', 'NDX',
                  'DAX', 'GER40', 'GER30', 'FTSE', 'UK100', 'CAC40', 'FRA40',
                  'EU50', 'STOXX', 'NIKKEI', 'JP225', 'HSI', 'ASX200', 'AUS200',
                  'IBEX', 'ESP35', 'MIB40', 'SA40', 'US2000', 'RUSSELL']

# Metal commodities
METAL_KEYWORDS = ['XAU', 'XAG', 'XPT', 'XPD', 'GOLD', 'SILVER', 'PLATINUM',
                  'PALLADIUM', 'COPPER']

# Energy commodities
ENERGY_KEYWORDS = ['WTI', 'BRENT', 'OIL', 'CRUDE', 'NGAS', 'NATGAS', 'UKO', 'USO']

# Soft commodities
SOFT_KEYWORDS = ['WHEAT', 'CORN', 'SOYBEAN', 'COFFEE', 'COCOA', 'SUGAR',
                 'COTTON', 'ORANGE']


def categorize_symbol(symbol: str, description: str = "") -> AssetClass:
    """
    Categorize a symbol into its asset class.

    Args:
        symbol: Trading symbol (e.g., 'EURUSD', 'BTCUSD')
        description: Optional description from broker

    Returns:
        AssetClass enum value
    """
    # Clean symbol (remove + suffix, etc.)
    clean = symbol.upper().replace('+', '').replace('.', '').replace('-', '')

    # Check forex majors
    if clean in FOREX_MAJORS or any(clean.startswith(m) or clean.endswith(m)
                                     for m in ['EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF']):
        return AssetClass.FOREX_MAJOR

    # Check forex minors
    if clean in FOREX_MINORS:
        return AssetClass.FOREX_MINOR

    # Check crypto
    if any(kw in clean for kw in CRYPTO_KEYWORDS):
        return AssetClass.CRYPTO

    # Check indices
    if any(kw in clean for kw in INDEX_KEYWORDS):
        return AssetClass.INDEX

    # Check metals
    if any(kw in clean for kw in METAL_KEYWORDS):
        return AssetClass.COMMODITY_METAL

    # Check energy
    if any(kw in clean for kw in ENERGY_KEYWORDS):
        return AssetClass.COMMODITY_ENERGY

    # Check soft commodities
    if any(kw in clean for kw in SOFT_KEYWORDS):
        return AssetClass.COMMODITY_SOFT

    # Check description for clues
    desc_upper = description.upper()
    if 'FOREX' in desc_upper or 'CURRENCY' in desc_upper:
        return AssetClass.FOREX_EXOTIC
    if 'CRYPTO' in desc_upper:
        return AssetClass.CRYPTO
    if 'INDEX' in desc_upper or 'INDICE' in desc_upper:
        return AssetClass.INDEX
    if 'STOCK' in desc_upper or 'SHARE' in desc_upper:
        return AssetClass.STOCK
    if 'ETF' in desc_upper:
        return AssetClass.ETF
    if 'BOND' in desc_upper:
        return AssetClass.BOND
    if 'METAL' in desc_upper or 'GOLD' in desc_upper or 'SILVER' in desc_upper:
        return AssetClass.COMMODITY_METAL
    if 'OIL' in desc_upper or 'GAS' in desc_upper or 'ENERGY' in desc_upper:
        return AssetClass.COMMODITY_ENERGY

    # Check if it looks like a forex pair (6 chars, known currencies)
    currencies = ['USD', 'EUR', 'GBP', 'JPY', 'CHF', 'AUD', 'CAD', 'NZD',
                  'SGD', 'HKD', 'SEK', 'NOK', 'DKK', 'PLN', 'ZAR', 'TRY',
                  'MXN', 'CNH', 'CNY', 'THB', 'INR']

    if len(clean) == 6:
        base = clean[:3]
        quote = clean[3:]
        if base in currencies and quote in currencies:
            return AssetClass.FOREX_EXOTIC

    return AssetClass.UNKNOWN


@dataclass
class SymbolInfo:
    """Symbol specification data."""
    symbol: str
    description: str
    asset_class: str
    digits: int
    contract_size: float
    min_volume: float
    max_volume: float
    volume_step: float
    pip_size: float
    currency_base: str
    currency_quote: str
    currency_profit: str
    margin_mode: str
    trade_mode: str
    tick_size: float
    stops_level: int
    freeze_level: int
    last_updated: str


class MetaAPISync:
    """MetaAPI data synchronization manager."""

    def __init__(self, token: str = None, account_id: str = None):
        """
        Initialize MetaAPI sync.

        Args:
            token: MetaAPI token (or from METAAPI_TOKEN env var)
            account_id: Account ID (or from METAAPI_ACCOUNT_ID env var, or auto-detect)
        """
        self.token = token or os.getenv('METAAPI_TOKEN')
        self.account_id = account_id or os.getenv('METAAPI_ACCOUNT_ID')

        if not self.token:
            raise ValueError("METAAPI_TOKEN not set. Add to .env file or pass as argument.")

        self.api: Optional[MetaApi] = None
        self.account = None
        self.connection = None

        # Output directories
        self.data_dir = Path("data")
        self.master_dir = self.data_dir / "master"
        self.symbols_dir = self.data_dir / "symbols"
        self.symbols_dir.mkdir(parents=True, exist_ok=True)

    async def connect(self):
        """Connect to MetaAPI and get account."""
        print("Connecting to MetaAPI...")
        self.api = MetaApi(token=self.token)

        # Get accounts using pagination API
        accounts_result = await self.api.metatrader_account_api.get_accounts_with_infinite_scroll_pagination()
        accounts = accounts_result.get('items', []) if isinstance(accounts_result, dict) else accounts_result

        if not accounts:
            raise ValueError("No MetaTrader accounts found. Please provision an account at https://app.metaapi.cloud/")

        print(f"Found {len(accounts)} account(s)")

        # Find account
        if self.account_id:
            self.account = next((a for a in accounts if a.id == self.account_id), None)
            if not self.account:
                raise ValueError(f"Account {self.account_id} not found")
        else:
            # Use first account
            self.account = accounts[0]
            self.account_id = self.account.id
            print(f"Using account: {self.account_id} ({self.account.name})")

        # Check deployment state
        if self.account.state != 'DEPLOYED':
            print(f"Account state: {self.account.state}. Deploying...")
            await self.account.deploy()
            await self.account.wait_deployed()

        # Wait for connection
        print("Waiting for API connection...")
        self.connection = self.account.get_rpc_connection()
        await self.connection.connect()
        await self.connection.wait_synchronized()

        print("Connected!")
        return self.account

    async def disconnect(self):
        """Disconnect from MetaAPI."""
        if self.connection:
            await self.connection.close()
        if self.api:
            await self.api.close()

    async def list_accounts(self) -> List[Dict]:
        """List all available MetaTrader accounts."""
        api = MetaApi(token=self.token)

        # Use pagination API to get accounts
        accounts_result = await api.metatrader_account_api.get_accounts_with_infinite_scroll_pagination()
        accounts = accounts_result.get('items', []) if isinstance(accounts_result, dict) else accounts_result

        result = []
        if not accounts:
            print("No MetaTrader accounts found.")
            print("Please provision an account at https://app.metaapi.cloud/")
        else:
            print(f"Found {len(accounts)} account(s):")
            for acc in accounts:
                info = {
                    'id': acc.id,
                    'name': acc.name,
                    'type': acc.type,
                    'platform': acc.platform,
                    'state': acc.state,
                    'broker': getattr(acc, 'broker', 'N/A'),
                    'server': getattr(acc, 'server', 'N/A'),
                }
                result.append(info)
                print(f"  {acc.id}: {acc.name} ({info['broker']}) [{acc.state}]")

        await api.close()
        return result

    async def get_symbols(self) -> List[str]:
        """Get all available symbols from the account."""
        if not self.connection:
            await self.connect()

        symbols = await self.connection.get_symbols()
        print(f"Found {len(symbols)} symbols")
        return symbols

    async def get_symbol_specification(self, symbol: str) -> Optional[Dict]:
        """Get specification for a single symbol."""
        try:
            spec = await self.connection.get_symbol_specification(symbol)
            return spec
        except Exception as e:
            print(f"  Error getting spec for {symbol}: {e}")
            return None

    async def get_all_specifications(self, symbols: List[str] = None) -> List[SymbolInfo]:
        """
        Get specifications for all symbols.

        Returns list of SymbolInfo objects.
        """
        if not self.connection:
            await self.connect()

        if symbols is None:
            symbols = await self.get_symbols()

        print(f"Fetching specifications for {len(symbols)} symbols...")

        results = []
        for i, symbol in enumerate(symbols):
            if (i + 1) % 20 == 0:
                print(f"  Progress: {i + 1}/{len(symbols)}")

            spec = await self.get_symbol_specification(symbol)
            if spec:
                # Categorize
                description = spec.get('description', '')
                asset_class = categorize_symbol(symbol, description)

                info = SymbolInfo(
                    symbol=symbol,
                    description=description,
                    asset_class=asset_class.value,
                    digits=spec.get('digits', 5),
                    contract_size=spec.get('contractSize', 1.0),
                    min_volume=spec.get('minVolume', 0.01),
                    max_volume=spec.get('maxVolume', 100.0),
                    volume_step=spec.get('volumeStep', 0.01),
                    pip_size=spec.get('pipSize', 0.0001),
                    currency_base=spec.get('currencyBase', ''),
                    currency_quote=spec.get('currencyQuote', ''),
                    currency_profit=spec.get('currencyProfit', ''),
                    margin_mode=spec.get('marginMode', ''),
                    trade_mode=spec.get('tradeMode', ''),
                    tick_size=spec.get('tickSize', 0.0),
                    stops_level=spec.get('stopsLevel', 0),
                    freeze_level=spec.get('freezeLevel', 0),
                    last_updated=datetime.now().isoformat(),
                )
                results.append(info)

            # Small delay to avoid rate limiting
            await asyncio.sleep(0.05)

        return results

    def save_symbols(self, symbols: List[SymbolInfo]):
        """Save symbol specifications to JSON and CSV atomically."""
        # Convert to dicts
        data = [asdict(s) for s in symbols]

        # Save as JSON (atomic)
        json_path = self.symbols_dir / "symbol_specs.json"
        atomic_write_json(json_path, data)
        print(f"Saved {len(symbols)} symbols to {json_path}")

        # Save as CSV for easy viewing (atomic)
        df = pd.DataFrame(data)
        csv_path = self.symbols_dir / "symbol_specs.csv"
        atomic_write_csv(csv_path, df, index=False)
        print(f"Saved to {csv_path}")

        # Save categorized summary
        summary = df.groupby('asset_class').size().to_dict()
        print("\nSymbols by Asset Class:")
        for cls, count in sorted(summary.items()):
            print(f"  {cls}: {count}")

        # Save per-category files (atomic)
        for asset_class in df['asset_class'].unique():
            subset = df[df['asset_class'] == asset_class]
            cat_path = self.symbols_dir / f"symbols_{asset_class}.csv"
            atomic_write_csv(cat_path, subset, index=False)

    async def download_historical_data(
        self,
        symbols: List[str] = None,
        timeframe: str = "1h",
        days: int = 365,
        asset_classes: List[str] = None,
    ):
        """
        Download historical OHLCV data for symbols.

        Args:
            symbols: List of symbols to download (or all if None)
            timeframe: Candle timeframe (1m, 5m, 15m, 30m, 1h, 4h, 1d, 1w)
            days: Number of days of history to download
            asset_classes: Filter by asset classes (e.g., ['forex_major', 'crypto'])
        """
        if not self.connection:
            await self.connect()

        # Get symbols if not specified
        if symbols is None:
            # Load from saved specs if available
            specs_path = self.symbols_dir / "symbol_specs.json"
            if specs_path.exists():
                with open(specs_path) as f:
                    specs = json.load(f)

                # Filter by asset class if specified
                if asset_classes:
                    specs = [s for s in specs if s['asset_class'] in asset_classes]

                symbols = [s['symbol'] for s in specs]
            else:
                symbols = await self.get_symbols()

        print(f"Downloading {timeframe} data for {len(symbols)} symbols ({days} days)...")

        # Map timeframe to MetaAPI format
        tf_map = {
            'm1': '1m', '1m': '1m', 'm5': '5m', '5m': '5m',
            'm15': '15m', '15m': '15m', 'm30': '30m', '30m': '30m',
            'h1': '1h', '1h': '1h', 'h4': '4h', '4h': '4h',
            'd1': '1d', '1d': '1d', 'w1': '1w', '1w': '1w',
        }
        mt_timeframe = tf_map.get(timeframe.lower(), timeframe)

        # Calculate start time
        end_time = datetime.now()
        start_time = end_time - timedelta(days=days)

        self.master_dir.mkdir(parents=True, exist_ok=True)

        # Chunking configuration
        CHUNK_SIZE = 1000  # Max candles per request
        MAX_RETRIES = 3
        RATE_LIMIT_DELAY = 0.2  # seconds between requests
        CONCURRENT_DOWNLOADS = 3  # parallel symbol downloads

        successful = 0
        failed = []

        # Semaphore for rate limiting concurrent downloads
        semaphore = asyncio.Semaphore(CONCURRENT_DOWNLOADS)

        async def download_symbol_data(symbol: str, idx: int) -> Tuple[str, bool, int]:
            """Download data for a single symbol with chunking."""
            async with semaphore:
                try:
                    all_candles = []
                    current_end = end_time
                    chunk_num = 0

                    # Download in chunks, going backwards in time
                    while current_end > start_time:
                        chunk_num += 1

                        # Retry logic for each chunk
                        for retry in range(MAX_RETRIES):
                            try:
                                candles = await self.account.get_historical_candles(
                                    symbol=symbol,
                                    timeframe=mt_timeframe,
                                    start_time=current_end,
                                    limit=CHUNK_SIZE,
                                )
                                break
                            except Exception as chunk_err:
                                if retry < MAX_RETRIES - 1:
                                    await asyncio.sleep(1.0 * (retry + 1))  # Exponential backoff
                                else:
                                    raise chunk_err

                        if not candles:
                            break

                        all_candles.extend(candles)

                        # Move to earlier time for next chunk
                        oldest = min(c['time'] for c in candles)
                        if oldest >= current_end:
                            break
                        current_end = oldest

                        # Rate limiting between chunks
                        await asyncio.sleep(RATE_LIMIT_DELAY)

                    if all_candles:
                        # Convert to DataFrame
                        df = pd.DataFrame(all_candles)
                        df['time'] = pd.to_datetime(df['time'])
                        df = df.sort_values('time')
                        df = df.drop_duplicates(subset=['time'])

                        # Rename columns to MT5 format
                        df = df.rename(columns={
                            'open': 'Open',
                            'high': 'High',
                            'low': 'Low',
                            'close': 'Close',
                            'tickVolume': 'Volume',
                            'spread': 'Spread',
                        })

                        # Format filename
                        start_str = df['time'].min().strftime('%Y%m%d%H%M')
                        end_str = df['time'].max().strftime('%Y%m%d%H%M')
                        tf_str = timeframe.upper()

                        # Clean symbol for filename
                        clean_symbol = symbol.replace('/', '').replace('\\', '')
                        filename = f"{clean_symbol}_{tf_str}_{start_str}_{end_str}.csv"

                        # Save atomically
                        filepath = self.master_dir / filename

                        # Add date/time columns for MT5 format
                        df['Date'] = df['time'].dt.strftime('%Y.%m.%d')
                        df['Time'] = df['time'].dt.strftime('%H:%M:%S')

                        # Save in MT5 format (atomic write)
                        output_cols = ['Date', 'Time', 'Open', 'High', 'Low', 'Close', 'Volume']
                        if 'Spread' in df.columns:
                            output_cols.append('Spread')

                        atomic_write_csv(
                            filepath,
                            df[output_cols],
                            sep='\t',
                            index=False,
                            header=['<DATE>', '<TIME>', '<OPEN>', '<HIGH>',
                                   '<LOW>', '<CLOSE>', '<TICKVOL>'] +
                                   (['<SPREAD>'] if 'Spread' in output_cols else [])
                        )

                        print(f"[{idx}/{len(symbols)}] {symbol}: OK ({len(df)} candles, {chunk_num} chunks)")
                        return symbol, True, len(df)
                    else:
                        print(f"[{idx}/{len(symbols)}] {symbol}: No data")
                        return symbol, False, 0

                except Exception as e:
                    print(f"[{idx}/{len(symbols)}] {symbol}: Error - {e}")
                    return symbol, False, 0

        # Download all symbols with concurrency
        print(f"\nDownloading with {CONCURRENT_DOWNLOADS} concurrent connections, {CHUNK_SIZE} candles per chunk...")

        tasks = [download_symbol_data(sym, i + 1) for i, sym in enumerate(symbols)]
        results = await asyncio.gather(*tasks)

        # Tally results
        for symbol, success, count in results:
            if success:
                successful += 1
            else:
                failed.append(symbol)

        print(f"\nCompleted: {successful} successful, {len(failed)} failed")
        if failed:
            print(f"Failed symbols: {', '.join(failed[:20])}")

    async def weekly_update(self):
        """
        Perform weekly update of symbol info and recent data.

        Should be run on Saturday night via cron.
        """
        print("=" * 60)
        print("WEEKLY UPDATE - MetaAPI Data Sync")
        print(f"Started: {datetime.now()}")
        print("=" * 60)

        try:
            await self.connect()

            # Update symbol specifications
            print("\n[1/3] Updating symbol specifications...")
            specs = await self.get_all_specifications()
            self.save_symbols(specs)

            # Download recent H1 data (7 days)
            print("\n[2/3] Downloading H1 data (7 days)...")
            await self.download_historical_data(timeframe='1h', days=7)

            # Download recent H4 data (30 days)
            print("\n[3/3] Downloading H4 data (30 days)...")
            await self.download_historical_data(timeframe='4h', days=30)

            print("\n" + "=" * 60)
            print("WEEKLY UPDATE COMPLETE")
            print(f"Finished: {datetime.now()}")
            print("=" * 60)

        finally:
            await self.disconnect()


async def main():
    parser = argparse.ArgumentParser(
        description='MetaAPI Cloud Data Sync',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List all connected accounts
  python scripts/metaapi_sync.py --list-accounts

  # Download symbol specifications
  python scripts/metaapi_sync.py --symbols

  # Download 1 year of H1 data
  python scripts/metaapi_sync.py --history --timeframe 1h --days 365

  # Full sync (specs + H1 + H4)
  python scripts/metaapi_sync.py --full-sync

  # Weekly update (for cron)
  python scripts/metaapi_sync.py --weekly-update
        """
    )

    parser.add_argument('--list-accounts', action='store_true',
                       help='List all MetaTrader accounts')
    parser.add_argument('--symbols', action='store_true',
                       help='Download symbol specifications')
    parser.add_argument('--history', action='store_true',
                       help='Download historical OHLCV data')
    parser.add_argument('--timeframe', default='1h',
                       help='Timeframe for historical data (1m, 5m, 15m, 30m, 1h, 4h, 1d)')
    parser.add_argument('--days', type=int, default=365,
                       help='Days of history to download')
    parser.add_argument('--asset-classes', nargs='+',
                       help='Filter by asset classes (forex_major, crypto, etc.)')
    parser.add_argument('--full-sync', action='store_true',
                       help='Full sync (symbols + all timeframe data)')
    parser.add_argument('--weekly-update', action='store_true',
                       help='Weekly update mode for cron job')
    parser.add_argument('--account-id', help='Specific account ID to use')

    args = parser.parse_args()

    # List accounts mode
    if args.list_accounts:
        sync = MetaAPISync()
        await sync.list_accounts()
        return

    # Create sync instance
    sync = MetaAPISync(account_id=args.account_id)

    try:
        # Weekly update mode
        if args.weekly_update:
            await sync.weekly_update()
            return

        # Connect
        await sync.connect()

        # Download symbols
        if args.symbols or args.full_sync:
            specs = await sync.get_all_specifications()
            sync.save_symbols(specs)

        # Download history
        if args.history or args.full_sync:
            timeframes = ['1h', '4h'] if args.full_sync else [args.timeframe]
            for tf in timeframes:
                await sync.download_historical_data(
                    timeframe=tf,
                    days=args.days,
                    asset_classes=args.asset_classes,
                )

    finally:
        await sync.disconnect()


if __name__ == "__main__":
    # Install python-dotenv if needed
    try:
        from dotenv import load_dotenv
    except ImportError:
        import subprocess
        subprocess.run([sys.executable, '-m', 'pip', 'install', 'python-dotenv', '-q'])
        from dotenv import load_dotenv

    asyncio.run(main())
