"""
MetaAPI Bulk Data Download
==========================
Downloads data for all asset classes, organizes properly,
with atomic saves and proper naming conventions.
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import asyncio
import json
import tempfile
import shutil

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from metaapi_cloud_sdk import MetaApi
except ImportError:
    print("Install: pip install metaapi-cloud-sdk")
    sys.exit(1)

# =====================================================
# CREDENTIALS
# =====================================================
API_TOKEN = "eyJhbGciOiJSUzUxMiIsInR5cCI6IkpXVCJ9.eyJfaWQiOiJjMTdhODAwNThhOWE3OWE0NDNkZjBlOGM1NDZjZjlmMSIsImFjY2Vzc1J1bGVzIjpbeyJpZCI6InRyYWRpbmctYWNjb3VudC1tYW5hZ2VtZW50LWFwaSIsIm1ldGhvZHMiOlsidHJhZGluZy1hY2NvdW50LW1hbmFnZW1lbnQtYXBpOnJlc3Q6cHVibGljOio6KiJdLCJyb2xlcyI6WyJyZWFkZXIiLCJ3cml0ZXIiXSwicmVzb3VyY2VzIjpbIio6JFVTRVJfSUQkOioiXX0seyJpZCI6Im1ldGFhcGktcmVzdC1hcGkiLCJtZXRob2RzIjpbIm1ldGFhcGktYXBpOnJlc3Q6cHVibGljOio6KiJdLCJyb2xlcyI6WyJyZWFkZXIiLCJ3cml0ZXIiXSwicmVzb3VyY2VzIjpbIio6JFVTRVJfSUQkOioiXX0seyJpZCI6Im1ldGFhcGktcnBjLWFwaSIsIm1ldGhvZHMiOlsibWV0YWFwaS1hcGk6d3M6cHVibGljOio6KiJdLCJyb2xlcyI6WyJyZWFkZXIiLCJ3cml0ZXIiXSwicmVzb3VyY2VzIjpbIio6JFVTRVJfSUQkOioiXX0seyJpZCI6Im1ldGFhcGktcmVhbC10aW1lLXN0cmVhbWluZy1hcGkiLCJtZXRob2RzIjpbIm1ldGFhcGktYXBpOndzOnB1YmxpYzoqOioiXSwicm9sZXMiOlsicmVhZGVyIiwid3JpdGVyIl0sInJlc291cmNlcyI6WyIqOiRVU0VSX0lEJDoqIl19LHsiaWQiOiJtZXRhc3RhdHMtYXBpIiwibWV0aG9kcyI6WyJtZXRhc3RhdHMtYXBpOnJlc3Q6cHVibGljOio6KiJdLCJyb2xlcyI6WyJyZWFkZXIiLCJ3cml0ZXIiXSwicmVzb3VyY2VzIjpbIio6JFVTRVJfSUQkOioiXX0seyJpZCI6InJpc2stbWFuYWdlbWVudC1hcGkiLCJtZXRob2RzIjpbInJpc2stbWFuYWdlbWVudC1hcGk6cmVzdDpwdWJsaWM6KjoqIl0sInJvbGVzIjpbInJlYWRlciIsIndyaXRlciJdLCJyZXNvdXJjZXMiOlsiKjokVVNFUl9JRCQ6KiJdfSx7ImlkIjoiY29weWZhY3RvcnktYXBpIiwibWV0aG9kcyI6WyJjb3B5ZmFjdG9yeS1hcGk6cmVzdDpwdWJsaWM6KjoqIl0sInJvbGVzIjpbInJlYWRlciIsIndyaXRlciJdLCJyZXNvdXJjZXMiOlsiKjokVVNFUl9JRCQ6KiJdfSx7ImlkIjoibXQtbWFuYWdlci1hcGkiLCJtZXRob2RzIjpbIm10LW1hbmFnZXItYXBpOnJlc3Q6ZGVhbGluZzoqOioiLCJtdC1tYW5hZ2VyLWFwaTpyZXN0OnB1YmxpYzoqOioiXSwicm9sZXMiOlsicmVhZGVyIiwid3JpdGVyIl0sInJlc291cmNlcyI6WyIqOiRVU0VSX0lEJDoqIl19LHsiaWQiOiJiaWxsaW5nLWFwaSIsIm1ldGhvZHMiOlsiYmlsbGluZy1hcGk6cmVzdDpwdWJsaWM6KjoqIl0sInJvbGVzIjpbInJlYWRlciJdLCJyZXNvdXJjZXMiOlsiKjokVVNFUl9JRCQ6KiJdfV0sImlnbm9yZVJhdGVMaW1pdHMiOmZhbHNlLCJ0b2tlbklkIjoiMjAyMTAyMTMiLCJpbXBlcnNvbmF0ZWQiOmZhbHNlLCJyZWFsVXNlcklkIjoiYzE3YTgwMDU4YTlhNzlhNDQzZGYwZThjNTQ2Y2Y5ZjEiLCJpYXQiOjE3NjcxMjY5NzQsImV4cCI6MTc3NDkwMjk3NH0.MNG5qH4ufgoKivTCTuvfVywtTYgYhkIEWCLoff9F1tP3MvGLNRhHNwe2dyMppSTr5mzEFlkF1VRlpFthpq2KnOUvCATFNUM04cUYJcpcv6Arp_Pf653Lrtm1DK2Br4NZYQr9eh_ZndXIN2qm2QYSAi2W5wXovAaMkLPjs1x2J1G4ZxFM48u7xrqCci0Sri2dhCLNI6eVX9-VlfLJb4iYJqbKcS7GacodmtUHQqzKKusazLPoEe0cJmVPVj0h5OwXiWnZRH07VY9e9s3i-5BzHp9syGVDh7rU3D7IU8jCaB8oBWl6S49MW-wpY41_cdxf3eo53CN0MY3GikfZbusgO_2xAxxBfbsmMIC9l0g2TiUIuATEfMILPzcAhCjKE35AAc0JEbXw0XxBWyIZoCAcdqI2FuyyMOyddKfSQ7y7kkW_0tu5d9P8p-HUdE5FEI_rEHbfxfEy4CLI9LY_5ZycuhZwrnOyKLS_CPX4iFtdTT40eHynaeNv8ok8_h_wirm5YQuFv_YL0u0HqTqiy5Q_f-vDJVLob7et779DsBj9myILCFGg7RlwzEcxsZNGCkbNRvsCjZE7HwqQy2IjqGNo5vI8AiEfHD0c3PGfPhdKqKS5mBa7w4md-90T_Um9VnUXHZ8EnQlrVCnP8NpsfCWGQRDPMepd_D1lvL6XjxaVMfM"

ACCOUNT_ID = "e8f8c21a-32b5-40b0-9bf7-672e8ffab91f"

# =====================================================
# PREFERRED SYMBOLS BY ASSET CLASS (fallback list)
# =====================================================
PREFERRED_SYMBOLS = {
    'forex': ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'EURJPY', 'GBPJPY', 'USDCHF', 'USDCAD', 'NZDUSD', 'EURGBP'],
    'crypto': ['BTCUSD', 'ETHUSD', 'LTCUSD', 'XRPUSD', 'BCHUSD', 'ADAUSD', 'DOTUSD', 'SOLUSD', 'BNBUSD', 'LINKUSD'],
    'indices': ['US30', 'US500', 'NAS100', 'GER40', 'UK100', 'JP225', 'AUS200', 'FRA40', 'EU50', 'HK50'],
    'metals': ['XAUUSD', 'XAGUSD', 'XPTUSD', 'XPDUSD', 'COPPER', 'XAUEUR', 'XAUAUD', 'XAGEUR'],
    'energy': ['USOIL', 'UKOIL', 'NGAS', 'BRENT', 'WTI', 'HEATING', 'NATGAS'],
}

# Pattern matchers for auto-discovery
ASSET_CLASS_PATTERNS = {
    'forex': [
        # Major pairs
        r'^(EUR|GBP|USD|JPY|CHF|CAD|AUD|NZD)(EUR|GBP|USD|JPY|CHF|CAD|AUD|NZD)\+?$',
    ],
    'crypto': [
        r'^(BTC|ETH|LTC|XRP|BCH|ADA|DOT|SOL|BNB|LINK|DOGE|AVAX)(USD|EUR|JPY)\+?$',
        r'^(BITCOIN|ETHEREUM)',
    ],
    'indices': [
        r'^(US30|US500|US2000|NAS100|DJ30|SP500|SPX)',
        r'^(GER40|GER30|DAX|UK100|FTSE|JP225|Nikkei|NI225)',
        r'^(AUS200|FRA40|EU50|HK50|CHINA)',
    ],
    'metals': [
        r'^(XAU|XAG|XPT|XPD|GOLD|SILVER|PLATINUM|PALLADIUM)',
        r'^COPPER',
    ],
    'energy': [
        r'^(USOIL|UKOIL|WTI|BRENT|NGAS|NATGAS|OIL|CRUDE)',
        r'^(CL|BRN|NG)',
    ],
}

# Timeframes to download
TIMEFRAMES = ['1h', '4h']  # H1 and H4


def classify_symbol(symbol: str) -> str:
    """Classify a symbol into an asset class based on patterns."""
    import re
    symbol_upper = symbol.upper().rstrip('+')  # Remove ECN suffix for matching

    for asset_class, patterns in ASSET_CLASS_PATTERNS.items():
        for pattern in patterns:
            if re.match(pattern, symbol_upper, re.IGNORECASE):
                return asset_class
    return None


def discover_available_symbols(available: list, max_per_class: int = 6) -> dict:
    """
    Intelligently discover and categorize available symbols.

    Returns dict of {asset_class: [(symbol, is_ecn), ...]}
    """
    discovered = {cls: [] for cls in ASSET_CLASS_PATTERNS.keys()}

    for symbol in available:
        asset_class = classify_symbol(symbol)
        if asset_class:
            is_ecn = symbol.endswith('+')
            discovered[asset_class].append((symbol, is_ecn))

    # Sort each class: ECN first, then alphabetically
    for cls in discovered:
        discovered[cls].sort(key=lambda x: (not x[1], x[0]))
        # Limit to max_per_class
        discovered[cls] = discovered[cls][:max_per_class]

    return discovered


def atomic_write_csv(df: pd.DataFrame, filepath: Path, sep: str = '\t'):
    """Write CSV atomically - write to temp file then rename."""
    filepath.parent.mkdir(parents=True, exist_ok=True)

    # Write to temp file in same directory
    temp_fd, temp_path = tempfile.mkstemp(
        suffix='.csv.tmp',
        dir=filepath.parent,
        prefix=filepath.stem + '_'
    )
    try:
        df.to_csv(temp_path, index=False, sep=sep)
        # Atomic rename
        shutil.move(temp_path, filepath)
    except Exception as e:
        # Clean up temp file on error
        if Path(temp_path).exists():
            Path(temp_path).unlink()
        raise e


# Symbol aliases - map common names to broker-specific symbols
SYMBOL_ALIASES = {
    # Indices
    'US30': ['DJ30', 'DJI30', 'DJ30ft', 'DOW30', 'DJIA', 'US30Cash'],
    'US500': ['SP500', 'SPX500', 'S&P500', 'US500Cash', 'SPX'],
    'JP225': ['Nikkei225', 'JPN225', 'NI225', 'NIKKEI', 'JP225Cash'],
    'NAS100': ['NASDAQ', 'NDX100', 'USTEC', 'NAS100Cash'],
    'GER40': ['DAX40', 'DE40', 'GER30', 'DAX', 'GER40Cash'],
    'UK100': ['FTSE100', 'UK100Cash', 'FTSE', 'UKX'],
    # Energy
    'USOIL': ['WTI', 'CRUDEOIL', 'USOUSD', 'CL', 'OIL', 'OILUSD', 'WTIUSD'],
    'UKOIL': ['BRENT', 'UKOUSD', 'BRN', 'BRENTOIL', 'BRNUSD'],
    'NGAS': ['NATGAS', 'NATURALGAS', 'NG', 'NGASUSD'],
    'BRENT': ['UKOIL', 'UKOUSD', 'BRN', 'BRNUSD'],
    'WTI': ['USOIL', 'WTIUSD', 'CL', 'USOUSD'],
    'HEATING': ['HO', 'HEATINGOIL', 'HOUSD'],
    # Crypto alternatives
    'BTCUSD': ['BITCOIN', 'BTC', 'XBTUSD'],
    'ETHUSD': ['ETHEREUM', 'ETH'],
    # Metals
    'COPPER': ['COPPER-C', 'HG', 'XCUUSD'],
}


def find_symbol_match(target: str, available: list, prefer_ecn: bool = True) -> str:
    """
    Find matching symbol from available list with comprehensive alias support.

    prefer_ecn: If True, prefer ECN symbols (+ suffix) for tighter spreads
    """
    target_upper = target.upper()

    # ECN suffixes first (tighter spreads), then standard
    if prefer_ecn:
        suffix_order = ['+', '', 'm', '.pro', '_SB', 'Cash', '-C', 'ft']
    else:
        suffix_order = ['', '+', 'm', '.pro', '_SB', 'Cash', '-C', 'ft']

    # Try exact match first (with ECN preference)
    for suffix in suffix_order:
        candidate = target + suffix
        if candidate in available:
            return candidate
        # Case insensitive
        matches = [s for s in available if s.upper() == (target_upper + suffix.upper())]
        if matches:
            return matches[0]

    # Try aliases (with ECN preference)
    aliases = SYMBOL_ALIASES.get(target_upper, [])
    for alias in aliases:
        for suffix in suffix_order:
            candidate = alias + suffix
            if candidate in available:
                return candidate
            matches = [s for s in available if s.upper() == (alias.upper() + suffix.upper())]
            if matches:
                return matches[0]

    # Try partial match (last resort) - prefer ECN if multiple matches
    matches = [s for s in available if target_upper in s.upper()]
    if matches:
        # Sort: ECN first (+), then by length (shorter = more specific)
        matches.sort(key=lambda x: ('+' not in x, len(x)))
        return matches[0]

    return None


# Concurrency control with dynamic throttling
MAX_CONCURRENT_DOWNLOADS = 4  # Initial parallel downloads
MIN_CONCURRENT_DOWNLOADS = 1  # Minimum when rate limited

class DynamicThrottler:
    """Adjusts concurrency based on rate limit responses."""

    def __init__(self, initial: int = 4, min_val: int = 1, max_val: int = 6):
        self.current = initial
        self.min_val = min_val
        self.max_val = max_val
        self.semaphore = asyncio.Semaphore(initial)
        self.rate_limit_count = 0
        self.success_count = 0
        self._lock = asyncio.Lock()

    async def on_rate_limit(self):
        """Reduce concurrency when rate limited."""
        async with self._lock:
            self.rate_limit_count += 1
            if self.current > self.min_val:
                self.current -= 1
                print(f"\n    ⚡ Throttling down to {self.current} concurrent", flush=True)

    async def on_success(self):
        """Consider increasing concurrency after successes."""
        async with self._lock:
            self.success_count += 1
            # Every 10 successes without rate limit, try increasing
            if self.success_count >= 10 and self.rate_limit_count == 0:
                if self.current < self.max_val:
                    self.current += 1
                    print(f"\n    ⚡ Throttling up to {self.current} concurrent", flush=True)
                self.success_count = 0

    async def acquire(self):
        await self.semaphore.acquire()

    def release(self):
        self.semaphore.release()


async def download_one(
    account, symbol: str, tf: str, asset_class: str, original: str,
    start_time, end_time, throttler: DynamicThrottler, task_id: int, total_tasks: int
) -> dict:
    """Download a single symbol/timeframe with dynamic throttling."""
    from datetime import timezone

    await throttler.acquire()
    try:
        tf_label = 'H1' if tf == '1h' else 'H4'
        output_dir = project_root / "data" / "master" / asset_class
        output_dir.mkdir(parents=True, exist_ok=True)

        # Check for existing file (resume capability)
        existing = list(output_dir.glob(f"{symbol}_{tf_label}_*.csv"))
        if existing:
            try:
                existing_df = pd.read_csv(existing[0], sep='\t')
                existing_df.columns = [c.lower().replace('<', '').replace('>', '') for c in existing_df.columns]
                last_date = pd.to_datetime(existing_df['date'] + ' ' + existing_df['time']).max()
                last_date = last_date.replace(tzinfo=timezone.utc)
                chunk_start = last_date + timedelta(hours=1 if tf == '1h' else 4)
                print(f"  [{task_id}/{total_tasks}] {symbol} {tf_label} (resume)...", end=" ", flush=True)
            except Exception:
                chunk_start = start_time
                existing_df = None
                print(f"  [{task_id}/{total_tasks}] {symbol} {tf_label}...", end=" ", flush=True)
        else:
            chunk_start = start_time
            existing_df = None
            print(f"  [{task_id}/{total_tasks}] {symbol} {tf_label}...", end=" ", flush=True)

        try:
            # Chunk through history
            all_candles = []
            chunk_end = end_time
            max_retries = 5
            backoff = 1.0

            while chunk_start < chunk_end:
                for retry in range(max_retries):
                    try:
                        candles = await account.get_historical_candles(
                            symbol=symbol,
                            timeframe=tf,
                            start_time=chunk_start,
                            limit=1000
                        )
                        break
                    except Exception as e:
                        if 'rate' in str(e).lower() or '429' in str(e):
                            await throttler.on_rate_limit()
                            wait = backoff * (2 ** retry)
                            await asyncio.sleep(wait)
                        else:
                            raise

                if not candles:
                    break

                all_candles.extend(candles)

                # Move to next chunk
                last_time = pd.to_datetime(candles[-1]['time'])
                if last_time.tzinfo is None:
                    last_time = last_time.replace(tzinfo=timezone.utc)
                chunk_start = last_time + timedelta(hours=1 if tf == '1h' else 4)

                await asyncio.sleep(0.3)  # Rate limit between chunks

            if not all_candles or len(all_candles) < 100:
                print(f"⚠️ {len(all_candles) if all_candles else 0} bars")
                return {'status': 'failed', 'symbol': symbol, 'tf': tf_label, 'error': 'insufficient data'}

            # Convert to DataFrame
            df = pd.DataFrame(all_candles)
            df['time'] = pd.to_datetime(df['time'])
            df = df.drop_duplicates(subset=['time']).sort_values('time')

            # Naming convention
            start_str = df['time'].iloc[0].strftime('%Y%m%d%H%M')
            end_str = df['time'].iloc[-1].strftime('%Y%m%d%H%M')
            filename = f"{symbol}_{tf_label}_{start_str}_{end_str}.csv"
            output_file = output_dir / filename

            # Remove old files
            for old in existing:
                if old != output_file:
                    old.unlink()

            # Prepare export format
            df_export = pd.DataFrame({
                '<DATE>': df['time'].dt.strftime('%Y.%m.%d'),
                '<TIME>': df['time'].dt.strftime('%H:%M:%S'),
                '<OPEN>': df['open'],
                '<HIGH>': df['high'],
                '<LOW>': df['low'],
                '<CLOSE>': df['close'],
                '<TICKVOL>': df['tickVolume'],
            })

            atomic_write_csv(df_export, output_file, sep='\t')
            print(f"✅ {len(df):,}")

            await throttler.on_success()
            return {
                'status': 'success',
                'symbol': symbol,
                'original': original,
                'asset_class': asset_class,
                'timeframe': tf_label,
                'bars': len(df),
                'start': str(df['time'].iloc[0]),
                'end': str(df['time'].iloc[-1]),
                'file': str(output_file),
            }

        except Exception as e:
            print(f"❌ {e}")
            return {'status': 'failed', 'symbol': symbol, 'tf': tf_label, 'error': str(e)}
    finally:
        throttler.release()


async def download_all():
    """Download H1 and H4 data for 6 instruments per asset class (PARALLEL)."""

    print("\n" + "="*70)
    print("METAAPI BULK DATA DOWNLOAD (PARALLEL)")
    print("="*70)
    print(f"  Asset classes: {len(PREFERRED_SYMBOLS)}")
    print(f"  Instruments per class: 6")
    print(f"  Timeframes: H1, H4")
    print(f"  History: 2 years")
    print(f"  Concurrency: {MAX_CONCURRENT_DOWNLOADS} parallel downloads")

    # Connect
    print("\n[1] Connecting to MetaAPI...")
    api = MetaApi(API_TOKEN)
    accounts = await api.metatrader_account_api.get_accounts_with_infinite_scroll_pagination()
    account = next((a for a in accounts if a.id == ACCOUNT_ID), accounts[0])

    if getattr(account, 'state', None) != 'DEPLOYED':
        await account.deploy()
    await account.wait_connected()
    print(f"    ✅ Connected: {account.name}")

    connection = account.get_rpc_connection()
    await connection.connect()
    await connection.wait_synchronized()
    print("    ✅ Synchronized")

    # Get available symbols
    print("\n[2] Discovering available symbols...")
    available_symbols = await connection.get_symbols()
    print(f"    ✅ {len(available_symbols)} total symbols available")

    # Intelligent auto-discovery: classify and select best symbols per class
    discovered = discover_available_symbols(available_symbols, max_per_class=6)

    print("\n    AUTO-DISCOVERED SYMBOLS (ECN preferred):")
    symbols_to_download = []
    for asset_class in ['forex', 'crypto', 'indices', 'metals', 'energy']:
        class_symbols = discovered.get(asset_class, [])
        if class_symbols:
            symbol_list = ', '.join([f"{s}{'*' if ecn else ''}" for s, ecn in class_symbols])
            print(f"    {asset_class}: {symbol_list}")
            for symbol, is_ecn in class_symbols:
                symbols_to_download.append((symbol, asset_class, symbol))
        else:
            # Fallback to preferred list with matching
            print(f"    {asset_class}: [fallback matching...]")
            for target in PREFERRED_SYMBOLS.get(asset_class, [])[:6]:
                match = find_symbol_match(target, available_symbols)
                if match:
                    symbols_to_download.append((match, asset_class, target))
                    print(f"      ✓ {target} → {match}")
                else:
                    print(f"      ✗ {target} → NOT FOUND")

    print(f"\n    * = ECN (tighter spreads)")
    print(f"    Total: {len(symbols_to_download)} symbols × {len(TIMEFRAMES)} timeframes = {len(symbols_to_download) * len(TIMEFRAMES)} files")

    # Date range (2 years) - use UTC for consistency with MetaAPI
    from datetime import timezone
    end_time = datetime.now(timezone.utc)
    start_time = end_time - timedelta(days=730)

    # Download with PARALLEL execution
    print(f"\n[3] Downloading ({start_time.date()} to {end_time.date()})...")

    total_tasks = len(symbols_to_download) * len(TIMEFRAMES)
    print(f"    Launching {total_tasks} downloads with dynamic throttling...")

    # Create dynamic throttler
    throttler = DynamicThrottler(initial=MAX_CONCURRENT_DOWNLOADS, min_val=1, max_val=6)

    # Build task list
    tasks = []
    task_id = 0
    for symbol, asset_class, original in symbols_to_download:
        for tf in TIMEFRAMES:
            task_id += 1
            tasks.append(
                download_one(
                    account, symbol, tf, asset_class, original,
                    start_time, end_time, throttler, task_id, total_tasks
                )
            )

    # Run all downloads in parallel (throttler limits concurrency)
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Process results
    downloaded = []
    failed = []
    for result in results:
        if isinstance(result, Exception):
            failed.append(('unknown', 'unknown', str(result)))
        elif result.get('status') == 'success':
            downloaded.append(result)
        else:
            failed.append((result.get('symbol', '?'), result.get('tf', '?'), result.get('error', 'unknown')))

    # Cleanup
    await connection.close()

    # Summary
    print("\n" + "="*70)
    print("DOWNLOAD SUMMARY")
    print("="*70)

    print(f"\n  ✅ Downloaded: {len(downloaded)} files")
    print(f"  ❌ Failed: {len(failed)} files")

    if downloaded:
        total_bars = sum(d['bars'] for d in downloaded)
        print(f"  📊 Total bars: {total_bars:,}")

        # By asset class
        print("\n  By asset class:")
        by_class = {}
        for d in downloaded:
            key = d['asset_class']
            by_class.setdefault(key, {'h1': 0, 'h4': 0, 'bars': 0})
            by_class[key]['bars'] += d['bars']
            if d['timeframe'] == 'H1':
                by_class[key]['h1'] += 1
            else:
                by_class[key]['h4'] += 1

        for ac in sorted(by_class.keys()):
            stats = by_class[ac]
            print(f"    {ac}: {stats['h1']} H1 + {stats['h4']} H4 = {stats['bars']:,} bars")

    # Save manifest
    manifest_file = project_root / "data" / "master" / "download_manifest.json"
    manifest = {
        'downloaded_at': datetime.now().isoformat(),
        'broker': 'Vantage',
        'total_files': len(downloaded),
        'total_bars': sum(d['bars'] for d in downloaded) if downloaded else 0,
        'symbols': downloaded,
        'failed': [{'symbol': s, 'tf': t, 'error': e} for s, t, e in failed],
    }

    with open(manifest_file, 'w') as f:
        json.dump(manifest, f, indent=2)
    print(f"\n  📄 Manifest: {manifest_file}")

    print("\n" + "="*70)
    print("DATA STRUCTURE:")
    print("="*70)
    print("""
    data/master/
    ├── forex/
    │   ├── EURUSD_H1_*.csv
    │   ├── EURUSD_H4_*.csv
    │   └── ...
    ├── crypto/
    │   ├── BTCUSD_H1_*.csv
    │   └── ...
    ├── indices/
    ├── metals/
    └── energy/
    """)

    print("\n✅ READY FOR PHYSICS EXPLORATION!")
    print("   Run: python rl_exploration_framework.py --data-dir data/master")

    return downloaded


if __name__ == "__main__":
    asyncio.run(download_all())
