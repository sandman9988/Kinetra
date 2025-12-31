"""
MetaAPI Bulk Data Download
==========================
Downloads data for all asset classes, organizes properly,
with atomic saves and proper naming conventions.
"""

import sys
import os
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

# Load environment variables from .env file if available
try:
    from dotenv import load_dotenv
    load_dotenv(project_root / '.env')
except ImportError:
    pass

# =====================================================
# CREDENTIALS - Load from environment variables
# =====================================================
API_TOKEN = os.environ.get('METAAPI_TOKEN')
ACCOUNT_ID = os.environ.get('METAAPI_ACCOUNT_ID')

if not API_TOKEN or not ACCOUNT_ID:
    print("ERROR: Missing credentials!")
    print("Please set METAAPI_TOKEN and METAAPI_ACCOUNT_ID environment variables.")
    print("You can:")
    print("  1. Copy .env.example to .env and fill in your credentials")
    print("  2. Export them: export METAAPI_TOKEN=your_token")
    print("                 export METAAPI_ACCOUNT_ID=your_account_id")
    sys.exit(1)

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


# Concurrency control - max out 32-thread AMD 5950
MAX_CONCURRENT_DOWNLOADS = 32  # Network I/O bound, throttler backs off on rate limits


class ProgressTracker:
    """Global progress tracker with heartbeat."""

    def __init__(self, total_tasks: int):
        self.total_tasks = total_tasks
        self.completed = 0
        self.in_progress = 0
        self.chunks_fetched = 0
        self.bars_fetched = 0
        self.start_time = datetime.now()
        self._lock = asyncio.Lock()

    async def start_task(self):
        async with self._lock:
            self.in_progress += 1

    async def complete_task(self, bars: int = 0):
        async with self._lock:
            self.in_progress -= 1
            self.completed += 1
            self.bars_fetched += bars

    async def chunk_done(self, bars: int):
        async with self._lock:
            self.chunks_fetched += 1
            self.bars_fetched += bars

    def status_line(self) -> str:
        elapsed = (datetime.now() - self.start_time).total_seconds()
        rate = self.bars_fetched / elapsed if elapsed > 0 else 0
        pct = (self.completed / self.total_tasks * 100) if self.total_tasks > 0 else 0
        return f"[{self.completed}/{self.total_tasks}] {pct:.0f}% | 🔄{self.in_progress} parallel | {self.bars_fetched:,} bars @ {rate:.0f}/sec"


class DynamicThrottler:
    """Simple bounded semaphore with rate limit detection."""

    def __init__(self, max_concurrent: int = 6):
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.rate_limited = False
        self._lock = asyncio.Lock()

    async def on_rate_limit(self):
        """Signal rate limit hit - adds global delay."""
        async with self._lock:
            if not self.rate_limited:
                self.rate_limited = True
                print(f"\n    ⚡ Rate limit detected - adding delays", flush=True)

    async def acquire(self):
        await self.semaphore.acquire()
        # If rate limited, add small delay before each request
        if self.rate_limited:
            await asyncio.sleep(0.5)

    def release(self):
        self.semaphore.release()


async def download_one(
    account, symbol: str, tf: str, asset_class: str, original: str,
    start_time, end_time, throttler: DynamicThrottler, progress: ProgressTracker
) -> dict:
    """Download a single symbol/timeframe with dynamic throttling and heartbeat."""
    from datetime import timezone

    await throttler.acquire()
    await progress.start_task()
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
            except Exception:
                chunk_start = start_time
                existing_df = None
        else:
            chunk_start = start_time
            existing_df = None

        try:
            # Chunk through history
            all_candles = []
            chunk_end = end_time
            max_retries = 5
            backoff = 1.0
            chunk_num = 0

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
                chunk_num += 1

                # HEARTBEAT: Print progress every chunk
                print(f"\r  💓 {progress.status_line()} | {symbol} {tf_label} chunk {chunk_num}", end="", flush=True)

                # Move to next chunk
                last_time = pd.to_datetime(candles[-1]['time'])
                if last_time.tzinfo is None:
                    last_time = last_time.replace(tzinfo=timezone.utc)
                chunk_start = last_time + timedelta(hours=1 if tf == '1h' else 4)

                await asyncio.sleep(0.05)  # Faster chunk delay

            if not all_candles or len(all_candles) < 100:
                print(f"\n  ⚠️ {symbol} {tf_label}: only {len(all_candles) if all_candles else 0} bars")
                await progress.complete_task(0)
                return {'status': 'failed', 'symbol': symbol, 'tf': tf_label, 'error': 'insufficient data'}

            # Process and write in thread pool (non-blocking)
            def process_and_write():
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
                return len(df), output_file

            # Run in thread pool - doesn't block other downloads
            bar_count, output_file = await asyncio.to_thread(process_and_write)
            await progress.complete_task(bar_count)
            print(f"\n  ✅ {symbol} {tf_label}: {bar_count:,} bars saved")

            return {
                'status': 'success',
                'symbol': symbol,
                'original': original,
                'asset_class': asset_class,
                'timeframe': tf_label,
                'bars': bar_count,
                'file': str(output_file),
            }

        except Exception as e:
            await progress.complete_task(0)
            print(f"\n  ❌ {symbol} {tf_label}: {e}")
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
    print(f"    💓 Heartbeat shows real-time progress\n")

    # Create throttler and progress tracker
    throttler = DynamicThrottler(max_concurrent=MAX_CONCURRENT_DOWNLOADS)
    progress = ProgressTracker(total_tasks)

    # Build task list
    tasks = []
    for symbol, asset_class, original in symbols_to_download:
        for tf in TIMEFRAMES:
            tasks.append(
                download_one(
                    account, symbol, tf, asset_class, original,
                    start_time, end_time, throttler, progress
                )
            )

    # Run all downloads in parallel (throttler limits concurrency)
    results = await asyncio.gather(*tasks, return_exceptions=True)
    print()  # Newline after heartbeat

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
