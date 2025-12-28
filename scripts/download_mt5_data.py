#!/usr/bin/env python3
"""
Download historical data from MT5 Terminal

Requires:
- MetaTrader 5 terminal running (Windows or Wine)
- pip install MetaTrader5

Usage:
    python scripts/download_mt5_data.py

Instruments to download (customizable):
- Crypto: BTCUSD, ETHUSD
- Forex: EURUSD, GBPUSD, USDJPY
- Commodities: XAUUSD (Gold), COPPER-C
- Indices: US500, US30
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta

sys.path.insert(0, str(Path(__file__).parent.parent))

from kinetra.mt5_connector import MT5Connector, MT5_AVAILABLE

# Configuration
INSTRUMENTS = [
    # Crypto
    "BTCUSD", "ETHUSD", "LTCUSD",
    # Forex majors
    "EURUSD", "GBPUSD", "USDJPY", "AUDUSD",
    # Commodities
    "XAUUSD", "COPPER-C", "USOIL",
    # Indices
    "US500", "US30", "GER40",
]

TIMEFRAMES = ["M30", "H1", "H4", "D1"]

# How much history to download
BARS_TO_DOWNLOAD = {
    "M30": 50000,   # ~2 years
    "H1": 30000,    # ~3.5 years
    "H4": 10000,    # ~5 years
    "D1": 5000,     # ~20 years
}


def download_all():
    """Download data for all configured instruments and timeframes."""

    if not MT5_AVAILABLE:
        print("ERROR: MetaTrader5 package not installed!")
        print("Install with: pip install MetaTrader5")
        print("\nNote: Requires Windows or Wine on Linux")
        return False

    connector = MT5Connector()

    print("=" * 70)
    print("MT5 DATA DOWNLOADER")
    print("=" * 70)

    if not connector.connect():
        print("\nERROR: Failed to connect to MT5 terminal!")
        print("Make sure MetaTrader 5 is running.")
        return False

    print(f"\nConnected to MT5!")
    account = connector.get_account_info()
    if account:
        print(f"Account: {account['login']}")
        print(f"Balance: {account['balance']} {account['currency']}")

    # Get available symbols
    available_symbols = set(connector.get_symbols())
    print(f"\nTotal symbols available: {len(available_symbols)}")

    project_root = Path(__file__).parent.parent
    downloaded = []
    failed = []

    for symbol in INSTRUMENTS:
        # Try exact match or with suffix (brokers add suffixes like .r, .raw)
        actual_symbol = None
        if symbol in available_symbols:
            actual_symbol = symbol
        else:
            # Try common suffixes
            for suffix in ['', '.r', '.raw', 'm', 'micro', '_SB']:
                test = symbol + suffix
                if test in available_symbols:
                    actual_symbol = test
                    break

        if not actual_symbol:
            print(f"\n⚠ Symbol not found: {symbol}")
            failed.append((symbol, "not found"))
            continue

        for tf in TIMEFRAMES:
            bars = BARS_TO_DOWNLOAD.get(tf, 10000)
            print(f"\nDownloading {actual_symbol} {tf} ({bars} bars)...")

            df = connector.get_ohlcv(actual_symbol, tf, count=bars)

            if df is None or len(df) == 0:
                print(f"  ✗ Failed to get data")
                failed.append((f"{symbol}_{tf}", "no data"))
                continue

            # Save to CSV
            filename = f"{symbol}_{tf}_{df['time'].iloc[0].strftime('%Y%m%d%H%M')}_{df['time'].iloc[-1].strftime('%Y%m%d%H%M')}.csv"
            filepath = project_root / filename

            df.to_csv(filepath, index=False, sep='\t')

            print(f"  ✓ Saved {len(df)} bars to {filename}")
            downloaded.append(filename)

    connector.disconnect()

    print("\n" + "=" * 70)
    print("DOWNLOAD SUMMARY")
    print("=" * 70)
    print(f"\nDownloaded: {len(downloaded)} files")
    for f in downloaded:
        print(f"  ✓ {f}")

    if failed:
        print(f"\nFailed: {len(failed)} items")
        for name, reason in failed:
            print(f"  ✗ {name}: {reason}")

    return True


def list_available_symbols():
    """List all available symbols from MT5."""
    if not MT5_AVAILABLE:
        print("MetaTrader5 package not installed")
        return

    connector = MT5Connector()
    if not connector.connect():
        print("Failed to connect to MT5")
        return

    symbols = connector.get_symbols()
    print(f"\nAvailable symbols ({len(symbols)}):")

    # Group by category
    crypto = [s for s in symbols if 'BTC' in s or 'ETH' in s or 'LTC' in s or 'XRP' in s]
    forex = [s for s in symbols if s.endswith('USD') and s not in crypto and 'XAU' not in s]
    commodities = [s for s in symbols if 'XAU' in s or 'COPPER' in s or 'OIL' in s or 'GAS' in s]
    indices = [s for s in symbols if any(x in s for x in ['US500', 'US30', 'GER', 'UK100', 'JPN', 'NDX'])]

    print("\nCrypto:")
    for s in sorted(crypto)[:20]:
        print(f"  {s}")

    print("\nForex (major pairs):")
    for s in sorted(forex)[:20]:
        print(f"  {s}")

    print("\nCommodities:")
    for s in sorted(commodities)[:20]:
        print(f"  {s}")

    print("\nIndices:")
    for s in sorted(indices)[:20]:
        print(f"  {s}")

    connector.disconnect()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download MT5 historical data")
    parser.add_argument("--list", action="store_true", help="List available symbols")
    args = parser.parse_args()

    if args.list:
        list_available_symbols()
    else:
        download_all()
