#!/usr/bin/env python3
"""
Extract MT5 Symbol Specifications

Pulls complete contract specifications from MT5 terminal and saves to JSON.
Run this periodically to keep specs up-to-date with broker changes.

Usage:
    python3 scripts/extract_mt5_specs.py
    python3 scripts/extract_mt5_specs.py --symbols EURUSD BTCUSD XAUUSD
    python3 scripts/extract_mt5_specs.py --all  # Extract all visible symbols
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from kinetra.mt5_spec_extractor import extract_specs_from_mt5


# Default symbols to extract (common trading instruments)
DEFAULT_SYMBOLS = [
    # Forex majors
    "EURUSD", "GBPUSD", "USDJPY", "USDCHF",
    "AUDUSD", "NZDUSD", "USDCAD",

    # Forex minors
    "EURJPY", "GBPJPY", "EURGBP",

    # Crypto
    "BTCUSD", "ETHUSD",

    # Indices
    "US500", "US30", "NAS100", "DJ30",

    # Metals
    "XAUUSD", "XAGUSD",

    # Energy
    "XTIUSD", "XBRUSD",

    # Commodities
    "COPPER-C",
]


def main():
    parser = argparse.ArgumentParser(
        description="Extract MT5 symbol specifications to JSON",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        '--symbols',
        nargs='+',
        help='Specific symbols to extract (default: common instruments)'
    )

    parser.add_argument(
        '--all',
        action='store_true',
        help='Extract all visible symbols from MT5'
    )

    parser.add_argument(
        '--output',
        type=str,
        default='data/master/instrument_specs.json',
        help='Output JSON file path (default: data/master/instrument_specs.json)'
    )

    args = parser.parse_args()

    # Determine which symbols to extract
    if args.all:
        print("Extracting ALL visible symbols from MT5...")
        symbols = get_all_mt5_symbols()
    elif args.symbols:
        symbols = args.symbols
    else:
        symbols = DEFAULT_SYMBOLS

    print(f"Extracting {len(symbols)} symbols...")
    print(f"Output: {args.output}")
    print()

    # Extract specs
    try:
        specs = extract_specs_from_mt5(symbols, args.output)
        print()
        print(f"✓ Successfully extracted {len(specs)} symbol specifications")
        print(f"✓ Saved to {args.output}")

        # Print summary
        print()
        print("Summary:")
        for symbol, spec in specs.items():
            print(f"  {symbol:12s} - {spec.asset_class.value:10s} "
                  f"(spread: {spec.spread_typical:.1f}, "
                  f"swap: {spec.swap_long:+.2f}/{spec.swap_short:+.2f})")

    except Exception as e:
        print(f"✗ Error: {e}")
        sys.exit(1)


def get_all_mt5_symbols():
    """Get all visible symbols from MT5 Market Watch."""
    try:
        import MetaTrader5 as mt5
    except ImportError:
        print("Error: MetaTrader5 module not installed")
        print("Install with: pip install MetaTrader5")
        sys.exit(1)

    if not mt5.initialize():
        print(f"Error: MT5 initialization failed: {mt5.last_error()}")
        sys.exit(1)

    try:
        # Get all symbols
        symbols = mt5.symbols_get()
        if symbols is None:
            print("Error: Failed to get symbols from MT5")
            return DEFAULT_SYMBOLS

        # Filter to visible/tradeable symbols
        symbol_names = [s.name for s in symbols if s.visible and s.trade_mode == 4]
        return symbol_names

    finally:
        mt5.shutdown()


if __name__ == "__main__":
    main()
