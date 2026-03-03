#!/usr/bin/env python3
"""
DATA DISCOVERY - Find what ACTUALLY exists
===========================================

NO ASSUMPTIONS. Just scan the filesystem and report reality.
"""
import re
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "master_standardized"

def discover_all_data():
    """Scan filesystem and discover ALL available data."""

    if not DATA_DIR.exists():
        return []

    results = []

    # Find all CSV files
    for csv_file in sorted(DATA_DIR.glob("*.csv")):
        # Parse filename: SYMBOL_TIMEFRAME.csv
        match = re.match(r'([A-Z0-9]+)_([A-Z0-9]+)(?:_.*)?\.csv$', csv_file.name)
        if not match:
            continue

        symbol = match.group(1)
        timeframe = match.group(2)

        try:
            df = pd.read_csv(csv_file)
            num_bars = len(df)

            # Get date range if possible
            date_range = None
            if 'time' in df.columns:
                date_range = (str(df['time'].iloc[0]), str(df['time'].iloc[-1]))
            elif 'timestamp' in df.columns:
                date_range = (str(df['timestamp'].iloc[0]), str(df['timestamp'].iloc[-1]))

            results.append({
                'symbol': symbol,
                'timeframe': timeframe,
                'bars': num_bars,
                'file': csv_file.name,
                'size_mb': csv_file.stat().st_size / (1024 * 1024),
                'date_range': date_range
            })
        except Exception as e:
            print(f"⚠️  Error reading {csv_file.name}: {e}")

    return results

def print_discovery():
    """Print discovered data in useful format."""
    data = discover_all_data()

    if not data:
        print("❌ No data found in", DATA_DIR)
        return

    print(f"\n{'='*80}")
    print(f"DISCOVERED DATA ({len(data)} files)")
    print(f"{'='*80}\n")

    # Group by symbol
    by_symbol = {}
    for item in data:
        symbol = item['symbol']
        if symbol not in by_symbol:
            by_symbol[symbol] = []
        by_symbol[symbol].append(item)

    print(f"Found {len(by_symbol)} unique symbols:")
    for symbol in sorted(by_symbol.keys()):
        items = by_symbol[symbol]
        tfs = [item['timeframe'] for item in items]
        total_bars = sum(item['bars'] for item in items)
        print(f"  {symbol}: {len(items)} timeframes ({', '.join(sorted(tfs))}) - {total_bars:,} total bars")

    # Group by timeframe
    print("\nTimeframes found:")
    by_tf = {}
    for item in data:
        tf = item['timeframe']
        if tf not in by_tf:
            by_tf[tf] = []
        by_tf[tf].append(item)

    for tf in sorted(by_tf.keys()):
        items = by_tf[tf]
        symbols = [item['symbol'] for item in items]
        print(f"  {tf}: {len(items)} symbols ({', '.join(sorted(symbols))})")

    # Usable data (>= 1000 bars)
    usable = [item for item in data if item['bars'] >= 1000]
    print(f"\n✅ USABLE ({len(usable)}/{len(data)} combinations with >=1000 bars):")
    for item in sorted(usable, key=lambda x: (x['symbol'], x['timeframe'])):
        print(f"  {item['symbol']:10} {item['timeframe']:5} - {item['bars']:6,} bars")

    # Insufficient data
    insufficient = [item for item in data if item['bars'] < 1000]
    if insufficient:
        print(f"\n⚠️  INSUFFICIENT ({len(insufficient)} combinations with <1000 bars):")
        for item in sorted(insufficient, key=lambda x: (x['symbol'], x['timeframe'])):
            print(f"  {item['symbol']:10} {item['timeframe']:5} - {item['bars']:6,} bars")

def get_usable_combinations(min_bars=1000):
    """
    Get list of usable symbol+timeframe combinations.
    
    Returns:
        List of (symbol, timeframe, bars) tuples
    """
    data = discover_all_data()
    return [(item['symbol'], item['timeframe'], item['bars'])
            for item in data if item['bars'] >= min_bars]

if __name__ == "__main__":
    print_discovery()

    # Export as JSON for menu consumption
    import json
    data = discover_all_data()
    output_file = PROJECT_ROOT / "data" / "available_data.json"
    output_file.parent.mkdir(exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"\n📄 Exported to: {output_file}")
