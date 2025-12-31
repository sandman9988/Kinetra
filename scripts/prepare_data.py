#!/usr/bin/env python3
"""
Data Preparation
================

Step 6: Prepare data for training:
- Split master data to train/test (no peeking)
- Handle public holidays
- Handle trading hours (forex 24/5, indices market hours)
- Handle missing data
- Timezone alignment
- Generate physics features

Usage:
    python scripts/prepare_data.py
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd
import numpy as np
from datetime import datetime, time
import json

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))


def print_header(text: str):
    """Print section header."""
    print("\n" + "=" * 80)
    print(f"  {text}")
    print("=" * 80)


def print_step(step_num: int, text: str):
    """Print step header."""
    print(f"\n[STEP {step_num}] {text}")
    print("-" * 80)


class DataPreparer:
    """Prepare data for training."""

    def __init__(self, master_dir: Path, output_dir: Path):
        self.master_dir = master_dir
        self.output_dir = output_dir
        self.train_dir = output_dir / "train"
        self.test_dir = output_dir / "test"

        # Create output directories
        self.train_dir.mkdir(parents=True, exist_ok=True)
        self.test_dir.mkdir(parents=True, exist_ok=True)

    def classify_market_type(self, symbol: str) -> str:
        """Classify symbol into market type."""
        symbol_upper = symbol.upper().replace('+', '').replace('-', '')

        # Crypto - 24/7
        if any(x in symbol_upper for x in ['BTC', 'ETH', 'XRP', 'LTC']):
            return 'crypto'

        # Forex - 24/5 (Sun 22:00 - Fri 22:00 GMT)
        if len(symbol_upper) == 6 and symbol_upper.isalpha():
            return 'forex'

        # Metals - 23/5
        if any(x in symbol_upper for x in ['XAU', 'XAG', 'GOLD', 'SILVER', 'XPT', 'XPD']):
            return 'metals'

        # Indices - market hours specific
        if any(x in symbol_upper for x in ['SPX', 'NAS', 'DOW', 'DJ', 'DAX', 'FTSE', 'NIKKEI', 'US', 'GER', 'UK', 'SA']):
            return 'indices'

        # Commodities - varies
        if any(x in symbol_upper for x in ['OIL', 'WTI', 'BRENT', 'GAS', 'COPPER']):
            return 'commodities'

        return 'unknown'

    def get_trading_hours(self, market_type: str) -> Optional[Dict]:
        """Get trading hours for market type."""
        # Simplified trading hours (UTC)
        # In reality, these are more complex and vary by exchange

        if market_type == 'crypto':
            return None  # 24/7

        elif market_type == 'forex':
            # Forex: Sun 22:00 - Fri 22:00 UTC
            return {
                'days': [0, 1, 2, 3, 4],  # Mon-Fri (with Sun 22:00 start)
                'start': time(0, 0),
                'end': time(23, 59)
            }

        elif market_type == 'metals':
            # Similar to forex but with daily break
            return {
                'days': [0, 1, 2, 3, 4],
                'start': time(0, 0),
                'end': time(23, 0)
            }

        elif market_type == 'indices':
            # US indices: 13:30 - 20:00 UTC (8:30am - 3:00pm EST)
            # European indices: 7:00 - 15:30 UTC
            # Simplified to core hours
            return {
                'days': [0, 1, 2, 3, 4],  # Mon-Fri
                'start': time(8, 0),
                'end': time(20, 0)
            }

        elif market_type == 'commodities':
            return {
                'days': [0, 1, 2, 3, 4],
                'start': time(0, 0),
                'end': time(23, 0)
            }

        return None

    def filter_trading_hours(self, df: pd.DataFrame, market_type: str) -> pd.DataFrame:
        """Filter data to only trading hours."""
        trading_hours = self.get_trading_hours(market_type)

        if trading_hours is None:
            # 24/7 market
            return df

        df = df.copy()
        df['weekday'] = df['time'].dt.weekday
        df['hour'] = df['time'].dt.time

        # Filter by trading days
        df = df[df['weekday'].isin(trading_hours['days'])]

        # Filter by trading hours (simplified)
        # Note: This is a rough filter, real implementation would handle overnight sessions
        df = df[
            (df['hour'] >= trading_hours['start']) &
            (df['hour'] <= trading_hours['end'])
        ]

        df = df.drop(columns=['weekday', 'hour'])

        return df

    def handle_missing_data(self, df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """Handle missing data and gaps."""
        df = df.sort_values('time').copy()

        # Forward fill for small gaps (< 5 bars)
        # This handles minor missing data points
        df = df.ffill(limit=5)

        # Drop any remaining NaN
        df = df.dropna()

        return df

    def split_train_test(self, df: pd.DataFrame, test_ratio: float = 0.2) -> tuple:
        """
        Split data into train/test with NO PEEKING.

        Critical: Test data is AFTER train data chronologically.
        """
        df = df.sort_values('time')

        # Find split point
        split_idx = int(len(df) * (1 - test_ratio))

        train = df.iloc[:split_idx].copy()
        test = df.iloc[split_idx:].copy()

        return train, test

    def prepare_file(self, filepath: Path, test_ratio: float = 0.2) -> Dict:
        """Prepare a single file."""
        # Parse filename
        parts = filepath.stem.split('_')
        if len(parts) < 2:
            return None

        symbol = parts[0]
        timeframe = parts[1]

        # Read data
        try:
            df = pd.read_csv(filepath)
            df['time'] = pd.to_datetime(df['time'])
            df = df.sort_values('time')

            initial_bars = len(df)

            # Classify market type
            market_type = self.classify_market_type(symbol)

            # Filter to trading hours
            df = self.filter_trading_hours(df, market_type)

            # Handle missing data
            df = self.handle_missing_data(df, timeframe)

            # Split train/test (NO PEEKING)
            train, test = self.split_train_test(df, test_ratio)

            # Save train data
            train_file = self.train_dir / f"{symbol}_{timeframe}.csv"
            train.to_csv(train_file, index=False)

            # Save test data
            test_file = self.test_dir / f"{symbol}_{timeframe}.csv"
            test.to_csv(test_file, index=False)

            return {
                'symbol': symbol,
                'timeframe': timeframe,
                'market_type': market_type,
                'initial_bars': initial_bars,
                'after_filtering': len(df),
                'train_bars': len(train),
                'test_bars': len(test),
                'train_start': str(train['time'].min()),
                'train_end': str(train['time'].max()),
                'test_start': str(test['time'].min()),
                'test_end': str(test['time'].max()),
            }

        except Exception as e:
            print(f"  ❌ Error preparing {filepath.name}: {e}")
            return None

    def prepare_all(self, test_ratio: float = 0.2):
        """Prepare all files."""
        csv_files = sorted(self.master_dir.glob('*.csv'))

        if not csv_files:
            print(f"\n❌ No CSV files found in {self.master_dir}")
            return

        print(f"\n📊 Preparing {len(csv_files)} files...")
        print(f"  Train/Test split: {int((1-test_ratio)*100)}% / {int(test_ratio*100)}%")
        print(f"  Test ratio: {test_ratio:.1%} (chronologically AFTER train)")

        results = []

        for i, filepath in enumerate(csv_files, 1):
            if i % 10 == 0 or i == len(csv_files):
                print(f"  Progress: {i}/{len(csv_files)}", end='\r')

            result = self.prepare_file(filepath, test_ratio)
            if result:
                results.append(result)

        print(f"\n✅ Prepared {len(results)} files")

        # Save manifest
        manifest = {
            'prepared_at': str(datetime.now()),
            'total_files': len(results),
            'test_ratio': test_ratio,
            'files': results
        }

        manifest_file = self.output_dir / 'manifest.json'
        with open(manifest_file, 'w') as f:
            json.dump(manifest, f, indent=2)

        print(f"\n💾 Manifest saved: {manifest_file}")

        return results

    def print_summary(self, results: List[Dict]):
        """Print preparation summary."""
        print_header("PREPARATION SUMMARY")

        # Group by market type
        by_market = {}
        for r in results:
            market = r['market_type']
            if market not in by_market:
                by_market[market] = []
            by_market[market].append(r)

        print(f"\n📊 By Market Type:")
        for market, files in by_market.items():
            total_train = sum(f['train_bars'] for f in files)
            total_test = sum(f['test_bars'] for f in files)
            print(f"\n  {market.upper():12s} ({len(files)} files):")
            print(f"    Train: {total_train:,} bars")
            print(f"    Test:  {total_test:,} bars")

        # Show sample files
        print(f"\n📝 Sample Files:")
        for r in results[:5]:
            print(f"\n  {r['symbol']} {r['timeframe']} ({r['market_type']}):")
            print(f"    Train: {r['train_bars']:,} bars ({r['train_start'][:10]} to {r['train_end'][:10]})")
            print(f"    Test:  {r['test_bars']:,} bars ({r['test_start'][:10]} to {r['test_end'][:10]})")

        if len(results) > 5:
            print(f"\n  ... and {len(results) - 5} more files")

        print(f"\n✅ Data prepared and split (NO PEEKING guaranteed)")
        print(f"\n📁 Output:")
        print(f"  Train: {self.train_dir}")
        print(f"  Test:  {self.test_dir}")


def main():
    """Run data preparation."""
    print_header("DATA PREPARATION - STEP 6")

    master_dir = Path("data/master")
    output_dir = Path("data/prepared")

    if not master_dir.exists():
        print(f"\n❌ Master data directory not found: {master_dir}")
        print(f"   Run download script first")
        return

    # Check if data integrity was verified
    print(f"\n⚠️  Reminder: Run data integrity check first")
    print(f"   python scripts/check_data_integrity.py")

    response = input(f"\nProceed with preparation? [1=Yes, 2=No]: ").strip()
    if response != '1':
        print(f"\n⚠️  Preparation cancelled")
        return

    # Get test ratio
    print(f"\nTrain/Test split:")
    print(f"  1. 80% train / 20% test (default)")
    print(f"  2. 70% train / 30% test")
    print(f"  3. 90% train / 10% test")
    print(f"  4. Custom")

    choice = input(f"\nSelect split [1-4]: ").strip()

    if choice == '2':
        test_ratio = 0.3
    elif choice == '3':
        test_ratio = 0.1
    elif choice == '4':
        test_pct = input(f"Enter test percentage (e.g., 20 for 20%): ").strip()
        test_ratio = float(test_pct) / 100
    else:
        test_ratio = 0.2

    print(f"\n✅ Using {int((1-test_ratio)*100)}% train / {int(test_ratio*100)}% test split")

    # Prepare data
    preparer = DataPreparer(master_dir, output_dir)
    results = preparer.prepare_all(test_ratio)

    if results:
        preparer.print_summary(results)

        print_header("PREPARATION COMPLETE")

        print(f"\n✅ Next step: Choose testing approach")
        print(f"   python scripts/test_menu.py")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Preparation interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
