#!/usr/bin/env python3
"""
Convert MT5 Format Files
========================

Converts MT5 export format to standard format:
- Combines <DATE> and <TIME> into 'time'
- Renames columns: <OPEN> -> open, <HIGH> -> high, etc.
- Keeps essential columns only

Usage:
    python scripts/convert_mt5_format.py
"""

import sys
from pathlib import Path
import pandas as pd

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))


def convert_file(filepath: Path) -> bool:
    """Convert a single MT5 format file."""
    try:
        # Read the file
        df = pd.read_csv(filepath, sep='\t')

        # Check if it's MT5 format
        if '<DATE>' not in df.columns or '<TIME>' not in df.columns:
            return False  # Already in correct format or different issue

        # Combine date and time
        df['time'] = pd.to_datetime(df['<DATE>'] + ' ' + df['<TIME>'])

        # Rename columns
        column_mapping = {
            '<OPEN>': 'open',
            '<HIGH>': 'high',
            '<LOW>': 'low',
            '<CLOSE>': 'close',
            '<TICKVOL>': 'volume',
            '<SPREAD>': 'spread'
        }

        df = df.rename(columns=column_mapping)

        # Keep only essential columns
        essential_cols = ['time', 'open', 'high', 'low', 'close']
        if 'volume' in df.columns:
            essential_cols.append('volume')
        if 'spread' in df.columns:
            essential_cols.append('spread')

        df = df[[c for c in essential_cols if c in df.columns]]

        # Save back to file (overwrite)
        df.to_csv(filepath, index=False)

        return True

    except Exception as e:
        print(f"  ❌ {filepath.name}: {e}")
        return False


def main():
    """Convert all MT5 format files in data/master."""
    print("=" * 80)
    print("  CONVERT MT5 FORMAT FILES")
    print("=" * 80)

    data_dir = Path("data/master")

    if not data_dir.exists():
        print(f"\n❌ Data directory not found: {data_dir}")
        return

    csv_files = list(data_dir.glob("*.csv"))

    if not csv_files:
        print(f"\n❌ No CSV files found in {data_dir}")
        return

    print(f"\n📁 Found {len(csv_files)} CSV files")
    print("\n🔄 Converting MT5 format to standard format...")
    print("   (Combining <DATE>+<TIME> → time, renaming columns)")

    converted = 0
    skipped = 0
    failed = 0

    for i, filepath in enumerate(csv_files, 1):
        if i % 10 == 0:
            print(f"  Progress: {i}/{len(csv_files)}", end='\r')

        result = convert_file(filepath)

        if result:
            converted += 1
        elif result is False:
            skipped += 1
        else:
            failed += 1

    print(f"  Completed: {len(csv_files)}/{len(csv_files)}")

    print("\n" + "=" * 80)
    print("  CONVERSION COMPLETE")
    print("=" * 80)

    print(f"\n📊 Results:")
    print(f"  ✅ Converted: {converted}")
    print(f"  ⏭️  Skipped:   {skipped}")
    print(f"  ❌ Failed:    {failed}")

    if converted > 0:
        print(f"\n✅ Converted {converted} files to standard format!")
        print("\n💡 Next step: Run integrity check again")
        print("   python scripts/check_data_integrity.py")
    else:
        print("\n⚠️  No files needed conversion")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
