#!/usr/bin/env python3
"""
Data Integrity Checker
======================

Step 5: Check downloaded data for:
- Download failures
- Interruptions
- Stoppages
- Missing data
- Corrupt files
- Data quality issues

Usage:
    python scripts/check_data_integrity.py
"""

import sys
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from collections import defaultdict

from pandas import DataFrame

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


class IntegrityIssue:
    """Represents a data integrity issue."""

    def __init__(self, severity: str, category: str, file: str, description: str):
        self.severity = severity  # CRITICAL, WARNING, INFO
        self.category = category  # MISSING, CORRUPT, GAPS, INCOMPLETE, QUALITY
        self.file = file
        self.description = description


class DataIntegrityChecker:
    """Check data integrity and quality."""

    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.issues: List[IntegrityIssue] = []
        self.files_checked = 0
        self.files_passed = 0
        self.files_failed = 0

    def check_file_exists(self, filepath: Path) -> bool:
        """Check if file exists and is readable."""
        if not filepath.exists():
            self.issues.append(IntegrityIssue(
                'CRITICAL', 'MISSING', filepath.name,
                'File does not exist'
            ))
            return False

        if filepath.stat().st_size == 0:
            self.issues.append(IntegrityIssue(
                'CRITICAL', 'CORRUPT', filepath.name,
                'File is empty (0 bytes)'
            ))
            return False

        return True

    def check_csv_format(self, filepath: Path) -> tuple[bool, None] | tuple[bool, DataFrame]:
        """Check if CSV is valid and has required columns."""
        try:
            df = pd.read_csv(filepath)

            # Check for required columns
            required = ['time', 'open', 'high', 'low', 'close']
            missing = [col for col in required if col not in df.columns]

            if missing:
                self.issues.append(IntegrityIssue(
                    'CRITICAL', 'CORRUPT', filepath.name,
                    f'Missing required columns: {", ".join(missing)}'
                ))
                return False, None

            # Check if time column is valid
            try:
                df['time'] = pd.to_datetime(df['time'])
            except Exception as e:
                self.issues.append(IntegrityIssue(
                    'CRITICAL', 'CORRUPT', filepath.name,
                    f'Invalid time column: {e}'
                ))
                return False, None

            return True, df

        except Exception as e:
            self.issues.append(IntegrityIssue(
                'CRITICAL', 'CORRUPT', filepath.name,
                f'Failed to read CSV: {e}'
            ))
            return False, None

    def check_data_completeness(self, df: pd.DataFrame, filepath: Path, timeframe: str):
        """Check for data completeness."""
        # Minimum bars threshold
        min_bars = {
            'M15': 2000,   # ~3 weeks
            'M30': 1000,   # ~3 weeks
            'H1': 500,     # ~3 weeks
            'H4': 180,     # ~1 month
            'D1': 60       # ~2 months
        }

        threshold = min_bars.get(timeframe, 500)

        if len(df) < threshold:
            self.issues.append(IntegrityIssue(
                'WARNING', 'INCOMPLETE', filepath.name,
                f'Only {len(df)} bars, expected at least {threshold}'
            ))

    def check_data_gaps(self, df: pd.DataFrame, filepath: Path, timeframe: str):
        """Check for gaps in time series."""
        df = df.sort_values('time').copy()

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
        median_diff = time_diffs.median()

        # Find large gaps (>3x expected interval)
        # Account for weekends/holidays
        threshold = timedelta(minutes=expected_interval * 3)
        large_gaps = time_diffs[time_diffs > threshold]

        if len(large_gaps) > 0:
            # Filter out weekend gaps for forex (Fri-Mon)
            significant_gaps = []
            for idx in large_gaps.index:
                gap = time_diffs.loc[idx]
                prev_time = df.loc[idx - 1, 'time']

                # Check if gap is weekend (Friday to Monday)
                is_weekend = (prev_time.weekday() == 4 and  # Friday
                             df.loc[idx, 'time'].weekday() == 0 and  # Monday
                             gap <= timedelta(days=3))

                if not is_weekend:
                    significant_gaps.append((prev_time, df.loc[idx, 'time'], gap))

            if significant_gaps:
                gap_summary = f'{len(significant_gaps)} significant gaps found'
                if len(significant_gaps) <= 3:
                    gap_details = '; '.join([
                        f'{start.strftime("%Y-%m-%d %H:%M")} → {end.strftime("%Y-%m-%d %H:%M")} ({gap})'
                        for start, end, gap in significant_gaps
                    ])
                    gap_summary += f': {gap_details}'

                self.issues.append(IntegrityIssue(
                    'WARNING', 'GAPS', filepath.name,
                    gap_summary
                ))

    def check_data_quality(self, df: pd.DataFrame, filepath: Path):
        """Check data quality (price validity, duplicates, etc)."""
        # Check for negative or zero prices
        price_cols = ['open', 'high', 'low', 'close']
        for col in price_cols:
            if (df[col] <= 0).any():
                count = (df[col] <= 0).sum()
                self.issues.append(IntegrityIssue(
                    'CRITICAL', 'QUALITY', filepath.name,
                    f'{count} rows with invalid {col} price (<= 0)'
                ))

        # Check for high/low consistency
        invalid_hl = (df['low'] > df['high']).sum()
        if invalid_hl > 0:
            self.issues.append(IntegrityIssue(
                'CRITICAL', 'QUALITY', filepath.name,
                f'{invalid_hl} rows where low > high'
            ))

        # Check for open/close within high/low
        invalid_o = ((df['open'] > df['high']) | (df['open'] < df['low'])).sum()
        invalid_c = ((df['close'] > df['high']) | (df['close'] < df['low'])).sum()

        if invalid_o > 0:
            self.issues.append(IntegrityIssue(
                'WARNING', 'QUALITY', filepath.name,
                f'{invalid_o} rows where open outside high/low range'
            ))

        if invalid_c > 0:
            self.issues.append(IntegrityIssue(
                'WARNING', 'QUALITY', filepath.name,
                f'{invalid_c} rows where close outside high/low range'
            ))

        # Check for duplicate timestamps
        duplicates = df['time'].duplicated().sum()
        if duplicates > 0:
            self.issues.append(IntegrityIssue(
                'WARNING', 'QUALITY', filepath.name,
                f'{duplicates} duplicate timestamps'
            ))

        # Check for NaN values
        nan_counts = df[price_cols].isna().sum()
        if nan_counts.any():
            nan_summary = ', '.join([f'{col}:{count}' for col, count in nan_counts.items() if count > 0])
            self.issues.append(IntegrityIssue(
                'WARNING', 'QUALITY', filepath.name,
                f'NaN values: {nan_summary}'
            ))

    def check_file(self, filepath: Path) -> bool:
        """Run all checks on a single file."""
        self.files_checked += 1

        # Parse filename to get timeframe
        parts = filepath.stem.split('_')
        timeframe = parts[1] if len(parts) > 1 else 'UNKNOWN'

        # Check 1: File exists
        if not self.check_file_exists(filepath):
            self.files_failed += 1
            return False

        # Check 2: Valid CSV format
        valid, df = self.check_csv_format(filepath)
        if not valid:
            self.files_failed += 1
            return False

        # Check 3: Data completeness
        self.check_data_completeness(df, filepath, timeframe)

        # Check 4: Data gaps
        self.check_data_gaps(df, filepath, timeframe)

        # Check 5: Data quality
        self.check_data_quality(df, filepath)

        # If no CRITICAL issues for this file, it passed
        file_issues = [i for i in self.issues if i.file == filepath.name and i.severity == 'CRITICAL']
        if file_issues:
            self.files_failed += 1
            return False
        else:
            self.files_passed += 1
            return True

    def check_all(self):
        """Check all CSV files in data directory."""
        csv_files = sorted(self.data_dir.glob('*.csv'))

        if not csv_files:
            print(f"\n❌ No CSV files found in {self.data_dir}")
            return

        print(f"\n🔍 Checking {len(csv_files)} files...")

        for i, filepath in enumerate(csv_files, 1):
            if i % 10 == 0 or i == len(csv_files):
                print(f"  Progress: {i}/{len(csv_files)} files checked", end='\r')
            self.check_file(filepath)

        print(f"\n✅ Checked {self.files_checked} files")

    def print_report(self):
        """Print integrity check report."""
        print_header("DATA INTEGRITY REPORT")

        # Summary
        print(f"\n📊 Summary:")
        print(f"  Files checked:  {self.files_checked}")
        print(f"  ✅ Passed:      {self.files_passed}")
        print(f"  ❌ Failed:      {self.files_failed}")

        if not self.issues:
            print(f"\n🎉 All files passed integrity checks!")
            return True

        # Group issues by severity and category
        by_severity = defaultdict(list)
        for issue in self.issues:
            by_severity[issue.severity].append(issue)

        # Print CRITICAL issues
        if 'CRITICAL' in by_severity:
            print(f"\n❌ CRITICAL Issues ({len(by_severity['CRITICAL'])}):")
            for issue in by_severity['CRITICAL'][:20]:  # Show first 20
                print(f"  [{issue.category}] {issue.file}")
                print(f"    {issue.description}")
            if len(by_severity['CRITICAL']) > 20:
                print(f"  ... and {len(by_severity['CRITICAL']) - 20} more critical issues")

        # Print WARNING issues
        if 'WARNING' in by_severity:
            print(f"\n⚠️  Warnings ({len(by_severity['WARNING'])}):")

            # Group warnings by category
            by_category = defaultdict(list)
            for issue in by_severity['WARNING']:
                by_category[issue.category].append(issue)

            for category, issues in by_category.items():
                print(f"\n  {category} ({len(issues)} files):")
                for issue in issues[:5]:  # Show first 5
                    print(f"    {issue.file}: {issue.description}")
                if len(issues) > 5:
                    print(f"    ... and {len(issues) - 5} more {category} warnings")

        # Recommendations
        print(f"\n💡 Recommendations:")

        if self.files_failed > 0:
            print(f"  1. Re-download failed files ({self.files_failed} files)")
            print(f"     python scripts/download_interactive.py")

        if 'GAPS' in [i.category for i in self.issues]:
            print(f"  2. Run data preparation to handle gaps and missing data")
            print(f"     python scripts/prepare_data.py")

        if 'QUALITY' in [i.category for i in self.issues]:
            print(f"  3. Review data quality issues before training")

        return self.files_failed == 0


def main():
    """Run data integrity check."""
    print_header("DATA INTEGRITY CHECK - STEP 5")

    data_dir = Path("data/master")

    if not data_dir.exists():
        print(f"\n❌ Data directory not found: {data_dir}")
        print(f"   Run download script first")
        return

    checker = DataIntegrityChecker(data_dir)
    checker.check_all()
    passed = checker.print_report()

    if passed:
        print(f"\n✅ Data integrity verified!")
        print(f"\n📁 Next step: Prepare data")
        print(f"   python scripts/prepare_data.py")
    else:
        print(f"\n⚠️  Data integrity issues detected")
        print(f"   Fix critical issues before proceeding")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Check interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
