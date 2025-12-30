#!/usr/bin/env python3
"""
Data Cutoff Standardization Utility

Ensures all training data has consistent temporal boundaries to prevent:
- Lookahead bias (training on future data)
- Temporal leakage across instruments
- Inconsistent episode lengths

Usage:
    python scripts/standardize_data_cutoff.py --analyze
    python scripts/standardize_data_cutoff.py --cutoff "2025-12-26 20:00"
    python scripts/standardize_data_cutoff.py --auto  # Uses earliest common cutoff
"""

import argparse
import os
import re
import shutil
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd


def parse_filename_timestamp(filename: str) -> Tuple[str, str, datetime, datetime]:
    """
    Parse instrument, timeframe, start, and end timestamps from filename.

    Format: INSTRUMENT_TIMEFRAME_YYYYMMDDHHMM_YYYYMMDDHHMM.csv (12 digits each)
    Example: BTCUSD_H1_202401020000_202512282200.csv
    """
    # Try 12-digit format first (YYYYMMDDHHMM), then 14-digit (YYYYMMDDHHMMSS)
    pattern_12 = r"(.+?)_([A-Z0-9]+)_(\d{12})_(\d{12})\.csv"
    pattern_14 = r"(.+?)_([A-Z0-9]+)_(\d{14})_(\d{14})\.csv"

    match = re.match(pattern_12, filename)
    fmt = "%Y%m%d%H%M"

    if not match:
        match = re.match(pattern_14, filename)
        fmt = "%Y%m%d%H%M%S"

    if not match:
        raise ValueError(f"Cannot parse filename: {filename}")

    instrument, timeframe, start_str, end_str = match.groups()

    start_dt = datetime.strptime(start_str, fmt)
    end_dt = datetime.strptime(end_str, fmt)

    return instrument, timeframe, start_dt, end_dt


def analyze_cutoffs(data_dir: str) -> Dict:
    """Analyze temporal cutoffs across all datasets."""
    data_path = Path(data_dir)
    csv_files = list(data_path.glob("*.csv"))

    if not csv_files:
        print(f"No CSV files found in {data_dir}")
        return {}

    cutoffs = defaultdict(list)
    by_instrument = defaultdict(dict)
    earliest_end = None
    latest_end = None

    for csv_file in csv_files:
        try:
            instrument, timeframe, start_dt, end_dt = parse_filename_timestamp(csv_file.name)

            cutoffs[end_dt].append({
                "file": csv_file.name,
                "instrument": instrument,
                "timeframe": timeframe,
                "start": start_dt,
                "end": end_dt,
            })

            by_instrument[instrument][timeframe] = end_dt

            if earliest_end is None or end_dt < earliest_end:
                earliest_end = end_dt
            if latest_end is None or end_dt > latest_end:
                latest_end = end_dt

        except ValueError as e:
            print(f"  [SKIP] {csv_file.name}: {e}")

    return {
        "cutoffs": dict(cutoffs),
        "by_instrument": dict(by_instrument),
        "earliest_end": earliest_end,
        "latest_end": latest_end,
        "total_files": len(csv_files),
    }


def print_analysis(analysis: Dict):
    """Print cutoff analysis in a readable format."""
    print("\n" + "=" * 80)
    print("  DATA CUTOFF ANALYSIS")
    print("=" * 80)

    print(f"\nTotal files: {analysis['total_files']}")
    print(f"Earliest cutoff: {analysis['earliest_end']}")
    print(f"Latest cutoff:   {analysis['latest_end']}")

    gap = analysis['latest_end'] - analysis['earliest_end']
    print(f"Gap:             {gap.days} days, {gap.seconds // 3600} hours")

    print("\n[CUTOFFS BY DATE]")
    print("-" * 60)

    for cutoff_dt in sorted(analysis['cutoffs'].keys()):
        files = analysis['cutoffs'][cutoff_dt]
        instruments = set(f['instrument'] for f in files)
        print(f"  {cutoff_dt.strftime('%Y-%m-%d %H:%M')}: {len(files):3d} files | "
              f"{len(instruments)} instruments")

    print("\n[BY INSTRUMENT - EARLIEST CUTOFF]")
    print("-" * 60)

    # Find instruments with earliest cutoffs
    instrument_min = {}
    for inst, timeframes in analysis['by_instrument'].items():
        min_cutoff = min(timeframes.values())
        instrument_min[inst] = min_cutoff

    for inst, cutoff in sorted(instrument_min.items(), key=lambda x: x[1]):
        gap_from_latest = analysis['latest_end'] - cutoff
        gap_str = f"-{gap_from_latest.days}d {gap_from_latest.seconds // 3600}h" if gap_from_latest.total_seconds() > 0 else "latest"
        print(f"  {inst:<20}: {cutoff.strftime('%Y-%m-%d %H:%M')} ({gap_str})")

    print("\n[RECOMMENDATION]")
    print("-" * 60)

    # Count how many files would be affected at different cutoffs
    cutoff_options = sorted(analysis['cutoffs'].keys())

    print("  Standardize to | Files at cutoff | Files to truncate | Data loss")
    print("  " + "-" * 60)

    for cutoff in cutoff_options[-5:]:  # Show last 5 options
        at_cutoff = len(analysis['cutoffs'][cutoff])
        to_truncate = sum(len(f) for dt, f in analysis['cutoffs'].items() if dt > cutoff)

        if cutoff == analysis['earliest_end']:
            print(f"  {cutoff.strftime('%Y-%m-%d %H:%M')} | {at_cutoff:3d} files       | {to_truncate:3d} files         | SAFEST (no bias)")
        else:
            print(f"  {cutoff.strftime('%Y-%m-%d %H:%M')} | {at_cutoff:3d} files       | {to_truncate:3d} files         |")

    print(f"\n  Recommended: Standardize to {analysis['earliest_end'].strftime('%Y-%m-%d %H:%M')}")
    print(f"  Command: python scripts/standardize_data_cutoff.py --cutoff \"{analysis['earliest_end'].strftime('%Y-%m-%d %H:%M')}\"")


def truncate_to_cutoff(
    data_dir: str,
    cutoff: datetime,
    output_dir: Optional[str] = None,
    dry_run: bool = True,
) -> Dict:
    """
    Truncate all datasets to a common cutoff time.

    Args:
        data_dir: Directory containing CSV files
        cutoff: Target cutoff datetime
        output_dir: Output directory (default: data_dir + "_standardized")
        dry_run: If True, only report what would be done

    Returns:
        Summary of operations
    """
    data_path = Path(data_dir)

    if output_dir is None:
        output_dir = str(data_path) + "_standardized"

    output_path = Path(output_dir)

    if not dry_run:
        output_path.mkdir(parents=True, exist_ok=True)

    results = {
        "truncated": [],
        "copied": [],
        "skipped": [],
        "errors": [],
    }

    csv_files = list(data_path.glob("*.csv"))

    print(f"\n{'[DRY RUN] ' if dry_run else ''}Truncating to cutoff: {cutoff}")
    print(f"Output directory: {output_path}")
    print("-" * 60)

    for csv_file in csv_files:
        try:
            instrument, timeframe, start_dt, end_dt = parse_filename_timestamp(csv_file.name)

            if end_dt <= cutoff:
                # File ends before cutoff - copy as-is
                if not dry_run:
                    shutil.copy(csv_file, output_path / csv_file.name)
                results["copied"].append(csv_file.name)
                print(f"  [COPY] {csv_file.name} (ends at {end_dt})")

            elif start_dt >= cutoff:
                # File starts after cutoff - skip entirely
                results["skipped"].append(csv_file.name)
                print(f"  [SKIP] {csv_file.name} (starts after cutoff)")

            else:
                # File needs truncation
                if not dry_run:
                    # Read tab-separated data (MT5 format)
                    df = pd.read_csv(csv_file, sep='\t')

                    # Normalize column names
                    df.columns = [c.lower().replace('<', '').replace('>', '') for c in df.columns]

                    # Combine date + time into datetime
                    if 'date' in df.columns and 'time' in df.columns:
                        df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'])
                        time_col = 'datetime'
                    elif 'datetime' in df.columns:
                        df['datetime'] = pd.to_datetime(df['datetime'])
                        time_col = 'datetime'
                    else:
                        # Assume first column is time
                        time_col = df.columns[0]
                        df[time_col] = pd.to_datetime(df[time_col])

                    # Filter by cutoff
                    df_truncated = df[df[time_col] <= cutoff]

                    if len(df_truncated) == 0:
                        results["skipped"].append(csv_file.name)
                        print(f"  [SKIP] {csv_file.name} (no data before cutoff)")
                        continue

                    # Generate new filename
                    new_end = df_truncated[time_col].max()
                    new_end_str = new_end.strftime("%Y%m%d%H%M")
                    start_str = start_dt.strftime("%Y%m%d%H%M")
                    new_filename = f"{instrument}_{timeframe}_{start_str}_{new_end_str}.csv"

                    # Save in same tab-separated format
                    # Restore original column format
                    df_truncated = df_truncated.drop(columns=['datetime'], errors='ignore')
                    df_truncated.columns = ['<' + c.upper() + '>' for c in df_truncated.columns]
                    df_truncated.to_csv(output_path / new_filename, sep='\t', index=False)

                rows_before = "?" if dry_run else len(pd.read_csv(csv_file))
                rows_after = "?" if dry_run else len(df_truncated)

                results["truncated"].append({
                    "file": csv_file.name,
                    "original_end": end_dt,
                    "new_end": cutoff,
                })
                print(f"  [TRUNC] {csv_file.name}: {end_dt} -> {cutoff}")

        except Exception as e:
            results["errors"].append({"file": csv_file.name, "error": str(e)})
            print(f"  [ERROR] {csv_file.name}: {e}")

    # Summary
    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    print(f"  Truncated: {len(results['truncated'])} files")
    print(f"  Copied:    {len(results['copied'])} files")
    print(f"  Skipped:   {len(results['skipped'])} files")
    print(f"  Errors:    {len(results['errors'])} files")

    if dry_run:
        print("\n  This was a DRY RUN. To apply changes, add --apply flag.")
    else:
        print(f"\n  Standardized data saved to: {output_path}")

    return results


def analyze_gaps(data_dir: str, timeframe: str = "H1") -> Dict:
    """
    Analyze gaps in data (weekends, holidays, unusual gaps).

    Returns dict with gap statistics and flagged anomalies.
    """
    from datetime import timedelta

    data_path = Path(data_dir)

    # Expected gap by timeframe (in hours)
    expected_gaps = {
        "M15": 0.25,
        "M30": 0.5,
        "H1": 1.0,
        "H4": 4.0,
    }
    expected_gap_hours = expected_gaps.get(timeframe, 1.0)

    # Weekend gap threshold (forex closes Friday 5pm NY, opens Sunday 5pm NY = ~48h)
    weekend_gap_hours = 48

    # Find matching files
    pattern = f"*_{timeframe}_*.csv"
    files = list(data_path.glob(pattern))

    if not files:
        print(f"No files matching {pattern} in {data_dir}")
        return {}

    all_gaps = []
    holiday_gaps = []

    for csv_file in files[:5]:  # Sample 5 files for speed
        try:
            df = pd.read_csv(csv_file, sep='\t')
            df.columns = [c.lower().replace('<', '').replace('>', '') for c in df.columns]

            if 'date' in df.columns and 'time' in df.columns:
                df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'])
            else:
                continue

            df = df.sort_values('datetime')

            # Calculate gaps
            df['gap_hours'] = df['datetime'].diff().dt.total_seconds() / 3600

            # Flag unusual gaps (not weekends)
            for idx, row in df.iterrows():
                gap = row['gap_hours']
                if pd.isna(gap):
                    continue

                dt = row['datetime']

                # Skip normal weekend gaps (Friday evening to Sunday evening)
                is_weekend_gap = (
                    dt.weekday() == 6 and  # Sunday
                    gap >= 40 and gap <= 60
                )

                if gap > expected_gap_hours * 2 and not is_weekend_gap:
                    all_gaps.append({
                        "file": csv_file.name,
                        "datetime": dt,
                        "gap_hours": gap,
                        "weekday": dt.strftime("%A"),
                    })

                    # Flag potential holidays (gap > 24h on weekday)
                    if gap > 24 and dt.weekday() < 5:
                        holiday_gaps.append({
                            "datetime": dt,
                            "gap_hours": gap,
                            "likely_holiday": True,
                        })

        except Exception as e:
            print(f"  [ERROR] {csv_file.name}: {e}")

    return {
        "total_gaps_found": len(all_gaps),
        "holiday_gaps": len(holiday_gaps),
        "gaps": all_gaps[:20],  # First 20
        "holidays": holiday_gaps,
    }


def print_gap_analysis(gaps: Dict):
    """Print gap analysis results."""
    print("\n" + "=" * 60)
    print("  GAP ANALYSIS (Holidays, Unusual Gaps)")
    print("=" * 60)

    print(f"\nTotal unusual gaps found: {gaps.get('total_gaps_found', 0)}")
    print(f"Potential holiday gaps: {gaps.get('holiday_gaps', 0)}")

    if gaps.get('holidays'):
        print("\n[LIKELY HOLIDAYS]")
        print("-" * 40)
        for h in gaps['holidays'][:10]:
            dt = h['datetime']
            print(f"  {dt.strftime('%Y-%m-%d %A')}: {h['gap_hours']:.1f}h gap")

    if gaps.get('gaps'):
        print("\n[UNUSUAL GAPS (sample)]")
        print("-" * 40)
        for g in gaps['gaps'][:10]:
            print(f"  {g['datetime']}: {g['gap_hours']:.1f}h ({g['weekday']})")


def main():
    parser = argparse.ArgumentParser(
        description="Standardize temporal cutoffs across training data"
    )
    parser.add_argument(
        "--data-dir",
        default="data/master",
        help="Directory containing CSV data files"
    )
    parser.add_argument(
        "--analyze",
        action="store_true",
        help="Analyze current cutoffs without making changes"
    )
    parser.add_argument(
        "--cutoff",
        type=str,
        help="Target cutoff datetime (e.g., '2025-12-26 20:00')"
    )
    parser.add_argument(
        "--auto",
        action="store_true",
        help="Automatically use earliest common cutoff"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Output directory for standardized data"
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually apply changes (default is dry run)"
    )
    parser.add_argument(
        "--gaps",
        action="store_true",
        help="Analyze gaps in data (holidays, weekends, unusual gaps)"
    )
    parser.add_argument(
        "--timeframe",
        default="H1",
        help="Timeframe for gap analysis (default: H1)"
    )

    args = parser.parse_args()

    # Gap analysis
    if args.gaps:
        gaps = analyze_gaps(args.data_dir, args.timeframe)
        print_gap_analysis(gaps)
        return

    # Always analyze first
    analysis = analyze_cutoffs(args.data_dir)

    if not analysis:
        return

    if args.analyze or (not args.cutoff and not args.auto):
        print_analysis(analysis)
        return

    # Determine cutoff
    if args.auto:
        cutoff = analysis['earliest_end']
        print(f"\n[AUTO] Using earliest common cutoff: {cutoff}")
    else:
        cutoff = datetime.strptime(args.cutoff, "%Y-%m-%d %H:%M")

    # Truncate
    truncate_to_cutoff(
        data_dir=args.data_dir,
        cutoff=cutoff,
        output_dir=args.output_dir,
        dry_run=not args.apply,
    )


if __name__ == "__main__":
    main()
