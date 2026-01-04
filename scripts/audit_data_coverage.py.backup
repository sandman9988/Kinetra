#!/usr/bin/env python3
"""
Data Coverage Audit Script for Kinetra
=======================================

Analyzes available data coverage across instruments and timeframes.
Identifies gaps and generates actionable reports.

Usage:
    # Basic audit
    python scripts/audit_data_coverage.py

    # With detailed report
    python scripts/audit_data_coverage.py --report data/coverage_report.csv

    # Check specific instruments
    python scripts/audit_data_coverage.py --instruments BTCUSD EURUSD

    # Minimum bars threshold
    python scripts/audit_data_coverage.py --min-bars 1000

Philosophy:
- Identify data gaps before they cause test failures
- Prioritize high-value instrument/timeframe combinations
- Guide data acquisition efforts
"""

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Project root
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import Kinetra components (graceful fallback)
try:
    from kinetra.mt5_connector import MT5Connector

    MT5_AVAILABLE = True
except ImportError:
    MT5_AVAILABLE = False
    print("Warning: MT5Connector not available")


# =============================================================================
# CONFIGURATION
# =============================================================================

# Instruments to check (5 asset classes)
INSTRUMENTS = {
    "crypto": ["BTCUSD", "ETHUSD"],
    "forex": ["EURUSD", "GBPUSD", "USDJPY"],
    "indices": ["US30", "NAS100", "SPX500"],
    "metals": ["XAUUSD", "XAGUSD"],
    "commodities": ["USOIL", "UKOIL"],
}

# Timeframes to check
TIMEFRAMES = ["M15", "M30", "H1", "H4", "D1"]

# Minimum bars for "good" coverage
MIN_BARS_REQUIRED = 1000  # ~40 days H1, ~1000 days D1

# Data sources (in priority order)
DATA_SOURCES = [
    "data/master_standardized",  # CSV cache
    "metaapi",  # MetaAPI connector
    "mt5",  # Direct MT5 (if available)
]


# =============================================================================
# UTILITIES
# =============================================================================


class Colors:
    """ANSI colors for terminal output."""

    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BLUE = "\033[94m"
    BOLD = "\033[1m"
    ENDC = "\033[0m"


def print_header(msg: str) -> None:
    """Print formatted header."""
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'=' * 70}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.BLUE}{msg.center(70)}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'=' * 70}{Colors.ENDC}\n")


def print_success(msg: str) -> None:
    """Print success message."""
    print(f"{Colors.GREEN}✅ {msg}{Colors.ENDC}")


def print_warning(msg: str) -> None:
    """Print warning message."""
    print(f"{Colors.YELLOW}⚠️  {msg}{Colors.ENDC}")


def print_error(msg: str) -> None:
    """Print error message."""
    print(f"{Colors.RED}❌ {msg}{Colors.ENDC}")


# =============================================================================
# DATA CHECKERS
# =============================================================================


def check_csv_data(instrument: str, timeframe: str, data_dir: Path) -> Tuple[bool, int, Dict]:
    """
    Check if CSV data exists and is sufficient.

    Returns:
        (exists, num_bars, metadata)
    """
    csv_path = data_dir / f"{instrument}_{timeframe}.csv"

    if not csv_path.exists():
        return False, 0, {}

    try:
        df = pd.read_csv(csv_path)
        num_bars = len(df)

        # Extract metadata
        metadata = {
            "path": str(csv_path),
            "size_mb": csv_path.stat().st_size / (1024 * 1024),
            "columns": list(df.columns),
            "date_range": None,
        }

        # Try to extract date range
        if "time" in df.columns:
            metadata["date_range"] = (
                df["time"].iloc[0],
                df["time"].iloc[-1],
            )
        elif "timestamp" in df.columns:
            metadata["date_range"] = (
                df["timestamp"].iloc[0],
                df["timestamp"].iloc[-1],
            )

        return True, num_bars, metadata

    except Exception as e:
        print_warning(f"Error reading {csv_path}: {e}")
        return False, 0, {}


def check_metaapi_data(instrument: str, timeframe: str) -> Tuple[bool, int, Dict]:
    """
    Check if MetaAPI can provide data (requires connection).

    Returns:
        (available, estimated_bars, metadata)
    """
    # Placeholder - would require actual MetaAPI connection
    # For now, return False (can be implemented later)
    return False, 0, {"source": "metaapi", "status": "not_checked"}


def check_mt5_data(instrument: str, timeframe: str) -> Tuple[bool, int, Dict]:
    """
    Check if MT5 can provide data (requires MT5 connection).

    Returns:
        (available, estimated_bars, metadata)
    """
    if not MT5_AVAILABLE:
        return False, 0, {"source": "mt5", "status": "not_available"}

    # Placeholder - would require actual MT5 connection
    # For now, return False
    return False, 0, {"source": "mt5", "status": "not_checked"}


# =============================================================================
# COVERAGE ANALYZER
# =============================================================================


class CoverageAnalyzer:
    """Analyzes data coverage across instruments and timeframes."""

    def __init__(
        self,
        instruments: Optional[List[str]] = None,
        timeframes: Optional[List[str]] = None,
        min_bars: int = MIN_BARS_REQUIRED,
        data_dir: Optional[Path] = None,
    ):
        """
        Initialize analyzer.

        Args:
            instruments: List of instruments to check (default: all)
            timeframes: List of timeframes to check (default: all)
            min_bars: Minimum bars for "good" coverage
            data_dir: Data directory (default: data/master_standardized)
        """
        # Flatten instrument dict if using defaults
        if instruments is None:
            instruments = []
            for asset_class, inst_list in INSTRUMENTS.items():
                instruments.extend(inst_list)

        self.instruments = instruments
        self.timeframes = timeframes or TIMEFRAMES
        self.min_bars = min_bars
        self.data_dir = data_dir or (PROJECT_ROOT / "data" / "master_standardized")

        self.results = {}

    def analyze(self) -> Dict:
        """
        Run coverage analysis.

        Returns:
            Coverage results dictionary
        """
        print_header("DATA COVERAGE ANALYSIS")

        total_combos = len(self.instruments) * len(self.timeframes)
        print(
            f"Checking {len(self.instruments)} instruments × {len(self.timeframes)} timeframes = {total_combos} combinations"
        )
        print(f"Minimum bars required: {self.min_bars}")
        print(f"Data directory: {self.data_dir}\n")

        # Check each combination
        for instrument in self.instruments:
            self.results[instrument] = {}

            for timeframe in self.timeframes:
                # Check CSV first (primary source)
                exists, num_bars, metadata = check_csv_data(instrument, timeframe, self.data_dir)

                # Determine status
                if exists and num_bars >= self.min_bars:
                    status = "good"
                    icon = "✅"
                elif exists and num_bars > 0:
                    status = "partial"
                    icon = "⚠️"
                else:
                    status = "missing"
                    icon = "❌"

                self.results[instrument][timeframe] = {
                    "status": status,
                    "num_bars": num_bars,
                    "exists": exists,
                    "metadata": metadata,
                }

                # Print status
                print(f"{icon} {instrument:8} {timeframe:4} - {num_bars:5} bars ({status})")

        return self.results

    def generate_summary(self) -> Dict:
        """Generate summary statistics."""
        total = 0
        good = 0
        partial = 0
        missing = 0

        for instrument in self.results:
            for timeframe in self.results[instrument]:
                result = self.results[instrument][timeframe]
                total += 1

                if result["status"] == "good":
                    good += 1
                elif result["status"] == "partial":
                    partial += 1
                else:
                    missing += 1

        coverage_pct = (good / total * 100) if total > 0 else 0

        return {
            "total_combinations": total,
            "good_coverage": good,
            "partial_coverage": partial,
            "missing": missing,
            "coverage_percentage": coverage_pct,
        }

    def print_summary(self) -> None:
        """Print coverage summary."""
        summary = self.generate_summary()

        print_header("COVERAGE SUMMARY")
        print(f"Total combinations:  {summary['total_combinations']}")
        print(
            f"Good coverage:       {summary['good_coverage']} ({summary['good_coverage'] / summary['total_combinations'] * 100:.1f}%)"
        )
        print(
            f"Partial coverage:    {summary['partial_coverage']} ({summary['partial_coverage'] / summary['total_combinations'] * 100:.1f}%)"
        )
        print(
            f"Missing:             {summary['missing']} ({summary['missing'] / summary['total_combinations'] * 100:.1f}%)"
        )
        print(f"\nOverall coverage:    {summary['coverage_percentage']:.1f}%")

        # Status indicator
        if summary["coverage_percentage"] >= 80:
            print_success("Coverage is GOOD (≥80%)")
        elif summary["coverage_percentage"] >= 50:
            print_warning("Coverage is PARTIAL (50-80%)")
        else:
            print_error("Coverage is POOR (<50%)")

    def identify_gaps(self) -> List[Tuple[str, str]]:
        """Identify missing data combinations."""
        gaps = []

        for instrument in self.results:
            for timeframe in self.results[instrument]:
                result = self.results[instrument][timeframe]
                if result["status"] == "missing":
                    gaps.append((instrument, timeframe))

        return gaps

    def identify_priority_gaps(self) -> List[Tuple[str, str, str]]:
        """
        Identify high-priority gaps (commonly tested combinations).

        Returns:
            List of (instrument, timeframe, reason)
        """
        priority_combos = [
            ("BTCUSD", "H1", "crypto_primary"),
            ("BTCUSD", "H4", "crypto_primary"),
            ("BTCUSD", "D1", "crypto_primary"),
            ("EURUSD", "H1", "forex_primary"),
            ("EURUSD", "H4", "forex_primary"),
            ("EURUSD", "D1", "forex_primary"),
            ("US30", "H1", "index_primary"),
            ("US30", "H4", "index_primary"),
            ("XAUUSD", "H1", "metal_primary"),
            ("XAUUSD", "H4", "metal_primary"),
        ]

        priority_gaps = []

        for instrument, timeframe, reason in priority_combos:
            if instrument in self.results and timeframe in self.results[instrument]:
                result = self.results[instrument][timeframe]
                if result["status"] in ["missing", "partial"]:
                    priority_gaps.append((instrument, timeframe, reason))

        return priority_gaps

    def export_csv(self, output_path: Path) -> None:
        """Export results to CSV."""
        rows = []

        for instrument in self.results:
            for timeframe in self.results[instrument]:
                result = self.results[instrument][timeframe]

                rows.append(
                    {
                        "instrument": instrument,
                        "timeframe": timeframe,
                        "status": result["status"],
                        "num_bars": result["num_bars"],
                        "exists": result["exists"],
                        "size_mb": result["metadata"].get("size_mb", 0),
                    }
                )

        df = pd.DataFrame(rows)
        df.to_csv(output_path, index=False)

        print_success(f"Coverage report saved to: {output_path}")

    def export_json(self, output_path: Path) -> None:
        """Export results to JSON."""
        export_data = {
            "timestamp": datetime.now().isoformat(),
            "summary": self.generate_summary(),
            "results": self.results,
            "gaps": self.identify_gaps(),
            "priority_gaps": self.identify_priority_gaps(),
        }

        with open(output_path, "w") as f:
            json.dump(export_data, f, indent=2)

        print_success(f"Coverage report saved to: {output_path}")


# =============================================================================
# MAIN
# =============================================================================


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Audit data coverage for Kinetra",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--instruments",
        nargs="+",
        help="Specific instruments to check (default: all)",
    )
    parser.add_argument(
        "--timeframes",
        nargs="+",
        help="Specific timeframes to check (default: all)",
    )
    parser.add_argument(
        "--min-bars",
        type=int,
        default=MIN_BARS_REQUIRED,
        help=f"Minimum bars for good coverage (default: {MIN_BARS_REQUIRED})",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        help="Data directory (default: data/master_standardized)",
    )
    parser.add_argument(
        "--report",
        type=Path,
        help="Export coverage report to CSV (e.g., data/coverage_report.csv)",
    )
    parser.add_argument(
        "--json",
        type=Path,
        help="Export coverage report to JSON (e.g., data/coverage_report.json)",
    )
    parser.add_argument(
        "--show-gaps",
        action="store_true",
        help="Show detailed gap analysis",
    )

    args = parser.parse_args()

    # Create analyzer
    analyzer = CoverageAnalyzer(
        instruments=args.instruments,
        timeframes=args.timeframes,
        min_bars=args.min_bars,
        data_dir=args.data_dir,
    )

    # Run analysis
    analyzer.analyze()

    # Print summary
    analyzer.print_summary()

    # Show gaps if requested
    if args.show_gaps:
        print_header("GAP ANALYSIS")

        gaps = analyzer.identify_gaps()
        if gaps:
            print(f"Found {len(gaps)} missing combinations:\n")
            for instrument, timeframe in gaps:
                print(f"  ❌ {instrument} {timeframe}")
        else:
            print_success("No gaps found!")

        print("\n")
        priority_gaps = analyzer.identify_priority_gaps()
        if priority_gaps:
            print(f"High-priority gaps ({len(priority_gaps)}):\n")
            for instrument, timeframe, reason in priority_gaps:
                print(f"  🔴 {instrument} {timeframe} ({reason})")
        else:
            print_success("All high-priority combinations covered!")

    # Export reports
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        analyzer.export_csv(args.report)

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        analyzer.export_json(args.json)

    # Return status code based on coverage
    summary = analyzer.generate_summary()
    if summary["coverage_percentage"] >= 80:
        return 0  # Good coverage
    elif summary["coverage_percentage"] >= 50:
        return 1  # Partial coverage
    else:
        return 2  # Poor coverage


if __name__ == "__main__":
    sys.exit(main())
