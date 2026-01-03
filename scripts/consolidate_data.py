#!/usr/bin/env python3
"""
Data Consolidation Script for Kinetra
======================================

Consolidates data from various subdirectories into the standard format
expected by the exhaustive testing framework.

Usage:
    # Create symlinks (recommended - saves space)
    python scripts/consolidate_data.py --symlink

    # Copy files (if symlinks not supported)
    python scripts/consolidate_data.py --copy

    # Preview only (no changes)
    python scripts/consolidate_data.py --dry-run

Philosophy:
- Leverage existing data across the project
- Use symlinks to avoid duplication
- Standardize file naming for test framework
"""

import argparse
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Set

# Project root
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# =============================================================================
# CONFIGURATION
# =============================================================================

# Source directories to scan
SOURCE_DIRS = [
    "data/master_standardized/crypto",
    "data/master_standardized/forex",
    "data/master_standardized/indices",
    "data/master_standardized/metals",
    "data/master_standardized/energy",
    "data/runs/berserker_run3/data",
]

# Target directory
TARGET_DIR = "data/master_standardized"

# Instrument name mappings (source → target)
INSTRUMENT_MAPPINGS = {
    # Crypto
    "BTCUSD": "BTCUSD",
    "BTCJPY": "BTCJPY",
    "ETHUSD": "ETHUSD",
    "ETHEUR": "ETHEUR",
    # Forex
    "EURUSD": "EURUSD",
    "EURUSD+": "EURUSD",
    "GBPUSD": "GBPUSD",
    "GBPUSD+": "GBPUSD",
    "USDJPY": "USDJPY",
    "USDJPY+": "USDJPY",
    "AUDUSD": "AUDUSD",
    "AUDUSD+": "AUDUSD",
    "EURJPY": "EURJPY",
    "EURJPY+": "EURJPY",
    "AUDJPY": "AUDJPY",
    "AUDJPY+": "AUDJPY",
    # Indices
    "US30": "US30",
    "DJ30ft": "US30",
    "NAS100": "NAS100",
    "SPX500": "SPX500",
    "BVSPX": "SPX500",
    "Nikkei225": "NAS100",  # Map to NAS100 as placeholder
    # Metals
    "XAUUSD": "XAUUSD",
    "XAUUSD+": "XAUUSD",
    "XAGUSD": "XAGUSD",
    "XAGUSD+": "XAGUSD",
    # Energy
    "USOIL": "USOIL",
    "UKOUSD": "UKOIL",
    "UKOIL": "UKOIL",
}

# Timeframe mappings
TIMEFRAME_MAPPINGS = {
    "M15": "M15",
    "M30": "M30",
    "H1": "H1",
    "H4": "H4",
    "D1": "D1",
}


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


def print_info(msg: str) -> None:
    """Print info message."""
    print(f"{Colors.BLUE}ℹ️  {msg}{Colors.ENDC}")


# =============================================================================
# FILE PARSER
# =============================================================================


def parse_filename(filename: str) -> Dict[str, str]:
    """
    Parse filename to extract instrument and timeframe.

    Expected format: INSTRUMENT_TIMEFRAME_YYYYMMDDHHMMSS_YYYYMMDDHHMMSS.csv

    Returns:
        Dict with 'instrument', 'timeframe', 'start', 'end' (or None if parse fails)
    """
    parts = filename.replace(".csv", "").split("_")

    if len(parts) < 2:
        return None

    instrument = parts[0]
    timeframe = parts[1]

    # Map to standard names
    instrument_std = INSTRUMENT_MAPPINGS.get(instrument, instrument)
    timeframe_std = TIMEFRAME_MAPPINGS.get(timeframe, timeframe)

    result = {
        "instrument": instrument_std,
        "timeframe": timeframe_std,
        "start": parts[2] if len(parts) > 2 else None,
        "end": parts[3] if len(parts) > 3 else None,
        "original_instrument": instrument,
        "original_timeframe": timeframe,
    }

    return result


def generate_target_filename(instrument: str, timeframe: str) -> str:
    """Generate standard target filename."""
    return f"{instrument}_{timeframe}.csv"


# =============================================================================
# DATA CONSOLIDATOR
# =============================================================================


class DataConsolidator:
    """Consolidates data files from various sources."""

    def __init__(self, target_dir: Path, source_dirs: List[str], dry_run: bool = False):
        """
        Initialize consolidator.

        Args:
            target_dir: Target directory for consolidated data
            source_dirs: List of source directories to scan
            dry_run: If True, only preview actions without executing
        """
        self.target_dir = PROJECT_ROOT / target_dir
        self.source_dirs = [PROJECT_ROOT / d for d in source_dirs]
        self.dry_run = dry_run

        self.files_found = {}  # (instrument, timeframe) -> source_path
        self.actions = []

    def scan_sources(self) -> None:
        """Scan source directories for data files."""
        print_header("SCANNING SOURCE DIRECTORIES")

        for source_dir in self.source_dirs:
            if not source_dir.exists():
                print_warning(f"Directory not found: {source_dir}")
                continue

            print_info(f"Scanning: {source_dir}")
            count = 0

            for csv_file in source_dir.glob("*.csv"):
                parsed = parse_filename(csv_file.name)

                if not parsed:
                    continue

                instrument = parsed["instrument"]
                timeframe = parsed["timeframe"]

                # Skip if not in standard instruments
                if instrument not in INSTRUMENT_MAPPINGS.values():
                    continue

                # Skip if not in standard timeframes
                if timeframe not in TIMEFRAME_MAPPINGS.values():
                    continue

                key = (instrument, timeframe)

                # Keep first occurrence (or choose latest based on file size)
                if key not in self.files_found:
                    self.files_found[key] = csv_file
                    count += 1
                else:
                    # Choose larger file (more data)
                    existing_size = self.files_found[key].stat().st_size
                    new_size = csv_file.stat().st_size

                    if new_size > existing_size:
                        self.files_found[key] = csv_file
                        print_info(
                            f"  Replacing {instrument}_{timeframe} with larger file ({new_size / 1024 / 1024:.1f} MB)"
                        )

            print_success(f"Found {count} usable files in {source_dir.name}")

        print_success(f"\nTotal unique combinations found: {len(self.files_found)}")

    def plan_consolidation(self, method: str = "symlink") -> None:
        """
        Plan consolidation actions.

        Args:
            method: 'symlink' or 'copy'
        """
        print_header(f"PLANNING CONSOLIDATION ({method.upper()})")

        for (instrument, timeframe), source_path in sorted(self.files_found.items()):
            target_filename = generate_target_filename(instrument, timeframe)
            target_path = self.target_dir / target_filename

            action = {
                "instrument": instrument,
                "timeframe": timeframe,
                "source": source_path,
                "target": target_path,
                "method": method,
                "exists": target_path.exists(),
            }

            self.actions.append(action)

            # Print action
            status = "EXISTS" if action["exists"] else "NEW"
            size_mb = source_path.stat().st_size / (1024 * 1024)
            print(
                f"{'⚠️' if action['exists'] else '📄'} {instrument:8} {timeframe:4} → {target_filename:20} ({size_mb:6.1f} MB) [{status}]"
            )

        print_success(f"\nPlanned {len(self.actions)} actions")

        # Summary
        new_count = sum(1 for a in self.actions if not a["exists"])
        existing_count = sum(1 for a in self.actions if a["exists"])

        print_info(f"New files: {new_count}")
        print_info(f"Existing files (will skip): {existing_count}")

    def execute_consolidation(self, overwrite: bool = False) -> None:
        """
        Execute planned consolidation.

        Args:
            overwrite: If True, overwrite existing files
        """
        if self.dry_run:
            print_header("DRY RUN - NO CHANGES MADE")
            print_info("Remove --dry-run to execute actions")
            return

        print_header("EXECUTING CONSOLIDATION")

        # Create target directory
        self.target_dir.mkdir(parents=True, exist_ok=True)

        success_count = 0
        skip_count = 0
        error_count = 0

        for action in self.actions:
            target_path = action["target"]
            source_path = action["source"]

            # Skip existing unless overwrite
            if target_path.exists() and not overwrite:
                print_warning(f"Skipping {target_path.name} (already exists)")
                skip_count += 1
                continue

            # Remove existing if overwriting
            if target_path.exists() and overwrite:
                if target_path.is_symlink():
                    target_path.unlink()
                else:
                    target_path.unlink()

            try:
                if action["method"] == "symlink":
                    # Create relative symlink
                    target_path.symlink_to(source_path.resolve())
                    print_success(f"Linked: {target_path.name}")
                else:
                    # Copy file
                    shutil.copy2(source_path, target_path)
                    print_success(f"Copied: {target_path.name}")

                success_count += 1

            except Exception as e:
                print_error(f"Failed: {target_path.name} - {e}")
                error_count += 1

        # Summary
        print_header("CONSOLIDATION COMPLETE")
        print_success(f"Successful: {success_count}")
        if skip_count > 0:
            print_info(f"Skipped: {skip_count}")
        if error_count > 0:
            print_error(f"Errors: {error_count}")


# =============================================================================
# MAIN
# =============================================================================


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Consolidate data files for Kinetra testing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Action selection
    action_group = parser.add_mutually_exclusive_group(required=True)
    action_group.add_argument(
        "--symlink",
        action="store_true",
        help="Create symlinks (recommended - saves space)",
    )
    action_group.add_argument(
        "--copy",
        action="store_true",
        help="Copy files (use if symlinks not supported)",
    )
    action_group.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview actions without executing",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing files",
    )
    parser.add_argument(
        "--target-dir",
        type=Path,
        default=TARGET_DIR,
        help=f"Target directory (default: {TARGET_DIR})",
    )

    args = parser.parse_args()

    # Determine method
    if args.symlink:
        method = "symlink"
        dry_run = False
    elif args.copy:
        method = "copy"
        dry_run = False
    else:
        method = "symlink"
        dry_run = True

    # Create consolidator
    consolidator = DataConsolidator(
        target_dir=args.target_dir, source_dirs=SOURCE_DIRS, dry_run=dry_run
    )

    # Scan sources
    consolidator.scan_sources()

    # Plan consolidation
    consolidator.plan_consolidation(method=method)

    # Execute (unless dry run)
    consolidator.execute_consolidation(overwrite=args.overwrite)

    # Run coverage audit after consolidation
    if not dry_run:
        print_header("RUNNING COVERAGE AUDIT")
        print_info("Verifying consolidated data...")

        # Import and run audit
        try:
            import subprocess

            result = subprocess.run(
                [
                    sys.executable,
                    "scripts/audit_data_coverage.py",
                    "--show-gaps",
                ],
                cwd=PROJECT_ROOT,
            )
            return result.returncode
        except Exception as e:
            print_warning(f"Could not run coverage audit: {e}")
            return 0

    return 0


if __name__ == "__main__":
    sys.exit(main())
