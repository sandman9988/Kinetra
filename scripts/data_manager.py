#!/usr/bin/env python3
"""
Kinetra Data Manager - Master Script
=====================================

Consolidated data management following 3-tier architecture:
  TIER 1: Master (raw, unmanipulated, per-broker)
  TIER 2: Prepared (validated, standardized, per-broker)
  TIER 3: Test Snapshots (frozen, reproducible)

REPLACES:
- prepare_data.py
- prepare_exploration_data.py
- standardize_data_cutoff.py
- parallel_data_prep.py
- check_and_fill_data.py
- check_data_integrity.py

Usage:
    # Download raw data (Tier 1)
    python scripts/data_manager.py download --broker VantageInternational

    # Prepare data (Tier 1 → Tier 2)
    python scripts/data_manager.py prepare --broker VantageInternational

    # Create test snapshot (Tier 2 → Tier 3)
    python scripts/data_manager.py snapshot --universe full --name "2026-01-04_baseline"

    # Validate snapshot
    python scripts/data_manager.py validate --snapshot "2026-01-04_baseline"

    # Archive old snapshot
    python scripts/data_manager.py archive --snapshot "2025-12-01_old"

__version__ = "1.0.0"
"""

import argparse
import hashlib
import json
import multiprocessing as mp
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

# Project root
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import after path setup
try:
    from kinetra.persistence_manager import get_persistence_manager
except ImportError:
    print("⚠️  Warning: PersistenceManager not available, using basic file ops")
    get_persistence_manager = None

try:
    from kinetra.cpu_utils import get_optimal_workers
except ImportError:
    # Fallback if cpu_utils not available
    def get_optimal_workers(workload_type: str = "balanced") -> int:
        return max(2, (mp.cpu_count() or 4) // 2)


# ============================================================================
# Data Classes
# ============================================================================


@dataclass
class DataQuality:
    """Data quality metrics."""

    nan_count: int = 0
    duplicate_timestamps: int = 0
    invalid_ohlc: int = 0
    zero_volume_bars: int = 0
    gaps_detected: int = 0


@dataclass
class Gap:
    """Gap in time series."""

    start: str
    end: str
    reason: str
    bars_missing: int


@dataclass
class Holiday:
    """Market holiday."""

    date: str
    name: str
    market_closed: bool


@dataclass
class StatisticalFingerprint:
    """Statistical signature for reproducibility."""

    bars: int
    date_start: str
    date_end: str
    open_mean: float
    high_max: float
    low_min: float
    close_std: float
    volume_mean: float
    checksum_sha256: str


# ============================================================================
# Directory Structure Manager
# ============================================================================


class DirectoryManager:
    """Manage 3-tier directory structure."""

    def __init__(self, base_dir: Path = None):
        self.base_dir = base_dir or PROJECT_ROOT / "data"
        self.master_dir = self.base_dir / "master"
        self.prepared_dir = self.base_dir / "prepared"
        self.snapshots_dir = self.base_dir / "test_snapshots"
        self.archives_dir = self.base_dir / "archives"

    def ensure_structure(self):
        """Create directory structure if not exists."""
        for dir_path in [
            self.master_dir,
            self.prepared_dir,
            self.snapshots_dir,
            self.archives_dir,
        ]:
            dir_path.mkdir(parents=True, exist_ok=True)

    def get_master_path(self, source: str, broker: str) -> Path:
        """Get master data path for broker."""
        path = self.master_dir / source / broker
        path.mkdir(parents=True, exist_ok=True)
        return path

    def get_prepared_path(self, source: str, broker: str) -> Path:
        """Get prepared data path for broker."""
        path = self.prepared_dir / source / broker
        path.mkdir(parents=True, exist_ok=True)
        return path

    def get_snapshot_path(self, snapshot_name: str) -> Path:
        """Get snapshot path."""
        path = self.snapshots_dir / snapshot_name
        return path

    def get_archive_path(self, snapshot_name: str) -> Path:
        """Get archive path."""
        path = self.archives_dir / snapshot_name
        return path


# ============================================================================
# Data Validator
# ============================================================================


class DataValidator:
    """Validate data at each tier with statistical rigor."""

    @staticmethod
    def validate_master(df: pd.DataFrame, filepath: Path) -> Tuple[bool, List[str]]:
        """Validate Tier 1 master data."""
        errors = []

        # Check not empty
        if len(df) == 0:
            errors.append("Empty dataset")
            return False, errors

        # Check basic structure (flexible column names)
        df_lower = df.copy()
        df_lower.columns = (
            df_lower.columns.str.lower().str.strip().str.replace("<", "").str.replace(">", "")
        )

        required = ["open", "high", "low", "close"]
        missing = [col for col in required if col not in df_lower.columns]
        if missing:
            errors.append(f"Missing required columns: {missing}")

        return len(errors) == 0, errors

    @staticmethod
    def validate_prepared(df: pd.DataFrame) -> Tuple[bool, List[str], DataQuality]:
        """Validate Tier 2 prepared data."""
        errors = []
        quality = DataQuality()

        # Check required columns (standardized)
        required = ["time", "open", "high", "low", "close", "volume"]
        missing = [col for col in required if col not in df.columns]
        if missing:
            errors.append(f"Missing standardized columns: {missing}")
            return False, errors, quality

        # Check NaN in OHLC
        ohlc_cols = ["open", "high", "low", "close"]
        quality.nan_count = int(df[ohlc_cols].isna().sum().sum())
        if quality.nan_count > 0:
            errors.append(f"NaN values in OHLC: {quality.nan_count}")

        # Check duplicate timestamps
        quality.duplicate_timestamps = int(df["time"].duplicated().sum())
        if quality.duplicate_timestamps > 0:
            errors.append(f"Duplicate timestamps: {quality.duplicate_timestamps}")

        # Check OHLC logic
        invalid_high = (df["high"] < df["low"]).sum()
        invalid_open = ((df["open"] > df["high"]) | (df["open"] < df["low"])).sum()
        invalid_close = ((df["close"] > df["high"]) | (df["close"] < df["low"])).sum()
        quality.invalid_ohlc = int(invalid_high + invalid_open + invalid_close)
        if quality.invalid_ohlc > 0:
            errors.append(f"Invalid OHLC relationships: {quality.invalid_ohlc}")

        # Check monotonic time
        if not df["time"].is_monotonic_increasing:
            errors.append("Time is not monotonically increasing")

        # Count zero volume bars (warning, not error)
        quality.zero_volume_bars = int((df["volume"] == 0).sum())

        return len(errors) == 0, errors, quality

    @staticmethod
    def calculate_fingerprint(df: pd.DataFrame) -> StatisticalFingerprint:
        """Calculate statistical fingerprint for reproducibility."""
        csv_bytes = df.to_csv(index=False).encode()
        checksum = hashlib.sha256(csv_bytes).hexdigest()

        return StatisticalFingerprint(
            bars=len(df),
            date_start=str(df["time"].min()),
            date_end=str(df["time"].max()),
            open_mean=float(df["open"].mean()),
            high_max=float(df["high"].max()),
            low_min=float(df["low"].min()),
            close_std=float(df["close"].std()),
            volume_mean=float(df["volume"].mean()),
            checksum_sha256=checksum,
        )


# ============================================================================
# Gap Analyzer
# ============================================================================


class GapAnalyzer:
    """Detect and classify gaps - NEVER fill blindly."""

    TRADING_HOURS = {
        "FOREX": {"hours": "24x5", "weekend_gap": True},
        "CRYPTO": {"hours": "24x7", "weekend_gap": False},
        "INDICES": {"hours": "business_hours", "weekend_gap": True},
        "COMMODITIES": {"hours": "varies", "weekend_gap": True},
    }

    @staticmethod
    def detect_gaps(df: pd.DataFrame, expected_interval_minutes: int = 60) -> List[Gap]:
        """Detect gaps in time series."""
        gaps = []

        df["time"] = pd.to_datetime(df["time"])
        df = df.sort_values("time")

        time_diffs = df["time"].diff()
        expected_delta = pd.Timedelta(minutes=expected_interval_minutes)

        # Find gaps larger than expected
        gap_mask = time_diffs > expected_delta * 1.5  # 50% tolerance

        for idx in df[gap_mask].index:
            if idx == 0:
                continue

            prev_time = df.loc[idx - 1, "time"]
            curr_time = df.loc[idx, "time"]
            gap_duration = curr_time - prev_time

            # Classify gap
            reason = GapAnalyzer._classify_gap(prev_time, curr_time, gap_duration)
            bars_missing = int(gap_duration.total_seconds() / 60 / expected_interval_minutes) - 1

            gaps.append(
                Gap(
                    start=prev_time.isoformat(),
                    end=curr_time.isoformat(),
                    reason=reason,
                    bars_missing=bars_missing,
                )
            )

        return gaps

    @staticmethod
    def _classify_gap(start: pd.Timestamp, end: pd.Timestamp, duration: pd.Timedelta) -> str:
        """Classify gap type."""
        # Weekend gap (Friday close to Monday open)
        if start.weekday() == 4 and end.weekday() == 0:  # Friday to Monday
            if duration < pd.Timedelta(days=3):
                return "weekend"

        # Holiday (weekday gap)
        if start.weekday() < 5 and end.weekday() < 5:
            if duration > pd.Timedelta(hours=24):
                return "holiday"

        # Short gap (likely broker issue)
        if duration < pd.Timedelta(hours=2):
            return "data_missing"

        # Long gap (likely outage)
        return "broker_outage"


# ============================================================================
# Parallel Worker Function (Module Level for Pickling)
# ============================================================================


def _prepare_single_worker(args: tuple) -> bool:
    """Worker function for parallel data preparation.

    Must be at module level for multiprocessing pickling.

    Args:
        args: Tuple of (filepath_str, output_dir_str, validate)

    Returns:
        True if preparation succeeded
    """
    filepath_str, output_dir_str, validate = args
    filepath = Path(filepath_str)
    output_dir = Path(output_dir_str)

    try:
        # Create validator and gap analyzer in worker process
        validator = DataValidator()
        gap_analyzer = GapAnalyzer()

        # Read file
        df = pd.read_csv(filepath, sep=None, engine="python")

        # Validate master data
        if validate:
            valid, errors = validator.validate_master(df, filepath)
            if not valid:
                return False

        # Standardize column names
        df = DataPreparator._standardize_columns(df)

        # Validate prepared data
        if validate:
            valid, errors, quality = validator.validate_prepared(df)
            if not valid:
                return False
        else:
            quality = DataQuality()

        # Detect gaps
        timeframe = DataPreparator._detect_timeframe(filepath.stem)
        interval_minutes = DataPreparator._timeframe_to_minutes(timeframe)
        gaps = gap_analyzer.detect_gaps(df, interval_minutes)

        # Calculate fingerprint
        fingerprint = validator.calculate_fingerprint(df)

        # Save prepared data
        output_name = filepath.stem.replace("_raw", "") + ".csv"
        output_path = output_dir / output_name
        df.to_csv(output_path, index=False)

        # Save metadata
        metadata = {
            "source_file": str(filepath),
            "preparation_timestamp": datetime.now(timezone.utc).isoformat(),
            "preparation_version": "1.0.0",
            "statistics": asdict(fingerprint),
            "data_quality": asdict(quality),
            "gaps": [asdict(g) for g in gaps],
            "holidays": [],
            "transformations_applied": [
                "standardize_column_names",
                "validate_ohlc_logic",
                "detect_gaps",
            ],
        }

        meta_path = output_path.with_suffix(".meta.json")
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)

        return True

    except Exception as e:
        print(f"❌ Worker failed for {filepath.name}: {e}")
        return False


# ============================================================================
# Data Preparator
# ============================================================================


class DataPreparator:
    """Transform Tier 1 → Tier 2 with validation."""

    def __init__(self, source_dir: Path, output_dir: Path):
        self.source_dir = source_dir
        self.output_dir = output_dir
        self.validator = DataValidator()
        self.gap_analyzer = GapAnalyzer()

    def prepare_all(self, validate: bool = True, parallel: bool = True) -> Dict[str, bool]:
        """Prepare all files in source directory.

        Args:
            validate: Whether to validate data at each stage
            parallel: Use multiprocessing for parallel preparation (default: True)

        Returns:
            Dict mapping filename to success status
        """
        results = {}

        csv_files = list(self.source_dir.glob("*_raw.csv"))
        if not csv_files:
            # Fallback: any CSV file
            csv_files = list(self.source_dir.glob("*.csv"))

        n_files = len(csv_files)
        print(f"\n📊 Preparing {n_files} file(s) from {self.source_dir.name}")

        # Use parallel processing for multiple files
        if parallel and n_files > 1:
            workers = min(get_optimal_workers("balanced"), n_files)
            print(f"🚀 Using {workers} parallel workers")

            # Prepare arguments for parallel execution
            prepare_args = [(str(fp), str(self.output_dir), validate) for fp in csv_files]

            with ProcessPoolExecutor(max_workers=workers) as executor:
                futures = {
                    executor.submit(_prepare_single_worker, args): args[0] for args in prepare_args
                }

                for future in tqdm(as_completed(futures), total=n_files, desc="Preparing"):
                    filepath_str = futures[future]
                    filename = Path(filepath_str).name
                    try:
                        success = future.result()
                        results[filename] = success
                    except Exception as e:
                        print(f"❌ Failed to prepare {filename}: {e}")
                        results[filename] = False
        else:
            # Sequential processing for single file or when parallel disabled
            for filepath in tqdm(csv_files, desc="Preparing"):
                try:
                    success = self.prepare_single(filepath, validate=validate)
                    results[filepath.name] = success
                except Exception as e:
                    print(f"❌ Failed to prepare {filepath.name}: {e}")
                    results[filepath.name] = False

        # Summary
        success_count = sum(results.values())
        print(f"\n✅ Successfully prepared: {success_count}/{len(results)}")

        return results

    def prepare_single(
        self, filepath: Path, validate: bool = True, output_dir: Path = None
    ) -> bool:
        """Prepare single file: standardize, validate, detect gaps.

        Args:
            filepath: Path to source file
            validate: Whether to validate data
            output_dir: Override output directory (used by parallel worker)

        Returns:
            True if preparation succeeded
        """
        output_dir = output_dir or self.output_dir
        # Read file
        df = pd.read_csv(filepath, sep=None, engine="python")

        # Validate master data
        if validate:
            valid, errors = self.validator.validate_master(df, filepath)
            if not valid:
                print(f"⚠️  Master validation failed for {filepath.name}: {errors}")
                return False

        # Standardize column names
        df = self._standardize_columns(df)

        # Validate prepared data
        if validate:
            valid, errors, quality = self.validator.validate_prepared(df)
            if not valid:
                print(f"❌ Prepared validation failed for {filepath.name}: {errors}")
                return False

        # Detect gaps
        timeframe = self._detect_timeframe(filepath.stem)
        interval_minutes = self._timeframe_to_minutes(timeframe)
        gaps = self.gap_analyzer.detect_gaps(df, interval_minutes)

        # Calculate fingerprint
        fingerprint = self.validator.calculate_fingerprint(df)

        # Save prepared data
        output_name = filepath.stem.replace("_raw", "") + ".csv"
        output_path = self.output_dir / output_name
        df.to_csv(output_path, index=False)

        # Save metadata
        metadata = {
            "source_file": str(filepath),
            "preparation_timestamp": datetime.now(timezone.utc).isoformat(),
            "preparation_version": __version__,
            "statistics": asdict(fingerprint),
            "data_quality": asdict(quality),
            "gaps": [asdict(g) for g in gaps],
            "holidays": [],  # TODO: Implement holiday detection
            "transformations_applied": [
                "standardize_column_names",
                "validate_ohlc_logic",
                "detect_gaps",
            ],
        }

        meta_path = output_path.with_suffix(".meta.json")
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)

        return True

    @staticmethod
    def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
        """Standardize column names to: time, open, high, low, close, volume."""
        # Normalize to lowercase, remove brackets
        df.columns = df.columns.str.lower().str.strip().str.replace("<", "").str.replace(">", "")

        # Map common column names
        column_map = {
            "date": "time",
            "datetime": "time",
            "timestamp": "time",
            "tickvol": "volume",
            "vol": "volume",
            "tick_volume": "volume",
        }
        df.rename(columns=column_map, inplace=True)

        # Ensure time column exists
        if "time" in df.columns:
            df["time"] = pd.to_datetime(df["time"])
        elif "date" in df.columns and "time" in df.columns:
            # Combine date and time columns if separate
            df["time"] = pd.to_datetime(df["date"].astype(str) + " " + df["time"].astype(str))

        # Ensure volume exists (create if missing)
        if "volume" not in df.columns:
            df["volume"] = 0

        # Select and order standard columns
        standard_cols = ["time", "open", "high", "low", "close", "volume"]
        df = df[standard_cols]

        return df

    @staticmethod
    def _detect_timeframe(filename: str) -> str:
        """Detect timeframe from filename."""
        filename_upper = filename.upper()
        for tf in ["M1", "M5", "M15", "M30", "H1", "H4", "D1", "W1", "MN1"]:
            if tf in filename_upper:
                return tf
        return "H1"  # Default

    @staticmethod
    def _timeframe_to_minutes(timeframe: str) -> int:
        """Convert timeframe to minutes."""
        tf_map = {
            "M1": 1,
            "M5": 5,
            "M15": 15,
            "M30": 30,
            "H1": 60,
            "H4": 240,
            "D1": 1440,
            "W1": 10080,
            "MN1": 43200,
        }
        return tf_map.get(timeframe.upper(), 60)


# ============================================================================
# Snapshot Creator
# ============================================================================


class SnapshotCreator:
    """Create Tier 3 immutable test snapshots."""

    UNIVERSES = {
        "full": {"description": "All brokers, all asset classes, all symbols"},
        "forex_only": {"description": "Forex pairs only"},
        "crypto_only": {"description": "Crypto pairs only"},
        "top_5": {"description": "Top 5 symbols by data quality"},
    }

    def __init__(self, prepared_dir: Path, snapshots_dir: Path):
        self.prepared_dir = prepared_dir
        self.snapshots_dir = snapshots_dir

    def create_snapshot(self, universe: str, name: str, purpose: str = "General testing") -> bool:
        """Create immutable test snapshot."""
        print(f"\n📸 Creating snapshot: {name}")
        print(f"   Universe: {universe}")
        print(f"   Purpose: {purpose}")

        # Create snapshot directory
        snapshot_path = self.snapshots_dir / name
        if snapshot_path.exists():
            print(f"❌ Snapshot already exists: {name}")
            return False

        snapshot_path.mkdir(parents=True)
        data_path = snapshot_path / "data"
        data_path.mkdir()

        # Select files based on universe
        files = self._select_files_for_universe(universe)
        print(f"   Selected {len(files)} file(s)")

        # Copy files to snapshot
        copied_files = []
        for src_file in tqdm(files, desc="Copying"):
            dest_file = data_path / src_file.name
            shutil.copy2(src_file, dest_file)

            # Calculate fingerprint
            df = pd.read_csv(dest_file)
            fingerprint = DataValidator.calculate_fingerprint(df)

            copied_files.append(
                {
                    "file": src_file.name,
                    "symbol": self._extract_symbol(src_file.stem),
                    "timeframe": self._extract_timeframe(src_file.stem),
                    "bars": fingerprint.bars,
                    "checksum_sha256": fingerprint.checksum_sha256,
                }
            )

        # Create manifest
        manifest = {
            "snapshot_id": name,
            "created_timestamp": datetime.now(timezone.utc).isoformat(),
            "purpose": purpose,
            "created_by": f"{Path(__file__).name} v{__version__}",
            "universe": universe,
            "files": copied_files,
            "immutable": True,
            "archive_date": None,
        }

        manifest_path = snapshot_path / "manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

        print(f"✅ Snapshot created: {snapshot_path}")
        return True

    def _select_files_for_universe(self, universe: str) -> List[Path]:
        """Select files based on universe definition."""
        all_files = list(self.prepared_dir.rglob("*.csv"))

        if universe == "full":
            return all_files

        elif universe == "forex_only":
            forex_pairs = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "EURJPY"]
            return [f for f in all_files if any(pair in f.stem for pair in forex_pairs)]

        elif universe == "crypto_only":
            crypto_pairs = ["BTC", "ETH"]
            return [f for f in all_files if any(pair in f.stem for pair in crypto_pairs)]

        elif universe == "top_5":
            # TODO: Implement quality-based selection
            return all_files[:5]

        return all_files

    @staticmethod
    def _extract_symbol(filename: str) -> str:
        """Extract symbol from filename."""
        parts = filename.split("_")
        return parts[0] if parts else "UNKNOWN"

    @staticmethod
    def _extract_timeframe(filename: str) -> str:
        """Extract timeframe from filename."""
        for tf in ["M1", "M5", "M15", "M30", "H1", "H4", "D1", "W1", "MN1"]:
            if tf in filename.upper():
                return tf
        return "H1"


# ============================================================================
# Snapshot Validator
# ============================================================================


class SnapshotValidator:
    """Validate snapshot integrity and immutability."""

    def __init__(self, snapshot_path: Path):
        self.snapshot_path = snapshot_path

    def validate(self) -> Tuple[bool, List[str]]:
        """Validate snapshot integrity."""
        errors = []

        # Check manifest exists
        manifest_path = self.snapshot_path / "manifest.json"
        if not manifest_path.exists():
            errors.append("Manifest file missing")
            return False, errors

        # Load manifest
        with open(manifest_path, "r") as f:
            manifest = json.load(f)

        # Check immutability flag
        if not manifest.get("immutable", False):
            errors.append("Snapshot is not marked as immutable")

        # Validate each file
        data_path = self.snapshot_path / "data"
        for file_info in manifest.get("files", []):
            filepath = data_path / file_info["file"]

            if not filepath.exists():
                errors.append(f"File missing: {file_info['file']}")
                continue

            # Verify checksum
            df = pd.read_csv(filepath)
            current_checksum = DataValidator.calculate_fingerprint(df).checksum_sha256

            if current_checksum != file_info.get("checksum_sha256"):
                errors.append(f"Checksum mismatch: {file_info['file']}")

        return len(errors) == 0, errors


# ============================================================================
# Main CLI
# ============================================================================


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Kinetra Data Manager - 3-Tier Data Architecture",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Prepare data (Tier 1 → Tier 2)
  python scripts/data_manager.py prepare --broker VantageInternational

  # Create test snapshot (Tier 2 → Tier 3)
  python scripts/data_manager.py snapshot --universe full --name "2026-01-04_baseline"

  # Validate snapshot
  python scripts/data_manager.py validate --snapshot "2026-01-04_baseline"
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # Prepare command
    prepare_parser = subparsers.add_parser("prepare", help="Prepare data (Tier 1 → Tier 2)")
    prepare_parser.add_argument("--broker", required=True, help="Broker name")
    prepare_parser.add_argument(
        "--source", default="metaapi", help="Data source (metaapi, mt5_local)"
    )
    prepare_parser.add_argument("--no-validate", action="store_true", help="Skip validation")

    # Snapshot command
    snapshot_parser = subparsers.add_parser(
        "snapshot", help="Create test snapshot (Tier 2 → Tier 3)"
    )
    snapshot_parser.add_argument(
        "--universe",
        required=True,
        choices=["full", "forex_only", "crypto_only", "top_5"],
        help="Universe definition",
    )
    snapshot_parser.add_argument("--name", required=True, help="Snapshot name")
    snapshot_parser.add_argument("--purpose", default="General testing", help="Purpose")

    # Validate command
    validate_parser = subparsers.add_parser("validate", help="Validate snapshot integrity")
    validate_parser.add_argument("--snapshot", required=True, help="Snapshot name")

    # Archive command
    archive_parser = subparsers.add_parser("archive", help="Archive old snapshot")
    archive_parser.add_argument("--snapshot", required=True, help="Snapshot name")
    archive_parser.add_argument("--reason", default="Replaced", help="Archive reason")

    args = parser.parse_args()

    # Initialize directory manager
    dir_mgr = DirectoryManager()
    dir_mgr.ensure_structure()

    if args.command == "prepare":
        # Prepare data
        source_path = dir_mgr.get_master_path(args.source, args.broker)
        output_path = dir_mgr.get_prepared_path(args.source, args.broker)

        if not source_path.exists() or not list(source_path.glob("*.csv")):
            print(f"❌ No data found in: {source_path}")
            sys.exit(1)

        preparator = DataPreparator(source_path, output_path)
        results = preparator.prepare_all(validate=not args.no_validate)

        if all(results.values()):
            print(f"\n✅ All files prepared successfully")
            sys.exit(0)
        else:
            print(f"\n⚠️  Some files failed to prepare")
            sys.exit(1)

    elif args.command == "snapshot":
        # Create snapshot
        prepared_path = dir_mgr.prepared_dir
        snapshots_path = dir_mgr.snapshots_dir

        creator = SnapshotCreator(prepared_path, snapshots_path)
        success = creator.create_snapshot(args.universe, args.name, args.purpose)

        sys.exit(0 if success else 1)

    elif args.command == "validate":
        # Validate snapshot
        snapshot_path = dir_mgr.get_snapshot_path(args.snapshot)

        if not snapshot_path.exists():
            print(f"❌ Snapshot not found: {args.snapshot}")
            sys.exit(1)

        validator = SnapshotValidator(snapshot_path)
        valid, errors = validator.validate()

        if valid:
            print(f"✅ Snapshot valid: {args.snapshot}")
            sys.exit(0)
        else:
            print(f"❌ Snapshot validation failed:")
            for error in errors:
                print(f"   - {error}")
            sys.exit(1)

    elif args.command == "archive":
        # Archive snapshot
        snapshot_path = dir_mgr.get_snapshot_path(args.snapshot)
        archive_path = dir_mgr.get_archive_path(args.snapshot)

        if not snapshot_path.exists():
            print(f"❌ Snapshot not found: {args.snapshot}")
            sys.exit(1)

        if archive_path.exists():
            print(f"❌ Archive already exists: {args.snapshot}")
            sys.exit(1)

        # Move to archive
        shutil.move(str(snapshot_path), str(archive_path))

        # Update manifest
        manifest_path = archive_path / "manifest.json"
        with open(manifest_path, "r") as f:
            manifest = json.load(f)

        manifest["archive_date"] = datetime.now(timezone.utc).isoformat()
        manifest["archive_reason"] = args.reason

        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

        print(f"✅ Snapshot archived: {args.snapshot}")
        sys.exit(0)

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
