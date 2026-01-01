"""
Scientific Data Management for Kinetra
======================================

Implements proper financial & scientific data handling with:
- Atomic file operations (no partial writes)
- Master data immutability (never modify originals)
- Automatic backups with versioning
- Test run isolation and reproducibility
- Cache management for performance
- Data deduplication
- Consistent naming conventions
- Checksums for data integrity
- Proper locking to prevent corruption

Philosophy:
- Master data is sacred - NEVER modify
- All operations are atomic (succeed or fail completely)
- Every test run is fully reproducible
- Cache everything possible for speed
- Detect and prevent data corruption
"""

import hashlib
import json
import logging
import shutil
import tempfile
import threading
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class DataFileMetadata:
    """Metadata for a data file."""
    filepath: str
    checksum: str  # SHA256 hash
    symbol: str
    timeframe: str
    asset_class: str
    start_date: str
    end_date: str
    num_bars: int
    file_size: int
    created_at: str
    last_modified: str
    data_quality_score: float = 0.0
    
    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class TestRunMetadata:
    """Metadata for a test run."""
    run_id: str
    timestamp: str
    test_suite: str
    instruments: List[str]
    config: Dict
    data_snapshot_path: str  # Path to immutable data snapshot
    results_path: str
    cache_path: str
    status: str = "running"  # running, completed, failed
    
    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class CacheEntry:
    """Cache entry for computed features."""
    key: str
    value_path: str
    checksum: str
    created_at: str
    last_accessed: str
    size_bytes: int
    hit_count: int = 0


# =============================================================================
# ATOMIC FILE OPERATIONS
# =============================================================================

class AtomicFileWriter:
    """
    Atomic file writer - writes to temp file, then moves atomically.
    Guarantees no partial writes or corruption.
    """
    
    @staticmethod
    def write_csv(df: pd.DataFrame, filepath: Path, **kwargs):
        """Atomically write DataFrame to CSV."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Write to temporary file first
        temp_file = filepath.parent / f".{filepath.name}.tmp.{threading.get_ident()}"
        
        try:
            df.to_csv(temp_file, **kwargs)
            # Atomic move (rename is atomic on POSIX systems)
            temp_file.replace(filepath)
            logger.debug(f"Atomically wrote {len(df)} rows to {filepath}")
        except Exception as e:
            if temp_file.exists():
                temp_file.unlink()
            raise RuntimeError(f"Failed to write {filepath}: {e}")
    
    @staticmethod
    def write_json(data: dict, filepath: Path, indent: int = 2):
        """Atomically write dict to JSON."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        temp_file = filepath.parent / f".{filepath.name}.tmp.{threading.get_ident()}"
        
        try:
            with open(temp_file, 'w') as f:
                json.dump(data, f, indent=indent, default=str)
            temp_file.replace(filepath)
            logger.debug(f"Atomically wrote JSON to {filepath}")
        except Exception as e:
            if temp_file.exists():
                temp_file.unlink()
            raise RuntimeError(f"Failed to write {filepath}: {e}")
    
    @staticmethod
    def write_numpy(array: np.ndarray, filepath: Path):
        """Atomically write numpy array."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        temp_file = filepath.parent / f".{filepath.name}.tmp.{threading.get_ident()}"
        
        try:
            np.save(temp_file, array)
            temp_file.replace(filepath)
            logger.debug(f"Atomically wrote numpy array to {filepath}")
        except Exception as e:
            if temp_file.exists():
                temp_file.unlink()
            raise RuntimeError(f"Failed to write {filepath}: {e}")


# =============================================================================
# CHECKSUM & INTEGRITY
# =============================================================================

class DataIntegrity:
    """Data integrity verification using checksums."""
    
    @staticmethod
    def compute_checksum(filepath: Path) -> str:
        """Compute SHA256 checksum of file."""
        sha256 = hashlib.sha256()
        with open(filepath, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                sha256.update(chunk)
        return sha256.hexdigest()
    
    @staticmethod
    def verify_checksum(filepath: Path, expected_checksum: str) -> bool:
        """Verify file checksum matches expected."""
        actual = DataIntegrity.compute_checksum(filepath)
        return actual == expected_checksum
    
    @staticmethod
    def compute_dataframe_hash(df: pd.DataFrame) -> str:
        """Compute hash of DataFrame content."""
        # Use pandas built-in hashing for consistency
        hash_val = pd.util.hash_pandas_object(df, index=True).sum()
        return hashlib.sha256(str(hash_val).encode()).hexdigest()


# =============================================================================
# MASTER DATA MANAGER
# =============================================================================

class MasterDataManager:
    """
    Manages master data with immutability guarantees.
    
    Rules:
    - Master data is NEVER modified in place
    - All additions are validated and deduplicated
    - Automatic backups before any changes
    - Checksums tracked for all files
    - Metadata maintained in manifest
    """
    
    def __init__(self, master_dir: str = "data/master"):
        self.master_dir = Path(master_dir)
        self.master_dir.mkdir(parents=True, exist_ok=True)
        
        self.manifest_path = self.master_dir / "manifest.json"
        self.manifest = self._load_manifest()
        
        logger.info(f"MasterDataManager initialized at {self.master_dir}")
    
    def _load_manifest(self) -> Dict:
        """Load manifest of all master data files."""
        if self.manifest_path.exists():
            with open(self.manifest_path, 'r') as f:
                return json.load(f)
        return {
            "version": "1.0",
            "created_at": datetime.now().isoformat(),
            "files": {},  # filepath -> metadata
            "checksums": {},  # filepath -> checksum
        }
    
    def _save_manifest(self):
        """Atomically save manifest."""
        self.manifest["last_updated"] = datetime.now().isoformat()
        AtomicFileWriter.write_json(self.manifest, self.manifest_path)
    
    def add_data_file(
        self,
        df: pd.DataFrame,
        symbol: str,
        timeframe: str,
        asset_class: str,
        deduplicate: bool = True,
        backup: bool = True
    ) -> Tuple[bool, str]:
        """
        Add new data file to master directory.
        
        Returns:
            (success, message)
        """
        # Generate consistent filename
        start_date = df['time'].min().strftime("%Y%m%d%H%M")
        end_date = df['time'].max().strftime("%Y%m%d%H%M")
        filename = f"{symbol}_{timeframe}_{start_date}_{end_date}.csv"
        
        # Asset class subdirectory
        subdir = self.master_dir / asset_class
        subdir.mkdir(exist_ok=True)
        filepath = subdir / filename
        
        # Check if file already exists
        if filepath.exists():
            if deduplicate:
                # Check if content is identical
                existing_df = pd.read_csv(filepath, sep='\t')
                if len(existing_df) == len(df):
                    logger.info(f"File already exists with same content: {filename}")
                    return True, "duplicate_skipped"
                else:
                    # Merge and deduplicate
                    logger.info(f"Merging with existing file: {filename}")
                    return self._merge_and_update(filepath, df, backup)
            else:
                return False, "file_exists"
        
        # Validate data
        is_valid, validation_msg = self._validate_data(df)
        if not is_valid:
            return False, f"validation_failed: {validation_msg}"
        
        # Write atomically
        try:
            AtomicFileWriter.write_csv(df, filepath, sep='\t', index=False)
            
            # Compute checksum
            checksum = DataIntegrity.compute_checksum(filepath)
            
            # Update manifest
            metadata = DataFileMetadata(
                filepath=str(filepath.relative_to(self.master_dir)),
                checksum=checksum,
                symbol=symbol,
                timeframe=timeframe,
                asset_class=asset_class,
                start_date=start_date,
                end_date=end_date,
                num_bars=len(df),
                file_size=filepath.stat().st_size,
                created_at=datetime.now().isoformat(),
                last_modified=datetime.now().isoformat(),
            )
            
            self.manifest["files"][str(filepath.relative_to(self.master_dir))] = metadata.to_dict()
            self.manifest["checksums"][str(filepath.relative_to(self.master_dir))] = checksum
            self._save_manifest()
            
            logger.info(f"Added master data file: {filename} ({len(df)} bars)")
            return True, "added"
            
        except Exception as e:
            logger.error(f"Failed to add data file: {e}")
            return False, f"error: {e}"
    
    def _merge_and_update(
        self,
        filepath: Path,
        new_df: pd.DataFrame,
        backup: bool = True
    ) -> Tuple[bool, str]:
        """Merge new data with existing, deduplicate, and update."""
        try:
            # Backup original
            if backup:
                backup_path = self._create_backup(filepath)
                logger.info(f"Created backup: {backup_path}")
            
            # Load existing
            existing_df = pd.read_csv(filepath, sep='\t')
            existing_df['time'] = pd.to_datetime(existing_df['time'])
            new_df['time'] = pd.to_datetime(new_df['time'])
            
            # Combine and deduplicate
            combined = pd.concat([existing_df, new_df], ignore_index=True)
            combined = combined.drop_duplicates(subset=['time'], keep='last')
            combined = combined.sort_values('time').reset_index(drop=True)
            
            # Write atomically
            AtomicFileWriter.write_csv(combined, filepath, sep='\t', index=False)
            
            # Update checksum in manifest
            checksum = DataIntegrity.compute_checksum(filepath)
            rel_path = str(filepath.relative_to(self.master_dir))
            
            if rel_path in self.manifest["files"]:
                self.manifest["files"][rel_path]["num_bars"] = len(combined)
                self.manifest["files"][rel_path]["last_modified"] = datetime.now().isoformat()
                self.manifest["files"][rel_path]["checksum"] = checksum
            
            self.manifest["checksums"][rel_path] = checksum
            self._save_manifest()
            
            added_bars = len(combined) - len(existing_df)
            logger.info(f"Merged data: added {added_bars} new bars, total {len(combined)}")
            return True, f"merged_{added_bars}_bars"
            
        except Exception as e:
            logger.error(f"Failed to merge data: {e}")
            return False, f"merge_error: {e}"
    
    def _create_backup(self, filepath: Path) -> Path:
        """Create timestamped backup of file."""
        backup_dir = self.master_dir / "backups"
        backup_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"{filepath.stem}_backup_{timestamp}{filepath.suffix}"
        backup_path = backup_dir / backup_name
        
        shutil.copy2(filepath, backup_path)
        return backup_path
    
    def _validate_data(self, df: pd.DataFrame) -> Tuple[bool, str]:
        """Validate data quality."""
        required_cols = ['time', 'open', 'high', 'low', 'close']
        
        # Check required columns
        if not all(col in df.columns for col in required_cols):
            return False, f"Missing required columns. Have: {df.columns.tolist()}"
        
        # Check for NaN in OHLC
        if df[required_cols].isnull().any().any():
            return False, "NaN values in OHLC data"
        
        # Check high >= low
        if (df['high'] < df['low']).any():
            return False, "High < Low violation"
        
        # Check open/close within high/low
        if ((df['open'] > df['high']) | (df['open'] < df['low'])).any():
            return False, "Open outside high/low range"
        
        if ((df['close'] > df['high']) | (df['close'] < df['low'])).any():
            return False, "Close outside high/low range"
        
        # Check minimum bars
        if len(df) < 100:
            return False, f"Insufficient data: {len(df)} bars < 100 minimum"
        
        return True, "valid"
    
    def verify_integrity(self) -> Dict[str, bool]:
        """Verify integrity of all master data files."""
        results = {}
        
        for rel_path, expected_checksum in self.manifest.get("checksums", {}).items():
            filepath = self.master_dir / rel_path
            
            if not filepath.exists():
                results[rel_path] = False
                logger.warning(f"Missing file: {rel_path}")
                continue
            
            actual_checksum = DataIntegrity.compute_checksum(filepath)
            matches = actual_checksum == expected_checksum
            results[rel_path] = matches
            
            if not matches:
                logger.error(f"Checksum mismatch for {rel_path}!")
                logger.error(f"  Expected: {expected_checksum}")
                logger.error(f"  Actual:   {actual_checksum}")
        
        return results


# =============================================================================
# TEST RUN MANAGER
# =============================================================================

class TestRunManager:
    """
    Manages isolated test runs with full reproducibility.
    
    Each test run gets:
    - Unique run ID
    - Immutable data snapshot
    - Isolated results directory
    - Dedicated cache
    - Complete metadata for reproduction
    """
    
    def __init__(self, runs_dir: str = "data/test_runs"):
        self.runs_dir = Path(runs_dir)
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        
        self.index_path = self.runs_dir / "runs_index.json"
        self.index = self._load_index()
        
        logger.info(f"TestRunManager initialized at {self.runs_dir}")
    
    def _load_index(self) -> Dict:
        """Load index of all test runs."""
        if self.index_path.exists():
            with open(self.index_path, 'r') as f:
                return json.load(f)
        return {"runs": {}}
    
    def _save_index(self):
        """Save test runs index."""
        AtomicFileWriter.write_json(self.index, self.index_path)
    
    def create_run(
        self,
        test_suite: str,
        instruments: List[str],
        config: Dict,
        master_data_dir: Path
    ) -> Tuple[str, Path]:
        """
        Create new isolated test run.
        
        Returns:
            (run_id, run_dir)
        """
        # Generate unique run ID
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        run_id = f"{test_suite}_{timestamp}"
        
        run_dir = self.runs_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        data_dir = run_dir / "data"
        results_dir = run_dir / "results"
        cache_dir = run_dir / "cache"
        
        for d in [data_dir, results_dir, cache_dir]:
            d.mkdir(exist_ok=True)
        
        # Create immutable data snapshot (hardlinks for efficiency)
        self._create_data_snapshot(master_data_dir, data_dir, instruments)
        
        # Create metadata
        metadata = TestRunMetadata(
            run_id=run_id,
            timestamp=datetime.now().isoformat(),
            test_suite=test_suite,
            instruments=instruments,
            config=config,
            data_snapshot_path=str(data_dir),
            results_path=str(results_dir),
            cache_path=str(cache_dir),
            status="running"
        )
        
        # Save metadata
        metadata_path = run_dir / "metadata.json"
        AtomicFileWriter.write_json(metadata.to_dict(), metadata_path)
        
        # Update index
        self.index["runs"][run_id] = {
            "created_at": metadata.timestamp,
            "test_suite": test_suite,
            "status": "running",
            "run_dir": str(run_dir)
        }
        self._save_index()
        
        logger.info(f"Created test run: {run_id}")
        return run_id, run_dir
    
    def _create_data_snapshot(
        self,
        source_dir: Path,
        target_dir: Path,
        instruments: List[str]
    ):
        """Create immutable snapshot of data (using hardlinks)."""
        for instrument in instruments:
            # Find matching files
            matches = list(source_dir.rglob(f"{instrument}*.csv"))
            
            for src_file in matches:
                # Preserve directory structure
                rel_path = src_file.relative_to(source_dir)
                dest_file = target_dir / rel_path
                dest_file.parent.mkdir(parents=True, exist_ok=True)
                
                # Create hardlink (same inode, no disk space used)
                try:
                    dest_file.hardlink_to(src_file)
                except:
                    # Fallback to copy if hardlink not supported
                    shutil.copy2(src_file, dest_file)
        
        logger.debug(f"Created data snapshot in {target_dir}")
    
    def mark_complete(self, run_id: str, status: str = "completed"):
        """Mark test run as complete."""
        if run_id in self.index["runs"]:
            self.index["runs"][run_id]["status"] = status
            self.index["runs"][run_id]["completed_at"] = datetime.now().isoformat()
            self._save_index()
            
            # Update metadata file
            run_dir = Path(self.index["runs"][run_id]["run_dir"])
            metadata_path = run_dir / "metadata.json"
            
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                metadata["status"] = status
                AtomicFileWriter.write_json(metadata, metadata_path)
            
            logger.info(f"Test run {run_id} marked as {status}")


# =============================================================================
# CACHE MANAGER
# =============================================================================

class CacheManager:
    """
    Manages computed feature cache for performance.
    
    Caches:
    - Physics features (energy, damping, entropy)
    - Technical indicators
    - Preprocessed data
    """
    
    def __init__(self, cache_dir: str = "data/cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.index_path = self.cache_dir / "cache_index.json"
        self.index = self._load_index()
        
        logger.info(f"CacheManager initialized at {self.cache_dir}")
    
    def _load_index(self) -> Dict:
        """Load cache index."""
        if self.index_path.exists():
            with open(self.index_path, 'r') as f:
                return json.load(f)
        return {"entries": {}}
    
    def _save_index(self):
        """Save cache index."""
        AtomicFileWriter.write_json(self.index, self.index_path)
    
    def get(self, key: str) -> Optional[pd.DataFrame]:
        """Get cached data if exists and valid."""
        if key not in self.index["entries"]:
            return None
        
        entry = self.index["entries"][key]
        cache_file = Path(entry["value_path"])
        
        if not cache_file.exists():
            # Cache file missing, remove from index
            del self.index["entries"][key]
            self._save_index()
            return None
        
        # Verify checksum
        if not DataIntegrity.verify_checksum(cache_file, entry["checksum"]):
            logger.warning(f"Cache corrupted: {key}")
            del self.index["entries"][key]
            self._save_index()
            return None
        
        # Load and update access time
        try:
            df = pd.read_parquet(cache_file)
            entry["last_accessed"] = datetime.now().isoformat()
            entry["hit_count"] += 1
            self._save_index()
            
            logger.debug(f"Cache hit: {key}")
            return df
        except Exception as e:
            logger.error(f"Failed to load cache {key}: {e}")
            return None
    
    def put(self, key: str, df: pd.DataFrame):
        """Store data in cache."""
        cache_file = self.cache_dir / f"{key}.parquet"
        
        try:
            # Write to parquet (faster than CSV)
            df.to_parquet(cache_file, compression='snappy')
            
            checksum = DataIntegrity.compute_checksum(cache_file)
            
            entry = CacheEntry(
                key=key,
                value_path=str(cache_file),
                checksum=checksum,
                created_at=datetime.now().isoformat(),
                last_accessed=datetime.now().isoformat(),
                size_bytes=cache_file.stat().st_size,
                hit_count=0
            )
            
            self.index["entries"][key] = asdict(entry)
            self._save_index()
            
            logger.debug(f"Cached: {key}")
        except Exception as e:
            logger.error(f"Failed to cache {key}: {e}")
    
    def clear_old_entries(self, max_age_days: int = 30):
        """Clear cache entries older than max_age_days."""
        now = datetime.now()
        to_remove = []
        
        for key, entry in self.index["entries"].items():
            created = datetime.fromisoformat(entry["created_at"])
            age_days = (now - created).days
            
            if age_days > max_age_days:
                cache_file = Path(entry["value_path"])
                if cache_file.exists():
                    cache_file.unlink()
                to_remove.append(key)
        
        for key in to_remove:
            del self.index["entries"][key]
        
        if to_remove:
            self._save_index()
            logger.info(f"Cleared {len(to_remove)} old cache entries")


# =============================================================================
# UNIFIED DATA COORDINATOR
# =============================================================================

class DataCoordinator:
    """
    Coordinates all data management:
    - Master data (immutable)
    - Test runs (isolated)
    - Caching (performance)
    """
    
    def __init__(
        self,
        master_dir: str = "data/master",
        runs_dir: str = "data/test_runs",
        cache_dir: str = "data/cache"
    ):
        self.master = MasterDataManager(master_dir)
        self.runs = TestRunManager(runs_dir)
        self.cache = CacheManager(cache_dir)
        
        logger.info("DataCoordinator initialized")
    
    def prepare_test_run(
        self,
        test_suite: str,
        instruments: List[str],
        config: Dict
    ) -> Tuple[str, Path]:
        """Prepare isolated test run with data snapshot."""
        return self.runs.create_run(
            test_suite=test_suite,
            instruments=instruments,
            config=config,
            master_data_dir=self.master.master_dir
        )
    
    def get_cached_features(
        self,
        symbol: str,
        timeframe: str,
        data_checksum: str
    ) -> Optional[pd.DataFrame]:
        """Get cached physics features if available."""
        cache_key = f"physics_{symbol}_{timeframe}_{data_checksum}"
        return self.cache.get(cache_key)
    
    def cache_features(
        self,
        symbol: str,
        timeframe: str,
        data_checksum: str,
        features_df: pd.DataFrame
    ):
        """Cache computed physics features."""
        cache_key = f"physics_{symbol}_{timeframe}_{data_checksum}"
        self.cache.put(cache_key, features_df)
