"""
Data Management System for Kinetra
===================================

Manages data downloads with:
- Broker/Account/Asset class organization
- Atomic downloads with chunking & integrity checks
- Gap detection and continuous filling
- Raw → Training data standardization pipeline

Directory Structure:
    data/
    ├── raw/                                    # Raw broker data (immutable, append-only)
    │   └── {broker}/                           # e.g., "VantageInternational"
    │       └── {account_login}/                # e.g., "12345678"
    │           ├── forex/                      # Asset class subfolder
    │           │   ├── EURJPY+_M15_*.csv       # Raw OHLCV data
    │           │   ├── EURJPY+_M15_metadata.json
    │           │   └── EURJPY+_M15_integrity.json
    │           ├── metals/
    │           ├── indices/
    │           └── crypto/
    ├── training/                               # Standardized training data (regenerated fresh each time)
    │   ├── forex/
    │   │   ├── EURJPY_M15_standardized.parquet # Standardized format
    │   │   └── EURJPY_M15_stats.json           # Normalization stats
    │   ├── metals/
    │   └── indices/
    └── cache/                                  # Temporary downloads

Philosophy:
- **Raw data**: Immutable, append-only, source of truth
- **Training data**: Ephemeral, regenerated fresh before each training run
- **Consistency**: Same raw data + same preprocessing = identical training data
"""

import hashlib
import json
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class DataChunk:
    """Represents a downloaded data chunk."""

    symbol: str
    timeframe: str
    start_time: datetime
    end_time: datetime
    bars_count: int
    file_path: Path
    checksum: str
    downloaded_at: datetime


class DataManager:
    """
    Manages data downloads with atomic operations and integrity checks.

    Features:
    - Atomic downloads with chunking
    - Integrity verification
    - Gap detection and filling
    - Raw → Training standardization (regenerated fresh each time)
    """

    def __init__(self, base_dir: Path = None):
        """
        Initialize data manager.

        Args:
            base_dir: Base directory for all data (default: ./data)
        """
        if base_dir is None:
            # Default to project_root/data
            base_dir = Path(__file__).parent.parent / "data"

        self.base_dir = base_dir
        self.raw_dir = self.base_dir / "raw"
        self.training_dir = self.base_dir / "training"
        self.cache_dir = self.base_dir / "cache"

        # Create directories
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.training_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get_asset_class_folder(self, symbol: str) -> str:
        """
        Determine asset class from symbol.

        Args:
            symbol: Trading symbol (e.g., "EURJPY+", "XAUUSD+")

        Returns:
            Asset class folder name
        """
        symbol_clean = symbol.replace("+", "").replace("-", "").upper()

        # Forex pairs (6 chars, all letters)
        if len(symbol_clean) == 6 and symbol_clean.isalpha():
            return "forex"

        # Metals
        if symbol_clean.startswith(("XAU", "XAG", "GOLD", "SILVER")):
            return "metals"

        # Crypto
        if "BTC" in symbol_clean or "ETH" in symbol_clean or "USDT" in symbol_clean:
            return "crypto"

        # Indices
        if any(
            idx in symbol_clean for idx in ["SPX", "NDX", "DJI", "DAX", "FTSE", "NAS100", "US30"]
        ):
            return "indices"

        # Energy
        if any(e in symbol_clean for e in ["WTI", "BRENT", "OIL", "GAS"]):
            return "energy"

        # Default to forex
        return "forex"

    def get_raw_data_path(self, broker: str, account: str, symbol: str, timeframe: str) -> Path:
        """
        Get path for raw data storage.

        Args:
            broker: Broker name (e.g., "VantageInternational")
            account: Account login/ID
            symbol: Trading symbol
            timeframe: Timeframe (e.g., "M15")

        Returns:
            Path to raw data directory
        """
        asset_class = self.get_asset_class_folder(symbol)
        path = self.raw_dir / broker / str(account) / asset_class
        path.mkdir(parents=True, exist_ok=True)
        return path

    def get_training_data_path(self, symbol: str, timeframe: str) -> Path:
        """
        Get path for training data storage.

        Args:
            symbol: Trading symbol
            timeframe: Timeframe

        Returns:
            Path to training data directory
        """
        asset_class = self.get_asset_class_folder(symbol)
        path = self.training_dir / asset_class
        path.mkdir(parents=True, exist_ok=True)
        return path

    def calculate_checksum(self, data: pd.DataFrame) -> str:
        """Calculate checksum for data integrity."""
        # Use hash of concatenated values
        data_str = data.to_csv(index=False)
        return hashlib.sha256(data_str.encode()).hexdigest()[:16]

    def save_raw_data_chunk(
        self,
        data: pd.DataFrame,
        broker: str,
        account: str,
        symbol: str,
        timeframe: str,
        metadata: Dict = None,
    ) -> DataChunk:
        """
        Save a chunk of raw data atomically.

        Args:
            data: DataFrame with OHLCV data
            broker: Broker name
            account: Account ID
            symbol: Symbol
            timeframe: Timeframe
            metadata: Additional metadata

        Returns:
            DataChunk with file info
        """
        # Get storage path
        raw_path = self.get_raw_data_path(broker, account, symbol, timeframe)

        # Create filename with timestamp range
        start_time = data["time"].iloc[0]
        end_time = data["time"].iloc[-1]
        start_str = start_time.strftime("%Y%m%d%H%M")
        end_str = end_time.strftime("%Y%m%d%H%M")

        filename = f"{symbol}_{timeframe}_{start_str}_{end_str}.csv"
        file_path = raw_path / filename

        # Calculate checksum
        checksum = self.calculate_checksum(data)

        # Atomic write: write to temp, then rename
        temp_path = self.cache_dir / f"temp_{filename}"
        data.to_csv(temp_path, index=False)
        shutil.move(str(temp_path), str(file_path))

        # Save metadata
        metadata_file = raw_path / f"{symbol}_{timeframe}_metadata.json"
        meta = metadata or {}
        meta.update(
            {
                "symbol": symbol,
                "timeframe": timeframe,
                "broker": broker,
                "account": account,
                "asset_class": self.get_asset_class_folder(symbol),
                "bars_count": len(data),
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "file": filename,
                "checksum": checksum,
                "downloaded_at": datetime.now().isoformat(),
            }
        )

        with open(metadata_file, "w") as f:
            json.dump(meta, f, indent=2)

        # Create DataChunk
        chunk = DataChunk(
            symbol=symbol,
            timeframe=timeframe,
            start_time=start_time,
            end_time=end_time,
            bars_count=len(data),
            file_path=file_path,
            checksum=checksum,
            downloaded_at=datetime.now(),
        )

        return chunk

    def verify_chunk_integrity(self, chunk: DataChunk) -> bool:
        """
        Verify integrity of a data chunk.

        Args:
            chunk: DataChunk to verify

        Returns:
            True if integrity check passes
        """
        if not chunk.file_path.exists():
            return False

        # Read data
        data = pd.read_csv(chunk.file_path)

        # Verify checksum
        checksum = self.calculate_checksum(data)
        if checksum != chunk.checksum:
            return False

        # Verify bar count
        if len(data) != chunk.bars_count:
            return False

        return True

    def detect_gaps(
        self, broker: str, account: str, symbol: str, timeframe: str
    ) -> List[Tuple[datetime, datetime]]:
        """
        Detect gaps in downloaded data.

        Args:
            broker: Broker name
            account: Account ID
            symbol: Symbol
            timeframe: Timeframe

        Returns:
            List of (gap_start, gap_end) tuples
        """
        raw_path = self.get_raw_data_path(broker, account, symbol, timeframe)

        # Load all chunks
        pattern = f"{symbol}_{timeframe}_*.csv"
        files = sorted(raw_path.glob(pattern))

        if not files:
            return []

        # Parse timestamps from filenames
        chunks = []
        for f in files:
            parts = f.stem.split("_")
            if len(parts) >= 4:
                try:
                    start = datetime.strptime(parts[-2], "%Y%m%d%H%M")
                    end = datetime.strptime(parts[-1], "%Y%m%d%H%M")
                    chunks.append((start, end))
                except ValueError:
                    continue

        # Sort chunks
        chunks.sort()

        # Find gaps
        gaps = []
        for i in range(len(chunks) - 1):
            current_end = chunks[i][1]
            next_start = chunks[i + 1][0]

            # If there's a gap > 1 bar
            timeframe_minutes = self._get_timeframe_minutes(timeframe)
            expected_next = current_end + timedelta(minutes=timeframe_minutes)

            if next_start > expected_next:
                gaps.append((current_end, next_start))

        return gaps

    def _get_timeframe_minutes(self, timeframe: str) -> int:
        """Convert timeframe string to minutes."""
        mapping = {"M1": 1, "M5": 5, "M15": 15, "M30": 30, "H1": 60, "H4": 240, "D1": 1440}
        return mapping.get(timeframe, 15)

    def merge_chunks(self, broker: str, account: str, symbol: str, timeframe: str) -> pd.DataFrame:
        """
        Merge all chunks for a symbol into continuous DataFrame.

        Args:
            broker: Broker name
            account: Account ID
            symbol: Symbol
            timeframe: Timeframe

        Returns:
            Merged DataFrame
        """
        raw_path = self.get_raw_data_path(broker, account, symbol, timeframe)

        # Load all chunks
        pattern = f"{symbol}_{timeframe}_*.csv"
        files = sorted(raw_path.glob(pattern))

        if not files:
            return pd.DataFrame()

        # Read and concatenate - Vectorized with list comprehension
        def read_and_parse(f):
            df = pd.read_csv(f)
            df["time"] = pd.to_datetime(df["time"])
            return df

        dfs = [read_and_parse(f) for f in files]

        # Merge and remove duplicates
        merged = pd.concat(dfs, ignore_index=True)
        merged = merged.drop_duplicates(subset=["time"])
        merged = merged.sort_values("time").reset_index(drop=True)

        return merged

    def prepare_training_data(
        self,
        broker: str,
        account: str,
        symbol: str,
        timeframe: str,
        features: List[str] = None,
        force_refresh: bool = True,
    ) -> Path:
        """
        Prepare standardized training data from raw data.

        **This regenerates fresh from raw data each time for consistency.**

        Process:
        1. Merge all raw chunks
        2. Calculate features
        3. Normalize
        4. Save to training directory

        Args:
            broker: Broker name
            account: Account ID
            symbol: Symbol
            timeframe: Timeframe
            features: List of features to compute
            force_refresh: Always regenerate (default True for consistency)

        Returns:
            Path to standardized file
        """
        training_path = self.get_training_data_path(symbol, timeframe)
        symbol_clean = symbol.replace("+", "").replace("-", "")
        output_file = training_path / f"{symbol_clean}_{timeframe}_standardized.parquet"

        # Always regenerate for consistency (unless explicitly disabled)
        if output_file.exists() and not force_refresh:
            print(f"⚠️  Using existing training data: {output_file}")
            print(f"   Set force_refresh=True to regenerate")
            return output_file

        print(f"\n{'=' * 80}")
        print(f"PREPARING TRAINING DATA (fresh from raw)")
        print(f"{'=' * 80}")
        print(f"Symbol: {symbol}")
        print(f"Timeframe: {timeframe}")

        # Merge raw chunks
        data = self.merge_chunks(broker, account, symbol, timeframe)

        if data.empty:
            raise ValueError(f"No raw data found for {symbol} {timeframe}")

        print(f"✅ Merged {len(data)} bars from raw data")

        # Calculate returns
        data["return"] = data["close"].pct_change()
        data["log_return"] = np.log(data["close"] / data["close"].shift(1))

        # Calculate volatility
        data["volatility"] = data["return"].rolling(20).std()

        # Calculate volume change
        data["volume_change"] = data["volume"].pct_change()

        # Normalize prices (min-max scaling per chunk to preserve distribution)
        price_cols = ["open", "high", "low", "close"]
        for col in price_cols:
            data[f"{col}_norm"] = (data[col] - data[col].min()) / (
                data[col].max() - data[col].min() + 1e-8
            )

        # Save statistics for denormalization
        stats = {
            "symbol": symbol,
            "timeframe": timeframe,
            "broker": broker,
            "account": account,
            "asset_class": self.get_asset_class_folder(symbol),
            "price_min": {col: float(data[col].min()) for col in price_cols},
            "price_max": {col: float(data[col].max()) for col in price_cols},
            "volume_mean": float(data["volume"].mean()),
            "volume_std": float(data["volume"].std()),
            "bars_count": len(data),
            "start_time": data["time"].iloc[0].isoformat(),
            "end_time": data["time"].iloc[-1].isoformat(),
            "created_at": datetime.now().isoformat(),
            "source": "raw_data_merge",
        }

        # Save standardized data as Parquet (more efficient than CSV)
        data.to_parquet(output_file, index=False, compression="snappy")

        # Save stats
        stats_file = training_path / f"{symbol_clean}_{timeframe}_stats.json"
        with open(stats_file, "w") as f:
            json.dump(stats, f, indent=2)

        print(f"✅ Standardized {len(data)} bars → {output_file}")
        print(f"✅ Stats saved → {stats_file}")
        print(f"{'=' * 80}\n")

        return output_file

    def get_data_summary(self, broker: str, account: str) -> Dict:
        """
        Get summary of all downloaded data.

        Args:
            broker: Broker name
            account: Account ID

        Returns:
            Summary dictionary
        """
        account_path = self.raw_dir / broker / str(account)

        if not account_path.exists():
            return {}

        summary = {
            "broker": broker,
            "account": account,
            "asset_classes": {},
            "total_symbols": 0,
            "total_bars": 0,
        }

        # Scan each asset class
        for asset_class_dir in account_path.iterdir():
            if not asset_class_dir.is_dir():
                continue

            asset_class = asset_class_dir.name
            summary["asset_classes"][asset_class] = {
                "symbols": [],
                "total_bars": 0,
            }

            # Scan for metadata files
            for meta_file in asset_class_dir.glob("*_metadata.json"):
                with open(meta_file) as f:
                    meta = json.load(f)

                symbol = meta.get("symbol")
                if symbol not in summary["asset_classes"][asset_class]["symbols"]:
                    summary["asset_classes"][asset_class]["symbols"].append(symbol)
                    summary["total_symbols"] += 1

                bars = meta.get("bars_count", 0)
                summary["asset_classes"][asset_class]["total_bars"] += bars
                summary["total_bars"] += bars

        return summary

    def create_encrypted_backup(
        self, broker: str, account: str, backup_dir: Path = None, password: str = None
    ) -> Path:
        """
        Create encrypted backup of raw data.

        **Raw data becomes irreplaceable over time** - brokers often only
        allow downloading last 2 years. Encrypted backups protect this
        valuable asset.

        Args:
            broker: Broker name
            account: Account ID
            backup_dir: Backup directory (default: ./data/backups)
            password: Encryption password (required)

        Returns:
            Path to encrypted backup file
        """
        import base64
        import getpass
        import tarfile

        from cryptography.fernet import Fernet
        from cryptography.hazmat.primitives import hashes
        from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

        if password is None:
            password = getpass.getpass("Enter backup password: ")

        # Setup backup directory
        if backup_dir is None:
            backup_dir = self.base_dir / "backups"
        backup_dir.mkdir(parents=True, exist_ok=True)

        # Get source data
        source_path = self.raw_dir / broker / str(account)
        if not source_path.exists():
            raise ValueError(f"No data found for {broker}/{account}")

        # Create tar archive
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        tar_filename = f"{broker}_{account}_{timestamp}.tar.gz"
        tar_path = self.cache_dir / tar_filename

        print(f"\n{'=' * 80}")
        print(f"CREATING ENCRYPTED BACKUP")
        print(f"{'=' * 80}")
        print(f"Broker: {broker}")
        print(f"Account: {account}")
        print(f"Source: {source_path}")

        with tarfile.open(tar_path, "w:gz") as tar:
            tar.add(source_path, arcname=f"{broker}_{account}")

        file_size_mb = tar_path.stat().st_size / (1024 * 1024)
        print(f"✅ Created archive: {tar_path.name} ({file_size_mb:.1f} MB)")

        # Derive encryption key from password
        salt = b"kinetra_backup_salt_v1"  # In production, use random salt and store it
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(password.encode()))
        fernet = Fernet(key)

        # Encrypt archive
        with open(tar_path, "rb") as f:
            data = f.read()

        encrypted_data = fernet.encrypt(data)

        # Save encrypted backup
        backup_filename = f"{broker}_{account}_{timestamp}.encrypted"
        backup_path = backup_dir / backup_filename

        with open(backup_path, "wb") as f:
            f.write(encrypted_data)

        encrypted_size_mb = backup_path.stat().st_size / (1024 * 1024)
        print(f"✅ Encrypted backup: {backup_path.name} ({encrypted_size_mb:.1f} MB)")

        # Clean up tar
        tar_path.unlink()

        # Save metadata
        metadata = {
            "broker": broker,
            "account": account,
            "created_at": datetime.now().isoformat(),
            "original_size_mb": file_size_mb,
            "encrypted_size_mb": encrypted_size_mb,
            "file": backup_filename,
        }

        meta_path = backup_dir / f"{broker}_{account}_{timestamp}.json"
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"✅ Metadata saved: {meta_path.name}")
        print(f"{'=' * 80}\n")

        print(f"⚠️  IMPORTANT: Store your password securely!")
        print(f"   Without it, the backup cannot be restored.")

        return backup_path

    def restore_from_backup(
        self, backup_file: Path, password: str = None, restore_dir: Path = None
    ) -> Path:
        """
        Restore raw data from encrypted backup.

        Args:
            backup_file: Path to encrypted backup file
            password: Decryption password (required)
            restore_dir: Where to restore (default: overwrites existing)

        Returns:
            Path to restored data
        """
        import base64
        import getpass
        import tarfile

        from cryptography.fernet import Fernet
        from cryptography.hazmat.primitives import hashes
        from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

        if password is None:
            password = getpass.getpass("Enter backup password: ")

        print(f"\n{'=' * 80}")
        print(f"RESTORING FROM ENCRYPTED BACKUP")
        print(f"{'=' * 80}")
        print(f"Backup: {backup_file.name}")

        # Derive decryption key
        salt = b"kinetra_backup_salt_v1"
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(password.encode()))
        fernet = Fernet(key)

        # Decrypt
        with open(backup_file, "rb") as f:
            encrypted_data = f.read()

        try:
            decrypted_data = fernet.decrypt(encrypted_data)
            print(f"✅ Decryption successful")
        except Exception as e:
            print(f"❌ Decryption failed: {e}")
            print(f"   Check your password and try again.")
            raise

        # Extract tar
        tar_path = self.cache_dir / "restore_temp.tar.gz"
        with open(tar_path, "wb") as f:
            f.write(decrypted_data)

        # Determine restore location
        if restore_dir is None:
            restore_dir = self.raw_dir

        # Extract
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(restore_dir)

        print(f"✅ Restored to: {restore_dir}")

        # Clean up
        tar_path.unlink()

        print(f"{'=' * 80}\n")

        return restore_dir

    def list_backups(self, backup_dir: Path = None) -> List[Dict]:
        """
        List all available backups.

        Args:
            backup_dir: Backup directory (default: ./data/backups)

        Returns:
            List of backup metadata dictionaries
        """
        if backup_dir is None:
            backup_dir = self.base_dir / "backups"

        if not backup_dir.exists():
            return []

        backups = []
        for meta_file in sorted(backup_dir.glob("*.json")):
            with open(meta_file) as f:
                meta = json.load(f)
            backups.append(meta)

        return backups


if __name__ == "__main__":
    print(__doc__)
