"""
Atomic Persistence Module

Crash-safe saving for:
- RL model checkpoints (weights, optimizer, replay buffer)
- MarketWatch streaming data
- Training state and metrics
- Configuration and hyperparameters

All writes are ATOMIC: write to temp file, then rename.
This ensures no partial/corrupt files on crash.
"""

import os
import json
import gzip
import pickle
import shutil
import tempfile
import hashlib
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional, Any, Union
from enum import Enum
import numpy as np

# Try torch for RL model persistence
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class CheckpointType(Enum):
    """Types of checkpoints we can save."""
    RL_MODEL = "rl_model"
    REPLAY_BUFFER = "replay_buffer"
    TRAINING_STATE = "training_state"
    MARKET_DATA = "market_data"
    CONFIG = "config"
    METRICS = "metrics"


@dataclass
class CheckpointMeta:
    """Metadata for a checkpoint."""
    checkpoint_type: str
    timestamp: str
    version: int
    checksum: str
    size_bytes: int
    extra: Dict[str, Any] = field(default_factory=dict)


def _compute_checksum(data: bytes) -> str:
    """Compute SHA256 checksum of data."""
    return hashlib.sha256(data).hexdigest()[:16]


def _atomic_write(path: Path, data: bytes) -> None:
    """
    Atomically write data to file.

    Uses write-to-temp-then-rename pattern for crash safety.
    On POSIX systems, rename is atomic.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Write to temp file in same directory (important for atomic rename)
    fd, temp_path = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp"
    )

    try:
        os.write(fd, data)
        os.fsync(fd)  # Ensure data is on disk
        os.close(fd)

        # Atomic rename
        os.rename(temp_path, path)

        # Sync parent directory (for durability)
        dir_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)

    except Exception:
        # Cleanup temp file on error
        os.close(fd)
        if os.path.exists(temp_path):
            os.unlink(temp_path)
        raise


def _atomic_read(path: Path) -> bytes:
    """Read file contents."""
    with open(path, 'rb') as f:
        return f.read()


class AtomicCheckpointer:
    """
    Atomic checkpointing for RL models and data.

    Features:
    - Atomic writes (crash-safe)
    - Versioned checkpoints with rotation
    - Checksum verification
    - Compression for large data
    - Auto-recovery from latest valid checkpoint

    Usage:
        checkpointer = AtomicCheckpointer("./checkpoints")

        # Save RL model
        checkpointer.save_rl_model(model, optimizer, step=1000)

        # Load latest
        state = checkpointer.load_latest_rl_model()
        model.load_state_dict(state['model'])
    """

    def __init__(
        self,
        base_dir: Union[str, Path],
        max_checkpoints: int = 5,
        compress: bool = True
    ):
        """
        Initialize checkpointer.

        Args:
            base_dir: Directory for checkpoints
            max_checkpoints: Max checkpoints to keep per type (rotation)
            compress: Whether to gzip compress data
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.max_checkpoints = max_checkpoints
        self.compress = compress

        # Subdirectories by type
        for ctype in CheckpointType:
            (self.base_dir / ctype.value).mkdir(exist_ok=True)

    def _get_checkpoint_path(
        self,
        ctype: CheckpointType,
        version: int
    ) -> Path:
        """Get path for a specific checkpoint version."""
        ext = ".pkl.gz" if self.compress else ".pkl"
        return self.base_dir / ctype.value / f"checkpoint_v{version:06d}{ext}"

    def _get_meta_path(self, ctype: CheckpointType, version: int) -> Path:
        """Get path for checkpoint metadata."""
        return self.base_dir / ctype.value / f"checkpoint_v{version:06d}.meta.json"

    def _get_latest_version(self, ctype: CheckpointType) -> int:
        """Find latest checkpoint version for a type."""
        type_dir = self.base_dir / ctype.value
        versions = []

        for f in type_dir.glob("checkpoint_v*.meta.json"):
            try:
                v = int(f.stem.split("_v")[1].split(".")[0])
                versions.append(v)
            except (ValueError, IndexError):
                continue

        return max(versions) if versions else 0

    def _rotate_checkpoints(self, ctype: CheckpointType):
        """Remove old checkpoints beyond max_checkpoints."""
        type_dir = self.base_dir / ctype.value

        # Find all versions
        versions = []
        for f in type_dir.glob("checkpoint_v*.meta.json"):
            try:
                v = int(f.stem.split("_v")[1].split(".")[0])
                versions.append(v)
            except (ValueError, IndexError):
                continue

        versions.sort(reverse=True)

        # Remove old ones
        for v in versions[self.max_checkpoints:]:
            for ext in [".pkl", ".pkl.gz", ".meta.json"]:
                path = type_dir / f"checkpoint_v{v:06d}{ext}"
                if path.exists():
                    path.unlink()

    def _serialize(self, data: Any) -> bytes:
        """Serialize data to bytes."""
        raw = pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)
        if self.compress:
            return gzip.compress(raw)
        return raw

    def _deserialize(self, data: bytes) -> Any:
        """Deserialize bytes to data."""
        if self.compress:
            data = gzip.decompress(data)
        return pickle.loads(data)

    def save(
        self,
        ctype: CheckpointType,
        data: Any,
        extra_meta: Optional[Dict] = None
    ) -> int:
        """
        Save a checkpoint atomically.

        Args:
            ctype: Type of checkpoint
            data: Data to save (must be pickleable)
            extra_meta: Additional metadata to store

        Returns:
            Version number of saved checkpoint
        """
        version = self._get_latest_version(ctype) + 1

        # Serialize data
        serialized = self._serialize(data)
        checksum = _compute_checksum(serialized)

        # Create metadata
        meta = CheckpointMeta(
            checkpoint_type=ctype.value,
            timestamp=datetime.utcnow().isoformat(),
            version=version,
            checksum=checksum,
            size_bytes=len(serialized),
            extra=extra_meta or {}
        )

        # Save data atomically
        data_path = self._get_checkpoint_path(ctype, version)
        _atomic_write(data_path, serialized)

        # Save metadata atomically
        meta_path = self._get_meta_path(ctype, version)
        meta_json = json.dumps(asdict(meta), indent=2).encode('utf-8')
        _atomic_write(meta_path, meta_json)

        # Rotate old checkpoints
        self._rotate_checkpoints(ctype)

        return version

    def load(
        self,
        ctype: CheckpointType,
        version: Optional[int] = None,
        verify_checksum: bool = True
    ) -> Optional[Any]:
        """
        Load a checkpoint.

        Args:
            ctype: Type of checkpoint
            version: Specific version (None = latest)
            verify_checksum: Whether to verify data integrity

        Returns:
            Loaded data or None if not found
        """
        if version is None:
            version = self._get_latest_version(ctype)

        if version == 0:
            return None

        data_path = self._get_checkpoint_path(ctype, version)
        meta_path = self._get_meta_path(ctype, version)

        if not data_path.exists() or not meta_path.exists():
            return None

        # Load and verify
        try:
            data = _atomic_read(data_path)

            if verify_checksum:
                with open(meta_path, 'r') as f:
                    meta = json.load(f)
                checksum = _compute_checksum(data)
                if checksum != meta['checksum']:
                    print(f"Checksum mismatch for {data_path}")
                    return None

            return self._deserialize(data)

        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            return None

    def load_latest_valid(self, ctype: CheckpointType) -> Optional[Any]:
        """
        Load the latest valid checkpoint, trying older ones if corrupt.

        Auto-recovery: walks backwards through versions until finding
        a valid checkpoint.
        """
        latest = self._get_latest_version(ctype)

        for version in range(latest, 0, -1):
            data = self.load(ctype, version, verify_checksum=True)
            if data is not None:
                if version < latest:
                    print(f"Recovered from older checkpoint v{version} (latest was v{latest})")
                return data

        return None

    # =========================================================================
    # RL-specific methods
    # =========================================================================

    def save_rl_model(
        self,
        model: Any,
        optimizer: Any = None,
        step: int = 0,
        episode: int = 0,
        metrics: Optional[Dict] = None
    ) -> int:
        """
        Save RL model checkpoint.

        Args:
            model: PyTorch model or state dict
            optimizer: Optional optimizer
            step: Training step
            episode: Episode number
            metrics: Training metrics
        """
        if HAS_TORCH and hasattr(model, 'state_dict'):
            model_state = model.state_dict()
        else:
            model_state = model

        data = {
            'model': model_state,
            'optimizer': optimizer.state_dict() if optimizer and hasattr(optimizer, 'state_dict') else optimizer,
            'step': step,
            'episode': episode,
            'metrics': metrics or {},
        }

        return self.save(
            CheckpointType.RL_MODEL,
            data,
            extra_meta={'step': step, 'episode': episode}
        )

    def load_latest_rl_model(self) -> Optional[Dict]:
        """Load latest valid RL model checkpoint."""
        return self.load_latest_valid(CheckpointType.RL_MODEL)

    def save_replay_buffer(
        self,
        buffer: Any,
        step: int = 0
    ) -> int:
        """
        Save replay buffer.

        For large buffers, considers using memory-mapped files
        or chunked saving for efficiency.
        """
        # If buffer is very large, we might want to save differently
        # For now, pickle the whole thing
        data = {
            'buffer': buffer,
            'step': step,
            'size': len(buffer) if hasattr(buffer, '__len__') else 0,
        }

        return self.save(
            CheckpointType.REPLAY_BUFFER,
            data,
            extra_meta={'step': step}
        )

    def load_latest_replay_buffer(self) -> Optional[Dict]:
        """Load latest valid replay buffer."""
        return self.load_latest_valid(CheckpointType.REPLAY_BUFFER)

    # =========================================================================
    # MarketWatch data methods
    # =========================================================================

    def save_market_data(
        self,
        data: Dict,
        symbol: str,
        timeframe: str
    ) -> int:
        """
        Save market data snapshot.

        Args:
            data: OHLCV data (dict or DataFrame)
            symbol: Symbol name
            timeframe: Timeframe (H1, M15, etc.)
        """
        import pandas as pd

        if isinstance(data, pd.DataFrame):
            # Convert to dict for pickling
            save_data = {
                'columns': list(data.columns),
                'index': data.index.tolist(),
                'values': data.values.tolist(),
                'index_name': data.index.name,
            }
        else:
            save_data = data

        return self.save(
            CheckpointType.MARKET_DATA,
            {'data': save_data, 'symbol': symbol, 'timeframe': timeframe},
            extra_meta={'symbol': symbol, 'timeframe': timeframe}
        )

    def load_latest_market_data(self) -> Optional[Dict]:
        """Load latest market data."""
        return self.load_latest_valid(CheckpointType.MARKET_DATA)

    # =========================================================================
    # Training state methods
    # =========================================================================

    def save_training_state(
        self,
        state: Dict,
        step: int = 0
    ) -> int:
        """
        Save complete training state.

        Includes everything needed to resume training:
        - Hyperparameters
        - Learning rate schedules
        - Random states
        - Normalization stats
        """
        # Capture random states for reproducibility
        state['numpy_rng_state'] = np.random.get_state()

        if HAS_TORCH:
            state['torch_rng_state'] = torch.get_rng_state()
            if torch.cuda.is_available():
                state['cuda_rng_state'] = torch.cuda.get_rng_state_all()

        state['step'] = step

        return self.save(
            CheckpointType.TRAINING_STATE,
            state,
            extra_meta={'step': step}
        )

    def load_latest_training_state(self) -> Optional[Dict]:
        """Load latest training state and restore RNG."""
        state = self.load_latest_valid(CheckpointType.TRAINING_STATE)

        if state is None:
            return None

        # Restore random states
        if 'numpy_rng_state' in state:
            np.random.set_state(state['numpy_rng_state'])

        if HAS_TORCH and 'torch_rng_state' in state:
            torch.set_rng_state(state['torch_rng_state'])
            if torch.cuda.is_available() and 'cuda_rng_state' in state:
                torch.cuda.set_rng_state_all(state['cuda_rng_state'])

        return state

    # =========================================================================
    # Metrics methods
    # =========================================================================

    def save_metrics(self, metrics: Dict, step: int = 0) -> int:
        """Save training/performance metrics."""
        return self.save(
            CheckpointType.METRICS,
            {'metrics': metrics, 'step': step},
            extra_meta={'step': step}
        )

    def load_all_metrics(self) -> List[Dict]:
        """Load all saved metrics for analysis."""
        type_dir = self.base_dir / CheckpointType.METRICS.value
        all_metrics = []

        for meta_file in sorted(type_dir.glob("*.meta.json")):
            try:
                version = int(meta_file.stem.split("_v")[1].split(".")[0])
                data = self.load(CheckpointType.METRICS, version)
                if data:
                    all_metrics.append(data)
            except (ValueError, IndexError, FileNotFoundError):
                continue

        return all_metrics


class StreamingDataPersister:
    """
    Persist streaming market data with periodic snapshots.

    For real-time data that needs to survive crashes.
    Uses append-only log with periodic compaction.
    """

    def __init__(
        self,
        base_dir: Union[str, Path],
        symbol: str,
        flush_interval: int = 100,  # Flush every N records
        snapshot_interval: int = 1000  # Snapshot every N records
    ):
        self.base_dir = Path(base_dir) / "streaming" / symbol
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.symbol = symbol
        self.flush_interval = flush_interval
        self.snapshot_interval = snapshot_interval

        self.buffer: List[Dict] = []
        self.record_count = 0
        self.log_file = self.base_dir / "append.log"
        self.snapshot_file = self.base_dir / "snapshot.pkl.gz"

        # Recover from crash
        self._recover()

    def _recover(self):
        """Recover state from disk after restart."""
        # Load snapshot
        if self.snapshot_file.exists():
            try:
                with gzip.open(self.snapshot_file, 'rb') as f:
                    snapshot = pickle.load(f)
                self.buffer = snapshot.get('data', [])
                self.record_count = snapshot.get('count', 0)
                print(f"Recovered {len(self.buffer)} records from snapshot")
            except Exception as e:
                print(f"Snapshot recovery failed: {e}")

        # Replay log
        if self.log_file.exists():
            try:
                with open(self.log_file, 'r') as f:
                    for line in f:
                        record = json.loads(line.strip())
                        self.buffer.append(record)
                        self.record_count += 1
                print(f"Replayed {self.record_count} records from log")
            except Exception as e:
                print(f"Log replay failed: {e}")

    def append(self, record: Dict):
        """
        Append a record to the stream.

        Periodically flushes to disk for durability.
        """
        self.buffer.append(record)
        self.record_count += 1

        # Flush to log
        if self.record_count % self.flush_interval == 0:
            self._flush_log()

        # Create snapshot
        if self.record_count % self.snapshot_interval == 0:
            self._create_snapshot()

    def _flush_log(self):
        """Flush recent records to append log."""
        # Append only new records since last flush
        start_idx = max(0, len(self.buffer) - self.flush_interval)
        new_records = self.buffer[start_idx:]

        with open(self.log_file, 'a') as f:
            for record in new_records:
                f.write(json.dumps(record) + '\n')
            f.flush()
            os.fsync(f.fileno())

    def _create_snapshot(self):
        """Create atomic snapshot and truncate log."""
        snapshot = {
            'data': self.buffer,
            'count': self.record_count,
            'timestamp': datetime.utcnow().isoformat(),
        }

        # Atomic write snapshot
        data = gzip.compress(pickle.dumps(snapshot))
        _atomic_write(self.snapshot_file, data)

        # Truncate log (records are now in snapshot)
        open(self.log_file, 'w').close()

    def get_all(self) -> List[Dict]:
        """Get all buffered records."""
        return self.buffer.copy()

    def get_dataframe(self):
        """Get records as DataFrame."""
        import pandas as pd
        if not self.buffer:
            return pd.DataFrame()
        return pd.DataFrame(self.buffer)


def create_checkpointer(base_dir: str = "./checkpoints") -> AtomicCheckpointer:
    """Create a default checkpointer."""
    return AtomicCheckpointer(base_dir)
