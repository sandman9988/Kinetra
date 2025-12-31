"""
Kinetra DevOps Module
=====================

Comprehensive utilities for:
- Deduplication (code and data)
- Parallelization with auto-scaling
- GPU auto-detection and optimization
- Git sync utilities
- Security scanning
- Environment management
- Real-time monitoring
"""

from .dedup import CodeDeduplicator, DataDeduplicator, find_duplicates
from .env_manager import EnvManager, setup_environment, verify_environment
from .file_safety import (
    AtomicWriter,
    BackupManager,
    IntegrityChecker,
    SafeFileHandler,
    atomic_write,
    create_backup,
    safe_read,
    safe_write,
    verify_integrity,
)
from .git_sync import GitSync, auto_sync, check_sync_status
from .gpu import GPUManager, auto_detect_gpu, get_gpu_info
from .monitor import FolderMonitor, PerformanceTracker, start_monitoring
from .parallel import AutoScaler, ParallelExecutor, get_optimal_config
from .security import SecurityScanner, check_secrets, scan_codebase

__all__ = [
    # Deduplication
    "DataDeduplicator",
    "CodeDeduplicator",
    "find_duplicates",
    # Parallelization
    "AutoScaler",
    "ParallelExecutor",
    "get_optimal_config",
    # GPU
    "GPUManager",
    "auto_detect_gpu",
    "get_gpu_info",
    # Git Sync
    "GitSync",
    "auto_sync",
    "check_sync_status",
    # Security
    "SecurityScanner",
    "scan_codebase",
    "check_secrets",
    # Environment
    "EnvManager",
    "setup_environment",
    "verify_environment",
    # Monitoring
    "FolderMonitor",
    "start_monitoring",
    "PerformanceTracker",
    # File Safety
    "AtomicWriter",
    "BackupManager",
    "IntegrityChecker",
    "SafeFileHandler",
    "atomic_write",
    "safe_write",
    "safe_read",
    "create_backup",
    "verify_integrity",
]
