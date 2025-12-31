"""
Kinetra Configuration
=====================

Centralized configuration for parallelization and system settings.
"""

import os
import multiprocessing as mp
from pathlib import Path


# =============================================================================
# PARALLELIZATION SETTINGS
# =============================================================================

# Default max workers - can be overridden via KINETRA_MAX_WORKERS env var
# Defaults to min(cpu_count, 32) for high-core systems like AMD 5950
_default_max_workers = min(mp.cpu_count(), 32)
MAX_WORKERS = int(os.environ.get("KINETRA_MAX_WORKERS", _default_max_workers))

# Network I/O workers (for downloads, API calls)
# Can be higher than CPU workers since network I/O is not CPU-bound
MAX_NETWORK_WORKERS = int(os.environ.get("KINETRA_MAX_NETWORK_WORKERS", MAX_WORKERS))


# =============================================================================
# GPU SETTINGS
# =============================================================================

# Auto-detect GPU availability (ROCm for AMD, CUDA for NVIDIA)
def detect_gpu():
    """Detect available GPU backend."""
    try:
        import torch
        if torch.cuda.is_available():
            # Check if this is ROCm (AMD) or CUDA (NVIDIA)
            if hasattr(torch.version, 'hip') and torch.version.hip is not None:
                return "rocm"
            return "cuda"
    except ImportError:
        pass
    return "cpu"

GPU_BACKEND = os.environ.get("KINETRA_GPU_BACKEND", detect_gpu())
USE_GPU = GPU_BACKEND in ("cuda", "rocm")


# =============================================================================
# DATA PATHS
# =============================================================================

# Base data directory - can be overridden via KINETRA_DATA_DIR
_project_root = Path(__file__).parent.parent
DATA_DIR = Path(os.environ.get("KINETRA_DATA_DIR", _project_root / "data"))
