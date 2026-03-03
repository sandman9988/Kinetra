"""
Kinetra Configuration
=====================

Centralized configuration for parallelization and system settings.

Worker / concurrency philosophy
--------------------------------
Three completely separate buckets — never mix them:

  CPU / Disk workers  (ProcessPoolExecutor, ThreadPoolExecutor for CSV reads)
  ──────────────────
  Default = 50 % of logical threads.  On an AMD 5950X (16 cores / 32 threads)
  that is 16 workers — exactly the physical core count, giving full throughput
  without SMT contention or OS starvation.

  Override globally : KINETRA_CPU_FRACTION=0.75   (fraction, 0.05–1.0)
  Override absolute : KINETRA_MAX_WORKERS=20       (takes priority)

  Network / broker concurrency  (asyncio semaphores for MetaAPI streams)
  ─────────────────────────────
  This is a RATE LIMIT, not a CPU limit.  The MetaAPI broker back-end
  serialises or throttles more than ~3–5 simultaneous historical-candle
  requests from the same account regardless of how many CPU cores you have.
  Empirically tested: 4+ streams caused multi-minute delays on
  VantageInternational-Demo.

  Default  : 3   (safe for all known MetaAPI brokers)
  Override : KINETRA_DL_CONCURRENCY=4   (use with caution; test your broker)
  Hard cap : 5   (absolute ceiling; raise only if your broker explicitly
                  supports higher throughput on historical data endpoints)
"""

import multiprocessing as mp
import os
from pathlib import Path
from typing import Literal

# =============================================================================
# CPU / DISK PARALLELIZATION
# =============================================================================

# Fraction of logical threads to use for CPU-bound and disk-bound work.
# 0.5 == physical core count on SMT systems (AMD 5950X: 0.5 × 32 = 16).
# Override via KINETRA_CPU_FRACTION (e.g. 0.75 for overnight batch jobs).
_cpu_fraction: float = float(os.environ.get("KINETRA_CPU_FRACTION", "0.5"))
_cpu_fraction = max(0.05, min(1.0, _cpu_fraction))  # clamp to [0.05, 1.0]

_logical_threads: int = mp.cpu_count() or 4
_default_cpu_workers: int = max(1, int(_logical_threads * _cpu_fraction))

# Absolute override takes priority over the fraction.
MAX_WORKERS: int = int(os.environ.get("KINETRA_MAX_WORKERS", _default_cpu_workers))

# =============================================================================
# NETWORK / BROKER CONCURRENCY  — rate-limit controlled, NOT CPU-derived
# =============================================================================

# Safe default for MetaAPI historical-candle streams (empirically validated).
_METAAPI_RATE_SAFE: int = 3
# Hard ceiling — even permissive brokers rarely support more than 5 concurrent
# historical streams without queuing / rate-limit errors.
_METAAPI_RATE_MAX: int = 5

MAX_NETWORK_WORKERS: int = min(
    _METAAPI_RATE_MAX,
    int(
        os.environ.get(
            "KINETRA_DL_CONCURRENCY",
            os.environ.get("KINETRA_MAX_NETWORK_WORKERS", _METAAPI_RATE_SAFE),
        )
    ),
)


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
            if hasattr(torch.version, "hip") and torch.version.hip is not None:
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

_project_root = Path(__file__).parent.parent

# Public export — import PROJECT_ROOT from kinetra.config instead of
# recomputing Path(__file__).parent.parent in every script (DRY-14).
PROJECT_ROOT: Path = _project_root

DATA_DIR = Path(os.environ.get("KINETRA_DATA_DIR", _project_root / "data"))


def resolve_project_path(
    path: str | Path,
    *,
    recursive: bool = False,
    kind: Literal["any", "file", "dir"] = "any",
) -> Path:
    """Resolve *path* relative to ``PROJECT_ROOT`` unless already absolute.

    Parameters
    ----------
    path:
        Absolute path, or a project-relative path.
    recursive:
        If ``True`` and direct ``PROJECT_ROOT / path`` does not exist, search
        recursively under ``PROJECT_ROOT`` for a unique match.
    kind:
        Optional type constraint when recursive search is enabled.
    """
    p = Path(path)
    if p.is_absolute():
        return p

    direct = PROJECT_ROOT / p
    if direct.exists() or not recursive:
        return direct

    matches = [m for m in PROJECT_ROOT.rglob(str(p))]
    if kind == "file":
        matches = [m for m in matches if m.is_file()]
    elif kind == "dir":
        matches = [m for m in matches if m.is_dir()]

    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise ValueError(
            f"Ambiguous recursive project path '{path}': "
            f"{', '.join(str(m.relative_to(PROJECT_ROOT)) for m in matches[:8])}"
        )
    return direct


def resolve_project_folder(name: str, *, recursive: bool = True) -> Path:
    """Resolve a folder by name from project root.

    If *name* is a direct top-level folder, it is returned. Otherwise,
    recursive search is used (default) and must resolve to a unique directory.
    """
    return resolve_project_path(name, recursive=recursive, kind="dir")
