#!/usr/bin/env python3
"""
CPU Utilities - Adaptive Performance Optimization
==================================================

Auto-detect CPU capabilities and provide optimal worker/concurrency settings
for different workload types.

Features:
- CPU core/thread detection
- Adaptive worker count (CPU-bound tasks)
- Adaptive concurrency (I/O-bound tasks)
- CPU brand/model detection
- Memory-aware scaling

Usage:
    from kinetra.cpu_utils import get_optimal_workers, get_optimal_concurrency

    # For CPU-intensive tasks (data prep, feature extraction)
    workers = get_optimal_workers()  # e.g., 24 on AMD 5950X

    # For I/O-intensive tasks (downloads, API calls)
    concurrency = get_optimal_concurrency()  # e.g., 48 on AMD 5950X
"""

import os
import platform
import sys
from dataclasses import dataclass
from typing import Optional


@dataclass
class CPUInfo:
    """CPU information and capabilities."""

    logical_cores: int  # Total threads (including hyperthreading)
    physical_cores: Optional[int]  # Physical cores (if detectable)
    brand: str  # CPU brand/model
    platform: str  # OS platform
    has_smt: bool  # SMT/HyperThreading enabled


def get_cpu_info() -> CPUInfo:
    """
    Detect CPU information.

    Returns:
        CPUInfo object with CPU details
    """
    logical_cores = os.cpu_count() or 4
    physical_cores = None
    brand = "Unknown"
    has_smt = False

    try:
        # Try to get physical core count
        if sys.platform == "linux":
            try:
                with open("/proc/cpuinfo") as f:
                    cpuinfo = f.read()
                # Count unique physical IDs
                physical_ids = set()
                for line in cpuinfo.split("\n"):
                    if line.startswith("physical id"):
                        physical_ids.add(line.split(":")[1].strip())
                if physical_ids:
                    # Count cores per socket
                    cores_per_socket = 0
                    for line in cpuinfo.split("\n"):
                        if line.startswith("cpu cores"):
                            cores_per_socket = int(line.split(":")[1].strip())
                            break
                    if cores_per_socket:
                        physical_cores = len(physical_ids) * cores_per_socket

                # Get CPU brand
                for line in cpuinfo.split("\n"):
                    if line.startswith("model name"):
                        brand = line.split(":")[1].strip()
                        break
            except Exception:
                pass

        elif sys.platform == "darwin":  # macOS
            try:
                import subprocess

                result = subprocess.run(
                    ["sysctl", "-n", "hw.physicalcpu"],
                    capture_output=True,
                    text=True,
                    timeout=1,
                )
                if result.returncode == 0:
                    physical_cores = int(result.stdout.strip())

                result = subprocess.run(
                    ["sysctl", "-n", "machdep.cpu.brand_string"],
                    capture_output=True,
                    text=True,
                    timeout=1,
                )
                if result.returncode == 0:
                    brand = result.stdout.strip()
            except Exception:
                pass

        elif sys.platform == "win32":  # Windows
            try:
                import subprocess

                result = subprocess.run(
                    ["wmic", "cpu", "get", "NumberOfCores"],
                    capture_output=True,
                    text=True,
                    timeout=2,
                )
                if result.returncode == 0:
                    lines = result.stdout.strip().split("\n")
                    if len(lines) > 1:
                        physical_cores = int(lines[1].strip())

                result = subprocess.run(
                    ["wmic", "cpu", "get", "name"],
                    capture_output=True,
                    text=True,
                    timeout=2,
                )
                if result.returncode == 0:
                    lines = result.stdout.strip().split("\n")
                    if len(lines) > 1:
                        brand = lines[1].strip()
            except Exception:
                pass

    except Exception:
        pass

    # Detect SMT/HyperThreading
    if physical_cores and logical_cores > physical_cores:
        has_smt = True
    elif logical_cores >= 8:
        # Heuristic: assume SMT on modern CPUs with 8+ threads
        has_smt = True
        physical_cores = logical_cores // 2

    return CPUInfo(
        logical_cores=logical_cores,
        physical_cores=physical_cores,
        brand=brand,
        platform=platform.system(),
        has_smt=has_smt,
    )


def get_optimal_workers(
    workload_type: str = "balanced", min_workers: int = 2, max_workers: int = 32
) -> int:
    """
    Determine optimal worker count for CPU-intensive parallel processing.

    Strategy:
    - Use physical cores (not hyperthreads) for CPU-bound tasks
    - Reserve 1-2 cores for system/other processes
    - Scale based on workload intensity

    Args:
        workload_type: 'light', 'balanced', or 'heavy'
            - light: Leave more headroom (e.g., UI tasks)
            - balanced: Good for mixed workloads (default)
            - heavy: Max performance (e.g., batch processing)
        min_workers: Minimum workers to use
        max_workers: Maximum workers to use

    Returns:
        Optimal number of worker processes

    Examples:
        # AMD 5950X (16 cores / 32 threads)
        get_optimal_workers('light')    -> 12 (leave headroom)
        get_optimal_workers('balanced') -> 24 (75% of threads)
        get_optimal_workers('heavy')    -> 30 (nearly all threads)

        # Intel i7 (8 cores / 16 threads)
        get_optimal_workers('balanced') -> 12
    """
    try:
        cpu_info = get_cpu_info()
        logical_cores = cpu_info.logical_cores

        # Use physical cores if available, else estimate
        if cpu_info.physical_cores:
            physical_cores = cpu_info.physical_cores
        else:
            # Estimate: assume 2:1 SMT ratio on modern CPUs
            physical_cores = logical_cores // 2 if logical_cores >= 8 else logical_cores

        # Choose worker count based on workload type
        if workload_type == "light":
            # Use ~50% of logical cores (leave plenty of headroom)
            workers = int(logical_cores * 0.5)
        elif workload_type == "heavy":
            # Use ~95% of logical cores (max performance)
            workers = int(logical_cores * 0.95)
        else:  # balanced (default)
            # Use ~75% of logical cores (good balance)
            # This approximates physical core count on SMT systems
            workers = int(logical_cores * 0.75)

        # Apply min/max bounds
        workers = max(min_workers, min(workers, max_workers))

        return workers

    except Exception:
        # Fallback to conservative default
        return 8


def get_optimal_concurrency(
    workload_type: str = "network", min_concurrency: int = 8, max_concurrency: int = 64
) -> int:
    """
    Determine optimal concurrency for I/O-bound async operations.

    Strategy:
    - I/O-bound tasks benefit from higher concurrency than CPU count
    - Network I/O can handle 2-3x CPU count
    - Disk I/O should be more conservative
    - Scale based on CPU capabilities (more cores = can handle more I/O)

    Args:
        workload_type: 'disk', 'network', or 'mixed'
            - disk: Conservative (2x CPU count)
            - network: Aggressive (3x CPU count) - default
            - mixed: Balanced (2.5x CPU count)
        min_concurrency: Minimum concurrency
        max_concurrency: Maximum concurrency

    Returns:
        Optimal concurrency level

    Examples:
        # AMD 5950X (32 threads)
        get_optimal_concurrency('disk')    -> 48  (1.5x threads)
        get_optimal_concurrency('network') -> 64  (2x threads)
        get_optimal_concurrency('mixed')   -> 56  (1.75x threads)

        # Intel i7 (16 threads)
        get_optimal_concurrency('network') -> 32
    """
    try:
        cpu_info = get_cpu_info()
        logical_cores = cpu_info.logical_cores

        # Choose concurrency multiplier based on workload
        if workload_type == "disk":
            # Disk I/O: 1.5x logical cores
            multiplier = 1.5
        elif workload_type == "mixed":
            # Mixed I/O: 1.75x logical cores
            multiplier = 1.75
        else:  # network (default)
            # Network I/O: 2x logical cores
            multiplier = 2.0

        concurrency = int(logical_cores * multiplier)

        # Apply min/max bounds
        concurrency = max(min_concurrency, min(concurrency, max_concurrency))

        return concurrency

    except Exception:
        # Fallback to reasonable default
        return 24


def get_memory_gb() -> float:
    """
    Get total system memory in GB.

    Returns:
        Total RAM in gigabytes
    """
    try:
        if sys.platform == "linux":
            with open("/proc/meminfo") as f:
                meminfo = f.read()
            for line in meminfo.split("\n"):
                if line.startswith("MemTotal:"):
                    kb = int(line.split()[1])
                    return kb / (1024 * 1024)  # KB -> GB

        elif sys.platform == "darwin":  # macOS
            import subprocess

            result = subprocess.run(
                ["sysctl", "-n", "hw.memsize"], capture_output=True, text=True, timeout=1
            )
            if result.returncode == 0:
                bytes_mem = int(result.stdout.strip())
                return bytes_mem / (1024**3)  # Bytes -> GB

        elif sys.platform == "win32":  # Windows
            import subprocess

            result = subprocess.run(
                ["wmic", "computersystem", "get", "totalphysicalmemory"],
                capture_output=True,
                text=True,
                timeout=2,
            )
            if result.returncode == 0:
                lines = result.stdout.strip().split("\n")
                if len(lines) > 1:
                    bytes_mem = int(lines[1].strip())
                    return bytes_mem / (1024**3)  # Bytes -> GB

    except Exception:
        pass

    return 8.0  # Default assumption


def print_system_info():
    """Print detailed system information."""
    cpu_info = get_cpu_info()
    memory_gb = get_memory_gb()

    print("\n" + "=" * 70)
    print("  SYSTEM INFORMATION")
    print("=" * 70)
    print(f"\nCPU: {cpu_info.brand}")
    print(f"Platform: {cpu_info.platform}")
    print(f"Logical cores: {cpu_info.logical_cores}")
    if cpu_info.physical_cores:
        print(f"Physical cores: {cpu_info.physical_cores}")
    print(f"SMT/HyperThreading: {'Yes' if cpu_info.has_smt else 'No'}")
    print(f"Memory: {memory_gb:.1f} GB")

    print("\n" + "-" * 70)
    print("  RECOMMENDED SETTINGS")
    print("-" * 70)

    print("\nCPU-Intensive Tasks (Data Prep, Feature Extraction):")
    print(f"  Light workload:    {get_optimal_workers('light')} workers")
    print(f"  Balanced workload: {get_optimal_workers('balanced')} workers ⭐")
    print(f"  Heavy workload:    {get_optimal_workers('heavy')} workers")

    print("\nI/O-Intensive Tasks (Downloads, API Calls):")
    print(f"  Disk I/O:    {get_optimal_concurrency('disk')} concurrent")
    print(f"  Mixed I/O:   {get_optimal_concurrency('mixed')} concurrent")
    print(f"  Network I/O: {get_optimal_concurrency('network')} concurrent ⭐")

    print("\n" + "=" * 70 + "\n")


if __name__ == "__main__":
    # Demo/diagnostic mode
    print_system_info()

    # Example usage
    print("Example Usage:")
    print("-" * 70)
    print("from kinetra.cpu_utils import get_optimal_workers, get_optimal_concurrency")
    print()
    print("# For data prep (CPU-bound)")
    print(f"workers = get_optimal_workers()  # {get_optimal_workers()}")
    print()
    print("# For downloads (I/O-bound)")
    print(f"concurrency = get_optimal_concurrency()  # {get_optimal_concurrency()}")
    print()
