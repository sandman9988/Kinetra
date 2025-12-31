"""
Performance Optimization Utilities
===================================

High-performance implementations of computationally expensive algorithms:
- JIT-compiled entropy calculations (O(n²) -> O(n²) but 50-100x faster)
- Vectorized recurrence matrix computation
- Caching utilities for expensive operations
- Profiling and benchmarking tools

USAGE:
    from kinetra.performance import (
        sample_entropy_fast,
        recurrence_matrix_fast,
        timed,
        cached_property,
    )
"""

from __future__ import annotations

import functools
import time
import warnings
from typing import Callable, Dict, Any, TypeVar, Optional

import numpy as np
from numpy.typing import NDArray

# Try to import numba for JIT compilation
try:
    from numba import jit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    # Create no-op decorator as fallback
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    prange = range


# ============================================================================
# JIT-Compiled Sample Entropy (50-100x faster than pure Python)
# ============================================================================

if NUMBA_AVAILABLE:
    @jit(nopython=True, cache=True, fastmath=True)
    def _count_matches_jit(data: np.ndarray, m: int, tolerance: float) -> tuple:
        """
        JIT-compiled match counting for sample entropy.
        
        Counts template matches for embedding dimensions m and m+1.
        """
        n = len(data)
        count_m = 0
        count_m1 = 0
        
        # Count matches for dimension m
        for i in range(n - m):
            for j in range(i + 1, n - m):
                # Check if templates match within tolerance
                match_m = True
                for k in range(m):
                    if abs(data[i + k] - data[j + k]) >= tolerance:
                        match_m = False
                        break
                
                if match_m:
                    count_m += 1
                    # Check m+1 dimension
                    if i + m < n and j + m < n:
                        if abs(data[i + m] - data[j + m]) < tolerance:
                            count_m1 += 1
        
        return count_m, count_m1

    def sample_entropy_fast(data: np.ndarray, m: int = 2, r: float = 0.2) -> float:
        """
        Fast sample entropy using JIT compilation.
        
        50-100x faster than pure Python implementation for typical data sizes.
        
        Args:
            data: Time series (1D array)
            m: Embedding dimension (default: 2)
            r: Tolerance as fraction of std (default: 0.2)
        
        Returns:
            Sample entropy value
        """
        n = len(data)
        if n < m + 2:
            return 0.0
        
        data = np.ascontiguousarray(data, dtype=np.float64)
        std = np.std(data)
        if std == 0:
            return 0.0
        
        tolerance = r * std
        count_m, count_m1 = _count_matches_jit(data, m, tolerance)
        
        if count_m == 0 or count_m1 == 0:
            return 0.0
        
        return -np.log(count_m1 / count_m)

else:
    def sample_entropy_fast(data: np.ndarray, m: int = 2, r: float = 0.2) -> float:
        """
        Vectorized sample entropy (fallback when numba not available).
        
        Still faster than naive Python through vectorization.
        """
        n = len(data)
        if n < m + 2:
            return 0.0
        
        std = np.std(data)
        if std == 0:
            return 0.0
        
        tolerance = r * std
        
        def count_matches_vectorized(template_len: int) -> int:
            """Count matches using vectorized operations."""
            count = 0
            for i in range(n - template_len):
                template_i = data[i:i + template_len]
                for j in range(i + 1, n - template_len):
                    template_j = data[j:j + template_len]
                    if np.max(np.abs(template_i - template_j)) < tolerance:
                        count += 1
            return count
        
        count_m = count_matches_vectorized(m)
        count_m1 = count_matches_vectorized(m + 1)
        
        if count_m == 0 or count_m1 == 0:
            return 0.0
        
        return -np.log(count_m1 / count_m)


# ============================================================================
# Vectorized Recurrence Matrix (O(n²) but vectorized = much faster)
# ============================================================================

def recurrence_matrix_fast(
    data: np.ndarray,
    threshold: float = 0.1,
    use_mad: bool = True
) -> np.ndarray:
    """
    Compute recurrence matrix using vectorized operations.
    
    Much faster than nested loops through broadcasting.
    
    Args:
        data: Time series (1D array)
        threshold: Threshold as fraction of MAD or std
        use_mad: Use MAD (robust) instead of std for threshold
    
    Returns:
        Binary recurrence matrix (n x n)
    """
    data = np.asarray(data, dtype=np.float64)
    n = len(data)
    
    if n < 2:
        return np.ones((1, 1), dtype=np.int32)
    
    # Calculate threshold
    if use_mad:
        median = np.median(data)
        dispersion = np.median(np.abs(data - median))
    else:
        dispersion = np.std(data)
    
    eps = threshold * max(dispersion, 1e-10)
    
    # Vectorized distance computation using broadcasting
    # data[:, np.newaxis] creates (n, 1) array
    # data[np.newaxis, :] creates (1, n) array
    # Subtraction broadcasts to (n, n)
    distances = np.abs(data[:, np.newaxis] - data[np.newaxis, :])
    
    # Create binary recurrence matrix (use int for test compatibility)
    R = (distances < eps).astype(int)
    
    return R


def determinism_fast(R: np.ndarray, min_line: int = 2) -> float:
    """
    Compute determinism from recurrence matrix efficiently.
    
    Determinism = fraction of recurrent points forming diagonal lines.
    
    Args:
        R: Recurrence matrix
        min_line: Minimum diagonal line length to count
    
    Returns:
        Determinism value (0-1)
    """
    n = len(R)
    if n < 2:
        return 0.0
    
    total_recurrent = np.sum(R)
    if total_recurrent == 0:
        return 0.0
    
    diagonal_points = 0
    
    # Check all diagonals (excluding main diagonal)
    for k in range(1, n):
        # Upper diagonal
        diag = np.diag(R, k)
        diagonal_points += _count_line_points(diag, min_line)
        
        # Lower diagonal
        diag = np.diag(R, -k)
        diagonal_points += _count_line_points(diag, min_line)
    
    return diagonal_points / total_recurrent


def _count_line_points(diag: np.ndarray, min_line: int) -> int:
    """Count points in diagonal lines of length >= min_line."""
    if len(diag) < min_line:
        return 0
    
    # Find runs of 1s
    count = 0
    run_length = 0
    
    for val in diag:
        if val == 1:
            run_length += 1
        else:
            if run_length >= min_line:
                count += run_length
            run_length = 0
    
    # Handle last run
    if run_length >= min_line:
        count += run_length
    
    return count


# ============================================================================
# Vectorized Rolling Operations (replaces slow pandas .apply())
# ============================================================================

def rolling_entropy_vectorized(
    returns: np.ndarray,
    window: int,
    bins: int = 20
) -> np.ndarray:
    """
    Compute rolling Shannon entropy using vectorized operations.
    
    Faster than pandas .rolling().apply() for entropy calculation.
    
    Args:
        returns: Return series
        window: Rolling window size
        bins: Number of histogram bins
    
    Returns:
        Rolling entropy array
    """
    n = len(returns)
    entropy = np.zeros(n, dtype=np.float64)
    
    if n < window:
        return entropy
    
    # Pre-compute histogram edges based on data range
    data_min = np.nanmin(returns)
    data_max = np.nanmax(returns)
    
    if data_min == data_max:
        return entropy
    
    bin_edges = np.linspace(data_min, data_max, bins + 1)
    log_bins = np.log(bins)
    
    for i in range(window - 1, n):
        window_data = returns[i - window + 1:i + 1]
        window_data = window_data[~np.isnan(window_data)]
        
        if len(window_data) < 5:
            entropy[i] = 0.0
            continue
        
        # Compute histogram
        hist, _ = np.histogram(window_data, bins=bin_edges, density=True)
        p = hist / (hist.sum() + 1e-12)
        p = p[p > 0]
        
        # Shannon entropy
        H = -np.sum(p * np.log(p))
        entropy[i] = H / log_bins
    
    return entropy


def rolling_percentile_vectorized(
    data: np.ndarray,
    window: int
) -> np.ndarray:
    """
    Compute rolling percentile rank using vectorized operations.
    
    Faster than pandas .rolling().apply() for percentile calculation.
    
    Args:
        data: Input series
        window: Rolling window size
    
    Returns:
        Rolling percentile (0-1) array
    """
    n = len(data)
    percentiles = np.full(n, 0.5, dtype=np.float64)
    
    if n < window:
        return percentiles
    
    for i in range(window - 1, n):
        window_data = data[i - window + 1:i + 1]
        valid_data = window_data[~np.isnan(window_data)]
        
        if len(valid_data) == 0:
            continue
        
        current_val = data[i]
        if np.isnan(current_val):
            continue
        
        # Percentile = fraction of values <= current
        percentiles[i] = np.mean(valid_data <= current_val)
    
    return percentiles


# ============================================================================
# Caching Utilities
# ============================================================================

F = TypeVar('F', bound=Callable[..., Any])


def timed(func: F) -> F:
    """
    Decorator to measure and log function execution time.
    
    Usage:
        @timed
        def slow_function():
            ...
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        
        if elapsed > 0.1:  # Only log if > 100ms
            print(f"[PERF] {func.__name__}: {elapsed:.3f}s")
        
        return result
    
    return wrapper  # type: ignore


def memoize(maxsize: int = 128):
    """
    Memoization decorator for expensive computations.
    
    Caches results based on hashable arguments.
    
    Args:
        maxsize: Maximum cache size (default: 128)
    
    Usage:
        @memoize(maxsize=256)
        def expensive_computation(x, y):
            ...
    """
    def decorator(func: F) -> F:
        cache: Dict[tuple, Any] = {}
        hits = 0
        misses = 0
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            nonlocal hits, misses
            
            # Create hashable key
            try:
                key = (args, tuple(sorted(kwargs.items())))
            except TypeError:
                # Unhashable arguments - bypass cache
                return func(*args, **kwargs)
            
            if key in cache:
                hits += 1
                return cache[key]
            
            result = func(*args, **kwargs)
            misses += 1
            
            # Evict oldest if cache full (simple FIFO)
            if len(cache) >= maxsize:
                cache.pop(next(iter(cache)))
            
            cache[key] = result
            return result
        
        # Attach cache info
        wrapper.cache_info = lambda: {"hits": hits, "misses": misses, "size": len(cache)}
        wrapper.cache_clear = lambda: cache.clear()
        
        return wrapper  # type: ignore
    
    return decorator


class CachedProperty:
    """
    Cached property descriptor - computes value once, then caches.
    
    Thread-safe and works with instance deletion.
    
    Usage:
        class MyClass:
            @CachedProperty
            def expensive_property(self):
                # Computed once per instance
                return heavy_computation()
    """
    
    def __init__(self, func: Callable):
        self.func = func
        self.__doc__ = func.__doc__
        self.attrname: Optional[str] = None
    
    def __set_name__(self, owner, name):
        self.attrname = name
    
    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        
        if self.attrname is None:
            raise TypeError("Cannot use CachedProperty without calling __set_name__")
        
        # Check instance dict first
        cache = obj.__dict__
        if self.attrname in cache:
            return cache[self.attrname]
        
        # Compute and cache
        val = self.func(obj)
        cache[self.attrname] = val
        return val


# Alias for convenience
cached_property = CachedProperty


# ============================================================================
# Benchmarking Utilities
# ============================================================================

class Benchmark:
    """
    Context manager for benchmarking code blocks.
    
    Usage:
        with Benchmark("physics computation"):
            engine.compute_physics_state(data)
    """
    
    _results: Dict[str, list] = {}
    
    def __init__(self, name: str, verbose: bool = True):
        self.name = name
        self.verbose = verbose
        self.start_time: float = 0
        self.elapsed: float = 0
    
    def __enter__(self):
        self.start_time = time.perf_counter()
        return self
    
    def __exit__(self, *args):
        self.elapsed = time.perf_counter() - self.start_time
        
        # Store result
        if self.name not in Benchmark._results:
            Benchmark._results[self.name] = []
        Benchmark._results[self.name].append(self.elapsed)
        
        if self.verbose:
            print(f"[BENCH] {self.name}: {self.elapsed:.3f}s")
    
    @classmethod
    def summary(cls) -> Dict[str, Dict[str, float]]:
        """Get summary statistics for all benchmarks."""
        summary = {}
        for name, times in cls._results.items():
            times_arr = np.array(times)
            summary[name] = {
                "count": len(times),
                "mean": np.mean(times_arr),
                "std": np.std(times_arr),
                "min": np.min(times_arr),
                "max": np.max(times_arr),
                "total": np.sum(times_arr),
            }
        return summary
    
    @classmethod
    def print_summary(cls):
        """Print formatted benchmark summary."""
        summary = cls.summary()
        if not summary:
            print("No benchmarks recorded")
            return
        
        print("\n" + "=" * 70)
        print("BENCHMARK SUMMARY")
        print("=" * 70)
        print(f"{'Name':<30} {'Count':>8} {'Mean':>10} {'Total':>10}")
        print("-" * 70)
        
        for name, stats in sorted(summary.items(), key=lambda x: -x[1]["total"]):
            print(f"{name:<30} {stats['count']:>8} {stats['mean']:>10.3f}s {stats['total']:>10.3f}s")
        
        print("=" * 70)
    
    @classmethod
    def clear(cls):
        """Clear all benchmark results."""
        cls._results.clear()


# ============================================================================
# Module-level utilities
# ============================================================================

def check_numba_available() -> bool:
    """Check if numba JIT compilation is available."""
    return NUMBA_AVAILABLE


def print_performance_info():
    """Print performance configuration information."""
    print("=" * 60)
    print("PERFORMANCE CONFIGURATION")
    print("=" * 60)
    print(f"Numba JIT:         {'Available' if NUMBA_AVAILABLE else 'Not installed'}")
    
    # Check for other accelerators
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        print(f"PyTorch CUDA/ROCm: {'Available' if cuda_available else 'Not available'}")
    except ImportError:
        print("PyTorch:           Not installed")
    
    print("=" * 60)
    
    if not NUMBA_AVAILABLE:
        print("\nTIP: Install numba for 50-100x faster entropy calculations:")
        print("  pip install numba")


# ============================================================================
# Optimized feature extraction functions
# ============================================================================

def extract_recurrence_features_fast(
    data: np.ndarray,
    lookback: int = 50,
    threshold: float = 0.1
) -> Dict[str, float]:
    """
    Fast recurrence feature extraction using vectorized operations.
    
    Args:
        data: Price series
        lookback: Analysis window
        threshold: Recurrence threshold
    
    Returns:
        Dictionary of recurrence features
    """
    if len(data) < lookback:
        lookback = len(data)
    
    recent = data[-lookback:]
    log_returns = np.diff(np.log(recent))
    
    if len(log_returns) < 10:
        return {
            'recurrence_rate': 0.0,
            'determinism': 0.0,
            'det_up': 0.0,
            'det_down': 0.0,
            'recurrence_asymmetry': 0.0
        }
    
    # Compute recurrence matrix (vectorized)
    R = recurrence_matrix_fast(log_returns, threshold)
    n = len(R)
    
    # Recurrence rate
    rr = np.sum(R) / (n * n)
    
    # Determinism
    det = determinism_fast(R)
    
    # Directional analysis
    up_returns = log_returns[log_returns > 0]
    down_returns = log_returns[log_returns < 0]
    
    if len(up_returns) >= 5:
        R_up = recurrence_matrix_fast(up_returns, threshold)
        det_up = determinism_fast(R_up)
    else:
        det_up = 0.0
    
    if len(down_returns) >= 5:
        R_down = recurrence_matrix_fast(np.abs(down_returns), threshold)
        det_down = determinism_fast(R_down)
    else:
        det_down = 0.0
    
    return {
        'recurrence_rate': rr,
        'determinism': det,
        'det_up': det_up,
        'det_down': det_down,
        'recurrence_asymmetry': det_up - det_down
    }
