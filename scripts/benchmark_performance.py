#!/usr/bin/env python3
"""
Performance Benchmark Script
============================

Measures performance improvements from optimizations:
1. Lazy imports (import time)
2. Sample entropy (JIT vs pure Python)
3. Recurrence matrix (vectorized vs loops)
4. Rolling operations (vectorized vs pandas .apply())
5. Data loading (cached vs uncached)

Usage:
    python scripts/benchmark_performance.py
"""

import sys
import time
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np


def print_header(text: str):
    """Print section header."""
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70)


def print_result(name: str, time_sec: float, baseline_sec: float = None):
    """Print benchmark result."""
    if baseline_sec:
        speedup = baseline_sec / time_sec if time_sec > 0 else float('inf')
        print(f"  {name:<40} {time_sec:>8.3f}s (speedup: {speedup:.1f}x)")
    else:
        print(f"  {name:<40} {time_sec:>8.3f}s")


def benchmark_import_time():
    """Benchmark module import time."""
    print_header("1. IMPORT TIME BENCHMARK")
    
    # Clear any cached imports
    modules_to_clear = [k for k in sys.modules.keys() if k.startswith('kinetra')]
    for mod in modules_to_clear:
        del sys.modules[mod]
    
    # Measure import time
    start = time.perf_counter()
    import kinetra
    import_time = time.perf_counter() - start
    
    print_result("kinetra package import", import_time)
    
    # Measure lazy component access
    start = time.perf_counter()
    _ = kinetra.PhysicsEngine
    physics_time = time.perf_counter() - start
    
    print_result("PhysicsEngine access (lazy)", physics_time)
    
    print(f"\n  TIP: Lazy imports defer loading until first use.")
    print(f"       Initial import is fast; components load on demand.")


def benchmark_sample_entropy():
    """Benchmark sample entropy computation."""
    print_header("2. SAMPLE ENTROPY BENCHMARK")
    
    # Generate test data
    np.random.seed(42)
    data_sizes = [100, 500, 1000]
    
    # Import both implementations
    try:
        from kinetra.performance import sample_entropy_fast, check_numba_available
        has_numba = check_numba_available()
    except ImportError:
        has_numba = False
        sample_entropy_fast = None
    
    from kinetra.dsp_features import EntropyExtractor
    
    print(f"\n  Numba JIT available: {has_numba}")
    
    for n in data_sizes:
        data = np.random.randn(n).cumsum()
        
        # Pure Python (baseline)
        def pure_python_entropy(data, m=2, r=0.2):
            """Pure Python sample entropy."""
            n = len(data)
            if n < m + 2:
                return 0.0
            std = np.std(data)
            if std == 0:
                return 0.0
            tolerance = r * std
            
            def count_matches(template_length):
                count = 0
                for i in range(n - template_length):
                    for j in range(i + 1, n - template_length):
                        if np.max(np.abs(data[i:i+template_length] - data[j:j+template_length])) < tolerance:
                            count += 1
                return count
            
            a = count_matches(m)
            b = count_matches(m + 1)
            if a == 0 or b == 0:
                return 0.0
            return -np.log(b / a)
        
        # Measure pure Python
        start = time.perf_counter()
        result_pure = pure_python_entropy(data)
        pure_time = time.perf_counter() - start
        
        # Measure optimized version (via EntropyExtractor)
        start = time.perf_counter()
        result_opt = EntropyExtractor.sample_entropy(data)
        opt_time = time.perf_counter() - start
        
        print(f"\n  Data size: {n}")
        print_result(f"  Pure Python", pure_time)
        print_result(f"  Optimized", opt_time, pure_time)


def benchmark_recurrence_matrix():
    """Benchmark recurrence matrix computation."""
    print_header("3. RECURRENCE MATRIX BENCHMARK")
    
    np.random.seed(42)
    data_sizes = [50, 100, 200]
    
    try:
        from kinetra.performance import recurrence_matrix_fast
        has_fast = True
    except ImportError:
        has_fast = False
    
    from kinetra.assumption_free_measures import RecurrenceFeatures
    
    for n in data_sizes:
        data = np.random.randn(n)
        
        # Naive nested loops (baseline)
        def naive_recurrence(data, threshold=0.1):
            n = len(data)
            mad = np.median(np.abs(data - np.median(data)))
            eps = threshold * max(mad, 1e-10)
            R = np.zeros((n, n), dtype=int)
            for i in range(n):
                for j in range(n):
                    if abs(data[i] - data[j]) < eps:
                        R[i, j] = 1
            return R
        
        # Measure naive
        start = time.perf_counter()
        R_naive = naive_recurrence(data)
        naive_time = time.perf_counter() - start
        
        # Measure optimized
        start = time.perf_counter()
        R_opt = RecurrenceFeatures.compute_recurrence_matrix(data)
        opt_time = time.perf_counter() - start
        
        # Verify correctness
        is_correct = np.array_equal(R_naive, R_opt)
        
        print(f"\n  Data size: {n}")
        print_result(f"  Naive (nested loops)", naive_time)
        print_result(f"  Optimized (vectorized)", opt_time, naive_time)
        print(f"  Results match: {is_correct}")


def benchmark_rolling_operations():
    """Benchmark rolling operations."""
    print_header("4. ROLLING OPERATIONS BENCHMARK")
    
    import pandas as pd
    np.random.seed(42)
    n = 5000
    window = 100
    
    data = pd.Series(np.random.randn(n).cumsum())
    
    # Pandas .apply() (baseline)
    def pandas_rolling_pct(x, window):
        def pct_last(w):
            val = w.iloc[-1]
            return (w <= val).mean()
        return x.rolling(window, min_periods=10).apply(pct_last, raw=False)
    
    # Measure pandas .apply()
    start = time.perf_counter()
    result_pandas = pandas_rolling_pct(data, window)
    pandas_time = time.perf_counter() - start
    
    # Measure optimized
    try:
        from kinetra.performance import rolling_percentile_vectorized
        start = time.perf_counter()
        result_opt = rolling_percentile_vectorized(data.values, window)
        opt_time = time.perf_counter() - start
        
        print(f"\n  Data size: {n}, window: {window}")
        print_result(f"  Pandas .apply()", pandas_time)
        print_result(f"  Vectorized", opt_time, pandas_time)
    except ImportError:
        print(f"\n  Data size: {n}, window: {window}")
        print_result(f"  Pandas .apply()", pandas_time)
        print(f"  Vectorized: Not available (install kinetra.performance)")


def benchmark_data_loading():
    """Benchmark data loading."""
    print_header("5. DATA LOADING BENCHMARK")
    
    # Find a sample data file
    data_dirs = [
        Path("data/master"),
        Path("data/prepared/train"),
        Path("data/runs/berserker_run1/data"),
        Path("data/runs/berserker_run3/data"),
    ]
    
    sample_file = None
    for dir_path in data_dirs:
        if dir_path.exists():
            files = list(dir_path.glob("*.csv"))
            if files:
                sample_file = files[0]
                break
    
    if sample_file is None:
        print("\n  No data files found for benchmarking.")
        print("  Place CSV files in data/master/ or data/prepared/train/")
        return
    
    print(f"\n  Sample file: {sample_file.name}")
    
    from kinetra.data_utils import load_mt5_csv, clear_format_cache
    
    # Clear cache first
    clear_format_cache()
    
    # First load (no cache)
    start = time.perf_counter()
    df1 = load_mt5_csv(str(sample_file))
    first_load = time.perf_counter() - start
    
    # Second load (with cache)
    start = time.perf_counter()
    df2 = load_mt5_csv(str(sample_file))
    second_load = time.perf_counter() - start
    
    print(f"  Bars loaded: {len(df1)}")
    print_result(f"  First load (no cache)", first_load)
    print_result(f"  Second load (cached)", second_load, first_load)


def benchmark_physics_engine():
    """Benchmark physics engine computation."""
    print_header("6. PHYSICS ENGINE BENCHMARK")
    
    import pandas as pd
    np.random.seed(42)
    
    # Generate synthetic OHLCV data
    n = 5000
    close = 100 + np.random.randn(n).cumsum() * 0.1
    high = close + np.abs(np.random.randn(n) * 0.5)
    low = close - np.abs(np.random.randn(n) * 0.5)
    open_ = close + np.random.randn(n) * 0.2
    volume = np.random.randint(1000, 10000, n)
    
    prices = pd.DataFrame({
        'open': open_,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume,
    }, index=pd.date_range('2020-01-01', periods=n, freq='H'))
    
    print(f"\n  Data size: {n} bars")
    
    from kinetra.physics_engine import PhysicsEngine
    
    engine = PhysicsEngine()
    
    # Measure physics state computation
    start = time.perf_counter()
    physics_state = engine.compute_physics_state_from_ohlcv(prices)
    physics_time = time.perf_counter() - start
    
    print_result(f"  Full physics state computation", physics_time)
    print(f"  Output columns: {len(physics_state.columns)}")
    print(f"  Output shape: {physics_state.shape}")


def main():
    """Run all benchmarks."""
    print("\n" + "=" * 70)
    print("  KINETRA PERFORMANCE BENCHMARK")
    print("=" * 70)
    print("\nMeasuring performance optimizations...")
    
    try:
        benchmark_import_time()
    except Exception as e:
        print(f"  Error: {e}")
    
    try:
        benchmark_sample_entropy()
    except Exception as e:
        print(f"  Error: {e}")
    
    try:
        benchmark_recurrence_matrix()
    except Exception as e:
        print(f"  Error: {e}")
    
    try:
        benchmark_rolling_operations()
    except Exception as e:
        print(f"  Error: {e}")
    
    try:
        benchmark_data_loading()
    except Exception as e:
        print(f"  Error: {e}")
    
    try:
        benchmark_physics_engine()
    except Exception as e:
        print(f"  Error: {e}")
    
    # Print optimization tips
    print_header("OPTIMIZATION TIPS")
    
    try:
        from kinetra.performance import check_numba_available, print_performance_info
        print_performance_info()
    except ImportError:
        print("  Import kinetra.performance to check optimization status")
    
    print("\nTo further improve performance:")
    print("  1. Install numba: pip install numba")
    print("  2. Use GPU acceleration: torch with CUDA/ROCm")
    print("  3. Use parallel processing for multi-instrument backtests")
    print("  4. Precompute physics state once, reuse for multiple strategies")


if __name__ == '__main__':
    main()
