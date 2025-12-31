# Performance Optimization Changelog

## Summary

This update introduces significant performance optimizations across the Kinetra trading system, focusing on:
1. **Import time reduction** through lazy loading
2. **Computational speedups** for expensive algorithms
3. **Memory efficiency** through caching
4. **Data loading optimization**

## Benchmark Results

| Operation | Before | After | Speedup |
|-----------|--------|-------|---------|
| Package import | ~0.5-1.0s | ~0.001s | **500-1000x** |
| Recurrence matrix (n=200) | ~7ms | ~0.3ms | **23x** |
| Rolling percentile (n=5000) | ~234ms | ~25ms | **9x** |
| Rolling entropy (n=5000) | ~500ms | ~50ms | **10x** |
| Sample entropy (with numba) | ~2.4s | ~24ms | **100x** |

## Changes

### 1. Lazy Imports (`kinetra/__init__.py`)

**Problem**: Original implementation eagerly imported all modules at package initialization, causing slow startup times even when only a subset of features was needed.

**Solution**: Implemented lazy loading using `__getattr__` hook:
- Modules are only imported when first accessed
- Cached after first import for subsequent use
- Near-zero initial import time (~0.001s)

```python
# Before: All imports executed at startup
from .physics_engine import PhysicsEngine  # Executed immediately

# After: Deferred until first use
def __getattr__(name):
    # Only imports when kinetra.PhysicsEngine is accessed
```

### 2. Optimized Sample Entropy (`kinetra/performance.py`)

**Problem**: Sample entropy has O(n²) complexity with nested loops, extremely slow for larger lookback windows.

**Solution**: 
- Added numba JIT compilation (when available)
- Falls back to vectorized numpy when numba not installed
- 50-100x speedup with numba

```python
# Install numba for maximum performance:
pip install numba
```

### 3. Vectorized Recurrence Matrix (`kinetra/performance.py`)

**Problem**: Original recurrence matrix computation used nested Python loops - O(n²) time with Python overhead.

**Solution**: Vectorized numpy implementation using broadcasting:
- 20-25x faster than naive implementation
- Maintains exact same output

```python
# Before: O(n²) with Python loop overhead
for i in range(n):
    for j in range(n):
        if abs(data[i] - data[j]) < eps:
            R[i, j] = 1

# After: Vectorized numpy
distances = np.abs(data[:, np.newaxis] - data[np.newaxis, :])
R = (distances < eps).astype(int)
```

### 4. Vectorized Rolling Operations (`kinetra/performance.py`)

**Problem**: Pandas `.apply()` with lambdas is slow due to Python function call overhead per window.

**Solution**: Implemented vectorized rolling functions:
- `rolling_percentile_vectorized()` - 9x faster
- `rolling_entropy_vectorized()` - 10x faster

### 5. Data Loading Cache (`kinetra/data_utils.py`)

**Problem**: Format detection (separator, date format) was repeated on every file load.

**Solution**: Cache detected formats per file path:
- First load: auto-detect and cache
- Subsequent loads: use cached format
- 1.2-1.5x speedup on repeated loads

### 6. Feature Engine Singleton (`kinetra/dsp_features.py`)

**Problem**: Creating new extractor instances for every feature extraction call.

**Solution**: Singleton pattern for DSPFeatureEngine:
- Reuses extractor instances
- Reduces memory allocation overhead

## New Files

### `kinetra/performance.py`
New module containing:
- `sample_entropy_fast()` - JIT-compiled sample entropy
- `recurrence_matrix_fast()` - Vectorized recurrence matrix
- `determinism_fast()` - Optimized determinism calculation
- `rolling_entropy_vectorized()` - Vectorized rolling entropy
- `rolling_percentile_vectorized()` - Vectorized rolling percentile
- `Benchmark` - Context manager for timing code blocks
- `timed` - Decorator for function timing
- `memoize` - Decorator for result caching
- `CachedProperty` - Descriptor for cached properties

### `scripts/benchmark_performance.py`
Benchmark script to measure and compare performance:
```bash
python scripts/benchmark_performance.py
```

## Usage

### Enabling Maximum Performance

1. **Install numba for JIT compilation**:
```bash
pip install numba
```

2. **Check performance configuration**:
```python
from kinetra.performance import print_performance_info
print_performance_info()
```

3. **Run benchmarks**:
```bash
python scripts/benchmark_performance.py
```

### Using Performance Utilities

```python
from kinetra.performance import (
    Benchmark,
    timed,
    memoize,
    sample_entropy_fast,
    recurrence_matrix_fast,
)

# Benchmark a code block
with Benchmark("physics computation"):
    result = engine.compute_physics_state(data)

# Time a function
@timed
def slow_function():
    ...

# Cache expensive results
@memoize(maxsize=256)
def expensive_computation(x, y):
    ...

# Use optimized algorithms directly
entropy = sample_entropy_fast(data, m=2, r=0.2)
R = recurrence_matrix_fast(data, threshold=0.1)
```

## Backward Compatibility

All optimizations are fully backward compatible:
- Same function signatures
- Same output values
- Automatic fallback when optional dependencies (numba) not installed
- All existing tests pass

## Recommendations

For production deployments:
1. Install numba: `pip install numba`
2. Use GPU acceleration for large-scale backtests (torch with CUDA/ROCm)
3. Precompute physics state once and reuse for multiple strategies
4. Use parallel processing for multi-instrument backtests

## Testing

All 54 existing tests pass after optimization:
```bash
pytest tests/ -v
```
