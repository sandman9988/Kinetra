# Vectorization Quick Reference Guide

**Rule:** Explicit Python loops are the last resort. Prefer NumPy vectorized ops, Pandas column operations, and broadcasting.

---

## 🚫 Anti-Patterns to Avoid

### 1. DataFrame.iterrows()
```python
# ❌ BAD - Extremely slow
for idx, row in df.iterrows():
    df.at[idx, 'new_col'] = row['a'] + row['b']

# ✅ GOOD - Vectorized
df['new_col'] = df['a'] + df['b']
```

### 2. range(len()) with .iloc[]
```python
# ❌ BAD - Slow indexing in loop
for i in range(len(df)):
    result.append(df.iloc[i]['price'] * df.iloc[i]['volume'])

# ✅ GOOD - Direct array operations
result = (df['price'] * df['volume']).tolist()
# Or even better - keep as NumPy array
result = (df['price'] * df['volume']).values
```

### 3. List append() in loops
```python
# ❌ BAD - Slow accumulation
results = []
for x in arr:
    results.append(x ** 2)

# ✅ GOOD - Vectorized
results = arr ** 2

# ✅ ALSO GOOD - List comprehension (when vectorization not possible)
results = [complex_function(x) for x in arr]
```

### 4. DataFrame append() in loops
```python
# ❌ BAD - Quadratic complexity
df_result = pd.DataFrame()
for item in items:
    df_result = df_result.append({'col': item}, ignore_index=True)

# ✅ GOOD - Collect then create
data = [{'col': item} for item in items]
df_result = pd.DataFrame(data)
```

### 5. Manual cumulative operations
```python
# ❌ BAD - Loop for cumulative sum
cumsum = []
total = 0
for val in values:
    total += val
    cumsum.append(total)

# ✅ GOOD - Use np.cumsum
cumsum = np.cumsum(values)

# ✅ GOOD - Cumulative product for price evolution
prices = initial_price * np.cumprod(1 + returns)
```

---

## ✅ Vectorization Patterns

### Pattern 1: Element-wise Operations
```python
# Use broadcasting for element-wise operations
df['spread_cost'] = df['price'] * df['spread_pct'] / 100
df['adjusted_price'] = np.where(df['direction'] == 1, 
                                 df['price'] + df['spread'],
                                 df['price'] - df['spread'])
```

### Pattern 2: Conditional Logic
```python
# Use np.where or np.select
# Simple condition
df['signal'] = np.where(df['momentum'] > 0, 1, -1)

# Multiple conditions
conditions = [
    df['momentum'] > threshold_high,
    df['momentum'] < threshold_low
]
choices = [1, -1]
df['signal'] = np.select(conditions, choices, default=0)
```

### Pattern 3: Aggregations
```python
# GroupBy operations
df.groupby('asset_class').agg({
    'return': 'mean',
    'volatility': 'std',
    'sharpe': lambda x: x.mean() / x.std()
})

# Rolling windows
df['sma_20'] = df['close'].rolling(20).mean()
df['volatility'] = df['returns'].rolling(20).std()
```

### Pattern 4: Array Slicing
```python
# Vectorized lookback windows
recent_prices = prices[-lookback:]
returns = prices[1:] / prices[:-1] - 1

# Compute all at once
up_moves = np.maximum(returns, 0)
down_moves = np.minimum(returns, 0)
```

### Pattern 5: Broadcasting
```python
# Broadcast operations across dimensions
# Shape: (n_samples, n_features) @ (n_features,) -> (n_samples,)
predictions = features @ weights

# Element-wise with different shapes
matrix = np.random.randn(100, 10)
column_means = matrix.mean(axis=0)  # Shape: (10,)
centered = matrix - column_means    # Broadcasting!
```

### Pattern 6: Index-based Joins
```python
# Instead of looping to find matching indices
# ❌ BAD
for idx, signal in signals.iterrows():
    matching_data = data[data.index == signal['time']]

# ✅ GOOD - Merge once
merged = signals.merge(data, left_on='time', right_index=True, how='left')
```

### Pattern 7: Apply Functions
```python
# When vectorization truly isn't possible, use .apply()
# Still faster than iterrows()

# Row-wise
df['result'] = df.apply(lambda row: complex_function(row['a'], row['b']), axis=1)

# Column-wise (even better)
df['result'] = df['column'].apply(single_arg_function)

# Multiple columns with unpacking
df['result'] = df[['a', 'b', 'c']].apply(lambda x: func(*x), axis=1)
```

### Pattern 8: Batch Operations
```python
# Process all items at once

# ❌ BAD
entry_features = []
for i in entry_indices:
    entry_features.append(features.iloc[i])

# ✅ GOOD
entry_features = features.iloc[entry_indices]

# With prefix/suffix
entry_features = features.iloc[entry_indices].add_prefix('entry_')
```

---

## 🎯 Common Transformations

### Financial Calculations
```python
# Returns
simple_returns = prices.pct_change()
log_returns = np.log(prices / prices.shift(1))

# Cumulative returns
cum_returns = (1 + returns).cumprod() - 1

# Drawdown
cummax = equity.cummax()
drawdown = (equity - cummax) / cummax

# Sharpe ratio (vectorized)
sharpe = returns.mean() / returns.std() * np.sqrt(252)
```

### Technical Indicators
```python
# Moving averages
sma = prices.rolling(window).mean()
ema = prices.ewm(span=window).mean()

# Bollinger bands
middle = prices.rolling(20).mean()
std = prices.rolling(20).std()
upper = middle + 2 * std
lower = middle - 2 * std

# RSI (vectorized)
delta = prices.diff()
gain = np.where(delta > 0, delta, 0)
loss = np.where(delta < 0, -delta, 0)
avg_gain = pd.Series(gain).rolling(14).mean()
avg_loss = pd.Series(loss).rolling(14).mean()
rs = avg_gain / avg_loss
rsi = 100 - (100 / (1 + rs))
```

### Signal Processing
```python
# Streak detection (vectorized)
returns_sign = np.sign(returns)
sign_changes = np.diff(returns_sign != 0)
streak_ids = np.cumsum(sign_changes)

# Pattern matching
patterns = np.lib.stride_tricks.sliding_window_view(data, window_size)
# Now process all windows at once
```

---

## 🔧 Performance Optimization Techniques

### Pre-allocation
```python
# When loops are unavoidable, pre-allocate
result = np.empty(n, dtype=float)
for i in range(n):
    result[i] = expensive_function(data[i])

# Better than: result.append(expensive_function(data[i]))
```

### Numba JIT
```python
from numba import jit

@jit(nopython=True)
def complex_loop(arr):
    """For truly unavoidable loops."""
    result = np.empty(len(arr))
    for i in range(len(arr)):
        # Complex logic that can't be vectorized
        result[i] = some_computation(arr[i])
    return result
```

### Cython (for extreme cases)
```python
# Only when Numba isn't enough
# cython: boundscheck=False, wraparound=False
import numpy as np
cimport numpy as np

def fast_function(np.ndarray[np.float64_t, ndim=1] arr):
    cdef int i
    cdef int n = arr.shape[0]
    cdef np.ndarray[np.float64_t, ndim=1] result = np.empty(n)
    
    for i in range(n):
        result[i] = arr[i] * arr[i]
    
    return result
```

---

## 📊 Benchmarking

Always benchmark your optimizations:

```python
import timeit

# Original
def original():
    result = []
    for x in data:
        result.append(x ** 2)
    return result

# Vectorized
def vectorized():
    return data ** 2

# Benchmark
n_runs = 1000
time_original = timeit.timeit(original, number=n_runs)
time_vectorized = timeit.timeit(vectorized, number=n_runs)

print(f"Speedup: {time_original / time_vectorized:.1f}x")
```

---

## 🎓 Best Practices

1. **Think in columns, not rows** - DataFrames are column-oriented
2. **Extract to NumPy when possible** - `df['col'].values` is faster than Series ops
3. **Use .loc/.iloc sparingly** - Direct column access is faster
4. **Avoid chaining operations that copy** - Each operation creates new objects
5. **Use inplace=True carefully** - Not always faster, test it
6. **Profile before optimizing** - Use `cProfile` or `line_profiler`
7. **Readable > Clever** - Don't sacrifice clarity for micro-optimizations

---

## 🧪 Testing Vectorized Code

```python
import numpy as np
import pandas as pd

def test_vectorization_correctness():
    """Ensure vectorized version matches loop version."""
    test_data = np.random.randn(100)
    
    # Original loop version
    loop_result = []
    for x in test_data:
        loop_result.append(x ** 2 + 2 * x + 1)
    loop_result = np.array(loop_result)
    
    # Vectorized version
    vec_result = test_data ** 2 + 2 * test_data + 1
    
    # Should be identical (within floating point precision)
    np.testing.assert_allclose(loop_result, vec_result)
    print("✅ Vectorization is correct!")

def test_edge_cases():
    """Test edge cases."""
    # Empty array
    assert len(vectorized_func(np.array([]))) == 0
    
    # Single element
    single = vectorized_func(np.array([1.0]))
    assert len(single) == 1
    
    # With NaN
    with_nan = vectorized_func(np.array([1.0, np.nan, 3.0]))
    assert len(with_nan) == 3
```

---

## 📚 Resources

- [NumPy Documentation](https://numpy.org/doc/stable/)
- [Pandas Performance Tips](https://pandas.pydata.org/docs/user_guide/enhancingperf.html)
- [Effective Pandas Patterns](https://effectivepandas.com/)
- [NumPy for MATLAB Users](https://numpy.org/doc/stable/user/numpy-for-matlab-users.html)

---

## 🚀 Quick Wins Checklist

- [ ] Replace all `.iterrows()` with vectorized operations
- [ ] Convert `range(len())` loops to direct array indexing
- [ ] Use `np.cumsum/cumprod` instead of manual accumulation
- [ ] Replace conditionals with `np.where()` or `np.select()`
- [ ] Batch process DataFrames with `.apply()` instead of row loops
- [ ] Pre-compute reusable arrays outside loops
- [ ] Use broadcasting for multi-dimensional operations
- [ ] Profile code to find actual bottlenecks

---

**Remember:** The goal is faster, more maintainable code. Always verify correctness and measure performance gains!