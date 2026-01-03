# Vectorization Audit Report
**Date:** 2024
**Project:** Kinetra Trading System
**Rule Applied:** Prefer Vectorization Over Python Loops

## Executive Summary

This audit identifies Python loops throughout the Kinetra project that can be replaced with NumPy/Pandas vectorized operations for improved performance. Explicit Python loops should be the last resort - prefer vectorized operations, broadcasting, and Pandas column operations.

---

## 🔴 HIGH PRIORITY - Critical Performance Impact

### 1. `kinetra/assumption_free_measures.py`

#### **DirectionalOrderFlow.extract_features** (Lines 317-384)
**Current Code:**
```python
for i in range(len(subset)):
    vol = subset[vol_col].iloc[i] if vol_col else 1.0
    buy, sell = DirectionalOrderFlow.compute_bar_pressure(
        subset['open'].iloc[i],
        subset['high'].iloc[i],
        subset['low'].iloc[i],
        subset['close'].iloc[i],
        vol
    )
    buy_pressures.append(buy)
    sell_pressures.append(sell)
```

**Vectorized Solution:**
```python
# Vectorize the entire computation
opens = subset['open'].values
highs = subset['high'].values
lows = subset['low'].values
closes = subset['close'].values
vols = subset[vol_col].values if vol_col else np.ones(len(subset))

# Compute all at once using NumPy broadcasting
buy_pressures = (highs - opens) * vols
sell_pressures = (opens - lows) * vols
```

**Impact:** Major - This runs in every feature extraction cycle.

---

#### **AsymmetricReturns.extract_features - Streak Analysis** (Lines 105-116)
**Current Code:**
```python
for i in range(len(up) - 1, -1, -1):
    bar_return = up[i] + down[i]
    bar_sign = np.sign(bar_return)
    if bar_sign == last_sign and bar_sign != 0:
        current_streak += 1
        streak_magnitude += bar_return
        streak_returns.append(bar_return)
    else:
        break
```

**Vectorized Solution:**
```python
# Vectorized streak detection
returns = up + down
signs = np.sign(returns)
last_sign = signs[-1]

# Find where sign changes
sign_changes = np.diff(signs != last_sign)
if len(sign_changes) > 0:
    last_change = len(signs) - np.where(sign_changes)[0][-1] - 1
else:
    last_change = len(signs)

# Slice to get current streak
streak_returns = returns[-last_change:] if last_sign == signs[-last_change] else []
current_streak = len(streak_returns) * int(last_sign)
streak_magnitude = np.sum(streak_returns)
```

**Impact:** High - Called frequently in feature extraction.

---

### 2. `kinetra/realistic_backtester.py`

#### **RealisticBacktester.run - Signal Processing** (Lines 587-650)
**Current Code:**
```python
for idx, signal in signals.iterrows():
    signal_time = signal['time']
    action = signal['action']
    candle = data[data.index == signal_time]
    # ... process each signal individually
```

**Vectorized Solution:**
```python
# Pre-merge signals with data for vectorized operations
signals_with_data = signals.merge(
    data.reset_index(), 
    left_on='time', 
    right_on='index', 
    how='left'
)

# Vectorized freeze zone checks
freeze_mask = np.isin(
    signals_with_data['time'].dt.hour * 60 + signals_with_data['time'].dt.minute,
    freeze_times
)

# Vectorized spread calculations
signals_with_data['fill_price'] = np.where(
    signals_with_data['action'].isin(['open_long']),
    signals_with_data['close'] + signals_with_data['spread'],
    signals_with_data['close'] - signals_with_data['spread']
)

# Then iterate only over valid, filtered signals (much smaller set)
```

**Impact:** Critical - This is in the main backtest loop.

---

### 3. `scripts/testing/run_comprehensive_backtest.py`

#### **extract_trade_details** (Lines 94-150)
**Current Code:**
```python
for idx, trade in trade_df.iterrows():
    entry_time = trade['EntryTime']
    exit_time = trade['ExitTime']
    # ... extract physics features for each trade
    for col in physics_features.columns:
        try:
            entry_physics[f'entry_{col}'] = float(physics_features.iloc[entry_idx][col])
            exit_physics[f'exit_{col}'] = float(physics_features.iloc[exit_idx][col])
        except:
            entry_physics[f'entry_{col}'] = 0.0
            exit_physics[f'exit_{col}'] = 0.0
```

**Vectorized Solution:**
```python
# Vectorized index finding
entry_indices = physics_features.index.get_indexer(
    trade_df['EntryTime'], 
    method='nearest'
)
exit_indices = physics_features.index.get_indexer(
    trade_df['ExitTime'], 
    method='nearest'
)

# Vectorized feature extraction
entry_features = physics_features.iloc[entry_indices].add_prefix('entry_')
exit_features = physics_features.iloc[exit_indices].add_prefix('exit_')

# Combine with trade data
trades_enhanced = pd.concat([
    trade_df.reset_index(drop=True),
    entry_features.reset_index(drop=True),
    exit_features.reset_index(drop=True)
], axis=1)

# Convert to records only at the end
trades = trades_enhanced.to_dict('records')
```

**Impact:** High - Runs for every backtest with many trades.

---

## 🟡 MEDIUM PRIORITY - Moderate Performance Impact

### 4. `kinetra/agent_factory.py`

#### **AgentAdapter.train_episode - Manual Update Loop** (Lines 211-221)
**Current Code:**
```python
for i in range(len(states) - 1):
    self.update(
        states[i],
        int(actions[i]),
        float(rewards[i]),
        states[i + 1],
        i == len(states) - 2,
    )
```

**Vectorized Solution:**
```python
# Check if agent supports batch updates
if hasattr(self.agent, 'batch_update'):
    # Pass entire arrays at once
    self.agent.batch_update(
        states[:-1],  # All states except last
        actions[:-1].astype(int),
        rewards[:-1].astype(float),
        states[1:],   # Next states
        np.arange(len(states)-1) == len(states)-2  # Done flags
    )
else:
    # Fall back to loop only if necessary
    for i in range(len(states) - 1):
        self.update(...)
```

**Impact:** Medium - Depends on agent implementation.

---

### 5. `kinetra/backtest_engine.py`

#### **BacktestEngine._shuffle_returns - Monte Carlo** (Lines 1086-1087)
**Current Code:**
```python
for r in shuffled_returns:
    new_close.append(new_close[-1] * (1 + r))
```

**Vectorized Solution:**
```python
# Vectorized cumulative product
price_multipliers = 1 + shuffled_returns
new_close = initial_price * np.cumprod(price_multipliers)
```

**Impact:** Medium - Monte Carlo simulations are expensive.

---

### 6. `kinetra/backtest_optimizer.py`

#### **DistributionAnalyzer.bootstrap_confidence_interval** (Lines 186-187)
**Current Code:**
```python
for _ in range(n_iterations):
    sample = np.random.choice(returns, size=n, replace=True)
    bootstrap_stats.append(statistic(sample))
```

**Vectorized Solution:**
```python
# Generate all bootstrap samples at once
all_samples = np.random.choice(
    returns, 
    size=(n_iterations, n), 
    replace=True
)

# Apply statistic to all samples (if vectorizable)
if statistic == np.mean:
    bootstrap_stats = np.mean(all_samples, axis=1)
elif statistic == np.std:
    bootstrap_stats = np.std(all_samples, axis=1)
else:
    # Vectorize with apply_along_axis if possible
    bootstrap_stats = np.apply_along_axis(statistic, 1, all_samples)
```

**Impact:** Medium - Bootstrap is already expensive.

---

## 🟢 LOW PRIORITY - Minor Performance Impact

### 7. `e2e_testing_framework.py`

#### **InstrumentRegistry.get_all_instruments** (Lines 204-208)
**Current Code:**
```python
all_instruments = []
for instruments in cls.FALLBACK_INSTRUMENTS.values():
    all_instruments.extend(instruments)
```

**Vectorized Solution:**
```python
# Use itertools.chain for flattening
from itertools import chain
all_instruments = list(chain.from_iterable(cls.FALLBACK_INSTRUMENTS.values()))
```

**Impact:** Low - Only runs once at initialization.

---

### 8. `kinetra/gpu_testing.py`

#### **create_test_scenarios_from_df** (Lines 522-530)
**Current Code:**
```python
for _, row in df_results.iterrows():
    key = (row["instrument"], row["timeframe"])
    if key in data_pool:
        scenario = TestScenario(...)
        scenarios.append(scenario)
```

**Vectorized Solution:**
```python
# Filter DataFrame first
valid_df = df_results[
    df_results.apply(
        lambda row: (row['instrument'], row['timeframe']) in data_pool,
        axis=1
    )
]

# Create scenarios using list comprehension or apply
scenarios = [
    TestScenario(
        instrument=row['instrument'],
        timeframe=row['timeframe'],
        asset_class=row['asset_class'],
        data=data_pool[(row['instrument'], row['timeframe'])]
    )
    for _, row in valid_df.iterrows()
]
```

**Impact:** Low - Test setup only.

---

### 9. Various Print/Display Loops

Multiple files contain loops for printing results:
- `e2e_testing_framework.py` (Lines 703-717)
- `scripts/analysis/pathfinder_deep_dive.py` (Lines 305-309)
- `scripts/testing/run_full_backtest.py` (Lines 169-173, 185-189)

**Note:** These are acceptable - they're I/O bound, not compute bound. Vectorization not needed.

---

## 📊 Summary Statistics

| Priority | Count | Estimated Speedup |
|----------|-------|------------------|
| 🔴 High  | 3     | 10-100x          |
| 🟡 Medium| 3     | 2-10x            |
| 🟢 Low   | 2     | 1.1-2x           |

---

## 🛠️ Implementation Guidelines

### General Principles

1. **Always prefer:**
   - NumPy vectorized operations (`np.sum`, `np.mean`, `np.where`, etc.)
   - Pandas column operations (`df['col'].method()`)
   - Broadcasting for element-wise operations
   - `np.cumprod`, `np.cumsum` for cumulative operations

2. **Avoid:**
   - `.iterrows()` on DataFrames (very slow)
   - `.iloc[i]` in loops
   - `.append()` in loops (list or DataFrame)
   - Manual accumulation when cumulative functions exist

3. **When loops are unavoidable:**
   - Keep them tight and local
   - Pre-allocate arrays when possible
   - Consider numba `@jit` decoration
   - Use list comprehensions over explicit loops

### Testing Strategy

After each vectorization:
1. **Correctness:** Verify output matches original
2. **Performance:** Benchmark with `timeit` or `cProfile`
3. **Edge Cases:** Test with empty arrays, single elements, NaN values

---

## 🎯 Recommended Implementation Order

1. **Phase 1 - Critical Path** (Week 1)
   - `DirectionalOrderFlow.extract_features` 
   - `RealisticBacktester.run` signal processing
   - `extract_trade_details` 

2. **Phase 2 - Feature Engineering** (Week 2)
   - `AsymmetricReturns` streak analysis
   - `AgentAdapter.train_episode`

3. **Phase 3 - Optimization Tools** (Week 3)
   - Bootstrap confidence intervals
   - Monte Carlo shuffle

4. **Phase 4 - Cleanup** (Week 4)
   - Remaining low-priority items
   - Code review and benchmarking

---

## 📝 Notes

- Some loops iterate over heterogeneous data structures (test configurations, display) - these are acceptable
- Database/file I/O loops are fine - they're I/O bound
- Loops that call complex external APIs may not benefit from vectorization
- Always benchmark - sometimes a clear loop is better than obscure vectorization

---

## ✅ Action Items

- [ ] Create unit tests for each function before vectorizing
- [ ] Set up benchmarking infrastructure
- [ ] Vectorize high-priority items
- [ ] Document performance improvements
- [ ] Update coding guidelines to enforce vectorization
- [ ] Add pre-commit hooks to flag `.iterrows()` usage

---

**End of Report**