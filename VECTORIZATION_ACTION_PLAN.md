# Vectorization Action Plan - Kinetra Project

**Date:** 2024
**Status:** 🔴 NEEDS ATTENTION
**Total Violations Found:** 657 (56 High Priority, 601 Medium Priority)

---

## 📊 Executive Summary

The Kinetra codebase has **657 vectorization violations** across 192 files. The "Prefer Vectorization Over Python Loops" rule is being violated extensively, creating significant performance bottlenecks.

### Current State
- **🔴 High Priority:** 56 violations (DataFrame.iterrows, range(len()) with iloc)
- **🟡 Medium Priority:** 601 violations (list.append in loops)
- **💰 Estimated Performance Gains:** 10-100x speedup for critical paths

### Impact Assessment
- **Backtesting:** Primary performance bottleneck in signal processing and trade analysis
- **Feature Engineering:** DirectionalOrderFlow and other measures running in tight loops
- **Optimization:** Bootstrap and Monte Carlo simulations could be 50-100x faster

---

## 🎯 Strategic Priorities

### Phase 1: Critical Path (Week 1-2) - 80% Impact
**Target:** High-priority violations in hot paths

1. **`kinetra/assumption_free_measures.py`**
   - DirectionalOrderFlow.extract_features (line 317-384)
   - AsymmetricReturns streak analysis (line 105-116)
   - **Impact:** Called thousands of times per backtest
   - **Expected Speedup:** 50-100x

2. **`kinetra/realistic_backtester.py`**
   - Signal processing loop (line 587-650)
   - **Impact:** Main backtest loop
   - **Expected Speedup:** 10-20x

3. **`scripts/testing/run_comprehensive_backtest.py`**
   - extract_trade_details (line 94-150)
   - **Impact:** Trade analysis for every backtest
   - **Expected Speedup:** 20-50x

### Phase 2: Feature Engineering (Week 3-4) - 15% Impact
**Target:** Remaining high-priority in feature extraction

4. **`kinetra/liquidity_features.py`** (2 violations)
5. **`scripts/research/fat_candle_forensics.py`** (3 violations)
6. **Agent training loops** (various files)

### Phase 3: Optimization Infrastructure (Week 5-6) - 5% Impact
**Target:** Medium-priority in optimization tools

7. **`kinetra/backtest_optimizer.py`** (13 violations)
8. Bootstrap and Monte Carlo functions
9. Genetic algorithm population handling

---

## 🔧 Implementation Strategy

### Step 1: Setup (Day 1)
```bash
# Install linter as pre-commit hook
python scripts/vectorization_linter.py --install-hook

# Run baseline benchmark
python scripts/benchmark_performance.py --baseline
```

### Step 2: Create Test Suite (Day 2-3)
For each function to be vectorized:
1. Extract current implementation to `_original_` function
2. Create unit test comparing original vs vectorized
3. Include edge cases (empty arrays, NaN, single element)
4. Add performance benchmark

**Example Test Template:**
```python
def test_vectorized_order_flow():
    """Test that vectorized version matches original."""
    # Generate test data
    test_df = create_test_ohlcv_data(n=1000)
    
    # Run both versions
    original = DirectionalOrderFlow.extract_features_original(test_df)
    vectorized = DirectionalOrderFlow.extract_features(test_df)
    
    # Assert equality
    for key in original:
        np.testing.assert_allclose(
            original[key], 
            vectorized[key], 
            rtol=1e-10,
            err_msg=f"Mismatch in {key}"
        )
    
    # Benchmark
    time_original = timeit.timeit(lambda: extract_original(test_df), number=100)
    time_vectorized = timeit.timeit(lambda: extract_vectorized(test_df), number=100)
    speedup = time_original / time_vectorized
    
    assert speedup > 5, f"Expected >5x speedup, got {speedup:.1f}x"
```

### Step 3: Vectorize High Priority (Week 1-2)

#### Example 1: DirectionalOrderFlow
**Before:**
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

**After:**
```python
# Extract arrays once
opens = subset['open'].values
highs = subset['high'].values
lows = subset['low'].values
closes = subset['close'].values
vols = subset[vol_col].values if vol_col else np.ones(len(subset))

# Vectorized computation
buy_pressures = (highs - opens) * vols
sell_pressures = (opens - lows) * vols
```

#### Example 2: Trade Detail Extraction
**Before:**
```python
for idx, trade in trade_df.iterrows():
    for col in physics_features.columns:
        entry_physics[f'entry_{col}'] = float(physics_features.iloc[entry_idx][col])
```

**After:**
```python
# Vectorized index finding
entry_indices = physics_features.index.get_indexer(
    trade_df['EntryTime'], method='nearest'
)
exit_indices = physics_features.index.get_indexer(
    trade_df['ExitTime'], method='nearest'
)

# Batch feature extraction
entry_features = physics_features.iloc[entry_indices].add_prefix('entry_')
exit_features = physics_features.iloc[exit_indices].add_prefix('exit_')

# Combine
trades_enhanced = pd.concat([
    trade_df.reset_index(drop=True),
    entry_features.reset_index(drop=True),
    exit_features.reset_index(drop=True)
], axis=1)
```

### Step 4: Continuous Monitoring
```bash
# Add to CI/CD pipeline
python scripts/vectorization_linter.py --severity high

# Weekly performance tracking
python scripts/benchmark_performance.py --compare-baseline
```

---

## 📈 Success Metrics

### Performance KPIs
- [ ] Full backtest runtime: **Target 50% reduction** (from ~10min to ~5min)
- [ ] Feature extraction: **Target 80% reduction** (from ~2min to ~24sec)
- [ ] Monte Carlo simulations: **Target 90% reduction** (from ~30min to ~3min)

### Code Quality KPIs
- [ ] High-priority violations: **0** (currently 56)
- [ ] Medium-priority violations: **<100** (currently 601)
- [ ] Test coverage of vectorized functions: **100%**
- [ ] Performance regression tests: **All passing**

---

## 🚧 Risk Management

### Risks & Mitigations

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Numerical precision differences | High | Low | Comprehensive unit tests with `assert_allclose` |
| Edge case bugs (NaN, empty) | Medium | High | Explicit edge case testing |
| Performance regression | Low | High | Automated benchmarking in CI |
| Breaking existing code | Medium | High | Keep original implementations, gradual rollout |

### Rollback Plan
1. All original implementations preserved with `_original` suffix
2. Feature flag for new vectorized code: `USE_VECTORIZED=True`
3. A/B testing in production: 50% traffic to vectorized, 50% to original
4. Automated alerts if results differ by >0.01%

---

## 👥 Team Responsibilities

### Week 1-2: Critical Path
- **Engineer 1:** DirectionalOrderFlow vectorization
- **Engineer 2:** RealisticBacktester signal processing
- **Engineer 3:** Test suite creation and benchmarking

### Week 3-4: Feature Engineering
- **Engineer 1:** Remaining liquidity/volatility features
- **Engineer 2:** Fat candle forensics and research scripts
- **Engineer 3:** Agent training loop optimization

### Week 5-6: Infrastructure
- **Engineer 1:** Backtest optimizer vectorization
- **Engineer 2:** Monte Carlo/Bootstrap improvements
- **All:** Code review, documentation, performance validation

---

## 📚 Resources & Tools

### Documentation
- [Vectorization Quick Reference](./VECTORIZATION_GUIDE.md)
- [Full Audit Report](./vectorization_audit_report.md)
- [NumPy Performance Tips](https://numpy.org/doc/stable/user/c-info.python-as-glue.html)

### Tools
- **Linter:** `python scripts/vectorization_linter.py`
- **Profiler:** `python -m cProfile -o profile.stats script.py`
- **Visualizer:** `snakeviz profile.stats`
- **Line profiler:** `kernprof -l -v script.py`

### Benchmarking Template
```python
import timeit
import numpy as np

def benchmark_function(func, *args, n_runs=1000):
    """Benchmark a function."""
    timer = timeit.Timer(lambda: func(*args))
    times = timer.repeat(repeat=5, number=n_runs)
    avg_time = np.mean(times) / n_runs
    std_time = np.std(times) / n_runs
    return avg_time, std_time

# Usage
time_old, std_old = benchmark_function(old_implementation, test_data)
time_new, std_new = benchmark_function(new_implementation, test_data)
speedup = time_old / time_new

print(f"Old: {time_old*1000:.3f} ± {std_old*1000:.3f} ms")
print(f"New: {time_new*1000:.3f} ± {std_new*1000:.3f} ms")
print(f"Speedup: {speedup:.1f}x")
```

---

## 🎓 Learning Resources

### Internal Training (Week 1)
- **Session 1:** NumPy Broadcasting Fundamentals (2h)
- **Session 2:** Pandas Vectorization Patterns (2h)
- **Session 3:** Performance Profiling Workshop (2h)

### External Resources
- [Effective Pandas Book](https://effectivepandas.com/)
- [NumPy Tutorial](https://numpy.org/doc/stable/user/quickstart.html)
- [Pandas Performance Guide](https://pandas.pydata.org/docs/user_guide/enhancingperf.html)

---

## 📅 Timeline

```
Week 1: Setup + High Priority Batch 1
├── Day 1-2: Test infrastructure
├── Day 3-4: DirectionalOrderFlow vectorization
└── Day 5: Testing and benchmarking

Week 2: High Priority Batch 2
├── Day 1-3: RealisticBacktester vectorization
├── Day 4-5: Trade extraction vectorization
└── Day 5: Integration testing

Week 3-4: Medium Priority
├── Feature engineering optimizations
├── Research script improvements
└── Agent training loops

Week 5-6: Infrastructure & Cleanup
├── Optimizer improvements
├── Documentation
└── Performance validation

Week 7: Buffer & Deployment
├── Final testing
├── Production deployment
└── Monitoring setup
```

---

## ✅ Acceptance Criteria

### Phase 1 Complete When:
- [ ] All 56 high-priority violations resolved
- [ ] 100% unit test coverage for vectorized functions
- [ ] Minimum 10x speedup demonstrated on benchmarks
- [ ] No numerical differences >1e-10 from original
- [ ] Code review approved by 2+ engineers

### Project Complete When:
- [ ] <50 total violations remaining (92% reduction)
- [ ] Backtest runtime reduced by 50%+
- [ ] Feature extraction 80% faster
- [ ] CI/CD pipeline includes vectorization checks
- [ ] Team trained on vectorization best practices
- [ ] Documentation complete and published

---

## 🚀 Quick Start

```bash
# 1. Review current violations
python scripts/vectorization_linter.py --severity high

# 2. Read the guide
cat VECTORIZATION_GUIDE.md

# 3. Run baseline benchmarks
python scripts/benchmark_performance.py --baseline

# 4. Start with highest impact file
# See: vectorization_audit_report.md Section "HIGH PRIORITY"

# 5. Create branch
git checkout -b vectorization/directional-order-flow

# 6. Implement, test, benchmark, commit
# 7. Create PR with performance benchmarks included
```

---

## 📞 Support

- **Questions:** Post in #vectorization-project Slack channel
- **Code Review:** Tag @performance-team
- **Blockers:** Escalate to Engineering Lead
- **Performance Regressions:** Alert @on-call immediately

---

**Last Updated:** 2024
**Next Review:** After Phase 1 completion
**Owner:** Performance Engineering Team