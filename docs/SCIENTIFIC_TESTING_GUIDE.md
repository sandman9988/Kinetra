# Scientific Testing Framework - Complete Guide

## Overview

The Kinetra Scientific Testing Framework is a comprehensive, statistically rigorous testing system designed to discover, validate, and backtest trading strategies through first-principles analysis and empirical validation.

## Philosophy

**"We don't know what we don't know"**

This framework systematically explores all possibilities with statistical rigor, automatic error handling, and continuous validation. It focuses on:

1. **Empirical**: Grounded in observed patterns from data
2. **First-Principled**: Anchored in fundamental market dynamics
3. **Statistically Sound**: Incorporates controls, significance testing, and overfitting prevention

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Master Orchestrator                       │
│           (run_scientific_testing.py)                        │
└────────────┬────────────────────────────────────────────────┘
             │
┌────────────┴────────────────────────────────────────────────┐
│  Phase 1: Data Validation & Quality                         │
│  - Stationarity tests                                        │
│  - Asymmetry detection (skew/kurtosis)                      │
│  - Basic integrity checks                                    │
└────────────┬────────────────────────────────────────────────┘
             │
┌────────────┴────────────────────────────────────────────────┐
│  Phase 2: Discovery Methods                                  │
│  ┌──────────────────────────────────────────────────┐       │
│  │ Hidden Dimension Discovery (PCA, ICA)            │       │
│  │ Chaos Theory Analysis (Lyapunov, Hurst)          │       │
│  │ Adversarial Discovery (GAN-style filtering)      │       │
│  │ Meta-Learning (Feature combination discovery)    │       │
│  └──────────────────────────────────────────────────┘       │
└────────────┬────────────────────────────────────────────────┘
             │
┌────────────┴────────────────────────────────────────────────┐
│  Phase 3: Statistical Validation                             │
│  - PBO (Probability of Backtest Overfitting)                │
│  - CPCV (Combinatorially Purged Cross-Validation)           │
│  - Bootstrap confidence intervals                            │
│  - Monte Carlo permutation tests                             │
└────────────┬────────────────────────────────────────────────┘
             │
┌────────────┴────────────────────────────────────────────────┐
│  Phase 4: Integrated Backtesting                             │
│  - Realistic cost modeling (spread, commission, slippage)   │
│  - Walk-forward validation                                   │
│  - Efficiency metrics (MFE/MAE, Pythagorean)                │
│  - Regime-specific validation                                │
└────────────┬────────────────────────────────────────────────┘
             │
┌────────────┴────────────────────────────────────────────────┐
│  Phase 5: Reporting & Synthesis                              │
│  - Comprehensive reports                                     │
│  - Performance rankings                                      │
│  - Statistical significance summaries                        │
└──────────────────────────────────────────────────────────────┘
```

## Components

### 1. Test Executor (`test_executor.py`)

Automated test execution with auto-fixing and continuation capabilities.

**Features:**
- Automatic error detection and remediation
- Progress checkpointing and resumption
- Retry logic with exponential backoff
- Statistical validation at every step

**Example:**
```python
from kinetra.test_executor import TestExecutor, ExecutionConfig

config = ExecutionConfig(
    name="my_test_suite",
    max_retries=3,
    auto_fix_enabled=True,
    continue_on_failure=True
)

executor = TestExecutor(config)

results = executor.execute_test_suite(
    test_functions=[test_func1, test_func2],
    test_names=['test1', 'test2']
)
```

### 2. Discovery Methods (`discovery_methods.py`)

Multiple discovery methods to find patterns we wouldn't think to look for manually.

**Available Methods:**

#### Hidden Dimension Discovery
Discovers latent factors using dimensionality reduction:
- PCA (Principal Component Analysis)
- ICA (Independent Component Analysis)
- Future: Autoencoders, t-SNE, UMAP

**Example:**
```python
from kinetra.discovery_methods import HiddenDimensionDiscovery

method = HiddenDimensionDiscovery()
result = method.discover(
    data=market_data,
    config={'methods': ['pca', 'ica'], 'latent_dims': [8, 16]}
)
```

#### Chaos Theory Analysis
Analyzes markets through chaos theory lens:
- Lyapunov exponents (sensitivity to initial conditions)
- Hurst exponent (trending vs. mean-reverting)
- Approximate entropy (randomness measure)

**Example:**
```python
from kinetra.discovery_methods import ChaosTheoryDiscovery

method = ChaosTheoryDiscovery()
result = method.discover(data=market_data, config={})

# Result contains:
# - lyapunov_exponent (positive = chaotic)
# - hurst_exponent (>0.5 = trending, <0.5 = mean-reverting)
# - approximate_entropy (higher = more random)
```

#### Adversarial Discovery
GAN-style approach where discriminator filters out random patterns:
- Permutation tests for statistical significance
- Feature survival analysis
- P-value corrections (Bonferroni)

**Example:**
```python
from kinetra.discovery_methods import AdversarialDiscovery

method = AdversarialDiscovery()
result = method.discover(data=market_data, config={})

# Only features that survive adversarial testing are returned
significant_features = result.optimal_parameters['significant_features']
```

#### Meta-Learning Discovery
Learns which feature combinations work best:
- Systematic feature combination testing
- Cross-context validation
- Combination importance scoring

### 3. Statistical Rigor (`test_executor.py`)

Comprehensive statistical validation tools:

#### PBO (Probability of Backtest Overfitting)
Measures likelihood that in-sample performance is due to overfitting.

**Interpretation:**
- PBO < 0.05: Good (low overfitting risk)
- PBO > 0.5: Bad (likely overfit)

**Example:**
```python
from kinetra.test_executor import StatisticalRigor

validator = StatisticalRigor()

pbo = validator.calculate_pbo(
    returns_is=in_sample_returns,
    returns_oos=out_of_sample_returns,
    n_trials=1000
)

if pbo < 0.05:
    print("Strategy likely has real edge")
else:
    print("Strategy may be overfit")
```

#### CPCV (Combinatorially Purged Cross-Validation)
Prevents information leakage in financial time series:
- Purges overlapping observations
- Adds embargo periods between train/test
- Uses combinatorial splits

**Example:**
```python
splits = validator.combinatorially_purged_cv(
    data=market_data,
    n_splits=5,
    embargo_pct=0.01
)

for train_idx, test_idx in splits:
    # Train on train_idx, validate on test_idx
    pass
```

#### Bootstrap Confidence Intervals
Non-parametric confidence interval estimation:

**Example:**
```python
lower, upper = validator.bootstrap_confidence_interval(
    data=returns,
    statistic_func=np.mean,
    n_bootstrap=1000,
    confidence_level=0.95
)

print(f"95% CI for mean return: [{lower:.2%}, {upper:.2%}]")
```

#### Monte Carlo Permutation Test
Tests if two samples are significantly different:

**Example:**
```python
p_value = validator.monte_carlo_permutation_test(
    sample1=strategy_a_returns,
    sample2=strategy_b_returns,
    n_permutations=10000
)

if p_value < 0.05:
    print("Strategies are significantly different")
```

### 4. Integrated Backtester (`integrated_backtester.py`)

Realistic backtesting with full cost modeling.

**Features:**
- Spread, commission, and slippage modeling
- MFE/MAE (Maximum Favorable/Adverse Excursion) tracking
- Pythagorean efficiency (shortest path vs. actual path)
- Walk-forward validation
- Statistical significance testing

**Example:**
```python
from kinetra.integrated_backtester import IntegratedBacktester, BacktestConfig

backtester = IntegratedBacktester(output_dir="backtest_results")

config = BacktestConfig(
    name="my_strategy",
    strategy_type="discovered",
    spread_pips=2.0,
    commission_per_lot=7.0,
    slippage_pips=0.5
)

result = backtester.backtest_discovered_strategy(
    strategy_config=strategy_config,
    data=market_data,
    config=config
)

print(f"Sharpe Ratio: {result.sharpe_ratio:.2f}")
print(f"Win Rate: {result.win_rate:.2%}")
print(f"MFE Captured: {result.mfe_captured_pct:.1f}%")
print(f"Statistically Significant: {result.is_statistically_significant}")
```

### 5. Master Orchestrator (`run_scientific_testing.py`)

Complete testing programme execution.

**Example:**
```bash
# Full scientific testing program
python scripts/run_scientific_testing.py --full

# Quick validation run (fewer discoveries, faster)
python scripts/run_scientific_testing.py --quick

# Specific phase only
python scripts/run_scientific_testing.py --phase discovery

# Custom output directory
python scripts/run_scientific_testing.py --full --output-dir my_results
```

## Usage Examples

### Example 1: Quick Validation Run

```python
from scripts.run_scientific_testing import ScientificTestingOrchestrator

orchestrator = ScientificTestingOrchestrator(output_dir="quick_test_results")

# Run quick mode (abbreviated tests)
orchestrator.run_full_programme(quick_mode=True)
```

**Output:**
```
================================================================================
PHASE 1: DATA PREPARATION AND VALIDATION
================================================================================

Validating 6 instruments...
CL-OIL_D1: Validated - 2000 bars, skew=-0.15, kurtosis=2.34
AUDCAD+_H4: Validated - 5000 bars, skew=0.08, kurtosis=1.98
...
Validated 6/6 instruments

================================================================================
PHASE 2: DISCOVERY METHOD EXECUTION
================================================================================

Running 2 discovery methods on 6 instruments...

Discovering patterns in CL-OIL_D1...
Running discovery: hidden_dimensions
PCA 8D explained variance: 0.827
Running discovery: chaos_theory
Lyapunov: 0.0523, Hurst: 0.58, Entropy: 1.24

...

================================================================================
PHASE 3: STATISTICAL VALIDATION
================================================================================

CL-OIL_D1 - hidden_dimensions: PASS (p=0.023)
CL-OIL_D1 - chaos_theory: PASS (p=0.041)
...

Statistical validation: 10/12 discoveries passed

================================================================================
PHASE 4: INTEGRATED BACKTESTING
================================================================================

Backtesting: CL-OIL_D1_hidden_dimensions
  Sharpe: 1.42, Win Rate: 58.3%
...

Completed 10 backtests

================================================================================
PHASE 5: GENERATING FINAL REPORT
================================================================================

Final report saved to scientific_testing_results/reports/...
```

### Example 2: Custom Discovery Pipeline

```python
from kinetra.discovery_methods import DiscoveryMethodRunner

runner = DiscoveryMethodRunner()

# Run specific methods
results = runner.run_all_discoveries(
    data=market_data,
    methods=['chaos_theory', 'adversarial'],
    config={
        'chaos_theory': {},
        'adversarial': {}
    }
)

# Check results
for method_name, result in results.items():
    print(f"\n{method_name}:")
    print(f"  Patterns found: {len(result.discovered_patterns)}")
    print(f"  Statistically significant: {result.statistical_significance}")
    print(f"  P-value: {result.p_value:.4f}")
    
    if result.statistical_significance:
        print(f"  Top features: {list(result.feature_importance.keys())[:5]}")
```

### Example 3: Manual Test Execution with Auto-Fix

```python
from kinetra.test_executor import TestExecutor, ExecutionConfig

def risky_test(param1, param2):
    """Test that might fail."""
    if param1 < 10:
        raise ValueError("param1 too small")
    return {'success': True, 'value': param1 * param2}

config = ExecutionConfig(
    name="risky_tests",
    max_retries=3,
    auto_fix_enabled=True,
    continue_on_failure=True
)

executor = TestExecutor(config, output_dir="test_execution")

results = executor.execute_test_suite(
    test_functions=[risky_test, risky_test],
    test_names=['test1', 'test2'],
    contexts=[
        {'param1': 5, 'param2': 10},   # Will fail initially
        {'param1': 20, 'param2': 5}    # Should pass
    ]
)

# Check results
for name, result in results.items():
    if result['success']:
        print(f"{name}: PASSED after {result['attempts']} attempt(s)")
    else:
        print(f"{name}: FAILED - {result['error']}")

# Load checkpoint to resume later
checkpoint = executor.load_checkpoint()
```

## Statistical Metrics

### Primary Metrics

1. **Sharpe Ratio** (>1.0 target)
   - Risk-adjusted returns
   - `sharpe = mean(returns) / std(returns) * sqrt(252)`

2. **Omega Ratio** (>1.2 target)
   - Probability-weighted ratio of gains vs. losses
   - More robust than Sharpe

3. **Calmar Ratio** (>1.5 target)
   - Return / Maximum Drawdown
   - Measures risk-adjusted performance

4. **Sortino Ratio** (>2.0 target)
   - Like Sharpe but only penalizes downside volatility

### Efficiency Metrics

1. **MFE Captured** (>60% target)
   - Percentage of Maximum Favorable Excursion captured
   - Measures exit timing quality

2. **MAE Ratio** (<2.0 target)
   - Maximum Adverse Excursion / PnL
   - Lower is better

3. **Pythagorean Efficiency** (>0.7 target)
   - Shortest path to profit / Actual path
   - Higher means straighter path to profit

### Statistical Tests

1. **P-value** (<0.05 for significance)
   - Tests if returns are significantly different from zero

2. **PBO** (<0.05 good, >0.5 bad)
   - Probability of Backtest Overfitting

3. **Effect Size (Cohen's d)**
   - 0.2: Small effect
   - 0.5: Medium effect
   - 0.8: Large effect

## Best Practices

### 1. Always Validate Statistically

```python
# Bad: Trust results without validation
result = backtest(strategy)
if result.sharpe_ratio > 1.0:
    deploy(strategy)

# Good: Validate with multiple tests
result = backtest(strategy)

# Check PBO
pbo = calculate_pbo(result.returns_is, result.returns_oos)
if pbo > 0.05:
    print("Warning: Potential overfitting")
    return

# Check statistical significance
if not result.is_statistically_significant:
    print("Warning: Results not statistically significant")
    return

# Check multiple metrics
if (result.sharpe_ratio > 1.0 and
    result.omega_ratio > 1.2 and
    result.calmar_ratio > 1.5 and
    pbo < 0.05):
    deploy(strategy)
```

### 2. Use Cross-Validation

```python
# Bad: Single train/test split
train_data = data[:1000]
test_data = data[1000:]

# Good: CPCV with embargo
from kinetra.test_executor import StatisticalRigor

validator = StatisticalRigor()
splits = validator.combinatorially_purged_cv(data, n_splits=5, embargo_pct=0.01)

results = []
for train_idx, test_idx in splits:
    result = backtest(data.iloc[train_idx], data.iloc[test_idx])
    results.append(result)

# Average metrics across folds
avg_sharpe = np.mean([r.sharpe_ratio for r in results])
```

### 3. Check for Overfitting

```python
# Multiple checks for overfitting
from kinetra.test_executor import StatisticalRigor

validator = StatisticalRigor()

# 1. PBO test
pbo = validator.calculate_pbo(returns_is, returns_oos)
print(f"PBO: {pbo:.3f} {'✓ PASS' if pbo < 0.05 else '✗ FAIL'}")

# 2. IS vs OOS degradation
is_sharpe = np.mean(returns_is) / np.std(returns_is)
oos_sharpe = np.mean(returns_oos) / np.std(returns_oos)
degradation = (is_sharpe - oos_sharpe) / is_sharpe

print(f"OOS degradation: {degradation:.1%} {'✓ PASS' if degradation < 0.2 else '✗ FAIL'}")

# 3. Bootstrap CI
lower, upper = validator.bootstrap_confidence_interval(returns_oos, np.mean)
print(f"95% CI for OOS return: [{lower:.2%}, {upper:.2%}]")

# 4. Permutation test
p_value = validator.monte_carlo_permutation_test(returns_is, returns_oos)
print(f"IS vs OOS p-value: {p_value:.4f} {'✓ SAME' if p_value > 0.05 else '✗ DIFFERENT'}")
```

### 4. Run Discovery Methods

```python
# Don't rely on manual feature selection
# Let discovery methods find patterns

from kinetra.discovery_methods import DiscoveryMethodRunner

runner = DiscoveryMethodRunner()

results = runner.run_all_discoveries(
    data=market_data,
    methods=['hidden_dimensions', 'chaos_theory', 'adversarial', 'meta_learning']
)

# Use only statistically significant discoveries
for method_name, result in results.items():
    if result.statistical_significance and result.p_value < 0.05:
        print(f"✓ {method_name}: Found {len(result.discovered_patterns)} patterns")
        
        # Use optimal parameters for strategy
        strategy = build_strategy(result.optimal_parameters)
        backtest(strategy)
```

## Troubleshooting

### Issue: Tests taking too long

**Solution:** Use quick mode
```bash
python scripts/run_scientific_testing.py --quick
```

Or reduce discovery methods:
```python
results = runner.run_all_discoveries(
    data=data,
    methods=['chaos_theory']  # Just one method
)
```

### Issue: Memory errors

**Solution:** Reduce batch sizes in config
```python
config = ExecutionConfig(
    name="test",
    max_retries=3
)
# Auto-fixer will automatically reduce episodes on MemoryError
```

### Issue: Import errors

**Solution:** Install dependencies
```bash
pip install numpy pandas scipy scikit-learn
```

The AutoFixer will attempt to install missing modules automatically.

### Issue: No patterns found

**Solution:** Check data quality first
```bash
python scripts/run_scientific_testing.py --phase data
```

Ensure:
- Data has >= 1000 bars
- Returns have non-zero variance
- Data is relatively stationary

## API Reference

See individual module docstrings for complete API documentation:

```python
from kinetra import test_executor
from kinetra import discovery_methods
from kinetra import integrated_backtester

help(test_executor.TestExecutor)
help(discovery_methods.HiddenDimensionDiscovery)
help(integrated_backtester.IntegratedBacktester)
```

## Contributing

When adding new discovery methods:

1. Inherit from `DiscoveryMethod` base class
2. Implement `discover()` method
3. Return `DiscoveryResult` with statistical validation
4. Add tests to `test_scientific_framework.py`
5. Update this documentation

## References

- **PBO**: Bailey, D. H., et al. (2014). "The Probability of Backtest Overfitting"
- **CPCV**: de Prado, M. L. (2018). "Advances in Financial Machine Learning"
- **Chaos Theory**: Peters, E. E. (1994). "Fractal Market Analysis"
- **Hurst Exponent**: Mandelbrot, B. B. (1997). "Fractals and Scaling in Finance"
