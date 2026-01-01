# Backtesting Consolidation Plan

## Consolidation Strategy

Merging 6 files (5,568 lines) into unified `kinetra/backtesting/` package (~2,500 lines).

### Source Files Analysis

| File | Lines | Unique Features | Keep As |
|------|-------|-----------------|---------|
| **backtest_engine.py** | 1,098 | ML/RL integration, GPU, Monte Carlo | **PRIMARY BASE** |
| **realistic_backtester.py** | 1,038 | MT5 freeze zones, dynamic spreads, stops validation | Mode: 'realistic' |
| **physics_backtester.py** | 977 | Physics indicators, backtesting.py lib | Mode: 'physics' |
| **backtest_optimizer.py** | 1,029 | Walk-forward, grid search, hyperparam tuning | optimizer.py |
| **integrated_backtester.py** | 634 | Testing framework bridge | adapters/testing_framework.py |
| **portfolio_backtest.py** | 792 | Multi-instrument, portfolio metrics | portfolio.py |

### Target Package Structure

```
kinetra/backtesting/
├── __init__.py               # Package exports
├── CONSOLIDATION_PLAN.md     # This file
├── core.py                   # UnifiedBacktester (from backtest_engine.py)
├── costs.py                  # Cost models (all sources)
├── constraints.py            # MT5 constraints (from realistic_backtester.py)
├── metrics.py                # Performance metrics (all sources)
├── execution.py              # Trade execution (all sources)
├── optimizer.py              # Parameter optimization (from backtest_optimizer.py)
├── portfolio.py              # Portfolio features (from portfolio_backtest.py)
└── adapters/
    ├── __init__.py
    ├── testing_framework.py  # Testing framework adapter
    └── backtesting_py.py     # backtesting.py wrapper
```

### Feature Consolidation Matrix

#### Cost Modeling
| Feature | Source | Target Module |
|---------|--------|---------------|
| Fixed spreads | backtest_engine.py | costs.py → FixedCostModel |
| Dynamic spreads | realistic_backtester.py | costs.py → DynamicCostModel |
| Commission | All | costs.py → CostModel (base) |
| Slippage | backtest_engine.py | costs.py → CostModel |
| Swap/rollover | backtest_engine.py | costs.py → CostModel |

#### Realism Features
| Feature | Source | Target Module |
|---------|--------|---------------|
| MT5 freeze zones | realistic_backtester.py | constraints.py |
| Stops validation | realistic_backtester.py | constraints.py |
| MT5 error codes | realistic_backtester.py | constraints.py |
| Market hours | realistic_backtester.py | constraints.py |

#### Performance
| Feature | Source | Target Module |
|---------|--------|---------------|
| Parallelization | backtest_engine.py | core.py |
| GPU support | backtest_engine.py | core.py |
| Monte Carlo | backtest_engine.py | core.py |

#### Metrics
| Feature | Source | Target Module |
|---------|--------|---------------|
| Sharpe ratio | All (5 different ways!) | metrics.py → standardized |
| Omega ratio | backtest_engine.py | metrics.py |
| Z-factor | backtest_engine.py | metrics.py |
| MFE/MAE | backtest_engine.py | metrics.py |
| Regime breakdown | realistic_backtester.py | metrics.py |

### Mode-Based Architecture

```python
class UnifiedBacktester:
    """
    Unified backtesting engine with mode selection.
    
    Modes:
    - 'standard': Fast, basic cost modeling
    - 'realistic': MT5-accurate with freeze zones
    - 'physics': Physics-based strategies
    - 'portfolio': Multi-instrument backtesting
    """
    
    def __init__(
        self,
        mode: str = 'standard',
        cost_model: Optional[CostModel] = None,
        constraints: Optional[Constraints] = None,
        parallel: bool = True,
        gpu: bool = False
    ):
        # Configure based on mode
        if mode == 'realistic':
            self.cost_model = cost_model or DynamicCostModel()
            self.constraints = MT5Constraints()
        elif mode == 'physics':
            self.physics_engine = PhysicsEngine()
        # ...
```

### Code Duplication Elimination

#### Before (Duplicated 6x)
```python
# Sharpe ratio calculated 5 different ways across files!
# backtest_engine.py
sharpe = mean(returns) / std(returns) * sqrt(252)

# realistic_backtester.py
sharpe = annualized_return / annualized_volatility

# physics_backtester.py  
sharpe = stats['Sharpe Ratio']  # From backtesting.py
```

#### After (ONE implementation)
```python
# kinetra/backtesting/metrics.py
class MetricsCalculator:
    @staticmethod
    def sharpe_ratio(returns: pd.Series, periods_per_year: int = 252) -> float:
        """Standardized Sharpe ratio calculation."""
        if len(returns) < 2:
            return 0.0
        return returns.mean() / returns.std() * np.sqrt(periods_per_year)
```

### Migration Guide

```python
# OLD - Different backtesters for different purposes
from kinetra.backtest_engine import BacktestEngine
from kinetra.realistic_backtester import RealisticBacktester
from kinetra.physics_backtester import PhysicsBacktester

# Standard backtest
bt1 = BacktestEngine()
results1 = bt1.run(strategy, data)

# MT5-realistic backtest
bt2 = RealisticBacktester()
results2 = bt2.run(strategy, data)

# Physics backtest
bt3 = PhysicsBacktester()
results3 = bt3.run(physics_strategy, data)

# NEW - One backtester, mode selection
from kinetra.backtesting import UnifiedBacktester

# Standard backtest
bt = UnifiedBacktester(mode='standard')
results1 = bt.run(strategy, data)

# MT5-realistic backtest
bt = UnifiedBacktester(mode='realistic', enable_freeze_zones=True)
results2 = bt.run(strategy, data)

# Physics backtest
bt = UnifiedBacktester(mode='physics')
results3 = bt.run(physics_strategy, data)
```

### Benefits

1. **54% Code Reduction**
   - Before: 5,568 lines across 6 files
   - After: ~2,500 lines in unified package
   - Eliminated: ~3,000 lines of duplication

2. **Consistent Interface**
   - ONE way to backtest
   - Mode selection for different scenarios
   - Consistent results format

3. **Easier Testing**
   - Test each module independently
   - Shared test utilities
   - Better coverage

4. **Better Performance**
   - Shared code paths → better caching
   - Unified parallelization
   - Single GPU optimization

### Implementation Steps

1. ✅ Create package structure
2. ⏳ Extract metrics calculation → `metrics.py`
3. ⏳ Extract cost models → `costs.py`
4. ⏳ Extract MT5 constraints → `constraints.py`
5. ⏳ Extract trade execution → `execution.py`
6. ⏳ Build core UnifiedBacktester → `core.py`
7. ⏳ Extract optimizer → `optimizer.py`
8. ⏳ Extract portfolio → `portfolio.py`
9. ⏳ Create adapters
10. ⏳ Update imports
11. ⏳ Deprecate old files
12. ⏳ Integration testing

### Backward Compatibility

```python
# kinetra/backtest_engine.py (deprecated wrapper)
import warnings
from kinetra.backtesting import UnifiedBacktester

class BacktestEngine(UnifiedBacktester):
    def __init__(self, **kwargs):
        warnings.warn(
            "BacktestEngine is deprecated. Use kinetra.backtesting.UnifiedBacktester instead.",
            DeprecationWarning,
            stacklevel=2
        )
        super().__init__(mode='standard', **kwargs)
```

## Status

- [x] Package structure created
- [ ] Metrics module extracted
- [ ] Cost models extracted
- [ ] Constraints extracted
- [ ] Execution logic extracted
- [ ] Core backtester built
- [ ] Optimizer extracted
- [ ] Portfolio extracted
- [ ] Adapters created
- [ ] Imports updated
- [ ] Old files deprecated
- [ ] Tests passing
