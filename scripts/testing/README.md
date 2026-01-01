# Testing Scripts

Backtesting, validation, and testing framework scripts.

## Main Testing Framework

- **`unified_test_framework.py`** - Main entry point for unified testing
- **`run_scientific_testing.py`** - Run scientific testing programme
- **`example_testing_framework.py`** - Testing framework examples

## Backtest Runners

- **`run_full_backtest.py`** - Full physics backtest runner
- **`run_physics_backtest.py`** - Physics-specific backtesting
- **`batch_backtest.py`** - Batch backtesting
- **`rl_backtest.py`** - RL agent backtesting
- **`run_comprehensive_backtest.py`** - Comprehensive backtest suite

## Integration Tests

- **`integrate_realistic_backtest.py`** - Realistic backtest integration
- **`demo_backtest_improvements.py`** - Backtest improvements demo
- **`test_framework_integration.py`** - Framework integration tests
- **`test_end_to_end.py`** - End-to-end tests

## Component Tests

- **`test_berserker_strategy.py`** - Berserker strategy tests
- **`test_doppelganger_triad.py`** - Triad system tests
- **`test_freeze_zones.py`** - Freeze zone validation
- **`test_physics.py`** - Physics engine tests
- **`test_sac.py`** - SAC agent tests
- **`test_strategies.py`** - Strategy tests

## Validation Tests

- **`validate_btc_h1_layer1.py`** - BTC H1 layer 1 validation
- **`validate_mql5_compliance.py`** - MQL5 compliance validation
- **`validate_theorems.py`** - Theorem validation
- **`validate_thesis.py`** - Thesis validation
- **`verify_calculations.py`** - Calculation verification

## Data & Infrastructure Tests

- **`test_real_data_backtest.py`** - Real data backtesting
- **`test_trade_lifecycle.py`** - Trade lifecycle tests
- **`test_trade_lifecycle_real_data.py`** - Real data trade lifecycle
- **`test_transaction_log.py`** - Transaction log tests
- **`test_mt5_friction.py`** - MT5 friction modeling
- **`test_mt5_logger.py`** - MT5 logger tests
- **`test_numerical_safety.py`** - Numerical safety tests

## Performance Tests

- **`test_backtest_numerical_validation.py`** - Numerical validation
- **`test_parallel_performance.py`** - Parallel performance tests
- **`test_performance_module.py`** - Performance module tests

## Specialized Tests

- **`test_energy_recovery_hypotheses.py`** - Energy recovery tests
- **`test_experience_replay.py`** - Experience replay tests
- **`test_exploration_strategies.py`** - Exploration strategy tests
- **`test_friction_costs.py`** - Friction cost modeling
- **`test_regime_filtering.py`** - Regime filtering tests
- **`test_marginal_gains.py`** - Marginal gains analysis
- **`test_multi_instrument.py`** - Multi-instrument tests
- **`test_portfolio_health.py`** - Portfolio health tests

## Quick Start

```bash
# Run unified test framework
python scripts/testing/unified_test_framework.py --quick

# Run scientific testing
python scripts/testing/run_scientific_testing.py --full

# Run physics backtest
python scripts/testing/run_full_backtest.py data/master/forex/EURUSD_H1_*.csv

# Validate theorems
python scripts/testing/validate_theorems.py
```
