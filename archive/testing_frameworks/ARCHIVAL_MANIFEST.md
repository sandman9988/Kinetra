# Testing Framework Archival Manifest

**Archive Date**: 2025-01-04  
**Version**: 1.0.0  
**Consolidation Initiative**: Kinetra Project Cleanup

---

## Summary

| Metric | Count |
|--------|-------|
| Files Archived | 55 |
| Files Retained | 8 |
| Reduction | ~87% |

---

## Retained Files (Canonical)

These files remain in `scripts/testing/` as the canonical testing infrastructure:

| File | Purpose | Version |
|------|---------|---------|
| `comprehensive_e2e_test.py` | **CANONICAL** End-to-end testing framework | v2.1.0 |
| `batch_backtest.py` | Batch backtest runner | v1.0.0 |
| `conftest.py` | pytest configuration | - |
| `test_metaapi_auth.py` | MetaAPI authentication tests | v1.0.0 |
| `test_mt5_authentication.py` | MT5 authentication tests | v1.0.0 |
| `test_mt5_friction.py` | MT5 friction cost tests | v1.0.0 |
| `test_mt5_logger.py` | MT5 logging tests | v1.0.0 |
| `test_mt5_vantage_full.py` | Vantage broker integration tests | v1.0.0 |
| `README.md` | Testing documentation | - |
| `E2E_TEST_README.md` | E2E test documentation | - |
| `run_continuous_menu_test.sh` | Shell script for continuous testing | - |

---

## Archived Files

All files moved to `archive/testing_frameworks/legacy/`:

### Menu/Integration Tests (5 files)
- `continuous_menu_test.py`
- `exercise_menu_continuous.py`
- `exercise_menu_with_real_data.py`
- `test_menu.py`
- `test_all_menu_options.py`

### Framework Files (4 files)
- `unified_test_framework.py`
- `example_testing_framework.py`
- `continuous_fix_pipeline.py`
- `AUTOMATED_AUDIT_FIX.py`

### Backtest Test Scripts (9 files)
- `run_comprehensive_backtest.py`
- `run_full_backtest.py`
- `run_exploration_backtest.py`
- `run_physics_backtest.py`
- `run_scientific_testing.py`
- `run_live_test.py`
- `integrate_realistic_backtest.py`
- `demo_backtest_improvements.py`
- `rl_backtest.py`

### E2E Test Scripts (5 files)
- `test_end_to_end.py`
- `test_e2e_symbols_timeframes.py`
- `test_framework_integration.py`
- `test_real_data_backtest.py`
- `test_infrastructure_modules.py`

### Physics/Strategy Tests (11 files)
- `test_physics_demo.py`
- `test_strategies.py`
- `test_berserker_strategy.py`
- `test_freeze_zones.py`
- `test_friction_costs.py`
- `test_energy_recovery_hypotheses.py`
- `test_doppelganger_triad.py`
- `test_regime_filtering.py`
- `test_exploration_strategies.py`
- `test_experience_replay.py`
- `test_sac.py`

### Performance/Numerical Tests (5 files)
- `test_parallel_performance.py`
- `test_performance_module.py`
- `test_numerical_safety.py`
- `test_backtest_numerical_validation.py`
- `test_backtest_trend.py`

### Integration Tests (6 files)
- `test_p0_p5_integration.py`
- `test_multi_instrument.py`
- `test_portfolio_health.py`
- `test_trade_lifecycle.py`
- `test_trade_lifecycle_real_data.py`
- `test_transaction_log.py`

### Validation Scripts (5 files)
- `validate_btc_h1_layer1.py`
- `validate_mql5_compliance.py`
- `validate_theorems.py`
- `validate_thesis.py`
- `verify_calculations.py`

### Misc Scripts (5 files)
- `generate_diagnostic_report.py`
- `multi_tf_test.py`
- `phase2_validation.py`
- `test_marginal_gains.py`
- `test_grafana_export.py`

---

## Functionality Migration Map

| Archived Feature | Now Located In |
|------------------|----------------|
| Menu exercising | `comprehensive_e2e_test.py` → `--menu` flag |
| Auto-fix pipeline | `comprehensive_e2e_test.py` → `--fix` flag |
| E2E validation | `comprehensive_e2e_test.py` → `--full` mode |
| Quick smoke tests | `comprehensive_e2e_test.py` → `--quick` mode |
| Backtest runs | `batch_backtest.py` or menu option |
| Physics unit tests | `tests/test_physics.py` |
| Backtest unit tests | `tests/test_backtest_engine.py` |
| RL unit tests | `tests/test_rl_agent.py` |
| Data validation | `scripts/data_manager.py validate` |

---

## Usage After Consolidation

### Quick E2E Test
```bash
python scripts/testing/comprehensive_e2e_test.py --quick --fix
```

### Full E2E Test with Report
```bash
python scripts/testing/comprehensive_e2e_test.py --full --fix --report
```

### Unit Tests
```bash
pytest tests/ -v
```

### Batch Backtest
```bash
python scripts/batch_backtest.py --instrument BTCUSD --timeframe H1
```

### Data Validation
```bash
python scripts/data_manager.py validate --snapshot "baseline"
```

---

## Restoration Policy

**DO NOT** restore archived files directly. Instead:

1. Check if functionality exists in canonical framework
2. If missing, enhance `comprehensive_e2e_test.py`
3. Follow versioning rules (increment MINOR for features)
4. Document changes in CHANGELOG

---

## Verification

After archival, verify tests still pass:

```bash
# Quick verification
python scripts/testing/comprehensive_e2e_test.py --quick --fix

# Full verification
pytest tests/ -v
```

---

## Related Documents

- `archive/testing_frameworks/legacy/README.md` - Legacy files documentation
- `scripts/testing/README.md` - Current testing guide
- `AGENT_RULES_MASTER.md` - Consolidation rules
- `docs/TESTING_FRAMEWORK.md` - Testing architecture

---

**Archived by**: Kinetra Consolidation Initiative  
**Approved by**: Project Guidelines  
**Status**: ✅ Complete