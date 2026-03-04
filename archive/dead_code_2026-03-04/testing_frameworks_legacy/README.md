# Legacy Testing Frameworks Archive

**Archive Date**: 2025-01-XX  
**Archived By**: Consolidation Initiative  
**Canonical Replacement**: `scripts/testing/comprehensive_e2e_test.py` (v2.1.0)

---

## Overview

This directory contains legacy test scripts and frameworks that have been archived as part of the Kinetra project consolidation initiative. These files have been superseded by the canonical `comprehensive_e2e_test.py` framework.

**DO NOT USE THESE FILES FOR NEW DEVELOPMENT**

---

## Why These Were Archived

1. **Duplication**: Multiple test frameworks existed with overlapping functionality
2. **Inconsistent Standards**: Different testing approaches and validation criteria
3. **Maintenance Burden**: ~63 test files with significant overlap
4. **Consolidation Goal**: Single source of truth for E2E testing

---

## Archived Files

### Menu/Integration Test Frameworks
| File | Original Purpose | Status |
|------|------------------|--------|
| `continuous_menu_test.py` | Continuous menu exercising | Merged into comprehensive_e2e_test.py |
| `exercise_menu_continuous.py` | Menu automation | Merged into comprehensive_e2e_test.py |
| `exercise_menu_with_real_data.py` | Menu with real data | Merged into comprehensive_e2e_test.py |
| `test_menu.py` | Menu unit tests | Merged into comprehensive_e2e_test.py |
| `test_all_menu_options.py` | Full menu coverage | Merged into comprehensive_e2e_test.py |

### Framework Files
| File | Original Purpose | Status |
|------|------------------|--------|
| `unified_test_framework.py` | Unified testing approach | Superseded by comprehensive_e2e_test.py |
| `example_testing_framework.py` | Example framework | Superseded by comprehensive_e2e_test.py |
| `continuous_fix_pipeline.py` | Auto-fix pipeline | Merged into comprehensive_e2e_test.py --fix |

### Backtest Test Scripts
| File | Original Purpose | Status |
|------|------------------|--------|
| `run_comprehensive_backtest.py` | Full backtest runs | Use batch_backtest.py instead |
| `run_full_backtest.py` | Full backtest | Use batch_backtest.py instead |
| `run_exploration_backtest.py` | Exploration backtest | Use batch_backtest.py with flags |
| `run_physics_backtest.py` | Physics-focused backtest | Use batch_backtest.py |
| `integrate_realistic_backtest.py` | Realistic integration | Superseded |
| `demo_backtest_improvements.py` | Demo script | No longer needed |

### Specific Test Scripts
| File | Original Purpose | Status |
|------|------------------|--------|
| `test_end_to_end.py` | E2E tests | Merged into comprehensive_e2e_test.py |
| `test_e2e_symbols_timeframes.py` | Symbol/TF E2E | Merged into comprehensive_e2e_test.py |
| `test_framework_integration.py` | Framework integration | Merged |
| `test_real_data_backtest.py` | Real data tests | Merged |
| `test_infrastructure_modules.py` | Infrastructure tests | Merged |
| `test_p0_p5_integration.py` | Integration tests | Merged |

### Validation Scripts
| File | Original Purpose | Status |
|------|------------------|--------|
| `validate_btc_h1_layer1.py` | BTC H1 validation | Use data_manager.py validate |
| `validate_mql5_compliance.py` | MQL5 compliance | Preserved if needed |
| `validate_theorems.py` | Theorem validation | Keep for theorem CI |
| `validate_thesis.py` | Thesis validation | Academic reference |
| `verify_calculations.py` | Calculation verification | Merged into tests |

### MetaAPI/MT5 Test Scripts
| File | Original Purpose | Status |
|------|------------------|--------|
| `test_metaapi_auth.py` | MetaAPI authentication | Keep for connectivity tests |
| `test_mt5_authentication.py` | MT5 auth | Keep for connectivity tests |
| `test_mt5_friction.py` | MT5 friction costs | Merged into physics tests |
| `test_mt5_logger.py` | MT5 logging | Merged |
| `test_mt5_vantage_full.py` | Vantage integration | Keep for broker tests |

### Physics/Strategy Test Scripts
| File | Original Purpose | Status |
|------|------------------|--------|
| `test_physics_demo.py` | Physics demo | Merged into unit tests |
| `test_strategies.py` | Strategy tests | Merged into unit tests |
| `test_berserker_strategy.py` | Berserker tests | Merged |
| `test_freeze_zones.py` | Freeze zone tests | Merged |
| `test_friction_costs.py` | Friction tests | Merged |
| `test_energy_recovery_hypotheses.py` | Energy recovery | Merged |

### Performance Test Scripts
| File | Original Purpose | Status |
|------|------------------|--------|
| `test_parallel_performance.py` | Parallel perf | Benchmark suite |
| `test_performance_module.py` | Performance module | Merged |

---

## Current Canonical Testing Structure

```
scripts/testing/
├── comprehensive_e2e_test.py  # ✅ CANONICAL E2E (v2.1.0)
├── conftest.py                # ✅ pytest configuration
├── README.md                  # ✅ Testing documentation
├── E2E_TEST_README.md         # ✅ E2E specific docs
└── batch_backtest.py          # ✅ Batch backtest runner

tests/                         # ✅ pytest unit/integration tests
├── test_physics.py
├── test_backtest_engine.py
├── test_rl_agent.py
└── ...
```

---

## How to Use Canonical Framework

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

---

## Restoration Policy

If you need functionality from an archived file:

1. **Check if it exists in the canonical framework first**
2. **If not, propose enhancement to `comprehensive_e2e_test.py`**
3. **Do NOT restore archived files directly**
4. **Follow versioning rules (increment MINOR for new features)**

---

## Contact

For questions about archived files or the canonical framework, refer to:
- `AGENT_RULES_MASTER.md` - Consolidation rules
- `docs/TESTING_FRAMEWORK.md` - Testing documentation
- `scripts/testing/README.md` - Current testing guide

---

**Remember**: Consolidation reduces maintenance burden and ensures consistency. Always enhance the canonical framework rather than creating new files.