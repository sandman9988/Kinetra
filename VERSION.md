# Kinetra Version Manifest

**Last Updated**: 2025-01-04  
**Document Version**: 1.2.0

---

## Overview

This document tracks version numbers for all canonical modules, scripts, and frameworks in the Kinetra project. All components follow semantic versioning (MAJOR.MINOR.PATCH).

### Versioning Rules

- **MAJOR**: Breaking changes, incompatible API changes
- **MINOR**: New features, backward-compatible additions
- **PATCH**: Bug fixes, backward-compatible fixes

---

## Core Modules (`kinetra/`)

| Module | Version | Status | Description |
|--------|---------|--------|-------------|
| `physics_engine.py` | 1.0.0 | ✅ Active | Physics-based market modeling (energy, friction, entropy) |
| `backtest_engine.py` | 1.2.0 | ✅ Active | Backtest engine with friction costs and optimized Monte Carlo |
| `rl_agent.py` | 1.0.0 | ✅ Active | PPO-based reinforcement learning agent |
| `rl_neural_agent.py` | 1.0.0 | ✅ Active | Neural network RL agent variants |
| `sb3_agents.py` | 1.0.0 | ✅ Active | Stable-Baselines3 agent implementations |
| `risk_management.py` | 1.0.0 | ✅ Active | Risk-of-Ruin and position sizing |
| `data_loader.py` | 1.0.0 | ✅ Active | Data loading and normalization |
| `symbol_spec.py` | 1.0.0 | ✅ Active | Instrument specifications |
| `berserker_strategy.py` | 1.0.0 | ✅ Active | Berserker trading strategy |
| `high_performance_engine.py` | 1.0.0 | ✅ Active | GPU-accelerated computations |
| `financial_audit.py` | 1.0.0 | ✅ Active | Numerical safety and audit compliance |
| `persistence_manager.py` | 1.0.0 | ✅ Active | Atomic save and backup management |

---

## Canonical Scripts (`scripts/`)

| Script | Version | Status | Description |
|--------|---------|--------|-------------|
| `batch_backtest.py` | 1.1.0 | ✅ Canonical | Batch backtesting pipeline with parallel processing |
| `data_manager.py` | 1.0.0 | ✅ Canonical | Data preparation and snapshot management (parallel) |
| `download_metaapi.py` | 1.0.0 | ✅ Canonical | MetaAPI data download |

---

## Testing Framework (`scripts/testing/`)

| Component | Version | Status | Description |
|-----------|---------|--------|-------------|
| `comprehensive_e2e_test.py` | 2.1.0 | ✅ Canonical | **CANONICAL** End-to-end testing framework |
| `e2e_stepover_test.py` | 1.0.0 | ✅ Active | Step-by-step E2E test with detailed logging |
| `batch_backtest.py` | 1.0.0 | ✅ Active | Testing batch backtest runner |
| `conftest.py` | - | ✅ Active | pytest configuration |
| `test_metaapi_auth.py` | 1.0.0 | ✅ Active | MetaAPI authentication tests |
| `test_mt5_*.py` | 1.0.0 | ✅ Active | MT5/broker integration tests |

---

## Menu System

| Component | Version | Status | Description |
|-----------|---------|--------|-------------|
| `kinetra_menu.py` | 2.0.0 | ✅ Canonical | **CANONICAL** Main menu interface |

---

## Archived Components

See `archive/` directory for legacy files. Archived components are NOT versioned and should NOT be used.

| Archive Location | Contents | Archive Date |
|------------------|----------|--------------|
| `archive/testing_frameworks/legacy/` | 55 legacy test files | 2025-01-04 |
| `archive/menus/` | Legacy menu implementations | 2025-01-XX |
| `archive/scripts/` | Legacy utility scripts | 2025-01-XX |

---

## Version History

### 2025-01-04: E2E Testing & MC Optimization v1.2.0

**Changes:**
- Created `e2e_stepover_test.py` (v1.0.0) - comprehensive step-by-step E2E test with detailed logging
- Fixed Monte Carlo multiprocessing pickle issue (BacktestEngine v1.1.0)
- Optimized Monte Carlo with fast vectorized worker (BacktestEngine v1.2.0)
  - ~300x speedup per run (1000ms → 3.2ms)
  - Now runs 100 iterations in 318ms vs 3 runs in 3.1s before
- All 8 E2E test steps passing (Environment, Data Discovery, Data Loading, Physics, Backtest, RL Agent, Monte Carlo, Integration)

---

### 2025-01-04: Consolidation Initiative v1.1.0

**Changes:**
- Added `__version__` to all core modules
- Created VERSION.md manifest
- Archived 55 legacy test files (~87% reduction)
- Established canonical testing framework (comprehensive_e2e_test.py v2.1.0)
- Established canonical menu (kinetra_menu.py v2.0.0)
- Established canonical data manager (data_manager.py v1.0.0)
- Added multiprocessing to batch_backtest.py (v1.1.0)
- Added parallel data preparation to data_manager.py
- Vectorized physics_engine.py functions (~50x speedup)
- Vectorized backtest_engine.py Monte Carlo shuffle (~5x speedup)
- Updated AGENT_RULES_MASTER.md to v3.0 with consolidation/parallelization rules

**Breaking Changes:**
- Legacy test scripts moved to archive (use comprehensive_e2e_test.py instead)
- Duplicate menus removed (use kinetra_menu.py instead)

---

## Checking Versions

### Python Import
```python
from kinetra.physics_engine import __version__
print(f"Physics Engine: {__version__}")

from kinetra.backtest_engine import __version__
print(f"Backtest Engine: {__version__}")

from kinetra.rl_agent import __version__
print(f"RL Agent: {__version__}")
```

### CLI Check
```bash
# Check if archived comprehensive_e2e_test.py runs
python archive/production_cleanup_2026-03-03/scripts/testing/comprehensive_e2e_test.py --version

# Check batch_backtest.py
python scripts/batch_backtest.py --help
```

---

## Updating Versions

When making changes:

1. **Bug fix**: Increment PATCH (1.0.0 → 1.0.1)
2. **New feature**: Increment MINOR (1.0.0 → 1.1.0)
3. **Breaking change**: Increment MAJOR (1.0.0 → 2.0.0)

**Always:**
- Update `__version__` in the module
- Update this VERSION.md
- Add entry to "Version History" in module docstring
- Run tests before committing

---

## Dependencies

| Package | Required Version | Purpose |
|---------|------------------|---------|
| Python | ≥3.10 | Core language |
| NumPy | ≥1.24.0 | Vectorized operations |
| Pandas | ≥2.0.0 | Data manipulation |
| PyTorch | ≥2.0.0 | Neural networks (optional) |
| Stable-Baselines3 | ≥2.0.0 | RL algorithms |
| scikit-learn | ≥1.3.0 | ML utilities |
| MetaAPI SDK | ≥26.0.0 | Broker connectivity |

---

## Related Documents

- `AGENT_RULES_MASTER.md` - Versioning and consolidation rules
- `archive/testing_frameworks/ARCHIVAL_MANIFEST.md` - Archive details
- `archive/production_cleanup_2026-03-03/repo/docs/TESTING_FRAMEWORK.md` - Testing documentation
- `CHANGELOG.md` - Project-wide changelog

---

**Maintained by**: Kinetra Project  
**Contact**: See repository maintainers
