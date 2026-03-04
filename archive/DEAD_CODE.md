# Dead Code Identification — Kinetra

**Last updated: 2026-03-04**

This document identifies modules that appear to be unused or deprecated in the current trading pipeline.

## Active Code Path

The main entry point is `scripts/renko_engine.py`, which imports:

```
kinetra/renko/
├── brick_engine.py          # Renko brick construction
├── filters.py               # Signal filtering
├── trading_engine.py        # Core trading engine (LIVE)
├── ctrader_dispatcher.py    # cTrader broker integration
├── live_trader.py           # Live trading utilities
├── dsp.py                   # Brick size optimization
├── backtest.py              # Backtesting engine

kinetra/connectors/
├── ctrader_connector.py     # cTrader TCP/API connection
├── hot_standby.py           # Failover support

kinetra/monitoring/
├── connection_health.py     # Health monitoring service

kinetra/
├── preflight_enhanced.py    # Pre-live safety checks
├── dns_hardening.py         # DNS resilience
├── friction_cost.py         # Trading costs
├── data_utils.py            # Data loading
```

## Dead Code Candidates

### 1. MetaTrader 5 (MT5) Integration
**Files:**
- `kinetra/mt5_bridge.py` - MT5 bridge (unused)
- `kinetra/mt5_live.py` - MT5 live trading (unused)
- `kinetra/mt5_connector.py` - MT5 connector (minimal use)

**Status:** Deprecated — Project now uses cTrader Open API exclusively.

**Archive candidate:** `archive/mt5_legacy/`

---

### 2. Deep Reinforcement Learning (DRL)
**Files:**
- `kinetra/drl_dueling_dqn.py` - Dueling DQN agent
- `kinetra/experience_replay.py` - Experience replay buffer
- `kinetra/reward_shaping.py` - RL reward shaping
- `kinetra/rl_agent.py` - RL agent base
- `kinetra/rl_gpu_trainer.py` - GPU RL training
- `kinetra/rl_neural_agent.py` - Neural agent
- `kinetra/rl_physics_env.py` - Physics environment for RL
- `kinetra/sb3_agents.py` - Stable-Baselines3 agents
- `kinetra/rl/` - RL subpackage

**Status:** Unused — Current strategy uses deterministic Renko brick rules, not RL.

**Archive candidate:** `archive/drl_research/`

---

### 3. Physics Engine
**Files:**
- `kinetra/physics_backtester.py` - Physics-based backtester
- `kinetra/physics_engine.py` - Physics calculations
- `kinetra/physics_v7.py` - Physics v7 variant

**Status:** Partially used — `physics_engine.py` referenced in some scripts but not in main trading pipeline.

**Archive candidate:** `archive/physics_experiments/`

---

### 4. Exploration Lab
**Files:**
- `kinetra/exploration_lab/orchestrator.py` - Exploration orchestrator

**Status:** Unused — Research code, not part of production pipeline.

**Archive candidate:** `archive/exploration_lab/`

---

### 5. Alternative Strategies
**Files:**
- `kinetra/berserker_strategy.py` - Berserker strategy
- `kinetra/triad_system.py` - Triad system
- `kinetra/doppelganger_triad.py` - Doppelganger variant
- `kinetra/strategies_v7.py` - Strategy v7

**Status:** Unused — Current pipeline uses Renko filter-based strategy.

**Archive candidate:** `archive/strategy_experiments/`

---

### 6. Backtest Optimizers
**Files:**
- `kinetra/backtest_optimizer.py` - Backtest optimization
- `kinetra/hpo_optimizer.py` - Hyperparameter optimization
- `kinetra/integrated_backtester.py` - Integrated backtester

**Status:** Unused — Current pipeline uses fixed parameters from DSP analysis.

**Archive candidate:** `archive/optimization_tools/`

---

### 7. Additional Utilities (Low Usage)
**Files:**
- `kinetra/high_performance_engine.py` - Unused performance variant
- `kinetra/network_resilience.py` - Superseded by `hot_standby.py`
- `kinetra/health_monitor.py` - Superseded by `monitoring/connection_health.py`
- `kinetra/health_score.py` - Unused

**Status:** Superseded by newer implementations.

---

## Archive Recommendation

```
archive/
├── mt5_legacy/
│   ├── mt5_bridge.py
│   ├── mt5_live.py
│   └── mt5_connector.py
├── drl_research/
│   ├── drl_dueling_dqn.py
│   ├── experience_replay.py
│   ├── reward_shaping.py
│   ├── rl_agent.py
│   ├── rl_gpu_trainer.py
│   ├── rl_neural_agent.py
│   ├── rl_physics_env.py
│   ├── sb3_agents.py
│   └── rl/
├── physics_experiments/
│   ├── physics_backtester.py
│   ├── physics_engine.py
│   └── physics_v7.py
├── exploration_lab/
│   └── orchestrator.py
├── strategy_experiments/
│   ├── berserker_strategy.py
│   ├── triad_system.py
│   ├── doppelganger_triad.py
│   └── strategies_v7.py
└── optimization_tools/
    ├── backtest_optimizer.py
    ├── hpo_optimizer.py
    └── integrated_backtester.py
```

## Verification Script

```bash
#!/bin/bash
# verify_active_imports.sh

echo "=== Active imports in scripts/renko_engine.py ==="
grep -E "^from kinetra|^import kinetra" scripts/renko_engine.py

echo ""
echo "=== Active imports in kinetra/renko/ ==="
grep -rE "^from kinetra|^import kinetra" kinetra/renko/ | grep -v "^Binary" | head -20

echo ""
echo "=== Unused module check ==="
for f in kinetra/physics_engine.py kinetra/drl_dueling_dqn.py kinetra/mt5_bridge.py; do
    count=$(grep -r "from kinetra.$(basename $f .py)" scripts/ kinetra/renko/ 2>/dev/null | wc -l)
    echo "$f: $count references"
done
```

## Before Archiving

1. **Backup verification:** Ensure all files are in git
2. **Update imports:** Check for any cross-dependencies
3. **Update tests:** Move related tests to `tests/archive/`
4. **Update documentation:** Note archival in relevant docs

## Active Code Inventory

| Module | Status | Used By |
|--------|--------|---------|
| `renko/brick_engine.py` | ✅ Active | `renko_engine.py` |
| `renko/filters.py` | ✅ Active | `renko_engine.py`, `brick_engine.py` |
| `renko/trading_engine.py` | ✅ Active | `renko_engine.py` |
| `renko/ctrader_dispatcher.py` | ✅ Active | `renko_engine.py` |
| `renko/dsp.py` | ✅ Active | `renko_engine.py` |
| `renko/backtest.py` | ✅ Active | `renko_engine.py` |
| `connectors/ctrader_connector.py` | ✅ Active | `renko_engine.py` |
| `connectors/hot_standby.py` | ✅ Active | `ctrader_connector.py` |
| `preflight_enhanced.py` | ✅ Active | `renko_engine.py` |
| `monitoring/connection_health.py` | ✅ Active | `renko_engine.py` |
| `dns_hardening.py` | ✅ Active | `ctrader_connector.py` |

## Total Code Stats

```
# Lines of code (approximate)
Active production code:     ~8,000 lines
Test suite:                 ~2,500 lines
Dead code candidates:       ~15,000 lines
Total codebase:             ~25,500 lines

Potential reduction: 60%
```

---

## Notes

- Dead code is kept for reference during transition period
- Archive process should be gradual and reversible
- Document all archival decisions
- Keep git history for recovery if needed

## Archival Status (2026-03-04)

### Completed Archivals

All files listed below have been moved to `archive/dead_code_2026-03-04/`:

#### MT5 Legacy (`mt5_legacy/`)
- mt5_bridge.py
- mt5_live.py
- mt5_connector.py
- mt5_spec_extractor.py

#### DRL Research (`drl_research/`)
- drl_dueling_dqn.py
- experience_replay.py
- reward_shaping.py
- rl_agent.py
- rl_gpu_trainer.py
- rl_neural_agent.py
- rl_physics_env.py
- sb3_agents.py
- rl/ (directory)
- agent_factory.py

#### Physics Experiments (`physics_experiments/`)
- physics_engine.py
- physics_backtester.py
- physics_v7.py
- exploration_lab/ (directory)

#### Strategy Experiments (`strategy_experiments/`)
- berserker_strategy.py
- triad_system.py
- doppelganger_triad.py
- strategies_v7.py

#### Optimization Tools (`optimization_tools/`)
- backtest_optimizer.py
- hpo_optimizer.py
- integrated_backtester.py

#### Superseded Utilities (`superseded_utilities/`)
- high_performance_engine.py
- network_resilience.py
- health_monitor.py
- health_score.py

#### Legacy Scripts (`legacy_scripts/`)
- train.py
- run_hpo.py
- train_dqn.py
- hunger_games_mvp.py
- kientra_alpha_pipeline.py
- detect_silent_failures.py
- benchmark_performance.py
- audit_data_coverage.py
- download/download_mt5_data.py
- download/parallel_data_prep.py
- download/extract_mt5_specs.py
- download/prepare_exploration_data.py
