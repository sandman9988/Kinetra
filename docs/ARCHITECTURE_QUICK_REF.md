# Kinetra Architecture - Quick Reference
## Visual Summary & Entry Points

**Last Updated**: 2026-01-01  
**Purpose**: One-page visual reference for navigating Kinetra architecture

---

## 🎯 Current State Summary

```
Research Tooling Setup Phase
├─ Data Pipeline: ✅ COMPLETE (87 instruments loaded)
├─ Testing Framework: 🔄 80% BUILT (needs integration plumbing)
├─ Agent System: ⚠️ DESIGNED (awaiting test results for algorithm choice)
└─ Production Deployment: ❌ NOT STARTED (Q2-Q3 2026)
```

---

## 🗺️ System Map (3 Layers)

```
┌────────────────────────────────────────────────────────────────────┐
│                    LAYER 1: DATA PIPELINE (✅ DONE)                 │
├────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  MT5 Terminal → extract_mt5_specs.py → instrument_specs.json       │
│       ↓                                        ↓                     │
│  data/master/*.csv                    (Real broker data)            │
│  (87 datasets)                                                      │
│       ↓                                                             │
│  MultiInstrumentLoader → UnifiedDataLoader → DataPackage           │
│                                                                      │
│  Available Markets:                                                 │
│  • Forex: AUDJPY, AUDUSD, EURJPY, GBPJPY, GBPUSD                   │
│  • Crypto: BTCUSD, BTCJPY, ETHEUR, XRPJPY                          │
│  • Indices: DJ30ft, NAS100, Nikkei225, GER40, US2000, SA40, EU50   │
│  • Metals: XAUUSD, XAUAUD, XAGUSD, XPTUSD                          │
│  • Energy: UKOUSD                                                   │
│  • Commodities: COPPER                                              │
│                                                                      │
│  Timeframes: M15, M30, H1, H4                                       │
└────────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────────┐
│              LAYER 2: TESTING FRAMEWORK (🔄 INTEGRATION)            │
├────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │           CORE SUITES (6)          │   DISCOVERY SUITES (12) │  │
│  ├──────────────────────────────────────────────────────────────┤  │
│  │ • control (MA/RSI baseline)        │ • hidden (PCA/ICA)      │  │
│  │ • physics (energy/damping/entropy) │ • meta (MAML)           │  │
│  │ • rl (PPO/SAC/A2C/TD3)            │ • chaos (Lyapunov)      │  │
│  │ • specialization (asset/tf/regime) │ • quantum (superpose)   │  │
│  │ • stacking (ensemble)              │ • info_theory (entropy) │  │
│  │ • triad (Inc/Comp/Res)            │ • combinatorial (combos)│  │
│  │                                    │ • adversarial (GAN)     │  │
│  │                                    │ • cross_regime          │  │
│  │                                    │ • cross_asset           │  │
│  │                                    │ • mtf (multi-timeframe) │  │
│  │                                    │ • emergent (evolution)  │  │
│  │                                    │ • deep_ensemble (all)   │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                              ↓                                       │
│                  Statistical Validation                             │
│                  (p < 0.01, Bonferroni/FDR)                         │
│                              ↓                                       │
│                      test_results/*.json                            │
│                              ↓                                       │
│                   ANSWERS TO RESEARCH QUESTIONS:                    │
│         • Best algorithm? (PPO vs TD3 vs SAC vs Quant)              │
│         • Specialization? (Asset vs Regime vs Universal)            │
│         • Market focus? (Which crypto/forex pairs?)                 │
│         • Alpha source? (Physics vs Chaos vs Hidden?)               │
└────────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────────┐
│         LAYER 3: AGENT SYSTEM (⚠️ DESIGNED, NOT DEPLOYED)           │
├────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Based on test results, deploy:                                    │
│                                                                      │
│  Option A: TRIAD SYSTEM (if tests show it wins)                    │
│  ┌──────────────┬──────────────┬──────────────┐                   │
│  │  Incumbent   │  Competitor  │  Researcher  │                   │
│  │  (Live)      │  (Shadow A)  │  (Shadow B)  │                   │
│  │              │              │              │                   │
│  │  PPO/TD3/    │  Alternative │  Online      │                   │
│  │  Quant?      │  Algorithm?  │  Learning?   │                   │
│  │  [Frozen]    │  [Frozen]    │  [Training]  │                   │
│  └──────────────┴──────────────┴──────────────┘                   │
│                         ↓                                           │
│              Promotion Logic (Sharpe > 0.3)                         │
│                                                                      │
│  Option B: UNIVERSAL AGENT (if tests show triad unnecessary)       │
│  ┌────────────────────────────────────────────┐                   │
│  │  Single Agent (PPO/TD3/Quant - TBD)        │                   │
│  │  Trained on all instruments                │                   │
│  └────────────────────────────────────────────┘                   │
│                                                                      │
│  Option C: SPECIALISTS (if tests show specialization wins)         │
│  ┌─────────────┬─────────────┬─────────────┐                      │
│  │ Forex Agent │ Crypto Agent│ Metals Agent│ ... etc              │
│  └─────────────┴─────────────┴─────────────┘                      │
│                                                                      │
│  All routes through:                                                │
│         ↓                                                           │
│  Risk Management (Circuit Breakers, RoR, CHS)                      │
│         ↓                                                           │
│  Order Execution (MT5 Connector)                                   │
└────────────────────────────────────────────────────────────────────┘
```

---

## 🚀 Quick Start Commands

### Data Collection (✅ DONE)
```bash
# Already complete: 87 datasets in data/master/
ls data/master/  # See all available instruments
```

### Testing Framework (🔄 RUN THESE)
```bash
# Quick validation (10 min)
python scripts/testing/unified_test_framework.py --quick

# Specific suite
python scripts/testing/unified_test_framework.py --suite physics
python scripts/testing/unified_test_framework.py --suite chaos

# Compare approaches
python scripts/testing/unified_test_framework.py --compare control physics rl

# Full suite (hours)
python scripts/testing/unified_test_framework.py --full

# EXTREME mode (days - all 18 suites)
python scripts/testing/unified_test_framework.py --extreme
```

### Exploration Scripts
```bash
# Specialization comparison
python scripts/training/explore_specialization.py

# Universal vs specialist
python scripts/training/explore_universal.py

# Triad training
python scripts/training/train_triad.py
```

### Analysis
```bash
# View latest results
cat test_results/test_*.json | jq '.'

# Generate comparison plots
python scripts/testing/unified_test_framework.py --compare control physics rl chaos
open test_results/comparison.png
```

---

## 🔌 Integration Status

| Integration Point | Priority | Effort | Status | Blocker |
|-------------------|----------|--------|--------|---------|
| Testing → RL Agents | 🔥 P1 | 2-3 days | ❌ TODO | None |
| Discovery Methods → Tests | 🔥 P2 | 3-5 days | ❌ TODO | None |
| Physics → Environments | 🔥 P2 | 1-2 days | ⚠️ Partial | None |
| Results → Analytics | 🟡 P3 | 1-2 days | ❌ TODO | None |
| Training → Unified Interface | 🟢 P4 | 1 day | ❌ TODO | None |

**Total**: 8-13 days (1.5-2.5 weeks) to complete integration

---

## 📊 Key Metrics Dashboard

### Research Phase (Current)
```
┌────────────────────────────────────────────┐
│ Datasets Available      87 / 87    [100%] │
│ Data Quality           High        [✅]   │
│ Test Suites Designed   18 / 18    [100%] │
│ Test Suites Wired      6 / 18     [ 33%] │
│ Integration Complete   2 / 5      [ 40%] │
│ GPU Utilization        TBD        [---]  │
└────────────────────────────────────────────┘
```

### Alpha Discovery (Next - After Integration)
```
┌────────────────────────────────────────────┐
│ Target Omega Ratio     > 2.7              │
│ Target Sharpe Ratio    > 1.5              │
│ Target Win Rate        > 55%              │
│ Statistical Sig (p)    < 0.01             │
│ Effect Size (Cohen's d)> 0.5              │
└────────────────────────────────────────────┘
```

### Production (Future - Q2-Q3 2026)
```
┌────────────────────────────────────────────┐
│ Composite Health Score > 0.90             │
│ Energy Captured        > 65%              │
│ MFE Captured           > 60%              │
│ Max Drawdown           < 15%              │
└────────────────────────────────────────────┘
```

---

## 🗂️ File Navigation

### 📖 Documentation (Start Here)
| File | Purpose |
|------|---------|
| **docs/ARCHITECTURE_COMPLETE.md** | Full system design (24KB) |
| **docs/INTEGRATION_GUIDE.md** | Step-by-step integration tasks (29KB) |
| **docs/TESTING_FRAMEWORK.md** | Testing framework philosophy & usage |
| **README.md** | Project overview & quick start |
| **AI_AGENT_INSTRUCTIONS.md** | First-principles philosophy |

### 🔧 Core Framework
| File | Purpose |
|------|---------|
| `kinetra/testing_framework.py` | Main testing engine |
| `kinetra/data_loader.py` | Data loading & validation |
| `kinetra/physics_engine.py` | Physics state computation |
| `kinetra/backtest_engine.py` | Monte Carlo backtesting |

### 🤖 Agents
| File | Purpose |
|------|---------|
| `kinetra/rl_agent.py` | PPO agent |
| `kinetra/rl_neural_agent.py` | DQN agent |
| `kinetra/triad_system.py` | Incumbent/Competitor/Researcher |
| `kinetra/doppelganger_triad.py` | Shadow agent wrapper |
| `specialist_agents.py` | Asset class specialists |

### 🧪 Testing Scripts
| File | Purpose |
|------|---------|
| `scripts/testing/unified_test_framework.py` | Main test interface |
| `scripts/training/explore_specialization.py` | Specialization comparison |
| `scripts/training/train_triad.py` | Triad training |
| `rl_exploration_framework.py` | Multi-instrument RL env |

---

## ❓ Research Questions (To Be Answered)

### Q1: Best Algorithm?
**Options**: PPO vs TD3 vs SAC vs Quant strategies  
**Test Suite**: `--suite rl`  
**Metric**: Sharpe ratio, Omega ratio, statistical significance  
**Answer**: TBD (run tests)

### Q2: Specialization Strategy?
**Options**: Asset class vs Regime vs Timeframe vs Universal  
**Test Suite**: `--suite specialization`  
**Metric**: Edge robustness (consistency across instruments)  
**Answer**: TBD (run tests)

### Q3: Market Focus?
**Options**: Which crypto? Which forex? Which metals?  
**Test Suite**: `--compare` with asset filters  
**Metric**: Alpha per instrument (Omega > 2.7, p < 0.01)  
**Answer**: TBD (run tests)

### Q4: Alpha Source?
**Options**: Physics vs Chaos vs Hidden dimensions vs Traditional  
**Test Suite**: `--compare control physics chaos hidden`  
**Metric**: Incremental alpha vs control group  
**Answer**: TBD (run tests)

### Q5: Triad Necessity?
**Options**: Triad vs Universal agent  
**Test Suite**: `--compare triad rl`  
**Metric**: Drift detection value, promotion frequency, stability  
**Answer**: TBD (run tests)

---

## 🎯 Q1 2026 Roadmap

### Week 1-2: Integration Plumbing (CURRENT)
- [ ] Create AgentFactory.py
- [ ] Wire Testing Framework ↔ RL Agents
- [ ] Implement discovery methods (chaos, hidden, etc.)
- [ ] Create UnifiedTradingEnv
- [ ] Build ResultsAnalyzer

### Week 3-4: Testing Campaign
- [ ] Run all 18 test suites
- [ ] Validate statistical significance (p < 0.01)
- [ ] Generate comparison plots
- [ ] Document preliminary findings

### Week 5-6: Analysis & Decision
- [ ] Answer Q1: Best algorithm
- [ ] Answer Q2: Specialization strategy
- [ ] Answer Q3: Market focus
- [ ] Answer Q4: Alpha sources
- [ ] Answer Q5: Triad necessity

### Week 7-10: Production Architecture Design
- [ ] Design final system based on test results
- [ ] Implement winning strategies
- [ ] Build monitoring infrastructure
- [ ] Prepare for paper trading

### Week 11-12: Validation & Documentation
- [ ] Out-of-sample validation
- [ ] Risk management testing
- [ ] Complete documentation
- [ ] Prepare deployment plan

---

## 🚨 Critical Dependencies

```
┌─────────────────────────────────────────────────────┐
│ BLOCKER: Integration plumbing must complete before  │
│ meaningful testing can begin.                       │
│                                                      │
│ Priority 1 (Week 1):                                │
│   ✅ Data pipeline complete                         │
│   ❌ Testing Framework ↔ RL Agents (CRITICAL)       │
│                                                      │
│ Without P1 integration:                             │
│   • Cannot train RL agents via testing framework    │
│   • Cannot compare algorithms systematically        │
│   • Cannot answer research questions                │
│                                                      │
│ With P1 integration:                                │
│   ✅ Can run all 18 test suites                     │
│   ✅ Can discover alpha sources                     │
│   ✅ Can design production architecture             │
└─────────────────────────────────────────────────────┘
```

---

## 📞 Support & Resources

**Documentation**:
- Architecture: `docs/ARCHITECTURE_COMPLETE.md`
- Integration: `docs/INTEGRATION_GUIDE.md`
- Testing: `docs/TESTING_FRAMEWORK.md`
- Specialization: `scripts/README_SPECIALIZATION_EXPLORER.md`

**Key Scripts**:
- Main test interface: `scripts/testing/unified_test_framework.py`
- Master workflow: `scripts/master_workflow.py`
- Exploration framework: `rl_exploration_framework.py`

**Data Locations**:
- Raw data: `data/master/*.csv` (87 files)
- Instrument specs: `instrument_specs.json`
- Test results: `test_results/*.json`
- Backups: `data/backups/`

---

## 🎓 Design Philosophy Reminders

1. **First Principles, Zero Assumptions**
   - No magic numbers (all thresholds adaptive)
   - No linearity assumptions
   - Question everything

2. **Let the Data Speak**
   - Testing framework answers questions
   - Statistical validation (p < 0.01)
   - Effect sizes matter (Cohen's d > 0.5)

3. **Defense in Depth**
   - Unit tests → Integration → Monte Carlo → Statistical → Health monitoring

4. **Exploration vs Exploitation**
   - Research: Open, learn everything
   - Paper: Soft gates, warnings
   - Live: Hard gates, protection

---

**QUICK REF VERSION 1.0** | 2026-01-01 | Research Phase

*For detailed information, see ARCHITECTURE_COMPLETE.md and INTEGRATION_GUIDE.md*
