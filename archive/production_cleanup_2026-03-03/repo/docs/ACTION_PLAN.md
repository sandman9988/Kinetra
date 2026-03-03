# Kinetra Action Plan - Critical Path to Empirical Testing

**Last Updated**: 2026-01-01  
**Status**: Integration → Testing → Empirical Discoveries → Theorems  
**Priority**: CRITICAL

---

## 🎯 Mission Statement

**Get to the MotherLoad empirical testing ASAP to produce empirically backed hypotheses & theorems**

All architectural decisions (algorithm choice, specialization, market focus) will be informed by empirical testing results, not assumptions.

---

## 📋 Critical Path (3 Phases)

### Phase 1: Integration Wiring (Week 1-2) 🔥 URGENT

**Goal**: Wire up all components so testing framework can run end-to-end

**Tasks** (in priority order):

| Priority | Task | Effort | Deliverable | Status |
|----------|------|--------|-------------|--------|
| **P0** | DSP-SuperPot Integration | 2-3 days | `kinetra/superpot_dsp.py` | ❌ TODO |
| **P1** | Testing Framework ↔ RL Agents | 2-3 days | `kinetra/agent_factory.py` | ❌ TODO |
| **P2** | Physics ↔ Test Environments | 1-2 days | `kinetra/trading_env.py` unified | ⚠️ Partial |
| **P3** | Discovery Methods | 3-5 days | Chaos, hidden dims implemented | ❌ TODO |
| **P4** | Results Analytics | 1-2 days | `kinetra/results_analyzer.py` | ❌ TODO |
| **P5** | Unified Training CLI | 1 day | `scripts/train.py` | ❌ TODO |

**Total Estimate**: 11-16 days (2-3 weeks)

**Acceptance Criteria**:
```bash
# Must be able to run:
python scripts/testing/unified_test_framework.py --quick
# → Completes without errors
# → Generates test_results/*.json
# → Uses DSP-detected cycles (not fixed periods)
# → Asymmetric features (up/down separate)
```

### Phase 2: Integration Testing (Week 3) 🔥 CRITICAL

**Goal**: Validate all integrations work correctly

**Tests to Run**:

1. **P0 Validation**: DSP-driven features
   ```bash
   python scripts/analysis/superpot_dsp_driven.py --episodes 10
   # Verify: No fixed periods in output, dominant_scale used
   ```

2. **P1 Validation**: RL agent training
   ```bash
   python scripts/train.py --agent ppo --strategy universal --episodes 10
   # Verify: Agent trains, metrics tracked
   ```

3. **End-to-End Integration**:
   ```bash
   python scripts/testing/unified_test_framework.py --compare control physics rl
   # Verify: All 3 suites run, statistical comparison generated
   ```

4. **Statistical Validation**:
   - Check p-values < 0.01 for significance
   - Check effect sizes (Cohen's d > 0.5)
   - Verify Bonferroni/FDR corrections applied

**Acceptance Criteria**:
- ✅ All test suites run without errors
- ✅ DSP features show adaptive cycles (not 5, 10, 20)
- ✅ Asymmetric measurements (up ≠ down)
- ✅ Statistical validation applied
- ✅ Results saved to JSON with full metadata

### Phase 3: MotherLoad Empirical Testing (Week 4+) 🎯 GOAL

**Goal**: Discover alpha sources empirically, produce theorems

**The MotherLoad Test Campaign**:

```bash
# 1. Complete SuperPot Exploration (All measurements, all instruments)
python scripts/analysis/superpot_dsp_driven.py \
    --episodes 500 \
    --all-instruments \
    --all-measurements \
    --prune-adaptive

# Expected Output:
# → Universal features (survive ALL asset classes)
# → Class-specific features (crypto vs forex vs metals)
# → Instrument-specific features
# → Pruned features (empirically useless)
```

```bash
# 2. Algorithm Comparison (Answer Q1: Best algorithm?)
python scripts/testing/unified_test_framework.py \
    --compare ppo td3 sac quant \
    --instruments crypto forex metals \
    --episodes 200 \
    --statistical-validation

# Expected Output:
# → Sharpe ratio by algorithm (with confidence intervals)
# → Omega ratio comparison
# → Statistical significance (p-values)
# → Winner: Algorithm X with p < 0.01
```

```bash
# 3. Specialization Comparison (Answer Q2: Best specialization?)
python scripts/training/explore_specialization.py \
    --episodes 200 \
    --all-strategies

# Expected Output:
# → Universal vs Asset-class vs Regime vs Timeframe
# → Edge robustness (consistency metric)
# → Winner: Strategy Y with lowest variance
```

```bash
# 4. Physics vs Chaos vs Traditional (Answer Q4: Alpha source?)
python scripts/testing/unified_test_framework.py \
    --compare control physics chaos hidden \
    --extreme \
    --statistical-validation

# Expected Output:
# → Incremental alpha vs control group
# → Which features survive pruning
# → Statistical significance
# → Winner: Approach Z with effect size > 0.8
```

**Empirical Discoveries Expected**:

| Research Question | Test Suite | Output | Theorem Candidate |
|-------------------|------------|--------|-------------------|
| Q1: Best algorithm? | `--compare ppo td3 sac` | Winner algorithm | "Algorithm X optimal for regime Y" |
| Q2: Specialization? | `explore_specialization.py` | Optimal strategy | "Asset-class specialization > universal" |
| Q3: Market focus? | Instrument filtering | Alpha by instrument | "Crypto pairs {BTC, ETH} show alpha" |
| Q4: Alpha source? | `--compare control physics chaos` | Feature survival | "Physics features > traditional TA" |
| Q5: Triad needed? | `--compare triad rl` | Performance delta | "Triad adds +15% Sharpe vs single" |

**Theorem Production Process**:

1. **Empirical Result**: E.g., "PPO achieves Sharpe 1.8, TD3 achieves 1.3, p < 0.001"
2. **Hypothesis**: "PPO is superior for crypto markets in underdamped regime"
3. **Validation**: Test on out-of-sample data (Jul-Dec 2025)
4. **Theorem**: If validated (p < 0.01), formalize as theorem
5. **Documentation**: Add to `docs/EMPIRICAL_THEOREMS.md`

---

## 🚦 Current Status Dashboard

### Integration Progress

```
P0: DSP-SuperPot     [░░░░░░░░░░] 0%  - Not started
P1: Testing ↔ RL     [░░░░░░░░░░] 0%  - Not started
P2: Physics ↔ Env    [████░░░░░░] 40% - Partially wired
P3: Discovery        [░░░░░░░░░░] 0%  - Stubs only
P4: Analytics        [░░░░░░░░░░] 0%  - Not started
P5: Unified CLI      [░░░░░░░░░░] 0%  - Not started

Overall Progress: [█░░░░░░░░░] 10%
```

### Blockers

| ID | Blocker | Impact | Solution |
|----|---------|--------|----------|
| B1 | Fixed periods in superpot | Philosophy violation | P0: DSP integration |
| B2 | RL agents not connected | Can't run tests | P1: AgentFactory |
| B3 | No results analysis | Can't interpret outcomes | P4: ResultsAnalyzer |

**No technical blockers** - All dependencies exist, just need wiring.

---

## 📊 Success Metrics

### Phase 1 Success (Integration Complete)

- ✅ All 18 test suites executable via CLI
- ✅ DSP-driven features (no fixed periods)
- ✅ Asymmetric measurements (up/down separate)
- ✅ Statistical validation pipeline working
- ✅ Results saved in standardized JSON format

### Phase 2 Success (Integration Tested)

- ✅ End-to-end test passes (control vs physics vs RL)
- ✅ Statistical significance calculated (p-values)
- ✅ Effect sizes computed (Cohen's d)
- ✅ Comparison plots generated
- ✅ No errors in 100+ episode runs

### Phase 3 Success (Empirical Theorems Produced)

- ✅ At least 3 empirically backed theorems with p < 0.01
- ✅ Universal features identified (survive all asset classes)
- ✅ Algorithm choice justified empirically
- ✅ Specialization strategy proven statistically
- ✅ Alpha sources ranked by incremental improvement
- ✅ Production architecture designed from empirical results

**Target Theorems** (Examples):

1. **Theorem 1 (Algorithm)**: "PPO with adaptive reward shaping achieves Omega > 2.7 on crypto markets with statistical significance p < 0.001"

2. **Theorem 2 (Specialization)**: "Asset-class specialization reduces variance by 35% vs universal agents (p < 0.01, effect size = 0.82)"

3. **Theorem 3 (Alpha Source)**: "Physics-based features (energy, damping, entropy) contribute 60% of incremental alpha vs traditional TA (p < 0.001)"

4. **Theorem 4 (Cycle Detection)**: "DSP-detected dominant_scale outperforms fixed 20-period lookback by 28% in Sharpe ratio across all regimes (p < 0.001)"

5. **Theorem 5 (Asymmetry)**: "Asymmetric feature extraction (up/down separate) yields 15% higher edge robustness than symmetric (p < 0.01)"

---

## 🔧 Immediate Next Actions (This Week)

### Day 1-2: P0 - DSP Integration

**Action**: Create `kinetra/superpot_dsp.py`

```bash
# 1. Copy structure from superpot_explorer.py
cp scripts/analysis/superpot_explorer.py kinetra/superpot_dsp.py

# 2. Replace all fixed periods with DSP-detected cycles
# - Remove: lookback = 20
# - Add: dominant_scale = wavelet_features['dominant_scale']

# 3. Make asymmetric (up/down separate)
# - Remove: abs(returns)
# - Add: up_returns, down_returns (separate)

# 4. Test
python -c "from kinetra.superpot_dsp import DSPSuperPotExtractor; print('OK')"
```

**Validation**:
```bash
grep -n "5\|10\|20" kinetra/superpot_dsp.py
# Should return ZERO fixed period references
```

### Day 3-4: P1 - Agent Factory

**Action**: Create `kinetra/agent_factory.py`

See `docs/INTEGRATION_GUIDE.md` Task 1 for complete code example.

**Validation**:
```bash
python -c "
from kinetra.agent_factory import AgentFactory
agent = AgentFactory.create('ppo', state_dim=64, action_dim=4, config={})
print('Agent created:', type(agent))
"
```

### Day 5: P2 - Unified Environment

**Action**: Create `kinetra/trading_env.py`

Consolidate `rl_exploration_framework.py` TradingEnv with physics integration.

**Validation**:
```bash
python -c "
from kinetra.trading_env import UnifiedTradingEnv
env = UnifiedTradingEnv(instruments=[], use_physics=True)
print('Environment ready')
"
```

### Day 6-7: Integration Testing

**Action**: Run quick tests on all integrations

```bash
# Test P0
python scripts/analysis/superpot_dsp_driven.py --episodes 5 --quick

# Test P1
python scripts/train.py --agent ppo --episodes 5

# Test P2
python scripts/testing/test_framework_integration.py
```

---

## 📚 Documentation References

| Document | Purpose | When to Use |
|----------|---------|-------------|
| `ARCHITECTURE_COMPLETE.md` | System design overview | Understanding big picture |
| `INTEGRATION_GUIDE.md` | Step-by-step wiring tasks | During implementation |
| `ARCHITECTURE_QUICK_REF.md` | One-page reference | Quick lookups |
| `ACTION_PLAN.md` (this doc) | Critical path roadmap | Daily standup, tracking |
| `AI_AGENT_INSTRUCTIONS.md` | First-principles philosophy | Design decisions |
| `TESTING_FRAMEWORK.md` | Testing methodology | Running tests |

---

## 🎯 Definition of Done

**Phase 1 Complete** when:
- All P0-P5 tasks marked ✅
- `python scripts/testing/unified_test_framework.py --quick` succeeds
- No fixed periods (5, 10, 20) in active code paths
- All features asymmetric (up/down separate)

**Phase 2 Complete** when:
- All integration tests pass
- Statistical validation working (p-values calculated)
- Comparison plots generated automatically
- Documentation updated with any discovered issues

**Phase 3 Complete** when:
- At least 3 theorems produced with p < 0.01
- `docs/EMPIRICAL_THEOREMS.md` created with formal statements
- Production architecture designed from empirical results
- Research questions (Q1-Q5) answered definitively

---

## 🚀 Let's Get to Work!

**Current Focus**: Start P0 (DSP integration) immediately

**Expected Timeline**:
- Week 1-2: Integration wiring
- Week 3: Integration testing
- Week 4+: MotherLoad empirical testing → Theorems

**Success Indicator**: When we can say with statistical confidence (p < 0.01):
> "Algorithm X using specialization strategy Y on instruments Z achieves Omega > 2.7 through alpha source W"

That's an empirically backed theorem, not an assumption.

---

**END OF ACTION PLAN**

*This is a living document - update status as tasks complete*
