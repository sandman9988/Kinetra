# Exhaustive Testing Framework: Action Plan & Roadmap

**Status**: Production-Ready with Enhancement Opportunities  
**Version**: 1.0  
**Last Updated**: 2025-01-03  
**Maintainer**: Kinetra Team

---

## Executive Summary

The Exhaustive Testing Framework is **operational and validated** with:

✅ **Core Infrastructure Complete**
- AgentFactory with 6 agents (PPO, DQN, Linear Q, Incumbent, Competitor, Researcher)
- Unified agent interface via AgentAdapter
- Real-data testing with MetaAPI/CSV fallback
- CI/CD integration with fast (PR) and full (nightly) modes
- Test orchestration script with parallel execution
- Interactive dashboard with static HTML export
- GPU testing scaffolding
- Comprehensive documentation

✅ **Validation Status**
- Agent factory self-test: **PASSED** (6/6 agents instantiate)
- Unit tests: **PASSING** (CI mode ~3 min, full mode available)
- Integration tests: **PASSING**
- CI pipeline: **OPERATIONAL** (GitHub Actions configured)

🎯 **Next Steps**: Enhance functionality, optimize performance, expand coverage

---

## Priority Matrix

### 🔴 High Priority (Do Now - Next 1-2 Weeks)

#### 1. Fix Dashboard Static Export in CI
**Status**: Code exists but not integrated  
**Impact**: High - Enables visual artifact tracking  
**Effort**: Low (2-4 hours)

**Problem**: Dashboard can generate static HTML, but CI doesn't upload it yet.

**Solution**:
```yaml
# Add to .github/workflows/ci.yml exhaustive-tests job
- name: Generate dashboard report
  if: always()
  run: |
    python -c "
    from kinetra.test_dashboard import TestDashboard
    dashboard = TestDashboard()
    dashboard.generate_static_report('test_report.html')
    "

- name: Upload dashboard artifact
  uses: actions/upload-artifact@v3
  if: always()
  with:
    name: test-dashboard
    path: test_report.html
    retention-days: 30
```

**Files to modify**:
- `.github/workflows/ci.yml`
- `scripts/run_exhaustive_tests.py` (optional: add `--generate-dashboard` flag)

**Acceptance Criteria**:
- [ ] CI uploads `test_report.html` as artifact
- [ ] Report viewable in GitHub Actions UI
- [ ] Includes heatmaps, agent comparisons, metrics distributions

---

#### 2. Document GPU Installation Instructions
**Status**: Code exists, docs incomplete  
**Impact**: Medium - Enables GPU acceleration  
**Effort**: Low (1-2 hours)

**Problem**: Users don't know how to install CUDA/ROCm torch wheels.

**Solution**: Add section to `docs/EXHAUSTIVE_TESTING_GUIDE.md`

```markdown
### GPU Acceleration Setup

#### NVIDIA GPUs (CUDA)
```bash
# Uninstall CPU-only torch
pip uninstall torch torchvision

# Install CUDA version (example for CUDA 12.1)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

#### AMD GPUs (ROCm)
```bash
# Uninstall CPU-only torch
pip uninstall torch torchvision

# Install ROCm version (Linux only, example for ROCm 6.0)
pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.0
```

#### Verify GPU
```bash
python -c "import torch; print(f'GPU available: {torch.cuda.is_available()}')"
python -m kinetra.gpu_testing --benchmark
```
```

**Files to modify**:
- `docs/EXHAUSTIVE_TESTING_GUIDE.md`
- `README.md` (add GPU setup quick reference)

**Acceptance Criteria**:
- [ ] Clear NVIDIA/AMD installation steps
- [ ] Verification commands documented
- [ ] Benchmark usage explained

---

#### 3. Expand Real-Data Coverage
**Status**: Infrastructure ready, data limited  
**Impact**: High - Improves test quality  
**Effort**: Medium (4-8 hours)

**Problem**: Some asset/timeframe combos may lack real data.

**Solution**:
1. **Audit current data coverage**:
   ```bash
   python scripts/audit_data_coverage.py --report data/coverage_report.csv
   ```

2. **Fetch missing data**:
   ```bash
   python scripts/fetch_missing_data.py --source metaapi --fill-gaps
   ```

3. **Add CSV fallbacks** for commonly tested combos:
   - BTCUSD: M15, M30, H1, H4, D1
   - EURUSD: H1, H4, D1
   - US30: H1, H4
   - XAUUSD: H1, H4

**Files to create**:
- `scripts/audit_data_coverage.py` - Check which combos have data
- `scripts/fetch_missing_data.py` - Automated data fetching
- `data/master_standardized/{instrument}_{timeframe}.csv` - Fallback CSVs

**Acceptance Criteria**:
- [ ] Coverage report shows >80% data availability
- [ ] Top 10 asset/timeframe combos have CSV fallbacks
- [ ] CI can run without MetaAPI dependency

---

### 🟡 Medium Priority (Next 2-4 Weeks)

#### 4. Hyperparameter Optimization Integration
**Status**: Not started  
**Impact**: High - Improves agent performance  
**Effort**: Medium-High (12-20 hours)

**Objective**: Automate HPO for all agents using Optuna.

**Design**:
```python
# kinetra/hpo_optimizer.py

from typing import Dict, Any
import optuna
from kinetra.agent_factory import AgentFactory

class HPOOptimizer:
    """Hyperparameter optimization for Kinetra agents."""
    
    def optimize_agent(
        self, 
        agent_type: str,
        n_trials: int = 100,
        metric: str = "omega_ratio"
    ) -> Dict[str, Any]:
        """Find optimal hyperparameters for agent type."""
        
        def objective(trial):
            # Suggest hyperparameters
            config = {
                "learning_rate": trial.suggest_float("lr", 1e-5, 1e-2, log=True),
                "gamma": trial.suggest_float("gamma", 0.9, 0.999),
                "clip_range": trial.suggest_float("clip_range", 0.1, 0.3),
                # ... more params
            }
            
            # Create agent
            agent = AgentFactory.create(agent_type, **config)
            
            # Run backtest
            results = run_backtest(agent)
            
            return results[metric]
        
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials)
        
        return study.best_params
```

**Usage**:
```bash
# Optimize single agent
python -m kinetra.hpo_optimizer --agent ppo --trials 100

# Optimize all agents
python scripts/optimize_all_agents.py --trials 50 --parallel 3
```

**Files to create**:
- `kinetra/hpo_optimizer.py` - Core HPO logic
- `scripts/optimize_all_agents.py` - Batch optimization
- `configs/hpo_search_spaces.yaml` - Per-agent search spaces
- `data/hpo_results/{agent}_best_params.json` - Export best configs

**Acceptance Criteria**:
- [ ] HPO runs for all 6 agents
- [ ] Best params exported to JSON
- [ ] CI job runs HPO weekly (optional)
- [ ] Performance improvement >10% on key metrics

---

#### 5. Live Dashboard Integration
**Status**: Dashboard exists, needs real-time wiring  
**Impact**: Medium - Better UX during long runs  
**Effort**: Medium (8-12 hours)

**Objective**: Show test progress in real-time during exhaustive runs.

**Design**:
```python
# In test harness, write intermediate results
def test_combination(agent, asset, timeframe, regime):
    result = run_test(...)
    
    # Append to live results CSV
    append_result("data/live_results.csv", result)
    
    # Trigger dashboard refresh (via file watch or websocket)
    signal_dashboard_update()
```

**Implementation Options**:

**Option A: File Watching**
- Dashboard watches `data/live_results.csv`
- Auto-reloads every 5 seconds
- Simple, no infrastructure needed

**Option B: WebSocket**
- Test runner pushes updates via WebSocket
- Dashboard subscribes to updates
- Real-time, but more complex

**Recommendation**: Start with Option A (file watching).

**Files to modify**:
- `kinetra/test_dashboard.py` - Add file watcher callback
- `tests/test_exhaustive_combinations.py` - Write intermediate results
- `scripts/run_exhaustive_tests.py` - Add `--live-dashboard` flag

**Acceptance Criteria**:
- [ ] Dashboard updates during test runs
- [ ] Progress bar shows completion %
- [ ] Heatmap fills in as results arrive
- [ ] No performance degradation from I/O

---

#### 6. Agent Coverage Expansion
**Status**: 6 agents working, more could be added  
**Impact**: Medium - More comprehensive testing  
**Effort**: Medium (10-16 hours per agent family)

**Objective**: Add standard RL baselines for comparison.

**Candidates**:
1. **A3C** (Asynchronous Advantage Actor-Critic)
2. **SAC** (Soft Actor-Critic)
3. **TD3** (Twin Delayed DDPG)

**Option 1: Use Stable-Baselines3**
```python
# kinetra/sb3_agents.py
from stable_baselines3 import A3C, SAC, TD3
from kinetra.agent_factory import AgentFactory

class SB3Adapter:
    """Adapter for Stable-Baselines3 agents."""
    
    def __init__(self, model_class, **kwargs):
        self.model = model_class(**kwargs)
    
    def select_action(self, state, explore=True):
        action, _ = self.model.predict(state, deterministic=not explore)
        return action
    
    def update(self, state, action, reward, next_state, done):
        # SB3 handles learning internally via .learn()
        pass

# Register with factory
AgentFactory.register_agent(
    "a3c",
    lambda **kw: SB3Adapter(A3C, **kw),
    "A3C (Asynchronous Actor-Critic) - Baseline"
)
```

**Option 2: Custom Implementations**
- More control, fits Kinetra philosophy
- More effort (3-5 days per agent)

**Recommendation**: Option 1 for quick comparison, Option 2 if outperformance needed.

**Files to create**:
- `kinetra/sb3_agents.py` - SB3 adapter
- `tests/test_sb3_agents.py` - Unit tests
- `configs/sb3_defaults.yaml` - Default hyperparameters

**Acceptance Criteria**:
- [ ] A3C, SAC, TD3 registered in AgentFactory
- [ ] Pass same tests as existing agents
- [ ] Performance benchmarked vs PPO/DQN baseline

---

### 🟢 Low Priority / Research (2-3 Months)

#### 7. Meta-Learning Across Agents
**Status**: Research direction  
**Impact**: High (if successful) - Adaptive agent selection  
**Effort**: High (20-40 hours)

**Concept**: Learn which agent performs best in which regime.

**Approach**:
```python
# Meta-agent that selects from agent pool
class MetaAgent:
    """Ensemble that learns agent selection policy."""
    
    def __init__(self, agents: List[Agent]):
        self.agents = agents
        self.selector = train_meta_policy(agents)
    
    def select_action(self, state):
        # Meta-policy chooses agent
        agent_idx = self.selector.predict(state)
        agent = self.agents[agent_idx]
        
        # Chosen agent makes decision
        return agent.select_action(state)
```

**Research Questions**:
- Can we learn regime → agent mapping?
- Does ensemble beat single best agent?
- How to handle agent switching costs?

**Milestones**:
1. Implement naive ensemble (equal weights)
2. Add learned weights (e.g., contextual bandits)
3. Add meta-RL policy (PPO over agent selection)
4. Validate on out-of-sample data

**Files to create**:
- `kinetra/meta_agent.py`
- `kinetra/meta_learning.py`
- `experiments/meta_learning/`

---

#### 8. GPU Batch Training Optimization
**Status**: Scaffolding exists, needs implementation  
**Impact**: Medium - 2-5x speedup on large runs  
**Effort**: Medium-High (12-24 hours)

**Objective**: Batch-train agents on GPU for faster exhaustive tests.

**Current State**: `kinetra/gpu_testing.py` has stubs.

**Implementation**:
```python
# Add to agent classes
class KinetraAgent:
    def batch_update(self, batch: Dict[str, torch.Tensor]):
        """GPU-accelerated batch update."""
        states = batch["states"]      # (N, state_dim)
        actions = batch["actions"]    # (N,)
        rewards = batch["rewards"]    # (N,)
        
        # Vectorized loss computation
        loss = self.compute_loss(states, actions, rewards)
        
        # Single backward pass
        loss.backward()
        self.optimizer.step()
```

**Usage**:
```python
from kinetra.gpu_testing import GPUBatchProcessor

processor = GPUBatchProcessor()
results = processor.batch_train_agents(
    agents=[agent1, agent2, agent3],
    data=training_data,
    batch_size=512
)
```

**Challenges**:
- Not all agents support batching (Linear Q doesn't need it)
- Memory constraints on large batches
- Synchronization between agents

**Files to modify**:
- `kinetra/rl_agent.py` - Add `batch_update()` to PPO
- `kinetra/rl_neural_agent.py` - Add `batch_update()` to DQN
- `kinetra/gpu_testing.py` - Implement batch processor

**Acceptance Criteria**:
- [ ] 2x+ speedup on GPU vs CPU for PPO/DQN
- [ ] Memory-efficient batching (auto-adjust batch size)
- [ ] Graceful fallback to CPU if GPU OOM

---

#### 9. Historical Test Run Database
**Status**: Not started  
**Impact**: Low-Medium - Better tracking over time  
**Effort**: Medium (8-16 hours)

**Objective**: Store all test runs in SQLite DB for historical analysis.

**Schema**:
```sql
CREATE TABLE test_runs (
    run_id TEXT PRIMARY KEY,
    timestamp DATETIME,
    mode TEXT,  -- 'ci' or 'full'
    git_commit TEXT,
    total_tests INTEGER,
    passed INTEGER,
    failed INTEGER,
    elapsed_seconds REAL
);

CREATE TABLE test_results (
    result_id INTEGER PRIMARY KEY,
    run_id TEXT,
    agent_type TEXT,
    asset TEXT,
    timeframe TEXT,
    regime TEXT,
    test_type TEXT,
    omega_ratio REAL,
    z_factor REAL,
    chs REAL,
    p_value REAL,
    FOREIGN KEY (run_id) REFERENCES test_runs(run_id)
);
```

**Usage**:
```python
from kinetra.test_database import TestDatabase

db = TestDatabase("test_history.db")
db.save_run(run_id, results)

# Query historical performance
trend = db.get_metric_trend("omega_ratio", agent="ppo", days=30)
```

**Benefits**:
- Track performance over time
- Detect regressions
- Identify best configurations

**Files to create**:
- `kinetra/test_database.py`
- `scripts/analyze_test_history.py`
- `test_history.db` (ignored in .gitignore)

---

## Operational Checklist

### Daily Development
```bash
# Verify agent factory
python -m kinetra.agent_factory

# Run fast tests
KINETRA_CI_MODE=1 pytest tests/test_exhaustive_combinations.py -v

# Check specific agent
pytest tests/test_exhaustive_combinations.py::test_all_agents -v
```

### Before PR Merge
```bash
# Run CI-mode exhaustive tests
python scripts/run_exhaustive_tests.py --ci-mode --all

# Check code quality
make lint
make format

# Verify no secrets
grep -r "METAAPI_TOKEN\s*=" kinetra/ tests/
```

### Weekly (Nightly CI)
```bash
# Full exhaustive tests (automated via GitHub Actions)
python scripts/run_exhaustive_tests.py --full --all --parallel 4

# HPO (after implemented)
python scripts/optimize_all_agents.py --trials 50
```

### Before Release
```bash
# Full exhaustive tests with coverage
python scripts/run_exhaustive_tests.py --full --all --coverage

# Generate final report
python -m kinetra.test_dashboard --static --output release_report.html

# Tag empirical theorems
git add docs/EMPIRICAL_THEOREMS.md
git commit -m "Update empirical theorems for v1.X.X"
```

---

## Success Metrics

### Short-Term (1 Month)
- [ ] Dashboard integrated into CI artifacts
- [ ] GPU instructions documented
- [ ] >80% real-data coverage for top 10 combos
- [ ] CI run time <10 min (fast mode), <2 hrs (full mode)

### Medium-Term (3 Months)
- [ ] HPO implemented for all agents
- [ ] Live dashboard during test runs
- [ ] 3+ new agent types (A3C, SAC, TD3)
- [ ] Performance improvement >15% vs baseline

### Long-Term (6 Months)
- [ ] Meta-learning agent operational
- [ ] GPU batch training 3x+ faster
- [ ] Test history database with 100+ runs
- [ ] Automated regression detection

---

## Risk Mitigation

### Risk: MetaAPI Rate Limits
**Mitigation**: CSV fallbacks + cached data + retry logic

### Risk: GPU Memory Exhaustion
**Mitigation**: Auto-adjust batch size + CPU fallback

### Risk: Test Flakiness
**Mitigation**: Statistical thresholds + retry failed tests + seed control

### Risk: CI Timeouts
**Mitigation**: Parallel execution + smart subset selection + incremental results

---

## Dependencies & Prerequisites

### Installed
- ✅ pytest, pytest-xdist, pytest-cov
- ✅ plotly, dash, dash-bootstrap-components
- ✅ numpy, pandas, torch
- ✅ AgentFactory with 6 agents

### To Install (Optional)
- ⚠️ optuna (for HPO)
- ⚠️ stable-baselines3 (for A3C/SAC/TD3)
- ⚠️ kaleido (for static plotly export - already in pyproject.toml)
- ⚠️ CUDA/ROCm torch (for GPU)

---

## Contributing Guidelines

### Adding New Tests
1. Add to `tests/test_exhaustive_combinations.py`
2. Ensure CI-mode subset support
3. Include statistical validation (p < 0.01)
4. Document in `docs/EXHAUSTIVE_TESTING_GUIDE.md`

### Adding New Agents
1. Implement agent class
2. Register in `kinetra/agent_factory.py`
3. Add unit tests
4. Run exhaustive tests in CI mode
5. Document in README

### Modifying Test Harness
1. Preserve backward compatibility
2. Update both CI and full modes
3. Test with `--ci-mode` and `--full`
4. Update documentation

---

## References

- **Main Docs**: `docs/EXHAUSTIVE_TESTING_GUIDE.md`
- **Quick Ref**: `EXHAUSTIVE_TESTING_QUICKREF.md`
- **Patch Summary**: `EXHAUSTIVE_TESTING_PATCH_SUMMARY.md`
- **CI Config**: `.github/workflows/ci.yml`
- **Orchestrator**: `scripts/run_exhaustive_tests.py`
- **Dashboard**: `kinetra/test_dashboard.py`
- **Agent Factory**: `kinetra/agent_factory.py`

---

## Contact & Support

For questions or issues:
1. Check documentation in `docs/`
2. Run self-tests: `python -m kinetra.agent_factory`
3. Review recent test results in `data/`
4. Check CI logs in GitHub Actions

---

**Last Updated**: 2025-01-03  
**Next Review**: 2025-02-01  
**Status**: ✅ Production-Ready, 🚀 Enhancement-Ready