# Kinetra Exhaustive Testing - Quick Reference Card

**Version**: 1.0 | **Date**: 2024-01-03 | **Status**: ✅ Production Ready

---

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Verify AgentFactory (all 6 agents)
python -m kinetra.agent_factory

# Fast CI test (~3 minutes)
KINETRA_CI_MODE=1 pytest tests/test_exhaustive_combinations.py::test_all_agents -v

# Or use test runner
python scripts/run_exhaustive_tests.py --ci-mode --test-type unit
```

---

## 📊 Test Modes

| Mode | Duration | Combinations | Use Case |
|------|----------|--------------|----------|
| **CI Mode** | 5-10 min | 24 | PRs, daily dev |
| **Full Mode** | 1-2 hours | 600+ | Releases, validation |

```bash
# CI Mode (fast)
export KINETRA_CI_MODE=1
python scripts/run_exhaustive_tests.py --ci-mode --test-type unit

# Full Mode (exhaustive)
export KINETRA_CI_MODE=0
python scripts/run_exhaustive_tests.py --full --all --parallel 4
```

---

## 🤖 Agent Factory

### All 6 Agents

| Agent | Type | Description |
|-------|------|-------------|
| `ppo` | PPO | Proximal Policy Optimization |
| `dqn` | DQN | Deep Q-Network |
| `linear_q` | Q-Learning | Fast linear Q |
| `incumbent` | Triad | PPO-style (stable) |
| `competitor` | Triad | A2C-style (aggressive) |
| `researcher` | Triad | SAC-style (exploration) |

### Usage

```python
from kinetra.agent_factory import AgentFactory

# Create single agent
agent = AgentFactory.create('ppo', state_dim=43, action_dim=4)

# Create with unified interface (recommended)
wrapped = AgentFactory.create_wrapped('ppo', state_dim=43, action_dim=4)
action = wrapped.select_action(state, explore=True)
wrapped.update(state, action, reward, next_state, done)

# Create all agents
all_agents = AgentFactory.create_all(state_dim=43, action_dim=4, wrapped=True)
```

---

## 🧪 Test Types

| Type | Purpose | Duration (CI) | Duration (Full) |
|------|---------|---------------|-----------------|
| **Unit** | CHS, RoR validation | 3 min | 30 min |
| **Integration** | Agent training | 3 min | 30 min |
| **Monte Carlo** | Statistical significance | 8 min | 60-90 min |
| **Walk-Forward** | Regime stability | 5 min | 45 min |

```bash
# Run specific test type
python scripts/run_exhaustive_tests.py --ci-mode --test-type unit
python scripts/run_exhaustive_tests.py --ci-mode --test-type integration
python scripts/run_exhaustive_tests.py --ci-mode --test-type monte_carlo
python scripts/run_exhaustive_tests.py --ci-mode --test-type walk_forward

# Run all test types
python scripts/run_exhaustive_tests.py --ci-mode --all
```

---

## ⚙️ Test Runner Options

```bash
python scripts/run_exhaustive_tests.py [OPTIONS]

Required (choose one):
  --ci-mode              Fast subset testing (5-10 min)
  --full                 Full exhaustive testing (1-2 hours)

Optional:
  --test-type TYPE       unit|integration|monte_carlo|walk_forward
  --all                  Run all test types
  --parallel N           Use N parallel workers
  --coverage             Generate coverage report
  --stop-on-fail         Stop on first failure
  --skip-verify          Skip pre-flight checks
  --report-dir DIR       Custom report directory
```

---

## 📈 Performance Targets

### Production (Full Mode)

| Metric | Target | Purpose |
|--------|--------|---------|
| Omega Ratio | > 2.7 | Asymmetric returns |
| Z-Factor | > 2.5 | Statistical significance |
| % Energy Captured | > 65% | Physics efficiency |
| CHS | > 0.90 | System stability |
| p-value | < 0.01 | Statistical rigor |
| RoR | < 5% | Capital preservation |

### CI Mode (Relaxed)

| Metric | Target | Reason |
|--------|--------|--------|
| Omega Ratio | > 1.0 | Fewer MC runs |
| Z-Factor | > 1.0 | Smaller sample |
| CHS | > 0.5 | Quick validation |
| p-value | < 0.10 | Less stringent |
| RoR | < 0.5 | Acceptable for subset |

---

## 🔧 Common Commands

### Development

```bash
# Quick agent verification
python -m kinetra.agent_factory

# Fast test before commit
KINETRA_CI_MODE=1 pytest tests/test_exhaustive_combinations.py::test_all_agents -v

# Unit tests only (fastest)
python scripts/run_exhaustive_tests.py --ci-mode --test-type unit

# With coverage
python scripts/run_exhaustive_tests.py --ci-mode --coverage
```

### Pre-Release

```bash
# Full validation
python scripts/run_exhaustive_tests.py --full --all --parallel 4

# Trigger via commit message
git commit -m "Release v2.0 [exhaustive]"
```

### Troubleshooting

```bash
# Check dependencies
python scripts/run_exhaustive_tests.py --ci-mode --test-type unit

# Skip verification
python scripts/run_exhaustive_tests.py --ci-mode --skip-verify --test-type unit

# Stop on first failure
python scripts/run_exhaustive_tests.py --ci-mode --all --stop-on-fail
```

---

## 📁 Output Files

```
data/
├── exhaustive_results_unit_YYYYMMDD_HHMM.csv
├── exhaustive_results_integration_YYYYMMDD_HHMM.csv
├── exhaustive_results_monte_carlo_YYYYMMDD_HHMM.csv
└── exhaustive_results_walk_forward_YYYYMMDD_HHMM.csv

plots/
├── heatmap_unit_YYYYMMDD_HHMM.png
├── heatmap_integration_YYYYMMDD_HHMM.png
├── heatmap_monte_carlo_YYYYMMDD_HHMM.png
└── heatmap_walk_forward_YYYYMMDD_HHMM.png

logs/
└── exhaustive.log

test_results/
└── test_report_YYYYMMDD_HHMMSS.json

docs/
└── EMPIRICAL_THEOREMS.md (auto-updated)
```

---

## 🐛 Troubleshooting

### Import Errors

```bash
# Problem: ImportError
# Fix: Install dependencies
pip install -r requirements.txt
```

### Tests Too Slow

```bash
# Problem: Full mode takes too long
# Fix: Use CI mode for development
KINETRA_CI_MODE=1 pytest tests/test_exhaustive_combinations.py -v

# Or parallelize
python scripts/run_exhaustive_tests.py --full --parallel 4
```

### Agent Creation Failures

```bash
# Problem: ValueError: Unknown agent type
# Fix: Verify AgentFactory
python -m kinetra.agent_factory
```

### Data Loading Failures

```bash
# Problem: WARNING: Skipped BTCUSD H1
# Fix: Set MetaAPI token or use CSV cache
export METAAPI_TOKEN=your_token
# Or place CSV files in data/master_standardized/
```

---

## 📚 Documentation

- **Quick Ref**: `EXHAUSTIVE_TESTING_QUICKREF.md` (this file)
- **Full Guide**: `docs/EXHAUSTIVE_TESTING_GUIDE.md`
- **Patch Summary**: `EXHAUSTIVE_TESTING_PATCH_SUMMARY.md`
- **Copilot Instructions**: `.github/copilot-instructions.md`

---

## ✅ Pre-Commit Checklist

```bash
# 1. Verify AgentFactory
python -m kinetra.agent_factory

# 2. Run fast tests
KINETRA_CI_MODE=1 pytest tests/test_exhaustive_combinations.py::test_all_agents -v

# 3. Quick validation
python scripts/run_exhaustive_tests.py --ci-mode --test-type unit

# 4. (Optional) Full validation before major changes
python scripts/run_exhaustive_tests.py --full --all --parallel 4
```

---

## 🎯 CI/CD Pipeline

### GitHub Actions Jobs

| Job | Trigger | Duration | Purpose |
|-----|---------|----------|---------|
| **fast-tests** | Every PR | 20-30 min | Quick validation |
| **exhaustive-tests** | Nightly/[exhaustive] | 2-3 hours | Full validation |
| **agent-factory-tests** | Every PR | 5 min | Agent verification |
| **code-quality** | Every PR | 10 min | Linting, formatting |

### Environment Variables

```yaml
env:
  KINETRA_CI_MODE: 1  # Enable CI mode (subset testing)
  # KINETRA_CI_MODE: 0  # Disable CI mode (full testing)
```

---

## 🔬 Research Use Cases

### Compare All Agents

```python
from kinetra.agent_factory import AgentFactory
import numpy as np

# Create all agents
agents = AgentFactory.create_all(state_dim=43, action_dim=4, wrapped=True)

# Compare performance
results = {}
for agent_type, agent in agents.items():
    rewards = run_backtest(agent, data)
    results[agent_type] = {
        'mean': np.mean(rewards),
        'std': np.std(rewards),
        'sharpe': np.mean(rewards) / (np.std(rewards) + 1e-10)
    }

# Print comparison
for agent_type, stats in sorted(results.items(), 
                                 key=lambda x: x[1]['sharpe'], 
                                 reverse=True):
    print(f"{agent_type:12} - Sharpe: {stats['sharpe']:.2f}")
```

### Add New Agent

```python
from kinetra.agent_factory import AgentFactory

# 1. Implement agent class
class MyNewAgent:
    def __init__(self, state_dim: int, action_dim: int, **kwargs):
        self.state_dim = state_dim
        self.action_dim = action_dim
    
    def select_action(self, state, explore=True):
        return np.random.randint(0, self.action_dim)
    
    def update(self, state, action, reward, next_state, done):
        pass

# 2. Register with factory
AgentFactory.register_agent(
    name='my_agent',
    agent_class=MyNewAgent,
    description='My awesome agent',
    default_params={'state_dim': 43, 'action_dim': 4},
    param_mapping={'state_dim': 'state_dim', 'action_dim': 'action_dim'}
)

# 3. Test it
agent = AgentFactory.create('my_agent', state_dim=43, action_dim=4)
```

---

## 📞 Support

```bash
# Check status
python -m kinetra.agent_factory
python scripts/run_exhaustive_tests.py --ci-mode --test-type unit

# Get help
python scripts/run_exhaustive_tests.py --help

# Read full documentation
cat docs/EXHAUSTIVE_TESTING_GUIDE.md
```

---

## 🏆 Success Criteria

✅ All 6 agents instantiate via AgentFactory  
✅ CI mode tests pass in < 10 minutes  
✅ Full mode tests pass in < 3 hours  
✅ p-value < 0.01 in production  
✅ Omega Ratio > 2.7 in production  
✅ No NaN/Inf in numerical calculations  
✅ Statistical validation on real data  

**Framework Status**: 🚀 Production Ready

---

**End of Quick Reference**