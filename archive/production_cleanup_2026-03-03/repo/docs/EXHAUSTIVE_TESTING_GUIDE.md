# Exhaustive Testing Framework for Kinetra

**Production-Grade, All-Agents, Real-Data Testing System**

Version: 1.0  
Last Updated: 2024-01-03

---

## Table of Contents

1. [Overview](#overview)
2. [Quick Start](#quick-start)
3. [Architecture](#architecture)
4. [Agent Factory](#agent-factory)
5. [Test Modes](#test-modes)
6. [Running Tests](#running-tests)
7. [CI/CD Integration](#cicd-integration)
8. [Test Types](#test-types)
9. [Data Pipeline](#data-pipeline)
10. [Performance Targets](#performance-targets)
11. [Troubleshooting](#troubleshooting)
12. [Contributing](#contributing)

---

## Overview

The Exhaustive Testing Framework validates Kinetra across **ALL** meaningful combinations:

- **6 Agent Types**: PPO, DQN, Linear Q, Incumbent, Competitor, Researcher
- **5 Asset Classes**: Crypto, Forex, Indices, Metals, Commodities
- **5 Timeframes**: M15, M30, H1, H4, D1
- **4 Regimes**: All, High Energy, Low Energy, Low Volatility
- **4 Test Types**: Unit, Integration, Monte Carlo, Walk-Forward

**Total Possible Combinations**: 6 × 5 × 5 × 4 × 4 = **2,400 test scenarios**

### Philosophy

- ✅ **First Principles** - No assumptions, everything validated
- ✅ **Real Data** - MetaAPI/MT5 data or CSV fallbacks
- ✅ **Statistical Rigor** - p < 0.01 required for significance
- ✅ **Vectorized Operations** - NumPy first, Python loops last resort
- ✅ **NaN Shields** - Numerical stability everywhere
- ✅ **CI-Friendly** - Fast subset mode for PRs, full mode for releases

---

## Quick Start

### Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Verify agent factory
python -m kinetra.agent_factory
```


### GPU Acceleration Setup (Optional)

Kinetra supports GPU acceleration for faster agent training during exhaustive tests. GPU setup is **optional** but can provide 2-5x speedup for neural network-based agents (PPO, DQN).

#### NVIDIA GPUs (CUDA)

```bash
# 1. Verify CUDA availability
nvidia-smi  # Should show your GPU

# 2. Uninstall CPU-only torch (if installed)
pip uninstall torch torchvision

# 3. Install CUDA-enabled torch
# For CUDA 12.1 (check your CUDA version with: nvcc --version)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# For CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# 4. Verify GPU is available to PyTorch
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0)}')"
```

#### AMD GPUs (ROCm - Linux only)

```bash
# 1. Verify ROCm installation
rocm-smi  # Should show your GPU

# 2. Uninstall CPU-only torch
pip uninstall torch torchvision

# 3. Install ROCm-enabled torch
# For ROCm 6.0 (check your ROCm version with: rocm-smi --showproductname)
pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.0

# For ROCm 5.7
pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm5.7

# 4. Verify GPU is available
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'Device count: {torch.cuda.device_count()}')"
```

#### Verify GPU Acceleration

```bash
# Run GPU benchmark
python -m kinetra.gpu_testing --benchmark

# Expected output:
# ✅ GPU detected: NVIDIA GeForce RTX 3090
# Benchmark: 1000 iterations
#   CPU: 12.34s
#   GPU: 2.56s (4.8x speedup)
```

#### Troubleshooting

**Issue**: `torch.cuda.is_available()` returns `False`
- **NVIDIA**: Ensure CUDA drivers match PyTorch CUDA version
- **AMD**: Ensure ROCm version matches PyTorch ROCm version
- Reinstall with correct index URL

**Issue**: Out of memory errors during tests
- Reduce batch size in agent config
- GPU auto-adjusts batch size in `kinetra/gpu_testing.py`
- Tests gracefully fallback to CPU if GPU OOM

**Issue**: Slower performance on GPU
- Small batches don't benefit from GPU
- Ensure `batch_size >= 128` for GPU advantage
- Profile with: `python -m kinetra.gpu_testing --benchmark --verbose`



### Run Fast Tests (CI Mode)

```bash
# Quick validation (5-10 minutes)
python scripts/run_exhaustive_tests.py --ci-mode --test-type unit

# Or use pytest directly
KINETRA_CI_MODE=1 pytest tests/test_exhaustive_combinations.py -v
```

### Run Full Tests (Exhaustive Mode)

```bash
# Full validation (1-2 hours)
python scripts/run_exhaustive_tests.py --full --all

# Or with parallelization
python scripts/run_exhaustive_tests.py --full --all --parallel 4
```

---

## Architecture

### Component Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    Exhaustive Test Harness                  │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐ │
│  │ Agent Factory│───▶│ Data Pipeline│───▶│ Test Engine  │ │
│  └──────────────┘    └──────────────┘    └──────────────┘ │
│         │                    │                    │        │
│         ▼                    ▼                    ▼        │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐ │
│  │ 6 Agents     │    │ Real Data    │    │ 4 Test Types │ │
│  │ - PPO        │    │ - MetaAPI    │    │ - Unit       │ │
│  │ - DQN        │    │ - MT5        │    │ - Integration│ │
│  │ - Linear Q   │    │ - CSV Cache  │    │ - Monte Carlo│ │
│  │ - Incumbent  │    │ - Regimes    │    │ - Walk-Fwd   │ │
│  │ - Competitor │    └──────────────┘    └──────────────┘ │
│  │ - Researcher │                                          │
│  └──────────────┘                                          │
│                                                             │
├─────────────────────────────────────────────────────────────┤
│                    Statistical Validation                   │
│  • Omega Ratio > 2.7   • Z-Factor > 2.5   • p < 0.01      │
└─────────────────────────────────────────────────────────────┘
```

### Directory Structure

```
Kinetra/
├── kinetra/
│   ├── agent_factory.py          # Unified agent creation
│   ├── rl_agent.py               # PPO agent
│   ├── rl_neural_agent.py        # DQN agent
│   ├── rl_physics_env.py         # Linear Q agent
│   ├── triad_system.py           # Triad agents (Incumbent, Competitor, Researcher)
│   ├── backtest_engine.py        # Monte Carlo validation
│   └── physics_engine.py         # Energy-based features
├── tests/
│   ├── test_exhaustive_combinations.py  # Main test suite
│   ├── test_physics.py           # Physics unit tests
│   ├── test_risk.py              # Risk management tests
│   └── test_integration.py       # Integration tests
├── scripts/
│   └── run_exhaustive_tests.py   # Test orchestration
├── data/
│   ├── master_standardized/      # CSV data cache
│   └── exhaustive_results_*.csv  # Test results
├── docs/
│   ├── EXHAUSTIVE_TESTING_GUIDE.md (this file)
│   └── EMPIRICAL_THEOREMS.md     # Validated discoveries
└── .github/
    └── workflows/
        └── ci.yml                # GitHub Actions config
```

---

## Agent Factory

### Overview

The `AgentFactory` provides a **unified interface** for all agent types, eliminating the need for custom integration code for each agent.

### Agent Registry

| Agent Type   | Class              | Description                          | Style  |
|--------------|--------------------|--------------------------------------|--------|
| `ppo`        | `KinetraAgent`     | Proximal Policy Optimization         | PPO    |
| `dqn`        | `NeuralAgent`      | Deep Q-Network with replay           | DQN    |
| `linear_q`   | `SimpleRLAgent`    | Linear Q-Learning (fast)             | Q-Lrn  |
| `incumbent`  | `IncumbentAgent`   | PPO-style Triad (stable)             | Triad  |
| `competitor` | `CompetitorAgent`  | A2C-style Triad (aggressive)         | Triad  |
| `researcher` | `ResearcherAgent`  | SAC-style Triad (exploration)        | Triad  |

### Usage Examples

```python
from kinetra.agent_factory import AgentFactory

# Create a single agent
agent = AgentFactory.create('ppo', state_dim=43, action_dim=4)

# Create with custom config
agent = AgentFactory.create('dqn', state_dim=43, action_dim=4, 
                            config={'lr': 1e-3, 'gamma': 0.95})

# Create with unified interface (wrapped)
wrapped = AgentFactory.create_wrapped('ppo', state_dim=43, action_dim=4)
action = wrapped.select_action(state, explore=True)
wrapped.update(state, action, reward, next_state, done)

# Create all agents for comparison
all_agents = AgentFactory.create_all(state_dim=43, action_dim=4, wrapped=True)
for agent_type, agent in all_agents.items():
    print(f"{agent_type}: {agent}")
```

### Unified Interface

All agents (when wrapped) implement:

```python
class UnifiedAgentInterface:
    def select_action(self, state: np.ndarray, explore: bool = True) -> int:
        """Select action given state."""
        pass
    
    def update(self, state, action, reward, next_state, done) -> Optional[Dict]:
        """Update from experience tuple."""
        pass
```

This interface **automatically adapts** to different agent APIs:
- PPO: `deterministic` parameter → `explore`
- DQN/Linear Q: `training` parameter → `explore`
- Triad: `explore` parameter (native)

---

## Test Modes

### CI Mode (Fast)

**Purpose**: Quick validation for pull requests  
**Duration**: 5-10 minutes  
**Coverage**: Subset of combinations

```bash
# Enable via environment variable
export KINETRA_CI_MODE=1

# Or via script
python scripts/run_exhaustive_tests.py --ci-mode
```

**CI Mode Configuration**:
- Asset Classes: Crypto, Forex (2 of 5)
- Timeframes: H1, D1 (2 of 5)
- Agents: PPO, DQN, Incumbent (3 of 6)
- Regimes: All, High Energy (2 of 4)
- Monte Carlo Runs: 10 (vs 100 in full mode)

**Total CI Combinations**: 2 × 2 × 3 × 2 = **24 scenarios**

### Full Mode (Exhaustive)

**Purpose**: Complete validation for releases  
**Duration**: 1-2 hours  
**Coverage**: All combinations

```bash
# Disable CI mode
export KINETRA_CI_MODE=0

# Or via script
python scripts/run_exhaustive_tests.py --full --all
```

**Full Mode Configuration**:
- Asset Classes: All 5
- Timeframes: All 5
- Agents: All 6
- Regimes: All 4
- Monte Carlo Runs: 100

**Total Full Combinations**: 5 × 5 × 6 × 4 = **600 base scenarios** × 4 test types = **2,400 total**

---

## Running Tests

### Using pytest Directly

```bash
# Run all exhaustive tests (CI mode)
KINETRA_CI_MODE=1 pytest tests/test_exhaustive_combinations.py -v

# Run specific test type
KINETRA_CI_MODE=1 pytest tests/test_exhaustive_combinations.py::TestExhaustiveCombinations::test_all_combos[unit] -v

# Run with parallelization
KINETRA_CI_MODE=1 pytest tests/test_exhaustive_combinations.py -n 4

# Run with coverage
KINETRA_CI_MODE=1 pytest tests/test_exhaustive_combinations.py --cov=kinetra --cov-report=html
```

### Using Test Runner Script

```bash
# CI mode - unit tests only
python scripts/run_exhaustive_tests.py --ci-mode --test-type unit

# CI mode - all test types
python scripts/run_exhaustive_tests.py --ci-mode --all

# Full mode - specific test type
python scripts/run_exhaustive_tests.py --full --test-type monte_carlo --parallel 4

# Full mode - everything with coverage
python scripts/run_exhaustive_tests.py --full --all --parallel 4 --coverage
```

### Script Options

| Option            | Description                               |
|-------------------|-------------------------------------------|
| `--ci-mode`       | Fast subset testing (5-10 min)            |
| `--full`          | Full exhaustive testing (1-2 hours)       |
| `--test-type`     | Specific test: unit/integration/mc/wf     |
| `--all`           | Run all 4 test types                      |
| `--parallel N`    | Use N parallel workers                    |
| `--coverage`      | Generate coverage report                  |
| `--stop-on-fail`  | Stop on first failure                     |
| `--skip-verify`   | Skip pre-flight checks                    |
| `--report-dir`    | Custom report directory                   |

---

## CI/CD Integration

### GitHub Actions Workflow

The CI pipeline has **4 jobs**:

#### 1. Fast Tests (Pull Requests)

Runs on every PR, uses CI mode:

```yaml
- name: Run exhaustive combinations (CI subset)
  env:
    KINETRA_CI_MODE: 1
  run: pytest tests/test_exhaustive_combinations.py -v
  timeout-minutes: 15
```

#### 2. Exhaustive Tests (Nightly)

Runs full suite at 2 AM UTC:

```yaml
schedule:
  - cron: "0 2 * * *"
  
- name: Run full exhaustive tests
  env:
    KINETRA_CI_MODE: 0
  run: pytest tests/test_exhaustive_combinations.py -v -n 4
  timeout-minutes: 180
```

#### 3. Agent Factory Verification

Quick check that all agents instantiate:

```yaml
- name: Test AgentFactory
  run: python -m kinetra.agent_factory
  timeout-minutes: 2
```

#### 4. Code Quality

Linting, formatting, type checking:

```yaml
- name: Run ruff (lint)
  run: ruff check kinetra/ tests/
```

### Triggering Full Tests Manually

Add `[exhaustive]` to commit message:

```bash
git commit -m "Major refactor [exhaustive]"
```

This triggers full exhaustive tests even on a PR.

---

## Test Types

### 1. Unit Tests

**Purpose**: Validate individual components  
**Duration**: ~1-2 minutes (CI mode)

**What's Tested**:
- Composite Health Score (CHS) calculation
- Risk-of-Ruin (RoR) computation
- Physics state calculation
- Reward shaping

**Thresholds** (relaxed for synthetic data):
- CHS > 0.5
- RoR < 0.5

```python
def _test_unit(self, result: Dict, state: Dict, df_regime: pd.DataFrame) -> Dict:
    rewards = adaptive_reward_shaping(state, df_regime)
    chs = compute_chs(energy_capture, omega_val, stability)
    ror = compute_ror(rewards["mu"], rewards["sigma"], rewards["X_t"])
    result["valid"] = chs > 0.5 and ror < 0.5
    return result
```

### 2. Integration Tests

**Purpose**: Validate agent training and action selection  
**Duration**: ~2-3 minutes (CI mode)

**What's Tested**:
- Agent instantiation via AgentFactory
- Action selection with explore/exploit modes
- Single-episode training loop
- Update mechanism

**Validation**:
- Valid actions (0-3)
- Finite rewards
- Non-zero training steps

```python
def _test_integration(self, result: Dict, agent_type: str, df_regime: pd.DataFrame) -> Dict:
    agent = create_agent(agent_type)
    total_reward, num_steps = train_agent_episode(agent, df_regime)
    action = test_agent_action_selection(agent, test_state)
    result["valid"] = 0 <= action < 4 and num_steps > 0 and np.isfinite(total_reward)
    return result
```

### 3. Monte Carlo Tests

**Purpose**: Statistical significance of backtest results  
**Duration**: ~5-10 minutes (CI mode, 10 runs); ~30-60 minutes (full mode, 100 runs)

**What's Tested**:
- Backtest stability across shuffled data
- Omega Ratio calculation
- Z-Factor for significance
- Bootstrap confidence intervals

**Thresholds** (relaxed for synthetic data):
- Omega Ratio > 1.0 (target: 2.7)
- Z-Factor > 1.0 (target: 2.5)
- p-value < 0.10 (target: 0.01)

```python
def _test_monte_carlo(self, result: Dict, df_regime: pd.DataFrame, symbol_spec: SymbolSpec) -> Dict:
    mc_df = monte_carlo_backtest(df_regime, symbol_spec, n_runs=MC_RUNS)
    mc_pnls = mc_df["total_net_pnl"].values
    result["omega"] = self.omega_ratio(mc_pnls)
    result["z_factor"] = self.z_factor(mc_pnls)
    _, p_value = stats.ttest_1samp(mc_pnls, 0)
    result["valid"] = result["omega"] > 1.0 and result["z_factor"] > 1.0 and p_value < 0.10
    return result
```

### 4. Walk-Forward Tests

**Purpose**: Regime stability across train/test splits  
**Duration**: ~3-5 minutes (CI mode)

**What's Tested**:
- Physics state consistency across splits
- Regime detection stability
- Out-of-sample validation

**Validation**:
- Energy correlation > 0.7
- Damping correlation > 0.6
- Entropy correlation > 0.6

```python
def _test_walk_forward(self, result: Dict, df_regime: pd.DataFrame, engine: PhysicsEngine) -> Dict:
    split = len(df_regime) // 2
    train_state = engine.compute_physics_state_from_ohlcv(df_regime[:split])
    test_state = engine.compute_physics_state_from_ohlcv(df_regime[split:])
    # Compute correlations between train/test states
    result["valid"] = all_correlations > threshold
    return result
```

---

## Data Pipeline

### Data Sources (Priority Order)

1. **MetaAPI** (live data, requires token)
2. **MT5 Direct** (local MT5 installation)
3. **CSV Cache** (`data/master_standardized/`)
4. **Graceful Skip** (if all fail)

### Data Preparation Pipeline

```python
def prepare_real_data(df_raw: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """
    Prepare real data for testing.
    
    Steps:
    1. Validate OHLCV columns
    2. Compute log returns (vectorized)
    3. Fill gaps (forward fill + interpolation)
    4. Clip outliers (adaptive, 99.9th percentile)
    5. Add NaN shields
    6. Validate sufficient data (>100 bars)
    """
```

### Regime Slicing

Regimes are identified using physics features:

```python
def regime_slice(df: pd.DataFrame, regime: str) -> pd.DataFrame:
    if regime == "all":
        return df
    elif regime == "high_energy":
        return df[df["energy"] > df["energy"].quantile(0.75)]
    elif regime == "low_energy":
        return df[df["energy"] < df["energy"].quantile(0.25)]
    elif regime == "low_vol":
        return df[df["volatility"] < df["volatility"].quantile(0.25)]
```

### Data Quality Checks

Before testing, data must pass:
- ✅ Minimum 100 bars
- ✅ Non-zero variance in returns
- ✅ No infinite values
- ✅ < 5% NaN values
- ✅ Valid OHLCV relationships (H ≥ C ≥ L)

---

## Performance Targets

### Production Targets (Full Mode)

| Metric                  | Target | Purpose                         |
|-------------------------|--------|---------------------------------|
| **Omega Ratio**         | > 2.7  | Asymmetric returns              |
| **Z-Factor**            | > 2.5  | Statistical edge significance   |
| **% Energy Captured**   | > 65%  | Physics alignment efficiency    |
| **Composite Health**    | > 0.90 | System stability                |
| **% MFE Captured**      | > 60%  | Execution quality               |
| **p-value**             | < 0.01 | Statistical significance        |
| **Risk-of-Ruin**        | < 5%   | Capital preservation            |

### CI Targets (Relaxed for Speed)

| Metric                  | CI Target | Relaxation Reason               |
|-------------------------|-----------|----------------------------------|
| **Omega Ratio**         | > 1.0     | Fewer MC runs, synthetic data    |
| **Z-Factor**            | > 1.0     | Smaller sample size              |
| **CHS**                 | > 0.5     | Quick validation                 |
| **p-value**             | < 0.10    | Less stringent for CI            |
| **RoR**                 | < 0.5     | Acceptable for subset tests      |

---

## Output and Artifacts

### Generated Files

```
data/
├── exhaustive_results_unit_20240103_1530.csv
├── exhaustive_results_integration_20240103_1535.csv
├── exhaustive_results_monte_carlo_20240103_1545.csv
└── exhaustive_results_walk_forward_20240103_1600.csv

plots/
├── heatmap_unit_20240103_1530.png
├── heatmap_integration_20240103_1535.png
├── heatmap_monte_carlo_20240103_1545.png
└── heatmap_walk_forward_20240103_1600.png

logs/
└── exhaustive.log

docs/
└── EMPIRICAL_THEOREMS.md (updated automatically)

test_results/
└── test_report_20240103_153045.json
```

### CSV Format

```csv
instrument,timeframe,asset_class,agent_type,regime,test_type,valid,omega,z_factor,chs,ror,p_value,error
BTCUSD,H1,crypto,ppo,all,unit,True,,,0.85,0.03,,
EURUSD,D1,forex,dqn,high_energy,monte_carlo,True,3.2,2.8,,,0.005,
```

### Heatmap Visualization

Automatically generated heatmaps show:
- **Rows**: Instrument × Timeframe
- **Columns**: Agent × Regime
- **Color**: Success/Failure or metric value

---

## Troubleshooting

### Common Issues

#### 1. Import Errors

```
ImportError: cannot import name 'Categorical'
```

**Fix**: Updated in `kinetra/rl_agent.py`:
```python
from torch.distributions import Categorical
```

#### 2. Tests Too Slow

**Problem**: Full mode takes too long

**Solutions**:
- Use CI mode for development: `--ci-mode`
- Parallelize: `--parallel 4`
- Test specific type: `--test-type unit`

#### 3. Data Loading Failures

```
WARNING: Skipped BTCUSD H1: No data available
```

**Solutions**:
- Set MetaAPI token: `export METAAPI_TOKEN=your_token`
- Use CSV cache: Place files in `data/master_standardized/`
- Tests will skip gracefully if no data available

#### 4. Agent Creation Failures

```
ValueError: Unknown agent type: 'ppo'
```

**Fix**: Ensure `agent_factory.py` is up to date with all 6 agents

**Verify**:
```bash
python -m kinetra.agent_factory
```

#### 5. Memory Issues

**Problem**: Out of memory with full mode

**Solutions**:
- Reduce parallel workers: `--parallel 2`
- Run test types separately
- Increase swap space
- Use cloud instance with more RAM

---

## Contributing

### Adding a New Agent

1. **Implement the agent class**:
```python
class MyNewAgent:
    def __init__(self, state_dim: int, action_dim: int, **kwargs):
        self.state_dim = state_dim
        self.action_dim = action_dim
    
    def select_action(self, state: np.ndarray, explore: bool = True) -> int:
        # Your logic here
        return action
    
    def update(self, state, action, reward, next_state, done):
        # Your logic here
        pass
```

2. **Register with AgentFactory**:
```python
from kinetra.agent_factory import AgentFactory

AgentFactory.register_agent(
    name='my_new_agent',
    agent_class=MyNewAgent,
    description='My awesome new agent',
    default_params={'state_dim': 43, 'action_dim': 4},
    param_mapping={'state_dim': 'state_dim', 'action_dim': 'action_dim'}
)
```

3. **Test it**:
```bash
python -m kinetra.agent_factory
KINETRA_CI_MODE=1 pytest tests/test_exhaustive_combinations.py::test_all_agents -v
```

### Adding a New Test Type

1. **Add test method** in `TestExhaustiveCombinations`:
```python
def _test_my_new_type(self, result: Dict, ...) -> Dict:
    # Your test logic
    result["valid"] = your_validation_logic
    return result
```

2. **Wire it up** in `_run_single_combo`:
```python
elif test_type == "my_new_type":
    result = self._test_my_new_type(result, ...)
```

3. **Add to parametrize**:
```python
@pytest.mark.parametrize("test_type", ["unit", "integration", "monte_carlo", "walk_forward", "my_new_type"])
```

---

## Best Practices

### ✅ DO

- Use AgentFactory for all agent creation
- Enable CI mode for development
- Run full mode before releases
- Check coverage: `--coverage`
- Parallelize when possible: `--parallel 4`
- Validate with statistical tests (p < 0.01)
- Use vectorized NumPy operations
- Add NaN shields to numerical calculations

### ❌ DON'T

- Hardcode agent creation (use AgentFactory)
- Skip pre-flight verification in production
- Ignore test failures in CI mode
- Use magic numbers without justification
- Add Python loops where NumPy can vectorize
- Remove working code without strong justification

---

## References

- **Main Instructions**: `AI_AGENT_INSTRUCTIONS.md`
- **Copilot Instructions**: `.github/copilot-instructions.md`
- **Theorem Proofs**: `docs/theorem_proofs.md`
- **Empirical Theorems**: `docs/EMPIRICAL_THEOREMS.md`
- **Testing Framework**: `docs/TESTING_FRAMEWORK.md`

---

## Summary

The Exhaustive Testing Framework provides **production-grade validation** across all agents, data, and regimes:

- ✅ **6 agents** tested uniformly via AgentFactory
- ✅ **CI mode** for fast PR feedback (5-10 min)
- ✅ **Full mode** for release validation (1-2 hours)
- ✅ **Real data** from MetaAPI/MT5 or CSV cache
- ✅ **Statistical rigor** with p < 0.01 threshold
- ✅ **4 test types** covering unit, integration, Monte Carlo, walk-forward
- ✅ **Automated CI/CD** via GitHub Actions
- ✅ **Comprehensive reports** with CSVs, heatmaps, and logs

**Ready to validate! 🚀**