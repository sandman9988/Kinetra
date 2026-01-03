# Exhaustive Testing Framework: Quick Start Guide

**Version**: 1.1  
**Last Updated**: 2025-01-03  
**Status**: Production-Ready ✅

---

## 🚀 What's New

### Recent Enhancements (v1.1)
1. ✅ **Dashboard Auto-Generation in CI** - Visual test reports automatically uploaded
2. ✅ **GPU Setup Documentation** - Clear NVIDIA/AMD installation instructions
3. ✅ **Enhanced Test Orchestration** - `--generate-dashboard` flag added

---

## ⚡ Quick Commands

### Verify System
```bash
# Check agent factory (6 agents should instantiate)
python -m kinetra.agent_factory

# Should output:
# ✅ ppo, dqn, linear_q, incumbent, competitor, researcher
# All agent factory tests passed ✅
```

### Run Fast Tests (5-10 minutes)
```bash
# CI mode - subset of combinations
KINETRA_CI_MODE=1 pytest tests/test_exhaustive_combinations.py -v

# Or use orchestrator
python scripts/run_exhaustive_tests.py --ci-mode
```

### Run Full Tests (1-2 hours)
```bash
# All combinations, all test types
python scripts/run_exhaustive_tests.py --full --all

# With parallelization (4 workers)
python scripts/run_exhaustive_tests.py --full --all --parallel 4
```

### Generate Dashboard Report
```bash
# After tests complete, generate visual report
python scripts/run_exhaustive_tests.py --ci-mode --generate-dashboard

# Or manually
python -c "
from kinetra.test_dashboard import TestDashboard
dashboard = TestDashboard()
dashboard.generate_static_report('test_report.html')
"

# Open report
open test_report.html  # macOS
xdg-open test_report.html  # Linux
```

---

## 🎯 Common Workflows

### Daily Development
```bash
# 1. Verify agents working
python -m kinetra.agent_factory

# 2. Run quick tests
KINETRA_CI_MODE=1 pytest tests/test_exhaustive_combinations.py::test_all_agents -v

# 3. Check code quality
make lint
make format
```

### Before PR
```bash
# Run full CI-mode suite
python scripts/run_exhaustive_tests.py --ci-mode --all --generate-dashboard

# Check coverage
pytest tests/ --cov=kinetra --cov-report=term-missing
```

### Weekly/Nightly (Automated in CI)
```bash
# Full exhaustive validation
python scripts/run_exhaustive_tests.py --full --all --parallel 4
```

---

## 🖥️ GPU Acceleration Setup (Optional)

GPU setup can provide **2-5x speedup** for neural agents (PPO, DQN).

### NVIDIA (CUDA)
```bash
# 1. Check CUDA version
nvidia-smi

# 2. Install CUDA-enabled PyTorch (example: CUDA 12.1)
pip uninstall torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 3. Verify
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

# 4. Benchmark
python -m kinetra.gpu_testing --benchmark
```

### AMD (ROCm - Linux only)
```bash
# 1. Check ROCm version
rocm-smi

# 2. Install ROCm-enabled PyTorch (example: ROCm 6.0)
pip uninstall torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.0

# 3. Verify
python -c "import torch; print(f'GPU Count: {torch.cuda.device_count()}')"

# 4. Benchmark
python -m kinetra.gpu_testing --benchmark
```

**Troubleshooting**: See `docs/EXHAUSTIVE_TESTING_GUIDE.md` → GPU Acceleration Setup

---

## 📊 Dashboard Features

The interactive dashboard (`test_report.html`) includes:

- **Heatmaps**: Agent × Regime × Timeframe performance
- **Agent Comparison**: Side-by-side metrics (Omega, Z-Factor, CHS, RoR)
- **Regime Analysis**: Performance breakdown by market regime
- **Metrics Distribution**: Statistical distribution of key metrics
- **Summary Stats**: Pass/fail rates, coverage, significance

**Access in CI**: Download from GitHub Actions → Artifacts → `test-dashboard`

---

## 🧪 Test Types

| Type | What it Tests | CI Time | Full Time |
|------|--------------|---------|-----------|
| **Unit** | Individual agent logic | ~3 min | ~15 min |
| **Integration** | End-to-end pipeline | ~3 min | ~20 min |
| **Monte Carlo** | Statistical significance (100 runs) | ~5 min | ~60 min |
| **Walk-Forward** | Out-of-sample validation | ~4 min | ~45 min |

**CI Mode**: Runs subset (10-20 combos per test type)  
**Full Mode**: Runs all 2,400 combinations

---

## 📁 Key Files

### Core System
- `kinetra/agent_factory.py` - Unified agent creation (6 agents)
- `kinetra/test_dashboard.py` - Interactive visualizations
- `kinetra/gpu_testing.py` - GPU acceleration utilities

### Testing
- `tests/test_exhaustive_combinations.py` - Main test harness
- `scripts/run_exhaustive_tests.py` - Test orchestration
- `.github/workflows/ci.yml` - CI/CD pipeline

### Documentation
- `docs/EXHAUSTIVE_TESTING_GUIDE.md` - Complete guide (768 lines)
- `EXHAUSTIVE_TESTING_ACTION_PLAN.md` - Roadmap & priorities
- `EXHAUSTIVE_TESTING_QUICKREF.md` - Command reference
- `EXHAUSTIVE_TESTING_IMPLEMENTATION_SUMMARY.md` - Recent changes

### Results
- `data/exhaustive_results_*.csv` - Test results (CSV)
- `test_results/test_report_*.json` - Summary reports (JSON)
- `test_results/test_dashboard_*.html` - Visual reports (HTML)
- `plots/*.png` - Generated plots

---

## 🔍 Troubleshooting

### Agent Factory Fails
```bash
# Symptom: ImportError or NameError
# Fix: Check dependencies
pip install -r requirements.txt
pip install -e ".[dev]"
```

### Tests Timeout in CI
```bash
# Symptom: CI jobs exceed time limits
# Fix: Already handled - CI mode runs subset
# If needed, reduce parallel workers:
python scripts/run_exhaustive_tests.py --ci-mode --parallel 2
```

### Dashboard Not Generated
```bash
# Symptom: Missing plotly/dash
# Fix: Install visualization dependencies
pip install plotly dash dash-bootstrap-components

# Or install all optional deps
pip install -e ".[dev,visualization]"
```

### GPU Not Detected
```bash
# Symptom: torch.cuda.is_available() returns False
# Fix: Ensure correct PyTorch build installed
# NVIDIA: Use cu121 or cu118 index URL
# AMD: Use rocm6.0 or rocm5.7 index URL
# See GPU section above
```

### Out of Memory (GPU)
```bash
# Symptom: CUDA out of memory errors
# Fix: Tests auto-adjust batch size
# Or reduce manually in agent config
# Graceful fallback to CPU built-in
```

---

## 📈 Performance Targets

| Metric | Target | Purpose |
|--------|--------|---------|
| **Omega Ratio** | > 2.7 | Asymmetric returns |
| **Z-Factor** | > 2.5 | Statistical edge |
| **CHS** | > 0.90 | System health |
| **RoR** | < 0.05 | Risk of ruin |
| **p-value** | < 0.01 | Significance |

---

## 🎓 Learn More

### Full Documentation
- **Complete Guide**: `docs/EXHAUSTIVE_TESTING_GUIDE.md`
- **Action Plan**: `EXHAUSTIVE_TESTING_ACTION_PLAN.md`
- **Implementation**: `EXHAUSTIVE_TESTING_IMPLEMENTATION_SUMMARY.md`

### Architecture Deep-Dive
- **Agent Factory**: Unified interface for 6 agent types
- **Test Harness**: Parameterized combinations (agent × asset × timeframe × regime)
- **Data Pipeline**: MetaAPI → CSV cache → Regime detection
- **Validation**: Monte Carlo (100 runs), statistical tests (p < 0.01)

### CI/CD Pipeline
- **Fast Tests**: PR validation (5-10 min)
- **Exhaustive Tests**: Nightly/release (2 hrs)
- **Agent Tests**: Factory verification (2 min)
- **Code Quality**: Lint, format, type checks

---

## 🚦 Status Indicators

### System Health
```bash
# All should pass:
✅ python -m kinetra.agent_factory  # 6 agents
✅ pytest tests/test_exhaustive_combinations.py::test_all_agents
✅ pytest tests/test_physics.py
✅ pytest tests/test_risk.py
```

### CI Health
- **Fast Tests**: Should complete in < 15 minutes
- **Exhaustive Tests**: Should complete in < 3 hours
- **Coverage**: Should be > 80%
- **Dashboard**: Should upload as artifact

---

## 💡 Tips & Best Practices

### Speed Up Development
```bash
# Run specific agent only
pytest tests/test_exhaustive_combinations.py -k "ppo"

# Run specific regime only
pytest tests/test_exhaustive_combinations.py -k "high_energy"

# Skip slow Monte Carlo tests
pytest tests/test_exhaustive_combinations.py -k "not monte_carlo"
```

### Parallel Execution
```bash
# Use all CPU cores
python scripts/run_exhaustive_tests.py --full --parallel auto

# Or specific number
python scripts/run_exhaustive_tests.py --full --parallel 4
```

### Coverage Reports
```bash
# Generate HTML coverage report
pytest tests/ --cov=kinetra --cov-report=html
open htmlcov/index.html
```

### Dashboard Customization
```python
from kinetra.test_dashboard import TestDashboard

dashboard = TestDashboard()

# Launch interactive (localhost:8050)
dashboard.launch(port=8050, debug=True)

# Generate static report
dashboard.generate_static_report('custom_report.html')
```

---

## 📞 Support

### Getting Help
1. Check documentation in `docs/`
2. Run self-tests: `python -m kinetra.agent_factory`
3. Review test results in `data/exhaustive_results_*.csv`
4. Check CI logs in GitHub Actions
5. Review action plan: `EXHAUSTIVE_TESTING_ACTION_PLAN.md`

### Common Issues
- **Import errors**: Run `pip install -r requirements.txt`
- **Test failures**: Check `KINETRA_CI_MODE` environment variable
- **Slow tests**: Use `--ci-mode` or `--parallel`
- **Missing data**: Tests skip missing combinations (expected)
- **GPU issues**: See GPU setup section above

---

## 🎯 Next Steps

### Just Getting Started?
1. Run `python -m kinetra.agent_factory` to verify setup
2. Run `python scripts/run_exhaustive_tests.py --ci-mode` for quick test
3. Review generated dashboard: `test_results/test_dashboard_*.html`
4. Read full guide: `docs/EXHAUSTIVE_TESTING_GUIDE.md`

### Ready for Production?
1. Run full suite: `python scripts/run_exhaustive_tests.py --full --all`
2. Enable GPU (optional): See GPU section above
3. Set up CI: Already configured in `.github/workflows/ci.yml`
4. Monitor dashboard artifacts in GitHub Actions

### Want to Contribute?
1. Read `EXHAUSTIVE_TESTING_ACTION_PLAN.md` for roadmap
2. Pick a priority item (HPO, live dashboard, new agents)
3. Add tests for new features
4. Generate dashboard to visualize improvements

---

**Last Updated**: 2025-01-03  
**Version**: 1.1  
**Status**: ✅ Production-Ready

For detailed information, see `docs/EXHAUSTIVE_TESTING_GUIDE.md`
