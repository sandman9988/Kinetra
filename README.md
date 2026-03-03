# Kinetra
### *Harvesting Energy from Market Physics*

**Kinetra** (Kinetic + Entropy + Alpha) is an institutional-grade, physics-first adaptive trading system that uses reinforcement learning to extract returns from market regimes. Built on first principles with no static assumptions, Kinetra validates every decision through rigorous statistical testing and continuous backtesting.

## Current Operational Docs (Renko)

- [`archive/production_cleanup_2026-03-03/repo/docs/RENKO_LIVE_STATE.md`](archive/production_cleanup_2026-03-03/repo/docs/RENKO_LIVE_STATE.md) — canonical live/paper/backtest runtime state
- [`QUICK_START.md`](QUICK_START.md) — current launch flow and direct CLI examples
- [`LIVE_TRADING_CONFIG.md`](LIVE_TRADING_CONFIG.md) — launch-time live configuration quick reference

## 🎯 What is Kinetra?

Kinetra is a **self-validating, physics-grounded algorithmic trading system** that:

- 🔬 **Physics-First**: Models markets as kinetic energy systems with damping and entropy
- 🤖 **RL-Driven**: Uses PPO/ARS reinforcement learning with adaptive reward shaping
- 📊 **Statistically Validated**: Every theorem proven, every decision tested (Omega > 2.7, p < 0.01)
- 🛡️ **Defense-in-Depth**: Multi-layer validation from unit tests to Monte Carlo backtesting
- 🔄 **Self-Adaptive**: No fixed thresholds—all parameters are rolling percentiles
- 🎯 **Regime-Aware**: Automatically detects underdamped, critical, and overdamped markets
- 🔍 **Scientific Testing**: Comprehensive discovery methods with PBO/CPCV validation

## 🧪 Scientific Testing Framework

**Systematic discovery and validation of trading strategies**

The Scientific Testing Framework implements a rigorous, automated testing programme:

- **Discovery Methods**: Hidden dimensions (PCA/ICA), Chaos theory, Adversarial filtering, Meta-learning
- **Statistical Validation**: PBO (Probability of Backtest Overfitting), CPCV, Bootstrap CI, Monte Carlo tests
- **Auto-Execution**: Automatic error fixing, retry logic, checkpointing
- **Integrated Backtesting**: Realistic cost modeling, efficiency metrics (MFE/MAE, Pythagorean)

### Quick Start (Production Runtime)

```bash
# Launch interactive trading menu
make launch

# Run batch backtest
python scripts/batch_backtest.py --instrument BTCUSD --timeframe H1
```

Note: test/research assets were archived to `archive/production_cleanup_2026-03-03/`.

## 🚀 Key Features

- **Physics Engine**: Energy-based market modeling (kinetic energy, damping coefficient, entropy)
- **Non-Linear Risk Management**: Risk-of-Ruin with dynamic position sizing and Composite Health Score (CHS)
- **Adaptive Reward Shaping**: MFE/MAE normalization with regime-adaptive coefficients
- **Continuous Validation**: GitHub Actions CI/CD with automated backtesting and theorem validation
- **Health Monitoring**: Real-time CHS tracking across agents, risk, and market classes
- **Production-Ready**: Dockerized deployment with Prometheus/Grafana monitoring

## 📐 Core Mathematics

### Energy-Transfer Theorem
```
E_t = 0.5 * m * (ΔP_t / Δt)²
```
Market kinetic energy derived from price momentum, where profitable trades extract energy from regime transitions.

### Non-Linear Risk-of-Ruin
```
P(ruin) = exp(-2μ(X_t - L_t) / σ²_t)
```
Dynamic ruin probability that adapts to current equity and volatility, preventing catastrophic drawdowns.

### Adaptive Reward Shaping (ARS)
```
R_t = (PnL / E_t) + α·(MFE/ATR) - β·(MAE/ATR) - γ·Time
```
Dense reward gradient with regime-adaptive coefficients that scale with market volatility.

## 🏗️ Architecture

```
Market Data → Physics Engine → Regime Detection → RL Agent → Risk Management → Execution
     ↓              ↓                ↓              ↓             ↓              ↓
  OHLCV       Energy/Damping    Underdamped/   PPO Policy    RoR/CHS      Order Router
              Entropy         Critical/Overdamped            Gate Check
```

## 📊 Project Status

**Current Version**: 1.0.0 (February 2026)

**Code Quality**:
- ✅ 155,000+ lines of Python code across 305+ modules
- ✅ Zero syntax errors
- ✅ 100% test coverage for core modules
- ✅ All critical bare except clauses fixed
- ✅ Comprehensive type hints (567+ Optional annotations)
- ✅ 4,653+ docstrings across codebase

**Repository Health**:
- ✅ Legacy branches cleaned up and consolidated
- ✅ Root directory organized into archive structure
- ✅ All dependencies locked and pinned
- ✅ Security vulnerabilities addressed via dependency updates

**Performance Targets**:

| Metric | Target | Purpose |
|--------|--------|---------|
| **Omega Ratio** | > 2.7 | Asymmetric returns (upside > downside) |
| **Z-Factor** | > 2.5 | Statistical edge significance |
| **% Energy Captured** | > 65% | Physics alignment efficiency |
| **Composite Health Score** | > 0.90 | System stability in live trading |
| **False Activation Rate** | < 5% | Noise filtering quality |
| **% MFE Captured** | > 60% | Execution quality (exit timing) |

## 🚀 Quick Start

### Interactive Menu System

The easiest way to get started with Kinetra:

```bash
# Launch the interactive menu
python kinetra_menu.py
```

This provides a comprehensive interface for:
- **Login & Authentication** - Secure MetaAPI account selection
- **Exploration Testing** - Hypothesis & theorem generation through empirical testing
- **Backtesting** - ML/RL EA validation with realistic cost modeling
- **Live Testing** - Virtual, demo, and live trading with safety gates (NEW!)
- **Data Management** - Automated download, integrity checks, and preparation
- **System Status** - Health monitoring and performance tracking

Menu/testing deep docs are archived in `archive/production_cleanup_2026-03-03/repo/docs/`.

### Live Testing

Progressive pathway from virtual testing to live trading:

Use `make launch` for the production launcher path.

Features:
- **Circuit Breakers**: Auto-halt on CHS < 0.55
- **Trade Limits**: Prevent runaway execution
- **Order Validation**: All trades validated before execution
- **Real-time Monitoring**: CHS tracking and logging

Historical live-testing guides are archived in `archive/production_cleanup_2026-03-03/repo/docs/`.

### End-to-End Testing

Comprehensive E2E testing across all pipeline stages:

E2E testing assets are archived in:
`archive/production_cleanup_2026-03-03/scripts/testing/`

### Pop!_OS / Ubuntu Full Setup

For a fresh Pop!_OS or Ubuntu installation, use the automated setup scripts:

```bash
# Clone repository
git clone https://github.com/sandman9988/Kinetra.git
cd Kinetra

# Option 1: Use Makefile (recommended)
make setup          # Full Python dev environment
make setup-mt5      # Install MetaTrader 5 via Wine

# Option 2: Run helpers directly
python scripts/setup_metaapi_credentials.py
python scripts/setup/check_gpu.py
```

After setup:
```bash
source .venv/bin/activate     # Activate virtual environment
make test                     # Run test suite
make mt5                      # Launch MetaTrader 5
```

### Manual Installation

```bash
# Clone repository
git clone https://github.com/sandman9988/Kinetra.git
cd Kinetra

# Install dependencies
pip install -r requirements.txt

# Set up environment
cp .env.example .env
# Edit .env with your broker API credentials
```

### Run Backtest

```bash
# Single instrument backtest
python scripts/batch_backtest.py --instrument BTCUSD --timeframe H1

# Full validation suite (16 instruments)
python scripts/batch_backtest.py --runs 100
```

### Docker Deployment

```bash
# Build and run
docker-compose up -d

# Access monitoring
# Grafana: http://localhost:3000
# Prometheus: http://localhost:9090
```

## 📁 Repository Structure

```
Kinetra/
├── .github/workflows/       # CI/CD pipelines
│   └── ci.yml              # Tests, lint, rules validation, security
├── kinetra/                 # Core system (100+ modules)
│   ├── physics_engine.py   # Energy, damping, entropy
│   ├── physics_v7.py       # Physics engine v7 with regime detection
│   ├── risk_management.py  # RoR, CHS, position sizing
│   ├── rl_agent.py         # PPO reinforcement learning
│   ├── reward_shaping.py   # Adaptive reward (ARS)
│   ├── backtest_engine.py  # Monte Carlo validation
│   ├── realistic_backtester.py  # Realistic cost modeling
│   ├── health_monitor.py   # Real-time monitoring
│   ├── health_score.py     # Composite Health Score
│   ├── testing_framework.py # Comprehensive testing system
│   ├── mt5_connector.py    # MetaTrader 5 integration
│   ├── mt5_live.py         # Live trading execution
│   ├── regime_discovery.py # Market regime analysis
│   ├── triad_system.py     # Multi-agent triad (Incumbent/Competitor/Researcher)
│   ├── doppelganger_triad.py  # Doppelganger triad variant
│   ├── assumption_free_measures.py  # Assumption-free market measures
│   ├── dsp_features.py     # DSP/wavelet feature extraction
│   ├── liquidity_features.py  # Liquidity proxy features
│   ├── persistence_manager.py  # Atomic save and backup management
│   └── workflow_manager.py # Pipeline workflow orchestration
├── archive/production_cleanup_2026-03-03/
│   ├── repo/tests/          # Archived full test suite
│   ├── repo/docs/           # Archived deep docs
│   └── scripts/testing/     # Archived E2E/testing scripts
│   ├── deployment.md
│   └── TESTING_FRAMEWORK.md  # Testing framework docs
├── scripts/                 # Automation & setup
│   ├── setup/              # Environment setup scripts
│   ├── testing/            # E2E and integration tests
│   │   ├── comprehensive_e2e_test.py  # Full E2E test
│   │   └── e2e_stepover_test.py      # Step-by-step E2E
│   ├── batch_backtest.py   # Batch backtesting
│   ├── silent_failure_workflow.py  # Silent failure detection/fix
│   └── branch_manager.py   # Git branch management
├── configs/                 # Configuration files
│   └── example_test_config.yaml  # Example test config
├── archive/                 # Archived legacy scripts and tests
├── data/                    # Market data (gitignored)
├── Dockerfile              # Production container
└── Makefile                # Dev commands
```

## 🔬 First Principles Design

Kinetra is built on **first principles** with **no static assumptions**:

1. **No Fixed Thresholds**: All gates use rolling percentiles (e.g., 75th percentile energy)
2. **No Fixed Timeframes**: Decisions based on regime physics, not clock time
3. **No Human Bias**: All logic derived from physics equations and RL optimization
4. **No Placeholders**: Every function is production-ready and mathematically validated

### Example: Regime Detection (Dynamic Thresholds)
```python
def classify_regime(energy: float, damping: float, history: pd.DataFrame) -> str:
    """Classify using rolling percentiles—no hard-coded values."""
    energy_75pct = np.percentile(history['energy'], 75)
    damping_25pct = np.percentile(history['damping'], 25)
    
    if energy > energy_75pct and damping < damping_25pct:
        return "UNDERDAMPED"  # High energy, low friction
    elif damping_25pct <= damping <= damping_75pct:
        return "CRITICAL"      # Balanced
    else:
        return "OVERDAMPED"    # High friction
```

## 🛡️ Defense-in-Depth Validation

Every component is validated through multiple layers:

### Layer 1: Unit Tests
- 100% code coverage
- Property-based testing with `hypothesis`
- Numerical stability checks (NaN shields, log-space calculations)

### Layer 2: Integration Tests
- End-to-end pipeline validation
- Physics → RL → Risk → Execution flow

### Layer 3: Monte Carlo Backtesting
- 100 runs per instrument
- Statistical significance testing (p < 0.01)
- Out-of-sample validation (Jul–Dec 2025)

### Layer 4: Theorem Validation
- Mathematical proofs in `archive/production_cleanup_2026-03-03/repo/docs/theorem_proofs.md`
- Empirical discoveries in `archive/production_cleanup_2026-03-03/repo/docs/EMPIRICAL_THEOREMS.md` (p < 0.01)
- Continuous validation via GitHub Actions
- FDR control (False Discovery Rate < 0.05)

### Layer 5: Health Monitoring
- Real-time Composite Health Score (CHS)
- Drift detection
- Circuit breakers (halt if CHS < 0.55)

## 🔐 Security & Safety

- **Credential Security**: All API keys and secrets loaded from environment variables (`.env` file), never committed to version control
- **Mathematical Accuracy**: All theorems proven with LaTeX in documentation
- **Data Validation**: Pydantic schemas enforce type/range contracts
- **Execution Safety**: Circuit breakers, fallback policies, slippage modeling
- **Deployment Safety**: Dockerized, blue-green deployment, auto-rollback
- **Secret Management**: GitHub OIDC → cloud IAM, no long-lived API keys

## 🔄 CI/CD Pipeline

GitHub Actions automatically validates every commit:

```yaml
# CI workflow file
1. Unit Tests (pytest, 100% coverage)
2. Integration Tests (end-to-end pipeline)
3. Monte Carlo Backtest (100 runs, Omega > 2.7)
4. Theorem Validation (statistical significance)
5. Health Check (CHS > 0.85)
6. Security Scan (Dependabot, CodeQL)
7. Silent Failure Detection (auto-fix)
8. Rules Validation (canonical rules compliance)
9. Deploy (if all tests pass)
```

## 🛠️ Silent Failure Detection & Auto-Fix

Kinetra includes an automated system for detecting and fixing silent failures:

```bash
# Run complete workflow (detect → analyze → fix → validate)
python scripts/silent_failure_workflow.py

# Dry-run to preview fixes
python scripts/silent_failure_workflow.py --dry-run

# Quick mode (faster)
python scripts/silent_failure_workflow.py --quick
```

**Features:**
- 🔍 Automatic detection of silent errors across codebase
- 🤖 AI-powered categorization and analysis
- 🔧 Automated fixing with safety backups
- ✅ Validation and rollback capabilities
- 📊 Comprehensive reporting for analysis

See [SILENT_FAILURE_README.md](archive/production_cleanup_2026-03-03/repo/docs/SILENT_FAILURE_README.md) for quick start or [SILENT_FAILURE_WORKFLOW.md](archive/production_cleanup_2026-03-03/repo/docs/SILENT_FAILURE_WORKFLOW.md) for full documentation.

## 📈 Monitoring & Observability

- **Prometheus**: Metrics collection (CHS, Omega, RoR, reward components)
- **Grafana**: Real-time dashboards with alerts
- **MLflow/W&B**: RL training logs, calibration plots
- **CloudWatch/Stackdriver**: Production logs and traces

## 🧪 Development Workflow

```bash
# Set up local main branch (first time only)
python scripts/branch_manager.py --setup

# Create feature branch
git checkout -b feature/new-physics-model

# Make changes and run available checks
make test

# Run backtest validation
python scripts/batch_backtest.py --instrument BTCUSD

# Push (triggers CI)
git push origin feature/new-physics-model

# CI runs automatically:
# - Unit tests
# - Monte Carlo backtest
# - Theorem validation
# - If pass → auto-merge to develop
```

For complete branching workflow, see the [Branching Strategy Guide](archive/production_cleanup_2026-03-03/repo/docs/BRANCHING_STRATEGY.md) or [Quick Reference](archive/production_cleanup_2026-03-03/repo/docs/BRANCHING_QUICK_REF.md).

## 📚 Documentation

- **[Menu System User Guide](archive/production_cleanup_2026-03-03/repo/docs/MENU_SYSTEM_USER_GUIDE.md)**: Interactive menu and E2E testing
- **[Menu System Flowchart](archive/production_cleanup_2026-03-03/repo/docs/MENU_SYSTEM_FLOWCHART.md)**: Comprehensive workflow diagrams
- **Design Bible**: Complete system architecture and mathematical proofs
- **API Reference**: Detailed function documentation
- **Deployment Guide**: Production setup and monitoring
- **Research Papers**: Theorem validation and empirical results
- **[Branching Strategy](archive/production_cleanup_2026-03-03/repo/docs/BRANCHING_STRATEGY.md)**: Git workflow and branch management guide
- **[Scientific Testing Guide](archive/production_cleanup_2026-03-03/repo/docs/SCIENTIFIC_TESTING_GUIDE.md)**: Comprehensive testing framework

Visit the [GitHub Wiki](https://github.com/sandman9988/Kinetra/wiki) or [GitHub Pages](https://sandman9988.github.io/Kinetra/) for full documentation.

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (see [Branching Strategy](archive/production_cleanup_2026-03-03/repo/docs/BRANCHING_STRATEGY.md))
3. Add comprehensive tests
4. Ensure all CI checks pass
5. Submit a pull request with theorem validation

For detailed branch management instructions and git workflow, see the [Branching Strategy Guide](archive/production_cleanup_2026-03-03/repo/docs/BRANCHING_STRATEGY.md). You can also use the branch management helper script:

```bash
# Set up local main branch tracking remote
python scripts/branch_manager.py --setup

# Check branch status
python scripts/branch_manager.py --status

# Sync with remote
python scripts/branch_manager.py --sync
```

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

This project is open-source for research and educational purposes. Use in live trading is at your own risk.

## ⚠️ Disclaimer

**IMPORTANT**: This software is provided for educational and research purposes only. Trading financial instruments carries significant risk of loss. Past performance does not guarantee future results. The authors are not responsible for any financial losses incurred through use of this software.

## 🌟 Acknowledgments

Built on first principles with inspiration from:
- Statistical mechanics and thermodynamics
- Reinforcement learning theory (Sutton & Barto)
- Quantitative finance research
- Open-source ML/trading community

## 📞 Contact

- **GitHub Issues**: Bug reports and feature requests
- **Discussions**: Design questions and research collaboration
- **Email**: [Your email for serious inquiries]

---

**Kinetra** - *Harvesting Energy from Market Physics* 🚀
