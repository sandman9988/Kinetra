# Kinetra Quick Reference Card

**Last Updated**: 2026-01-04  
**Status**: ✅ ALL SYSTEMS OPERATIONAL

---

## 🚀 Quick Start (First Time Setup Complete)

Your MetaAPI credentials are validated and the system is ready to use!

### Immediate Next Step: Download Data

```bash
python scripts/download/download_market_data.py --instrument BTCUSD --timeframe H1
```

---

## 📋 Common Commands

### Data Operations

```bash
# Test MetaAPI connection
python scripts/download/test_metaapi_connection.py

# Download market data
python scripts/download/download_market_data.py --instrument BTCUSD --timeframe H1

# Discover available instruments/timeframes
python scripts/download/discover_available_data.py

# View data coverage
python scripts/data/view_data_coverage.py
```

### Training & Testing

```bash
# Quick RL training
python scripts/training/quick_rl_training.py

# Batch backtest (100 Monte Carlo runs)
python scripts/batch_backtest.py --instrument BTCUSD --timeframe H1 --runs 100

# Single backtest
python scripts/run_backtest.py --instrument BTCUSD --timeframe H1
```

### Menu System

```bash
# Production menu (main interface)
python kinetra_production_menu.py

# Testing menu
python scripts/testing/test_menu.py
```

---

## 🔧 Development Commands

```bash
# Setup development environment
make setup
# or
pip install -e ".[dev]"

# Run tests
make test
pytest tests/ -v

# Code quality
make lint      # Run Ruff linter
make format    # Format with Black

# Coverage report
pytest tests/ --cov=kinetra --cov-report=html
```

---

## 📊 Project Structure

```
Kinetra/
├── kinetra/                    # Core package
│   ├── physics/                # Physics engine (energy, entropy)
│   ├── rl/                     # Reinforcement learning
│   ├── risk/                   # Risk management
│   ├── execution/              # Order execution
│   ├── menu_state_tracker.py   # Workflow state tracking
│   └── exploration_lab/        # Autonomous experiments
├── scripts/                    # Executable scripts
│   ├── download/               # Data download tools
│   ├── data/                   # Data analysis
│   ├── training/               # Model training
│   ├── testing/                # Testing tools
│   └── batch_backtest.py       # Monte Carlo validation
├── data/                       # Data storage
│   ├── master/                 # Primary data files
│   ├── backups/                # Automatic backups
│   └── menu_state.json         # Workflow state
├── tests/                      # Test suite
├── docs/                       # Design documentation
└── .env                        # Credentials (NOT in git)
```

---

## 🎯 Typical Workflow

### 1. Setup (✅ COMPLETE)
- [x] Install dependencies
- [x] Configure MetaAPI credentials
- [x] Validate connection

### 2. Data Preparation (← YOU ARE HERE)
```bash
# Download data
python scripts/download/download_market_data.py --instrument BTCUSD --timeframe H1

# Verify coverage
python scripts/data/view_data_coverage.py
```

### 3. Training
```bash
# Quick training run
python scripts/training/quick_rl_training.py

# Full training with monitoring
python scripts/training/train_rl_agent.py --instrument BTCUSD --timeframe H1
```

### 4. Backtesting
```bash
# Monte Carlo validation (100 runs)
python scripts/batch_backtest.py --instrument BTCUSD --timeframe H1 --runs 100
```

### 5. Production Deployment
```bash
# Access production menu for deployment options
python kinetra_production_menu.py
```

---

## 🔍 Troubleshooting

### MetaAPI Connection Issues

**Check environment variables**:
```bash
env | grep -i metaapi  # Should show nothing (credentials loaded from .env)
```

**If you see placeholder variables**:
```bash
unset METAAPI_TOKEN
unset METAAPI_ACCOUNT_ID
```

**Test connection**:
```bash
python scripts/download/test_metaapi_connection.py
```

**Diagnostic tool**:
```bash
python scripts/fix_metaapi_env.py
```

### Data Issues

**Check data directory**:
```bash
ls -lh data/master/
```

**View coverage**:
```bash
python scripts/data/view_data_coverage.py
```

**Re-download if needed**:
```bash
python scripts/download/download_market_data.py --instrument BTCUSD --timeframe H1 --force
```

### Test Failures

**Run specific test**:
```bash
pytest tests/test_physics.py -v
```

**Run with debugging**:
```bash
pytest tests/test_physics.py -v -s --pdb
```

**Check diagnostics**:
```bash
pytest tests/ --collect-only  # List all tests
pytest tests/ -v --tb=short   # Short traceback format
```

---

## 📖 Documentation

### Key Documents
- **Complete Rules**: `AGENT_RULES_MASTER.md` - Comprehensive agent guidelines
- **Session Status**: `SESSION_COMPLETION_STATUS.md` - Current session summary
- **Quick Start**: `QUICK_START_WORKFLOW.md` - Step-by-step workflow
- **Copilot Instructions**: `.github/copilot-instructions.md` - AI assistant rules

### Design Documentation
- **Design Bible**: `docs/` directory - Complete architecture
- **Theorem Proofs**: `docs/theorem_proofs.md` - Mathematical validation
- **Empirical Theorems**: `docs/EMPIRICAL_THEOREMS.md` - Data-driven discoveries
- **Testing Framework**: `docs/TESTING_FRAMEWORK.md` - Testing guidelines

---

## 🎯 Performance Targets

| Metric | Target | Purpose |
|--------|--------|---------|
| **Omega Ratio** | > 2.7 | Asymmetric returns |
| **Z-Factor** | > 2.5 | Statistical edge significance |
| **% Energy Captured** | > 65% | Physics alignment efficiency |
| **Composite Health Score** | > 0.90 | System stability |
| **% MFE Captured** | > 60% | Execution quality |

---

## 🔐 Security Checklist

- [x] `.env` file in `.gitignore`
- [x] No credentials in version control
- [x] PersistenceManager atomic saves enabled
- [x] Backup directory configured
- [x] Environment variables cleared

---

## 🚦 Current Status

**MetaAPI**: ✅ Connected and validated  
**Account**: ✅ DEPLOYED (ID: e8f8c21a-32b5-40b0-9bf7-672e8ffab91f)  
**Credentials**: ✅ Working from `.env` file  
**Environment**: ✅ Clean (no variable conflicts)  
**Data Pipeline**: ✅ Ready to download  
**Training**: ✅ Ready to train  
**Testing**: ✅ Operational  

**Blockers**: NONE ✅

---

## ⚡ Quick Commands Cheatsheet

```bash
# Environment
env | grep -i metaapi          # Check env vars (should be empty)
source .env                    # Load credentials (usually automatic)

# Data
python scripts/download/download_market_data.py --instrument BTCUSD --timeframe H1
python scripts/download/discover_available_data.py
python scripts/data/view_data_coverage.py

# Training
python scripts/training/quick_rl_training.py
python scripts/training/train_rl_agent.py --instrument BTCUSD --timeframe H1

# Testing
make test                      # Run all tests
pytest tests/ -v               # Verbose test output
make lint                      # Check code quality
make format                    # Format code

# Backtesting
python scripts/batch_backtest.py --instrument BTCUSD --timeframe H1 --runs 100

# Menus
python kinetra_production_menu.py    # Main menu
python scripts/testing/test_menu.py  # Testing menu

# Diagnostics
python scripts/fix_metaapi_env.py                    # Diagnose env issues
python scripts/download/test_metaapi_connection.py   # Test connection
```

---

## 🎓 Core Principles

1. **No Magic Numbers**: Use rolling percentiles, adaptive thresholds
2. **Physics First**: Energy, entropy, friction guide all decisions
3. **Statistical Validation**: p < 0.01 for all claims
4. **Vectorization**: NumPy/Pandas ops over Python loops
5. **Defense in Depth**: Multiple validation layers
6. **Zero Assumptions**: Let the data guide decisions

---

## 📞 Need Help?

1. Check `SESSION_COMPLETION_STATUS.md` for detailed status
2. Run diagnostic: `python scripts/fix_metaapi_env.py`
3. Test connection: `python scripts/download/test_metaapi_connection.py`
4. Review `AGENT_RULES_MASTER.md` for complete guidelines
5. Check `docs/` for design documentation

---

**You're ready to proceed!** Start with downloading data. 🚀

---

## 📊 Smart Download Menu (NEW!)

### Quick Start
```bash
python scripts/download/smart_download_menu.py
```

### Menu Options Summary
```
1. Quick Start      - Top 5 per class (~50 files, 5-10 min) ⭐ RECOMMENDED
2. Standard         - Top 10 per class (~100 files, 10-20 min)
3. Extended         - Top 20 per class (~200 files, 20-40 min)
4. Full Download    - All symbols (~900+ files, 2-4 hours)
5-9. Single Class   - Forex/Crypto/Indices/Metals/Energy only
10. Custom          - Fine-grained control
11. Resume          - Continue interrupted download
12. Status          - View current download status
```

### Auto Data Prep
- **Enabled by default** - data prep runs in parallel with downloads
- Start training as soon as first symbols complete
- Physics features extracted (energy, entropy, momentum)
- Quality validation (only high-quality data ready for training)

### Workflow Example
```
1. Run: python scripts/download/smart_download_menu.py
2. Select: 1 (Quick Start)
3. Wait: 5-10 minutes
4. Result: 50 files ready for training
5. Start training immediately!
```

---
