# Kinetra Quick Start Workflow
**Get from Zero to Trading in 5 Steps**

---

## 🚀 Quick Reference

```bash
# 1. Fix environment (if needed)
python scripts/fix_metaapi_env.py

# 2. Test credentials
python scripts/download/test_metaapi_connection.py

# 3. Run production menu
python kinetra_production_menu.py

# 4. Check workflow status
python -m kinetra.menu_state_tracker status

# 5. Run comprehensive tests
python scripts/testing/test_all_menu_options.py --quick
```

---

## 📋 Prerequisites

### Required
- Python 3.10+
- MetaAPI account (free tier works)
- Git repository cloned

### Optional
- CUDA GPU (for faster training)
- MT5 terminal (local data source)

---

## 🎯 The 5-Step Workflow

### Step 1: Setup Credentials ⚙️

**Check Current Status**:
```bash
python scripts/fix_metaapi_env.py
```

**If Issues Found**:
```bash
# Option A: Unset environment variables
unset METAAPI_TOKEN METAAPI_ACCOUNT_ID

# Option B: Use helper script
source scripts/unset_metaapi_env.sh

# Option C: Open new terminal (cleanest)
```

**Configure Credentials**:
```bash
# Via menu
python kinetra_production_menu.py
# Select: 1 → 1 (Configure MetaAPI Credentials)

# Or direct script
python scripts/download/setup_metaapi_credentials.py
```

**Verify**:
```bash
python scripts/download/test_metaapi_connection.py
```

**Expected Output**:
```
✅ ACCOUNT FOUND
✅ Account is DEPLOYED and ready to use
✅ CONNECTION TEST PASSED
```

---

### Step 2: Download Data 📊

**Discover What's Available**:
```bash
python scripts/discover_available_data.py
```

**Download Data**:
```bash
# Bulk download (all asset classes)
python scripts/download/metaapi_bulk_download.py

# Or single instrument
python scripts/download/download_metaapi.py --symbol EURUSD --timeframe H1
```

**Progress Tracking**:
- Real-time progress bars (tqdm)
- ETA estimates
- Resume capability
- Atomic saves (no data loss)

**Verify Data**:
```bash
python scripts/download/check_data_integrity.py
```

---

### Step 3: Train Agent 🤖

**Quick Training** (Recommended First Run):
```bash
# Via menu
python kinetra_production_menu.py
# Select: 3 → 1 (Quick RL Training)

# Or direct script
python scripts/train_rl_agent.py --instrument EURUSD --timeframe H1
```

**Advanced Training**:
```bash
# SuperPot agent
python scripts/train_superpot_agent.py

# Agent comparison
python scripts/compare_agents.py

# Scientific exploration
python scripts/exploration/run_comprehensive_exploration.py
```

**Monitor Progress**:
```bash
# View training logs
tail -f logs/training.log

# Check composite health score
python scripts/monitor_composite_health.py
```

---

### Step 4: Backtest Strategy 📈

**Quick Backtest**:
```bash
python scripts/batch_backtest.py --instrument EURUSD --timeframe H1
```

**Batch Backtest** (Multiple Instruments):
```bash
python scripts/batch_backtest.py \
  --instruments EURUSD GBPUSD BTCUSD \
  --timeframes H1 H4 \
  --mc-runs 100
```

**Monte Carlo Validation**:
```bash
python scripts/run_monte_carlo_validation.py \
  --instrument EURUSD \
  --timeframe H1 \
  --runs 1000
```

**Performance Targets**:
- Omega Ratio > 2.7
- Z-Factor > 2.5
- % Energy Captured > 65%
- Composite Health Score > 0.90
- % MFE Captured > 60%

---

### Step 5: Monitor & Optimize 🔍

**Check Workflow Progress**:
```bash
python -m kinetra.menu_state_tracker status
```

**Output**:
```
=== Workflow Status ===

Breadcrumb: Setup ✅ → Data ✅ → Training 🔄 → Backtesting • → Deployment •

Overall Progress: 45/100 (45.0%)

Stage Progress:
  ✅ setup: 2/2 (100.0%)
  ✅ data: 4/4 (100.0%)
  🔄 training: 1/2 (50.0%)
  • backtesting: 0/3 (0.0%)
  • deployment: 0/1 (0.0%)

Next Step: 3.2
```

**Run Tests**:
```bash
# Quick tests
python scripts/testing/test_all_menu_options.py --quick

# Full test suite
python scripts/run_exhaustive_tests.py
```

**Generate Reports**:
```bash
# Performance analysis
python scripts/analyze_performance.py

# Risk analysis
python scripts/analyze_risk.py

# Regime analysis
python scripts/analyze_regimes.py
```

---

## 🔬 Exploration Lab (Advanced)

**What It Does**:
- Autonomous discovery of new measurements
- Agent architecture evolution
- Pattern mining
- Rigorous validation (p < 0.01)

**Start Exploration**:
```bash
python -m kinetra.exploration_lab.orchestrator continuous
```

**Check Discoveries**:
```bash
# View experiments
ls experiments/lab/experiments/

# View production candidates
ls experiments/lab/candidates/

# Get summary
python -m kinetra.exploration_lab.orchestrator status
```

**Validation Gates** (All Must Pass):
- p_value < 0.01
- omega_ratio > 2.7
- z_factor > 2.5
- monte_carlo_percentile > 95.0
- physics_alignment_score > 0.80
- composite_health_score > 0.90

---

## 🛠️ Troubleshooting

### MetaAPI Connection Fails

**Symptom**: `❌ Connection test failed`

**Fix**:
```bash
# 1. Run diagnostic
python scripts/fix_metaapi_env.py

# 2. Follow action plan output

# 3. Verify fix
python scripts/download/test_metaapi_connection.py
```

**Common Causes**:
- Environment variable override (placeholder values)
- Token expired (regenerate at MetaAPI)
- Account not deployed (deploy at MetaAPI dashboard)
- Network/firewall issues

---

### Missing Data

**Symptom**: `❌ No data found for EURUSD H1`

**Fix**:
```bash
# 1. Discover what's available
python scripts/discover_available_data.py

# 2. Download missing data
python scripts/download/metaapi_bulk_download.py

# 3. Verify integrity
python scripts/download/check_data_integrity.py
```

---

### Training Fails

**Symptom**: `❌ Training failed: NaN in rewards`

**Fix**:
```bash
# 1. Check data quality
python scripts/download/check_data_integrity.py

# 2. Validate physics calculations
python scripts/testing/test_physics_demo.py

# 3. Check logs
tail -100 logs/training.log

# 4. Reduce learning rate
python scripts/train_rl_agent.py --learning-rate 1e-5
```

---

### Backtest Results Poor

**Symptom**: Omega < 2.7, Z-Factor < 2.5

**Actions**:
1. Check if model trained sufficiently
2. Validate data quality and coverage
3. Check regime alignment
4. Review physics calculations
5. Run Monte Carlo to verify significance

**Debug**:
```bash
# Physics validation
python scripts/testing/test_physics_demo.py

# Numerical stability
python scripts/testing/test_numerical_safety.py

# Regime analysis
python scripts/analyze_regimes.py
```

---

## 📊 Workflow State Tracking

**View Status**:
```bash
python -m kinetra.menu_state_tracker status
```

**Mark Step Complete**:
```bash
python -m kinetra.menu_state_tracker complete 1.1
```

**Reset Progress**:
```bash
python -m kinetra.menu_state_tracker reset
```

**State File Location**: `data/menu_state.json`

---

## 🎯 Production Checklist

Before going live, ensure:

- [ ] **Credentials**: ✅ MetaAPI connection verified
- [ ] **Data**: 
  - [ ] ✅ Downloaded for target instruments
  - [ ] ✅ Integrity validated
  - [ ] ✅ Coverage > 90%
- [ ] **Training**:
  - [ ] ✅ Model trained (>100k steps)
  - [ ] ✅ Convergence achieved
  - [ ] ✅ Composite Health > 0.90
- [ ] **Backtesting**:
  - [ ] ✅ Omega > 2.7
  - [ ] ✅ Z-Factor > 2.5
  - [ ] ✅ Monte Carlo validated (p < 0.01)
  - [ ] ✅ Out-of-sample tested
- [ ] **Risk**:
  - [ ] ✅ Risk-of-Ruin < 1%
  - [ ] ✅ Position sizing validated
  - [ ] ✅ Circuit breakers configured
- [ ] **Monitoring**:
  - [ ] ✅ Health monitoring active
  - [ ] ✅ Alerts configured
  - [ ] ✅ Logging enabled

---

## 🔄 Continuous Workflow

**Daily**:
```bash
# 1. Update data
python scripts/download/metaapi_bulk_download.py

# 2. Check health
python scripts/monitor_composite_health.py

# 3. Review performance
python scripts/analyze_performance.py
```

**Weekly**:
```bash
# 1. Retrain models
python scripts/train_rl_agent.py

# 2. Run full backtest
python scripts/batch_backtest.py --mc-runs 1000

# 3. Update reports
python scripts/generate_reports.py
```

**Monthly**:
```bash
# 1. Archive old data
python scripts/download/backup_data.py

# 2. Run exploration lab
python -m kinetra.exploration_lab.orchestrator continuous

# 3. Promote validated discoveries
python scripts/promote_discoveries.py
```

---

## 📚 Additional Resources

### Documentation
- `README.md` - Project overview
- `AGENT_RULES_MASTER.md` - Complete agent rules
- `STATUS_REPORT_ALL_ACTIONS.md` - Latest system status
- `docs/` - Complete design documentation

### Scripts Reference
- `scripts/download/` - Data acquisition
- `scripts/training/` - Model training
- `scripts/backtesting/` - Strategy validation
- `scripts/testing/` - Test suite
- `scripts/exploration/` - Scientific discovery

### Key Modules
- `kinetra/physics_engine.py` - Physics calculations
- `kinetra/rl_agent.py` - RL agent implementations
- `kinetra/risk_management.py` - Risk controls
- `kinetra/menu_state_tracker.py` - Workflow tracking
- `kinetra/exploration_lab/` - Autonomous discovery

---

## 🆘 Getting Help

**Check Diagnostics**:
```bash
python scripts/fix_metaapi_env.py
```

**Run Health Check**:
```bash
python scripts/monitor_composite_health.py
```

**View Logs**:
```bash
# Recent errors
tail -100 logs/kinetra.log | grep ERROR

# Training progress
tail -f logs/training.log

# Backtest results
cat logs/backtest.log
```

**Test System**:
```bash
python scripts/testing/test_all_menu_options.py --quick
```

---

## 💡 Pro Tips

1. **Start Small**: Begin with one instrument (EURUSD H1)
2. **Validate Early**: Test credentials before bulk downloads
3. **Monitor Progress**: Use state tracker to stay on course
4. **Question Everything**: No magic numbers, validate assumptions
5. **Use Exploration Lab**: Let it discover patterns autonomously
6. **Check Health Often**: Composite Health Score is your north star
7. **Trust the Physics**: Energy, friction, entropy are real in markets
8. **Rigorous Testing**: p < 0.01 for statistical significance
9. **Monte Carlo Everything**: 100+ runs for validation
10. **Production First**: Consolidate, test, validate before exploring

---

## ✅ Success Criteria

**You're Ready When**:
- ✅ Credentials verified
- ✅ Data downloaded (>3 months per instrument)
- ✅ Model trained (Composite Health > 0.90)
- ✅ Backtest passing (Omega > 2.7, Z > 2.5)
- ✅ Monte Carlo validated (p < 0.01)
- ✅ All menu tests passing (>80%)

**Then**:
```bash
# Start live monitoring
python scripts/start_live_monitoring.py

# Begin paper trading
python scripts/start_paper_trading.py

# Eventually: Live trading (with extreme caution)
python scripts/start_live_trading.py --dry-run
```

---

## 🎓 Learning Path

**Week 1**: Setup & Data
- Configure credentials
- Download data for 3 instruments
- Validate data integrity
- Understand physics calculations

**Week 2**: Training
- Train first agent (EURUSD H1)
- Monitor composite health
- Understand RL rewards
- Study regime detection

**Week 3**: Backtesting
- Run single backtests
- Batch test multiple instruments
- Monte Carlo validation
- Analyze performance metrics

**Week 4**: Production
- Full system test
- Deploy monitoring
- Paper trading
- Continuous improvement

---

**Remember**: Kinetra is built on first principles with zero assumptions. Question everything, validate rigorously, and let the data guide you.

**Good luck! 🚀**