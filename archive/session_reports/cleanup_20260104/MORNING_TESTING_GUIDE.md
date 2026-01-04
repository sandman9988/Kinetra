# KINETRA MORNING TESTING GUIDE
================================

**Quick Start for Real-World Testing - 2026-01-04**

---

## PRE-FLIGHT CHECKLIST

Before you start:

- [ ] Python environment activated (`.venv` or conda)
- [ ] Terminal ready
- [ ] MetaAPI credentials handy (if using)
- [ ] Coffee ☕

---

## FASTEST PATH TO TESTING (5 Minutes)

```bash
# 1. Start the production menu
python kinetra_production_menu.py

# 2. Check what data you already have
Main Menu → 2 → 1
# This discovers existing data files

# 3. Run a quick backtest on existing data
Main Menu → 4 → 1
# Select any symbol/timeframe that shows as "usable"

# 4. View results
Main Menu → 4 → 5
```

**Expected Output**:
```
✅ Results from batch_backtest_results.csv:
  symbol    omega_train  win_train  trigger    mfe    chs
  BTCUSD    3.2          62.5%      log_E_t    0.68   0.85
```

---

## COMPLETE WORKFLOW (30 Minutes)

### Step 1: System Check (2 min)

```bash
python kinetra_production_menu.py

# Menu: 5 → 1 (System Status & Diagnostics)
```

**What to Look For**:
```
✅ Data ready | ✅ Creds | ✅ MetaAPI | ✅ GPU
💡 Next Step: <guided suggestion>
```

### Step 2: Setup Credentials (3 min) - SKIP IF ALREADY DONE

```bash
# Menu: 1 → 1 (Configure MetaAPI Credentials)
```

**You'll Need**:
- MetaAPI Token (from https://app.metaapi.cloud)
- Account ID (from your MetaAPI dashboard)

**What Happens**:
- Credentials saved to `.env`
- Encrypted storage (secure)
- Verified with API call

### Step 3: Discover Existing Data (1 min)

```bash
# Menu: 2 → 1 (Discover Available Data)
```

**Output**:
```
Found 15 unique symbols:
  BTCUSD: 3 timeframes (H1, H4, D1) - 26,280 total bars
  EURUSD: 2 timeframes (H1, H4) - 17,520 total bars
  ...

✅ USABLE (12/15 combinations with >=1000 bars)

📄 Exported to: data/available_data.json
```

### Step 4: Download Fresh Data (5-10 min) - OPTIONAL

```bash
# Menu: 2 → 2 (Download Data - MetaAPI)
# OR
# Menu: 2 → 3 (Download Data - MT5 Local)
```

**Recommended for Testing**:
- **Crypto**: BTCUSD, ETHUSD (high volatility, clear signals)
- **Forex**: EURUSD, GBPUSD (liquid, reliable)
- **Timeframes**: H1 (best for testing), H4 (slower)
- **Date Range**: Last 2 years (2023-2024)

### Step 5: Quick Backtest (10 min)

```bash
# Menu: 4 → 1 (Quick Backtest)
```

**Interactive Prompts**:
```
Symbol (default: BTCUSD): BTCUSD
Timeframe (default: H1): H1

Backtesting BTCUSD H1...
```

**What It Does**:
1. Loads BTCUSD H1 data
2. Splits 70/30 (train/test)
3. Applies non-linear prep (log-returns, median filtering)
4. Trains SuperPot PPO agent
5. Evaluates on train set (Omega, Win%)
6. Evaluates on out-of-sample set
7. Checks survival criteria:
   - Omega > 2.7 ✅
   - Win rate > 55% ✅
   - OOS drop < 5% ✅
   - KS test p > 0.05 ✅
8. Computes triggers, harvesters, risk
9. Saves results

**Success Looks Like**:
```
✅ Batch backtest complete: Results in data/batch_backtest_results.csv

BTCUSD 2023: Ω_train=3.2, Win=62.5%, drop=2.1%, KS_p=0.12 | Survives: True
```

### Step 6: View Results (2 min)

```bash
# Menu: 4 → 5 (View Backtest Results)
```

**Key Metrics**:
- **Omega Ratio**: > 2.7 = PASS (asymmetric returns)
- **Win Rate**: > 55% = PASS (consistency)
- **OOS Drop**: < 5% = PASS (generalization)
- **CHS**: > 0.90 = EXCELLENT system health

---

## WHAT TO TEST FOR REAL-WORLD READINESS

### 1. Data Integrity ✅

```bash
# Menu: 2 → 4 (Check Data Integrity)
```

**Pass Criteria**:
- No NaN values
- No duplicate timestamps
- Chronological order
- Realistic price ranges

### 2. Single Symbol Backtest ✅

```bash
# Menu: 4 → 1 (Quick Backtest)
# Test: BTCUSD H1
```

**Pass Criteria**:
- Omega > 2.7
- Win rate > 55%
- No crashes/errors
- Results saved successfully

### 3. Multi-Symbol Batch ✅

```bash
# Menu: 4 → 2 (Batch Backtest)
# Test: BTCUSD, EURUSD, ETHUSD (3 symbols)
```

**Pass Criteria**:
- All 3 symbols complete
- At least 1 survives criteria
- Results CSV contains all rows
- No memory leaks

### 4. Monte Carlo Validation (OPTIONAL - Takes Time) ⏱️

```bash
# Menu: 4 → 3 (Monte Carlo Validation)
# 100 runs × 3 symbols = ~1 hour
```

**Pass Criteria**:
- Mean Omega > 2.7
- 95% CI above threshold
- p-value < 0.01

---

## TROUBLESHOOTING QUICK FIXES

### Problem: "No data found"

**Solution**:
```bash
# Check if files exist
ls data/master_standardized/*.csv

# If empty, download some
# Menu: 2 → 2
# Choose: BTCUSD, H1, 2023-01-01 to 2024-12-31
```

### Problem: "MetaAPI connection failed"

**Solution**:
```bash
# Re-run credential setup
# Menu: 1 → 1

# Verify token at: https://app.metaapi.cloud
# Check account ID is correct
```

### Problem: "Training fails with NaN"

**Solution**:
```bash
# Check data integrity first
# Menu: 2 → 4

# If data is corrupt, re-download
# Menu: 2 → 2
```

### Problem: "Script interrupted"

**Recovery**:
- Press Ctrl+C once (graceful shutdown)
- Re-run the menu
- Menu system will detect previous state
- Safe to continue where you left off

---

## EXPECTED TIMINGS

| Task | Time | Notes |
|------|------|-------|
| System check | 30s | Instant status |
| Data discovery | 1m | Scans filesystem |
| Download 1 symbol/TF | 2-5m | Depends on bars |
| Quick backtest | 5-10m | Includes training |
| Batch backtest (3 symbols) | 15-30m | Parallel if GPU |
| Monte Carlo (100 runs) | 1-2h | Statistical validation |

---

## SUCCESS INDICATORS

### Green Flags ✅

- **Omega > 2.7**: System is capturing asymmetric returns
- **Win rate > 55%**: Consistent edge
- **OOS drop < 5%**: Model generalizes well
- **No errors in logs**: Clean execution
- **CHS > 0.90**: Excellent system health

### Yellow Flags ⚠️

- **Omega 2.0-2.7**: Marginal, needs optimization
- **Win rate 50-55%**: Weak edge
- **OOS drop 5-10%**: Some overfitting
- **Warnings in logs**: Review but not critical

### Red Flags 🚨

- **Omega < 2.0**: No edge, STOP
- **Win rate < 50%**: Losing strategy
- **OOS drop > 10%**: Severe overfitting
- **Errors/crashes**: Fix before continuing
- **CHS < 0.55**: Circuit breaker, halt trading

---

## WHAT TO REPORT

### Minimum Test Report

```
Date: 2026-01-04
Time: 09:00 AM
System: Kinetra Production Menu v1.0

DATA:
- Symbols tested: BTCUSD, EURUSD
- Timeframes: H1
- Bars: 8,760 (BTCUSD), 8,760 (EURUSD)
- Date range: 2023-01-01 to 2023-12-31

RESULTS:
- BTCUSD: Omega=3.2, Win=62.5%, CHS=0.85 ✅
- EURUSD: Omega=2.9, Win=58.1%, CHS=0.82 ✅

ISSUES:
- None

STATUS: READY FOR EXTENDED TESTING
```

### Extended Test Report (If Time Permits)

Include:
- Monte Carlo statistics (mean, std, CI)
- Agent comparison results (PPO vs DQN)
- Measurement impact analysis
- Plots (equity curve, drawdown)

---

## NEXT STEPS AFTER MORNING TESTING

### If All Tests Pass ✅

1. **Extended Backtest**: All symbols, all timeframes
2. **Walk-Forward Analysis**: Rolling window validation
3. **Live Virtual Trading**: Paper trading mode
4. **Demo Account**: Small real money test

### If Issues Found ⚠️

1. **Review logs**: `Menu → 5 → 6`
2. **Check diagnostics**: `Menu → 5 → 1`
3. **Re-download data**: `Menu → 2 → 2`
4. **Run tests**: `Menu → 5 → 5`

---

## QUICK COMMAND REFERENCE

```bash
# Start menu
python kinetra_production_menu.py

# Direct script access (if needed)
python scripts/discover_available_data.py
python scripts/batch_backtest.py --symbols BTCUSD --tf H1
python scripts/training/train_rl.py --episodes 100

# Check logs
tail -f logs/batch_backtest.log

# View results
cat data/batch_backtest_results.csv
```

---

## FILES TO CHECK AFTER TESTING

✅ **Generated Files**:
- `data/available_data.json` (discovery output)
- `data/batch_backtest_results.csv` (backtest results)
- `logs/batch_backtest.log` (execution log)
- `.env` (credentials - check it exists)

✅ **Results to Review**:
- Omega ratio per symbol
- Win rate consistency
- OOS degradation
- System health score

---

## SAFETY GATES

Before ANY real money:

1. ✅ Monte Carlo validation (100+ runs)
2. ✅ Statistical significance (p < 0.01)
3. ✅ Out-of-sample validation
4. ✅ Walk-forward analysis
5. ✅ Risk-of-Ruin < 1%
6. ✅ Composite Health Score > 0.90
7. ✅ Live virtual trading (30+ days)
8. ✅ Demo account (90+ days)

**ONLY THEN** consider micro-lot real money.

---

## EMERGENCY CONTACTS

- **Documentation**: `WORKFLOW_DATA_PATHS.md`
- **Design Rules**: `AGENT_RULES_MASTER.md`
- **Troubleshooting**: `docs/TROUBLESHOOTING.md`
- **Testing Guide**: `docs/TESTING_FRAMEWORK.md`

---

**Good luck with your morning testing! 🚀**

Remember: 
- Physics-first, NO assumptions
- Question everything
- Let the data guide you
- Statistical validation is mandatory
- Safety gates are non-negotiable

**Start small, validate thoroughly, scale carefully.**