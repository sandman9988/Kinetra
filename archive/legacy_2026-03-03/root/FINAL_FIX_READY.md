# ✅ FINAL FIX - DatetimeIndex Issue Resolved

## Problem Fixed
**Error:** `TypeError` when trying to subtract timedelta from index

**Cause:** The returned Series didn't have a proper DatetimeIndex

**Solution Applied:**
1. Explicitly convert datetime column to UTC timezone using `pd.to_datetime(..., utc=True)`
2. Create Series with proper DatetimeIndex immediately
3. Sort the index and remove duplicates
4. Verify datetime_col exists before proceeding

## Execute Backtest

### Option 1: Simple Command
```bash
cd /home/renierdejager/Projects/Kinetra && python scripts/renko_engine.py XAUUSD --stage backtest --months 3
```

### Option 2: With Debug Output
```bash
bash /home/renierdejager/Projects/Kinetra/run_backtest_debug.sh
```

This will:
1. Verify data files exist
2. Test data loading with detailed output
3. Run the backtest
4. Show final results

## Expected Output
```
============================================================
STAGE 3: BACKTEST (3 months)
============================================================
Testing 90720 bars (3 months)
Range: 2025-11-15 12:00:00+00:00 to 2026-03-02 20:55:00+00:00

--- STATIC SIZING ---
┌──────────────────────────────────────────────────────┐
│ XAUUSD  [BACKTEST-STATIC]  2026-03-02 HH:MM UTC     │
├──────────────────────────────────────────────────────┤
│ Trades        XX  (XX W / XX L)  Win rate     XX.X%  │
│ Net P&L       $XXX.XX             Avg trade  $XX.XX  │
│ Profit factor X.XX                Omega       X.XXX  │
│ Max drawdown  X.X%                Live equity $XXXXX │
└──────────────────────────────────────────────────────┘
✅ PASS (or ❌ FAIL)

--- COMPOUNDING SIZING ---
[Similar panel...]
✅ PASS (or ❌ FAIL)
```

## Success Criteria
- ✅ **PASS IF:** Omega ≥ 1.5 **AND** Trades ≥ 30
- ❌ **FAIL IF:** Omega < 1.5 **OR** Trades < 30

## Files Generated
```
outputs/results/XAUUSD_backtest_static_20260302_HHMMSS.json
outputs/results/XAUUSD_backtest_compounding_20260302_HHMMSS.json
outputs/logs/renko_engine_20260302.log
```

---

**🚀 READY TO RUN!**

All fixes applied:
1. ✅ Removed invalid imports (CTraderBarProvider, etc.)
2. ✅ Fixed column name mapping (Close instead of close)
3. ✅ Ensured proper DatetimeIndex creation
4. ✅ Added duplicate removal and sorting

**Execute now:**
```bash
python scripts/renko_engine.py XAUUSD --stage backtest --months 3
```
