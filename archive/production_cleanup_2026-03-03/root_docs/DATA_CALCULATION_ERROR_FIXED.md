# Data Calculation Error - FIXED
**Date:** 2026-03-02  
**Issue:** DSP profile reported 10.6 years of XAUUSD M1 data - calculation error

---

## The Error

**Old DSP profile stated:**
```json
"n_bars": 128161,
"years": 10.595320767195767
```

**Calculation error:**
- Used `bars_per_hour = 2.0` (M30 setting)
- 128,161 bars / (2 × 24 × 252) = 128,161 / 12,096 = **10.595 years** ❌ WRONG

---

## The Fix

**Correct calculation with M1 data:**
- Correct `bars_per_hour = 60.0` (M1 setting)
- 128,161 bars / (60 × 24 × 252) = 128,161 / 362,880 = **0.353 years** ✓ CORRECT

**What this means:**
- 0.353 years × 365 days/year = **129 calendar days**
- 0.353 years × 252 trading days/year = **89 trading days** ≈ **~4.2 months**

---

## Root Cause

The DSP was run with **M1 data but using M30's `bars_per_hour` default value**.

**Old code (BUGGY):**
```python
closes = load_m1_data(symbol)  # M1 data
dsp_result = run_dsp(closes, symbol)  # bars_per_hour defaults to 2.0 (M30!)
```

**New code (FIXED):**
```python
closes = load_m1_data(symbol)  # M1 data  
dsp_result = run_dsp(closes, symbol, bars_per_hour=60.0)  # Correct timeframe!
```

---

## Files Updated

1. **data/master_standardized/ctrader/pepperstone/metals/XAUUSD/dsp_profile.json**
   - Changed `"years"` from `10.595320767195767` to `0.35324754098360654`
   - Added note explaining the correction

---

## Implication for 3-Month Backtest

**Good news:** We have EXACTLY ~4.2 months of data, so:
- **3-month backtest** ✓ Fully possible (89 days available, need 90 days)
- **Full 3-year backtest** ❌ NOT possible (only 4.2 months available)

**Backtest command should be:**
```bash
python scripts/renko_engine.py XAUUSD --stage backtest --months 3
```

This will use ~90 days of the available ~129 calendar days of data.

---

## Timeline

- **Data covers:** ~4.2 months (approximately 89-90 trading days)
- **Start:** Unknown (need to check CSV first/last timestamps)
- **End:** Unknown (need to check CSV first/last timestamps)
- **Bars:** 128,161 M1 bars

---

## Verification Math

```
1 year = 252 trading days = 362,880 M1 bars (60 bars/hr × 24 × 252)
0.353 years = 89 trading days = 128,161 M1 bars ✓

3 months = 63 trading days = 90,720 M1 bars
Available = 128,161 M1 bars ✓ (enough for 3+ month backtest)
```

---

## Summary

- **Error:** Calculation was off by ~30× (10.6 years vs 0.35 years)
- **Cause:** Using M30's `bars_per_hour=2.0` with M1 data
- **Fix:** Use correct `bars_per_hour=60.0` for M1
- **Status:** ✅ CORRECTED in DSP profile JSON
- **Result:** ~4.2 months of data available, sufficient for 3-month backtest

---

**Next step:** Run `python scripts/renko_engine.py XAUUSD --stage backtest --months 3`
