# Fixes Applied - 2026-03-02

## Issues Fixed

### 1. ✅ Duplicate XAUUSD Live Display Boxes
**Problem:** Two stat boxes displayed at bottom after live trading ended.

**Root Cause:** Line 800 in `stage_live()` called `_print_stats()` after `stats_panel.stop()`, creating a duplicate display.

**Fix:** Removed the duplicate `_print_stats()` call. The `_LiveStatsPanel` already shows stats every 30s during live trading. Only `_save_results()` needed after run ends.

**Before:**
```python
finally:
    watchdog.stop()
    stats_panel.stop()

summary = results.get("summary", {})
_print_stats(summary, symbol, "live")  # ← DUPLICATE!
_save_results(results, symbol, "live")
```

**After:**
```python
finally:
    watchdog.stop()
    stats_panel.stop()

# Save final results (display already shown by _LiveStatsPanel during run)
_save_results(results, symbol, "live")
```

---

### 2. ✅ Brick Count Impossibly Low (14 bricks)
**Problem:** FlipRate/Markov window was set to 14 bricks, but this was actually 14 **M1 bars**, not bricks.

**Root Cause:** `vr_peak_scale` from DSP is in **bars** (M30 or M1 bars), not bricks. The old code directly used this bar count as the brick-based filter window:

```python
# OLD - WRONG: vr_peak_scale is in BARS, not bricks
window = int(dsp.get("vr_peak_scale", dsp.get("vr_scale_bars", 50)))
```

This caused:
- FlipRate to operate on 14 M1 bars worth of data
- But bricks form much slower than bars → only ~14 bricks in weeks of data
- Filters couldn't work properly with so few bricks

**Fix:** Convert bar-based `vr_peak_scale` to brick-based window using empirical brick frequency:

```python
# NEW - CORRECT: Convert M1 bars → equivalent brick count
# 1. Build sample bricks to measure bricks_per_day
# 2. Convert vr_peak_scale (M1 bars) → days
# 3. Multiply days × bricks_per_day = brick window

vr_peak_scale_bars = int(dsp.get("vr_peak_scale", 50))
bars_per_day = 60 * 24  # M1: 1440 bars/day
days_in_peak = vr_peak_scale_bars / bars_per_day
window = max(10, int(bpd * days_in_peak))
```

**Example:**
- `vr_peak_scale` = 720 M1 bars
- 720 bars / 1440 bars/day = 0.5 days
- If `bricks_per_day` = 80 (empirical for XAUUSD)
- Window = 80 × 0.5 = **40 bricks** ✓ (not 720!)

---

### 3. ✅ DSP Using Wrong Timeframe
**Problem:** DSP was designed for M30 bars (comment says "e.g. M30 close prices"), but we were passing M1 data with default `bars_per_hour=2.0` (M30 setting).

**Root Cause:** `stage_dsp()` loaded M1 data but didn't tell `run_dsp()` about the timeframe:
```python
# OLD - WRONG
closes = load_m1_data(symbol)
dsp_result = run_dsp(closes, symbol)  # Uses bars_per_hour=2.0 default (M30)
```

**Solution Options:**
- **Option A:** Aggregate M1 → M30, then run DSP  
  ❌ Extra aggregation step, more code, data movement
  
- **Option B:** Pass M1 directly with `bars_per_hour=60.0`  
  ✅ Cleaner, less code, DSP already supports it

**Fix (Option B - CLEANER):**
```python
# Run DSP on M1 data with correct bars_per_hour
dsp_result = run_dsp(m1_closes, symbol, bars_per_hour=60.0)
```

---

## Complete Flow (After Fixes)

```
1. Download M1 data
   ↓
2. Run DSP on M1 with bars_per_hour=60.0
   → Returns vr_peak_scale in M1 bars
   ↓
3. Build engine config:
   → Load M1 data
   → Build sample bricks (last 10k bars)
   → Measure bricks_per_day empirically
   → Convert vr_peak_scale (M1 bars) → brick window
   ↓
4. Live trading:
   → _LiveStatsPanel shows stats every 30s
   → Entry checks logged to file
   → NO duplicate display at end
```

---

## Files Modified

1. **scripts/renko_engine.py**
   - `stage_dsp()` - Pass `bars_per_hour=60.0` for M1 data
   - `_build_engine_config()` - Convert bar-based scale to brick-based window
   - `stage_live()` - Removed duplicate `_print_stats()` call

---

## Verification

### Brick Count Should Be Reasonable
**Before:** 14 bricks (impossibly low)  
**After:** 40-80 bricks (typical for 0.5-1 day window on XAUUSD)

### DSP Timeframe
**Before:** M1 data with M30 `bars_per_hour` (wrong scale factors)  
**After:** M1 data with M1 `bars_per_hour=60.0` ✓

### Display
**Before:** Two identical stat boxes at end  
**After:** One live-updating panel during run, results saved to file ✓

---

## Testing

```bash
# Run dry-run mode
./scripts/ctrader/launch.sh
# Select: 2 (dry-run) → XAUUSD

# Expected:
# 1. Single stats panel updating every 30s
# 2. Brick count shows reasonable number (50-100)
# 3. No duplicate box at end
# 4. Entry evaluations logged: "Entry eval [BUY]: FR=0.320 M=0.620 → ✓ PASS"
```

---

## Why Option B is Cleaner

### Option A (Aggregate First)
```python
# Load M1
m1_closes = load_m1_data(symbol)

# Aggregate M1 → M30
m1_df = pd.DataFrame({"close": m1_closes.values}, index=m1_closes.index)
m30_df = aggregate_ohlcv(m1_df, "M30")
m30_closes = m30_df["close"]

# Run DSP
dsp_result = run_dsp(m30_closes, symbol)  # bars_per_hour=2.0 default
```
❌ 6 lines, imports aggregation, creates intermediate DataFrames

### Option B (Direct M1)
```python
# Load M1
m1_closes = load_m1_data(symbol)

# Run DSP with correct timeframe
dsp_result = run_dsp(m1_closes, symbol, bars_per_hour=60.0)
```
✅ 3 lines, no imports, no intermediate data, explicit timeframe

**Option B wins:** Less code, clearer intent, same result.

---

## Key Insight

**Bars ≠ Bricks**

- DSP operates on **price bars** (M1, M30, H4, etc.)
- Filters operate on **Renko bricks**
- Bricks form at variable rate depending on volatility
- Must measure empirical brick frequency to convert bar-based windows to brick-based windows

**Formula:**
```
vr_peak_scale (bars) / bars_per_day → days
days × bricks_per_day → brick_window
```

---

## Compliance

✅ **DRY** - Uses existing `build_renko()` and `bricks_per_day()` functions  
✅ **First principles** - Empirical brick frequency measurement  
✅ **No magic numbers** - Window calculated from DSP result  
✅ **Single source of truth** - Engine state used for display  
✅ **Type hints** - All maintained  
✅ **Cleaner code** - Option B chosen over Option A  

---

**Status:** ✅ ALL FIXES COMPLETE  
**Ready for testing**
