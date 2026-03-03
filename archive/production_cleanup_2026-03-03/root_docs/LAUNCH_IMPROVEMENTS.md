# Live Trading Launcher Improvements
**Date:** 2026-03-02

## Issues Fixed

### 1. ✅ Incorrect Equity Display
**Problem:** The live trading screen showed hardcoded `$10,000` equity but the actual account balance was `$48,589.59` (shown in preflight checks).

**Fix:** Reordered the `stage_live()` function to:
1. Connect to broker FIRST
2. Run preflight checks to get real account balance
3. Update `cfg.initial_equity` with real balance
4. THEN show settings with correct equity

**Before:**
```
Equity:      $10,000
```

**After:**
```
Equity:      $48,589.59
```

**File:** `scripts/renko_engine.py` - `stage_live()` function

---

### 2. ✅ No Entry Evaluation Feedback
**Problem:** User couldn't see if the engine was actively checking for trade opportunities during live trading.

**Fixes Applied:**

#### A. Added Entry Evaluation Logging
Every time a brick flip occurs, the engine now logs:
- Direction (BUY/SELL)
- FlipRate and Markov values
- Pass/Fail status
- Current price

**Example log output:**
```
Entry eval [BUY]: FR=0.320 M=0.620 → ✓ PASS  (price=2654.50)
  → Opening BUY 0.486 lots @ 2654.50

Entry eval [SELL]: FR=0.420 M=0.480 → ✗ FAIL  (price=2651.30)
```

**File:** `kinetra/renko/trading_engine.py` - `_process_brick()` method

#### B. Enhanced Live Stats Panel
The stats panel now shows:
- Number of bricks formed
- Time since last brick

**File:** `scripts/renko_engine.py` - `_LiveStatsPanel` class

#### C. Added Startup Monitoring Info
When live trading starts, the console now shows:
```
📊 Monitoring XAUUSD M1 bars → Renko bricks ($1.00)
🎯 Evaluating entries: FlipRate < 0.35, Markov > 0.55
⏱️  Stats refresh every 30s  •  Entry checks logged to file
```

**File:** `scripts/renko_engine.py` - `stage_live()` function

---

## Complete Launch Flow

```bash
# Run the interactive launcher
./scripts/ctrader/launch.sh
```

**User sees:**
1. Mode selection (paper/dry-run/LIVE)
2. Symbol selection (XAUUSD/NAS100/etc)
3. Gate selection (micro/small/full) - for LIVE mode only
4. Settings summary with:
   - Brick size
   - Stop loss
   - Session times
   - Loss cluster brake settings
   - **Real account equity** ✅ FIXED
   - Trailing MAE settings
   - Order gate status
5. Live risk acknowledgment (type `I_UNDERSTAND_LIVE_RISK`)
6. Preflight checks (TCP, auth, account, symbol)
7. **Entry monitoring info** ✅ NEW
8. Live stats panel with periodic updates
9. **Entry evaluation logs** ✅ NEW (in log file)

---

## Log File Location

All entry evaluations are logged to:
```
outputs/logs/<SYMBOL>_<YYYYMMDD>.log
```

Example:
```
outputs/logs/XAUUSD_20260302.log
```

**Log entries include:**
- Bar timestamps
- Brick formations
- Entry evaluations (filter values + pass/fail)
- Trade opens/closes
- Equity updates

---

## Technical Details

### Equity Fix
**Changed execution order in `stage_live()`:**
```python
# OLD: Show settings → Connect → Preflight → Update equity (too late!)
# NEW: Connect → Preflight → Update equity → Show settings ✓
```

### Entry Feedback
**Added at 3 levels:**
1. **Log file** - Detailed entry eval every brick flip
2. **Console** - Startup info showing filter thresholds
3. **Stats panel** - Brick count and last activity time

**Location in code:**
- Entry logging: `kinetra/renko/trading_engine.py:430-445`
- Console output: `scripts/renko_engine.py:760-762`
- Stats enhancement: `scripts/renko_engine.py:280-305`

---

## Testing

To verify the fixes:

```bash
# 1. Run dry-run mode (connects to live bars, paper orders)
./scripts/ctrader/launch.sh
# Select: 2 (dry-run) → XAUUSD → [Enter]

# 2. Check equity is correct (should match preflight balance)
# 3. Watch console for brick monitoring messages
# 4. Check log file for entry evaluations:
tail -f outputs/logs/XAUUSD_$(date +%Y%m%d).log
```

---

## Files Modified

1. **scripts/renko_engine.py**
   - `stage_live()` - Reordered to show correct equity
   - `_LiveStatsPanel` - Added brick count tracking
   - Console output - Added monitoring info

2. **kinetra/renko/trading_engine.py**
   - `_process_brick()` - Added entry evaluation logging

---

## Compliance with Copilot Instructions

✅ **No magic numbers** - All thresholds from config  
✅ **DRY principle** - Used existing `evaluate_entry()` function  
✅ **First principles** - Shows actual filter values, not derived metrics  
✅ **Rich/console** - Used existing Rich setup, no new dependencies  
✅ **Logging** - Used existing `kinetra.renko` logger  
✅ **Type hints** - Maintained in all modified functions  

---

## Next Steps (Future Enhancements)

**Potential improvements:**
1. Add brick formation sound alert (optional)
2. Show current FlipRate/Markov in live stats panel
3. Add "bars since last brick" counter
4. Optional Telegram/Discord notifications for entries
5. Real-time chart overlay (separate window)

**Note:** These are suggestions only - current implementation provides all essential feedback.
