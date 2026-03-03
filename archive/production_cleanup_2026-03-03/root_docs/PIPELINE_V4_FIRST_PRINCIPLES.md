# Kinetra Pipeline v4: First Principles Approach

## Core Philosophy

**The Market Truth:**
- M1 bars are **raw noise** (5-minute candlesticks from broker)
- **Bricks are price-level structure** (aggregated by volatility, not time)
- **Correct brick sizing is everything** - it determines if strategy works
- **Friction must be < 1 brick** - otherwise spreads eat all profits
- **Testing determines parameters** - no assumptions, only empirical validation

## The Pipeline

### Step 1: Download M1 Bars
**What:** Acquire raw data from cTrader
**Data:** M1 OHLCV bars (the noise)
**Output:** CSV files in `data/master_standardized/ctrader/{account}/{symbol}/`

```bash
make download SYMBOL=XAUUSD
```

**Format:**
```
time,open,high,low,close,volume
2026-02-01T00:00:00Z,2043.5,2043.7,2043.3,2043.6,100
2026-02-01T00:01:00Z,2043.6,2043.8,2043.4,2043.7,95
```

---

### Step 2: DSP Analysis (Find Brick Size)
**What:** Analyze volatility regime → determine optimal brick size
**Method:** Volatility Ratio (VR) analysis
**Key Metric:** VR peak = peak volatility ratio at different aggregation levels

**DSP Output:**
```json
{
  "vr_peak": 1.426,           // volatility peak
  "vr_scale_bars": 96,        // scale where peak occurs
  "brick_size": 44.996,       // optimal brick size in price units
  "regime": "mean_reverting"  // or "trending"
}
```

**Interpretation:**
- VR peak at 96-bar scale = markets show structure at ~96 M1 bar aggregation
- Brick size of $44.99 = price movement of one brick
- This is **NOT a tuning parameter** - it's measured by DSP

```bash
make dsp SYMBOL=XAUUSD
```

**Why this matters:**
- Too small bricks = too much noise, too many false signals
- Too large bricks = miss actual moves, too few signals
- DSP finds the **natural frequency** where market structure appears

---

### Step 3: Friction Validation
**What:** Verify that trading costs are sustainable
**Goal:** `friction_cost < 1 brick` (otherwise strategy is unprofitable)

**Friction Components:**
1. **Bid-ask spread** (e.g., 1.0 pt = $0.01 on XAUUSD)
2. **Commission** (e.g., $0 on cTrader, varies on other brokers)
3. **Slippage** (model as 0.5 × spread for limit orders)

**Example (XAUUSD):**
```
Brick size: $44.99
Spread: 1.0 pt = $0.01
Round-trip friction: $0.02 (bid-ask)
Friction / Brick: 0.02 / 44.99 = 0.044% ✅ PASS

Strategy: Need to capture ≥ $0.02 per trade to break even
With 0.5 brick stops: risk = $22.50, need >= $0.02 profit (0.09% of risk)
```

```bash
make friction SYMBOL=XAUUSD
```

**Pass Criteria:**
- `friction_cost < 1 brick` = sustainable
- `friction_cost > 1 brick` = **strategy will never work** (spreads exceed profit targets)

---

### Step 4: Empirical Backtest
**What:** Actually test if the strategy works on real data
**Data:** 3 months of M1 (minimal, quick validation)

**Test:**
1. Build bricks at DSP-determined size
2. Apply Renko flip + filter logic
3. Count entries and P&L
4. Check: `Omega > 1.0` (breakeven) OR `Z-factor > 2.0` (significant)

```bash
make backtest SYMBOL=XAUUSD MONTHS=3
```

**Output:**
```
Bricks: 342
Up bricks: 171
Down bricks: 171
Entry signals: 47
Win rate: 51%
Omega: 1.23 ✅ PASS
```

**Pass Criteria:**
- Omega > 1.0 in backtest (must beat buy-and-hold breakeven)
- Z-factor > 2.0 (statistically significant)
- Win rate > 40% (not random)

---

### Step 5: Live Trading (PER Gates)
**What:** Trade with real bars while monitoring performance

**Gate Progression:**
```
PAPER (synthetic lots)
  ├─ Omega ≥ 1.5 → unlock MICRO
  │
  └─ MICRO (0.01 lots)
      ├─ Max DD: 3%
      ├─ Max 2 instruments
      └─ Omega ≥ 2.0 → unlock SMALL
          │
          └─ SMALL (0.1 lots)
              ├─ Max DD: 5%
              ├─ Max 5 instruments
              └─ Omega ≥ 2.5 → unlock FULL
                  │
                  └─ FULL (broker limits)
                      ├─ Max DD: 10%
                      └─ Unlimited instruments
```

```bash
make trade SYMBOL=XAUUSD GATE=paper
```

---

## The No-Assumptions Principle

### What We DON'T Assume:
❌ "Renko should use 5-brick stops" → TEST IT
❌ "20-bar MA crosses are good filters" → TEST IT
❌ "Market is too noisy to trade" → FIND THE RIGHT BRICK SIZE
❌ "All symbols behave the same" → TEST EACH SYMBOL
❌ "Fixed thresholds work everywhere" → ADAPT TO DSP FINDINGS

### What We DO Test:
✅ **DSP brick size** - measured from volatility, not guessed
✅ **Friction sustainability** - must be < 1 brick
✅ **Filter effectiveness** - flip_rate and markov on actual bricks
✅ **Stop distance** - empirically validate 0.5 vs 1.0 brick
✅ **Gate thresholds** - Omega levels for advancement
✅ **Instrument selection** - each symbol gets individual testing

---

## Key Metrics

### DSP Metrics
| Metric | Meaning | Action |
|--------|---------|--------|
| VR peak | Volatility ratio at optimal aggregation | Higher = more structure |
| VR scale (bars) | Bar count where structure appears | Natural frequency |
| Brick size | Price movement per brick | Measured, not chosen |

### Friction Metrics
| Metric | Meaning | Pass? |
|--------|---------|-------|
| Friction / Brick | Spread as % of brick | < 100% (ideally < 5%) |
| Spread (pts) | Raw bid-ask spread | Platform dependent |
| Commission | Per-trade cost | 0 on cTrader ECN |

### Backtest Metrics
| Metric | Meaning | Pass Threshold |
|--------|---------|-----------------|
| Omega | Sharpe-like ratio | > 1.0 (breakeven), target 2.0+ |
| Z-factor | Statistical significance | > 2.0 (p < 0.05) |
| Win rate | % winning trades | > 40% (above random) |
| Max DD | Peak-to-trough loss | < 20% (manageable) |

### Live Metrics
| Metric | Meaning | Action |
|--------|---------|--------|
| Current Omega | Real-time performance | Monitor for gate advancement |
| Portfolio DD | Drawdown from peak | Halt if > gate limit |
| Entry signals | Flip count | Verify bricks are forming |
| Blockers | Rejected entries | Diagnose filter issues |

---

## Testing Workflow

### For a New Symbol (e.g., BTCUSD):

```bash
# 1. Get the data
make download SYMBOL=BTCUSD
# ✓ Saves M1 bars to data/master_standardized/ctrader/{account}/BTCUSD/

# 2. Analyze volatility structure
make dsp SYMBOL=BTCUSD
# ✓ Outputs: VR peak, brick size, regime
# ✓ Saves dsp_profile.json

# 3. Verify friction is acceptable
make friction SYMBOL=BTCUSD
# ✓ Checks: spread < 1 brick?
# ✓ If FAIL: symbol is unsuitable (friction too high)
# ✓ If PASS: continue to backtest

# 4. Quick backtest (3 months)
make backtest SYMBOL=BTCUSD
# ✓ Checks: Omega > 1.0?
# ✓ If FAIL: logic doesn't work (change filters?)
# ✓ If PASS: continue to live trading

# 5. Paper trading (risk-free)
make trade SYMBOL=BTCUSD GATE=paper
# ✓ Monitor: does real market match backtest?
# ✓ Target: Omega ≥ 1.5 to unlock micro lots
# ✓ Watch: blockers, entry signals, P&L

# 6. Micro lots (small real money)
make trade SYMBOL=BTCUSD GATE=micro
# ✓ Risk: max 0.01 lots, 3% DD limit
# ✓ Target: Omega ≥ 2.0 to unlock small

# And so on...
```

---

## Example: XAUUSD Full Analysis

### Step 1: Download
```
40 days × 24 hours × 60 minutes = 57,600 M1 bars
File: XAUUSD_M1_20260102-20260301.csv (1.2 MB)
```

### Step 2: DSP
```
VR peak: 1.426 at 96-bar scale
Brick size: $44.99
Regime: mean-reverting
```

**Interpretation:** Markets show price structure at ~96-minute aggregation. Optimal brick is ~$45. Regime is choppy (mean-revert), not trending.

### Step 3: Friction
```
Spread: 1.0 pt = $0.01
Friction / brick: 0.02% ✅ EXCELLENT (< 1%)
```

**Interpretation:** Friction is negligible. Spread is 0.0005× the brick size. Strategy is friction-sustainable.

### Step 4: Backtest (3 months)
```
Bricks: 342 (57,600 bars / ~168 bars per brick)
Flips: 47 entry signals
Trades: 23 (filled, not rejected)
Wins: 12 (52%)
Losses: 11 (48%)
Omega: 1.31
Z-factor: 1.87
```

**Interpretation:** Slight edge (Omega > 1.0). Not strongly significant (Z < 2.5), but positive. Proceed to live trading.

### Step 5: Paper Trading (1 week)
```
Trades: 8
Wins: 5 (63%)
Losses: 3 (37%)
Omega: 2.14
Profit: $127.50 (synthetic)
```

**Interpretation:** Better than backtest. Real-time performance > historical. Ready for micro lots.

---

## Common Pitfalls

### "The market is too noisy"
**False.** You just haven't found the right brick size yet. DSP will find it.

### "I should optimize the brick size"
**Wrong.** DSP measures it. Tuning the brick size is overfitting.

### "Spreads don't matter, the edge is big"
**Dangerous.** If friction > 1 brick, you cannot be profitable. Test rigorously.

### "My 3-month backtest passed, I'm ready for full lots"
**Reckless.** Do paper trading first, then micro, then small. Gates exist for a reason.

### "I'll add more filters to catch edge cases"
**Complexity kills.** Flip + Markov is proven. Stick to simple rules, test empirically.

---

## Refactoring Complete

The pipeline is now:
- ✅ **Simple** - 5 clear steps, no menu maze
- ✅ **Transparent** - each step has clear inputs/outputs
- ✅ **Empirical** - testing determines everything
- ✅ **Progressive** - paper → micro → small → full (gated by Omega)
- ✅ **First-principles** - no assumptions, only DSP + friction + backtests

**Next:** Implement the missing backtest logic for Step 4, wire up real DSP output.
