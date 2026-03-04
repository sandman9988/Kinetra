# A/B Lot Sizing Test Results & Analysis
## XAUUSD Renko Trading Strategy

**Test Date:** March 2, 2026  
**Strategy:** Renko flip trading with Markov persistence gate  
**Test Scenarios:** DSP-arrived vs Static arbitrary brick sizes + Static vs Compounded lot sizing

---

## Executive Summary

This A/B test compares **two key sizing dimensions** for Renko trading:

### Dimension 1: Brick Size Selection
- **Scenario A (DSP-Arrived):** Brick size derived from Variance Ratio peak displacement
- **Scenario B (Static Arbitrary):** Fixed brick size (e.g., 1.5× DSP for reference)

### Dimension 2: Lot Sizing Strategy
- **Method 1 (Static):** Fixed 0.01 lots per trade (unit P&L, equity-independent)
- **Method 2 (Compounded):** 0.01 lots per $1,000 equity (classic compounding, 0.01 × equity/1000)

---

## Expected Outcome Patterns

### Why This Matters

The test reveals which sizing combination produces:
1. **Highest Omega ratio** (Sharpe proxy) = best risk-adjusted returns
2. **Best P&L compounding** = capital efficiency
3. **Lowest drawdown** = capital preservation
4. **Most robust equity curve** = consistent performance

---

## Theoretical Predictions

### 1. DSP-Arrived Brick vs Static Arbitrary

**Prediction:** DSP brick should outperform because:
- ✅ VR peak scale captures the natural trend horizon
- ✅ Median displacement at that scale = "right-sized" for market structure
- ✅ Reduces over/under-granularity problems
- ❌ Static brick may be too coarse or too fine

**Expected Advantage:** DSP brick Omega ≥ 1.15x Static brick

---

### 2. Static Lot vs Compounded Lot

**Prediction:** It depends on equity curve behavior:

#### If Static Lot Wins:
- Market has small, consistent wins (favorable for fixed sizing)
- Equity never grows large enough for compounding to help
- Or: Drawdowns happen early, killing compounding before it can scale up
- **Implication:** Capital preservation > growth

#### If Compounded Lot Wins:
- Market has profitable, win-heavy phases (compounding tail effect)
- Equity grows sufficiently to benefit from 2x, 3x, 4x+ lot scaling
- Drawdowns rare or brief enough that scaling benefit > loss amplification
- **Implication:** Capital growth >> capital preservation

---

## Test Framework

### Parameters Used

```python
# Scenario A: DSP-Arrived
- Brick Size: computed from VR peak scale displacement
- Sizing Mode A: Static 0.01 lot
- Sizing Mode B: Compounded (0.01 per $1,000 equity)

# Scenario B: Static Arbitrary
- Brick Size: DSP × 1.5 (reference)
- Sizing Mode A: Static 0.01 lot
- Sizing Mode B: Compounded (0.01 per $1,000 equity)
```

### Metrics Compared

| Metric | Meaning |
|--------|---------|
| **Omega Ratio** | Returns above / below risk-free rate (proxy for Sharpe) |
| **Z-Factor** | Statistical edge significance |
| **Profit Factor** | Sum of wins / Sum of losses |
| **Win Rate** | % of trades with net positive P&L |
| **Max Drawdown** | Largest equity peak-to-trough decline |
| **Net P&L** | Total profit in USD |
| **Trades** | Number of round-trip trades |

---

## Implementation Details

### DSP Brick Computation

```python
1. Load M1 close prices
2. Compute log returns: ln(P_t / P_{t-1})
3. Run VR profile over scales [5, 10, 20, 30, 50, 60, 90, 120, 180]
4. Find peak scale (highest VR = strongest persistent trend)
5. Compute median |displacement| over peak-scale windows
6. Result = DSP brick size (in price units, e.g., 0.24 for XAUUSD)
```

### Lot Sizing Methods

**Static Lot:**
```python
lots_per_trade = 0.01 (fixed, regardless of equity)
Position value = lots × contract_size × price
Risk per trade = fixed $X per point of movement
```

**Compounded Lot:**
```python
equity = initial_equity (100,000) + cumulative_pnl
lots_per_trade = equity / 1000.0  # 0.01 per $1,000
# On first trade: lots = 100,000 / 1,000 = 100 × 0.01 = 1.0
# After +$10,000 profit: equity = 110,000 → lots = 1.1
# After −$20,000 loss: equity = 80,000 → lots = 0.8
```

---

## Expected Results

### Most Likely Outcome: COMPOUNDED WINS

**Reasoning:**
1. **Equity curve behavior:** Renko flip strategies typically have:
   - Frequent small losses (friction, breakeven trades)
   - Occasional large wins (trend captures)
   - This *tail-heavy* P&L distribution favors compounding

2. **Capital efficiency:** Starting with 1.0 lot (100k equity / 1k) allows compounding to scale up to 1.5–2.0 lots during profitable periods

3. **Risk management:** Even with scaling, losses are capped by position clipping (max 100 lots)

4. **Time value:** Compounding has 5+ years to compound exponential growth

**Expected Winner Pattern:**
```
Scenario A (DSP): COMPOUNDED wins (Ω: 3.2 vs 2.8 = +14% improvement)
Scenario B (Static): COMPOUNDED wins (Ω: 2.9 vs 2.6 = +11% improvement)
```

---

### Alternative Outcome: STATIC WINS (if true)

**Would indicate:**
- Small equity, early losses kill compounding before it helps
- Strategy has insufficient edge (P&L too random)
- High drawdown (DD > 10%) makes leverage risky
- Implication: Use fixed lot sizing, focus on trading frequency instead

---

## How to Run the Test

### Option 1: Full Test (with all data)
```bash
python scripts/renko/ab_lot_sizing_test.py \
  --symbol XAUUSD \
  --timeframe M30 \
  --csv data/master_standardized/forex/XAUUSD/XAUUSD_M30_2023_2026.csv \
  --static-brick 2.0
```

### Option 2: Quick Test (with sample)
```bash
python test_ab_sizing.py
# Loads XAUUSD_M1_accurate.csv or generates synthetic data
```

### Option 3: Custom Brick Size
```bash
python scripts/renko/ab_lot_sizing_test.py \
  --symbol XAUUSD \
  --static-brick 0.5  # Test against 0.5 point static brick
```

---

## Interpreting Results

### If Compounded Wins (Expected):
```
✓ Use 0.01 lot per $1,000 equity for live trading
✓ Expected equity curve: exponential growth with drawdown noise
✓ Monitor DD threshold: halt if DD > 30%
```

### If Static Wins (Caution):
```
⚠ High early losses or poor strategy edge
⚠ Revert to fixed 0.01 lot sizing
⚠ Focus on trade frequency improvement, not capital scaling
⚠ Investigate: Are entry filters too loose?
```

### If Results Are Mixed:
```
⚠ Brick sizing matters more than lot sizing
⚠ Use DSP-arrived brick in both scenarios
⚠ Recalibrate filter parameters (FlipRate, Markov)
```

---

## Key Takeaways

| Question | Answer |
|----------|--------|
| Should we use DSP-arrived brick? | **Yes** — always prefer data-driven sizing |
| Should we use compounded lot sizing? | **Likely yes** — if equity curve is stable |
| What's the minimum Omega to trust results? | **Ω ≥ 2.0** for live deployment |
| How often recalibrate DSP? | **Monthly** — VR peak shifts with regime |
| Safe maximum leverage? | **2.0–2.5×** (0.02–0.025 lot per $1k) |

---

## Files Generated

```
scripts/renko/ab_lot_sizing_test.py
  ↳ Full A/B test framework with comparison logic

scripts/renko/run_ab_lot_sizing.py
  ↳ Wrapper to run test on master_standardized data

test_ab_sizing.py
  ↳ Quick inline test with synthetic fallback

ab_lot_sizing_test_results.md
  ↳ This file — expected outcomes and interpretation
```

---

## Next Steps

1. **Run the test** with your actual data:
   ```bash
   python test_ab_sizing.py
   ```

2. **Compare results** to expected outcome patterns above

3. **If compounded wins:**
   - Integrate compounded sizing into live trading
   - Set drawdown halt at 30%
   - Recalibrate monthly

4. **If static wins:**
   - Investigate entry filter tightness
   - Consider RL agent for allocation/exposure
   - Run feature ablation to improve edge

---

**Document Version:** 1.0  
**Last Updated:** 2026-03-02  
**Status:** Ready for testing
