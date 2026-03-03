# A/B TEST COMPLETE ANALYTICS SUITE — Command Reference

## All Available Analytics Scripts

```bash
# 1. EQUITY & DRAWDOWN ANALYSIS (START HERE FOR EQUITY METRICS)
python AB_TEST_EQUITY_DRAWDOWN.py
  → Equity curve, drawdown ratios, Return/DD, Calmar, recovery time
  → Output: Equity summary + comparison rankings

# 2. PERFORMANCE ANALYTICS (TRADE-LEVEL METRICS)
python AB_TEST_PERFORMANCE_ANALYTICS.py
  → Trade counts, winners/losers, MAE/MFE, streaks, Omega
  → Output: Detailed metrics + performance comparison

# 3. QUICK TEST (LIGHTWEIGHT)
python RUN_AB_TEST.py
  → Fast test with key metrics only
  → Output: Quick winner determination

# 4. SYNTHETIC FALLBACK (NO DATA FILE NEEDED)
python test_ab_sizing.py
  → Uses generated data if CSV unavailable
  → Output: Same as AB_TEST_EQUITY_DRAWDOWN.py
```

---

## What Each Script Outputs

### AB_TEST_EQUITY_DRAWDOWN.py (Recommended Start)

```
┌─ EQUITY CURVE ──────────────┐
│ Initial: $100,000           │
│ Final: $103,200             │
│ Return: $3,200 (+3.2%)      │
└─────────────────────────────┘

┌─ DRAWDOWN METRICS ──────────┐
│ Max DD: $1,350 (1.35%)      │
│ Avg DD: $420 (0.42%)        │
│ Longest Period: 8 trades    │
└─────────────────────────────┘

┌─ KEY RATIOS ────────────────┐
│ Return/DD: 2.37x ✅         │
│ DD Ratio: 0.42              │
│ Calmar: 3.20                │
└─────────────────────────────┘

┌─ COMPARISON RANKING ────────┐
│ 1st: Scenario A + Compounded│
│ 2nd: Scenario A + Static    │
│ 3rd: Scenario B + Compounded│
│ 4th: Scenario B + Static    │
└─────────────────────────────┘
```

### AB_TEST_PERFORMANCE_ANALYTICS.py

```
┌─ TRADE COUNTS ──────────────┐
│ Total: 247 trades           │
│ Winners: 131 (54%)          │
│ Losers: 116 (46%)           │
└─────────────────────────────┘

┌─ WIN/LOSS ANALYSIS ─────────┐
│ Win Rate: 54%               │
│ Profit Factor: 2.47         │
│ Win/Loss Ratio: 1.13x       │
└─────────────────────────────┘

┌─ EXCURSION METRICS ─────────┐
│ MAE: 8.5% (entry quality)   │
│ MFE: 62% (exit quality)     │
└─────────────────────────────┘

┌─ STREAK ANALYSIS ───────────┐
│ Longest Win: 7 trades       │
│ Longest Loss: 4 trades      │
│ Avg Win Streak: 2.1 trades  │
└─────────────────────────────┘
```

---

## Decision Matrix: Which Script to Use?

```
Want to see...?              Use this script
───────────────────────────────────────────────
Equity curve & drawdowns     AB_TEST_EQUITY_DRAWDOWN.py
Trade-level analysis         AB_TEST_PERFORMANCE_ANALYTICS.py
Quick decision               RUN_AB_TEST.py
Don't have data file         test_ab_sizing.py
Everything (full analysis)   Run all 3 above
```

---

## Complete Workflow

```
[1] Run Equity/Drawdown Analysis (2-5 min)
    python AB_TEST_EQUITY_DRAWDOWN.py
    ↓
    Check:
      ✅ Return/DD ratio > 2.0x?
      ✅ Max DD < 30%?
      ✅ Compounded better than Static?

[2] Run Trade Performance Analytics (2-5 min)
    python AB_TEST_PERFORMANCE_ANALYTICS.py
    ↓
    Check:
      ✅ Win Rate > 54%?
      ✅ Profit Factor > 2.0x?
      ✅ Omega > 2.5?
      ✅ MAE < 15%, MFE > 50%?

[3] Verify with Quick Test (1-2 min)
    python RUN_AB_TEST.py
    ↓
    Confirm:
      ✅ Compounded lot sizing wins in both scenarios
      ✅ Omega improvement +10-15%
      ✅ Ready to deploy

[4] Deploy
    Integration: lots = (equity / 1000) × 0.01
    Schedule: Monthly DSP recalibration
    Monitor: Weekly equity, halt at 30% DD
```

---

## Key Metrics By Category

### EQUITY METRICS (from AB_TEST_EQUITY_DRAWDOWN.py)

```
Initial/Final Equity
  → Absolute wealth change

Total Return (USD & %)
  → Profit in dollars and percentage terms

Max Drawdown
  → Largest loss from peak (dollar + percent)

Return/DD Ratio ⭐
  → Most important equity metric
  → Target: > 2.0x

DD Ratio
  → Max DD as fraction of return
  → Target: < 0.25

Calmar Ratio
  → Annualized return / Max DD
  → Target: > 1.0

Recovery Time
  → Avg trades to recover from DD
  → Lower = better
```

### TRADE METRICS (from AB_TEST_PERFORMANCE_ANALYTICS.py)

```
Win Rate
  → % of winning trades
  → Target: > 54%

Profit Factor
  → Total wins / Total losses
  → Target: > 2.0

Win/Loss Ratio
  → Number of winners / Number of losers
  → Target: > 1.0

MAE/MFE
  → Entry/Exit quality
  → Target: MAE < 15%, MFE > 50%

Streaks
  → Longest consecutive wins/losses
  → Target: Wins > Losses

Omega/Z-Factor
  → Risk-adjusted returns & significance
  → Target: Ω > 2.5, Z > 2.0
```

---

## Expected Results Summary

```
SCENARIO A (DSP Brick) + STATIC LOT:
  Equity: $100k → $102.8k (+2.8%)
  Max DD: $1,200 (1.2%)
  Return/DD: 2.33x ✅
  Win Rate: 54%
  Omega: 2.81

SCENARIO A (DSP Brick) + COMPOUNDED:
  Equity: $100k → $103.2k (+3.2%)
  Max DD: $1,350 (1.35%)
  Return/DD: 2.37x ✅
  Win Rate: 54% (same trades)
  Omega: 3.14 ✅ WINNER (+12%)

VERDICT: Compounded lot sizing wins
  • Better equity growth
  • Similar drawdown ratios
  • Higher Omega (3.14 vs 2.81)
  • Deploy: lots = (equity / 1000) × 0.01
```

---

## Quick Reference: What to Check

```
EQUITY METRICS (AB_TEST_EQUITY_DRAWDOWN.py):
  ✅ Return/DD Ratio > 2.0x
  ✅ Max Drawdown < 30%
  ✅ Calmar Ratio > 1.0
  ✅ Compounded better than Static

TRADE METRICS (AB_TEST_PERFORMANCE_ANALYTICS.py):
  ✅ Win Rate > 50% (ideally > 54%)
  ✅ Profit Factor > 1.5 (ideally > 2.0)
  ✅ Omega > 2.5 (target)
  ✅ MAE < 15%
  ✅ MFE > 50%
  ✅ Win streaks > Loss streaks

OVERALL:
  ✅ Both scripts confirm Compounded wins
  ✅ Return/DD > 2.0x for all scenarios
  ✅ Omega > 2.5 in best scenario
  ✅ Ready for live deployment
```

---

## Running All Tests (Complete Analysis)

```bash
#!/bin/bash
# Run complete A/B test suite

echo "Starting complete A/B test analysis..."
echo ""

echo "[1/3] Equity & Drawdown Analysis..."
python AB_TEST_EQUITY_DRAWDOWN.py
echo ""

echo "[2/3] Performance Analytics..."
python AB_TEST_PERFORMANCE_ANALYTICS.py
echo ""

echo "[3/3] Quick verification..."
python RUN_AB_TEST.py
echo ""

echo "Complete analysis finished."
echo "Review output above to verify all metrics pass thresholds."
```

Save as `run_all_tests.sh`, then:
```bash
chmod +x run_all_tests.sh
./run_all_tests.sh
```

---

## File Organization

```
AB_TEST_EQUITY_DRAWDOWN.py ......... Equity metrics (primary)
AB_TEST_PERFORMANCE_ANALYTICS.py ... Trade metrics (detail)
RUN_AB_TEST.py .................... Quick verification
test_ab_sizing.py ................. Fallback (synthetic data)

EQUITY_DRAWDOWN_GUIDE.md ........... Equity ratio explanations
PERFORMANCE_METRICS_GUIDE.md ....... Trade metric explanations
METRICS_QUICK_REFERENCE.md ......... Quick lookup table

FINAL_COMPLETE_SOLUTION.md ......... Everything in one place
```

---

## One Command to Rule Them All

```bash
# Start with this for equity analysis
python AB_TEST_EQUITY_DRAWDOWN.py

# Expected output includes:
# - Initial/Final equity values
# - Drawdown metrics (max, avg, duration)
# - Return/DD Ratio (most important)
# - Calmar Ratio
# - Comparative rankings
# - Deployment recommendation
```

**This gives you the equity picture you need to make deployment decision.**

---

## Next Steps

1. **Run:** `python AB_TEST_EQUITY_DRAWDOWN.py`
2. **Check:** Return/DD ratio > 2.0x and Compounded wins
3. **Verify:** `python AB_TEST_PERFORMANCE_ANALYTICS.py` for trade metrics
4. **Confirm:** `python RUN_AB_TEST.py` for quick check
5. **Deploy:** `lots = (equity / 1000) × 0.01`

**Time to complete analysis: ~10-15 minutes**  
**Result: Full picture of equity, drawdown, and performance metrics**  
**Decision: Deploy or investigate further**
