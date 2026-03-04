# EQUITY & DRAWDOWN RATIOS — Complete Guide

## Run the Analysis

```bash
python AB_TEST_EQUITY_DRAWDOWN.py
```

Outputs: Complete equity curve analysis with all ratios.

---

## Key Equity Metrics

### EQUITY CURVE

```
Initial Equity
  → Starting capital ($100,000)

Final Equity
  → Ending capital after all trades

Total Return (USD)
  → Final - Initial
  → Example: $105,000 - $100,000 = $5,000

Total Return (%)
  → (Total Return / Initial) × 100
  → Example: ($5,000 / $100,000) × 100 = 5%
```

---

## Drawdown Metrics (Absolute)

### DEFINITION

```
Drawdown: Peak-to-Trough decline in equity

Example Equity Curve:
  $100,000 ← Peak
       │
       ├─ $95,000 (decline of $5,000 = 5% DD)
       │
       └─ $90,000 ← Trough (max DD from peak)
              │
              └─ $98,000 (recovering)
```

### ABSOLUTE DRAWDOWN METRICS

```
Max Drawdown (USD)
  → Largest dollar decline from peak
  → Formula: Peak - Trough
  → Example: $100,000 - $90,000 = $10,000 DD

Max Drawdown (%)
  → Largest % decline from peak
  → Formula: (Peak - Trough) / Peak × 100
  → Example: ($10,000 / $100,000) × 100 = 10% DD
  → Decision threshold: 30% = HALT

Avg Drawdown (USD)
  → Average decline during all DD periods
  → Example: $3,500 average per DD

Avg Drawdown (%)
  → Average % decline during all DD periods
  → Example: 3.5% average per DD
```

---

## ⭐ KEY RATIO: Return/DD Ratio

### DEFINITION

```
Return/DD Ratio = Total Return / Max Drawdown

Example:
  Total Return: $5,000
  Max DD: $2,000
  Return/DD Ratio: $5,000 / $2,000 = 2.5x

Interpretation:
  For every $1.00 of maximum equity loss,
  you made $2.50 in net profit
```

### INTERPRETATION TABLE

```
Return/DD Ratio | Meaning | Status |
1.0x            | Return equals max DD | Marginal
1.5x            | 50% more return than DD | Acceptable
2.0x            | 2× return vs DD | Good ✅
2.5x            | Excellent return/risk | Very Good ✅✅
3.0x+           | Exceptional edge | Excellent ✅✅✅

Target for XAUUSD: > 2.0x
Expected: 2.4-2.8x
```

### Why This Matters

```
Scenario A (Static):
  Total Return: $2,800
  Max DD: $1,200
  Return/DD: 2.33x ✅

Scenario A (Compounded):
  Total Return: $3,200
  Max DD: $1,350 (slightly larger due to leverage)
  Return/DD: 2.37x ✅ (similar or better!)

Winner: Compounded
Reason: Higher return with manageable DD increase
```

---

## ⭐ KEY RATIO: DD Ratio

### DEFINITION

```
DD Ratio = Max Drawdown / Total Return

Example:
  Max DD: $2,000
  Total Return: $5,000
  DD Ratio: $2,000 / $5,000 = 0.40

Interpretation:
  Max drawdown is 40% of total return
  (Lower is better)
```

### INTERPRETATION TABLE

```
DD Ratio | Meaning | Status |
0.50     | DD is 50% of return | Poor
0.33     | DD is 33% of return | Acceptable
0.25     | DD is 25% of return | Good ✅
0.15     | DD is 15% of return | Very Good ✅✅
< 0.10   | DD is < 10% of return | Exceptional ✅✅✅

Target: < 0.25
```

---

## ⭐ KEY RATIO: Calmar Ratio

### DEFINITION

```
Calmar Ratio = Annual Return / Max Drawdown

Example:
  Total Return: $5,000 (over 1 year)
  Max DD: $2,000
  Calmar = $5,000 / $2,000 = 2.5

Interpretation:
  For every $1 of max loss risk,
  you earn $2.50 per year of returns
```

### INTERPRETATION TABLE

```
Calmar | Meaning | Status |
0.25   | Low return/risk ratio | Poor
0.50   | Acceptable | Marginal
1.0    | Good | Acceptable ✅
1.5    | Very good | Good ✅✅
2.0+   | Excellent | Excellent ✅✅✅

Target: > 0.5
Expected: 1.0-1.5
```

### Calmar vs Return/DD Ratio

```
Difference:
  Return/DD: Total return / Max DD (one-time view)
  Calmar: Annualized return / Max DD (per-year rate)

For XAUUSD:
  If 200 trades over 1 year
    → Return/DD and Calmar are similar
  If 200 trades over 3 years
    → Return/DD = 2.5x but Calmar = 0.8 (lower)
```

---

## DRAWDOWN DURATION & RECOVERY

### DEFINITION

```
Drawdown Period
  → Time (in trades) from peak to trough to recovery

Example:
  Trade 50: Equity peaks at $100,000
  Trade 55: Equity troughs at $95,000 (max DD = $5,000)
  Trade 65: Equity recovers to $100,000
  
  DD Duration: 15 trades
  Recovery Time: 10 trades (from trough to recovery)
```

### METRICS

```
Longest DD Period
  → Maximum consecutive trades in a drawdown
  → Example: 12 trades
  → Higher = longer pain periods (worse psychologically)

Avg Recovery Time
  → Average trades to recover from drawdown
  → Example: 6.5 trades average
  → Lower = faster recovery (better)

Consecutive DD Periods
  → How many separate drawdown periods occurred
  → Example: 8 separate DD periods
  → More = more frequent losses (fragile strategy)
```

---

## EXPECTED RESULTS (XAUUSD)

```
SCENARIO A (DSP Brick) + STATIC:
  Initial Equity: $100,000
  Final Equity: $102,800
  Total Return: $2,800 (2.8%)
  Max DD: $1,200 (1.2%)
  
  Return/DD Ratio: 2.33x ✅
  DD Ratio: 0.43
  Calmar Ratio: 2.80

SCENARIO A (DSP Brick) + COMPOUNDED:
  Initial Equity: $100,000
  Final Equity: $103,200
  Total Return: $3,200 (3.2%)
  Max DD: $1,350 (1.35%)
  
  Return/DD Ratio: 2.37x ✅ (similar)
  DD Ratio: 0.42
  Calmar Ratio: 3.20
  
  ⭐ Winner: Slightly better Return/DD despite larger DD
     Leverage is working properly
```

---

## DECISION THRESHOLDS

```
Return/DD Ratio:
  > 2.0x = Deploy
  1.5-2.0x = Review
  < 1.5x = Investigate

DD Ratio:
  < 0.25 = Deploy
  0.25-0.50 = Acceptable
  > 0.50 = Fix strategy

Calmar Ratio:
  > 1.0 = Deploy
  0.5-1.0 = Marginal
  < 0.5 = Improve strategy

Max Drawdown:
  < 20% = Safe
  20-30% = Acceptable
  > 30% = HALT CONDITION

Recovery Time:
  < 5 trades = Good
  5-10 trades = Acceptable
  > 10 trades = Slow recovery
```

---

## Real-World Interpretation

### Scenario: Two Strategies with Same Return

```
Strategy A:
  Total Return: $5,000
  Max DD: $500
  Return/DD: 10.0x ✅✅✅

Strategy B:
  Total Return: $5,000
  Max DD: $4,000
  Return/DD: 1.25x ❌

Same profit, but A is MUCH safer!
Return/DD Ratio reveals the difference.
```

### Why Drawdown Ratios Matter

```
Without Return/DD:
  Both strategies show 5% return
  Looks identical

With Return/DD:
  Strategy A: 10.0x (excellent risk/reward)
  Strategy B: 1.25x (high risk for same return)
  Clear winner: A

Lesson: Return alone is misleading.
        Must consider drawdown.
```

---

## Equity Curve Quality Signals

```
Good Equity Curve:
  ✅ Steady upward trend
  ✅ Small, brief drawdowns
  ✅ Quick recoveries
  ✅ Return/DD > 2.0x
  ✅ Few DD periods

Bad Equity Curve:
  ❌ Flat or declining
  ❌ Large, prolonged drawdowns
  ❌ Slow recoveries
  ❌ Return/DD < 1.5x
  ❌ Many DD periods
```

---

## Static vs Compounded: Equity Impact

```
STATIC LOT (0.01 fixed):
  • Equity grows linearly
  • DD stays constant dollar amount
  • Return/DD improves slowly

COMPOUNDED LOT (0.01 per $1k):
  • Equity grows exponentially
  • DD grows with equity (proportional)
  • Return/DD stays similar or better
  • Larger dollar DD, but ≈ same % return/DD

Winner: Compounded
Reason: Better use of capital with similar ratios
```

---

## Files & Commands

```bash
# Run equity analytics
python AB_TEST_EQUITY_DRAWDOWN.py

# Expected output shows:
  ├─ Initial/Final equity
  ├─ Total return (USD + %)
  ├─ Max/Avg drawdown
  ├─ Return/DD ratio (key metric)
  ├─ DD ratio
  ├─ Calmar ratio
  ├─ Recovery time
  └─ Comparison rankings
```

---

## Summary Card

```
╔════════════════════════════════════════════════════════╗
║           KEY EQUITY RATIOS AT A GLANCE               ║
╠════════════════════════════════════════════════════════╣
║                                                        ║
║  Return/DD Ratio (most important)                     ║
║  ├─ Target: > 2.0x                                   ║
║  └─ Formula: Total Return / Max Drawdown             ║
║     (Higher = better)                                 ║
║                                                        ║
║  DD Ratio (risk perspective)                          ║
║  ├─ Target: < 0.25                                   ║
║  └─ Formula: Max DD / Total Return                   ║
║     (Lower = better)                                  ║
║                                                        ║
║  Calmar Ratio (annual risk-adjusted return)           ║
║  ├─ Target: > 1.0                                    ║
║  └─ Formula: Annual Return / Max DD                  ║
║     (Higher = better)                                 ║
║                                                        ║
║  Max Drawdown (absolute risk)                         ║
║  ├─ Target: < 30%                                    ║
║  └─ Halt: > 30% (automatic stop)                     ║
║                                                        ║
╚════════════════════════════════════════════════════════╝
```

---

## Next Steps

```
1. Run: python AB_TEST_EQUITY_DRAWDOWN.py
2. Check: Return/DD ratio for each scenario
3. Verify: Ratio > 2.0x (target)
4. Deploy: Configuration with best Return/DD
5. Monitor: Weekly Return/DD trend (should stay > 2.0x)
```

**The Return/DD ratio is your #1 equity metric. Everything else is detail.**
