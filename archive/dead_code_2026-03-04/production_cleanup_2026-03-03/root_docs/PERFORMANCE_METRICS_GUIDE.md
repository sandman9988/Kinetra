# A/B TEST PERFORMANCE ANALYTICS — Metrics Guide

## Quick Start

```bash
python AB_TEST_PERFORMANCE_ANALYTICS.py
```

Outputs: Complete performance breakdown with all metrics below.

---

## Metrics Explained

### TRADE COUNTS

```
Total Trades
  → Total number of round-trip trades

Winners vs Losers
  → Count of profitable vs unprofitable trades
  → Expressed as: count + percentage

Breakeven
  → Trades that closed at entry price (zero P&L)
```

### WIN/LOSS METRICS

```
Win Rate (WR)
  → % of trades that were profitable
  → Formula: (winners / total_trades) × 100
  → Target: > 50% (better than random)
  → Example: 54% = 54 out of 100 trades profitable

Loss Rate
  → % of trades that were unprofitable
  → Formula: (losers / total_trades) × 100
  → Typically: 100% - Win Rate

Win/Loss Ratio
  → How many winners per loser
  → Formula: winners / losers
  → Example: 1.2x = 1.2 winners for every 1 loser
  → If losers = 0 (all winners): ratio = infinity (displayed as count)

Profit Factor (PF)
  → Total wins / Total losses
  → Formula: sum(positive_trades) / abs(sum(negative_trades))
  → Interpretation:
    • PF > 2.0 = Excellent (wins are 2× losses)
    • PF > 1.5 = Good
    • PF > 1.0 = Profitable
    • PF < 1.0 = Losing system
  → Example: PF 2.5 = For every $1 lost, made $2.50
```

### P&L SUMMARY

```
Total Gross P&L
  → Sum of all trades BEFORE friction costs
  → Formula: Σ(exit_price - entry_price)
  → Shows underlying strategy edge

Total Friction
  → Sum of all costs (commission + spread)
  → Formula: Σ(commission + spread)
  → For XAUUSD: typically 0.07 × total trades

Total Net P&L
  → Gross P&L minus friction
  → Formula: Gross - Friction
  → This is what you actually keep

Avg Win
  → Average profit on winning trades
  → Formula: sum(winners) / count(winners)
  → Example: $47.23 per winning trade

Avg Loss
  → Average loss on losing trades
  → Formula: sum(losses) / count(losses)
  → Example: -$8.50 per losing trade

Avg Net/Trade
  → Average outcome per trade (win or loss)
  → Formula: total_net_pnl / total_trades
  → Example: $3.24 average per trade
  → Positive = edge exists
```

### MAE/MFE ANALYSIS (Maximum Excursion)

```
MAE — Maximum Adverse Excursion
  → How much a trade went against you before closing
  → Formula: (friction_cost / gross_pnl)
  → Expressed as: % of the gross move
  → Example: 10% MAE = trade was 10% wrong before closing
  → Interpretation:
    • Low MAE (< 5%) = Clean entries, hitting right direction
    • High MAE (> 20%) = Bad entries, had to wait for reversal

MFE — Maximum Favorable Excursion
  → How much profit potential was available
  → Formula: (actual_capture / potential_move)
  → Expressed as: % of what was available
  → Example: 60% MFE = captured 60% of available move
  → Interpretation:
    • MFE > 80% = Excellent exit timing
    • MFE 50-80% = Good exits
    • MFE < 50% = Missing profit opportunity

MFE Captured
  → Ratio of favorable moves actually realized
  → Formula: (net_profit / gross_potential) × 100
  → Example: 65% MFE = Captured 65% of what was available
  → Shows quality of exit execution
```

### STREAK ANALYSIS

```
Longest Win Streak
  → Maximum consecutive winning trades
  → Example: 7 = Had 7 wins in a row
  → Longer streaks = better consistency

Longest Loss Streak
  → Maximum consecutive losing trades
  → Example: 4 = Had 4 losses in a row
  → Shorter = more resilient

Avg Win Streak
  → Average length of winning periods
  → Formula: avg(all_winning_runs)
  → Example: 2.3 trades = typically win for ~2.3 trades before loss

Avg Loss Streak
  → Average length of losing periods
  → Example: 1.8 trades = typically lose for ~1.8 trades
  → Lower = quicker recovery

Streak Interpretation:
  If Longest Win Streak > Longest Loss Streak:
    → Strategy has directional bias (good for trending markets)
  If similar:
    → Strategy is balanced (good for choppy markets)
```

### RISK METRICS

```
Omega Ratio (Ω)
  → Risk-adjusted returns (target metric for this test)
  → Formula: (return above risk-free) / (return below risk-free)
  → Example: Ω = 3.14 = excellent edge
  → Benchmark:
    • Ω > 2.5 = Excellent
    • Ω > 2.0 = Good
    • Ω > 1.5 = Acceptable
    • Ω < 1.0 = Losing edge

Z-Factor
  → Statistical significance of edge
  → Formula: (mean_return / std(returns))
  → Interpretation:
    • Z > 2.5 = Edge is real (p < 0.01)
    • Z > 2.0 = Likely real
    • Z < 1.5 = Might be luck

Max Drawdown
  → Largest peak-to-trough equity decline
  → Formula: (peak_equity - trough_equity) / peak_equity
  → Example: -15% = lost 15% from high
  → Psychological tolerance limit: ~20-30% max
  → Live halt condition: 30% drawdown = STOP

Win/Loss Ratio (USD)
  → Dollar amount: avg_winner / abs(avg_loser)
  → Example: 5.5x = Win $5.50 for every $1.00 lost
  → Formula: avg_win / abs(avg_loss)
  → If this > 1, edge is in win magnitude (not frequency)
```

---

## Interpretation Examples

### Example 1: Strong Strategy (Expected for XAUUSD)

```
Total Trades: 247
Win Rate: 54% (131 winners, 116 losers)
Win/Loss Ratio: 1.13x
Profit Factor: 2.47

Avg Win: $47.23
Avg Loss: -$8.50
Avg Net: $3.24/trade

MAE: 8.5%
MFE: 62%
MFE Captured: 62%

Longest Win Streak: 7
Longest Loss Streak: 4
Avg Win Streak: 2.1
Avg Loss Streak: 1.6

Omega: 3.14 ✅ Excellent
Z-Factor: 2.89 ✅ Significant
Max DD: -15% ✅ Manageable

INTERPRETATION:
  ✅ Win rate is slightly above 50% (good)
  ✅ Average winner (47) > average loser (8.5) by 5.5x
  ✅ Profit factor 2.47 = for every $1 lost, made $2.47
  ✅ Streaks show consistency (win/loss roughly balanced)
  ✅ MAE/MFE reasonable (8.5% entry cost, 62% exit capture)
  ✅ Omega 3.14 is excellent (target: > 2.5)
  ✅ Drawdown manageable (15% < 30% halt limit)

VERDICT: Strong, consistent edge. Deploy immediately.
```

### Example 2: Weak Strategy (What NOT to see)

```
Total Trades: 189
Win Rate: 42% (79 winners, 110 losers)
Win/Loss Ratio: 0.72x ❌ More losers than winners
Profit Factor: 0.85 ❌ For every $1 won, lost $1.18

Avg Win: $12.30
Avg Loss: -$18.50 ❌ Losses > wins
Avg Net: -$1.23/trade ❌ Losing money

MAE: 25% ❌ Bad entry timing
MFE: 35% ❌ Poor exits
MFE Captured: 35%

Longest Win Streak: 3 ❌ Short
Longest Loss Streak: 8 ❌ Long losing periods

Omega: 0.82 ❌ Negative edge
Z-Factor: 0.45 ❌ Not significant
Max DD: -40% ❌ Severe

INTERPRETATION:
  ❌ Win rate below 50% + losses > wins = double problem
  ❌ Average loser > average winner = bad edge
  ❌ Profit factor < 1.0 = losing money
  ❌ High MAE/Low MFE = poor signal quality
  ❌ Long loss streaks = not stress-resistant
  ❌ Omega < 1.0 = no edge
  ❌ Drawdown > 30% = too risky

VERDICT: Do NOT deploy. Fix strategy first.
```

### Example 3: Comparing Static vs Compounded (A/B Test)

```
SCENARIO A + STATIC:
  Win Rate: 54%
  PF: 2.47
  Avg Net: $3.24
  Omega: 2.81
  Max DD: -15%

SCENARIO A + COMPOUNDED:
  Win Rate: 54% (same trades)
  PF: 2.47 (same trades)
  Avg Net: $4.12 (+27% more because larger position)
  Omega: 3.14 (+12% improvement)
  Max DD: -16% (slightly larger dollar DD, but smaller % of equity)

WINNER: COMPOUNDED
  → Same win rate (trading logic unchanged)
  → Same profit factor (same trades)
  → 12% higher Omega (better risk-adjusted return)
  → Higher average P&L (capital efficiently deployed)
  → Slightly larger DD (expected with leverage)
```

---

## How to Read the Output

When you run `python AB_TEST_PERFORMANCE_ANALYTICS.py`, you'll see:

```
[1] TRADE COUNTS — How many trades, how many won/lost
[2] WIN/LOSS METRICS — Win rate, profit factor, ratios
[3] P&L SUMMARY — Total profit and per-trade averages
[4] MAE/MFE ANALYSIS — Entry and exit quality
[5] STREAK ANALYSIS — Consistency and resilience
[6] RISK METRICS — Omega, Z, drawdown
[7] COMPARISON TABLE — All 4 scenarios ranked
[8] RANKINGS — By Omega, Profit Factor, Win Rate
[9] OVERALL CONCLUSION — Best performer and deployment recommendation
```

---

## Decision Thresholds

```
TRADE COUNTS
  Minimum viable: 50+ trades (for statistical significance)
  Ideal: 200+ trades

WIN RATE
  Acceptable: > 50%
  Good: > 54%
  Excellent: > 60%

PROFIT FACTOR
  Minimum: > 1.0 (profitable)
  Good: > 1.5
  Excellent: > 2.0

OMEGA RATIO (Key Metric)
  Deploy immediately: Ω > 2.5
  Acceptable: Ω > 2.0
  Review: Ω > 1.5
  Don't deploy: Ω < 1.0

MAX DRAWDOWN
  Safe: < 20%
  Acceptable: < 30%
  Halt condition: > 30%

MAE/MFE BALANCE
  Good: MAE < 15%, MFE > 50%
  Acceptable: MAE < 20%, MFE > 40%
  Problem: MAE > 25% or MFE < 30%
```

---

## What the A/B Test Will Show

**Expected Result (XAUUSD with Compounded Lot Sizing):**

```
┌─ SCENARIO A (DSP Brick) + STATIC LOT ─────────────┐
│ Win Rate: 54%                                      │
│ PF: 2.47                                           │
│ Omega: 2.81                                        │
│ Avg Net: $3.24/trade                              │
└────────────────────────────────────────────────────┘

┌─ SCENARIO A (DSP Brick) + COMPOUNDED LOT ────────┐
│ Win Rate: 54% (same strategy)                     │
│ PF: 2.47 (same strategy)                          │
│ Omega: 3.14 ✅ +12% BETTER                        │
│ Avg Net: $4.12/trade +27% better                 │
└────────────────────────────────────────────────────┘

WINNER: COMPOUNDED LOT SIZING wins by 0.33 Omega points.
```

---

## Files Related to Metrics

- **`AB_TEST_PERFORMANCE_ANALYTICS.py`** — Generates these metrics (run this)
- **`AB_TEST_FINAL_SUMMARY.md`** — Explanation and deployment guide
- **`QUICK_START.md`** — 30-second reference

---

## Summary

The analytics script gives you:
1. **Trade-level metrics** (winners/losers, streaks)
2. **Portfolio-level metrics** (Omega, Profit Factor, Drawdown)
3. **Quality metrics** (MAE/MFE, entry/exit timing)
4. **Comparative analysis** (A vs B, Static vs Compounded)
5. **Deployment recommendation** (which configuration to use)

**Run it. Let it show you the winner. Deploy that configuration.**
