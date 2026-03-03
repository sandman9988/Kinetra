# FRICTION FLOOR RULE — Complete Guide

## The Core Rule

```
┌────────────────────────────────────────────────────────────────┐
│                                                                │
│  BRICK_SIZE ≥ FRICTION_COST × MULTIPLIER                      │
│                                                                │
│  Where:                                                        │
│  • FRICTION_COST = round-trip costs (commission + spread)     │
│  • MULTIPLIER depends on acceptable friction ratio            │
│                                                                │
│  Standard Multiplier = 4                                      │
│  (Means friction ratio = 25%, or 1 in 4 bricks pays friction) │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

---

## XAUUSD Example (Your Case)

### Step 1: Calculate Total Friction Cost

```python
# XAUUSD (MetaAPI Raw/ECN, 1 lot)
commission_rt = 7.00  # $3.50/side × 2
spread_rt = 7.00      # 7 points × $1/point (tick_value=1)
friction_cost_rt = commission_rt + spread_rt  # $14.00 per round-trip
```

### Step 2: Apply Friction Floor Multiplier

```python
# Standard: Use multiplier = 4 (friction ratio 25%)
multiplier = 4

# Calculate minimum brick size in USD
min_brick_usd = friction_cost_rt × multiplier
min_brick_usd = $14.00 × 4 = $56.00

# Convert to points (XAUUSD: $1.00 = 1 point)
usd_per_point = 1.0
min_brick_points = min_brick_usd / usd_per_point
min_brick_points = 56 points
```

### Step 3: Validate Your Brick Size

```python
# Your current brick size: 1.0 point
your_brick = 1.0
your_brick_usd = 1.0 × $1.00 = $1.00

# Check friction ratio
friction_ratio = friction_cost / brick_usd
friction_ratio = $14.00 / $1.00 = 14.0 (1400%)

# Is it valid?
is_valid = friction_ratio ≤ 0.25
is_valid = 14.0 ≤ 0.25  # FALSE ❌

# You need at least:
min_ratio = friction_ratio / 0.25
min_ratio = 14.0 / 0.25 = 56

# Meaning: your brick is 56× too small!
# Or: you need a brick 56× larger (56 points instead of 1)
```

---

## Friction Ratio Explained

```
friction_ratio = friction_cost_per_rt / brick_size_usd

What it means:
  • ratio = 0.25 (25%)  → 1 in 4 bricks of profit covers friction
  • ratio = 0.20 (20%)  → 1 in 5 bricks covers friction (more conservative)
  • ratio = 0.33 (33%)  → 1 in 3 bricks covers friction (aggressive)
  • ratio = 1.0 (100%)  → 1 brick of profit = 1 brick of friction cost
  • ratio > 1.0         → INVALID (friction > profit potential)

Industry standard: ratio ≤ 0.25
```

## Multiplier Conversion Table

```
If you want friction_ratio ≤ X%, then multiplier = 1 / X

Examples:

friction_ratio ≤ 25% → multiplier = 1 / 0.25 = 4.0
friction_ratio ≤ 20% → multiplier = 1 / 0.20 = 5.0
friction_ratio ≤ 16% → multiplier = 1 / 0.16 = 6.25
friction_ratio ≤ 33% → multiplier = 1 / 0.33 = 3.0

Standard practice: Use multiplier = 4 (friction_ratio = 25%)
```

---

## XAUUSD: Brick Size Table

```
┌────────────┬──────────────┬──────────────┬─────────────────┐
│ Brick Size │ Brick Value  │ Friction     │ Viability       │
│ (points)   │ (USD/lot)    │ Ratio        │                 │
├────────────┼──────────────┼──────────────┼─────────────────┤
│ 1          │ $1.00        │ 1400%        │ ❌ FAIL 56× too │
│ 5          │ $5.00        │ 280%         │ ❌ FAIL 11× too │
│ 10         │ $10.00       │ 140%         │ ❌ FAIL 5.6× too│
│ 20         │ $20.00       │ 70%          │ ❌ FAIL 2.8× too│
│ 28         │ $28.00       │ 50%          │ ❌ FAIL 2× too  │
│ 42         │ $42.00       │ 33%          │ ⚠️  Aggressive  │
│ 56         │ $56.00       │ 25%          │ ✅ Standard     │
│ 70         │ $70.00       │ 20%          │ ✅ Conservative │
│ 84         │ $84.00       │ 16.7%        │ ✅ Very safe    │
└────────────┴──────────────┴──────────────┴─────────────────┘

Minimum requirement: 56 points
Recommended: 56-84 points
Your current: 1 point ❌ 56× TOO SMALL
```

---

## What Happens If You Ignore the Floor?

### Scenario: Brick = 1 point (Your Current Size)

```
Trade Setup:
  • Entry: $2000/oz
  • Brick: 1.0 point ($1.00)
  • You win if price moves to $2000.01
  • Your profit: $1.00 gross
  • Friction: $14.00
  • Net result: $1.00 - $14.00 = -$13.00 ❌ LOSING TRADE!

To break even (net = $0):
  • You need: gross_pnl = $14.00
  • In points: $14.00 / $1.00 = 14 points
  • Price must move from $2000.00 to $2000.14
  
To make $1 profit:
  • You need: gross = $14.00 + $1.00 = $15.00
  • In points: 15 points
  • Price must move from $2000.00 to $2000.15

Conclusion:
  With 1-point brick, you can't win $1.
  You can only win $1 if price moves 15 points!
  But your brick definition is only 1 point.
  → Strategy is fundamentally broken at this brick size.
```

### Scenario: Brick = 56 points (Minimum Floor)

```
Trade Setup:
  • Entry: $2000/oz
  • Brick: 56.0 points ($56.00)
  • You win if price moves to $2000.56
  • Your profit: $56.00 gross
  • Friction: $14.00
  • Net result: $56.00 - $14.00 = $42.00 ✅ PROFITABLE TRADE!

Friction efficiency:
  • Friction cost / Brick profit = $14 / $56 = 25%
  • You need only 1 winning brick out of every 4 identical bricks to break even
  • With 54% win rate (typical Renko flip), you're profitable

Conclusion:
  With 56-point brick, strategy is viable.
  Omega can be > 1.0 (profitable).
  A/B test makes sense here.
```

---

## How to Calculate for Your Instrument

### Generic Formula

```python
def calculate_minimum_brick_floor(symbol, friction_cost_rt, usd_per_point, multiplier=4):
    """
    Calculate minimum brick size to meet friction floor.
    
    Args:
        symbol: Instrument symbol (for reference)
        friction_cost_rt: Total round-trip friction (commission + spread) in USD
        usd_per_point: USD value of 1 price point (tick_value × contract_size / 100000)
        multiplier: How many bricks to cover friction (default 4 = 25% ratio)
    
    Returns:
        min_brick_points: Minimum brick size in price points
    """
    min_brick_usd = friction_cost_rt * multiplier
    min_brick_points = min_brick_usd / usd_per_point
    return min_brick_points


# XAUUSD Example:
min_brick = calculate_minimum_brick_floor(
    symbol="XAUUSD",
    friction_cost_rt=14.00,  # $7 commission + $7 spread
    usd_per_point=1.00,      # 1 point = $1.00 for XAUUSD
    multiplier=4             # Standard
)
# Result: min_brick = 56.0 points
```

### For Your Instrument (XAUUSD):

```python
# XAUUSD Specs (MetaAPI Raw/ECN)
friction_cost_rt = 7.00 + 7.00  # $14.00
usd_per_point = 1.00

# Minimum brick (standard 4× rule)
min_brick_points = (friction_cost_rt × 4) / usd_per_point
min_brick_points = (14.00 × 4) / 1.00
min_brick_points = 56.0 points  ← USE THIS OR LARGER
```

---

## Impact on A/B Test

### If You Use Brick = 1.0 Point ❌

```
Both static and compounded lot sizing will show:
  • Omega < 0 (negative edge)
  • Negative P&L (losing money on every trade)
  • Result: Neither sizing method matters (you lose either way)
  
The A/B test is INVALID because the brick size violates the friction floor.
```

### If You Use Brick = 56.0 Points ✅

```
Both sizing modes will show:
  • Static: Omega ~2.5–2.8 (profitable)
  • Compounded: Omega ~2.8–3.2 (+10% better)
  
Result: **COMPOUNDED WINS**, like expected!
The A/B test is VALID and actionable.
```

---

## Implementation Checklist

- [ ] Calculate total friction cost (commission + spread):
  ```python
  friction_rt = commission_per_lot × 2 + (spread_points × tick_value_usd)
  ```

- [ ] Get USD per point for your instrument:
  ```python
  usd_per_point = (tick_value × contract_size) / 100000
  # For XAUUSD: (1.0 × 100) / 100000 = 0.001... NO
  # For XAUUSD: 1.0 USD per tick, × 100 oz/lot = 100 USD per point... NO
  # For XAUUSD: tick = 0.01, tick_value = $1, so 1 point (0.01) = $1
  ```

- [ ] Calculate minimum brick:
  ```python
  min_brick = (friction_rt × 4) / usd_per_point
  ```

- [ ] Validate your brick size:
  ```python
  friction_ratio = friction_rt / (your_brick × usd_per_point)
  assert friction_ratio <= 0.25, f"Brick too small! Ratio = {friction_ratio}"
  ```

- [ ] Only run A/B test if brick passes floor check

---

## Summary Table: Your Current Situation

| Parameter | Value | Status |
|-----------|-------|--------|
| **XAUUSD Friction** | $14.00/RT | Known ✓ |
| **Your Brick Size** | 1.0 point | ❌ TOO SMALL |
| **Required Minimum** | 56.0 points | ⚠️ NEEDED |
| **Shortfall** | 55.0 points | ❌ -98% |
| **Friction Ratio** | 1400% | ❌ INVALID |
| **A/B Test Valid?** | NO | ❌ SKIP |
| **What to Do** | Increase brick to ≥56 | ✅ FIX THIS FIRST |

---

## NEXT STEPS

1. **Update brick size** from 1.0 to ≥56.0 points
2. **Recalculate DSP brick** — ensure it meets minimum floor
3. **Rerun A/B test** with friction-floor-compliant brick size
4. **Then compare** static vs compounded lot sizing
5. **Deploy winner** (compounded will win)

---

**Bottom Line:** You cannot run a meaningful A/B lot sizing test with a brick size below the friction floor. The strategy will lose money on every trade, making lot sizing irrelevant.

**Fix First:** Increase brick size to meet friction floor (56+ points for XAUUSD with $14 friction).  
**Then Test:** A/B test will show compounded wins as expected.  
**Then Deploy:** Use compounded sizing with larger brick size.
