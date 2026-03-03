# XAUUSD Pricing Terminology — Clarification

## What is a "Point" for XAUUSD?

There are two possible interpretations:

### Interpretation A: 1 Point = 1 Tick (0.01 price units) = $1.00/lot

```
XAUUSD Quote Example: $2345.67/oz

If price moves from $2345.67 to $2345.68:
  • Price change: $0.01 (one tick)
  • One "point" in this context = $0.01 price movement
  • Value per lot: 0.01 × 100 oz = $1.00

So: 1 point = $1.00 per lot ✓
```

### Interpretation B: 1 Point = 1 Price Unit = $100.00/lot

```
If price moves from $2345.00 to $2346.00:
  • Price change: $1.00 (one full price unit)
  • One "point" in this context = $1.00 price movement
  • Value per lot: 1.00 × 100 oz = $100.00

So: 1 point = $100.00 per lot
```

---

## Which One Is Correct?

For **MetaAPI XAUUSD**, the standard is:

```
XAUUSD MetaAPI Specs:
├─ Quote: USD per ounce (e.g., $2345.67)
├─ Contract size: 100 oz per lot
├─ Tick size: 0.01 (minimum price movement)
├─ Tick value: $1.00 per tick per lot
│
└─ "Point" terminology:
   • In forex context: 1 point = 1 pip = 0.0001 (not applicable to XAUUSD)
   • In XAUUSD context: 1 point = 1 tick = 0.01 price units = $1.00/lot ✓
```

**ANSWER: 1 point = $1.00/lot** (Interpretation A is correct)

---

## Proof

```
XAUUSD Price: $2345.67/oz

Movement of 1 point (0.01):
  New price: $2345.68/oz
  Profit for 1 lot (100 oz): 0.01 × 100 = $1.00 ✓

Movement of 56 points (0.56):
  New price: $2346.23/oz
  Profit for 1 lot (100 oz): 0.56 × 100 = $56.00 ✓

This matches our friction floor calculation:
  Friction: $14.00
  Floor: $14.00 × 4 = $56.00
  Floor in points: $56.00 / $1.00 per point = 56 points ✓
```

---

## Impact on Your Brick Size

### If 1 point = $1.00 (CORRECT) ✓

```
Your brick: 1.0 point = $1.00/lot
Friction: $14.00/lot
Ratio: $14.00 / $1.00 = 1400% ❌ INVALID

Minimum floor: 56 points = $56.00/lot
This makes sense and is viable ✓
```

### If 1 point = $100.00 (WRONG) ❌

```
Your brick: 1.0 point = $100.00/lot
Friction: $14.00/lot
Ratio: $14.00 / $100.00 = 14% ✅ Would be valid!

But this interpretation is incorrect for XAUUSD MetaAPI
```

---

## Confirmed for Your Calculations

```
✅ CORRECT INTERPRETATION:
   1 point = 0.01 price units = $1.00 per lot

✅ YOUR FRICTION COSTS:
   $14.00 total per round-trip (correct)

✅ MINIMUM BRICK FLOOR:
   56 points = $56.00 per lot (required)

✅ YOUR CURRENT BRICK:
   1.0 point = $1.00 per lot (56× too small)

ACTION: Increase brick to ≥56 points
```

---

## Summary Table

| Term | Symbol | Value | USD/lot |
|------|--------|-------|---------|
| 1 Tick | 0.01 | 0.01 price units | $1.00 |
| 1 Point | 0.01 | 0.01 price units | $1.00 |
| 1 Full Unit | 1.00 | 1.00 price units | $100.00 |
| **Your Brick** | 1.0pt | 0.01 price units | **$1.00** |
| **Friction Floor** | 56pt | 0.56 price units | **$56.00** |

---

**Yes, 1 point = $1.00/lot for XAUUSD MetaAPI.**

This confirms:
- Your brick size (1 point) is indeed $1.00/lot
- Your friction ($14.00) is 14× your brick value
- You need to increase brick to 56 points ($56/lot) minimum
- Then A/B test will be valid
