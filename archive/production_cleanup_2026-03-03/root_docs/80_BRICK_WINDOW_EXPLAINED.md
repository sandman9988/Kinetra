# The 80-Brick Rolling Window Explained
**Date:** 2026-03-02

## What is the 80-Brick Window?

The **80-brick window** is the **lookback period** for calculating the FlipRate and Markov filters that determine whether a color change is tradeable.

---

## How It Works

### When a New Brick Forms

```
Current brick sequence (last 80 bricks):
[+1, +1, +1, -1, -1, -1, +1, +1, ... +1, +1, -1]  ← NEW brick just formed
 └─────────────── 80 bricks ──────────────────┘
```

### Step 1: Calculate FlipRate (Rolling)
**Question:** How choppy is the market over the last 80 bricks?

```python
# Count flips (color changes) in last 80 bricks
flips = count_where(brick[i] != brick[i-1]) over last 80 bricks

# FlipRate = proportion of flips
FlipRate = flips / 80
```

**Example:**
- If 15 flips in last 80 bricks → FlipRate = 15/80 = 0.19 (trending - good!)
- If 40 flips in last 80 bricks → FlipRate = 40/80 = 0.50 (choppy - bad!)

**Rule:** Only trade if **FlipRate < 0.35** (not too choppy)

---

### Step 2: Calculate Markov Stickiness (Rolling)
**Question:** When a brick is UP, what % of the time does the NEXT brick stay UP?

```python
# Look at transitions in last 80 bricks
# For all transitions where brick[i-1] == +1:
#   Did brick[i] also == +1?

# Count UP→UP transitions
uu_count = count_where(brick[i-1] == +1 AND brick[i] == +1)

# Count all UP→ANY transitions  
up_count = count_where(brick[i-1] == +1)

# P(UP→UP)
pUU = uu_count / up_count
```

Similarly for down bricks:
```python
# P(DOWN→DOWN)
pDD = dd_count / down_count
```

**Example:**
- If 45 of 50 UP bricks were followed by another UP brick → pUU = 45/50 = 0.90 (strong persistence!)
- If 25 of 50 UP bricks were followed by another UP brick → pUU = 25/50 = 0.50 (random walk)

**Rule:** Only trade if **Markov > 0.55** (direction persists)

---

## Entry Decision Logic

```
NEW brick forms with direction = +1 (UP, green)
PREVIOUS brick was direction = -1 (DOWN, red)

→ COLOR CHANGE detected!

Check filters over LAST 80 BRICKS:
  1. FlipRate = 0.22  ← 18 flips in 80 bricks (trending)
  2. pUU = 0.68       ← UP bricks persist 68% of the time

Gate checks:
  ✓ FlipRate (0.22) < 0.35  → PASS (not choppy)
  ✓ pUU (0.68) > 0.55       → PASS (UP persists)
  ✓ Color change occurred   → PASS

→ TRADE LONG! Enter at this brick's close price.
```

---

## Why 80 Bricks?

**Original calculation (FIXED):**
```
DSP finds vr_peak_scale = 720 M1 bars (strongest trend persistence scale)

Convert to bricks:
  720 M1 bars / 1440 bars per day = 0.5 days
  0.5 days × 160 bricks/day (empirical for XAUUSD) = 80 bricks
```

**Meaning:** The market shows strongest trend persistence over a ~0.5 day (12 hour) window, which equals about 80 Renko bricks for XAUUSD.

---

## What Happens with Wrong Window?

### Too Small (e.g., 14 bricks - the bug we fixed!)
```
Last 14 bricks: [+1, +1, -1, -1, +1, +1, +1, -1, -1, +1, +1, +1, -1, -1]
                                                                    ↑ NEW
FlipRate = 9/14 = 0.64  ← Looks VERY choppy (but it's just noise in small sample)
```
**Problem:** Not enough history to distinguish signal from noise.

### Too Large (e.g., 500 bricks)
```
Last 500 bricks includes data from 3 days ago...
Market regime has changed since then!
```
**Problem:** Stale data, slow to adapt to regime shifts.

### Just Right (80 bricks ≈ 0.5 days)
```
Enough history to filter noise
Recent enough to track current regime
```

---

## The Complete Strategy

```
ENTRY TRIGGER:
  1. Color change occurs (brick flips from red → green or green → red)
  2. FlipRate (last 80 bricks) < 0.35  ← Not choppy
  3. Markov (last 80 bricks) > 0.55    ← Direction persists

ENTRY:
  Open position at brick close price

STOP:
  1 brick (backtest) or 0.5 brick (live)

EXIT:
  First brick of opposite color (color change back)
```

---

## Real Example (XAUUSD)

**Live trading scenario:**
```
Time: 14:30 UTC
Last 80 bricks span: 14:00 - 14:30 (last ~30 minutes)
Bricks in that time: 80 bricks at $1.00/brick on XAUUSD

Brick sequence (last 10 shown):
  [...70 earlier bricks...] [+1, +1, +1, +1, +1, -1, -1, -1, -1, -1]
                                                            ↑ PREVIOUS
                                                               ↑ NEW = +1 (color flip!)

Filters (calculated over all 80):
  FlipRate = 0.28  ← 22 flips in 80 bricks
  pUU = 0.62       ← UP bricks persist 62% of time

Decision:
  ✓ Color changed (was -1, now +1)
  ✓ FlipRate 0.28 < 0.35  → Trending market
  ✓ pUU 0.62 > 0.55       → UP direction persists
  
→ BUY 0.486 lots @ $2654.50
→ Stop @ $2653.50 (1 brick below)
→ Exit when first RED brick forms
```

---

## Summary

| Concept | Value | Purpose |
|---------|-------|---------|
| **Window** | 80 bricks | Lookback period for filters |
| **FlipRate** | < 0.35 | Reject choppy markets |
| **Markov pUU/pDD** | > 0.55 | Confirm direction persists |
| **Entry** | Color change | Brick flips color |
| **Stop** | 1 brick | Fixed distance |
| **Exit** | Color change | First opposite brick |

**The 80-brick window is NOT the entry signal.**  
**It's the FILTER that determines if a color change is worth trading.**

---

**Status:** ✅ Window calculation FIXED (was 14 bars, now 80 bricks)  
**Result:** Filters now have enough history to distinguish signal from noise
