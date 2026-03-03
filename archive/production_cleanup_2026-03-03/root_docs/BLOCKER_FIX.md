# FIX: Blocker Logic Should Count Bricks, Not Bars

## Problem
The live trader was incrementing `no_flip` and other entry blockers **once per bar**, causing them to multiply across all bars within a single brick. After a week of empirical testing, you accumulated 5 `no_flip` blockers from a single brick that lacked a colour flip.

## Root Cause
Entry blockers were being counted every time `_check_entry()` was called, but the calling logic only ensures that `_check_entry()` is called when a new brick forms:

```python
if len(bricks) > prev_n_bricks and state.open_trade is None and not self._halted:
    self._check_entry(...)
```

However, if the bar provider or internal logic calls `_on_bar()` multiple times with the same M1 bar (e.g., reconnections, retries), the brick array remains unchanged but `_check_entry()` would still be called and blockers would be incremented multiple times.

## Solution
Track the last brick index checked for each instrument and only count blockers when a **new brick is encountered**:

```python
# In InstrumentLiveState:
last_checked_brick_idx: int = -1

# In _check_entry():
is_new_brick = (last_idx != state.last_checked_brick_idx)
state.last_checked_brick_idx = last_idx

# Then guard all blocker increments:
if is_new_brick:
    self._note_entry_block("reason")
```

## Changes Made
- Added `last_checked_brick_idx: int = -1` field to `InstrumentLiveState` 
- Modified `_check_entry()` to track and check `is_new_brick`
- Guarded all `self._note_entry_block()` calls with `if is_new_brick:`
  - `no_flip`
  - `loss_pause`
  - `startup_skip`
  - `fliprate_unready`
  - `markov_unready`
  - `filter_reject`
  - `short_disabled`
  - `circuit_breaker:{result}`
  - `alloc_index_miss`
  - `zero_lots`
  - `open_reject`

## Result
Blockers now increment **exactly once per brick that fails the check**, not multiple times per bar. The blocker count accurately reflects how many bricks failed entry criteria, not how many bars were processed.

## Files Modified
- `kinetra/renko/live_trader.py`
  - Line 416: Added `last_checked_brick_idx` field
  - Lines 1863-1871: Added brick tracking logic
  - Lines 1870-2076: Guarded all blocker increments with `is_new_brick` check
