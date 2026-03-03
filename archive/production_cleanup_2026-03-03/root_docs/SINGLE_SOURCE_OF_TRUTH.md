# Single Source of Truth - Stats Display
**Date:** 2026-03-02

## Problem
The live stats display was at risk of becoming a separate source of truth, potentially recalculating metrics independently from the engine's internal state.

## Solution: Direct Engine State Access

### Architecture
```
RenkoEngine (SINGLE SOURCE OF TRUTH)
    ├── _completed: List[LiveTrade]      # All closed trades
    ├── _live_equity: float              # Current account equity  
    ├── _cumulative_pnl: float           # Total P&L
    ├── _in_pos: bool                    # Currently in position?
    ├── _pos_dir: int                    # 1=LONG, -1=SHORT
    ├── _entry_price: float              # Current position entry
    ├── _entry_lots: float               # Current position size
    └── _dir_deque: deque                # Brick sequence history
         │
         └──> _make_results() ──> summary dict
                                   │
                                   └──> _stats_panel() ──> Display
```

### Key Principles

1. **Engine owns all state**
   - Trades, equity, position, bricks
   - NO duplicate tracking

2. **Display reads, never calculates**
   - `_stats_panel()` receives `summary` dict from `engine._make_results()`
   - Live state (position, bricks) read directly from `engine._in_pos`, `engine._dir_deque`
   - NO recalculation of metrics

3. **Analytics layer is canonical**
   - `analyze_trades()` computes all metrics ONCE from `_completed` trades
   - Omega, profit factor, drawdown, etc. come from this single computation

## Implementation

### 1. Stats Panel Enhancement
```python
def _stats_panel(summary: dict, symbol: str, mode: str, engine: Optional["RenkoEngine"] = None) -> Panel:
    """Build a rich Panel from engine state (single source of truth)."""
    
    # Summary metrics from engine._make_results()
    n = summary.get("n_trades", 0)
    net = summary.get("net_usd", 0.0)
    eq = summary.get("final_equity", 0.0)  # = engine._live_equity
    om = summary.get("omega", 0.0)
    
    # Live engine state (optional, for live mode)
    if engine is not None:
        pos_str = "FLAT"
        if engine._in_pos:  # Direct read from engine
            dir_str = "LONG" if engine._pos_dir == 1 else "SHORT"
            pos_str = f"{dir_str} {engine._entry_lots:.3f} lots @ ${engine._entry_price:.2f}"
        
        brick_count = len(engine._dir_deque)  # Direct read from engine
```

**Before (risk of duplication):**
```python
# Could have done this (WRONG):
net_pnl = sum(t.net_usd for t in trades)  # Recalculating!
```

**After (single source):**
```python
# Always do this (RIGHT):
net_pnl = summary.get("net_usd", 0.0)  # From engine._make_results()
```

### 2. Live Stats Panel Loop
```python
def _loop(self) -> None:
    while not self._stop.wait(self._interval):
        # Get results from ENGINE (not recalculate)
        summary = self._engine._make_results().get("summary", {})
        
        # Pass ENGINE reference for live state
        self._live.update(_stats_panel(summary, self._engine.cfg.symbol, "live", self._engine))
        
        # Log position from ENGINE state
        pos = "LONG" if self._engine._in_pos and self._engine._pos_dir == 1 else (
            "SHORT" if self._engine._in_pos and self._engine._pos_dir == -1 else "FLAT"
        )
        LOG.info("... pos=%s bricks=%d", pos, len(self._engine._dir_deque))
```

### 3. Engine Internal Flow
```python
class RenkoEngine:
    def _process_brick(self, ...):
        # Exit logic
        if self._in_pos:
            # Calculate P&L
            gross, friction, net = self._simulate_pnl(...)
            
            # Update state
            self._completed.append(trade)
            self._cumulative_pnl += net
            self._live_equity = self.cfg.initial_equity + self._cumulative_pnl
            
            # Sync with broker if live
            if dispatcher:
                broker_eq = dispatcher.get_equity()
                if broker_eq:
                    self._live_equity = broker_eq  # Override with real balance
    
    def _make_results(self):
        # Single analytics computation
        analytics = analyze_trades(self._completed, initial_equity=self.cfg.initial_equity)
        return {
            "summary": {
                "n_trades": analytics.n_trades,
                "net_usd": analytics.net_pnl,        # From analytics
                "omega": analytics.omega,            # From analytics
                "final_equity": self._live_equity,   # From engine state
                ...
            }
        }
```

## Data Flow Guarantees

### Equity Tracking
```
Broker balance (live only)
    ↓
engine._live_equity (synced after each trade)
    ↓
summary["final_equity"]
    ↓
Display shows: ${eq:,.2f}
```

**Single source:** `engine._live_equity`

### Position Tracking
```
engine._in_pos (bool)
engine._pos_dir (int)  
engine._entry_price (float)
engine._entry_lots (float)
    ↓
Display shows: "LONG 0.486 lots @ $2654.50"
```

**Single source:** `engine._in_pos`, `engine._pos_dir`, etc.

### Trade Metrics
```
engine._completed (List[LiveTrade])
    ↓
analyze_trades() → TradeAnalytics
    ↓
summary["omega"], summary["profit_factor"], etc.
    ↓
Display shows metrics
```

**Single source:** `engine._completed` → `analyze_trades()`

## Display Examples

### Before (minimal info)
```
┌──────────────────────────────────────────────────┐
│ XAUUSD  [LIVE]  2026-03-02 14:30 UTC            │
├──────────────────────────────────────────────────┤
│ Trades         3  (2W / 1L)    Win rate    66.7% │
│ Net P&L        $245.00         Avg trade  $81.67 │
│ Profit factor  2.50            Omega       3.420 │
│ Max drawdown   1.2%            Final equity $10,245│
└──────────────────────────────────────────────────┘
```

### After (live engine state)
```
┌──────────────────────────────────────────────────┐
│ XAUUSD  [LIVE]  2026-03-02 14:30 UTC            │
├──────────────────────────────────────────────────┤
│ Position    LONG 0.486 lots @ $2654.50  Bricks    127│
│ Trades         3  (2W / 1L)    Win rate    66.7% │
│ Net P&L        $245.00         Avg trade  $81.67 │
│ Profit factor  2.50            Omega       3.420 │
│ Max drawdown   1.2%            Live equity $48,834│
└──────────────────────────────────────────────────┘
```

**All values from engine state - ZERO recalculation**

## Verification Checklist

- [x] **Stats panel** receives `engine` reference
- [x] **Live position** read from `engine._in_pos`, `engine._pos_dir`
- [x] **Brick count** read from `len(engine._dir_deque)`
- [x] **Equity** from `summary["final_equity"]` which = `engine._live_equity`
- [x] **All metrics** from `summary` dict = `engine._make_results()`
- [x] **No duplication** - NO recalculation anywhere in display code

## Files Modified

1. **scripts/renko_engine.py**
   - `_stats_panel()` - Added `engine` parameter for live state
   - `_LiveStatsPanel._loop()` - Pass engine reference
   - Added `TYPE_CHECKING` import for type hint

2. **No changes to trading_engine.py**
   - Already had `_make_results()` as single source
   - Display code just needed to USE it properly

## Anti-Patterns Avoided

### ❌ BAD - Recalculating metrics
```python
def _stats_panel(trades, ...):
    net = sum(t.net_usd for t in trades)  # Duplicate calculation!
    winners = [t for t in trades if t.net_usd > 0]
    win_rate = len(winners) / len(trades)  # Duplicate!
```

### ✅ GOOD - Reading from engine
```python
def _stats_panel(summary, engine, ...):
    net = summary.get("net_usd", 0.0)     # From engine._make_results()
    win_rate = summary.get("win_rate", 0) # From analyze_trades()
    position = engine._in_pos             # Direct engine state
```

### ❌ BAD - Tracking position separately
```python
class _LiveStatsPanel:
    def __init__(self):
        self._my_position = None  # Duplicate state!
    
    def on_trade_open(self, ...):
        self._my_position = {...}  # Out of sync risk!
```

### ✅ GOOD - Reading engine state
```python
class _LiveStatsPanel:
    def _loop(self):
        if self._engine._in_pos:  # Read from engine
            pos_str = f"{self._engine._pos_dir}..."
```

## Benefits

1. **No sync issues** - Only one place updates state (engine)
2. **No calculation bugs** - Analytics computed once in `analyze_trades()`
3. **Live accuracy** - Display shows EXACTLY what engine sees
4. **Performance** - No duplicate work
5. **Maintainability** - Change metric logic in ONE place

## Testing

```python
# Verify single source of truth:
engine = RenkoEngine(cfg)
results = engine.backtest(closes)

# These MUST match:
assert results["summary"]["final_equity"] == engine._live_equity
assert results["summary"]["n_trades"] == len(engine._completed)

# Display uses same data:
panel = _stats_panel(results["summary"], "TEST", "backtest", engine)
# Panel shows engine._live_equity, len(engine._dir_deque), etc.
```

## Compliance

✅ **DRY** - Single calculation of all metrics  
✅ **First principles** - Display shows raw state, not derivatives  
✅ **Type safety** - `Optional["RenkoEngine"]` with TYPE_CHECKING  
✅ **No magic numbers** - All from engine config  
✅ **Logging** - Engine state logged to file  

---

**Summary:** The display is now a pure **read-only view** of engine state. The engine is the **single source of truth** for all trading state, metrics, and calculations. Zero duplication, zero drift risk.
