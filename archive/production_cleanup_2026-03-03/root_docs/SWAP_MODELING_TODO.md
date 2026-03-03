# Swap Cost Modeling — TODO

## Current Status

**Backtest mode**: Swap costs are **$0** (not modeled)
- This **understates friction** for positions held overnight
- Average hold time is 0.2h, so most trades close same-day
- However, some positions may span multiple days

**Live mode**: Swap costs are **$0** (not yet implemented)
- Should use actual swap charged by broker

## Impact on Results

From 24-month XAUUSD backtest (compounding):
- Total friction: **$96.92M**
  - Spread: $89.12M (92%)
  - Commission: $7.80M (8%)
  - **Swap: $0.00 (not modeled)**

For strategies with longer hold times, swap could be significant:
- XAUUSD swap rate: ~$10/lot/day (varies by broker)
- If avg hold = 24h and avg position = 10 lots → ~$100/trade
- With 56,148 trades → potential **$5.6M additional friction**

## Implementation Plan

### Backtest Mode

```python
# Estimate swap from hold duration
hold_days = (exit_time - entry_time).total_seconds() / 86400
swap_rate = spec.swap_long_usd_per_lot_per_eff_day if direction == 1 else spec.swap_short_usd_per_lot_per_eff_day

# Triple swap day handling (Wednesday = 3× charge)
if exit_time.weekday() == 2:  # Wednesday
    swap_multiplier = 3.0
else:
    swap_multiplier = 1.0

swap_usd = swap_rate * lots * hold_days * swap_multiplier
```

### Live Mode

```python
# Use actual swap from broker dispatcher
swap_usd = dispatcher.get_swap_charged(order_id)
# Or compute from spec if not available from broker
```

## Data Requirements

1. **Swap rates** from `contract_spec.json`:
   - `swap_long_usd_per_lot_per_eff_day`
   - `swap_short_usd_per_lot_per_eff_day`
   - `triple_swap_day` (usually 3 = Wednesday)

2. **Hold duration** per trade:
   - Already tracked: `entry_time`, `exit_time`

3. **Triple swap calendar**:
   - Need to know which day has 3× swap (broker-specific)
   - Usually Wednesday for forex/metals

## Priority

**Low** for current strategy:
- Average hold time is 0.2h (12 minutes)
- Most trades close before swap accrual
- Spread + commission = 99%+ of friction costs

**Medium** if strategy evolves:
- Longer-term positions (hold >4h)
- Overnight holding becomes material
- Accuracy requirements increase for live deployment

## References

- `kinetra/friction_cost.py`: `InstrumentSpec.swap_long_usd_per_lot_per_eff_day`
- `kinetra/renko/trading_engine.py`: `_simulate_pnl()` line 350
- `kinetra/renko/live_trader.py`: `LiveTrade.swap_usd`
