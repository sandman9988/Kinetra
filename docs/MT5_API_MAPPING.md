# MT5 API Mapping

This document maps Kinetra's trading system to MetaTrader 5's CTrade API to ensure sim-to-real alignment.

## Trade Operations Mapping

### Position Operations

| MT5 CTrade Method | Kinetra Equivalent | Status | Notes |
|-------------------|-------------------|--------|-------|
| `PositionOpen()` | `RealisticBacktester.run()` with 'open_long'/'open_short' signal | ✅ | Validates freeze zones, stop levels before opening |
| `PositionModify()` | Planned | ⏳ | Will add modify_position() with freeze zone checks |
| `PositionClose()` | `RealisticBacktester.run()` with 'close' signal | ✅ | Applies spread, commission, swap, slippage |
| `PositionClosePartial()` | Planned | ⏳ | Partial close with prorated costs |
| `PositionCloseBy()` | Not implemented | ❌ | Requires hedging mode support |

### Order Operations

| MT5 CTrade Method | Kinetra Equivalent | Status | Notes |
|-------------------|-------------------|--------|-------|
| `OrderOpen()` | Pending orders system | ⏳ | Will add pending order queue |
| `OrderModify()` | Planned | ⏳ | Modify pending order with freeze zone checks |
| `OrderDelete()` | Planned | ⏳ | Cancel pending order |

### Simplified Trade Methods

| MT5 CTrade Method | Kinetra Signal Format | Status |
|-------------------|----------------------|--------|
| `Buy()` | `{'action': 'open_long', 'sl': X, 'tp': Y, 'volume': Z}` | ✅ |
| `Sell()` | `{'action': 'open_short', 'sl': X, 'tp': Y, 'volume': Z}` | ✅ |
| `BuyLimit()` | Pending order type | ⏳ |
| `BuyStop()` | Pending order type | ⏳ |
| `SellLimit()` | Pending order type | ⏳ |
| `SellStop()` | Pending order type | ⏳ |

## Parameter Settings Mapping

| MT5 Setting | Kinetra Equivalent | Implementation |
|-------------|-------------------|----------------|
| `SetAsyncMode()` | Not needed (backtest is synchronous) | N/A |
| `SetExpertMagicNumber()` | Agent ID / Magic number tracking | Planned |
| `SetDeviationInPoints()` | Slippage model | ✅ Implemented in `simulate_fill()` |
| `SetTypeFilling()` | `SymbolSpec.filling_mode` | ✅ FOK/IOC/BOC |
| `SetMarginMode()` | `SymbolSpec.trade_calc_mode` | ✅ Forex/CFD/Futures |

## Request/Result Tracking

### Request Parameters (MqlTradeRequest)

| MT5 Field | Kinetra Equivalent | Status |
|-----------|-------------------|--------|
| `action` | Signal 'action' field | ✅ |
| `symbol` | `SymbolSpec.symbol` | ✅ |
| `volume` | Signal 'volume' field | ✅ |
| `price` | Current candle price | ✅ |
| `stoplimit` | Signal 'sl'/'tp' fields | ✅ |
| `sl` | Signal 'sl' field | ✅ |
| `tp` | Signal 'tp' field | ✅ |
| `deviation` | Slippage tolerance | ✅ |
| `type` | Order type (market/limit/stop) | ⏳ |
| `type_filling` | `filling_mode` | ✅ |
| `type_time` | Expiration mode | ⏳ |
| `expiration` | Order expiration time | ⏳ |
| `comment` | Trade comment/metadata | ⏳ |
| `position` | Position ticket | ⏳ |
| `position_by` | Opposite position (for close by) | ❌ |
| `magic` | Expert magic number | ⏳ |

### Result Codes (MqlTradeResult)

| MT5 Retcode | Kinetra Error Code | Implementation |
|-------------|-------------------|----------------|
| `TRADE_RETCODE_DONE` (10009) | `MT5ErrorCode.SUCCESS` | ✅ |
| `TRADE_RETCODE_INVALID_STOPS` (10016) | `MT5ErrorCode.INVALID_STOPS` | ✅ |
| `TRADE_RETCODE_FROZEN` (10029) | `MT5ErrorCode.FROZEN` | ✅ |
| `TRADE_RETCODE_INVALID_FILL` (10030) | `MT5ErrorCode.INVALID_FILL` | ✅ |
| Other error codes | Planned | ⏳ |

### Check Results (MqlTradeCheckResult)

| MT5 Field | Kinetra Equivalent | Status |
|-----------|-------------------|--------|
| `retcode` | Validation result code | ✅ |
| `balance` | Account balance after trade | ✅ |
| `equity` | Account equity after trade | ✅ |
| `profit` | Floating P&L | ✅ |
| `margin` | Required margin | ⏳ |
| `margin_free` | Free margin after trade | ⏳ |
| `margin_level` | Margin level % | ⏳ |
| `comment` | Error/warning message | ✅ |

## Symbol Properties Mapping

### Integer Properties (ENUM_SYMBOL_INFO_INTEGER)

| MT5 Property | Kinetra SymbolSpec Field | Status |
|-------------|-------------------------|--------|
| `SYMBOL_DIGITS` | `digits` | ✅ |
| `SYMBOL_SPREAD` | `spread_typical` | ✅ |
| `SYMBOL_SPREAD_FLOAT` | Dynamic spread from CSV | ✅ |
| `SYMBOL_TRADE_STOPS_LEVEL` | `trade_stops_level` | ✅ |
| `SYMBOL_TRADE_FREEZE_LEVEL` | `trade_freeze_level` | ✅ |
| `SYMBOL_TRADE_CALC_MODE` | `trade_calc_mode` | ✅ |
| `SYMBOL_TRADE_MODE` | `trade_mode` | ✅ |
| `SYMBOL_TRADE_EXEMODE` | `order_mode` | ✅ |
| `SYMBOL_SWAP_MODE` | `swap_type` | ⚠️ Partial (points only) |
| `SYMBOL_SWAP_ROLLOVER3DAYS` | `swap_triple_day` | ✅ |
| `SYMBOL_FILLING_MODE` | `filling_mode` | ✅ |
| `SYMBOL_ORDER_MODE` | `order_mode` | ✅ |

### Double Properties (ENUM_SYMBOL_INFO_DOUBLE)

| MT5 Property | Kinetra SymbolSpec Field | Status |
|-------------|-------------------------|--------|
| `SYMBOL_BID` | Current candle 'close' - spread/2 | ✅ |
| `SYMBOL_ASK` | Current candle 'close' + spread/2 | ✅ |
| `SYMBOL_POINT` | `point` | ✅ |
| `SYMBOL_TRADE_TICK_SIZE` | `tick_size` | ✅ |
| `SYMBOL_TRADE_CONTRACT_SIZE` | `contract_size` | ✅ |
| `SYMBOL_VOLUME_MIN` | `volume_min` | ✅ |
| `SYMBOL_VOLUME_MAX` | `volume_max` | ✅ |
| `SYMBOL_VOLUME_STEP` | `volume_step` | ✅ |
| `SYMBOL_SWAP_LONG` | `swap_long` | ✅ |
| `SYMBOL_SWAP_SHORT` | `swap_short` | ✅ |
| `SYMBOL_SWAP_SUNDAY` through `SYMBOL_SWAP_SATURDAY` | Not implemented | ⏳ |

### String Properties (ENUM_SYMBOL_INFO_STRING)

| MT5 Property | Kinetra SymbolSpec Field | Status |
|-------------|-------------------------|--------|
| `SYMBOL_DESCRIPTION` | `description` | ✅ |
| `SYMBOL_CURRENCY_BASE` | `base_currency` | ✅ |
| `SYMBOL_CURRENCY_PROFIT` | `quote_currency` | ✅ |
| `SYMBOL_CURRENCY_MARGIN` | `margin_currency` | ✅ |
| `SYMBOL_SECTOR` | `asset_class` | ✅ |
| `SYMBOL_INDUSTRY` | Planned | ⏳ |

## Friction Costs Mapping

### Cost Components

| Cost Type | MT5 Implementation | Kinetra Implementation | Status |
|-----------|-------------------|------------------------|--------|
| **Spread** | Bid-Ask difference | Dynamic per-candle from CSV | ✅ |
| **Commission** | Per lot, per side | `commission_per_lot * volume * 2` | ✅ |
| **Swap** | Daily rollover | `swap_points * point * contract_size * days_held` | ✅ |
| **Triple Swap** | 3x on configured day | Wednesday (configurable) 3x multiplier | ✅ |
| **Slippage** | Market execution | Physics-based from volume | ✅ |

### Swap Calculation Modes (ENUM_SYMBOL_SWAP_MODE)

| Mode | MT5 ID | Kinetra Status |
|------|--------|----------------|
| Points | `SYMBOL_SWAP_MODE_POINTS` | ✅ Implemented |
| Currency (symbol) | `SYMBOL_SWAP_MODE_CURRENCY_SYMBOL` | ⏳ Planned |
| Currency (margin) | `SYMBOL_SWAP_MODE_CURRENCY_MARGIN` | ⏳ Planned |
| Currency (deposit) | `SYMBOL_SWAP_MODE_CURRENCY_DEPOSIT` | ⏳ Planned |
| Currency (profit) | `SYMBOL_SWAP_MODE_CURRENCY_PROFIT` | ⏳ Planned |
| Interest (current) | `SYMBOL_SWAP_MODE_INTEREST_CURRENT` | ⏳ Planned |
| Interest (open) | `SYMBOL_SWAP_MODE_INTEREST_OPEN` | ⏳ Planned |
| Reopen (current) | `SYMBOL_SWAP_MODE_REOPEN_CURRENT` | ⏳ Planned |
| Reopen (bid) | `SYMBOL_SWAP_MODE_REOPEN_BID` | ⏳ Planned |

## Trade Execution Modes Mapping

### Execution Types (ENUM_SYMBOL_TRADE_EXECUTION)

| Mode | MT5 ID | Kinetra Support |
|------|--------|-----------------|
| Request Execution | `SYMBOL_TRADE_EXECUTION_REQUEST` | ⏳ |
| Instant Execution | `SYMBOL_TRADE_EXECUTION_INSTANT` | ⏳ |
| Market Execution | `SYMBOL_TRADE_EXECUTION_MARKET` | ✅ Default |
| Exchange Execution | `SYMBOL_TRADE_EXECUTION_EXCHANGE` | ⏳ |

### Filling Policies (SYMBOL_FILLING_MODE)

| Policy | MT5 ID | Kinetra Support |
|--------|--------|-----------------|
| Fill or Kill (FOK) | `SYMBOL_FILLING_FOK` | ✅ |
| Immediate or Cancel (IOC) | `SYMBOL_FILLING_IOC` | ✅ |
| Book or Cancel (BOC) | `SYMBOL_FILLING_BOC` | ⏳ |
| Return | No ID (default) | ✅ |

## Constraint Validation Mapping

### Stop Level Validation

```python
# MT5: Validates SL/TP distance from current price
SYMBOL_TRADE_STOPS_LEVEL = 100  # points

# Kinetra: validate_stops()
def validate_stops(current_price, sl, tp, stops_level):
    if abs(current_price - sl) < stops_level * point:
        return MT5ErrorCode.INVALID_STOPS
    if abs(current_price - tp) < stops_level * point:
        return MT5ErrorCode.INVALID_STOPS
    return MT5ErrorCode.SUCCESS
```

### Freeze Zone Validation

```python
# MT5: Blocks modifications near session close
SYMBOL_TRADE_FREEZE_LEVEL = 50  # points

# Kinetra: is_in_freeze_zone()
def is_in_freeze_zone(signal_time):
    # Check if within session close window
    if near_session_close and within_freeze_distance:
        return True  # Block modification
    return False
```

## History Classes Mapping

### CHistoryOrderInfo

| MT5 Property | Kinetra Trade Field | Status |
|-------------|-------------------|--------|
| `TimeSetup()` | `entry_time` | ✅ |
| `TimeDone()` | `exit_time` | ✅ |
| `OrderType()` | `direction` (1=long, -1=short) | ✅ |
| `VolumeInitial()` | `volume` | ✅ |
| `PriceOpen()` | `entry_price` | ✅ |
| `StopLoss()` | Tracked separately | ⏳ |
| `TakeProfit()` | Tracked separately | ⏳ |
| `Symbol()` | Symbol name from spec | ✅ |
| `Magic()` | Agent ID | ⏳ |
| `PositionId()` | Position tracking | ⏳ |

### CDealInfo

| MT5 Property | Kinetra Trade Field | Status |
|-------------|-------------------|--------|
| `Time()` | Trade timestamp | ✅ |
| `DealType()` | Entry/Exit | ✅ |
| `Volume()` | `volume` | ✅ |
| `Price()` | `entry_price` / `exit_price` | ✅ |
| `Commission()` | `commission` | ✅ |
| `Swap()` | `swap` | ✅ |
| `Profit()` | `pnl` | ✅ |
| `Symbol()` | Symbol name | ✅ |
| `Magic()` | Agent ID | ⏳ |

## Testing Coverage

| Test File | MT5 Operations Tested | Status |
|-----------|----------------------|--------|
| `test_friction_costs.py` | Spread, commission, swap, triple swap | ✅ 5/5 pass |
| `test_end_to_end.py` | Full pipeline with MT5 constraints | ✅ 9/9 pass |
| `test_trade_lifecycle_real_data.py` | Freeze zones, stop levels, costs | ✅ |
| `test_doppelganger_triad.py` | Agent management | ✅ 8/8 pass |
| `test_portfolio_health.py` | Health monitoring | ✅ 7/7 pass |

## Legend

- ✅ Fully implemented
- ⚠️ Partially implemented
- ⏳ Planned
- ❌ Not planned (out of scope)

## Notes

1. **Async Mode**: Not needed for backtesting (synchronous by design)
2. **Hedging Mode**: Not currently supported (netting mode only)
3. **Pending Orders**: Will be added in future release
4. **Position Modification**: Will track SL/TP changes with freeze zone validation
5. **Swap Modes**: Currently only "points" mode supported, others planned

## Validation Strategy

To ensure sim-to-real alignment:

1. ✅ All friction costs calculated using MT5 formulas
2. ✅ Freeze zones and stop levels enforced
3. ✅ Error codes match MT5 return codes
4. ✅ Symbol specifications mirror MT5 properties
5. ⏳ Will add position modification tracking
6. ⏳ Will add pending order queue
7. ⏳ Will add margin requirement calculations

## Account Information Mapping (CAccountInfo)

### Account State Properties

| MT5 CAccountInfo Method | Kinetra Equivalent | Status | Notes |
|------------------------|-------------------|--------|-------|
| `Login()` | Account ID tracking | ⏳ | Will add account_id field |
| `TradeMode()` | Account type (demo/real/contest) | ⏳ | Metadata field |
| `Leverage()` | `RiskManager` leverage setting | ✅ | Used in margin calculations |
| `StopoutMode()` | Margin call threshold | ⏳ | Percentage or money terms |
| `MarginMode()` | Margin calculation mode | ✅ | Retail/Exchange/Hedging |
| `TradeAllowed()` | Always true in backtest | ✅ | Live trading flag |
| `TradeExpert()` | Always true (expert trading) | ✅ | EA permission flag |
| `LimitOrders()` | Max pending orders | ⏳ | Broker limit |

### Account Balance Properties

| MT5 CAccountInfo Method | Kinetra Equivalent | Status | Implementation |
|------------------------|-------------------|--------|----------------|
| `Balance()` | `BacktestResult.final_balance` | ✅ | Tracked throughout backtest |
| `Credit()` | Not applicable (backtest) | N/A | Broker credit |
| `Profit()` | `BacktestResult.total_pnl` | ✅ | Floating + realized P&L |
| `Equity()` | `balance + floating_pnl` | ✅ | Real-time equity calculation |
| `Margin()` | Position margin requirement | ⏳ | Sum of all position margins |
| `FreeMargin()` | `equity - margin` | ⏳ | Available for new trades |
| `MarginLevel()` | `(equity / margin) * 100` | ⏳ | Margin level percentage |
| `MarginCall()` | Stop-out threshold | ⏳ | Usually 80-120% |
| `MarginStopOut()` | Force close threshold | ⏳ | Usually 20-50% |

### Account Text Properties

| MT5 CAccountInfo Method | Kinetra Equivalent | Status |
|------------------------|-------------------|--------|
| `Name()` | User/agent name | ⏳ |
| `Server()` | Broker server | ⏳ |
| `Currency()` | Account currency (USD/EUR/etc) | ✅ |
| `Company()` | Broker name | ⏳ |

### Risk Validation Methods

| MT5 CAccountInfo Method | Kinetra Equivalent | Status | Notes |
|------------------------|-------------------|--------|-------|
| `OrderProfitCheck()` | Pre-trade profit estimation | ⏳ | Estimate potential profit |
| `MarginCheck()` | Pre-trade margin validation | ⏳ | Check if enough margin |
| `FreeMarginCheck()` | Free margin after trade | ⏳ | Validate before opening |
| `MaxLotCheck()` | Max position size calculation | ✅ | `RiskManager.calculate_position_size()` |

## Terminal Information Mapping (CTerminalInfo)

### Terminal State Properties

| MT5 CTerminalInfo Method | Kinetra Equivalent | Status | Notes |
|-------------------------|-------------------|--------|-------|
| `Build()` | Software version tracking | ⏳ | Build number |
| `IsConnected()` | Always true (backtest) | ✅ | Connection status |
| `IsDLLsAllowed()` | Not applicable | N/A | DLL permission |
| `IsTradeAllowed()` | Always true (backtest) | ✅ | Manual trading permission |
| `IsEmailEnabled()` | Alert system | ⏳ | Email notifications |
| `IsFtpEnabled()` | Not implemented | ❌ | FTP reports |
| `MaxBars()` | Data buffer size | ✅ | Max candles in memory |
| `CodePage()` | UTF-8 encoding | ✅ | Character encoding |
| `CPUCores()` | System info | ⏳ | Parallelization |
| `MemoryPhysical()` | System RAM | ⏳ | Memory monitoring |
| `MemoryTotal()` | Process memory limit | ⏳ | Memory tracking |
| `MemoryAvailable()` | Free memory | ⏳ | Memory monitoring |
| `MemoryUsed()` | Process memory usage | ⏳ | Memory tracking |
| `IsX64()` | Platform architecture | ✅ | 64-bit Python |
| `OpenCLSupport()` | GPU support | ❌ | Not implemented |
| `DiskSpace()` | Available disk space | ⏳ | Storage monitoring |

### Terminal Path Properties

| MT5 CTerminalInfo Method | Kinetra Equivalent | Status |
|-------------------------|-------------------|--------|
| `Language()` | System locale | ⏳ |
| `Name()` | "Kinetra Backtester" | ✅ |
| `Company()` | "Kinetra Team" | ✅ |
| `Path()` | Installation directory | ✅ |
| `DataPath()` | `./data/` directory | ✅ |
| `CommonDataPath()` | Shared data directory | ⏳ |

## Account Margin Calculation

### Margin Formula Mapping

Different asset classes use different margin formulas:

#### Forex (SYMBOL_CALC_MODE_FOREX)

```python
# MT5 Formula:
margin = (lots * contract_size) / leverage * margin_rate

# Kinetra Implementation:
def calculate_margin_forex(volume, contract_size, price, leverage):
    return (volume * contract_size * price) / leverage
```

#### CFD (SYMBOL_CALC_MODE_CFD)

```python
# MT5 Formula:
margin = lots * contract_size * market_price * margin_rate

# Kinetra Implementation:
def calculate_margin_cfd(volume, contract_size, price, margin_rate=1.0):
    return volume * contract_size * price * margin_rate
```

#### Futures (SYMBOL_CALC_MODE_FUTURES)

```python
# MT5 Formula:
margin = lots * initial_margin * margin_rate

# Kinetra Implementation:
def calculate_margin_futures(volume, initial_margin, margin_rate=1.0):
    return volume * initial_margin * margin_rate
```

### Margin Monitoring

| Metric | Formula | Threshold | Action |
|--------|---------|-----------|--------|
| **Margin Level** | `(equity / margin) * 100` | >100% | Normal trading |
| **Margin Call** | Margin level drops below 100-120% | 100% | Warning, no new positions |
| **Stop Out** | Margin level drops below 20-50% | 50% | Force close positions |

```python
# Example: Margin level check
def check_margin_level(equity, margin):
    if margin == 0:
        return float('inf')  # No positions
    
    margin_level = (equity / margin) * 100
    
    if margin_level < 50:  # Stop out
        return "STOP_OUT"
    elif margin_level < 100:  # Margin call
        return "MARGIN_CALL"
    else:
        return "OK"
```

## Integration with Portfolio Health Monitor

The `PortfolioHealthMonitor` class integrates account state monitoring:

| Health Pillar | Account Metrics Used | Weight |
|---------------|---------------------|--------|
| Return & Efficiency | Balance, Equity, Profit | 25% |
| Downside Risk | Max Drawdown, Margin Level | 30% |
| Structural Stability | Correlation, Position count | 25% |
| Behavioral Health | Trade frequency, Win rate | 20% |

### Automated Risk Scaling

```python
# Based on margin level and health score
def calculate_risk_multiplier(margin_level, health_score):
    if margin_level < 100:  # Margin call
        return 0.0  # No new trades
    elif health_score < 40:  # CRITICAL
        return 0.25  # 75% risk reduction
    elif health_score < 60:  # DEGRADED
        return 0.50  # 50% risk reduction
    elif health_score < 80:  # WARNING
        return 0.70  # 30% risk reduction
    else:  # HEALTHY
        return 1.0  # Normal risk
```

## Future Enhancements

### Account Management

1. ⏳ **Real-time margin tracking**: Calculate required margin per position
2. ⏳ **Margin call simulation**: Stop-out when margin level drops below threshold
3. ⏳ **Multi-account support**: Track multiple accounts simultaneously
4. ⏳ **Account hedging mode**: Support for hedging (multiple positions same symbol)

### Risk Controls

1. ⏳ **Max daily loss**: Stop trading when daily loss exceeds threshold
2. ⏳ **Max position count**: Limit concurrent positions
3. ⏳ **Max volume per symbol**: Prevent over-concentration
4. ⏳ **Account balance protection**: Stop when equity drops below threshold

### Monitoring

1. ⏳ **Memory usage tracking**: Monitor process memory consumption
2. ⏳ **Disk space monitoring**: Alert when storage is low
3. ⏳ **Connection health**: Track data feed latency and connectivity
4. ⏳ **Performance metrics**: CPU usage, trade execution speed

