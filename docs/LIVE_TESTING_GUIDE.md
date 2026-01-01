# Kinetra Live Testing Guide

## Overview

The Live Testing module provides a safe, progressive pathway from virtual testing to live trading with comprehensive safety gates and monitoring.

## Testing Modes

### 1. Virtual Trading (Paper Trading)

**Purpose**: Test agent logic without any real connection.

**Features**:
- No MT5 connection required
- Uses synthetic data stream
- Safe testing environment
- Identical code to live trading
- Real-time CHS monitoring

**Usage**:
```bash
# Via menu
python kinetra_menu.py
# Select: 4 (Live Testing) > 1 (Virtual Trading)

# Via command line
python scripts/testing/run_live_test.py --mode virtual --symbol EURUSD --duration 60
```

**When to use**:
- Initial agent development
- Quick sanity checks
- Testing new strategies
- Debugging agent logic

### 2. Demo Account Testing

**Purpose**: Test with real market data and execution on demo account.

**Requirements**:
- MT5 terminal running
- Demo account configured
- MetaTrader5 Python package installed (`pip install MetaTrader5`)

**Features**:
- Real MT5 connection
- Real market data
- Real trade execution (demo money)
- Full validation before live

**Usage**:
```bash
# Via menu
python kinetra_menu.py
# Select: 4 (Live Testing) > 2 (Demo Account Testing)

# Via command line
python scripts/testing/run_live_test.py --mode demo --symbol EURUSD --duration 30
```

**Safety checks**:
- Requires explicit confirmation
- Limited max trades (default: 5)
- CHS circuit breakers active
- All trades validated

**When to use**:
- After successful virtual testing
- Before live deployment
- Testing MT5 integration
- Validating real market behavior

### 3. Connection Testing

**Purpose**: Verify MT5 connection and account access.

**Usage**:
```bash
# Via menu
python kinetra_menu.py
# Select: 4 (Live Testing) > 3 (Test MT5 Connection)

# Via command line
python scripts/testing/run_live_test.py --test-connection
```

**Verifies**:
- MT5 terminal is running
- Python can connect to MT5
- Account is accessible
- Automated trading is enabled
- Account balance and margin

## Safety Features

### Circuit Breakers

Automatically halt trading when Composite Health Score (CHS) drops below threshold:

```python
# Default threshold
chs_threshold = 0.55

# Circuit breaker activates when:
if CHS < 0.55:
    halt_trading()
    log_warning("Circuit breaker activated")
```

**Recovery**:
- Monitor CHS continuously
- Resume when CHS > (threshold + 0.1)
- Manual override not allowed

### Trade Limits

Prevent runaway execution:

```python
--max-trades 10    # Maximum trades per session
--duration 60      # Maximum duration in minutes
```

**Automatic stop**:
- Stops after max trades reached
- Stops after duration elapsed
- Stops on circuit breaker (if CHS doesn't recover)

### Order Validation

All orders validated before execution:

```python
from kinetra.order_validator import OrderValidator

validator = OrderValidator(spec, auto_adjust_stops=True)

# Validates:
# - Stop loss meets minimum distance
# - Take profit meets minimum distance
# - Not in freeze zone
# - Volume within limits
# - Price within trading hours
```

**Auto-adjustment**:
- Invalid stops automatically adjusted
- Ensures MT5 compliance
- Prevents rejected orders

## Testing Progression

**ALWAYS follow this order:**

### Step 1: Virtual Trading
```
Duration: 1-2 hours
Max Trades: 20-50
Success Criteria:
  ✓ No crashes
  ✓ Agent logic works correctly
  ✓ Orders are valid
  ✓ CHS > 0.75 consistently
```

### Step 2: Demo Account Testing
```
Duration: 1-7 days
Max Trades: 100-500
Success Criteria:
  ✓ Real market data handled correctly
  ✓ Orders execute successfully
  ✓ CHS > 0.90 consistently
  ✓ Omega Ratio > 2.7
  ✓ % Energy Captured > 65%
  ✓ No validator rejections
```

### Step 3: Live Trading (NOT in menu - requires approval)
```
⚠️  CRITICAL: Only proceed after:
  1. Successful demo testing for minimum 1 week
  2. All performance targets met
  3. Manual review of all trades
  4. Risk management approved
  5. Kill switch tested

Start with:
  - Minimal capital
  - Low max trades (5-10)
  - Continuous monitoring
  - Manual review every day
```

## Performance Targets

Before moving to next stage:

| Metric | Target | Description |
|--------|--------|-------------|
| **CHS** | > 0.90 | Composite Health Score |
| **Omega Ratio** | > 2.7 | Upside/downside probability ratio |
| **% Energy Captured** | > 65% | Physics-based efficiency |
| **Z-Factor** | > 2.5 | Statistical robustness |
| **Validator Rejections** | 0 | All orders valid |
| **Circuit Breaker Events** | < 5% | Minimal halts |

## Command Line Options

```bash
python scripts/testing/run_live_test.py [OPTIONS]

Options:
  --mode {virtual,demo,live}    Testing mode (default: virtual)
  --symbol SYMBOL               Trading symbol (default: EURUSD)
  --agent {ppo,dqn,linear,berserker,triad}
                                Agent type (default: ppo)
  --duration MINUTES            Test duration in minutes (default: 60)
  --max-trades N                Maximum trades (default: 10)
  --chs-threshold FLOAT         CHS circuit breaker (default: 0.55)
  --test-connection             Test MT5 connection only

Examples:
  # Virtual trading, 2 hours, max 20 trades
  python scripts/testing/run_live_test.py --mode virtual --duration 120 --max-trades 20
  
  # Demo testing, short session
  python scripts/testing/run_live_test.py --mode demo --duration 30 --max-trades 5
  
  # Test different agent
  python scripts/testing/run_live_test.py --mode virtual --agent dqn
  
  # Connection test only
  python scripts/testing/run_live_test.py --test-connection
```

## Setup Instructions

### Installing MT5 Python Package

```bash
pip install MetaTrader5
```

### Configuring MT5 for Automated Trading

1. **Launch MT5 Terminal**
   - Open MetaTrader 5
   - Login to your account (demo or live)

2. **Enable Automated Trading**
   - Tools → Options → Expert Advisors
   - ✓ Allow automated trading
   - ✓ Allow DLL imports
   - ✓ Disable "Ask manual confirmation"

3. **Verify Settings**
   ```bash
   python scripts/testing/run_live_test.py --test-connection
   ```

### Troubleshooting

**Error: "MetaTrader5 not installed"**
```bash
pip install MetaTrader5
```

**Error: "Failed to connect to MT5"**
- Make sure MT5 terminal is running
- Check that account is logged in
- Verify automated trading is enabled

**Error: "Symbol not found"**
- Check symbol spelling (e.g., EURUSD vs EURUSD+)
- Verify symbol is available in your broker
- Use Market Watch in MT5 to confirm

**High validator rejections**
- Check stop levels: `trade_stops_level` in symbol info
- Verify freeze levels: `trade_freeze_level`
- Use auto-adjust: `OrderValidator(spec, auto_adjust_stops=True)`

## Monitoring

### Real-time Logs

All live tests log to console with timestamps:

```
2024-01-01 15:00:00 - INFO - STARTING LIVE TEST - VIRTUAL MODE
2024-01-01 15:00:00 - INFO - Symbol: EURUSD
2024-01-01 15:00:00 - INFO - Agent: PPO
2024-01-01 15:00:05 - INFO - ✅ Virtual mode initialized
2024-01-01 15:00:05 - INFO - 🚀 Test started - monitoring for circuit breakers...
2024-01-01 15:01:23 - INFO - ✅ Trade 1: open_long @ 1.08503
2024-01-01 15:02:15 - WARNING - ⚠️ CIRCUIT BREAKER ACTIVATED - CHS 0.51 < 0.55
2024-01-01 15:03:00 - INFO - Iteration 10 - Trades: 1, CHS: 0.51
```

### CHS History

Track Composite Health Score over time:

```python
# CHS calculated every iteration
# Circuit breaker activates if CHS < threshold
# History saved in results
results['chs_history'] = [0.75, 0.78, 0.82, ...]
```

### Trade Log

Every trade is logged with full details:

```python
{
    'timestamp': '2024-01-01T15:01:23',
    'action': 'open_long',
    'price': 1.08500,
    'sl': 1.08480,
    'tp': 1.08540,
    'success': True,
    'fill_price': 1.08503,
    'error': None
}
```

## Best Practices

### DO:
- ✅ Start with virtual trading
- ✅ Test on demo for at least 1 week
- ✅ Monitor CHS continuously
- ✅ Keep detailed logs
- ✅ Review all trades manually
- ✅ Respect circuit breakers
- ✅ Use conservative trade limits

### DON'T:
- ❌ Skip virtual testing
- ❌ Deploy to live without demo validation
- ❌ Override circuit breakers
- ❌ Ignore validator rejections
- ❌ Test with large position sizes
- ❌ Run unmonitored overnight
- ❌ Disable safety features

## Integration with Menu System

The live testing module is fully integrated into the Kinetra menu:

```
Main Menu → 4. Live Testing
├── 1. Virtual Trading (Paper Trading)
│   ├─→ Configure: symbol, agent, duration, limits
│   ├─→ Run test
│   └─→ View results
├── 2. Demo Account Testing
│   ├─→ Safety confirmation
│   ├─→ Configure: symbol, agent, duration, limits
│   ├─→ Run test
│   └─→ View results
├── 3. Test MT5 Connection
│   ├─→ Verify MT5 running
│   ├─→ Check account access
│   └─→ Display account info
└── 4. View Live Testing Guide
    └─→ Complete documentation
```

## Security Notes

- **Never commit credentials**: Use `.env` file (ignored by git)
- **Use demo first**: Always validate on demo before live
- **Monitor continuously**: Never run unattended initially
- **Start small**: Minimal position sizes and trade counts
- **Kill switch ready**: Know how to stop immediately
- **Review logs**: Check every trade manually at first

## Support

For issues or questions:
1. Check logs in `logs/` directory
2. Review this guide
3. Test connection: `python scripts/testing/run_live_test.py --test-connection`
4. Start with virtual mode to debug

---

**Remember**: Live trading involves real financial risk. Always test thoroughly in virtual and demo modes first!
