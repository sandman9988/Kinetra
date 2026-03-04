# Preflight Checks — Live Trading Safety

Last updated: 2026-03-04

## Overview

Before executing any **REAL ORDERS**, the system runs comprehensive preflight checks. These checks validate the entire trading pipeline from network connectivity to broker execution.

**Enhanced preflight includes:**
- DNS resolution validation
- TCP connection pool health
- Heartbeat/keep-alive verification
- Broker session validation
- Account balance & margin checks
- Symbol resolution
- Market hours validation
- Hot standby verification
- Connection health service check
- Optional test order execution

## Quick Reference

```bash
# Run with preflight (recommended)
python scripts/renko_engine.py XAUUSD --stage live --live-size micro \
  --preflight-test-order --preflight-lots 0.01 \
  --ack-live I_UNDERSTAND_LIVE_RISK

# Skip test order (faster)
python scripts/renko_engine.py XAUUSD --stage live --live-size micro
```

## Check Details

### 1. DNS Resolution
```
[1/10] DNS resolution... ✓ PASS
  demo.ctraderapi.com -> 1.2.3.4, 5.6.7.8 (45.2ms)
```

**What it checks:**
- Broker endpoint hostname resolves to valid IP
- Multiple DNS resolvers tested (if configured)
- Resolution time < 1000ms

**Failure mode:**
- Network misconfiguration
- DNS server issues
- Broker endpoint changed

**Action on fail:** Block trading

---

### 2. TCP Connection Pool
```
[2/10] TCP connection pool... ✓ PASS
  Connected to demo.ctraderapi.com:5035
  Hot-standby: UP
```

**What it checks:**
- Active TCP connection to broker
- Hot standby status (if enabled)
- Endpoint accessibility

**Failure mode:**
- Firewall blocking
- Network partition
- Broker maintenance

**Action on fail:** Block trading

---

### 3. Heartbeat/Keep-Alive
```
[3/10] Heartbeat... ✓ PASS (23.1ms)
```

**What it checks:**
- Round-trip latency to broker
- Keep-alive message success
- Latency < 500ms (ideal), < 1000ms (acceptable)

**Failure mode:**
- Network congestion
- Broker overload
- Route instability

**Action on fail:** Block if > 1000ms

---

### 4. Broker Session
```
[4/10] Broker session... ✓ PASS
  Account: 12345
  Broker: Pepperstone
  Environment: demo
```

**What it checks:**
- Valid API credentials
- Active trading session
- Account not locked/suspended

**Failure mode:**
- Expired credentials
- Wrong password/token
- Account suspended

**Action on fail:** Block trading

---

### 5. Account Balance
```
[5/10] Account balance... ✓ PASS ($1,250.00 USD)
```

**What it checks:**
- Current balance > minimum ($100 default)
- Recommended: $1000+ for small lots

**Failure mode:**
- Insufficient funds
- Wrong account

**Action on fail:** Block trading

---

### 6. Margin Available
```
[6/10] Margin available... ✓ PASS ($1,250.00)
  Used: $0.00
```

**What it checks:**
- Free margin > minimum ($50 default)
- Not over-leveraged

**Failure mode:**
- Existing positions using margin
- Account near stop-out

**Action on fail:** Block trading

---

### 7. Symbol Resolution
```
[7/10] Symbol resolution... ✓ PASS
  XAUUSD -> symbolId: 41
  Digits: 2
```

**What it checks:**
- Symbol exists on broker
- Symbol ID resolved
- Digits/precision known

**Failure mode:**
- Symbol not available
- Wrong symbol name
- Trading suspended

**Action on fail:** Block trading

---

### 8. Market Hours
```
[8/10] Market hours... ✓ PASS
  Market open (2026-03-04 12:00 UTC)
```

**What it checks:**
- Market currently open
- Not weekend (Fri 20:00 - Sun 22:00 UTC)
- Not maintenance (00:00-00:01 UTC)

**XAUUSD Hours:**
- Open: Sunday 22:00 UTC
- Close: Friday 20:00 UTC

**Failure mode:**
- Weekend trading attempted
- Holiday closure

**Action on fail:** Warning (allows trading for 24h markets)

---

### 9. Hot Standby
```
[9/10] Hot standby... ✓ PASS
  Status: UP
  Failover count: 0
```

**What it checks:** (if `CTRADER_HOT_STANDBY=1`)
- Standby connection ready
- Failover history

**Failure mode:**
- Standby connection failed
- Single point of failure

**Action on fail:** Warning (degraded mode)

---

### 10. Connection Health
```
[10/10] Connection health... ✓ PASS
  Status: HEALTHY
  Latency: 23.1ms (p95: 25.3ms)
  Packet loss: 0.0%
```

**What it checks:**
- Health service validation
- Recent latency samples
- Packet loss rate

**Status levels:**
- **HEALTHY** - All metrics normal
- **DEGRADED** - Elevated latency/jitter
- **UNHEALTHY** - Packet loss or failures
- **CRITICAL** - Connection down

**Action on fail:** Warning if DEGRADED, Block if UNHEALTHY/CRITICAL

---

### 11. Test Order (Optional)
```
[11/11] Test order (0.01 lot)...
    Submitting test BUY order... ✓ FILLED
    Closing test position... ✓ CLOSED
    Test P&L: $-0.50
    ✓ PASS
```

**What it checks:**
- Full order lifecycle: Open → Fill → Close
- Market execution with real spread
- Position tracking
- Fill confirmation

**Cost:** ~$0.50 (spread + commission)

**When to use:**
- First time trading new symbol
- After broker changes
- After system updates

**Action on fail:** Block trading

---

## Configuration

### Environment Variables

```bash
# DNS
export KINETRA_DNS_USE_PUBLIC_RESOLVERS=1
export KINETRA_DNS_RESOLVERS=1.1.1.1,8.8.8.8,9.9.9.9

# Thresholds
export KINETRA_PREFLIGHT_MIN_BALANCE=100.0
export KINETRA_PREFLIGHT_MAX_LATENCY_MS=500.0
export KINETRA_PREFLIGHT_MAX_DNS_LATENCY_MS=1000.0

# Hot standby
export CTRADER_HOT_STANDBY=1
export CTRADER_HOT_STANDBY_START_RETRIES=2
```

### Code Configuration

```python
from kinetra.preflight_enhanced import PreflightConfig

config = PreflightConfig(
    symbol="XAUUSD",
    min_balance_usd=100.0,
    min_margin_available_usd=50.0,
    max_latency_ms=500.0,
    require_hot_standby=False,
    check_market_hours=True,
    enable_health_service=True,
)
```

---

## Failure Handling

### Blocking vs Warning

| Check | Fail Action | Recovery |
|-------|-------------|----------|
| DNS Resolution | Block | Fix network/DNS |
| TCP Connection | Block | Check firewall |
| Heartbeat | Block if >1000ms | Check latency |
| Broker Session | Block | Renew credentials |
| Balance | Block | Deposit funds |
| Margin | Block | Close positions |
| Symbol | Block | Verify symbol |
| Market Hours | Warn | Wait for open |
| Hot Standby | Warn | Fix standby |
| Health | Warn/Block | Check network |
| Test Order | Block | Check broker |

### Preflight Output Examples

**All Pass:**
```
============================================================
PREFLIGHT CHECK RESULTS
Timestamp: 2026-03-04 12:00:00 UTC
Duration: 1250.5ms
Passed: 11/11
============================================================

✅ PASSED:
  ✓ dns_resolution: demo.ctraderapi.com -> 1.2.3.4 (45.2ms)
  ✓ tcp_connection: Connected to demo.ctraderapi.com:5035
  ...

🟢 ALL CHECKS PASSED - Safe to trade
============================================================
```

**With Failures:**
```
============================================================
PREFLIGHT CHECK RESULTS
Timestamp: 2026-03-04 12:00:00 UTC
Duration: 2500.3ms
Passed: 8/11
============================================================

❌ BLOCKING ISSUES:
  ✗ account_balance: Balance: $50.00 (min: $100.00)
  ✗ broker_session: Session expired

⚠️  WARNINGS:
  ! market_hours: Market closed (weekend)

🔴 PREFLIGHT FAILED - Trading blocked
Blocking reasons:
  - account_balance: Balance: $50.00 (min: $100.00)
  - broker_session: Session expired
============================================================
```

---

## Standalone Usage

### Programmatic

```python
from kinetra.preflight_enhanced import EnhancedPreflight, PreflightConfig
from kinetra.connectors.ctrader_connector import build_connector

# Create connector
connector = build_connector()

# Configure preflight
config = PreflightConfig(
    symbol="XAUUSD",
    min_balance_usd=100.0,
)

# Run checks
preflight = EnhancedPreflight(connector, config)
result = preflight.run_all_checks()

# Check result
if result.can_trade:
    print("Safe to trade")
else:
    print(f"Blocked: {result.blocking_reasons}")
```

### CLI

```bash
# Using Python directly
python -c "
from kinetra.preflight_enhanced import run_preflight_cli
from kinetra.connectors.ctrader_connector import build_connector
connector = build_connector()
run_preflight_cli(connector, 'XAUUSD')
"
```

---

## Troubleshooting

### "DNS resolution failed"
```bash
# Check DNS
nslookup demo.ctraderapi.com

# Use public resolvers
export KINETRA_DNS_USE_PUBLIC_RESOLVERS=1
```

### "TCP connection failed"
```bash
# Test connectivity
nc -zv demo.ctraderapi.com 5035

# Check firewall
sudo iptables -L | grep 5035
```

### "Heartbeat latency too high"
```bash
# Test latency
ping demo.ctraderapi.com

# Consider VPS closer to broker
```

### "Insufficient balance"
- Deposit funds
- Reduce `--preflight-lots` if using test order

### "Market closed"
- XAUUSD: Sun 22:00 - Fri 20:00 UTC
- Check broker holiday schedule

---

## Best Practices

1. **Always run preflight** before live trading
2. **Use test order** on first run or after changes
3. **Monitor warnings** - they indicate degraded conditions
4. **Fix blocking issues** immediately
5. **Log all preflight results** for audit trail

---

## See Also

- [QUICK_START.md](QUICK_START.md) - Quick start guide
- [LIVE_TRADING_CONFIG.md](LIVE_TRADING_CONFIG.md) - Configuration
- [PIPELINE.md](PIPELINE.md) - Trading pipeline
- `kinetra/preflight_enhanced.py` - Implementation
