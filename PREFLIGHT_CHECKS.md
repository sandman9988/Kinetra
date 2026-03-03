# Preflight Checks — Live Trading Safety

> Canonical runtime status and current behavior are tracked in
> [`archive/production_cleanup_2026-03-03/repo/docs/RENKO_LIVE_STATE.md`](archive/production_cleanup_2026-03-03/repo/docs/RENKO_LIVE_STATE.md).

## Overview

Before executing any **REAL ORDERS**, the live trading stage runs comprehensive preflight checks to verify:

1. ✅ **DNS Resolution** — Broker endpoints resolve correctly
2. ✅ **Network Latency** — Connection speed is acceptable (<500ms ideal)
3. ✅ **Authentication** — Valid credentials and active session
4. ✅ **Account Balance** — Sufficient funds for minimum lot size
5. ✅ **Symbol Tradability** — Instrument is available for trading
6. ✅ **Optional Test Order** — Full open + close cycle with a tiny lot size when explicitly enabled

## Check Details

### 1. DNS Resolution
```
[1/6] DNS resolution... ✓ PASS
```
**What it checks:**
- Broker endpoint hostname resolves to valid IP
- No DNS hijacking or poisoning detected

**Failure mode:**
- Network misconfiguration
- DNS server issues
- Broker endpoint changed

### 2. Network Latency
```
[2/6] Network latency... ✓ PASS (45ms)
```
**What it checks:**
- Round-trip time to broker server
- Acceptable: <500ms (green), 500-1000ms (yellow warning), >1000ms (red fail)

**Failure mode:**
- Network congestion
- Geographic distance from broker servers
- Firewall/proxy interference

### 3. Authentication
```
[3/6] Authentication... ✓ PASS (account: 45841299)
```
**What it checks:**
- Valid API credentials
- Active trading session
- Account not locked or suspended

**Failure mode:**
- Expired credentials
- Wrong password/token
- Account suspended by broker

### 4. Account Balance
```
[4/6] Account balance... ✓ PASS ($1,250.00 USD)
```
**What it checks:**
- Current account balance
- Minimum $100 for micro lots (0.01)
- Minimum $1,000 recommended for small lots (0.10)

**Failure mode:**
- Insufficient funds
- Margin already in use
- Account currency mismatch

### 5. Symbol Tradability
```
[5/6] Symbol 'XAUUSD' tradable... ✓ PASS (symbolId: 41)
```
**What it checks:**
- Symbol exists on broker platform
- Symbol is currently tradable (not suspended)
- Symbol ID resolved for order submission

**Failure mode:**
- Symbol delisted
- Trading halted by broker
- Wrong symbol name

### 6. Optional Test Order (Open + Close)
```
[6/6] Test order (0.01 lot)...
    Submitting test BUY order... ✓ FILLED
    Closing test position... ✓ CLOSED
    Test P&L: $-0.50
    ✓ PASS
```
**What it checks (when enabled):**
- **Full order lifecycle**: Open → Fill → Close
- **Market execution**: Real bid/ask spread
- **Stop-loss placement**: Verified with broker
- **Position tracking**: Position ID returned and used
- **Minimum lot size**: 0.01 lots (micro)

**Failure modes:**
- Order rejected (insufficient margin, invalid parameters)
- Fill timeout (network issue, market closed)
- Close failed (position already closed, broker error)

**⚠️ Important:** This submits a **REAL ORDER** and immediately closes it.
It is only run when you pass:

```bash
--preflight-test-order --preflight-lots 0.01 --ack-live I_UNDERSTAND_LIVE_RISK
```

The small loss (spread + commission) is the cost of verification.

---

## Preflight Output Example

```
Running preflight checks...
  [PASS] connection       connector authenticated
  [PASS] account_snapshot balance=$48,589.59
  [PASS] symbol_resolution XAUUSD -> 41
  [PASS] symbol_digits    digits=2
✅ Preflight passed
```

---

## Failure Handling

If any check fails:

1. **Trading is aborted** — No real orders will be placed
2. **Error message displayed** — Specific failure reason shown
3. **Connection closed** — Clean disconnect from broker
4. **Manual intervention required** — Fix issue before retrying

### Example Failure

```
Running preflight checks...
  [1/6] DNS resolution... ✓ PASS
  [2/6] Network latency... ✓ PASS (48ms)
  [3/6] Authentication... ✓ PASS (account: 45841299)
  [4/6] Account balance... ✗ FAIL ($50.00 - insufficient)
    Minimum balance $100 required for micro lots
  [5/6] Symbol 'XAUUSD' tradable... (skipped)
  [6/6] Test order (0.01 lot)... (skipped)

  Preflight: 3/6 checks passed
❌ Preflight checks failed - aborting live trading
```

---

## Skipping Preflight

**Dry-run mode** (`--dry-run`) does not run live execution preflight because:
- No real orders are submitted
- Paper dispatcher doesn't require broker connection
- Safe for testing configuration

```bash
python scripts/renko_engine.py XAUUSD --stage live --dry-run
# No preflight checks (paper trading)
```

---

## Best Practices

1. **Run preflight before every live session** — Network conditions change
2. **Don't skip failed checks** — They protect your capital
3. **Monitor test order P&L** — Should be ~spread + commission
4. **Keep minimum balance** — At least $100 for micro lots
5. **Verify symbol ID** — Ensure trading correct instrument

---

## Troubleshooting

### "DNS resolution failed"
- Check internet connection
- Try `ping demo.ctraderapi.com`
- Verify DNS settings

### "Network latency too high"
- Check network congestion
- Consider VPS closer to broker servers
- Close bandwidth-heavy applications

### "Authentication failed"
- Verify `.env.openapi` credentials
- Check account status with broker
- Renew API token if expired

### "Insufficient balance"
- Deposit funds
- Reduce lot size (use `--live-size micro`)
- Close other positions to free margin

### "Symbol not found"
- Verify symbol name (case-sensitive)
- Check if symbol is tradable on your account type
- Contact broker if symbol should be available

### "Test order failed"
- Check market hours (forex closed weekends)
- Verify stop-loss distance (not too tight)
- Ensure sufficient margin for 0.01 lots

---

## See Also

- `LIVE_TRADING_CONFIG.md` — Full live trading configuration reference
- `SWAP_MODELING_TODO.md` — Swap cost modeling limitations
- `scripts/ctrader/launch.sh` — Interactive launcher with preflight
