# Live Trading Configuration — Kinetra Renko Engine

Last updated: 2026-03-04

## Quick Start

```bash
# Dry run (paper trading)
python scripts/renko_engine.py XAUUSD --stage live --live-size micro --dry-run

# Live micro lots with preflight
python scripts/renko_engine.py XAUUSD --stage live --live-size micro \
  --preflight-test-order --preflight-lots 0.01 \
  --ack-live I_UNDERSTAND_LIVE_RISK

# Live trading (full)
python scripts/renko_engine.py XAUUSD --stage live --live-size full \
  --ack-live I_UNDERSTAND_LIVE_RISK
```

## Configuration Methods

Priority (highest to lowest):
1. CLI arguments
2. Environment variables (`.env`)
3. Default values in code

## Essential Environment Variables

Create `.env` in project root:

```bash
# Required: cTrader OpenAPI credentials
CTRADER_CLIENT_ID=your_client_id
CTRADER_CLIENT_SECRET=your_client_secret
CTRADER_ACCESS_TOKEN=your_access_token
CTRADER_ACCOUNT_ID=your_account_id

# Required: Environment (demo/live)
CTRADER_ENVIRONMENT=demo

# Optional: Connection settings
CTRADER_ENDPOINT=demo.ctraderapi.com
CTRADER_PORT=5035

# Optional: Timeouts
CTRADER_FILL_WAIT_S=5.0           # Fill timeout (default: 5.0)
CTRADER_CLOSE_WAIT_S=5.0          # Close timeout (default: 5.0)

# Optional: Safety
CTRADER_MAX_FILL_FAILURES=5       # Circuit breaker threshold (default: 5)
CTRADER_HOT_STANDBY=1             # Enable hot standby (default: 0)
```

## CLI Arguments

### Stage Selection
```bash
--stage {download,dsp,backtest,paper,live}
    Trading stage to run
    
--live-size {micro,full}
    Lot size for live trading (default: micro)
    
--dry-run
    Simulate live trading without real orders
```

### Preflight Options
```bash
--preflight-test-order
    Execute test order before trading (validates full pipeline)
    
--preflight-lots 0.01
    Lot size for test order (default: 0.01)
    
--ack-live I_UNDERSTAND_LIVE_RISK
    Required acknowledgment for live trading
```

### Strategy Parameters
```bash
--brick-size 1.0
    Renko brick size in price points
    
--profile {aggressive,balanced,conservative}
    Filter profile (default: balanced)
    
--target-trades-per-day 15.0
    Target trade frequency
    
--months 24
    Backtest period (months)
```

## Connection Settings

### Basic Connection
```bash
CTRADER_ENDPOINT=demo.ctraderapi.com
CTRADER_PORT=5035
CTRADER_TLS=1
```

### Hot Standby (High Availability)
```bash
CTRADER_HOT_STANDBY=1
CTRADER_HOT_STANDBY_START_RETRIES=2
```

Enables automatic failover on connection failure.

### DNS Configuration
```bash
KINETRA_DNS_USE_PUBLIC_RESOLVERS=1
KINETRA_DNS_RESOLVERS=1.1.1.1,8.8.8.8,9.9.9.9
```

For production trading, use reliable DNS resolvers.

## Timeout Configuration

| Timeout | Variable | Default | Description |
|---------|----------|---------|-------------|
| Fill Wait | `CTRADER_FILL_WAIT_S` | 5.0 | Max time to wait for fill confirmation |
| Close Wait | `CTRADER_CLOSE_WAIT_S` | 5.0 | Max time to wait for close confirmation |
| Heartbeat | `KINETRA_PREFLIGHT_MAX_LATENCY_MS` | 500 | Max acceptable latency |

Adjust based on your network conditions:
- Low latency (<50ms): Use defaults
- High latency (>200ms): Increase to 10.0s

## Safety Configuration

### Circuit Breaker
```bash
CTRADER_MAX_FILL_FAILURES=5
```

Trading halts after N consecutive fill failures. Manual reset required via menu or restart.

### Preflight Thresholds
```bash
# Minimum balance to allow trading
KINETRA_PREFLIGHT_MIN_BALANCE=100.0

# Minimum free margin
KINETRA_PREFLIGHT_MIN_MARGIN=50.0

# Max acceptable latency
KINETRA_PREFLIGHT_MAX_LATENCY_MS=500.0
```

### Position Limits
```python
# In kinetra/renko/strategies.py or config
max_position_size_usd = 1000.0    # Max $ exposure per trade
max_open_positions = 1            # Only 1 position at a time
```

## Lot Sizing

### Micro Lots (0.01)
```bash
python scripts/renko_engine.py XAUUSD --stage live --live-size micro
```
- XAUUSD: ~$0.10 per point
- Good for testing with <$1000

### Full Lots
```bash
python scripts/renko_engine.py XAUUSD --stage live --live-size full
```
- XAUUSD: ~$10.00 per point
- Requires $10,000+ balance recommended

## Account Types

### Demo Account
```bash
CTRADER_ENVIRONMENT=demo
CTRADER_ENDPOINT=demo.ctraderapi.com
```
- No real money risk
- Same execution as live
- Use for testing and development

### Live Account
```bash
CTRADER_ENVIRONMENT=live
CTRADER_ENDPOINT=live.ctraderapi.com
```
- Real money at risk
- Requires live credentials
- Ensure preflight passes

## Monitoring Configuration

### Health Checks
```python
# In code
health_service = ConnectionHealthService(
    connector=connector,
    check_interval=30.0,        # Check every 30 seconds
    latency_threshold_ms=500,    # Degraded if > 500ms
    critical_threshold_ms=1000,  # Critical if > 1000ms
    enable_preemptive_failover=True
)
```

### Dashboard Settings
```python
# Auto-refresh interval
dashboard_refresh_interval = 5.0  # seconds

# Metrics window
metrics_window_days = 7
```

## State Persistence

### Automatic Saving
State auto-saved on:
- Fill confirmation
- Trade completion
- Graceful shutdown

### Manual Save
Press `s` in the live dashboard.

### Storage Location
```python
# Default
KINETRA_PERSIST_DIR=./.workflow_state

# Production
KINETRA_PERSIST_DIR=/var/kinetra/state
```

## Complete Example Config

### Development
```bash
# .env
CTRADER_ENVIRONMENT=demo
CTRADER_CLIENT_ID=your_demo_id
CTRADER_CLIENT_SECRET=your_demo_secret
CTRADER_ACCESS_TOKEN=your_demo_token
CTRADER_ACCOUNT_ID=12345

CTRADER_FILL_WAIT_S=5.0
CTRADER_MAX_FILL_FAILURES=5
```

### Production
```bash
# .env
CTRADER_ENVIRONMENT=live
CTRADER_CLIENT_ID=your_live_id
CTRADER_CLIENT_SECRET=your_live_secret
CTRADER_ACCESS_TOKEN=your_live_token
CTRADER_ACCOUNT_ID=67890

CTRADER_ENDPOINT=live.ctraderapi.com
CTRADER_HOT_STANDBY=1

CTRADER_FILL_WAIT_S=10.0
CTRADER_MAX_FILL_FAILURES=3

KINETRA_DNS_USE_PUBLIC_RESOLVERS=1
KINETRA_PERSIST_DIR=/var/kinetra/state
```

## Troubleshooting

### "Connection timeout"
- Check firewall settings
- Verify endpoint and port
- Increase `CTRADER_FILL_WAIT_S`

### "Session expired"
- Renew access token
- Check `CTRADER_ACCESS_TOKEN`
- Verify `CTRADER_ENVIRONMENT`

### "Insufficient balance"
- Deposit funds
- Use micro lots
- Check `KINETRA_PREFLIGHT_MIN_BALANCE`

### "Circuit breaker triggered"
- Check logs for fill failures
- Restart to reset (only if confident)
- Investigate broker issues

## See Also

- [QUICK_START.md](QUICK_START.md) - Quick reference
- [PREFLIGHT_CHECKS.md](PREFLIGHT_CHECKS.md) - Preflight documentation
- [PIPELINE.md](PIPELINE.md) - Trading pipeline
- [TRADING_ARCHITECTURE.md](TRADING_ARCHITECTURE.md) - System architecture
