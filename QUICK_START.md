# Kinetra Quick Start (Current)

Last updated: 2026-03-04

## Overview

Kinetra is a production-ready Renko brick trading system with:
- **Async order execution** - Two-thread architecture prevents bar drops during order latency
- **Health monitoring** - Real-time connection health and fill failure tracking
- **Circuit breakers** - Automatic trading halt on consecutive fill failures
- **Enhanced preflight** - DNS, TCP, heartbeat, and margin validation before trading
- **Atomic persistence** - Crash recovery with state and trade persistence

## 1) Launch Interactive Menu

```bash
make launch
```

## 2) Direct CLI Commands

### Backtest (Historical Validation)
```bash
# Quick 3-month backtest
python scripts/renko_engine.py XAUUSD --stage backtest --months 3

# Full 24-month backtest
python scripts/renko_engine.py XAUUSD --stage backtest --months 24
```

### Paper Trading (Live Data, Simulated Fills)
```bash
python scripts/renko_engine.py XAUUSD --stage paper
```

### Live Dry-Run (Live Data, Paper Orders)
```bash
python scripts/renko_engine.py XAUUSD --stage live --dry-run --live-size micro
```

### LIVE Trading (REAL ORDERS - USE WITH CAUTION)
```bash
# Micro lots (0.01) - recommended for testing
python scripts/renko_engine.py XAUUSD --stage live --live-size micro

# With execution preflight (recommended)
python scripts/renko_engine.py XAUUSD --stage live --live-size micro \
  --preflight-test-order --preflight-lots 0.01 \
  --ack-live I_UNDERSTAND_LIVE_RISK
```

## 3) Enhanced Preflight Checks

The enhanced preflight validates 10 critical items:

1. **DNS Resolution** - Broker endpoints resolve correctly
2. **TCP Connection Pool** - Active connection verified
3. **Heartbeat/Keep-Alive** - Latency < 500ms
4. **Broker Session** - Valid credentials and session
5. **Account Balance** - ≥ $100 minimum
6. **Margin Available** - Sufficient free margin
7. **Symbol Resolution** - Symbol tradable
8. **Market Hours** - Market open (XAUUSD: Sun 22:00 - Fri 20:00 UTC)
9. **Hot Standby** - Failover ready (if enabled)
10. **Connection Health** - Health service validation

```bash
# Run standalone preflight
python -c "
from kinetra.preflight_enhanced import run_preflight_cli
from kinetra.connectors.ctrader_connector import build_connector
connector = build_connector()
run_preflight_cli(connector, 'XAUUSD')
"
```

## 4) Health Monitoring Dashboard

The live dashboard shows real-time health metrics:

```
[dim]Health & Execution[/]
Health status   [green]HEALTHY[/]
Health latency  45.2ms
Fill success    [green]100.0%[/]
Consec failures [green]0[/]
Orders          5/5 OK, 0 fail
```

### Health Status Levels
- **HEALTHY** (green) - All metrics normal
- **DEGRADED** (yellow) - Elevated latency or minor issues
- **UNHEALTHY** (red) - High packet loss or multiple failures
- **CRITICAL** (red) - Connection down or thread crash

## 5) Circuit Breaker

Automatic trading halt on fill failures:

```bash
# Default: halt after 5 consecutive fill failures
export CTRADER_MAX_FILL_FAILURES=5

# Stricter: halt after 3 failures
export CTRADER_MAX_FILL_FAILURES=3

# Manual reset (after resolving issue):
# In Python console or new process
from kinetra.renko.ctrader_dispatcher import CTraderOrderDispatcher
dispatcher.reset_circuit_breaker()
```

## 6) Strategy Overrides

```bash
python scripts/renko_engine.py XAUUSD --stage live --live-size micro \
  --brick-size 1.0 \
  --stop-bricks 0.5 \
  --target-risk 100 \
  --fliprate-window 431 \
  --markov-window 431 \
  --fliprate-threshold 0.35 \
  --markov-threshold 0.55
```

### Key Parameters
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--brick-size` | 1.0 | Renko brick size in price units |
| `--stop-bricks` | 0.5 | Stop distance in bricks |
| `--target-risk` | 100 | Risk USD per trade |
| `--fliprate-window` | 431 | Flip rate lookback window |
| `--markov-window` | 431 | Markov stickiness window |
| `--fliprate-threshold` | 0.35 | Max flip rate for entry |
| `--markov-threshold` | 0.55 | Min stickiness for entry |

## 7) State Persistence

Engine state is automatically persisted to `~/.kinetra/state/<symbol>/`:

```
~/.kinetra/state/XAUUSD/
├── engine_state.json   # Current position state
└── trades.jsonl        # Completed trades log
```

Persistence triggers:
- After fill confirmation
- After trade completion (exit)
- On graceful shutdown

## 8) Environment Variables

```bash
# Connection
export CTRADER_HOT_STANDBY=1                    # Enable hot standby
export CTRADER_FILL_WAIT_S=5.0                  # Fill timeout (reduced from 30s)
export CTRADER_MAX_FILL_FAILURES=5              # Circuit breaker threshold

# Health Monitoring
export KINETRA_LIVE_HEALTH_PROBE_INTERVAL_S=15  # Health check interval
export KINETRA_LIVE_STALE_BRICK_PROBE_S=90      # Stale brick detection

# Preflight
export KINETRA_DNS_USE_PUBLIC_RESOLVERS=1       # Use public DNS
export KINETRA_DNS_RESOLVERS=1.1.1.1,8.8.8.8   # Custom DNS servers

# Persistence
export KINETRA_PERSIST_DIR=/custom/path         # Custom state directory
```

## 9) Troubleshooting

### "Circuit breaker triggered"
```bash
# Check fill metrics
python -c "
from kinetra.renko.ctrader_dispatcher import CTraderOrderDispatcher
print(dispatcher.get_fill_metrics())
"

# Reset after fixing issue
dispatcher.reset_circuit_breaker()
```

### "Health status CRITICAL"
- Check network connectivity: `ping demo.ctraderapi.com`
- Verify credentials in `.env.openapi`
- Check thread health in logs for `[HEALTH]` messages

### "Fill queue full"
- This is a critical error - indicates order executor thread blocked
- Check for `[ORDER-EXECUTOR]` error messages
- Restart trading process

### "Preflight failed"
- Each check is logged with `[PASS]` or `[FAIL]`
- Blocking issues prevent trading
- Warnings allow trading with caution

### Thread Crash Detection
The system monitors thread health:
```
[HEALTH] Bar processor thread has stopped!
[HEALTH] Order executor thread has stopped!
```

If seen:
1. Stop trading immediately
2. Check logs for crash details
3. Restart process

## 10) Safety Checklist

Before going live:

- [ ] Backtest shows Omega ≥ 1.5
- [ ] Backtest shows ≥ 30 trades
- [ ] Paper trading validates signals
- [ ] Dry-run confirms connectivity
- [ ] Preflight all checks pass
- [ ] Health status is HEALTHY
- [ ] Fill success rate > 95%
- [ ] Circuit breaker not triggered
- [ ] Lot size appropriate for equity
- [ ] Stop loss configured
- [ ] Persistence directory writable
- [ ] You understand the risks

**⚠️ NEVER trade more than you can afford to lose.**

## 11) Pipeline Flow

```
1. DSP Analysis (one-time)
   └── Calculate optimal brick size

2. Backtest Validation
   └── Historical performance verification

3. Paper Trading
   └── Live signals, simulated execution

4. Dry-Run
   └── Live data, paper orders

5. LIVE Trading
   └── Real orders with full safety
```

See [PIPELINE.md](PIPELINE.md) for detailed pipeline documentation.

## 12) Key Files

| File | Purpose |
|------|---------|
| `kinetra/renko/trading_engine.py` | Core trading engine |
| `kinetra/renko/ctrader_dispatcher.py` | cTrader integration |
| `kinetra/monitoring/connection_health.py` | Health monitoring |
| `kinetra/preflight_enhanced.py` | Enhanced preflight checks |
| `scripts/renko_engine.py` | Main CLI entry point |

## See Also

- [LIVE_TRADING_CONFIG.md](LIVE_TRADING_CONFIG.md) - Detailed configuration
- [PREFLIGHT_CHECKS.md](PREFLIGHT_CHECKS.md) - Preflight check details
- [PIPELINE.md](PIPELINE.md) - Trading pipeline documentation
- `archive/production_cleanup_2026-03-03/repo/docs/RENKO_LIVE_STATE.md` - Live state semantics
