# Kinetra Trading Pipeline

**Chronological flow from data to live trading**

This document describes the complete pipeline from raw data ingestion through to live order execution.

## Pipeline Overview

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Raw Data  │───▶│  DSP/Brick  │───▶│  Backtest   │───▶│   Paper     │───▶│    LIVE     │
│   Download  │    │    Size     │    │ Validation  │    │  Trading    │    │  Trading    │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
      │                   │                   │                   │                   │
   [M1 CSV]          [Profile]          [Metrics]          [Signals]          [Orders]
   [cTrader]          [bricks/day]       [Omega>1.5]        [Live bars]        [Real $]
```

---

## Stage 1: Data Acquisition

**Purpose:** Download and cache historical M1 data for analysis

### Input
- Symbol (e.g., XAUUSD)
- Date range
- Source (cTrader, MT5, etc.)

### Process
```bash
python scripts/renko_engine.py XAUUSD --stage download
```

### Output
- `data/XAUUSD/m1/` - M1 CSV files
- Format: `time,open,high,low,close,volume,spread`

### Validation
- Data completeness check
- Gap detection
- Quality scoring

---

## Stage 2: DSP Analysis (Digital Signal Processing)

**Purpose:** Calculate optimal brick size based on signal-to-noise ratio

### Input
- M1 historical data
- Target: Brick size > friction costs

### Process
```bash
python scripts/renko_engine.py XAUUSD --stage dsp
```

### Algorithm
1. Calculate Renko bricks for range of brick sizes
2. Compute bricks per day (velocity)
3. Estimate signal-to-noise ratio
4. Select size where S/N > threshold

### Output
- `data/XAUUSD/dsp_profile.json`
```json
{
  "symbol": "XAUUSD",
  "optimal_brick_size": 1.0,
  "bricks_per_day": 12.5,
  "signal_to_noise": 2.3,
  "friction_costs": 0.15
}
```

### Success Criteria
- Brick size generates ≥ 5 bricks/day
- Signal-to-noise ratio > 2.0
- Brick size > 3× spread

---

## Stage 3: Backtest Validation

**Purpose:** Validate strategy performance on historical data

### Input
- M1 data
- DSP profile (brick size)
- Strategy parameters

### Process
```bash
# Quick validation (3 months)
python scripts/renko_engine.py XAUUSD --stage backtest --months 3

# Full validation (24 months)
python scripts/renko_engine.py XAUUSD --stage backtest --months 24
```

### Engine Flow
1. Build Renko bricks from M1 closes
2. Calculate filters (flip rate, Markov stickiness)
3. Evaluate entry/exit signals
4. Apply sizing and risk rules
5. Calculate performance metrics

### Output
- Trade list with entry/exit details
- Performance summary:
  - Net P&L
  - Win rate
  - Omega ratio
  - Sharpe ratio
  - Max drawdown

### Success Criteria
| Metric | Threshold | Rationale |
|--------|-----------|-----------|
| Omega | ≥ 1.5 | Profit/loss ratio |
| Trades | ≥ 30 | Statistical significance |
| Win Rate | 35-65% | Realistic range |
| Max DD | < 10% | Risk tolerance |

### Failure Handling
If backtest fails criteria:
- Adjust strategy parameters
- Try different brick size
- Reject symbol for trading

---

## Stage 4: Paper Trading

**Purpose:** Validate live signal generation without real orders

### Input
- Live M1 bar feed from broker
- Same strategy as backtest

### Process
```bash
python scripts/renko_engine.py XAUUSD --stage paper
```

### Architecture
```
cTrader SpotEvent ──▶ CTraderBarProvider ──▶ TradingEngine
     (M1 bars)           (decode)             (signals only)
```

### Key Differences from Backtest
- Real-time data feed latency
- Live spread variation
- Broker session handling
- No actual order execution

### Monitoring
- Signal frequency matches backtest
- Entry/exit timing
- Filter values in real-time

### Success Criteria
- Signals generated at expected frequency
- Filter values stable
- No exceptions or errors

---

## Stage 5: Dry-Run (Live Data, Paper Orders)

**Purpose:** Test full execution path without risking capital

### Input
- Live M1 bar feed
- Paper order dispatcher (simulated fills)

### Process
```bash
python scripts/renko_engine.py XAUUSD --stage live --dry-run --live-size micro
```

### Architecture
```
Bar Feed ──▶ Engine ──▶ PaperDispatcher ──▶ Simulated Fills
                           (latency simulation)
```

### Key Features
- Same order flow as live
- Simulated fill latency
- Position tracking
- State persistence testing

### Validation
- Orders generated correctly
- Fill tracking works
- State persistence
- Dashboard displays correctly

---

## Stage 6: Enhanced Preflight

**Purpose:** Comprehensive safety checks before real trading

### Process
```bash
python scripts/renko_engine.py XAUUSD --stage live --live-size micro \
  --preflight-test-order --preflight-lots 0.01 \
  --ack-live I_UNDERSTAND_LIVE_RISK
```

### Checks

| # | Check | Action on Fail |
|---|-------|----------------|
| 1 | DNS Resolution | Block trading |
| 2 | TCP Connection | Block trading |
| 3 | Heartbeat Latency | Block if > 1000ms |
| 4 | Broker Session | Block trading |
| 5 | Account Balance | Block if < $100 |
| 6 | Margin Available | Block if insufficient |
| 7 | Symbol Resolution | Block trading |
| 8 | Market Hours | Warn if closed |
| 9 | Hot Standby | Block if required |
| 10 | Health Service | Warn if degraded |
| 11 | Test Order (optional) | Block if fails |

### Output
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
  ✓ heartbeat: Heartbeat latency: 23.1ms, connected: True
  ...

🟢 ALL CHECKS PASSED - Safe to trade
============================================================
```

---

## Stage 7: LIVE Trading

**Purpose:** Execute real orders with full safety systems

### Launch
```bash
python scripts/renko_engine.py XAUUSD --stage live --live-size micro
```

### Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         LIVE TRADING SYSTEM                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐         │
│  │ cTrader API  │────▶│ Bar Provider │────▶│   Engine     │         │
│  │  (M1 bars)   │     │  (decode)    │     │ (strategies) │         │
│  └──────────────┘     └──────────────┘     └──────┬───────┘         │
│                                                    │                  │
│                           ┌────────────────────────┘                  │
│                           │                                          │
│                           ▼                                          │
│                    ┌──────────────┐                                 │
│                    │ _bar_queue   │                                 │
│                    │   (1024)     │                                 │
│                    └──────┬───────┘                                 │
│                           │                                          │
│          ┌────────────────┼────────────────┐                        │
│          │                │                │                        │
│          ▼                ▼                ▼                        │
│   ┌──────────────┐ ┌──────────────┐ ┌──────────────┐               │
│   │ Bar Processor│ │Order Executor│ │Health Monitor│               │
│   │ (time-critical│ │ (blocking OK)│ │ (periodic)   │               │
│   └──────┬───────┘ └──────┬───────┘ └──────────────┘               │
│          │                │                                        │
│          │                ▼                                        │
│          │         ┌──────────────┐                                │
│          │         │   Broker     │                                │
│          │         │ (real orders)│                                │
│          │         └──────┬───────┘                                │
│          │                │                                        │
│          │         ┌──────┴───────┐                                │
│          │         │   _fill_queue │                                │
│          │         │    (256)      │                                │
│          │         └──────┬───────┘                                │
│          │                │                                        │
│          └────────────────┘                                        │
│                           │                                         │
│                           ▼                                         │
│                    ┌──────────────┐                                │
│                    │ State Update │                                │
│                    │ Persistence  │                                │
│                    └──────────────┘                                │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### Safety Systems

#### 1. Thread Isolation
- **Bar Processor**: Processes bars, NEVER blocks
- **Order Executor**: Handles blocking broker calls
- **Fill Queue**: Async communication between threads

#### 2. Health Monitoring
- Periodic heartbeat checks (every 30s)
- Latency tracking
- Packet loss detection
- Thread liveness monitoring

#### 3. Circuit Breaker
- Triggers after N consecutive fill failures
- Default: 5 failures
- Configurable: `CTRADER_MAX_FILL_FAILURES`
- Manual reset required

#### 4. State Persistence
- Auto-save on fill confirmation
- Auto-save on trade completion
- Atomic JSON writes
- Recovery on restart

#### 5. Live Dashboard
```
┌─────────────────────────────────────────────────┐
│ XAUUSD LIVE SNAPSHOT  2026-03-04 12:00:00 UTC  │
├─────────────────────────────────────────────────┤
│ [dim]Trade Performance[/]                       │
│ Mode            LIVE                            │
│ Symbol          XAUUSD                          │
│ Renko Bricks    42                              │
│ Trades          3                               │
│ Net P&L         $45.50                          │
│                                                 │
│ [dim]Health & Execution[/]                      │
│ Health status   [green]HEALTHY[/]               │
│ Health latency  45.2ms                          │
│ Fill success    [green]100.0%[/]                │
│ Consec failures [green]0[/]                     │
│ Orders          3/3 OK, 0 fail                  │
│                                                 │
│ [dim]Connection[/]                              │
│ Status          [green]UP[/]                    │
│ Endpoint        demo.ctraderapi.com:5035        │
│ Failovers       0                               │
└─────────────────────────────────────────────────┘
```

---

## Error Handling & Recovery

### Fill Timeout
```
1. Order sent to broker
2. Wait for execution event (5s timeout)
3. If timeout:
   a. Reconcile position with broker
   b. If position found → treat as filled
   c. If not found → mark as failed
4. Update state
5. Persist to disk
```

### Connection Loss
```
1. Health monitor detects disconnect
2. Status changes to CRITICAL
3. Circuit breaker triggers (if enabled)
4. Log error with full context
5. Attempt reconnect (if hot standby)
```

### Thread Crash
```
1. Monitor thread detects stopped thread
2. Log CRITICAL alert: "[HEALTH] X thread has stopped!"
3. Stop trading immediately
4. Preserve state
5. Manual restart required
```

---

## Pipeline Validation Gates

| Gate | Requirement | Failure Action |
|------|-------------|----------------|
| DSP | S/N > 2.0 | Reject symbol |
| Backtest | Omega ≥ 1.5, Trades ≥ 30 | Adjust parameters |
| Paper | Signals match expected | Debug strategy |
| Dry-Run | Orders flow correctly | Fix dispatcher |
| Preflight | All checks pass | Resolve issues |
| Live | Health = HEALTHY | Pause trading |

---

## Environment Configuration

### Minimum Viable Config
```bash
# .env.openapi
CTRADER_CLIENT_ID=your_client_id
CTRADER_CLIENT_SECRET=your_secret
CTRADER_ACCESS_TOKEN=your_token
CTRADER_ACCOUNT_ID=12345
CTRADER_ENVIRONMENT=demo

# Optional tuning
CTRADER_FILL_WAIT_S=5.0
CTRADER_MAX_FILL_FAILURES=5
```

### Production Config
```bash
# Enhanced reliability
CTRADER_HOT_STANDBY=1
KINETRA_DNS_USE_PUBLIC_RESOLVERS=1
KINETRA_PERSIST_DIR=/var/kinetra/state
```

---

## Monitoring & Observability

### Log Files
- `outputs/logs/renko_engine_YYYYMMDD.log`
- Look for: `[FILL-CONFIRMED]`, `[EXIT]`, `[HEALTH]`, `[CRITICAL]`

### Metrics (Prometheus/Grafana)
- `kinetra_orders_submitted_total`
- `kinetra_orders_filled_total`
- `kinetra_orders_failed_total`
- `kinetra_fill_latency_ms`
- `kinetra_health_status`
- `kinetra_circuit_breaker_triggered`

### Alerts
- Circuit breaker triggered
- Health status CRITICAL
- Thread crash detected
- Fill success rate < 80%
- Consecutive failures ≥ 3

---

## See Also

- [QUICK_START.md](QUICK_START.md) - Quick reference
- [LIVE_TRADING_CONFIG.md](LIVE_TRADING_CONFIG.md) - Configuration details
- [PREFLIGHT_CHECKS.md](PREFLIGHT_CHECKS.md) - Preflight documentation
- `kinetra/renko/trading_engine.py` - Core engine
- `kinetra/renko/ctrader_dispatcher.py` - Broker integration
