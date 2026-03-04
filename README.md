# Kinetra
### *Institutional-Grade Renko Brick Trading Engine*

**Kinetra** is a production-ready Renko brick trading system for XAUUSD (Gold) via cTrader Open API. Built for reliability, safety, and deterministic execution.

## 🚀 Current Status

**Active Trading Pipeline:** Renko Brick System for XAUUSD

| Component | Status | Notes |
|-----------|--------|-------|
| Data Pipeline | ✅ Production | M1 download, DSP analysis |
| Backtesting | ✅ Production | 3mo/3yr validation |
| Paper Trading | ✅ Production | Live signal validation |
| Live Trading | ✅ Production | cTrader integration with full safety |
| Test Suite | ✅ Comprehensive | 15+ test files, 2500+ lines |
| Documentation | ✅ Current | All docs reflect current state |

## 📋 Quick Start

```bash
# 1. Download historical data
python scripts/renko_engine.py XAUUSD --stage download

# 2. Run DSP analysis (find optimal brick size)
python scripts/renko_engine.py XAUUSD --stage dsp

# 3. Backtest (3 months quick validation)
python scripts/renko_engine.py XAUUSD --stage backtest --months 3

# 4. Paper trading (validate live signals)
python scripts/renko_engine.py XAUUSD --stage paper

# 5. Live micro lots with preflight
python scripts/renko_engine.py XAUUSD --stage live --live-size micro \
  --preflight-test-order --preflight-lots 0.01 \
  --ack-live I_UNDERSTAND_LIVE_RISK
```

## 📚 Documentation

| Document | Purpose |
|----------|---------|
| [QUICK_START.md](QUICK_START.md) | Quick command reference |
| [LIVE_TRADING_CONFIG.md](LIVE_TRADING_CONFIG.md) | Live trading configuration |
| [PREFLIGHT_CHECKS.md](PREFLIGHT_CHECKS.md) | Pre-live safety checks |
| [PIPELINE.md](PIPELINE.md) | Complete trading pipeline |
| [TRADING_ARCHITECTURE.md](TRADING_ARCHITECTURE.md) | System architecture |
| [DEAD_CODE.md](DEAD_CODE.md) | Unused code inventory |

## 🏗️ Architecture

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Raw M1    │───▶│  DSP Brick  │───▶│  Backtest   │───▶│    LIVE     │
│   Data      │    │    Size     │    │ Validation  │    │  Trading    │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
                                                         │
                              ┌──────────────────────────┘
                              ▼
                    ┌─────────────────┐
                    │  Safety Systems │
                    │ ─────────────── │
                    │ • Preflight     │
                    │ • Health Monitor│
                    │ • Circuit Breaker│
                    │ • State Persist │
                    └─────────────────┘
```

### Core Components

| Component | File | Purpose |
|-----------|------|---------|
| Trading Engine | `kinetra/renko/trading_engine.py` | Signal generation, position management |
| cTrader Dispatcher | `kinetra/renko/ctrader_dispatcher.py` | Broker order execution |
| Brick Engine | `kinetra/renko/brick_engine.py` | Renko brick construction |
| Filters | `kinetra/renko/filters.py` | Signal quality filters |
| DSP | `kinetra/renko/dsp.py` | Brick size optimization |
| Preflight | `kinetra/preflight_enhanced.py` | Pre-live safety checks |
| Health Monitor | `kinetra/monitoring/connection_health.py` | Connection health tracking |

## ⚙️ Configuration

### Minimum Required (.env)
```bash
CTRADER_CLIENT_ID=your_client_id
CTRADER_CLIENT_SECRET=your_secret
CTRADER_ACCESS_TOKEN=your_token
CTRADER_ACCOUNT_ID=12345
CTRADER_ENVIRONMENT=demo
```

### Optional Tuning
```bash
CTRADER_FILL_WAIT_S=5.0           # Fill timeout
CTRADER_MAX_FILL_FAILURES=5        # Circuit breaker threshold
CTRADER_HOT_STANDBY=1              # Enable hot standby
KINETRA_DNS_USE_PUBLIC_RESOLVERS=1 # Use public DNS
```

## 🧪 Test Suite

```bash
# Run all tests
pytest tests/ -v

# Run specific test files
pytest tests/test_execution_waiter.py -v
pytest tests/test_trade_lifecycle.py -v
pytest tests/test_fill_failure_tracking.py -v
pytest tests/test_health_monitoring.py -v
```

### Test Coverage

| Test File | Lines | Coverage |
|-----------|-------|----------|
| `test_execution_waiter.py` | 375 | Fill matching, timeouts |
| `test_trade_lifecycle.py` | 268 | Entry/exit, fill races |
| `test_fill_failure_tracking.py` | 200 | Circuit breaker, metrics |
| `test_health_monitoring.py` | 180 | Health service, transitions |
| `test_renko_engine_fill_race.py` | 150 | Async fill processing |
| `test_renko_backtest_latency.py` | 100 | Performance benchmarks |

## 🛡️ Safety Systems

### 1. Enhanced Preflight (11 Checks)
- DNS resolution
- TCP connection pool
- Heartbeat latency
- Broker session
- Account balance
- Margin available
- Symbol resolution
- Market hours
- Hot standby status
- Connection health
- Test order execution (optional)

### 2. Connection Health Monitor
- Periodic health checks (30s interval)
- Latency tracking with anomaly detection
- Packet loss monitoring
- Automatic preemptive failover

### 3. Circuit Breaker
- Triggers after N consecutive fill failures
- Default: 5 failures
- Manual reset required

### 4. Thread Safety
- Bar processor: time-critical, never blocks
- Order executor: handles blocking I/O
- Fill queue: async inter-thread communication
- State lock: protects all position mutations

### 5. State Persistence
- Auto-save on fill confirmation
- Auto-save on trade completion
- Atomic JSON writes
- Recovery on restart

## 📊 Pipeline Validation Gates

| Gate | Requirement |
|------|-------------|
| DSP | Signal/Noise > 2.0, Bricks/day ≥ 5 |
| Backtest | Omega ≥ 1.5, Trades ≥ 30 |
| Paper | Signals match expected frequency |
| Preflight | All 11 checks pass |
| Live | Health = HEALTHY |

## 🔄 Live Dashboard

```
┌─────────────────────────────────────────────────┐
│ XAUUSD LIVE SNAPSHOT  2026-03-04 12:00:00 UTC  │
├─────────────────────────────────────────────────┤
│ Mode            LIVE                            │
│ Symbol          XAUUSD                          │
│ Renko Bricks    42                              │
│ Trades          3                               │
│ Net P&L         $45.50                          │
│                                                 │
│ Health status   HEALTHY                         │
│ Health latency  45.2ms                          │
│ Fill success    100.0%                          │
│ Consec failures 0                               │
│ Orders          3/3 OK, 0 fail                  │
│                                                 │
│ Status          UP                              │
│ Endpoint        demo.ctraderapi.com:5035        │
│ Failovers       0                               │
└─────────────────────────────────────────────────┘
```

## 📝 Recent Changes

### March 2026 — Production Hardening

**New Features:**
- ✅ Atomic persistence with crash recovery
- ✅ DNS hardening with resolver pool
- ✅ Hot standby with automatic failover
- ✅ Enhanced preflight (11 checks)
- ✅ Connection health monitoring
- ✅ Fill failure tracking & circuit breaker
- ✅ Close position verification
- ✅ Thread health monitoring
- ✅ Comprehensive test suite (15+ files)

**Critical Fixes:**
- Fixed close position false success bug
- Fixed fill queue blocking issue
- Fixed exit result checking
- Fixed silent persistence failures

**Documentation:**
- Updated all docs to current state
- Created PIPELINE.md for complete flow
- Created TRADING_ARCHITECTURE.md
- Created DEAD_CODE.md for cleanup

## 🗂️ Repository Structure

```
Kinetra/
├── kinetra/
│   ├── renko/              # Core trading engine
│   │   ├── trading_engine.py      # Main engine
│   │   ├── ctrader_dispatcher.py  # Broker integration
│   │   ├── brick_engine.py        # Renko construction
│   │   ├── filters.py             # Signal filters
│   │   ├── dsp.py                 # Brick optimization
│   │   └── backtest.py            # Backtesting
│   ├── connectors/         # cTrader connection
│   │   ├── ctrader_connector.py
│   │   └── hot_standby.py
│   ├── monitoring/         # Health & telemetry
│   │   └── connection_health.py
│   ├── preflight_enhanced.py      # Safety checks
│   └── dns_hardening.py           # DNS resilience
├── scripts/
│   └── renko_engine.py     # Main entry point
├── tests/                  # Comprehensive test suite
├── configs/                # Configuration files
├── data/                   # Market data (gitignored)
└── docs/                   # Documentation
```

## 🔧 Troubleshooting

### "DNS resolution failed"
```bash
export KINETRA_DNS_USE_PUBLIC_RESOLVERS=1
```

### "Connection timeout"
```bash
# Increase timeout
export CTRADER_FILL_WAIT_S=10.0
```

### "Circuit breaker triggered"
```bash
# Check logs for fill failures
# Reset by restarting after fixing issue
```

### "Preflight failed"
```bash
# Run with verbose output
python scripts/renko_engine.py XAUUSD --stage live \
  --preflight-test-order --ack-live I_UNDERSTAND_LIVE_RISK
```

## 📈 Performance Targets

| Metric | Target | Measurement |
|--------|--------|-------------|
| Omega Ratio | ≥ 1.5 | Profit/loss ratio |
| Win Rate | 35-65% | Realistic range |
| Max Drawdown | < 10% | Risk tolerance |
| Fill Success | > 95% | Execution quality |
| Health Latency | < 500ms | Connection quality |

## ⚠️ Disclaimer

**IMPORTANT**: This software is provided for educational and research purposes only. Trading financial instruments carries significant risk of loss. Past performance does not guarantee future results. The authors are not responsible for any financial losses incurred through use of this software.

Always start with demo/paper trading. Use micro lots when going live. Never risk more than you can afford to lose.

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

---

**Kinetra** - *Institutional-Grade Renko Trading* 🚀
