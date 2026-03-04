# Kinetra Live Trading — Implementation Summary

## Overview

Complete live trading implementation with comprehensive safety checks, configuration options, and broker integration via cTrader Open API.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Kinetra Live Trading                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐      │
│  │   Launcher   │───▶│  Renko       │───▶│  cTrader     │      │
│  │  (bash)      │    │  Engine      │    │  Connector   │      │
│  └──────────────┘    └──────────────┘    └──────────────┘      │
│         │                   │                   │                │
│         ▼                   ▼                   ▼                │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐      │
│  │  CLI Options │    │  Preflight   │    │  Order       │      │
│  │  & Defaults  │    │  Checks      │    │  Dispatcher  │      │
│  └──────────────┘    └──────────────┘    └──────────────┘      │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Components

### 1. Interactive Launcher (`scripts/ctrader/launch.sh`)

**Features:**
- Menu-driven interface
- Symbol selection (NAS100, XAUUSD)
- Gate selection (micro, small, full)
- Auto-loads symbol-specific defaults
- Confirmation prompt for live trading

**Usage:**
```bash
make launch
# Select: 5) LIVE
# Select symbol
# Select gate
# Type: I_UNDERSTAND_LIVE_RISK
```

### 2. CLI Options (`scripts/renko_engine.py`)

| Option | Description | Default |
|--------|-------------|---------|
| `--live-size` | Lot size tier | micro |
| `--sizing-mode` | Position sizing | static |
| `--stop-bricks` | Stop loss (bricks) | 1.0 |
| `--brick-size` | Brick size ($) | from DSP |
| `--trailing-stop` | Enable trailing | off |
| `--monday-start` | Session start UTC | from DSP |
| `--friday-end` | Session end UTC | from DSP |
| `--target-risk` | Risk USD/trade | from DSP |
| `--dry-run` | Paper trading | off |

### 3. Preflight Checks (`_run_preflight_checks()`)

**6 Safety Checks:**

1. **DNS Resolution** — Broker endpoint validation
2. **Network Latency** — <500ms ideal, <1000ms acceptable
3. **Authentication** — Valid credentials & session
4. **Account Balance** — ≥$100 for micro lots
5. **Symbol Tradability** — Instrument available
6. **Test Order** — Full open + close cycle (0.01 lots)

**Behavior:**
- Runs automatically before live trading
- Skipped for dry-run mode
- Aborts on any failure
- Detailed error reporting

### 4. Order Dispatcher (`CTraderOrderDispatcher`)

**Capabilities:**
- Market order submission with hard stop-loss
- Position closing by position ID
- Fill event tracking via execution events
- Spread estimation from live quotes
- Volume conversion (lots → cTrader units)

**Order Flow:**
```
Open Position:
1. Generate clientOrderId
2. Register execution waiter
3. Send ProtoOANewOrderReq
4. Wait for ProtoOAExecutionEvent
5. Extract positionId
6. Return OrderResult

Close Position:
1. Send ProtoOAClosePositionReq
2. Wait for response
3. Return success/failure
```

### 5. Configuration Defaults

**Symbol-specific defaults** (from `launch.sh`):

```bash
# XAUUSD
DEFAULTS_BRICK[XAUUSD]="1.0"
DEFAULTS_STOP_LIVE[XAUUSD]="0.5"
DEFAULTS_MONDAY[XAUUSD]="03:00"
DEFAULTS_FRIDAY[XAUUSD]="20:55"
DEFAULTS_TARGET_RISK[XAUUSD]="100.0"

# NAS100
DEFAULTS_BRICK[NAS100]="5.0"
DEFAULTS_STOP_LIVE[NAS100]="1.0"
DEFAULTS_MONDAY[NAS100]="14:30"
DEFAULTS_FRIDAY[NAS100]="21:00"
```

**Broker contract spec** (from `contract_spec.json`):
- `volume_max`: 50.0 lots (XAUUSD)
- `spread_points`: 22.0 ticks
- `commission_per_lot`: $3.50
- `tick_size`: 0.01
- `contract_size`: 100 oz

---

## Workflows

### Full Live Trading Workflow

```bash
# 1. DSP Analysis (first time)
python scripts/renko_engine.py XAUUSD --stage dsp

# 2. Backtest Validation
python scripts/renko_engine.py XAUUSD --stage backtest --months 24

# 3. Paper Trading
python scripts/renko_engine.py XAUUSD --stage paper

# 4. Dry-Run (live data, paper orders)
python scripts/renko_engine.py XAUUSD --stage live --dry-run

# 5. LIVE Trading (REAL ORDERS)
python scripts/renko_engine.py XAUUSD --stage live --live-size micro
```

### Configuration Examples

**Conservative Live (XAUUSD):**
```bash
python scripts/renko_engine.py XAUUSD --stage live \
  --live-size micro \
  --sizing-mode static \
  --stop-bricks 0.5
```

**Aggressive Compounding:**
```bash
python scripts/renko_engine.py XAUUSD --stage live \
  --live-size full \
  --sizing-mode compounding \
  --stop-bricks 1.0 \
  --trailing-stop
```

**Custom Trading Hours:**
```bash
python scripts/renko_engine.py XAUUSD --stage live \
  --monday-start 03:00 \
  --friday-end 20:55
```

---

## Safety Features

### 1. Preflight Verification
- Network connectivity confirmed
- Broker authentication verified
- Account balance checked
- Symbol availability confirmed
- Test order executed

### 2. Lot Size Limits
- `micro`: 0.01 lots (max)
- `small`: 0.10 lots (max)
- `full`: broker volume_max (50.0 for XAUUSD)

### 3. Risk Management
- Stop-loss enforced on every order
- Target risk per trade configurable
- Drawdown monitoring (from defaults)
- Trading hours enforced

### 4. Confirmation Prompts
- Interactive launcher requires explicit confirmation
- Must type `I_UNDERSTAND_LIVE_RISK` exactly
- Clear warnings about real money at risk

### 5. Graceful Shutdown
- Ctrl+C stops trading cleanly
- Final stats printed
- Connection closed properly
- Position state preserved

---

## Monitoring & Stats

### Periodic Stats (every 60s)
```
╭──────────── XAUUSD  [LIVE]  2026-03-03 00:00 UTC ────────────╮
│   Trades               15  (9W / 6L)   Win rate        60.0%  │
│   Net P&L                    $27.50    Total return     2.8%  │
│   Omega                      2.15      Z-Factor         1.82  │
│   Gross profit               $45.00    Gross loss     -$17.50  │
│   → Spread cost              -$6.60    → Commission   -$0.53   │
│   → Swap                     -$0.00    → Total fric.  -$7.13   │
│   Max drawdown               -0.08%    Max DD $        -$4.20  │
╰────────────────────────────────────────────────────────────────╯
```

### Final Results
```
════════════════════════════════════════════════════════════════
  LIVE TRADING RESULTS
════════════════════════════════════════════════════════════════
  Trades:  47
  Net P&L: $125.50
  Omega:   2.34

⚠️  Always verify orders in cTrader terminal
```

---

## Friction Cost Breakdown

**Tracked per trade:**
- **Spread cost**: Bid-ask spread × lots
- **Commission**: ECN commission × lots
- **Swap**: Overnight financing (TODO: implement)
- **Total friction**: Sum of all costs

**Typical XAUUSD costs:**
- Spread: ~$22.00 per round-trip (22 ticks)
- Commission: ~$3.50 per lot
- Swap: ~$10/lot/day (not yet modeled)

---

## Known Limitations

### 1. Swap Modeling (TODO)
- Current: $0 (not modeled)
- Impact: Understates friction for overnight positions
- Priority: Low (avg hold time 0.2h)
- See: `SWAP_MODELING_TODO.md`

### 2. Trailing Stop (TODO)
- CLI flag exists but not yet implemented in engine
- Planned: 50% MFE after 2 bricks
- Requires: Engine modification

### 3. Multi-Symbol (TODO)
- Current: Single symbol per session
- Planned: Portfolio trading with allocation agent

---

## Files Modified

| File | Changes |
|------|---------|
| `scripts/renko_engine.py` | Live stage, CLI options, preflight checks |
| `scripts/ctrader/launch.sh` | Auto-build live options from defaults |
| `kinetra/renko/trading_engine.py` | Friction breakdown (spread/commission/swap) |
| `kinetra/renko/live_trader.py` | LiveTrade friction fields |
| `kinetra/renko/trade_analytics.py` | Friction analytics |
| `data/.../contract_spec.json` | Fixed spread (22 ticks), contract_size (100 oz) |

---

## Documentation

| Document | Purpose |
|----------|---------|
| `LIVE_TRADING_CONFIG.md` | Configuration reference |
| `PREFLIGHT_CHECKS.md` | Preflight safety checks |
| `SWAP_MODELING_TODO.md` | Swap modeling limitations |
| `LIVE_TRADING_IMPLEMENTATION.md` | This summary |

---

## Testing Checklist

- [x] CLI options parse correctly
- [x] Preflight checks compile
- [x] Dry-run mode works (paper dispatcher)
- [x] Live mode uses CTraderOrderDispatcher
- [x] Symbol defaults load correctly
- [x] Broker volume_max used for "full" tier
- [x] Friction breakdown displays correctly
- [ ] **Live test with broker** (requires credentials)
- [ ] **Test order execution** (requires live account)

---

## Next Steps

1. **Test with live broker connection**
   - Verify preflight checks pass
   - Confirm test order executes
   - Monitor fill latency

2. **Implement swap modeling**
   - Use hold duration × swap rate
   - Handle triple swap day (Wednesday)
   - Update friction breakdown

3. **Add trailing stop**
   - Track MFE per trade
   - Move stop after 2 bricks
   - Lock in 50% of MFE

4. **Multi-symbol support**
   - Portfolio allocation
   - Correlation monitoring
   - Risk distribution

---

## Support

For issues or questions:
1. Check `PREFLIGHT_CHECKS.md` for troubleshooting
2. Review `LIVE_TRADING_CONFIG.md` for configuration
3. Verify broker credentials in `.env.openapi`
4. Test with `--dry-run` before going live

**⚠️ Always start with dry-run mode to validate configuration before trading real money.**
