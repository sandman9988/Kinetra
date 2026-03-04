# Kinetra Pipeline v3 Refactoring

## Overview
Replaced the complex multi-menu system (`kinetra_menu.py` with 1,253 lines) with a simplified pipeline that focuses on:
1. **Qualification screening** (Stage 1 & 3)
2. **Live trading** with PER gates (paper → micro → small → full)
3. **Status monitoring**

## New Pipeline Architecture

### Entry Points

```
make screen    → python scripts/pipeline_v3.py screen
make trade     → python scripts/pipeline_v3.py trade
make status    → python scripts/pipeline_v3.py status
```

### Qualification Flow

```
STAGE 1: DSP Screen (3 months)
├─ Quick brick size validation
├─ 100-200 trade backtest
├─ Gate: Omega > 0.5 OR Z > 1.0
└─ If PASS → Stage 3

STAGE 3: Multi-Window Backtest (1-3 years)
├─ Load full history data
├─ 70/30 IS/OOS rolling windows
├─ Window step: 3 months
├─ Gate: Omega_IS ≥ 1.5 AND Omega_OOS ≥ 1.2 AND Survival ≥ 80%
└─ If PASS → QUALIFY

QUALIFICATION.JSON
├─ qualified: bool
├─ omega: float
├─ omega_oos: float
├─ n_trades: int
├─ survival_rate: float
├─ disqualification_reason: str (if not qualified)
└─ ... (other fields)
```

### Trading Flow

```
PAPER GATE
├─ No real orders
├─ Lots = paper_lots (0.01)
├─ Target: Omega ≥ 1.5
└─ If pass → unlock MICRO

MICRO GATE (0.01 lots)
├─ Real orders, 0.01 max
├─ Max DD: 3%
├─ Max instruments: 2
├─ Target: Omega ≥ 2.0
└─ If pass → unlock SMALL

SMALL GATE (0.1 lots)
├─ Real orders, 0.1 max
├─ Max DD: 5%
├─ Max instruments: 5
├─ Target: Omega ≥ 2.5
└─ If pass → unlock FULL

FULL GATE
├─ Real orders, broker limits
├─ Max DD: 10%
├─ Unlimited instruments
└─ Live deployment
```

## Files Changed

### New Files
- `scripts/pipeline_v3.py` — Main CLI for pipeline (screen / trade / status)
- `scripts/renko/stage1_dsp_screen.py` — Stage 1 implementation (DSP validation)
- `scripts/renko/stage3_multiwindow_backtest.py` — Stage 3 implementation (full backtest)

### Modified Files
- `Makefile` — Updated with new make targets (screen, trade, status)
- `kinetra/renko/live_trader.py` — Added `skip_qualification` flag + blocker fix

### Deprecated Files
- `kinetra_menu.py` — Kept for now, but replaced by pipeline_v3.py
- `scripts/ctrader/launch.sh` — Still works via `make launch` but not recommended

## Usage Examples

### 1. Check System Status
```bash
make status
# Shows: total symbols, qualified count, screening status
```

### 2. Qualify a New Symbol
```bash
make screen
# Prompts: select symbol
# Runs: Stage 1 (DSP) + Stage 3 (multi-window)
# Writes: qualification.json
```

### 3. Start Paper Trading
```bash
make trade
# Prompts: symbol, gate (paper/micro/small/full), mode (paper/dry_run/live)
# Launches: RenkoLiveTrader with appropriate config
```

### 4. Skip Qualification (Testing)
```bash
python scripts/pipeline_v3.py trade --symbol XAUUSD --gate paper
# Will ask: "Symbol not qualified. Proceed anyway?" → Yes
# Launches with skip_qualification=True
```

## Configuration

### LiveTraderConfig Changes
```python
# New field:
skip_qualification: bool = False  # If True, allow trading unqualified symbols
```

### Blocker Counting Fix
- **Previous:** Blockers incremented once per bar
- **Now:** Blockers only counted once per brick that fails the check
- **Implementation:** Track `last_checked_brick_idx` per instrument
- **Result:** Blocker counts now match actual brick events

## Transition Plan

### Phase 1 (NOW)
- ✅ Pipeline v3 created and tested
- ✅ Blocker fix deployed
- ✅ Makefile updated

### Phase 2 (NEXT)
- Implement Stage 1 (DSP screen) with real backtest logic
- Implement Stage 3 (multi-window backtest) with rolling window logic
- Add data loading utilities

### Phase 3 (LATER)
- Archive kinetra_menu.py
- Remove legacy launcher scripts
- Consolidate to pipeline_v3.py only

## Testing

```bash
# Test status
python scripts/pipeline_v3.py status

# Test screen (dry-run)
python scripts/pipeline_v3.py screen --symbol XAUUSD

# Test trade (dry-run, skip qual)
python scripts/pipeline_v3.py trade --symbol XAUUSD --gate paper --mode dry_run
```

## Key Design Decisions

1. **No menu system** — Direct CLI with click (simpler, faster)
2. **Qualification is blocking** — All symbols must pass before trading
3. **Loose Stage 1 gate** — Just filter obvious losers (Omega > 0.5)
4. **Strict Stage 3 gate** — Require robust multi-window validation (Omega_IS ≥ 1.5, OOS ≥ 1.2)
5. **PER gates are progressive** — Paper → Micro → Small → Full requires continuous performance
6. **Skip option for testing** — Can force-trade unqualified symbols with `--skip-qual` flag

## Metrics & Monitoring

### Qualification Metrics
- **Omega_IS**: In-sample Renko performance ratio
- **Omega_OOS**: Out-of-sample validation
- **Survival Rate**: % of rolling windows that achieved breakeven or better
- **Win Rate**: % winning trades
- **Max DD**: Maximum drawdown during backtest

### Live Trading Metrics
- **Session PnL**: Cumulative P&L in USD
- **Omega**: Current paper/live performance ratio
- **Portfolio DD**: Current drawdown vs peak equity
- **Entry Signals**: Count of valid flip signals
- **Blockers**: Count of rejected entry attempts by reason

## Notes

- The `no_flip` blocker now only increments when you actually try to enter on a non-flip brick
- Blocker counts should match observable brick events in the UI
- Each PER gate has strict pass criteria (Omega thresholds are enforced)
- Paper trading is mandatory first step before any real money
