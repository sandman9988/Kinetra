# Kinetra Live Trading Configuration

> Canonical status has moved to [`docs/RENKO_LIVE_STATE.md`](docs/RENKO_LIVE_STATE.md).
> This file is now an operator quick-reference for launch commands only.

## Quick Start

```bash
# Interactive launcher (recommended)
make launch

# Or direct command line
python scripts/renko_engine.py XAUUSD --stage live [OPTIONS]
```

## Configuration Options

### Lot Size (`--live-size`)

| Tier | Lot Ceiling | Use Case |
|------|-------------|----------|
| `micro` | 0.01 lots | Testing, minimal risk |
| `small` | 0.10 lots | Small accounts |
| `full` | 50.0 lots (broker max) | Full position sizing |

**Default:** `micro`

### Sizing Mode

Sizing mode is currently selected by profile/runtime config in `scripts/renko_engine.py`.
Use strategy overrides (`--target-risk`, `--stop-bricks`, thresholds/windows) to tune behavior.

### Stop Loss (`--stop-bricks`)

Stop loss distance in Renko bricks.

**Default:** `1.0` (from broker defaults)

**Example:** `--stop-bricks 0.5` for tighter stops

### Brick Size (`--brick-size`)

Override the DSP-profile brick size.

**Default:** From DSP profile (e.g., $1.00 for XAUUSD)

### Execution Preflight (`--preflight-test-order`)

Optional real micro-order open/close verification before entering live loop.

Requires:
- `--preflight-test-order`
- `--preflight-lots <value>`
- `--ack-live I_UNDERSTAND_LIVE_RISK`

### Trading Hours (`--monday-start`, `--friday-end`)

Session window in UTC.

**Defaults (XAUUSD):**
- Monday: `03:00` UTC
- Friday: `20:55` UTC

### Target Risk (`--target-risk`)

Risk USD per trade for `risk_based` sizing.

**Default:** $100.00 (from broker defaults)

### Dry Run (`--dry-run`)

Paper trading with live data — NO REAL ORDERS.

**Use for:** Testing configuration before going live

---

## Symbol-Specific Defaults

Defaults are loaded from `scripts/ctrader/launch.sh`:

```bash
# XAUUSD defaults
DEFAULTS_BRICK[XAUUSD]="1.0"
DEFAULTS_STOP_LIVE[XAUUSD]="0.5"
DEFAULTS_MONDAY[XAUUSD]="03:00"
DEFAULTS_FRIDAY[XAUUSD]="20:55"
DEFAULTS_DD_HALT_LIVE[XAUUSD]="0.02"
DEFAULTS_TARGET_RISK[XAUUSD]="100.0"

# NAS100 defaults
DEFAULTS_BRICK[NAS100]="5.0"
DEFAULTS_STOP_LIVE[NAS100]="1.0"
DEFAULTS_MONDAY[NAS100]="14:30"
DEFAULTS_FRIDAY[NAS100]="21:00"
```

---

## Example Commands

### Conservative Live Trading (XAUUSD)
```bash
python scripts/renko_engine.py XAUUSD --stage live \
  --live-size micro \
  --stop-bricks 0.5
```

### Aggressive Compounding (XAUUSD)
```bash
python scripts/renko_engine.py XAUUSD --stage live \
  --live-size full \
  --stop-bricks 1.0
```

### Dry-Run Testing
```bash
python scripts/renko_engine.py XAUUSD --stage live \
  --live-size full \
  --stop-bricks 0.5 \
  --dry-run
```

### Custom Trading Hours
```bash
python scripts/renko_engine.py XAUUSD --stage live \
  --monday-start 03:00 \
  --friday-end 20:55
```

---

## Full Workflow

1. **DSP Analysis** (first time only)
   ```bash
   python scripts/renko_engine.py XAUUSD --stage dsp
   ```

2. **Backtest Validation**
   ```bash
   python scripts/renko_engine.py XAUUSD --stage backtest --months 24
   ```

3. **Paper Trading**
   ```bash
   python scripts/renko_engine.py XAUUSD --stage paper
   ```

4. **Dry-Run** (live data, paper orders)
   ```bash
   python scripts/renko_engine.py XAUUSD --stage live --dry-run
   ```

5. **LIVE Trading** (REAL ORDERS)
   ```bash
   python scripts/renko_engine.py XAUUSD --stage live --live-size micro
   ```
   Optional execution preflight:
   ```bash
   python scripts/renko_engine.py XAUUSD --stage live --live-size micro \
     --preflight-test-order --preflight-lots 0.01 \
     --ack-live I_UNDERSTAND_LIVE_RISK
   ```

---

## Safety Checklist

Before going live:

- [ ] Backtest shows Omega ≥ 1.5
- [ ] Backtest shows ≥ 30 trades
- [ ] Paper trading validates live execution
- [ ] Dry-run confirms broker connectivity
- [ ] Lot size appropriate for account equity
- [ ] Stop loss configured correctly
- [ ] Trading hours match your availability
- [ ] You understand the risks

**⚠️ NEVER trade more than you can afford to lose.**

---

## Troubleshooting

### "cTrader connector not available"
```bash
pip install ctrader-open-api
```

### "No DSP profile found"
```bash
python scripts/renko_engine.py XAUUSD --stage dsp
```

### "Failed to connect to cTrader"
- Check `.env.openapi` credentials
- Verify network connectivity
- Check cTrader account status

### Wrong lot size
- Verify `--live-size` matches your intent
- Check broker `volume_max` in `contract_spec.json`

---

## See Also

- [`docs/RENKO_LIVE_STATE.md`](docs/RENKO_LIVE_STATE.md) — Canonical live state and snapshot semantics
- `SWAP_MODELING_TODO.md` — Swap cost modeling limitations
- `scripts/ctrader/launch.sh` — Interactive launcher defaults
