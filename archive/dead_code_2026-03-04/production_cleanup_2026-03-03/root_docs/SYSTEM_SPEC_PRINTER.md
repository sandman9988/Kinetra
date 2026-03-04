# ✅ SYSTEM SPECIFICATION PRINTER IMPLEMENTED

## What's New

Before backtest results, the system now displays a comprehensive **Trading System Specification** table showing:

### Trading Parameters Displayed

**Instrument & Broker**
- Symbol: XAUUSD
- Contract Size: 100 oz
- Tick Size: $0.01 per tick

**Brick Configuration**
- Brick Size: $1.00 (from DSP)
- Brick Window: 40-50 bricks (empirically measured)

**Entry & Exit Rules**
- Entry Signal: Colour flip + filters (2-brick direction change)
- FlipRate Gate: < 0.35 (reject choppy markets)
- Markov Gate: > 0.55 (require direction persistence)
- Stop Loss (SL): 0.5-1.0 bricks ($0.50-$1.00)
- Exit Signal: Colour change (first opposite brick)
- Trailing Stop: Off (uses fixed stop)

**Friction & Costs** (from contract_spec.json)
- Spread: 22 ticks ($0.22)
- Commission: $7.00/lot (ECN round-trip)
- Swap Long: -$1.017/day (cost to hold long)
- Swap Short: +$0.478/day (earn on shorts)
- Triple Swap: Wednesday (3× on Wed)

**Sizing**
- Initial Equity: $10,000.00
- Risk per Trade: $100.00
- Lot Ceiling: 10.0 lots (compounding), 0.01 lots (static)
- Sizing Mode: Dual scenario (static + compounding)

**Session & Hours**
- Trading Hours: 24/5 (Monday-Friday, UTC)
- Week Start: Monday 00:00 UTC
- Week Close: Friday 24:00 UTC (swap settlement)

**Physics & DSP**
- VR Peak Scale: 12 M30 bars (trend persistence)
- Regime: RANDOM_WALK (from DSP analysis)

**Performance Targets**
- Min Omega: ≥ 1.5 (statistical significance)
- Min Trades: ≥ 30 (sample size)
- Target Win Rate: > 50%
- Target MFE/MAE: > 1.5 (execution quality)

---

## Example Output

```
════════════════════════════════════════════════════════════════════════════════
                    TRADING SYSTEM SPECIFICATION
════════════════════════════════════════════════════════════════════════════════
Parameter                     Value                    Notes
────────────────────────────────────────────────────────────────────────────────
Symbol                        XAUUSD                   cTrader Pepperstone ECN
Contract Size                 100 oz                   Standard gold lot
Tick Size                     $0.01                    Value per tick: $1.00
Brick Size                    $1.00                    Price movement threshold
Brick Window                  43 bricks                Filter lookback period (~0.25 days)
Entry Signal                  Colour flip + filters    2-brick direction change required
FlipRate Gate                 < 0.35                   Reject choppy markets (% flips)
Markov Gate                   > 0.55                   Require direction persistence
Stop Loss (SL)                0.5 brick                $0.50 fixed distance
Exit Signal                   Colour change (opposite) Exit on first reversal brick
Trailing Stop                 Off                      Standard 1-brick fixed stop used
Spread                        22.0 ticks               $0.22 per round-trip
Commission                    $7.00/lot                ECN round-trip per standard lot
Swap Long                     -$1.017/day              Cost to hold long positions overnight
Swap Short                    +$0.478/day              Earn on short positions (positive carry)
Triple Swap                   Wednesday                3× swap charged on Wednesdays
Initial Equity                $10,000.00               Starting account balance
Risk per Trade                $100.00                  Target USD at risk per position
Lot Ceiling                   10.00 lots               Maximum position size cap
Sizing Mode (Backtest)        Dual scenario            Static (0.01) + Compounding (max 10.0)
Trading Hours                 24/5 (Mon-Fri)           No weekend trading (forex session)
Week Start                    Monday 00:00 UTC         New trading week begins
Week Close                    Friday 24:00 UTC         End of week, swap settlement
VR Peak Scale                 12 M30 bars              Trend persistence peak from DSP
Regime                        RANDOM_WALK              Market classification from DSP
Min Omega                     ≥ 1.5                    Statistical significance gate
Min Trades                    ≥ 30                     Sample size sufficiency
Target Win Rate               > 50%                    More winners than losers
Target MFE/MAE                > 1.5                    Execution quality (capture favorable moves)
════════════════════════════════════════════════════════════════════════════════

============================================================
STAGE 3: BACKTEST (3 months)
============================================================
Testing 90720 bars (3 months)

--- STATIC SIZING ---
[Stats panel with detailed metrics...]
✅ PASS

--- COMPOUNDING SIZING ---
[Stats panel with detailed metrics...]
✅ PASS
```

---

## How It Works

1. **Before backtest runs**, system loads:
   - `contract_spec.json` (broker calibration: spread, commission, swap)
   - `dsp_profile.json` (physics analysis: brick size, regime, VR scale)
   - Engine config (filters, stops, windows)

2. **Prints spec table** showing ALL parameters

3. **Runs backtest** with both sizing scenarios

4. **Shows performance** with full stats (trades, omega, MAE/MFE, streaks, etc.)

---

## Execute with System Spec Output

```bash
cd /home/renierdejager/Projects/Kinetra && python scripts/renko_engine.py XAUUSD --stage backtest --months 3
```

**Output Flow:**
1. System Specification table (all parameters)
2. Static sizing backtest results
3. Compounding sizing backtest results

---

## Benefits

✅ **Transparency** - See exactly what parameters are being tested  
✅ **Auditability** - Full system documented before results  
✅ **Reproducibility** - Can verify settings match intent  
✅ **Debugging** - Easy to spot calibration issues  
✅ **Validation** - Confirm contract specs are loaded correctly

---

**Status: READY FOR BACKTEST** 🚀
