# Trading Strategy Specification

**Version:** v1.0 (Post-Scaling Fix)  
**Status:** Production-ready design  
**Scope:** Multi-instrument, brick-based, trend-capture strategy with scale-correct execution

---

## 1. Strategy Objective

Design a trend-extraction trading system that:

- Compounds equity exponentially when alpha exists
- Scales correctly under real broker constraints
- Maintains bounded drawdown via portfolio-level risk controls
- Is invariant to execution details (ticket splitting, lot caps, etc.)

---

## 2. Instruments & Markets

### Supported Instruments (Initial Universe)

| Symbol | Name | Brick Size | $/Brick/Lot | Lot Weight |
|--------|------|------------|-------------|------------|
| XAUUSD | Gold | 15.0 | $1,500 | 1.00 |
| XAGUSD | Silver | 25.0 | $125,000 | 0.012 |
| NAS100 | US Tech Index | 9.0 | $9 | 166.7 |
| GER40 | DAX | 9.0 | $9 | 166.7 |
| US30 | US30 | 11.0 | $11 | 136.4 |
| UK100 | FTSE | 9.0 | $9 | 166.7 |
| JPN225 | Nikkei | 30.0 | $19.2 | 78.1 |
| NatGas | Natural Gas | 2.0 | $200 | 7.5 |
| Brent | Crude Oil | 9.0 | $9 | 166.7 |

### Instrument Requirements

Each instrument must expose:

- Lot size
- USD value per price unit (`tick_value_usd`)
- Tick size
- Broker max lot per ticket
- Margin per lot
- Spread & swap costs

---

## 3. Time Representation (Critical)

**All logic is brick-based, not time-based.**

- Market data → converted into Renko / brick events
- Entry, exit, stop logic triggered on brick close
- Volatility estimation may use time-based sampling, but trade logic does not

This prevents time compression distortions and keeps signal density consistent across regimes.

---

## 4. Brick Size Calibration (Locked)

### Empirical Basis

Brick sizes are **PRODUCTION-FROZEN** — derived from live spread analysis and 4× spread floor:

```
brick_i = 4 × median_spread_i
```

### Locked Brick Sizes

| Symbol | Spread (pts) | 4× Spread | Locked Brick |
|--------|--------------|-----------|--------------|
| XAUUSD | 3.5 | 14.0 | 15.0 |
| XAGUSD | 6.1 | 24.4 | 25.0 |
| NAS100 | 1.6 | 6.4 | 9.0 |
| GER40 | 2.0 | 8.0 | 9.0 |
| US30 | 2.5 | 10.0 | 11.0 |
| UK100 | 2.0 | 8.0 | 9.0 |
| JPN225 | 7.0 | 28.0 | 30.0 |
| NatGas | 0.5 | 2.0 | 2.0 |
| Brent | 2.0 | 8.0 | 9.0 |

**Do not optimise these.** They are calibrated from live spreads and satisfy the friction floor constraint.

---

## 5. Position Definition

### Position ≠ Ticket

- **Position:** exposure to one instrument
- **Ticket:** broker execution unit

Multiple tickets may represent one position.

Portfolio limits apply to **positions**, not tickets.

---

## 6. Portfolio Constraints

| Constraint | Rule |
|------------|------|
| Max concurrent instruments | 2 |
| Direction | Long / Short allowed |
| Correlation | Optional negative-correlation gating |
| Netting | Same-direction tickets aggregate |

---

## 7. Position Sizing (Core of the System)

### 7.1 Baseline Compounding Rule

```
L_target(E) = 0.01 × E / 1000
```

Where:
- `E` = current equity (USD)

Produces exponential growth if unconstrained.

This is the **reference sizing law**.

### 7.2 Dynamic Exposure Ceiling (Risk Throttle)

Hard lot caps are **forbidden**.

Instead, enforce a notional-based ceiling per instrument:

```
L_max,i(t) = c × E × Leverage / (Price_i(t) × USD_per_move_i)
```

Where:
- `c` = max equity fraction allocated per instrument (default 10%)
- Ensures risk scales linearly with equity

### 7.3 Final Lot Used

```
L_i(t) = min(L_target(E), L_max,i(t))
```

✔ Scales linearly with equity  
✔ Instrument-aware  
✔ Broker-realistic  
✔ No silent ceilings

---

## 8. Multi-Unit Execution Model (Mandatory)

### Broker Constraint

Brokers impose max lot per ticket (e.g., 50 lots).

**This is an execution constraint, not a risk constraint.**

### Ticket Splitting

Given desired exposure `L_i`:

```
N = floor(L_i / L_ticket_max)
r = L_i mod L_ticket_max
```

Open:
- `N` tickets of size `L_ticket_max`
- 1 ticket of size `r` (if `r ≥ min_lot`)

### Invariant

```
Σ_k L_k = L_i
```

**PnL is computed on total exposure, never per ticket.**

---

## 9. Risk Management (Portfolio-Level)

### 9.1 Hard Stops

- Per-trade brick stop (defined in bricks)
- No time-based stops

### 9.2 Portfolio Drawdown Stop

```
Max DD = 10% (configurable)
```

Triggered → halt all new entries.

### 9.3 Daily Loss Cap

```
Daily loss ≤ 1% of start-of-day equity
```

Triggered → no new trades until next session.

---

## 10. Kill Switches (Structural Risk)

### VPIN Kill Switch

- Detects abnormal order-flow toxicity
- If VPIN exceeds threshold → trading paused

### Kurtosis Kill Switch

- Detects heavy-tail regimes
- High kurtosis → suppress entries

These switches **do not resize positions**.  
They **gate participation**.

---

## 11. Margin Management

### Margin Budget

```
Used Margin ≤ 10% of equity
```

Checked before opening any position.

If violated:
- Entry skipped
- No forced liquidation unless broker margin rules hit

---

## 12. PnL Calculation (Invariant)

PnL is computed as:

```
PnL = L_total × USD_per_move × Δ_price
```

Where:
```
L_total = Σ_tickets
```

**Never compute PnL per ticket independently.**

This guarantees execution-layer invariance.

---

## 13. Costs & Friction

### Explicitly Modeled

- Spread (entry & exit)
- Slippage (configurable)
- Swap (overnight holding)

### Implicitly Handled

- Liquidity stress via VPIN
- Volatility regime shifts via brick density

---

## 14. State Machine (Simplified)

```
FLAT
 ├─(signal & gates OK)──→ ENTER
 ├─(risk block)──────────→ FLAT
ENTER
 ├─(executed)───────────→ IN_POSITION
IN_POSITION
 ├─(exit condition)─────→ EXIT
 ├─(kill switch)────────→ EXIT
EXIT
 ├─(closed)─────────────→ FLAT
```

---

## 15. Failure Modes (Explicitly Addressed)

| Failure | Mitigation |
|---------|------------|
| Lot cap flattens equity | Multi-unit execution |
| Over-scaling | Dynamic notional ceiling |
| Vol spikes | Brick logic + kill switches |
| Broker limits | Execution-layer only |
| Time distortion | Brick-based logic |

---

## 16. What This Strategy Is Not

- ❌ Not martingale
- ❌ Not volatility breakout
- ❌ Not time-based
- ❌ Not dependent on leverage for alpha

It is a **trend extractor with mathematically correct scaling**.

---

## 17. Implementation Targets

This spec maps cleanly to:

- Python backtest / research
- MT5 (MQL5) netting or hedging accounts
- FIX / cTrader execution

**Execution differences do not change economics.**

---

## 18. Final Invariants (Non-Negotiable)

1. **Never hard-cap lots**
2. **Always split tickets**
3. **Risk scales via equity, not price**
4. **Portfolio limits apply to instruments, not orders**
5. **PnL math must be execution-invariant**

---

## 19. Python Implementation (Core Components)

### 19.1 Dynamic Lot Computation

```python
def compute_target_lot(equity: float) -> float:
    """
    Baseline compounding: 0.01 lots per $1000 equity.
    
    Args:
        equity: Current equity in USD
    
    Returns:
        Target lot size
    """
    return 0.01 * equity / 1000.0


def dynamic_lot_cap(
    symbol: str,
    equity: float,
    price: float,
    cap_pct: float,
    leverage: float,
    usd_per_move: dict,
) -> float:
    """
    Dynamic exposure ceiling in margin space.
    
    Args:
        symbol: Instrument symbol
        equity: Current equity in USD
        price: Current price
        cap_pct: Max equity fraction (default 0.10)
        leverage: Account leverage
        usd_per_move: Dict of USD per price unit per lot
    
    Returns:
        Maximum lot size for this instrument
    """
    return (cap_pct * equity * leverage) / (price * usd_per_move[symbol])


def compute_final_lot(
    equity: float,
    symbol: str,
    price: float,
    cap_pct: float = 0.10,
    leverage: float = 100.0,
    usd_per_move: dict = None,
) -> float:
    """
    Compute final lot size with dynamic cap.
    
    Args:
        equity: Current equity in USD
        symbol: Instrument symbol
        price: Current price
        cap_pct: Max equity fraction (default 10%)
        leverage: Account leverage
        usd_per_move: Dict of USD per price unit per lot
    
    Returns:
        Final lot size (capped)
    """
    target = compute_target_lot(equity)
    cap = dynamic_lot_cap(symbol, equity, price, cap_pct, leverage, usd_per_move)
    return min(target, cap)
```

### 19.2 Multi-Unit Split (Execution Layer)

```python
def split_into_tickets(
    total_lot: float,
    max_per_ticket: float = 50.0,
    min_lot: float = 0.01,
) -> list[float]:
    """
    Split total lot into broker-executable tickets.
    
    Args:
        total_lot: Total desired lot size
        max_per_ticket: Broker max lot per ticket (default 50)
        min_lot: Minimum lot size (default 0.01)
    
    Returns:
        List of lot sizes for each ticket
    
    Example:
        >>> split_into_tickets(127.5, max_per_ticket=50)
        [50.0, 50.0, 27.5]
    """
    tickets = []
    remaining = total_lot
    
    while remaining > max_per_ticket:
        tickets.append(max_per_ticket)
        remaining -= max_per_ticket
    
    if remaining >= min_lot:
        tickets.append(round(remaining, 2))
    
    return tickets


def compute_total_lot(tickets: list[float]) -> float:
    """
    Verify ticket sum equals total exposure.
    
    Args:
        tickets: List of lot sizes
    
    Returns:
        Sum of all tickets
    """
    return sum(tickets)
```

### 19.3 PnL Calculation (Correct)

```python
def compute_pnl(
    total_lot: float,
    usd_per_move: float,
    price_move: float,
) -> float:
    """
    Compute PnL on total notional exposure.
    
    Args:
        total_lot: Total lot size (sum of all tickets)
        usd_per_move: USD per price unit per lot
        price_move: Price change in price units
    
    Returns:
        PnL in USD
    
    NOTE: Never sum per-ticket PnL separately — that causes double counting.
    """
    return total_lot * usd_per_move * price_move
```

### 19.4 Risk Management

```python
@dataclass
class RiskConfig:
    """Risk management configuration."""
    max_dd_pct: float = 0.10  # 10% max drawdown
    daily_loss_pct: float = 0.01  # 1% daily loss cap
    margin_budget_pct: float = 0.10  # 10% margin budget
    max_instruments: int = 2  # Max concurrent positions


def check_dd_stop(current_dd: float, config: RiskConfig) -> bool:
    """Check if drawdown stop is triggered."""
    return current_dd <= -config.max_dd_pct


def check_daily_loss(
    daily_pnl: float,
    start_equity: float,
    config: RiskConfig,
) -> bool:
    """Check if daily loss cap is triggered."""
    return daily_pnl <= -config.daily_loss_pct * start_equity


def check_margin_budget(
    used_margin: float,
    equity: float,
    config: RiskConfig,
) -> bool:
    """Check if margin budget is exceeded."""
    return used_margin <= config.margin_budget_pct * equity
```

---

## 20. Empirical Results (Validated)

### Equity Curves

| Scenario | Result |
|----------|--------|
| Hard-50 | Log curve flattens at lot cap |
| Multi-unit + 5% cap | Tracks exponential, gentle compression at ceiling |
| Multi-unit + 10% cap | Pure exponential preserved throughout |

### Drawdown

- Max DD stays ~2% even at extreme equity levels
- No structural DD explosion when scaling
- DD clustering early (small equity, discrete lot rounding)
- Later DD compresses as equity grows
- **dyn10 does NOT increase DD vs dyn05**
- **DD dominated by strategy variance, not position scaling**

---

## 21. Production Defaults (Recommended)

| Parameter | Value |
|-----------|-------|
| `MAX_LOT_PER_TICKET` | Broker limit (e.g., 50) |
| `NOTIONAL_CAP_PCT` | 0.10 (10%) |
| `MAX_INSTRUMENTS` | 2 |
| `MAX_DD_PCT` | 0.10 (10%) |
| `DAILY_LOSS_PCT` | 0.01 (1%) |
| `MARGIN_BUDGET_PCT` | 0.10 (10%) |
| `LEVERAGE` | 100 (typical for forex/CFD) |

---

## 22. What This Enables Next

1. **Swap-aware scaling tests** — stress under swap drag + widened spread regimes
2. **Spread-shock stress regimes** — validate DD bounds under adverse conditions
3. **Live shadow execution** — confidence that PnL math is invariant

---

## 23. Change Log

| Version | Date | Changes |
|---------|------|---------|
| v1.0 | 2025-03-02 | Initial spec post-scaling fix |

---

## 24. References

- Empirical A/B lot sizing test results
- Brick calibration from live spread analysis
- Multi-unit execution model validation
- Dynamic exposure ceiling testing