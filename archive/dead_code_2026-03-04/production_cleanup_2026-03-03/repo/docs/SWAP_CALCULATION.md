# Swap (Rollover) Calculation

## Overview

Forex brokers charge swap (rollover) fees when positions are held overnight. The swap fee varies by broker, symbol, and day of the week.

## Triple Swap Day

**IMPORTANT**: Triple swap day is **NOT always Wednesday**. It varies by broker:

| Broker Type | Triple Swap Day | Rationale |
|-------------|----------------|-----------|
| Most MT5/MT4 brokers | **Wednesday** | Rollover to Thursday accounts for weekend (Sat + Sun + Wed) |
| Some brokers | **Friday** | Rollover to Monday accounts for weekend |
| Islamic accounts | **None** | No swap charged (swap-free) |
| Some crypto brokers | **Sunday** | Different market hours |

## Configuration

Set triple swap day in `SymbolSpec`:

```python
spec = SymbolSpec(
    symbol="EURUSD",
    swap_long=-3.5,
    swap_short=1.0,
    swap_triple_day="wednesday",  # Can be any day of the week
)
```

Supported values:
- `"monday"`, `"tuesday"`, `"wednesday"`, `"thursday"`, `"friday"`, `"saturday"`, `"sunday"`

## Swap Calculation Formula

### Basic Formula (Points Mode)

```python
swap_per_lot_per_day = swap_points * point * contract_size
total_swap = swap_per_lot_per_day * volume * days_held
```

### With Triple Swap

```python
# Count days between entry and exit
normal_days = days_held - triple_swap_count
triple_days = triple_swap_count  # Number of triple swap days crossed

total_swap = (
    swap_per_lot_per_day * volume * normal_days +
    swap_per_lot_per_day * volume * triple_days * 3  # 3x on triple day
)
```

## Example: Wednesday Triple Swap

```python
# Entry: Monday 00:00
# Exit: Friday 00:00 (4 days later)

# Swap charged:
# Monday -> Tuesday: 1x
# Tuesday -> Wednesday: 1x
# Wednesday -> Thursday: 3x  (triple swap day)
# Thursday -> Friday: 1x

# Total: 1 + 1 + 3 + 1 = 6 swap days over 4 calendar days
```

## Example: Friday Triple Swap

```python
# Entry: Monday 00:00
# Exit: Sunday 00:00 (6 days later)

# Swap charged:
# Monday -> Tuesday: 1x
# Tuesday -> Wednesday: 1x
# Wednesday -> Thursday: 1x
# Thursday -> Friday: 1x
# Friday -> Saturday: 3x  (triple swap day)
# Saturday -> Sunday: 1x

# Total: 1 + 1 + 1 + 1 + 3 + 1 = 8 swap days over 6 calendar days
```

## Per-Day Swap Multipliers (MT5 Full Specification)

MT5 supports per-day swap multipliers via `SYMBOL_SWAP_SUNDAY` through `SYMBOL_SWAP_SATURDAY`:

```cpp
// MT5 Example
SYMBOL_SWAP_MONDAY = 1.0      // Normal swap
SYMBOL_SWAP_TUESDAY = 1.0     // Normal swap
SYMBOL_SWAP_WEDNESDAY = 3.0   // Triple swap
SYMBOL_SWAP_THURSDAY = 1.0    // Normal swap
SYMBOL_SWAP_FRIDAY = 1.0      // Normal swap
SYMBOL_SWAP_SATURDAY = 0.0    // No swap (market closed)
SYMBOL_SWAP_SUNDAY = 0.0      // No swap (market closed)
```

**Current Kinetra Implementation**: Simplified model with single `swap_triple_day`

**Planned Enhancement**: Full per-day multiplier support

## Swap Types (ENUM_SYMBOL_SWAP_MODE)

| Mode | Description | Formula | Kinetra Status |
|------|-------------|---------|----------------|
| **Points** | Swap in price points | `swap_points * point * contract_size * days` | ✅ Implemented |
| **Currency (Symbol)** | Swap in base currency | `swap_value * days` | ⏳ Planned |
| **Currency (Margin)** | Swap in margin currency | `swap_value * days` | ⏳ Planned |
| **Currency (Deposit)** | Swap in account currency | `swap_value * days` | ⏳ Planned |
| **Interest (Current)** | Annual interest on current price | `(price * rate / 360) * days` | ⏳ Planned |
| **Interest (Open)** | Annual interest on open price | `(open_price * rate / 360) * days` | ⏳ Planned |
| **Reopen (Current)** | Reopen at current price | `(current - open ± swap_points) * contract_size` | ⏳ Planned |
| **Reopen (Bid)** | Reopen at bid price | `(bid - open ± swap_points) * contract_size` | ⏳ Planned |

## No Swap (Intraday Trading)

Positions opened and closed within the same day (< 24 hours) incur **NO swap charges**:

```python
if days_held < 1.0:
    return 0.0  # No swap for intraday
```

## Islamic (Swap-Free) Accounts

For Islamic/swap-free accounts, set swap rates to zero:

```python
spec = SymbolSpec(
    symbol="EURUSD",
    swap_long=0.0,      # No swap
    swap_short=0.0,     # No swap
    swap_triple_day="wednesday",  # Not used (swap is 0)
)
```

## Testing

Run swap calculation tests:

```bash
# Test default Wednesday triple swap
python3 scripts/test_friction_costs.py

# Test Friday triple swap
python3 tests/test_swap_friday.py
```

## Real-World Examples

### EURUSD (IC Markets - Wednesday Triple)
```python
SymbolSpec(
    symbol="EURUSD",
    swap_long=-3.5,      # -3.5 points per day
    swap_short=1.0,      # +1.0 points per day (credit)
    swap_triple_day="wednesday",
)
```

### GBPJPY (Pepperstone - Friday Triple)
```python
SymbolSpec(
    symbol="GBPJPY",
    swap_long=-8.2,
    swap_short=2.4,
    swap_triple_day="friday",  # Some brokers use Friday
)
```

### BTCUSD (Crypto - No Triple Swap)
```python
SymbolSpec(
    symbol="BTCUSD",
    swap_long=-0.05,     # Daily financing
    swap_short=-0.05,    # Both sides pay
    swap_triple_day="sunday",  # Varies by broker
)
```

## Implementation Details

### Code Location

- **Calculation**: `kinetra/realistic_backtester.py::calculate_swap()`
- **Configuration**: `kinetra/market_microstructure.py::SymbolSpec.swap_triple_day`
- **Tests**: `scripts/test_friction_costs.py::test_triple_swap()`

### Algorithm

```python
def calculate_swap(direction, volume, entry_time, exit_time):
    days_held = (exit_time - entry_time).total_seconds() / 86400

    if days_held < 1:
        return 0.0  # No swap for intraday

    # Get swap rate
    swap_points = spec.swap_long if direction == 1 else spec.swap_short
    swap_per_day = swap_points * spec.point * spec.contract_size

    # Map triple swap day to weekday number
    day_map = {'monday': 0, 'tuesday': 1, 'wednesday': 2, ...}
    triple_day = day_map.get(spec.swap_triple_day.lower(), 2)

    # Count triple swap days between entry and exit
    triple_count = 0
    current = entry_time.replace(hour=0, minute=0, second=0)
    end = exit_time.replace(hour=0, minute=0, second=0)

    while current < end:
        if current.weekday() == triple_day:
            triple_count += 1
        current += timedelta(days=1)

    # Calculate total
    normal_days = int(days_held) - triple_count
    total_swap = (
        swap_per_day * volume * normal_days +
        swap_per_day * volume * triple_count * 3
    )

    return total_swap
```

## Broker-Specific Configuration

Always check your broker's swap specifications:

1. **Triple swap day**: Check contract specifications
2. **Swap rates**: Updated regularly by broker
3. **Swap-free options**: Available for Islamic accounts
4. **Crypto swaps**: May differ from forex (daily financing vs rollover)

## References

- [MetaTrader 5 Documentation - Swap Calculation](https://www.mql5.com/en/docs/constants/environment_state/marketinfoconstants#enum_symbol_swap_mode)
- [IC Markets Swap Rates](https://www.icmarkets.com/global/en/trading-conditions/swap-rates)
- [Pepperstone Swap Schedule](https://www.pepperstone.com/en/trading-conditions/swap-rates)

## Multi-Asset Swap Configuration

### Different Triple Swap Days Per Asset Class

Brokers often use **different triple swap days** for different asset classes:

```python
# Forex: Wednesday triple swap
forex_spec = SymbolSpec(
    symbol="EURUSD",
    asset_class=AssetClass.FOREX,
    swap_long=-3.5,
    swap_short=1.0,
    swap_triple_day="wednesday",  # ✅ Wednesday for forex
)

# Indices: Friday triple swap
index_spec = SymbolSpec(
    symbol="US500",
    asset_class=AssetClass.INDEX,
    swap_long=-2.5,
    swap_short=-1.0,
    swap_triple_day="friday",  # ✅ Friday for indices
)

# Crypto: No triple swap (24/7 market)
crypto_spec = SymbolSpec(
    symbol="BTCUSD",
    asset_class=AssetClass.CRYPTO,
    swap_long=-0.05,
    swap_short=-0.05,
    swap_triple_day="sunday",  # ✅ Varies by broker
)

# Metals: Friday triple swap
metal_spec = SymbolSpec(
    symbol="XAUUSD",
    asset_class=AssetClass.METAL,
    swap_long=-5.0,
    swap_short=1.0,
    swap_triple_day="friday",  # ✅ Friday for metals
)
```

### Mixed Swap Types (Points vs Percentage)

Different symbols can use different swap calculation methods:

```python
# Forex: Swap in points
forex_spec = SymbolSpec(
    symbol="EURUSD",
    swap_long=-3.5,      # Points
    swap_short=1.0,      # Points
    swap_type="points",  # ✅ Points mode
)

# Indices: Swap in percentage
index_spec = SymbolSpec(
    symbol="US500",
    swap_long=-0.02,     # -2% annual
    swap_short=-0.01,    # -1% annual
    swap_type="percentage",  # ⚠️ Not yet implemented
)

# Crypto: Daily financing rate
crypto_spec = SymbolSpec(
    symbol="BTCUSD",
    swap_long=-0.05,     # -5% annual
    swap_short=-0.05,    # -5% annual
    swap_type="interest",  # ⚠️ Not yet implemented
)
```

**Current Status**: Only `swap_type="points"` is fully implemented. Other modes planned.

## Real-World Broker Examples

### IC Markets (Australia)

```python
# Forex - Wednesday triple swap, points mode
ic_markets_specs = {
    "EURUSD": SymbolSpec(
        symbol="EURUSD",
        swap_long=-3.5,
        swap_short=1.0,
        swap_type="points",
        swap_triple_day="wednesday",
    ),
    "GBPJPY": SymbolSpec(
        symbol="GBPJPY",
        swap_long=-8.2,
        swap_short=2.4,
        swap_type="points",
        swap_triple_day="wednesday",
    ),
}

# Indices - Friday triple swap
ic_markets_specs.update({
    "US500": SymbolSpec(
        symbol="US500",
        swap_long=-2.5,
        swap_short=-1.0,
        swap_type="points",
        swap_triple_day="friday",  # Different from forex!
    ),
})

# Metals - Friday triple swap
ic_markets_specs.update({
    "XAUUSD": SymbolSpec(
        symbol="XAUUSD",
        swap_long=-5.0,
        swap_short=1.0,
        swap_type="points",
        swap_triple_day="friday",  # Different from forex!
    ),
})
```

### Pepperstone (UK)

```python
# Forex - Wednesday triple swap
pepperstone_specs = {
    "EURUSD": SymbolSpec(
        symbol="EURUSD",
        swap_long=-3.2,
        swap_short=0.8,
        swap_type="points",
        swap_triple_day="wednesday",
    ),
}

# Indices - No triple swap on some symbols
pepperstone_specs.update({
    "GER40": SymbolSpec(
        symbol="GER40",
        swap_long=-0.02,  # Percentage
        swap_short=-0.01,
        swap_type="percentage",  # ⚠️ Planned
        swap_triple_day="friday",
    ),
})
```

### Islamic Accounts (Swap-Free)

```python
# All symbols have zero swap
islamic_spec = SymbolSpec(
    symbol="EURUSD",
    swap_long=0.0,      # No swap
    swap_short=0.0,     # No swap
    swap_type="points",
    swap_triple_day="wednesday",  # Not used
)
```

## Symbol-Specific Configuration

Since `SymbolSpec` is created **per symbol**, you have full flexibility:

```python
# Load different specs for different symbols
specs = {}

# Forex symbols (Wednesday triple)
for symbol in ["EURUSD", "GBPUSD", "USDJPY"]:
    specs[symbol] = load_forex_spec(symbol, triple_day="wednesday")

# Index CFDs (Friday triple)
for symbol in ["US500", "US30", "GER40"]:
    specs[symbol] = load_index_spec(symbol, triple_day="friday")

# Crypto (24/7, different swap logic)
for symbol in ["BTCUSD", "ETHUSD"]:
    specs[symbol] = load_crypto_spec(symbol, triple_day="sunday")

# Use the correct spec when backtesting
backtester = RealisticBacktester(
    spec=specs["EURUSD"],  # ✅ Use symbol-specific spec
    initial_capital=10000.0,
)
```

## Fetching Broker Specifications

### From MT5 API (Live)

```python
import MetaTrader5 as mt5

def fetch_swap_config(symbol: str) -> dict:
    """Fetch swap configuration from MT5."""
    # Select symbol
    if not mt5.symbol_select(symbol, True):
        raise ValueError(f"Symbol {symbol} not found")
    
    # Get swap info
    info = mt5.symbol_info(symbol)
    
    return {
        'swap_long': info.swap_long,
        'swap_short': info.swap_short,
        'swap_mode': info.swap_mode,  # ENUM_SYMBOL_SWAP_MODE
        'swap_triple_day': info.swap_rollover3days,  # 0=Sun, 1=Mon, ..., 6=Sat
    }

# Example usage
eurusd_swap = fetch_swap_config("EURUSD")
print(f"EURUSD swap long: {eurusd_swap['swap_long']}")
print(f"Triple swap day: {eurusd_swap['swap_triple_day']}")  # 3 = Wednesday
```

### From CSV/JSON Configuration

```python
# config/swap_specs.json
{
    "EURUSD": {
        "swap_long": -3.5,
        "swap_short": 1.0,
        "swap_type": "points",
        "swap_triple_day": "wednesday"
    },
    "US500": {
        "swap_long": -2.5,
        "swap_short": -1.0,
        "swap_type": "points",
        "swap_triple_day": "friday"
    }
}
```

```python
import json

def load_swap_specs(config_file: str) -> dict:
    """Load swap specifications from JSON."""
    with open(config_file, 'r') as f:
        return json.load(f)

# Usage
swap_specs = load_swap_specs("config/swap_specs.json")

spec = SymbolSpec(
    symbol="EURUSD",
    **swap_specs["EURUSD"],  # ✅ Load from config
)
```

## Best Practices

1. **Always verify with your broker**: Swap rates and triple swap days can change
2. **Check asset class**: Don't assume all symbols use the same triple swap day
3. **Update regularly**: Swap rates change based on interest rates
4. **Test before live**: Validate swap calculations match broker statements
5. **Use symbol-specific specs**: Don't share specs between different symbols

## Warning: Swap Rate Changes

⚠️ **Swap rates are NOT constant**:
- Central bank rate changes affect swap rates
- Brokers update swap rates regularly (weekly/monthly)
- Major economic events can cause sudden changes
- Always fetch current rates before backtesting recent data

```python
# Example: Fetch current rates before backtest
def update_swap_rates(spec: SymbolSpec) -> SymbolSpec:
    """Update swap rates from broker API."""
    current_rates = fetch_swap_config(spec.symbol)
    
    spec.swap_long = current_rates['swap_long']
    spec.swap_short = current_rates['swap_short']
    
    return spec

# Usage
spec = SymbolSpec(symbol="EURUSD", ...)
spec = update_swap_rates(spec)  # ✅ Get current rates
```

