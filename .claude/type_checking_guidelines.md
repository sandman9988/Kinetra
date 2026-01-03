# Type Checking Guidelines for Kinetra

> **⚠️ CANONICAL RULES:** All agent rules are consolidated in [`AGENT_RULES_MASTER.md`](../AGENT_RULES_MASTER.md)
> 
> This file provides **type checking specific guidelines**. For complete rules, see the master document.

---

## BasedPyRight Error Prevention

These guidelines help avoid common type checking errors in the Kinetra codebase.

### 1. `reportOptionalMemberAccess` - Check Optional types for None

**Problem**: Accessing members on `Optional[T]` types without checking for `None`.

```python
# BAD - Error: "attribute" is not a known attribute of "None"
result = self.data_quality_report.completeness_pct

# GOOD - Check for None first
if self.data_quality_report is not None:
    result = self.data_quality_report.completeness_pct
```

**Also applies to ternary expressions**:
```python
# BAD - Still triggers the error
pnl_str = f"${self.result.net_pnl:.2f}" if self.result else "N/A"

# GOOD - Explicit if/else block
if self.result is not None:
    pnl_str = f"${self.result.net_pnl:.2f}"
else:
    pnl_str = "N/A"
```

### 2. `reportInvalidTypeForm` - Proper type annotations

**Problem**: Using variables or conditional imports in type annotations.

```python
# BAD - Variable not allowed in type expression
calendar: Optional[MarketCalendar] = None  # When MarketCalendar is conditionally imported

# GOOD - Use TYPE_CHECKING block for type-only imports
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .market_calendar import MarketCalendar

# Then use string quotes for forward references
calendar: Optional["MarketCalendar"] = None
```

**For conditional imports where runtime value may be None**:
```python
# BAD
try:
    from .module import SomeClass
except ImportError:
    SomeClass = None  # Now SomeClass is None at runtime

def func(param: Optional[SomeClass]):  # Error: SomeClass is None
    ...

# GOOD - Use TYPE_CHECKING and Any fallback
from typing import TYPE_CHECKING, Any
if TYPE_CHECKING:
    from .module import SomeClass
else:
    SomeClass = Any  # Fallback for runtime
```

### 3. `reportPossiblyUnboundVariable` - Ensure variables are bound

**Problem**: Variables may not be bound in all code paths.

```python
# BAD - get_calendar_for_symbol may not be defined
try:
    from .market_calendar import get_calendar_for_symbol
    AVAILABLE = True
except ImportError:
    AVAILABLE = False

if AVAILABLE:
    calendar = get_calendar_for_symbol(symbol)  # Possibly unbound

# GOOD - Provide a default or check the import result
from typing import Callable, Optional

_get_calendar: Optional[Callable[..., Any]] = None
try:
    from .market_calendar import get_calendar_for_symbol as _get_calendar_impl
    _get_calendar = _get_calendar_impl
except ImportError:
    pass

if _get_calendar is not None:
    calendar = _get_calendar(symbol)
```

### 4. `reportArgumentType` - Match argument types exactly

**Problem**: Passing incompatible types to function parameters.

```python
# BAD - pd.Index is not datetime
bar_time = data.index[i]  # Type: Index | datetime
self._close_position(..., exit_time=bar_time)  # Expects datetime

# GOOD - Explicit conversion
bar_time = pd.to_datetime(data.index[i])  # Now definitely datetime
self._close_position(..., exit_time=bar_time)
```

### 5. Generic Types - Always specify type parameters

**Problem**: Using bare `List`, `Dict`, `Callable` without type arguments.

```python
# BAD - Missing type arguments
from typing import List, Dict
trades: List = []
config: Dict = {}

# GOOD - Specify type parameters
from typing import List, Dict, Any
trades: List[Trade] = []
config: Dict[str, Any] = {}
```

### 6. `from __future__ import annotations` - Defer annotation evaluation

When using many forward references, add this at the top of the file:

```python
from __future__ import annotations

# Now all annotations are strings by default
# No need to quote forward references
class MyClass:
    def method(self) -> MyClass:  # Works without quotes
        ...
```

### 7. Pandas Type Stubs

Many pandas-related "Unknown" types come from missing type stubs.

**Solution**: Install pandas-stubs:
```bash
pip install pandas-stubs
```

Or use explicit type annotations:
```python
# Explicit typing for pandas operations
row: pd.Series = data.iloc[i]
close_price: float = float(row["close"])
```

### 8. Unannotated Class Attributes

**Problem**: Class attributes without type annotations in non-final classes.

```python
# BAD
class MyEngine:
    def __init__(self):
        self.trades = []  # No type annotation

# GOOD - Add type annotation
class MyEngine:
    trades: List[Trade]
    
    def __init__(self):
        self.trades = []
```

Or use dataclass for automatic annotations:
```python
from dataclasses import dataclass

@dataclass
class MyEngine:
    trades: List[Trade] = field(default_factory=list)
```

---

## Quick Checklist Before Committing

1. [ ] All `Optional[T]` types checked for `None` before member access
2. [ ] All generic types have type parameters (`List[T]`, `Dict[K, V]`)
3. [ ] Conditional imports handled with `TYPE_CHECKING` block
4. [ ] Variables initialized in all code paths
5. [ ] Pandas operations have explicit type annotations where needed
6. [ ] Class attributes have type annotations

## Suppressing Type Errors

When a type error is unavoidable (e.g., pandas stubs limitations), use pyright-specific ignore comments:

```python
# Pyright-specific ignore (preferred for basedpyright)
value: pd.Timestamp = data.index[i]  # pyright: ignore[reportAssignmentType]

# Standard type ignore (works but less specific)
value = some_call()  # type: ignore[assignment]
```

Common ignore rules:
- `reportAssignmentType` - Type assignment issues
- `reportArgumentType` - Argument type mismatches
- `reportOptionalMemberAccess` - Accessing Optional without None check
- `reportUnknownMemberType` - Unknown member types (often pandas)

## Running Type Checks

```bash
# Check specific files
basedpyright kinetra/backtest_engine.py

# Check with JSON output for parsing
basedpyright --outputjson kinetra/ 2>/dev/null | python3 -c "..."

# Check with ruff for unused imports/variables
ruff check kinetra/backtest_engine.py --select=F401,F841
```

---

## Additional Resources

- **Complete Rules**: [`AGENT_RULES_MASTER.md`](../AGENT_RULES_MASTER.md) - **START HERE**
- **GitHub Copilot Quick Reference**: [`.github/copilot-instructions.md`](../.github/copilot-instructions.md)
- **Claude/Zed Quick Reference**: [`.claude/instructions.md`](instructions.md)

---

**For complete, comprehensive rules → See [`AGENT_RULES_MASTER.md`](../AGENT_RULES_MASTER.md)**
