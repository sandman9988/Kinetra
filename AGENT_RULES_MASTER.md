# KINETRA AGENT RULES - MASTER REFERENCE
**Version:** 2.0  
**Last Updated:** 2024-01-09  
**Status:** Canonical - Single Source of Truth

---

## 📋 TABLE OF CONTENTS

1. [Meta-Rules (Conversation & Compression)](#1-meta-rules)
2. [Core Philosophy (First Principles)](#2-core-philosophy)
3. [Data Safety & Integrity](#3-data-safety--integrity)
4. [Performance - Vectorization & Optimization](#4-performance---vectorization--optimization)
5. [Memory & Efficiency](#5-memory--efficiency)
6. [I/O & Concurrency](#6-io--concurrency)
7. [Determinism & Reproducibility](#7-determinism--reproducibility)
8. [Backtesting Engine](#8-backtesting-engine)
9. [Experiment Safety & Validation](#9-experiment-safety--validation)
10. [MetaAPI Connector](#10-metaapi-connector)
11. [Logging & Error Handling](#11-logging--error-handling)
12. [Security & Hard Prohibitions](#12-security--hard-prohibitions)
13. [Code Quality & Style](#13-code-quality--style)
14. [Type Checking & Documentation](#14-type-checking--documentation)
15. [Physics-First Approach](#15-physics-first-approach)
16. [Testing Requirements](#16-testing-requirements)
17. [Deliverables & Validation](#17-deliverables--validation)

---

## 1. META-RULES

### 1.1 Conversation Continuity

**ALWAYS:**
- ✅ Continue as if nothing was lost
- ✅ Never re-ask settled questions
- ✅ Never contradict earlier constraints
- ✅ Never re-introduce rejected ideas
- ✅ Maintain naming, terminology, and assumptions consistently

### 1.2 Compression Rules

**Compress automatically when:**
- Context window pressure risks losing architecture/rules/constraints
- Long background explanations are complete and no longer evolving
- Repeated clarifications or restatements appear
- Execution moves from design → implementation
- **Do NOT wait for user approval**

**When Compressing, Replace Prior Turns With:**
```
Context Snapshot:
├── Facts & Constraints
├── Active Objectives
├── Active Constraints
├── Open Questions
└── Do-Not-Break List
```

**Compression MUST Retain:**
- System goals and non-negotiables
- Architectural boundaries (layers, responsibilities)
- Explicit prohibitions and safety rules
- Definitions of terms with special meaning
- Decisions already made (even if provisional)
- Open questions still unresolved

**Aggressively Remove:**
- Explanatory prose once intent is clear
- Repeated rationale or philosophy
- Historical step-by-step narration
- Redundant examples
- Large illustrative but non-binding text

**Replace with canonical summaries, NOT paraphrases**

**If Compression Would Cause Ambiguity:**
- Preserve both paths as explicit alternatives
- Mark the fork clearly
- Do NOT silently choose

**Do NOT announce compression** unless it affects user-visible behavior

---

## 2. CORE PHILOSOPHY

### 2.1 First Principles, Zero Assumptions

**THE ONLY ASSUMPTION:** Physics is real (energy, friction, entropy exist in markets)

**NEVER:**
- ❌ Use magic numbers (20-period MA, 14-period RSI, 2% stops)
- ❌ Use traditional TA indicators (ATR, BB, RSI, MACD, ADX) without physics justification
- ❌ Assume linearity without proof (no Pearson correlation, linear regression unless approved)
- ❌ Use fixed thresholds ("if energy > 0.8")
- ❌ Use static values ("volume spike > 1.5x")
- ❌ Apply universal rules across markets without exploration
- ❌ Use time-based or calendar-based filters
- ❌ Remove or modify working code without strong justification
- ❌ Use placeholders (no TODOs, stubs, "assume...", or partial implementations)

**ALWAYS:**
- ✅ Start from thermodynamic/physical first principles
- ✅ Use rolling, adaptive distributions (NO fixed periods)
- ✅ Validate per-market, per-regime, per-timeframe
- ✅ **EXPLORE before implementing** - let data decide
- ✅ Question everything (even these rules!)
- ✅ Convert metrics to percentiles (adaptive to current distribution)
- ✅ **Asymmetric by default** (up/down separated, NEVER combined)
- ✅ Let RL discover patterns (provide features, not rules)

### 2.2 No TA Indicators - Physics Only

**What We DON'T Use:**
- NO traditional TA indicators (ATR, Bollinger Bands, RSI, MACD, Stochastic, ADX, etc.)
- NO hardcoded thresholds
- NO static values
- NO time-based filters
- NO rules - only features for RL to discover patterns

**What We DO Use - Physics-Based Primitives:**

| Concept | Formula | Physical Analogue |
|---------|---------|-------------------|
| **Kinetic Energy** | E = ½mv² = ½ × velocity² | Energy in motion |
| **Damping** | ζ = σ(v) / μ(\|v\|) | Energy dissipation / friction |
| **Reynolds Number** | Re = (v × L × ρ) / μ | Laminar vs turbulent flow |
| **Viscosity** | μ = resistance / flow | Market friction |
| **Volume Loss** | ∫(vol_noise²) dt | Entropy production |
| **Energy Dissipation** | ∫(price_jerk²) dt | Predicts exhaustion/reversal |
| **Liquidity Loss** | spread × vol⁻¹ | Slippage trap |
| **Spring Stiffness** | k = volume / Δprice | Resistance to displacement |
| **Phase Space** | (position, momentum) | State confinement |
| **Entropy** | Shannon entropy | Predictability measure |

### 2.3 Adaptive Percentiles (Not Static Thresholds)

**Every metric MUST be converted to its position in rolling distribution:**

```python
# ✅ CORRECT: Adaptive percentile
feature_pct = feature.rolling(window).apply(
    lambda x: (x.iloc[-1] > x.iloc[:-1]).mean()
)
# Returns 0-1: where does current value sit in recent history?

# ❌ WRONG: Static threshold
if feature > 0.8:  # NOT adaptive!
```

### 2.4 DSP-Driven Adaptive Windows

**NO hardcoded window sizes** (no "20 bars", no "500 bars")

**Use Digital Signal Processing to find natural cycles:**

```python
# FFT decomposes signal into frequency components
fft_vals = np.fft.fft(detrended_velocity)
power = np.abs(fft_vals) ** 2

# Find dominant periods (natural cycles in THIS instrument's data)
short_period = first_dominant_period   # Fast cycle
long_period = second_dominant_period   # Slow cycle

# Use these as windows - derived from data, NOT hardcoded
lookback = short_period
window = long_period
```

**Why:** Gold has different cycles than BTC than EURUSD. Summer has different cycles than winter.

### 2.5 Adaptive Volatility Estimators (Market-Specific)

**DO NOT use simple std dev!** Use market-appropriate estimators:

- **Indices/Forex:** Yang-Zhang (accounts for gaps)
- **Commodities:** Rogers-Satchell (drift-independent)
- **Crypto:** Realized Variance (high-frequency)
- **ALL with adaptive lookbacks** (via DSP cycle detection, NO fixed periods)

### 2.6 Regime-Aware Filtering

**NEVER take a trade without 3-regime confluence:**

**1. Physics Regime (from physics_engine.py):**
- **LAMINAR:** Smooth trend, low jerk → **TRADE**
- **UNDERDAMPED:** Momentum with oscillation → **TRADE** (wider stops)
- **OVERDAMPED:** Mean-reverting → **BLOCK** trend following
- **CHAOTIC:** High jerk, unpredictable → **BLOCK ALL**

**2. Volatility Regime (adaptive vol estimator):**
- **LOW:** Mean-reversion dominant
- **NORMAL:** Trend-following viable
- **HIGH:** Chaos, avoid trading

**3. Momentum Regime (from ROC or dsp_trend_dir):**
- **WEAK:** Chop, avoid
- **MODERATE:** Pullback/swing trades
- **STRONG:** Breakout trades

**Trading Signal = Specialist Signal AND (Physics ∈ {LAMINAR, UNDERDAMPED}) AND (Vol ≠ HIGH) AND (Momentum ≠ WEAK)**

### 2.7 Omega Reward (Pythagorean Path Efficiency)

**Reward function with NO static coefficients:**

```python
# Goal: Maximum displacement via shortest path
# Total excursion = Pythagorean distance in MFE/MAE space
total_excursion = sqrt(MFE² + MAE²)

# Path efficiency = how direct was the path?
path_efficiency = |PnL| / total_excursion

# Omega = signed reward (direction matters)
omega = PnL × path_efficiency
```

**Why this works:**
- Clean move (MFE only): high omega (large PnL, small excursion)
- Whipsaw (high MFE + high MAE): low omega (excursion dominates)
- Loss with excursion: negative omega (penalty)

**NO static weights, NO arbitrary coefficients - pure geometry**

### 2.8 Meta-Assumption: We Don't Know How to Specialize

**Before asking:** "Should crypto specialists use different stops than forex?"

**Ask first:**
- Should we even specialize by asset class?
- Maybe specialize by regime (LAMINAR vs CHAOTIC)?
- Maybe specialize by timeframe (M15 vs H4)?
- Maybe specialize by volatility regime?
- Maybe ONE universal agent is optimal?

**Let the data tell us!**

### 2.9 Temporal Non-Stationarity

**Even IF exploration discovers that:**
- Asset class specialists work best (today)
- LAMINAR regime is tradeable (today)
- Energy-based stops are optimal (today)

**These findings can CHANGE as markets evolve:**
- Crypto correlation with indices shifts
- Central bank policy changes forex dynamics
- Algorithmic trading changes intraday patterns
- Crisis regimes invalidate normal-regime rules

**THEREFORE:**
- Continuous re-exploration (weekly/monthly)
- Doppelgänger system detects drift (Shadow A vs Live)
- Health scoring triggers re-training
- Never assume today's optimal = tomorrow's optimal

---

## 3. DATA SAFETY & INTEGRITY

### 3.1 Data Safety (#1 Priority - NEVER LOSE USER DATA)

**User data (especially `data/master/` CSVs) is IRREPLACEABLE**
- Downloads take hours
- Losing them is **UNACCEPTABLE**

**Mandatory Before ANY Data Operation:**

1. ✅ **ALWAYS use `PersistenceManager.atomic_save()`** - Never raw file writes
2. ✅ **ALWAYS backup before git operations** - `git rm --cached` can delete files
3. ✅ **CHECK `.gitignore`** before commits - Large files must NEVER be tracked
4. ✅ **NEVER assume backups exist** - Verify before dangerous operations

**Atomic Save Pattern:**

```python
from kinetra.persistence_manager import get_persistence_manager

pm = get_persistence_manager(backup_dir="data/backups", max_backups=10)

# Atomic save: backup → temp write → atomic rename → auto-recovery
pm.atomic_save(
    filepath="data/master/BTCUSD_H1.csv",
    content=df,
    writer=lambda path, data: data.to_csv(path, index=False)
)

# Restore if needed
pm.restore_latest("data/master/BTCUSD_H1.csv")
```

**How Atomic Save Works:**
1. Create timestamped backup of existing file (if exists)
2. Write to temporary file in same directory
3. Atomic rename (OS guarantees either full success or full failure)
4. Automatic recovery on failure

**Git Safety Rules:**

```bash
# DANGEROUS - Backup first!
git rm --cached data/master/*.csv
git clean -fd
git pull
git reset --hard

# SAFE - Always backup first
python scripts/backup_data.py
git pull
```

**Branch Management:**
- `main` branch is production-ready (protected)
- Feature branches are short-lived (merge within 1-2 weeks)
- Always create PRs for changes to `main`
- Use descriptive branch names: `feature/`, `fix/`, `refactor/`, `docs/`
- Clean up merged branches regularly

**.gitignore Critical Patterns:**

```gitignore
# Large data files (NEVER commit)
data/master/
data/prepared/
data/test/
data/backups/

# Allow only gitkeep
!data/.gitkeep
```

**Recovery Procedures:**

If data was accidentally deleted:
1. Check git stash: `git stash list; git stash pop`
2. Restore from backups: `python scripts/backup_data.py --restore`
3. Check container vs local machine (files not synced by default)
4. Last resort: Re-download (slow, avoid at all costs)

### 3.2 Data Validation (Reject Invalid Data)

**Reject data if ANY of these fail:**

```python
# Non-monotonic timestamps
assert data.index.is_monotonic_increasing, "REJECT: Non-monotonic timestamps"

# Duplicated bars
assert ~data.index.duplicated().any(), "REJECT: Duplicated bars"

# Invalid OHLC
assert (data['high'] >= data['low']).all(), "REJECT: high < low"
assert data['open'].between(data['low'], data['high']).all(), "REJECT: open outside [low, high]"
assert data['close'].between(data['low'], data['high']).all(), "REJECT: close outside [low, high]"

# Impossible volume
assert (data['volume'] >= 0).all(), "REJECT: Negative volume"
assert data['volume'].notna().all(), "REJECT: Non-numeric volume"
```

### 3.3 Market-Type Aware Rules

```python
# Forex: Remove weekends, enforce session continuity
if market_type == "forex":
    data = remove_weekends(data)
    validate_session_continuity(data)

# Crypto: Enforce 24/7 continuity (NO weekend removal)
if market_type == "crypto":
    validate_24_7_continuity(data)
```

### 3.4 Quality Report (MANDATORY)

**Every dataset MUST emit a quality_report:**

```python
quality_report = {
    "bar_count": int,
    "missing_count": int,
    "gap_stats": {
        "max_gap_hours": float,
        "total_gaps": int,
        "gap_distribution": {...}
    },
    "outlier_stats": {
        "extreme_moves": int,
        "z_scores_above_3": int
    },
    "integrity_flags": [
        "weekend_data_present",  # For crypto
        "gaps_detected",
        "outliers_detected"
    ]
}
```

**All transformations MUST be audited:**
- What changed
- How many rows
- Why (reason/logic)

### 3.5 Standardized Data Output

**Data prep MUST output a standardized object:**

```python
data_package = {
    "prices": OHLCV + timestamp,
    "symbol_spec": {  # Real MT5/MetaAPI specs
        "spread": float,
        "commission": float,
        "swap_long": float,
        "swap_short": float,
        "min_lot": float,
        "lot_step": float,
        "stop_level": int,
        "margin_rate": float
    },
    "market_type": AssetClass enum,  # Auto-detected
    "quality_report": {...},
    "feature_matrix": (if produced) + feature_registry_metadata
}
```

**Feature engineering MUST be asymmetric by default** (up/down separated, NEVER combined)

---

## 4. PERFORMANCE - VECTORIZATION & OPTIMIZATION

### 4.1 Explicit Python Loops = Last Resort

**PREFER (in order):**

**1. NumPy Vectorized Operations** ✅
```python
# ✅ CORRECT
energy = 0.5 * velocity ** 2

# ❌ WRONG
energy = np.empty_like(velocity)
for i in range(len(velocity)):
    energy[i] = 0.5 * velocity[i] ** 2
```

**2. Pandas Column Operations** ✅
```python
# ✅ CORRECT
df['energy_pct'] = df['energy'].rolling(window).rank(pct=True)

# ❌ WRONG
for i in range(len(df)):
    df.loc[i, 'energy_pct'] = ...
```

**3. Broadcasting** ✅
```python
# ✅ CORRECT
result = arr_2d + arr_1d[:, np.newaxis]

# ❌ WRONG
for i in range(arr_2d.shape[0]):
    for j in range(arr_2d.shape[1]):
        result[i, j] = arr_2d[i, j] + arr_1d[i]
```

**4. Built-in Functions** ✅
```python
# Prefer: sum, min, max, map, filter, zip over manual loops
total = sum(values)  # NOT: total = 0; for v in values: total += v
```

**5. Libraries with C/C++ Backends** ✅
- NumPy, Pandas, PyTorch, SciPy
- **Reimplementing optimized primitives is PROHIBITED**

**If Looping Is Unavoidable:**
- Keep it tight and local
- Cache attribute lookups to locals
- Inline trivial functions if profiling shows overhead

### 4.2 Algorithmic Improvements > Micro-Optimizations

**Choose Optimal Data Structures:**

```python
# ✅ O(1) lookup
fast_lookup = {key: value}  # dict
unique_items = {item}       # set

# ❌ O(n) scan - Avoid on large lists
if item in large_list:  # Linear scan - BAD for hot paths
```

**Prefer algorithmic improvements over micro-optimizations**

**Do NOT trade clarity for speed unless complexity demands it**

### 4.3 Never Optimize Blindly

**Every optimization MUST include:**

```python
# 1. Baseline timing
import time
start = time.perf_counter()
result_before = old_implementation()
baseline_time = time.perf_counter() - start

# 2. Post-change timing
start = time.perf_counter()
result_after = new_implementation()
optimized_time = time.perf_counter() - start

# 3. Workload description
print(f"Dataset: {len(data)} rows, {n_features} features")
print(f"Speedup: {baseline_time / optimized_time:.2f}x")

# 4. Verification
assert np.allclose(result_before, result_after), "Results differ!"
```

**Performance claims without benchmarks are INVALID**

**Always Profile Before and After:**
- Use `cProfile`, `line_profiler`, or equivalent
- Optimization without evidence is invalid
- Never optimize blindly

### 4.4 JIT / Native Acceleration Escalation Path

**Escalate ONLY if profiling proves Python is the bottleneck:**

```python
# Level 1: Vectorization (ALWAYS TRY FIRST)
energy = 0.5 * velocity ** 2

# Level 2: Numba JIT (if vectorization insufficient)
from numba import jit

@jit(nopython=True)
def compute_energy(velocity):
    energy = np.empty_like(velocity)
    for i in range(len(velocity)):
        energy[i] = 0.5 * velocity[i] ** 2
    return energy

# Level 3: Cython (static typing, if Numba insufficient)
# cython: boundscheck=False, wraparound=False
cdef double[:] compute_energy(double[:] velocity):
    ...

# Level 4: GPU (CUDA / PyTorch, if still bottleneck)
energy = 0.5 * torch.tensor(velocity, device='cuda') ** 2

# Level 5: Distributed (Dask / Spark, if data-parallel bottleneck)
import dask.array as da
energy = 0.5 * da.from_array(velocity) ** 2
```

**Skipping levels requires justification:**
- Document why earlier levels insufficient
- Provide profiling evidence
- Justify complexity cost

### 4.5 Minimize Python Overhead in Hot Paths

**Reduce:**

```python
# ❌ BAD: Repeated attribute lookups
for i in range(len(data)):
    result = self.config.params.threshold * data[i]  # 3 lookups per iteration

# ✅ GOOD: Cache to local variable
threshold = self.config.params.threshold  # Cache once
for i in range(len(data)):
    result = threshold * data[i]  # Direct access

# ❌ BAD: Global variable access in loop
for i in range(n):
    x = GLOBAL_CONSTANT * values[i]

# ✅ GOOD: Cache global to local
constant = GLOBAL_CONSTANT
for i in range(n):
    x = constant * values[i]
```

**In Hot Paths, MINIMIZE:**
- Function calls
- Attribute lookups
- Global variable access

**Globals in hot paths are FORBIDDEN**

**Inline Trivial Functions (When Profiling Shows Overhead):**

```python
# ❌ BAD: Function call overhead in hot path
def square(x):
    return x * x

for val in large_array:
    result = square(val)  # Function call overhead

# ✅ GOOD: Inline when profiling shows this matters
for val in large_array:
    result = val * val  # Direct computation
```

---

## 5. MEMORY & EFFICIENCY

### 5.1 Lazy Evaluation by Default

**Use generators and iterators where full materialization is unnecessary**

```python
# ✅ GOOD: Generators (lazy)
def load_data_generator(files):
    for file in files:
        yield pd.read_csv(file)

for df in load_data_generator(file_list):
    process(df)  # Stream, don't materialize all

# ❌ BAD: Eager materialization (unless required)
all_data = [pd.read_csv(f) for f in files]  # Loads ALL into memory

# ✅ GOOD: Iterators where full materialization unnecessary
features = (compute_features(row) for row in data.itertuples())

# ❌ BAD: Eager when streaming possible
features = [compute_features(row) for row in data.itertuples()]
```

**Stream data; do not load entire datasets unless required**

**Eager evaluation MUST be justified:**

```python
# Justify when you MUST materialize:
# Reason: Need random access for Monte Carlo sampling
all_episodes = list(episode_generator())  # Justified: random access needed
```

### 5.2 Memory Discipline

**Avoid Unnecessary Allocations:**

```python
# ✅ GOOD: In-place operations
array *= 2

# ❌ BAD: Creates new array
array = array * 2

# ✅ GOOD: Preallocation
result = np.empty(shape, dtype=float)

# ✅ GOOD: Reuse buffers
buffer = np.empty(1000)
for batch in batches:
    buffer[:len(batch)] = batch
    process(buffer[:len(batch)])
```

**Prefer:**
- In-place operations
- Preallocation
- Reuse of buffers
- Generator expressions over materialized lists when possible

**Do NOT build large intermediate objects unless required**

**Monitor Memory:**

```python
import gc

# Delete large objects explicitly
del large_df
gc.collect()

# Memory growth: <50 MB per iteration
```

### 5.3 Caching Rules

**Cache ONLY:**
- Pure functions
- Deterministic outputs

**Use bounded caches:**

```python
from functools import lru_cache

@lru_cache(maxsize=128)  # Bounded cache
def expensive_pure_function(x):
    return complex_computation(x)
```

**NEVER cache:**
- I/O-dependent functions
- Stateful functions
- Non-deterministic functions

---

## 6. I/O & CONCURRENCY

### 6.1 Batch I/O Operations

```python
# ✅ GOOD: Batch I/O
with open(file, 'w') as f:
    f.writelines(all_lines)  # Single write

# ❌ BAD: I/O in tight loop
for line in lines:
    with open(file, 'a') as f:
        f.write(line)  # Multiple writes - SLOW
```

**Rules:**
- Avoid I/O in tight loops
- Buffer writes and reads
- Async or bulk APIs preferred

### 6.2 Concurrency

**Choose Correctly:**

```python
# CPU-bound: multiprocessing (NOT threading - GIL prevents parallelism)
from multiprocessing import Pool
with Pool(processes=4) as pool:
    results = pool.map(cpu_intensive_func, data)

# I/O-bound: async or threading
import asyncio
async def fetch_data(): ...
```

**Rules:**
- **Never assume parallelism improves performance—MEASURE**
- Shared state must be minimized or eliminated
- No locks in hot paths unless profiling proves necessary

---

## 7. DETERMINISM & REPRODUCIBILITY

### 7.1 Determinism First

**Identical inputs MUST produce identical outputs:**

```python
# Seed all randomness
import numpy as np
import torch
import random

np.random.seed(42)
torch.manual_seed(42)
random.seed(42)

# Fix execution order where relevant
data = data.sort_index()  # Deterministic ordering

# Pin versions in requirements
# numpy==1.24.0 (not numpy>=1.24.0)
```

**Performance improvements MUST NOT change semantics unless explicitly intended**

### 7.2 No Silent Failure

**Every exception is either:**
- Handled with a defined outcome, OR
- Causes a hard fail with context

```python
# ✅ GOOD: Defined outcome
try:
    result = risky_operation()
except SpecificError as e:
    log.warning("operation_failed", error=str(e))
    result = fallback_value  # Defined outcome

# ✅ GOOD: Hard fail with context
try:
    result = critical_operation()
except Exception as e:
    log.error("critical_failure", error=str(e), context={...})
    raise  # Hard fail

# ❌ BAD: Silent failure
try:
    result = operation()
except:
    pass  # NEVER do this
```

---

## 8. BACKTESTING ENGINE

### 8.1 Backtest Requirements

**Backtests MUST be:**

**1. Reproducible**
```python
# Seed, deterministic ordering
np.random.seed(42)
torch.manual_seed(42)
data = data.sort_index()  # Deterministic order
```

**2. Separated into Train/Valid/Test with Explicit Dates**
```python
# EXPLICIT dates, NO overlap
train_data = data["2020-01-01":"2021-12-31"]
valid_data = data["2022-01-01":"2022-06-30"]
test_data = data["2022-07-01":"2023-12-31"]

# Log split dates
log.info("data_split",
    train_start="2020-01-01", train_end="2021-12-31",
    valid_start="2022-01-01", valid_end="2022-06-30",
    test_start="2022-07-01", test_end="2023-12-31"
)
```

**3. Free of Lookahead Bias**
```python
# Strict causal feature computation
# Feature at time t can ONLY use data from t-1 and earlier
for i in range(lookback, len(data)):
    features[i] = compute_features(data[:i])  # Only past data
    # NEVER use data[i:] or data[i+1:]
```

### 8.2 Execution Model (MUST Be Explicit)

```python
execution_model = {
    "fills": "market" | "limit" | "VWAP",
    "slippage": slippage_model(volatility, liquidity),
    "spread": symbol_spec.spread,  # Real MT5 data
    "commission": symbol_spec.commission,  # Real MT5 data
    "swaps": {
        "long": symbol_spec.swap_long,
        "short": symbol_spec.swap_short
    },
    "broker_constraints": {
        "min_lot": symbol_spec.min_lot,
        "lot_step": symbol_spec.lot_step,
        "stop_level": symbol_spec.stop_level,
        "margin_required": symbol_spec.margin_rate
    }
}
```

**Use REAL MT5 specs from `instrument_specs.json`:**
- BTCUSD: `swap_long=-18% annual`
- EURUSD: `swap_long=-12.16 points`
- XAUUSD: `spread=0.35 points`

### 8.3 Metrics (MUST Include Minimum)

**Per-Trade Metrics:**
- Return series
- Drawdowns (running, max)
- Sharpe ratio
- **Omega ratio** (target > 2.7)
- Calmar ratio
- Profit Factor (PF)
- Win rate
- Exposure time
- % MFE Captured (target > 60%)
- % Energy Captured (target > 65%)

**Per-Instrument + Portfolio Aggregates:**

```python
metrics = {
    "BTCUSD": {
        "sharpe": 1.8,
        "omega": 2.9,
        "max_dd": -0.15,
        ...
    },
    "EURUSD": {...},
    "XAUUSD": {...},
    "portfolio": {
        "total_return": 0.45,
        "sharpe": 2.1,
        "omega": 3.2,
        "max_drawdown": -0.12,
        "correlation_matrix": [[1, 0.3, 0.1], ...],
        "composite_health_score": 0.92
    }
}
```

### 8.4 Optimization (MUST Include)

**1. Multiple Testing Correction (When Comparing Many Configs):**

```python
# Bonferroni correction
alpha_corrected = 0.01 / n_comparisons

# Or FDR (False Discovery Rate)
from statsmodels.stats.multitest import fdrcorrection
reject, p_adjusted = fdrcorrection(p_values, alpha=0.01)
```

**2. Effect Sizes (NOT Just P-Values):**

```python
# Cohen's d
def cohens_d(group1, group2):
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2))
    return (np.mean(group1) - np.mean(group2)) / pooled_std

effect_size = cohens_d(strategy_returns, baseline_returns)
assert effect_size > 0.5, "Effect size too small (< medium)"
```

### 8.5 Reconstructability

**Every backtest MUST be reconstructable from saved artifacts alone:**

```python
# Save everything needed to reproduce
artifacts = {
    "config": full_config_snapshot,
    "results": results_json,
    "plots": plot_files,
    "logs": log_files,
    "metadata": {
        "dataset_hash": hash_of_data,
        "code_version_hash": git_commit_sha,
        "timestamp": datetime.now().isoformat(),
        "seed": 42
    }
}
```

**Never overwrite results:**
- Use atomic writes
- Use versioned directories (`results/run_001/`, `results/run_002/`)

---

## 9. EXPERIMENT SAFETY & VALIDATION

### 9.1 No "Wins" Accepted Unless

**All THREE required:**

1. ✅ **Out-of-sample performance reported**
   ```python
   # Report BOTH in-sample and out-of-sample
   results = {
       "train": {"sharpe": 2.1, "omega": 2.9, ...},
       "valid": {"sharpe": 1.8, "omega": 2.5, ...},
       "test": {"sharpe": 1.7, "omega": 2.4, ...}  # Out-of-sample
   }
   ```

2. ✅ **Robustness checks run**
   - Bootstrap / Monte Carlo (100+ runs minimum)
   - Regime slices (LAMINAR vs CHAOTIC vs OVERDAMPED)
   - Instrument slices (crypto vs forex vs metals)

3. ✅ **Results include confidence intervals** (where applicable)
   ```python
   # Bootstrap confidence interval
   sharpe_ci = bootstrap_ci(returns, metric=sharpe_ratio, n_boot=1000, alpha=0.05)
   print(f"Sharpe: {sharpe:.2f} (95% CI: [{sharpe_ci[0]:.2f}, {sharpe_ci[1]:.2f}])")
   ```

### 9.2 If Result Looks Too Good

**Run Leakage Checks:**

```python
# 1. Check for data leakage
check_feature_leakage(features, labels, split_dates)
check_label_leakage(train, test)
check_split_logic(train_idx, test_idx)

# 2. Check for timestamp leaks
assert train.index.max() < test.index.min(), "Temporal leakage!"

# 3. Check for feature computation leaks
# Features at time t should ONLY use data up to t-1
```

**Run Shuffle Test Baselines:**

```python
# Randomize labels/returns - should destroy performance
shuffled_labels = np.random.permutation(labels)
baseline_performance = train_on_shuffled(shuffled_labels)

# If performance still "good" → LEAKAGE DETECTED
assert actual_sharpe >> baseline_sharpe, "Shuffle test failed - leakage suspected"
```

### 9.3 Explorer Requirements

**The explorer MUST:**

- Store every run with full config snapshot
- Store artifacts: results JSON, plots (if any), logs
- Tag runs with dataset hashes + code version hash
- **Never overwrite results** - use atomic writes + versioned directories

```python
# Run directory structure
results/
├── run_001_20240109_143022/
│   ├── config.json
│   ├── results.json
│   ├── plots/
│   ├── logs/
│   └── metadata.json (dataset_hash, code_hash, timestamp)
├── run_002_20240109_150315/
│   └── ...
```

---

## 10. METAAPI CONNECTOR

### 10.1 Treat Connectivity as UNRELIABLE by Default

**Failure Modes:**
- Disconnects
- Reconnects
- Stale sockets
- Partial responses

### 10.2 MUST Implement

**1. Bounded Retries with Backoff + Jitter:**

```python
import time
import random

def api_call_with_retry(func, max_retries=3):
    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            # Exponential backoff with jitter
            backoff = (2 ** attempt) + random.uniform(0, 1)
            log.warning("retry", attempt=attempt+1, backoff=backoff, error=str(e))
            time.sleep(backoff)
```

**2. Circuit Breaker:**

```python
# Trip on error-rate / timeout thresholds
if error_rate > 0.5 or timeout_count > 3:
    circuit_breaker.open()
    log.alert("circuit_breaker_open", reason="connectivity_issues")
    # Stop requests until health recovers
```

**3. Health Checks:**

```python
health_metrics = {
    "latency_ms": measure_latency(),
    "disconnect_count": count_disconnects(),
    "last_heartbeat": datetime.now(),
    "error_rate": errors / total_requests,
    "circuit_breaker_state": "closed" | "open" | "half_open"
}
```

### 10.3 Every Request MUST Have

```python
import uuid

request = {
    "request_id": str(uuid.uuid4()),  # Unique ID for tracing
    "timeout": 30,  # Explicit timeout in seconds
    # Structured logging (inputs redacted, outputs summarized)
}

# Log request
log.info("metaapi_request",
    request_id=request["request_id"],
    endpoint="***REDACTED***",  # Don't log credentials
    timeout=30
)

# Log response
log.info("metaapi_response",
    request_id=request["request_id"],
    status_code=200,
    rows_returned=len(data),  # Summary, not full data
    duration_ms=123
)
```

### 10.4 Multi-Account Support (4 Broker Accounts)

**MUST:**
- Isolate account configs
- Independent health state per account
- **NO cross-account state bleed**

```python
accounts = {
    "account_1": {
        "config": {"account_id": "...", "token": "..."},
        "health": {"latency": 50, "error_rate": 0.01, ...}
    },
    "account_2": {
        "config": {...},
        "health": {...}
    },
    "account_3": {
        "config": {...},
        "health": {...}
    },
    "account_4": {
        "config": {...},
        "health": {...}
    },
}

# Each account fully isolated - no shared state
```

---

## 11. LOGGING & ERROR HANDLING

### 11.1 Structured Logs Only (JSON)

```python
import structlog

log = structlog.get_logger()

# ✅ GOOD: Structured logging
log.info("backtest_start",
    run_id=run_id,
    dataset_id=dataset_id,
    instrument=instrument,
    timeframe=timeframe,
    seed=seed,
    train_start="2020-01-01",
    train_end="2021-12-31"
)

# Timing for each pipeline stage
log.info("stage_complete",
    stage="feature_extraction",
    duration_ms=123,
    rows_processed=10000
)

# Error context (stack trace + input references)
log.error("feature_error",
    error=str(e),
    error_type=type(e).__name__,
    traceback=traceback.format_exc(),
    input_file=input_file,
    row_number=i
)

# ❌ BAD: Unstructured logging
print("Starting backtest...")  # NO!
log.info(f"Processing {instrument}")  # NO! Not structured
```

### 11.2 Error Handling

**No Silent Failure:**

```python
# ✅ GOOD: Explicit error handling
try:
    result = operation()
except SpecificError as e:
    log.error("operation_failed", error=str(e))
    raise  # Or return defined fallback

# ❌ BAD: Silent failure
try:
    result = operation()
except:
    pass  # FORBIDDEN
```

---

## 12. SECURITY & HARD PROHIBITIONS

### 12.1 No Live Order Placement Code

```python
# ❌ FORBIDDEN - This codebase is for research/backtesting only
def place_live_order(symbol, quantity, price):
    broker_api.submit_order(...)  # NEVER IMPLEMENT

# ✅ CORRECT - Paper trading / simulation only
def simulate_order(symbol, quantity, price):
    log.info("PAPER_TRADE", symbol=symbol, qty=quantity, price=price)
    return simulated_fill
```

### 12.2 No Credential Leakage

```python
# ❌ FORBIDDEN: Hardcoded credentials
API_KEY = "sk_live_abc123..."  # NEVER

# ❌ FORBIDDEN: Logging credentials
log.info(f"Using API key: {api_key}")  # NEVER

# ✅ CORRECT: Environment variables
import os
api_key = os.getenv("METAAPI_TOKEN")
if not api_key:
    raise ValueError("METAAPI_TOKEN not set")

# ✅ CORRECT: Redact in logs
log.info("request", token="***REDACTED***", account_id="12345")
```

**Keys/tokens NEVER logged or hardcoded**

### 12.3 No Online Learning in Backtests

**Unless explicitly flagged and isolated:**

```python
# ❌ FORBIDDEN: Silent online learning in backtest
for t in range(len(data)):
    prediction = model.predict(data[t])
    model.fit(data[t], target[t])  # Updates model during backtest!

# ✅ CORRECT: Explicitly flagged
if online_learning_enabled:
    log.warning("online_learning_active", mode="walk_forward")
    for t in range(len(data)):
        prediction = model.predict(data[t])
        model.fit(data[t], target[t])  # Explicitly enabled
else:
    # Standard backtest - no model updates
    predictions = model.predict(data)
```

---

## 13. CODE QUALITY & STYLE

### 13.1 Python Style

**Follow:**
- **PEP 8** conventions
- **Black** for code formatting (line length: 100)
- **Ruff** for linting (select: E, F, I, W)
- Target Python 3.10+
- **Type hints for ALL function signatures**
- Prefer explicit over implicit

```python
# ✅ GOOD: Full type hints
def compute_energy(
    velocity: np.ndarray,
    mass: float = 1.0
) -> np.ndarray:
    """
    Compute kinetic energy.
    
    Args:
        velocity: Price velocity (log returns)
        mass: Effective mass (default 1.0)
        
    Returns:
        Kinetic energy array
    """
    return 0.5 * mass * (velocity ** 2)

# ❌ BAD: No type hints
def compute_energy(velocity, mass=1.0):
    return 0.5 * mass * (velocity ** 2)
```

### 13.2 Readability vs Speed

**Default to readable, idiomatic Python**

**Sacrifice readability ONLY when:**
- Profiling proves necessity
- Gain is material (>2x speedup)
- Code is documented with rationale

```python
# ✅ GOOD: Readable
energy = 0.5 * velocity ** 2

# ❌ BAD: Premature optimization (unless profiling proves necessary)
# Optimized version with Numba JIT
@jit(nopython=True)
def compute_energy_jit(v):
    e = np.empty_like(v)
    for i in range(len(v)):
        e[i] = 0.5 * v[i] * v[i]
    return e
# Only use if profiling shows vectorized version is bottleneck
```

### 13.3 Incremental Changes

**Changes MUST be:**
- Incremental and minimal
- Do NOT refactor unrelated areas
- Keep public interfaces stable unless migration provided
- One logical change per commit

```python
# ✅ GOOD: Focused change
# Fix energy calculation precision
energy = 0.5 * mass * velocity ** 2  # Changed from velocity^2 to velocity**2

# ❌ BAD: Unrelated changes mixed in
# Fix energy calculation + refactor unrelated module + update docs
```

### 13.4 Every New Module MUST Include

1. **Unit tests for core logic**
   ```python
   # tests/test_physics.py
   def test_compute_energy():
       velocity = np.array([1.0, 2.0, 3.0])
       energy = compute_energy(velocity)
       expected = np.array([0.5, 2.0, 4.5])
       assert np.allclose(energy, expected)
   ```

2. **At least one integration test for end-to-end pipeline**
   ```python
   # tests/test_integration.py
   def test_backtest_pipeline():
       data = load_test_data()
       results = run_backtest(data, config)
       assert results["sharpe"] > 1.0
   ```

### 13.5 Function Purity & Side Effects

**Functions MUST be referentially transparent unless explicitly marked:**

```python
# ✅ GOOD: Pure function
def compute_energy(velocity: np.ndarray) -> np.ndarray:
    return 0.5 * velocity ** 2

# ❌ BAD: Hidden side effects
_energy_cache = []  # Global state

def compute_energy(velocity):
    global _energy_cache
    energy = 0.5 * velocity ** 2
    _energy_cache.append(energy)  # Side effect!
    return energy

# ✅ GOOD: Explicitly marked if side effects necessary
def compute_and_cache_energy(velocity: np.ndarray, cache: list) -> np.ndarray:
    """Compute energy and append to cache (SIDE EFFECT: modifies cache)"""
    energy = 0.5 * velocity ** 2
    cache.append(energy)  # Explicit in signature and docstring
    return energy
```

**Side effects inside loops require justification:**

```python
# ❌ BAD: Side effects in loop (unless justified)
for i in range(len(data)):
    result = process(data[i])
    global_state.update(result)  # Side effect - avoid

# ✅ GOOD: Collect results, update once
results = [process(data[i]) for i in range(len(data))]
global_state.update_batch(results)
```

---

## 14. TYPE CHECKING & DOCUMENTATION

### 14.1 Type Checking (BasedPyRight)

**Check Optional Types for None:**

```python
# ✅ GOOD: Check for None
if self.data_quality_report is not None:
    result = self.data_quality_report.completeness_pct

# ❌ BAD: Access Optional without check
result = self.data_quality_report.completeness_pct  # Error if None
```

**Use TYPE_CHECKING for Conditional Imports:**

```python
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from .market_calendar import MarketCalendar

calendar: Optional["MarketCalendar"] = None
```

**Generic Types with Parameters:**

```python
from typing import List, Dict, Any

# ✅ GOOD
trades: List[Trade] = []
config: Dict[str, Any] = {}

# ❌ BAD: Missing type parameters
trades: List = []
config: Dict = {}
```

**Annotate Class Attributes:**

```python
from typing import List
from dataclasses import dataclass, field

# ✅ GOOD: Dataclass with type annotations
@dataclass
class MyEngine:
    trades: List[Trade] = field(default_factory=list)
    config: Dict[str, Any] = field(default_factory=dict)

# ✅ GOOD: Manual annotations
class MyEngine:
    trades: List[Trade]
    config: Dict[str, Any]
    
    def __init__(self):
        self.trades = []
        self.config = {}
```

### 14.2 Documentation

**All public functions/classes MUST have docstrings:**

```python
def compute_energy(
    velocity: np.ndarray,
    mass: float = 1.0
) -> np.ndarray:
    """
    Compute kinetic energy from velocity.
    
    Based on classical mechanics: E = ½mv²
    
    Args:
        velocity: Price velocity (log returns), shape (n,)
        mass: Effective mass (default 1.0)
        
    Returns:
        Kinetic energy array, shape (n,)
        
    Example:
        >>> velocity = np.array([0.01, 0.02, -0.01])
        >>> energy = compute_energy(velocity)
        >>> print(energy)
        [5.0e-05 2.0e-04 5.0e-05]
    """
    return 0.5 * mass * (velocity ** 2)
```

**Include mathematical formulas in LaTeX format where relevant:**

```python
def compute_damping(velocity_std: float, velocity_mean: float) -> float:
    """
    Compute damping coefficient (energy dissipation).
    
    Formula:
        ζ = σ(v) / μ(|v|)
    
    Where:
        σ(v) = standard deviation of velocity
        μ(|v|) = mean of absolute velocity
    
    Args:
        velocity_std: Standard deviation of velocity
        velocity_mean: Mean of absolute velocity
        
    Returns:
        Damping coefficient (dimensionless)
    """
    return velocity_std / velocity_mean if velocity_mean > 0 else 0.0
```

**Keep README and documentation in sync with code**

---

## 15. PHYSICS-FIRST APPROACH

### 15.1 Existing Physics Measurements (60+ features)

**From `physics_engine.py`:**

**Kinematics:**
- `velocity` (log-return)
- `acceleration` (Δ velocity)
- `jerk` (Δ acceleration) - best fat candle predictor

**Energy:**
- `kinetic_energy` (½mv²)
- `potential_energy` (1 / long-vol) - stored/compressed energy
- `eta` (KE / PE) - efficiency ratio

**Fluid Dynamics:**
- `reynolds` (trend / noise) - laminar vs turbulent
- `damping` / `zeta` (σ(v) / μ(|v|)) - friction
- `viscosity` (resistance to flow)
- `liquidity` (volume / price impact)

**Thermodynamics:**
- `entropy` (Shannon entropy of returns)
- `buying_pressure` (BP)

**From `physics_v7.py`:**
- `body_ratio` (|C-O| / (H-L))
- `energy` (body_ratio² × vol_ewma)
- `damping` (range expansion/contraction)

**All converted to rolling percentiles (0-1):**
- `KE_pct`, `Re_m_pct`, `zeta_pct`, `Hs_pct`, `PE_pct`, `eta_pct`, `velocity_pct`, `jerk_pct`

**NO traditional indicators. These physics measures capture everything needed.**

### 15.2 GPU Requirements

**Training REQUIRES GPU acceleration. CPU training is 100x slower.**

**Check GPU availability:**

```python
import torch

print(f"CUDA: {torch.cuda.is_available()}")
print(f"Device count: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"Device name: {torch.cuda.get_device_name(0)}")
```

**For AMD GPUs (ROCm):**

```bash
pip install torch --index-url https://download.pytorch.org/whl/rocm6.0

# Environment variables for RX 7600 / RDNA3:
export HSA_OVERRIDE_GFX_VERSION=11.0.0
export HIP_VISIBLE_DEVICES=0

# For RX 6000 series / RDNA2:
export HSA_OVERRIDE_GFX_VERSION=10.3.0
export HIP_VISIBLE_DEVICES=0
```

**For NVIDIA GPUs (CUDA):**

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

**CRITICAL: If no GPU detected, DO NOT proceed with training. Fix GPU first.**

The code will detect AMD ROCm automatically via `torch.version.hip`.

---

## 16. TESTING REQUIREMENTS

### 16.1 Defense-in-Depth (Multi-Layer Validation)

**1. Unit Tests** (`pytest`)
- **100% code coverage required** for new features
- Property-based testing with `hypothesis` for mathematical functions
- Numerical stability checks (NaN shields, overflow)

```python
from hypothesis import given, strategies as st

@given(st.lists(st.floats(min_value=-10, max_value=10), min_size=1, max_size=1000))
def test_energy_always_positive(velocities):
    """Property: Energy must always be non-negative"""
    velocity = np.array(velocities)
    energy = compute_energy(velocity)
    assert np.all(energy >= 0), "Energy cannot be negative"
```

**2. Integration Tests**
- End-to-end pipeline validation
- Physics → RL → Risk → Execution flow

**3. Monte Carlo Backtesting**
- **100 runs per instrument minimum**
- Statistical significance testing (**p < 0.01**)
- Out-of-sample validation required

**4. Theorem Validation**
- Mathematical proofs must be documented in `docs/theorem_proofs.md`
- Continuous validation via CI/CD

**5. Health Monitoring**
- Real-time Composite Health Score (CHS)
- Circuit breakers (halt if **CHS < 0.55**)

### 16.2 Running Tests

```bash
# Run all tests
make test
# or
pytest tests/ -v

# Run specific test file
pytest tests/test_physics.py -v

# Run with coverage
pytest tests/ --cov=kinetra --cov-report=html

# Run property-based tests
pytest tests/ -v --hypothesis-show-statistics
```

### 16.3 Performance Targets

| Metric | Target | Purpose |
|--------|--------|---------|
| **Omega Ratio** | > 2.7 | Asymmetric returns |
| **Z-Factor** | > 2.5 | Statistical edge significance |
| **% Energy Captured** | > 65% | Physics alignment efficiency |
| **Composite Health Score** | > 0.90 | System stability |
| **% MFE Captured** | > 60% | Execution quality |
| **Test Coverage** | 100% | Code quality |

---

## 17. DELIVERABLES & VALIDATION

### 17.1 When Delivering Changes, Provide

**1. Exact files/modules changed:**
```
Modified:
- kinetra/physics_engine.py (lines 45-67)
- kinetra/backtest_engine.py (lines 123-145)

Added:
- kinetra/new_module.py
- tests/test_new_module.py
```

**2. Concise list of behavioral changes:**
```
Changes:
- Energy calculation now uses Kahan summation for numerical stability
- Backtest engine now saves intermediate results every 100 episodes
- Added circuit breaker that halts if error rate > 50%
```

**3. Validation steps to run (commands):**
```bash
# Unit tests
pytest tests/test_physics.py::test_energy_numerical_stability -v

# Integration test
pytest tests/test_integration.py::test_backtest_with_circuit_breaker -v

# Manual verification
python scripts/verify_changes.py --instrument BTCUSD --timeframe H1
```

**Never claim something "works" without stating what was run to validate**

### 17.2 Commit Messages

```bash
# ✅ GOOD: Clear, descriptive
git commit -m "Add Kahan summation to energy calculation for numerical stability

- Fixes floating point accumulation errors in long backtests
- Verified with property-based tests (hypothesis)
- Benchmark shows <1% performance impact"

# ❌ BAD: Vague
git commit -m "fix bug"
git commit -m "update code"
```

### 17.3 Standardization

**Standardize across:**
- `/project` root
- `/.github` workflows and docs
- `/agents` (AI agent configurations)

**Consistent:**
- Naming conventions
- File structure
- Documentation format
- Code style

---

## 📋 QUICK REFERENCE CHECKLIST

**Before committing code, verify:**

### Core Philosophy
- [ ] No magic numbers (all thresholds derived/configurable)
- [ ] No traditional TA indicators (physics only)
- [ ] No linear assumptions (unless explicitly approved)
- [ ] Asymmetric features (up/down separate)
- [ ] Adaptive percentiles (not static thresholds)

### Performance
- [ ] Vectorized (no explicit loops unless profiled)
- [ ] Benchmarks provided (if optimization)
- [ ] Profiling evidence (before/after)
- [ ] Algorithmic improvements prioritized over micro-opts

### Data & Safety
- [ ] Data validated (timestamps, OHLC, volume)
- [ ] Quality report emitted
- [ ] Atomic saves (PersistenceManager)
- [ ] Backed up before git operations
- [ ] No credentials leaked

### Reproducibility
- [ ] Deterministic (seeded RNG, stable ordering)
- [ ] Train/valid/test splits explicit
- [ ] No lookahead bias
- [ ] Execution model explicit (fills, slippage, spread, commission, swaps)

### Validation
- [ ] Out-of-sample performance reported
- [ ] Robustness checks run (Monte Carlo / regime slices)
- [ ] Effect sizes + p-values (not just p-values)
- [ ] Confidence intervals included
- [ ] Leakage checks passed

### Code Quality
- [ ] Type hints for all functions
- [ ] Docstrings for public APIs
- [ ] Tests included (unit + integration)
- [ ] Structured logging (JSON)
- [ ] No silent failures
- [ ] Incremental changes only
- [ ] Public interfaces stable

### Testing
- [ ] 100% code coverage (new features)
- [ ] Property-based tests (hypothesis)
- [ ] Integration tests pass
- [ ] Monte Carlo backtests (100+ runs)

---

## 🎯 PHILOSOPHY SUMMARY

> **"We don't know what we don't know. The market will teach us through exploration, not through assumptions."**

> **"If you can't explain it with physics (energy, friction, viscosity, entropy), you don't understand it."**

> **"Crypto is not stocks. Stocks are not forex. One rule does not fit all."**

> **"Never lose user data. EVER."**

> **"Vectorize first, optimize later, and only when profiling proves it necessary."**

> **"Determinism first: identical inputs must produce identical outputs."**

> **"No silent failures. Every exception has a defined outcome or causes a hard fail with context."**

---

**This is the canonical rulebook. All rules consolidated. Single source of truth.** 🚀

**Version:** 2.0  
**Status:** ACTIVE - Read before every task  
**Last Updated:** 2024-01-09