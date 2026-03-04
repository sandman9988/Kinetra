

# KINETRA AGENT RULES - MASTER REFERENCE
**Version:** 4.11  
**Last Updated:** 2026-02-28  
**Status:** Canonical - Single Source of Truth
**Copilot Sync Marker:** `AGENT_RULES_MASTER.md@v4.11`

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
10. [Broker Connectivity (MetaAPI / MT5 / cTrader)](#10-broker-connectivity-metaapi--mt5--ctrader)
11. [Logging & Error Handling](#11-logging--error-handling)
12. [Security & Hard Prohibitions](#12-security--hard-prohibitions)
13. [Code Quality & Style](#13-code-quality--style)
14. [Type Checking & Documentation](#14-type-checking--documentation)
15. [Physics-First Approach](#15-physics-first-approach)
16. [Testing Requirements](#16-testing-requirements)
17. [Deliverables & Validation](#17-deliverables--validation)
18. [Consolidation & Versioning](#18-consolidation--versioning)
19. [Parallelization & Performance](#19-parallelization--performance)
20. [Feature Engineering Intelligence](#20-feature-engineering-intelligence)
21. [Empirical Ablation Results (canonical_v2.2)](#21-empirical-ablation-results-canonical_v22)
22. [Data Foundation Fix](#22-data-foundation-fix-️-current-priority) ⭐ CURRENT PRIORITY
23. [Data Pipeline Architecture](#23-data-pipeline-architecture)
24. [Known Tech-Debt (Audit 2026-02-23)](#24-known-tech-debt-audit-2026-02-23)
25. [Menu Restructure — Pipeline Correction](#25-menu-restructure--pipeline-correction) ✅ COMPLETE
26. [Gate-Logic Bugfixes (2026-02-27)](#26-gate-logic-bugfixes-2026-02-27)
27. [DRY Violations Register (2026-03-01)](#27-dry-violations-register-2026-03-01) ⭐ ACTIVE
28. [Multi-Broker Architecture (2026-03-01)](#28-multi-broker-architecture-2026-03-01) 📐 DESIGN
29. [Sprint 5 Architecture — Industrialisation (2026-03-04)](#29-sprint-5-architecture--industrialisation-2026-03-04) ⭐ CURRENT
30. [Menu & Terminal UI Styling Guide](#30-menu--terminal-ui-styling-guide)
31. [Path Resolution Policy (2026-03-04)](#31-path-resolution-policy-2026-03-04) ⭐ ACTIVE

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

### 2.10 Data Preparation Philosophy — What Belongs in Prepared Files

**The Prepare Data step (pipeline step 5) computes ONLY physics-justified features.
Fixed-period TA derivatives are intentionally excluded.**

**INCLUDE in prepared files:**
- Raw OHLCV normalised (timezone, format, holiday tags, session window)
- Physics state from `PhysicsEngine`: `velocity`, `energy`, `damping`, `entropy`,
  `reynolds`, `potential`, `eta`, `BP`, `pe`, `liquidity`, `viscosity`
- Rolling-percentile adaptive sensors: `KE_pct`, `Hs_pct`, `Re_m_pct`, `zeta_pct`,
  `PE_pct`, `eta_pct`, `velocity_pct`, `jerk_pct`
- Regime labels: `cluster`, `regime`, `regime_age_frac`

**NEVER include in prepared files (magic-number TA derivatives):**
- ❌ `MA_{6,12,24,48}`, `EMA_N` — fixed-period moving averages
- ❌ `ATR_14`, `atr_pct` — 14-period ATR (magic number)
- ❌ `vol_ma_N`, `range_ma_N` — fixed-window volume/range averages
- ❌ `close_vs_ma_N` — price-vs-MA derivatives
- ❌ `consec_up`, `consec_down` — run-length counters (no physics basis)
- ❌ `vol_spike`, `big_move` — fixed-multiplier threshold features

**Rationale:** Everything in the excluded list is either:
1. Already captured adaptively by the PhysicsEngine (`reynolds` ≈ trend/noise,
   `damping` ≈ mean-reversion, `_pct` sensors ≈ rolling rank), OR
2. A magic number that violates §2.1 and §2.4.

Fixed-period derivatives pollute additive feature testing (§20.5) by mixing TA
assumptions with physics-derived signals, making ablation results uninterpretable.
Evaluate any candidate derived feature through the additive testing pipeline
(Menu 3 › option 2) — not by baking it into every prepared file.

**Denoising is SUPPLEMENTARY, not pre-processing:**
- Denoise Data (menu item 14) is locked until Prepare Data (step 11) is complete.
- Denoising raw OHLCV before physics feature extraction destroys the market
  microstructure signals (sharp transitions, volatility clustering, regime changes)
  that the PhysicsEngine is specifically designed to detect.
- Use denoised data only as an ablation experiment input — compare denoised vs raw
  as alternative feature sources, never replace the raw physics pipeline with it.

---

## 3. DATA SAFETY & INTEGRITY

### 3.1 Data Safety (#1 Priority - NEVER LOSE USER DATA)

**User data (especially `data/master_standardized/` CSVs) is IRREPLACEABLE**
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
    filepath="data/master_standardized/metaapi/forex/BTCUSD/BTCUSD_M1_202401010000_202412312359.csv",
    content=df,
    writer=lambda path, data: data.to_csv(path, index=False)
)

# Restore if needed
pm.restore_latest("data/master_standardized/metaapi/forex/BTCUSD/BTCUSD_M1_202401010000_202412312359.csv")
```

**How Atomic Save Works:**
1. Create timestamped backup of existing file (if exists)
2. Write to temporary file in same directory
3. Atomic rename (OS guarantees either full success or full failure)
4. Automatic recovery on failure

**Git Safety Rules:**

```bash
# DANGEROUS - Backup first!
git rm --cached data/master_standardized/**/*.csv
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
data/master_standardized/
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

## 10. BROKER CONNECTIVITY (MetaAPI / MT5 / cTrader)

> **Multi-broker context:** Kinetra currently connects to MT5 brokers via MetaAPI (cloud)
> and the local MT5 Python package.  cTrader Open API integration is **planned** (see §28).
> All rules in this section apply to **any** broker connector — MetaAPI, MT5, and future
> cTrader.  When refactoring MetaAPI code, design for broker-neutrality where practical
> (see §28 for the canonical abstraction boundary).

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
- **NO broker-type assumptions** — account may be MetaAPI, MT5, or cTrader

```python
accounts = {
    "account_1": {
        "broker_type": "metaapi",  # or "mt5" or "ctrader"
        "config": {"account_id": "...", "token": "..."},
        "health": {"latency": 50, "error_rate": 0.01, ...}
    },
    "account_2": {
        "broker_type": "ctrader",
        "config": {"client_id": "...", "access_token": "..."},
        "health": {...}
    },
    # ...
}

# Each account fully isolated - no shared state, no cross-broker bleed
```

### 10.5 Broker-Neutral Refactoring Rule

When modifying any broker-specific code (MetaAPI scripts, `mt5_connector.py`,
`poll_symbol_specs.py`, download scripts), **ask yourself**:

> "Could this logic work for a different broker with a different API?"

If **yes** → extract into a broker-neutral helper (e.g. FX enrichment, save/merge
logic, OHLCV normalisation, retry/backoff).

If **no** → keep it broker-specific but behind a clear interface boundary so a
future cTrader implementation can provide its own version.

See §28 for the full multi-broker architecture design.

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
- **Ruff ≥ 0.15** for linting — see `[tool.ruff.lint]` in `pyproject.toml`
- Target Python 3.10+
- **Type hints for ALL function signatures**
- Prefer explicit over implicit

---

#### 13.1.1 Ruff Configuration (canonical — `pyproject.toml`)

```toml
[tool.ruff]
line-length = 100
exclude = ["archive", ".venv", "__pycache__", "*.egg-info"]

[tool.ruff.lint]
select = ["E", "F", "I", "W"]
ignore = [
    "E501",   # line-too-long — handled by Black
    "E741",   # ambiguous variable name — 'l' = low price is idiomatic in OHLCV/trading code
]

[tool.ruff.lint.per-file-ignores]
# Scripts prepend sys.path before kinetra imports — E402 is intentional
"scripts/**" = ["E402"]
"tests/**"   = ["E402", "F401"]
"kinetra/dsp_features.py"   = ["E402"]
"kinetra/physics_engine.py" = ["E402"]
"kinetra/config.py"         = ["E402"]

# superpot_explorer uses `features[fi] = ...; fi += 1` array-fill idiom
"scripts/analysis/superpot_explorer.py" = ["E702"]

# Column-mapping tables use aligned `if lc == x: col_map[c] = y` — intentional
"scripts/research/liquidity_gate_analysis.py" = ["E701"]
"scripts/research/mfe_runway_sweep.py"        = ["E701"]
"scripts/feature_ablation_sweep.py"           = ["E701"]

# Test helpers where bare-except / optional imports are acceptable
"scripts/testing/**" = ["E722", "F401"]
```

**Run ruff (check only):**
```bash
ruff check .
```

**Run ruff with autofix (safe + unsafe):**
```bash
ruff check . --fix --unsafe-fixes
```

> The `archive/` directory is excluded from all ruff checks — it contains legacy
> code that is intentionally not maintained.

---

#### 13.1.2 Ruff-Aware Coding Rules

These rules must be followed in **all new code** so that ruff stays at zero violations.

**Imports**

| Rule | Requirement |
|------|-------------|
| `F401` | Remove unused imports. Optional/conditional imports inside `try/except ImportError` blocks must have `# noqa: F401` on the import line. |
| `F402` | Never reuse a module-level import name as a loop variable (e.g., `for stats in ...` shadows `from scipy import stats`). |
| `E402` | All imports go at the top of the file. When `sys.path` manipulation is unavoidable before imports (scripts only), the file must be covered by the `scripts/**` per-file ignore — do **not** scatter `# noqa: E402` inline. |
| `I001` | Import order: stdlib → third-party → local. Black/ruff-isort handles this automatically on `--fix`. |

```python
# ✅ GOOD: optional dep inside try/except
try:
    from kinetra.performance import sample_entropy_fast  # noqa: F401
    _OPTIMIZED = True
except ImportError:
    _OPTIMIZED = False

# ❌ BAD: bare unused import at module level
import pandas_market_calendars  # never used
```

**Exceptions**

| Rule | Requirement |
|------|-------------|
| `E722` | Never use bare `except:`. Always specify at least `except Exception:` or a more specific type. |

```python
# ✅ GOOD
try:
    result = risky_call()
except ValueError as e:
    log.warning("bad value: %s", e)
except Exception:
    pass  # swallow intentionally — document why

# ❌ BAD
try:
    result = risky_call()
except:          # bare except — catches SystemExit, KeyboardInterrupt, etc.
    pass
```

**Variable names**

| Rule | Requirement |
|------|-------------|
| `E741` | Globally ignored for `l` (= *low* price in OHLCV) — this is the **only** exception. Do not use `O` (looks like zero) or `I` (looks like one) as variable names anywhere. |

```python
# ✅ GOOD: l = low price is allowed
o, h, l, c, v = df["open"], df["high"], df["low"], df["close"], df["volume"]

# ❌ BAD: O and I as variable names
O = order_book   # looks like zero
I = identity     # looks like one
```

**Statement style**

| Rule | Requirement |
|------|-------------|
| `E702` | Do not put multiple statements on one line with `;`, except inside `scripts/analysis/superpot_explorer.py` where the `features[fi] = …; fi += 1` array-fill idiom is pre-approved. |
| `E701` | Do not put `if`/`for`/`while` body on the same line as the colon, except in the three approved research scripts (see per-file-ignores). |
| `E712` | Use `is True` / `is False` / truthiness — never `== True` or `== False`. |

```python
# ✅ GOOD
if condition:
    do_something()

# ❌ BAD
if condition: do_something()   # E701
a = 1; b = 2                   # E702 (outside approved files)
if flag == True:               # E712
```

**Forward references & TYPE_CHECKING**

When a type annotation references a class that is only imported inside a function
(to avoid circular imports), use `TYPE_CHECKING`:

```python
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from kinetra.data import DataManager  # canonical (DRY-10 Phase C)

def process(dm: "DataManager") -> None: ...
```

**Undefined names (F821)**

- Never reference a variable before it is assigned.
- Constants used across functions must be defined at module level, not inside a
  docstring or in a different scope.
- Loop variables that shadow module-level names cause `F402` — rename the loop
  variable instead.

**Duplicate dict keys (F601)**

Lazy-loader dicts (like `_LAZY_MODULES` in `kinetra/__init__.py`) must not repeat
the same string key. The last definition wins silently — a guaranteed source of
hard-to-find bugs. Run `ruff check` before adding any new entry.

**Return outside function (F706)**

`return` is only valid inside a function or method body. In `if __name__ == "__main__"`
blocks, use `sys.exit(code)` to terminate early.

---

#### 13.1.3 Pre-Commit Checklist (ruff)

Before every commit, the following must pass with **zero violations**:

```bash
ruff check .          # linting
ruff check . --diff   # preview any auto-fixable changes
```

If violations are introduced, fix them before committing — do **not** add blanket
`# noqa` suppressions without a documented reason.

Acceptable `# noqa` patterns:
- `# noqa: F401` — optional import inside `try/except ImportError`
- `# noqa: E402` — **only** if the file is not already covered by a `per-file-ignores` glob

Everything else requires either a real fix or a new entry in `[tool.ruff.lint.per-file-ignores]`
with a comment explaining the rationale.

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

### 15.1 Core Physics Features (prepare-time output, ~20 columns)

> **Note:** The "60+ features" figure referred to a previous implementation that
> included magic-number TA derivatives (`MA_6`, `ATR_14`, etc.). These have been
> removed. See §2.10 for the canonical list of what belongs in prepared files.
> Additional features are evaluated and added only via Additive Feature Testing
> (Menu 3 › option 2), never baked into the base prepared dataset.

**From `physics_engine.py` — always computed at prepare time:**

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

## 18. CONSOLIDATION & VERSIONING

### 18.1 Consolidation Rules

**NEVER create duplicate files. ALWAYS enhance existing canonical files.**

**CANONICAL FILES (Single Source of Truth):**

| Category | Canonical File | Version |
|----------|----------------|---------|
| **Menu** | `kinetra_menu.py` | v2.1.0 |
| **E2E Testing** | `scripts/testing/comprehensive_e2e_test.py` | v2.1.0 |
| **Data Management** | `scripts/data_manager.py` | v1.0.0 |
| **Batch Backtest** | `scripts/batch_backtest.py` | v1.1.0 |
| **Physics Engine** | `kinetra/physics_engine.py` | v1.0.0 |
| **Backtest Engine** | `kinetra/backtest_engine.py` | v1.0.0 |
| **RL Agent** | `kinetra/rl_agent.py` | v1.0.0 |
| **Data Prep (parallel)** | `scripts/download/parallel_data_prep.py` | v1.1.0 |
| **Denoise Filters** | `kinetra/denoise_filters.py` | v1.1.0 |

**When Adding New Functionality:**
1. ✅ Check if canonical file exists for that category
2. ✅ Enhance existing file (add features, fix bugs)
3. ✅ Increment version appropriately (MAJOR.MINOR.PATCH)
4. ✅ Update VERSION.md manifest
5. ❌ NEVER create new file if canonical exists
6. ❌ NEVER fork/copy existing files

**When Removing Old/Duplicate Files:**
1. ✅ Move to `archive/` directory (never delete)
2. ✅ Create README.md in archive explaining what was archived
3. ✅ Update references in codebase
4. ✅ Document in ARCHIVAL_MANIFEST.md

### 18.2 Versioning Rules

**All modules MUST have version constants:**

```python
__version__ = "1.0.0"
__author__ = "Kinetra Project"

"""
Module docstring with Version History:
    1.1.0 (2025-01-04): Added feature X
    1.0.0 (2025-01-03): Initial versioned release
"""
```

**Semantic Versioning (MAJOR.MINOR.PATCH):**
- **MAJOR**: Breaking changes, incompatible API changes
- **MINOR**: New features, backward-compatible additions
- **PATCH**: Bug fixes, backward-compatible fixes

**Version Increment Checklist:**
- [ ] Updated `__version__` in module
- [ ] Updated Version History in docstring
- [ ] Updated VERSION.md manifest
- [ ] Tests pass with new version
- [ ] CHANGELOG entry added (if significant)

### 18.3 Archive Policy

**Archive Structure:**
```
archive/
├── testing_frameworks/
│   ├── legacy/           # Old test files
│   └── ARCHIVAL_MANIFEST.md
├── menus/                # Old menu implementations
├── scripts/              # Old utility scripts
└── README.md             # Archive overview
```

**Restoration Policy:**
1. ❌ NEVER restore archived files directly
2. ✅ Extract needed functionality into canonical file
3. ✅ Follow versioning rules for the enhancement
4. ✅ Document the migration

---

## 19. PARALLELIZATION & PERFORMANCE

### 19.1 Parallelization Strategy

**Use the right tool for the task:**

| Task Type | Tool | Example |
|-----------|------|---------|
| **CPU-bound** | `multiprocessing` / `ProcessPoolExecutor` | Data prep, feature extraction, backtests |
| **I/O-bound** | `asyncio` / `ThreadPoolExecutor` | Downloads, API calls, file reads |
| **GPU-accelerated** | PyTorch / CuPy | Neural networks, large matrix ops |

### 19.2 CPU-Adaptive Worker Selection

**ALWAYS use adaptive worker counts:**

```python
from kinetra.cpu_utils import get_optimal_workers, get_optimal_concurrency

# CPU-intensive tasks (data prep, backtests)
workers = get_optimal_workers("balanced")  # ~75% of logical cores

# I/O-intensive tasks (downloads, API calls)
concurrency = get_optimal_concurrency("network")  # ~2x logical cores
```

**Worker Count Guidelines:**
- `"light"`: 50% of cores (leave headroom for UI/system)
- `"balanced"`: 75% of cores (default, good balance)
- `"heavy"`: 95% of cores (max performance, batch jobs)

### 19.3 Parallel Processing Patterns

**ProcessPoolExecutor for CPU-bound:**

```python
from concurrent.futures import ProcessPoolExecutor, as_completed
from kinetra.cpu_utils import get_optimal_workers

def process_parallel(tasks):
    workers = get_optimal_workers("balanced")
    results = []
    
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(process_single, task): task for task in tasks}
        
        for future in as_completed(futures):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                logging.error(f"Task failed: {e}")
    
    return results
```

**Asyncio for I/O-bound:**

```python
import asyncio
from kinetra.cpu_utils import get_optimal_concurrency

async def download_parallel(urls):
    concurrency = get_optimal_concurrency("network")
    semaphore = asyncio.Semaphore(concurrency)
    
    async def download_with_limit(url):
        async with semaphore:
            return await download(url)
    
    return await asyncio.gather(*[download_with_limit(url) for url in urls])
```

### 19.4 Vectorization Requirements

**Python loops are LAST RESORT. Prefer:**

1. **NumPy vectorized ops** (fastest)
2. **Pandas column operations** (fast, readable)
3. **Broadcasting** (efficient for array ops)
4. **np.select / np.where** (conditional assignment)

**Vectorization Patterns:**

```python
# ❌ SLOW: Python loop
for i in range(len(arr)):
    if condition[i]:
        result[i] = value

# ✅ FAST: NumPy boolean indexing
result[condition] = value

# ❌ SLOW: Python loop for cumulative
total = 0
for x in arr:
    total += x
    result.append(total)

# ✅ FAST: NumPy cumsum
result = np.cumsum(arr)

# ❌ SLOW: Multiple if/elif in loop
for i in range(len(arr)):
    if cond1[i]:
        result[i] = val1
    elif cond2[i]:
        result[i] = val2

# ✅ FAST: np.select
result = np.select([cond1, cond2], [val1, val2], default=default_val)
```

### 19.5 Performance Benchmarking

**Before optimizing, MEASURE:**

```python
import time

start = time.perf_counter()
# ... code to benchmark ...
elapsed = time.perf_counter() - start
print(f"Elapsed: {elapsed:.3f}s")
```

**Document speedups:**
```python
# Vectorized version: ~50x faster than loop
# Before: 500ms for 100k elements
# After: 10ms for 100k elements
```

---

## 20. FEATURE ENGINEERING INTELLIGENCE

### 20.1 Canonical Feature Set History

The H4 observation vector has gone through deliberate, evidence-driven evolution.
Every change is OOS-validated before being locked in.

| Version | Features | Key Change | OOS Outcome |
|---------|----------|------------|-------------|
| **v2.1** | 75 | Baseline (4 validated MI wins over v1) | Positive OOS gap |
| **v2.2** | 75 | Removed 3 WEAK features (`volume_surge`, `cvd`, `tail_ratio`); added 3 UNIVERSAL replacements (`vol_ts_32`, `te_asymmetry`, `vol_of_vol`) | Validated — all 3 additions are IMPORTANT_KEEP in LOO sweep |

**Canonical file:** `results/h4_prime_discovery/h4_canonical_features.txt`

---

### 20.2 Feature Classification System

All 149 MeasurementEngine features are classified after cross-category MI analysis:

| Class | Count | Rule |
|-------|-------|------|
| **UNIVERSAL** | 43 | Strong MI signal in ≥ 4 / 5 categories — always include |
| **MULTI_CATEGORY** | 25 | Strong in 2–3 categories — include globally, gate per-category if needed |
| **CATEGORY_SPECIFIC** | 26 | Strong in exactly 1 category — use only in that category's profile |
| **WEAK** | 55 | Low MI across all categories — exclude from policy obs vector |

**Classification source:** `results/category_mi_analysis/`
**Profile definitions:** `configs/feature_profiles.json`

---

### 20.3 Risk-Layer Routing (Architecture)

The following features are **NOT** policy observation features.
They carry risk/tail information and MUST be routed to the risk layer:

| Feature | Destination | Role |
|---------|-------------|------|
| `tail_ratio` | `AdaptiveRewardShaper` | Asymmetrically scale α/β (MFE/MAE weight) |
| `kurtosis` / `excess_kurtosis` | `RiskManager.calculate_risk_health_score()` | Kurtosis penalty component |
| `vpin` | `RiskManager` + action mask | Toxic flow gate (blocks trades above threshold) |

**Rationale:** These are distributional/tail metrics — they describe the *risk environment*, not a predictive market signal. Feeding them into the RL policy vector causes the agent to try to predict them, which adds noise without edge.

**Implementation:**
- `RiskManager.calculate_risk_health_score()` accepts `excess_kurtosis`
- `RiskManager.tail_position_factor()` helper scales position size
- `AdaptiveRewardShaper` accepts `tail_ratio` and adjusts α/β asymmetrically

---

### 20.4 Per-Category Profile Rules

From the category OOS comparison (category_oos_comparison.py):

| Category | Decision | Rationale |
|----------|----------|-----------|
| **energy** | ✅ Deploy category profile (+4 additions) | Only category to cross Ω=1.0. Adds: `pacf_lag2`, `trend_quality`, `pv_divergence`, `release_potential_v2` |
| **forex** | ✅ Global canonical only | No additions needed — already dominant |
| **metals** | ✅ Global canonical only | Profile wins on XPDUSD but loses on XAUUSD/XAGUSD — not robust |
| **indices** | ❌ Remove `wt_phase_cos_L1/L2` additions | Overfit on CHINA50/GER40/FRA40 — instrument set too heterogeneous |
| **crypto** | ❌ Fix instrument set first | Need BTC/ETH/SOL/ADA — current set (ABNB, BCH pairs, BTCEUR) is not a clean crypto set |

**Energy profile additions are the ONLY validated per-category win to date.**

**Before adding any category-specific feature:**
1. Run `scripts/category_oos_comparison.py --category <cat>` for the candidate additions
2. Require Test Ω improvement > +0.02 AND no catastrophic overfitting (Train Ω − Test Ω gap < 0.30)
3. Confirm improvement holds on ≥ 70% of instruments in that category

---

### 20.5 Feature Addition / Removal Protocol

**NEVER add or remove a canonical feature without following this protocol:**

```
1. Propose      →  State the feature name, its MI class, and the hypothesis
2. Fast screen  →  scripts/feature_ablation_sweep.py --fast --feature <name>
                   A fast-mode ★ SIGNIFICANT result is a SCREENING SIGNAL only.
                   It is NOT sufficient to authorise removal.
3. Full confirm →  scripts/feature_ablation_sweep.py (no --fast) --feature <name>
                   --episodes-train 400 --episodes-test 150
                   Full-data 95% CI lower bound > 0 is REQUIRED to proceed.
                   Fast-mode "significant" removals that flip to KEEP on full data
                   are artefacts of the 3 000-bar window — see §21.0.
4. Combination  →  If removal is confirmed, test removing it alongside any
                   co-correlated candidates (check MI correlation matrix).
                   Cross-feature dependency can reverse individual verdicts.
5. Lock in      →  Update h4_canonical_features.txt, bump version comment
                   Update configs/feature_profiles.json metadata
                   Record outcome in Section 21 of this document
```

**The ΔΩ thresholds for LOO ablation (full-data run only):**
- `ΔΩ > +0.02`  → CANDIDATE_REMOVAL (removing improves OOS)
- `ΔΩ < −0.02`  → IMPORTANT_KEEP (removing hurts OOS — feature is protected)
- `|ΔΩ| ≤ 0.02` → NEUTRAL (no meaningful impact — keep, but lowest priority)

**★ SIGNIFICANT** = 95% bootstrap CI lower bound strictly on the correct side of zero
(full-data run). Fast-mode ★ SIGNIFICANT is for screening only — see §21.0 for
the bar-cap artefact phenomenon.

**Empirically observed flip rate:** In the 2026-02-21 confirmatory run, 6 of 9
fast-mode ★ SIGNIFICANT candidates flipped to IMPORTANT_KEEP on full data.
Treat all fast-mode removal signals as unconfirmed hypotheses until full-data validated.

---

### 20.6 Ablation Sweep Infrastructure

**Script:** `scripts/feature_ablation_sweep.py`

```bash
# Fast global LOO sweep (all 91 instruments, bar-capped at 3000)
python scripts/feature_ablation_sweep.py --fast

# Parallel (8 workers — ~5× faster, fork-safe on Linux)
python scripts/feature_ablation_sweep.py --fast --workers 8

# Resume interrupted sweep
python scripts/feature_ablation_sweep.py --fast --resume

# Validate a specific proposed addition or removal
python scripts/feature_ablation_sweep.py --fast --feature <feature_name>

# Only test the v2.2 new additions
python scripts/feature_ablation_sweep.py --fast --subset v22_changes
```

**Outputs:**
- `results/feature_ablation/ablation_summary.csv` — all 75 rows sorted by ΔΩ descending
- `results/feature_ablation/ablation_report.txt` — human-readable ranked report
- `results/feature_ablation/per_feature/{feature}.json` — raw PnLs + bootstrap CI
- `results/feature_ablation/baseline.json` — cached baseline run (reused with `--resume`)

---

## 21. EMPIRICAL ABLATION RESULTS (canonical_v2.2)

> **Two-stage protocol completed: fast-mode screening → full-data confirmatory.**
> Results from both stages are recorded here. The confirmatory stage is authoritative.

---

### 21.0 ⚠️ CRITICAL METHODOLOGICAL FINDING: Fast-Mode Bar-Cap Artefacts

**The 3 000-bar fast-mode cap produces systematic false-positive removal signals.**

When the confirmatory full-data run (up to 6 438 bars per instrument, 400+150 episodes)
was run against the 9 features that appeared as ★ SIGNIFICANT CANDIDATE_REMOVAL in the
fast-mode sweep, **6 of the 9 flipped to IMPORTANT_KEEP**:

| Feature | Fast-mode ΔΩ | Full-data ΔΩ | Verdict change |
|---------|-------------|-------------|----------------|
| `xwt_coherence_L2` | +0.892 ★ | **−0.262** | ❌ FLIP → KEEP |
| `hht_imf_energy_entropy` | +0.866 ★ | **−0.202** | ❌ FLIP → KEEP |
| `hht_amplitude_mod` | +0.817 ★ | **−0.190** | ❌ FLIP → KEEP |
| `wt_phase_sin_L5` | +0.700 ★ | **−0.226** | ❌ FLIP → KEEP |
| `xwt_coherence_L1` | +0.603 ★ | **+0.334 ★** | ✅ Confirmed removal |
| `ou_speed` | +0.529 ★ | **+0.026** | ➡️ Downgraded to neutral |
| `te_vol_to_price` | +0.361 ★ | **+0.162** | ⬇️ Downgraded to tentative |
| `te_price_to_vol` | +0.302 ★ | **−0.082** | ❌ FLIP → KEEP |
| `spectral_power_ratio` | +0.293 ★ | **−0.115** | ❌ FLIP → KEEP |

**Why this happens:** The 3 000-bar window samples only the most recent ~2.5 years of
H4 data. Many XWT/HHT/spectral features encode multi-regime structure that spans
longer history. In a single-regime window those signals appear as noise; across the full
multi-regime history they are load-bearing. The fast-mode sweep is therefore valid only
as a **screening tool** to generate candidates — it must never be used to authorise removal.

**Rule (added to §20.5):** The full-data confirmatory run is the ONLY authoritative gate.
Fast-mode ★ SIGNIFICANT is a screening signal, not a removal decision.

---

### 21.1 Stage 1 — Fast-Mode Screening Sweep

**Date:** 2026-02-21 · **Protocol:** `--fast` (3 000 bar cap), 91 instruments, 200+80 episodes, seed=42, 1 000 bootstrap resamples
**Baseline:** Train Ω = 0.7360 · Test Ω = 0.7252
**Output:** `results/feature_ablation/` (75 per-feature JSON files)

Complete ranked table (fast-mode) — **for screening reference only**:

| Rank | Feature | Fast ΔΩ | CI 95% | p-val | Fast Abl. Ω | Confirmed? |
|------|---------|---------|--------|-------|-------------|-----------|
| 1 | `xwt_coherence_L2` | +0.892 | [+0.535,+1.311] | 0.000 | 1.617 | ❌ FLIP (keep) |
| 2 | `hht_imf_energy_entropy` | +0.866 | [+0.516,+1.266] | 0.000 | 1.591 | ❌ FLIP (keep) |
| 3 | `hht_amplitude_mod` | +0.817 | [+0.473,+1.203] | 0.000 | 1.542 | ❌ FLIP (keep) |
| 4 | `wt_phase_sin_L5` | +0.700 | [+0.325,+1.192] | 0.000 | 1.426 | ❌ FLIP (keep) |
| 5 | `xwt_coherence_L1` | +0.603 | [+0.269,+1.002] | 0.000 | 1.328 | ✅ Confirmed |
| 6 | `ou_speed` | +0.529 | [+0.191,+0.912] | 0.001 | 1.255 | ➡️ Neutral |
| 7 | `te_vol_to_price` | +0.361 | [+0.086,+0.662] | 0.003 | 1.086 | ⬇️ Tentative |
| 8 | `te_price_to_vol` | +0.302 | [+0.001,+0.616] | 0.025 | 1.027 | ❌ FLIP (keep) |
| 9 | `spectral_power_ratio` | +0.293 | [+0.026,+0.598] | 0.014 | 1.019 | ❌ FLIP (keep) |
| 10 | `xwt_phase_sin_L3` | +0.195 | [−0.091,+0.475] | 0.086 | 0.920 | ⏳ Pending |
| 11 | `hht_freq_shift` | +0.195 | [−0.045,+0.451] | 0.053 | 0.920 | ⏳ Pending |
| 12 | `cvd_fast_slow_div` | +0.188 | [−0.052,+0.453] | 0.061 | 0.913 | ⏳ Pending |
| 13 | `vol_regime_ratio` | +0.184 | [−0.063,+0.452] | 0.079 | 0.909 | ⏳ Pending |
| 14 | `wavelet_pe_L3` | +0.174 | [−0.105,+0.493] | 0.089 | 0.900 | ⏳ Pending |
| 15 | `te_asymmetry` | +0.107 | [−0.133,+0.381] | 0.218 | 0.832 | ⏳ Pending |
| 16 | `cvd_slow` | +0.106 | [−0.135,+0.353] | 0.202 | 0.831 | ⏳ Pending |

Neutral features (|ΔΩ| ≤ 0.02, fast-mode): `rogers_satchell_vol` (+0.017),
`wt_phase_sin_L2` (+0.016), `xwt_coherence_L5` (+0.015), `wavelet_energy_L2` (−0.002),
`pacf_lag4` (−0.007), `vol_ts_convexity` (−0.007), `xwt_phase_cos_L4` (−0.008).

---

### 21.2 Stage 2 — Full-Data Confirmatory Run (Authoritative)

**Date:** 2026-02-21 · **Protocol:** full bar history (up to 6 438 bars), 91 instruments, 400+150 episodes, seed=42, 1 000 bootstrap resamples
**Baseline:** Train Ω = 0.7247 · **Test Ω = 0.9019** (higher than fast-mode — more history = better signal)
**Output:** `results/feature_ablation_confirmatory/`

| Rank | Feature | Full-data ΔΩ | CI 95% [lo, hi] | p-val | Full Abl. Ω | Final Verdict |
|------|---------|-------------|-----------------|-------|-------------|---------------|
| 1 | `xwt_coherence_L1` | **+0.334** | [+0.086, +0.583] | 0.001 | 1.236 | ✅ **REMOVE — confirmed ★** |
| 2 | `te_vol_to_price` | +0.162 | [−0.047, +0.392] | 0.071 | 1.064 | ⚠️ Tentative — needs deeper check |
| 3 | `ou_speed` | +0.026 | [−0.171, +0.233] | 0.403 | 0.928 | ➡️ Neutral — keep for now |
| 4 | `te_price_to_vol` | −0.082 | [−0.301, +0.111] | 0.781 | 0.820 | 🔒 IMPORTANT_KEEP |
| 5 | `spectral_power_ratio` | −0.115 | [−0.285, +0.060] | 0.906 | 0.787 | 🔒 IMPORTANT_KEEP |
| 6 | `hht_amplitude_mod` | −0.190 | [−0.355, −0.010] | 0.984 | 0.712 | 🔒 IMPORTANT_KEEP ★ |
| 7 | `hht_imf_energy_entropy` | −0.202 | [−0.353, −0.044] | 0.992 | 0.700 | 🔒 IMPORTANT_KEEP ★ |
| 8 | `wt_phase_sin_L5` | −0.226 | [−0.405, −0.054] | 0.998 | 0.676 | 🔒 IMPORTANT_KEEP ★ |
| 9 | `xwt_coherence_L2` | −0.262 | [−0.432, −0.104] | 1.000 | 0.640 | 🔒 IMPORTANT_KEEP ★ |

**Confirmed action — `xwt_coherence_L1` only:**
This is the single feature confirmed for removal across both stages. It is a CANDIDATE
for the v2.3 canonical update. Before removal is locked in, run it as part of the
combination-removal test (§21.5 Priority 1) to confirm no cross-feature dependency.

---

### 21.3 Fast-Mode IMPORTANT_KEEP — Strongly Protected (authoritative, fast-mode only)

Features with ΔΩ ≤ −0.30 in fast-mode sweep. Not re-run in confirmatory (no reason to).
These are **protected**. Do not remove or gate without extraordinary evidence.

| Feature | Fast ΔΩ | Abl. Test Ω |
|---------|---------|-------------|
| `cvd_fast_energy` | −0.523 | 0.203 |
| `wt_phase_sin_L3` | −0.442 | 0.283 |
| `skewness` | −0.441 | 0.285 |
| `bvc_imbalance` | −0.436 | 0.289 |
| `wavelet_pe_L4` | −0.414 | 0.311 |
| `xwt_phase_cos_L3` | −0.389 | 0.336 |
| `vol_ts_32` | −0.385 | 0.341 |
| `xwt_phase_sin_L4` | −0.383 | 0.342 |
| `mfdfa_alpha_mean` | −0.338 | 0.388 |
| `fft_phase` | −0.333 | 0.392 |
| `wavelet_pe_L2` | −0.310 | 0.415 |
| `wavelet_energy_L5` | −0.308 | 0.417 |
| `xwt_coherence_L3` | −0.308 | 0.418 |
| `wavelet_energy_ratio` | −0.307 | 0.418 |
| `wt_phase_sin_L4` | −0.302 | 0.423 |

> `vol_ts_32` (ΔΩ = −0.385, p = 1.000, fast) is a v2.2 addition — fully validated.
> `skewness` and `bvc_imbalance` are among the strongest single protectors of OOS edge.
> The wavelet energy hierarchy (L2–L5) and XWT coherence (L3–L5 — note: L1 is the
> exception) together account for the deepest OOS degradation when removed.
> The multi-scale spectral structure is **load-bearing** — do not thin this group.

---

### 21.4 v2.2 Addition Audit (two-stage)

| Feature | Fast ΔΩ | Full-data ΔΩ | Final Status |
|---------|---------|-------------|--------------|
| `vol_ts_32` | −0.385 ★ (keep) | not re-run (protected) | ✅ Fully validated — keep |
| `vol_of_vol` | −0.230 (keep) | not re-run | ✅ Validated — keep |
| `te_asymmetry` | +0.107 (tentative) | not yet run | ⏳ Pending confirmatory |

> v2.2 is net-positive: 2 of 3 additions strongly validated. `te_asymmetry` is
> tentative globally but likely valuable in the energy category profile.
> Do not remove until a full-data per-category confirmatory run is done.

---

### 21.5 Ablation Next Steps — ⏸️ PAUSED (resume after §22 data fix is complete)

> **All ablation work is on hold until the data foundation is clean.**
> Ablation results computed on contaminated/duplicate instrument sets are unreliable.
> Resume in the order below once §22 sign-off is complete.

**Queued Priority A — v2.3 combination gate:**
Remove `xwt_coherence_L1` alone and alongside `te_vol_to_price` on full clean data.
Accept only if full-data ΔΩ > +0.02 with CI lower bound > 0.

```bash
python scripts/feature_ablation_sweep.py \
  --feature xwt_coherence_L1 --feature te_vol_to_price \
  --episodes-train 400 --episodes-test 150 \
  --outdir results/feature_ablation_v23_candidate
```

**Queued Priority B — Confirmatory runs for fast-mode tentatives (ranks 10–16):**
`xwt_phase_sin_L3`, `hht_freq_shift`, `cvd_fast_slow_div`, `vol_regime_ratio`,
`wavelet_pe_L3`, `te_asymmetry`, `cvd_slow` — full-data re-runs required.
Most expected to flip to KEEP given the 67% observed flip rate.

**Queued Priority C — Energy profile ablation:**
Isolate which of the 4 energy additions (`pacf_lag2`, `trend_quality`,
`pv_divergence`, `release_potential_v2`) drives the Ω=1.055 win.

**DO NOT start any of the above until `§22.5 Sign-off Checklist` is fully checked.**

---

## 22. DATA FOUNDATION FIX ⭐ CURRENT PRIORITY

> **This section is the active workstream.** All ablation and feature work is blocked
> on having a clean, correctly categorised instrument set.

---

### 22.0 Why This Matters

Every OOS result in §21 was computed on a contaminated instrument set:
- Crypto-classified instruments included a stock (ABNB) and cross-pairs (BTCBCH, BTCXAU…)
- Forex category contained 11 crypto/fiat pairs (BTCJPY, ETHEUR, SOLJPY, etc.)
- Six forex symbols existed both with and without broker suffix `+` (duplicate data)
- Indices had spot/futures duplicates (`GER40` vs `GER40ft`, etc.)
- Root-level orphan CSVs (old naming convention, not tracked in `available_data.json`)

**Impact:** Category-OOS comparisons and per-category feature profiles are only valid
once each category contains a clean, homogeneous, non-overlapping instrument set.

---

### 22.1 Canonical Instrument Sets (Target State)

#### Crypto — 7 clean USD-denominated spot pairs
| Keep | Reason |
|------|--------|
| `BTCUSD` | Primary crypto benchmark |
| `ETHUSD` | Second largest, different regime |
| `SOLUSD` | High-volatility alt |
| `ADAUSD` | Lower-volatility alt |
| `XRPUSD` | High-liquidity alt |
| `BNBUSD` | Exchange token, unique regime |
| `DOTUSD` | Protocol token, keep if bars ≥ MIN_BARS |
| `LTCUSD` | Keep if bars ≥ MIN_BARS — otherwise drop |

**Remove from crypto:**
- `ABNB` — Airbnb stock. Not crypto. Delete from crypto folder.
- `ADAJPY`, `BCHJPY` — crypto/fiat pairs duplicated in forex. Remove from crypto.
- `BTCBCH`, `BTCETH`, `BTCLTC`, `BTCXAU` — crypto cross-pairs. Remove.
- `ETHBCH`, `ETHLTC`, `ETHXAU` — crypto cross-pairs. Remove.
- `BTCEUR`, `BCHUSD` — non-USD quote. Remove (EUR-base adds FX noise).
- `BTCO`, `NETH25` — unknown/unclassified instruments. Investigate then remove if not canonical crypto.

#### Forex — clean major/minor/exotic FX pairs only
**Resolve broker-suffix duplicates** — keep the `+` version (more recent, more bars)
and delete the bare version for the 6 overlapping pairs:
`AUDUSD`, `EURUSD`, `GBPUSD`, `NZDUSD`, `USDCAD`, `USDJPY`

**Remove from forex:**
- All crypto/fiat pairs: `ADAJPY`, `BCHJPY`, `BTCEUR`, `BTCJPY`, `ETHEUR`,
  `ETHJPY`, `LTCJPY`, `SOLJPY`, `USDTJPY`, `XLMJPY`, `XRPJPY`
- `XAGAUD` — silver pair, belongs in metals
- `EURIBOR3M` — interest rate instrument, not a forex pair
- `USDX` — dollar index, belongs in indices

#### Metals — clean precious/industrial metals
**Resolve XAUUSD duplicates** — keep `XAUUSD+` (broker canonical), remove `XAUUSD.crp`.
Current set is otherwise clean: XAU*, XAG*, COPPER-C, XPDUSD, XPTUSD.

Add `XAGAUD` moved from forex.

#### Indices — resolve spot vs futures duplicates
**Policy: keep spot, remove futures (`ft` suffix)** unless the futures version has
significantly more bars. Affected pairs:
`DJ30/DJ30ft`, `FRA40/FRA40ft`, `GER40/GER40ft`, `NAS100/NAS100ft`,
`UK100/UK100ft`, `CHINA50/CHINA50ft`

Investigate `AVAUSD`, `BERAUSD`, `CHINAH`, `SGDJ`, `FRAS`, `EURIBOR3M`, `USDX` —
classify correctly or remove if not standard equity indices.

#### Orphan root-level CSVs
Move or delete the 6 × 4 = 24 old-format flat files in `data/master_standardized/`:
`AUDJPY_{H1,H4,M15,M30}.csv`, `AUDUSD_*`, `BTCJPY_*`, `ETHEUR_*`, `US30_*`, `XAUUSD_*`
These use old naming (no date-range suffix) and are not tracked in `available_data.json`.
Superseded versions exist in the correct category subfolders — delete the orphans.

---

### 22.2 Data Menu & Script Improvements Required

**Menu restructure completed 2026-02-22.** The data management menu now follows
strict pipeline-order steps with context-aware gates. Remaining gaps:

**Completed ✅**

| Item | What was done |
|------|--------------|
| Pipeline order enforced | Items renumbered into 7 stepped sections (ACQUIRE→VALIDATE→ORGANISE→SPECS→PREPARE→RANK→RESEARCH) |
| Denoise gated | Item 14 locked until Prepare Data (step 11) complete — cannot be run before physics features |
| Prepare Data description fixed | Removed false "60+ features" and "splits train/test" claims; shows only physics-justified outputs |
| Soft gates on Prepare | Contextual warnings when integrity check skipped or specs not polled; hard block on critical T1 issues |
| Pipeline progress banner | Step-by-step ✅/⚠️/○ status shown at top of menu on every redraw |
| `suggest_next_step()` updated | All option numbers and pipeline ordering updated to match new menu layout |
| Specs before discovery | `suggest_next_step()` now recommends polling specs before running Prime Discovery |
| `add_derived_features` removed | Magic-number TA derivatives (`MA_N`, `ATR_14`, etc.) purged from `parallel_data_prep.py` |
| `project_root` bug fixed | `parallel_data_prep.py` path was `parent.parent` (→ `scripts/`); corrected to `parent.parent.parent` |
| Denoise filter fallbacks | Hardcoded `21` and `51` fallback windows replaced with `max(5|7, len(prices)//10)` |

**Still required ⏳**

| Gap | Fix Needed |
|-----|-----------|
| No instrument curation / recategorisation step | Add Menu 2 item: "Curate Instrument Sets" (move/delete misclassified symbols, resolve `+` duplicates) |
| Consolidate runs dry-run only from menu | Allow live run with explicit confirm + atomic backup |
| Discover does not auto-refresh after consolidate/move | Chain `discover_data()` after any mutating operation |
| No bar-count summary per category | Add to "View Data Coverage" — show min/max/median bars per category |
| No duplicate detection | Add duplicate symbol checker (same base, different suffix) to integrity check |
| Crypto set is not enforced | `_discover_all_instruments()` in ablation sweep needs a `--category-filter` |
| `available_data.json` paths use old flat naming for some entries | Re-run discover after all curation to regenerate clean manifest |

---

### 22.3 Script Fixes Required

#### `scripts/discover_available_data.py`
- Must scan **only** the category subdirectories (`crypto/`, `forex/`, `metals/`, `indices/`, `energy/`)
- Must skip orphan root-level CSVs (already not tracked — confirm this stays the case)
- Add `--validate` flag: report duplicates, misclassified symbols, short-history instruments

#### `scripts/feature_ablation_sweep.py`
- `_discover_all_instruments()` currently reads from `available_data.json`
- After curation, re-run discover to regenerate the manifest — then ablation automatically uses the clean set
- No code change needed in ablation script itself; the manifest is the single source of truth

#### `scripts/category_oos_comparison.py`
- Verify it routes correctly after manifest regeneration
- After crypto set is fixed, re-run `--category crypto` to get a clean crypto baseline

---

### 22.4 Execution Order

```
Step 1  Backup current data/master_standardized/ to data/backups/pre_curation_<date>/
Step 2  Delete orphan root-level CSVs (24 files, superseded)
Step 3  Crypto curation: remove ABNB, cross-pairs, non-USD quotes
Step 4  Forex curation: remove crypto/fiat pairs, remove non-forex instruments,
        resolve + duplicates (delete bare versions of the 6 overlap pairs)
Step 5  Metals curation: remove XAUUSD.crp, add XAGAUD (moved from forex)
Step 6  Indices curation: decide spot vs ft for 6 duplicate pairs, investigate unknowns
Step 7  Run scripts/discover_available_data.py to regenerate available_data.json
Step 8  Verify counts: crypto ≥ 6, forex ≥ 40, metals ≥ 8, indices ≥ 12, energy ≥ 8
Step 9  Run scripts/audit_data_coverage.py — confirm no NaN-heavy or sub-MIN_BARS files
Step 10 Re-run category_oos_comparison.py --category crypto (new clean baseline)
Step 11 §22.5 sign-off — then unblock §21.5 ablation queue
```

---

### 22.5 Sign-off Checklist (required before resuming ablation)

- [ ] No orphan CSVs in `data/master_standardized/` root
- [ ] Crypto: only USD-denominated spot pairs, no stocks, no cross-pairs
- [ ] Forex: no crypto/fiat pairs, no interest-rate instruments, + duplicates resolved
- [ ] Metals: XAUUSD deduplicated, XAGAUD present
- [ ] Indices: no spot/futures duplicates for the same underlying
- [ ] `available_data.json` regenerated and verified clean
- [ ] Bar counts per category meet MIN_BARS = 600 threshold for all H4 instruments
- [ ] `kinetra_menu.py` data management section updated with curation tooling (see §22.2)
- [ ] Crypto category OOS baseline re-run with clean set

---

## 23. DATA PIPELINE ARCHITECTURE

### 23.1 Canonical Pipeline Order (enforced by menu)

> **⚠️ SUPERSEDED by §25.2** — The pipeline order below reflects the pre-2026-02-25
> menu layout. It has critical sequence inversions (Prepare before Scientific
> Discovery, no curation step, no hypothesis gates). See §25 for the corrected
> 6-phase pipeline with explicit gates. This section is retained for historical
> reference only — do NOT use it as the implementation target.

```
STEP 1  ACQUIRE      (Menu 2 › 1-4)
  1. Discover Available Data     — scan filesystem, build available_data.json
  2. Download Data via MetaAPI   — fetch OHLCV history for all candidate instruments
  3. Download Data via MT5       — alternative: pull from local MT5 terminal
  4. Fill Data Gaps              — re-download missing bars only

STEP 2  VALIDATE     (Menu 2 › 5-6)
  5. Check Data Integrity        — gaps, OHLC validity, dupes, bar counts
                                   ⚠️  Run BEFORE Prepare — corrupt data → corrupt features
  6. View Data Coverage          — bar counts per symbol / TF

STEP 3  ORGANISE     (Menu 2 › 7-8)
  7. Consolidate & Clean Data    — merge dupes, standardize naming
  8. Reorganize into Folders     — per-symbol dirs + contract_spec.json

STEP 4  BROKER SPECS (Menu 2 › 9-10)
  9. Poll Symbol Specs           — fetch spread/swap/commission from broker
                                   ⚠️  Run BEFORE Prime Discovery — needed for friction score
 10. List Broker Symbols         — diagnose NotFoundException

STEP 5  PREPARE      (Menu 2 › 11)
 11. Prepare Data                — OHLCV → core physics features only (see §2.10)
                                   Soft-warns if steps 5 or 9 were skipped
                                   Hard-blocks if critical T1 integrity issues exist

STEP 6  RANK         (Menu 2 › 12-13 / Menu 3 › 0-1)
 12. Prime Instrument Discovery  — rolling wavelet + friction → Tier-1 JSON
 13. Wavelet Feature Pipeline    — DWT db4, 6 levels, 512-bar lookback on Tier-1

SUPPLEMENTARY RESEARCH (Menu 2 › 14) — LOCKED until step 11 complete
 14. Denoise Data (Research)     — compare denoised vs raw as ablation inputs only
                                   NEVER use as a replacement for raw physics pipeline
```

### 23.2 What Prepare Data Does NOT Do

| Claim (old) | Reality | Why removed |
|---|---|---|
| "Splits into train/test sets" | Not done at prepare time | Split is a deliberate decision made at training time, not buried in file prep |
| "60+ physics features" | ~20 core physics features | The extra ~40 were magic-number TA derivatives now removed (§2.10) |
| Fixed rolling windows [6,12,24,48] | Removed | Magic numbers — captured adaptively by PhysicsEngine |
| `ATR_14` | Removed | Classic TA magic number |
| `vol_spike` (×2 threshold) | Removed | Fixed multiplier = static threshold (§2.1 violation) |

### 23.3 Denoising Workflow (Research Only)

The `denoise_filters.py` module provides four non-linear methods:

| Method | Window selection | Status |
|---|---|---|
| Savitzky-Golay | FFT dominant cycle; fallback `max(7, n//10)` | ✅ Adaptive |
| Median | FFT dominant cycle; fallback `max(5, n//10)` (odd) | ✅ Adaptive |
| LOWESS | `frac=0.05` of series length | ✅ Adaptive |
| Wavelet (db4) | Level = `min(5, log2(n)−1)`; threshold = σ√(2 log n) | ✅ Principled (Donoho & Johnstone universal threshold) |

**All fallbacks are data-adaptive (scale with series length). No magic numbers.**

Correct usage:
```
physics features (from raw OHLCV)  ← canonical pipeline
        ↓
additive testing: does denoised input improve OOS Ω over raw input?
        ↓
if yes → add denoised version as an additional feature, not a replacement
```

---

**This is the canonical rulebook. All rules consolidated. Single source of truth.** 🚀

---

## 24. KNOWN TECH-DEBT (Audit 2026-02-23)

This section records gaps identified during a full say-vs-do audit.
Items marked ✅ FIXED are already remediated. Items marked ⏳ OPEN are tracked
here so they are not forgotten and are addressed incrementally.

### 24.1 Remediated (✅ FIXED)

Items below were fixed across two audit sprints (2026-02-23).

| # | File(s) | Issue | Fix Applied |
|---|---|---|---|
| T-01 | `pyproject.toml` | `make test` crashed with `ModuleNotFoundError: No module named 'kinetra'` — no `pythonpath` in pytest config | Added `pythonpath = ["."]` to `[tool.pytest.ini_options]` |
| T-02 | `kinetra/integrated_backtester.py` | Win-acceptance gate used `p < 0.05` instead of `p < 0.01` (§9.1) | Changed to `p < 0.01` |
| T-03 | `kinetra/results_analyzer.py` | Winner selection and result filtering used `p < 0.05` (§9.1) | Changed all win-gate comparisons to `p < 0.01`; chart label for orange bar annotated as "marginal — does NOT meet §9.1 win gate" |
| T-04 | `kinetra/discovery_methods.py` | Latent-dimension significance gate at `p < 0.05`; adversarial survival gate at `p < 0.05` (§9.1) | Both gates tightened to `p < 0.01` |
| T-05 | `kinetra/testing_framework.py` | `test_asymmetry` used `p < 0.05` — ambiguous as to whether it was a win gate | Added explicit comment: this is a distributional-difference detection test (KS), NOT a §9.1 win-acceptance gate; the `p < 0.05` threshold is intentional for this use |
| T-06 | `kinetra/trading_env.py` | `_compute_atr(14)` — magic-number ATR window (§2.2 / §2.4) | Replaced with `_compute_adaptive_volatility()`: window derived from 1/e autocorrelation decay lag of true range, bounded to [10, min(200, n//5)] |
| T-07 | `kinetra/trading_env.py` | `_normalize_features()` listed dead feature names (`momentum_5`, `momentum_20`, `energy_ma5`, `energy_ma20`, `energy_std5`) left over from `add_derived_features()` removal — silent no-ops | Removed dead names from normalization list |
| T-08 | `kinetra/realistic_trading_env.py` | `_compute_atr(14)` and `_compute_bb_width(20, 2.0)` — magic-number windows for ATR and Bollinger (§2.2 / §2.4) | Replaced with `_compute_adaptive_atr()` and `_compute_adaptive_band_width()`: both use the `_adaptive_window()` helper (1/e ACF decay lag, bounded to [10, 200]); BB ±2σ fixed multiplier replaced by dimensionless 2σ/μ dispersion ratio |
| T-09 | `scripts/download/parallel_data_prep.py` | Docstring still claimed "60+ measurements" after `add_derived_features()` removal (§2.10) | Updated docstring to accurately describe PhysicsEngine output columns and reference §2.10 |
| T-10 | `kinetra/high_performance_engine.py` | RSI(14), MACD(12/26/9), SMA(20), EMA(20), BB(20) with fixed magic-period defaults; `momentum_strategy`/`trend_strategy`/`volatility_strategy` used hardcoded TA thresholds (§2.2) | Full TA implementation archived to `archive/legacy-scripts/high_performance_engine.py`. Production shim emits `DeprecationWarning` on import; TA classes replaced by `ImportError` tombstones; data-plumbing dataclasses (`TickData`, `BarData`, `Signal`) retained for backward compat |
| T-11 | `kinetra/data_manager.py:434` | `rolling(20)` hardcoded for volatility computation (§2.4) | Replaced with `rolling_volatility(data["close"].values)` from `kinetra.volatility_utils` — window derived adaptively from ACF 1/e decay of \|log-returns\| |
| T-12 | `kinetra/rl_gpu_trainer.py:310-311` | `rolling(50)` for phase-space price/momentum normalisation (§2.4) | `_phase_window = adaptive_atr_window(df["close"].values)` — window now derived from the data's own volatility memory; imported from `kinetra.volatility_utils` |
| T-13 | `kinetra/stress_test.py:583` | `rolling(20)` for ATR approximation in gap injection (§2.4) | Replaced with `adaptive_atr(high, low, close)` from `kinetra.volatility_utils`; NaN fallback to `\|high-low\|` for early bars |
| T-14 | `kinetra/outcome_labeller.py` | `DEFAULT_ATR_WINDOW = 14` hardcoded magic number (§2.4) | Changed `DEFAULT_ATR_WINDOW` to `None`; `compute_atr()` and `ForwardOutcomeLabeller` now accept `window=None` → resolved adaptively via `adaptive_atr_window()` from `kinetra.volatility_utils`; explicit `atr_window=14` still works for reproducibility |
| T-15 | 9 modules (`discovery_methods`, `integrated_backtester`, `test_executor`, `unified_data_manager`, `testing_framework`, `doppelganger_triad`, `portfolio_health`, `live_dashboard`, `exploration_lab/orchestrator`) | Module-level `logging.basicConfig(...)` calls hijacked the root logger for the entire process (§11.1) | Created `kinetra/log_config.py` with `JSONFormatter`, `PlainTextFormatter`, and `configure_logging()` (idempotent, env-var overridable). Removed all 9 `basicConfig` calls: 5 production modules now use `logging.getLogger(__name__)` only; 4 script/`__main__` blocks call `configure_logging()`. `kinetra_menu.py` `main()` calls `configure_logging(json=False)` at startup |
| T-20 | `outcome_labeller`, `physics_backtester`, `runway_engine`, `stress_test`, `data_manager` | Five duplicate `compute_atr` implementations scattered across the codebase (§13.3) | Extracted canonical implementations to `kinetra/volatility_utils.py`: `compute_tr()`, `rolling_atr()`, `wilder_atr()`, `adaptive_atr_window()`, `adaptive_atr()`, `rolling_volatility()`. Callers for T-11–T-14 updated to import from there; existing callers in `physics_backtester` and `runway_engine` tracked for migration (T-20b — LOW, next touch) |
| T-20b | `kinetra/physics_backtester.py`, `kinetra/runway_engine.py` | Both modules still carried their own `compute_atr` / `_compute_atr` duplicates predating `volatility_utils` (§13.3) | `physics_backtester.compute_atr()` now delegates to `wilder_atr` from `kinetra.volatility_utils` (import alias `_wilder_atr`); NaN prefix filled for Backtesting.py compatibility. `runway_engine._compute_atr()` now delegates to `wilder_atr` from `kinetra.volatility_utils` (import alias `_volatility_wilder_atr`). Both local wrappers retained for backward-compat call-site stability. |
| T-16 | `kinetra/persistence_manager.py`, `kinetra/data_manager.py`, `kinetra/unified_data_manager.py`, `kinetra/spread_gate.py`, `kinetra/trigger_predictor.py`, `kinetra/data_discovery.py` | `print()` calls in production package code (should be structured logging) — hot-path modules addressed first (§11.1) | Added `import logging` + `logger = logging.getLogger(__name__)` to modules that lacked it. Converted all `print()` calls in production methods to `logger.info/debug/warning/error`. `persistence_manager` — all 12 prints in `atomic_save`, `restore_latest`, `cleanup_old_backups` converted. `data_manager` — all 32 prints in `prepare_training_data`, `create_backup`, `restore_backup` converted. `unified_data_manager.print_summary()` → delegates to new `log_summary()` structured method. `spread_gate`, `trigger_predictor`, `data_discovery` — CLI/`__main__` and validation-function prints converted. Core hot-path modules (`physics_engine`, `trading_env`, `backtest_engine`) were already clean. |
| T-17 | `kinetra/results_manager.py`, `kinetra/test_executor.py`, `kinetra/integrated_backtester.py`, `kinetra/unified_data_manager.py` | Raw `.to_csv()` / `.to_parquet()` / `open(..., 'w')` writes that bypass atomic safety (§3.1) | All writes in these four modules now go through `kinetra.data.atomic_ops.atomic_write()` (temp-file + atomic rename). `results_manager`: config.json, summary.json, trades.parquet, equity_curve.parquet. `test_executor`: checkpoint.json, results JSON. `integrated_backtester`: result JSON, backtest report TXT. `unified_data_manager`: integrity report JSON, CSV/parquet export. |
| T-21 | `tests/test_menu_system.py`, `tests/test_menu_workflow.py`, `tests/test_live_testing_integration.py`, `tests/test_system_stress.py` | Test functions returned non-None values (`True`/`False`/`Dict`) causing 50 `PytestReturnNotNoneWarning` per run (§16.1) | Removed all `return True/False` from test function bodies; changed `except: return False` blocks to `raise` so exceptions propagate to pytest. Tests that depend on `e2e_testing_framework` (unavailable) now use `pytest.importorskip`. Tests that tested a now-removed/renamed API (`show_main_menu`, `MenuConfig`, `show_live_testing_menu`, etc.) now carry `@pytest.mark.xfail(strict=False, reason=...)` documenting the pre-existing API mismatch. Result: **0 `PytestReturnNotNoneWarning`**, 517 passed, 76 skipped, 16 xfailed (all xfails are pre-existing failures now properly classified). |

### 24.2 Open Tech-Debt (⏳ OPEN — address incrementally)

#### MEDIUM priority

| # | File(s) | Issue | §Rule | Remediation Plan |
|---|---|---|---|---|
| T-16 (remainder) | `kinetra/mt5_bridge.py` (62), `kinetra/menu_ux.py` (59), `kinetra/cpu_utils.py` (40), `kinetra/parallel.py` (36), `kinetra/doppelganger_triad.py` (26), `kinetra/portfolio_health.py` (21), `kinetra/mt5_live.py` (20) and others | ~400 remaining `print()` calls in interactive/UI/live-trading modules not yet converted | §11.1 | Convert incrementally; `menu_ux.py` interactive prompts may intentionally use `print` for console UX — evaluate case by case |
| T-17 (remainder) | `scripts/` directory, `kinetra/devops/`, `kinetra/exploration_lab/orchestrator.py`, `kinetra/silent_failure_logger.py`, `kinetra/market_calendar.py`, `kinetra/testing_framework.py` | Remaining raw file writes in scripts and secondary modules | §3.1 | Migrate scripts on next touch; scripts are lower priority than core modules |
| T-18 | Throughout `kinetra/` (~1219 functions) | ~46 % of functions (those without `->` return type annotation) lack return type hints | §14.1 | Incremental: add return types starting from public API surface of `physics_engine.py`, `backtest_engine.py`, `trading_env.py` |

#### LOW priority (style / maintenance)

| # | File(s) | Issue | §Rule | Remediation Plan |
|---|---|---|---|---|
| T-19 | `kinetra/dsp_features.py`, `kinetra/discovery_methods.py` | `for i in range(len(...))` explicit index loops where `enumerate` or direct array ops would be cleaner and faster | §4.1 | Vectorise on next touch of these files |
| T-22 | `tests/test_menu_workflow.py`, `tests/test_live_testing_integration.py`, `tests/test_system_stress.py`, `tests/test_menu_system.py` | 16 tests marked `xfail` because they test a removed/renamed API (`show_main_menu` → `print_main_menu`, `MenuConfig`, live-testing functions). Pre-existing bugs now properly documented. | §16.1 | Update tests to match current `kinetra_menu` API; implement or stub missing functions (`show_live_testing_menu`, `run_virtual_trading`, `MenuConfig`) |

### 24.3 New Modules Added (sprints 2–3)

| Module | Purpose |
|---|---|
| `kinetra/volatility_utils.py` | Canonical volatility helpers: `compute_tr`, `rolling_atr`, `wilder_atr`, `adaptive_atr_window` (ACF 1/e decay), `adaptive_atr`, `rolling_volatility`. Single source of truth replacing 5 duplicated `compute_atr` implementations. |
| `kinetra/log_config.py` | Centralised logging configuration: `JSONFormatter`, `PlainTextFormatter`, `configure_logging()` (idempotent, env-var driven), `get_logger()`. Replaces all `logging.basicConfig` calls. Entry-points call `configure_logging()` once at startup. |

### 24.4 Sprint 3 Changes (2026-02-24)

| Item | Files Changed | Summary |
|---|---|---|
| T-20b | `kinetra/runway_engine.py`, `kinetra/physics_backtester.py` | Both delegated to `volatility_utils.wilder_atr`; local wrappers retained as thin shims |
| T-16 (hot-path) | `kinetra/persistence_manager.py`, `kinetra/data_manager.py`, `kinetra/unified_data_manager.py`, `kinetra/spread_gate.py`, `kinetra/trigger_predictor.py`, `kinetra/data_discovery.py` | Added `logging` infrastructure and converted all production `print()` to `logger.*` |
| T-17 (core) | `kinetra/results_manager.py`, `kinetra/test_executor.py`, `kinetra/integrated_backtester.py`, `kinetra/unified_data_manager.py` | All data writes now go through `kinetra.data.atomic_ops.atomic_write()` |
| T-21 | `tests/test_menu_system.py`, `tests/test_menu_workflow.py`, `tests/test_live_testing_integration.py`, `tests/test_system_stress.py` | Removed all `return True/False`; added `pytest.importorskip` for missing `e2e_testing_framework`; added `@pytest.mark.xfail` for 16 tests that test non-existent API |
| Ruff | All modified files | Zero violations maintained throughout |
| Tests | Full suite | **517 passed, 76 skipped, 16 xfailed, 0 failed, 26 warnings** (down from 50 warnings) |

### 24.4 Audit Methodology

The audit compared AGENT_RULES_MASTER.md (this document) against the actual
codebase by:

1. Running `ruff check .` — confirmed zero violations pre-audit and post-fix.
2. Running `pytest tests/ -q` — confirmed 536 passed, 73 skipped, 0 failed post-fix.
3. Grepping for: magic-number rolling windows, `p < 0.05` significance gates,
   `logging.basicConfig`, `print(`, raw file writes, TA indicator names
   (ATR, RSI, MACD, Bollinger, SMA, EMA).
4. Verifying `pythonpath` in pytest config and package discoverability.
5. Reviewing `_normalize_features()` for dead/stale feature references.
6. Sprint-2 additions: smoke-tested `volatility_utils` ACF adaptive window on
   synthetic GBM data (n=500); confirmed all finite, NaN-shielded, window in
   [5, 252]; confirmed `ForwardOutcomeLabeller(atr_window=None)` (adaptive
   default) and `ForwardOutcomeLabeller(atr_window=14)` (legacy fixed) both
   produce correct 40-column DataFrames.

All findings are recorded above; none are silently discarded.

---

## 25. MENU RESTRUCTURE — PIPELINE CORRECTION ✅ COMPLETE

> **Canonical plan:** [`../repo/docs/WORKFLOW.md`](../repo/docs/WORKFLOW.md)
>
> This section summarises the findings and corrected pipeline. The plan document
> contains full file-level change manifests, gate definitions, implementation
> phases, and sign-off checklists. Read it before starting implementation.

### 25.0 Why This Restructure Is Needed

A critical review of `kinetra_menu.py` (2026-02-25) found that the pipeline
**assumes its way to alpha instead of testing its way there**. The menu is
well-built as a UI (context badges, progress banners, gating) but has deep
structural problems in the scientific pipeline it enforces:

1. Physics features are computed on ALL data BEFORE asking "is there structure?"
2. Instrument curation (§22, the declared #1 priority) has zero menu tooling
3. A hardcoded 8-instrument Tier-1 universe (4 gold pairs) violates §2.1 and §2.8
4. The pipeline pushes forward even when every signal says "stop — no alpha"
5. Specialist training code exists but is unreachable from the menu
6. Backtesting options are half stubs ("under development")
7. No hypothesis validation gates between pipeline phases

### 25.1 Current Pipeline Order (Broken)

```
 1  Credentials → 2 Discover → 3 Download → 4 Integrity → 5 Specs
 6  Prepare ALL data (blanket)          ← WRONG: before structure discovery
 7  Scientific Discovery                ← WRONG: after prepare
 8  Prime Discovery → 9 Wavelet → 10 Additive → 11 Agent Comparison
12  Train → 13 Backtest
```

**Key inversions:**
- Step 6 (Prepare) runs on ALL ~878 files before step 7 (Scientific Discovery)
  determines which instruments have exploitable structure
- No curation step between download and processing — contaminated data flows through
- Scientific Discovery (step 7) is optional — can be skipped entirely
- No gate checks whether any step actually found something before proceeding

### 25.2 Corrected Pipeline Order (Target State)

```
PHASE 1: DATA FOUNDATION  (Menu 2)
  1  Credentials
  2  Discover data on filesystem
  3  Download raw OHLCV (broad universe, H4 primary)
  4  Validate integrity
  5  ★ Curate instrument sets (§22 — NEW)
  6  Poll broker specs
  7  Organise per-symbol folders
  ── GATE 1: Clean data foundation ──
     All curated, integrity OK, MIN_BARS met, manifest regenerated

PHASE 2: STRUCTURE DISCOVERY  (Menu 3, Part A — on RAW OHLCV)
  8  Comprehensive Exploration (ELEVATED from "supplementary")
     → What does "trend" / "reversion" mean per class? From DATA.
  9  Scientific Discovery (PCA/ICA/DFA/PE/null-hypothesis)
     → Is there non-random structure? Which instruments pass null test?
 10  Prime Instrument Discovery (wavelet persistence + friction)
     → Tier-1 / Tier-2 / Skip — informed by step 9 results
  ── GATE 2: Structure confirmed ──
     Sci Disc run, ≥1 instrument passes null test, Tier-1 set non-empty
     If zero qualify → STOP. Do not prepare data for random walks.

PHASE 3: TARGETED PREPARATION  (Menu 2, gated on Phase 2)
 11  Prepare Data — ONLY Tier-1/Tier-2 qualifying instruments
 12  ★ Prepared-data quality audit (NEW)
  ── GATE 3: Prepared data valid ──
     Zero NaN/Inf, no degenerate features, regimes differentiate behavior
     ⚠️  If wavelet features already exist, Prepare is not required
        (wavelet operates on raw H4 CSVs, not prepared data)

PHASE 4: FEATURE VALIDATION  (Menu 3, Part B)
 13  Wavelet Feature Pipeline (band order data-derived, not hardcoded)
       ⚠️  Operates on RAW H4 CSVs in master_standardized/, NOT prepared data
       Gate: has_t1_data AND (has_prepared OR has_wavelet)
 14  Additive Feature Testing (D2 baseline → +bands, Omega-gated)
 15  Feature Ablation Sweep (optional — §20.5 protocol)
  ── GATE 4: Features carry signal ──
     D2 baseline Omega > 1.0 after ≥200 episodes
     If Omega < 1.0 → WARN (investigate, don't blindly proceed)

PHASE 5: AGENT SELECTION  (Menu 3, Part C)
 16  Agent Comparison (TQC vs DreamerV3 vs regime-aware B&H)
  ── GATE 5: Agent beats baseline ──
     ≥1 agent beats B&H on ≥70% of Tier-1 instruments after costs
     If none beat baseline → no learnable edge in current feature set
 17  Train winner(s) — production config (500+ episodes)

PHASE 6: VALIDATION  (Menu 4)
 18  Walk-forward analysis (temporal stability)
 19  Monte Carlo validation (100+ runs, p < 0.01)
 20  OOS backtest with full friction model
  ── GATE 6: Validated alpha (§9.1 — ALL THREE required) ──
     OOS reported, robustness checks passed, confidence intervals included
```

### 25.3 Critical Findings Summary

| ID | Severity | Finding | Location |
|----|----------|---------|----------|
| C1 | 🚨 | Prepare Data before Scientific Discovery — total inversion | `suggest_next_step()` L596–610 |
| C2 | 🚨 | No instrument curation step — §22 not implemented | Menu 2 — missing entirely |
| C3 | 🚨 | Hardcoded gold-biased Tier-1 universe (4/8 gold) | `_TIER1_UNIVERSE_DEFAULT` L206–303 |
| C4 | 🚨 | Wavelet additive order hardcoded, not data-derived | `WAVELET_ADDITIVE_ORDER` L354–380 |
| C5 | 🔴 | Scientific Discovery optional and bypassable | `suggest_next_step()` L633–661 |
| C6 | 🔴 | No failure handling — pipeline never says "stop" | `suggest_next_step()` L582–726 |
| C7 | 🔴 | Comprehensive Exploration misplaced as "supplementary" | `menu_exploration_training()` L2634 |
| C8 | 🔴 | No prepared-data quality audit | `prepare_data()` L1931–2008 |
| C9 | 🟠 | Denoise outputs to wrong directory (`data/prepared/`) | `denoise_data()` L2540 |
| C10 | 🟠 | `train_specialists()` unreachable from menu | L3676–3765 — not wired |
| C11 | 🟠 | Backtesting stubs: walk-forward + report = "under development" | L4187–4239 |
| C12 | 🟠 | Status badges track completion, not quality | `SystemStatus` L382–478 |

### 25.4 Hardcoded Assumptions to Remove

| Item | Location | Fix |
|------|----------|-----|
| 8-instrument Tier-1 default (4 gold) | L206–303 | Remove; require discovery |
| Wavelet band order D2→D3→…→D1 | L354–380 | Read from analysis output |
| Backtest years `2023 2024` | L4127 | Derive from data on disk |
| MC runs = 50 | L4159 | Change to 100 (§16.1) |
| MC instruments = `available_symbols[:3]` | L4180 | Use Tier-1 |
| Denoise output `data/prepared/` | L2540 | `data/prepared_standardized/denoised/` |
| Prepared threshold `>= 0.5` of total | L847 | Based on Tier-1 coverage |

### 25.5 New Status Fields Required

Add to `SystemStatus` dataclass to support quality-aware gating:

```python
# Phase gates
instruments_curated: bool = False
available_data_regenerated: bool = False

# Scientific Discovery quality (not just "has it run?")
scientific_discovery_has_structure: Optional[bool] = None
scientific_discovery_n_structured: int = 0
scientific_discovery_n_non_random: int = 0

# Prepared-data audit
prepared_data_audited: bool = False
prepared_data_quality_ok: bool = False

# Agent comparison results
agent_comparison_done: bool = False
agent_beats_baseline: Optional[bool] = None
agent_best_name: Optional[str] = None
agent_best_omega: Optional[float] = None
```

`check_system_status()` must read result JSON files to populate these — not just
check whether the JSON exists.

### 25.6 New Files Required

| File | Purpose |
|------|---------|
| `scripts/data/curate_instruments.py` | Instrument curation — implements §22 |
| `scripts/data/audit_prepared_data.py` | Post-prepare quality audit |

### 25.7 Implementation Phases

Implement in this order — each phase is independently testable:

**Phase A — Foundation Fixes** (unblocks everything else):
Remove hardcoded Tier-1 default, fix MC runs/years/denoise path, handle empty
universe. These are small, independent, zero-risk changes.

**Phase B — Pipeline Reordering** (core restructure):
Rewrite `suggest_next_step()`, restructure Menu 2 and Menu 3, add curation step,
gate Prepare on Phase 2, add instrument filtering to prepare scripts.

**Phase C — Gate System** (failure-aware pipeline):
Add quality fields to `SystemStatus`, read result quality from JSON files, make
`suggest_next_step()` branch on results (warn when no structure / low Omega / no
agent edge).

**Phase D — Cleanup** (stubs, dead code, polish):
Wire `train_specialists()`, grey out stubs, remove Menu 2/3 duplication for
ranking steps, update progress banners.

### 25.8 Key Principle

> **We are building a HYPOTHESIS-TESTING pipeline, not an ASSUMPTION pipeline.**
>
> Every phase transition must answer **"did we find something?"** — not just
> **"did we run the script?"**
>
> A pipeline that reaches Phase 6 without ever confirming structure exists
> has validated nothing. The gates exist to catch this.


## 26. GATE-LOGIC BUGFIXES (2026-02-27)

Three bugs discovered when exercising the live menu against the actual repository
state (additive step 2/7, 210 historical `.joblib` artifacts, wavelet features
computed from raw H4 CSVs without Prepare Data having been run).

### 26.1 BUG 1 — `models_trained` poisoned by historical artifacts (HIGH)

**Root cause:** `check_system_status()` sets `models_trained=True` whenever any
model file (`.joblib`, `.pt`, `.pkl`, `.zip`) exists in `models/`, `checkpoints/`,
or `data/runs/`. 210 old `.joblib` files from a prior pipeline run caused this
flag to be `True` even though the current pipeline was only at additive step 2/7
and had never run agent comparison.

**Three cascading effects fixed:**

| Location | Before | After |
|----------|--------|-------|
| Main menu explore badge (`print_main_menu`) | `✅` (short-circuits all progress branches) | `⚠️ (additive step 2/7 Ω=4.40)` or `⚠️ (stale models — re-run pipeline)` |
| Menu 3 pipeline banner `_ps6` | `⚪Compare ✅Trained` (impossible state) | `⚪Compare ○Trained` (inactive — compare not done) |
| Menu 4 backtest gates (`menu_backtesting`) | All 4 options unlocked | All 4 options locked until `agent_comparison_done` |

**Fix:** All UI gates now require `agent_comparison_done AND models_trained`
instead of `models_trained` alone. The filesystem scan is unchanged (it still
detects all model files), but the flag is no longer trusted alone for gate
decisions.

### 26.2 BUG 2 — Wavelet gate too strict (MEDIUM)

**Root cause:** `_wavelet_available = has_t1_data and has_prepared` required
prepared data, but `run_wavelet_analysis.py` operates on **raw H4 CSVs** in
`master_standardized/`, not on prepared data. When `data_prepared=False` but
wavelet features already existed on disk (71/81 computed), the wavelet pipeline
was locked.

**Fix:** `_wavelet_available = has_t1_data and (has_prepared or has_wavelet)`.
Lock reason text restructured for clarity.

### 26.3 BUG 3 — Pipeline order: Prepare vs Wavelet (DESIGN)

**Root cause:** `suggest_next_step()` suggested "Prepare Data" even when wavelet
features already existed on disk. The audit suggestion also fired when
`data_prepared=False` (nothing to audit).

**Fix (two locations in `suggest_next_step()`):**
1. Prepare suggestion guarded: `if not self.data_prepared and not self.wavelet_features_ready:`
2. Audit suggestion guarded: `if self.data_prepared and not self.prepared_data_audited:`

### 26.4 Regression Tests

12 new tests in `tests/test_menu_gates.py`:

| Class | Tests | Covers |
|-------|-------|--------|
| `TestStaleModelsDoNotPoisonBadges` | 6 | Stale models don't unlock ✅ badge, don't green pipeline banner, don't unlock backtest; legitimate trained path still works; stale warning shown when no additive progress |
| `TestWaveletGateRelaxed` | 4 | Wavelet available with existing features, locked without both, still requires Tier-1 data, normal prepared-data path still works |
| `TestPipelineOrderWaveletVsPrepare` | 2 | Suggest additive (not prepare) when wavelet done without prepare; don't regress to Phase 3 when additive is active |

**Total test suite:** 175 passed (86 unit + 89 integration), ruff zero violations.

### 26.5 Key Invariants (enforced by tests)

These invariants MUST be maintained in all future changes:

1. **`models_trained` alone NEVER unlocks UI badges or backtest gates** — always
   requires `agent_comparison_done` as a co-condition.
2. **Wavelet pipeline availability** = `has_t1_data AND (has_prepared OR has_wavelet)` —
   wavelet operates on raw H4 CSVs, not prepared data.
3. **`suggest_next_step()` skips "Prepare Data"** when `wavelet_features_ready=True`.
4. **Audit suggestion only fires** when `data_prepared=True` — no audit if nothing
   to audit.

---

## 27. DRY VIOLATIONS REGISTER (2026-03-01)

> **Canonical plan:** [`../repo/docs/HOUSEKEEPING_AUDIT.md`](../repo/docs/HOUSEKEEPING_AUDIT.md)
>
> This section summarises the DRY (Don't Repeat Yourself) audit findings and
> the governing rules. The plan document contains the full per-violation
> breakdown, canonical home locations, and acceptance criteria.

### 27.0 DRY Rules — What Agents Must Follow

**NEVER:**
- ❌ Copy-paste a helper function into a new file — find or create the shared module
- ❌ Redefine a constant that already exists elsewhere (e.g. `MAX_FILL_BARS`)
- ❌ Add a standalone `omega_ratio()` / `z_factor()` / ATR function to a script —
  import from `kinetra.backtesting.metrics` or `kinetra.volatility_utils`
- ❌ Add blacklist / gap-scan logic inline in a new script — import from
  `kinetra.data_gap_tools`
- ❌ Define `_detect_agent_comparison()` or the wavelet step reader again —
  use `kinetra.model_manifest.detect_agent_comparison()` and
  `kinetra.model_manifest.read_wavelet_step()`

**ALWAYS:**
- ✅ Before writing a new utility function, `grep` for existing implementations
- ✅ When adding a function used by 2+ callers, put it in a `kinetra/` library
  module — not inline in a script or `kinetra_menu.py`
- ✅ Import constants from their authoritative source; never redeclare with a
  "keep in sync" comment
- ✅ Check `../repo/docs/HOUSEKEEPING_AUDIT.md` before touching any of the open items
  (DRY-01 through DRY-15) to avoid making the duplication worse

### 27.1 Canonical Module Locations (Single Source of Truth)

| Concept | Canonical Module | Status |
|---------|-----------------|--------|
| Agent comparison detection | `kinetra.model_manifest.detect_agent_comparison()` | ✅ EXISTS |
| Wavelet step reader | `kinetra.model_manifest.read_wavelet_step()` | ✅ EXISTS |
| Agent CMP file path | `kinetra.model_manifest.AGENT_CMP_RELPATH` | ✅ EXISTS |
| Blacklist read / write | `kinetra.data_gap_tools.read_blacklist()` / `write_blacklist()` | ✅ EXISTS |
| `is_blacklisted()` check | `kinetra.data_gap_tools.is_blacklisted()` | ✅ EXISTS |
| Gap scan / classify | `kinetra.data_gap_tools.scan_and_classify_gaps()` | ✅ EXISTS |
| Max fill bars constant | `kinetra.market_calendar.MAX_FILL_BARS` | ✅ EXISTS — import, don't redefine |
| Omega ratio (module-level) | `kinetra.backtesting.metrics.omega_ratio` | ✅ EXISTS |
| Omega ratio (class method) | `kinetra.backtesting.metrics.MetricsCalculator.omega_ratio` | ✅ EXISTS |
| Z-factor (returns array) | `kinetra.backtesting.metrics.calculate_z_factor` | ✅ EXISTS |
| Z-factor (trade dicts) | `kinetra.backtesting.metrics.MetricsCalculator.z_factor` | ✅ EXISTS |
| ATR / adaptive volatility | `kinetra.volatility_utils` | ✅ EXISTS |
| Project root path | `kinetra.config.PROJECT_ROOT` | ✅ EXISTS — import, never inline (DRY-14 ✅ DONE) |
| Friction / instrument spec | `kinetra.friction_cost.InstrumentSpec` | ✅ EXISTS (canonical) |
| Load CSV data (MT5 format) | `kinetra.data_utils.load_mt5_csv` | ✅ EXISTS (legacy — prefer `load_broker_csv`) |
| Load CSV data (broker-neutral) | `kinetra.data_utils.load_broker_csv` | ✅ EXISTS (canonical — §28 Phase 2) |
| OHLCV normalisation | `kinetra.broker.normalize_ohlcv` | ✅ EXISTS (§28 Phase 2) |
| OHLCV schema detection | `kinetra.broker.detect_ohlcv_schema` | ✅ EXISTS (§28 Phase 2) |
| Broker connection ABC | `kinetra.broker.BrokerConnection` | ✅ EXISTS (§28 Phase 2) |
| Broker data handler ABC | `kinetra.broker.BrokerDataHandler` | ✅ EXISTS (§28 Phase 2) |
| Broker spec handler ABC | `kinetra.broker.BrokerSpecHandler` | ✅ EXISTS (§28 Phase 2) |
| Broker identity metadata | `kinetra.broker.BrokerInfo` | ✅ EXISTS (§28 Phase 2) |
| Broker source detection | `kinetra.broker.detect_broker_source` | ✅ EXISTS (§28 Phase 2) |
| Symbol suffix stripping | `kinetra.broker.strip_broker_suffix` | ✅ EXISTS (§28 Phase 2) |
| Category inference | `kinetra.broker.infer_category` | ✅ EXISTS (§28 Phase 2) |
| Timeframe normalisation | `kinetra.broker.canonical_timeframe` | ✅ EXISTS (§28 Phase 2) |
| InstrumentSpec (source-aware) | `kinetra.friction_cost.InstrumentSpec.from_broker_json()` | ✅ EXISTS (§28 Phase 2) |
| FX enrichment (spec polling) | `kinetra.spec_utils.enrich_with_fx_rates` | ✅ EXISTS (§28 Phase 2) |
| Spec save/merge orchestration | `kinetra.spec_utils.save_specs` | ✅ EXISTS (§28 Phase 2) |
| Symbol discovery | `kinetra.spec_utils.discover_symbols` | ✅ EXISTS (§28 Phase 2) |
| Quote currency inference | `kinetra.spec_utils.infer_quote_currency` | ✅ EXISTS (§28 Phase 2) |
| Instrument dir lookup | `kinetra.spec_utils.find_instrument_dir` | ✅ EXISTS (§28 Phase 2) |
| Contract spec loading | `kinetra.spec_utils.load_contract_spec` | ✅ EXISTS (§28 Phase 2) |
| Core data manager | `kinetra.data.DataManager` | ✅ EXISTS (canonical) |
| Download manager | `kinetra.data.download.DownloadManager` | ✅ EXISTS |
| Backtesting (standard) | `kinetra.backtesting.core.UnifiedBacktester(mode='standard')` | ✅ EXISTS |
| Backtesting (MT5-realistic) | `kinetra.backtesting.core.UnifiedBacktester(mode='realistic')` | ✅ EXISTS |
| Backtesting (physics) | `kinetra.backtesting.core.UnifiedBacktester(mode='physics')` | ✅ EXISTS |
| Backtesting (portfolio) | `kinetra.backtesting.core.UnifiedBacktester(mode='portfolio')` | ✅ EXISTS |
| Trade / TradeDirection types | `kinetra.backtesting.types` | ✅ EXISTS (canonical) |
| Portfolio health dataclass | `kinetra.portfolio_health.PortfolioHealthScore` | ✅ EXISTS |
| Strategy health scorer | `kinetra.health_score.CompositeHealthScore` | ✅ EXISTS |

### 27.2 Open Violations Summary

All Tier-1 violations (DRY-01 through DRY-07) and most Tier-2/3 violations are resolved.
The remaining open items are high-blast-radius Phase C migrations:

| ID | Description | Tier | Status |
|----|-------------|------|--------|
| DRY-09 | `symbol_spec` / `symbol_specs` / `symbol_info` caller migration | 2 | Phase B done — Phase C (caller migration) deferred |
| DRY-10 | `data_manager` / `unified_data_manager` caller migration | 2 | ✅ Phase C DONE (Sprint 8) — shim files archived to `archive/kinetra/` |
| DRY-11 | Standalone backtester files → `UnifiedBacktester` caller migration | 2 | Phase C partial (Sprint 8) — compat bridges added; `hpo_optimizer.py`, `batch_backtest_consolidated.py` deferred |

All other violations (DRY-01–08, DRY-12–16, DRY-D1–D3) are ✅ DONE.
See `../repo/docs/HOUSEKEEPING_AUDIT.md` for the full status table and per-item notes.

### 27.3 New-Code Rule

When writing new code that needs **any** of the concepts listed in §27.1:

1. Import from the canonical location.
2. If the canonical location does not yet have the function (marked ⏳), do **not**
   write a local copy — implement it in the canonical module first, then import.
3. If you are fixing a DRY violation (collapsing N copies → 1), run the full
   test suite before committing: `pytest tests/ -q && ruff check .`

### 27.4 Previously Remediated DRY Issues

| ID | Description | Resolution | Sprint |
|----|-------------|-----------|--------|
| T-20 | `compute_atr()` — 5 duplicate implementations | `kinetra/volatility_utils.py` created | Pre-DRY |
| T-20b | `runway_engine` / `physics_backtester` ATR duplicates | delegated to `volatility_utils` | Pre-DRY |
| DRY-01 | `_detect_agent_comparison()` — 7 copies | `kinetra.model_manifest.detect_agent_comparison` | Sprint 1 |
| DRY-02 | Wavelet step reader — 8 boilerplate blocks | `kinetra.model_manifest.read_wavelet_step` | Sprint 1 |
| DRY-03 | `_AGENT_CMP_FILE` — 10 constant definitions | `kinetra.model_manifest.AGENT_CMP_RELPATH` | Sprint 1 |
| DRY-04 | `_is_blacklisted()` — 2 copies | `kinetra.data_gap_tools.is_blacklisted` | Sprint 2 |
| DRY-05 | `_read_blacklist()` — 2 copies | `kinetra.data_gap_tools.read_blacklist` | Sprint 2 |
| DRY-06 | `_scan_and_classify_gaps()` stranded in menu | `kinetra.data_gap_tools.scan_and_classify_gaps` | Sprint 2 |
| DRY-07 | `_MAX_FILL_BARS` redefined in menu | `kinetra.market_calendar.MAX_FILL_BARS` imported | Sprint 1 |
| DRY-08 | `omega_ratio()` — 6+ implementations | `kinetra.backtesting.metrics` canonical | Sprint 3 |
| DRY-09 | `SymbolSpec`/`SymbolInfo`/`InstrumentSpec` — 5+ defs | DeprecationWarning Phase A done | Sprint 5 |
| DRY-10 | `DataManager`/`UnifiedDataManager` — 3 classes | Phase C DONE — `data_manager.py` + `unified_data_manager.py` archived to `archive/kinetra/`; ImportError regression guards in tests | Sprint 6–8 |
| DRY-11 | Backtester — 5 standalone files | Phase C partial — `run_backtest()`/`monte_carlo_validation()` compat bridges on `UnifiedBacktester`; `__init__.py` `TradeDirection` → `backtesting.types`; `integrated_backtester`, `mt5_connector`, scripts/testing migrated | Sprint 5–8 |
| DRY-12 | `CompositeHealthScore` — 2 classes same name | Renamed to `PortfolioHealthScore` in `portfolio_health.py` | Sprint 4 |
| DRY-13 | `calculate_z_factor()` — 2 implementations | `kinetra.backtesting.metrics.calculate_z_factor` canonical | Sprint 3 |
| DRY-14 | `project_root = Path(...)` — ~40 inline copies in scripts/ | All migrated to `from kinetra.config import PROJECT_ROOT`; depth bugs fixed | Sprint 3–4, Sprint 7 |
| DRY-15 | `load_csv_data` / `load_mt5_csv` overlap | `mt5_connector.load_csv_data` is thin wrapper over `data_utils.load_mt5_csv` | Sprint 3 |
| DRY-16 | `sys.path.insert` in every script | Acknowledged BY DESIGN; not a defect | — |
| DRY-D1 | Three stale consolidation plans | Cross-references and status checklists updated | Sprint 5 |
| DRY-D2 | `docs/ACTION_PLAN.md` outdated | Archived to `archive/docs/ACTION_PLAN.md` | Sprint 5 |
| DRY-D3 | AGENT_RULES_MASTER / copilot overlap | Acknowledged BY DESIGN | — |

---

## 28. MULTI-BROKER ARCHITECTURE (2026-03-01)

> **Status:** 📐 DESIGN — no cTrader code yet.  This section establishes the
> abstraction boundaries so that current MetaAPI refactoring work builds toward
> multi-broker support without rework.
>
> **Target broker:** Pepperstone cTrader (account available for testing).
> **Protocol:** cTrader Open API (Protobuf over WebSocket + OAuth2).
> **Phased approach:** MetaAPI must work end-to-end first.  cTrader implementation
> starts only after MetaAPI pipeline is production-stable.

### 28.1 The Broker-Neutral Boundary (CRITICAL INVARIANT)

The entire Kinetra pipeline is split at a hard boundary:

```
BROKER-AWARE (upstream)          │  BROKER-BLIND (downstream)
─────────────────────────────────│──────────────────────────────────────
  MetaAPI connector              │  data/master_standardized/ CSVs
  cTrader connector (planned)    │  InstrumentSpec dataclass
  MT5 local connector            │  PhysicsEngine, feature engineering
  Download handlers              │  Backtesting, training, RL envs
  Spec pollers                   │  Reward orchestrator, manifests
  Format converters              │  All menu gates & pipeline logic
```

**Rule:** Everything downstream of `master_standardized/` and `InstrumentSpec` MUST
remain broker-blind.  No module in the physics/feature/backtest/training pipeline
may import MetaAPI, MT5, or cTrader symbols.

**Corollary:** All broker differences (symbol naming, OHLCV format, spec field names,
auth protocol, order model) MUST be normalised before crossing the boundary.

### 28.2 Abstraction Points — What to Extract When Refactoring MetaAPI Code

When modifying any broker-specific module, look for opportunities to extract
broker-neutral logic into shared helpers.  The following table identifies the
planned abstraction points:

| Concern | Broker-specific (keep separate) | Broker-neutral (extract & share) |
|---------|--------------------------------|----------------------------------|
| **Connection** | Auth flow (token vs OAuth2), WebSocket protocol, SDK calls | Retry/backoff, circuit breaker, health scoring, session caching |
| **Download** | API pagination, rate limiting, field names | OHLCV normalisation, progress tracking, parallel orchestration |
| **Spec polling** | RPC call, response field mapping | FX rate enrichment, USD conversion, save/merge to `contract_spec.json`, symbol discovery iteration |
| **OHLCV format** | Column names, timestamp precision | Canonical schema validation, dedup, sort, gap detection |
| **Order execution** | Order type enums, fill policies, API calls | Order validation (already via `OrderValidator`), margin checks |
| **Symbol naming** | Broker-specific names (e.g. `XAUUSD+` vs `GOLD`) | Symbol alias resolution, canonical name registry |

### 28.3 Existing Broker-Neutral Assets (Already DRY-Ready)

These modules are ALREADY broker-blind — no changes needed for cTrader:

| Module | Why it's safe |
|--------|--------------|
| `kinetra/broker.py` → `BrokerConnection`, `BrokerDataHandler`, `BrokerSpecHandler` | **NEW (Phase 2)** — ABCs that all broker adapters implement; no SDK imports |
| `kinetra/broker.py` → `normalize_ohlcv()`, `detect_ohlcv_schema()` | **NEW (Phase 2)** — canonical OHLCV normalisation chokepoint; broker-agnostic |
| `kinetra/broker.py` → `BrokerInfo`, `OHLCVSchema`, `SpecPollResult` | **NEW (Phase 2)** — data classes for broker metadata; no SDK dependency |
| `kinetra/broker.py` → `strip_broker_suffix()`, `infer_category()`, `canonical_timeframe()` | **NEW (Phase 2)** — symbol/timeframe utilities; pure string logic |
| `kinetra/spec_utils.py` → `enrich_with_fx_rates()`, `save_specs()`, `discover_symbols()` | **NEW (Phase 2)** — spec polling orchestration extracted from `poll_symbol_specs.py`; broker-blind |
| `kinetra/spec_utils.py` → `find_instrument_dir()`, `load_contract_spec()` | **NEW (Phase 2)** — directory lookup and spec loading; no SDK calls |
| `kinetra/data_utils.py` → `load_broker_csv()` | **NEW (Phase 2)** — canonical broker-neutral CSV loader; auto-detects format |
| `kinetra/friction_cost.py` → `InstrumentSpec` | Reads `contract_spec.json` — doesn't care who wrote it |
| `kinetra/friction_cost.py` → `InstrumentSpec.from_broker_json()` | **NEW (Phase 2)** — source-aware factory; delegates to `from_polled_json()` |
| `kinetra/friction_cost.py` → `FrictionCalculator` | Works on `InstrumentSpec` fields only |
| `kinetra/order_validator.py` → `OrderValidator` | Validates against `InstrumentSpec` — no broker SDK calls |
| `kinetra/broker_compliance.py` → `BrokerComplianceValidator` | Works on spec data, not live API |
| `kinetra/network_resilience.py` → `Connection(ABC)` | Already abstract — `ConnectionManager`, `ReconnectHandler`, `ServerPool` are broker-neutral |
| `kinetra/data/download.py` → `DataSource` | Already has `name: str` supporting `'metaapi'`, `'mt5'`, `'csv'` — add `'ctrader'` |
| `kinetra/data_utils.py` → `load_mt5_csv()` | Normalises CSV → canonical OHLCV schema (legacy — prefer `load_broker_csv`) |
| All of `kinetra/physics_engine.py`, `kinetra/dsp_features.py`, backtesting, training, reward orchestrator | Consume OHLCV DataFrames + `InstrumentSpec` only |

### 28.4 Key cTrader Differences to Design For

| Aspect | MetaAPI / MT5 | cTrader Open API |
|--------|---------------|------------------|
| **Auth** | Simple token + account ID | OAuth2 (client_id, client_secret, access_token, refresh_token + auto-renewal) |
| **Protocol** | REST + WebSocket (JSON) | Protobuf over WebSocket (binary) |
| **Symbol IDs** | String names (`EURUSD`, `XAUUSD+`) | Numeric symbol IDs + string names (may differ from MT5) |
| **Timeframes** | MT5 enum (`H1`, `H4`) | cTrader period types (different enum values) |
| **OHLCV format** | `time, open, high, low, close, tick_volume` | Different field names, timestamps in milliseconds |
| **Spec fields** | `tickSize, tickValue, contractSize, digits` | `pipPosition, stepVolume, maxVolume, lotSize, digits` |
| **Order model** | `ORDER_TYPE_BUY/SELL`, `ORDER_FILLING_FOK/IOC` | `MARKET/LIMIT/STOP`, different fill policies |
| **Account model** | Login + server | ctid (cTrader ID) + account number |
| **Swap reporting** | Points per day | May be in currency or percentage |
| **SDK dependency** | `metaapi-cloud-sdk` (pip) | `ctrader-open-api` or custom Protobuf stubs |

### 28.5 Phased Implementation Plan

**Phase 1 — MetaAPI First (CURRENT):**
Complete the MetaAPI pipeline end-to-end.  No cTrader code.  But when
refactoring MetaAPI modules, follow the §28.2 extraction guidance — don't
bake MetaAPI assumptions into shared logic.

**Phase 2 — Pre-cTrader Prep (✅ COMPLETE):**
Broker-neutral ABCs and shared utilities extracted from MetaAPI code.
Pure refactoring — no new broker, no new features.  134 tests in
`tests/test_broker_abstractions.py`.

New modules and additions:

- `kinetra/broker.py` (**NEW**) — Broker ABCs and normalisation:
  - `BrokerConnection(ABC)` — auth, lifecycle, health scoring
  - `BrokerDataHandler(ABC)` — historical OHLCV download interface
  - `BrokerSpecHandler(ABC)` — instrument spec polling interface
  - `BrokerInfo` — frozen dataclass for broker identity metadata
  - `BrokerCapability` — enum of advertised capabilities
  - `ConnectionState` — enum for connection lifecycle
  - `OHLCVSchema` — column layout descriptor for broker CSV formats
  - `SpecPollResult` — result of polling a single instrument's spec
  - `normalize_ohlcv()` — canonical OHLCV normalisation chokepoint
  - `detect_ohlcv_schema()` — auto-detect CSV format from headers
  - `detect_broker_source()` — identify which broker produced a file
  - `canonical_timeframe()` — normalise timeframe strings (`"4h"` → `"H4"`)
  - `strip_broker_suffix()` — remove ECN suffixes (`"XAUUSD+"` → `"XAUUSD"`)
  - `infer_category()` — asset category from symbol name
  - Pre-built schemas: `MT5_SCHEMA`, `METAAPI_SCHEMA`, `CTRADER_SCHEMA`
  - `CANONICAL_OHLCV_COLUMNS`, `TIMEFRAME_MINUTES` constants
- `kinetra/spec_utils.py` (**NEW**) — Broker-neutral spec polling orchestration:
  - `enrich_with_fx_rates()` — FX-corrected USD cost fields
  - `compute_quote_usd_rates()` — quote-currency → USD rate map
  - `save_specs()` — save/merge to `contract_spec.json` with field ownership
  - `discover_symbols()` — scan `master_standardized/` for symbols
  - `find_instrument_dir()` / `canonical_instrument_dir()` — directory lookup
  - `infer_quote_currency()` — 3-char quote currency from symbol name
  - `load_contract_spec()` / `list_available_contract_specs()` — disk loading
  - `print_spec_summary()` / `print_save_result()` — operator-facing output
  - `BROKER_FIELDS` / `MANUALLY_REVIEWED_FIELDS` — field ownership constants
  - `SaveResult` — outcome dataclass with counts and warnings
- `kinetra/data_utils.py` — `load_broker_csv()` added:
  - Canonical broker-neutral CSV loader with `format='auto'|'mt5'|'metaapi'|'ctrader'`
  - Auto-detects format from column headers via `detect_ohlcv_schema()`
  - Normalises to canonical OHLCV schema via `normalize_ohlcv()`
  - Replaces `load_mt5_csv()` as the recommended entry point for new code
- `kinetra/friction_cost.py` — `InstrumentSpec.from_broker_json()` added:
  - Source-aware factory method (`source='metaapi'|'mt5'|'ctrader'|'default'|'csv'`)
  - Delegates to `from_polled_json()` then stamps the `source` field
  - Placeholder for cTrader field remapping (Phase 3)
- All new symbols registered in `kinetra/__init__.py` lazy import map
- `DataSource.name` already supports `'ctrader'` as a valid string value

**Phase 3 — cTrader Integration (FUTURE):**
Implement cTrader-specific modules:

- `kinetra/ctrader_connector.py` — connection, data, execution
- `scripts/download/ctrader_downloader.py` — download handler
- `scripts/data/poll_ctrader_specs.py` — spec polling (thin, uses shared save/merge)
- `InstrumentSpec.from_ctrader_json()` — field mapping
- OAuth2 token management (refresh flow, secure `.env` storage)
- Menu wiring: Setup & Auth section, Download section, Spec polling
- Tests: connection mock, spec round-trip, OHLCV normalisation

### 28.6 Hard Rules for Multi-Broker Code

- ❌ **Never** import `metaapi_cloud_sdk`, `MetaTrader5`, or any broker SDK in
  library modules under `kinetra/` (except dedicated connector files and
  `poll_symbol_specs.py`).  All broker SDK imports must be behind
  `try/except ImportError`.
- ❌ **Never** assume symbol names are the same across brokers.  Pepperstone MT5
  uses `XAUUSD`; Pepperstone cTrader may use `XAUUSD` or `GOLD` or a numeric ID.
  The canonical name is whatever is in `master_standardized/<category>/<SYMBOL>/`.
- ❌ **Never** hardcode MetaAPI-specific field names (e.g. `tickSize`, `tickValue`)
  outside of the MetaAPI adapter layer.  `InstrumentSpec` is the canonical schema.
- ❌ **Never** add broker-specific logic to `PhysicsEngine`, `FrictionCalculator`,
  `OrderValidator`, backtesting engines, RL environments, or training scripts.
- ✅ **Always** normalise OHLCV to the canonical schema (`time, open, high, low,
  close, volume`) before writing to `master_standardized/`.
- ✅ **Always** normalise spec data to `InstrumentSpec` fields before writing to
  `contract_spec.json`.
- ✅ **Always** use `try/except ImportError` for broker SDK imports so the system
  degrades gracefully when a broker's package is not installed.
- ✅ **Always** isolate broker connection state — no cross-broker session bleed.
  A MetaAPI session failure must not affect cTrader connectivity and vice versa.
- ✅ **When refactoring MetaAPI code**, ask: "Would a cTrader implementation need
  to duplicate this logic?"  If yes, extract the shared part first.

### 28.7 Broker Account Registry (Design Intent)

```
.env (or secure vault):
  # MetaAPI (current)
  METAAPI_TOKEN=...
  METAAPI_ACCOUNT_ID=...

  # cTrader (planned — Pepperstone)
  CTRADER_CLIENT_ID=...
  CTRADER_CLIENT_SECRET=...
  CTRADER_ACCESS_TOKEN=...
  CTRADER_REFRESH_TOKEN=...
  CTRADER_ACCOUNT_ID=...
```

The menu's Setup & Auth section (Menu 1) will gain a cTrader subsection
alongside the existing MetaAPI and MT5 subsections.  `SystemStatus` will
gain `ctrader_available` and `ctrader_connected` fields mirroring the
existing `metaapi_available` / `metaapi_connected` pattern.

### 28.8 Data Pipeline — Broker Source is Transparent After STEP 1

The canonical pipeline (§23) is broker-agnostic from STEP 2 onwards:

```
STEP 1  ACQUIRE  ← broker-specific (MetaAPI / MT5 / cTrader)
                    writes to master_standardized/ in canonical OHLCV schema
────────────────── BROKER BOUNDARY ──────────────────────────────────────
STEP 2  VALIDATE ← broker-blind (reads canonical CSVs)
STEP 3  CURATE   ← broker-blind
STEP 4  SPECS    ← spec polling is broker-specific, but contract_spec.json
                    schema is broker-neutral (InstrumentSpec)
STEP 5+ ...      ← all broker-blind
```

The `contract_spec.json` files in each instrument folder contain a
`broker_friction` section whose schema is defined by `InstrumentSpec`.
A cTrader spec poller would write the **same JSON schema** — downstream
code cannot tell (and should not care) which broker produced the spec.

### 28.9 Symbol Alias Resolution (Future Consideration)

Different brokers may use different names for the same instrument:

```
MT5 (Pepperstone):     XAUUSD
cTrader (Pepperstone): XAUUSD  (likely same, but not guaranteed)
MT5 (ICMarkets):       XAUUSD.raw
```

When cTrader integration begins, a symbol alias map may be needed:

```
data/master_standardized/metals/XAUUSD/
  ├── contract_spec.json      (canonical spec)
  ├── XAUUSD_H4_*.csv         (data from any broker)
  └── broker_aliases.json      (optional: {"metaapi": "XAUUSD", "ctrader": "XAUUSD"})
```

This is a Phase 3 concern — do not pre-build it.  But be aware of it when
designing symbol lookup logic.

---

## 29. SPRINT 5 ARCHITECTURE — INDUSTRIALISATION (2026-03-04) ✅ ALL SPRINTS 5A/5B/5C COMPLETE

> **Status:** ⭐ CURRENT PRIORITY
>
> This section captures all architectural decisions from the design review
> sessions of 2026-03-04, covering: data QC findings on XAUUSD M1, broker
> data source strategy, strategy discovery conclusions, industrialisation gaps,
> and RL regime-adaptation placement.  All agents must follow these rules.

---

### 29.1 Strategy Discovery Conclusions — LOCKED ✅

**Sprint 5 completion summary (2026-03-04 → complete):**
- Sprint 5A: `RiskParams` (loss-cluster breaker), `scaled_filter_params()`, `session_break_minutes` param in `build_renko()` ✅
- Sprint 5B: `session.py` (detect_session_break, SessionProfile, clamp_spikes), `qualify.py` (qualify_instrument, QualificationRegistry, CalibrationDriftDetector), `scripts/renko/qualify_instruments.py` CLI ✅
- Sprint 5C: `orchestrator.py` (run_full_pipeline, run_qualification_only, PortfolioPipelineResult, private helpers), `PortfolioDaySnapshot.vr_drift` + `recalibration_pending` fields, `N_RISK_OBS_FEATURES` bumped 8→10, comprehensive test suites ✅
- Total Sprint 5 tests added: ~500+ (test_renko_qualify.py, test_renko_orchestrator.py expansions, test_renko_rl.py RL drift tests) ✅


The Renko trading core is **fully validated and locked**.  Do not re-open signal
research.  The following conclusions are canonical:

| Decision | Conclusion | Rationale |
|---|---|---|
| Entry | Colour flip only | Structural, not predictive |
| Gate | Fixed Markov persistence | Walk-forward stable; fixed ≈ optimised |
| Exit | FlipExit (colour change) | Deterministic, no discretion |
| Stop | 1 brick (backtest), 0.5 brick (live) | Worst-case proxy |
| Brick size | DSP-derived, structural parameter | Dominates everything; not a tuning knob |
| Rejected filters | TMA, PSAR, entropy/DFA, higher-order Markov | Reduced participation without improving equity |
| Chop handling | Loss-cluster breaker (damage control) | Not prediction — participation throttle |
| Sizing | Fixed risk (1R per trade) | Best in walk-forward; vol-target improves stability but reduces returns |

**❌ Never re-introduce TMA, PSAR, entropy filters, or higher-order Markov.**
**❌ Never treat brick size as an optimisation parameter — it is a structural measurement.**
**✅ Chop is a risk problem, not a signal problem.** The loss-cluster breaker is the correct mechanism.

---

### 29.2 Data QC — Broker Fingerprint Requirements

Every M1 file **MUST** be fingerprinted before brick construction.  The XAUUSD
analysis revealed:

| Finding | XAUUSD Value | Action Required |
|---|---|---|
| Daily missing window | 21:00–21:59 UTC (60 min) | Detect per instrument; set as session break |
| Dominant gap UTC time | 22:01 (MetaAPI broker) | Store in `SessionProfile` |
| Post-rollover spikes | Clusters at 22:00–23:01 UTC | Session break prevents burst artifacts |
| Weekend bars | ~120 min/day (Sunday reopen) | Normal; do not strip |
| OHLC integrity | 100% clean | Verify per instrument |
| Spike rate | 91 / 1,056,687 bars (0.009%) | Clamp, never interpolate |

**Mandatory QC metrics per instrument file (automated, gates DSP):**
1. Coverage ratio (rows / expected minutes between endpoints)
2. Duplicate timestamps
3. Top-10 gap sizes + counts
4. Dominant missing minute-of-day range (the "session break" window)
5. Spike count (ratio > 20× rolling median AND absolute move > threshold)
6. OHLC consistency (high ≥ low, open/close within [low, high], volume ≥ 0)

**Spike guard (data hygiene, not smoothing):**
- Flag: `|Δclose| > max(threshold_abs, 20 × rolling_median_abs_Δ over 720 bars)`
- Action: **clamp** to `prev_close ± threshold` — never drop (dropping creates a gap)
- Log every clamp
- Fail-fast if clamp rate exceeds 0.01% of bars

---

### 29.3 Session Break — CRITICAL Missing Feature

`build_renko()` currently treats the close series as continuous.  **This is wrong.**
The daily rollover gap (e.g. 21:00–21:59 UTC on MetaAPI) creates brick bursts at
resumption that corrupt FlipRate, Markov, and VPIN baselines.

**Required change to `kinetra/renko/brick_engine.py`:**

```python
def build_renko(
    closes: pd.Series,
    brick_size: float,
    *,
    return_ref: bool = False,
    session_break_minutes: float = 30.0,   # ← NEW PARAMETER
) -> pd.DataFrame:
    """
    session_break_minutes: when gap between consecutive bars exceeds this
    threshold, reset the reference price to the new bar's close instead of
    spanning the gap.  Prevents post-rollover brick bursts.
    Set to 0 to disable (legacy behaviour).
    """
```

**Behaviour on session break:**
- When `dt[i] >= session_break_minutes`, set `ref = float(vals[i])` and continue
- Do NOT emit bricks across the gap
- Optionally mark a `session_break` column for downstream use

**This affects every downstream result.**  Fix before running any new backtests.

**❌ Never build bricks across a gap ≥ 30 minutes without session break detection.**
**✅ Always pass `session_break_minutes` from the instrument's `SessionProfile`.**

---

### 29.4 Broker Data Source Strategy

**Canonical decisions (locked):**

| Question | Answer |
|---|---|
| Overlay MetaAPI + cTrader data for same instrument? | ❌ No — correlated duplicates with different artifacts |
| Primary research feed | cTrader (tighter spreads, cleaner UTC alignment for live) |
| Brick sequence transfers cross-broker? | ✅ Yes — market-intrinsic |
| FlipRate / Markov parameters transfer? | ✅ Yes — market-intrinsic |
| Friction floor transfers? | ❌ No — must recalibrate per broker |
| VPIN baseline transfers? | ❌ No — volume is incommensurable across brokers |
| Session break UTC time transfers? | ❌ No — broker server-timezone dependent |
| Circuit breaker thresholds transfer? | ❌ No — must recalibrate per broker |

**What belongs in `spread_profile.json` (addition to Phase 1 spec):**
```json
{
  "broker_source": "pepperstone_ctrader",
  "server_timezone_offset_hours": 0,
  "daily_gap_utc_start": "22:00",
  "daily_gap_utc_end": "23:00"
}
```

**Cross-broker robustness test:**
Run full backtest pipeline on cTrader data, then re-run on MetaAPI data for same
instruments and period.  If Omega and Z-factor are within 15%, strategy is
broker-agnostic.  If they diverge significantly, the divergence is in the cost
model (friction), not the signal.

**❌ Never use cTrader spread data to qualify instruments for MetaAPI deployment.**
**❌ Never use MetaAPI volume to compute VPIN baselines for cTrader deployment.**
**✅ Always store `broker_source` in `spread_profile.json` alongside spread percentiles.**

---

### 29.5 Five Missing Components — Sprint 5 Build List

These are the gaps between the current codebase and a fully industrialised
pipeline.  They must be built in the order listed.

#### 29.5.1 Sprint 5A — Fix the Foundation (before any new backtests)

**A1: Session break in `build_renko()`** — see §29.3
- File: `kinetra/renko/brick_engine.py`
- Parameter: `session_break_minutes: float = 30.0`
- Resets ref price on gap; does not emit bricks across break

**A2: `RiskParams` + loss-cluster breaker in `backtest_instrument()`**
- File: `kinetra/renko/backtest.py`
- New dataclass:
```python
@dataclass(frozen=True, slots=True)
class RiskParams:
    loss_cluster_window: int = 5       # look back N trades
    loss_cluster_threshold: int = 4    # if >= N consecutive losses → pause
    loss_cluster_cooldown: int = 10    # wait M bricks before re-enabling
    dd_throttle_pct: float = 0.10      # reduce sizing if rolling DD > X%
    dd_halt_pct: float = 0.20          # halt if rolling DD > Y%
```
- `backtest_instrument()` gains `risk_params: Optional[RiskParams] = None`
- This is a **participation throttle**, not a signal filter
- Validated empirically: reduces drawdown, improves Omega/PF at cost of participation

**A3: `scaled_filter_params()` in `kinetra/renko/dsp.py`**
- Derives `FilterParams` from DSP output — replaces XAUUSD-calibrated magic defaults
- Signature:
```python
def scaled_filter_params(dsp_result: DSPResult, bricks_per_day: float) -> FilterParams:
    """
    Derive filter parameters that scale with brick characteristics.
    fliprate_window  = f(bricks_per_day)   — more bricks → larger window
    markov_window    = g(bricks_per_day)
    fliprate_threshold / markov_threshold  = h(friction_ratio)
    """
```
- Must be validated empirically on ≥ 3 instruments before locking

#### 29.5.2 Sprint 5B — Qualification Pipeline

**B1: `kinetra/renko/qualify.py`** — canonical per-instrument qualification module
```
qualify_instrument(symbol, closes, spread_profile, contract_spec, config)
  → QualificationResult (dataclass)
  → writes data/renko_qualified/<SYMBOL>/qualification.json
  → idempotent: skip if file exists and data_end matches
```
- Chains: `detect_session_break()` → `run_dsp()` → `scaled_filter_params()` →
  `sweep_brick_sizes()` → `walk_forward_instrument()` → `stress_test_friction()`
- Emits `QualificationResult`:
```python
@dataclass
class QualificationResult:
    symbol: str
    qualified: bool
    qualified_at: str           # ISO timestamp
    data_end: str               # ISO timestamp (for staleness check)
    broker_source: str
    session_break_utc: str      # "21:00"
    dsp_brick_size: float
    selected_brick_size: float
    filter_params: FilterParams
    risk_params: RiskParams
    is_omega: float
    oos_omega: float
    oos_z: float
    friction_ratio: float
    cluster: str
    disqualified_reason: Optional[str] = None
    recalibration_due: bool = False
```

**B2: `QualificationRegistry`** in `kinetra/renko/qualify.py`
```python
class QualificationRegistry:
    def load_all(self, data_dir) -> Dict[str, QualificationResult]
    def get_qualified(self) -> Dict[str, QualificationResult]
    def disqualify(self, symbol, reason)   # for instrument retirement
    def needs_recalibration(self, symbol) -> bool
```

**B3: `kinetra/renko/session.py`** — broker fingerprinting module
```python
@dataclass
class SessionProfile:
    broker_source: str
    daily_gap_utc_start: str    # "21:00"
    daily_gap_utc_end: str      # "21:59"
    gap_duration_minutes: int   # 60
    weekend_bars_present: bool
    session_break_minutes: float  # threshold for build_renko()

def detect_session_break(df_m1: pd.DataFrame) -> SessionProfile:
    """Detect dominant gap window from M1 data (automated QC analysis)."""
```

**B4: `scripts/renko/qualify_instruments.py`** — CLI wrapper
- Parallel across instruments (CPU-adaptive workers)
- `--instrument XAUUSD` or `--all` or `--category metals`
- Reads M1 CSVs, runs full qualification, writes `qualification.json`
- Reports pass/fail table

#### 29.5.3 Sprint 5C — Portfolio Orchestrator & Menu Wiring ✅ COMPLETE

**C1: `kinetra/renko/orchestrator.py`** — full pipeline runner ✅ EXISTS
```python
def run_full_pipeline(
    m1_data: Dict[str, pd.DataFrame],
    spread_specs: Dict[str, Tuple[float, float]],
    *,
    output_dir: Optional[Path] = None,
    results_dir: Optional[Path] = None,
    broker_source: str = "unknown",
    force: bool = False,
    n_workers: int = 1,
    run_mc: bool = False,
    mc_runs: int = DEFAULT_MC_RUNS,
    mc_seed: Optional[int] = None,
    run_tail_risk: bool = True,
    min_portfolio_instruments: int = PORTFOLIO_MIN_INSTRUMENTS,
    portfolio_min_omega: float = PORTFOLIO_MIN_OMEGA,
    portfolio_min_z: float = PORTFOLIO_MIN_Z,
    keep_closes_for_mc: bool = True,
) -> PortfolioPipelineResult:
    """
    qualify → build_portfolio → backtest_portfolio →
    monte_carlo → tail_risk_analysis → write results/renko/portfolio_result.json
    """
```

**C2: `PortfolioDaySnapshot` drift fields** ✅ COMPLETE (Sprint 5C 2026-03-04)
- `vr_drift: float = 0.0` — mean VR change across instruments vs calibration baseline
- `recalibration_pending: float = 0.0` — fraction of instruments flagging drift
- `N_RISK_OBS_FEATURES` bumped from 8 → 10 (obs[8]=vr_drift, obs[9]=recalibration_pending)
- `_build_observation()` updated with indices 8 and 9

**C3: `check_system_status()` wired** ✅ EXISTS in `kinetra_menu.py`
- Reads `results/renko/portfolio_result.json`
- Sets `renko_backtest_done = True` when file exists
- Updates `renko_backtest_portfolio_omega` from file
- `renko_qualified_count`, `renko_qualification_done`, `renko_drift_flags` all populated

**C4: Tests** ✅ COMPLETE
- `tests/test_renko_orchestrator.py` — 137 tests (was 102; added TestDeriveM30Closes,
  TestBuildAndBacktestPortfolio, TestRunPerInstrumentMCNoCloses,
  TestRunPerInstrumentMCWithCloses, TestComputeTailRisk, TestRunFullPipelineWithMC,
  TestRunFullPipelineWithTailRisk, TestDeploymentReadyGate, TestPortfolioDaySnapshotDriftFields)
- `tests/test_renko_rl.py` — 131 tests (was 128; added drift field tests, obs[8]/obs[9] assertions)

---

### 29.6 Regime Change & RL Adaptation — Phased Design

Regime change acts at three distinct time horizons, each requiring a different
mechanism.  Do not conflate them.

#### Horizon Map

| Horizon | Mechanism | Where in Kinetra | Phase |
|---|---|---|---|
| Minutes | Hard circuit breakers (VPIN > extreme, DD > limit) | `monitoring/circuit_breakers.py` | ✅ Done |
| Hours–Days | Layer 3 RL exposure scalar (VPIN + spread + corr obs) | `rl/risk_env.py` `RiskOverlayEnv` | ✅ Done |
| Days–Weeks | Loss-cluster breaker (participation throttle) | `backtest.py` `RiskParams` | Sprint 5A |
| Weeks–Months | Drift-triggered recalibration (re-run DSP + sweep) | `renko/qualify.py` `recalibrate_instrument()` + `CalibrationDriftDetector` + CLI `recalibrate_instruments.py` | ✅ Sprint 6 |
| Months–Years | Layer 2 RL with live `vr_current` feed (learns to down-weight structurally degraded instruments) | `rl/portfolio_env.py` `InstrumentContext.recalibrate()` (called by `recalibrate_instrument()`) | ✅ Sprint 6 |
| Years | Instrument retirement (`QualificationRegistry.disqualify()`) | `renko/qualify.py` | Sprint 6 |

#### Short-horizon (minutes–days): Already correct

`RiskOverlayEnv` observes VPIN, spread, corr, vol regime and learns the exposure
scalar.  Circuit breakers enforce hard limits.  **Do not change this.**

#### Medium-horizon (weeks–months): Recalibration, NOT RL ✅ CalibrationDriftDetector exists in qualify.py

RL is the **wrong tool** for recalibrating brick size and filter parameters.
`run_dsp()` computes in 50ms what RL would need thousands of episodes to approximate.

**`CalibrationDriftDetector`** triggers recalibration when ANY of:
- Rolling VR at current brick scale drops below 0.95 × calibration baseline
- Bricks per day shifts > 30% from `QualificationResult` baseline
- Friction ratio shifts > 20% (spread regime change)
- Walk-forward OOS Omega < `oos_min_omega` over rolling 90 days

```python
# Add to kinetra/renko/qualify.py
class CalibrationDriftDetector:
    def check(
        self,
        symbol: str,
        current_closes: pd.Series,
        qualification: QualificationResult,
    ) -> bool:
        """Return True if recalibration is needed."""
```

**Add to `PortfolioDaySnapshot` (for Layer 3 observation):**
```python
vr_drift: float = 0.0           # mean VR change across instruments vs baseline
recalibration_pending: float = 0.0  # fraction of instruments flagging drift
```

#### Long-horizon (months–years): RL with live structural observations 🔲 Sprint 6

**Missing mechanism: `vr_current` is static in `InstrumentContext`.**

```python
# InstrumentContext currently:
vr_current: float = 1.0   # set at construction, NEVER updated
```

For long-horizon regime adaptation, add `recalibrate()`:
```python
def recalibrate(
    self,
    new_closes: pd.Series,
    new_dsp_result: DSPResult,
    new_filter_params: FilterParams,
) -> None:
    """
    Update structural observations after a recalibration cycle.
    Called by CalibrationDriftDetector when drift is confirmed.
    Updates: vr_current, brick_size, fliprate_threshold, markov_threshold.
    Records recalibrated_at timestamp.
    """
```

The Layer 2 RL agent already observes `vr_current` — when this is updated after
each recalibration cycle, the agent naturally learns to reduce allocation weight
on instruments whose VR has declined.  **No reward function changes needed.**

**This is a Sprint 6 concern** — requires online learning infrastructure and live
data.  Do not implement early; the deterministic Layer 1 pipeline must be stable
first.

#### Why RL should NOT recalibrate brick size or filter thresholds directly

1. Brick size has a well-defined derivation: VR peak scale → median displacement.  There is no reward signal that improves on this.
2. Filter thresholds are derived from `scaled_filter_params(dsp_result, bricks_per_day)`.  RL cannot learn this relationship faster than the analytic function computes it.
3. RL operates in the allocation/exposure space (Layer 2/3).  Calibration is a Layer 0 data-processing concern.  Mixing the layers creates unstable training.

**✅ RL adapts WITHIN a calibrated parameter set (allocation weights, exposure scalar).**
**❌ RL never calibrates the parameter set itself.**

---

### 29.7 Canonical New Modules — Sprint 5 DRY Table Addition

| What you need | Import from |
|---|---|
| Per-instrument qualification | `kinetra.renko.qualify.qualify_instrument` ✅ Sprint 5B COMPLETE |
| Qualification registry | `kinetra.renko.qualify.QualificationRegistry` ✅ Sprint 5B COMPLETE |
| Qualification result | `kinetra.renko.qualify.QualificationResult` ✅ Sprint 5B COMPLETE |
| Session break detection | `kinetra.renko.session.detect_session_break` ✅ Sprint 5B COMPLETE |
| Session profile | `kinetra.renko.session.SessionProfile` ✅ Sprint 5B COMPLETE |
| Loss-cluster risk params | `kinetra.renko.backtest.RiskParams` ✅ Sprint 5A COMPLETE |
| Scaled filter params | `kinetra.renko.dsp.scaled_filter_params` ✅ Sprint 5A COMPLETE |
| Calibration drift detector | `kinetra.renko.qualify.CalibrationDriftDetector` ✅ Sprint 5C COMPLETE |
| Portfolio pipeline orchestrator | `kinetra.renko.orchestrator.run_full_pipeline` ✅ Sprint 5C COMPLETE |
| Portfolio pipeline result | `kinetra.renko.orchestrator.PortfolioPipelineResult` ✅ Sprint 5C COMPLETE |
| InstrumentContext recalibration | `kinetra.rl.portfolio_env.InstrumentContext.recalibrate` ✅ exists (Sprint 6) |
| Drift-triggered recalibration pipeline | `kinetra.renko.qualify.recalibrate_instrument` ✅ Sprint 6 COMPLETE |
| Recalibration result record | `kinetra.renko.qualify.RecalibrationResult` ✅ Sprint 6 COMPLETE |
| Recalibration CLI | `scripts/renko/recalibrate_instruments.py` ✅ Sprint 6 COMPLETE |

**Hard rules for Sprint 5–6 new modules:**
- ❌ **Never** inline session break detection — use `kinetra.renko.session.detect_session_break`
- ❌ **Never** hardcode `FilterParams` defaults for a new instrument — use `scaled_filter_params()`
- ❌ **Never** build a qualification pipeline inline in a script — use `qualify_instrument()`
- ❌ **Never** assemble `instrument_data` dict manually for `build_portfolio()` — use `QualificationRegistry.get_qualified()`
- ❌ **Never** use RL to calibrate brick size or filter thresholds — use `run_dsp()` + `scaled_filter_params()`
- ❌ **Never** inline drift-triggered recalibration logic — use `recalibrate_instrument()` from `kinetra.renko.qualify`
- ❌ **Never** call `InstrumentContext.recalibrate()` directly without going through `recalibrate_instrument()` in production — the wrapper also updates the registry and persists the audit log
- ✅ **Always** run `detect_session_break()` before calling `build_renko()` on a new data file
- ✅ **Always** pass `session_break_minutes` from `SessionProfile` to `build_renko()`
- ✅ **Always** store `broker_source` and session gap UTC times in `spread_profile.json`
- ✅ **Always** use `RiskParams` for loss-cluster and DD throttle configuration
- ✅ **Always** run `CalibrationDriftDetector.check()` on a monthly schedule in live deployment
- ✅ **Always** call `recalibrate_instrument()` (not inline DSP re-run) when drift is confirmed — it chains session detect → DSP → filter params → context update → registry clear → audit log
- ✅ **Always** check `recalibration_log.json` in `data/renko_qualified/<SYMBOL>/` for the per-instrument recalibration history; never trust stale parameters after a known drift event

---

### 29.8 Updated Sprint Table

| Sprint | Focus | Status |
|---|---|---|
| Sprint 1 | Foundation: brick_engine, filters, dsp, aggregation, download_core | ✅ COMPLETE |
| Sprint 2 | Portfolio + Backtesting: backtest.py, portfolio.py | ✅ COMPLETE |
| Sprint 3 | RL Environments + Reward: reward.py, portfolio_env.py, risk_env.py | ✅ COMPLETE |
| Sprint 4 | VPIN + Circuit Breakers + Training: vpin.py, circuit_breakers.py, training scripts | ✅ COMPLETE |
| **Sprint 5A** | **Fix foundation: session break in build_renko(), RiskParams, scaled_filter_params()** | ✅ COMPLETE |
| **Sprint 5B** | **Qualification pipeline: qualify.py, session.py, QualificationRegistry, CalibrationDriftDetector** | ✅ COMPLETE |
| **Sprint 5C** | **Portfolio orchestrator: orchestrator.py, PortfolioDaySnapshot drift fields, menu wiring, 2026 tests** | ✅ COMPLETE |
| **Sprint 6** | **Online learning: recalibrate_instrument() pipeline, CalibrationDriftDetector → context wiring, recalibrate_instruments.py CLI, paper trading, PER gates** | 🔲 IN PROGRESS |
| Sprint 7 | cTrader live connector, multi-broker deployment | 🔲 PLANNED |

---

### 29.9 Updated Menu Structure

```
MENU 1 — Setup & Authentication
  1-3  MetaAPI (configure, test, select account)
  4-5  MT5 (configure, test)
  6    View configuration

MENU 2 — Data Foundation  (pipeline order enforced)
  1-4  ACQUIRE (discover, download M1 MetaAPI, download M1 MT5, fill gaps)
  15   AGGREGATE (M1 → M5/M15/M30/H1/H4 + Renko — needs M1 specifically)
  5-6  VALIDATE (integrity check, coverage)
  7    CURATE (§22)
  8-9  BROKER SPECS (poll, list symbols)
  10-11 ORGANISE (consolidate, reorganise)
  12   PREPARE (physics features — Gate 2 required)
  13   AUDIT (prepared data quality)
  14   DENOISE (research — needs prepared data)
  16   ARCHIVE LEGACY (one-time: move non-M1 files)

MENU 3 — Exploration & Training
  ── PHASE 2: STRUCTURE DISCOVERY ──
  1    Comprehensive Exploration
  2    Scientific Discovery
  3    Prime Instrument Discovery
  ── PHASE 4: FEATURE VALIDATION ──
  4    Wavelet Feature Pipeline
  5    Additive Feature Testing
  6    Feature Ablation Sweep
  ── PHASE 5a: AGENT SELECTION (Physics/Wavelet) ──
  7    Agent Comparison
  8    Train Production Agent
  9    Train Specialists
  ── PHASE 5b: RENKO TRAINING (parallel — needs M1, sequentially gated) ──
  10   Reward Weight Sweep        ← always available with M1
  11   Renko Agent Comparison     ← requires 10 done
  12   Train Risk Agent (L3)      ← requires 11 done
  13   Train Allocation Agent (L2)← requires 12 done
  14   View Training Results

MENU 4 — Backtesting & Validation
  ── PHYSICS / WAVELET (requires trained agent) ──
  1    Quick Backtest
  2    Batch Backtest
  3    Monte Carlo Validation
  4    Walk-Forward Analysis
  ── RENKO — QUALIFICATION (deterministic, needs M1) ──
  7    Qualify Instruments        ← NEW Sprint 5B: runs full qualification pipeline
  8    View Qualification Registry← NEW Sprint 5B: table of qualified instruments
  ── RENKO — BACKTESTING (deterministic Layer 1 — no agent needed) ──
  9    Renko Instrument Backtest
  10   Renko Portfolio Backtest
  11   Renko Walk-Forward
  12   Renko Monte Carlo
  13   Renko Friction Stress Test
  ── RESULTS ──
  5    View Backtest Results
  6    Generate Performance Report

MENU 5 — System Tools & Monitoring
  1    System Status & Diagnostics
  2    Manifest Health Check
  3    Circuit Breaker Status
  4    Calibration Drift Status    ← NEW Sprint 5C: per-instrument drift flags
  5    Cache Management
  6    Backup Data
  7    Clean Temporary Files
  8    Run Tests
  9    View Logs
```

**Gate logic additions for Menu 4 Renko options:**
- Options 7–8 (Qualify): require `m1_data_available`
- Options 9–13 (Backtest): require `m1_data_available`; options 10–13 additionally
  require at least 1 qualified instrument in the registry (`renko_qualified_count > 0`)
- Portfolio backtest (10): requires `renko_qualified_count >= 3` (minimum meaningful portfolio)

**`SystemStatus` new fields (Sprint 5B/C):**
```python
renko_qualified_count: int = 0          # number of instruments passing qualification
renko_qualification_done: bool = False  # at least 1 qualified instrument exists
renko_drift_flags: int = 0              # instruments flagging recalibration_due
```

---

## 30. MENU & TERMINAL UI STYLING GUIDE

> **Status:** ✅ CANONICAL — enforced on every menu edit
>
> All visual output in `kinetra_menu.py` follows a strict layered style system.
> Never introduce new visual constructs without updating this section first.
> When in doubt: match what already exists rather than inventing something new.

---

### 30.1 ANSI Colour & Style Helpers

All colour/style output goes through the helper functions defined at the top of
`kinetra_menu.py`.  They degrade gracefully to plain text on non-TTY output
(pipes, log files, CI) because `_ANSI_SUPPORTED = sys.stdout.isatty()`.

```python
_bold(text)    # bold white      — option numbers, header titles
_dim(text)     # dim/grey        — locked items, lock reason, hint text
_yellow(text)  # yellow          # warnings, missing prerequisites
_green(text)   # green           # success states (use sparingly; ✅ preferred)
_red(text)     # red             # errors (use sparingly; ❌ preferred)
_cyan(text)    # cyan            # informational labels (rare; ✅/💡 preferred)
```

**Hard rules:**
- ❌ **Never** use raw ANSI escape sequences (`\033[…m`) anywhere outside these helpers
- ❌ **Never** call `colorama`, `rich`, `termcolor`, or any third-party colour library
- ❌ **Never** add a new `_colour()` helper without also adding its `_A_COLOUR` constant
  and ensuring it is guarded by `_ANSI_SUPPORTED`
- ✅ Prefer emoji state icons (`✅ ❌ ⚠️ 🚨`) over colour for status — they survive
  copy-paste into logs and Slack

---

### 30.2 Header Hierarchy

There are exactly **two** header types.  Never add a third.

#### Submenu header — `print_submenu_header(text, status)`

Used **once per menu loop iteration** at the top of every `while True:` menu function.

```
════════════════════════════════════════════════════════════════════════════════
  📊 DATA FOUNDATION                         ← bold, emoji prefix, ALL CAPS
════════════════════════════════════════════════════════════════════════════════

  📊 ✅ Raw (230 files M1) | ✅ Curated | …  ← status badges (≤ 8)
  💡 Next: Aggregate M1 → M30 (Menu 2 › 12) ← suggest_next_step() or absent
```

- Bar character: `═` (U+2550), width 80
- Title: `_bold(text)` with 2-space indent
- Always followed by `status.get_status_line()` and optionally `suggest_next_step()`
- Title format: `EMOJI  ALL CAPS NOUN` (two spaces after emoji when emoji is present)

#### Action screen header — `print_header(text)`

Used **once at the top** of every individual action function (the screen that
runs after the user picks an option).

```
────────────────────────────────────────────────────────────────────────────────
  🧱 Renko Instrument Backtest  (Menu 4 › 3)   ← bold, title-case, location ref
────────────────────────────────────────────────────────────────────────────────
```

- Bar character: `─` (U+2500), width 80
- Title: `_bold(text)` with 2-space indent
- **Always includes a menu location reference** in the format `(Menu N › M)` at
  the end of the title — this is mandatory so users can navigate back
- Title format: `EMOJI Label  (Menu N › M)` — two spaces before the location ref

**Menu location reference format:**

| Situation | Format |
|---|---|
| Top-level option | `(Menu 2 › 6)` |
| Sub-option within a screen | `(Menu 2 › 6 › 2)` |
| Layer annotation | `(Menu 3 › 12 — Layer 3)` |
| Planned feature | `(Menu 4 › 9 — PLANNED)` |

---

### 30.3 Section Divider Lines

Section dividers visually separate groups of options within a single menu screen.
They are printed with `print(...)`, **never** embedded inside a multi-line string
literal (doing so would make them literal text, not separators).

**Canonical format — exactly 80 characters total:**

```
  ── LABEL TEXT ──────────────────────────────────────────────────────────────
```

- 2 leading spaces
- `──` (two U+2500 chars) + one space
- Label text (see conventions below)
- One space
- Enough `─` fill chars to reach exactly **80 characters total**

**Generating the correct fill programmatically:**
```python
def _sep(label: str, total: int = 80) -> str:
    prefix = "  ── "
    suffix = " "
    fill = "─" * (total - len(prefix) - len(label) - len(suffix))
    return f"{prefix}{label}{suffix}{fill}"
```

**Hard rules:**
- ❌ **Never** let a divider line be any length other than 80 chars
- ❌ **Never** put a `print("  ── …")` call inside a `print("""…""")` block —
  always break the string before the divider and issue a separate `print()` call
- ❌ **Never** use `═` (submenu bar char) for section dividers — only `─`
- ✅ Always verify length after writing: `len("  ── LABEL ───…") == 80`

**Label text conventions:**

| Context | Style | Example |
|---|---|---|
| Pipeline steps | `STEP N · NOUN  (note)` | `STEP 2 · VALIDATE` |
| Menu sections | `ALL CAPS` | `MAINTENANCE` |
| Sub-sections | `Title Case` | `Layer 3 Architecture` |
| Technical contexts | `noun phrase` | `Sweep configuration` |
| Dynamic content | sentence fragment | `Agents compared on IDENTICAL Renko data sequences` |

---

### 30.4 Menu Item Lines — `_print_menu_item()`

All numbered option lines in a menu **must** go through `_print_menu_item()`.
Never `print()` a raw option line with manual bold/dim formatting.

```python
_print_menu_item(
    number="6",                      # str — option number shown to user
    label="Fill M1 Data Gaps",       # str — short noun phrase, title case
    hint="(extend M1 to today + schedule daily cron/systemd)",  # str — parens, lowercase
    available=locks["6"][0],         # bool — gates the item
    lock_reason=locks["6"][1],       # str — shown as "🔒 needs: …" when locked
    width=38,                        # int — label column width (default 38)
)
```

**Rendered output:**

```
# Available:
  6. Fill M1 Data Gaps                (extend M1 to today + …)

# Locked:
  6. Fill M1 Data Gaps  🔒 needs: download data first (option 2, 3, or 4)
  ^^^^ entire line is dim/greyed
```

**Label conventions:**
- Short noun phrase — typically 3–6 words, title case
- Leading emoji is fine when the action screen also has that emoji in its header
- Never end the label with punctuation

**Hint conventions:**
- Always wrapped in `(parentheses)`
- Lowercase, imperative or descriptive
- For status hints (e.g. `← ✅ 19 qualified`): no parentheses, use `←` arrow
- Keep short: fits on one terminal line alongside the label

---

### 30.5 Status Badge Line — `get_status_line()`

The status line appears under every submenu header.  It is a `|`-separated
sequence of short badges, capped at **8 badges maximum** (the rest become `+N more`).

**Badge format rules:**

| State | Format | Example |
|---|---|---|
| OK / complete | `✅ Noun (detail)` | `✅ Raw (230 files M1)` |
| Warning / stale | `⚠️  Noun (detail)` | `⚠️  Specs (14d old)` |
| Critical | `🚨 Noun (N crit)` | `🚨 Integrity (2 crit)` |
| Missing / error | `❌ Noun` | `❌ No data` |
| Dynamic source | `📡 Noun (detail)` | `📡 T1 (8★ 4★★)` |
| Trained/ready | `✅ Label✅` | `Renko: Sweep✅ Cmp✅ L3✅` |

**Hard rules:**
- ❌ **Never** exceed 8 badges — the `+N more` truncation must be respected
- ❌ **Never** use `_bold()` or `_red()` inside badges — icons carry the state
- ✅ Badge order is fixed: Data → Curation → Integrity → Specs → Discovery →
  Renko Training → Qualification → Broker/Infra

---

### 30.6 Workflow Suggestion Line — `suggest_next_step()`

Printed as `  💡 <suggestion>` immediately below the status line when there is a
clear next step. Absent when the pipeline is complete or the next step is ambiguous.

- Prefix: `💡` (always — do not change)
- One sentence, imperative: `"Aggregate M1 → M30 (Menu 2 › 12)"`
- Includes the menu location so the user can act immediately
- Never use colour helpers — plain text only

---

### 30.7 Emoji Icon Vocabulary

Use emoji **consistently** — same icon always means the same thing.  Introduce
a new emoji only when no existing one fits, and document it here.

**State icons** (highest priority — override label emoji in status contexts):

| Icon | Meaning |
|---|---|
| `✅` | complete / healthy / configured |
| `❌` | missing / failed / not configured |
| `⚠️` | warning / stale / degraded |
| `🚨` | critical / blocking / must fix now |
| `🔒` | locked — prerequisite not met |
| `💡` | next-step suggestion |
| `🚧` | planned / not yet implemented |

**Action screen title emoji** (prefix of `print_header` titles):

| Icon | Domain |
|---|---|
| `🗺️` | Exploration / mapping |
| `🔭` | Scientific discovery |
| `🎯` | Targeting / optimisation / discovery |
| `⚖️` | Comparison / balance |
| `🛡️` | Risk / protection / safety |
| `📊` | Data / results / reporting |
| `📋` | Registry / table / list |
| `🔍` | Search / qualify / inspect |
| `🧱` | Renko bricks / backtest |
| `🧹` | Curation / cleanup |
| `🔁` | Refresh / rewrite |
| `🔄` | Drift / recalibration cycle |
| `📈` | Backtesting & Validation menu |
| `🔐` | Authentication / credentials |
| `🛠️` | System tools / maintenance |

**Infrastructure & pipeline icons:**

| Icon | Meaning |
|---|---|
| `📡` | Dynamic / live data source |
| `🚀` | Script launched |
| `🏦` | Broker context |
| `🔌` | Connection / socket |
| `⏱` / `⏳` | Timing / elapsed |

**Hard rules:**
- ❌ **Never** use a heart, star, sparkle, or decorative emoji anywhere in the menu
- ❌ **Never** use `✔` or `✗` — use `✅` and `❌` exclusively
- ❌ **Never** place an emoji mid-sentence in a hint string — hints are plain text
- ✅ Use emoji only at the **start** of a title, label, or badge — never trailing

---

### 30.8 Inline Text Conventions

**Arrows and separators in text:**

| Symbol | Use |
|---|---|
| `→` (U+2192) | pipeline flow: `M1 → M30` |
| `←` (U+2190) | status annotation on a menu item: `← ✅ 19 qualified` |
| `›` (U+203A) | menu path: `Menu 2 › 12` |
| `·` (U+00B7) | inline separator in labels: `STEP 2 · VALIDATE` |
| `│` (U+2502) | vertical rule in ASCII box diagrams only |

**Number and quantity formatting:**
- File counts: `230 files` (no comma for < 10 000)
- Ages: `14d ago`, `today` (never `0d ago`)
- Omega ratios: `Ω=6.07` (Unicode Omega, not `O=`)
- Percentages: `79%` (no space before `%`)
- Layer references: `L2`, `L3` (not `layer 2`, not `Layer2`)

**Lock reason strings** (passed as `lock_reason=` to `_print_menu_item`):
- Lowercase, imperative, actionable: `"download data first (option 2, 3, or 4)"`
- Always tells the user exactly what to do and where
- Never starts with "You need to…" or "Please…"
- References menu option numbers, not function names

---

### 30.9 `run_script()` Output Block

When a script is launched via `run_script()`, the output is automatically wrapped:

```
  🚀  Running:  python /path/to/script.py --args
  ────────────────────────────────────────────────────────────────────────────

  … script stdout …

  ────────────────────────────────────────────────────────────────────────────
  ✅  Completed in 4m 12s
```

- The inner rule is `─` × 76 (with 2-space indent = 78 chars total)
- Elapsed time uses `_format_elapsed()`: `"42s"`, `"4m 12s"`
- ❌ **Never** replicate this wrapper manually — always use `run_script()`

---

### 30.10 "Press Enter to continue" Prompt

After every action function that launches a script or displays results, the
calling menu loop must pause with:

```python
input("\n  📌  Press Enter to continue…")
```

- Exact string — do not vary the wording or the emoji
- The `\n` before it provides visual breathing room after script output
- ❌ **Never** call `confirm_action()` for this — that expects `y/n`
- ❌ **Never** omit it — without it the menu redraws immediately, losing the output

---

### 30.11 New Screen Checklist

When adding a new action screen to the menu, verify all of the following:

```
[ ] print_header("EMOJI Label  (Menu N › M)") — first line of function body
[ ] Menu location ref present and correct
[ ] All options listed via _print_menu_item() — no raw print() for option lines
[ ] Locked items pass available=False with a clear lock_reason
[ ] Section dividers are exactly 80 chars — verify with len()
[ ] No print("  ── …") inside a """…""" string block
[ ] run_script() used for all subprocess calls — no subprocess.run() directly
[ ] input("\n  📌  Press Enter to continue…") at the call site in the menu loop
[ ] Emoji prefix matches the domain table in §30.7
[ ] New menu option added to _data_menu_locks() or equivalent gate function
[ ] New menu option wired into the while-loop elif chain
[ ] valid_choices list updated to include the new option number
[ ] print_header title added to copilot-instructions.md Menu Structure table
```

---

### 30.12 ASCII Block-Code / Box-Art Diagrams

Box-art diagrams appear in two places in the menu:

1. **Architecture diagrams** — rendered inline before a training/config prompt
   (e.g. the Layer 2/3 three-tier stack shown before training options 12 and 13)
2. **Splash/logo block** — the `╔═╗ … ╚═╝` banner rendered once at startup in `main()`

Both must follow the rules below.  Never create a third category without updating
this section.

---

#### 30.12.1 Inner-arch width — the single source of truth

Every box in a diagram shares one **inner-arch** constant: the number of terminal
display columns between the left border character and the right border character
(not counting either border).

```
inner_arch = 65   ← display columns (terminal cells), NOT Python len()
```

The outer width of every line is therefore:

```
outer = inner_arch + 2   # 1 left-border char + inner + 1 right-border char
                         # = 67 terminal cells for inner_arch=65
```

With a 4-space print-indent (`    `) the total terminal width per line is:

```
total = 4 + outer = 4 + 67 = 71 terminal cells
```

**Hard rule:** all boxes in one diagram block must share the same `inner_arch`.
Mismatched widths produce ragged right borders.

---

#### 30.12.2 Title bar formula

The title bar fills the inner-arch with dashes and a centred label:

```
┌──────────────────── LABEL TEXT ──── … ──────────────────────┐
```

The number of `─` chars on each side of the label is determined by:

```python
label = " LABEL TEXT "          # one space each side inside the dashes
left_dashes  = 20               # empirical left padding — keep consistent per diagram
right_dashes = inner_arch - left_dashes - len(label)
title_bar = "─" * left_dashes + label + "─" * right_dashes
```

For `inner_arch = 65` and `left_dashes = 20`:

```
inner_arch = 65
label      = " HARD CIRCUIT BREAKERS "   # 24 chars
left       = 20
right      = 65 - 20 - 24 = 21
→  ──────────────────── HARD CIRCUIT BREAKERS ─────────────────────
   (20 dashes)          (24 chars)                (21 dashes)  = 65 ✅
```

Canonical left padding is **20** dashes for all architecture diagrams in this menu.
Do not vary it without updating all sibling boxes.

---

#### 30.12.3 Content line formula

Content lines pad their text to exactly `inner_arch` display columns:

```python
content = "  Output: exposure_scalar ∈ [0.0, 1.0]"
pad     = inner_arch - len(content)   # = 65 - 39 = 26 spaces
line    = "│" + content + " " * pad + "│"
```

**Always use Python `len()` for ALL box width calculations** — no exceptions.
Static hardcoded box strings (like splash screens) are fine, but any dynamic
width calculation must use `len()` for simplicity and consistency.

---

#### 30.12.4 Canonical architecture diagram template

The three-tier Renko runtime architecture is the canonical reference.  Copy this
exact template whenever adding a new architecture diagram:

```python
_I = 65  # inner_arch — display columns between borders

def _box(title: str, lines: list[str]) -> list[str]:
    left = 20
    label = f" {title} "
    right = _I - left - len(label)
    top    = "┌" + "─" * left + label + "─" * right + "┐"
    bottom = "└" + "─" * _I + "┘"
    body   = ["│" + ln + " " * (_I - len(ln)) + "│" for ln in lines]
    return [top, *body, bottom]

_arrow = " " * 33 + "▼"   # centred under a 71-col block (4-indent + 67 outer)

_circuit = _box("HARD CIRCUIT BREAKERS", [
    "  NON-NEGOTIABLE — not learned",
])
_l3 = _box("LAYER 3 — RISK RL AGENT  ◄── THIS", [
    "  Output: exposure_scalar ∈ [0.0, 1.0]",
    '  Question: "Should we be in the market at all right now?"',
])
_l2 = _box("LAYER 2 — ALLOCATION RL AGENT  ◄── THIS", [
    "  Output: weight[i] ∈ [0, 1] per instrument",
    '  Question: "How much capital in each instrument right now?"',
])
_l1 = _box("LAYER 1 — DETERMINISTIC ENGINE", [
    "  Renko flip + filter → trade at allocated weight",
])

indent = "    "
for line in [*_circuit, _arrow, *_l3]:
    print(indent + line)
```

Rendered output (each line = 71 terminal cells):

```
    ┌──────────────────── HARD CIRCUIT BREAKERS ─────────────────────┐
    │  NON-NEGOTIABLE — not learned                                   │
    └─────────────────────────────────────────────────────────────────┘
                                 ▼
    ┌──────────────────── LAYER 3 — RISK RL AGENT  ◄── THIS ─────────┐
    │  Output: exposure_scalar ∈ [0.0, 1.0]                          │
    │  Question: "Should we be in the market at all right now?"       │
    └─────────────────────────────────────────────────────────────────┘
```

---

#### 30.12.5 Splash / logo block — static strings only

The startup splash uses a **static hardcoded triple-quoted string** with
**double-line** borders (`╔ ═ ╗ / ╚ ═ ╝ / ║`).

**Two valid approaches:**

**Option 1: Static string (PREFERRED for splash screens):**
```python
print("""
╔═══════════════════════════════════════════════════════════════════════════╗
║                                                                           ║
║   ██╗  ██╗██╗███╗   ██╗███████╗████████╗██████╗  █████╗                 ║
║   ██║ ██╔╝██║████╗  ██║██╔════╝╚══██╔══╝██╔══██╗██╔══██╗                ║
║   █████╔╝ ██║██╔██╗ ██║█████╗     ██║   ██████╔╝███████║                ║
║   ██╔═██╗ ██║██║╚██╗██║██╔══╝     ██║   ██╔══██╗██╔══██║                ║
║   ██║  ██╗██║██║ ╚████║███████╗   ██║   ██║  ██║██║  ██║                ║
║   ╚═╝  ╚═╝╚═╝╚═╝  ╚═══╝╚══════╝   ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═╝                ║
║                                                                           ║
╚═══════════════════════════════════════════════════════════════════════════╝
""")
```

**Option 2: Dynamic with `len()` (for simple ASCII boxes):**
```python
_INNER = 75   # display columns between borders

def _row(content: str) -> str:
    """Build a row using len() - works for ASCII content."""
    return "║" + content + " " * (_INNER - len(content)) + "║"
```

**Hard rules for splash screens:**
- ✅ **Static hardcoded string** — no width calculation needed (current implementation)
- ✅ **Dynamic with `len()`** — only if content is pure ASCII
- ❌ **Never use `wcwidth.wcswidth()`** — adds unnecessary dependency and complexity
- ❌ **Never** use `len()` to pad splash rows — always `wcswidth()`
- ✅ `_INNER = 75` is the canonical splash inner width for static hardcoded strings
- ❌ **Never use `wcwidth`** — adds unnecessary dependency; static strings or `len()` are sufficient

---

#### 30.12.6 Inline box for planned / status screens

Static "planned feature" boxes (like the Performance Report screen) use `─` /
`│` / `┌ ┘` single-line borders and a fixed inner width of **73** display
columns (2-space print-indent + 1 border + 73 inner + 1 border = **77** total):

```
  ┌─────────────────────────────────────────────────────────────────────────┐
  │  STATUS: PLANNED — not yet implemented                                  │
  │                                                                         │
  │  The performance report will generate:                                  │
  │    • HTML report …                                                      │
  └─────────────────────────────────────────────────────────────────────────┘
```

Rules:
- Print-indent is **2 spaces** (matching section dividers)
- Inner width is **73** — `─` × 73 top/bottom bars
- Content lines: `  │  text` + padding to column 75 + `  │`
- ❌ **Never** mix architecture-diagram inner_arch (65) with inline-box inner width (73)
- ✅ **Always use `len()` for width calculations** — all content is ASCII

---

#### 30.12.7 Box-art checklist

```
[ ] All boxes in one diagram share the same inner_arch (canonical: 65)
[ ] Title bar formula: 20 left dashes + " LABEL " + fill dashes = inner_arch
[ ] Content lines padded to exactly inner_arch with ASCII spaces
[ ] Arrow connector is " " * 33 + "▼" (centres under 71-col block)
[ ] Splash block uses wcswidth() — never len() — inner width 75
[ ] Planned-feature inline box uses inner width 73, 2-space print-indent
[ ] No box-art inside a triple-quoted string that is indented in source
    (indentation becomes part of the output and shifts the right border)
[ ] Verify all rendered line lengths before committing:
      python -c "from wcwidth import wcswidth; print(wcswidth('    ┌─…─┐'))"
```

---

#### 30.13 End-to-End Refactor Workflow (2026-02-28) ⭐ ACTIVE

This workflow is mandatory for all Renko/portfolio changes until further notice.

##### 30.13.1 Non-negotiable execution order

1. **Data contract + QC gate**
2. **Canonical dataset build**
3. **Rolling OOS evaluation + drift checks**
4. **Portfolio composition + risk overlay evaluation**
5. **Only then** RL allocation/risk experiments

No step may be skipped. If step 1 fails, all downstream work is blocked.

##### 30.13.2 Data preparation hard requirements

- Every run must reference a specific date range and data manifest/hash.
- Session breaks must be explicit metadata; never inferred silently.
- Spread stats (P50/P90/P95) must be persisted and used in backtests.
- Brick floor must include a causal rolling-friction component:
  - `brick_min >= x * typical(rolling_spread_window)`
  - optional tail guard: `max(..., z * p95(rolling_spread_window))`
  - rolling windows must be trailing-only (no future bars; no look-ahead bias).
- QC report must include: duplicates, gaps, missing-minute profile, spike events, timezone status.
- No interpolation/forward-fill across session gaps for signal generation.

##### 30.13.3 Rolling OOS + drift hard requirements

- Use stitched TEST-only metrics for decision making.
- Produce rolling 30/60/90-day comparison vs qualification baseline.
- Track at minimum: Omega, return/maxDD, Calmar, time-in-DD, trade frequency, loss-cluster rate.
- Drift states are mandatory:
  - `watch`: mild degradation
  - `throttle`: persistent degradation (conservative risk profile forced)
  - `requalify`: hard breach (instrument or portfolio re-qualification)
  - `halt`: risk or integrity breach

##### 30.13.4 RL scope constraints

- RL may optimize **portfolio allocation and risk overlays only**.
- RL must not modify deterministic Layer 1 signal logic (flip, Markov gate, FlipExit, stop model).
- RL must not bypass circuit breakers, risk caps, or QC gates.
- RL value claim is accepted only if it beats baseline at equal or lower risk.

##### 30.13.5 Immediate next implementation step

Focus on end-to-end hardening (not new signal features):

1. Enforce rolling-friction brick floor in qualification/backtest selection.
2. Add rolling OOS drift dashboard outputs (instrument + portfolio).
3. Add portfolio composition policy checks (cluster/concentration/concurrency).
4. Add risk-management profile executor (conservative/optimal/aggressive) with audit trail.

---

## 31. Path Resolution Policy (2026-03-04)

### 31.1 Canonical root resolution

- All runtime filesystem paths MUST resolve from `kinetra.config.PROJECT_ROOT`.
- Relative cwd-based paths are forbidden in runtime modules.
- Use `kinetra.config.resolve_project_path()` for user-provided or default relative paths.

### 31.2 Canonical data roots

- Primary market data root: `data/master_standardized/`.
- Derived aggregate cache root: `data/aggregated/`.
- Hidden control-plane caches/registries: `data/.cache/`.
- Legacy `data/master/` paths are non-canonical and must not be introduced in new code.

### 31.3 Runtime defaults

- DataManager default base dir: `kinetra.config.DATA_DIR` (supports `KINETRA_DATA_DIR`).
- Renko default outputs:
  - `data/renko_qualified/`
  - `results/renko/`
- Menu/context checks must use `PROJECT_ROOT`-anchored paths (never cwd-sensitive `Path("...")`).

### 31.4 Contract spec isolation reminder

- Contract specs are broker/account-scoped artefacts.
- Never mix spec payload provenance across broker/account trees.
- Run contract-spec hygiene before qualification when download/control-plane runs.
