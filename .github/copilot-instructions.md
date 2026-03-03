# GitHub Copilot Instructions for Kinetra
<!-- Synced from consolidated repo rules v4.11 on 2026-02-28 -->
<!-- Last updated: 2026-03-04 — reflects Sprint 4 COMPLETE + §29 Sprint 5 Architecture, plus §31 path-resolution policy (PROJECT_ROOT-anchored runtime paths, data/master_standardized canonical, resolve_project_path for relative defaults) -->

> **⚠️ CANONICAL RULES:** All agent rules are consolidated in [`AGENT_RULES_MASTER.md`](../README.md)
>
> This file provides a **quick reference** for GitHub Copilot. For complete current guidance, see README and QUICK_START.

---

## Quick Reference

```bash
# Setup
make setup              # Full development environment
pip install -e ".[dev]" # Alternative: install with dev dependencies

# Development
make test               # Run all tests
make lint               # Lint code with Ruff
make format             # Format code with Black
pytest archive/production_cleanup_2026-03-03/repo/tests/test_physics.py -v  # Run specific test

# Ruff (zero violations required — enforced on every commit)
ruff check .                     # Check entire project
ruff check . --fix --unsafe-fixes  # Autofix everything fixable
ruff check . --statistics        # Summary by rule code

# Menu (canonical entry point)
python kinetra_menu.py

# Common Commands
python scripts/batch_backtest.py --instrument BTCUSD --timeframe H1
```

---

## Project Overview

**Kinetra** (Kinetic + Entropy + Alpha) is an institutional-grade, physics-first adaptive trading
system that uses reinforcement learning to extract returns from market regimes. Built on first
principles with **no static assumptions**, Kinetra validates every decision through rigorous
statistical testing and continuous backtesting.

**Renko Kinetra** is the active redesign around Renko price-space representation with three
core simplifications: (1) M1-only downloads, derive all higher TFs by aggregation; (2) one
strategy class (Renko flip with regime filter, no physics features); (3) RL for allocation
and risk management, not entry/exit signals.  See `archive/production_cleanup_2026-03-03/repo/docs/TRADING_STRATEGY_SPEC.md`.

---

## Data Pipeline — Canonical Step Order

The Data Management menu (Menu 2) enforces a strict top-to-bottom pipeline.
**Never skip or reorder steps** — each gate exists to protect data integrity.

```
STEP 1  ACQUIRE      (Menu 2 › 1-4)   Download M1 OHLCV (M1-only architecture)
STEP 1b AGGREGATE    (Menu 2 › 15)    M1 → M5/M15/M30/H1/H4 + Renko (needs M1 specifically)
STEP 2  VALIDATE     (Menu 2 › 5-6)   Check integrity BEFORE any processing
STEP 3  CURATE       (Menu 2 › 7)     §22 — fix misclassified, dupes, MIN_BARS
STEP 4  BROKER SPECS (Menu 2 › 8-9)   Poll friction costs BEFORE discovery
STEP 5  ORGANISE     (Menu 2 › 10-11) Consolidate, per-symbol folders
──────  GATE 2       (Menu 3 › 1-3)   Structure Discovery must confirm structure
STEP 6  PREPARE      (Menu 2 › 12)    OHLCV → core physics features ONLY
STEP 7  AUDIT        (Menu 2 › 13)    Validate prepared data quality (NaN, schema)
──────  RESEARCH     (Menu 2 › 14)    Denoise — LOCKED until step 6 done
──────  PHASE 4      (Menu 3 › 4-5)   Wavelet → Additive (operates on RAW H4, not prepared)
──────  LEGACY       (Menu 2 › 16)    Archive non-M1 files
```

**Wavelet Feature Pipeline operates on raw H4 CSVs** in `master_standardized/`, NOT on
prepared data. If wavelet features already exist on disk, the pipeline can proceed to
Additive Feature Testing without Prepare Data having been run. The `suggest_next_step()`
and Menu 3 gate logic reflect this — Prepare Data is skipped if wavelet features exist.

**Gate 2 (Structure Confirmed) is required before Prepare Data:**
- Comprehensive Exploration (Menu 3 › 1) must have run
- Scientific Discovery (Menu 3 › 2) must find non-random structure
- Prime Instrument Discovery (Menu 3 › 3) must classify Tier-1 instruments
- If no structure found → pipeline STOPS — no alpha to find

**What Prepare Data (step 6 / option 12) outputs — physics-justified features only:**
- `velocity`, `energy`, `damping`, `entropy`, `reynolds`, `potential`, `eta`
- `BP`, `pe`, `liquidity`, `viscosity`
- Rolling-percentile sensors: `KE_pct`, `Hs_pct`, `Re_m_pct`, `zeta_pct`, …
- Regime: `cluster`, `regime`, `regime_age_frac`
- Holiday/session tags, normalised OHLCV

**What Prepare Data NEVER outputs (removed):**
- ❌ `MA_{6,12,24,48}`, `ATR_14`, `atr_pct` — magic-number TA
- ❌ `vol_ma_N`, `range_ma_N`, `close_vs_ma_N` — fixed-window derivatives
- ❌ `vol_spike`, `big_move`, `consec_up/down` — fixed-threshold features

These are either already captured adaptively by PhysicsEngine or must be
evaluated through Additive Feature Testing (Menu 3 › 5), not baked into files.

**Denoising is supplementary, not pre-processing:**
- Item 14 is hard-locked until Prepare Data (step 6) is complete.
- Denoising raw OHLCV destroys the entropy/regime signals PhysicsEngine detects.
- Use denoised data only as an ablation experiment input (Menu 3 › 5).
- Denoised output goes to `data/prepared_standardized/denoised/`, NOT the main prepared dir.

---

## Core Philosophy: First-Principles, Zero Assumptions

**CRITICAL**: Question everything! Even established "best practices" are hypotheses to explore,
not commandments.

### NEVER:
- ❌ Use magic numbers (20-period MA, 14-period RSI, etc.)
- ❌ Use traditional TA indicators without physics justification
- ❌ Assume linearity without proof
- ❌ Use fixed thresholds (e.g., stop at 2% ATR)
- ❌ Apply universal rules across markets without exploration
- ❌ Remove or modify working code without strong justification
- ❌ Add fixed-period TA derivatives to prepared data files (see Data Pipeline above)
- ❌ Denoise raw OHLCV before physics feature extraction

### ALWAYS:
- ✅ Start from thermodynamic/physical first principles
- ✅ Use rolling, adaptive distributions (NO fixed periods)
- ✅ Validate per-market, per-regime, per-timeframe
- ✅ Explore before implementing
- ✅ Question assumptions
- ✅ Let the data guide decisions
- ✅ **Prefer vectorization over Python loops** (NumPy/Pandas ops, broadcasting)
- ✅ Follow the canonical pipeline step order (ACQUIRE → VALIDATE → CURATE → SPECS → ORGANISE → [Gate 2: Structure Discovery] → PREPARE → AUDIT → RESEARCH → WAVELET → ADDITIVE)
- ✅ Require `agent_comparison_done` before trusting `models_trained` for UI gates (historical model artifacts on disk are not pipeline-current)
- ✅ **Design broker code for multi-broker reuse** — when refactoring MetaAPI modules, extract broker-neutral logic (see §28 in AGENT_RULES_MASTER.md)

**THE ONLY ASSUMPTION**: Physics is real (energy, friction, entropy exist in markets)

---

## Coding Standards

### Python Style
- Follow **PEP 8** conventions
- Use **Black** for code formatting (line length: 100)
- Use **Ruff ≥ 0.15** for linting — `pyproject.toml` is the single source of truth
- Target Python 3.10+
- Use type hints for all function signatures
- Prefer explicit over implicit

### Ruff Configuration (canonical)

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
"scripts/**"  = ["E402"]          # sys.path manipulation before imports is intentional
"tests/**"    = ["E402", "F401"]
"kinetra/dsp_features.py"   = ["E402"]
"kinetra/physics_engine.py" = ["E402"]
"kinetra/config.py"         = ["E402"]
"archive/production_cleanup_2026-03-03/scripts/analysis/superpot_explorer.py"       = ["E702"]  # features[fi]=…; fi+=1 idiom
"archive/production_cleanup_2026-03-03/scripts/research/analyze_results.py" = ["E701"]  # aligned column-map tables
"archive/production_cleanup_2026-03-03/scripts/research/measurement_toolkit.py"        = ["E701"]
"archive/production_cleanup_2026-03-03/scripts/analysis/superpot_complete.py"           = ["E701"]
"scripts/testing/**" = ["E722", "F401"]
```

> The `archive/` directory is **excluded** from all ruff checks — legacy code, not maintained.

### Ruff-Aware Coding Rules (zero-violation contract)

**Imports**
- `F401` — Remove unused imports. Optional deps inside `try/except ImportError` must have
  `# noqa: F401` on the import line.
- `F402` — Never reuse a module-level import name as a loop variable
  (e.g., `for stats in ...` shadows `from scipy import stats` — rename the loop variable).
- `E402` — All imports go at the top. If `sys.path` manipulation is unavoidable first
  (scripts only), rely on the `scripts/**` per-file ignore — do **not** scatter
  `# noqa: E402` inline.
- `I001` — Import order: stdlib → third-party → local. Ruff `--fix` handles this automatically.

```python
# ✅ GOOD: optional dep with noqa
try:
    from kinetra.performance import sample_entropy_fast  # noqa: F401
    _OPTIMIZED = True
except ImportError:
    _OPTIMIZED = False

# ❌ BAD: bare unused import at module level
import pandas_market_calendars   # never referenced
```

**Exceptions**
- `E722` — Never `except:`. Always `except Exception:` or a more specific type.

```python
# ✅ GOOD
try:
    result = risky_call()
except ValueError as e:
    log.warning("bad value: %s", e)
except Exception:
    pass  # intentional swallow — explain why in a comment

# ❌ BAD — catches SystemExit, KeyboardInterrupt, etc.
try:
    result = risky_call()
except:
    pass
```

**Variable names**
- `E741` is globally ignored for `l` (= *low* price in OHLCV) — the **only** exception.
  Never use `O` (zero lookalike) or `I` (one lookalike) anywhere.

```python
# ✅ GOOD: l = low price is allowed
o, h, l, c, v = df["open"], df["high"], df["low"], df["close"], df["volume"]

# ❌ BAD
O = order_book   # looks like zero
I = identity     # looks like one
```

**Statement style**
- `E702` — No semicoloned multi-statement lines outside `superpot_explorer.py`.
- `E701` — No `if cond: body` on one line outside the three approved research scripts.
- `E712` — Never `== True` / `== False`; use `is True` / `is False` or truthiness.

```python
# ✅ GOOD
if condition:
    do_something()

# ❌ BAD
if condition: do_something()   # E701
a = 1; b = 2                   # E702
if flag == True:               # E712
```

**Forward references**

When a type annotation references a class imported only inside a function (circular-import
avoidance), use `TYPE_CHECKING`:

```python
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from kinetra.unified_data_manager import UnifiedDataManager

def process(dm: "UnifiedDataManager") -> None: ...
```

**Undefined names (F821)**
- Never reference a variable before it is assigned.
- Constants used across functions must live at module level — not inside a docstring or a
  nested scope.

**Duplicate dict keys (F601)**
- Lazy-loader dicts (e.g., `_LAZY_MODULES` in `kinetra/__init__.py`) must not repeat the same
  string key. The last definition wins silently — a guaranteed hidden bug.

**Return outside function (F706)**
- `return` is only valid inside a function/method. In `if __name__ == "__main__"` blocks,
  use `sys.exit(code)` to terminate early.

**Pre-commit gate**

```bash
ruff check .   # must exit 0 before any commit
```

Acceptable `# noqa` suppressions:
- `# noqa: F401` — optional import inside `try/except ImportError`
- `# noqa: E402` — only if the file is **not** already covered by a per-file-ignores glob

Everything else: real fix or a new `[tool.ruff.lint.per-file-ignores]` entry with a rationale
comment.

### Code Quality
- **100% code coverage** for new features
- Property-based testing with `hypothesis` for mathematical functions
- Numerical stability checks (NaN shields, log-space calculations)
- Use Pydantic schemas for data validation

### Vectorization (MANDATORY)
```python
# ✅ PREFER: NumPy vectorized ops
energy = 0.5 * velocity ** 2

# ✅ PREFER: Pandas column operations
df["energy_pct"] = df["energy"].rolling(window).rank(pct=True)

# ✅ PREFER: Broadcasting
result = arr_2d + arr_1d[:, np.newaxis]

# ❌ LAST RESORT: Explicit Python loops
for i in range(len(data)):  # Only if unavoidable, keep tight
    ...
```

---

## Testing Requirements

### Multi-Layer Validation (Defense-in-Depth)

1. **Unit Tests** (`pytest`)
   - 100% code coverage required
   - Property-based testing with `hypothesis`
   - Numerical stability checks

2. **Integration Tests**
   - End-to-end pipeline validation
   - Physics → RL → Risk → Execution flow

3. **Monte Carlo Backtesting**
   - 100 runs per instrument minimum
   - Statistical significance testing (p < 0.01)
   - Out-of-sample validation required

4. **Theorem Validation**
   - Mathematical proofs must be documented
   - Continuous validation via CI/CD

5. **Health Monitoring**
   - Real-time Composite Health Score (CHS)
   - Circuit breakers (halt if CHS < 0.55)

### Running Tests
```bash
# Run all tests
make test
# or
pytest tests/ -v

# Run specific test file
pytest archive/production_cleanup_2026-03-03/repo/tests/test_physics.py -v

# Run with coverage
pytest tests/ --cov=kinetra --cov-report=html
```

---

## Data Safety & Integrity

### NEVER LOSE USER DATA (#1 Priority)

**Mandatory Before ANY Data Operation:**

1. ✅ **ALWAYS use `PersistenceManager.atomic_save()`** — Never raw file writes
2. ✅ **ALWAYS backup before git operations** — `git rm --cached` can delete files
3. ✅ **CHECK `.gitignore`** before commits — Large files must NEVER be tracked
4. ✅ **NEVER assume backups exist** — Verify before dangerous operations

```python
from kinetra.persistence_manager import get_persistence_manager

pm = get_persistence_manager(backup_dir="data/backups", max_backups=10)
pm.atomic_save(
    filepath="data/master/",
    content=df,
    writer=lambda path, data: data.to_csv(path, index=False),
)
```

---

## Security and Safety

### Credential Security
- **NEVER** commit API keys, secrets, or credentials
- Use environment variables (`.env` file) for sensitive data
- Reference `.env.example` for required variables

### Execution Safety
- Circuit breakers for abnormal conditions
- Fallback policies for error handling
- Slippage modeling in backtest
- Risk-of-Ruin (RoR) gates before execution

---

## Performance Targets

| Metric                    | Target | Purpose                        |
|---------------------------|--------|--------------------------------|
| **Omega Ratio**           | > 2.7  | Asymmetric returns             |
| **Z-Factor**              | > 2.5  | Statistical edge significance  |
| **% Energy Captured**     | > 65%  | Physics alignment efficiency   |
| **Composite Health Score**| > 0.90 | System stability               |
| **% MFE Captured**        | > 60%  | Execution quality              |

---

## Common Patterns

### Dynamic Thresholds (No Magic Numbers)
```python
# ✅ GOOD: Rolling percentiles
energy_75pct = np.percentile(history["energy"], 75)
if energy > energy_75pct:
    pass  # High energy regime

# ❌ BAD: Fixed threshold
if energy > 0.5:  # Magic number!
    pass
```

### NaN Protection
```python
# ✅ GOOD: Shield against NaN
value = np.clip(raw_value, MIN_VALUE, MAX_VALUE)
if not np.isfinite(value):
    value = fallback_value

# ❌ BAD: No protection
value = raw_value  # Could be NaN/Inf
```

### Validation Pattern
```python
# ✅ GOOD: Validate assumptions
assert len(data) > MIN_SAMPLES, "Insufficient data"
assert data["price"].notna().all(), "NaN in price data"
assert omega_ratio > 2.7, f"Omega {omega_ratio:.2f} below threshold"

# Include statistical validation
p_value = statistical_test(results)
assert p_value < 0.01, f"Not statistically significant (p={p_value})"
```

---

## When in Doubt

1. Check [`AGENT_RULES_MASTER.md`](../README.md) for complete rules
2. Run `ruff check .` — fix all violations before proceeding
3. Validate with statistical tests (p < 0.01)
4. Write comprehensive tests first
5. Question assumptions — "Is this a magic number?"
6. Reference physics first principles
7. Check [`archive/production_cleanup_2026-03-03/repo/docs/WORKFLOW.md`](../archive/production_cleanup_2026-03-03/repo/docs/WORKFLOW.md) for active pipeline work
8. **Before writing any utility function**, grep for an existing implementation and check [`archive/production_cleanup_2026-03-03/repo/docs/TRADING_STRATEGY_SPEC.md`](../archive/production_cleanup_2026-03-03/repo/docs/TRADING_STRATEGY_SPEC.md) for the canonical home
9. **Before adding any menu output**, read §30 of `AGENT_RULES_MASTER.md` — the UI styling guide

---

## Additional Resources

- **Renko Design Spec**: [`archive/production_cleanup_2026-03-03/repo/docs/TRADING_STRATEGY_SPEC.md`](../archive/production_cleanup_2026-03-03/repo/docs/TRADING_STRATEGY_SPEC.md) — **7-phase architecture** (Sprint 1–2 complete)
- **Renko Thread Context**: [`archive/production_cleanup_2026-03-03/repo/docs/TRADING_STRATEGY_SPEC.md`](../archive/production_cleanup_2026-03-03/repo/docs/TRADING_STRATEGY_SPEC.md) — decision history & empirical basis
- **Complete Rules**: [`AGENT_RULES_MASTER.md`](../README.md) — **START HERE** (v4.9)
  - §30   — **Menu & Terminal UI Styling Guide** ✅ CANONICAL ← **read before any menu edit**
  - §29   — **Sprint 5 Architecture — Industrialisation** ⭐ CURRENT
  - §28   — **Multi-broker architecture** (cTrader planned) 📐 DESIGN
  - §27   — **DRY violations register** ⭐ ACTIVE
  - §26   — Gate-logic bugfixes (2026-02-27)
  - §25   — Menu restructure & pipeline correction ✅ COMPLETE
  - §22   — Data foundation fix (clean instrument sets)
  - §23   — Canonical pipeline architecture & step order
  - §2.10 — Data preparation philosophy (what belongs in prepared files)
  - §20   — Feature engineering & additive testing protocol
  - §21   — Empirical ablation results (canonical_v2.2)
- **DRY Remediation Plan**: [`archive/production_cleanup_2026-03-03/repo/docs/TRADING_STRATEGY_SPEC.md`](../archive/production_cleanup_2026-03-03/repo/docs/TRADING_STRATEGY_SPEC.md) — **19 violations tracked; Tier 1 (DRY-01–07) are Sprint 1–2**
- **Menu Restructure Plan**: [`archive/production_cleanup_2026-03-03/repo/docs/WORKFLOW.md`](../archive/production_cleanup_2026-03-03/repo/docs/WORKFLOW.md) — **Phases A–D COMPLETE; all sign-off items done**
- **Instrument Selection**: `archive/production_cleanup_2026-03-03/repo/docs/TRADING_STRATEGY_SPEC.md`
- **Design Bible**: Complete architecture in `docs/` directory
- **Theorem Proofs**: Mathematical validation in `QUICK_START.md`
- **Empirical Theorems**: Data-driven discoveries in `QUICK_START.md`
- **Testing Guide**: `QUICK_START.md`
- **Operator Playbook**: `QUICK_START.md` — remediation procedures, gate debugging, manifest management
- **Reward Consolidation**: `archive/production_cleanup_2026-03-03/repo/docs/IMPLEMENTATION_SUMMARY.md` — RewardOrchestrator consolidation
- **Migration Guide**: `archive/production_cleanup_2026-03-03/repo/docs/INTEGRATION_GUIDE.md` — legacy ARS → RewardOrchestrator

---

## Menu & Terminal UI Styling — Quick Reference (§30)

> **Full rules → [`AGENT_RULES_MASTER.md §30`](../README.md)**
> Never introduce new visual constructs without reading §30 first.

### Header Types (exactly two — never add a third)

| Type | Function | Bar char | Width | When used |
|---|---|---|---|---|
| **Submenu header** | `print_submenu_header(text, status)` | `═` (U+2550) | 80 | Once per `while True:` menu loop |
| **Action screen header** | `print_header(text)` | `─` (U+2500) | 80 | Once per action function |

**`print_header` title format:** `EMOJI Label  (Menu N › M)` — location ref is **mandatory**.

```python
print_header("🧱 Renko Instrument Backtest  (Menu 4 › 3)")
print_header("🛡️  Train Renko Risk Agent  (Menu 3 › 12 — Layer 3)")
print_header("🚧 Generate Performance Report  (Menu 4 › 9 — PLANNED)")
```

### Section Divider Lines

Always exactly **80 characters** total.  Always a standalone `print()` — **never** inside a `"""..."""` block.

```python
# Generate correctly:
def _sep(label: str, total: int = 80) -> str:
    prefix, suffix = "  ── ", " "
    return f"{prefix}{label}{suffix}{'─' * (total - len(prefix) - len(label) - len(suffix))}"

print(_sep("STEP 2 · VALIDATE"))          # "  ── STEP 2 · VALIDATE ─────…" (80 chars)
print(_sep("Training configuration (Layer 3)"))
```

### ANSI Colour Helpers — use only these

```python
_bold(text)    # option numbers, header titles
_dim(text)     # locked items, hints
_yellow(text)  # warnings / missing prerequisites
_green(text)   # success states (prefer ✅ emoji instead)
_red(text)     # errors (prefer ❌ emoji instead)
_cyan(text)    # informational labels (rare)
```

❌ **Never** use raw `\033[…m` sequences.  ❌ **Never** import `colorama`, `rich`, or `termcolor`.

### `_print_menu_item()` — all option lines go through this

```python
_print_menu_item(
    number="6",
    label="Fill M1 Data Gaps",          # title case, no trailing punct
    hint="(extend M1 to today + cron)", # (parens), lowercase
    available=locks["6"][0],
    lock_reason="download data first (option 2, 3, or 4)",  # lowercase, actionable
)
```

### Emoji Vocabulary

| Icon | Meaning / domain |
|---|---|
| `✅` | complete / healthy / configured |
| `❌` | missing / failed |
| `⚠️` | warning / stale |
| `🚨` | critical / blocking |
| `🔒` | locked — prerequisite not met |
| `💡` | next-step suggestion (prefix only) |
| `🚧` | planned / not implemented |
| `🗺️` | exploration / mapping |
| `🔭` | scientific discovery |
| `🎯` | targeting / optimisation |
| `⚖️` | comparison |
| `🛡️` | risk / protection |
| `📊` | data / results / reports |
| `📋` | registry / table |
| `🔍` | search / qualify |
| `🧱` | Renko / backtest |
| `🧹` | curation / cleanup |
| `🔁` | refresh / rewrite |
| `🔄` | drift / recalibration |
| `📈` | Backtesting menu |
| `🔐` | credentials |
| `🛠️` | system tools |

❌ Never use hearts, stars, sparkles, or decorative emoji.  ❌ Never use `✔` / `✗` — use `✅` / `❌`.

### Inline Text Conventions

| Symbol | Use |
|---|---|
| `→` | pipeline flow: `M1 → M30` |
| `←` | status annotation: `← ✅ 19 qualified` |
| `›` | menu path: `Menu 2 › 12` |
| `·` | inline label separator: `STEP 2 · VALIDATE` |

- Ages: `14d ago`, `today` (never `0d ago`)
- Omega: `Ω=6.07` (Unicode, not `O=`)
- Layers: `L2`, `L3` (not `layer 2`)

### "Press Enter" Prompt — exact wording, no variation

```python
input("\n  📌  Press Enter to continue…")
```

### New Screen Checklist

```
[ ] print_header("EMOJI Label  (Menu N › M)") is the first line
[ ] All options use _print_menu_item() — no raw print() for option lines
[ ] Section dividers are exactly 80 chars — verify with len()
[ ] No print("  ── …") inside a """…""" block
[ ] run_script() used for all subprocess calls
[ ] input("\n  📌  Press Enter to continue…") at the call site
[ ] New option added to _data_menu_locks() or equivalent gate function
[ ] New option wired into the while-loop elif chain
[ ] valid_choices list updated
```

### ASCII Block-Code / Box-Art Diagrams (§30.12)

> **Full rules → [`AGENT_RULES_MASTER.md §30.12`](../README.md)**

Two categories exist — never add a third without updating §30.12.

**1. Architecture diagrams** (Layer 2/3 training screens)

- `inner_arch = 65` — display columns between border chars (canonical for all arch diagrams)
- Outer line width: `4-space indent + 1 border + 65 inner + 1 border = 71 terminal cells`
- Title bar: `20 left dashes + " LABEL " + fill dashes = 65`
- Content lines: `len()` is sufficient (ASCII only)
- Arrow connector between boxes: `" " * 33 + "▼"`

```python
_I = 65  # inner_arch

def _box(title: str, lines: list[str]) -> list[str]:
    left = 20
    label = f" {title} "
    right = _I - left - len(label)
    top    = "┌" + "─" * left + label + "─" * right + "┐"
    bottom = "└" + "─" * _I + "┘"
    body   = ["│" + ln + " " * (_I - len(ln)) + "│" for ln in lines]
    return [top, *body, bottom]

_arrow = " " * 33 + "▼"
indent = "    "
for line in [*_box("LAYER 3 — RISK RL AGENT  ◄── THIS", [
    "  Output: exposure_scalar ∈ [0.0, 1.0]",
    '  Question: "Should we be in the market at all right now?"',
]), _arrow, *_box("LAYER 1 — DETERMINISTIC ENGINE", [
    "  Renko flip + filter → trade at allocated weight",
])]:
    print(indent + line)
```

❌ **Never** embed arch boxes in a triple-quoted string — source indentation shifts the right border.

**2. Splash / logo block** — `_splash()` in `main()`

- `_INNER = 75`, double-line borders `╔ ═ ╗ / ║ / ╚ ═ ╝`
- Contains block-art `█` characters — **must use `wcwidth.wcswidth()`**, never `len()`
- Import inside the function with `except ImportError` fallback for CI

```python
try:
    from wcwidth import wcswidth as _wcswidth
except ImportError:  # noqa: F401
    _wcswidth = len  # type: ignore[assignment]

_INNER = 75

def _row(content: str) -> str:
    w = _wcswidth(content)
    return "║" + content + " " * (_INNER - w) + "║"
```

❌ **Never** replace `_splash()` with a static string — it misaligns on 2-column-char terminals.

**3. Inline planned-feature box** (e.g. Performance Report screen)

- 2-space print-indent, `─ / │ / ┌ ┘` single-line borders, inner width **73**
- `len()` is sufficient (ASCII content only)
- Total line width: `2 indent + 1 border + 73 inner + 1 border = 77 terminal cells`

❌ **Never** mix `inner_arch=65` (arch diagrams) with `inner=73` (inline boxes).

**Box-art checklist:**
```
[ ] All boxes in one diagram share inner_arch=65
[ ] Title bar: 20 left dashes + " LABEL " + fill = 65 exactly
[ ] Content lines padded with ASCII spaces to exactly inner_arch
[ ] Arrow connector: " " * 33 + "▼"
[ ] Splash uses wcswidth() — inner width 75, double-line borders
[ ] Planned-feature box: inner width 73, 2-space indent, single-line borders
[ ] No box-art inside a triple-quoted string that is indented in source
[ ] Verify widths: python -c "from wcwidth import wcswidth; print(wcswidth('    ┌─…─┐'))"
```

---

- **Operator Health CLI**: `scripts/monitor_daemon.py` — `--json` for external alerting

---

**Remember**: Kinetra is built on first principles with rigorous validation. Every decision should
be justified mathematically, tested statistically, and validated continuously. The ruff linter is
always at **zero violations** — keep it that way.

---

## Multi-Broker Architecture — Quick Reference (§28)

> **cTrader Open API** integration is **planned** (Pepperstone cTrader account available).
> MetaAPI must work end-to-end first.  Current refactoring should build toward multi-broker
> support without adding cTrader code yet.

### The Broker-Neutral Boundary (CRITICAL)

Everything downstream of `data/master_standardized/` and `InstrumentSpec` is **broker-blind**.
No physics, feature, backtest, training, or reward module may import a broker SDK.

```
BROKER-AWARE (upstream)          │  BROKER-BLIND (downstream)
─────────────────────────────────│──────────────────────────────────
  MetaAPI / cTrader connector    │  master_standardized/ CSVs
  Download handlers              │  InstrumentSpec dataclass
  Spec pollers                   │  PhysicsEngine, features, training
  Format converters              │  Backtesting, reward orchestrator
```

### When Refactoring MetaAPI Code — Ask:

> "Would a cTrader implementation need to duplicate this logic?"

- **Yes** → extract into a broker-neutral helper (FX enrichment, save/merge, OHLCV normalisation, retry/backoff)
- **No** → keep broker-specific but behind a clear interface

### Multi-Broker Hard Rules

- ❌ **Never** import broker SDKs in `kinetra/` library modules (except dedicated connectors) — always `try/except ImportError`
- ❌ **Never** assume symbol names are identical across brokers (`XAUUSD` vs `GOLD`)
- ❌ **Never** hardcode MetaAPI field names (e.g. `tickSize`) outside the MetaAPI adapter — `InstrumentSpec` is the canonical schema
- ❌ **Never** add broker-specific logic to physics, backtesting, training, or reward modules
- ✅ **Always** normalise OHLCV to canonical schema (`time, open, high, low, close, volume`) before writing to `master_standardized/`
- ✅ **Always** normalise specs to `InstrumentSpec` fields before writing `contract_spec.json`
- ✅ **Always** isolate broker connection state — no cross-broker session bleed

### Phased Plan

1. **Phase 1 (CURRENT):** MetaAPI end-to-end.  No cTrader code.  But extract shared logic when refactoring.
2. **Phase 2 (✅ COMPLETE):** Broker-neutral ABCs and shared utilities extracted:
   - `kinetra/broker_compliance.py` — `BrokerConnection`, `BrokerDataHandler`, `BrokerSpecHandler` ABCs; `normalize_ohlcv()`, `detect_ohlcv_schema()`, `OHLCVSchema`, `BrokerInfo`, `SpecPollResult`; timeframe/symbol/category utilities; pre-built schemas (`MT5_SCHEMA`, `METAAPI_SCHEMA`, `CTRADER_SCHEMA`)
   - `kinetra/friction_cost.py` — FX enrichment (`enrich_with_fx_rates`), save/merge (`save_specs`), symbol discovery (`discover_symbols`), quote-currency inference, dir lookup, contract spec loading — all extracted from `poll_symbol_specs.py`
   - `kinetra/data_utils.py` — `load_broker_csv()` canonical broker-neutral CSV loader with auto-detection
   - `kinetra/friction_cost.py` — `InstrumentSpec.from_broker_json(raw, source='metaapi'|'mt5'|'ctrader')` source-aware factory
   - 134 tests in `archive/production_cleanup_2026-03-03/repo/tests/test_integration.py`
3. **Phase 3 (future):** `kinetra/connectors/ctrader_connector.py`, `poll_ctrader_specs.py`, OAuth2 token management, menu wiring.

> **Broker data source strategy (§29.4):** Use **cTrader as primary research feed** (tighter spreads, cleaner UTC alignment). Do NOT overlay MetaAPI + cTrader data for the same instrument — correlated duplicates with different artifacts. Brick sequences and filter parameters transfer cross-broker; friction floor, VPIN baseline, session break UTC, and circuit breaker thresholds are broker-specific and must be recalibrated per broker. Always store `broker_source` and session gap UTC times in `spread_profile.json`.

See [`AGENT_RULES_MASTER.md §28`](../README.md) for the complete design.

---

## DRY — Quick Reference (§27)

### Canonical Locations — import, don't reinvent

| What you need | Import from |
|---------------|-------------|
| Per-instrument qualification | `kinetra.renko.qualify.qualify_instrument` 🔲 Sprint 5B |
| Qualification registry | `kinetra.renko.qualify.QualificationRegistry` 🔲 Sprint 5B |
| Qualification result | `kinetra.renko.qualify.QualificationResult` 🔲 Sprint 5B |
| Calibration drift detector | `kinetra.renko.qualify.CalibrationDriftDetector` 🔲 Sprint 5C |
| Session break detection | `kinetra.renko.session.detect_session_break` 🔲 Sprint 5B |
| Session profile | `kinetra.renko.session.SessionProfile` 🔲 Sprint 5B |
| Loss-cluster risk params | `kinetra.renko.backtest.RiskParams` 🔲 Sprint 5A |
| Scaled filter params | `kinetra.renko.dsp.scaled_filter_params` 🔲 Sprint 5A |
| Portfolio pipeline orchestrator | `kinetra.renko.orchestrator.run_full_pipeline` 🔲 Sprint 5C |
| Portfolio pipeline result | `kinetra.renko.orchestrator.PortfolioPipelineResult` 🔲 Sprint 5C |
| InstrumentContext recalibration | `kinetra.rl.portfolio_env.InstrumentContext.recalibrate` ✅ exists (Sprint 6) |
| Agent comparison detection | `kinetra.model_manifest.detect_agent_comparison()` ✅ exists |
| Wavelet step reader | `kinetra.model_manifest.read_wavelet_step()` ✅ exists |
| Agent CMP file path | `kinetra.model_manifest.AGENT_CMP_RELPATH` ✅ exists |
| Blacklist read / write | `kinetra.data_gap_tools.read_blacklist()` / `write_blacklist()` ✅ exists |
| `is_blacklisted()` check | `kinetra.data_gap_tools.is_blacklisted()` ✅ exists |
| Gap scan / classify | `kinetra.data_gap_tools.scan_and_classify_gaps()` ✅ exists |
| Max fill bars constant | `kinetra.market_calendar.MAX_FILL_BARS` ✅ exists |
| Omega ratio (module-level) | `kinetra.backtesting.metrics.omega_ratio` ✅ exists |
| Omega ratio (class method) | `kinetra.backtesting.metrics.MetricsCalculator.omega_ratio` ✅ exists |
| Z-factor (returns array) | `kinetra.backtesting.metrics.calculate_z_factor` ✅ exists |
| Z-factor (trade dicts) | `kinetra.backtesting.metrics.MetricsCalculator.z_factor` ✅ exists |
| ATR / adaptive volatility | `kinetra.volatility_utils` ✅ exists |
| Project root path | `kinetra.config.PROJECT_ROOT` ✅ exists |
| Friction / instrument spec | `kinetra.friction_cost.InstrumentSpec` ✅ canonical (replaces deprecated `symbol_spec`, `symbol_specs`, `symbol_info`) |
| Fast CSV time-column read | `kinetra.csv_reader.read_time_column_fast` ✅ exists (PyArrow — 38× faster; gap scanner hot path) |
| Fast CSV OHLCV read | `kinetra.csv_reader.read_ohlcv_fast` ✅ exists (PyArrow — 25× faster; aggregation, integrity hot paths) |
| Fast CSV general read | `kinetra.csv_reader.read_csv_fast` ✅ exists (PyArrow with auto-fallback to pandas) |
| PyArrow availability check | `kinetra.csv_reader.has_pyarrow` ✅ exists |
| Load CSV data (broker-neutral) | `kinetra.data_utils.load_broker_csv` ✅ canonical (§28 Phase 2 — auto-detects MT5/MetaAPI/cTrader format) |
| Load CSV data (MT5 format) | `kinetra.data_utils.load_mt5_csv` ✅ exists (legacy — prefer `load_broker_csv`) |
| OHLCV normalisation | `kinetra.broker.normalize_ohlcv` ✅ exists (§28 Phase 2 — canonical normalisation chokepoint) |
| OHLCV schema detection | `kinetra.broker.detect_ohlcv_schema` ✅ exists (§28 Phase 2) |
| Broker connection ABC | `kinetra.broker.BrokerConnection` ✅ exists (§28 Phase 2) |
| Broker data handler ABC | `kinetra.broker.BrokerDataHandler` ✅ exists (§28 Phase 2) |
| Broker spec handler ABC | `kinetra.broker.BrokerSpecHandler` ✅ exists (§28 Phase 2) |
| Broker identity metadata | `kinetra.broker.BrokerInfo` ✅ exists (§28 Phase 2) |
| Symbol suffix stripping | `kinetra.broker.strip_broker_suffix` ✅ exists (§28 Phase 2) |
| Category inference | `kinetra.broker.infer_category` ✅ exists (§28 Phase 2) |
| Timeframe normalisation | `kinetra.broker.canonical_timeframe` ✅ exists (§28 Phase 2) |
| InstrumentSpec (source-aware) | `kinetra.friction_cost.InstrumentSpec.from_broker_json()` ✅ exists (§28 Phase 2) |
| FX enrichment (spec polling) | `kinetra.spec_utils.enrich_with_fx_rates` ✅ exists (§28 Phase 2) |
| Spec save/merge orchestration | `kinetra.spec_utils.save_specs` ✅ exists (§28 Phase 2) |
| Symbol discovery | `kinetra.spec_utils.discover_symbols` ✅ exists (§28 Phase 2) |
| Quote currency inference | `kinetra.spec_utils.infer_quote_currency` ✅ exists (§28 Phase 2) |
| Instrument dir lookup | `kinetra.spec_utils.find_instrument_dir` ✅ exists (§28 Phase 2) |
| Contract spec loading | `kinetra.spec_utils.load_contract_spec` ✅ exists (§28 Phase 2) |
| Core data manager | `kinetra.data.DataManager` ✅ canonical (replaces deprecated `data_manager`, `unified_data_manager`) |
| Download manager | `kinetra.data.download.DownloadManager` ✅ canonical |
| Integrity checker | `kinetra.data.integrity.IntegrityChecker` ✅ canonical |
| Cache manager | `kinetra.data.cache.CacheManager` ✅ canonical |
| Backtesting (standard) | `kinetra.backtesting.core.UnifiedBacktester(mode='standard')` ✅ canonical (replaces deprecated `backtest_engine`) |
| Backtesting (MT5-realistic) | `kinetra.backtesting.core.UnifiedBacktester(mode='realistic')` ✅ canonical (replaces deprecated `realistic_backtester`) |
| Backtesting (physics) | `kinetra.backtesting.core.UnifiedBacktester(mode='physics')` ✅ canonical (replaces deprecated `physics_backtester`) |
| Backtesting (portfolio) | `kinetra.backtesting.core.UnifiedBacktester(mode='portfolio')` ✅ canonical (replaces deprecated `portfolio_backtest`) |
| Portfolio health dataclass | `kinetra.portfolio_health.PortfolioHealthScore` ✅ exists (renamed from `CompositeHealthScore`) |
| Strategy health scorer | `kinetra.health_score.CompositeHealthScore` ✅ exists (multi-factor, unambiguous) |
| Renko brick construction | `kinetra.renko.brick_engine.build_renko` ✅ exists |
| Renko FlipRate filter | `kinetra.renko.filters.flip_rate` ✅ exists |
| Renko Markov stickiness | `kinetra.renko.filters.markov_stickiness` ✅ exists |
| Renko entry evaluation | `kinetra.renko.filters.evaluate_entry` ✅ exists |
| Renko DSP analysis | `kinetra.renko.dsp.run_dsp` ✅ exists |
| Renko friction floor | `kinetra.renko.dsp.compute_friction_floor` ✅ exists |
| VPIN (scalar) | `kinetra.renko.vpin.compute_vpin` ✅ exists (Sprint 4) |
| VPIN time series | `kinetra.renko.vpin.vpin_timeseries` ✅ exists (Sprint 4) |
| VPIN baseline stats | `kinetra.renko.vpin.vpin_baseline` ✅ exists (Sprint 4) |
| VPIN multi-instrument | `kinetra.renko.vpin.compute_vpin_multi` ✅ exists (Sprint 4) |
| VPIN normalisation | `kinetra.renko.vpin.normalise_vpin` ✅ exists (Sprint 4) |
| VPIN regime classification | `kinetra.renko.vpin.classify_vpin_regime` ✅ exists (Sprint 4) |
| VPIN extreme detection | `kinetra.renko.vpin.is_vpin_extreme` ✅ exists (Sprint 4) |
| VPIN auto bucket sizing | `kinetra.renko.vpin.auto_bucket_size` ✅ exists (Sprint 4) |
| Circuit breaker manager | `kinetra.monitoring.circuit_breakers.CircuitBreakerManager` ✅ exists (Sprint 4) |
| Circuit breaker evaluation | `kinetra.monitoring.circuit_breakers.evaluate_circuit_breakers` ✅ exists (Sprint 4) |
| VPIN breaker check | `kinetra.monitoring.circuit_breakers.check_vpin_breaker` ✅ exists (Sprint 4) |
| Drawdown breaker check | `kinetra.monitoring.circuit_breakers.check_drawdown_breaker` ✅ exists (Sprint 4) |
| Renko instrument backtest | `kinetra.renko.backtest.backtest_instrument` ✅ exists |
| Renko portfolio backtest | `kinetra.renko.backtest.backtest_portfolio` ✅ exists |
| Renko walk-forward IS/OOS | `kinetra.renko.backtest.walk_forward_instrument` ✅ exists |
| Renko Monte Carlo validation | `kinetra.renko.backtest.monte_carlo_instrument` ✅ exists |
| Renko brick sweep (full) | `kinetra.renko.backtest.sweep_brick_sizes` ✅ exists |
| Renko friction stress test | `kinetra.renko.backtest.stress_test_friction` ✅ exists |
| Renko filter params config | `kinetra.renko.backtest.FilterParams` ✅ exists |
| Renko trade record | `kinetra.renko.backtest.RenkoTrade` ✅ exists |
| Renko cluster assignment | `kinetra.renko.portfolio.get_cluster` ✅ exists |
| Renko cluster taxonomy | `kinetra.renko.portfolio.CLUSTER_MAP` ✅ exists |
| Renko equal-risk sizing | `kinetra.renko.portfolio.equal_risk_weights` ✅ exists |
| Renko cluster capping | `kinetra.renko.portfolio.apply_cluster_caps` ✅ exists |
| Renko spot/futures dedup | `kinetra.renko.portfolio.deduplicate_underlyings` ✅ exists |
| Renko portfolio equity curve | `kinetra.renko.portfolio.build_portfolio_equity` ✅ exists |
| Renko portfolio builder | `kinetra.renko.portfolio.build_portfolio` ✅ exists |
| Renko Herfindahl index | `kinetra.renko.portfolio.herfindahl_index` ✅ exists |
| Renko tail-risk analysis | `kinetra.renko.portfolio.tail_risk_analysis` ✅ exists |
| Renko USD/point estimation | `kinetra.renko.portfolio.estimate_usd_per_point` ✅ exists |
| Renko allocation reward | `kinetra.rl.reward.compute_allocation_reward` ✅ exists |
| Renko risk reward | `kinetra.rl.reward.compute_risk_reward` ✅ exists |
| Renko trade reward | `kinetra.rl.reward.compute_trade_reward` ✅ exists |
| Renko portfolio reward | `kinetra.rl.reward.compute_portfolio_reward` ✅ exists |
| Renko terminal reward | `kinetra.rl.reward.compute_terminal_reward` ✅ exists |
| Renko allocation config | `kinetra.rl.reward.AllocationRewardConfig` ✅ exists |
| Renko risk config | `kinetra.rl.reward.RiskRewardConfig` ✅ exists |
| Renko trade outcome | `kinetra.rl.reward.TradeOutcome` ✅ exists |
| Renko reward tracker | `kinetra.rl.reward.RewardTracker` ✅ exists |
| Renko risk reward tracker | `kinetra.rl.reward.RiskRewardTracker` ✅ exists |
| Renko portfolio env | `kinetra.rl.portfolio_env.RenkoPortfolioEnv` ✅ exists |
| Renko instrument context | `kinetra.rl.portfolio_env.InstrumentContext` ✅ exists |
| Renko portfolio env config | `kinetra.rl.portfolio_env.PortfolioEnvConfig` ✅ exists |
| Renko risk overlay env | `kinetra.rl.risk_env.RiskOverlayEnv` ✅ exists |
| Renko portfolio day snapshot | `kinetra.rl.risk_env.PortfolioDaySnapshot` ✅ exists |
| Renko risk env config | `kinetra.rl.risk_env.RiskEnvConfig` ✅ exists |
| M1 → higher TF aggregation | `kinetra.aggregation.aggregate_ohlcv` ✅ exists |
| M1 download chunking | `scripts.download.download_core.backward_chunk_download` ✅ exists |
| Train allocation agent (L2) | `scripts/train.py` ✅ exists (Sprint 4) |
| Train risk agent (L3) | `scripts/train.py` ✅ exists (Sprint 4) |
| Compare Renko agents (L2+L3) | `archive/production_cleanup_2026-03-03/scripts/training/explore_compare_agents.py` ✅ exists (Sprint 4) |
| Reward weight sweep (L2+L3) | `scripts/run_hpo.py` ✅ exists (Sprint 4) |

### Hard Rules (§29 additions)

- ❌ **Never** call `build_renko()` on a new data file without first running `detect_session_break()` — gaps ≥ 30 min across rollover create brick bursts that corrupt FlipRate, Markov, and VPIN
- ❌ **Never** hardcode `FilterParams` defaults for a new instrument — use `scaled_filter_params(dsp_result, bricks_per_day)` (Sprint 5A)
- ❌ **Never** inline qualification logic in a script — use `qualify_instrument()` from `kinetra.renko.qualify`
- ❌ **Never** assemble `instrument_data` dict manually for `build_portfolio()` — use `QualificationRegistry.get_qualified()`
- ❌ **Never** use RL to calibrate brick size or filter thresholds — use `run_dsp()` + `scaled_filter_params()`
- ❌ **Never** overlay MetaAPI + cTrader data for the same instrument as "diversity" — correlated duplicates, not independent variation
- ❌ **Never** use cTrader spread data to qualify instruments for MetaAPI deployment (or vice versa)
- ❌ **Never** re-open signal research — the Renko core (flip + fixed Markov gate) is locked (§29.1)
- ❌ **Never** treat brick size as an optimisation parameter — it is a structural DSP measurement
- ❌ **Never** duplicate loss-cluster / DD throttle logic inline — use `kinetra.renko.backtest.RiskParams`
- ✅ **Always** pass `session_break_minutes` from `SessionProfile` to `build_renko()`
- ✅ **Always** store `broker_source` and session gap UTC times in `spread_profile.json`
- ✅ **Always** run `CalibrationDriftDetector.check()` on a monthly schedule in live deployment (Sprint 5C+)
- ✅ **RL adapts WITHIN a calibrated parameter set** (allocation weights, exposure scalar) — never calibrates the parameter set itself

### Hard Rules (pre-existing)

- ❌ **Never** copy `_detect_agent_comparison()` into another training script
- ❌ **Never** add another `omega_ratio()` standalone function to a script — import from `kinetra.backtesting.metrics`
- ❌ **Never** add another `calculate_z_factor()` function to any module — import from `kinetra.backtesting.metrics`
- ❌ **Never** redefine `MAX_FILL_BARS` with a "keep in sync" comment
- ❌ **Never** add blacklist / gap-scan logic inline — it belongs in `scripts/download/check_and_fill_data.py`
- ❌ **Never** write `project_root = Path(__file__).parent.parent` in library code — import `PROJECT_ROOT` from `kinetra.config`
- ❌ **Never** use `portfolio_health.CompositeHealthScore` in new code — use `PortfolioHealthScore` instead
- ❌ **Never** import from `kinetra.symbol_spec`, `kinetra.symbol_specs`, or `kinetra.symbol_info` in new code — use `kinetra.friction_cost.InstrumentSpec` (DRY-09)
- ❌ **Never** import from `kinetra.data_manager` or `kinetra.unified_data_manager` in new code — use `kinetra.data.DataManager` (DRY-10)
- ❌ **Never** import from `kinetra.backtest_engine`, `kinetra.realistic_backtester`, `kinetra.physics_backtester`, `kinetra.backtest_optimizer`, `kinetra.integrated_backtester`, or `kinetra.portfolio_backtest` in new code — use `kinetra.backtesting.core.UnifiedBacktester` (DRY-11)
- ❌ **Never** import broker SDKs (`metaapi_cloud_sdk`, `MetaTrader5`, cTrader) in library modules except dedicated connectors — always behind `try/except ImportError` (§28)
- ❌ **Never** hardcode MetaAPI-specific field names outside the MetaAPI adapter layer — use `InstrumentSpec` (§28)
- ❌ **Never** write a new OHLCV CSV loader — use `kinetra.data_utils.load_broker_csv` (§28 Phase 2)
- ❌ **Never** duplicate PyArrow CSV logic — import from `kinetra.csv_reader` (`read_time_column_fast`, `read_ohlcv_fast`, `read_csv_fast`)
- ❌ **Never** use `pd.read_csv(..., engine='pyarrow')` directly — it lacks the timestamp type-hint that gives the real speedup; use `kinetra.csv_reader` instead
- ❌ **Never** use `pd.read_csv` + `pd.to_datetime` in hot paths (gap scanner, aggregation, integrity checker) — use `kinetra.csv_reader.read_time_column_fast` or `read_ohlcv_fast` for 25–38× speedup
- ❌ **Never** duplicate FX enrichment or spec save/merge logic — use `kinetra.spec_utils` (§28 Phase 2)
- ❌ **Never** duplicate symbol discovery or category inference — use `kinetra.spec_utils.discover_symbols` / `kinetra.broker.infer_category` (§28 Phase 2)
- ❌ **Never** duplicate `build_renko()` — import from `kinetra.renko.brick_engine`
- ❌ **Never** duplicate `flip_rate()` or `markov_stickiness()` — import from `kinetra.renko.filters`
- ❌ **Never** duplicate cluster taxonomy — import `CLUSTER_MAP` / `get_cluster()` from `kinetra.renko.portfolio`
- ❌ **Never** duplicate VPIN computation — import from `kinetra.renko.vpin` (Sprint 4)
- ❌ **Never** duplicate circuit breaker logic — import from `kinetra.monitoring.circuit_breakers` (Sprint 4)
- ❌ **Never** inline VPIN extreme detection — use `kinetra.renko.vpin.is_vpin_extreme` (Sprint 4)
- ❌ **Never** hardcode drawdown/VPIN/spread/correlation safety thresholds — use `CircuitBreakerConfig` (Sprint 4)
- ❌ **Never** implement non-learned safety limits outside `kinetra.monitoring.circuit_breakers` — all hard limits are centralised there (Sprint 4)
- ❌ **Never** duplicate equity curve merging — use `kinetra.renko.portfolio.build_portfolio_equity`
- ❌ **Never** inline Renko backtesting logic — use `kinetra.renko.backtest.backtest_instrument` / `backtest_portfolio`
- ❌ **Never** duplicate Renko reward logic — import from `kinetra.rl.reward`
- ❌ **Never** build a new Renko RL env — use `kinetra.rl.portfolio_env.RenkoPortfolioEnv` (Layer 2) or `kinetra.rl.risk_env.RiskOverlayEnv` (Layer 3)
- ❌ **Never** duplicate `InstrumentContext` — import from `kinetra.rl.portfolio_env`
- ❌ **Never** duplicate `PortfolioDaySnapshot` — import from `kinetra.rl.risk_env`
- ✅ **Always** check `archive/production_cleanup_2026-03-03/repo/docs/TRADING_STRATEGY_SPEC.md` before touching any open DRY-NN item
- ✅ **Always** run `pytest tests/ -q && ruff check .` after any DRY consolidation
- ✅ **When refactoring MetaAPI code**, extract broker-neutral logic into shared helpers where a future cTrader implementation would otherwise duplicate it (§28)
- ✅ **When building a new broker adapter**, implement `BrokerConnection`, `BrokerDataHandler`, and `BrokerSpecHandler` from `kinetra.broker` (§28 Phase 2)

**Key changes (2026-03-01) — DRY Audit & Remediation Plan (Sprints 1–5 complete):**
- `archive/production_cleanup_2026-03-03/repo/docs/TRADING_STRATEGY_SPEC.md` — Full DRY violations register (19 items, Tiers 1–4); v1.3 as of Sprint 5:
  - **Tier 1 (✅ ALL DONE, Sprint 1–2):** DRY-01 `_detect_agent_comparison` ×7, DRY-02 wavelet step reader ×8, DRY-03 `_AGENT_CMP_FILE` ×10, DRY-04 `_is_blacklisted` ×2, DRY-05 `_read_blacklist` ×2, DRY-06 `_scan_and_classify_gaps` stranded in menu, DRY-07 `_MAX_FILL_BARS` redefined
  - **Tier 2 (✅ DONE: DRY-08, DRY-12, DRY-13 — Sprint 3–4; DRY-09/10/11 Phase A — Sprint 5):** DRY-08 `omega_ratio` canonical + thin wrappers; DRY-12 `CompositeHealthScore` → `PortfolioHealthScore` rename; DRY-13 `calculate_z_factor` canonical module-level function; DRY-09 `symbol_spec`/`symbol_specs`/`symbol_info` DeprecationWarning added; DRY-10 `data_manager`/`unified_data_manager` DeprecationWarning added; DRY-11 all 6 standalone backtester files DeprecationWarning added — Phase B caller migration deferred to Sprint 6+ (high blast radius)
  - **Tier 3 (✅ DONE: DRY-15; DRY-14 PARTIAL — Sprint 3–4):** DRY-15 `load_csv_data` thin wrapper over `load_mt5_csv`; DRY-14 `PROJECT_ROOT` exported from `kinetra.config` — library files updated, `scripts/analysis/` one-offs deferred
  - **Tier 4 (✅ DONE: DRY-D1, DRY-D2 — Sprint 5):** DRY-D1 consolidation plan checklists updated; `archive/production_cleanup_2026-03-03/repo/docs/HOUSEKEEPING_AUDIT.md` superseded note confirmed; DRY-D2 `archive/production_cleanup_2026-03-03/repo/docs/ACTION_PLAN.md` replaced with archive pointer, original at `archive/production_cleanup_2026-03-03/repo/docs/ACTION_PLAN.md`; DRY-D3 copilot/master overlap (by design)
- `archive/production_cleanup_2026-03-03/repo/tests/test_integration.py` — **NEW**: 62 tests (DRY-09/10/11 DeprecationWarning content + backward compat; canonical classes confirmed clean; DRY-D1/D2 doc assertions; cross-cutting redefinition guard; `_reload_and_collect_warnings()` with sys.modules save/restore to prevent cross-test pollution)
- `archive/production_cleanup_2026-03-03/repo/tests/test_backtest_engine_comprehensive.py` — **NEW** (Sprint 3–4): 71 tests (omega_ratio, calculate_z_factor, MetricsCalculator regressions, thin wrapper contracts, PortfolioHealthScore rename, PROJECT_ROOT export)
- `AGENT_RULES_MASTER.md` — **§27 added**: DRY rules, canonical module table, open violations summary, new-code rule, previously-remediated cross-reference

**Key recent changes (2026-02-27) — Model Provenance Manifest + Reward Orchestrator Integration:**
- `kinetra/config.py` — **NEW**: Model provenance manifest system:
  - `TrainingManifest` dataclass: version, pipeline_run_id, git_commit, wavelet step, agent comparison status, training config, model file inventory, expiry
  - `write_manifest()` / `read_manifest()` / `validate_manifest()`: atomic JSON write, tolerant read (unknown fields ignored, missing fields default), 3-gate validation (age, agent_comparison, ≥50% file presence)
  - `discover_model_files()`: recursive scan of `models/`, `checkpoints/`, `data/runs/` for all model extensions
  - `build_manifest_from_status()`: convenience factory for pipeline scripts
  - CLI: `python -m kinetra.model_manifest` for operator diagnostics
- `kinetra_menu.py` — **SystemStatus + gate updates**:
  - `SystemStatus` gains `models_trained_current`, `manifest_age_days`, `manifest_run_id`, `manifest_rejection_reason` fields
  - `check_system_status()` reads and validates `results/training_manifest` — sets `models_trained_current` only when manifest is valid, not expired, agent comparison done, and ≥50% model files present
  - Main menu explore badge: prefers `models_trained_current` (manifest-validated) over legacy `models_trained AND agent_comparison_done`
  - Menu 3 pipeline banner `_ps6` (Trained): prefers `models_trained_current`
  - Menu 4 backtest gate `has_trained`: prefers `models_trained_current`
- `kinetra/__init__.py` — Lazy imports registered: `TrainingManifest`, `ManifestValidation`, `write_manifest`, `read_manifest`, `validate_manifest`, `build_manifest_from_status`, `discover_model_files`
- `kinetra/reward_shaping.py` — **RewardOrchestrator** fully integrated into `_WaveletTradingEnv` (already done in prior session):
  - `_step_orchestrated()`: routes raw P&L through orchestrator with MFE/MAE tracking, force-close handling, Omega terminal bonus
  - `_build_orchestrator()`: factory from `InstrumentRewardConfig` with horizon calibration
  - Per-episode CSV training logs: `results/wavelet/training_logs/step_N/<SYMBOL>_training_log.csv`
  - CLI flag: `--use-orchestrator` on `run_additive_step.py`
- `archive/production_cleanup_2026-03-03/repo/tests/test_integration.py` — **NEW**: 76 tests in 12 classes:
  - `TestTrainingManifestCreation` (8): defaults, auto-fields, unique run IDs
  - `TestWriteReadRoundTrip` (9): serialisation, corruption resilience, unknown field tolerance, auto-discovery
  - `TestValidateManifest` (11): agent comparison gate, expiry, custom max_age, partial file presence, unparseable timestamps
  - `TestDiscoverModelFiles` (8): multi-extension scanning, recursive, sorted output, ignores non-model files
  - `TestBuildManifestFromStatus` (4): convenience factory
  - `TestManifestValidation` (2): summary formatting
  - `TestEdgeCases` (7): empty JSON, arrays, empty paths, overwrites, naive datetimes
  - `TestCIGateSmoke` (6): 210 stale files blocked, manifest without comparison blocked, expired blocked, deleted files blocked, fresh replaces stale
  - `TestSystemStatusIntegration` (4): simulated check_system_status flow, has_trained gate logic
  - `TestProvenanceChain` (4): unique run IDs, git commit, training config preservation, wavelet step tracking
  - `TestPerformance` (2): write+read+validate < 100ms, discover 200 files < 500ms
  - `TestE2ETrainingManifestFlow` (10): **NEW** — end-to-end CI integration tests verifying training script → manifest → check_system_status → gate unlock flow; covers agent comparison manifest write, additive step with/without comparison, expired manifest (strict vs legacy fallback), retrain after expiry, model file deletion, latest-manifest-wins, pipeline banner logic, dry-run skip, full pipeline happy path
- `QUICK_START.md` — **NEW**: Operator remediation guide:
  - §1 Stale model artefacts: 3 remediation options (re-run pipeline, clean up, write manifest)
  - §2 Pipeline recovery: no-structure, audit failure, agent comparison failure, wavelet gate lock
  - §3 Common failures: training collapse, reckless trading, MetaAPI, import errors
  - §4 Gate debugging cheatsheet: table of all gates + required flags + unlock steps
  - §5 Manifest management: viewing, lifecycle diagram, manual inspection, expiry extension
  - §6 Data integrity: quality checks, common problems table, safe operations
  - §7 Reward orchestrator diagnostics: training log analysis, healthy ranges, config overrides
- `scripts/train.py` — **Manifest auto-write**: at end of `main()` (when `--commit`), writes training manifest via `build_manifest_from_status()` + `write_manifest()` with wavelet step, agent comparison status (detected from `results/exploration/`), and full training config (bands, episodes, seed, friction mode, orchestrator flag, aggregate/median Omega, trade count)
- `archive/production_cleanup_2026-03-03/scripts/training/explore_compare_agents.py` — **Manifest auto-write**: at end of `main()`, writes training manifest with `agent_comparison_done=True`, best agent name/omega from comparison results, wavelet step from disk, and training config (agents tested, episodes, baseline info, agents beating baseline)
- `archive/production_cleanup_2026-03-03/scripts/training/train_rl.py` — **Manifest auto-write**: writes manifest at end of `main()` with DQN agent type, backend info (PyTorch/NumPy), timeframe, symbol filter, instruments list, and physics state dim
- `archive/production_cleanup_2026-03-03/scripts/training/train_berserker.py` — **Manifest auto-write**: writes manifest at end of `main()` with strategy=berserker, episodes per timeframe, parallel instruments count, and run directory
- `archive/production_cleanup_2026-03-03/scripts/training/train_sniper.py` — **Manifest auto-write**: writes manifest at end of training via `_write_training_manifest()` helper with strategy=sniper, episodes, and run directory
- `archive/production_cleanup_2026-03-03/scripts/training/train_triad.py` — **Manifest auto-write**: writes manifest at end of `main()` with strategy=triad, role, per-agent performance metrics, file counts, and elapsed time
- `archive/production_cleanup_2026-03-03/scripts/training/train_fast_multi.py` — **Manifest auto-write**: writes manifest at end of `main()` with DQN agent type, episodes, instrument count, and instrument names
- `archive/production_cleanup_2026-03-03/scripts/training/train_with_metrics.py` — **Manifest auto-write**: writes manifest at end of `main()` with DQN agent type, episodes, data path, metrics port, and device
- `archive/production_cleanup_2026-03-03/scripts/training/train_rl_gpu.py` — **Manifest auto-write**: writes manifest at end of `main()` with DQN_GPU agent type, episodes, hidden sizes, learning rate, batch size, instruments, and device
- `archive/production_cleanup_2026-03-03/scripts/training/train_rl_physics.py` — **Manifest auto-write**: writes manifest at end of `main()` with DQN_physics agent type, instruments list, walk-forward fold count, and cross-instrument flag
- `kinetra/single_symbol_env.py` — **RewardOrchestrator integration**: optional `orchestrator` parameter added to `SingleSymbolRLEnv.__init__()`, trade close metadata tracking (MFE/MAE/bars held), `_route_through_orchestrator()` method mirroring `TradingEnv` and `RealisticTradingEnv` pattern, episode-end Omega bonus, full backward compatibility when `orchestrator=None`
- `kinetra/unified_trading_env.py` — **RewardOrchestrator integration**: optional `orchestrator` parameter added to `UnifiedTradingEnv.__init__()`, `_record_trade_close()` helper, `_route_through_orchestrator()` method, MFE/MAE tracking per position, episode-end Omega bonus, full backward compatibility when `orchestrator=None`
- `GitHub Actions workflow (`CI workflow YAML`)` — **NEW CI job `gate-invariants`**: runs menu gate tests, pipeline integration tests, model manifest tests, reward orchestrator tests, manifest CLI diagnostics, verifies all training scripts import manifest functions, and validates gate-logic invariants in `kinetra_menu.py`
- **Total tests:** 349 passed (86 menu gates + 89 integration + 98 reward orchestrator + 76 model manifest), ruff zero violations

**Manifest auto-write integration (all training scripts):**
- `run_additive_step.py` writes manifest on `--commit` (skipped on `--no-commit` / dry-run)
- `explore_compare_agents.py` writes manifest after agent comparison completes
- `train_rl.py`, `train_berserker.py`, `train_sniper.py`, `train_triad.py`, `train_fast_multi.py`, `train_with_metrics.py`, `train_rl_gpu.py`, `train_rl_physics.py` — all write manifests at end of `main()`
- All scripts detect agent comparison status from `results/exploration/`
- All scripts populate `training_config` with script name, parameters, and results
- All scripts use `_detect_agent_comparison()` helper (reads both `status=COMPLETE` and legacy `best_overall` format)
- Manifest includes auto-discovered model files via `discover_model_files()`
- `check_system_status()` reads manifest → `models_trained_current` unlocks gates

**RewardOrchestrator coverage (all RL environments):**
- `TradingEnv` (kinetra/trading_env.py) — ✅ orchestrator integrated
- `RealisticTradingEnv` (kinetra/realistic_trading_env.py) — ✅ orchestrator integrated
- `_WaveletTradingEnv` (scripts/train.py) — ✅ orchestrator integrated
- `SingleSymbolRLEnv` (kinetra/single_symbol_env.py) — ✅ orchestrator integrated (this session)
- `UnifiedTradingEnv` (kinetra/unified_trading_env.py) — ✅ orchestrator integrated (this session)

**Previous changes (2026-02-27) — Stale-Model / Wavelet-Gate / Pipeline-Order Bugfixes:**
- `kinetra_menu.py` — **3 bug fixes, 5 code locations:**
  - **BUG 1 (HIGH):** `models_trained` poisoned by 210 historical `.joblib` artifacts — all UI gates now require `agent_comparison_done AND models_trained`:
    - Main menu explore badge: shows `⚠️ (additive step 2/7 Ω=4.40)` or `⚠️ (stale models — re-run pipeline)` instead of false `✅`
    - Menu 3 pipeline banner `_ps6` (Trained): uses `agent_comparison_done and models_trained` — impossible `⚪Compare ✅Trained` state eliminated
    - Menu 4 backtest gates: `has_trained` requires both flags — stale models no longer unlock backtesting
  - **BUG 2 (MEDIUM):** Wavelet pipeline (Menu 3 option 4) locked despite features already computed — gate relaxed from `has_t1_data and has_prepared` to `has_t1_data and (has_prepared or has_wavelet)` since wavelet operates on raw H4 CSVs
  - **BUG 3 (DESIGN):** `suggest_next_step()` no longer suggests "Prepare Data" when wavelet features already exist; audit suggestion guarded on `data_prepared=True`
- `archive/production_cleanup_2026-03-03/repo/tests/test_menu_system.py` — **12 new regression tests** in 3 classes:
  - `TestStaleModelsDoNotPoisonBadges` (6 tests): stale models don't unlock ✅ badge, don't green pipeline banner, don't unlock backtest; legitimate trained path still works
  - `TestWaveletGateRelaxed` (4 tests): wavelet available with existing features, locked without both, still requires Tier-1 data
  - `TestPipelineOrderWaveletVsPrepare` (2 tests): suggest additive (not prepare) when wavelet done without prepare; don't regress to Phase 3 when additive is active

**Previous changes (2026-02-26) — Path & Menu Bugfixes:**
- `scripts/data_manager.py` — **SyntaxError fixed**: `global MIN_BARS_H4` moved to top of `main()` (Python 3.12 rejects use-before-global-declaration)
- `scripts/download/fetch_broker_spec_from_metaapi.py` — **Output path corrected**: header and save summary now show canonical `contract_spec.json` in instrument folders as primary, legacy `data/symbol_specs/` as secondary
- `kinetra_menu.py` — **Multiple fixes**:
  - `check_system_status()` spec detection: now reads `contract_spec.json` from instrument folders (canonical) first, falls back to legacy `data/symbol_specs/`
  - `curate_instruments()` menu wiring: removed invalid `--dry-run` flag (script defaults to dry-run when `--apply` absent)
  - `poll_symbol_specs()` display text: updated to show canonical instrument-folder path
  - `print_main_menu()`: Phase 2 sub-step "not run" hints now only shown when user is actually in Phase 2, not when past it into additive testing; shows current additive step progress when applicable
- `archive/production_cleanup_2026-03-03/repo/tests/test_integration.py` — NEW: 89 integration tests for full pipeline flows

**Key changes (2026-02-25) — Menu Restructure Implementation (COMPLETE):**
- `kinetra_menu.py` — **MAJOR RESTRUCTURE**: 6-phase pipeline with failure-aware gates
  - Menu 2 ("Data Foundation"): ACQUIRE(1-4) → VALIDATE(5-6) → CURATE(7) → SPECS(8-9) → ORGANISE(10-11) → PREPARE(12, Gate 2 required) → AUDIT(13) → RESEARCH(14)
  - Menu 3 ("Exploration & Training"): Phase 2 (CompExplore→SciDisc→Prime), Phase 4 (Wavelet→Additive→Ablation), Phase 5 (AgentCmp→Train→Specialists)
  - `suggest_next_step()` rewritten with full 6-phase gate-aware pipeline
  - Gate 2 (Structure Confirmed) now **blocks** Prepare Data until Scientific Discovery + Prime Discovery complete
  - `check_system_status()` reads curation marker, sci-disc quality, prepared-data audit, agent comparison results
  - Hardcoded Tier-1 universe removed — discovery required
  - Wavelet additive order loaded from file, not hardcoded
  - MC default: 50→100 runs; backtest years derived from data; denoise output path fixed
  - **B5**: Prepare Data now accepts `--instruments` / `--tier` filter (menu offers Tier-1 / Tier-1+2 / all / custom)
  - **D2**: Walk-forward and Performance Report marked as 🚧 PLANNED with detailed capability cards
- `scripts/download/parallel_data_prep.py` — `--instruments` and `--tier` CLI filters added; resolves symbols from discovery JSON
- `scripts/data_manager.py` — NEW: §22 instrument curation (dry-run / apply / report)
- `scripts/download/check_data_integrity.py` — NEW: prepared-data quality audit (NaN, schema, stationarity, regime)
- `archive/production_cleanup_2026-03-03/repo/tests/test_menu_system.py` — NEW: 74 unit tests for gate logic (pipeline order, Gate 2 blocking, status parsing, filter resolution)
- `archive/production_cleanup_2026-03-03/repo/tests/test_integration.py` — NEW: 89 integration tests for full pipeline flows (happy path Phase 1→6, failure paths, gate lock matrix, malformed marker resilience, recovery scenarios, dynamic Tier-1 loading)
- `archive/production_cleanup_2026-03-03/repo/docs/WORKFLOW.md` — All sign-off checklist items complete

**Canonical paths (after reorganisation):**
- Symbol specs: `data/master_standardized/<category>/<SYMBOL>/contract_spec.json` (canonical), `data/symbol_specs/<SYMBOL>.json` (legacy)
- Raw data: `data/master_standardized/<category>/<SYMBOL>/<SYMBOL>_<TF>_*.csv`
- Prepared data: `data/prepared_standardized/train/<category>/<SYMBOL>/<stem>.parquet`

**All restructure work COMPLETE (see MENU_RESTRUCTURE_PLAN.md):**
- ~~B5~~: ✅ DONE — `--instruments` / `--tier` filter in `parallel_data_prep.py` + menu tier selection UI
- ~~D2~~: ✅ DONE — Walk-forward and Performance Report marked as 🚧 PLANNED with detailed cards
- ~~Unit tests~~: ✅ DONE — 86 tests in `archive/production_cleanup_2026-03-03/repo/tests/test_menu_system.py` (pipeline order, Gate 2, status parsing, filter resolution, stale-model regression, wavelet gate, pipeline order design)
- ~~Integration tests~~: ✅ DONE — 89 tests in `archive/production_cleanup_2026-03-03/repo/tests/test_integration.py` (happy path Phase 1→6, 6 failure paths, gate lock matrix at every stage, malformed marker resilience, recovery scenarios, dynamic Tier-1 loading, partial preparation, curation dry-run vs apply, symbol specs freshness, backtest detection, retroactive unlock)
- ~~Path/menu bugfixes~~: ✅ DONE — curate SyntaxError, spec canonical path, menu context hints, dry-run flag
- ~~Stale-model/gate bugfixes~~: ✅ DONE — `models_trained` provenance guard, wavelet gate relaxed, pipeline order design fix

**Previous changes (2026-02-22–24):**
- `parallel_data_prep.py` — `project_root` path bug fixed; `add_derived_features()` removed
- `kinetra/denoise_filters.py` — magic-number fallback windows replaced with adaptive sizing
- `AGENT_RULES_MASTER.md` — §2.10 and §23 added; §24 tech-debt audit completed

**Key gate-logic invariants (enforced by tests):**
- `models_trained` alone NEVER unlocks UI badges or backtest gates — always requires `agent_comparison_done`
- `models_trained_current` (manifest-backed) is the preferred gate flag — `models_trained AND agent_comparison_done` is the legacy fallback
- Manifest validation requires ALL of: `agent_comparison_done=True`, age < `expires_days`, ≥50% listed model files present on disk
- Wavelet pipeline availability: `has_t1_data AND (has_prepared OR has_wavelet)` — not `has_prepared` alone
- `suggest_next_step()` skips "Prepare Data" when `wavelet_features_ready=True` (wavelet operates on raw H4 CSVs)
- Audit suggestion only fires when `data_prepared=True` (no audit if nothing to audit)
- **M1-specific detection**: `m1_data_available` field in `SystemStatus` — checks for `*_M1_*.csv` files specifically; `data_ready` alone could be satisfied by legacy H4 files
- **Renko pipeline is PARALLEL to Physics/Wavelet pipeline** — only needs `m1_data_available`, NOT Phase 5a completion
- **Renko training is sequentially gated**: sweep (10) → agent cmp (11) → risk L3 (12) → alloc L2 (13) — each requires its predecessor
- **Renko backtesting needs NO trained agent** — deterministic Layer 1; only requires `m1_data_available` (Menu 4 options 7-11)
- **Physics/Wavelet backtesting requires trained agent** — `models_trained_current` or `(models_trained AND agent_comparison_done)` (Menu 4 options 1-4)
- **Status line capped at 8 badges** — shows `+N more` when truncated to prevent terminal wrapping

**Canonical manifest path:**
- Training manifest: `results/training_manifest` — written by training scripts, read by `check_system_status()`

**Key changes (2026-03-03) — Menu Review & Fix:**
- `kinetra_menu.py` — **14 issues fixed**:
  - **Issue 1 (HIGH):** `suggest_next_step()` now treats Renko as a **parallel pipeline** — Renko suggestions appear after Phase 5a (or whenever `m1_data_available=True`), not blocked behind full Physics/Wavelet pipeline completion
  - **Issue 2 (HIGH):** Menu 4 now has **Renko backtesting options** (7-11): Instrument Backtest, Portfolio Backtest, Walk-Forward, Monte Carlo, Friction Stress Test — all deterministic Layer 1, no agent needed
  - **Issue 3 (MEDIUM):** Added `m1_data_available` field to `SystemStatus` — fast `os.scandir` check for `*_M1_*.csv` files; aggregation and Renko gates now use this instead of `data_ready`
  - **Issue 4 (MEDIUM):** Renko training options **sequentially gated**: option 10 (sweep) → 11 (cmp) → 12 (risk L3) → 13 (alloc L2); skipping steps no longer possible
  - **Issue 5 (MEDIUM):** Physics/Wavelet backtesting (Menu 4 options 1-4) and Renko backtesting (Menu 4 options 7-11) have **separate gates** — Renko needs only M1 data
  - **Issue 6 (MEDIUM):** Tier-1 data check message updated for M1-only architecture — says "download M1 + aggregate" not "download H4"
  - **Issue 7 (MEDIUM):** Circuit Breaker Status added to **Menu 5 option 3** — view config, persisted state, and reset cooldowns
  - **Issue 8 (LOW):** Options 15 (aggregate) and 16 (archive) moved into `_data_menu_locks()` for single-source-of-truth
  - **Issue 9 (LOW):** Option 10 gap in Menu 3 filled — Renko options renumbered 10-13 (was 11-14), View Results is now option 14 (was 15)
  - **Issue 10 (LOW):** Status line capped at 8 badges + `+N more` to prevent terminal wrapping
  - **Issue 11 (LOW):** Menu 5 (System Tools) renumbered 1-8 to accommodate circuit breaker option
  - `_data_menu_locks()` docstring updated to reflect aggregate (15) and archive (16) steps
  - Renko pipeline progress banner uses `m1_data_available` instead of `data_ready`
  - `suggest_next_step()` docstring updated with Phase 5b parallel designation and option numbers
- `SystemStatus` — **3 new fields**: `m1_data_available` (bool), `renko_backtest_done` (bool), `renko_backtest_age_days` (Optional[float]), `renko_backtest_portfolio_omega` (Optional[float])
- `check_system_status()` — **2 new detection blocks**: M1 file scan (3-level `os.scandir`), Renko backtest results (`results/renko/backtest/*.json`)
- **5 new menu functions**: `renko_instrument_backtest()`, `renko_portfolio_backtest()`, `renko_walk_forward()`, `renko_monte_carlo()`, `renko_friction_stress_test()`, `circuit_breaker_status()`
- `archive/production_cleanup_2026-03-03/repo/tests/test_menu_system.py` — **22 new tests** (108 total):
  - `TestDataMenuLocks`: 3 new tests (aggregate M1 gate, aggregate locked without M1, archive in locks)
  - `TestGetStatusLine`: 1 new test (badge cap at max)
  - `TestM1DataAvailable` (NEW class, 3 tests): default false, requires M1 for Renko, appears with M1
  - `TestRenkoParallelPipeline` (NEW class, 2 tests): suggested after Phase 1 with M1, not suggested without M1
  - `TestRenkoSequentialGating` (NEW class, 5 tests): only sweep initially, cmp after sweep, risk after cmp, alloc after risk, alloc locked without risk
  - `TestRenkoBacktestingNoAgent` (NEW class, 3 tests): available without model, physics locked without agent, status fields default
  - Existing fixtures updated: `m1_data_available=True` added to Phase 1+ fixtures
- `archive/production_cleanup_2026-03-03/repo/tests/test_integration.py` — **4 tests updated, 1 new test** (93 total):
  - `_ProjectBuilder.add_m1_data()` method added — creates `*_M1_*.csv` stub files
  - Renko pipeline tests now call `.add_m1_data()` so `m1_data_available=True`
  - Option number references updated (11→10, 12→11)
  - `test_phase5b_renko_not_suggested_without_m1` — NEW: verifies fallthrough to validation when only H4 data exists
  - `test_phase5_agent` — tolerates badge cap truncation

**For complete, comprehensive rules → See [`AGENT_RULES_MASTER.md`](../README.md)**

---

## Renko Kinetra Architecture — Quick Reference

> **Strategy Discovery Status: LOCKED (§29.1)**
> The Renko trading core is fully validated. Do not re-open signal research.
> Entry: colour flip. Gate: fixed Markov persistence. Exit: FlipExit. Stop: 1 brick (backtest) / 0.5 brick (live).
> Brick size is a structural DSP measurement, not a tuning parameter.
> Chop is a risk problem (loss-cluster breaker), not a signal problem.
> Rejected and must not return: TMA, PSAR, entropy/DFA filters, higher-order Markov.

> **Design Spec:** [`archive/production_cleanup_2026-03-03/repo/docs/TRADING_STRATEGY_SPEC.md`](../archive/production_cleanup_2026-03-03/repo/docs/TRADING_STRATEGY_SPEC.md)
> **Empirical basis:** Portfolio Omega 6.07, Z 28.89, 79% OOS survival, 100% friction stress survival (19 instruments, 8 clusters)

### What Changed (Current Kinetra → Renko Kinetra)

| Dimension | Current Kinetra | Renko Kinetra |
|-----------|----------------|---------------|
| Data source | M1 + M30 + H4 downloaded separately | **M1 only**, aggregate up |
| Features | PhysicsEngine (velocity, energy, entropy) | **VR + FlipRate + Markov** on brick sequence |
| Pipeline | 12-step prepare data | **4-step**: download → validate → aggregate → compute |
| Strategy | Multiple types, RL-learned signals | **One**: Renko flip + regime filter |
| RL task | Entry/exit signals | **Allocation weights + risk management** |
| Primary TF | H4 | **M1** (source), **M30** (DSP), **bricks** (trading) |
| Stop | Fixed or adaptive | **1 brick** (backtest) / **0.5 brick** (live) |
| Exit | Various | **Colour change** (first opposite brick) |

### Data QC — Broker Fingerprint (§29.2, required before brick construction)

Every M1 file must be fingerprinted before DSP / brick construction:
1. Coverage ratio (rows / expected minutes)
2. Dominant missing minute-of-day range → **session break UTC window**
3. Top-10 gap sizes + counts
4. Spike count (ratio > 20× rolling median AND abs move > threshold)
5. OHLC integrity

**Session break rule (§29.3):** `build_renko()` requires `session_break_minutes` from `SessionProfile`. Gaps ≥ 30 min reset the reference price — no bricks emitted across the gap. This prevents post-rollover brick bursts (e.g. XAUUSD: 21:00–21:59 UTC daily gap on MetaAPI).

### Build-Time Pipeline (7 Phases)

```
PHASE 1 ─ DATA FOUNDATION           ✅ COMPLETE
  Download M1 → Validate → Aggregate → Specs

PHASE 2 ─ DSP INSTRUMENT FILTERING  ✅ COMPLETE
  VR → Spread floor → Brick sizing → Quality gate → Candidates

PHASE 3 ─ TRADING LOGIC             ✅ COMPLETE (Sprint 2)
  Filter comparison → Entry variants → Stop model → Scaling rules

PHASE 4 ─ PORTFOLIO CONSTRUCTION    ✅ COMPLETE (Sprint 2)
  Clusters → Equal-risk sizing → Walk-forward → Stress test

PHASE 5 ─ RL INTEGRATION            ✅ COMPLETE (Sprint 3)
  Env design → Reward sweep → Agent comparison → Train → Validate

PHASE 6 ─ BACKTESTING               🔲 Sprint 5B/5C
  Qualification pipeline → Portfolio construction → MC + Regime + Tail risk

PHASE 7 ─ PROGRESSIVE LIVE          🔲 Sprint 6+
  Paper → Micro (0.01) → Small (0.1) → Full lots
  + Online recalibration: CalibrationDriftDetector → InstrumentContext.recalibrate()
```

### Three-Layer Runtime Architecture

```
┌──────────────────────── HARD CIRCUIT BREAKERS ────────────────────────┐
│  VPIN > extreme → flatten       DD > limit → reduce 50%              │
│  Spread > 3× normal → halt      Correlation → 1 → cap                │
│  ────────────────── NON-NEGOTIABLE, NOT LEARNED ─────────────────────│
└───────────────────────────┬──────────────────────────────────────────┘
                            ▼
┌──────────────────── LAYER 3 — RISK RL AGENT ─────────────────────────┐
│  Output: portfolio_exposure_scalar ∈ [0.0, 1.0]                       │
│  Question: "Should we be in the market at all right now?"             │
└───────────────────────────┬──────────────────────────────────────────┘
                            ▼
┌──────────────────── LAYER 2 — ALLOCATION RL AGENT ───────────────────┐
│  Output: weight[i] ∈ [0.0, 1.0] per instrument                       │
│  Question: "How much capital in each instrument right now?"           │
└───────────────────────────┬──────────────────────────────────────────┘
                            ▼
┌──────────────────── LAYER 1 — DETERMINISTIC ENGINE ──────────────────┐
│  M1 feed → brick construction → filter evaluation                     │
│  Flip + filter pass → TRADE at allocated weight                       │
│  Stop: 0.5 brick (live) / 1 brick (backtest)                         │
│  Exit: colour change (first opposite brick)                           │
│  ──────────────── NO LEARNING, PURE RULES ───────────────────────────│
└─────────────────────────────────────────────────────────────────────┘
```

### Regime Change & RL Adaptation — Horizon Map (§29.6)

| Horizon | Mechanism | Status |
|---|---|---|
| Minutes | Hard circuit breakers (VPIN > extreme, DD > limit) | ✅ Done |
| Hours–Days | Layer 3 RL exposure scalar (VPIN + spread + corr obs) | ✅ Done |
| Days–Weeks | Loss-cluster breaker `RiskParams` (participation throttle) | 🔲 Sprint 5A |
| Weeks–Months | `CalibrationDriftDetector` → re-run DSP + sweep | 🔲 Sprint 5C |
| Months–Years | Layer 2 RL with live `vr_current` via `InstrumentContext.recalibrate()` | 🔲 Sprint 6 |
| Years | Instrument retirement `QualificationRegistry.disqualify()` | 🔲 Sprint 6 |

**Key rule:** RL adapts allocation/exposure WITHIN a calibrated parameter set. RL never calibrates brick size or filter thresholds — that is `run_dsp()` + `scaled_filter_params()`.

### Canonical Renko Modules — import, don't reinvent

| What you need | Import from |
|---------------|-------------|
| **RL — Reward** | |
| Allocation reward (combined) | `kinetra.rl.reward.compute_allocation_reward` ✅ **NEW** |
| Trade-level reward | `kinetra.rl.reward.compute_trade_reward` ✅ **NEW** |
| Portfolio-level reward | `kinetra.rl.reward.compute_portfolio_reward` ✅ **NEW** |
| Terminal reward | `kinetra.rl.reward.compute_terminal_reward` ✅ **NEW** |
| Risk reward (Layer 3) | `kinetra.rl.reward.compute_risk_reward` ✅ **NEW** |
| Allocation reward config | `kinetra.rl.reward.AllocationRewardConfig` ✅ **NEW** |
| Risk reward config | `kinetra.rl.reward.RiskRewardConfig` ✅ **NEW** |
| Trade outcome record | `kinetra.rl.reward.TradeOutcome` ✅ **NEW** |
| Portfolio state snapshot | `kinetra.rl.reward.PortfolioState` ✅ **NEW** |
| Episode terminal state | `kinetra.rl.reward.EpisodeTerminalState` ✅ **NEW** |
| Reward tracker (diagnostics) | `kinetra.rl.reward.RewardTracker` ✅ **NEW** |
| Risk reward tracker | `kinetra.rl.reward.RiskRewardTracker` ✅ **NEW** |
| **RL — Environments** | |
| Portfolio allocation env | `kinetra.rl.portfolio_env.RenkoPortfolioEnv` ✅ **NEW** |
| Instrument context (env input) | `kinetra.rl.portfolio_env.InstrumentContext` ✅ **NEW** |
| Portfolio env config | `kinetra.rl.portfolio_env.PortfolioEnvConfig` ✅ **NEW** |
| Risk overlay env | `kinetra.rl.risk_env.RiskOverlayEnv` ✅ **NEW** |
| Portfolio day snapshot | `kinetra.rl.risk_env.PortfolioDaySnapshot` ✅ **NEW** |
| Risk env config | `kinetra.rl.risk_env.RiskEnvConfig` ✅ **NEW** |
| Snapshots from trades factory | `kinetra.rl.risk_env.RiskOverlayEnv.snapshots_from_trades` ✅ **NEW** |
| **Renko — Core** | |
| Build Renko bricks | `kinetra.renko.brick_engine.build_renko` ✅ |
| Brick summary stats | `kinetra.renko.brick_engine.brick_summary` ✅ |
| Bricks per day | `kinetra.renko.brick_engine.bricks_per_day` ✅ |
| FlipRate filter | `kinetra.renko.filters.flip_rate` ✅ |
| Markov stickiness | `kinetra.renko.filters.markov_stickiness` ✅ |
| KER (optional tertiary) | `kinetra.renko.filters.ker` ✅ |
| Entry evaluation (scalar) | `kinetra.renko.filters.evaluate_entry` ✅ |
| Entry evaluation (vectorized) | `kinetra.renko.filters.evaluate_entries_vectorized` ✅ |
| VR profile | `kinetra.renko.dsp.vr_profile` ✅ |
| Brick sizing from VR scale | `kinetra.renko.dsp.brick_from_scale` ✅ |
| Regime classification | `kinetra.renko.dsp.classify_regime` ✅ |
| Friction floor | `kinetra.renko.dsp.compute_friction_floor` ✅ |
| Full DSP analysis | `kinetra.renko.dsp.run_dsp` ✅ |
| DSP brick sweep | `kinetra.renko.dsp.sweep_brick_sizes` (DSP-level) ✅ |
| **VPIN (scalar)** | `kinetra.renko.vpin.compute_vpin` ✅ **NEW (Sprint 4)** |
| **VPIN time series** | `kinetra.renko.vpin.vpin_timeseries` ✅ **NEW (Sprint 4)** |
| **VPIN baseline stats** | `kinetra.renko.vpin.vpin_baseline` ✅ **NEW (Sprint 4)** |
| **VPIN multi-instrument** | `kinetra.renko.vpin.compute_vpin_multi` ✅ **NEW (Sprint 4)** |
| **VPIN normalisation** | `kinetra.renko.vpin.normalise_vpin` ✅ **NEW (Sprint 4)** |
| **VPIN z-score normalisation** | `kinetra.renko.vpin.normalise_vpin_zscore` ✅ **NEW (Sprint 4)** |
| **VPIN regime classification** | `kinetra.renko.vpin.classify_vpin_regime` ✅ **NEW (Sprint 4)** |
| **VPIN extreme detection** | `kinetra.renko.vpin.is_vpin_extreme` ✅ **NEW (Sprint 4)** |
| **VPIN auto bucket sizing** | `kinetra.renko.vpin.auto_bucket_size` ✅ **NEW (Sprint 4)** |
| **VPIN baseline dataclass** | `kinetra.renko.vpin.VPINBaseline` ✅ **NEW (Sprint 4)** |
| **VPIN bucket dataclass** | `kinetra.renko.vpin.VPINBucket` ✅ **NEW (Sprint 4)** |
| **VPIN time series dataclass** | `kinetra.renko.vpin.VPINTimeSeries` ✅ **NEW (Sprint 4)** |
| **Circuit breaker manager** | `kinetra.monitoring.circuit_breakers.CircuitBreakerManager` ✅ **NEW (Sprint 4)** |
| **Circuit breaker evaluation** | `kinetra.monitoring.circuit_breakers.evaluate_circuit_breakers` ✅ **NEW (Sprint 4)** |
| **VPIN breaker** | `kinetra.monitoring.circuit_breakers.check_vpin_breaker` ✅ **NEW (Sprint 4)** |
| **Drawdown breaker** | `kinetra.monitoring.circuit_breakers.check_drawdown_breaker` ✅ **NEW (Sprint 4)** |
| **Spread breaker** | `kinetra.monitoring.circuit_breakers.check_spread_breaker` ✅ **NEW (Sprint 4)** |
| **Correlation breaker** | `kinetra.monitoring.circuit_breakers.check_correlation_breaker` ✅ **NEW (Sprint 4)** |
| **Breaker action enum** | `kinetra.monitoring.circuit_breakers.BreakerAction` ✅ **NEW (Sprint 4)** |
| **Breaker config** | `kinetra.monitoring.circuit_breakers.CircuitBreakerConfig` ✅ **NEW (Sprint 4)** |
| **Breaker state** | `kinetra.monitoring.circuit_breakers.CircuitBreakerState` ✅ **NEW (Sprint 4)** |
| **Breaker portfolio snapshot** | `kinetra.monitoring.circuit_breakers.PortfolioSnapshot` ✅ **NEW (Sprint 4)** |
| **Instrument backtest** | `kinetra.renko.backtest.backtest_instrument` ✅ **NEW** |
| **Portfolio backtest** | `kinetra.renko.backtest.backtest_portfolio` ✅ **NEW** |
| **Walk-forward IS/OOS** | `kinetra.renko.backtest.walk_forward_instrument` ✅ **NEW** |
| **Monte Carlo validation** | `kinetra.renko.backtest.monte_carlo_instrument` ✅ **NEW** |
| **Brick sweep (full backtest)** | `kinetra.renko.backtest.sweep_brick_sizes` ✅ **NEW** |
| **Friction stress test** | `kinetra.renko.backtest.stress_test_friction` ✅ **NEW** |
| **Filter params config** | `kinetra.renko.backtest.FilterParams` ✅ **NEW** |
| **Stop/exit config** | `kinetra.renko.backtest.StopParams` ✅ **NEW** |
| **Renko trade record** | `kinetra.renko.backtest.RenkoTrade` ✅ **NEW** |
| **Cluster assignment** | `kinetra.renko.portfolio.get_cluster` ✅ **NEW** |
| **Cluster taxonomy** | `kinetra.renko.portfolio.CLUSTER_MAP` ✅ **NEW** |
| **Equal-risk sizing** | `kinetra.renko.portfolio.equal_risk_weights` ✅ **NEW** |
| **Cluster capping** | `kinetra.renko.portfolio.apply_cluster_caps` ✅ **NEW** |
| **Spot/futures dedup** | `kinetra.renko.portfolio.deduplicate_underlyings` ✅ **NEW** |
| **Portfolio equity curve** | `kinetra.renko.portfolio.build_portfolio_equity` ✅ **NEW** |
| **Full portfolio builder** | `kinetra.renko.portfolio.build_portfolio` ✅ **NEW** |
| **Herfindahl index** | `kinetra.renko.portfolio.herfindahl_index` ✅ **NEW** |
| **Max drawdown** | `kinetra.renko.portfolio.max_drawdown` ✅ **NEW** |
| **Calmar ratio** | `kinetra.renko.portfolio.calmar_ratio` ✅ **NEW** |
| **CVaR / Expected Shortfall** | `kinetra.renko.portfolio.cvar` ✅ **NEW** |
| **Tail-risk analysis** | `kinetra.renko.portfolio.tail_risk_analysis` ✅ **NEW** |
| **USD/point estimation** | `kinetra.renko.portfolio.estimate_usd_per_point` ✅ **NEW** |
| **Worst-period finder** | `kinetra.renko.portfolio.find_worst_period` ✅ **NEW** |
| **Correlation stress** | `kinetra.renko.portfolio.stress_correlation_one` ✅ **NEW** |
| M1 → higher TF aggregation | `kinetra.aggregation.aggregate_ohlcv` ✅ |
| Batch aggregation | `kinetra.aggregation.aggregate_batch` ✅ |
| M1 download chunking | `scripts.download.download_core.backward_chunk_download` ✅ |

### Renko Hard Rules

- ❌ **Never** duplicate `build_renko()` — import from `kinetra.renko.brick_engine`
- ❌ **Never** duplicate `flip_rate()` or `markov_stickiness()` — import from `kinetra.renko.filters`
- ❌ **Never** duplicate cluster taxonomy — import `CLUSTER_MAP` or `get_cluster()` from `kinetra.renko.portfolio`
- ❌ **Never** duplicate omega/z-factor computation — import from `kinetra.backtesting.metrics` (used inside `kinetra.renko.backtest`)
- ❌ **Never** duplicate equity curve merging — use `kinetra.renko.portfolio.build_portfolio_equity`
- ❌ **Never** duplicate VPIN computation — import `compute_vpin` / `vpin_timeseries` / `vpin_baseline` from `kinetra.renko.vpin`
- ❌ **Never** inline VPIN extreme detection — use `kinetra.renko.vpin.is_vpin_extreme` with a `VPINBaseline`
- ❌ **Never** duplicate circuit breaker logic — import from `kinetra.monitoring.circuit_breakers`
- ❌ **Never** hardcode drawdown/VPIN/spread/correlation safety thresholds in scripts — use `CircuitBreakerConfig` from `kinetra.monitoring.circuit_breakers`
- ❌ **Never** implement non-learned safety limits outside `kinetra.monitoring.circuit_breakers` — all hard limits are centralised there
- ❌ **Never** use intrabar (high/low) data for Renko construction — close-only prevents lookahead bias
- ❌ **Never** use fixed-period TA (MA_20, RSI_14, etc.) in the Renko pipeline — all features are VR/FlipRate/Markov on brick sequences
- ❌ **Never** use denoised data as Renko input — Renko is inherently noise-filtered
- ❌ **Never** aggregate from anything other than M1 — all higher TFs are derived via `kinetra.aggregation`
- ❌ **Never** hardcode filter thresholds across instruments — use `FilterParams` and empirical/DSP scaling
- ❌ **Never** duplicate Renko reward functions — import from `kinetra.rl.reward`
- ❌ **Never** build a new Renko RL env — use `RenkoPortfolioEnv` (Layer 2) or `RiskOverlayEnv` (Layer 3)
- ❌ **Never** duplicate `InstrumentContext` — import from `kinetra.rl.portfolio_env`
- ❌ **Never** duplicate `PortfolioDaySnapshot` — import from `kinetra.rl.risk_env`
- ✅ **Always** gate bricks on friction floor: `brick_size ≥ max(Y × spread_p50, Z × spread_p95)`
- ✅ **Always** gate instruments on friction ratio: `spread / brick ≤ 0.25`
- ✅ **Always** run walk-forward (70/30 IS/OOS) before trusting backtest results
- ✅ **Always** run friction stress (1.5× and 2.0× costs) before including an instrument
- ✅ **Always** apply cluster caps (max 3 per cluster, max 35% weight) to prevent concentration
- ✅ **Always** use equal-risk sizing: `1R = 0.5 × brick_size × usd_per_point`
- ✅ **Always** use `kinetra.renko.backtest.backtest_instrument()` for single-instrument Renko backtests
- ✅ **Always** use `kinetra.renko.backtest.backtest_portfolio()` for portfolio-level backtests
- ✅ **Always** use `kinetra.rl.portfolio_env.RenkoPortfolioEnv` for allocation RL training
- ✅ **Always** use `kinetra.rl.risk_env.RiskOverlayEnv` for risk overlay RL training
- ✅ **Always** use `kinetra.rl.reward.compute_allocation_reward()` for Layer 2 rewards
- ✅ **Always** use `kinetra.rl.reward.compute_risk_reward()` for Layer 3 rewards
- ✅ **Always** run `CircuitBreakerManager.evaluate()` before executing any Layer 2/3 agent decisions in live mode
- ✅ **Always** compute VPIN baselines per-instrument from historical data before using VPIN in circuit breakers or risk env
- ✅ **Always** use `kinetra.renko.vpin.compute_vpin_multi()` for portfolio-level VPIN (feeds `vpin_mean` and `vpin_max` observations)

### Data Pipeline (Renko Kinetra)

```
DOWNLOAD M1 (scripts/download/download_market_data.py)
  │  MetaAPI/cTrader → backward_chunk_download() → exhaust broker history
  ▼
BROKER FINGERPRINT (kinetra/renko/session.py — 🔲 Sprint 5B)
  │  detect_session_break() → SessionProfile (gap UTC, duration, weekend bars)
  │  QC metrics: coverage, gaps, spikes, OHLC integrity
  ▼
VALIDATE (scripts/download/check_and_fill_data.py)
  │  Gap scan, blacklist, integrity checks
  ▼
AGGREGATE (kinetra/aggregation.py + scripts/download/parallel_data_prep.py)
  │  M1 → M5/M15/M30/H1/H4 — cache, not source — always re-derivable
  ▼
DSP ANALYSIS (kinetra/renko/dsp.py)
  │  VR profile → peak scale → brick size → scaled_filter_params() → regime
  │  scaled_filter_params(): derives FilterParams from DSP output (🔲 Sprint 5A)
  ▼
QUALIFY INSTRUMENT (kinetra/renko/qualify.py — 🔲 Sprint 5B)
  │  qualify_instrument() → QualificationResult → qualification.json
  │  Chains: session detect → DSP → scaled params → sweep → walk-forward → stress
  │  Idempotent: skip if qualification.json exists and data_end matches
  ▼
BRICK SWEEP (kinetra/renko/backtest.py)
  │  sweep_brick_sizes() → quality gates → select best Omega
  │  build_renko() called with session_break_minutes from SessionProfile (🔲 Sprint 5A)
  ▼
BACKTEST (kinetra/renko/backtest.py)
  │  backtest_instrument(risk_params=RiskParams()) — loss-cluster breaker (🔲 Sprint 5A)
  │  walk_forward_instrument() → stress_test_friction()
  ▼
PORTFOLIO (kinetra/renko/portfolio.py + kinetra/renko/orchestrator.py — 🔲 Sprint 5C)
  │  QualificationRegistry.get_qualified() → build_portfolio()
  │  cluster caps → equal-risk sizing → equity curve
  ▼
PORTFOLIO BACKTEST (kinetra/renko/backtest.py + orchestrator.py)
  │  backtest_portfolio() → monte_carlo_instrument() → tail_risk_analysis()
  │  → writes outputs/renko_results/portfolio_result.json
  ▼
REWARD SWEEP (Menu 3 › 10 — ✅ COMPLETE)
  │  reward_sweep.py → grid/random search → Pareto-optimal reward weights
  ▼
AGENT COMPARISON (Menu 3 › 11 — ✅ COMPLETE)
  │  explore_compare_renko_agents.py → L2+L3 baseline vs RL agents
  ▼
RL TRAINING (Menu 3 › 12-13 — ✅ COMPLETE)
  │  train_risk_agent.py (Layer 3 FIRST, option 12)
  │  → train_allocation_agent.py (Layer 2, option 13)
  ▼
RENKO BACKTESTING (Menu 4 › 9-13 — deterministic, no agent needed)
  │  Instrument backtest (9) → Portfolio backtest (10) → Walk-forward (11)
  │  → Monte Carlo (12) → Friction stress test (13)
  ▼
LIVE DEPLOYMENT (Sprint 6+ — planned)
     Paper → PER Gate 1 (micro) → Gate 2 (small) → Gate 3 (full)
     + Monthly: CalibrationDriftDetector.check() → recalibrate() if drift confirmed
```

### Directory Structure (Renko)

```
kinetra/renko/
├── __init__.py          # Package exports (all symbols from all submodules)
├── brick_engine.py      # build_renko(session_break_minutes=30.0), brick_summary(),
│                        #   bricks_per_day() — 🔲 Sprint 5A: add session_break_minutes param
├── filters.py           # flip_rate(), markov_stickiness(), ker(), evaluate_entry()
├── dsp.py               # vr_profile(), brick_from_scale(), run_dsp(), sweep_brick_sizes()
│                        #   🔲 Sprint 5A: + scaled_filter_params(dsp_result, bricks_per_day)
├── backtest.py          # ✅ backtest_instrument(risk_params=RiskParams()), backtest_portfolio(),
│                        #   walk_forward, monte_carlo, sweep_brick_sizes, stress_test_friction
│                        #   🔲 Sprint 5A: + RiskParams (loss-cluster breaker, DD throttle)
├── portfolio.py         # ✅ CLUSTER_MAP, get_cluster(), equal_risk_weights(),
│                        #   apply_cluster_caps(), build_portfolio(), build_portfolio_equity(),
│                        #   herfindahl_index(), calmar_ratio(), cvar(), tail_risk_analysis()
├── vpin.py              # ✅ Sprint 4: compute_vpin(), vpin_timeseries(), vpin_baseline(),
│                        #   compute_vpin_multi(), normalise_vpin(), classify_vpin_regime(),
│                        #   is_vpin_extreme(), auto_bucket_size(), VPINBaseline, VPINBucket
├── qualify.py           # 🔲 Sprint 5B: qualify_instrument(), QualificationResult,
│                        #   QualificationRegistry, CalibrationDriftDetector
├── session.py           # 🔲 Sprint 5B: detect_session_break(), SessionProfile
│                        #   (broker fingerprinting — gap UTC, duration, weekend bars)
├── orchestrator.py      # 🔲 Sprint 5C: run_full_pipeline(), PortfolioPipelineResult
kinetra/rl/                     # ✅ Sprint 3 — RL environments + reward
├── __init__.py          # Package exports (all symbols from all submodules)
├── reward.py            # ✅ compute_allocation_reward(), compute_risk_reward(),
│                        #   AllocationRewardConfig, RiskRewardConfig, TradeOutcome,
│                        #   PortfolioState, EpisodeTerminalState, RewardTracker
├── portfolio_env.py     # ✅ RenkoPortfolioEnv (Layer 2 — Allocation),
│                        #   InstrumentContext (✅ Sprint 6: recalibrate() method done),
│                        #   PortfolioEnvConfig, Gymnasium API
├── risk_env.py          # ✅ RiskOverlayEnv (Layer 3 — Risk),
│                        #   PortfolioDaySnapshot (🔲 Sprint 5C: + vr_drift, recalibration_pending),
│                        #   RiskEnvConfig, circuit breakers
kinetra/monitoring/             # ✅ Sprint 4 — Non-negotiable safety infrastructure
├── __init__.py          # Package exports
├── circuit_breakers.py  # ✅ Sprint 4: CircuitBreakerManager, evaluate_circuit_breakers(),
│                        #   check_vpin_breaker(), check_drawdown_breaker(),
│                        #   check_spread_breaker(), check_correlation_breaker(),
│                        #   BreakerAction, BreakerType, BreakerResult, EvaluationResult,
│                        #   CircuitBreakerConfig, CircuitBreakerState, PortfolioSnapshot
kinetra/aggregation.py   # M1 → higher TF aggregation (aggregate_ohlcv, aggregate_batch)
scripts/download/
├── download_core.py     # ✅ Canonical shared download module (backward_chunk_download)
├── download_metaapi.py  # Simple CLI wrapper
├── metaapi_bulk_download.py  # Parallel bulk wrapper
├── download_interactive.py   # Interactive menu wrapper
scripts/data/
├── aggregate_timeframes.py   # CLI: M1 → higher TFs + optional Renko generation
scripts/renko/
├── qualify_instruments.py    # 🔲 Sprint 5B: CLI — parallel instrument qualification
│                              #   --instrument / --all / --category; writes qualification.json
├── renko_backtest.py          # 🔲 Sprint 5C: CLI — Renko backtesting (instrument, portfolio,
│                              #   walk-forward, monte-carlo, stress-friction — Menu 4 options 9-13)
scripts/training/              # ✅ Sprint 4 — Renko RL training infrastructure
├── train_allocation_agent.py  # ✅ Layer 2 training (Menu 3 › 13, RenkoPortfolioEnv)
├── train_risk_agent.py        # ✅ Layer 3 training (Menu 3 › 12, RiskOverlayEnv)
├── explore_compare_renko_agents.py  # ✅ Agent comparison (Menu 3 › 11, 5 agent types)
├── reward_sweep.py            # ✅ Reward weight sweep (Menu 3 › 10, grid/random search)
tests/
├── test_renko.py        # 147 tests (brick_engine, filters, dsp, pipeline)
├── test_renko_backtest.py  # 131 tests (backtest, portfolio, walk-forward,
│                           #   MC, sweep, stress, cluster, sizing, equity, tail-risk)
├── test_renko_rl.py     # 128 tests (reward functions, portfolio_env,
│                        #   risk_env, trackers, numerical stability, full pipeline)
├── test_renko_vpin.py   # ✅ Sprint 4: 117 tests (BVC, bucketing, timeseries, baseline,
│                        #   normalisation, regime, extreme, auto-sizing, multi-instrument,
│                        #   column resolution, data containers, numerical stability, integration)
├── test_circuit_breakers.py  # ✅ Sprint 4: 165 tests (VPIN/DD/spread/correlation breakers,
│                              #   aggregate evaluator, manager, cooldowns, transitions,
│                              #   weight application, state, config validation, edge cases)
├── test_aggregation.py  # Aggregation tests
├── test_renko_qualify.py     # 🔲 Sprint 5B: qualification pipeline, session detect, registry
├── test_renko_orchestrator.py # 🔲 Sprint 5C: full pipeline, drift detection, menu wiring
├── test_menu_gates.py   # ✅ 108 tests (pipeline order, gates, M1 detection, Renko parallel,
│                        #   sequential gating, backtest gates, status line cap, badge logic)
├── test_pipeline_integration.py  # ✅ 93 tests (happy path, failure paths, gate lock matrix,
│                                 #   M1 detection, Renko pipeline order, status line)
```

### Sprint Progress

| Sprint | Focus | Status |
|--------|-------|--------|
| **Sprint 1** | Foundation: brick_engine, filters, dsp, aggregation, download_core | ✅ COMPLETE |
| **Sprint 2** | Portfolio + Backtesting: backtest.py, portfolio.py | ✅ COMPLETE |
| **Sprint 3** | RL Environments + Reward: reward.py, portfolio_env.py, risk_env.py | ✅ COMPLETE |
| **Sprint 4** | VPIN + Circuit Breakers + Training: vpin.py, circuit_breakers.py, training scripts | ✅ COMPLETE |
| **Sprint 5A** | Fix foundation: session break in build_renko(), RiskParams loss-cluster breaker, scaled_filter_params() | ✅ COMPLETE |
| **Sprint 5B** | Qualification pipeline: qualify.py, session.py, QualificationRegistry, CalibrationDriftDetector, qualify_instruments.py CLI | ✅ COMPLETE |
| **Sprint 5C** | Portfolio orchestrator: orchestrator.py, PortfolioDaySnapshot drift fields (vr_drift/recalibration_pending), N_RISK_OBS_FEATURES 8→10, menu wiring, comprehensive tests | ✅ COMPLETE |
| **Sprint 6** | Online learning: InstrumentContext.recalibrate(), CalibrationDriftDetector live, paper trading, PER gates | 🔲 CURRENT |
| **Sprint 7** | cTrader live connector, multi-broker deployment | 🔲 PLANNED |

### Test Counts (Renko + Menu)

| Module | Tests | Status |
|--------|-------|--------|
| `test_renko.py` (brick_engine + filters + dsp) | 147 | ✅ |
| `test_renko_backtest.py` (backtest + portfolio) | 131 | ✅ |
| `test_renko_rl.py` (reward + portfolio_env + risk_env) | 131 | ✅ Sprint 5C: +3 drift field tests |
| `test_renko_vpin.py` (VPIN: BVC, bucketing, baseline, normalisation) | 117 | ✅ |
| `test_circuit_breakers.py` (breakers, manager, cooldowns, weights) | 165 | ✅ |
| `test_renko_qualify.py` (session, qualify, registry, drift detector) | 207 | ✅ Sprint 5B/5C |
| `test_renko_orchestrator.py` (orchestrator, MC, tail-risk, drift wiring) | 137 | ✅ Sprint 5C: +35 tests |
| `test_aggregation.py` | varies | ✅ |
| `test_menu_gates.py` (pipeline gates, M1, Renko gating, badges) | 108 | ✅ |
| `test_pipeline_integration.py` (end-to-end pipeline flows) | 93 | ✅ |
| **Total Renko + monitoring + menu tests** | **1236+** | ✅ |
| **Total project tests** | **2026+ passed** | ✅ |

### New Modules — Sprint 5 (§29.7)

| Module | Purpose | Sprint |
|---|---|---|
| `kinetra/renko/session.py` | Broker fingerprinting: `detect_session_break()`, `SessionProfile`, `clamp_spikes()` | 5B ✅ |
| `kinetra/renko/qualify.py` | `qualify_instrument()`, `QualificationResult`, `QualificationRegistry`, `CalibrationDriftDetector` | 5B ✅ |
| `kinetra/renko/orchestrator.py` | `run_full_pipeline()`, `run_qualification_only()`, `PortfolioPipelineResult`, private helpers | 5C ✅ |
| `kinetra/renko/dsp.py` addition | `scaled_filter_params(dsp_result, bricks_per_day) → FilterParams` | 5A ✅ |
| `kinetra/rl/portfolio_env.py` additions | `PortfolioDaySnapshot.vr_drift`, `PortfolioDaySnapshot.recalibration_pending`, `N_RISK_OBS_FEATURES=10` | 5C ✅ |
| `kinetra/renko/backtest.py` addition | `RiskParams` dataclass (loss-cluster window/threshold/cooldown, DD throttle/halt) | 5A |
| `kinetra/renko/brick_engine.py` addition | `session_break_minutes: float = 30.0` param in `build_renko()` | 5A |
| `scripts/renko/stage1_dsp_screen.py` | CLI: parallel qualification across instruments | 5B |
| `scripts/renko/stage3_multiwindow_backtest.py` | CLI: all Menu 4 Renko backtest modes | 5C |

### Training Scripts (Sprint 4) — Menu 3 Options 10-13

| Script | Menu Option | Layer | Purpose |
|--------|-------------|-------|---------|
| `reward_sweep.py` | 10 | 2+3 | Sweep w1/w2/w3 (Layer 2) and CRB/MRP/URP (Layer 3) reward weights empirically |
| `explore_compare_renko_agents.py` | 11 | 2+3 | Compare agent architectures (Uniform/FullExposure baseline, LinearAC, MLP-Small, MLP-Large, PPO-style) |
| `train_risk_agent.py` | 12 | 3 | Train risk overlay agent using `RiskOverlayEnv` — exposure scalar learning with asymmetric reward |
| `train_allocation_agent.py` | 13 | 2 | Train allocation agent using `RenkoPortfolioEnv` — actor-critic (PyTorch) or linear policy (NumPy) |

### Renko Qualification — Menu 4 Options 7-8 (Sprint 5B, needs M1)

| Menu Option | Mode | Purpose |
|---|---|---|
| 7 | Qualify Instruments | Full qualification pipeline per instrument → `qualification.json` |
| 8 | View Qualification Registry | Table of qualified instruments, omega, cluster, drift flags |

Gate: requires `m1_data_available`.

### Renko Backtesting — Menu 4 Options 9-13 (deterministic, no agent needed)

| Menu Option | Mode | Purpose |
|---|---|---|
| 9 | Instrument Backtest | Single-instrument Renko backtest (flip+filter+loss-cluster strategy) |
| 10 | Portfolio Backtest | All qualified instruments with cluster caps and equal-risk sizing |
| 11 | Walk-Forward | 70/30 IS/OOS per-instrument robustness validation |
| 12 | Monte Carlo | Shuffled entry order, 100+ runs, p < 0.01 significance |
| 13 | Friction Stress | 1.0×/1.5×/2.0× broker costs — survival gate |

Gate: options 9–13 require `m1_data_available`; options 10–13 additionally require `renko_qualified_count >= 1`; option 10 (portfolio) requires `renko_qualified_count >= 3`.


### Menu Structure — Quick Reference

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
  ── RENKO — QUALIFICATION (needs M1 — Sprint 5B) ──
  7    Qualify Instruments        ← full qualification pipeline per instrument
  8    View Qualification Registry← table: qualified instruments, omega, cluster, drift
  ── RENKO — BACKTESTING (deterministic Layer 1 — NO agent needed) ──
  9    Renko Instrument Backtest  ← requires m1_data_available
  10   Renko Portfolio Backtest   ← requires renko_qualified_count >= 3
  11   Renko Walk-Forward         ← requires renko_qualified_count >= 1
  12   Renko Monte Carlo          ← requires renko_qualified_count >= 1
  13   Renko Friction Stress Test ← requires renko_qualified_count >= 1
  ── RESULTS ──
  5    View Backtest Results
  6    Generate Performance Report

MENU 5 — System Tools & Monitoring
  1    System Status & Diagnostics
  2    Manifest Health Check
  3    Circuit Breaker Status      ← view config, state, reset cooldowns
  4    Calibration Drift Status    ← NEW Sprint 5C: per-instrument drift flags, recalibration_due
  5    Cache Management
  6    Backup Data
  7    Clean Temporary Files
  8    Run Tests
  9    View Logs
```

**New `SystemStatus` fields (Sprint 5B/C):**
- `renko_qualified_count: int` — number of instruments with `qualified=True` in registry
- `renko_qualification_done: bool` — `renko_qualified_count >= 1`
- `renko_drift_flags: int` — instruments with `recalibration_due=True`

**For complete, comprehensive rules → See [`AGENT_RULES_MASTER.md`](../README.md)**
