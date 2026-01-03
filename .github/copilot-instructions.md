# GitHub Copilot Instructions for Kinetra

> **⚠️ CANONICAL RULES:** All agent rules are consolidated in [`AGENT_RULES_MASTER.md`](../AGENT_RULES_MASTER.md)
> 
> This file provides a **quick reference** for GitHub Copilot. For complete rules, see the master document.

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
pytest tests/test_physics.py -v  # Run specific test

# Common Commands
python scripts/batch_backtest.py --instrument BTCUSD --timeframe H1
```

---

## Project Overview

**Kinetra** (Kinetic + Entropy + Alpha) is an institutional-grade, physics-first adaptive trading system that uses reinforcement learning to extract returns from market regimes. Built on first principles with **no static assumptions**, Kinetra validates every decision through rigorous statistical testing and continuous backtesting.

---

## Core Philosophy: First-Principles, Zero Assumptions

**CRITICAL**: Question everything! Even established "best practices" are hypotheses to explore, not commandments.

### NEVER:
- ❌ Use magic numbers (20-period MA, 14-period RSI, etc.)
- ❌ Use traditional TA indicators without physics justification
- ❌ Assume linearity without proof
- ❌ Use fixed thresholds (e.g., stop at 2% ATR)
- ❌ Apply universal rules across markets without exploration
- ❌ Remove or modify working code without strong justification

### ALWAYS:
- ✅ Start from thermodynamic/physical first principles
- ✅ Use rolling, adaptive distributions (NO fixed periods)
- ✅ Validate per-market, per-regime, per-timeframe
- ✅ Explore before implementing
- ✅ Question assumptions
- ✅ Let the data guide decisions
- ✅ **Prefer vectorization over Python loops** (NumPy/Pandas ops, broadcasting)

**THE ONLY ASSUMPTION**: Physics is real (energy, friction, entropy exist in markets)

---

## Coding Standards

### Python Style
- Follow **PEP 8** conventions
- Use **Black** for code formatting (line length: 100)
- Use **Ruff** for linting (select: E, F, I, W)
- Target Python 3.10+
- Use type hints for all function signatures
- Prefer explicit over implicit

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
df['energy_pct'] = df['energy'].rolling(window).rank(pct=True)

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
pytest tests/test_physics.py -v

# Run with coverage
pytest tests/ --cov=kinetra --cov-report=html
```

---

## Data Safety & Integrity

### NEVER LOSE USER DATA (#1 Priority)

**Mandatory Before ANY Data Operation:**

1. ✅ **ALWAYS use `PersistenceManager.atomic_save()`** - Never raw file writes
2. ✅ **ALWAYS backup before git operations** - `git rm --cached` can delete files
3. ✅ **CHECK `.gitignore`** before commits - Large files must NEVER be tracked
4. ✅ **NEVER assume backups exist** - Verify before dangerous operations

```python
from kinetra.persistence_manager import get_persistence_manager

pm = get_persistence_manager(backup_dir="data/backups", max_backups=10)
pm.atomic_save(
    filepath="data/master/BTCUSD_H1.csv",
    content=df,
    writer=lambda path, data: data.to_csv(path, index=False)
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

| Metric | Target | Purpose |
|--------|--------|---------|
| **Omega Ratio** | > 2.7 | Asymmetric returns |
| **Z-Factor** | > 2.5 | Statistical edge significance |
| **% Energy Captured** | > 65% | Physics alignment efficiency |
| **Composite Health Score** | > 0.90 | System stability |
| **% MFE Captured** | > 60% | Execution quality |

---

## Common Patterns

### Dynamic Thresholds (No Magic Numbers)
```python
# ✅ GOOD: Rolling percentiles
energy_75pct = np.percentile(history['energy'], 75)
if energy > energy_75pct:
    # High energy regime
    
# ❌ BAD: Fixed threshold
if energy > 0.5:  # Magic number!
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
assert data['price'].notna().all(), "NaN in price data"
assert omega_ratio > 2.7, f"Omega {omega_ratio:.2f} below threshold"

# Include statistical validation
p_value = statistical_test(results)
assert p_value < 0.01, f"Not statistically significant (p={p_value})"
```

---

## When in Doubt

1. Check [`AGENT_RULES_MASTER.md`](../AGENT_RULES_MASTER.md) for complete rules
2. Validate with statistical tests (p < 0.01)
3. Write comprehensive tests first
4. Run the full test suite before committing
5. Question assumptions - "Is this a magic number?"
6. Reference physics first principles
7. Ensure backward compatibility with existing tests

---

## Additional Resources

- **Complete Rules**: [`AGENT_RULES_MASTER.md`](../AGENT_RULES_MASTER.md) - **START HERE**
- **Design Bible**: Complete architecture in `docs/` directory
- **Theorem Proofs**: Mathematical validation in `docs/theorem_proofs.md`
- **Empirical Theorems**: Data-driven discoveries in `docs/EMPIRICAL_THEOREMS.md`
- **Testing Guide**: `docs/TESTING_FRAMEWORK.md`
- **AI Instructions**: `archive/status-reports/AI_AGENT_INSTRUCTIONS.md`

---

**Remember**: Kinetra is built on first principles with rigorous validation. Every decision should be justified mathematically, tested statistically, and validated continuously.

**For complete, comprehensive rules → See [`AGENT_RULES_MASTER.md`](../AGENT_RULES_MASTER.md)**