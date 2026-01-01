# GitHub Copilot Instructions for Kinetra

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

## Project Overview

**Kinetra** (Kinetic + Entropy + Alpha) is an institutional-grade, physics-first adaptive trading system that uses reinforcement learning to extract returns from market regimes. Built on first principles with **no static assumptions**, Kinetra validates every decision through rigorous statistical testing and continuous backtesting.

## Core Philosophy: First-Principles, Zero Assumptions

**CRITICAL**: Question everything! Even established "best practices" are hypotheses to explore, not commandments.

### NEVER:
- Use magic numbers (20-period MA, 14-period RSI, etc.)
- Assume linearity without proof
- Use fixed thresholds (e.g., stop at 2% ATR)
- Apply universal rules across markets without exploration
- Implement TA indicators without physics justification
- Remove or modify working code without strong justification

### ALWAYS:
- Start from thermodynamic/physical first principles
- Use rolling, adaptive distributions (NO fixed periods)
- Validate per-market, per-regime, per-timeframe
- Explore before implementing
- Question assumptions
- Let the data guide decisions

**THE ONLY ASSUMPTION**: Physics is real (energy, friction, entropy exist in markets)

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

### Documentation
- Add docstrings for all public functions and classes
- Include mathematical formulas in LaTeX format where relevant
- Reference theorem proofs in `docs/theorem_proofs.md`
- Document empirical findings in `docs/EMPIRICAL_THEOREMS.md` (p < 0.01)
- Keep README and documentation in sync with code

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

## Building and Linting

### Setup Development Environment
```bash
# Recommended: Use make for automated setup
make setup          # Full Python dev environment

# Alternative: Manual installation
pip install -r requirements.txt
pip install -e ".[dev]"  # Install with dev dependencies (pytest, black, ruff, etc.)
```

**Note**: Dev dependencies (pytest, black, ruff, mypy, hypothesis) are required for development work. These are specified in `pyproject.toml` under `[project.optional-dependencies]`.

### Code Quality
```bash
make lint           # Run Ruff linting
make format         # Format with Black
```

## Security and Safety

### Credential Security
- **NEVER** commit API keys, secrets, or credentials
- Use environment variables (`.env` file) for sensitive data
- Reference `.env.example` for required variables

### Data Validation
- Use Pydantic schemas to enforce type/range contracts
- Validate all external inputs
- Implement NaN shields for numerical calculations

### Execution Safety
- Circuit breakers for abnormal conditions
- Fallback policies for error handling
- Slippage modeling in backtest
- Risk-of-Ruin (RoR) gates before execution

## Architecture Patterns

### Core Components
- **Physics Engine**: Energy-based market modeling (`kinetra/physics_engine.py`)
- **Risk Management**: Non-linear Risk-of-Ruin, CHS (`kinetra/risk_management.py`)
- **RL Agent**: PPO reinforcement learning (`kinetra/rl_agent.py`)
- **Reward Shaping**: Adaptive reward with MFE/MAE (`kinetra/reward_shaping.py`)
- **Backtest Engine**: Monte Carlo validation (`kinetra/backtest_engine.py`)

### Data Flow
```
Market Data → Physics Engine → Regime Detection → RL Agent → Risk Management → Execution
```

### Key Mathematics
```python
# Energy-Transfer Theorem
E_t = 0.5 * m * (ΔP_t / Δt)²

# Non-Linear Risk-of-Ruin
P(ruin) = exp(-2μ(X_t - L_t) / σ²_t)

# Adaptive Reward Shaping
R_t = (PnL / E_t) + α·(MFE/ATR) - β·(MAE/ATR) - γ·Time
```

## Performance Targets

| Metric | Target | Purpose |
|--------|--------|---------|
| **Omega Ratio** | > 2.7 | Asymmetric returns |
| **Z-Factor** | > 2.5 | Statistical edge significance |
| **% Energy Captured** | > 65% | Physics alignment efficiency |
| **Composite Health Score** | > 0.90 | System stability |
| **% MFE Captured** | > 60% | Execution quality |

## Common Patterns

### Dynamic Thresholds (No Magic Numbers)
```python
# GOOD: Rolling percentiles
energy_75pct = np.percentile(history['energy'], 75)
if energy > energy_75pct:
    # High energy regime
    
# BAD: Fixed threshold
if energy > 0.5:  # ❌ Magic number!
```

### NaN Protection
```python
# GOOD: Shield against NaN
value = np.clip(raw_value, MIN_VALUE, MAX_VALUE)
if not np.isfinite(value):
    value = fallback_value

# BAD: No protection
value = raw_value  # ❌ Could be NaN/Inf
```

### Validation Pattern
```python
# GOOD: Validate assumptions
assert len(data) > MIN_SAMPLES, "Insufficient data"
assert data['price'].notna().all(), "NaN in price data"
assert omega_ratio > 2.7, f"Omega {omega_ratio:.2f} below threshold"

# Include statistical validation
p_value = statistical_test(results)
assert p_value < 0.01, f"Not statistically significant (p={p_value})"
```

## Dependencies

### Core Libraries
- `numpy` - Numerical computing
- `pandas` - Data manipulation
- `scipy` - Scientific computing
- `pydantic` - Data validation
- `scikit-learn` - Machine learning utilities
- `hmmlearn` - Hidden Markov Models

### Development Tools
- `pytest` - Testing framework
- `pytest-cov` - Coverage reporting
- `hypothesis` - Property-based testing
- `black` - Code formatting
- `ruff` - Linting
- `mypy` - Type checking

**Note**: Refer to `pyproject.toml` for exact version requirements and additional dependencies.

## File Organization

```
kinetra/
├── physics_engine.py      # Energy, damping, entropy calculations
├── risk_management.py     # RoR, CHS, position sizing
├── rl_agent.py           # Reinforcement learning (PPO)
├── reward_shaping.py     # Adaptive reward (ARS)
├── backtest_engine.py    # Monte Carlo validation
├── health_monitor.py     # Real-time monitoring
├── testing_framework.py  # Comprehensive testing system
└── mt5_connector.py      # MetaTrader 5 integration

tests/
├── test_physics.py       # Physics engine tests
├── test_risk.py          # Risk management tests
├── test_integration.py   # End-to-end tests
└── ...
```

## Git Workflow

### Branch Naming
- Feature: `feature/description`
- Bug fix: `fix/description`
- Refactor: `refactor/description`

### Commit Messages
- Use clear, descriptive messages
- Reference issue numbers when applicable
- Example: `Add energy-based regime detection (#42)`

## Additional Resources

- **Design Bible**: Complete architecture in `docs/` directory
- **Theorem Proofs**: Mathematical validation in `docs/theorem_proofs.md`
- **Empirical Theorems**: Data-driven discoveries in `docs/EMPIRICAL_THEOREMS.md`
- **Testing Guide**: `docs/TESTING_FRAMEWORK.md`
- **API Reference**: Inline docstrings and function documentation
- **Main Instructions**: `AI_AGENT_INSTRUCTIONS.md` for detailed system philosophy

## When in Doubt

1. Check existing code patterns in the same module
2. Validate with statistical tests (p < 0.01)
3. Write comprehensive tests first
4. Run the full test suite before committing
5. Question assumptions - "Is this a magic number?"
6. Reference physics first principles
7. Ensure backward compatibility with existing tests

---

**Remember**: Kinetra is built on first principles with rigorous validation. Every decision should be justified mathematically, tested statistically, and validated continuously.
