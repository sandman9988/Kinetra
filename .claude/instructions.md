# Kinetra AI Development Instructions

> **⚠️ CANONICAL RULES:** All agent rules are consolidated in [`AGENT_RULES_MASTER.md`](../AGENT_RULES_MASTER.md)
> 
> This file provides a **quick reference** for Claude/Zed AI agents. For complete rules, see the master document.

---

## Quick Reference

**For comprehensive, authoritative rules → See [`AGENT_RULES_MASTER.md`](../AGENT_RULES_MASTER.md)**

---

## ⚠️ MANDATORY: Core Philosophy Summary

**EVERYTHING is derived from first principles. NO static rules.**

### What We DON'T Use
- ❌ NO traditional TA indicators (ATR, BB, RSI, MACD, ADX, etc.)
- ❌ NO hardcoded thresholds (no "if energy > 0.8")
- ❌ NO static values (no "volume spike > 1.5x")
- ❌ NO time-based filters
- ❌ NO magic numbers
- ❌ NO rules - only features for RL to discover patterns

### What We DO Use
- ✅ Physics (energy, damping, entropy, viscosity, Reynolds)
- ✅ Thermodynamics (energy states, phase transitions)
- ✅ Kinematics (velocity, acceleration, jerk, momentum)
- ✅ Rolling percentiles (adaptive to current distribution)
- ✅ Probability distributions (where in the distribution is current value?)
- ✅ First principles derivation

**THE ONLY ASSUMPTION**: Physics is real (energy, friction, entropy exist in markets)

---

## Physics First Principles (Quick Reference)

| Concept | Formula | Meaning |
|---------|---------|---------|
| Kinetic Energy | E = ½mv² = ½ × velocity² | Energy in motion |
| Damping | ζ = σ(v) / μ(\|v\|) | Energy dissipation |
| Reynolds | Re = (v × L × ρ) / μ | Laminar vs turbulent |
| Viscosity | μ = resistance / flow | Market friction |
| Spring Stiffness | k = F / x = volume / Δprice | Resistance to displacement |
| Phase Space | (position, momentum) | State confinement |
| Entropy | S = disorder measure | Predictability |

---

## Adaptive Percentiles (NO Magic Numbers)

Every metric should be converted to its position in the rolling distribution:

```python
# ✅ CORRECT: Adaptive percentile
feature_pct = feature.rolling(window).apply(
    lambda x: (x.iloc[-1] > x.iloc[:-1]).mean()
)
# Returns 0-1: where does current value sit in recent history?

# ❌ WRONG: Static threshold
if feature > 0.8:  # NO! This is not adaptive
```

---

## Vectorization (MANDATORY)

Explicit Python loops are the **last resort**.

**Prefer:**
- ✅ NumPy vectorized ops: `energy = 0.5 * velocity ** 2`
- ✅ Pandas column operations: `df['energy_pct'] = df['energy'].rolling(window).rank(pct=True)`
- ✅ Broadcasting: `result = arr_2d + arr_1d[:, np.newaxis]`

**Avoid:**
- ❌ Explicit Python loops (only if unavoidable, keep tight)

---

## Data Safety & Integrity (#1 Priority)

### NEVER LOSE USER DATA

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

## Testing Requirements (Multi-Layer Validation)

1. **Unit Tests** (`pytest`)
   - 100% code coverage required
   - Property-based testing with `hypothesis`
   - Numerical stability checks

2. **Integration Tests**
   - End-to-end pipeline validation

3. **Monte Carlo Backtesting**
   - 100 runs per instrument minimum
   - Statistical significance testing (p < 0.01)
   - Out-of-sample validation required

4. **Theorem Validation**
   - Mathematical proofs must be documented

5. **Health Monitoring**
   - Real-time Composite Health Score (CHS)
   - Circuit breakers (halt if CHS < 0.55)

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

## GPU Requirements

Training REQUIRES GPU acceleration. CPU training is 100x slower.

**Check GPU availability:**
```python
import torch
print(f"CUDA: {torch.cuda.is_available()}")
print(f"Devices: {torch.cuda.device_count()}")
```

**For AMD GPUs (ROCm):**
```bash
pip install torch --index-url https://download.pytorch.org/whl/rocm6.0
export HSA_OVERRIDE_GFX_VERSION=11.0.0  # For RX 7600/RDNA3
export HIP_VISIBLE_DEVICES=0
```

**For NVIDIA GPUs (CUDA):**
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

**CRITICAL**: If no GPU detected, DO NOT proceed with training. Fix GPU first.

---

## Security & Hard Prohibitions

### NEVER:
- ❌ Commit API keys, secrets, or credentials
- ❌ Write code that places live orders (backtest/paper only)
- ❌ Hardcode credentials
- ❌ Expose sensitive data in logs

### ALWAYS:
- ✅ Use environment variables (`.env` file) for sensitive data
- ✅ Reference `.env.example` for required variables
- ✅ Use circuit breakers for abnormal conditions
- ✅ Model slippage in backtest

---

## Summary

> **"We don't trade rules. We provide physics state. RL discovers edges."**

Every feature should be:
1. Derived from first principles (physics)
2. Expressed as a percentile (adaptive)
3. Instrument-agnostic (works everywhere)
4. Fed to RL for pattern discovery (not rules)

---

## Additional Resources

- **Complete Rules**: [`AGENT_RULES_MASTER.md`](../AGENT_RULES_MASTER.md) - **START HERE**
- **GitHub Copilot Quick Reference**: [`.github/copilot-instructions.md`](../.github/copilot-instructions.md)
- **Type Checking Guidelines**: [`.claude/type_checking_guidelines.md`](type_checking_guidelines.md)
- **Design Bible**: Complete architecture in `docs/` directory
- **Theorem Proofs**: Mathematical validation in `docs/theorem_proofs.md`
- **Empirical Theorems**: Data-driven discoveries in `docs/EMPIRICAL_THEOREMS.md`

---

**Remember**: Kinetra is built on first principles with rigorous validation. Every decision should be justified mathematically, tested statistically, and validated continuously.

**For complete, comprehensive rules → See [`AGENT_RULES_MASTER.md`](../AGENT_RULES_MASTER.md)**