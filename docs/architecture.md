# Kinetra Architecture

## Overview

Kinetra is a physics-first, adaptive algorithmic trading system that uses reinforcement learning to extract returns from market regimes. The system models markets as kinetic energy systems with damping and entropy, enabling regime-aware decision making without static assumptions.

## Core Components

### 1. Physics Engine (`physics_engine.py`)

Models markets using classical mechanics principles:

- **Kinetic Energy**: `E_t = 0.5 * m * (ΔP_t / Δt)²`
  - Represents market momentum
  - Higher energy → stronger trends
  
- **Damping Coefficient**: `ζ = volatility / mean_absolute_return`
  - Represents market friction/resistance
  - Higher damping → more ranging behavior
  
- **Entropy**: `H = -Σ p_i * log(p_i)`
  - Represents market disorder/uncertainty
  - Higher entropy → less predictable

**Regime Classification** (dynamic, percentile-based):
- **Underdamped**: High energy, low damping → trending markets
- **Critical**: Balanced → transitional
- **Overdamped**: High damping → ranging markets

### 2. Risk Management (`risk_management.py`)

Multi-layer risk control system:

- **Non-Linear Risk-of-Ruin**: `P(ruin) = exp(-2μ(X_t - L_t) / σ²)`
  - Adapts to current equity and volatility
  - Circuit breaker if RoR > 10%

- **Composite Health Score (CHS)**:
  - `CHS = 0.4 * CHS_agents + 0.3 * CHS_risk + 0.3 * CHS_class`
  - Agent health: win rate, win/loss ratio, Omega
  - Risk health: RoR, drawdown, volatility
  - Classification health: regime stability, energy capture

- **Dynamic Position Sizing**:
  - Scaled by RoR and CHS
  - Optional Kelly criterion adjustment
  - Zero position if gates fail

### 3. RL Agent (`rl_agent.py`)

Proximal Policy Optimization (PPO) agent:

- **State Space**: [energy, damping, entropy, position, pnl, ...]
- **Action Space**: [hold, buy, sell, close]
- **Policy Network**: Kinematic MLP with physics-aware architecture
- **Value Network**: Estimates expected returns

### 4. Reward Shaping (`reward_shaping.py`)

Adaptive Reward Shaping (ARS):

```
R_t = (PnL / E_t) + α·(MFE/ATR) - β·(MAE/ATR) - γ·Time
```

Where:
- `α, β` are regime-adaptive (scale with volatility)
- MFE = Maximum Favorable Excursion (best unrealized profit)
- MAE = Maximum Adverse Excursion (worst unrealized loss)
- Time penalty discourages holding losing positions

### 5. Backtest Engine (`backtest_engine.py`)

Monte Carlo validation framework:

- Run 100+ backtests with different random seeds
- Statistical significance testing (p < 0.01)
- Out-of-sample validation (Jul–Dec 2025)
- Performance metrics: Omega, Z-factor, %Energy Captured

### 6. Health Monitor (`health_monitor.py`)

Real-time system monitoring:

- Continuous CHS tracking
- Circuit breakers (halt if CHS < 0.55)
- Drift detection
- Performance anomaly alerts

## Data Flow

```
Market Data (OHLCV)
       ↓
Physics Engine
  → Energy, Damping, Entropy
       ↓
Regime Classification
  → Underdamped/Critical/Overdamped
       ↓
RL Agent
  → Policy selects action
       ↓
Risk Management
  → Check RoR, CHS gates
  → Calculate position size
       ↓
Execution
  → Place order (if gates pass)
       ↓
Health Monitor
  → Track CHS, detect drift
```

## First Principles Design

1. **No Fixed Thresholds**: All gates use rolling percentiles
2. **No Fixed Timeframes**: Decisions based on regime, not clock
3. **No Human Bias**: All logic from physics + RL optimization
4. **No Placeholders**: Production-ready code only

## Performance Targets

| Metric | Target | Purpose |
|--------|--------|---------|
| Omega Ratio | > 2.7 | Asymmetric returns |
| Z-Factor | > 2.5 | Statistical edge |
| % Energy Captured | > 65% | Physics alignment |
| CHS | > 0.90 | System stability |
| False Activation Rate | < 5% | Noise filtering |
| % MFE Captured | > 60% | Execution quality |

## Mathematical Proofs

See `theorem_proofs.md` for detailed derivations of:
- Energy-Transfer Theorem
- Risk-of-Ruin Optimality
- ARS Reward Gradient
- CHS Calibration

## References

- Sutton & Barto: Reinforcement Learning
- Statistical Mechanics principles
- Kelly Criterion
- Modern Portfolio Theory
