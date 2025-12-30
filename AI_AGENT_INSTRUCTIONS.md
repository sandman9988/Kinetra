# AI AGENT PERSISTENT INSTRUCTIONS
## Read This at the Start of Every Prompt

---

## 🎯 CORE PHILOSOPHY: First-Principles, Zero Assumptions

**NEVER**:
- Use magic numbers (20-period MA, 14-period RSI, etc.)
- Assume linearity (Pearson correlation, linear regression without proof)
- Use fixed thresholds (stop at 2% ATR, etc.)
- Apply universal rules across markets (crypto ≠ stocks ≠ forex)
- Trust Pareto distributions without validation
- Implement TA indicators without physics justification

**ALWAYS**:
- Start from thermodynamic/physical first principles
- Use rolling, adaptive distributions (NO fixed periods)
- Validate per-market, per-regime
- Explore before implementing
- Question everything, even established wisdom

---

## 🏗️ SYSTEM ARCHITECTURE

### 1. Multi-Layer Structure

```
Data Pipeline:
  MT5 Terminal
  → extract_mt5_specs.py
  → instrument_specs.json
  → UnifiedDataLoader
  → DataPackage
  → Exploration Engine / Backtester

Trading System:
  Specialist Agents (per asset class)
  → Doppelgänger Triads (Live, Shadow A, Shadow B)
  → Regime Filters (Physics + Volatility + Momentum)
  → Portfolio Health Monitoring
  → Self-Healing Actions
```

### 2. Asset Class Specialists

Each market type has DIFFERENT physics:
- **Crypto** (BTCUSD): High energy, low liquidity, 24/7, gap risk dominant
- **Forex** (EURUSD): Mean-reverting, rollover costs, 24/5, liquidity cycles
- **Indices** (NAS100): Momentum persistence, session-dependent, vol clustering
- **Metals** (XAUUSD): 23-hour trading, safe-haven flows, maintenance windows
- **Energy** (XTIUSD): Inventory-driven, contango/backwardation, storage costs
- **Shares/ETFs**: Corporate actions, exchange hours, earnings gaps

**CRITICAL**: What works for crypto trailing stops will NOT work for forex!

### 3. Doppelgänger System

For EVERY specialist agent, maintain 3 versions:
- **Live**: Currently trading
- **Shadow A**: Frozen checkpoint (drift detection)
- **Shadow B**: Online learning (candidate replacement)

**Promotion Rule**: Shadow B replaces Live if:
1. Statistical edge validated (>100 trades)
2. Sharpe ratio improvement >0.3
3. Max drawdown not worse than Live
4. Health score >80

---

## 🔬 MEASUREMENT FRAMEWORK

### Physics-Based Primitives (NOT TA)

For ANY measurement (trailing stop, entry signal, sizing), start with:

| Primitive | Physical Analogue | Formula | Why It Matters |
|-----------|-------------------|---------|----------------|
| **Volume Loss** | Entropy production | `∫(vol_noise²) dt` | Measures useless volume (noise vs signal) |
| **Energy Dissipation** | Kinetic loss | `∫(price_jerk²) dt` | Predicts exhaustion/reversal |
| **Liquidity Loss** | Friction | `spread × vol⁻¹` | High = slippage trap |
| **Viscosity** | Response lag | `VPIN lag` | High = overdamped = mean-reversion |
| **Nonlinearity** | Chaos | `Lyapunov exponent` | Quantifies predictability |
| **Laminar Flow** | Regime stability | `low jerk + age` | Only trends worth following |

### Adaptive Volatility Estimators

**DO NOT** use simple std dev! Use market-appropriate estimators:

- **Indices/Forex**: Yang-Zhang (accounts for gaps)
- **Commodities**: Rogers-Satchell (drift-independent)
- **Crypto**: Realized Variance (high-frequency)

**ALL** with adaptive lookbacks (via cycle period or change-point detection).

---

## 🎲 EXPLORATION-DRIVEN DEVELOPMENT

### For ANY New Feature (e.g., Trailing Stop)

**Step 1: Define Parametric Family**
```python
# Example: Trailing stop for crypto
def stop_distance(state, weights):
    return (
        weights['energy'] * energy_dissipation_norm +
        weights['viscosity'] * viscosity_norm +
        weights['liquidity'] * liquidity_loss_norm +
        weights['regime'] * (1.0 if regime == "LAMINAR" else 2.0)
    )
```

**Step 2: Explore Parameter Space**
- Use multi-armed bandit or PBT (population-based training)
- Reward = **fitness score**, NOT raw PnL
- Run per-market, per-regime

**Step 3: Validate Fitness**
```python
def trailing_stop_fitness(trade):
    energy_capture = trade.net_pnl / trade.mfe  # 0-1
    leakage = 1 - (trade.mae / trade.mfe)       # Penalize MAE
    regime_bonus = 1.2 if trade.regime in ["LAMINAR", "UNDERDAMPED"] else 0.5
    return energy_capture * leakage * regime_bonus
```

**Step 4: Stack Non-Linearly**
```python
# Meta-learner discovers interactions
X = [stop_energy, stop_visc, stop_vol, momentum_regime, vol_regime]
y = optimal_stop_distance_from_hindsight
meta_model.fit(X, y)  # XGBoost or tiny NN
```

**Step 5: Specialize Per Market**
- The system discovers autonomously that:
  - Crypto: energy + gap risk dominant
  - Indices: regime age + vol clustering
  - Forex: trailing stops often inferior to fixed MFE

---

## 🚦 REGIME-AWARE FILTERING

**NEVER** take a trade without 3-regime confluence:

### 1. Physics Regime (from physics_engine.py)
- **LAMINAR**: Smooth trend, low jerk → TRADE
- **UNDERDAMPED**: Momentum with oscillation → TRADE (with wider stops)
- **OVERDAMPED**: Mean-reverting → BLOCK trend following
- **CHAOTIC**: High jerk, unpredictable → BLOCK ALL

### 2. Volatility Regime (adaptive vol estimator)
- **LOW**: Mean-reversion dominant
- **NORMAL**: Trend-following viable
- **HIGH**: Chaos, avoid trading

### 3. Momentum Regime (from ROC or dsp_trend_dir)
- **WEAK**: Chop, avoid
- **MODERATE**: Pullback/swing trades
- **STRONG**: Breakout trades

**Trading Signal** = Specialist Signal AND (Physics Regime ∈ {LAMINAR, UNDERDAMPED}) AND (Vol Regime ≠ HIGH) AND (Momentum Regime ≠ WEAK)

---

## 📊 DATA PACKAGE INTEGRATION

When loading data for exploration or backtesting:

```python
from kinetra.data_loader import UnifiedDataLoader
from kinetra.data_package import DataPackage

# ALWAYS use UnifiedDataLoader (instrument-agnostic)
loader = UnifiedDataLoader(validate=True, verbose=True)
pkg = loader.load("data/master/BTCUSD_H1_*.csv")

# DataPackage contains:
# - prices: OHLCV
# - physics_state: 64-dim features (if computed)
# - symbol_spec: Real MT5 specs (swaps, margins, spreads)
# - market_type: AssetClass enum (auto-detected)
# - quality_report: Validation results

# Use appropriate format for your engine
backtest_data = pkg.to_backtest_engine_format()
rl_state, rl_prices = pkg.to_rl_environment_format()
```

**CRITICAL**: The specs in `pkg.symbol_spec` are REAL MT5 data, not hardcoded!
- BTCUSD: swap_long=-18% (annual), swap_type="percent_annual"
- EURUSD: swap_long=-12.16 points, swap_type="points"

Use these for **accurate cost calculations** in backtests.

---

## 🏥 PORTFOLIO HEALTH MONITORING

Track 4 pillars (continuous scoring):

| Pillar | Metrics | Weight |
|--------|---------|--------|
| **Return & Efficiency** | CAGR, Sharpe, Omega, Calmar | 25% |
| **Downside Risk** | Max DD, Ulcer Index, Recovery Time | 30% |
| **Structural Stability** | Avg correlation, Eigenvalue crowding | 25% |
| **Behavioral Health** | Edge decay, promotion frequency | 20% |

### Health Score → Actions

- **>80**: Normal operation
- **60-80**: Reduce risk 30%, increase monitoring
- **40-60**: Retrain underperformers, add hedges
- **<40**: Go flat, emergency retraining

---

## 🚫 ANTI-PATTERNS TO AVOID

1. **"Let's use 14-period RSI"** → NO! Why 14? Explore or use adaptive period
2. **"ATR is ATR everywhere"** → NO! Yang-Zhang for indices, Rogers-Satchell for commodities
3. **"This works for BTCUSD, ship it"** → NO! Validate on EURUSD, NAS100, XAUUSD separately
4. **"PnL is up, we're good"** → NO! Check health score, edge decay, regime breakdown
5. **"Hardcode stop at 2%"** → NO! Explore stop = f(energy, viscosity, liquidity, regime)
6. **"Assume normal distribution"** → NO! Use regime-conditioned quantiles

---

## ✅ SUCCESS CRITERIA

For ANY feature/agent/measurement:

1. **No magic numbers**: All parameters explored or adaptively estimated
2. **Physics-consistent**: Thermodynamic justification exists
3. **Market-specific**: Crypto ≠ Stocks ≠ Forex validation
4. **Regime-aware**: Only trades when physics permits profit
5. **Self-improving**: New data → new exploration → better rules

---

## 🔄 CURRENT INTEGRATION POINTS

### Existing System Components
- ✅ `rl_exploration_framework.py`: TradingEnv, RewardShaper, LinearQAgent, MultiInstrumentLoader
- ✅ `kinetra/physics_engine.py`: Physics state computation (64-dim)
- ✅ `kinetra/backtest_engine.py`: Friction-aware backtesting
- ✅ `kinetra/data_package.py`: Standardized data container
- ✅ `kinetra/data_loader.py`: Auto-loads instrument specs from JSON
- ✅ `kinetra/mt5_spec_extractor.py`: Extracts real broker data

### Next Integration Steps
1. Update `MultiInstrumentLoader` to use `UnifiedDataLoader`
2. Add regime filters to `TradingEnv`
3. Implement `DoppelgangerTriad` wrapper for agents
4. Add `PortfolioHealthScore` monitoring
5. Build exploration framework for trailing stops

---

## 💡 WHEN IN DOUBT

Ask yourself:
1. **Is this assumption-free?** (No magic numbers?)
2. **Is this physics-based?** (Thermodynamic justification?)
3. **Is this market-specific?** (Validated per asset class?)
4. **Is this explored or hardcoded?** (Did we discover or assume?)
5. **Does this work in ALL regimes?** (Or only LAMINAR?)

If ANY answer is "no", **stop and explore first**.

---

## 📝 REMEMBER

> "We don't know what we don't know. The market will teach us through exploration, not through assumptions."

> "If you can't explain it with physics (energy, friction, viscosity, entropy), you don't understand it."

> "Crypto is not stocks. Stocks are not forex. One rule does not fit all."

---

**Last Updated**: 2024-12-30
**Version**: 1.0
**Status**: Active - Read before every prompt
