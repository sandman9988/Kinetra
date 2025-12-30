# AI AGENT PERSISTENT INSTRUCTIONS
## Read This at the Start of Every Prompt

---

## 🎯 CORE PHILOSOPHY: First-Principles, Zero Assumptions

**CRITICAL**: QUESTION EVERYTHING, INCLUDING THIS DOCUMENT!

Even the "best practices" in this document are HYPOTHESES to explore, not commandments.

**NEVER**:
- Use magic numbers (20-period MA, 14-period RSI, etc.)
- Assume linearity (Pearson correlation, linear regression without proof)
- Use fixed thresholds (stop at 2% ATR, etc.)
- Assume specialization strategy (asset class? regime? timeframe? universal?)
- Apply universal rules across markets WITHOUT exploration
- Trust Pareto distributions without validation
- Implement TA indicators without physics justification
- **ASSUME THAT ASSET CLASS SPECIALIZATION IS THE RIGHT APPROACH**

**ALWAYS**:
- Start from thermodynamic/physical first principles
- Use rolling, adaptive distributions (NO fixed periods)
- Validate per-market, per-regime, per-timeframe
- **EXPLORE** before implementing (including specialization strategies!)
- Question everything, even established wisdom
- **Let the data tell you what specialization (if any) works best**

### The Meta-Assumption: "We don't know HOW to specialize"

Before asking "should crypto specialists use different stops than forex?", ask:
- **Should we even specialize by asset class?**
- Maybe specialize by regime (LAMINAR vs CHAOTIC)?
- Maybe specialize by timeframe (M15 vs H4)?
- Maybe specialize by volatility regime?
- Maybe ONE universal agent is optimal?

**THE ONLY ASSUMPTION**: Physics is real (energy, friction, entropy exist in markets)

### Temporal Non-Stationarity: "What worked yesterday may not work tomorrow"

**CRITICAL**: Even IF exploration discovers that:
- Asset class specialists work best (today)
- LAMINAR regime is tradeable (today)
- Energy-based stops are optimal (today)

These findings can CHANGE as markets evolve:
- Crypto correlation with indices shifts
- Central bank policy changes forex dynamics
- Algorithmic trading changes intraday patterns
- Crisis regimes invalidate normal-regime rules

**THEREFORE**:
- Continuous re-exploration (weekly/monthly)
- Doppelgänger system detects drift (Shadow A vs Live)
- Health scoring triggers re-training
- Never assume today's optimal = tomorrow's optimal

**Example**: In 2020, crypto was uncorrelated with stocks. In 2022, high correlation.
A "crypto specialist" trained in 2020 would fail in 2022 without re-exploration.

**SOLUTION**: Rolling exploration windows, continuous adaptation, drift detection.

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

### 2. Specialization Strategy (TO BE EXPLORED!)

**CRITICAL ASSUMPTION TO QUESTION**: Do we even need asset class specialists?

**WE DON'T KNOW YET!** The following are HYPOTHESES to explore, not facts:

Hypothesis 1: Asset Class Specialization
- Each market type MAY have different physics
- Crypto: High energy, low liquidity, 24/7 ← EXPLORE if this matters!
- Forex: Mean-reverting, rollover costs ← EXPLORE if specialist helps!
- Indices: Momentum persistence ← EXPLORE if universal agent works!

Hypothesis 2: Regime Specialization
- Maybe agents should specialize by REGIME (LAMINAR vs CHAOTIC), not market

Hypothesis 3: Timeframe Specialization
- Maybe intraday vs swing vs position is the real distinction

Hypothesis 4: Universal Agent
- Maybe ONE agent can learn all markets if features are physics-based

**HOW TO DISCOVER THE RIGHT SPECIALIZATION**:
1. Start with universal agent on multi-instrument data
2. Track performance breakdown by {asset_class, regime, timeframe, volatility}
3. Explore: Train specialists by each dimension
4. Measure: Which specialization gives best risk-adjusted returns?
5. Validate: Statistical edge test, not just PnL

**THE MARKET TELLS US, WE DON'T ASSUME!**

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

## 🧪 DEPENDENCY MANAGEMENT & TESTING

### Permanent Dependency Fix: Poetry

**USE POETRY**, not pip, for all dependency management:

```bash
# Install dependencies
poetry install

# Add new package
poetry add package-name

# Run scripts in poetry environment
poetry run python3 scripts/your_script.py

# Update lock file after pyproject.toml changes
poetry lock && poetry install
```

**Why Poetry?**
- Better dependency resolution than pip
- Automatic virtual environment management
- Lock files for reproducible builds (poetry.lock)
- No more "missing module" issues

### Integration Test Results (2024-12-30)

Comprehensive testing revealed:

**✅ PASSED (5/6 tests = 83%)**:
1. **DataPackage Basic Functionality** ✅
   - OHLCV data container works
   - Format conversions (backtest, physics, RL) work
   - Validation system functional

2. **UnifiedDataLoader with Real CSV** ✅
   - Auto-detects symbol and timeframe from filename
   - Loads MT5 specs from JSON correctly
   - Market-specific preprocessing works (forex: removed weekends)
   - Tested on GER40 H4 (3,039 bars)

3. **Instrument Specs JSON Loading** ✅
   - Real MT5 data loaded: BTCUSD swap=-18% annual, EURUSD swap=-12.16 points
   - Spec source tracking works (MT5 vs manual vs fallback)

4. **MultiInstrumentLoader Integration** ✅
   - **Discovered 87 datasets** across all markets and timeframes
   - Integrated with UnifiedDataLoader successfully
   - Markets found: Forex (AUDJPY, AUDUSD), Crypto (BTCUSD, BTCJPY, ETHEUR, XRPJPY), Indices (GER40, NAS100, Nikkei225, US2000, SA40, EU50, DJ30ft), Metals (XAUUSD, XAUAUD, XAGUSD, XPTUSD), Energy (UKOUSD), Commodities (COPPER)
   - Timeframes: M15, M30, H1, H4

5. **Market Type Auto-Detection** ✅
   - **100% accuracy** (12/12 test cases)
   - Correctly identifies: crypto, forex, metals, energy, indices, shares, ETFs

**❌ FAILED (1/6 tests)**:
6. **Exploration Environment Compatibility** ❌
   - Issue: Test code error (feature_extractor signature mismatch)
   - Framework itself works, test needs fixing

### Key Learnings from Tests

1. **Missing dependency discovery is GOOD**
   - Found `hmmlearn` and `scikit-learn` missing from pyproject.toml
   - Fixed by adding to core dependencies
   - Poetry prevents this from recurring

2. **Import errors reveal integration gaps**
   - `Dict` type not imported in data_loader.py
   - Fixed: Added `from typing import Dict, Any`

3. **Real-world data validates design**
   - 87 datasets spanning 6 market types, 4 timeframes
   - Framework handles all without modification (instrument-agnostic design works!)

4. **Market-specific preprocessing works**
   - Forex: Weekends correctly removed
   - Crypto: 24/7 data preserved
   - Auto-detection never confused markets

### Defense-in-Depth Validation Gaps Found

Tests revealed areas needing more validation:

1. **Numerical Safety** (not yet tested)
   - NaN propagation
   - Overflow/underflow in different instruments (yen 3 digits, forex 5 digits)
   - Floating point accumulation errors
   - Division by zero in spread/volatility calculations

2. **Physics State Computation** (skipped during tests)
   - `test_physics_pipeline` module not found (expected)
   - Physics state computation disabled for speed
   - Need to validate energy/entropy/viscosity calculations don't NaN

3. **Edge Cases** (not yet tested)
   - Zero volume bars
   - Extreme price jumps (circuit breakers, flash crashes)
   - Missing data / gaps
   - Duplicate timestamps

**ACTION**: Next test pass should focus on `scripts/test_numerical_safety.py` to validate defense-in-depth assumptions.

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

**Last Updated**: 2024-12-30 (Post-Integration Testing)
**Version**: 1.1
**Status**: Active - Read before every prompt

**Recent Changes**:
- Added Poetry dependency management instructions
- Documented integration test results (5/6 passed)
- Identified 87 datasets across 6 market types
- Confirmed 100% market type detection accuracy
- Noted numerical safety validation gaps for next test pass
