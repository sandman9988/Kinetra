# Backtester Comparison: BacktestEngine vs RealisticBacktester

## Overview

This document compares the existing `BacktestEngine` (kinetra/backtest_engine.py) with the new `RealisticBacktester` (kinetra/realistic_backtester.py) to identify features that should be ported over.

## Feature Comparison

### ✅ Already in RealisticBacktester

| Feature | BacktestEngine | RealisticBacktester | Notes |
|---------|----------------|---------------------|-------|
| **MFE/MAE Tracking** | ✅ Lines 61-62, 516-526 | ✅ Lines 59-60 | Max Favorable/Adverse Excursion |
| **Spread Cost** | ✅ Line 451 | ✅ Dynamic per-candle | RealisticBacktester uses ACTUAL spread from data |
| **Commission** | ✅ Lines 452, 495-500 | ✅ Lines 47-53 | Both use SymbolSpec |
| **Slippage** | ✅ Lines 453-454, 498-499 | ✅ Lines 54-56 | RealisticBacktester more sophisticated |
| **Swap/Holding Costs** | ✅ Lines 503-505 | ✅ Line 52 | Overnight interest charges |
| **Regime Classification** | ✅ Lines 57-58, 273-278 | ✅ Lines 62-65 | Physics-based regime tracking |
| **Trade Direction** | ✅ Enum Lines 26-30 | ✅ Int (1/-1) | Different representation |
| **Freeze Zones** | ❌ NOT PRESENT | ✅ Lines 101-118 | **CRITICAL MT5 constraint** |
| **Stop Validation** | ❌ NOT PRESENT | ✅ Lines 120-142 | **CRITICAL MT5 constraint** |
| **MT5 Error Codes** | ❌ NOT PRESENT | ✅ Lines 25-33 | Simulates realistic rejections |

### ❌ Missing from RealisticBacktester (Should Port)

| Feature | BacktestEngine Location | Priority | Reason |
|---------|------------------------|----------|---------|
| **Risk-Based Position Sizing** | Lines 441-448 | 🔴 **HIGH** | Critical for realistic equity management |
| **Equity Curve Tracking** | Lines 216, 317-318, 623 | 🔴 **HIGH** | Needed for drawdown, Sharpe, etc. |
| **Sortino Ratio** | Lines 659-667 | 🟡 **MEDIUM** | Better than Sharpe for asymmetric returns |
| **CVaR (Tail Risk)** | Lines 670-677 | 🟡 **MEDIUM** | Measures downside tail risk |
| **Omega Ratio** | Lines 679-683 | 🟡 **MEDIUM** | Gain/loss threshold metric |
| **Z-Factor** | Lines 685-695 | 🟡 **MEDIUM** | Statistical edge metric |
| **Energy Captured %** | Lines 697-702 | 🟢 **LOW** | Physics-specific metric |
| **MFE Capture %** | Lines 704-707 | 🟡 **MEDIUM** | Trade execution efficiency |
| **Monte Carlo Validation** | Lines 738-801 | 🔴 **HIGH** | Robustness testing |
| **RL Agent Integration** | Lines 542-597, 364-367 | 🟡 **MEDIUM** | _build_agent_state() for RL |
| **Timeframe-Aware Annualization** | Lines 638-651 | 🔴 **HIGH** | Proper Sharpe scaling for different timeframes |
| **Mark-to-Market Tracking** | Lines 316-318, 528-540 | 🟡 **MEDIUM** | Live unrealized P&L |

### 🐛 Bugs in BacktestEngine (Fix When Porting)

| Bug | Location | Issue | Fix |
|-----|----------|-------|-----|
| **Missing `self.timeframe`** | Line 650 | References undefined attribute | Add to `__init__()` or pass as parameter |
| **Missing `self.min_margin_level`** | Line 728 | Never initialized or tracked | Remove or implement margin tracking |

## Architecture Differences

### BacktestEngine
- **Strengths**:
  - Comprehensive metrics (Sortino, CVaR, Omega, Z-factor)
  - Monte Carlo validation for robustness
  - RL agent integration
  - Physics-based signal generation
  - Risk-based position sizing

- **Weaknesses**:
  - NO freeze zone enforcement (sim-to-real gap!)
  - NO stop validation (sim-to-real gap!)
  - Fixed spread assumption (unrealistic)
  - Limited error simulation

### RealisticBacktester
- **Strengths**:
  - Enforces MT5 freeze zones
  - Validates stop distances
  - Dynamic per-candle spread
  - MT5 error code simulation
  - Regime-aware performance breakdown

- **Weaknesses**:
  - Missing advanced metrics (Sortino, CVaR, Omega)
  - No Monte Carlo validation
  - No RL agent integration
  - No risk-based position sizing
  - Basic Sharpe calculation (not timeframe-aware)

## Recommendations

### Priority 1: Critical Features (Port Immediately)

1. **Risk-Based Position Sizing** (`backtest_engine.py:441-448`)
   ```python
   # Calculate position size based on risk
   risk_amount = self.equity * self.risk_per_trade
   lots = min(
       risk_amount / (spec.spread_points * spec.tick_value * 2),
       spec.volume_max,
   )
   lots = max(lots, spec.volume_min)
   lots = round(lots / spec.volume_step) * spec.volume_step
   ```

2. **Equity Curve Tracking** (`backtest_engine.py:216, 317-318`)
   ```python
   self.equity_history: List[float] = [initial_capital]
   # After each bar
   mark_to_market = self._calculate_mtm(open_position, row["close"], spec)
   equity_value = self.equity + mark_to_market
   self.equity_history.append(equity_value)
   ```

3. **Timeframe-Aware Sharpe** (`backtest_engine.py:638-656`)
   ```python
   timeframe_bars_per_year = {
       "M1": 525600, "M5": 105120, "M15": 35040, "M30": 17520,
       "H1": 8760, "H4": 2190, "D1": 252, "W1": 52, "MN": 12,
   }
   bars_per_year = timeframe_bars_per_year.get(timeframe, 252)
   annualization = np.sqrt(bars_per_year)
   sharpe = (returns.mean() / returns.std()) * annualization
   ```

4. **Monte Carlo Validation** (`backtest_engine.py:738-801`)
   - Essential for assessing strategy robustness
   - Shuffle returns or bootstrap sample
   - Run backtest N times with shuffled data
   - Check if results hold across random permutations

### Priority 2: Important Metrics (Port Soon)

5. **Sortino Ratio** - Downside deviation (more relevant than Sharpe)
6. **CVaR (Conditional Value at Risk)** - Tail risk measurement
7. **Omega Ratio** - Gain/loss threshold metric
8. **MFE Capture %** - Trade execution efficiency

### Priority 3: Advanced Features (Port When Needed)

9. **RL Agent Integration** - For agent-based backtesting
10. **Z-Factor** - Statistical edge metric
11. **Energy Captured %** - Physics-specific metric

## Implementation Plan

### Phase 1: Enhance RealisticBacktester Metrics (Now)
- Add equity curve tracking
- Add risk-based position sizing
- Fix Sharpe ratio (timeframe-aware)
- Add Sortino, CVaR, Omega ratios
- Add MFE capture percentage

### Phase 2: Add Robustness Testing (Next)
- Port Monte Carlo validation
- Add return shuffling method
- Add bootstrap method

### Phase 3: Add Advanced Integration (Later)
- Port RL agent integration (_build_agent_state)
- Add Z-factor calculation
- Add energy-based metrics

## Key Insight: Complementary Strengths

**The ideal approach**: Use **BOTH** backtesters in sequence!

1. **Training Phase**: Use `TradingEnv` (fast RL training)
2. **Validation Phase 1**: Use `BacktestEngine` (comprehensive metrics)
3. **Validation Phase 2**: Use `RealisticBacktester` (MT5 constraints)
4. **Deployment**: Use `OrderExecutor` with live MT5 connection

This way you get:
- ✅ Comprehensive metrics (from BacktestEngine)
- ✅ Realistic constraints (from RealisticBacktester)
- ✅ Robustness validation (Monte Carlo)
- ✅ Regime-specific analysis (RealisticBacktester)

## Conclusion

**RealisticBacktester** is more aligned with the **live trading reality** (freeze zones, stop validation, dynamic spread), but **BacktestEngine** has superior **statistical analysis** (Sortino, CVaR, Monte Carlo).

**Solution**: Port key metrics from BacktestEngine into RealisticBacktester to create a unified, comprehensive, realistic backtesting engine.

This creates a **best-of-both-worlds** solution that prevents sim-to-real gap while providing deep statistical insights.
