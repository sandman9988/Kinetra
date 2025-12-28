# Kinetra AI Development Instructions

## CRITICAL: Physics-Only Approach

**DO NOT use traditional technical analysis indicators:**
- NO ATR (Average True Range)
- NO Bollinger Bands
- NO RSI, MACD, Stochastic
- NO Moving Average crossovers
- NO Fibonacci levels
- NO Support/Resistance lines
- NO time-of-day filters

**USE physics-derived metrics ONLY:**

### Core Physics Features
| Metric | Formula | Meaning |
|--------|---------|---------|
| Energy | 0.5 × velocity² | Kinetic energy of price movement |
| Damping | σ(velocity) / mean(|velocity|) | Energy dissipation rate |
| Entropy | σ / mean | Disorder/predictability of returns |
| Reynolds | (velocity × range × volume) / σ | Laminar vs turbulent flow |
| Viscosity | range / volume | Market friction |
| Torque | imbalance × acceleration | Rotational force on price |

### Compression Detection (NOT Bollinger Bands)
| Metric | Meaning |
|--------|---------|
| Phase Compression | Bounding box in (price, momentum) phase space |
| Suppression Ratio | Volume / Energy - hidden force absorption |
| Entropy Proxy | CV of returns - low = repetitive = coiled |
| Spring Stiffness | Volume / |velocity| - market resistance |

### Regime Filtering (NOT time-of-day)
| Metric | Meaning |
|--------|---------|
| Fragile Regime | Low liquidity + high spread + declining volume |
| Favorable Regime | Good liquidity + tight spread |
| Friction | Spread % from symbol_info |

## Philosophy

> "Bollinger Bands see a quiet market. Physics compression sees STORED POTENTIAL ENERGY."

The agent must DISCOVER patterns through RL, not be fed rules.
No hardcoded thresholds. All features as adaptive percentiles.

## Data Sources
- Friction: symbol_info (spread, swap, margin)
- Liquidity: volume percentile
- Energy: derived from price velocity

## What the Agent Learns
1. WHEN to enter: Energy release (compression → trigger → acceleration)
2. WHEN NOT to trade: Fragile regime (high friction)
3. HOW to exit: MFE/MAE efficiency
