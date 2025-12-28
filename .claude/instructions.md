# Kinetra AI Development Instructions

## CORE PHILOSOPHY

**EVERYTHING is derived from first principles. NO static rules.**

### What We DON'T Use
- NO traditional TA indicators (ATR, BB, RSI, MACD, etc.)
- NO hardcoded thresholds (no "if energy > 0.8")
- NO static values (no "volume spike > 1.5x")
- NO time-based filters
- NO magic numbers
- NO rules - only features for RL to discover patterns

### What We DO Use
- Physics (energy, damping, entropy, viscosity, Reynolds)
- Thermodynamics (energy states, phase transitions)
- Kinematics (velocity, acceleration, jerk, momentum)
- Rolling percentiles (adaptive to current distribution)
- Probability distributions (where in the distribution is current value?)
- First principles derivation

## ADAPTIVE PERCENTILES

Every metric should be converted to its position in the rolling distribution:

```python
# CORRECT: Adaptive percentile
feature_pct = feature.rolling(window).apply(
    lambda x: (x.iloc[-1] > x.iloc[:-1]).mean()
)
# Returns 0-1: where does current value sit in recent history?

# WRONG: Static threshold
if feature > 0.8:  # NO! This is not adaptive
```

## PHYSICS FIRST PRINCIPLES

| Concept | Formula | Meaning |
|---------|---------|---------|
| Kinetic Energy | E = ½mv² = ½ × velocity² | Energy in motion |
| Damping | ζ = σ(v) / μ(\|v\|) | Energy dissipation |
| Reynolds | Re = (v × L × ρ) / μ | Laminar vs turbulent |
| Viscosity | μ = resistance / flow | Market friction |
| Spring Stiffness | k = F / x = volume / Δprice | Resistance to displacement |
| Phase Space | (position, momentum) | State confinement |
| Entropy | S = disorder measure | Predictability |

## DYNAMIC REGIME DETECTION

Regimes are detected by WHERE current values sit in their distributions:

```python
# Compression = multiple physics metrics in extreme percentiles
phase_compressed = phase_compression_pct > 0.8  # Top 20%
high_suppression = suppression_ratio_pct > 0.7  # Top 30%
low_entropy = entropy_proxy_pct < 0.3          # Bottom 30%
```

But even these percentile cutoffs should ideally be learned, not fixed.

## SELF-HEALING / ADAPTIVE

The system must work across:
- Any instrument (BTCUSD, EURUSD, COPPER, etc.)
- Any timeframe (M15, H1, H4, D1)
- Any market regime
- Any volatility environment

This is achieved by:
1. All features as percentiles (0-1 range)
2. No instrument-specific parameters
3. RL discovers what works, not hardcoded rules

## WHAT RL DISCOVERS

The neural network learns:
1. WHEN energy is about to release (compression → trigger → acceleration)
2. WHEN NOT to trade (fragile regime - high friction)
3. WHICH direction (long vs short based on physics state)
4. WHEN to exit (MFE/MAE efficiency)

We provide FEATURES. RL finds PATTERNS.

## OMEGA REWARD (Pythagorean Path Efficiency)

The reward function must have NO static coefficients. We use Pythagorean geometry:

```python
# Goal: Maximum displacement from entry via shortest path
# Agent should exit as FAR from entry as possible (in right direction)
# but via the SHORTEST path (no yo-yo whipsaw)

# Total excursion = Pythagorean distance in MFE/MAE space
total_excursion = sqrt(MFE² + MAE²)

# Path efficiency = how direct was the path?
path_efficiency = |PnL| / total_excursion

# Omega = signed reward (direction matters)
omega = PnL × path_efficiency
```

Why this works:
- Clean move (MFE only): high omega (large PnL, small excursion)
- Whipsaw (high MFE + high MAE): low omega (excursion dominates)
- Loss with excursion: negative omega (penalty)

NO static weights, NO arbitrary coefficients - pure geometry.

## PROBABILITY PREDICTORS

Instead of rules, we compute probabilities:
- P(move up | high energy + acceleration up)
- P(continuation | low Reynolds + positive momentum)
- P(reversal | high entropy + extreme range position)

These are IMPLICIT in the neural network weights, not explicit rules.

## EXAMPLE: Energy Release

Wrong approach:
```python
if bb_width < 0.02 and volume > 2 * avg_volume:
    enter_trade()  # NO! Static rules
```

Correct approach:
```python
# Compute physics features
physics_compression = composite of phase/suppression/entropy/stiffness
was_compressed = physics_compression.rolling(5).max() == 1

# Let RL learn what to do with these features
state = [all physics percentiles + position info]
action = network.predict(state)  # RL decides
```

## FRICTION = VISCOSITY (Not Time)

Market friction comes from:
- Spread (from symbol_info)
- Low liquidity (volume percentile)
- High viscosity (price impact)

NOT from:
- Time of day
- Calendar events
- Session boundaries

## PARETO ANALYSIS: FAT CANDLES

"Fat candle" is RELATIVE to the instrument's distribution:
- What's fat for crypto differs from gold differs from forex
- Summer volatility differs from winter
- Use rolling percentile, NOT static threshold

```python
# "Fat" = top 20% of THIS instrument's recent distribution
candle_magnitude_pct = magnitude.rolling(500).apply(
    lambda x: (x.iloc[-1] > x.iloc[:-1]).mean()
)
is_fat_candle = candle_magnitude_pct >= 0.8  # Pareto: 20% → 80% of gains
```

Track physics state BEFORE fat candles:
- prior_phase_compression
- prior_suppression
- prior_entropy
- compression_buildup

RL learns: "When physics looks like X, fat candle probability increases"

## SEASONALITY / REGIME ADAPTATION

Use multiple rolling windows:
- Short window (20 bars): Current regime
- Long window (500 bars): Seasonal baseline

```python
vol_regime_ratio = short_vol / long_vol
# >1 = high vol regime (summer?), <1 = low vol regime (winter?)
```

This lets RL learn regime-dependent patterns WITHOUT hardcoded dates.

## SUMMARY

> "We don't trade rules. We provide physics state. RL discovers edges."

Every feature should be:
1. Derived from first principles (physics)
2. Expressed as a percentile (adaptive)
3. Instrument-agnostic (works everywhere)
4. Fed to RL for pattern discovery (not rules)
