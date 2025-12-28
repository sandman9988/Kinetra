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

## SUMMARY

> "We don't trade rules. We provide physics state. RL discovers edges."

Every feature should be:
1. Derived from first principles (physics)
2. Expressed as a percentile (adaptive)
3. Instrument-agnostic (works everywhere)
4. Fed to RL for pattern discovery (not rules)
