# SUPERPOT Empirical Testing

## Overview

**SUPERPOT** (SUPERior POTential) is a comprehensive empirical testing framework that implements the "throw all measurements in, prune worst" philosophy. It discovers which features matter through rigorous statistical testing rather than assumptions.

## Philosophy

### Core Principles

1. **NO Assumptions**: Don't assume what matters - test empirically
2. **Statistical Rigor**: All claims require p < 0.01 significance
3. **Adaptive Pruning**: Remove features based on statistical evidence, not fixed schedules
4. **Universal vs Specific**: Discover features that work universally vs those specific to asset classes or instruments
5. **Effect Size Matters**: Statistical significance alone isn't enough - require Cohen's d > 0.2

### What Makes SUPERPOT Different

Unlike traditional feature selection that relies on:
- Domain expert intuition
- Fixed feature sets (e.g., "RSI, MACD, Bollinger Bands")
- Arbitrary thresholds (e.g., "top 20 features")

SUPERPOT uses:
- **Empirical validation**: Let the market data decide
- **Statistical testing**: Spearman correlation (no linearity assumption)
- **Multiple testing correction**: Bonferroni for p-value adjustment
- **Effect size**: Cohen's d to ensure practical significance
- **Adaptive thresholds**: Prune based on significance, not arbitrary counts

## Architecture

### Components

```
UnifiedMeasurementExtractor
  ├── PhysicsExtractor (150+ features)
  │   ├── Kinematics (velocity, acceleration, jerk, snap, crackle, pop)
  │   ├── Energy (kinetic, potential, efficiency)
  │   ├── Flow Dynamics (Reynolds, damping, viscosity)
  │   ├── Thermodynamics (entropy, phase compression)
  │   ├── Field Theory (gradients, divergence, pressure)
  │   ├── Microstructure (spread, volume dynamics)
  │   ├── Cross-Interactions (composite signals)
  │   ├── Asymmetric Tails (directional risk)
  │   ├── Order Flow (CVD, imbalance, toxicity)
  │   └── Chaos/Complexity (Lyapunov, Hurst, fractal)
  │
  └── SuperPotExtractor (150+ features)
      ├── Price Action
      ├── Volume Dynamics
      ├── Volatility (multiple estimators)
      ├── Momentum
      ├── Entropy & Chaos
      ├── Tail Behavior
      ├── Microstructure
      ├── Higher Moments
      ├── Regime Indicators
      └── Cross-Feature Interactions

Total: 300+ measurements
```

### AdaptiveFeatureTracker

Tracks feature importance using:

1. **Spearman Correlation**: Rank-based correlation (no linearity assumption)
   - Measures monotonic relationship with reward
   
2. **Effect Size (Cohen's d)**: Practical significance
   ```
   d = (μ_win - μ_lose) / σ_pooled
   ```
   - Small: d = 0.2
   - Medium: d = 0.5
   - Large: d = 0.8

3. **Statistical Significance**: P-value with Bonferroni correction
   ```
   α_corrected = α / n_active_features
   ```
   - Default: α = 0.01 (1% false positive rate)
   - Bonferroni ensures family-wise error rate control

### Pruning Strategy

**Adaptive Pruning** removes features that are:

1. **Statistically insignificant**: p > 0.05 (after correction)
2. **Low effect size**: Cohen's d < 0.2
3. **Low importance score**: Combined metric < threshold

**Fallback**: If no features meet pruning criteria, remove bottom 10% by importance

**Safety**: Always keep minimum 20 features (configurable)

## Usage

### Basic Usage

```bash
# Default: 100 episodes, all asset classes
python scripts/analysis/superpot_empirical.py

# Quick test (50 episodes, fewer steps)
python scripts/analysis/superpot_empirical.py --quick

# Filter by asset class
python scripts/analysis/superpot_empirical.py --asset-class crypto

# Filter by timeframe
python scripts/analysis/superpot_empirical.py --timeframe H1

# Combined filters
python scripts/analysis/superpot_empirical.py --asset-class forex --timeframe H4
```

### Advanced Options

```bash
python scripts/analysis/superpot_empirical.py \
  --episodes 200 \
  --max-steps 1000 \
  --prune-every 25 \
  --max-files 50 \
  --asset-class crypto \
  --timeframe M15
```

**Parameters**:
- `--episodes`: Total training episodes (default: 100)
- `--max-steps`: Max steps per episode (default: 500)
- `--prune-every`: Prune features every N episodes (default: 20)
- `--max-files`: Max data files to use (default: 30)
- `--asset-class`: Filter by asset class (crypto, forex, metals, commodities, indices)
- `--timeframe`: Filter by timeframe (M15, M30, H1, H4, D1)
- `--quick`: Quick test mode (50 episodes, 200 steps, 10 files)

## Output

### Console Output

```
╔══════════════════════════════════════════════════════════════════════╗
║                  SUPERPOT EMPIRICAL TESTING                          ║
║     Execute comprehensive empirical testing with ALL measurements    ║
║              Prune worst, discover what matters (p < 0.01)           ║
╚══════════════════════════════════════════════════════════════════════╝

🔍 Discovering data files...
📁 Using 10 data files

📊 Asset classes: {'crypto': 10}
⏱️  Timeframes: {'M30': 2, 'H4': 3, 'M15': 4, 'H1': 1}

🔧 Initializing measurement extractors...
📊 UnifiedMeasurementExtractor initialized with 23 measurements

🧪 Starting empirical testing:
   Features: 23
   Episodes: 50
   Adaptive pruning every 20 episodes
   Statistical validation: p < 0.01 (Bonferroni corrected)

Ep  10: R=   +1.3 PnL=$    +14 | Active: 23/23 | ε=1.000
Ep  20: R=   -0.9 PnL=$     -8 | Active: 23/23 | ε=1.000

🗑️  PRUNED 3 features:
   - feature_xyz
   - feature_abc
   - feature_def
   Remaining: 20 features

Ep  30: R=   -5.4 PnL=$    -53 | Active: 20/23 | ε=0.950

======================================================================
SUPERPOT EMPIRICAL TESTING RESULTS
======================================================================

📊 Performance Metrics:
   Episodes completed: 50
   Avg reward: -1.10 ± 6.20
   Avg PnL: $-10 ± $62
   Win rate: 44.0%
   Time: 17.1s

🔬 Feature Discovery:
   Initial features: 23
   Surviving features: 20
   Pruned: 3 (13.0%)

⭐ STATISTICALLY SIGNIFICANT FEATURES (p < 0.01, Bonferroni corrected):
   Found: 2

   1. volume_ratio_10                        p=0.000123 d=0.456
   2. energy_momentum_product_5              p=0.000891 d=0.389

🏆 TOP FEATURES BY IMPORTANCE SCORE:
    1. volume_ratio_10                          score=0.0176 p=0.0001 d=0.456 ***
    2. energy_momentum_product_5                score=0.0141 p=0.0009 d=0.389 ***
    3. right_tail_5                             score=0.0123 p=0.0173 d=0.250 *
    ...

💾 Results saved: results/superpot/empirical_20260101_111113.json
```

### JSON Output

Results are saved to `results/superpot/empirical_{timestamp}.json`:

```json
{
  "timestamp": "20260101_111113",
  "config": {
    "episodes": 50,
    "max_steps": 200,
    "prune_every": 20,
    "asset_class": "crypto",
    "timeframe": null
  },
  "metrics": {
    "episodes": 50,
    "avg_reward": -1.10,
    "std_reward": 6.20,
    "avg_pnl": -10.0,
    "std_pnl": 62.0,
    "win_rate": 0.44,
    "total_time": 17.1
  },
  "features": {
    "initial": 23,
    "surviving": 20,
    "pruned": 3
  },
  "top_features": [
    {
      "name": "volume_ratio_10",
      "score": 0.0176,
      "p_value": 0.0001,
      "effect_size": 0.456
    },
    ...
  ],
  "statistically_significant": [
    {
      "name": "volume_ratio_10",
      "p_value": 0.000123,
      "effect_size": 0.456
    },
    ...
  ]
}
```

## Interpreting Results

### Statistical Markers

- `***`: p < 0.001 (highly significant)
- `**`: p < 0.01 (significant)
- `*`: p < 0.05 (marginally significant)
- (none): p ≥ 0.05 (not significant)

### Effect Size Guidelines

| Cohen's d | Interpretation | Practical Significance |
|-----------|----------------|------------------------|
| < 0.2 | Negligible | Not practically useful |
| 0.2 - 0.5 | Small | Minimal practical impact |
| 0.5 - 0.8 | Medium | Moderate practical impact |
| > 0.8 | Large | Strong practical impact |

### Feature Categories

Results will reveal:

1. **Universal Features**: Survive across all asset classes and timeframes
   - These are fundamental market dynamics
   - Should be included in all models

2. **Class-Specific Features**: Only significant for certain asset classes
   - e.g., "crypto features" vs "forex features"
   - Suggests specialization strategy

3. **Instrument-Specific Features**: Only work for individual instruments
   - Indicates overfitting risk
   - May not generalize

4. **Timeframe-Specific Features**: Vary by trading horizon
   - Scalping features ≠ swing trading features
   - Suggests different feature sets per timeframe

## Workflow

### 1. Broad Discovery

Start with all measurements, all asset classes:

```bash
python scripts/analysis/superpot_empirical.py \
  --episodes 200 \
  --max-files 50
```

**Discover**: Which features are universally important?

### 2. Class-Specific Testing

Test each asset class separately:

```bash
for class in crypto forex metals commodities indices; do
  python scripts/analysis/superpot_empirical.py \
    --asset-class $class \
    --episodes 200
done
```

**Discover**: Which features are class-specific?

### 3. Timeframe-Specific Testing

Test each timeframe separately:

```bash
for tf in M15 M30 H1 H4 D1; do
  python scripts/analysis/superpot_empirical.py \
    --timeframe $tf \
    --episodes 200
done
```

**Discover**: Which features matter per timeframe?

### 4. Cross-Validation

Test on held-out data to validate findings:

```bash
python scripts/analysis/superpot_empirical.py \
  --episodes 100 \
  --max-files 20  # Different files than training
```

**Validate**: Do significant features generalize?

### 5. Theorem Formulation

Document findings that meet criteria:
- p < 0.01 (after Bonferroni correction)
- Cohen's d > 0.5 (medium effect)
- Validated out-of-sample
- Reproducible across runs

Add to `docs/EMPIRICAL_THEOREMS.md`:

```markdown
## Theorem: Volume Ratio Predictive Power (Crypto, H1)

**Statement**: Volume ratio over 10 periods significantly predicts profitability 
in crypto markets on H1 timeframe.

**Evidence**:
- p-value: 0.000123 (Bonferroni corrected, α = 0.01)
- Effect size: d = 0.456 (small-to-medium)
- Sample size: n = 200 episodes, 50,000+ observations
- Out-of-sample validation: p = 0.000567, d = 0.412

**Implication**: Include volume_ratio_10 in all crypto H1 models.
```

## Best Practices

### 1. Sample Size

- Minimum: 100 episodes for initial discovery
- Recommended: 200+ episodes for reliable results
- Statistical power: Need >30 episodes for effect detection

### 2. Multiple Testing

Always apply Bonferroni correction:
```
α_effective = 0.01 / n_features
```

With 300 features: α_effective = 0.00003 (very conservative)

### 3. Avoid Data Leakage

- Use different files for cross-validation
- Don't re-use same data for hypothesis generation and testing
- Report both in-sample and out-of-sample results

### 4. Document Everything

- Save all JSON outputs
- Record configuration parameters
- Note any anomalies or issues
- Track which data files were used

### 5. Reproducibility

- Use random seed for reproducibility
- Save full configuration
- Document software versions
- Archive data snapshots

## Common Pitfalls

### ❌ Don't Do This

1. **P-hacking**: Running multiple tests and reporting only significant ones
   - Solution: Pre-register hypotheses or use Bonferroni correction

2. **Cherry-picking**: Only reporting best-performing features
   - Solution: Report all results, including null findings

3. **Overfitting**: Using same data for discovery and validation
   - Solution: Always validate on held-out data

4. **Ignoring effect size**: Reporting p < 0.01 with d = 0.1
   - Solution: Require both significance AND meaningful effect size

5. **Fixed thresholds**: "Remove bottom 20 features"
   - Solution: Use statistical criteria for pruning

### ✅ Do This

1. **Pre-register**: Decide criteria before testing
2. **Correct for multiple testing**: Use Bonferroni or FDR
3. **Report effect sizes**: Cohen's d alongside p-values
4. **Validate out-of-sample**: Test on different data
5. **Document negative results**: "Feature X was NOT significant"

## Integration with Kinetra

### Feature Selection for Models

Use SUPERPOT results to:

1. **Initialize feature sets**: Start with statistically significant features
2. **Prune models**: Remove features with p > 0.05
3. **Asset class models**: Use class-specific features
4. **Timeframe optimization**: Adapt features to trading horizon

### Example Integration

```python
from scripts.analysis.superpot_empirical import AdaptiveFeatureTracker

# Load SUPERPOT results
with open('results/superpot/empirical_crypto_h1.json') as f:
    results = json.load(f)

# Get significant features (p < 0.01)
significant_features = [
    f['name'] for f in results['statistically_significant']
]

# Use in model initialization
model = TradingModel(features=significant_features)
```

## Future Enhancements

### Planned Features

1. **Hierarchical Testing**: Test universal → class → instrument in sequence
2. **Feature Interactions**: Test combinations (e.g., "volume × momentum")
3. **Regime-Conditional**: Different features for different market regimes
4. **Temporal Stability**: Track feature importance over time
5. **Cross-Instrument Validation**: Test on similar instruments

### Research Questions

1. Are there truly universal features? Or is everything context-dependent?
2. What's the optimal feature set size? 20? 50? 100?
3. Do feature importances change over time (non-stationarity)?
4. Can we predict which features will matter based on market conditions?
5. What's the relationship between feature count and overfitting?

## References

### Statistical Methods

- **Spearman Correlation**: Non-parametric rank correlation (no linearity assumption)
- **Cohen's d**: Standardized effect size measure
- **Bonferroni Correction**: Conservative multiple testing correction
- **FDR (Benjamini-Hochberg)**: Less conservative alternative (future)

### Related Work

- `scripts/analysis/superpot_explorer.py` - Original SuperPot implementation
- `scripts/analysis/superpot_physics.py` - Physics-based measurements
- `scripts/analysis/superpot_complete.py` - Complete cross-testing
- `scripts/analysis/superpot_by_class.py` - Asset class specific

### Documentation

- `AI_AGENT_INSTRUCTIONS.md` - Keyword: SUPERPOT
- `docs/EMPIRICAL_THEOREMS.md` - Validated theorems (p < 0.01)
- `docs/ARCHITECTURE_COMPLETE.md` - System architecture

---

**Remember**: The goal is DISCOVERY, not confirmation. Let the data surprise you.
