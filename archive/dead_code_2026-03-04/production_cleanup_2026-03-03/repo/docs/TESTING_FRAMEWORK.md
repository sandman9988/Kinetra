# Comprehensive Testing Framework Documentation

## Overview

The Kinetra Testing Framework is a rigorous, scientific testing system designed to discover alpha through first principles while exploring dimensions we don't even know exist yet.

**Core Philosophy: "We don't know what we don't know, and we can't see what we can't even see"**

## Key Features

### 1. Scientific Rigor
- **Control Groups**: Standard indicators (MA, RSI, MACD, BB) as baseline
- **Statistical Validation**: Only keeps statistically significant results
- **Multiple Testing Correction**: Bonferroni and FDR corrections
- **Effect Size Calculation**: Cohen's d for practical significance

### 2. First Principles Adherence
- **No Magic Numbers**: All thresholds are adaptive/rolling percentiles
- **Non-linearity**: Tests for and enforces non-linear relationships
- **Asymmetry**: Long and short treated differently
- **No Fixed Periods**: Everything adapts to market regime

### 3. Efficiency Metrics
- **MFE/MAE Analysis**: Maximum Favorable/Adverse Excursion
- **Pythagorean Efficiency**: Shortest path vs actual path to profit
- **Trailing Stop Optimization**: Dynamic stop mechanisms
- **Path Efficiency**: Measures directness of profit capture

### 4. GPU Acceleration
- **Full GPU Utilization**: Monitors and maximizes GPU usage
- **ROCm/CUDA Support**: Works with both AMD and NVIDIA
- **Parallel Execution**: Multiple tests in parallel

### 5. Exploration of Unknown Dimensions

The framework goes beyond traditional testing to explore what we might be missing:

#### Hidden Dimension Discovery
- Autoencoders to find latent features
- PCA, t-SNE, UMAP for dimensionality reduction
- ICA for independent component analysis

#### Meta-Learning
- MAML (Model-Agnostic Meta-Learning)
- Learn which features matter across contexts
- Automatic feature selection

#### Cross-Domain Analysis
- **Cross-Regime**: Study regime transitions
- **Cross-Asset**: Transfer learning between asset classes
- **Multi-Timeframe**: Fuse signals across timeframes

#### Emergent Behavior
- Evolution Strategies (ES)
- Genetic algorithms
- Swarm intelligence
- Discover behaviors we wouldn't think to program

#### Adversarial Discovery
- GAN-style approach: Generator finds patterns, Discriminator validates
- Filters noise, keeps only real alpha
- Statistical validation at p < 0.01

#### Quantum-Inspired
- Superposition of strategies
- Collapse to best based on observation
- Correlation-based entanglement

#### Chaos Theory
- Lyapunov exponents
- Strange attractors
- Fractal dimensions
- Find order in apparent randomness

#### Information Theory
- Mutual information
- Transfer entropy
- Causality detection
- Information flow between instruments

#### Combinatorial Explosion
- Test massive feature combinations
- Pairs, triplets, quadruplets, etc.
- GPU-accelerated evaluation
- Statistical pruning

## Test Suites

### Core Suites

1. **Control**: Standard indicators baseline
2. **Physics**: Energy, damping, entropy (first principles)
3. **RL**: PPO, SAC, A2C reinforcement learning
4. **Specialization**: By asset class, regime, or timeframe
5. **Stacking**: Ensemble of multiple models
6. **Triad**: Incumbent/Competitor/Researcher system

### Advanced Suites

7. **Hidden Dimension Discovery**: Find latent features
8. **Meta-Learning**: Learn to learn
9. **Cross-Regime**: Study transitions
10. **Cross-Asset**: Transfer learning
11. **Multi-Timeframe Fusion**: Temporal patterns
12. **Emergent Behavior**: Evolutionary discovery
13. **Adversarial Discovery**: GAN-style validation
14. **Quantum-Inspired**: Strategy superposition
15. **Chaos Theory**: Deterministic patterns
16. **Information Theory**: Information flow
17. **Combinatorial Explosion**: Massive combinations
18. **Deep Ensemble**: Stack everything

## Usage

### Quick Test (10 minutes)
```bash
python scripts/unified_test_framework.py --quick
```

### Full Test Suite (standard tests)
```bash
python scripts/unified_test_framework.py --full
```

### EXTREME Mode (ALL combinations)
```bash
python scripts/unified_test_framework.py --extreme
```

### Specific Suite
```bash
# Traditional tests
python scripts/unified_test_framework.py --suite control
python scripts/unified_test_framework.py --suite physics
python scripts/unified_test_framework.py --suite rl

# Advanced discovery
python scripts/unified_test_framework.py --suite hidden
python scripts/unified_test_framework.py --suite meta
python scripts/unified_test_framework.py --suite chaos
python scripts/unified_test_framework.py --suite quantum
python scripts/unified_test_framework.py --suite adversarial
python scripts/unified_test_framework.py --suite combinatorial
```

### Compare Multiple Suites
```bash
python scripts/unified_test_framework.py --compare control physics rl hidden chaos
```

### Filter by Asset Class or Timeframe
```bash
# Only crypto and forex
python scripts/unified_test_framework.py --full --asset-classes crypto forex

# Only H1 and H4 timeframes
python scripts/unified_test_framework.py --full --timeframes H1 H4

# Combine filters
python scripts/unified_test_framework.py --extreme \
    --asset-classes crypto \
    --timeframes H1 H4 \
    --max-instruments 2
```

### Custom Episodes
```bash
python scripts/unified_test_framework.py --full --episodes 50
```

## Output

### Results Directory Structure
```
test_results/
├── test_results_YYYYMMDD_HHMMSS.json    # Raw results
├── plots/
│   ├── comparison_sharpe_ratio_*.png
│   ├── comparison_omega_ratio_*.png
│   ├── efficiency_metrics_*.png
│   ├── gpu_utilization_*.png
│   └── first_principles_*.png
└── reports/
    └── summary_YYYYMMDD_HHMMSS.txt
```

### Plots Generated

1. **Performance Comparison**: Box plots of Sharpe, Omega, drawdown
2. **Efficiency Metrics**: MFE captured and Pythagorean efficiency
3. **GPU Utilization**: Average GPU usage per test
4. **First Principles Validation**: Non-linearity, asymmetry, no magic numbers

### Metrics Tracked

#### Performance
- Total Return
- Sharpe Ratio
- Omega Ratio
- Maximum Drawdown
- Win Rate
- Number of Trades

#### Efficiency
- MFE Captured %
- MAE Ratio
- Pythagorean Efficiency
- Average Trade Duration

#### First Principles
- Is Non-Linear (boolean)
- Is Asymmetric (boolean)
- Uses Magic Numbers (boolean - should be False)

#### System
- GPU Utilization %
- Statistical Significance
- P-values
- Effect Sizes

## Statistical Validation

All results are filtered for statistical significance:

- **Minimum Sample Size**: 30 observations
- **Significance Level**: α = 0.05
- **Multiple Testing Correction**: Bonferroni or FDR
- **Effect Size**: Cohen's d calculated for all comparisons

Only results that pass statistical tests are kept and reported.

## Integration with Existing Scripts

The framework integrates with existing test scripts:

- `explore_specialization.py` → Specialization suites
- `train_triad.py` → Triad suites
- `superpot_by_class.py` → Asset class analysis
- `superpot_complete.py` → Complete exploration

But provides a unified interface and rigorous statistical validation.

## GPU Optimization

The framework monitors and optimizes GPU usage:

1. **Auto-detection**: Detects ROCm (AMD) or CUDA (NVIDIA)
2. **Utilization Monitoring**: Samples GPU usage during tests
3. **Target**: Aims for 80%+ GPU utilization
4. **Reporting**: Shows average GPU usage in plots

## Philosophy: Unknown Unknowns

Traditional testing asks: "Does strategy X work?"

This framework asks:
1. "What strategies exist that we haven't thought of?" (Emergent)
2. "What features matter that we can't see?" (Hidden Dimensions)
3. "What relationships exist between markets?" (Cross-Asset, Info Theory)
4. "What happens during transitions?" (Cross-Regime)
5. "Are markets random or chaotic?" (Chaos Theory)
6. "What's the optimal combination of everything?" (Deep Ensemble)

By exploring these questions systematically, we discover alpha that would otherwise remain hidden.

## Roadmap

### Phase 1 (Current): Framework Foundation
- [x] Core testing infrastructure
- [x] Statistical validation
- [x] Efficiency metrics
- [x] GPU monitoring
- [x] Advanced test suites
- [x] Visualization

### Phase 2: Implementation Integration
- [ ] Connect to actual RL training loops
- [ ] Implement hidden dimension discovery
- [ ] Implement meta-learning algorithms
- [ ] Implement chaos theory analysis
- [ ] Implement information theory metrics

### Phase 3: Live Trading Evolution
- [ ] Real-time testing
- [ ] Continuous adaptation
- [ ] Competition between strategies
- [ ] Auto-deployment of winners
- [ ] Replay learning integration

### Phase 4: Trading Ecosystem
- [ ] Multi-agent competition
- [ ] Strategy marketplace
- [ ] Alpha discovery network
- [ ] Self-healing mechanisms
- [ ] Full autonomy

## Best Practices

1. **Start Small**: Use `--quick` for testing
2. **Control First**: Always run control group for baseline
3. **Statistical Rigor**: Don't trust single runs - need statistical significance
4. **GPU Check**: Monitor GPU utilization - should be 80%+
5. **Document**: Keep notes on what works and why
6. **Iterate**: Use results to inform next tests
7. **Extreme Mode Carefully**: `--extreme` runs 20+ test suites - takes hours/days

## Troubleshooting

### No Instruments Found
```bash
# Check data directory
ls -la data/master/
ls -la data/

# Specify custom data directory in code
```

### Low GPU Utilization
- Check GPU drivers: `rocm-smi` or `nvidia-smi`
- Ensure PyTorch sees GPU: `python -c "import torch; print(torch.cuda.is_available())"`
- Increase batch size or parallel tests

### Tests Running Too Long
- Reduce `--episodes`
- Use `--quick` mode
- Limit `--max-instruments`
- Filter by `--asset-classes` or `--timeframes`

## Examples

### Example 1: Quick Validation
```bash
# 10-minute test to validate setup
python scripts/unified_test_framework.py --quick
```

### Example 2: Control vs Physics
```bash
# Compare traditional vs first principles
python scripts/unified_test_framework.py --compare control physics
```

### Example 3: Discover Hidden Features
```bash
# What features exist that we can't see?
python scripts/unified_test_framework.py --suite hidden --episodes 200
```

### Example 4: Chaos vs Random
```bash
# Are markets chaotic or random?
python scripts/unified_test_framework.py --suite chaos
```

### Example 5: Full Scientific Study
```bash
# Comprehensive scientific analysis
python scripts/unified_test_framework.py --extreme \
    --asset-classes crypto forex metals \
    --timeframes H1 H4 D1 \
    --max-instruments 3 \
    --output-dir results/study_$(date +%Y%m%d)
```

## Citation

If you use this framework in research, please cite:

```
Kinetra Testing Framework (2025)
A comprehensive, first-principles testing system for algorithmic trading
https://github.com/sandman9988/Kinetra
```

## License

MIT License - see LICENSE file

## Contact

For questions, issues, or contributions:
- GitHub Issues: https://github.com/sandman9988/Kinetra/issues
- Discussions: https://github.com/sandman9988/Kinetra/discussions
