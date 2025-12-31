# Testing Scripts

This directory contains the comprehensive testing framework for Kinetra.

## Main Testing Framework

**`unified_test_framework.py`** - The main testing interface

Runs scientific tests across multiple dimensions to discover alpha.

### Quick Start

```bash
# Quick test (10 minutes)
python scripts/unified_test_framework.py --quick

# Run specific suite
python scripts/unified_test_framework.py --suite control
python scripts/unified_test_framework.py --suite physics
python scripts/unified_test_framework.py --suite chaos

# Compare approaches
python scripts/unified_test_framework.py --compare control physics rl

# Full test suite
python scripts/unified_test_framework.py --full

# EXTREME mode - all 18 test suites
python scripts/unified_test_framework.py --extreme
```

## Test Suites Available

### Core Suites (6)
1. **control** - Standard indicators baseline (MA, RSI, MACD)
2. **physics** - First principles (energy, damping, entropy)
3. **rl** - Reinforcement learning (PPO, SAC, A2C)
4. **specialization** - Agent specialization strategies
5. **stacking** - Ensemble methods
6. **triad** - Incumbent/Competitor/Researcher

### Discovery Suites (12)
7. **hidden** - Hidden dimension discovery (autoencoders, PCA)
8. **meta** - Meta-learning (MAML)
9. **cross_regime** - Regime transition analysis
10. **cross_asset** - Transfer learning across assets
11. **mtf** - Multi-timeframe fusion
12. **emergent** - Emergent behavior (evolution strategies)
13. **adversarial** - Adversarial discovery (GAN-style)
14. **quantum** - Quantum-inspired superposition
15. **chaos** - Chaos theory (Lyapunov exponents)
16. **info_theory** - Information theory (entropy, causality)
17. **combinatorial** - Massive feature combinations
18. **deep_ensemble** - Stack everything

## Examples

**`example_testing_framework.py`** - Learn by example

```bash
python scripts/example_testing_framework.py
```

Shows:
- Minimal usage
- Comprehensive testing
- Custom efficiency metrics
- Statistical validation

## Legacy Test Scripts

These scripts are integrated into the unified framework but can still be run standalone:

- `explore_specialization.py` - Agent specialization exploration
- `train_triad.py` - Triad system training
- `superpot_by_class.py` - Asset class analysis
- `superpot_complete.py` - Complete exploration

## Configuration

Example configurations are in `configs/`:

- `example_test_config.yaml` - Full configuration example

## Output

Results are saved to `test_results/` (gitignored):

```
test_results/
├── test_results_YYYYMMDD_HHMMSS.json  # Raw results
├── plots/                              # Visualizations
│   ├── comparison_sharpe_ratio_*.png
│   ├── efficiency_metrics_*.png
│   └── gpu_utilization_*.png
└── reports/                            # Summary reports
```

## Documentation

See `docs/TESTING_FRAMEWORK.md` for comprehensive documentation.

## Philosophy

**"We don't know what we don't know, and we can't see what we can't even see"**

The framework explores:
- Traditional approaches (control group)
- First principles (physics)
- Machine learning (RL)
- Hidden dimensions (what we can't see)
- Emergent patterns (what we haven't thought of)
- Cross-domain relationships (what transcends markets)

By testing systematically across all dimensions, we discover alpha that would otherwise remain hidden.
