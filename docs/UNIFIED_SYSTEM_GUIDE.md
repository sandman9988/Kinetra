# Unified Testing, Data, and Backtesting System - Complete Guide

## Overview

This document provides a comprehensive guide to the fully integrated testing system that consolidates:
1. All existing test scripts
2. Data download and preparation
3. Discovery testing (hidden dimensions, chaos theory, etc.)
4. Automatic backtesting of discovered strategies
5. Statistical validation and reporting

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    UNIFIED TESTING SYSTEM                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌────────────────────────────────────────────────────────┐    │
│  │  1. DATA MANAGEMENT (UnifiedDataManager)               │    │
│  │  - Download data (MetaAPI, MT5, CSV)                   │    │
│  │  - Validate integrity and quality                      │    │
│  │  - Prepare instruments for testing                     │    │
│  └────────────────────────────────────────────────────────┘    │
│                            │                                     │
│                            ▼                                     │
│  ┌────────────────────────────────────────────────────────┐    │
│  │  2. TESTING FRAMEWORK (18 Test Suites)                 │    │
│  │  Core (6):                                              │    │
│  │  - Control (MA, RSI, etc)                              │    │
│  │  - Physics (energy, damping, entropy)                  │    │
│  │  - RL (PPO, SAC, A2C)                                  │    │
│  │  - Specialization (asset/regime/timeframe)             │    │
│  │  - Stacking (ensemble)                                  │    │
│  │  - Triad (incumbent/competitor/researcher)              │    │
│  │                                                          │    │
│  │  Discovery (12):                                        │    │
│  │  - Hidden Dimensions (autoencoders, PCA)               │    │
│  │  - Meta-Learning (MAML)                                │    │
│  │  - Cross-Regime (transitions)                          │    │
│  │  - Cross-Asset (transfer learning)                     │    │
│  │  - Multi-Timeframe Fusion                              │    │
│  │  - Emergent Behavior (evolution)                       │    │
│  │  - Adversarial Discovery (GAN)                         │    │
│  │  - Quantum-Inspired (superposition)                    │    │
│  │  - Chaos Theory (Lyapunov)                             │    │
│  │  - Information Theory (entropy)                        │    │
│  │  - Combinatorial Explosion                             │    │
│  │  - Deep Ensemble (all combined)                        │    │
│  └────────────────────────────────────────────────────────┘    │
│                            │                                     │
│                            ▼                                     │
│  ┌────────────────────────────────────────────────────────┐    │
│  │  3. INTEGRATED BACKTESTER                               │    │
│  │  - Convert discoveries to tradeable strategies         │    │
│  │  - Run realistic backtests                             │    │
│  │  - Calculate comprehensive metrics                     │    │
│  │  - Statistical validation                              │    │
│  │  - Generate reports                                    │    │
│  └────────────────────────────────────────────────────────┘    │
│                            │                                     │
│                            ▼                                     │
│  ┌────────────────────────────────────────────────────────┐    │
│  │  4. RESULTS & VISUALIZATION                             │    │
│  │  - Statistical filtering                               │    │
│  │  - Performance plots                                   │    │
│  │  - Efficiency metrics                                  │    │
│  │  - Backtest reports                                    │    │
│  └────────────────────────────────────────────────────────┘    │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Quick Start

### 1. Minimal Example (1 minute)
```bash
# Quick test with 2 instruments
python scripts/unified_test_framework.py --quick
```

### 2. Standard Workflow (15-30 minutes)
```bash
# Full test suite with data validation
python scripts/unified_test_framework.py \
    --full \
    --validate-data \
    --asset-classes crypto forex \
    --timeframes H1 H4
```

### 3. Complete Scientific Study (hours/days)
```bash
# EXTREME mode: All 18 test suites + backtesting
python scripts/unified_test_framework.py \
    --extreme \
    --validate-data \
    --backtest \
    --asset-classes crypto forex metals \
    --timeframes M15 H1 H4 D1 \
    --max-instruments 5 \
    --min-quality 0.8 \
    --output-dir results/full_study_$(date +%Y%m%d)
```

## Complete Features List

### Data Management
- ✅ Automatic instrument discovery
- ✅ Data quality validation (integrity checks)
- ✅ Quality scoring (0-1 scale)
- ✅ Multiple source support (MetaAPI, MT5, CSV)
- ✅ Atomic operations with rollback
- ✅ Gap detection
- ✅ Asset classification

### Testing Framework
- ✅ 18 test suites (6 core + 12 discovery)
- ✅ Statistical validation (Bonferroni, FDR)
- ✅ First principles validation (non-linear, asymmetric, no magic numbers)
- ✅ GPU monitoring and optimization
- ✅ Efficiency metrics (MFE/MAE, Pythagorean)
- ✅ Same instruments across all tests ("apples to apples")
- ✅ Control group (standard indicators baseline)

### Discovery Methods
- ✅ Hidden dimensions (PCA, autoencoders, UMAP)
- ✅ Meta-learning (learn what works)
- ✅ Cross-regime analysis (transitions)
- ✅ Cross-asset transfer learning
- ✅ Multi-timeframe fusion
- ✅ Emergent behavior (evolution strategies)
- ✅ Adversarial discovery (GAN-style)
- ✅ Quantum-inspired superposition
- ✅ Chaos theory analysis
- ✅ Information theory (causality detection)
- ✅ Combinatorial feature testing
- ✅ Deep ensemble stacking

### Backtesting
- ✅ Automatic strategy conversion from discoveries
- ✅ Realistic cost modeling (spread, commission, slippage)
- ✅ Risk management (stops, position sizing)
- ✅ Comprehensive metrics (Sharpe, Omega, Sortino, Calmar)
- ✅ Statistical validation of backtest results
- ✅ MFE/MAE analysis
- ✅ Trade-by-trade tracking

### Reporting & Visualization
- ✅ Performance comparison plots
- ✅ Efficiency metrics plots
- ✅ GPU utilization monitoring
- ✅ First principles validation charts
- ✅ Statistical significance filtering
- ✅ JSON and text reports
- ✅ Backtest reports

## Command Reference

### Basic Commands
```bash
# Quick test
python scripts/unified_test_framework.py --quick

# Full suite
python scripts/unified_test_framework.py --full

# EXTREME mode (all combinations)
python scripts/unified_test_framework.py --extreme
```

### Specific Test Suites
```bash
# Run control group
python scripts/unified_test_framework.py --suite control

# Run physics
python scripts/unified_test_framework.py --suite physics

# Run discovery suites
python scripts/unified_test_framework.py --suite hidden
python scripts/unified_test_framework.py --suite chaos
python scripts/unified_test_framework.py --suite quantum
python scripts/unified_test_framework.py --suite meta
```

### Comparison Studies
```bash
# Compare control vs physics vs chaos
python scripts/unified_test_framework.py --compare control physics chaos

# Compare all discovery methods
python scripts/unified_test_framework.py --compare hidden meta chaos quantum
```

### Data Management
```bash
# Validate data before testing
python scripts/unified_test_framework.py --full --validate-data

# Set minimum quality threshold
python scripts/unified_test_framework.py --full --min-quality 0.85

# Auto-download missing data (requires configuration)
python scripts/unified_test_framework.py --full --auto-download
```

### Backtesting
```bash
# Test and backtest discoveries
python scripts/unified_test_framework.py --suite chaos --backtest

# Full workflow with backtesting
python scripts/unified_test_framework.py --extreme --backtest
```

### Filtering
```bash
# Only crypto
python scripts/unified_test_framework.py --full --asset-classes crypto

# Only H1 and H4
python scripts/unified_test_framework.py --full --timeframes H1 H4

# Combine filters
python scripts/unified_test_framework.py --full \
    --asset-classes crypto forex \
    --timeframes H1 H4 \
    --max-instruments 2
```

### Custom Episodes
```bash
# Override episode count
python scripts/unified_test_framework.py --full --episodes 50
```

## Output Structure

```
test_results/
├── test_results_20251231_205300.json     # Raw test results
│
├── plots/                                 # Visualizations
│   ├── comparison_sharpe_ratio_*.png
│   ├── comparison_omega_ratio_*.png
│   ├── efficiency_metrics_*.png
│   ├── gpu_utilization_*.png
│   └── first_principles_*.png
│
├── backtests/                             # Backtest results
│   ├── chaos_strategy_*.json
│   ├── hidden_dimension_strategy_*.json
│   └── backtest_report_*.txt
│
└── metadata/                              # Data integrity reports
    ├── BTCUSD_H1_integrity.json
    └── EURUSD_H1_integrity.json
```

## Programmatic Usage

### Example: Custom Test
```python
from kinetra.testing_framework import TestingFramework, TestConfiguration, InstrumentSpec
from kinetra.unified_data_manager import quick_setup

# Setup data
manager = quick_setup()
instruments = manager.prepare_for_testing(
    asset_classes=['crypto'],
    timeframes=['H1'],
    max_per_class=2
)

# Create framework
framework = TestingFramework()

# Add custom test
config = TestConfiguration(
    name="my_custom_test",
    description="Custom chaos theory test",
    instruments=instruments,
    agent_type="chaos",
    agent_config={"lyapunov_threshold": 0.5},
    episodes=100,
    use_gpu=True
)
framework.add_test(config)

# Run
results = framework.run_all_tests()
framework.generate_report()
```

### Example: Backtest Discovery
```python
from kinetra.integrated_backtester import IntegratedBacktester
import pandas as pd

# Initialize
backtester = IntegratedBacktester()

# Discovered strategy configuration
strategy_config = {
    'type': 'chaos_theory',
    'features': ['lyapunov_exponent', 'fractal_dimension'],
    'thresholds': {'lyapunov_entry': 0.5}
}

# Load data
data = pd.read_csv('data/master/BTCUSD_H1.csv')

# Backtest
result = backtester.backtest_discovered_strategy(strategy_config, data)

print(f"Total Return: {result.total_return:.2%}")
print(f"Sharpe Ratio: {result.sharpe_ratio:.2f}")
print(f"Win Rate: {result.win_rate:.2%}")
```

## Integration with Existing Code

The unified system integrates with existing Kinetra components:

### Data Sources
- `scripts/download_interactive.py` → `UnifiedDataManager`
- `scripts/prepare_data.py` → `UnifiedDataManager.prepare_for_testing()`
- `kinetra/data_manager.py` → `UnifiedDataManager`

### Test Scripts
- `scripts/explore_specialization.py` → Specialization suites
- `scripts/train_triad.py` → Triad suites
- `scripts/superpot_by_class.py` → Asset class testing
- `scripts/superpot_complete.py` → Complete exploration

### Backtesting
- `scripts/run_comprehensive_backtest.py` → `IntegratedBacktester`
- `kinetra/backtest_engine.py` → Used by `IntegratedBacktester`

## Performance Tips

### GPU Optimization
```bash
# Monitor GPU usage
python scripts/unified_test_framework.py --full --validate-data

# Check results for GPU utilization
# Target: 80%+ GPU usage
```

### Parallel Execution
The framework automatically uses available CPU cores for:
- Data validation
- Monte Carlo simulations
- Parallel test execution

### Memory Management
For large datasets:
```bash
# Reduce instruments per class
python scripts/unified_test_framework.py --extreme --max-instruments 2

# Reduce episodes
python scripts/unified_test_framework.py --extreme --episodes 50
```

## Troubleshooting

### No Instruments Found
```bash
# Check data directory
ls -la data/master/

# Validate data
python scripts/unified_test_framework.py --quick --validate-data

# Use data manager directly
python -c "from kinetra.unified_data_manager import quick_setup; m = quick_setup(); m.print_summary()"
```

### Low GPU Utilization
- Increase batch size in RL configs
- Use `--extreme` mode for more parallel tests
- Check `nvidia-smi` or `rocm-smi`

### Statistical Insignificance
- Increase `--episodes` for more data
- Use `--full` or `--extreme` for more test combinations
- Check `min_sample_size` in test configurations

## Philosophy: Unknown Unknowns

**"We don't know what we don't know, and we can't see what we can't even see"**

Traditional testing:
- ❌ Tests predefined strategies
- ❌ Uses fixed parameters
- ❌ Assumes known features matter

This framework:
- ✅ Discovers new dimensions (hidden, latent)
- ✅ Tests what emerges (evolution, GAN)
- ✅ Explores unknown relationships (cross-regime, information flow)
- ✅ Validates scientifically (statistics, first principles)
- ✅ Backtests automatically (convert to strategy)

## Next Steps

1. **Run Quick Test**: Validate setup
   ```bash
   python scripts/unified_test_framework.py --quick
   ```

2. **Review Results**: Check output in `test_results/`

3. **Run Discovery**: Try unknown dimension exploration
   ```bash
   python scripts/unified_test_framework.py --suite chaos --backtest
   ```

4. **Full Study**: Run complete scientific analysis
   ```bash
   python scripts/unified_test_framework.py --extreme --backtest
   ```

5. **Analyze**: Review plots, reports, and backtest results

## Support

- Documentation: `docs/TESTING_FRAMEWORK.md`
- Examples: `scripts/example_testing_framework.py`
- Testing Guide: `scripts/README_TESTING.md`
- Main README: `README.md`

## License

MIT License - see LICENSE file
