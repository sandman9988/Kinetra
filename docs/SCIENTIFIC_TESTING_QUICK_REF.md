# Scientific Testing Framework - Quick Reference

## Quick Start

### 1. Run Full Testing Programme

```bash
# Complete scientific testing (2-4 hours)
python scripts/run_scientific_testing.py --full

# Quick validation run (10-20 minutes)
python scripts/run_scientific_testing.py --quick

# Local-only mode (no git sync)
python scripts/run_scientific_testing.py --full --no-git-sync
```

### 2. Run Specific Phase

```bash
# Data validation only
python scripts/run_scientific_testing.py --phase data

# Discovery methods only
python scripts/run_scientific_testing.py --phase discovery

# Backtesting only
python scripts/run_scientific_testing.py --phase backtest
```

### 3. Git Sync Management

```bash
# Check git sync status
python scripts/run_scientific_testing.py --check-sync

# Run without git sync (for local development)
python scripts/run_scientific_testing.py --quick --no-git-sync
```

## Git Sync Integration

The framework automatically integrates with the DevOps module to:
- ✅ Check sync status before running tests
- ✅ Pull latest changes from remote if behind
- ✅ Display sync status after test completion
- ✅ Support local-only mode with `--no-git-sync`

**Benefits:**
- Ensures local and remote code stay synchronized
- Prevents running outdated code
- Supports both online (synced) and offline (local) workflows

## Common Commands

### Run Discovery Methods

```python
from kinetra.discovery_methods import DiscoveryMethodRunner

runner = DiscoveryMethodRunner()
results = runner.run_all_discoveries(
    data=market_data,
    methods=['chaos_theory', 'adversarial']
)
```

### Statistical Validation

```python
from kinetra.test_executor import StatisticalRigor

validator = StatisticalRigor()

# Check for overfitting
pbo = validator.calculate_pbo(returns_is, returns_oos)
print(f"PBO: {pbo:.3f} ({'PASS' if pbo < 0.05 else 'FAIL'})")

# Cross-validation
splits = validator.combinatorially_purged_cv(data, n_splits=5)

# Bootstrap CI
lower, upper = validator.bootstrap_confidence_interval(returns)
```

### Backtest Strategy

```python
from kinetra.integrated_backtester import IntegratedBacktester

backtester = IntegratedBacktester()
result = backtester.backtest_discovered_strategy(
    strategy_config=config,
    data=market_data
)

print(f"Sharpe: {result.sharpe_ratio:.2f}")
print(f"Win Rate: {result.win_rate:.2%}")
print(f"Significant: {result.is_statistically_significant}")
```

## Key Metrics

| Metric | Good | Bad | Description |
|--------|------|-----|-------------|
| **PBO** | <0.05 | >0.5 | Probability of overfitting |
| **Sharpe Ratio** | >1.0 | <0.5 | Risk-adjusted returns |
| **Omega Ratio** | >1.2 | <1.0 | Gains/losses ratio |
| **Win Rate** | >55% | <45% | Percentage of winning trades |
| **MFE Captured** | >60% | <40% | Exit efficiency |
| **p-value** | <0.05 | >0.05 | Statistical significance |

## Discovery Methods

| Method | Purpose | Key Output |
|--------|---------|------------|
| **Hidden Dimensions** | Find latent factors | PCA/ICA components |
| **Chaos Theory** | Measure predictability | Lyapunov, Hurst, Entropy |
| **Adversarial** | Filter noise | Statistically significant features |
| **Meta-Learning** | Optimal combinations | Feature importance |

## File Structure

```
scientific_testing_results/
├── data_validation/
│   └── validation_report.json
├── discovery/
│   ├── SYMBOL_TF_discoveries.json
│   └── ...
├── backtests/
│   ├── strategy_name_TIMESTAMP.json
│   └── ...
└── reports/
    └── scientific_testing_report_TIMESTAMP.txt
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Tests too slow | Use `--quick` mode |
| Memory errors | Auto-fixer will reduce batch sizes |
| Import errors | Auto-fixer will install missing packages |
| No patterns found | Check data quality with `--phase data` |

## Statistical Checks Checklist

- [ ] PBO < 0.05
- [ ] p-value < 0.05
- [ ] OOS degradation < 20%
- [ ] Bootstrap CI doesn't contain 0
- [ ] Sharpe > 1.0
- [ ] Omega > 1.2
- [ ] Win Rate > 50%
- [ ] CPCV shows consistent performance

## Best Practices

1. ✅ **Always validate statistically** - Never trust backtest alone
2. ✅ **Use CPCV** - Prevent information leakage
3. ✅ **Check PBO** - Guard against overfitting
4. ✅ **Run discovery methods** - Find unknown patterns
5. ✅ **Test auto-fix** - Ensure robust execution

## Next Steps

1. Read full guide: `docs/SCIENTIFIC_TESTING_GUIDE.md`
2. Run quick test: `python scripts/run_scientific_testing.py --quick`
3. Review results in `scientific_testing_results/`
4. Adjust and re-run as needed
