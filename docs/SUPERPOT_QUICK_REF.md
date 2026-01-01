# SUPERPOT Quick Reference

## One-Line Commands

```bash
# Quick test
python scripts/analysis/superpot_empirical.py --quick

# Test crypto markets
python scripts/analysis/superpot_empirical.py --asset-class crypto

# Test H1 timeframe
python scripts/analysis/superpot_empirical.py --timeframe H1

# Full run (200 episodes)
python scripts/analysis/superpot_empirical.py --episodes 200 --max-files 50
```

## Output Interpretation

### Significance Levels
- `***` = p < 0.001 (highly significant)
- `**` = p < 0.01 (significant - theorem candidate)
- `*` = p < 0.05 (marginally significant)

### Effect Size (Cohen's d)
- d < 0.2 = negligible
- d = 0.2-0.5 = small
- d = 0.5-0.8 = medium ✓ (useful)
- d > 0.8 = large ✓✓ (very useful)

### Empirical Theorem Criteria
1. ✓ p < 0.01 (after Bonferroni correction)
2. ✓ Cohen's d > 0.5 (medium+ effect)
3. ✓ Validated out-of-sample
4. ✓ Reproducible

## Workflow

```bash
# 1. Broad discovery (all asset classes)
python scripts/analysis/superpot_empirical.py --episodes 200

# 2. Class-specific testing
for class in crypto forex metals; do
  python scripts/analysis/superpot_empirical.py --asset-class $class --episodes 200
done

# 3. Compare results
ls -lh results/superpot/

# 4. Extract top features
grep "STATISTICALLY SIGNIFICANT" results/superpot/*.json
```

## Key Metrics

| Metric | Good | Excellent |
|--------|------|-----------|
| Win Rate | > 50% | > 60% |
| Avg PnL | > $0 | > $50 |
| Surviving Features | 20-50 | 10-30 |
| Statistically Significant | > 5 | > 10 |

## Common Issues

### No episodes completing
- Check data file format (tab-separated, angle brackets)
- Verify column names (open, high, low, close)
- Ensure prices are numeric (convert if needed)

### No significant features (p > 0.01)
- Increase sample size (--episodes 200+)
- Check if data quality is sufficient
- Verify feature extraction is working

### Too many features pruned
- Increase min_keep threshold in code
- Check if features are too correlated
- Verify statistical calculations

## File Locations

| Path | Description |
|------|-------------|
| `scripts/analysis/superpot_empirical.py` | Main script |
| `results/superpot/` | Output JSON files |
| `docs/SUPERPOT_EMPIRICAL_TESTING.md` | Full documentation |
| `docs/EMPIRICAL_THEOREMS.md` | Validated theorems |

## Quick Checks

```bash
# Check if script exists
ls -lh scripts/analysis/superpot_empirical.py

# Run help
python scripts/analysis/superpot_empirical.py --help

# Quick 10-episode test
python scripts/analysis/superpot_empirical.py --episodes 10 --max-steps 100

# View recent results
cat results/superpot/empirical_*.json | jq '.metrics'
```

## Next Steps After Running

1. Review statistically significant features
2. Check effect sizes (d > 0.5?)
3. Compare across asset classes
4. Validate on different data
5. Document as theorems if criteria met
6. Integrate into trading models

---

**Quick Rule**: If p < 0.01 AND d > 0.5 AND validated → It's a theorem!
