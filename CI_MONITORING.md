# CI Monitoring Quick Reference

**Repository**: [sandman9988/Kinetra](https://github.com/sandman9988/Kinetra)  
**Last Updated**: 2026-01-03  
**Status**: 🟢 Active Monitoring

---

## Quick Links

### GitHub Actions Dashboard
🔗 **Main Dashboard**: https://github.com/sandman9988/Kinetra/actions

### Specific Workflows
- **Fast Tests**: https://github.com/sandman9988/Kinetra/actions/workflows/ci.yml?query=event%3Apush
- **Exhaustive Tests**: https://github.com/sandman9988/Kinetra/actions/workflows/ci.yml?query=event%3Aschedule
- **Rules Validation**: Filter by job name in workflow runs

### Recent Commits
- **Latest Push**: https://github.com/sandman9988/Kinetra/commit/2dff6d7
- **Rules Validation Commit**: https://github.com/sandman9988/Kinetra/commit/b1122fd
- **Exhaustive Testing Framework**: https://github.com/sandman9988/Kinetra/commit/c366902

---

## Current CI Jobs

### 1. Fast Tests (Every Push/PR)
**Purpose**: Quick validation of core functionality  
**Runtime**: ~4-5 minutes  
**Triggers**: Push to main, Pull requests  

**What It Tests**:
- Agent factory functionality (6 agent types)
- Core physics engine
- Unit tests (subset in CI mode)
- Integration tests

**Expected Status**: ✅ Should pass on every commit

### 2. Exhaustive Tests (Nightly)
**Purpose**: Full test matrix validation  
**Runtime**: ~30-60 minutes (depending on data coverage)  
**Triggers**: 
- Scheduled: Daily at 02:00 UTC
- Manual: workflow_dispatch

**What It Tests**:
- All agent types × all instrument/timeframe combinations
- Multi-regime validation
- Performance benchmarks
- Statistical significance tests

**Artifacts Generated**:
- `test-dashboard` - Interactive HTML report
- `test_report_*.json` - Machine-readable results

**Expected Status**: ⚠️ May have skipped tests due to missing data (currently 45% coverage)

### 3. Rules Validation (Every Push/PR)
**Purpose**: Enforce coding standards and security  
**Runtime**: ~2-3 minutes  
**Triggers**: Push to main, Pull requests

**What It Checks**:
- Canonical rules references (AGENT_RULES_MASTER.md)
- Banned patterns (traditional TA indicators)
- Hardcoded credentials detection
- Vectorization compliance

**Expected Status**: ✅ Should pass (just added - verify first run)

---

## Monitoring Checklist

### After Every Push
- [ ] Check fast-tests job completes successfully
- [ ] Verify rules-validation job passes
- [ ] Review any new warnings or deprecations
- [ ] Check build time (should be <5 min for fast tests)

### Daily (After 02:00 UTC)
- [ ] Verify exhaustive-tests completed
- [ ] Download test-dashboard artifact
- [ ] Open test_report.html in browser
- [ ] Review any failed test combinations
- [ ] Check for performance regressions

### Weekly
- [ ] Review test coverage trends
- [ ] Analyze dashboard metrics (Omega, Z-factor, etc.)
- [ ] Identify missing data combinations
- [ ] Update data coverage reports

---

## How to Access Artifacts

### Step-by-Step
1. Go to https://github.com/sandman9988/Kinetra/actions
2. Click on the latest "Exhaustive Tests" workflow run
3. Scroll down to the "Artifacts" section
4. Download `test-dashboard` (ZIP file)
5. Unzip and open `test_report.html` in a web browser

### Expected Artifact Contents
```
test-dashboard.zip
└── test_report.html          # Main dashboard (interactive)
└── test_report_*.json        # Raw test results (optional)
```

---

## CI Status Badges

Add to README.md:

```markdown
[![Fast Tests](https://github.com/sandman9988/Kinetra/actions/workflows/ci.yml/badge.svg?event=push)](https://github.com/sandman9988/Kinetra/actions/workflows/ci.yml)
```

---

## Troubleshooting

### Fast Tests Failing
**Check**:
1. Review pytest output in job logs
2. Look for import errors or missing dependencies
3. Check if AgentFactory verification passed
4. Verify CI_MODE environment variable is set

**Common Issues**:
- Missing dependencies → Update `pyproject.toml`
- Import errors → Check Python path
- Agent factory failures → Run `python -m kinetra.agent_factory` locally

### Exhaustive Tests Timing Out
**Check**:
1. How many test combinations ran before timeout?
2. Is data coverage too high for nightly run?
3. Are there performance regressions?

**Solutions**:
- Reduce test matrix in CI mode
- Parallelize more aggressively
- Skip slow combinations in nightly run

### Rules Validation Failing
**Check**:
1. Review banned pattern matches
2. Check for hardcoded credentials
3. Verify AGENT_RULES_MASTER.md references

**Common Issues**:
- False positives for TA indicators → Update regex patterns
- Credential detection → Use environment variables
- Missing rules references → Add to code comments

### Dashboard Not Generating
**Check**:
1. Look for plotly/dash import errors
2. Verify `--generate-dashboard` flag is set
3. Check test results JSON exists

**Solutions**:
- Install visualization dependencies: `pip install plotly dash`
- Run locally: `python scripts/run_exhaustive_tests.py --generate-dashboard`
- Check logs for specific error messages

---

## Performance Targets

Monitor these metrics in the dashboard:

| Metric | Target | Alert If |
|--------|--------|----------|
| **Omega Ratio** | > 2.7 | < 2.5 |
| **Z-Factor** | > 2.5 | < 2.0 |
| **% Energy Captured** | > 65% | < 60% |
| **Composite Health Score** | > 0.90 | < 0.85 |
| **% MFE Captured** | > 60% | < 55% |
| **Fast Test Runtime** | < 5 min | > 7 min |
| **Exhaustive Test Runtime** | < 60 min | > 90 min |

---

## CI Configuration Files

### Primary CI File
**Path**: `.github/workflows/ci.yml`

**Key Sections**:
```yaml
jobs:
  fast-tests:        # Lines 20-80
  exhaustive-tests:  # Lines 82-150
  rules-validation:  # Lines 177-235
```

### Test Configuration
**Path**: `tests/conftest.py`

**Environment Variables**:
- `KINETRA_CI_MODE=1` - Enable fast subset testing
- `PYTEST_PARALLEL=auto` - Auto-detect CPU cores

---

## Manual Workflow Triggers

### Trigger Exhaustive Tests Manually
1. Go to: https://github.com/sandman9988/Kinetra/actions/workflows/ci.yml
2. Click "Run workflow" dropdown (top-right)
3. Select branch: `main`
4. Click "Run workflow" button

### Or via GitHub CLI
```bash
gh workflow run ci.yml
```

---

## Notification Setup (Optional)

### Slack Notifications
Add to `.github/workflows/ci.yml`:
```yaml
- name: Notify Slack
  if: failure()
  uses: 8398a7/action-slack@v3
  with:
    status: ${{ job.status }}
    webhook_url: ${{ secrets.SLACK_WEBHOOK }}
```

### Email Notifications
GitHub automatically sends emails for:
- Failed workflow runs
- First failure after success
- Recovery (failure → success)

Configure at: https://github.com/settings/notifications

---

## Data Coverage Monitoring

### Current Status
- **Total Combinations**: 60 (10 instruments × 6 timeframes)
- **Available**: 27 combinations (45%)
- **Missing**: 33 combinations (55%)

### High-Priority Gaps
1. BTCUSD D1 (crypto_primary)
2. EURUSD H1 (forex_primary)
3. EURUSD H4 (forex_primary)
4. EURUSD D1 (forex_primary)

### Update Coverage
```bash
# Audit current coverage
python scripts/audit_data_coverage.py --show-gaps \
  --report data/coverage_report.csv

# Consolidate new data
python scripts/consolidate_data.py --symlink

# Re-run audit
python scripts/audit_data_coverage.py --show-gaps
```

---

## Next Steps After CI Validation

### Immediate (Post-Push)
1. ✅ Monitor first fast-tests run (commit 2dff6d7)
2. ✅ Verify rules-validation job passes
3. ⏳ Wait for nightly exhaustive-tests (02:00 UTC)
4. 📥 Download and review test-dashboard artifact

### Short-Term (This Week)
1. Acquire missing data for EURUSD/BTCUSD high-priority combos
2. Validate GPU testing on real hardware (if available)
3. Review dashboard metrics for baseline performance
4. Document any CI issues or improvements needed

### Medium-Term (This Month)
1. Implement HPO integration with Optuna
2. Add live dashboard streaming for long runs
3. Expand agent portfolio (A3C, SAC, TD3)
4. Increase data coverage to 80% target

---

## Contact & Support

### Repository Issues
Report CI problems: https://github.com/sandman9988/Kinetra/issues/new

### Documentation
- **Testing Guide**: `docs/EXHAUSTIVE_TESTING_GUIDE.md`
- **Validation Report**: `EXHAUSTIVE_TESTING_VALIDATION.md`
- **Action Plan**: `EXHAUSTIVE_TESTING_ACTION_PLAN.md`
- **Quick Start**: `EXHAUSTIVE_TESTING_QUICKSTART.md`

---

**Status**: 🟢 All systems operational  
**Last CI Check**: 2026-01-03 22:30 UTC  
**Next Scheduled Run**: Daily at 02:00 UTC