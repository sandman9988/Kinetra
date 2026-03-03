# Silent Failure Workflow - Quick Reference

## Quick Start

```bash
# Run complete workflow (dry-run first)
python scripts/silent_failure_workflow.py --dry-run

# If dry-run looks good, apply fixes
python scripts/silent_failure_workflow.py

# Quick mode (faster)
python scripts/silent_failure_workflow.py --quick
```

## What It Does

1. **Detects** silent failures by testing all modules
2. **Analyzes** failures and categorizes them
3. **Fixes** failures automatically (with backups)
4. **Validates** that fixes work correctly
5. **Reports** comprehensive results

## Output

- **Logs**: `data/logs/silent_failures/failures_YYYYMMDD.jsonl`
- **Reports**: `data/logs/silent_failures/failure_report_*.json`
- **Backups**: `data/backups/failure_fixes/*.bak`

## Manual Steps

```bash
# Step 1: Detect
python scripts/detect_silent_failures.py --mode quick

# Step 2: Analyze
python scripts/fix_silent_failures.py --analyze

# Step 3: Fix (dry-run)
python scripts/fix_silent_failures.py --fix --dry-run

# Step 4: Fix (apply)
python scripts/fix_silent_failures.py --fix --auto

# Step 5: Validate
python scripts/fix_silent_failures.py --validate
```

## Example Output

```
============================================================
Silent Failure Detection - Mode: quick
============================================================

🔍 Testing Kinetra modules...
  Found 96 modules to test
  ✓ kinetra
  ✗ kinetra.backtest_engine
  ...

============================================================
Detection Summary
============================================================
Modules tested:    96
  Succeeded:       15
  Failed:          81
Failures found:    42

============================================================
Failure Breakdown
============================================================

By Category:
  import_error              35
  type_error                 5
  attribute_error            2

Top Failures:
  1. ModuleNotFoundError (count: 25)
     File: kinetra/physics_engine.py:31
     Message: No module named 'sklearn'...
```

## Safety Features

✓ Backups created before any changes  
✓ Dry-run mode to preview changes  
✓ Validation after fixes  
✓ Rollback from backups if needed  

## Full Documentation

See [./SILENT_FAILURE_WORKFLOW.md](./SILENT_FAILURE_WORKFLOW.md) for complete documentation.
