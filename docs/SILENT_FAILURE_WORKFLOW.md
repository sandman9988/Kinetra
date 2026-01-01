# Silent Failure Detection and Fixing Workflow

This document describes the automated workflow for detecting and fixing silent failures in the Kinetra codebase.

## Overview

The workflow consists of three main components:

1. **Detection** (`detect_silent_failures.py`) - Discovers silent failures by testing the codebase
2. **Fixing** (`fix_silent_failures.py`) - Analyzes and automatically fixes detected failures
3. **Orchestration** (`silent_failure_workflow.py`) - Coordinates the complete workflow

## Quick Start

### Run Complete Workflow

```bash
# Dry run (analyze without applying fixes)
python scripts/silent_failure_workflow.py --dry-run

# Apply fixes automatically
python scripts/silent_failure_workflow.py

# Quick mode (faster, less comprehensive)
python scripts/silent_failure_workflow.py --quick
```

### Manual Step-by-Step

```bash
# Step 1: Detect failures
python scripts/detect_silent_failures.py --mode comprehensive

# Step 2: Analyze failures
python scripts/fix_silent_failures.py --analyze

# Step 3: Apply fixes
python scripts/fix_silent_failures.py --fix --auto

# Step 4: Validate fixes
python scripts/fix_silent_failures.py --validate
```

## Components

### 1. Silent Failure Logger (`kinetra/silent_failure_logger.py`)

Core logging system that captures silent failures across the codebase.

**Features:**
- Automatic categorization (import errors, type errors, etc.)
- Severity levels (low, medium, high, critical)
- Thread-safe concurrent logging
- JSON export for AI analysis
- Pattern detection for recurring issues

**Usage in Code:**

```python
from kinetra.silent_failure_logger import log_failure

try:
    risky_operation()
except Exception as e:
    log_failure(e, context={"operation": "risky_op"})
    # Continue execution (silent failure)
```

**Decorator Usage:**

```python
from kinetra.silent_failure_logger import log_failures

@log_failures(context={"module": "data_loading"})
def load_data():
    # Failures logged automatically
    pass
```

### 2. Failure Detector (`scripts/detect_silent_failures.py`)

Systematically tests the codebase to discover silent failures.

**Detection Modes:**

- **Quick**: Fast module import testing (~30 seconds)
- **Comprehensive**: Full testing with data loading and execution (~5-10 minutes)
- **Targeted**: Test specific modules

**Usage:**

```bash
# Quick detection
python scripts/detect_silent_failures.py --mode quick

# Comprehensive detection
python scripts/detect_silent_failures.py --mode comprehensive

# Target specific module
python scripts/detect_silent_failures.py --mode targeted --module kinetra.data_loader
```

**What It Tests:**

- ✓ All kinetra module imports
- ✓ Script syntax and structure
- ✓ Data loading operations
- ✓ Physics engine calculations
- ✓ Known issues (e.g., InstrumentData len())

### 3. Automated Fixer (`scripts/fix_silent_failures.py`)

Analyzes detected failures and applies automated fixes.

**Fix Strategies:**

1. **Add __len__ method** - Adds missing len() support to dataclasses
2. **Fix import errors** - Adds missing imports
3. **Fix type errors** - Corrects type mismatches
4. **Add error handling** - Wraps risky operations
5. **Manual review** - Flags complex issues for human review

**Usage:**

```bash
# Analyze failures
python scripts/fix_silent_failures.py --analyze

# Dry run (see what would be fixed)
python scripts/fix_silent_failures.py --fix --dry-run

# Apply fixes
python scripts/fix_silent_failures.py --fix --auto

# Validate fixes
python scripts/fix_silent_failures.py --validate
```

**Safety Features:**

- ✓ Automatic backups before modifications
- ✓ Dry-run mode to preview changes
- ✓ Validation after fixes
- ✓ Rollback capability

### 4. Workflow Orchestrator (`scripts/silent_failure_workflow.py`)

Coordinates the complete detection → analysis → fix → validate workflow.

**Workflow Phases:**

1. **Detection** - Run comprehensive testing
2. **Analysis** - Categorize and prioritize failures
3. **Fixing** - Apply automated fixes
4. **Validation** - Ensure fixes work
5. **Reporting** - Generate comprehensive reports

**Usage:**

```bash
# Complete workflow
python scripts/silent_failure_workflow.py

# Dry run (no changes)
python scripts/silent_failure_workflow.py --dry-run

# Quick mode
python scripts/silent_failure_workflow.py --quick

# Individual phases
python scripts/silent_failure_workflow.py --detect-only
python scripts/silent_failure_workflow.py --analyze-only
python scripts/silent_failure_workflow.py --fix-only
```

## Output and Reports

### Log Files

Failures are logged to `data/logs/silent_failures/`:

```
data/logs/silent_failures/
├── failures_20241231.jsonl      # Daily JSONL log
├── failure_report_20241231_143022.json  # Comprehensive report
└── ...
```

### Backups

Original files are backed up before modification:

```
data/backups/failure_fixes/
├── explore_interactive.py.20241231_143530.bak
├── exploration_integration.py.20241231_143531.bak
└── ...
```

### Report Format

```json
{
  "metadata": {
    "generated_at": "2024-12-31T14:35:30",
    "session_id": "20241231_143022"
  },
  "statistics": {
    "total_unique_failures": 15,
    "total_failure_count": 47,
    "by_category": {
      "type_error": 8,
      "import_error": 5,
      "attribute_error": 2
    },
    "by_severity": {
      "medium": 10,
      "low": 5
    }
  },
  "failures": [...]
}
```

## Example: Fixing the InstrumentData len() Issue

The workflow automatically handles the example issue you provided:

### Problem

```python
# In scripts/explore_interactive.py line 293
print(f"  {key}: {len(data)} bars")  # ❌ TypeError: object of type 'InstrumentData' has no len()
```

### Automated Fixes

The workflow applies two fixes:

1. **Add __len__ method to InstrumentData class:**

```python
@dataclass
class InstrumentData:
    # ... existing fields ...
    
    def __len__(self) -> int:
        """Return the number of data points."""
        if hasattr(self, 'data') and hasattr(self.data, '__len__'):
            return len(self.data)
        elif hasattr(self, 'prices') and hasattr(self.prices, '__len__'):
            return len(self.prices)
        return 0
```

2. **Fix the len() call in explore_interactive.py:**

```python
# Defensive version
print(f"  {key}: {len(data.data) if hasattr(data, 'data') else len(data.prices) if hasattr(data, 'prices') else 'N/A'} bars")
```

### Verification

After fixes are applied, the workflow:
- ✓ Creates backups of original files
- ✓ Applies both fixes
- ✓ Validates that InstrumentData has __len__
- ✓ Generates report with success status

## AI Agent Integration

The failure logs are designed for AI agent analysis:

### For AI Agents

```python
from kinetra.silent_failure_logger import get_failure_logger

# Get all failures
logger = get_failure_logger()
failures = logger.get_failures()

# Filter by category/severity
high_priority = logger.get_failures(severity="high")
import_errors = logger.get_failures(category="import_error")

# Get statistics for pattern analysis
stats = logger.get_statistics()

# Export for external tools
report_path = logger.export_report()
```

### AI-Friendly Features

- Structured JSON format
- Categorization and tagging
- Stack traces and context
- Deduplication and aggregation
- Pattern detection

## Best Practices

### For Developers

1. **Use the logger in new code:**
   ```python
   from kinetra.silent_failure_logger import log_failure
   
   try:
       optional_feature()
   except ImportError as e:
       log_failure(e, context={"feature": "optional"})
       # Gracefully degrade
   ```

2. **Run detection before releases:**
   ```bash
   python scripts/silent_failure_workflow.py --quick
   ```

3. **Review backups before committing fixes:**
   ```bash
   ls -la data/backups/failure_fixes/
   ```

### For CI/CD

Add to your CI pipeline:

```yaml
# .github/workflows/ci.yml
- name: Detect Silent Failures
  run: |
    python scripts/detect_silent_failures.py --mode quick
    python scripts/fix_silent_failures.py --analyze
```

## Troubleshooting

### "Module not found" errors

Run from project root:
```bash
cd /path/to/Kinetra
python scripts/silent_failure_workflow.py
```

### Validation fails after fixes

1. Check backups in `data/backups/failure_fixes/`
2. Review changes manually
3. Restore from backup if needed:
   ```bash
   cp data/backups/failure_fixes/file.py.TIMESTAMP.bak path/to/file.py
   ```

### No failures detected

This is good! But if you expected some:
- Try comprehensive mode: `--mode comprehensive`
- Check specific modules: `--module kinetra.specific_module`
- Review log files in `data/logs/silent_failures/`

## Advanced Usage

### Custom Fix Strategies

Extend the fixer with custom strategies:

```python
# In scripts/fix_silent_failures.py

def fix_custom_pattern(self, failure: FailureRecord) -> bool:
    """Custom fix strategy."""
    # Implement your fix logic
    pass

# Add to determine_fix_strategy()
if "custom_pattern" in failure.exception_message:
    return "fix_custom_pattern"
```

### Integration with External Tools

Export failures for external analysis:

```python
from kinetra.silent_failure_logger import get_failure_logger

logger = get_failure_logger()
report = logger.export_report(Path("analysis/failures.json"))

# Use with external tools
# - Load into pandas for statistical analysis
# - Send to monitoring systems
# - Feed to ML models for prediction
```

## Future Enhancements

Planned improvements:

- [ ] Machine learning for fix suggestion
- [ ] Integration with GitHub Issues
- [ ] Automatic PR creation for fixes
- [ ] Real-time failure monitoring dashboard
- [ ] Regression detection (new vs. existing failures)
- [ ] Performance impact analysis
- [ ] Collaborative review workflow

## Support

For issues or questions:

1. Check existing failures: `cat data/logs/silent_failures/failures_*.jsonl`
2. Review documentation: This file
3. Open GitHub issue with failure report attached
