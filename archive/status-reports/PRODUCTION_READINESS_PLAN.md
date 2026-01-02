# Production Readiness Plan
## Continuous Testing, Fixing, and Verification

**Goal**: Systematically fix thousands of errors to achieve production-ready status

**Status**: Initial framework created  
**Last Updated**: 2026-01-01

---

## Overview

The system currently has thousands of errors preventing production deployment. We need a systematic approach to:
1. **Identify** all errors through comprehensive testing
2. **Categorize** errors by type, severity, and fix-ability
3. **Prioritize** fixes based on impact
4. **Apply** fixes automatically where possible
5. **Verify** fixes don't introduce regressions
6. **Iterate** until production-ready

---

## Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                  CONTINUOUS FIX PIPELINE                     │
└─────────────────────────────────────────────────────────────┘
           │
           ├──▶ [1] RUN TEST SUITE
           │     ├─ Menu exerciser
           │     ├─ Real data tests
           │     ├─ Data preparation
           │     ├─ Backtest validation
           │     └─ Import tests
           │
           ├──▶ [2] COLLECT & CATEGORIZE ERRORS
           │     ├─ dtype incompatibility (CRITICAL)
           │     ├─ StopIteration (HIGH)
           │     ├─ KeyError (HIGH)
           │     ├─ FileNotFound (HIGH)
           │     ├─ AttributeError (MEDIUM)
           │     └─ Others...
           │
           ├──▶ [3] PRIORITIZE BY SEVERITY
           │     └─ CRITICAL > HIGH > MEDIUM > LOW
           │
           ├──▶ [4] APPLY FIXES
           │     ├─ Automated (where possible)
           │     ├─ Semi-automated (with confirmation)
           │     └─ Manual (documentation provided)
           │
           ├──▶ [5] VERIFY FIXES
           │     └─ Re-run failed tests
           │
           └──▶ [6] ITERATE
                 └─ Loop until error count = 0
```

---

## Error Categories

### CRITICAL (Blocking All Operations)

#### 1. DType Incompatibility
**Pattern**: `resolved dtypes are not compatible with add.reduce`  
**Impact**: ALL backtesting strategies fail (BERSERKER, SNIPER, MULTI_AGENT_V7)  
**Root Cause**: String columns being used in mathematical operations  
**Auto-Fixable**: Yes  
**Fix Strategy**:
```python
# 1. Identify string columns
string_cols = df.select_dtypes(include=['object']).columns

# 2. Convert to numeric
for col in ['open', 'high', 'low', 'close', 'volume']:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# 3. Handle MT5 format
df.columns = [col.strip('<>').lower() for col in df.columns]
df['time'] = pd.to_datetime(df['date'] + ' ' + df['time'])
```

**Affected Files**:
- `scripts/download/prepare_data.py`
- `kinetra/berserker_strategy.py`
- `kinetra/backtest_engine.py`
- All strategy files

**Estimated Occurrences**: 1000s (every backtest run)

---

### HIGH Priority

#### 2. Missing Column Names ('time' not found)
**Pattern**: `KeyError.*time|'time' not found`  
**Impact**: Data preparation fails for all 87 CSV files  
**Root Cause**: MT5 format uses `<DATE>`, `<TIME>` instead of `time`  
**Auto-Fixable**: Yes  
**Fix Strategy**:
```bash
# Run converter first
python scripts/download/convert_mt5_format.py

# Or update prepare_data.py to handle MT5 format
```

**Affected Files**:
- All 87 CSV files in `data/master/`
- `scripts/download/prepare_data.py`

**Estimated Occurrences**: 87 (one per file)

#### 3. StopIteration in Submenus
**Pattern**: `StopIteration`  
**Impact**: Menu crashes during automated testing  
**Root Cause**: Missing exception handler  
**Auto-Fixable**: Yes (partially done)  
**Fix Strategy**:
```python
except (EOFError, StopIteration):
    sys.exit(0)
```

**Affected Files**:
- Menu submenu functions (exploration, backtesting)

**Estimated Occurrences**: ~7 remaining

---

### MEDIUM Priority

#### 4. Timeout Errors
**Pattern**: `Timeout|timed out`  
**Impact**: Long-running operations block pipeline  
**Auto-Fixable**: Yes  
**Fix Strategy**:
- Add progress bars
- Optimize algorithms
- Increase timeouts
- Use async operations

**Estimated Occurrences**: Unknown (discovered during testing)

#### 5. Attribute/Key Errors
**Impact**: Runtime failures  
**Auto-Fixable**: Yes  
**Fix Strategy**:
```python
# KeyError fix
value = dict.get(key, default)

# AttributeError fix
if hasattr(obj, 'attribute'):
    value = obj.attribute
```

**Estimated Occurrences**: Unknown

---

## Fix Strategies

### Automated Fixes

#### Strategy 1: DType Conversion
```python
def fix_dtype_incompatibility(df: pd.DataFrame) -> pd.DataFrame:
    """Convert string columns to appropriate numeric types."""
    # Detect and convert numeric columns
    for col in df.columns:
        if col in ['open', 'high', 'low', 'close', 'volume', 'tickvol']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Handle datetime
    if 'time' not in df.columns and 'date' in df.columns:
        df['time'] = pd.to_datetime(df['date'] + ' ' + df['time'])
        df = df.drop(columns=['date'])
    
    return df
```

#### Strategy 2: Column Name Standardization
```python
def standardize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Convert MT5 format to standard format."""
    # Remove angle brackets and lowercase
    df.columns = [col.strip('<>').lower() for col in df.columns]
    
    # Combine date/time if needed
    if 'date' in df.columns and 'time' in df.columns:
        df['time'] = pd.to_datetime(
            df['date'].astype(str) + ' ' + df['time'].astype(str)
        )
        df = df.drop(columns=['date'])
    
    return df
```

#### Strategy 3: Exception Handling
```python
def add_exception_handlers(file_path: str):
    """Add StopIteration handlers to Python file."""
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Replace EOFError with (EOFError, StopIteration)
    content = content.replace(
        'except EOFError:',
        'except (EOFError, StopIteration):'
    )
    
    with open(file_path, 'w') as f:
        f.write(content)
```

---

## Testing Workflow

### Phase 1: Discovery (Current)
```bash
# Run comprehensive tests
python scripts/testing/continuous_fix_pipeline.py --max-cycles 1

# Output: Error inventory with categories
```

### Phase 2: Automated Fixing
```bash
# Apply automated fixes
python scripts/testing/continuous_fix_pipeline.py --auto-fix --max-cycles 10

# Output: 
# - Fixes applied
# - Verification results
# - Remaining errors
```

### Phase 3: Manual Fixes
```bash
# For errors requiring manual intervention:
# 1. Review error report
# 2. Apply fixes based on documentation
# 3. Re-run pipeline
```

### Phase 4: Verification
```bash
# Full regression test
python scripts/testing/continuous_fix_pipeline.py --max-cycles 1

# Target: 0 errors
```

---

## Metrics & KPIs

### Success Criteria
| Metric | Target | Current | Status |
|--------|--------|---------|--------|
| Total Errors | 0 | ~1000s | 🔴 |
| Critical Errors | 0 | 2+ | 🔴 |
| High Priority | 0 | 10+ | 🔴 |
| Test Pass Rate | 100% | 56% | 🟡 |
| Auto-Fix Rate | >70% | TBD | ⚪ |
| Manual Fixes Required | <30% | TBD | ⚪ |

### Progress Tracking
```
Cycle 1: 1000 errors → Apply fixes → Cycle 2: 800 errors → ...
                                                ↓
                                        Target: 0 errors
```

---

## Implementation Phases

### Phase 1: Framework Setup ✅ DONE
- [x] Created continuous_fix_pipeline.py
- [x] Error categorization system
- [x] Automated test suite
- [x] Logging and reporting

### Phase 2: Critical Fixes (Week 1)
- [ ] Fix dtype incompatibility globally
- [ ] Convert all MT5 CSV files
- [ ] Update prepare_data.py for MT5 format
- [ ] Add dtype validation to all strategies
- [ ] Test: Target <100 critical errors

### Phase 3: High Priority Fixes (Week 1-2)
- [ ] Fix all StopIteration errors
- [ ] Fix column name mismatches
- [ ] Add missing file checks
- [ ] Install missing dependencies
- [ ] Test: Target <50 high priority errors

### Phase 4: Medium/Low Priority (Week 2-3)
- [ ] Fix timeout issues
- [ ] Add attribute/key checks
- [ ] Optimize slow operations
- [ ] Add progress bars everywhere
- [ ] Test: Target <10 errors

### Phase 5: Verification & Polish (Week 3-4)
- [ ] Full regression testing
- [ ] Performance benchmarking
- [ ] Documentation updates
- [ ] Production deployment checklist
- [ ] Test: Target 0 errors

---

## Daily Workflow

### Morning: Run Pipeline
```bash
cd /home/runner/work/Kinetra/Kinetra
python scripts/testing/continuous_fix_pipeline.py --auto-fix --max-cycles 5
```

### Midday: Review Results
- Check error report
- Identify new issues
- Plan manual fixes

### Afternoon: Apply Manual Fixes
- Code changes for non-auto-fixable errors
- Update documentation
- Add tests for fixes

### Evening: Verify
```bash
python scripts/testing/continuous_fix_pipeline.py --max-cycles 1
```

### Track Progress
- Update metrics
- Commit fixes
- Update this document

---

## Tools

### 1. Continuous Fix Pipeline
**File**: `scripts/testing/continuous_fix_pipeline.py`  
**Usage**:
```bash
# Dry run (no fixes)
python scripts/testing/continuous_fix_pipeline.py

# Auto-fix mode
python scripts/testing/continuous_fix_pipeline.py --auto-fix

# Custom cycles
python scripts/testing/continuous_fix_pipeline.py --auto-fix --max-cycles 20
```

### 2. Menu Exerciser
**File**: `scripts/testing/exercise_menu_continuous.py`  
**Usage**:
```bash
python scripts/testing/exercise_menu_continuous.py --iterations 10
```

### 3. Real Data Exerciser
**File**: `scripts/testing/exercise_menu_with_real_data.py`  
**Usage**:
```bash
python scripts/testing/exercise_menu_with_real_data.py --profile
```

---

## Logging & Reports

### Logs Location
```
logs/continuous_pipeline/
├── pipeline_20260101_120000.log  # Detailed logs
├── report_20260101_120530.json   # Test results
└── dtype_fix_guide.md            # Fix documentation
```

### Report Format
```json
{
  "cycles": 5,
  "total_errors": 1000,
  "errors_fixed": 400,
  "errors_remaining": 600,
  "errors_by_category": {
    "dtype_incompatibility": 500,
    "missing_column": 87,
    "stopiteration": 7,
    ...
  },
  "fixes_applied": [...]
}
```

---

## Emergency Procedures

### If Pipeline Gets Stuck
1. Check logs: `tail -f logs/continuous_pipeline/*.log`
2. Identify bottleneck
3. Kill process: `Ctrl+C`
4. Fix bottleneck
5. Restart with lower cycle count

### If Auto-Fixes Break System
1. Git revert: `git reset --hard HEAD~1`
2. Review fix that caused issue
3. Update fix strategy
4. Re-run with manual review

### If Error Count Increases
1. New errors introduced by fixes
2. Review recent changes
3. Add regression tests
4. Fix introduced bugs

---

## Success Indicators

### Ready for Production When:
- ✅ All tests pass (100% success rate)
- ✅ Zero critical errors
- ✅ Zero high priority errors
- ✅ <5 medium priority errors (documented as known issues)
- ✅ All 87 CSV files load successfully
- ✅ Data preparation completes without errors
- ✅ At least one strategy runs successfully
- ✅ Menu navigation works without crashes
- ✅ Performance meets targets (see MENU_EXERCISE_SUMMARY.md)

---

## Next Steps

### Immediate (Today)
1. Run continuous pipeline for first full cycle
2. Generate error inventory
3. Apply critical dtype fixes
4. Verify fixes

### Short Term (This Week)
1. Fix all critical errors
2. Convert MT5 CSV format
3. Fix StopIteration errors
4. Achieve <100 total errors

### Medium Term (This Month)
1. Fix all high priority errors
2. Optimize performance bottlenecks
3. Add comprehensive progress bars
4. Achieve <10 total errors

### Long Term (Next Month)
1. Production deployment
2. Continuous monitoring
3. Automated regression testing
4. Performance optimization

---

## Resources

- **Main Summary**: `MENU_EXERCISE_SUMMARY.md`
- **Pipeline Code**: `scripts/testing/continuous_fix_pipeline.py`
- **Error Reports**: `logs/continuous_pipeline/`
- **Fix Guides**: `logs/continuous_pipeline/dtype_fix_guide.md`

---

**Last Updated**: 2026-01-01  
**Status**: Framework created, ready for first full cycle  
**Next Action**: Run `python scripts/testing/continuous_fix_pipeline.py --auto-fix`
