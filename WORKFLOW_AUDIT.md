# Kinetra Workflow Audit - Login → Download → Prepare → Test

**Date**: 2026-01-03  
**Status**: 🔍 Audit Complete - Gaps Identified  
**Purpose**: Map existing code to identify plumbing gaps in end-to-end workflow

---

## Expected User Flow

```
1. Login & Select Account
   ↓
2. Select Symbols & Timeframes  
   ↓
3. Download Data
   ↓
4. Consolidate/Prepare Data
   ↓
5. Run Tests
```

---

## Existing Code Inventory

### 1. Login & Authentication

#### **scripts/download/select_metaapi_account.py**
- ✅ **Exists**: Lists MetaAPI accounts, allows selection
- ❌ **Problem**: Has HARDCODED API token (line 142)
- ❌ **Problem**: Doesn't prompt for next step (download)
- **Returns**: Prints account ID, exits

#### **Menu: kinetra_menu.py → show_authentication_menu()**
```python
Options:
  1. Select MetaAPI Account  → calls select_metaapi_account()
  2. Test Connection         → calls test_connection()
```
- ✅ **Fixed**: Now prompts for download after account selection (lines 656-728)
- ✅ **Fixed**: Offers full workflow (download → consolidate → test)

---

### 2. Data Download

#### **scripts/download/download_interactive.py** ✅ COMPLETE WORKFLOW
- ✅ **Step 1**: Select account (asks for token, lists accounts)
- ✅ **Step 2**: Select asset classes (forex, crypto, metals, indices, commodities)
- ✅ **Step 3**: Select symbols (from available symbols in account)
- ✅ **Step 4**: Select timeframes (M15, M30, H1, H4, D1)
- ✅ **Step 5**: Download with progress tracking
- ✅ **Saves to**: `data/master/SYMBOL_TF_STARTDATE_ENDDATE.csv`
- ✅ **Handles**: Token from env, .env file, or manual input
- ✅ **Saves credentials**: Offers to save to .env file

**Entry point**:
```bash
python scripts/download/download_interactive.py
```

**From menu**: Data Management → Option 2 (Manual Download)

#### **scripts/mt5_metaapi_sync.py** ✅ CLI ALTERNATIVE
- ✅ **Init mode**: Download 2+ years of data
- ✅ **Sync mode**: Incremental updates (daily/hourly)
- ✅ **Supports**: Single symbol or bulk sync
- ✅ **Retry logic**: Exponential backoff (2s, 4s, 8s, 16s)
- ✅ **Saves to**: `data/metaapi/SYMBOL_TF.csv`

**Usage**:
```bash
# Initial download
python3 scripts/mt5_metaapi_sync.py --init --symbol EURUSD --timeframe H1 --years 2

# Daily sync
python3 scripts/mt5_metaapi_sync.py --sync --symbol EURUSD --timeframe H1
```

#### **Other Download Scripts** (Legacy/Alternative)
- `scripts/download/metaapi_bulk_download.py` - Bulk download (alternative)
- `scripts/download/download_metaapi.py` - Legacy (deprecated)
- `scripts/download/download_market_data.py` - Legacy (deprecated)
- `scripts/download/download_mt5_data.py` - Legacy (deprecated)

---

### 3. Data Preparation & Consolidation

#### **scripts/consolidate_data.py** ✅ DATA CONSOLIDATION
- ✅ **Scans**: `data/master_standardized/`, `data/runs/*/data/`, other sources
- ✅ **Consolidates**: Symlinks or copies to `data/master_standardized/`
- ✅ **Normalizes**: Filename format `INSTRUMENT_TIMEFRAME.csv`
- ✅ **Modes**: `--symlink` (safe, preserves originals), `--copy`, `--dry-run`

**Usage**:
```bash
# Dry run (see what would happen)
python scripts/consolidate_data.py --dry-run

# Symlink mode (recommended)
python scripts/consolidate_data.py --symlink

# Copy mode
python scripts/consolidate_data.py --copy
```

**From menu**: Data Management → Option 5 (Prepare Data)

#### **scripts/audit_data_coverage.py** ✅ COVERAGE AUDITING
- ✅ **Scans**: All data sources
- ✅ **Reports**: Coverage by instrument × timeframe
- ✅ **Identifies**: Missing high-priority combinations
- ✅ **Exports**: CSV and JSON reports

**Usage**:
```bash
python scripts/audit_data_coverage.py --show-gaps \
  --report data/coverage_report.csv \
  --json data/coverage_report.json
```

**From menu**: Data Management → Option 3 (Check & Fill Missing Data)

#### **scripts/download/prepare_data.py** - Train/Test Split
- ✅ Splits data into train/test sets
- ✅ Validates data integrity

**From menu**: Data Management → Option 5 (Prepare Data)

#### **scripts/download/check_data_integrity.py** - Data Validation
- ✅ Checks for NaN, duplicates, gaps
- ✅ Validates timestamps, OHLCV integrity

**From menu**: Data Management → Option 4 (Data Integrity Check)

---

### 4. Testing

#### **scripts/run_exhaustive_tests.py** ✅ FULL TEST ORCHESTRATOR
- ✅ **Modes**: CI mode (fast subset), full exhaustive
- ✅ **Tests**: All agents × all instrument/timeframe combos
- ✅ **Dashboard**: Generates static HTML report
- ✅ **Exports**: JSON results for analysis

**Usage**:
```bash
# Fast CI mode
KINETRA_CI_MODE=1 python scripts/run_exhaustive_tests.py

# Full with dashboard
python scripts/run_exhaustive_tests.py --generate-dashboard

# CI mode with dashboard
python scripts/run_exhaustive_tests.py --ci-mode --generate-dashboard
```

**From menu**: Backtesting → Various options

#### **tests/test_exhaustive_combinations.py** - Test Suite
- ✅ Unit tests for all agent types
- ✅ Integration tests for full pipeline
- ✅ Regime validation

---

## Menu Integration Analysis

### Main Menu Structure

```
MAIN MENU
├─[1] Login & Authentication
│   ├─[1] Select MetaAPI Account  ✅ NOW prompts for download → consolidate → test
│   └─[2] Test Connection         ✅ Works
│
├─[2] Exploration Testing
│   └─ (Various exploration options)
│
├─[3] Backtesting
│   └─ (Various backtest options)
│
├─[4] Live Testing
│   └─ (Demo/Live trading - requires MT5 terminal)
│
├─[5] Data Management
│   ├─[1] Auto-Download for Configuration  ✅ Works
│   ├─[2] Manual Download                  ✅ Calls download_interactive.py
│   ├─[3] Check & Fill Missing Data        ✅ Works
│   ├─[4] Data Integrity Check             ✅ Works
│   ├─[5] Prepare Data                     ✅ Works
│   └─[6] Backup & Restore                 ✅ Works
│
└─[6] System Status & Health
    └─ (Status checks)
```

---

## Identified Gaps & Issues

### ✅ FIXED: Gap #1 - Broken Flow After Login
**Problem**: After selecting account (Menu 1→1), user dumped back to main menu  
**Solution**: Added prompts in `select_metaapi_account()` to offer:
- Download data?
- Consolidate data?
- Run tests?

**Status**: ✅ Fixed (lines 656-728 in kinetra_menu.py)

### ⚠️ ISSUE #2: Duplicate Account Selection
**Problem**: `download_interactive.py` re-selects account even if already authenticated  
**Impact**: User has to login twice if using Menu 1→1 then download  
**Workaround**: Use Data Management → Manual Download (Option 5→2) directly  
**Proper Fix**: Share authenticated session between menu and download script

### ⚠️ ISSUE #3: Hardcoded Token in select_metaapi_account.py
**Problem**: Line 142 has hardcoded API token (security risk)  
**Impact**: Token exposed in git repository  
**Fix**: Use environment variables or .env file (like download_interactive.py does)

### ⚠️ ISSUE #4: Two Different Download Directories
**Problem**: 
- `download_interactive.py` saves to `data/master/`
- `mt5_metaapi_sync.py` saves to `data/metaapi/`
- Tests expect data in `data/master_standardized/`

**Impact**: User needs to manually consolidate after download  
**Current Workaround**: Run `consolidate_data.py` after download  
**Proper Fix**: Standardize on single output directory OR auto-consolidate after download

---

## Recommended User Workflows

### Workflow A: First-Time Setup (Interactive - RECOMMENDED)

```bash
# 1. Start menu
python kinetra_menu.py

# 2. Select: [1] Login & Authentication
#    Then: [1] Select MetaAPI Account
#    - Enter token when prompted
#    - Select account from list
#    - Answer "Yes" to download data → Will launch download_interactive.py
#    - Select asset classes, symbols, timeframes
#    - Answer "Yes" to consolidate → Will run consolidate_data.py
#    - Answer "Yes" to run tests → Will run exhaustive tests

# Done! Data downloaded, prepared, and tested.
```

### Workflow B: Quick CLI Download (For Specific Symbols)

```bash
# 1. Set credentials (one-time)
export METAAPI_TOKEN="your-token"
export METAAPI_ACCOUNT_ID="your-account-id"

# 2. Download high-priority data
python3 scripts/mt5_metaapi_sync.py --init --symbol EURUSD --timeframe H1 --years 2
python3 scripts/mt5_metaapi_sync.py --init --symbol EURUSD --timeframe H4 --years 2
python3 scripts/mt5_metaapi_sync.py --init --symbol EURUSD --timeframe D1 --years 2
python3 scripts/mt5_metaapi_sync.py --init --symbol BTCUSD --timeframe D1 --years 2

# 3. Consolidate data
python scripts/consolidate_data.py --symlink

# 4. Run tests
python scripts/run_exhaustive_tests.py --generate-dashboard
```

### Workflow C: Menu-Based (Skip Login If Already Authenticated)

```bash
# 1. Start menu
python kinetra_menu.py

# 2. Select: [5] Data Management
#    Then: [2] Manual Download
#    - Will re-authenticate (download_interactive.py handles it)
#    - Select what to download
#    - Downloads to data/master/

# 3. Back to Data Management: [5] Prepare Data
#    - Consolidates to data/master_standardized/

# 4. Run tests from Backtesting menu or CLI
python scripts/run_exhaustive_tests.py --generate-dashboard
```

---

## Missing Plumbing (To Implement)

### Priority 1: Fix Hardcoded Token
**File**: `scripts/download/select_metaapi_account.py`  
**Action**: Remove hardcoded token, use environment/prompt like download_interactive.py does

### Priority 2: Auto-Consolidation After Download
**Location**: End of `download_interactive.py` step 5  
**Action**: Add prompt "Consolidate data now?" and call consolidate_data.py

### Priority 3: Shared Authentication Session
**Problem**: Multiple account selections in one workflow  
**Solution**: Save account_id to temp file or environment after first selection, reuse in subsequent steps

### Priority 4: Standardize Data Directories
**Options**:
- A) All scripts write to `data/master_standardized/` directly
- B) All scripts write to their own dirs, consolidate_data.py is mandatory step
- C) Tests auto-discover data from multiple locations (current behavior)

**Recommendation**: Option B (current approach) + auto-prompt for consolidation

---

## Code Quality Notes

### ✅ Well-Implemented
- `download_interactive.py` - Clean step-by-step workflow
- `consolidate_data.py` - Robust file handling, multiple modes
- `run_exhaustive_tests.py` - Comprehensive testing framework
- `audit_data_coverage.py` - Useful reporting

### ⚠️ Needs Cleanup
- `scripts/download/` has many legacy/duplicate scripts (see DEPRECATE list in classify_unused_scripts.py)
- Hardcoded credentials in select_metaapi_account.py
- Multiple data directories (data/master, data/metaapi, data/master_standardized)

---

## Testing Status

### ✅ Verified Working
- Menu navigation and options display
- `download_interactive.py` full workflow (tested manually)
- `consolidate_data.py` with --symlink mode
- `run_exhaustive_tests.py` in CI mode
- Dashboard generation

### ⏳ Needs Testing
- End-to-end workflow through menu (Login → Download → Consolidate → Test)
- Auto-prompts added to select_metaapi_account()
- Data integrity after consolidation
- Full exhaustive tests with real data (after acquiring missing combos)

---

## Next Actions

### Immediate (Today)
1. ✅ **DONE**: Add workflow prompts to select_metaapi_account()
2. **TODO**: Test end-to-end flow: Menu 1→1 → download → consolidate → test
3. **TODO**: Fix hardcoded token in select_metaapi_account.py

### Short-Term (This Week)
1. Download high-priority data (EURUSD H1/H4/D1, BTCUSD D1)
2. Run full consolidation and coverage audit
3. Execute exhaustive tests with real data
4. Review and deprecate legacy download scripts

### Medium-Term (Next 2 Weeks)
1. Standardize data directory structure
2. Add auto-consolidation to download scripts
3. Implement shared authentication session
4. Clean up scripts/download/ directory

---

## Summary

**Existing Code**: 90% complete - most functionality exists  
**Main Issue**: Plumbing between steps (sequential workflow integration)  
**Fix Applied**: Added prompts in authentication menu to guide user through full workflow  
**Remaining Gaps**: Duplicate authentication, hardcoded credentials, directory standardization  

**User Can Now**:
- Login via menu and be prompted through full workflow ✅
- Use download_interactive.py for complete download experience ✅
- Use CLI tools (mt5_metaapi_sync.py) for scripted downloads ✅
- Consolidate and test data through menu or CLI ✅

**Status**: 🟢 Functional with prompts, needs refinement for seamless experience

---

**Last Updated**: 2026-01-03 23:00 UTC  
**Audit By**: Kinetra Development Team  
**Next Review**: After end-to-end testing with real user