# Consolidation Plan - No More Duplicates!

## Problem
We created new files instead of enhancing existing ones:
- ❌ `smart_download_menu.py` (new) → Should enhance `download_interactive.py`
- ❌ `scripts/data/parallel_data_prep.py` (new) → Duplicate of `scripts/download/parallel_data_prep.py`
- ❌ `metaapi_bulk_download.py` (enhanced) → Good, but needs version number

## Solution: Enhance Existing, Add Versions

### 1. Enhance `download_interactive.py` → v2.0
- Merge `smart_download_menu.py` features into existing file
- Add version: `__version__ = "2.0.0"`
- Keep all existing functionality
- Add: Top-N selection, auto-prep, CPU optimization

### 2. Enhance `parallel_data_prep.py` → v2.0
- Already exists in `scripts/download/`
- Add version: `__version__ = "2.0.0"`
- Merge our enhanced version (physics features, CPU utils)
- Delete duplicate in `scripts/data/`

### 3. Enhance `metaapi_bulk_download.py` → v2.0
- Already enhanced with progress bars
- Add version: `__version__ = "2.0.0"`
- Document changes in header

### 4. Keep New Modules (Not Duplicates)
- ✅ `kinetra/cpu_utils.py` (new utility, not duplicate)
- ✅ `kinetra/menu_state_tracker.py` (new utility, not duplicate)
- ✅ `kinetra/exploration_lab/` (new feature, not duplicate)

## Version Scheme
```
__version__ = "MAJOR.MINOR.PATCH"

MAJOR: Breaking changes
MINOR: New features (backward compatible)
PATCH: Bug fixes
```

## Action Items
1. Add `__version__` to all modified scripts
2. Merge `smart_download_menu.py` → `download_interactive.py`
3. Delete duplicate `scripts/data/parallel_data_prep.py`
4. Update `scripts/download/parallel_data_prep.py` with enhancements
5. Add changelog comments to each file
