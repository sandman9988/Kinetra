# Data Management Consolidation Plan

## Consolidation Strategy

Merging 3 files (2,087 lines) into unified `kinetra/data/` package:

### Source Files
1. **data_manager.py** (764 lines)
   - Broker/account/asset class organization
   - Raw data immutability (append-only)
   - Gap detection
   - Atomic chunked downloads
   
2. **data_management.py** (737 lines)
   - Atomic file operations
   - Test run isolation
   - Feature caching
   - Data deduplication
   - Checksum verification
   
3. **unified_data_manager.py** (590 lines)
   - Download workflows (MetaAPI, MT5, CSV)
   - Testing framework integration
   - Data validation
   - Training/test splits

### Target Package Structure

```
kinetra/data/
├── __init__.py           # Package exports
├── manager.py            # Core DataManager (from data_manager.py)
├── download.py           # DownloadManager (from unified_data_manager.py)
├── integrity.py          # IntegrityChecker (from data_manager.py + unified)
├── cache.py              # CacheManager (from data_management.py)
├── test_isolation.py     # TestRunManager (from data_management.py)
└── atomic_ops.py         # AtomicFileWriter (from data_management.py)
```

### Feature Mapping

| Feature | Source File | Target Module |
|---------|-------------|---------------|
| Raw data organization | data_manager.py | manager.py |
| Atomic downloads | data_manager.py | manager.py |
| Gap detection | data_manager.py | integrity.py |
| Broker/account structure | data_manager.py | manager.py |
| Atomic file operations | data_management.py | atomic_ops.py |
| Test run isolation | data_management.py | test_isolation.py |
| Feature caching | data_management.py | cache.py |
| Checksums | data_management.py | integrity.py |
| Download workflows | unified_data_manager.py | download.py |
| MetaAPI integration | unified_data_manager.py | download.py |
| Testing framework bridge | unified_data_manager.py | manager.py |

### Implementation Steps

1. ✅ Create package structure (`kinetra/data/`)
2. ⏳ Extract atomic operations → `atomic_ops.py`
3. ⏳ Extract cache management → `cache.py`
4. ⏳ Extract test isolation → `test_isolation.py`
5. ⏳ Consolidate integrity checks → `integrity.py`
6. ⏳ Consolidate download logic → `download.py`
7. ⏳ Create core manager → `manager.py`
8. ⏳ Update imports across codebase
9. ⏳ Test consolidated package
10. ⏳ Deprecate old files

### Migration Guide

```python
# OLD
from kinetra.data_manager import DataManager
from kinetra.unified_data_manager import UnifiedDataManager
from kinetra.data_management import AtomicFileWriter, CacheEntry

# NEW
from kinetra.data import DataManager, DownloadManager
from kinetra.data.atomic_ops import AtomicFileWriter
from kinetra.data.cache import CacheManager, CacheEntry
```

### Benefits

1. **Single import**: `from kinetra.data import DataManager`
2. **No duplication**: Each feature in one place
3. **Clear separation**: Download, integrity, caching, testing
4. **Easier testing**: Smaller, focused modules
5. **Better docs**: Each module has specific purpose

### Backward Compatibility

Keep old files as thin wrappers during transition:

```python
# kinetra/data_manager.py (deprecated)
import warnings
from kinetra.data import DataManager

warnings.warn(
    "kinetra.data_manager is deprecated. Use kinetra.data instead.",
    DeprecationWarning,
    stacklevel=2
)

__all__ = ['DataManager']
```

## Status

- [x] Package structure created
- [ ] Atomic operations extracted
- [ ] Cache manager extracted
- [ ] Test isolation extracted
- [ ] Integrity checker consolidated
- [ ] Download manager consolidated
- [ ] Core manager consolidated
- [ ] Imports updated
- [ ] Tests passing
- [ ] Old files deprecated
