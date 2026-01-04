# Data Management Architecture: Statistical Rigor & Reproducibility

**Version:** 1.0.0  
**Date:** 2026-01-04  
**Purpose:** Establish data provenance, statistical rigor, and test reproducibility

---

## Core Philosophy

> **"Never modify master data. All transformations are derived and versioned."**

### Key Principles

1. **Immutability** - Master data is append-only, never modified
2. **Provenance** - Every data file has clear lineage (broker → prepared → test)
3. **Reproducibility** - Tests use frozen snapshots with manifests
4. **Statistical Rigor** - Validation at every transformation step
5. **Isolation** - Test data never contaminates master or prepared data

---

## Data Hierarchy: 3-Tier Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  TIER 1: MASTER DATA (Raw, Unmanipulated)                  │
│  • Per-broker storage                                       │
│  • Append-only (never modify)                              │
│  • Source of truth                                          │
└─────────────────────────────────────────────────────────────┘
                            ↓
            (Standardization, Gap Detection)
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  TIER 2: PREPARED DATA (Processed, Still Per-Broker)       │
│  • Standardized format                                      │
│  • Gaps marked (not filled)                                │
│  • Trading hours validated                                  │
│  • Public holidays flagged                                  │
└─────────────────────────────────────────────────────────────┘
                            ↓
            (Snapshot Creation with Manifest)
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  TIER 3: TEST DATA (Frozen Snapshots)                      │
│  • Immutable archives with timestamps                       │
│  • Manifest tracks: date, source, stats                    │
│  • Used for backtests (reproducible)                       │
│  • Old test data archived (never deleted)                  │
└─────────────────────────────────────────────────────────────┘
```

---

## Directory Structure

```
data/
├── master/                          # TIER 1: Raw data per broker
│   ├── metaapi/                     # MetaAPI broker data
│   │   ├── VantageInternational/   # Broker name
│   │   │   ├── BTCUSD_H1_raw.csv   # Raw format from broker
│   │   │   ├── XAUUSD_H1_raw.csv
│   │   │   └── .manifest.json      # Download metadata
│   │   └── ICMarkets/
│   │       └── ...
│   └── mt5_local/                   # MT5 local terminal data
│       └── BrokerName/
│           └── ...
│
├── prepared/                        # TIER 2: Processed per broker
│   ├── metaapi/
│   │   ├── VantageInternational/
│   │   │   ├── BTCUSD_H1.csv       # Standardized format
│   │   │   ├── BTCUSD_H1.meta.json # Stats, gaps, holidays
│   │   │   ├── XAUUSD_H1.csv
│   │   │   └── XAUUSD_H1.meta.json
│   │   └── ICMarkets/
│   │       └── ...
│   └── mt5_local/
│       └── ...
│
├── test_snapshots/                  # TIER 3: Frozen test data
│   ├── 2026-01-04_universe_all/    # Full universe snapshot
│   │   ├── data/                   # Actual data files
│   │   │   ├── BTCUSD_H1.csv
│   │   │   └── ...
│   │   ├── manifest.json           # What, when, from where
│   │   └── statistics.json         # Statistical fingerprint
│   │
│   ├── 2026-01-04_forex_only/      # Forex-only snapshot
│   │   └── ...
│   │
│   └── 2025-12-01_crypto_baseline/ # Historical baseline
│       └── ...
│
└── archives/                        # OLD test snapshots
    ├── 2025-11-15_baseline/
    └── 2025-10-01_initial/
```

---

## Metadata Schemas

### Master Data Manifest (`.manifest.json`)

```json
{
  "broker": "VantageInternational",
  "source": "MetaAPI",
  "account_id": "e8f8c21a-32b5-40b0-9bf7-672e8ffab91f",
  "download_timestamp": "2026-01-04T10:30:00Z",
  "symbols": [
    {
      "symbol": "BTCUSD",
      "timeframe": "H1",
      "bars": 17004,
      "date_range": {
        "start": "2024-01-02T00:00:00Z",
        "end": "2026-01-04T10:00:00Z"
      },
      "raw_file": "BTCUSD_H1_raw.csv",
      "checksum_sha256": "a3f2c9d8e1b4..."
    }
  ]
}
```

### Prepared Data Metadata (`BTCUSD_H1.meta.json`)

```json
{
  "source_broker": "VantageInternational",
  "source_file": "../../master/metaapi/VantageInternational/BTCUSD_H1_raw.csv",
  "preparation_timestamp": "2026-01-04T11:00:00Z",
  "preparation_version": "1.2.0",
  
  "statistics": {
    "total_bars": 17004,
    "date_range": {
      "start": "2024-01-02T00:00:00Z",
      "end": "2026-01-04T10:00:00Z"
    },
    "trading_hours": {
      "24x7": true,
      "gaps_detected": 12,
      "gap_analysis": "weekends only"
    },
    "data_quality": {
      "nan_count": 0,
      "duplicate_timestamps": 0,
      "invalid_ohlc": 0,
      "zero_volume_bars": 34
    }
  },
  
  "gaps": [
    {
      "start": "2024-01-06T22:00:00Z",
      "end": "2024-01-08T00:00:00Z",
      "reason": "weekend",
      "bars_missing": 26
    }
  ],
  
  "holidays": [
    {
      "date": "2024-12-25",
      "name": "Christmas",
      "market_closed": true
    }
  ],
  
  "transformations_applied": [
    "standardize_column_names",
    "validate_ohlc_logic",
    "detect_gaps",
    "flag_holidays"
  ]
}
```

### Test Snapshot Manifest (`manifest.json`)

```json
{
  "snapshot_id": "2026-01-04_universe_all",
  "created_timestamp": "2026-01-04T12:00:00Z",
  "purpose": "Full universe baseline for Q1 2026 testing",
  "created_by": "scripts/create_test_snapshot.py v1.0.0",
  
  "scope": {
    "brokers": ["VantageInternational"],
    "asset_classes": ["forex", "crypto", "indices", "commodities"],
    "symbols": 48,
    "timeframes": ["M15", "M30", "H1", "H4"]
  },
  
  "source_data": {
    "prepared_data_path": "../../prepared/metaapi/VantageInternational",
    "preparation_date": "2026-01-04"
  },
  
  "files": [
    {
      "symbol": "BTCUSD",
      "timeframe": "H1",
      "file": "data/BTCUSD_H1.csv",
      "bars": 17004,
      "checksum_sha256": "d4e9f1a2b3c5..."
    }
  ],
  
  "statistics_fingerprint": {
    "total_bars": 815441,
    "date_range": {
      "start": "2024-01-02T00:00:00Z",
      "end": "2026-01-04T10:00:00Z"
    },
    "checksum_combined": "f7a8c2d1e4b9..."
  },
  
  "usage": {
    "tests_run": [],
    "immutable": true,
    "archive_date": null
  }
}
```

---

## Statistical Rigor Framework

### 1. Data Validation Pipeline

**Every transformation step validates:**

```python
class DataValidator:
    """Validate data at each tier with statistical checks."""
    
    CHECKS = {
        "TIER_1_MASTER": [
            "check_file_integrity",      # SHA256 matches download
            "check_not_empty",            # > 0 bars
            "check_timestamp_format",     # Valid datetime
            "check_basic_structure",      # Has OHLCV columns
        ],
        
        "TIER_2_PREPARED": [
            "check_no_nan_in_ohlc",       # NaN forbidden in OHLC
            "check_ohlc_logic",           # High ≥ Low, etc.
            "check_no_duplicates",        # Unique timestamps
            "check_monotonic_time",       # Time always increases
            "check_gap_detection",        # Gaps properly flagged
            "check_trading_hours",        # Within expected hours
            "check_statistical_sanity",   # Returns not impossible
        ],
        
        "TIER_3_TEST_SNAPSHOT": [
            "check_immutability",         # File hasn't changed
            "check_manifest_match",       # Files match manifest
            "check_reproducibility",      # Same data = same stats
            "check_statistical_fingerprint", # Checksum matches
        ]
    }
```

### 2. Gap Analysis & Handling

**NEVER fill gaps blindly. Document them:**

```python
class GapAnalyzer:
    """Detect and classify gaps - DO NOT fill."""
    
    GAP_TYPES = {
        "weekend": {
            "expected": True,
            "action": "flag_only",
            "impact": "none"
        },
        "holiday": {
            "expected": True,
            "action": "mark_in_metadata",
            "impact": "low"
        },
        "market_close": {
            "expected": True,
            "action": "validate_trading_hours",
            "impact": "medium"
        },
        "data_missing": {
            "expected": False,
            "action": "alert_and_log",
            "impact": "high"
        },
        "broker_outage": {
            "expected": False,
            "action": "re-download_if_possible",
            "impact": "critical"
        }
    }
```

**Gap Filling Policy:**

- ✅ **Flag gaps** in metadata
- ✅ **Document reason** (weekend, holiday, outage)
- ❌ **NEVER forward-fill** for backtesting (introduces look-ahead bias)
- ❌ **NEVER interpolate** OHLC (creates fake data)
- ✅ **Mark as invalid** in backtest if critical gap exists

### 3. Trading Hours & Public Holidays

**Market-Specific Calendars:**

```python
TRADING_HOURS = {
    "FOREX": {
        "schedule": "24x5",  # 24 hours, 5 days/week
        "closed": ["Saturday", "Sunday"],
        "holidays": "inherit_from_USD"
    },
    "CRYPTO": {
        "schedule": "24x7",  # 24 hours, 7 days/week
        "closed": [],
        "holidays": []
    },
    "US_INDICES": {
        "schedule": "09:30-16:00 EST",
        "closed": ["Saturday", "Sunday"],
        "holidays": "NYSE_calendar",
        "early_close": ["day_before_holiday"]
    },
    "COMMODITIES": {
        "XAUUSD": "23:00-22:00 GMT (Sun-Fri)",
        "UKOIL": "01:00-23:00 GMT (Mon-Fri)"
    }
}

PUBLIC_HOLIDAYS = {
    "USD": [
        {"date": "2024-12-25", "name": "Christmas", "market": "closed"},
        {"date": "2024-07-04", "name": "Independence Day", "market": "closed"},
        {"date": "2024-11-28", "name": "Thanksgiving", "market": "early_close"}
    ],
    "GBP": [...],
    "JPY": [...]
}
```

### 4. Statistical Fingerprinting

**Each dataset has a unique statistical signature:**

```python
def calculate_statistical_fingerprint(df: pd.DataFrame) -> dict:
    """Create reproducible statistical signature."""
    return {
        "bars": len(df),
        "date_range": {
            "start": df['time'].min().isoformat(),
            "end": df['time'].max().isoformat()
        },
        "ohlc_stats": {
            "open_mean": float(df['open'].mean()),
            "high_max": float(df['high'].max()),
            "low_min": float(df['low'].min()),
            "close_std": float(df['close'].std())
        },
        "volume_stats": {
            "mean": float(df['volume'].mean()),
            "median": float(df['volume'].median())
        },
        "gaps": count_gaps(df),
        "checksum": hashlib.sha256(df.to_csv(index=False).encode()).hexdigest()
    }
```

**Reproducibility Check:**

```python
def verify_snapshot_unchanged(snapshot_path: Path) -> bool:
    """Verify test snapshot hasn't been modified."""
    manifest = json.load(open(snapshot_path / "manifest.json"))
    current_fingerprint = calculate_fingerprint(snapshot_path / "data")
    
    return current_fingerprint == manifest['statistics_fingerprint']
```

---

## Test Universe Selector

### Hierarchical Test Combinations

```python
class TestUniverseSelector:
    """Select test data based on hierarchical criteria."""
    
    UNIVERSES = {
        "full": {
            "brokers": "all",
            "asset_classes": "all",
            "symbols": "all",
            "timeframes": "all"
        },
        
        "multi_broker": {
            "brokers": ["VantageInternational", "ICMarkets"],
            "asset_classes": "all",
            "symbols": "common_only"  # Only symbols on both brokers
        },
        
        "single_broker_full": {
            "brokers": ["VantageInternational"],
            "asset_classes": "all",
            "symbols": "all"
        },
        
        "asset_class_forex": {
            "brokers": "all",
            "asset_classes": ["forex"],
            "symbols": "all"
        },
        
        "single_symbol": {
            "brokers": "all",
            "asset_classes": "any",
            "symbols": ["BTCUSD"]
        },
        
        "top_5_per_class": {
            "brokers": ["VantageInternational"],
            "asset_classes": "all",
            "symbols": "top_5_by_volume_per_class"
        }
    }
```

### Test Combinations Matrix

```
┌─────────────────────────────────────────────────────────────────────┐
│ Test Level    │ Brokers      │ Asset Classes │ Symbols   │ Purpose  │
├─────────────────────────────────────────────────────────────────────┤
│ UNIVERSE      │ All          │ All           │ All       │ Full sys │
│ MULTI-BROKER  │ 2+           │ All           │ Common    │ Arbitrag │
│ BROKER        │ 1            │ All           │ All       │ Isolate  │
│ ASSET-CLASS   │ All          │ 1             │ All       │ Regime   │
│ MULTI-SYMBOL  │ All          │ Any           │ 5-10      │ Portfolio│
│ SINGLE-SYMBOL │ All          │ Any           │ 1         │ Deep     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Data Management Scripts

### 1. Master Data Downloader (Enhanced)

```bash
# Download raw data per broker (never overwrites existing)
python scripts/download/download_master_data.py \
    --broker VantageInternational \
    --symbols BTCUSD,XAUUSD,GBPUSD \
    --timeframes H1,H4 \
    --append-only  # Never modifies existing files

# Output: data/master/metaapi/VantageInternational/
```

### 2. Data Preparation Pipeline

```bash
# Transform master → prepared (with validation)
python scripts/data/prepare_data.py \
    --source data/master/metaapi/VantageInternational \
    --output data/prepared/metaapi/VantageInternational \
    --validate-gaps \
    --detect-holidays \
    --generate-metadata

# Output: data/prepared/ with .meta.json files
```

### 3. Test Snapshot Creator

```bash
# Create immutable test snapshot
python scripts/data/create_test_snapshot.py \
    --universe full \
    --name "2026-01-04_universe_all" \
    --source data/prepared \
    --purpose "Q1 2026 baseline"

# Output: data/test_snapshots/2026-01-04_universe_all/
```

### 4. Snapshot Validator

```bash
# Verify snapshot integrity before test
python scripts/data/validate_snapshot.py \
    --snapshot data/test_snapshots/2026-01-04_universe_all \
    --check-fingerprint \
    --check-immutability

# Returns: 0 if valid, 1 if corrupted
```

---

## Statistical Validation Checklist

Before ANY backtest:

- [ ] **Master data validated** - SHA256 matches download
- [ ] **Prepared data validated** - All quality checks pass
- [ ] **Test snapshot created** - Frozen with manifest
- [ ] **Fingerprint verified** - Data matches manifest checksum
- [ ] **Gaps documented** - All gaps flagged in metadata
- [ ] **Holidays flagged** - Public holidays marked
- [ ] **Trading hours validated** - Data within expected hours
- [ ] **Statistical sanity** - Returns, spreads within bounds
- [ ] **Reproducibility confirmed** - Re-running prep gives same result

---

## Archival Policy

### When to Archive Test Snapshots

Archive snapshots when:
1. **New baseline created** - Old snapshot becomes historical reference
2. **Data updated** - Master data significantly changed
3. **3 months old** - Automatic archival for space management
4. **Test complete** - Mark as "test_complete" in manifest

### Archive Structure

```
data/archives/
├── 2025-12-01_baseline/
│   ├── manifest.json         # Original manifest
│   ├── test_results.json     # What tests used this
│   └── data.tar.gz           # Compressed data
└── README.md                 # Archival log
```

**NEVER delete archived data** - disk is cheap, reproducibility is priceless.

---

## Reproducibility Guarantee

**Every backtest must be reproducible:**

```python
# Backtest manifest
{
    "backtest_id": "bt_20260104_btcusd_h1",
    "timestamp": "2026-01-04T14:30:00Z",
    "snapshot": "2026-01-04_universe_all",
    "snapshot_fingerprint": "f7a8c2d1e4b9...",
    "code_version": "git_commit_sha",
    "config": {...},
    "results": {...}
}
```

**Guarantee:** Same snapshot + same code + same config = **identical results**

---

## Example Workflows

### Workflow 1: Add New Broker Data

```bash
# 1. Download master data (append-only)
python scripts/download/download_master_data.py \
    --broker ICMarkets --symbols BTCUSD --timeframes H1

# 2. Prepare data with validation
python scripts/data/prepare_data.py \
    --source data/master/metaapi/ICMarkets \
    --output data/prepared/metaapi/ICMarkets

# 3. Verify prepared data
python scripts/data/validate_prepared_data.py \
    --broker ICMarkets

# 4. Create test snapshot (if needed)
python scripts/data/create_test_snapshot.py \
    --universe multi_broker --name "2026-01-04_ic_vantage"
```

### Workflow 2: Run Backtest with Reproducibility

```bash
# 1. Validate snapshot before use
python scripts/data/validate_snapshot.py \
    --snapshot data/test_snapshots/2026-01-04_universe_all

# 2. Run backtest (uses frozen snapshot)
python scripts/batch_backtest.py \
    --data-snapshot 2026-01-04_universe_all \
    --symbols BTCUSD --timeframes H1

# 3. Results include snapshot fingerprint for reproducibility
```

### Workflow 3: Update Data (Preserve History)

```bash
# 1. Download new data (appends to master)
python scripts/download/download_master_data.py \
    --broker VantageInternational --append-only

# 2. Re-prepare (creates new prepared files)
python scripts/data/prepare_data.py \
    --source data/master/metaapi/VantageInternational \
    --output data/prepared/metaapi/VantageInternational

# 3. Create NEW snapshot (old one archived automatically)
python scripts/data/create_test_snapshot.py \
    --universe full --name "2026-01-05_updated"

# 4. Archive old snapshot
python scripts/data/archive_snapshot.py \
    --snapshot 2026-01-04_universe_all \
    --reason "replaced_by_2026-01-05_updated"
```

---

## Consistency Checks

### Cross-Broker Validation

For multi-broker tests:

```python
def validate_cross_broker_consistency(broker1_data, broker2_data, symbol):
    """Ensure different brokers' data is statistically similar."""
    
    # Check timestamp alignment
    assert are_timestamps_aligned(broker1_data, broker2_data)
    
    # Check price correlation (should be > 0.99 for same symbol)
    corr = np.corrcoef(broker1_data['close'], broker2_data['close'])[0,1]
    assert corr > 0.99, f"Low correlation: {corr}"
    
    # Check spread differences (acceptable range)
    spread1 = (broker1_data['high'] - broker1_data['low']).mean()
    spread2 = (broker2_data['high'] - broker2_data['low']).mean()
    assert abs(spread1 - spread2) / spread1 < 0.1, "Spreads differ significantly"
```

---

## Version Control Integration

```bash
# Track data manifests in git (NOT the actual data)
git add data/master/*/.manifest.json
git add data/prepared/*/*.meta.json
git add data/test_snapshots/*/manifest.json
git commit -m "data: Add snapshot manifest for 2026-01-04 baseline"

# Data files themselves in .gitignore
# Use DVC or similar for actual data version control
```

---

## Key Takeaways

1. **Master = Immutable** - Never modify raw broker data
2. **Prepared = Validated** - All transformations documented
3. **Test = Frozen** - Snapshots guarantee reproducibility
4. **Gaps = Documented** - Never blindly fill, always flag
5. **Holidays = Flagged** - Market closures marked in metadata
6. **Statistics = Fingerprinted** - Every dataset has unique signature
7. **Tests = Hierarchical** - Universe → Broker → Asset → Symbol
8. **Archives = Permanent** - Old test data never deleted
9. **Reproducibility = Guaranteed** - Same data + code = same results

---

**Status:** Framework designed, ready for implementation  
**Next Step:** Implement data management scripts following this architecture