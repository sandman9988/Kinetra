# USABLE DATA PERMUTATIONS

**Last Updated:** $(date +%Y-%m-%d)

## Current Reality

### ✅ USABLE COMBINATIONS (27 total)

**Format:** Symbol | Timeframes | Bar Count

#### Crypto (4 combinations)
- BTCUSD: M15 (67995), M30 (34003), H1 (17004), H4 (4280)

#### Forex (3 combinations)  
- GBPUSD: M15 (49535), H1 (12384), H4 (3096)

#### Indices (8 combinations)
- US30: M15 (46901), M30 (23455), H1 (11732), H4 (3069)
- NAS100: M15 (46905), M30 (23456), H1 (11732), H4 (3068)

#### Metals (8 combinations)
- XAUUSD: M15 (47056), M30 (23535), H1 (11775), H4 (3083)
- XAGUSD: M15 (22891), M30 (11448), H1 (5728), H4 (1500)

#### Commodities (4 combinations)
- UKOIL: M15 (43769), M30 (21888), H1 (10950), H4 (3077)

### ❌ MISSING COMBINATIONS (33 total)

**Cannot use these until downloaded:**

- BTCUSD D1
- ETHUSD: ALL timeframes (M15, M30, H1, H4, D1)
- EURUSD: ALL timeframes (M15, M30, H1, H4, D1)
- GBPUSD: M30, D1
- USDJPY: ALL timeframes (M15, M30, H1, H4, D1)
- US30, NAS100: D1 only
- SPX500: ALL timeframes (M15, M30, H1, H4, D1)
- XAUUSD, XAGUSD: D1 only
- USOIL: ALL timeframes (M15, M30, H1, H4, D1)
- UKOIL: D1 only

## Test Permutation Matrix

### What Can We Test RIGHT NOW?

**Per-Symbol Tests:**
- 6 symbols × 1-4 timeframes each = 27 unique combinations

**Per-Timeframe Tests:**
- M15: 6 symbols (BTCUSD, GBPUSD, US30, NAS100, XAUUSD, XAGUSD, UKOIL)
- M30: 5 symbols (BTCUSD, US30, NAS100, XAUUSD, XAGUSD, UKOIL)
- H1: 6 symbols (BTCUSD, GBPUSD, US30, NAS100, XAUUSD, XAGUSD, UKOIL)
- H4: 6 symbols (BTCUSD, GBPUSD, US30, NAS100, XAUUSD, XAGUSD, UKOIL)
- D1: 0 symbols ❌

**Per-Asset-Class Tests:**
- Crypto: 1 symbol (BTCUSD) × 4 TFs = 4 combinations
- Forex: 1 symbol (GBPUSD) × 3 TFs = 3 combinations
- Indices: 2 symbols × 4 TFs = 8 combinations
- Metals: 2 symbols × 4 TFs = 8 combinations
- Commodities: 1 symbol (UKOIL) × 4 TFs = 4 combinations

**Cross-Asset Tests:**
- Same TF, different assets: 4 asset classes with overlap
- Same asset, different TFs: Up to 4 TFs per symbol

### What Scripts Support What?

**Scripts that work with CURRENT data (27 combinations):**

1. `scripts/batch_backtest.py`
   - Can run: Single symbol, single TF
   - Can run: All TFs for one symbol
   - Can run: All symbols for one TF
   - Can run: ALL 27 combinations

2. `scripts/run_exhaustive_tests.py`
   - Requires: data/master_standardized/*.csv
   - Tests: All usable combinations found
   - Modes: --ci-mode (fast) or --full (exhaustive)

3. `scripts/analysis/superpot_*.py`
   - Can test across all 27 combinations
   - Prunes worst features empirically

**Scripts that CANNOT run yet (need missing data):**

- Any script requiring EURUSD (0/5 TFs available)
- Any script requiring D1 timeframe (0/12 symbols)
- Any multi-asset tests requiring >2 forex pairs

## Download Strategy

### To Complete Coverage (60/60 = 100%):

**Priority 1: Get D1 data for existing symbols (6 downloads)**
```bash
python scripts/mt5_metaapi_sync.py --init --symbol BTCUSD --timeframe D1 --years 3
python scripts/mt5_metaapi_sync.py --init --symbol GBPUSD --timeframe D1 --years 3
python scripts/mt5_metaapi_sync.py --init --symbol US30 --timeframe D1 --years 2
python scripts/mt5_metaapi_sync.py --init --symbol NAS100 --timeframe D1 --years 2
python scripts/mt5_metaapi_sync.py --init --symbol XAUUSD --timeframe D1 --years 3
python scripts/mt5_metaapi_sync.py --init --symbol UKOIL --timeframe D1 --years 2
```

**Priority 2: Add missing major symbols (20 downloads)**
- EURUSD: 5 TFs × 2-3 years each
- ETHUSD: 5 TFs × 2 years each  
- USDJPY: 5 TFs × 2-3 years each
- SPX500: 5 TFs × 2 years each

**Priority 3: Fill gaps in existing symbols (7 downloads)**
- GBPUSD M30, XAGUSD D1, XAUUSD D1, etc.

### Estimated Time/Data:
- Each download: 2-10 minutes depending on years/TF
- Total for 100% coverage: ~33 downloads = 1-3 hours
- Disk space: ~500MB-1GB total

## Script Compatibility Matrix

| Script Type | Requires | Works with 27? | Works with 60? |
|-------------|----------|----------------|----------------|
| batch_backtest.py | Any symbol+TF | ✅ Yes | ✅ Yes |
| run_exhaustive_tests.py | data/master_standardized/ | ✅ Yes | ✅ Yes |
| superpot_*.py | Multiple symbols | ✅ Limited | ✅ Full |
| Multi-asset analysis | >=3 symbols/class | ❌ Partial | ✅ Yes |
| Walk-forward (D1) | Daily data | ❌ No | ✅ Yes |
| Regime analysis | Multiple TFs | ✅ Yes | ✅ Better |

## Menu Design Implications

**Menu MUST:**
1. Show ACTUAL available combinations (27 now, 60 when complete)
2. Let user select from AVAILABLE data only
3. Gracefully skip missing combinations
4. Offer to download missing data on-demand

**Menu should NOT:**
5. Assume any symbol/TF exists
6. Have hardcoded lists that don't match reality
7. Fail silently when data missing

**Example Menu Flow:**
```
Select Symbols:
  [x] BTCUSD (4 TFs available)
  [ ] ETHUSD (0 TFs available - download?)
  [x] GBPUSD (3 TFs available)
  ...

Select Timeframes:
  [x] M15 (6 symbols)
  [ ] M30 (5 symbols)
  [x] H1 (6 symbols)
  [ ] H4 (6 symbols)
  [ ] D1 (0 symbols - download?)

Combinations selected: 12 (BTCUSD M15+H1, GBPUSD M15+H1, ...)
Missing: 0
Ready to run? [Y/n]
```

## Reality Check

**Right now, we can:**
- ✅ Test 6 different symbols
- ✅ Test 4 different timeframes (M15, M30, H1, H4)
- ✅ Test 5 different asset classes
- ✅ Run 27 unique backtests
- ❌ Test NO daily (D1) strategies
- ❌ Test NO EURUSD strategies
- ❌ Compare >1 forex pair

**After full download, we can:**
- ✅ Test 12 symbols
- ✅ Test 5 timeframes (including D1)
- ✅ Run 60 unique backtests
- ✅ Full asset class comparisons
- ✅ Multi-timeframe regime analysis

