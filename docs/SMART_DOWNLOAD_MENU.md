# Smart Download Menu Documentation

**Last Updated**: 2026-01-04  
**Script**: `scripts/download/smart_download_menu.py`

---

## 🎯 Overview

The Smart Download Menu provides an interactive, flexible interface for downloading market data with intelligent symbol selection, parallel processing, and automatic data preparation.

**Key Features**:
- ✅ Download top N symbols per asset class
- ✅ Start data prep as symbols complete (parallel pipeline)
- ✅ Progress bars with ETA
- ✅ Resume capability
- ✅ Flexible selection (quick start, selective, or full download)

---

## 🚀 Quick Start

### Launch Menu
```bash
python scripts/download/smart_download_menu.py
```

### Recommended First-Time Use
Select **Option 1: Quick Start** for fastest results:
- Top 5 symbols per asset class
- 2 timeframes (H1, H4)
- ~50 files
- Auto data prep enabled
- **Ready to train in 5-10 minutes**

---

## 📋 Menu Options

### 📊 QUICK START OPTIONS

#### 1. Quick Start (Top 5 per class) - **RECOMMENDED**
```
Asset classes: forex, crypto, indices, metals, energy
Symbols per class: 5
Timeframes: H1, H4
Total: ~50 files
Time: 5-10 minutes
```

**Use Case**: Get started quickly, test the system

**Example Symbols**:
- Forex: EURUSD+, GBPUSD+, USDJPY+, AUDUSD+, USDCHF+
- Crypto: BTCUSD, ETHUSD, LTCUSD, XRPUSD, BCHUSD
- Indices: US30, US500, NAS100, GER40, UK100
- Metals: XAUUSD+, XAGUSD+, XPTUSD+, XPDUSD+, COPPER
- Energy: USOIL, UKOIL, NGAS, BRENT, WTI

#### 2. Standard (Top 10 per class)
```
Asset classes: All 5
Symbols per class: 10
Total: ~100 files
Time: 10-20 minutes
```

**Use Case**: Good balance for multi-asset portfolio testing

#### 3. Extended (Top 20 per class)
```
Asset classes: All 5
Symbols per class: 20
Total: ~200 files
Time: 20-40 minutes
```

**Use Case**: Comprehensive coverage for production systems

#### 4. Full Download (All available symbols)
```
Asset classes: All 5
Symbols per class: All available
Total: ~900+ files
Time: 2-4 hours
```

**Use Case**: Maximum coverage, overnight downloads

---

### 🎯 SELECTIVE OPTIONS

#### 5. Forex Only
```
Prompt: How many forex symbols? (1-50)
Example: 10
Result: Top 10 forex pairs (EURUSD+, GBPUSD+, etc.)
```

**Use Case**: Focus on forex trading

#### 6. Crypto Only
```
Prompt: How many crypto symbols? (1-30)
Example: 5
Result: Top 5 crypto pairs (BTCUSD, ETHUSD, etc.)
```

**Use Case**: Cryptocurrency-focused strategies

#### 7. Indices Only
```
Prompt: How many index symbols? (1-20)
Example: 5
Result: Top 5 indices (US30, US500, NAS100, etc.)
```

**Use Case**: Stock index trading

#### 8. Metals Only
```
Prompt: How many metal symbols? (1-15)
Example: 4
Result: Gold, silver, platinum, palladium
```

**Use Case**: Precious metals trading

#### 9. Energy Only
```
Prompt: How many energy symbols? (1-15)
Example: 3
Result: Oil, gas contracts
```

**Use Case**: Commodities/energy trading

---

### ⚙️ ADVANCED OPTIONS

#### 10. Custom Selection
```
Interactive prompts:
  - Asset classes: forex,crypto,metals
  - Symbols per class: 5
  - Timeframes: 1h,4h
  - History (days): 730
  - Concurrency: 32
  - Auto prep: Y
```

**Use Case**: Fine-grained control over download configuration

#### 11. Resume Previous Download
```
Loads previous download manifest
Retries failed/skipped symbols
```

**Use Case**: Continue interrupted downloads

#### 12. View Download Status
```
Shows:
  - Last download date
  - Files downloaded
  - Files skipped
  - Bars per asset class
```

**Use Case**: Check what's already downloaded

---

## 🔄 Parallel Data Prep Pipeline

### What is Auto Prep?

When enabled, the system **automatically prepares data for training** as soon as each symbol completes downloading. This means:

1. **Symbol downloads** → Immediately queued for prep
2. **Physics features extracted** (energy, entropy, momentum)
3. **Data validated** (quality checks)
4. **Ready for training** while other symbols still download

### Benefits

| Without Auto Prep | With Auto Prep |
|-------------------|----------------|
| Download 50 files (10 min) | Download 50 files (10 min) |
| Wait for all to complete | **Start prep immediately** |
| Run data prep (5 min) | **Prep happens in parallel** |
| Total: **15 minutes** | Total: **10 minutes** |

**You can start training as soon as the first few symbols are ready!**

---

## 📊 Symbol Selection Logic

### ECN Preference
```
Available: [EURUSD, EURUSD+, EURUSD.m]
Selected:  EURUSD+  (ECN - tighter spreads)
```

ECN symbols (marked with `+`) have tighter spreads and lower trading costs.

### Top-N Priority
1. **Match from priority list** (EURUSD, GBPUSD, BTCUSD, etc.)
2. **Prefer ECN suffix** (`+`)
3. **Shortest name** (most specific)
4. **Pattern matching** if not in priority list

### Pattern Matching
```python
forex:   ^(EUR|GBP|USD|JPY|CHF|CAD|AUD|NZD){2}\+?$
crypto:  ^(BTC|ETH|LTC|XRP|BCH|ADA|DOT|SOL|BNB)(USD|EUR|JPY)\+?$
indices: ^(US30|US500|NAS100|GER40|UK100|JP225)
metals:  ^(XAU|XAG|XPT|XPD|GOLD|SILVER|COPPER)
energy:  ^(USOIL|UKOIL|WTI|BRENT|NGAS|CL|NG)
```

---

## ⚡ Performance & Optimization

### Concurrency Levels

| Symbols | Recommended Concurrency | Time (Est.) |
|---------|------------------------|-------------|
| 1-20    | 16                     | 5-10 min    |
| 21-50   | 24                     | 10-20 min   |
| 51-100  | 32                     | 20-40 min   |
| 100+    | 32-48                  | 1-4 hours   |

**Higher concurrency** = Faster downloads but more risk of rate limiting

### Parallel Data Prep

| Workers | CPU Usage | Speed    |
|---------|-----------|----------|
| 4       | ~50%      | Standard |
| 8       | ~100%     | Fast     |
| 16      | ~200%     | Very Fast (if 16+ cores) |

**More workers** = Faster prep but higher CPU usage

---

## 📋 Example Workflows

### Workflow 1: Quick Test (Beginner)
```
1. Run: python scripts/download/smart_download_menu.py
2. Select: 1 (Quick Start)
3. Confirm: Y
4. Wait: 5-10 minutes
5. Result: 50 files ready for training
```

### Workflow 2: Forex Strategy Development
```
1. Run menu
2. Select: 5 (Forex Only)
3. Enter: 10 symbols
4. Confirm: Y
5. Wait: 8-15 minutes
6. Result: 20 forex files (10 symbols × 2 timeframes)
```

### Workflow 3: Multi-Asset Portfolio
```
1. Run menu
2. Select: 2 (Standard)
3. Confirm: Y
4. Wait: 15-25 minutes
5. Result: 100 files across all asset classes
6. Start training while waiting (auto prep enabled)
```

### Workflow 4: Custom Configuration
```
1. Run menu
2. Select: 10 (Custom)
3. Asset classes: forex,crypto
4. Symbols per class: 15
5. Timeframes: 1h,4h
6. History: 730 days
7. Concurrency: 24
8. Auto prep: Y
9. Confirm: Y
10. Result: 60 files (15 forex + 15 crypto × 2 TF)
```

---

## 🎓 Physics-First Data Prep

### Features Extracted (All Vectorized)

#### Energy
```python
energy = 0.5 * velocity^2
# Where: velocity = price_change / price
```

**Purpose**: Measure market kinetic energy (momentum magnitude)

#### Entropy
```python
entropy = -Σ(p * log(p))
# Where: p = probability from returns histogram
```

**Purpose**: Measure market disorder/predictability

#### Friction
```python
spread_friction = median(spread_pct)
volatility_friction = median(rolling_std)
```

**Purpose**: Estimate trading costs

#### Momentum
```python
momentum = price * velocity
```

**Purpose**: Directional energy (positive/negative)

### Quality Checks

| Check | Threshold | Action |
|-------|-----------|--------|
| Missing data | < 5% | Clean rows |
| Duplicates | 0 required | Remove |
| OHLC anomalies | 0 required | Remove |
| Total quality | ≥ 0.95 | Accept |
| Minimum bars | ≥ 1000 | Accept |

**Files below thresholds are skipped (not ready for training)**

---

## 📁 Output Structure

### Downloaded Data
```
data/master/
├── forex/
│   ├── EURUSD+_H1_202401050000_202601040000.csv
│   ├── EURUSD+_H4_202401050000_202601040000.csv
│   └── ...
├── crypto/
│   ├── BTCUSD_H1_202401050000_202601040000.csv
│   └── ...
└── download_manifest.json
```

### Prepared Data (Auto Prep)
```
data/prepared/
├── EURUSD+_H1_prepared.csv
├── EURUSD+_H4_prepared.csv
├── BTCUSD_H1_prepared.csv
└── ...
```

### Prepared Data Format
```csv
# Physics-First Prepared Data
# Spread Friction: 0.000123
# Volatility Friction: 0.008456
# Bars: 15234
datetime,open,high,low,close,volume,energy,energy_pct,entropy,entropy_rate,momentum,momentum_pct
2024-01-05 00:00:00,1.0950,1.0955,1.0948,1.0952,1234,0.00012,0.00015,2.456,0.001,0.123,0.145
...
```

---

## 🛡️ Error Handling

### Symbol Not Found
```
⚠️ BCHJPY H1: Symbol not found on broker
Status: SKIPPED (continues with other symbols)
```

### Insufficient Data
```
⚠️ ADAJPY H4: Only 45 bars (need 100+)
Status: SKIPPED
```

### Low Quality
```
⚠️ CHINAH H1: Low quality: 0.87 (need 0.95+)
Status: SKIPPED (not ready for training)
```

### Network Error
```
❌ XAUUSD H1: Network timeout
Status: FAILED (retries exhausted)
```

---

## 📊 Progress Display

### Download Progress
```
📊 Downloads: 45%|████████████          | 27/60 [05:23<06:12, 11.29s/file]

  ✅ BTCUSD H1: 15,234 bars saved
  ✅ EURUSD H4: 8,456 bars saved
  ⚠️ ADAJPY H1: No data available from broker
```

### Data Prep Progress
```
📊 Preparing data: 80%|████████████████  | 40/50 [00:45<00:11, 1.11file/s]

  ✅ BTCUSD H1: 15,234 bars (Q=0.98)
  ✅ EURUSD H4: 8,456 bars (Q=0.97)
  ⚠️ CHINAH H1: Low quality: 0.87 (need 0.95+)
```

---

## 🎯 Best Practices

### 1. Start Small, Scale Up
```
First run: Option 1 (Quick Start - 50 files)
Test training pipeline
Then: Option 2 or 3 for full coverage
```

### 2. Use Auto Prep
```
Always enable auto prep (default: Y)
Start training on early completions
Don't wait for all downloads
```

### 3. Monitor Quality
```
After download, check quality scores
Files with Q < 0.95 are skipped
Review skipped symbols
```

### 4. Resume if Interrupted
```
Option 11: Resume Previous Download
Retries failed/skipped symbols
Avoids re-downloading successful files
```

### 5. Verify Before Training
```
Option 12: View Download Status
Check total bars, quality distribution
Ensure sufficient data per asset class
```

---

## 🚦 Next Steps After Download

### 1. Verify Data
```bash
python scripts/download/smart_download_menu.py
# Select: 12 (View Download Status)
```

### 2. Check Prepared Data
```bash
ls -lh data/prepared/
# Should see *_prepared.csv files
```

### 3. Start Training
```bash
python scripts/training/quick_rl_training.py
# Or use the production menu
```

---

## 📞 Troubleshooting

### Problem: "No data files found"
**Solution**: Run download first (Option 1-9)

### Problem: "Low quality" warnings
**Cause**: Data has gaps, duplicates, or anomalies  
**Solution**: Normal - system automatically skips low-quality data

### Problem: Many symbols skipped
**Cause**: Broker doesn't offer those symbols  
**Solution**: Normal - check download manifest for details

### Problem: Slow download
**Cause**: Rate limiting or low concurrency  
**Solution**: Reduce concurrency (Option 10 - Custom) or wait

### Problem: Connection timeout
**Cause**: Network issues or MetaAPI unavailable  
**Solution**: Check internet connection, retry later

---

## 📖 Related Documentation

- **Download Behavior**: `docs/DOWNLOAD_BEHAVIOR.md`
- **Progress Examples**: `docs/PROGRESS_BAR_EXAMPLE.txt`
- **Enhancement Summary**: `ENHANCEMENT_SUMMARY.md`
- **Quick Reference**: `QUICK_REFERENCE.md`

---

**Status**: ✅ READY TO USE

Start with **Option 1 (Quick Start)** and be training in 10 minutes! 🚀