# MetaAPI Quick Login & Data Download Guide

**Last Updated**: 2026-01-03  
**Status**: ✅ Production Ready

---

## Quick Start — 3 Commands to Get Data

### Option 1: Main Menu (Interactive — RECOMMENDED)

```bash
# Start the Kinetra menu system
python kinetra_menu.py
```

Then:
1. Select **`1`** - Login & Authentication
2. Select **`1`** - Select MetaAPI Account
3. Follow prompts to enter your credentials
4. Select **`5`** - Data Management
5. Select data download option

### Option 2: Direct Script (Command Line)

```bash
# Set your credentials (one-time setup)
export METAAPI_TOKEN="your-token-here"
export METAAPI_ACCOUNT_ID="your-account-id-here"

# Download data for EURUSD H1 (2 years)
python3 scripts/mt5_metaapi_sync.py --init --symbol EURUSD --timeframe H1 --years 2

# Download data for BTCUSD D1 (2 years)
python3 scripts/mt5_metaapi_sync.py --init --symbol BTCUSD --timeframe D1 --years 2
```

---

## Step 1: Get MetaAPI Credentials

### A. Sign Up (Free Tier Available)

1. **Go to**: https://app.metaapi.cloud/
2. **Sign up** (use residential IP, not VPN/datacenter)
3. **Add MT5 account**:
   - Demo account (recommended for testing)
   - Or live account (for production)
4. **Get credentials**:
   - Click on your account → Copy **API Token**
   - Copy **Account ID** from dashboard

### B. Save Credentials

**Option A: Environment Variables** (Recommended)
```bash
# Add to ~/.bashrc or ~/.zshrc
export METAAPI_TOKEN="eyJhbGciOiJSUzUxMiIsInR5cCI6IkpXVCJ9..."
export METAAPI_ACCOUNT_ID="a1b2c3d4-e5f6-7890-abcd-ef1234567890"

# Reload shell
source ~/.bashrc
```

**Option B: `.env` File** (Alternative)
```bash
# Create .env file in Kinetra root
cat > .env << EOF
METAAPI_TOKEN=eyJhbGciOiJSUzUxMiIsInR5cCI6IkpXVCJ9...
METAAPI_ACCOUNT_ID=a1b2c3d4-e5f6-7890-abcd-ef1234567890
EOF
```

---

## Step 2: Download Data

### Main Menu Method (Interactive)

```bash
python kinetra_menu.py
```

**Menu Navigation**:
```
Main Menu
  → [1] Login & Authentication
      → [1] Select MetaAPI Account
          → Enter token when prompted
          → Enter account ID when prompted
          → Save credentials? [y/N]: y
  
  → [5] Data Management
      → [Download option - follow prompts]
```

### Command Line Method (Fast)

#### High-Priority Symbols (Close Data Gaps)

```bash
# EURUSD - Missing H1, H4, D1
python3 scripts/mt5_metaapi_sync.py --init --symbol EURUSD --timeframe H1 --years 2
python3 scripts/mt5_metaapi_sync.py --init --symbol EURUSD --timeframe H4 --years 2
python3 scripts/mt5_metaapi_sync.py --init --symbol EURUSD --timeframe D1 --years 2

# BTCUSD - Missing D1
python3 scripts/mt5_metaapi_sync.py --init --symbol BTCUSD --timeframe D1 --years 2
```

#### Download Multiple Symbols/Timeframes

```bash
# All EUR pairs, H1 timeframe
for symbol in EURUSD EURJPY EURGBP; do
  python3 scripts/mt5_metaapi_sync.py --init --symbol $symbol --timeframe H1 --years 2
done

# BTCUSD all timeframes
for tf in M15 M30 H1 H4 D1; do
  python3 scripts/mt5_metaapi_sync.py --init --symbol BTCUSD --timeframe $tf --years 2
done
```

---

## Step 3: Verify Data

### Check Downloaded Files

```bash
# List downloaded data
ls -lh data/metaapi/

# Should see files like:
# EURUSD_H1.csv
# EURUSD_H4.csv
# BTCUSD_D1.csv
```

### Verify Data Quality

```bash
# Quick data check (first and last 5 rows)
head -5 data/metaapi/EURUSD_H1.csv
tail -5 data/metaapi/EURUSD_H1.csv

# Row count
wc -l data/metaapi/EURUSD_H1.csv
# Expected: ~17,520 rows for 2 years H1 (365 days * 24 hours * 2 years)
```

### Run Coverage Audit

```bash
# Check what data you now have
python scripts/audit_data_coverage.py --show-gaps

# Expected output: Coverage should increase from 45% toward 80%+
```

---

## Step 4: Daily Sync (Incremental Updates)

### Manual Sync

```bash
# Sync single symbol (updates with latest bars)
python3 scripts/mt5_metaapi_sync.py --sync --symbol EURUSD --timeframe H1

# Sync all configured symbols
python3 scripts/mt5_metaapi_sync.py --sync-all
```

### Automated Sync (Cron)

```bash
# Open crontab
crontab -e

# Add this line (runs daily at 2 AM UTC)
0 2 * * * cd /path/to/Kinetra && /path/to/python3 scripts/mt5_metaapi_sync.py --sync-all >> logs/metaapi_sync.log 2>&1
```

---

## Troubleshooting

### Error: "MetaAPI not installed"

```bash
pip install metaapi-cloud-sdk>=27.0.0
# Or
pip install -r requirements.txt
```

### Error: "Token not found"

```bash
# Check environment variables
echo $METAAPI_TOKEN
echo $METAAPI_ACCOUNT_ID

# If empty, set them:
export METAAPI_TOKEN="your-token"
export METAAPI_ACCOUNT_ID="your-account-id"
```

### Error: "Connection timeout"

**Causes**:
- Using VPN or datacenter IP (MetaAPI blocks these)
- Wrong account ID
- Account not synchronized yet

**Solutions**:
1. Use residential IP (turn off VPN)
2. Wait 2-3 minutes for account to synchronize
3. Check account status at https://app.metaapi.cloud/

### Error: "Insufficient data returned"

```bash
# Try smaller date range first
python3 scripts/mt5_metaapi_sync.py --init --symbol EURUSD --timeframe H1 --years 1

# Then extend incrementally
python3 scripts/mt5_metaapi_sync.py --sync --symbol EURUSD --timeframe H1
```

---

## Data Format

### CSV Structure

```csv
timestamp,open,high,low,close,volume
2024-01-01 00:00:00,1.10450,1.10480,1.10420,1.10455,1234
2024-01-01 01:00:00,1.10455,1.10490,1.10440,1.10470,1567
...
```

### Using the Data

```python
# Load data in Python
import pandas as pd

df = pd.read_csv('data/metaapi/EURUSD_H1.csv', parse_dates=['timestamp'])
df.set_index('timestamp', inplace=True)

print(f"Loaded {len(df)} bars")
print(f"Date range: {df.index.min()} to {df.index.max()}")
```

### Integration with Kinetra

Data is automatically discovered if placed in:
- `data/metaapi/` (recommended)
- `data/master_standardized/` (for testing framework)

---

## Quick Reference

### Essential Commands

```bash
# 1. Main menu (interactive)
python kinetra_menu.py

# 2. Initial download
python3 scripts/mt5_metaapi_sync.py --init --symbol SYMBOL --timeframe TF --years 2

# 3. Daily sync
python3 scripts/mt5_metaapi_sync.py --sync --symbol SYMBOL --timeframe TF

# 4. Sync all
python3 scripts/mt5_metaapi_sync.py --sync-all

# 5. Check coverage
python scripts/audit_data_coverage.py --show-gaps
```

### Supported Symbols

**Forex**: EURUSD, GBPUSD, USDJPY, EURJPY, EURGBP, etc.  
**Crypto**: BTCUSD, ETHUSD  
**Indices**: US30, NAS100, SPX500  
**Commodities**: XAUUSD (gold), XAGUSD (silver), USOIL, UKOIL

### Supported Timeframes

- **M1** - 1 minute
- **M5** - 5 minutes
- **M15** - 15 minutes
- **M30** - 30 minutes
- **H1** - 1 hour
- **H4** - 4 hours
- **D1** - Daily

---

## High-Priority Downloads (Close Data Gaps)

Based on current 45% coverage, download these to reach 80%+:

```bash
# Critical gaps (forex primary)
python3 scripts/mt5_metaapi_sync.py --init --symbol EURUSD --timeframe H1 --years 2
python3 scripts/mt5_metaapi_sync.py --init --symbol EURUSD --timeframe H4 --years 2
python3 scripts/mt5_metaapi_sync.py --init --symbol EURUSD --timeframe D1 --years 2

# Critical gaps (crypto primary)
python3 scripts/mt5_metaapi_sync.py --init --symbol BTCUSD --timeframe D1 --years 2

# High-value additions (indices)
python3 scripts/mt5_metaapi_sync.py --init --symbol SPX500 --timeframe H1 --years 2
python3 scripts/mt5_metaapi_sync.py --init --symbol SPX500 --timeframe H4 --years 2
python3 scripts/mt5_metaapi_sync.py --init --symbol SPX500 --timeframe D1 --years 2

# Gold (high volatility, good for testing)
python3 scripts/mt5_metaapi_sync.py --init --symbol XAUUSD --timeframe D1 --years 2
```

After downloading these 8 combinations, re-run coverage audit:

```bash
python scripts/audit_data_coverage.py --show-gaps
# Expected: 60-70% coverage (up from 45%)
```

---

## Next Steps After Data Download

1. **Consolidate Data**:
   ```bash
   python scripts/consolidate_data.py --symlink
   ```

2. **Run Tests**:
   ```bash
   python scripts/run_exhaustive_tests.py --generate-dashboard
   ```

3. **Check Dashboard**:
   - Open `test_results/test_dashboard_*.html` in browser
   - Review performance metrics

4. **Set Up Daily Sync**:
   - Add cron job for automated updates
   - Monitor data quality daily

---

## Documentation References

- **Setup Guide**: `docs/METAAPI_SETUP.md` (comprehensive, 417 lines)
- **Quick Start**: `docs/METAAPI_QUICKSTART.md` (5-minute guide, 159 lines)
- **Usage Example**: `examples/use_metaapi_data.py` (ML pipeline example)
- **Sync Script**: `scripts/mt5_metaapi_sync.py` (main data manager)

---

**Status**: 🟢 Ready to Use  
**Free Tier**: 10+ symbols supported  
**Recommended Start**: EURUSD H1 (most liquid, best for testing)

---

**🚀 Quick Command to Get Started Right Now:**

```bash
# 1. Set credentials (replace with your actual values)
export METAAPI_TOKEN="your-token-here"
export METAAPI_ACCOUNT_ID="your-account-id-here"

# 2. Download your first dataset (EURUSD H1, 2 years)
python3 scripts/mt5_metaapi_sync.py --init --symbol EURUSD --timeframe H1 --years 2

# 3. Verify it worked
ls -lh data/metaapi/EURUSD_H1.csv
head data/metaapi/EURUSD_H1.csv
```

**That's it! You're ready to download real MT5 data.** 🎉