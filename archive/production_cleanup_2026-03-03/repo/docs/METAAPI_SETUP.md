# MetaAPI Historical Data Download & Sync

Complete guide for downloading and maintaining MT5 historical market data using MetaAPI.

## 🎯 What You Can Do

- ✅ Download 2+ years of OHLC candle data for any MT5 symbol
- ✅ Incremental sync (daily/hourly updates without re-downloading everything)
- ✅ Reliable data pipeline with retry logic and partial candle handling
- ✅ Works on any platform (Linux/Mac/Windows) - no MT5 terminal required
- ✅ Free tier supports 10+ symbols with reasonable limits

## 📋 Prerequisites

### 1. Sign Up for MetaAPI (Free Tier)

**IMPORTANT:** You must sign up from a **residential or mobile IP** (not VPN/proxy/cloud server).

1. **Turn off VPN/proxy** if you're using one
2. **Connect to home Wi-Fi or mobile hotspot**
3. Go to: https://app.metaapi.cloud/sign-up
4. Create account with your email

**If you get "IP restricted" error:**
- You're on a VPN, cloud server, or datacenter IP
- Switch to your home internet or mobile hotspot
- Once signed up, you can use MetaAPI from anywhere

### 2. Add Your MT5 Account

Two options:

**Option A: Use MetaQuotes Demo (Easiest)**
1. Go to https://web.metatrader.app/
2. Sign in with demo credentials
3. In MetaAPI dashboard → Add Account
4. Enter your MetaQuotes demo credentials
5. Wait for deployment (~2 minutes)

**Option B: Use Your Broker's MT5**
1. Have MT5 account (demo or live) from any broker
2. In MetaAPI dashboard → Add Account
3. Enter broker server, login, password
4. Wait for deployment

### 3. Get API Credentials

1. In MetaAPI dashboard → **API Tokens**
2. Copy your **API Token**
3. Go to **Accounts** → copy your **Account ID**

### 4. Install Python Package

```bash
pip install metaapi-cloud-sdk pandas numpy
```

### 5. Set Environment Variables

**Linux/Mac:**
```bash
export METAAPI_TOKEN="your-token-here"
export METAAPI_ACCOUNT_ID="your-account-id-here"
```

**Windows (PowerShell):**
```powershell
$env:METAAPI_TOKEN="your-token-here"
$env:METAAPI_ACCOUNT_ID="your-account-id-here"
```

**Permanent (Linux/Mac) - Add to `~/.bashrc` or `~/.zshrc`:**
```bash
echo 'export METAAPI_TOKEN="your-token-here"' >> ~/.bashrc
echo 'export METAAPI_ACCOUNT_ID="your-account-id-here"' >> ~/.bashrc
source ~/.bashrc
```

## 🚀 Usage

### Initial Download (2 Years)

Download 2 years of EURUSD H1 data:

```bash
python3 scripts/mt5_metaapi_sync.py \
  --init \
  --symbol EURUSD \
  --timeframe H1 \
  --years 2
```

**Output:**
- `data/metaapi/EURUSD_H1_history.csv` - OHLCV candle data
- `data/metaapi/EURUSD_H1_metadata.json` - Last sync timestamp

### Daily Sync (Extend Data)

Add new candles since last sync:

```bash
python3 scripts/mt5_metaapi_sync.py \
  --sync \
  --symbol EURUSD \
  --timeframe H1
```

**What happens:**
- Loads last sync timestamp from metadata
- Downloads only new candles since then
- Refreshes last 2 candles (handles partial candles)
- Appends to existing CSV (no duplicates)
- Updates metadata

### Sync Multiple Symbols

```bash
python3 scripts/mt5_metaapi_sync.py \
  --sync-all \
  --symbols EURUSD,GBPUSD,USDJPY,BTCUSD \
  --timeframes H1,H4
```

This syncs all symbol/timeframe combinations (8 files in this example).

## 🤖 Automation (Cron Jobs)

### Daily Sync at 2 AM UTC

**Linux/Mac:**

```bash
# Open crontab editor
crontab -e

# Add this line (replace paths):
0 2 * * * cd /home/user/Kinetra && /usr/bin/python3 scripts/mt5_metaapi_sync.py --sync-all --symbols EURUSD,GBPUSD --timeframes H1 >> /var/log/metaapi_sync.log 2>&1
```

**Verify:**
```bash
crontab -l
```

### Hourly Sync (Every Hour)

```bash
0 * * * * cd /home/user/Kinetra && /usr/bin/python3 scripts/mt5_metaapi_sync.py --sync --symbol EURUSD --timeframe M15
```

## 📊 Data Format

### CSV Output

```csv
time,open,high,low,close,volume,tick_volume,returns,range_pct,body_pct
2023-01-01 00:00:00+00:00,1.0700,1.0710,1.0695,1.0705,1234,5678,0.0005,0.14,0.05
2023-01-01 01:00:00+00:00,1.0705,1.0720,1.0700,1.0715,2345,6789,0.0009,0.19,0.09
...
```

### Metadata JSON

```json
{
  "symbol": "EURUSD",
  "timeframe": "H1",
  "last_sync_time": "2024-12-30T14:00:00+00:00",
  "last_sync_timestamp": 1735570800.0,
  "total_bars": 17520,
  "updated_at": "2024-12-30T15:30:00.123456"
}
```

## 🧪 Using Data in ML/Trading

### Load Data

```python
import pandas as pd

# Load historical data
df = pd.read_csv('data/metaapi/EURUSD_H1_history.csv',
                 index_col='time',
                 parse_dates=True)

print(f"Loaded {len(df)} bars")
print(f"Period: {df.index[0]} to {df.index[-1]}")
print(df.head())
```

### Feature Engineering

```python
# Data already includes:
# - returns: percentage change
# - range_pct: (high-low)/close * 100
# - body_pct: |close-open|/close * 100

# Add custom features
df['sma_20'] = df['close'].rolling(20).mean()
df['volatility'] = df['returns'].rolling(20).std()
df['rsi'] = compute_rsi(df['close'], 14)  # your RSI function

# Remove NaN from rolling calculations
df = df.dropna()
```

### Train/Test Split

```python
from sklearn.model_selection import train_test_split

# Time-series split (no shuffle!)
train_size = int(len(df) * 0.8)
train_df = df.iloc[:train_size]
test_df = df.iloc[train_size:]

print(f"Train: {len(train_df)} bars ({train_df.index[0]} to {train_df.index[-1]})")
print(f"Test: {len(test_df)} bars ({test_df.index[0]} to {test_df.index[-1]})")
```

## ⚙️ Configuration

### Timeframes Supported

- `M1` - 1 minute
- `M5` - 5 minutes
- `M15` - 15 minutes
- `M30` - 30 minutes
- `H1` - 1 hour (recommended for training)
- `H4` - 4 hours
- `D1` - 1 day

### Download Limits

Free tier MetaAPI limits:
- **Max candles per request:** ~10,000-50,000 (broker-dependent)
- **Rate limit:** ~100 requests/minute
- **Symbols:** 10-15 symbols recommended

The script automatically chunks large downloads (e.g., 2 years of M1 = ~20-100 requests).

### Historical Depth

Depends on broker:
- **MetaQuotes Demo:** ~2-3 years for major pairs
- **Live brokers:** Varies (check with your broker)

If you need more history, try:
1. IC Markets MT5 Demo (good historical depth)
2. Pepperstone MT5 Demo
3. Multiple brokers and merge data

## 🛡️ Best Practices

### 1. Handle Partial Candles

The script automatically refreshes the **last 2 candles** on each sync to ensure they're finalized (not still forming).

### 2. UTC Timezone

All timestamps are **UTC** by default. Convert to your timezone if needed:

```python
df.index = df.index.tz_convert('America/New_York')
```

### 3. Data Quality Checks

Built-in validation:
- ✅ Fills NaN values (forward-fill → backward-fill)
- ✅ Ensures positive prices (clips to 0.00001 minimum)
- ✅ Ensures non-negative volume
- ✅ Removes duplicate timestamps

### 4. Version Your Data

Keep snapshots for reproducible experiments:

```bash
# Before major updates
cp -r data/metaapi data/metaapi_2024_12_30
```

### 5. Monitor Sync Logs

Add logging to cron jobs:

```bash
0 2 * * * python3 scripts/mt5_metaapi_sync.py --sync-all >> logs/sync.log 2>&1
```

Check logs:
```bash
tail -f logs/sync.log
```

## 🐛 Troubleshooting

### "IP address restricted" during signup

**Solution:** Sign up from residential IP (home Wi-Fi or mobile hotspot), not VPN/cloud/datacenter.

### "Connection failed" or "Deployment failed"

**Solutions:**
1. Check MetaAPI dashboard - is account deployed?
2. Verify credentials: `echo $METAAPI_TOKEN` and `echo $METAAPI_ACCOUNT_ID`
3. Try re-deploying account in dashboard
4. Check MT5 account is active (demo accounts expire)

### "No data received"

**Solutions:**
1. Check symbol name is correct (e.g., `EURUSD` not `EUR/USD`)
2. Some brokers use suffixes (e.g., `EURUSDm`, `EURUSD.raw`)
3. Try a different symbol (e.g., `XAUUSD`, `BTCUSD`)
4. Verify broker has historical data for that timeframe

### Network timeouts / Rate limits

The script has **built-in retry logic** with exponential backoff:
- Retries: 4 attempts
- Delays: 2s → 4s → 8s → 16s

If still failing:
- Check your internet connection
- MetaAPI may be under maintenance (check status page)
- Free tier may have hit quota (wait 24h or upgrade)

### Duplicate or missing candles

**Solutions:**
1. Delete metadata file and re-sync:
   ```bash
   rm data/metaapi/EURUSD_H1_metadata.json
   python3 scripts/mt5_metaapi_sync.py --sync --symbol EURUSD --timeframe H1
   ```
2. Re-download from scratch:
   ```bash
   rm data/metaapi/EURUSD_H1_*
   python3 scripts/mt5_metaapi_sync.py --init --symbol EURUSD --timeframe H1 --years 2
   ```

## 📈 Advanced Usage

### Custom Output Directory

```bash
python3 scripts/mt5_metaapi_sync.py \
  --init \
  --symbol BTCUSD \
  --timeframe H1 \
  --output data/crypto
```

### Multiple Timeframes for Same Symbol

```bash
# Download H1
python3 scripts/mt5_metaapi_sync.py --init --symbol EURUSD --timeframe H1 --years 2

# Download H4
python3 scripts/mt5_metaapi_sync.py --init --symbol EURUSD --timeframe H4 --years 5

# Sync both daily
python3 scripts/mt5_metaapi_sync.py --sync-all --symbols EURUSD --timeframes H1,H4
```

### Integrate with Your Training Pipeline

```python
import subprocess
import pandas as pd

# Sync data before training
subprocess.run([
    'python3', 'scripts/mt5_metaapi_sync.py',
    '--sync', '--symbol', 'EURUSD', '--timeframe', 'H1'
], check=True)

# Load fresh data
df = pd.read_csv('data/metaapi/EURUSD_H1_history.csv',
                 index_col='time', parse_dates=True)

# Train model
train_model(df)
```

## 🔗 Resources

- **MetaAPI Docs:** https://metaapi.cloud/docs/
- **MetaAPI Dashboard:** https://app.metaapi.cloud/
- **Python SDK:** https://github.com/agiliumtrade-ai/metaapi-python-sdk
- **MT5 Terminal (web):** https://web.metatrader.app/

## 📝 Summary

| Command | Purpose |
|---------|---------|
| `--init` | Initial download (e.g., 2 years) |
| `--sync` | Extend existing data with new candles |
| `--sync-all` | Sync multiple symbols/timeframes |
| `--symbol` | Symbol to trade (e.g., EURUSD) |
| `--timeframe` | Candle size (M1, M5, H1, H4, D1) |
| `--years` | Years of history for --init |
| `--output` | Custom output directory |

**Typical Workflow:**

1. **Sign up** from residential IP → get credentials
2. **Initial download:** `--init --symbol EURUSD --timeframe H1 --years 2`
3. **Daily sync:** `--sync --symbol EURUSD --timeframe H1` (via cron)
4. **Load data** in Python → train models → backtest → deploy

✅ **You now have a production-ready MT5 data pipeline!**
