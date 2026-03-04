# MetaAPI Quick Start Guide

Get MT5 historical data in 5 minutes.

## 📋 Prerequisites Checklist

- [ ] Python 3.8+
- [ ] Internet connection
- [ ] Home/Mobile internet (not VPN/cloud IP) for signup only

## 🚀 5-Minute Setup

### Step 1: Install Dependencies (30 seconds)

```bash
cd Kinetra
pip install metaapi-cloud-sdk pandas numpy
```

### Step 2: Sign Up for MetaAPI (2 minutes)

**CRITICAL:** Use home Wi-Fi or mobile hotspot (not VPN/cloud)

1. Go to: https://app.metaapi.cloud/sign-up
2. Create account with email
3. Click confirmation email

### Step 3: Add MT5 Account (2 minutes)

**Option A - MetaQuotes Demo (Easiest):**
1. In MetaAPI dashboard → "Add Account"
2. Select "MetaQuotes Demo"
3. Wait ~2 minutes for deployment

**Option B - Your Broker:**
1. Have MT5 demo/live credentials
2. Add account in dashboard
3. Enter server, login, password

### Step 4: Get API Credentials (30 seconds)

1. Dashboard → **API Tokens** → copy token
2. Dashboard → **Accounts** → copy account ID

### Step 5: Set Environment Variables (30 seconds)

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

### Step 6: Download Data (1 minute)

```bash
python3 scripts/mt5_metaapi_sync.py \
  --init \
  --symbol EURUSD \
  --timeframe H1 \
  --years 2
```

**Output:**
```
🔌 Connecting to MetaAPI...
✅ Connected to: MetaQuotes-Demo
📊 Downloading EURUSD H1 from 2022-12-30 to 2024-12-30
  ✓ Got 8760 candles (2022-12-30 to 2023-12-30)
  ✓ Got 8760 candles (2023-12-30 to 2024-12-30)
✅ Total candles downloaded: 17520
💾 Saved 17520 bars to: EURUSD_H1_history.csv
💾 Metadata saved: EURUSD_H1_metadata.json

✅ Initial download complete!
   Bars: 17520
   Period: 2022-12-30 to 2024-12-30
```

### Step 7: Use the Data

```bash
python3 examples/use_metaapi_data.py --symbol EURUSD --timeframe H1
```

## 📅 Daily Auto-Sync (Optional)

Add to crontab for daily updates at 2 AM:

```bash
crontab -e
```

Add this line:
```
0 2 * * * cd /home/user/Kinetra && python3 scripts/mt5_metaapi_sync.py --sync --symbol EURUSD --timeframe H1
```

## 🎯 Common Use Cases

### Download Multiple Symbols

```bash
# Download EURUSD, GBPUSD, BTCUSD
for symbol in EURUSD GBPUSD BTCUSD; do
  python3 scripts/mt5_metaapi_sync.py --init --symbol $symbol --timeframe H1 --years 2
done
```

### Sync All Daily

```bash
python3 scripts/mt5_metaapi_sync.py \
  --sync-all \
  --symbols EURUSD,GBPUSD,BTCUSD \
  --timeframes H1,H4
```

### Load in Python

```python
import pandas as pd

df = pd.read_csv('data/metaapi/EURUSD_H1_history.csv',
                 index_col='time', parse_dates=True)

print(f"{len(df)} bars from {df.index[0]} to {df.index[-1]}")
```

## 🐛 Troubleshooting

| Problem | Solution |
|---------|----------|
| "IP restricted" on signup | Use home/mobile IP, not VPN |
| "Connection failed" | Check credentials are set: `echo $METAAPI_TOKEN` |
| "No data received" | Verify symbol name (e.g., `EURUSD` not `EUR/USD`) |
| Network timeout | Built-in retry logic - wait for completion |

## 📚 Next Steps

- ✅ Read full documentation: `docs/METAAPI_SETUP.md`
- ✅ Try ML example: `examples/use_metaapi_data.py`
- ✅ Integrate with your trading strategy
- ✅ Set up automated daily sync via cron

## ✅ You're Done!

You now have:
- ✅ 2 years of EURUSD H1 data
- ✅ Script for daily sync
- ✅ Example ML pipeline
- ✅ Production-ready data pipeline

Happy trading! 📈
