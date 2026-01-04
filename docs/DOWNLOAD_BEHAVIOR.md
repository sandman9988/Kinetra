# Download Behavior Documentation

**Last Updated**: 2026-01-04  
**Applies To**: MetaAPI Bulk Download (`scripts/download/metaapi_bulk_download.py`)

---

## 🎯 Overview

This document explains how the MetaAPI bulk download script handles various scenarios, including "0 bars" situations, auto-retry logic, and failure categorization.

---

## 📊 Download Status Categories

### ✅ **SUCCESS** - Downloaded Successfully
- **Condition**: Downloaded ≥ 100 bars
- **Action**: Data saved to `data/master/{asset_class}/{symbol}_{tf}_{start}_{end}.csv`
- **Progress**: Counted in successful downloads
- **Message**: `✅ {symbol} {tf}: {count} bars saved`

### ⏭️ **UP_TO_DATE** - Already Current
- **Condition**: Existing file is within 24 hours of requested end date
- **Action**: Skip download, keep existing file
- **Progress**: Not counted as download or failure
- **Message**: `Already up to date (latest: {date})`
- **Reason**: Avoids redundant API calls for recently downloaded data

### ⚠️ **SKIPPED** - No Data Available
- **Condition**: 0 bars returned from broker OR < 100 bars after retries
- **Action**: Skip symbol, don't save file
- **Progress**: Counted separately from failures
- **Message**: `⚠️ {symbol} {tf}: {reason}`
- **Common Reasons**:
  - `No data available from broker (symbol may not exist or no history)`
  - `Insufficient data: only {n} bars (need 100+)`
  - `Symbol not found on broker`
  - `No new data (existing: {n} bars)`

### ❌ **FAILED** - Error During Download
- **Condition**: Exception/error during download process
- **Action**: Skip symbol, log error
- **Progress**: Counted as failed
- **Message**: `❌ {symbol} {tf}: {error}`
- **Common Reasons**:
  - Network errors
  - API errors (non-rate-limit)
  - Data processing errors

---

## 🔄 Auto-Retry Logic

### Symbol-Level Retry (Up to 3 Attempts)
```
Attempt 1 → 0 bars → Wait 2s → Retry
Attempt 2 → 0 bars → Wait 2s → Retry
Attempt 3 → 0 bars → SKIP (no data available)
```

**Why Retry?**
- Temporary broker glitches
- API synchronization delays
- Network hiccups

**Why Stop After 3?**
- Avoid wasting API quota
- Prevent infinite loops
- Symbol likely doesn't exist or has no history

### Chunk-Level Retry (Up to 5 Attempts Per Chunk)
```
Chunk request → Rate limit (429) → Wait 1s → Retry
Chunk request → Rate limit (429) → Wait 2s → Retry
Chunk request → Rate limit (429) → Wait 4s → Retry
Chunk request → Success → Continue
```

**Handles**:
- Rate limiting (429 errors)
- Temporary network errors
- Transient API issues

**Exponential Backoff**: `wait_time = 1s × 2^retry_number`

### Empty Chunk Detection
```
Chunk 1 → 1000 bars → Continue
Chunk 2 → 0 bars → Count=1, skip ahead 7 days
Chunk 3 → 0 bars → Count=2, skip ahead 7 days
Chunk 4 → 0 bars → Count=3, STOP (no more data)
```

**Why Stop After 3 Empty Chunks?**
- Avoid scanning entire history for sparse data
- Indicates end of available data reached
- Prevents unnecessary API calls

---

## 📋 What "0 Bars" Means

### Scenario 1: **Symbol Not Offered by Broker**
```
Example: BCHJPY on Vantage (may not be offered)
Result: 0 bars → SKIPPED
Reason: "No data available from broker"
Action: None - symbol not tradeable on this account
```

### Scenario 2: **Symbol Exists, No Historical Data**
```
Example: Newly listed instrument
Result: 0 bars → SKIPPED
Reason: "No data available from broker (symbol may not exist or no history)"
Action: Try again in the future when history accumulates
```

### Scenario 3: **Already Up to Date**
```
Example: Downloaded yesterday, requested today
Result: 0 NEW bars (existing file has data)
Reason: "No new data (existing: 5000 bars)"
Status: UP_TO_DATE (not SKIPPED)
Action: Existing file retained, no download needed
```

### Scenario 4: **Temporary Broker Issue**
```
Example: Broker maintenance window
Result: 0 bars on attempt 1 → RETRY
Action: Wait 2s, retry up to 3 times
Final: If still 0 bars → SKIPPED
```

### Scenario 5: **Symbol Name Mismatch**
```
Example: BTCUSD vs BTCUSD+ vs BTCUSD.m
Result: 0 bars for wrong suffix → SKIPPED
Reason: "Symbol not found on broker"
Action: Script auto-discovers correct suffix (ECN preferred)
```

---

## 🎯 Smart Symbol Discovery

### ECN Preference
```
Available symbols: [EURUSD, EURUSD+, EURUSD.m]
Selection: EURUSD+ (ECN - tighter spreads)
Suffix: + indicates ECN pricing
```

### Pattern Matching
```
forex:   ^(EUR|GBP|USD|JPY|CHF|CAD|AUD|NZD){2}\+?$
crypto:  ^(BTC|ETH|LTC|XRP|BCH|ADA)(USD|EUR|JPY)\+?$
indices: ^(US30|US500|NAS100|GER40|UK100|JP225)
metals:  ^(XAU|XAG|XPT|XPD|GOLD|SILVER)
energy:  ^(USOIL|UKOIL|WTI|BRENT|NGAS|CL|NG)
```

### Fallback Strategy
```
1. Try exact match: BTCUSD
2. Try ECN suffix: BTCUSD+
3. Try broker suffix: BTCUSD.m
4. Try partial match: *BTCUSD*
5. If none found → SKIP with "Symbol not found"
```

---

## 📈 Progress Bar Display

### New Format (with `tqdm`)
```
📊 Downloads: 45%|████████████          | 27/60 [05:23<06:12, 11.29s/file]
📈 Bars: 125,432 bars @ 388.5bar/s
```

**Benefits**:
- Real-time ETA (estimated time remaining)
- Accurate rate calculation
- No flickering heartbeat symbols
- Clean, professional output

### Old Format (heartbeat)
```
💓 [27/60] 45% | 🔄32 parallel | 125,432 bars @ 389/sec | BTCUSD H1 chunk 5
```

**Issues**:
- Cluttered display
- No ETA
- Heartbeat symbol distracting
- Hard to read in logs

---

## 🛡️ Soft-Block vs Hard-Fail

### Soft-Block (SKIP) - Recommended ✅
**When**: Symbol has no data available
**Action**: Skip symbol, continue with others
**Rationale**:
- Doesn't stop entire download batch
- Other symbols may still succeed
- Can retry later (broker may add data)
- Logged in manifest for review

### Hard-Fail (ERROR) - Avoided ❌
**When**: Critical system error (disk full, API auth fail)
**Action**: Stop entire batch
**Rationale**:
- Only for errors that affect ALL downloads
- Prevents data corruption
- User intervention required

### Decision Tree
```
0 bars received
├─ Symbol not found → SKIP (soft-block)
├─ Rate limit → RETRY (with backoff)
├─ Network error → RETRY (up to 3x)
├─ After 3 retries still 0 bars → SKIP (soft-block)
└─ API auth error → FAIL (hard-fail, stop batch)
```

---

## 📝 Download Manifest

### Location
```
data/master/download_manifest.json
```

### Schema
```json
{
  "downloaded_at": "2026-01-04T12:00:00",
  "broker": "Vantage",
  "total_files": 45,
  "up_to_date": 10,
  "skipped": 12,
  "failed": 3,
  "total_bars": 1250000,
  "symbols": [
    {
      "status": "success",
      "symbol": "BTCUSD",
      "tf": "H1",
      "bars": 15000,
      "asset_class": "crypto",
      "file": "data/master/crypto/BTCUSD_H1_20240105_20260104.csv"
    }
  ],
  "skipped_symbols": [
    {
      "status": "skipped",
      "symbol": "BCHJPY",
      "tf": "H4",
      "bars": 0,
      "reason": "No data available from broker"
    }
  ],
  "up_to_date_symbols": [
    {
      "status": "up_to_date",
      "symbol": "EURUSD",
      "tf": "H1",
      "bars": 5000,
      "reason": "Already up to date (latest: 2026-01-03)"
    }
  ],
  "failed_symbols": [
    {
      "symbol": "UNKNOWN",
      "tf": "H1",
      "reason": "Network timeout"
    }
  ]
}
```

---

## 🔍 Troubleshooting

### Problem: Many symbols showing "0 bars"

**Diagnosis Steps**:
1. Check manifest: `cat data/master/download_manifest.json`
2. Look for common reason:
   - All same reason → Systematic issue
   - Different reasons → Symbol-specific

**Common Causes**:
| Reason | Cause | Solution |
|--------|-------|----------|
| "Symbol not found" | Broker doesn't offer it | Remove from symbol list |
| "No data available" | New symbol, no history yet | Wait for history to accumulate |
| "Already up to date" | Recently downloaded | Normal - no action needed |
| "Insufficient data" | < 100 bars available | Wait for more history |

### Problem: All downloads skipped

**Check**:
1. MetaAPI account status: `python scripts/download/test_metaapi_connection.py`
2. Account is DEPLOYED (not UNDEPLOYED)
3. Subscription active (not expired)
4. Token has correct permissions

### Problem: Slow download rate

**Causes**:
| Symptom | Cause | Solution |
|---------|-------|----------|
| `⚡ Rate limit detected` | Too many concurrent requests | Reduce `MAX_CONCURRENT_DOWNLOADS` |
| All downloads slow | Broker throttling | Add delays between chunks |
| Single symbol slow | Large history | Normal - wait for completion |

---

## 🎓 Best Practices

### 1. Start with Small Test
```bash
# Test with 1 symbol first
python scripts/download/metaapi_bulk_download.py --symbols BTCUSD --timeframes H1
```

### 2. Review Manifest
```bash
# After download, check what was skipped
cat data/master/download_manifest.json | jq '.skipped_symbols'
```

### 3. Retry Skipped Later
```bash
# Some symbols may become available later
# Re-run download after a week/month
```

### 4. Monitor Rate Limits
```bash
# If you see rate limit warnings:
# Reduce MAX_CONCURRENT_DOWNLOADS in .env:
echo "KINETRA_MAX_NETWORK_WORKERS=16" >> .env
```

### 5. Keep Manifest History
```bash
# Archive old manifests before re-downloading
mv data/master/download_manifest.json \
   data/master/download_manifest_$(date +%Y%m%d).json
```

---

## 📊 Expected Results

### Typical Download (Vantage Demo)
```
Total symbols: 30
Timeframes: 2 (H1, H4)
Total files attempted: 60

Results:
  ✅ Downloaded: 40-45 files (67-75%)
  ⏭️  Up to date: 0 files (first run)
  ⚠️  Skipped: 10-15 files (17-25%)
  ❌ Failed: 0-5 files (0-8%)

Common skipped symbols:
  - Exotic crypto pairs (ADAJPY, BCHJPY)
  - Niche indices (CHINAH)
  - Uncommon energy symbols
```

### Re-run Same Day
```
Total files attempted: 60

Results:
  ✅ Downloaded: 5-10 files (new data since last run)
  ⏭️  Up to date: 35-40 files (no new bars)
  ⚠️  Skipped: 10-15 files (still no data)
  ❌ Failed: 0 files
```

---

## 🚀 Performance Tips

### Optimize Concurrency
```python
# In .env file:
KINETRA_MAX_NETWORK_WORKERS=24  # Default: 32

# Lower = slower but safer (less rate limiting)
# Higher = faster but more rate limit risk
```

### Optimize Time Range
```python
# Download 1 year instead of 2
# Modify in script:
start_time = end_time - timedelta(days=365)  # Instead of 730
```

### Optimize Symbol Selection
```python
# Only download symbols you'll actually trade
# Edit PREFERRED_SYMBOLS dict in script
```

---

## 📞 Need Help?

1. **Check Logs**: `tail -f logs/download.log` (if logging enabled)
2. **Test Connection**: `python scripts/download/test_metaapi_connection.py`
3. **Review Manifest**: `cat data/master/download_manifest.json`
4. **Check This Doc**: `docs/DOWNLOAD_BEHAVIOR.md` (you are here)
5. **Report Issue**: Include manifest and log excerpt

---

**End of Download Behavior Documentation**