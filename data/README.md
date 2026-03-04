# Data Directory

## Structure

```
data/
├── master/                    # Raw historical data by asset class
│   ├── crypto/               # BTC, ETH, XRP
│   ├── forex/                # AUD, EUR, GBP pairs
│   ├── metals/               # XAU, XAG, XPT, Copper
│   ├── energy/               # Oil, Gas
│   └── indices/              # Indices
│
├── master_standardized/      # Standardized M1 data (active pipeline)
│   └── ctrader/
│       └── pepperstone/      # By broker → category → symbol
│           ├── metals/XAUUSD/
│           └── indices/NAS100/
│
├── renko_qualified/         # DSP qualification results
│   ├── XAUUSD/qualification.json
│   └── NAS100/qualification.json
│
└── backups/                 # Backup data (e.g., e2e)
```

## Active Data Path

The pipeline uses: `data/master_standardized/ctrader/pepperstone/<category>/<symbol>/`

- Category: `metals`, `forex`, `indices`, `energy`, `crypto`
- Symbol: Uppercase, e.g., `XAUUSD`

## Downloading Data

```bash
# Download via cTrader
python scripts/ctrader/download_ctrader_history.py --symbol XAUUSD --days 90 --resume

# Auto-download during backtest
python scripts/renko_engine.py XAUUSD --stage backtest --auto-download
```
