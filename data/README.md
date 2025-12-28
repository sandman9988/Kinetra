# Data Directory

Place your MT5 CSV exports here.

## Supported Formats
- Tab-delimited (MT5 default export)
- Comma-separated
- Semicolon-separated

## Expected Columns
- Date, Time (or DateTime)
- Open, High, Low, Close
- Volume (or TickVol)

## Usage
```bash
python scripts/run_full_backtest.py data/*.csv
```

