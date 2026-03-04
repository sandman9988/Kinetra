#!/bin/bash
# Debug backtest script with better error handling

cd /home/renierdejager/Projects/Kinetra

echo "Creating output directories..."
mkdir -p outputs/results outputs/logs

echo "Verifying data file exists..."
ls -lh data/master_standardized/ctrader/pepperstone/metals/XAUUSD/XAUUSD_M1_*.csv

echo ""
echo "Testing data loading..."
python3 << 'EOF'
import sys
sys.path.insert(0, '.')
from pathlib import Path
import pandas as pd
from kinetra.data_utils import load_mt5_csv

data_dir = Path("data/master_standardized/ctrader/pepperstone/metals/XAUUSD")
m1_files = list(data_dir.glob("*_M1_*.csv"))

if not m1_files:
    print("❌ No M1 files found!")
    sys.exit(1)

print(f"✓ Found M1 files: {[f.name for f in m1_files]}")

# Load first M1 file
df = load_mt5_csv(str(m1_files[0]))
print(f"✓ Loaded {len(df)} rows")
print(f"✓ Columns: {list(df.columns)}")
print(f"✓ Close col: {'Close' in df.columns or 'close' in df.columns}")
print(f"✓ DateTime col: {'DateTime' in df.columns}")
print(f"✓ Date col: {'Date' in df.columns}")

# Test Series creation
close_col = "Close" if "Close" in df.columns else "close"
datetime_col = None
for col in ["DateTime", "Date", "Time"]:
    if col in df.columns:
        datetime_col = col
        break

if datetime_col:
    series = pd.Series(
        df[close_col].values,
        index=pd.to_datetime(df[datetime_col], utc=True)
    )
    print(f"✓ Series created with {len(series)} values")
    print(f"✓ Index type: {type(series.index)}")
    print(f"✓ Index range: {series.index[0]} to {series.index[-1]}")
else:
    print("❌ No datetime column found!")
    sys.exit(1)

print("\n✓ All checks passed!")
EOF

if [ $? -ne 0 ]; then
    echo ""
    echo "❌ Data loading failed!"
    exit 1
fi

echo ""
echo "Running backtest..."
python scripts/renko_engine.py XAUUSD --stage backtest --months 3

echo ""
echo "✓ Done! Check outputs/ directory for results."
