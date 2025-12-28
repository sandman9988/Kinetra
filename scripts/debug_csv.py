#!/usr/bin/env python3
"""Debug CSV loading."""
import sys
import pandas as pd
import numpy as np

filepath = sys.argv[1] if len(sys.argv) > 1 else None
if not filepath:
    print("Usage: python debug_csv.py <csv_file>")
    sys.exit(1)

print(f"Loading: {filepath}")

# Step 1: Raw read
print("\n1. Raw read with tab separator:")
df = pd.read_csv(filepath, sep='\t', encoding='utf-8-sig', engine='python')
print(f"   Columns: {list(df.columns)}")
print(f"   Shape: {df.shape}")
print(f"   First row: {df.iloc[0].tolist()}")

# Step 2: Clean column names
print("\n2. Cleaning column names:")
def clean_col(c):
    c = str(c).strip().lower().replace('\r', '')
    if c.startswith('<') and c.endswith('>'):
        c = c[1:-1]
    return c

df.columns = [clean_col(c) for c in df.columns]
print(f"   Columns: {list(df.columns)}")

# Step 3: Map columns
print("\n3. Mapping columns:")
column_map = {
    'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close',
    'tickvol': 'Volume', 'vol': 'Volume', 'date': 'Date', 'time': 'Time',
}
df.columns = [column_map.get(c, c) for c in df.columns]
print(f"   Columns: {list(df.columns)}")

# Step 4: Combine date/time
print("\n4. Combining Date + Time:")
if 'Date' in df.columns and 'Time' in df.columns:
    print(f"   Date sample: {df['Date'].iloc[0]}")
    print(f"   Time sample: {df['Time'].iloc[0]}")
    df['DateTime'] = df['Date'].astype(str) + ' ' + df['Time'].astype(str)
    print(f"   DateTime sample: {df['DateTime'].iloc[0]}")

# Step 5: Parse datetime
print("\n5. Parsing datetime:")
try:
    df['DateTime'] = pd.to_datetime(df['DateTime'], format='%Y.%m.%d %H:%M:%S')
    print(f"   Success! Sample: {df['DateTime'].iloc[0]}")
    df = df.set_index('DateTime')
except Exception as e:
    print(f"   Error: {e}")

# Step 6: Select OHLCV
print("\n6. Selecting OHLCV columns:")
print(f"   Available: {list(df.columns)}")
df = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
print(f"   Selected: {list(df.columns)}")

# Step 7: Convert to numeric
print("\n7. Converting to numeric:")
for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')
print(f"   Dtypes: {df.dtypes.tolist()}")

# Step 8: Drop NaN
df = df.dropna()
print(f"\n8. Final shape: {df.shape}")
print(f"   Index type: {type(df.index)}")
print(f"   First 3 rows:\n{df.head(3)}")

print("\nSUCCESS - Data loaded correctly!")
