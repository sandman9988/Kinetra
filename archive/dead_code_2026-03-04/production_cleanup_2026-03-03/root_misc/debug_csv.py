#!/usr/bin/env python3
"""Debug script to check CSV columns."""

import sys

sys.path.insert(0, "/home/renierdejager/Projects/Kinetra")

from pathlib import Path

from kinetra.data_utils import load_mt5_csv

csv_path = Path(
    "/home/renierdejager/Projects/Kinetra/data/master_standardized/ctrader/pepperstone/metals/XAUUSD/XAUUSD_M1_accurate.csv"
)

print(f"Loading: {csv_path}")
df = load_mt5_csv(str(csv_path))

print(f"\nDataFrame shape: {df.shape}")
print(f"Columns: {list(df.columns)}")
print("\nFirst few rows:")
print(df.head())
print("\nData types:")
print(df.dtypes)
print(f"\nIndex type: {type(df.index)}")
print(f"Index: {df.index[:5]}")
