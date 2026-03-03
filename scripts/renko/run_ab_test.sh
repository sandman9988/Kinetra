#!/usr/bin/env python3
"""
Quick A/B Lot Sizing Test Runner
=================================

Runs the A/B test with the XAUUSD_M1_accurate.csv file.
"""

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Use the existing XAUUSD M1 data
csv_file = PROJECT_ROOT / "XAUUSD_M1_accurate.csv"

if not csv_file.exists():
    print(f"Error: {csv_file} not found")
    sys.exit(1)

print(f"Running A/B lot sizing test with {csv_file.name}...")
print()

cmd = [
    sys.executable,
    str(PROJECT_ROOT / "scripts" / "renko" / "ab_lot_sizing_test.py"),
    "--symbol", "XAUUSD",
    "--timeframe", "M1",
    "--csv", str(csv_file),
    "--static-brick", "2.0",
]

result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
sys.exit(result.returncode)
