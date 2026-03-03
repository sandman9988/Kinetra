#!/usr/bin/env python3
"""
Run A/B Lot Sizing Test with Sample Data
=========================================
"""

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from kinetra.friction_cost import get_calculator
from scripts.renko.ab_lot_sizing_test import compute_dsp_brick_size, print_summary, run_ab_test


def main():
    csv_file = PROJECT_ROOT / "XAUUSD_M1_accurate.csv"

    if not csv_file.exists():
        print(f"Error: {csv_file} not found")
        return 1

    print(f"Loading XAUUSD M1 data from {csv_file.name}...")

    try:
        # Read with proper datetime parsing
        df = pd.read_csv(csv_file, parse_dates=[0], index_col=0)

        # Handle various column name possibilities
        if "close" in df.columns:
            closes = df["close"]
        elif "Close" in df.columns:
            closes = df["Close"]
        elif len(df.columns) > 0:
            closes = df.iloc[:, 0]  # First column
        else:
            print("Error: No price data found in CSV")
            return 1

        closes = closes.astype(float).dropna()

        if len(closes) < 100:
            print(f"Warning: Only {len(closes)} bars available (need ≥100)")
            if len(closes) < 10:
                print("  Cannot run meaningful test with so few bars")
                return 1

        print(f"  Loaded {len(closes)} bars")
        print(f"  Price range: {closes.min():.2f} - {closes.max():.2f}")

    except Exception as e:
        print(f"Error reading CSV: {e}")
        return 1

    # Compute DSP brick
    print("\nComputing DSP brick size...")
    dsp_brick = compute_dsp_brick_size(closes, "M1")
    print(f"  DSP Brick: {dsp_brick:.4f}")

    # Static arbitrary brick (1.5x DSP as reference)
    static_brick = dsp_brick * 1.5
    print(f"  Static Brick (1.5× DSP): {static_brick:.4f}")

    # Get friction calculator
    try:
        calc = get_calculator("XAUUSD")
        print("  Friction calculator loaded")
    except Exception as e:
        print(f"  Warning: Friction calculator unavailable ({e})")
        calc = None

    # Run A/B test
    try:
        result_a, result_b = run_ab_test(
            symbol="XAUUSD",
            closes=closes,
            dsp_brick_size=dsp_brick,
            static_brick_size=static_brick,
            calc=calc,
            timeframe="M1",
        )

        # Print summary
        print_summary(result_a, result_b)

        return 0

    except Exception as e:
        print(f"Error running test: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
