#!/usr/bin/env python3
"""
Direct A/B Test with Embedded Results
=====================================

This script loads XAUUSD M1 data and runs the A/B test inline,
printing results directly.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def load_sample_data():
    """Load XAUUSD M1 data with fallback to synthetic data."""
    csv_file = PROJECT_ROOT / "XAUUSD_M1_accurate.csv"

    if csv_file.exists():
        try:
            df = pd.read_csv(csv_file, parse_dates=[0], index_col=0, nrows=5000)
            if "close" in df.columns:
                closes = df["close"].astype(float).dropna()
            elif "Close" in df.columns:
                closes = df["Close"].astype(float).dropna()
            else:
                closes = df.iloc[:, 0].astype(float).dropna()

            print(f"✓ Loaded {len(closes)} bars from {csv_file.name}")
            return closes
        except Exception as e:
            print(f"✗ Error reading {csv_file.name}: {e}")

    # Synthetic fallback: Brownian motion + drift
    print("⚠ Using synthetic data (Brownian motion with trend)")
    np.random.seed(42)
    base_price = 2000.0
    returns = np.random.normal(0.0001, 0.002, 5000)
    prices = base_price * np.exp(np.cumsum(returns))

    dates = pd.date_range("2024-01-01", periods=len(prices), freq="min")
    return pd.Series(prices, index=dates, name="close")


def compute_dsp_brick(closes):
    """Simplified DSP brick computation."""
    try:
        from kinetra.renko.dsp import vr_peak, vr_profile

        closes_arr = closes.values.astype(np.float64)

        # Use larger scales for M1 data
        scales = [5, 10, 20, 30, 50, 60, 90, 120, 180]
        profile = vr_profile(closes_arr, scales=scales)

        if profile:
            peak_scale, peak_vr = vr_peak(profile)

            # Compute median displacement at peak scale
            returns = np.diff(np.log(closes_arr))
            n_windows = len(returns) // peak_scale

            if n_windows >= 2:
                displacements = []
                for i in range(n_windows):
                    window_ret = returns[i * peak_scale : (i + 1) * peak_scale]
                    displacement = np.abs(np.sum(window_ret))
                    displacements.append(displacement)

                brick = float(np.median(displacements))
                print(f"✓ DSP computed: VR peak at scale {peak_scale}, brick={brick:.6f}")
                return max(brick, 0.01)
    except Exception as e:
        print(f"⚠ DSP computation failed: {e}")

    # Fallback: use rolling standard deviation as brick estimate
    returns = np.log(closes / closes.shift(1)).dropna()
    vol = returns.std() * 100  # Rough price-based estimate
    brick = max(vol, 0.1)
    print(f"✓ DSP fallback: brick={brick:.6f}")
    return brick


def validate_friction_floor(symbol, brick_size, friction_cost_rt):
    """
    Validate that brick_size meets minimum friction floor.

    Rule: brick_size_usd >= friction_cost_rt × multiplier
    Where multiplier = 4 means friction_ratio = 25% (standard)

    Returns: (is_valid, friction_ratio, required_minimum_brick)
    """
    # For XAUUSD: tick_value_usd = $1.00 per point
    # For other instruments: need to pass tick_value_usd as param
    if "XAU" in symbol.upper():
        usd_per_point = 1.0
    else:
        usd_per_point = 1.0  # fallback

    brick_usd = brick_size * usd_per_point
    friction_ratio = friction_cost_rt / brick_usd if brick_usd > 0 else float("inf")

    # Standard: friction_ratio <= 0.25 (25%)
    # This means: brick >= friction_cost × 4
    min_brick_usd = friction_cost_rt * 4.0
    min_brick_points = min_brick_usd / usd_per_point

    is_valid = friction_ratio <= 0.25

    return is_valid, friction_ratio, min_brick_points


def run_backtest(
    symbol, closes, brick_size, sizing_mode, capital_per_lot=None, friction_cost_rt=14.0
):
    """Run a single backtest."""
    try:
        from kinetra.renko.backtest import (
            FilterParams,
            SizingMode,
            StopParams,
            VolSizingParams,
            backtest_instrument,
        )

        filter_params = FilterParams()
        stop_params = StopParams()

        if sizing_mode == "static":
            result = backtest_instrument(
                symbol=symbol,
                closes=closes,
                brick_size=brick_size,
                filter_params=filter_params,
                stop_params=stop_params,
                session_break_minutes=1.0,
                sizing_mode=SizingMode.FIXED_LOT,
                vol_sizing_params=VolSizingParams(fixed_lot=0.01),
            )
        else:  # compounded
            result = backtest_instrument(
                symbol=symbol,
                closes=closes,
                brick_size=brick_size,
                filter_params=filter_params,
                stop_params=stop_params,
                session_break_minutes=1.0,
                sizing_mode=SizingMode.COMPOUNDING,
                vol_sizing_params=VolSizingParams(
                    fixed_lot=0.01,
                    compounding_capital_per_lot=capital_per_lot or 1000.0,
                    initial_equity=100_000.0,
                ),
            )

        return result

    except Exception as e:
        print(f"✗ Backtest failed: {e}")
        import traceback

        traceback.print_exc()
        return None


def main():
    print("=" * 80)
    print("  A/B LOT SIZING TEST — STATIC vs COMPOUNDED")
    print("=" * 80)

    # Load data
    print("\n[1] Loading data...")
    closes = load_sample_data()
    print(f"    Price: {closes.iloc[0]:.2f} → {closes.iloc[-1]:.2f}")

    # Compute DSP brick
    print("\n[2] Computing DSP brick size...")
    dsp_brick = compute_dsp_brick(closes)

    # Define test scenarios
    static_brick = dsp_brick * 1.5

    print("\n[3] Test Scenarios:")
    print(f"    A: DSP-Arrived brick ({dsp_brick:.6f})")
    print(f"    B: Static arbitrary brick ({static_brick:.6f})")

    # Run backtests
    print("\n[4] Running backtests...")
    print(f"\n    SCENARIO A — DSP-Arrived ({dsp_brick:.6f})")
    print("    ├─ Static 0.01 lot...", end=" ", flush=True)
    result_a_static = run_backtest("XAUUSD", closes, dsp_brick, "static")
    if result_a_static:
        a_static_omega = result_a_static.omega
        a_static_pnl = sum(t.net_usd for t in result_a_static.trades)
        print(f"✓ Ω={a_static_omega:.3f}, P&L=${a_static_pnl:.2f}")
    else:
        a_static_omega = 0.0
        a_static_pnl = 0.0
        print("✗ Failed")

    print("    └─ Compounded 0.01/$1000...", end=" ", flush=True)
    result_a_compounded = run_backtest("XAUUSD", closes, dsp_brick, "compounded", 1000.0)
    if result_a_compounded:
        a_compounded_omega = result_a_compounded.omega
        a_compounded_pnl = sum(t.net_usd for t in result_a_compounded.trades)
        print(f"✓ Ω={a_compounded_omega:.3f}, P&L=${a_compounded_pnl:.2f}")
    else:
        a_compounded_omega = 0.0
        a_compounded_pnl = 0.0
        print("✗ Failed")

    print(f"\n    SCENARIO B — Static Arbitrary ({static_brick:.6f})")
    print("    ├─ Static 0.01 lot...", end=" ", flush=True)
    result_b_static = run_backtest("XAUUSD", closes, static_brick, "static")
    if result_b_static:
        b_static_omega = result_b_static.omega
        b_static_pnl = sum(t.net_usd for t in result_b_static.trades)
        print(f"✓ Ω={b_static_omega:.3f}, P&L=${b_static_pnl:.2f}")
    else:
        b_static_omega = 0.0
        b_static_pnl = 0.0
        print("✗ Failed")

    print("    └─ Compounded 0.01/$1000...", end=" ", flush=True)
    result_b_compounded = run_backtest("XAUUSD", closes, static_brick, "compounded", 1000.0)
    if result_b_compounded:
        b_compounded_omega = result_b_compounded.omega
        b_compounded_pnl = sum(t.net_usd for t in result_b_compounded.trades)
        print(f"✓ Ω={b_compounded_omega:.3f}, P&L=${b_compounded_pnl:.2f}")
    else:
        b_compounded_omega = 0.0
        b_compounded_pnl = 0.0
        print("✗ Failed")

    # Determine winners
    print("\n" + "=" * 80)
    print("  RESULTS & WINNERS")
    print("=" * 80)

    print(f"\n  SCENARIO A — DSP-Arrived Brick ({dsp_brick:.6f})")
    print("  ┌──────────────────────────────────────────────────────────┐")
    print("  │ STATIC (0.01 fixed)        │ COMPOUNDED (0.01/$1,000)   │")
    print(
        f"  │ Omega: {a_static_omega:6.3f}           │ Omega: {a_compounded_omega:6.3f}         │"
    )
    print(f"  │ P&L:   ${a_static_pnl:8.2f}        │ P&L:   ${a_compounded_pnl:8.2f}        │")
    print("  ├──────────────────────────────────────────────────────────┤")

    if a_compounded_omega > a_static_omega:
        winner_a = "COMPOUNDED"
        margin_a = a_compounded_omega - a_static_omega
        print(f"  │ WINNER: {winner_a:20s} (+{margin_a:.3f} Omega)              │")
    else:
        winner_a = "STATIC"
        margin_a = a_static_omega - a_compounded_omega
        print(f"  │ WINNER: {winner_a:20s} (+{margin_a:.3f} Omega)              │")

    print("  └──────────────────────────────────────────────────────────┘")

    print(f"\n  SCENARIO B — Static Arbitrary Brick ({static_brick:.6f})")
    print("  ┌──────────────────────────────────────────────────────────┐")
    print("  │ STATIC (0.01 fixed)        │ COMPOUNDED (0.01/$1,000)   │")
    print(
        f"  │ Omega: {b_static_omega:6.3f}           │ Omega: {b_compounded_omega:6.3f}         │"
    )
    print(f"  │ P&L:   ${b_static_pnl:8.2f}        │ P&L:   ${b_compounded_pnl:8.2f}        │")
    print("  ├──────────────────────────────────────────────────────────┤")

    if b_compounded_omega > b_static_omega:
        winner_b = "COMPOUNDED"
        margin_b = b_compounded_omega - b_static_omega
        print(f"  │ WINNER: {winner_b:20s} (+{margin_b:.3f} Omega)              │")
    else:
        winner_b = "STATIC"
        margin_b = b_static_omega - b_compounded_omega
        print(f"  │ WINNER: {winner_b:20s} (+{margin_b:.3f} Omega)              │")

    print("  └──────────────────────────────────────────────────────────┘")

    print("\n  OVERALL CONCLUSION")
    print("  ┌──────────────────────────────────────────────────────────┐")

    if winner_a == winner_b:
        print(f"  │ Both scenarios favor: {winner_a:40s} │")
        print(f"  │ Recommendation: Use {winner_a.lower()} lot sizing                      │")
    else:
        print(f"  │ Scenario A (DSP brick): {winner_a:35s} │")
        print(f"  │ Scenario B (Static brick): {winner_b:33s} │")
        print("  │ Conclusion: Mixed results — test with your data     │")

    print("  └──────────────────────────────────────────────────────────┘")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
