#!/usr/bin/env python3
"""
A/B Lot Sizing Comparison Test — XAUUSD Friction-Aware
=======================================================

Compares static (0.01 fixed) vs compounded (0.01 per $1,000) lot sizing
across two brick size scenarios:
  - A: DSP-arrived brick size (from variance ratio peak, in price units)
  - B: Static arbitrary brick size (fixed reference, in price units)

XAUUSD Friction Model:
  • Commission: $3.50 entry + $3.50 exit = $7.00/lot RT
  • Spread: 7 points × $1/point = $7.00/lot
  • Total: $14.00 per round-trip (1 lot)

Terminology:
  • Price unit: USD/oz for XAUUSD (1.0 = $1.00/oz = $100/lot)
  • Brick size is in price units (not pips/points)
  • Friction ratio = friction_cost / (brick_size_usd) → target ≤ 25%

Usage:
    python scripts/renko/ab_lot_sizing_test.py --symbol XAUUSD --timeframe M30
"""

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from kinetra.friction_cost import get_calculator
from kinetra.renko.backtest import (
    FilterParams,
    SizingMode,
    StopParams,
    VolSizingParams,
    backtest_instrument,
)
from kinetra.renko.dsp import vr_peak, vr_profile

# ══════════════════════════════════════════════════════════════════════════════
# Dataclass for A/B Results
# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class LotSizingABResult:
    """Result of comparing static vs compounded lot sizing."""

    symbol: str
    brick_scenario: str  # "DSP-Arrived" or "Static Arbitrary"
    dsp_brick_size: float
    test_brick_size: float

    # Static lot results (0.01 fixed)
    static_omega: float
    static_z: float
    static_trades: int
    static_pnl: float
    static_dd: float

    # Compounded lot results (0.01 per $1,000)
    compounded_omega: float
    compounded_z: float
    compounded_trades: int
    compounded_pnl: float
    compounded_dd: float

    # Winner determination
    winner: str  # "static" or "compounded"
    winner_reason: str


def compute_dsp_brick_size(closes: pd.Series, timeframe: str = "M30") -> float:
    """
    Compute DSP brick size from variance ratio peak.

    Parameters
    ----------
    closes : pd.Series
        Close prices indexed by datetime.
    timeframe : str
        Timeframe (M30, H1, H4, etc.).

    Returns
    -------
    float
        DSP brick size (median displacement at peak scale).
    """
    closes_arr = closes.values.astype(np.float64)

    if len(closes_arr) < 100:
        return 1.0  # fallback

    # Compute VR profile
    try:
        profile = vr_profile(closes_arr, scales=[2, 4, 6, 8, 12, 16, 24, 32, 48])
        if not profile:
            return 1.0

        peak_scale, peak_vr = vr_peak(profile)

        # Compute displacement at peak scale (brick size estimate)
        returns = np.diff(np.log(closes_arr))
        n_windows = len(returns) // peak_scale
        if n_windows < 1:
            return 1.0

        displacements = []
        for i in range(n_windows):
            window_ret = returns[i * peak_scale : (i + 1) * peak_scale]
            displacement = np.abs(np.sum(window_ret))
            displacements.append(displacement)

        brick_size = float(np.median(displacements)) if displacements else 1.0
        return max(brick_size, 0.1)  # ensure minimum

    except Exception as e:
        print(f"  Warning: DSP calculation failed ({e}), using fallback")
        return 1.0


def run_ab_test(
    symbol: str,
    closes: pd.Series,
    dsp_brick_size: float,
    static_brick_size: float,
    calc=None,
    timeframe: str = "M30",
) -> Tuple[LotSizingABResult, LotSizingABResult]:
    """
    Run A/B tests for both brick size scenarios.

    Returns
    -------
    (result_a, result_b)
        A: DSP-arrived brick size
        B: Static arbitrary brick size
    """

    # Default filter and stop params
    filter_params = FilterParams()
    stop_params = StopParams()

    # Session break for M30
    session_break_minutes = 30.0 if timeframe in ["M30", "H1"] else 1440.0

    print(f"\n{'=' * 80}")
    print(f"  A/B LOT SIZING TEST — {symbol}")
    print(f"{'=' * 80}")

    # ── Scenario A: DSP-Arrived Brick Size ──────────────────────────────────
    print(f"\n  SCENARIO A: DSP-Arrived Brick Size ({dsp_brick_size:.4f})")
    print(f"  {'─' * 76}")

    print("    Testing STATIC lot (0.01 fixed)...")
    result_a_static = backtest_instrument(
        symbol=symbol,
        closes=closes,
        brick_size=dsp_brick_size,
        filter_params=filter_params,
        stop_params=stop_params,
        calc=calc,
        session_break_minutes=session_break_minutes,
        sizing_mode=SizingMode.FIXED_LOT,
        vol_sizing_params=VolSizingParams(fixed_lot=0.01),
    )

    pnl_a_static = float(sum(t.net_usd for t in result_a_static.trades))
    print(
        f"      ✓ Omega={result_a_static.omega:.3f}, Z={result_a_static.z_factor:.2f}, "
        f"Trades={len(result_a_static.trades)}, P&L=${pnl_a_static:.2f}, "
        f"DD=${result_a_static.max_dd_usd:.2f}"
    )

    print("    Testing COMPOUNDED lot (0.01 per $1,000)...")
    result_a_compounded = backtest_instrument(
        symbol=symbol,
        closes=closes,
        brick_size=dsp_brick_size,
        filter_params=filter_params,
        stop_params=stop_params,
        calc=calc,
        session_break_minutes=session_break_minutes,
        sizing_mode=SizingMode.COMPOUNDING,
        vol_sizing_params=VolSizingParams(
            fixed_lot=0.01,
            compounding_capital_per_lot=1000.0,
            initial_equity=100_000.0,
        ),
    )

    pnl_a_compounded = float(sum(t.net_usd for t in result_a_compounded.trades))
    print(
        f"      ✓ Omega={result_a_compounded.omega:.3f}, Z={result_a_compounded.z_factor:.2f}, "
        f"Trades={len(result_a_compounded.trades)}, P&L=${pnl_a_compounded:.2f}, "
        f"DD=${result_a_compounded.max_dd_usd:.2f}"
    )

    winner_a = "compounded" if result_a_compounded.omega > result_a_static.omega else "static"
    reason_a = f"Higher Omega: {max(result_a_compounded.omega, result_a_static.omega):.3f}"

    result_a = LotSizingABResult(
        symbol=symbol,
        brick_scenario="DSP-Arrived",
        dsp_brick_size=dsp_brick_size,
        test_brick_size=dsp_brick_size,
        static_omega=result_a_static.omega,
        static_z=result_a_static.z_factor,
        static_trades=len(result_a_static.trades),
        static_pnl=pnl_a_static,
        static_dd=result_a_static.max_dd_usd,
        compounded_omega=result_a_compounded.omega,
        compounded_z=result_a_compounded.z_factor,
        compounded_trades=len(result_a_compounded.trades),
        compounded_pnl=pnl_a_compounded,
        compounded_dd=result_a_compounded.max_dd_usd,
        winner=winner_a,
        winner_reason=reason_a,
    )

    print(f"\n    WINNER (Scenario A): {winner_a.upper()} — {reason_a}")

    # ── Scenario B: Static Arbitrary Brick Size ──────────────────────────────
    print(f"\n  SCENARIO B: Static Arbitrary Brick Size ({static_brick_size:.4f})")
    print(f"  {'─' * 76}")

    print("    Testing STATIC lot (0.01 fixed)...")
    result_b_static = backtest_instrument(
        symbol=symbol,
        closes=closes,
        brick_size=static_brick_size,
        filter_params=filter_params,
        stop_params=stop_params,
        calc=calc,
        session_break_minutes=session_break_minutes,
        sizing_mode=SizingMode.FIXED_LOT,
        vol_sizing_params=VolSizingParams(fixed_lot=0.01),
    )

    pnl_b_static = float(sum(t.net_usd for t in result_b_static.trades))
    print(
        f"      ✓ Omega={result_b_static.omega:.3f}, Z={result_b_static.z_factor:.2f}, "
        f"Trades={len(result_b_static.trades)}, P&L=${pnl_b_static:.2f}, "
        f"DD=${result_b_static.max_dd_usd:.2f}"
    )

    print("    Testing COMPOUNDED lot (0.01 per $1,000)...")
    result_b_compounded = backtest_instrument(
        symbol=symbol,
        closes=closes,
        brick_size=static_brick_size,
        filter_params=filter_params,
        stop_params=stop_params,
        calc=calc,
        session_break_minutes=session_break_minutes,
        sizing_mode=SizingMode.COMPOUNDING,
        vol_sizing_params=VolSizingParams(
            fixed_lot=0.01,
            compounding_capital_per_lot=1000.0,
            initial_equity=100_000.0,
        ),
    )

    pnl_b_compounded = float(sum(t.net_usd for t in result_b_compounded.trades))
    print(
        f"      ✓ Omega={result_b_compounded.omega:.3f}, Z={result_b_compounded.z_factor:.2f}, "
        f"Trades={len(result_b_compounded.trades)}, P&L=${pnl_b_compounded:.2f}, "
        f"DD=${result_b_compounded.max_dd_usd:.2f}"
    )

    winner_b = "compounded" if result_b_compounded.omega > result_b_static.omega else "static"
    reason_b = f"Higher Omega: {max(result_b_compounded.omega, result_b_static.omega):.3f}"

    result_b = LotSizingABResult(
        symbol=symbol,
        brick_scenario="Static Arbitrary",
        dsp_brick_size=dsp_brick_size,
        test_brick_size=static_brick_size,
        static_omega=result_b_static.omega,
        static_z=result_b_static.z_factor,
        static_trades=len(result_b_static.trades),
        static_pnl=pnl_b_static,
        static_dd=result_b_static.max_dd_usd,
        compounded_omega=result_b_compounded.omega,
        compounded_z=result_b_compounded.z_factor,
        compounded_trades=len(result_b_compounded.trades),
        compounded_pnl=pnl_b_compounded,
        compounded_dd=result_b_compounded.max_dd_usd,
        winner=winner_b,
        winner_reason=reason_b,
    )

    print(f"\n    WINNER (Scenario B): {winner_b.upper()} — {reason_b}")

    return result_a, result_b


def print_summary(
    result_a: LotSizingABResult, result_b: LotSizingABResult, symbol: str = "XAUUSD"
) -> None:
    """Print summary comparison of both scenarios with friction analysis."""
    print(f"\n{'=' * 80}")
    print(f"  SUMMARY: A/B LOT SIZING TEST RESULTS — {symbol}")
    print(f"{'=' * 80}")

    # XAUUSD friction details
    if symbol.upper() == "XAUUSD":
        print("\n  FRICTION ANALYSIS (XAUUSD):")
        print("  ├─ Commission: $7.00 ($3.50 each way)")
        print("  ├─ Spread: $7.00 (7 points × $1/point)")
        print("  ├─ Total Friction: $14.00 per round-trip")
        print(
            f"  ├─ Scenario A Friction Ratio: {result_a.static_pnl / result_a.static_pnl if result_a.static_pnl > 0 else 0:.1%}"
        )
        print(
            f"  └─ Scenario B Friction Ratio: {result_b.static_pnl / result_b.static_pnl if result_b.static_pnl > 0 else 0:.1%}"
        )

    print(f"\n  SCENARIO A — DSP-Arrived Brick ({result_a.test_brick_size:.4f})")
    print("  ┌────────────────────────────────────────────────────────────────┐")
    print("  │ STATIC (0.01 fixed)                                             │")
    print(
        f"  │   Omega: {result_a.static_omega:6.3f}  Z-Factor: {result_a.static_z:5.2f}  Trades: {result_a.static_trades:3d}       │"
    )
    print(
        f"  │   P&L: ${result_a.static_pnl:9.2f}  Max DD: ${result_a.static_dd:9.2f}             │"
    )
    print("  ├────────────────────────────────────────────────────────────────┤")
    print("  │ COMPOUNDED (0.01 per $1,000)                                   │")
    print(
        f"  │   Omega: {result_a.compounded_omega:6.3f}  Z-Factor: {result_a.compounded_z:5.2f}  Trades: {result_a.compounded_trades:3d}       │"
    )
    print(
        f"  │   P&L: ${result_a.compounded_pnl:9.2f}  Max DD: ${result_a.compounded_dd:9.2f}             │"
    )
    print("  ├────────────────────────────────────────────────────────────────┤")
    print(f"  │ WINNER: {result_a.winner.upper():20s}                                │")
    print(f"  │ Reason: {result_a.winner_reason:48s} │")
    print("  └────────────────────────────────────────────────────────────────┘")

    print(f"\n  SCENARIO B — Static Arbitrary Brick ({result_b.test_brick_size:.4f})")
    print("  ┌────────────────────────────────────────────────────────────────┐")
    print("  │ STATIC (0.01 fixed)                                             │")
    print(
        f"  │   Omega: {result_b.static_omega:6.3f}  Z-Factor: {result_b.static_z:5.2f}  Trades: {result_b.static_trades:3d}       │"
    )
    print(
        f"  │   P&L: ${result_b.static_pnl:9.2f}  Max DD: ${result_b.static_dd:9.2f}             │"
    )
    print("  ├────────────────────────────────────────────────────────────────┤")
    print("  │ COMPOUNDED (0.01 per $1,000)                                   │")
    print(
        f"  │   Omega: {result_b.compounded_omega:6.3f}  Z-Factor: {result_b.compounded_z:5.2f}  Trades: {result_b.compounded_trades:3d}       │"
    )
    print(
        f"  │   P&L: ${result_b.compounded_pnl:9.2f}  Max DD: ${result_b.compounded_dd:9.2f}             │"
    )
    print("  ├────────────────────────────────────────────────────────────────┤")
    print(f"  │ WINNER: {result_b.winner.upper():20s}                                │")
    print(f"  │ Reason: {result_b.winner_reason:48s} │")
    print("  └────────────────────────────────────────────────────────────────┘")

    print("\n  OVERALL WINNER")
    print("  ┌────────────────────────────────────────────────────────────────┐")

    if result_a.winner == result_b.winner:
        print(f"  │ Both scenarios favor: {result_a.winner.upper():43s} │")
    else:
        print(f"  │ Scenario A favors: {result_a.winner.upper():45s} │")
        print(f"  │ Scenario B favors: {result_b.winner.upper():45s} │")

    print("  └────────────────────────────────────────────────────────────────┘")


def main() -> None:
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="A/B test: static vs compounded lot sizing with DSP vs static brick sizes"
    )
    parser.add_argument("--symbol", default="XAUUSD", help="Instrument symbol (default: XAUUSD)")
    parser.add_argument("--timeframe", default="M30", help="Timeframe (default: M30)")
    parser.add_argument(
        "--static-brick",
        type=float,
        default=1.0,
        help="Static brick size for Scenario B (default: 1.0)",
    )
    parser.add_argument("--csv", help="Path to CSV file with close prices (time, close columns)")

    args = parser.parse_args()

    symbol = args.symbol
    timeframe = args.timeframe
    static_brick_size = args.static_brick

    # Load data
    if args.csv:
        csv_path = Path(args.csv)
        if not csv_path.exists():
            print(f"Error: CSV file not found: {csv_path}")
            sys.exit(1)

        df = pd.read_csv(csv_path, parse_dates=["time"], index_col="time")
        closes = df["close"]
    else:
        # Try to load from master_standardized
        master_dir = PROJECT_ROOT / "data" / "master_standardized"

        # Find the instrument folder
        symbol_upper = symbol.upper()
        instr_dirs = list(master_dir.glob(f"*/{symbol_upper}"))

        if not instr_dirs:
            print(f"Error: No data found for {symbol}")
            print(f"  Checked: {master_dir}/*/{symbol_upper}")
            print("  Available CSV test data can be provided via --csv")
            sys.exit(1)

        instr_dir = instr_dirs[0]

        # Find CSV file matching timeframe
        csv_files = list(instr_dir.glob(f"{symbol_upper}_{timeframe}_*.csv"))

        if not csv_files:
            print(f"Error: No {timeframe} CSV found for {symbol}")
            print(f"  Checked: {instr_dir}/{symbol_upper}_{timeframe}_*.csv")
            sys.exit(1)

        csv_path = csv_files[0]
        print(f"Loading data from: {csv_path}")

        df = pd.read_csv(csv_path, parse_dates=["time"], index_col="time")
        closes = df["close"]

    if len(closes) < 100:
        print(f"Error: Insufficient data ({len(closes)} bars, need ≥100)")
        sys.exit(1)

    print(f"Loaded {len(closes)} bars of {symbol} {timeframe}")

    # Compute DSP brick size
    print("\nComputing DSP brick size...")
    dsp_brick_size = compute_dsp_brick_size(closes, timeframe)
    print(f"  DSP Brick Size: {dsp_brick_size:.4f}")

    # Get friction calculator
    try:
        calc = get_calculator(symbol)
        print(f"  Loaded friction calculator for {symbol}")
    except Exception as e:
        print(f"  Warning: Could not load friction calculator ({e})")
        calc = None

    # Run A/B test
    result_a, result_b = run_ab_test(
        symbol=symbol,
        closes=closes,
        dsp_brick_size=dsp_brick_size,
        static_brick_size=static_brick_size,
        calc=calc,
        timeframe=timeframe,
    )

    # Print summary
    print_summary(result_a, result_b)

    print(f"\n{'=' * 80}")


if __name__ == "__main__":
    main()
