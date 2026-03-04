#!/usr/bin/env python3
"""
XAUUSD Brick Validation — Fixed vs Compounded Sizing Comparison

Uses existing Renko engine components (DRY) to validate the $1.00 brick framework
with both FIXED_LOT and COMPOUNDING position sizing modes.

Compares:
  - Omega, Z-factor, Profit factor
  - Win rate, Max DD, Trade count
  - Equity curve shape
  - MAE (Max Adverse Excursion), MFE (Max Favorable Excursion)
  - Total PnL and per-trade metrics

Usage:
    python scripts/renko/validate_brick_xauusd.py
"""

from __future__ import annotations

import json
import logging
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

# ── Renko Engine Components (DRY) ─────────────────────────────────────────────
from kinetra.renko.backtest import (
    FilterParams,
    RiskParams,
    SizingMode,
    StopParams,
    VolSizingParams,
    backtest_instrument,
)
from kinetra.renko.brick_engine import build_renko
from kinetra.renko.dsp import (
    SpreadProfile,
    compute_full_friction_floor,
    run_dsp,
    scaled_filter_params,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Contract Specs (Pepperstone cTrader ECN) ───────────────────────────────────
CONTRACT_SIZE = 100.0  # oz per lot
COMMISSION_RT = 7.0  # $3.50/side × 2 = $7.00 round-trip
TICK_SIZE = 0.01
MIN_LOT = 0.01
USD_PER_POINT = 1.0  # $1 per 0.01 price move per lot
BROKER = "pepperstone_demo_45841299"
INSTRUMENT = "XAUUSD"
CATEGORY = "metals"

# ── Brick Configuration ───────────────────────────────────────────────────────
BRICK_SIZE_ABS = 1.00  # $1.00 absolute price
FRICTION_MULT = 4.0


@dataclass
class BacktestComparison:
    """Results from comparing FIXED vs COMPOUNDING sizing."""

    # Common parameters
    brick_size: float
    friction_ratio: float
    floor_brick_ratio: float
    vr_peak_value: float
    regime: str

    # Fixed lot results
    fixed_omega: float
    fixed_z_factor: float
    fixed_profit_factor: float
    fixed_win_rate: float
    fixed_max_dd_pct: float
    fixed_n_trades: int
    fixed_total_pnl: float
    fixed_avg_trade: float
    fixed_mae_pct: float
    fixed_mfe_pct: float
    fixed_final_equity: float

    # Compounded results
    comp_omega: float
    comp_z_factor: float
    comp_profit_factor: float
    comp_win_rate: float
    comp_max_dd_pct: float
    comp_n_trades: int
    comp_total_pnl: float
    comp_avg_trade: float
    comp_mae_pct: float
    comp_mfe_pct: float
    comp_final_equity: float

    # Improvement metrics
    omega_improvement_pct: float
    z_factor_improvement_pct: float
    pnl_improvement_pct: float
    final_equity_improvement_pct: float


def load_data() -> pd.DataFrame:
    """Load XAUUSD M1 data from canonical location."""
    data_path = (
        project_root
        / "data"
        / "master_standardized"
        / "ctrader"
        / BROKER
        / CATEGORY
        / INSTRUMENT
        / f"{INSTRUMENT}_M1_202511022301_202603020927.csv"
    )

    if not data_path.exists():
        data_path = project_root / "XAUUSD_M1_accurate.csv"

    if not data_path.exists():
        raise FileNotFoundError(f"Data not found: {data_path}")

    df = pd.read_csv(data_path)
    df.columns = df.columns.str.lower()
    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"], utc=True)
        df = df.set_index("time")

    return df


def compute_spread_profile(df: pd.DataFrame) -> SpreadProfile:
    """Compute spread profile from M1 data."""
    if "spread" not in df.columns:
        spread_vals = (df["high"] - df["low"]).values * 0.1
    else:
        spread_vals = df["spread"].values

    spread_vals = spread_vals[spread_vals > 0]

    return SpreadProfile.from_points(
        symbol=INSTRUMENT,
        tick_size=TICK_SIZE,
        p50_pts=float(np.percentile(spread_vals, 50)),
        p75_pts=float(np.percentile(spread_vals, 75)),
        p95_pts=float(np.percentile(spread_vals, 95)),
        source="csv_m1",
        n_bars=len(spread_vals),
    )


def compute_mae_mfe(trades: List, equity_curve: List[float]) -> tuple[float, float]:
    """
    Compute MAE (Max Adverse Excursion) and MFE (Max Favorable Excursion).

    MAE = Average friction as % of gross P&L (entry cost)
    MFE = Average net P&L as % of gross P&L (exit capture)

    These measure entry efficiency and exit efficiency.
    """
    if not trades:
        return 0.0, 0.0

    mae_values = []
    mfe_values = []

    for trade in trades:
        gross = abs(getattr(trade, "gross_usd", 0))
        net = abs(getattr(trade, "net_usd", 0))
        friction = getattr(trade, "friction_usd", 0)

        if gross > 0:
            mae_values.append(friction / gross)  # Entry cost as % of gross
            mfe_values.append(net / gross)  # Exit capture as % of gross

    return (
        float(np.mean(mae_values)) * 100 if mae_values else 0.0,
        float(np.mean(mfe_values)) * 100 if mfe_values else 0.0,
    )


def run_backtest_comparison(
    closes: pd.Series,
    brick_size: float,
    filter_params: FilterParams,
    stop_params: StopParams,
    risk_params: RiskParams,
    spread: SpreadProfile,
    session_break_minutes: float = 30.0,
    initial_equity: float = 10000.0,
) -> BacktestComparison:
    """
    Run backtest with both FIXED_LOT and COMPOUNDING sizing modes.

    Uses existing Renko engine components (DRY):
    - backtest_instrument with SizingMode
    - InstrumentBacktestResult for metrics
    """

    # ── DSP Analysis ─────────────────────────────────────────────────────────────
    dsp_result = run_dsp(closes.values, symbol=INSTRUMENT, bars_per_hour=60.0)

    # ── Build bricks for bricks_per_day calculation ─────────────────────────────
    bricks = build_renko(closes, brick_size, session_break_minutes=session_break_minutes)
    days = (closes.index.max() - closes.index.min()).total_seconds() / 86400
    bricks_per_day = len(bricks) / max(days, 1)

    # ── Fixed Lot Backtest ─────────────────────────────────────────────────────
    fixed_vol_params = VolSizingParams(
        fixed_lot=MIN_LOT,
        initial_equity=initial_equity,
        compounding_capital_per_lot=1000.0,  # Not used for FIXED_LOT
    )

    fixed_result = backtest_instrument(
        symbol=INSTRUMENT,
        closes=closes,
        brick_size=brick_size,
        filter_params=filter_params,
        stop_params=stop_params,
        risk_params=risk_params,
        sizing_mode=SizingMode.FIXED_LOT,
        vol_sizing_params=fixed_vol_params,
        session_break_minutes=session_break_minutes,
    )

    # ── Compounding Backtest ─────────────────────────────────────────────────────
    comp_vol_params = VolSizingParams(
        fixed_lot=MIN_LOT,
        initial_equity=initial_equity,
        compounding_capital_per_lot=1000.0,  # lot = equity / $1000
    )

    comp_result = backtest_instrument(
        symbol=INSTRUMENT,
        closes=closes,
        brick_size=brick_size,
        filter_params=filter_params,
        stop_params=stop_params,
        risk_params=risk_params,
        sizing_mode=SizingMode.COMPOUNDING,
        vol_sizing_params=comp_vol_params,
        session_break_minutes=session_break_minutes,
    )

    # ── Compute MAE/MFE ───────────────────────────────────────────────────────
    fixed_mae, fixed_mfe = compute_mae_mfe(fixed_result.trades, fixed_result.equity_curve)
    comp_mae, comp_mfe = compute_mae_mfe(comp_result.trades, comp_result.equity_curve)

    # ── Calculate improvements ────────────────────────────────────────────────
    omega_imp = (
        (comp_result.omega - fixed_result.omega) / fixed_result.omega * 100
        if fixed_result.omega != 0
        else 0
    )
    z_imp = (
        (comp_result.z_factor - fixed_result.z_factor) / fixed_result.z_factor * 100
        if fixed_result.z_factor != 0
        else 0
    )
    pnl_imp = (
        (comp_result.equity_curve[-1] - fixed_result.equity_curve[-1])
        / abs(fixed_result.equity_curve[-1])
        * 100
        if fixed_result.equity_curve[-1] != 0
        else 0
    )
    equity_imp = (
        (comp_result.equity_curve[-1] - fixed_result.equity_curve[-1])
        / abs(fixed_result.equity_curve[-1])
        * 100
        if fixed_result.equity_curve[-1] != 0
        else 0
    )

    # ── Friction floor ─────────────────────────────────────────────────────────
    floor = compute_full_friction_floor(
        spread=spread,
        dsp_brick=brick_size,
        commission_rt=COMMISSION_RT,
        contract_size=CONTRACT_SIZE,
        friction_mult=FRICTION_MULT,
    )
    friction_per_unit = (COMMISSION_RT / CONTRACT_SIZE) + spread.spread_p50_price
    friction_ratio = friction_per_unit / brick_size

    return BacktestComparison(
        brick_size=brick_size,
        friction_ratio=friction_ratio,
        floor_brick_ratio=floor.floor_price / brick_size,
        vr_peak_value=dsp_result.vr_peak_value,
        regime=dsp_result.regime,
        fixed_omega=fixed_result.omega,
        fixed_z_factor=fixed_result.z_factor,
        fixed_profit_factor=fixed_result.profit_factor,
        fixed_win_rate=fixed_result.win_rate,
        fixed_max_dd_pct=_max_dd_pct(fixed_result.equity_curve),
        fixed_n_trades=len(fixed_result.trades),
        fixed_total_pnl=fixed_result.equity_curve[-1],
        fixed_avg_trade=fixed_result.avg_net_per_trade,
        fixed_mae_pct=fixed_mae,
        fixed_mfe_pct=fixed_mfe,
        fixed_final_equity=fixed_result.equity_curve[-1],
        comp_omega=comp_result.omega,
        comp_z_factor=comp_result.z_factor,
        comp_profit_factor=comp_result.profit_factor,
        comp_win_rate=comp_result.win_rate,
        comp_max_dd_pct=_max_dd_pct(comp_result.equity_curve),
        comp_n_trades=len(comp_result.trades),
        comp_total_pnl=comp_result.equity_curve[-1],
        comp_avg_trade=comp_result.avg_net_per_trade,
        comp_mae_pct=comp_mae,
        comp_mfe_pct=comp_mfe,
        comp_final_equity=comp_result.equity_curve[-1],
        omega_improvement_pct=omega_imp,
        z_factor_improvement_pct=z_imp,
        pnl_improvement_pct=pnl_imp,
        final_equity_improvement_pct=equity_imp,
    )


def _max_dd_pct(equity_curve: List[float]) -> float:
    """Calculate max drawdown as percentage."""
    if len(equity_curve) < 2:
        return 0.0

    equity = np.array(equity_curve)
    peak = np.maximum.accumulate(equity)
    dd = (peak - equity) / np.maximum(peak, 1e-10)
    return float(np.max(dd))


def print_section(title: str) -> None:
    logger.info("")
    logger.info("=" * 70)
    logger.info(f"  {title}")
    logger.info("=" * 70)


def print_subsection(title: str) -> None:
    logger.info("")
    logger.info("-" * 70)
    logger.info(f"  {title}")
    logger.info("-" * 70)


def main():
    print_section("XAUUSD BRICK VALIDATION — FIXED vs COMPOUNDING")

    # ── Load data ───────────────────────────────────────────────────────────────
    df = load_data()
    date_start = df.index.min()
    date_end = df.index.max()
    days = (date_end - date_start).total_seconds() / 86400

    logger.info(f"Data: {date_start} to {date_end}")
    logger.info(f"Duration: {days:.0f} days ({days / 30:.1f} months)")
    logger.info(f"Bars: {len(df)}")

    # ── Spread profile ────────────────────────────────────────────────────────
    spread = compute_spread_profile(df)

    print_subsection("CONTRACT & FRICTION")
    logger.info(f"  Contract size:    {CONTRACT_SIZE} oz/lot")
    logger.info(f"  Commission:       ${COMMISSION_RT:.2f} RT")
    logger.info(f"  Spread p50:       ${spread.spread_p50_price:.4f}")
    logger.info(f"  Brick size:       ${BRICK_SIZE_ABS:.2f}")

    friction_per_unit = (COMMISSION_RT / CONTRACT_SIZE) + spread.spread_p50_price
    floor = compute_full_friction_floor(spread, BRICK_SIZE_ABS, COMMISSION_RT, CONTRACT_SIZE)

    logger.info(f"  Friction/unit:    ${friction_per_unit:.4f}")
    logger.info(f"  Friction ratio:   {friction_per_unit / BRICK_SIZE_ABS:.1%}")
    logger.info(f"  Floor:            ${floor.floor_price:.4f}")
    logger.info(f"  Brick/floor:     {BRICK_SIZE_ABS / floor.floor_price:.1f}×")

    # ── Filter params ────────────────────────────────────────────────────────────
    closes = df["close"]
    bricks_temp = build_renko(closes, BRICK_SIZE_ABS)
    bricks_per_day = len(bricks_temp) / max(days, 1)

    dsp_result = run_dsp(closes.values, symbol=INSTRUMENT, bars_per_hour=60.0)
    filter_params = scaled_filter_params(bricks_per_day)

    print_subsection("FILTER PARAMS")
    logger.info(f"  VR peak:          {dsp_result.vr_peak_value:.4f} ({dsp_result.regime})")
    logger.info(f"  FlipRate window:  {filter_params.fliprate_window}")
    logger.info(f"  FlipRate thresh:  {filter_params.fliprate_threshold:.2f}")
    logger.info(f"  Markov window:    {filter_params.markov_window}")
    logger.info(f"  Markov thresh:    {filter_params.markov_threshold:.2f}")

    # ── Run comparison ──────────────────────────────────────────────────────────
    stop_params = StopParams(stop_bricks=1.0, exit_on_colour_change=True, allow_short=True)
    risk_params = RiskParams(
        loss_cluster_window=10,
        loss_cluster_threshold=0.7,
        loss_cluster_cooldown=5,
        dd_throttle_pct=0.5,
        dd_halt_pct=0.2,
    )

    comparison = run_backtest_comparison(
        closes=closes,
        brick_size=BRICK_SIZE_ABS,
        filter_params=filter_params,
        stop_params=stop_params,
        risk_params=risk_params,
        spread=spread,
        initial_equity=10000.0,
    )

    # ── Print comparison ───────────────────────────────────────────────────────
    print_subsection("FIXED vs COMPOUNDING COMPARISON")
    logger.info("")
    logger.info(f"  {'Metric':<25} {'FIXED_LOT':>15} {'COMPOUNDING':>15} {'Δ':>12}")
    logger.info(f"  {'-' * 67}")

    logger.info(
        f"  {'Omega':<25} {comparison.fixed_omega:>15.3f} {comparison.comp_omega:>15.3f} {comparison.omega_improvement_pct:>+11.1f}%"
    )
    logger.info(
        f"  {'Z-factor':<25} {comparison.fixed_z_factor:>15.2f} {comparison.comp_z_factor:>15.2f} {comparison.z_factor_improvement_pct:>+11.1f}%"
    )
    logger.info(
        f"  {'Profit factor':<25} {comparison.fixed_profit_factor:>15.2f} {comparison.comp_profit_factor:>15.2f}"
    )
    logger.info(
        f"  {'Win rate':<25} {comparison.fixed_win_rate:>14.1%} {comparison.comp_win_rate:>14.1%}"
    )
    logger.info(
        f"  {'Max DD':<25} {comparison.fixed_max_dd_pct:>14.2%} {comparison.comp_max_dd_pct:>14.2%}"
    )
    logger.info(
        f"  {'# Trades':<25} {comparison.fixed_n_trades:>15} {comparison.comp_n_trades:>15}"
    )
    logger.info(
        f"  {'Total PnL':<25} ${comparison.fixed_total_pnl:>14,.0f} ${comparison.comp_total_pnl:>14,.0f} {comparison.pnl_improvement_pct:>+11.1f}%"
    )
    logger.info(
        f"  {'Final equity':<25} ${comparison.fixed_final_equity:>14,.0f} ${comparison.comp_final_equity:>14,.0f} {comparison.final_equity_improvement_pct:>+11.1f}%"
    )
    logger.info(
        f"  {'MAE (entry cost)':<25} {comparison.fixed_mae_pct:>14.2f}% {comparison.comp_mae_pct:>14.2f}%"
    )
    logger.info(
        f"  {'MFE (exit capture)':<25} {comparison.fixed_mfe_pct:>14.2f}% {comparison.comp_mfe_pct:>14.2f}%"
    )

    # ── Verdict ────────────────────────────────────────────────────────────────
    print_subsection("VERDICT")
    logger.info("")
    logger.info(f"  Friction ratio:     {comparison.friction_ratio:.1%} (target: ≤25%)")
    logger.info(
        f"  Brick above floor:  {'✅ YES' if comparison.floor_brick_ratio < 1.0 else '❌ NO'}"
    )
    logger.info(f"  Regime:             {comparison.regime}")
    logger.info("")

    if comparison.omega_improvement_pct > 0:
        logger.info(f"  ✅ COMPOUNDING improves Omega by {comparison.omega_improvement_pct:.1f}%")
    else:
        logger.info(
            f"  ⚠️  COMPOUNDING reduces Omega by {abs(comparison.omega_improvement_pct):.1f}%"
        )

    if comparison.pnl_improvement_pct > 0:
        logger.info(f"  ✅ COMPOUNDING improves PnL by {comparison.pnl_improvement_pct:.1f}%")
    else:
        logger.info(f"  ⚠️  COMPOUNDING reduces PnL by {abs(comparison.pnl_improvement_pct):.1f}%")

    # ── Save JSON ───────────────────────────────────────────────────────────────
    output = {
        "validation": {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "instrument": INSTRUMENT,
            "broker": BROKER,
            "date_start": date_start.isoformat(),
            "date_end": date_end.isoformat(),
            "days": days,
            "brick_size": BRICK_SIZE_ABS,
        },
        "contract": {
            "contract_size_oz": CONTRACT_SIZE,
            "commission_rt_usd": COMMISSION_RT,
            "tick_size": TICK_SIZE,
            "min_lot": MIN_LOT,
            "usd_per_point": USD_PER_POINT,
        },
        "friction": {
            "friction_per_unit": friction_per_unit,
            "friction_ratio": comparison.friction_ratio,
            "floor_price": floor.floor_price,
            "floor_brick_ratio": comparison.floor_brick_ratio,
        },
        "dsp": {
            "vr_peak_value": comparison.vr_peak_value,
            "regime": comparison.regime,
            "bricks_per_day": bricks_per_day,
        },
        "filter_params": {
            "fliprate_window": filter_params.fliprate_window,
            "fliprate_threshold": filter_params.fliprate_threshold,
            "markov_window": filter_params.markov_window,
            "markov_threshold": filter_params.markov_threshold,
        },
        "comparison": {
            "fixed_lot": {
                "omega": comparison.fixed_omega,
                "z_factor": comparison.fixed_z_factor,
                "profit_factor": comparison.fixed_profit_factor,
                "win_rate": comparison.fixed_win_rate,
                "max_dd_pct": comparison.fixed_max_dd_pct,
                "n_trades": comparison.fixed_n_trades,
                "total_pnl": comparison.fixed_total_pnl,
                "final_equity": comparison.fixed_final_equity,
                "mae_pct": comparison.fixed_mae_pct,
                "mfe_pct": comparison.fixed_mfe_pct,
            },
            "compounding": {
                "omega": comparison.comp_omega,
                "z_factor": comparison.comp_z_factor,
                "profit_factor": comparison.comp_profit_factor,
                "win_rate": comparison.comp_win_rate,
                "max_dd_pct": comparison.comp_max_dd_pct,
                "n_trades": comparison.comp_n_trades,
                "total_pnl": comparison.comp_total_pnl,
                "final_equity": comparison.comp_final_equity,
                "mae_pct": comparison.comp_mae_pct,
                "mfe_pct": comparison.comp_mfe_pct,
            },
            "improvement": {
                "omega_pct": comparison.omega_improvement_pct,
                "z_factor_pct": comparison.z_factor_improvement_pct,
                "pnl_pct": comparison.pnl_improvement_pct,
                "equity_pct": comparison.final_equity_improvement_pct,
            },
        },
    }

    output_dir = (
        project_root / "data" / "master_standardized" / "ctrader" / BROKER / CATEGORY / INSTRUMENT
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "brick_validation.json"

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)

    print_section("JSON OUTPUT")
    logger.info(f"Saved to: {output_path}")
    logger.info("")
    logger.info(json.dumps(output, indent=2, default=str))


if __name__ == "__main__":
    main()
