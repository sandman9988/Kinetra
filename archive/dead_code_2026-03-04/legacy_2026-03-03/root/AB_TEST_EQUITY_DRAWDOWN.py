#!/usr/bin/env python3
"""
A/B Lot Sizing Test — Equity Curve & Drawdown Analytics
=========================================================

Comprehensive equity and drawdown metrics:
  • Equity curve tracking
  • Drawdown ratios and metrics
  • Recovery analysis
  • Calmar ratio
  • Return/Drawdown ratios
  • Consecutive drawdown periods
"""

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


@dataclass
class EquityMetrics:
    """Equity curve and drawdown statistics."""

    symbol: str
    brick_scenario: str
    sizing_mode: str

    # Equity summary
    initial_equity: float
    final_equity: float
    total_return_usd: float
    total_return_pct: float

    # Trade performance
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    profit_factor: float

    # Drawdown metrics
    max_dd_usd: float
    max_dd_pct: float
    avg_dd_usd: float
    avg_dd_pct: float

    # Drawdown ratios (key metrics)
    return_dd_ratio: float  # Total return / Max DD (higher = better)
    dd_ratio: float  # Max DD / Total return (lower = better)
    calmar_ratio: float  # Annual return / Max DD (higher = better)

    # Drawdown recovery
    recovery_trades_avg: float  # Avg trades to recover from DD
    longest_dd_duration: int  # Trades in longest DD period
    consecutive_dd_periods: int  # How many separate DD periods

    # Risk-adjusted
    omega: float
    z_factor: float
    sharpe_ratio: float

    # Quality metrics
    mae_pct: float  # Avg max adverse excursion
    mfe_pct: float  # Avg max favorable excursion
    avg_trade_pnl: float


def compute_equity_metrics(
    trades: List,
    omega: float,
    z_factor: float,
    max_dd_usd: float,
    brick_scenario: str,
    sizing_mode: str,
    symbol: str = "XAUUSD",
    initial_equity: float = 100_000.0,
) -> EquityMetrics:
    """Compute comprehensive equity and drawdown metrics."""

    if not trades:
        return EquityMetrics(
            symbol=symbol,
            brick_scenario=brick_scenario,
            sizing_mode=sizing_mode,
            initial_equity=initial_equity,
            final_equity=initial_equity,
            total_return_usd=0.0,
            total_return_pct=0.0,
            total_trades=0,
            winning_trades=0,
            losing_trades=0,
            win_rate=0.0,
            profit_factor=0.0,
            max_dd_usd=0.0,
            max_dd_pct=0.0,
            avg_dd_usd=0.0,
            avg_dd_pct=0.0,
            return_dd_ratio=0.0,
            dd_ratio=0.0,
            calmar_ratio=0.0,
            recovery_trades_avg=0.0,
            longest_dd_duration=0,
            consecutive_dd_periods=0,
            omega=omega,
            z_factor=z_factor,
            sharpe_ratio=0.0,
            mae_pct=0.0,
            mfe_pct=0.0,
            avg_trade_pnl=0.0,
        )

    # Build equity curve
    equity = initial_equity
    equity_curve = [equity]
    peaks = [equity]
    current_peak = equity

    for trade in trades:
        equity += trade.net_usd
        equity_curve.append(equity)

        if equity > current_peak:
            current_peak = equity
        peaks.append(current_peak)

    # Calculate drawdowns
    drawdowns = []
    drawdown_usd = []
    drawdown_pct = []
    dd_periods = []  # List of (start_idx, end_idx) for each DD period
    current_dd_start = None

    for i, (eq, peak) in enumerate(zip(equity_curve, peaks)):
        dd = peak - eq
        dd_pct = (dd / peak * 100) if peak > 0 else 0
        drawdowns.append(dd)
        drawdown_usd.append(dd)
        drawdown_pct.append(dd_pct)

        # Track DD periods
        if dd > 0:  # In drawdown
            if current_dd_start is None:
                current_dd_start = i
        else:  # Out of drawdown
            if current_dd_start is not None:
                dd_periods.append((current_dd_start, i))
                current_dd_start = None

    if current_dd_start is not None:
        dd_periods.append((current_dd_start, len(equity_curve)))

    # Equity metrics
    final_equity = equity_curve[-1]
    total_return_usd = final_equity - initial_equity
    total_return_pct = (total_return_usd / initial_equity) * 100

    # Trade counts
    net_pnls = [t.net_usd for t in trades]
    winning_trades = sum(1 for p in net_pnls if p > 0)
    losing_trades = sum(1 for p in net_pnls if p < 0)
    win_rate = (winning_trades / len(trades)) * 100 if trades else 0

    sum_wins = sum(p for p in net_pnls if p > 0)
    sum_losses = abs(sum(p for p in net_pnls if p < 0))
    profit_factor = (sum_wins / sum_losses) if sum_losses > 0 else (10.0 if sum_wins > 0 else 0.0)

    # Drawdown statistics
    max_dd_value = max(drawdown_usd) if drawdown_usd else 0
    max_dd_pct_value = max(drawdown_pct) if drawdown_pct else 0
    avg_dd_usd = (
        np.mean([d for d in drawdown_usd if d > 0]) if any(d > 0 for d in drawdown_usd) else 0
    )
    avg_dd_pct = (
        np.mean([d for d in drawdown_pct if d > 0]) if any(d > 0 for d in drawdown_pct) else 0
    )

    # Drawdown ratios (KEY METRICS)
    return_dd_ratio = total_return_usd / max_dd_value if max_dd_value > 0 else 0
    dd_ratio = max_dd_value / total_return_usd if total_return_usd > 0 else float("inf")

    # Calmar ratio (annual return / max DD)
    # Assuming ~250 trading days/year, ~20 trading days/month
    years = len(trades) / 20 / 12 if trades else 0  # Rough estimate
    annual_return = (total_return_usd / years) if years > 0 else 0
    calmar_ratio = annual_return / max_dd_value if max_dd_value > 0 else 0

    # Recovery analysis
    recovery_trades_list = []
    for start, end in dd_periods:
        recovery_trades = end - start
        recovery_trades_list.append(recovery_trades)

    recovery_trades_avg = np.mean(recovery_trades_list) if recovery_trades_list else 0
    longest_dd = max([(end - start) for start, end in dd_periods]) if dd_periods else 0

    # MAE/MFE
    mae_values = []
    mfe_values = []
    for t in trades:
        if abs(t.gross_usd) > 0:
            mae_values.append(t.friction_usd / abs(t.gross_usd))
            mfe_values.append(abs(t.net_usd) / abs(t.gross_usd))

    avg_mae = np.mean(mae_values) if mae_values else 0
    avg_mfe = np.mean(mfe_values) if mfe_values else 0
    avg_trade = np.mean(net_pnls) if net_pnls else 0

    # Sharpe (simplified: return / std)
    returns_arr = np.array([t.net_usd for t in trades])
    sharpe = (np.mean(returns_arr) / np.std(returns_arr)) if np.std(returns_arr) > 0 else 0

    return EquityMetrics(
        symbol=symbol,
        brick_scenario=brick_scenario,
        sizing_mode=sizing_mode,
        initial_equity=initial_equity,
        final_equity=final_equity,
        total_return_usd=total_return_usd,
        total_return_pct=total_return_pct,
        total_trades=len(trades),
        winning_trades=winning_trades,
        losing_trades=losing_trades,
        win_rate=win_rate,
        profit_factor=profit_factor,
        max_dd_usd=max_dd_value,
        max_dd_pct=max_dd_pct_value,
        avg_dd_usd=avg_dd_usd,
        avg_dd_pct=avg_dd_pct,
        return_dd_ratio=return_dd_ratio,
        dd_ratio=dd_ratio,
        calmar_ratio=calmar_ratio,
        recovery_trades_avg=recovery_trades_avg,
        longest_dd_duration=longest_dd,
        consecutive_dd_periods=len(dd_periods),
        omega=omega,
        z_factor=z_factor,
        sharpe_ratio=sharpe,
        mae_pct=avg_mae * 100,
        mfe_pct=avg_mfe * 100,
        avg_trade_pnl=avg_trade,
    )


def print_equity_summary(metrics: EquityMetrics) -> None:
    """Print equity curve and drawdown summary."""

    print(f"\n{'=' * 80}")
    print(f"  EQUITY & DRAWDOWN ANALYSIS — {metrics.symbol}")
    print(f"  Scenario: {metrics.brick_scenario} | Sizing: {metrics.sizing_mode}")
    print(f"{'=' * 80}")

    # Equity summary
    print("\n  EQUITY CURVE")
    print(f"  ├─ Initial Equity: ${metrics.initial_equity:,.2f}")
    print(f"  ├─ Final Equity: ${metrics.final_equity:,.2f}")
    print(f"  ├─ Total Return (USD): ${metrics.total_return_usd:,.2f}")
    print(f"  └─ Total Return (%): {metrics.total_return_pct:.2f}%")

    # Trade summary
    print("\n  TRADING PERFORMANCE")
    print(f"  ├─ Total Trades: {metrics.total_trades}")
    print(f"  ├─ Winners: {metrics.winning_trades} ({metrics.win_rate:.1f}%)")
    print(f"  ├─ Losers: {metrics.losing_trades}")
    print(f"  ├─ Profit Factor: {metrics.profit_factor:.2f}")
    print(f"  └─ Avg Trade P&L: ${metrics.avg_trade_pnl:.2f}")

    # Drawdown metrics
    print("\n  DRAWDOWN METRICS (ABSOLUTE)")
    print(f"  ├─ Max Drawdown (USD): ${metrics.max_dd_usd:.2f}")
    print(f"  ├─ Max Drawdown (%): {metrics.max_dd_pct:.2f}%")
    print(f"  ├─ Avg Drawdown (USD): ${metrics.avg_dd_usd:.2f}")
    print(f"  ├─ Avg Drawdown (%): {metrics.avg_dd_pct:.2f}%")
    print(f"  ├─ Longest DD Period: {metrics.longest_dd_duration} trades")
    print(f"  └─ Total DD Periods: {metrics.consecutive_dd_periods}")

    # KEY RATIOS
    print("\n  ⭐ KEY DRAWDOWN RATIOS")
    print(f"  ├─ Return/DD Ratio: {metrics.return_dd_ratio:.2f}x")
    print("  │  (How many $ return per $1 max DD)")
    print("  │  Target: > 2.0x (higher = better)")
    print("  │")
    print(f"  ├─ DD Ratio: {metrics.dd_ratio:.2f}")
    print("  │  (Max DD as fraction of total return)")
    print("  │  Target: < 1.0 (lower = better)")
    print("  │")
    print(f"  ├─ Calmar Ratio: {metrics.calmar_ratio:.2f}")
    print("  │  (Annual return / max DD)")
    print("  │  Target: > 0.5 (higher = better)")
    print("  │")
    print(f"  └─ Avg Recovery Time: {metrics.recovery_trades_avg:.1f} trades")
    print("     (Trades to recover from avg DD)")

    # Risk metrics
    print("\n  RISK METRICS")
    print(f"  ├─ Omega Ratio: {metrics.omega:.3f}")
    print(f"  ├─ Z-Factor: {metrics.z_factor:.2f}")
    print(f"  ├─ Sharpe Ratio: {metrics.sharpe_ratio:.2f}")
    print(f"  ├─ MAE (entry cost): {metrics.mae_pct:.1f}%")
    print(f"  └─ MFE (exit capture): {metrics.mfe_pct:.1f}%")

    # Status
    print("\n  DEPLOYMENT STATUS")
    status_checks = []
    status_checks.append(("Total Return > 0", metrics.total_return_usd > 0))
    status_checks.append(("Omega > 2.5", metrics.omega > 2.5))
    status_checks.append(("Max DD < 30%", metrics.max_dd_pct < 30))
    status_checks.append(("Return/DD Ratio > 1.5x", metrics.return_dd_ratio > 1.5))
    status_checks.append(("Calmar Ratio > 0.3", metrics.calmar_ratio > 0.3))

    for check_name, passed in status_checks:
        symbol = "✅" if passed else "❌"
        print(f"  {symbol} {check_name}")


def compare_equity_metrics(metrics_list: List[EquityMetrics]) -> None:
    """Compare equity metrics across scenarios."""
    print(f"\n{'=' * 80}")
    print("  COMPARATIVE EQUITY ANALYSIS")
    print(f"{'=' * 80}")

    print("\n  RETURN vs DRAWDOWN COMPARISON")
    print(f"  {'Scenario':<35} {'Return':<12} {'Max DD':<12} {'Return/DD':<10}")
    print(f"  {'-' * 80}")

    for m in metrics_list:
        scenario = f"{m.brick_scenario} / {m.sizing_mode}"
        print(
            f"  {scenario:<35} {m.total_return_pct:>10.2f}%  "
            f"{m.max_dd_pct:>10.2f}%  {m.return_dd_ratio:>8.2f}x"
        )

    # Rank by key ratios
    print("\n  RANKING BY RETURN/DD RATIO (Higher = Better)")
    sorted_by_rdd = sorted(metrics_list, key=lambda m: m.return_dd_ratio, reverse=True)
    for i, m in enumerate(sorted_by_rdd, 1):
        scenario = f"{m.brick_scenario} / {m.sizing_mode}"
        print(f"    {i}. {scenario:<40} {m.return_dd_ratio:.2f}x")

    print("\n  RANKING BY CALMAR RATIO (Higher = Better)")
    sorted_by_calmar = sorted(metrics_list, key=lambda m: m.calmar_ratio, reverse=True)
    for i, m in enumerate(sorted_by_calmar, 1):
        scenario = f"{m.brick_scenario} / {m.sizing_mode}"
        print(f"    {i}. {scenario:<40} {m.calmar_ratio:.2f}")

    print("\n  RANKING BY TOTAL RETURN")
    sorted_by_return = sorted(metrics_list, key=lambda m: m.total_return_usd, reverse=True)
    for i, m in enumerate(sorted_by_return, 1):
        scenario = f"{m.brick_scenario} / {m.sizing_mode}"
        print(f"    {i}. {scenario:<40} ${m.total_return_usd:>10,.2f}")


def run_equity_analytics():
    """Run A/B test with equity and drawdown analytics."""
    from kinetra.renko.backtest import (
        FilterParams,
        SizingMode,
        StopParams,
        VolSizingParams,
        backtest_instrument,
    )

    print("=" * 80)
    print("  A/B LOT SIZING TEST — EQUITY & DRAWDOWN ANALYTICS")
    print("=" * 80)

    # Load data
    print("\n[1] Loading data...")
    csv_file = PROJECT_ROOT / "XAUUSD_M1_accurate.csv"

    if not csv_file.exists():
        print("  ⚠ Using synthetic data fallback")
        np.random.seed(42)
        base_price = 2000.0
        returns = np.random.normal(0.0001, 0.002, 5000)
        prices = base_price * np.exp(np.cumsum(returns))
        closes = pd.Series(prices, name="close")
    else:
        df = pd.read_csv(csv_file, parse_dates=[0], index_col=0, nrows=10000)
        closes = df.iloc[:, 0].astype(float).dropna() if len(df.columns) > 0 else None

    if closes is None or len(closes) < 100:
        print("  ❌ Insufficient data")
        return

    print(f"  ✓ Loaded {len(closes)} bars")

    # Compute DSP brick
    print("\n[2] Computing DSP brick...")
    try:
        from kinetra.renko.dsp import vr_peak, vr_profile

        closes_arr = closes.values.astype(np.float64)
        scales = [5, 10, 20, 30, 50, 60, 90, 120]
        profile = vr_profile(closes_arr, scales=scales)
        if profile:
            peak_scale, _ = vr_peak(profile)
            returns = np.diff(np.log(closes_arr))
            n_windows = len(returns) // peak_scale
            displacements = []
            for i in range(max(1, n_windows)):
                window_ret = returns[i * peak_scale : (i + 1) * peak_scale]
                displacement = np.abs(np.sum(window_ret))
                displacements.append(displacement)
            dsp_brick = float(np.median(displacements)) if displacements else 0.5
        else:
            dsp_brick = 0.5
    except Exception as e:
        dsp_brick = 0.5

    static_brick = dsp_brick * 1.5
    print(f"  DSP brick: {dsp_brick:.4f}")
    print(f"  Static brick: {static_brick:.4f}")

    # Run tests
    print("\n[3] Running backtests with equity tracking...")
    filter_params = FilterParams()
    stop_params = StopParams()

    metrics_list = []

    scenarios = [
        ("DSP-Arrived", dsp_brick),
        ("Static Arbitrary", static_brick),
    ]

    for scenario_name, brick_size in scenarios:
        for sizing_mode, sizing_enum in [
            ("Static", SizingMode.FIXED_LOT),
            ("Compounded", SizingMode.COMPOUNDING),
        ]:
            print(f"  {scenario_name} + {sizing_mode}...", end=" ", flush=True)

            try:
                if sizing_enum == SizingMode.FIXED_LOT:
                    vol_params = VolSizingParams(fixed_lot=0.01)
                else:
                    vol_params = VolSizingParams(
                        fixed_lot=0.01,
                        compounding_capital_per_lot=1000.0,
                        initial_equity=100_000.0,
                    )

                result = backtest_instrument(
                    symbol="XAUUSD",
                    closes=closes,
                    brick_size=brick_size,
                    filter_params=filter_params,
                    stop_params=stop_params,
                    session_break_minutes=1.0,
                    sizing_mode=sizing_enum,
                    vol_sizing_params=vol_params,
                )

                metrics = compute_equity_metrics(
                    trades=result.trades,
                    omega=result.omega,
                    z_factor=result.z_factor,
                    max_dd_usd=result.max_dd_usd,
                    brick_scenario=scenario_name,
                    sizing_mode=sizing_mode,
                )

                metrics_list.append(metrics)
                print(
                    f"✓ ${metrics.final_equity:,.0f} final equity, "
                    f"Return/DD={metrics.return_dd_ratio:.2f}x"
                )

            except Exception as e:
                print(f"✗ Failed: {e}")

    # Print summaries
    print("\n" + "=" * 80)
    print("  DETAILED EQUITY SUMMARIES")
    print("=" * 80)

    for metrics in metrics_list:
        print_equity_summary(metrics)

    # Comparison
    if len(metrics_list) > 1:
        compare_equity_metrics(metrics_list)

    # Final verdict
    print(f"\n{'=' * 80}")
    print("  FINAL VERDICT")
    print(f"{'=' * 80}")

    sorted_by_return_dd = sorted(metrics_list, key=lambda m: m.return_dd_ratio, reverse=True)
    if sorted_by_return_dd:
        winner = sorted_by_return_dd[0]
        print("\n  🏆 BEST RETURN/DRAWDOWN RATIO:")
        print(f"     Scenario: {winner.brick_scenario}")
        print(f"     Sizing: {winner.sizing_mode}")
        print(f"     Return/DD: {winner.return_dd_ratio:.2f}x")
        print(f"     Total Return: ${winner.total_return_usd:,.2f}")
        print(f"     Max DD: ${winner.max_dd_usd:,.2f}")
        print(f"     Status: {'✅ DEPLOY' if winner.return_dd_ratio > 1.5 else '⚠️ REVIEW'}")


if __name__ == "__main__":
    run_equity_analytics()
