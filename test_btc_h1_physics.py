"""
BTC H1 Physics Engine Test Pipeline

This script demonstrates:
1. Loading BTC H1 data and computing Layer-1 physics sensors
2. Regime clustering (GMM) and regime-age tracking
3. Running a baseline physics-based strategy
4. Analyzing results with CVaR and regime-aware metrics
5. Validating universal truths empirically
"""

import math
import sys
import warnings
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from kinetra.backtest_engine import BacktestEngine, TradeDirection

# Import Kinetra modules
from kinetra.physics_engine import PhysicsEngine
from kinetra.symbol_spec import SymbolSpec
from kinetra.volatility import (
    forward_energy_release,
    rogers_satchell,
)
from kinetra.volatility import (
    potential_energy as compute_pe,
)


def load_btc_h1_data(filepath: str) -> pd.DataFrame:
    """Load and clean BTC H1 OHLCV data."""
    print(f"\n{'=' * 60}")
    print(f"LOADING DATA: {filepath}")
    print(f"{'=' * 60}")

    # Try to detect format
    try:
        # First try tab-separated
        df = pd.read_csv(filepath, sep="\t", header=None, dtype=str)
        if df.shape[1] < 6:
            # Try comma-separated
            df = pd.read_csv(filepath, sep=",", header=None, dtype=str)
    except Exception:
        # Default to comma
        df = pd.read_csv(filepath, sep=",", header=None, dtype=str)

    # Assign column names based on typical format
    cols = ["date", "time", "open", "high", "low", "close", "volume", "tick_vol", "spread"]
    if df.shape[1] >= len(cols):
        df = df.iloc[:, : len(cols)]
        df.columns = cols
    else:
        # Minimal format
        min_cols = ["date", "time", "open", "high", "low", "close", "volume"]
        df = df.iloc[:, : len(min_cols)]
        df.columns = min_cols

    # Parse datetime
    try:
        df["datetime"] = pd.to_datetime(df["date"].astype(str) + " " + df["time"].astype(str))
    except Exception:
        try:
            df["datetime"] = pd.to_datetime(df["date"])
        except Exception:
            df["datetime"] = pd.date_range(start="2024-01-02", periods=len(df), freq="1H")

    df.set_index("datetime", inplace=True)
    df.sort_index(inplace=True)

    # Convert OHLCV to float
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["close"])

    print(f"Loaded {len(df)} bars")
    print(f"Date range: {df.index[0]} to {df.index[-1]}")
    print(f"Price range: ${df['close'].min():,.2f} to ${df['close'].max():,.2f}")

    return df


def compute_physics_state(df: pd.DataFrame, physics: PhysicsEngine) -> pd.DataFrame:
    """Compute full physics state with Layer-1 sensors."""
    print(f"\n{'=' * 60}")
    print("COMPUTING PHYSICS STATE")
    print(f"{'=' * 60}")

    close = df["close"]
    volume = df.get("volume") if "volume" in df.columns else None
    high = df.get("high") if "high" in df.columns else None
    low = df.get("low") if "low" in df.columns else None

    # Compute physics state
    physics_state = physics.compute_physics_state(
        prices=close,
        volume=volume,
        high=high,
        low=low,
        include_percentiles=True,
        include_kinematics=True,
        include_flow=True,
    )

    print(f"Computed physics state with {len(physics_state.columns)} columns")

    # Show regime distribution
    regime_counts = physics_state["regime"].value_counts()
    print("\nRegime distribution:")
    for regime, count in regime_counts.items():
        pct = count / len(physics_state) * 100
        print(f"  {regime}: {count} bars ({pct:.1f}%)")

    return physics_state


def analyze_regime_quality(
    physics_state: pd.DataFrame, df: pd.DataFrame
) -> Dict[str, Dict[str, float]]:
    """Analyze return distribution by regime (empirical test of universal truths)."""
    print(f"\n{'=' * 60}")
    print("REGIME QUALITY ANALYSIS (Universal Truths Test)")
    print(f"{'=' * 60}")

    # Compute returns
    log_returns = np.log(df["close"]).diff()

    # Create analysis dataframe
    analysis = pd.DataFrame(
        {
            "regime": physics_state["regime"],
            "cluster": physics_state["cluster"],
            "KE_pct": physics_state.get("KE_pct", 0.5),
            "Re_m_pct": physics_state.get("Re_m_pct", 0.5),
            "zeta_pct": physics_state.get("zeta_pct", 0.5),
            "PE_pct": physics_state.get("PE_pct", 0.5),
            "Hs_pct": physics_state.get("Hs_pct", 0.5),
            "eta_pct": physics_state.get("eta_pct", 0.5),
            "return": log_returns,
        }
    )

    results = {}

    for regime in analysis["regime"].unique():
        if regime == "UNKNOWN":
            continue
        sub = analysis[analysis["regime"] == regime]
        returns = sub["return"].dropna()

        if len(returns) < 10:
            continue

        # Basic stats
        mean_ret = returns.mean()
        std_ret = returns.std()

        # Annualized Sharpe (assuming H1)
        if std_ret > 0:
            sharpe = mean_ret / std_ret * np.sqrt(8760)
        else:
            sharpe = 0.0

        # CVaR
        q5 = returns.quantile(0.05)
        q1 = returns.quantile(0.01)
        cvar_95 = returns[returns <= q5].mean()
        cvar_99 = returns[returns <= q1].mean()

        # Omega
        gains = returns[returns > 0].sum()
        losses = abs(returns[returns <= 0].sum())
        omega = gains / losses if losses > 0 else float("inf")

        results[regime] = {
            "bars": len(returns),
            "mean_return": mean_ret,
            "std_return": std_ret,
            "sharpe_h1": sharpe,
            "cvar_95": cvar_95,
            "cvar_99": cvar_99,
            "omega": omega,
            "avg_KE_pct": sub["KE_pct"].mean(),
            "avg_Re_pct": sub["Re_m_pct"].mean(),
            "avg_zeta_pct": sub["zeta_pct"].mean(),
            "avg_PE_pct": sub["PE_pct"].mean(),
            "avg_Hs_pct": sub["Hs_pct"].mean(),
        }

        print(f"\n{regime}:")
        print(f"  Bars: {len(returns)}")
        print(f"  Mean return: {mean_ret:.6f} ({mean_ret * 10000:.2f} bps)")
        print(f"  Std return: {std_ret:.6f}")
        print(f"  Sharpe (H1): {sharpe:.2f}")
        print(f"  CVaR 95%: {cvar_95:.6f}")
        print(f"  CVaR 99%: {cvar_99:.6f}")
        print(f"  Omega: {omega:.2f}")
        print(f"  Avg KE_pct: {sub['KE_pct'].mean():.3f}")
        print(f"  Avg Re_m_pct: {sub['Re_m_pct'].mean():.3f}")
        print(f"  Avg zeta_pct: {sub['zeta_pct'].mean():.3f}")

    return results


def create_btc_symbol_spec() -> SymbolSpec:
    """Create SymbolSpec for BTCUSD."""
    from kinetra.symbol_spec import CommissionType, SwapType

    spec = SymbolSpec(
        symbol="BTCUSD",
        tick_size=0.01,
        tick_value=0.01,  # Per unit
        contract_size=1.0,
        spread_points=2.0,  # Typical 2 pip spread
        spread_cost_multiplier=1.0,
        commission=CommissionType.PER_LOT,
        commission_value=0.0,  # Crypto typically no commission
        swap=SwapType.POINTS,
        swap_long=-0.01,  # Daily swap for long
        swap_short=0.005,  # Daily swap for short
        volume_min=0.01,
        volume_max=10.0,
        volume_step=0.01,
        margin_percent=10.0,  # 10% margin requirement
        stop_out_level=0.5,  # 50% stop-out
    )

    return spec


def physics_based_signal(
    row: pd.Series,
    physics_state: pd.DataFrame,
    bar_index: int,
) -> int:
    """
    Physics-based trading signal.

    Rules (based on empirical analysis):
    - Long only in UNDERDAMPED or LAMINAR regimes
    - High Reynolds (Re_m_pct > 0.7) = clean trend
    - Low damping (zeta_pct < 0.4) = low friction
    - High PE (PE_pct > 0.6) = stored energy
    """
    if bar_index >= len(physics_state):
        return 0

    regime = str(physics_state.iloc[bar_index].get("regime", "UNKNOWN"))

    # Skip bad regimes
    if regime in ["OVERDAMPED", "UNKNOWN"]:
        return 0

    # Extract Layer-1 sensors
    re_pct = physics_state.iloc[bar_index].get("Re_m_pct", 0.5)
    zeta_pct = physics_state.iloc[bar_index].get("zeta_pct", 0.5)
    pe_pct = physics_state.iloc[bar_index].get("PE_pct", 0.5)
    hs_pct = physics_state.iloc[bar_index].get("Hs_pct", 0.5)

    # Regime-specific entry conditions
    if regime == "UNDERDAMPED":
        # Trend regime: require high Re, low zeta
        if re_pct > 0.7 and zeta_pct < 0.4:
            return 1  # Long
    elif regime == "LAMINAR":
        # Clean flow: moderate Re, low entropy
        if re_pct > 0.5 and hs_pct < 0.4 and pe_pct > 0.5:
            return 1  # Long
    elif regime == "BREAKOUT":
        # Breakout: require high PE + low entropy
        if pe_pct > 0.6 and hs_pct < 0.5:
            return 1  # Long

    return 0


def run_backtest(df: pd.DataFrame, symbol_spec: SymbolSpec) -> Tuple[Any, pd.DataFrame]:
    """Run the backtest with physics-based signals."""
    print(f"\n{'=' * 60}")
    print("RUNNING BACKTEST")
    print(f"{'=' * 60}")

    engine = BacktestEngine(
        initial_capital=100000.0,
        risk_per_trade=0.02,  # 2% risk per trade
        max_positions=1,
        timeframe="H1",
    )

    result = engine.run_backtest(
        data=df,
        symbol_spec=symbol_spec,
        signal_func=physics_based_signal,
    )

    return result, engine.physics.compute_physics_state(df["close"])


def analyze_trade_results(result: Any) -> None:
    """Analyze and display backtest results."""
    print(f"\n{'=' * 60}")
    print("BACKTEST RESULTS")
    print(f"{'=' * 60}")

    print(f"\nTotal trades: {result.total_trades}")
    print(f"Win rate: {result.win_rate * 100:.1f}%")
    print(f"Winning trades: {result.winning_trades}")
    print(f"Losing trades: {result.losing_trades}")

    print(f"\nP&L Summary:")
    print(f"  Gross profit: ${result.gross_profit:+,.2f}")
    print(f"  Gross loss: ${result.gross_loss:+,.2f}")
    print(f"  Total costs: ${result.total_costs:+,.2f}")
    print(f"  Net P&L: ${result.total_net_pnl:+,.2f}")

    print(f"\nCost Breakdown:")
    print(f"  Spread: ${result.total_spread_cost:+,.2f}")
    print(f"  Commission: ${result.total_commission:+,.2f}")
    print(f"  Slippage: ${result.total_slippage:+,.2f}")
    print(f"  Swap: ${result.total_swap_cost:+,.2f}")

    print(f"\nRisk Metrics:")
    print(f"  Max drawdown: ${result.max_drawdown:,.2f} ({result.max_drawdown_pct:.2f}%)")
    print(f"  Sharpe ratio: {result.sharpe_ratio:.2f}")
    print(f"  Sortino ratio: {result.sortino_ratio:.2f}")
    print(f"  Omega ratio: {result.omega_ratio:.2f}")
    print(f"  Z-factor: {result.z_factor:.2f}")

    # CVaR metrics (new)
    if hasattr(result, "cvar_95"):
        print(f"\nCVaR Metrics (Downside Risk):")
        print(f"  CVaR 95%: {result.cvar_95 * 100:.3f}%")
        print(f"  CVaR 99%: {result.cvar_99 * 100:.3f}%")

    # Energy metrics
    print(f"\nPhysics Metrics:")
    print(f"  Energy captured %: {result.energy_captured_pct * 100:.1f}%")
    print(f"  MFE capture %: {result.mfe_capture_pct * 100:.1f}%")

    # Min margin level
    if hasattr(result, "min_margin_level") and result.min_margin_level != float("inf"):
        print(f"\nMargin:")
        print(f"  Min margin level: {result.min_margin_level:.1f}%")

    return result


def analyze_trade_lifecycle(result: Any) -> None:
    """Analyze trade lifecycle and physics metrics."""
    print(f"\n{'=' * 60}")
    print("TRADE LIFECYCLE ANALYSIS")
    print(f"{'=' * 60}")

    if not result.trades:
        print("No trades to analyze")
        return

    # Compute aggregate lifecycle metrics
    trades = result.trades

    # MFE efficiency stats
    mfe_efficiencies = [t.mfe_efficiency for t in trades]
    avg_mfe_eff = np.mean(mfe_efficiencies)

    # MAE efficiency stats
    mae_efficiencies = [t.mae_efficiency for t in trades]
    avg_mae_eff = np.mean(mae_efficiencies)

    # Stop efficiency
    stop_efficiencies = [t.stop_efficiency for t in trades if t.stop_efficiency > 0]
    avg_stop_eff = np.mean(stop_efficiencies) if stop_efficiencies else 0.0

    # R-multiple distribution
    r_multiples = [t.r_multiple for t in trades]

    # Physics metrics
    avg_energy_entry = np.mean([t.energy_at_entry for t in trades])
    avg_pe_entry = np.mean([t.pe_at_entry for t in trades])

    # Regime at entry distribution
    regime_counts = {}
    for t in trades:
        r = t.regime_at_entry
        regime_counts[r] = regime_counts.get(r, 0) + 1

    print(f"\nTrade Quality Metrics:")
    print(f"  Avg MFE efficiency: {avg_mfe_eff * 100:.1f}%")
    print(f"  Avg MAE efficiency: {avg_mae_eff * 100:.1f}%")
    print(f"  Avg stop efficiency: {avg_stop_eff * 100:.1f}%")

    print(f"\nR-Multiple Distribution:")
    print(f"  Min: {min(r_multiples):.2f}")
    print(f"  Max: {max(r_multiple for r_multiple in r_multiples):.2f}")
    print(f"  Mean: {np.mean(r_multiples):.2f}")
    print(f"  Median: {np.median(r_multiples):.2f}")

    print(f"\nPhysics at Entry:")
    print(f"  Avg energy: {avg_energy_entry:.4f}")
    print(f"  Avg PE: {avg_pe_entry:.3f}")

    print(f"\nRegime at Entry:")
    for regime, count in sorted(regime_counts.items(), key=lambda x: -x[1]):
        pct = count / len(trades) * 100
        print(f"  {regime}: {count} trades ({pct:.1f}%)")

    # Show top 5 trades by net P&L
    print(f"\nTop 5 Trades by Net P&L:")
    sorted_trades = sorted(trades, key=lambda t: t.net_pnl, reverse=True)[:5]
    for i, t in enumerate(sorted_trades, 1):
        print(
            f"  {i}. {t.entry_time.strftime('%Y-%m-%d %H:%M')} | "
            f"P&L: ${t.net_pnl:+.2f} | "
            f"MFE: {t.mfe_efficiency * 100:.0f}% | "
            f"Regime: {t.regime_at_entry}"
        )

    # Show worst 5 trades
    print(f"\nWorst 5 Trades by Net P&L:")
    sorted_trades = sorted(trades, key=lambda t: t.net_pnl)[:5]
    for i, t in enumerate(sorted_trades, 1):
        print(
            f"  {i}. {t.entry_time.strftime('%Y-%m-%d %H:%M')} | "
            f"P&L: ${t.net_pnl:+.2f} | "
            f"MAE: {t.mae_efficiency * 100:.0f}% | "
            f"Regime: {t.regime_at_entry}"
        )


def generate_summary_report(
    df: pd.DataFrame,
    physics_state: pd.DataFrame,
    regime_results: Dict[str, Dict[str, float]],
    backtest_result: Any,
) -> str:
    """Generate a summary report."""
    report = []

    report.append("=" * 60)
    report.append("BTC H1 PHYSICS BACKTEST SUMMARY REPORT")
    report.append("=" * 60)
    report.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    report.append(f"\nData Period:")
    report.append(f"  Start: {df.index[0]}")
    report.append(f"  End: {df.index[-1]}")
    report.append(f"  Bars: {len(df)}")

    report.append(f"\nRegime Distribution:")
    regime_dist = physics_state["regime"].value_counts()
    for regime, count in regime_dist.items():
        pct = count / len(physics_state) * 100
        report.append(f"  {regime}: {count} ({pct:.1f}%)")

    report.append(f"\nRegime Quality (from empirical analysis):")
    for regime, stats in regime_results.items():
        report.append(f"  {regime}:")
        report.append(f"    Sharpe (H1): {stats.get('sharpe_h1', 0):.2f}")
        report.append(f"    CVaR 95%: {stats.get('cvar_95', 0) * 100:.3f}%")
        report.append(f"    Omega: {stats.get('omega', 0):.2f}")

    report.append(f"\nBacktest Results:")
    report.append(f"  Total trades: {backtest_result.total_trades}")
    report.append(f"  Win rate: {backtest_result.win_rate * 100:.1f}%")
    report.append(f"  Net P&L: ${backtest_result.total_net_pnl:+,.2f}")
    report.append(f"  Sharpe ratio: {backtest_result.sharpe_ratio:.2f}")
    report.append(f"  Max drawdown: {backtest_result.max_drawdown_pct:.2f}%")
    report.append(f"  CVaR 95%: {getattr(backtest_result, 'cvar_95', 0) * 100:.3f}%")
    report.append(f"  Energy captured: {backtest_result.energy_captured_pct * 100:.1f}%")

    report.append(f"\nConclusion:")

    # Determine best and worst regimes
    valid_regimes = {k: v for k, v in regime_results.items() if "sharpe_h1" in v}
    if valid_regimes:
        best_regime = max(valid_regimes.items(), key=lambda x: x[1].get("sharpe_h1", 0))
        worst_regime = min(valid_regimes.items(), key=lambda x: x[1].get("sharpe_h1", 0))

        report.append(
            f"  Best performing regime: {best_regime[0]} (Sharpe: {best_regime[1].get('sharpe_h1', 0):.2f})"
        )
        report.append(
            f"  Worst performing regime: {worst_regime[0]} (Sharpe: {worst_regime[1].get('sharpe_h1', 0):.2f})"
        )

        # Universal truth validation
        report.append(f"\nUniversal Truth Validation:")
        report.append(
            f"  - High Re_m + Low ζ → Positive Sharpe: {best_regime[1].get('sharpe_h1', 0) > 0}"
        )
        report.append(f"  - Energy captured > 0: {backtest_result.energy_captured_pct > 0}")
        report.append(f"  - MFE capture > 0: {backtest_result.mfe_capture_pct > 0}")

    report.append("\n" + "=" * 60)

    return "\n".join(report)


def main():
    """Main execution function."""
    warnings.filterwarnings("ignore")

    # Configuration
    DATA_PATH = "data/BTCUSD_H1_202401020000_202512282200.csv"

    print("\n" + "=" * 60)
    print("BTC H1 PHYSICS ENGINE TEST PIPELINE")
    print("=" * 60)

    # Step 1: Load data
    df = load_btc_h1_data(DATA_PATH)

    # Step 2: Initialize physics engine
    physics = PhysicsEngine(
        vel_window=1,
        damping_window=64,
        entropy_window=64,
        re_slow=24,
        re_fast=6,
        pe_window=72,
        pct_window=500,
        n_clusters=4,
        random_state=42,
    )

    # Step 3: Compute physics state
    physics_state = compute_physics_state(df, physics)

    # Step 4: Analyze regime quality (universal truths test)
    regime_results = analyze_regime_quality(physics_state, df)

    # Step 5: Create symbol spec and run backtest
    symbol_spec = create_btc_symbol_spec()
    backtest_result, _ = run_backtest(df, symbol_spec)

    # Step 6: Analyze results
    analyze_trade_results(backtest_result)
    analyze_trade_lifecycle(backtest_result)

    # Step 7: Generate summary report
    report = generate_summary_report(df, physics_state, regime_results, backtest_result)
    print("\n" + report)

    # Save report to file
    report_path = "btc_h1_backtest_report.txt"
    with open(report_path, "w") as f:
        f.write(report)
    print(f"\nReport saved to: {report_path}")

    # Save trade log
    trade_log_path = "btc_h1_trade_log.csv"
    trade_df = backtest_result.trade_log()
    trade_df.to_csv(trade_log_path, index=False)
    print(f"Trade log saved to: {trade_log_path}")

    print(f"\n{'=' * 60}")
    print("PIPELINE COMPLETE")
    print(f"{'=' * 60}\n")

    return backtest_result


if __name__ == "__main__":
    result = main()
