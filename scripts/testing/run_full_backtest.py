#!/usr/bin/env python3
"""
Full Physics Backtest Runner with Plotting

Runs all physics strategies (v7.0) on each CSV file and plots results.

Usage:
    python scripts/run_full_backtest.py /path/to/your/data/*.csv
    python scripts/run_full_backtest.py /home/renier/QuantumHunter/*.csv

Example:
    python scripts/run_full_backtest.py data/master/forex/EURUSD_H1_*.csv data/master/crypto/BTCUSD_H1_*.csv
"""

import os
import sys
import warnings
from datetime import datetime
from pathlib import Path

warnings.filterwarnings("ignore")

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from backtesting import Backtest

from kinetra.data_utils import get_data_summary, load_mt5_csv, validate_ohlcv
from kinetra.physics_backtester import (
    DampingReversionStrategy,
    EnergyMomentumStrategy,
    EntropyVolatilityStrategy,
    MultiPhysicsStrategy,
)
from kinetra.physics_v7 import (
    PhysicsEngineV7,
    calculate_omega_ratio,
    calculate_z_factor,
    validate_theorem_targets,
)

# Import Kinetra modules
from kinetra.strategies_v7 import (
    BerserkerStrategy,
    MultiAgentV7Strategy,
    SniperStrategy,
    list_v7_strategies,
)

# All strategies to test
ALL_STRATEGIES = {
    # v7.0 Strategies (Berserker/Sniper)
    "berserker": BerserkerStrategy,
    "sniper": SniperStrategy,
    "multi_agent_v7": MultiAgentV7Strategy,
    # Original Physics Strategies
    "energy_momentum": EnergyMomentumStrategy,
    "damping_reversion": DampingReversionStrategy,
    "entropy_volatility": EntropyVolatilityStrategy,
    "multi_physics": MultiPhysicsStrategy,
}


def load_csv_flexible(filepath: str) -> pd.DataFrame:
    """Load CSV with flexible format detection using data_utils."""
    # Use the robust MT5 loader from data_utils
    df = load_mt5_csv(filepath)

    # Ensure datetime index for backtesting.py
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.date_range(start="2020-01-01", periods=len(df), freq="h")

    return df


def run_backtest(df: pd.DataFrame, strategy_class, cash: float = 100000) -> dict:
    """Run a single backtest and return results."""
    try:
        bt = Backtest(
            df,
            strategy_class,
            cash=cash,
            commission=0.001,
            trade_on_close=True,
            exclusive_orders=True,
        )
        stats = bt.run()

        return {
            "success": True,
            "return_pct": float(stats["Return [%]"]),
            "sharpe": float(stats["Sharpe Ratio"]) if not np.isnan(stats["Sharpe Ratio"]) else 0,
            "max_dd": float(stats["Max. Drawdown [%]"]),
            "win_rate": float(stats["Win Rate [%]"]) if not np.isnan(stats["Win Rate [%]"]) else 0,
            "trades": int(stats["# Trades"]),
            "profit_factor": float(stats["Profit Factor"])
            if not np.isnan(stats["Profit Factor"])
            else 0,
            "final_equity": float(stats["Equity Final [$]"]),
            "buy_hold": float(stats["Buy & Hold Return [%]"]),
            "exposure": float(stats["Exposure Time [%]"]),
            "stats": stats,
            "bt": bt,
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "return_pct": 0,
            "sharpe": 0,
            "max_dd": 0,
            "win_rate": 0,
            "trades": 0,
            "profit_factor": 0,
            "final_equity": cash,
            "buy_hold": 0,
            "exposure": 0,
        }


def run_all_strategies(df: pd.DataFrame, strategies: dict = None) -> pd.DataFrame:
    """Run all strategies on a dataset."""
    if strategies is None:
        strategies = ALL_STRATEGIES

    results = []
    for name, strategy_class in strategies.items():
        print(f"  Running {name}...", end=" ")
        result = run_backtest(df, strategy_class)
        result["strategy"] = name
        results.append(result)

        if result["success"]:
            print(f"Return: {result['return_pct']:+.2f}%, Trades: {result['trades']}")
        else:
            print(f"FAILED: {result.get('error', 'Unknown error')}")

    return pd.DataFrame(results)


def plot_comparison(results_df: pd.DataFrame, title: str = "Strategy Comparison"):
    """Plot strategy comparison chart."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(title, fontsize=14, fontweight="bold")

    # Filter successful strategies
    df = results_df[results_df["success"]].copy()

    if len(df) == 0:
        print("No successful backtests to plot")
        return fig

    # Sort by return
    df = df.sort_values("return_pct", ascending=True)

    # 1. Returns comparison
    ax1 = axes[0, 0]
    colors = ["green" if x > 0 else "red" for x in df["return_pct"]]
    bars = ax1.barh(df["strategy"], df["return_pct"], color=colors, alpha=0.7)
    ax1.axvline(x=0, color="black", linewidth=0.5)
    ax1.axvline(
        x=df["buy_hold"].iloc[0],
        color="blue",
        linestyle="--",
        label=f"Buy&Hold: {df['buy_hold'].iloc[0]:.1f}%",
    )
    ax1.set_xlabel("Return (%)")
    ax1.set_title("Strategy Returns")
    ax1.legend()

    # 2. Risk-adjusted (Sharpe vs Return)
    ax2 = axes[0, 1]
    scatter = ax2.scatter(
        df["return_pct"], df["sharpe"], c=df["max_dd"].abs(), cmap="RdYlGn_r", s=100, alpha=0.7
    )
    # Vectorized annotations
    for strategy, return_pct, sharpe in zip(
        df["strategy"].values, df["return_pct"].values, df["sharpe"].values
    ):
        ax2.annotate(strategy, (return_pct, sharpe), fontsize=8, ha="center", va="bottom")
    ax2.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax2.axvline(x=0, color="gray", linestyle="--", alpha=0.5)
    ax2.set_xlabel("Return (%)")
    ax2.set_ylabel("Sharpe Ratio")
    ax2.set_title("Risk-Adjusted Performance")
    plt.colorbar(scatter, ax=ax2, label="Max Drawdown (%)")

    # 3. Win Rate vs Profit Factor
    ax3 = axes[1, 0]
    valid_pf = df[df["profit_factor"] > 0]
    if len(valid_pf) > 0:
        ax3.scatter(
            valid_pf["win_rate"], valid_pf["profit_factor"], s=valid_pf["trades"] * 2, alpha=0.7
        )
        # Vectorized annotations
        for strategy, win_rate, profit_factor in zip(
            valid_pf["strategy"].values,
            valid_pf["win_rate"].values,
            valid_pf["profit_factor"].values,
        ):
            ax3.annotate(strategy, (win_rate, profit_factor), fontsize=8, ha="center", va="bottom")
    ax3.axhline(y=1, color="red", linestyle="--", label="Break-even")
    ax3.set_xlabel("Win Rate (%)")
    ax3.set_ylabel("Profit Factor")
    ax3.set_title("Win Rate vs Profit Factor (size = # trades)")
    ax3.legend()

    # 4. Summary table
    ax4 = axes[1, 1]
    ax4.axis("off")

    # Create summary table
    summary_data = df[["strategy", "return_pct", "sharpe", "max_dd", "win_rate", "trades"]].copy()
    summary_data.columns = ["Strategy", "Return%", "Sharpe", "MaxDD%", "WinRate%", "Trades"]

    table = ax4.table(
        cellText=summary_data.round(2).values,
        colLabels=summary_data.columns,
        cellLoc="center",
        loc="center",
        colColours=["lightblue"] * 6,
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)
    ax4.set_title("Summary")

    plt.tight_layout()
    return fig


def plot_equity_curves(df: pd.DataFrame, results_df: pd.DataFrame, title: str = "Equity Curves"):
    """Plot equity curves for all strategies."""
    fig, ax = plt.subplots(figsize=(14, 7))

    # Vectorized equity curve plotting
    for success, bt_obj, strategy in zip(
        results_df["success"].values,
        results_df["bt"].values if "bt" in results_df.columns else [None] * len(results_df),
        results_df["strategy"].values
        if "strategy" in results_df.columns
        else [None] * len(results_df),
    ):
        if success and bt_obj is not None:
            try:
                stats = bt_obj.run()
                equity = stats._equity_curve["Equity"]
                ax.plot(equity.index, equity.values, label=strategy, alpha=0.8)
            except Exception as exc:
                warnings.warn(
                    f"Failed to plot equity curve for strategy {strategy if strategy else 'unknown'}: {exc}",
                    RuntimeWarning,
                )

    # Buy and hold
    buy_hold = 100000 * (df["Close"] / df["Close"].iloc[0])
    ax.plot(df.index, buy_hold.values, label="Buy & Hold", linestyle="--", color="black", alpha=0.5)

    ax.set_xlabel("Date")
    ax.set_ylabel("Equity ($)")
    ax.set_title(title)
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def analyze_physics(df: pd.DataFrame):
    """Analyze physics state of the data."""
    engine = PhysicsEngineV7()
    state = engine.compute_physics_state(df)

    print("\n  Physics Analysis:")
    print(f"    Energy (mean):    {state['energy'].mean():.6f}")
    print(f"    Damping (mean):   {state['damping'].mean():.4f}")
    print(f"    Entropy (mean):   {state['entropy'].mean():.4f}")

    regime_dist = state["regime"].value_counts(normalize=True) * 100
    print(f"    Regime distribution:")
    for regime, pct in regime_dist.items():
        print(f"      {regime}: {pct:.1f}%")

    agent_dist = state["active_agent"].value_counts(normalize=True) * 100
    print(f"    Agent activations:")
    for agent, pct in agent_dist.items():
        print(f"      {agent}: {pct:.1f}%")

    return state


def main():
    if len(sys.argv) < 2:
        raise ValueError(
            "Usage: python run_full_backtest.py <csv_file1> [csv_file2] ...\n\n"
            "Example:\n"
            "  python run_full_backtest.py data/*.csv\n"
            "  python run_full_backtest.py /home/renier/QuantumHunter/*.csv"
        )

    csv_files = sys.argv[1:]

    print("=" * 70)
    print(" KINETRA PHYSICS BACKTESTER")
    print(" Energy-Transfer Trading Theorem v7.0")
    print("=" * 70)
    print(f"\nFiles to process: {len(csv_files)}")
    print(f"Strategies: {list(ALL_STRATEGIES.keys())}")

    all_results = []

    for filepath in csv_files:
        if not os.path.exists(filepath):
            print(f"\nSkipping {filepath} - file not found")
            continue

        filename = os.path.basename(filepath)
        print(f"\n{'=' * 70}")
        print(f"Processing: {filename}")
        print("=" * 70)

        # Load data
        try:
            df = load_csv_flexible(filepath)
            print(f"  Loaded {len(df)} bars")
            print(f"  Date range: {df.index[0]} to {df.index[-1]}")
            print(f"  Price range: {df['Close'].min():.5f} - {df['Close'].max():.5f}")
        except Exception as e:
            print(f"  ERROR loading data: {e}")
            continue

        # Analyze physics
        try:
            state = analyze_physics(df)
        except Exception as e:
            print(f"  Physics analysis failed: {e}")

        # Run backtests
        print("\n  Running backtests...")
        results = run_all_strategies(df)
        results["file"] = filename
        all_results.append(results)

        # Best strategy
        best = results[results["success"]].sort_values("return_pct", ascending=False)
        if len(best) > 0:
            print(f"\n  BEST: {best.iloc[0]['strategy']} with {best.iloc[0]['return_pct']:+.2f}%")

        # Plot results
        try:
            fig1 = plot_comparison(results, f"{filename} - Strategy Comparison")
            fig1.savefig(
                f"results_{filename.replace('.csv', '')}_comparison.png",
                dpi=150,
                bbox_inches="tight",
            )
            print(f"  Saved: results_{filename.replace('.csv', '')}_comparison.png")

            fig2 = plot_equity_curves(df, results, f"{filename} - Equity Curves")
            fig2.savefig(
                f"results_{filename.replace('.csv', '')}_equity.png", dpi=150, bbox_inches="tight"
            )
            print(f"  Saved: results_{filename.replace('.csv', '')}_equity.png")

            plt.close("all")
        except Exception as e:
            print(f"  Plotting failed: {e}")

    # Combined summary
    if len(all_results) > 1:
        print("\n" + "=" * 70)
        print(" COMBINED SUMMARY")
        print("=" * 70)

        combined = pd.concat(all_results, ignore_index=True)

        # Average performance by strategy
        avg_perf = (
            combined[combined["success"]]
            .groupby("strategy")
            .agg(
                {
                    "return_pct": "mean",
                    "sharpe": "mean",
                    "max_dd": "mean",
                    "win_rate": "mean",
                    "trades": "sum",
                }
            )
            .round(2)
        )

        print("\nAverage Performance by Strategy:")
        print(avg_perf.sort_values("return_pct", ascending=False).to_string())

        # Save combined results
        combined.to_csv("backtest_results_combined.csv", index=False)
        print("\nSaved: backtest_results_combined.csv")

    print("\n" + "=" * 70)
    print(" BACKTEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
