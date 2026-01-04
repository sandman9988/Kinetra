#!/usr/bin/env python3
"""
BTC H1 Layer-1 Validation Script

Validates the enhanced physics engine and Layer-1 sensors on BTC hourly data:
1. Load and clean BTC H1 data
2. Compute Layer-1 physics state
3. Run regime clustering (GMM)
4. Validate "universal truths" per regime
5. Run simple baseline strategy to validate engine

This script serves as the reference implementation for validating
the physics-based approach before extending to other instruments.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from kinetra.backtest_engine import BacktestEngine, BacktestResult
from kinetra.physics_engine import PhysicsEngine
from kinetra.single_symbol_env import DummyPhysicsAgent, SingleSymbolRLEnv, run_episode
from kinetra.symbol_spec import SymbolSpec


def load_btc_h1_data(filepath: str | Path) -> pd.DataFrame:
    """
    Load BTC H1 data from MT5 export format.

    MT5 format: tab-separated with <DATE>, <TIME>, <OPEN>, <HIGH>, <LOW>, <CLOSE>, <VOL>
    """
    print(f"Loading data from: {filepath}")

    df = pd.read_csv(
        filepath,
        sep="\t",
        parse_dates={"datetime": ["<DATE>", "<TIME>"]},
        date_format="%Y.%m.%d %H:%M:%S",
    )

    # Rename columns to lowercase
    df.columns = [c.replace("<", "").replace(">", "").lower() for c in df.columns]

    # Set datetime index
    df = df.set_index("datetime")
    df = df.sort_index()

    # Rename vol to volume if needed
    if "vol" in df.columns and "volume" not in df.columns:
        df = df.rename(columns={"vol": "volume"})
    if "tickvol" in df.columns and "volume" not in df.columns:
        df = df.rename(columns={"tickvol": "volume"})

    # Validate data
    print(f"  Date range: {df.index[0]} to {df.index[-1]}")
    print(f"  Total bars: {len(df):,}")
    print(f"  Columns: {list(df.columns)}")

    # Check for issues
    duplicates = df.index.duplicated().sum()
    if duplicates > 0:
        print(f"  WARNING: {duplicates} duplicate timestamps, removing...")
        df = df[~df.index.duplicated(keep="first")]

    # Check for backward time jumps
    time_diffs = df.index.to_series().diff()
    backward_jumps = (time_diffs < pd.Timedelta(0)).sum()
    if backward_jumps > 0:
        print(f"  WARNING: {backward_jumps} backward time jumps detected!")

    return df


def compute_layer1_physics(df: pd.DataFrame) -> pd.DataFrame:
    """Compute full Layer-1 physics state from OHLCV data."""
    print("\nComputing Layer-1 physics state...")

    engine = PhysicsEngine(lookback=20)

    physics_state = engine.compute_physics_state(
        prices=df["close"],
        volume=df["volume"] if "volume" in df.columns else None,
        high=df["high"],
        low=df["low"],
        open_price=df["open"],
        include_percentiles=True,
        include_kinematics=True,
        include_flow=("volume" in df.columns),
    )

    print(f"  Physics columns: {list(physics_state.columns)}")
    print(f"  Shape: {physics_state.shape}")

    return physics_state


def analyze_regime_statistics(df: pd.DataFrame, physics: pd.DataFrame) -> dict:
    """
    Analyze statistics per regime/cluster.

    Returns dict with per-regime metrics.
    """
    print("\n" + "=" * 60)
    print("REGIME STATISTICS")
    print("=" * 60)

    # Use physics which already has close column
    combined = physics.copy()

    # Calculate returns for metrics (use the close from physics)
    combined["returns"] = combined["close"].pct_change()

    results = {}

    for cluster in sorted(combined["cluster"].dropna().unique()):
        cluster = int(cluster)
        mask = combined["cluster"] == cluster
        subset = combined[mask]

        if len(subset) < 10:
            continue

        returns = subset["returns"].dropna()

        # Basic stats
        count = len(subset)
        pct = count / len(combined) * 100

        # Return metrics
        mean_ret = returns.mean() * 100  # Convert to %
        std_ret = returns.std() * 100

        # Sharpe (annualized for H1: 8760 bars/year)
        if std_ret > 0:
            sharpe = (mean_ret / std_ret) * np.sqrt(8760)
        else:
            sharpe = 0.0

        # CVaR 95
        q05 = returns.quantile(0.05)
        cvar_95 = returns[returns <= q05].mean() * 100 if len(returns[returns <= q05]) > 0 else 0

        # CVaR 99
        q01 = returns.quantile(0.01)
        cvar_99 = returns[returns <= q01].mean() * 100 if len(returns[returns <= q01]) > 0 else 0

        # Physics averages
        avg_ke_pct = subset["KE_pct"].mean() if "KE_pct" in subset else 0
        avg_re_pct = subset["Re_m_pct"].mean() if "Re_m_pct" in subset else 0
        avg_zeta_pct = subset["zeta_pct"].mean() if "zeta_pct" in subset else 0
        avg_hs_pct = subset["Hs_pct"].mean() if "Hs_pct" in subset else 0
        avg_bp = subset["BP"].mean() if "BP" in subset else 0.5

        # Regime label
        regime_labels = {0: "UNDERDAMPED", 1: "CRITICAL", 2: "OVERDAMPED", 3: "BREAKOUT"}
        label = regime_labels.get(cluster, f"CLUSTER_{cluster}")

        results[cluster] = {
            "label": label,
            "count": count,
            "pct": pct,
            "mean_ret_bps": mean_ret * 100,  # basis points
            "sharpe": sharpe,
            "cvar_95": cvar_95,
            "cvar_99": cvar_99,
            "avg_KE_pct": avg_ke_pct,
            "avg_Re_pct": avg_re_pct,
            "avg_zeta_pct": avg_zeta_pct,
            "avg_Hs_pct": avg_hs_pct,
            "avg_BP": avg_bp,
        }

        print(f"\nCluster {cluster} ({label}):")
        print(f"  Bars: {count:,} ({pct:.1f}%)")
        print(f"  Mean Return: {mean_ret * 100:.2f} bps/bar")
        print(f"  Sharpe (annualized): {sharpe:.2f}")
        print(f"  CVaR 95: {cvar_95:.4f}%")
        print(f"  CVaR 99: {cvar_99:.4f}%")
        print(f"  Avg KE%: {avg_ke_pct:.2f}, Re%: {avg_re_pct:.2f}, ζ%: {avg_zeta_pct:.2f}")

    return results


def validate_universal_truths(regime_stats: dict) -> None:
    """
    Validate hypothesized "universal truths" about physics regimes.

    Expected:
    - High Re_m + low zeta → better trend following (UNDERDAMPED)
    - High PE_pct + low entropy → bigger forward releases
    - BREAKOUT cluster should have highest absolute returns
    """
    print("\n" + "=" * 60)
    print("UNIVERSAL TRUTH VALIDATION")
    print("=" * 60)

    # Check UNDERDAMPED (cluster 0) has best Sharpe
    if 0 in regime_stats and 2 in regime_stats:
        underdamped_sharpe = regime_stats[0]["sharpe"]
        overdamped_sharpe = regime_stats[2]["sharpe"]

        if underdamped_sharpe > overdamped_sharpe:
            print(
                f"✓ UNDERDAMPED Sharpe ({underdamped_sharpe:.2f}) > OVERDAMPED ({overdamped_sharpe:.2f})"
            )
        else:
            print(
                f"✗ UNDERDAMPED Sharpe ({underdamped_sharpe:.2f}) <= OVERDAMPED ({overdamped_sharpe:.2f})"
            )

    # Check UNDERDAMPED has high Re% and low zeta%
    if 0 in regime_stats:
        re_pct = regime_stats[0]["avg_Re_pct"]
        zeta_pct = regime_stats[0]["avg_zeta_pct"]

        if re_pct > 0.5:
            print(f"✓ UNDERDAMPED has high Re% ({re_pct:.2f} > 0.5)")
        else:
            print(f"✗ UNDERDAMPED has low Re% ({re_pct:.2f})")

        if zeta_pct < 0.5:
            print(f"✓ UNDERDAMPED has low ζ% ({zeta_pct:.2f} < 0.5)")
        else:
            print(f"✗ UNDERDAMPED has high ζ% ({zeta_pct:.2f})")

    # Check OVERDAMPED has lower absolute returns
    if 0 in regime_stats and 2 in regime_stats:
        underdamped_ret = abs(regime_stats[0]["mean_ret_bps"])
        overdamped_ret = abs(regime_stats[2]["mean_ret_bps"])

        if underdamped_ret > overdamped_ret:
            print(
                f"✓ UNDERDAMPED |returns| ({underdamped_ret:.2f}bps) > OVERDAMPED ({overdamped_ret:.2f}bps)"
            )
        else:
            print(
                f"✗ UNDERDAMPED |returns| ({underdamped_ret:.2f}bps) <= OVERDAMPED ({overdamped_ret:.2f}bps)"
            )

    # Check BREAKOUT has highest CVaR (most volatile)
    if 3 in regime_stats:
        breakout_cvar = abs(regime_stats[3]["cvar_95"])
        other_cvars = [abs(v["cvar_95"]) for k, v in regime_stats.items() if k != 3]

        if other_cvars and breakout_cvar > max(other_cvars):
            print(f"✓ BREAKOUT has highest CVaR95 ({breakout_cvar:.4f}%)")
        elif other_cvars:
            print(
                f"? BREAKOUT CVaR95 ({breakout_cvar:.4f}%) vs max others ({max(other_cvars):.4f}%)"
            )


def create_btc_symbol_spec() -> SymbolSpec:
    """Create SymbolSpec for BTCUSD."""
    return SymbolSpec(
        symbol="BTCUSD",
        description="Bitcoin vs US Dollar",
        base_currency="BTC",
        quote_currency="USD",
        tick_size=0.01,
        tick_value=0.01,  # $0.01 per 0.01 price move per 1 lot
        contract_size=1.0,  # 1 BTC per lot
        digits=2,
        volume_min=0.01,
        volume_max=100.0,
        volume_step=0.01,
        spread_points=5000.0,  # 50 USD spread (5000 * 0.01 tick)
        spread_min=1000.0,
        spread_max=20000.0,
        slippage_avg=2000.0,  # 20 USD slippage (2000 * 0.01 tick)
        slippage_max=10000.0,
        margin_initial=0.10,  # 10% margin (10x leverage)
        margin_maintenance=0.05,
        stop_out_level=0.50,
    )


def simple_physics_signal(row: pd.Series, physics_state: pd.DataFrame, bar_index: int) -> int:
    """
    Simple physics-based signal function for baseline validation.

    Long when:
    - KE_pct > 0.6 (high energy)
    - zeta_pct < 0.5 (low friction)
    - cluster in [0, 3] (UNDERDAMPED or BREAKOUT)
    - BP > 0.55 (buying pressure)

    Short when:
    - Same energy/friction conditions
    - BP < 0.45 (selling pressure)

    Flat: otherwise (CRITICAL or OVERDAMPED regimes)
    """
    if bar_index >= len(physics_state):
        return 0

    ps = physics_state.iloc[bar_index]

    # Extract features with defaults
    ke_pct = ps.get("KE_pct", 0.5)
    zeta_pct = ps.get("zeta_pct", 0.5)
    eta_pct = ps.get("eta_pct", 0.5)
    cluster = int(ps.get("cluster", 1))
    bp = ps.get("BP", 0.5)
    velocity = ps.get("velocity", 0.0)

    # Only trade in favorable regimes (UNDERDAMPED=0 or BREAKOUT=3)
    if cluster not in [0, 3]:
        return 0

    # Energy and friction conditions
    high_energy = ke_pct > 0.6
    low_friction = zeta_pct < 0.5

    # Combined momentum condition
    if not (high_energy and low_friction):
        return 0

    # Direction from buying pressure and velocity
    if bp > 0.55 and velocity > 0:
        return 1  # Long
    elif bp < 0.45 and velocity < 0:
        return -1  # Short

    return 0  # Flat


def run_baseline_backtest(df: pd.DataFrame, physics: pd.DataFrame) -> BacktestResult:
    """Run baseline physics strategy backtest."""
    print("\n" + "=" * 60)
    print("BASELINE STRATEGY BACKTEST")
    print("=" * 60)

    spec = create_btc_symbol_spec()

    engine = BacktestEngine(
        initial_capital=100000.0,
        risk_per_trade=0.01,
        max_positions=1,
        timeframe="H1",
        data_quality_check=False,  # Skip for now
    )

    result = engine.run_backtest(
        data=df,
        symbol_spec=spec,
        signal_func=simple_physics_signal,
    )

    print(f"\nResults:")
    print(f"  Total Trades: {result.total_trades}")
    print(f"  Win Rate: {result.win_rate * 100:.1f}%")
    print(f"  Net P&L: ${result.total_net_pnl:,.2f}")
    print(f"  Sharpe Ratio: {result.sharpe_ratio:.2f}")
    print(f"  Sortino Ratio: {result.sortino_ratio:.2f}")
    print(f"  Max Drawdown: {result.max_drawdown_pct:.1f}%")
    print(f"  CVaR 95: {result.cvar_95 * 100:.4f}%")
    print(f"  CVaR 99: {result.cvar_99 * 100:.4f}%")
    print(f"  Omega Ratio: {result.omega_ratio:.2f}")
    print(f"  MFE Capture: {result.mfe_capture_pct * 100:.1f}%")
    print(f"  Energy Captured: {result.energy_captured_pct * 100:.1f}%")

    # Cost breakdown
    print(f"\nCost Breakdown:")
    print(f"  Spread: ${result.total_spread_cost:,.2f}")
    print(f"  Commission: ${result.total_commission:,.2f}")
    print(f"  Slippage: ${result.total_slippage:,.2f}")
    print(f"  Swap: ${result.total_swap_cost:,.2f}")
    print(f"  Total Costs: ${result.total_costs:,.2f}")

    return result


def analyze_trade_lifecycle(result: BacktestResult, n_trades: int = 3) -> None:
    """Print detailed lifecycle for a few trades."""
    if not result.trades:
        print("\nNo trades to analyze")
        return

    print("\n" + "=" * 60)
    print("TRADE LIFECYCLE ANALYSIS")
    print("=" * 60)

    # Analyze first few profitable and losing trades
    winners = [t for t in result.trades if t.net_pnl > 0][:n_trades]
    losers = [t for t in result.trades if t.net_pnl <= 0][:n_trades]

    for trade in winners + losers:
        result.print_trade_lifecycle(trade.trade_id)


def test_rl_environment(df: pd.DataFrame) -> None:
    """Test the RL environment with random and heuristic policies."""
    print("\n" + "=" * 60)
    print("RL ENVIRONMENT TEST")
    print("=" * 60)

    try:
        # Create environment
        env = SingleSymbolRLEnv(
            data=df,
            initial_equity=100_000.0,
            start_offset=500,  # Skip warmup
            spread_pct=0.0005,
            slippage_pct=0.0002,
        )

        print(f"  Environment created successfully")
        print(f"  Observation dim: {env.observation_dim}")
        print(f"  Action dim: {env.action_dim}")
        print(f"  Total bars: {env.n_bars}")
        print(f"  Start offset: {env.start_offset}")

        # Test with random policy
        print("\n  Running random policy...")
        random_stats = run_episode(env, random_policy=True)
        print(f"    Final equity: ${random_stats['final_equity']:,.2f}")
        print(f"    Total return: {random_stats['total_return_pct']:.2f}%")
        print(f"    Sharpe ratio: {random_stats['sharpe_ratio']:.2f}")
        print(f"    Max drawdown: {random_stats['max_drawdown_pct']:.1f}%")
        print(f"    Trade count: {random_stats['trade_count']}")

        # Test with physics-based agent
        print("\n  Running physics-based agent...")
        agent = DummyPhysicsAgent()
        agent_stats = run_episode(env, agent=agent)
        print(f"    Final equity: ${agent_stats['final_equity']:,.2f}")
        print(f"    Total return: {agent_stats['total_return_pct']:.2f}%")
        print(f"    Sharpe ratio: {agent_stats['sharpe_ratio']:.2f}")
        print(f"    Max drawdown: {agent_stats['max_drawdown_pct']:.1f}%")
        print(f"    Trade count: {agent_stats['trade_count']}")

        # Compare
        print("\n  Comparison:")
        if agent_stats["total_return_pct"] > random_stats["total_return_pct"]:
            print(
                f"    ✓ Physics agent outperforms random by {agent_stats['total_return_pct'] - random_stats['total_return_pct']:.2f}%"
            )
        else:
            print(
                f"    ✗ Random outperforms physics agent by {random_stats['total_return_pct'] - agent_stats['total_return_pct']:.2f}%"
            )

    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback

        traceback.print_exc()


def main():
    """Main validation pipeline."""
    print("=" * 60)
    print("BTC H1 LAYER-1 VALIDATION")
    print("=" * 60)

    # 1. Load data
    data_path = project_root / "data" / "master" / "BTCUSD_H1_202401020000_202512282200.csv"

    if not data_path.exists():
        print(f"ERROR: Data file not found: {data_path}")
        return

    df = load_btc_h1_data(data_path)

    # 2. Compute physics
    physics = compute_layer1_physics(df)

    # 3. Analyze regime statistics
    regime_stats = analyze_regime_statistics(df, physics)

    # 4. Validate universal truths
    validate_universal_truths(regime_stats)

    # 5. Run baseline backtest
    result = run_baseline_backtest(df, physics)

    # 6. Analyze trade lifecycle (sample)
    if result.total_trades > 0:
        analyze_trade_lifecycle(result, n_trades=2)

    # 7. Test RL environment
    test_rl_environment(df)

    print("\n" + "=" * 60)
    print("VALIDATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
