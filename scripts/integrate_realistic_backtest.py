#!/usr/bin/env python3
"""
Integration Guide: Realistic Backtesting in Exploration Framework

Shows how to use RealisticBacktester to get accurate, repeatable
exploration results that transfer to live trading.

Key idea: After training, validate with realistic backtest BEFORE deploying.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np

from kinetra.realistic_backtester import RealisticBacktester, BacktestResult
from kinetra.market_microstructure import SymbolSpec, AssetClass
from kinetra.data_loader import UnifiedDataLoader


def validate_exploration_run(
    agent,
    data: pd.DataFrame,
    spec: SymbolSpec,
    verbose: bool = True
) -> BacktestResult:
    """
    Validate trained agent with realistic backtesting.

    This is the CRITICAL step that prevents sim-to-real gap:
    - Training uses simple environment (fast iteration)
    - Validation uses realistic backtest (catches issues before live)

    Args:
        agent: Trained RL agent
        data: OHLCV data with 'spread' column
        spec: SymbolSpec with freeze zones and stops levels
        verbose: Print warnings for constraint violations

    Returns:
        BacktestResult with regime breakdown

    Usage:
        # After training
        result = validate_exploration_run(agent, data, spec)

        # Check for overfitting
        if result.regime_performance['physics_overdamped']['sharpe'] < 0:
            print("Agent loses in chop → Deploy with regime filter!")

        # Check for constraint violations
        if result.total_invalid_stops > 0:
            print(f"Agent tried {result.total_invalid_stops} invalid stops!")
            print("→ Adjust agent SL logic or increase safe_stop_distance()")
    """
    # Generate signals from agent
    signals = generate_signals_from_agent(agent, data)

    # Run realistic backtest
    backtester = RealisticBacktester(
        spec=spec,
        enable_slippage=True,
        enable_freeze_zones=True,
        enable_stop_validation=True,
        verbose=verbose
    )

    result = backtester.run(
        data=data,
        signals=signals,
        classify_regimes=True
    )

    # Print summary
    if verbose:
        print_validation_summary(result, spec)

    return result


def generate_signals_from_agent(agent, data: pd.DataFrame) -> pd.DataFrame:
    """
    Generate trading signals from trained agent.

    Agent decides: when to enter, direction, SL, TP.
    Returns signals DataFrame for backtest.
    """
    signals = []

    # Reset agent
    # state = agent.reset(data)

    for idx in range(len(data)):
        # Get current state
        # action = agent.select_action(state)

        # Placeholder: Random signals for demonstration
        if np.random.rand() > 0.95:  # 5% entry rate
            signals.append({
                'time': data.index[idx],
                'action': 'open_long' if np.random.rand() > 0.5 else 'open_short',
                'price': data.iloc[idx]['close'],
                'sl': data.iloc[idx]['close'] * 0.98,  # 2% SL
                'tp': data.iloc[idx]['close'] * 1.04,  # 4% TP
                'volume': 1.0,
            })

        # Random close
        if len(signals) > 0 and signals[-1]['action'].startswith('open'):
            if np.random.rand() > 0.9:
                signals.append({
                    'time': data.index[idx],
                    'action': 'close',
                    'price': data.iloc[idx]['close'],
                })

    return pd.DataFrame(signals)


def print_validation_summary(result: BacktestResult, spec: SymbolSpec):
    """Print validation summary with regime breakdown."""
    print("\n" + "="*70)
    print("REALISTIC BACKTEST VALIDATION")
    print("="*70)

    # Overall metrics
    print(f"\nOverall Performance:")
    print(f"  Total Trades: {result.total_trades}")
    print(f"  Win Rate: {result.win_rate*100:.1f}%")
    print(f"  Total P&L: ${result.total_pnl:,.0f}")
    print(f"  Return: {result.total_return_pct*100:.1f}%")
    print(f"  Sharpe Ratio: {result.sharpe_ratio:.3f}")
    print(f"  Max Drawdown: {result.max_drawdown*100:.1f}%")

    # Realistic costs
    print(f"\nRealistic Costs:")
    print(f"  Spread Cost: ${result.total_spread_cost:,.0f}")
    print(f"  Commission: ${result.total_commission:,.0f}")
    print(f"  Swap: ${result.total_swap:,.0f}")
    print(f"  Slippage: ${result.total_slippage:,.0f}")

    total_cost = result.total_spread_cost + result.total_commission + abs(result.total_swap) + result.total_slippage
    print(f"  TOTAL: ${total_cost:,.0f} ({total_cost/abs(result.total_pnl)*100:.1f}% of gross P&L)")

    # Quality metrics
    print(f"\nQuality Metrics:")
    print(f"  Avg MFE: ${result.avg_mfe:.2f}")
    print(f"  Avg MAE: ${result.avg_mae:.2f}")
    print(f"  MFE/MAE Ratio: {result.avg_mfe_mae_ratio:.2f}")

    # Constraint violations (CRITICAL)
    print(f"\nConstraint Violations:")
    print(f"  Freeze Zone Violations: {result.total_freeze_violations}")
    print(f"  Invalid Stops: {result.total_invalid_stops}")
    print(f"  Rejected Orders: {result.total_rejected_orders}")

    if result.total_freeze_violations + result.total_invalid_stops > 0:
        print(f"\n  ⚠️  WARNING: Agent violates broker constraints!")
        print(f"  → Fix agent logic OR adjust spec:")
        print(f"     - Use spec.get_safe_stop_distance() for SL placement")
        print(f"     - Check spec.is_in_freeze_zone() before modifications")

    # Regime breakdown (detect overfitting)
    print(f"\nRegime Performance Breakdown:")
    print(f"  (Negative Sharpe = LOSING in that regime)")

    for regime_key, metrics in sorted(result.regime_performance.items()):
        regime_type, regime_name = regime_key.split('_', 1)
        sharpe = metrics['sharpe']
        status = "✓" if sharpe > 0 else "✗"

        print(f"  {status} {regime_key:30s}: Sharpe={sharpe:+.3f}, Trades={metrics['trades']}, "
              f"WinRate={metrics['win_rate']*100:.1f}%")

    # Recommendations
    print(f"\n" + "="*70)
    print("RECOMMENDATIONS")
    print("="*70)

    # Check for regime-specific losses
    losing_regimes = [k for k, v in result.regime_performance.items() if v['sharpe'] < 0]

    if losing_regimes:
        print(f"\n⚠️  Agent LOSES money in these regimes:")
        for regime in losing_regimes:
            print(f"  - {regime}")
        print(f"\n→ SOLUTION: Deploy with RegimeFilter to avoid these conditions!")
        print(f"   Example: RegimeFilter(physics_regimes={{PhysicsRegime.LAMINAR}})")
    else:
        print(f"\n✓ Agent profitable across all regimes (robust!)")

    # Check cost ratio
    cost_ratio = total_cost / abs(result.total_pnl) if result.total_pnl != 0 else 0
    if cost_ratio > 0.5:
        print(f"\n⚠️  Costs are {cost_ratio*100:.0f}% of gross P&L (too high!)")
        print(f"→ Agent is overtrading or holding too short")
        print(f"→ Increase min trade duration or reduce trade frequency")
    else:
        print(f"\n✓ Costs are {cost_ratio*100:.0f}% of gross P&L (acceptable)")


def compare_simple_vs_realistic(agent, data: pd.DataFrame, spec: SymbolSpec):
    """
    Compare simple backtest (what agents see during training)
    vs realistic backtest (what happens in live trading).

    This shows the sim-to-real gap.
    """
    print("\n" + "="*70)
    print("SIM-TO-REAL GAP ANALYSIS")
    print("="*70)

    # Simple backtest (no costs, no constraints)
    simple_result = run_simple_backtest(agent, data)

    # Realistic backtest (full MT5 constraints)
    realistic_result = validate_exploration_run(agent, data, spec, verbose=False)

    # Compare
    print(f"\n{'Metric':<30s} {'Simple':>15s} {'Realistic':>15s} {'Gap':>15s}")
    print("-"*70)

    metrics = [
        ('Sharpe Ratio', simple_result['sharpe'], realistic_result.sharpe_ratio),
        ('Total P&L', simple_result['pnl'], realistic_result.total_pnl),
        ('Win Rate', simple_result['win_rate'], realistic_result.win_rate),
        ('Max Drawdown', simple_result['max_dd'], realistic_result.max_drawdown),
    ]

    for metric_name, simple_val, realistic_val in metrics:
        gap = realistic_val - simple_val
        gap_pct = (gap / simple_val * 100) if simple_val != 0 else 0

        print(f"{metric_name:<30s} {simple_val:>15.3f} {realistic_val:>15.3f} {gap_pct:>14.1f}%")

    # Interpretation
    print(f"\n" + "="*70)
    print("INTERPRETATION")
    print("="*70)

    sharpe_gap = realistic_result.sharpe_ratio - simple_result['sharpe']

    if sharpe_gap < -0.3:
        print(f"\n⚠️  LARGE SIM-TO-REAL GAP: Sharpe drops {abs(sharpe_gap):.2f}")
        print(f"   Simple backtest: {simple_result['sharpe']:.3f}")
        print(f"   Realistic backtest: {realistic_result.sharpe_ratio:.3f}")
        print(f"\n   → Agent learned on unrealistic conditions!")
        print(f"   → Re-train with realistic constraints from day 1")
    else:
        print(f"\n✓ Small sim-to-real gap ({abs(sharpe_gap):.2f})")
        print(f"   Agent should transfer well to live trading!")


def run_simple_backtest(agent, data: pd.DataFrame) -> dict:
    """
    Simple backtest without constraints (what agents see during training).

    Returns dict with basic metrics for comparison.
    """
    # Placeholder: Random results for demonstration
    return {
        'sharpe': 1.5,
        'pnl': 5000.0,
        'win_rate': 0.55,
        'max_dd': 0.15,
    }


def main():
    """Demonstrate realistic backtest integration."""
    print("Realistic Backtest Integration Example")
    print("="*70)

    # Load data
    print("\n1. Loading data...")
    loader = UnifiedDataLoader(verbose=False)

    # For demo, create synthetic data
    data = pd.DataFrame({
        'open': np.random.randn(1000).cumsum() + 100,
        'high': np.random.randn(1000).cumsum() + 101,
        'low': np.random.randn(1000).cumsum() + 99,
        'close': np.random.randn(1000).cumsum() + 100,
        'volume': np.random.randint(1000, 10000, 1000),
        'spread': np.random.randint(10, 30, 1000),  # Dynamic spread!
    }, index=pd.date_range('2024-01-01', periods=1000, freq='H'))

    # Create spec with realistic constraints
    spec = SymbolSpec(
        symbol="EURUSD",
        asset_class=AssetClass.FOREX,
        digits=5,
        trade_stops_level=15,  # 15 points minimum SL distance
        trade_freeze_level=10,  # 10 points freeze zone
        spread_typical=15.0,
        swap_long=-12.16,
        swap_short=4.37,
    )

    print(f"   Symbol: {spec.symbol}")
    print(f"   Stops Level: {spec.trade_stops_level} points")
    print(f"   Freeze Level: {spec.trade_freeze_level} points")

    # Placeholder agent (in production, use trained agent)
    print("\n2. Creating agent...")
    agent = None  # Replace with trained agent

    # Run realistic validation
    print("\n3. Running realistic backtest validation...")
    result = validate_exploration_run(agent, data, spec, verbose=True)

    # Compare simple vs realistic
    print("\n4. Comparing simple vs realistic backtest...")
    compare_simple_vs_realistic(agent, data, spec)

    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print("\nRealistic backtesting reveals:")
    print("1. Regime-specific performance (detect overfitting)")
    print("2. Realistic costs (spread, commission, swap, slippage)")
    print("3. Constraint violations (freeze zones, invalid stops)")
    print("4. Sim-to-real gap (simple vs realistic backtest)")
    print("\n→ Use this BEFORE deploying to live trading!")


if __name__ == "__main__":
    main()
