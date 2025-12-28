#!/usr/bin/env python3
"""
Physics-Based Backtesting Runner

Empirically tests thermodynamics/physics/kinematics trading strategies
using REAL MT5 market data.

Usage:
    python scripts/run_physics_backtest.py --data path/to/mt5_data.csv
    python scripts/run_physics_backtest.py --data path/to/data.csv --strategy multi_physics
    python scripts/run_physics_backtest.py --data path/to/data.csv --compare-all
    python scripts/run_physics_backtest.py --data path/to/data.csv --walk-forward

Examples:
    # Run single strategy
    python scripts/run_physics_backtest.py --data data/EURUSD_H1.csv --strategy energy_momentum

    # Compare all physics strategies
    python scripts/run_physics_backtest.py --data data/BTCUSD_H1.csv --compare-all

    # Walk-forward validation
    python scripts/run_physics_backtest.py --data data/XAUUSD_H1.csv --walk-forward --train-bars 1000

    # Optimize parameters
    python scripts/run_physics_backtest.py --data data/EURUSD_H1.csv --optimize --strategy multi_physics
"""

import argparse
import sys
import json
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np

from kinetra.physics_backtester import (
    PhysicsBacktestRunner,
    list_strategies,
    calculate_physics_metrics,
    STRATEGY_REGISTRY
)
from kinetra.data_utils import (
    load_mt5_csv,
    load_mt5_history,
    preprocess_mt5_data,
    get_data_summary,
    validate_ohlcv,
    split_data,
    create_walk_forward_windows
)
from kinetra.physics_engine import PhysicsEngine


def print_header(title: str):
    """Print formatted header."""
    print("\n" + "=" * 70)
    print(f" {title}")
    print("=" * 70)


def print_section(title: str):
    """Print section header."""
    print(f"\n--- {title} ---")


def format_pct(value: float) -> str:
    """Format percentage value."""
    return f"{value:+.2f}%" if not np.isnan(value) else "N/A"


def format_ratio(value: float) -> str:
    """Format ratio value."""
    return f"{value:.3f}" if not np.isnan(value) else "N/A"


def run_single_backtest(args):
    """Run a single strategy backtest."""
    print_header(f"PHYSICS BACKTEST: {args.strategy.upper()}")

    # Load data
    print_section("Loading MT5 Data")
    df = load_mt5_csv(args.data)

    if args.start_date or args.end_date:
        df = load_mt5_history(args.data, args.start_date, args.end_date)

    df = preprocess_mt5_data(df, remove_weekends=not args.include_weekends)

    # Print data summary
    summary = get_data_summary(df)
    print(f"Data file:     {args.data}")
    print(f"Date range:    {summary['start_date']} to {summary['end_date']}")
    print(f"Total bars:    {summary['total_bars']:,}")
    print(f"Price range:   {summary['price_low']:.5f} - {summary['price_high']:.5f}")
    print(f"Buy & Hold:    {format_pct(summary['total_return_pct'])}")

    # Run backtest
    print_section("Running Backtest")
    runner = PhysicsBacktestRunner(
        cash=args.cash,
        commission=args.commission,
        margin=args.margin
    )

    results = runner.run(df, strategy=args.strategy)

    # Print results
    print_section("RESULTS")
    print(f"Strategy:           {results['strategy']}")
    print(f"Return:             {format_pct(results['return_pct'])}")
    print(f"Buy & Hold Return:  {format_pct(results['buy_hold_return_pct'])}")
    print(f"Sharpe Ratio:       {format_ratio(results['sharpe_ratio'])}")
    print(f"Sortino Ratio:      {format_ratio(results['sortino_ratio'])}")
    print(f"Max Drawdown:       {format_pct(results['max_drawdown_pct'])}")
    print(f"Win Rate:           {format_pct(results['win_rate'])}")
    print(f"Profit Factor:      {format_ratio(results['profit_factor'])}")
    print(f"Total Trades:       {results['total_trades']}")
    print(f"Avg Trade:          {format_pct(results['avg_trade_pct'])}")
    print(f"Exposure:           {format_pct(results['exposure_pct'])}")
    print(f"Final Equity:       ${results['final_equity']:,.2f}")
    print(f"SQN:                {format_ratio(results['sqn'])}")

    # Physics metrics
    print_section("PHYSICS METRICS")
    physics_metrics = calculate_physics_metrics(df, results)
    print(f"Total Energy:       {physics_metrics['total_energy']:.2f}")
    print(f"Energy Captured:    {physics_metrics['energy_captured_pct']:.1f}%")
    print(f"Avg Entropy:        {physics_metrics['avg_entropy']:.4f}")
    print(f"Avg Damping:        {physics_metrics['avg_damping']:.4f}")
    print(f"Regime Distribution:")
    print(f"  - Underdamped:    {physics_metrics['regime_underdamped_pct']:.1f}%")
    print(f"  - Critical:       {physics_metrics['regime_critical_pct']:.1f}%")
    print(f"  - Overdamped:     {physics_metrics['regime_overdamped_pct']:.1f}%")

    # Save results if requested
    if args.output:
        output_data = {
            'timestamp': datetime.now().isoformat(),
            'data_file': args.data,
            'data_summary': summary,
            'results': results,
            'physics_metrics': physics_metrics
        }
        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"\nResults saved to: {args.output}")

    return results


def run_strategy_comparison(args):
    """Compare all physics strategies."""
    print_header("PHYSICS STRATEGY COMPARISON")

    # Load data
    print_section("Loading MT5 Data")
    df = load_mt5_csv(args.data)
    df = preprocess_mt5_data(df, remove_weekends=not args.include_weekends)

    summary = get_data_summary(df)
    print(f"Data file:     {args.data}")
    print(f"Date range:    {summary['start_date']} to {summary['end_date']}")
    print(f"Total bars:    {summary['total_bars']:,}")
    print(f"Buy & Hold:    {format_pct(summary['total_return_pct'])}")

    # Run comparison
    print_section("Running Strategy Comparison")
    runner = PhysicsBacktestRunner(
        cash=args.cash,
        commission=args.commission,
        margin=args.margin
    )

    comparison = runner.compare_strategies(df)

    # Sort by return
    comparison = comparison.sort_values('return_pct', ascending=False)

    # Print comparison table
    print_section("STRATEGY RANKING BY RETURN")
    print(f"{'Strategy':<25} {'Return':>10} {'Sharpe':>8} {'MaxDD':>10} {'WinRate':>8} {'Trades':>7}")
    print("-" * 70)

    for _, row in comparison.iterrows():
        print(f"{row['strategy']:<25} "
              f"{format_pct(row['return_pct']):>10} "
              f"{format_ratio(row['sharpe_ratio']):>8} "
              f"{format_pct(row['max_drawdown_pct']):>10} "
              f"{format_pct(row['win_rate']):>8} "
              f"{int(row['total_trades']):>7}")

    # Best strategy summary
    print_section("BEST PERFORMERS")
    best_return = comparison.iloc[0]
    best_sharpe = comparison.loc[comparison['sharpe_ratio'].idxmax()]
    best_winrate = comparison.loc[comparison['win_rate'].idxmax()]

    print(f"Best Return:     {best_return['strategy']} ({format_pct(best_return['return_pct'])})")
    print(f"Best Sharpe:     {best_sharpe['strategy']} ({format_ratio(best_sharpe['sharpe_ratio'])})")
    print(f"Best Win Rate:   {best_winrate['strategy']} ({format_pct(best_winrate['win_rate'])})")

    # Save results
    if args.output:
        comparison.to_csv(args.output, index=False)
        print(f"\nResults saved to: {args.output}")

    return comparison


def run_walk_forward(args):
    """Run walk-forward validation."""
    print_header("WALK-FORWARD VALIDATION")

    # Load data
    print_section("Loading MT5 Data")
    df = load_mt5_csv(args.data)
    df = preprocess_mt5_data(df, remove_weekends=not args.include_weekends)

    summary = get_data_summary(df)
    print(f"Data file:     {args.data}")
    print(f"Date range:    {summary['start_date']} to {summary['end_date']}")
    print(f"Total bars:    {summary['total_bars']:,}")

    # Create walk-forward windows
    windows = create_walk_forward_windows(
        df,
        train_bars=args.train_bars,
        test_bars=args.test_bars,
        step_bars=args.step_bars
    )

    print(f"\nWalk-forward windows: {len(windows)}")
    print(f"Train bars: {args.train_bars}, Test bars: {args.test_bars}, Step: {args.step_bars}")

    # Run walk-forward
    print_section(f"Running Walk-Forward: {args.strategy.upper()}")

    runner = PhysicsBacktestRunner(
        cash=args.cash,
        commission=args.commission,
        margin=args.margin
    )

    results = []
    for i, (train_df, test_df) in enumerate(windows):
        # Optimize on train
        if args.optimize_each_window:
            # Simple optimization - could be extended
            pass

        # Test on out-of-sample
        try:
            result = runner.run(test_df, strategy=args.strategy)
            result['window'] = i
            result['train_start'] = str(train_df.index[0])
            result['train_end'] = str(train_df.index[-1])
            result['test_start'] = str(test_df.index[0])
            result['test_end'] = str(test_df.index[-1])
            results.append(result)
            print(f"Window {i+1:3d}/{len(windows)}: Return = {format_pct(result['return_pct'])}")
        except Exception as e:
            print(f"Window {i+1:3d}/{len(windows)}: FAILED - {e}")

    results_df = pd.DataFrame(results)

    # Summary statistics
    print_section("WALK-FORWARD SUMMARY")
    print(f"Windows tested:     {len(results_df)}")
    print(f"Mean Return:        {format_pct(results_df['return_pct'].mean())}")
    print(f"Median Return:      {format_pct(results_df['return_pct'].median())}")
    print(f"Std Return:         {format_pct(results_df['return_pct'].std())}")
    print(f"Win Rate (windows): {format_pct((results_df['return_pct'] > 0).mean() * 100)}")
    print(f"Best Window:        {format_pct(results_df['return_pct'].max())}")
    print(f"Worst Window:       {format_pct(results_df['return_pct'].min())}")

    # Cumulative metrics
    total_return = ((1 + results_df['return_pct']/100).prod() - 1) * 100
    print(f"\nCumulative Return:  {format_pct(total_return)}")

    # Save results
    if args.output:
        results_df.to_csv(args.output, index=False)
        print(f"\nResults saved to: {args.output}")

    return results_df


def run_optimization(args):
    """Run parameter optimization."""
    print_header(f"PARAMETER OPTIMIZATION: {args.strategy.upper()}")

    # Load data
    print_section("Loading MT5 Data")
    df = load_mt5_csv(args.data)
    df = preprocess_mt5_data(df, remove_weekends=not args.include_weekends)

    summary = get_data_summary(df)
    print(f"Data file:     {args.data}")
    print(f"Date range:    {summary['start_date']} to {summary['end_date']}")
    print(f"Total bars:    {summary['total_bars']:,}")

    # Define parameter ranges based on strategy
    param_ranges = {
        'lookback': range(10, 50, 5),
        'energy_threshold_pct': range(60, 90, 5),
    }

    print_section("Optimizing Parameters")
    print(f"Parameter ranges: {param_ranges}")

    runner = PhysicsBacktestRunner(
        cash=args.cash,
        commission=args.commission,
        margin=args.margin
    )

    try:
        best_params, stats = runner.optimize(
            df,
            strategy=args.strategy,
            param_ranges=param_ranges,
            maximize=args.optimize_for
        )

        print_section("OPTIMIZATION RESULTS")
        print(f"Optimized for: {args.optimize_for}")
        print(f"Best parameters: {best_params}")
        print(f"Best {args.optimize_for}: {stats[args.optimize_for]}")

    except Exception as e:
        print(f"Optimization failed: {e}")
        return None

    return best_params


def analyze_physics_state(args):
    """Analyze physics state of the data."""
    print_header("PHYSICS STATE ANALYSIS")

    # Load data
    df = load_mt5_csv(args.data)
    df = preprocess_mt5_data(df, remove_weekends=not args.include_weekends)

    summary = get_data_summary(df)
    print(f"Data file:     {args.data}")
    print(f"Date range:    {summary['start_date']} to {summary['end_date']}")
    print(f"Total bars:    {summary['total_bars']:,}")

    # Compute physics state
    print_section("Computing Physics State")
    engine = PhysicsEngine(lookback=20)
    state = engine.compute_physics_state(df['Close'], df.get('Volume'))

    # Statistics
    print_section("PHYSICS STATISTICS")
    print(f"\nKinetic Energy:")
    print(f"  Mean:     {state['energy'].mean():.6f}")
    print(f"  Std:      {state['energy'].std():.6f}")
    print(f"  Max:      {state['energy'].max():.6f}")
    print(f"  75th pct: {state['energy'].quantile(0.75):.6f}")

    print(f"\nDamping Coefficient:")
    print(f"  Mean:     {state['damping'].mean():.4f}")
    print(f"  Std:      {state['damping'].std():.4f}")
    print(f"  25th pct: {state['damping'].quantile(0.25):.4f}")
    print(f"  75th pct: {state['damping'].quantile(0.75):.4f}")

    print(f"\nEntropy:")
    print(f"  Mean:     {state['entropy'].mean():.4f}")
    print(f"  Std:      {state['entropy'].std():.4f}")
    print(f"  Max:      {state['entropy'].max():.4f}")

    # Regime distribution
    print_section("REGIME DISTRIBUTION")
    regime_counts = state['regime'].value_counts()
    for regime, count in regime_counts.items():
        pct = count / len(state) * 100
        print(f"  {regime:<15}: {count:>6} bars ({pct:.1f}%)")

    # Regime transitions
    regime_changes = (state['regime'] != state['regime'].shift()).sum()
    avg_regime_duration = len(state) / regime_changes if regime_changes > 0 else len(state)
    print(f"\nRegime changes:     {regime_changes}")
    print(f"Avg regime duration: {avg_regime_duration:.1f} bars")

    return state


def main():
    parser = argparse.ArgumentParser(
        description='Physics-Based Backtesting for Kinetra',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # Required arguments
    parser.add_argument('--data', '-d', required=True,
                        help='Path to MT5 CSV data file')

    # Mode selection
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument('--compare-all', action='store_true',
                            help='Compare all physics strategies')
    mode_group.add_argument('--walk-forward', action='store_true',
                            help='Run walk-forward validation')
    mode_group.add_argument('--optimize', action='store_true',
                            help='Optimize strategy parameters')
    mode_group.add_argument('--analyze', action='store_true',
                            help='Analyze physics state only (no trading)')

    # Strategy selection
    parser.add_argument('--strategy', '-s', default='multi_physics',
                        choices=list_strategies(),
                        help='Trading strategy (default: multi_physics)')

    # Data options
    parser.add_argument('--start-date', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', help='End date (YYYY-MM-DD)')
    parser.add_argument('--include-weekends', action='store_true',
                        help='Include weekend data')

    # Backtest parameters
    parser.add_argument('--cash', type=float, default=100000,
                        help='Initial capital (default: 100000)')
    parser.add_argument('--commission', type=float, default=0.001,
                        help='Commission per trade (default: 0.001)')
    parser.add_argument('--margin', type=float, default=1.0,
                        help='Margin requirement (default: 1.0)')

    # Walk-forward parameters
    parser.add_argument('--train-bars', type=int, default=500,
                        help='Training window size (default: 500)')
    parser.add_argument('--test-bars', type=int, default=100,
                        help='Test window size (default: 100)')
    parser.add_argument('--step-bars', type=int, default=50,
                        help='Step size between windows (default: 50)')
    parser.add_argument('--optimize-each-window', action='store_true',
                        help='Optimize parameters for each window')

    # Optimization parameters
    parser.add_argument('--optimize-for', default='Return [%]',
                        help='Metric to optimize (default: Return [%%])')

    # Output
    parser.add_argument('--output', '-o', help='Output file for results')

    args = parser.parse_args()

    # Validate data file exists
    if not Path(args.data).exists():
        print(f"Error: Data file not found: {args.data}")
        sys.exit(1)

    # Run appropriate mode
    try:
        if args.compare_all:
            run_strategy_comparison(args)
        elif args.walk_forward:
            run_walk_forward(args)
        elif args.optimize:
            run_optimization(args)
        elif args.analyze:
            analyze_physics_state(args)
        else:
            run_single_backtest(args)

        print("\n" + "=" * 70)
        print(" BACKTEST COMPLETE")
        print("=" * 70 + "\n")

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
