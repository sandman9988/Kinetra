#!/usr/bin/env python3
"""
Comprehensive Physics Backtest Runner with Detailed Logging

Runs all physics strategies on each CSV file with extensive per-trade
and per-run statistics for reward shaping and composite health scores.

Output:
- Per-trade details (JSON + CSV) for RL reward shaping
- Per-run metrics for composite health score
- Physics state analysis per trade

Usage:
    python scripts/run_comprehensive_backtest.py /path/to/data/*.csv
"""

import sys
import os
import json
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from backtesting import Backtest

# Import Kinetra modules
from kinetra.strategies_v7 import (
    BerserkerStrategy,
    SniperStrategy,
    MultiAgentV7Strategy,
)
from kinetra.physics_v7 import (
    PhysicsEngineV7,
    calculate_omega_ratio,
    compute_fractal_dimension_katz,
    compute_sample_entropy_fast,
    compute_vpin_proxy,
    compute_symc_ratio,
    compute_ftle_fast,
    OnlineMetricsTracker,
)
from kinetra.data_utils import load_mt5_csv, get_data_summary


# V7 Strategies only (for RL training)
V7_STRATEGIES = {
    'berserker': BerserkerStrategy,
    'sniper': SniperStrategy,
    'multi_agent_v7': MultiAgentV7Strategy,
}


def compute_physics_features(df: pd.DataFrame, window: int = 50) -> pd.DataFrame:
    """Compute all physics features for the dataset."""
    close = df['Close'].values
    high = df['High'].values
    low = df['Low'].values
    volume = df['Volume'].values
    returns = df['Close'].pct_change().fillna(0).values

    features = pd.DataFrame(index=df.index)

    # Fractal Dimension
    features['fractal_dim'] = compute_fractal_dimension_katz(close, window)

    # Sample Entropy
    features['sample_entropy'] = compute_sample_entropy_fast(close, window)

    # VPIN (toxicity)
    features['vpin'] = compute_vpin_proxy(close, volume, window)

    # SymC Ratio (damping)
    features['symc'] = compute_symc_ratio(high, low, close, volume, window // 2)

    # FTLE (chaos)
    features['ftle'] = compute_ftle_fast(close, window)

    # Volatility
    features['volatility'] = pd.Series(returns).rolling(window).std().values

    # Return momentum
    features['momentum'] = pd.Series(close).pct_change(window).values

    return features


def extract_trade_details(stats, df: pd.DataFrame, physics_features: pd.DataFrame) -> list:
    """Extract detailed per-trade information including physics state."""
    trades = []

    if not hasattr(stats, '_trades') or stats._trades is None or len(stats._trades) == 0:
        return trades

    trade_df = stats._trades

    for idx, trade in trade_df.iterrows():
        entry_time = trade['EntryTime']
        exit_time = trade['ExitTime']

        # Find closest index for physics features
        try:
            entry_idx = physics_features.index.get_indexer([entry_time], method='nearest')[0]
            exit_idx = physics_features.index.get_indexer([exit_time], method='nearest')[0]
        except:
            entry_idx = 0
            exit_idx = len(physics_features) - 1

        # Physics state at entry
        entry_physics = {}
        exit_physics = {}

        for col in physics_features.columns:
            try:
                entry_physics[f'entry_{col}'] = float(physics_features.iloc[entry_idx][col])
                exit_physics[f'exit_{col}'] = float(physics_features.iloc[exit_idx][col])
            except:
                entry_physics[f'entry_{col}'] = 0.0
                exit_physics[f'exit_{col}'] = 0.0

        # Calculate trade metrics
        entry_price = float(trade['EntryPrice'])
        exit_price = float(trade['ExitPrice'])
        pnl = float(trade['PnL'])
        return_pct = float(trade['ReturnPct'])

        # Duration in bars
        try:
            duration_bars = int(trade['Duration'].total_seconds() / 3600)  # Assuming hourly
        except:
            duration_bars = 1

        # Direction
        is_long = entry_price < exit_price if pnl > 0 else entry_price > exit_price

        trade_info = {
            'trade_id': int(idx),
            'entry_time': str(entry_time),
            'exit_time': str(exit_time),
            'entry_price': entry_price,
            'exit_price': exit_price,
            'pnl': pnl,
            'return_pct': return_pct,
            'duration_bars': duration_bars,
            'is_long': is_long,
            'size': float(trade['Size']),
            **entry_physics,
            **exit_physics,
        }

        # Add price excursions if available
        try:
            trade_info['max_favorable_excursion'] = float(trade.get('MFE', 0))
            trade_info['max_adverse_excursion'] = float(trade.get('MAE', 0))
        except:
            pass

        trades.append(trade_info)

    return trades


def compute_run_metrics(stats, trades: list) -> dict:
    """Compute comprehensive run metrics for reward shaping."""
    metrics = {}

    # Basic metrics
    metrics['total_return_pct'] = float(stats['Return [%]'])
    metrics['final_equity'] = float(stats['Equity Final [$]'])
    metrics['max_drawdown_pct'] = float(stats['Max. Drawdown [%]'])
    metrics['num_trades'] = int(stats['# Trades'])
    metrics['win_rate'] = float(stats['Win Rate [%]']) if not np.isnan(stats['Win Rate [%]']) else 0.0
    metrics['profit_factor'] = float(stats['Profit Factor']) if not np.isnan(stats['Profit Factor']) else 0.0
    metrics['sharpe_ratio'] = float(stats['Sharpe Ratio']) if not np.isnan(stats['Sharpe Ratio']) else 0.0
    metrics['exposure_time_pct'] = float(stats['Exposure Time [%]'])
    metrics['buy_hold_return_pct'] = float(stats['Buy & Hold Return [%]'])

    # Calculate Omega ratio from trades
    if trades:
        returns = [t['return_pct'] / 100 for t in trades]
        returns_series = pd.Series(returns)
        metrics['omega_ratio'] = calculate_omega_ratio(returns_series, threshold=0.0)

        # Sortino ratio (downside deviation)
        downside_returns = returns_series[returns_series < 0]
        if len(downside_returns) > 0:
            downside_std = downside_returns.std()
            if downside_std > 0:
                metrics['sortino_ratio'] = returns_series.mean() / downside_std * np.sqrt(252)
            else:
                metrics['sortino_ratio'] = 0.0
        else:
            metrics['sortino_ratio'] = float('inf') if returns_series.mean() > 0 else 0.0

        # Calmar ratio (return / max drawdown)
        if metrics['max_drawdown_pct'] != 0:
            metrics['calmar_ratio'] = metrics['total_return_pct'] / abs(metrics['max_drawdown_pct'])
        else:
            metrics['calmar_ratio'] = 0.0

        # Tail ratios
        if len(returns) > 10:
            sorted_returns = sorted(returns)
            n = len(sorted_returns)
            bottom_10 = sorted_returns[:max(1, n//10)]
            top_10 = sorted_returns[-max(1, n//10):]

            metrics['avg_win'] = float(np.mean([r for r in returns if r > 0])) if any(r > 0 for r in returns) else 0.0
            metrics['avg_loss'] = float(np.mean([r for r in returns if r < 0])) if any(r < 0 for r in returns) else 0.0
            metrics['win_loss_ratio'] = abs(metrics['avg_win'] / metrics['avg_loss']) if metrics['avg_loss'] != 0 else 0.0
            metrics['tail_ratio'] = abs(np.mean(top_10) / np.mean(bottom_10)) if np.mean(bottom_10) != 0 else 0.0

            # Skewness and kurtosis
            metrics['return_skewness'] = float(returns_series.skew())
            metrics['return_kurtosis'] = float(returns_series.kurtosis())

        # Duration stats
        durations = [t['duration_bars'] for t in trades]
        metrics['avg_duration_bars'] = float(np.mean(durations))
        metrics['max_duration_bars'] = int(max(durations))
        metrics['min_duration_bars'] = int(min(durations))

        # Win/loss streaks
        win_streak = 0
        loss_streak = 0
        max_win_streak = 0
        max_loss_streak = 0

        for t in trades:
            if t['pnl'] > 0:
                win_streak += 1
                loss_streak = 0
                max_win_streak = max(max_win_streak, win_streak)
            else:
                loss_streak += 1
                win_streak = 0
                max_loss_streak = max(max_loss_streak, loss_streak)

        metrics['max_win_streak'] = max_win_streak
        metrics['max_loss_streak'] = max_loss_streak

        # Physics feature averages at entry
        physics_cols = [k for k in trades[0].keys() if k.startswith('entry_')]
        for col in physics_cols:
            values = [t.get(col, 0) for t in trades]
            metrics[f'avg_{col}'] = float(np.mean(values))
            metrics[f'std_{col}'] = float(np.std(values))

        # Winning vs losing trade physics comparison
        winning_trades = [t for t in trades if t['pnl'] > 0]
        losing_trades = [t for t in trades if t['pnl'] <= 0]

        for col in physics_cols:
            if winning_trades:
                metrics[f'win_avg_{col}'] = float(np.mean([t.get(col, 0) for t in winning_trades]))
            if losing_trades:
                metrics[f'loss_avg_{col}'] = float(np.mean([t.get(col, 0) for t in losing_trades]))

    return metrics


def compute_ulcer_index(equity_curve: pd.Series) -> float:
    """Calculate Ulcer Index - measures downside volatility."""
    peak = equity_curve.expanding().max()
    drawdown_pct = (equity_curve - peak) / peak * 100
    ulcer_index = np.sqrt(np.mean(drawdown_pct ** 2))
    return float(ulcer_index)


def run_backtest_with_logging(
    df: pd.DataFrame,
    strategy_class,
    physics_features: pd.DataFrame,
    cash: float = 100000
) -> dict:
    """Run backtest with comprehensive logging."""
    result = {
        'success': False,
        'strategy': strategy_class.__name__,
        'trades': [],
        'metrics': {},
        'error': None
    }

    try:
        bt = Backtest(
            df,
            strategy_class,
            cash=cash,
            commission=0.001,
            trade_on_close=True,
            exclusive_orders=True
        )
        stats = bt.run()

        # Extract trade details
        trades = extract_trade_details(stats, df, physics_features)

        # Compute run metrics
        metrics = compute_run_metrics(stats, trades)

        # Add Ulcer Index from equity curve
        try:
            equity = stats._equity_curve['Equity']
            metrics['ulcer_index'] = compute_ulcer_index(equity)

            # UPI (Ulcer Performance Index) = Return / Ulcer Index
            if metrics['ulcer_index'] > 0:
                metrics['ulcer_performance_index'] = metrics['total_return_pct'] / metrics['ulcer_index']
            else:
                metrics['ulcer_performance_index'] = 0.0
        except:
            metrics['ulcer_index'] = 0.0
            metrics['ulcer_performance_index'] = 0.0

        result['success'] = True
        result['trades'] = trades
        result['metrics'] = metrics

    except Exception as e:
        result['error'] = str(e)
        import traceback
        result['traceback'] = traceback.format_exc()

    return result


def analyze_dataset_physics(df: pd.DataFrame) -> dict:
    """Compute physics analysis for the entire dataset."""
    engine = PhysicsEngineV7()
    state = engine.compute_physics_state(df)

    analysis = {
        'bars': len(df),
        'start_date': str(df.index[0]),
        'end_date': str(df.index[-1]),
        'price_start': float(df['Close'].iloc[0]),
        'price_end': float(df['Close'].iloc[-1]),
        'price_change_pct': float((df['Close'].iloc[-1] / df['Close'].iloc[0] - 1) * 100),

        # Physics means
        'energy_mean': float(state['energy'].mean()),
        'energy_std': float(state['energy'].std()),
        'damping_mean': float(state['damping'].mean()),
        'damping_std': float(state['damping'].std()),
        'entropy_mean': float(state['entropy'].mean()),
        'entropy_std': float(state['entropy'].std()),

        # Regime distribution
        'regime_distribution': state['regime'].value_counts(normalize=True).to_dict(),

        # Agent activation distribution
        'agent_distribution': state['active_agent'].value_counts(normalize=True).to_dict(),
    }

    return analysis


def save_results(
    filepath: str,
    strategy_name: str,
    result: dict,
    dataset_analysis: dict,
    output_dir: str
):
    """Save comprehensive results to files."""
    os.makedirs(output_dir, exist_ok=True)

    base_name = Path(filepath).stem
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Full result JSON
    full_result = {
        'file': filepath,
        'strategy': strategy_name,
        'timestamp': timestamp,
        'dataset_analysis': dataset_analysis,
        'metrics': result['metrics'],
        'trades': result['trades'],
        'success': result['success'],
        'error': result.get('error')
    }

    json_path = os.path.join(output_dir, f'{base_name}_{strategy_name}_{timestamp}.json')
    with open(json_path, 'w') as f:
        json.dump(full_result, f, indent=2, default=str)

    return json_path


def main():
    if len(sys.argv) < 2:
        print("Usage: python run_comprehensive_backtest.py <csv_file1> [csv_file2] ...")
        print("\nExample:")
        print("  python scripts/run_comprehensive_backtest.py data/*.csv")
        print("  python scripts/run_comprehensive_backtest.py /path/to/BTCUSD_H1.csv")
        sys.exit(1)

    csv_files = sys.argv[1:]
    output_dir = 'backtest_results'

    print("=" * 80)
    print(" KINETRA COMPREHENSIVE PHYSICS BACKTESTER")
    print(" Energy-Transfer Trading Theorem v7.0")
    print(" Detailed Logging for RL Reward Shaping")
    print("=" * 80)
    print(f"\nFiles to process: {len(csv_files)}")
    print(f"Strategies: {list(V7_STRATEGIES.keys())}")
    print(f"Output directory: {output_dir}")

    all_metrics = []
    all_trades = []

    for filepath in csv_files:
        if not os.path.exists(filepath):
            print(f"\nSkipping {filepath} - file not found")
            continue

        filename = os.path.basename(filepath)
        print(f"\n{'='*80}")
        print(f"Processing: {filename}")
        print("=" * 80)

        # Load data
        try:
            df = load_mt5_csv(filepath)

            # Ensure datetime index
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.date_range(start='2020-01-01', periods=len(df), freq='h')

            print(f"  Loaded {len(df)} bars")
            print(f"  Date range: {df.index[0]} to {df.index[-1]}")
            print(f"  Price: {df['Close'].iloc[0]:.5f} -> {df['Close'].iloc[-1]:.5f}")
        except Exception as e:
            print(f"  ERROR loading data: {e}")
            continue

        # Compute physics features
        print("  Computing physics features...")
        physics_features = compute_physics_features(df)

        # Dataset analysis
        print("  Analyzing dataset physics...")
        dataset_analysis = analyze_dataset_physics(df)

        print(f"    Energy: {dataset_analysis['energy_mean']:.6f} ± {dataset_analysis['energy_std']:.6f}")
        print(f"    Damping: {dataset_analysis['damping_mean']:.4f} ± {dataset_analysis['damping_std']:.4f}")
        print(f"    Regimes: {dataset_analysis['regime_distribution']}")
        print(f"    Agents: {dataset_analysis['agent_distribution']}")

        # Run each strategy
        print("\n  Running strategies with detailed logging...")

        for strategy_name, strategy_class in V7_STRATEGIES.items():
            print(f"\n  [{strategy_name.upper()}]")

            result = run_backtest_with_logging(df, strategy_class, physics_features)

            if result['success']:
                m = result['metrics']
                print(f"    Return: {m['total_return_pct']:+.2f}%")
                print(f"    Trades: {m['num_trades']}")
                print(f"    Win Rate: {m['win_rate']:.1f}%")
                print(f"    Omega: {m.get('omega_ratio', 0):.3f}")
                print(f"    Sortino: {m.get('sortino_ratio', 0):.3f}")
                print(f"    Ulcer: {m.get('ulcer_index', 0):.3f}")
                print(f"    Max DD: {m['max_drawdown_pct']:.2f}%")

                if m['num_trades'] > 0:
                    print(f"    Avg Duration: {m.get('avg_duration_bars', 0):.1f} bars")
                    print(f"    Win/Loss Ratio: {m.get('win_loss_ratio', 0):.2f}")
                    print(f"    Max Win Streak: {m.get('max_win_streak', 0)}")
                    print(f"    Max Loss Streak: {m.get('max_loss_streak', 0)}")

                # Save detailed results
                json_path = save_results(filepath, strategy_name, result, dataset_analysis, output_dir)
                print(f"    Saved: {json_path}")

                # Collect for summary
                metric_row = {
                    'file': filename,
                    'strategy': strategy_name,
                    **m
                }
                all_metrics.append(metric_row)

                # Collect trades
                for trade in result['trades']:
                    trade['file'] = filename
                    trade['strategy'] = strategy_name
                    all_trades.append(trade)
            else:
                print(f"    FAILED: {result.get('error', 'Unknown error')}")

    # Save combined results
    if all_metrics:
        print("\n" + "=" * 80)
        print(" SAVING COMBINED RESULTS")
        print("=" * 80)

        os.makedirs(output_dir, exist_ok=True)

        # Metrics summary
        metrics_df = pd.DataFrame(all_metrics)
        metrics_path = os.path.join(output_dir, 'all_run_metrics.csv')
        metrics_df.to_csv(metrics_path, index=False)
        print(f"  Run metrics: {metrics_path}")

        # All trades
        if all_trades:
            trades_df = pd.DataFrame(all_trades)
            trades_path = os.path.join(output_dir, 'all_trades.csv')
            trades_df.to_csv(trades_path, index=False)
            print(f"  Trade details: {trades_path}")
            print(f"  Total trades logged: {len(all_trades)}")

        # Summary by strategy
        print("\n  SUMMARY BY STRATEGY:")
        print("-" * 60)

        summary = metrics_df.groupby('strategy').agg({
            'total_return_pct': ['mean', 'std'],
            'omega_ratio': 'mean',
            'sortino_ratio': 'mean',
            'ulcer_index': 'mean',
            'num_trades': 'sum',
            'win_rate': 'mean',
            'max_drawdown_pct': 'mean'
        }).round(3)

        print(summary.to_string())

        # Save summary
        summary_path = os.path.join(output_dir, 'strategy_summary.csv')
        summary.to_csv(summary_path)
        print(f"\n  Strategy summary: {summary_path}")

    print("\n" + "=" * 80)
    print(" COMPREHENSIVE BACKTEST COMPLETE")
    print("=" * 80)
    print(f"\nResults saved to: {output_dir}/")
    print("Use these files for:")
    print("  - all_run_metrics.csv: Composite health score inputs")
    print("  - all_trades.csv: RL reward shaping features")
    print("  - *.json: Full per-run details with physics state")


if __name__ == '__main__':
    main()
