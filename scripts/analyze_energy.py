#!/usr/bin/env python3
"""
Energy Analysis Script

Analyzes potential energy across a backtest period to identify:
- Top third of energy releases (long opportunities)
- Top third of energy releases (short opportunities)
- Regime distribution
- Energy capture potential

Usage:
    python scripts/analyze_energy.py --symbol BTCUSD --timeframe H1
"""

import argparse
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from kinetra import PhysicsEngine, load_csv_data
from kinetra.plotting import setup_style, MATPLOTLIB_AVAILABLE

if MATPLOTLIB_AVAILABLE:
    import matplotlib.pyplot as plt


def analyze_energy(data: pd.DataFrame, lookback: int = 20) -> pd.DataFrame:
    """
    Compute energy metrics for each bar.

    Returns DataFrame with:
    - energy: kinetic energy
    - momentum: price momentum (direction indicator)
    - energy_long: energy for upward moves (positive momentum)
    - energy_short: energy for downward moves (negative momentum)
    """
    engine = PhysicsEngine(lookback=lookback)

    # Compute physics state
    physics = engine.compute_physics_state(data['close'])

    # Add to dataframe
    result = data.copy()
    result['energy'] = physics['energy']
    result['damping'] = physics['damping']
    result['entropy'] = physics['entropy']
    result['regime'] = physics['regime']

    # Calculate momentum direction
    result['momentum'] = data['close'].diff(lookback)
    result['momentum_pct'] = data['close'].pct_change(lookback) * 100

    # Separate long vs short energy
    # Long: positive momentum (price going up) with high energy
    # Short: negative momentum (price going down) with high energy
    result['energy_long'] = np.where(result['momentum'] > 0, result['energy'], 0)
    result['energy_short'] = np.where(result['momentum'] < 0, result['energy'], 0)

    # Calculate potential profit per unit energy
    # This shows how much price moved per unit energy
    result['efficiency'] = np.abs(result['momentum']) / (result['energy'] + 1e-10)

    return result


def identify_top_energy_periods(
    df: pd.DataFrame,
    percentile: float = 66.67,  # Top third
    direction: str = "long"
) -> pd.DataFrame:
    """
    Identify periods with top energy values.

    Args:
        df: DataFrame with energy columns
        percentile: Threshold percentile (66.67 = top third)
        direction: "long" or "short"

    Returns:
        DataFrame filtered to high-energy periods
    """
    col = f"energy_{direction}"
    if col not in df.columns:
        raise ValueError(f"Column {col} not found")

    # Only look at non-zero energy in the specified direction
    non_zero = df[df[col] > 0].copy()

    if len(non_zero) == 0:
        return pd.DataFrame()

    threshold = np.percentile(non_zero[col], percentile)
    high_energy = non_zero[non_zero[col] >= threshold].copy()

    # Add rank
    high_energy['energy_rank'] = high_energy[col].rank(ascending=False)
    high_energy['energy_percentile'] = high_energy[col].rank(pct=True) * 100

    return high_energy.sort_values(col, ascending=False)


def compute_distribution_stats(values: np.ndarray, name: str) -> dict:
    """
    Compute full distribution statistics for a set of values.
    """
    if len(values) == 0:
        return {"name": name, "count": 0}

    return {
        "name": name,
        "count": len(values),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "median": float(np.median(values)),
        "skewness": float(pd.Series(values).skew()),
        "kurtosis": float(pd.Series(values).kurtosis()),
        "percentiles": {
            "p5": float(np.percentile(values, 5)),
            "p10": float(np.percentile(values, 10)),
            "p25": float(np.percentile(values, 25)),
            "p33": float(np.percentile(values, 33.33)),
            "p50": float(np.percentile(values, 50)),
            "p66": float(np.percentile(values, 66.67)),
            "p75": float(np.percentile(values, 75)),
            "p90": float(np.percentile(values, 90)),
            "p95": float(np.percentile(values, 95)),
        },
    }


def compute_pareto_analysis(values: np.ndarray) -> dict:
    """
    Find the Pareto point: what % of observations give 80% of total energy?

    Returns:
        Dict with pareto analysis results
    """
    if len(values) == 0:
        return {}

    # Sort descending
    sorted_vals = np.sort(values)[::-1]
    total = np.sum(sorted_vals)

    if total == 0:
        return {}

    # Cumulative sum
    cumsum = np.cumsum(sorted_vals)

    # Find how many observations to get 80% of energy
    target_80 = total * 0.80
    n_for_80 = np.searchsorted(cumsum, target_80) + 1
    pct_for_80 = n_for_80 / len(values) * 100

    # Also find 50%, 90%, 95% thresholds
    target_50 = total * 0.50
    target_90 = total * 0.90
    target_95 = total * 0.95

    n_for_50 = np.searchsorted(cumsum, target_50) + 1
    n_for_90 = np.searchsorted(cumsum, target_90) + 1
    n_for_95 = np.searchsorted(cumsum, target_95) + 1

    # Energy contribution by percentile bands
    # Top 1%, 5%, 10%, 20% of observations
    top_1_idx = max(1, int(len(values) * 0.01))
    top_5_idx = max(1, int(len(values) * 0.05))
    top_10_idx = max(1, int(len(values) * 0.10))
    top_20_idx = max(1, int(len(values) * 0.20))

    return {
        "total_energy": float(total),
        "pareto_80": {
            "pct_observations": float(pct_for_80),
            "n_observations": int(n_for_80),
            "threshold_value": float(sorted_vals[n_for_80 - 1]) if n_for_80 <= len(sorted_vals) else 0,
        },
        "energy_by_observation_pct": {
            "top_1_pct": float(np.sum(sorted_vals[:top_1_idx]) / total * 100),
            "top_5_pct": float(np.sum(sorted_vals[:top_5_idx]) / total * 100),
            "top_10_pct": float(np.sum(sorted_vals[:top_10_idx]) / total * 100),
            "top_20_pct": float(np.sum(sorted_vals[:top_20_idx]) / total * 100),
        },
        "observations_for_cumulative": {
            "pct_for_50_energy": float(n_for_50 / len(values) * 100),
            "pct_for_80_energy": float(pct_for_80),
            "pct_for_90_energy": float(n_for_90 / len(values) * 100),
            "pct_for_95_energy": float(n_for_95 / len(values) * 100),
        },
    }


def summarize_energy_opportunities(
    df: pd.DataFrame,
    high_long: pd.DataFrame,
    high_short: pd.DataFrame
) -> dict:
    """
    Create summary statistics for energy opportunities.
    """
    total_bars = len(df)

    # Long opportunities
    long_count = len(high_long)
    long_total_energy = high_long['energy_long'].sum() if len(high_long) > 0 else 0
    long_avg_momentum = high_long['momentum_pct'].mean() if len(high_long) > 0 else 0
    long_avg_energy = high_long['energy_long'].mean() if len(high_long) > 0 else 0

    # Short opportunities
    short_count = len(high_short)
    short_total_energy = high_short['energy_short'].sum() if len(high_short) > 0 else 0
    short_avg_momentum = high_short['momentum_pct'].mean() if len(high_short) > 0 else 0
    short_avg_energy = high_short['energy_short'].mean() if len(high_short) > 0 else 0

    # Regime distribution
    regime_dist = df['regime'].value_counts(normalize=True) * 100

    # Energy by regime
    energy_by_regime = df.groupby('regime')['energy'].agg(['mean', 'sum', 'count'])

    # Full distribution stats
    all_energy = df['energy'].values
    long_energy = df[df['energy_long'] > 0]['energy_long'].values
    short_energy = df[df['energy_short'] > 0]['energy_short'].values

    return {
        "period": {
            "total_bars": total_bars,
            "start": str(df.iloc[0].get('time', 'N/A')),
            "end": str(df.iloc[-1].get('time', 'N/A')),
        },
        "long_opportunities": {
            "count": long_count,
            "pct_of_total": long_count / total_bars * 100,
            "total_energy": long_total_energy,
            "avg_energy": long_avg_energy,
            "avg_momentum_pct": long_avg_momentum,
        },
        "short_opportunities": {
            "count": short_count,
            "pct_of_total": short_count / total_bars * 100,
            "total_energy": short_total_energy,
            "avg_energy": short_avg_energy,
            "avg_momentum_pct": short_avg_momentum,
        },
        "regime_distribution": regime_dist.to_dict() if hasattr(regime_dist, 'to_dict') else {},
        "energy_by_regime": energy_by_regime.to_dict() if hasattr(energy_by_regime, 'to_dict') else {},
        "distributions": {
            "all_energy": compute_distribution_stats(all_energy, "All Energy"),
            "long_energy": compute_distribution_stats(long_energy, "Long Energy"),
            "short_energy": compute_distribution_stats(short_energy, "Short Energy"),
        },
        "pareto": {
            "all_energy": compute_pareto_analysis(all_energy),
            "long_energy": compute_pareto_analysis(long_energy),
            "short_energy": compute_pareto_analysis(short_energy),
        },
    }


def plot_energy_analysis(
    df: pd.DataFrame,
    high_long: pd.DataFrame,
    high_short: pd.DataFrame,
    symbol: str,
    save_path: Path = None,
    show: bool = True
):
    """Generate energy analysis plots."""
    if not MATPLOTLIB_AVAILABLE:
        print("matplotlib required for plotting")
        return

    setup_style()
    fig = plt.figure(figsize=(16, 12))

    # 1. Price with high energy periods highlighted
    ax1 = fig.add_subplot(3, 2, 1)
    ax1.plot(df['close'].values, color='gray', linewidth=0.5, alpha=0.7)

    # Highlight high energy long periods
    for idx in high_long.index:
        pos = df.index.get_loc(idx)
        ax1.axvline(x=pos, color='green', alpha=0.3, linewidth=1)

    # Highlight high energy short periods
    for idx in high_short.index:
        pos = df.index.get_loc(idx)
        ax1.axvline(x=pos, color='red', alpha=0.3, linewidth=1)

    ax1.set_title(f'{symbol} Price with High Energy Periods')
    ax1.set_ylabel('Price')

    # 2. Energy time series
    ax2 = fig.add_subplot(3, 2, 2)
    ax2.fill_between(range(len(df)), 0, df['energy_long'].values,
                     color='green', alpha=0.5, label='Long Energy')
    ax2.fill_between(range(len(df)), 0, -df['energy_short'].values,
                     color='red', alpha=0.5, label='Short Energy')
    ax2.set_title('Energy Over Time (Long vs Short)')
    ax2.set_ylabel('Energy')
    ax2.legend()

    # 3. Energy distribution - Long
    ax3 = fig.add_subplot(3, 2, 3)
    long_energy = df[df['energy_long'] > 0]['energy_long']
    if len(long_energy) > 0:
        ax3.hist(long_energy, bins=50, color='green', alpha=0.7, edgecolor='white')
        threshold = np.percentile(long_energy, 66.67)
        ax3.axvline(x=threshold, color='darkgreen', linestyle='--', linewidth=2,
                    label=f'Top Third Threshold: {threshold:.2f}')
    ax3.set_title('Long Energy Distribution')
    ax3.set_xlabel('Energy')
    ax3.set_ylabel('Frequency')
    ax3.legend()

    # 4. Energy distribution - Short
    ax4 = fig.add_subplot(3, 2, 4)
    short_energy = df[df['energy_short'] > 0]['energy_short']
    if len(short_energy) > 0:
        ax4.hist(short_energy, bins=50, color='red', alpha=0.7, edgecolor='white')
        threshold = np.percentile(short_energy, 66.67)
        ax4.axvline(x=threshold, color='darkred', linestyle='--', linewidth=2,
                    label=f'Top Third Threshold: {threshold:.2f}')
    ax4.set_title('Short Energy Distribution')
    ax4.set_xlabel('Energy')
    ax4.set_ylabel('Frequency')
    ax4.legend()

    # 5. Regime distribution
    ax5 = fig.add_subplot(3, 2, 5)
    regime_counts = df['regime'].value_counts()
    colors = {'UNDERDAMPED': '#2E86AB', 'CRITICAL': '#F4A261', 'OVERDAMPED': '#E63946'}
    bar_colors = [colors.get(r, '#888888') for r in regime_counts.index]
    ax5.bar(regime_counts.index, regime_counts.values, color=bar_colors, alpha=0.8)
    ax5.set_title('Regime Distribution')
    ax5.set_ylabel('Count')

    # 6. Energy by Regime
    ax6 = fig.add_subplot(3, 2, 6)
    energy_by_regime = df.groupby('regime')['energy'].mean()
    bar_colors = [colors.get(r, '#888888') for r in energy_by_regime.index]
    ax6.bar(energy_by_regime.index, energy_by_regime.values, color=bar_colors, alpha=0.8)
    ax6.set_title('Average Energy by Regime')
    ax6.set_ylabel('Avg Energy')

    plt.suptitle(f'{symbol} Energy Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', facecolor='white', dpi=150)
        print(f"Plot saved to: {save_path}")

    if show:
        plt.show()

    return fig


def main():
    parser = argparse.ArgumentParser(description='Analyze potential energy opportunities')
    parser.add_argument('--data', type=str, help='Path to CSV data file')
    parser.add_argument('--symbol', type=str, default='BTCUSD', help='Symbol name')
    parser.add_argument('--lookback', type=int, default=20, help='Lookback period for energy calculation')
    parser.add_argument('--percentile', type=float, default=66.67, help='Percentile threshold for top energy')
    parser.add_argument('--save', type=str, help='Path to save plot')
    parser.add_argument('--no-plot', action='store_true', help='Skip plotting')

    args = parser.parse_args()

    # Find data file
    if args.data:
        data_path = Path(args.data)
    else:
        # Look for matching file in current directory
        project_root = Path(__file__).parent.parent
        csv_files = list(project_root.glob(f"*{args.symbol}*.csv"))
        if not csv_files:
            print(f"No CSV file found for {args.symbol}")
            print("Available files:")
            for f in project_root.glob("*.csv"):
                print(f"  {f.name}")
            return
        data_path = csv_files[0]
        print(f"Using: {data_path.name}")

    # Load data
    print(f"\nLoading data from {data_path}...")
    data = load_csv_data(str(data_path))
    print(f"Loaded {len(data)} bars")

    # Analyze energy
    print(f"\nComputing energy with lookback={args.lookback}...")
    df = analyze_energy(data, lookback=args.lookback)

    # Identify top energy periods
    print(f"\nIdentifying top {100 - args.percentile:.1f}% energy opportunities...")
    high_long = identify_top_energy_periods(df, args.percentile, "long")
    high_short = identify_top_energy_periods(df, args.percentile, "short")

    # Summary
    summary = summarize_energy_opportunities(df, high_long, high_short)

    print("\n" + "=" * 60)
    print(f"ENERGY ANALYSIS: {args.symbol}")
    print("=" * 60)

    print(f"\nPeriod: {summary['period']['total_bars']} bars")
    print(f"  Start: {summary['period']['start']}")
    print(f"  End: {summary['period']['end']}")

    print(f"\n📈 LONG Opportunities (Top Third):")
    lo = summary['long_opportunities']
    print(f"  Count: {lo['count']} ({lo['pct_of_total']:.1f}% of bars)")
    print(f"  Total Energy: {lo['total_energy']:.2f}")
    print(f"  Avg Energy: {lo['avg_energy']:.4f}")
    print(f"  Avg Momentum: {lo['avg_momentum_pct']:.2f}%")

    print(f"\n📉 SHORT Opportunities (Top Third):")
    so = summary['short_opportunities']
    print(f"  Count: {so['count']} ({so['pct_of_total']:.1f}% of bars)")
    print(f"  Total Energy: {so['total_energy']:.2f}")
    print(f"  Avg Energy: {so['avg_energy']:.4f}")
    print(f"  Avg Momentum: {so['avg_momentum_pct']:.2f}%")

    print(f"\n🎯 Regime Distribution:")
    for regime, pct in summary['regime_distribution'].items():
        print(f"  {regime}: {pct:.1f}%")

    # Print full distribution statistics
    print("\n" + "=" * 60)
    print("ENERGY DISTRIBUTION (Bell Curve Stats)")
    print("=" * 60)

    for dist_name, dist in summary['distributions'].items():
        if dist['count'] == 0:
            continue

        print(f"\n{dist['name']}:")
        print(f"  Count:    {dist['count']}")
        print(f"  Mean:     {dist['mean']:.6f}")
        print(f"  Std Dev:  {dist['std']:.6f}")
        print(f"  Skewness: {dist['skewness']:.3f} {'(right-tailed)' if dist['skewness'] > 0 else '(left-tailed)'}")
        print(f"  Kurtosis: {dist['kurtosis']:.3f} {'(heavy tails)' if dist['kurtosis'] > 0 else '(light tails)'}")

        print(f"  Percentiles:")
        p = dist['percentiles']
        print(f"    5%:   {p['p5']:.6f}")
        print(f"    25%:  {p['p25']:.6f}")
        print(f"    33%:  {p['p33']:.6f}  <-- Lower third threshold")
        print(f"    50%:  {p['p50']:.6f}  (median)")
        print(f"    66%:  {p['p66']:.6f}  <-- Upper third threshold")
        print(f"    75%:  {p['p75']:.6f}")
        print(f"    95%:  {p['p95']:.6f}")
        print(f"  Range:    [{dist['min']:.6f}, {dist['max']:.6f}]")

    # Pareto Analysis
    print("\n" + "=" * 60)
    print("PARETO ANALYSIS (Energy Concentration)")
    print("=" * 60)

    for name, pareto in summary['pareto'].items():
        if not pareto:
            continue

        print(f"\n{name.replace('_', ' ').title()}:")
        print(f"  Total Energy: {pareto['total_energy']:,.2f}")

        p80 = pareto['pareto_80']
        print(f"\n  80% of energy captured by:")
        print(f"    {p80['pct_observations']:.1f}% of observations ({p80['n_observations']:,} bars)")
        print(f"    Threshold: energy >= {p80['threshold_value']:,.2f}")

        e_pct = pareto['energy_by_observation_pct']
        print(f"\n  Energy contribution by top bars:")
        print(f"    Top  1% of bars: {e_pct['top_1_pct']:5.1f}% of energy")
        print(f"    Top  5% of bars: {e_pct['top_5_pct']:5.1f}% of energy")
        print(f"    Top 10% of bars: {e_pct['top_10_pct']:5.1f}% of energy")
        print(f"    Top 20% of bars: {e_pct['top_20_pct']:5.1f}% of energy")

        obs = pareto['observations_for_cumulative']
        print(f"\n  Observations needed for cumulative energy:")
        print(f"    50% energy: {obs['pct_for_50_energy']:5.1f}% of bars")
        print(f"    80% energy: {obs['pct_for_80_energy']:5.1f}% of bars")
        print(f"    90% energy: {obs['pct_for_90_energy']:5.1f}% of bars")
        print(f"    95% energy: {obs['pct_for_95_energy']:5.1f}% of bars")

    print("\n" + "=" * 60)

    # Show top 10 long opportunities
    print("\n📈 Top 10 LONG Energy Bars:")
    if len(high_long) > 0:
        top_long = high_long.head(10)[['close', 'energy_long', 'momentum_pct', 'regime']]
        print(top_long.to_string())

    print("\n📉 Top 10 SHORT Energy Bars:")
    if len(high_short) > 0:
        top_short = high_short.head(10)[['close', 'energy_short', 'momentum_pct', 'regime']]
        print(top_short.to_string())

    # Plot
    if not args.no_plot:
        save_path = Path(args.save) if args.save else None
        plot_energy_analysis(df, high_long, high_short, args.symbol, save_path, show=True)


if __name__ == "__main__":
    main()
