"""
Plotting Module for Backtest Visualization

Generates publication-quality plots for:
- Equity curves
- Trade distributions
- Drawdown analysis
- Cost breakdown
- Physics regime analysis
- Monte Carlo distributions
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime

# Import plotting libraries with fallback
try:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from matplotlib.ticker import FuncFormatter
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Warning: matplotlib not available. Install with: pip install matplotlib")

try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False


def setup_style():
    """Set up consistent plot styling."""
    if not MATPLOTLIB_AVAILABLE:
        return

    plt.style.use('seaborn-v0_8-darkgrid')
    plt.rcParams.update({
        'figure.figsize': (12, 6),
        'figure.dpi': 100,
        'savefig.dpi': 150,
        'font.size': 10,
        'axes.titlesize': 12,
        'axes.labelsize': 10,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 9,
        'lines.linewidth': 1.5,
    })


def currency_formatter(x, pos):
    """Format axis labels as currency."""
    if abs(x) >= 1e6:
        return f'${x/1e6:.1f}M'
    elif abs(x) >= 1e3:
        return f'${x/1e3:.1f}K'
    else:
        return f'${x:.0f}'


def plot_equity_curve(
    equity: pd.Series,
    title: str = "Equity Curve",
    initial_capital: float = None,
    save_path: Optional[Path] = None,
    show: bool = True
) -> Optional[plt.Figure]:
    """
    Plot equity curve with drawdown overlay.

    Args:
        equity: Series of equity values
        title: Plot title
        initial_capital: Initial capital for reference line
        save_path: Path to save figure
        show: Whether to display the plot

    Returns:
        matplotlib Figure object
    """
    if not MATPLOTLIB_AVAILABLE:
        print("matplotlib required for plotting")
        return None

    setup_style()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), height_ratios=[3, 1])

    # Equity curve
    ax1.plot(equity.values, color='#2E86AB', linewidth=1.5, label='Equity')

    if initial_capital:
        ax1.axhline(y=initial_capital, color='gray', linestyle='--',
                    linewidth=1, label=f'Initial: ${initial_capital:,.0f}')

    ax1.fill_between(range(len(equity)), initial_capital or equity.iloc[0],
                     equity.values, alpha=0.3, color='#2E86AB')

    ax1.set_title(title, fontsize=14, fontweight='bold')
    ax1.set_ylabel('Equity ($)')
    ax1.yaxis.set_major_formatter(FuncFormatter(currency_formatter))
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)

    # Calculate drawdown
    rolling_max = equity.expanding().max()
    drawdown = (equity - rolling_max) / rolling_max * 100

    # Drawdown subplot
    ax2.fill_between(range(len(drawdown)), 0, drawdown.values,
                     color='#E63946', alpha=0.7)
    ax2.set_ylabel('Drawdown (%)')
    ax2.set_xlabel('Bar')
    ax2.set_ylim(drawdown.min() * 1.1, 5)
    ax2.grid(True, alpha=0.3)

    # Add max drawdown annotation
    max_dd_idx = drawdown.idxmin()
    max_dd_val = drawdown.min()
    ax2.annotate(f'Max DD: {max_dd_val:.1f}%',
                 xy=(max_dd_idx, max_dd_val),
                 xytext=(max_dd_idx + len(equity)*0.05, max_dd_val * 0.7),
                 arrowprops=dict(arrowstyle='->', color='#E63946'),
                 fontsize=9, color='#E63946')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', facecolor='white')

    if show:
        plt.show()

    return fig


def plot_trade_distribution(
    trades_df: pd.DataFrame,
    title: str = "Trade P&L Distribution",
    save_path: Optional[Path] = None,
    show: bool = True
) -> Optional[plt.Figure]:
    """
    Plot distribution of trade P&L.

    Args:
        trades_df: DataFrame with trade data (must have 'net_pnl' column)
        title: Plot title
        save_path: Path to save figure
        show: Whether to display

    Returns:
        matplotlib Figure object
    """
    if not MATPLOTLIB_AVAILABLE:
        return None

    setup_style()
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. P&L histogram
    ax1 = axes[0, 0]
    pnl = trades_df['net_pnl']
    colors = ['#2E86AB' if x >= 0 else '#E63946' for x in pnl]

    ax1.hist(pnl, bins=30, color='#2E86AB', alpha=0.7, edgecolor='white')
    ax1.axvline(x=0, color='black', linestyle='--', linewidth=1)
    ax1.axvline(x=pnl.mean(), color='#F4A261', linestyle='-', linewidth=2,
                label=f'Mean: ${pnl.mean():.2f}')
    ax1.set_title('P&L Distribution')
    ax1.set_xlabel('P&L ($)')
    ax1.set_ylabel('Frequency')
    ax1.legend()

    # 2. Win/Loss pie chart
    ax2 = axes[0, 1]
    winners = (pnl > 0).sum()
    losers = (pnl <= 0).sum()
    colors_pie = ['#2E86AB', '#E63946']
    explode = (0.05, 0)
    ax2.pie([winners, losers], labels=['Winners', 'Losers'],
            colors=colors_pie, explode=explode, autopct='%1.1f%%',
            startangle=90, shadow=True)
    ax2.set_title(f'Win Rate: {winners}/{winners+losers} ({winners/(winners+losers)*100:.1f}%)')

    # 3. Cumulative P&L
    ax3 = axes[1, 0]
    cumulative = pnl.cumsum()
    ax3.plot(cumulative.values, color='#2E86AB', linewidth=1.5)
    ax3.fill_between(range(len(cumulative)), 0, cumulative.values,
                     where=cumulative >= 0, color='#2E86AB', alpha=0.3)
    ax3.fill_between(range(len(cumulative)), 0, cumulative.values,
                     where=cumulative < 0, color='#E63946', alpha=0.3)
    ax3.axhline(y=0, color='black', linestyle='--', linewidth=1)
    ax3.set_title('Cumulative P&L')
    ax3.set_xlabel('Trade #')
    ax3.set_ylabel('Cumulative P&L ($)')
    ax3.yaxis.set_major_formatter(FuncFormatter(currency_formatter))

    # 4. P&L by trade sequence
    ax4 = axes[1, 1]
    colors = ['#2E86AB' if x >= 0 else '#E63946' for x in pnl]
    ax4.bar(range(len(pnl)), pnl, color=colors, alpha=0.7)
    ax4.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax4.set_title('P&L by Trade')
    ax4.set_xlabel('Trade #')
    ax4.set_ylabel('P&L ($)')

    plt.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', facecolor='white')

    if show:
        plt.show()

    return fig


def plot_cost_breakdown(
    costs: Dict[str, float],
    gross_pnl: float,
    title: str = "Trading Costs Breakdown",
    save_path: Optional[Path] = None,
    show: bool = True
) -> Optional[plt.Figure]:
    """
    Plot breakdown of trading costs.

    Args:
        costs: Dict with cost categories {spread, commission, slippage, swap}
        gross_pnl: Gross P&L for comparison
        save_path: Path to save figure
        show: Whether to display

    Returns:
        matplotlib Figure object
    """
    if not MATPLOTLIB_AVAILABLE:
        return None

    setup_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # 1. Cost pie chart
    labels = list(costs.keys())
    values = [abs(v) for v in costs.values()]
    colors = ['#2E86AB', '#F4A261', '#E76F51', '#E9C46A']

    ax1.pie(values, labels=labels, colors=colors[:len(labels)],
            autopct=lambda p: f'${p*sum(values)/100:.0f}\n({p:.1f}%)',
            startangle=90, shadow=True)
    ax1.set_title(f'Cost Breakdown (Total: ${sum(values):,.2f})')

    # 2. Gross vs Net comparison
    total_costs = sum(values)
    net_pnl = gross_pnl - total_costs

    categories = ['Gross P&L', 'Costs', 'Net P&L']
    bar_values = [gross_pnl, -total_costs, net_pnl]
    bar_colors = ['#2E86AB', '#E63946', '#2A9D8F' if net_pnl >= 0 else '#E63946']

    bars = ax2.bar(categories, bar_values, color=bar_colors, alpha=0.8)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax2.set_title('P&L Impact of Costs')
    ax2.set_ylabel('Amount ($)')
    ax2.yaxis.set_major_formatter(FuncFormatter(currency_formatter))

    # Add value labels on bars
    for bar, val in zip(bars, bar_values):
        height = bar.get_height()
        ax2.annotate(f'${val:,.0f}',
                     xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 3 if height >= 0 else -15),
                     textcoords="offset points",
                     ha='center', va='bottom' if height >= 0 else 'top',
                     fontsize=10, fontweight='bold')

    plt.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', facecolor='white')

    if show:
        plt.show()

    return fig


def plot_regime_analysis(
    trades_df: pd.DataFrame,
    title: str = "Performance by Market Regime",
    save_path: Optional[Path] = None,
    show: bool = True
) -> Optional[plt.Figure]:
    """
    Plot performance breakdown by market regime.

    Args:
        trades_df: DataFrame with trade data (must have 'regime_at_entry', 'net_pnl')
        save_path: Path to save figure
        show: Whether to display

    Returns:
        matplotlib Figure object
    """
    if not MATPLOTLIB_AVAILABLE or 'regime_at_entry' not in trades_df.columns:
        return None

    setup_style()
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    regime_colors = {
        'UNDERDAMPED': '#2E86AB',
        'CRITICAL': '#F4A261',
        'OVERDAMPED': '#E63946',
    }

    # Group by regime
    regime_stats = trades_df.groupby('regime_at_entry').agg({
        'net_pnl': ['sum', 'mean', 'count'],
    }).round(2)
    regime_stats.columns = ['total_pnl', 'avg_pnl', 'count']

    regimes = regime_stats.index.tolist()
    colors = [regime_colors.get(r, '#888888') for r in regimes]

    # 1. Total P&L by regime
    ax1 = axes[0]
    bars = ax1.bar(regimes, regime_stats['total_pnl'], color=colors, alpha=0.8)
    ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax1.set_title('Total P&L by Regime')
    ax1.set_ylabel('P&L ($)')
    ax1.yaxis.set_major_formatter(FuncFormatter(currency_formatter))

    # 2. Trade count by regime
    ax2 = axes[1]
    ax2.bar(regimes, regime_stats['count'], color=colors, alpha=0.8)
    ax2.set_title('Trade Count by Regime')
    ax2.set_ylabel('# Trades')

    # 3. Average P&L by regime
    ax3 = axes[2]
    ax3.bar(regimes, regime_stats['avg_pnl'], color=colors, alpha=0.8)
    ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax3.set_title('Average P&L by Regime')
    ax3.set_ylabel('Avg P&L ($)')

    plt.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', facecolor='white')

    if show:
        plt.show()

    return fig


def plot_monte_carlo_results(
    mc_results: pd.DataFrame,
    metric: str = "total_net_pnl",
    title: str = "Monte Carlo Distribution",
    save_path: Optional[Path] = None,
    show: bool = True
) -> Optional[plt.Figure]:
    """
    Plot Monte Carlo simulation results.

    Args:
        mc_results: DataFrame from monte_carlo_validation
        metric: Which metric to plot distribution of
        save_path: Path to save figure
        show: Whether to display

    Returns:
        matplotlib Figure object
    """
    if not MATPLOTLIB_AVAILABLE:
        return None

    setup_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    values = mc_results[metric]

    # 1. Distribution histogram
    ax1.hist(values, bins=30, color='#2E86AB', alpha=0.7, edgecolor='white')
    ax1.axvline(x=values.mean(), color='#E63946', linestyle='-', linewidth=2,
                label=f'Mean: ${values.mean():,.2f}')
    ax1.axvline(x=values.median(), color='#F4A261', linestyle='--', linewidth=2,
                label=f'Median: ${values.median():,.2f}')

    # Add confidence intervals
    p5 = np.percentile(values, 5)
    p95 = np.percentile(values, 95)
    ax1.axvline(x=p5, color='gray', linestyle=':', linewidth=1,
                label=f'5th %ile: ${p5:,.2f}')
    ax1.axvline(x=p95, color='gray', linestyle=':', linewidth=1,
                label=f'95th %ile: ${p95:,.2f}')

    ax1.set_title(f'{metric} Distribution ({len(values)} runs)')
    ax1.set_xlabel(metric)
    ax1.set_ylabel('Frequency')
    ax1.legend()

    # 2. Box plot for multiple metrics
    if SEABORN_AVAILABLE:
        metrics_to_plot = ['total_net_pnl', 'sharpe_ratio', 'omega_ratio', 'max_drawdown']
        available_metrics = [m for m in metrics_to_plot if m in mc_results.columns]

        if available_metrics:
            # Normalize for comparison
            normalized = mc_results[available_metrics].apply(
                lambda x: (x - x.min()) / (x.max() - x.min()) if x.max() != x.min() else 0
            )
            normalized_melted = normalized.melt(var_name='Metric', value_name='Normalized Value')
            sns.boxplot(data=normalized_melted, x='Metric', y='Normalized Value', ax=ax2)
            ax2.set_title('Metric Distributions (Normalized)')
            ax2.tick_params(axis='x', rotation=45)
    else:
        ax2.text(0.5, 0.5, 'Install seaborn for additional plots',
                 transform=ax2.transAxes, ha='center')

    plt.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', facecolor='white')

    if show:
        plt.show()

    return fig


def generate_all_plots(
    result: Dict,
    trades_df: pd.DataFrame,
    equity_series: pd.Series,
    output_dir: Path,
    show: bool = False
):
    """
    Generate all standard plots for a backtest run.

    Args:
        result: Backtest result dict
        trades_df: DataFrame of trades
        equity_series: Equity curve series
        output_dir: Directory to save plots
        show: Whether to display plots
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Equity curve
    plot_equity_curve(
        equity_series,
        title="Equity Curve",
        save_path=output_dir / "equity_curve.png",
        show=show
    )

    # 2. Trade distribution
    if not trades_df.empty:
        plot_trade_distribution(
            trades_df,
            title="Trade Analysis",
            save_path=output_dir / "trade_distribution.png",
            show=show
        )

    # 3. Cost breakdown
    costs = result.get("cost_breakdown", {})
    if costs:
        plot_cost_breakdown(
            costs,
            result.get("total_gross_pnl", 0),
            title="Cost Analysis",
            save_path=output_dir / "cost_breakdown.png",
            show=show
        )

    # 4. Regime analysis
    if 'regime_at_entry' in trades_df.columns:
        plot_regime_analysis(
            trades_df,
            title="Regime Performance",
            save_path=output_dir / "regime_analysis.png",
            show=show
        )

    print(f"Plots saved to: {output_dir}")


if __name__ == "__main__":
    # Test with sample data
    np.random.seed(42)

    # Sample equity curve
    equity = pd.Series(np.cumsum(np.random.randn(500) * 100) + 10000)

    # Sample trades
    trades_df = pd.DataFrame({
        'net_pnl': np.random.randn(50) * 200,
        'regime_at_entry': np.random.choice(['UNDERDAMPED', 'CRITICAL', 'OVERDAMPED'], 50)
    })

    # Test plots
    plot_equity_curve(equity, show=True)
    plot_trade_distribution(trades_df, show=True)
