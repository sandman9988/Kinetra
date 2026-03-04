"""
Trade Analytics - MAE, MFE, Streaks, and Performance Metrics
=============================================================

Computes comprehensive trade statistics from a list of completed trades.

Enhanced metrics include:
- Z-factor: statistical significance of trading edge
- Sharpe ratio: risk-adjusted returns
- Calmar ratio: return vs max drawdown
- MAE/MFE: Maximum Adverse/Favorable Excursion (USD and ratio)
- Holding time statistics
"""

from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np


@dataclass
class TradeAnalytics:
    """Comprehensive trade performance analytics."""

    # Trade counts
    n_trades: int = 0
    n_winners: int = 0
    n_losers: int = 0
    win_rate: float = 0.0

    # P&L
    gross_profit: float = 0.0
    gross_loss: float = 0.0
    net_pnl: float = 0.0
    avg_trade: float = 0.0
    avg_winner: float = 0.0
    avg_loser: float = 0.0

    # Friction cost breakdown
    total_spread_usd: float = 0.0
    total_commission_usd: float = 0.0
    total_swap_usd: float = 0.0
    total_friction_usd: float = 0.0
    avg_spread_per_trade: float = 0.0
    avg_commission_per_trade: float = 0.0
    avg_swap_per_trade: float = 0.0
    friction_pct_of_gross: float = 0.0  # Total friction as % of gross profit

    # Risk-adjusted ratios
    profit_factor: float = 0.0
    omega: float = 0.0
    expectancy: float = 0.0
    z_factor: float = 0.0  # Statistical significance of edge
    sharpe_ratio: float = 0.0  # Risk-adjusted returns (annualized)
    calmar_ratio: float = 0.0  # Return / Max DD

    # Streaks
    max_win_streak: int = 0
    max_loss_streak: int = 0
    current_streak: int = 0

    # MAE/MFE (Maximum Adverse/Favorable Excursion)
    avg_mae_usd: float = 0.0  # Average MAE in USD
    avg_mfe_usd: float = 0.0  # Average MFE in USD
    mfe_mae_ratio: float = 0.0  # MFE/MAE ratio (>1 is good)
    avg_mae_pct: float = 0.0  # MAE as % of target risk (legacy)
    avg_mfe_pct: float = 0.0  # MFE as % of target risk (legacy)

    # Holding time
    avg_holding_hours: float = 0.0  # Average trade duration in hours
    avg_winner_hours: float = 0.0  # Average winner duration
    avg_loser_hours: float = 0.0  # Average loser duration
    avg_run_capture: float = 0.0  # Mean held_bricks / trend_run_bricks
    median_run_capture: float = 0.0  # Median held_bricks / trend_run_bricks
    run_capture_samples: int = 0  # Number of trades with capture diagnostics

    # Drawdown
    max_drawdown_pct: float = 0.0
    max_drawdown_usd: float = 0.0

    # Equity
    final_equity: float = 0.0
    total_return_pct: float = 0.0


def calculate_streaks(trades: List[Any]) -> Dict[str, int]:
    """Calculate win/loss streaks from trade list."""
    max_win = 0
    max_loss = 0
    current_win = 0
    current_loss = 0

    for trade in trades:
        is_win = getattr(trade, "net_usd", 0) > 0

        if is_win:
            current_win += 1
            current_loss = 0
            max_win = max(max_win, current_win)
        else:
            current_loss += 1
            current_win = 0
            max_loss = max(max_loss, current_loss)

    return {
        "max_win_streak": max_win,
        "max_loss_streak": max_loss,
        "current_streak": current_win if current_win > 0 else -current_loss,
    }


def calculate_drawdown(equity_curve: np.ndarray) -> tuple[float, float]:
    """Calculate max drawdown from equity curve.

    Returns:
        Tuple of (max_drawdown_pct, max_drawdown_usd)
    """
    if len(equity_curve) == 0:
        return 0.0, 0.0

    peak = np.maximum.accumulate(equity_curve)
    drawdown_usd = equity_curve - peak
    max_dd_usd = float(np.min(drawdown_usd))  # Most negative (as USD loss)

    # Avoid division by zero
    peak_at_max_dd = peak[np.argmin(drawdown_usd)]
    if peak_at_max_dd > 0:
        max_dd_pct = float(max_dd_usd / peak_at_max_dd) * 100  # As percentage
    else:
        max_dd_pct = 0.0

    return max_dd_pct, abs(max_dd_usd)


def calculate_z_factor(winners: np.ndarray, losers: np.ndarray) -> float:
    """
    Calculate Z-factor (statistical significance of trading edge).

    Z-factor = (Avg Win - Avg Loss) / Std(Losses)

    Interpretation:
    - Z > 2.5: Statistically significant edge
    - Z > 1.5: Moderate edge
    - Z < 1.0: Weak edge

    Args:
        winners: Array of winning trade P&Ls
        losers: Array of losing trade P&Ls (negative values)

    Returns:
        Z-factor value
    """
    if len(winners) == 0 or len(losers) == 0:
        return 0.0

    avg_win = float(winners.mean())
    avg_loss = float(abs(losers.mean()))  # Convert to positive
    std_loss = float(losers.std())

    if std_loss < 1e-10:
        return 0.0

    return (avg_win - avg_loss) / std_loss


def calculate_sharpe_ratio(
    returns: np.ndarray,
    periods_per_year: int = 252 * 24,  # Default: hourly data (24*252)
    risk_free_rate: float = 0.0,
) -> float:
    """
    Calculate annualized Sharpe ratio from trade returns.

    Args:
        returns: Array of per-trade returns (as decimals, not USD)
        periods_per_year: Trading periods per year for annualization
        risk_free_rate: Annual risk-free rate (default 0.0)

    Returns:
        Annualized Sharpe ratio
    """
    if len(returns) < 2:
        return 0.0

    excess_returns = returns - (risk_free_rate / periods_per_year)
    std = excess_returns.std()

    if std < 1e-10:
        return 0.0

    return float(excess_returns.mean() / std * np.sqrt(periods_per_year))


def calculate_calmar_ratio(
    total_return_pct: float,
    max_drawdown_pct: float,
) -> float:
    """
    Calculate Calmar ratio (annualized return / max drawdown).

    Args:
        total_return_pct: Total return as percentage
        max_drawdown_pct: Max drawdown as percentage (positive value)

    Returns:
        Calmar ratio (higher is better)
    """
    if max_drawdown_pct < 1e-10:
        return float("inf") if total_return_pct > 0 else 0.0

    return total_return_pct / max_drawdown_pct


def extract_mae_mfe(trades: List[Any]) -> tuple[float, float, float]:
    """
    Extract MAE/MFE metrics from trade objects.

    Looks for 'max_adverse_excursion' and 'max_favorable_excursion' attributes.

    Returns:
        Tuple of (avg_mae_usd, avg_mfe_usd, mfe_mae_ratio)
    """
    maes = []
    mfes = []

    for trade in trades:
        mae = getattr(trade, "max_adverse_excursion", None)
        mfe = getattr(trade, "max_favorable_excursion", None)

        if mae is not None and mae != 0:
            maes.append(abs(float(mae)))
        if mfe is not None and mfe != 0:
            mfes.append(abs(float(mfe)))

    avg_mae = float(np.mean(maes)) if maes else 0.0
    avg_mfe = float(np.mean(mfes)) if mfes else 0.0

    if avg_mae > 0:
        mfe_mae_ratio = avg_mfe / avg_mae
    else:
        mfe_mae_ratio = 0.0

    return avg_mae, avg_mfe, mfe_mae_ratio


def extract_holding_times(trades: List[Any]) -> tuple[float, float, float]:
    """
    Extract holding time statistics from trade objects.

    Looks for 'holding_hours' property or calculates from entry/exit times.

    Returns:
        Tuple of (avg_holding_hours, avg_winner_hours, avg_loser_hours)
    """
    all_hours = []
    winner_hours = []
    loser_hours = []

    for trade in trades:
        # Try holding_hours property first
        hours = getattr(trade, "holding_hours", None)

        # Fall back to calculating from timestamps
        if hours is None:
            entry_time = getattr(trade, "entry_time", None)
            exit_time = getattr(trade, "exit_time", None)
            if entry_time is not None and exit_time is not None:
                try:
                    # Handle both datetime and pandas Timestamp
                    delta = exit_time - entry_time
                    hours = delta.total_seconds() / 3600.0
                except (AttributeError, TypeError):
                    hours = None

        if hours is not None and hours > 0:
            all_hours.append(float(hours))
            if getattr(trade, "net_usd", 0) > 0:
                winner_hours.append(float(hours))
            else:
                loser_hours.append(float(hours))

    avg_all = float(np.mean(all_hours)) if all_hours else 0.0
    avg_winner = float(np.mean(winner_hours)) if winner_hours else 0.0
    avg_loser = float(np.mean(loser_hours)) if loser_hours else 0.0

    return avg_all, avg_winner, avg_loser


def extract_run_capture(trades: List[Any]) -> tuple[float, float, int]:
    """
    Extract trend run-capture diagnostics.

    Expects trades to optionally carry:
    - ``n_bricks_held``: bricks held in position
    - ``trend_run_bricks``: total bricks in the captured trend run
    """
    captures: List[float] = []
    for trade in trades:
        held = getattr(trade, "n_bricks_held", None)
        run = getattr(trade, "trend_run_bricks", None)
        if held is None or run is None:
            continue
        try:
            held_f = float(held)
            run_f = float(run)
        except (TypeError, ValueError):
            continue
        if run_f <= 0:
            continue
        capture = held_f / run_f
        capture = min(max(capture, 0.0), 1.0)
        captures.append(float(capture))

    if not captures:
        return 0.0, 0.0, 0
    arr = np.asarray(captures, dtype=np.float64)
    return float(arr.mean()), float(np.median(arr)), int(arr.size)


def analyze_trades(
    trades: List[Any],
    initial_equity: float = 10000.0,
) -> TradeAnalytics:
    """
    Analyze a list of trades and compute comprehensive metrics.

    Args:
        trades: List of trade objects with 'net_usd' attribute (and optionally
                'max_adverse_excursion', 'max_favorable_excursion', 'holding_hours',
                'entry_time', 'exit_time', 'spread_usd', 'commission_usd', 'swap_usd')
        initial_equity: Starting account balance

    Returns:
        TradeAnalytics dataclass with all computed metrics
    """
    if not trades:
        return TradeAnalytics(final_equity=initial_equity)

    nets = np.array([float(getattr(t, "net_usd", 0)) for t in trades])
    n_trades = len(trades)

    winners = nets[nets > 0]
    losers = nets[nets < 0]

    n_winners = len(winners)
    n_losers = len(losers)
    win_rate = n_winners / n_trades if n_trades > 0 else 0.0

    gross_profit = float(winners.sum()) if len(winners) > 0 else 0.0
    gross_loss = float(abs(losers.sum())) if len(losers) > 0 else 0.0
    net_pnl = float(nets.sum())

    # Friction cost breakdown
    total_spread = sum(float(getattr(t, "spread_usd", 0)) for t in trades)
    total_commission = sum(float(getattr(t, "commission_usd", 0)) for t in trades)
    total_swap = sum(float(getattr(t, "swap_usd", 0)) for t in trades)
    total_friction = total_spread + total_commission + total_swap
    avg_spread = total_spread / n_trades if n_trades > 0 else 0.0
    avg_commission = total_commission / n_trades if n_trades > 0 else 0.0
    avg_swap = total_swap / n_trades if n_trades > 0 else 0.0
    friction_pct = (total_friction / gross_profit * 100) if gross_profit > 0 else 0.0

    # Basic ratios
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")
    omega = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    # Expectancy = (Win% * Avg Win) - (Loss% * Avg Loss)
    avg_winner = float(winners.mean()) if len(winners) > 0 else 0.0
    avg_loser = float(abs(losers.mean())) if len(losers) > 0 else 0.0
    expectancy = (win_rate * avg_winner) - ((1 - win_rate) * avg_loser)

    # Z-factor (statistical significance)
    z_factor = calculate_z_factor(winners, losers)

    # Streaks
    streaks = calculate_streaks(trades)

    # Equity curve and drawdown
    equity = initial_equity + np.cumsum(nets)
    final_equity = float(equity[-1]) if len(equity) > 0 else initial_equity
    max_dd_pct, max_dd_usd = calculate_drawdown(equity)

    # Total return
    total_return_pct = ((final_equity - initial_equity) / initial_equity) * 100

    # Sharpe ratio from per-trade returns
    # Use equity returns (percentage change per trade)
    if len(equity) > 1:
        equity_returns = np.diff(equity) / equity[:-1]
        sharpe = calculate_sharpe_ratio(equity_returns)
    else:
        sharpe = 0.0

    # Calmar ratio
    calmar = calculate_calmar_ratio(total_return_pct, abs(max_dd_pct))

    # MAE/MFE
    avg_mae_usd, avg_mfe_usd, mfe_mae_ratio = extract_mae_mfe(trades)

    # Holding times
    avg_holding, avg_winner_hrs, avg_loser_hrs = extract_holding_times(trades)
    avg_capture, median_capture, capture_samples = extract_run_capture(trades)

    return TradeAnalytics(
        n_trades=n_trades,
        n_winners=n_winners,
        n_losers=n_losers,
        win_rate=win_rate,
        gross_profit=gross_profit,
        gross_loss=gross_loss,
        net_pnl=net_pnl,
        avg_trade=net_pnl / n_trades if n_trades > 0 else 0.0,
        avg_winner=avg_winner,
        avg_loser=avg_loser,
        total_spread_usd=total_spread,
        total_commission_usd=total_commission,
        total_swap_usd=total_swap,
        total_friction_usd=total_friction,
        avg_spread_per_trade=avg_spread,
        avg_commission_per_trade=avg_commission,
        avg_swap_per_trade=avg_swap,
        friction_pct_of_gross=friction_pct,
        profit_factor=profit_factor,
        omega=omega,
        expectancy=expectancy,
        z_factor=z_factor,
        sharpe_ratio=sharpe,
        calmar_ratio=calmar,
        max_win_streak=streaks["max_win_streak"],
        max_loss_streak=streaks["max_loss_streak"],
        current_streak=streaks["current_streak"],
        avg_mae_usd=avg_mae_usd,
        avg_mfe_usd=avg_mfe_usd,
        mfe_mae_ratio=mfe_mae_ratio,
        avg_holding_hours=avg_holding,
        avg_winner_hours=avg_winner_hrs,
        avg_loser_hours=avg_loser_hrs,
        avg_run_capture=avg_capture,
        median_run_capture=median_capture,
        run_capture_samples=capture_samples,
        max_drawdown_pct=max_dd_pct,
        max_drawdown_usd=max_dd_usd,
        final_equity=final_equity,
        total_return_pct=total_return_pct,
    )


def print_analytics(analytics: TradeAnalytics, symbol: str = "") -> None:
    """Print formatted analytics report."""
    prefix = f"{symbol}: " if symbol else ""

    print(f"\n{prefix}Trade Analytics")
    print("=" * 60)
    print(
        f"Trades:         {analytics.n_trades} ({analytics.n_winners} W / {analytics.n_losers} L)"
    )
    print(f"Win Rate:       {analytics.win_rate:.1%}")
    print(f"Net P&L:        ${analytics.net_pnl:,.2f}")
    print(f"Total Return:   {analytics.total_return_pct:.2f}%")
    print()
    print("─" * 60)
    print("P&L METRICS")
    print("─" * 60)
    print(f"Avg Trade:      ${analytics.avg_trade:,.2f}")
    print(f"Avg Winner:     ${analytics.avg_winner:,.2f}")
    print(f"Avg Loser:      ${analytics.avg_loser:,.2f}")
    print(f"Profit Factor:  {analytics.profit_factor:.2f}")
    print(f"Expectancy:     ${analytics.expectancy:,.2f}")
    print()
    print("─" * 60)
    print("RISK-ADJUSTED METRICS")
    print("─" * 60)
    print(f"Omega:          {analytics.omega:.3f}")
    print(f"Z-Factor:       {analytics.z_factor:.2f}")
    print(f"Sharpe Ratio:   {analytics.sharpe_ratio:.2f}")
    print(f"Calmar Ratio:   {analytics.calmar_ratio:.2f}")
    print()
    print("─" * 60)
    print("RISK METRICS")
    print("─" * 60)
    print(f"Max Drawdown:   {analytics.max_drawdown_pct:.2f}% (${analytics.max_drawdown_usd:,.0f})")
    print(f"Max Win Streak: {analytics.max_win_streak}")
    print(f"Max Loss Streak:{analytics.max_loss_streak}")
    if analytics.avg_mae_usd > 0 or analytics.avg_mfe_usd > 0:
        print(f"Avg MAE:        ${analytics.avg_mae_usd:,.2f}")
        print(f"Avg MFE:        ${analytics.avg_mfe_usd:,.2f}")
        print(f"MFE/MAE Ratio:  {analytics.mfe_mae_ratio:.2f}")
    if analytics.avg_holding_hours > 0:
        print()
        print("─" * 60)
        print("TIMING METRICS")
        print("─" * 60)
        print(f"Avg Hold Time:  {analytics.avg_holding_hours:.1f}h")
        print(f"Avg Winner:     {analytics.avg_winner_hours:.1f}h")
        print(f"Avg Loser:      {analytics.avg_loser_hours:.1f}h")
    if analytics.run_capture_samples > 0:
        print(f"Run Capture μ:  {analytics.avg_run_capture:.2%}")
        print(f"Run Capture 50%:{analytics.median_run_capture:.2%}")
