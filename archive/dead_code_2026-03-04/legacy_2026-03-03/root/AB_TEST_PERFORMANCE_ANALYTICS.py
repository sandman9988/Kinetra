#!/usr/bin/env python3
"""
A/B Lot Sizing Test — Performance Analytics
============================================

Comprehensive metrics with proper risk-based brick sizing:
  • Locked brick sizes from empirical calibration (4× spread floor)
  • $ risk per brick calculation from tick_size and tick_value_usd
  • Lot sizing normalised to equal risk across instruments
  • Starting equity: $1,000 (matching previous benchmark)
  • Full data backtest (not limited to 10,000 bars)
  • Full parameter logging for repeatability
"""

import json
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

# Canonical data path for XAUUSD M1 data
DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "master_standardized"
    / "ctrader"
    / "peperstone"
    / "metals"
    / "XAUUSD"
    / "XAUUSD_M1_accurate.csv"
)

# Output directory for results
RESULTS_DIR = PROJECT_ROOT / "results" / "ab_test"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ══════════════════════════════════════════════════════════════════════════════
# LOCKED BRICK SIZES (from empirical calibration)
# Source: Live spread analysis + 4× spread floor + DSP validation
# These are PRODUCTION-FROZEN - do not optimise
# ══════════════════════════════════════════════════════════════════════════════
LOCKED_BRICK_SIZES = {
    "XAUUSD": 15.0,  # 4× ~3.5 point spread = 14.0, locked at 15.0
    "XAGUSD": 25.0,  # 4× ~6.1 point spread = 24.4, locked at 25.0
    "NAS100": 9.0,  # 4× ~1.6 point spread = 6.4, locked at 9.0
    "GER40": 9.0,  # 4× ~2.0 point spread = 8.0, locked at 9.0
    "US30": 11.0,  # 4× ~2-3 point spread = 8-12, locked at 11.0
    "UK100": 9.0,  # 4× ~2.0 point spread = 8.0, locked at 9.0
    "JPN225": 30.0,  # 4× ~7.0 point spread = 28.0, locked at 30.0
    "NatGas": 2.0,  # 4× ~0.5 point spread = 2.0
    "Brent": 9.0,  # 4× ~2.0 point spread = 8.0, locked at 9.0
}

# ══════════════════════════════════════════════════════════════════════════════
# CONTRACT SPECS (tick_size, tick_value_usd, contract_size)
# Source: Pepperstone cTrader contract_spec.json
# ══════════════════════════════════════════════════════════════════════════════
CONTRACT_SPECS = {
    "XAUUSD": {
        "tick_size": 0.01,
        "tick_value_usd": 1.00,  # $1 per tick per lot
        "contract_size": 100.0,
        "spread_points": 3.5,  # Live typical
    },
    "XAGUSD": {
        "tick_size": 0.001,
        "tick_value_usd": 5.00,  # $5 per tick per lot
        "contract_size": 5000.0,
        "spread_points": 6.1,  # Live typical
    },
    "NAS100": {
        "tick_size": 0.1,
        "tick_value_usd": 0.10,  # $0.10 per tick per lot
        "contract_size": 10.0,
        "spread_points": 1.6,  # Live typical
    },
    "GER40": {
        "tick_size": 0.1,
        "tick_value_usd": 0.10,  # $0.10 per tick per lot
        "contract_size": 10.0,
        "spread_points": 2.0,  # Live typical
    },
    "US30": {
        "tick_size": 0.1,
        "tick_value_usd": 0.10,  # $0.10 per tick per lot
        "contract_size": 10.0,
        "spread_points": 2.5,  # Live typical
    },
    "UK100": {
        "tick_size": 0.1,
        "tick_value_usd": 0.10,  # $0.10 per tick per lot
        "contract_size": 10.0,
        "spread_points": 2.0,  # Live typical
    },
}

# Calculate $ risk per brick per lot
# $/brick/lot = (brick / tick_size) × tick_value_usd
for symbol, spec in CONTRACT_SPECS.items():
    brick = LOCKED_BRICK_SIZES[symbol]
    spec["usd_per_brick_per_lot"] = (brick / spec["tick_size"]) * spec["tick_value_usd"]

# ══════════════════════════════════════════════════════════════════════════════
# LOT WEIGHTS (risk-normalised, XAUUSD = 1.0)
# w_i ∝ 1 / ($ risk per brick_i)
# Normalised so XAUUSD = 1.0
# ══════════════════════════════════════════════════════════════════════════════
XAUUSD_RISK = CONTRACT_SPECS["XAUUSD"]["usd_per_brick_per_lot"]
LOT_WEIGHTS = {}
for symbol, spec in CONTRACT_SPECS.items():
    risk = spec["usd_per_brick_per_lot"]
    LOT_WEIGHTS[symbol] = XAUUSD_RISK / risk  # Inverse of risk, normalised

# Lot weights (for reference):
# XAUUSD: 1.00 (baseline)
# XAGUSD: 0.012 (brick is $125K/lot - tiny lots needed)
# NAS100: 166.7 (brick is $9/lot - huge lots needed)
# GER40: 166.7
# US30: 136.4
# UK100: 166.7


@dataclass
class FrictionFloor:
    """Friction cost analysis for brick size validation."""

    symbol: str
    spread_points: float
    tick_size: float
    tick_value_usd: float
    contract_size: float
    spread_usd_per_lot: float
    commission_per_lot: float
    total_friction_per_lot: float
    friction_per_point: float
    min_brick_size: float  # 4× spread floor
    locked_brick_size: float
    usd_per_brick_per_lot: float
    brick_to_friction_ratio: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DrawdownAnalysis:
    """Drawdown metrics from equity curve."""

    max_dd_usd: float
    max_dd_pct: float
    max_dd_duration_bars: int
    avg_dd_duration_bars: float
    recovery_factor: float
    calmar_ratio: float
    dd_episodes: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PerformanceStats:
    """Complete performance statistics."""

    symbol: str
    brick_scenario: str
    sizing_mode: str

    # Time period (for repeatability)
    start_date: str
    end_date: str
    n_bars: int
    n_bricks: int
    years: float

    # Brick and friction
    brick_size: float
    brick_to_friction_ratio: float
    usd_per_brick_per_lot: float

    # Trade counts
    total_trades: int
    winning_trades: int
    losing_trades: int
    breakeven_trades: int

    # Win/Loss metrics
    win_rate: float
    loss_rate: float
    win_loss_ratio: float

    # P&L metrics
    total_gross_pnl: float
    total_friction: float
    total_net_pnl: float
    avg_win: float
    avg_loss: float
    avg_net: float
    median_trade: float

    # Risk metrics
    profit_factor: float
    omega: float
    z_factor: float
    max_dd_usd: float

    # Drawdown analysis
    drawdown: DrawdownAnalysis

    # Excursion metrics
    avg_mae_pct: float
    avg_mfe_pct: float
    mfe_captured_pct: float

    # Streak metrics
    longest_win_streak: int
    longest_loss_streak: int
    avg_win_streak: float
    avg_loss_streak: float

    # Trade frequency
    trades_per_year: float
    avg_holding_bars: float

    # Initial equity
    initial_equity: float
    final_equity: float
    total_return_pct: float

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["drawdown"] = self.drawdown.to_dict()
        return d


def compute_drawdown_analysis(
    equity_curve: List[float], total_return: float, years: float
) -> DrawdownAnalysis:
    """Compute comprehensive drawdown metrics from equity curve."""
    equity = np.array(equity_curve)

    if len(equity) < 2:
        return DrawdownAnalysis(
            max_dd_usd=0.0,
            max_dd_pct=0.0,
            max_dd_duration_bars=0,
            avg_dd_duration_bars=0.0,
            recovery_factor=0.0,
            calmar_ratio=0.0,
            dd_episodes=0,
        )

    running_max = np.maximum.accumulate(equity)
    dd = equity - running_max
    dd_pct = np.where(running_max > 0, dd / running_max, 0)

    max_dd_usd = float(dd.min())
    max_dd_pct = float(dd_pct.min())

    # Find DD episodes
    in_dd = dd < 0
    dd_episodes = 0
    dd_durations = []
    current_duration = 0

    for i in range(len(in_dd)):
        if in_dd[i]:
            current_duration += 1
        else:
            if current_duration > 0:
                dd_durations.append(current_duration)
                current_duration = 0

    if current_duration > 0:
        dd_durations.append(current_duration)

    dd_episodes = len(dd_durations)
    max_dd_duration = max(dd_durations) if dd_durations else 0
    avg_dd_duration = np.mean(dd_durations) if dd_durations else 0.0

    # Recovery factor
    recovery_factor = abs(total_return / max_dd_usd) if max_dd_usd != 0 else 0.0

    # Calmar ratio
    annual_return = total_return / years if years > 0 else 0
    calmar_ratio = abs(annual_return / max_dd_pct) if max_dd_pct != 0 else 0.0

    return DrawdownAnalysis(
        max_dd_usd=max_dd_usd,
        max_dd_pct=max_dd_pct,
        max_dd_duration_bars=max_dd_duration,
        avg_dd_duration_bars=float(avg_dd_duration),
        recovery_factor=recovery_factor,
        calmar_ratio=calmar_ratio,
        dd_episodes=dd_episodes,
    )


def compute_performance_stats(
    trades: List,
    equity_curve: List[float],
    omega: float,
    z_factor: float,
    max_dd_usd: float,
    brick_scenario: str,
    sizing_mode: str,
    brick_size: float,
    brick_to_friction_ratio: float,
    usd_per_brick_per_lot: float,
    start_date: str,
    end_date: str,
    n_bars: int,
    n_bricks: int,
    years: float,
    initial_equity: float = 1000.0,
    symbol: str = "XAUUSD",
) -> PerformanceStats:
    """Compute comprehensive performance statistics from trades."""

    final_equity = equity_curve[-1] if equity_curve else initial_equity
    total_return = final_equity - initial_equity
    total_return_pct = (total_return / initial_equity) * 100 if initial_equity > 0 else 0

    if not trades:
        return PerformanceStats(
            symbol=symbol,
            brick_scenario=brick_scenario,
            sizing_mode=sizing_mode,
            start_date=start_date,
            end_date=end_date,
            n_bars=n_bars,
            n_bricks=n_bricks,
            years=years,
            brick_size=brick_size,
            brick_to_friction_ratio=brick_to_friction_ratio,
            usd_per_brick_per_lot=usd_per_brick_per_lot,
            total_trades=0,
            winning_trades=0,
            losing_trades=0,
            breakeven_trades=0,
            win_rate=0.0,
            loss_rate=0.0,
            win_loss_ratio=0.0,
            total_gross_pnl=0.0,
            total_friction=0.0,
            total_net_pnl=0.0,
            avg_win=0.0,
            avg_loss=0.0,
            avg_net=0.0,
            median_trade=0.0,
            profit_factor=0.0,
            omega=omega,
            z_factor=z_factor,
            max_dd_usd=max_dd_usd,
            drawdown=compute_drawdown_analysis(equity_curve, 0, years),
            avg_mae_pct=0.0,
            avg_mfe_pct=0.0,
            mfe_captured_pct=0.0,
            longest_win_streak=0,
            longest_loss_streak=0,
            avg_win_streak=0.0,
            avg_loss_streak=0.0,
            trades_per_year=0.0,
            avg_holding_bars=0.0,
            initial_equity=initial_equity,
            final_equity=final_equity,
            total_return_pct=total_return_pct,
        )

    # Count outcomes
    net_pnls = [t.net_usd for t in trades]
    winners = [p for p in net_pnls if p > 0]
    losers = [p for p in net_pnls if p < 0]
    breakeven = [p for p in net_pnls if p == 0]

    total_trades = len(trades)
    winning_trades = len(winners)
    losing_trades = len(losers)
    breakeven_trades = len(breakeven)

    win_rate = winning_trades / total_trades if total_trades > 0 else 0.0
    loss_rate = losing_trades / total_trades if total_trades > 0 else 0.0
    win_loss_ratio = (
        winning_trades / losing_trades
        if losing_trades > 0
        else (winning_trades if winning_trades > 0 else 0.0)
    )

    # P&L metrics
    total_gross = sum(t.gross_usd for t in trades)
    total_friction = sum(t.friction_usd for t in trades)
    total_net = sum(net_pnls)

    avg_win = sum(winners) / len(winners) if winners else 0.0
    avg_loss = sum(losers) / len(losers) if losers else 0.0
    avg_net = total_net / total_trades if total_trades > 0 else 0.0
    median_trade = float(np.median(net_pnls))

    # Profit factor
    sum_wins = sum(p for p in net_pnls if p > 0)
    sum_losses = abs(sum(p for p in net_pnls if p < 0))
    profit_factor = sum_wins / sum_losses if sum_losses > 0 else (10.0 if sum_wins > 0 else 0.0)

    # MAE/MFE
    mae_values = []
    mfe_values = []
    for t in trades:
        if abs(t.gross_usd) > 0:
            mae_pct = t.friction_usd / abs(t.gross_usd)
            mae_values.append(mae_pct)
            mfe_pct = abs(t.net_usd) / abs(t.gross_usd)
            mfe_values.append(mfe_pct)

    avg_mae_pct = np.mean(mae_values) if mae_values else 0.0
    avg_mfe_pct = np.mean(mfe_values) if mfe_values else 0.0
    mfe_captured = sum(mfe_values) / len(mfe_values) if mfe_values else 0.0

    # Streak analysis
    streaks = []
    current_streak = 1
    current_is_win = net_pnls[0] > 0

    for i in range(1, len(net_pnls)):
        is_win = net_pnls[i] > 0
        if is_win == current_is_win:
            current_streak += 1
        else:
            streaks.append((current_is_win, current_streak))
            current_is_win = is_win
            current_streak = 1

    streaks.append((current_is_win, current_streak))

    win_streaks = [s for is_win, s in streaks if is_win]
    loss_streaks = [s for is_win, s in streaks if not is_win]

    longest_win_streak = max(win_streaks) if win_streaks else 0
    longest_loss_streak = max(loss_streaks) if loss_streaks else 0
    avg_win_streak = np.mean(win_streaks) if win_streaks else 0.0
    avg_loss_streak = np.mean(loss_streaks) if loss_streaks else 0.0

    # Trade frequency
    trades_per_year = len(trades) / max(years, 0.001)
    avg_holding_bars = np.mean([t.n_bricks_held for t in trades]) if trades else 0.0

    # Drawdown analysis
    drawdown = compute_drawdown_analysis(equity_curve, total_net, years)

    return PerformanceStats(
        symbol=symbol,
        brick_scenario=brick_scenario,
        sizing_mode=sizing_mode,
        start_date=start_date,
        end_date=end_date,
        n_bars=n_bars,
        n_bricks=n_bricks,
        years=years,
        brick_size=brick_size,
        brick_to_friction_ratio=brick_to_friction_ratio,
        usd_per_brick_per_lot=usd_per_brick_per_lot,
        total_trades=total_trades,
        winning_trades=winning_trades,
        losing_trades=losing_trades,
        breakeven_trades=breakeven_trades,
        win_rate=win_rate,
        loss_rate=loss_rate,
        win_loss_ratio=win_loss_ratio,
        total_gross_pnl=total_gross,
        total_friction=total_friction,
        total_net_pnl=total_net,
        avg_win=avg_win,
        avg_loss=avg_loss,
        avg_net=avg_net,
        median_trade=median_trade,
        profit_factor=profit_factor,
        omega=omega,
        z_factor=z_factor,
        max_dd_usd=max_dd_usd,
        drawdown=drawdown,
        avg_mae_pct=avg_mae_pct,
        avg_mfe_pct=avg_mfe_pct,
        mfe_captured_pct=mfe_captured,
        longest_win_streak=longest_win_streak,
        longest_loss_streak=longest_loss_streak,
        avg_win_streak=avg_win_streak,
        avg_loss_streak=avg_loss_streak,
        trades_per_year=trades_per_year,
    )
