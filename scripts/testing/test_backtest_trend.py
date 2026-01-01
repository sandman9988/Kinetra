#!/usr/bin/env python3
"""
Incremental Backtest Test - T3 MA Crossover + ADX + Chandelier Exit Strategy

Tests the backtest engine with a trend-following strategy:
- T3 Moving Average crossover (Tim Tillson, 1998)
- ADX filter with +DI/-DI directional confirmation
- Chandelier Exit for trailing stops

Usage:
    python scripts/test_backtest_trend.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from kinetra.backtest_engine import BacktestEngine, BacktestResult
from kinetra.symbol_spec import get_symbol_spec
from kinetra.volatility import (
    analyze_energy_distribution,
    energy_efficiency,
    forward_energy_release,
    potential_energy,
)

# =============================================================================
# T3 MOVING AVERAGE (Tim Tillson, 1998)
# =============================================================================


def calculate_t3(data: pd.Series, period: int, v: float = 0.7) -> pd.Series:
    """
    Calculate T3 Moving Average (Tim Tillson, 1998).

    T3 is a six-stage EMA cascade that reduces lag while maintaining smoothness.
    Each stage uses the SAME period.

    Args:
        data: Price series
        period: EMA period for each of the 6 stages (e.g., 10 for fast, 30 for slow)
        v: Volume factor (0.7-0.9 typical, 0.7 is standard)
    """

    def ema(src: pd.Series, length: int) -> pd.Series:
        return src.ewm(span=length, adjust=False).mean()

    e1 = ema(data, period)
    e2 = ema(e1, period)
    e3 = ema(e2, period)
    e4 = ema(e3, period)
    e5 = ema(e4, period)
    e6 = ema(e5, period)

    c1 = -v * v * v
    c2 = 3 * v * v + 3 * v * v * v
    c3 = -6 * v * v - 3 * v - 3 * v * v * v
    c4 = 1 + 3 * v + v * v * v + 3 * v * v

    t3 = c1 * e6 + c2 * e5 + c3 * e4 + c4 * e3
    return t3


# =============================================================================
# ADX WITH +DI/-DI DIRECTIONAL INDICATORS
# =============================================================================


def calculate_adx(
    high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    Calculate ADX, +DI, -DI.

    ADX > 20-25 indicates a trending market.
    +DI > -DI indicates bullish direction.
    -DI > +DI indicates bearish direction.

    Returns:
        (adx, plus_di, minus_di)
    """
    up_move = high.diff()
    down_move = low.diff() * -1

    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0)

    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    atr = tr.rolling(window=period).mean()

    plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)

    dx = (plus_di - minus_di).abs() / (plus_di + minus_di) * 100
    adx = dx.rolling(window=period).mean()

    return adx, plus_di, minus_di


# =============================================================================
# CHANDELIER EXIT
# =============================================================================


def calculate_chandelier_exit(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    atr_period: int = 14,
    multiplier: float = 3.0,
) -> tuple[pd.Series, pd.Series]:
    """
    Calculate Chandelier Exit for both long and short positions.

    Returns:
        (long_stop, short_stop)
    """
    tr = np.maximum(
        high - low,
        np.maximum(abs(high - close.shift(1)), abs(low - close.shift(1))),
    )
    atr = pd.Series(tr, index=close.index).rolling(atr_period).mean()

    long_stop = high.rolling(atr_period).max() - multiplier * atr
    short_stop = low.rolling(atr_period).min() + multiplier * atr

    return long_stop, short_stop


# =============================================================================
# T3 + ADX + CHANDELIER STRATEGY
# =============================================================================


def apply_strategy(
    df: pd.DataFrame,
    t3_fast: int = 10,
    t3_slow: int = 30,
    t3_v: float = 0.7,
    adx_period: int = 14,
    adx_threshold: float = 20.0,
    atr_period: int = 14,
    atr_multiplier: float = 3.0,
    spread_gate: float | None = None,  # Only enter if spread <= X * median spread
) -> pd.DataFrame:
    """
    Apply T3 + ADX + Chandelier Exit strategy.

    Entry Rules:
    - Long: Fast T3 crosses above Slow T3 AND ADX > threshold AND +DI > -DI
    - Short: Fast T3 crosses below Slow T3 AND ADX > threshold AND -DI > +DI

    Exit Rules:
    - Long: Price falls below Chandelier long stop
    - Short: Price rises above Chandelier short stop

    Returns:
        DataFrame with signals, indicators, and position state
    """
    df = df.copy()

    # Calculate T3 MAs
    df["T3_fast"] = calculate_t3(df["close"], t3_fast, t3_v)
    df["T3_slow"] = calculate_t3(df["close"], t3_slow, t3_v)

    # Calculate ADX with directional indicators
    df["ADX"], df["DI+"], df["DI-"] = calculate_adx(df["high"], df["low"], df["close"], adx_period)

    # Calculate Chandelier Exit
    df["chandelier_long"], df["chandelier_short"] = calculate_chandelier_exit(
        df["high"], df["low"], df["close"], atr_period, atr_multiplier
    )

    # Crossover detection
    df["fast_above_slow"] = df["T3_fast"] > df["T3_slow"]
    df["cross_above"] = df["fast_above_slow"] & (~df["fast_above_slow"].shift(1).fillna(False))
    df["cross_below"] = (~df["fast_above_slow"]) & (df["fast_above_slow"].shift(1).fillna(False))

    # ADX filter with directional confirmation
    df["strong_trend"] = df["ADX"] > adx_threshold
    df["bullish"] = df["DI+"] > df["DI-"]
    df["bearish"] = df["DI-"] > df["DI+"]

    # Spread gate filter (only enter when spread is reasonable)
    # Use MIN spread as baseline - represents best execution conditions
    if spread_gate is not None and "spread" in df.columns:
        min_spread = df["spread"].min()
        # Handle case where min is 0 (use 1 as floor)
        min_spread = max(min_spread, 1.0)
        max_allowed_spread = min_spread * spread_gate
        df["spread_ok"] = df["spread"] <= max_allowed_spread
    else:
        df["spread_ok"] = True

    # Entry conditions (ADX + DI confirmation + spread gate)
    df["long_entry"] = df["cross_above"] & df["strong_trend"] & df["bullish"] & df["spread_ok"]
    df["short_entry"] = df["cross_below"] & df["strong_trend"] & df["bearish"] & df["spread_ok"]

    # Initialize position tracking
    df["position"] = 0  # -1: short, 0: flat, 1: long
    df["exit_signal"] = False

    in_long = False
    in_short = False

    for i in range(1, len(df)):
        # Handle entries (only if flat)
        if df["long_entry"].iloc[i] and not in_long and not in_short:
            in_long = True
            df.iloc[i, df.columns.get_loc("position")] = 1
        elif df["short_entry"].iloc[i] and not in_short and not in_long:
            in_short = True
            df.iloc[i, df.columns.get_loc("position")] = -1

        # Handle exits via Chandelier
        if in_long:
            df.iloc[i, df.columns.get_loc("position")] = 1
            if df["low"].iloc[i] < df["chandelier_long"].iloc[i]:
                in_long = False
                df.iloc[i, df.columns.get_loc("exit_signal")] = True
                df.iloc[i, df.columns.get_loc("position")] = 0
        elif in_short:
            df.iloc[i, df.columns.get_loc("position")] = -1
            if df["high"].iloc[i] > df["chandelier_short"].iloc[i]:
                in_short = False
                df.iloc[i, df.columns.get_loc("exit_signal")] = True
                df.iloc[i, df.columns.get_loc("position")] = 0

    return df


class TrendStrategy:
    """
    T3 + ADX + Chandelier Exit strategy wrapper for BacktestEngine.

    Converts pre-computed signals to backtest engine format.
    """

    def __init__(self, strategy_df: pd.DataFrame):
        self.df = strategy_df
        self.prev_position = 0

    def get_signal(
        self,
        row: pd.Series,
        physics_state: pd.DataFrame,
        bar_index: int,
    ) -> int:
        """
        Generate trading signal for backtest engine.

        Returns: 1 (long), -1 (short), 0 (flat/exit)
        """
        if bar_index >= len(self.df):
            return 0

        current_position = int(self.df["position"].iloc[bar_index])
        exit_signal = bool(self.df["exit_signal"].iloc[bar_index])

        # Check for new entry
        if self.prev_position == 0 and current_position != 0:
            self.prev_position = current_position
            return current_position  # 1 for long, -1 for short

        # Check for exit
        if exit_signal:
            exit_dir = -self.prev_position  # Opposite direction to close
            self.prev_position = 0
            return exit_dir

        # No change
        self.prev_position = current_position
        return 0


# =============================================================================
# DATA LOADING
# =============================================================================


def load_data(filepath: str) -> pd.DataFrame:
    """Load MT5 exported CSV data."""
    df = pd.read_csv(
        filepath,
        sep="\t",
        parse_dates={"datetime": ["<DATE>", "<TIME>"]},
    )

    # Rename columns to standard format
    df = df.rename(
        columns={
            "<OPEN>": "open",
            "<HIGH>": "high",
            "<LOW>": "low",
            "<CLOSE>": "close",
            "<TICKVOL>": "volume",
            "<VOL>": "real_volume",
            "<SPREAD>": "spread",
        }
    )

    # Set datetime as index
    df = df.set_index("datetime")

    return df


# =============================================================================
# RESULT DISPLAY
# =============================================================================


def print_result(result: BacktestResult, symbol: str, timeframe: str) -> None:
    """Print backtest results in a formatted way."""
    print("\n" + "=" * 60)
    print(f"BACKTEST RESULTS: {symbol} {timeframe} - T3 + ADX + Chandelier")
    print("=" * 60)

    print(f"\nTrade Statistics:")
    print(f"  Total Trades:    {result.total_trades}")
    print(f"  Winning Trades:  {result.winning_trades}")
    print(f"  Losing Trades:   {result.losing_trades}")
    print(f"  Win Rate:        {result.win_rate:.1%}")

    print(f"\nP&L Summary:")
    print(f"  Gross Profit:    ${result.gross_profit:,.2f}")
    print(f"  Gross Loss:      ${result.gross_loss:,.2f}")
    print(f"  Total Gross P&L: ${result.total_gross_pnl:,.2f}")
    print(f"  Total Costs:     ${result.total_costs:,.2f}")
    print(f"  Net P&L:         ${result.total_net_pnl:,.2f}")

    print(f"\nCost Breakdown:")
    print(f"  Spread Cost:     ${result.total_spread_cost:,.2f}")
    print(f"  Commission:      ${result.total_commission:,.2f}")
    print(f"  Slippage:        ${result.total_slippage:,.2f}")
    print(f"  Swap Cost:       ${result.total_swap_cost:,.2f}")

    print(f"\nRisk Metrics:")
    print(f"  Max Drawdown:    ${result.max_drawdown:,.2f} ({result.max_drawdown_pct:.1f}%)")
    print(f"  Sharpe Ratio:    {result.sharpe_ratio:.2f}")
    print(f"  Sortino Ratio:   {result.sortino_ratio:.2f}")
    print(f"  Omega Ratio:     {result.omega_ratio:.2f}")
    print(f"  Z-Factor:        {result.z_factor:.2f}")

    print(f"\nPhysics Metrics:")
    print(f"  Energy Captured: {result.energy_captured_pct:.1%}")
    print(f"  MFE Capture:     {result.mfe_capture_pct:.1%}")

    print("=" * 60)


# =============================================================================
# MAIN TEST
# =============================================================================


def main():
    """Run incremental backtest test."""

    # Configuration - T3 crossover per Tillson specification
    symbol = "XAUUSD+"
    timeframe = "H1"
    t3_fast = 10  # Fast T3 period
    t3_slow = 30  # Slow T3 period
    t3_v = 0.7  # T3 volume factor (standard)
    adx_period = 14  # ADX period
    adx_threshold = 20.0  # Only trade when ADX > 20 (trending)
    atr_period = 14  # ATR period for Chandelier Exit
    atr_multiplier = 2.0  # Tighter Chandelier (was 3.0) - reduces holding time
    initial_capital = 10000.0
    fixed_lots = 0.1  # Fixed position size
    spread_gate = 7.0  # Only enter if spread <= 7x min spread (~normal spread)

    # Find data file
    data_dir = Path(__file__).parent.parent / "data" / "master"
    data_files = list(data_dir.glob(f"{symbol}_{timeframe}_*.csv"))

    if not data_files:
        print(f"No data found for {symbol} {timeframe}")
        print(f"Available files: {list(data_dir.glob('*.csv'))[:5]}")
        return

    data_file = data_files[0]
    print(f"Loading data from: {data_file}")

    # Load data
    data = load_data(str(data_file))
    print(f"Loaded {len(data)} bars from {data.index[0]} to {data.index[-1]}")

    # Get symbol specification
    try:
        symbol_spec = get_symbol_spec(symbol.replace("+", ""))  # Remove + suffix
    except KeyError:
        print(f"Symbol spec not found for {symbol}, using XAUUSD default")
        symbol_spec = get_symbol_spec("XAUUSD")

    # Update symbol spec with actual spread from data
    actual_spread = (
        data["spread"].median() if "spread" in data.columns else symbol_spec.spread_points
    )
    symbol_spec.spread_points = int(actual_spread)
    symbol_spec.spread_min = (
        int(data["spread"].min()) if "spread" in data.columns else symbol_spec.spread_min
    )
    symbol_spec.spread_max = (
        int(data["spread"].max()) if "spread" in data.columns else symbol_spec.spread_max
    )

    print(f"\nSymbol Spec: {symbol_spec.symbol}")
    print(f"  Spread: {symbol_spec.spread_points} points (from data, median)")
    print(f"  Spread Range: {symbol_spec.spread_min} - {symbol_spec.spread_max} points")
    print(f"  Tick Size: {symbol_spec.tick_size}")
    print(f"  Tick Value: ${symbol_spec.tick_value}")

    # Print spread distribution
    print_spread_distribution(data)

    # Print potential energy distribution
    print_energy_distribution(data)

    # Apply strategy
    print(f"\nApplying T3 + ADX + Chandelier strategy...")
    print(f"  T3 Fast: {t3_fast}, T3 Slow: {t3_slow}, v: {t3_v}")
    print(f"  ADX Period: {adx_period}, Threshold: {adx_threshold}")
    print(f"  Chandelier: ATR({atr_period}) x {atr_multiplier}")
    print(f"  Spread Gate: {spread_gate}x min spread")

    strategy_df = apply_strategy(
        data,
        t3_fast=t3_fast,
        t3_slow=t3_slow,
        t3_v=t3_v,
        adx_period=adx_period,
        adx_threshold=adx_threshold,
        atr_period=atr_period,
        atr_multiplier=atr_multiplier,
        spread_gate=spread_gate,
    )

    # Count signals
    long_entries = strategy_df["long_entry"].sum()
    short_entries = strategy_df["short_entry"].sum()
    exit_signals = strategy_df["exit_signal"].sum()
    print(
        f"Generated {long_entries} long entries, {short_entries} short entries, {exit_signals} exits"
    )

    # Create strategy wrapper
    strategy = TrendStrategy(strategy_df)

    # Run backtest
    print(f"\nRunning backtest with ${initial_capital:,.0f} initial capital...")
    print(f"  Fixed lot size: {fixed_lots}")

    engine = BacktestEngine(
        initial_capital=initial_capital,
        risk_per_trade=0.02,  # 2% risk
        max_positions=1,
        use_physics_signals=False,  # Use our custom signals
        data_quality_check=False,  # Skip for now
    )

    # Override position sizing to use fixed lots
    original_open_position = engine._open_position

    def fixed_lot_open_position(
        symbol,
        direction,
        price,
        time,
        spec,
        energy,
        regime,
        stop_distance=None,
        atr_value=None,
        pe_at_entry=0.0,
        forward_release_5=0.0,
        forward_release_10=0.0,
    ):
        """Override to use fixed lot size."""
        trade = original_open_position(
            symbol,
            direction,
            price,
            time,
            spec,
            energy,
            regime,
            stop_distance=stop_distance,
            atr_value=atr_value,
            pe_at_entry=pe_at_entry,
            forward_release_5=forward_release_5,
            forward_release_10=forward_release_10,
        )
        # Replace with fixed lots
        trade.lots = fixed_lots
        # Recalculate costs with fixed lots (full costs on entry)
        trade.spread_cost = spec.spread_cost(fixed_lots, price)
        trade.commission = spec.commission.calculate_commission(
            fixed_lots, fixed_lots * spec.contract_size * price
        )
        trade.slippage = spec.slippage_avg * spec.tick_value * fixed_lots
        return trade

    engine._open_position = fixed_lot_open_position

    result = engine.run_backtest(
        data=strategy_df,
        symbol_spec=symbol_spec,
        signal_func=strategy.get_signal,
    )

    # Print results
    print_result(result, symbol, timeframe)

    # Print full trade log (last 20 trades) with efficiency metrics
    result.print_trade_log(last_n=20, detailed=True)

    # Holding time analysis
    if result.trades:
        holding_hours = [t.holding_hours for t in result.trades]
        print(f"\nHolding Time Analysis:")
        print(f"  Mean:   {sum(holding_hours) / len(holding_hours):.1f} hours")
        print(f"  Min:    {min(holding_hours):.1f} hours")
        print(f"  Max:    {max(holding_hours):.1f} hours")
        overnight = sum(1 for h in holding_hours if h > 24)
        print(
            f"  Overnight (>24h): {overnight} trades ({100 * overnight / len(holding_hours):.1f}%)"
        )

    # Export trade log to CSV
    trade_log_df = result.trade_log()
    if not trade_log_df.empty:
        log_path = Path(__file__).parent / "trade_log.csv"
        trade_log_df.to_csv(log_path, index=False)
        print(f"\nTrade log exported to: {log_path}")

    # Show detailed lifecycle for a winning trade with many TS adjustments
    if result.trades:
        # Find trade with most trailing stop adjustments
        best_trade = max(result.trades, key=lambda t: len(t.stop_adjustments))
        if best_trade.stop_adjustments:
            result.print_trade_lifecycle(best_trade.trade_id)

    # Show signal quality analysis
    print("\n" + "=" * 60)
    print("SIGNAL QUALITY ANALYSIS")
    print("=" * 60)

    # Analyze ADX at entry
    entry_bars = strategy_df[strategy_df["long_entry"] | strategy_df["short_entry"]]
    if len(entry_bars) > 0:
        print(f"\nADX at entries:")
        print(f"  Mean:   {entry_bars['ADX'].mean():.1f}")
        print(f"  Min:    {entry_bars['ADX'].min():.1f}")
        print(f"  Max:    {entry_bars['ADX'].max():.1f}")

        # DI spread at entries
        di_spread = (entry_bars["DI+"] - entry_bars["DI-"]).abs()
        print(f"\nDI Spread at entries (|+DI - -DI|):")
        print(f"  Mean:   {di_spread.mean():.1f}")
        print(f"  Min:    {di_spread.min():.1f}")
        print(f"  Max:    {di_spread.max():.1f}")

    print("=" * 60)

    # PE trade analysis (Pareto/80-20)
    print_pe_trade_analysis(result)


def print_energy_distribution(data: pd.DataFrame) -> None:
    """Print potential energy distribution analysis."""
    print("\n" + "=" * 80)
    print("POTENTIAL ENERGY DISTRIBUTION ANALYSIS")
    print("=" * 80)

    # Calculate PE
    pe = potential_energy(data, period=14).dropna()
    fer = forward_energy_release(data, forward_bars=5).dropna()

    # Align series
    common_idx = pe.index.intersection(fer.index)
    pe = pe.loc[common_idx]
    fer = fer.loc[common_idx]

    print(f"\n### POTENTIAL ENERGY (Compression/Squeeze) ###")
    print(f"Count:     {len(pe):,}")
    print(f"Mean:      {pe.mean():.4f}")
    print(f"Std Dev:   {pe.std():.4f}")
    print(f"Min:       {pe.min():.4f}")
    print(f"Max:       {pe.max():.4f}")

    print(f"\nPercentiles:")
    for p in [25, 50, 75, 90, 95, 99]:
        print(f"  P{p}:       {pe.quantile(p / 100):.4f}")

    # Energy release analysis
    print(f"\n### FORWARD ENERGY RELEASE (Next 5 bars) ###")
    print(f"Mean Move:   {fer.mean() * 100:.3f}%")
    print(f"Median Move: {fer.median() * 100:.3f}%")
    print(f"Max Move:    {fer.max() * 100:.3f}%")

    # Efficiency: PE vs Actual Release
    print(f"\n### ENERGY EFFICIENCY (Release / PE) ###")
    efficiency = fer / pe.clip(lower=0.01)
    print(f"Mean Efficiency:   {efficiency.mean():.2f}x")
    print(f"Median Efficiency: {efficiency.median():.2f}x")

    # High PE conditions
    high_pe_threshold = pe.quantile(0.75)
    extreme_pe_threshold = pe.quantile(0.90)

    print(f"\n### HIGH PE CONDITIONS (Squeeze Detection) ###")
    print(f"High PE Threshold (P75):    {high_pe_threshold:.4f}")
    print(f"Extreme PE Threshold (P90): {extreme_pe_threshold:.4f}")

    high_pe_bars = pe[pe > high_pe_threshold]
    extreme_pe_bars = pe[pe > extreme_pe_threshold]

    # Forward release after high PE
    high_pe_release = fer.loc[high_pe_bars.index]
    extreme_pe_release = fer.loc[extreme_pe_bars.index]
    normal_release = fer.loc[pe[pe <= high_pe_threshold].index]

    print(f"\nForward Release Comparison:")
    print(f"  Normal PE:  {normal_release.mean() * 100:.3f}% avg move ({len(normal_release)} bars)")
    print(
        f"  High PE:    {high_pe_release.mean() * 100:.3f}% avg move ({len(high_pe_release)} bars)"
    )
    print(
        f"  Extreme PE: {extreme_pe_release.mean() * 100:.3f}% avg move ({len(extreme_pe_bars)} bars)"
    )

    # Lift factor
    if normal_release.mean() > 0:
        high_lift = high_pe_release.mean() / normal_release.mean()
        extreme_lift = extreme_pe_release.mean() / normal_release.mean()
        print(f"\nLift Factors (vs Normal):")
        print(f"  High PE Lift:    {high_lift:.2f}x")
        print(f"  Extreme PE Lift: {extreme_lift:.2f}x")

    # Pareto analysis
    print(f"\n### PARETO ANALYSIS (80/20 Rule) ###")
    # Sort by forward release
    sorted_fer = fer.sort_values(ascending=False)
    cumsum = sorted_fer.cumsum() / sorted_fer.sum()

    # Find what % of bars capture 80% of total movement
    bars_for_80pct = (cumsum <= 0.80).sum()
    pct_bars_for_80 = bars_for_80pct / len(sorted_fer) * 100

    print(f"Top {pct_bars_for_80:.1f}% of bars capture 80% of total energy release")
    print(f"That's {bars_for_80pct} out of {len(sorted_fer)} bars")

    # What PE level captures top movers?
    top_20pct_idx = sorted_fer.head(int(len(sorted_fer) * 0.20)).index
    pe_of_top_movers = pe.loc[top_20pct_idx]
    print(f"\nPE of Top 20% Movers:")
    print(f"  Mean PE:   {pe_of_top_movers.mean():.4f}")
    print(f"  Median PE: {pe_of_top_movers.median():.4f}")
    print(f"  % with High PE (>P75): {(pe_of_top_movers > high_pe_threshold).mean() * 100:.1f}%")

    print("=" * 80)


def print_pe_trade_analysis(result: BacktestResult) -> None:
    """Analyze PE metrics across trades - Pareto analysis for ML/RL guidance."""
    if not result.trades:
        print("No trades to analyze")
        return

    print("\n" + "=" * 80)
    print("POTENTIAL ENERGY TRADE ANALYSIS (Pareto/80-20)")
    print("=" * 80)

    # Extract PE metrics from trades
    trades = result.trades
    pe_data = []
    for t in trades:
        pe_data.append(
            {
                "trade_id": t.trade_id,
                "pe_at_entry": t.pe_at_entry,
                "forward_release_5": t.forward_release_5,
                "net_pnl": t.net_pnl,
                "gross_pnl": t.gross_pnl,
                "mfe": t.mfe,
                "mfe_efficiency": t.mfe_efficiency,
                "direction": t.direction.value,
            }
        )

    df = pd.DataFrame(pe_data)

    # Basic PE stats at entry
    print(f"\n### PE AT TRADE ENTRIES ###")
    print(f"Mean PE:      {df['pe_at_entry'].mean():.4f}")
    print(f"Median PE:    {df['pe_at_entry'].median():.4f}")
    print(f"Max PE:       {df['pe_at_entry'].max():.4f}")

    # Segment trades by PE level
    high_pe_threshold = 0.4  # Top quartile from distribution
    extreme_pe_threshold = 0.6  # Top decile

    low_pe = df[df["pe_at_entry"] <= high_pe_threshold]
    high_pe = df[
        (df["pe_at_entry"] > high_pe_threshold) & (df["pe_at_entry"] <= extreme_pe_threshold)
    ]
    extreme_pe = df[df["pe_at_entry"] > extreme_pe_threshold]

    print(f"\n### PE SEGMENTATION ###")
    print(f"Low PE (<=0.4):      {len(low_pe)} trades ({len(low_pe) / len(df) * 100:.1f}%)")
    print(f"High PE (0.4-0.6):   {len(high_pe)} trades ({len(high_pe) / len(df) * 100:.1f}%)")
    print(f"Extreme PE (>0.6):   {len(extreme_pe)} trades ({len(extreme_pe) / len(df) * 100:.1f}%)")

    # Performance by PE segment
    print(f"\n### PERFORMANCE BY PE SEGMENT ###")

    def segment_stats(segment, name):
        if len(segment) == 0:
            print(f"\n{name}: No trades")
            return
        winners = segment[segment["net_pnl"] > 0]
        win_rate = len(winners) / len(segment) * 100
        avg_pnl = segment["net_pnl"].mean()
        total_pnl = segment["net_pnl"].sum()
        avg_mfe_eff = segment["mfe_efficiency"].mean() * 100
        print(f"\n{name}:")
        print(f"  Trades:       {len(segment)}")
        print(f"  Win Rate:     {win_rate:.1f}%")
        print(f"  Avg Net P&L:  ${avg_pnl:+.2f}")
        print(f"  Total P&L:    ${total_pnl:+.2f}")
        print(f"  Avg MFE Eff:  {avg_mfe_eff:.1f}%")

    segment_stats(low_pe, "Low PE (<=0.4)")
    segment_stats(high_pe, "High PE (0.4-0.6)")
    segment_stats(extreme_pe, "Extreme PE (>0.6)")

    # Pareto analysis: which % of trades contribute to 80% of profits
    print(f"\n### PARETO ANALYSIS (80/20 Rule) ###")

    # Sort trades by net P&L (descending)
    profitable = df[df["net_pnl"] > 0].sort_values("net_pnl", ascending=False)

    if len(profitable) > 0:
        total_profit = profitable["net_pnl"].sum()
        cumsum = profitable["net_pnl"].cumsum()

        # Find trades that contribute 80% of profits
        trades_for_80pct = (cumsum <= total_profit * 0.80).sum() + 1
        pct_trades = trades_for_80pct / len(df) * 100

        print(f"Total profitable trades: {len(profitable)}")
        print(f"Total profit: ${total_profit:,.2f}")
        print(f"Top {pct_trades:.1f}% of trades capture 80% of profits")
        print(f"That's {trades_for_80pct} out of {len(df)} trades")

        # PE characteristics of top profit contributors
        top_trades = profitable.head(trades_for_80pct)
        print(f"\nPE of Top Profit Contributors:")
        print(f"  Mean PE:   {top_trades['pe_at_entry'].mean():.4f}")
        print(f"  Median PE: {top_trades['pe_at_entry'].median():.4f}")

        # Compare to losers
        losers = df[df["net_pnl"] <= 0]
        if len(losers) > 0:
            print(f"\nPE of Losing Trades:")
            print(f"  Mean PE:   {losers['pe_at_entry'].mean():.4f}")
            print(f"  Median PE: {losers['pe_at_entry'].median():.4f}")

    # Forward release vs actual capture
    print(f"\n### FORWARD RELEASE vs ACTUAL CAPTURE ###")
    df["forward_5_pct"] = df["forward_release_5"] * 100
    print(f"Avg Forward Release (5 bars): {df['forward_5_pct'].mean():.3f}%")
    print(f"Avg MFE Captured:             {df['mfe'].mean():.2f} pts")

    # Correlation analysis
    if len(df) > 5:
        pe_pnl_corr = df["pe_at_entry"].corr(df["net_pnl"])
        pe_mfe_corr = df["pe_at_entry"].corr(df["mfe"])
        forward_pnl_corr = df["forward_release_5"].corr(df["net_pnl"])

        print(f"\n### CORRELATIONS ###")
        print(f"PE vs Net P&L:       {pe_pnl_corr:+.3f}")
        print(f"PE vs MFE:           {pe_mfe_corr:+.3f}")
        print(f"Forward vs Net P&L:  {forward_pnl_corr:+.3f}")

    # NOTE: No "guidance" provided - this strategy is a test harness only.
    # ML/RL agents will discover optimal patterns through empirical exploration,
    # not from correlations derived from a broken momentum strategy.
    print(f"\n### NOTE ###")
    print(f"  This T3+ADX strategy is a TEST HARNESS for engine validation.")
    print(f"  Correlations above reflect THIS strategy's behavior, not market truth.")
    print(f"  ML/RL agents will derive guidance from empirical testing - not from here.")
    print("=" * 80)


def print_spread_distribution(data: pd.DataFrame) -> None:
    """Print spread distribution analysis."""
    if "spread" not in data.columns:
        print("No spread data available")
        return

    spread = data["spread"]

    print("\n" + "=" * 60)
    print("SPREAD DISTRIBUTION ANALYSIS")
    print("=" * 60)

    print(f"\nBasic Statistics:")
    print(f"  Count:     {len(spread):,}")
    print(f"  Mean:      {spread.mean():.2f} points")
    print(f"  Median:    {spread.median():.2f} points")
    print(f"  Std Dev:   {spread.std():.2f} points")
    print(f"  Min:       {spread.min():.0f} points")
    print(f"  Max:       {spread.max():.0f} points")

    print(f"\nPercentiles:")
    for p in [25, 50, 75, 90, 95, 99]:
        print(f"  P{p}:       {spread.quantile(p / 100):.1f} points")

    # Distribution by multiplier of MIN (best execution baseline)
    min_spread = max(spread.min(), 1.0)  # Floor at 1 if min is 0
    print(f"\nDistribution by Min Multiple ({min_spread:.1f} points = best execution):")
    for mult in [1.0, 2.0, 3.0, 4.0, 5.0, 7.0, 10.0]:
        threshold = min_spread * mult
        pct_below = (spread <= threshold).sum() / len(spread) * 100
        print(f"  <= {mult:.0f}x min ({threshold:.1f} pts): {pct_below:5.1f}% of bars")

    # Spread by hour of day (if datetime index)
    if hasattr(data.index, "hour"):
        print(f"\nMedian Spread by Hour (UTC):")
        hourly = spread.groupby(data.index.hour).median()
        for hour in [0, 4, 8, 12, 16, 20]:
            if hour in hourly.index:
                print(f"  {hour:02d}:00 - {hourly[hour]:.1f} points")

    print("=" * 60)


if __name__ == "__main__":
    main()
