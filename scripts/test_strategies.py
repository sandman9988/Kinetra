#!/usr/bin/env python3
"""
Test Strategies for Backtesting Engine Validation

Three strategy types with simple, low-lag indicators:
1. Trend Following (momentum-based)
2. Mean Reversion (oversold/overbought)
3. Breakout (range expansion)

Uses rolling periods with adaptive thresholds (no magic numbers).
All thresholds are percentile-based, adapting to each instrument's characteristics.

Usage:
    python scripts/test_strategies.py --symbol BTCUSD --data data/master/BTCUSD_H1.csv
    python scripts/test_strategies.py --symbol EURUSD --stress-test
"""

import argparse
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from kinetra.backtest_engine import BacktestEngine, BacktestResult
from kinetra.data_quality import validate_data
from kinetra.market_calendar import get_calendar_for_symbol
from kinetra.physics_engine import PhysicsEngine
from kinetra.stress_test import StressTestEngine, quick_stress_test
from kinetra.symbol_spec import SymbolSpec, get_symbol_spec

# === STRATEGY BASE CLASS ===


@dataclass
class StrategyConfig:
    """Configuration for a test strategy."""

    name: str
    lookback: int = 20
    description: str = ""


class BaseStrategy:
    """Base class for test strategies."""

    def __init__(self, config: StrategyConfig):
        self.config = config
        self.name = config.name
        self.lookback = config.lookback

    def signal(self, row: pd.Series, physics: pd.DataFrame, bar_idx: int) -> int:
        """
        Generate trading signal.

        Args:
            row: Current bar OHLCV data
            physics: Physics state DataFrame
            bar_idx: Current bar index

        Returns:
            1 = buy, -1 = sell, 0 = hold
        """
        raise NotImplementedError

    def get_signal_func(self) -> Callable:
        """Return signal function for backtest engine."""
        return self.signal


# === TREND FOLLOWING STRATEGY ===


class TrendStrategy(BaseStrategy):
    """
    Simple momentum strategy using physics-based flow detection.

    Entry: Energy increasing + Damping low + price momentum aligned
    Exit: Energy decreasing or Damping increasing (flow becoming turbulent)

    Uses rolling percentiles, NOT fixed thresholds.
    No magic numbers - all adaptive to the instrument.
    """

    def __init__(self, lookback: int = 20):
        super().__init__(
            StrategyConfig(
                name="Trend Following",
                lookback=lookback,
                description="Physics-based momentum with laminar flow detection",
            )
        )

    def signal(self, row: pd.Series, physics: pd.DataFrame, bar_idx: int) -> int:
        if bar_idx < self.lookback or bar_idx >= len(physics):
            return 0

        # Get physics state at current bar
        physics_row = physics.iloc[bar_idx]

        # Adaptive thresholds via percentiles
        energy_pct = physics_row.get("energy_pct", 0.5)
        damping_pct = physics_row.get("damping_pct", 0.5)

        # Price velocity (momentum direction)
        # Compute simple momentum from close prices
        price_momentum = (
            row["close"]
            - physics.get("close", pd.Series([row["close"]])).iloc[max(0, bar_idx - self.lookback)]
            if "close" in physics.columns
            else 0
        )

        # Trend conditions (laminar flow)
        # High energy + low damping = strong trend
        laminar_flow = energy_pct > 0.7 and damping_pct < 0.3

        if laminar_flow:
            if price_momentum > 0:
                return 1  # Long
            elif price_momentum < 0:
                return -1  # Short

        return 0


# === MEAN REVERSION STRATEGY ===


class MeanReversionStrategy(BaseStrategy):
    """
    Fade extremes when market is overdamped (high friction, low energy).

    Entry: Overdamped regime + price at distribution extreme
    Exit: Price returns to mean or energy increases (trend starting)

    No hardcoded z-score thresholds - uses adaptive percentiles.
    """

    def __init__(self, lookback: int = 20):
        super().__init__(
            StrategyConfig(
                name="Mean Reversion",
                lookback=lookback,
                description="Fade extremes in high-friction regimes",
            )
        )

    def signal(self, row: pd.Series, physics: pd.DataFrame, bar_idx: int) -> int:
        if bar_idx < self.lookback or bar_idx >= len(physics):
            return 0

        physics_row = physics.iloc[bar_idx]

        # Get regime and damping
        regime = physics_row.get("regime", "unknown")
        damping_pct = physics_row.get("damping_pct", 0.5)
        energy_pct = physics_row.get("energy_pct", 0.5)

        # Mean reversion conditions (overdamped, high friction)
        ranging_market = regime == "OVERDAMPED" or (damping_pct > 0.7 and energy_pct < 0.3)

        if not ranging_market:
            return 0

        # Calculate price position in rolling distribution
        # Use simple percentile rank instead of fixed z-score
        close = row["close"]

        # Get recent closes for comparison
        lookback_start = max(0, bar_idx - self.lookback)
        if "close" in physics.columns:
            recent_closes = physics["close"].iloc[lookback_start:bar_idx]
        else:
            return 0

        if len(recent_closes) < 5:
            return 0

        # Calculate percentile rank of current close
        price_pct = (recent_closes < close).mean()

        # Extreme conditions (adaptive, not fixed thresholds)
        if price_pct < 0.1:  # Bottom 10%
            return 1  # Buy oversold
        elif price_pct > 0.9:  # Top 10%
            return -1  # Sell overbought

        return 0


# === BREAKOUT STRATEGY ===


class BreakoutStrategy(BaseStrategy):
    """
    Enter on range expansion with increasing energy.

    Entry: Price breaks N-bar high/low + Energy surge
    Exit: Energy decreasing or volatility contracting

    Targets "fat candles" - Berserker-style approach.
    """

    def __init__(self, lookback: int = 20):
        super().__init__(
            StrategyConfig(
                name="Breakout",
                lookback=lookback,
                description="Range expansion with energy surge detection",
            )
        )

    def signal(self, row: pd.Series, physics: pd.DataFrame, bar_idx: int) -> int:
        if bar_idx < self.lookback or bar_idx >= len(physics):
            return 0

        physics_row = physics.iloc[bar_idx]

        # Energy and jerk (rate of energy change)
        energy_pct = physics_row.get("energy_pct", 0.5)

        # Calculate jerk approximation (energy acceleration)
        if bar_idx > 1:
            prev_energy = physics.iloc[bar_idx - 1].get("energy_pct", 0.5)
            energy_delta = energy_pct - prev_energy
        else:
            energy_delta = 0

        # Breakout conditions: energy surge
        energy_surge = energy_pct > 0.8 and energy_delta > 0.1

        if not energy_surge:
            return 0

        # Check price breakout vs recent range
        close = row["close"]
        lookback_start = max(0, bar_idx - self.lookback)

        if "high" in physics.columns and "low" in physics.columns:
            recent_high = physics["high"].iloc[lookback_start:bar_idx].max()
            recent_low = physics["low"].iloc[lookback_start:bar_idx].min()
        else:
            return 0

        # Breakout detection
        if close > recent_high:
            return 1  # Long breakout
        elif close < recent_low:
            return -1  # Short breakdown

        return 0


# === VALIDATION RUNNER ===


def run_strategy_validation(
    data: pd.DataFrame,
    symbol_spec: SymbolSpec,
    strategies: List[BaseStrategy] = None,
    initial_capital: float = 10000.0,
    stress_test: bool = False,
    verbose: bool = True,
) -> Dict[str, BacktestResult]:
    """
    Run all strategies on data to validate backtesting engine.

    Checks:
    1. Traditional rules-based execution
    2. Data quality handling
    3. Gap/event handling
    4. Cost modeling accuracy

    Args:
        data: OHLCV DataFrame
        symbol_spec: Instrument specification
        strategies: List of strategies (default: all three)
        initial_capital: Starting capital
        stress_test: Whether to run stress tests
        verbose: Print progress

    Returns:
        Dictionary of strategy name -> BacktestResult
    """
    if strategies is None:
        strategies = [
            TrendStrategy(),
            MeanReversionStrategy(),
            BreakoutStrategy(),
        ]

    # Validate data quality first
    if verbose:
        print(f"Validating data quality for {symbol_spec.symbol}...")

    quality_report = validate_data(data, symbol_spec.symbol)

    if verbose:
        print(f"  Bars: {quality_report.total_bars:,}")
        print(f"  Completeness: {quality_report.completeness_pct:.1%}")
        print(f"  Quality Score: {quality_report.quality_score:.1f}/100")
        print()

    # Compute physics state once
    physics = PhysicsEngine()
    physics_state = physics.compute_physics_state(data["close"])

    # Add OHLC to physics for strategy access
    physics_state["open"] = data["open"].values
    physics_state["high"] = data["high"].values
    physics_state["low"] = data["low"].values
    physics_state["close"] = data["close"].values

    results = {}

    for strategy in strategies:
        if verbose:
            print(f"Running {strategy.name}...")

        engine = BacktestEngine(
            initial_capital=initial_capital,
            data_quality_check=True,
            timeframe="H1",
        )

        # Wrap signal function to pass physics state
        def make_signal_func(strat, phys):
            def signal_func(row, physics_df, bar_idx):
                return strat.signal(row, phys, bar_idx)

            return signal_func

        result = engine.run_backtest(
            data=data,
            symbol_spec=symbol_spec,
            signal_func=make_signal_func(strategy, physics_state),
        )

        results[strategy.name] = result

        if verbose:
            print(f"  Trades: {result.total_trades}")
            print(f"  Win Rate: {result.win_rate:.1%}")
            print(f"  Net P&L: ${result.total_net_pnl:,.2f}")
            print(f"  Sharpe: {result.sharpe_ratio:.2f}")
            print(f"  Max DD: {result.max_drawdown_pct:.1%}")
            print()

        # Run stress test if requested
        if stress_test:
            if verbose:
                print(f"  Running stress tests for {strategy.name}...")

            stress_report = quick_stress_test(
                data=data,
                signal_func=make_signal_func(strategy, physics_state),
                symbol_spec=symbol_spec,
            )

            results[f"{strategy.name}_stress"] = stress_report

            if verbose:
                print(f"  Robustness Score: {stress_report.robustness_score:.1f}/100")
                print(f"  Survival Rate: {stress_report.survival_rate:.1%}")
                print()

    return results


def print_comparison(results: Dict[str, BacktestResult]):
    """Print comparison table of strategy results."""
    print("\n" + "=" * 70)
    print("STRATEGY COMPARISON")
    print("=" * 70)

    headers = ["Strategy", "Trades", "Win%", "Net P&L", "Sharpe", "MaxDD%", "Z-Factor"]

    # Header row
    print(
        f"{headers[0]:<20} {headers[1]:>8} {headers[2]:>8} {headers[3]:>12} "
        f"{headers[4]:>8} {headers[5]:>8} {headers[6]:>8}"
    )
    print("-" * 70)

    for name, result in results.items():
        if "_stress" in name:
            continue  # Skip stress test results in main comparison

        print(
            f"{name:<20} {result.total_trades:>8} {result.win_rate * 100:>7.1f}% "
            f"${result.total_net_pnl:>10,.0f} {result.sharpe_ratio:>8.2f} "
            f"{result.max_drawdown_pct:>7.1f}% {result.z_factor:>8.2f}"
        )

    print("=" * 70)


def load_data(filepath: str) -> pd.DataFrame:
    """Load OHLCV data from CSV."""
    df = pd.read_csv(filepath)

    # Standardize column names
    column_map = {
        "<DATE>": "date",
        "<TIME>": "time",
        "<OPEN>": "open",
        "<HIGH>": "high",
        "<LOW>": "low",
        "<CLOSE>": "close",
        "<TICKVOL>": "volume",
        "<VOL>": "volume",
        "<SPREAD>": "spread",
    }

    df.columns = [column_map.get(c, c.lower()) for c in df.columns]

    # Combine date and time if separate
    if "date" in df.columns and "time" in df.columns:
        df["datetime"] = pd.to_datetime(df["date"] + " " + df["time"])
        df.set_index("datetime", inplace=True)
    elif "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"])
        df.set_index("time", inplace=True)

    return df


def main():
    parser = argparse.ArgumentParser(description="Test backtesting strategies")
    parser.add_argument("--symbol", type=str, default="BTCUSD", help="Symbol to test")
    parser.add_argument("--data", type=str, help="Path to CSV data file")
    parser.add_argument("--capital", type=float, default=10000.0, help="Initial capital")
    parser.add_argument("--stress-test", action="store_true", help="Run stress tests")
    parser.add_argument(
        "--strategy",
        type=str,
        choices=["trend", "mr", "breakout", "all"],
        default="all",
        help="Strategy to test",
    )

    args = parser.parse_args()

    # Load data
    if args.data:
        data_path = args.data
    else:
        # Try to find data file
        data_dir = Path(__file__).parent.parent / "data" / "master"
        candidates = list(data_dir.glob(f"{args.symbol}*.csv"))
        if candidates:
            data_path = str(candidates[0])
        else:
            print(f"No data file found for {args.symbol}")
            print(f"Please specify --data or place CSV in {data_dir}")
            sys.exit(1)

    print(f"Loading data from {data_path}...")
    data = load_data(data_path)
    print(f"Loaded {len(data):,} bars")

    # Get symbol spec
    try:
        symbol_spec = get_symbol_spec(args.symbol)
    except KeyError:
        print(f"Symbol spec not found for {args.symbol}, using defaults")
        symbol_spec = SymbolSpec(symbol=args.symbol)

    # Select strategies
    strategies = []
    if args.strategy == "all":
        strategies = [TrendStrategy(), MeanReversionStrategy(), BreakoutStrategy()]
    elif args.strategy == "trend":
        strategies = [TrendStrategy()]
    elif args.strategy == "mr":
        strategies = [MeanReversionStrategy()]
    elif args.strategy == "breakout":
        strategies = [BreakoutStrategy()]

    # Run validation
    results = run_strategy_validation(
        data=data,
        symbol_spec=symbol_spec,
        strategies=strategies,
        initial_capital=args.capital,
        stress_test=args.stress_test,
        verbose=True,
    )

    # Print comparison
    print_comparison(results)

    print("\nValidation complete!")


if __name__ == "__main__":
    main()
