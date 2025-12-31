"""
Walk-Forward Validation Engine

Implements proper out-of-sample testing with variable windows:
- In-sample (IS): Train/optimize parameters
- Out-of-sample (OOS): Test on unseen data
- Rolling or anchored windows
- Variable window sizes for robustness testing
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

import numpy as np
import pandas as pd

from .backtest_engine import BacktestEngine, BacktestResult
from .config import MAX_WORKERS
from .symbol_spec import SymbolSpec


class WindowType(Enum):
    """Type of walk-forward window."""

    ROLLING = "rolling"  # Fixed-size rolling window
    ANCHORED = "anchored"  # Expanding window from anchor point
    EXPANDING = "expanding"  # Alias for anchored


@dataclass
class WalkForwardWindow:
    """A single walk-forward window (IS + OOS period)."""

    window_id: int

    # In-sample period
    is_start: datetime
    is_end: datetime

    # Out-of-sample period
    oos_start: datetime
    oos_end: datetime

    # Bar counts (defaults must come after non-defaults)
    is_bars: int = 0
    oos_bars: int = 0

    # Results
    is_result: Optional[BacktestResult] = None
    oos_result: Optional[BacktestResult] = None

    # Optimized parameters (if any)
    optimized_params: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_duration_days(self) -> float:
        return (self.is_end - self.is_start).total_seconds() / 86400

    @property
    def oos_duration_days(self) -> float:
        return (self.oos_end - self.oos_start).total_seconds() / 86400

    @property
    def is_oos_ratio(self) -> float:
        """Ratio of IS to OOS period."""
        if self.oos_duration_days == 0:
            return float("inf")
        return self.is_duration_days / self.oos_duration_days


@dataclass
class WalkForwardResult:
    """Complete walk-forward validation results."""

    windows: List[WalkForwardWindow]

    # Aggregate OOS metrics
    total_oos_trades: int = 0
    total_oos_net_pnl: float = 0.0
    oos_win_rate: float = 0.0
    oos_sharpe: float = 0.0
    oos_max_drawdown: float = 0.0

    # Robustness metrics
    oos_consistency: float = 0.0  # % of OOS windows profitable
    is_oos_correlation: float = 0.0  # Correlation between IS and OOS performance
    degradation_ratio: float = 0.0  # OOS performance / IS performance

    # Combined equity curve
    combined_equity: Optional[pd.Series] = None

    def to_dict(self) -> Dict:
        return {
            "num_windows": len(self.windows),
            "total_oos_trades": self.total_oos_trades,
            "total_oos_net_pnl": self.total_oos_net_pnl,
            "oos_win_rate": self.oos_win_rate,
            "oos_sharpe": self.oos_sharpe,
            "oos_max_drawdown": self.oos_max_drawdown,
            "oos_consistency": self.oos_consistency,
            "is_oos_correlation": self.is_oos_correlation,
            "degradation_ratio": self.degradation_ratio,
        }


class WalkForwardEngine:
    """
    Walk-forward validation engine.

    Splits data into IS/OOS windows and validates strategy performance
    on truly out-of-sample data.

    Usage:
        engine = WalkForwardEngine(
            is_bars=500,      # 500 bars for training
            oos_bars=100,     # 100 bars for testing
            step_bars=100,    # Roll forward 100 bars
            window_type=WindowType.ROLLING
        )

        result = engine.run(
            data=df,
            symbol_spec=spec,
            train_func=my_train_function,  # Optional optimization
            signal_func=my_signal_function
        )
    """

    def __init__(
        self,
        is_bars: int = 500,
        oos_bars: int = 100,
        step_bars: Optional[int] = None,
        window_type: WindowType = WindowType.ROLLING,
        min_is_bars: int = 100,
        initial_capital: float = 100000.0,
    ):
        """
        Initialize walk-forward engine.

        Args:
            is_bars: Number of bars for in-sample period
            oos_bars: Number of bars for out-of-sample period
            step_bars: Bars to step forward (default = oos_bars)
            window_type: Rolling or anchored windows
            min_is_bars: Minimum IS bars required
            initial_capital: Starting capital for each window
        """
        self.is_bars = is_bars
        self.oos_bars = oos_bars
        self.step_bars = step_bars or oos_bars
        self.window_type = window_type
        self.min_is_bars = min_is_bars
        self.initial_capital = initial_capital

    def generate_windows(
        self,
        data: pd.DataFrame,
    ) -> List[Tuple[int, int, int, int]]:
        """
        Generate IS/OOS window indices.

        Returns:
            List of (is_start, is_end, oos_start, oos_end) tuples
        """
        n = len(data)
        windows = []

        if self.window_type == WindowType.ANCHORED:
            # Anchored: IS always starts at 0, expands
            is_start = 0
            oos_start = self.is_bars

            while oos_start + self.oos_bars <= n:
                is_end = oos_start
                oos_end = oos_start + self.oos_bars

                if is_end - is_start >= self.min_is_bars:
                    windows.append((is_start, is_end, oos_start, oos_end))

                oos_start += self.step_bars
        else:
            # Rolling: Fixed-size IS window
            is_start = 0

            while is_start + self.is_bars + self.oos_bars <= n:
                is_end = is_start + self.is_bars
                oos_start = is_end
                oos_end = oos_start + self.oos_bars

                windows.append((is_start, is_end, oos_start, oos_end))

                is_start += self.step_bars

        return windows

    def run(
        self,
        data: pd.DataFrame,
        symbol_spec: SymbolSpec,
        signal_func: Optional[Callable] = None,
        train_func: Optional[Callable] = None,
        agent_factory: Optional[Callable] = None,
    ) -> WalkForwardResult:
        """
        Run walk-forward validation.

        Args:
            data: OHLCV DataFrame
            symbol_spec: Instrument specification
            signal_func: Signal generation function
            train_func: Optional training/optimization function
                        Signature: train_func(is_data, symbol_spec) -> optimized_params
            agent_factory: Optional factory to create RL agent
                          Signature: agent_factory(is_data) -> agent

        Returns:
            WalkForwardResult with all window results
        """
        windows = self.generate_windows(data)

        if len(windows) == 0:
            raise ValueError(
                f"Not enough data for walk-forward. Need at least "
                f"{self.is_bars + self.oos_bars} bars, got {len(data)}"
            )

        wf_windows: List[WalkForwardWindow] = []
        all_oos_equity = []

        def process_window(args):
            """Process single window - for parallel execution."""
            i, is_start, is_end, oos_start, oos_end = args
            is_data = data.iloc[is_start:is_end].copy()
            oos_data = data.iloc[oos_start:oos_end].copy()

            # Get timestamps
            is_start_time = (
                is_data.index[0] if hasattr(is_data.index, "__getitem__") else datetime.now()
            )
            is_end_time = (
                is_data.index[-1] if hasattr(is_data.index, "__getitem__") else datetime.now()
            )
            oos_start_time = (
                oos_data.index[0] if hasattr(oos_data.index, "__getitem__") else datetime.now()
            )
            oos_end_time = (
                oos_data.index[-1] if hasattr(oos_data.index, "__getitem__") else datetime.now()
            )

            wf_window = WalkForwardWindow(
                window_id=i,
                is_start=is_start_time,
                is_end=is_end_time,
                is_bars=len(is_data),
                oos_start=oos_start_time,
                oos_end=oos_end_time,
                oos_bars=len(oos_data),
            )

            # Optional: Train/optimize on IS data
            agent = None
            current_signal_func = signal_func

            if train_func is not None:
                optimized_params = train_func(is_data, symbol_spec)
                wf_window.optimized_params = optimized_params

            if agent_factory is not None:
                agent = agent_factory(is_data)

            # Run IS backtest
            is_engine = BacktestEngine(initial_capital=self.initial_capital)
            wf_window.is_result = is_engine.run_backtest(
                is_data, symbol_spec, current_signal_func, agent
            )

            # Run OOS backtest (with same parameters/agent)
            oos_engine = BacktestEngine(initial_capital=self.initial_capital)
            wf_window.oos_result = oos_engine.run_backtest(
                oos_data, symbol_spec, current_signal_func, agent
            )

            return wf_window

        # Parallel walk-forward - use configured max workers
        n_workers = min(mp.cpu_count(), len(windows), MAX_WORKERS)
        window_args = [(i, *w) for i, w in enumerate(windows)]

        if n_workers > 1 and len(windows) >= 4:
            # Parallel execution
            with ProcessPoolExecutor(max_workers=n_workers) as executor:
                futures = {executor.submit(process_window, args): args[0] for args in window_args}
                for future in as_completed(futures):
                    wf_window = future.result()
                    wf_windows.append(wf_window)
                    if wf_window.oos_result and wf_window.oos_result.equity_curve is not None:
                        all_oos_equity.append(wf_window.oos_result.equity_curve)
            # Sort by window_id to maintain order
            wf_windows.sort(key=lambda w: w.window_id)
        else:
            # Sequential for small runs
            for args in window_args:
                wf_window = process_window(args)
                wf_windows.append(wf_window)
                if wf_window.oos_result and wf_window.oos_result.equity_curve is not None:
                    all_oos_equity.append(wf_window.oos_result.equity_curve)

        # Calculate aggregate metrics
        result = self._calculate_aggregate_metrics(wf_windows, all_oos_equity)

        return result

    def _calculate_aggregate_metrics(
        self,
        windows: List[WalkForwardWindow],
        oos_equity_curves: List[pd.Series],
    ) -> WalkForwardResult:
        """Calculate aggregate walk-forward metrics."""

        # Collect OOS results
        oos_results = [w.oos_result for w in windows if w.oos_result is not None]
        is_results = [w.is_result for w in windows if w.is_result is not None]

        if not oos_results:
            return WalkForwardResult(windows=windows)

        # Total OOS trades and P&L
        total_trades = sum(r.total_trades for r in oos_results)
        total_pnl = sum(r.total_net_pnl for r in oos_results)

        # OOS win rate
        total_wins = sum(r.winning_trades for r in oos_results)
        win_rate = total_wins / total_trades if total_trades > 0 else 0

        # OOS Sharpe (average of window Sharpes)
        sharpes = [r.sharpe_ratio for r in oos_results if r.sharpe_ratio != 0]
        avg_sharpe = np.mean(sharpes) if sharpes else 0

        # Max drawdown across all OOS periods
        max_dd = max(r.max_drawdown for r in oos_results) if oos_results else 0

        # Consistency: % of profitable OOS windows
        profitable_windows = sum(1 for r in oos_results if r.total_net_pnl > 0)
        consistency = profitable_windows / len(oos_results) if oos_results else 0

        # IS-OOS correlation
        is_pnls = [r.total_net_pnl for r in is_results]
        oos_pnls = [r.total_net_pnl for r in oos_results]

        if len(is_pnls) > 1 and len(oos_pnls) > 1:
            min_len = min(len(is_pnls), len(oos_pnls))
            correlation = np.corrcoef(is_pnls[:min_len], oos_pnls[:min_len])[0, 1]
            if np.isnan(correlation):
                correlation = 0
        else:
            correlation = 0

        # Degradation ratio: OOS performance / IS performance
        is_total_pnl = sum(r.total_net_pnl for r in is_results)
        degradation = total_pnl / is_total_pnl if is_total_pnl != 0 else 0

        # Combine OOS equity curves
        combined_equity = None
        if oos_equity_curves:
            # Chain equity curves together
            combined = []
            running_equity = self.initial_capital

            for curve in oos_equity_curves:
                if len(curve) > 0:
                    # Scale curve to start from running equity
                    scale_factor = running_equity / curve.iloc[0] if curve.iloc[0] != 0 else 1
                    scaled = curve * scale_factor
                    combined.extend(scaled.tolist())
                    running_equity = scaled.iloc[-1]

            combined_equity = pd.Series(combined)

        return WalkForwardResult(
            windows=windows,
            total_oos_trades=total_trades,
            total_oos_net_pnl=total_pnl,
            oos_win_rate=win_rate,
            oos_sharpe=avg_sharpe,
            oos_max_drawdown=max_dd,
            oos_consistency=consistency,
            is_oos_correlation=correlation,
            degradation_ratio=degradation,
            combined_equity=combined_equity,
        )

    def run_variable_windows(
        self,
        data: pd.DataFrame,
        symbol_spec: SymbolSpec,
        signal_func: Optional[Callable] = None,
        is_bar_sizes: List[int] = None,
        oos_bar_sizes: List[int] = None,
    ) -> Dict[Tuple[int, int], WalkForwardResult]:
        """
        Run walk-forward with variable window sizes for robustness testing.

        Args:
            data: OHLCV DataFrame
            symbol_spec: Instrument specification
            signal_func: Signal generation function
            is_bar_sizes: List of IS sizes to test (default: [250, 500, 750, 1000])
            oos_bar_sizes: List of OOS sizes to test (default: [50, 100, 150, 200])

        Returns:
            Dict mapping (is_bars, oos_bars) -> WalkForwardResult
        """
        if is_bar_sizes is None:
            is_bar_sizes = [250, 500, 750, 1000]
        if oos_bar_sizes is None:
            oos_bar_sizes = [50, 100, 150, 200]

        results = {}

        for is_bars in is_bar_sizes:
            for oos_bars in oos_bar_sizes:
                # Skip if not enough data
                if is_bars + oos_bars > len(data):
                    continue

                engine = WalkForwardEngine(
                    is_bars=is_bars,
                    oos_bars=oos_bars,
                    step_bars=oos_bars,
                    window_type=self.window_type,
                    initial_capital=self.initial_capital,
                )

                try:
                    result = engine.run(data, symbol_spec, signal_func)
                    results[(is_bars, oos_bars)] = result
                except ValueError:
                    # Not enough data for this configuration
                    continue

        return results

    def analyze_robustness(
        self,
        variable_results: Dict[Tuple[int, int], WalkForwardResult],
    ) -> pd.DataFrame:
        """
        Analyze robustness across variable window configurations.

        Args:
            variable_results: Results from run_variable_windows()

        Returns:
            DataFrame with metrics for each configuration
        """
        rows = []

        for (is_bars, oos_bars), result in variable_results.items():
            rows.append(
                {
                    "is_bars": is_bars,
                    "oos_bars": oos_bars,
                    "is_oos_ratio": is_bars / oos_bars,
                    "num_windows": len(result.windows),
                    "oos_trades": result.total_oos_trades,
                    "oos_pnl": result.total_oos_net_pnl,
                    "oos_win_rate": result.oos_win_rate,
                    "oos_sharpe": result.oos_sharpe,
                    "oos_consistency": result.oos_consistency,
                    "is_oos_correlation": result.is_oos_correlation,
                    "degradation_ratio": result.degradation_ratio,
                }
            )

        df = pd.DataFrame(rows)

        if len(df) > 0:
            # Sort by consistency then Sharpe
            df = df.sort_values(["oos_consistency", "oos_sharpe"], ascending=[False, False])

        return df
