"""Rolling OOS drift monitoring utilities (no-lookahead by construction)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import numpy as np
import pandas as pd

from kinetra.renko.backtest import (
    FilterParams,
    InstrumentBacktestResult,
    StopParams,
    backtest_instrument,
)


@dataclass(frozen=True, slots=True)
class FoldResult:
    fold_idx: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    train_omega: float
    train_z: float
    train_trades: int
    oos_omega: float
    oos_z: float
    oos_trades: int
    oos_pnl_usd: float
    oos_max_dd_usd: float


@dataclass(frozen=True, slots=True)
class RollingOOSResult:
    symbol: str
    folds: List[FoldResult]
    stitched_oos_omega: float
    stitched_oos_z: float
    stitched_oos_trades: int
    stitched_oos_pnl_usd: float
    stitched_oos_max_dd_usd: float


def _ensure_series(closes: pd.Series) -> pd.Series:
    s = closes.sort_index()
    if not isinstance(s.index, pd.DatetimeIndex):
        raise ValueError("closes index must be DatetimeIndex")
    if s.index.tz is None:
        s.index = s.index.tz_localize("UTC")
    return s.dropna()


def _max_drawdown(equity: Sequence[float]) -> float:
    arr = np.asarray(list(equity), dtype=float)
    if arr.size == 0:
        return 0.0
    peak = np.maximum.accumulate(arr)
    dd = arr - peak
    return float(dd.min())


def _metrics(returns: np.ndarray) -> tuple[float, float]:
    from kinetra.backtesting.metrics import calculate_z_factor, omega_ratio

    if len(returns) == 0:
        return 0.0, 0.0
    return float(omega_ratio(returns)), float(calculate_z_factor(returns))


def _run_segment(
    symbol: str,
    closes: pd.Series,
    brick_size: float,
    filter_params: FilterParams,
    stop_params: StopParams,
    session_break_minutes: float,
) -> InstrumentBacktestResult:
    return backtest_instrument(
        symbol=symbol,
        closes=closes,
        brick_size=brick_size,
        filter_params=filter_params,
        stop_params=stop_params,
        session_break_minutes=session_break_minutes,
    )


def rolling_oos_instrument(
    *,
    symbol: str,
    closes: pd.Series,
    brick_size: float,
    filter_params: FilterParams,
    stop_params: StopParams,
    train_days: int = 180,
    test_days: int = 30,
    step_days: int = 30,
    embargo_minutes: int = 30,
    min_train_bars: int = 200,
    min_test_bars: int = 60,
    session_break_minutes: float = 30.0,
) -> RollingOOSResult:
    """Run rolling train->test evaluation with strict no-lookahead slicing.

    Anti-lookahead guarantees:
    - Each fold test interval is strictly after its train interval.
    - Optional embargo removes bars immediately preceding test_start from train set.
    - Strategy parameters are fixed inputs; no per-fold optimization.
    - Stitched OOS metrics are computed from fold TEST trades only.
    """
    s = _ensure_series(closes)
    if len(s) < (min_train_bars + min_test_bars):
        return RollingOOSResult(symbol, [], 0.0, 0.0, 0, 0.0, 0.0)

    start = s.index.min()
    end = s.index.max()

    train_td = pd.Timedelta(days=train_days)
    test_td = pd.Timedelta(days=test_days)
    step_td = pd.Timedelta(days=step_days)
    embargo_td = pd.Timedelta(minutes=embargo_minutes)

    folds: List[FoldResult] = []
    stitched_returns: list[float] = []
    stitched_equity: list[float] = [0.0]
    stitched_trades = 0

    test_start = start + train_td
    fold_idx = 0

    while True:
        test_end = test_start + test_td
        if test_end > end:
            break

        train_end = test_start - embargo_td
        train_start = train_end - train_td
        if train_start < start:
            test_start = test_start + step_td
            continue

        train = s[(s.index >= train_start) & (s.index < train_end)]
        test = s[(s.index >= test_start) & (s.index < test_end)]
        if len(train) < min_train_bars or len(test) < min_test_bars:
            test_start = test_start + step_td
            continue

        train_res = _run_segment(
            symbol=symbol,
            closes=train,
            brick_size=brick_size,
            filter_params=filter_params,
            stop_params=stop_params,
            session_break_minutes=session_break_minutes,
        )
        oos_res = _run_segment(
            symbol=symbol,
            closes=test,
            brick_size=brick_size,
            filter_params=filter_params,
            stop_params=stop_params,
            session_break_minutes=session_break_minutes,
        )

        fold_returns = np.array([t.net_usd for t in oos_res.trades], dtype=float)
        if fold_returns.size:
            stitched_returns.extend(fold_returns.tolist())
            for x in fold_returns:
                stitched_equity.append(stitched_equity[-1] + float(x))

        stitched_trades += len(oos_res.trades)
        folds.append(
            FoldResult(
                fold_idx=fold_idx,
                train_start=train_start.isoformat(),
                train_end=train_end.isoformat(),
                test_start=test_start.isoformat(),
                test_end=test_end.isoformat(),
                train_omega=float(train_res.omega),
                train_z=float(train_res.z_factor),
                train_trades=len(train_res.trades),
                oos_omega=float(oos_res.omega),
                oos_z=float(oos_res.z_factor),
                oos_trades=len(oos_res.trades),
                oos_pnl_usd=float(np.sum(fold_returns)) if fold_returns.size else 0.0,
                oos_max_dd_usd=float(oos_res.max_dd_usd),
            )
        )
        fold_idx += 1
        test_start = test_start + step_td

    if not stitched_returns:
        return RollingOOSResult(symbol, folds, 0.0, 0.0, 0, 0.0, 0.0)

    ret_arr = np.asarray(stitched_returns, dtype=float)
    omega, z = _metrics(ret_arr)
    return RollingOOSResult(
        symbol=symbol,
        folds=folds,
        stitched_oos_omega=float(omega),
        stitched_oos_z=float(z),
        stitched_oos_trades=stitched_trades,
        stitched_oos_pnl_usd=float(np.sum(ret_arr)),
        stitched_oos_max_dd_usd=float(_max_drawdown(stitched_equity)),
    )
