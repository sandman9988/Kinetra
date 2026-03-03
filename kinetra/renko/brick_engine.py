"""
Renko Brick Engine
==================

Close-only Renko brick construction from price series.

Design decisions (from RENKO_KINETRA_DESIGN_SPEC.md §6.1):
  - Classic Renko: price must move ≥1 brick in the same direction to add,
    ≥2 bricks in the opposite direction to reverse.
  - Close-only construction: avoids intrabar lookahead bias.
  - Brick size in price units (not pips, not percentage).
  - Returns DataFrame: brick_open, brick_close, direction (+1/-1), time.

This module is the canonical Renko builder for Kinetra.  All other scripts
should import from here rather than maintaining local copies.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd


@dataclass(frozen=True, slots=True)
class BrickSummary:
    """Lightweight summary statistics for a Renko brick sequence."""

    n_bricks: int
    n_up: int
    n_down: int
    bricks_per_bar: float  # bricks / source bars
    longest_run: int  # longest consecutive same-direction run
    mean_run_length: float  # mean same-direction run length


def build_renko(
    closes: pd.Series,
    brick_size: float,
    *,
    return_ref: bool = False,
    session_break_minutes: float = 30.0,
) -> pd.DataFrame:
    """
    Construct Renko bricks from a close-price series.

    Uses ONLY close prices to avoid intrabar lookahead bias — the single
    most important correctness requirement for Renko backtesting integrity.
    (High/Low from aggregated bars would give knowledge of intrabar price
    paths not available at bar open, invalidating any backtest.)

    Parameters
    ----------
    closes : pd.Series
        Close prices indexed by datetime (or any ordered index).
        Must be non-empty and contain finite values.
    brick_size : float
        Fixed brick size in **price units** (not pips, not percentage).
        Must be strictly positive.
    return_ref : bool, default False
        If True, include a ``ref_price`` column showing the running
        reference price after each brick.  Useful for debugging.
    session_break_minutes : float, default 30.0
        Minimum gap in minutes between consecutive bars to trigger a session break.
        When a gap >= this threshold is detected, the reference price is reset to
        the current close price, preventing brick bursts across session boundaries.
        Bricks produced from the bar immediately after the break are marked with
        session_break=True for diagnostics.

    Returns
    -------
    pd.DataFrame
        Columns: ``brick_open``, ``brick_close``, ``direction`` (+1/−1),
        ``time``, ``session_break``.  Empty DataFrame (with correct columns) when no bricks
        can be constructed.

    Raises
    ------
    ValueError
        If ``brick_size <= 0`` or ``closes`` contains non-finite values.

    Notes
    -----
    The algorithm is O(B) where B = total bricks produced.  For typical
    Renko configurations this is significantly smaller than the source
    bar count, making the inner loop fast in practice.
    """
    _cols = ["brick_open", "brick_close", "direction", "time", "session_break"]
    if return_ref:
        _cols = [*_cols, "ref_price"]

    # ── Input validation ────────────────────────────────────────────────
    if brick_size <= 0:
        raise ValueError(f"brick_size must be > 0, got {brick_size}")

    if session_break_minutes < 0:
        raise ValueError(f"session_break_minutes must be >= 0, got {session_break_minutes}")
    if session_break_minutes == 0:
        warnings.warn(
            "session_break_minutes=0 disables session-break detection. "
            "Brick bursts across session boundaries will not be suppressed.",
            UserWarning,
            stacklevel=2,
        )

    if closes.empty:
        return pd.DataFrame(columns=_cols)

    vals = closes.values.astype(np.float64)
    if not np.all(np.isfinite(vals)):
        raise ValueError(
            "closes contains non-finite values (NaN/Inf); "
            "clean the series before building Renko bricks"
        )

    # ── Core brick construction loop ────────────────────────────────────
    brick_opens: List[float] = []
    brick_closes: List[float] = []
    directions: List[int] = []
    times: list = []
    session_breaks: List[bool] = []
    ref_prices: List[float] = []

    # Reserve list capacity (CPython hint — no-op but harmless)
    for lst in (brick_opens, brick_closes, directions, times, session_breaks, ref_prices):
        lst.clear()

    ref = float(vals[0])
    idx = closes.index  # preserve original index type

    for i in range(1, len(vals)):
        current_time = idx[i]
        prev_time = idx[i - 1]
        session_break_detected = False
        if isinstance(current_time, pd.Timestamp) and isinstance(prev_time, pd.Timestamp):
            dt_minutes = (current_time - prev_time).total_seconds() / 60.0
            if dt_minutes > session_break_minutes:
                session_break_detected = True
                ref = float(vals[i])  # Reset reference price to current close

        price = float(vals[i])
        diff = price - ref
        n = int(abs(diff) / brick_size)
        if n >= 1:
            direction = 1 if diff > 0 else -1
            ts = idx[i]
            for j in range(n):
                b_open = ref + direction * j * brick_size
                b_close = ref + direction * (j + 1) * brick_size
                brick_opens.append(b_open)
                brick_closes.append(b_close)
                directions.append(direction)
                times.append(ts)
                session_breaks.append(session_break_detected)
            ref = ref + direction * n * brick_size
            if return_ref:
                ref_prices.extend([ref] * n)
            # Reset flag after producing bricks for this bar
            session_break_detected = False

    # ── Build result DataFrame ──────────────────────────────────────────
    if not brick_opens:
        return pd.DataFrame(columns=_cols)

    result = pd.DataFrame(
        {
            "brick_open": np.array(brick_opens, dtype=np.float64),
            "brick_close": np.array(brick_closes, dtype=np.float64),
            "direction": np.array(directions, dtype=np.int8),
            "time": times,
            "session_break": np.array(session_breaks, dtype=bool),
        }
    )
    # Coerce to datetime with UTC when possible
    result["time"] = pd.to_datetime(result["time"], utc=True, errors="coerce")

    if return_ref:
        result["ref_price"] = np.array(ref_prices, dtype=np.float64)

    return result


class IncrementalRenkoBuilder:
    """
    Stateful, one-bar-at-a-time Renko brick construction.

    Produces the same bricks as :func:`build_renko` when fed the same
    price sequence.  Used by :class:`~kinetra.renko.trading_engine.RenkoEngine`
    for streaming (paper/live) operation.

    Parameters
    ----------
    brick_size : float
        Fixed brick size in price units.  Must be > 0.
    session_break_minutes : float
        Gap threshold in minutes to reset the reference price (same
        semantics as :func:`build_renko`).
    """

    def __init__(self, brick_size: float, session_break_minutes: float = 30.0) -> None:
        if brick_size <= 0:
            raise ValueError(f"brick_size must be > 0, got {brick_size}")
        self.brick_size = brick_size
        self.session_break_minutes = session_break_minutes
        self._ref: Optional[float] = None
        self._last_time: Optional[pd.Timestamp] = None

    def update(self, price: float, time: pd.Timestamp) -> List[tuple]:
        """
        Process one close price bar.

        Returns a list of ``(brick_open, brick_close, direction)`` tuples —
        empty when no new bricks formed.
        """
        if not np.isfinite(price):
            return []

        if self._ref is None:
            self._ref = price
            self._last_time = time
            return []

        # Session-break: reset reference to prevent burst bricks across gaps
        if self._last_time is not None and self.session_break_minutes > 0:
            try:
                gap = (time - self._last_time).total_seconds() / 60.0
                if gap > self.session_break_minutes:
                    self._ref = price
                    self._last_time = time
                    return []
            except Exception:
                pass

        self._last_time = time
        diff = price - self._ref
        n = int(abs(diff) / self.brick_size)
        if n < 1:
            return []

        direction = 1 if diff > 0 else -1
        bricks = []
        for j in range(n):
            b_open = self._ref + direction * j * self.brick_size
            b_close = self._ref + direction * (j + 1) * self.brick_size
            bricks.append((b_open, b_close, direction))
        self._ref = self._ref + direction * n * self.brick_size
        return bricks


def brick_summary(bricks: pd.DataFrame) -> Optional[BrickSummary]:
    """
    Compute summary statistics for a Renko brick DataFrame.

    Parameters
    ----------
    bricks : pd.DataFrame
        Output of :func:`build_renko`.

    Returns
    -------
    BrickSummary or None
        None if the input is empty.
    """
    if bricks.empty or "direction" not in bricks.columns:
        return None

    dirs = bricks["direction"].values.astype(np.int8)
    n = len(dirs)
    n_up = int(np.sum(dirs == 1))
    n_down = int(np.sum(dirs == -1))

    # ── Run-length statistics ───────────────────────────────────────────
    if n <= 1:
        return BrickSummary(
            n_bricks=n,
            n_up=n_up,
            n_down=n_down,
            bricks_per_bar=0.0,
            longest_run=n,
            mean_run_length=float(n),
        )

    # Identify run boundaries: where direction changes
    changes = np.concatenate(([0], np.where(dirs[1:] != dirs[:-1])[0] + 1, [n]))
    run_lengths = np.diff(changes)

    longest_run = int(run_lengths.max()) if len(run_lengths) > 0 else 0
    mean_run = float(run_lengths.mean()) if len(run_lengths) > 0 else 0.0

    # bricks_per_bar: need distinct source bars
    if "time" in bricks.columns:
        n_source_bars = bricks["time"].nunique()
        bricks_per_bar = n / max(n_source_bars, 1)
    else:
        bricks_per_bar = 0.0

    return BrickSummary(
        n_bricks=n,
        n_up=n_up,
        n_down=n_down,
        bricks_per_bar=bricks_per_bar,
        longest_run=longest_run,
        mean_run_length=mean_run,
    )


def bricks_per_day(bricks: pd.DataFrame) -> float:
    """
    Compute the average number of Renko bricks produced per calendar day.

    Useful for scaling filter window sizes relative to brick frequency.

    Parameters
    ----------
    bricks : pd.DataFrame
        Output of :func:`build_renko` (must have a ``time`` column).

    Returns
    -------
    float
        Average bricks per calendar day, or 0.0 if insufficient data.
    """
    if bricks.empty or "time" not in bricks.columns:
        return 0.0

    times = pd.to_datetime(bricks["time"], utc=True, errors="coerce").dropna()
    if len(times) < 2:
        return 0.0

    span_days = (times.max() - times.min()).total_seconds() / 86400.0
    if span_days < 0.01:
        return 0.0

    return len(bricks) / span_days
