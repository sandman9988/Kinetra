"""
Renko Signal Filters
====================

Signal filters for Renko brick sequences: FlipRate and Markov stickiness.

Design decisions (from RENKO_KINETRA_DESIGN_SPEC.md §6.2):
  - **FlipRate** is the primary filter for flip-based entry systems.
    Low FlipRate → trending (long same-colour runs) → TRADE.
    High FlipRate → choppy (alternating colours) → DON'T TRADE.
  - **Markov stickiness** is the secondary filter.
    High P(UU) or P(DD) → direction persists → TRADE.
    Low stickiness → direction random → DON'T TRADE.

All filters operate on direction arrays (int8: +1 / -1) or brick close
prices, NOT on raw OHLCV.  They are pure functions with no side effects.

This module is the canonical home for Renko signal filters.  All scripts
should import from here (DRY compliance).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True, slots=True)
class FilterState:
    """Snapshot of all filter values at a single brick index.

    Useful for vectorized entry-rule evaluation and logging.
    """

    flip_rate: float
    pUU: float  # Markov P(Up→Up)
    pDD: float  # Markov P(Down→Down)


# ══════════════════════════════════════════════════════════════════════════════
# FlipRate — primary filter
# ══════════════════════════════════════════════════════════════════════════════


def flip_rate(
    directions: np.ndarray,
    window: int,
    *,
    min_periods: Optional[int] = None,
) -> np.ndarray:
    """
    Rolling flip rate over the last ``window`` bricks.

    A *flip* occurs when ``direction[t] != direction[t-1]``.
    FlipRate is the rolling mean of the binary flip indicator.

    Parameters
    ----------
    directions : np.ndarray
        Array of brick directions (+1 or -1), dtype int-like.
        Length must be ≥ 2 for any flips to be detected.
    window : int
        Rolling window size in bricks.  Must be ≥ 2.
    min_periods : int or None
        Minimum number of observations required for a valid value.
        Defaults to ``window`` (i.e. no partial windows).

    Returns
    -------
    np.ndarray (float64)
        Same length as ``directions``.  Leading values where the
        rolling window has insufficient data are ``NaN``.

    Raises
    ------
    ValueError
        If ``window < 2`` or ``directions`` has length < 2.

    Examples
    --------
    >>> import numpy as np
    >>> dirs = np.array([1, 1, 1, -1, -1, 1, -1, 1, 1, 1])
    >>> fr = flip_rate(dirs, window=5)
    >>> # First 4 values are NaN (window=5, min_periods=5)
    >>> float(fr[4])  # flips at idx 3 → 1 flip in 5-brick window
    0.2
    """
    if window < 2:
        raise ValueError(f"window must be ≥ 2, got {window}")

    dirs = np.asarray(directions, dtype=np.int8).ravel()
    n = len(dirs)

    if n < 2:
        raise ValueError(f"directions must have length ≥ 2 for flip detection, got {n}")

    if min_periods is None:
        min_periods = window

    # Binary flip indicator: flip[t] = 1 if direction changed from t-1 to t
    flips = np.empty(n, dtype=np.float64)
    flips[0] = 0.0  # no predecessor → no flip
    flips[1:] = (dirs[1:] != dirs[:-1]).astype(np.float64)

    # Rolling mean via pandas (handles NaN / min_periods cleanly)
    result = pd.Series(flips).rolling(window, min_periods=min_periods).mean().values
    return result


def flip_rate_expanding(directions: np.ndarray) -> np.ndarray:
    """
    Expanding (cumulative) flip rate — no fixed window.

    Useful for instruments where the optimal window is unknown or for
    regime-agnostic baseline measurements.

    Parameters
    ----------
    directions : np.ndarray
        Array of brick directions (+1 or -1).

    Returns
    -------
    np.ndarray (float64)
        Cumulative flip rate at each brick.  First element is 0.0.
    """
    dirs = np.asarray(directions, dtype=np.int8).ravel()
    n = len(dirs)
    if n < 2:
        return np.zeros(n, dtype=np.float64)

    flips = np.empty(n, dtype=np.float64)
    flips[0] = 0.0
    flips[1:] = (dirs[1:] != dirs[:-1]).astype(np.float64)

    cum_flips = np.cumsum(flips)
    counts = np.arange(1, n + 1, dtype=np.float64)
    return cum_flips / counts


# ══════════════════════════════════════════════════════════════════════════════
# Markov Stickiness — secondary filter
# ══════════════════════════════════════════════════════════════════════════════


def markov_stickiness(
    directions: np.ndarray,
    window: int,
    *,
    min_periods: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Rolling Markov transition stickiness probabilities.

    Computes rolling estimates of:
      - ``pUU`` = P(next=Up | current=Up)   — up-trend persistence
      - ``pDD`` = P(next=Down | current=Down) — down-trend persistence

    High stickiness → direction persists → TRADE.
    Low stickiness → direction is random → DON'T TRADE.

    Parameters
    ----------
    directions : np.ndarray
        Array of brick directions (+1 or -1), dtype int-like.
    window : int
        Rolling window size in bricks.  Must be ≥ 2.
    min_periods : int or None
        Minimum transitions required for a valid estimate.
        Defaults to ``window``.

    Returns
    -------
    pUU : np.ndarray (float64)
        Rolling P(Up→Up).  NaN where insufficient data.
    pDD : np.ndarray (float64)
        Rolling P(Down→Down).  NaN where insufficient data.

    Raises
    ------
    ValueError
        If ``window < 2`` or ``directions`` has length < 2.

    Notes
    -----
    For a fair-coin (random) Renko sequence, both pUU and pDD converge
    to 0.5.  Values significantly above 0.5 indicate regime persistence;
    values below 0.5 indicate mean-reversion (alternating colours).
    """
    if window < 2:
        raise ValueError(f"window must be ≥ 2, got {window}")

    dirs = np.asarray(directions, dtype=np.int8).ravel()
    n = len(dirs)

    if n < 2:
        raise ValueError(f"directions must have length ≥ 2, got {n}")

    if min_periods is None:
        min_periods = window

    # ── Build transition indicators ─────────────────────────────────────
    # For each pair (dirs[t-1], dirs[t]):
    #   is_up_given_up[t]   = 1 if dirs[t-1]==+1 AND dirs[t]==+1, else 0
    #   is_up[t]            = 1 if dirs[t-1]==+1 (denominator for pUU)
    #   is_down_given_down  = 1 if dirs[t-1]==-1 AND dirs[t]==-1
    #   is_down             = 1 if dirs[t-1]==-1 (denominator for pDD)

    prev = dirs[:-1]
    curr = dirs[1:]

    is_up = (prev == 1).astype(np.float64)
    is_down = (prev == -1).astype(np.float64)
    is_uu = ((prev == 1) & (curr == 1)).astype(np.float64)
    is_dd = ((prev == -1) & (curr == -1)).astype(np.float64)

    # Prepend NaN for index alignment (no transition at t=0)
    is_up = np.concatenate(([np.nan], is_up))
    is_down = np.concatenate(([np.nan], is_down))
    is_uu = np.concatenate(([np.nan], is_uu))
    is_dd = np.concatenate(([np.nan], is_dd))

    # ── Rolling sums ────────────────────────────────────────────────────
    sum_uu = pd.Series(is_uu).rolling(window, min_periods=min_periods).sum().values
    sum_up = pd.Series(is_up).rolling(window, min_periods=min_periods).sum().values
    sum_dd = pd.Series(is_dd).rolling(window, min_periods=min_periods).sum().values
    sum_down = pd.Series(is_down).rolling(window, min_periods=min_periods).sum().values

    # ── Conditional probabilities ───────────────────────────────────────
    # Guard against division by zero: if no up-states in the window, pUU = NaN
    with np.errstate(invalid="ignore", divide="ignore"):
        pUU = np.where(sum_up > 0, sum_uu / sum_up, np.nan)
        pDD = np.where(sum_down > 0, sum_dd / sum_down, np.nan)

    return pUU.astype(np.float64), pDD.astype(np.float64)


def markov_matrix(
    directions: np.ndarray,
    window: int,
    *,
    min_periods: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Full rolling 2×2 Markov transition matrix.

    Returns all four transition probabilities:
      pUU, pUD, pDU, pDD

    where pXY = P(next=Y | current=X).

    Parameters
    ----------
    directions : np.ndarray
        Brick directions (+1 or -1).
    window : int
        Rolling window size.  Must be ≥ 2.
    min_periods : int or None
        Minimum transitions for valid estimate.

    Returns
    -------
    pUU, pUD, pDU, pDD : np.ndarray (float64)
        Each array has the same length as ``directions``.
    """
    pUU, pDD = markov_stickiness(directions, window, min_periods=min_periods)
    pUD = np.where(np.isfinite(pUU), 1.0 - pUU, np.nan)
    pDU = np.where(np.isfinite(pDD), 1.0 - pDD, np.nan)
    return pUU, pUD, pDU, pDD


# ══════════════════════════════════════════════════════════════════════════════
# Permutation Entropy — non-linear regime filter
# ══════════════════════════════════════════════════════════════════════════════


def permutation_entropy(
    sequence: np.ndarray,
    order: int = 3,
    *,
    delay: int = 1,
) -> float:
    """Normalized permutation entropy in [0, 1] for a 1D sequence.

    Notes
    -----
    - Returns ``np.nan`` when there is not enough data.
    - Uses a stable tiny index tie-breaker so repeated values (common with
      Renko direction sequences) still produce deterministic ordinal patterns.
    """
    x = np.asarray(sequence, dtype=np.float64).ravel()
    n = len(x)
    if order < 2:
        raise ValueError(f"order must be >= 2, got {order}")
    if delay < 1:
        raise ValueError(f"delay must be >= 1, got {delay}")

    span = (order - 1) * delay + 1
    if n < span:
        return float("nan")

    n_patterns = n - span + 1
    patterns = {}
    tie_eps = 1e-12 * np.arange(order, dtype=np.float64)

    for i in range(n_patterns):
        window = x[i : i + span : delay]
        ranks = np.argsort(window + tie_eps, kind="mergesort")
        key = tuple(int(v) for v in ranks.tolist())
        patterns[key] = patterns.get(key, 0) + 1

    probs = np.array(list(patterns.values()), dtype=np.float64) / float(n_patterns)
    shannon = -np.sum(probs * np.log2(probs))
    max_h = math.log2(math.factorial(order))
    if max_h <= 0.0:
        return 0.0
    return float(np.clip(shannon / max_h, 0.0, 1.0))


# ══════════════════════════════════════════════════════════════════════════════
# Composite entry rule evaluation
# ══════════════════════════════════════════════════════════════════════════════


def evaluate_entry(
    direction: int,
    flip_rate_val: float,
    pUU: float,
    pDD: float,
    fliprate_threshold: float,
    markov_threshold: float,
) -> bool:
    """
    Evaluate whether a flip entry should be taken.

    Implements the canonical entry rule from the design spec (§6.2):

    .. code-block:: python

        allow_entry = (
            flip_rate <= fliprate_threshold
            and (
                (direction == +1 and pUU >= markov_threshold)
                or
                (direction == -1 and pDD >= markov_threshold)
            )
        )

    Parameters
    ----------
    direction : int
        Current brick direction (+1 for long, -1 for short).
    flip_rate_val : float
        Current FlipRate value.
    pUU : float
        Current Markov P(Up→Up).
    pDD : float
        Current Markov P(Down→Down).
    fliprate_threshold : float
        Maximum FlipRate to allow entry (lower = stricter).
    markov_threshold : float
        Minimum Markov stickiness to allow entry (higher = stricter).

    Returns
    -------
    bool
        True if the entry passes all filter gates.
    """
    # NaN values → reject (data not yet warm)
    if not np.isfinite(flip_rate_val) or not np.isfinite(pUU) or not np.isfinite(pDD):
        return False

    # Primary filter: FlipRate must be low enough
    if flip_rate_val > fliprate_threshold:
        return False

    # Secondary filter: Markov stickiness must be high enough
    if direction == 1:
        if pUU < markov_threshold:
            return False
    elif direction == -1:
        if pDD < markov_threshold:
            return False
    else:
        return False  # invalid direction

    return True


def evaluate_entries_vectorized(
    directions: np.ndarray,
    flip_rates: np.ndarray,
    pUU_arr: np.ndarray,
    pDD_arr: np.ndarray,
    fliprate_threshold: float,
    markov_threshold: float,
    *,
    is_flip: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Vectorized entry evaluation across all bricks.

    Parameters
    ----------
    directions : np.ndarray
        Brick directions (+1 / -1).
    flip_rates : np.ndarray
        FlipRate values (same length as directions).
    pUU_arr : np.ndarray
        Markov P(Up→Up) values.
    pDD_arr : np.ndarray
        Markov P(Down→Down) values.
    fliprate_threshold : float
        Maximum FlipRate for entry.
    markov_threshold : float
        Minimum Markov stickiness for entry.
    is_flip : np.ndarray or None
        Boolean mask where True = colour flip occurred.  If None,
        flips are computed from ``directions``.

    Returns
    -------
    np.ndarray (bool)
        Boolean mask: True at indices where entry is allowed.
    """
    dirs = np.asarray(directions, dtype=np.int8)
    fr = np.asarray(flip_rates, dtype=np.float64)
    pu = np.asarray(pUU_arr, dtype=np.float64)
    pd_arr = np.asarray(pDD_arr, dtype=np.float64)
    n = len(dirs)

    # Flip detection
    if is_flip is None:
        flips = np.zeros(n, dtype=bool)
        if n >= 2:
            flips[1:] = dirs[1:] != dirs[:-1]
    else:
        flips = np.asarray(is_flip, dtype=bool)

    # All values must be finite
    finite_mask = np.isfinite(fr) & np.isfinite(pu) & np.isfinite(pd_arr)

    # FlipRate gate
    fr_ok = fr <= fliprate_threshold

    # Markov gate: direction-dependent
    markov_ok = np.where(
        dirs == 1,
        pu >= markov_threshold,
        pd_arr >= markov_threshold,
    )

    return flips & finite_mask & fr_ok & markov_ok
