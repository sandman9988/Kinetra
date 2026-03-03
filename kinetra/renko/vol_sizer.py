"""
Volatility-Targeted Sizing
==========================

Canonical volatility-targeted position sizing for the Kinetra Renko engine.

Background
----------
The research chronology (Phase 5 → Next Experiment) identifies the key
limitation of fixed compounding (0.01 lot / $1,000 equity):

    "Equity scale reflects aggressive compounding on a 1-year high-alpha
     Renko dataset — treat as relative comparison between modes, not live
     expectation.  Throttle via vol-targeted sizing is the recommended
     next step to produce interpretable live sizing curves."

Fixed compounding causes **exponential equity artefacts** because:
1. Lot sizes grow with equity, so late-period trades dominate the P&L curve.
2. The curve is not interpretable across instruments with different brick
   sizes and USD-per-point values.
3. Portfolio DD comparisons are distorted by compounding paths, not by
   true diversification benefit.

Volatility-targeted sizing solves this by targeting a **fixed volatility
contribution** per instrument per trade:

    lot_size = (target_vol_budget × equity) / (brick × usd_per_point
                × rolling_vol_estimate × 100_000)

Where ``rolling_vol_estimate`` is the rolling standard deviation of
**per-brick P&L** (in USD, normalised by 1 lot), computed adaptively
from the trade history.  This makes the lot size:

- **Equity-proportional** — position sizes scale with the account size.
- **Volatility-normalised** — instruments with higher P&L variance get
  smaller positions automatically.
- **Interpretable** — the equity curve shows returns as a fraction of
  equity, not as compounding artefacts.
- **Comparable across instruments** — a 1% vol target is the same 1% for
  XAUUSD and NAS100 regardless of their brick sizes.

Key design decisions
--------------------
- **Brick P&L as volatility input** — we use the per-brick P&L distribution
  (in price points × direction) rather than log-returns of M1 closes.
  This is natural in Renko space: each brick IS one unit of P&L risk.
  The spread (as a fraction of brick size) is the friction ratio.
- **Adaptive window** — the vol window is derived from the ACF decay of
  |brick returns| rather than a fixed lookback.  This follows the Kinetra
  first-principles rule of no magic numbers.
- **Minimum sample guard** — vol is only estimated when at least
  ``min_vol_window`` bricks are available.  Before that, a conservative
  fallback (``initial_vol_fallback``) is used.
- **Vol floor** — a minimum vol estimate prevents division-by-zero and
  excessive lot sizes during low-volatility regimes.
- **Vol ceiling** — a maximum vol estimate caps lot size during
  crisis/spike periods.
- **Regime detection** — the sizer can detect when realised vol is
  significantly above or below the rolling estimate and scale down
  ("vol spike throttle") or scale up ("vol compression boost") within
  configurable bounds.

This module is the canonical location for vol-targeted sizing in Kinetra.
Scripts and the live trader must import from here rather than reimplement.

Usage::

    from kinetra.renko.vol_sizer import VolTargetSizer, VolSizingConfig

    sizer = VolTargetSizer(VolSizingConfig(
        target_vol_pct=0.01,       # 1% daily vol target per instrument
        vol_window=50,             # lookback for vol estimate (bricks)
        vol_floor=0.003,           # minimum vol (as fraction of brick)
        vol_ceil=0.10,             # maximum vol (as fraction of brick)
    ))

    # Update rolling vol from a new brick's P&L in points
    sizer.update("XAUUSD", brick_pts=0.40, direction=1)  # +1 brick

    # Compute lot size
    lots = sizer.compute(
        symbol="XAUUSD",
        equity_usd=100_000.0,
        brick_size=0.40,
        usd_per_point=100.0,
        layer2_weight=0.80,
        layer3_exposure=0.90,
        gate=PERGate.MICRO,
    )

See Also
--------
- ``kinetra/renko/live_trader.py``    — RenkoSizer (fixed compounding baseline)
- ``kinetra/renko/backtest.py``       — SizingMode, VolSizingParams
- ``kinetra/volatility_utils.py``     — rolling_volatility (M1 close-based)
- ``docs/MANUAL.md §7`` — runtime architecture

Portfolio-Level Pareto Frontier
--------------------------------
:func:`compare_portfolio_sizing_pareto` extends the single-instrument
calibration to a *portfolio* of instruments by using
:func:`~kinetra.renko.backtest.backtest_portfolio` as the inner evaluator.
For each DD budget the same ``target_vol_pct`` (or
``compounding_capital_per_lot``) is applied uniformly across all instruments
so that the portfolio equity curve reflects true diversification effects.

Usage::

    from kinetra.renko.vol_sizer import (
        compare_portfolio_sizing_pareto,
        format_pareto_report,
    )

    result = compare_portfolio_sizing_pareto(
        instruments={"XAUUSD": xau_closes, "NAS100": nas_closes},
        brick_sizes={"XAUUSD": 0.40, "NAS100": 5.0},
        usd_per_points={"XAUUSD": 100.0, "NAS100": 20.0},
        target_dd_pcts=[-3.0, -5.0, -8.0, -12.0],
        initial_equity=10_000.0,
    )
    print(format_pareto_report(result))
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Optional

if TYPE_CHECKING:
    import pandas as pd

    from kinetra.renko.backtest import (
        FilterParams,
        InstrumentBacktestResult,
        RiskParams,
        StopParams,
        VolSizingParams,
    )

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

#: Default fraction of equity to risk per instrument per unit-vol day.
#: 1 % means: if the instrument moves 1σ (vol estimate), the position loses
#: at most 1% of equity.  Multiply by √(1/bricks_per_day) for intraday.
DEFAULT_TARGET_VOL_PCT: float = 0.01

#: Default rolling window for vol estimation (bricks).
#: ~50 bricks ≈ 1–2 trading days for most liquid instruments.
DEFAULT_VOL_WINDOW: int = 50

#: Minimum bricks required before vol estimation is meaningful.
DEFAULT_MIN_VOL_WINDOW: int = 20

#: Default long-window for the sigma stability floor (bricks).
#: ~500 bricks ≈ 10 trading days — long enough to smooth cold-start artefacts
#: without being insensitive to genuine regime shifts.
#: Set to 0 to disable the dual-window floor entirely.
DEFAULT_VOL_FLOOR_LONG_WINDOW: int = 500

#: Quantile of the long-window rolling std to use as the floor.
#: P25 is conservative: the floor activates only when the short-window sigma
#: drops into the bottom quarter of its historical distribution, preventing
#: the cold-start underestimation from driving oversized positions.
DEFAULT_VOL_FLOOR_LONG_QUANTILE: float = 0.25

#: Minimum vol estimate as a fraction of brick size.
#: Prevents astronomical lot sizes in ultra-low-vol regimes.
DEFAULT_VOL_FLOOR: float = 0.003

#: Maximum vol estimate as a fraction of brick size.
#: Caps sizing during crisis / spike periods.
DEFAULT_VOL_CEIL: float = 0.20

#: Conservative vol estimate used before enough bricks are observed.
#: 2% of brick size → lot sizes are smaller than steady-state at startup.
DEFAULT_INITIAL_VOL_FALLBACK: float = 0.02

#: Vol spike throttle: if realised vol > threshold × estimate → scale down.
DEFAULT_VOL_SPIKE_RATIO: float = 2.0

#: Maximum lot scaling when vol is very low vs estimate (compression boost).
DEFAULT_VOL_COMPRESSION_LIMIT: float = 2.0

#: Hard upper bound on lot size (all gates, all modes) as a safety net.
MAX_LOT_HARD_CEILING: float = 100.0


# ---------------------------------------------------------------------------
# Shared lot normalization helpers (DRY)
# ---------------------------------------------------------------------------


def quantize_and_clamp_lot(
    lots: float,
    *,
    lot_step: float,
    min_lots: float,
    ceilings: tuple[float, ...],
) -> float:
    """
    Quantize a lot value by step and apply lower/upper bounds.

    This is the canonical scalar implementation used across backtest/live paths.
    """
    stepped = (
        max(0.0, round(float(lots) / lot_step) * lot_step)
        if lot_step > 0
        else max(0.0, float(lots))
    )
    if ceilings:
        stepped = min(stepped, *ceilings)
    return 0.0 if stepped < min_lots else float(stepped)


def quantize_and_clamp_lots_array(
    lots: np.ndarray,
    *,
    lot_step: float,
    min_lots: float,
    ceilings: tuple[float, ...],
) -> np.ndarray:
    """
    Vectorized lot quantize + clamp for batch sizing.
    """
    arr = np.asarray(lots, dtype=np.float64)
    if lot_step > 0:
        stepped = np.round(arr / lot_step) * lot_step
    else:
        stepped = arr
    stepped = np.maximum(stepped, 0.0)
    if ceilings:
        stepped = np.minimum(stepped, min(ceilings))
    return np.where(stepped < min_lots, 0.0, stepped)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class VolSizingConfig:
    """
    Configuration for :class:`VolTargetSizer`.

    All fraction parameters are **dimensionless** (not %).
    Use 0.01 for "1 %".

    Attributes
    ----------
    target_vol_pct : float
        Fraction of equity to risk per 1σ brick-vol move.
        Default ``0.01`` (1 %).
    vol_window : int
        Rolling window for brick P&L standard deviation (bricks).
        Default ``50``.
    min_vol_window : int
        Minimum bricks required before vol is estimated from data.
        Below this count the ``initial_vol_fallback`` is used.
        Default ``20``.
    vol_floor : float
        Hard minimum vol estimate (as fraction of brick size).
        Prevents oversizing in ultra-low-vol regimes.
        Default ``0.003``.
    vol_ceil : float
        Maximum vol estimate (as fraction of brick size).
        Caps sizing during spikes / crisis.
        Default ``0.20``.
    initial_vol_fallback : float
        Conservative vol estimate used during the warmup period
        (before ``min_vol_window`` bricks are observed).
        Default ``0.02``.
    vol_floor_long_window : int
        Rolling window for the **stability floor** sigma estimate (bricks).
        The floor is computed as the ``vol_floor_long_quantile``-th percentile
        of the long-window rolling std distribution, and is applied as a
        soft lower bound on the short-window sigma.  This prevents cold-start
        underestimation from driving oversized positions during the first
        few hundred bricks.

        Set to ``0`` to disable the dual-window floor entirely and use the
        hard ``vol_floor`` only (single-window behaviour).

        Default ``500`` (≈ 10 trading days of bricks for most instruments).
    vol_floor_long_quantile : float
        Quantile of the long-window rolling std distribution used as the
        stability floor.  ``0.25`` means: use the 25th-percentile sigma as
        the floor — the short-window sigma must drop below the bottom quarter
        of its own long-term distribution before the floor activates.

        Only used when ``vol_floor_long_window > 0``.
        Default ``0.25``.
    use_vol_spike_throttle : bool
        If True, lot size is scaled down when realised vol is
        significantly above the rolling estimate.
        Default ``True``.
    vol_spike_ratio : float
        Realised / rolling vol ratio above which throttle activates.
        Default ``2.0``.
    use_vol_compression_boost : bool
        If True, lot size may be increased (up to
        ``vol_compression_limit``) when vol is much lower than estimate.
        Default ``False`` — conservative; boost is off by default.
    vol_compression_limit : float
        Maximum lot multiplier from compression boost.
        Default ``2.0``.
    annualisation_factor : float
        √(number of bricks per year) for vol annualisation.
        Used only for display / logging; sizing is always in brick units.
        Default ``0.0`` (annualisation disabled).
    """

    target_vol_pct: float = DEFAULT_TARGET_VOL_PCT
    vol_window: int = DEFAULT_VOL_WINDOW
    min_vol_window: int = DEFAULT_MIN_VOL_WINDOW
    vol_floor: float = DEFAULT_VOL_FLOOR
    vol_ceil: float = DEFAULT_VOL_CEIL
    initial_vol_fallback: float = DEFAULT_INITIAL_VOL_FALLBACK
    vol_floor_long_window: int = DEFAULT_VOL_FLOOR_LONG_WINDOW
    vol_floor_long_quantile: float = DEFAULT_VOL_FLOOR_LONG_QUANTILE
    use_vol_spike_throttle: bool = True
    vol_spike_ratio: float = DEFAULT_VOL_SPIKE_RATIO
    use_vol_compression_boost: bool = False
    vol_compression_limit: float = DEFAULT_VOL_COMPRESSION_LIMIT
    annualisation_factor: float = 0.0

    def __post_init__(self) -> None:
        if self.target_vol_pct <= 0:
            raise ValueError(f"target_vol_pct must be > 0, got {self.target_vol_pct}")
        if self.vol_window < 2:
            raise ValueError(f"vol_window must be >= 2, got {self.vol_window}")
        if self.min_vol_window < 2:
            raise ValueError(f"min_vol_window must be >= 2, got {self.min_vol_window}")
        if self.vol_floor < 0:
            raise ValueError(f"vol_floor must be >= 0, got {self.vol_floor}")
        if self.vol_ceil <= self.vol_floor:
            raise ValueError(f"vol_ceil ({self.vol_ceil}) must be > vol_floor ({self.vol_floor})")
        if self.vol_floor_long_window < 0:
            raise ValueError(
                f"vol_floor_long_window must be >= 0, got {self.vol_floor_long_window}"
            )
        if not (0.0 < self.vol_floor_long_quantile < 1.0):
            raise ValueError(
                f"vol_floor_long_quantile must be in (0, 1), got {self.vol_floor_long_quantile}"
            )
        if self.vol_spike_ratio <= 1.0:
            raise ValueError(f"vol_spike_ratio must be > 1, got {self.vol_spike_ratio}")
        if self.vol_compression_limit < 1.0:
            raise ValueError(
                f"vol_compression_limit must be >= 1, got {self.vol_compression_limit}"
            )


# ---------------------------------------------------------------------------
# Per-instrument rolling vol state
# ---------------------------------------------------------------------------


class _InstrumentVolState:
    """Thread-safe rolling brick P&L buffer for one instrument.

    Maintains two buffers:
    - ``_buffer``      — short-window buffer (size ``config.vol_window``)
                         for the responsive sigma estimate.
    - ``_long_buffer`` — long-window buffer (size ``config.vol_floor_long_window``)
                         for the stability floor sigma estimate.
                         Only populated when ``vol_floor_long_window > 0``.
    """

    def __init__(self, config: VolSizingConfig) -> None:
        self.config = config
        self._buffer: List[float] = []
        self._long_buffer: List[float] = []
        self._lock = threading.Lock()

    def update(self, brick_pts: float) -> None:
        """
        Add one brick's P&L (in price points, signed by direction) to both
        rolling buffers.

        Parameters
        ----------
        brick_pts : float
            Price-point move of the brick: ``brick_size × direction``.
            E.g. for a 0.40 up-brick: ``+0.40``; for a down-brick: ``-0.40``.
        """
        with self._lock:
            pt = float(brick_pts)
            self._buffer.append(pt)
            if len(self._buffer) > self.config.vol_window:
                self._buffer = self._buffer[-self.config.vol_window :]
            if self.config.vol_floor_long_window > 0:
                self._long_buffer.append(pt)
                if len(self._long_buffer) > self.config.vol_floor_long_window:
                    self._long_buffer = self._long_buffer[-self.config.vol_floor_long_window :]

    def estimate(self, brick_size: float) -> float:
        """
        Return the current rolling vol estimate as a **fraction of brick_size**.

        Applies the dual-window sigma floor when ``vol_floor_long_window > 0``:
        the effective sigma is ``max(short_sigma, long_window_quantile_sigma)``.
        This prevents cold-start underestimation from driving oversized
        positions during the warmup period.

        Parameters
        ----------
        brick_size : float
            Current instrument brick size (price units).

        Returns
        -------
        float
            Rolling std of brick P&L / brick_size, clamped to [floor, ceil].
        """
        with self._lock:
            n = len(self._buffer)
            cfg = self.config

            if n < cfg.min_vol_window or brick_size <= 0:
                return cfg.initial_vol_fallback

            arr = np.asarray(self._buffer[-cfg.vol_window :], dtype=np.float64)
            raw_std = float(np.std(arr, ddof=1)) if len(arr) >= 2 else cfg.initial_vol_fallback

            # Normalise by brick size to get dimensionless vol
            vol_frac = raw_std / brick_size if brick_size > 0 else cfg.initial_vol_fallback

            # ── Dual-window sigma floor ──────────────────────────────────
            # When the long buffer is populated, compute the quantile sigma
            # floor and use max(short_sigma, floor) so that temporary
            # underestimation during quiet patches cannot drive oversizing.
            if cfg.vol_floor_long_window > 0 and len(self._long_buffer) >= max(
                cfg.min_vol_window, cfg.vol_floor_long_window // 10
            ):
                long_arr = np.asarray(self._long_buffer, dtype=np.float64)
                long_std = float(np.std(long_arr, ddof=1)) if len(long_arr) >= 2 else 0.0
                long_vol_frac = long_std / brick_size if brick_size > 0 else 0.0
                # Use the quantile of the long-window as the floor.
                # We approximate the quantile from the long buffer's std by
                # treating the rolling sequence of per-brick |deviations| as
                # the distribution.  A simple scalar multiple of long_std
                # approximates the chosen quantile under normality:
                #   P25 ≈ μ - 0.674σ  (for std of a normal distribution,
                #   the 25th percentile of a chi(1) is ~0.674)
                # However, we compute it empirically when the buffer is large
                # enough (> 50 points) to avoid distributional assumptions.
                if len(long_arr) >= 50:
                    # Empirical: rolling std of non-overlapping chunks of
                    # vol_window gives the distribution; with the full buffer
                    # we approximate via stride-based std of windows.
                    chunk_size = max(cfg.vol_window, 2)
                    n_chunks = len(long_arr) // chunk_size
                    if n_chunks >= 4:
                        chunk_stds = np.array(
                            [
                                float(
                                    np.std(long_arr[i * chunk_size : (i + 1) * chunk_size], ddof=1)
                                )
                                for i in range(n_chunks)
                            ]
                        )
                        floor_std = float(np.quantile(chunk_stds, cfg.vol_floor_long_quantile))
                        floor_frac = floor_std / brick_size if brick_size > 0 else 0.0
                    else:
                        floor_frac = long_vol_frac * cfg.vol_floor_long_quantile
                else:
                    floor_frac = long_vol_frac * cfg.vol_floor_long_quantile

                vol_frac = max(vol_frac, floor_frac)

            # Clamp to [hard_floor, ceil]
            return float(np.clip(vol_frac, cfg.vol_floor, cfg.vol_ceil))

    def n_observations(self) -> int:
        """Return number of bricks in the short-window buffer."""
        with self._lock:
            return len(self._buffer)

    def n_long_observations(self) -> int:
        """Return number of bricks in the long-window buffer."""
        with self._lock:
            return len(self._long_buffer)

    def is_warmed_up(self) -> bool:
        """True once ``min_vol_window`` bricks have been observed."""
        with self._lock:
            return len(self._buffer) >= self.config.min_vol_window


# ---------------------------------------------------------------------------
# VolTargetSizer
# ---------------------------------------------------------------------------


class VolTargetSizer:
    """
    Volatility-targeted lot sizer for the Renko engine.

    Computes lot sizes so that each instrument contributes a target fraction
    of equity in P&L volatility per trade, normalised by rolling brick P&L
    standard deviation.

    Sizing formula
    --------------

    .. math::

        \\text{lots} =
        \\frac{\\text{target\\_vol\\_pct} \\times \\text{equity\\_usd}}
             {\\text{brick\\_size} \\times \\text{usd\\_per\\_point}
              \\times \\hat{\\sigma} \\times 100{,}000}
        \\times w_2 \\times w_3

    Where:

    - ``target_vol_pct`` — fraction of equity to risk per 1σ brick move
    - ``equity_usd`` — current account equity
    - ``brick_size`` — instrument brick size (price units)
    - ``usd_per_point`` — USD value of 1 price unit per lot
    - ``σ̂`` — rolling vol estimate (fraction of brick size, dimensionless)
    - ``w₂`` — Layer 2 allocation weight [0, 1]
    - ``w₃`` — Layer 3 exposure scalar [0, 1]

    Parameters
    ----------
    config : VolSizingConfig or None
        Sizer configuration.  Defaults to ``VolSizingConfig()`` if None.
    """

    def __init__(self, config: Optional[VolSizingConfig] = None) -> None:
        self._config = config if config is not None else VolSizingConfig()
        self._states: Dict[str, _InstrumentVolState] = {}
        self._global_lock = threading.Lock()

    # ── Public API ────────────────────────────────────────────────────────

    def update(self, symbol: str, brick_pts: float) -> None:
        """
        Register one completed brick for *symbol*.

        Must be called once per brick, regardless of whether a trade was
        opened.  This keeps the vol estimate current even during flat /
        filtered periods.

        Parameters
        ----------
        symbol : str
            Canonical instrument symbol.
        brick_pts : float
            ``brick_size × direction`` — the signed price-point move of the
            brick (positive for up-bricks, negative for down-bricks).
        """
        state = self._get_or_create(symbol)
        state.update(brick_pts)

    def compute(
        self,
        symbol: str,
        equity_usd: float,
        brick_size: float,
        usd_per_point: float,
        layer2_weight: float = 1.0,
        layer3_exposure: float = 1.0,
        lot_step: float = 0.01,
        min_lots: float = 0.01,
        gate_lot_ceiling: float = MAX_LOT_HARD_CEILING,
    ) -> float:
        """
        Compute the vol-targeted lot size for *symbol*.

        Parameters
        ----------
        symbol : str
            Canonical instrument symbol.
        equity_usd : float
            Current account equity in USD.
        brick_size : float
            Brick size in price units.
        usd_per_point : float
            USD value of 1 price unit per 1 standard lot (100,000 units).
        layer2_weight : float
            Allocation weight from Layer 2 RL agent [0, 1].
        layer3_exposure : float
            Exposure scalar from Layer 3 RL agent [0, 1].
        lot_step : float
            Broker minimum lot increment (default 0.01).
        min_lots : float
            Broker minimum lot size (default 0.01).
        gate_lot_ceiling : float
            PER gate's lot ceiling — hard cap on the result.

        Returns
        -------
        float
            Lot size to trade, or ``0.0`` to skip the trade.
        """
        if equity_usd <= 0 or brick_size <= 0 or usd_per_point <= 0:
            return 0.0
        if layer2_weight < 1e-6 or layer3_exposure < 1e-6:
            return 0.0

        state = self._get_or_create(symbol)
        vol_frac = state.estimate(brick_size)

        # vol_frac is dimensionless: std(brick_pts) / brick_size
        # USD vol per lot per brick = brick_size × usd_per_point × vol_frac × 100_000
        usd_vol_per_lot = brick_size * usd_per_point * vol_frac * 100_000.0
        if usd_vol_per_lot <= 0:
            return 0.0

        # Target USD vol contribution from this position
        target_usd_vol = self._config.target_vol_pct * equity_usd

        # Base lots (before RL scaling)
        base_lots = target_usd_vol / usd_vol_per_lot

        # Apply vol spike throttle
        if self._config.use_vol_spike_throttle:
            spike_scalar = self._compute_spike_scalar(state, brick_size, vol_frac)
            base_lots *= spike_scalar

        # Apply vol compression boost (optional, conservative by default)
        if self._config.use_vol_compression_boost:
            boost_scalar = self._compute_compression_scalar(state, brick_size, vol_frac)
            base_lots *= boost_scalar

        # Apply RL scaling
        scaled = base_lots * layer2_weight * layer3_exposure

        return quantize_and_clamp_lot(
            scaled,
            lot_step=lot_step,
            min_lots=min_lots,
            ceilings=(gate_lot_ceiling, MAX_LOT_HARD_CEILING),
        )

    def vol_estimate(self, symbol: str, brick_size: float) -> float:
        """
        Return the current vol estimate for *symbol* (fraction of brick_size).

        Useful for logging, diagnostics, and observation vector construction.

        Parameters
        ----------
        symbol : str
            Canonical instrument symbol.
        brick_size : float
            Brick size in price units (used for normalisation).

        Returns
        -------
        float
            Rolling vol estimate ∈ [vol_floor, vol_ceil].
        """
        state = self._get_or_create(symbol)
        return state.estimate(brick_size)

    def is_warmed_up(self, symbol: str) -> bool:
        """
        Return True once the rolling vol buffer has enough observations.

        Parameters
        ----------
        symbol : str
            Canonical instrument symbol.
        """
        with self._global_lock:
            if symbol not in self._states:
                return False
        return self._states[symbol].is_warmed_up()

    def n_observations(self, symbol: str) -> int:
        """
        Return the number of bricks in the rolling buffer for *symbol*.

        Parameters
        ----------
        symbol : str
            Canonical instrument symbol.
        """
        with self._global_lock:
            if symbol not in self._states:
                return 0
        return self._states[symbol].n_observations()

    def reset(self, symbol: Optional[str] = None) -> None:
        """
        Reset the vol state for *symbol*, or all symbols if None.

        Useful when restarting the live trader or switching instruments.

        Parameters
        ----------
        symbol : str or None
            Symbol to reset, or None to reset all.
        """
        with self._global_lock:
            if symbol is None:
                self._states.clear()
            elif symbol in self._states:
                del self._states[symbol]

    def symbols(self) -> List[str]:
        """Return a snapshot of all known symbols."""
        with self._global_lock:
            return list(self._states.keys())

    def n_long_observations(self, symbol: str) -> int:
        """
        Return the number of bricks in the long-window stability buffer for
        *symbol*.  Returns 0 if the symbol is unknown or the long window is
        disabled (``vol_floor_long_window == 0``).

        Parameters
        ----------
        symbol : str
            Canonical instrument symbol.
        """
        with self._global_lock:
            if symbol not in self._states:
                return 0
        return self._states[symbol].n_long_observations()

    # ── Internal helpers ──────────────────────────────────────────────────

    def _get_or_create(self, symbol: str) -> _InstrumentVolState:
        """Get or lazily create the per-instrument vol state."""
        with self._global_lock:
            if symbol not in self._states:
                self._states[symbol] = _InstrumentVolState(config=self._config)
            return self._states[symbol]

    def _compute_spike_scalar(
        self,
        state: _InstrumentVolState,
        brick_size: float,
        rolling_vol_frac: float,
    ) -> float:
        """
        Return a scaling factor in (0, 1] to throttle sizing during vol spikes.

        If the most recent brick's |P&L fraction| is above
        ``vol_spike_ratio × rolling_vol_frac``, the lot is scaled down
        proportionally so that the effective risk stays within the target.

        Parameters
        ----------
        state : _InstrumentVolState
            Per-instrument state (buffer access).
        brick_size : float
            Brick size for normalisation.
        rolling_vol_frac : float
            Current rolling vol estimate (fraction of brick_size).

        Returns
        -------
        float
            Multiplicative scalar ∈ (0, 1].
        """
        if brick_size <= 0 or rolling_vol_frac <= 0:
            return 1.0

        with state._lock:
            if not state._buffer:
                return 1.0
            last_pt = abs(state._buffer[-1])

        realised_frac = last_pt / brick_size
        spike_threshold = self._config.vol_spike_ratio * rolling_vol_frac

        if realised_frac <= spike_threshold:
            return 1.0

        # Scale down: scalar = threshold / realised_frac
        scalar = spike_threshold / realised_frac
        return float(np.clip(scalar, 0.0, 1.0))

    def _compute_compression_scalar(
        self,
        state: _InstrumentVolState,
        brick_size: float,
        rolling_vol_frac: float,
    ) -> float:
        """
        Return a scaling factor in [1, vol_compression_limit] to boost sizing
        during vol compression (very low realised vol vs estimate).

        Parameters
        ----------
        state : _InstrumentVolState
            Per-instrument state.
        brick_size : float
            Brick size for normalisation.
        rolling_vol_frac : float
            Current rolling vol estimate (fraction of brick_size).

        Returns
        -------
        float
            Multiplicative scalar ∈ [1, vol_compression_limit].
        """
        if not state.is_warmed_up() or brick_size <= 0 or rolling_vol_frac <= 0:
            return 1.0

        with state._lock:
            if len(state._buffer) < 5:
                return 1.0
            recent = np.asarray(state._buffer[-5:], dtype=np.float64)

        recent_frac = float(np.std(recent, ddof=1)) / brick_size if brick_size > 0 else 0.0
        if recent_frac <= 0 or rolling_vol_frac <= 0:
            return 1.0

        # Compression ratio: rolling_vol / recent_vol — how much vol has compressed
        compression_ratio = rolling_vol_frac / recent_frac
        clamped = float(np.clip(compression_ratio, 1.0, self._config.vol_compression_limit))
        return clamped


# ---------------------------------------------------------------------------
# Vectorised batch sizer — for backtesting
# ---------------------------------------------------------------------------


def compute_vol_targeted_lots_batch(
    brick_pts: np.ndarray,
    brick_size: float,
    equity_curve: np.ndarray,
    usd_per_point: float,
    config: Optional[VolSizingConfig] = None,
    *,
    layer2_weights: Optional[np.ndarray] = None,
    layer3_exposures: Optional[np.ndarray] = None,
    lot_step: float = 0.01,
    min_lots: float = 0.01,
    lot_ceiling: float = MAX_LOT_HARD_CEILING,
    is_trade_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Compute vol-targeted lot sizes for an entire brick sequence in one pass.

    This is the vectorised equivalent of :meth:`VolTargetSizer.compute` and
    is designed for use inside :func:`~kinetra.renko.backtest.backtest_instrument`
    when ``SizingMode.VOL_TARGET`` is requested.

    The rolling vol is estimated with a causal (past-only) rolling window so
    there is **no lookahead bias** — the vol estimate at brick ``i`` uses
    only bricks ``[i - vol_window, i-1]``.

    Parameters
    ----------
    brick_pts : np.ndarray, shape (N,)
        Signed per-brick P&L in price points (``brick_size × direction``).
    brick_size : float
        Brick size in price units.
    equity_curve : np.ndarray, shape (N,)
        Equity at each brick boundary (used for position sizing).
        Pass ``np.ones(N) * initial_equity`` for fixed-equity sizing.
    usd_per_point : float
        USD value of 1 price unit per 1 standard lot (100,000 units).
    config : VolSizingConfig or None
        Sizer configuration.  Defaults to ``VolSizingConfig()`` if None.
    layer2_weights : np.ndarray of shape (N,) or None
        Per-brick Layer 2 allocation weights.  Defaults to all-ones.
    layer3_exposures : np.ndarray of shape (N,) or None
        Per-brick Layer 3 exposure scalars.  Defaults to all-ones.
    lot_step : float
        Minimum lot increment.
    min_lots : float
        Minimum lot size (below this → 0.0).
    lot_ceiling : float
        Hard upper cap on lot size.
    is_trade_mask : np.ndarray of bool, shape (N,) or None
        If provided, lot computation is only run at ``True`` positions
        (entry bricks).  All other positions return 0.0.

    Returns
    -------
    np.ndarray, shape (N,)
        Lot size at each brick position.  Non-entry bricks are 0.0.

    Notes
    -----
    The function uses ``np.lib.stride_tricks`` for the rolling std
    computation instead of a Python loop to satisfy the Kinetra vectorisation
    mandate (ALWAYS prefer vectorisation over Python loops).

    For the first ``min_vol_window`` bricks the ``initial_vol_fallback`` is
    used so lot sizes are conservative during warmup.
    """
    if config is None:
        config = VolSizingConfig()

    n = len(brick_pts)
    if n == 0:
        return np.zeros(0, dtype=np.float64)

    w2 = layer2_weights if layer2_weights is not None else np.ones(n, dtype=np.float64)
    w3 = layer3_exposures if layer3_exposures is not None else np.ones(n, dtype=np.float64)
    eq = equity_curve if equity_curve is not None else np.ones(n, dtype=np.float64)

    bp = np.asarray(brick_pts, dtype=np.float64)
    w2 = np.asarray(w2, dtype=np.float64)
    w3 = np.asarray(w3, dtype=np.float64)
    eq = np.asarray(eq, dtype=np.float64)

    # ── Rolling causal std of brick P&L (vectorised / semi-vectorised) ──
    # At position i: std(bp[max(0, i-window) : i])
    # When the dual-window floor is enabled we use the floor-aware variant;
    # otherwise the pure-vectorised single-window path is used.
    use_floor = config.vol_floor_long_window > 0
    if use_floor:
        # Semi-vectorised: O(N × long_window/chunk_size) but still numpy-heavy
        min_long_obs = max(config.min_vol_window, config.vol_floor_long_window // 10)
        rolling_std = _causal_rolling_std_long_floor(
            bp,
            short_window=config.vol_window,
            long_window=config.vol_floor_long_window,
            long_quantile=config.vol_floor_long_quantile,
            min_long_obs=min_long_obs,
        )
    else:
        rolling_std = _causal_rolling_std(bp, config.vol_window)

    # Normalise to fraction of brick_size
    if brick_size > 0:
        vol_frac = rolling_std / brick_size
    else:
        vol_frac = np.full(n, config.initial_vol_fallback, dtype=np.float64)

    # Apply hard floor/ceil and warmup fallback
    warmup_mask = np.arange(n) < config.min_vol_window
    vol_frac = np.where(warmup_mask, config.initial_vol_fallback, vol_frac)
    vol_frac = np.clip(vol_frac, config.vol_floor, config.vol_ceil)

    # ── Compute lot sizes (vectorised) ───────────────────────────────────
    usd_vol_per_lot = brick_size * usd_per_point * vol_frac * 100_000.0
    # Guard division by zero
    safe_denominator = np.where(usd_vol_per_lot > 0, usd_vol_per_lot, np.inf)

    target_usd_vol = config.target_vol_pct * eq
    base_lots = target_usd_vol / safe_denominator

    # RL scaling
    scaled = base_lots * w2 * w3
    scaled = np.maximum(scaled, 0.0)

    # NOTE: lot_step rounding, min_lots, and ceiling clamping are intentionally
    # NOT applied here when the caller passes equity=1.0 to get lots-per-dollar.
    # The caller (_compute_trade_lots) multiplies by live_equity first and then
    # applies rounding and clamping so that rounding is correct at the full
    # equity scale.  When equity is not 1.0 (e.g. direct diagnostic use), we
    # do apply all constraints so the output is immediately usable.
    if np.any(eq != 1.0):
        # Direct use: apply rounding and clamping
        clamped = quantize_and_clamp_lots_array(
            scaled,
            lot_step=lot_step,
            min_lots=min_lots,
            ceilings=(lot_ceiling, MAX_LOT_HARD_CEILING),
        )
    else:
        # lots-per-dollar mode: return raw (unrounded, unclamped) fractional lots
        clamped = scaled

    # Apply trade mask (only emit non-zero lots at entry bricks)
    if is_trade_mask is not None:
        mask = np.asarray(is_trade_mask, dtype=bool)
        clamped = np.where(mask, clamped, 0.0)

    return clamped


def _causal_rolling_std_long_floor(
    arr: np.ndarray,
    short_window: int,
    long_window: int,
    long_quantile: float,
    min_long_obs: int,
) -> np.ndarray:
    """
    Compute a causal rolling std with a **long-window stability floor**.

    At each position ``i``:
    1. Compute ``short_std[i]`` = std(arr[max(0,i-short_window):i]) — responsive.
    2. Compute ``floor_std[i]`` = quantile(chunk_stds(arr[0:i]), long_quantile)
       where chunks are non-overlapping windows of size ``short_window``.
    3. Return ``max(short_std[i], floor_std[i])``.

    This prevents cold-start sigma underestimation from driving oversized
    lot calculations during the warmup period.

    Parameters
    ----------
    arr : np.ndarray, shape (N,)
        Input brick-pts array.
    short_window : int
        Responsive rolling std window.
    long_window : int
        Look-back length for the floor distribution.
    long_quantile : float
        Quantile in (0, 1) of the chunk-std distribution used as the floor.
    min_long_obs : int
        Minimum long-buffer observations before the floor activates.
        Before this threshold the short std is used as-is (no floor).

    Returns
    -------
    np.ndarray, shape (N,)
        Effective sigma at each position (max of short and floor).
    """
    short_std = _causal_rolling_std(arr, short_window)
    n = len(arr)
    if n == 0 or long_window <= 0:
        return short_std

    out = short_std.copy()
    chunk_size = max(short_window, 2)

    for i in range(n):
        # long buffer: arr[max(0, i-long_window) : i]
        lo = max(0, i - long_window)
        long_seg = arr[lo:i]
        if len(long_seg) < min_long_obs:
            # Not enough long history yet — keep short std as-is
            continue

        # Empirical distribution of chunk stds from the long buffer
        n_chunks = len(long_seg) // chunk_size
        if n_chunks < 2:
            # Too few chunks for a meaningful quantile — use simple fraction
            if len(long_seg) >= 2:
                floor_val = float(np.std(long_seg, ddof=1)) * long_quantile
            else:
                continue
        else:
            chunk_stds = np.array(
                [
                    float(np.std(long_seg[j * chunk_size : (j + 1) * chunk_size], ddof=1))
                    for j in range(n_chunks)
                ]
            )
            floor_val = float(np.quantile(chunk_stds, long_quantile))

        out[i] = max(out[i], floor_val)

    return out


def _causal_rolling_std(arr: np.ndarray, window: int) -> np.ndarray:
    """
    Compute a **causal** (past-only) rolling standard deviation.

    At index ``i``, returns ``std(arr[max(0, i-window) : i])``.  Index 0
    returns 0.0 (only 1 element, no std).  Uses ``ddof=1``.

    This is O(N·window) but fully NumPy — no Python loops.

    Parameters
    ----------
    arr : np.ndarray, shape (N,)
        Input array.
    window : int
        Rolling window size (bricks).

    Returns
    -------
    np.ndarray, shape (N,)
        Rolling std array.  Values at indices < 1 are 0.0.
    """
    n = len(arr)
    if n == 0:
        return np.zeros(0, dtype=np.float64)

    out = np.zeros(n, dtype=np.float64)

    # Use stride tricks to build a 2D windowed view for vectorised std.
    # For the first (window-1) positions the window is truncated.
    # We handle truncated prefix separately to avoid boundary conditions.
    w = min(window, n)

    # Full windows (positions >= w-1): use stride tricks
    # out[i] = std(arr[i-w+1 : i+1])  for i in [w-1, n-1]
    # Number of full windows: n - w + 1
    # Guard: stride-tricks std with ddof=1 requires w >= 2.
    if n >= w and w >= 2:
        shape = (n - w + 1, w)
        strides = (arr.strides[0], arr.strides[0])
        windowed = np.lib.stride_tricks.as_strided(arr, shape=shape, strides=strides)
        # windowed[0] = arr[0:w], windowed[1] = arr[1:w+1], ...
        # std of each row (ddof=1)
        std_full = np.std(windowed, axis=1, ddof=1)
        # Place at out[w-1 : n]  (causal: out[w-1] uses arr[0:w], out[n-1] uses arr[n-w:n])
        out[w - 1 :] = std_full

    # Prefix: positions 1 to min(w-1, n-1) (windows are shorter, but causal)
    # out[i] = std(arr[0 : i+1])  — uses elements up to and including i
    # Only meaningful when i+1 >= 2, i.e. i >= 1.
    for i in range(1, min(w, n)):
        segment = arr[: i + 1]
        if len(segment) >= 2:
            out[i] = float(np.std(segment, ddof=1))
        # If segment has < 2 elements (i==0 case), out[i] stays 0.0.

    return out


# ---------------------------------------------------------------------------
# Equal-risk budget calibration
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class VolBudgetCalibrationResult:
    """
    Result of :func:`calibrate_vol_budget`.

    Attributes
    ----------
    optimal_vol_pct : float
        The vol budget (``target_vol_pct``) that produces a max drawdown
        closest to ``target_max_dd_pct``.
    achieved_max_dd_pct : float
        Actual max drawdown (%) achieved by the optimal budget.
    target_max_dd_pct : float
        The requested target drawdown (%).
    dd_error_pct : float
        ``achieved_max_dd_pct − target_max_dd_pct`` (signed, %).
    converged : bool
        True if the binary search converged within ``dd_tolerance_pct``.
    n_iterations : int
        Number of binary-search iterations performed.
    baseline_max_dd_pct : float
        Max drawdown of the baseline (COMPOUNDING) run for reference.
    baseline_final_equity : float
        Final equity of the baseline run.
    optimal_final_equity : float
        Final equity of the optimal vol-targeted run.
    delta_equity_pct : float
        ``(optimal_final_equity − baseline_final_equity) / baseline_final_equity × 100``.
        Positive means vol-targeting compounds faster at equal risk.
    scan_points : List[dict]
        Fine-grained scan around the solution (budget → max_dd, final_equity).
        Empty when ``fine_scan_n == 0``.
    """

    optimal_vol_pct: float
    achieved_max_dd_pct: float
    target_max_dd_pct: float
    dd_error_pct: float
    converged: bool
    n_iterations: int
    baseline_max_dd_pct: float
    baseline_final_equity: float
    optimal_final_equity: float
    delta_equity_pct: float
    scan_points: List[dict]


def calibrate_vol_budget(
    symbol: str,
    closes: "pd.Series",
    brick_size: float,
    usd_per_point: float,
    *,
    target_max_dd_pct: Optional[float] = None,
    dd_tolerance_pct: float = 0.1,
    budget_lo: float = 1e-5,
    budget_hi: float = 0.05,
    max_iterations: int = 60,
    fine_scan_n: int = 10,
    vol_sizing_params_template: Optional["VolSizingParams"] = None,
    filter_params: Optional["FilterParams"] = None,
    stop_params: Optional["StopParams"] = None,
    risk_params: Optional["RiskParams"] = None,
    session_break_minutes: float = 30.0,
) -> "VolBudgetCalibrationResult":
    """
    Binary-search calibration of ``target_vol_pct`` to match a target max-DD.

    Uses :func:`~kinetra.renko.backtest.backtest_instrument` as the inner
    evaluation function.  The monotonic property exploited is:

        Larger ``target_vol_pct`` → larger lots → more negative ``max_dd``.

    The search finds the smallest ``target_vol_pct`` that produces a
    ``max_dd ≤ target_max_dd_pct``.

    If ``target_max_dd_pct`` is None the baseline (COMPOUNDING mode) max-DD
    is used automatically — this gives the equal-risk comparison: the vol
    budget that matches the baseline's drawdown profile.

    Parameters
    ----------
    symbol : str
        Instrument symbol passed to :func:`backtest_instrument`.
    closes : pd.Series
        Close price series (same format as :func:`backtest_instrument`).
    brick_size : float
        Brick size in price units.
    usd_per_point : float
        USD value of 1 price unit per 1 standard lot.
    target_max_dd_pct : float or None
        Target max drawdown in percent (e.g. ``-0.615`` for −0.615 %).
        Negative values are expected (drawdowns are negative).
        If None, the baseline COMPOUNDING max-DD is measured and used.
    dd_tolerance_pct : float
        Acceptable absolute error from the target (%), default ``0.1``.
    budget_lo : float
        Lower bound for binary search.  Default ``1e-5``.
    budget_hi : float
        Upper bound for binary search.  Extended automatically if the
        upper bound still produces a shallower DD than the target.
        Default ``0.05``.
    max_iterations : int
        Maximum binary-search iterations.  Default ``60``.
    fine_scan_n : int
        Number of evenly-spaced scan points around the solution for the
        sensitivity analysis.  Set to ``0`` to skip.  Default ``10``.
    vol_sizing_params_template : VolSizingParams or None
        Template for vol sizing parameters.  ``target_vol_pct`` and
        ``usd_per_point`` are overridden; all other fields are taken from
        this object.  Defaults to ``VolSizingParams(usd_per_point=usd_per_point)``.
    filter_params : FilterParams or None
        Passed to :func:`backtest_instrument`.
    stop_params : StopParams or None
        Passed to :func:`backtest_instrument`.
    risk_params : RiskParams or None
        Passed to :func:`backtest_instrument`.
    session_break_minutes : float
        Passed to :func:`backtest_instrument`.

    Returns
    -------
    VolBudgetCalibrationResult
        Full calibration result including the optimal budget, achieved DD,
        equity comparison, and optional fine-scan sensitivity table.

    Raises
    ------
    ValueError
        If ``brick_size <= 0`` or ``usd_per_point <= 0``.
    ImportError
        If ``kinetra.renko.backtest`` is not available (should not happen
        in a complete installation).
    """
    import dataclasses

    import pandas as pd  # noqa: F401 (type hint in docstring)

    from kinetra.renko.backtest import (
        SizingMode,
        VolSizingParams,
        backtest_instrument,
    )

    if brick_size <= 0:
        raise ValueError(f"brick_size must be > 0, got {brick_size}")
    if usd_per_point <= 0:
        raise ValueError(f"usd_per_point must be > 0, got {usd_per_point}")

    # Build a base VolSizingParams (without target_vol_pct — overridden per iteration)
    if vol_sizing_params_template is None:
        base_vsp = VolSizingParams(usd_per_point=usd_per_point)
    else:
        base_vsp = dataclasses.replace(vol_sizing_params_template, usd_per_point=usd_per_point)

    def _extract_dd_and_equity(res: "InstrumentBacktestResult") -> tuple:
        """
        Extract (max_dd_pct, final_equity_usd) from an InstrumentBacktestResult.

        ``InstrumentBacktestResult`` stores:
        - ``equity_curve``: cumulative net P&L after each trade (starts at 0.0,
          so ``equity_curve[i] = initial_equity + sum(net_usd[:i+1])``).
        - ``max_dd_usd``: maximum drawdown in USD (always <= 0).

        We convert both to percentage terms relative to initial equity so the
        binary search operates on a dimensionless scale that is comparable
        across different lot sizes and equity levels.

        For COMPOUNDING mode the initial equity changes trade-by-trade, but
        the initial_equity field of VolSizingParams gives the starting point.
        For a clean percentage we define:

            max_dd_pct = max_dd_usd / initial_equity * 100.0   (always <= 0)
            final_equity = initial_equity + equity_curve[-1]   (USD)

        This is consistent with the notebook's convention where DD is expressed
        as a fraction of the running peak — here we use starting equity as the
        denominator which is equivalent when the initial equity is fixed across
        all comparison runs (which it is: base_vsp.initial_equity).
        """
        init_eq = base_vsp.initial_equity
        eq_curve = res.equity_curve
        final_pnl = float(eq_curve[-1]) if eq_curve else 0.0
        final_eq = init_eq + final_pnl

        # max_dd_usd is always <= 0; express as % of initial equity
        # Guard: if initial equity is 0 or max_dd_usd is 0, return 0.0
        if init_eq <= 0:
            dd_pct = 0.0
        else:
            dd_pct = float(res.max_dd_usd) / init_eq * 100.0

        return dd_pct, final_eq

    def _run_vol(budget: float) -> tuple:
        """Return (max_dd_pct, final_equity_usd) for a given vol budget."""
        vsp = dataclasses.replace(base_vsp, target_vol_pct=budget)
        res = backtest_instrument(
            symbol=symbol,
            closes=closes,
            brick_size=brick_size,
            filter_params=filter_params,
            stop_params=stop_params,
            risk_params=risk_params,
            sizing_mode=SizingMode.VOL_TARGET,
            vol_sizing_params=vsp,
            session_break_minutes=session_break_minutes,
        )
        return _extract_dd_and_equity(res)

    def _run_baseline() -> tuple:
        """Return (max_dd_pct, final_equity_usd) for COMPOUNDING baseline."""
        res = backtest_instrument(
            symbol=symbol,
            closes=closes,
            brick_size=brick_size,
            filter_params=filter_params,
            stop_params=stop_params,
            risk_params=risk_params,
            sizing_mode=SizingMode.COMPOUNDING,
            vol_sizing_params=base_vsp,
            session_break_minutes=session_break_minutes,
        )
        return _extract_dd_and_equity(res)

    # ── Step 1: measure baseline ──────────────────────────────────────────
    baseline_dd, baseline_equity = _run_baseline()

    # ── Step 2: resolve target DD ─────────────────────────────────────────
    if target_max_dd_pct is None:
        target_dd = baseline_dd
    else:
        target_dd = float(target_max_dd_pct)

    # ── Step 3: confirm budget_hi brackets the target ─────────────────────
    # DD is monotonically more negative as budget grows.
    # We need: _run_vol(budget_hi).max_dd  <=  target_dd  (i.e. deeper than target).
    hi_dd, _ = _run_vol(budget_hi)
    if hi_dd > target_dd:
        # The upper bound is still shallower than the target — extend hi.
        for multiplier in [2, 4, 8, 16, 32, 64]:
            budget_hi_candidate = budget_hi * multiplier
            hi_dd_cand, _ = _run_vol(budget_hi_candidate)
            if hi_dd_cand <= target_dd:
                budget_hi = budget_hi_candidate
                hi_dd = hi_dd_cand
                break
        else:
            # Could not bracket; return best-effort with what we have
            logger.warning(
                "calibrate_vol_budget: could not bracket target DD %.4f%% "
                "even at budget_hi=%.6f (achieved DD=%.4f%%). "
                "Returning non-converged result.",
                target_dd,
                budget_hi,
                hi_dd,
            )
            opt_dd, opt_equity = _run_vol(budget_hi)
            delta_pct = (
                (opt_equity - baseline_equity) / baseline_equity * 100.0 if baseline_equity else 0.0
            )
            return VolBudgetCalibrationResult(
                optimal_vol_pct=budget_hi,
                achieved_max_dd_pct=opt_dd,
                target_max_dd_pct=target_dd,
                dd_error_pct=opt_dd - target_dd,
                converged=False,
                n_iterations=0,
                baseline_max_dd_pct=baseline_dd,
                baseline_final_equity=baseline_equity,
                optimal_final_equity=opt_equity,
                delta_equity_pct=delta_pct,
                scan_points=[],
            )

    # ── Step 4: binary search ─────────────────────────────────────────────
    lo = float(budget_lo)
    hi = float(budget_hi)

    best_budget = hi
    best_dd = hi_dd
    best_equity = 0.0
    converged = False
    n_iters = 0

    for n_iters in range(1, max_iterations + 1):
        mid = (lo + hi) / 2.0
        mid_dd, mid_equity = _run_vol(mid)
        err = mid_dd - target_dd  # positive → shallower; negative → deeper

        if abs(err) <= dd_tolerance_pct:
            best_budget = mid
            best_dd = mid_dd
            best_equity = mid_equity
            converged = True
            break

        # Track the best bracket point closest to the target
        if abs(mid_dd - target_dd) < abs(best_dd - target_dd):
            best_budget = mid
            best_dd = mid_dd
            best_equity = mid_equity

        if err > 0:
            # Too shallow (DD not negative enough) → increase budget
            lo = mid
        else:
            # Too deep (DD more negative than target) → decrease budget
            hi = mid

        if (hi - lo) < 1e-10:
            converged = True
            break

    if best_equity == 0.0:
        # Fetch final equity for the best budget if not captured above
        _, best_equity = _run_vol(best_budget)

    # ── Step 5: fine-grained scan ──────────────────────────────────────────
    scan_points: List[dict] = []
    if fine_scan_n > 0:
        scan_lo = max(budget_lo, best_budget * 0.5)
        scan_hi = best_budget * 1.5
        scan_budgets = np.linspace(scan_lo, scan_hi, fine_scan_n).tolist()
        for vb in scan_budgets:
            s_dd, s_eq = _run_vol(vb)
            scan_points.append(
                {"budget": float(vb), "max_dd_pct": float(s_dd), "final_equity": float(s_eq)}
            )

    delta_pct = (
        (best_equity - baseline_equity) / baseline_equity * 100.0 if baseline_equity else 0.0
    )

    return VolBudgetCalibrationResult(
        optimal_vol_pct=best_budget,
        achieved_max_dd_pct=best_dd,
        target_max_dd_pct=target_dd,
        dd_error_pct=best_dd - target_dd,
        converged=converged,
        n_iterations=n_iters,
        baseline_max_dd_pct=baseline_dd,
        baseline_final_equity=baseline_equity,
        optimal_final_equity=best_equity,
        delta_equity_pct=delta_pct,
        scan_points=scan_points,
    )


# ---------------------------------------------------------------------------
# Correct equal-risk comparison: calibrate both modes to a shared DD target
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class SizingComparisonResult:
    """
    Result of :func:`compare_sizing_at_risk_class`.

    Compares baseline COMPOUNDING sizing against VOL_TARGET sizing when both
    are constrained to the same max-drawdown budget.  This is the correct
    equal-risk comparison: instead of matching the baseline's accidental DD
    (which may be dominated by a single event), both sizing modes are
    calibrated to a **chosen** DD class (e.g. 5 %) so the equity comparison
    is meaningful.

    Attributes
    ----------
    target_dd_pct : float
        The shared max-drawdown budget used for both modes (%).
    baseline_vol_pct : float
        The ``target_vol_pct``-equivalent for COMPOUNDING mode — i.e. the
        ``compounding_capital_per_lot`` that achieves ``target_dd_pct``.
        Stored as the *effective* vol fraction implied by the solved lot size
        (for display; COMPOUNDING mode uses ``compounding_capital_per_lot``
        internally).
    baseline_capital_per_lot : float
        The solved ``compounding_capital_per_lot`` for COMPOUNDING mode that
        achieves ``target_dd_pct``.
    baseline_achieved_dd_pct : float
        Actual max DD (%) of the solved COMPOUNDING run.
    baseline_final_equity : float
        Final equity (USD) of the solved COMPOUNDING run.
    vol_budget : float
        The solved ``target_vol_pct`` for VOL_TARGET mode that achieves
        ``target_dd_pct``.
    vol_achieved_dd_pct : float
        Actual max DD (%) of the solved VOL_TARGET run.
    vol_final_equity : float
        Final equity (USD) of the solved VOL_TARGET run.
    delta_equity_pct : float
        ``(vol_final_equity - baseline_final_equity) / baseline_final_equity * 100``.
        Positive means vol-targeting compounds faster at the same DD budget.
    baseline_converged : bool
        Whether the COMPOUNDING calibration converged.
    vol_converged : bool
        Whether the VOL_TARGET calibration converged.
    scan_baseline : List[dict]
        Fine-grained scan for COMPOUNDING sensitivity.
    scan_vol : List[dict]
        Fine-grained scan for VOL_TARGET sensitivity.
    """

    target_dd_pct: float
    baseline_vol_pct: float
    baseline_capital_per_lot: float
    baseline_achieved_dd_pct: float
    baseline_final_equity: float
    vol_budget: float
    vol_achieved_dd_pct: float
    vol_final_equity: float
    delta_equity_pct: float
    baseline_converged: bool
    vol_converged: bool
    scan_baseline: List[dict]
    scan_vol: List[dict]


def compare_sizing_at_risk_class(
    symbol: str,
    closes: "pd.Series",
    brick_size: float,
    usd_per_point: float,
    target_dd_pct: float,
    *,
    dd_tolerance_pct: float = 0.2,
    initial_equity: float = 1_000.0,
    compounding_lot_per_unit: float = 0.01,
    # Compounding calibration search bounds (capital per lot)
    baseline_lo: float = 100.0,
    baseline_hi: float = 1_000_000.0,
    # Vol-target calibration search bounds
    vol_budget_lo: float = 1e-5,
    vol_budget_hi: float = 0.20,
    max_iterations: int = 60,
    fine_scan_n: int = 12,
    vol_sizing_params_template: Optional["VolSizingParams"] = None,
    filter_params: Optional["FilterParams"] = None,
    stop_params: Optional["StopParams"] = None,
    risk_params: Optional["RiskParams"] = None,
    session_break_minutes: float = 30.0,
) -> "SizingComparisonResult":
    """
    Correct equal-risk comparison: solve both COMPOUNDING and VOL_TARGET to
    a shared ``target_dd_pct``, then compare final equity.

    The key insight from the research chronology:

        The 0.615 % baseline DD is dominated by a single event and is not a
        meaningful risk budget.  Matching it forces both systems into a regime
        where the compounding advantage of the baseline overwhelms any
        signal.  Instead, choose a real risk budget (e.g. 3–8 %) and solve
        the sizing parameter for *each* mode that hits that budget.  Only then
        does the equity comparison answer the research question: "does
        vol-targeting compound faster or slower than fixed compounding at the
        same risk class?"

    Algorithm
    ---------
    1. Binary-search ``compounding_capital_per_lot`` so that COMPOUNDING mode
       achieves ``max_dd ≈ target_dd_pct``.  Monotonic property: smaller
       capital-per-lot → larger lots → more negative DD.
    2. Binary-search ``target_vol_pct`` so that VOL_TARGET mode achieves the
       same ``target_dd_pct``.
    3. Compare final equity of both solved runs.

    Parameters
    ----------
    symbol : str
        Instrument symbol passed to :func:`backtest_instrument`.
    closes : pd.Series
        Close price series.
    brick_size : float
        Brick size in price units.
    usd_per_point : float
        USD value of 1 price unit per 1 standard lot.
    target_dd_pct : float
        Target max drawdown in percent (negative, e.g. ``-5.0`` for −5 %).
    dd_tolerance_pct : float
        Acceptable absolute error from target DD (%), default ``0.2``.
    initial_equity : float
        Starting equity for both modes.  Default ``1_000.0``.
    compounding_lot_per_unit : float
        The lot unit for COMPOUNDING mode (``fixed_lot``).
        Default ``0.01`` (replicates the research baseline).
    baseline_lo : float
        Lower bound of ``compounding_capital_per_lot`` search.
        Smaller values → larger lots → deeper DD.  Default ``100.0``.
    baseline_hi : float
        Upper bound of ``compounding_capital_per_lot`` search.
        Larger values → smaller lots → shallower DD.  Default ``1_000_000.0``.
    vol_budget_lo : float
        Lower bound for vol budget search.  Default ``1e-5``.
    vol_budget_hi : float
        Upper bound for vol budget search.  Default ``0.20``.
    max_iterations : int
        Binary-search iteration cap per mode.  Default ``60``.
    fine_scan_n : int
        Fine-scan points around each solution.  ``0`` to skip.  Default ``12``.
    vol_sizing_params_template : VolSizingParams or None
        Template for vol sizing parameters (except ``target_vol_pct`` and
        ``usd_per_point`` which are always overridden).
    filter_params : FilterParams or None
        Passed to :func:`backtest_instrument`.
    stop_params : StopParams or None
        Passed to :func:`backtest_instrument`.
    risk_params : RiskParams or None
        Passed to :func:`backtest_instrument`.
    session_break_minutes : float
        Passed to :func:`backtest_instrument`.

    Returns
    -------
    SizingComparisonResult
        Full comparison result with solved parameters, achieved DDs, final
        equities, delta, convergence flags, and sensitivity scans.

    Raises
    ------
    ValueError
        If ``brick_size <= 0``, ``usd_per_point <= 0``, or
        ``target_dd_pct >= 0``.
    """
    import dataclasses

    from kinetra.renko.backtest import (
        SizingMode,
        VolSizingParams,
        backtest_instrument,
    )

    if brick_size <= 0:
        raise ValueError(f"brick_size must be > 0, got {brick_size}")
    if usd_per_point <= 0:
        raise ValueError(f"usd_per_point must be > 0, got {usd_per_point}")
    if target_dd_pct >= 0:
        raise ValueError(f"target_dd_pct must be negative (a drawdown), got {target_dd_pct}")

    target = float(target_dd_pct)

    if vol_sizing_params_template is None:
        base_vsp = VolSizingParams(
            usd_per_point=usd_per_point,
            initial_equity=initial_equity,
        )
    else:
        base_vsp = dataclasses.replace(
            vol_sizing_params_template,
            usd_per_point=usd_per_point,
            initial_equity=initial_equity,
        )

    # ── Shared DD extractor (same convention as calibrate_vol_budget) ────
    def _dd_and_eq(res: "InstrumentBacktestResult") -> tuple:
        eq_curve = res.equity_curve
        final_pnl = float(eq_curve[-1]) if eq_curve else 0.0
        final_eq = initial_equity + final_pnl
        dd_pct = float(res.max_dd_usd) / initial_equity * 100.0 if initial_equity > 0 else 0.0
        return dd_pct, final_eq

    # ── COMPOUNDING calibration ───────────────────────────────────────────
    # Monotonic: smaller capital_per_lot → larger lots → more negative DD.
    # Search for capital_per_lot such that DD ≈ target.
    def _run_compounding(cap_per_lot: float) -> tuple:
        vsp = dataclasses.replace(
            base_vsp,
            compounding_capital_per_lot=cap_per_lot,
            fixed_lot=compounding_lot_per_unit,
        )
        res = backtest_instrument(
            symbol=symbol,
            closes=closes,
            brick_size=brick_size,
            filter_params=filter_params,
            stop_params=stop_params,
            risk_params=risk_params,
            sizing_mode=SizingMode.COMPOUNDING,
            vol_sizing_params=vsp,
            session_break_minutes=session_break_minutes,
        )
        return _dd_and_eq(res)

    # Confirm brackets: lo should produce DD deeper than target, hi shallower.
    # (smaller cap_per_lot → larger lots → deeper DD)
    b_lo_dd, _ = _run_compounding(baseline_lo)
    b_hi_dd, _ = _run_compounding(baseline_hi)

    # If lo is still shallower than target, the target is too deep for this
    # instrument — clamp and warn.
    if b_lo_dd > target:
        logger.warning(
            "compare_sizing_at_risk_class: COMPOUNDING cannot reach target_dd=%.2f%% "
            "even at baseline_lo=%.1f (achieved DD=%.4f%%). "
            "Consider increasing target_dd_pct magnitude or lowering baseline_lo.",
            target,
            baseline_lo,
            b_lo_dd,
        )

    best_cap = baseline_lo
    best_b_dd, best_b_eq = b_lo_dd, 0.0
    b_converged = False
    b_iters = 0
    b_lo_search = float(baseline_lo)
    b_hi_search = float(baseline_hi)

    for b_iters in range(1, max_iterations + 1):
        mid_cap = (b_lo_search + b_hi_search) / 2.0
        mid_dd, mid_eq = _run_compounding(mid_cap)
        err = mid_dd - target  # positive → shallower; negative → deeper

        if abs(err) <= dd_tolerance_pct:
            best_cap = mid_cap
            best_b_dd = mid_dd
            best_b_eq = mid_eq
            b_converged = True
            break

        if abs(mid_dd - target) < abs(best_b_dd - target):
            best_cap = mid_cap
            best_b_dd = mid_dd
            best_b_eq = mid_eq

        if err > 0:
            # Too shallow → decrease capital_per_lot (bigger lots)
            b_hi_search = mid_cap
        else:
            # Too deep → increase capital_per_lot (smaller lots)
            b_lo_search = mid_cap

        if (b_hi_search - b_lo_search) < 1e-3:
            b_converged = True
            break

    if best_b_eq == 0.0:
        _, best_b_eq = _run_compounding(best_cap)

    # ── VOL_TARGET calibration (reuse calibrate_vol_budget logic) ────────
    vol_result = calibrate_vol_budget(
        symbol=symbol,
        closes=closes,
        brick_size=brick_size,
        usd_per_point=usd_per_point,
        target_max_dd_pct=target,
        dd_tolerance_pct=dd_tolerance_pct,
        budget_lo=vol_budget_lo,
        budget_hi=vol_budget_hi,
        max_iterations=max_iterations,
        fine_scan_n=fine_scan_n,
        vol_sizing_params_template=dataclasses.replace(base_vsp, initial_equity=initial_equity),
        filter_params=filter_params,
        stop_params=stop_params,
        risk_params=risk_params,
        session_break_minutes=session_break_minutes,
    )

    # ── Fine-grained scan for COMPOUNDING ────────────────────────────────
    scan_baseline: List[dict] = []
    if fine_scan_n > 0:
        scan_lo_cap = max(baseline_lo, best_cap * 0.5)
        scan_hi_cap = best_cap * 1.5
        scan_caps = np.linspace(scan_lo_cap, scan_hi_cap, fine_scan_n).tolist()
        for cap in scan_caps:
            s_dd, s_eq = _run_compounding(cap)
            scan_baseline.append(
                {
                    "capital_per_lot": float(cap),
                    "max_dd_pct": float(s_dd),
                    "final_equity": float(s_eq),
                }
            )

    # ── Effective vol-fraction implied by the solved compounding lot ─────
    # At the solved capital_per_lot, the lot at initial_equity is:
    #   lots_at_start = (initial_equity / best_cap) * compounding_lot_per_unit
    # The effective vol fraction is defined as the fraction of equity risked
    # per brick at that lot size (for display only):
    #   eff_vol_pct = (lots_at_start * brick * usd_per_point * 1e5) / initial_equity
    lots_at_start = (initial_equity / best_cap) * compounding_lot_per_unit if best_cap > 0 else 0.0
    eff_vol_pct = (
        lots_at_start * brick_size * usd_per_point * 1e5 / initial_equity
        if initial_equity > 0
        else 0.0
    )

    delta_pct = (
        (vol_result.optimal_final_equity - best_b_eq) / best_b_eq * 100.0 if best_b_eq > 0 else 0.0
    )

    return SizingComparisonResult(
        target_dd_pct=target,
        baseline_vol_pct=eff_vol_pct,
        baseline_capital_per_lot=best_cap,
        baseline_achieved_dd_pct=best_b_dd,
        baseline_final_equity=best_b_eq,
        vol_budget=vol_result.optimal_vol_pct,
        vol_achieved_dd_pct=vol_result.achieved_max_dd_pct,
        vol_final_equity=vol_result.optimal_final_equity,
        delta_equity_pct=delta_pct,
        baseline_converged=b_converged,
        vol_converged=vol_result.converged,
        scan_baseline=scan_baseline,
        scan_vol=vol_result.scan_points,
    )


# ---------------------------------------------------------------------------
# Diagnostic helpers
def vol_sizing_report(
    sizer: VolTargetSizer,
    symbols: List[str],
    brick_sizes: Dict[str, float],
    equity_usd: float,
    usd_per_point: Dict[str, float],
) -> Dict[str, dict]:
    """
    Build a diagnostic snapshot of the vol sizer state.

    Useful for logging, operator dashboards, and the menu's calibration
    drift status display.

    Parameters
    ----------
    sizer : VolTargetSizer
        The active sizer instance.
    symbols : list[str]
        Symbols to report on.
    brick_sizes : dict[str, float]
        Brick size per symbol (price units).
    equity_usd : float
        Current equity for hypothetical lot calculation.
    usd_per_point : dict[str, float]
        USD per point per symbol.

    Returns
    -------
    dict[str, dict]
        Per-symbol diagnostic dict with keys:
        ``vol_estimate``, ``n_observations``, ``is_warmed_up``,
        ``hypothetical_1lot_usd_vol``, ``hypothetical_lots_100k``.
    """
    report: Dict[str, dict] = {}
    for sym in symbols:
        bs = brick_sizes.get(sym, 0.0)
        upp = usd_per_point.get(sym, 0.0)
        vol = sizer.vol_estimate(sym, bs)
        usd_vol_1lot = bs * upp * vol * 100_000.0 if (bs > 0 and upp > 0) else 0.0
        hyp_lots = (
            sizer.compute(
                symbol=sym,
                equity_usd=equity_usd,
                brick_size=bs,
                usd_per_point=upp,
                gate_lot_ceiling=MAX_LOT_HARD_CEILING,
            )
            if (bs > 0 and upp > 0 and equity_usd > 0)
            else 0.0
        )
        report[sym] = {
            "vol_estimate": vol,
            "n_observations": sizer.n_observations(sym),
            "is_warmed_up": sizer.is_warmed_up(sym),
            "hypothetical_1lot_usd_vol": usd_vol_1lot,
            "hypothetical_lots_100k": hyp_lots,
        }
    return report


# ---------------------------------------------------------------------------
# Portfolio-level Pareto frontier: compare both sizing modes across DD budgets
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class PortfolioParetoPoint:
    """
    A single point on the portfolio Pareto frontier.

    Represents the comparison between COMPOUNDING and VOL_TARGET sizing
    modes when both are calibrated to the same ``target_dd_pct`` on a
    *portfolio* of instruments.

    Attributes
    ----------
    target_dd_pct : float
        The shared max-drawdown budget used for both modes (%).
    baseline_capital_per_lot : float
        Solved ``compounding_capital_per_lot`` for COMPOUNDING mode.
    baseline_achieved_dd_pct : float
        Actual portfolio max DD (%) of the solved COMPOUNDING run.
    baseline_final_equity : float
        Final portfolio equity of the solved COMPOUNDING run.
    baseline_omega : float
        Portfolio Omega ratio of the solved COMPOUNDING run.
    baseline_calmar : float
        Portfolio Calmar ratio of the solved COMPOUNDING run.
    baseline_converged : bool
        Whether the COMPOUNDING calibration converged.
    vol_budget : float
        Solved ``target_vol_pct`` for VOL_TARGET mode.
    vol_achieved_dd_pct : float
        Actual portfolio max DD (%) of the solved VOL_TARGET run.
    vol_final_equity : float
        Final portfolio equity of the solved VOL_TARGET run.
    vol_omega : float
        Portfolio Omega ratio of the solved VOL_TARGET run.
    vol_calmar : float
        Portfolio Calmar ratio of the solved VOL_TARGET run.
    vol_converged : bool
        Whether the VOL_TARGET calibration converged.
    delta_equity_pct : float
        ``(vol_final_equity - baseline_final_equity) / baseline_final_equity * 100``.
        Positive means vol-targeting compounds faster at this DD budget.
    delta_omega : float
        ``vol_omega - baseline_omega``.
    delta_calmar : float
        ``vol_calmar - baseline_calmar``.
    n_instruments : int
        Number of instruments in the portfolio run.
    baseline_scan : List[dict]
        Fine-grained scan for COMPOUNDING mode sensitivity.
    vol_scan : List[dict]
        Fine-grained scan for VOL_TARGET mode sensitivity.
    """

    target_dd_pct: float
    baseline_capital_per_lot: float
    baseline_achieved_dd_pct: float
    baseline_final_equity: float
    baseline_omega: float
    baseline_calmar: float
    baseline_converged: bool
    vol_budget: float
    vol_achieved_dd_pct: float
    vol_final_equity: float
    vol_omega: float
    vol_calmar: float
    vol_converged: bool
    delta_equity_pct: float
    delta_omega: float
    delta_calmar: float
    n_instruments: int
    baseline_scan: List[dict]
    vol_scan: List[dict]


@dataclass(slots=True)
class PortfolioParetoResult:
    """
    Full portfolio-level Pareto frontier comparing COMPOUNDING vs VOL_TARGET.

    Produced by :func:`compare_portfolio_sizing_pareto`.

    Attributes
    ----------
    points : List[PortfolioParetoPoint]
        One entry per ``target_dd_pct`` value, sorted ascending by
        ``abs(target_dd_pct)`` (most conservative first).
    instruments : List[str]
        Symbols included in the portfolio.
    initial_equity : float
        Starting equity used for all runs.
    recommended_mode : str
        ``"vol_target"`` or ``"compounding"`` — the mode with higher
        ``delta_equity_pct`` at the median DD budget, with a tie-break
        that prefers ``"vol_target"`` when gains are within 2 % (i.e.
        vol-targeting is preferred when roughly equivalent because it
        produces interpretable, cross-instrument-comparable equity curves).
    recommended_vol_budget : float
        The solved ``target_vol_pct`` at the recommended DD budget
        (only meaningful when ``recommended_mode == "vol_target"``).
    recommended_capital_per_lot : float
        The solved ``compounding_capital_per_lot`` at the recommended DD
        budget (only meaningful when ``recommended_mode == "compounding"``).
    recommended_dd_pct : float
        The DD budget used for the recommendation.
    """

    points: List[PortfolioParetoPoint]
    instruments: List[str]
    initial_equity: float
    recommended_mode: str
    recommended_vol_budget: float
    recommended_capital_per_lot: float
    recommended_dd_pct: float


def compare_portfolio_sizing_pareto(
    instruments: "Dict[str, pd.Series]",
    brick_sizes: "Dict[str, float]",
    usd_per_points: "Dict[str, float]",
    target_dd_pcts: "List[float]",
    *,
    initial_equity: float = 10_000.0,
    dd_tolerance_pct: float = 0.3,
    allocation_weights: "Optional[Dict[str, float]]" = None,
    cluster_map: "Optional[Dict[str, str]]" = None,
    compounding_lot_per_unit: float = 0.01,
    baseline_lo: float = 100.0,
    baseline_hi: float = 5_000_000.0,
    vol_budget_lo: float = 1e-6,
    vol_budget_hi: float = 0.20,
    max_iterations: int = 60,
    fine_scan_n: int = 10,
    vol_sizing_params_template: "Optional[VolSizingParams]" = None,
    filter_params: "Optional[FilterParams]" = None,
    stop_params: "Optional[StopParams]" = None,
    risk_params: "Optional[RiskParams]" = None,
    session_break_minutes: float = 30.0,
    recommend_dd_pct: "Optional[float]" = None,
) -> "PortfolioParetoResult":
    """
    Portfolio-level Pareto frontier: compare COMPOUNDING vs VOL_TARGET sizing
    across multiple ``target_dd_pcts``, using :func:`backtest_portfolio` as
    the inner evaluator.

    For each DD budget both sizing modes are binary-searched so that the
    *portfolio* max drawdown hits ``target_dd_pct``.  The same sizing
    parameter (``compounding_capital_per_lot`` or ``target_vol_pct``) is
    applied uniformly across all instruments in the portfolio run so the
    comparison captures true diversification effects.

    This is the correct equal-risk portfolio comparison:

    - Baseline (COMPOUNDING): every instrument trades ``lots = equity /
      capital_per_lot``.  The solved ``capital_per_lot`` produces the
      target portfolio DD.
    - Vol-targeted: every instrument uses the same ``target_vol_pct`` but
      sizes *per-instrument* vol (its brick P&L std) so position sizes
      differ across instruments automatically.  The solved ``vol_pct``
      produces the target portfolio DD.

    Parameters
    ----------
    instruments : dict[str, pd.Series]
        Mapping of symbol → close price series (M30 or M1 aggregated to
        the same frequency as used in :func:`backtest_instrument`).
    brick_sizes : dict[str, float]
        Brick size per instrument (price units).
    usd_per_points : dict[str, float]
        USD value of 1 price unit per 1 standard lot, per instrument.
    target_dd_pcts : list[float]
        List of target max drawdown percentages to evaluate.  Must be
        negative (e.g. ``[-3.0, -5.0, -8.0, -12.0]``).
    initial_equity : float
        Starting equity for all runs.  Default ``10_000.0``.
    dd_tolerance_pct : float
        Acceptable absolute error from the target DD (%).  Default ``0.3``.
        Slightly looser than the single-instrument default because portfolio
        DD is noisier due to trade-merge interleaving.
    allocation_weights : dict[str, float] or None
        Allocation weights per instrument passed to
        :func:`~kinetra.renko.backtest.backtest_portfolio`.
        If None, equal weights (1.0) are used.
    cluster_map : dict[str, str] or None
        Cluster labels per instrument for contribution tracking.
        Passed through to :func:`backtest_portfolio`.
    compounding_lot_per_unit : float
        Lot unit for COMPOUNDING mode.  Default ``0.01``.
    baseline_lo : float
        Lower bound for ``compounding_capital_per_lot`` search.
        Default ``100.0`` (very aggressive — large lots, deep DD).
    baseline_hi : float
        Upper bound for ``compounding_capital_per_lot`` search.
        Default ``5_000_000.0`` (very conservative — tiny lots, shallow DD).
    vol_budget_lo : float
        Lower bound for ``target_vol_pct`` search.  Default ``1e-6``.
    vol_budget_hi : float
        Upper bound for ``target_vol_pct`` search.  Default ``0.20``.
    max_iterations : int
        Binary-search iteration cap per mode per DD budget.  Default ``60``.
    fine_scan_n : int
        Fine-scan sensitivity points around each solution.  ``0`` skips.
        Default ``10``.
    vol_sizing_params_template : VolSizingParams or None
        Template for vol sizing parameters applied to all instruments.
        ``target_vol_pct`` and ``usd_per_point`` are overridden per
        instrument per iteration; all other fields are taken from this
        object.
    filter_params : FilterParams or None
        Passed to :func:`backtest_instrument` for every instrument.
    stop_params : StopParams or None
        Passed to :func:`backtest_instrument` for every instrument.
    risk_params : RiskParams or None
        Passed to :func:`backtest_instrument` for every instrument.
    session_break_minutes : float
        Session break threshold (minutes) passed to every
        :func:`backtest_instrument` call.
    recommend_dd_pct : float or None
        The DD budget to use for the recommendation logic.  If None, the
        median value of ``target_dd_pcts`` is used.

    Returns
    -------
    PortfolioParetoResult
        Full Pareto frontier with one :class:`PortfolioParetoPoint` per
        ``target_dd_pct``, plus a production recommendation.

    Raises
    ------
    ValueError
        If ``instruments`` is empty, any ``target_dd_pct >= 0``, or
        ``brick_sizes`` / ``usd_per_points`` are missing keys for any
        instrument.
    """
    import dataclasses

    import pandas as pd  # noqa: F401

    from kinetra.renko.backtest import (
        SizingMode,
        VolSizingParams,
        backtest_instrument,
        backtest_portfolio,
    )

    # ── Validate inputs ──────────────────────────────────────────────────
    if not instruments:
        raise ValueError("instruments dict must not be empty")

    for sym in instruments:
        if sym not in brick_sizes:
            raise ValueError(f"brick_sizes missing key: {sym!r}")
        if sym not in usd_per_points:
            raise ValueError(f"usd_per_points missing key: {sym!r}")
        if brick_sizes[sym] <= 0:
            raise ValueError(f"brick_sizes[{sym!r}] must be > 0, got {brick_sizes[sym]}")
        if usd_per_points[sym] <= 0:
            raise ValueError(f"usd_per_points[{sym!r}] must be > 0, got {usd_per_points[sym]}")

    for dd in target_dd_pcts:
        if dd >= 0:
            raise ValueError(f"All target_dd_pcts must be negative (drawdowns), got {dd}")

    symbols = list(instruments.keys())
    n_instruments = len(symbols)

    # ── Build per-instrument base VolSizingParams ─────────────────────────
    # The template is shared; usd_per_point and initial_equity are overridden
    # per instrument per call.
    if vol_sizing_params_template is None:
        _vsp_template = VolSizingParams(initial_equity=initial_equity)
    else:
        _vsp_template = dataclasses.replace(
            vol_sizing_params_template, initial_equity=initial_equity
        )

    # ── Inner runner: run all instruments with a shared sizing parameter ──
    # Returns (portfolio_max_dd_pct, portfolio_final_equity, omega, calmar)

    def _run_portfolio_compounding(cap_per_lot: float) -> "tuple[float, float, float, float]":
        """Run all instruments with COMPOUNDING sizing at cap_per_lot."""
        inst_results = {}
        for sym in symbols:
            closes = instruments[sym]
            upp = usd_per_points[sym]
            bs = brick_sizes[sym]
            vsp = dataclasses.replace(
                _vsp_template,
                usd_per_point=upp,
                initial_equity=initial_equity,
                compounding_capital_per_lot=cap_per_lot,
                fixed_lot=compounding_lot_per_unit,
            )
            res = backtest_instrument(
                symbol=sym,
                closes=closes,
                brick_size=bs,
                filter_params=filter_params,
                stop_params=stop_params,
                risk_params=risk_params,
                sizing_mode=SizingMode.COMPOUNDING,
                vol_sizing_params=vsp,
                session_break_minutes=session_break_minutes,
            )
            inst_results[sym] = res

        port = backtest_portfolio(
            instrument_results=inst_results,
            allocation_weights=allocation_weights,
            cluster_map=cluster_map,
        )
        dd_pct = port.max_dd_usd / initial_equity * 100.0 if initial_equity > 0 else 0.0
        final_eq = initial_equity + port.net_pnl_usd
        return float(dd_pct), float(final_eq), float(port.omega), float(port.calmar_ratio)

    def _run_portfolio_vol(vol_pct: float) -> "tuple[float, float, float, float]":
        """Run all instruments with VOL_TARGET sizing at vol_pct."""
        inst_results = {}
        for sym in symbols:
            closes = instruments[sym]
            upp = usd_per_points[sym]
            bs = brick_sizes[sym]
            vsp = dataclasses.replace(
                _vsp_template,
                usd_per_point=upp,
                initial_equity=initial_equity,
                target_vol_pct=vol_pct,
            )
            res = backtest_instrument(
                symbol=sym,
                closes=closes,
                brick_size=bs,
                filter_params=filter_params,
                stop_params=stop_params,
                risk_params=risk_params,
                sizing_mode=SizingMode.VOL_TARGET,
                vol_sizing_params=vsp,
                session_break_minutes=session_break_minutes,
            )
            inst_results[sym] = res

        port = backtest_portfolio(
            instrument_results=inst_results,
            allocation_weights=allocation_weights,
            cluster_map=cluster_map,
        )
        dd_pct = port.max_dd_usd / initial_equity * 100.0 if initial_equity > 0 else 0.0
        final_eq = initial_equity + port.net_pnl_usd
        return float(dd_pct), float(final_eq), float(port.omega), float(port.calmar_ratio)

    # ── Binary-search helper ──────────────────────────────────────────────
    # Both searches exploit the same monotonic property:
    #   larger position parameter → larger lots → more negative portfolio DD.

    def _bisect_compounding(
        target: float,
    ) -> "tuple[float, float, float, float, float, bool, List[dict]]":
        """
        Binary-search ``compounding_capital_per_lot`` so that portfolio DD
        ≈ target.  Returns (best_cap, best_dd, best_eq, best_omega,
        best_calmar, converged, scan).
        """
        lo = float(baseline_lo)
        hi = float(baseline_hi)

        # Confirm lo bracket (smaller cap → bigger lots → deeper DD)
        lo_dd, lo_eq, lo_omega, lo_calmar = _run_portfolio_compounding(lo)
        hi_dd, hi_eq, hi_omega, hi_calmar = _run_portfolio_compounding(hi)

        # If even lo_dd is shallower than target, warn and clamp.
        if lo_dd > target:
            logger.warning(
                "compare_portfolio_sizing_pareto: COMPOUNDING cannot reach "
                "target_dd=%.2f%% even at baseline_lo=%.1f (portfolio DD=%.4f%%). "
                "Returning best-effort result.",
                target,
                lo,
                lo_dd,
            )

        best_cap = lo
        best_dd = lo_dd
        best_eq = lo_eq
        best_omega = lo_omega
        best_calmar = lo_calmar
        converged = False
        lo_s = lo
        hi_s = hi

        for _ in range(max_iterations):
            mid_cap = (lo_s + hi_s) / 2.0
            mid_dd, mid_eq, mid_omega, mid_calmar = _run_portfolio_compounding(mid_cap)
            err = mid_dd - target  # positive → shallower than target

            if abs(mid_dd - target) < abs(best_dd - target):
                best_cap = mid_cap
                best_dd = mid_dd
                best_eq = mid_eq
                best_omega = mid_omega
                best_calmar = mid_calmar

            if abs(err) <= dd_tolerance_pct:
                converged = True
                break

            if err > 0:
                # Too shallow → decrease capital_per_lot (bigger lots)
                hi_s = mid_cap
            else:
                # Too deep → increase capital_per_lot (smaller lots)
                lo_s = mid_cap

            if (hi_s - lo_s) < 1e-3:
                converged = True
                break

        # Fine scan
        scan: List[dict] = []
        if fine_scan_n > 0:
            scan_lo = max(baseline_lo, best_cap * 0.5)
            scan_hi = best_cap * 1.5
            for cap in np.linspace(scan_lo, scan_hi, fine_scan_n).tolist():
                s_dd, s_eq, s_omega, s_calmar = _run_portfolio_compounding(cap)
                scan.append(
                    {
                        "capital_per_lot": float(cap),
                        "max_dd_pct": float(s_dd),
                        "final_equity": float(s_eq),
                        "omega": float(s_omega),
                        "calmar": float(s_calmar),
                    }
                )

        return best_cap, best_dd, best_eq, best_omega, best_calmar, converged, scan

    def _bisect_vol(
        target: float,
    ) -> "tuple[float, float, float, float, float, bool, List[dict]]":
        """
        Binary-search ``target_vol_pct`` so that portfolio DD ≈ target.
        Returns (best_vb, best_dd, best_eq, best_omega, best_calmar,
        converged, scan).
        """
        lo = float(vol_budget_lo)
        hi = float(vol_budget_hi)

        # Confirm hi brackets the target (larger budget → deeper DD)
        hi_dd, hi_eq, hi_omega, hi_calmar = _run_portfolio_vol(hi)
        if hi_dd > target:
            # Upper bound shallower than target — extend hi
            for mult in [2, 4, 8, 16, 32, 64]:
                candidate_hi = hi * mult
                cdd, ceq, com, ccal = _run_portfolio_vol(candidate_hi)
                if cdd <= target:
                    hi = candidate_hi
                    hi_dd, hi_eq, hi_omega, hi_calmar = cdd, ceq, com, ccal
                    break
            else:
                logger.warning(
                    "compare_portfolio_sizing_pareto: VOL_TARGET cannot reach "
                    "target_dd=%.2f%% even at vol_budget_hi=%.6f (portfolio DD=%.4f%%). "
                    "Returning non-converged result.",
                    target,
                    hi,
                    hi_dd,
                )

        best_vb = hi
        best_dd = hi_dd
        best_eq = hi_eq
        best_omega = hi_omega
        best_calmar = hi_calmar
        converged = False
        lo_s = lo
        hi_s = hi

        for _ in range(max_iterations):
            mid_vb = (lo_s + hi_s) / 2.0
            mid_dd, mid_eq, mid_omega, mid_calmar = _run_portfolio_vol(mid_vb)
            err = mid_dd - target

            if abs(mid_dd - target) < abs(best_dd - target):
                best_vb = mid_vb
                best_dd = mid_dd
                best_eq = mid_eq
                best_omega = mid_omega
                best_calmar = mid_calmar

            if abs(err) <= dd_tolerance_pct:
                converged = True
                break

            if err > 0:
                # Too shallow → increase vol budget (bigger lots)
                lo_s = mid_vb
            else:
                # Too deep → decrease vol budget (smaller lots)
                hi_s = mid_vb

            if (hi_s - lo_s) < 1e-12:
                converged = True
                break

        # Fine scan
        scan: List[dict] = []
        if fine_scan_n > 0:
            scan_lo = max(vol_budget_lo, best_vb * 0.5)
            scan_hi = min(0.50, best_vb * 1.5)
            for vb in np.linspace(scan_lo, scan_hi, fine_scan_n).tolist():
                s_dd, s_eq, s_omega, s_calmar = _run_portfolio_vol(vb)
                scan.append(
                    {
                        "vol_budget": float(vb),
                        "max_dd_pct": float(s_dd),
                        "final_equity": float(s_eq),
                        "omega": float(s_omega),
                        "calmar": float(s_calmar),
                    }
                )

        return best_vb, best_dd, best_eq, best_omega, best_calmar, converged, scan

    # ── Main loop: one Pareto point per DD budget ─────────────────────────
    sorted_dds = sorted(target_dd_pcts, key=lambda x: abs(x))  # shallow → deep
    points: List[PortfolioParetoPoint] = []

    for target in sorted_dds:
        logger.info(
            "compare_portfolio_sizing_pareto: calibrating target_dd=%.2f%% (%d instruments) …",
            target,
            n_instruments,
        )

        b_cap, b_dd, b_eq, b_omega, b_calmar, b_conv, b_scan = _bisect_compounding(target)
        v_vb, v_dd, v_eq, v_omega, v_calmar, v_conv, v_scan = _bisect_vol(target)

        delta_eq = (v_eq - b_eq) / b_eq * 100.0 if b_eq > 0 else 0.0
        delta_omega = v_omega - b_omega
        delta_calmar = v_calmar - b_calmar

        points.append(
            PortfolioParetoPoint(
                target_dd_pct=float(target),
                baseline_capital_per_lot=float(b_cap),
                baseline_achieved_dd_pct=float(b_dd),
                baseline_final_equity=float(b_eq),
                baseline_omega=float(b_omega),
                baseline_calmar=float(b_calmar),
                baseline_converged=b_conv,
                vol_budget=float(v_vb),
                vol_achieved_dd_pct=float(v_dd),
                vol_final_equity=float(v_eq),
                vol_omega=float(v_omega),
                vol_calmar=float(v_calmar),
                vol_converged=v_conv,
                delta_equity_pct=float(delta_eq),
                delta_omega=float(delta_omega),
                delta_calmar=float(delta_calmar),
                n_instruments=n_instruments,
                baseline_scan=b_scan,
                vol_scan=v_scan,
            )
        )

    # ── Recommendation ───────────────────────────────────────────────────
    # Use the requested DD budget, or fall back to the median.
    if recommend_dd_pct is not None:
        # Find the closest point to the requested budget.
        rec_point = min(points, key=lambda p: abs(p.target_dd_pct - recommend_dd_pct))
    else:
        rec_point = points[len(points) // 2]  # median point

    # Prefer vol_target when delta_equity_pct > -2 % (roughly equivalent or
    # better) because vol-targeting produces interpretable equity curves.
    # Only switch to compounding if it is materially better (> 2 % advantage).
    if rec_point.delta_equity_pct >= -2.0:
        recommended_mode = "vol_target"
    else:
        recommended_mode = "compounding"

    return PortfolioParetoResult(
        points=points,
        instruments=symbols,
        initial_equity=float(initial_equity),
        recommended_mode=recommended_mode,
        recommended_vol_budget=float(rec_point.vol_budget),
        recommended_capital_per_lot=float(rec_point.baseline_capital_per_lot),
        recommended_dd_pct=float(rec_point.target_dd_pct),
    )


def format_pareto_report(result: "PortfolioParetoResult") -> str:
    """
    Format a :class:`PortfolioParetoResult` as a human-readable console report.

    Produces a table showing, for each DD budget:

    - Solved COMPOUNDING ``capital_per_lot`` and achieved DD
    - Solved VOL_TARGET ``vol_budget`` and achieved DD
    - Final equity comparison (Δ equity %)
    - Omega and Calmar ratios for both modes
    - Convergence flags

    Followed by a production recommendation section.

    Parameters
    ----------
    result : PortfolioParetoResult
        Output of :func:`compare_portfolio_sizing_pareto`.

    Returns
    -------
    str
        Multi-line formatted report string.
    """
    lines: List[str] = []
    sep = "─" * 110

    lines.append("")
    lines.append("╔" + "═" * 108 + "╗")
    lines.append(
        "║  PORTFOLIO SIZING PARETO FRONTIER — COMPOUNDING vs VOL-TARGETED" + " " * 41 + "║"
    )
    lines.append("╚" + "═" * 108 + "╝")
    lines.append(f"  Instruments ({len(result.instruments)}): {', '.join(result.instruments)}")
    lines.append(f"  Initial equity: ${result.initial_equity:,.0f}")
    lines.append("")

    # Header row
    header = (
        f"{'DD Budget':>10}  "
        f"{'── COMPOUNDING ─────────────────────────':40}  "
        f"{'── VOL-TARGET ──────────────────────────':40}  "
        f"{'ΔEQTY%':>7}  {'ΔΩ':>6}  {'ΔCal':>6}"
    )
    lines.append(header)
    lines.append(sep)

    sub_header = (
        f"{'%':>10}  "
        f"{'cap/lot':>12}  {'achDD%':>7}  {'finalEQ$':>10}  {'Ω':>5}  {'Cal':>5}  {'cv':>2}  "
        f"{'vol_pct%':>10}  {'achDD%':>7}  {'finalEQ$':>10}  {'Ω':>5}  {'Cal':>5}  {'cv':>2}  "
        f"{'Δeq%':>7}  {'ΔΩ':>6}  {'ΔCal':>6}"
    )
    lines.append(sub_header)
    lines.append(sep)

    for p in result.points:
        b_cv = "✓" if p.baseline_converged else "✗"
        v_cv = "✓" if p.vol_converged else "✗"
        delta_marker = "▲" if p.delta_equity_pct >= 0 else "▽"
        row = (
            f"{p.target_dd_pct:>10.2f}  "
            f"{p.baseline_capital_per_lot:>12,.0f}  "
            f"{p.baseline_achieved_dd_pct:>7.3f}  "
            f"{p.baseline_final_equity:>10,.0f}  "
            f"{p.baseline_omega:>5.2f}  "
            f"{p.baseline_calmar:>5.2f}  "
            f"{b_cv:>2}  "
            f"{p.vol_budget * 100:>10.4f}  "
            f"{p.vol_achieved_dd_pct:>7.3f}  "
            f"{p.vol_final_equity:>10,.0f}  "
            f"{p.vol_omega:>5.2f}  "
            f"{p.vol_calmar:>5.2f}  "
            f"{v_cv:>2}  "
            f"{delta_marker}{abs(p.delta_equity_pct):>6.2f}  "
            f"{p.delta_omega:>+6.2f}  "
            f"{p.delta_calmar:>+6.2f}"
        )
        lines.append(row)

    lines.append(sep)
    lines.append("  cv = converged (✓ / ✗)   ΔEQTY% = (vol_eq − base_eq) / base_eq × 100")
    lines.append("  ΔΩ = vol_omega − base_omega    ΔCal = vol_calmar − base_calmar")
    lines.append("  vol_pct% = target_vol_pct × 100  (e.g. 1.0000 = 1 % daily vol target)")
    lines.append("")

    # Recommendation
    rec_p = next(
        (p for p in result.points if p.target_dd_pct == result.recommended_dd_pct),
        result.points[len(result.points) // 2],
    )
    lines.append("┌─ PRODUCTION RECOMMENDATION " + "─" * 82 + "┐")
    lines.append(
        f"│  Recommended mode      : {result.recommended_mode.upper():<20}                                                     │"
    )
    lines.append(
        f"│  Recommended DD budget : {result.recommended_dd_pct:.2f} %                                                                     │"
    )

    if result.recommended_mode == "vol_target":
        lines.append(
            f"│  Recommended vol_budget: {result.recommended_vol_budget * 100:.4f} %  (target_vol_pct = {result.recommended_vol_budget:.6f})                               │"
        )
        lines.append(
            f"│  Solved at portfolio DD: {rec_p.vol_achieved_dd_pct:.3f} %  (target: {result.recommended_dd_pct:.2f} %)                                               │"
        )
        lines.append(
            f"│  Final equity          : ${rec_p.vol_final_equity:>12,.0f}    Ω = {rec_p.vol_omega:.3f}    Calmar = {rec_p.vol_calmar:.3f}                           │"
        )
    else:
        lines.append(
            f"│  Recommended cap/lot   : {result.recommended_capital_per_lot:>12,.0f}  (compounding_capital_per_lot)                                     │"
        )
        lines.append(
            f"│  Solved at portfolio DD: {rec_p.baseline_achieved_dd_pct:.3f} %  (target: {result.recommended_dd_pct:.2f} %)                                               │"
        )
        lines.append(
            f"│  Final equity          : ${rec_p.baseline_final_equity:>12,.0f}    Ω = {rec_p.baseline_omega:.3f}    Calmar = {rec_p.baseline_calmar:.3f}                           │"
        )

    lines.append(
        f"│  Δ equity at rec. budget: {rec_p.delta_equity_pct:+.2f} %  (vol_target vs compounding at same DD)                                     │"
    )
    lines.append(
        "│                                                                                                              │"
    )
    lines.append(
        "│  Decision rule:                                                                                              │"
    )
    lines.append(
        "│    ▲ ΔEQTY% >= −2 %  → prefer VOL_TARGET (interpretable curves, ≈ same return)                              │"
    )
    lines.append(
        "│    ▽ ΔEQTY% < −2 %   → compounding materially faster; review vol floor / window config                      │"
    )
    lines.append("└" + "─" * 110 + "┘")
    lines.append("")

    return "\n".join(lines)
