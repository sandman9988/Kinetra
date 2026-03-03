"""
Renko Portfolio Environment (Layer 2 — Allocation)
===================================================

Gymnasium-compatible RL environment for portfolio allocation across
Renko-traded instruments.  The agent controls **allocation weights**
(capital distribution), while entry/exit signals are handled by the
deterministic Renko engine (Layer 1).

Design source: ``RENKO_KINETRA_DESIGN_SPEC.md §8.1``

Architecture
------------
::

    ┌─────────────────── LAYER 2 — ALLOCATION RL AGENT ────────────────────┐
    │  Observation: per-instrument features × N + portfolio-level features  │
    │  Action: weight[i] ∈ [0.0, 1.0] per instrument                       │
    │  Question: "How much capital in each instrument right now?"            │
    └───────────────────────────┬───────────────────────────────────────────┘
                                ▼
    ┌─────────────────── LAYER 1 — DETERMINISTIC ENGINE ───────────────────┐
    │  Brick construction → filter evaluation → signal → trade at weight    │
    │  Stop: 1 brick (backtest) / 0.5 brick (live)                          │
    │  Exit: colour change (first opposite brick)                           │
    └─────────────────────────────────────────────────────────────────────┘

Observation Space (per instrument × N + portfolio-level)
--------------------------------------------------------
Per-instrument (9 features):
  0. friction_ratio       : spread / brick            [0, 1]
  1. vr_current           : rolling VR at peak scale  [0.5, 2.0]
  2. flip_rate            : rolling FlipRate           [0, 1]
  3. markov_stickiness    : (pUU + pDD) / 2           [0, 1]
  4. rolling_omega        : omega over last N trades   [0, 20]  (clipped)
  5. rolling_z            : z-factor over last N       [-5, 20] (clipped)
  6. position_active      : currently in a trade?      {0, 1}
  7. time_in_position     : normalised hold time       [0, 1]
  8. drawdown_from_peak   : per-instrument DD          [-1, 0]

Portfolio-level (4 features):
  0. portfolio_dd         : current portfolio drawdown [-1, 0]
  1. herfindahl           : cluster concentration      [0, 1]
  2. n_active_positions   : normalised count           [0, 1]
  3. correlation_state    : rolling entry concurrence  [0, 1]

Total observation dim: 9 × N_instruments + 4

Action Space
------------
  Continuous Box([0, 1]^N) — allocation weight per instrument.

Step Modes
----------
  ``per_day``  : advance one calendar day per step (default)
  ``per_brick``: advance one brick arrival per step

This module is the canonical Layer 2 environment for Renko Kinetra.
All training scripts should import from here.

See Also
--------
- ``kinetra.rl.reward`` — reward functions
- ``kinetra.rl.risk_env`` — Layer 3 risk overlay environment
- ``kinetra.renko.backtest`` — deterministic backtester
- ``kinetra.renko.portfolio`` — portfolio construction
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    import gymnasium as gym
    from gymnasium import spaces

    GYM_AVAILABLE = True
except ImportError:
    gym = None  # type: ignore[assignment]
    spaces = None  # type: ignore[assignment]
    GYM_AVAILABLE = False

from kinetra.renko.filters import evaluate_entry, flip_rate, markov_stickiness
from kinetra.renko.portfolio import get_cluster, herfindahl_index

if TYPE_CHECKING:
    from kinetra.renko.backtest import FilterParams
    from kinetra.renko.dsp import DSPResult

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# Instrument context — pre-computed data for one instrument
# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class InstrumentContext:
    """
    Pre-computed data for one instrument in the portfolio environment.

    This should be created ONCE per instrument before the environment is
    instantiated.  All heavy computation (brick building, filter signals)
    is done here, not inside the env step loop.

    Attributes
    ----------
    symbol : str
        Instrument symbol (e.g. ``"XAUUSD"``).
    bricks : pd.DataFrame
        Renko bricks from :func:`~kinetra.renko.brick_engine.build_renko`.
        Must have columns: ``brick_open``, ``brick_close``, ``direction``,
        ``time``.
    brick_size : float
        Brick size in price units.
    friction_ratio : float
        Spread / brick size ratio (0–1, lower is better).
    vr_current : float
        Current VR at peak scale (from DSP analysis).
    usd_per_point : float
        USD per price point (for P&L conversion).
    cluster : str
        Cluster assignment for this instrument.
    fliprate_threshold : float
        FlipRate threshold for entry gating.
    markov_threshold : float
        Markov stickiness threshold for entry gating.
    stop_bricks : float
        Stop distance in bricks (1.0 for backtest, 0.5 for live).
    exit_on_colour_change : bool
        Whether to exit on first opposite-direction brick.
    allow_short : bool
        Whether short entries are allowed.
    fliprate_window : int
        Window for FlipRate computation.
    markov_window : int
        Window for Markov stickiness computation.
    """

    symbol: str
    bricks: pd.DataFrame
    brick_size: float
    friction_ratio: float = 0.1
    vr_current: float = 1.0
    usd_per_point: float = 1.0
    cluster: str = ""
    fliprate_threshold: float = 0.35
    markov_threshold: float = 0.55
    stop_bricks: float = 1.0
    exit_on_colour_change: bool = True
    allow_short: bool = True
    fliprate_window: int = 50
    markov_window: int = 50
    recalibrated_at: Optional[datetime] = field(default=None)

    def __post_init__(self) -> None:
        """Derive cluster if not provided and pre-compute filter arrays."""
        if not self.cluster:
            self.cluster = get_cluster(self.symbol)

        # Pre-compute filter arrays for the entire brick sequence
        if not self.bricks.empty and "direction" in self.bricks.columns:
            dirs = self.bricks["direction"].values.astype(np.int8)
            self._flip_rates = flip_rate(dirs, self.fliprate_window)
            pUU, pDD = markov_stickiness(dirs, self.markov_window)
            self._markov_pUU = pUU
            self._markov_pDD = pDD
            self._directions = dirs
            self._brick_closes = self.bricks["brick_close"].values.astype(np.float64)

            # Parse times to datetime
            times_raw = self.bricks["time"].values
            self._brick_times: List[datetime] = []
            for t in times_raw:
                ts = pd.Timestamp(t)
                if ts.tzinfo is None:
                    ts = ts.tz_localize("UTC")
                self._brick_times.append(ts.to_pydatetime())
        else:
            self._flip_rates = np.array([], dtype=np.float64)
            self._markov_pUU = np.array([], dtype=np.float64)
            self._markov_pDD = np.array([], dtype=np.float64)
            self._directions = np.array([], dtype=np.int8)
            self._brick_closes = np.array([], dtype=np.float64)
            self._brick_times = []

    @property
    def n_bricks(self) -> int:
        """Total number of bricks."""
        return len(self._brick_closes)

    def recalibrate(
        self,
        new_closes: "pd.Series",
        new_dsp_result: "DSPResult",
        new_filter_params: "FilterParams",
        session_break_minutes: float = 30.0,
    ) -> None:
        """Update structural observations after a recalibration cycle.

        Called by :class:`~kinetra.renko.qualify.CalibrationDriftDetector`
        when drift is confirmed on a live instrument.  Rebuilds the brick
        sequence from new M30 closes and updates all derived arrays so the
        Layer 2 RL agent observes fresh structural state on the next episode.

        Parameters
        ----------
        new_closes : pd.Series
            Up-to-date M30 close prices for this instrument.
        new_dsp_result : DSPResult
            Fresh DSP analysis result (from :func:`~kinetra.renko.dsp.run_dsp`).
        new_filter_params : FilterParams
            New filter thresholds derived via
            :func:`~kinetra.renko.dsp.scaled_filter_params`.
        session_break_minutes : float
            Gap threshold for brick construction — must come from
            :func:`~kinetra.renko.session.detect_session_break` (§29.3).

        Notes
        -----
        - ``vr_current``, ``brick_size``, ``fliprate_threshold``, and
          ``markov_threshold`` are all updated atomically.
        - The internal pre-computed arrays (``_flip_rates``, ``_directions``,
          etc.) are fully rebuilt so no stale signal leaks into RL episodes.
        - ``recalibrated_at`` is set to the current UTC time.
        - The RL agent does **not** need retraining: it already observes
          ``vr_current`` as a feature.  An updated value naturally shifts the
          agent's allocation weight without any reward-function changes.
        """
        from kinetra.renko.brick_engine import build_renko

        # 1. Rebuild brick sequence with the new brick size and session break.
        new_brick_size = new_dsp_result.dsp_brick_size
        new_bricks = build_renko(
            new_closes,
            brick_size=new_brick_size,
            session_break_minutes=session_break_minutes,
        )

        # 2. Update scalar structural observations.
        self.brick_size = new_brick_size
        self.vr_current = new_dsp_result.vr_peak_value
        self.fliprate_threshold = new_filter_params.fliprate_threshold
        self.markov_threshold = new_filter_params.markov_threshold
        self.fliprate_window = new_filter_params.fliprate_window
        self.markov_window = new_filter_params.markov_window

        # 3. Rebuild derived arrays (same logic as __post_init__).
        self.bricks = new_bricks
        if not new_bricks.empty and "direction" in new_bricks.columns:
            dirs = new_bricks["direction"].values.astype(np.int8)
            self._flip_rates = flip_rate(dirs, self.fliprate_window)
            pUU, pDD = markov_stickiness(dirs, self.markov_window)
            self._markov_pUU = pUU
            self._markov_pDD = pDD
            self._directions = dirs
            self._brick_closes = new_bricks["brick_close"].values.astype(np.float64)
            times_raw = new_bricks["time"].values
            self._brick_times = []
            for t in times_raw:
                ts = pd.Timestamp(t)
                if ts.tzinfo is None:
                    ts = ts.tz_localize("UTC")
                self._brick_times.append(ts.to_pydatetime())
        else:
            self._flip_rates = np.array([], dtype=np.float64)
            self._markov_pUU = np.array([], dtype=np.float64)
            self._markov_pDD = np.array([], dtype=np.float64)
            self._directions = np.array([], dtype=np.int8)
            self._brick_closes = np.array([], dtype=np.float64)
            self._brick_times = []

        # 4. Stamp recalibration time.
        self.recalibrated_at = datetime.now(tz=timezone.utc)

        logger.info(
            "InstrumentContext.recalibrate [%s]: brick_size %.5f → %.5f, "
            "vr_current → %.3f, fliprate_threshold → %.3f, "
            "markov_threshold → %.3f, n_bricks=%d",
            self.symbol,
            new_brick_size,
            new_brick_size,
            self.vr_current,
            self.fliprate_threshold,
            self.markov_threshold,
            len(self._brick_closes),
        )


# ══════════════════════════════════════════════════════════════════════════════
# Per-instrument position state
# ══════════════════════════════════════════════════════════════════════════════


@dataclass(slots=True)
class _PositionState:
    """Mutable position state for one instrument during simulation."""

    in_position: bool = False
    direction: int = 0  # +1 long, -1 short
    entry_price: float = 0.0
    entry_time: Optional[datetime] = None
    entry_brick_idx: int = 0
    # Rolling metrics
    cumulative_pnl: float = 0.0
    peak_pnl: float = 0.0
    trade_returns: deque = field(default_factory=lambda: deque(maxlen=50))
    n_trades: int = 0
    # Brick cursor (current position in the brick array)
    brick_cursor: int = 0
    # MFE/MAE tracking for open position
    running_mfe: float = 0.0
    running_mae: float = 0.0

    def reset(self, start_brick: int = 0) -> None:
        """Reset position state for a new episode."""
        self.in_position = False
        self.direction = 0
        self.entry_price = 0.0
        self.entry_time = None
        self.entry_brick_idx = 0
        self.cumulative_pnl = 0.0
        self.peak_pnl = 0.0
        self.trade_returns.clear()
        self.n_trades = 0
        self.brick_cursor = start_brick
        self.running_mfe = 0.0
        self.running_mae = 0.0


# ══════════════════════════════════════════════════════════════════════════════
# Environment configuration
# ══════════════════════════════════════════════════════════════════════════════


@dataclass(slots=True)
class PortfolioEnvConfig:
    """
    Configuration for the RenkoPortfolioEnv.

    Attributes
    ----------
    step_mode : str
        ``"per_day"`` or ``"per_brick"``.  Controls how many bricks are
        processed per env step.
    episode_days : int
        Length of one episode in calendar days.  Only used when
        ``step_mode="per_day"``.
    episode_bricks : int
        Length of one episode in total bricks processed.  Only used when
        ``step_mode="per_brick"``.
    rolling_metric_window : int
        Window size for rolling omega/z-factor computation (trades).
    max_hold_hours : float
        Maximum hold time for normalisation of ``time_in_position``.
    initial_equity : float
        Starting equity for each episode (for drawdown normalisation).
    random_start : bool
        If True, start at a random offset within the data on each reset.
    seed : int or None
        Random seed for reproducibility.
    warmup_bricks : int
        Minimum bricks before the filter signals are valid.
    """

    step_mode: str = "per_day"
    episode_days: int = 180
    episode_bricks: int = 5000
    rolling_metric_window: int = 50
    max_hold_hours: float = 720.0  # 30 days
    initial_equity: float = 10000.0
    random_start: bool = True
    seed: Optional[int] = None
    warmup_bricks: int = 55  # max(fliprate_window, markov_window) + buffer


# ══════════════════════════════════════════════════════════════════════════════
# Features per instrument (constants)
# ══════════════════════════════════════════════════════════════════════════════

N_PER_INSTRUMENT_FEATURES = 9
N_PORTFOLIO_FEATURES = 4

_OMEGA_LO, _OMEGA_HI = 0.0, 20.0
_Z_LO, _Z_HI = -5.0, 20.0


# ══════════════════════════════════════════════════════════════════════════════
# RenkoPortfolioEnv
# ══════════════════════════════════════════════════════════════════════════════


class RenkoPortfolioEnv:
    """
    RL environment for portfolio allocation across Renko instruments.

    This is the Layer 2 (allocation) environment in the Renko Kinetra
    three-layer architecture.  The agent outputs allocation weights per
    instrument; the deterministic Renko engine handles entry/exit signals.

    Implements the Gymnasium ``Env`` interface (``reset``, ``step``,
    ``observation_space``, ``action_space``).  When gymnasium is not
    installed, the class still works but ``observation_space`` and
    ``action_space`` are plain dicts instead of ``gymnasium.spaces``.

    Parameters
    ----------
    instruments : list of InstrumentContext
        Pre-computed instrument data.  At least one instrument required.
    config : PortfolioEnvConfig or None
        Environment configuration.
    reward_config : AllocationRewardConfig or None
        Reward function configuration (from ``kinetra.rl.reward``).

    Raises
    ------
    ValueError
        If no instruments are provided or all instruments have empty
        brick data.

    Examples
    --------
    ::

        from kinetra.rl.portfolio_env import RenkoPortfolioEnv, InstrumentContext
        from kinetra.renko.brick_engine import build_renko

        ctx = InstrumentContext(
            symbol="XAUUSD",
            bricks=build_renko(closes, brick_size=5.0),
            brick_size=5.0,
        )
        env = RenkoPortfolioEnv(instruments=[ctx])
        obs, info = env.reset()
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
    """

    # Gymnasium metadata
    metadata: Dict[str, Any] = {"render_modes": []}

    def __init__(
        self,
        instruments: Sequence[InstrumentContext],
        config: Optional[PortfolioEnvConfig] = None,
        reward_config: Optional[Any] = None,
    ) -> None:
        if not instruments:
            raise ValueError("At least one InstrumentContext is required")

        self._instruments = list(instruments)
        self._n_instruments = len(self._instruments)
        self._config = config or PortfolioEnvConfig()
        self._reward_config = reward_config

        # Lazy import to avoid circular deps
        self._reward_module: Optional[Any] = None

        # Validate that at least one instrument has bricks
        valid = [ctx for ctx in self._instruments if ctx.n_bricks > 0]
        if not valid:
            raise ValueError("All instruments have empty brick data — cannot create environment")

        # Cluster lookup for portfolio-level tracking
        self._cluster_map: Dict[str, str] = {ctx.symbol: ctx.cluster for ctx in self._instruments}
        self._symbol_to_idx: Dict[str, int] = {
            ctx.symbol: i for i, ctx in enumerate(self._instruments)
        }

        # ── Observation / Action spaces ──────────────────────────────
        self._obs_dim = N_PER_INSTRUMENT_FEATURES * self._n_instruments + N_PORTFOLIO_FEATURES

        if GYM_AVAILABLE and spaces is not None:
            self.observation_space = spaces.Box(
                low=-1.0,
                high=20.0,
                shape=(self._obs_dim,),
                dtype=np.float32,
            )
            self.action_space = spaces.Box(
                low=0.0,
                high=1.0,
                shape=(self._n_instruments,),
                dtype=np.float32,
            )
        else:
            # Fallback when gymnasium is not installed
            self.observation_space = {
                "shape": (self._obs_dim,),
                "low": -1.0,
                "high": 20.0,
                "dtype": "float32",
            }
            self.action_space = {
                "shape": (self._n_instruments,),
                "low": 0.0,
                "high": 1.0,
                "dtype": "float32",
            }

        # ── Per-instrument state ─────────────────────────────────────
        self._positions: List[_PositionState] = [_PositionState() for _ in self._instruments]

        # ── Portfolio-level state ────────────────────────────────────
        self._portfolio_equity: float = 0.0
        self._portfolio_peak: float = 0.0
        self._portfolio_dd: float = 0.0
        self._current_weights: np.ndarray = np.ones(self._n_instruments, dtype=np.float32)
        self._step_count: int = 0
        self._episode_trades: List[Any] = []
        self._episode_returns: List[float] = []
        self._current_day: Optional[datetime] = None
        self._episode_end_day: Optional[datetime] = None

        # ── Entry concurrence tracking ───────────────────────────────
        self._recent_entry_times: deque = deque(maxlen=100)

        # ── RNG ──────────────────────────────────────────────────────
        self._rng = np.random.default_rng(self._config.seed)

        # ── Reward tracker (optional) ────────────────────────────────
        self._reward_tracker: Optional[Any] = None

        logger.info(
            "RenkoPortfolioEnv created: %d instruments, obs_dim=%d, step_mode=%s, episode_days=%d",
            self._n_instruments,
            self._obs_dim,
            self._config.step_mode,
            self._config.episode_days,
        )

    # ──────────────────────────────────────────────────────────────────────
    # Properties
    # ──────────────────────────────────────────────────────────────────────

    @property
    def n_instruments(self) -> int:
        """Number of instruments in the portfolio."""
        return self._n_instruments

    @property
    def instruments(self) -> List[InstrumentContext]:
        """List of instrument contexts."""
        return list(self._instruments)

    @property
    def obs_dim(self) -> int:
        """Observation vector dimension."""
        return self._obs_dim

    @property
    def current_weights(self) -> np.ndarray:
        """Current allocation weights (copy)."""
        return self._current_weights.copy()

    @property
    def portfolio_equity(self) -> float:
        """Current portfolio equity (cumulative P&L)."""
        return self._portfolio_equity

    @property
    def step_count(self) -> int:
        """Steps taken in the current episode."""
        return self._step_count

    # ──────────────────────────────────────────────────────────────────────
    # Gymnasium interface: reset
    # ──────────────────────────────────────────────────────────────────────

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset the environment for a new episode.

        Parameters
        ----------
        seed : int or None
            Optional seed override for this episode.
        options : dict or None
            Optional reset options (currently unused).

        Returns
        -------
        tuple[np.ndarray, dict]
            ``(observation, info)`` following the Gymnasium API.
        """
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        # ── Determine start positions for each instrument ────────────
        warmup = self._config.warmup_bricks

        if self._config.random_start:
            # Find the time range where all instruments have data
            min_start_brick = self._find_min_common_start(warmup)
            max_start_brick = self._find_max_common_start(warmup)

            if max_start_brick > min_start_brick:
                random_offset = self._rng.integers(0, max_start_brick - min_start_brick)
            else:
                random_offset = 0

            for i, ctx in enumerate(self._instruments):
                effective_warmup = min(warmup, max(ctx.n_bricks - 2, 0))
                start = effective_warmup + random_offset
                start = min(start, max(ctx.n_bricks - 2, 0))
                self._positions[i].reset(start_brick=start)
        else:
            for i, ctx in enumerate(self._instruments):
                effective_warmup = min(warmup, max(ctx.n_bricks - 2, 0))
                self._positions[i].reset(start_brick=effective_warmup)

        # ── Reset portfolio state ────────────────────────────────────
        self._portfolio_equity = 0.0
        self._portfolio_peak = 0.0
        self._portfolio_dd = 0.0
        self._current_weights = np.ones(self._n_instruments, dtype=np.float32) / self._n_instruments
        self._step_count = 0
        self._episode_trades = []
        self._episode_returns = []
        self._recent_entry_times.clear()

        # ── Determine episode time bounds ────────────────────────────
        self._current_day = self._find_earliest_current_time()
        if self._current_day is not None:
            self._episode_end_day = self._current_day + timedelta(days=self._config.episode_days)
        else:
            self._episode_end_day = None

        # ── Reset reward tracker ─────────────────────────────────────
        if self._reward_tracker is not None:
            self._reward_tracker.reset()

        obs = self._build_observation()
        info = self._build_info()
        return obs, info

    # ──────────────────────────────────────────────────────────────────────
    # Gymnasium interface: step
    # ──────────────────────────────────────────────────────────────────────

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one environment step.

        1. Apply allocation weights from action.
        2. Advance simulation (process bricks for one day or one brick).
        3. Execute pending Renko signals at current weights.
        4. Compute reward.

        Parameters
        ----------
        action : np.ndarray
            Allocation weights, shape ``(n_instruments,)``.
            Values are clipped to [0, 1].

        Returns
        -------
        tuple
            ``(observation, reward, terminated, truncated, info)``
            following the Gymnasium API.
        """
        # ── Validate and apply action ────────────────────────────────
        weights = np.asarray(action, dtype=np.float32).flatten()
        if len(weights) != self._n_instruments:
            # Pad or truncate to match
            padded = np.zeros(self._n_instruments, dtype=np.float32)
            n = min(len(weights), self._n_instruments)
            padded[:n] = weights[:n]
            weights = padded

        weights = np.clip(weights, 0.0, 1.0)
        self._current_weights = weights

        # ── Track previous equity for reward ─────────────────────────
        prev_equity = self._portfolio_equity

        # ── Advance simulation ───────────────────────────────────────
        step_trades: List[Any] = []

        if self._config.step_mode == "per_brick":
            trades = self._advance_one_brick()
            step_trades.extend(trades)
        else:  # per_day
            trades = self._advance_one_day()
            step_trades.extend(trades)

        self._step_count += 1

        # ── Record trades ────────────────────────────────────────────
        for trade_info in step_trades:
            self._episode_trades.append(trade_info)
            self._episode_returns.append(trade_info["net_pnl"])

        # ── Update portfolio-level state ─────────────────────────────
        self._portfolio_peak = max(self._portfolio_peak, self._portfolio_equity)
        if self._portfolio_peak > 0:
            self._portfolio_dd = (
                self._portfolio_equity - self._portfolio_peak
            ) / self._portfolio_peak
        else:
            self._portfolio_dd = min(0.0, self._portfolio_equity - self._portfolio_peak)

        # ── Check termination ────────────────────────────────────────
        terminated = self._check_terminated()
        truncated = self._check_truncated()

        # ── Compute reward ───────────────────────────────────────────
        reward = self._compute_step_reward(
            step_trades=step_trades,
            prev_equity=prev_equity,
            terminated=terminated or truncated,
        )

        # ── Build observation and info ───────────────────────────────
        obs = self._build_observation()
        info = self._build_info()
        info["step_trades"] = len(step_trades)
        info["step_pnl"] = sum(t["net_pnl"] for t in step_trades)

        return obs, float(reward), terminated, truncated, info

    # ──────────────────────────────────────────────────────────────────────
    # Simulation: advance by one day
    # ──────────────────────────────────────────────────────────────────────

    def _advance_one_day(self) -> List[Dict[str, Any]]:
        """
        Process all bricks across all instruments for one calendar day.

        Returns list of completed trade dicts.
        """
        if self._current_day is None:
            return []

        day_end = self._current_day + timedelta(days=1)
        all_trades: List[Dict[str, Any]] = []

        for i, ctx in enumerate(self._instruments):
            pos = self._positions[i]
            weight = float(self._current_weights[i])

            # Process bricks until we pass the current day
            while pos.brick_cursor < ctx.n_bricks:
                if pos.brick_cursor >= len(ctx._brick_times):
                    break

                brick_time = ctx._brick_times[pos.brick_cursor]
                if brick_time >= day_end:
                    break  # This brick belongs to the next day

                trades = self._process_brick(i, weight)
                all_trades.extend(trades)
                pos.brick_cursor += 1

        self._current_day = day_end
        return all_trades

    # ──────────────────────────────────────────────────────────────────────
    # Simulation: advance by one brick (earliest across all instruments)
    # ──────────────────────────────────────────────────────────────────────

    def _advance_one_brick(self) -> List[Dict[str, Any]]:
        """
        Process the next brick (the one with earliest time across instruments).

        Returns list of completed trade dicts (usually 0 or 1).
        """
        # Find instrument with the earliest next brick
        earliest_time: Optional[datetime] = None
        earliest_idx: Optional[int] = None

        for i, ctx in enumerate(self._instruments):
            pos = self._positions[i]
            if pos.brick_cursor < ctx.n_bricks and pos.brick_cursor < len(ctx._brick_times):
                t = ctx._brick_times[pos.brick_cursor]
                if earliest_time is None or t < earliest_time:
                    earliest_time = t
                    earliest_idx = i

        if earliest_idx is None:
            return []

        weight = float(self._current_weights[earliest_idx])
        trades = self._process_brick(earliest_idx, weight)
        self._positions[earliest_idx].brick_cursor += 1

        # Update current day tracker
        if earliest_time is not None:
            self._current_day = earliest_time

        return trades

    # ──────────────────────────────────────────────────────────────────────
    # Core brick processing (deterministic engine)
    # ──────────────────────────────────────────────────────────────────────

    def _process_brick(
        self,
        inst_idx: int,
        weight: float,
    ) -> List[Dict[str, Any]]:
        """
        Process a single brick for one instrument.

        Handles exit checks (stop, colour change) and entry checks
        (flip + filter pass) deterministically.

        Parameters
        ----------
        inst_idx : int
            Index into ``self._instruments``.
        weight : float
            Current allocation weight for this instrument.

        Returns
        -------
        list[dict]
            List of completed trade info dicts (0 or 1 entries).
        """
        ctx = self._instruments[inst_idx]
        pos = self._positions[inst_idx]
        trades: List[Dict[str, Any]] = []

        bi = pos.brick_cursor
        if bi >= ctx.n_bricks:
            return trades

        price = float(ctx._brick_closes[bi])
        bd = int(ctx._directions[bi])
        bt = ctx._brick_times[bi] if bi < len(ctx._brick_times) else None

        # ── Check exits ──────────────────────────────────────────────
        if pos.in_position:
            stop_distance = ctx.stop_bricks * ctx.brick_size
            stop_hit = (pos.direction == 1 and price <= pos.entry_price - stop_distance) or (
                pos.direction == -1 and price >= pos.entry_price + stop_distance
            )
            colour_change = ctx.exit_on_colour_change and bd != pos.direction

            if stop_hit or colour_change:
                reason = "stop" if stop_hit else "colour_change"
                trade = self._close_position(inst_idx, price, bt, reason, weight)
                if trade:
                    trades.append(trade)
            else:
                # Update MFE/MAE for open position
                unrealised = (price - pos.entry_price) * pos.direction * ctx.usd_per_point
                pos.running_mfe = max(pos.running_mfe, unrealised)
                pos.running_mae = min(pos.running_mae, unrealised)

        # ── Check entries ────────────────────────────────────────────
        if not pos.in_position and bi > 0:
            prev_dir = int(ctx._directions[bi - 1]) if bi > 0 else 0
            is_flip = bd != prev_dir

            if is_flip and weight > 0.01:  # Don't enter at near-zero weight
                # Get filter values
                fr_val = (
                    float(ctx._flip_rates[bi])
                    if bi < len(ctx._flip_rates) and np.isfinite(ctx._flip_rates[bi])
                    else float("nan")
                )
                pUU_val = (
                    float(ctx._markov_pUU[bi])
                    if bi < len(ctx._markov_pUU) and np.isfinite(ctx._markov_pUU[bi])
                    else float("nan")
                )
                pDD_val = (
                    float(ctx._markov_pDD[bi])
                    if bi < len(ctx._markov_pDD) and np.isfinite(ctx._markov_pDD[bi])
                    else float("nan")
                )

                entry_ok = evaluate_entry(
                    direction=bd,
                    flip_rate_val=fr_val,
                    pUU=pUU_val,
                    pDD=pDD_val,
                    fliprate_threshold=ctx.fliprate_threshold,
                    markov_threshold=ctx.markov_threshold,
                )

                if entry_ok:
                    if bd == 1 or (bd == -1 and ctx.allow_short):
                        pos.in_position = True
                        pos.direction = bd
                        pos.entry_price = price
                        pos.entry_time = bt
                        pos.entry_brick_idx = bi
                        pos.running_mfe = 0.0
                        pos.running_mae = 0.0
                        # Track entry concurrence
                        if bt is not None:
                            self._recent_entry_times.append(bt)

        return trades

    def _close_position(
        self,
        inst_idx: int,
        exit_price: float,
        exit_time: Optional[datetime],
        reason: str,
        weight: float,
    ) -> Optional[Dict[str, Any]]:
        """
        Close the current position for an instrument and record the trade.

        Returns trade info dict, or None if no position was open.
        """
        ctx = self._instruments[inst_idx]
        pos = self._positions[inst_idx]

        if not pos.in_position:
            return None

        # Compute P&L
        gross_pts = (exit_price - pos.entry_price) * pos.direction
        gross_usd = gross_pts * ctx.usd_per_point

        # Estimate friction (simplified: friction_ratio × |gross|)
        friction_usd = abs(gross_usd) * ctx.friction_ratio
        net_usd = gross_usd - friction_usd

        # Scale by allocation weight
        scaled_net = net_usd * weight
        scaled_gross = gross_usd * weight
        scaled_friction = friction_usd * weight

        # Holding time
        holding_hours = 0.0
        if pos.entry_time is not None and exit_time is not None:
            holding_hours = (exit_time - pos.entry_time).total_seconds() / 3600.0

        # Build trade record
        trade = {
            "symbol": ctx.symbol,
            "direction": pos.direction,
            "entry_price": pos.entry_price,
            "exit_price": exit_price,
            "entry_time": pos.entry_time,
            "exit_time": exit_time,
            "gross_pts": gross_pts,
            "gross_usd": scaled_gross,
            "friction_usd": scaled_friction,
            "net_pnl": scaled_net,
            "exit_reason": reason,
            "brick_size": ctx.brick_size,
            "weight": weight,
            "mfe": pos.running_mfe * weight,
            "mae": pos.running_mae * weight,
            "holding_hours": holding_hours,
            "cluster": ctx.cluster,
        }

        # Update position state
        pos.cumulative_pnl += scaled_net
        pos.peak_pnl = max(pos.peak_pnl, pos.cumulative_pnl)
        pos.trade_returns.append(scaled_net)
        pos.n_trades += 1
        pos.in_position = False
        pos.direction = 0
        pos.entry_price = 0.0
        pos.entry_time = None
        pos.running_mfe = 0.0
        pos.running_mae = 0.0

        # Update portfolio equity
        self._portfolio_equity += scaled_net

        return trade

    # ──────────────────────────────────────────────────────────────────────
    # Observation builder
    # ──────────────────────────────────────────────────────────────────────

    def _build_observation(self) -> np.ndarray:
        """
        Build the observation vector.

        Layout: [inst_0_features, inst_1_features, ..., portfolio_features]
        """
        obs = np.zeros(self._obs_dim, dtype=np.float32)
        offset = 0

        for i, ctx in enumerate(self._instruments):
            pos = self._positions[i]
            bi = min(pos.brick_cursor, ctx.n_bricks - 1) if ctx.n_bricks > 0 else 0

            # 0. friction_ratio [0, 1]
            obs[offset + 0] = np.clip(ctx.friction_ratio, 0.0, 1.0)

            # 1. vr_current [0.5, 2.0]
            obs[offset + 1] = np.clip(ctx.vr_current, 0.5, 2.0)

            # 2. flip_rate (current) [0, 1]
            if bi < len(ctx._flip_rates) and np.isfinite(ctx._flip_rates[bi]):
                obs[offset + 2] = np.clip(float(ctx._flip_rates[bi]), 0.0, 1.0)
            else:
                obs[offset + 2] = 0.5  # neutral default

            # 3. markov_stickiness ((pUU + pDD) / 2) [0, 1]
            if bi < len(ctx._markov_pUU) and np.isfinite(ctx._markov_pUU[bi]):
                pUU = float(ctx._markov_pUU[bi])
                pDD = float(ctx._markov_pDD[bi]) if np.isfinite(ctx._markov_pDD[bi]) else 0.5
                obs[offset + 3] = np.clip((pUU + pDD) / 2.0, 0.0, 1.0)
            else:
                obs[offset + 3] = 0.5

            # 4. rolling_omega [0, 20]
            omega = self._compute_rolling_omega(i)
            obs[offset + 4] = np.clip(omega, _OMEGA_LO, _OMEGA_HI)

            # 5. rolling_z [-5, 20]
            z = self._compute_rolling_z(i)
            obs[offset + 5] = np.clip(z, _Z_LO, _Z_HI)

            # 6. position_active {0, 1}
            obs[offset + 6] = 1.0 if pos.in_position else 0.0

            # 7. time_in_position (normalised) [0, 1]
            if pos.in_position and pos.entry_time is not None and bi < len(ctx._brick_times):
                current_time = ctx._brick_times[bi]
                hold_hours = (current_time - pos.entry_time).total_seconds() / 3600.0
                obs[offset + 7] = np.clip(hold_hours / self._config.max_hold_hours, 0.0, 1.0)
            else:
                obs[offset + 7] = 0.0

            # 8. drawdown_from_peak [-1, 0]
            if pos.peak_pnl > 0:
                dd = (pos.cumulative_pnl - pos.peak_pnl) / pos.peak_pnl
                obs[offset + 8] = np.clip(dd, -1.0, 0.0)
            else:
                obs[offset + 8] = 0.0

            offset += N_PER_INSTRUMENT_FEATURES

        # ── Portfolio-level features ─────────────────────────────────
        # 0. portfolio_dd [-1, 0]
        obs[offset + 0] = np.clip(self._portfolio_dd, -1.0, 0.0)

        # 1. herfindahl [0, 1]
        cluster_weights = self._compute_cluster_weights()
        hhi = herfindahl_index(cluster_weights) if cluster_weights else 1.0
        obs[offset + 1] = np.clip(hhi, 0.0, 1.0)

        # 2. n_active_positions (normalised) [0, 1]
        n_active = sum(1 for pos in self._positions if pos.in_position)
        obs[offset + 2] = n_active / max(self._n_instruments, 1)

        # 3. correlation_state (entry concurrence) [0, 1]
        obs[offset + 3] = self._compute_entry_concurrence()

        return obs

    # ──────────────────────────────────────────────────────────────────────
    # Reward computation
    # ──────────────────────────────────────────────────────────────────────

    def _compute_step_reward(
        self,
        step_trades: List[Dict[str, Any]],
        prev_equity: float,
        terminated: bool,
    ) -> float:
        """
        Compute the reward for this step using the canonical reward module.

        Falls back to a simple P&L-based reward if the reward module is
        not available.
        """
        try:
            return self._compute_reward_canonical(step_trades, prev_equity, terminated)
        except Exception as exc:
            logger.debug("Reward computation fallback: %s", exc)
            return self._compute_reward_simple(step_trades, prev_equity, terminated)

    def _compute_reward_canonical(
        self,
        step_trades: List[Dict[str, Any]],
        prev_equity: float,
        terminated: bool,
    ) -> float:
        """Compute reward using the three-component canonical reward."""
        if self._reward_module is None:
            from kinetra.rl.reward import (
                AllocationRewardConfig,
                EpisodeTerminalState,
                PortfolioState,
                TradeOutcome,
                compute_allocation_reward,
            )

            self._reward_module = {
                "compute_allocation_reward": compute_allocation_reward,
                "TradeOutcome": TradeOutcome,
                "PortfolioState": PortfolioState,
                "EpisodeTerminalState": EpisodeTerminalState,
                "AllocationRewardConfig": AllocationRewardConfig,
            }

        compute_fn = self._reward_module["compute_allocation_reward"]
        TradeOutcome = self._reward_module["TradeOutcome"]
        PortfolioState = self._reward_module["PortfolioState"]
        EpisodeTerminalState = self._reward_module["EpisodeTerminalState"]

        # Build trade outcomes
        outcomes = []
        for t in step_trades:
            outcomes.append(
                TradeOutcome(
                    symbol=t["symbol"],
                    net_pnl=t["net_pnl"],
                    gross_pnl=t["gross_usd"],
                    friction_usd=t["friction_usd"],
                    allocation_weight=t["weight"],
                    mfe=t.get("mfe", 0.0),
                    mae=t.get("mae", 0.0),
                    holding_hours=t.get("holding_hours", 0.0),
                    cluster=t.get("cluster", "unknown"),
                )
            )

        # Build portfolio state
        active_clusters: Dict[str, int] = {}
        for i, pos in enumerate(self._positions):
            if pos.in_position:
                cl = self._cluster_map.get(self._instruments[i].symbol, "unknown")
                active_clusters[cl] = active_clusters.get(cl, 0) + 1

        cluster_weights = self._compute_cluster_weights()
        hhi = herfindahl_index(cluster_weights) if cluster_weights else 1.0
        n_active = sum(1 for p in self._positions if p.in_position)

        port_state = PortfolioState(
            equity=self._portfolio_equity,
            prev_equity=prev_equity,
            peak_equity=self._portfolio_peak,
            drawdown=self._portfolio_dd,
            n_active_positions=n_active,
            n_instruments=self._n_instruments,
            active_clusters=active_clusters,
            herfindahl=hhi,
            n_distinct_clusters_active=len(active_clusters),
        )

        # Build terminal state if episode is ending
        terminal = None
        if terminated:
            returns = np.array(self._episode_returns, dtype=np.float64)
            omega = self._compute_portfolio_omega(returns)
            max_dd = self._portfolio_dd
            years = self._config.episode_days / 365.25
            from kinetra.renko.portfolio import calmar_ratio as calmar_fn

            calmar = calmar_fn(sum(self._episode_returns), max_dd, years)

            terminal = EpisodeTerminalState(
                portfolio_omega=omega,
                portfolio_calmar=calmar,
                portfolio_herfindahl=hhi,
                total_trades=len(self._episode_trades),
                total_net_pnl=sum(self._episode_returns),
                max_drawdown=max_dd,
            )

        return compute_fn(
            trade_outcomes=outcomes,
            portfolio_state=port_state,
            terminal=terminal,
            config=self._reward_config,
        )

    def _compute_reward_simple(
        self,
        step_trades: List[Dict[str, Any]],
        prev_equity: float,
        terminated: bool,
    ) -> float:
        """Simple fallback reward: equity change normalised by instrument count."""
        equity_delta = self._portfolio_equity - prev_equity
        reward = equity_delta / max(self._n_instruments, 1)

        # Small terminal bonus for positive episode
        if terminated and self._portfolio_equity > 0:
            reward += 0.1

        return float(np.clip(reward, -10.0, 10.0))

    # ──────────────────────────────────────────────────────────────────────
    # Rolling metric helpers
    # ──────────────────────────────────────────────────────────────────────

    def _compute_rolling_omega(self, inst_idx: int) -> float:
        """Compute rolling Omega ratio for an instrument from recent trades."""
        pos = self._positions[inst_idx]
        if len(pos.trade_returns) < 3:
            return 1.0  # neutral default

        returns = np.array(list(pos.trade_returns), dtype=np.float64)
        try:
            from kinetra.backtesting.metrics import omega_ratio

            return float(omega_ratio(returns))
        except Exception:
            # Ultra-simple fallback
            gains = float(returns[returns > 0].sum())
            losses = float(-returns[returns <= 0].sum())
            if losses > 0:
                return gains / losses
            return 10.0 if gains > 0 else 0.0

    def _compute_rolling_z(self, inst_idx: int) -> float:
        """Compute rolling Z-factor for an instrument from recent trades."""
        pos = self._positions[inst_idx]
        if len(pos.trade_returns) < 3:
            return 0.0  # neutral default

        returns = np.array(list(pos.trade_returns), dtype=np.float64)
        try:
            from kinetra.backtesting.metrics import calculate_z_factor

            return float(calculate_z_factor(returns))
        except Exception:
            n = len(returns)
            if n < 2:
                return 0.0
            mean = returns.mean()
            std = returns.std()
            if std < 1e-12:
                return 0.0
            return float(mean * np.sqrt(n) / std)

    def _compute_portfolio_omega(self, returns: np.ndarray) -> float:
        """Compute portfolio-level Omega from episode returns."""
        if len(returns) < 2:
            return 0.0
        try:
            from kinetra.backtesting.metrics import omega_ratio

            return float(omega_ratio(returns))
        except Exception:
            gains = float(returns[returns > 0].sum())
            losses = float(-returns[returns <= 0].sum())
            if losses > 0:
                return gains / losses
            return 10.0 if gains > 0 else 0.0

    # ──────────────────────────────────────────────────────────────────────
    # Portfolio-level helpers
    # ──────────────────────────────────────────────────────────────────────

    def _compute_cluster_weights(self) -> Dict[str, float]:
        """Compute cluster-level weight sums from current allocation."""
        cluster_weights: Dict[str, float] = {}
        for i, ctx in enumerate(self._instruments):
            w = float(self._current_weights[i])
            cl = ctx.cluster
            cluster_weights[cl] = cluster_weights.get(cl, 0.0) + w
        return cluster_weights

    def _compute_entry_concurrence(self) -> float:
        """
        Compute entry concurrence: fraction of recent entries that
        occurred close together in time (within 1 hour).

        High concurrence suggests correlated instruments entering
        simultaneously — a risk signal.
        """
        if len(self._recent_entry_times) < 2:
            return 0.0

        times = sorted(self._recent_entry_times)
        n_concurrent = 0
        n_pairs = 0

        # Check last 20 entries for concurrence
        recent = list(times)[-20:]
        for j in range(len(recent)):
            for k in range(j + 1, len(recent)):
                n_pairs += 1
                dt = abs((recent[k] - recent[j]).total_seconds())
                if dt < 3600:  # within 1 hour
                    n_concurrent += 1

        if n_pairs == 0:
            return 0.0
        return np.clip(n_concurrent / n_pairs, 0.0, 1.0)

    # ──────────────────────────────────────────────────────────────────────
    # Termination checks
    # ──────────────────────────────────────────────────────────────────────

    def _check_terminated(self) -> bool:
        """
        Check if the episode should terminate (natural end).

        Termination occurs when:
        - All instruments have exhausted their brick data, OR
        - The episode time limit has been reached (per_day mode), OR
        - The episode brick limit has been reached (per_brick mode).
        """
        # All instruments exhausted
        all_done = all(
            self._positions[i].brick_cursor >= ctx.n_bricks
            for i, ctx in enumerate(self._instruments)
        )
        if all_done:
            # Force-close all open positions
            self._force_close_all()
            return True

        # Time limit (per_day mode)
        if self._config.step_mode == "per_day":
            if self._current_day is not None and self._episode_end_day is not None:
                if self._current_day >= self._episode_end_day:
                    self._force_close_all()
                    return True

        # Brick limit (per_brick mode)
        if self._config.step_mode == "per_brick":
            total_bricks = sum(p.brick_cursor for p in self._positions)
            if total_bricks >= self._config.episode_bricks:
                self._force_close_all()
                return True

        return False

    def _check_truncated(self) -> bool:
        """
        Check if the episode should be truncated (safety limit).

        Truncation is a safety net for unexpected conditions.
        """
        # Truncate if step count exceeds a generous limit
        max_steps = self._config.episode_days * 2 if self._config.step_mode == "per_day" else 50000
        return self._step_count >= max_steps

    def _force_close_all(self) -> None:
        """Force-close all open positions at current prices."""
        for i, ctx in enumerate(self._instruments):
            pos = self._positions[i]
            if pos.in_position and ctx.n_bricks > 0:
                bi = min(pos.brick_cursor, ctx.n_bricks - 1)
                price = float(ctx._brick_closes[bi])
                bt = ctx._brick_times[bi] if bi < len(ctx._brick_times) else None
                weight = float(self._current_weights[i])
                trade = self._close_position(i, price, bt, "end_of_data", weight)
                if trade:
                    self._episode_trades.append(trade)
                    self._episode_returns.append(trade["net_pnl"])

    # ──────────────────────────────────────────────────────────────────────
    # Time helpers
    # ──────────────────────────────────────────────────────────────────────

    def _find_earliest_current_time(self) -> Optional[datetime]:
        """Find the earliest brick time at current cursor positions."""
        earliest: Optional[datetime] = None
        for i, ctx in enumerate(self._instruments):
            bi = self._positions[i].brick_cursor
            if bi < len(ctx._brick_times):
                t = ctx._brick_times[bi]
                if earliest is None or t < earliest:
                    earliest = t
        return earliest

    def _find_min_common_start(self, warmup: int) -> int:
        """Find the minimum warmup brick count across instruments."""
        if not self._instruments:
            return 0
        return warmup

    def _find_max_common_start(self, warmup: int) -> int:
        """
        Find the maximum start offset that leaves enough bricks for
        an episode in all instruments.
        """
        if not self._instruments:
            return 0

        # Minimum bricks needed for a useful episode
        min_bricks_needed = 100

        max_starts = []
        for ctx in self._instruments:
            available = ctx.n_bricks - warmup - min_bricks_needed
            max_starts.append(max(0, available))

        if not max_starts:
            return warmup

        return warmup + min(max_starts)

    # ──────────────────────────────────────────────────────────────────────
    # Info builder
    # ──────────────────────────────────────────────────────────────────────

    def _build_info(self) -> Dict[str, Any]:
        """Build the info dict returned by reset() and step()."""
        n_active = sum(1 for p in self._positions if p.in_position)
        total_trades = len(self._episode_trades)

        return {
            "portfolio_equity": self._portfolio_equity,
            "portfolio_dd": self._portfolio_dd,
            "portfolio_peak": self._portfolio_peak,
            "n_active_positions": n_active,
            "total_trades": total_trades,
            "total_pnl": sum(self._episode_returns),
            "step": self._step_count,
            "weights": self._current_weights.tolist(),
        }

    # ──────────────────────────────────────────────────────────────────────
    # Utility: sample action
    # ──────────────────────────────────────────────────────────────────────

    def sample_action(self) -> np.ndarray:
        """
        Sample a random action from the action space.

        Returns
        -------
        np.ndarray
            Random allocation weights, shape ``(n_instruments,)``.
        """
        if GYM_AVAILABLE and hasattr(self.action_space, "sample"):
            return self.action_space.sample()
        return self._rng.uniform(0.0, 1.0, size=self._n_instruments).astype(np.float32)

    # ──────────────────────────────────────────────────────────────────────
    # Utility: get episode summary
    # ──────────────────────────────────────────────────────────────────────

    def episode_summary(self) -> Dict[str, Any]:
        """
        Compute a summary of the current/completed episode.

        Returns
        -------
        dict
            Keys: total_trades, total_pnl, portfolio_equity, max_dd,
            omega, win_rate, per_instrument (dict of per-instrument stats).
        """
        returns = np.array(self._episode_returns, dtype=np.float64)
        omega = self._compute_portfolio_omega(returns)

        n_wins = int(np.sum(returns > 0)) if len(returns) > 0 else 0
        win_rate = n_wins / len(returns) if len(returns) > 0 else 0.0

        per_inst: Dict[str, Dict[str, Any]] = {}
        for i, ctx in enumerate(self._instruments):
            pos = self._positions[i]
            per_inst[ctx.symbol] = {
                "n_trades": pos.n_trades,
                "cumulative_pnl": pos.cumulative_pnl,
                "peak_pnl": pos.peak_pnl,
                "bricks_processed": pos.brick_cursor,
                "in_position": pos.in_position,
            }

        return {
            "total_trades": len(self._episode_trades),
            "total_pnl": float(sum(self._episode_returns)),
            "portfolio_equity": self._portfolio_equity,
            "max_dd": self._portfolio_dd,
            "omega": omega,
            "win_rate": win_rate,
            "steps": self._step_count,
            "per_instrument": per_inst,
        }

    # ──────────────────────────────────────────────────────────────────────
    # String repr
    # ──────────────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        symbols = [ctx.symbol for ctx in self._instruments]
        return (
            f"RenkoPortfolioEnv(n_instruments={self._n_instruments}, "
            f"obs_dim={self._obs_dim}, step_mode='{self._config.step_mode}', "
            f"symbols={symbols})"
        )
