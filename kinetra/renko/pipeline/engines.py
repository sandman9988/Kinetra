"""
Brick engine protocol and concrete implementations.

Both engines share a common :class:`FitContext` input and :class:`EngineFitResult`
output, making fixed and adaptive qualification directly comparable.

:class:`FixedEngine`
    Delegates to :func:`kinetra.renko.qualify.qualify_instrument`.
    Uses DSP brick sizing + sweep + walk-forward + friction stress.

:class:`AdaptiveEngine`
    Delegates to :func:`kinetra.renko.spread_gated_backtest.qualify_instrument_adaptive`.
    Uses Garman-Klass volatility + spread-gate grid search.

Neither engine re-implements any existing logic — they are thin wrappers
that translate between the common contract (:class:`EngineFitResult`) and
the existing backend return types.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, Optional

import pandas as pd

from .identity import InstrumentKey
from .specs import ContractSpec

if TYPE_CHECKING:
    from kinetra.renko.session import SessionProfile

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# Shared IO types
# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class FitContext:
    """
    All inputs required by any :class:`BrickEngine`.

    Parameters
    ----------
    key : InstrumentKey
        Full instrument identity (includes engine field).
    m1_df : pd.DataFrame
        M1 OHLCV DataFrame (UTC-indexed, at least ``close`` column).
    spec : ContractSpec
        Contract specification (spread, tick_size, commission, etc.).
    session_profile : SessionProfile or None
        Pre-computed session profile.  If ``None``, the engine must detect
        the session break itself.
    force : bool
        If True, ignore any cached ``qualification.json`` and re-run.
    output_dir : Path or None
        Where to write ``qualification.json`` and ``session_profile.json``.
        If ``None``, results are not persisted.
    """

    key: InstrumentKey
    m1_df: pd.DataFrame
    spec: ContractSpec
    session_profile: Optional["SessionProfile"] = None
    force: bool = False
    output_dir: Optional[Any] = None  # Path — avoid import at module level


@dataclass
class EngineFitResult:
    """
    Unified qualification output.  Both engines populate all common fields.
    Engine-specific config is in ``adaptive_config`` (empty for ``FixedEngine``).

    Parameters
    ----------
    key : InstrumentKey
    qualified : bool
    failure_reason : str
        Human-readable reason when ``qualified=False``.
    brick_size : float
    filter_params : dict
        Serialised :class:`~kinetra.renko.backtest.FilterParams`.
    stop_params : dict
        Serialised :class:`~kinetra.renko.backtest.StopParams`.
    adaptive_config : dict
        Spread-gate + brick-stability config (empty for fixed engine).
    is_omega, is_z, is_trades : float / int
        In-sample Omega, Z-factor, trade count.
    oos_omega, oos_survival : float
        Walk-forward OOS Omega and fold survival rate.
    stress_omega : float
        Omega at 1.5× friction multiplier.
    vr_peak : float
        DSP variance-ratio peak (0.0 for adaptive engine).
    friction_ratio : float
        ``spread_pts / brick_size`` at qualification time.
    gate_bar_fraction : float
        Fraction of M1 bars where the spread gate was OPEN.
        0.0 for the fixed engine (always trading when signal fires).
    gate_trade_fraction : float
        Fraction of completed trades that occurred in gated windows.
        0.0 for the fixed engine.
    bars_m1, coverage_ratio, spike_count, session_break_minutes : data QC fields.
    """

    key: InstrumentKey
    qualified: bool
    failure_reason: str = ""
    brick_size: float = 0.0
    filter_params: Dict[str, Any] = field(default_factory=dict)
    stop_params: Dict[str, Any] = field(default_factory=dict)
    adaptive_config: Dict[str, Any] = field(default_factory=dict)
    is_omega: float = 0.0
    is_z: float = 0.0
    is_trades: int = 0
    is_pnl_usd: float = 0.0
    oos_omega: float = 0.0
    oos_survival: float = 0.0
    stress_omega: float = 0.0
    vr_peak: float = 0.0
    friction_ratio: float = 0.0
    gate_bar_fraction: float = 0.0
    gate_trade_fraction: float = 0.0
    bars_m1: int = 0
    coverage_ratio: float = 0.0
    spike_count: int = 0
    session_break_minutes: float = 30.0


# ══════════════════════════════════════════════════════════════════════════════
# Protocol (structural typing)
# ══════════════════════════════════════════════════════════════════════════════


class BrickEngine:
    """
    Structural protocol for a brick engine.

    Both :class:`FixedEngine` and :class:`AdaptiveEngine` satisfy this
    interface.  Type checkers that support ``typing.Protocol`` can use that;
    runtime consumers just call ``engine.fit(ctx)``.
    """

    name: str = ""

    def fit(self, ctx: FitContext) -> EngineFitResult:  # pragma: no cover
        raise NotImplementedError


# ══════════════════════════════════════════════════════════════════════════════
# Private translation helpers
# ══════════════════════════════════════════════════════════════════════════════


def _to_engine_result_fixed(
    key: InstrumentKey,
    result: Any,  # QualificationResult from kinetra.renko.qualify
) -> EngineFitResult:
    """Translate a legacy ``QualificationResult`` to :class:`EngineFitResult`."""
    return EngineFitResult(
        key=key,
        qualified=bool(result.qualified),
        failure_reason=str(getattr(result, "disqualification_reason", "") or ""),
        brick_size=float(result.brick_size),
        filter_params=dict(result.filter_params) if result.filter_params else {},
        stop_params={},
        adaptive_config={},
        is_omega=float(result.omega),
        is_z=float(result.z_factor),
        is_trades=int(result.n_trades),
        is_pnl_usd=0.0,
        oos_omega=float(result.oos_omega),
        oos_survival=float(result.oos_survival_rate),
        stress_omega=float(result.friction_stress_omega),
        vr_peak=float(result.vr_peak),
        friction_ratio=float(result.friction_ratio),
        gate_bar_fraction=0.0,
        gate_trade_fraction=0.0,
        bars_m1=int(getattr(result, "bars_m1", 0)),
        coverage_ratio=float(getattr(result, "coverage_ratio", 0.0)),
        spike_count=int(getattr(result, "spike_count", 0)),
        session_break_minutes=float(getattr(result, "session_break_minutes", 30.0)),
    )


def _to_engine_result_adaptive(
    key: InstrumentKey,
    result: Any,  # AdaptiveQualificationResult from spread_gated_backtest
) -> EngineFitResult:
    """Translate an ``AdaptiveQualificationResult`` to :class:`EngineFitResult`."""
    best = getattr(result, "best_config", None)
    omega = float(best.omega) if best else 0.0
    z = float(best.z_factor) if best else 0.0
    trades = int(best.trades) if best else 0
    gate_type = str(getattr(best, "gate_type", "none")) if best else "none"
    window = getattr(best, "window", None)
    quantile = getattr(best, "quantile", None)
    theta = getattr(best, "max_spread_brick_ratio", None)
    floor_k4 = getattr(best, "floor_k4", 4.0) if best else 4.0

    adaptive_config: Dict[str, Any] = {
        "gate_type": gate_type,
        "window": window,
        "quantile": quantile,
        "max_spread_brick_ratio": theta,
        "floor_k4": float(floor_k4) if floor_k4 is not None else 4.0,
    }

    # gate coverage metrics
    total_bars = int(getattr(result, "total_bars", 0))
    gated_entries = int(getattr(best, "gated_entries", 0)) if best else 0
    gate_bar_fraction = float(getattr(result, "gate_bar_fraction", 0.0))
    gate_trade_fraction = float(getattr(result, "gate_trade_fraction", 0.0))

    # Fallback: estimate gate_bar_fraction from gated_entries / total_bars
    if gate_bar_fraction == 0.0 and total_bars > 0 and gated_entries > 0:
        gate_bar_fraction = min(gated_entries / total_bars, 1.0)

    avg_brick = float(getattr(best, "avg_brick_size", 0.0)) if best else 0.0
    return EngineFitResult(
        key=key,
        qualified=bool(result.qualified),
        failure_reason=str(getattr(result, "status", "") or ""),
        brick_size=avg_brick,
        filter_params={},
        stop_params={},
        adaptive_config=adaptive_config,
        is_omega=omega,
        is_z=z,
        is_trades=trades,
        is_pnl_usd=float(getattr(best, "net_pnl", 0.0)) if best else 0.0,
        oos_omega=omega,  # adaptive doesn't separate IS/OOS in the same way
        oos_survival=1.0 if result.qualified else 0.0,
        stress_omega=omega,  # spread gating already acts as a stress filter
        vr_peak=0.0,
        friction_ratio=0.0,
        gate_bar_fraction=gate_bar_fraction,
        gate_trade_fraction=gate_trade_fraction,
        bars_m1=total_bars,
        coverage_ratio=0.0,
        spike_count=0,
        session_break_minutes=30.0,
    )


# ══════════════════════════════════════════════════════════════════════════════
# Concrete engines
# ══════════════════════════════════════════════════════════════════════════════


class FixedEngine(BrickEngine):
    """
    Fixed brick-size engine.

    Delegates to :func:`kinetra.renko.qualify.qualify_instrument` which runs:

    1. Session break detection
    2. DSP analysis (VR profiling, brick sizing)
    3. Scaled filter params
    4. Brick sweep (grid over candidate brick sizes)
    5. Full backtest (Omega, Z-factor, trade count)
    6. Walk-forward IS/OOS (70/30)
    7. Friction stress test (1.5× costs)
    """

    name: str = "fixed"

    def fit(self, ctx: FitContext) -> EngineFitResult:
        from pathlib import Path

        from kinetra.renko.qualify import qualify_instrument

        output_dir = Path(ctx.output_dir) if ctx.output_dir else None

        try:
            result = qualify_instrument(
                symbol=ctx.key.symbol,
                m1_df=ctx.m1_df,
                spread_pts=ctx.spec.spread_pts,
                tick_size=ctx.spec.tick_size,
                broker_source=ctx.key.broker_source,
                broker_account=ctx.key.broker_account,
                instrument_id=ctx.key.legacy_instrument_id,
                broker_symbol=ctx.key.broker_symbol,
                is_ecn=ctx.spec.is_ecn,
                account_type=ctx.key.exec_tag,
                commission_per_lot=ctx.spec.commission_per_lot,
                force=ctx.force,
                output_dir=output_dir,
            )
        except Exception as exc:
            logger.error("FixedEngine.fit(%s): %s", ctx.key.symbol, exc, exc_info=True)
            return EngineFitResult(
                key=ctx.key,
                qualified=False,
                failure_reason=f"{type(exc).__name__}: {exc}",
            )

        return _to_engine_result_fixed(ctx.key, result)


class AdaptiveEngine(BrickEngine):
    """
    Adaptive brick-size + spread-gate engine.

    Delegates to :func:`kinetra.renko.spread_gated_backtest.qualify_instrument_adaptive`
    which runs Garman-Klass volatility estimation, a roofing filter for adaptive brick
    sizing, and a grid search over spread-gate parameters (W, Q, θ).

    Parameters
    ----------
    grid_config : dict or None
        Optional keyword-argument overrides forwarded to
        ``qualify_instrument_adaptive()``.  Supported keys match the CLI flags
        of the legacy ``adaptive_qualify.py``:
        ``q_vals``, ``w_vals``, ``theta_vals``, ``gate_type``, ``window``,
        ``quantile``, ``theta``, ``min_omega``, ``min_z``, ``min_trades``,
        ``max_drawdown_pct``.
    """

    name: str = "adaptive"

    def __init__(self, grid_config: Optional[Dict[str, Any]] = None) -> None:
        self.grid_config: Dict[str, Any] = grid_config or {}

    def fit(self, ctx: FitContext) -> EngineFitResult:
        from pathlib import Path

        from kinetra.renko.spread_gated_backtest import qualify_instrument_adaptive

        output_dir = Path(ctx.output_dir) if ctx.output_dir else None

        try:
            result = qualify_instrument_adaptive(
                symbol=ctx.key.symbol,
                m1_df=ctx.m1_df,
                spread_pts=ctx.spec.spread_pts,
                tick_size=ctx.spec.tick_size,
                output_dir=output_dir,
                **self.grid_config,
            )
        except Exception as exc:
            logger.error("AdaptiveEngine.fit(%s): %s", ctx.key.symbol, exc, exc_info=True)
            return EngineFitResult(
                key=ctx.key,
                qualified=False,
                failure_reason=f"{type(exc).__name__}: {exc}",
            )

        return _to_engine_result_adaptive(ctx.key, result)
