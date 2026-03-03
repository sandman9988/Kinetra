"""
Pipeline qualification orchestrator.

:class:`QualificationPolicy` is the single source of truth for all
qualification gate thresholds.  Every CLI, test, and downstream consumer
should import thresholds from here rather than duplicating magic numbers.

:func:`qualify_instrument_pipeline` runs one engine on one instrument and
returns a :class:`~.registry.PipelineQualificationResult`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone

# Import module-level constants so QualificationPolicy defaults stay in sync
from kinetra.renko.qualify import (
    QUALIFY_FRICTION_MIN_OMEGA,
    QUALIFY_FRICTION_MULT,
    QUALIFY_MAX_FRICTION_RATIO,
    QUALIFY_MIN_OMEGA,
    QUALIFY_MIN_OOS_OMEGA,
    QUALIFY_MIN_OOS_SURVIVAL,
    QUALIFY_MIN_TRADES,
    QUALIFY_MIN_VR,
    QUALIFY_MIN_Z,
)

from .engines import BrickEngine, EngineFitResult, FitContext
from .registry import (
    DataMeta,
    DriftState,
    FrictionMeta,
    GateMetrics,
    ISMetrics,
    MetricsMeta,
    OOSMetrics,
    ParamsMeta,
    PipelineQualificationResult,
    StressMetrics,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class QualificationPolicy:
    """
    Single source of truth for all qualification gate thresholds.

    Defaults mirror the module-level constants in
    :mod:`kinetra.renko.qualify` so there is one definition, not two.

    Parameters
    ----------
    min_vr : float
        Minimum DSP variance-ratio peak to consider trending.
    max_friction_ratio : float
        Maximum ``spread_pts / brick_size`` allowed.
    min_is_omega : float
        Minimum in-sample Omega ratio.
    min_z : float
        Minimum Z-factor.
    min_trades : int
        Minimum number of trades for a credible backtest.
    min_oos_omega : float
        Minimum walk-forward OOS Omega.
    min_oos_survival : float
        Minimum fraction of walk-forward folds passing the OOS omega gate.
    stress_cost_mult : float
        Friction multiplier for the stress test (default 1.5×).
    min_stress_omega : float
        Minimum Omega at ``stress_cost_mult`` × costs.
    oos_split : float
        Fraction of data used for OOS in walk-forward (not currently
        enforced by QualificationPolicy itself, informational only).
    """

    min_vr: float = QUALIFY_MIN_VR
    max_friction_ratio: float = QUALIFY_MAX_FRICTION_RATIO
    min_is_omega: float = QUALIFY_MIN_OMEGA
    min_z: float = QUALIFY_MIN_Z
    min_trades: int = QUALIFY_MIN_TRADES
    min_oos_omega: float = QUALIFY_MIN_OOS_OMEGA
    min_oos_survival: float = QUALIFY_MIN_OOS_SURVIVAL
    stress_cost_mult: float = QUALIFY_FRICTION_MULT
    min_stress_omega: float = QUALIFY_FRICTION_MIN_OMEGA
    oos_split: float = 0.30

    def describe(self) -> str:
        """Return a human-readable threshold legend (no hard-coded strings)."""
        lines = [
            "Qualification gates:",
            f"  min_vr              >= {self.min_vr}",
            f"  max_friction_ratio  <= {self.max_friction_ratio}",
            f"  min_is_omega        >= {self.min_is_omega}",
            f"  min_z               >= {self.min_z}",
            f"  min_trades          >= {self.min_trades}",
            f"  min_oos_omega       >= {self.min_oos_omega}",
            f"  min_oos_survival    >= {self.min_oos_survival}",
            f"  stress_cost_mult    =  {self.stress_cost_mult}×",
            f"  min_stress_omega    >= {self.min_stress_omega}",
            f"  oos_split           =  {self.oos_split:.0%}",
        ]
        return "\n".join(lines)


DEFAULT_POLICY = QualificationPolicy()


def qualify_instrument_pipeline(
    ctx: FitContext,
    engine: BrickEngine,
    *,
    policy: QualificationPolicy = DEFAULT_POLICY,
) -> PipelineQualificationResult:
    """
    Run one engine on one instrument and return a unified result.

    Parameters
    ----------
    ctx : FitContext
        All inputs (key, m1_df, spec, session_profile, force, output_dir).
    engine : BrickEngine
        The engine to use (``FixedEngine`` or ``AdaptiveEngine``).
    policy : QualificationPolicy
        Gate thresholds (default: ``DEFAULT_POLICY``).

    Returns
    -------
    PipelineQualificationResult
        Full qualification record ready for :class:`~.registry.PipelineRegistry`.
    """
    fit: EngineFitResult = engine.fit(ctx)
    now_iso = datetime.now(timezone.utc).isoformat()

    # ── Build sub-schemas from EngineFitResult ────────────────────────────────
    data = DataMeta(
        bars_m1=fit.bars_m1,
        start="",
        end="",
        coverage_ratio=fit.coverage_ratio,
        spike_count=fit.spike_count,
        session_break_minutes=fit.session_break_minutes,
    )
    friction = FrictionMeta(
        spread_pts=ctx.spec.spread_pts,
        commission_per_lot=ctx.spec.commission_per_lot,
        tick_size=ctx.spec.tick_size,
        friction_ratio=fit.friction_ratio,
        stress_cost_mult=policy.stress_cost_mult,
    )
    params = ParamsMeta(
        brick_size=fit.brick_size,
        filter_params=fit.filter_params,
        stop_params=fit.stop_params,
        adaptive=fit.adaptive_config,
    )
    metrics = MetricsMeta(
        vr_peak=fit.vr_peak,
        is_=ISMetrics(
            omega=fit.is_omega,
            z=fit.is_z,
            trades=fit.is_trades,
            pnl_usd=fit.is_pnl_usd,
        ),
        oos=OOSMetrics(
            omega=fit.oos_omega,
            z=0.0,
            trades=0,
            survival=fit.oos_survival,
        ),
        stress=StressMetrics(
            omega=fit.stress_omega,
            z=0.0,
        ),
        gate=GateMetrics(
            bar_fraction=fit.gate_bar_fraction,
            trade_fraction=fit.gate_trade_fraction,
        ),
    )

    return PipelineQualificationResult(
        instrument_id=ctx.key.instrument_id,
        key=ctx.key,
        qualified=fit.qualified,
        engine=engine.name,
        qualified_at=now_iso,
        data=data,
        friction=friction,
        params=params,
        metrics=metrics,
        drift=DriftState(),
        failure_reason=fit.failure_reason,
    )
