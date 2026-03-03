"""
Pipeline pool builder for the unified Renko framework.

:class:`TierPolicy` consolidates all tier threshold constants from
:mod:`kinetra.renko.instrument_pool` into a single frozen dataclass
with a ``describe()`` method so CLIs can print thresholds without
hard-coding legend strings.

:func:`build_pool` reads the :class:`~.registry.PipelineRegistry` only
(no qualification logic) and delegates tier assignment, scoring, cluster
capping, and deduplication to the existing
:func:`~kinetra.renko.instrument_pool.build_pool_from_results`.
It adds ``min_gate_bar_fraction`` enforcement for the adaptive engine:
an adaptive result cannot reach Tier 1 if its gate was open for fewer
than ``min_gate_bar_fraction`` of bars.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, List, Optional

from kinetra.renko.instrument_pool import (
    TIER1_MAX_FRICTION_RATIO,
    TIER1_MIN_FRICTION_STRESS_OMEGA,
    TIER1_MIN_IS_OMEGA,
    TIER1_MIN_OOS_OMEGA,
    TIER1_MIN_OOS_SURVIVAL,
    TIER1_MIN_TRADES,
    TIER1_MIN_Z,
    InstrumentPool,
    build_pool_from_results,
)
from kinetra.renko.qualify import (
    QUALIFY_MAX_FRICTION_RATIO,
    QUALIFY_MIN_OMEGA,
    QUALIFY_MIN_OOS_OMEGA,
    QUALIFY_MIN_OOS_SURVIVAL,
    QUALIFY_MIN_TRADES,
    QUALIFY_MIN_Z,
)

from .registry import PipelineQualificationResult, PipelineRegistry

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TierPolicy:
    """
    Single source of truth for pool tier thresholds.

    Tier 1 — deploy immediately (all must pass):
        IS Omega, OOS Omega, Z-factor, OOS survival, friction stress, trade
        count, friction ratio.

    Tier 2 — viable (qualification pass + below Tier 1 on ≥ 1 axis).

    Tier 3 — watch list (qualified but recalibration_due=True or below Tier 2).

    Adaptive-specific:
        ``min_gate_bar_fraction`` — adaptive Tier 1 requires gate open ≥ this
        fraction of bars.  Prevents deploying a strategy that only trades in
        rare micro-windows.
    """

    tier1_min_is_omega: float = TIER1_MIN_IS_OMEGA
    tier1_min_oos_omega: float = TIER1_MIN_OOS_OMEGA
    tier1_min_z: float = TIER1_MIN_Z
    tier1_min_oos_survival: float = TIER1_MIN_OOS_SURVIVAL
    tier1_min_stress_omega: float = TIER1_MIN_FRICTION_STRESS_OMEGA
    tier1_min_trades: int = TIER1_MIN_TRADES
    tier1_max_friction_ratio: float = TIER1_MAX_FRICTION_RATIO

    # Qualification gates (used for Tier 2 / Tier 3 boundaries)
    qual_min_omega: float = QUALIFY_MIN_OMEGA
    qual_min_oos_omega: float = QUALIFY_MIN_OOS_OMEGA
    qual_min_z: float = QUALIFY_MIN_Z
    qual_min_oos_survival: float = QUALIFY_MIN_OOS_SURVIVAL
    qual_min_trades: int = QUALIFY_MIN_TRADES
    qual_max_friction_ratio: float = QUALIFY_MAX_FRICTION_RATIO

    # Adaptive engine: minimum gate coverage to reach Tier 1
    min_gate_bar_fraction: float = 0.35

    # Portfolio slot limits
    max_cluster_instruments: int = 3
    max_cluster_weight: float = 0.35

    def describe(self) -> str:
        """Return a human-readable threshold legend for CLI --thresholds output."""
        lines = [
            "Pool tier thresholds:",
            "",
            "  TIER 1 (all must pass):",
            f"    min IS omega        >= {self.tier1_min_is_omega}",
            f"    min OOS omega       >= {self.tier1_min_oos_omega}",
            f"    min Z-factor        >= {self.tier1_min_z}",
            f"    min OOS survival    >= {self.tier1_min_oos_survival:.0%}",
            f"    min stress omega    >= {self.tier1_min_stress_omega}",
            f"    min trades          >= {self.tier1_min_trades}",
            f"    max friction ratio  <= {self.tier1_max_friction_ratio}",
            f"    min gate bar frac   >= {self.min_gate_bar_fraction:.0%}  (adaptive only)",
            "",
            "  TIER 2 (qualification pass + below Tier 1 on ≥ 1 axis):",
            f"    min IS omega        >= {self.qual_min_omega}",
            f"    min OOS omega       >= {self.qual_min_oos_omega}",
            f"    min Z-factor        >= {self.qual_min_z}",
            "",
            "  Portfolio slot limits:",
            f"    max per cluster     {self.max_cluster_instruments}",
            f"    max cluster weight  {self.max_cluster_weight:.0%}",
        ]
        return "\n".join(lines)


DEFAULT_TIER_POLICY = TierPolicy()


def _to_legacy_result(result: PipelineQualificationResult) -> Any:
    """
    Convert a :class:`~.registry.PipelineQualificationResult` to the legacy
    ``QualificationResult`` format expected by ``build_pool_from_results()``.
    """
    from kinetra.renko.qualify import QualificationResult

    return QualificationResult(
        symbol=result.key.symbol,
        qualified=result.qualified,
        disqualified=not result.qualified,
        disqualification_reason=result.failure_reason or None,
        broker_source=result.key.broker_source,
        broker_account=result.key.broker_account,
        instrument_id=result.instrument_id,
        broker_symbol=result.key.broker_symbol,
        spread_pts=result.friction.spread_pts,
        tick_size=result.friction.tick_size,
        commission_per_lot=result.friction.commission_per_lot,
        cluster="unknown",  # will be re-derived by build_pool_from_results
        brick_size=result.params.brick_size,
        vr_peak=result.metrics.vr_peak,
        friction_ratio=result.friction.friction_ratio,
        omega=result.metrics.is_.omega,
        z_factor=result.metrics.is_.z,
        n_trades=result.metrics.is_.trades,
        oos_omega=result.metrics.oos.omega,
        oos_survival_rate=result.metrics.oos.survival,
        friction_stress_omega=result.metrics.stress.omega,
        filter_params=result.params.filter_params,
        data_start=result.data.start,
        data_end=result.data.end,
        qualified_at=result.qualified_at,
        recalibration_due=result.drift.recalibration_due,
        drift_reason=result.drift.drift_reason or None,
        pipeline_version=result.pipeline_version,
        # New fields (safe defaults if not present)
        engine=result.engine,
        gate_bar_fraction=result.metrics.gate.bar_fraction,
        gate_trade_fraction=result.metrics.gate.trade_fraction,
    )


def build_pool(
    registry: PipelineRegistry,
    policy: TierPolicy = DEFAULT_TIER_POLICY,
    *,
    engines: Optional[List[str]] = None,
) -> InstrumentPool:
    """
    Build an :class:`~kinetra.renko.instrument_pool.InstrumentPool` from the registry.

    Parameters
    ----------
    registry : PipelineRegistry
        Loaded registry (both engines present if ``--engine both`` was used).
    policy : TierPolicy
        Tier and slot limit policy.
    engines : list[str] or None
        Filter to specific engines (e.g. ``["fixed"]``).  None = all engines.

    Returns
    -------
    InstrumentPool
        Tiered, cluster-capped, deduplicated pool.

    Notes
    -----
    Adaptive Tier 1 enforcement
        If ``result.engine == "adaptive"`` and
        ``result.metrics.gate.bar_fraction < policy.min_gate_bar_fraction``,
        the result is downgraded to Tier 2 by setting ``friction_stress_omega = 0``
        in the translated legacy record before passing to ``build_pool_from_results``.
    """
    all_results = registry.get_qualified(engine=None)
    if engines is not None:
        all_results = [r for r in all_results if r.engine in engines]

    if not all_results:
        logger.warning("build_pool: no qualified results found in registry")

    legacy_results = []
    for r in all_results:
        legacy = _to_legacy_result(r)
        # Enforce min_gate_bar_fraction for adaptive Tier 1
        if (
            r.engine == "adaptive"
            and r.metrics.gate.bar_fraction < policy.min_gate_bar_fraction
        ):
            # Zero out stress_omega so Tier 1 gate fails; instrument falls to Tier 2
            object.__setattr__(legacy, "friction_stress_omega", 0.0) if hasattr(legacy, "__slots__") else setattr(legacy, "friction_stress_omega", 0.0)
            logger.debug(
                "build_pool: %s adaptive gate_bar_fraction=%.2f < %.2f → capped to Tier 2",
                r.instrument_id,
                r.metrics.gate.bar_fraction,
                policy.min_gate_bar_fraction,
            )
        legacy_results.append(legacy)

    pool = build_pool_from_results(legacy_results)
    return pool
