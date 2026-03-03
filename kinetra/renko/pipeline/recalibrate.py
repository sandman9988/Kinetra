"""
Engine-aware recalibration for the unified Renko pipeline.

:func:`recalibrate` recomputes session break + DSP + engine params on
recent data for a qualified instrument, appends a recalibration_log.json
entry, and updates the registry record.

Identity fields (symbol, broker_source, broker_account, exec_tag, engine)
are never modified — only the derived params (brick_size, filter_params,
stop_params, adaptive_config) and the drift state are updated.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from .engines import BrickEngine, FitContext
from .qualify import DEFAULT_POLICY, QualificationPolicy, qualify_instrument_pipeline
from .registry import DriftState, PipelineQualificationResult, PipelineRegistry

logger = logging.getLogger(__name__)


def recalibrate(
    result: PipelineQualificationResult,
    m1_df: pd.DataFrame,
    engine: BrickEngine,
    registry: PipelineRegistry,
    *,
    output_dir: Optional[Path] = None,
    policy: QualificationPolicy = DEFAULT_POLICY,
) -> PipelineQualificationResult:
    """
    Recalibrate one instrument in-place.

    The pipeline runs the full engine fit on *m1_df* (which should be the
    most recent data available), then merges the updated params back into
    the existing record.  Identity fields are preserved.

    Parameters
    ----------
    result : PipelineQualificationResult
        The existing registry record to update.
    m1_df : pd.DataFrame
        Most recent M1 data (UTC-indexed, at least ``close`` column).
    engine : BrickEngine
        Engine to use (should match ``result.engine``).
    registry : PipelineRegistry
        Registry to persist the updated record into.
    output_dir : Path or None
        Where to write ``qualification.json`` (defaults to registry root /
        instrument_id).
    policy : QualificationPolicy
        Qualification gates.

    Returns
    -------
    PipelineQualificationResult
        The updated record (also persisted to the registry).
    """
    from .specs import ContractSpec

    iid = result.instrument_id
    now_iso = datetime.now(timezone.utc).isoformat()

    if output_dir is None:
        output_dir = registry._root / iid

    # Reconstruct a minimal ContractSpec from the existing friction metadata
    spec = ContractSpec(
        symbol=result.key.symbol,
        broker_symbol=result.key.broker_symbol,
        broker_source=result.key.broker_source,
        spread_pts=result.friction.spread_pts,
        tick_size=result.friction.tick_size,
        commission_per_lot=result.friction.commission_per_lot,
        contract_size=0.0,
        tick_value_usd=0.0,
        pip_value_usd=0.0,
        volume_min=0.0,
        volume_step=0.0,
        volume_max=0.0,
        swap_long_points=0.0,
        swap_short_points=0.0,
        is_ecn=None,
        account_type=result.key.exec_tag,
        usd_per_price_unit=0.0,
    )

    ctx = FitContext(
        key=result.key,
        m1_df=m1_df,
        spec=spec,
        force=True,
        output_dir=output_dir,
    )

    new_result = qualify_instrument_pipeline(ctx, engine, policy=policy)

    # ── Merge: keep original identity + qualified_at; update params/metrics ──
    updated = PipelineQualificationResult(
        instrument_id=result.instrument_id,
        key=result.key,           # identity preserved
        qualified=new_result.qualified,
        engine=result.engine,     # engine preserved
        qualified_at=result.qualified_at,  # original qualification timestamp
        data=new_result.data,
        friction=new_result.friction,
        params=new_result.params,
        metrics=new_result.metrics,
        drift=DriftState(
            recalibration_due=False,   # cleared after successful recalibration
            drift_reason="",
            last_checked_at=now_iso,
        ),
        failure_reason=new_result.failure_reason,
        pipeline_version=new_result.pipeline_version,
    )

    # ── Append recalibration log entry ────────────────────────────────────────
    log_entry: Dict[str, Any] = {
        "recalibrated_at": now_iso,
        "symbol": result.key.symbol,
        "engine": result.engine,
        "success": updated.qualified,
        "old_brick_size": result.params.brick_size,
        "new_brick_size": updated.params.brick_size,
        "old_is_omega": result.metrics.is_.omega,
        "new_is_omega": updated.metrics.is_.omega,
        "old_vr_peak": result.metrics.vr_peak,
        "new_vr_peak": updated.metrics.vr_peak,
        "reason": result.drift.drift_reason or "manual",
    }
    try:
        registry.append_recalibration_log(iid, log_entry)
    except Exception as exc:
        logger.warning("Could not append recalibration log for %s: %s", iid, exc)

    # ── Persist updated record ────────────────────────────────────────────────
    try:
        registry.save_result(updated)
        logger.info(
            "recalibrate(%s): success=%s omega=%s→%s brick=%s→%s",
            iid,
            updated.qualified,
            result.metrics.is_.omega,
            updated.metrics.is_.omega,
            result.params.brick_size,
            updated.params.brick_size,
        )
    except Exception as exc:
        logger.error("recalibrate(%s): failed to persist: %s", iid, exc)

    return updated
