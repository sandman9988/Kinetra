"""
Unified Renko Pipeline Package
===============================

Collapses the five overlapping qualification/drift/pool pipelines into
a single coherent framework with:

- One canonical instrument identity (:class:`InstrumentKey`)
- One registry schema (:class:`PipelineQualificationResult`) — written by
  both engines, backwards-compatible with Sprint ≤5B ``qualification.json``
- Two engines (:class:`FixedEngine`, :class:`AdaptiveEngine`) sharing a
  common :class:`BrickEngine` interface
- One qualification policy (:class:`QualificationPolicy`)
- One drift model (:func:`check_drift`)
- One pool builder (:func:`build_pool`) with engine-aware tier enforcement

Quick-start
-----------
::

    from kinetra.renko.pipeline import (
        InstrumentKey, FixedEngine, AdaptiveEngine,
        discover_candidates, qualify_instrument_pipeline,
        PipelineRegistry, build_pool,
    )

    # Discover instruments
    candidates = discover_candidates(Path("data/master_standardized"), engine="fixed")

    # Qualify with fixed engine
    registry = PipelineRegistry(Path("data/renko_qualified"))
    registry.load()
    engine = FixedEngine()
    for cand in candidates:
        ctx = FitContext(key=cand.key, m1_df=..., spec=cand.spec)
        result = qualify_instrument_pipeline(ctx, engine)
        registry.save_result(result)

    # Build tiered pool
    pool = build_pool(registry)
    pool.save(Path("results/renko/instrument_pool.json"))

See Also
--------
:mod:`kinetra.renko.pipeline.identity`    — InstrumentKey
:mod:`kinetra.renko.pipeline.specs`       — ContractSpec, load_contract_spec
:mod:`kinetra.renko.pipeline.discovery`   — Candidate, discover_candidates
:mod:`kinetra.renko.pipeline.engines`     — BrickEngine, FixedEngine, AdaptiveEngine
:mod:`kinetra.renko.pipeline.qualify`     — QualificationPolicy, qualify_instrument_pipeline
:mod:`kinetra.renko.pipeline.registry`    — PipelineQualificationResult, PipelineRegistry
:mod:`kinetra.renko.pipeline.drift`       — DriftCheckResult, check_drift
:mod:`kinetra.renko.pipeline.recalibrate` — recalibrate
:mod:`kinetra.renko.pipeline.pool`        — TierPolicy, build_pool
"""

from .discovery import Candidate, discover_candidates
from .drift import DriftCheckResult, check_drift
from .engines import (
    AdaptiveEngine,
    BrickEngine,
    EngineFitResult,
    FitContext,
    FixedEngine,
)
from .identity import InstrumentKey
from .pool import DEFAULT_TIER_POLICY, TierPolicy, build_pool
from .qualify import DEFAULT_POLICY, QualificationPolicy, qualify_instrument_pipeline
from .recalibrate import recalibrate
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
    PipelineRegistry,
    StressMetrics,
)
from .specs import ContractSpec, load_contract_spec

__all__ = [
    # identity
    "InstrumentKey",
    # specs
    "ContractSpec",
    "load_contract_spec",
    # discovery
    "Candidate",
    "discover_candidates",
    # engines
    "BrickEngine",
    "FixedEngine",
    "AdaptiveEngine",
    "FitContext",
    "EngineFitResult",
    # qualify
    "QualificationPolicy",
    "qualify_instrument_pipeline",
    "DEFAULT_POLICY",
    # registry
    "PipelineQualificationResult",
    "PipelineRegistry",
    "DataMeta",
    "FrictionMeta",
    "ParamsMeta",
    "ISMetrics",
    "OOSMetrics",
    "StressMetrics",
    "GateMetrics",
    "MetricsMeta",
    "DriftState",
    # drift
    "DriftCheckResult",
    "check_drift",
    # recalibrate
    "recalibrate",
    # pool
    "TierPolicy",
    "build_pool",
    "DEFAULT_TIER_POLICY",
]
