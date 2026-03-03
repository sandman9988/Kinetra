"""
Kinetra - Renko-Only Adaptive Trading System

A self-validating, reinforcement learning-based algorithmic trading system
built on Renko price-space representation.
"""

__version__ = "3.0.0"
__author__ = "Kinetra Team"

import importlib
import importlib.util
from typing import TYPE_CHECKING

# Type checking imports (only used by type checkers, not at runtime)
if TYPE_CHECKING:
    from .friction_cost import (
        BuyHoldCalculator as BuyHoldCalculator,
    )
    from .friction_cost import (
        BuyHoldResult as BuyHoldResult,
    )
    from .friction_cost import (
        FrictionBreakdown as FrictionBreakdown,
    )
    from .friction_cost import (
        FrictionCalculator as FrictionCalculator,
    )
    from .friction_cost import (
        InstrumentSpec as InstrumentSpec,
    )
    from .friction_cost import (
        RoundTripResult as RoundTripResult,
    )
    from .friction_cost import (
        clear_cache as clear_cache,
    )
    from .friction_cost import (
        env_friction_pct as env_friction_pct,
    )
    from .friction_cost import (
        get_buy_hold_baseline as get_buy_hold_baseline,
    )
    from .friction_cost import (
        get_calculator as get_calculator,
    )
    from .friction_cost import (
        list_available_specs as list_available_specs,
    )
    from .friction_cost import (
        load_spec as load_spec,
    )


# Module mapping for lazy imports — Renko-only architecture
_LAZY_MODULES = {
    # ── Project root path ────────────────────────────────────────────────
    "PROJECT_ROOT": "config",
    "resolve_project_path": "config",
    # ── Friction Cost Calculator (canonical) ──────────────────────────────
    "InstrumentSpec": "friction_cost",
    "FrictionBreakdown": "friction_cost",
    "FrictionCalculator": "friction_cost",
    "RoundTripResult": "friction_cost",
    "BuyHoldResult": "friction_cost",
    "BuyHoldCalculator": "friction_cost",
    "load_spec": "friction_cost",
    "get_calculator": "friction_cost",
    "env_friction_pct": "friction_cost",
    "get_buy_hold_baseline": "friction_cost",
    "list_available_specs": "friction_cost",
    "clear_cache": "friction_cost",
    "get_calculator_with_data": "friction_cost",
    # ── Backtesting metric helpers ───────────────────────────────────────
    "omega_ratio": "backtesting.metrics",
    "calculate_z_factor": "backtesting.metrics",
    # ── Aggregation (M1 → higher timeframes) ──────────────────────────────
    "aggregate_ohlcv": "aggregation",
    # ══════════════════════════════════════════════════════════════════════
    # RENKO MODULES — canonical Renko pipeline
    # ══════════════════════════════════════════════════════════════════════
    # ── Renko Brick Engine ────────────────────────────────────────────────
    "build_renko": "renko.brick_engine",
    "brick_summary": "renko.brick_engine",
    "bricks_per_day": "renko.brick_engine",
    "BrickSummary": "renko.brick_engine",
    # ── Renko Filters ─────────────────────────────────────────────────────
    "flip_rate": "renko.filters",
    "markov_stickiness": "renko.filters",
    "evaluate_entry": "renko.filters",
    "evaluate_entries_vectorized": "renko.filters",
    # ── Renko DSP ─────────────────────────────────────────────────────────
    "vr_profile": "renko.dsp",
    "brick_from_scale": "renko.dsp",
    "classify_regime": "renko.dsp",
    "compute_friction_floor": "renko.dsp",
    "compute_rolling_friction_floor": "renko.dsp",
    "run_dsp": "renko.dsp",
    "scaled_filter_params": "renko.dsp",
    "DSPResult": "renko.dsp",
    "FrictionFloor": "renko.dsp",
    "SpreadProfile": "renko.dsp",
    # ── Renko Backtest ────────────────────────────────────────────────────
    "backtest_instrument": "renko.backtest",
    "backtest_portfolio": "renko.backtest",
    "walk_forward_instrument": "renko.backtest",
    "monte_carlo_instrument": "renko.backtest",
    "sweep_brick_sizes": "renko.backtest",
    "stress_test_friction": "renko.backtest",
    "FilterParams": "renko.backtest",
    "StopParams": "renko.backtest",
    "RiskParams": "renko.backtest",
    "SizingMode": "renko.backtest",
    "VolSizingParams": "renko.backtest",
    "RenkoTrade": "renko.backtest",
    "RenkoEmpiricalCore": "renko.backtest",
    "xauusd_empirical_core": "renko.backtest",
    "InstrumentBacktestResult": "renko.backtest",
    "PortfolioBacktestResult": "renko.backtest",
    "WalkForwardResult": "renko.backtest",
    "MonteCarloResult": "renko.backtest",
    "FrictionStressResult": "renko.backtest",
    "SweepPoint": "renko.backtest",
    "BrickSweepResult": "renko.backtest",
    # ── Renko Portfolio ───────────────────────────────────────────────────
    "get_cluster": "renko.portfolio",
    "get_cluster_members": "renko.portfolio",
    "cluster_summary": "renko.portfolio",
    "CLUSTER_MAP": "renko.portfolio",
    "ALL_CLUSTERS": "renko.portfolio",
    "equal_risk_weights": "renko.portfolio",
    "apply_cluster_caps": "renko.portfolio",
    "deduplicate_underlyings": "renko.portfolio",
    "build_portfolio_equity": "renko.portfolio",
    "build_portfolio": "renko.portfolio",
    "herfindahl_index": "renko.portfolio",
    "max_drawdown": "renko.portfolio",
    "max_drawdown_duration_days": "renko.portfolio",
    "calmar_ratio": "renko.portfolio",
    "cvar": "renko.portfolio",
    "tail_risk_analysis": "renko.portfolio",
    "estimate_usd_per_point": "renko.portfolio",
    "find_worst_period": "renko.portfolio",
    "stress_correlation_one": "renko.portfolio",
    "PortfolioConfig": "renko.portfolio",
    "PortfolioConstruction": "renko.portfolio",
    "SizingInfo": "renko.portfolio",
    "ClusterStats": "renko.portfolio",
    "TailRiskReport": "renko.portfolio",
    "WorstPeriodResult": "renko.portfolio",
    "CorrelationStressResult": "renko.portfolio",
    # ── Renko Session ─────────────────────────────────────────────────────
    "detect_session_break": "renko.session",
    "session_break_minutes_for": "renko.session",
    "SessionProfile": "renko.session",
    "GapRecord": "renko.session",
    "QCMetrics": "renko.session",
    "clamp_spikes": "renko.session",
    "DEFAULT_SESSION_BREAK_MINUTES": "renko.session",
    # ── Renko Qualify ─────────────────────────────────────────────────────
    "qualify_instrument": "renko.qualify",
    "recalibrate_instrument": "renko.qualify",
    "QualificationResult": "renko.qualify",
    "QualificationRegistry": "renko.qualify",
    "CalibrationDriftDetector": "renko.qualify",
    "DriftCheckResult": "renko.qualify",
    "RecalibrationResult": "renko.qualify",
    "QUALIFY_MIN_OMEGA": "renko.qualify",
    "QUALIFY_MIN_Z": "renko.qualify",
    "QUALIFY_MIN_OOS_OMEGA": "renko.qualify",
    "QUALIFY_MIN_OOS_SURVIVAL": "renko.qualify",
    "QUALIFY_FRICTION_MULT": "renko.qualify",
    "QUALIFY_FRICTION_MIN_OMEGA": "renko.qualify",
    "QUALIFY_MIN_TRADES": "renko.qualify",
    "QUALIFY_MIN_VR": "renko.qualify",
    "QUALIFY_MAX_FRICTION_RATIO": "renko.qualify",
    "DRIFT_VR_CHANGE_THRESHOLD": "renko.qualify",
    "DRIFT_OMEGA_CHANGE_THRESHOLD": "renko.qualify",
    "DRIFT_MIN_NEW_BARS": "renko.qualify",
    # ── Renko Orchestrator ────────────────────────────────────────────────
    "run_full_pipeline": "renko.orchestrator",
    "run_qualification_only": "renko.orchestrator",
    "load_pipeline_result": "renko.orchestrator",
    "PortfolioPipelineResult": "renko.orchestrator",
    "InstrumentPipelineResult": "renko.orchestrator",
    # ── Renko VPIN ────────────────────────────────────────────────────────
    "compute_vpin": "renko.vpin",
    "vpin_timeseries": "renko.vpin",
    "vpin_baseline": "renko.vpin",
    "compute_vpin_multi": "renko.vpin",
    "normalise_vpin": "renko.vpin",
    "normalise_vpin_zscore": "renko.vpin",
    "vpin_excess_kurtosis": "renko.vpin",
    "classify_vpin_regime": "renko.vpin",
    "is_vpin_extreme": "renko.vpin",
    "auto_bucket_size": "renko.vpin",
    "VPINBaseline": "renko.vpin",
    "VPINBucket": "renko.vpin",
    "VPINTimeSeries": "renko.vpin",
    # ── Renko Live Trader ─────────────────────────────────────────────────
    "RenkoLiveTrader": "renko.live_trader",
    "LiveTraderConfig": "renko.live_trader",
    "LiveTrade": "renko.live_trader",
    "InstrumentLiveState": "renko.live_trader",
    "TradeDirection": "renko.live_trader",
    "OrderResult": "renko.live_trader",
    "OrderDispatcher": "renko.live_trader",
    "PaperDispatcher": "renko.live_trader",
    "BarProvider": "renko.live_trader",
    "HistoricalBarProvider": "renko.live_trader",
    "RenkoSizer": "renko.live_trader",
    "VolTargetSizerLive": "renko.live_trader",
    "PERGate": "renko.live_trader",
    "AllocationAgent": "renko.live_trader",
    "RiskAgent": "renko.live_trader",
    "UniformAllocationAgent": "renko.live_trader",
    "FullExposureRiskAgent": "renko.live_trader",
    "evaluate_per_gate": "renko.live_trader",
    "load_agent_from_file": "renko.live_trader",
    # ── Renko Vol Sizer ───────────────────────────────────────────────────
    "VolTargetSizer": "renko.vol_sizer",
    "VolSizingConfig": "renko.vol_sizer",
    "vol_sizing_report": "renko.vol_sizer",
    "compute_vol_targeted_lots_batch": "renko.vol_sizer",
    "PortfolioParetoResult": "renko.vol_sizer",
    "PortfolioParetoPoint": "renko.vol_sizer",
    # ── Renko Policy ──────────────────────────────────────────────────────
    "XAUUSDRenkoProductionPolicy": "renko.policy",
    "xauusd_production_policy": "renko.policy",
    "xauusd_signal_lock": "renko.policy",
    "DrawdownThrottlePolicy": "renko.policy",
    "RegimeSizingPolicy": "renko.policy",
    "BrickScalingPolicy": "renko.policy",
    # ── Renko Instrument Pool ─────────────────────────────────────────────
    "InstrumentPool": "renko.instrument_pool",
    "PoolEntry": "renko.instrument_pool",
    "ClusterPoolStats": "renko.instrument_pool",
    "build_instrument_pool": "renko.instrument_pool",
    "build_pool_from_results": "renko.instrument_pool",
    "TIER1_MIN_IS_OMEGA": "renko.instrument_pool",
    "TIER1_MIN_OOS_OMEGA": "renko.instrument_pool",
    "TIER1_MIN_Z": "renko.instrument_pool",
    "TIER1_MIN_OOS_SURVIVAL": "renko.instrument_pool",
    "TIER1_MIN_FRICTION_STRESS_OMEGA": "renko.instrument_pool",
    "TIER1_MIN_TRADES": "renko.instrument_pool",
    "TIER1_MAX_FRICTION_RATIO": "renko.instrument_pool",
    # ── Renko Spread Gated Backtest ───────────────────────────────────────
    "qualify_instrument_adaptive": "renko.spread_gated_backtest",
    "AdaptiveBacktestConfig": "renko.spread_gated_backtest",
    "AdaptiveBacktestResult": "renko.spread_gated_backtest",
    "SpreadGateConfig": "renko.spread_gated_backtest",
    "GateType": "renko.spread_gated_backtest",
    "build_variable_renko": "renko.spread_gated_backtest",
    "garman_klass_sigma": "renko.spread_gated_backtest",
    "roofing_filter": "renko.spread_gated_backtest",
    # ── Renko Pipeline ────────────────────────────────────────────────────
    "qualify_instrument_pipeline": "renko.pipeline",
    "discover_candidates": "renko.pipeline",
    "PipelineRegistry": "renko.pipeline",
    "PipelineQualificationResult": "renko.pipeline",
    "QualificationPolicy": "renko.pipeline",
    "TierPolicy": "renko.pipeline",
    "BrickEngine": "renko.pipeline",
    "FixedEngine": "renko.pipeline",
    "AdaptiveEngine": "renko.pipeline",
    "EngineFitResult": "renko.pipeline",
    "ContractSpec": "renko.pipeline",
    "InstrumentKey": "renko.pipeline",
    "Candidate": "renko.pipeline",
    "check_drift": "renko.pipeline",
    "recalibrate": "renko.pipeline",
    "build_pool": "renko.pipeline",
    "load_contract_spec": "renko.pipeline",
    "DEFAULT_POLICY": "renko.pipeline",
    "DEFAULT_TIER_POLICY": "renko.pipeline",
    # ── Renko Drift ───────────────────────────────────────────────────────
    "rolling_oos_instrument": "renko.drift",
    # ── Circuit Breakers ──────────────────────────────────────────────────
    "CircuitBreakerManager": "monitoring.circuit_breakers",
    "PortfolioSnapshot": "monitoring.circuit_breakers",
    # ── cTrader Connector ─────────────────────────────────────────────────
    "CTraderConnector": "connectors.ctrader_connector",
    "CTraderCredentials": "connectors.ctrader_connector",
    "build_connector": "connectors.ctrader_connector",
    # ── DNS Hardening ─────────────────────────────────────────────────────
    "DNSHardeningPolicy": "dns_hardening",
    "resolve_and_validate_host": "dns_hardening",
}

# Cache for loaded modules
_loaded_modules = {}


def __getattr__(name: str):
    """
    Lazy import handler - only imports modules when accessed.
    """
    if name in _LAZY_MODULES:
        module_name = _LAZY_MODULES[name]

        # Check cache first
        if module_name not in _loaded_modules:
            try:
                _loaded_modules[module_name] = importlib.import_module(
                    f".{module_name}", package="kinetra"
                )
            except ImportError as e:
                # Handle optional dependencies gracefully
                if "torch" in str(e).lower() or "pytorch" in str(e).lower():
                    return None
                raise

        return getattr(_loaded_modules[module_name], name)

    raise AttributeError(f"module 'kinetra' has no attribute '{name}'")


def __dir__():
    """Return available attributes for IDE autocomplete support."""
    return list(_LAZY_MODULES.keys()) + ["__version__", "__author__"]
