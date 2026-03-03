"""
Kinetra Renko Package
=====================

Renko-native trading infrastructure: brick construction, signal filters,
DSP instrument analysis, backtesting, and portfolio management.

Modules
-------
brick_engine
    Close-only Renko brick construction from price series.
filters
    Signal filters for Renko sequences: FlipRate, Markov stickiness.
dsp
    Variance Ratio profiling, brick sizing, friction floor, regime classification.
backtest
    Two-mode Renko backtester (instrument + portfolio), walk-forward,
    Monte Carlo validation, brick sweep, and stress testing.
portfolio
    Cluster management, equal-risk sizing, portfolio equity curve,
    diversification controls, tail-risk analysis.
session
    Broker fingerprinting and session-break detection (Sprint 5B).
qualify
    Instrument qualification pipeline, registry, and drift detection (Sprint 5B).
orchestrator
    Full portfolio pipeline orchestrator: qualify → portfolio → backtest → MC (Sprint 5C).
live_trader
    Renko-native paper/live trader with incremental brick construction, PER capital gates,
    Layer 1/2/3 runtime integration, and broker-neutral OrderDispatcher (Sprint 6).
ctrader_dispatcher
    cTrader concrete implementations of OrderDispatcher and BarProvider (Sprint 6).
    Uses native ProtoOASubscribeLiveTrendbarReq for real-time M1 bars and
    ProtoOANewOrderReq for execution.  Guards the SDK import via
    kinetra.connectors.ctrader_connector — all other Renko modules remain broker-blind.
    cTrader is the primary live-trading feed (§29.4): tighter spreads, cleaner UTC
    alignment, persistent TCP socket (no cloud relay latency).
vol_sizer
    Volatility-targeted position sizing (Sprint 6 — Next Experiment).
    VolTargetSizer: lots = (target_vol_pct × equity) / (brick × usd_per_point × vol̂ × 100k)
    compute_vol_targeted_lots_batch: vectorised causal rolling-vol batch sizer for backtests.
    Replaces fixed compounding (0.01 lot / $1,000) with interpretable, cross-instrument-
    comparable equity curves.
spread_gated_backtest
    Adaptive Renko backtesting with spread-based gating (2026-03-04).
    Uses Garman-Klass volatility + roofing filter for adaptive brick sizing,
    spread gating (percentile/ratio/both modes) to filter high-spread periods,
    and grid search optimization over gate parameters (W, Q, θ).
    Replaces strict fixed-brick qualification for instruments that fail
    traditional gates but have massive profit potential.
"""

from kinetra.renko.backtest import (
    BrickSweepResult,
    FilterParams,
    FrictionStressResult,
    InstrumentBacktestResult,
    MonteCarloResult,
    PortfolioBacktestResult,
    RenkoEmpiricalCore,
    RenkoTrade,
    RiskParams,
    SizingMode,
    StopParams,
    SweepPoint,
    VolSizingParams,
    WalkForwardResult,
    backtest_instrument,
    backtest_portfolio,
    monte_carlo_instrument,
    stress_test_friction,
    sweep_brick_sizes,
    walk_forward_instrument,
    xauusd_empirical_core,
)
from kinetra.renko.brick_engine import BrickSummary, brick_summary, bricks_per_day, build_renko
from kinetra.renko.dsp import (
    DSPResult,
    FrictionFloor,
    SpreadProfile,
    brick_from_scale,
    build_rolling_spread_profile,
    classify_regime,
    compute_friction_floor,
    compute_full_friction_floor,
    compute_rolling_friction_floor,
    run_dsp,
    scaled_filter_params,
    vr_profile,
)
from kinetra.renko.filters import (
    evaluate_entries_vectorized,
    evaluate_entry,
    flip_rate,
    markov_stickiness,
)
from kinetra.renko.live_trader import (
    AllocationAgent,
    BarProvider,
    FullExposureRiskAgent,
    HistoricalBarProvider,
    InstrumentLiveState,
    LiveTrade,
    LiveTraderConfig,
    OrderDispatcher,
    OrderResult,
    PaperDispatcher,
    PERGate,
    RenkoLiveTrader,
    RenkoSizer,
    RiskAgent,
    TradeDirection,
    UniformAllocationAgent,
    VolTargetSizerLive,
    evaluate_per_gate,
    load_agent_from_file,
)
from kinetra.renko.policy import (
    BrickScalingPolicy,
    DrawdownThrottlePolicy,
    RegimeSizingPolicy,
    XAUUSDRenkoProductionPolicy,
    xauusd_production_policy,
    xauusd_signal_lock,
)
from kinetra.renko.vol_sizer import (
    PortfolioParetoPoint,
    PortfolioParetoResult,
    SizingComparisonResult,
    VolBudgetCalibrationResult,
    VolSizingConfig,
    VolTargetSizer,
    calibrate_vol_budget,
    compare_portfolio_sizing_pareto,
    compare_sizing_at_risk_class,
    compute_vol_targeted_lots_batch,
    format_pareto_report,
    vol_sizing_report,
)

try:
    from kinetra.renko.ctrader_dispatcher import (
        CTraderBarProvider,
        CTraderOrderDispatcher,
        build_ctrader_session,
    )

    _CTRADER_DISPATCHER_AVAILABLE = True
except ImportError:
    _CTRADER_DISPATCHER_AVAILABLE = False


from kinetra.renko.instrument_pool import (
    TIER1_MAX_FRICTION_RATIO,
    TIER1_MIN_FRICTION_STRESS_OMEGA,
    TIER1_MIN_IS_OMEGA,
    TIER1_MIN_OOS_OMEGA,
    TIER1_MIN_OOS_SURVIVAL,
    TIER1_MIN_TRADES,
    TIER1_MIN_Z,
    ClusterPoolStats,
    InstrumentPool,
    PoolEntry,
    build_instrument_pool,
    build_pool_from_results,
)
from kinetra.renko.orchestrator import (
    InstrumentPipelineResult,
    PortfolioPipelineResult,
    load_pipeline_result,
    run_full_pipeline,
    run_qualification_only,
)
from kinetra.renko.pipeline import (  # noqa: E402  # Sprint 6A
    DEFAULT_POLICY as PIPELINE_DEFAULT_POLICY,
)
from kinetra.renko.pipeline import (
    DEFAULT_TIER_POLICY,
    AdaptiveEngine,
    BrickEngine,
    Candidate,
    ContractSpec,
    EngineFitResult,
    FitContext,
    FixedEngine,
    InstrumentKey,
    PipelineQualificationResult,
    PipelineRegistry,
    QualificationPolicy,
    TierPolicy,
    check_drift,
    discover_candidates,
    load_contract_spec,
    qualify_instrument_pipeline,
)
from kinetra.renko.pipeline import (
    DriftCheckResult as PipelineDriftCheckResult,
)
from kinetra.renko.pipeline import (
    build_pool as build_pipeline_pool,
)
from kinetra.renko.pipeline import (
    recalibrate as recalibrate_pipeline,
)
from kinetra.renko.portfolio import (
    ALL_CLUSTERS,
    CLUSTER_MAP,
    ClusterStats,
    CorrelationStressResult,
    PortfolioConfig,
    PortfolioConstruction,
    SizingInfo,
    TailRiskReport,
    WorstPeriodResult,
    apply_cluster_caps,
    build_portfolio,
    build_portfolio_equity,
    calmar_ratio,
    cluster_summary,
    cvar,
    deduplicate_underlyings,
    equal_risk_weights,
    estimate_usd_per_point,
    find_worst_period,
    get_cluster,
    get_cluster_members,
    herfindahl_index,
    max_drawdown,
    max_drawdown_duration_days,
    stress_correlation_one,
    tail_risk_analysis,
)
from kinetra.renko.qualify import (
    DRIFT_MIN_NEW_BARS,
    DRIFT_OMEGA_CHANGE_THRESHOLD,
    DRIFT_VR_CHANGE_THRESHOLD,
    QUALIFY_FRICTION_MIN_OMEGA,
    QUALIFY_FRICTION_MULT,
    QUALIFY_MAX_FRICTION_RATIO,
    QUALIFY_MIN_OMEGA,
    QUALIFY_MIN_OOS_OMEGA,
    QUALIFY_MIN_OOS_SURVIVAL,
    QUALIFY_MIN_TRADES,
    QUALIFY_MIN_VR,
    QUALIFY_MIN_Z,
    CalibrationDriftDetector,
    DriftCheckResult,
    QualificationRegistry,
    QualificationResult,
    RecalibrationResult,
    qualify_instrument,
    recalibrate_instrument,
)
from kinetra.renko.session import (
    DEFAULT_SESSION_BREAK_MINUTES,
    GapRecord,
    QCMetrics,
    SessionProfile,
    clamp_spikes,
    detect_session_break,
    session_break_minutes_for,
)
from kinetra.renko.spread_gated_backtest import (
    AdaptiveBacktestConfig,
    AdaptiveBacktestResult,
    AdaptiveBacktestStatus,
    AdaptiveQualificationResult,
    BrickStabilityConfig,
    GateType,
    SpreadGateConfig,
    backtest_strict,
    build_variable_renko,
    calculate_omega,
    find_pareto_frontier,
    garman_klass_sigma,
    lots_from_capital,
    print_qualification_summary,
    qualify_instrument_adaptive,
    risk_metrics,
    roofing_filter,
    run_grid_search,
)
from kinetra.renko.vpin import (
    VPINBaseline,
    VPINBucket,
    VPINTimeSeries,
    auto_bucket_size,
    classify_vpin_regime,
    compute_vpin,
    compute_vpin_multi,
    is_vpin_extreme,
    normalise_vpin,
    normalise_vpin_zscore,
    vpin_baseline,
    vpin_excess_kurtosis,
    vpin_timeseries,
)

__all__ = [
    # pipeline (Sprint 6A) — unified qualification framework
    "InstrumentKey",
    "ContractSpec",
    "load_contract_spec",
    "Candidate",
    "discover_candidates",
    "BrickEngine",
    "FixedEngine",
    "AdaptiveEngine",
    "FitContext",
    "EngineFitResult",
    "QualificationPolicy",
    "PIPELINE_DEFAULT_POLICY",
    "TierPolicy",
    "DEFAULT_TIER_POLICY",
    "PipelineQualificationResult",
    "PipelineRegistry",
    "PipelineDriftCheckResult",
    "check_drift",
    "qualify_instrument_pipeline",
    "recalibrate_pipeline",
    "build_pipeline_pool",
    # instrument_pool — tiering, scoring, pool construction
    "PoolEntry",
    "ClusterPoolStats",
    "InstrumentPool",
    "TIER1_MIN_IS_OMEGA",
    "TIER1_MIN_OOS_OMEGA",
    "TIER1_MIN_Z",
    "TIER1_MIN_OOS_SURVIVAL",
    "TIER1_MIN_FRICTION_STRESS_OMEGA",
    "TIER1_MIN_TRADES",
    "TIER1_MAX_FRICTION_RATIO",
    "build_instrument_pool",
    "build_pool_from_results",
    # live_trader (Sprint 6)
    "PERGate",
    "TradeDirection",
    "LiveTrade",
    "InstrumentLiveState",
    "RenkoSizer",
    "VolTargetSizerLive",
    "OrderResult",
    "OrderDispatcher",
    "PaperDispatcher",
    "AllocationAgent",
    "RiskAgent",
    "UniformAllocationAgent",
    "FullExposureRiskAgent",
    "BarProvider",
    "HistoricalBarProvider",
    "LiveTraderConfig",
    "RenkoLiveTrader",
    "evaluate_per_gate",
    "load_agent_from_file",
    # vol_sizer (Sprint 6 — volatility-targeted sizing)
    "VolSizingConfig",
    "VolTargetSizer",
    "VolTargetSizerLive",
    "VolBudgetCalibrationResult",
    "SizingComparisonResult",
    "PortfolioParetoPoint",
    "PortfolioParetoResult",
    "calibrate_vol_budget",
    "compare_sizing_at_risk_class",
    "compare_portfolio_sizing_pareto",
    "compute_vol_targeted_lots_batch",
    "format_pareto_report",
    "vol_sizing_report",
    # policy (post-discovery deployment profile)
    "DrawdownThrottlePolicy",
    "RegimeSizingPolicy",
    "BrickScalingPolicy",
    "XAUUSDRenkoProductionPolicy",
    "xauusd_production_policy",
    "xauusd_signal_lock",
    # ctrader_dispatcher (Sprint 6 — optional, requires ctrader-open-api)
    "CTraderBarProvider",
    "CTraderOrderDispatcher",
    "build_ctrader_session",
    # orchestrator
    "run_full_pipeline",
    "run_qualification_only",
    "load_pipeline_result",
    "PortfolioPipelineResult",
    "InstrumentPipelineResult",
    # session
    "DEFAULT_SESSION_BREAK_MINUTES",
    "GapRecord",
    "QCMetrics",
    "SessionProfile",
    "clamp_spikes",
    "detect_session_break",
    "session_break_minutes_for",
    # spread_gated_backtest (adaptive, spread-gated Renko)
    "AdaptiveBacktestConfig",
    "AdaptiveBacktestResult",
    "AdaptiveBacktestStatus",
    "AdaptiveQualificationResult",
    "BrickStabilityConfig",
    "GateType",
    "SpreadGateConfig",
    "backtest_strict",
    "build_variable_renko",
    "calculate_omega",
    "find_pareto_frontier",
    "garman_klass_sigma",
    "lots_from_capital",
    "print_qualification_summary",
    "qualify_instrument_adaptive",
    "risk_metrics",
    "roofing_filter",
    "run_grid_search",
    # qualify
    "QUALIFY_MIN_OMEGA",
    "QUALIFY_MIN_Z",
    "QUALIFY_MIN_OOS_OMEGA",
    "QUALIFY_MIN_OOS_SURVIVAL",
    "QUALIFY_FRICTION_MULT",
    "QUALIFY_FRICTION_MIN_OMEGA",
    "QUALIFY_MIN_TRADES",
    "QUALIFY_MIN_VR",
    "QUALIFY_MAX_FRICTION_RATIO",
    "DRIFT_VR_CHANGE_THRESHOLD",
    "DRIFT_OMEGA_CHANGE_THRESHOLD",
    "DRIFT_MIN_NEW_BARS",
    "QualificationResult",
    "QualificationRegistry",
    "CalibrationDriftDetector",
    "DriftCheckResult",
    "RecalibrationResult",
    "qualify_instrument",
    "recalibrate_instrument",
    # vpin
    "compute_vpin",
    "vpin_timeseries",
    "vpin_baseline",
    "compute_vpin_multi",
    "normalise_vpin",
    "normalise_vpin_zscore",
    "vpin_excess_kurtosis",
    "classify_vpin_regime",
    "is_vpin_extreme",
    "auto_bucket_size",
    "VPINBucket",
    "VPINBaseline",
    "VPINTimeSeries",
    # brick_engine
    "build_renko",
    "brick_summary",
    "bricks_per_day",
    "BrickSummary",
    # filters
    "flip_rate",
    "markov_stickiness",
    "evaluate_entry",
    "evaluate_entries_vectorized",
    # dsp
    "vr_profile",
    "brick_from_scale",
    "classify_regime",
    "compute_friction_floor",
    "compute_full_friction_floor",
    "compute_rolling_friction_floor",
    "build_rolling_spread_profile",
    "scaled_filter_params",
    "run_dsp",
    "DSPResult",
    "FrictionFloor",
    "SpreadProfile",
    # backtest
    "backtest_instrument",
    "backtest_portfolio",
    "walk_forward_instrument",
    "monte_carlo_instrument",
    "stress_test_friction",
    "sweep_brick_sizes",
    "xauusd_empirical_core",
    "FilterParams",
    "StopParams",
    "RiskParams",
    "SizingMode",
    "VolSizingParams",
    "RenkoEmpiricalCore",
    "RenkoTrade",
    "InstrumentBacktestResult",
    "PortfolioBacktestResult",
    "WalkForwardResult",
    "MonteCarloResult",
    "FrictionStressResult",
    "SweepPoint",
    "BrickSweepResult",
    # portfolio
    "CLUSTER_MAP",
    "ALL_CLUSTERS",
    "get_cluster",
    "get_cluster_members",
    "cluster_summary",
    "PortfolioConfig",
    "SizingInfo",
    "ClusterStats",
    "PortfolioConstruction",
    "estimate_usd_per_point",
    "equal_risk_weights",
    "apply_cluster_caps",
    "deduplicate_underlyings",
    "build_portfolio_equity",
    "build_portfolio",
    "herfindahl_index",
    "max_drawdown",
    "max_drawdown_duration_days",
    "calmar_ratio",
    "cvar",
    "stress_correlation_one",
    "CorrelationStressResult",
    "find_worst_period",
    "WorstPeriodResult",
    "tail_risk_analysis",
    "TailRiskReport",
]
