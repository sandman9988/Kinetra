"""
Kinetra - Physics-First Adaptive Trading System

A self-validating, reinforcement learning-based algorithmic trading system
that models markets as kinetic energy systems with damping and entropy.

Physics principles:
- Kinetic Energy: E = 0.5 * m * v² (market momentum)
- Damping: ζ = friction / (2 * √(k*m)) (market resistance)
- Entropy: H = -Σ p_i * log(p_i) (market disorder)

PERFORMANCE OPTIMIZATION: This module uses lazy imports to minimize load time.
Components are only imported when first accessed.
"""

__version__ = "1.0.0"
__author__ = "Kinetra Team"

import importlib
import importlib.util
from typing import TYPE_CHECKING

# Initialize silent failure logger early
from .silent_failure_logger import (
    SilentFailureLogger,
    log_failure,
    log_failures,
    get_failure_logger,
    FailureCategory,
    FailureSeverity,
)

# Type checking imports (only used by type checkers, not at runtime)
if TYPE_CHECKING:
    from .assumption_free_measures import AssumptionFreeEngine, extract_assumption_free_features
    from .backtest_engine import TradeDirection
    from .data_utils import load_mt5_csv, load_mt5_history, validate_ohlcv
    from .dsp_features import DSPFeatureEngine, extract_dsp_features
    from .health_monitor import HealthMonitor
    from .health_score import CompositeHealthScore, RewardShaper
    from .liquidity_features import LiquidityFeatureEngine, extract_liquidity_features
    from .market_microstructure import AdaptiveFrictionTracker, AssetClass, FrictionModel
    from .mt5_bridge import MT5Bridge
    from .mt5_connector import MT5Connector, MT5Session, load_csv_data
    from .persistence import AtomicCheckpointer, StreamingDataPersister
    from .physics_backtester import EnergyMomentumStrategy, PhysicsBacktestRunner
    from .physics_engine import (
        PhysicsEngine,
        calculate_damping,
        calculate_energy,
        calculate_entropy,
    )
    from .physics_v7 import AgentType, PhysicsEngineV7, PhysicsState, RegimeState
    from .realistic_backtester import BacktestResult, RealisticBacktester, Trade
    from .regime_discovery import RegimeDiscoveryEngine, discover_regimes
    from .reward_shaping import AdaptiveRewardShaper
    from .risk_management import RiskManager, calculate_risk_of_ruin, composite_health_score
    from .rl_agent import KinetraAgent
    from .strategies_v7 import BerserkerStrategy, SniperStrategy
    from .symbol_spec import CommissionSpec, CommissionType, SwapSpec, SwapType, SymbolSpec
    from .trading_env import Action, Position, TradingEnv
    from .trigger_predictor import Direction, TriggerPrediction, TriggerPredictor
    from .workflow_manager import WorkflowManager


# Module mapping for lazy imports
_LAZY_MODULES = {
    # Physics Engine
    "PhysicsEngine": "physics_engine",
    "calculate_energy": "physics_engine",
    "calculate_damping": "physics_engine",
    "calculate_entropy": "physics_engine",

    # Risk Management
    "RiskManager": "risk_management",
    "calculate_risk_of_ruin": "risk_management",
    "composite_health_score": "risk_management",

    # RL Agent
    "KinetraAgent": "rl_agent",

    # Reward Shaping
    "AdaptiveRewardShaper": "reward_shaping",

    # Backtest Engines
    "RealisticBacktester": "realistic_backtester",
    "Trade": "realistic_backtester",
    "BacktestResult": "realistic_backtester",
    "TradeDirection": "backtest_engine",

    # Physics Backtester
    "PhysicsBacktestRunner": "physics_backtester",
    "EnergyMomentumStrategy": "physics_backtester",
    "DampingReversionStrategy": "physics_backtester",
    "EntropyVolatilityStrategy": "physics_backtester",
    "AccelerationTrendStrategy": "physics_backtester",
    "MultiPhysicsStrategy": "physics_backtester",
    "ThermodynamicEquilibriumStrategy": "physics_backtester",
    "list_strategies": "physics_backtester",
    "get_strategy": "physics_backtester",
    "calculate_physics_metrics": "physics_backtester",

    # Health Monitor
    "HealthMonitor": "health_monitor",

    # MT5 Connector
    "MT5Connector": "mt5_connector",
    "MT5Session": "mt5_connector",
    "load_csv_data": "mt5_connector",

    # Symbol Specifications
    "SymbolSpec": "symbol_spec",
    "SwapSpec": "symbol_spec",
    "CommissionSpec": "symbol_spec",
    "SwapType": "symbol_spec",
    "CommissionType": "symbol_spec",
    "get_symbol_spec": "symbol_spec",
    "fetch_mt5_symbol_spec": "symbol_spec",
    "DEFAULT_SPECS": "symbol_spec",

    # Trigger Predictor
    "TriggerPredictor": "trigger_predictor",
    "TriggerPrediction": "trigger_predictor",
    "Direction": "trigger_predictor",

    # Trading Environment
    "TradingEnv": "trading_env",
    "Action": "trading_env",
    "Position": "trading_env",

    # Data Utilities
    "load_mt5_csv": "data_utils",
    "load_mt5_history": "data_utils",
    "validate_ohlcv": "data_utils",
    "preprocess_mt5_data": "data_utils",
    "get_data_summary": "data_utils",
    "split_data": "data_utils",
    "create_walk_forward_windows": "data_utils",

    # Physics v7.0
    "PhysicsEngineV7": "physics_v7",
    "PhysicsState": "physics_v7",
    "AgentType": "physics_v7",
    "RegimeState": "physics_v7",
    "EnergyWeightedExitManager": "physics_v7",
    "calculate_omega_ratio": "physics_v7",
    "calculate_z_factor": "physics_v7",
    "calculate_energy_captured": "physics_v7",
    "validate_theorem_targets": "physics_v7",

    # v7.0 Strategies
    "BerserkerStrategy": "strategies_v7",
    "SniperStrategy": "strategies_v7",
    "MultiAgentV7Strategy": "strategies_v7",
    "list_v7_strategies": "strategies_v7",
    "get_v7_strategy": "strategies_v7",

    # Health Score & Reward Shaping
    "RewardShaper": "health_score",
    "CompositeHealthScore": "health_score",
    "TradeReward": "health_score",
    "HealthScore": "health_score",
    "compute_reward_from_trade": "health_score",
    "compute_health_from_metrics": "health_score",

    # Market Microstructure & Friction
    "AssetClass": "market_microstructure",
    "TradingMode": "market_microstructure",
    "FrictionModel": "market_microstructure",
    "AdaptiveFrictionTracker": "market_microstructure",
    "compute_friction_series": "market_microstructure",
    "SYMBOL_SPECS": "market_microstructure",

    # MT5 Bridge
    "MT5Bridge": "mt5_bridge",
    "save_bridge_server_script": "mt5_bridge",

    # Atomic Persistence
    "AtomicCheckpointer": "persistence",
    "StreamingDataPersister": "persistence",
    "CheckpointType": "persistence",
    "create_checkpointer": "persistence",

    # DSP Features (Assumption-Free)
    "DSPFeatureEngine": "dsp_features",
    "WaveletExtractor": "dsp_features",
    "HilbertExtractor": "dsp_features",
    "EntropyExtractor": "dsp_features",
    "DirectionalWaveletExtractor": "dsp_features",
    "extract_dsp_features": "dsp_features",

    # Liquidity Proxies
    "LiquidityFeatureEngine": "liquidity_features",
    "CVDExtractor": "liquidity_features",
    "AmihudExtractor": "liquidity_features",
    "RangeImpactExtractor": "liquidity_features",
    "VolumeImbalanceExtractor": "liquidity_features",
    "extract_liquidity_features": "liquidity_features",

    # Assumption-Free Measures
    "AsymmetricReturns": "assumption_free_measures",
    "RankBasedMeasures": "assumption_free_measures",
    "DirectionalVolatility": "assumption_free_measures",
    "DirectionalOrderFlow": "assumption_free_measures",
    "PermutationPatterns": "assumption_free_measures",
    "RecurrenceFeatures": "assumption_free_measures",
    "TailBehavior": "assumption_free_measures",
    "AssumptionFreeEngine": "assumption_free_measures",
    "extract_assumption_free_features": "assumption_free_measures",

    # Regime Discovery
    "RegimeDiscoveryEngine": "regime_discovery",
    "RegimeDiscoveryResult": "regime_discovery",
    "RegimeProfile": "regime_discovery",
    "TransitionPrecursor": "regime_discovery",
    "CrossAssetRegimeAnalyzer": "regime_discovery",
    "discover_regimes": "regime_discovery",

    # Triad System (Incumbent/Competitor/Researcher)
    "TriadSystem": "triad_system",
    "AgentRole": "triad_system",
    "RegimeState": "triad_system",
    "ImbalanceState": "triad_system",
    "ImbalanceExtractor": "triad_system",
    "RegimeDetector": "triad_system",
    "MetaController": "triad_system",
    "IncumbentAgent": "triad_system",
    "CompetitorAgent": "triad_system",
    "ResearcherAgent": "triad_system",
    "create_triad_system": "triad_system",

    # Workflow Management
    "WorkflowManager": "workflow_manager",

    # Financial Audit
    "SafeMath": "financial_audit",
    "DigitNormalizer": "financial_audit",
    "PnLCalculator": "financial_audit",
    "RiskMetricsCalculator": "financial_audit",
    "AuditTrail": "financial_audit",
    "AuditIssue": "financial_audit",
    "AuditSeverity": "financial_audit",

    # Position Manager
    "PositionManager": "position_manager",
    "Position": "position_manager",
    "PositionState": "position_manager",
    "PositionSide": "position_manager",
    "PositionEvent": "position_manager",

    # Broker Compliance
    "BrokerComplianceValidator": "broker_compliance",
    "GracefulExecutor": "broker_compliance",
    "ValidationResult": "broker_compliance",
    "ValidationIssue": "broker_compliance",
    "ThrottleState": "broker_compliance",
    "with_graceful_failure": "broker_compliance",

    # Backtest Optimizer
    "BayesianOptimizer": "backtest_optimizer",
    "GeneticOptimizer": "backtest_optimizer",
    "ParetoOptimizer": "backtest_optimizer",
    "ParameterSpace": "backtest_optimizer",
    "Parameter": "backtest_optimizer",
    "DistributionAnalyzer": "backtest_optimizer",
    "ObjectiveFunction": "backtest_optimizer",
    "calculate_objective": "backtest_optimizer",

    # Performance Module
    "RingBuffer": "performance",
    "TickBuffer": "performance",
    "BarBuffer": "performance",
    "LRUCache": "performance",
    "ComputeCache": "performance",
    "cached_property": "performance",
    "AsyncExecutor": "performance",
    "AsyncDataStream": "performance",
    "Timer": "performance",
    "OnTickHandler": "performance",
    "OnTimerHandler": "performance",
    "ParallelProcessor": "performance",
    "PerformanceMetrics": "performance",
    "PerformanceMonitor": "performance",

    # Network Resilience
    "ConnectionManager": "network_resilience",
    "ServerPool": "network_resilience",
    "ServerEndpoint": "network_resilience",
    "LatencyMonitor": "network_resilience",
    "PacketTracker": "network_resilience",
    "ReconnectHandler": "network_resilience",
    "RedundantConnection": "network_resilience",
    "ConnectionState": "network_resilience",
    "StressTest": "network_resilience",

    # Hardware Optimizer
    "HardwareDetector": "hardware_optimizer",
    "HardwareOptimizer": "hardware_optimizer",
    "HardwareProfile": "hardware_optimizer",
    "OptimizedConfig": "hardware_optimizer",
    "CPUInfo": "hardware_optimizer",
    "MemoryInfo": "hardware_optimizer",
    "GPUInfo": "hardware_optimizer",
    "PerformanceTier": "hardware_optimizer",
    "auto_configure": "hardware_optimizer",

    # Trading Costs
    "TradingCostCalculator": "trading_costs",
    "TradingCostSpec": "trading_costs",
    "TradeCosts": "trading_costs",
    "SwapCalendar": "trading_costs",
    "CostAnalyzer": "trading_costs",
    "get_forex_major_spec": "trading_costs",
    "get_index_cfd_spec": "trading_costs",
    "get_crypto_spec": "trading_costs",

    # Symbol Info
    "SymbolInfo": "symbol_info",
    "get_symbol_info": "symbol_info",
    "register_symbol": "symbol_info",
    "list_symbols": "symbol_info",
    "SYMBOL_REGISTRY": "symbol_info",
    "calculate_position_value": "symbol_info",
    "calculate_pip_value": "symbol_info",
    "calculate_profit_loss": "symbol_info",

    # MQL5 Trade Classes
    "CAccountInfo": "mql5_trade_classes",
    "CSymbolInfo": "mql5_trade_classes",
    "COrderInfo": "mql5_trade_classes",
    "CHistoryOrderInfo": "mql5_trade_classes",
    "CPositionInfo": "mql5_trade_classes",
    "CDealInfo": "mql5_trade_classes",
    "CTrade": "mql5_trade_classes",
    "CTerminalInfo": "mql5_trade_classes",
    "MqlTradeRequest": "mql5_trade_classes",
    "MqlTradeResult": "mql5_trade_classes",
    "ENUM_ORDER_TYPE": "mql5_trade_classes",
    "ENUM_POSITION_TYPE": "mql5_trade_classes",
    "ENUM_DEAL_TYPE": "mql5_trade_classes",
    "ENUM_TRADE_RETCODE": "mql5_trade_classes",

    # High Performance Engine
    "HighPerformanceEngine": "high_performance_engine",
    "HighPerformanceDataFeed": "high_performance_engine",
    "CachedIndicatorEngine": "high_performance_engine",
    "SignalGenerator": "high_performance_engine",
    "TickData": "high_performance_engine",
    "BarData": "high_performance_engine",
    "Signal": "high_performance_engine",
}

# Cache for loaded modules
_loaded_modules = {}


def __getattr__(name: str):
    """
    Lazy import handler - only imports modules when accessed.

    This significantly reduces initial import time by deferring
    heavy imports until they are actually needed.
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
                    if name in ("KinetraAgent", "TradingEnv", "Action", "Position"):
                        return None
                raise

        return getattr(_loaded_modules[module_name], name)

    raise AttributeError(f"module 'kinetra' has no attribute '{name}'")


def __dir__():
    """Return available attributes for IDE autocomplete support."""
    return list(_LAZY_MODULES.keys()) + ["__version__", "__author__", "_RL_AVAILABLE"]


# Check if RL components are available (for backward compatibility)
_RL_AVAILABLE = importlib.util.find_spec("torch") is not None


__all__ = [
    # Physics Engine
    "PhysicsEngine",
    "calculate_energy",
    "calculate_damping",
    "calculate_entropy",
    # Risk Management
    "RiskManager",
    "calculate_risk_of_ruin",
    "composite_health_score",
    # RL Agent
    "KinetraAgent",
    # Reward Shaping
    "AdaptiveRewardShaper",
    # Backtest Engines
    "RealisticBacktester",
    "PhysicsBacktestRunner",
    # Physics Strategies
    "EnergyMomentumStrategy",
    "DampingReversionStrategy",
    "EntropyVolatilityStrategy",
    "AccelerationTrendStrategy",
    "MultiPhysicsStrategy",
    "ThermodynamicEquilibriumStrategy",
    "list_strategies",
    "get_strategy",
    "calculate_physics_metrics",
    # Health Monitor
    "HealthMonitor",
    # Data Utilities
    "load_mt5_csv",
    "load_mt5_history",
    "validate_ohlcv",
    "preprocess_mt5_data",
    "get_data_summary",
    "split_data",
    "create_walk_forward_windows",
    # Physics v7.0
    "PhysicsEngineV7",
    "PhysicsState",
    "AgentType",
    "RegimeState",
    "EnergyWeightedExitManager",
    "calculate_omega_ratio",
    "calculate_z_factor",
    "calculate_energy_captured",
    "validate_theorem_targets",
    # v7.0 Strategies
    "BerserkerStrategy",
    "SniperStrategy",
    "MultiAgentV7Strategy",
    "list_v7_strategies",
    "get_v7_strategy",
    # Health Score & Reward Shaping
    "RewardShaper",
    "CompositeHealthScore",
    "TradeReward",
    "HealthScore",
    "compute_reward_from_trade",
    "compute_health_from_metrics",
    # Market Microstructure & Friction
    "SymbolSpec",
    "AssetClass",
    "TradingMode",
    "FrictionModel",
    "AdaptiveFrictionTracker",
    "get_symbol_spec",
    "compute_friction_series",
    # MT5 Bridge
    "MT5Bridge",
    "save_bridge_server_script",
    # Atomic Persistence
    "AtomicCheckpointer",
    "StreamingDataPersister",
    "CheckpointType",
    "create_checkpointer",
    "MT5Connector",
    "MT5Session",
    "load_csv_data",
    # Symbol specifications
    "SwapSpec",
    "CommissionSpec",
    "SwapType",
    "CommissionType",
    "fetch_mt5_symbol_spec",
    "DEFAULT_SPECS",
    # Backtest types
    "Trade",
    "TradeDirection",
    "BacktestResult",
    # Trigger prediction
    "TriggerPredictor",
    "TriggerPrediction",
    "Direction",
    # RL environment
    "TradingEnv",
    "Action",
    "Position",
    # DSP Features (Assumption-Free)
    "DSPFeatureEngine",
    "WaveletExtractor",
    "HilbertExtractor",
    "EntropyExtractor",
    "DirectionalWaveletExtractor",
    "extract_dsp_features",
    # Liquidity Proxies
    "LiquidityFeatureEngine",
    "CVDExtractor",
    "AmihudExtractor",
    "RangeImpactExtractor",
    "VolumeImbalanceExtractor",
    "extract_liquidity_features",
    # Assumption-Free Measures
    "AsymmetricReturns",
    "RankBasedMeasures",
    "DirectionalVolatility",
    "DirectionalOrderFlow",
    "PermutationPatterns",
    "RecurrenceFeatures",
    "TailBehavior",
    "AssumptionFreeEngine",
    "extract_assumption_free_features",
    # Regime Discovery
    "RegimeDiscoveryEngine",
    "RegimeDiscoveryResult",
    "RegimeProfile",
    "TransitionPrecursor",
    "CrossAssetRegimeAnalyzer",
    "discover_regimes",
    # Workflow Management
    "WorkflowManager",
    # Performance Module
    "RingBuffer",
    "TickBuffer",
    "BarBuffer",
    "LRUCache",
    "ComputeCache",
    "cached_property",
    "AsyncExecutor",
    "AsyncDataStream",
    "Timer",
    "OnTickHandler",
    "OnTimerHandler",
    "ParallelProcessor",
    "PerformanceMetrics",
    "PerformanceMonitor",
]
