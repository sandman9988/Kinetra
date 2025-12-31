"""
Kinetra - Physics-First Adaptive Trading System

A self-validating, reinforcement learning-based algorithmic trading system
that models markets as kinetic energy systems with damping and entropy.

Physics principles:
- Kinetic Energy: E = 0.5 * m * v² (market momentum)
- Damping: ζ = friction / (2 * √(k*m)) (market resistance)
- Entropy: H = -Σ p_i * log(p_i) (market disorder)
"""

__version__ = "1.0.0"
__author__ = "Kinetra Team"

from .physics_engine import PhysicsEngine, calculate_energy, calculate_damping, calculate_entropy
from .risk_management import RiskManager, calculate_risk_of_ruin, composite_health_score
from .rl_agent import KinetraAgent
from .reward_shaping import AdaptiveRewardShaper
from .backtest_engine import BacktestEngine
from .health_monitor import HealthMonitor
from .mt5_connector import MT5Connector, MT5Session, load_csv_data
from .symbol_spec import (
    SymbolSpec, SwapSpec, CommissionSpec, SwapType, CommissionType,
    get_symbol_spec, fetch_mt5_symbol_spec, DEFAULT_SPECS
)
from .backtest_engine import Trade, TradeDirection, BacktestResult
from .trigger_predictor import TriggerPredictor, TriggerPrediction, Direction

# RL components require PyTorch
try:
    from .rl_agent import KinetraAgent
    from .trading_env import TradingEnv, Action, Position
    _RL_AVAILABLE = True
except ImportError:
    KinetraAgent = None
    TradingEnv = None
    Action = None
    Position = None
    _RL_AVAILABLE = False

# Physics backtester (using backtesting.py)
from .physics_backtester import (
    PhysicsBacktestRunner,
    EnergyMomentumStrategy,
    DampingReversionStrategy,
    EntropyVolatilityStrategy,
    AccelerationTrendStrategy,
    MultiPhysicsStrategy,
    ThermodynamicEquilibriumStrategy,
    list_strategies,
    get_strategy,
    calculate_physics_metrics,
)

# Data utilities for MT5
from .data_utils import (
    load_mt5_csv,
    load_mt5_history,
    validate_ohlcv,
    preprocess_mt5_data,
    get_data_summary,
    split_data,
    create_walk_forward_windows,
)

# Physics v7.0 (Energy-Transfer Theorem)
from .physics_v7 import (
    PhysicsEngineV7,
    PhysicsState,
    AgentType,
    RegimeState,
    EnergyWeightedExitManager,
    calculate_omega_ratio,
    calculate_z_factor,
    calculate_energy_captured,
    validate_theorem_targets,
)

# v7.0 Strategies (Berserker, Sniper)
from .strategies_v7 import (
    BerserkerStrategy,
    SniperStrategy,
    MultiAgentV7Strategy,
    list_v7_strategies,
    get_v7_strategy,
)

# Health Score & Reward Shaping
from .health_score import (
    RewardShaper,
    CompositeHealthScore,
    TradeReward,
    HealthScore,
    compute_reward_from_trade,
    compute_health_from_metrics,
)

# Market Microstructure & Friction
from .market_microstructure import (
    SymbolSpec,
    AssetClass,
    TradingMode,
    FrictionModel,
    AdaptiveFrictionTracker,
    get_symbol_spec,
    compute_friction_series,
)

# MT5 Bridge (Direct, MetaAPI, Bridge modes)
from .mt5_bridge import (
    MT5Bridge,
    save_bridge_server_script,
)

# Atomic Persistence (crash-safe checkpointing)
from .persistence import (
    AtomicCheckpointer,
    StreamingDataPersister,
    CheckpointType,
    create_checkpointer,
)

# DSP-Driven Features (Assumption-Free)
from .dsp_features import (
    DSPFeatureEngine,
    WaveletExtractor,
    HilbertExtractor,
    EntropyExtractor,
    HurstExtractor,
    extract_dsp_features,
)

# Liquidity Proxies (Asymmetric Order Flow)
from .liquidity_features import (
    LiquidityFeatureEngine,
    CVDExtractor,
    AmihudExtractor,
    RangeImpactExtractor,
    VolumeImbalanceExtractor,
    extract_liquidity_features,
)

# Unsupervised Regime Discovery
from .regime_discovery import (
    RegimeDiscoveryEngine,
    RegimeDiscoveryResult,
    RegimeProfile,
    TransitionPrecursor,
    CrossAssetRegimeAnalyzer,
    discover_regimes,
)

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
    "BacktestEngine",
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
    "SymbolSpec",
    "SwapSpec",
    "CommissionSpec",
    "SwapType",
    "CommissionType",
    "get_symbol_spec",
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
    "HurstExtractor",
    "extract_dsp_features",
    # Liquidity Proxies
    "LiquidityFeatureEngine",
    "CVDExtractor",
    "AmihudExtractor",
    "RangeImpactExtractor",
    "VolumeImbalanceExtractor",
    "extract_liquidity_features",
    # Regime Discovery
    "RegimeDiscoveryEngine",
    "RegimeDiscoveryResult",
    "RegimeProfile",
    "TransitionPrecursor",
    "CrossAssetRegimeAnalyzer",
    "discover_regimes",
]
