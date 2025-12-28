"""
Kinetra - Physics-First Adaptive Trading System

A self-validating, reinforcement learning-based algorithmic trading system
that models markets as kinetic energy systems with damping and entropy.
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
from .trading_env import TradingEnv, Action, Position

__all__ = [
    "PhysicsEngine",
    "calculate_energy",
    "calculate_damping",
    "calculate_entropy",
    "RiskManager",
    "calculate_risk_of_ruin",
    "composite_health_score",
    "KinetraAgent",
    "AdaptiveRewardShaper",
    "BacktestEngine",
    "HealthMonitor",
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
]
