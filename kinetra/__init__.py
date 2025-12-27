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
]
