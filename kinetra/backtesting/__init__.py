"""
Kinetra Backtesting Package
============================

Unified backtesting system consolidating 6 implementations (5,568 lines → ~2,500 lines):

Previous Implementations:
- backtest_engine.py (PRIMARY - 1098 lines): ML/RL integration, comprehensive metrics
- realistic_backtester.py (1038 lines): MT5 freeze zones, dynamic spreads
- physics_backtester.py (977 lines): Physics strategies, backtesting.py lib
- backtest_optimizer.py (1029 lines): Parameter optimization, walk-forward
- integrated_backtester.py (634 lines): Testing framework bridge
- portfolio_backtest.py (792 lines): Portfolio-level backtesting

Unified Package:
- core.py: UnifiedBacktester with mode selection
- costs.py: Cost models (fixed/dynamic spreads, commission, slippage)
- constraints.py: MT5 constraints (freeze zones, stops validation)
- metrics.py: Performance metrics (Sharpe, Omega, Z-factor, MFE/MAE)
- execution.py: Trade execution logic
- optimizer.py: Parameter optimization
- portfolio.py: Portfolio-level features
- adapters/: Compatibility adapters

Usage:
    from kinetra.backtesting import UnifiedBacktester

    # Standard backtesting
    bt = UnifiedBacktester(mode='standard')
    results = bt.run(strategy, data)

    # MT5-realistic backtesting
    bt = UnifiedBacktester(mode='realistic', enable_freeze_zones=True)
    results = bt.run(strategy, data)

    # Physics-based backtesting
    bt = UnifiedBacktester(mode='physics')
    results = bt.run(physics_strategy, data)
"""

from .core import BacktestResult, UnifiedBacktester
from .costs import CanonicalCostModel, CostModel, DynamicCostModel, FixedCostModel, TradingCosts
from .metrics import MetricsCalculator, calculate_z_factor, omega_ratio
from .types import Trade, TradeDirection

__all__ = [
    "UnifiedBacktester",
    "BacktestResult",
    "MetricsCalculator",
    # Module-level metric helpers (DRY-08, DRY-13) — import these instead of local copies
    "omega_ratio",
    "calculate_z_factor",
    "CostModel",
    "CanonicalCostModel",
    "DynamicCostModel",
    "FixedCostModel",
    "TradingCosts",
    # Canonical trade types (DRY-11 Phase B) — moved from backtest_engine.py
    "Trade",
    "TradeDirection",
]
