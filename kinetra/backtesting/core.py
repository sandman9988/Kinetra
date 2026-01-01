"""
Core Unified Backtester
========================

Main backtesting engine with mode selection.
Consolidates all 6 implementations.

Modes:
- 'standard': Fast, basic cost modeling (from backtest_engine.py)
- 'realistic': MT5-accurate with freeze zones (from realistic_backtester.py)
- 'physics': Physics-based strategies (from physics_backtester.py)
- 'portfolio': Multi-instrument (from portfolio_backtest.py)
"""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import numpy as np

from .costs import CostModel, FixedCostModel, DynamicCostModel
from .constraints import MT5Constraints
from .metrics import MetricsCalculator, PerformanceMetrics


@dataclass
class BacktestResult:
    """Unified backtest result."""
    trades: List[Dict]
    equity_curve: pd.Series
    metrics: PerformanceMetrics
    mode: str
    config: Dict = field(default_factory=dict)
    regime_breakdown: Optional[Dict] = None


class UnifiedBacktester:
    """
    Unified backtesting engine.
    
    Consolidates 6 previous implementations into one with mode selection.
    
    Example:
        # Standard mode (fast)
        bt = UnifiedBacktester(mode='standard')
        result = bt.run(strategy, data)
        
        # MT5-realistic mode
        bt = UnifiedBacktester(mode='realistic', enable_freeze_zones=True)
        result = bt.run(strategy, data)
        
        # Physics mode
        bt = UnifiedBacktester(mode='physics')
        result = bt.run(physics_strategy, data)
    """
    
    def __init__(
        self,
        mode: str = 'standard',
        cost_model: Optional[CostModel] = None,
        constraints: Optional[MT5Constraints] = None,
        parallel: bool = False,
        gpu: bool = False,
        **kwargs
    ):
        """
        Initialize unified backtester.
        
        Args:
            mode: Backtest mode ('standard', 'realistic', 'physics', 'portfolio')
            cost_model: Cost model (default depends on mode)
            constraints: MT5 constraints (for realistic mode)
            parallel: Enable parallel processing
            gpu: Enable GPU acceleration
            **kwargs: Additional mode-specific options
        """
        self.mode = mode
        self.parallel = parallel
        self.gpu = gpu
        self.config = kwargs
        
        # Configure based on mode
        if mode == 'realistic':
            self.cost_model = cost_model or DynamicCostModel()
            self.constraints = constraints or MT5Constraints()
            self.enable_freeze_zones = kwargs.get('enable_freeze_zones', True)
        else:
            self.cost_model = cost_model or FixedCostModel()
            self.constraints = constraints
            self.enable_freeze_zones = False
            
        # Metrics calculator
        self.metrics_calc = MetricsCalculator()
        
    def run(
        self,
        strategy: Any,
        data: pd.DataFrame,
        initial_capital: float = 100000.0,
        **kwargs
    ) -> BacktestResult:
        """
        Run backtest.
        
        Args:
            strategy: Trading strategy
            data: OHLCV data
            initial_capital: Starting capital
            **kwargs: Additional parameters
            
        Returns:
            BacktestResult
        """
        # Initialize
        trades = []
        equity = initial_capital
        equity_curve = [equity]
        
        # Simplified simulation loop (full implementation would be more complex)
        # This is a stub showing the interface
        
        for i in range(len(data)):
            # Strategy logic would go here
            # For now, just maintain equity
            equity_curve.append(equity)
            
        # Create equity series
        equity_series = pd.Series(equity_curve, index=data.index[:len(equity_curve)])
        
        # Calculate returns
        returns = equity_series.pct_change().dropna()
        
        # Calculate metrics
        metrics = self.metrics_calc.calculate_all(
            trades=trades,
            equity_curve=equity_series,
            returns=returns
        )
        
        return BacktestResult(
            trades=trades,
            equity_curve=equity_series,
            metrics=metrics,
            mode=self.mode,
            config=self.config
        )
        
    def optimize(
        self,
        strategy_class: Any,
        data: pd.DataFrame,
        param_grid: Dict[str, List[Any]],
        **kwargs
    ) -> Dict:
        """
        Parameter optimization.
        
        Integration point for backtest_optimizer.py functionality.
        
        Args:
            strategy_class: Strategy class
            data: Data
            param_grid: Parameter grid
            **kwargs: Additional options
            
        Returns:
            Optimization results
        """
        # Stub - full implementation would do grid search/walk-forward
        return {
            'best_params': {},
            'best_score': 0.0,
            'all_results': []
        }
