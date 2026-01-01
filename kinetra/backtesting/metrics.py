"""
Performance Metrics
===================

Standardized performance metrics calculation.
Consolidated from all 6 backtest implementations.

Features:
- Sharpe ratio (standardized - was calculated 5 different ways!)
- Omega ratio
- Z-factor
- MFE/MAE analysis
- Regime-based breakdown
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from dataclasses import dataclass


@dataclass
class PerformanceMetrics:
    """Standard performance metrics."""
    total_return: float
    sharpe_ratio: float
    omega_ratio: float
    z_factor: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    avg_mfe: float  # Average Maximum Favorable Excursion
    avg_mae: float  # Average Maximum Adverse Excursion
    total_trades: int


class MetricsCalculator:
    """
    Standardized metrics calculation.
    
    Eliminates the 5 different Sharpe ratio calculations!
    """
    
    @staticmethod
    def sharpe_ratio(
        returns: pd.Series,
        periods_per_year: int = 252,
        risk_free_rate: float = 0.0
    ) -> float:
        """
        Calculate Sharpe ratio (STANDARDIZED).
        
        Args:
            returns: Return series
            periods_per_year: Scaling factor (252 for daily, 252*24 for hourly)
            risk_free_rate: Risk-free rate (annualized)
            
        Returns:
            Sharpe ratio
        """
        if len(returns) < 2:
            return 0.0
            
        excess_returns = returns - (risk_free_rate / periods_per_year)
        
        if excess_returns.std() == 0:
            return 0.0
            
        return (
            excess_returns.mean() / excess_returns.std() * np.sqrt(periods_per_year)
        )
        
    @staticmethod
    def omega_ratio(returns: pd.Series, threshold: float = 0.0) -> float:
        """
        Calculate Omega ratio (probability-weighted gain/loss).
        
        Args:
            returns: Return series
            threshold: Return threshold
            
        Returns:
            Omega ratio (>1 is good, >2.7 is target)
        """
        returns_above = returns[returns > threshold] - threshold
        returns_below = threshold - returns[returns < threshold]
        
        gains = returns_above.sum()
        losses = returns_below.sum()
        
        if losses == 0:
            return np.inf if gains > 0 else 0.0
            
        return gains / losses
        
    @staticmethod
    def z_factor(trades: List[Dict]) -> float:
        """
        Calculate Z-factor (statistical significance of edge).
        
        Args:
            trades: List of trades with 'pnl' field
            
        Returns:
            Z-factor (>2.5 is statistically significant)
        """
        if len(trades) < 2:
            return 0.0
            
        pnls = np.array([t['pnl'] for t in trades])
        
        wins = pnls[pnls > 0]
        losses = pnls[pnls < 0]
        
        if len(wins) == 0 or len(losses) == 0:
            return 0.0
            
        avg_win = wins.mean()
        avg_loss = abs(losses.mean())
        std_loss = losses.std()
        
        if std_loss == 0:
            return 0.0
            
        return (avg_win - avg_loss) / std_loss
        
    @staticmethod
    def max_drawdown(equity_curve: pd.Series) -> float:
        """
        Calculate maximum drawdown.
        
        Args:
            equity_curve: Equity over time
            
        Returns:
            Max drawdown as percentage (negative)
        """
        running_max = equity_curve.expanding().max()
        drawdown = (equity_curve - running_max) / running_max
        return drawdown.min()
        
    @staticmethod
    def calculate_all(
        trades: List[Dict],
        equity_curve: pd.Series,
        returns: pd.Series
    ) -> PerformanceMetrics:
        """
        Calculate all metrics.
        
        Args:
            trades: List of trades
            equity_curve: Equity over time
            returns: Return series
            
        Returns:
            PerformanceMetrics object
        """
        # Basic stats
        total_return = (equity_curve.iloc[-1] / equity_curve.iloc[0] - 1) if len(equity_curve) > 0 else 0.0
        
        # Win rate and profit factor
        winning_trades = [t for t in trades if t['pnl'] > 0]
        losing_trades = [t for t in trades if t['pnl'] < 0]
        
        win_rate = len(winning_trades) / len(trades) if trades else 0.0
        
        total_wins = sum(t['pnl'] for t in winning_trades)
        total_losses = abs(sum(t['pnl'] for t in losing_trades))
        profit_factor = total_wins / total_losses if total_losses > 0 else 0.0
        
        # MFE/MAE
        avg_mfe = np.mean([t.get('mfe', 0) for t in trades]) if trades else 0.0
        avg_mae = np.mean([t.get('mae', 0) for t in trades]) if trades else 0.0
        
        return PerformanceMetrics(
            total_return=total_return,
            sharpe_ratio=MetricsCalculator.sharpe_ratio(returns),
            omega_ratio=MetricsCalculator.omega_ratio(returns),
            z_factor=MetricsCalculator.z_factor(trades),
            max_drawdown=MetricsCalculator.max_drawdown(equity_curve),
            win_rate=win_rate,
            profit_factor=profit_factor,
            avg_mfe=avg_mfe,
            avg_mae=avg_mae,
            total_trades=len(trades)
        )
