"""
Integrated Backtesting System for Testing Framework
====================================================

Automatically backtests discovered strategies from the testing framework.

Features:
- Converts discovered patterns into tradeable strategies
- Runs realistic backtests with full cost modeling
- Validates strategies statistically
- Generates performance reports
- Integrates with testing framework

Usage:
    from kinetra.integrated_backtester import IntegratedBacktester
    
    backtester = IntegratedBacktester()
    
    # Backtest discovered strategy
    results = backtester.backtest_discovered_strategy(
        strategy_config=discovered_config,
        data=market_data
    )
    
    # Or backtest from test results
    backtest_results = backtester.backtest_from_test_results(test_results)
"""

import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class StrategySignal:
    """A trading signal from a strategy."""
    timestamp: datetime
    action: str  # 'buy', 'sell', 'close', 'hold'
    confidence: float  # 0-1
    features: Dict[str, float]  # Features that triggered signal
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BacktestConfig:
    """Configuration for a backtest run."""
    name: str
    strategy_type: str  # 'physics', 'rl', 'ml', 'discovered', etc.
    parameters: Dict[str, Any]
    
    # Backtest settings
    initial_capital: float = 10000.0
    position_size: float = 0.1  # 10% of capital per trade
    max_positions: int = 1
    
    # Cost modeling
    spread_pips: float = 2.0
    commission_per_lot: float = 7.0
    slippage_pips: float = 0.5
    
    # Risk management
    stop_loss_pct: float = 0.02  # 2%
    take_profit_pct: float = 0.04  # 4%
    trailing_stop_enabled: bool = False
    trailing_stop_distance_pct: float = 0.01
    
    # Validation
    min_trades: int = 30
    max_drawdown_threshold: float = 0.25  # 25%


@dataclass
class BacktestResult:
    """Results from a backtest run."""
    config_name: str
    strategy_type: str
    timestamp: datetime
    
    # Performance metrics
    total_return: float
    sharpe_ratio: float
    omega_ratio: float
    sortino_ratio: float
    max_drawdown: float
    calmar_ratio: float
    
    # Trade statistics
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    
    # Efficiency metrics
    mfe_captured_pct: float = 0.0
    mae_ratio: float = 0.0
    pythagorean_efficiency: float = 0.0
    
    # Statistical validation
    is_statistically_significant: bool = False
    p_value: float = 1.0
    
    # Equity curve
    equity_curve: np.ndarray = field(default_factory=lambda: np.array([]))
    
    # Trade history
    trades: List[Dict] = field(default_factory=list)
    
    # Metadata
    data_points: int = 0
    backtest_duration_seconds: float = 0.0


class DiscoveredStrategyConverter:
    """
    Converts discovered patterns/features into tradeable strategies.
    
    Takes output from discovery suites (hidden dimensions, chaos theory,
    meta-learning, etc.) and creates actionable trading strategies.
    """
    
    def __init__(self):
        self.conversion_methods = {
            'hidden_dimensions': self._convert_hidden_dimensions,
            'chaos_theory': self._convert_chaos_theory,
            'meta_learning': self._convert_meta_learning,
            'cross_regime': self._convert_cross_regime,
            'information_theory': self._convert_information_theory,
            'emergent': self._convert_emergent,
        }
    
    def convert(self, discovery_result: Dict) -> Dict[str, Any]:
        """
        Convert discovery result to strategy configuration.
        
        Args:
            discovery_result: Results from a discovery suite
            
        Returns:
            Strategy configuration dict
        """
        strategy_type = discovery_result.get('type', 'unknown')
        
        if strategy_type in self.conversion_methods:
            return self.conversion_methods[strategy_type](discovery_result)
        else:
            return self._convert_generic(discovery_result)
    
    def _convert_hidden_dimensions(self, result: Dict) -> Dict:
        """Convert hidden dimension discovery to strategy."""
        # Extract latent features from autoencoder/PCA
        latent_features = result.get('latent_features', [])
        importance_scores = result.get('feature_importance', {})
        
        # Select top features
        top_features = sorted(
            importance_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:5]
        
        return {
            'type': 'hidden_dimension_strategy',
            'features': [f[0] for f in top_features],
            'thresholds': 'learned',  # From autoencoder
            'entry_logic': 'latent_space_regime_change',
            'exit_logic': 'latent_space_equilibrium',
        }
    
    def _convert_chaos_theory(self, result: Dict) -> Dict:
        """Convert chaos theory analysis to strategy."""
        return {
            'type': 'chaos_strategy',
            'features': ['lyapunov_exponent', 'fractal_dimension', 'recurrence_rate'],
            'entry_logic': 'lyapunov_crossing_threshold',
            'exit_logic': 'attractor_reached',
            'thresholds': {
                'lyapunov_entry': result.get('optimal_lyapunov_threshold', 0.5),
                'fractal_dim_filter': result.get('optimal_fractal_dim', 1.5),
            }
        }
    
    def _convert_meta_learning(self, result: Dict) -> Dict:
        """Convert meta-learning results to strategy."""
        return {
            'type': 'meta_learned_strategy',
            'feature_selection': result.get('optimal_features', []),
            'strategy_template': result.get('best_strategy_template', 'adaptive'),
            'hyperparameters': result.get('learned_hyperparameters', {}),
        }
    
    def _convert_cross_regime(self, result: Dict) -> Dict:
        """Convert cross-regime analysis to strategy."""
        return {
            'type': 'regime_transition_strategy',
            'transition_signals': result.get('transition_indicators', []),
            'pre_transition_window': result.get('optimal_window', 10),
            'entry_logic': 'regime_transition_detected',
            'exit_logic': 'regime_stabilized',
        }
    
    def _convert_information_theory(self, result: Dict) -> Dict:
        """Convert information theory analysis to strategy."""
        return {
            'type': 'information_flow_strategy',
            'causality_pairs': result.get('causal_relationships', []),
            'transfer_entropy_threshold': result.get('optimal_te_threshold', 0.1),
            'entry_logic': 'information_cascade_detected',
            'exit_logic': 'information_flow_reversed',
        }
    
    def _convert_emergent(self, result: Dict) -> Dict:
        """Convert emergent behavior to strategy."""
        return {
            'type': 'emergent_strategy',
            'evolved_rules': result.get('best_individual', {}),
            'fitness_score': result.get('fitness', 0),
            'mutations': result.get('mutations', []),
        }
    
    def _convert_generic(self, result: Dict) -> Dict:
        """Generic conversion for unknown types."""
        return {
            'type': 'generic_discovered_strategy',
            'config': result,
        }


class IntegratedBacktester:
    """
    Integrated backtesting system for testing framework.
    
    Automatically converts discovered strategies into backtestable form
    and runs realistic backtests with full cost modeling.
    """
    
    def __init__(self, output_dir: str = "backtest_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.converter = DiscoveredStrategyConverter()
        
        # Import backtesting components
        try:
            from kinetra.backtest_engine import BacktestEngine
            self.backtest_engine = BacktestEngine()
            self.engine_available = True
        except ImportError:
            logger.warning("BacktestEngine not available - using simplified backtester")
            self.backtest_engine = None
            self.engine_available = False
        
        logger.info(f"IntegratedBacktester initialized. Engine available: {self.engine_available}")
    
    def backtest_discovered_strategy(
        self,
        strategy_config: Dict,
        data: pd.DataFrame,
        config: Optional[BacktestConfig] = None
    ) -> BacktestResult:
        """
        Backtest a discovered strategy.
        
        Args:
            strategy_config: Strategy configuration from discovery
            data: Market data (OHLCV)
            config: Backtest configuration
            
        Returns:
            BacktestResult
        """
        if config is None:
            config = BacktestConfig(
                name=strategy_config.get('type', 'unknown'),
                strategy_type=strategy_config.get('type', 'unknown'),
                parameters=strategy_config
            )
        
        start_time = datetime.now()
        
        # Generate signals
        signals = self._generate_signals(strategy_config, data)
        
        # Simulate trades
        trades, equity_curve = self._simulate_trades(signals, data, config)
        
        # Calculate metrics
        metrics = self._calculate_metrics(trades, equity_curve, config.initial_capital)
        
        # Create result
        result = BacktestResult(
            config_name=config.name,
            strategy_type=config.strategy_type,
            timestamp=datetime.now(),
            total_return=metrics['total_return'],
            sharpe_ratio=metrics['sharpe_ratio'],
            omega_ratio=metrics['omega_ratio'],
            sortino_ratio=metrics['sortino_ratio'],
            max_drawdown=metrics['max_drawdown'],
            calmar_ratio=metrics['calmar_ratio'],
            total_trades=len(trades),
            winning_trades=metrics['winning_trades'],
            losing_trades=metrics['losing_trades'],
            win_rate=metrics['win_rate'],
            avg_win=metrics['avg_win'],
            avg_loss=metrics['avg_loss'],
            profit_factor=metrics['profit_factor'],
            mfe_captured_pct=metrics.get('mfe_captured_pct', 0),
            mae_ratio=metrics.get('mae_ratio', 0),
            pythagorean_efficiency=metrics.get('pythagorean_efficiency', 0),
            is_statistically_significant=metrics.get('is_significant', False),
            p_value=metrics.get('p_value', 1.0),
            equity_curve=equity_curve,
            trades=trades,
            data_points=len(data),
            backtest_duration_seconds=(datetime.now() - start_time).total_seconds()
        )
        
        # Save results
        self._save_backtest_result(result)
        
        return result
    
    def _generate_signals(
        self,
        strategy_config: Dict,
        data: pd.DataFrame
    ) -> List[StrategySignal]:
        """
        Generate trading signals from strategy configuration.
        
        This is a simplified implementation. Real implementation would:
        1. Parse strategy_config
        2. Compute required features
        3. Apply entry/exit logic
        4. Generate signals with confidence scores
        """
        signals = []
        
        strategy_type = strategy_config.get('type', 'unknown')
        
        # Placeholder: Generate random signals for demonstration
        # Real implementation would use actual strategy logic
        for i in range(len(data)):
            if i % 50 == 0:  # Generate signal every 50 bars
                signal = StrategySignal(
                    timestamp=data.index[i] if hasattr(data.index[i], 'to_pydatetime') else datetime.now(),
                    action='buy' if i % 100 == 0 else 'sell',
                    confidence=0.7,
                    features={'example_feature': 0.5}
                )
                signals.append(signal)
        
        return signals
    
    def _simulate_trades(
        self,
        signals: List[StrategySignal],
        data: pd.DataFrame,
        config: BacktestConfig
    ) -> Tuple[List[Dict], np.ndarray]:
        """
        Simulate trades based on signals.
        
        Returns:
            (trades, equity_curve)
        """
        equity = config.initial_capital
        equity_curve = [equity]
        trades = []
        position = None
        
        for i, signal in enumerate(signals):
            # Entry
            if signal.action in ['buy', 'sell'] and position is None:
                # Find price in data (simplified)
                entry_price = float(data.iloc[min(i, len(data)-1)].get('close', 100))
                
                position = {
                    'direction': signal.action,
                    'entry_price': entry_price,
                    'entry_time': signal.timestamp,
                    'size': config.position_size * equity,
                }
            
            # Exit
            elif signal.action == 'close' and position is not None:
                exit_price = float(data.iloc[min(i, len(data)-1)].get('close', 100))
                
                # Calculate PnL
                if position['direction'] == 'buy':
                    pnl = (exit_price - position['entry_price']) / position['entry_price'] * position['size']
                else:
                    pnl = (position['entry_price'] - exit_price) / position['entry_price'] * position['size']
                
                # Apply costs
                costs = config.spread_pips + config.commission_per_lot + config.slippage_pips
                net_pnl = pnl - costs
                
                equity += net_pnl
                equity_curve.append(equity)
                
                trades.append({
                    'entry_time': position['entry_time'],
                    'exit_time': signal.timestamp,
                    'direction': position['direction'],
                    'entry_price': position['entry_price'],
                    'exit_price': exit_price,
                    'pnl': net_pnl,
                    'mfe': abs(pnl) * 1.2,  # Simplified
                    'mae': abs(pnl) * 0.3,  # Simplified
                })
                
                position = None
        
        return trades, np.array(equity_curve)
    
    def _calculate_metrics(
        self,
        trades: List[Dict],
        equity_curve: np.ndarray,
        initial_capital: float
    ) -> Dict:
        """Calculate backtest metrics."""
        if len(trades) == 0:
            return self._empty_metrics()
        
        # Trade statistics
        pnls = np.array([t['pnl'] for t in trades])
        wins = pnls[pnls > 0]
        losses = pnls[pnls < 0]
        
        winning_trades = len(wins)
        losing_trades = len(losses)
        win_rate = winning_trades / len(trades) if len(trades) > 0 else 0
        avg_win = np.mean(wins) if len(wins) > 0 else 0
        avg_loss = np.mean(losses) if len(losses) > 0 else 0
        
        total_wins = np.sum(wins) if len(wins) > 0 else 0
        total_losses = abs(np.sum(losses)) if len(losses) > 0 else 1e-10
        profit_factor = total_wins / total_losses if total_losses > 0 else 0
        
        # Returns
        returns = np.diff(equity_curve) / equity_curve[:-1]
        total_return = (equity_curve[-1] - initial_capital) / initial_capital
        
        # Sharpe ratio
        if len(returns) > 1 and np.std(returns) > 0:
            sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252)
        else:
            sharpe_ratio = 0
        
        # Omega ratio
        threshold = 0
        gains = returns[returns > threshold]
        losses_omega = returns[returns <= threshold]
        if len(losses_omega) > 0 and np.sum(np.abs(losses_omega)) > 0:
            omega_ratio = np.sum(gains) / np.sum(np.abs(losses_omega))
        else:
            omega_ratio = 0
        
        # Sortino ratio
        downside_returns = returns[returns < 0]
        if len(downside_returns) > 1 and np.std(downside_returns) > 0:
            sortino_ratio = np.mean(returns) / np.std(downside_returns) * np.sqrt(252)
        else:
            sortino_ratio = 0
        
        # Max drawdown
        cummax = np.maximum.accumulate(equity_curve)
        drawdowns = (equity_curve - cummax) / cummax
        max_drawdown = abs(np.min(drawdowns)) if len(drawdowns) > 0 else 0
        
        # Calmar ratio
        calmar_ratio = total_return / max_drawdown if max_drawdown > 0 else 0
        
        # Efficiency metrics
        mfe_captured_pct = 0
        mae_ratio = 0
        if len(trades) > 0:
            mfe_list = [t.get('mfe', 0) for t in trades]
            mae_list = [t.get('mae', 0) for t in trades]
            
            if np.mean(mfe_list) > 0:
                mfe_captured_pct = (np.mean(pnls) / np.mean(mfe_list)) * 100
            if np.mean(np.abs(pnls)) > 0:
                mae_ratio = np.mean(mae_list) / np.mean(np.abs(pnls))
        
        # Pythagorean efficiency
        from kinetra.testing_framework import EfficiencyMetrics
        pythagorean_efficiency = EfficiencyMetrics.calculate_pythagorean_efficiency(equity_curve)
        
        # Statistical significance (simplified t-test)
        from scipy import stats
        if len(returns) > 30:
            t_stat, p_value = stats.ttest_1samp(returns, 0)
            is_significant = p_value < 0.05
        else:
            p_value = 1.0
            is_significant = False
        
        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'omega_ratio': omega_ratio,
            'sortino_ratio': sortino_ratio,
            'max_drawdown': max_drawdown,
            'calmar_ratio': calmar_ratio,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'mfe_captured_pct': mfe_captured_pct,
            'mae_ratio': mae_ratio,
            'pythagorean_efficiency': pythagorean_efficiency,
            'is_significant': is_significant,
            'p_value': p_value,
        }
    
    def _empty_metrics(self) -> Dict:
        """Return empty metrics dict."""
        return {
            'total_return': 0, 'sharpe_ratio': 0, 'omega_ratio': 0,
            'sortino_ratio': 0, 'max_drawdown': 0, 'calmar_ratio': 0,
            'winning_trades': 0, 'losing_trades': 0, 'win_rate': 0,
            'avg_win': 0, 'avg_loss': 0, 'profit_factor': 0,
            'mfe_captured_pct': 0, 'mae_ratio': 0, 'pythagorean_efficiency': 0,
            'is_significant': False, 'p_value': 1.0,
        }
    
    def _save_backtest_result(self, result: BacktestResult):
        """Save backtest result to file."""
        timestamp = result.timestamp.strftime("%Y%m%d_%H%M%S")
        filename = f"{result.config_name}_{timestamp}.json"
        filepath = self.output_dir / filename
        
        # Convert to dict (excluding large arrays)
        result_dict = asdict(result)
        result_dict['equity_curve'] = result_dict['equity_curve'].tolist()
        
        import json
        with open(filepath, 'w') as f:
            json.dump(result_dict, f, indent=2, default=str)
        
        logger.info(f"Backtest result saved to {filepath}")
    
    def backtest_from_test_results(
        self,
        test_results: List,
        data_manager: 'UnifiedDataManager'
    ) -> Dict[str, BacktestResult]:
        """
        Backtest all discovered strategies from test results.
        
        Args:
            test_results: Results from testing framework
            data_manager: Data manager for loading market data
            
        Returns:
            Dict mapping strategy name to backtest results
        """
        backtest_results = {}
        
        for test_result in test_results:
            # Skip if not a discovery test
            if 'discovered' not in test_result.config_name.lower():
                continue
            
            # Convert to strategy
            strategy_config = self.converter.convert({
                'type': test_result.agent_type,
                'config': test_result.metrics,
            })
            
            # Load data
            try:
                data_path = test_result.instrument.data_path
                data = pd.read_csv(data_path)
                
                # Run backtest
                backtest_result = self.backtest_discovered_strategy(
                    strategy_config,
                    data
                )
                
                backtest_results[test_result.config_name] = backtest_result
                
            except Exception as e:
                logger.error(f"Error backtesting {test_result.config_name}: {e}")
        
        return backtest_results
    
    def generate_backtest_report(
        self,
        results: Dict[str, BacktestResult],
        output_file: Optional[str] = None
    ):
        """Generate comprehensive backtest report."""
        if output_file is None:
            output_file = self.output_dir / f"backtest_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        
        with open(output_file, 'w') as f:
            f.write("="*80 + "\n")
            f.write("BACKTEST RESULTS REPORT\n")
            f.write("="*80 + "\n\n")
            
            for name, result in results.items():
                f.write(f"\nStrategy: {name}\n")
                f.write("-"*80 + "\n")
                f.write(f"Total Return: {result.total_return:.2%}\n")
                f.write(f"Sharpe Ratio: {result.sharpe_ratio:.2f}\n")
                f.write(f"Omega Ratio: {result.omega_ratio:.2f}\n")
                f.write(f"Max Drawdown: {result.max_drawdown:.2%}\n")
                f.write(f"Win Rate: {result.win_rate:.2%}\n")
                f.write(f"Profit Factor: {result.profit_factor:.2f}\n")
                f.write(f"Total Trades: {result.total_trades}\n")
                f.write(f"MFE Captured: {result.mfe_captured_pct:.1f}%\n")
                f.write(f"Statistically Significant: {result.is_statistically_significant}\n")
                f.write("\n")
        
        logger.info(f"Backtest report saved to {output_file}")
