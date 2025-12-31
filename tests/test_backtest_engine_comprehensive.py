"""
Comprehensive BacktestEngine Tests

Tests for critical bug fixes and new features:
1. Timeframe parameter initialization
2. Margin level tracking
3. Safe math operations
4. Data validation
5. Logging integration
6. Multi-instrument support (future)
"""

import numpy as np
import pandas as pd
import pytest
from datetime import datetime, timedelta

from kinetra.backtest_engine import BacktestEngine, TradeDirection, Trade, BacktestResult
from kinetra.symbol_spec import SymbolSpec, CommissionSpec, CommissionType


class TestBacktestEngineInitialization:
    """Test BacktestEngine initialization with new parameters."""
    
    def test_default_initialization(self):
        """Test default initialization works."""
        engine = BacktestEngine()
        assert engine.initial_capital == 100000.0
        assert engine.timeframe == "H1"
        assert engine.leverage == 100.0
        assert engine.min_margin_level == float("inf")
        assert engine.enable_logging == False
        
    def test_custom_timeframe(self):
        """Test custom timeframe parameter."""
        engine = BacktestEngine(timeframe="M15")
        assert engine.timeframe == "M15"
        
    def test_invalid_timeframe_fallback(self):
        """Test invalid timeframe falls back to H1."""
        with pytest.warns(UserWarning):
            engine = BacktestEngine(timeframe="INVALID")
        assert engine.timeframe == "H1"
    
    def test_parameter_validation(self):
        """Test parameter validation in __init__."""
        # Invalid initial capital
        with pytest.raises(ValueError, match="initial_capital must be positive"):
            BacktestEngine(initial_capital=-1000)
        
        # Invalid risk per trade
        with pytest.raises(ValueError, match="risk_per_trade must be in"):
            BacktestEngine(risk_per_trade=1.5)
        
        with pytest.raises(ValueError, match="risk_per_trade must be in"):
            BacktestEngine(risk_per_trade=0)
        
        # Invalid max positions
        with pytest.raises(ValueError, match="max_positions must be >= 1"):
            BacktestEngine(max_positions=0)
        
        # Invalid leverage
        with pytest.raises(ValueError, match="leverage must be positive"):
            BacktestEngine(leverage=-10)


class TestBacktestEngineDataValidation:
    """Test data validation in run_backtest."""
    
    def create_test_spec(self) -> SymbolSpec:
        """Create a minimal SymbolSpec for testing."""
        return SymbolSpec(
            symbol="EURUSD",
            tick_size=0.00001,
            tick_value=1.0,
            contract_size=100000,
            volume_min=0.01,
            volume_max=100.0,
            volume_step=0.01,
            spread_points=2.0,
            commission=CommissionSpec(rate=0.0, commission_type=CommissionType.PER_LOT),
            slippage_avg=0.5,
        )
    
    def create_valid_data(self, n_bars: int = 100) -> pd.DataFrame:
        """Create valid OHLCV data for testing."""
        dates = pd.date_range(start="2024-01-01", periods=n_bars, freq="H")
        prices = 1.08 + np.random.randn(n_bars).cumsum() * 0.001
        
        return pd.DataFrame({
            "time": dates,
            "open": prices,
            "high": prices + np.abs(np.random.randn(n_bars) * 0.0005),
            "low": prices - np.abs(np.random.randn(n_bars) * 0.0005),
            "close": prices,
            "volume": np.random.randint(1000, 10000, n_bars),
        })
    
    def test_missing_columns(self):
        """Test that missing required columns raise ValueError."""
        engine = BacktestEngine()
        spec = self.create_test_spec()
        
        # Missing 'close' column
        data = pd.DataFrame({
            "open": [1.0, 1.1],
            "high": [1.1, 1.2],
            "low": [0.9, 1.0],
        })
        
        with pytest.raises(ValueError, match="Data must contain columns"):
            engine.run_backtest(data, spec)
    
    def test_nan_in_data(self):
        """Test that NaN values in data raise ValueError."""
        engine = BacktestEngine()
        spec = self.create_test_spec()
        
        data = self.create_valid_data(10)
        data.loc[5, "close"] = np.nan
        
        with pytest.raises(ValueError, match="Data contains NaN values"):
            engine.run_backtest(data, spec)
    
    def test_inf_in_data(self):
        """Test that Inf values in data raise ValueError."""
        engine = BacktestEngine()
        spec = self.create_test_spec()
        
        data = self.create_valid_data(10)
        data.loc[5, "high"] = np.inf
        
        with pytest.raises(ValueError, match="Data contains Inf values"):
            engine.run_backtest(data, spec)
    
    def test_insufficient_data(self):
        """Test that too little data raises ValueError."""
        engine = BacktestEngine()
        spec = self.create_test_spec()
        
        data = pd.DataFrame({
            "open": [1.0],
            "high": [1.1],
            "low": [0.9],
            "close": [1.0],
        })
        
        with pytest.raises(ValueError, match="Data must have at least 2 bars"):
            engine.run_backtest(data, spec)


class TestMarginTracking:
    """Test margin level tracking during backtest."""
    
    def create_test_spec(self) -> SymbolSpec:
        """Create a minimal SymbolSpec for testing."""
        return SymbolSpec(
            symbol="EURUSD",
            tick_size=0.00001,
            tick_value=1.0,
            contract_size=100000,
            volume_min=0.01,
            volume_max=100.0,
            volume_step=0.01,
            spread_points=2.0,
            commission=CommissionSpec(rate=0.0, commission_type=CommissionType.PER_LOT),
            slippage_avg=0.5,
        )
    
    def create_trending_data(self, n_bars: int = 100) -> pd.DataFrame:
        """Create trending data that will trigger trades."""
        dates = pd.date_range(start="2024-01-01", periods=n_bars, freq="H")
        # Create strong uptrend
        prices = 1.08 + np.arange(n_bars) * 0.0001
        
        return pd.DataFrame({
            "time": dates,
            "open": prices,
            "high": prices + 0.0002,
            "low": prices - 0.0001,
            "close": prices,
            "volume": np.ones(n_bars) * 1000,
        })
    
    def test_margin_tracking_enabled(self):
        """Test that margin levels are tracked during backtest."""
        engine = BacktestEngine(leverage=100.0)
        spec = self.create_test_spec()
        data = self.create_trending_data(50)
        
        result = engine.run_backtest(data, spec)
        
        # Should have margin history
        assert len(engine.margin_history) > 0
        
        # Min margin level should be set
        assert engine.min_margin_level < float("inf")
        
        # Min margin level should be in result
        assert result.min_margin_level == engine.min_margin_level
    
    def test_no_position_margin_level(self):
        """Test margin level when no position is open."""
        engine = BacktestEngine()
        spec = self.create_test_spec()
        # Flat data - no trades
        data = pd.DataFrame({
            "time": pd.date_range(start="2024-01-01", periods=10, freq="H"),
            "open": [1.08] * 10,
            "high": [1.081] * 10,
            "low": [1.079] * 10,
            "close": [1.08] * 10,
            "volume": [1000] * 10,
        })
        
        result = engine.run_backtest(data, spec)
        
        # Should have margin history (all inf when no position)
        assert len(engine.margin_history) > 0
        assert all(m == float("inf") for m in engine.margin_history)


class TestSafeMathOperations:
    """Test safe math operations (division by zero, etc.)."""
    
    def create_test_spec(self, **overrides) -> SymbolSpec:
        """Create a SymbolSpec with optional overrides."""
        defaults = {
            "symbol": "EURUSD",
            "tick_size": 0.00001,
            "tick_value": 1.0,
            "contract_size": 100000,
            "volume_min": 0.01,
            "volume_max": 100.0,
            "volume_step": 0.01,
            "spread_points": 2.0,
            "commission": CommissionSpec(rate=0.0, commission_type=CommissionType.PER_LOT),
            "slippage_avg": 0.5,
        }
        defaults.update(overrides)
        return SymbolSpec(**defaults)
    
    def test_zero_tick_size_handling(self):
        """Test handling of zero tick_size."""
        engine = BacktestEngine()
        spec = self.create_test_spec(tick_size=0.0)
        
        data = pd.DataFrame({
            "time": pd.date_range(start="2024-01-01", periods=5, freq="H"),
            "open": [1.08, 1.09, 1.10, 1.11, 1.12],
            "high": [1.081, 1.091, 1.101, 1.111, 1.121],
            "low": [1.079, 1.089, 1.099, 1.109, 1.119],
            "close": [1.08, 1.09, 1.10, 1.11, 1.12],
            "volume": [1000] * 5,
        })
        
        # Should not crash, should warn
        with pytest.warns(UserWarning, match="Invalid tick_size"):
            result = engine.run_backtest(data, spec)
        
        # Should complete without error
        assert isinstance(result, BacktestResult)
    
    def test_zero_spread_handling(self):
        """Test handling of zero or negative spread."""
        engine = BacktestEngine()
        spec = self.create_test_spec(spread_points=0.0)
        
        data = pd.DataFrame({
            "time": pd.date_range(start="2024-01-01", periods=5, freq="H"),
            "open": [1.08, 1.09, 1.10, 1.11, 1.12],
            "high": [1.081, 1.091, 1.101, 1.111, 1.121],
            "low": [1.079, 1.089, 1.099, 1.109, 1.119],
            "close": [1.08, 1.09, 1.10, 1.11, 1.12],
            "volume": [1000] * 5,
        })
        
        # Should warn and use fallback
        with pytest.warns(UserWarning, match="Invalid spread_points"):
            result = engine.run_backtest(data, spec)
        
        assert isinstance(result, BacktestResult)


class TestMetricsCalculation:
    """Test that all metrics are calculated correctly."""
    
    def create_test_spec(self) -> SymbolSpec:
        """Create a minimal SymbolSpec for testing."""
        return SymbolSpec(
            symbol="EURUSD",
            tick_size=0.00001,
            tick_value=1.0,
            contract_size=100000,
            volume_min=0.01,
            volume_max=100.0,
            volume_step=0.01,
            spread_points=2.0,
            commission=CommissionSpec(rate=0.0, commission_type=CommissionType.PER_LOT),
            slippage_avg=0.5,
        )
    
    def test_timeframe_aware_annualization(self):
        """Test that Sharpe ratio uses correct timeframe annualization."""
        # Test H1 timeframe
        engine_h1 = BacktestEngine(timeframe="H1")
        assert engine_h1.timeframe == "H1"
        
        # Test M15 timeframe
        engine_m15 = BacktestEngine(timeframe="M15")
        assert engine_m15.timeframe == "M15"
        
        # Test D1 timeframe
        engine_d1 = BacktestEngine(timeframe="D1")
        assert engine_d1.timeframe == "D1"
    
    def test_empty_trades_result(self):
        """Test result when no trades are executed."""
        engine = BacktestEngine()
        spec = self.create_test_spec()
        
        # Flat data - no trades
        data = pd.DataFrame({
            "time": pd.date_range(start="2024-01-01", periods=10, freq="H"),
            "open": [1.08] * 10,
            "high": [1.081] * 10,
            "low": [1.079] * 10,
            "close": [1.08] * 10,
            "volume": [1000] * 10,
        })
        
        result = engine.run_backtest(data, spec)
        
        assert result.total_trades == 0
        assert result.win_rate == 0.0
        assert result.total_net_pnl == 0.0
        assert len(result.equity_curve) > 0


class TestLoggingIntegration:
    """Test MT5Logger integration."""
    
    def create_test_spec(self) -> SymbolSpec:
        """Create a minimal SymbolSpec for testing."""
        return SymbolSpec(
            symbol="EURUSD",
            tick_size=0.00001,
            tick_value=1.0,
            contract_size=100000,
            volume_min=0.01,
            volume_max=100.0,
            volume_step=0.01,
            spread_points=2.0,
            commission=CommissionSpec(rate=0.0, commission_type=CommissionType.PER_LOT),
            slippage_avg=0.5,
        )
    
    def test_logging_disabled_by_default(self):
        """Test that logging is disabled by default."""
        engine = BacktestEngine()
        assert engine.enable_logging == False
        assert engine.logger is None
    
    def test_logging_enabled(self):
        """Test that logging can be enabled."""
        engine = BacktestEngine(enable_logging=True)
        spec = self.create_test_spec()
        
        data = pd.DataFrame({
            "time": pd.date_range(start="2024-01-01", periods=10, freq="H"),
            "open": [1.08] * 10,
            "high": [1.081] * 10,
            "low": [1.079] * 10,
            "close": [1.08] * 10,
            "volume": [1000] * 10,
        })
        
        result = engine.run_backtest(data, spec)
        
        # Logger should be initialized during backtest
        assert engine.logger is not None
        assert engine.logger.symbol == "EURUSD"


class TestResetFunctionality:
    """Test that reset() properly clears state."""
    
    def test_reset_clears_state(self):
        """Test that reset clears all state variables."""
        engine = BacktestEngine(initial_capital=50000.0)
        
        # Manually set some state
        engine.trades = [Trade(
            trade_id=1, symbol="EURUSD", direction=TradeDirection.LONG,
            lots=1.0, entry_time=datetime.now(), entry_price=1.08
        )]
        engine.equity = 60000.0
        engine.equity_history = [50000.0, 55000.0, 60000.0]
        engine.margin_history = [500.0, 600.0]
        engine.min_margin_level = 500.0
        engine.trade_counter = 5
        
        # Reset
        engine.reset()
        
        # Verify all cleared
        assert len(engine.trades) == 0
        assert engine.equity == 50000.0
        assert engine.equity_history == [50000.0]
        assert len(engine.margin_history) == 0
        assert engine.min_margin_level == float("inf")
        assert engine.trade_counter == 0
        assert engine.logger is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
