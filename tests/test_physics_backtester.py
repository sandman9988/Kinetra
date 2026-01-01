"""
Tests for Physics-Based Backtester

Tests the thermodynamics/physics/kinematics trading strategies
using the backtesting.py framework.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from kinetra.physics_backtester import (
    PhysicsBacktestRunner,
    BasePhysicsStrategy,
    EnergyMomentumStrategy,
    DampingReversionStrategy,
    EntropyVolatilityStrategy,
    AccelerationTrendStrategy,
    MultiPhysicsStrategy,
    ThermodynamicEquilibriumStrategy,
    list_strategies,
    get_strategy,
    calculate_physics_metrics,
    compute_kinetic_energy,
    compute_damping,
    compute_entropy,
    compute_velocity,
    compute_acceleration,
    compute_atr,
)

from kinetra.data_utils import (
    validate_ohlcv,
    preprocess_mt5_data,
    get_data_summary,
    split_data,
    create_walk_forward_windows,
)


# =============================================================================
# FIXTURES - Using real-like OHLCV structure (but with minimal test data)
# =============================================================================

@pytest.fixture
def minimal_ohlcv_data():
    """Create minimal OHLCV data for unit tests (100 bars)."""
    np.random.seed(42)
    n = 100

    # Create price series with some trend
    close = 100.0 + np.cumsum(np.random.randn(n) * 0.5)
    close = np.maximum(close, 10.0)  # Ensure positive

    # Create OHLC from close
    high = close * (1 + np.abs(np.random.randn(n) * 0.01))
    low = close * (1 - np.abs(np.random.randn(n) * 0.01))
    open_prices = np.roll(close, 1)
    open_prices[0] = close[0]

    # Ensure OHLC validity
    high = np.maximum(high, np.maximum(open_prices, close))
    low = np.minimum(low, np.minimum(open_prices, close))

    df = pd.DataFrame({
        'Open': open_prices,
        'High': high,
        'Low': low,
        'Close': close,
        'Volume': np.random.randint(100000, 1000000, n)
    })

    df.index = pd.date_range(start='2023-01-01', periods=n, freq='h')
    return df


@pytest.fixture
def extended_ohlcv_data():
    """Create extended OHLCV data for backtesting (500 bars)."""
    np.random.seed(42)
    n = 500

    # Create trending price series
    trend = np.linspace(0, 20, n)
    noise = np.cumsum(np.random.randn(n) * 0.5)
    close = 100.0 + trend + noise
    close = np.maximum(close, 10.0)

    high = close * (1 + np.abs(np.random.randn(n) * 0.01))
    low = close * (1 - np.abs(np.random.randn(n) * 0.01))
    open_prices = np.roll(close, 1)
    open_prices[0] = close[0]

    high = np.maximum(high, np.maximum(open_prices, close))
    low = np.minimum(low, np.minimum(open_prices, close))

    df = pd.DataFrame({
        'Open': open_prices,
        'High': high,
        'Low': low,
        'Close': close,
        'Volume': np.random.randint(100000, 1000000, n)
    })

    df.index = pd.date_range(start='2023-01-01', periods=n, freq='h')
    return df


# =============================================================================
# PHYSICS INDICATOR TESTS
# =============================================================================

class TestPhysicsIndicators:
    """Test physics indicator calculations."""

    def test_kinetic_energy_non_negative(self, minimal_ohlcv_data):
        """Kinetic energy must be non-negative (E = 0.5mv² >= 0)."""
        energy = compute_kinetic_energy(minimal_ohlcv_data['Close'].values)
        assert (energy >= 0).all(), "Kinetic energy cannot be negative"

    def test_kinetic_energy_formula(self, minimal_ohlcv_data):
        """Verify E = 0.5 * m * v² formula."""
        close = minimal_ohlcv_data['Close']
        mass = 1.0

        energy = compute_kinetic_energy(close.values, mass)
        velocity = close.diff().fillna(0.0)
        expected = 0.5 * mass * velocity ** 2

        # Check non-NaN values match
        np.testing.assert_array_almost_equal(
            energy[1:],
            expected.values[1:],
            decimal=10
        )

    def test_damping_non_negative(self, minimal_ohlcv_data):
        """Damping coefficient must be non-negative."""
        damping = compute_damping(minimal_ohlcv_data['Close'].values)
        assert (damping >= 0).all(), "Damping cannot be negative"

    def test_entropy_non_negative(self, minimal_ohlcv_data):
        """Shannon entropy must be non-negative."""
        entropy = compute_entropy(minimal_ohlcv_data['Close'].values)
        assert (entropy >= 0).all(), "Entropy cannot be negative"

    def test_velocity_calculation(self, minimal_ohlcv_data):
        """Velocity is first derivative of price."""
        close = minimal_ohlcv_data['Close']
        velocity = compute_velocity(close.values)
        expected = close.diff().fillna(0.0).values
        np.testing.assert_array_almost_equal(velocity, expected, decimal=10)

    def test_acceleration_calculation(self, minimal_ohlcv_data):
        """Acceleration is second derivative of price."""
        close = minimal_ohlcv_data['Close']
        acceleration = compute_acceleration(close.values)
        expected = close.diff().diff().fillna(0.0).values
        np.testing.assert_array_almost_equal(acceleration, expected, decimal=10)

    def test_atr_positive(self, minimal_ohlcv_data):
        """ATR must be positive."""
        df = minimal_ohlcv_data
        atr = compute_atr(df['High'].values, df['Low'].values, df['Close'].values)
        # After warmup period - use valid (non-NaN from fillna)
        assert (atr > 0).all(), "ATR must be positive"


# =============================================================================
# STRATEGY TESTS
# =============================================================================

class TestStrategyRegistry:
    """Test strategy registration and access."""

    def test_list_strategies(self):
        """All strategies should be listed."""
        strategies = list_strategies()
        expected = [
            'energy_momentum',
            'damping_reversion',
            'entropy_volatility',
            'acceleration_trend',
            'multi_physics',
            'thermodynamic'
        ]
        assert sorted(strategies) == sorted(expected)

    def test_get_strategy_valid(self):
        """Get valid strategy by name."""
        strategy = get_strategy('multi_physics')
        assert strategy == MultiPhysicsStrategy

    def test_get_strategy_invalid(self):
        """Invalid strategy name should raise."""
        with pytest.raises(ValueError):
            get_strategy('invalid_strategy')


class TestStrategyExecution:
    """Test strategy execution."""

    def test_energy_momentum_runs(self, extended_ohlcv_data):
        """Energy momentum strategy should run without errors."""
        runner = PhysicsBacktestRunner(cash=10000)
        results = runner.run(extended_ohlcv_data, strategy='energy_momentum')
        assert 'return_pct' in results
        assert 'total_trades' in results

    def test_damping_reversion_runs(self, extended_ohlcv_data):
        """Damping reversion strategy should run without errors."""
        runner = PhysicsBacktestRunner(cash=10000)
        results = runner.run(extended_ohlcv_data, strategy='damping_reversion')
        assert 'return_pct' in results

    def test_entropy_volatility_runs(self, extended_ohlcv_data):
        """Entropy volatility strategy should run without errors."""
        runner = PhysicsBacktestRunner(cash=10000)
        results = runner.run(extended_ohlcv_data, strategy='entropy_volatility')
        assert 'return_pct' in results

    def test_acceleration_trend_runs(self, extended_ohlcv_data):
        """Acceleration trend strategy should run without errors."""
        runner = PhysicsBacktestRunner(cash=10000)
        results = runner.run(extended_ohlcv_data, strategy='acceleration_trend')
        assert 'return_pct' in results

    def test_multi_physics_runs(self, extended_ohlcv_data):
        """Multi-physics strategy should run without errors."""
        runner = PhysicsBacktestRunner(cash=10000)
        results = runner.run(extended_ohlcv_data, strategy='multi_physics')
        assert 'return_pct' in results

    def test_thermodynamic_runs(self, extended_ohlcv_data):
        """Thermodynamic strategy should run without errors."""
        runner = PhysicsBacktestRunner(cash=10000)
        results = runner.run(extended_ohlcv_data, strategy='thermodynamic')
        assert 'return_pct' in results


class TestBacktestResults:
    """Test backtest result structure and validity."""

    def test_result_keys(self, extended_ohlcv_data):
        """Results should have all required keys."""
        runner = PhysicsBacktestRunner()
        results = runner.run(extended_ohlcv_data, strategy='multi_physics')

        required_keys = [
            'strategy', 'return_pct', 'buy_hold_return_pct',
            'sharpe_ratio', 'sortino_ratio', 'max_drawdown_pct',
            'win_rate', 'profit_factor', 'total_trades',
            'avg_trade_pct', 'exposure_pct', 'final_equity', 'sqn'
        ]

        for key in required_keys:
            assert key in results, f"Missing key: {key}"

    def test_max_drawdown_non_positive(self, extended_ohlcv_data):
        """Max drawdown should be non-positive (it's a loss)."""
        runner = PhysicsBacktestRunner()
        results = runner.run(extended_ohlcv_data, strategy='multi_physics')
        assert results['max_drawdown_pct'] <= 0

    def test_exposure_bounded(self, extended_ohlcv_data):
        """Exposure should be between 0 and 100."""
        runner = PhysicsBacktestRunner()
        results = runner.run(extended_ohlcv_data, strategy='multi_physics')
        assert 0 <= results['exposure_pct'] <= 100

    def test_final_equity_positive(self, extended_ohlcv_data):
        """Final equity should be positive."""
        runner = PhysicsBacktestRunner(cash=10000)
        results = runner.run(extended_ohlcv_data, strategy='multi_physics')
        assert results['final_equity'] > 0


class TestStrategyComparison:
    """Test strategy comparison functionality."""

    def test_compare_returns_dataframe(self, extended_ohlcv_data):
        """Comparison should return DataFrame."""
        runner = PhysicsBacktestRunner()
        comparison = runner.compare_strategies(
            extended_ohlcv_data,
            strategies=['energy_momentum', 'damping_reversion']
        )
        assert isinstance(comparison, pd.DataFrame)
        assert len(comparison) == 2

    def test_compare_all_strategies(self, extended_ohlcv_data):
        """Compare all strategies should work."""
        runner = PhysicsBacktestRunner()
        comparison = runner.compare_strategies(extended_ohlcv_data)
        assert len(comparison) == len(list_strategies())


# =============================================================================
# DATA UTILITY TESTS
# =============================================================================

class TestDataValidation:
    """Test OHLCV data validation."""

    def test_valid_data_passes(self, minimal_ohlcv_data):
        """Valid OHLCV data should pass validation."""
        is_valid, msg = validate_ohlcv(minimal_ohlcv_data)
        assert is_valid, f"Validation failed: {msg}"

    def test_missing_column_fails(self, minimal_ohlcv_data):
        """Missing required column should fail."""
        df = minimal_ohlcv_data.drop('Close', axis=1)
        is_valid, msg = validate_ohlcv(df)
        assert not is_valid
        assert 'Close' in msg

    def test_negative_prices_fail(self, minimal_ohlcv_data):
        """Negative prices should fail validation."""
        df = minimal_ohlcv_data.copy()
        df.loc[df.index[0], 'Close'] = -1.0
        is_valid, msg = validate_ohlcv(df)
        assert not is_valid

    def test_invalid_high_low_fails(self, minimal_ohlcv_data):
        """High < Low should fail validation."""
        df = minimal_ohlcv_data.copy()
        # Make High less than Low
        df.loc[df.index[0], 'High'] = df.loc[df.index[0], 'Low'] - 1
        is_valid, msg = validate_ohlcv(df)
        assert not is_valid


class TestDataSplitting:
    """Test data splitting functions."""

    def test_split_maintains_order(self, extended_ohlcv_data):
        """Split should maintain chronological order."""
        train, val, test = split_data(extended_ohlcv_data)

        # Train ends before validation starts
        assert train.index[-1] < val.index[0]
        # Validation ends before test starts
        assert val.index[-1] < test.index[0]

    def test_split_proportions(self, extended_ohlcv_data):
        """Split should roughly match proportions."""
        train, val, test = split_data(extended_ohlcv_data, 0.7, 0.15)
        total = len(extended_ohlcv_data)

        assert abs(len(train) / total - 0.7) < 0.02
        assert abs(len(val) / total - 0.15) < 0.02

    def test_walk_forward_windows(self, extended_ohlcv_data):
        """Walk-forward windows should be created correctly."""
        windows = create_walk_forward_windows(
            extended_ohlcv_data,
            train_bars=100,
            test_bars=50,
            step_bars=25
        )

        assert len(windows) > 0

        for train_df, test_df in windows:
            assert len(train_df) == 100
            assert len(test_df) == 50
            # Train ends before test starts
            assert train_df.index[-1] < test_df.index[0]


class TestDataSummary:
    """Test data summary function."""

    def test_summary_keys(self, minimal_ohlcv_data):
        """Summary should have all required keys."""
        summary = get_data_summary(minimal_ohlcv_data)

        required_keys = [
            'start_date', 'end_date', 'total_bars',
            'price_start', 'price_end', 'price_high', 'price_low',
            'total_return_pct', 'annualized_volatility'
        ]

        for key in required_keys:
            assert key in summary, f"Missing key: {key}"

    def test_summary_values_correct(self, minimal_ohlcv_data):
        """Summary values should be calculated correctly."""
        summary = get_data_summary(minimal_ohlcv_data)

        assert summary['total_bars'] == len(minimal_ohlcv_data)
        assert summary['price_high'] == minimal_ohlcv_data['High'].max()
        assert summary['price_low'] == minimal_ohlcv_data['Low'].min()


# =============================================================================
# PHYSICS METRICS TESTS
# =============================================================================

class TestPhysicsMetrics:
    """Test physics-specific metrics calculation."""

    def test_physics_metrics_keys(self, extended_ohlcv_data):
        """Physics metrics should have all required keys."""
        runner = PhysicsBacktestRunner()
        results = runner.run(extended_ohlcv_data, strategy='multi_physics')
        metrics = calculate_physics_metrics(extended_ohlcv_data, results)

        required_keys = [
            'total_energy', 'energy_captured_pct',
            'avg_entropy', 'avg_damping',
            'regime_underdamped_pct', 'regime_critical_pct', 'regime_overdamped_pct'
        ]

        for key in required_keys:
            assert key in metrics, f"Missing key: {key}"

    def test_energy_non_negative(self, extended_ohlcv_data):
        """Total energy should be non-negative."""
        runner = PhysicsBacktestRunner()
        results = runner.run(extended_ohlcv_data, strategy='multi_physics')
        metrics = calculate_physics_metrics(extended_ohlcv_data, results)

        assert metrics['total_energy'] >= 0

    def test_regime_percentages_sum(self, extended_ohlcv_data):
        """Regime percentages should sum to approximately 100."""
        runner = PhysicsBacktestRunner()
        results = runner.run(extended_ohlcv_data, strategy='multi_physics')
        metrics = calculate_physics_metrics(extended_ohlcv_data, results)

        total = (metrics['regime_underdamped_pct'] +
                 metrics['regime_critical_pct'] +
                 metrics['regime_overdamped_pct'] +
                 metrics.get('regime_laminar_pct', 0) +
                 metrics.get('regime_breakout_pct', 0))

        assert abs(total - 100) < 1  # Within 1% due to rounding


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestIntegration:
    """End-to-end integration tests."""

    def test_full_backtest_pipeline(self, extended_ohlcv_data):
        """Full backtest pipeline should work end-to-end."""
        # Validate data
        is_valid, _ = validate_ohlcv(extended_ohlcv_data)
        assert is_valid

        # Get summary
        summary = get_data_summary(extended_ohlcv_data)
        assert summary['total_bars'] > 0

        # Run backtest
        runner = PhysicsBacktestRunner()
        results = runner.run(extended_ohlcv_data, strategy='multi_physics')

        # Calculate physics metrics
        metrics = calculate_physics_metrics(extended_ohlcv_data, results)

        # All should complete without error
        assert results['final_equity'] > 0
        assert metrics['total_energy'] >= 0

    def test_walk_forward_pipeline(self, extended_ohlcv_data):
        """Walk-forward analysis pipeline should work."""
        runner = PhysicsBacktestRunner()

        wf_results = runner.monte_carlo(
            extended_ohlcv_data,
            strategy='energy_momentum',
            n_runs=5,
            sample_pct=0.5
        )

        assert len(wf_results) > 0
        assert 'return_pct' in wf_results.columns


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
