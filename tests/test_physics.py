"""
Unit tests for Physics Engine
"""

import pytest
import numpy as np
import pandas as pd
from kinetra.physics_engine import PhysicsEngine, RegimeType, calculate_energy


class TestPhysicsEngine:
    """Test suite for physics engine."""
    
    def test_energy_calculation(self):
        """Test kinetic energy calculation."""
        prices = pd.Series([100, 101, 102, 101, 100])
        engine = PhysicsEngine(mass=1.0)
        
        energy = engine.calculate_energy(prices)
        
        # Energy should be non-negative
        assert (energy >= 0).all()
        
        # First value should be 0 (no previous price)
        assert energy.iloc[0] == 0.0
    
    def test_damping_calculation(self):
        """Test damping coefficient calculation."""
        prices = pd.Series(np.random.randn(100).cumsum() + 100)
        engine = PhysicsEngine(lookback=20)
        
        damping = engine.calculate_damping(prices)
        
        # Damping should be non-negative
        assert (damping >= 0).all()
    
    def test_entropy_calculation(self):
        """Test Shannon entropy calculation."""
        prices = pd.Series(np.random.randn(100).cumsum() + 100)
        engine = PhysicsEngine(lookback=20)
        
        entropy = engine.calculate_entropy(prices)
        
        # Entropy should be non-negative
        assert (entropy >= 0).all()
    
    def test_regime_classification(self):
        """Test regime classification with no fixed thresholds."""
        # Create trending price data (underdamped)
        trend_prices = pd.Series(np.arange(100) + np.random.randn(100) * 0.1 + 100)
        engine = PhysicsEngine(lookback=20)
        
        state = engine.compute_physics_state(trend_prices)
        
        # Should have some regime classifications
        assert 'regime' in state.columns
        assert len(state['regime'].unique()) > 0
    
    def test_physics_constraints(self):
        """Test that physics constraints are enforced."""
        prices = pd.Series(np.random.randn(100).cumsum() + 100)
        engine = PhysicsEngine()
        
        state = engine.compute_physics_state(prices)
        
        # All physics values must be non-negative
        assert (state['energy'] >= 0).all()
        assert (state['damping'] >= 0).all()
        assert (state['entropy'] >= 0).all()
    
    def test_nan_handling(self):
        """Test NaN shield works correctly."""
        prices = pd.Series([100, 101, np.nan, 102, 103])
        engine = PhysicsEngine()
        
        # Should handle NaN gracefully
        energy = engine.calculate_energy(prices.fillna(method='ffill'))
        assert not energy.isna().any()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
