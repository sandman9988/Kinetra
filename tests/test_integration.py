"""
Integration tests for complete system.
"""

import pytest
import numpy as np
import pandas as pd
from kinetra.physics_engine import PhysicsEngine
from kinetra.risk_management import RiskManager
from kinetra.rl_agent import KinetraAgent


class TestIntegration:
    """Integration tests for full pipeline."""
    
    def test_physics_to_risk_pipeline(self):
        """Test physics engine -> risk management flow."""
        # Generate synthetic market data
        np.random.seed(42)
        prices = pd.Series(np.random.randn(200).cumsum() + 100)
        
        # Compute physics state
        physics = PhysicsEngine(lookback=20)
        state = physics.compute_physics_state(prices)
        
        # Calculate returns
        returns = prices.pct_change().dropna()
        
        # Risk management
        risk_mgr = RiskManager()
        ror = risk_mgr.calculate_risk_of_ruin(
            current_equity=100000,
            ruin_level=50000,
            returns=returns
        )
        
        # Should complete without errors
        assert 0.0 <= ror <= 1.0
        assert len(state) == len(prices)
    
    def test_end_to_end_decision(self):
        """Test complete decision pipeline."""
        # Generate data
        np.random.seed(42)
        prices = pd.Series(np.random.randn(100).cumsum() + 100)
        
        # Physics
        physics = PhysicsEngine()
        state = physics.compute_physics_state(prices)
        
        # Risk
        risk_mgr = RiskManager()
        returns = prices.pct_change().dropna()
        ror = risk_mgr.calculate_risk_of_ruin(100000, 50000, returns)
        
        # Agent (placeholder) - use state_dim=10 to match test state
        agent = KinetraAgent(state_dim=10)
        action = agent.select_action(np.zeros(10))
        
        # Should complete pipeline
        assert state is not None
        assert 0.0 <= ror <= 1.0
        assert action in [0, 1, 2, 3]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
