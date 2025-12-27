"""
Unit tests for Risk Management
"""

import pytest
import numpy as np
import pandas as pd
from kinetra.risk_management import RiskManager, calculate_risk_of_ruin


class TestRiskManagement:
    """Test suite for risk management."""
    
    def test_risk_of_ruin_calculation(self):
        """Test non-linear RoR calculation."""
        # Create profitable returns
        returns = pd.Series(np.random.randn(100) * 0.01 + 0.001)
        
        manager = RiskManager()
        ror = manager.calculate_risk_of_ruin(
            current_equity=100000,
            ruin_level=50000,
            returns=returns
        )
        
        # RoR should be in [0, 1]
        assert 0.0 <= ror <= 1.0
        
        # Positive expected return -> lower RoR
        assert ror < 0.5
    
    def test_agent_health_score(self):
        """Test agent health score calculation."""
        manager = RiskManager()
        
        chs = manager.calculate_agent_health_score(
            win_rate=0.60,
            avg_win_loss_ratio=2.0,
            omega_ratio=3.0
        )
        
        # CHS should be in [0, 1]
        assert 0.0 <= chs <= 1.0
        
        # Good metrics should give high score
        assert chs > 0.5
    
    def test_risk_health_score(self):
        """Test risk health score calculation."""
        manager = RiskManager()
        
        chs = manager.calculate_risk_health_score(
            risk_of_ruin=0.05,
            max_drawdown=0.10,
            volatility=0.15
        )
        
        # CHS should be in [0, 1]
        assert 0.0 <= chs <= 1.0
    
    def test_composite_health_score(self):
        """Test composite health score."""
        manager = RiskManager()
        
        chs = manager.composite_health_score(
            chs_agents=0.70,
            chs_risk=0.80,
            chs_class=0.75
        )
        
        # CHS should be weighted average
        assert 0.0 <= chs <= 1.0
        assert 0.70 <= chs <= 0.80
    
    def test_position_sizing_with_gates(self):
        """Test position sizing respects risk gates."""
        manager = RiskManager(max_risk_of_ruin=0.10, min_health_score=0.55)
        
        # Test normal conditions
        position = manager.calculate_position_size(
            equity=100000,
            risk_of_ruin=0.05,
            health_score=0.80
        )
        assert position > 0
        
        # Test RoR gate
        position = manager.calculate_position_size(
            equity=100000,
            risk_of_ruin=0.15,  # Too high
            health_score=0.80
        )
        assert position == 0  # Circuit breaker
        
        # Test health gate
        position = manager.calculate_position_size(
            equity=100000,
            risk_of_ruin=0.05,
            health_score=0.40  # Too low
        )
        assert position == 0  # Circuit breaker
    
    def test_risk_gates(self):
        """Test risk gate checking."""
        returns = pd.Series(np.random.randn(100) * 0.01 + 0.001)
        manager = RiskManager()
        
        # Good conditions
        passed, msg = manager.check_risk_gates(
            current_equity=100000,
            ruin_level=50000,
            returns=returns,
            health_score=0.80
        )
        assert passed
        assert "passed" in msg.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
