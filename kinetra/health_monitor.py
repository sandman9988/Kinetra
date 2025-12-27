"""
Health Monitoring and Circuit Breakers
"""

from typing import Dict, Tuple
import numpy as np


class HealthMonitor:
    """
    Real-time health monitoring with circuit breakers.
    """
    
    def __init__(self, circuit_breaker_threshold: float = 0.55):
        """Initialize health monitor."""
        self.circuit_breaker_threshold = circuit_breaker_threshold
        self.health_history = []
    
    def check_health(self, chs: float) -> Tuple[bool, str]:
        """Check if system is healthy."""
        self.health_history.append(chs)
        
        if chs < self.circuit_breaker_threshold:
            return False, f"Circuit breaker triggered: CHS {chs:.2f} < {self.circuit_breaker_threshold:.2f}"
        
        return True, "System healthy"
    
    def get_health_metrics(self) -> Dict:
        """Get current health metrics."""
        if not self.health_history:
            return {"current_chs": 0.0, "avg_chs": 0.0, "min_chs": 0.0}
        
        return {
            "current_chs": self.health_history[-1],
            "avg_chs": np.mean(self.health_history),
            "min_chs": np.min(self.health_history)
        }
