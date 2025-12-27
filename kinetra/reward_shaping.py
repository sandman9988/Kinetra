"""
Adaptive Reward Shaping (ARS)

Implements MFE/MAE-based reward with regime-adaptive coefficients.
"""

import numpy as np
import pandas as pd
from typing import Dict


class AdaptiveRewardShaper:
    """
    Adaptive Reward Shaping with MFE/MAE normalization.
    
    R_t = (PnL / E_t) + α·(MFE/ATR) - β·(MAE/ATR) - γ·Time
    
    Where α, β are regime-adaptive.
    """
    
    def __init__(self, base_alpha: float = 0.15, base_beta: float = 0.10, gamma: float = 0.01):
        """Initialize reward shaper."""
        self.base_alpha = base_alpha
        self.base_beta = base_beta
        self.gamma = gamma
    
    def calculate_reward(
        self,
        pnl: float,
        energy: float,
        mfe: float,
        mae: float,
        atr: float,
        time_in_trade: int
    ) -> float:
        """Calculate adaptive reward."""
        # Physics-aligned base reward
        reward = pnl / (energy + 1e-6)
        
        # MFE/MAE components
        reward += self.base_alpha * (mfe / (atr + 1e-6))
        reward -= self.base_beta * (mae / (atr + 1e-6))
        
        # Time penalty
        reward -= self.gamma * time_in_trade
        
        return float(reward)
