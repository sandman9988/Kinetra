"""
Unified Trading Environment
============================

Combines physics integration, regime detection, and multi-instrument support
into a single, standardized environment for testing framework.

This is the UNIFIED version that integrates all environments for P2.

Key Features:
- Physics state computation (64-dim) via PhysicsEngine
- Regime classification (laminar/chaotic/etc.)
- Optional regime filtering
- Multi-instrument support
- Standardized observation/action spaces

Philosophy: One environment, many modes, zero assumptions.
"""

import logging
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class TradingMode(Enum):
    """Trading mode determines behavior and constraints."""
    EXPLORATION = "exploration"  # Open for learning, no filtering
    VALIDATION = "validation"    # Regime filtering enabled
    PRODUCTION = "production"    # Full risk controls


class ActionType(Enum):
    """Trading actions."""
    HOLD = 0
    BUY = 1
    SELL = 2
    CLOSE = 3


class UnifiedTradingEnv:
    """
    Unified trading environment with physics integration.
    
    Features:
    - Physics state computation (64-dim)
    - Regime classification (laminar/chaotic/etc)
    - Regime filtering (optional, mode-dependent)
    - Multi-instrument support
    
    Observation Space: OHLCV (5) + Physics (64) + Position (3) = 72-dim
    Action Space: Discrete(4) - HOLD, BUY, SELL, CLOSE
    """
    
    def __init__(
        self,
        data: pd.DataFrame,
        mode: TradingMode = TradingMode.EXPLORATION,
        use_physics: bool = True,
        regime_filter: bool = False,
        initial_balance: float = 10000.0,
        **kwargs
    ):
        """
        Args:
            data: DataFrame with OHLCV data
            mode: Trading mode (exploration/validation/production)
            use_physics: Whether to compute physics state
            regime_filter: Whether to filter trades by regime
            initial_balance: Starting capital
        """
        self.data = data
        self.mode = mode
        self.use_physics = use_physics
        self.regime_filter = regime_filter
        self.initial_balance = initial_balance
        
        # State tracking
        self.current_idx = 0
        self.position = 0  # -1 short, 0 flat, 1 long
        self.entry_price = 0.0
        self.balance = initial_balance
        self.unrealized_pnl = 0.0
        self.trade_history = []
        
        # Physics engine (lazy init if needed)
        self._physics_engine = None
        self._physics_state_cache = {}
        
        # Observation/action space definitions
        self.observation_dim = 5 + (64 if use_physics else 0) + 3  # OHLCV + physics + position
        self.action_dim = 4  # HOLD, BUY, SELL, CLOSE
        
        logger.info(
            f"UnifiedTradingEnv initialized: mode={mode.value}, "
            f"use_physics={use_physics}, regime_filter={regime_filter}, "
            f"obs_dim={self.observation_dim}"
        )
    
    @property
    def physics_engine(self):
        """Lazy init physics engine."""
        if self._physics_engine is None and self.use_physics:
            try:
                from kinetra.physics_engine import PhysicsEngine
                self._physics_engine = PhysicsEngine()
                logger.info("Physics engine initialized")
            except Exception as e:
                logger.warning(f"Failed to init physics engine: {e}")
                self.use_physics = False
        return self._physics_engine
    
    def reset(self) -> np.ndarray:
        """
        Reset environment to initial state.
        
        Returns:
            Initial observation
        """
        self.current_idx = 100  # Start after warmup
        self.position = 0
        self.entry_price = 0.0
        self.balance = self.initial_balance
        self.unrealized_pnl = 0.0
        self.trade_history = []
        self._physics_state_cache = {}
        
        return self._get_observation()
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        """
        Execute action and return next state.
        
        Args:
            action: Action to execute (0-3)
            
        Returns:
            (next_state, reward, done, info)
        """
        # Get current market data
        current_bar = self.data.iloc[self.current_idx]
        current_price = current_bar['close']
        
        # Compute physics state (if enabled)
        regime = "UNKNOWN"
        if self.use_physics and self.physics_engine:
            physics_state = self._compute_physics_state()
            regime = self._classify_regime(physics_state)
        
        # Regime filter (if enabled and not in EXPLORATION mode)
        info = {'regime': regime, 'regime_blocked': False}
        if self.regime_filter and self.mode != TradingMode.EXPLORATION:
            if not self._is_tradeable_regime(regime):
                # Block trade in non-tradeable regimes
                action = ActionType.HOLD.value
                info['regime_blocked'] = True
        
        # Execute action
        reward = 0.0
        
        if action == ActionType.BUY.value:
            if self.position <= 0:  # Can buy if flat or short
                if self.position < 0:
                    # Close short first
                    pnl = (self.entry_price - current_price) / self.entry_price
                    self.balance += self.balance * pnl
                    reward += pnl * 100  # Scale for reward
                    self.trade_history.append({
                        'entry': self.entry_price,
                        'exit': current_price,
                        'pnl': pnl,
                        'direction': 'short'
                    })
                # Enter long
                self.position = 1
                self.entry_price = current_price
        
        elif action == ActionType.SELL.value:
            if self.position >= 0:  # Can sell if flat or long
                if self.position > 0:
                    # Close long first
                    pnl = (current_price - self.entry_price) / self.entry_price
                    self.balance += self.balance * pnl
                    reward += pnl * 100
                    self.trade_history.append({
                        'entry': self.entry_price,
                        'exit': current_price,
                        'pnl': pnl,
                        'direction': 'long'
                    })
                # Enter short
                self.position = -1
                self.entry_price = current_price
        
        elif action == ActionType.CLOSE.value:
            if self.position != 0:
                # Close current position
                if self.position > 0:
                    pnl = (current_price - self.entry_price) / self.entry_price
                    direction = 'long'
                else:
                    pnl = (self.entry_price - current_price) / self.entry_price
                    direction = 'short'
                
                self.balance += self.balance * pnl
                reward += pnl * 100
                self.trade_history.append({
                    'entry': self.entry_price,
                    'exit': current_price,
                    'pnl': pnl,
                    'direction': direction
                })
                self.position = 0
                self.entry_price = 0.0
        
        # Update unrealized P&L
        if self.position != 0:
            if self.position > 0:
                self.unrealized_pnl = (current_price - self.entry_price) / self.entry_price
            else:
                self.unrealized_pnl = (self.entry_price - current_price) / self.entry_price
            reward += self.unrealized_pnl * 0.1  # Small reward for unrealized gains
        else:
            self.unrealized_pnl = 0.0
        
        # Advance time
        self.current_idx += 1
        done = self.current_idx >= len(self.data) - 1
        
        # Get next observation
        next_state = self._get_observation()
        
        # Add info
        info.update({
            'balance': self.balance,
            'position': self.position,
            'unrealized_pnl': self.unrealized_pnl,
            'num_trades': len(self.trade_history)
        })
        
        return next_state, reward, done, info
    
    def _get_observation(self) -> np.ndarray:
        """Build observation vector."""
        current_bar = self.data.iloc[self.current_idx]
        
        # OHLCV features (normalized)
        price = current_bar['close']
        ohlcv = np.array([
            current_bar['open'] / price if price != 0 else 1,
            current_bar['high'] / price if price != 0 else 1,
            current_bar['low'] / price if price != 0 else 1,
            1.0,  # close/close = 1
            current_bar['volume'] / 1e6 if 'volume' in current_bar else 0,
        ], dtype=np.float32)
        
        # Physics features (64-dim) if enabled
        if self.use_physics and self.physics_engine:
            physics = self._compute_physics_state()
        else:
            physics = np.zeros(64, dtype=np.float32)
        
        # Position info
        position_info = np.array([
            self.position,  # -1, 0, or 1
            self.entry_price / price if price != 0 and self.position != 0 else 0,
            self.unrealized_pnl
        ], dtype=np.float32)
        
        # Concatenate all features
        obs = np.concatenate([ohlcv, physics, position_info])
        
        # Ensure correct shape
        if len(obs) != self.observation_dim:
            logger.warning(f"Observation size mismatch: {len(obs)} != {self.observation_dim}")
            obs = np.pad(obs, (0, max(0, self.observation_dim - len(obs))))[:self.observation_dim]
        
        return obs.astype(np.float32)
    
    def _compute_physics_state(self) -> np.ndarray:
        """Compute physics state for current position."""
        # Check cache
        if self.current_idx in self._physics_state_cache:
            return self._physics_state_cache[self.current_idx]
        
        # Compute physics state
        if self.physics_engine:
            try:
                lookback = 100
                start_idx = max(0, self.current_idx - lookback)
                data_slice = self.data.iloc[start_idx:self.current_idx+1]
                
                # Use physics engine to compute state
                state = self.physics_engine.compute_state(data_slice)
                
                # Ensure correct size (64-dim)
                if len(state) < 64:
                    state = np.pad(state, (0, 64 - len(state)))
                state = state[:64]
                
                # Cache result
                self._physics_state_cache[self.current_idx] = state
                return state
                
            except Exception as e:
                logger.warning(f"Physics state computation failed: {e}")
                return np.zeros(64, dtype=np.float32)
        
        return np.zeros(64, dtype=np.float32)
    
    def _classify_regime(self, physics_state: np.ndarray) -> str:
        """
        Classify market regime from physics state.
        
        Simple classification based on energy and damping.
        """
        if len(physics_state) < 2:
            return "UNKNOWN"
        
        energy = physics_state[0] if len(physics_state) > 0 else 0
        damping = physics_state[1] if len(physics_state) > 1 else 0
        
        if energy > 0.7:
            if damping < 0.3:
                return "CHAOTIC"  # High energy, low damping
            else:
                return "TRENDING"  # High energy, high damping
        else:
            if damping < 0.3:
                return "LAMINAR"  # Low energy, low damping
            else:
                return "RANGING"  # Low energy, high damping
    
    def _is_tradeable_regime(self, regime: str) -> bool:
        """Determine if regime is tradeable."""
        # In validation/production modes, avoid chaotic regimes
        if self.mode == TradingMode.PRODUCTION:
            return regime in ["TRENDING", "LAMINAR"]
        elif self.mode == TradingMode.VALIDATION:
            return regime != "CHAOTIC"
        else:
            return True  # Exploration mode: trade all regimes
    
    def get_trade_history(self) -> List[Dict]:
        """Get trade history."""
        return self.trade_history
    
    def get_metrics(self) -> Dict:
        """Calculate performance metrics."""
        if not self.trade_history:
            return {
                'total_pnl': 0,
                'num_trades': 0,
                'win_rate': 0,
                'avg_pnl': 0
            }
        
        pnls = [t['pnl'] for t in self.trade_history]
        wins = [p for p in pnls if p > 0]
        
        return {
            'total_pnl': sum(pnls),
            'num_trades': len(pnls),
            'win_rate': len(wins) / len(pnls) if pnls else 0,
            'avg_pnl': np.mean(pnls) if pnls else 0,
            'final_balance': self.balance
        }
