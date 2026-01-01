#!/usr/bin/env python3
"""
Kinetra Triad System: Incumbent, Competitor, Researcher
========================================================

Core principle: Markets are about IMBALANCES, not equilibrium.
- Price moves when imbalances can no longer be absorbed
- Institutions amplify/prolong imbalances
- Edge comes from timing the RESOLUTION of imbalance

Architecture:
- Incumbent (PPO): Exploit stable regimes, rock-solid execution
- Competitor (A2C): Challenge incumbent, optimize transitions  
- Researcher (SAC): Explore new alpha, maximum entropy exploration

No assumptions:
- No magic numbers (thresholds from rolling history percentiles)
- No linearity (only asymmetric/signed features)
- No stationarity (adaptive to regime changes)
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Callable
from enum import Enum
from collections import deque
from abc import ABC, abstractmethod
import warnings

from numpy import dtype, ndarray, signedinteger
from numpy._typing import _32Bit, _64Bit

warnings.filterwarnings('ignore')


# =============================================================================
# ENUMS & CONFIG
# =============================================================================

class AgentRole(Enum):
    """Trading role determines reward structure and risk tolerance."""
    TRADER = "trader"              # Short-term PnL focus
    RISK_MANAGER = "risk_manager"  # Drawdown/VaR focus  
    PORTFOLIO_MANAGER = "portfolio_manager"  # Sharpe/allocation focus


class RegimeState(Enum):
    """Market regime states based on imbalance characteristics."""
    BALANCED = "balanced"          # Low net flow, high liquidity
    BUILDUP = "buildup"            # Imbalance accumulating, not released
    RELEASE = "release"            # Imbalance threshold breached, trending
    CRISIS = "crisis"              # Liquidity collapse, panic feedback


class ImbalanceDirection(Enum):
    """Direction of detected imbalance."""
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"


# =============================================================================
# IMBALANCE FEATURES (Assumption-Free)
# =============================================================================

@dataclass
class ImbalanceState:
    """
    Core imbalance measurements - all ASYMMETRIC, no averages.
    
    These are the sensors that detect market pressure before release.
    """
    # Order Flow Imbalance
    cvd: float = 0.0                    # Cumulative Volume Delta (signed)
    cvd_divergence: float = 0.0         # CVD vs price divergence
    buy_pressure: float = 0.5           # Buy volume ratio [0,1]
    sell_acceleration: float = 0.0      # Rate of sell pressure increase
    
    # Liquidity Imbalance  
    amihud_illiquidity: float = 0.0     # Amihud ratio (|return|/volume)
    spread_rank: float = 0.5            # Current spread vs history percentile
    depth_asymmetry: float = 0.0        # Bid depth - Ask depth (normalized)
    
    # Behavioral/Information Imbalance
    tail_asymmetry: float = 0.0         # Left tail - Right tail magnitude
    skewness_signed: float = 0.0        # Directional skew (not squared)
    entropy: float = 1.0                # Permutation entropy [0,1]
    
    # Time/Regime Imbalance
    wavelet_energy_skew: float = 0.0    # Energy asymmetry across scales
    hilbert_phase_persistence: float = 0.0  # How long current phase held
    recurrence_determinism: float = 0.5     # Structure vs chaos [0,1]
    
    # Derived
    net_pressure: float = 0.0           # Overall directional pressure
    imbalance_magnitude: float = 0.0    # How extreme is the imbalance
    
    def to_array(self) -> np.ndarray:
        """Convert to feature array for RL agents."""
        return np.array([
            self.cvd, self.cvd_divergence, self.buy_pressure, self.sell_acceleration,
            self.amihud_illiquidity, self.spread_rank, self.depth_asymmetry,
            self.tail_asymmetry, self.skewness_signed, self.entropy,
            self.wavelet_energy_skew, self.hilbert_phase_persistence, self.recurrence_determinism,
            self.net_pressure, self.imbalance_magnitude
        ], dtype=np.float32)
    
    @property
    def direction(self) -> ImbalanceDirection:
        """Determine overall imbalance direction."""
        if self.net_pressure > 0.3:
            return ImbalanceDirection.BULLISH
        elif self.net_pressure < -0.3:
            return ImbalanceDirection.BEARISH
        return ImbalanceDirection.NEUTRAL


class ImbalanceExtractor:
    """
    Extract imbalance features from OHLCV data.
    
    All features are:
    - Asymmetric (signed, directional)
    - Non-parametric (ranks, medians, percentiles)
    - Rolling (no fixed thresholds)
    """
    
    def __init__(self, lookback: int = 100):
        self.lookback = lookback
        self.history = deque(maxlen=lookback * 2)
    
    def extract(self, df: pd.DataFrame, idx: int) -> ImbalanceState:
        """Extract imbalance state at given index."""
        if idx < self.lookback:
            return ImbalanceState()
        
        window = df.iloc[max(0, idx - self.lookback):idx + 1]
        
        o = window['open'].values
        h = window['high'].values
        l = window['low'].values
        c = window['close'].values
        v = window['volume'].values if 'volume' in window else np.ones(len(window))
        
        state = ImbalanceState()
        
        # === Order Flow Imbalance ===
        # CVD: Signed volume based on close vs open
        signed_vol = np.sign(c - o) * v
        state.cvd = np.sum(signed_vol[-20:]) / (np.sum(v[-20:]) + 1e-10)
        
        # CVD Divergence: Price up but CVD down (or vice versa)
        price_change = (c[-1] - c[-20]) / (c[-20] + 1e-10)
        cvd_change = state.cvd
        state.cvd_divergence = price_change - cvd_change  # Positive = bullish divergence
        
        # Buy pressure from candle body position
        body_pos = (c - l) / (h - l + 1e-10)
        state.buy_pressure = np.mean(body_pos[-10:])
        
        # Sell acceleration: Is selling speeding up?
        sell_vol = np.where(c < o, v, 0)
        if len(sell_vol) >= 10:
            recent_sell = np.mean(sell_vol[-5:])
            prior_sell = np.mean(sell_vol[-10:-5])
            state.sell_acceleration = (recent_sell - prior_sell) / (prior_sell + 1e-10)
        
        # === Liquidity Imbalance ===
        returns = np.diff(c) / (c[:-1] + 1e-10)
        state.amihud_illiquidity = np.median(np.abs(returns) / (v[1:] + 1e-10)) * 1e6
        
        # Spread proxy from high-low range
        ranges = (h - l) / (c + 1e-10)
        state.spread_rank = (ranges[-1] - np.percentile(ranges, 10)) / (np.percentile(ranges, 90) - np.percentile(ranges, 10) + 1e-10)
        
        # Depth asymmetry from upper/lower wicks
        upper_wick = (h - np.maximum(o, c)) / (h - l + 1e-10)
        lower_wick = (np.minimum(o, c) - l) / (h - l + 1e-10)
        state.depth_asymmetry = np.mean(lower_wick[-10:] - upper_wick[-10:])  # Positive = more buying pressure
        
        # === Behavioral Imbalance ===
        # Tail asymmetry: Compare left vs right tail magnitudes
        down_moves = returns[returns < 0]
        up_moves = returns[returns > 0]
        left_tail = np.percentile(down_moves, 10) if len(down_moves) > 5 else 0
        right_tail = np.percentile(up_moves, 90) if len(up_moves) > 5 else 0
        state.tail_asymmetry = abs(left_tail) - abs(right_tail)  # Positive = fatter left tail (bearish)
        
        # Signed skewness (not squared!)
        if len(returns) > 10:
            mean_ret = np.median(returns)  # Use median, not mean
            state.skewness_signed = np.mean(np.sign(returns - mean_ret) * (returns - mean_ret) ** 2)
        
        # Permutation entropy for chaos detection
        state.entropy = self._permutation_entropy(c[-20:])
        
        # === Time/Regime Imbalance ===
        # Wavelet energy skew (simplified: compare short vs long period volatility)
        short_vol = np.std(returns[-5:]) if len(returns) >= 5 else 0
        long_vol = np.std(returns[-20:]) if len(returns) >= 20 else short_vol
        state.wavelet_energy_skew = (short_vol - long_vol) / (long_vol + 1e-10)
        
        # Phase persistence: How long have we been in current direction?
        signs = np.sign(returns[-20:])
        if len(signs) > 0:
            current_sign = signs[-1]
            persistence = 0
            for s in reversed(signs):
                if s == current_sign:
                    persistence += 1
                else:
                    break
            state.hilbert_phase_persistence = persistence / 20
        
        # Recurrence determinism (simplified)
        state.recurrence_determinism = 1 - state.entropy
        
        # === Derived ===
        state.net_pressure = (
            0.3 * state.cvd +
            0.2 * state.buy_pressure * 2 - 1 +
            0.2 * state.depth_asymmetry +
            0.15 * (-state.tail_asymmetry) +
            0.15 * state.skewness_signed * 10
        )
        state.net_pressure = np.clip(state.net_pressure, -1, 1)
        
        state.imbalance_magnitude = np.sqrt(
            state.cvd ** 2 +
            state.cvd_divergence ** 2 +
            state.amihud_illiquidity ** 2 +
            state.tail_asymmetry ** 2
        )
        
        return state
    
    def _permutation_entropy(self, series: np.ndarray, order: int = 3) -> float:
        """Compute permutation entropy for chaos detection."""
        if len(series) < order + 1:
            return 1.0
        
        # Create ordinal patterns
        patterns = []
        for i in range(len(series) - order):
            pattern = tuple(np.argsort(series[i:i + order]))
            patterns.append(pattern)
        
        if not patterns:
            return 1.0
        
        # Count pattern frequencies
        from collections import Counter
        counts = Counter(patterns)
        probs = np.array(list(counts.values())) / len(patterns)
        
        # Entropy
        entropy = -np.sum(probs * np.log(probs + 1e-10))
        import math
        max_entropy = np.log(min(len(patterns), math.factorial(order)))
        
        return entropy / (max_entropy + 1e-10)


# =============================================================================
# REGIME DETECTOR (No Assumptions)
# =============================================================================

class RegimeDetector:
    """
    Detect market regimes from imbalance states.
    
    Uses rolling history percentiles - NO MAGIC NUMBERS.
    Thresholds lock at session start, roll daily.
    """
    
    def __init__(self, history_window: int = 500):
        self.history_window = history_window
        self.history: deque = deque(maxlen=history_window)
        self.locked_thresholds: Optional[Dict] = None
    
    def update(self, state: ImbalanceState):
        """Add new state to history."""
        self.history.append(state.to_array())
    
    def lockdown_thresholds(self):
        """
        Lock thresholds from historical percentiles.
        Call at session start - no updates during live trading.
        """
        if len(self.history) < 50:
            # Not enough history, use defaults
            self.locked_thresholds = {
                'imbalance_breach': 0.5,
                'illiquidity_spike': 0.7,
                'entropy_collapse': 0.3,
                'cvd_divergence_extreme': 0.4,
            }
            return
        
        arr = np.array(self.history)
        
        self.locked_thresholds = {
            'imbalance_breach': np.percentile(np.abs(arr[:, 13]), 80),  # net_pressure
            'illiquidity_spike': np.percentile(arr[:, 4], 90),  # amihud
            'entropy_collapse': np.percentile(arr[:, 9], 20),  # entropy
            'cvd_divergence_extreme': np.percentile(np.abs(arr[:, 1]), 85),  # cvd_divergence
        }
    
    def detect(self, state: ImbalanceState) -> RegimeState:
        """Detect current regime from imbalance state."""
        if self.locked_thresholds is None:
            self.lockdown_thresholds()
        
        t = self.locked_thresholds
        
        # Crisis: Liquidity collapse + high imbalance
        if (state.amihud_illiquidity > t['illiquidity_spike'] and 
            state.imbalance_magnitude > t['imbalance_breach']):
            return RegimeState.CRISIS
        
        # Release: Imbalance threshold breached, entropy dropping (structure emerging)
        if (state.imbalance_magnitude > t['imbalance_breach'] and
            state.entropy < t['entropy_collapse']):
            return RegimeState.RELEASE
        
        # Buildup: Divergence building, not yet released
        if abs(state.cvd_divergence) > t['cvd_divergence_extreme']:
            return RegimeState.BUILDUP
        
        # Balanced: Everything within normal ranges
        return RegimeState.BALANCED


# =============================================================================
# BASE AGENT (Abstract)
# =============================================================================

class TriadAgent(ABC):
    """Base class for Triad agents."""
    
    def __init__(
        self,
        state_dim: int,
        n_actions: int,
        role: AgentRole,
        learning_rate: float = 0.001,
        gamma: float = 0.95,
    ):
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.role = role
        self.lr = learning_rate
        self.gamma = gamma
        
        # Performance tracking
        self.episode_rewards: List[float] = []
        self.episode_pnls: List[float] = []
    
    @abstractmethod
    def select_action(self, state: np.ndarray, explore: bool = True) -> int:
        """Select action given state."""
        pass
    
    @abstractmethod
    def update(self, state: np.ndarray, action: int, reward: float, 
               next_state: np.ndarray, done: bool):
        """Update agent from experience."""
        pass
    
    def compute_reward(self, pnl: float, drawdown: float, sharpe: float) -> float:
        """Compute role-specific reward."""
        if self.role == AgentRole.TRADER:
            # Focus on PnL with small transaction cost penalty
            return pnl * 100 - 0.01
        elif self.role == AgentRole.RISK_MANAGER:
            # Focus on minimizing drawdown
            return pnl * 50 - drawdown * 200 + max(0, sharpe) * 10
        elif self.role == AgentRole.PORTFOLIO_MANAGER:
            # Focus on risk-adjusted returns
            return sharpe * 50 + pnl * 30 - drawdown * 50
        return pnl


# =============================================================================
# INCUMBENT AGENT (PPO-style)
# =============================================================================

class IncumbentAgent(TriadAgent):
    """
    Incumbent: PPO-style agent for stable exploitation.
    
    - Rock-solid, low-variance updates
    - Exploits confirmed regimes
    - Best for: Stable trending or ranging markets
    """
    
    def __init__(self, state_dim: int, n_actions: int, role: AgentRole, **kwargs):
        super().__init__(state_dim, n_actions, role, **kwargs)
        
        # Policy network (linear for speed, can upgrade to MLP)
        self.policy_W = np.random.randn(state_dim, n_actions) * 0.1
        self.policy_b = np.zeros(n_actions)
        
        # Value network
        self.value_W = np.random.randn(state_dim) * 0.1
        self.value_b = 0.0
        
        # PPO hyperparams
        self.clip_epsilon = 0.2
        self.entropy_coef = 0.01
        
        # Trajectory buffer
        self.buffer: List[Tuple] = []
    
    def _softmax(self, x: np.ndarray) -> np.ndarray:
        x = np.clip(x, -20, 20)
        e = np.exp(x - np.max(x))
        p = e / (e.sum() + 1e-10)
        if np.any(np.isnan(p)):
            return np.ones(len(x)) / len(x)
        return p
    
    def select_action(self, state: np.ndarray, explore: bool = True) -> int:
        state = np.nan_to_num(state, nan=0, posinf=0, neginf=0)
        logits = state @ self.policy_W + self.policy_b
        probs = self._softmax(logits)
        
        if explore:
            action = np.random.choice(self.n_actions, p=probs)
        else:
            action = np.argmax(probs)
        
        # Store for PPO update
        self.buffer.append((state, action, np.log(probs[action] + 1e-10), 0))
        return action
    
    def update(self, state: np.ndarray, action: int, reward: float,
               next_state: np.ndarray, done: bool):
        """PPO-style clipped update."""
        if self.buffer:
            self.buffer[-1] = (*self.buffer[-1][:3], reward)
        
        if done and len(self.buffer) > 1:
            # Compute returns with GAE
            returns = []
            G = 0
            for _, _, _, r in reversed(self.buffer):
                G = r + self.gamma * G
                returns.insert(0, G)
            returns = np.array(returns)
            returns = (returns - returns.mean()) / (returns.std() + 1e-10)
            
            # PPO update
            for (s, a, old_log_prob, _), R in zip(self.buffer, returns):
                s = np.nan_to_num(s, nan=0, posinf=0, neginf=0)
                
                # Value update
                value = s @ self.value_W + self.value_b
                advantage = R - value
                self.value_W += self.lr * 0.5 * advantage * s
                self.value_b += self.lr * 0.5 * advantage
                
                # Policy update with clipping
                logits = s @ self.policy_W + self.policy_b
                probs = self._softmax(logits)
                new_log_prob = np.log(probs[a] + 1e-10)
                
                ratio = np.exp(new_log_prob - old_log_prob)
                clipped_ratio = np.clip(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon)
                
                # Use minimum of clipped and unclipped
                policy_loss = -min(ratio * advantage, clipped_ratio * advantage)
                
                # Gradient approximation
                grad = np.zeros(self.n_actions)
                grad[a] = policy_loss
                self.policy_W -= self.lr * np.outer(s, grad)
                self.policy_b -= self.lr * grad
            
            self.buffer = []


# =============================================================================
# COMPETITOR AGENT (A2C-style)
# =============================================================================

class CompetitorAgent(TriadAgent):
    """
    Competitor: A2C-style agent for fast adaptation.
    
    - Challenges Incumbent with incremental improvements
    - Fast parallel updates
    - Best for: Transitions, regime shifts
    """
    
    def __init__(self, state_dim: int, n_actions: int, role: AgentRole, **kwargs):
        super().__init__(state_dim, n_actions, role, **kwargs)
        
        # Actor
        self.actor_W = np.random.randn(state_dim, n_actions) * 0.1
        self.actor_b = np.zeros(n_actions)
        
        # Critic
        self.critic_W = np.random.randn(state_dim) * 0.1
        self.critic_b = 0.0
        
        # A2C hyperparams
        self.entropy_coef = 0.05  # Higher entropy for more exploration than Incumbent
    
    def _softmax(self, x: np.ndarray) -> np.ndarray:
        x = np.clip(x, -20, 20)
        e = np.exp(x - np.max(x))
        p = e / (e.sum() + 1e-10)
        if np.any(np.isnan(p)):
            return np.ones(len(x)) / len(x)
        return p
    
    def select_action(self, state: np.ndarray, explore: bool = True) -> int | ndarray[
        tuple[Any, ...], dtype[signedinteger[_32Bit | _64Bit]]] | ndarray[tuple[Any, ...], dtype[Any]] | signedinteger[
                                                                            _32Bit | _64Bit] | Any:
        state = np.nan_to_num(state, nan=0, posinf=0, neginf=0)
        logits = state @ self.actor_W + self.actor_b
        probs = self._softmax(logits)
        
        if explore:
            return np.random.choice(self.n_actions, p=probs)
        return np.argmax(probs)
    
    def update(self, state: np.ndarray, action: int, reward: float,
               next_state: np.ndarray, done: bool):
        """A2C online update."""
        state = np.nan_to_num(state, nan=0, posinf=0, neginf=0)
        next_state = np.nan_to_num(next_state, nan=0, posinf=0, neginf=0)
        
        # Critic values
        value = state @ self.critic_W + self.critic_b
        next_value = 0 if done else next_state @ self.critic_W + self.critic_b
        
        # TD error (advantage)
        td_target = reward + self.gamma * next_value
        advantage = td_target - value
        
        # Critic update
        self.critic_W += self.lr * advantage * state
        self.critic_b += self.lr * advantage
        
        # Actor update
        logits = state @ self.actor_W + self.actor_b
        probs = self._softmax(logits)
        
        # Policy gradient with entropy bonus
        entropy = -np.sum(probs * np.log(probs + 1e-10))
        
        grad = -probs.copy()
        grad[action] += 1
        grad *= advantage
        grad += self.entropy_coef * (np.log(probs + 1e-10) + 1)  # Entropy gradient
        
        self.actor_W += self.lr * np.outer(state, grad)
        self.actor_b += self.lr * grad


# =============================================================================
# RESEARCHER AGENT (SAC-style)
# =============================================================================

class ResearcherAgent(TriadAgent):
    """
    Researcher: SAC-style agent for maximum exploration.
    
    - Maximum entropy objective (explore novel alphas)
    - Off-policy with replay buffer
    - Best for: Uncertain regimes, new alpha discovery
    """
    
    def __init__(self, state_dim: int, n_actions: int, role: AgentRole, **kwargs):
        super().__init__(state_dim, n_actions, role, **kwargs)
        
        # Twin Q-networks
        self.Q1_W = np.random.randn(state_dim, n_actions) * 0.1
        self.Q1_b = np.zeros(n_actions)
        self.Q2_W = np.random.randn(state_dim, n_actions) * 0.1
        self.Q2_b = np.zeros(n_actions)
        
        # Policy network
        self.policy_W = np.random.randn(state_dim, n_actions) * 0.1
        self.policy_b = np.zeros(n_actions)
        
        # SAC hyperparams
        self.alpha = 0.2  # Entropy temperature (high = more exploration)
        self.target_entropy = -np.log(1.0 / n_actions)  # Target entropy
        
        # Replay buffer
        self.buffer: deque = deque(maxlen=10000)
        self.batch_size = 32
    
    def _softmax(self, x: np.ndarray) -> np.ndarray:
        x = np.clip(x, -20, 20)
        e = np.exp(x - np.max(x))
        p = e / (e.sum() + 1e-10)
        if np.any(np.isnan(p)):
            return np.ones(len(x)) / len(x)
        return p
    
    def select_action(self, state: np.ndarray, explore: bool = True) -> int | ndarray[
        tuple[Any, ...], dtype[signedinteger[_32Bit | _64Bit]]] | ndarray[tuple[Any, ...], dtype[Any]] | signedinteger[
                                                                            _32Bit | _64Bit] | Any:
        state = np.nan_to_num(state, nan=0, posinf=0, neginf=0)
        logits = state @ self.policy_W + self.policy_b
        probs = self._softmax(logits)
        
        # SAC always samples for maximum entropy
        if explore:
            return np.random.choice(self.n_actions, p=probs)
        return np.argmax(probs)
    
    def update(self, state: np.ndarray, action: int, reward: float,
               next_state: np.ndarray, done: bool):
        """SAC off-policy update with entropy maximization."""
        state = np.nan_to_num(state, nan=0, posinf=0, neginf=0)
        next_state = np.nan_to_num(next_state, nan=0, posinf=0, neginf=0)
        
        self.buffer.append((state, action, reward, next_state, done))
        
        if len(self.buffer) < self.batch_size:
            return
        
        # Sample batch
        indices = np.random.choice(len(self.buffer), self.batch_size, replace=False)
        batch = [self.buffer[i] for i in indices]
        
        for s, a, r, ns, d in batch:
            # Q-target with entropy
            next_probs = self._softmax(ns @ self.policy_W + self.policy_b)
            next_log_probs = np.log(next_probs + 1e-10)
            
            next_q1 = ns @ self.Q1_W + self.Q1_b
            next_q2 = ns @ self.Q2_W + self.Q2_b
            next_q = np.minimum(next_q1, next_q2)
            
            # Soft value target
            next_value = np.sum(next_probs * (next_q - self.alpha * next_log_probs))
            target = r + (0 if d else self.gamma * next_value)
            
            # Q updates
            q1 = s @ self.Q1_W + self.Q1_b
            q2 = s @ self.Q2_W + self.Q2_b
            
            td1 = target - q1[a]
            td2 = target - q2[a]
            
            self.Q1_W[:, a] += self.lr * td1 * s
            self.Q1_b[a] += self.lr * td1
            self.Q2_W[:, a] += self.lr * td2 * s
            self.Q2_b[a] += self.lr * td2
            
            # Policy update (maximize Q + entropy)
            probs = self._softmax(s @ self.policy_W + self.policy_b)
            log_probs = np.log(probs + 1e-10)
            q_vals = np.minimum(s @ self.Q1_W + self.Q1_b, s @ self.Q2_W + self.Q2_b)
            
            # Gradient: increase prob of high Q, increase entropy
            policy_grad = probs * (q_vals - self.alpha * log_probs - np.sum(probs * (q_vals - self.alpha * log_probs)))
            
            self.policy_W += self.lr * 0.1 * np.outer(s, policy_grad)
            self.policy_b += self.lr * 0.1 * policy_grad


# =============================================================================
# META-CONTROLLER
# =============================================================================

class MetaController:
    """
    Meta-controller for Triad handoffs.
    
    Decides which agent should dominate based on:
    - Regime state
    - Bayesian surprise (information gain)
    - Rolling performance
    """
    
    def __init__(self, surprise_threshold: float = 0.5):
        self.surprise_threshold = surprise_threshold
        self.performance_history = {
            'incumbent': deque(maxlen=100),
            'competitor': deque(maxlen=100),
            'researcher': deque(maxlen=100),
        }
        self.active_agent = 'incumbent'
    
    def compute_surprise(self, state: ImbalanceState, prediction: float, actual: float) -> float:
        """Compute Bayesian surprise from prediction error."""
        return abs(actual - prediction) / (abs(prediction) + 1e-10)
    
    def select_agent(self, regime: RegimeState, surprise: float) -> str:
        """Select which agent should act."""
        # High surprise → Researcher explores
        if surprise > self.surprise_threshold:
            return 'researcher'
        
        # Crisis → Researcher (unknown territory)
        if regime == RegimeState.CRISIS:
            return 'researcher'
        
        # Buildup/transition → Competitor challenges
        if regime == RegimeState.BUILDUP:
            return 'competitor'
        
        # Release/trending → Incumbent exploits
        if regime == RegimeState.RELEASE:
            return 'incumbent'
        
        # Balanced → Compare performance, let Competitor challenge
        inc_perf = np.mean(self.performance_history['incumbent']) if self.performance_history['incumbent'] else 0
        comp_perf = np.mean(self.performance_history['competitor']) if self.performance_history['competitor'] else 0
        
        if comp_perf > inc_perf * 1.1:  # Competitor 10% better
            return 'competitor'
        
        return 'incumbent'
    
    def update_performance(self, agent_name: str, reward: float):
        """Track agent performance."""
        self.performance_history[agent_name].append(reward)
    
    def get_ensemble_action(self, actions: Dict[str, int], weights: Optional[Dict[str, float]] = None) -> int:
        """Ensemble voting (optional, for more stable decisions)."""
        if weights is None:
            weights = {'incumbent': 0.5, 'competitor': 0.3, 'researcher': 0.2}
        
        # Weighted vote
        votes = {}
        for agent, action in actions.items():
            votes[action] = votes.get(action, 0) + weights.get(agent, 0.33)
        
        return max(votes, key=votes.get)


# =============================================================================
# TRIAD SYSTEM (Main Interface)
# =============================================================================

class TriadSystem:
    """
    Complete Triad Trading System.
    
    Markets are about IMBALANCES:
    - Detect imbalance accumulation
    - Time the resolution
    - Adapt to regime shifts
    
    Usage:
        system = TriadSystem(role=AgentRole.TRADER)
        system.lockdown_thresholds(historical_data)
        
        for bar in live_data:
            action = system.step(bar)
            # Execute trade based on action
    """
    
    def __init__(
        self,
        role: AgentRole = AgentRole.TRADER,
        n_actions: int = 4,  # Hold, Buy, Sell, Close
        lookback: int = 100,
        history_window: int = 500,
    ):
        self.role = role
        self.n_actions = n_actions
        
        # Core components
        self.imbalance_extractor = ImbalanceExtractor(lookback=lookback)
        self.regime_detector = RegimeDetector(history_window=history_window)
        self.meta_controller = MetaController()
        
        # State dimension from ImbalanceState
        state_dim = 15  # Number of features in ImbalanceState
        
        # Triad agents
        self.agents = {
            'incumbent': IncumbentAgent(state_dim, n_actions, role),
            'competitor': CompetitorAgent(state_dim, n_actions, role),
            'researcher': ResearcherAgent(state_dim, n_actions, role),
        }
        
        # Tracking
        self.current_regime = RegimeState.BALANCED
        self.current_imbalance = ImbalanceState()
        self.last_prediction = 0.0
        self.bar_count = 0
    
    def lockdown_thresholds(self, historical_df: pd.DataFrame):
        """
        Lock thresholds from historical data.
        Call at session start.
        """
        for i in range(50, len(historical_df)):
            state = self.imbalance_extractor.extract(historical_df, i)
            self.regime_detector.update(state)
        
        self.regime_detector.lockdown_thresholds()
        print(f"✅ Thresholds locked: {self.regime_detector.locked_thresholds}")
    
    def step(self, df: pd.DataFrame, idx: int, explore: bool = True) -> Tuple[int, Dict]:
        """
        Process one bar and return action.
        
        Returns:
            action: int (0=Hold, 1=Buy, 2=Sell, 3=Close)
            info: dict with regime, agent, imbalance state
        """
        self.bar_count += 1
        
        # Extract imbalance state
        self.current_imbalance = self.imbalance_extractor.extract(df, idx)
        state = self.current_imbalance.to_array()
        
        # Update regime detector
        self.regime_detector.update(self.current_imbalance)
        self.current_regime = self.regime_detector.detect(self.current_imbalance)
        
        # Compute surprise
        actual_price_change = 0
        if idx > 0:
            actual_price_change = (df.iloc[idx]['close'] - df.iloc[idx-1]['close']) / df.iloc[idx-1]['close']
        surprise = self.meta_controller.compute_surprise(
            self.current_imbalance, self.last_prediction, actual_price_change
        )
        
        # Select active agent
        active_agent = self.meta_controller.select_agent(self.current_regime, surprise)
        
        # Get action from active agent
        action = self.agents[active_agent].select_action(state, explore=explore)
        
        # Store prediction for next surprise calculation
        self.last_prediction = self.current_imbalance.net_pressure * 0.001  # Simple prediction
        
        info = {
            'regime': self.current_regime.value,
            'agent': active_agent,
            'imbalance_direction': self.current_imbalance.direction.value,
            'net_pressure': self.current_imbalance.net_pressure,
            'imbalance_magnitude': self.current_imbalance.imbalance_magnitude,
            'surprise': surprise,
        }
        
        return action, info
    
    def update(self, reward: float, done: bool = False):
        """Update all agents with reward."""
        state = self.current_imbalance.to_array()
        
        for name, agent in self.agents.items():
            agent.update(state, 0, reward, state, done)
            self.meta_controller.update_performance(name, reward)
    
    def get_state_summary(self) -> str:
        """Get human-readable state summary."""
        return (
            f"Regime: {self.current_regime.value.upper()} | "
            f"Direction: {self.current_imbalance.direction.value} | "
            f"Pressure: {self.current_imbalance.net_pressure:+.3f} | "
            f"Magnitude: {self.current_imbalance.imbalance_magnitude:.3f}"
        )


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================

def create_triad_system(
    role: str = 'trader',
    n_actions: int = 4,
) -> TriadSystem:
    """
    Create a Triad system for the specified role.
    
    Args:
        role: 'trader', 'risk_manager', or 'portfolio_manager'
        n_actions: Number of actions (default 4: Hold, Buy, Sell, Close)
    
    Returns:
        Configured TriadSystem
    """
    role_map = {
        'trader': AgentRole.TRADER,
        'risk_manager': AgentRole.RISK_MANAGER,
        'portfolio_manager': AgentRole.PORTFOLIO_MANAGER,
    }
    
    return TriadSystem(
        role=role_map.get(role, AgentRole.TRADER),
        n_actions=n_actions,
    )


# =============================================================================
# TESTING
# =============================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("KINETRA TRIAD SYSTEM TEST")
    print("Markets are about IMBALANCES")
    print("=" * 60)
    
    # Create synthetic test data
    np.random.seed(42)
    n_bars = 500
    
    # Generate price with regime changes
    prices = [100.0]
    for i in range(n_bars - 1):
        if i < 150:  # Ranging
            change = np.random.randn() * 0.002
        elif i < 250:  # Buildup
            change = np.random.randn() * 0.003 + 0.0005
        elif i < 350:  # Trending release
            change = np.random.randn() * 0.005 + 0.002
        else:  # Crisis then recovery
            change = np.random.randn() * 0.01 - 0.001
        prices.append(prices[-1] * (1 + change))
    
    df = pd.DataFrame({
        'open': prices,
        'high': [p * (1 + abs(np.random.randn() * 0.003)) for p in prices],
        'low': [p * (1 - abs(np.random.randn() * 0.003)) for p in prices],
        'close': [p * (1 + np.random.randn() * 0.001) for p in prices],
        'volume': [np.random.randint(1000, 10000) for _ in prices],
    })
    
    # Test Triad system
    system = create_triad_system(role='trader')
    
    # Lockdown thresholds from first 200 bars
    system.lockdown_thresholds(df.iloc[:200])
    
    # Run through remaining bars
    print("\nRunning simulation...")
    regime_counts = {}
    agent_counts = {}
    
    for i in range(200, len(df)):
        action, info = system.step(df, i)
        
        regime_counts[info['regime']] = regime_counts.get(info['regime'], 0) + 1
        agent_counts[info['agent']] = agent_counts.get(info['agent'], 0) + 1
        
        # Simulate reward based on action
        if i < len(df) - 1:
            price_change = (df.iloc[i+1]['close'] - df.iloc[i]['close']) / df.iloc[i]['close']
            if action == 1:  # Buy
                reward = price_change * 100
            elif action == 2:  # Sell
                reward = -price_change * 100
            else:
                reward = -0.001  # Small cost for holding/closing
            system.update(reward)
        
        if i % 50 == 0:
            print(f"Bar {i}: {system.get_state_summary()}")
    
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"\nRegime distribution: {regime_counts}")
    print(f"Agent usage: {agent_counts}")
    print("\n✅ Triad System test complete!")
