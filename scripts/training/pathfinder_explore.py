#!/usr/bin/env python3
"""
PATHFINDER EXPLORATION
======================

"We don't know what we don't know" - explore without assumptions.

Uses cached data, tests multiple agents, shows results fast.
"""

import sys
import time
import warnings
from pathlib import Path
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
from numpy import dtype, ndarray, signedinteger, bool, unsignedinteger
from numpy._typing import _32Bit, _64Bit, _16Bit, _8Bit

sys.path.insert(0, str(Path(__file__).parent.parent))
warnings.filterwarnings('ignore')

# Import the fixed framework
from rl_exploration_framework import (
    get_rl_state_features, 
    get_rl_feature_names,
)

# =============================================================================
# DATA LOADING
# =============================================================================

def load_csv_direct(filepath: Path) -> pd.DataFrame:
    """Load MT5 CSV directly."""
    df = pd.read_csv(filepath, sep='\t')
    df.columns = [c.strip('<>').lower() for c in df.columns]
    df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'])
    df.set_index('datetime', inplace=True)
    df = df.rename(columns={'tickvol': 'volume'})
    return df[['open', 'high', 'low', 'close', 'volume', 'spread']]


def compute_simple_physics(df: pd.DataFrame, lookback: int = 20) -> pd.DataFrame:
    """
    Compute simplified physics state for exploration.
    No external dependencies - works standalone.
    """
    physics = pd.DataFrame(index=df.index)
    close = df['close'].values
    volume = df['volume'].values
    high = df['high'].values
    low = df['low'].values
    
    # Returns and derivatives
    returns = pd.Series(close).pct_change().fillna(0).values
    
    # Velocity (return)
    physics['v'] = returns
    
    # Acceleration (change in return)
    physics['a'] = pd.Series(returns).diff().fillna(0).values
    
    # Jerk (change in acceleration)
    physics['j'] = physics['a'].diff().fillna(0).values
    physics['jerk_z'] = (physics['j'] - physics['j'].rolling(lookback).mean()) / (physics['j'].rolling(lookback).std() + 1e-10)
    
    # Energy
    physics['energy'] = 0.5 * physics['v'] ** 2
    physics['PE'] = 1.0 / (pd.Series(volume).rolling(lookback).mean() + 1e-10)
    physics['eta'] = physics['energy'] / (physics['PE'] + 1e-10)
    
    # Volatility
    physics['vol_rs'] = pd.Series(returns).rolling(lookback).std().fillna(0).values
    physics['vol_yz'] = physics['vol_rs'] * np.sqrt(252)  # Annualized
    physics['vol_gk'] = pd.Series(np.log(high / low + 1e-10) ** 2).rolling(lookback).mean().fillna(0).values
    
    # Z-scores
    for col in ['vol_rs', 'vol_yz']:
        physics[f'{col}_z'] = (physics[col] - physics[col].rolling(100).mean()) / (physics[col].rolling(100).std() + 1e-10)
    
    # Damping
    physics['damping'] = physics['vol_rs'] / (np.abs(physics['v']) + 1e-10)
    physics['viscosity'] = physics['damping'] * physics['vol_rs']
    physics['visc_z'] = (physics['viscosity'] - physics['viscosity'].rolling(lookback).mean()) / (physics['viscosity'].rolling(lookback).std() + 1e-10)
    
    # Entropy (approximation using rolling range)
    price_range = (high - low) / (close + 1e-10)
    physics['entropy'] = pd.Series(price_range).rolling(lookback).std().fillna(0).values
    physics['entropy_z'] = (physics['entropy'] - physics['entropy'].rolling(100).mean()) / (physics['entropy'].rolling(100).std() + 1e-10)
    
    # Reynolds (momentum / viscosity)
    physics['reynolds'] = np.abs(physics['v']) / (physics['viscosity'] + 1e-10)
    
    # Lyapunov proxy (sensitivity to initial conditions)
    physics['lyapunov_proxy'] = np.abs(physics['a']) / (np.abs(physics['v']) + 1e-10)
    physics['lyap_z'] = (physics['lyapunov_proxy'] - physics['lyapunov_proxy'].rolling(lookback).mean()) / (physics['lyapunov_proxy'].rolling(lookback).std() + 1e-10)
    physics['local_dim'] = 2.0 + physics['lyapunov_proxy'].clip(-1, 1)
    
    # Tail risk (CVaR approximation)
    physics['cvar_95'] = pd.Series(returns).rolling(lookback).apply(lambda x: x[x < x.quantile(0.05)].mean() if len(x[x < x.quantile(0.05)]) > 0 else 0, raw=False).fillna(0).values
    down_returns = np.minimum(returns, 0)
    up_returns = np.maximum(returns, 0)
    physics['cvar_asymmetry'] = pd.Series(np.abs(down_returns)).rolling(lookback).mean().fillna(0).values / (pd.Series(np.abs(up_returns)).rolling(lookback).mean().fillna(1e-10).values + 1e-10)
    
    # Higher moments
    physics['skewness'] = pd.Series(returns).rolling(lookback).skew().fillna(0).values
    physics['kurtosis'] = pd.Series(returns).rolling(lookback).kurt().fillna(0).values
    physics['kurtosis_z'] = (physics['kurtosis'] - physics['kurtosis'].rolling(100).mean()) / (physics['kurtosis'].rolling(100).std() + 1e-10)
    physics['skewness_z'] = (physics['skewness'] - physics['skewness'].rolling(100).mean()) / (physics['skewness'].rolling(100).std() + 1e-10)
    
    # Momentum
    physics['roc'] = pd.Series(close).pct_change(lookback).fillna(0).values
    physics['momentum_strength'] = physics['roc'] / (physics['vol_rs'] + 1e-10)
    
    # Composites
    physics['composite_jerk_entropy'] = physics['jerk_z'] * physics['entropy_z']
    physics['stack_jerk_entropy'] = (physics['jerk_z'] + physics['entropy_z']) / 2
    physics['stack_jerk_lyap'] = (physics['jerk_z'] + physics['lyap_z']) / 2
    physics['triple_stack'] = (physics['jerk_z'] + physics['entropy_z'] + physics['lyap_z']) / 3
    
    # Percentiles (empirical CDF)
    for col in ['energy', 'damping', 'entropy', 'lyapunov_proxy', 'local_dim', 'cvar_95', 'cvar_asymmetry', 
                'composite_jerk_entropy', 'triple_stack', 'PE', 'reynolds', 'eta']:
        if col in physics.columns:
            physics[f'{col}_pct'] = physics[col].rolling(500, min_periods=50).rank(pct=True).fillna(0.5).values
    
    # Regime detection (simple)
    physics['regime'] = 'LAMINAR'
    physics.loc[physics['vol_rs'] > physics['vol_rs'].rolling(100).quantile(0.8), 'regime'] = 'BREAKOUT'
    physics.loc[physics['damping'] > physics['damping'].rolling(100).quantile(0.8), 'regime'] = 'OVERDAMPED'
    physics.loc[(physics['momentum_strength'].abs() > 1) & (physics['vol_rs'] < physics['vol_rs'].rolling(100).quantile(0.5)), 'regime'] = 'UNDERDAMPED'
    
    physics['regime_age_frac'] = 0.5  # Placeholder
    
    # Additional features
    physics['adaptive_trail_mult'] = 2.0 + physics['vol_rs_z'].clip(-1, 1)
    physics['vol_ratio_yz_rs'] = physics['vol_yz'] / (physics['vol_rs'] + 1e-10)
    physics['vol_term_structure'] = 1.0
    
    # DSP placeholders
    physics['dsp_roofing'] = physics['v'].rolling(5).mean() - physics['v'].rolling(20).mean()
    physics['dsp_roofing_z'] = (physics['dsp_roofing'] - physics['dsp_roofing'].rolling(100).mean()) / (physics['dsp_roofing'].rolling(100).std() + 1e-10)
    physics['dsp_trend'] = np.sign(physics['roc'])
    physics['dsp_trend_dir'] = physics['dsp_trend']
    physics['dsp_cycle_period'] = 24
    
    # VPIN proxy (volume imbalance)
    vol_ma = pd.Series(volume).rolling(lookback).mean().fillna(1).values
    physics['vpin'] = 0.5 + 0.5 * np.sign(returns) * (volume / (vol_ma + 1e-10))
    physics['vpin_z'] = (physics['vpin'] - 0.5) / 0.1
    physics['vpin_pct'] = physics['vpin'].rolling(500, min_periods=50).rank(pct=True).fillna(0.5).values
    physics['buy_pressure'] = (physics['vpin'] > 0.5).astype(float)
    
    # Tail risk
    physics['tail_risk'] = physics['cvar_95'].abs() * physics['kurtosis'].clip(0, 10)
    physics['jb_proxy_z'] = physics['skewness_z'] ** 2 + 0.25 * physics['kurtosis_z'] ** 2
    
    return physics.fillna(0)


# =============================================================================
# AGENTS
# =============================================================================

class LinearQAgent:
    """Linear Q-learning with feature tracking."""
    
    def __init__(self, state_dim: int = 64, n_actions: int = 4, lr: float = 0.01):
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.lr = lr
        self.W = np.random.randn(state_dim, n_actions) * 0.1  # Larger initial weights
        self.b = np.zeros(n_actions)
        self.b[1] = 0.1  # Slight bias toward buy
        self.b[2] = 0.1  # Slight bias toward sell
        self.epsilon = 0.5  # Start with more exploration
        self.epsilon_decay = 0.99
        self.epsilon_min = 0.05
        self.gamma = 0.95
        self.name = "LinearQ"
        
        # Track feature importance
        self.feature_usage = np.zeros(state_dim)
    
    def get_q_values(self, state: np.ndarray) -> np.ndarray:
        return state @ self.W + self.b
    
    def act(self, state: np.ndarray, training: bool = True) -> int | bool | bool | unsignedinteger[_8Bit] | \
                                                               unsignedinteger[_16Bit] | unsignedinteger[_32Bit] | \
                                                               unsignedinteger[_32Bit | _64Bit] | unsignedinteger[
                                                                   _64Bit] | signedinteger[_8Bit] | signedinteger[
                                                                   _16Bit] | signedinteger[_32Bit] | signedinteger[
                                                                   _32Bit | _64Bit] | signedinteger[_64Bit] | ndarray[
                                                                   tuple[Any, ...], dtype[
                                                                       signedinteger[_32Bit | _64Bit]]] | ndarray[
                                                                   tuple[Any, ...], dtype[bool]] | ndarray[
                                                                   tuple[Any, ...], dtype[signedinteger[_8Bit]]] | \
                                                               ndarray[tuple[Any, ...], dtype[signedinteger[_16Bit]]] | \
                                                               ndarray[tuple[Any, ...], dtype[signedinteger[_32Bit]]] | \
                                                               ndarray[tuple[Any, ...], dtype[signedinteger[_64Bit]]] | \
                                                               ndarray[tuple[Any, ...], dtype[unsignedinteger[_8Bit]]] | \
                                                               ndarray[
                                                                   tuple[Any, ...], dtype[unsignedinteger[_16Bit]]] | \
                                                               ndarray[
                                                                   tuple[Any, ...], dtype[unsignedinteger[_32Bit]]] | \
                                                               ndarray[
                                                                   tuple[Any, ...], dtype[unsignedinteger[_64Bit]]] | \
                                                               ndarray[tuple[Any, ...], dtype[
                                                                   unsignedinteger[_32Bit | _64Bit]]]:
        if training and np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        
        q_values = self.get_q_values(state)
        
        # Track which features contributed most
        if not training:
            for a in range(self.n_actions):
                contributions = np.abs(state * self.W[:, a])
                self.feature_usage += contributions
        
        return np.argmax(q_values)
    
    def update(self, state, action, reward, next_state, done):
        q_values = self.get_q_values(state)
        next_q = self.get_q_values(next_state)
        target = reward + (0 if done else self.gamma * np.max(next_q))
        td_error = target - q_values[action]
        
        self.W[:, action] += self.lr * td_error * state
        self.b[action] += self.lr * td_error
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def get_top_features(self, feature_names: list, top_k: int = 10) -> list:
        """Get top features by usage."""
        if self.feature_usage.sum() == 0:
            return []
        importance = self.feature_usage / (self.feature_usage.sum() + 1e-10)
        indices = np.argsort(importance)[::-1][:top_k]
        return [(feature_names[i], importance[i]) for i in indices]


class PPOAgent:
    """Simple PPO-like agent (no PyTorch required)."""
    
    def __init__(self, state_dim: int = 64, n_actions: int = 4, lr: float = 0.001):
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.lr = lr
        
        # Policy network (linear)
        self.policy_W = np.random.randn(state_dim, n_actions) * 0.01
        self.policy_b = np.zeros(n_actions)
        
        # Value network (linear)
        self.value_W = np.random.randn(state_dim) * 0.01
        self.value_b = 0.0
        
        self.gamma = 0.99
        self.clip_ratio = 0.2
        self.name = "PPO"
        
        # Trajectory buffer
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
    
    def _softmax(self, x):
        x = np.nan_to_num(x, nan=0.0, posinf=10, neginf=-10)
        x = np.clip(x, -10, 10)
        exp_x = np.exp(x - np.max(x))
        result = exp_x / (exp_x.sum() + 1e-10)
        if np.any(np.isnan(result)) or np.any(result < 0):
            return np.ones(len(x)) / len(x)
        return result
    
    def get_action_probs(self, state: np.ndarray) -> np.ndarray:
        logits = state @ self.policy_W + self.policy_b
        return self._softmax(logits)
    
    def get_value(self, state: np.ndarray) -> float:
        return float(state @ self.value_W + self.value_b)
    
    def act(self, state: np.ndarray, training: bool = True) -> int | ndarray[
        tuple[Any, ...], dtype[signedinteger[_32Bit | _64Bit]]] | ndarray[tuple[Any, ...], dtype[Any]] | signedinteger[
                                                                   _32Bit | _64Bit] | Any:
        probs = self.get_action_probs(state)
        if training:
            action = np.random.choice(self.n_actions, p=probs)
            self.states.append(state)
            self.actions.append(action)
            self.log_probs.append(np.log(probs[action] + 1e-10))
            return action
        return np.argmax(probs)
    
    def store_reward(self, reward):
        self.rewards.append(reward)
    
    def update(self, state=None, action=None, reward=None, next_state=None, done=False):
        """PPO update at episode end."""
        if not done or len(self.states) < 2:
            if reward is not None:
                self.store_reward(reward)
            return
        
        # Compute returns
        returns = []
        G = 0
        for r in reversed(self.rewards):
            G = r + self.gamma * G
            returns.insert(0, G)
        returns = np.array(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-10)
        
        # PPO update
        for i, (s, a, old_log_prob, R) in enumerate(zip(self.states, self.actions, self.log_probs, returns)):
            # Advantage
            value = self.get_value(s)
            advantage = R - value
            
            # Policy gradient
            probs = self.get_action_probs(s)
            new_log_prob = np.log(probs[a] + 1e-10)
            ratio = np.exp(new_log_prob - old_log_prob)
            
            # Clipped objective
            clipped_ratio = np.clip(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio)
            policy_loss = -min(ratio * advantage, clipped_ratio * advantage)
            
            # Update policy
            grad = np.zeros(self.n_actions)
            grad[a] = policy_loss
            grad_W = np.outer(s, grad)
            self.policy_W -= self.lr * grad_W
            self.policy_b -= self.lr * grad
            
            # Update value
            value_loss = (R - value)
            self.value_W += self.lr * value_loss * s
            self.value_b += self.lr * value_loss
        
        # Clear buffers
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []


class SACAgent:
    """Simple SAC-like agent with entropy bonus."""
    
    def __init__(self, state_dim: int = 64, n_actions: int = 4, lr: float = 0.001):
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.lr = lr
        
        # Q networks (2 for twin Q)
        self.Q1_W = np.random.randn(state_dim, n_actions) * 0.01
        self.Q1_b = np.zeros(n_actions)
        self.Q2_W = np.random.randn(state_dim, n_actions) * 0.01
        self.Q2_b = np.zeros(n_actions)
        
        # Policy
        self.policy_W = np.random.randn(state_dim, n_actions) * 0.01
        self.policy_b = np.zeros(n_actions)
        
        self.gamma = 0.99
        self.alpha = 0.2  # Entropy coefficient
        self.name = "SAC"
        
        # Replay buffer
        self.buffer = []
        self.buffer_size = 10000
    
    def _softmax(self, x):
        x = np.nan_to_num(x, nan=0.0, posinf=10, neginf=-10)
        x = np.clip(x, -10, 10)  # Prevent overflow
        exp_x = np.exp(x - np.max(x))
        result = exp_x / (exp_x.sum() + 1e-10)
        # Ensure valid probabilities
        if np.any(np.isnan(result)) or np.any(result < 0):
            return np.ones(len(x)) / len(x)  # Uniform fallback
        return result
    
    def act(self, state: np.ndarray, training: bool = True) -> int | ndarray[
        tuple[Any, ...], dtype[signedinteger[_32Bit | _64Bit]]] | ndarray[tuple[Any, ...], dtype[Any]] | signedinteger[
                                                                   _32Bit | _64Bit] | Any:
        logits = state @ self.policy_W + self.policy_b
        probs = self._softmax(logits)
        
        if training:
            return np.random.choice(self.n_actions, p=probs)
        return np.argmax(probs)
    
    def update(self, state, action, reward, next_state, done):
        # Store transition
        self.buffer.append((state, action, reward, next_state, done))
        if len(self.buffer) > self.buffer_size:
            self.buffer.pop(0)
        
        if len(self.buffer) < 32:
            return
        
        # Sample batch
        indices = np.random.choice(len(self.buffer), 32, replace=False)
        batch = [self.buffer[i] for i in indices]
        
        for s, a, r, ns, d in batch:
            # Q targets
            next_probs = self._softmax(ns @ self.policy_W + self.policy_b)
            next_q1 = ns @ self.Q1_W + self.Q1_b
            next_q2 = ns @ self.Q2_W + self.Q2_b
            next_v = (next_probs * (np.minimum(next_q1, next_q2) - self.alpha * np.log(next_probs + 1e-10))).sum()
            
            target = r + (0 if d else self.gamma * next_v)
            
            # Update Q networks
            q1 = s @ self.Q1_W + self.Q1_b
            q2 = s @ self.Q2_W + self.Q2_b
            
            td1 = target - q1[a]
            td2 = target - q2[a]
            
            self.Q1_W[:, a] += self.lr * td1 * s
            self.Q1_b[a] += self.lr * td1
            self.Q2_W[:, a] += self.lr * td2 * s
            self.Q2_b[a] += self.lr * td2
            
            # Update policy (maximize Q + entropy)
            probs = self._softmax(s @ self.policy_W + self.policy_b)
            q_min = np.minimum(q1, q2)
            policy_target = q_min - self.alpha * np.log(probs + 1e-10)
            
            for act in range(self.n_actions):
                grad = probs[act] * (policy_target[act] - (probs * policy_target).sum())
                self.policy_W[:, act] += self.lr * 0.1 * grad * s
                self.policy_b[act] += self.lr * 0.1 * grad


# =============================================================================
# ENVIRONMENT
# =============================================================================

class TradingEnv:
    """Trading environment with physics state."""
    
    def __init__(self, data: pd.DataFrame, physics_state: pd.DataFrame, spread_pct: float = 0.0001):
        self.data = data
        self.physics_state = physics_state
        self.spread_pct = spread_pct
        
        self.current_bar = 0
        self.position = 0
        self.entry_price = 0
        self.entry_bar = 0
        self.balance = 10000
        self.trades = []
        
    def reset(self, start_bar: int = 100) -> np.ndarray:
        self.current_bar = start_bar
        self.position = 0
        self.entry_price = 0
        self.balance = 10000
        self.trades = []
        state = get_rl_state_features(self.physics_state, self.current_bar)
        return np.nan_to_num(state, nan=0.0, posinf=0.0, neginf=0.0)
    
    def step(self, action: int) -> tuple:
        """Actions: 0=hold, 1=buy, 2=sell, 3=close"""
        price = self.data.iloc[self.current_bar]['close']
        reward = 0
        done = False
        
        # Execute
        if action == 1 and self.position == 0:
            self.position = 1
            self.entry_price = price * (1 + self.spread_pct)
            self.entry_bar = self.current_bar
            
        elif action == 2 and self.position == 0:
            self.position = -1
            self.entry_price = price * (1 - self.spread_pct)
            self.entry_bar = self.current_bar
            
        elif action == 3 and self.position != 0:
            if self.position == 1:
                pnl_pct = (price - self.entry_price) / self.entry_price
            else:
                pnl_pct = (self.entry_price - price) / self.entry_price
            
            reward = pnl_pct * 100
            pnl_dollars = self.balance * 0.1 * pnl_pct
            self.balance += pnl_dollars
            self.trades.append({
                'pnl': pnl_dollars,
                'bars_held': self.current_bar - self.entry_bar,
                'direction': 'long' if self.position == 1 else 'short',
            })
            self.position = 0
        
        # Small holding penalty
        if self.position != 0:
            reward -= 0.001
        
        self.current_bar += 1
        
        if self.current_bar >= len(self.data) - 1:
            done = True
            if self.position != 0:
                price = self.data.iloc[self.current_bar]['close']
                if self.position == 1:
                    pnl_pct = (price - self.entry_price) / self.entry_price
                else:
                    pnl_pct = (self.entry_price - price) / self.entry_price
                reward += pnl_pct * 100
                self.balance += self.balance * 0.1 * pnl_pct
        
        next_state = get_rl_state_features(self.physics_state, self.current_bar)
        next_state = np.nan_to_num(next_state, nan=0.0, posinf=0.0, neginf=0.0)
        return next_state, reward, done, {}


# =============================================================================
# TRAINING
# =============================================================================

def train_and_evaluate(agent, env: TradingEnv, train_episodes: int = 30, eval_episodes: int = 5):
    """Train agent and evaluate."""
    
    # Training
    for ep in range(train_episodes):
        start = np.random.randint(100, len(env.data) - 600)
        state = env.reset(start_bar=start)
        
        for step in range(500):
            action = agent.act(state, training=True)
            next_state, reward, done, _ = env.step(action)
            agent.update(state, action, reward, next_state, done)
            state = next_state
            if done:
                break
    
    # Evaluation
    returns = []
    all_trades = []
    
    for ep in range(eval_episodes):
        start = np.random.randint(100, len(env.data) - 600)
        state = env.reset(start_bar=start)
        
        for _ in range(500):
            action = agent.act(state, training=False)
            next_state, reward, done, _ = env.step(action)
            state = next_state
            if done:
                break
        
        ret = ((env.balance - 10000) / 10000) * 100
        returns.append(ret)
        all_trades.extend(env.trades)
    
    wins = [t['pnl'] for t in all_trades if t['pnl'] > 0]
    losses = [t['pnl'] for t in all_trades if t['pnl'] <= 0]
    
    return {
        'avg_return': np.mean(returns),
        'std_return': np.std(returns),
        'total_trades': len(all_trades),
        'win_rate': len(wins) / len(all_trades) * 100 if all_trades else 0,
        'profit_factor': abs(sum(wins) / sum(losses)) if losses and sum(losses) != 0 else float('inf'),
        'avg_bars_held': np.mean([t['bars_held'] for t in all_trades]) if all_trades else 0,
    }


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("PATHFINDER EXPLORATION")
    print("=" * 70)
    print(f"Started: {datetime.now()}")
    print("\n'We don't know what we don't know' - exploring without assumptions\n")
    
    data_dir = Path("/workspace/data/runs/berserker_run3/data")
    
    # Test multiple symbols
    test_files = [
        ("XAUUSD", "XAUUSD+_H1_202401020100_202512262300.csv"),
        ("BTCUSD", "BTCUSD_H1_202401020000_202512282200.csv"),
        ("GBPUSD", "GBPUSD+_H1_202401020000_202512262300.csv"),
        ("NAS100", "NAS100_H1_202401020100_202512262300.csv"),
    ]
    
    all_results = []
    feature_names = get_rl_feature_names()
    
    for symbol, filename in test_files:
        filepath = data_dir / filename
        if not filepath.exists():
            print(f"⚠ {filename} not found")
            continue
        
        print(f"\n{'━' * 70}")
        print(f"Symbol: {symbol}")
        print(f"{'━' * 70}")
        
        # Load data
        data = load_csv_direct(filepath)
        data = data.iloc[-4000:]  # Last 4000 bars
        print(f"  Data: {len(data)} bars")
        
        # Compute physics
        print("  Computing physics state...")
        physics_state = compute_simple_physics(data)
        print(f"  Physics: {physics_state.shape[1]} features")
        
        # Create environment
        env = TradingEnv(data, physics_state)
        
        # Test agents
        agents = [
            LinearQAgent(state_dim=64, n_actions=4, lr=0.01),
            PPOAgent(state_dim=64, n_actions=4, lr=0.001),
            SACAgent(state_dim=64, n_actions=4, lr=0.001),
        ]
        
        for agent in agents:
            print(f"\n  Training {agent.name}...")
            start = time.time()
            
            metrics = train_and_evaluate(agent, env, train_episodes=30, eval_episodes=5)
            elapsed = time.time() - start
            
            print(f"    Time: {elapsed:.1f}s")
            print(f"    Return: {metrics['avg_return']:>7.2f}% ± {metrics['std_return']:.2f}%")
            print(f"    Trades: {metrics['total_trades']:>4} | Win Rate: {metrics['win_rate']:.1f}%")
            print(f"    Profit Factor: {metrics['profit_factor']:.2f} | Avg Hold: {metrics['avg_bars_held']:.1f} bars")
            
            # Feature importance (LinearQ only)
            if hasattr(agent, 'get_top_features'):
                top_features = agent.get_top_features(feature_names, top_k=5)
                if top_features:
                    print(f"    Top features: {', '.join([f[0] for f in top_features])}")
            
            all_results.append({
                'symbol': symbol,
                'agent': agent.name,
                **metrics,
            })
    
    # Summary
    print("\n" + "=" * 70)
    print("EXPLORATION SUMMARY")
    print("=" * 70)
    
    if all_results:
        df = pd.DataFrame(all_results)
        
        # By agent
        print("\n  BY AGENT (across all symbols):")
        agent_summary = df.groupby('agent').agg({
            'avg_return': 'mean',
            'win_rate': 'mean',
            'profit_factor': 'mean',
        }).round(2)
        
        print(f"\n  {'Agent':<10} {'Avg Return':>12} {'Win Rate':>12} {'Profit Factor':>14}")
        print("  " + "-" * 50)
        for agent_name, row in agent_summary.iterrows():
            pf = f"{row['profit_factor']:.2f}" if row['profit_factor'] < 100 else "∞"
            print(f"  {agent_name:<10} {row['avg_return']:>11.2f}% {row['win_rate']:>11.1f}% {pf:>14}")
        
        # Best overall
        best = df.loc[df['avg_return'].idxmax()]
        print(f"\n  🏆 Best: {best['agent']} on {best['symbol']} = {best['avg_return']:.2f}%")
        
        # By symbol
        print("\n  BY SYMBOL (best agent):")
        for symbol in df['symbol'].unique():
            sym_df = df[df['symbol'] == symbol]
            best_sym = sym_df.loc[sym_df['avg_return'].idxmax()]
            print(f"    {symbol}: {best_sym['agent']} = {best_sym['avg_return']:.2f}%")
    
    print(f"\n✅ Exploration complete at {datetime.now()}")
    print("\nTHE MARKET HAS SPOKEN - NOW WE KNOW!")


if __name__ == "__main__":
    main()
