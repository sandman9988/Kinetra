#!/usr/bin/env python3
"""
QUICK RL TEST - Test RL Agents Without Breaking
================================================

Uses cached data directly - shows results fast.
Tests LinearQ, and optionally PPO/SAC if available.
"""

import sys
import time
import warnings
from pathlib import Path
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
from numpy import bool, dtype, ndarray, signedinteger, unsignedinteger
from numpy._typing import _16Bit, _32Bit, _64Bit, _8Bit

sys.path.insert(0, str(Path(__file__).parent.parent))
warnings.filterwarnings('ignore')


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


def extract_features(data: pd.DataFrame, lookback: int = 20) -> np.ndarray:
    """Extract simple features for RL state."""
    features_list = []
    
    for i in range(len(data)):
        if i < lookback:
            features_list.append(np.zeros(32))
            continue
        
        window = data.iloc[i-lookback:i+1]
        close = window['close'].values
        volume = window['volume'].values
        high = window['high'].values
        low = window['low'].values
        
        # Returns
        returns = np.diff(close) / close[:-1]
        
        # Features (32-dim)
        features = [
            # Price momentum
            (close[-1] - close[0]) / close[0],  # Total return
            np.mean(returns),  # Mean return
            np.std(returns),  # Volatility
            returns[-1] if len(returns) > 0 else 0,  # Last return
            
            # Moving averages
            close[-1] / np.mean(close) - 1,  # Price vs MA
            np.mean(close[-5:]) / np.mean(close[-10:]) - 1 if len(close) >= 10 else 0,  # MA crossover
            
            # Range
            (high[-1] - low[-1]) / close[-1],  # Current range
            np.mean(high - low) / np.mean(close),  # Avg range
            
            # Volume
            volume[-1] / np.mean(volume) if np.mean(volume) > 0 else 1,  # Rel volume
            np.std(volume) / np.mean(volume) if np.mean(volume) > 0 else 0,  # Vol variability
            
            # Higher moments
            pd.Series(returns).skew() if len(returns) > 2 else 0,  # Skewness
            pd.Series(returns).kurtosis() if len(returns) > 3 else 0,  # Kurtosis
            
            # Z-scores
            (close[-1] - np.mean(close)) / (np.std(close) + 1e-10),  # Price z-score
            (returns[-1] - np.mean(returns)) / (np.std(returns) + 1e-10) if len(returns) > 0 else 0,  # Return z
            
            # Trend
            np.polyfit(range(len(close)), close, 1)[0] / close[-1] if len(close) > 1 else 0,  # Linear trend
            
            # Percentiles
            (np.sum(close < close[-1]) / len(close)),  # Price percentile
            (np.sum(volume < volume[-1]) / len(volume)) if len(volume) > 0 else 0.5,  # Volume pct
            
            # Additional features to reach 32
            np.max(returns) if len(returns) > 0 else 0,
            np.min(returns) if len(returns) > 0 else 0,
            np.max(close) / close[-1] - 1,  # Distance from high
            close[-1] / np.min(close) - 1,  # Distance from low
            np.mean(returns[-5:]) if len(returns) >= 5 else 0,  # Recent momentum
            np.std(returns[-5:]) if len(returns) >= 5 else 0,  # Recent vol
            (high[-1] - close[-1]) / (high[-1] - low[-1] + 1e-10),  # Upper shadow
            (close[-1] - low[-1]) / (high[-1] - low[-1] + 1e-10),  # Lower shadow
            np.corrcoef(range(len(close)), close)[0,1] if len(close) > 1 else 0,  # Trend strength
            volume[-1] / volume[-2] if len(volume) > 1 and volume[-2] > 0 else 1,  # Volume change
            close[-1] / close[-2] - 1 if len(close) > 1 else 0,  # 1-bar return
            close[-1] / close[-5] - 1 if len(close) > 5 else 0,  # 5-bar return
            close[-1] / close[-10] - 1 if len(close) > 10 else 0,  # 10-bar return
            np.mean(high[-5:]) / np.mean(low[-5:]) - 1 if len(high) >= 5 else 0,  # Range expansion
            np.std(close[-5:]) / np.std(close) if np.std(close) > 0 else 1,  # Vol ratio
        ]
        
        features_list.append(np.nan_to_num(features, nan=0, posinf=0, neginf=0))
    
    return np.array(features_list, dtype=np.float32)


# =============================================================================
# SIMPLE RL AGENTS
# =============================================================================

class LinearQAgent:
    """Simple linear Q-learning agent."""
    
    def __init__(self, state_dim: int = 32, n_actions: int = 4, lr: float = 0.01):
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.lr = lr
        
        # Linear weights
        self.W = np.random.randn(state_dim, n_actions) * 0.01
        self.b = np.zeros(n_actions)
        
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.gamma = 0.99
        
        self.name = "LinearQ"
    
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
        return np.argmax(self.get_q_values(state))
    
    def update(self, state, action, reward, next_state, done):
        q_values = self.get_q_values(state)
        next_q = self.get_q_values(next_state)
        
        target = reward + (0 if done else self.gamma * np.max(next_q))
        td_error = target - q_values[action]
        
        # Update weights
        self.W[:, action] += self.lr * td_error * state
        self.b[action] += self.lr * td_error
        
        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


class TabularQAgent:
    """Tabular Q-learning with state discretization."""
    
    def __init__(self, n_bins: int = 10, n_actions: int = 4, lr: float = 0.1):
        self.n_bins = n_bins
        self.n_actions = n_actions
        self.lr = lr
        self.gamma = 0.99
        
        self.Q = {}  # State -> Q-values dict
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        
        self.name = "TabularQ"
    
    def discretize(self, state: np.ndarray) -> tuple:
        # Use first 4 features discretized
        bins = np.linspace(-3, 3, self.n_bins)
        discrete = tuple(np.digitize(state[:4], bins))
        return discrete
    
    def get_q_values(self, state: np.ndarray) -> np.ndarray:
        key = self.discretize(state)
        if key not in self.Q:
            self.Q[key] = np.zeros(self.n_actions)
        return self.Q[key]
    
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
        return np.argmax(self.get_q_values(state))
    
    def update(self, state, action, reward, next_state, done):
        key = self.discretize(state)
        if key not in self.Q:
            self.Q[key] = np.zeros(self.n_actions)
        
        next_q = self.get_q_values(next_state)
        target = reward + (0 if done else self.gamma * np.max(next_q))
        
        self.Q[key][action] += self.lr * (target - self.Q[key][action])
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


# =============================================================================
# TRADING ENVIRONMENT
# =============================================================================

class TradingEnv:
    """Simple trading environment."""
    
    def __init__(self, data: pd.DataFrame, features: np.ndarray, spread_pct: float = 0.0001):
        self.data = data
        self.features = features
        self.spread_pct = spread_pct
        
        self.current_bar = 0
        self.position = 0  # -1, 0, 1
        self.entry_price = 0
        self.balance = 10000
        self.equity = 10000
        self.trades = []
        
    def reset(self, start_bar: int = 20) -> np.ndarray:
        self.current_bar = start_bar
        self.position = 0
        self.entry_price = 0
        self.balance = 10000
        self.equity = 10000
        self.trades = []
        return self.features[self.current_bar]
    
    def step(self, action: int) -> tuple:
        """
        Actions: 0=hold, 1=buy, 2=sell, 3=close
        """
        price = self.data.iloc[self.current_bar]['close']
        reward = 0
        done = False
        
        # Execute action
        if action == 1 and self.position == 0:  # Buy
            self.position = 1
            self.entry_price = price * (1 + self.spread_pct)
            
        elif action == 2 and self.position == 0:  # Sell short
            self.position = -1
            self.entry_price = price * (1 - self.spread_pct)
            
        elif action == 3 and self.position != 0:  # Close
            if self.position == 1:
                pnl_pct = (price - self.entry_price) / self.entry_price
            else:
                pnl_pct = (self.entry_price - price) / self.entry_price
            
            reward = pnl_pct * 100  # Scale reward
            pnl_dollars = self.balance * 0.1 * pnl_pct
            self.balance += pnl_dollars
            self.trades.append(pnl_dollars)
            
            self.position = 0
            self.entry_price = 0
        
        # Small penalty for holding
        if self.position != 0:
            reward -= 0.001
        
        # Move to next bar
        self.current_bar += 1
        
        if self.current_bar >= len(self.data) - 1:
            done = True
            # Close any open position
            if self.position != 0:
                price = self.data.iloc[self.current_bar]['close']
                if self.position == 1:
                    pnl_pct = (price - self.entry_price) / self.entry_price
                else:
                    pnl_pct = (self.entry_price - price) / self.entry_price
                reward += pnl_pct * 100
                self.balance += self.balance * 0.1 * pnl_pct
        
        next_state = self.features[min(self.current_bar, len(self.features)-1)]
        
        return next_state, reward, done, {}


# =============================================================================
# TRAINING
# =============================================================================

def train_agent(agent, env: TradingEnv, episodes: int = 20, max_steps: int = 1000,
                start_bars: list = None):
    """
    Train agent and return metrics.

    Args:
        agent: Agent to train
        env: Trading environment
        episodes: Number of training episodes
        max_steps: Max steps per episode
        start_bars: Fixed start positions for each episode (for fair comparison).
                   If None, uses random starts (NOT comparable across agents!)
    """

    episode_rewards = []
    episode_trades = []

    for ep in range(episodes):
        if start_bars is not None:
            start_bar = start_bars[ep]
        else:
            start_bar = np.random.randint(20, len(env.data) - max_steps - 10)
        state = env.reset(start_bar=start_bar)
        total_reward = 0
        
        for step in range(max_steps):
            action = agent.act(state, training=True)
            next_state, reward, done, _ = env.step(action)
            
            agent.update(state, action, reward, next_state, done)
            
            state = next_state
            total_reward += reward
            
            if done:
                break
        
        episode_rewards.append(total_reward)
        episode_trades.append(len(env.trades))
    
    return {
        'mean_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'total_trades': sum(episode_trades),
        'final_epsilon': getattr(agent, 'epsilon', 0),
    }


def evaluate_agent(agent, env: TradingEnv, episodes: int = 5, start_bars: list = None) -> dict:
    """
    Evaluate trained agent.

    Args:
        agent: Agent to evaluate
        env: Trading environment
        episodes: Number of evaluation episodes
        start_bars: Fixed start positions for each episode (for fair comparison).
                   If None, uses random starts (NOT comparable across agents!)
    """

    returns = []
    all_trades = []

    for ep in range(episodes):
        if start_bars is not None:
            start = start_bars[ep]
        else:
            start = np.random.randint(20, len(env.data) - 1000)
        state = env.reset(start_bar=start)
        
        for _ in range(1000):
            action = agent.act(state, training=False)
            next_state, reward, done, _ = env.step(action)
            state = next_state
            if done:
                break
        
        pct_return = ((env.balance - 10000) / 10000) * 100
        returns.append(pct_return)
        all_trades.extend(env.trades)
    
    wins = [t for t in all_trades if t > 0]
    losses = [t for t in all_trades if t <= 0]
    
    return {
        'avg_return': np.mean(returns),
        'std_return': np.std(returns),
        'total_trades': len(all_trades),
        'win_rate': len(wins) / len(all_trades) * 100 if all_trades else 0,
        'profit_factor': abs(sum(wins) / sum(losses)) if losses and sum(losses) != 0 else float('inf'),
    }


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("QUICK RL TEST - Train & Evaluate RL Agents")
    print("=" * 70)
    print(f"Started: {datetime.now()}\n")
    
    data_dir = Path("/workspace/data/runs/berserker_run3/data")
    
    test_files = [
        ("XAUUSD", "XAUUSD+_H1_202401020100_202512262300.csv"),
        ("BTCUSD", "BTCUSD_H1_202401020000_202512282200.csv"),
        ("GBPUSD", "GBPUSD+_H1_202401020000_202512262300.csv"),
    ]
    
    all_results = []
    
    for symbol, filename in test_files:
        filepath = data_dir / filename
        if not filepath.exists():
            continue
        
        print(f"\n{'━' * 70}")
        print(f"Symbol: {symbol}")
        print(f"{'━' * 70}")
        
        # Load data
        data = load_csv_direct(filepath)
        data = data.iloc[-3000:]  # Last 3000 bars for speed
        
        print(f"  Data: {len(data)} bars")
        
        # Extract features
        print("  Extracting features...")
        features = extract_features(data)
        print(f"  Features: {features.shape}")
        
        # Create environment
        env = TradingEnv(data, features)
        
        # Test agents
        agents = [
            LinearQAgent(state_dim=32, n_actions=4),
            TabularQAgent(n_bins=10, n_actions=4),
        ]
        
        for agent in agents:
            print(f"\n  Training {agent.name}...")
            start = time.time()
            
            # Train
            train_metrics = train_agent(agent, env, episodes=30, max_steps=500)
            train_time = time.time() - start
            
            # Evaluate
            eval_metrics = evaluate_agent(agent, env, episodes=5)
            
            print(f"    Train time: {train_time:.1f}s")
            print(f"    Avg Return: {eval_metrics['avg_return']:>7.2f}% ± {eval_metrics['std_return']:.2f}%")
            print(f"    Trades: {eval_metrics['total_trades']:>4} | Win Rate: {eval_metrics['win_rate']:.1f}%")
            
            all_results.append({
                'symbol': symbol,
                'agent': agent.name,
                'avg_return': eval_metrics['avg_return'],
                'std_return': eval_metrics['std_return'],
                'win_rate': eval_metrics['win_rate'],
                'trades': eval_metrics['total_trades'],
            })
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    if all_results:
        df = pd.DataFrame(all_results)
        
        print(f"\n{'Agent':<12} {'Symbol':<10} {'Return':>10} {'Win Rate':>10} {'Trades':>8}")
        print("-" * 55)
        for _, row in df.iterrows():
            print(f"{row['agent']:<12} {row['symbol']:<10} {row['avg_return']:>9.2f}% {row['win_rate']:>9.1f}% {row['trades']:>8}")
        
        # Best by agent type
        print("\n" + "─" * 55)
        for agent_name in df['agent'].unique():
            agent_df = df[df['agent'] == agent_name]
            avg = agent_df['avg_return'].mean()
            print(f"{agent_name} Average: {avg:.2f}%")
    
    print(f"\n✅ Completed at {datetime.now()}")


if __name__ == "__main__":
    main()
