#!/usr/bin/env python3
"""
PATHFINDER DEEP DIVE
====================

Focus on promising signals. More episodes, more symbols, find the edge.
"""

import sys
import time
import warnings
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
warnings.filterwarnings('ignore')

# =============================================================================
# SIMPLE FUNCTIONS - NO EXTERNAL DEPS
# =============================================================================

def load_csv(filepath: Path) -> pd.DataFrame:
    df = pd.read_csv(filepath, sep='\t')
    df.columns = [c.strip('<>').lower() for c in df.columns]
    df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'])
    df.set_index('datetime', inplace=True)
    df = df.rename(columns={'tickvol': 'volume'})
    return df[['open', 'high', 'low', 'close', 'volume', 'spread']]


def compute_features(df: pd.DataFrame, lookback: int = 20) -> np.ndarray:
    """Compute 32-dim features directly from OHLCV."""
    n = len(df)
    features = np.zeros((n, 32), dtype=np.float32)
    
    close = df['close'].values.astype(float)
    high = df['high'].values.astype(float)
    low = df['low'].values.astype(float)
    volume = df['volume'].values.astype(float)
    
    for i in range(lookback, n):
        c = close[i-lookback:i+1]
        h = high[i-lookback:i+1]
        l = low[i-lookback:i+1]
        v = volume[i-lookback:i+1]
        
        ret = np.diff(c) / (c[:-1] + 1e-10)
        
        c_mean = np.mean(c)
        c_std = np.std(c) + 1e-10
        v_mean = np.mean(v) + 1e-10
        ret_mean = np.mean(ret)
        ret_std = np.std(ret) + 1e-10
        
        try:
            corr = np.corrcoef(np.arange(len(c)), c)[0, 1]
            if np.isnan(corr):
                corr = 0
        except:
            corr = 0
        
        try:
            skew = float(pd.Series(ret).skew())
            if np.isnan(skew):
                skew = 0
        except:
            skew = 0
        
        try:
            kurt = float(pd.Series(ret).kurtosis())
            if np.isnan(kurt):
                kurt = 0
        except:
            kurt = 0
        
        f = np.array([
            (c[-1] - c[0]) / (c[0] + 1e-10),
            ret_mean,
            ret[-1],
            c[-1] / c_mean - 1,
            np.std(ret),
            c_std / c_mean,
            (h[-1] - l[-1]) / (c[-1] + 1e-10),
            np.mean(h - l) / c_mean,
            v[-1] / v_mean,
            np.std(v) / v_mean,
            (c[-1] - c_mean) / c_std,
            (ret[-1] - ret_mean) / ret_std,
            corr,
            np.sum(c < c[-1]) / len(c),
            np.sum(v < v[-1]) / len(v),
            np.max(ret),
            np.min(ret),
            c[-1] / np.max(c) - 1,
            c[-1] / np.min(c) - 1,
            np.mean(ret[-5:]),
            np.std(ret[-5:]),
            (h[-1] - c[-1]) / (h[-1] - l[-1] + 1e-10),
            (c[-1] - l[-1]) / (h[-1] - l[-1] + 1e-10),
            c[-1] / c[-5] - 1,
            c[-1] / c[-10] - 1,
            skew,
            kurt,
            np.mean(h[-5:]) / np.mean(l[-5:]) - 1,
            v[-1] / (v[-2] + 1e-10),
            np.std(c[-5:]) / c_std,
            (c[-1] - c[-2]) / (c[-2] + 1e-10),
            0,  # Padding
        ], dtype=np.float32)
        
        features[i] = np.nan_to_num(f, nan=0, posinf=0, neginf=0)
    
    return features


class TradingEnv:
    def __init__(self, data, features, spread_pct=0.0001):
        self.data = data
        self.features = features
        self.spread_pct = spread_pct
        self.reset()
    
    def reset(self, start=50):
        self.bar = start
        self.position = 0
        self.entry_price = 0
        self.balance = 10000
        self.trades = []
        return self.features[self.bar]
    
    def step(self, action):
        price = self.data.iloc[self.bar]['close']
        reward = 0
        
        if action == 1 and self.position == 0:  # Buy
            self.position = 1
            self.entry_price = price * (1 + self.spread_pct)
        elif action == 2 and self.position == 0:  # Sell
            self.position = -1
            self.entry_price = price * (1 - self.spread_pct)
        elif action == 3 and self.position != 0:  # Close
            pnl = ((price - self.entry_price) / self.entry_price) * self.position
            reward = pnl * 100
            self.balance *= (1 + pnl * 0.1)
            self.trades.append(pnl * self.balance * 0.1)
            self.position = 0
        
        if self.position != 0:
            reward -= 0.001
        
        self.bar += 1
        done = self.bar >= len(self.data) - 1
        
        if done and self.position != 0:
            pnl = ((self.data.iloc[self.bar]['close'] - self.entry_price) / self.entry_price) * self.position
            self.balance *= (1 + pnl * 0.1)
        
        return self.features[min(self.bar, len(self.features)-1)], reward, done


class PPOAgent:
    def __init__(self, state_dim=32, n_actions=4, lr=0.002):
        self.W = np.random.randn(state_dim, n_actions) * 0.1
        self.b = np.array([0, 0.05, 0.05, 0])  # Bias toward trading
        self.V_W = np.random.randn(state_dim) * 0.1
        self.V_b = 0
        self.lr = lr
        self.gamma = 0.95
        self.states, self.actions, self.rewards, self.log_probs = [], [], [], []
        self.name = "PPO"
    
    def _softmax(self, x):
        x = np.clip(x, -10, 10)
        e = np.exp(x - np.max(x))
        return e / (e.sum() + 1e-10)
    
    def act(self, state, train=True):
        probs = self._softmax(state @ self.W + self.b)
        if train:
            a = np.random.choice(4, p=probs)
            self.states.append(state)
            self.actions.append(a)
            self.log_probs.append(np.log(probs[a] + 1e-10))
            return a
        return np.argmax(probs)
    
    def store(self, r):
        self.rewards.append(r)
    
    def update(self):
        if len(self.states) < 2:
            self.states, self.actions, self.rewards, self.log_probs = [], [], [], []
            return
        
        # Compute returns
        G, returns = 0, []
        for r in reversed(self.rewards):
            G = r + self.gamma * G
            returns.insert(0, G)
        returns = np.array(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-10)
        
        # Update
        for s, a, lp, R in zip(self.states, self.actions, self.log_probs, returns):
            adv = R - (s @ self.V_W + self.V_b)
            probs = self._softmax(s @ self.W + self.b)
            ratio = np.exp(np.log(probs[a] + 1e-10) - lp)
            
            grad = np.zeros(4)
            grad[a] = -min(ratio, np.clip(ratio, 0.8, 1.2)) * adv
            self.W -= self.lr * np.outer(s, grad)
            self.b -= self.lr * grad
            
            self.V_W += self.lr * adv * s
            self.V_b += self.lr * adv
        
        self.states, self.actions, self.rewards, self.log_probs = [], [], [], []


def train_eval(agent, env, train_eps=50, eval_eps=10, max_steps=500):
    # Train
    for _ in range(train_eps):
        s = env.reset(np.random.randint(50, len(env.data) - max_steps - 10))
        for _ in range(max_steps):
            a = agent.act(s, train=True)
            ns, r, done = env.step(a)
            agent.store(r)
            s = ns
            if done:
                break
        agent.update()
    
    # Eval
    returns, trades = [], []
    for _ in range(eval_eps):
        s = env.reset(np.random.randint(50, len(env.data) - max_steps - 10))
        for _ in range(max_steps):
            a = agent.act(s, train=False)
            s, _, done = env.step(a)
            if done:
                break
        ret = (env.balance - 10000) / 100
        returns.append(ret)
        trades.extend(env.trades)
    
    wins = [t for t in trades if t > 0]
    losses = [t for t in trades if t <= 0]
    
    return {
        'return': np.mean(returns),
        'std': np.std(returns),
        'trades': len(trades),
        'win_rate': len(wins) / len(trades) * 100 if trades else 0,
        'pf': abs(sum(wins) / sum(losses)) if losses and sum(losses) != 0 else 999,
    }


def main():
    print("=" * 60)
    print("PATHFINDER DEEP DIVE")
    print("=" * 60)
    print(f"Started: {datetime.now()}\n")
    
    data_dir = Path("/workspace/data/runs/berserker_run3/data")
    
    # All available H1 files
    h1_files = list(data_dir.glob("*_H1_*.csv"))
    print(f"Found {len(h1_files)} H1 datasets\n")
    
    results = []
    
    for filepath in sorted(h1_files):
        symbol = filepath.stem.split('_')[0]
        print(f"━━━ {symbol} ━━━")
        
        try:
            data = load_csv(filepath)
            data = data.iloc[-5000:]  # Last 5000 bars
            features = compute_features(data)
            
            env = TradingEnv(data, features)
            agent = PPOAgent(state_dim=32)
            
            start = time.time()
            metrics = train_eval(agent, env, train_eps=50, eval_eps=10)
            elapsed = time.time() - start
            
            print(f"  Return: {metrics['return']:>6.2f}% | Win: {metrics['win_rate']:>5.1f}% | "
                  f"Trades: {metrics['trades']:>4} | PF: {min(metrics['pf'], 99):>5.1f} | {elapsed:.1f}s")
            
            results.append({'symbol': symbol, **metrics})
            
        except Exception as e:
            print(f"  Error: {e}")
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY - RANKED BY RETURN")
    print("=" * 60)
    
    df = pd.DataFrame(results).sort_values('return', ascending=False)
    
    print(f"\n{'Symbol':<12} {'Return':>8} {'Win%':>7} {'Trades':>7} {'PF':>7}")
    print("-" * 45)
    for _, r in df.iterrows():
        pf = f"{min(r['pf'], 99):.1f}" if r['pf'] < 99 else "∞"
        print(f"{r['symbol']:<12} {r['return']:>7.2f}% {r['win_rate']:>6.1f}% {r['trades']:>7} {pf:>7}")
    
    # Top performers
    print(f"\n🏆 TOP 3:")
    for i, (_, r) in enumerate(df.head(3).iterrows()):
        print(f"  {i+1}. {r['symbol']}: {r['return']:.2f}% ({r['win_rate']:.0f}% win rate)")
    
    # Average
    print(f"\n📊 Average: {df['return'].mean():.2f}% | Win Rate: {df['win_rate'].mean():.1f}%")
    
    print(f"\n✅ Complete at {datetime.now()}")


if __name__ == "__main__":
    main()
