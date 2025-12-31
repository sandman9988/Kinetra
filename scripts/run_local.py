#!/usr/bin/env python3
"""
KINETRA LOCAL RUNNER
====================
Save this as run_local.py in your Kinetra folder and run:
    python run_local.py

No external dependencies beyond numpy/pandas.
"""

import sys
import time
import warnings
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

print("""
╔═══════════════════════════════════════════════════════════════════╗
║           KINETRA PATHFINDER - LOCAL EXPLORATION                  ║
║                  "First in, find the path"                        ║
╚═══════════════════════════════════════════════════════════════════╝
""")

# =============================================================================
# FIND DATA
# =============================================================================

def find_data_dir():
    """Find the data directory."""
    possible_paths = [
        Path("data/runs/berserker_run3/data"),
        Path("data/master"),
        Path("data"),
        Path("../data"),
        Path.home() / "Kinetra/data/runs/berserker_run3/data",
        Path.home() / "Kinetra/data/master",
    ]
    
    for p in possible_paths:
        if p.exists() and list(p.glob("*.csv")):
            return p
    
    # Ask user
    print("Could not find data directory automatically.")
    user_path = input("Enter path to CSV data folder: ").strip()
    return Path(user_path)


def load_csv(filepath: Path) -> pd.DataFrame:
    """Load MT5 CSV."""
    df = pd.read_csv(filepath, sep='\t')
    df.columns = [c.strip('<>').lower() for c in df.columns]
    df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'])
    df.set_index('datetime', inplace=True)
    if 'tickvol' in df.columns:
        df = df.rename(columns={'tickvol': 'volume'})
    return df[['open', 'high', 'low', 'close', 'volume', 'spread']]


# =============================================================================
# FEATURES
# =============================================================================

def compute_features(df: pd.DataFrame, lookback: int = 20) -> np.ndarray:
    """32-dim features from OHLCV."""
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
        c_mean, c_std = np.mean(c), np.std(c) + 1e-10
        v_mean = np.mean(v) + 1e-10
        
        try:
            corr = np.corrcoef(np.arange(len(c)), c)[0, 1]
            corr = 0 if np.isnan(corr) else corr
        except:
            corr = 0
        
        f = np.array([
            (c[-1] - c[0]) / (c[0] + 1e-10),
            np.mean(ret), ret[-1], c[-1] / c_mean - 1,
            np.std(ret), c_std / c_mean,
            (h[-1] - l[-1]) / (c[-1] + 1e-10), np.mean(h - l) / c_mean,
            v[-1] / v_mean, np.std(v) / v_mean,
            (c[-1] - c_mean) / c_std, (ret[-1] - np.mean(ret)) / (np.std(ret) + 1e-10),
            corr, np.sum(c < c[-1]) / len(c), np.sum(v < v[-1]) / len(v),
            np.max(ret), np.min(ret), c[-1] / np.max(c) - 1, c[-1] / np.min(c) - 1,
            np.mean(ret[-5:]), np.std(ret[-5:]),
            (h[-1] - c[-1]) / (h[-1] - l[-1] + 1e-10),
            (c[-1] - l[-1]) / (h[-1] - l[-1] + 1e-10),
            c[-1] / c[-5] - 1, c[-1] / c[-10] - 1,
            0, 0,  # Placeholders for skew/kurt
            np.mean(h[-5:]) / np.mean(l[-5:]) - 1,
            v[-1] / (v[-2] + 1e-10), np.std(c[-5:]) / c_std,
            (c[-1] - c[-2]) / (c[-2] + 1e-10), 0
        ], dtype=np.float32)
        
        features[i] = np.nan_to_num(f, nan=0, posinf=0, neginf=0)
    
    return features


# =============================================================================
# TRADING ENV
# =============================================================================

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
        
        if action == 1 and self.position == 0:
            self.position = 1
            self.entry_price = price * (1 + self.spread_pct)
        elif action == 2 and self.position == 0:
            self.position = -1
            self.entry_price = price * (1 - self.spread_pct)
        elif action == 3 and self.position != 0:
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


# =============================================================================
# PPO AGENT
# =============================================================================

class PPOAgent:
    def __init__(self, state_dim=32, n_actions=4, lr=0.002):
        self.W = np.random.randn(state_dim, n_actions) * 0.1
        self.b = np.array([0, 0.05, 0.05, 0])
        self.V_W = np.random.randn(state_dim) * 0.1
        self.V_b = 0
        self.lr = lr
        self.gamma = 0.95
        self.states, self.actions, self.rewards, self.log_probs = [], [], [], []
    
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
        
        G, returns = 0, []
        for r in reversed(self.rewards):
            G = r + self.gamma * G
            returns.insert(0, G)
        returns = np.array(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-10)
        
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


# =============================================================================
# TRAIN & EVAL
# =============================================================================

def train_eval(agent, env, train_eps=50, eval_eps=10, max_steps=500):
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
    
    returns, trades = [], []
    for _ in range(eval_eps):
        s = env.reset(np.random.randint(50, len(env.data) - max_steps - 10))
        for _ in range(max_steps):
            a = agent.act(s, train=False)
            s, _, done = env.step(a)
            if done:
                break
        returns.append((env.balance - 10000) / 100)
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


# =============================================================================
# MAIN
# =============================================================================

def main():
    print(f"Started: {datetime.now()}\n")
    
    data_dir = find_data_dir()
    print(f"📁 Data directory: {data_dir}\n")
    
    csv_files = list(data_dir.glob("*_H1_*.csv")) + list(data_dir.glob("*_H4_*.csv"))
    if not csv_files:
        csv_files = list(data_dir.glob("*.csv"))[:10]
    
    print(f"Found {len(csv_files)} files to process\n")
    
    results = []
    
    for filepath in sorted(csv_files)[:10]:  # Limit to 10 for speed
        symbol = filepath.stem.split('_')[0]
        tf = filepath.stem.split('_')[1] if '_' in filepath.stem else 'H1'
        
        print(f"━━━ {symbol} {tf} ━━━")
        
        try:
            data = load_csv(filepath)
            data = data.iloc[-5000:]
            features = compute_features(data)
            
            env = TradingEnv(data, features)
            agent = PPOAgent(state_dim=32)
            
            start = time.time()
            metrics = train_eval(agent, env, train_eps=50, eval_eps=10)
            elapsed = time.time() - start
            
            status = "✓" if metrics['return'] > 0 else "○"
            print(f"  {status} Return: {metrics['return']:>6.2f}% | Win: {metrics['win_rate']:>5.1f}% | "
                  f"Trades: {metrics['trades']:>4} | {elapsed:.1f}s")
            
            results.append({'symbol': f"{symbol}_{tf}", **metrics})
            
        except Exception as e:
            print(f"  ✗ Error: {e}")
    
    # Summary
    if results:
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        
        df = pd.DataFrame(results).sort_values('return', ascending=False)
        
        print(f"\n{'Symbol':<15} {'Return':>8} {'Win%':>7} {'Trades':>7}")
        print("-" * 40)
        for _, r in df.iterrows():
            marker = "🏆" if r['return'] == df['return'].max() and r['return'] > 0 else "  "
            print(f"{marker}{r['symbol']:<13} {r['return']:>7.2f}% {r['win_rate']:>6.1f}% {r['trades']:>7}")
        
        positive = df[df['return'] > 0]
        if len(positive) > 0:
            print(f"\n✓ {len(positive)}/{len(df)} symbols profitable")
            print(f"✓ Best: {positive.iloc[0]['symbol']} = {positive.iloc[0]['return']:.2f}%")
        
        print(f"\n📊 Average: {df['return'].mean():.2f}%")
    
    print(f"\n✅ Complete at {datetime.now()}")
    print("\n" + "=" * 60)
    print("THE MARKET HAS SPOKEN - NOW WE KNOW!")
    print("=" * 60)


if __name__ == "__main__":
    main()
