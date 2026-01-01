#!/usr/bin/env python3
"""
STANDALONE EXPLORER - No broken dependencies
=============================================
Just works. 4 agents. All your data files.

python scripts/explorer_standalone.py
"""

import os
import sys
import time
import warnings
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

# =============================================================================
# CONFIG
# =============================================================================

DATA_PATHS = [
    "data/runs/berserker_run3/data",
    "data/master",
    "data",
]

TRAIN_EPISODES = 30
EVAL_EPISODES = 5
MAX_STEPS = 500
MAX_FILES = 50  # Limit for speed

# =============================================================================
# DATA LOADING
# =============================================================================

def find_data_dir():
    for p in DATA_PATHS:
        path = Path(p)
        if path.exists() and list(path.glob("*.csv")):
            return path
    # Try absolute
    for p in DATA_PATHS:
        path = Path.home() / "Kinetra" / p
        if path.exists() and list(path.glob("*.csv")):
            return path
    print("ERROR: No data directory found")
    print("Looked in:", DATA_PATHS)
    sys.exit(1)


def load_csv(filepath):
    df = pd.read_csv(filepath, sep='\t')
    df.columns = [c.strip('<>').lower() for c in df.columns]
    df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'])
    df.set_index('datetime', inplace=True)
    if 'tickvol' in df.columns:
        df = df.rename(columns={'tickvol': 'volume'})
    return df[['open', 'high', 'low', 'close', 'volume', 'spread']]


def compute_features(df, lookback=20):
    n = len(df)
    features = np.zeros((n, 32), dtype=np.float32)
    
    c = df['close'].values.astype(np.float64)
    h = df['high'].values.astype(np.float64)
    l = df['low'].values.astype(np.float64)
    v = df['volume'].values.astype(np.float64)
    
    for i in range(lookback, n):
        cw = c[i-lookback:i+1]
        hw = h[i-lookback:i+1]
        lw = l[i-lookback:i+1]
        vw = v[i-lookback:i+1]
        
        ret = np.diff(cw) / (cw[:-1] + 1e-10)
        cm = np.mean(cw)
        cs = np.std(cw) + 1e-10
        vm = np.mean(vw) + 1e-10
        rm = np.mean(ret)
        rs = np.std(ret) + 1e-10
        
        f = np.zeros(32)
        f[0] = (cw[-1] - cw[0]) / (cw[0] + 1e-10)
        f[1] = rm
        f[2] = ret[-1]
        f[3] = cw[-1] / cm - 1
        f[4] = np.std(ret)
        f[5] = cs / cm
        f[6] = (hw[-1] - lw[-1]) / (cw[-1] + 1e-10)
        f[7] = np.mean(hw - lw) / cm
        f[8] = vw[-1] / vm
        f[9] = np.std(vw) / vm
        f[10] = (cw[-1] - cm) / cs
        f[11] = (ret[-1] - rm) / rs
        f[12] = np.sum(cw < cw[-1]) / len(cw)
        f[13] = np.sum(vw < vw[-1]) / len(vw)
        f[14] = np.max(ret)
        f[15] = np.min(ret)
        f[16] = cw[-1] / np.max(cw) - 1
        f[17] = cw[-1] / np.min(cw) - 1
        f[18] = np.mean(ret[-5:]) if len(ret) >= 5 else 0
        f[19] = np.std(ret[-5:]) if len(ret) >= 5 else 0
        f[20] = (hw[-1] - cw[-1]) / (hw[-1] - lw[-1] + 1e-10)
        f[21] = (cw[-1] - lw[-1]) / (hw[-1] - lw[-1] + 1e-10)
        f[22] = cw[-1] / cw[-5] - 1 if len(cw) >= 5 else 0
        f[23] = cw[-1] / cw[-10] - 1 if len(cw) >= 10 else 0
        f[24] = np.mean(hw[-5:]) / (np.mean(lw[-5:]) + 1e-10) - 1
        f[25] = vw[-1] / (vw[-2] + 1e-10) if len(vw) >= 2 else 1
        f[26] = np.std(cw[-5:]) / cs if len(cw) >= 5 else 1
        f[27] = (cw[-1] - cw[-2]) / (cw[-2] + 1e-10) if len(cw) >= 2 else 0
        
        features[i] = np.nan_to_num(f, nan=0, posinf=0, neginf=0)
    
    return features


# =============================================================================
# ENVIRONMENT
# =============================================================================

class TradingEnv:
    def __init__(self, data, features, spread_pct=0.0001):
        self.data = data
        self.features = features
        self.spread_pct = spread_pct
        self.n = len(data)
    
    def reset(self, start=50):
        self.bar = start
        self.pos = 0
        self.entry = 0
        self.balance = 10000
        self.trades = []
        return self.features[self.bar]
    
    def step(self, action):
        price = self.data.iloc[self.bar]['close']
        reward = 0
        
        if action == 1 and self.pos == 0:  # Buy
            self.pos = 1
            self.entry = price * (1 + self.spread_pct)
        elif action == 2 and self.pos == 0:  # Sell
            self.pos = -1
            self.entry = price * (1 - self.spread_pct)
        elif action == 3 and self.pos != 0:  # Close
            pnl = ((price - self.entry) / self.entry) * self.pos
            reward = pnl * 100
            self.balance *= (1 + pnl * 0.1)
            self.trades.append(pnl * 10000 * 0.1)
            self.pos = 0
        
        if self.pos != 0:
            reward -= 0.001
        
        self.bar += 1
        done = self.bar >= self.n - 1
        
        if done and self.pos != 0:
            price = self.data.iloc[self.bar]['close']
            pnl = ((price - self.entry) / self.entry) * self.pos
            self.balance *= (1 + pnl * 0.1)
            self.trades.append(pnl * 10000 * 0.1)
        
        state = self.features[min(self.bar, len(self.features)-1)]
        return state, reward, done


# =============================================================================
# AGENTS
# =============================================================================

class LinearQAgent:
    name = "LinearQ"
    
    def __init__(self, sd=32, na=4):
        self.W = np.random.randn(sd, na) * 0.1
        self.b = np.array([0, 0.02, 0.02, 0])
        self.lr = 0.01
        self.gamma = 0.95
        self.eps = 0.3
    
    def act(self, s, train=True):
        if train and np.random.random() < self.eps:
            return np.random.randint(4)
        return np.argmax(s @ self.W + self.b)
    
    def update(self, s, a, r, ns, done):
        q = s @ self.W + self.b
        nq = ns @ self.W + self.b
        target = r + (0 if done else self.gamma * np.max(nq))
        td = target - q[a]
        self.W[:, a] += self.lr * td * s
        self.b[a] += self.lr * td
        self.eps = max(0.05, self.eps * 0.995)


class PPOAgent:
    name = "PPO"
    
    def __init__(self, sd=32, na=4):
        self.W = np.random.randn(sd, na) * 0.1
        self.b = np.array([0, 0.02, 0.02, 0])
        self.VW = np.random.randn(sd) * 0.1
        self.Vb = 0
        self.lr = 0.002
        self.gamma = 0.95
        self.buf = []
    
    def _soft(self, x):
        x = np.clip(x, -10, 10)
        e = np.exp(x - np.max(x))
        p = e / (e.sum() + 1e-10)
        if np.any(np.isnan(p)):
            return np.ones(4) / 4
        return p
    
    def act(self, s, train=True):
        p = self._soft(s @ self.W + self.b)
        if train:
            a = np.random.choice(4, p=p)
            self.buf.append((s, a, np.log(p[a] + 1e-10), 0))
            return a
        return np.argmax(p)
    
    def update(self, s, a, r, ns, done):
        if self.buf:
            self.buf[-1] = (self.buf[-1][0], self.buf[-1][1], self.buf[-1][2], r)
        
        if done and len(self.buf) > 1:
            G = 0
            returns = []
            for _, _, _, rr in reversed(self.buf):
                G = rr + self.gamma * G
                returns.insert(0, G)
            returns = np.array(returns)
            returns = (returns - returns.mean()) / (returns.std() + 1e-10)
            
            for (st, ac, lp, _), R in zip(self.buf, returns):
                adv = R - (st @ self.VW + self.Vb)
                p = self._soft(st @ self.W + self.b)
                ratio = np.exp(np.log(p[ac] + 1e-10) - lp)
                clip_ratio = np.clip(ratio, 0.8, 1.2)
                
                grad = np.zeros(4)
                grad[ac] = -min(ratio, clip_ratio) * adv
                self.W -= self.lr * np.outer(st, grad)
                self.b -= self.lr * grad
                self.VW += self.lr * 0.5 * adv * st
                self.Vb += self.lr * 0.5 * adv
            
            self.buf = []


class SACAgent:
    name = "SAC"
    
    def __init__(self, sd=32, na=4):
        self.Q1W = np.random.randn(sd, na) * 0.1
        self.Q1b = np.zeros(na)
        self.Q2W = np.random.randn(sd, na) * 0.1
        self.Q2b = np.zeros(na)
        self.PW = np.random.randn(sd, na) * 0.1
        self.Pb = np.array([0, 0.02, 0.02, 0])
        self.lr = 0.001
        self.gamma = 0.95
        self.alpha = 0.2
        self.buf = []
        self.buf_max = 5000
    
    def _soft(self, x):
        x = np.clip(x, -10, 10)
        e = np.exp(x - np.max(x))
        p = e / (e.sum() + 1e-10)
        if np.any(np.isnan(p)):
            return np.ones(4) / 4
        return p
    
    def act(self, s, train=True):
        p = self._soft(s @ self.PW + self.Pb)
        if train:
            return np.random.choice(4, p=p)
        return np.argmax(p)
    
    def update(self, s, a, r, ns, done):
        self.buf.append((s, a, r, ns, done))
        if len(self.buf) > self.buf_max:
            self.buf.pop(0)
        
        if len(self.buf) < 32:
            return
        
        idx = np.random.choice(len(self.buf), 32, replace=False)
        for i in idx:
            st, ac, rw, nst, dn = self.buf[i]
            
            np_ = self._soft(nst @ self.PW + self.Pb)
            nq1 = nst @ self.Q1W + self.Q1b
            nq2 = nst @ self.Q2W + self.Q2b
            nv = (np_ * (np.minimum(nq1, nq2) - self.alpha * np.log(np_ + 1e-10))).sum()
            target = rw + (0 if dn else self.gamma * nv)
            
            td1 = target - (st @ self.Q1W + self.Q1b)[ac]
            td2 = target - (st @ self.Q2W + self.Q2b)[ac]
            
            self.Q1W[:, ac] += self.lr * td1 * st
            self.Q1b[ac] += self.lr * td1
            self.Q2W[:, ac] += self.lr * td2 * st
            self.Q2b[ac] += self.lr * td2


class TD3Agent:
    name = "TD3"
    
    def __init__(self, sd=32, na=4):
        self.Q1W = np.random.randn(sd, na) * 0.1
        self.Q1b = np.zeros(na)
        self.Q2W = np.random.randn(sd, na) * 0.1
        self.Q2b = np.zeros(na)
        self.PW = np.random.randn(sd, na) * 0.1
        self.Pb = np.array([0, 0.02, 0.02, 0])
        self.lr = 0.001
        self.gamma = 0.95
        self.buf = []
        self.buf_max = 5000
        self.update_count = 0
        self.policy_delay = 2
    
    def _soft(self, x):
        x = np.clip(x, -10, 10)
        e = np.exp(x - np.max(x))
        p = e / (e.sum() + 1e-10)
        if np.any(np.isnan(p)):
            return np.ones(4) / 4
        return p
    
    def act(self, s, train=True):
        p = self._soft(s @ self.PW + self.Pb)
        if train and np.random.random() < 0.1:
            return np.random.randint(4)
        return np.argmax(p)
    
    def update(self, s, a, r, ns, done):
        self.buf.append((s, a, r, ns, done))
        if len(self.buf) > self.buf_max:
            self.buf.pop(0)
        
        if len(self.buf) < 32:
            return
        
        self.update_count += 1
        idx = np.random.choice(len(self.buf), 32, replace=False)
        
        for i in idx:
            st, ac, rw, nst, dn = self.buf[i]
            
            nq1 = nst @ self.Q1W + self.Q1b
            nq2 = nst @ self.Q2W + self.Q2b
            target = rw + (0 if dn else self.gamma * np.min([np.max(nq1), np.max(nq2)]))
            
            td1 = target - (st @ self.Q1W + self.Q1b)[ac]
            td2 = target - (st @ self.Q2W + self.Q2b)[ac]
            
            self.Q1W[:, ac] += self.lr * td1 * st
            self.Q1b[ac] += self.lr * td1
            self.Q2W[:, ac] += self.lr * td2 * st
            self.Q2b[ac] += self.lr * td2
        
        # Delayed policy update
        if self.update_count % self.policy_delay == 0:
            for i in idx[:16]:
                st = self.buf[i][0]
                q = np.minimum(st @ self.Q1W + self.Q1b, st @ self.Q2W + self.Q2b)
                best = np.argmax(q)
                p = self._soft(st @ self.PW + self.Pb)
                grad = -p.copy()
                grad[best] += 1
                self.PW += self.lr * 0.1 * np.outer(st, grad)
                self.Pb += self.lr * 0.1 * grad


# =============================================================================
# TRAINING
# =============================================================================

def train_eval(agent, env, train_eps, eval_eps, max_steps):
    # Train
    for _ in range(train_eps):
        start = np.random.randint(50, env.n - max_steps - 10)
        s = env.reset(start)
        for _ in range(max_steps):
            a = agent.act(s, train=True)
            ns, r, done = env.step(a)
            agent.update(s, a, r, ns, done)
            s = ns
            if done:
                break
    
    # Eval
    returns = []
    all_trades = []
    for _ in range(eval_eps):
        start = np.random.randint(50, env.n - max_steps - 10)
        s = env.reset(start)
        for _ in range(max_steps):
            a = agent.act(s, train=False)
            s, _, done = env.step(a)
            if done:
                break
        returns.append((env.balance - 10000) / 100)
        all_trades.extend(env.trades)
    
    wins = [t for t in all_trades if t > 0]
    losses = [t for t in all_trades if t <= 0]
    
    return {
        'return': np.mean(returns),
        'std': np.std(returns),
        'trades': len(all_trades),
        'win_rate': len(wins) / len(all_trades) * 100 if all_trades else 0,
        'pf': abs(sum(wins) / sum(losses)) if losses and sum(losses) != 0 else 999,
    }


def process_file(filepath):
    """Process a single file with all agents."""
    try:
        name = filepath.stem
        parts = name.split('_')
        symbol = parts[0]
        tf = parts[1] if len(parts) > 1 else 'H1'
        
        df = load_csv(filepath)
        if len(df) < 1000:
            return None
        
        df = df.iloc[-5000:]
        features = compute_features(df)
        env = TradingEnv(df, features)
        
        results = {'file': name, 'symbol': symbol, 'tf': tf}
        
        for AgentClass in [LinearQAgent, PPOAgent, SACAgent, TD3Agent]:
            agent = AgentClass()
            metrics = train_eval(agent, env, TRAIN_EPISODES, EVAL_EPISODES, MAX_STEPS)
            results[agent.name] = metrics['return']
            results[f'{agent.name}_win'] = metrics['win_rate']
            results[f'{agent.name}_trades'] = metrics['trades']
        
        return results
    except Exception as e:
        return {'file': str(filepath), 'error': str(e)}


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║                    KINETRA STANDALONE EXPLORER                       ║
║                   4 Agents × All Your Data Files                     ║
╚══════════════════════════════════════════════════════════════════════╝
""")
    
    start_time = time.time()
    print(f"Started: {datetime.now()}\n")
    
    # Find data
    data_dir = find_data_dir()
    print(f"📁 Data directory: {data_dir}")
    
    # Find all CSV files
    csv_files = sorted(list(data_dir.glob("**/*.csv")))[:MAX_FILES]
    print(f"📊 Found {len(csv_files)} files (processing up to {MAX_FILES})\n")
    
    if not csv_files:
        print("ERROR: No CSV files found!")
        return
    
    # Process files
    print("Processing files...\n")
    print(f"{'File':<30} {'LinearQ':>10} {'PPO':>10} {'SAC':>10} {'TD3':>10}")
    print("-" * 75)
    
    results = []
    for i, filepath in enumerate(csv_files):
        result = process_file(filepath)
        if result and 'error' not in result:
            best = max(result.get('LinearQ', -999), result.get('PPO', -999), 
                      result.get('SAC', -999), result.get('TD3', -999))
            marker = "★" if best > 0 else " "
            print(f"{marker} {result['file'][:28]:<28} "
                  f"{result.get('LinearQ', 0):>9.2f}% "
                  f"{result.get('PPO', 0):>9.2f}% "
                  f"{result.get('SAC', 0):>9.2f}% "
                  f"{result.get('TD3', 0):>9.2f}%")
            results.append(result)
        elif result:
            print(f"  {filepath.name[:28]:<28} ERROR: {result.get('error', 'unknown')[:30]}")
        
        # Progress
        if (i + 1) % 10 == 0:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed
            remaining = (len(csv_files) - i - 1) / rate if rate > 0 else 0
            print(f"  ... {i+1}/{len(csv_files)} files ({elapsed:.0f}s elapsed, ~{remaining:.0f}s remaining)")
    
    # Summary
    print("\n" + "=" * 75)
    print("SUMMARY")
    print("=" * 75)
    
    if results:
        df = pd.DataFrame(results)
        
        print("\n📈 AVERAGE RETURN BY AGENT:")
        for agent in ['LinearQ', 'PPO', 'SAC', 'TD3']:
            if agent in df.columns:
                avg = df[agent].mean()
                pos = (df[agent] > 0).sum()
                print(f"   {agent:<10}: {avg:>7.2f}% avg | {pos}/{len(df)} profitable")
        
        print("\n🏆 TOP 5 PERFORMERS:")
        df['best'] = df[['LinearQ', 'PPO', 'SAC', 'TD3']].max(axis=1)
        df['best_agent'] = df[['LinearQ', 'PPO', 'SAC', 'TD3']].idxmax(axis=1)
        top5 = df.nlargest(5, 'best')
        for _, row in top5.iterrows():
            print(f"   {row['file'][:25]:<25} {row['best_agent']:<8} {row['best']:>7.2f}%")
        
        print("\n📊 AGENT WINS (best performer per file):")
        wins = df['best_agent'].value_counts()
        for agent, count in wins.items():
            print(f"   {agent:<10}: {count} files ({count/len(df)*100:.0f}%)")
    
    elapsed = time.time() - start_time
    print(f"\n⏱️  Total time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"✅ Complete at {datetime.now()}")
    print("\n" + "=" * 75)
    print("THE MARKET HAS SPOKEN!")
    print("=" * 75)


if __name__ == "__main__":
    main()
