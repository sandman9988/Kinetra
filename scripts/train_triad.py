#!/usr/bin/env python3
"""
Triad Training System
=====================

Train Incumbent (PPO), Competitor (A2C), Researcher (SAC) on real data.

Core principle: Markets are about IMBALANCES
- No magic numbers (thresholds from rolling history)
- No linearity (only asymmetric features)
- No assumptions (let data reveal regimes)

Usage:
    python scripts/train_triad.py
    python scripts/train_triad.py --role trader --episodes 100
    python scripts/train_triad.py --role risk_manager --assets crypto forex
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import json
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from kinetra.triad_system import (
    TriadSystem, AgentRole, RegimeState, ImbalanceState,
    IncumbentAgent, CompetitorAgent, ResearcherAgent,
    ImbalanceExtractor, RegimeDetector, MetaController,
    create_triad_system
)


# =============================================================================
# DATA DISCOVERY
# =============================================================================

DATA_PATHS = [
    "data/master",
    "data/runs/berserker_run3/data",
    "data",
]


def discover_data_files() -> list:
    """Find all CSV data files."""
    files = []
    
    for base_path in DATA_PATHS:
        path = Path(base_path)
        if not path.exists():
            continue
        
        for csv_file in path.rglob("*.csv"):
            # Skip non-data files
            if 'symbol' in csv_file.name.lower() or 'info' in csv_file.name.lower():
                continue
            
            # Parse filename
            name = csv_file.stem
            parts = name.split('_')
            
            if len(parts) < 2:
                continue
            
            symbol = parts[0]
            timeframe = parts[1]
            
            # Classify asset class
            asset_class = classify_asset(symbol)
            
            files.append({
                'path': str(csv_file),
                'symbol': symbol,
                'timeframe': timeframe,
                'asset_class': asset_class,
            })
    
    return files


def classify_asset(symbol: str) -> str:
    """Classify symbol into asset class."""
    s = symbol.upper().replace('+', '').replace('-', '')
    
    if any(x in s for x in ['BTC', 'ETH', 'XRP', 'LTC']):
        return 'crypto'
    elif any(x in s for x in ['XAU', 'XAG', 'XPT', 'GOLD', 'SILVER', 'COPPER']):
        return 'metals'
    elif any(x in s for x in ['OIL', 'WTI', 'BRENT', 'GAS', 'GASOIL', 'UKOUSD']):
        return 'commodities'
    elif any(x in s for x in ['SPX', 'NAS', 'DOW', 'DJ', 'DAX', 'FTSE', 'NIKKEI', 
                               'US', 'GER', 'UK', 'SA', 'EU', '225', '100', '40', '30', '2000']):
        return 'indices'
    elif len(s) == 6 and s.isalpha():
        return 'forex'
    return 'unknown'


def load_data(filepath: str) -> pd.DataFrame:
    """Load and normalize CSV data."""
    df = pd.read_csv(filepath, sep='\t')
    
    # Normalize columns
    df.columns = [c.strip('<>').lower() for c in df.columns]
    
    # Required columns
    required = ['open', 'high', 'low', 'close']
    if not all(c in df.columns for c in required):
        # Try alternate separators
        df = pd.read_csv(filepath)
        df.columns = [c.strip('<>').lower() for c in df.columns]
    
    # Add volume if missing
    if 'volume' not in df.columns:
        if 'tickvol' in df.columns:
            df['volume'] = df['tickvol']
        else:
            df['volume'] = 1000
    
    # Ensure numeric
    for col in ['open', 'high', 'low', 'close', 'volume']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    df = df.dropna(subset=['open', 'high', 'low', 'close'])
    
    return df


# =============================================================================
# TRAINING ENVIRONMENT
# =============================================================================

class TriadTradingEnv:
    """
    Trading environment for Triad training.
    
    Actions: 0=Hold, 1=Buy, 2=Sell, 3=Close
    Rewards: Based on imbalance-aligned PnL
    """
    
    def __init__(self, df: pd.DataFrame, spread_pct: float = 0.0001):
        self.df = df
        self.spread_pct = spread_pct
        self.n_bars = len(df)
        
        # Imbalance extraction
        self.extractor = ImbalanceExtractor(lookback=50)
        
        # State
        self.bar = 0
        self.position = 0  # -1, 0, 1
        self.entry_price = 0
        self.balance = 10000
        self.peak_balance = 10000
        self.trades = []
    
    def reset(self, start_bar: int = 100) -> np.ndarray:
        """Reset environment."""
        self.bar = start_bar
        self.position = 0
        self.entry_price = 0
        self.balance = 10000
        self.peak_balance = 10000
        self.trades = []
        
        state = self.extractor.extract(self.df, self.bar)
        return state.to_array()
    
    def step(self, action: int) -> tuple:
        """Execute action, return (next_state, reward, done, info)."""
        price = self.df.iloc[self.bar]['close']
        reward = 0
        pnl = 0
        
        # Execute action
        if action == 1 and self.position == 0:  # Buy
            self.position = 1
            self.entry_price = price * (1 + self.spread_pct)
        
        elif action == 2 and self.position == 0:  # Sell
            self.position = -1
            self.entry_price = price * (1 - self.spread_pct)
        
        elif action == 3 and self.position != 0:  # Close
            if self.position == 1:
                pnl = (price * (1 - self.spread_pct) - self.entry_price) / self.entry_price
            else:
                pnl = (self.entry_price - price * (1 + self.spread_pct)) / self.entry_price
            
            self.balance *= (1 + pnl * 0.1)  # 10% of account per trade
            self.trades.append(pnl)
            self.position = 0
            reward = pnl * 100
        
        # Holding cost
        if self.position != 0:
            reward -= 0.001
        
        # Update peak for drawdown
        self.peak_balance = max(self.peak_balance, self.balance)
        
        # Advance bar
        self.bar += 1
        done = self.bar >= self.n_bars - 1
        
        # Force close at end
        if done and self.position != 0:
            price = self.df.iloc[self.bar]['close']
            if self.position == 1:
                pnl = (price - self.entry_price) / self.entry_price
            else:
                pnl = (self.entry_price - price) / self.entry_price
            self.balance *= (1 + pnl * 0.1)
            self.trades.append(pnl)
            reward += pnl * 100
        
        # Next state
        next_state = self.extractor.extract(self.df, self.bar)
        
        info = {
            'pnl': pnl,
            'balance': self.balance,
            'drawdown': (self.peak_balance - self.balance) / self.peak_balance,
            'position': self.position,
        }
        
        return next_state.to_array(), reward, done, info
    
    def get_stats(self) -> dict:
        """Get episode statistics."""
        if not self.trades:
            return {'total_pnl': 0, 'n_trades': 0, 'win_rate': 0, 'sharpe': 0}
        
        returns = np.array(self.trades)
        wins = returns[returns > 0]
        
        sharpe = np.mean(returns) / (np.std(returns) + 1e-10) * np.sqrt(252)
        
        return {
            'total_pnl': (self.balance - 10000),
            'n_trades': len(self.trades),
            'win_rate': len(wins) / len(self.trades) * 100 if self.trades else 0,
            'sharpe': sharpe,
            'max_drawdown': (self.peak_balance - self.balance) / self.peak_balance,
        }


# =============================================================================
# TRIAD TRAINER
# =============================================================================

class TriadTrainer:
    """
    Train Triad agents across multiple assets and timeframes.
    
    Tracks performance by:
    - Asset class
    - Timeframe  
    - Regime
    - Agent type
    """
    
    def __init__(self, role: str = 'trader'):
        self.role = role
        self.system = create_triad_system(role=role)
        
        # Results tracking
        self.results = defaultdict(lambda: defaultdict(list))
        self.regime_results = defaultdict(lambda: defaultdict(list))
        self.agent_performance = defaultdict(list)
    
    def train_on_file(self, file_info: dict, episodes: int = 10, 
                      max_steps: int = 500, verbose: bool = False) -> dict:
        """Train on a single data file."""
        try:
            df = load_data(file_info['path'])
            if len(df) < 200:
                return None
            
            # Use last 3000 bars max for speed
            df = df.iloc[-3000:].reset_index(drop=True)
            
            env = TriadTradingEnv(df)
            
            # Lockdown thresholds from first 100 bars
            warmup_df = df.iloc[:150]
            self.system.lockdown_thresholds(warmup_df)
            
            episode_results = []
            
            for ep in range(episodes):
                # Random start for variety
                start = np.random.randint(100, max(101, len(df) - max_steps - 10))
                state = env.reset(start_bar=start)
                
                total_reward = 0
                regime_counts = defaultdict(int)
                agent_counts = defaultdict(int)
                
                for step in range(max_steps):
                    # Get action from Triad system
                    action, info = self.system.step(df, env.bar, explore=True)
                    
                    # Track regime and agent
                    regime_counts[info['regime']] += 1
                    agent_counts[info['agent']] += 1
                    
                    # Execute
                    next_state, reward, done, step_info = env.step(action)
                    
                    # Update agents
                    self.system.update(reward, done)
                    
                    total_reward += reward
                    state = next_state
                    
                    if done:
                        break
                
                # Episode stats
                stats = env.get_stats()
                stats['total_reward'] = total_reward
                stats['dominant_regime'] = max(regime_counts, key=regime_counts.get) if regime_counts else 'unknown'
                stats['dominant_agent'] = max(agent_counts, key=agent_counts.get) if agent_counts else 'incumbent'
                
                episode_results.append(stats)
                
                if verbose and (ep + 1) % 5 == 0:
                    print(f"  Ep {ep+1}: R={total_reward:.1f} PnL=${stats['total_pnl']:.0f} "
                          f"WR={stats['win_rate']:.0f}% Agent={stats['dominant_agent']}")
            
            # Aggregate
            avg_pnl = np.mean([r['total_pnl'] for r in episode_results])
            avg_reward = np.mean([r['total_reward'] for r in episode_results])
            avg_wr = np.mean([r['win_rate'] for r in episode_results])
            
            return {
                'symbol': file_info['symbol'],
                'timeframe': file_info['timeframe'],
                'asset_class': file_info['asset_class'],
                'avg_pnl': avg_pnl,
                'avg_reward': avg_reward,
                'avg_win_rate': avg_wr,
                'episodes': episode_results,
            }
            
        except Exception as e:
            if verbose:
                print(f"  Error: {e}")
            return None
    
    def train_all(self, files: list, episodes_per_file: int = 10, 
                  max_files: int = 50, verbose: bool = True) -> dict:
        """Train on all files."""
        print(f"\n{'='*60}")
        print(f"TRIAD TRAINING - Role: {self.role.upper()}")
        print(f"{'='*60}")
        print(f"Files: {len(files)} (max {max_files})")
        print(f"Episodes per file: {episodes_per_file}")
        print(f"Agents: Incumbent (PPO), Competitor (A2C), Researcher (SAC)")
        print(f"{'='*60}\n")
        
        files = files[:max_files]
        start_time = time.time()
        
        all_results = []
        
        for i, file_info in enumerate(files):
            if verbose:
                print(f"[{i+1}/{len(files)}] {file_info['symbol']}_{file_info['timeframe']} "
                      f"({file_info['asset_class']})")
            
            result = self.train_on_file(file_info, episodes=episodes_per_file, verbose=verbose)
            
            if result:
                all_results.append(result)
                
                # Track by category
                self.results[file_info['asset_class']][file_info['timeframe']].append(result)
                
                # Track agent performance from last episodes
                for ep in result['episodes'][-3:]:
                    self.agent_performance[ep['dominant_agent']].append(ep['total_pnl'])
        
        elapsed = time.time() - start_time
        
        return {
            'results': all_results,
            'elapsed': elapsed,
            'n_files': len(all_results),
        }
    
    def print_summary(self):
        """Print training summary."""
        print(f"\n{'='*60}")
        print("TRIAD TRAINING SUMMARY")
        print(f"{'='*60}")
        
        # By asset class
        print("\n📊 BY ASSET CLASS:")
        for asset_class, tf_data in sorted(self.results.items()):
            all_pnls = []
            for tf, results in tf_data.items():
                all_pnls.extend([r['avg_pnl'] for r in results])
            
            if all_pnls:
                avg = np.mean(all_pnls)
                pos = sum(1 for p in all_pnls if p > 0)
                print(f"  {asset_class:15s}: ${avg:+8.0f} avg | {pos}/{len(all_pnls)} profitable")
        
        # By timeframe
        print("\n📊 BY TIMEFRAME:")
        tf_summary = defaultdict(list)
        for asset_class, tf_data in self.results.items():
            for tf, results in tf_data.items():
                tf_summary[tf].extend([r['avg_pnl'] for r in results])
        
        for tf, pnls in sorted(tf_summary.items()):
            avg = np.mean(pnls)
            pos = sum(1 for p in pnls if p > 0)
            print(f"  {tf:15s}: ${avg:+8.0f} avg | {pos}/{len(pnls)} profitable")
        
        # By agent
        print("\n📊 BY AGENT (when dominant):")
        for agent, pnls in sorted(self.agent_performance.items()):
            if pnls:
                avg = np.mean(pnls)
                pos = sum(1 for p in pnls if p > 0)
                print(f"  {agent:15s}: ${avg:+8.0f} avg | {pos}/{len(pnls)} wins")
        
        # Recommendations
        print(f"\n{'='*60}")
        print("🎯 RECOMMENDATIONS")
        print(f"{'='*60}")
        
        # Best asset class
        best_asset = max(self.results.items(), 
                        key=lambda x: np.mean([r['avg_pnl'] for tf in x[1].values() for r in tf]),
                        default=(None, None))
        if best_asset[0]:
            print(f"\n  Best asset class: {best_asset[0].upper()}")
        
        # Best agent
        best_agent = max(self.agent_performance.items(),
                        key=lambda x: np.mean(x[1]) if x[1] else -999,
                        default=(None, None))
        if best_agent[0]:
            print(f"  Best agent overall: {best_agent[0].upper()}")
        
        print("\n  THE MARKET HAS TOLD US - NOW WE KNOW!")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Train Triad System')
    parser.add_argument('--role', type=str, default='trader',
                       choices=['trader', 'risk_manager', 'portfolio_manager'],
                       help='Trading role')
    parser.add_argument('--episodes', type=int, default=10,
                       help='Episodes per file')
    parser.add_argument('--max-files', type=int, default=50,
                       help='Maximum files to process')
    parser.add_argument('--assets', nargs='+', default=None,
                       help='Asset classes to include (e.g., crypto forex)')
    parser.add_argument('--timeframes', nargs='+', default=None,
                       help='Timeframes to include (e.g., H1 H4)')
    parser.add_argument('--save', action='store_true',
                       help='Save results to JSON')
    
    args = parser.parse_args()
    
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║                    KINETRA TRIAD TRAINING                            ║
║         Incumbent (PPO) • Competitor (A2C) • Researcher (SAC)        ║
║                  Markets are about IMBALANCES                        ║
╚══════════════════════════════════════════════════════════════════════╝
""")
    
    # Discover data
    print("📁 Discovering data files...")
    files = discover_data_files()
    print(f"   Found {len(files)} files")
    
    # Filter by asset class
    if args.assets:
        files = [f for f in files if f['asset_class'] in args.assets]
        print(f"   Filtered to {len(files)} files (assets: {args.assets})")
    
    # Filter by timeframe
    if args.timeframes:
        files = [f for f in files if f['timeframe'] in args.timeframes]
        print(f"   Filtered to {len(files)} files (timeframes: {args.timeframes})")
    
    if not files:
        print("❌ No data files found!")
        return
    
    # Show distribution
    asset_counts = defaultdict(int)
    tf_counts = defaultdict(int)
    for f in files:
        asset_counts[f['asset_class']] += 1
        tf_counts[f['timeframe']] += 1
    
    print(f"\n   Asset classes: {dict(asset_counts)}")
    print(f"   Timeframes: {dict(tf_counts)}")
    
    # Train
    trainer = TriadTrainer(role=args.role)
    results = trainer.train_all(
        files,
        episodes_per_file=args.episodes,
        max_files=args.max_files,
        verbose=True
    )
    
    # Summary
    trainer.print_summary()
    
    # Save results
    if args.save:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = Path("results/triad")
        results_dir.mkdir(parents=True, exist_ok=True)
        
        output_file = results_dir / f"triad_training_{args.role}_{timestamp}.json"
        
        # Convert to serializable
        save_data = {
            'role': args.role,
            'timestamp': timestamp,
            'n_files': results['n_files'],
            'elapsed_seconds': results['elapsed'],
            'summary': {
                'by_asset': {k: {'n_files': len([r for tf in v.values() for r in tf]),
                                'avg_pnl': np.mean([r['avg_pnl'] for tf in v.values() for r in tf])}
                            for k, v in trainer.results.items()},
                'by_agent': {k: {'n_episodes': len(v), 'avg_pnl': np.mean(v) if v else 0}
                            for k, v in trainer.agent_performance.items()},
            }
        }
        
        with open(output_file, 'w') as f:
            json.dump(save_data, f, indent=2)
        
        print(f"\n💾 Results saved: {output_file}")
    
    print(f"\n⏱️  Total time: {results['elapsed']:.1f}s")
    print("\n" + "="*60)
    print("TRIAD TRAINING COMPLETE")
    print("="*60)


if __name__ == '__main__':
    main()
