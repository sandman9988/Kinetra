#!/usr/bin/env python3
"""
SUPERPOT COMPLETE EXPLORATION
=============================

Run ALL combinations:
1. By Asset Class (crypto, forex, metals, commodities, indices)
2. By Timeframe (M15, M30, H1, H4)
3. By Role (trader, risk_manager, portfolio_manager)
4. Cross-combinations

Find what features matter WHERE and for WHOM.

Usage:
    python scripts/superpot_complete.py
    python scripts/superpot_complete.py --quick  # Fast test
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import json
import time
import warnings
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy import bool, dtype, ndarray, signedinteger, unsignedinteger, floating
from numpy._typing import _16Bit, _32Bit, _64Bit, _8Bit

warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.superpot_explorer import (
    SuperPotExtractor, FeatureImportanceTracker, 
    load_data, discover_files
)


# =============================================================================
# CLASSIFICATION
# =============================================================================

def classify_asset(symbol: str) -> str:
    s = symbol.upper().replace('+', '').replace('-', '')
    if any(x in s for x in ['BTC', 'ETH', 'XRP', 'LTC', 'DOGE', 'SOL']):
        return 'crypto'
    elif any(x in s for x in ['XAU', 'XAG', 'XPT', 'XPD', 'GOLD', 'SILVER', 'COPPER']):
        return 'metals'
    elif any(x in s for x in ['OIL', 'WTI', 'BRENT', 'GAS', 'GASOIL', 'UKOUSD']):
        return 'commodities'
    elif any(x in s for x in ['SPX', 'NAS', 'DOW', 'DJ', 'DAX', 'FTSE', 'NIKKEI', 
                               'US30', 'US500', 'US100', 'GER', 'UK100', 'SA40', 'EU50']):
        return 'indices'
    elif len(s) == 6 and s.isalpha():
        return 'forex'
    return 'unknown'


def classify_timeframe(tf: str) -> str:
    tf = tf.upper()
    if tf in ['M1', 'M5', 'M15']:
        return 'scalp'
    elif tf in ['M30', 'H1']:
        return 'intraday'
    elif tf in ['H4', 'D1']:
        return 'swing'
    return tf


# =============================================================================
# ROLE-AWARE AGENT
# =============================================================================

@dataclass
class RoleConfig:
    """Configuration for different trading roles."""
    name: str
    pnl_weight: float
    drawdown_weight: float
    sharpe_weight: float
    holding_penalty: float
    description: str


ROLE_CONFIGS = {
    'trader': RoleConfig(
        name='trader',
        pnl_weight=1.0,
        drawdown_weight=0.1,
        sharpe_weight=0.1,
        holding_penalty=0.001,
        description='Aggressive PnL focus'
    ),
    'risk_manager': RoleConfig(
        name='risk_manager',
        pnl_weight=0.3,
        drawdown_weight=1.0,
        sharpe_weight=0.5,
        holding_penalty=0.005,
        description='Drawdown minimization focus'
    ),
    'portfolio_manager': RoleConfig(
        name='portfolio_manager',
        pnl_weight=0.5,
        drawdown_weight=0.3,
        sharpe_weight=1.0,
        holding_penalty=0.002,
        description='Risk-adjusted returns focus'
    ),
}


class RoleAwareAgent:
    """Agent with role-specific reward shaping."""
    
    def __init__(self, n_features: int, n_actions: int, role: str):
        self.n_features = n_features
        self.n_actions = n_actions
        self.role = role
        self.config = ROLE_CONFIGS[role]
        
        # Policy
        self.W = np.random.randn(n_features, n_actions) * 0.01
        self.b = np.zeros(n_actions)
        
        # Value
        self.V_W = np.random.randn(n_features) * 0.01
        self.V_b = 0.0
        
        self.lr = 0.001
        self.gamma = 0.95
        self.epsilon = 0.3
        
        # Tracking for role-specific metrics
        self.peak_balance = 10000
        self.returns_history = []
    
    def _softmax(self, x):
        x = np.clip(x, -20, 20)
        e = np.exp(x - np.max(x))
        p = e / (e.sum() + 1e-10)
        return np.ones(len(x)) / len(x) if np.any(np.isnan(p)) else p
    
    def compute_reward(self, pnl: float, balance: float, position: int) -> floating[Any]:
        """Compute role-specific reward."""
        # Drawdown
        self.peak_balance = max(self.peak_balance, balance)
        drawdown = (self.peak_balance - balance) / self.peak_balance
        
        # Sharpe proxy
        self.returns_history.append(pnl)
        if len(self.returns_history) > 20:
            self.returns_history = self.returns_history[-20:]
        sharpe = np.mean(self.returns_history) / (np.std(self.returns_history) + 1e-10)
        
        # Holding penalty
        holding_cost = self.config.holding_penalty if position != 0 else 0
        
        # Role-weighted reward
        reward = (
            self.config.pnl_weight * pnl * 100 -
            self.config.drawdown_weight * drawdown * 100 +
            self.config.sharpe_weight * sharpe * 10 -
            holding_cost
        )
        
        return reward
    
    def select_action(self, features: np.ndarray, explore: bool = True) -> int | bool | bool | unsignedinteger[_8Bit] | \
                                                                           unsignedinteger[_16Bit] | unsignedinteger[
                                                                               _32Bit] | unsignedinteger[
                                                                               _32Bit | _64Bit] | unsignedinteger[
                                                                               _64Bit] | signedinteger[_8Bit] | \
                                                                           signedinteger[_16Bit] | signedinteger[
                                                                               _32Bit] | signedinteger[
                                                                               _32Bit | _64Bit] | signedinteger[
                                                                               _64Bit] | ndarray[tuple[Any, ...], dtype[
        signedinteger[_32Bit | _64Bit]]] | ndarray[tuple[Any, ...], dtype[bool]] | ndarray[tuple[Any, ...], dtype[
        signedinteger[_8Bit]]] | ndarray[tuple[Any, ...], dtype[signedinteger[_16Bit]]] | ndarray[
                                                                               tuple[Any, ...], dtype[
                                                                                   signedinteger[_32Bit]]] | ndarray[
                                                                               tuple[Any, ...], dtype[
                                                                                   signedinteger[_64Bit]]] | ndarray[
                                                                               tuple[Any, ...], dtype[
                                                                                   unsignedinteger[_8Bit]]] | ndarray[
                                                                               tuple[Any, ...], dtype[
                                                                                   unsignedinteger[_16Bit]]] | ndarray[
                                                                               tuple[Any, ...], dtype[
                                                                                   unsignedinteger[_32Bit]]] | ndarray[
                                                                               tuple[Any, ...], dtype[
                                                                                   unsignedinteger[_64Bit]]] | ndarray[
                                                                               tuple[Any, ...], dtype[
                                                                                   unsignedinteger[_32Bit | _64Bit]]]:
        features = np.nan_to_num(features, nan=0, posinf=0, neginf=0)
        if len(features) != self.n_features:
            padded = np.zeros(self.n_features)
            padded[:min(len(features), self.n_features)] = features[:self.n_features]
            features = padded
        
        logits = features @ self.W + self.b
        probs = self._softmax(logits)
        
        if explore and np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        return np.argmax(probs)
    
    def update(self, features, action, reward, next_features, done):
        features = np.nan_to_num(features, nan=0, posinf=0, neginf=0)
        next_features = np.nan_to_num(next_features, nan=0, posinf=0, neginf=0)
        
        if len(features) != self.n_features:
            padded = np.zeros(self.n_features)
            padded[:min(len(features), self.n_features)] = features[:self.n_features]
            features = padded
        if len(next_features) != self.n_features:
            padded = np.zeros(self.n_features)
            padded[:min(len(next_features), self.n_features)] = next_features[:self.n_features]
            next_features = padded
        
        value = features @ self.V_W + self.V_b
        next_value = 0 if done else next_features @ self.V_W + self.V_b
        td_error = reward + self.gamma * next_value - value
        
        self.V_W += self.lr * td_error * features
        self.V_b += self.lr * td_error
        
        probs = self._softmax(features @ self.W + self.b)
        grad = -probs.copy()
        grad[action] += 1
        grad *= td_error
        
        self.W += self.lr * np.outer(features, grad)
        self.b += self.lr * grad
        
        self.epsilon = max(0.05, self.epsilon * 0.9995)
    
    def reset_episode(self):
        self.peak_balance = 10000
        self.returns_history = []


# =============================================================================
# DIMENSION TRAINER
# =============================================================================

class DimensionTrainer:
    """Train on a specific dimension (class, timeframe, role combination)."""
    
    def __init__(self, dimension_name: str, files: list, role: str = 'trader'):
        self.dimension_name = dimension_name
        self.files = files
        self.role = role
        
        self.extractor = SuperPotExtractor(lookback=50)
        self.tracker = FeatureImportanceTracker(
            self.extractor.n_features,
            self.extractor.feature_names
        )
        self.agent = RoleAwareAgent(self.extractor.n_features, 4, role)
        
        self.rewards = []
        self.pnls = []
        self.drawdowns = []
    
    def train(self, episodes: int, prune_every: int, prune_count: int,
              max_steps: int = 500, verbose: bool = False) -> dict:
        
        for ep in range(episodes):
            file_info = self.files[np.random.randint(len(self.files))]
            
            try:
                df = load_data(file_info['path'])
                if len(df) < 200:
                    continue
                
                df = df.iloc[-2000:].reset_index(drop=True)
                
                start_bar = np.random.randint(100, max(101, len(df) - max_steps - 10))
                bar = start_bar
                position = 0
                entry_price = 0
                balance = 10000
                peak_balance = 10000
                episode_reward = 0
                
                self.agent.reset_episode()
                
                for step in range(max_steps):
                    if bar >= len(df) - 1:
                        break
                    
                    features = self.extractor.extract(df, bar)
                    active_features = self.tracker.mask_features(features)
                    
                    action = self.agent.select_action(active_features, explore=True)
                    
                    price = df.iloc[bar]['close']
                    pnl = 0
                    
                    if action == 1 and position == 0:
                        position = 1
                        entry_price = price * 1.0001
                    elif action == 2 and position == 0:
                        position = -1
                        entry_price = price * 0.9999
                    elif action == 3 and position != 0:
                        if position == 1:
                            pnl = (price * 0.9999 - entry_price) / entry_price
                        else:
                            pnl = (entry_price - price * 1.0001) / entry_price
                        balance *= (1 + pnl * 0.1)
                        position = 0
                    
                    peak_balance = max(peak_balance, balance)
                    reward = self.agent.compute_reward(pnl, balance, position)
                    
                    bar += 1
                    next_features = self.extractor.extract(df, bar) if bar < len(df) else features
                    active_next = self.tracker.mask_features(next_features)
                    
                    self.tracker.record(features, action, reward)
                    self.agent.update(active_features, action, reward, active_next, bar >= len(df) - 1)
                    
                    episode_reward += reward
                
                final_pnl = balance - 10000
                max_dd = (peak_balance - balance) / peak_balance
                
                self.rewards.append(episode_reward)
                self.pnls.append(final_pnl)
                self.drawdowns.append(max_dd)
                
                # Prune
                if (ep + 1) % prune_every == 0 and self.tracker.n_active > prune_count + 20:
                    self.tracker.prune(prune_count)
                    active_indices = np.where(self.tracker.active_mask)[0]
                    self.agent.W = self.agent.W[active_indices]
                    self.agent.V_W = self.agent.V_W[active_indices]
                    self.agent.n_features = self.tracker.n_active
            
            except:
                continue
        
        return {
            'dimension': self.dimension_name,
            'role': self.role,
            'episodes': len(self.rewards),
            'avg_reward': float(np.mean(self.rewards)) if self.rewards else 0,
            'avg_pnl': float(np.mean(self.pnls)) if self.pnls else 0,
            'avg_drawdown': float(np.mean(self.drawdowns)) if self.drawdowns else 0,
            'win_rate': sum(1 for p in self.pnls if p > 0) / len(self.pnls) * 100 if self.pnls else 0,
            'surviving_features': self.tracker.get_active_features(),
            'top_features': self.tracker.get_top_features(30),
            'n_surviving': self.tracker.n_active,
        }


# =============================================================================
# CROSS-DIMENSIONAL ANALYSIS
# =============================================================================

def analyze_across_dimensions(results: Dict[str, dict]) -> dict:
    """Analyze feature importance across all dimensions."""
    
    # Count feature survival
    feature_survival = defaultdict(lambda: {'dimensions': [], 'scores': []})
    
    for dim_name, result in results.items():
        for feature in result['surviving_features']:
            feature_survival[feature]['dimensions'].append(dim_name)
        
        for feature, score in result['top_features']:
            feature_survival[feature]['scores'].append(score)
    
    # Categorize
    all_dims = list(results.keys())
    n_dims = len(all_dims)
    
    # Universal = in ALL dimensions
    universal = [f for f, data in feature_survival.items() 
                 if len(data['dimensions']) == n_dims]
    
    # Very common = in 75%+ dimensions
    very_common = [f for f, data in feature_survival.items()
                   if len(data['dimensions']) >= n_dims * 0.75 and f not in universal]
    
    # Common = in 50%+ dimensions
    common = [f for f, data in feature_survival.items()
              if len(data['dimensions']) >= n_dims * 0.5 and f not in universal and f not in very_common]
    
    # Compute aggregate scores
    rankings = []
    for feature, data in feature_survival.items():
        if data['scores']:
            rankings.append({
                'feature': feature,
                'avg_score': np.mean(data['scores']),
                'n_dimensions': len(data['dimensions']),
                'coverage': len(data['dimensions']) / n_dims,
                'dimensions': data['dimensions'],
            })
    
    rankings.sort(key=lambda x: x['avg_score'] * x['coverage'], reverse=True)
    
    return {
        'universal': universal,
        'very_common': very_common,
        'common': common,
        'rankings': rankings,
        'feature_survival': dict(feature_survival),
    }


def compare_roles(results: Dict[str, dict]) -> dict:
    """Compare feature importance across roles."""
    
    role_features = defaultdict(lambda: defaultdict(list))
    
    for dim_name, result in results.items():
        role = result['role']
        for feature, score in result['top_features'][:20]:
            role_features[role][feature].append(score)
    
    # Compute role-specific rankings
    role_rankings = {}
    for role, features in role_features.items():
        rankings = [(f, np.mean(scores)) for f, scores in features.items()]
        rankings.sort(key=lambda x: x[1], reverse=True)
        role_rankings[role] = rankings[:20]
    
    # Find role-specific features
    all_features = set()
    for role, rankings in role_rankings.items():
        all_features.update([f for f, _ in rankings])
    
    role_specific = defaultdict(list)
    shared_across_roles = []
    
    for feature in all_features:
        roles_with_feature = [r for r, rankings in role_rankings.items() 
                             if any(f == feature for f, _ in rankings)]
        if len(roles_with_feature) == 1:
            role_specific[roles_with_feature[0]].append(feature)
        elif len(roles_with_feature) == len(role_rankings):
            shared_across_roles.append(feature)
    
    return {
        'role_rankings': role_rankings,
        'role_specific': dict(role_specific),
        'shared_across_roles': shared_across_roles,
    }


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='SuperPot Complete Exploration')
    parser.add_argument('--quick', action='store_true', help='Quick test mode')
    parser.add_argument('--episodes', type=int, default=80, help='Episodes per dimension')
    parser.add_argument('--prune-every', type=int, default=15, help='Prune interval')
    parser.add_argument('--prune-count', type=int, default=8, help='Features to prune')
    
    args = parser.parse_args()
    
    if args.quick:
        args.episodes = 30
        args.prune_every = 10
        args.prune_count = 5
    
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║                  SUPERPOT COMPLETE EXPLORATION                       ║
║                                                                      ║
║   Dimensions: Asset Class × Timeframe × Role                         ║
║   Find what matters WHERE and for WHOM                               ║
╚══════════════════════════════════════════════════════════════════════╝
""")
    
    # Discover files
    print("📁 Discovering data files...")
    all_files = discover_files()
    
    # Classify
    for f in all_files:
        f['asset_class'] = classify_asset(f['symbol'])
        f['tf_group'] = classify_timeframe(f['timeframe'])
    
    # Filter unknown
    all_files = [f for f in all_files if f['asset_class'] != 'unknown']
    
    # Group by dimensions
    by_class = defaultdict(list)
    by_timeframe = defaultdict(list)
    
    for f in all_files:
        by_class[f['asset_class']].append(f)
        by_timeframe[f['timeframe']].append(f)
    
    print(f"\n📊 Data distribution:")
    print(f"   Asset classes: {dict((k, len(v)) for k, v in by_class.items())}")
    print(f"   Timeframes: {dict((k, len(v)) for k, v in by_timeframe.items())}")
    
    roles = ['trader', 'risk_manager', 'portfolio_manager']
    
    # Calculate total combinations
    n_class_combos = len([c for c, files in by_class.items() if len(files) >= 3]) * len(roles)
    n_tf_combos = len([t for t, files in by_timeframe.items() if len(files) >= 3]) * len(roles)
    total = n_class_combos + n_tf_combos
    
    print(f"\n🔬 Training combinations: {total}")
    print(f"   Episodes per combo: {args.episodes}")
    print(f"   Estimated time: {total * args.episodes * 0.5 / 60:.0f} min")
    
    start_time = time.time()
    all_results = {}
    combo_num = 0
    
    # =========================================================================
    # TRAIN BY ASSET CLASS × ROLE
    # =========================================================================
    print(f"\n{'='*70}")
    print("PHASE 1: ASSET CLASS × ROLE")
    print(f"{'='*70}")
    
    for asset_class, files in sorted(by_class.items()):
        if len(files) < 3:
            continue
        
        for role in roles:
            combo_num += 1
            dim_name = f"{asset_class}_{role}"
            
            print(f"\n[{combo_num}/{total}] {dim_name} ({len(files)} files)")
            
            trainer = DimensionTrainer(dim_name, files, role)
            result = trainer.train(
                episodes=args.episodes,
                prune_every=args.prune_every,
                prune_count=args.prune_count,
            )
            all_results[dim_name] = result
            
            print(f"  → PnL=${result['avg_pnl']:+.0f} DD={result['avg_drawdown']*100:.1f}% "
                  f"Features={result['n_surviving']}")
    
    # =========================================================================
    # TRAIN BY TIMEFRAME × ROLE
    # =========================================================================
    print(f"\n{'='*70}")
    print("PHASE 2: TIMEFRAME × ROLE")
    print(f"{'='*70}")
    
    for timeframe, files in sorted(by_timeframe.items()):
        if len(files) < 3:
            continue
        
        for role in roles:
            combo_num += 1
            dim_name = f"{timeframe}_{role}"
            
            print(f"\n[{combo_num}/{total}] {dim_name} ({len(files)} files)")
            
            trainer = DimensionTrainer(dim_name, files, role)
            result = trainer.train(
                episodes=args.episodes,
                prune_every=args.prune_every,
                prune_count=args.prune_count,
            )
            all_results[dim_name] = result
            
            print(f"  → PnL=${result['avg_pnl']:+.0f} DD={result['avg_drawdown']*100:.1f}% "
                  f"Features={result['n_surviving']}")
    
    elapsed = time.time() - start_time
    
    # =========================================================================
    # ANALYSIS
    # =========================================================================
    print(f"\n{'='*70}")
    print("CROSS-DIMENSIONAL ANALYSIS")
    print(f"{'='*70}")
    
    cross_analysis = analyze_across_dimensions(all_results)
    role_analysis = compare_roles(all_results)
    
    # Universal features
    print(f"\n🌍 UNIVERSAL FEATURES (all dimensions):")
    if cross_analysis['universal']:
        for f in cross_analysis['universal'][:15]:
            print(f"   ✓ {f}")
    else:
        print("   (none)")
    
    # Very common
    print(f"\n🔗 VERY COMMON FEATURES (75%+ dimensions):")
    for f in cross_analysis['very_common'][:15]:
        print(f"   • {f}")
    
    # Top rankings
    print(f"\n🏆 TOP FEATURES BY COMBINED SCORE:")
    print(f"{'Feature':<40s} {'Score':>8s} {'Coverage':>10s}")
    print("-" * 60)
    for r in cross_analysis['rankings'][:25]:
        coverage_pct = r['coverage'] * 100
        combined = r['avg_score'] * r['coverage']
        print(f"{r['feature']:<40s} {combined:>8.4f} {coverage_pct:>9.0f}%")
    
    # Role comparison
    print(f"\n{'='*70}")
    print("ROLE-SPECIFIC ANALYSIS")
    print(f"{'='*70}")
    
    for role in roles:
        config = ROLE_CONFIGS[role]
        print(f"\n📋 {role.upper()} ({config.description}):")
        
        # Performance for this role
        role_results = [r for dim, r in all_results.items() if r['role'] == role]
        avg_pnl = np.mean([r['avg_pnl'] for r in role_results])
        avg_dd = np.mean([r['avg_drawdown'] for r in role_results])
        
        print(f"   Avg PnL: ${avg_pnl:+.0f} | Avg DD: {avg_dd*100:.1f}%")
        
        if role in role_analysis['role_rankings']:
            print(f"   Top features:")
            for feat, score in role_analysis['role_rankings'][role][:8]:
                print(f"     • {feat} ({score:.4f})")
    
    # Role-specific features
    print(f"\n🎯 ROLE-SPECIFIC FEATURES:")
    for role, features in role_analysis['role_specific'].items():
        if features:
            print(f"\n   {role.upper()} only:")
            for f in features[:5]:
                print(f"     - {f}")
    
    # Shared across roles
    print(f"\n🤝 SHARED ACROSS ALL ROLES:")
    for f in role_analysis['shared_across_roles'][:10]:
        print(f"   ✓ {f}")
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    print(f"\n{'='*70}")
    print("SUMMARY & RECOMMENDATIONS")
    print(f"{'='*70}")
    
    print(f"\n📊 Dimensions trained: {len(all_results)}")
    print(f"⏱️  Total time: {elapsed/60:.1f} min")
    
    print(f"\n💡 KEY INSIGHTS:")
    
    # Universal truths
    if cross_analysis['universal']:
        print(f"\n   1. UNIVERSAL TRUTHS ({len(cross_analysis['universal'])} features):")
        print(f"      These work EVERYWHERE - use them always:")
        for f in cross_analysis['universal'][:5]:
            print(f"        ✓ {f}")
    
    # Best for each role
    print(f"\n   2. ROLE OPTIMIZATION:")
    for role in roles:
        if role in role_analysis['role_rankings'] and role_analysis['role_rankings'][role]:
            top_feat = role_analysis['role_rankings'][role][0][0]
            print(f"      {role}: Focus on '{top_feat}'")
    
    # Role-specific edges
    if any(role_analysis['role_specific'].values()):
        print(f"\n   3. SPECIALIZED EDGES:")
        for role, features in role_analysis['role_specific'].items():
            if features:
                print(f"      {role} unique: {features[0]}")
    
    print(f"\n   THE MARKET HAS SPOKEN!")
    
    # Save results
    results_dir = Path("results/superpot")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = results_dir / f"superpot_complete_{timestamp}.json"
    
    output = {
        'timestamp': timestamp,
        'elapsed_minutes': elapsed / 60,
        'n_dimensions': len(all_results),
        'episodes_per_dim': args.episodes,
        'universal_features': cross_analysis['universal'],
        'very_common_features': cross_analysis['very_common'],
        'top_rankings': [
            {'feature': r['feature'], 'score': float(r['avg_score'] * r['coverage']), 
             'coverage': float(r['coverage'])}
            for r in cross_analysis['rankings'][:50]
        ],
        'role_rankings': {
            role: [(str(f), float(s)) for f, s in rankings[:20]]
            for role, rankings in role_analysis['role_rankings'].items()
        },
        'role_specific': {k: list(v) for k, v in role_analysis['role_specific'].items()},
        'shared_across_roles': role_analysis['shared_across_roles'],
        'by_dimension': {
            dim: {
                'role': r['role'],
                'avg_pnl': r['avg_pnl'],
                'avg_drawdown': r['avg_drawdown'],
                'n_surviving': r['n_surviving'],
                'top_5': [(str(f), float(s)) for f, s in r['top_features'][:5]],
            }
            for dim, r in all_results.items()
        }
    }
    
    with open(results_file, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n💾 Results saved: {results_file}")
    
    print(f"\n{'='*70}")
    print("SUPERPOT COMPLETE EXPLORATION FINISHED")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
