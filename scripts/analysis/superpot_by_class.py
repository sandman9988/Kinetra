#!/usr/bin/env python3
"""
SUPERPOT BY ASSET CLASS
=======================

Run SuperPot separately for each asset class:
- Crypto
- Forex
- Metals
- Commodities
- Indices

Then find COMMON DENOMINATORS - features that survive across ALL classes.

These are the universal truths the market is telling us.

Usage:
    python scripts/superpot_by_class.py
    python scripts/superpot_by_class.py --episodes 100 --prune-every 20
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

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.superpot_explorer import (
    SuperPotExtractor, FeatureImportanceTracker, SuperPotAgent,
    load_data, discover_files
)


# =============================================================================
# ASSET CLASS CLASSIFICATION
# =============================================================================

def classify_asset(symbol: str) -> str:
    """Classify symbol into asset class."""
    s = symbol.upper().replace('+', '').replace('-', '')
    
    if any(x in s for x in ['BTC', 'ETH', 'XRP', 'LTC', 'DOGE', 'SOL']):
        return 'crypto'
    elif any(x in s for x in ['XAU', 'XAG', 'XPT', 'XPD', 'GOLD', 'SILVER', 'COPPER']):
        return 'metals'
    elif any(x in s for x in ['OIL', 'WTI', 'BRENT', 'GAS', 'GASOIL', 'UKOUSD', 'USOIL']):
        return 'commodities'
    elif any(x in s for x in ['SPX', 'NAS', 'DOW', 'DJ', 'DAX', 'FTSE', 'NIKKEI', 
                               'US30', 'US500', 'US100', 'GER', 'UK100', 'SA40', 
                               'EU50', '225', '100', '40', '30', '2000']):
        return 'indices'
    elif len(s) == 6 and s.isalpha():
        return 'forex'
    return 'unknown'


def group_files_by_class(files: list) -> dict:
    """Group files by asset class."""
    grouped = defaultdict(list)
    
    for f in files:
        asset_class = classify_asset(f['symbol'])
        if asset_class != 'unknown':
            f['asset_class'] = asset_class
            grouped[asset_class].append(f)
    
    return dict(grouped)


# =============================================================================
# SINGLE CLASS TRAINER
# =============================================================================

class AssetClassTrainer:
    """Train SuperPot on a single asset class."""
    
    def __init__(self, asset_class: str, files: list):
        self.asset_class = asset_class
        self.files = files
        
        self.extractor = SuperPotExtractor(lookback=50)
        self.tracker = FeatureImportanceTracker(
            self.extractor.n_features, 
            self.extractor.feature_names
        )
        self.agent = SuperPotAgent(
            self.extractor.n_features, 
            n_actions=4, 
            name=f"SuperPot_{asset_class}"
        )
        
        self.rewards = []
        self.pnls = []
    
    def train(self, episodes: int, prune_every: int, prune_count: int, 
              max_steps: int = 500, verbose: bool = True) -> dict:
        """Train on this asset class."""
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"  TRAINING: {self.asset_class.upper()}")
            print(f"  Files: {len(self.files)}")
            print(f"{'='*60}")
        
        for ep in range(episodes):
            # Pick random file from this class
            file_info = self.files[np.random.randint(len(self.files))]
            
            try:
                df = load_data(file_info['path'])
                if len(df) < 200:
                    continue
                
                df = df.iloc[-2000:].reset_index(drop=True)
                
                # Episode
                start_bar = np.random.randint(100, max(101, len(df) - max_steps - 10))
                bar = start_bar
                position = 0
                entry_price = 0
                balance = 10000
                episode_reward = 0
                
                for step in range(max_steps):
                    if bar >= len(df) - 1:
                        break
                    
                    features = self.extractor.extract(df, bar)
                    active_features = self.tracker.mask_features(features)
                    
                    action = self.agent.select_action(active_features, explore=True)
                    
                    price = df.iloc[bar]['close']
                    reward = 0
                    
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
                        reward = pnl * 100
                        position = 0
                    
                    if position != 0:
                        reward -= 0.001
                    
                    bar += 1
                    next_features = self.extractor.extract(df, bar) if bar < len(df) else features
                    active_next = self.tracker.mask_features(next_features)
                    
                    self.tracker.record(features, action, reward)
                    self.agent.update(active_features, action, reward, active_next, bar >= len(df) - 1)
                    
                    episode_reward += reward
                
                pnl = balance - 10000
                self.rewards.append(episode_reward)
                self.pnls.append(pnl)
                
                if verbose and (ep + 1) % 10 == 0:
                    avg_r = np.mean(self.rewards[-10:])
                    avg_pnl = np.mean(self.pnls[-10:])
                    print(f"  Ep {ep+1:3d}: R={avg_r:+7.1f} PnL=${avg_pnl:+7.0f} | "
                          f"Features: {self.tracker.n_active}/{self.extractor.n_features}")
                
                # Prune
                if (ep + 1) % prune_every == 0 and self.tracker.n_active > prune_count + 20:
                    pruned = self.tracker.prune(prune_count)
                    if verbose:
                        print(f"  🗑️  Pruned {len(pruned)} features → {self.tracker.n_active} remaining")
                    
                    # Resize agent
                    active_indices = np.where(self.tracker.active_mask)[0]
                    self.agent.W = self.agent.W[active_indices]
                    self.agent.V_W = self.agent.V_W[active_indices]
                    self.agent.n_features = self.tracker.n_active
            
            except Exception as e:
                continue
        
        # Get results
        surviving = self.tracker.get_active_features()
        top_features = self.tracker.get_top_features(30)
        pruned = self.tracker.get_pruned_features()
        
        return {
            'asset_class': self.asset_class,
            'episodes': len(self.rewards),
            'avg_reward': float(np.mean(self.rewards)) if self.rewards else 0,
            'avg_pnl': float(np.mean(self.pnls)) if self.pnls else 0,
            'win_rate': sum(1 for p in self.pnls if p > 0) / len(self.pnls) * 100 if self.pnls else 0,
            'surviving_features': surviving,
            'top_features': top_features,
            'pruned_features': pruned,
            'n_surviving': len(surviving),
            'n_pruned': len(pruned),
        }


# =============================================================================
# CROSS-CLASS ANALYSIS
# =============================================================================

def find_common_features(results: dict) -> dict:
    """Find features that survive across multiple asset classes."""
    
    # Count how many classes each feature survives in
    feature_survival = defaultdict(list)
    
    for asset_class, result in results.items():
        for feature in result['surviving_features']:
            feature_survival[feature].append(asset_class)
    
    # Categorize
    all_classes = list(results.keys())
    n_classes = len(all_classes)
    
    universal = []  # Survive in ALL classes
    common = []     # Survive in 3+ classes
    specific = defaultdict(list)  # Class-specific
    
    for feature, classes in feature_survival.items():
        if len(classes) == n_classes:
            universal.append(feature)
        elif len(classes) >= 3:
            common.append((feature, classes))
        elif len(classes) == 1:
            specific[classes[0]].append(feature)
    
    return {
        'universal': universal,
        'common': common,
        'specific': dict(specific),
        'feature_survival': dict(feature_survival),
    }


def compute_feature_rankings(results: dict) -> list:
    """Compute average ranking across all classes."""
    
    feature_scores = defaultdict(list)
    
    for asset_class, result in results.items():
        for i, (feature, score) in enumerate(result['top_features']):
            # Convert rank to score (higher = better)
            rank_score = (30 - i) / 30  # Top feature gets 1.0
            feature_scores[feature].append({
                'class': asset_class,
                'importance': score,
                'rank_score': rank_score,
            })
    
    # Compute aggregate scores
    aggregated = []
    for feature, scores in feature_scores.items():
        avg_importance = np.mean([s['importance'] for s in scores])
        avg_rank = np.mean([s['rank_score'] for s in scores])
        n_classes = len(scores)
        
        # Combined score: importance * rank * class coverage
        combined = avg_importance * avg_rank * (n_classes / len(results))
        
        aggregated.append({
            'feature': feature,
            'avg_importance': avg_importance,
            'avg_rank_score': avg_rank,
            'n_classes': n_classes,
            'combined_score': combined,
            'classes': [s['class'] for s in scores],
        })
    
    # Sort by combined score
    aggregated.sort(key=lambda x: x['combined_score'], reverse=True)
    
    return aggregated


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='SuperPot by Asset Class')
    parser.add_argument('--episodes', type=int, default=100, help='Episodes per class')
    parser.add_argument('--prune-every', type=int, default=20, help='Prune every N episodes')
    parser.add_argument('--prune-count', type=int, default=10, help='Features to prune')
    parser.add_argument('--max-steps', type=int, default=500, help='Max steps per episode')
    parser.add_argument('--classes', nargs='+', default=None, 
                       help='Specific classes (crypto forex metals commodities indices)')
    
    args = parser.parse_args()
    
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║                   SUPERPOT BY ASSET CLASS                            ║
║         Find UNIVERSAL features that work across ALL markets         ║
╚══════════════════════════════════════════════════════════════════════╝
""")
    
    # Discover and group files
    print("📁 Discovering data files...")
    all_files = discover_files()
    grouped = group_files_by_class(all_files)
    
    print(f"\n📊 Files by asset class:")
    for cls, files in sorted(grouped.items()):
        print(f"   {cls:15s}: {len(files)} files")
    
    # Filter classes if specified
    if args.classes:
        grouped = {k: v for k, v in grouped.items() if k in args.classes}
    
    if not grouped:
        print("❌ No data files found!")
        return
    
    # Train each class
    start_time = time.time()
    results = {}
    
    for asset_class, files in sorted(grouped.items()):
        if len(files) < 3:
            print(f"\n⚠️  Skipping {asset_class} (only {len(files)} files)")
            continue
        
        trainer = AssetClassTrainer(asset_class, files)
        result = trainer.train(
            episodes=args.episodes,
            prune_every=args.prune_every,
            prune_count=args.prune_count,
            max_steps=args.max_steps,
            verbose=True
        )
        results[asset_class] = result
    
    elapsed = time.time() - start_time
    
    # Cross-class analysis
    print(f"\n{'='*70}")
    print("CROSS-CLASS ANALYSIS")
    print(f"{'='*70}")
    
    common_analysis = find_common_features(results)
    rankings = compute_feature_rankings(results)
    
    # Print results per class
    print(f"\n📊 RESULTS BY ASSET CLASS:")
    print("-" * 70)
    
    for asset_class, result in sorted(results.items()):
        print(f"\n{asset_class.upper()}:")
        print(f"  Episodes: {result['episodes']} | Avg PnL: ${result['avg_pnl']:+.0f} | "
              f"Win rate: {result['win_rate']:.0f}%")
        print(f"  Surviving features: {result['n_surviving']}/{result['n_surviving'] + result['n_pruned']}")
        print(f"  Top 5:")
        for i, (feat, score) in enumerate(result['top_features'][:5]):
            print(f"    {i+1}. {feat:<35s} ({score:.4f})")
    
    # Universal features
    print(f"\n{'='*70}")
    print("🌍 UNIVERSAL FEATURES (survive in ALL classes)")
    print(f"{'='*70}")
    
    if common_analysis['universal']:
        for i, feat in enumerate(common_analysis['universal'][:20]):
            print(f"  {i+1:2d}. {feat}")
    else:
        print("  No features survived in ALL classes")
    
    # Common features (3+ classes)
    print(f"\n{'='*70}")
    print("🔗 COMMON FEATURES (survive in 3+ classes)")
    print(f"{'='*70}")
    
    if common_analysis['common']:
        for feat, classes in common_analysis['common'][:20]:
            print(f"  {feat:<40s} [{', '.join(classes)}]")
    else:
        print("  No features common to 3+ classes")
    
    # Top ranked features overall
    print(f"\n{'='*70}")
    print("🏆 TOP FEATURES BY COMBINED RANKING")
    print(f"{'='*70}")
    print(f"{'Feature':<40s} {'Score':>8s} {'Imp':>8s} {'Classes':>8s}")
    print("-" * 70)
    
    for i, r in enumerate(rankings[:30]):
        classes_str = ','.join([c[:3] for c in r['classes']])
        print(f"{r['feature']:<40s} {r['combined_score']:>8.4f} {r['avg_importance']:>8.4f} {classes_str:>8s}")
    
    # Class-specific features
    print(f"\n{'='*70}")
    print("🎯 CLASS-SPECIFIC FEATURES (unique to one class)")
    print(f"{'='*70}")
    
    for cls, features in sorted(common_analysis['specific'].items()):
        if features:
            print(f"\n{cls.upper()}:")
            for feat in features[:5]:
                print(f"  - {feat}")
            if len(features) > 5:
                print(f"  ... and {len(features) - 5} more")
    
    # Summary statistics
    print(f"\n{'='*70}")
    print("📈 SUMMARY")
    print(f"{'='*70}")
    
    n_universal = len(common_analysis['universal'])
    n_common = len(common_analysis['common'])
    n_specific = sum(len(f) for f in common_analysis['specific'].values())
    
    print(f"\n  Universal features (all classes):  {n_universal}")
    print(f"  Common features (3+ classes):      {n_common}")
    print(f"  Class-specific features:           {n_specific}")
    
    # Key insight
    print(f"\n{'='*70}")
    print("💡 KEY INSIGHT")
    print(f"{'='*70}")
    
    if common_analysis['universal']:
        print(f"\n  The market has revealed {n_universal} UNIVERSAL TRUTHS:")
        print(f"  These features matter regardless of asset class:")
        for feat in common_analysis['universal'][:10]:
            print(f"    ✓ {feat}")
    elif rankings:
        print(f"\n  Top features with strongest cross-class signal:")
        for r in rankings[:5]:
            print(f"    ✓ {r['feature']} (score: {r['combined_score']:.4f})")
    
    print(f"\n  THE MARKET HAS SPOKEN!")
    
    # Save results
    results_dir = Path("results/superpot")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = results_dir / f"superpot_by_class_{timestamp}.json"
    
    output = {
        'timestamp': timestamp,
        'episodes_per_class': args.episodes,
        'elapsed_seconds': elapsed,
        'classes_trained': list(results.keys()),
        'by_class': {
            cls: {
                'avg_pnl': r['avg_pnl'],
                'win_rate': r['win_rate'],
                'n_surviving': r['n_surviving'],
                'top_features': [(str(f), float(s)) for f, s in r['top_features'][:20]],
            }
            for cls, r in results.items()
        },
        'universal_features': common_analysis['universal'],
        'common_features': [(str(f), c) for f, c in common_analysis['common']],
        'top_rankings': [
            {
                'feature': str(r['feature']),
                'combined_score': float(r['combined_score']),
                'avg_importance': float(r['avg_importance']),
                'n_classes': int(r['n_classes']),
            }
            for r in rankings[:50]
        ],
    }
    
    with open(results_file, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n💾 Results saved: {results_file}")
    print(f"⏱️  Total time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    
    print(f"\n{'='*70}")
    print("SUPERPOT BY CLASS COMPLETE")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
