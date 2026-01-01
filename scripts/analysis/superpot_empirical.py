#!/usr/bin/env python3
"""
SUPERPOT EMPIRICAL TESTING
==========================

Execute comprehensive empirical testing with ALL measurements:
1. Combine physics-based + traditional measurements (~300+ features)
2. Test across all asset classes, timeframes, and instruments
3. Prune worst performers adaptively
4. Discover universal vs class-specific vs instrument-specific features
5. Statistical validation (p < 0.01)
6. Generate empirical theorems

Philosophy:
- NO assumptions about what matters
- Let data decide through rigorous testing
- Adaptive pruning (not fixed intervals)
- Statistical significance required for all claims

Usage:
    python scripts/analysis/superpot_empirical.py
    python scripts/analysis/superpot_empirical.py --quick
    python scripts/analysis/superpot_empirical.py --asset-class crypto
    python scripts/analysis/superpot_empirical.py --timeframe H1
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime
from collections import defaultdict, Counter
import json
import time
import warnings
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent.parent))

# Import existing extractors
try:
    from scripts.analysis.superpot_physics import PhysicsExtractor
except (ImportError, ModuleNotFoundError):
    PhysicsExtractor = None

try:
    from scripts.analysis.superpot_explorer import SuperPotExtractor
except (ImportError, ModuleNotFoundError):
    SuperPotExtractor = None


# =============================================================================
# UNIFIED MEASUREMENT EXTRACTOR
# =============================================================================

class UnifiedMeasurementExtractor:
    """
    Combine ALL possible measurements:
    - Physics-based (~150 features)
    - Traditional TA-derived (~150 features)
    - Total: 300+ measurements
    
    No filtering. No assumptions. Everything goes in.
    """
    
    def __init__(self, lookback: int = 50):
        self.lookback = lookback
        self.physics_extractor = None
        self.traditional_extractor = None
        
        # Try to use existing extractors if available
        if PhysicsExtractor:
            self.physics_extractor = PhysicsExtractor(lookback)
        if SuperPotExtractor:
            self.traditional_extractor = SuperPotExtractor(lookback)
        
        self.feature_names = self._build_feature_names()
        print(f"📊 UnifiedMeasurementExtractor initialized with {len(self.feature_names)} measurements")
    
    def _build_feature_names(self) -> List[str]:
        """Build complete list of all measurement names."""
        names = []
        
        # Physics-based features
        if self.physics_extractor:
            physics_names = [f"phys_{n}" for n in self.physics_extractor.feature_names]
            names.extend(physics_names)
        
        # Traditional features
        if self.traditional_extractor:
            trad_names = [f"trad_{n}" for n in self.traditional_extractor.feature_names]
            names.extend(trad_names)
        
        # If extractors not available, use minimal set
        if not names:
            names = self._build_minimal_features()
        
        return names
    
    def _build_minimal_features(self) -> List[str]:
        """Minimal feature set if extractors not available."""
        return [
            # Basic price action
            'return_1', 'return_5', 'return_10', 'return_20',
            'volatility_5', 'volatility_10', 'volatility_20',
            'momentum_5', 'momentum_10', 'momentum_20',
            
            # Volume
            'volume_ratio_5', 'volume_ratio_10',
            'volume_trend_5', 'volume_trend_10',
            
            # Energy proxies
            'kinetic_energy_5', 'kinetic_energy_10',
            'potential_energy_5', 'potential_energy_10',
            
            # Entropy
            'entropy_5', 'entropy_10',
            
            # Tail behavior
            'left_tail_5', 'right_tail_5',
            'tail_asymmetry_5',
        ]
    
    @property
    def n_features(self) -> int:
        return len(self.feature_names)
    
    def extract(self, df: pd.DataFrame, idx: int) -> np.ndarray:
        """Extract ALL measurements at given index."""
        if idx < self.lookback:
            return np.zeros(self.n_features, dtype=np.float32)
        
        features = []
        
        # Extract physics features
        if self.physics_extractor:
            phys_features = self.physics_extractor.extract(df, idx)
            features.extend(phys_features)
        
        # Extract traditional features
        if self.traditional_extractor:
            trad_features = self.traditional_extractor.extract(df, idx)
            features.extend(trad_features)
        
        # Fallback to minimal extraction
        if not features:
            features = self._extract_minimal(df, idx)
        
        return np.array(features, dtype=np.float32)
    
    def _extract_minimal(self, df: pd.DataFrame, idx: int) -> List[float]:
        """Minimal feature extraction if extractors not available."""
        window = df.iloc[max(0, idx - self.lookback):idx + 1]
        
        c = window['close'].values.astype(np.float64)
        h = window['high'].values.astype(np.float64)
        l = window['low'].values.astype(np.float64)
        v = window['volume'].values.astype(np.float64) if 'volume' in window else np.ones(len(window))
        
        ret = np.diff(c) / (c[:-1] + 1e-10)
        
        features = []
        fi = 0
        
        # Returns
        features.append(float(ret[-1]) if len(ret) > 0 else 0.0)
        features.append(float(np.mean(ret[-5:])) if len(ret) >= 5 else 0.0)
        features.append(float(np.mean(ret[-10:])) if len(ret) >= 10 else 0.0)
        features.append(float(np.mean(ret[-20:])) if len(ret) >= 20 else 0.0)
        
        # Volatility
        features.append(float(np.std(ret[-5:])) if len(ret) >= 5 else 0.0)
        features.append(float(np.std(ret[-10:])) if len(ret) >= 10 else 0.0)
        features.append(float(np.std(ret[-20:])) if len(ret) >= 20 else 0.0)
        
        # Momentum
        features.append(float((c[-1] - c[-5]) / (c[-5] + 1e-10)) if len(c) >= 5 else 0.0)
        features.append(float((c[-1] - c[-10]) / (c[-10] + 1e-10)) if len(c) >= 10 else 0.0)
        features.append(float((c[-1] - c[-20]) / (c[-20] + 1e-10)) if len(c) >= 20 else 0.0)
        
        # Volume
        vm = np.mean(v) + 1e-10
        features.append(float(v[-1] / vm))
        features.append(float(np.mean(v[-5:]) / vm) if len(v) >= 5 else 1.0)
        features.append(float(np.polyfit(np.arange(5), v[-5:], 1)[0] / vm) if len(v) >= 5 else 0.0)
        features.append(float(np.polyfit(np.arange(10), v[-10:], 1)[0] / vm) if len(v) >= 10 else 0.0)
        
        # Energy proxies
        features.append(float(np.std(ret[-5:]) ** 2) if len(ret) >= 5 else 0.0)
        features.append(float(np.std(ret[-10:]) ** 2) if len(ret) >= 10 else 0.0)
        features.append(float((h[-1] - l[-1]) / (c[-1] + 1e-10)))
        if len(c) >= 5:
            pe_5 = float(np.mean((h[-5:] - l[-5:]) / (c[-5:] + 1e-10)))
            features.append(pe_5)
        else:
            features.append(0.0)
        
        # Entropy
        def shannon_entropy(arr, bins=10):
            if len(arr) < 5:
                return 0.5
            hist, _ = np.histogram(arr, bins=bins)
            hist = hist / (np.sum(hist) + 1e-10)
            hist = hist[hist > 0]
            return float(-np.sum(hist * np.log(hist + 1e-10)) / np.log(bins))
        
        features.append(shannon_entropy(ret[-5:]) if len(ret) >= 5 else 0.5)
        features.append(shannon_entropy(ret[-10:]) if len(ret) >= 10 else 0.5)
        
        # Tails
        left_tail = float(np.percentile(ret[-5:], 5)) if len(ret) >= 5 else 0.0
        right_tail = float(np.percentile(ret[-5:], 95)) if len(ret) >= 5 else 0.0
        features.append(left_tail)
        features.append(right_tail)
        features.append(right_tail - left_tail)
        
        return np.array(features, dtype=np.float32)


# =============================================================================
# ADAPTIVE FEATURE IMPORTANCE TRACKER
# =============================================================================

class AdaptiveFeatureTracker:
    """
    Track feature importance with statistical rigor.
    
    Metrics:
    - Correlation with reward (Spearman, not Pearson - no linearity assumption)
    - Win/loss separation (effect size)
    - Consistency across episodes
    - Statistical significance (p-value)
    """
    
    def __init__(self, n_features: int, feature_names: List[str]):
        self.n_features = n_features
        self.feature_names = feature_names.copy()
        self.active_mask = np.ones(n_features, dtype=bool)
        
        # Importance tracking
        self.feature_history: List[np.ndarray] = []
        self.reward_history: List[float] = []
        self.action_history: List[int] = []
        
        # Statistical metrics
        self.importance_scores = np.zeros(n_features)
        self.p_values = np.ones(n_features)
        self.effect_sizes = np.zeros(n_features)
        
        # Pruning history
        self.pruned_features: List[str] = []
        self.pruning_rounds = 0
    
    @property
    def n_active(self) -> int:
        return np.sum(self.active_mask)
    
    def record(self, features: np.ndarray, action: int, reward: float):
        """Record observation for importance calculation."""
        self.feature_history.append(features.copy())
        self.reward_history.append(reward)
        self.action_history.append(action)
        
        # Keep history bounded
        if len(self.feature_history) > 20000:
            self.feature_history = self.feature_history[-10000:]
            self.reward_history = self.reward_history[-10000:]
            self.action_history = self.action_history[-10000:]
    
    def calculate_importance(self) -> Dict[str, np.ndarray]:
        """
        Calculate feature importance with statistical validation.
        
        Returns:
            Dict with 'scores', 'p_values', 'effect_sizes'
        """
        if len(self.feature_history) < 100:
            return {
                'scores': np.ones(self.n_features),
                'p_values': np.ones(self.n_features),
                'effect_sizes': np.zeros(self.n_features)
            }
        
        features = np.array(self.feature_history)
        rewards = np.array(self.reward_history)
        
        scores = np.zeros(self.n_features)
        p_values = np.ones(self.n_features)
        effect_sizes = np.zeros(self.n_features)
        
        for i in range(self.n_features):
            if not self.active_mask[i]:
                scores[i] = -999
                continue
            
            f = features[:, i]
            
            # Skip if no variance
            if np.std(f) < 1e-10:
                continue
            
            # Spearman correlation (no linearity assumption)
            corr, p_val = stats.spearmanr(f, rewards)
            if not np.isnan(corr):
                scores[i] = abs(corr)
                p_values[i] = p_val
            
            # Effect size (Cohen's d) between winning and losing episodes
            win_mask = rewards > 0
            n_win = np.sum(win_mask)
            n_lose = np.sum(~win_mask)
            
            if n_win > 10 and n_lose > 10:
                win_mean = np.mean(f[win_mask])
                lose_mean = np.mean(f[~win_mask])
                pooled_std = np.sqrt(
                    ((n_win - 1) * np.var(f[win_mask]) + 
                     (n_lose - 1) * np.var(f[~win_mask])) / 
                    (n_win + n_lose - 2)
                )
                
                if pooled_std > 1e-10:
                    cohens_d = abs(win_mean - lose_mean) / pooled_std
                    effect_sizes[i] = cohens_d
                    
                    # Boost score if effect size is large
                    if cohens_d > 0.5:
                        scores[i] += cohens_d * 0.3
        
        self.importance_scores = scores
        self.p_values = p_values
        self.effect_sizes = effect_sizes
        
        return {
            'scores': scores,
            'p_values': p_values,
            'effect_sizes': effect_sizes
        }
    
    def prune_adaptive(self, min_keep: int = 20) -> List[str]:
        """
        Adaptive pruning based on statistical insignificance.
        
        Prune features that are:
        1. Not statistically significant (p > 0.05)
        2. Low effect size (Cohen's d < 0.2)
        3. Low importance score
        
        Keep at least min_keep features.
        """
        metrics = self.calculate_importance()
        scores = metrics['scores']
        p_vals = metrics['p_values']
        effect_sizes = metrics['effect_sizes']
        
        pruned = []
        active_indices = np.where(self.active_mask)[0]
        
        # Find candidates for pruning
        for idx in active_indices:
            # Don't prune if we're at minimum
            if self.n_active <= min_keep:
                break
            
            # Prune if statistically insignificant AND low effect size
            if p_vals[idx] > 0.05 and effect_sizes[idx] < 0.2 and scores[idx] < 0.1:
                self.active_mask[idx] = False
                pruned.append(self.feature_names[idx])
        
        # If nothing was pruned by statistical criteria, prune worst performers
        if len(pruned) == 0 and self.n_active > min_keep * 2:
            active_indices = np.where(self.active_mask)[0]
            active_scores = scores[active_indices]
            sorted_indices = active_indices[np.argsort(active_scores)]
            
            # Prune bottom 10%
            n_to_prune = max(1, int(len(sorted_indices) * 0.1))
            n_to_prune = min(n_to_prune, len(sorted_indices) - min_keep)
            
            for i in range(n_to_prune):
                idx = sorted_indices[i]
                self.active_mask[idx] = False
                pruned.append(self.feature_names[idx])
        
        self.pruned_features.extend(pruned)
        self.pruning_rounds += 1
        
        return pruned
    
    def get_top_features(self, n: int = 20) -> List[Tuple[str, float, float, float]]:
        """
        Get top N features by importance.
        
        Returns:
            List of (name, score, p_value, effect_size)
        """
        metrics = self.calculate_importance()
        scores = metrics['scores']
        p_vals = metrics['p_values']
        effect_sizes = metrics['effect_sizes']
        
        active_indices = np.where(self.active_mask)[0]
        sorted_indices = active_indices[np.argsort(scores[active_indices])[::-1]]
        
        return [
            (self.feature_names[i], scores[i], p_vals[i], effect_sizes[i])
            for i in sorted_indices[:n]
        ]
    
    def get_statistically_significant(self, alpha: float = 0.01) -> List[Tuple[str, float, float]]:
        """
        Get features that are statistically significant at level alpha.
        
        Returns:
            List of (name, p_value, effect_size)
        """
        metrics = self.calculate_importance()
        p_vals = metrics['p_values']
        effect_sizes = metrics['effect_sizes']
        
        # Apply Bonferroni correction for multiple testing
        alpha_corrected = alpha / self.n_active
        
        significant = []
        active_indices = np.where(self.active_mask)[0]
        
        for idx in active_indices:
            if p_vals[idx] < alpha_corrected:
                significant.append((
                    self.feature_names[idx],
                    p_vals[idx],
                    effect_sizes[idx]
                ))
        
        # Sort by p-value (most significant first)
        significant.sort(key=lambda x: x[1])
        
        return significant
    
    def mask_features(self, features: np.ndarray) -> np.ndarray:
        """Return only active features."""
        return features[self.active_mask]


# =============================================================================
# SIMPLE AGENT FOR EMPIRICAL TESTING
# =============================================================================

class SimpleRLAgent:
    """
    Simple Q-learning agent for empirical testing.
    Focus is on feature discovery, not sophisticated RL.
    """
    
    def __init__(self, n_features: int, n_actions: int = 4):
        self.n_features = n_features
        self.n_actions = n_actions
        
        # Linear Q-function: Q(s, a) = W[a] · s
        self.W = np.random.randn(n_actions, n_features) * 0.01
        
        # Adam optimizer
        self.V_W = np.zeros_like(self.W)
        self.m_W = np.zeros_like(self.W)
        
        self.lr = 0.001
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.05
        
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.adam_t = 0
    
    def select_action(self, state: np.ndarray, explore: bool = True) -> int:
        """Select action using ε-greedy."""
        if explore and np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        
        # Clip state to avoid overflow
        state = np.clip(state, -10, 10)
        q_values = self.W @ state
        return int(np.argmax(q_values))
    
    def update(self, state: np.ndarray, action: int, reward: float, 
               next_state: np.ndarray, done: bool):
        """Update Q-function with TD error."""
        state = np.clip(state, -10, 10)
        next_state = np.clip(next_state, -10, 10)
        
        # TD target
        if done:
            target = reward
        else:
            target = reward + self.gamma * np.max(self.W @ next_state)
        
        # Current Q-value
        current = self.W[action] @ state
        
        # TD error
        td_error = target - current
        
        # Gradient
        grad = -td_error * state
        
        # Adam update
        self.adam_t += 1
        self.m_W[action] = self.beta1 * self.m_W[action] + (1 - self.beta1) * grad
        self.V_W[action] = self.beta2 * self.V_W[action] + (1 - self.beta2) * (grad ** 2)
        
        m_hat = self.m_W[action] / (1 - self.beta1 ** self.adam_t)
        v_hat = self.V_W[action] / (1 - self.beta2 ** self.adam_t)
        
        self.W[action] -= self.lr * m_hat / (np.sqrt(v_hat) + 1e-8)
        
        # Decay epsilon
        if done:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


# =============================================================================
# ASSET CLASSIFICATION
# =============================================================================

def classify_asset(symbol: str) -> str:
    """Classify asset by type."""
    s = symbol.upper().replace('+', '').replace('-', '')
    
    if any(x in s for x in ['BTC', 'ETH', 'XRP', 'LTC', 'DOGE', 'SOL', 'ADA']):
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
    """Classify timeframe by trading style."""
    tf = tf.upper()
    if tf in ['M1', 'M5', 'M15']:
        return 'scalp'
    elif tf in ['M30', 'H1']:
        return 'intraday'
    elif tf in ['H4', 'D1']:
        return 'swing'
    return tf


# =============================================================================
# DATA LOADING
# =============================================================================

def load_data(filepath: str) -> pd.DataFrame:
    """Load and clean data."""
    df = pd.read_csv(filepath, sep='\t')
    
    # Normalize column names - remove angle brackets and convert to lowercase
    df.columns = df.columns.str.replace('<', '').str.replace('>', '').str.lower().str.strip()
    
    # Handle different naming conventions
    rename_map = {
        'tickvol': 'volume',
    }
    df.rename(columns=rename_map, inplace=True)
    
    # Ensure required columns
    required = ['open', 'high', 'low', 'close']
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing column: {col} in {filepath}")
    
    # Convert price columns to float
    for col in required:
        df[col] = pd.to_numeric(df[col], errors='coerce').astype(float)
    
    # Handle volume - prefer 'vol' if present, otherwise use 'volume'
    if 'vol' in df.columns and df['vol'].sum() > 0:
        df['volume'] = pd.to_numeric(df['vol'], errors='coerce').astype(float)
    elif 'volume' not in df.columns or df['volume'].sum() == 0:
        df['volume'] = 1.0
    else:
        df['volume'] = pd.to_numeric(df['volume'], errors='coerce').astype(float)
    
    # Remove rows with NaN in required columns
    df = df.dropna(subset=required)
    
    # Basic sanity check - ensure OHLC relationship holds
    df = df[(df['high'] >= df['low']) & 
            (df['high'] >= df['open']) & 
            (df['high'] >= df['close']) &
            (df['low'] <= df['open']) & 
            (df['low'] <= df['close'])]
    
    return df


def discover_files(base_paths: Optional[List[str]] = None) -> List[dict]:
    """Find all data files."""
    if base_paths is None:
        base_paths = ["data/master", "data/runs/berserker_run3/data", "data"]
    
    files = []
    for base in base_paths:
        path = Path(base)
        if not path.exists():
            continue
        
        for f in path.rglob("*.csv"):
            if 'symbol' in f.name.lower() or 'info' in f.name.lower():
                continue
            
            parts = f.stem.split('_')
            if len(parts) >= 2:
                symbol = parts[0]
                timeframe = parts[1]
                asset_class = classify_asset(symbol)
                tf_class = classify_timeframe(timeframe)
                
                files.append({
                    'path': str(f),
                    'symbol': symbol,
                    'timeframe': timeframe,
                    'asset_class': asset_class,
                    'tf_class': tf_class
                })
    
    return files


# =============================================================================
# EMPIRICAL TESTING ENGINE
# =============================================================================

def run_empirical_testing(
    files: List[dict],
    extractor: UnifiedMeasurementExtractor,
    tracker: AdaptiveFeatureTracker,
    episodes: int = 100,
    max_steps: int = 500,
    prune_every: int = 20,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Run empirical testing with adaptive pruning.
    
    Returns:
        Results dictionary with performance metrics and feature rankings
    """
    agent = SimpleRLAgent(tracker.n_active, n_actions=4)
    
    all_rewards = []
    all_pnls = []
    episode_info = []
    
    start_time = time.time()
    
    for ep in range(episodes):
        # Select random file
        file_info = files[np.random.randint(len(files))]
        
        try:
            df = load_data(file_info['path'])
            if len(df) < 200:
                continue
            
            # Use recent data
            df = df.iloc[-2000:].reset_index(drop=True)
            
            # Initialize episode
            start_bar = np.random.randint(100, max(101, len(df) - max_steps - 10))
            bar = start_bar
            position = 0
            entry_price = 0
            balance = 10000
            episode_reward = 0
            trades = []
            
            # Episode loop
            for step in range(max_steps):
                if bar >= len(df) - 1:
                    break
                
                # Extract features
                features = extractor.extract(df, bar)
                active_features = tracker.mask_features(features)
                
                # Agent action
                action = agent.select_action(active_features, explore=True)
                
                # Execute action
                price = df.iloc[bar]['close']
                reward = 0
                
                if action == 1 and position == 0:  # Buy
                    position = 1
                    entry_price = price * 1.0001  # Slippage
                elif action == 2 and position == 0:  # Sell
                    position = -1
                    entry_price = price * 0.9999  # Slippage
                elif action == 3 and position != 0:  # Close
                    if position == 1:
                        pnl = (price * 0.9999 - entry_price) / entry_price
                    else:
                        pnl = (entry_price - price * 1.0001) / entry_price
                    
                    balance *= (1 + pnl * 0.1)  # 10% position size
                    reward = pnl * 100
                    position = 0
                    
                    trades.append(pnl)
                
                # Holding penalty
                if position != 0:
                    reward -= 0.001
                
                # Next state
                bar += 1
                next_features = extractor.extract(df, bar) if bar < len(df) else features
                active_next = tracker.mask_features(next_features)
                
                # Track for importance
                tracker.record(features, action, reward)
                
                # Update agent
                agent.update(active_features, action, reward, active_next, bar >= len(df) - 1)
                
                episode_reward += reward
            
            # End of episode
            pnl = balance - 10000
            all_rewards.append(episode_reward)
            all_pnls.append(pnl)
            
            episode_info.append({
                'episode': ep,
                'reward': episode_reward,
                'pnl': pnl,
                'trades': len(trades),
                'symbol': file_info['symbol'],
                'timeframe': file_info['timeframe'],
                'asset_class': file_info['asset_class']
            })
            
            # Progress
            if verbose and (ep + 1) % 10 == 0:
                avg_r = np.mean(all_rewards[-10:])
                avg_pnl = np.mean(all_pnls[-10:])
                print(f"Ep {ep+1:3d}: R={avg_r:+7.1f} PnL=${avg_pnl:+7.0f} | "
                      f"Active: {tracker.n_active}/{extractor.n_features} | "
                      f"ε={agent.epsilon:.3f}")
            
            # Adaptive pruning
            if (ep + 1) % prune_every == 0 and tracker.n_active > 30:
                pruned = tracker.prune_adaptive(min_keep=20)
                
                if verbose and pruned:
                    print(f"\n🗑️  PRUNED {len(pruned)} features:")
                    for f in pruned[:5]:
                        print(f"   - {f}")
                    if len(pruned) > 5:
                        print(f"   ... and {len(pruned) - 5} more")
                    print(f"   Remaining: {tracker.n_active} features\n")
                
                # Resize agent
                if pruned:
                    active_indices = np.where(tracker.active_mask)[0]
                    agent.W = agent.W[:, active_indices]
                    agent.V_W = agent.V_W[:, active_indices]
                    agent.m_W = agent.m_W[:, active_indices]
                    agent.n_features = tracker.n_active
        
        except (ValueError, KeyError, IndexError, TypeError) as e:
            if verbose:
                import traceback
                print(f"Error in episode {ep}: {e}")
                if ep == 0:  # Show traceback for first error
                    traceback.print_exc()
            continue
    
    elapsed = time.time() - start_time
    
    # Final metrics
    results = {
        'episodes': len(all_rewards),
        'total_time': elapsed,
        'avg_reward': float(np.mean(all_rewards)),
        'std_reward': float(np.std(all_rewards)),
        'avg_pnl': float(np.mean(all_pnls)),
        'std_pnl': float(np.std(all_pnls)),
        'win_rate': float(sum(1 for p in all_pnls if p > 0) / len(all_pnls)) if all_pnls else 0,
        'initial_features': extractor.n_features,
        'surviving_features': tracker.n_active,
        'pruned_count': extractor.n_features - tracker.n_active,
        'top_features': tracker.get_top_features(30),
        'statistically_significant': tracker.get_statistically_significant(alpha=0.01),
        'episode_info': episode_info,
    }
    
    return results


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='SUPERPOT Empirical Testing - ALL measurements, adaptive pruning'
    )
    parser.add_argument('--episodes', type=int, default=100, 
                       help='Total episodes to run')
    parser.add_argument('--max-steps', type=int, default=500,
                       help='Max steps per episode')
    parser.add_argument('--prune-every', type=int, default=20,
                       help='Prune features every N episodes')
    parser.add_argument('--max-files', type=int, default=30,
                       help='Max data files to use')
    parser.add_argument('--asset-class', type=str, default=None,
                       help='Filter by asset class (crypto, forex, metals, etc.)')
    parser.add_argument('--timeframe', type=str, default=None,
                       help='Filter by timeframe (M15, H1, H4, etc.)')
    parser.add_argument('--quick', action='store_true',
                       help='Quick test (fewer episodes)')
    
    args = parser.parse_args()
    
    # Quick mode
    if args.quick:
        args.episodes = 50
        args.max_steps = 200
        args.max_files = 10
    
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║                  SUPERPOT EMPIRICAL TESTING                          ║
║     Execute comprehensive empirical testing with ALL measurements    ║
║              Prune worst, discover what matters (p < 0.01)           ║
╚══════════════════════════════════════════════════════════════════════╝
""")
    
    # Discover data
    print("🔍 Discovering data files...")
    all_files = discover_files()
    
    # Filter by asset class
    if args.asset_class:
        all_files = [f for f in all_files if f['asset_class'] == args.asset_class.lower()]
        print(f"   Filtered to {args.asset_class}: {len(all_files)} files")
    
    # Filter by timeframe
    if args.timeframe:
        all_files = [f for f in all_files if f['timeframe'].upper() == args.timeframe.upper()]
        print(f"   Filtered to {args.timeframe}: {len(all_files)} files")
    
    # Limit files
    files = all_files[:args.max_files]
    
    print(f"📁 Using {len(files)} data files")
    if not files:
        print("❌ No data files found!")
        return
    
    # Show file distribution
    asset_dist = Counter(f['asset_class'] for f in files)
    tf_dist = Counter(f['timeframe'] for f in files)
    print(f"\n📊 Asset classes: {dict(asset_dist)}")
    print(f"⏱️  Timeframes: {dict(tf_dist)}\n")
    
    # Initialize
    print("🔧 Initializing measurement extractors...")
    extractor = UnifiedMeasurementExtractor(lookback=50)
    tracker = AdaptiveFeatureTracker(extractor.n_features, extractor.feature_names)
    
    print(f"\n🧪 Starting empirical testing:")
    print(f"   Features: {extractor.n_features}")
    print(f"   Episodes: {args.episodes}")
    print(f"   Adaptive pruning every {args.prune_every} episodes")
    print(f"   Statistical validation: p < 0.01 (Bonferroni corrected)\n")
    
    # Run empirical testing
    results = run_empirical_testing(
        files=files,
        extractor=extractor,
        tracker=tracker,
        episodes=args.episodes,
        max_steps=args.max_steps,
        prune_every=args.prune_every,
        verbose=True
    )
    
    # Display results
    print(f"\n{'='*70}")
    print("SUPERPOT EMPIRICAL TESTING RESULTS")
    print(f"{'='*70}")
    
    print(f"\n📊 Performance Metrics:")
    print(f"   Episodes completed: {results['episodes']}")
    print(f"   Avg reward: {results['avg_reward']:+.2f} ± {results['std_reward']:.2f}")
    print(f"   Avg PnL: ${results['avg_pnl']:+.0f} ± ${results['std_pnl']:.0f}")
    print(f"   Win rate: {results['win_rate']*100:.1f}%")
    print(f"   Time: {results['total_time']:.1f}s")
    
    print(f"\n🔬 Feature Discovery:")
    print(f"   Initial features: {results['initial_features']}")
    print(f"   Surviving features: {results['surviving_features']}")
    print(f"   Pruned: {results['pruned_count']} ({results['pruned_count']/results['initial_features']*100:.1f}%)")
    
    # Statistically significant features
    sig_features = results['statistically_significant']
    print(f"\n⭐ STATISTICALLY SIGNIFICANT FEATURES (p < 0.01, Bonferroni corrected):")
    print(f"   Found: {len(sig_features)}\n")
    
    if sig_features:
        for i, (name, p_val, effect_size) in enumerate(sig_features[:20], 1):
            print(f"   {i:2d}. {name:<40s} p={p_val:.6f} d={effect_size:.3f}")
    else:
        print("   (None found - increase sample size or relax alpha)")
    
    # Top features by importance
    print(f"\n🏆 TOP FEATURES BY IMPORTANCE SCORE:")
    top_features = results['top_features']
    for i, (name, score, p_val, effect_size) in enumerate(top_features[:20], 1):
        sig_marker = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
        print(f"   {i:2d}. {name:<40s} score={score:.4f} p={p_val:.4f} d={effect_size:.3f} {sig_marker}")
    
    # Save results
    results_dir = Path("results/superpot")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Filter suffix
    suffix = ""
    if args.asset_class:
        suffix += f"_{args.asset_class}"
    if args.timeframe:
        suffix += f"_{args.timeframe}"
    
    results_file = results_dir / f"empirical{suffix}_{timestamp}.json"
    
    # Prepare output (convert tuples to lists for JSON)
    output = {
        'timestamp': timestamp,
        'config': {
            'episodes': args.episodes,
            'max_steps': args.max_steps,
            'prune_every': args.prune_every,
            'asset_class': args.asset_class,
            'timeframe': args.timeframe,
        },
        'metrics': {
            'episodes': int(results['episodes']),
            'avg_reward': float(results['avg_reward']) if not np.isnan(results['avg_reward']) else 0.0,
            'std_reward': float(results['std_reward']) if not np.isnan(results['std_reward']) else 0.0,
            'avg_pnl': float(results['avg_pnl']) if not np.isnan(results['avg_pnl']) else 0.0,
            'std_pnl': float(results['std_pnl']) if not np.isnan(results['std_pnl']) else 0.0,
            'win_rate': float(results['win_rate']),
            'total_time': float(results['total_time']),
        },
        'features': {
            'initial': int(results['initial_features']),
            'surviving': int(results['surviving_features']),
            'pruned': int(results['pruned_count']),
        },
        'top_features': [
            {
                'name': name,
                'score': float(score),
                'p_value': float(p_val),
                'effect_size': float(effect_size)
            }
            for name, score, p_val, effect_size in results['top_features']
        ],
        'statistically_significant': [
            {
                'name': name,
                'p_value': float(p_val),
                'effect_size': float(effect_size)
            }
            for name, p_val, effect_size in results['statistically_significant']
        ],
    }
    
    with open(results_file, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n💾 Results saved: {results_file}")
    print("\n" + "="*70)
    print("EMPIRICAL TESTING COMPLETE")
    print("="*70)
    print("\nNext steps:")
    print("1. Review statistically significant features")
    print("2. Test universal vs class-specific vs instrument-specific")
    print("3. Validate findings out-of-sample")
    print("4. Document empirical theorems (p < 0.01)")


if __name__ == '__main__':
    main()
