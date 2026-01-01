#!/usr/bin/env python3
"""
Comprehensive Exploration Runner with PHYSICS-Based Discovery Engine
=====================================================================

Uses PHYSICS-ONLY measurements. NO traditional indicators.
Discovers what works per asset class - NO assumptions.

Key Physics Features:
- Kinematics: velocity, acceleration, jerk, snap, crackle, pop (6 derivatives)
- Energy: kinetic, potential (compression + displacement), efficiency
- Flow dynamics: Reynolds number, damping, viscosity, liquidity
- Thermodynamics: entropy, entropy rate, phase compression
- Field theory: gradients, divergence, buying pressure, body ratio
- Microstructure: spread percentile, volume surge, volume trend

What We DON'T Use:
- NO RSI, MACD, Bollinger Bands, ADX, Aroon
- NO hardcoded periods
- NO magic numbers

Philosophy: RL discovers patterns from physics state - not from traditional indicators.
"""

import json
import shutil
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

# Import measurement framework using direct file imports to avoid kinetra/__init__.py
# which requires backtesting library
import importlib.util

def _load_module(name: str, file_path: str):
    """Load a module directly from file path."""
    spec = importlib.util.spec_from_file_location(name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

try:
    _kinetra_path = Path(__file__).parent / 'kinetra'

    # Load modules directly
    _measurements = _load_module('measurements', _kinetra_path / 'measurements.py')
    _composite = _load_module('composite_stacking', _kinetra_path / 'composite_stacking.py')
    _integration = _load_module('exploration_integration', _kinetra_path / 'exploration_integration.py')
    _multi_agent = _load_module('multi_agent_design', _kinetra_path / 'multi_agent_design.py')
    _trend_discovery = _load_module('trend_discovery', _kinetra_path / 'trend_discovery.py')

    # Import classes
    MeasurementEngine = _measurements.MeasurementEngine
    CorrelationExplorer = _measurements.CorrelationExplorer

    ExplorationEngine = _composite.ExplorationEngine
    CompositeStacker = _composite.CompositeStacker
    ClassDiscoveryEngine = _composite.ClassDiscoveryEngine

    ExplorationDataLoader = _integration.ExplorationDataLoader
    ExplorationFeatureExtractor = _integration.ExplorationFeatureExtractor
    DiscoveryReporter = _integration.DiscoveryReporter

    AssetClass = _multi_agent.AssetClass
    get_asset_class = _multi_agent.get_asset_class
    get_instrument_profile = _multi_agent.get_instrument_profile
    INSTRUMENT_PROFILES = _multi_agent.INSTRUMENT_PROFILES
    TradeValidator = _multi_agent.TradeValidator
    VolatilityEstimator = _multi_agent.VolatilityEstimator

    UnifiedStrategyLearner = _trend_discovery.UnifiedStrategyLearner
    TrendDefinitionLearner = _trend_discovery.TrendDefinitionLearner

    MEASUREMENT_FRAMEWORK_AVAILABLE = True
except Exception as e:
    print(f"[WARN] Measurement framework not fully available: {e}")
    import traceback
    traceback.print_exc()
    MEASUREMENT_FRAMEWORK_AVAILABLE = False

from rl_exploration_framework import (
    RewardShaper,
    LinearQAgent,
    FeatureTracker,
)


# =============================================================================
# ASSET CLASS TAXONOMY
# =============================================================================

ASSET_CLASS_MAP = {
    # Forex (Mean-Reverting, 24/5)
    "AUDJPY+": "Forex", "AUDUSD+": "Forex", "EURJPY+": "Forex",
    "GBPJPY+": "Forex", "GBPUSD+": "Forex",

    # Crypto (24/7, Trending, High Vol)
    "BTCJPY": "Crypto", "BTCUSD": "Crypto", "ETHEUR": "Crypto", "XRPJPY": "Crypto",

    # Equity Indices (Exchange Hours, Gap Risk)
    "DJ30ft": "EquityIndex", "NAS100": "EquityIndex", "Nikkei225": "EquityIndex",
    "EU50": "EquityIndex", "GER40": "EquityIndex", "SA40": "EquityIndex", "US2000": "EquityIndex",

    # Precious Metals (Trending, NOT Forex despite broker classification)
    "XAGUSD": "PreciousMetals", "XAUAUD+": "PreciousMetals",
    "XAUUSD+": "PreciousMetals", "XPTUSD": "PreciousMetals",

    # Energy & Industrial Commodities (Inventory Cycles)
    "COPPER-C": "EnergyCommodities", "UKOUSD": "EnergyCommodities",
}


def get_asset_class_from_key(instrument_key: str) -> str:
    """Get asset class from instrument key."""
    for prefix, cls in ASSET_CLASS_MAP.items():
        if instrument_key.startswith(prefix.rstrip('+')):
            return cls
    return "Forex"  # Default


# =============================================================================
# TRADING ENVIRONMENT WITH FULL MEASUREMENTS
# =============================================================================

class ComprehensiveTradingEnv:
    """
    Trading environment using comprehensive measurements.

    Features:
    - Full measurement engine (50+ features)
    - Composite signal stacking
    - Class-specific reward shaping
    - Trade validation
    - Strategy definition learning (what is "trend" for this class?)
    """

    def __init__(
        self,
        inst_measurements,  # InstrumentMeasurements object
        feature_extractor: ExplorationFeatureExtractor,
        reward_shaper: RewardShaper,
        strategy_learner: Optional['UnifiedStrategyLearner'] = None,
        trade_validator: Optional[TradeValidator] = None,
        max_bars: int = 500,
    ):
        self.inst_meas = inst_measurements
        self.extractor = feature_extractor
        self.reward_shaper = reward_shaper
        self.strategy_learner = strategy_learner
        self.validator = trade_validator or TradeValidator()
        self.max_bars = max_bars

        # Data arrays
        self.close = inst_measurements.raw_measurements.get('roc_5', None)
        if self.close is None:
            # Fallback - compute from ROC
            self.n_bars = len(list(inst_measurements.raw_measurements.values())[0])
        else:
            self.n_bars = len(self.close)

        # Get close prices from raw data
        # We need to reload close prices since measurements don't store them
        self._load_prices()

        # State
        self.current_bar = 0
        self.position = 0  # -1, 0, 1
        self.entry_price = 0.0
        self.entry_bar = 0
        self.mfe = 0.0
        self.mae = 0.0

        # Episode stats
        self.trades: List[Dict] = []
        self.total_pnl = 0.0

        # State/action dims
        self.state_dim = len(feature_extractor.get_feature_names())
        self.n_actions = 3  # Sell, Hold, Buy

    def _load_prices(self):
        """Load price data from original file."""
        # Find the CSV file for this instrument
        data_dir = Path("data/master_standardized")
        if not data_dir.exists():
            data_dir = Path("data/master")

        pattern = f"{self.inst_meas.instrument_key.split('_')[0]}*_{self.inst_meas.instrument_key.split('_')[1]}_*.csv"
        files = list(data_dir.glob(pattern))

        if files:
            df = pd.read_csv(files[0], sep='\t')
            df.columns = [c.lower().replace('<', '').replace('>', '') for c in df.columns]
            self.close_prices = df['close'].values.astype(float)
            self.high_prices = df['high'].values.astype(float)
            self.low_prices = df['low'].values.astype(float)
            self.spread = df.get('spread', pd.Series(np.ones(len(df)))).values.astype(float)
            self.volume = df.get('tickvol', df.get('vol', pd.Series(np.ones(len(df))))).values.astype(float)
        else:
            # Fallback
            self.close_prices = np.ones(self.n_bars)
            self.high_prices = np.ones(self.n_bars)
            self.low_prices = np.ones(self.n_bars)
            self.spread = np.ones(self.n_bars)
            self.volume = np.ones(self.n_bars)

    def reset(self) -> np.ndarray:
        """Reset environment for new episode."""
        # Start after warmup period for indicators
        self.current_bar = 200
        self.position = 0
        self.entry_price = 0.0
        self.entry_bar = 0
        self.mfe = 0.0
        self.mae = 0.0
        self.trades = []
        self.total_pnl = 0.0

        return self._get_state()

    def _get_state(self) -> np.ndarray:
        """Get current state (feature vector)."""
        try:
            features = self.extractor.get_features(self.inst_meas, self.current_bar)
            return features
        except Exception:
            return np.zeros(self.state_dim)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Execute action and return (next_state, reward, done, info).

        Actions: 0=Sell, 1=Hold, 2=Buy
        """
        reward = 0.0
        info = {"action": action}

        # Get current physics measurements
        bar_meas = {}
        for name in self.inst_meas.normalized_measurements:
            if self.current_bar < len(self.inst_meas.normalized_measurements[name]):
                val = self.inst_meas.normalized_measurements[name][self.current_bar]
                if np.isfinite(val):
                    bar_meas[name] = val

        # Use physics-based spread and volume measures
        spread_pct = bar_meas.get('spread_pct_pct', 0.5)  # Spread percentile
        volume_surge = bar_meas.get('volume_surge_pct', 0.5)  # Volume surge percentile

        # Current price
        close = self.close_prices[self.current_bar]
        high = self.high_prices[self.current_bar]
        low = self.low_prices[self.current_bar]

        # Position logic
        if self.position == 0:
            # No position - can open
            if action == 2:  # Buy
                self.position = 1
                self.entry_price = close
                self.entry_bar = self.current_bar
                self.mfe = 0.0
                self.mae = 0.0
                info["opened"] = "long"

            elif action == 0:  # Sell
                self.position = -1
                self.entry_price = close
                self.entry_bar = self.current_bar
                self.mfe = 0.0
                self.mae = 0.0
                info["opened"] = "short"

        else:
            # Have position - update MAE/MFE
            if self.position == 1:  # Long
                unrealized = (close - self.entry_price) / self.entry_price * 100
                max_favorable = (high - self.entry_price) / self.entry_price * 100
                max_adverse = (low - self.entry_price) / self.entry_price * 100
            else:  # Short
                unrealized = (self.entry_price - close) / self.entry_price * 100
                max_favorable = (self.entry_price - low) / self.entry_price * 100
                max_adverse = (self.entry_price - high) / self.entry_price * 100

            self.mfe = max(self.mfe, max_favorable)
            self.mae = min(self.mae, max_adverse)

            # Check for close
            should_close = False
            if self.position == 1 and action == 0:  # Long -> Sell
                should_close = True
            elif self.position == -1 and action == 2:  # Short -> Buy
                should_close = True

            if should_close:
                # Close position
                bars_held = self.current_bar - self.entry_bar
                pnl = unrealized

                # Get entry and exit features for reward shaping
                entry_features = self.extractor.get_features(self.inst_meas, self.entry_bar)
                exit_features = self._get_state()

                # Shape reward
                reward = self.reward_shaper.shape_reward(
                    raw_pnl=pnl,
                    mae=self.mae,
                    mfe=self.mfe,
                    bars_held=bars_held,
                    entry_features=entry_features,
                    exit_features=exit_features,
                    physics_state=pd.DataFrame(),  # TODO: Add physics state
                    bar_index=self.current_bar,
                )

                # Record trade with physics state
                trade = {
                    "entry_bar": self.entry_bar,
                    "exit_bar": self.current_bar,
                    "direction": "long" if self.position == 1 else "short",
                    "pnl": pnl,
                    "mae": self.mae,
                    "mfe": self.mfe,
                    "bars_held": bars_held,
                    "spread_pct_entry": bar_meas.get('spread_pct_pct', 0.5),
                    "volume_surge_entry": bar_meas.get('volume_surge_pct', 0.5),
                    "kinetic_energy_entry": bar_meas.get('kinetic_energy_pct', 0.5),
                    "reynolds_entry": bar_meas.get('reynolds_pct', 0.5),
                    "entropy_entry": bar_meas.get('entropy_pct', 0.5),
                }
                self.trades.append(trade)
                self.total_pnl += pnl

                # Record for discovery
                self.extractor.record_trade(
                    self.inst_meas,
                    self.entry_bar, self.current_bar,
                    pnl, self.mae, self.mfe, bars_held
                )

                # Record for strategy definition learning
                if self.strategy_learner is not None:
                    # Get entry measurements for learning what defines "trend"
                    entry_meas = {
                        name: self.inst_meas.normalized_measurements[name][self.entry_bar]
                        for name in self.inst_meas.normalized_measurements
                        if self.entry_bar < len(self.inst_meas.normalized_measurements[name])
                        and np.isfinite(self.inst_meas.normalized_measurements[name][self.entry_bar])
                    }

                    self.strategy_learner.record_trade(
                        asset_class=self.inst_meas.asset_class,
                        instrument=self.inst_meas.instrument_key,
                        entry_measurements=entry_meas,
                        pnl=pnl,
                        mae=self.mae,
                        mfe=self.mfe,
                        bars_held=bars_held,
                        direction=self.position,  # 1=long, -1=short
                    )

                # Reset position
                self.position = 0
                self.entry_price = 0.0
                info["closed"] = trade["direction"]
                info["pnl"] = pnl

        # Advance bar
        self.current_bar += 1

        # Check done
        done = (
            self.current_bar >= self.n_bars - 1 or
            self.current_bar >= 200 + self.max_bars
        )

        # Force close at end
        if done and self.position != 0:
            close = self.close_prices[min(self.current_bar, self.n_bars - 1)]
            if self.position == 1:
                pnl = (close - self.entry_price) / self.entry_price * 100
            else:
                pnl = (self.entry_price - close) / self.entry_price * 100

            self.total_pnl += pnl
            self.trades.append({
                "entry_bar": self.entry_bar,
                "exit_bar": self.current_bar,
                "direction": "long" if self.position == 1 else "short",
                "pnl": pnl,
                "mae": self.mae,
                "mfe": self.mfe,
                "bars_held": self.current_bar - self.entry_bar,
                "forced_close": True,
            })

        next_state = self._get_state() if not done else np.zeros(self.state_dim)

        info["bar"] = self.current_bar
        info["position"] = self.position
        info["total_pnl"] = self.total_pnl

        return next_state, reward, done, info

    def get_episode_stats(self) -> Dict:
        """Get stats for completed episode."""
        n_trades = len(self.trades)
        wins = [t for t in self.trades if t["pnl"] > 0]
        losses = [t for t in self.trades if t["pnl"] < 0]

        return {
            "instrument": self.inst_meas.instrument_key,
            "asset_class": self.inst_meas.asset_class,
            "trades": n_trades,
            "wins": len(wins),
            "losses": len(losses),
            "win_rate": len(wins) / n_trades if n_trades > 0 else 0,
            "total_pnl": self.total_pnl,
            "avg_pnl": self.total_pnl / n_trades if n_trades > 0 else 0,
            "avg_mae": np.mean([t["mae"] for t in self.trades]) if self.trades else 0,
            "avg_mfe": np.mean([t["mfe"] for t in self.trades]) if self.trades else 0,
        }


# =============================================================================
# MAIN EXPLORATION RUNNER
# =============================================================================

def run_comprehensive_exploration(
    data_dir: str = "data/master",
    episodes: int = 100,
    verbose: bool = True,
):
    """
    Run comprehensive exploration with discovery engine.
    """
    print("=" * 80)
    print("  COMPREHENSIVE EXPLORATION WITH DISCOVERY ENGINE")
    print("  50+ measurements | 5 signal generators | Class-specific learning")
    print("=" * 80)

    # Standardize data first
    print("\n[STEP 1] Standardizing data...")
    standardized_dir = standardize_data(data_dir)

    # Load with full measurements
    print("\n[STEP 2] Loading instruments with full measurements...")
    loader = ExplorationDataLoader(standardized_dir)
    n_loaded = loader.load_all(verbose=verbose)

    if n_loaded == 0:
        print("\n" + "=" * 80)
        print("[ERROR] No instruments loaded!")
        print("=" * 80)
        print("\n🔍 Troubleshooting:")
        print(f"  1. Check that data files exist in: {standardized_dir}")
        print(f"  2. Files should be in subdirectories (crypto/, forex/, etc.)")
        print(f"  3. File format: INSTRUMENT_TIMEFRAME_START_END.csv")
        print(f"  4. Run data download: python scripts/download/download_interactive.py")
        print(f"\n📁 Directory contents:")
        data_path = Path(standardized_dir)
        if data_path.exists():
            subdirs = [d.name for d in data_path.iterdir() if d.is_dir()]
            print(f"  Subdirectories: {subdirs if subdirs else 'None found'}")
            csv_count = len(list(data_path.glob("**/*.csv")))
            print(f"  Total CSV files: {csv_count}")
        else:
            print(f"  ❌ Directory does not exist: {standardized_dir}")
        print("=" * 80)
        return None

    # Create feature extractor
    extractor = ExplorationFeatureExtractor()

    # Create strategy learner - discovers what "trend" and "MR" mean per class
    strategy_learner = UnifiedStrategyLearner()
    print("\n[STEP 3] Strategy learner initialized")
    print("  Will discover what 'trend' and 'mean reversion' mean per class")

    # Create class-specific reward shapers
    reward_shapers = {
        'Forex': RewardShaper.from_asset_class('Forex'),
        'EquityIndex': RewardShaper.from_asset_class('EquityIndex'),
        'Crypto': RewardShaper.from_asset_class('Crypto'),
        'PreciousMetals': RewardShaper.from_asset_class('PreciousMetals'),
        'EnergyCommodities': RewardShaper.from_asset_class('EnergyCommodities'),
    }

    print("\n[STEP 4] Reward shapers by class:")
    for cls, shaper in reward_shapers.items():
        print(f"  {cls}: MAE_w={shaper.mae_penalty_weight}, "
              f"trend={shaper.trend_bonus_weight}, hold={shaper.holding_bonus_weight}")

    # Get state dim from first instrument
    first_inst = list(loader.instruments.values())[0]
    state_dim = len(extractor.get_feature_names())

    print(f"\n[STEP 4] Feature dimension: {state_dim}")
    print(f"  Feature names: {extractor.get_feature_names()[:10]}...")

    # Create agent
    agent = LinearQAgent(
        state_dim=state_dim,
        n_actions=3,
        learning_rate=0.05,
        gamma=0.9,
    )

    # Tracking
    cumulative = {
        'total_reward': 0.0,
        'total_pnl': 0.0,
        'total_trades': 0,
        'episodes': 0,
    }

    per_class = defaultdict(lambda: {
        'total_reward': 0.0,
        'total_pnl': 0.0,
        'trades': 0,
        'episodes': 0,
    })

    per_instrument = defaultdict(lambda: {
        'rewards': [],
        'pnls': [],
        'trades': [],
    })

    instrument_list = list(loader.instruments.keys())
    epsilon = 1.0

    print(f"\n[STEP 5] Running {episodes} episodes...")
    print("=" * 80)

    # Training loop
    for episode in range(episodes):
        # Round-robin instrument selection
        inst_key = instrument_list[episode % len(instrument_list)]
        inst_meas = loader.instruments[inst_key]

        # Get class-specific reward shaper
        asset_class = inst_meas.asset_class
        reward_shaper = reward_shapers.get(asset_class, reward_shapers['Forex'])

        # Create env with strategy learner
        env = ComprehensiveTradingEnv(
            inst_measurements=inst_meas,
            feature_extractor=extractor,
            reward_shaper=reward_shaper,
            strategy_learner=strategy_learner,
        )

        # Run episode
        state = env.reset()
        total_reward = 0.0

        for step in range(500):
            action = agent.select_action(state, epsilon)
            next_state, reward, done, info = env.step(action)
            agent.update(state, action, reward, next_state, done)
            total_reward += reward
            state = next_state
            if done:
                break

        epsilon = max(0.1, epsilon * 0.995)

        # Get stats
        stats = env.get_episode_stats()

        # Update tracking
        cumulative['total_reward'] += total_reward
        cumulative['total_pnl'] += stats['total_pnl']
        cumulative['total_trades'] += stats['trades']
        cumulative['episodes'] += 1

        per_class[asset_class]['total_reward'] += total_reward
        per_class[asset_class]['total_pnl'] += stats['total_pnl']
        per_class[asset_class]['trades'] += stats['trades']
        per_class[asset_class]['episodes'] += 1

        per_instrument[inst_key]['rewards'].append(total_reward)
        per_instrument[inst_key]['pnls'].append(stats['total_pnl'])
        per_instrument[inst_key]['trades'].append(stats['trades'])

        # Heartbeat every 10 episodes
        if (episode + 1) % 10 == 0:
            avg_r = cumulative['total_reward'] / cumulative['episodes']
            avg_pnl = cumulative['total_pnl'] / cumulative['episodes']
            print(f"  EP {episode+1:3d}/{episodes} | {inst_key:<20} | "
                  f"R={total_reward:+6.1f} | PnL={stats['total_pnl']:+5.2f}% | "
                  f"Trades={stats['trades']:2d} | ε={epsilon:.2f} | "
                  f"Cum: R={avg_r:+.1f}, PnL={avg_pnl:+.2f}%")

    # Final report
    print("\n" + "=" * 80)
    print("  EXPLORATION RESULTS")
    print("=" * 80)

    print(f"\n[CUMULATIVE]")
    print(f"  Episodes: {cumulative['episodes']}")
    print(f"  Total Reward: {cumulative['total_reward']:+.1f}")
    print(f"  Avg Reward: {cumulative['total_reward'] / cumulative['episodes']:+.2f}")
    print(f"  Total PnL: {cumulative['total_pnl']:+.2f}%")
    print(f"  Avg PnL: {cumulative['total_pnl'] / cumulative['episodes']:+.3f}%")

    print(f"\n[BY ASSET CLASS]")
    print(f"  {'Class':<20} {'Eps':>5} {'Avg R':>10} {'Avg PnL%':>10} {'Trades':>8}")
    print("  " + "-" * 55)
    for cls in sorted(per_class.keys()):
        stats = per_class[cls]
        if stats['episodes'] > 0:
            avg_r = stats['total_reward'] / stats['episodes']
            avg_pnl = stats['total_pnl'] / stats['episodes']
            print(f"  {cls:<20} {stats['episodes']:>5} {avg_r:>+10.2f} {avg_pnl:>+10.3f}% {stats['trades']:>8}")

    # Discovery report
    print("\n" + "=" * 80)
    print("  DISCOVERIES")
    print("=" * 80)

    discoveries = extractor.get_discoveries()

    # Signal reliability per class
    print("\n[SIGNAL RELIABILITY BY CLASS]")
    for cls, profile in discoveries.get('class_profiles', {}).items():
        print(f"\n  {cls}:")
        print(f"    Dominant: {profile.get('dominant_signal', 'unknown')}")
        for sig, rel in profile.get('generator_reliability', {}).items():
            bar = "█" * int(rel * 20)
            print(f"      {sig:<20}: {rel:.3f} {bar}")

    # Inverse relationships
    inversions = discoveries.get('inverse_relationships', [])
    if inversions:
        print("\n[INVERSE RELATIONSHIPS (High Vol vs Low Vol)]")
        for inv in inversions[:5]:
            print(f"  {inv['measurement_1']} vs {inv['measurement_2']}: "
                  f"{inv['low_vol_correlation']:+.2f} → {inv['high_vol_correlation']:+.2f} "
                  f"({inv['type']})")

    # STRATEGY DEFINITION DISCOVERY - What "trend" means per class
    print("\n" + "=" * 80)
    print("  TREND DEFINITION DISCOVERY")
    print("  What measurements ACTUALLY define 'trending' per class")
    print("=" * 80)

    for asset_class in sorted(set(per_class.keys())):
        profile = strategy_learner.get_class_strategy_profile(asset_class)

        print(f"\n  {asset_class}:")

        # Trend definition
        trend = profile.get('trend', {})
        trend_def = trend.get('definition', {})
        if trend_def.get('top_predictors'):
            print(f"    TREND defined by:")
            for pred in trend_def['top_predictors'][:5]:
                corr = pred['correlation']
                name = pred['measurement']
                bar = "█" * int(abs(corr) * 15)
                sign = "+" if corr > 0 else "-"
                print(f"      {sign}{abs(corr):.3f} {name:<35} {bar}")
            print(f"    Win rate: {trend.get('win_rate', 0):.1%}, Avg PnL: {trend.get('avg_pnl', 0):.3f}%")

        # MR definition
        mr = profile.get('mean_reversion', {})
        mr_def = mr.get('definition', {})
        if mr_def.get('top_predictors'):
            print(f"    MEAN-REVERSION defined by:")
            for pred in mr_def['top_predictors'][:3]:
                corr = pred['correlation']
                name = pred['measurement']
                sign = "+" if corr > 0 else "-"
                print(f"      {sign}{abs(corr):.3f} {name}")
            print(f"    Win rate: {mr.get('win_rate', 0):.1%}, Avg PnL: {mr.get('avg_pnl', 0):.3f}%")

        print(f"    → RECOMMENDED: {profile.get('recommended_strategy', 'unknown').upper()}")

    # Cross-class comparison
    print("\n" + "-" * 60)
    print("  CROSS-CLASS COMPARISON: Same measurement, different meaning")
    print("-" * 60)

    comparison = strategy_learner.trend_learner.compare_trend_definitions()
    if not comparison.empty:
        print(f"\n  {'Class':<20} {'#1 Predictor':<25} {'#2 Predictor':<25}")
        print("  " + "-" * 70)
        for _, row in comparison.iterrows():
            cls = row.get('asset_class', 'unknown')
            p1 = row.get('predictor_1', 'N/A')
            p2 = row.get('predictor_2', 'N/A')
            print(f"  {cls:<20} {str(p1):<25} {str(p2):<25}")

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    # Collect strategy definitions
    strategy_definitions = {}
    for asset_class in per_class.keys():
        profile = strategy_learner.get_class_strategy_profile(asset_class)
        strategy_definitions[asset_class] = {
            'trend_definition': profile.get('trend', {}).get('definition', {}),
            'mr_definition': profile.get('mean_reversion', {}).get('definition', {}),
            'trend_win_rate': profile.get('trend', {}).get('win_rate', 0),
            'mr_win_rate': profile.get('mean_reversion', {}).get('win_rate', 0),
            'recommended': profile.get('recommended_strategy', 'unknown'),
        }

    results = {
        "timestamp": timestamp,
        "episodes": episodes,
        "cumulative": cumulative,
        "per_class": {k: dict(v) for k, v in per_class.items()},
        "discoveries": {
            k: v if not isinstance(v, pd.DataFrame) else v.to_dict()
            for k, v in discoveries.items()
        },
        "strategy_definitions": strategy_definitions,
    }

    results_file = results_dir / f"comprehensive_exploration_{timestamp}.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n[SAVED] {results_file}")

    return results


def standardize_data(source_dir: str) -> str:
    """Standardize data to common end date."""
    source_path = Path(source_dir)
    output_dir = str(source_path) + "_standardized"
    output_path = Path(output_dir)

    # Search for CSV files in root and all subdirectories
    csv_files = list(source_path.glob("*.csv"))
    csv_files.extend(source_path.glob("**/*.csv"))
    csv_files = list(set(csv_files))  # Remove duplicates
    
    if not csv_files:
        print(f"  [WARN] No CSV files found in {source_dir}")
        return source_dir

    # Find earliest end date
    import re
    end_dates = []
    for f in csv_files:
        match = re.search(r'_(\d{12})\.csv$', f.name)
        if match:
            ts = match.group(1)
            end_dates.append(datetime.strptime(ts, "%Y%m%d%H%M"))

    if not end_dates:
        return source_dir

    cutoff = min(end_dates)
    print(f"  Cutoff: {cutoff}")

    output_path.mkdir(exist_ok=True)

    # Clear old files (including in subdirectories)
    for old in output_path.glob("**/*.csv"):
        old.unlink()

    truncated = 0
    copied = 0

    for csv_file in csv_files:
        try:
            match = re.search(r'_(\d{12})\.csv$', csv_file.name)
            if match:
                ts = datetime.strptime(match.group(1), "%Y%m%d%H%M")
                
                # Preserve subdirectory structure
                rel_path = csv_file.relative_to(source_path)
                dest_file = output_path / rel_path
                dest_file.parent.mkdir(parents=True, exist_ok=True)
                
                if ts <= cutoff:
                    shutil.copy(csv_file, dest_file)
                    copied += 1
                else:
                    # Truncate
                    df = pd.read_csv(csv_file, sep='\t')
                    df.columns = [c.lower().replace('<', '').replace('>', '') for c in df.columns]

                    if 'date' in df.columns and 'time' in df.columns:
                        date_str = df['date'].astype(str).str.replace('.', '-', regex=False)
                        df['datetime'] = pd.to_datetime(date_str + ' ' + df['time'].astype(str))
                        df = df[df['datetime'] <= cutoff]

                        if len(df) > 0:
                            new_end = df['datetime'].max().strftime("%Y%m%d%H%M")
                            parts = csv_file.stem.split('_')
                            new_name = f"{parts[0]}_{parts[1]}_{parts[2]}_{new_end}.csv"

                            df = df.drop(columns=['datetime'])
                            df.columns = ['<' + c.upper() + '>' for c in df.columns]
                            
                            # Save to appropriate subdirectory
                            new_dest = dest_file.parent / new_name
                            df.to_csv(new_dest, sep='\t', index=False)
                            truncated += 1
        except Exception as e:
            print(f"    [WARN] {csv_file.name}: {e}")

    print(f"  Standardized: {truncated} truncated, {copied} copied")
    return output_dir


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Comprehensive exploration with discovery engine"
    )
    parser.add_argument("--data-dir", default="data/master")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--verbose", action="store_true", default=True)
    args = parser.parse_args()

    if not MEASUREMENT_FRAMEWORK_AVAILABLE:
        print("[ERROR] Measurement framework not available!")
        print("Make sure kinetra/measurements.py and related files exist.")
        sys.exit(1)

    run_comprehensive_exploration(
        data_dir=args.data_dir,
        episodes=args.episodes,
        verbose=args.verbose,
    )
