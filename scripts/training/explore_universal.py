#!/usr/bin/env python3
"""
Universal Agent Baseline - Exploration Step 1
==============================================

First principles approach:
1. Train ONE universal agent on ALL instruments
2. Track performance by asset_class, regime, timeframe, volatility
3. Establish baseline before exploring specialization

THE MARKET TELLS US, WE DON'T ASSUME!

Usage:
    python scripts/explore_universal.py
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
import json
from collections import defaultdict

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from rl_exploration_framework import (
    MultiInstrumentLoader,
    MultiInstrumentEnv,
    RewardShaper,
    LinearQAgent,
    FeatureTracker,
)


def get_rl_state_features(physics_state: pd.DataFrame, bar_index: int) -> np.ndarray:
    """Extract ungated feature vector for RL exploration.

    First-principles approach: NO assumptions, NO gating, NO filtering.
    Expose ALL physics measures to let the agent discover optimal combinations.

    "We don't know what we don't know" - allow exploration of ALL feature space.

    Returns 64-dimensional state vector (all normalized to ~N(0,1)):
    - Kinematic: v, a, j, jerk_z (derivatives)
    - Energetic: energy, PE, eta (kinetic/potential/efficiency)
    - Damping: damping, zeta, viscosity, visc_z
    - Information: entropy, entropy_z, reynolds
    - Chaos: lyapunov_proxy, lyap_z, local_dim
    - Tail risk: cvar_95, cvar_asymmetry
    - Stacked: composite, triple_stack, stack_jerk_entropy, stack_jerk_lyap
    - Regime: one-hot encoded (OVERDAMPED, UNDERDAMPED, LAMINAR, BREAKOUT)
    - Momentum: roc, momentum_strength
    - Adaptive: adaptive_trail_mult
    - Percentiles: all _pct features (empirical CDFs)
    - Volatility: YZ, RS, GK, PK estimators + ratios
    - DSP: Ehlers roofing filter, trend, cycle period
    - VPIN: Order flow toxicity proxy, buy pressure
    - Higher moments: Kurtosis, skewness, tail_risk, JB proxy
    """
    if bar_index >= len(physics_state):
        return np.zeros(64)  # Return zeros for out-of-bounds

    ps = physics_state.iloc[bar_index]

    # Build ungated feature vector - let the agent discover what matters
    features = [
        # === KINEMATICS (derivatives) ===
        ps.get("v", 0),                    # velocity (1st derivative)
        ps.get("a", 0),                    # acceleration (2nd derivative)
        ps.get("j", 0),                    # jerk (3rd derivative)
        ps.get("jerk_z", 0),               # z-scored jerk

        # === ENERGETICS ===
        ps.get("energy", 0),               # kinetic energy (0.5 * v^2)
        ps.get("PE", 0),                   # potential energy (1/vol)
        ps.get("eta", 0),                  # efficiency (KE/PE)
        ps.get("energy_pct", 0.5),         # percentile

        # === DAMPING / FRICTION ===
        ps.get("damping", 0),              # damping ratio (zeta)
        ps.get("viscosity", 0),            # viscosity proxy
        ps.get("visc_z", 0),               # z-scored viscosity
        ps.get("damping_pct", 0.5),        # percentile

        # === INFORMATION / ENTROPY ===
        ps.get("entropy", 0),              # spectral entropy
        ps.get("entropy_z", 0),            # z-scored entropy
        ps.get("reynolds", 0),             # Reynolds number (trend/noise)
        ps.get("entropy_pct", 0.5),        # percentile

        # === CHAOS MEASURES ===
        ps.get("lyapunov_proxy", 0),       # Lyapunov divergence rate
        ps.get("lyap_z", 0),               # z-scored Lyapunov
        ps.get("local_dim", 2.0),          # local correlation dimension
        ps.get("lyapunov_proxy_pct", 0.5), # percentile
        ps.get("local_dim_pct", 0.5),      # percentile

        # === TAIL RISK (CVaR) ===
        ps.get("cvar_95", 0),              # 95% CVaR (expected shortfall)
        ps.get("cvar_asymmetry", 1.0),     # upside/downside tail ratio
        ps.get("cvar_95_pct", 0.5),        # percentile
        ps.get("cvar_asymmetry_pct", 0.5), # percentile

        # === STACKED COMPOSITES (non-linear combinations) ===
        ps.get("composite_jerk_entropy", 0),   # jerk * exp(entropy)
        ps.get("stack_jerk_entropy", 0),       # jerk_z * exp(entropy_z)
        ps.get("stack_jerk_lyap", 0),          # jerk_z * |lyap_z|
        ps.get("triple_stack", 0),             # jerk_z * exp(entropy_z) * lyap_z
        ps.get("composite_pct", 0.5),          # percentile
        ps.get("triple_stack_pct", 0.5),       # percentile

        # === MOMENTUM ===
        ps.get("roc", 0),                  # rate of change
        ps.get("momentum_strength", 0),    # rolling |ROC| mean

        # === REGIME (one-hot) ===
        1.0 if ps.get("regime") == "OVERDAMPED" else 0.0,
        1.0 if ps.get("regime") == "UNDERDAMPED" else 0.0,
        1.0 if ps.get("regime") == "LAMINAR" else 0.0,
        1.0 if ps.get("regime") == "BREAKOUT" else 0.0,

        # === REGIME AGE ===
        ps.get("regime_age_frac", 0),      # normalized time in current regime

        # === ADAPTIVE TRAIL ===
        ps.get("adaptive_trail_mult", 2.0), # physics-based trail multiplier

        # === ADDITIONAL PERCENTILES ===
        ps.get("PE_pct", 0.5),
        ps.get("reynolds_pct", 0.5),
        ps.get("eta_pct", 0.5),

        # === ADVANCED VOLATILITY (YZ/RS/GK/PK) ===
        ps.get("vol_rs", 0),               # Rogers-Satchell (drift-robust)
        ps.get("vol_yz", 0),               # Yang-Zhang (most efficient)
        ps.get("vol_gk", 0),               # Garman-Klass (classic)
        ps.get("vol_rs_z", 0),             # z-scored RS
        ps.get("vol_yz_z", 0),             # z-scored YZ
        ps.get("vol_ratio_yz_rs", 1.0),    # YZ/RS ratio (gap risk)
        ps.get("vol_term_structure", 1.0), # short/long vol ratio (stress)

        # === DSP (Ehlers Filters) ===
        ps.get("dsp_roofing", 0),          # Roofing filter (cycle isolation)
        ps.get("dsp_roofing_z", 0),        # z-scored roofing
        ps.get("dsp_trend", 0),            # Instantaneous trend
        ps.get("dsp_trend_dir", 0),        # Trend direction (-1, 0, 1)
        ps.get("dsp_cycle_period", 24),    # Estimated cycle period

        # === VPIN (Order Flow Toxicity) ===
        ps.get("vpin", 0.5),               # VPIN proxy (0-1)
        ps.get("vpin_z", 0),               # z-scored VPIN
        ps.get("vpin_pct", 0.5),           # VPIN percentile
        ps.get("buy_pressure", 0.5),       # Buy volume ratio

        # === HIGHER MOMENTS (Kurtosis/Skewness) ===
        ps.get("kurtosis", 0),             # Excess kurtosis (fat tails)
        ps.get("kurtosis_z", 0),           # z-scored kurtosis
        ps.get("skewness", 0),             # Skewness (tail asymmetry)
        ps.get("skewness_z", 0),           # z-scored skewness
        ps.get("tail_risk", 0),            # kurtosis_z * (-skewness_z) (crash risk)
        ps.get("jb_proxy_z", 0),           # Jarque-Bera proxy (non-normality)
    ]

    return np.array(features, dtype=np.float32)


def get_rl_feature_names() -> list:
    """Get feature names for interpretability and debugging."""
    return [
        # Kinematics
        "v", "a", "j", "jerk_z",
        # Energetics
        "energy", "PE", "eta", "energy_pct",
        # Damping
        "damping", "viscosity", "visc_z", "damping_pct",
        # Entropy/Information
        "entropy", "entropy_z", "reynolds", "entropy_pct",
        # Chaos
        "lyapunov_proxy", "lyap_z", "local_dim", "lyapunov_proxy_pct", "local_dim_pct",
        # Tail risk
        "cvar_95", "cvar_asymmetry", "cvar_95_pct", "cvar_asymmetry_pct",
        # Stacked composites
        "composite_jerk_entropy", "stack_jerk_entropy", "stack_jerk_lyap", "triple_stack",
        "composite_pct", "triple_stack_pct",
        # Momentum
        "roc", "momentum_strength",
        # Regime (one-hot)
        "regime_OVERDAMPED", "regime_UNDERDAMPED", "regime_LAMINAR", "regime_BREAKOUT",
        "regime_age_frac",
        # Adaptive
        "adaptive_trail_mult",
        "PE_pct", "reynolds_pct", "eta_pct",
        # Advanced volatility (YZ/RS/GK/PK)
        "vol_rs", "vol_yz", "vol_gk", "vol_rs_z", "vol_yz_z",
        "vol_ratio_yz_rs", "vol_term_structure",
        # DSP (Ehlers filters)
        "dsp_roofing", "dsp_roofing_z", "dsp_trend", "dsp_trend_dir", "dsp_cycle_period",
        # VPIN (order flow toxicity)
        "vpin", "vpin_z", "vpin_pct", "buy_pressure",
        # Higher moments
        "kurtosis", "kurtosis_z", "skewness", "skewness_z", "tail_risk", "jb_proxy_z",
    ]


def classify_symbol(symbol: str) -> str:
    """Classify symbol into asset class."""
    symbol_upper = symbol.upper().replace('+', '').replace('-', '')

    if 'BTC' in symbol_upper or 'ETH' in symbol_upper or 'XRP' in symbol_upper or 'LTC' in symbol_upper:
        return 'crypto'
    elif len(symbol_upper) == 6 and symbol_upper.isalpha():
        return 'forex'
    elif any(x in symbol_upper for x in ['XAU', 'XAG', 'XPT', 'XPD', 'SILVER', 'GOLD']):
        return 'metals'
    # Commodities - check before indices to avoid UKOUSD matching 'UK'
    elif any(x in symbol_upper for x in ['OIL', 'WTI', 'BRENT', 'GAS', 'COPPER']):
        return 'commodities'
    # EU50 = Euro Stoxx 50, UK indices, SA40 = South Africa 40
    elif any(x in symbol_upper for x in ['SPX', 'NAS', 'DOW', 'DJ', 'DAX', 'FTSE', 'NIKKEI', 'US', 'GER', 'UK', 'SA', 'EU']):
        return 'indices'
    else:
        return 'unknown'


def classify_volatility(stats: dict) -> str:
    """Classify episode volatility."""
    # Use a simple volatility proxy from episode stats
    # In real implementation, would use ATR or similar
    pnl_std = stats.get('pnl_std', 0)

    if pnl_std < 100:
        return 'low'
    elif pnl_std < 500:
        return 'medium'
    else:
        return 'high'


def print_header(text: str):
    """Print section header."""
    print("\n" + "=" * 80)
    print(f"  {text}")
    print("=" * 80)


def main():
    """Run universal agent baseline exploration."""
    print_header("UNIVERSAL AGENT BASELINE - EXPLORATION")

    print("\n📋 Approach:")
    print("  • ONE universal agent")
    print("  • ALL instruments (forex, crypto, indices, metals, commodities)")
    print("  • ALL timeframes (M15, M30, H1, H4)")
    print("  • Track performance by asset_class × regime × timeframe × volatility")
    print("\n  THE MARKET TELLS US, WE DON'T ASSUME!")

    # Load training data
    train_dir = Path("data/prepared/train")

    if not train_dir.exists():
        print(f"\n❌ Training data not found: {train_dir}")
        print(f"   Run: python scripts/prepare_data.py")
        return

    print(f"\n📥 Loading training data from {train_dir}...")

    loader = MultiInstrumentLoader(data_dir=str(train_dir), verbose=True)
    loader.load_all()

    if not loader.instruments:
        print("\n❌ No instruments loaded")
        return

    print(f"\n✅ Loaded {len(loader.instruments)} instruments")

    # Show breakdown
    by_class = defaultdict(list)
    for key in loader.instruments.keys():
        symbol = key.split('_')[0]
        asset_class = classify_symbol(symbol)
        by_class[asset_class].append(key)

    print(f"\n📊 Breakdown by asset class:")
    for asset_class, instruments in sorted(by_class.items()):
        print(f"  {asset_class:12s}: {len(instruments)} instruments")

    # Setup training
    print(f"\n⚙️  Setting up universal agent...")

    # Risk-averse reward shaper
    reward_shaper = RewardShaper(
        pnl_weight=1.0,
        edge_ratio_weight=0.3,
        mae_penalty_weight=2.5,  # Penalize adverse excursion
        regime_bonus_weight=0.2,
        entropy_alignment_weight=0.1,
    )

    # Multi-instrument environment
    env = MultiInstrumentEnv(
        loader=loader,
        feature_extractor=get_rl_state_features,
        reward_shaper=reward_shaper,
        sampling_mode="round_robin",  # Sample all instruments fairly
    )

    # Universal agent (ONE agent for ALL instruments)
    agent = LinearQAgent(
        state_dim=env.state_dim,
        n_actions=env.n_actions,
        learning_rate=0.05,
        gamma=0.9,
    )

    # Feature tracker
    feature_names = get_rl_feature_names()
    tracker = FeatureTracker(feature_names)

    print(f"\n✅ Universal agent ready:")
    print(f"  State dim: {env.state_dim}")
    print(f"  Actions: {env.n_actions} (HOLD, LONG, SHORT, CLOSE)")
    print(f"  Instruments: {len(loader.instruments)}")

    # Interactive training configuration
    print(f"\n🎯 Training Configuration:")
    print(f"\n1. How many episodes to train?")
    print(f"   • Quick test: 20 episodes (~3 min)")
    print(f"   • Standard: 50 episodes (~7 min)")
    print(f"   • Thorough: 100 episodes (~15 min)")
    print(f"   • Custom: Enter your own")

    episodes_choice = input("\nSelect [1=Quick, 2=Standard, 3=Thorough, or enter number]: ").strip()

    if episodes_choice == '1':
        episodes = 20
    elif episodes_choice == '2' or episodes_choice == '':
        episodes = 50  # Default
    elif episodes_choice == '3':
        episodes = 100
    else:
        try:
            episodes = int(episodes_choice)
            if episodes < 1:
                print("⚠️  Invalid, using default 50")
                episodes = 50
        except ValueError:
            print("⚠️  Invalid input, using default 50")
            episodes = 50

    print(f"\n✅ Will train for {episodes} episodes")

    # Exploration strategy
    print(f"\n2. Exploration strategy:")
    print(f"   1. Aggressive (ε: 1.0 → 0.05, decay 0.93)")
    print(f"   2. Balanced (ε: 1.0 → 0.10, decay 0.95) [default]")
    print(f"   3. Conservative (ε: 1.0 → 0.20, decay 0.97)")

    strategy_choice = input("\nSelect strategy [1-3, default=2]: ").strip()

    if strategy_choice == '1':
        epsilon = 1.0
        epsilon_decay = 0.93
        epsilon_min = 0.05
        print("✅ Aggressive exploration")
    elif strategy_choice == '3':
        epsilon = 1.0
        epsilon_decay = 0.97
        epsilon_min = 0.20
        print("✅ Conservative exploration")
    else:
        epsilon = 1.0
        epsilon_decay = 0.95
        epsilon_min = 0.1
        print("✅ Balanced exploration")

    # Performance tracking by breakdown
    performance_breakdown = defaultdict(lambda: {
        'episodes': 0,
        'total_reward': 0,
        'total_pnl': 0,
        'rewards': [],
        'pnls': []
    })

    print(f"\n🏃 Training universal agent for {episodes} episodes...")
    print(f"{'Ep':>4s} {'ε':>6s} {'R_avg':>10s} {'PnL_avg':>12s} {'Instrument':>20s} {'Class':>12s} {'Regime':>12s}")
    print("-" * 90)

    for episode in range(episodes):
        state = env.reset()
        total_reward = 0

        for step in range(500):
            action = agent.select_action(state, epsilon)
            tracker.record(state, action)
            next_state, reward, done, info = env.step(action)
            agent.update(state, action, reward, next_state, done)
            total_reward += reward
            state = next_state
            if done:
                break

        epsilon = max(epsilon_min, epsilon * epsilon_decay)

        # Get episode stats
        stats = env.get_episode_stats()
        pnl = stats.get("total_pnl", 0)
        instrument = stats.get("instrument", "unknown")
        regime = stats.get("regime", "unknown")

        # Extract breakdown dimensions
        symbol = instrument.split('_')[0]
        timeframe = instrument.split('_')[1] if '_' in instrument else 'unknown'
        asset_class = classify_symbol(symbol)
        volatility = classify_volatility(stats)

        # Record to breakdown
        breakdown_keys = [
            f"asset_class:{asset_class}",
            f"regime:{regime}",
            f"timeframe:{timeframe}",
            f"volatility:{volatility}",
            f"{asset_class}:{regime}",
            f"{asset_class}:{timeframe}",
            f"{regime}:{timeframe}",
            f"all:all"  # Overall
        ]

        for key in breakdown_keys:
            perf = performance_breakdown[key]
            perf['episodes'] += 1
            perf['total_reward'] += total_reward
            perf['total_pnl'] += pnl
            perf['rewards'].append(total_reward)
            perf['pnls'].append(pnl)

        # Print progress
        if (episode + 1) % 5 == 0 or episode < 5:
            overall = performance_breakdown['all:all']
            avg_r = overall['total_reward'] / overall['episodes']
            avg_pnl = overall['total_pnl'] / overall['episodes']
            print(f"{episode+1:4d} {epsilon:6.3f} {avg_r:+10.2f} ${avg_pnl:+12,.0f} {instrument:>20s} {asset_class:>12s} {regime:>12s}")

    # Analysis
    print_header("UNIVERSAL AGENT RESULTS")

    overall = performance_breakdown['all:all']
    print(f"\n📊 Overall Performance:")
    print(f"  Episodes:     {overall['episodes']}")
    print(f"  Total Reward: {overall['total_reward']:+.1f}")
    print(f"  Avg Reward:   {overall['total_reward'] / overall['episodes']:+.2f}")
    print(f"  Total P&L:    ${overall['total_pnl']:+,.0f}")
    print(f"  Avg P&L:      ${overall['total_pnl'] / overall['episodes']:+,.0f}")

    # Performance by asset class
    print(f"\n📈 By Asset Class:")
    asset_classes = [k for k in performance_breakdown.keys() if k.startswith('asset_class:')]
    for key in sorted(asset_classes):
        asset_class = key.split(':')[1]
        perf = performance_breakdown[key]
        if perf['episodes'] > 0:
            avg_r = perf['total_reward'] / perf['episodes']
            avg_pnl = perf['total_pnl'] / perf['episodes']
            print(f"  {asset_class:12s}: {perf['episodes']:3d} eps | R={avg_r:+8.2f} | PnL=${avg_pnl:+10,.0f}")

    # Performance by regime
    print(f"\n🌊 By Regime:")
    regimes = [k for k in performance_breakdown.keys() if k.startswith('regime:')]
    for key in sorted(regimes):
        regime = key.split(':')[1]
        perf = performance_breakdown[key]
        if perf['episodes'] > 0:
            avg_r = perf['total_reward'] / perf['episodes']
            avg_pnl = perf['total_pnl'] / perf['episodes']
            print(f"  {regime:12s}: {perf['episodes']:3d} eps | R={avg_r:+8.2f} | PnL=${avg_pnl:+10,.0f}")

    # Performance by timeframe
    print(f"\n⏱️  By Timeframe:")
    timeframes = [k for k in performance_breakdown.keys() if k.startswith('timeframe:')]
    for key in sorted(timeframes):
        timeframe = key.split(':')[1]
        perf = performance_breakdown[key]
        if perf['episodes'] > 0:
            avg_r = perf['total_reward'] / perf['episodes']
            avg_pnl = perf['total_pnl'] / perf['episodes']
            print(f"  {timeframe:12s}: {perf['episodes']:3d} eps | R={avg_r:+8.2f} | PnL=${avg_pnl:+10,.0f}")

    # Top feature correlations
    print(f"\n🔍 Top Features by Action:")
    top_features = tracker.get_top_features_per_action(top_k=3)
    action_names = ['HOLD', 'LONG', 'SHORT', 'CLOSE']
    for action_idx, features in top_features.items():
        action_name = action_names[action_idx] if action_idx < len(action_names) else f"Action{action_idx}"
        top_str = ", ".join([f"{name}({corr:+.3f})" for name, corr in features])
        print(f"  {action_name:6s}: {top_str}")

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path("results/exploration")
    results_dir.mkdir(parents=True, exist_ok=True)
    results_file = results_dir / f"universal_baseline_{timestamp}.json"

    # Convert breakdown to serializable format
    breakdown_serializable = {}
    for key, perf in performance_breakdown.items():
        breakdown_serializable[key] = {
            'episodes': perf['episodes'],
            'avg_reward': float(perf['total_reward'] / perf['episodes']) if perf['episodes'] > 0 else 0,
            'avg_pnl': float(perf['total_pnl'] / perf['episodes']) if perf['episodes'] > 0 else 0,
        }

    results = {
        'timestamp': timestamp,
        'approach': 'universal_agent',
        'episodes': episodes,
        'instruments': list(loader.instruments.keys()),
        'overall': {
            'episodes': overall['episodes'],
            'total_reward': float(overall['total_reward']),
            'avg_reward': float(overall['total_reward'] / overall['episodes']),
            'total_pnl': float(overall['total_pnl']),
            'avg_pnl': float(overall['total_pnl'] / overall['episodes']),
        },
        'performance_breakdown': breakdown_serializable,
        'top_features': top_features,
    }

    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n💾 Results saved: {results_file}")

    print_header("BASELINE ESTABLISHED")

    print(f"\n✅ Key Findings:")
    print(f"  • Universal agent trained on ALL data")
    print(f"  • Performance measured across all dimensions")
    print(f"  • Baseline established for comparison")

    print(f"\n🔬 Next Steps:")
    print(f"  1. Analyze which asset classes perform differently")
    print(f"  2. Explore measurement impact per class")
    print(f"  3. Test if specialization improves performance")
    print(f"\n  Run: python scripts/explore_measurements.py")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
