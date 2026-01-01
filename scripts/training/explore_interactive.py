#!/usr/bin/env python3
"""
Interactive Exploration Training
=================================

One unified process:
1. Check available data
2. Show what we have
3. Confirm to proceed
4. Run exploration training
5. Show results

Usage:
    python scripts/explore_interactive.py
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
import json

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
    """Extract ungated 64-dim feature vector for RL exploration."""
    if bar_index >= len(physics_state):
        return np.zeros(64)

    ps = physics_state.iloc[bar_index]

    features = [
        # Kinematics
        ps.get("v", 0), ps.get("a", 0), ps.get("j", 0), ps.get("jerk_z", 0),
        # Energetics
        ps.get("energy", 0), ps.get("PE", 0), ps.get("eta", 0), ps.get("energy_pct", 0.5),
        # Damping
        ps.get("damping", 0), ps.get("viscosity", 0), ps.get("visc_z", 0), ps.get("damping_pct", 0.5),
        # Entropy
        ps.get("entropy", 0), ps.get("entropy_z", 0), ps.get("reynolds", 0), ps.get("entropy_pct", 0.5),
        # Chaos
        ps.get("lyapunov_proxy", 0), ps.get("lyap_z", 0), ps.get("local_dim", 2.0),
        ps.get("lyapunov_proxy_pct", 0.5), ps.get("local_dim_pct", 0.5),
        # Tail risk
        ps.get("cvar_95", 0), ps.get("cvar_asymmetry", 1.0),
        ps.get("cvar_95_pct", 0.5), ps.get("cvar_asymmetry_pct", 0.5),
        # Composites
        ps.get("composite_jerk_entropy", 0), ps.get("stack_jerk_entropy", 0),
        ps.get("stack_jerk_lyap", 0), ps.get("triple_stack", 0),
        ps.get("composite_pct", 0.5), ps.get("triple_stack_pct", 0.5),
        # Momentum
        ps.get("roc", 0), ps.get("momentum_strength", 0),
        # Regime one-hot
        1.0 if ps.get("regime") == "OVERDAMPED" else 0.0,
        1.0 if ps.get("regime") == "UNDERDAMPED" else 0.0,
        1.0 if ps.get("regime") == "LAMINAR" else 0.0,
        1.0 if ps.get("regime") == "BREAKOUT" else 0.0,
        ps.get("regime_age_frac", 0),
        # Adaptive
        ps.get("adaptive_trail_mult", 2.0),
        ps.get("PE_pct", 0.5), ps.get("reynolds_pct", 0.5), ps.get("eta_pct", 0.5),
        # Advanced volatility
        ps.get("vol_rs", 0), ps.get("vol_yz", 0), ps.get("vol_gk", 0),
        ps.get("vol_rs_z", 0), ps.get("vol_yz_z", 0),
        ps.get("vol_ratio_yz_rs", 1.0), ps.get("vol_term_structure", 1.0),
        # DSP
        ps.get("dsp_roofing", 0), ps.get("dsp_roofing_z", 0),
        ps.get("dsp_trend", 0), ps.get("dsp_trend_dir", 0), ps.get("dsp_cycle_period", 24),
        # VPIN
        ps.get("vpin", 0.5), ps.get("vpin_z", 0),
        ps.get("vpin_pct", 0.5), ps.get("buy_pressure", 0.5),
        # Higher moments
        ps.get("kurtosis", 0), ps.get("kurtosis_z", 0),
        ps.get("skewness", 0), ps.get("skewness_z", 0),
        ps.get("tail_risk", 0), ps.get("jb_proxy_z", 0),
    ]

    return np.array(features, dtype=np.float32)


def get_rl_feature_names() -> list:
    """Get feature names for interpretability."""
    return [
        "v", "a", "j", "jerk_z",
        "energy", "PE", "eta", "energy_pct",
        "damping", "viscosity", "visc_z", "damping_pct",
        "entropy", "entropy_z", "reynolds", "entropy_pct",
        "lyapunov_proxy", "lyap_z", "local_dim", "lyapunov_proxy_pct", "local_dim_pct",
        "cvar_95", "cvar_asymmetry", "cvar_95_pct", "cvar_asymmetry_pct",
        "composite_jerk_entropy", "stack_jerk_entropy", "stack_jerk_lyap", "triple_stack",
        "composite_pct", "triple_stack_pct",
        "roc", "momentum_strength",
        "regime_OVERDAMPED", "regime_UNDERDAMPED", "regime_LAMINAR", "regime_BREAKOUT",
        "regime_age_frac",
        "adaptive_trail_mult",
        "PE_pct", "reynolds_pct", "eta_pct",
        "vol_rs", "vol_yz", "vol_gk", "vol_rs_z", "vol_yz_z",
        "vol_ratio_yz_rs", "vol_term_structure",
        "dsp_roofing", "dsp_roofing_z", "dsp_trend", "dsp_trend_dir", "dsp_cycle_period",
        "vpin", "vpin_z", "vpin_pct", "buy_pressure",
        "kurtosis", "kurtosis_z", "skewness", "skewness_z", "tail_risk", "jb_proxy_z",
    ]


def print_header(text):
    """Print section header."""
    print("\n" + "="*80)
    print(f"  {text}")
    print("="*80)


def print_step(step_num, text):
    """Print step header."""
    print(f"\n[STEP {step_num}] {text}")
    print("-" * 80)


def check_data_quality(csv_files, symbols):
    """Check data quality and detect issues."""
    issues = []

    # Expected timeframes
    expected_tfs = {'M15', 'M30', 'H1', 'H4'}

    # Check for missing timeframes
    incomplete_symbols = []
    for symbol, tfs in symbols.items():
        tfs_set = set(tfs)
        missing_tfs = expected_tfs - tfs_set
        if missing_tfs:
            incomplete_symbols.append((symbol, missing_tfs))

    if incomplete_symbols:
        issues.append(('missing_timeframes', incomplete_symbols))

    # Check for gaps in data (sample a few files)
    files_with_gaps = []
    for csv_file in csv_files[:20]:  # Check first 20 files
        try:
            df = pd.read_csv(csv_file)
            if 'time' in df.columns:
                df['time'] = pd.to_datetime(df['time'])
                df = df.sort_values('time')

                # Check for large gaps (more than 2x expected interval)
                time_diffs = df['time'].diff()
                median_diff = time_diffs.median()
                large_gaps = (time_diffs > median_diff * 2).sum()

                if large_gaps > 0:
                    files_with_gaps.append((csv_file.name, large_gaps))
        except:
            pass

    if files_with_gaps:
        issues.append(('data_gaps', files_with_gaps))

    return issues


def main():
    """Run interactive exploration training."""

    print_header("KINETRA INTERACTIVE EXPLORATION")

    data_dir = Path("data/master")

    # ========================================================================
    # STEP 1: CHECK AVAILABLE DATA
    # ========================================================================
    print_step(1, "Checking available data...")

    if not data_dir.exists():
        print(f"\n❌ Data directory not found: {data_dir}")
        print("   Run download script first to get market data")
        return

    csv_files = list(data_dir.glob("*.csv"))

    if not csv_files:
        print(f"\n❌ No CSV files found in {data_dir}")
        print("   Run download script to get market data")
        return

    print(f"\n✅ Found {len(csv_files)} data files")

    # Group by symbol and timeframe
    symbols = {}
    for f in csv_files:
        parts = f.stem.split('_')
        if len(parts) >= 2:
            symbol = parts[0]
            timeframe = parts[1]
            if symbol not in symbols:
                symbols[symbol] = []
            symbols[symbol].append(timeframe)

    print(f"\nAvailable symbols ({len(symbols)}):")
    for symbol, timeframes in sorted(symbols.items())[:20]:  # Show first 20
        tf_str = ', '.join(sorted(set(timeframes)))
        print(f"  {symbol:15s} : {tf_str}")

    if len(symbols) > 20:
        print(f"  ... and {len(symbols) - 20} more symbols")

    # ========================================================================
    # STEP 1.5: CHECK DATA QUALITY
    # ========================================================================
    print(f"\n🔍 Checking data quality...")

    issues = check_data_quality(csv_files, symbols)

    if issues:
        print(f"\n⚠️  Data quality issues detected:\n")

        for issue_type, details in issues:
            if issue_type == 'missing_timeframes':
                print(f"  📊 Missing timeframes ({len(details)} symbols):")
                for symbol, missing in details[:10]:
                    print(f"     {symbol}: missing {', '.join(sorted(missing))}")
                if len(details) > 10:
                    print(f"     ... and {len(details) - 10} more symbols")

            elif issue_type == 'data_gaps':
                print(f"\n  ⏱️  Data gaps detected ({len(details)} files):")
                for filename, gap_count in details[:5]:
                    print(f"     {filename}: {gap_count} gaps")
                if len(details) > 5:
                    print(f"     ... and {len(details) - 5} more files")

        print(f"\n💡 Recommendations:")
        print(f"  1. Fetch missing data: python scripts/download_metaapi.py")
        print(f"  2. Run data preparation to handle:")
        print(f"     • Public holidays")
        print(f"     • Trading hours (forex 24/5, indices market hours)")
        print(f"     • Missing data interpolation")
        print(f"     • Timezone alignment")

        response = input(f"\nContinue with existing data anyway? [y/N]: ").strip().lower()
        if response not in ['y', 'yes']:
            print(f"\n⚠️  Exploration cancelled")
            print(f"\nNext steps:")
            print(f"  1. Fix data issues (download missing data, run data prep)")
            print(f"  2. Run this script again")
            return

        print(f"\n⚠️  Proceeding with existing data (may have gaps/issues)")
    else:
        print(f"✅ No major data quality issues detected")

    # ========================================================================
    # STEP 2: CONFIRM TO PROCEED
    # ========================================================================
    print_step(2, "Ready to train exploration agent")

    print("\nTraining configuration:")
    print("  • Multi-instrument exploration")
    print("  • Physics-based features (energy, entropy, damping)")
    print("  • Linear Q-learning agent")
    print("  • Risk-averse reward shaping (MAE penalty)")
    print("  • 35 episodes (default)")

    response = input("\nProceed with training? [Y/n]: ").strip().lower()
    if response and response not in ['y', 'yes']:
        print("\n⚠️  Training cancelled by user")
        return

    # ========================================================================
    # STEP 3: LOAD DATA
    # ========================================================================
    print_step(3, "Loading instruments...")

    loader = MultiInstrumentLoader(data_dir=str(data_dir), verbose=True)
    loader.load_all()

    if not loader.instruments:
        print("\n❌ Failed to load any instruments")
        return

    print(f"\n✅ Loaded {len(loader.instruments)} instruments")
    for key, data in list(loader.instruments.items())[:10]:
        print(f"  {key}: {len(data)} bars")

    if len(loader.instruments) > 10:
        print(f"  ... and {len(loader.instruments) - 10} more instruments")

    # ========================================================================
    # STEP 4: SETUP TRAINING
    # ========================================================================
    print_step(4, "Setting up exploration training...")

    # Reward shaper with risk-averse config
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
        sampling_mode="round_robin",
    )

    # Linear Q agent
    agent = LinearQAgent(
        state_dim=env.state_dim,
        n_actions=env.n_actions,
        learning_rate=0.05,
        gamma=0.9,
    )

    # Feature tracker
    feature_names = get_rl_feature_names()
    tracker = FeatureTracker(feature_names)

    print(f"\n✅ Training setup complete")
    print(f"  State dimension: {env.state_dim}")
    print(f"  Actions: {env.n_actions} (HOLD, LONG, SHORT, CLOSE)")
    print(f"  Instruments: {len(loader.instruments)}")

    # ========================================================================
    # STEP 5: RUN TRAINING
    # ========================================================================
    print_step(5, "Running exploration training...")

    episodes = 35
    all_rewards = []
    all_pnl = []
    per_instrument = {key: [] for key in loader.instruments}
    epsilon = 1.0

    print(f"\nTraining for {episodes} episodes...")
    print(f"{'Episode':>8s} {'ε':>8s} {'Reward':>12s} {'PnL':>15s} {'Instrument':>15s}")
    print("-" * 80)

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

        epsilon = max(0.1, epsilon * 0.95)

        stats = env.get_episode_stats()
        all_rewards.append(total_reward)
        all_pnl.append(stats.get("total_pnl", 0))

        instrument = stats.get("instrument", "unknown")
        per_instrument[instrument].append({
            "episode": episode,
            "reward": total_reward,
            "pnl": stats.get("total_pnl", 0),
        })

        # Print every episode
        if (episode + 1) % 5 == 0 or episode < 5:
            recent_r = np.mean(all_rewards[-5:]) if len(all_rewards) >= 5 else np.mean(all_rewards)
            recent_pnl = np.mean(all_pnl[-5:]) if len(all_pnl) >= 5 else np.mean(all_pnl)
            print(f"{episode+1:8d} {epsilon:8.3f} {recent_r:+12.2f} ${recent_pnl:+14,.0f} {instrument:>15s}")

    # ========================================================================
    # STEP 6: SHOW RESULTS
    # ========================================================================
    print_step(6, "Training complete! Analyzing results...")

    top_features = tracker.get_top_features_per_action(top_k=5)

    # Per-instrument summary
    inst_summary = {}
    for key, eps in per_instrument.items():
        if eps:
            inst_summary[key] = {
                "episodes": len(eps),
                "avg_reward": float(np.mean([e["reward"] for e in eps])),
                "avg_pnl": float(np.mean([e["pnl"] for e in eps])),
            }

    # Print results
    print_header("EXPLORATION RESULTS")

    print(f"\n📊 Overall Performance:")
    print(f"  Total Episodes:  {episodes}")
    print(f"  Total Reward:    {sum(all_rewards):+.1f}")
    print(f"  Avg Reward:      {np.mean(all_rewards):+.2f}")
    print(f"  Total P&L:       ${sum(all_pnl):+,.0f}")
    print(f"  Avg P&L:         ${np.mean(all_pnl):+,.0f}")

    print(f"\n📈 Per Instrument (top 10 by episodes):")
    sorted_inst = sorted(inst_summary.items(), key=lambda x: x[1]['episodes'], reverse=True)
    for key, stats in sorted_inst[:10]:
        print(f"  {key:20s}: {stats['episodes']:3d} eps | R={stats['avg_reward']:+8.2f} | PnL=${stats['avg_pnl']:+10,.0f}")

    print(f"\n🔍 Top Features Discovered:")
    action_names = ['HOLD', 'LONG', 'SHORT', 'CLOSE']
    for action_idx, features in top_features.items():
        action_name = action_names[action_idx] if action_idx < len(action_names) else f"Action{action_idx}"
        top_3 = features[:3]
        top_str = ", ".join([f"{name}({corr:+.3f})" for name, corr in top_3])
        print(f"  {action_name:6s}: {top_str}")

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    results_file = results_dir / f"exploration_{timestamp}.json"

    results = {
        "timestamp": timestamp,
        "episodes": episodes,
        "instruments": list(loader.instruments.keys()),
        "total_reward": float(sum(all_rewards)),
        "avg_reward": float(np.mean(all_rewards)),
        "total_pnl": float(sum(all_pnl)),
        "avg_pnl": float(np.mean(all_pnl)),
        "per_instrument": inst_summary,
        "top_features": top_features,
    }

    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n💾 Results saved: {results_file}")

    print_header("EXPLORATION COMPLETE!")

    print("\n✅ Next steps:")
    print("  • Review results in results/ directory")
    print("  • Analyze feature importance patterns")
    print("  • Try different reward configurations")
    print("  • Train for more episodes for convergence")
    print("  • Run with specific instruments or timeframes")


if __name__ == "__main__":
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
