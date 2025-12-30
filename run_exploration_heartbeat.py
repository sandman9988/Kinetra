#!/usr/bin/env python3
"""
Exploration Runner with Live Heartbeat Stats
Shows real-time progress per instrument, per run, cumulative, and per portfolio.

Automatically standardizes data (temporal alignment) before each run.
"""
import json
import shutil
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from rl_exploration_framework import (
    MultiInstrumentLoader,
    MultiInstrumentEnv,
    RewardShaper,
    LinearQAgent,
    FeatureTracker,
)
from test_physics_pipeline import get_rl_state_features, get_rl_feature_names


# Asset class mapping
ASSET_CLASS = {
    "AUDJPY+": "Forex", "AUDUSD+": "Forex", "EURJPY+": "Forex",
    "GBPJPY+": "Forex", "GBPUSD+": "Forex",
    "BTCJPY": "Crypto", "BTCUSD": "Crypto", "ETHEUR": "Crypto", "XRPJPY": "Crypto",
    "DJ30ft": "Index", "NAS100": "Index", "Nikkei225": "Index",
    "EU50": "Index", "GER40": "Index", "SA40": "Index", "US2000": "Index",
    "COPPER-C": "Commodity", "UKOUSD": "Commodity",
    "XAGUSD": "Metals", "XAUAUD+": "Metals", "XAUUSD+": "Metals", "XPTUSD": "Metals",
}


def standardize_data(source_dir: str, output_dir: str = None) -> str:
    """
    Standardize all data files to the earliest common end date.
    Returns path to standardized data directory.
    """
    source_path = Path(source_dir)
    if not source_path.exists():
        raise FileNotFoundError(f"Data directory not found: {source_dir}")

    # Default output is source_standardized
    if output_dir is None:
        output_dir = str(source_path) + "_standardized"
    output_path = Path(output_dir)

    csv_files = list(source_path.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files in {source_dir}")

    print(f"\n[STANDARDIZE] Processing {len(csv_files)} files...")

    # Parse filename to get end timestamp
    def parse_end_timestamp(filename):
        """Extract end timestamp from filename like AUDJPY+_H1_202401020000_202512262000.csv"""
        import re
        match = re.search(r'_(\d{12})\.csv$', filename)
        if match:
            ts = match.group(1)
            return datetime.strptime(ts, "%Y%m%d%H%M")
        return None

    # Find earliest end date across all files
    end_dates = []
    for f in csv_files:
        end_dt = parse_end_timestamp(f.name)
        if end_dt:
            end_dates.append(end_dt)

    if not end_dates:
        print("[WARN] Could not parse timestamps, using raw data")
        return source_dir

    cutoff = min(end_dates)
    print(f"[STANDARDIZE] Cutoff: {cutoff} (earliest end date)")

    # Create output directory
    output_path.mkdir(exist_ok=True)

    # Clear existing files
    for old_file in output_path.glob("*.csv"):
        old_file.unlink()

    truncated = 0
    copied = 0
    errors = 0

    for csv_file in csv_files:
        try:
            end_dt = parse_end_timestamp(csv_file.name)

            if end_dt and end_dt <= cutoff:
                # File ends at or before cutoff - copy as-is
                shutil.copy(csv_file, output_path / csv_file.name)
                copied += 1
            else:
                # Needs truncation - read and filter
                df = pd.read_csv(csv_file, sep='\t')
                df.columns = [c.lower().replace('<', '').replace('>', '') for c in df.columns]

                if 'date' in df.columns and 'time' in df.columns:
                    # Handle MT5 date format: 2024.01.02 (dots)
                    date_str = df['date'].astype(str).str.replace('.', '-', regex=False)
                    df['datetime'] = pd.to_datetime(date_str + ' ' + df['time'].astype(str))
                else:
                    continue  # Skip files without proper datetime

                # Filter by cutoff
                df_truncated = df[df['datetime'] <= cutoff]

                if len(df_truncated) == 0:
                    continue

                # Generate new filename with updated end timestamp
                new_end = df_truncated['datetime'].max()
                new_end_str = new_end.strftime("%Y%m%d%H%M")

                # Parse instrument and timeframe from original filename
                parts = csv_file.stem.split('_')
                if len(parts) >= 3:
                    instrument = parts[0]
                    timeframe = parts[1]
                    start_str = parts[2]
                    new_filename = f"{instrument}_{timeframe}_{start_str}_{new_end_str}.csv"
                else:
                    new_filename = csv_file.name

                # Save in original format
                df_truncated = df_truncated.drop(columns=['datetime'], errors='ignore')
                df_truncated.columns = ['<' + c.upper() + '>' for c in df_truncated.columns]
                df_truncated.to_csv(output_path / new_filename, sep='\t', index=False)
                truncated += 1

        except Exception as e:
            print(f"  [WARN] {csv_file.name}: {e}")
            errors += 1

    print(f"[STANDARDIZE] Done: {truncated} truncated, {copied} copied, {errors} errors")
    print(f"[STANDARDIZE] Output: {output_path}")

    return str(output_path)


def get_asset_class(instrument_key):
    """Get asset class from instrument key."""
    for prefix, cls in ASSET_CLASS.items():
        if instrument_key.startswith(prefix):
            return cls
    return "Other"


def print_heartbeat(
    episode: int,
    total_episodes: int,
    instrument: str,
    reward: float,
    pnl: float,
    trades: int,
    cumulative: dict,
    portfolio: dict,
):
    """Print formatted heartbeat stats. PnL is now in PERCENTAGE terms."""
    asset_class = get_asset_class(instrument)

    # Per-episode stats (PnL is now percentage, not absolute dollars)
    print(f"\n{'─'*80}")
    print(f"│ EP {episode:3d}/{total_episodes} │ {instrument:<20} │ {asset_class:<10} │")
    print(f"├{'─'*80}")
    print(f"│ Reward: {reward:+10.2f} │ PnL: {pnl:+8.2f}% │ Trades: {trades:3d} │")

    # Cumulative stats
    print(f"├{'─'*80}")
    print(f"│ CUMULATIVE: Total R={cumulative['total_reward']:+.1f} │ "
          f"Total PnL={cumulative['total_pnl']:+.2f}% │ "
          f"Avg R={cumulative['avg_reward']:+.2f} │")

    # Portfolio breakdown
    print(f"├{'─'*80}")
    print(f"│ PORTFOLIO BY ASSET CLASS:")
    for cls, stats in sorted(portfolio.items()):
        if stats['episodes'] > 0:
            avg_r = stats['total_reward'] / stats['episodes']
            avg_pnl = stats['total_pnl'] / stats['episodes']
            print(f"│   {cls:<12}: Eps={stats['episodes']:3d} │ "
                  f"Avg R={avg_r:+8.2f} │ Avg PnL={avg_pnl:+8.2f}% │")
    print(f"└{'─'*80}")


def run_with_heartbeat(
    data_dir: str = "data/master",
    config_name: str = "Heartbeat_Run",
    mae_w: float = 2.5,
    lr: float = 0.05,
    gamma: float = 0.9,
    episodes: int = 50,
    skip_standardize: bool = False,
):
    """Run exploration with live heartbeat output.

    Automatically standardizes data (temporal alignment) before each run
    unless skip_standardize=True.
    """
    print("=" * 80)
    print(f"  EXPLORATION WITH HEARTBEAT: {config_name}")
    print(f"  MAE_w={mae_w}, LR={lr}, Gamma={gamma}, Episodes={episodes}")
    print("=" * 80)

    # Step 1: Standardize data (all files to same end date)
    if not skip_standardize:
        try:
            data_dir = standardize_data(data_dir)
        except FileNotFoundError as e:
            print(f"[ERROR] {e}")
            return None

    # Load instruments
    loader = MultiInstrumentLoader(data_dir=data_dir, verbose=True)
    loader.load_all()

    if not loader.instruments:
        print("[ERROR] No instruments found!")
        return None

    n_instruments = len(loader.instruments)
    print(f"\n[READY] {n_instruments} instruments loaded")
    print("=" * 80)

    # Create reward shaper
    reward_shaper = RewardShaper(
        pnl_weight=1.0,
        edge_ratio_weight=0.3,
        mae_penalty_weight=mae_w,
        regime_bonus_weight=0.2,
        entropy_alignment_weight=0.1,
    )

    # Create env
    env = MultiInstrumentEnv(
        loader=loader,
        feature_extractor=get_rl_state_features,
        reward_shaper=reward_shaper,
        sampling_mode="round_robin",
    )

    # Create agent
    agent = LinearQAgent(
        state_dim=env.state_dim,
        n_actions=env.n_actions,
        learning_rate=lr,
        gamma=gamma,
    )

    feature_names = get_rl_feature_names()
    tracker = FeatureTracker(feature_names)

    # Tracking structures
    cumulative = {
        'total_reward': 0.0,
        'total_pnl': 0.0,
        'total_trades': 0,
        'episodes': 0,
        'avg_reward': 0.0,
    }

    portfolio = defaultdict(lambda: {
        'total_reward': 0.0,
        'total_pnl': 0.0,
        'total_trades': 0,
        'episodes': 0,
    })

    per_instrument = defaultdict(lambda: {
        'rewards': [],
        'pnls': [],
        'trades': [],
    })

    epsilon = 1.0

    # Training loop with heartbeat
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

        # Get episode stats
        stats = env.get_episode_stats()
        instrument = stats["instrument"]
        pnl = stats.get("total_pnl", 0)
        trades = stats["trades"]
        asset_class = get_asset_class(instrument)

        # Update cumulative
        cumulative['total_reward'] += total_reward
        cumulative['total_pnl'] += pnl
        cumulative['total_trades'] += trades
        cumulative['episodes'] += 1
        cumulative['avg_reward'] = cumulative['total_reward'] / cumulative['episodes']

        # Update portfolio
        portfolio[asset_class]['total_reward'] += total_reward
        portfolio[asset_class]['total_pnl'] += pnl
        portfolio[asset_class]['total_trades'] += trades
        portfolio[asset_class]['episodes'] += 1

        # Update per-instrument
        per_instrument[instrument]['rewards'].append(total_reward)
        per_instrument[instrument]['pnls'].append(pnl)
        per_instrument[instrument]['trades'].append(trades)

        # Print heartbeat
        print_heartbeat(
            episode=episode + 1,
            total_episodes=episodes,
            instrument=instrument,
            reward=total_reward,
            pnl=pnl,
            trades=trades,
            cumulative=cumulative,
            portfolio=dict(portfolio),
        )

    # Final summary
    print("\n" + "=" * 80)
    print("  FINAL RESULTS (PnL in PERCENTAGE terms for cross-instrument comparison)")
    print("=" * 80)

    print(f"\n[CUMULATIVE]")
    print(f"  Total Episodes: {cumulative['episodes']}")
    print(f"  Total Reward:   {cumulative['total_reward']:+.1f}")
    print(f"  Total PnL:      {cumulative['total_pnl']:+.2f}%")
    print(f"  Avg PnL/Ep:     {cumulative['total_pnl'] / cumulative['episodes'] if cumulative['episodes'] > 0 else 0:+.2f}%")
    print(f"  Avg Reward:     {cumulative['avg_reward']:+.2f}")

    print(f"\n[PORTFOLIO BY ASSET CLASS]")
    print(f"  {'Class':<12} {'Episodes':>8} {'Avg Reward':>12} {'Avg PnL%':>12} {'Total PnL%':>12}")
    print("  " + "-" * 60)
    for cls in sorted(portfolio.keys()):
        stats = portfolio[cls]
        if stats['episodes'] > 0:
            avg_r = stats['total_reward'] / stats['episodes']
            avg_pnl = stats['total_pnl'] / stats['episodes']
            print(f"  {cls:<12} {stats['episodes']:>8} {avg_r:>+12.2f} {avg_pnl:>+12.2f}% {stats['total_pnl']:>+11.2f}%")

    print(f"\n[TOP 10 INSTRUMENTS BY AVG REWARD]")
    inst_avg = []
    for inst, data in per_instrument.items():
        if len(data['rewards']) > 0:
            avg_r = np.mean(data['rewards'])
            avg_pnl = np.mean(data['pnls'])
            inst_avg.append((inst, avg_r, avg_pnl, len(data['rewards'])))

    inst_avg.sort(key=lambda x: x[1], reverse=True)
    print(f"  {'Instrument':<25} {'Eps':>5} {'Avg Reward':>12} {'Avg PnL%':>12}")
    print("  " + "-" * 56)
    for inst, avg_r, avg_pnl, eps in inst_avg[:10]:
        print(f"  {inst:<25} {eps:>5} {avg_r:>+12.2f} {avg_pnl:>+12.2f}%")

    print(f"\n[BOTTOM 5 INSTRUMENTS]")
    for inst, avg_r, avg_pnl, eps in inst_avg[-5:]:
        print(f"  {inst:<25} {eps:>5} {avg_r:>+12.2f} {avg_pnl:>+12.2f}%")

    # Top features
    print(f"\n[TOP FEATURES LEARNED]")
    top_features = tracker.get_top_features_per_action(top_k=5)
    for action, features in top_features.items():
        top_3 = features[:3]
        top_str = ", ".join([f"{name}({corr:+.3f})" for name, corr in top_3])
        print(f"  {action}: {top_str}")

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results = {
        "config": config_name,
        "cumulative": cumulative,
        "portfolio": {k: dict(v) for k, v in portfolio.items()},
        "per_instrument": {k: {
            "avg_reward": float(np.mean(v['rewards'])) if v['rewards'] else 0,
            "avg_pnl": float(np.mean(v['pnls'])) if v['pnls'] else 0,
            "episodes": len(v['rewards']),
        } for k, v in per_instrument.items()},
        "top_features": top_features,
    }

    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    results_file = results_dir / f"heartbeat_{config_name}_{timestamp}.json"

    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n[SAVED] {results_file}")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run RL exploration with automatic data standardization"
    )
    parser.add_argument("--data-dir", default="data/master",
                        help="Source data directory (will be standardized)")
    parser.add_argument("--config", default="Heartbeat_FullDataset")
    parser.add_argument("--mae-w", type=float, default=2.5)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--gamma", type=float, default=0.9)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--skip-standardize", action="store_true",
                        help="Skip data standardization (use raw data)")
    args = parser.parse_args()

    run_with_heartbeat(
        data_dir=args.data_dir,
        config_name=args.config,
        mae_w=args.mae_w,
        lr=args.lr,
        gamma=args.gamma,
        episodes=args.episodes,
        skip_standardize=args.skip_standardize,
    )
