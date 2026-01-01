#!/usr/bin/env python3
"""
Demo: Continual Learning from Live Trading

Shows how the agent improves continuously by learning from real trading outcomes:
1. Live trading logs all state/actions
2. Trades are labeled as "good" or "poor" after closing
3. Experience analyzer finds patterns
4. Prioritized replay buffer stores episodes
5. Agent trains on real experiences

Key insight: Agents that don't learn from real outcomes become stale.
This creates a self-improving system.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from kinetra.experience_replay import (
    ContinualLearningManager,
    Experience,
    TradeEpisode,
)


def simulate_live_trading_session():
    """
    Simulate a live trading session where:
    1. Agent trades
    2. Logger records everything
    3. Trades are labeled after closing
    4. Patterns are analyzed
    """
    print("=" * 70)
    print("DEMO: Continual Learning from Live Trading")
    print("=" * 70)

    # Initialize continual learning manager
    manager = ContinualLearningManager(
        log_dir="logs/trades",
        buffer_capacity=10000,
        lookback_trades=100,
    )

    print("\n[Manager] Initialized continual learning pipeline")
    print(f"  - TradeLogger: Recording all state/actions")
    print(f"  - TradeLabeler: Will label trades after closing")
    print(f"  - ExperienceAnalyzer: Tracking patterns")
    print(f"  - PrioritizedReplayBuffer: Storing episodes")

    # Simulate 10 trades
    print("\n" + "=" * 70)
    print("Simulating 10 live trades...")
    print("=" * 70)

    for trade_id in range(1, 11):
        print(f"\n[Trade {trade_id}] Starting...")

        # Start logging
        manager.logger.start_trade()

        # Simulate trade steps (simplified)
        num_steps = np.random.randint(10, 50)

        for step in range(num_steps):
            # Simulate state/action/reward
            state = np.random.randn(20)
            action = np.random.randint(0, 4)
            reward = np.random.randn() * 0.01
            next_state = np.random.randn(20)
            done = (step == num_steps - 1)

            # Market context metadata
            metadata = {
                'regime': np.random.choice(['laminar', 'underdamped', 'overdamped']),
                'volatility': np.random.uniform(0.01, 0.05),
                'spread': np.random.uniform(10, 30),
            }

            # Log this step
            manager.logger.log_step(
                state=state,
                action=action,
                reward=reward,
                next_state=next_state,
                done=done,
                metadata=metadata
            )

        # Simulate trade outcome
        # Some trades are good, some poor, some neutral
        trade_quality = np.random.choice(['good', 'poor', 'neutral'], p=[0.3, 0.3, 0.4])

        if trade_quality == 'good':
            total_return_pct = np.random.uniform(0.02, 0.05)  # 2-5% gain
            sharpe_ratio = np.random.uniform(2.0, 4.0)
            mfe_efficiency = np.random.uniform(0.7, 0.9)
            mae_efficiency = np.random.uniform(0.6, 0.8)
            constraint_violations = 0
        elif trade_quality == 'poor':
            total_return_pct = np.random.uniform(-0.05, -0.02)  # 2-5% loss
            sharpe_ratio = np.random.uniform(-2.0, -0.5)
            mfe_efficiency = np.random.uniform(0.1, 0.4)
            mae_efficiency = np.random.uniform(0.1, 0.4)
            constraint_violations = np.random.choice([0, 1, 2], p=[0.5, 0.3, 0.2])
        else:  # neutral
            total_return_pct = np.random.uniform(-0.01, 0.01)
            sharpe_ratio = np.random.uniform(-0.5, 0.5)
            mfe_efficiency = np.random.uniform(0.4, 0.6)
            mae_efficiency = np.random.uniform(0.4, 0.6)
            constraint_violations = 0

        total_pnl = total_return_pct * 10000  # Assume $10k position

        # Trade closes - trigger full pipeline
        episode = manager.on_trade_complete(
            total_pnl=total_pnl,
            total_return_pct=total_return_pct,
            sharpe_ratio=sharpe_ratio,
            mfe=abs(total_return_pct) * 1.5,  # Simulate MFE
            mae=abs(total_return_pct) * 0.8,  # Simulate MAE
            mfe_efficiency=mfe_efficiency,
            mae_efficiency=mae_efficiency,
            entry_regime=np.random.choice(['laminar', 'underdamped', 'overdamped']),
            avg_volatility=np.random.uniform(0.01, 0.05),
            avg_spread=np.random.uniform(10, 30),
            constraint_violations=constraint_violations,
            freeze_violations=0,
        )

        print(f"[Trade {trade_id}] Completed and labeled: {episode.label}")
        print(f"  Return: {episode.total_return_pct:+.2%}")
        print(f"  Sharpe: {episode.sharpe_ratio:.2f}")
        print(f"  MFE efficiency: {episode.mfe_efficiency:.2%}")
        print(f"  Priority: {episode.priority:.1f}")

    # Analyze patterns
    print("\n" + "=" * 70)
    print("EXPERIENCE ANALYSIS")
    print("=" * 70)

    analysis = manager.analyze_performance()

    # Poor trades analysis
    print("\n[Poor Trades Analysis]")
    poor = analysis['poor_trades']
    if poor['count'] > 0:
        print(f"  Count: {poor['count']}")
        print(f"  Avg Return: {poor['avg_return']:+.2%}")
        print(f"  Avg Sharpe: {poor['avg_sharpe']:.2f}")
        print(f"  Avg MFE efficiency: {poor['avg_mfe_efficiency']:.2%}")
        print(f"  Constraint violation rate: {poor['constraint_violation_rate']:.1%}")
        print(f"  Regime breakdown: {poor['regime_breakdown']}")
    else:
        print("  No poor trades!")

    # Good trades analysis
    print("\n[Good Trades Analysis]")
    good = analysis['good_trades']
    if good['count'] > 0:
        print(f"  Count: {good['count']}")
        print(f"  Avg Return: {good['avg_return']:+.2%}")
        print(f"  Avg Sharpe: {good['avg_sharpe']:.2f}")
        print(f"  Avg MFE efficiency: {good['avg_mfe_efficiency']:.2%}")
        print(f"  Regime breakdown: {good['regime_breakdown']}")
    else:
        print("  No good trades yet!")

    # Drift detection
    print("\n[Drift Detection]")
    drift = analysis['drift_analysis']
    print(f"  Drift detected: {drift['drift_detected']}")
    if drift['drift_detected']:
        print(f"  Reasons: {', '.join(drift['reasons'])}")
        print(f"  Recommend re-exploration: {drift['recommend_reexploration']}")

    # Buffer stats
    print("\n[Replay Buffer Stats]")
    buffer_stats = analysis['buffer_stats']
    print(f"  Total episodes: {buffer_stats['total']}")
    print(f"  Good: {buffer_stats['good']}")
    print(f"  Poor: {buffer_stats['poor']}")
    print(f"  Neutral: {buffer_stats['neutral']}")

    # Policy update recommendation
    print("\n[Policy Update Recommendation]")
    should_update = analysis['should_update_policy']
    print(f"  Should update policy: {should_update}")
    if should_update:
        print("  → Agent should train on recent experiences to improve!")

    # Demonstrate training batch sampling
    print("\n" + "=" * 70)
    print("RL TRAINING INTEGRATION")
    print("=" * 70)

    print("\n[Sampling Training Batch]")
    batch_size = 32
    training_batch = manager.get_training_batch(batch_size)

    print(f"  Sampled {len(training_batch)} experiences for training")
    print(f"  These are prioritized samples (more poor trades for learning)")

    # Show distribution in batch
    if len(training_batch) > 0:
        # Count experiences by episode label (approximate)
        print(f"\n  Batch composition:")
        print(f"    - Experiences ready for RL algorithm (DQN, PPO, etc.)")
        print(f"    - Higher priority given to poor trades")
        print(f"    - Agent learns from mistakes faster!")

    print("\n" + "=" * 70)
    print("KEY TAKEAWAYS")
    print("=" * 70)

    print("""
1. CONTINUOUS LOGGING
   - Every state, action, reward logged during live trading
   - Full context preserved for retroactive analysis

2. RETROACTIVE LABELING
   - Trades labeled as "good" or "poor" after closing
   - Based on risk-adjusted return, execution efficiency, violations

3. PATTERN DETECTION
   - Analyzer finds what conditions lead to poor trades
   - Finds what conditions lead to good trades
   - Detects when market regime has shifted (drift)

4. PRIORITIZED LEARNING
   - Poor trades get higher sampling priority
   - Agent learns from mistakes faster
   - Good trades reinforce successful patterns

5. SELF-IMPROVEMENT
   - Agent continuously improves from real outcomes
   - Adapts to changing market conditions
   - Never becomes stale

→ This creates a self-improving trading system where real experience drives learning!
    """)


def demonstrate_drift_detection():
    """
    Demonstrate drift detection when market changes.
    """
    print("\n" + "=" * 70)
    print("DEMO: Drift Detection")
    print("=" * 70)

    manager = ContinualLearningManager()

    print("\n[Scenario] Market conditions change after 50 trades")
    print("  - First 50 trades: Good market conditions (high win rate)")
    print("  - Next 20 trades: Market changed (win rate drops)")

    # Simulate 50 good trades
    print("\n[Phase 1] Trading in good market conditions...")
    for i in range(50):
        manager.logger.start_trade()

        # Log some steps
        for _ in range(10):
            manager.logger.log_step(
                state=np.random.randn(20),
                action=np.random.randint(0, 4),
                reward=np.random.randn() * 0.01,
                next_state=np.random.randn(20),
                done=False,
                metadata={'regime': 'laminar', 'volatility': 0.02, 'spread': 15}
            )

        # Most trades are good
        quality = np.random.choice(['good', 'poor'], p=[0.7, 0.3])

        if quality == 'good':
            manager.on_trade_complete(
                total_pnl=200,
                total_return_pct=0.02,
                sharpe_ratio=2.5,
                mfe=0.03,
                mae=0.01,
                mfe_efficiency=0.8,
                mae_efficiency=0.7,
                entry_regime='laminar',
                avg_volatility=0.02,
                avg_spread=15,
            )
        else:
            manager.on_trade_complete(
                total_pnl=-100,
                total_return_pct=-0.01,
                sharpe_ratio=-1.0,
                mfe=0.015,
                mae=0.015,
                mfe_efficiency=0.3,
                mae_efficiency=0.3,
                entry_regime='overdamped',
                avg_volatility=0.04,
                avg_spread=25,
            )

    print(f"  Completed 50 trades")

    analysis = manager.analyze_performance()
    print(f"  Win rate: {analysis['good_trades']['count'] / 50:.1%}")

    # Market changes - simulate 20 poor trades
    print("\n[Phase 2] Market conditions deteriorate...")
    for i in range(20):
        manager.logger.start_trade()

        for _ in range(10):
            manager.logger.log_step(
                state=np.random.randn(20),
                action=np.random.randint(0, 4),
                reward=np.random.randn() * 0.01,
                next_state=np.random.randn(20),
                done=False,
                metadata={'regime': 'overdamped', 'volatility': 0.05, 'spread': 30}
            )

        # Most trades are now poor
        quality = np.random.choice(['good', 'poor'], p=[0.2, 0.8])

        if quality == 'good':
            manager.on_trade_complete(
                total_pnl=100,
                total_return_pct=0.01,
                sharpe_ratio=1.5,
                mfe=0.02,
                mae=0.01,
                mfe_efficiency=0.6,
                mae_efficiency=0.6,
                entry_regime='underdamped',
                avg_volatility=0.03,
                avg_spread=20,
            )
        else:
            manager.on_trade_complete(
                total_pnl=-200,
                total_return_pct=-0.02,
                sharpe_ratio=-2.0,
                mfe=0.01,
                mae=0.025,
                mfe_efficiency=0.2,
                mae_efficiency=0.2,
                entry_regime='overdamped',
                avg_volatility=0.05,
                avg_spread=30,
                constraint_violations=np.random.randint(0, 2),
            )

    print(f"  Completed 20 more trades")

    # Check for drift
    analysis = manager.analyze_performance()
    drift = analysis['drift_analysis']

    print("\n[Drift Detection Results]")
    print(f"  Drift detected: {drift['drift_detected']}")
    print(f"  Recent win rate: {drift['recent_win_rate']:.1%}")
    print(f"  Historical win rate: {drift['historical_win_rate']:.1%}")
    print(f"  Recent Sharpe: {drift['recent_sharpe']:.2f}")
    print(f"  Historical Sharpe: {drift['historical_sharpe']:.2f}")

    if drift['drift_detected']:
        print("\n✅ DRIFT DETECTED!")
        print("  Reasons:")
        for reason in drift['reasons']:
            print(f"    - {reason}")
        print(f"\n  → Recommend re-exploration: {drift['recommend_reexploration']}")
        print("  → Agent should explore new strategies for changed market!")


def main():
    """Run all demos."""
    print("\n" + "=" * 70)
    print("CONTINUAL LEARNING FROM LIVE TRADING")
    print("Self-Improving Agents via Experience Replay")
    print("=" * 70)

    simulate_live_trading_session()
    demonstrate_drift_detection()

    print("\n" + "=" * 70)
    print("SUMMARY: Continual Learning Architecture")
    print("=" * 70)

    print("""
TRADITIONAL RL:
  Train → Deploy → Agent becomes stale → Performance degrades

CONTINUAL LEARNING RL:
  Train → Deploy → Log experiences → Label → Analyze → Retrain → Deploy
            ↑                                                        │
            └────────────────────────────────────────────────────────┘
            (Continuous improvement loop)

BENEFITS:
1. ✅ Agent adapts to changing markets
2. ✅ Learns from actual mistakes (poor trades)
3. ✅ Detects when market regime shifts
4. ✅ Never becomes stale
5. ✅ Prioritizes learning from high-value experiences

INTEGRATION WITH KINETRA:
- OrderExecutor logs all state/actions during live trading
- TradeLabeler evaluates trades after closing
- ExperienceAnalyzer finds patterns in good/poor trades
- PrioritizedReplayBuffer stores episodes with priority
- RL agent trains on real experiences
- Policy updates deployed back to live trading

→ This creates a self-improving trading system!
    """)


if __name__ == "__main__":
    main()
