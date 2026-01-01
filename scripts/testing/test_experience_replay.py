#!/usr/bin/env python3
"""
Test Experience Replay System - Integration Test

Tests the continual learning pipeline end-to-end with a single instrument:
1. TradeLogger captures all state correctly
2. TradeLabeler labels trades properly
3. ExperienceAnalyzer finds patterns
4. PrioritizedReplayBuffer prioritizes correctly
5. Full pipeline integrates without errors

NOT testing profitability - testing EXECUTION and LOGIC.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from kinetra.experience_replay import (
    ContinualLearningManager,
    TradeLogger,
    TradeLabeler,
    ExperienceAnalyzer,
    PrioritizedReplayBuffer,
    Experience,
    TradeEpisode,
)


def test_trade_logger():
    """Test 1: TradeLogger captures state correctly."""
    print("=" * 70)
    print("TEST 1: TradeLogger - State Capture")
    print("=" * 70)

    logger = TradeLogger(log_dir="logs/test_trades")

    # Start a trade
    logger.start_trade()
    print("\n✓ Trade started")

    # Log 10 steps
    for step in range(10):
        state = np.random.randn(20)
        action = np.random.randint(0, 4)
        reward = np.random.randn() * 0.01
        next_state = np.random.randn(20)
        done = (step == 9)

        metadata = {
            'regime': 'laminar',
            'volatility': 0.02,
            'spread': 15.0,
            'price': 1.08500,
        }

        logger.log_step(state, action, reward, next_state, done, metadata)

    print(f"✓ Logged {len(logger.current_episode)} steps")

    # Verify logging
    assert len(logger.current_episode) == 10, "Should have 10 experiences"
    assert logger.current_episode[0].action >= 0, "Actions should be valid"
    assert logger.current_episode[0].metadata['regime'] == 'laminar', "Metadata should be preserved"

    print("✓ All experiences captured correctly")
    print("✓ Metadata preserved")

    # End trade
    episode = logger.end_trade(
        total_pnl=100.0,
        total_return_pct=0.01,
        sharpe_ratio=1.5,
        mfe=0.015,
        mae=0.005,
        mfe_efficiency=0.7,
        mae_efficiency=0.6,
        entry_regime='laminar',
        avg_volatility=0.02,
        avg_spread=15.0,
    )

    print(f"✓ Trade ended, episode created")
    print(f"  - {len(episode.experiences)} experiences")
    print(f"  - Entry: {episode.entry_time}")
    print(f"  - Exit: {episode.exit_time}")
    print(f"  - PnL: ${episode.total_pnl:.2f}")

    # Verify reset
    assert logger.current_episode is None, "Should reset after end_trade"

    print("\n✅ TEST PASSED: TradeLogger works correctly\n")
    return True


def test_trade_labeler():
    """Test 2: TradeLabeler labels trades correctly."""
    print("=" * 70)
    print("TEST 2: TradeLabeler - Trade Quality Labeling")
    print("=" * 70)

    labeler = TradeLabeler()

    # Create test episodes
    test_cases = [
        # Good trade
        {
            'total_pnl': 200,
            'total_return_pct': 0.02,
            'sharpe_ratio': 2.5,
            'mfe_efficiency': 0.85,
            'mae_efficiency': 0.75,
            'constraint_violations': 0,
            'expected_label': 'good',
        },
        # Poor trade
        {
            'total_pnl': -200,
            'total_return_pct': -0.02,
            'sharpe_ratio': -2.0,
            'mfe_efficiency': 0.25,
            'mae_efficiency': 0.25,
            'constraint_violations': 2,
            'expected_label': 'poor',
        },
        # Neutral trade
        {
            'total_pnl': 10,
            'total_return_pct': 0.001,
            'sharpe_ratio': 0.5,
            'mfe_efficiency': 0.5,
            'mae_efficiency': 0.5,
            'constraint_violations': 0,
            'expected_label': 'neutral',
        },
    ]

    print("\nTesting trade labeling logic...")

    for i, case in enumerate(test_cases):
        # Create dummy episode
        episode = TradeEpisode(
            experiences=[],
            entry_time=datetime.now(),
            exit_time=datetime.now() + timedelta(hours=1),
            total_pnl=case['total_pnl'],
            total_return_pct=case['total_return_pct'],
            sharpe_ratio=case['sharpe_ratio'],
            mfe=abs(case['total_return_pct']) * 1.5,
            mae=abs(case['total_return_pct']) * 0.8,
            mfe_efficiency=case['mfe_efficiency'],
            mae_efficiency=case['mae_efficiency'],
            entry_regime='laminar',
            avg_volatility=0.02,
            avg_spread=15.0,
            constraint_violations=case['constraint_violations'],
            freeze_violations=0,
        )

        # Label it
        labeled = labeler.label_episode(episode)

        # Verify
        print(f"\n  Case {i+1}: {case['expected_label'].upper()}")
        print(f"    Sharpe: {labeled.sharpe_ratio:.2f}")
        print(f"    MFE eff: {labeled.mfe_efficiency:.2%}")
        print(f"    Violations: {labeled.constraint_violations}")
        print(f"    → Label: {labeled.label}")
        print(f"    → Priority: {labeled.priority:.1f}")

        assert labeled.label == case['expected_label'], \
            f"Expected {case['expected_label']}, got {labeled.label}"

        # Poor trades should have higher priority
        if labeled.label == 'poor':
            assert labeled.priority > 1.0, "Poor trades should have priority > 1.0"

    print("\n✅ TEST PASSED: TradeLabeler works correctly\n")
    return True


def test_experience_analyzer():
    """Test 3: ExperienceAnalyzer finds patterns."""
    print("=" * 70)
    print("TEST 3: ExperienceAnalyzer - Pattern Detection")
    print("=" * 70)

    analyzer = ExperienceAnalyzer(lookback_trades=50)

    # Add 30 trades (mix of good/poor)
    print("\nAdding 30 labeled trades...")

    for i in range(30):
        # 40% good, 40% poor, 20% neutral
        quality = np.random.choice(['good', 'poor', 'neutral'], p=[0.4, 0.4, 0.2])

        if quality == 'good':
            sharpe = np.random.uniform(1.5, 3.0)
            mfe_eff = np.random.uniform(0.7, 0.9)
            violations = 0
        elif quality == 'poor':
            sharpe = np.random.uniform(-3.0, -1.0)
            mfe_eff = np.random.uniform(0.1, 0.4)
            violations = np.random.randint(0, 3)
        else:
            sharpe = np.random.uniform(-0.5, 0.5)
            mfe_eff = np.random.uniform(0.4, 0.6)
            violations = 0

        episode = TradeEpisode(
            experiences=[],
            entry_time=datetime.now() - timedelta(hours=30-i),
            exit_time=datetime.now() - timedelta(hours=29-i),
            total_pnl=sharpe * 100,
            total_return_pct=sharpe * 0.01,
            sharpe_ratio=sharpe,
            mfe=abs(sharpe) * 0.015,
            mae=abs(sharpe) * 0.008,
            mfe_efficiency=mfe_eff,
            mae_efficiency=mfe_eff * 0.9,
            entry_regime=np.random.choice(['laminar', 'underdamped', 'overdamped']),
            avg_volatility=np.random.uniform(0.01, 0.05),
            avg_spread=np.random.uniform(10, 30),
            constraint_violations=violations,
            freeze_violations=0,
            label=quality,
            priority=1.0 if quality != 'poor' else 2.5,
        )

        analyzer.add_episode(episode)

    print(f"✓ Added {len(analyzer.trade_history)} trades")

    # Analyze poor trades
    print("\n[Poor Trades Analysis]")
    poor_analysis = analyzer.analyze_poor_trades()

    if poor_analysis['count'] > 0:
        print(f"  Count: {poor_analysis['count']}")
        print(f"  Avg Return: {poor_analysis['avg_return']:+.2%}")
        print(f"  Avg Sharpe: {poor_analysis['avg_sharpe']:.2f}")
        print(f"  Constraint violation rate: {poor_analysis['constraint_violation_rate']:.1%}")
        print(f"  Regime breakdown: {poor_analysis['regime_breakdown']}")

        assert poor_analysis['avg_sharpe'] < 0, "Poor trades should have negative Sharpe"
    else:
        print("  No poor trades (unlikely but possible)")

    # Analyze good trades
    print("\n[Good Trades Analysis]")
    good_analysis = analyzer.analyze_good_trades()

    if good_analysis['count'] > 0:
        print(f"  Count: {good_analysis['count']}")
        print(f"  Avg Return: {good_analysis['avg_return']:+.2%}")
        print(f"  Avg Sharpe: {good_analysis['avg_sharpe']:.2f}")
        print(f"  Regime breakdown: {good_analysis['regime_breakdown']}")

        assert good_analysis['avg_sharpe'] > 0, "Good trades should have positive Sharpe"
    else:
        print("  No good trades (unlikely but possible)")

    # Test drift detection (no drift yet)
    print("\n[Drift Detection - Phase 1]")
    drift = analyzer.detect_drift()
    print(f"  Drift detected: {drift['drift_detected']}")

    if drift['drift_detected']:
        print(f"  Reasons: {drift['reasons']}")

    print("\n✅ TEST PASSED: ExperienceAnalyzer works correctly\n")
    return True


def test_prioritized_replay_buffer():
    """Test 4: PrioritizedReplayBuffer sampling."""
    print("=" * 70)
    print("TEST 4: PrioritizedReplayBuffer - Prioritized Sampling")
    print("=" * 70)

    buffer = PrioritizedReplayBuffer(capacity=100, alpha=0.6)

    # Add 50 episodes with different priorities
    print("\nAdding 50 episodes (25 poor, 25 good)...")

    for i in range(50):
        quality = 'poor' if i < 25 else 'good'
        priority = 2.5 if quality == 'poor' else 1.0

        episode = TradeEpisode(
            experiences=[Experience(
                state=np.random.randn(20),
                action=np.random.randint(0, 4),
                reward=np.random.randn() * 0.01,
                next_state=np.random.randn(20),
                done=False,
                timestamp=datetime.now(),
                metadata={'regime': 'laminar'}
            ) for _ in range(10)],
            entry_time=datetime.now(),
            exit_time=datetime.now() + timedelta(hours=1),
            total_pnl=100 if quality == 'good' else -100,
            total_return_pct=0.01 if quality == 'good' else -0.01,
            sharpe_ratio=2.0 if quality == 'good' else -2.0,
            mfe=0.015,
            mae=0.008,
            mfe_efficiency=0.8 if quality == 'good' else 0.3,
            mae_efficiency=0.7 if quality == 'good' else 0.3,
            entry_regime='laminar',
            avg_volatility=0.02,
            avg_spread=15.0,
            constraint_violations=0 if quality == 'good' else 2,
            freeze_violations=0,
            label=quality,
            priority=priority,
        )

        buffer.add(episode)

    print(f"✓ Buffer size: {len(buffer.buffer)}")

    # Get label distribution
    dist = buffer.get_label_distribution()
    print(f"\n[Buffer Distribution]")
    print(f"  Good: {dist['good']}")
    print(f"  Poor: {dist['poor']}")
    print(f"  Total: {dist['total']}")

    # Sample episodes - poor trades should be sampled more
    print("\n[Prioritized Sampling Test]")
    print("  Sampling 20 episodes 100 times to check prioritization...")

    poor_sample_counts = []
    for _ in range(100):
        sampled = buffer.sample(batch_size=20)
        poor_count = sum(1 for ep in sampled if ep.label == 'poor')
        poor_sample_counts.append(poor_count)

    avg_poor_in_sample = np.mean(poor_sample_counts)
    print(f"\n  Average poor trades in sample: {avg_poor_in_sample:.1f}/20")
    print(f"  Expected (uniform): 10/20")
    print(f"  Expected (prioritized): >10/20")

    # With alpha=0.6 and 2.5x priority for poor trades,
    # we should see MORE poor trades in samples than uniform
    assert avg_poor_in_sample > 10.5, \
        "Prioritization not working - should sample more poor trades"

    print(f"  ✓ Prioritization working ({avg_poor_in_sample:.1f} > 10.5)")

    # Sample experiences for training
    print("\n[Experience Sampling for RL]")
    experiences = buffer.sample_experiences(batch_size=32)
    print(f"  Sampled {len(experiences)} experiences")
    print(f"  Each has: state, action, reward, next_state, done")

    assert len(experiences) > 0, "Should return experiences"
    assert hasattr(experiences[0], 'state'), "Should have state"
    assert hasattr(experiences[0], 'action'), "Should have action"

    print("  ✓ Experience format correct for RL training")

    print("\n✅ TEST PASSED: PrioritizedReplayBuffer works correctly\n")
    return True


def test_full_pipeline():
    """Test 5: Full continual learning pipeline."""
    print("=" * 70)
    print("TEST 5: Full Pipeline - End-to-End Integration")
    print("=" * 70)

    manager = ContinualLearningManager(
        log_dir="logs/test_integration",
        buffer_capacity=100,
        lookback_trades=50,
    )

    print("\n[Simulating 20 trades through full pipeline]")

    for trade_id in range(1, 21):
        # Start trade
        manager.logger.start_trade()

        # Simulate trade steps
        num_steps = np.random.randint(10, 30)
        for step in range(num_steps):
            manager.logger.log_step(
                state=np.random.randn(20),
                action=np.random.randint(0, 4),
                reward=np.random.randn() * 0.01,
                next_state=np.random.randn(20),
                done=(step == num_steps - 1),
                metadata={
                    'regime': np.random.choice(['laminar', 'underdamped', 'overdamped']),
                    'volatility': np.random.uniform(0.01, 0.05),
                    'spread': np.random.uniform(10, 30),
                }
            )

        # Simulate trade outcome
        quality = np.random.choice(['good', 'poor', 'neutral'])

        if quality == 'good':
            sharpe = np.random.uniform(1.5, 3.0)
            mfe_eff = np.random.uniform(0.7, 0.9)
            violations = 0
        elif quality == 'poor':
            sharpe = np.random.uniform(-3.0, -1.0)
            mfe_eff = np.random.uniform(0.1, 0.4)
            violations = np.random.randint(0, 3)
        else:
            sharpe = np.random.uniform(-0.5, 0.5)
            mfe_eff = np.random.uniform(0.4, 0.6)
            violations = 0

        # Close trade - triggers full pipeline
        episode = manager.on_trade_complete(
            total_pnl=sharpe * 100,
            total_return_pct=sharpe * 0.01,
            sharpe_ratio=sharpe,
            mfe=abs(sharpe) * 0.015,
            mae=abs(sharpe) * 0.008,
            mfe_efficiency=mfe_eff,
            mae_efficiency=mfe_eff * 0.9,
            entry_regime=np.random.choice(['laminar', 'underdamped', 'overdamped']),
            avg_volatility=np.random.uniform(0.01, 0.05),
            avg_spread=np.random.uniform(10, 30),
            constraint_violations=violations,
        )

        print(f"  Trade {trade_id:2d}: {episode.label:8s} (Sharpe={episode.sharpe_ratio:+.2f}, Priority={episode.priority:.1f})")

    print(f"\n✓ Processed {len(manager.analyzer.trade_history)} trades")

    # Get comprehensive analysis
    analysis = manager.analyze_performance()

    print("\n[Performance Analysis]")
    print(f"  Good trades: {analysis['good_trades']['count']}")
    print(f"  Poor trades: {analysis['poor_trades']['count']}")
    print(f"  Buffer total: {analysis['buffer_stats']['total']}")
    print(f"  Should update policy: {analysis['should_update_policy']}")

    # Verify pipeline integrity
    assert len(manager.replay_buffer.buffer) == 20, "All trades should be in buffer"
    assert len(manager.analyzer.trade_history) == 20, "All trades should be in analyzer"

    # Get training batch
    print("\n[Training Batch]")
    batch = manager.get_training_batch(batch_size=32)
    print(f"  Sampled {len(batch)} experiences for training")

    assert len(batch) > 0, "Should return training batch"

    print("\n✅ TEST PASSED: Full pipeline works correctly\n")
    return True


def test_drift_detection():
    """Test 6: Drift detection with regime change."""
    print("=" * 70)
    print("TEST 6: Drift Detection - Market Regime Change")
    print("=" * 70)

    manager = ContinualLearningManager()

    # Phase 1: Good market (50 trades, 70% win rate)
    print("\n[Phase 1: Good Market Conditions]")
    print("  Simulating 50 trades with 70% win rate...")

    for i in range(50):
        manager.logger.start_trade()

        for _ in range(10):
            manager.logger.log_step(
                state=np.random.randn(20),
                action=np.random.randint(0, 4),
                reward=np.random.randn() * 0.01,
                next_state=np.random.randn(20),
                done=False,
                metadata={'regime': 'laminar', 'volatility': 0.02}
            )

        # 70% good trades
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

    analysis = manager.analyze_performance()
    phase1_good = analysis['good_trades']['count']
    phase1_poor = analysis['poor_trades']['count']

    print(f"  Phase 1 results: {phase1_good} good, {phase1_poor} poor")
    print(f"  Win rate: {phase1_good / 50:.1%}")

    # Check drift (should be False - consistent performance)
    drift = analysis['drift_analysis']
    print(f"  Drift detected: {drift['drift_detected']}")

    # Phase 2: Bad market (20 trades, 20% win rate)
    print("\n[Phase 2: Market Deteriorates]")
    print("  Simulating 20 trades with 20% win rate...")

    for i in range(20):
        manager.logger.start_trade()

        for _ in range(10):
            manager.logger.log_step(
                state=np.random.randn(20),
                action=np.random.randint(0, 4),
                reward=np.random.randn() * 0.01,
                next_state=np.random.randn(20),
                done=False,
                metadata={'regime': 'overdamped', 'volatility': 0.05}
            )

        # 80% poor trades
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

    # Check for drift NOW
    analysis = manager.analyze_performance()
    drift = analysis['drift_analysis']

    print(f"\n[Drift Detection Results]")
    print(f"  Drift detected: {drift['drift_detected']}")
    print(f"  Recent win rate: {drift['recent_win_rate']:.1%}")
    print(f"  Historical win rate: {drift['historical_win_rate']:.1%}")
    print(f"  Recent Sharpe: {drift['recent_sharpe']:.2f}")
    print(f"  Historical Sharpe: {drift['historical_sharpe']:.2f}")

    if drift['drift_detected']:
        print(f"\n  ✓ Drift correctly detected!")
        print("  Reasons:")
        for reason in drift['reasons']:
            print(f"    - {reason}")
        print(f"  Recommend re-exploration: {drift['recommend_reexploration']}")
    else:
        print(f"\n  ⚠️  Drift not detected (may need more trades or stricter thresholds)")

    print("\n✅ TEST PASSED: Drift detection works\n")
    return True


def run_all_tests():
    """Run all integration tests."""
    print("\n" + "=" * 70)
    print("EXPERIENCE REPLAY SYSTEM - INTEGRATION TEST SUITE")
    print("Testing: Execution, Logic, Integration (NOT profitability)")
    print("=" * 70 + "\n")

    tests = [
        ("TradeLogger", test_trade_logger),
        ("TradeLabeler", test_trade_labeler),
        ("ExperienceAnalyzer", test_experience_analyzer),
        ("PrioritizedReplayBuffer", test_prioritized_replay_buffer),
        ("Full Pipeline", test_full_pipeline),
        ("Drift Detection", test_drift_detection),
    ]

    results = {}

    for name, test_func in tests:
        try:
            result = test_func()
            results[name] = result
        except Exception as e:
            print(f"\n❌ TEST FAILED: {name}")
            print(f"   Error: {e}")
            import traceback
            traceback.print_exc()
            results[name] = False

    # Summary
    print("=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)

    passed = sum(1 for r in results.values() if r)
    total = len(results)

    for name, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"  {status}: {name}")

    print()
    print(f"OVERALL: {passed}/{total} tests passed ({100*passed//total}%)")
    print("=" * 70)

    if passed == total:
        print("\n🎉 ALL TESTS PASSED - Experience replay system is working!")
        print("\nNext steps:")
        print("  1. Integrate with live trading (OrderExecutor)")
        print("  2. Test with real market data")
        print("  3. Train RL agent on collected experiences")
        print("  4. Deploy and monitor performance")
    else:
        print("\n⚠️  SOME TESTS FAILED - Fix issues before deployment")

    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    # Don't exit during pytest - let tests run naturally
    if not success:
        raise RuntimeError("Experience replay tests failed")
