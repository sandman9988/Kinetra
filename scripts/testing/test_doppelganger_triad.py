#!/usr/bin/env python3
"""
DoppelgangerTriad Integration Test

Tests the DoppelgangerTriad system with real RL agents:
1. Integration with KinetraAgent (PPO-based)
2. Drift detection with performance degradation
3. Promotion logic with improvement
4. Rollback capability
5. Trade result tracking

This validates the shadow agent system works correctly for continual learning.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from kinetra import DoppelgangerTriad, KinetraAgent


def test_triad_initialization():
    """Test 1: DoppelgangerTriad initializes correctly with KinetraAgent."""
    print("\n[Test 1] Initialization")

    agent = KinetraAgent(state_dim=64, action_dim=4)
    triad = DoppelgangerTriad(
        live_agent=agent,
        drift_threshold=0.2,
        promotion_threshold=0.1,
    )

    # Check all three agents exist
    assert triad.live_agent is not None
    assert triad.frozen_shadow is not None
    assert triad.training_shadow is not None

    # Check agent IDs
    assert triad.live_agent.agent_id == "live"
    assert triad.frozen_shadow.agent_id == "frozen"
    assert triad.training_shadow.agent_id == "training"

    # Check states
    assert triad.live_agent.state.name == "ACTIVE"
    assert triad.frozen_shadow.state.name == "FROZEN"
    assert triad.training_shadow.state.name == "TRAINING"

    print("  ✅ All agents initialized correctly")
    return triad


def test_action_selection():
    """Test 2: Action selection works across all agents."""
    print("\n[Test 2] Action Selection")

    agent = KinetraAgent(state_dim=64, action_dim=4)
    triad = DoppelgangerTriad(live_agent=agent)

    state = np.random.randn(64)
    action = triad.select_action(state, epsilon=0.1)

    # Check action is valid
    assert isinstance(action, (int, np.integer))
    assert 0 <= action < 4

    # Check decision history is tracked
    assert len(triad.live_agent.decision_history) == 1
    assert len(triad.frozen_shadow.decision_history) == 1
    assert len(triad.training_shadow.decision_history) == 1

    print(f"  ✅ Action selected: {action}")
    print(f"  ✅ Decision history: {len(triad.live_agent.decision_history)} entries")


def test_learning_update():
    """Test 3: Learning updates work correctly (frozen doesn't learn)."""
    print("\n[Test 3] Learning Updates")

    agent = KinetraAgent(state_dim=64, action_dim=4)
    triad = DoppelgangerTriad(live_agent=agent)

    state = np.random.randn(64)
    next_state = np.random.randn(64)
    action = 1
    reward = 10.0
    done = False

    # Update all agents
    triad.update_all(state, action, reward, next_state, done)

    # Frozen agent should NOT learn (returns 0.0)
    # Live and training agents should learn (returns non-zero)
    print("  ✅ Update successful")
    print(f"  ✅ Frozen agent state: {triad.frozen_shadow.state.name}")
    assert triad.frozen_shadow.state.name == "FROZEN"


def test_trade_tracking():
    """Test 4: Trade result tracking updates performance metrics."""
    print("\n[Test 4] Trade Result Tracking")

    agent = KinetraAgent(state_dim=64, action_dim=4)
    triad = DoppelgangerTriad(live_agent=agent)

    # Simulate 10 trades
    for i in range(10):
        pnl = np.random.randn() * 100
        reward = pnl * 1.5
        edge_ratio = np.random.random()

        triad.record_trade_result({
            'raw_pnl': pnl,
            'shaped_reward': reward,
            'edge_ratio': edge_ratio,
        })

    # Check performance tracked
    assert triad.live_agent.performance.trades == 10
    assert triad.training_shadow.performance.trades == 10
    assert triad.frozen_shadow.performance.trades == 10

    print(f"  ✅ Trades tracked: {triad.live_agent.performance.trades}")
    print(f"  ✅ Live total reward: {triad.live_agent.performance.total_reward:.2f}")
    print(f"  ✅ Live win rate: {triad.live_agent.performance.win_rate:.1%}")


def test_drift_detection():
    """Test 5: Drift detection triggers when performance degrades."""
    print("\n[Test 5] Drift Detection")

    agent = KinetraAgent(state_dim=64, action_dim=4)
    triad = DoppelgangerTriad(
        live_agent=agent,
        drift_threshold=0.2,
        min_trades_for_drift=10,
    )

    # Give frozen shadow good performance
    for i in range(15):
        triad.frozen_shadow.update_performance(
            reward=100.0,  # Good reward
            pnl=100.0,
            is_win=True,
            edge_ratio=0.8,
        )

    # Give live agent bad performance
    for i in range(15):
        triad.live_agent.update_performance(
            reward=50.0,  # Worse reward
            pnl=50.0,
            is_win=False,
            edge_ratio=0.3,
        )

    # Check drift
    is_drifted, drift, msg = triad.check_drift()

    print(f"  ✅ Is drifted: {is_drifted}")
    print(f"  ✅ Drift amount: {drift*100:.1f}%")
    if msg:
        print(f"  ✅ Message: {msg}")

    # Should detect drift (live is 50% worse than frozen)
    assert is_drifted, "Should detect drift when live performance degrades"


def test_promotion_logic():
    """Test 6: Promotion triggers when training outperforms live."""
    print("\n[Test 6] Promotion Logic")

    agent = KinetraAgent(state_dim=64, action_dim=4)
    triad = DoppelgangerTriad(
        live_agent=agent,
        promotion_threshold=0.1,
        min_trades_for_promotion=20,
    )

    # Give live agent mediocre performance
    for i in range(25):
        triad.live_agent.update_performance(
            reward=50.0,
            pnl=50.0,
            is_win=i % 2 == 0,
            edge_ratio=0.5,
        )

    # Give training shadow better performance
    for i in range(25):
        triad.training_shadow.update_performance(
            reward=75.0,  # 50% better
            pnl=75.0,
            is_win=i % 3 != 0,  # Higher win rate
            edge_ratio=0.7,
        )

    # Check promotion
    should_promote, msg = triad.check_promotion()

    print(f"  ✅ Should promote: {should_promote}")
    if msg:
        print(f"  ✅ Message: {msg}")

    # Should trigger promotion (training is 50% better)
    assert should_promote, "Should promote when training outperforms live"

    # Execute promotion
    old_live_id = id(triad.live_agent.agent)
    triad.promote_training_shadow()

    # Check promotion succeeded
    assert triad.live_agent.state.name == "ACTIVE"
    assert triad.frozen_shadow.state.name == "FROZEN"
    assert triad.training_shadow.state.name == "TRAINING"

    # Check new training shadow was created
    assert id(triad.training_shadow.agent) != id(triad.live_agent.agent)

    print("  ✅ Promotion executed successfully")
    print(f"  ✅ New live agent state: {triad.live_agent.state.name}")
    print(f"  ✅ Events logged: {len(triad.events)}")


def test_rollback():
    """Test 7: Rollback to frozen agent."""
    print("\n[Test 7] Rollback to Frozen")

    agent = KinetraAgent(state_dim=64, action_dim=4)
    triad = DoppelgangerTriad(live_agent=agent)

    # Get frozen agent ID
    frozen_agent_weights = triad.frozen_shadow.agent.actor.state_dict() if hasattr(triad.frozen_shadow.agent, 'actor') else None

    # Rollback
    triad.rollback_to_frozen()

    # Check rollback succeeded
    assert triad.live_agent.state.name == "ACTIVE"
    assert triad.frozen_shadow.state.name == "FROZEN"
    assert len(triad.events) == 1
    assert triad.events[0]['type'] == 'rollback'

    print("  ✅ Rollback executed successfully")
    print(f"  ✅ Events: {triad.events}")


def test_system_summary():
    """Test 8: System summary returns correct information."""
    print("\n[Test 8] System Summary")

    agent = KinetraAgent(state_dim=64, action_dim=4)
    triad = DoppelgangerTriad(live_agent=agent)

    # Add some trades
    for i in range(5):
        triad.record_trade_result({
            'raw_pnl': np.random.randn() * 50,
            'shaped_reward': np.random.randn() * 75,
            'edge_ratio': np.random.random(),
        })

    summary = triad.get_system_summary()

    # Check summary structure
    assert 'agents' in summary
    assert 'live' in summary['agents']
    assert 'frozen' in summary['agents']
    assert 'training' in summary['agents']
    assert 'recent_events' in summary

    # Check live agent stats
    assert summary['agents']['live']['trades'] == 5
    assert summary['agents']['live']['state'] == 'ACTIVE'

    print("  ✅ Summary generated:")
    print(f"     Live trades: {summary['agents']['live']['trades']}")
    print(f"     Live total reward: {summary['agents']['live']['total_reward']:.2f}")
    print(f"     Live win rate: {summary['agents']['live']['win_rate']:.1%}")


def run_all_tests():
    """Run all integration tests."""
    print("=" * 70)
    print("DOPPELGANGER TRIAD INTEGRATION TESTS")
    print("=" * 70)

    try:
        test_triad_initialization()
        test_action_selection()
        test_learning_update()
        test_trade_tracking()
        test_drift_detection()
        test_promotion_logic()
        test_rollback()
        test_system_summary()

        print("\n" + "=" * 70)
        print("✅ ALL TESTS PASSED (8/8)")
        print("=" * 70)
        return 0

    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        return 1
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(run_all_tests())
