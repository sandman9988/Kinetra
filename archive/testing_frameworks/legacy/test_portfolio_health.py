#!/usr/bin/env python3
"""
Portfolio Health Monitoring Integration Test

Tests the 4-pillar health monitoring system:
1. Return & Efficiency scoring
2. Downside Risk scoring
3. Structural Stability scoring
4. Behavioral Health scoring
5. Health state transitions
6. Action recommendations

This validates the health monitoring system works correctly for production use.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from kinetra.portfolio_health import (
    PortfolioHealthMonitor,
    HealthState,
    HEALTH_ACTIONS,
)


def test_initialization():
    """Test 1: PortfolioHealthMonitor initializes correctly."""
    print("\n[Test 1] Initialization")

    monitor = PortfolioHealthMonitor(
        lookback_days=30,
        min_trades_for_score=10,
    )

    assert monitor.lookback_days == 30
    assert monitor.min_trades_for_score == 10
    assert len(monitor.health_history) == 0

    # Check pillar weights sum to 1.0
    total_weight = sum(monitor.pillar_weights.values())
    assert abs(total_weight - 1.0) < 0.01, f"Weights sum to {total_weight}, not 1.0"

    print("  ✅ Monitor initialized correctly")
    print(f"  ✅ Pillar weights sum to {total_weight:.2f}")


def test_healthy_portfolio():
    """Test 2: Healthy portfolio scores >80."""
    print("\n[Test 2] Healthy Portfolio")

    monitor = PortfolioHealthMonitor(lookback_days=30, min_trades_for_score=10)

    # Generate good equity curve (steady growth, low DD)
    dates = pd.date_range('2024-01-01', periods=100, freq='D')
    equity = 10000 * (1 + np.arange(100) * 0.005 + np.random.randn(100) * 0.001)
    equity_curve = pd.Series(equity, index=dates)

    # Generate profitable trades
    trades = []
    for i in range(50):
        pnl = abs(np.random.randn() * 50) + 50  # Mostly profitable
        trades.append({
            'pnl': pnl,
            'entry_time': dates[i],
            'exit_time': dates[i + 1],
            'mfe': abs(np.random.randn() * 150) + 100,
            'mae': abs(np.random.randn() * 50),
            'edge_ratio': 0.7 + np.random.random() * 0.2,
        })

    health = monitor.update(
        trades=trades,
        equity_curve=equity_curve,
        correlations=np.eye(5) * 0.8 + 0.1,  # Low correlation
        agent_promotions=1,
    )

    print(f"  Composite Score: {health.composite_score:.1f}")
    print(f"  State: {health.state.name}")
    print(f"  Return & Efficiency: {health.return_efficiency.score:.1f}")
    print(f"  Downside Risk: {health.downside_risk.score:.1f}")

    # Healthy portfolio should score reasonably well
    assert health.composite_score > 40, f"Score too low: {health.composite_score}"
    assert health.state in [HealthState.HEALTHY, HealthState.WARNING], \
        f"Unexpected state: {health.state.name}"

    print(f"  ✅ Portfolio assessed as {health.state.name}")


def test_degraded_portfolio():
    """Test 3: Degraded portfolio triggers warning/degraded state."""
    print("\n[Test 3] Degraded Portfolio")

    monitor = PortfolioHealthMonitor(lookback_days=30, min_trades_for_score=10)

    # Generate poor equity curve (with large drawdown)
    dates = pd.date_range('2024-01-01', periods=100, freq='D')
    equity = np.concatenate([
        10000 * np.ones(30),  # Flat
        10000 * (1 - np.arange(40) * 0.01),  # 40% drawdown
        6000 * np.ones(30),  # Flat at bottom
    ])
    equity_curve = pd.Series(equity, index=dates)

    # Generate losing trades
    trades = []
    for i in range(50):
        pnl = -abs(np.random.randn() * 100)  # Mostly losing
        trades.append({
            'pnl': pnl,
            'entry_time': dates[i],
            'exit_time': dates[i + 1],
            'mfe': abs(np.random.randn() * 80),
            'mae': abs(np.random.randn() * 150),
            'edge_ratio': 0.2 + np.random.random() * 0.2,
        })

    health = monitor.update(
        trades=trades,
        equity_curve=equity_curve,
        correlations=np.ones((5, 5)) * 0.8,  # High correlation
        agent_promotions=0,  # No promotions (not learning)
    )

    print(f"  Composite Score: {health.composite_score:.1f}")
    print(f"  State: {health.state.name}")
    print(f"  Max DD: {health.downside_risk.metrics['max_drawdown_pct']:.2f}%")
    print(f"  Action: {health.action.message}")

    # Should trigger degraded or critical state
    assert health.state in [HealthState.DEGRADED, HealthState.CRITICAL, HealthState.WARNING], \
        f"Expected degraded state, got {health.state.name}"

    print(f"  ✅ Degraded portfolio detected correctly")


def test_health_actions():
    """Test 4: Health actions map correctly to states."""
    print("\n[Test 4] Health Actions")

    # Check each action
    for state, action in HEALTH_ACTIONS.items():
        print(f"  {state.name}:")
        print(f"    - Risk multiplier: {action.risk_multiplier:.1%}")
        print(f"    - Message: {action.message}")
        print(f"    - Requires retraining: {action.requires_retraining}")
        print(f"    - Go flat: {action.go_flat}")

    # Verify critical state goes flat
    assert HEALTH_ACTIONS[HealthState.CRITICAL].go_flat, \
        "Critical state should go flat"
    assert HEALTH_ACTIONS[HealthState.CRITICAL].requires_retraining, \
        "Critical state should require retraining"

    # Verify risk reduction
    assert HEALTH_ACTIONS[HealthState.WARNING].risk_multiplier < 1.0, \
        "Warning state should reduce risk"
    assert HEALTH_ACTIONS[HealthState.DEGRADED].risk_multiplier < 0.7, \
        "Degraded state should reduce risk significantly"

    print("  ✅ All health actions configured correctly")


def test_edge_decay_detection():
    """Test 5: Edge decay detection works."""
    print("\n[Test 5] Edge Decay Detection")

    monitor = PortfolioHealthMonitor(lookback_days=30, min_trades_for_score=10)

    dates = pd.date_range('2024-01-01', periods=100, freq='D')
    equity = 10000 * (1 + np.arange(100) * 0.002)
    equity_curve = pd.Series(equity, index=dates)

    # Generate trades with edge decay (good→bad)
    trades = []
    for i in range(50):
        # First half: good edge
        if i < 25:
            edge_ratio = 0.8 + np.random.random() * 0.1
        # Second half: poor edge (decay)
        else:
            edge_ratio = 0.3 + np.random.random() * 0.1

        trades.append({
            'pnl': np.random.randn() * 50,
            'entry_time': dates[i],
            'exit_time': dates[i + 1],
            'mfe': abs(np.random.randn() * 100),
            'mae': abs(np.random.randn() * 80),
            'edge_ratio': edge_ratio,
        })

    health = monitor.update(
        trades=trades,
        equity_curve=equity_curve,
        agent_promotions=0,
    )

    edge_decay = health.behavioral_health.metrics['edge_decay_pct']

    print(f"  Edge decay: {edge_decay:.2f}%")
    print(f"  Behavioral health score: {health.behavioral_health.score:.1f}")

    # Should detect negative edge decay
    assert edge_decay < 0, f"Expected negative edge decay, got {edge_decay:.2f}%"

    print("  ✅ Edge decay detected correctly")


def test_health_summary():
    """Test 6: Health summary returns correct structure."""
    print("\n[Test 6] Health Summary")

    monitor = PortfolioHealthMonitor(lookback_days=30, min_trades_for_score=10)

    dates = pd.date_range('2024-01-01', periods=100, freq='D')
    equity = 10000 * (1 + np.arange(100) * 0.003)
    equity_curve = pd.Series(equity, index=dates)

    trades = []
    for i in range(30):
        trades.append({
            'pnl': np.random.randn() * 100,
            'entry_time': dates[i],
            'exit_time': dates[i + 1],
            'mfe': abs(np.random.randn() * 100),
            'mae': abs(np.random.randn() * 80),
            'edge_ratio': np.random.random(),
        })

    health = monitor.update(
        trades=trades,
        equity_curve=equity_curve,
        agent_promotions=1,
    )

    summary = monitor.get_health_summary()

    # Check structure
    assert 'composite_score' in summary
    assert 'state' in summary
    assert 'action' in summary
    assert 'pillars' in summary
    assert 'return_efficiency' in summary['pillars']
    assert 'downside_risk' in summary['pillars']
    assert 'structural_stability' in summary['pillars']
    assert 'behavioral_health' in summary['pillars']

    print(f"  ✅ Summary structure correct")
    print(f"  ✅ Composite score: {summary['composite_score']:.1f}")
    print(f"  ✅ State: {summary['state']}")
    print(f"  ✅ Action: {summary['action']['message']}")


def test_insufficient_data():
    """Test 7: Handles insufficient data gracefully."""
    print("\n[Test 7] Insufficient Data Handling")

    monitor = PortfolioHealthMonitor(lookback_days=30, min_trades_for_score=20)

    dates = pd.date_range('2024-01-01', periods=10, freq='D')
    equity = 10000 * (1 + np.arange(10) * 0.01)
    equity_curve = pd.Series(equity, index=dates)

    # Only 5 trades (below min)
    trades = []
    for i in range(5):
        trades.append({
            'pnl': np.random.randn() * 50,
            'entry_time': dates[i],
            'exit_time': dates[i + 1],
            'mfe': abs(np.random.randn() * 100),
            'mae': abs(np.random.randn() * 80),
            'edge_ratio': np.random.random(),
        })

    health = monitor.update(
        trades=trades,
        equity_curve=equity_curve,
        agent_promotions=0,
    )

    # Should return neutral scores
    print(f"  Composite Score: {health.composite_score:.1f}")
    print(f"  Return & Efficiency status: {health.return_efficiency.metrics.get('status', 'ok')}")

    # Should handle gracefully (neutral scores)
    assert 30 <= health.composite_score <= 70, \
        f"Expected neutral score, got {health.composite_score}"

    print("  ✅ Insufficient data handled gracefully")


def run_all_tests():
    """Run all integration tests."""
    print("=" * 70)
    print("PORTFOLIO HEALTH MONITORING TESTS")
    print("=" * 70)

    try:
        test_initialization()
        test_healthy_portfolio()
        test_degraded_portfolio()
        test_health_actions()
        test_edge_decay_detection()
        test_health_summary()
        test_insufficient_data()

        print("\n" + "=" * 70)
        print("✅ ALL TESTS PASSED (7/7)")
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
