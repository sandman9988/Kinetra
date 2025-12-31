#!/usr/bin/env python3
"""
DoppelgangerTriad: Shadow Agent Risk Management System

A three-agent architecture for continuous improvement with drift detection:
1. Live Agent: Primary trading agent (executes actual trades)
2. Shadow A (Frozen): Frozen checkpoint for drift detection
3. Shadow B (Training): Continuously retrained candidate for promotion

This system enables:
- Drift detection: Compare live vs frozen to detect performance degradation
- Continuous improvement: Shadow B learns from new data
- Safe promotion: Only promote shadow if it outperforms live
- Rollback capability: Restore from frozen if live degrades

Designed for LIVE TRADING first, backtest compatibility second.
"""

import copy
import logging
from dataclasses import dataclass
from datetime import datetime
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# =============================================================================
# SHADOW AGENT ARCHITECTURE
# =============================================================================

class ShadowAgentState(Enum):
    """State of a shadow agent."""
    ACTIVE = auto()      # Currently trading
    FROZEN = auto()      # Frozen for comparison
    TRAINING = auto()    # Being retrained
    CANDIDATE = auto()   # Ready for promotion


@dataclass
class AgentPerformance:
    """Performance metrics for an agent."""
    total_reward: float = 0.0
    total_pnl: float = 0.0
    trades: int = 0
    win_rate: float = 0.0
    avg_edge_ratio: float = 0.0
    sharpe: float = 0.0
    last_updated: Optional[datetime] = None


class ShadowAgent:
    """
    Shadow agent in the DoppelgangerTriad system.

    Each shadow agent wraps a base agent and tracks its performance.
    Compatible with any agent that has:
    - select_action(state, epsilon) -> action
    - update(state, action, reward, next_state, done) -> loss
    - get_q_values(state) -> np.ndarray (optional, for drift detection)
    """

    def __init__(
        self,
        agent: Any,  # Base agent (KinetraAgent, LinearQAgent, etc.)
        agent_id: str,
        state: ShadowAgentState = ShadowAgentState.ACTIVE,
    ):
        self.agent = agent
        self.agent_id = agent_id
        self.state = state
        self.performance = AgentPerformance()
        self.creation_time = datetime.now()
        self.frozen_at: Optional[datetime] = None

        # Track decisions for drift detection
        self.decision_history: List[Tuple[np.ndarray, int, float]] = []

    def select_action(self, state: np.ndarray, epsilon: float = 0.1) -> int:
        """Select action and track for drift detection."""
        action = self.agent.select_action(state, epsilon)

        # Track decision for drift detection (if agent supports get_q_values)
        if hasattr(self.agent, 'get_q_values'):
            q_values = self.agent.get_q_values(state)
            self.decision_history.append((state.copy(), action, q_values.max()))
        else:
            # For agents without Q-values, just track action
            self.decision_history.append((state.copy(), action, 0.0))

        # Keep only recent history
        if len(self.decision_history) > 1000:
            self.decision_history = self.decision_history[-1000:]

        return action

    def update(self, state, action, reward, next_state, done):
        """Update agent (only if not frozen)."""
        if self.state == ShadowAgentState.FROZEN:
            return 0.0  # Frozen agents don't learn

        # Handle different agent types
        # PPO agents (KinetraAgent): store_transition() then update()
        if hasattr(self.agent, 'store_transition'):
            # PPO-style agent - no update per step
            return 0.0
        # Q-learning agents (LinearQAgent): update(state, action, reward, next_state, done)
        elif hasattr(self.agent, 'update'):
            # Check signature of update method
            import inspect
            sig = inspect.signature(self.agent.update)
            if len(sig.parameters) > 1:  # Has parameters beyond self
                return self.agent.update(state, action, reward, next_state, done)
            else:  # PPO-style update with no args
                return self.agent.update()
        else:
            return 0.0

    def freeze(self):
        """Freeze agent (stop learning, preserve state)."""
        self.state = ShadowAgentState.FROZEN
        self.frozen_at = datetime.now()

    def unfreeze(self):
        """Unfreeze agent (resume learning)."""
        self.state = ShadowAgentState.TRAINING

    def clone(self, new_id: str) -> "ShadowAgent":
        """Create a clone of this agent."""
        agent_copy = copy.deepcopy(self.agent)
        new_shadow = ShadowAgent(
            agent=agent_copy,
            agent_id=new_id,
            state=ShadowAgentState.TRAINING,
        )
        return new_shadow

    def update_performance(
        self,
        reward: float,
        pnl: float,
        is_win: bool,
        edge_ratio: float,
    ):
        """Update performance metrics."""
        self.performance.total_reward += reward
        self.performance.total_pnl += pnl
        self.performance.trades += 1

        # Rolling win rate
        total_wins = self.performance.win_rate * (self.performance.trades - 1)
        if is_win:
            total_wins += 1
        self.performance.win_rate = total_wins / self.performance.trades

        # Rolling edge ratio
        total_edge = self.performance.avg_edge_ratio * (self.performance.trades - 1)
        total_edge += edge_ratio
        self.performance.avg_edge_ratio = total_edge / self.performance.trades

        self.performance.last_updated = datetime.now()


class DoppelgangerTriad:
    """
    DoppelgangerTriad: Three-Agent Risk Management System.

    Architecture:
    1. Live Agent: The primary agent that actually executes trades
    2. Shadow Agent A (Frozen): Frozen copy of live agent for drift detection
    3. Shadow Agent B (Training): Continuously retrained candidate for promotion

    Key Features:
    - Drift detection: Compare live vs frozen to detect performance degradation
    - Continuous improvement: Shadow B learns from new data
    - Safe promotion: Only promote shadow if it outperforms live
    - Rollback capability: Restore from frozen if live degrades
    - Designed for live trading first, backtest compatibility second

    Usage:
        # Initialize with any RL agent
        from kinetra import KinetraAgent, DoppelgangerTriad

        agent = KinetraAgent(state_dim=64, action_dim=4)
        triad = DoppelgangerTriad(
            live_agent=agent,
            drift_threshold=0.2,  # 20% performance drop triggers warning
            promotion_threshold=0.1,  # 10% improvement triggers promotion
        )

        # Use in trading loop
        action = triad.select_action(state, physics_state, epsilon=0.1)

        # After trade completes
        triad.record_trade_result({
            'raw_pnl': pnl,
            'shaped_reward': reward,
            'edge_ratio': mfe / hypotenuse,
        })

        # Periodic checks
        is_drifted, drift, msg = triad.check_drift()
        should_promote, msg = triad.check_promotion()

        if should_promote:
            triad.promote_training_shadow()
    """

    def __init__(
        self,
        live_agent: Any,
        drift_threshold: float = 0.2,  # 20% performance drift triggers warning
        promotion_threshold: float = 0.1,  # 10% better to promote
        min_trades_for_drift: int = 20,  # Minimum trades before drift detection
        min_trades_for_promotion: int = 30,  # Minimum trades before promotion
    ):
        self.drift_threshold = drift_threshold
        self.promotion_threshold = promotion_threshold
        self.min_trades_for_drift = min_trades_for_drift
        self.min_trades_for_promotion = min_trades_for_promotion

        # Initialize shadow agents
        self.live_agent = ShadowAgent(
            agent=live_agent,
            agent_id="live",
            state=ShadowAgentState.ACTIVE,
        )

        self.frozen_shadow = self.live_agent.clone("frozen")
        self.frozen_shadow.freeze()

        self.training_shadow = self.live_agent.clone("training")
        self.training_shadow.state = ShadowAgentState.TRAINING

        # Logger
        self.logger = logging.getLogger("doppelganger_triad")

        # Event history
        self.events: List[Dict] = []

    def select_action(
        self,
        state: np.ndarray,
        epsilon: float = 0.1,
    ) -> int:
        """
        Select action using the live agent.

        Args:
            state: Current state vector
            epsilon: Exploration rate

        Returns:
            action: Selected action (int)
        """
        # All agents get to see the state (for parallel evaluation)
        live_action = self.live_agent.select_action(state, epsilon)
        _ = self.frozen_shadow.select_action(state, 0.0)  # No exploration
        _ = self.training_shadow.select_action(state, epsilon * 0.5)  # Less exploration

        # But only live agent's action is used
        return live_action

    def update_all(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ):
        """Update all agents (respecting their states)."""
        # Live agent learns
        self.live_agent.update(state, action, reward, next_state, done)

        # Frozen doesn't learn (handled in ShadowAgent.update)
        self.frozen_shadow.update(state, action, reward, next_state, done)

        # Training shadow learns more aggressively
        self.training_shadow.update(state, action, reward, next_state, done)

    def record_trade_result(
        self,
        trade_data: Dict,
    ):
        """
        Record trade result for all tracking systems.

        Args:
            trade_data: Dictionary with keys:
                - raw_pnl: Raw P&L (float)
                - shaped_reward: Shaped reward (float)
                - edge_ratio: MFE efficiency (0-1)
        """
        # Extract metrics
        pnl = trade_data.get("raw_pnl", 0)
        reward = trade_data.get("shaped_reward", pnl)
        is_win = pnl > 0
        edge_ratio = trade_data.get("edge_ratio", 0.5)

        # Update agent performances
        self.live_agent.update_performance(reward, pnl, is_win, edge_ratio)
        self.training_shadow.update_performance(reward, pnl, is_win, edge_ratio)
        # Frozen shadow also tracks (for comparison)
        self.frozen_shadow.update_performance(reward, pnl, is_win, edge_ratio)

    def check_drift(self) -> Tuple[bool, float, str]:
        """
        Check for performance drift between live and frozen agents.

        Returns:
            (is_drifted, drift_amount, message)
        """
        if self.live_agent.performance.trades < self.min_trades_for_drift:
            return False, 0.0, "Insufficient trades for drift detection"

        live_perf = self.live_agent.performance
        frozen_perf = self.frozen_shadow.performance

        # Compare average rewards per trade
        live_avg = live_perf.total_reward / max(1, live_perf.trades)
        frozen_avg = frozen_perf.total_reward / max(1, frozen_perf.trades)

        if frozen_avg > 0:
            drift = (frozen_avg - live_avg) / frozen_avg
        else:
            drift = 0.0

        is_drifted = drift > self.drift_threshold

        if is_drifted:
            msg = f"Performance drift detected: {drift*100:.1f}% worse than frozen"
            self.logger.warning(msg)
            self.events.append({
                "type": "drift_detected",
                "timestamp": datetime.now().isoformat(),
                "drift": drift,
                "live_avg": live_avg,
                "frozen_avg": frozen_avg,
            })
            return True, drift, msg

        return False, drift, ""

    def check_promotion(self) -> Tuple[bool, str]:
        """
        Check if training shadow should be promoted to live.

        Returns:
            (should_promote, message)
        """
        if self.training_shadow.performance.trades < self.min_trades_for_promotion:
            return False, "Insufficient trades for promotion evaluation"

        live_perf = self.live_agent.performance
        training_perf = self.training_shadow.performance

        # Compare average rewards
        live_avg = live_perf.total_reward / max(1, live_perf.trades)
        training_avg = training_perf.total_reward / max(1, training_perf.trades)

        if live_avg > 0:
            improvement = (training_avg - live_avg) / live_avg
        else:
            improvement = training_avg if training_avg > 0 else 0

        should_promote = improvement > self.promotion_threshold

        if should_promote:
            msg = f"Training shadow ready for promotion: {improvement*100:.1f}% improvement"
            self.logger.info(msg)
            return True, msg

        return False, ""

    def promote_training_shadow(self):
        """Promote training shadow to live, demote live to frozen."""
        self.logger.info("Promoting training shadow to live agent")

        # Current live becomes new frozen
        old_live = self.live_agent
        old_live.freeze()

        # Training becomes new live
        self.training_shadow.state = ShadowAgentState.ACTIVE
        self.live_agent = self.training_shadow

        # Old frozen is discarded, old live is new frozen
        self.frozen_shadow = old_live

        # Create new training shadow from new live
        self.training_shadow = self.live_agent.clone("training")
        self.training_shadow.state = ShadowAgentState.TRAINING

        self.events.append({
            "type": "promotion",
            "timestamp": datetime.now().isoformat(),
            "new_live_perf": self.live_agent.performance.total_reward,
        })

    def rollback_to_frozen(self):
        """Rollback live agent to frozen version (restore known-good state)."""
        self.logger.warning("Rolling back to frozen agent")

        # Clone frozen to become new live
        restored = self.frozen_shadow.clone("live")
        restored.state = ShadowAgentState.ACTIVE  # Set as active (no unfreeze needed)

        # Keep current frozen as is
        self.live_agent = restored

        # Create new training from restored
        self.training_shadow = restored.clone("training")
        self.training_shadow.state = ShadowAgentState.TRAINING

        self.events.append({
            "type": "rollback",
            "timestamp": datetime.now().isoformat(),
        })

    def get_system_summary(self) -> Dict[str, Any]:
        """Get full system status summary."""
        return {
            "agents": {
                "live": {
                    "state": self.live_agent.state.name,
                    "trades": self.live_agent.performance.trades,
                    "total_reward": self.live_agent.performance.total_reward,
                    "total_pnl": self.live_agent.performance.total_pnl,
                    "win_rate": self.live_agent.performance.win_rate,
                },
                "frozen": {
                    "state": self.frozen_shadow.state.name,
                    "trades": self.frozen_shadow.performance.trades,
                    "frozen_at": self.frozen_shadow.frozen_at.isoformat()
                    if self.frozen_shadow.frozen_at else None,
                },
                "training": {
                    "state": self.training_shadow.state.name,
                    "trades": self.training_shadow.performance.trades,
                    "total_reward": self.training_shadow.performance.total_reward,
                },
            },
            "recent_events": self.events[-5:] if self.events else [],
        }


# =============================================================================
# DEMO
# =============================================================================

def demo_doppelganger_triad():
    """Demonstrate the DoppelgangerTriad system."""
    print("=" * 70)
    print("DOPPELGANGER TRIAD - DEMO")
    print("=" * 70)

    # Create mock agent
    class MockAgent:
        def __init__(self, state_dim=64, n_actions=4):
            self.state_dim = state_dim
            self.n_actions = n_actions
            self.weights = np.random.randn(n_actions, state_dim) * 0.01

        def select_action(self, state, epsilon=0.1):
            if np.random.random() < epsilon:
                return np.random.randint(self.n_actions)
            q = self.weights @ state
            return int(np.argmax(q))

        def update(self, state, action, reward, next_state, done):
            return 0.01

        def get_q_values(self, state):
            return self.weights @ state

    # Create triad system
    agent = MockAgent()
    triad = DoppelgangerTriad(
        live_agent=agent,
        drift_threshold=0.2,
        promotion_threshold=0.1,
    )

    print("\n[1] System initialized")
    print(f"    Live agent: {triad.live_agent.agent_id}")
    print(f"    Frozen shadow: {triad.frozen_shadow.agent_id}")
    print(f"    Training shadow: {triad.training_shadow.agent_id}")

    # Test action selection
    state = np.random.randn(64)
    action = triad.select_action(state, epsilon=0.1)

    print(f"\n[2] Action selection test:")
    print(f"    Action: {action}")

    # Simulate 50 trades
    print(f"\n[3] Simulating 50 trades...")
    for i in range(50):
        state = np.random.randn(64)
        next_state = np.random.randn(64)
        action = triad.select_action(state, epsilon=0.1)
        reward = np.random.randn() * 100
        done = i % 10 == 9

        triad.update_all(state, action, reward, next_state, done)

        if i % 5 == 4:  # Every 5 steps, record trade
            triad.record_trade_result({
                'raw_pnl': reward,
                'shaped_reward': reward * 1.5,
                'edge_ratio': np.random.random(),
            })

    # Check drift
    is_drifted, drift, msg = triad.check_drift()
    print(f"\n[4] Drift check:")
    print(f"    Is drifted: {is_drifted}")
    print(f"    Drift amount: {drift*100:.1f}%")
    if msg:
        print(f"    Message: {msg}")

    # Check promotion
    should_promote, msg = triad.check_promotion()
    print(f"\n[5] Promotion check:")
    print(f"    Should promote: {should_promote}")
    if msg:
        print(f"    Message: {msg}")

    # Show system summary
    print(f"\n[6] System Summary:")
    summary = triad.get_system_summary()
    print(f"    Live agent trades: {summary['agents']['live']['trades']}")
    print(f"    Live agent total reward: {summary['agents']['live']['total_reward']:.2f}")
    print(f"    Live agent win rate: {summary['agents']['live']['win_rate']:.1%}")
    print(f"    Training agent trades: {summary['agents']['training']['trades']}")
    print(f"    Training agent total reward: {summary['agents']['training']['total_reward']:.2f}")

    print("\n" + "=" * 70)
    print("DEMO COMPLETE")
    print("=" * 70)

    return triad


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    demo_doppelganger_triad()
