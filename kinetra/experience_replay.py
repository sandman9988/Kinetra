"""
Experience Replay System for Continual Learning from Live Trading

This system enables the agent to improve continuously by learning from
real trading outcomes. Trades are retroactively labeled as "good" or "poor"
and used for prioritized experience replay.

Key Components:
1. TradeLogger: Records all state during live trading
2. TradeLabeler: Retroactively labels trades based on quality metrics
3. ExperienceAnalyzer: Finds patterns in labeled trades
4. PrioritizedReplayBuffer: Stores episodes with priority
5. Continual learning integration

Architecture:
    Live Trading → Logger → Labeler → Analyzer → Replay → RL Agent

This creates a self-improving system where real outcomes drive learning.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, NamedTuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque
import pickle
from pathlib import Path


class Experience(NamedTuple):
    """Single experience tuple (s, a, r, s', done)."""
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool
    timestamp: datetime
    metadata: Dict  # Market context, regime, etc.


@dataclass
class TradeEpisode:
    """Complete trade episode with metadata."""
    experiences: List[Experience]

    # Trade outcomes
    entry_time: datetime
    exit_time: datetime
    total_pnl: float
    total_return_pct: float

    # Quality metrics
    sharpe_ratio: float
    mfe: float
    mae: float
    mfe_efficiency: float
    mae_efficiency: float

    # Market context
    entry_regime: str
    avg_volatility: float
    avg_spread: float

    # Constraint violations
    constraint_violations: int
    freeze_violations: int

    # Label (assigned retroactively)
    label: Optional[str] = None  # "good", "poor", "neutral"
    priority: float = 1.0  # Sampling priority


class TradeQualityMetrics:
    """Defines what makes a trade "good" or "poor"."""

    @staticmethod
    def calculate_trade_quality(episode: TradeEpisode) -> Tuple[str, float]:
        """
        Calculate trade quality label and priority.

        Returns:
            (label, priority) where:
            - label: "good", "poor", "neutral"
            - priority: Higher = more important for learning
        """
        score = 0.0
        priority = 1.0

        # Criterion 1: Risk-adjusted return (Sharpe-like)
        if episode.sharpe_ratio > 2.0:
            score += 3
        elif episode.sharpe_ratio > 1.0:
            score += 1
        elif episode.sharpe_ratio < -1.0:
            score -= 3
            priority += 2.0  # High priority to learn from losses
        elif episode.sharpe_ratio < 0:
            score -= 1
            priority += 1.0

        # Criterion 2: Execution efficiency (MFE capture)
        if episode.mfe_efficiency > 0.8:
            score += 2  # Captured most of potential profit
        elif episode.mfe_efficiency < 0.3:
            score -= 2  # Left too much on table
            priority += 1.5

        # Criterion 3: Risk management (MAE control)
        if episode.mae_efficiency > 0.7:
            score += 1  # Limited drawdown well
        elif episode.mae_efficiency < 0.3:
            score -= 2  # Poor drawdown control
            priority += 2.0  # Critical to learn

        # Criterion 4: Constraint violations (CRITICAL)
        if episode.constraint_violations > 0:
            score -= 5  # Major penalty
            priority += 3.0  # Must learn to avoid

        # Criterion 5: Absolute return
        if episode.total_return_pct > 0.02:  # > 2% return
            score += 2
        elif episode.total_return_pct < -0.02:  # > 2% loss
            score -= 2
            priority += 1.5

        # Determine label
        if score >= 4:
            label = "good"
        elif score <= -4:
            label = "poor"
        else:
            label = "neutral"

        return label, priority


class TradeLogger:
    """
    Logs all state, actions, and outcomes during live trading.

    This runs continuously in the background, recording every step
    so we can retroactively analyze what happened.
    """

    def __init__(self, log_dir: str = "logs/trades"):
        """
        Initialize trade logger.

        Args:
            log_dir: Directory to store trade logs
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Current episode being recorded
        self.current_episode: Optional[List[Experience]] = None
        self.episode_start_time: Optional[datetime] = None

        # Completed episodes (not yet labeled)
        self.unlabeled_episodes: List[TradeEpisode] = []

    def start_trade(self):
        """Start logging a new trade episode."""
        self.current_episode = []
        self.episode_start_time = datetime.now()

    def log_step(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        metadata: Dict
    ):
        """
        Log a single step during trading.

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Episode done flag
            metadata: Market context (regime, spread, volatility, etc.)
        """
        if self.current_episode is None:
            self.start_trade()

        experience = Experience(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            done=done,
            timestamp=datetime.now(),
            metadata=metadata
        )

        self.current_episode.append(experience)

    def end_trade(
        self,
        total_pnl: float,
        total_return_pct: float,
        sharpe_ratio: float,
        mfe: float,
        mae: float,
        mfe_efficiency: float,
        mae_efficiency: float,
        entry_regime: str,
        avg_volatility: float,
        avg_spread: float,
        constraint_violations: int = 0,
        freeze_violations: int = 0,
    ) -> TradeEpisode:
        """
        End current trade and create episode.

        Returns:
            TradeEpisode ready for labeling
        """
        if self.current_episode is None or len(self.current_episode) == 0:
            raise ValueError("No trade in progress")

        episode = TradeEpisode(
            experiences=self.current_episode,
            entry_time=self.episode_start_time,
            exit_time=datetime.now(),
            total_pnl=total_pnl,
            total_return_pct=total_return_pct,
            sharpe_ratio=sharpe_ratio,
            mfe=mfe,
            mae=mae,
            mfe_efficiency=mfe_efficiency,
            mae_efficiency=mae_efficiency,
            entry_regime=entry_regime,
            avg_volatility=avg_volatility,
            avg_spread=avg_spread,
            constraint_violations=constraint_violations,
            freeze_violations=freeze_violations,
        )

        # Reset for next trade
        self.current_episode = None
        self.episode_start_time = None

        # Store for labeling
        self.unlabeled_episodes.append(episode)

        return episode

    def save_episode(self, episode: TradeEpisode):
        """Save episode to disk for persistence."""
        timestamp = episode.entry_time.strftime("%Y%m%d_%H%M%S")
        filename = self.log_dir / f"episode_{timestamp}.pkl"

        with open(filename, 'wb') as f:
            pickle.dump(episode, f)


class TradeLabeler:
    """
    Retroactively labels trades as "good" or "poor" based on outcomes.

    This runs after trades close, analyzing the full trade lifecycle
    to determine quality and assign priority for learning.
    """

    def __init__(self):
        self.quality_metrics = TradeQualityMetrics()

    def label_episode(self, episode: TradeEpisode) -> TradeEpisode:
        """
        Label a trade episode as good/poor and assign priority.

        Args:
            episode: Unlabeled trade episode

        Returns:
            Episode with label and priority assigned
        """
        label, priority = self.quality_metrics.calculate_trade_quality(episode)

        episode.label = label
        episode.priority = priority

        return episode

    def label_batch(self, episodes: List[TradeEpisode]) -> List[TradeEpisode]:
        """Label a batch of episodes."""
        return [self.label_episode(ep) for ep in episodes]


class ExperienceAnalyzer:
    """
    Analyzes patterns in labeled trades to detect:
    1. What conditions lead to poor trades
    2. What conditions lead to good trades
    3. Whether market regime has shifted
    4. Whether agent policy needs updating
    """

    def __init__(self, lookback_trades: int = 100):
        """
        Initialize experience analyzer.

        Args:
            lookback_trades: Number of recent trades to analyze
        """
        self.lookback_trades = lookback_trades
        self.trade_history: deque = deque(maxlen=lookback_trades)

    def add_episode(self, episode: TradeEpisode):
        """Add labeled episode to history."""
        self.trade_history.append(episode)

    def analyze_poor_trades(self) -> Dict:
        """
        Analyze what went wrong in poor trades.

        Returns:
            Dict with analysis of poor trade patterns
        """
        poor_trades = [ep for ep in self.trade_history if ep.label == "poor"]

        if len(poor_trades) == 0:
            return {"count": 0}

        analysis = {
            "count": len(poor_trades),
            "avg_return": np.mean([ep.total_return_pct for ep in poor_trades]),
            "avg_sharpe": np.mean([ep.sharpe_ratio for ep in poor_trades]),
            "avg_mfe_efficiency": np.mean([ep.mfe_efficiency for ep in poor_trades]),
            "constraint_violation_rate": sum(ep.constraint_violations > 0 for ep in poor_trades) / len(poor_trades),

            # Regime analysis
            "regime_breakdown": self._regime_breakdown(poor_trades),

            # Timing analysis
            "avg_volatility": np.mean([ep.avg_volatility for ep in poor_trades]),
            "avg_spread": np.mean([ep.avg_spread for ep in poor_trades]),
        }

        return analysis

    def analyze_good_trades(self) -> Dict:
        """Analyze what worked well in good trades."""
        good_trades = [ep for ep in self.trade_history if ep.label == "good"]

        if len(good_trades) == 0:
            return {"count": 0}

        analysis = {
            "count": len(good_trades),
            "avg_return": np.mean([ep.total_return_pct for ep in good_trades]),
            "avg_sharpe": np.mean([ep.sharpe_ratio for ep in good_trades]),
            "avg_mfe_efficiency": np.mean([ep.mfe_efficiency for ep in good_trades]),

            # Regime analysis
            "regime_breakdown": self._regime_breakdown(good_trades),

            # Timing analysis
            "avg_volatility": np.mean([ep.avg_volatility for ep in good_trades]),
            "avg_spread": np.mean([ep.avg_spread for ep in good_trades]),
        }

        return analysis

    def _regime_breakdown(self, episodes: List[TradeEpisode]) -> Dict:
        """Count trades by entry regime."""
        regimes = {}
        for ep in episodes:
            regime = ep.entry_regime
            regimes[regime] = regimes.get(regime, 0) + 1
        return regimes

    def detect_drift(self) -> Dict:
        """
        Detect if market dynamics have shifted.

        Returns:
            Dict with drift analysis and re-exploration recommendation
        """
        if len(self.trade_history) < 20:
            return {"drift_detected": False, "reason": "Insufficient data"}

        # Compare recent performance to historical
        recent_trades = list(self.trade_history)[-20:]
        historical_trades = list(self.trade_history)[:-20]

        if len(historical_trades) == 0:
            return {"drift_detected": False, "reason": "No historical baseline"}

        recent_win_rate = sum(ep.label == "good" for ep in recent_trades) / len(recent_trades)
        historical_win_rate = sum(ep.label == "good" for ep in historical_trades) / len(historical_trades)

        recent_sharpe = np.mean([ep.sharpe_ratio for ep in recent_trades])
        historical_sharpe = np.mean([ep.sharpe_ratio for ep in historical_trades])

        # Drift detection criteria
        win_rate_drop = historical_win_rate - recent_win_rate
        sharpe_drop = historical_sharpe - recent_sharpe

        drift_detected = False
        reasons = []

        if win_rate_drop > 0.2:  # Win rate dropped by >20%
            drift_detected = True
            reasons.append(f"Win rate dropped from {historical_win_rate:.2%} to {recent_win_rate:.2%}")

        if sharpe_drop > 1.0:  # Sharpe dropped by >1.0
            drift_detected = True
            reasons.append(f"Sharpe dropped from {historical_sharpe:.2f} to {recent_sharpe:.2f}")

        return {
            "drift_detected": drift_detected,
            "reasons": reasons,
            "recommend_reexploration": drift_detected,
            "recent_win_rate": recent_win_rate,
            "historical_win_rate": historical_win_rate,
            "recent_sharpe": recent_sharpe,
            "historical_sharpe": historical_sharpe,
        }

    def should_update_policy(self) -> bool:
        """
        Determine if agent policy should be updated.

        Returns:
            True if enough new experiences warrant policy update
        """
        if len(self.trade_history) < 10:
            return False

        # Check if we have enough poor trades to learn from
        poor_trades = sum(ep.label == "poor" for ep in self.trade_history)

        # Update if >30% of recent trades are poor
        if poor_trades / len(self.trade_history) > 0.3:
            return True

        return False


class PrioritizedReplayBuffer:
    """
    Stores trade episodes with priority-based sampling.

    Higher priority episodes (e.g., poor trades with constraint violations)
    are sampled more frequently to accelerate learning from mistakes.
    """

    def __init__(self, capacity: int = 10000, alpha: float = 0.6):
        """
        Initialize prioritized replay buffer.

        Args:
            capacity: Maximum number of episodes to store
            alpha: Priority exponent (0 = uniform, 1 = fully prioritized)
        """
        self.capacity = capacity
        self.alpha = alpha
        self.buffer: deque = deque(maxlen=capacity)
        self.priorities: deque = deque(maxlen=capacity)

    def add(self, episode: TradeEpisode):
        """Add episode with its priority."""
        self.buffer.append(episode)
        self.priorities.append(episode.priority)

    def sample(self, batch_size: int) -> List[TradeEpisode]:
        """
        Sample batch of episodes using prioritized sampling.

        Args:
            batch_size: Number of episodes to sample

        Returns:
            List of sampled episodes
        """
        if len(self.buffer) < batch_size:
            batch_size = len(self.buffer)

        # Convert priorities to probabilities
        priorities = np.array(self.priorities) ** self.alpha
        probs = priorities / priorities.sum()

        # Sample indices
        indices = np.random.choice(len(self.buffer), size=batch_size, p=probs, replace=False)

        # Return sampled episodes
        return [self.buffer[i] for i in indices]

    def sample_experiences(self, batch_size: int) -> List[Experience]:
        """
        Sample individual experiences (not full episodes).

        Useful for standard RL training (DQN, PPO, etc.)
        """
        episodes = self.sample(batch_size=min(batch_size // 10, len(self.buffer)))

        # Flatten episodes into individual experiences
        experiences = []
        for episode in episodes:
            experiences.extend(episode.experiences)

        # Random sample to exact batch size
        if len(experiences) > batch_size:
            indices = np.random.choice(len(experiences), size=batch_size, replace=False)
            experiences = [experiences[i] for i in indices]

        return experiences

    def get_label_distribution(self) -> Dict:
        """Get distribution of labels in buffer."""
        labels = [ep.label for ep in self.buffer if ep.label is not None]

        return {
            "good": labels.count("good"),
            "poor": labels.count("poor"),
            "neutral": labels.count("neutral"),
            "total": len(labels),
        }

    def save(self, filepath: str):
        """Save buffer to disk."""
        with open(filepath, 'wb') as f:
            pickle.dump((self.buffer, self.priorities), f)

    def load(self, filepath: str):
        """Load buffer from disk."""
        with open(filepath, 'rb') as f:
            self.buffer, self.priorities = pickle.load(f)


class ContinualLearningManager:
    """
    Orchestrates the full continual learning pipeline:
    1. Log trades during live trading
    2. Label trades retroactively
    3. Analyze patterns
    4. Update replay buffer
    5. Trigger policy updates when needed
    """

    def __init__(
        self,
        log_dir: str = "logs/trades",
        buffer_capacity: int = 10000,
        lookback_trades: int = 100,
    ):
        """
        Initialize continual learning manager.

        Args:
            log_dir: Directory for trade logs
            buffer_capacity: Max replay buffer size
            lookback_trades: Number of trades for drift detection
        """
        self.logger = TradeLogger(log_dir=log_dir)
        self.labeler = TradeLabeler()
        self.analyzer = ExperienceAnalyzer(lookback_trades=lookback_trades)
        self.replay_buffer = PrioritizedReplayBuffer(capacity=buffer_capacity)

    def on_trade_complete(
        self,
        total_pnl: float,
        total_return_pct: float,
        sharpe_ratio: float,
        mfe: float,
        mae: float,
        mfe_efficiency: float,
        mae_efficiency: float,
        entry_regime: str,
        avg_volatility: float,
        avg_spread: float,
        constraint_violations: int = 0,
        freeze_violations: int = 0,
    ):
        """
        Called when a trade completes in live trading.

        This triggers the full pipeline:
        1. End logging
        2. Label trade
        3. Add to analyzer
        4. Add to replay buffer
        5. Check for drift
        """
        # 1. End logging and create episode
        episode = self.logger.end_trade(
            total_pnl=total_pnl,
            total_return_pct=total_return_pct,
            sharpe_ratio=sharpe_ratio,
            mfe=mfe,
            mae=mae,
            mfe_efficiency=mfe_efficiency,
            mae_efficiency=mae_efficiency,
            entry_regime=entry_regime,
            avg_volatility=avg_volatility,
            avg_spread=avg_spread,
            constraint_violations=constraint_violations,
            freeze_violations=freeze_violations,
        )

        # 2. Label trade
        episode = self.labeler.label_episode(episode)

        # 3. Add to analyzer
        self.analyzer.add_episode(episode)

        # 4. Add to replay buffer
        self.replay_buffer.add(episode)

        # 5. Save episode to disk
        self.logger.save_episode(episode)

        return episode

    def get_training_batch(self, batch_size: int) -> List[Experience]:
        """Get prioritized batch for RL training."""
        return self.replay_buffer.sample_experiences(batch_size)

    def analyze_performance(self) -> Dict:
        """Get comprehensive performance analysis."""
        return {
            "poor_trades": self.analyzer.analyze_poor_trades(),
            "good_trades": self.analyzer.analyze_good_trades(),
            "drift_analysis": self.analyzer.detect_drift(),
            "buffer_stats": self.replay_buffer.get_label_distribution(),
            "should_update_policy": self.analyzer.should_update_policy(),
        }
