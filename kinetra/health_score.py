"""
Composite Health Score and Reward Shaping

Computes multi-factor health scores for trading strategies and
provides reward signals for RL training.

NO FIXED THRESHOLDS - All values are normalized relative to
rolling distributions.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class TradeReward:
    """Reward components for a single trade."""
    pnl_reward: float           # Raw PnL contribution
    risk_adjusted: float        # Risk-adjusted component
    duration_penalty: float     # Duration efficiency
    regime_alignment: float     # Physics regime alignment
    total: float                # Final composite reward


@dataclass
class HealthScore:
    """Multi-factor health score for a trading run."""
    profitability: float        # Return-based score
    risk_efficiency: float      # Risk-adjusted metrics
    consistency: float          # Win rate, streak stability
    physics_alignment: float    # Regime/physics correctness
    composite: float            # Final weighted score
    components: Dict            # All sub-components


class RewardShaper:
    """
    Asymmetric Reward Function for RL Training.

    R = (PnL_net × α) - (σ × λ) - (Duration × δ) + (PhysicsBonus × φ)

    Where:
    - α = profit scaling (asymmetric: losses hurt more)
    - λ = volatility penalty
    - δ = duration decay
    - φ = physics alignment bonus

    All coefficients are ADAPTIVE based on rolling statistics.
    """

    def __init__(
        self,
        alpha_win: float = 1.0,
        alpha_loss: float = 2.0,      # Losses hurt 2x (asymmetric)
        lambda_vol: float = 0.5,      # Volatility penalty
        delta_duration: float = 0.01,  # Duration decay per bar
        phi_physics: float = 0.3,     # Physics bonus weight
    ):
        self.alpha_win = alpha_win
        self.alpha_loss = alpha_loss
        self.lambda_vol = lambda_vol
        self.delta_duration = delta_duration
        self.phi_physics = phi_physics

        # Rolling stats for normalization
        self.returns_history = []
        self.duration_history = []

    def compute_trade_reward(
        self,
        pnl_pct: float,
        duration_bars: int,
        volatility: float,
        physics_alignment: float = 0.0,  # -1 to 1 (wrong to right regime)
        entry_energy: float = 0.0,
        exit_energy: float = 0.0,
    ) -> TradeReward:
        """
        Compute reward for a single trade.

        Args:
            pnl_pct: Return percentage
            duration_bars: Trade duration in bars
            volatility: Market volatility during trade
            physics_alignment: How well trade aligned with physics regime
            entry_energy: Energy at entry
            exit_energy: Energy at exit

        Returns:
            TradeReward with component breakdown
        """
        # 1. Asymmetric PnL reward
        if pnl_pct >= 0:
            pnl_reward = pnl_pct * self.alpha_win
        else:
            pnl_reward = pnl_pct * self.alpha_loss  # More negative

        # 2. Risk-adjusted: normalize by volatility
        if volatility > 0:
            risk_adjusted = pnl_pct / volatility
        else:
            risk_adjusted = pnl_pct

        # 3. Duration penalty: exponential decay
        # Encourage efficient trades, penalize holding too long
        duration_penalty = -self.delta_duration * duration_bars

        # But bonus for profitable long holds
        if pnl_pct > 0 and duration_bars > 10:
            # Trend riding bonus
            duration_penalty += 0.005 * duration_bars

        # 4. Physics alignment bonus
        # +1 if entered in correct regime and exited well
        # -1 if counter-regime entry
        regime_bonus = physics_alignment * self.phi_physics

        # Energy capture bonus: reward capturing energy moves
        if entry_energy > 0:
            energy_capture = (exit_energy - entry_energy) / entry_energy
            energy_bonus = max(-0.2, min(0.2, energy_capture * 0.1))
        else:
            energy_bonus = 0.0

        # Total reward
        total = (
            pnl_reward
            - (volatility * self.lambda_vol)
            + duration_penalty
            + regime_bonus
            + energy_bonus
        )

        # Update history
        self.returns_history.append(pnl_pct)
        self.duration_history.append(duration_bars)

        return TradeReward(
            pnl_reward=pnl_reward,
            risk_adjusted=risk_adjusted,
            duration_penalty=duration_penalty,
            regime_alignment=regime_bonus,
            total=total
        )

    def compute_episode_reward(
        self,
        trades: List[Dict],
        final_equity: float,
        initial_equity: float,
        max_drawdown_pct: float
    ) -> float:
        """
        Compute reward for entire episode (run).

        Combines individual trade rewards with overall performance.
        """
        if not trades:
            return -1.0  # Penalty for no trades

        # Sum of trade rewards
        trade_rewards = []
        for trade in trades:
            reward = self.compute_trade_reward(
                pnl_pct=trade.get('return_pct', 0),
                duration_bars=trade.get('duration_bars', 1),
                volatility=trade.get('entry_volatility', 0.01),
                physics_alignment=trade.get('physics_alignment', 0),
                entry_energy=trade.get('entry_fractal_dim', 0),
                exit_energy=trade.get('exit_fractal_dim', 0)
            )
            trade_rewards.append(reward.total)

        avg_trade_reward = np.mean(trade_rewards)

        # Overall performance
        total_return = (final_equity - initial_equity) / initial_equity * 100

        # Drawdown penalty (severe for large drawdowns)
        dd_penalty = -abs(max_drawdown_pct) * 0.1

        # Combine
        episode_reward = (
            total_return * 0.5 +          # 50% weight on total return
            avg_trade_reward * 10.0 +     # Scaled trade rewards
            dd_penalty                     # Drawdown hurts
        )

        return episode_reward


class CompositeHealthScore:
    """
    Multi-factor health score for strategy evaluation.

    Components:
    1. Profitability (Omega, Return, PF)
    2. Risk Efficiency (Sortino, Ulcer, Calmar)
    3. Consistency (Win Rate, Streaks, Variance)
    4. Physics Alignment (Regime match, Energy capture)

    All scores normalized to 0-100 range using ADAPTIVE percentiles.
    """

    def __init__(self, historical_metrics: pd.DataFrame = None):
        """
        Initialize with optional historical data for normalization.

        Args:
            historical_metrics: DataFrame of past run metrics for percentile normalization
        """
        self.historical_metrics = historical_metrics
        self.percentile_cache = {}

        if historical_metrics is not None:
            self._compute_percentile_cache()

    def _compute_percentile_cache(self):
        """Compute percentile distributions from historical data."""
        if self.historical_metrics is None or len(self.historical_metrics) == 0:
            return

        metrics_to_cache = [
            'total_return_pct', 'omega_ratio', 'sortino_ratio',
            'ulcer_index', 'calmar_ratio', 'win_rate', 'profit_factor'
        ]

        for metric in metrics_to_cache:
            if metric in self.historical_metrics.columns:
                values = self.historical_metrics[metric].dropna()
                if len(values) > 0:
                    self.percentile_cache[metric] = {
                        'min': values.min(),
                        'max': values.max(),
                        'mean': values.mean(),
                        'std': values.std(),
                        'p25': values.quantile(0.25),
                        'p50': values.quantile(0.50),
                        'p75': values.quantile(0.75),
                    }

    def _normalize_to_score(
        self,
        value: float,
        metric: str,
        higher_is_better: bool = True
    ) -> float:
        """
        Normalize a metric value to 0-100 score.

        Uses historical percentiles if available, otherwise uses
        reasonable defaults based on metric type.
        """
        if metric in self.percentile_cache:
            cache = self.percentile_cache[metric]

            # Use z-score normalized to 0-100
            if cache['std'] > 0:
                z = (value - cache['mean']) / cache['std']
            else:
                z = 0

            # Convert z-score to 0-100 (z=-2 -> 0, z=0 -> 50, z=2 -> 100)
            score = 50 + z * 25

        else:
            # Default normalization based on metric type
            defaults = {
                'total_return_pct': (-50, 100),      # -50% to +100%
                'omega_ratio': (0.5, 3.0),           # 0.5 to 3.0
                'sortino_ratio': (-1, 3),            # -1 to 3
                'ulcer_index': (20, 0),              # 20 (bad) to 0 (good) - inverted
                'calmar_ratio': (-1, 5),             # -1 to 5
                'win_rate': (30, 70),                # 30% to 70%
                'profit_factor': (0.5, 2.5),         # 0.5 to 2.5
            }

            if metric in defaults:
                low, high = defaults[metric]
                if metric == 'ulcer_index':
                    # Invert - lower is better
                    score = 100 * (1 - (value - high) / (low - high))
                else:
                    score = 100 * (value - low) / (high - low)
            else:
                # Generic normalization
                score = 50 + value * 10

        # Clamp to 0-100
        return max(0, min(100, score))

    def compute_score(self, metrics: Dict) -> HealthScore:
        """
        Compute composite health score from run metrics.

        Args:
            metrics: Dictionary of run metrics (from compute_run_metrics)

        Returns:
            HealthScore with component breakdown
        """
        components = {}

        # 1. PROFITABILITY SCORE (0-100)
        # - Omega ratio (primary - handles fat tails)
        # - Total return
        # - Profit factor
        omega_score = self._normalize_to_score(
            metrics.get('omega_ratio', 1.0), 'omega_ratio')
        return_score = self._normalize_to_score(
            metrics.get('total_return_pct', 0), 'total_return_pct')
        pf_score = self._normalize_to_score(
            metrics.get('profit_factor', 1.0), 'profit_factor')

        profitability = (omega_score * 0.5 + return_score * 0.3 + pf_score * 0.2)
        components['omega_score'] = omega_score
        components['return_score'] = return_score
        components['profit_factor_score'] = pf_score

        # 2. RISK EFFICIENCY SCORE (0-100)
        # - Sortino (downside focus)
        # - Ulcer Index (pain)
        # - Calmar (return/drawdown)
        sortino_score = self._normalize_to_score(
            metrics.get('sortino_ratio', 0), 'sortino_ratio')
        ulcer_score = self._normalize_to_score(
            metrics.get('ulcer_index', 10), 'ulcer_index')
        calmar_score = self._normalize_to_score(
            metrics.get('calmar_ratio', 0), 'calmar_ratio')

        risk_efficiency = (sortino_score * 0.4 + ulcer_score * 0.35 + calmar_score * 0.25)
        components['sortino_score'] = sortino_score
        components['ulcer_score'] = ulcer_score
        components['calmar_score'] = calmar_score

        # 3. CONSISTENCY SCORE (0-100)
        # - Win rate
        # - Win/Loss streak ratio
        # - Return variance (lower is better)
        win_rate_score = self._normalize_to_score(
            metrics.get('win_rate', 50), 'win_rate')

        max_win = metrics.get('max_win_streak', 1)
        max_loss = metrics.get('max_loss_streak', 1)
        streak_ratio = max_win / (max_loss + 1)
        streak_score = min(100, streak_ratio * 25 + 25)  # 1:1 = 50, 3:1 = 100

        return_std = metrics.get('std_entry_volatility', 0.02)
        variance_score = max(0, 100 - return_std * 1000)  # Lower std = higher score

        consistency = (win_rate_score * 0.4 + streak_score * 0.3 + variance_score * 0.3)
        components['win_rate_score'] = win_rate_score
        components['streak_score'] = streak_score
        components['variance_score'] = variance_score

        # 4. PHYSICS ALIGNMENT SCORE (0-100)
        # Compare winning vs losing trade physics
        # Higher score = physics correctly predicting outcomes

        physics_score = 50  # Default neutral

        # Check if winning trades have different physics than losing trades
        win_fd = metrics.get('win_avg_entry_fractal_dim', 1.4)
        loss_fd = metrics.get('loss_avg_entry_fractal_dim', 1.4)

        win_symc = metrics.get('win_avg_entry_symc', 1.0)
        loss_symc = metrics.get('loss_avg_entry_symc', 1.0)

        win_vpin = metrics.get('win_avg_entry_vpin', 0.5)
        loss_vpin = metrics.get('loss_avg_entry_vpin', 0.5)

        # If winning trades have lower FD (trending), that's good
        if win_fd < loss_fd:
            physics_score += 15
        elif win_fd > loss_fd:
            physics_score -= 10

        # If winning trades have lower VPIN (less toxic), that's good
        if win_vpin < loss_vpin:
            physics_score += 15
        elif win_vpin > loss_vpin:
            physics_score -= 10

        # SymC differentiation
        symc_diff = abs(win_symc - loss_symc)
        physics_score += min(20, symc_diff * 20)  # Reward separation

        physics_score = max(0, min(100, physics_score))
        components['physics_score'] = physics_score

        # COMPOSITE SCORE
        # Weighted combination
        composite = (
            profitability * 0.35 +
            risk_efficiency * 0.30 +
            consistency * 0.20 +
            physics_score * 0.15
        )

        return HealthScore(
            profitability=profitability,
            risk_efficiency=risk_efficiency,
            consistency=consistency,
            physics_alignment=physics_score,
            composite=composite,
            components=components
        )


def compute_reward_from_trade(
    trade: Dict,
    reward_shaper: RewardShaper = None
) -> float:
    """
    Convenience function to compute reward from a trade dictionary.

    Args:
        trade: Trade dictionary from backtest
        reward_shaper: Optional RewardShaper instance

    Returns:
        Reward value for RL training
    """
    if reward_shaper is None:
        reward_shaper = RewardShaper()

    reward = reward_shaper.compute_trade_reward(
        pnl_pct=trade.get('return_pct', 0),
        duration_bars=trade.get('duration_bars', 1),
        volatility=trade.get('entry_volatility', 0.01),
        physics_alignment=0,  # Will be set by strategy
        entry_energy=trade.get('entry_fractal_dim', 0),
        exit_energy=trade.get('exit_fractal_dim', 0)
    )

    return reward.total


def compute_health_from_metrics(
    metrics: Dict,
    historical: pd.DataFrame = None
) -> float:
    """
    Convenience function to compute health score from metrics.

    Args:
        metrics: Run metrics dictionary
        historical: Optional historical data for normalization

    Returns:
        Composite health score (0-100)
    """
    scorer = CompositeHealthScore(historical)
    health = scorer.compute_score(metrics)
    return health.composite
