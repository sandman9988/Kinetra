#!/usr/bin/env python3
"""
Portfolio Health Monitoring System

Implements 4-pillar health scoring with automated re-exploration triggers:
1. Return & Efficiency (25%): CAGR, Sharpe, Omega, Calmar
2. Downside Risk (30%): Max DD, Ulcer Index, Recovery Time
3. Structural Stability (25%): Avg correlation, Eigenvalue crowding
4. Behavioral Health (20%): Edge decay, promotion frequency

Health scores trigger adaptive actions:
- >80: Normal operation
- 60-80: Reduce risk 30%, increase monitoring
- 40-60: Retrain underperformers, add hedges
- <40: Go flat, emergency retraining

Designed for live trading first, backtest compatibility second.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# =============================================================================
# HEALTH STATES AND ACTIONS
# =============================================================================

class HealthState(Enum):
    """Portfolio health state based on composite score."""
    CRITICAL = auto()    # <40: Emergency mode
    DEGRADED = auto()    # 40-60: Degraded performance
    WARNING = auto()     # 60-80: Warning state
    HEALTHY = auto()     # >80: Normal operation


@dataclass
class HealthAction:
    """Recommended action based on health state."""
    state: HealthState
    risk_multiplier: float  # Multiplier for position sizing
    message: str
    requires_retraining: bool = False
    requires_hedge: bool = False
    go_flat: bool = False


# Action mappings
HEALTH_ACTIONS = {
    HealthState.HEALTHY: HealthAction(
        state=HealthState.HEALTHY,
        risk_multiplier=1.0,
        message="Normal operation",
    ),
    HealthState.WARNING: HealthAction(
        state=HealthState.WARNING,
        risk_multiplier=0.7,  # Reduce risk 30%
        message="Reduce risk 30%, increase monitoring",
    ),
    HealthState.DEGRADED: HealthAction(
        state=HealthState.DEGRADED,
        risk_multiplier=0.5,  # Reduce risk 50%
        message="Retrain underperformers, add hedges",
        requires_retraining=True,
        requires_hedge=True,
    ),
    HealthState.CRITICAL: HealthAction(
        state=HealthState.CRITICAL,
        risk_multiplier=0.0,  # No new positions
        message="Go flat, emergency retraining",
        requires_retraining=True,
        go_flat=True,
    ),
}


# =============================================================================
# PILLAR SCORES
# =============================================================================

@dataclass
class PillarScore:
    """Score for a single health pillar."""
    name: str
    score: float  # 0-100
    weight: float  # Weight in composite score
    metrics: Dict[str, float] = field(default_factory=dict)
    last_updated: Optional[datetime] = None


@dataclass
class CompositeHealthScore:
    """Complete portfolio health assessment."""
    timestamp: datetime
    composite_score: float  # 0-100
    state: HealthState
    action: HealthAction

    # Individual pillars
    return_efficiency: PillarScore
    downside_risk: PillarScore
    structural_stability: PillarScore
    behavioral_health: PillarScore

    # Event tracking
    events: List[str] = field(default_factory=list)


# =============================================================================
# HEALTH CALCULATOR
# =============================================================================

class PortfolioHealthMonitor:
    """
    4-Pillar Portfolio Health Monitoring System.

    Tracks portfolio health across four dimensions and triggers
    automated actions when health degrades.

    Usage:
        monitor = PortfolioHealthMonitor(lookback_days=30)

        # Update with recent trades and equity
        monitor.update(
            trades=trades_list,
            equity_curve=equity_series,
            correlations=correlation_matrix,
        )

        # Get current health
        health = monitor.get_health_score()

        if health.state == HealthState.CRITICAL:
            # Go flat, trigger emergency retraining
            pass
    """

    def __init__(
        self,
        lookback_days: int = 30,
        min_trades_for_score: int = 20,
        logger: Optional[logging.Logger] = None,
    ):
        self.lookback_days = lookback_days
        self.min_trades_for_score = min_trades_for_score
        self.logger = logger or logging.getLogger("portfolio_health")

        # History tracking
        self.health_history: List[CompositeHealthScore] = []
        self.last_health: Optional[CompositeHealthScore] = None

        # Pillar weights (must sum to 1.0)
        self.pillar_weights = {
            'return_efficiency': 0.25,
            'downside_risk': 0.30,
            'structural_stability': 0.25,
            'behavioral_health': 0.20,
        }

    def update(
        self,
        trades: List[Dict],
        equity_curve: pd.Series,
        correlations: Optional[np.ndarray] = None,
        agent_promotions: int = 0,
    ) -> CompositeHealthScore:
        """
        Update health score with latest data.

        Args:
            trades: List of trade dictionaries with keys:
                - pnl, entry_time, exit_time, mfe, mae, etc.
            equity_curve: Pandas Series with timestamp index and equity values
            correlations: Optional correlation matrix between positions
            agent_promotions: Number of agent promotions in period

        Returns:
            CompositeHealthScore with all metrics
        """
        timestamp = datetime.now()

        # Calculate each pillar
        return_eff = self._calculate_return_efficiency(trades, equity_curve)
        downside = self._calculate_downside_risk(equity_curve)
        structural = self._calculate_structural_stability(trades, correlations)
        behavioral = self._calculate_behavioral_health(trades, agent_promotions)

        # Composite score (weighted average)
        composite = (
            return_eff.score * return_eff.weight +
            downside.score * downside.weight +
            structural.score * structural.weight +
            behavioral.score * behavioral.weight
        )

        # Determine state and action
        state = self._get_health_state(composite)
        action = HEALTH_ACTIONS[state]

        # Detect events
        events = self._detect_events(composite, state)

        # Create health score
        health = CompositeHealthScore(
            timestamp=timestamp,
            composite_score=composite,
            state=state,
            action=action,
            return_efficiency=return_eff,
            downside_risk=downside,
            structural_stability=structural,
            behavioral_health=behavioral,
            events=events,
        )

        # Log significant changes
        if self.last_health and state != self.last_health.state:
            self.logger.warning(
                f"Health state changed: {self.last_health.state.name} → {state.name} "
                f"(score: {composite:.1f})"
            )

        # Store history
        self.health_history.append(health)
        self.last_health = health

        return health

    def _calculate_return_efficiency(
        self,
        trades: List[Dict],
        equity_curve: pd.Series,
    ) -> PillarScore:
        """Calculate Return & Efficiency pillar (CAGR, Sharpe, Omega, Calmar)."""
        if len(trades) < self.min_trades_for_score:
            return PillarScore(
                name="Return & Efficiency",
                score=50.0,  # Neutral score
                weight=self.pillar_weights['return_efficiency'],
                metrics={'status': 'insufficient_data'},
            )

        # Calculate returns
        returns = equity_curve.pct_change().dropna()

        # CAGR (annualized)
        days = (equity_curve.index[-1] - equity_curve.index[0]).days
        if days > 0:
            total_return = equity_curve.iloc[-1] / equity_curve.iloc[0] - 1
            cagr = (1 + total_return) ** (365.0 / days) - 1
        else:
            cagr = 0.0

        # Sharpe ratio (annualized)
        if len(returns) > 1 and returns.std() > 0:
            sharpe = (returns.mean() / returns.std()) * np.sqrt(252)
        else:
            sharpe = 0.0

        # Omega ratio (gain/loss ratio above 0)
        gains = returns[returns > 0].sum()
        losses = abs(returns[returns < 0].sum())
        omega = gains / losses if losses > 0 else 0.0

        # Calmar ratio (CAGR / max DD)
        max_dd = self._calculate_max_drawdown(equity_curve)
        calmar = abs(cagr / max_dd) if max_dd < 0 else 0.0

        # Score (0-100): weighted combination
        # Sharpe >2 = excellent, Omega >2 = good, Calmar >3 = excellent
        sharpe_score = min(100, max(0, (sharpe / 2.0) * 50 + 50))
        omega_score = min(100, max(0, (omega / 2.0) * 50 + 50))
        calmar_score = min(100, max(0, (calmar / 3.0) * 50 + 50))
        cagr_score = min(100, max(0, (cagr / 0.5) * 50 + 50))  # 50% CAGR = excellent

        score = (sharpe_score * 0.3 + omega_score * 0.2 +
                 calmar_score * 0.3 + cagr_score * 0.2)

        return PillarScore(
            name="Return & Efficiency",
            score=score,
            weight=self.pillar_weights['return_efficiency'],
            metrics={
                'cagr': cagr,
                'sharpe': sharpe,
                'omega': omega,
                'calmar': calmar,
            },
            last_updated=datetime.now(),
        )

    def _calculate_downside_risk(
        self,
        equity_curve: pd.Series,
    ) -> PillarScore:
        """Calculate Downside Risk pillar (Max DD, Ulcer Index, Recovery Time)."""
        if len(equity_curve) < 2:
            return PillarScore(
                name="Downside Risk",
                score=50.0,
                weight=self.pillar_weights['downside_risk'],
                metrics={'status': 'insufficient_data'},
            )

        # Max drawdown (as percentage)
        max_dd = self._calculate_max_drawdown(equity_curve)
        max_dd_pct = abs(max_dd) * 100

        # Ulcer Index (measure of drawdown severity)
        ulcer = self._calculate_ulcer_index(equity_curve)

        # Recovery time (days to recover from max DD)
        recovery_days = self._calculate_recovery_time(equity_curve)

        # Score (0-100): lower risk = higher score
        # Max DD <10% = excellent, <20% = good, >30% = poor
        dd_score = min(100, max(0, 100 - max_dd_pct * 3))

        # Ulcer <5 = excellent, <10 = good, >20 = poor
        ulcer_score = min(100, max(0, 100 - ulcer * 5))

        # Recovery <30 days = excellent, <60 = good, >90 = poor
        recovery_score = min(100, max(0, 100 - (recovery_days / 90) * 100))

        score = dd_score * 0.4 + ulcer_score * 0.3 + recovery_score * 0.3

        return PillarScore(
            name="Downside Risk",
            score=score,
            weight=self.pillar_weights['downside_risk'],
            metrics={
                'max_drawdown_pct': max_dd_pct,
                'ulcer_index': ulcer,
                'recovery_days': recovery_days,
            },
            last_updated=datetime.now(),
        )

    def _calculate_structural_stability(
        self,
        trades: List[Dict],
        correlations: Optional[np.ndarray] = None,
    ) -> PillarScore:
        """Calculate Structural Stability pillar (Avg correlation, Eigenvalue crowding)."""
        if len(trades) < self.min_trades_for_score:
            return PillarScore(
                name="Structural Stability",
                score=50.0,
                weight=self.pillar_weights['structural_stability'],
                metrics={'status': 'insufficient_data'},
            )

        # Average correlation (between positions)
        if correlations is not None and correlations.size > 1:
            # Get upper triangle (excluding diagonal)
            mask = np.triu(np.ones_like(correlations, dtype=bool), k=1)
            avg_corr = abs(correlations[mask].mean()) if mask.any() else 0.0

            # Eigenvalue crowding (measure of concentration risk)
            eigenvalues = np.linalg.eigvalsh(correlations)
            eigenvalues = eigenvalues[eigenvalues > 0]  # Only positive
            if len(eigenvalues) > 1:
                max_eigen = eigenvalues.max()
                crowding = max_eigen / eigenvalues.sum()  # Concentration in largest
            else:
                crowding = 1.0
        else:
            avg_corr = 0.0
            crowding = 0.5  # Neutral

        # Score (0-100): lower correlation = higher score
        # Avg corr <0.3 = excellent, <0.5 = good, >0.7 = poor
        corr_score = min(100, max(0, 100 - avg_corr * 150))

        # Eigenvalue crowding <0.5 = well-diversified, >0.8 = concentrated
        crowding_score = min(100, max(0, 100 - crowding * 125))

        score = corr_score * 0.6 + crowding_score * 0.4

        return PillarScore(
            name="Structural Stability",
            score=score,
            weight=self.pillar_weights['structural_stability'],
            metrics={
                'avg_correlation': avg_corr,
                'eigenvalue_crowding': crowding,
            },
            last_updated=datetime.now(),
        )

    def _calculate_behavioral_health(
        self,
        trades: List[Dict],
        agent_promotions: int,
    ) -> PillarScore:
        """Calculate Behavioral Health pillar (Edge decay, promotion frequency)."""
        if len(trades) < self.min_trades_for_score:
            return PillarScore(
                name="Behavioral Health",
                score=50.0,
                weight=self.pillar_weights['behavioral_health'],
                metrics={'status': 'insufficient_data'},
            )

        # Edge decay (compare recent vs historical performance)
        edge_decay = self._calculate_edge_decay(trades)

        # Promotion frequency (adaptive learning health)
        # Optimal: 1-2 promotions per month (indicates continuous improvement)
        days_in_period = self.lookback_days
        promotions_per_month = agent_promotions * (30.0 / days_in_period)

        # Score (0-100)
        # Edge decay: 0 = no decay (100), -50% = severe decay (0)
        edge_score = min(100, max(0, 100 + edge_decay * 2))

        # Promotion frequency: 1-2/month = optimal (100), 0 or >4 = poor
        if 1.0 <= promotions_per_month <= 2.0:
            promo_score = 100
        elif promotions_per_month < 1.0:
            promo_score = promotions_per_month * 100  # Linear below 1
        else:
            promo_score = max(0, 100 - (promotions_per_month - 2) * 25)

        score = edge_score * 0.7 + promo_score * 0.3

        return PillarScore(
            name="Behavioral Health",
            score=score,
            weight=self.pillar_weights['behavioral_health'],
            metrics={
                'edge_decay_pct': edge_decay,
                'promotions_per_month': promotions_per_month,
            },
            last_updated=datetime.now(),
        )

    def _calculate_max_drawdown(self, equity: pd.Series) -> float:
        """Calculate maximum drawdown as percentage."""
        peak = equity.expanding(min_periods=1).max()
        dd = (equity - peak) / peak
        return dd.min()

    def _calculate_ulcer_index(self, equity: pd.Series) -> float:
        """Calculate Ulcer Index (RMS of drawdown)."""
        peak = equity.expanding(min_periods=1).max()
        dd = ((equity - peak) / peak) * 100  # As percentage
        ulcer = np.sqrt((dd ** 2).mean())
        return ulcer

    def _calculate_recovery_time(self, equity: pd.Series) -> float:
        """Calculate recovery time from max drawdown (in days)."""
        peak = equity.expanding(min_periods=1).max()
        dd = equity - peak

        # Find max DD point
        max_dd_idx = dd.idxmin()

        # Find recovery point (where equity exceeds previous peak)
        recovery_equity = equity[max_dd_idx:]
        max_dd_peak = peak[max_dd_idx]

        recovery_idx = recovery_equity[recovery_equity >= max_dd_peak].index

        if len(recovery_idx) > 0:
            recovery_days = (recovery_idx[0] - max_dd_idx).days
        else:
            # Not yet recovered - use time since max DD
            recovery_days = (equity.index[-1] - max_dd_idx).days

        return max(0, recovery_days)

    def _calculate_edge_decay(self, trades: List[Dict]) -> float:
        """Calculate edge decay (recent vs historical performance)."""
        if len(trades) < 10:
            return 0.0

        # Split trades into halves
        mid = len(trades) // 2
        recent = trades[mid:]
        historical = trades[:mid]

        # Calculate average edge ratio (MFE efficiency)
        recent_edge = np.mean([t.get('edge_ratio', 0.5) for t in recent])
        historical_edge = np.mean([t.get('edge_ratio', 0.5) for t in historical])

        if historical_edge > 0:
            decay_pct = ((recent_edge - historical_edge) / historical_edge) * 100
        else:
            decay_pct = 0.0

        return decay_pct

    def _get_health_state(self, composite_score: float) -> HealthState:
        """Map composite score to health state."""
        if composite_score >= 80:
            return HealthState.HEALTHY
        elif composite_score >= 60:
            return HealthState.WARNING
        elif composite_score >= 40:
            return HealthState.DEGRADED
        else:
            return HealthState.CRITICAL

    def _detect_events(
        self,
        composite_score: float,
        state: HealthState,
    ) -> List[str]:
        """Detect significant health events."""
        events = []

        # State change
        if self.last_health and state != self.last_health.state:
            events.append(f"State change: {self.last_health.state.name} → {state.name}")

        # Score drop
        if self.last_health:
            score_delta = composite_score - self.last_health.composite_score
            if score_delta < -10:
                events.append(f"Score drop: {score_delta:.1f} points")

        # Critical threshold
        if composite_score < 40:
            events.append("Critical health threshold breached")

        return events

    def get_health_score(self) -> Optional[CompositeHealthScore]:
        """Get most recent health score."""
        return self.last_health

    def get_health_summary(self) -> Dict:
        """Get summary of current health."""
        if not self.last_health:
            return {'status': 'no_data'}

        h = self.last_health
        return {
            'timestamp': h.timestamp.isoformat(),
            'composite_score': h.composite_score,
            'state': h.state.name,
            'action': {
                'message': h.action.message,
                'risk_multiplier': h.action.risk_multiplier,
                'requires_retraining': h.action.requires_retraining,
                'go_flat': h.action.go_flat,
            },
            'pillars': {
                'return_efficiency': {
                    'score': h.return_efficiency.score,
                    'weight': h.return_efficiency.weight,
                    'metrics': h.return_efficiency.metrics,
                },
                'downside_risk': {
                    'score': h.downside_risk.score,
                    'weight': h.downside_risk.weight,
                    'metrics': h.downside_risk.metrics,
                },
                'structural_stability': {
                    'score': h.structural_stability.score,
                    'weight': h.structural_stability.weight,
                    'metrics': h.structural_stability.metrics,
                },
                'behavioral_health': {
                    'score': h.behavioral_health.score,
                    'weight': h.behavioral_health.weight,
                    'metrics': h.behavioral_health.metrics,
                },
            },
            'events': h.events,
        }


# =============================================================================
# DEMO
# =============================================================================

def demo_portfolio_health():
    """Demonstrate the portfolio health monitoring system."""
    print("=" * 70)
    print("PORTFOLIO HEALTH MONITORING - DEMO")
    print("=" * 70)

    # Create monitor
    monitor = PortfolioHealthMonitor(lookback_days=30, min_trades_for_score=10)

    # Simulate equity curve (with some drawdown)
    dates = pd.date_range('2024-01-01', periods=100, freq='D')
    equity_values = 10000 * (1 + np.cumsum(np.random.randn(100) * 0.01))
    equity_curve = pd.Series(equity_values, index=dates)

    # Simulate trades
    trades = []
    for i in range(50):
        trades.append({
            'pnl': np.random.randn() * 100,
            'entry_time': dates[i],
            'exit_time': dates[i + 1],
            'mfe': abs(np.random.randn() * 150),
            'mae': abs(np.random.randn() * 80),
            'edge_ratio': np.random.random(),
        })

    # Update health
    health = monitor.update(
        trades=trades,
        equity_curve=equity_curve,
        correlations=np.random.randn(5, 5) * 0.3 + np.eye(5),
        agent_promotions=2,
    )

    print(f"\n[Health Assessment]")
    print(f"  Composite Score: {health.composite_score:.1f}")
    print(f"  State: {health.state.name}")
    print(f"  Action: {health.action.message}")
    print(f"  Risk Multiplier: {health.action.risk_multiplier:.1%}")

    print(f"\n[Pillar Scores]")
    print(f"  Return & Efficiency: {health.return_efficiency.score:.1f}")
    print(f"    - CAGR: {health.return_efficiency.metrics.get('cagr', 0):.2%}")
    print(f"    - Sharpe: {health.return_efficiency.metrics.get('sharpe', 0):.2f}")
    print(f"  Downside Risk: {health.downside_risk.score:.1f}")
    print(f"    - Max DD: {health.downside_risk.metrics.get('max_drawdown_pct', 0):.2f}%")
    print(f"  Structural Stability: {health.structural_stability.score:.1f}")
    print(f"  Behavioral Health: {health.behavioral_health.score:.1f}")

    if health.events:
        print(f"\n[Events]")
        for event in health.events:
            print(f"  - {event}")

    print("\n" + "=" * 70)
    print("DEMO COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    demo_portfolio_health()
