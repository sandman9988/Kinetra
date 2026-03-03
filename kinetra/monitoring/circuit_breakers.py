"""
Circuit Breakers — Non-Negotiable Safety Limits
================================================

Hard circuit breakers that sit **ABOVE** all RL agents in the Renko
Kinetra three-layer architecture.  These are deterministic, non-learned
safety limits that override any agent decision.

Design source: ``RENKO_KINETRA_DESIGN_SPEC.md §3.1``

Architecture
------------
::

    ┌──────────────────── HARD CIRCUIT BREAKERS ────────────────────────┐
    │  VPIN > extreme → flatten       DD > limit → reduce 50%          │
    │  Spread > 3× normal → halt      Correlation → 1 → cap            │
    │  ────────────────── NON-NEGOTIABLE, NOT LEARNED ─────────────────│
    └───────────────────────────┬──────────────────────────────────────┘
                                ▼
    ┌──────────────────── LAYER 3 — RISK RL AGENT ─────────────────────┐
    └───────────────────────────┬──────────────────────────────────────┘
                                ▼
    ┌──────────────────── LAYER 2 — ALLOCATION RL AGENT ───────────────┐
    └─────────────────────────────────────────────────────────────────┘

Breaker Types
-------------

1. **VPIN Breaker** — VPIN > extreme threshold → flatten all positions.
   Protects against informed-flow toxicity (adverse selection).

2. **Drawdown Breaker** — Portfolio drawdown > hard limit → reduce
   exposure by 50%.  If DD > critical → flatten.
   Protects against catastrophic loss.

3. **Spread Breaker** — Current spread > N× baseline → halt new entries.
   Protects against illiquid/wide-spread conditions.

4. **Correlation Breaker** — Entry concurrence exceeds threshold → cap
   cluster weights.  Protects against concentration risk.

Each breaker is implemented as a **pure function** that takes current
state and returns a ``BreakerResult``.  The ``CircuitBreakerManager``
aggregates all breakers and returns the most severe action.

Non-Negotiable Rules
--------------------
- These breakers **cannot be disabled** during live trading.
- They are **not learned** — thresholds are set by the operator.
- They **override** any RL agent decision (Layer 2 or Layer 3).
- The most severe action across all breakers always wins.
- All breaker triggers are **logged** with full context.
- Breaker state transitions produce alerts (configurable).

Usage
-----
::

    from kinetra.monitoring.circuit_breakers import (
        CircuitBreakerManager,
        CircuitBreakerConfig,
        PortfolioSnapshot,
    )

    manager = CircuitBreakerManager(config=CircuitBreakerConfig())

    snapshot = PortfolioSnapshot(
        portfolio_dd=-0.12,
        vpin_mean=0.45,
        vpin_max=0.78,
        spread_ratio_max=2.1,
        entry_concurrence=0.3,
        n_active_positions=5,
        n_instruments=10,
        cluster_weights={"fx_major": 0.4, "crypto": 0.3},
    )

    result = manager.evaluate(snapshot)
    if result.action != BreakerAction.NONE:
        print(f"BREAKER: {result.action.name} — {result.reason}")

See Also
--------
- ``kinetra.renko.vpin`` — VPIN computation (feeds VPIN breaker)
- ``kinetra.rl.risk_env`` — Layer 3 risk env (respects breaker output)
- ``kinetra.rl.portfolio_env`` — Layer 2 env (weights capped by breaker)
- ``kinetra.renko.portfolio`` — cluster weights, Herfindahl index
"""

from __future__ import annotations

import enum
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# Enums
# ══════════════════════════════════════════════════════════════════════════════


class BreakerAction(enum.IntEnum):
    """
    Action severity levels for circuit breakers.

    Ordered from least severe to most severe.  When multiple breakers
    trigger, the **most severe** action wins.

    Attributes
    ----------
    NONE : int
        No breaker active — normal operation.
    CAP_CLUSTER : int
        Cap cluster weights to configured maximum.
    HALT_NEW_ENTRIES : int
        Halt all new position entries; existing positions remain.
    REDUCE_EXPOSURE : int
        Reduce portfolio exposure by configured fraction (typically 50%).
    FLATTEN : int
        Close ALL positions immediately.  Most severe action.
    """

    NONE = 0
    CAP_CLUSTER = 1
    HALT_NEW_ENTRIES = 2
    REDUCE_EXPOSURE = 3
    FLATTEN = 4


class BreakerType(enum.Enum):
    """
    Identifies which circuit breaker triggered.

    Attributes
    ----------
    VPIN : str
        Volume-synchronized Probability of Informed Trading breaker.
    DRAWDOWN : str
        Portfolio drawdown breaker.
    SPREAD : str
        Spread regime breaker.
    CORRELATION : str
        Entry concurrence / correlation breaker.
    """

    VPIN = "vpin"
    DRAWDOWN = "drawdown"
    SPREAD = "spread"
    CORRELATION = "correlation"
    VPIN_KURTOSIS = "vpin_kurtosis"


# ══════════════════════════════════════════════════════════════════════════════
# Results
# ══════════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True, slots=True)
class BreakerResult:
    """
    Result from evaluating a single circuit breaker.

    Attributes
    ----------
    breaker_type : BreakerType
        Which breaker produced this result.
    action : BreakerAction
        The action to take (NONE if breaker is not triggered).
    triggered : bool
        True if the breaker threshold was exceeded.
    reason : str
        Human-readable description of why the breaker triggered
        (empty string if not triggered).
    current_value : float
        The current value of the metric being monitored.
    threshold : float
        The threshold that was (or was not) exceeded.
    severity : float
        How far past the threshold the metric is (0.0 if not triggered,
        > 0.0 if triggered).  Used for graduated responses.
    metadata : dict
        Additional context (breaker-specific).
    """

    breaker_type: BreakerType
    action: BreakerAction
    triggered: bool
    reason: str
    current_value: float
    threshold: float
    severity: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


# ══════════════════════════════════════════════════════════════════════════════
# Configuration
# ══════════════════════════════════════════════════════════════════════════════


@dataclass(slots=True)
class VPINBreakerConfig:
    """
    Configuration for the VPIN circuit breaker.

    Attributes
    ----------
    extreme_threshold : float
        VPIN level that triggers FLATTEN.  Should be set from the
        instrument's baseline p95 or p99 via ``vpin_baseline()``.
        Default 0.8 is a conservative starting point.
    elevated_threshold : float
        VPIN level that triggers HALT_NEW_ENTRIES.  Below extreme
        but above normal — stop opening new positions.
    enabled : bool
        If False, this breaker always returns NONE.
    use_max : bool
        If True, trigger on max VPIN across instruments.
        If False, trigger on mean VPIN.
    """

    extreme_threshold: float = 0.80
    elevated_threshold: float = 0.65
    enabled: bool = True
    use_max: bool = True


@dataclass(slots=True)
class VPINKurtosisBreakerConfig:
    """
    Configuration for VPIN kurtosis tail-risk breaker.

    Uses rolling excess kurtosis of VPIN values as a jump-risk detector.
    High kurtosis indicates toxic-flow tail clustering, where mean/max VPIN
    can still look benign while tail risk is already elevated.
    """

    elevated_threshold: float = 2.5
    extreme_threshold: float = 4.0
    enabled: bool = True


@dataclass(slots=True)
class DrawdownBreakerConfig:
    """
    Configuration for the drawdown circuit breaker.

    The drawdown breaker has two tiers:
      - **Reduce** tier: reduce exposure by ``reduction_fraction``
        (default 50%) when DD exceeds ``reduce_threshold``.
      - **Flatten** tier: close ALL positions when DD exceeds
        ``flatten_threshold``.

    Attributes
    ----------
    reduce_threshold : float
        Drawdown fraction (positive, e.g. 0.10 = 10%) that triggers
        exposure reduction.
    flatten_threshold : float
        Drawdown fraction that triggers full flatten.
    reduction_fraction : float
        How much to reduce exposure by (0.5 = halve all weights).
    enabled : bool
        If False, this breaker always returns NONE.
    """

    reduce_threshold: float = 0.10
    flatten_threshold: float = 0.20
    reduction_fraction: float = 0.50
    enabled: bool = True

    def __post_init__(self) -> None:
        if self.reduce_threshold > self.flatten_threshold:
            raise ValueError(
                f"reduce_threshold ({self.reduce_threshold}) must be <= "
                f"flatten_threshold ({self.flatten_threshold})"
            )


@dataclass(slots=True)
class SpreadBreakerConfig:
    """
    Configuration for the spread regime circuit breaker.

    Triggers when current spread exceeds a multiple of the baseline
    spread.  Prevents trading in illiquid conditions.

    Attributes
    ----------
    halt_multiplier : float
        Spread-to-baseline ratio that triggers HALT_NEW_ENTRIES.
        Default 3.0 = spread is 3× its normal level.
    flatten_multiplier : float
        Spread ratio that triggers FLATTEN (extreme illiquidity).
    enabled : bool
        If False, this breaker always returns NONE.
    """

    halt_multiplier: float = 3.0
    flatten_multiplier: float = 5.0
    enabled: bool = True

    def __post_init__(self) -> None:
        if self.halt_multiplier > self.flatten_multiplier:
            raise ValueError(
                f"halt_multiplier ({self.halt_multiplier}) must be <= "
                f"flatten_multiplier ({self.flatten_multiplier})"
            )


@dataclass(slots=True)
class CorrelationBreakerConfig:
    """
    Configuration for the correlation / entry concurrence breaker.

    Triggers when too many instruments are entering at the same time,
    indicating regime-wide risk (e.g., correlated crash).

    Attributes
    ----------
    concurrence_threshold : float
        Entry concurrence level (0–1) that triggers CAP_CLUSTER.
        0.7 means ≥70% of instruments entering within the same window.
    halt_threshold : float
        Concurrence level that triggers HALT_NEW_ENTRIES.
    max_cluster_weight : float
        Maximum weight for any single cluster when capping is active.
    enabled : bool
        If False, this breaker always returns NONE.
    """

    concurrence_threshold: float = 0.70
    halt_threshold: float = 0.90
    max_cluster_weight: float = 0.35
    enabled: bool = True


@dataclass(slots=True)
class CircuitBreakerConfig:
    """
    Top-level configuration aggregating all circuit breakers.

    Attributes
    ----------
    vpin : VPINBreakerConfig
        VPIN breaker configuration.
    drawdown : DrawdownBreakerConfig
        Drawdown breaker configuration.
    spread : SpreadBreakerConfig
        Spread breaker configuration.
    correlation : CorrelationBreakerConfig
        Correlation breaker configuration.
    cooldown_seconds : float
        Minimum seconds between successive breaker triggers of the
        same type.  Prevents oscillating on/off rapidly.
    log_triggers : bool
        If True, log all breaker triggers at WARNING level.
    """

    vpin: VPINBreakerConfig = field(default_factory=VPINBreakerConfig)
    vpin_kurtosis: VPINKurtosisBreakerConfig = field(default_factory=VPINKurtosisBreakerConfig)
    drawdown: DrawdownBreakerConfig = field(default_factory=DrawdownBreakerConfig)
    spread: SpreadBreakerConfig = field(default_factory=SpreadBreakerConfig)
    correlation: CorrelationBreakerConfig = field(default_factory=CorrelationBreakerConfig)
    cooldown_seconds: float = 60.0
    log_triggers: bool = True


# ══════════════════════════════════════════════════════════════════════════════
# Portfolio snapshot — input to the circuit breakers
# ══════════════════════════════════════════════════════════════════════════════


@dataclass(slots=True)
class PortfolioSnapshot:
    """
    Current portfolio state snapshot for circuit breaker evaluation.

    This is the minimal set of information the breaker system needs.
    It is intentionally separate from ``PortfolioDaySnapshot`` in the
    risk env — breakers run at higher frequency (per-tick or per-minute)
    and need fewer fields.

    Attributes
    ----------
    portfolio_dd : float
        Current portfolio drawdown as a negative fraction
        (e.g. -0.12 = 12% drawdown).  Must be ≤ 0.
    vpin_mean : float
        Mean VPIN across instruments.  Range [0, 1].
    vpin_max : float
        Max VPIN across instruments.  Range [0, 1].
    vpin_kurtosis : float
        Rolling excess kurtosis of VPIN.  0~Gaussian, >0 fat tails.
    spread_ratio_max : float
        Maximum current-spread / baseline-spread ratio across
        instruments.  1.0 = normal, 3.0 = 3× wider than baseline.
    entry_concurrence : float
        Fraction of instruments that entered within the same time
        window.  Range [0, 1].
    n_active_positions : int
        Number of instruments currently in a position.
    n_instruments : int
        Total number of instruments in the portfolio.
    cluster_weights : dict[str, float]
        Current allocation weight sum per cluster.
    timestamp : datetime or None
        When this snapshot was taken (for cooldown tracking).
    """

    portfolio_dd: float = 0.0
    vpin_mean: float = 0.0
    vpin_max: float = 0.0
    vpin_kurtosis: float = 0.0
    spread_ratio_max: float = 1.0
    entry_concurrence: float = 0.0
    n_active_positions: int = 0
    n_instruments: int = 1
    cluster_weights: Dict[str, float] = field(default_factory=dict)
    timestamp: Optional[datetime] = None


# ══════════════════════════════════════════════════════════════════════════════
# Breaker state
# ══════════════════════════════════════════════════════════════════════════════


@dataclass(slots=True)
class CircuitBreakerState:
    """
    Stateful tracking for the circuit breaker manager.

    Tracks which breakers are currently active, when they last
    triggered, and historical trigger counts.

    Attributes
    ----------
    active_breakers : dict[BreakerType, BreakerResult]
        Currently active breaker results (only triggered ones).
    last_trigger_time : dict[BreakerType, datetime]
        Timestamp of the most recent trigger per breaker type.
    trigger_counts : dict[BreakerType, int]
        Cumulative trigger count per breaker type.
    current_action : BreakerAction
        The most severe action currently in effect.
    history : list[tuple[datetime, BreakerResult]]
        Recent trigger history (bounded by ``max_history``).
    max_history : int
        Maximum history entries to retain.
    """

    active_breakers: Dict[BreakerType, BreakerResult] = field(default_factory=dict)
    last_trigger_time: Dict[BreakerType, datetime] = field(default_factory=dict)
    trigger_counts: Dict[BreakerType, int] = field(default_factory=dict)
    current_action: BreakerAction = BreakerAction.NONE
    history: List[Tuple[datetime, BreakerResult]] = field(default_factory=list)
    max_history: int = 1000

    def record_trigger(
        self,
        result: BreakerResult,
        timestamp: Optional[datetime] = None,
    ) -> None:
        """Record a breaker trigger in state."""
        ts = timestamp or datetime.now(tz=timezone.utc)

        self.active_breakers[result.breaker_type] = result
        self.last_trigger_time[result.breaker_type] = ts
        self.trigger_counts[result.breaker_type] = (
            self.trigger_counts.get(result.breaker_type, 0) + 1
        )

        self.history.append((ts, result))
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history :]

    def clear_breaker(self, breaker_type: BreakerType) -> None:
        """Clear an active breaker (condition no longer met)."""
        self.active_breakers.pop(breaker_type, None)

    def update_current_action(self) -> BreakerAction:
        """Recompute the most severe currently active action."""
        if not self.active_breakers:
            self.current_action = BreakerAction.NONE
        else:
            self.current_action = max(r.action for r in self.active_breakers.values())
        return self.current_action

    def reset(self) -> None:
        """Reset all state (e.g. at start of a new session)."""
        self.active_breakers.clear()
        self.last_trigger_time.clear()
        self.trigger_counts.clear()
        self.current_action = BreakerAction.NONE
        self.history.clear()

    @property
    def is_any_active(self) -> bool:
        """True if any breaker is currently triggered."""
        return len(self.active_breakers) > 0

    @property
    def active_types(self) -> List[BreakerType]:
        """List of currently active breaker types."""
        return list(self.active_breakers.keys())

    def summary(self) -> Dict[str, Any]:
        """Return a summary dict for logging/diagnostics."""
        return {
            "current_action": self.current_action.name,
            "n_active": len(self.active_breakers),
            "active_types": [t.value for t in self.active_breakers],
            "trigger_counts": {t.value: c for t, c in self.trigger_counts.items()},
            "total_triggers": sum(self.trigger_counts.values()),
        }


# ══════════════════════════════════════════════════════════════════════════════
# Individual breaker functions (pure, stateless)
# ══════════════════════════════════════════════════════════════════════════════


def check_vpin_breaker(
    snapshot: PortfolioSnapshot,
    config: Optional[VPINBreakerConfig] = None,
) -> BreakerResult:
    """
    Evaluate the VPIN circuit breaker.

    VPIN > extreme threshold → FLATTEN (close all positions).
    VPIN > elevated threshold → HALT_NEW_ENTRIES.

    The VPIN breaker protects against **informed-flow toxicity**.
    When VPIN is extreme, market makers are being adversely selected,
    spreads are about to widen, and prices will move against us.

    Parameters
    ----------
    snapshot : PortfolioSnapshot
        Current portfolio state with VPIN readings.
    config : VPINBreakerConfig or None
        Breaker configuration.  Uses defaults if None.

    Returns
    -------
    BreakerResult
        Breaker evaluation result.
    """
    if config is None:
        config = VPINBreakerConfig()

    if not config.enabled:
        return BreakerResult(
            breaker_type=BreakerType.VPIN,
            action=BreakerAction.NONE,
            triggered=False,
            reason="",
            current_value=0.0,
            threshold=config.extreme_threshold,
        )

    # Select the metric: max or mean VPIN
    vpin_value = snapshot.vpin_max if config.use_max else snapshot.vpin_mean

    # Tier 1: FLATTEN on extreme VPIN
    if vpin_value > config.extreme_threshold:
        severity = (vpin_value - config.extreme_threshold) / max(
            1.0 - config.extreme_threshold, 1e-9
        )
        return BreakerResult(
            breaker_type=BreakerType.VPIN,
            action=BreakerAction.FLATTEN,
            triggered=True,
            reason=(
                f"VPIN {'max' if config.use_max else 'mean'} = {vpin_value:.4f} > "
                f"extreme threshold {config.extreme_threshold:.4f} — FLATTEN ALL"
            ),
            current_value=vpin_value,
            threshold=config.extreme_threshold,
            severity=float(np.clip(severity, 0.0, 10.0)),
            metadata={
                "vpin_mean": snapshot.vpin_mean,
                "vpin_max": snapshot.vpin_max,
                "tier": "extreme",
            },
        )

    # Tier 2: HALT on elevated VPIN
    if vpin_value > config.elevated_threshold:
        severity = (vpin_value - config.elevated_threshold) / max(
            config.extreme_threshold - config.elevated_threshold, 1e-9
        )
        return BreakerResult(
            breaker_type=BreakerType.VPIN,
            action=BreakerAction.HALT_NEW_ENTRIES,
            triggered=True,
            reason=(
                f"VPIN {'max' if config.use_max else 'mean'} = {vpin_value:.4f} > "
                f"elevated threshold {config.elevated_threshold:.4f} — HALT NEW ENTRIES"
            ),
            current_value=vpin_value,
            threshold=config.elevated_threshold,
            severity=float(np.clip(severity, 0.0, 10.0)),
            metadata={
                "vpin_mean": snapshot.vpin_mean,
                "vpin_max": snapshot.vpin_max,
                "tier": "elevated",
            },
        )

    # Not triggered
    return BreakerResult(
        breaker_type=BreakerType.VPIN,
        action=BreakerAction.NONE,
        triggered=False,
        reason="",
        current_value=vpin_value,
        threshold=config.elevated_threshold,
    )


def check_drawdown_breaker(
    snapshot: PortfolioSnapshot,
    config: Optional[DrawdownBreakerConfig] = None,
) -> BreakerResult:
    """
    Evaluate the drawdown circuit breaker.

    DD > flatten_threshold → FLATTEN.
    DD > reduce_threshold  → REDUCE_EXPOSURE (by configured fraction).

    The drawdown breaker is the **last line of defence** against
    catastrophic loss.  It is non-negotiable.

    Parameters
    ----------
    snapshot : PortfolioSnapshot
        Current portfolio state.  ``portfolio_dd`` should be ≤ 0
        (e.g. -0.12 = 12% drawdown).
    config : DrawdownBreakerConfig or None
        Breaker configuration.

    Returns
    -------
    BreakerResult
        Breaker evaluation result.
    """
    if config is None:
        config = DrawdownBreakerConfig()

    if not config.enabled:
        return BreakerResult(
            breaker_type=BreakerType.DRAWDOWN,
            action=BreakerAction.NONE,
            triggered=False,
            reason="",
            current_value=0.0,
            threshold=config.reduce_threshold,
        )

    # Convert DD to positive fraction for comparison
    # portfolio_dd is negative (e.g. -0.12), thresholds are positive (0.10)
    dd_abs = abs(snapshot.portfolio_dd)

    # Tier 1: FLATTEN on catastrophic drawdown
    if dd_abs > config.flatten_threshold:
        severity = (dd_abs - config.flatten_threshold) / max(1.0 - config.flatten_threshold, 1e-9)
        return BreakerResult(
            breaker_type=BreakerType.DRAWDOWN,
            action=BreakerAction.FLATTEN,
            triggered=True,
            reason=(
                f"Portfolio DD = {dd_abs:.2%} > flatten threshold "
                f"{config.flatten_threshold:.2%} — FLATTEN ALL"
            ),
            current_value=dd_abs,
            threshold=config.flatten_threshold,
            severity=float(np.clip(severity, 0.0, 10.0)),
            metadata={
                "portfolio_dd": snapshot.portfolio_dd,
                "tier": "flatten",
                "reduction_fraction": 1.0,
            },
        )

    # Tier 2: REDUCE on significant drawdown
    if dd_abs > config.reduce_threshold:
        severity = (dd_abs - config.reduce_threshold) / max(
            config.flatten_threshold - config.reduce_threshold, 1e-9
        )
        return BreakerResult(
            breaker_type=BreakerType.DRAWDOWN,
            action=BreakerAction.REDUCE_EXPOSURE,
            triggered=True,
            reason=(
                f"Portfolio DD = {dd_abs:.2%} > reduce threshold "
                f"{config.reduce_threshold:.2%} — REDUCE EXPOSURE by "
                f"{config.reduction_fraction:.0%}"
            ),
            current_value=dd_abs,
            threshold=config.reduce_threshold,
            severity=float(np.clip(severity, 0.0, 10.0)),
            metadata={
                "portfolio_dd": snapshot.portfolio_dd,
                "tier": "reduce",
                "reduction_fraction": config.reduction_fraction,
            },
        )

    # Not triggered
    return BreakerResult(
        breaker_type=BreakerType.DRAWDOWN,
        action=BreakerAction.NONE,
        triggered=False,
        reason="",
        current_value=dd_abs,
        threshold=config.reduce_threshold,
    )


def check_vpin_kurtosis_breaker(
    snapshot: PortfolioSnapshot,
    config: Optional[VPINKurtosisBreakerConfig] = None,
) -> BreakerResult:
    """
    Evaluate VPIN kurtosis circuit breaker.

    - kurtosis > extreme_threshold  -> FLATTEN
    - kurtosis > elevated_threshold -> HALT_NEW_ENTRIES
    """
    if config is None:
        config = VPINKurtosisBreakerConfig()

    if not config.enabled:
        return BreakerResult(
            breaker_type=BreakerType.VPIN_KURTOSIS,
            action=BreakerAction.NONE,
            triggered=False,
            reason="",
            current_value=0.0,
            threshold=config.elevated_threshold,
        )

    k = float(snapshot.vpin_kurtosis)
    if not np.isfinite(k):
        return BreakerResult(
            breaker_type=BreakerType.VPIN_KURTOSIS,
            action=BreakerAction.NONE,
            triggered=False,
            reason="",
            current_value=k,
            threshold=config.elevated_threshold,
        )

    if k > config.extreme_threshold:
        sev = (k - config.extreme_threshold) / max(abs(config.extreme_threshold), 1e-9)
        return BreakerResult(
            breaker_type=BreakerType.VPIN_KURTOSIS,
            action=BreakerAction.FLATTEN,
            triggered=True,
            reason=(
                f"VPIN kurtosis={k:.3f} > extreme threshold "
                f"{config.extreme_threshold:.3f} — FLATTEN ALL"
            ),
            current_value=k,
            threshold=config.extreme_threshold,
            severity=float(np.clip(sev, 0.0, 10.0)),
            metadata={"tier": "extreme", "vpin_kurtosis": k},
        )

    if k > config.elevated_threshold:
        sev = (k - config.elevated_threshold) / max(
            config.extreme_threshold - config.elevated_threshold, 1e-9
        )
        return BreakerResult(
            breaker_type=BreakerType.VPIN_KURTOSIS,
            action=BreakerAction.HALT_NEW_ENTRIES,
            triggered=True,
            reason=(
                f"VPIN kurtosis={k:.3f} > elevated threshold "
                f"{config.elevated_threshold:.3f} — HALT NEW ENTRIES"
            ),
            current_value=k,
            threshold=config.elevated_threshold,
            severity=float(np.clip(sev, 0.0, 10.0)),
            metadata={"tier": "elevated", "vpin_kurtosis": k},
        )

    return BreakerResult(
        breaker_type=BreakerType.VPIN_KURTOSIS,
        action=BreakerAction.NONE,
        triggered=False,
        reason="",
        current_value=k,
        threshold=config.elevated_threshold,
    )


def check_spread_breaker(
    snapshot: PortfolioSnapshot,
    config: Optional[SpreadBreakerConfig] = None,
) -> BreakerResult:
    """
    Evaluate the spread regime circuit breaker.

    Spread > flatten_multiplier × baseline → FLATTEN.
    Spread > halt_multiplier × baseline    → HALT_NEW_ENTRIES.

    The spread breaker protects against **illiquid conditions**.
    Wide spreads mean higher friction, worse fills, and potential
    slippage cascades.

    Parameters
    ----------
    snapshot : PortfolioSnapshot
        Current portfolio state.  ``spread_ratio_max`` is the maximum
        current-spread / baseline-spread ratio across instruments.
    config : SpreadBreakerConfig or None
        Breaker configuration.

    Returns
    -------
    BreakerResult
        Breaker evaluation result.
    """
    if config is None:
        config = SpreadBreakerConfig()

    if not config.enabled:
        return BreakerResult(
            breaker_type=BreakerType.SPREAD,
            action=BreakerAction.NONE,
            triggered=False,
            reason="",
            current_value=1.0,
            threshold=config.halt_multiplier,
        )

    ratio = snapshot.spread_ratio_max

    # Tier 1: FLATTEN on extreme spread widening
    if ratio > config.flatten_multiplier:
        severity = (ratio - config.flatten_multiplier) / max(config.flatten_multiplier, 1e-9)
        return BreakerResult(
            breaker_type=BreakerType.SPREAD,
            action=BreakerAction.FLATTEN,
            triggered=True,
            reason=(
                f"Spread ratio = {ratio:.2f}× baseline > flatten multiplier "
                f"{config.flatten_multiplier:.1f}× — FLATTEN ALL"
            ),
            current_value=ratio,
            threshold=config.flatten_multiplier,
            severity=float(np.clip(severity, 0.0, 10.0)),
            metadata={"tier": "flatten"},
        )

    # Tier 2: HALT on significantly wide spreads
    if ratio > config.halt_multiplier:
        severity = (ratio - config.halt_multiplier) / max(
            config.flatten_multiplier - config.halt_multiplier, 1e-9
        )
        return BreakerResult(
            breaker_type=BreakerType.SPREAD,
            action=BreakerAction.HALT_NEW_ENTRIES,
            triggered=True,
            reason=(
                f"Spread ratio = {ratio:.2f}× baseline > halt multiplier "
                f"{config.halt_multiplier:.1f}× — HALT NEW ENTRIES"
            ),
            current_value=ratio,
            threshold=config.halt_multiplier,
            severity=float(np.clip(severity, 0.0, 10.0)),
            metadata={"tier": "halt"},
        )

    # Not triggered
    return BreakerResult(
        breaker_type=BreakerType.SPREAD,
        action=BreakerAction.NONE,
        triggered=False,
        reason="",
        current_value=ratio,
        threshold=config.halt_multiplier,
    )


def check_correlation_breaker(
    snapshot: PortfolioSnapshot,
    config: Optional[CorrelationBreakerConfig] = None,
) -> BreakerResult:
    """
    Evaluate the correlation / entry concurrence breaker.

    Concurrence > halt_threshold → HALT_NEW_ENTRIES.
    Concurrence > concurrence_threshold → CAP_CLUSTER.

    The correlation breaker protects against **concentration risk**.
    When many instruments enter simultaneously, diversification is
    illusory and a single regime shift can cause correlated losses.

    Parameters
    ----------
    snapshot : PortfolioSnapshot
        Current portfolio state.  ``entry_concurrence`` is the fraction
        of instruments entering within the same time window [0, 1].
    config : CorrelationBreakerConfig or None
        Breaker configuration.

    Returns
    -------
    BreakerResult
        Breaker evaluation result.
    """
    if config is None:
        config = CorrelationBreakerConfig()

    if not config.enabled:
        return BreakerResult(
            breaker_type=BreakerType.CORRELATION,
            action=BreakerAction.NONE,
            triggered=False,
            reason="",
            current_value=0.0,
            threshold=config.concurrence_threshold,
        )

    concurrence = snapshot.entry_concurrence

    # Tier 1: HALT on extreme concurrence
    if concurrence > config.halt_threshold:
        severity = (concurrence - config.halt_threshold) / max(1.0 - config.halt_threshold, 1e-9)
        return BreakerResult(
            breaker_type=BreakerType.CORRELATION,
            action=BreakerAction.HALT_NEW_ENTRIES,
            triggered=True,
            reason=(
                f"Entry concurrence = {concurrence:.2%} > halt threshold "
                f"{config.halt_threshold:.2%} — HALT NEW ENTRIES"
            ),
            current_value=concurrence,
            threshold=config.halt_threshold,
            severity=float(np.clip(severity, 0.0, 10.0)),
            metadata={
                "tier": "halt",
                "cluster_weights": dict(snapshot.cluster_weights),
                "n_active": snapshot.n_active_positions,
                "n_total": snapshot.n_instruments,
            },
        )

    # Tier 2: CAP_CLUSTER on elevated concurrence
    if concurrence > config.concurrence_threshold:
        severity = (concurrence - config.concurrence_threshold) / max(
            config.halt_threshold - config.concurrence_threshold, 1e-9
        )

        # Identify clusters that exceed the cap
        over_cap = {
            cluster: weight
            for cluster, weight in snapshot.cluster_weights.items()
            if weight > config.max_cluster_weight
        }

        return BreakerResult(
            breaker_type=BreakerType.CORRELATION,
            action=BreakerAction.CAP_CLUSTER,
            triggered=True,
            reason=(
                f"Entry concurrence = {concurrence:.2%} > threshold "
                f"{config.concurrence_threshold:.2%} — CAP CLUSTER weights "
                f"to {config.max_cluster_weight:.0%}"
            ),
            current_value=concurrence,
            threshold=config.concurrence_threshold,
            severity=float(np.clip(severity, 0.0, 10.0)),
            metadata={
                "tier": "cap",
                "max_cluster_weight": config.max_cluster_weight,
                "clusters_over_cap": over_cap,
                "cluster_weights": dict(snapshot.cluster_weights),
            },
        )

    # Not triggered
    return BreakerResult(
        breaker_type=BreakerType.CORRELATION,
        action=BreakerAction.NONE,
        triggered=False,
        reason="",
        current_value=concurrence,
        threshold=config.concurrence_threshold,
    )


# ══════════════════════════════════════════════════════════════════════════════
# Aggregate evaluator (pure function)
# ══════════════════════════════════════════════════════════════════════════════


def evaluate_circuit_breakers(
    snapshot: PortfolioSnapshot,
    config: Optional[CircuitBreakerConfig] = None,
) -> Tuple[BreakerAction, List[BreakerResult]]:
    """
    Evaluate ALL circuit breakers and return the aggregate result.

    The most severe action across all breakers wins.  All individual
    results are returned for diagnostics.

    This is a **pure function** — no state is modified.  Use
    ``CircuitBreakerManager`` for stateful tracking with cooldowns
    and history.

    Parameters
    ----------
    snapshot : PortfolioSnapshot
        Current portfolio state.
    config : CircuitBreakerConfig or None
        Configuration for all breakers.

    Returns
    -------
    action : BreakerAction
        The most severe action across all breakers.
    results : list[BreakerResult]
        Individual result from each breaker.

    Examples
    --------
    >>> snap = PortfolioSnapshot(portfolio_dd=-0.12, vpin_max=0.5)
    >>> action, results = evaluate_circuit_breakers(snap)
    >>> action == BreakerAction.REDUCE_EXPOSURE  # DD > 10% default
    True
    """
    if config is None:
        config = CircuitBreakerConfig()

    results = [
        check_vpin_breaker(snapshot, config.vpin),
        check_vpin_kurtosis_breaker(snapshot, config.vpin_kurtosis),
        check_drawdown_breaker(snapshot, config.drawdown),
        check_spread_breaker(snapshot, config.spread),
        check_correlation_breaker(snapshot, config.correlation),
    ]

    # Most severe action wins
    action = max(r.action for r in results)

    return action, results


# ══════════════════════════════════════════════════════════════════════════════
# Stateful manager
# ══════════════════════════════════════════════════════════════════════════════


class CircuitBreakerManager:
    """
    Stateful circuit breaker manager with cooldowns and history.

    Wraps the pure evaluation functions with:
    - State tracking (which breakers are currently active)
    - Cooldown enforcement (minimum time between re-triggers)
    - Trigger logging (at WARNING level)
    - History for diagnostics
    - Transition detection (newly triggered / newly cleared)

    The manager is designed to be called frequently (per-tick or
    per-minute) and returns the aggregate action efficiently.

    Parameters
    ----------
    config : CircuitBreakerConfig or None
        Configuration for all breakers.

    Usage
    -----
    ::

        manager = CircuitBreakerManager()

        # On each tick / minute:
        result = manager.evaluate(snapshot)
        if result.action == BreakerAction.FLATTEN:
            flatten_all_positions()
        elif result.action == BreakerAction.REDUCE_EXPOSURE:
            reduce_exposure(0.50)
        elif result.action == BreakerAction.HALT_NEW_ENTRIES:
            block_new_entries()
        elif result.action == BreakerAction.CAP_CLUSTER:
            cap_cluster_weights(0.35)
    """

    def __init__(self, config: Optional[CircuitBreakerConfig] = None) -> None:
        self._config = config or CircuitBreakerConfig()
        self._state = CircuitBreakerState()
        self._last_evaluation: Optional[Tuple[BreakerAction, List[BreakerResult]]] = None

    # ──────────────────────────────────────────────────────────────────
    # Properties
    # ──────────────────────────────────────────────────────────────────

    @property
    def config(self) -> CircuitBreakerConfig:
        """Current configuration."""
        return self._config

    @property
    def state(self) -> CircuitBreakerState:
        """Current breaker state."""
        return self._state

    @property
    def current_action(self) -> BreakerAction:
        """Most severe currently active action."""
        return self._state.current_action

    @property
    def is_any_active(self) -> bool:
        """True if any breaker is currently triggered."""
        return self._state.is_any_active

    @property
    def active_types(self) -> List[BreakerType]:
        """List of currently active breaker types."""
        return self._state.active_types

    # ──────────────────────────────────────────────────────────────────
    # Core evaluation
    # ──────────────────────────────────────────────────────────────────

    def evaluate(self, snapshot: PortfolioSnapshot) -> "EvaluationResult":
        """
        Evaluate all circuit breakers against the current snapshot.

        Updates internal state, enforces cooldowns, logs transitions,
        and returns the aggregate result.

        Parameters
        ----------
        snapshot : PortfolioSnapshot
            Current portfolio state.

        Returns
        -------
        EvaluationResult
            Aggregate action + individual results + transition info.
        """
        now = snapshot.timestamp or datetime.now(tz=timezone.utc)

        # Run pure evaluation
        action, results = evaluate_circuit_breakers(snapshot, self._config)

        # Track transitions
        newly_triggered: List[BreakerResult] = []
        newly_cleared: List[BreakerType] = []

        for result in results:
            bt = result.breaker_type
            was_active = bt in self._state.active_breakers

            if result.triggered:
                # Check cooldown
                if was_active and not self._is_cooldown_expired(bt, now):
                    prev = self._state.active_breakers.get(bt)
                    # Do not let cooldown block escalation to a more severe action.
                    if prev is not None and result.action <= prev.action:
                        continue  # still in cooldown — keep previous state

                if not was_active:
                    newly_triggered.append(result)

                self._state.record_trigger(result, now)

                if self._config.log_triggers:
                    logger.warning(
                        "CIRCUIT BREAKER %s: %s (value=%.4f, threshold=%.4f, severity=%.2f)",
                        result.action.name,
                        result.reason,
                        result.current_value,
                        result.threshold,
                        result.severity,
                    )
            else:
                if was_active:
                    newly_cleared.append(bt)
                    self._state.clear_breaker(bt)

                    if self._config.log_triggers:
                        logger.info(
                            "CIRCUIT BREAKER CLEARED: %s (value=%.4f < threshold=%.4f)",
                            bt.value,
                            result.current_value,
                            result.threshold,
                        )

        # Update aggregate action
        final_action = self._state.update_current_action()

        self._last_evaluation = (final_action, results)

        return EvaluationResult(
            action=final_action,
            results=results,
            newly_triggered=newly_triggered,
            newly_cleared=newly_cleared,
            state_summary=self._state.summary(),
        )

    def _is_cooldown_expired(
        self,
        breaker_type: BreakerType,
        now: datetime,
    ) -> bool:
        """Check if the cooldown period has expired for a breaker."""
        last_time = self._state.last_trigger_time.get(breaker_type)
        if last_time is None:
            return True

        # Ensure both datetimes are comparable
        if last_time.tzinfo is None:
            last_time = last_time.replace(tzinfo=timezone.utc)
        if now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)

        elapsed = (now - last_time).total_seconds()
        return elapsed >= self._config.cooldown_seconds

    # ──────────────────────────────────────────────────────────────────
    # Convenience queries
    # ──────────────────────────────────────────────────────────────────

    def should_flatten(self) -> bool:
        """True if the current state requires flattening all positions."""
        return self._state.current_action == BreakerAction.FLATTEN

    def should_reduce(self) -> bool:
        """True if exposure should be reduced (or flattened)."""
        return self._state.current_action >= BreakerAction.REDUCE_EXPOSURE

    def should_halt_entries(self) -> bool:
        """True if new entries should be blocked."""
        return self._state.current_action >= BreakerAction.HALT_NEW_ENTRIES

    def should_cap_clusters(self) -> bool:
        """True if cluster weights should be capped."""
        return self._state.current_action >= BreakerAction.CAP_CLUSTER

    def get_reduction_fraction(self) -> float:
        """
        Get the exposure reduction fraction.

        Returns 1.0 (flatten) if FLATTEN, the configured reduction
        fraction if REDUCE, or 0.0 (no reduction) otherwise.
        """
        if self._state.current_action == BreakerAction.FLATTEN:
            return 1.0
        elif self._state.current_action == BreakerAction.REDUCE_EXPOSURE:
            dd_result = self._state.active_breakers.get(BreakerType.DRAWDOWN)
            if dd_result is not None:
                return dd_result.metadata.get(
                    "reduction_fraction",
                    self._config.drawdown.reduction_fraction,
                )
            return self._config.drawdown.reduction_fraction
        return 0.0

    def get_max_cluster_weight(self) -> float:
        """
        Get the maximum cluster weight.

        Returns the configured cap if CAP_CLUSTER or higher is active,
        or 1.0 (no cap) otherwise.
        """
        if self.should_cap_clusters():
            return self._config.correlation.max_cluster_weight
        return 1.0

    # ──────────────────────────────────────────────────────────────────
    # Weight application
    # ──────────────────────────────────────────────────────────────────

    def apply_to_weights(
        self,
        weights: Dict[str, float],
        cluster_map: Optional[Dict[str, str]] = None,
    ) -> Dict[str, float]:
        """
        Apply circuit breaker constraints to portfolio weights.

        Modifies weights according to the current breaker state:
        - FLATTEN → all weights set to 0.
        - REDUCE_EXPOSURE → all weights multiplied by
          (1 − reduction_fraction).
        - HALT_NEW_ENTRIES → weights for instruments NOT currently in
          a position are set to 0 (not enforced here since we don't
          know position state — caller should handle).
        - CAP_CLUSTER → per-cluster weight sum capped.

        Parameters
        ----------
        weights : dict[str, float]
            Instrument → weight mapping (from Layer 2).
        cluster_map : dict[str, str] or None
            Instrument → cluster name mapping.  Needed for cluster
            capping.  If None, cluster capping is skipped.

        Returns
        -------
        dict[str, float]
            Adjusted weights.
        """
        action = self._state.current_action

        if action == BreakerAction.NONE:
            return dict(weights)

        adjusted = dict(weights)

        if action == BreakerAction.FLATTEN:
            return {k: 0.0 for k in adjusted}

        if action >= BreakerAction.REDUCE_EXPOSURE:
            reduction = self.get_reduction_fraction()
            scale = 1.0 - reduction
            adjusted = {k: v * scale for k, v in adjusted.items()}

        if action >= BreakerAction.CAP_CLUSTER and cluster_map is not None:
            adjusted = self._apply_cluster_cap(adjusted, cluster_map)

        return adjusted

    def _apply_cluster_cap(
        self,
        weights: Dict[str, float],
        cluster_map: Dict[str, str],
    ) -> Dict[str, float]:
        """Cap per-cluster weight sums to the configured maximum."""
        max_weight = self.get_max_cluster_weight()

        # Group instruments by cluster
        cluster_totals: Dict[str, float] = {}
        cluster_members: Dict[str, List[str]] = {}

        for instrument, weight in weights.items():
            cluster = cluster_map.get(instrument, "unknown")
            cluster_totals[cluster] = cluster_totals.get(cluster, 0.0) + weight
            if cluster not in cluster_members:
                cluster_members[cluster] = []
            cluster_members[cluster].append(instrument)

        # Scale down clusters that exceed the cap
        adjusted = dict(weights)
        for cluster, total in cluster_totals.items():
            if total > max_weight and total > 0:
                scale = max_weight / total
                for instrument in cluster_members[cluster]:
                    adjusted[instrument] = adjusted[instrument] * scale

        return adjusted

    # ──────────────────────────────────────────────────────────────────
    # Reset / reconfigure
    # ──────────────────────────────────────────────────────────────────

    def reset(self) -> None:
        """Reset all breaker state (e.g. at session start)."""
        self._state.reset()
        self._last_evaluation = None
        logger.info("CircuitBreakerManager reset")

    def update_config(self, config: CircuitBreakerConfig) -> None:
        """
        Update breaker configuration.

        Does NOT reset state — currently active breakers remain active
        until the next evaluation clears them.
        """
        self._config = config
        logger.info("CircuitBreakerManager configuration updated")

    # ──────────────────────────────────────────────────────────────────
    # Diagnostics
    # ──────────────────────────────────────────────────────────────────

    def summary(self) -> Dict[str, Any]:
        """
        Return a diagnostic summary of the manager state.

        Returns
        -------
        dict
            Keys: current_action, is_active, active_types,
            trigger_counts, config summary.
        """
        return {
            "current_action": self._state.current_action.name,
            "is_active": self.is_any_active,
            "active_types": [t.value for t in self.active_types],
            "should_flatten": self.should_flatten(),
            "should_reduce": self.should_reduce(),
            "should_halt_entries": self.should_halt_entries(),
            "should_cap_clusters": self.should_cap_clusters(),
            "reduction_fraction": self.get_reduction_fraction(),
            "max_cluster_weight": self.get_max_cluster_weight(),
            "state": self._state.summary(),
        }

    def __repr__(self) -> str:
        return (
            f"CircuitBreakerManager(action={self._state.current_action.name}, "
            f"active={[t.value for t in self.active_types]})"
        )


# ══════════════════════════════════════════════════════════════════════════════
# Evaluation result (returned by CircuitBreakerManager.evaluate)
# ══════════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    """
    Aggregate result from the circuit breaker manager.

    Attributes
    ----------
    action : BreakerAction
        Most severe action across all breakers.
    results : list[BreakerResult]
        Individual result from each breaker.
    newly_triggered : list[BreakerResult]
        Breakers that were NOT active before but are now.
    newly_cleared : list[BreakerType]
        Breaker types that were active before but cleared.
    state_summary : dict
        Diagnostic state summary.
    """

    action: BreakerAction
    results: List[BreakerResult]
    newly_triggered: List[BreakerResult] = field(default_factory=list)
    newly_cleared: List[BreakerType] = field(default_factory=list)
    state_summary: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_triggered(self) -> bool:
        """True if any breaker is currently triggered."""
        return self.action != BreakerAction.NONE

    @property
    def triggered_breakers(self) -> List[BreakerResult]:
        """List of triggered breaker results."""
        return [r for r in self.results if r.triggered]

    @property
    def has_transitions(self) -> bool:
        """True if any breaker changed state (newly triggered or cleared)."""
        return len(self.newly_triggered) > 0 or len(self.newly_cleared) > 0


# ══════════════════════════════════════════════════════════════════════════════
# Module exports
# ══════════════════════════════════════════════════════════════════════════════

__all__ = [
    # Enums
    "BreakerAction",
    "BreakerType",
    # Results
    "BreakerResult",
    "EvaluationResult",
    # Configuration
    "CircuitBreakerConfig",
    "VPINBreakerConfig",
    "VPINKurtosisBreakerConfig",
    "DrawdownBreakerConfig",
    "SpreadBreakerConfig",
    "CorrelationBreakerConfig",
    # State
    "CircuitBreakerState",
    "PortfolioSnapshot",
    # Manager
    "CircuitBreakerManager",
    # Individual breaker functions
    "check_vpin_breaker",
    "check_vpin_kurtosis_breaker",
    "check_drawdown_breaker",
    "check_spread_breaker",
    "check_correlation_breaker",
    # Aggregate evaluator
    "evaluate_circuit_breakers",
]
