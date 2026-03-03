"""
Renko Policy Profiles
=====================

Canonical policy objects for running Renko strategies after signal discovery.

The XAUUSD empirical signal core is treated as immutable; only sizing and
risk overlays are tunable from here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple

from kinetra.renko.backtest import RenkoEmpiricalCore, xauusd_empirical_core
from kinetra.renko.vol_sizer import VolSizingConfig


@dataclass(frozen=True, slots=True)
class DrawdownThrottlePolicy:
    """
    Portfolio-level drawdown throttling policy.

    ``drawdown`` values are fractions in [0, 1], where 0.10 means 10%.
    """

    reduce_at_drawdown: float = 0.06
    reduce_to_size_mult: float = 0.50
    halt_at_drawdown: float = 0.10
    resume_at_drawdown: float = 0.04


@dataclass(frozen=True, slots=True)
class RegimeSizingPolicy:
    """
    Coarse regime-based sizing multipliers.

    This policy never blocks signals; it scales exposure only.
    """

    enabled: bool = True
    # Session bucket -> lot multiplier.
    session_multipliers: Dict[str, float] = field(
        default_factory=lambda: {"asia": 0.85, "london": 1.00, "ny": 1.00}
    )


@dataclass(frozen=True, slots=True)
class BrickScalingPolicy:
    """
    Coarse volatility-bucket brick scaling.

    The base brick comes from the empirical core; scaling is bucketed and
    intentionally sparse to preserve Markov gate semantics.
    """

    enabled: bool = False
    low_vol_mult: float = 0.90
    normal_vol_mult: float = 1.00
    high_vol_mult: float = 1.10


@dataclass(frozen=True, slots=True)
class XAUUSDRenkoProductionPolicy:
    """
    Full production policy for XAUUSD Renko deployment.

    The signal core is immutable. Only sizing/risk overlays are adjustable.
    """

    signal_core: RenkoEmpiricalCore
    vol_sizing: VolSizingConfig
    drawdown: DrawdownThrottlePolicy
    regime_sizing: RegimeSizingPolicy
    brick_scaling: BrickScalingPolicy

    def as_immutable_summary(self) -> Dict[str, object]:
        """Return an operator-facing immutable summary dictionary."""
        return {
            "instrument": self.signal_core.instrument,
            "brick_size": self.signal_core.brick_size,
            "fliprate_threshold": self.signal_core.filter_params.fliprate_threshold,
            "markov_threshold": self.signal_core.filter_params.markov_threshold,
            "flip_exit": self.signal_core.stop_params.exit_on_colour_change,
            "stop_bricks": self.signal_core.stop_params.stop_bricks,
            "vol_target_pct": self.vol_sizing.target_vol_pct,
            "drawdown_halt": self.drawdown.halt_at_drawdown,
        }


def xauusd_production_policy() -> XAUUSDRenkoProductionPolicy:
    """
    Return the canonical post-discovery production policy for XAUUSD.

    Policy intent:
    - Freeze signal logic (entry/exit/filters/brick base assumptions)
    - Improve outcomes via sizing, drawdown control, and execution robustness
    """
    return XAUUSDRenkoProductionPolicy(
        signal_core=xauusd_empirical_core(),
        vol_sizing=VolSizingConfig(
            target_vol_pct=0.01,
            vol_window=50,
            min_vol_window=20,
            vol_floor=0.003,
            vol_ceil=0.20,
            initial_vol_fallback=0.02,
        ),
        drawdown=DrawdownThrottlePolicy(),
        regime_sizing=RegimeSizingPolicy(),
        brick_scaling=BrickScalingPolicy(),
    )


def xauusd_signal_lock() -> Tuple[str, float, float]:
    """Return a minimal immutable lock tuple: (instrument, brick_size, markov_threshold)."""
    core = xauusd_empirical_core()
    return (
        core.instrument,
        core.brick_size,
        core.filter_params.markov_threshold,
    )
