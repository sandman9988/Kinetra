"""
Unified drift detection for the Renko pipeline.

Two-stage drift model
---------------------
1. **Recent-window check** — backtest with existing params on the last
   ``recent_days`` of M1 data and compare Omega to the qualification baseline.
   This is a fast, low-cost check that runs on every scheduled drift sweep.

2. **Rolling OOS check** (optional, ``deep=True``) — delegate to
   :func:`kinetra.renko.drift.rolling_oos_instrument` which applies strict
   no-lookahead rolling train/test folds over the full dataset.  This is
   expensive and is used for recalibration-threshold validation rather than
   routine monitoring.

Returns a :class:`DriftCheckResult` whose fields map directly onto
:class:`~.registry.DriftState` in :class:`~.registry.PipelineQualificationResult`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone

import pandas as pd

from .registry import DriftState, PipelineQualificationResult

logger = logging.getLogger(__name__)


@dataclass
class DriftCheckResult:
    """
    Output of a drift check run.

    Parameters
    ----------
    instrument_id : str
    recalibration_due : bool
        True when any drift threshold was exceeded.
    drift_reason : str
        Human-readable description (empty if no drift detected).
    last_checked_at : str
        ISO-8601 UTC timestamp of this check.
    recent_omega : float
        Omega computed over the recent window (or 0.0 if not computed).
    baseline_omega : float
        Omega from the original qualification (the reference).
    rolling_oos_omega : float
        Stitched OOS Omega from rolling folds (0.0 if deep check skipped).
    """

    instrument_id: str
    recalibration_due: bool
    drift_reason: str
    last_checked_at: str
    recent_omega: float = 0.0
    baseline_omega: float = 0.0
    rolling_oos_omega: float = 0.0

    def to_drift_state(self) -> DriftState:
        """Convert to :class:`~.registry.DriftState` for registry persistence."""
        return DriftState(
            recalibration_due=self.recalibration_due,
            drift_reason=self.drift_reason,
            last_checked_at=self.last_checked_at,
        )


def check_drift(
    result: PipelineQualificationResult,
    m1_df: pd.DataFrame,
    *,
    recent_days: int = 60,
    omega_drop_warn: float = 0.15,
    omega_drop_halt: float = 0.50,
    deep: bool = False,
    rolling_train_days: int = 180,
    rolling_test_days: int = 30,
    rolling_step_days: int = 30,
    rolling_embargo_minutes: int = 30,
) -> DriftCheckResult:
    """
    Check an instrument for calibration drift.

    Parameters
    ----------
    result : PipelineQualificationResult
        The existing qualification record (provides params and baseline metrics).
    m1_df : pd.DataFrame
        Current M1 data (UTC-indexed, at least ``close`` column).
    recent_days : int
        Length of the recent-window check in calendar days.
    omega_drop_warn : float
        Fraction drop from baseline Omega that triggers a warning flag.
    omega_drop_halt : float
        Fraction drop from baseline Omega that triggers ``recalibration_due=True``.
    deep : bool
        If True, also run the rolling OOS check (expensive).
    rolling_train_days : int
        Train-window size for rolling OOS folds (used when ``deep=True``).
    rolling_test_days : int
        Test-window size for rolling OOS folds (used when ``deep=True``).
    rolling_step_days : int
        Step between rolling OOS folds (used when ``deep=True``).
    rolling_embargo_minutes : int
        Embargo between train and test in rolling OOS (used when ``deep=True``).

    Returns
    -------
    DriftCheckResult
        ``recalibration_due=True`` when Omega dropped ≥ ``omega_drop_halt``
        relative to the baseline, or when the deep rolling OOS check also flags.
    """
    now_iso = datetime.now(timezone.utc).isoformat()
    baseline_omega = result.metrics.is_.omega

    # ── Stage 1: recent-window backtest ──────────────────────────────────────
    recent_omega = _recent_window_omega(result, m1_df, recent_days=recent_days)

    reasons = []
    recalibration_due = False

    if baseline_omega > 0 and recent_omega > 0:
        drop = (baseline_omega - recent_omega) / baseline_omega
        if drop >= omega_drop_halt:
            recalibration_due = True
            reasons.append(
                f"Recent {recent_days}d omega {recent_omega:.2f} vs baseline "
                f"{baseline_omega:.2f} (drop={drop:.1%} ≥ halt={omega_drop_halt:.0%})"
            )
        elif drop >= omega_drop_warn:
            reasons.append(
                f"Recent {recent_days}d omega {recent_omega:.2f} vs baseline "
                f"{baseline_omega:.2f} (drop={drop:.1%} ≥ warn={omega_drop_warn:.0%})"
            )

    # ── Stage 2: rolling OOS check (optional) ────────────────────────────────
    rolling_oos_omega = 0.0
    if deep:
        rolling_oos_omega = _rolling_oos_omega(
            result,
            m1_df,
            train_days=rolling_train_days,
            test_days=rolling_test_days,
            step_days=rolling_step_days,
            embargo_minutes=rolling_embargo_minutes,
        )
        if baseline_omega > 0 and rolling_oos_omega > 0:
            deep_drop = (baseline_omega - rolling_oos_omega) / baseline_omega
            if deep_drop >= omega_drop_halt and not recalibration_due:
                recalibration_due = True
                reasons.append(
                    f"Rolling OOS omega {rolling_oos_omega:.2f} vs baseline "
                    f"{baseline_omega:.2f} (drop={deep_drop:.1%})"
                )

    drift_reason = "; ".join(reasons)

    return DriftCheckResult(
        instrument_id=result.instrument_id,
        recalibration_due=recalibration_due,
        drift_reason=drift_reason,
        last_checked_at=now_iso,
        recent_omega=recent_omega,
        baseline_omega=baseline_omega,
        rolling_oos_omega=rolling_oos_omega,
    )


# ══════════════════════════════════════════════════════════════════════════════
# Private helpers
# ══════════════════════════════════════════════════════════════════════════════


def _recent_window_omega(
    result: PipelineQualificationResult,
    m1_df: pd.DataFrame,
    recent_days: int,
) -> float:
    """
    Run a backtest on the last *recent_days* of data with the existing params.
    Returns Omega, or 0.0 on failure.
    """
    try:
        from kinetra.renko.backtest import (
            FilterParams,
            StopParams,
            backtest_instrument,
        )

        # Slice to recent window
        if hasattr(m1_df.index, "tz"):
            cutoff = m1_df.index[-1] - pd.Timedelta(days=recent_days)
            recent_df = m1_df[m1_df.index >= cutoff]
        else:
            recent_df = m1_df.iloc[-recent_days * 1440:]  # ~1440 M1 bars/day

        if len(recent_df) < 500:
            logger.debug(
                "check_drift(%s): not enough recent bars (%d) for recent-window check",
                result.instrument_id,
                len(recent_df),
            )
            return 0.0

        closes = recent_df["close"].values if "close" in recent_df.columns else recent_df.iloc[:, 3].values

        fp_dict = result.params.filter_params or {}
        try:
            fp = FilterParams(**{k: v for k, v in fp_dict.items() if k in FilterParams.__dataclass_fields__})  # type: ignore[attr-defined]
        except Exception:
            fp = FilterParams()

        sp_dict = result.params.stop_params or {}
        try:
            sp = StopParams(**{k: v for k, v in sp_dict.items() if k in StopParams.__dataclass_fields__})  # type: ignore[attr-defined]
        except Exception:
            sp = StopParams()

        bt = backtest_instrument(
            symbol=result.key.symbol,
            closes=closes,
            brick_size=result.params.brick_size,
            filter_params=fp,
            stop_params=sp,
            session_break_minutes=result.data.session_break_minutes,
        )
        return float(bt.omega)

    except Exception as exc:
        logger.warning(
            "check_drift(%s): recent-window backtest failed: %s",
            result.instrument_id,
            exc,
        )
        return 0.0


def _rolling_oos_omega(
    result: PipelineQualificationResult,
    m1_df: pd.DataFrame,
    *,
    train_days: int,
    test_days: int,
    step_days: int,
    embargo_minutes: int,
) -> float:
    """
    Delegate to :func:`kinetra.renko.drift.rolling_oos_instrument` and
    return the stitched OOS Omega.
    """
    try:
        from kinetra.renko.backtest import FilterParams, StopParams
        from kinetra.renko.drift import rolling_oos_instrument

        closes = m1_df["close"].values if "close" in m1_df.columns else m1_df.iloc[:, 3].values

        fp_dict = result.params.filter_params or {}
        try:
            fp = FilterParams(**{k: v for k, v in fp_dict.items() if k in FilterParams.__dataclass_fields__})  # type: ignore[attr-defined]
        except Exception:
            fp = FilterParams()

        sp_dict = result.params.stop_params or {}
        try:
            sp = StopParams(**{k: v for k, v in sp_dict.items() if k in StopParams.__dataclass_fields__})  # type: ignore[attr-defined]
        except Exception:
            sp = StopParams()

        oos_result = rolling_oos_instrument(
            symbol=result.key.symbol,
            closes=closes,
            brick_size=result.params.brick_size,
            filter_params=fp,
            stop_params=sp,
            train_days=train_days,
            test_days=test_days,
            step_days=step_days,
            embargo_minutes=embargo_minutes,
        )
        return float(oos_result.stitched_oos_omega)

    except Exception as exc:
        logger.warning(
            "check_drift(%s): rolling OOS failed: %s",
            result.instrument_id,
            exc,
        )
        return 0.0
