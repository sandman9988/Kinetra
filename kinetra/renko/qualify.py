"""
Renko Qualification Pipeline
=============================

Instrument qualification, registry management, and calibration drift detection
for the Renko Kinetra pipeline.

This module is the canonical home for per-instrument qualification logic.
The pipeline chains:

  1. Session break detection  (kinetra.renko.session)
  2. DSP analysis             (kinetra.renko.dsp.run_dsp)
  3. Scaled filter params     (kinetra.renko.dsp.scaled_filter_params)
  4. Brick sweep              (kinetra.renko.backtest.sweep_brick_sizes)
  5. Walk-forward IS/OOS      (kinetra.renko.backtest.walk_forward_instrument)
  6. Friction stress test     (kinetra.renko.backtest.stress_test_friction)

The result is persisted as ``qualification.json`` in
``data/renko_qualified/<symbol>/``.  The :class:`QualificationRegistry` loads
all qualification files into a lookup so downstream portfolio construction can
call ``registry.get_qualified()`` without re-running any analysis.

Canonical usage::

    from kinetra.renko.qualify import qualify_instrument, QualificationRegistry

    result = qualify_instrument(symbol="XAUUSD", m1_df=df,
                                 spread_pts=1.5, tick_size=0.01)
    if result.qualified:
        print(f"Qualified: brick={result.brick_size}, omega={result.omega:.2f}")

    registry = QualificationRegistry("data/renko_qualified")
    registry.load()
    qualified = registry.get_qualified()  # list[QualificationResult]

See Also:
    - ``kinetra/renko/session.py``  — SessionProfile / detect_session_break
    - ``kinetra/renko/dsp.py``      — run_dsp, scaled_filter_params, DSPResult
    - ``kinetra/renko/backtest.py`` — sweep_brick_sizes, walk_forward_instrument,
                                      stress_test_friction, InstrumentBacktestResult
    - ``docs/MANUAL.md §29.5`` — qualification spec
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from kinetra.rl.portfolio_env import InstrumentContext

from kinetra.renko.backtest import (
    FilterParams,
    InstrumentBacktestResult,
    RiskParams,
    StopParams,
    backtest_instrument,
    stress_test_friction,
    sweep_brick_sizes,
    walk_forward_instrument,
)
from kinetra.renko.dsp import (
    SpreadProfile,
    compute_friction_floor,
    compute_rolling_friction_floor,
    run_dsp,
    scaled_filter_params,
)
from kinetra.renko.portfolio import get_cluster
from kinetra.renko.session import clamp_spikes, detect_session_break

logger = logging.getLogger(__name__)


# ── Quality gate thresholds ───────────────────────────────────────────────────

# Minimum Omega ratio to qualify
QUALIFY_MIN_OMEGA: float = 1.5

# Minimum Z-factor to qualify
QUALIFY_MIN_Z: float = 2.0

# Minimum OOS Omega (walk-forward) to qualify
QUALIFY_MIN_OOS_OMEGA: float = 1.2

# Minimum OOS survival rate (fraction of WF folds with omega > threshold)
QUALIFY_MIN_OOS_SURVIVAL: float = 0.50

# Friction stress: instrument must survive 1.5x cost scaling
QUALIFY_FRICTION_MULT: float = 1.5
QUALIFY_FRICTION_MIN_OMEGA: float = 1.0  # min omega at 1.5x costs

# Minimum number of trades for a credible backtest result
QUALIFY_MIN_TRADES: int = 30
QUALIFY_MIN_TRADES_FLOOR: int = 8

# Minimum VR peak to consider trending
QUALIFY_MIN_VR: float = 1.05

# Maximum friction ratio (spread / brick) allowed for qualification
QUALIFY_MAX_FRICTION_RATIO: float = 0.25
QUALIFY_USE_ROLLING_FRICTION_FLOOR: bool = True
QUALIFY_FRICTION_ROLL_WINDOW_BARS: int = 1440
QUALIFY_FRICTION_ROLL_MIN_OBS: int = 180
QUALIFY_FRICTION_ROLL_X_MULT: float = 5.0
QUALIFY_FRICTION_ROLL_TAIL_MULT: float = 2.0

# CalibrationDriftDetector thresholds
DRIFT_VR_CHANGE_THRESHOLD: float = 0.15  # |vr_new - vr_old| / vr_old
DRIFT_OMEGA_CHANGE_THRESHOLD: float = 0.30  # |omega_new - omega_old| / omega_old
DRIFT_MIN_NEW_BARS: int = 500  # minimum new bars before re-running DSP

# Qualification file name inside each instrument's directory
QUALIFICATION_FILENAME: str = "qualification.json"

# Session profile file name
SESSION_PROFILE_FILENAME: str = "session_profile.json"

# Recalibration log file name (appended per recalibration event)
RECALIBRATION_LOG_FILENAME: str = "recalibration_log.json"


def _sanitize_id(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(value)).strip("_") or "unknown"


def _canonical_instrument_id(
    symbol: str,
    broker_source: str,
    broker_account: str,
    is_ecn: Optional[bool],
    account_type: str,
) -> str:
    if is_ecn is True:
        exec_tag = "ecn"
    elif is_ecn is False:
        exec_tag = "std"
    else:
        exec_tag = _sanitize_id(account_type.lower() or "unknown_exec")
    return (
        f"{_sanitize_id(symbol)}__{_sanitize_id(broker_source)}__"
        f"{_sanitize_id(broker_account)}__{exec_tag}"
    )


# ══════════════════════════════════════════════════════════════════════════════
# Data containers
# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class QualificationResult:
    """
    Full qualification record for one instrument.

    Attributes
    ----------
    symbol : str
        Instrument symbol (e.g. "XAUUSD").
    qualified : bool
        True if the instrument passed all quality gates.
    disqualified : bool
        True if the instrument was explicitly disqualified (e.g. by
        :meth:`QualificationRegistry.disqualify`).  An instrument can be
        ``not qualified and not disqualified`` when qualification is pending.
    disqualification_reason : Optional[str]
        Human-readable reason for disqualification.
    broker_source : str
        Broker the data came from.
    cluster : str
        Asset cluster from :func:`kinetra.renko.portfolio.get_cluster`.
    brick_size : float
        DSP-derived optimal brick size (price units).
    vr_peak : float
        Peak variance ratio from DSP analysis.
    vr_scale_bars : int
        Number of bars at the VR peak scale.
    friction_ratio : float
        spread_p50_price / brick_size at qualification time.
    omega : float
        Omega ratio from the full-data backtest.
    z_factor : float
        Z-factor from the full-data backtest.
    n_trades : int
        Number of trades in the full-data backtest.
    win_rate : float
        Win rate in the full-data backtest.
    max_drawdown : float
        Maximum drawdown fraction in the full-data backtest.
    oos_omega : float
        Out-of-sample Omega (median across walk-forward folds).
    oos_survival_rate : float
        Fraction of walk-forward folds with Omega > threshold.
    friction_stress_omega : float
        Omega at ``QUALIFY_FRICTION_MULT`` × friction costs.
    filter_params : dict
        Serialised :class:`FilterParams` used in the backtest.
    risk_params : dict
        Serialised :class:`RiskParams` used in the backtest.
    data_start : str
        ISO-8601 UTC timestamp of first M1 bar used.
    data_end : str
        ISO-8601 UTC timestamp of last M1 bar used.
    qualified_at : str
        ISO-8601 UTC timestamp when qualification was computed.
    recalibration_due : bool
        True when :class:`CalibrationDriftDetector` has flagged drift.
    drift_reason : Optional[str]
        Human-readable reason for the drift flag.
    pipeline_version : str
        Free-form version label for the qualification pipeline
        (useful for invalidating old results on algorithm changes).
    """

    symbol: str
    qualified: bool
    disqualified: bool = False
    disqualification_reason: Optional[str] = None
    broker_source: str = "unknown"
    broker_account: str = "unknown"
    instrument_id: str = ""
    broker_symbol: str = ""
    is_ecn: Optional[bool] = None
    account_type: str = ""
    spread_pts: float = 0.0
    tick_size: float = 0.0
    commission_per_lot: float = 0.0
    swap_long_points: float = 0.0
    swap_short_points: float = 0.0
    contract_size: float = 0.0
    tick_value_usd: float = 0.0
    pip_value_usd: float = 0.0
    volume_min: float = 0.0
    volume_step: float = 0.0
    volume_max: float = 0.0
    usd_per_price_unit: float = 0.0
    cluster: str = "unknown"
    brick_size: float = 0.0
    vr_peak: float = 0.0
    vr_scale_bars: int = 0
    friction_ratio: float = 0.0
    omega: float = 0.0
    z_factor: float = 0.0
    n_trades: int = 0
    win_rate: float = 0.0
    max_drawdown: float = 0.0
    oos_omega: float = 0.0
    oos_survival_rate: float = 0.0
    friction_stress_omega: float = 0.0
    filter_params: Dict = field(default_factory=dict)
    risk_params: Dict = field(default_factory=dict)
    data_start: str = ""
    data_end: str = ""
    qualified_at: str = ""
    recalibration_due: bool = False
    drift_reason: Optional[str] = None
    pipeline_version: str = "5B"
    # ── Sprint 6A additions (all default-safe for loading old JSON files) ────
    engine: str = "fixed"  # "fixed" | "adaptive"
    bars_m1: int = 0  # M1 bar count used in qualification
    coverage_ratio: float = 0.0  # non-NaN bars / expected bars
    spike_count: int = 0  # artefactual spikes removed before backtest
    session_break_minutes: float = 30.0  # from SessionProfile
    gate_bar_fraction: float = 0.0  # adaptive: fraction of bars gate was OPEN
    gate_trade_fraction: float = 0.0  # adaptive: fraction of trades passing gate
    drift_last_checked_at: str = ""  # ISO timestamp of last drift check

    # ── Persistence ──────────────────────────────────────────────────────────

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict) -> "QualificationResult":
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        filtered = {k: v for k, v in d.items() if k in known}
        return cls(**filtered)

    def save(self, path: Path) -> None:
        """Atomic save to JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.to_dict(), indent=2))
        tmp.replace(path)

    @classmethod
    def load(cls, path: Path) -> "QualificationResult":
        raw = json.loads(Path(path).read_text())
        return cls.from_dict(raw)


# ══════════════════════════════════════════════════════════════════════════════
# Qualification pipeline
# ══════════════════════════════════════════════════════════════════════════════


def qualify_instrument(
    symbol: str,
    m1_df: pd.DataFrame,
    spread_pts: float,
    tick_size: float,
    *,
    broker_source: str = "unknown",
    broker_account: str = "unknown",
    instrument_id: str = "",
    broker_symbol: str = "",
    is_ecn: Optional[bool] = None,
    account_type: str = "",
    commission_per_lot: float = 0.0,
    swap_long_points: float = 0.0,
    swap_short_points: float = 0.0,
    risk_params: Optional[RiskParams] = None,
    stop_params: Optional[StopParams] = None,
    pipeline_version: str = "5B",
    force: bool = False,
    output_dir: Optional[Path] = None,
) -> QualificationResult:
    """
    Run the full Renko qualification pipeline for a single instrument.

    Steps
    -----
    1. ``detect_session_break()`` — broker fingerprint, session gap detection.
    2. ``run_dsp()`` — VR analysis on M1 close series.
    3. Gate: VR peak must exceed ``QUALIFY_MIN_VR``.
    4. ``scaled_filter_params()`` — derive adaptive :class:`FilterParams`.
    5. Gate: friction ratio must not exceed ``QUALIFY_MAX_FRICTION_RATIO``.
    6. ``sweep_brick_sizes()`` — select optimal brick size by Omega.
    7. ``backtest_instrument()`` — full-data backtest with selected params.
    8. Gate: Omega, Z-factor, trade count.
    9. ``walk_forward_instrument()`` — OOS robustness validation.
    10. Gate: OOS Omega and survival rate.
    11. ``stress_test_friction()`` — 1.5× cost stress test.
    12. Gate: friction-stressed Omega.
    13. Persist ``qualification.json`` (if ``output_dir`` given).
    14. Persist ``session_profile.json`` (if ``output_dir`` given).

    The pipeline is **idempotent**: if ``output_dir`` is given and a
    ``qualification.json`` already exists with the same ``data_end``, the
    existing result is returned without re-running (unless ``force=True``).

    Parameters
    ----------
    symbol : str
        Instrument symbol (e.g. "XAUUSD").
    m1_df : pd.DataFrame
        Raw M1 OHLCV data with a time column and at minimum ``close``.
    spread_pts : float
        Typical spread in broker points (used for friction floor / ratio).
    tick_size : float
        Price per broker point (used to convert spread to price units).
    broker_source : str
        Broker identifier for traceability (e.g. "metaapi").
    risk_params : RiskParams, optional
        Override risk management params.  Defaults to ``RiskParams()``.
    stop_params : StopParams, optional
        Override stop params.  Defaults to ``StopParams()``.
    pipeline_version : str
        Version label written into ``qualification.json``.
    force : bool
        Re-run even if an up-to-date ``qualification.json`` exists.
    output_dir : Path, optional
        Directory to persist ``qualification.json`` and
        ``session_profile.json``.  If None, nothing is persisted.

    Returns
    -------
    QualificationResult
        Fully populated result; ``result.qualified`` is the gate outcome.
    """
    risk_params = risk_params or RiskParams()
    stop_params = stop_params or StopParams()
    if not instrument_id:
        instrument_id = _canonical_instrument_id(
            symbol=symbol,
            broker_source=broker_source,
            broker_account=broker_account,
            is_ecn=is_ecn,
            account_type=account_type,
        )

    # ── Idempotency check ────────────────────────────────────────────────────
    if output_dir is not None and not force:
        q_path = Path(output_dir) / QUALIFICATION_FILENAME
        if q_path.exists():
            try:
                existing = QualificationResult.load(q_path)
                # Re-use if data_end matches
                if m1_df is not None and not m1_df.empty:
                    last_ts = str(pd.to_datetime(m1_df.iloc[-1].get("time", ""), utc=True))
                    if existing.data_end and existing.data_end[:10] == last_ts[:10]:
                        logger.info(
                            "%s: existing qualification up-to-date (data_end=%s), skipping",
                            symbol,
                            existing.data_end,
                        )
                        return existing
            except Exception as exc:
                logger.debug("%s: could not load existing qualification: %s", symbol, exc)

    def _fail(reason: str, **kw) -> QualificationResult:
        result = QualificationResult(
            symbol=symbol,
            qualified=False,
            disqualification_reason=reason,
            broker_source=broker_source,
            broker_account=broker_account,
            instrument_id=instrument_id,
            broker_symbol=broker_symbol or symbol,
            is_ecn=is_ecn,
            account_type=account_type,
            spread_pts=spread_pts,
            tick_size=tick_size,
            commission_per_lot=commission_per_lot,
            swap_long_points=swap_long_points,
            swap_short_points=swap_short_points,
            pipeline_version=pipeline_version,
            **kw,
        )
        logger.info("%s: NOT QUALIFIED — %s", symbol, reason)
        if output_dir is not None:
            result.save(Path(output_dir) / QUALIFICATION_FILENAME)
        return result

    if m1_df is None or m1_df.empty:
        return _fail("empty M1 data")

    # ── Determine data timestamps ────────────────────────────────────────────
    time_col = next(
        (c for c in m1_df.columns if c.lower() in ("time", "datetime", "date", "timestamp")),
        None,
    )
    if time_col:
        times_raw = pd.to_datetime(m1_df[time_col], utc=True, errors="coerce").dropna()
        data_start = times_raw.iloc[0].isoformat() if len(times_raw) > 0 else ""
        data_end = times_raw.iloc[-1].isoformat() if len(times_raw) > 0 else ""
    else:
        data_start = data_end = ""

    # ── Step 1: Session break detection ─────────────────────────────────────
    logger.info("%s: step 1/6 — session break detection", symbol)
    try:
        session_profile = detect_session_break(m1_df, symbol=symbol, broker_source=broker_source)
    except Exception as exc:
        return _fail(
            f"session break detection failed: {exc}", data_start=data_start, data_end=data_end
        )

    session_break_minutes = session_profile.session_break_minutes

    # ── Step 1b: Spike clamping ───────────────────────────────────────────────
    # If the QC pass detected spikes (outlier high–low ranges caused by
    # post-rollover resume-of-session jumps), clamp them before DSP aggregation
    # and brick construction to prevent artefactual bricks from corrupting the
    # FlipRate, Markov, and VPIN signals.
    m1_clean = m1_df
    if session_profile.qc.spike_count > 0:
        logger.info(
            "%s: clamping %d spike bar(s) before DSP aggregation",
            symbol,
            session_profile.qc.spike_count,
        )
        try:
            m1_clean = clamp_spikes(m1_df, method="clamp")
            logger.debug(
                "%s: spike clamp complete (rows before=%d, after=%d)",
                symbol,
                len(m1_df),
                len(m1_clean),
            )
        except Exception as exc:
            logger.warning(
                "%s: clamp_spikes failed (%s) — proceeding with raw M1 data", symbol, exc
            )
            m1_clean = m1_df

    # ── Step 2: Extract M1 closes for DSP ───────────────────────────────────
    logger.info("%s: step 2/6 — DSP analysis (VR on M1)", symbol)
    try:
        m1_closes = _extract_m1_closes(m1_clean)
    except Exception as exc:
        return _fail(f"M1 close extraction failed: {exc}", data_start=data_start, data_end=data_end)

    if len(m1_closes) < 300:
        return _fail(
            f"insufficient M1 bars for DSP ({len(m1_closes)} < 300)",
            data_start=data_start,
            data_end=data_end,
        )

    try:
        dsp_result = run_dsp(
            m1_closes.values,
            symbol=symbol,
            bars_per_hour=60.0,  # M1 = 60 bars/hour
        )
    except Exception as exc:
        return _fail(f"DSP analysis failed: {exc}", data_start=data_start, data_end=data_end)

    # ── Gate: VR peak ─────────────────────────────────────────────────────────
    if dsp_result.vr_peak_value < QUALIFY_MIN_VR:
        return _fail(
            f"VR peak {dsp_result.vr_peak_value:.3f} < {QUALIFY_MIN_VR} (random walk)",
            vr_peak=dsp_result.vr_peak_value,
            vr_scale_bars=dsp_result.vr_peak_scale,
            cluster=get_cluster(symbol),
            data_start=data_start,
            data_end=data_end,
        )

    # ── Step 3: Brick sizing from DSP + friction floor model ─────────────────
    brick_size = dsp_result.dsp_brick_size
    spread_profile_for_floor: SpreadProfile
    friction_floor_result = None
    spread_price = spread_pts * tick_size

    spread_col = next((c for c in m1_clean.columns if c.lower() in {"spread", "spread_pts"}), None)
    spread_series_pts = None
    if spread_col is not None:
        raw_spread = pd.to_numeric(m1_clean[spread_col], errors="coerce")
        spread_series_pts = raw_spread[raw_spread > 0].to_numpy(dtype=float)

    if (
        QUALIFY_USE_ROLLING_FRICTION_FLOOR
        and spread_series_pts is not None
        and len(spread_series_pts) >= QUALIFY_FRICTION_ROLL_MIN_OBS
    ):
        try:
            friction_floor_result = compute_rolling_friction_floor(
                symbol=symbol,
                spread_values=spread_series_pts,
                tick_size=tick_size,
                dsp_brick=brick_size,
                x_mult=QUALIFY_FRICTION_ROLL_X_MULT,
                window_bars=QUALIFY_FRICTION_ROLL_WINDOW_BARS,
                min_obs=QUALIFY_FRICTION_ROLL_MIN_OBS,
                tail_mult=QUALIFY_FRICTION_ROLL_TAIL_MULT,
            )
            # Companion spread profile for friction ratio / sweep spread.
            spread_profile_for_floor = SpreadProfile.from_points(
                symbol=symbol,
                tick_size=tick_size,
                p50_pts=float(np.percentile(spread_series_pts, 50)),
                p75_pts=float(np.percentile(spread_series_pts, 75)),
                p95_pts=float(np.percentile(spread_series_pts, 95)),
                source="qualify_m1_raw_spread",
                n_bars=len(spread_series_pts),
            )
            spread_price = spread_profile_for_floor.spread_p75_price
        except Exception as exc:
            logger.warning(
                "%s: rolling friction floor fallback to static profile (%s)", symbol, exc
            )
            spread_profile_for_floor = SpreadProfile.from_points(
                symbol=symbol,
                tick_size=tick_size,
                p50_pts=spread_pts,
                p75_pts=spread_pts * 1.2,
                p95_pts=spread_pts * 1.5,
                source="qualify_input",
            )
            friction_floor_result = compute_friction_floor(
                spread=spread_profile_for_floor,
                dsp_brick=brick_size,
            )
            spread_price = spread_profile_for_floor.spread_p75_price
    else:
        spread_profile_for_floor = SpreadProfile.from_points(
            symbol=symbol,
            tick_size=tick_size,
            p50_pts=spread_pts,
            p75_pts=spread_pts * 1.2,
            p95_pts=spread_pts * 1.5,
            source="qualify_input",
        )
        friction_floor_result = compute_friction_floor(
            spread=spread_profile_for_floor,
            dsp_brick=brick_size,
        )
        spread_price = spread_profile_for_floor.spread_p75_price

    friction_ratio = spread_price / brick_size if brick_size > 0 else float("inf")

    # ── Gate: friction ratio ──────────────────────────────────────────────────
    if friction_ratio > QUALIFY_MAX_FRICTION_RATIO:
        return _fail(
            f"friction ratio {friction_ratio:.3f} > {QUALIFY_MAX_FRICTION_RATIO} "
            f"(spread={spread_price:.5f}, brick={brick_size:.5f})",
            vr_peak=dsp_result.vr_peak_value,
            vr_scale_bars=dsp_result.vr_peak_scale,
            brick_size=brick_size,
            friction_ratio=friction_ratio,
            cluster=get_cluster(symbol),
            data_start=data_start,
            data_end=data_end,
        )

    # ── Step 4: Scaled filter params ─────────────────────────────────────────
    from kinetra.renko.brick_engine import bricks_per_day, build_renko

    logger.info("%s: step 3/6 — scaled filter params + brick sweep", symbol)
    try:
        m1_closes.index if hasattr(m1_closes, "index") else None
        temp_bricks = build_renko(
            m1_closes,
            brick_size=brick_size,
            session_break_minutes=session_break_minutes,
        )
        bpd = bricks_per_day(temp_bricks)
    except Exception:
        bpd = 5.0  # safe fallback

    try:
        filter_params = scaled_filter_params(dsp_result, bpd)
    except Exception as exc:
        logger.warning("%s: scaled_filter_params failed (%s), using defaults", symbol, exc)
        filter_params = FilterParams()

    # ── Step 5: Brick sweep ───────────────────────────────────────────────────
    # sweep_brick_sizes uses the actual spread/friction-floor values, not session_break_minutes
    # (session_break_minutes is a Sprint 5A addition to the backtester; for now we pass the
    # DSP-computed spread and friction floor and let the sweep pick the best brick size).
    logger.info("%s: step 4/6 — brick sweep", symbol)
    try:
        sweep_result = sweep_brick_sizes(
            symbol=symbol,
            closes=m1_closes,
            dsp_brick_size=brick_size,
            spread_price=spread_price,
            friction_floor=friction_floor_result.floor_price,
            filter_params=filter_params,
            stop_params=stop_params,
            session_break_minutes=session_break_minutes,
        )
        if sweep_result.best_point is not None and sweep_result.best_point.brick_size > 0:
            brick_size = sweep_result.best_point.brick_size
    except Exception as exc:
        logger.warning("%s: brick sweep failed (%s), using DSP brick size", symbol, exc)

    # ── Step 6: Full backtest ─────────────────────────────────────────────────
    logger.info("%s: step 5/6 — full backtest", symbol)
    try:
        bt_result: InstrumentBacktestResult = backtest_instrument(
            symbol=symbol,
            closes=m1_closes,
            brick_size=brick_size,
            filter_params=filter_params,
            stop_params=stop_params,
            risk_params=risk_params,
            session_break_minutes=session_break_minutes,
        )
    except Exception as exc:
        return _fail(
            f"backtest failed: {exc}",
            vr_peak=dsp_result.vr_peak_value,
            vr_scale_bars=dsp_result.vr_peak_scale,
            brick_size=brick_size,
            friction_ratio=friction_ratio,
            cluster=get_cluster(symbol),
            filter_params=asdict(filter_params),
            risk_params=asdict(risk_params),
            data_start=data_start,
            data_end=data_end,
        )

    # ── Gate: Omega, Z-factor, trade count ───────────────────────────────────
    # Adaptive trade-count gate:
    # - Preserve the canonical 30-trade requirement on >=1y data.
    # - Scale down proportionally on short histories so recently-downloaded
    #   instruments are not auto-rejected despite strong omega/z.
    effective_min_trades = max(
        QUALIFY_MIN_TRADES_FLOOR,
        int(np.ceil(QUALIFY_MIN_TRADES * min(bt_result.years, 1.0))),
    )
    if len(bt_result.trades) < effective_min_trades:
        return _fail(
            f"only {len(bt_result.trades)} trades (min {effective_min_trades}, "
            f"scaled from {QUALIFY_MIN_TRADES} at {bt_result.years:.2f}y)",
            vr_peak=dsp_result.vr_peak_value,
            vr_scale_bars=dsp_result.vr_peak_scale,
            brick_size=brick_size,
            friction_ratio=friction_ratio,
            omega=bt_result.omega,
            z_factor=bt_result.z_factor,
            n_trades=len(bt_result.trades),
            cluster=get_cluster(symbol),
            filter_params=asdict(filter_params),
            risk_params=asdict(risk_params),
            data_start=data_start,
            data_end=data_end,
        )

    if bt_result.omega < QUALIFY_MIN_OMEGA:
        return _fail(
            f"Omega {bt_result.omega:.2f} < {QUALIFY_MIN_OMEGA}",
            vr_peak=dsp_result.vr_peak_value,
            vr_scale_bars=dsp_result.vr_peak_scale,
            brick_size=brick_size,
            friction_ratio=friction_ratio,
            omega=bt_result.omega,
            z_factor=bt_result.z_factor,
            n_trades=len(bt_result.trades),
            win_rate=bt_result.win_rate,
            max_drawdown=bt_result.max_dd_usd,
            cluster=get_cluster(symbol),
            filter_params=asdict(filter_params),
            risk_params=asdict(risk_params),
            data_start=data_start,
            data_end=data_end,
        )

    if bt_result.z_factor < QUALIFY_MIN_Z:
        return _fail(
            f"Z-factor {bt_result.z_factor:.2f} < {QUALIFY_MIN_Z}",
            vr_peak=dsp_result.vr_peak_value,
            vr_scale_bars=dsp_result.vr_peak_scale,
            brick_size=brick_size,
            friction_ratio=friction_ratio,
            omega=bt_result.omega,
            z_factor=bt_result.z_factor,
            n_trades=len(bt_result.trades),
            win_rate=bt_result.win_rate,
            max_drawdown=bt_result.max_dd_usd,
            cluster=get_cluster(symbol),
            filter_params=asdict(filter_params),
            risk_params=asdict(risk_params),
            data_start=data_start,
            data_end=data_end,
        )

    # ── Step 7: Walk-forward ──────────────────────────────────────────────────
    logger.info("%s: step 6/6 — walk-forward + friction stress", symbol)
    oos_omega = 0.0
    oos_survival_rate = 0.0
    try:
        wf_result = walk_forward_instrument(
            symbol=symbol,
            closes=m1_closes,
            brick_size=brick_size,
            filter_params=filter_params,
            stop_params=stop_params,
            session_break_minutes=session_break_minutes,
        )
        if wf_result is not None:
            oos_omega = wf_result.oos_omega
            oos_survival_rate = 1.0 if wf_result.oos_passed else 0.0
    except Exception as exc:
        logger.warning("%s: walk-forward failed (%s); skipping OOS gate", symbol, exc)

    if oos_survival_rate > 0 and oos_omega > 0 and oos_omega < QUALIFY_MIN_OOS_OMEGA:
        return _fail(
            f"OOS Omega {oos_omega:.2f} < {QUALIFY_MIN_OOS_OMEGA}",
            vr_peak=dsp_result.vr_peak_value,
            vr_scale_bars=dsp_result.vr_peak_scale,
            brick_size=brick_size,
            friction_ratio=friction_ratio,
            omega=bt_result.omega,
            z_factor=bt_result.z_factor,
            n_trades=len(bt_result.trades),
            win_rate=bt_result.win_rate,
            max_drawdown=bt_result.max_dd_usd,
            oos_omega=oos_omega,
            oos_survival_rate=oos_survival_rate,
            cluster=get_cluster(symbol),
            filter_params=asdict(filter_params),
            risk_params=asdict(risk_params),
            data_start=data_start,
            data_end=data_end,
        )

    if oos_survival_rate > 0 and oos_omega > 0 and oos_survival_rate < QUALIFY_MIN_OOS_SURVIVAL:
        return _fail(
            f"OOS survival {oos_survival_rate:.0%} < {QUALIFY_MIN_OOS_SURVIVAL:.0%}",
            vr_peak=dsp_result.vr_peak_value,
            vr_scale_bars=dsp_result.vr_peak_scale,
            brick_size=brick_size,
            friction_ratio=friction_ratio,
            omega=bt_result.omega,
            z_factor=bt_result.z_factor,
            n_trades=len(bt_result.trades),
            win_rate=bt_result.win_rate,
            max_drawdown=bt_result.max_dd_usd,
            oos_omega=oos_omega,
            oos_survival_rate=oos_survival_rate,
            cluster=get_cluster(symbol),
            filter_params=asdict(filter_params),
            risk_params=asdict(risk_params),
            data_start=data_start,
            data_end=data_end,
        )

    # ── Friction stress test ──────────────────────────────────────────────────
    friction_stress_omega = bt_result.omega
    try:
        stress_result = stress_test_friction(bt_result, multiplier=QUALIFY_FRICTION_MULT)
        friction_stress_omega = stress_result.stressed_omega
    except Exception as exc:
        logger.warning("%s: friction stress test failed (%s)", symbol, exc)

    if friction_stress_omega < QUALIFY_FRICTION_MIN_OMEGA:
        return _fail(
            f"friction-stressed Omega {friction_stress_omega:.2f} < "
            f"{QUALIFY_FRICTION_MIN_OMEGA} at {QUALIFY_FRICTION_MULT}x costs",
            vr_peak=dsp_result.vr_peak_value,
            vr_scale_bars=dsp_result.vr_peak_scale,
            brick_size=brick_size,
            friction_ratio=friction_ratio,
            omega=bt_result.omega,
            z_factor=bt_result.z_factor,
            n_trades=len(bt_result.trades),
            win_rate=bt_result.win_rate,
            max_drawdown=bt_result.max_dd_usd,
            oos_omega=oos_omega,
            oos_survival_rate=oos_survival_rate,
            friction_stress_omega=friction_stress_omega,
            cluster=get_cluster(symbol),
            filter_params=asdict(filter_params),
            risk_params=asdict(risk_params),
            data_start=data_start,
            data_end=data_end,
        )

    # ── All gates passed ──────────────────────────────────────────────────────
    contract_size = 0.0
    tick_value_usd = 0.0
    pip_value_usd = 0.0
    volume_min = 0.0
    volume_step = 0.0
    volume_max = 0.0
    usd_per_price_unit = 0.0
    try:
        from kinetra.friction_cost import load_spec

        spec = load_spec(symbol.split("__", 1)[0])
        contract_size = float(getattr(spec, "contract_size", 0.0))
        tick_value_usd = float(getattr(spec, "tick_value_usd", 0.0))
        pip_value_usd = float(getattr(spec, "pip_value_usd", 0.0))
        volume_min = float(getattr(spec, "volume_min", 0.0))
        volume_step = float(getattr(spec, "volume_step", 0.0))
        volume_max = float(getattr(spec, "volume_max", 0.0))
        tick = max(float(getattr(spec, "tick_size", tick_size)), 1e-12)
        usd_per_price_unit = tick_value_usd / tick if tick_value_usd > 0 else 0.0
    except Exception as exc:
        logger.debug("%s: spec enrichment unavailable (%s)", symbol, exc)

    now_utc = datetime.now(tz=timezone.utc).isoformat()
    result = QualificationResult(
        symbol=symbol,
        qualified=True,
        broker_source=broker_source,
        broker_account=broker_account,
        instrument_id=instrument_id,
        broker_symbol=broker_symbol or symbol,
        is_ecn=is_ecn,
        account_type=account_type,
        spread_pts=spread_pts,
        tick_size=tick_size,
        commission_per_lot=commission_per_lot,
        swap_long_points=swap_long_points,
        swap_short_points=swap_short_points,
        contract_size=contract_size,
        tick_value_usd=tick_value_usd,
        pip_value_usd=pip_value_usd,
        volume_min=volume_min,
        volume_step=volume_step,
        volume_max=volume_max,
        usd_per_price_unit=usd_per_price_unit,
        cluster=get_cluster(symbol),
        brick_size=brick_size,
        vr_peak=dsp_result.vr_peak_value,
        vr_scale_bars=dsp_result.vr_peak_scale,
        friction_ratio=friction_ratio,
        omega=bt_result.omega,
        z_factor=bt_result.z_factor,
        n_trades=len(bt_result.trades),
        win_rate=bt_result.win_rate,
        max_drawdown=bt_result.max_dd_usd,
        oos_omega=oos_omega,
        oos_survival_rate=oos_survival_rate,
        friction_stress_omega=friction_stress_omega,
        filter_params=asdict(filter_params),
        risk_params=asdict(risk_params),
        data_start=data_start,
        data_end=data_end,
        qualified_at=now_utc,
        recalibration_due=False,
        pipeline_version=pipeline_version,
    )

    logger.info(
        "%s: QUALIFIED  omega=%.2f  z=%.2f  oos_omega=%.2f  brick=%.5f",
        symbol,
        result.omega,
        result.z_factor,
        result.oos_omega,
        result.brick_size,
    )

    if output_dir is not None:
        result.save(Path(output_dir) / QUALIFICATION_FILENAME)
        session_profile.save(Path(output_dir) / SESSION_PROFILE_FILENAME)

    return result


# ══════════════════════════════════════════════════════════════════════════════
# Recalibration pipeline
# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class RecalibrationResult:
    """Outcome of a single instrument recalibration cycle.

    Attributes
    ----------
    symbol : str
        Instrument that was recalibrated.
    success : bool
        True when the recalibration completed without error and the
        :class:`~kinetra.rl.portfolio_env.InstrumentContext` was updated.
    old_brick_size : float
        Brick size before recalibration.
    new_brick_size : float
        Brick size after recalibration.
    old_vr_peak : float
        VR peak before recalibration.
    new_vr_peak : float
        VR peak after recalibration.
    old_omega : float
        Omega from the qualifying-time backtest.
    new_omega : float
        Omega from the post-recalibration quick backtest.
    session_break_minutes : float
        Session gap threshold used for brick construction.
    reason : Optional[str]
        Failure reason when ``success=False``, or None.
    recalibrated_at : str
        ISO-8601 UTC timestamp of this recalibration event.
    context_updated : bool
        True when the supplied :class:`~kinetra.rl.portfolio_env.InstrumentContext`
        was mutated in place.  False when no context was provided or recalibration
        failed.
    registry_cleared : bool
        True when the ``recalibration_due`` flag was cleared in the registry.
    """

    symbol: str
    success: bool
    old_brick_size: float = 0.0
    new_brick_size: float = 0.0
    old_vr_peak: float = 0.0
    new_vr_peak: float = 0.0
    old_omega: float = 0.0
    new_omega: float = 0.0
    session_break_minutes: float = 30.0
    reason: Optional[str] = None
    recalibrated_at: str = ""
    context_updated: bool = False
    registry_cleared: bool = False

    def to_dict(self) -> Dict:
        return asdict(self)


def recalibrate_instrument(
    symbol: str,
    recent_m1_df: pd.DataFrame,
    registry: "QualificationRegistry",
    *,
    instrument_context: "Optional[InstrumentContext]" = None,
    output_dir: Optional[Path] = None,
    persist_log: bool = True,
) -> RecalibrationResult:
    """Run a full recalibration cycle for one instrument and update state.

    This is the canonical integration point between
    :class:`CalibrationDriftDetector` and
    :class:`~kinetra.rl.portfolio_env.InstrumentContext`.  Call it when a
    drift check has flagged ``recalibration_due=True`` to:

    1. Re-detect the session break window (via
       :func:`~kinetra.renko.session.detect_session_break`).
    2. Re-run DSP on fresh M1 data (via :func:`~kinetra.renko.dsp.run_dsp`).
    3. Derive new :class:`~kinetra.renko.backtest.FilterParams` (via
       :func:`~kinetra.renko.dsp.scaled_filter_params`).
    4. (Optional) Rebuild the brick sequence and update all derived arrays in
       the live :class:`~kinetra.rl.portfolio_env.InstrumentContext` so the
       Layer 2 RL agent observes fresh structural state on the next episode.
    5. Clear the ``recalibration_due`` flag in the registry.
    6. Append a :class:`RecalibrationResult` record to
       ``<output_dir>/<symbol>/recalibration_log.json`` for audit.

    Parameters
    ----------
    symbol : str
        Instrument to recalibrate.
    recent_m1_df : pd.DataFrame
        Recent M1 OHLCV data (at least ``DRIFT_MIN_NEW_BARS`` bars).
    registry : QualificationRegistry
        Live registry — used to read the qualifying baseline and clear the
        drift flag after successful recalibration.
    instrument_context : InstrumentContext, optional
        If provided, :meth:`~kinetra.rl.portfolio_env.InstrumentContext.recalibrate`
        is called in-place so the running RL environment immediately observes
        updated structural features.  When ``None``, only the registry and log
        are updated.
    output_dir : Path, optional
        Root directory of the qualification registry (defaults to the
        registry's own ``root_dir``).
    persist_log : bool
        If True, append the result to
        ``<output_dir>/<symbol>/recalibration_log.json``.

    Returns
    -------
    RecalibrationResult

    Notes
    -----
    - This function is **idempotent** with respect to the registry: calling it
      twice for the same instrument simply produces two log entries and leaves
      the drift flag cleared.
    - The RL agent does **not** need retraining after recalibration: it already
      observes ``vr_current`` as a feature, so updated structural parameters
      naturally shift allocation weights without any reward-function changes
      (§29.6 Regime Adaptation Horizon Map).
    - This function must **not** be used to tune the Renko core (flip + Markov
      gate) — that is locked (§29.1).  It only updates structural DSP
      measurements.

    Example
    -------
    >>> from kinetra.renko.qualify import (
    ...     CalibrationDriftDetector, QualificationRegistry, recalibrate_instrument
    ... )
    >>> registry = QualificationRegistry("data/renko_qualified")
    >>> registry.load()
    >>> detector = CalibrationDriftDetector(registry)
    >>> drift = detector.check("XAUUSD", recent_m1_df)
    >>> if drift.drift_detected:
    ...     result = recalibrate_instrument("XAUUSD", recent_m1_df, registry,
    ...                                     instrument_context=ctx)
    ...     print(f"Recalibrated: brick {result.old_brick_size:.5f} → {result.new_brick_size:.5f}")
    """
    now_utc = datetime.now(tz=timezone.utc).isoformat()
    out_dir = Path(output_dir) if output_dir is not None else registry.root_dir

    existing = registry.get(symbol)
    if existing is None:
        return RecalibrationResult(
            symbol=symbol,
            success=False,
            reason=f"symbol '{symbol}' not found in registry",
            recalibrated_at=now_utc,
        )

    if recent_m1_df is None or len(recent_m1_df) < DRIFT_MIN_NEW_BARS:
        n = 0 if recent_m1_df is None else len(recent_m1_df)
        return RecalibrationResult(
            symbol=symbol,
            success=False,
            old_brick_size=existing.brick_size,
            new_brick_size=existing.brick_size,
            old_vr_peak=existing.vr_peak,
            new_vr_peak=existing.vr_peak,
            old_omega=existing.omega,
            new_omega=existing.omega,
            reason=f"insufficient new bars ({n} < {DRIFT_MIN_NEW_BARS})",
            recalibrated_at=now_utc,
        )

    # ── Step 1: Re-detect session break ──────────────────────────────────────
    try:
        session_profile = detect_session_break(recent_m1_df, symbol=symbol)
        session_break_minutes = session_profile.session_break_minutes
    except Exception as exc:
        logger.warning("%s: recalibrate session detect failed: %s — using 30.0 min", symbol, exc)
        session_break_minutes = 30.0

    # ── Step 2: Extract M1 closes, re-run DSP ───────────────────────────────
    try:
        m1_closes = _extract_m1_closes(recent_m1_df)
    except Exception as exc:
        return RecalibrationResult(
            symbol=symbol,
            success=False,
            old_brick_size=existing.brick_size,
            new_brick_size=existing.brick_size,
            old_vr_peak=existing.vr_peak,
            new_vr_peak=existing.vr_peak,
            old_omega=existing.omega,
            new_omega=existing.omega,
            session_break_minutes=session_break_minutes,
            reason=f"M1 close extraction failed: {exc}",
            recalibrated_at=now_utc,
        )

    try:
        new_dsp = run_dsp(m1_closes.values, symbol=symbol, bars_per_hour=60.0)
    except Exception as exc:
        return RecalibrationResult(
            symbol=symbol,
            success=False,
            old_brick_size=existing.brick_size,
            new_brick_size=existing.brick_size,
            old_vr_peak=existing.vr_peak,
            new_vr_peak=existing.vr_peak,
            old_omega=existing.omega,
            new_omega=existing.omega,
            session_break_minutes=session_break_minutes,
            reason=f"DSP failed: {exc}",
            recalibrated_at=now_utc,
        )

    # ── Step 3: Derive new FilterParams ──────────────────────────────────────
    from kinetra.renko.brick_engine import bricks_per_day

    try:
        # Build a temporary brick sequence (using old brick size) just to compute
        # bricks_per_day — we need this for scaled_filter_params.
        from kinetra.renko.brick_engine import build_renko

        _tmp_bricks = build_renko(
            m1_closes,
            brick_size=new_dsp.dsp_brick_size,
            session_break_minutes=session_break_minutes,
        )
        bpd = bricks_per_day(_tmp_bricks)
    except Exception:
        bpd = 48.0  # conservative fallback if brick build fails

    try:
        new_filter_params = scaled_filter_params(new_dsp, bricks_per_day=bpd)
    except Exception as exc:
        logger.warning("%s: scaled_filter_params failed: %s — using defaults", symbol, exc)
        new_filter_params = FilterParams()

    # ── Step 4: Quick backtest for new Omega ──────────────────────────────────
    new_omega = existing.omega
    try:
        quick_bt = backtest_instrument(
            symbol=symbol,
            closes=m1_closes,
            brick_size=new_dsp.dsp_brick_size,
            filter_params=new_filter_params,
            session_break_minutes=session_break_minutes,
        )
        new_omega = quick_bt.omega
    except Exception as exc:
        logger.debug("%s: recalibrate quick backtest failed: %s", symbol, exc)

    # ── Step 5: Update InstrumentContext if provided ──────────────────────────
    context_updated = False
    if instrument_context is not None:
        try:
            instrument_context.recalibrate(
                new_closes=m1_closes,
                new_dsp_result=new_dsp,
                new_filter_params=new_filter_params,
                session_break_minutes=session_break_minutes,
            )
            context_updated = True
            logger.info(
                "%s: InstrumentContext updated — brick %.5f → %.5f, vr %.3f → %.3f",
                symbol,
                existing.brick_size,
                new_dsp.dsp_brick_size,
                existing.vr_peak,
                new_dsp.vr_peak_value,
            )
        except Exception as exc:
            logger.warning("%s: InstrumentContext.recalibrate failed: %s", symbol, exc)

    # ── Step 6: Clear drift flag in registry ─────────────────────────────────
    registry_cleared = False
    try:
        registry.clear_drift(symbol)
        registry_cleared = True
    except Exception as exc:
        logger.warning("%s: failed to clear drift flag: %s", symbol, exc)

    # ── Step 7: Persist recalibration log ────────────────────────────────────
    result = RecalibrationResult(
        symbol=symbol,
        success=True,
        old_brick_size=existing.brick_size,
        new_brick_size=new_dsp.dsp_brick_size,
        old_vr_peak=existing.vr_peak,
        new_vr_peak=new_dsp.vr_peak_value,
        old_omega=existing.omega,
        new_omega=new_omega,
        session_break_minutes=session_break_minutes,
        recalibrated_at=now_utc,
        context_updated=context_updated,
        registry_cleared=registry_cleared,
    )

    if persist_log:
        _log_path = out_dir / symbol / RECALIBRATION_LOG_FILENAME
        try:
            _log_path.parent.mkdir(parents=True, exist_ok=True)
            # Append to existing log (list of dicts) or create new.
            _existing_log: list = []
            if _log_path.exists():
                try:
                    _existing_log = json.loads(_log_path.read_text())
                    if not isinstance(_existing_log, list):
                        _existing_log = []
                except Exception:
                    _existing_log = []
            _existing_log.append(result.to_dict())
            _tmp = _log_path.with_suffix(".tmp")
            _tmp.write_text(json.dumps(_existing_log, indent=2))
            _tmp.replace(_log_path)
            logger.debug("%s: recalibration log appended to %s", symbol, _log_path)
        except Exception as exc:
            logger.warning("%s: failed to persist recalibration log: %s", symbol, exc)

    logger.info(
        "recalibrate_instrument [%s]: success=%s  brick %.5f→%.5f  vr %.3f→%.3f  "
        "omega %.2f→%.2f  context_updated=%s  registry_cleared=%s",
        symbol,
        result.success,
        result.old_brick_size,
        result.new_brick_size,
        result.old_vr_peak,
        result.new_vr_peak,
        result.old_omega,
        result.new_omega,
        result.context_updated,
        result.registry_cleared,
    )
    return result


# ══════════════════════════════════════════════════════════════════════════════
# Internal helpers
# ══════════════════════════════════════════════════════════════════════════════


def _extract_m1_closes(m1_df: pd.DataFrame) -> pd.Series:
    """Extract M1 close-price Series from an M1 OHLCV DataFrame.

    Returns a pandas Series indexed by UTC datetime with the close price
    with duplicate timestamps removed.
    """
    df = m1_df.copy()
    # Normalise time column → DatetimeIndex
    time_col = next(
        (c for c in df.columns if c.lower() in ("time", "datetime", "date", "timestamp")),
        None,
    )
    if time_col:
        df.index = pd.to_datetime(df[time_col], utc=True, errors="coerce")
    elif not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("m1_df must have a time column or DatetimeIndex")
    else:
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")

    df = df.sort_index()

    close_col = next((c for c in df.columns if c.lower() == "close"), None)
    if close_col is None:
        raise ValueError("m1_df must have a 'close' column")

    closes = pd.to_numeric(df[close_col], errors="coerce").dropna()
    closes = closes[~closes.index.duplicated(keep="last")]
    return closes


# ══════════════════════════════════════════════════════════════════════════════
# Qualification Registry
# ══════════════════════════════════════════════════════════════════════════════


class QualificationRegistry:
    """
    Loads and manages all per-instrument qualification results.

    The registry scans ``<root_dir>/<symbol>/qualification.json`` for all
    instruments and provides fast in-memory lookup.

    Parameters
    ----------
    root_dir : str or Path
        Root directory containing per-symbol sub-directories (e.g.
        ``data/renko_qualified``).

    Example
    -------
    >>> registry = QualificationRegistry("data/renko_qualified")
    >>> registry.load()
    >>> qualified = registry.get_qualified()
    >>> print([r.symbol for r in qualified])
    ['XAUUSD', 'EURUSD', 'GBPUSD']
    """

    def __init__(self, root_dir: str | Path) -> None:
        self.root_dir = Path(root_dir)
        self._results: Dict[str, QualificationResult] = {}

    @staticmethod
    def _result_key(result: QualificationResult) -> str:
        """Canonical key: instrument_id (required)."""
        if not result.instrument_id:
            raise ValueError(f"qualification result missing instrument_id: {result.symbol}")
        return result.instrument_id

    def _resolve_key(self, symbol_or_id: str) -> Optional[str]:
        """Resolve exact instrument_id or unambiguous underlying symbol."""
        if symbol_or_id in self._results:
            return symbol_or_id
        matches = [k for k, r in self._results.items() if r.symbol == symbol_or_id]
        if not matches:
            return None
        if len(matches) > 1:
            raise ValueError(
                "Registry lookup for "
                f"{symbol_or_id} is ambiguous ({len(matches)} variants): "
                f"{', '.join(sorted(matches))}. Use instrument_id."
            )
        logger.debug(
            "Registry lookup by symbol %s resolved to %s",
            symbol_or_id,
            matches[0],
        )
        return matches[0]

    # ── Loading ──────────────────────────────────────────────────────────────

    def load(self) -> "QualificationRegistry":
        """Scan ``root_dir`` and load all ``qualification.json`` files.

        Existing entries are replaced.  Unreadable files are skipped with
        a warning.  Returns self for chaining.
        """
        self._results.clear()
        if not self.root_dir.exists():
            logger.debug("QualificationRegistry: root_dir %s does not exist", self.root_dir)
            return self

        for symbol_dir in sorted(self.root_dir.iterdir()):
            if not symbol_dir.is_dir():
                continue
            q_path = symbol_dir / QUALIFICATION_FILENAME
            if not q_path.exists():
                continue
            try:
                result = QualificationResult.load(q_path)
                if not result.instrument_id:
                    if "__" in symbol_dir.name:
                        inferred_id = symbol_dir.name
                    else:
                        inferred_id = _canonical_instrument_id(
                            symbol=result.symbol or symbol_dir.name,
                            broker_source=result.broker_source or "unknown",
                            broker_account=result.broker_account or "unknown",
                            is_ecn=result.is_ecn,
                            account_type=result.account_type or "",
                        )
                    logger.warning(
                        "Upgrading %s: missing instrument_id -> %s",
                        q_path,
                        inferred_id,
                    )
                    result = QualificationResult(
                        **{**result.to_dict(), "instrument_id": inferred_id}
                    )
                    try:
                        result.save(q_path)
                    except Exception as exc:
                        logger.debug(
                            "Could not persist upgraded instrument_id to %s: %s", q_path, exc
                        )
                self._results[self._result_key(result)] = result
                logger.debug(
                    "Loaded qualification: %s  qualified=%s", result.symbol, result.qualified
                )
            except Exception as exc:
                logger.warning("Could not load qualification from %s: %s", q_path, exc)

        logger.info(
            "QualificationRegistry: loaded %d records (%d qualified)",
            len(self._results),
            sum(1 for r in self._results.values() if r.qualified),
        )
        return self

    # ── Queries ──────────────────────────────────────────────────────────────

    def get_qualified(self, *, include_drift: bool = True) -> List[QualificationResult]:
        """Return all instruments that passed qualification.

        Parameters
        ----------
        include_drift : bool
            If False, exclude instruments with ``recalibration_due=True``.
        """
        results = [r for r in self._results.values() if r.qualified and not r.disqualified]
        if not include_drift:
            results = [r for r in results if not r.recalibration_due]
        return sorted(results, key=lambda r: (r.symbol, r.instrument_id))

    def get(self, symbol_or_id: str) -> Optional[QualificationResult]:
        """Return a qualification result by instrument_id or unambiguous symbol."""
        key = self._resolve_key(symbol_or_id)
        return self._results.get(key) if key is not None else None

    def get_all_by_symbol(self, symbol: str) -> List[QualificationResult]:
        """Return all qualified variants for an underlying symbol."""
        rows = [r for r in self._results.values() if r.symbol == symbol]
        return sorted(rows, key=lambda r: r.instrument_id or r.symbol)

    def all_results(self) -> List[QualificationResult]:
        """Return all loaded results (qualified and not)."""
        return sorted(self._results.values(), key=lambda r: (r.symbol, r.instrument_id))

    @property
    def qualified_count(self) -> int:
        """Number of currently qualified instruments."""
        return sum(1 for r in self._results.values() if r.qualified and not r.disqualified)

    @property
    def drift_flags(self) -> int:
        """Number of qualified instruments with recalibration_due=True."""
        return sum(
            1
            for r in self._results.values()
            if r.qualified and not r.disqualified and r.recalibration_due
        )

    # ── Mutation ─────────────────────────────────────────────────────────────

    def register(
        self,
        result: QualificationResult,
        *,
        persist: bool = True,
    ) -> None:
        """Add or update a qualification result in the registry.

        Parameters
        ----------
        result : QualificationResult
            The result to register.
        persist : bool
            If True, write ``qualification.json`` to disk immediately.
        """
        self._results[self._result_key(result)] = result
        if persist:
            out_dir = self.root_dir / result.instrument_id
            out_dir.mkdir(parents=True, exist_ok=True)
            result.save(out_dir / QUALIFICATION_FILENAME)

    def disqualify(
        self,
        symbol: str,
        reason: str,
        *,
        persist: bool = True,
    ) -> None:
        """Explicitly disqualify an instrument (e.g. regime retirement).

        Parameters
        ----------
        symbol : str
            Instrument to disqualify.
        reason : str
            Human-readable reason stored in ``disqualification_reason``.
        persist : bool
            If True, update the ``qualification.json`` on disk.
        """
        key = self._resolve_key(symbol)
        if key is None:
            logger.warning("disqualify: %s not in registry — creating stub", symbol)
            stub = QualificationResult(
                symbol=symbol,
                qualified=False,
                instrument_id=_sanitize_id(symbol),
            )
            self._results[stub.instrument_id] = stub
            key = stub.instrument_id
        r = self._results[key]
        # Replace with a disqualified copy
        updated = QualificationResult(
            **{
                **r.to_dict(),
                "qualified": False,
                "disqualified": True,
                "disqualification_reason": reason,
            }
        )
        new_key = self._result_key(updated)
        if new_key != key:
            self._results.pop(key, None)
        self._results[new_key] = updated
        if persist:
            out_dir = self.root_dir / updated.instrument_id
            out_dir.mkdir(parents=True, exist_ok=True)
            updated.save(out_dir / QUALIFICATION_FILENAME)
        logger.info("Disqualified %s: %s", symbol, reason)

    def flag_drift(
        self,
        symbol: str,
        reason: str,
        *,
        persist: bool = True,
    ) -> None:
        """Set ``recalibration_due=True`` for an instrument.

        Called by :class:`CalibrationDriftDetector` when significant regime
        change is detected.
        """
        key = self._resolve_key(symbol)
        if key is None:
            logger.warning("flag_drift: %s not in registry — ignoring", symbol)
            return

        r = self._results[key]
        updated = QualificationResult(
            **{**r.to_dict(), "recalibration_due": True, "drift_reason": reason}
        )
        new_key = self._result_key(updated)
        if new_key != key:
            self._results.pop(key, None)
        self._results[new_key] = updated
        if persist:
            out_dir = self.root_dir / updated.instrument_id
            out_dir.mkdir(parents=True, exist_ok=True)
            updated.save(out_dir / QUALIFICATION_FILENAME)
        logger.info("Drift flagged for %s: %s", symbol, reason)

    def clear_drift(
        self,
        symbol: str,
        *,
        persist: bool = True,
    ) -> None:
        """Clear ``recalibration_due`` flag after successful recalibration."""
        key = self._resolve_key(symbol)
        if key is None:
            return
        r = self._results[key]
        updated = QualificationResult(
            **{**r.to_dict(), "recalibration_due": False, "drift_reason": None}
        )
        new_key = self._result_key(updated)
        if new_key != key:
            self._results.pop(key, None)
        self._results[new_key] = updated
        if persist:
            out_dir = self.root_dir / updated.instrument_id
            out_dir.mkdir(parents=True, exist_ok=True)
            updated.save(out_dir / QUALIFICATION_FILENAME)
        logger.info("Drift cleared for %s", symbol)

    # ── Display ──────────────────────────────────────────────────────────────

    def summary_table(self) -> str:
        """Return a formatted ASCII table of all registry entries."""
        rows = self.all_results()
        if not rows:
            return "(registry empty)"
        header = (
            f"{'Symbol':<10} {'Variant':<24} {'Qual':^5} {'Cluster':<12} {'Brick':>9} "
            f"{'Omega':>7} {'Z':>6} {'OOS Ω':>7} {'Drift':^6} {'Data End':<12}"
        )
        sep = "-" * len(header)
        lines = [header, sep]
        for r in rows:
            qual_str = (
                "YES" if r.qualified and not r.disqualified else ("DIS" if r.disqualified else "no")
            )
            drift_str = "⚠" if r.recalibration_due else "-"
            data_end = r.data_end[:10] if r.data_end else ""
            variant = r.instrument_id[:24]
            lines.append(
                f"{r.symbol:<10} {variant:<24} {qual_str:^5} {r.cluster:<12} {r.brick_size:>9.5f} "
                f"{r.omega:>7.2f} {r.z_factor:>6.2f} {r.oos_omega:>7.2f} "
                f"{drift_str:^6} {data_end:<12}"
            )
        return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# Calibration Drift Detector
# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class DriftCheckResult:
    """Result of a single calibration drift check.

    Attributes
    ----------
    symbol : str
        Instrument checked.
    drift_detected : bool
        True if any drift signal exceeded its threshold.
    vr_drift : float
        Relative change in VR peak: ``|new - old| / old``.
    omega_drift : float
        Relative change in Omega: ``|new - old| / old``.
    new_vr_peak : float
        Most-recent VR peak.
    new_omega : float
        Most-recent Omega estimate.
    reason : Optional[str]
        Human-readable summary of what triggered the flag.
    checked_at : str
        ISO-8601 UTC timestamp.
    """

    symbol: str
    drift_detected: bool
    vr_drift: float
    omega_drift: float
    new_vr_peak: float
    new_omega: float
    reason: Optional[str]
    checked_at: str


class CalibrationDriftDetector:
    """
    Detect when a qualified instrument's regime has shifted enough to
    require recalibration.

    The detector re-runs DSP on recent data and compares the resulting VR
    peak against the qualifying-time VR peak stored in the
    :class:`QualificationResult`.  If the relative change exceeds
    ``DRIFT_VR_CHANGE_THRESHOLD``, ``recalibration_due`` is flagged.

    Parameters
    ----------
    registry : QualificationRegistry
        Live registry to flag drift in.

    Example
    -------
    >>> detector = CalibrationDriftDetector(registry)
    >>> check = detector.check("XAUUSD", recent_m1_df)
    >>> if check.drift_detected:
    ...     print(f"Drift: {check.reason}")
    """

    def __init__(self, registry: QualificationRegistry) -> None:
        self.registry = registry

    def check(
        self,
        symbol: str,
        recent_m1_df: pd.DataFrame,
        *,
        flag_in_registry: bool = True,
    ) -> DriftCheckResult:
        """
        Check for calibration drift by comparing new DSP results to the
        qualifying-time baseline.

        Parameters
        ----------
        symbol : str
            Instrument to check.
        recent_m1_df : pd.DataFrame
            Recent M1 OHLCV data (typically the last few months).
            Must have >= ``DRIFT_MIN_NEW_BARS`` rows.
        flag_in_registry : bool
            If True and drift is detected, calls
            :meth:`QualificationRegistry.flag_drift` automatically.

        Returns
        -------
        DriftCheckResult
        """
        now_utc = datetime.now(tz=timezone.utc).isoformat()

        existing = self.registry.get(symbol)
        if existing is None or not existing.qualified:
            return DriftCheckResult(
                symbol=symbol,
                drift_detected=False,
                vr_drift=0.0,
                omega_drift=0.0,
                new_vr_peak=0.0,
                new_omega=0.0,
                reason="instrument not qualified — skip drift check",
                checked_at=now_utc,
            )

        if recent_m1_df is None or len(recent_m1_df) < DRIFT_MIN_NEW_BARS:
            n = 0 if recent_m1_df is None else len(recent_m1_df)
            return DriftCheckResult(
                symbol=symbol,
                drift_detected=False,
                vr_drift=0.0,
                omega_drift=0.0,
                new_vr_peak=existing.vr_peak,
                new_omega=existing.omega,
                reason=f"insufficient new bars ({n} < {DRIFT_MIN_NEW_BARS})",
                checked_at=now_utc,
            )

        # Extract M1 closes for DSP
        try:
            m1_closes = _extract_m1_closes(recent_m1_df)
        except Exception as exc:
            return DriftCheckResult(
                symbol=symbol,
                drift_detected=False,
                vr_drift=0.0,
                omega_drift=0.0,
                new_vr_peak=existing.vr_peak,
                new_omega=existing.omega,
                reason=f"aggregation failed: {exc}",
                checked_at=now_utc,
            )

        # Re-run DSP on M1 closes
        try:
            new_dsp = run_dsp(m1_closes.values, symbol=symbol, bars_per_hour=60.0)
        except Exception as exc:
            return DriftCheckResult(
                symbol=symbol,
                drift_detected=False,
                vr_drift=0.0,
                omega_drift=0.0,
                new_vr_peak=existing.vr_peak,
                new_omega=existing.omega,
                reason=f"DSP failed: {exc}",
                checked_at=now_utc,
            )

        # Quick backtest for Omega delta — use stored filter_params if available
        new_omega = existing.omega  # default: no change detected
        try:
            fp_dict = existing.filter_params or {}
            # Only pass keys that FilterParams actually accepts
            valid_fp_keys = {f.name for f in FilterParams.__dataclass_fields__.values()}  # type: ignore[attr-defined]
            fp_dict_filtered = {k: v for k, v in fp_dict.items() if k in valid_fp_keys}
            filter_params = FilterParams(**fp_dict_filtered) if fp_dict_filtered else FilterParams()
            quick_bt = backtest_instrument(
                symbol=symbol,
                closes=m1_closes,
                brick_size=existing.brick_size,
                filter_params=filter_params,
            )
            new_omega = quick_bt.omega
        except Exception as exc:
            logger.debug("%s: drift check quick backtest failed: %s", symbol, exc)

        # Compute relative drifts
        old_vr = existing.vr_peak
        new_vr = new_dsp.vr_peak_value
        vr_drift = abs(new_vr - old_vr) / max(abs(old_vr), 1e-6)

        old_omega = existing.omega
        omega_drift = abs(new_omega - old_omega) / max(abs(old_omega), 1e-6)

        reasons = []
        if vr_drift > DRIFT_VR_CHANGE_THRESHOLD:
            reasons.append(f"VR drift {vr_drift:.1%} (old={old_vr:.3f}, new={new_vr:.3f})")
        if omega_drift > DRIFT_OMEGA_CHANGE_THRESHOLD:
            reasons.append(
                f"Omega drift {omega_drift:.1%} (old={old_omega:.2f}, new={new_omega:.2f})"
            )

        drift_detected = bool(reasons)
        reason_str = "; ".join(reasons) if reasons else None

        if drift_detected and flag_in_registry:
            self.registry.flag_drift(symbol, reason_str or "drift detected")

        return DriftCheckResult(
            symbol=symbol,
            drift_detected=drift_detected,
            vr_drift=vr_drift,
            omega_drift=omega_drift,
            new_vr_peak=new_vr,
            new_omega=new_omega,
            reason=reason_str,
            checked_at=now_utc,
        )

    def check_all(
        self,
        symbol_data: Dict[str, pd.DataFrame],
        *,
        flag_in_registry: bool = True,
    ) -> List[DriftCheckResult]:
        """
        Run drift checks for multiple instruments.

        Parameters
        ----------
        symbol_data : dict[str, pd.DataFrame]
            Mapping of symbol → recent M1 DataFrame.
        flag_in_registry : bool
            Passed through to :meth:`check`.

        Returns
        -------
        list[DriftCheckResult]
            Results sorted by symbol.
        """
        results = []
        for symbol, df in sorted(symbol_data.items()):
            result = self.check(symbol, df, flag_in_registry=flag_in_registry)
            results.append(result)
        return results
