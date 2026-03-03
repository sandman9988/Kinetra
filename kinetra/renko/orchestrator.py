"""
Renko Portfolio Pipeline Orchestrator
======================================

Sprint 5C — canonical orchestrator for the full Renko Kinetra portfolio pipeline.

Chains all upstream pipeline stages for a universe of instruments and produces
a single :class:`PortfolioPipelineResult` containing per-instrument
qualification results, the assembled portfolio backtest, and a concise
summary suitable for operator review.

Pipeline stages (all idempotent / incremental):

  1.  ``qualify_instrument()``     — per-instrument: session → DSP → sweep →
                                     backtest → walk-forward → stress test
  2.  ``QualificationRegistry``    — loads / persists qualification.json files
  3.  ``build_portfolio()``        — cluster caps, equal-risk sizing, dedup
  4.  ``backtest_portfolio()``     — time-ordered trade merge, equity curve
  5.  ``monte_carlo_instrument()`` — (optional) per-instrument MC validation
  6.  ``tail_risk_analysis()``     — (optional) CVaR / tail-risk stats

Design rules (§29 AGENT_RULES_MASTER.md):
  - ❌  Never duplicate qualification logic — call ``qualify_instrument()``
  - ❌  Never assemble ``instrument_data`` manually — use ``QualificationRegistry``
  - ❌  Never use RL to calibrate brick size — DSP only
  - ✅  Persist ``results/renko/portfolio_result.json`` after pipeline run
  - ✅  Parallelism is optional and controlled by ``n_workers``

Canonical usage::

    from kinetra.renko.orchestrator import run_full_pipeline, PortfolioPipelineResult

    result = run_full_pipeline(
        m1_data={
            "XAUUSD": xauusd_df,
            "EURUSD": eurusd_df,
        },
        spread_specs={
            "XAUUSD": (1.5, 0.01),   # (spread_pts, tick_size)
            "EURUSD": (0.8, 0.00001),
        },
        output_dir=Path("data/renko_qualified"),
        results_dir=Path("results/renko"),
        n_workers=4,
    )
    print(f"Portfolio Omega: {result.portfolio_omega:.2f}")
    print(f"Qualified: {result.n_qualified}/{result.n_instruments}")

See Also:
    - ``kinetra/renko/qualify.py``    — qualify_instrument, QualificationRegistry
    - ``kinetra/renko/backtest.py``   — backtest_portfolio, monte_carlo_instrument
    - ``kinetra/renko/portfolio.py``  — build_portfolio, tail_risk_analysis
    - ``docs/MANUAL.md §Phase6`` — backtesting specification
"""

from __future__ import annotations

import json
import logging
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from kinetra.aggregation import aggregate_ohlcv
from kinetra.config import resolve_project_path
from kinetra.renko.backtest import (
    InstrumentBacktestResult,
    MonteCarloResult,
    PortfolioBacktestResult,
    backtest_portfolio,
    monte_carlo_instrument,
)
from kinetra.renko.portfolio import (
    PortfolioConfig,
    build_portfolio,
    tail_risk_analysis,
)
from kinetra.renko.qualify import (
    QualificationRegistry,
    QualificationResult,
    qualify_instrument,
)

logger = logging.getLogger(__name__)

# ── Pipeline constants ────────────────────────────────────────────────────────

# Minimum qualified instruments to attempt portfolio construction
PORTFOLIO_MIN_INSTRUMENTS: int = 3

# Minimum portfolio Omega to consider deployment-ready
PORTFOLIO_MIN_OMEGA: float = 2.0

# Minimum portfolio Z-factor for deployment
PORTFOLIO_MIN_Z: float = 5.0

# Default output location for portfolio result
DEFAULT_PORTFOLIO_RESULT_FILENAME: str = "portfolio_result.json"

# Default MC runs
DEFAULT_MC_RUNS: int = 100


def _resolve_runtime_path(path: Optional[Path], default_rel: str) -> Path:
    if path is None:
        return resolve_project_path(default_rel)
    return resolve_project_path(path)


# ══════════════════════════════════════════════════════════════════════════════
# Result dataclasses
# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class InstrumentPipelineResult:
    """Per-instrument pipeline stage result.

    Attributes
    ----------
    symbol : str
        Instrument symbol.
    qualification : QualificationResult
        Full qualification result (qualified=True / False).
    backtest : InstrumentBacktestResult or None
        Full-data backtest result.  None if qualification failed.
    mc_result : MonteCarloResult or None
        Monte Carlo result.  None if ``run_mc=False`` or qualification failed.
    error : str or None
        Exception message if the pipeline raised unexpectedly.
    elapsed_s : float
        Wall-clock seconds spent qualifying this instrument.
    """

    symbol: str
    qualification: QualificationResult
    backtest: Optional[InstrumentBacktestResult] = None
    mc_result: Optional[MonteCarloResult] = None
    error: Optional[str] = None
    elapsed_s: float = 0.0

    @property
    def qualified(self) -> bool:
        return self.qualification.qualified


@dataclass
class PortfolioPipelineResult:
    """
    Complete portfolio pipeline result.

    Attributes
    ----------
    pipeline_run_id : str
        Unique ISO-timestamp run identifier.
    n_instruments : int
        Total instruments processed.
    n_qualified : int
        Instruments that passed all qualification gates.
    n_disqualified : int
        Instruments that failed at least one gate.
    instrument_results : list[InstrumentPipelineResult]
        Per-instrument stage results.
    portfolio : PortfolioBacktestResult or None
        Portfolio-level backtest.  None if fewer than
        ``PORTFOLIO_MIN_INSTRUMENTS`` qualified.
    portfolio_omega : float
        Portfolio Omega ratio (0.0 if no portfolio).
    portfolio_z : float
        Portfolio Z-factor (0.0 if no portfolio).
    portfolio_max_dd : float
        Portfolio maximum drawdown in USD (0.0 if no portfolio).
    tail_risk : dict or None
        Output of ``tail_risk_analysis()`` if run.
    deployment_ready : bool
        True when portfolio_omega ≥ threshold and Z ≥ threshold.
    allocation_weights : dict[str, float]
        Final allocation weights used for portfolio backtest.
    elapsed_total_s : float
        Total wall-clock seconds for the pipeline run.
    errors : list[str]
        Any non-fatal error messages accumulated during the run.
    """

    pipeline_run_id: str
    n_instruments: int
    n_qualified: int
    n_disqualified: int
    instrument_results: List[InstrumentPipelineResult] = field(default_factory=list)
    portfolio: Optional[PortfolioBacktestResult] = None
    portfolio_omega: float = 0.0
    portfolio_z: float = 0.0
    portfolio_max_dd: float = 0.0
    tail_risk: Optional[dict] = None
    deployment_ready: bool = False
    allocation_weights: Dict[str, float] = field(default_factory=dict)
    elapsed_total_s: float = 0.0
    errors: List[str] = field(default_factory=list)

    # ── Persistence ───────────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        """Serialise to a JSON-safe dict."""
        return {
            "pipeline_run_id": self.pipeline_run_id,
            "n_instruments": self.n_instruments,
            "n_qualified": self.n_qualified,
            "n_disqualified": self.n_disqualified,
            "portfolio_omega": self.portfolio_omega,
            "portfolio_z": self.portfolio_z,
            "portfolio_max_dd": self.portfolio_max_dd,
            "deployment_ready": self.deployment_ready,
            "allocation_weights": self.allocation_weights,
            "elapsed_total_s": self.elapsed_total_s,
            "errors": self.errors,
            "tail_risk": self.tail_risk,
            "instruments": {
                r.symbol: {
                    "qualified": r.qualified,
                    "omega": r.qualification.omega,
                    "z_factor": r.qualification.z_factor,
                    "oos_omega": r.qualification.oos_omega,
                    "brick_size": r.qualification.brick_size,
                    "cluster": r.qualification.cluster,
                    "friction_ratio": r.qualification.friction_ratio,
                    "n_trades": r.qualification.n_trades,
                    "elapsed_s": r.elapsed_s,
                    "error": r.error,
                }
                for r in self.instrument_results
            },
        }

    def save(self, path: Path) -> None:
        """Atomically write result JSON to *path*."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.to_dict(), indent=2))
        os.replace(tmp, path)
        logger.info("Portfolio pipeline result saved → %s", path)

    @classmethod
    def load(cls, path: Path) -> "PortfolioPipelineResult":
        """Load a previously saved result (summary only — no backtest objects)."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Portfolio result not found: {path}")
        raw = json.loads(path.read_text())
        return cls(
            pipeline_run_id=raw.get("pipeline_run_id", ""),
            n_instruments=raw.get("n_instruments", 0),
            n_qualified=raw.get("n_qualified", 0),
            n_disqualified=raw.get("n_disqualified", 0),
            portfolio_omega=raw.get("portfolio_omega", 0.0),
            portfolio_z=raw.get("portfolio_z", 0.0),
            portfolio_max_dd=raw.get("portfolio_max_dd", 0.0),
            deployment_ready=raw.get("deployment_ready", False),
            allocation_weights=raw.get("allocation_weights", {}),
            elapsed_total_s=raw.get("elapsed_total_s", 0.0),
            errors=raw.get("errors", []),
            tail_risk=raw.get("tail_risk"),
        )

    # ── Summary helpers ───────────────────────────────────────────────────────

    def summary_lines(self) -> List[str]:
        """Return a list of human-readable summary lines for CLI display."""
        lines: List[str] = []
        badge = "✅ DEPLOYMENT-READY" if self.deployment_ready else "⚠️  NOT READY"
        lines.append(f"  Portfolio Pipeline — {badge}")
        lines.append(f"  Run: {self.pipeline_run_id}")
        lines.append(f"  Instruments: {self.n_qualified}/{self.n_instruments} qualified")
        if self.portfolio_omega > 0:
            lines.append(f"  Portfolio Omega:    {self.portfolio_omega:.2f}")
            lines.append(f"  Portfolio Z-factor: {self.portfolio_z:.2f}")
            lines.append(f"  Max Drawdown:       {self.portfolio_max_dd:.2f} USD")
        if self.errors:
            lines.append(f"  ⚠️  {len(self.errors)} error(s) during run:")
            for e in self.errors[:5]:
                lines.append(f"      {e}")
            if len(self.errors) > 5:
                lines.append(f"      … (+{len(self.errors) - 5} more)")
        return lines

    def print_summary(self) -> None:
        """Print a formatted summary to stdout."""
        for line in self.summary_lines():
            print(line)


# ══════════════════════════════════════════════════════════════════════════════
# Worker helper (module-level for ProcessPoolExecutor pickling)
# ══════════════════════════════════════════════════════════════════════════════


def _qualify_worker(
    args: Tuple[str, pd.DataFrame, float, float, float, float, float, str, bool, Optional[Path]],
) -> InstrumentPipelineResult:
    """
    Worker function for parallel qualification.

    Runs :func:`qualify_instrument` for a single instrument and wraps the
    result in :class:`InstrumentPipelineResult`.  Must be module-level to be
    picklable by :class:`concurrent.futures.ProcessPoolExecutor`.

    Parameters
    ----------
    args : tuple
        ``(symbol, m1_df, spread_pts, tick_size, commission_per_lot, swap_long_points,``
        `` swap_short_points, broker_source, force, output_dir)``

    Returns
    -------
    InstrumentPipelineResult
    """
    import time

    (
        symbol,
        m1_df,
        spread_pts,
        tick_size,
        commission_per_lot,
        swap_long_points,
        swap_short_points,
        broker_source,
        force,
        output_dir,
    ) = args
    t0 = time.perf_counter()
    try:
        q_result = qualify_instrument(
            symbol=symbol,
            m1_df=m1_df,
            spread_pts=spread_pts,
            tick_size=tick_size,
            commission_per_lot=commission_per_lot,
            swap_long_points=swap_long_points,
            swap_short_points=swap_short_points,
            broker_source=broker_source,
            force=force,
            output_dir=output_dir,
        )
        elapsed = time.perf_counter() - t0
        return InstrumentPipelineResult(
            symbol=symbol,
            qualification=q_result,
            elapsed_s=elapsed,
        )
    except Exception as exc:
        elapsed = time.perf_counter() - t0
        logger.error("qualify_worker %s raised: %s", symbol, exc)
        dummy_qual = QualificationResult(
            symbol=symbol,
            qualified=False,
        )
        return InstrumentPipelineResult(
            symbol=symbol,
            qualification=dummy_qual,
            error=str(exc),
            elapsed_s=elapsed,
        )


# ══════════════════════════════════════════════════════════════════════════════
# Public API
# ══════════════════════════════════════════════════════════════════════════════


def run_full_pipeline(
    m1_data: Dict[str, pd.DataFrame],
    spread_specs: Dict[str, Tuple[float, ...]],
    *,
    output_dir: Optional[Path] = None,
    results_dir: Optional[Path] = None,
    broker_source: str = "unknown",
    force: bool = False,
    n_workers: int = 1,
    run_mc: bool = False,
    mc_runs: int = DEFAULT_MC_RUNS,
    mc_seed: Optional[int] = None,
    run_tail_risk: bool = True,
    min_portfolio_instruments: int = PORTFOLIO_MIN_INSTRUMENTS,
    portfolio_min_omega: float = PORTFOLIO_MIN_OMEGA,
    portfolio_min_z: float = PORTFOLIO_MIN_Z,
    keep_closes_for_mc: bool = True,
) -> PortfolioPipelineResult:
    """
    Run the full Renko portfolio pipeline for a universe of instruments.

    Pipeline stages
    ---------------
    1. **Qualify** each instrument (parallel when ``n_workers > 1``).
    2. **Assemble portfolio** from qualified instruments using the
       canonical cluster-capped equal-risk sizing rules.
    3. **Backtest portfolio** — time-ordered equity curve.
    4. **Monte Carlo** (optional, per-instrument) for robustness stats.
    5. **Tail-risk analysis** (optional) on the portfolio equity curve.
    6. **Persist** ``portfolio_result.json`` to ``results_dir``.

    Parameters
    ----------
    m1_data : dict[str, pd.DataFrame]
        Raw M1 OHLCV DataFrames keyed by symbol.
    spread_specs : dict[str, tuple[float, ...]]
        ``{symbol: (spread_pts, tick_size)}`` or
        ``{symbol: (spread_pts, tick_size, commission_per_lot, swap_long_points, swap_short_points)}``.
    output_dir : Path or None
        Root dir for per-instrument ``qualification.json`` and
        ``session_profile.json`` files
        (default: ``data/renko_qualified``).
    results_dir : Path or None
        Directory for ``portfolio_result.json``
        (default: ``results/renko``).
    broker_source : str
        Broker identifier written into qualification files.
    force : bool
        Re-qualify even if up-to-date ``qualification.json`` exists.
    n_workers : int
        Number of parallel worker processes for qualification.
        ``1`` → sequential (easier debugging).
    run_mc : bool
        If True, run per-instrument Monte Carlo after qualification.
    mc_runs : int
        Number of Monte Carlo runs per instrument.
    mc_seed : int or None
        Random seed for Monte Carlo reproducibility.
    run_tail_risk : bool
        If True, compute tail-risk stats on the portfolio equity curve.
    min_portfolio_instruments : int
        Minimum qualified instruments required to attempt portfolio
        construction.
    portfolio_min_omega : float
        Minimum portfolio Omega ratio for ``deployment_ready`` flag.
    portfolio_min_z : float
        Minimum portfolio Z-factor for ``deployment_ready`` flag.
    keep_closes_for_mc : bool, default True
        If True and ``run_mc=True``, derive M30 close series from the
        in-memory ``m1_data`` and pass them to
        :func:`_run_per_instrument_mc_with_closes` so Monte Carlo
        actually runs.  Set to False to skip MC even when ``run_mc=True``
        (useful for memory-constrained environments).

    Returns
    -------
    PortfolioPipelineResult
        Complete pipeline result with qualification details, portfolio
        metrics, and deployment-readiness assessment.
    """
    import time

    t_pipeline_start = time.perf_counter()
    run_id = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    # ── Resolve paths ─────────────────────────────────────────────────────────
    _output_dir = _resolve_runtime_path(output_dir, "data/renko_qualified")
    _results_dir = _resolve_runtime_path(results_dir, "results/renko")

    n_instruments = len(m1_data)
    errors: List[str] = []

    logger.info(
        "run_full_pipeline [%s]: %d instruments, n_workers=%d",
        run_id,
        n_instruments,
        n_workers,
    )

    # ── Stage 1: Qualify all instruments ─────────────────────────────────────
    instrument_results: List[InstrumentPipelineResult] = []

    if n_workers > 1 and n_instruments > 1:
        instrument_results = _qualify_parallel(
            m1_data=m1_data,
            spread_specs=spread_specs,
            output_dir=_output_dir,
            broker_source=broker_source,
            force=force,
            n_workers=n_workers,
            errors=errors,
        )
    else:
        instrument_results = _qualify_sequential(
            m1_data=m1_data,
            spread_specs=spread_specs,
            output_dir=_output_dir,
            broker_source=broker_source,
            force=force,
            errors=errors,
        )

    n_qualified = sum(1 for r in instrument_results if r.qualified)
    n_disqualified = n_instruments - n_qualified

    logger.info(
        "[%s] Qualification complete: %d/%d qualified",
        run_id,
        n_qualified,
        n_instruments,
    )

    # ── Stage 2: Portfolio construction ──────────────────────────────────────
    portfolio_result: Optional[PortfolioBacktestResult] = None
    portfolio_omega = 0.0
    portfolio_z = 0.0
    portfolio_max_dd = 0.0
    allocation_weights: Dict[str, float] = {}
    tail_risk_dict: Optional[dict] = None

    if n_qualified < min_portfolio_instruments:
        msg = (
            f"Only {n_qualified}/{min_portfolio_instruments} instruments qualified — "
            f"portfolio construction skipped."
        )
        logger.warning("[%s] %s", run_id, msg)
        errors.append(msg)
    else:
        qualified_results = [r for r in instrument_results if r.qualified]
        portfolio_result, allocation_weights = _build_and_backtest_portfolio(
            qualified_results=qualified_results,
            errors=errors,
            run_id=run_id,
        )

        if portfolio_result is not None:
            portfolio_omega = portfolio_result.omega
            portfolio_z = portfolio_result.z_factor
            portfolio_max_dd = portfolio_result.max_dd_usd

            # ── Stage 5: Tail-risk analysis ───────────────────────────────
            if run_tail_risk and portfolio_result.equity_curve:
                try:
                    tail_risk_dict = _compute_tail_risk(portfolio_result, qualified_results, run_id)
                except Exception as exc:
                    msg = f"Tail-risk analysis failed: {exc}"
                    logger.warning("[%s] %s", run_id, msg)
                    errors.append(msg)

    # ── Stage 4: Monte Carlo (per-instrument, optional) ───────────────────────
    if run_mc and n_qualified > 0:
        if keep_closes_for_mc:
            # Derive M30 closes from in-memory M1 data for each qualified instrument
            instrument_closes: Dict[str, pd.Series] = _derive_m30_closes(
                m1_data=m1_data,
                symbols=[r.symbol for r in instrument_results if r.qualified],
                errors=errors,
                run_id=run_id,
            )
            if instrument_closes:
                instrument_results = _run_per_instrument_mc_with_closes(
                    instrument_results=instrument_results,
                    instrument_closes=instrument_closes,
                    n_runs=mc_runs,
                    seed=mc_seed,
                    errors=errors,
                    run_id=run_id,
                )
            else:
                # No closes could be derived — fall back to no-op stub
                logger.warning(
                    "[%s] MC requested but no M30 closes could be derived; skipping Monte Carlo.",
                    run_id,
                )
                instrument_results = _run_per_instrument_mc(
                    instrument_results=instrument_results,
                    n_runs=mc_runs,
                    seed=mc_seed,
                    errors=errors,
                    run_id=run_id,
                )
        else:
            instrument_results = _run_per_instrument_mc(
                instrument_results=instrument_results,
                n_runs=mc_runs,
                seed=mc_seed,
                errors=errors,
                run_id=run_id,
            )

    # ── Deployment readiness ──────────────────────────────────────────────────
    deployment_ready = (
        portfolio_omega >= portfolio_min_omega
        and portfolio_z >= portfolio_min_z
        and n_qualified >= min_portfolio_instruments
    )

    elapsed_total = time.perf_counter() - t_pipeline_start

    result = PortfolioPipelineResult(
        pipeline_run_id=run_id,
        n_instruments=n_instruments,
        n_qualified=n_qualified,
        n_disqualified=n_disqualified,
        instrument_results=instrument_results,
        portfolio=portfolio_result,
        portfolio_omega=portfolio_omega,
        portfolio_z=portfolio_z,
        portfolio_max_dd=portfolio_max_dd,
        tail_risk=tail_risk_dict,
        deployment_ready=deployment_ready,
        allocation_weights=allocation_weights,
        elapsed_total_s=elapsed_total,
        errors=errors,
    )

    # ── Persist result ────────────────────────────────────────────────────────
    try:
        _results_dir.mkdir(parents=True, exist_ok=True)
        result.save(_results_dir / DEFAULT_PORTFOLIO_RESULT_FILENAME)
    except Exception as exc:
        msg = f"Failed to persist portfolio result: {exc}"
        logger.error("[%s] %s", run_id, msg)
        result.errors.append(msg)

    logger.info(
        "[%s] Pipeline complete in %.1fs — omega=%.2f z=%.2f deployment=%s",
        run_id,
        elapsed_total,
        portfolio_omega,
        portfolio_z,
        deployment_ready,
    )

    return result


def run_qualification_only(
    m1_data: Dict[str, pd.DataFrame],
    spread_specs: Dict[str, Tuple[float, ...]],
    *,
    output_dir: Optional[Path] = None,
    broker_source: str = "unknown",
    force: bool = False,
    n_workers: int = 1,
) -> QualificationRegistry:
    """
    Run qualification for all instruments and return a populated registry.

    Lighter entry point than :func:`run_full_pipeline` — skips portfolio
    construction, Monte Carlo, and tail-risk analysis.

    Parameters
    ----------
    m1_data : dict[str, pd.DataFrame]
        Raw M1 OHLCV DataFrames keyed by symbol.
    spread_specs : dict[str, tuple[float, ...]]
        ``{symbol: (spread_pts, tick_size)}`` or
        ``{symbol: (spread_pts, tick_size, commission_per_lot, swap_long_points, swap_short_points)}``.
    output_dir : Path or None
        Root dir for qualification files
        (default: ``data/renko_qualified``).
    broker_source : str
        Broker identifier.
    force : bool
        Re-qualify even if up-to-date files exist.
    n_workers : int
        Number of parallel workers.

    Returns
    -------
    QualificationRegistry
        Loaded registry with all qualification results.
    """
    _output_dir = _resolve_runtime_path(output_dir, "data/renko_qualified")
    errors: List[str] = []

    if n_workers > 1 and len(m1_data) > 1:
        _qualify_parallel(
            m1_data=m1_data,
            spread_specs=spread_specs,
            output_dir=_output_dir,
            broker_source=broker_source,
            force=force,
            n_workers=n_workers,
            errors=errors,
        )
    else:
        _qualify_sequential(
            m1_data=m1_data,
            spread_specs=spread_specs,
            output_dir=_output_dir,
            broker_source=broker_source,
            force=force,
            errors=errors,
        )

    registry = QualificationRegistry(_output_dir)
    registry.load()
    return registry


def load_pipeline_result(results_dir: Optional[Path] = None) -> Optional[PortfolioPipelineResult]:
    """
    Load the most recent ``portfolio_result.json`` from *results_dir*.

    Returns None if no result file exists yet.
    """
    _results_dir = _resolve_runtime_path(results_dir, "results/renko")
    result_path = _results_dir / DEFAULT_PORTFOLIO_RESULT_FILENAME
    if not result_path.exists():
        return None
    try:
        return PortfolioPipelineResult.load(result_path)
    except Exception as exc:
        logger.warning("Could not load pipeline result from %s: %s", result_path, exc)
        return None


# ══════════════════════════════════════════════════════════════════════════════
# Private helpers
# ══════════════════════════════════════════════════════════════════════════════


def _derive_m30_closes(
    m1_data: Dict[str, pd.DataFrame],
    symbols: List[str],
    errors: List[str],
    run_id: str,
) -> Dict[str, "pd.Series"]:
    """
    Derive M30 close series from in-memory M1 DataFrames.

    Aggregates each symbol's M1 data to M30 using
    :func:`kinetra.aggregation.aggregate_ohlcv`, then extracts the
    ``close`` column as a UTC-indexed :class:`pd.Series`.

    Parameters
    ----------
    m1_data : dict[str, pd.DataFrame]
        Raw M1 OHLCV DataFrames keyed by symbol.
    symbols : list[str]
        Symbols to derive closes for (subset of m1_data keys).
    errors : list[str]
        Accumulator for non-fatal error messages.
    run_id : str
        Pipeline run identifier (for log messages).

    Returns
    -------
    dict[str, pd.Series]
        M30 close series keyed by symbol.  Symbols for which aggregation
        failed are omitted.
    """
    closes: Dict[str, pd.Series] = {}
    for symbol in symbols:
        m1_df = m1_data.get(symbol)
        if m1_df is None or m1_df.empty:
            continue
        try:
            # Ensure a datetime index before aggregation
            df = m1_df.copy()
            time_col = next(
                (c for c in df.columns if c.lower() in ("time", "datetime", "date", "timestamp")),
                None,
            )
            if time_col and not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df[time_col], utc=True, errors="coerce")

            m30_df = aggregate_ohlcv(df, "M30")
            if m30_df is None:
                logger.debug("[%s] %s: aggregate_ohlcv returned None", run_id, symbol)
                continue

            close_col = next((c for c in m30_df.columns if c.lower() == "close"), None)
            if close_col is None:
                logger.debug("[%s] %s: no 'close' column after M30 aggregation", run_id, symbol)
                continue

            # Build a DatetimeIndex-backed series from the 'time' column that
            # aggregate_ohlcv always resets into the DataFrame after resampling.
            time_col_out = next(
                (c for c in m30_df.columns if c.lower() in ("time", "datetime")),
                None,
            )
            if time_col_out is not None:
                idx = pd.to_datetime(m30_df[time_col_out], utc=True, errors="coerce")
                series = pd.Series(m30_df[close_col].values, index=idx, dtype=float).dropna()
            elif isinstance(m30_df.index, pd.DatetimeIndex):
                series = m30_df[close_col].dropna()
            else:
                logger.debug(
                    "[%s] %s: cannot construct DatetimeIndex for M30 series", run_id, symbol
                )
                continue

            if len(series) >= 50:
                closes[symbol] = series
            else:
                logger.debug(
                    "[%s] %s: only %d M30 bars after aggregation — skipping MC",
                    run_id,
                    symbol,
                    len(series),
                )
        except Exception as exc:
            msg = f"{symbol}: M1→M30 aggregation for MC failed: {exc}"
            logger.warning("[%s] %s", run_id, msg)
            errors.append(msg)

    return closes


def _unpack_spread_spec(spec: Tuple[float, ...]) -> Tuple[float, float, float, float, float]:
    """Backward-compatible unpack for per-symbol friction spec tuples."""
    spread_pts = float(spec[0]) if len(spec) >= 1 else 1.0
    tick_size = float(spec[1]) if len(spec) >= 2 else 0.0001
    commission_per_lot = float(spec[2]) if len(spec) >= 3 else 0.0
    swap_long_points = float(spec[3]) if len(spec) >= 4 else 0.0
    swap_short_points = float(spec[4]) if len(spec) >= 5 else 0.0
    return spread_pts, tick_size, commission_per_lot, swap_long_points, swap_short_points


def _qualify_sequential(
    m1_data: Dict[str, pd.DataFrame],
    spread_specs: Dict[str, Tuple[float, ...]],
    output_dir: Path,
    broker_source: str,
    force: bool,
    errors: List[str],
) -> List[InstrumentPipelineResult]:
    """Qualify instruments one at a time (main process)."""
    import time

    results: List[InstrumentPipelineResult] = []
    for symbol, m1_df in m1_data.items():
        spec = spread_specs.get(symbol, (1.0, 0.0001))
        spread_pts, tick_size, commission_per_lot, swap_long_points, swap_short_points = (
            _unpack_spread_spec(spec)
        )
        sym_dir = output_dir / symbol
        t0 = time.perf_counter()
        try:
            q_result = qualify_instrument(
                symbol=symbol,
                m1_df=m1_df,
                spread_pts=spread_pts,
                tick_size=tick_size,
                commission_per_lot=commission_per_lot,
                swap_long_points=swap_long_points,
                swap_short_points=swap_short_points,
                broker_source=broker_source,
                force=force,
                output_dir=sym_dir,
            )
            elapsed = time.perf_counter() - t0
            results.append(
                InstrumentPipelineResult(
                    symbol=symbol,
                    qualification=q_result,
                    elapsed_s=elapsed,
                )
            )
        except Exception as exc:
            elapsed = time.perf_counter() - t0
            msg = f"{symbol}: qualification raised unexpectedly: {exc}"
            logger.error(msg)
            errors.append(msg)
            dummy = QualificationResult(symbol=symbol, qualified=False)
            results.append(
                InstrumentPipelineResult(
                    symbol=symbol,
                    qualification=dummy,
                    error=str(exc),
                    elapsed_s=elapsed,
                )
            )
    return results


def _qualify_parallel(
    m1_data: Dict[str, pd.DataFrame],
    spread_specs: Dict[str, Tuple[float, ...]],
    output_dir: Path,
    broker_source: str,
    force: bool,
    n_workers: int,
    errors: List[str],
) -> List[InstrumentPipelineResult]:
    """Qualify instruments using a process pool."""
    results: List[InstrumentPipelineResult] = []
    task_args = []
    for symbol, m1_df in m1_data.items():
        spec = spread_specs.get(symbol, (1.0, 0.0001))
        spread_pts, tick_size, commission_per_lot, swap_long_points, swap_short_points = (
            _unpack_spread_spec(spec)
        )
        sym_dir = output_dir / symbol
        task_args.append(
            (
                symbol,
                m1_df,
                spread_pts,
                tick_size,
                commission_per_lot,
                swap_long_points,
                swap_short_points,
                broker_source,
                force,
                sym_dir,
            )
        )

    futures_map = {}
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        for args in task_args:
            f = executor.submit(_qualify_worker, args)
            futures_map[f] = args[0]  # symbol

    for f in as_completed(futures_map):
        symbol = futures_map[f]
        try:
            res = f.result()
            results.append(res)
            if res.error:
                errors.append(f"{symbol}: {res.error}")
        except Exception as exc:
            msg = f"{symbol}: worker process raised: {exc}"
            logger.error(msg)
            errors.append(msg)
            dummy = QualificationResult(symbol=symbol, qualified=False)
            results.append(
                InstrumentPipelineResult(
                    symbol=symbol,
                    qualification=dummy,
                    error=str(exc),
                )
            )

    # Preserve original insertion order
    order = {args[0]: i for i, args in enumerate(task_args)}
    results.sort(key=lambda r: order.get(r.symbol, 9999))
    return results


def _build_and_backtest_portfolio(
    qualified_results: List[InstrumentPipelineResult],
    errors: List[str],
    run_id: str,
) -> Tuple[Optional[PortfolioBacktestResult], Dict[str, float]]:
    """
    Assemble and backtest a portfolio from qualified instruments.

    Uses the canonical pipeline:
      - ``deduplicate_underlyings()`` — remove spot/futures duplicates
      - ``equal_risk_weights()``      — 1R sizing per instrument
      - ``apply_cluster_caps()``      — max 3 per cluster, 35% weight
      - ``backtest_portfolio()``      — time-ordered trade merge

    Returns (PortfolioBacktestResult, allocation_weights) or (None, {}).
    """
    # Build instrument_data dict in the format build_portfolio() expects:
    # {symbol: {"brick_size", "usd_per_point", "omega", "friction_ratio"}}
    instrument_data: Dict[str, dict] = {}
    for r in qualified_results:
        q = r.qualification
        instrument_data[r.symbol] = {
            "brick_size": q.brick_size or 1e-4,
            "usd_per_point": 1.0,  # conservative default — recalibrated in live
            "omega": q.omega,
            "friction_ratio": q.friction_ratio or 0.0,
        }

    # Build portfolio via canonical builder
    try:
        portfolio_construction = build_portfolio(
            instrument_data=instrument_data,
            config=PortfolioConfig(),
        )
    except Exception as exc:
        msg = f"build_portfolio failed: {exc}"
        logger.error("[%s] %s", run_id, msg)
        errors.append(msg)
        return None, {}

    allocation_weights: Dict[str, float] = portfolio_construction.allocation_weights
    cluster_map: Dict[str, str] = {
        sym: info.cluster for sym, info in portfolio_construction.sizing.items()
    }
    instrument_backtests: Dict[str, InstrumentBacktestResult] = {}

    if not instrument_backtests:
        # Fallback: run individual backtests for each qualified instrument
        logger.info("[%s] Running per-instrument backtests for portfolio...", run_id)
        for r in qualified_results:
            if r.symbol not in portfolio_construction.instruments:
                continue
            if r.backtest is not None:
                instrument_backtests[r.symbol] = r.backtest
            else:
                # closes must be provided externally for live orchestration;
                # without in-memory closes we skip gracefully and log.
                logger.debug(
                    "[%s] %s: no in-memory closes available for re-run backtest; skipping",
                    run_id,
                    r.symbol,
                )

    if not instrument_backtests:
        msg = "No instrument backtests available for portfolio construction."
        logger.warning("[%s] %s", run_id, msg)
        errors.append(msg)
        return None, {}

    # ── Portfolio-level backtest ──────────────────────────────────────────────
    try:
        portfolio_result = backtest_portfolio(
            instrument_results=instrument_backtests,
            allocation_weights=allocation_weights,
            cluster_map=cluster_map if cluster_map else None,
        )
    except Exception as exc:
        msg = f"backtest_portfolio failed: {exc}"
        logger.error("[%s] %s", run_id, msg)
        errors.append(msg)
        return None, allocation_weights

    return portfolio_result, allocation_weights


def _compute_tail_risk(
    portfolio_result: PortfolioBacktestResult,
    qualified_results: List[InstrumentPipelineResult],
    run_id: str,
) -> Optional[dict]:
    """Compute tail-risk analysis using all portfolio trades merged.

    ``tail_risk_analysis()`` in ``kinetra.renko.portfolio`` takes a flat
    sequence of trade objects with ``.exit_time`` and ``.net_usd``
    attributes.  We merge all instrument trades in time order and pass them.
    """
    try:
        # Collect all trades across instruments
        all_trades = []
        for r in qualified_results:
            if r.backtest is not None:
                all_trades.extend(r.backtest.trades)

        # Supplement with portfolio-weighted trades if per-instrument missing
        if not all_trades:
            # Nothing to analyse
            return None

        # Sort by exit time
        all_trades.sort(key=lambda t: t.exit_time)

        report = tail_risk_analysis(trades=all_trades)

        # Convert TailRiskReport dataclass to JSON-safe dict
        from dataclasses import asdict as _asdict

        raw = _asdict(report)
        safe: dict = {}
        for k, v in raw.items():
            if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
                safe[k] = None
            elif isinstance(v, np.ndarray):
                safe[k] = v.tolist()
            else:
                safe[k] = v
        return safe
    except Exception as exc:
        logger.warning("[%s] tail_risk_analysis failed: %s", run_id, exc)
        return None


def _run_per_instrument_mc(
    instrument_results: List[InstrumentPipelineResult],
    n_runs: int,
    seed: Optional[int],
    errors: List[str],
    run_id: str,
) -> List[InstrumentPipelineResult]:
    """Run Monte Carlo for each qualified instrument in-place."""
    updated: List[InstrumentPipelineResult] = []
    for r in instrument_results:
        if not r.qualified or r.backtest is None:
            updated.append(r)
            continue
        try:
            # monte_carlo_instrument needs a closes pd.Series, not an equity curve.
            # Without in-memory M1/M30 data available here we skip gracefully.
            # Callers that want per-instrument MC should pass closes via a
            # separate instrument_closes dict (future Sprint 6 enhancement).
            logger.debug(
                "[%s] Monte Carlo for %s skipped — in-memory closes not available "
                "in _run_per_instrument_mc; wire closes dict in Sprint 6.",
                run_id if "run_id" in dir() else "?",
                r.symbol,
            )
            updated.append(r)
            continue
        except Exception as exc:
            msg = f"{r.symbol}: Monte Carlo failed: {exc}"
            logger.warning("%s", msg)
            errors.append(msg)
            updated.append(r)
    return updated


def _run_per_instrument_mc_with_closes(
    instrument_results: List[InstrumentPipelineResult],
    instrument_closes: Dict[str, "pd.Series"],
    n_runs: int,
    seed: Optional[int],
    errors: List[str],
    run_id: str,
) -> List[InstrumentPipelineResult]:
    """
    Run Monte Carlo for each qualified instrument when closes are available.

    This is the full implementation used when the caller provides in-memory
    M30 close series (e.g. during a full pipeline run that keeps closes in
    memory).  The simpler ``_run_per_instrument_mc`` skips MC when closes
    are not available.

    Parameters
    ----------
    instrument_results : list[InstrumentPipelineResult]
    instrument_closes : dict[str, pd.Series]
        M30 close series keyed by symbol.
    n_runs, seed, errors, run_id : see ``_run_per_instrument_mc``.
    """
    updated: List[InstrumentPipelineResult] = []
    for r in instrument_results:
        if not r.qualified:
            updated.append(r)
            continue
        closes = instrument_closes.get(r.symbol)
        if closes is None or len(closes) < 50:
            updated.append(r)
            continue
        try:
            mc = monte_carlo_instrument(
                symbol=r.symbol,
                closes=closes,
                brick_size=r.qualification.brick_size or 1e-4,
                n_runs=n_runs,
                seed=seed,
            )
            updated.append(
                InstrumentPipelineResult(
                    symbol=r.symbol,
                    qualification=r.qualification,
                    backtest=r.backtest,
                    mc_result=mc,
                    error=r.error,
                    elapsed_s=r.elapsed_s,
                )
            )
        except Exception as exc:
            msg = f"{r.symbol}: Monte Carlo failed: {exc}"
            logger.warning("[%s] %s", run_id, msg)
            errors.append(msg)
            updated.append(r)
    return updated
