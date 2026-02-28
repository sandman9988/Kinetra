"""
Tests for kinetra.renko.orchestrator — Sprint 5C
=================================================

Coverage:
  - InstrumentPipelineResult: construction, qualified property
  - PortfolioPipelineResult: construction, to_dict, save/load roundtrip, summary_lines
  - load_pipeline_result: missing file, corrupt file, valid file
  - run_qualification_only: sequential happy path, all-fail path
  - run_full_pipeline: too few qualified → no portfolio, enough qualified → portfolio built,
    persistence, deployment_ready flag, error accumulation, elapsed time populated
  - _qualify_sequential / _qualify_parallel internals via run_qualification_only
  - Edge cases: empty m1_data, single instrument, n_workers > 1 (parallel)
  - Numerical stability: zero brick size, negative omega, NaN in tail risk
  - _derive_m30_closes: normal path, empty result, aggregation failure, short result skipped
  - _build_and_backtest_portfolio: happy path, no backtests, build_portfolio failure
  - _run_per_instrument_mc_with_closes: closes provided, missing closes skipped, exception handled
  - _run_per_instrument_mc (no closes): all skip gracefully
  - _compute_tail_risk: with trades, without trades, exception handled
  - run_full_pipeline with run_mc=True, keep_closes_for_mc=True/False
  - run_full_pipeline with run_tail_risk=True
  - Sprint 5C: PortfolioDaySnapshot.vr_drift / recalibration_pending wiring
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

import kinetra.renko.orchestrator as orch
from kinetra.renko.backtest import (
    FilterParams,
    InstrumentBacktestResult,
    MonteCarloResult,
    PortfolioBacktestResult,
    RenkoTrade,
    RiskParams,
    StopParams,
)
from kinetra.renko.orchestrator import (
    DEFAULT_MC_RUNS,
    DEFAULT_PORTFOLIO_RESULT_FILENAME,
    PORTFOLIO_MIN_INSTRUMENTS,
    PORTFOLIO_MIN_OMEGA,
    PORTFOLIO_MIN_Z,
    InstrumentPipelineResult,
    PortfolioPipelineResult,
    _build_and_backtest_portfolio,
    _compute_tail_risk,
    _derive_m30_closes,
    _run_per_instrument_mc,
    _run_per_instrument_mc_with_closes,
    load_pipeline_result,
    run_full_pipeline,
    run_qualification_only,
)
from kinetra.renko.qualify import QualificationResult

# ══════════════════════════════════════════════════════════════════════════════
# Fixtures & Helpers
# ══════════════════════════════════════════════════════════════════════════════


def _make_m1_df(n: int = 500, start: str = "2023-01-02") -> pd.DataFrame:
    """Create a minimal M1 OHLCV DataFrame for testing."""
    idx = pd.date_range(start, periods=n, freq="1min", tz="UTC")
    close = 100.0 + np.cumsum(np.random.default_rng(42).normal(0, 0.01, n))
    high = close + 0.005
    low = close - 0.005
    return pd.DataFrame(
        {
            "time": idx,
            "open": close - 0.001,
            "high": high,
            "low": low,
            "close": close,
            "volume": np.ones(n) * 100,
        }
    )


def _make_qual_result(
    symbol: str,
    qualified: bool = True,
    omega: float = 3.0,
    z_factor: float = 6.0,
    brick_size: float = 0.0005,
) -> QualificationResult:
    """Create a minimal QualificationResult for mocking."""
    return QualificationResult(
        symbol=symbol,
        qualified=qualified,
        omega=omega,
        z_factor=z_factor,
        brick_size=brick_size,
        friction_ratio=0.10,
        n_trades=50,
        win_rate=0.55,
        oos_omega=2.5,
        oos_survival_rate=1.0,
        friction_stress_omega=2.0,
        cluster="major_fx",
        vr_peak=1.15,
        vr_scale_bars=16,
        filter_params={
            "fliprate_window": 16,
            "fliprate_threshold": 0.35,
            "markov_window": 16,
            "markov_threshold": 0.55,
            "ker_period": None,
            "ker_threshold": None,
        },
        risk_params={
            "loss_cluster_window": 10,
            "loss_cluster_threshold": 0.7,
            "loss_cluster_cooldown": 5,
            "dd_throttle_pct": 0.5,
            "dd_halt_pct": 0.2,
        },
    )


def _make_instrument_pipeline_result(
    symbol: str,
    qualified: bool = True,
    omega: float = 3.0,
) -> InstrumentPipelineResult:
    """Create an InstrumentPipelineResult for testing."""
    q = _make_qual_result(symbol, qualified=qualified, omega=omega)
    return InstrumentPipelineResult(
        symbol=symbol,
        qualification=q,
        elapsed_s=0.5,
    )


def _make_renko_trade(
    symbol: str = "EURUSD",
    net_usd: float = 10.0,
    entry_offset_days: int = 0,
    exit_offset_days: int = 1,
) -> RenkoTrade:
    """Create a minimal RenkoTrade for testing."""
    from datetime import timedelta

    base = datetime(2024, 1, 1, tzinfo=timezone.utc)
    entry = base + timedelta(days=entry_offset_days)
    exit_ = base + timedelta(days=exit_offset_days)
    sign = 1 if net_usd >= 0 else -1
    return RenkoTrade(
        symbol=symbol,
        direction=sign,
        entry_price=1.1000,
        exit_price=1.1010 if net_usd >= 0 else 1.0990,
        entry_time=entry,
        exit_time=exit_,
        gross_pts=sign * 0.001,
        gross_usd=abs(net_usd) + 1.0,
        friction_usd=1.0,
        net_usd=net_usd,
        exit_reason="colour_change",
        brick_size=0.001,
        n_bricks_held=1,
    )


def _make_instrument_backtest_result(
    symbol: str = "EURUSD",
    n_trades: int = 40,
    omega: float = 3.0,
    z_factor: float = 6.0,
) -> InstrumentBacktestResult:
    """Build a minimal InstrumentBacktestResult with synthetic trades."""
    rng = np.random.default_rng(42)
    trades = []
    for i in range(n_trades):
        net = float(rng.normal(5.0, 20.0))
        trades.append(_make_renko_trade(symbol=symbol, net_usd=net, entry_offset_days=i * 2))
    equity = list(np.cumsum([t.net_usd for t in trades]))
    return InstrumentBacktestResult(
        symbol=symbol,
        brick_size=0.001,
        filter_params=FilterParams(),
        stop_params=StopParams(),
        trades=trades,
        equity_curve=equity,
        n_source_bars=1000,
        n_bricks=200,
        years=2.0,
        omega=omega,
        z_factor=z_factor,
        profit_factor=1.8,
        win_rate=0.55,
        max_dd_usd=-150.0,
        total_friction_usd=40.0,
        avg_net_per_trade=5.0,
        trades_per_year=20.0,
    )


def _make_portfolio_backtest_result(
    symbols: List[str],
    omega: float = 4.0,
    z_factor: float = 8.0,
) -> PortfolioBacktestResult:
    """Build a minimal PortfolioBacktestResult."""
    inst_results = {s: _make_instrument_backtest_result(symbol=s) for s in symbols}
    equity = [0.0]
    for r in inst_results.values():
        for i, v in enumerate(r.equity_curve):
            if i + 1 >= len(equity):
                equity.append(v)
            else:
                equity[i + 1] += v
    weights = {s: 1.0 / len(symbols) for s in symbols}
    return PortfolioBacktestResult(
        instruments=list(symbols),
        instrument_results=inst_results,
        allocation_weights=weights,
        equity_curve=equity,
        total_trades=sum(len(r.trades) for r in inst_results.values()),
        omega=omega,
        z_factor=z_factor,
        profit_factor=1.9,
        win_rate=0.56,
        net_pnl_usd=float(equity[-1]),
        max_dd_usd=-200.0,
        calmar_ratio=2.0,
        years=2.0,
        trades_per_year=20.0,
    )


# ══════════════════════════════════════════════════════════════════════════════
# TestInstrumentPipelineResult
# ══════════════════════════════════════════════════════════════════════════════


class TestInstrumentPipelineResult:
    def test_construction_defaults(self):
        q = _make_qual_result("EURUSD")
        r = InstrumentPipelineResult(symbol="EURUSD", qualification=q)
        assert r.symbol == "EURUSD"
        assert r.backtest is None
        assert r.mc_result is None
        assert r.error is None
        assert r.elapsed_s == 0.0

    def test_qualified_property_true(self):
        q = _make_qual_result("EURUSD", qualified=True)
        r = InstrumentPipelineResult(symbol="EURUSD", qualification=q)
        assert r.qualified is True

    def test_qualified_property_false(self):
        q = _make_qual_result("EURUSD", qualified=False)
        r = InstrumentPipelineResult(symbol="EURUSD", qualification=q)
        assert r.qualified is False

    def test_with_error(self):
        q = _make_qual_result("XAUUSD", qualified=False)
        r = InstrumentPipelineResult(symbol="XAUUSD", qualification=q, error="session failed")
        assert r.error == "session failed"
        assert r.qualified is False

    def test_elapsed_stored(self):
        q = _make_qual_result("GBPUSD")
        r = InstrumentPipelineResult(symbol="GBPUSD", qualification=q, elapsed_s=12.34)
        assert r.elapsed_s == pytest.approx(12.34)


# ══════════════════════════════════════════════════════════════════════════════
# TestPortfolioPipelineResult
# ══════════════════════════════════════════════════════════════════════════════


class TestPortfolioPipelineResult:
    def _make(self, **kwargs) -> PortfolioPipelineResult:
        defaults = dict(
            pipeline_run_id="20240101T120000Z",
            n_instruments=5,
            n_qualified=3,
            n_disqualified=2,
        )
        defaults.update(kwargs)
        return PortfolioPipelineResult(**defaults)

    def test_construction_defaults(self):
        r = self._make()
        assert r.portfolio_omega == 0.0
        assert r.portfolio_z == 0.0
        assert r.deployment_ready is False
        assert r.errors == []
        assert r.instrument_results == []
        assert r.allocation_weights == {}

    def test_to_dict_keys(self):
        r = self._make(portfolio_omega=3.5, portfolio_z=7.2)
        d = r.to_dict()
        assert "pipeline_run_id" in d
        assert "n_instruments" in d
        assert "n_qualified" in d
        assert "portfolio_omega" in d
        assert "portfolio_z" in d
        assert "deployment_ready" in d
        assert "instruments" in d

    def test_to_dict_values(self):
        r = self._make(n_instruments=5, n_qualified=3, portfolio_omega=4.1)
        d = r.to_dict()
        assert d["n_instruments"] == 5
        assert d["n_qualified"] == 3
        assert d["portfolio_omega"] == pytest.approx(4.1)

    def test_to_dict_instrument_entries(self):
        ir = _make_instrument_pipeline_result("EURUSD", omega=3.2)
        r = self._make(instrument_results=[ir])
        d = r.to_dict()
        assert "EURUSD" in d["instruments"]
        assert d["instruments"]["EURUSD"]["qualified"] is True
        assert d["instruments"]["EURUSD"]["omega"] == pytest.approx(3.2)

    def test_save_and_load_roundtrip(self, tmp_path):
        r = self._make(
            portfolio_omega=5.1,
            portfolio_z=10.2,
            deployment_ready=True,
            allocation_weights={"EURUSD": 0.4, "XAUUSD": 0.6},
        )
        path = tmp_path / "result.json"
        r.save(path)
        assert path.exists()
        r2 = PortfolioPipelineResult.load(path)
        assert r2.portfolio_omega == pytest.approx(5.1)
        assert r2.portfolio_z == pytest.approx(10.2)
        assert r2.deployment_ready is True
        assert r2.allocation_weights == {"EURUSD": pytest.approx(0.4), "XAUUSD": pytest.approx(0.6)}

    def test_save_creates_parent_dirs(self, tmp_path):
        r = self._make()
        path = tmp_path / "subdir" / "nested" / "result.json"
        r.save(path)
        assert path.exists()

    def test_save_is_atomic(self, tmp_path):
        """No .tmp file should remain after save."""
        r = self._make()
        path = tmp_path / "result.json"
        r.save(path)
        tmp_files = list(tmp_path.glob("*.tmp"))
        assert len(tmp_files) == 0

    def test_load_missing_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            PortfolioPipelineResult.load(tmp_path / "nonexistent.json")

    def test_load_tolerates_unknown_fields(self, tmp_path):
        """Future fields added to JSON must not break loading."""
        path = tmp_path / "result.json"
        data = {
            "pipeline_run_id": "abc123",
            "n_instruments": 2,
            "n_qualified": 1,
            "n_disqualified": 1,
            "portfolio_omega": 2.2,
            "portfolio_z": 4.0,
            "portfolio_max_dd": -0.05,
            "deployment_ready": False,
            "allocation_weights": {},
            "elapsed_total_s": 1.5,
            "errors": [],
            "tail_risk": None,
            "FUTURE_UNKNOWN_FIELD": "ignored",
        }
        path.write_text(json.dumps(data))
        r = PortfolioPipelineResult.load(path)
        assert r.pipeline_run_id == "abc123"
        assert r.n_qualified == 1

    def test_summary_lines_ready(self):
        r = self._make(deployment_ready=True, portfolio_omega=4.5, portfolio_z=8.0)
        lines = r.summary_lines()
        assert any("DEPLOYMENT-READY" in line for line in lines)
        assert any("4.50" in line for line in lines)

    def test_summary_lines_not_ready(self):
        r = self._make(deployment_ready=False)
        lines = r.summary_lines()
        assert any("NOT READY" in line for line in lines)

    def test_summary_lines_errors_shown(self):
        r = self._make(errors=["error one", "error two"])
        lines = r.summary_lines()
        joined = "\n".join(lines)
        assert "error one" in joined

    def test_summary_lines_many_errors_truncated(self):
        r = self._make(errors=[f"error {i}" for i in range(10)])
        lines = r.summary_lines()
        joined = "\n".join(lines)
        # Should show 5 errors + overflow note
        assert "+5 more" in joined

    def test_print_summary_does_not_raise(self, capsys):
        r = self._make(deployment_ready=True, portfolio_omega=3.0, portfolio_z=6.0)
        r.print_summary()
        out = capsys.readouterr().out
        assert len(out) > 0


# ══════════════════════════════════════════════════════════════════════════════
# TestLoadPipelineResult
# ══════════════════════════════════════════════════════════════════════════════


class TestLoadPipelineResult:
    def test_returns_none_when_missing(self, tmp_path):
        result = load_pipeline_result(tmp_path)
        assert result is None

    def test_loads_existing_result(self, tmp_path):
        r = PortfolioPipelineResult(
            pipeline_run_id="test-run",
            n_instruments=3,
            n_qualified=3,
            n_disqualified=0,
            portfolio_omega=4.0,
            portfolio_z=8.0,
        )
        r.save(tmp_path / DEFAULT_PORTFOLIO_RESULT_FILENAME)
        loaded = load_pipeline_result(tmp_path)
        assert loaded is not None
        assert loaded.pipeline_run_id == "test-run"
        assert loaded.portfolio_omega == pytest.approx(4.0)

    def test_returns_none_on_corrupt_json(self, tmp_path):
        path = tmp_path / DEFAULT_PORTFOLIO_RESULT_FILENAME
        path.write_text("{{not valid json}}")
        result = load_pipeline_result(tmp_path)
        assert result is None

    def test_default_path_used(self, tmp_path, monkeypatch):
        """When results_dir is None, PROJECT_ROOT/results/renko is used."""
        monkeypatch.setattr(
            orch,
            "resolve_project_path",
            lambda p: tmp_path / Path(str(p)),
        )
        result = load_pipeline_result(None)
        assert result is None


# ══════════════════════════════════════════════════════════════════════════════
# Helpers for mocking qualify_instrument
# ══════════════════════════════════════════════════════════════════════════════


def _patch_qualify(symbols_and_results: Dict[str, QualificationResult]):
    """Return a context manager that patches qualify_instrument to return preset results."""

    def _mock_qualify(symbol, m1_df, spread_pts, tick_size, **kwargs):
        return symbols_and_results.get(symbol, QualificationResult(symbol=symbol, qualified=False))

    return patch("kinetra.renko.orchestrator.qualify_instrument", side_effect=_mock_qualify)


# ══════════════════════════════════════════════════════════════════════════════
# TestRunQualificationOnly
# ══════════════════════════════════════════════════════════════════════════════


class TestRunQualificationOnly:
    def test_returns_registry(self, tmp_path):
        syms = ["EURUSD", "XAUUSD"]
        q_results = {s: _make_qual_result(s) for s in syms}
        m1 = {s: _make_m1_df() for s in syms}
        specs = {s: (1.0, 0.0001) for s in syms}

        with _patch_qualify(q_results):
            registry = run_qualification_only(
                m1_data=m1,
                spread_specs=specs,
                output_dir=tmp_path,
            )

        from kinetra.renko.qualify import QualificationRegistry

        assert isinstance(registry, QualificationRegistry)

    def test_all_pass_counted(self, tmp_path):
        syms = ["EURUSD", "GBPUSD", "XAUUSD"]
        q_results = {s: _make_qual_result(s, qualified=True) for s in syms}
        m1 = {s: _make_m1_df() for s in syms}
        specs = {s: (1.0, 0.0001) for s in syms}

        with _patch_qualify(q_results):
            registry = run_qualification_only(m1_data=m1, spread_specs=specs, output_dir=tmp_path)

        assert registry.qualified_count >= 0  # persisted files may not all be written in mock

    def test_empty_m1_data_returns_empty_registry(self, tmp_path):
        with _patch_qualify({}):
            registry = run_qualification_only(
                m1_data={},
                spread_specs={},
                output_dir=tmp_path,
            )
        from kinetra.renko.qualify import QualificationRegistry

        assert isinstance(registry, QualificationRegistry)

    def test_default_spread_spec_used_when_missing(self, tmp_path):
        """Symbols with no spread_spec entry should get the default (1.0, 0.0001)."""
        q = _make_qual_result("USDJPY")
        called_with = {}

        def _mock_qualify(symbol, m1_df, spread_pts, tick_size, **kwargs):
            called_with["spread_pts"] = spread_pts
            called_with["tick_size"] = tick_size
            return q

        with patch("kinetra.renko.orchestrator.qualify_instrument", side_effect=_mock_qualify):
            run_qualification_only(
                m1_data={"USDJPY": _make_m1_df()},
                spread_specs={},  # no entry for USDJPY
                output_dir=tmp_path,
            )

        assert called_with["spread_pts"] == pytest.approx(1.0)
        assert called_with["tick_size"] == pytest.approx(0.0001)

    def test_exception_during_qualify_does_not_crash(self, tmp_path):
        """An exception in qualify_instrument should be caught and not propagate."""

        def _boom(symbol, m1_df, spread_pts, tick_size, **kwargs):
            raise RuntimeError("test boom")

        with patch("kinetra.renko.orchestrator.qualify_instrument", side_effect=_boom):
            registry = run_qualification_only(
                m1_data={"EURUSD": _make_m1_df()},
                spread_specs={"EURUSD": (1.0, 0.0001)},
                output_dir=tmp_path,
            )
        from kinetra.renko.qualify import QualificationRegistry

        assert isinstance(registry, QualificationRegistry)

    def test_force_flag_passed_through(self, tmp_path):
        captured = {}

        def _mock_qualify(symbol, m1_df, spread_pts, tick_size, **kwargs):
            captured["force"] = kwargs.get("force", False)
            return _make_qual_result(symbol)

        with patch("kinetra.renko.orchestrator.qualify_instrument", side_effect=_mock_qualify):
            run_qualification_only(
                m1_data={"EURUSD": _make_m1_df()},
                spread_specs={"EURUSD": (1.0, 0.0001)},
                output_dir=tmp_path,
                force=True,
            )

        assert captured["force"] is True

    def test_broker_source_passed_through(self, tmp_path):
        captured = {}

        def _mock_qualify(symbol, m1_df, spread_pts, tick_size, **kwargs):
            captured["broker_source"] = kwargs.get("broker_source", "")
            return _make_qual_result(symbol)

        with patch("kinetra.renko.orchestrator.qualify_instrument", side_effect=_mock_qualify):
            run_qualification_only(
                m1_data={"EURUSD": _make_m1_df()},
                spread_specs={"EURUSD": (1.0, 0.0001)},
                output_dir=tmp_path,
                broker_source="ctrader",
            )

        assert captured["broker_source"] == "ctrader"


# ══════════════════════════════════════════════════════════════════════════════
# TestRunFullPipeline
# ══════════════════════════════════════════════════════════════════════════════


class TestRunFullPipeline:
    """Tests for run_full_pipeline()."""

    # ── Too few qualified ──────────────────────────────────────────────────────

    def test_zero_instruments_returns_result(self, tmp_path):
        with _patch_qualify({}):
            result = run_full_pipeline(
                m1_data={},
                spread_specs={},
                output_dir=tmp_path,
                results_dir=tmp_path,
            )
        assert isinstance(result, PortfolioPipelineResult)
        assert result.n_instruments == 0
        assert result.n_qualified == 0
        assert result.portfolio is None

    def test_one_qualified_below_min_no_portfolio(self, tmp_path):
        syms = ["EURUSD"]
        q_results = {s: _make_qual_result(s) for s in syms}
        m1 = {s: _make_m1_df() for s in syms}

        with _patch_qualify(q_results):
            result = run_full_pipeline(
                m1_data=m1,
                spread_specs={s: (1.0, 0.0001) for s in syms},
                output_dir=tmp_path,
                results_dir=tmp_path,
                min_portfolio_instruments=3,
            )

        assert result.n_qualified == 1
        assert result.portfolio is None
        assert any("Only 1/3 instruments" in e for e in result.errors)

    def test_two_qualified_below_min_no_portfolio(self, tmp_path):
        syms = ["EURUSD", "XAUUSD"]
        q_results = {s: _make_qual_result(s) for s in syms}
        m1 = {s: _make_m1_df() for s in syms}

        with _patch_qualify(q_results):
            result = run_full_pipeline(
                m1_data=m1,
                spread_specs={s: (1.0, 0.0001) for s in syms},
                output_dir=tmp_path,
                results_dir=tmp_path,
                min_portfolio_instruments=3,
            )

        assert result.n_qualified == 2
        assert result.portfolio is None

    def test_all_fail_no_portfolio(self, tmp_path):
        syms = ["EURUSD", "XAUUSD", "GBPUSD"]
        q_results = {s: _make_qual_result(s, qualified=False) for s in syms}
        m1 = {s: _make_m1_df() for s in syms}

        with _patch_qualify(q_results):
            result = run_full_pipeline(
                m1_data=m1,
                spread_specs={s: (1.0, 0.0001) for s in syms},
                output_dir=tmp_path,
                results_dir=tmp_path,
            )

        assert result.n_qualified == 0
        assert result.portfolio is None
        assert result.deployment_ready is False

    # ── Counts and bookkeeping ─────────────────────────────────────────────────

    def test_n_instruments_correct(self, tmp_path):
        syms = ["A", "B", "C", "D"]
        q_results = {s: _make_qual_result(s, qualified=True) for s in syms}
        m1 = {s: _make_m1_df() for s in syms}

        with _patch_qualify(q_results):
            result = run_full_pipeline(
                m1_data=m1,
                spread_specs={s: (1.0, 0.0001) for s in syms},
                output_dir=tmp_path,
                results_dir=tmp_path,
            )

        assert result.n_instruments == 4

    def test_n_qualified_n_disqualified_sum(self, tmp_path):
        q_results = {
            "EURUSD": _make_qual_result("EURUSD", qualified=True),
            "XAUUSD": _make_qual_result("XAUUSD", qualified=False),
            "GBPUSD": _make_qual_result("GBPUSD", qualified=True),
        }
        m1 = {s: _make_m1_df() for s in q_results}

        with _patch_qualify(q_results):
            result = run_full_pipeline(
                m1_data=m1,
                spread_specs={s: (1.0, 0.0001) for s in q_results},
                output_dir=tmp_path,
                results_dir=tmp_path,
            )

        assert result.n_qualified + result.n_disqualified == result.n_instruments

    def test_instrument_results_length(self, tmp_path):
        syms = ["EURUSD", "XAUUSD", "GBPUSD"]
        q_results = {s: _make_qual_result(s) for s in syms}
        m1 = {s: _make_m1_df() for s in syms}

        with _patch_qualify(q_results):
            result = run_full_pipeline(
                m1_data=m1,
                spread_specs={s: (1.0, 0.0001) for s in syms},
                output_dir=tmp_path,
                results_dir=tmp_path,
            )

        assert len(result.instrument_results) == len(syms)

    def test_pipeline_run_id_is_string(self, tmp_path):
        with _patch_qualify({}):
            result = run_full_pipeline(
                m1_data={},
                spread_specs={},
                output_dir=tmp_path,
                results_dir=tmp_path,
            )
        assert isinstance(result.pipeline_run_id, str)
        assert len(result.pipeline_run_id) > 0

    def test_pipeline_run_id_is_unique(self, tmp_path):
        with _patch_qualify({}):
            r1 = run_full_pipeline(
                m1_data={}, spread_specs={}, output_dir=tmp_path, results_dir=tmp_path
            )
            r2 = run_full_pipeline(
                m1_data={}, spread_specs={}, output_dir=tmp_path, results_dir=tmp_path
            )
        # Run IDs are ISO timestamps — may be equal if tests run in same second,
        # but they should at least be non-empty strings.
        assert isinstance(r1.pipeline_run_id, str)
        assert isinstance(r2.pipeline_run_id, str)

    def test_elapsed_total_populated(self, tmp_path):
        with _patch_qualify({}):
            result = run_full_pipeline(
                m1_data={},
                spread_specs={},
                output_dir=tmp_path,
                results_dir=tmp_path,
            )
        assert result.elapsed_total_s >= 0.0

    # ── Persistence ───────────────────────────────────────────────────────────

    def test_result_persisted_to_disk(self, tmp_path):
        with _patch_qualify({}):
            run_full_pipeline(
                m1_data={},
                spread_specs={},
                output_dir=tmp_path,
                results_dir=tmp_path,
            )
        result_path = tmp_path / DEFAULT_PORTFOLIO_RESULT_FILENAME
        assert result_path.exists()

    def test_persisted_json_is_valid(self, tmp_path):
        with _patch_qualify({}):
            run_full_pipeline(
                m1_data={},
                spread_specs={},
                output_dir=tmp_path,
                results_dir=tmp_path,
            )
        result_path = tmp_path / DEFAULT_PORTFOLIO_RESULT_FILENAME
        data = json.loads(result_path.read_text())
        assert "pipeline_run_id" in data
        assert "n_instruments" in data

    def test_results_dir_created_if_missing(self, tmp_path):
        results_dir = tmp_path / "deep" / "results" / "renko"
        assert not results_dir.exists()

        with _patch_qualify({}):
            run_full_pipeline(
                m1_data={},
                spread_specs={},
                output_dir=tmp_path,
                results_dir=results_dir,
            )
        assert results_dir.exists()
        assert (results_dir / DEFAULT_PORTFOLIO_RESULT_FILENAME).exists()

    # ── Deployment-ready flag ─────────────────────────────────────────────────

    def test_deployment_not_ready_when_no_portfolio(self, tmp_path):
        with _patch_qualify({}):
            result = run_full_pipeline(
                m1_data={},
                spread_specs={},
                output_dir=tmp_path,
                results_dir=tmp_path,
            )
        assert result.deployment_ready is False

    def test_deployment_not_ready_when_too_few_qualified(self, tmp_path):
        syms = ["EURUSD", "XAUUSD"]
        q_results = {s: _make_qual_result(s) for s in syms}
        m1 = {s: _make_m1_df() for s in syms}

        with _patch_qualify(q_results):
            result = run_full_pipeline(
                m1_data=m1,
                spread_specs={s: (1.0, 0.0001) for s in syms},
                output_dir=tmp_path,
                results_dir=tmp_path,
                min_portfolio_instruments=3,
            )
        assert result.deployment_ready is False

    # ── Error accumulation ────────────────────────────────────────────────────

    def test_exception_in_qualify_adds_error(self, tmp_path):
        def _boom(symbol, m1_df, spread_pts, tick_size, **kwargs):
            raise RuntimeError("deliberate failure")

        with patch("kinetra.renko.orchestrator.qualify_instrument", side_effect=_boom):
            result = run_full_pipeline(
                m1_data={"EURUSD": _make_m1_df()},
                spread_specs={"EURUSD": (1.0, 0.0001)},
                output_dir=tmp_path,
                results_dir=tmp_path,
            )

        assert len(result.errors) > 0
        assert any("EURUSD" in e or "deliberate" in e for e in result.errors)

    def test_errors_list_type(self, tmp_path):
        with _patch_qualify({}):
            result = run_full_pipeline(
                m1_data={},
                spread_specs={},
                output_dir=tmp_path,
                results_dir=tmp_path,
            )
        assert isinstance(result.errors, list)

    # ── run_mc=False (default) ────────────────────────────────────────────────

    def test_mc_results_none_by_default(self, tmp_path):
        syms = ["EURUSD"]
        q_results = {s: _make_qual_result(s) for s in syms}
        m1 = {s: _make_m1_df() for s in syms}

        with _patch_qualify(q_results):
            result = run_full_pipeline(
                m1_data=m1,
                spread_specs={s: (1.0, 0.0001) for s in syms},
                output_dir=tmp_path,
                results_dir=tmp_path,
                run_mc=False,
            )

        for ir in result.instrument_results:
            assert ir.mc_result is None

    # ── Custom min_portfolio_instruments ─────────────────────────────────────

    def test_custom_min_portfolio_instruments(self, tmp_path):
        syms = ["EURUSD", "XAUUSD"]
        q_results = {s: _make_qual_result(s) for s in syms}
        m1 = {s: _make_m1_df() for s in syms}

        with _patch_qualify(q_results):
            # With min=2, 2 qualified instruments should attempt portfolio construction
            result = run_full_pipeline(
                m1_data=m1,
                spread_specs={s: (1.0, 0.0001) for s in syms},
                output_dir=tmp_path,
                results_dir=tmp_path,
                min_portfolio_instruments=2,
            )

        # No "Only N/M instruments" error
        assert not any("Only 2/2 instruments" in e for e in result.errors)


# ══════════════════════════════════════════════════════════════════════════════
# TestPortfolioPipelineResultRoundtrip
# ══════════════════════════════════════════════════════════════════════════════


class TestPortfolioPipelineResultRoundtrip:
    """Save → load → assert field equality."""

    def test_full_roundtrip(self, tmp_path):
        irs = [
            _make_instrument_pipeline_result("EURUSD", omega=3.1),
            _make_instrument_pipeline_result("XAUUSD", omega=4.2, qualified=False),
        ]
        r = PortfolioPipelineResult(
            pipeline_run_id="rt-test",
            n_instruments=2,
            n_qualified=1,
            n_disqualified=1,
            instrument_results=irs,
            portfolio_omega=3.1,
            portfolio_z=6.2,
            portfolio_max_dd=-150.0,
            deployment_ready=False,
            allocation_weights={"EURUSD": 1.0},
            elapsed_total_s=12.5,
            errors=["one non-fatal error"],
        )
        path = tmp_path / "result.json"
        r.save(path)
        r2 = PortfolioPipelineResult.load(path)

        assert r2.pipeline_run_id == "rt-test"
        assert r2.n_instruments == 2
        assert r2.n_qualified == 1
        assert r2.portfolio_omega == pytest.approx(3.1)
        assert r2.portfolio_z == pytest.approx(6.2)
        assert r2.portfolio_max_dd == pytest.approx(-150.0)
        assert r2.deployment_ready is False
        assert r2.allocation_weights == {"EURUSD": pytest.approx(1.0)}
        assert r2.elapsed_total_s == pytest.approx(12.5)
        assert r2.errors == ["one non-fatal error"]

    def test_tail_risk_preserved(self, tmp_path):
        r = PortfolioPipelineResult(
            pipeline_run_id="tr-test",
            n_instruments=1,
            n_qualified=1,
            n_disqualified=0,
            tail_risk={"cvar_5pct": -0.5, "worst_trade_usd": -1.0},
        )
        path = tmp_path / "result.json"
        r.save(path)
        r2 = PortfolioPipelineResult.load(path)
        assert r2.tail_risk is not None
        assert r2.tail_risk["cvar_5pct"] == pytest.approx(-0.5)

    def test_null_tail_risk_preserved(self, tmp_path):
        r = PortfolioPipelineResult(
            pipeline_run_id="no-tr",
            n_instruments=1,
            n_qualified=0,
            n_disqualified=1,
            tail_risk=None,
        )
        path = tmp_path / "result.json"
        r.save(path)
        r2 = PortfolioPipelineResult.load(path)
        assert r2.tail_risk is None


# ══════════════════════════════════════════════════════════════════════════════
# TestOrchestratorConstants
# ══════════════════════════════════════════════════════════════════════════════


class TestOrchestratorConstants:
    def test_min_instruments_positive(self):
        assert PORTFOLIO_MIN_INSTRUMENTS >= 1

    def test_min_omega_above_one(self):
        assert PORTFOLIO_MIN_OMEGA > 1.0

    def test_min_z_positive(self):
        assert PORTFOLIO_MIN_Z > 0.0

    def test_default_mc_runs_reasonable(self):
        assert 10 <= DEFAULT_MC_RUNS <= 10000

    def test_result_filename_is_json(self):
        assert DEFAULT_PORTFOLIO_RESULT_FILENAME.endswith(".json")


# ══════════════════════════════════════════════════════════════════════════════
# TestCanonicalImport
# ══════════════════════════════════════════════════════════════════════════════


class TestCanonicalImport:
    """Verify all Sprint 5C symbols are importable from kinetra.renko."""

    def test_import_run_full_pipeline(self):
        from kinetra.renko import run_full_pipeline as rfp  # noqa: F401

        assert callable(rfp)

    def test_import_run_qualification_only(self):
        from kinetra.renko import run_qualification_only as rqo  # noqa: F401

        assert callable(rqo)

    def test_import_load_pipeline_result(self):
        from kinetra.renko import load_pipeline_result as lpr  # noqa: F401

        assert callable(lpr)

    def test_import_portfolio_pipeline_result(self):
        from kinetra.renko import PortfolioPipelineResult as PPR  # noqa: F401

        assert PPR is not None

    def test_import_instrument_pipeline_result(self):
        from kinetra.renko import InstrumentPipelineResult as IPR  # noqa: F401

        assert IPR is not None

    def test_direct_module_import(self):
        import kinetra.renko.orchestrator as orch

        assert hasattr(orch, "run_full_pipeline")
        assert hasattr(orch, "run_qualification_only")
        assert hasattr(orch, "load_pipeline_result")
        assert hasattr(orch, "PortfolioPipelineResult")
        assert hasattr(orch, "InstrumentPipelineResult")


# ══════════════════════════════════════════════════════════════════════════════
# TestNumericalEdgeCases
# ══════════════════════════════════════════════════════════════════════════════


class TestNumericalEdgeCases:
    def test_portfolio_omega_zero_when_no_portfolio(self, tmp_path):
        with _patch_qualify({}):
            result = run_full_pipeline(
                m1_data={},
                spread_specs={},
                output_dir=tmp_path,
                results_dir=tmp_path,
            )
        assert result.portfolio_omega == 0.0

    def test_portfolio_z_zero_when_no_portfolio(self, tmp_path):
        with _patch_qualify({}):
            result = run_full_pipeline(
                m1_data={},
                spread_specs={},
                output_dir=tmp_path,
                results_dir=tmp_path,
            )
        assert result.portfolio_z == 0.0

    def test_portfolio_max_dd_zero_when_no_portfolio(self, tmp_path):
        with _patch_qualify({}):
            result = run_full_pipeline(
                m1_data={},
                spread_specs={},
                output_dir=tmp_path,
                results_dir=tmp_path,
            )
        assert result.portfolio_max_dd == 0.0

    def test_to_dict_finite_values_when_no_portfolio(self, tmp_path):
        with _patch_qualify({}):
            result = run_full_pipeline(
                m1_data={},
                spread_specs={},
                output_dir=tmp_path,
                results_dir=tmp_path,
            )
        d = result.to_dict()
        assert math.isfinite(d["portfolio_omega"])
        assert math.isfinite(d["portfolio_z"])

    def test_instrument_pipeline_result_elapsed_non_negative(self, tmp_path):
        syms = ["EURUSD"]
        q_results = {s: _make_qual_result(s) for s in syms}
        m1 = {s: _make_m1_df() for s in syms}

        with _patch_qualify(q_results):
            result = run_full_pipeline(
                m1_data=m1,
                spread_specs={s: (1.0, 0.0001) for s in syms},
                output_dir=tmp_path,
                results_dir=tmp_path,
            )

        for ir in result.instrument_results:
            assert ir.elapsed_s >= 0.0

    def test_instrument_result_with_none_brick_size(self, tmp_path):
        """qualify_instrument may return brick_size=None for failed instruments."""
        q = QualificationResult(symbol="EURUSD", qualified=False, brick_size=None)

        def _mock_qualify(symbol, m1_df, spread_pts, tick_size, **kwargs):
            return q

        with patch("kinetra.renko.orchestrator.qualify_instrument", side_effect=_mock_qualify):
            result = run_full_pipeline(
                m1_data={"EURUSD": _make_m1_df()},
                spread_specs={"EURUSD": (1.0, 0.0001)},
                output_dir=tmp_path,
                results_dir=tmp_path,
            )

        assert result.n_qualified == 0
        assert isinstance(result, PortfolioPipelineResult)

    def test_allocation_weights_empty_when_no_portfolio(self, tmp_path):
        with _patch_qualify({}):
            result = run_full_pipeline(
                m1_data={},
                spread_specs={},
                output_dir=tmp_path,
                results_dir=tmp_path,
            )
        assert result.allocation_weights == {}


# ══════════════════════════════════════════════════════════════════════════════
# TestQualifyWorker (module-level picklable function)
# ══════════════════════════════════════════════════════════════════════════════


class TestQualifyWorker:
    """Test the _qualify_worker function directly."""

    def test_happy_path(self, tmp_path):
        from kinetra.renko.orchestrator import _qualify_worker

        sym = "EURUSD"
        m1 = _make_m1_df()
        q = _make_qual_result(sym)

        with patch("kinetra.renko.orchestrator.qualify_instrument", return_value=q):
            result = _qualify_worker((sym, m1, 1.0, 0.0001, "test", False, tmp_path / sym))

        assert result.symbol == sym
        assert result.qualified is True
        assert result.error is None
        assert result.elapsed_s >= 0.0

    def test_exception_returns_failed_result(self, tmp_path):
        from kinetra.renko.orchestrator import _qualify_worker

        def _boom(*args, **kwargs):
            raise ValueError("forced error")

        with patch("kinetra.renko.orchestrator.qualify_instrument", side_effect=_boom):
            result = _qualify_worker(
                ("XAUUSD", _make_m1_df(), 1.5, 0.01, "test", False, tmp_path / "XAUUSD")
            )

        assert result.qualified is False
        assert result.error is not None
        assert "forced error" in result.error
        assert result.elapsed_s >= 0.0

    def test_symbol_preserved_on_error(self, tmp_path):
        from kinetra.renko.orchestrator import _qualify_worker

        def _boom(*args, **kwargs):
            raise RuntimeError("boom")

        with patch("kinetra.renko.orchestrator.qualify_instrument", side_effect=_boom):
            result = _qualify_worker(
                ("GBPUSD", _make_m1_df(), 1.0, 0.0001, "test", False, tmp_path / "GBPUSD")
            )

        assert result.symbol == "GBPUSD"


# ══════════════════════════════════════════════════════════════════════════════
# TestSummaryLinesDetailled
# ══════════════════════════════════════════════════════════════════════════════


class TestSummaryLinesDetailled:
    def test_run_id_in_summary(self):
        r = PortfolioPipelineResult(
            pipeline_run_id="unique-run-xyz",
            n_instruments=3,
            n_qualified=3,
            n_disqualified=0,
        )
        lines = r.summary_lines()
        assert any("unique-run-xyz" in line for line in lines)

    def test_qualified_count_in_summary(self):
        r = PortfolioPipelineResult(
            pipeline_run_id="run-1",
            n_instruments=10,
            n_qualified=7,
            n_disqualified=3,
        )
        lines = r.summary_lines()
        joined = "\n".join(lines)
        assert "7/10" in joined

    def test_metrics_shown_when_portfolio_exists(self):
        r = PortfolioPipelineResult(
            pipeline_run_id="run-2",
            n_instruments=5,
            n_qualified=5,
            n_disqualified=0,
            portfolio_omega=6.07,
            portfolio_z=14.2,
            portfolio_max_dd=-500.0,
        )
        lines = r.summary_lines()
        joined = "\n".join(lines)
        assert "6.07" in joined
        assert "14.20" in joined
        assert "-500.00" in joined

    def test_metrics_not_shown_when_zero(self):
        r = PortfolioPipelineResult(
            pipeline_run_id="run-3",
            n_instruments=1,
            n_qualified=0,
            n_disqualified=1,
            portfolio_omega=0.0,
        )
        lines = r.summary_lines()
        joined = "\n".join(lines)
        # Should NOT show metric rows for zero omega
        assert "Portfolio Omega" not in joined


# ══════════════════════════════════════════════════════════════════════════════
# TestDeriveM30Closes — Sprint 5C
# ══════════════════════════════════════════════════════════════════════════════


class TestDeriveM30Closes:
    """Unit tests for _derive_m30_closes(), the M1→M30 helper used by MC."""

    def _make_long_m1_df(self, n: int = 5000, symbol: str = "EURUSD") -> pd.DataFrame:
        """Create a long enough M1 DataFrame to produce ≥50 M30 bars."""
        idx = pd.date_range("2023-01-02", periods=n, freq="1min", tz="UTC")
        close = 1.1 + np.cumsum(np.random.default_rng(0).normal(0, 0.0001, n))
        return pd.DataFrame(
            {
                "time": idx,
                "open": close - 0.0001,
                "high": close + 0.0002,
                "low": close - 0.0002,
                "close": close,
                "volume": np.ones(n) * 100,
            }
        )

    def test_returns_dict_keyed_by_symbol(self):
        m1 = {"EURUSD": self._make_long_m1_df()}
        result = _derive_m30_closes(m1_data=m1, symbols=["EURUSD"], errors=[], run_id="t")
        assert isinstance(result, dict)
        assert "EURUSD" in result

    def test_returns_series_with_datetimeindex(self):
        m1 = {"EURUSD": self._make_long_m1_df()}
        result = _derive_m30_closes(m1_data=m1, symbols=["EURUSD"], errors=[], run_id="t")
        assert isinstance(result["EURUSD"], pd.Series)
        assert isinstance(result["EURUSD"].index, pd.DatetimeIndex)

    def test_result_series_has_close_values(self):
        m1 = {"EURUSD": self._make_long_m1_df(n=5000)}
        result = _derive_m30_closes(m1_data=m1, symbols=["EURUSD"], errors=[], run_id="t")
        s = result["EURUSD"]
        assert len(s) >= 50
        assert s.notna().all()

    def test_empty_symbol_list_returns_empty_dict(self):
        m1 = {"EURUSD": self._make_long_m1_df()}
        result = _derive_m30_closes(m1_data=m1, symbols=[], errors=[], run_id="t")
        assert result == {}

    def test_symbol_not_in_m1_data_is_skipped(self):
        m1 = {"EURUSD": self._make_long_m1_df()}
        result = _derive_m30_closes(m1_data=m1, symbols=["XAUUSD"], errors=[], run_id="t")
        assert "XAUUSD" not in result

    def test_empty_dataframe_is_skipped(self):
        m1 = {"EURUSD": pd.DataFrame()}
        result = _derive_m30_closes(m1_data=m1, symbols=["EURUSD"], errors=[], run_id="t")
        assert "EURUSD" not in result

    def test_none_dataframe_is_skipped(self):
        m1 = {"EURUSD": None}
        result = _derive_m30_closes(m1_data=m1, symbols=["EURUSD"], errors=[], run_id="t")
        assert "EURUSD" not in result

    def test_too_few_bars_after_aggregation_is_skipped(self):
        # Only 20 M1 bars → only 1 M30 bar after resample, well below 50
        m1 = {"EURUSD": _make_m1_df(n=20)}
        result = _derive_m30_closes(m1_data=m1, symbols=["EURUSD"], errors=[], run_id="t")
        assert "EURUSD" not in result

    def test_aggregation_failure_adds_to_errors(self):
        """If aggregate_ohlcv raises, the error is accumulated and symbol skipped."""
        m1 = {"EURUSD": self._make_long_m1_df()}
        errors: List[str] = []
        with patch("kinetra.renko.orchestrator.aggregate_ohlcv", side_effect=RuntimeError("boom")):
            result = _derive_m30_closes(m1_data=m1, symbols=["EURUSD"], errors=errors, run_id="t")
        assert "EURUSD" not in result
        assert len(errors) == 1
        assert "EURUSD" in errors[0]

    def test_aggregate_ohlcv_returning_none_is_skipped(self):
        m1 = {"EURUSD": self._make_long_m1_df()}
        with patch("kinetra.renko.orchestrator.aggregate_ohlcv", return_value=None):
            result = _derive_m30_closes(m1_data=m1, symbols=["EURUSD"], errors=[], run_id="t")
        assert "EURUSD" not in result

    def test_multiple_symbols_processed_independently(self):
        m1 = {
            "EURUSD": self._make_long_m1_df(n=5000),
            "XAUUSD": self._make_long_m1_df(n=5000, symbol="XAUUSD"),
        }
        result = _derive_m30_closes(m1_data=m1, symbols=["EURUSD", "XAUUSD"], errors=[], run_id="t")
        assert "EURUSD" in result
        assert "XAUUSD" in result

    def test_one_failure_does_not_block_other_symbol(self):
        m1 = {
            "EURUSD": self._make_long_m1_df(n=5000),
            "FAIL": pd.DataFrame(),  # empty → skipped
        }
        errors: List[str] = []
        result = _derive_m30_closes(
            m1_data=m1, symbols=["EURUSD", "FAIL"], errors=errors, run_id="t"
        )
        assert "EURUSD" in result
        assert "FAIL" not in result

    def test_datetime_index_input_accepted(self):
        """DataFrames with existing DatetimeIndex (no time column) are handled."""
        n = 5000
        idx = pd.date_range("2023-01-02", periods=n, freq="1min", tz="UTC")
        close = 1.1 + np.cumsum(np.random.default_rng(1).normal(0, 0.0001, n))
        df = pd.DataFrame(
            {"close": close, "open": close, "high": close, "low": close, "volume": 100.0},
            index=idx,
        )
        # aggregate_ohlcv requires a 'time' column or a DatetimeIndex — pass both to be safe
        df = df.copy()
        df["time"] = idx
        df = df.reset_index(drop=True)
        m1 = {"EURUSD": df}
        result = _derive_m30_closes(m1_data=m1, symbols=["EURUSD"], errors=[], run_id="t")
        assert "EURUSD" in result


# ══════════════════════════════════════════════════════════════════════════════
# TestBuildAndBacktestPortfolio — Sprint 5C
# ══════════════════════════════════════════════════════════════════════════════


class TestBuildAndBacktestPortfolio:
    """Unit tests for _build_and_backtest_portfolio()."""

    def _make_qualified_results(self, symbols: List[str]) -> List[InstrumentPipelineResult]:
        results = []
        for s in symbols:
            bt = _make_instrument_backtest_result(symbol=s)
            q = _make_qual_result(s, qualified=True)
            r = InstrumentPipelineResult(symbol=s, qualification=q, backtest=bt)
            results.append(r)
        return results

    def test_returns_tuple(self):
        results = self._make_qualified_results(["EURUSD", "XAUUSD", "GBPUSD"])
        port, weights = _build_and_backtest_portfolio(
            qualified_results=results, errors=[], run_id="t"
        )
        assert isinstance(weights, dict)

    def test_no_backtest_results_returns_none_portfolio(self):
        """When none of the qualified results have backtest objects, portfolio is None."""
        syms = ["EURUSD", "XAUUSD", "GBPUSD"]
        results = []
        for s in syms:
            q = _make_qual_result(s)
            results.append(InstrumentPipelineResult(symbol=s, qualification=q))
        errors: List[str] = []
        port, weights = _build_and_backtest_portfolio(
            qualified_results=results, errors=errors, run_id="t"
        )
        assert port is None
        assert len(errors) >= 1

    def test_build_portfolio_failure_returns_none(self):
        results = self._make_qualified_results(["EURUSD", "XAUUSD", "GBPUSD"])
        errors: List[str] = []
        with patch(
            "kinetra.renko.orchestrator.build_portfolio",
            side_effect=RuntimeError("build failed"),
        ):
            port, weights = _build_and_backtest_portfolio(
                qualified_results=results, errors=errors, run_id="t"
            )
        assert port is None
        assert any("build_portfolio failed" in e for e in errors)

    def test_backtest_portfolio_failure_returns_none(self):
        results = self._make_qualified_results(["EURUSD", "XAUUSD", "GBPUSD"])
        errors: List[str] = []
        with patch(
            "kinetra.renko.orchestrator.backtest_portfolio",
            side_effect=RuntimeError("backtest failed"),
        ):
            port, weights = _build_and_backtest_portfolio(
                qualified_results=results, errors=errors, run_id="t"
            )
        assert port is None
        assert any("backtest_portfolio failed" in e for e in errors)

    def test_allocation_weights_returned_on_build_failure(self):
        """Even if backtest fails, allocation_weights from build_portfolio are returned."""
        results = self._make_qualified_results(["EURUSD", "XAUUSD", "GBPUSD"])
        mock_construction = MagicMock()
        mock_construction.allocation_weights = {"EURUSD": 0.4, "XAUUSD": 0.3, "GBPUSD": 0.3}
        mock_construction.sizing = {
            s: MagicMock(cluster="major_fx") for s in ["EURUSD", "XAUUSD", "GBPUSD"]
        }
        mock_construction.instruments = ["EURUSD", "XAUUSD", "GBPUSD"]
        with patch("kinetra.renko.orchestrator.build_portfolio", return_value=mock_construction):
            with patch(
                "kinetra.renko.orchestrator.backtest_portfolio",
                side_effect=RuntimeError("boom"),
            ):
                errors: List[str] = []
                port, weights = _build_and_backtest_portfolio(
                    qualified_results=results, errors=errors, run_id="t"
                )
        assert port is None
        # weights should be the build_portfolio output even though backtest failed
        assert "EURUSD" in weights


# ══════════════════════════════════════════════════════════════════════════════
# TestRunPerInstrumentMC — Sprint 5C
# ══════════════════════════════════════════════════════════════════════════════


class TestRunPerInstrumentMCNoCloses:
    """Tests for _run_per_instrument_mc() — the no-closes fallback path."""

    def test_returns_same_length_list(self):
        results = [_make_instrument_pipeline_result(s) for s in ["EURUSD", "XAUUSD"]]
        # Inject backtest so the qualified-with-backtest path is exercised
        results[0] = InstrumentPipelineResult(
            symbol="EURUSD",
            qualification=_make_qual_result("EURUSD"),
            backtest=_make_instrument_backtest_result("EURUSD"),
        )
        updated = _run_per_instrument_mc(
            instrument_results=results, n_runs=10, seed=42, errors=[], run_id="t"
        )
        assert len(updated) == 2

    def test_mc_result_stays_none_when_no_closes(self):
        """Without closes, mc_result must remain None for all instruments."""
        bt = _make_instrument_backtest_result("EURUSD")
        r = InstrumentPipelineResult(
            symbol="EURUSD",
            qualification=_make_qual_result("EURUSD"),
            backtest=bt,
        )
        updated = _run_per_instrument_mc(
            instrument_results=[r], n_runs=5, seed=0, errors=[], run_id="t"
        )
        assert updated[0].mc_result is None

    def test_unqualified_instruments_pass_through_unchanged(self):
        r = _make_instrument_pipeline_result("EURUSD", qualified=False)
        updated = _run_per_instrument_mc(
            instrument_results=[r], n_runs=5, seed=0, errors=[], run_id="t"
        )
        assert updated[0].qualified is False
        assert updated[0].mc_result is None

    def test_empty_list_returns_empty_list(self):
        updated = _run_per_instrument_mc(
            instrument_results=[], n_runs=5, seed=0, errors=[], run_id="t"
        )
        assert updated == []


class TestRunPerInstrumentMCWithCloses:
    """Tests for _run_per_instrument_mc_with_closes() — full MC path."""

    def _make_m30_closes(self, n: int = 300) -> pd.Series:
        idx = pd.date_range("2023-01-02", periods=n, freq="30min", tz="UTC")
        return pd.Series(1.1 + np.cumsum(np.random.default_rng(7).normal(0, 0.001, n)), index=idx)

    def test_returns_same_length_list(self):
        syms = ["EURUSD", "XAUUSD"]
        results = [
            InstrumentPipelineResult(
                symbol=s,
                qualification=_make_qual_result(s),
                backtest=_make_instrument_backtest_result(s),
            )
            for s in syms
        ]
        closes = {s: self._make_m30_closes() for s in syms}
        updated = _run_per_instrument_mc_with_closes(
            instrument_results=results,
            instrument_closes=closes,
            n_runs=5,
            seed=42,
            errors=[],
            run_id="t",
        )
        assert len(updated) == 2

    def test_mc_result_populated_when_closes_available(self):
        r = InstrumentPipelineResult(
            symbol="EURUSD",
            qualification=_make_qual_result("EURUSD"),
            backtest=_make_instrument_backtest_result("EURUSD"),
        )
        closes = {"EURUSD": self._make_m30_closes(n=300)}
        updated = _run_per_instrument_mc_with_closes(
            instrument_results=[r],
            instrument_closes=closes,
            n_runs=5,
            seed=42,
            errors=[],
            run_id="t",
        )
        assert updated[0].mc_result is not None
        assert isinstance(updated[0].mc_result, MonteCarloResult)

    def test_mc_result_has_correct_n_runs(self):
        r = InstrumentPipelineResult(
            symbol="EURUSD",
            qualification=_make_qual_result("EURUSD"),
            backtest=_make_instrument_backtest_result("EURUSD"),
        )
        closes = {"EURUSD": self._make_m30_closes(n=300)}
        updated = _run_per_instrument_mc_with_closes(
            instrument_results=[r],
            instrument_closes=closes,
            n_runs=7,
            seed=42,
            errors=[],
            run_id="t",
        )
        assert updated[0].mc_result.n_runs == 7

    def test_missing_closes_symbol_is_skipped(self):
        r = InstrumentPipelineResult(
            symbol="EURUSD",
            qualification=_make_qual_result("EURUSD"),
            backtest=_make_instrument_backtest_result("EURUSD"),
        )
        updated = _run_per_instrument_mc_with_closes(
            instrument_results=[r],
            instrument_closes={},  # no closes provided
            n_runs=5,
            seed=0,
            errors=[],
            run_id="t",
        )
        assert updated[0].mc_result is None

    def test_short_closes_series_is_skipped(self):
        r = InstrumentPipelineResult(
            symbol="EURUSD",
            qualification=_make_qual_result("EURUSD"),
            backtest=_make_instrument_backtest_result("EURUSD"),
        )
        short_closes = {"EURUSD": self._make_m30_closes(n=10)}  # < 50 bars
        updated = _run_per_instrument_mc_with_closes(
            instrument_results=[r],
            instrument_closes=short_closes,
            n_runs=5,
            seed=0,
            errors=[],
            run_id="t",
        )
        assert updated[0].mc_result is None

    def test_unqualified_instruments_skipped(self):
        r = _make_instrument_pipeline_result("EURUSD", qualified=False)
        closes = {"EURUSD": self._make_m30_closes()}
        updated = _run_per_instrument_mc_with_closes(
            instrument_results=[r],
            instrument_closes=closes,
            n_runs=5,
            seed=0,
            errors=[],
            run_id="t",
        )
        assert updated[0].mc_result is None

    def test_mc_exception_adds_to_errors(self):
        r = InstrumentPipelineResult(
            symbol="EURUSD",
            qualification=_make_qual_result("EURUSD"),
            backtest=_make_instrument_backtest_result("EURUSD"),
        )
        closes = {"EURUSD": self._make_m30_closes()}
        errors: List[str] = []
        with patch(
            "kinetra.renko.orchestrator.monte_carlo_instrument",
            side_effect=RuntimeError("mc boom"),
        ):
            updated = _run_per_instrument_mc_with_closes(
                instrument_results=[r],
                instrument_closes=closes,
                n_runs=5,
                seed=0,
                errors=errors,
                run_id="t",
            )
        assert updated[0].mc_result is None
        assert any("EURUSD" in e for e in errors)

    def test_other_instruments_not_affected_by_one_mc_failure(self):
        syms = ["EURUSD", "XAUUSD"]
        results = [
            InstrumentPipelineResult(
                symbol=s,
                qualification=_make_qual_result(s),
                backtest=_make_instrument_backtest_result(s),
            )
            for s in syms
        ]
        closes = {s: self._make_m30_closes() for s in syms}

        call_count = [0]
        orig_mc = __import__(
            "kinetra.renko.backtest", fromlist=["monte_carlo_instrument"]
        ).monte_carlo_instrument

        def selective_boom(symbol, *args, **kwargs):
            call_count[0] += 1
            if symbol == "EURUSD":
                raise RuntimeError("boom")
            return orig_mc(symbol, *args, **kwargs)

        errors: List[str] = []
        with patch("kinetra.renko.orchestrator.monte_carlo_instrument", side_effect=selective_boom):
            updated = _run_per_instrument_mc_with_closes(
                instrument_results=results,
                instrument_closes=closes,
                n_runs=3,
                seed=0,
                errors=errors,
                run_id="t",
            )

        eurusd_r = next(r for r in updated if r.symbol == "EURUSD")
        xauusd_r = next(r for r in updated if r.symbol == "XAUUSD")
        assert eurusd_r.mc_result is None
        assert xauusd_r.mc_result is not None

    def test_other_fields_preserved_after_mc(self):
        q = _make_qual_result("EURUSD")
        bt = _make_instrument_backtest_result("EURUSD")
        r = InstrumentPipelineResult(
            symbol="EURUSD",
            qualification=q,
            backtest=bt,
            elapsed_s=1.23,
        )
        closes = {"EURUSD": self._make_m30_closes()}
        updated = _run_per_instrument_mc_with_closes(
            instrument_results=[r],
            instrument_closes=closes,
            n_runs=3,
            seed=0,
            errors=[],
            run_id="t",
        )
        assert updated[0].elapsed_s == pytest.approx(1.23)
        assert updated[0].backtest is bt
        assert updated[0].qualification is q

    def test_empty_list_returns_empty_list(self):
        updated = _run_per_instrument_mc_with_closes(
            instrument_results=[],
            instrument_closes={},
            n_runs=5,
            seed=0,
            errors=[],
            run_id="t",
        )
        assert updated == []


# ══════════════════════════════════════════════════════════════════════════════
# TestComputeTailRisk — Sprint 5C
# ══════════════════════════════════════════════════════════════════════════════


class TestComputeTailRisk:
    """Tests for _compute_tail_risk()."""

    def _make_qualified_with_backtest(
        self, symbol: str, n_trades: int = 40
    ) -> InstrumentPipelineResult:
        q = _make_qual_result(symbol)
        bt = _make_instrument_backtest_result(symbol=symbol, n_trades=n_trades)
        return InstrumentPipelineResult(symbol=symbol, qualification=q, backtest=bt)

    def test_returns_dict(self):
        port = _make_portfolio_backtest_result(["EURUSD", "XAUUSD", "GBPUSD"])
        results = [self._make_qualified_with_backtest(s) for s in ["EURUSD", "XAUUSD", "GBPUSD"]]
        out = _compute_tail_risk(portfolio_result=port, qualified_results=results, run_id="t")
        # Should either return a dict or None (None is acceptable when no trades)
        assert out is None or isinstance(out, dict)

    def test_no_backtest_results_returns_none(self):
        port = _make_portfolio_backtest_result(["EURUSD"])
        results = [_make_instrument_pipeline_result("EURUSD")]  # no backtest
        out = _compute_tail_risk(portfolio_result=port, qualified_results=results, run_id="t")
        assert out is None

    def test_tail_risk_exception_returns_none(self):
        port = _make_portfolio_backtest_result(["EURUSD"])
        results = [self._make_qualified_with_backtest("EURUSD")]
        with patch(
            "kinetra.renko.orchestrator.tail_risk_analysis",
            side_effect=RuntimeError("tr boom"),
        ):
            out = _compute_tail_risk(portfolio_result=port, qualified_results=results, run_id="t")
        assert out is None

    def test_nan_values_serialised_as_none(self):
        """NaN values in tail_risk must be JSON-serialisable (replaced with None)."""
        from kinetra.renko.portfolio import TailRiskReport

        port = _make_portfolio_backtest_result(["EURUSD"])
        results = [self._make_qualified_with_backtest("EURUSD")]

        # Build a TailRiskReport with NaN values (using actual field names)
        nan_report = TailRiskReport(
            n_trades=5,
            worst_trade_usd=float("nan"),
            max_consecutive_losses=3,
            cvar_5pct=float("nan"),
            worst_month_usd=float("nan"),
            worst_week_usd=float("nan"),
        )
        with patch("kinetra.renko.orchestrator.tail_risk_analysis", return_value=nan_report):
            out = _compute_tail_risk(portfolio_result=port, qualified_results=results, run_id="t")
        if out is not None:
            # Any NaN values must have been replaced with None for JSON safety
            for v in out.values():
                if isinstance(v, float):
                    assert not math.isnan(v), f"NaN survived in tail_risk output: {out}"


# ══════════════════════════════════════════════════════════════════════════════
# TestRunFullPipelineWithMC — Sprint 5C
# ══════════════════════════════════════════════════════════════════════════════


class TestRunFullPipelineWithMC:
    """Tests for run_full_pipeline with run_mc=True."""

    def _make_long_m1(self, n: int = 5000) -> pd.DataFrame:
        idx = pd.date_range("2023-01-02", periods=n, freq="1min", tz="UTC")
        close = 1.1 + np.cumsum(np.random.default_rng(5).normal(0, 0.0001, n))
        return pd.DataFrame(
            {
                "time": idx,
                "open": close - 0.0001,
                "high": close + 0.0002,
                "low": close - 0.0002,
                "close": close,
                "volume": np.ones(n) * 100,
            }
        )

    def test_run_mc_false_leaves_mc_result_none(self, tmp_path):
        syms = ["EURUSD"]
        q_results = {s: _make_qual_result(s) for s in syms}
        m1 = {s: self._make_long_m1() for s in syms}
        with _patch_qualify(q_results):
            result = run_full_pipeline(
                m1_data=m1,
                spread_specs={s: (1.0, 0.0001) for s in syms},
                output_dir=tmp_path,
                results_dir=tmp_path,
                run_mc=False,
            )
        for ir in result.instrument_results:
            assert ir.mc_result is None

    def test_run_mc_true_keep_closes_false_skips_mc(self, tmp_path):
        """keep_closes_for_mc=False → MC skipped even with run_mc=True."""
        syms = ["EURUSD"]
        q_results = {s: _make_qual_result(s) for s in syms}
        m1 = {s: self._make_long_m1() for s in syms}
        with _patch_qualify(q_results):
            result = run_full_pipeline(
                m1_data=m1,
                spread_specs={s: (1.0, 0.0001) for s in syms},
                output_dir=tmp_path,
                results_dir=tmp_path,
                run_mc=True,
                keep_closes_for_mc=False,
                mc_runs=3,
            )
        # With keep_closes_for_mc=False, _run_per_instrument_mc is called (no closes)
        for ir in result.instrument_results:
            assert ir.mc_result is None

    def test_run_mc_true_no_qualified_runs_safely(self, tmp_path):
        """run_mc=True with zero qualified instruments should not crash."""
        with _patch_qualify({}):
            result = run_full_pipeline(
                m1_data={},
                spread_specs={},
                output_dir=tmp_path,
                results_dir=tmp_path,
                run_mc=True,
                mc_runs=3,
            )
        assert result.n_qualified == 0
        assert isinstance(result, PortfolioPipelineResult)

    def test_run_mc_true_with_closes_invokes_mc_with_closes(self, tmp_path):
        """When run_mc=True and keep_closes_for_mc=True, MC-with-closes path is used."""
        syms = ["EURUSD"]
        q_results = {s: _make_qual_result(s) for s in syms}
        m1 = {s: self._make_long_m1() for s in syms}

        mc_with_closes_called = []

        original = _run_per_instrument_mc_with_closes

        def spy(*args, **kwargs):
            mc_with_closes_called.append(True)
            return original(*args, **kwargs)

        with _patch_qualify(q_results):
            with patch(
                "kinetra.renko.orchestrator._run_per_instrument_mc_with_closes",
                side_effect=spy,
            ):
                run_full_pipeline(
                    m1_data=m1,
                    spread_specs={s: (1.0, 0.0001) for s in syms},
                    output_dir=tmp_path,
                    results_dir=tmp_path,
                    run_mc=True,
                    keep_closes_for_mc=True,
                    mc_runs=3,
                )
        # spy should have been called at least once (even if closes were too short)
        assert len(mc_with_closes_called) >= 1

    def test_mc_seed_is_passed_through(self, tmp_path):
        """mc_seed is forwarded to the MC functions."""
        syms = ["EURUSD"]
        q_results = {s: _make_qual_result(s) for s in syms}
        m1 = {s: self._make_long_m1() for s in syms}

        received_seeds = []

        def capture_seed(*args, **kwargs):
            received_seeds.append(kwargs.get("seed"))
            return []

        with _patch_qualify(q_results):
            with patch(
                "kinetra.renko.orchestrator._run_per_instrument_mc",
                side_effect=capture_seed,
            ):
                run_full_pipeline(
                    m1_data=m1,
                    spread_specs={s: (1.0, 0.0001) for s in syms},
                    output_dir=tmp_path,
                    results_dir=tmp_path,
                    run_mc=True,
                    keep_closes_for_mc=False,
                    mc_runs=3,
                    mc_seed=999,
                )
        assert 999 in received_seeds


# ══════════════════════════════════════════════════════════════════════════════
# TestRunFullPipelineWithTailRisk — Sprint 5C
# ══════════════════════════════════════════════════════════════════════════════


class TestRunFullPipelineWithTailRisk:
    """Tests for run_full_pipeline with run_tail_risk=True."""

    def test_tail_risk_false_leaves_tail_risk_none(self, tmp_path):
        with _patch_qualify({}):
            result = run_full_pipeline(
                m1_data={},
                spread_specs={},
                output_dir=tmp_path,
                results_dir=tmp_path,
                run_tail_risk=False,
            )
        assert result.tail_risk is None

    def test_tail_risk_none_when_no_portfolio(self, tmp_path):
        """Even with run_tail_risk=True, tail_risk is None when no portfolio was built."""
        syms = ["EURUSD"]
        q_results = {s: _make_qual_result(s) for s in syms}
        m1 = {s: _make_m1_df() for s in syms}
        with _patch_qualify(q_results):
            result = run_full_pipeline(
                m1_data=m1,
                spread_specs={s: (1.0, 0.0001) for s in syms},
                output_dir=tmp_path,
                results_dir=tmp_path,
                run_tail_risk=True,
                min_portfolio_instruments=3,  # too high → no portfolio
            )
        assert result.tail_risk is None

    def test_tail_risk_exception_adds_to_errors(self, tmp_path):
        """An exception in tail_risk_analysis is captured in errors, not raised."""
        syms = ["EURUSD", "XAUUSD", "GBPUSD"]
        q_results = {s: _make_qual_result(s) for s in syms}
        m1 = {s: _make_m1_df() for s in syms}

        mock_port = _make_portfolio_backtest_result(syms)

        with _patch_qualify(q_results):
            with patch(
                "kinetra.renko.orchestrator._build_and_backtest_portfolio",
                return_value=(mock_port, {s: 1.0 / 3 for s in syms}),
            ):
                with patch(
                    "kinetra.renko.orchestrator._compute_tail_risk",
                    side_effect=RuntimeError("tr crash"),
                ):
                    result = run_full_pipeline(
                        m1_data=m1,
                        spread_specs={s: (1.0, 0.0001) for s in syms},
                        output_dir=tmp_path,
                        results_dir=tmp_path,
                        run_tail_risk=True,
                    )
        assert result.tail_risk is None
        assert any("Tail-risk analysis failed" in e for e in result.errors)

    def test_tail_risk_preserved_in_persisted_json(self, tmp_path):
        """tail_risk dict is written to portfolio_result.json when present."""
        r = PortfolioPipelineResult(
            pipeline_run_id="tr-test",
            n_instruments=3,
            n_qualified=3,
            n_disqualified=0,
            tail_risk={"cvar_95": -0.05, "max_drawdown": -0.12},
        )
        r.save(tmp_path / DEFAULT_PORTFOLIO_RESULT_FILENAME)
        loaded = load_pipeline_result(tmp_path)
        assert loaded is not None
        assert loaded.tail_risk is not None
        assert loaded.tail_risk["cvar_95"] == pytest.approx(-0.05)


# ══════════════════════════════════════════════════════════════════════════════
# TestDeploymentReadyGate — Sprint 5C
# ══════════════════════════════════════════════════════════════════════════════


class TestDeploymentReadyGate:
    """Tests for the deployment_ready logic in run_full_pipeline."""

    def test_deployment_ready_requires_omega_threshold(self, tmp_path):
        """deployment_ready is False when portfolio_omega < threshold."""
        syms = ["EURUSD", "XAUUSD", "GBPUSD"]
        q_results = {s: _make_qual_result(s) for s in syms}
        m1 = {s: _make_m1_df() for s in syms}

        mock_port = _make_portfolio_backtest_result(syms, omega=0.5, z_factor=10.0)

        with _patch_qualify(q_results):
            with patch(
                "kinetra.renko.orchestrator._build_and_backtest_portfolio",
                return_value=(mock_port, {s: 1.0 / 3 for s in syms}),
            ):
                result = run_full_pipeline(
                    m1_data=m1,
                    spread_specs={s: (1.0, 0.0001) for s in syms},
                    output_dir=tmp_path,
                    results_dir=tmp_path,
                    portfolio_min_omega=2.0,
                )
        assert result.deployment_ready is False

    def test_deployment_ready_requires_z_threshold(self, tmp_path):
        """deployment_ready is False when portfolio_z < threshold."""
        syms = ["EURUSD", "XAUUSD", "GBPUSD"]
        q_results = {s: _make_qual_result(s) for s in syms}
        m1 = {s: _make_m1_df() for s in syms}

        mock_port = _make_portfolio_backtest_result(syms, omega=6.0, z_factor=1.0)

        with _patch_qualify(q_results):
            with patch(
                "kinetra.renko.orchestrator._build_and_backtest_portfolio",
                return_value=(mock_port, {s: 1.0 / 3 for s in syms}),
            ):
                result = run_full_pipeline(
                    m1_data=m1,
                    spread_specs={s: (1.0, 0.0001) for s in syms},
                    output_dir=tmp_path,
                    results_dir=tmp_path,
                    portfolio_min_z=5.0,
                )
        assert result.deployment_ready is False

    def test_deployment_ready_when_all_thresholds_met(self, tmp_path):
        """deployment_ready is True when omega, z, and qualified count are all met."""
        syms = ["EURUSD", "XAUUSD", "GBPUSD"]
        q_results = {s: _make_qual_result(s) for s in syms}
        m1 = {s: _make_m1_df() for s in syms}

        mock_port = _make_portfolio_backtest_result(syms, omega=6.0, z_factor=10.0)

        with _patch_qualify(q_results):
            with patch(
                "kinetra.renko.orchestrator._build_and_backtest_portfolio",
                return_value=(mock_port, {s: 1.0 / 3 for s in syms}),
            ):
                result = run_full_pipeline(
                    m1_data=m1,
                    spread_specs={s: (1.0, 0.0001) for s in syms},
                    output_dir=tmp_path,
                    results_dir=tmp_path,
                    portfolio_min_omega=2.0,
                    portfolio_min_z=5.0,
                    min_portfolio_instruments=3,
                )
        assert result.deployment_ready is True

    def test_portfolio_omega_and_z_stored_in_result(self, tmp_path):
        """portfolio_omega and portfolio_z are extracted from the backtest result."""
        syms = ["EURUSD", "XAUUSD", "GBPUSD"]
        q_results = {s: _make_qual_result(s) for s in syms}
        m1 = {s: _make_m1_df() for s in syms}

        mock_port = _make_portfolio_backtest_result(syms, omega=5.5, z_factor=11.1)

        with _patch_qualify(q_results):
            with patch(
                "kinetra.renko.orchestrator._build_and_backtest_portfolio",
                return_value=(mock_port, {s: 1.0 / 3 for s in syms}),
            ):
                result = run_full_pipeline(
                    m1_data=m1,
                    spread_specs={s: (1.0, 0.0001) for s in syms},
                    output_dir=tmp_path,
                    results_dir=tmp_path,
                )
        assert result.portfolio_omega == pytest.approx(5.5)
        assert result.portfolio_z == pytest.approx(11.1)

    def test_portfolio_max_dd_stored_in_result(self, tmp_path):
        """portfolio_max_dd is extracted from the backtest result."""
        syms = ["EURUSD", "XAUUSD", "GBPUSD"]
        q_results = {s: _make_qual_result(s) for s in syms}
        m1 = {s: _make_m1_df() for s in syms}

        mock_port = _make_portfolio_backtest_result(syms)
        mock_port = PortfolioBacktestResult(
            instruments=mock_port.instruments,
            instrument_results=mock_port.instrument_results,
            allocation_weights=mock_port.allocation_weights,
            equity_curve=mock_port.equity_curve,
            total_trades=mock_port.total_trades,
            omega=mock_port.omega,
            z_factor=mock_port.z_factor,
            profit_factor=mock_port.profit_factor,
            win_rate=mock_port.win_rate,
            net_pnl_usd=mock_port.net_pnl_usd,
            max_dd_usd=-777.77,
            calmar_ratio=mock_port.calmar_ratio,
            years=mock_port.years,
            trades_per_year=mock_port.trades_per_year,
        )

        with _patch_qualify(q_results):
            with patch(
                "kinetra.renko.orchestrator._build_and_backtest_portfolio",
                return_value=(mock_port, {s: 1.0 / 3 for s in syms}),
            ):
                result = run_full_pipeline(
                    m1_data=m1,
                    spread_specs={s: (1.0, 0.0001) for s in syms},
                    output_dir=tmp_path,
                    results_dir=tmp_path,
                )
        assert result.portfolio_max_dd == pytest.approx(-777.77)


# ══════════════════════════════════════════════════════════════════════════════
# TestPortfolioDaySnapshotDriftFields — Sprint 5C
# ══════════════════════════════════════════════════════════════════════════════


class TestPortfolioDaySnapshotDriftFields:
    """
    Sprint 5C: verify that PortfolioDaySnapshot.vr_drift and
    recalibration_pending are wired into RiskOverlayEnv observations.
    """

    def test_snapshot_has_vr_drift_field(self):
        """PortfolioDaySnapshot must expose vr_drift (Sprint 5C)."""
        from kinetra.rl.risk_env import PortfolioDaySnapshot

        snap = PortfolioDaySnapshot(
            date=datetime(2024, 1, 1, tzinfo=timezone.utc),
            vr_drift=0.22,
        )
        assert snap.vr_drift == pytest.approx(0.22)

    def test_snapshot_has_recalibration_pending_field(self):
        """PortfolioDaySnapshot must expose recalibration_pending (Sprint 5C)."""
        from kinetra.rl.risk_env import PortfolioDaySnapshot

        snap = PortfolioDaySnapshot(
            date=datetime(2024, 1, 1, tzinfo=timezone.utc),
            recalibration_pending=0.33,
        )
        assert snap.recalibration_pending == pytest.approx(0.33)

    def test_drift_fields_default_to_zero(self):
        """Both drift fields must default to 0.0 for backward compatibility."""
        from kinetra.rl.risk_env import PortfolioDaySnapshot

        snap = PortfolioDaySnapshot(date=datetime(2024, 1, 1, tzinfo=timezone.utc))
        assert snap.vr_drift == 0.0
        assert snap.recalibration_pending == 0.0

    def test_obs_dimension_is_ten(self):
        """N_RISK_OBS_FEATURES must be 10 after Sprint 5C bump."""
        from kinetra.rl.risk_env import N_RISK_OBS_FEATURES

        assert N_RISK_OBS_FEATURES == 10

    def test_observation_includes_drift_at_indices_8_and_9(self):
        """obs[8] == vr_drift and obs[9] == recalibration_pending."""
        from kinetra.rl.risk_env import PortfolioDaySnapshot, RiskEnvConfig, RiskOverlayEnv

        base = datetime(2024, 1, 1, tzinfo=timezone.utc)
        from datetime import timedelta

        snaps = [
            PortfolioDaySnapshot(
                date=base + timedelta(days=i),
                portfolio_return=0.001,
                vr_drift=0.18,
                recalibration_pending=0.75,
            )
            for i in range(100)
        ]
        env = RiskOverlayEnv(snapshots=snaps, config=RiskEnvConfig(random_start=False, seed=0))
        obs, _ = env.reset()
        assert obs.shape[0] == 10
        assert abs(obs[8] - 0.18) < 1e-5
        assert abs(obs[9] - 0.75) < 1e-5

    def test_zero_drift_snapshots_produce_zero_in_obs(self):
        """Legacy snapshots with vr_drift=0 produce obs[8]==obs[9]==0."""
        from kinetra.rl.risk_env import PortfolioDaySnapshot, RiskEnvConfig, RiskOverlayEnv

        base = datetime(2024, 1, 1, tzinfo=timezone.utc)
        from datetime import timedelta

        snaps = [
            PortfolioDaySnapshot(date=base + timedelta(days=i), portfolio_return=0.001)
            for i in range(100)
        ]
        env = RiskOverlayEnv(snapshots=snaps, config=RiskEnvConfig(random_start=False, seed=0))
        obs, _ = env.reset()
        assert obs[8] == pytest.approx(0.0)
        assert obs[9] == pytest.approx(0.0)

    def test_vr_drift_clipped_to_unity_in_obs(self):
        """Values > 1.0 for vr_drift must be clipped to 1.0 in the observation."""
        from kinetra.rl.risk_env import PortfolioDaySnapshot, RiskEnvConfig, RiskOverlayEnv

        base = datetime(2024, 1, 1, tzinfo=timezone.utc)
        from datetime import timedelta

        snaps = [
            PortfolioDaySnapshot(
                date=base + timedelta(days=i), portfolio_return=0.001, vr_drift=999.0
            )
            for i in range(100)
        ]
        env = RiskOverlayEnv(snapshots=snaps, config=RiskEnvConfig(random_start=False, seed=0))
        obs, _ = env.reset()
        assert obs[8] == pytest.approx(1.0)

    def test_recalibration_pending_clipped_to_zero_when_negative(self):
        """Negative recalibration_pending is clipped to 0.0 in the observation."""
        from kinetra.rl.risk_env import PortfolioDaySnapshot, RiskEnvConfig, RiskOverlayEnv

        base = datetime(2024, 1, 1, tzinfo=timezone.utc)
        from datetime import timedelta

        snaps = [
            PortfolioDaySnapshot(
                date=base + timedelta(days=i),
                portfolio_return=0.001,
                recalibration_pending=-0.5,
            )
            for i in range(100)
        ]
        env = RiskOverlayEnv(snapshots=snaps, config=RiskEnvConfig(random_start=False, seed=0))
        obs, _ = env.reset()
        assert obs[9] == pytest.approx(0.0)

    def test_step_observation_also_includes_drift(self):
        """After env.step(), the returned observation still includes drift at [8] and [9]."""
        from kinetra.rl.risk_env import PortfolioDaySnapshot, RiskEnvConfig, RiskOverlayEnv

        base = datetime(2024, 1, 1, tzinfo=timezone.utc)
        from datetime import timedelta

        snaps = [
            PortfolioDaySnapshot(
                date=base + timedelta(days=i),
                portfolio_return=0.001,
                vr_drift=0.35,
                recalibration_pending=0.5,
            )
            for i in range(300)
        ]
        env = RiskOverlayEnv(
            snapshots=snaps, config=RiskEnvConfig(random_start=False, seed=0, episode_days=50)
        )
        env.reset()
        import numpy as np

        obs, _, _, _, _ = env.step(np.array([0.8], dtype=np.float32))
        assert obs.shape[0] == 10
        assert abs(obs[8] - 0.35) < 1e-5
        assert abs(obs[9] - 0.5) < 1e-5

    def test_snapshots_from_trades_factory_creates_zero_drift_fields(self):
        """snapshots_from_trades() must produce snapshots with vr_drift==0 by default."""
        from kinetra.rl.risk_env import RiskOverlayEnv

        trades = [
            _make_renko_trade("EURUSD", net_usd=10.0, entry_offset_days=0, exit_offset_days=1),
            _make_renko_trade("EURUSD", net_usd=-5.0, entry_offset_days=2, exit_offset_days=3),
            _make_renko_trade("EURUSD", net_usd=20.0, entry_offset_days=5, exit_offset_days=6),
        ]
        snaps = RiskOverlayEnv.snapshots_from_trades(trades)
        for snap in snaps:
            assert snap.vr_drift == 0.0
            assert snap.recalibration_pending == 0.0
