from types import SimpleNamespace

import pytest

from scripts.renko_engine import DriftAdaptationController


class _FakeEngine:
    def __init__(
        self,
        *,
        target_risk_usd: float = 100.0,
        stop_bricks: float = 1.0,
        fliprate_threshold: float = 0.35,
        markov_threshold: float = 0.55,
    ) -> None:
        self.cfg = SimpleNamespace(
            target_risk_usd=target_risk_usd,
            stop_bricks=stop_bricks,
            fliprate_threshold=fliprate_threshold,
            markov_threshold=markov_threshold,
        )
        self._summary = {"n_trades": 0, "omega": 0.0, "win_rate": 0.0}

    def set_summary(self, *, n_trades: int, omega: float, win_rate: float) -> None:
        self._summary = {
            "n_trades": n_trades,
            "omega": omega,
            "win_rate": win_rate,
        }

    def _make_results(self):
        return {"summary": dict(self._summary)}


def _mk_controller(engine: _FakeEngine, **opts) -> DriftAdaptationController:
    base = {
        "enabled": True,
        "every_bars": 100,
        "min_trades": 20,
        "low_omega": 1.0,
        "high_omega": 1.8,
        "low_wr": 0.45,
        "high_wr": 0.60,
        "step_risk": 0.10,
        "step_thr": 0.01,
        "step_stop": 0.05,
        "risk_min": 25.0,
        "risk_max": 200.0,
        "stop_min": 0.30,
        "stop_max": 1.50,
        "flip_min": 0.25,
        "flip_max": 0.45,
        "markov_min": 0.50,
        "markov_max": 0.70,
        "rollback_omega": 0.80,
    }
    base.update(opts)
    return DriftAdaptationController(engine, base)


def test_defensive_adaptation_reduces_risk_and_tightens_filters():
    engine = _FakeEngine()
    engine.set_summary(n_trades=30, omega=0.9, win_rate=0.40)
    ctrl = _mk_controller(engine)

    ctrl.maybe_adapt(100)

    assert engine.cfg.target_risk_usd == pytest.approx(90.0)
    assert engine.cfg.stop_bricks == pytest.approx(1.05)
    assert engine.cfg.fliprate_threshold == pytest.approx(0.36)
    assert engine.cfg.markov_threshold == pytest.approx(0.56)
    assert ctrl.adapt_count > 0


def test_offensive_adaptation_increases_risk_and_relaxes_filters():
    engine = _FakeEngine()
    engine.set_summary(n_trades=30, omega=2.1, win_rate=0.70)
    ctrl = _mk_controller(engine)

    ctrl.maybe_adapt(100)

    assert engine.cfg.target_risk_usd == pytest.approx(110.0)
    assert engine.cfg.stop_bricks == pytest.approx(0.95)
    assert engine.cfg.fliprate_threshold == pytest.approx(0.34)
    assert engine.cfg.markov_threshold == pytest.approx(0.54)
    assert ctrl.adapt_count > 0


def test_adaptation_respects_bounds():
    engine = _FakeEngine(
        target_risk_usd=26.0,
        stop_bricks=1.49,
        fliprate_threshold=0.449,
        markov_threshold=0.699,
    )
    engine.set_summary(n_trades=30, omega=0.7, win_rate=0.30)
    ctrl = _mk_controller(engine)

    ctrl.maybe_adapt(100)

    assert engine.cfg.target_risk_usd >= 25.0
    assert engine.cfg.stop_bricks <= 1.50
    assert engine.cfg.fliprate_threshold <= 0.45
    assert engine.cfg.markov_threshold <= 0.70


def test_rollback_restores_baseline_after_degradation():
    engine = _FakeEngine()
    ctrl = _mk_controller(engine)

    engine.set_summary(n_trades=30, omega=2.0, win_rate=0.70)
    ctrl.maybe_adapt(100)
    assert ctrl.adapt_count > 0
    assert engine.cfg.target_risk_usd != 100.0

    engine.set_summary(n_trades=30, omega=0.5, win_rate=0.30)
    ctrl.maybe_adapt(200)

    assert engine.cfg.target_risk_usd == pytest.approx(100.0)
    assert engine.cfg.stop_bricks == pytest.approx(1.0)
    assert engine.cfg.fliprate_threshold == pytest.approx(0.35)
    assert engine.cfg.markov_threshold == pytest.approx(0.55)
