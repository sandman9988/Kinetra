from __future__ import annotations

import importlib
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from kinetra.renko.brick_engine import build_renko
from kinetra.renko.filters import flip_rate, markov_stickiness
from kinetra.renko.trade_analytics import analyze_trades
from kinetra.renko.vol_sizer import quantize_and_clamp_lot, quantize_and_clamp_lots_array

RENKO_MODULES = [
    "kinetra.renko",
    "kinetra.renko.backtest",
    "kinetra.renko.brick_engine",
    "kinetra.renko.ctrader_dispatcher",
    "kinetra.renko.drift",
    "kinetra.renko.dsp",
    "kinetra.renko.filters",
    "kinetra.renko.instrument_pool",
    "kinetra.renko.live_trader",
    "kinetra.renko.orchestrator",
    "kinetra.renko.policy",
    "kinetra.renko.portfolio",
    "kinetra.renko.qualify",
    "kinetra.renko.session",
    "kinetra.renko.spread_gated_backtest",
    "kinetra.renko.trade_analytics",
    "kinetra.renko.trading_engine",
    "kinetra.renko.vol_sizer",
    "kinetra.renko.vpin",
    "kinetra.renko.pipeline",
    "kinetra.renko.pipeline.discovery",
    "kinetra.renko.pipeline.drift",
    "kinetra.renko.pipeline.engines",
    "kinetra.renko.pipeline.identity",
    "kinetra.renko.pipeline.pool",
    "kinetra.renko.pipeline.qualify",
    "kinetra.renko.pipeline.recalibrate",
    "kinetra.renko.pipeline.registry",
    "kinetra.renko.pipeline.specs",
]


@pytest.mark.parametrize("module_name", RENKO_MODULES)
def test_renko_modules_import(module_name: str) -> None:
    mod = importlib.import_module(module_name)
    assert mod is not None


def test_build_renko_handles_session_break_without_burst() -> None:
    idx = pd.to_datetime(
        [
            "2026-01-01 00:00:00+00:00",
            "2026-01-01 00:01:00+00:00",
            "2026-01-01 02:00:00+00:00",
            "2026-01-01 02:01:00+00:00",
        ],
        utc=True,
    )
    closes = pd.Series([100.0, 101.0, 120.0, 121.0], index=idx)

    bricks = build_renko(closes, brick_size=1.0, session_break_minutes=30.0)
    assert not bricks.empty
    # Gap reset suppresses the post-gap burst from 101 -> 120.
    assert len(bricks) == 2
    assert bool(bricks["session_break"].any()) is False
    assert bricks["time"].dt.tz is not None


def test_flip_rate_and_markov_shapes_and_ranges() -> None:
    directions = np.array([1, 1, -1, -1, 1, 1, 1, -1], dtype=np.int8)
    fr = flip_rate(directions, window=3, min_periods=2)
    puu, pdd = markov_stickiness(directions, window=4, min_periods=2)

    assert len(fr) == len(directions)
    assert len(puu) == len(directions)
    assert len(pdd) == len(directions)
    assert np.nanmax(fr) <= 1.0
    assert np.nanmin(fr) >= 0.0
    assert np.nanmax(puu) <= 1.0
    assert np.nanmin(puu) >= 0.0
    assert np.nanmax(pdd) <= 1.0
    assert np.nanmin(pdd) >= 0.0


@dataclass
class _Trade:
    net_usd: float
    entry_time: datetime
    exit_time: datetime
    spread_usd: float = 0.2
    commission_usd: float = 0.1
    swap_usd: float = 0.0
    max_adverse_excursion: float = -0.5
    max_favorable_excursion: float = 1.0


def test_analyze_trades_basic_metrics_are_consistent() -> None:
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    trades = [
        _Trade(10.0, base, base + timedelta(hours=2)),
        _Trade(-4.0, base + timedelta(hours=3), base + timedelta(hours=4)),
        _Trade(6.0, base + timedelta(hours=5), base + timedelta(hours=8)),
    ]
    a = analyze_trades(trades, initial_equity=1000.0)

    assert a.n_trades == 3
    assert a.n_winners == 2
    assert a.n_losers == 1
    assert a.net_pnl == pytest.approx(12.0)
    assert a.final_equity == pytest.approx(1012.0)
    assert a.max_drawdown_usd >= 0.0
    assert a.total_friction_usd > 0.0


def test_quantize_helpers_match_scalar_and_vector_paths() -> None:
    raw = np.array([0.004, 0.011, 0.097, 0.505], dtype=np.float64)
    vec = quantize_and_clamp_lots_array(
        raw,
        lot_step=0.01,
        min_lots=0.01,
        ceilings=(0.5, 2.0),
    )
    scalar = np.array(
        [quantize_and_clamp_lot(v, lot_step=0.01, min_lots=0.01, ceilings=(0.5, 2.0)) for v in raw],
        dtype=np.float64,
    )
    assert np.allclose(vec, scalar)
