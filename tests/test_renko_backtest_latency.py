from __future__ import annotations

import numpy as np
import pandas as pd

from kinetra.renko.backtest import (
    FilterParams,
    LatencyParams,
    StopParams,
    backtest_instrument,
)


def _sample_closes() -> pd.Series:
    rng = np.random.default_rng(0)
    steps = rng.choice([-1, 1], size=500, p=[0.45, 0.55])
    vals = 100.0 + np.cumsum(steps.astype(np.float64))
    idx = pd.date_range("2026-01-01", periods=len(vals), freq="min", tz="UTC")
    return pd.Series(vals, index=idx)


def _filter_params() -> FilterParams:
    return FilterParams(
        fliprate_window=10,
        fliprate_threshold=0.99,
        markov_window=10,
        markov_threshold=0.01,
    )


def _stop_params() -> StopParams:
    return StopParams(stop_bricks=10.0, exit_on_colour_change=True, allow_short=True)


def test_zero_latency_matches_default_behavior() -> None:
    closes = _sample_closes()
    base = backtest_instrument(
        "XAUUSD",
        closes,
        1.0,
        filter_params=_filter_params(),
        stop_params=_stop_params(),
        warmup_override=0,
    )
    zero = backtest_instrument(
        "XAUUSD",
        closes,
        1.0,
        filter_params=_filter_params(),
        stop_params=_stop_params(),
        warmup_override=0,
        latency_params=LatencyParams(),
    )

    assert len(base.trades) == len(zero.trades)
    assert [t.to_dict() for t in base.trades] == [t.to_dict() for t in zero.trades]


def test_latency_delays_fills() -> None:
    closes = _sample_closes()
    base = backtest_instrument(
        "XAUUSD",
        closes,
        1.0,
        filter_params=_filter_params(),
        stop_params=_stop_params(),
        warmup_override=0,
        latency_params=LatencyParams(),
    )
    delayed = backtest_instrument(
        "XAUUSD",
        closes,
        1.0,
        filter_params=_filter_params(),
        stop_params=_stop_params(),
        warmup_override=0,
        latency_params=LatencyParams(entry_latency_ms=61_000, exit_latency_ms=61_000),
    )

    assert base.trades
    assert delayed.trades
    b0 = base.trades[0]
    d0 = delayed.trades[0]
    assert d0.entry_time >= b0.entry_time
    assert d0.exit_time >= b0.exit_time
    assert (
        d0.entry_time > b0.entry_time
        or d0.exit_time > b0.exit_time
        or d0.entry_price != b0.entry_price
        or d0.exit_price != b0.exit_price
    )
