from __future__ import annotations

from types import SimpleNamespace

from kinetra.renko.trade_analytics import analyze_trades


def test_run_capture_aggregates_when_available() -> None:
    t1 = SimpleNamespace(
        net_usd=10.0,
        spread_usd=0.0,
        commission_usd=0.0,
        swap_usd=0.0,
        n_bricks_held=5,
        trend_run_bricks=10,
    )
    t2 = SimpleNamespace(
        net_usd=-4.0,
        spread_usd=0.0,
        commission_usd=0.0,
        swap_usd=0.0,
        n_bricks_held=3,
        trend_run_bricks=6,
    )
    analytics = analyze_trades([t1, t2], initial_equity=1000.0)

    assert analytics.run_capture_samples == 2
    assert analytics.avg_run_capture == 0.5
    assert analytics.median_run_capture == 0.5


def test_run_capture_zero_when_not_available() -> None:
    t1 = SimpleNamespace(net_usd=1.0, spread_usd=0.0, commission_usd=0.0, swap_usd=0.0)
    analytics = analyze_trades([t1], initial_equity=1000.0)

    assert analytics.run_capture_samples == 0
    assert analytics.avg_run_capture == 0.0
    assert analytics.median_run_capture == 0.0
