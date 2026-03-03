from datetime import datetime, timedelta, timezone

from kinetra.renko.trading_engine import EngineConfig, RenkoEngine


def _feed_prices(engine: RenkoEngine, prices: list[float]) -> None:
    t0 = datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc)
    for i, p in enumerate(prices):
        engine.process_bar(float(p), t0 + timedelta(minutes=i))


def test_brick_counter_does_not_use_rolling_window_length() -> None:
    cfg = EngineConfig(
        symbol="XAUUSD",
        brick_size=1.0,
        fliprate_window=2,
        markov_window=2,
        fliprate_threshold=1.0,
        markov_threshold=0.0,
    )
    engine = RenkoEngine(cfg)

    # 11 prices from 100..110 should yield 10 consecutive up-bricks.
    _feed_prices(engine, [float(x) for x in range(100, 111)])

    assert engine.bricks_processed == 10
    assert len(engine._dir_deque) < engine.bricks_processed


def test_no_brick_formed_keeps_brick_counter_zero() -> None:
    cfg = EngineConfig(symbol="XAUUSD", brick_size=1.0, fliprate_window=2, markov_window=2)
    engine = RenkoEngine(cfg)

    # No move reaches one full brick.
    _feed_prices(engine, [100.0, 100.2, 100.6, 100.4, 100.8])

    assert engine.bricks_processed == 0
    assert engine._last_eval["reason"] == "awaiting_bricks"


def test_flip_after_warmup_triggers_entry_evaluation() -> None:
    cfg = EngineConfig(
        symbol="XAUUSD",
        brick_size=1.0,
        fliprate_window=2,
        markov_window=2,
        fliprate_threshold=1.0,  # disable FR gate
        markov_threshold=0.0,  # disable Markov gate
    )
    engine = RenkoEngine(cfg)

    # Bricks: up, up, down (flip after warmup ready).
    _feed_prices(engine, [100.0, 101.0, 102.0, 101.0])

    assert engine.bricks_processed == 3
    assert engine._last_eval["is_flip"] is True
    assert engine._last_eval["warmup_ready"] is True
    # First post-warmup flip must be evaluated as a flip path (not no_flip/warmup).
    assert engine._last_eval["reason"] in {
        "filters_not_ready",
        "gate_reject",
        "entry_pass",
        "entered",
    }
