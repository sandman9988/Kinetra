from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from kinetra.renko.live_trader import OrderDispatcher, OrderResult, TradeDirection
from kinetra.renko.trading_engine import EngineConfig, RenkoEngine


@dataclass
class _StubDispatcher(OrderDispatcher):
    open_success: bool = True
    close_success: bool = True
    position_open: bool | None = None
    open_calls: int = 0
    close_calls: int = 0

    def open_position(
        self,
        symbol: str,
        direction: TradeDirection,
        lots: float,
        price: float,
        stop_price: float,
        comment: str = "",
    ) -> OrderResult:
        self.open_calls += 1
        if self.open_success:
            return OrderResult(success=True, order_id="OID-1", filled_price=price, filled_lots=lots)
        return OrderResult(success=False, error="simulated_open_failure")

    def close_position(
        self,
        symbol: str,
        order_id: str,
        price: float,
        lots: float,
        comment: str = "",
    ) -> OrderResult:
        self.close_calls += 1
        if not self.close_success:
            return OrderResult(success=False, order_id=order_id, error="simulated_close_failure")
        return OrderResult(success=True, order_id=order_id, filled_price=price, filled_lots=lots)

    def get_spread_pts(self, symbol: str) -> float:
        return 0.0

    def get_equity(self) -> float | None:
        return None

    def is_position_open(self, order_id: str) -> bool | None:
        return self.position_open


def _new_engine() -> RenkoEngine:
    cfg = EngineConfig(
        symbol="XAUUSD",
        brick_size=10.0,
        stop_bricks=0.5,
        startup_skip_flips=0,
        min_warmup_bricks=1,
        fliprate_threshold=0.35,
        markov_threshold=0.55,
    )
    return RenkoEngine(cfg, quiet_mode=True)


def test_waiting_for_fill_blocks_colour_change_exit() -> None:
    engine = _new_engine()
    t0 = pd.Timestamp("2026-03-03T10:00:00Z")

    engine._in_pos = True
    engine._waiting_for_fill = True
    engine._pos_dir = 1
    engine._entry_price = 100.0
    engine._entry_time = t0
    engine._entry_lots = 0.1

    # Colour flip to short without stop hit should NOT exit while waiting for fill.
    engine._process_brick(104.0, -1, t0, 0.10, 0.90, 0.90, dispatcher=None)

    assert engine._in_pos is True
    assert len(engine._completed) == 0
    assert engine._waiting_for_fill is True


def test_stop_exit_still_works_while_waiting_for_fill() -> None:
    engine = _new_engine()
    t0 = pd.Timestamp("2026-03-03T10:00:00Z")
    t1 = pd.Timestamp("2026-03-03T10:01:00Z")

    engine._in_pos = True
    engine._waiting_for_fill = True
    engine._pos_dir = 1
    engine._entry_price = 100.0
    engine._entry_time = t0
    engine._entry_lots = 0.1

    # stop_dist = 0.5 * 10 = 5; price=94 hits stop.
    engine._process_brick(94.0, -1, t1, 0.10, 0.90, 0.90, dispatcher=None)

    assert engine._in_pos is False
    assert engine._waiting_for_fill is False
    assert len(engine._completed) == 1
    assert engine._completed[0].exit_reason == "stop"


def test_fill_confirmation_clears_waiting_flag() -> None:
    engine = _new_engine()
    dispatcher = _StubDispatcher(open_success=True)
    t0 = pd.Timestamp("2026-03-03T10:00:00Z")

    engine._prev_dir = -1
    engine._brick_count = 1  # with min_warmup_bricks=1, warmup is ready after increment

    engine._process_brick(100.0, 1, t0, 0.10, 0.90, 0.90, dispatcher=dispatcher)

    assert dispatcher.open_calls == 1
    assert engine._in_pos is True
    assert engine._waiting_for_fill is False
    assert engine._open_order_id == "OID-1"


def test_open_failure_rolls_back_pending_position_state() -> None:
    engine = _new_engine()
    dispatcher = _StubDispatcher(open_success=False)
    t0 = pd.Timestamp("2026-03-03T10:00:00Z")

    engine._prev_dir = -1
    engine._brick_count = 1  # with min_warmup_bricks=1, warmup is ready after increment

    engine._process_brick(100.0, 1, t0, 0.10, 0.90, 0.90, dispatcher=dispatcher)

    assert dispatcher.open_calls == 1
    assert engine._in_pos is False
    assert engine._waiting_for_fill is False
    assert engine._open_order_id is None
    assert engine._entry_lots == 0.0
    assert engine._last_eval["reason"] == "order_submit_failed"


def test_backtest_mode_never_leaves_waiting_for_fill_and_marks_simulated_orders() -> None:
    cfg = EngineConfig(
        symbol="XAUUSD",
        brick_size=10.0,
        stop_bricks=0.5,
        startup_skip_flips=0,
        min_warmup_bricks=1,
        fliprate_window=2,
        markov_window=2,
        fliprate_threshold=1.0,
        markov_threshold=0.0,
    )
    engine = RenkoEngine(cfg, quiet_mode=True)

    idx = pd.date_range("2026-03-03T10:00:00Z", periods=16, freq="min")
    closes = pd.Series(
        [
            100.0,
            110.0,
            100.0,
            110.0,
            100.0,
            110.0,
            100.0,
            110.0,
            100.0,
            110.0,
            100.0,
            110.0,
            100.0,
            110.0,
            100.0,
            110.0,
        ],
        index=idx,
    )
    result = engine.backtest(closes)

    assert "error" not in result
    assert engine._waiting_for_fill is False

    simulated_entries = [
        s
        for s in result["signals"]
        if s.get("entry_ok") and float(s.get("requested_lots", 0.0)) > 0.0
    ]
    assert simulated_entries, "Expected at least one simulated backtest entry."
    assert all(s.get("order_success") for s in simulated_entries)
    assert all(s.get("order_id") == "BACKTEST" for s in simulated_entries)


def test_close_failure_keeps_position_open_and_no_local_trade_booked() -> None:
    engine = _new_engine()
    dispatcher = _StubDispatcher(open_success=True, close_success=False)
    t0 = pd.Timestamp("2026-03-03T10:00:00Z")
    t1 = pd.Timestamp("2026-03-03T10:01:00Z")

    engine._in_pos = True
    engine._waiting_for_fill = False
    engine._pos_dir = 1
    engine._entry_price = 100.0
    engine._entry_time = t0
    engine._entry_lots = 0.1
    engine._open_order_id = "201974987"
    engine._open_signal_id = "S-1"

    # stop_dist = 5.0, so 94.0 triggers stop close attempt
    engine._process_brick(94.0, -1, t1, 0.10, 0.90, 0.90, dispatcher=dispatcher)

    assert dispatcher.close_calls == 1
    assert engine._in_pos is True
    assert engine._open_order_id == "201974987"
    assert len(engine._completed) == 0
    assert engine._last_eval["reason"] == "close_dispatch_failed"


def test_colour_change_exit_tags_trend_run_capture() -> None:
    engine = _new_engine()
    t0 = pd.Timestamp("2026-03-03T10:00:00Z")
    t1 = pd.Timestamp("2026-03-03T10:01:00Z")

    engine._in_pos = True
    engine._waiting_for_fill = False
    engine._pos_dir = 1
    engine._entry_price = 100.0
    engine._entry_time = t0
    engine._entry_lots = 0.1
    engine._entry_brick_index = 1
    engine._brick_count = 10
    engine._current_run_dir = 1
    engine._current_run_len = 5

    # Flip to down without stop hit -> colour_change exit.
    engine._process_brick(101.0, -1, t1, 0.10, 0.90, 0.90, dispatcher=None)

    assert len(engine._completed) == 1
    trade = engine._completed[0]
    assert trade.exit_reason == "colour_change"
    assert getattr(trade, "trend_run_bricks", 0) == 5
    assert 0.0 <= float(getattr(trade, "run_capture", 0.0)) <= 1.0


def test_broker_flat_sync_closes_local_position_without_dispatch_close() -> None:
    engine = _new_engine()
    dispatcher = _StubDispatcher(position_open=False)
    t0 = pd.Timestamp("2026-03-03T10:00:00Z")
    t1 = pd.Timestamp("2026-03-03T10:01:00Z")

    engine._in_pos = True
    engine._waiting_for_fill = False
    engine._pos_dir = 1
    engine._entry_price = 100.0
    engine._entry_time = t0
    engine._entry_lots = 0.1
    engine._open_order_id = "202016826"
    engine._open_signal_id = "S-00000037-000001"
    engine._entry_brick_index = 1

    engine._process_brick(99.0, -1, t1, 0.10, 0.90, 0.90, dispatcher=dispatcher)

    assert dispatcher.close_calls == 0
    assert engine._in_pos is False
    assert engine._open_order_id is None
    assert len(engine._completed) == 1
    assert engine._completed[0].exit_reason == "broker_flat_sync"
    assert engine._last_eval["reason"] == "broker_flat_sync"
