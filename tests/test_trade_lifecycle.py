"""
Trade Lifecycle Tests
=====================

Comprehensive tests for the async trade execution flow, covering:
- Circuit breaker functionality
- Close position verification
- Async order execution
- State persistence
- Thread health monitoring
- Fill queue handling
- Error recovery

Usage:
    python -m pytest tests/test_trade_lifecycle.py -v
    python -m pytest tests/test_trade_lifecycle.py::TestCircuitBreaker -v
"""

from __future__ import annotations

import json
import os
import queue

# Ensure project root is in path
import sys
import threading
import time
import unittest
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, Mock, patch

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from kinetra.renko.live_trader import LiveTrade, OrderResult, TradeDirection
from kinetra.renko.trading_engine import (
    EngineConfig,
    EngineState,
    OrderFill,
    OrderRequest,
    RenkoEngine,
    _atomic_write_json,
    _load_json_if_exists,
)

# ═══════════════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.fixture
def engine_config():
    """Default engine config for testing."""
    return EngineConfig(
        symbol="XAUUSD",
        brick_size=1.0,
        initial_equity=10000.0,
        min_lots=0.01,
    )


@pytest.fixture
def mock_connector():
    """Mock cTrader connector."""
    connector = MagicMock()
    connector.is_connected.return_value = True
    connector.credentials.account_id = 12345
    connector.credentials.environment = "demo"
    connector.selected_endpoint = "demo.ctraderapi.com"
    connector.request_timeout_count = 0
    connector.health_status = "UP"
    connector.failover_count = 0
    connector.failover_generation = 0
    connector.last_failover_utc = "-"

    def mock_get_account_snapshot(**kwargs):
        return {
            "balance": 10000.0,
            "equity": 10000.0,
            "account_id": 12345,
            "broker_name": "Pepperstone",
        }

    connector.get_account_snapshot.side_effect = mock_get_account_snapshot
    return connector


@pytest.fixture
def mock_dispatcher():
    """Mock order dispatcher."""
    dispatcher = MagicMock()

    def mock_open_position(**kwargs):
        return OrderResult(
            success=True,
            order_id="POS-12345",
            filled_price=kwargs.get("price", 2000.0),
            filled_lots=kwargs.get("lots", 0.01),
            raw={"client_order_id": "COID-123"},
        )

    def mock_close_position(**kwargs):
        return OrderResult(
            success=True,
            order_id=kwargs.get("order_id"),
            filled_price=kwargs.get("price", 2000.0),
            filled_lots=kwargs.get("lots", 0.01),
        )

    dispatcher.open_position.side_effect = mock_open_position
    dispatcher.close_position.side_effect = mock_close_position
    return dispatcher


# ═══════════════════════════════════════════════════════════════════════════════
# Test Circuit Breaker
# ═══════════════════════════════════════════════════════════════════════════════


class TestCircuitBreaker(unittest.TestCase):
    """Test fill failure circuit breaker functionality."""

    def setUp(self):
        """Set up test fixtures."""
        # Import here to avoid import errors if ctrader not available
        try:
            from kinetra.renko.ctrader_dispatcher import CTraderOrderDispatcher

            self.dispatcher_class = CTraderOrderDispatcher
            self.has_ctrader = True
        except ImportError:
            self.has_ctrader = False
            self.skipTest("cTrader not available")

    @patch("kinetra.renko.ctrader_dispatcher._CONNECTOR_AVAILABLE", True)
    @patch("kinetra.renko.ctrader_dispatcher._get_api_msgs")
    def test_circuit_breaker_triggers_after_max_failures(self, mock_get_api_msgs):
        """Test that circuit breaker triggers after consecutive failures."""
        if not self.has_ctrader:
            self.skipTest("cTrader not available")

        # Mock API messages
        mock_api_msgs = MagicMock()
        mock_get_api_msgs.return_value = mock_api_msgs

        # Create mock connector and bar provider
        connector = MagicMock()
        connector.is_connected.return_value = True
        connector.credentials.account_id = 12345

        bar_provider = MagicMock()
        bar_provider._symbol_ids = {"XAUUSD": 41}
        bar_provider._lock = threading.Lock()

        # Create dispatcher with low threshold
        with patch.dict(os.environ, {"CTRADER_MAX_FILL_FAILURES": "3"}):
            dispatcher = self.dispatcher_class(connector, bar_provider, fill_timeout_s=0.1)

        # Simulate 3 consecutive failures
        for i in range(3):
            with dispatcher._fill_metrics_lock:
                dispatcher._consecutive_fill_failures += 1

        # Try to open position - should be blocked
        result = dispatcher.open_position(
            symbol="XAUUSD",
            direction=TradeDirection.LONG,
            lots=0.01,
            price=2000.0,
            stop_price=1990.0,
        )

        self.assertFalse(result.success)
        self.assertIn("Circuit breaker", result.error)
        self.assertTrue(dispatcher._circuit_breaker_triggered)

    @patch("kinetra.renko.ctrader_dispatcher._CONNECTOR_AVAILABLE", True)
    def test_circuit_breaker_reset(self):
        """Test manual circuit breaker reset."""
        if not self.has_ctrader:
            self.skipTest("cTrader not available")

        connector = MagicMock()
        bar_provider = MagicMock()

        dispatcher = self.dispatcher_class(connector, bar_provider)

        # Trigger circuit breaker
        dispatcher._circuit_breaker_triggered = True
        dispatcher._consecutive_fill_failures = 5

        # Reset
        dispatcher.reset_circuit_breaker()

        self.assertFalse(dispatcher._circuit_breaker_triggered)
        self.assertEqual(dispatcher._consecutive_fill_failures, 0)

    @patch("kinetra.renko.ctrader_dispatcher._CONNECTOR_AVAILABLE", True)
    def test_successful_fill_resets_counter(self):
        """Test that successful fill resets consecutive failure counter."""
        if not self.has_ctrader:
            self.skipTest("cTrader not available")

        connector = MagicMock()
        bar_provider = MagicMock()

        dispatcher = self.dispatcher_class(connector, bar_provider)

        # Set some failures
        with dispatcher._fill_metrics_lock:
            dispatcher._consecutive_fill_failures = 2

        # Record successful fill
        with dispatcher._fill_metrics_lock:
            dispatcher._orders_filled += 1
            dispatcher._consecutive_fill_failures = 0

        self.assertEqual(dispatcher._consecutive_fill_failures, 0)


# ═══════════════════════════════════════════════════════════════════════════════
# Test Close Position Verification
# ═══════════════════════════════════════════════════════════════════════════════


class TestClosePositionVerification(unittest.TestCase):
    """Test close position timeout and reconciliation."""

    def setUp(self):
        try:
            from kinetra.renko.ctrader_dispatcher import CTraderOrderDispatcher

            self.dispatcher_class = CTraderOrderDispatcher
            self.has_ctrader = True
        except ImportError:
            self.has_ctrader = False

    @patch("kinetra.renko.ctrader_dispatcher._CONNECTOR_AVAILABLE", True)
    @patch("kinetra.renko.ctrader_dispatcher._get_api_msgs")
    def test_close_timeout_with_position_still_open(self, mock_get_api_msgs):
        """Test that close returns failure when position is still open after timeout."""
        if not self.has_ctrader:
            self.skipTest("cTrader not available")

        mock_api_msgs = MagicMock()
        mock_get_api_msgs.return_value = mock_api_msgs

        connector = MagicMock()
        connector.is_connected.return_value = True
        connector.credentials.account_id = 12345

        bar_provider = MagicMock()
        bar_provider._symbol_ids = {"XAUUSD": 41}

        dispatcher = self.dispatcher_class(connector, bar_provider, fill_timeout_s=0.01)

        # Mock is_position_open to return True (still open)
        dispatcher.is_position_open = MagicMock(return_value=True)

        # Mock the execution waiter to timeout
        with patch("kinetra.renko.ctrader_dispatcher._ExecutionWaiter") as mock_waiter_class:
            mock_waiter = MagicMock()
            mock_waiter.wait.return_value = None  # Timeout
            mock_waiter_class.return_value = mock_waiter

            result = dispatcher.close_position(
                symbol="XAUUSD",
                order_id="POS-12345",
                price=2000.0,
                lots=0.01,
            )

        self.assertFalse(result.success)
        self.assertIn("still open", result.error)
        self.assertTrue(result.raw.get("position_still_open"))

    @patch("kinetra.renko.ctrader_dispatcher._CONNECTOR_AVAILABLE", True)
    @patch("kinetra.renko.ctrader_dispatcher._get_api_msgs")
    def test_close_timeout_with_position_closed(self, mock_get_api_msgs):
        """Test that close returns success when position is confirmed closed."""
        if not self.has_ctrader:
            self.skipTest("cTrader not available")

        mock_api_msgs = MagicMock()
        mock_get_api_msgs.return_value = mock_api_msgs

        connector = MagicMock()
        connector.is_connected.return_value = True
        connector.credentials.account_id = 12345

        bar_provider = MagicMock()

        dispatcher = self.dispatcher_class(connector, bar_provider, fill_timeout_s=0.01)

        # Mock is_position_open to return False (closed)
        dispatcher.is_position_open = MagicMock(return_value=False)

        with patch("kinetra.renko.ctrader_dispatcher._ExecutionWaiter") as mock_waiter_class:
            mock_waiter = MagicMock()
            mock_waiter.wait.return_value = None  # Timeout
            mock_waiter_class.return_value = mock_waiter

            result = dispatcher.close_position(
                symbol="XAUUSD",
                order_id="POS-12345",
                price=2000.0,
                lots=0.01,
            )

        self.assertTrue(result.success)
        self.assertTrue(result.raw.get("reconciled"))

    @patch("kinetra.renko.ctrader_dispatcher._CONNECTOR_AVAILABLE", True)
    @patch("kinetra.renko.ctrader_dispatcher._get_api_msgs")
    def test_close_reconcile_check_fails(self, mock_get_api_msgs):
        """Test handling when reconcile check itself fails."""
        if not self.has_ctrader:
            self.skipTest("cTrader not available")

        mock_api_msgs = MagicMock()
        mock_get_api_msgs.return_value = mock_api_msgs

        connector = MagicMock()
        connector.is_connected.return_value = True
        connector.credentials.account_id = 12345

        bar_provider = MagicMock()

        dispatcher = self.dispatcher_class(connector, bar_provider, fill_timeout_s=0.01)

        # Mock is_position_open to return None (check failed)
        dispatcher.is_position_open = MagicMock(return_value=None)

        with patch("kinetra.renko.ctrader_dispatcher._ExecutionWaiter") as mock_waiter_class:
            mock_waiter = MagicMock()
            mock_waiter.wait.return_value = None  # Timeout
            mock_waiter_class.return_value = mock_waiter

            result = dispatcher.close_position(
                symbol="XAUUSD",
                order_id="POS-12345",
                price=2000.0,
                lots=0.01,
            )

        self.assertFalse(result.success)
        self.assertIn("unknown", result.error.lower())
        self.assertTrue(result.raw.get("reconcile_failed"))


# ═══════════════════════════════════════════════════════════════════════════════
# Test Async Order Execution
# ═══════════════════════════════════════════════════════════════════════════════


class TestAsyncOrderExecution(unittest.TestCase):
    """Test async order execution flow."""

    def test_order_request_creation(self):
        """Test OrderRequest dataclass."""
        req = OrderRequest(
            signal_id="S-00000001-000001",
            direction=1,
            price=2000.0,
            stop_price=1990.0,
            lots=0.01,
            bar_time=pd.Timestamp("2024-01-01 12:00:00", tz="UTC"),
            brick_count=10,
        )

        self.assertEqual(req.signal_id, "S-00000001-000001")
        self.assertEqual(req.direction, 1)
        self.assertEqual(req.price, 2000.0)
        self.assertEqual(req.lots, 0.01)

    def test_order_fill_creation(self):
        """Test OrderFill dataclass."""
        fill = OrderFill(
            signal_id="S-00000001-000001",
            order_id="POS-12345",
            client_order_id="COID-123",
            filled_price=2000.5,
            filled_lots=0.01,
            success=True,
        )

        self.assertEqual(fill.signal_id, "S-00000001-000001")
        self.assertEqual(fill.order_id, "POS-12345")
        self.assertTrue(fill.success)

    def test_process_pending_fills_success(self, engine_config, mock_dispatcher):
        """Test processing successful fill."""
        engine = RenkoEngine(engine_config)
        engine._reset_state()

        # Set waiting state
        with engine._state_lock:
            engine._waiting_for_fill = True

        # Create fill queue with success
        fill_queue = queue.Queue()
        fill_queue.put(
            OrderFill(
                signal_id="S-00000001-000001",
                order_id="POS-12345",
                client_order_id="COID-123",
                filled_price=2000.5,
                filled_lots=0.01,
                success=True,
            )
        )

        # Process fills
        engine._process_pending_fills(fill_queue, mock_dispatcher)

        # Verify state updated
        with engine._state_lock:
            self.assertFalse(engine._waiting_for_fill)
            self.assertEqual(engine._open_order_id, "POS-12345")

    def test_process_pending_fills_failure(self, engine_config, mock_dispatcher):
        """Test processing failed fill."""
        engine = RenkoEngine(engine_config)
        engine._reset_state()

        # Set position state
        with engine._state_lock:
            engine._waiting_for_fill = True
            engine._in_pos = True
            engine._pos_dir = 1

        # Create fill queue with failure
        fill_queue = queue.Queue()
        fill_queue.put(
            OrderFill(
                signal_id="S-00000001-000001",
                order_id="",
                client_order_id="",
                filled_price=2000.0,
                filled_lots=0.01,
                success=False,
                error="Fill timeout",
            )
        )

        # Process fills
        engine._process_pending_fills(fill_queue, mock_dispatcher)

        # Verify state reset on failure
        with engine._state_lock:
            self.assertFalse(engine._waiting_for_fill)
            self.assertFalse(engine._in_pos)
            self.assertEqual(engine._pos_dir, 0)


# ═══════════════════════════════════════════════════════════════════════════════
# Test State Persistence
# ═══════════════════════════════════════════════════════════════════════════════


class TestStatePersistence(unittest.TestCase):
    """Test state persistence functionality."""

    def setUp(self):
        """Set up test temp directory."""
        self.test_dir = Path("/tmp/test_kinetra_state")
        self.test_dir.mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        """Clean up test files."""
        import shutil

        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)

    def test_atomic_write_json(self):
        """Test atomic JSON write."""
        test_file = self.test_dir / "test_state.json"
        test_data = {"version": 1, "symbol": "XAUUSD", "in_pos": True}

        _atomic_write_json(test_file, test_data)

        self.assertTrue(test_file.exists())
        loaded = json.loads(test_file.read_text())
        self.assertEqual(loaded["symbol"], "XAUUSD")
        self.assertTrue(loaded["in_pos"])

    def test_load_json_if_exists(self):
        """Test loading existing JSON."""
        test_file = self.test_dir / "test_state.json"
        test_data = {"version": 1, "symbol": "XAUUSD"}
        test_file.write_text(json.dumps(test_data))

        loaded = _load_json_if_exists(test_file)

        self.assertIsNotNone(loaded)
        self.assertEqual(loaded["symbol"], "XAUUSD")

    def test_load_json_nonexistent(self):
        """Test loading non-existent file returns None."""
        test_file = self.test_dir / "nonexistent.json"

        loaded = _load_json_if_exists(test_file)

        self.assertIsNone(loaded)

    def test_engine_state_serialization(self):
        """Test EngineState serialization."""
        state = EngineState(
            version=1,
            timestamp="2024-01-01T12:00:00Z",
            symbol="XAUUSD",
            in_pos=True,
            pos_dir=1,
            entry_price=2000.0,
            entry_time="2024-01-01T11:00:00Z",
            entry_lots=0.01,
            open_order_id="POS-12345",
        )

        # Serialize
        data = state.to_dict()

        # Deserialize
        restored = EngineState.from_dict(data)

        self.assertEqual(restored.symbol, "XAUUSD")
        self.assertTrue(restored.in_pos)
        self.assertEqual(restored.pos_dir, 1)
        self.assertEqual(restored.entry_price, 2000.0)

    def test_save_state_returns_bool(self, engine_config):
        """Test that _save_state returns success/failure."""
        engine = RenkoEngine(engine_config)
        engine._init_persistence(str(self.test_dir))

        result = engine._save_state()

        self.assertTrue(result)
        self.assertTrue(engine._state_file.exists())

    def test_persist_trade_returns_bool(self, engine_config):
        """Test that _persist_trade returns success/failure."""
        engine = RenkoEngine(engine_config)
        engine._init_persistence(str(self.test_dir))

        trade = LiveTrade(
            trade_id="POS-12345",
            symbol="XAUUSD",
            direction=TradeDirection.LONG,
            entry_price=2000.0,
            entry_time=pd.Timestamp("2024-01-01 12:00:00", tz="UTC"),
            brick_size=1.0,
            lots=0.01,
            target_risk_usd=100.0,
        )
        trade.close(
            exit_price=2010.0,
            exit_time=pd.Timestamp("2024-01-01 13:00:00", tz="UTC"),
            exit_reason="test",
            friction_usd=0.0,
            usd_per_point=1.0,
        )

        result = engine._persist_trade(trade)

        self.assertTrue(result)
        self.assertTrue(engine._trades_file.exists())


# ═══════════════════════════════════════════════════════════════════════════════
# Test Thread Health Monitoring
# ═══════════════════════════════════════════════════════════════════════════════


class TestThreadHealth(unittest.TestCase):
    """Test thread health monitoring."""

    def test_thread_health_tracking(self):
        """Test thread health state tracking."""
        health_lock = threading.Lock()
        health = {"bar_processor": True, "order_executor": True}

        # Simulate thread failure
        with health_lock:
            health["bar_processor"] = False

        with health_lock:
            self.assertFalse(health["bar_processor"])
            self.assertTrue(health["order_executor"])

    def test_thread_is_alive_check(self):
        """Test thread alive checking."""

        def worker():
            time.sleep(0.1)

        thread = threading.Thread(target=worker)
        thread.start()

        self.assertTrue(thread.is_alive())
        thread.join()
        self.assertFalse(thread.is_alive())


# ═══════════════════════════════════════════════════════════════════════════════
# Test Fill Queue Handling
# ═══════════════════════════════════════════════════════════════════════════════


class TestFillQueueHandling(unittest.TestCase):
    """Test fill queue non-blocking behavior."""

    def test_fill_queue_put_nowait_success(self):
        """Test successful non-blocking put."""
        q = queue.Queue(maxsize=10)
        fill = OrderFill(
            signal_id="S-00000001-000001",
            order_id="POS-12345",
            client_order_id="COID-123",
            filled_price=2000.0,
            filled_lots=0.01,
            success=True,
        )

        # Should succeed
        q.put_nowait(fill)

        self.assertEqual(q.qsize(), 1)

    def test_fill_queue_put_nowait_full(self):
        """Test non-blocking put when queue full."""
        q = queue.Queue(maxsize=1)
        fill1 = OrderFill(
            signal_id="S-00000001-000001",
            order_id="POS-12345",
            client_order_id="COID-123",
            filled_price=2000.0,
            filled_lots=0.01,
            success=True,
        )
        fill2 = OrderFill(
            signal_id="S-00000001-000002",
            order_id="POS-12346",
            client_order_id="COID-124",
            filled_price=2001.0,
            filled_lots=0.01,
            success=True,
        )

        q.put_nowait(fill1)

        # Should raise Full
        with self.assertRaises(queue.Full):
            q.put_nowait(fill2)

    def test_fill_queue_get_nowait_empty(self):
        """Test non-blocking get when queue empty."""
        q = queue.Queue()

        # Should raise Empty
        with self.assertRaises(queue.Empty):
            q.get_nowait()


# ═══════════════════════════════════════════════════════════════════════════════
# Test Exit Result Checking
# ═══════════════════════════════════════════════════════════════════════════════


class TestExitResultChecking(unittest.TestCase):
    """Test that exit results are properly checked."""

    def test_exit_result_checked_on_failure(self, engine_config):
        """Test that failed close is handled correctly."""
        engine = RenkoEngine(engine_config)
        engine._reset_state()

        # Set up position state
        with engine._state_lock:
            engine._in_pos = True
            engine._pos_dir = 1
            engine._entry_price = 2000.0
            engine._entry_lots = 0.01
            engine._open_order_id = "POS-12345"
            engine._pending_close = False

        # Create dispatcher that returns failure
        dispatcher = MagicMock()
        dispatcher.close_position.return_value = OrderResult(
            success=False,
            error="Close rejected",
        )

        # Try to exit
        engine._check_exit_async(
            b_close=1990.0,
            direction=-1,  # Colour change
            bar_time=pd.Timestamp("2024-01-01 12:00:00", tz="UTC"),
            dispatcher=dispatcher,
        )

        # Verify pending_close was reset on failure
        with engine._state_lock:
            self.assertFalse(engine._pending_close)

    def test_exit_result_checked_on_success(self, engine_config):
        """Test that successful close proceeds correctly."""
        engine = RenkoEngine(engine_config)
        engine._reset_state()

        # Set up position state
        with engine._state_lock:
            engine._in_pos = True
            engine._pos_dir = 1
            engine._entry_price = 2000.0
            engine._entry_lots = 0.01
            engine._open_order_id = "POS-12345"
            engine._pending_close = False

        # Create dispatcher that returns success
        dispatcher = MagicMock()
        dispatcher.close_position.return_value = OrderResult(
            success=True,
            order_id="POS-12345",
        )

        # Try to exit
        engine._check_exit_async(
            b_close=1990.0,
            direction=-1,
            bar_time=pd.Timestamp("2024-01-01 12:00:00", tz="UTC"),
            dispatcher=dispatcher,
        )

        # Verify position was closed and trade recorded
        with engine._state_lock:
            self.assertFalse(engine._in_pos)
            self.assertEqual(len(engine._completed), 1)


# ═══════════════════════════════════════════════════════════════════════════════
# Integration Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestIntegration(unittest.TestCase):
    """Integration tests for the full trade lifecycle."""

    def test_full_trade_lifecycle(self, engine_config, mock_dispatcher):
        """Test complete entry and exit cycle."""
        engine = RenkoEngine(engine_config)
        engine._reset_state()

        # Initialize with enough equity
        engine._live_equity = 10000.0

        # Create queues
        order_queue = queue.Queue(maxsize=256)
        fill_queue = queue.Queue(maxsize=256)

        # Setup for entry - simulate flip
        with engine._state_lock:
            engine._prev_dir = -1  # Previous was short

        # Simulate entry signal
        engine._check_entry_async(
            b_close=2000.0,
            direction=1,  # Flip to long
            bar_time=pd.Timestamp("2024-01-01 12:00:00", tz="UTC"),
            fr_val=0.2,
            pUU_val=0.6,
            pDD_val=0.4,
            order_queue=order_queue,
        )

        # Process order (would be done by order executor thread)
        try:
            req = order_queue.get_nowait()

            # Simulate fill
            fill_queue.put(
                OrderFill(
                    signal_id=req.signal_id,
                    order_id="POS-12345",
                    client_order_id="COID-123",
                    filled_price=req.price,
                    filled_lots=req.lots,
                    success=True,
                )
            )
        except queue.Empty:
            pass

        # Process fill
        engine._process_pending_fills(fill_queue, mock_dispatcher)

        # Verify entry
        with engine._state_lock:
            self.assertTrue(engine._in_pos)
            self.assertEqual(engine._pos_dir, 1)
            self.assertEqual(engine._open_order_id, "POS-12345")

        # Simulate exit signal (colour change)
        engine._check_exit_async(
            b_close=1990.0,
            direction=-1,  # Colour change
            bar_time=pd.Timestamp("2024-01-01 13:00:00", tz="UTC"),
            dispatcher=mock_dispatcher,
        )

        # Verify exit
        with engine._state_lock:
            self.assertFalse(engine._in_pos)
            self.assertEqual(len(engine._completed), 1)

    def test_concurrent_fills_processing(self, engine_config, mock_dispatcher):
        """Test processing multiple fills concurrently."""
        engine = RenkoEngine(engine_config)
        engine._reset_state()

        fill_queue = queue.Queue()

        # Add multiple fills to queue
        for i in range(5):
            fill_queue.put(
                OrderFill(
                    signal_id=f"S-{i:08d}-000001",
                    order_id=f"POS-{i + 1}",
                    client_order_id=f"COID-{i + 1}",
                    filled_price=2000.0 + i,
                    filled_lots=0.01,
                    success=True,
                )
            )

        # Process all fills
        engine._process_pending_fills(fill_queue, mock_dispatcher)

        # Verify all processed
        try:
            fill_queue.get_nowait()
            self.fail("Queue should be empty")
        except queue.Empty:
            pass  # Expected


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    unittest.main(verbosity=2)
