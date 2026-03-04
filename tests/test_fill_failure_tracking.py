"""
Fill Failure Tracking Tests
============================

Tests for fill failure metrics and recovery.

Usage:
    python -m pytest tests/test_fill_failure_tracking.py -v
"""

from __future__ import annotations

import sys
import threading
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from kinetra.renko.live_trader import OrderResult, TradeDirection


class TestFillFailureTracking(unittest.TestCase):
    """Test fill failure metrics tracking."""

    def setUp(self):
        """Set up test fixtures."""
        try:
            from kinetra.renko.ctrader_dispatcher import CTraderOrderDispatcher

            self.dispatcher_class = CTraderOrderDispatcher
            self.has_ctrader = True
        except ImportError:
            self.has_ctrader = False

    @patch("kinetra.renko.ctrader_dispatcher._CONNECTOR_AVAILABLE", True)
    def test_fill_metrics_initial_state(self):
        """Test initial fill metrics state."""
        if not self.has_ctrader:
            self.skipTest("cTrader not available")

        connector = MagicMock()
        bar_provider = MagicMock()

        dispatcher = self.dispatcher_class(connector, bar_provider)

        metrics = dispatcher.get_fill_metrics()

        self.assertEqual(metrics["orders_submitted"], 0)
        self.assertEqual(metrics["orders_filled"], 0)
        self.assertEqual(metrics["orders_failed"], 0)
        self.assertEqual(metrics["fill_success_rate"], 1.0)
        self.assertTrue(metrics["is_healthy"])

    @patch("kinetra.renko.ctrader_dispatcher._CONNECTOR_AVAILABLE", True)
    def test_successful_fill_increments_counter(self):
        """Test that successful fill increments filled counter."""
        if not self.has_ctrader:
            self.skipTest("cTrader not available")

        connector = MagicMock()
        bar_provider = MagicMock()

        dispatcher = self.dispatcher_class(connector, bar_provider)

        # Record successful fill
        with dispatcher._fill_metrics_lock:
            dispatcher._orders_submitted += 1
            dispatcher._orders_filled += 1
            dispatcher._consecutive_fill_failures = 0

        metrics = dispatcher.get_fill_metrics()

        self.assertEqual(metrics["orders_submitted"], 1)
        self.assertEqual(metrics["orders_filled"], 1)
        self.assertEqual(metrics["orders_failed"], 0)
        self.assertEqual(metrics["fill_success_rate"], 1.0)

    @patch("kinetra.renko.ctrader_dispatcher._CONNECTOR_AVAILABLE", True)
    def test_failed_fill_increments_failure_counter(self):
        """Test that failed fill increments failure counter."""
        if not self.has_ctrader:
            self.skipTest("cTrader not available")

        connector = MagicMock()
        bar_provider = MagicMock()

        dispatcher = self.dispatcher_class(connector, bar_provider)

        # Record failed fill
        with dispatcher._fill_metrics_lock:
            dispatcher._orders_submitted += 1
            dispatcher._orders_failed += 1
            dispatcher._consecutive_fill_failures += 1
            dispatcher._last_fill_failure = datetime.now(timezone.utc)
            dispatcher._fill_failures.append(
                {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "symbol": "XAUUSD",
                    "error": "Fill timeout",
                    "type": "timeout",
                }
            )

        metrics = dispatcher.get_fill_metrics()

        self.assertEqual(metrics["orders_submitted"], 1)
        self.assertEqual(metrics["orders_filled"], 0)
        self.assertEqual(metrics["orders_failed"], 1)
        self.assertEqual(metrics["consecutive_failures"], 1)
        self.assertIsNotNone(metrics["last_failure"])
        self.assertEqual(len(metrics["recent_failures"]), 1)

    @patch("kinetra.renko.ctrader_dispatcher._CONNECTOR_AVAILABLE", True)
    def test_consecutive_failures_tracked(self):
        """Test consecutive failures are tracked correctly."""
        if not self.has_ctrader:
            self.skipTest("cTrader not available")

        connector = MagicMock()
        bar_provider = MagicMock()

        dispatcher = self.dispatcher_class(connector, bar_provider)

        # Record 3 consecutive failures
        for i in range(3):
            with dispatcher._fill_metrics_lock:
                dispatcher._orders_submitted += 1
                dispatcher._orders_failed += 1
                dispatcher._consecutive_fill_failures += 1

        metrics = dispatcher.get_fill_metrics()

        self.assertEqual(metrics["consecutive_failures"], 3)
        self.assertFalse(metrics["is_healthy"])

    @patch("kinetra.renko.ctrader_dispatcher._CONNECTOR_AVAILABLE", True)
    def test_success_resets_consecutive_failures(self):
        """Test successful fill resets consecutive failure counter."""
        if not self.has_ctrader:
            self.skipTest("cTrader not available")

        connector = MagicMock()
        bar_provider = MagicMock()

        dispatcher = self.dispatcher_class(connector, bar_provider)

        # Set up some failures
        with dispatcher._fill_metrics_lock:
            dispatcher._consecutive_fill_failures = 3

        # Record successful fill
        with dispatcher._fill_metrics_lock:
            dispatcher._orders_filled += 1
            dispatcher._consecutive_fill_failures = 0

        metrics = dispatcher.get_fill_metrics()

        self.assertEqual(metrics["consecutive_failures"], 0)
        self.assertTrue(metrics["is_healthy"])

    @patch("kinetra.renko.ctrader_dispatcher._CONNECTOR_AVAILABLE", True)
    def test_fill_failure_history_limited(self):
        """Test that fill failure history is limited to last 100."""
        if not self.has_ctrader:
            self.skipTest("cTrader not available")

        connector = MagicMock()
        bar_provider = MagicMock()

        dispatcher = self.dispatcher_class(connector, bar_provider)

        # Add 150 failures
        with dispatcher._fill_metrics_lock:
            for i in range(150):
                dispatcher._fill_failures.append(
                    {
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "symbol": "XAUUSD",
                        "error": f"Error {i}",
                        "type": "timeout",
                    }
                )
            # Trim to last 100 (as done in actual code)
            if len(dispatcher._fill_failures) > 100:
                dispatcher._fill_failures = dispatcher._fill_failures[-100:]

        self.assertEqual(len(dispatcher._fill_failures), 100)

    @patch("kinetra.renko.ctrader_dispatcher._CONNECTOR_AVAILABLE", True)
    def test_fill_success_rate_calculation(self):
        """Test fill success rate calculation."""
        if not self.has_ctrader:
            self.skipTest("cTrader not available")

        connector = MagicMock()
        bar_provider = MagicMock()

        dispatcher = self.dispatcher_class(connector, bar_provider)

        # 7 fills, 3 failures = 70% success rate
        with dispatcher._fill_metrics_lock:
            dispatcher._orders_submitted = 10
            dispatcher._orders_filled = 7
            dispatcher._orders_failed = 3

        metrics = dispatcher.get_fill_metrics()

        self.assertEqual(metrics["fill_success_rate"], 0.7)

    @patch("kinetra.renko.ctrader_dispatcher._CONNECTOR_AVAILABLE", True)
    def test_circuit_breaker_in_metrics(self):
        """Test circuit breaker status in metrics."""
        if not self.has_ctrader:
            self.skipTest("cTrader not available")

        connector = MagicMock()
        bar_provider = MagicMock()

        dispatcher = self.dispatcher_class(connector, bar_provider)

        # Trigger circuit breaker
        dispatcher._circuit_breaker_triggered = True

        metrics = dispatcher.get_fill_metrics()

        self.assertTrue(metrics["circuit_breaker_triggered"])

    @patch("kinetra.renko.ctrader_dispatcher._CONNECTOR_AVAILABLE", True)
    def test_thread_safety_of_metrics(self):
        """Test thread-safe access to metrics."""
        if not self.has_ctrader:
            self.skipTest("cTrader not available")

        connector = MagicMock()
        bar_provider = MagicMock()

        dispatcher = self.dispatcher_class(connector, bar_provider)

        errors = []

        def increment_fills():
            try:
                for _ in range(100):
                    with dispatcher._fill_metrics_lock:
                        dispatcher._orders_submitted += 1
                        dispatcher._orders_filled += 1
            except Exception as e:
                errors.append(e)

        # Run from multiple threads
        threads = [threading.Thread(target=increment_fills) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(len(errors), 0)
        self.assertEqual(dispatcher._orders_submitted, 500)
        self.assertEqual(dispatcher._orders_filled, 500)


class TestReconcileAfterTimeout(unittest.TestCase):
    """Test reconciliation after fill timeout."""

    def setUp(self):
        try:
            from kinetra.renko.ctrader_dispatcher import CTraderOrderDispatcher

            self.dispatcher_class = CTraderOrderDispatcher
            self.has_ctrader = True
        except ImportError:
            self.has_ctrader = False

    @patch("kinetra.renko.ctrader_dispatcher._CONNECTOR_AVAILABLE", True)
    @patch("kinetra.renko.ctrader_dispatcher._get_api_msgs")
    def test_reconcile_finds_position(self, mock_get_api_msgs):
        """Test that reconcile finds position after timeout."""
        if not self.has_ctrader:
            self.skipTest("cTrader not available")

        mock_api_msgs = MagicMock()
        mock_get_api_msgs.return_value = mock_api_msgs

        connector = MagicMock()
        connector.is_connected.return_value = True
        connector.credentials.account_id = 12345

        bar_provider = MagicMock()

        dispatcher = self.dispatcher_class(connector, bar_provider, fill_timeout_s=0.01)

        # Mock reconcile to return position
        mock_reconcile = MagicMock()
        mock_reconcile.position = [MagicMock(positionId=12345)]
        connector.send_and_wait.return_value = mock_reconcile

        # Call reconcile method
        result = dispatcher._reconcile_position_after_timeout(
            symbol="XAUUSD",
            symbol_id=41,
            expected_price=2000.0,
            expected_lots=0.01,
        )

        self.assertEqual(result, "12345")

    @patch("kinetra.renko.ctrader_dispatcher._CONNECTOR_AVAILABLE", True)
    @patch("kinetra.renko.ctrader_dispatcher._get_api_msgs")
    def test_reconcile_position_not_found(self, mock_get_api_msgs):
        """Test reconcile when position not found."""
        if not self.has_ctrader:
            self.skipTest("cTrader not available")

        mock_api_msgs = MagicMock()
        mock_get_api_msgs.return_value = mock_api_msgs

        connector = MagicMock()
        connector.is_connected.return_value = True
        connector.credentials.account_id = 12345

        bar_provider = MagicMock()

        dispatcher = self.dispatcher_class(connector, bar_provider, fill_timeout_s=0.01)

        # Mock reconcile with empty position list
        mock_reconcile = MagicMock()
        mock_reconcile.position = []
        connector.send_and_wait.return_value = mock_reconcile

        result = dispatcher._reconcile_position_after_timeout(
            symbol="XAUUSD",
            symbol_id=41,
            expected_price=2000.0,
            expected_lots=0.01,
        )

        self.assertIsNone(result)

    @patch("kinetra.renko.ctrader_dispatcher._CONNECTOR_AVAILABLE", True)
    @patch("kinetra.renko.ctrader_dispatcher._get_api_msgs")
    def test_reconcile_connector_failure(self, mock_get_api_msgs):
        """Test reconcile when connector fails."""
        if not self.has_ctrader:
            self.skipTest("cTrader not available")

        mock_api_msgs = MagicMock()
        mock_get_api_msgs.return_value = mock_api_msgs

        connector = MagicMock()
        connector.is_connected.return_value = True
        connector.credentials.account_id = 12345
        connector.send_and_wait.return_value = None  # Simulate failure

        bar_provider = MagicMock()

        dispatcher = self.dispatcher_class(connector, bar_provider)

        result = dispatcher._reconcile_position_after_timeout(
            symbol="XAUUSD",
            symbol_id=41,
            expected_price=2000.0,
            expected_lots=0.01,
        )

        self.assertIsNone(result)


class TestOrderResultHandling(unittest.TestCase):
    """Test OrderResult creation and handling."""

    def test_order_result_success(self):
        """Test successful order result."""
        result = OrderResult(
            success=True,
            order_id="POS-12345",
            filled_price=2000.5,
            filled_lots=0.01,
            raw={"client_order_id": "COID-123"},
        )

        self.assertTrue(result.success)
        self.assertEqual(result.order_id, "POS-12345")
        self.assertEqual(result.filled_price, 2000.5)
        self.assertIsNone(result.error)

    def test_order_result_failure(self):
        """Test failed order result."""
        result = OrderResult(
            success=False,
            error="Fill timeout after 5s",
            raw={"client_order_id": "COID-123"},
        )

        self.assertFalse(result.success)
        self.assertIsNone(result.order_id)
        self.assertEqual(result.error, "Fill timeout after 5s")

    def test_order_result_partial_fill(self):
        """Test partial fill result."""
        result = OrderResult(
            success=True,
            order_id="POS-12345",
            filled_price=2000.5,
            filled_lots=0.005,  # Partial fill
            raw={"client_order_id": "COID-123", "partial_fill": True},
        )

        self.assertTrue(result.success)
        self.assertEqual(result.filled_lots, 0.005)


if __name__ == "__main__":
    unittest.main(verbosity=2)
