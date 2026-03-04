"""
Health Monitoring Tests
=======================

Tests for connection health monitoring and preflight checks.

Usage:
    python -m pytest tests/test_health_monitoring.py -v
"""

from __future__ import annotations

import sys
import threading
import time
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from kinetra.monitoring.connection_health import (
    ConnectionHealthService,
    HealthCheckResult,
    HealthMetrics,
    HealthStatus,
    format_health_report,
)
from kinetra.preflight_enhanced import (
    CheckSeverity,
    EnhancedPreflight,
    PreflightCheck,
    PreflightConfig,
    PreflightResult,
)

# ═══════════════════════════════════════════════════════════════════════════════
# Connection Health Service Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestConnectionHealthService(unittest.TestCase):
    """Test ConnectionHealthService functionality."""

    def setUp(self):
        """Set up mock connector."""
        self.connector = MagicMock()
        self.connector.is_connected.return_value = True

    def test_health_service_initialization(self):
        """Test health service initializes correctly."""
        service = ConnectionHealthService(self.connector)

        self.assertEqual(service._connector, self.connector)
        self.assertFalse(service._running)

    def test_health_status_healthy(self):
        """Test healthy status evaluation."""
        service = ConnectionHealthService(
            self.connector,
            latency_threshold_ms=500.0,
            latency_critical_ms=1000.0,
        )

        # Create healthy metrics
        metrics = HealthMetrics(
            timestamp_utc=datetime.now(timezone.utc),
            latency_ms=100.0,
            latency_samples=[100.0, 110.0, 105.0],
            packet_loss_rate=0.0,
            heartbeat_success_rate=1.0,
            consecutive_failures=0,
            total_checks=10,
        )

        status = service._evaluate_status(metrics)

        self.assertEqual(status, HealthStatus.HEALTHY)

    def test_health_status_degraded_high_latency(self):
        """Test degraded status on high latency."""
        service = ConnectionHealthService(
            self.connector,
            latency_threshold_ms=100.0,
            latency_critical_ms=500.0,
        )

        metrics = HealthMetrics(
            timestamp_utc=datetime.now(timezone.utc),
            latency_ms=200.0,  # Above threshold
            latency_samples=[200.0],
            packet_loss_rate=0.0,
            heartbeat_success_rate=1.0,
            consecutive_failures=0,
            total_checks=10,
        )

        status = service._evaluate_status(metrics)

        self.assertEqual(status, HealthStatus.DEGRADED)

    def test_health_status_critical_on_failures(self):
        """Test critical status on consecutive failures."""
        service = ConnectionHealthService(
            self.connector,
            consecutive_failures_threshold=3,
        )

        self.connector.is_connected.return_value = True

        metrics = HealthMetrics(
            timestamp_utc=datetime.now(timezone.utc),
            latency_ms=100.0,
            consecutive_failures=3,  # At threshold
            total_checks=10,
        )

        status = service._evaluate_status(metrics)

        self.assertEqual(status, HealthStatus.CRITICAL)

    def test_health_status_critical_on_disconnect(self):
        """Test critical status when not connected."""
        service = ConnectionHealthService(self.connector)

        self.connector.is_connected.return_value = False

        metrics = HealthMetrics(
            timestamp_utc=datetime.now(timezone.utc),
            latency_ms=100.0,
            consecutive_failures=0,
            total_checks=10,
        )

        status = service._evaluate_status(metrics)

        self.assertEqual(status, HealthStatus.CRITICAL)

    def test_should_failover_on_critical(self):
        """Test preemptive failover triggers on critical."""
        service = ConnectionHealthService(
            self.connector,
            enable_preemptive_failover=True,
        )

        # Mock connector with failover capability
        self.connector._atomic_failover = MagicMock(return_value=True)

        metrics = HealthMetrics(
            timestamp_utc=datetime.now(timezone.utc),
            latency_ms=100.0,
            consecutive_failures=0,
            total_checks=10,
        )
        metrics.status = HealthStatus.CRITICAL

        should_failover = service._should_failover(metrics)

        self.assertTrue(should_failover)

    def test_should_not_failover_when_disabled(self):
        """Test no failover when disabled."""
        service = ConnectionHealthService(
            self.connector,
            enable_preemptive_failover=False,
        )

        metrics = HealthMetrics(
            timestamp_utc=datetime.now(timezone.utc),
            latency_ms=100.0,
            consecutive_failures=0,
            total_checks=10,
        )
        metrics.status = HealthStatus.CRITICAL

        should_failover = service._should_failover(metrics)

        self.assertFalse(should_failover)

    def test_packet_loss_calculation(self):
        """Test packet loss rate calculation."""
        service = ConnectionHealthService(self.connector)

        # Simulate heartbeat history with failures
        service._heartbeat_history = [True, True, False, True, False]

        loss_rate = service._calculate_packet_loss()

        self.assertEqual(loss_rate, 0.4)  # 2/5 failures

    def test_latency_statistics(self):
        """Test latency statistics calculation."""
        metrics = HealthMetrics(
            timestamp_utc=datetime.now(timezone.utc),
            latency_ms=100.0,
            latency_samples=[50.0, 100.0, 150.0, 200.0, 250.0],
        )

        self.assertAlmostEqual(metrics.latency_std_ms, 79.06, places=1)
        self.assertEqual(metrics.latency_p95_ms, 250.0)

    def test_status_change_callback(self):
        """Test status change callback is triggered."""
        service = ConnectionHealthService(self.connector)

        callback_mock = MagicMock()
        service.on_status_change(callback_mock)

        # Simulate status change
        service._notify_status_change(HealthStatus.HEALTHY, HealthStatus.DEGRADED)

        callback_mock.assert_called_once_with(HealthStatus.HEALTHY, HealthStatus.DEGRADED)

    def test_format_health_report(self):
        """Test health report formatting."""
        metrics = HealthMetrics(
            timestamp_utc=datetime.now(timezone.utc),
            latency_ms=100.0,
            latency_samples=[100.0, 110.0],
            packet_loss_rate=0.0,
            heartbeat_success_rate=1.0,
            consecutive_failures=0,
            total_checks=10,
        )

        result = HealthCheckResult(
            status=HealthStatus.HEALTHY,
            metrics=metrics,
            checks={
                "connected": (True, "connected"),
                "latency_ok": (True, "latency 100.0ms < 500.0ms"),
            },
            recommendations=[],
        )

        report = format_health_report(result)

        self.assertIn("Health Status: HEALTHY", report)
        self.assertIn("Latency: 100.0ms", report)
        self.assertIn("connected", report)


# ═══════════════════════════════════════════════════════════════════════════════
# Preflight Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestPreflightChecks(unittest.TestCase):
    """Test enhanced preflight checks."""

    def setUp(self):
        """Set up mock connector."""
        self.connector = MagicMock()
        self.connector.is_connected.return_value = True
        self.connector.selected_endpoint = "demo.ctraderapi.com:5035"

        def mock_get_account_snapshot(**kwargs):
            return {
                "balance": 1000.0,
                "equity": 1000.0,
                "margin_used": 0.0,
                "account_id": 12345,
                "broker_name": "Pepperstone",
            }

        self.connector.get_account_snapshot.side_effect = mock_get_account_snapshot
        self.connector.find_symbol_id.return_value = 41
        self.connector.get_digits.return_value = 2

    def test_preflight_config_defaults(self):
        """Test preflight config defaults."""
        config = PreflightConfig(symbol="XAUUSD")

        self.assertEqual(config.symbol, "XAUUSD")
        self.assertEqual(config.min_balance_usd, 100.0)
        self.assertTrue(config.validate_dns)
        self.assertTrue(config.check_market_hours)

    def test_preflight_all_checks_pass(self):
        """Test preflight when all checks pass."""
        config = PreflightConfig(
            symbol="XAUUSD",
            enable_health_service=False,
            validate_dns=False,  # Skip DNS for unit test
        )

        preflight = EnhancedPreflight(self.connector, config)

        # Run checks
        result = preflight.run_all_checks()

        self.assertTrue(result.can_trade)
        self.assertEqual(result.passed_count, result.total_count)
        self.assertEqual(len(result.blocking_reasons), 0)

    def test_preflight_fails_on_low_balance(self):
        """Test preflight blocks on insufficient balance."""
        # Mock low balance
        self.connector.get_account_snapshot.return_value = {
            "balance": 50.0,  # Below minimum
            "equity": 50.0,
        }

        config = PreflightConfig(
            symbol="XAUUSD",
            min_balance_usd=100.0,
            enable_health_service=False,
            validate_dns=False,
        )

        preflight = EnhancedPreflight(self.connector, config)
        result = preflight.run_all_checks()

        self.assertFalse(result.can_trade)
        self.assertTrue(any("balance" in reason.lower() for reason in result.blocking_reasons))

    def test_preflight_fails_on_not_connected(self):
        """Test preflight blocks when not connected."""
        self.connector.is_connected.return_value = False

        config = PreflightConfig(
            symbol="XAUUSD",
            enable_health_service=False,
            validate_dns=False,
        )

        preflight = EnhancedPreflight(self.connector, config)
        result = preflight.run_all_checks()

        self.assertFalse(result.can_trade)

    def test_preflight_fails_on_symbol_not_found(self):
        """Test preflight blocks when symbol not found."""
        self.connector.find_symbol_id.return_value = None

        config = PreflightConfig(
            symbol="INVALID",
            enable_health_service=False,
            validate_dns=False,
        )

        preflight = EnhancedPreflight(self.connector, config)
        result = preflight.run_all_checks()

        self.assertFalse(result.can_trade)
        self.assertTrue(any("symbol" in reason.lower() for reason in result.blocking_reasons))

    def test_preflight_result_properties(self):
        """Test PreflightResult properties."""
        check1 = PreflightCheck(
            name="test1",
            passed=True,
            severity=CheckSeverity.INFO,
            message="OK",
            duration_ms=10.0,
        )
        check2 = PreflightCheck(
            name="test2",
            passed=False,
            severity=CheckSeverity.WARNING,
            message="Warning",
            duration_ms=20.0,
        )
        check3 = PreflightCheck(
            name="test3",
            passed=False,
            severity=CheckSeverity.BLOCKING,
            message="Blocking",
            duration_ms=30.0,
        )

        result = PreflightResult(
            timestamp_utc=datetime.now(timezone.utc),
            checks=[check1, check2, check3],
            total_duration_ms=60.0,
        )

        self.assertFalse(result.can_trade)
        self.assertEqual(result.passed_count, 1)
        self.assertEqual(result.total_count, 3)
        self.assertEqual(len(result.blocking_reasons), 1)
        self.assertEqual(len(result.warnings), 1)
        self.assertEqual(result.get_check("test1").passed, True)
        self.assertIsNone(result.get_check("nonexistent"))

    def test_preflight_check_is_blocking(self):
        """Test PreflightCheck is_blocking property."""
        blocking = PreflightCheck(
            name="blocking",
            passed=False,
            severity=CheckSeverity.BLOCKING,
            message="Error",
        )
        warning = PreflightCheck(
            name="warning",
            passed=False,
            severity=CheckSeverity.WARNING,
            message="Warning",
        )
        passed = PreflightCheck(
            name="passed",
            passed=True,
            severity=CheckSeverity.INFO,
            message="OK",
        )

        self.assertTrue(blocking.is_blocking)
        self.assertFalse(warning.is_blocking)
        self.assertFalse(passed.is_blocking)

    def test_preflight_format_report(self):
        """Test preflight report formatting."""
        checks = [
            PreflightCheck(
                name="check1",
                passed=True,
                severity=CheckSeverity.INFO,
                message="OK",
            ),
            PreflightCheck(
                name="check2",
                passed=False,
                severity=CheckSeverity.BLOCKING,
                message="Failed",
            ),
        ]

        from kinetra.preflight_enhanced import format_preflight_report

        result = PreflightResult(
            timestamp_utc=datetime.now(timezone.utc),
            checks=checks,
            total_duration_ms=100.0,
        )

        report = format_preflight_report(result, verbose=True)

        self.assertIn("PREFLIGHT CHECK RESULTS", report)
        self.assertIn("BLOCKING", report)
        self.assertIn("Failed", report)


# ═══════════════════════════════════════════════════════════════════════════════
# Integration Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestHealthIntegration(unittest.TestCase):
    """Integration tests for health monitoring."""

    def test_health_service_start_stop(self):
        """Test health service start and stop."""
        connector = MagicMock()
        connector.is_connected.return_value = True

        service = ConnectionHealthService(connector)

        # Start monitoring
        service.start_monitoring()
        self.assertTrue(service._running)
        self.assertIsNotNone(service._monitor_thread)

        # Let it run briefly
        time.sleep(0.2)

        # Stop monitoring
        service.stop_monitoring()
        self.assertFalse(service._running)

    def test_force_health_check(self):
        """Test forced health check."""
        connector = MagicMock()
        connector.is_connected.return_value = True

        service = ConnectionHealthService(connector)

        result = service.force_health_check()

        self.assertIsInstance(result, HealthCheckResult)
        self.assertIsNotNone(result.status)
        self.assertIsNotNone(result.metrics)

    def test_health_check_with_telemetry(self):
        """Test health check with telemetry emission."""
        connector = MagicMock()
        telemetry = MagicMock()

        service = ConnectionHealthService(connector)
        service.set_telemetry(telemetry)

        # Perform check
        result = service.force_health_check()

        # Verify telemetry was called
        telemetry.emit.assert_called_once()
        call_args = telemetry.emit.call_args
        self.assertEqual(call_args.kwargs["stream"], "connection_health")


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    unittest.main(verbosity=2)
