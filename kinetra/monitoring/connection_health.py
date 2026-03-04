"""
Connection Health Monitoring Service
=====================================

Proactive health monitoring for broker connections with:
- Periodic heartbeat verification
- Latency trending and anomaly detection
- Packet loss monitoring
- Automatic preemptive failover
- Health metrics export to telemetry

Usage::

    from kinetra.monitoring.connection_health import ConnectionHealthService

    health_svc = ConnectionHealthService(connector)
    health_svc.start_monitoring()

    # Check health status
    status = health_svc.get_health_status()
    if status.is_healthy:
        print(f"Latency: {status.latency_ms:.1f}ms")

    health_svc.stop_monitoring()
"""

from __future__ import annotations

import logging
import statistics
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, auto
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Tuple

if TYPE_CHECKING:
    from kinetra.connectors.ctrader_connector import CTraderConnector, HotStandbyCTraderConnector

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """Connection health status."""

    HEALTHY = auto()  # All metrics within normal bounds
    DEGRADED = auto()  # Some metrics concerning but functional
    UNHEALTHY = auto()  # Metrics outside acceptable bounds
    CRITICAL = auto()  # Connection failure imminent or occurred
    UNKNOWN = auto()  # Not enough data to determine


@dataclass
class HealthMetrics:
    """Snapshot of connection health metrics."""

    timestamp_utc: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    latency_ms: float = 0.0
    latency_samples: List[float] = field(default_factory=list)
    packet_loss_rate: float = 0.0
    heartbeat_success_rate: float = 1.0
    consecutive_failures: int = 0
    total_failures: int = 0
    total_checks: int = 0
    last_heartbeat_utc: Optional[datetime] = None
    failover_count: int = 0

    @property
    def latency_std_ms(self) -> float:
        """Standard deviation of latency samples."""
        if len(self.latency_samples) < 2:
            return 0.0
        return statistics.stdev(self.latency_samples)

    @property
    def latency_p95_ms(self) -> float:
        """95th percentile latency."""
        if not self.latency_samples:
            return 0.0
        sorted_samples = sorted(self.latency_samples)
        idx = int(len(sorted_samples) * 0.95)
        return sorted_samples[min(idx, len(sorted_samples) - 1)]


@dataclass
class HealthCheckResult:
    """Result of a health check evaluation."""

    status: HealthStatus
    metrics: HealthMetrics
    checks: Dict[str, Tuple[bool, str]] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)

    @property
    def is_healthy(self) -> bool:
        return self.status == HealthStatus.HEALTHY

    @property
    def needs_action(self) -> bool:
        return self.status in (HealthStatus.UNHEALTHY, HealthStatus.CRITICAL)


class ConnectionHealthService:
    """
    Proactive health monitoring for broker connections.

    Monitors connection quality in real-time and triggers alerts or
    automatic failover before complete failure occurs.

    Parameters
    ----------
    connector : CTraderConnector or HotStandbyCTraderConnector
        The connector to monitor
    check_interval_seconds : float
        Seconds between health checks (default: 30)
    latency_window_size : int
        Number of samples to keep for latency statistics (default: 100)
    latency_threshold_ms : float
        Latency above which connection is considered degraded (default: 500)
    latency_critical_ms : float
        Latency above which connection is critical (default: 1000)
    packet_loss_threshold : float
        Packet loss rate above which to alert (default: 0.01 = 1%)
    consecutive_failures_threshold : int
        Failures before marking unhealthy (default: 3)
    enable_preemptive_failover : bool
        Whether to trigger failover on degradation (default: False)
    """

    # Default configuration
    DEFAULT_CHECK_INTERVAL_S = 30.0
    DEFAULT_LATENCY_THRESHOLD_MS = 500.0
    DEFAULT_LATENCY_CRITICAL_MS = 1000.0
    DEFAULT_PACKET_LOSS_THRESHOLD = 0.01
    DEFAULT_CONSECUTIVE_FAILURES = 3
    DEFAULT_LATENCY_WINDOW = 100

    def __init__(
        self,
        connector: CTraderConnector | HotStandbyCTraderConnector,
        check_interval_seconds: float = DEFAULT_CHECK_INTERVAL_S,
        latency_window_size: int = DEFAULT_LATENCY_WINDOW,
        latency_threshold_ms: float = DEFAULT_LATENCY_THRESHOLD_MS,
        latency_critical_ms: float = DEFAULT_LATENCY_CRITICAL_MS,
        packet_loss_threshold: float = DEFAULT_PACKET_LOSS_THRESHOLD,
        consecutive_failures_threshold: int = DEFAULT_CONSECUTIVE_FAILURES,
        enable_preemptive_failover: bool = False,
    ):
        self._connector = connector
        self._check_interval = check_interval_seconds
        self._latency_window = latency_window_size
        self._latency_threshold_ms = latency_threshold_ms
        self._latency_critical_ms = latency_critical_ms
        self._packet_loss_threshold = packet_loss_threshold
        self._consecutive_failures_threshold = consecutive_failures_threshold
        self._enable_preemptive_failover = enable_preemptive_failover

        # State
        self._running = False
        self._monitor_thread: Optional[threading.Thread] = None
        self._lock = threading.RLock()

        # Metrics history
        self._latency_history: List[float] = []
        self._heartbeat_history: List[bool] = []
        self._last_metrics = HealthMetrics()

        # Statistics
        self._total_checks = 0
        self._consecutive_failures = 0
        self._total_failures = 0
        self._last_heartbeat_time: Optional[datetime] = None

        # Callbacks
        self._on_status_change: List[Callable[[HealthStatus, HealthStatus], None]] = []
        self._on_degradation: List[Callable[[HealthMetrics], None]] = []
        self._on_failover_recommended: List[Callable[[str], None]] = []

        # Telemetry reference (set externally)
        self._telemetry = None

    def set_telemetry(self, telemetry) -> None:
        """Set telemetry sink for health events."""
        self._telemetry = telemetry

    def on_status_change(self, callback: Callable[[HealthStatus, HealthStatus], None]) -> None:
        """Register callback for status changes (old_status, new_status)."""
        self._on_status_change.append(callback)

    def on_degradation(self, callback: Callable[[HealthMetrics], None]) -> None:
        """Register callback for degradation detection."""
        self._on_degradation.append(callback)

    def on_failover_recommended(self, callback: Callable[[str], None]) -> None:
        """Register callback when failover is recommended (reason)."""
        self._on_failover_recommended.append(callback)

    def start_monitoring(self) -> None:
        """Start background health monitoring."""
        if self._running:
            return

        self._running = True
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()
        logger.info("[HealthService] Started monitoring (interval=%.1fs)", self._check_interval)

    def stop_monitoring(self) -> None:
        """Stop background health monitoring."""
        self._running = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=self._check_interval + 5.0)
        logger.info("[HealthService] Stopped monitoring")

    def _monitor_loop(self) -> None:
        """Main monitoring loop."""
        while self._running:
            try:
                self._perform_health_check()
            except Exception as exc:
                logger.error("[HealthService] Health check error: %s", exc)

            # Sleep with interrupt handling
            for _ in range(int(self._check_interval)):
                if not self._running:
                    break
                time.sleep(1.0)

    def _perform_health_check(self) -> None:
        """Execute a single health check cycle."""
        check_time = datetime.now(timezone.utc)

        # Check 1: Connection state
        is_connected = self._check_connection()

        # Check 2: Heartbeat latency
        heartbeat_ok, latency_ms = self._check_heartbeat()

        # Check 3: Update metrics
        with self._lock:
            self._total_checks += 1

            if heartbeat_ok and latency_ms > 0:
                self._latency_history.append(latency_ms)
                self._consecutive_failures = 0
                self._last_heartbeat_time = check_time
                # Keep only last N samples
                if len(self._latency_history) > self._latency_window:
                    self._latency_history = self._latency_history[-self._latency_window :]
            else:
                self._consecutive_failures += 1
                self._total_failures += 1

            self._heartbeat_history.append(heartbeat_ok)
            if len(self._heartbeat_history) > self._latency_window:
                self._heartbeat_history = self._heartbeat_history[-self._latency_window :]

            # Build metrics snapshot
            metrics = HealthMetrics(
                timestamp_utc=check_time,
                latency_ms=latency_ms if heartbeat_ok else float("inf"),
                latency_samples=self._latency_history.copy(),
                packet_loss_rate=self._calculate_packet_loss(),
                heartbeat_success_rate=sum(self._heartbeat_history) / len(self._heartbeat_history)
                if self._heartbeat_history
                else 1.0,
                consecutive_failures=self._consecutive_failures,
                total_failures=self._total_failures,
                total_checks=self._total_checks,
                last_heartbeat_utc=self._last_heartbeat_time,
                failover_count=getattr(self._connector, "failover_count", 0),
            )

            old_status = (
                self._last_metrics.status
                if hasattr(self._last_metrics, "status")
                else HealthStatus.UNKNOWN
            )
            new_status = self._evaluate_status(metrics)
            metrics.status = new_status  # type: ignore

            self._last_metrics = metrics

        # Emit telemetry
        self._emit_health_telemetry(metrics, new_status)

        # Trigger callbacks if status changed
        if new_status != old_status:
            self._notify_status_change(old_status, new_status)
            logger.warning(
                "[HealthService] Status change: %s -> %s (latency=%.1fms, failures=%d)",
                old_status.name if isinstance(old_status, HealthStatus) else old_status,
                new_status.name,
                metrics.latency_ms,
                metrics.consecutive_failures,
            )

        # Check for degradation
        if new_status == HealthStatus.DEGRADED:
            self._notify_degradation(metrics)

        # Check for preemptive failover
        if self._should_failover(metrics):
            reason = f"latency={metrics.latency_ms:.1f}ms, failures={metrics.consecutive_failures}"
            self._notify_failover_recommended(reason)

    def _check_connection(self) -> bool:
        """Check if connector is connected."""
        try:
            return bool(self._connector.is_connected())
        except Exception as exc:
            logger.debug("[HealthService] Connection check failed: %s", exc)
            return False

    def _check_heartbeat(self) -> Tuple[bool, float]:
        """Send heartbeat and measure round-trip time.

        Returns (success, latency_ms)
        """
        start_time = time.perf_counter()
        try:
            # For HotStandby, heartbeat both connections
            if hasattr(self._connector, "send_heartbeat"):
                self._connector.send_heartbeat()
            else:
                # Direct connector
                from kinetra.connectors.ctrader_connector import CTraderConnector

                if isinstance(self._connector, CTraderConnector):
                    self._connector.send_heartbeat()

            # For accurate measurement, we'd need a ping-pong
            # For now, approximate with local timing
            latency_ms = (time.perf_counter() - start_time) * 1000
            return True, latency_ms

        except Exception as exc:
            logger.debug("[HealthService] Heartbeat failed: %s", exc)
            return False, float("inf")

    def _calculate_packet_loss(self) -> float:
        """Calculate packet loss rate from heartbeat history."""
        if not self._heartbeat_history:
            return 0.0
        failures = sum(1 for ok in self._heartbeat_history if not ok)
        return failures / len(self._heartbeat_history)

    def _evaluate_status(self, metrics: HealthMetrics) -> HealthStatus:
        """Evaluate overall health status from metrics."""
        # CRITICAL: Multiple consecutive failures or extreme latency
        if metrics.consecutive_failures >= self._consecutive_failures_threshold:
            return HealthStatus.CRITICAL

        if metrics.latency_ms > self._latency_critical_ms:
            return HealthStatus.CRITICAL

        if not self._connector.is_connected():
            return HealthStatus.CRITICAL

        # UNHEALTHY: High packet loss or poor success rate
        if metrics.packet_loss_rate > self._packet_loss_threshold * 2:
            return HealthStatus.UNHEALTHY

        if metrics.heartbeat_success_rate < 0.95:
            return HealthStatus.UNHEALTHY

        # DEGRADED: Elevated latency or minor packet loss
        if metrics.latency_ms > self._latency_threshold_ms:
            return HealthStatus.DEGRADED

        if metrics.packet_loss_rate > self._packet_loss_threshold:
            return HealthStatus.DEGRADED

        if metrics.latency_std_ms > 100:  # High jitter
            return HealthStatus.DEGRADED

        # HEALTHY: All metrics normal
        if metrics.total_checks >= 3:  # Need some history
            return HealthStatus.HEALTHY

        return HealthStatus.UNKNOWN

    def _should_failover(self, metrics: HealthMetrics) -> bool:
        """Determine if preemptive failover should be triggered."""
        if not self._enable_preemptive_failover:
            return False

        # Only failover if we have a hot standby
        if not hasattr(self._connector, "_atomic_failover"):
            return False

        # Failover on critical status
        if metrics.status == HealthStatus.CRITICAL:
            return True

        # Failover on sustained degradation
        if metrics.status == HealthStatus.DEGRADED and metrics.consecutive_failures >= 2:
            return True

        return False

    def _notify_status_change(self, old_status: HealthStatus, new_status: HealthStatus) -> None:
        """Notify status change callbacks."""
        for callback in self._on_status_change:
            try:
                callback(old_status, new_status)
            except Exception as exc:
                logger.error("[HealthService] Status callback error: %s", exc)

    def _notify_degradation(self, metrics: HealthMetrics) -> None:
        """Notify degradation callbacks."""
        for callback in self._on_degradation:
            try:
                callback(metrics)
            except Exception as exc:
                logger.error("[HealthService] Degradation callback error: %s", exc)

    def _notify_failover_recommended(self, reason: str) -> None:
        """Notify failover recommended callbacks."""
        for callback in self._on_failover_recommended:
            try:
                callback(reason)
            except Exception as exc:
                logger.error("[HealthService] Failover callback error: %s", exc)

    def _emit_health_telemetry(self, metrics: HealthMetrics, status: HealthStatus) -> None:
        """Emit health metrics to telemetry."""
        if self._telemetry is None:
            return

        try:
            self._telemetry.emit(
                stream="connection_health",
                component="connection_health_service",
                event_type="health_check",
                status=status.name.lower(),
                payload={
                    "latency_ms": round(metrics.latency_ms, 2),
                    "latency_p95_ms": round(metrics.latency_p95_ms, 2),
                    "latency_std_ms": round(metrics.latency_std_ms, 2),
                    "packet_loss_rate": round(metrics.packet_loss_rate, 4),
                    "heartbeat_success_rate": round(metrics.heartbeat_success_rate, 4),
                    "consecutive_failures": metrics.consecutive_failures,
                    "total_checks": metrics.total_checks,
                },
            )
        except Exception as exc:
            logger.debug("[HealthService] Telemetry emit failed: %s", exc)

    def get_health_status(self) -> HealthCheckResult:
        """Get current health status snapshot."""
        with self._lock:
            metrics = HealthMetrics(
                timestamp_utc=self._last_metrics.timestamp_utc,
                latency_ms=self._last_metrics.latency_ms,
                latency_samples=self._latency_history.copy(),
                packet_loss_rate=self._last_metrics.packet_loss_rate,
                heartbeat_success_rate=self._last_metrics.heartbeat_success_rate,
                consecutive_failures=self._consecutive_failures,
                total_failures=self._total_failures,
                total_checks=self._total_checks,
                last_heartbeat_utc=self._last_heartbeat_time,
                failover_count=getattr(self._connector, "failover_count", 0),
            )
            status = self._evaluate_status(metrics)
            metrics.status = status  # type: ignore

        # Build detailed checks
        checks = {
            "connected": (self._connector.is_connected(), "TCP connection"),
            "latency_ok": (
                metrics.latency_ms < self._latency_threshold_ms,
                f"latency {metrics.latency_ms:.1f}ms < {self._latency_threshold_ms:.1f}ms",
            ),
            "packet_loss_ok": (
                metrics.packet_loss_rate < self._packet_loss_threshold,
                f"loss {metrics.packet_loss_rate:.2%} < {self._packet_loss_threshold:.2%}",
            ),
            "heartbeat_ok": (
                metrics.heartbeat_success_rate > 0.95,
                f"success {metrics.heartbeat_success_rate:.1%}",
            ),
        }

        # Generate recommendations
        recommendations = []
        if status == HealthStatus.CRITICAL:
            recommendations.append("Immediate failover recommended")
            recommendations.append("Check network connectivity")
        elif status == HealthStatus.UNHEALTHY:
            recommendations.append("Consider failover")
            recommendations.append("Monitor packet loss")
        elif status == HealthStatus.DEGRADED:
            if metrics.latency_ms > self._latency_threshold_ms:
                recommendations.append(f"High latency detected ({metrics.latency_ms:.1f}ms)")
            if metrics.packet_loss_rate > self._packet_loss_threshold:
                recommendations.append(f"Packet loss detected ({metrics.packet_loss_rate:.2%})")

        return HealthCheckResult(
            status=status,
            metrics=metrics,
            checks=checks,
            recommendations=recommendations,
        )

    def force_health_check(self) -> HealthCheckResult:
        """Force immediate health check and return result."""
        self._perform_health_check()
        return self.get_health_status()


def format_health_report(result: HealthCheckResult) -> str:
    """Format health check result as human-readable string."""
    lines = [
        f"Health Status: {result.status.name}",
        f"Latency: {result.metrics.latency_ms:.1f}ms (p95: {result.metrics.latency_p95_ms:.1f}ms, std: {result.metrics.latency_std_ms:.1f}ms)",
        f"Packet Loss: {result.metrics.packet_loss_rate:.2%}",
        f"Heartbeat Success: {result.metrics.heartbeat_success_rate:.1%}",
        f"Consecutive Failures: {result.metrics.consecutive_failures}",
        f"Total Checks: {result.metrics.total_checks}",
        "",
        "Checks:",
    ]

    for name, (ok, detail) in result.checks.items():
        status = "✓" if ok else "✗"
        lines.append(f"  {status} {name}: {detail}")

    if result.recommendations:
        lines.extend(["", "Recommendations:"])
        for rec in result.recommendations:
            lines.append(f"  ! {rec}")

    return "\n".join(lines)
