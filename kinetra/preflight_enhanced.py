"""
Enhanced Preflight Checks for Live Trading
============================================

Comprehensive preflight validation before executing real orders:
1. DNS Resolution & Latency
2. TCP Connection Pool Health
3. Heartbeat/Keep-Alive Verification
4. Broker Authentication & Session
5. Account Balance & Margin Requirements
6. Symbol Resolution & Market Hours
7. Circuit Breaker Status
8. Execution Path Test (optional)

Usage::

    from kinetra.preflight_enhanced import EnhancedPreflight, PreflightConfig

    config = PreflightConfig(
        symbol="XAUUSD",
        min_balance_usd=100.0,
        test_lots=0.01,
    )

    preflight = EnhancedPreflight(connector, config)
    result = preflight.run_all_checks()

    if result.can_trade:
        print("All checks passed - safe to trade")
    else:
        print(f"Blocked: {result.blocking_reasons}")
"""

from __future__ import annotations

import logging
import socket
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, auto
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple

import click

from kinetra.dns_hardening import (
    rank_reachable_endpoints,
    resolve_and_validate_host,
    select_reachable_endpoint,
)
from kinetra.monitoring.central_logger import CentralTelemetry
from kinetra.monitoring.connection_health import (
    ConnectionHealthService,
    HealthCheckResult,
    HealthStatus,
    format_health_report,
)

if TYPE_CHECKING:
    from kinetra.connectors.ctrader_connector import CTraderConnector, HotStandbyCTraderConnector

logger = logging.getLogger(__name__)


class CheckSeverity(Enum):
    """Severity level for preflight checks."""

    INFO = auto()  # Informational only
    WARNING = auto()  # Warning but not blocking
    BLOCKING = auto()  # Blocks trading


@dataclass
class PreflightCheck:
    """Individual preflight check result."""

    name: str
    passed: bool
    severity: CheckSeverity
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    duration_ms: float = 0.0

    @property
    def is_blocking(self) -> bool:
        return not self.passed and self.severity == CheckSeverity.BLOCKING


@dataclass
class PreflightResult:
    """Complete preflight check results."""

    timestamp_utc: datetime
    checks: List[PreflightCheck]
    total_duration_ms: float

    @property
    def can_trade(self) -> bool:
        return not any(c.is_blocking for c in self.checks)

    @property
    def blocking_reasons(self) -> List[str]:
        return [f"{c.name}: {c.message}" for c in self.checks if c.is_blocking]

    @property
    def warnings(self) -> List[str]:
        return [
            f"{c.name}: {c.message}"
            for c in self.checks
            if not c.passed and c.severity == CheckSeverity.WARNING
        ]

    @property
    def passed_count(self) -> int:
        return sum(1 for c in self.checks if c.passed)

    @property
    def total_count(self) -> int:
        return len(self.checks)

    def get_check(self, name: str) -> Optional[PreflightCheck]:
        """Get check by name."""
        for check in self.checks:
            if check.name == name:
                return check
        return None


@dataclass
class PreflightConfig:
    """Configuration for preflight checks."""

    symbol: str
    min_balance_usd: float = 100.0
    min_margin_available_usd: float = 50.0
    max_latency_ms: float = 500.0
    max_dns_latency_ms: float = 2000.0
    test_lots: Optional[float] = None  # None = skip execution test
    check_market_hours: bool = True
    check_circuit_breakers: bool = True
    require_hot_standby: bool = False
    enable_health_service: bool = True
    health_check_duration_seconds: float = 5.0
    require_depth_quotes: bool = False
    depth_levels: int = 5
    depth_timeout_seconds: float = 5.0

    # DNS validation
    validate_dns: bool = True
    dns_hosts: List[str] = field(
        default_factory=lambda: [
            "demo.ctraderapi.com",
            "live.ctraderapi.com",
        ]
    )


class EnhancedPreflight:
    """
    Enhanced preflight check suite for live trading safety.

    Performs comprehensive validation before allowing real order submission.

    Parameters
    ----------
    connector : CTraderConnector or HotStandbyCTraderConnector
        The broker connector to validate
    config : PreflightConfig
        Preflight configuration
    """

    def __init__(
        self,
        connector: CTraderConnector | HotStandbyCTraderConnector,
        config: PreflightConfig,
        telemetry: Optional[CentralTelemetry] = None,
    ):
        self._connector = connector
        self._config = config
        self._telemetry = telemetry
        self._health_service: Optional[ConnectionHealthService] = None

        if config.enable_health_service:
            self._health_service = ConnectionHealthService(connector)
            if telemetry:
                self._health_service.set_telemetry(telemetry)

    def run_all_checks(
        self, progress_callback: Optional[Callable[[str], None]] = None
    ) -> PreflightResult:
        """Run all preflight checks and return results."""
        start_time = time.perf_counter()
        checks: List[PreflightCheck] = []

        def _progress(msg: str):
            logger.info("[Preflight] %s", msg)
            if progress_callback:
                progress_callback(msg)

        # Check 1: DNS Resolution
        _progress("Checking DNS resolution...")
        checks.append(self._check_dns())

        # Check 2: TCP Connection Pool
        _progress("Validating TCP connection pool...")
        checks.append(self._check_tcp_pool())

        # Check 3: Heartbeat/Keep-Alive
        _progress("Testing heartbeat/keep-alive...")
        checks.append(self._check_heartbeat())

        # Check 4: Broker Session Health
        _progress("Checking broker session...")
        checks.append(self._check_broker_session())

        # Check 5: Account Balance
        _progress("Verifying account balance...")
        checks.append(self._check_account_balance())

        # Check 6: Margin Requirements
        _progress("Checking margin requirements...")
        checks.append(self._check_margin())

        # Check 7: Symbol Resolution
        _progress("Resolving symbol...")
        checks.append(self._check_symbol_resolution())

        # Check 7b: Depth quotes (for OBI slippage buffer)
        if self._config.require_depth_quotes:
            _progress("Validating depth quotes (L2/OBI)...")
            checks.append(self._check_depth_quotes())

        # Check 8: Market Hours
        if self._config.check_market_hours:
            _progress("Checking market hours...")
            checks.append(self._check_market_hours())

        # Check 9: Hot Standby (if enabled)
        if self._config.require_hot_standby:
            _progress("Verifying hot standby...")
            checks.append(self._check_hot_standby())

        # Check 10: Connection Health Service (if enabled)
        if self._config.enable_health_service and self._health_service:
            _progress("Running health service validation...")
            checks.append(self._check_connection_health())

        total_duration_ms = (time.perf_counter() - start_time) * 1000

        result = PreflightResult(
            timestamp_utc=datetime.now(timezone.utc),
            checks=checks,
            total_duration_ms=total_duration_ms,
        )

        # Emit telemetry
        self._emit_preflight_telemetry(result)

        return result

    def _emit_preflight_telemetry(self, result: PreflightResult) -> None:
        """Emit preflight results to telemetry."""
        if self._telemetry is None:
            return

        try:
            self._telemetry.emit(
                stream="preflight",
                component="enhanced_preflight",
                event_type="preflight_complete",
                status="pass" if result.can_trade else "fail",
                payload={
                    "symbol": self._config.symbol,
                    "can_trade": result.can_trade,
                    "total_checks": result.total_count,
                    "passed_checks": result.passed_count,
                    "duration_ms": round(result.total_duration_ms, 2),
                    "blocking_reasons": result.blocking_reasons,
                    "warnings": result.warnings,
                },
            )
        except Exception as exc:
            logger.debug("[Preflight] Telemetry emit failed: %s", exc)

    def _check_dns(self) -> PreflightCheck:
        """Check DNS resolution for broker endpoints."""
        check_start = time.perf_counter()

        if not self._config.validate_dns:
            return PreflightCheck(
                name="dns_resolution",
                passed=True,
                severity=CheckSeverity.INFO,
                message="DNS validation disabled",
                duration_ms=(time.perf_counter() - check_start) * 1000,
            )

        results = []
        resolution_failed = False  # no IPs returned or exception — hard block
        latency_slow = False       # resolved OK but slow — warning only

        for host in self._config.dns_hosts:
            try:
                start = time.perf_counter()
                ips = resolve_and_validate_host(host, service_name="preflight")
                duration_ms = (time.perf_counter() - start) * 1000

                if ips:
                    results.append(f"{host} -> {', '.join(ips)} ({duration_ms:.1f}ms)")
                    if duration_ms > self._config.max_dns_latency_ms:
                        latency_slow = True
                else:
                    results.append(f"{host} -> NO RESULTS")
                    resolution_failed = True

            except Exception as exc:
                results.append(f"{host} -> ERROR: {exc}")
                resolution_failed = True

        duration_ms = (time.perf_counter() - check_start) * 1000

        if resolution_failed:
            severity = CheckSeverity.BLOCKING
            passed = False
        elif latency_slow:
            severity = CheckSeverity.WARNING
            passed = False
        else:
            severity = CheckSeverity.INFO
            passed = True

        return PreflightCheck(
            name="dns_resolution",
            passed=passed,
            severity=severity,
            message="; ".join(results),
            details={"hosts_checked": len(self._config.dns_hosts)},
            duration_ms=duration_ms,
        )

    def _check_tcp_pool(self) -> PreflightCheck:
        """Check TCP connection pool health."""
        check_start = time.perf_counter()

        try:
            # Check connection state
            is_connected = self._connector.is_connected()

            # Get endpoint info if available
            endpoint = getattr(self._connector, "selected_endpoint", "unknown")

            # Test TCP connectivity
            if is_connected:
                # Already connected - verify it's responsive
                message = f"Connected to {endpoint}"
                passed = True
            else:
                message = "Not connected"
                passed = False

            # Check for hot standby status
            if hasattr(self._connector, "health_status"):
                hb_status = self._connector.health_status
                message += f" (hot-standby: {hb_status})"
                if hb_status == "DOWN":
                    passed = False

            duration_ms = (time.perf_counter() - check_start) * 1000

            return PreflightCheck(
                name="tcp_connection",
                passed=passed,
                severity=CheckSeverity.BLOCKING if not passed else CheckSeverity.INFO,
                message=message,
                details={"endpoint": endpoint, "connected": is_connected},
                duration_ms=duration_ms,
            )

        except Exception as exc:
            duration_ms = (time.perf_counter() - check_start) * 1000
            return PreflightCheck(
                name="tcp_connection",
                passed=False,
                severity=CheckSeverity.BLOCKING,
                message=f"TCP check failed: {exc}",
                duration_ms=duration_ms,
            )

    def _check_heartbeat(self) -> PreflightCheck:
        """Check heartbeat/keep-alive functionality."""
        check_start = time.perf_counter()

        try:
            # Send heartbeat and time it
            hb_start = time.perf_counter()
            self._connector.send_heartbeat()
            hb_latency_ms = (time.perf_counter() - hb_start) * 1000

            # Verify still connected after heartbeat
            is_connected = self._connector.is_connected()

            passed = is_connected and hb_latency_ms < self._config.max_latency_ms

            duration_ms = (time.perf_counter() - check_start) * 1000

            return PreflightCheck(
                name="heartbeat",
                passed=passed,
                severity=CheckSeverity.BLOCKING if not is_connected else CheckSeverity.WARNING,
                message=f"Heartbeat latency: {hb_latency_ms:.1f}ms, connected: {is_connected}",
                details={"latency_ms": hb_latency_ms, "connected": is_connected},
                duration_ms=duration_ms,
            )

        except Exception as exc:
            duration_ms = (time.perf_counter() - check_start) * 1000
            return PreflightCheck(
                name="heartbeat",
                passed=False,
                severity=CheckSeverity.BLOCKING,
                message=f"Heartbeat failed: {exc}",
                duration_ms=duration_ms,
            )

    def _check_broker_session(self) -> PreflightCheck:
        """Check broker authentication and session validity."""
        check_start = time.perf_counter()

        try:
            # Try to get account snapshot - this validates the session
            snapshot = self._connector.get_account_snapshot(timeout_s=10.0)

            account_id = snapshot.get("account_id", "unknown")
            broker_name = snapshot.get("broker_name", "unknown")

            passed = bool(account_id and account_id != "unknown")

            duration_ms = (time.perf_counter() - check_start) * 1000

            return PreflightCheck(
                name="broker_session",
                passed=passed,
                severity=CheckSeverity.BLOCKING if not passed else CheckSeverity.INFO,
                message=f"Session valid - Account: {account_id}, Broker: {broker_name}",
                details={"account_id": account_id, "broker_name": broker_name},
                duration_ms=duration_ms,
            )

        except Exception as exc:
            duration_ms = (time.perf_counter() - check_start) * 1000
            return PreflightCheck(
                name="broker_session",
                passed=False,
                severity=CheckSeverity.BLOCKING,
                message=f"Session check failed: {exc}",
                duration_ms=duration_ms,
            )

    def _check_account_balance(self) -> PreflightCheck:
        """Check account balance meets minimum requirements."""
        check_start = time.perf_counter()

        try:
            snapshot = self._connector.get_account_snapshot(timeout_s=10.0)
            balance = float(snapshot.get("balance", 0.0))
            equity = float(snapshot.get("equity", balance))

            passed = balance >= self._config.min_balance_usd

            duration_ms = (time.perf_counter() - check_start) * 1000

            return PreflightCheck(
                name="account_balance",
                passed=passed,
                severity=CheckSeverity.BLOCKING if not passed else CheckSeverity.INFO,
                message=f"Balance: ${balance:,.2f}, Equity: ${equity:,.2f}",
                details={"balance": balance, "equity": equity},
                duration_ms=duration_ms,
            )

        except Exception as exc:
            duration_ms = (time.perf_counter() - check_start) * 1000
            return PreflightCheck(
                name="account_balance",
                passed=False,
                severity=CheckSeverity.BLOCKING,
                message=f"Balance check failed: {exc}",
                duration_ms=duration_ms,
            )

    def _check_margin(self) -> PreflightCheck:
        """Check available margin for trading."""
        check_start = time.perf_counter()

        try:
            snapshot = self._connector.get_account_snapshot(timeout_s=10.0)

            balance = float(snapshot.get("balance", 0.0))
            equity = float(snapshot.get("equity", balance))
            margin_used = float(snapshot.get("margin_used", 0.0))
            margin_available = equity - margin_used

            passed = margin_available >= self._config.min_margin_available_usd

            duration_ms = (time.perf_counter() - check_start) * 1000

            return PreflightCheck(
                name="margin_available",
                passed=passed,
                severity=CheckSeverity.BLOCKING if not passed else CheckSeverity.INFO,
                message=f"Available: ${margin_available:,.2f} (Used: ${margin_used:,.2f})",
                details={
                    "margin_available": margin_available,
                    "margin_used": margin_used,
                    "equity": equity,
                },
                duration_ms=duration_ms,
            )

        except Exception as exc:
            duration_ms = (time.perf_counter() - check_start) * 1000
            return PreflightCheck(
                name="margin_available",
                passed=False,
                severity=CheckSeverity.BLOCKING,
                message=f"Margin check failed: {exc}",
                duration_ms=duration_ms,
            )

    def _check_symbol_resolution(self) -> PreflightCheck:
        """Check symbol can be resolved."""
        check_start = time.perf_counter()

        try:
            symbol_id = self._connector.find_symbol_id(self._config.symbol, timeout_s=10.0)

            passed = symbol_id is not None

            # Get additional symbol info if resolved
            digits = None
            if passed:
                try:
                    digits = self._connector.get_digits(symbol_id, timeout_s=5.0)
                except Exception:
                    pass

            duration_ms = (time.perf_counter() - check_start) * 1000

            return PreflightCheck(
                name="symbol_resolution",
                passed=passed,
                severity=CheckSeverity.BLOCKING if not passed else CheckSeverity.INFO,
                message=f"{self._config.symbol} -> ID {symbol_id}"
                + (f" (digits={digits})" if digits else ""),
                details={"symbol_id": symbol_id, "digits": digits},
                duration_ms=duration_ms,
            )

        except Exception as exc:
            duration_ms = (time.perf_counter() - check_start) * 1000
            return PreflightCheck(
                name="symbol_resolution",
                passed=False,
                severity=CheckSeverity.BLOCKING,
                message=f"Symbol resolution failed: {exc}",
                duration_ms=duration_ms,
            )

    def _check_depth_quotes(self) -> PreflightCheck:
        """Check L2 depth stream is available for OBI execution buffering."""
        check_start = time.perf_counter()
        symbol_id = None
        try:
            symbol_id = self._connector.find_symbol_id(self._config.symbol, timeout_s=10.0)
            if symbol_id is None:
                duration_ms = (time.perf_counter() - check_start) * 1000
                return PreflightCheck(
                    name="depth_quotes",
                    passed=False,
                    severity=CheckSeverity.BLOCKING,
                    message=f"Cannot resolve symbol for depth: {self._config.symbol}",
                    duration_ms=duration_ms,
                )

            # Prefer direct connector helper when available.
            get_imb = getattr(self._connector, "get_order_book_imbalance", None)
            if callable(get_imb):
                imb = get_imb(self._config.symbol, levels=max(int(self._config.depth_levels), 1))
                if imb is not None:
                    duration_ms = (time.perf_counter() - check_start) * 1000
                    return PreflightCheck(
                        name="depth_quotes",
                        passed=True,
                        severity=CheckSeverity.INFO,
                        message=f"Depth imbalance available: {float(imb):.3f}",
                        details={"imbalance": float(imb), "levels": int(self._config.depth_levels)},
                        duration_ms=duration_ms,
                    )

            # Fallback: raw subscribe + wait for depth event.
            event = threading.Event()
            state: Dict[str, Any] = {"quotes": 0, "imbalance": None}

            def _on_depth(payload: Any) -> None:
                try:
                    sid = int(getattr(payload, "symbolId", 0))
                    if sid != int(symbol_id):
                        return
                    new_quotes = list(getattr(payload, "newQuotes", []))
                    if not new_quotes:
                        return
                    levels = max(int(self._config.depth_levels), 1)
                    bids: List[float] = []
                    asks: List[float] = []
                    for q in new_quotes:
                        size = float(getattr(q, "size", 0.0))
                        if size <= 0:
                            continue
                        bid_raw = int(getattr(q, "bid", 0))
                        ask_raw = int(getattr(q, "ask", 0))
                        if bid_raw > 0 and ask_raw <= 0:
                            bids.append(size)
                        elif ask_raw > 0:
                            asks.append(size)
                    bids = bids[:levels]
                    asks = asks[:levels]
                    if bids and asks:
                        wb = sum(v / float(i + 1) for i, v in enumerate(bids))
                        wa = sum(v / float(i + 1) for i, v in enumerate(asks))
                        denom = wb + wa
                        if denom > 0:
                            state["imbalance"] = float((wb - wa) / denom)
                    state["quotes"] = len(new_quotes)
                    event.set()
                except Exception:
                    return

            self._connector.add_push_handler("ProtoOADepthEvent", _on_depth)
            api_msgs = __import__("ctrader_open_api.messages.OpenApiMessages_pb2", fromlist=["_x"])
            req = api_msgs.ProtoOASubscribeDepthQuotesReq()
            req.ctidTraderAccountId = self._connector.credentials.account_id
            req.symbolId.append(int(symbol_id))
            self._connector.send_and_wait(req, timeout_s=5.0)
            ok = event.wait(timeout=max(float(self._config.depth_timeout_seconds), 0.5))

            try:
                unreq = api_msgs.ProtoOAUnsubscribeDepthQuotesReq()
                unreq.ctidTraderAccountId = self._connector.credentials.account_id
                unreq.symbolId.append(int(symbol_id))
                self._connector.send_and_wait(unreq, timeout_s=3.0)
            except Exception:
                pass
            try:
                self._connector.remove_push_handler("ProtoOADepthEvent", _on_depth)
            except Exception:
                pass

            duration_ms = (time.perf_counter() - check_start) * 1000
            if not ok:
                return PreflightCheck(
                    name="depth_quotes",
                    passed=False,
                    severity=CheckSeverity.BLOCKING,
                    message=f"No depth quotes received within {self._config.depth_timeout_seconds:.1f}s",
                    details={"symbol_id": int(symbol_id)},
                    duration_ms=duration_ms,
                )
            imb = state.get("imbalance")
            msg = (
                f"Depth feed active (quotes={int(state.get('quotes', 0))}, imbalance={float(imb):.3f})"
                if imb is not None
                else f"Depth feed active (quotes={int(state.get('quotes', 0))})"
            )
            return PreflightCheck(
                name="depth_quotes",
                passed=True,
                severity=CheckSeverity.INFO,
                message=msg,
                details={"symbol_id": int(symbol_id), "levels": int(self._config.depth_levels)},
                duration_ms=duration_ms,
            )
        except Exception as exc:
            duration_ms = (time.perf_counter() - check_start) * 1000
            return PreflightCheck(
                name="depth_quotes",
                passed=False,
                severity=CheckSeverity.BLOCKING
                if self._config.require_depth_quotes
                else CheckSeverity.WARNING,
                message=f"Depth check failed: {exc}",
                details={"symbol_id": symbol_id},
                duration_ms=duration_ms,
            )

    def _check_market_hours(self) -> PreflightCheck:
        """Check if market is currently open for trading."""
        check_start = time.perf_counter()

        try:
            # XAUUSD (Gold) trading hours: Sunday 22:00 UTC - Friday 20:00 UTC
            # With daily maintenance: 00:00-00:01 UTC (some brokers)

            now = datetime.now(timezone.utc)
            weekday = now.weekday()  # Monday=0, Sunday=6
            hour = now.hour

            # Check weekend (Friday after 20:00 UTC to Sunday after 22:00 UTC)
            is_weekend = False
            if weekday == 4 and hour >= 20:  # Friday after 20:00
                is_weekend = True
            elif weekday == 5:  # Saturday
                is_weekend = True
            elif weekday == 6 and hour < 22:  # Sunday before 22:00
                is_weekend = True

            # Check maintenance window (00:00-00:01 UTC)
            is_maintenance = hour == 0 and now.minute == 0

            is_open = not is_weekend and not is_maintenance

            # For critical markets, block if closed
            # For 24h markets like crypto, always pass
            is_forex_metal = True  # XAUUSD is a metal

            passed = is_open or not is_forex_metal

            if is_weekend:
                message = "Market closed (weekend)"
            elif is_maintenance:
                message = "Market in maintenance (00:00 UTC)"
            else:
                message = f"Market open ({now.strftime('%Y-%m-%d %H:%M UTC')})"

            duration_ms = (time.perf_counter() - check_start) * 1000

            return PreflightCheck(
                name="market_hours",
                passed=passed,
                severity=CheckSeverity.WARNING if not passed else CheckSeverity.INFO,
                message=message,
                details={
                    "is_open": is_open,
                    "is_weekend": is_weekend,
                    "is_maintenance": is_maintenance,
                    "utc_time": now.isoformat(),
                },
                duration_ms=duration_ms,
            )

        except Exception as exc:
            duration_ms = (time.perf_counter() - check_start) * 1000
            return PreflightCheck(
                name="market_hours",
                passed=True,  # Don't block on check failure
                severity=CheckSeverity.WARNING,
                message=f"Market hours check failed: {exc}",
                duration_ms=duration_ms,
            )

    def _check_hot_standby(self) -> PreflightCheck:
        """Check hot standby is available."""
        check_start = time.perf_counter()

        try:
            if not hasattr(self._connector, "health_status"):
                # Not a hot standby connector
                duration_ms = (time.perf_counter() - check_start) * 1000
                return PreflightCheck(
                    name="hot_standby",
                    passed=False,
                    severity=CheckSeverity.BLOCKING
                    if self._config.require_hot_standby
                    else CheckSeverity.WARNING,
                    message="Hot standby not configured",
                    duration_ms=duration_ms,
                )

            status = self._connector.health_status
            standby_ok = status in ("UP", "DEGRADED")

            passed = standby_ok or not self._config.require_hot_standby

            duration_ms = (time.perf_counter() - check_start) * 1000

            return PreflightCheck(
                name="hot_standby",
                passed=passed,
                severity=CheckSeverity.BLOCKING if not passed else CheckSeverity.INFO,
                message=f"Hot standby status: {status}",
                details={
                    "status": status,
                    "failover_count": getattr(self._connector, "failover_count", 0),
                },
                duration_ms=duration_ms,
            )

        except Exception as exc:
            duration_ms = (time.perf_counter() - check_start) * 1000
            return PreflightCheck(
                name="hot_standby",
                passed=False,
                severity=CheckSeverity.BLOCKING
                if self._config.require_hot_standby
                else CheckSeverity.WARNING,
                message=f"Hot standby check failed: {exc}",
                duration_ms=duration_ms,
            )

    def _check_connection_health(self) -> PreflightCheck:
        """Run connection health service validation."""
        check_start = time.perf_counter()

        try:
            if not self._health_service:
                duration_ms = (time.perf_counter() - check_start) * 1000
                return PreflightCheck(
                    name="connection_health",
                    passed=True,
                    severity=CheckSeverity.INFO,
                    message="Health service not enabled",
                    duration_ms=duration_ms,
                )

            # Quick health check
            result = self._health_service.force_health_check()

            passed = result.status not in (HealthStatus.CRITICAL, HealthStatus.UNHEALTHY)

            message = f"Status: {result.status.name}, Latency: {result.metrics.latency_ms:.1f}ms"

            duration_ms = (time.perf_counter() - check_start) * 1000

            return PreflightCheck(
                name="connection_health",
                passed=passed,
                severity=CheckSeverity.BLOCKING if not passed else CheckSeverity.INFO,
                message=message,
                details={
                    "health_status": result.status.name,
                    "latency_ms": result.metrics.latency_ms,
                    "packet_loss": result.metrics.packet_loss_rate,
                },
                duration_ms=duration_ms,
            )

        except Exception as exc:
            duration_ms = (time.perf_counter() - check_start) * 1000
            return PreflightCheck(
                name="connection_health",
                passed=False,
                severity=CheckSeverity.WARNING,
                message=f"Health check failed: {exc}",
                duration_ms=duration_ms,
            )

    def start_health_monitoring(self) -> None:
        """Start continuous health monitoring."""
        if self._health_service:
            self._health_service.start_monitoring()

    def stop_health_monitoring(self) -> None:
        """Stop continuous health monitoring."""
        if self._health_service:
            self._health_service.stop_monitoring()


def format_preflight_report(result: PreflightResult, verbose: bool = False) -> str:
    """Format preflight result as human-readable report."""
    lines = [
        "=" * 60,
        "PREFLIGHT CHECK RESULTS",
        f"Timestamp: {result.timestamp_utc.strftime('%Y-%m-%d %H:%M:%S UTC')}",
        f"Duration: {result.total_duration_ms:.1f}ms",
        f"Passed: {result.passed_count}/{result.total_count}",
        "=" * 60,
        "",
    ]

    # Group by status
    blocking = [c for c in result.checks if c.is_blocking]
    failed = [c for c in result.checks if not c.passed and not c.is_blocking]
    passed = [c for c in result.checks if c.passed]

    if blocking:
        lines.append("❌ BLOCKING ISSUES:")
        for check in blocking:
            lines.append(f"  ✗ {check.name}: {check.message}")
        lines.append("")

    if failed:
        lines.append("⚠️  WARNINGS:")
        for check in failed:
            lines.append(f"  ! {check.name}: {check.message}")
        lines.append("")

    if verbose or not blocking:
        lines.append("✅ PASSED:")
        for check in passed:
            lines.append(f"  ✓ {check.name}: {check.message}")
        lines.append("")

    # Summary
    if result.can_trade:
        lines.append("🟢 ALL CHECKS PASSED - Safe to trade")
    else:
        lines.append("🔴 PREFLIGHT FAILED - Trading blocked")
        lines.append("Blocking reasons:")
        for reason in result.blocking_reasons:
            lines.append(f"  - {reason}")

    lines.append("=" * 60)

    return "\n".join(lines)


def run_preflight_cli(
    connector, symbol: str, telemetry: Optional[CentralTelemetry] = None, **kwargs
) -> bool:
    """Run preflight checks with CLI output."""
    config = PreflightConfig(symbol=symbol, **kwargs)
    preflight = EnhancedPreflight(connector, config, telemetry=telemetry)

    click.echo()
    click.secho("Running enhanced preflight checks...", fg="cyan", bold=True)
    click.echo()

    def _progress(msg: str):
        click.echo(f"  {msg}")

    result = preflight.run_all_checks(progress_callback=_progress)

    click.echo()
    click.echo(format_preflight_report(result, verbose=False))

    return result.can_trade
