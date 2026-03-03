"""
cTrader Open API Connector — Shared Transport Layer
=====================================================

Canonical reusable connector for the cTrader Open API Protobuf interface.

This module extracts the Twisted-reactor-in-a-thread pattern that was
previously duplicated across two scripts:

    scripts/ctrader/fetch_balance.py        — CTraderClient
    scripts/ctrader/download_ctrader_history.py — CTraderDownloadClient

Both scripts had identical implementations of:
    - Twisted reactor lifecycle (start / stop / _run_reactor)
    - Request/response routing (_send_and_wait / _do_send / _on_response)
    - App authentication (_on_connected / _on_app_auth_*)
    - Account authentication (_do_account_auth)
    - Heartbeat handling (_on_message)
    - Symbol resolution (resolve_symbols / find_symbol_id)
    - Symbol digit caching (_fetch_symbol_digits / get_digits)

This module consolidates all of that into :class:`CTraderConnector`.
The scripts become thin wrappers that call the canonical connector.

Architecture (§28 Multi-Broker Phase 3)
----------------------------------------
- **Broker-aware boundary**: This module is the ONLY place in the
  ``kinetra/`` library that imports ``ctrader_open_api`` and ``twisted``.
  All downstream code (backtesting, RL, live trader) is broker-blind.
- **DRY**: Scripts import from here instead of reimplementing transport.
- **Reuse**: Both :class:`~kinetra.renko.ctrader_dispatcher.CTraderBarProvider`
  (live M1 bar feed + spot subscription) and the historical downloader use
  the same connector instance with the same authenticated session.

Hard rules (§28 AGENT_RULES_MASTER.md)
----------------------------------------
- ❌ Never import ``ctrader_open_api`` or ``twisted`` outside this file
  within the ``kinetra/`` library.
- ❌ Never duplicate _send_and_wait / reactor lifecycle in scripts —
  import :class:`CTraderConnector` instead.
- ✅ Always guard the import with ``try/except ImportError``.
- ✅ Always call ``stop()`` in a ``finally`` block.
- ✅ Always pass ``install_signal_handlers=False`` to ``reactor.run()``
  when running in a daemon thread.

Usage::

    from kinetra.connectors.ctrader_connector import CTraderConnector, CTraderCredentials

    creds = CTraderCredentials.from_env()
    conn = CTraderConnector(creds)
    ok = conn.start(timeout_s=30.0)          # connect + app auth + account auth
    assert ok, "Connection failed"

    # Resolve symbol name → cTrader symbolId
    symbol_id = conn.find_symbol_id("XAUUSD")

    # Send any Protobuf request and wait for response
    req = api_msgs.ProtoOAGetTrendbarsReq()
    ...
    resp = conn.send_and_wait(req, timeout_s=15.0)

    # Register a callback for unsolicited push messages (SpotEvent, ExecutionEvent, …)
    conn.add_push_handler("ProtoOASpotEvent", my_callback)

    conn.stop()

See Also
--------
- ``scripts/ctrader/download_ctrader_history.py`` — historical downloader (uses this)
- ``scripts/ctrader/fetch_balance.py``            — balance fetcher (uses this)
- ``kinetra/renko/ctrader_dispatcher.py``         — live OrderDispatcher + BarProvider
- ``AGENT_RULES_MASTER.md §28``                   — multi-broker architecture rules
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from kinetra.dns_hardening import (
    expand_endpoint_candidates,
    rank_reachable_endpoints,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional SDK import — guard so that broker-blind modules stay importable
# ---------------------------------------------------------------------------

try:
    from ctrader_open_api import Client, EndPoints, Protobuf, TcpProtocol
    from ctrader_open_api.messages import OpenApiCommonMessages_pb2 as _common_msgs
    from ctrader_open_api.messages import OpenApiMessages_pb2 as _api_msgs
    from twisted.internet import reactor as _reactor

    _CTRADER_AVAILABLE = True
except ImportError:
    _CTRADER_AVAILABLE = False
    Client = None  # type: ignore[assignment,misc]
    EndPoints = None  # type: ignore[assignment,misc]
    Protobuf = None  # type: ignore[assignment,misc]
    TcpProtocol = None  # type: ignore[assignment,misc]
    _common_msgs = None  # type: ignore[assignment]
    _api_msgs = None  # type: ignore[assignment]
    _reactor = None  # type: ignore[assignment]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: cTrader TrendbarPeriod value for M1.
CTRADER_M1_PERIOD: int = 1

#: Default response timeout (seconds) for individual Protobuf requests.
DEFAULT_REQUEST_TIMEOUT_S: float = 15.0

#: Default connect + auth timeout (seconds).
DEFAULT_CONNECT_TIMEOUT_S: float = 30.0

#: Inter-request delay (seconds) to stay within the SDK's 5 msg/s cap.
DEFAULT_INTER_REQUEST_SLEEP_S: float = 0.25

#: Symbol aliases: canonical Kinetra name → cTrader/Pepperstone broker name.
#: Discovered empirically from Pepperstone cTrader demo account
#: (ctid 45841299, login 5179095, folder slug: pepperstone_demo_5179095).
DEFAULT_SYMBOL_ALIASES: Dict[str, str] = {
    "JP225": "JPN225",
    "USOIL": "SpotCrude",
    "UKOIL": "SpotBrent",
    "NGAS": "NatGas",
}


def _connector_env_value(name: str) -> Optional[str]:
    """
    Resolve connector runtime settings from process env or env files.

    Resolution order:
    1. ``os.environ``
    2. ``<project_root>/.env.openapi``
    3. ``<project_root>/.env``
    """
    if name in os.environ:
        return os.environ.get(name)

    project_root = Path(__file__).resolve().parent.parent.parent
    merged: Dict[str, str] = {}
    for p in [project_root / ".env", project_root / ".env.openapi"]:
        if p.exists():
            merged.update(_parse_env_file(p))
    return merged.get(name)


def _connector_env_list(name: str) -> List[str]:
    raw = _connector_env_value(name) or ""
    return [part.strip() for part in raw.split(",") if part.strip()]


def _connector_env_bool(name: str, default: bool) -> bool:
    raw = _connector_env_value(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _last_endpoint_cache_path(environment: str) -> Path:
    project_root = Path(__file__).resolve().parent.parent.parent
    cache_dir = project_root / ".cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    env = (environment or "demo").strip().lower()
    return cache_dir / f"ctrader_last_endpoint_{env}.txt"


def _read_last_endpoint(environment: str) -> Optional[str]:
    p = _last_endpoint_cache_path(environment)
    try:
        if not p.exists():
            return None
        val = p.read_text(encoding="utf-8").strip()
        return val or None
    except Exception:
        return None


def _write_last_endpoint(environment: str, endpoint: str) -> None:
    if not endpoint:
        return
    p = _last_endpoint_cache_path(environment)
    try:
        p.write_text(endpoint.strip(), encoding="utf-8")
    except Exception:
        logger.debug("[cTrader] Could not persist last endpoint cache", exc_info=True)


def _endpoint_pool_cache_path(environment: str) -> Path:
    project_root = Path(__file__).resolve().parent.parent.parent
    cache_dir = project_root / ".cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    env = (environment or "demo").strip().lower()
    return cache_dir / f"ctrader_endpoint_pool_{env}.json"


def _read_endpoint_pool(environment: str) -> List[str]:
    p = _endpoint_pool_cache_path(environment)
    try:
        if not p.exists():
            return []
        payload = json.loads(p.read_text(encoding="utf-8"))
        rows = payload.get("endpoints", [])
        return [str(x).strip() for x in rows if str(x).strip()]
    except Exception:
        return []


def _hot_standby_state_path(environment: str) -> Path:
    project_root = Path(__file__).resolve().parent.parent.parent
    state_dir = project_root / "outputs" / "state"
    state_dir.mkdir(parents=True, exist_ok=True)
    env = (environment or "demo").strip().lower()
    return state_dir / f"ctrader_hot_standby_state_{env}.json"


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def _write_endpoint_pool(environment: str, endpoints: List[str]) -> None:
    p = _endpoint_pool_cache_path(environment)
    try:
        payload = {
            "updated_at_utc": int(time.time()),
            "endpoints": [e for e in endpoints if e],
        }
        p.write_text(json.dumps(payload), encoding="utf-8")
    except Exception:
        logger.debug("[cTrader] Could not persist endpoint pool cache", exc_info=True)


# ---------------------------------------------------------------------------
# Credentials dataclass
# ---------------------------------------------------------------------------


@dataclass
class CTraderCredentials:
    """cTrader Open API OAuth2 credentials.

    Parameters
    ----------
    client_id:
        OAuth2 application client ID (from https://openapi.ctrader.com/).
    client_secret:
        OAuth2 application client secret.
    access_token:
        OAuth2 access token (obtained via the ``--auth`` flow in
        ``scripts/ctrader/fetch_balance.py``).
    account_id:
        cTrader ctidTraderAccountId (integer).
    environment:
        ``"demo"`` or ``"live"``.  Selects the correct cTrader endpoint.
    """

    client_id: str
    client_secret: str
    access_token: str
    account_id: int
    environment: str = "demo"

    @classmethod
    def from_env(
        cls,
        env_file: Optional[str] = None,
        openapi_env_file: Optional[str] = None,
    ) -> "CTraderCredentials":
        """Load credentials from environment variables / ``.env`` files.

        Resolution order (later sources override earlier ones for the same key):
        1. ``<project_root>/.env`` — general project credentials.
        2. ``<project_root>/.env.openapi`` — cTrader-specific overrides.
        3. The explicit *env_file* path (if supplied).
        4. The explicit *openapi_env_file* path (if supplied).
        5. Process environment variables (``os.environ``).

        Parameters
        ----------
        env_file:
            Optional path to an additional ``.env``-style file to load.
        openapi_env_file:
            Optional path to a cTrader-specific ``.env.openapi`` file.

        Returns
        -------
        CTraderCredentials

        Raises
        ------
        ValueError
            If any required credential is missing after all sources are
            exhausted.
        """
        # Attempt dotenv loading (optional dependency)
        _project_root = Path(__file__).resolve().parent.parent.parent
        _loaded: Dict[str, str] = {}

        for path in [
            _project_root / ".env",
            _project_root / ".env.openapi",
            Path(env_file) if env_file else None,
            Path(openapi_env_file) if openapi_env_file else None,
        ]:
            if path is not None and path.exists():
                _loaded.update(_parse_env_file(path))

        def _get(key: str) -> str:
            # env vars take highest precedence over file values
            return os.environ.get(key, _loaded.get(key, "")).strip()

        client_id = _get("CTRADER_CLIENT_ID")
        client_secret = _get("CTRADER_CLIENT_SECRET")
        access_token = _get("CTRADER_ACCESS_TOKEN")
        raw_account = _get("CTRADER_ACCOUNT_ID")
        environment = _get("CTRADER_ENVIRONMENT") or "demo"

        missing = [
            name
            for name, val in [
                ("CTRADER_CLIENT_ID", client_id),
                ("CTRADER_CLIENT_SECRET", client_secret),
                ("CTRADER_ACCESS_TOKEN", access_token),
                ("CTRADER_ACCOUNT_ID", raw_account),
            ]
            if not val
        ]
        if missing:
            raise ValueError(
                f"Missing cTrader credentials: {', '.join(missing)}. "
                "Run 'python scripts/ctrader/fetch_balance.py --auth' to complete OAuth2 "
                "authorization and set these in .env.openapi."
            )

        try:
            account_id = int(raw_account)
        except ValueError:
            raise ValueError(f"CTRADER_ACCOUNT_ID must be an integer, got: {raw_account!r}")

        return cls(
            client_id=client_id,
            client_secret=client_secret,
            access_token=access_token,
            account_id=account_id,
            environment=environment,
        )


# ---------------------------------------------------------------------------
# Push-handler registry
# ---------------------------------------------------------------------------


@dataclass
class _PushHandlerRegistry:
    """Maps Protobuf message names to lists of registered callbacks.

    The connector calls all registered handlers when an unsolicited push
    message arrives (SpotEvent, ExecutionEvent, etc.).  Handlers are called
    on the Twisted reactor thread — they must not block.
    """

    _handlers: Dict[str, List[Callable[[Any], None]]] = field(default_factory=dict)

    def add(self, message_name: str, handler: Callable[[Any], None]) -> None:
        """Register *handler* for messages named *message_name*."""
        self._handlers.setdefault(message_name, []).append(handler)

    def remove(self, message_name: str, handler: Callable[[Any], None]) -> None:
        """Unregister a previously registered handler."""
        lst = self._handlers.get(message_name, [])
        try:
            lst.remove(handler)
        except ValueError:
            pass

    def dispatch(self, message_name: str, payload: Any) -> None:
        """Call all handlers registered for *message_name*."""
        for handler in self._handlers.get(message_name, []):
            try:
                handler(payload)
            except Exception:
                logger.exception("[cTrader] Push handler for %s raised an exception", message_name)


# ---------------------------------------------------------------------------
# CTraderConnector
# ---------------------------------------------------------------------------


class CTraderConnector:
    """Reusable cTrader Open API connector — Twisted transport + auth.

    Wraps the Twisted-based ``ctrader_open_api`` client in a synchronous
    interface.  A single Twisted reactor runs in a dedicated daemon thread;
    all Protobuf requests are dispatched onto the reactor via
    ``reactor.callFromThread`` and results are returned via
    ``threading.Event``.

    Authentication lifecycle
    ------------------------
    ``start()`` performs three steps in order:

    1. **TCP + TLS connection** — waits for ``_on_connected``.
    2. **Application authentication** — ``ProtoOAApplicationAuthReq``
       (identifies the OAuth2 app to the server).
    3. **Account authentication** — ``ProtoOAAccountAuthReq``
       (authorises the specific trading account).

    After ``start()`` returns ``True``, the connector is ready for
    symbol resolution, data requests, and push subscriptions.

    Push message routing
    --------------------
    Register callbacks for unsolicited server-push messages (e.g.
    ``"ProtoOASpotEvent"``, ``"ProtoOAExecutionEvent"``) via
    :meth:`add_push_handler`.  Handlers are called synchronously on the
    reactor thread — **they must not block**.

    Thread safety
    -------------
    :meth:`send_and_wait` is safe to call from any thread.
    :meth:`add_push_handler` / :meth:`remove_push_handler` are safe to call
    before ``start()`` and during live operation.

    Parameters
    ----------
    credentials:
        cTrader OAuth2 credentials (see :class:`CTraderCredentials`).
    symbol_aliases:
        Optional dict of ``{canonical_name: broker_name}`` overrides.
        Defaults to :data:`DEFAULT_SYMBOL_ALIASES`.
    request_timeout_s:
        Default timeout for individual ``send_and_wait`` calls.
    """

    def __init__(
        self,
        credentials: CTraderCredentials,
        symbol_aliases: Optional[Dict[str, str]] = None,
        request_timeout_s: float = DEFAULT_REQUEST_TIMEOUT_S,
    ) -> None:
        if not _CTRADER_AVAILABLE:
            raise ImportError(
                "ctrader_open_api and twisted are required. "
                "Install them with: pip install ctrader-open-api"
            )
        self.credentials = credentials
        self.request_timeout_s = request_timeout_s
        self._symbol_aliases: Dict[str, str] = (
            symbol_aliases if symbol_aliases is not None else dict(DEFAULT_SYMBOL_ALIASES)
        )

        # Twisted state
        self._client: Any = None
        self._reactor_thread: Optional[threading.Thread] = None

        # Auth synchronisation
        self._connected_event = threading.Event()
        self._app_auth_done = threading.Event()
        self._app_auth_ok: bool = False
        self._acct_auth_done = threading.Event()
        self._acct_auth_ok: bool = False

        # Request/response routing: msg_id → (Event, response)
        self._responses: Dict[str, Any] = {}
        self._response_events: Dict[str, threading.Event] = {}
        self._msg_counter: int = 0
        self._lock = threading.Lock()

        # Clean-shutdown flag — suppresses error-log noise from in-flight deferreds
        self._shutting_down: bool = False

        # Symbol caches
        self._symbol_map: Dict[str, int] = {}  # name → symbolId
        self._symbol_names: Dict[int, str] = {}  # symbolId → name
        self._symbol_digits: Dict[int, int] = {}  # symbolId → price digits
        self._symbol_cache_lock = threading.Lock()

        # Broker metadata (populated after account auth)
        self.broker_title: str = ""
        self.account_login: int = 0
        self._selected_endpoint: str = ""
        self._request_timeout_count: int = 0
        self._app_auth_attempts: int = 0
        self._max_app_auth_attempts: int = max(
            int(_connector_env_value("CTRADER_APP_AUTH_MAX_ATTEMPTS") or "3"),
            1,
        )
        self._app_auth_retry_delay_s: float = max(
            float(_connector_env_value("CTRADER_APP_AUTH_RETRY_DELAY_S") or "1.0"),
            0.2,
        )
        self._app_auth_timeout_s: float = max(
            float(
                _connector_env_value("CTRADER_APP_AUTH_TIMEOUT_S")
                or str(max(float(self.request_timeout_s), 10.0))
            ),
            5.0,
        )

        # Push-message handler registry
        self._push_handlers = _PushHandlerRegistry()
        # Reconnect guard / timeout tracking
        self._reconnect_lock = threading.Lock()
        self._consecutive_request_timeouts: int = 0

    # =========================================================================
    # Public lifecycle
    # =========================================================================

    def start(self, timeout_s: float = DEFAULT_CONNECT_TIMEOUT_S) -> bool:
        """Connect to cTrader, authenticate the app and the account.

        Parameters
        ----------
        timeout_s:
            Maximum seconds to wait for each auth step.

        Returns
        -------
        bool
            ``True`` if all three steps (connect, app auth, account auth)
            succeeded within *timeout_s*.  ``False`` otherwise.

        Raises
        ------
        ImportError
            If ``ctrader_open_api`` / ``twisted`` are not installed.
        RuntimeError
            If the reactor is already running (can only start once).
        """
        host = (
            EndPoints.PROTOBUF_DEMO_HOST
            if self.credentials.environment == "demo"
            else EndPoints.PROTOBUF_LIVE_HOST
        )
        port = EndPoints.PROTOBUF_PORT
        service_name = f"ctrader-{self.credentials.environment}"
        alt_hosts = _connector_env_list("CTRADER_ALT_ENDPOINTS")
        cross_env_host = (
            EndPoints.PROTOBUF_LIVE_HOST
            if self.credentials.environment == "demo"
            else EndPoints.PROTOBUF_DEMO_HOST
        )
        allow_cross_env_fallback = _connector_env_bool(
            "CTRADER_ALLOW_CROSS_ENV_ENDPOINT_FALLBACK", False
        )
        if allow_cross_env_fallback and cross_env_host not in alt_hosts:
            alt_hosts = [*alt_hosts, cross_env_host]
        include_ips = _connector_env_bool("CTRADER_INCLUDE_RESOLVED_IP_FALLBACKS", True)
        probe_timeout_raw = _connector_env_value("CTRADER_ENDPOINT_PROBE_TIMEOUT_S") or "2.0"
        probe_timeout_s = float(probe_timeout_raw)
        probe_retries = max(int(_connector_env_value("CTRADER_ENDPOINT_PROBE_RETRIES") or "2"), 0)
        probe_retry_sleep_s = max(
            float(_connector_env_value("CTRADER_ENDPOINT_PROBE_RETRY_SLEEP_S") or "0.75"), 0.0
        )
        allow_direct_on_probe_fail = _connector_env_bool(
            "CTRADER_ALLOW_DIRECT_CONNECT_ON_PROBE_FAIL", True
        )

        primary_candidates = expand_endpoint_candidates(
            [host],
            include_resolved_ips=include_ips,
            service_name=service_name,
        )
        secondary_candidates = expand_endpoint_candidates(
            alt_hosts,
            include_resolved_ips=include_ips,
            service_name=service_name,
        )
        if not primary_candidates and not secondary_candidates:
            logger.error("[cTrader] No DNS-validated endpoint candidates available")
            return False

        # Promote previously low-latency endpoint pool first.
        cached_pool = _read_endpoint_pool(self.credentials.environment)
        if cached_pool:
            p_pool = [e for e in cached_pool if e in primary_candidates]
            s_pool = [e for e in cached_pool if e in secondary_candidates]
            if p_pool:
                primary_candidates = p_pool + [c for c in primary_candidates if c not in p_pool]
                logger.info("[cTrader] Promoting cached primary endpoint pool: %s", p_pool)
            if s_pool:
                secondary_candidates = s_pool + [c for c in secondary_candidates if c not in s_pool]
                logger.info("[cTrader] Promoting cached secondary endpoint pool: %s", s_pool)

        # Prefer last-known-good endpoint first (if still in candidate set).
        last_ep = _read_last_endpoint(self.credentials.environment)
        if last_ep and last_ep in primary_candidates:
            primary_candidates = [last_ep, *[c for c in primary_candidates if c != last_ep]]
            logger.info("[cTrader] Promoting cached endpoint candidate: %s", last_ep)
        elif last_ep and last_ep in secondary_candidates:
            secondary_candidates = [last_ep, *[c for c in secondary_candidates if c != last_ep]]
            logger.info("[cTrader] Promoting cached secondary endpoint candidate: %s", last_ep)

        selected: Optional[str] = None
        ranked_for_cache: List[str] = []
        for attempt in range(probe_retries + 1):
            ranked_primary = rank_reachable_endpoints(
                primary_candidates,
                port=port,
                timeout_s=probe_timeout_s,
                service_name=service_name,
            )
            if ranked_primary:
                primary_candidates = [ep for ep, _lat in ranked_primary]
                selected = primary_candidates[0]
                ranked_for_cache = list(primary_candidates)
                logger.info(
                    "[cTrader] Lowest-latency PRIMARY endpoint selected: %s (%.1fms)",
                    ranked_primary[0][0],
                    ranked_primary[0][1],
                )
                break

            if secondary_candidates:
                ranked_secondary = rank_reachable_endpoints(
                    secondary_candidates,
                    port=port,
                    timeout_s=probe_timeout_s,
                    service_name=service_name,
                )
                if ranked_secondary:
                    secondary_candidates = [ep for ep, _lat in ranked_secondary]
                    selected = secondary_candidates[0]
                    ranked_for_cache = list(secondary_candidates)
                    logger.warning(
                        "[cTrader] Falling back to SECONDARY endpoint tier, selected: %s (%.1fms)",
                        ranked_secondary[0][0],
                        ranked_secondary[0][1],
                    )
                    break

            if attempt < probe_retries:
                logger.warning(
                    "[cTrader] Endpoint probe attempt %d/%d failed; retrying in %.2fs",
                    attempt + 1,
                    probe_retries + 1,
                    probe_retry_sleep_s,
                )
                time.sleep(probe_retry_sleep_s)

        if selected and ranked_for_cache:
            _write_endpoint_pool(self.credentials.environment, ranked_for_cache)

        if not selected:
            fallback_pool = primary_candidates if primary_candidates else secondary_candidates
            if not allow_direct_on_probe_fail or not fallback_pool:
                logger.error(
                    "[cTrader] No reachable endpoints in candidate set (primary=%s secondary=%s)",
                    primary_candidates,
                    secondary_candidates,
                )
                return False
            selected = fallback_pool[0]
            logger.warning(
                "[cTrader] All endpoint probes failed; falling back to direct connect attempt: %s:%d",
                selected,
                port,
            )

        self._client = Client(selected, port, TcpProtocol)
        self._selected_endpoint = str(selected)
        self._client.setConnectedCallback(self._on_connected)
        self._client.setDisconnectedCallback(self._on_disconnected)
        self._client.setMessageReceivedCallback(self._on_message)

        self._reactor_thread = threading.Thread(
            target=self._run_reactor,
            daemon=True,
            name="ctrader-reactor",
        )
        self._reactor_thread.start()

        # Step 1: TCP + TLS
        if not self._connected_event.wait(timeout=timeout_s):
            logger.error("[cTrader] Connection timeout after %.0fs", timeout_s)
            return False

        # Step 2: App auth
        if not self._app_auth_done.wait(timeout=timeout_s):
            logger.error("[cTrader] Application auth timeout after %.0fs", timeout_s)
            return False
        if not self._app_auth_ok:
            logger.error("[cTrader] Application auth failed")
            return False

        # Step 3: Account auth
        self._do_account_auth(timeout_s=timeout_s)
        if not self._acct_auth_done.wait(timeout=timeout_s):
            logger.error("[cTrader] Account auth timeout after %.0fs", timeout_s)
            return False
        if not self._acct_auth_ok:
            logger.error("[cTrader] Account auth failed")
            return False

        logger.info(
            "[cTrader] Connected: account=%d  broker=%s  env=%s",
            self.credentials.account_id,
            self.broker_title or "unknown",
            self.credentials.environment,
        )
        _write_last_endpoint(self.credentials.environment, selected)
        return True

    def stop(self) -> None:
        """Gracefully shut down the reactor and disconnect.

        Sets ``_shutting_down`` before stopping so that any in-flight
        deferred callbacks suppress their errback logging (prevents
        Twisted's "Unhandled error in Deferred" noise on shutdown).
        """
        self._shutting_down = True
        try:
            if self._client is not None:
                _reactor.callFromThread(self._client.stopService)
            # Brief pause to let pending deferreds flush before reactor stop.
            time.sleep(0.15)
            if _reactor.running:
                _reactor.callFromThread(_reactor.stop)
        except Exception:
            logger.debug("[cTrader] Exception during stop (usually harmless)", exc_info=True)

    def is_connected(self) -> bool:
        """Return ``True`` if the TCP connection is established and authed."""
        return (
            self._connected_event.is_set()
            and self._app_auth_ok
            and self._acct_auth_ok
            and not self._shutting_down
        )

    @property
    def selected_endpoint(self) -> str:
        """Current endpoint host/IP selected for this connector session."""
        return self._selected_endpoint

    @property
    def request_timeout_count(self) -> int:
        """Total request timeouts observed for this connector session."""
        return int(self._request_timeout_count)

    def send_heartbeat(self) -> None:
        """Send a client-initiated ProtoHeartbeatEvent to keep the session alive.

        Must be called periodically (every ~25 s) to prevent the broker from
        closing the idle TCP connection (typical broker idle timeout: 60 s).
        Fire-and-forget — no response is expected.
        """
        if not self.is_connected() or self._client is None:
            return
        try:

            def _send() -> None:
                if self._client is not None:
                    try:
                        d = self._client.send(_common_msgs.ProtoHeartbeatEvent())
                        d.addErrback(lambda _: None)
                    except Exception:
                        pass

            _reactor.callFromThread(_send)
        except Exception:
            pass

    # =========================================================================
    # Public request interface
    # =========================================================================

    def send_and_wait(
        self,
        message: Any,
        timeout_s: Optional[float] = None,
    ) -> Optional[Any]:
        """Send a Protobuf message and block until the response arrives.

        Safe to call from any thread.  The message is dispatched onto the
        Twisted reactor thread; the calling thread blocks on a
        ``threading.Event`` until the response or a timeout.

        Parameters
        ----------
        message:
            Any ``ctrader_open_api`` Protobuf request object.
        timeout_s:
            Response timeout.  Defaults to ``self.request_timeout_s``.

        Returns
        -------
        Any
            The decoded Protobuf response payload, or ``None`` on timeout
            or if the connector is shutting down.
        """
        if timeout_s is None:
            timeout_s = self.request_timeout_s

        # Allow auth-phase requests (e.g. account auth) once TCP is up.
        # Only fail fast when transport is unavailable or connector is shutting down.
        if self._client is None or self._shutting_down or not self._connected_event.is_set():
            logger.warning(
                "[cTrader] Request skipped: transport is not connected; attempting reconnect"
            )
            if not self._attempt_reconnect(timeout_s=max(float(timeout_s), 5.0)):
                return None

        msg_id = self._next_msg_id()
        event = threading.Event()

        with self._lock:
            self._response_events[msg_id] = event

        _reactor.callFromThread(self._do_send, message, msg_id)

        if not event.wait(timeout=timeout_s):
            with self._lock:
                self._response_events.pop(msg_id, None)
                self._request_timeout_count += 1
                self._consecutive_request_timeouts += 1
            logger.warning("[cTrader] Request timeout after %.0fs (msg_id=%s)", timeout_s, msg_id)
            if self._consecutive_request_timeouts >= 3:
                self._attempt_reconnect(timeout_s=max(float(timeout_s), 5.0))
            return None

        with self._lock:
            self._response_events.pop(msg_id, None)
            self._consecutive_request_timeouts = 0
            return self._responses.pop(msg_id, None)

    # =========================================================================
    # Push-handler registration
    # =========================================================================

    def add_push_handler(
        self,
        message_name: str,
        handler: Callable[[Any], None],
    ) -> None:
        """Register *handler* for unsolicited push messages named *message_name*.

        Common message names:
        - ``"ProtoOASpotEvent"``       — live bid/ask + completed trendbars
        - ``"ProtoOAExecutionEvent"``  — order fills, position updates
        - ``"ProtoOAOrderErrorEvent"`` — order rejection notifications

        Handlers are called on the Twisted reactor thread.  **They must not
        block** — use a queue or ``threading.Event`` to hand off to the
        calling thread if needed.

        Parameters
        ----------
        message_name:
            The ``DESCRIPTOR.name`` of the Protobuf message type.
        handler:
            Callable receiving the decoded Protobuf payload as its only
            argument.
        """
        self._push_handlers.add(message_name, handler)

    def remove_push_handler(
        self,
        message_name: str,
        handler: Callable[[Any], None],
    ) -> None:
        """Unregister a previously registered push handler."""
        self._push_handlers.remove(message_name, handler)

    # =========================================================================
    # Symbol resolution
    # =========================================================================

    def resolve_symbols(
        self,
        names: List[str],
        timeout_s: float = DEFAULT_REQUEST_TIMEOUT_S,
    ) -> Dict[str, int]:
        """Resolve a list of symbol names to cTrader symbolIds.

        Populates ``self._symbol_map`` and ``self._symbol_names`` caches.
        Returns only the symbols that were successfully resolved.

        Parameters
        ----------
        names:
            List of canonical symbol names (e.g. ``["XAUUSD", "NAS100"]``).
        timeout_s:
            Per-request timeout for the symbols-list API call.

        Returns
        -------
        dict
            ``{name: symbol_id}`` for all resolved symbols.
        """
        # Fetch the full symbol list if the cache is empty
        with self._symbol_cache_lock:
            if not self._symbol_map:
                self._populate_symbol_cache(timeout_s=timeout_s)

        resolved: Dict[str, int] = {}
        for name in names:
            sid = self.find_symbol_id(name)
            if sid is not None:
                resolved[name] = sid
            else:
                logger.warning("[cTrader] Could not resolve symbol: %s", name)
        return resolved

    def find_symbol_id(
        self,
        name: str,
        timeout_s: float = DEFAULT_REQUEST_TIMEOUT_S,
    ) -> Optional[int]:
        """Find the cTrader symbolId for a given symbol name.

        Resolution order:
        1. Exact match in cache.
        2. Alias lookup (``DEFAULT_SYMBOL_ALIASES``).
        3. Suffix stripping (e.g. ``XAUUSD+`` → ``XAUUSD``).
        4. Case-insensitive prefix match in cache.
        5. Refresh the cache and retry once.

        Parameters
        ----------
        name:
            Canonical or broker symbol name.
        timeout_s:
            Timeout for a cache-refresh API call.

        Returns
        -------
        int or None
        """
        with self._symbol_cache_lock:
            if not self._symbol_map:
                self._populate_symbol_cache(timeout_s=timeout_s)

        return self._resolve_name(name, timeout_s=timeout_s)

    def symbol_name_for_id(self, symbol_id: int) -> str:
        """Return the symbol name for a given cTrader symbolId, or ``"<id>"``."""
        return self._symbol_names.get(symbol_id, f"<{symbol_id}>")

    def get_digits(
        self,
        symbol_id: int,
        timeout_s: float = DEFAULT_REQUEST_TIMEOUT_S,
    ) -> int:
        """Return the number of price decimal digits for a symbol.

        Fetches from the cTrader API on first access; cached thereafter.

        Parameters
        ----------
        symbol_id:
            cTrader integer symbolId.
        timeout_s:
            Request timeout.

        Returns
        -------
        int
            Number of decimal digits (e.g. 5 for EURUSD, 2 for XAUUSD).
            Returns 5 as a safe fallback if the lookup fails.
        """
        with self._symbol_cache_lock:
            if symbol_id in self._symbol_digits:
                return self._symbol_digits[symbol_id]

        digits = self._fetch_symbol_digits(symbol_id, timeout_s=timeout_s)
        if digits is not None:
            with self._symbol_cache_lock:
                self._symbol_digits[symbol_id] = digits
            return digits

        # Fallback: infer from symbol name
        name = self.symbol_name_for_id(symbol_id)
        fallback = _infer_digits_from_name(name)
        logger.debug("[cTrader] get_digits fallback for %s (id=%d): %d", name, symbol_id, fallback)
        return fallback

    def get_account_snapshot(
        self,
        timeout_s: float = DEFAULT_REQUEST_TIMEOUT_S,
    ) -> Dict[str, Any]:
        """
        Poll broker/account summary from cTrader Open API.

        Returns keys:
        - broker_name
        - trader_login
        - account_id
        - balance
        - used_margin
        - account_type
        - deposit_asset_id
        - money_digits
        """
        trader = self._fetch_trader_snapshot(timeout_s=timeout_s)
        if trader is None:
            raise RuntimeError("Failed to fetch trader snapshot")

        money_digits = int(getattr(trader, "moneyDigits", 2) or 2)
        scale = 10 ** max(money_digits, 0)

        balance_raw = int(getattr(trader, "balance", 0) or 0)
        balance = balance_raw / scale
        broker_name = str(getattr(trader, "brokerName", "") or "")
        trader_login = int(getattr(trader, "traderLogin", 0) or 0)
        deposit_asset_id = int(getattr(trader, "depositAssetId", 0) or 0)
        account_type = int(getattr(trader, "accountType", 0) or 0)

        # Reconcile carries current open positions including usedMargin.
        used_margin_raw = 0
        rec_req = _api_msgs.ProtoOAReconcileReq()
        rec_req.ctidTraderAccountId = self.credentials.account_id
        rec = self.send_and_wait(rec_req, timeout_s=timeout_s)
        if rec is not None and not hasattr(rec, "errorCode"):
            for pos in getattr(rec, "position", []):
                used_margin_raw += int(getattr(pos, "usedMargin", 0) or 0)
                pm = int(getattr(pos, "moneyDigits", money_digits) or money_digits)
                if pm >= 0:
                    money_digits = pm
                    scale = 10**pm
        used_margin = used_margin_raw / scale

        return {
            "broker_name": broker_name,
            "trader_login": trader_login,
            "account_id": self.credentials.account_id,
            "balance": float(balance),
            "used_margin": float(used_margin),
            "account_type": account_type,
            "deposit_asset_id": deposit_asset_id,
            "money_digits": money_digits,
        }

    def _fetch_trader_snapshot(self, timeout_s: float = DEFAULT_REQUEST_TIMEOUT_S) -> Optional[Any]:
        """Fetch ProtoOATrader model for the authenticated account."""
        req = _api_msgs.ProtoOATraderReq()
        req.ctidTraderAccountId = self.credentials.account_id
        trader_res = self.send_and_wait(req, timeout_s=timeout_s)
        if (
            trader_res is None
            or hasattr(trader_res, "errorCode")
            or not hasattr(trader_res, "trader")
        ):
            return None
        return trader_res.trader

    # =========================================================================
    # Internal: Twisted reactor
    # =========================================================================

    def _run_reactor(self) -> None:
        """Run the Twisted reactor (blocking) in the reactor daemon thread."""
        self._client.startService()
        try:
            _reactor.run(installSignalHandlers=False)
        except Exception:
            logger.debug("[cTrader] Reactor stopped", exc_info=True)

    # =========================================================================
    # Internal: request/response plumbing
    # =========================================================================

    def _next_msg_id(self) -> str:
        with self._lock:
            self._msg_counter += 1
            return f"msg_{self._msg_counter}"

    def _do_send(self, message: Any, msg_id: str) -> None:
        """Send *message* on the reactor thread.  Called via callFromThread."""
        if self._shutting_down:
            # Signal the waiting thread immediately with a None result.
            with self._lock:
                self._responses[msg_id] = None
                ev = self._response_events.get(msg_id)
            if ev:
                ev.set()
            return

        d = self._client.send(
            message,
            clientMsgId=msg_id,
            responseTimeoutInSeconds=int(self.request_timeout_s),
        )
        d.addCallback(lambda resp: self._on_response(msg_id, resp))
        d.addErrback(lambda failure: self._on_response_error(msg_id, failure))
        # Final suppression errback — swallows CancelledError during shutdown.
        d.addErrback(lambda _: None)

    def _on_response(self, msg_id: str, raw: Any) -> None:
        """Decode and store a response; wake the waiting thread."""
        try:
            payload = Protobuf.extract(raw)
        except Exception:
            payload = raw

        with self._lock:
            self._responses[msg_id] = payload
            ev = self._response_events.get(msg_id)
        if ev:
            ev.set()

    def _on_response_error(self, msg_id: str, failure: Any) -> None:
        """Store the failure object; wake the waiting thread."""
        with self._lock:
            self._responses[msg_id] = failure
            ev = self._response_events.get(msg_id)
        if ev:
            ev.set()

    # =========================================================================
    # Internal: Twisted connection callbacks
    # =========================================================================

    def _on_connected(self, client: Any) -> None:
        """Twisted callback: TCP+TLS connection established."""
        self._app_auth_done.clear()
        self._acct_auth_done.clear()
        self._app_auth_ok = False
        self._acct_auth_ok = False
        self._app_auth_attempts = 0
        self._connected_event.set()
        logger.debug("[cTrader] TCP connected — sending app auth")
        self._send_app_auth(client)

    def _send_app_auth(self, client: Any) -> None:
        """Send app auth request with explicit timeout and retry tracking."""
        self._app_auth_attempts += 1
        req = _api_msgs.ProtoOAApplicationAuthReq()
        req.clientId = self.credentials.client_id
        req.clientSecret = self.credentials.client_secret
        # Use explicit timeout; SDK default (~5s) is too aggressive on noisy links.
        d = client.send(req, responseTimeoutInSeconds=max(int(self._app_auth_timeout_s), 10))
        d.addCallback(self._on_app_auth_response)
        d.addErrback(self._on_app_auth_error)
        d.addErrback(lambda _: None)

    def _on_disconnected(self, client: Any, reason: Any) -> None:
        """Twisted callback: connection dropped."""
        self._connected_event.clear()
        self._app_auth_done.clear()
        self._acct_auth_done.clear()
        self._app_auth_ok = False
        self._acct_auth_ok = False
        if not self._shutting_down:
            logger.warning("[cTrader] Disconnected: %s", reason)

    def _attempt_reconnect(self, timeout_s: float = 10.0) -> bool:
        """Best-effort reconnect + re-auth for an existing connector session."""
        if self._shutting_down or self._client is None:
            return False
        if self.is_connected():
            return True
        if not self._reconnect_lock.acquire(blocking=False):
            # Another thread is already reconnecting; wait briefly for outcome.
            return self._connected_event.wait(timeout=max(timeout_s, 1.0)) and self.is_connected()
        try:
            self._connected_event.clear()
            self._app_auth_done.clear()
            self._acct_auth_done.clear()
            self._app_auth_ok = False
            self._acct_auth_ok = False
            try:
                _reactor.callFromThread(self._client.startService)
            except Exception:
                logger.debug("[cTrader] Reconnect startService failed", exc_info=True)
                return False

            if not self._connected_event.wait(timeout=timeout_s):
                logger.warning(
                    "[cTrader] Reconnect failed: TCP not established in %.1fs", timeout_s
                )
                return False
            if not self._app_auth_done.wait(timeout=timeout_s) or not self._app_auth_ok:
                logger.warning("[cTrader] Reconnect failed: app auth")
                return False

            self._do_account_auth(timeout_s=timeout_s)
            if not self._acct_auth_done.wait(timeout=timeout_s) or not self._acct_auth_ok:
                logger.warning("[cTrader] Reconnect failed: account auth")
                return False

            logger.info("[cTrader] Reconnect successful")
            self._consecutive_request_timeouts = 0
            return True
        finally:
            self._reconnect_lock.release()

    def _on_app_auth_response(self, raw: Any) -> None:
        """Twisted callback: application auth response."""
        try:
            payload = Protobuf.extract(raw)
            name = payload.DESCRIPTOR.name
            if name == "ProtoOAErrorRes":
                code = getattr(payload, "errorCode", "?")
                desc = getattr(payload, "description", "")
                logger.error("[cTrader] App auth error: %s — %s", code, desc)
                self._app_auth_ok = False
            else:
                self._app_auth_ok = True
                logger.debug("[cTrader] App auth OK")
        except Exception as exc:
            logger.error("[cTrader] App auth parse error: %s", exc)
            self._app_auth_ok = False
        finally:
            self._app_auth_done.set()

    def _on_app_auth_error(self, failure: Any) -> None:
        """Twisted errback: application auth failure."""
        remaining = self._max_app_auth_attempts - self._app_auth_attempts
        if (
            not self._shutting_down
            and self._connected_event.is_set()
            and self._client is not None
            and remaining > 0
        ):
            logger.warning(
                "[cTrader] App auth attempt %d/%d failed (%s); retrying in %.1fs",
                self._app_auth_attempts,
                self._max_app_auth_attempts,
                failure,
                self._app_auth_retry_delay_s,
            )

            def _retry() -> None:
                if (
                    self._client is None
                    or self._shutting_down
                    or not self._connected_event.is_set()
                ):
                    return
                self._send_app_auth(self._client)

            _reactor.callFromThread(
                lambda: _reactor.callLater(self._app_auth_retry_delay_s, _retry)
            )
            return

        logger.error(
            "[cTrader] App auth failed after %d attempt(s): %s",
            self._app_auth_attempts,
            failure,
        )
        self._app_auth_ok = False
        self._app_auth_done.set()

    def _on_message(self, client: Any, raw_message: Any) -> None:
        """Twisted callback: unsolicited push message.

        Handles heartbeat keep-alive and dispatches all other messages
        to registered push handlers.
        """
        try:
            payload = Protobuf.extract(raw_message)
        except Exception:
            return

        name = payload.DESCRIPTOR.name

        if name == "ProtoHeartbeatEvent":
            # Echo heartbeat back — fire-and-forget with errback suppression.
            d = client.send(_common_msgs.ProtoHeartbeatEvent())
            d.addErrback(lambda _: None)
            return

        # Dispatch to registered handlers (SpotEvent, ExecutionEvent, etc.)
        self._push_handlers.dispatch(name, payload)

    # =========================================================================
    # Internal: account authentication
    # =========================================================================

    def _do_account_auth(self, timeout_s: float = DEFAULT_REQUEST_TIMEOUT_S) -> None:
        """Send account auth request and wait for response."""
        req = _api_msgs.ProtoOAAccountAuthReq()
        req.ctidTraderAccountId = self.credentials.account_id
        req.accessToken = self.credentials.access_token

        resp = self.send_and_wait(req, timeout_s=timeout_s)

        if resp is None:
            logger.error("[cTrader] Account auth timeout (account %d)", self.credentials.account_id)
            self._acct_auth_ok = False
        elif hasattr(resp, "errorCode"):
            code = getattr(resp, "errorCode", "?")
            desc = getattr(resp, "description", "")
            logger.error("[cTrader] Account auth error: %s — %s", code, desc)
            self._acct_auth_ok = False
        else:
            self._acct_auth_ok = True
            logger.debug("[cTrader] Account auth OK (account %d)", self.credentials.account_id)
            trader = self._fetch_trader_snapshot(timeout_s=timeout_s)
            if trader is not None:
                self.broker_title = str(getattr(trader, "brokerName", "") or "")
                self.account_login = int(getattr(trader, "traderLogin", 0) or 0)
            else:
                logger.debug(
                    "[cTrader] Could not fetch trader profile after auth (account %d)",
                    self.credentials.account_id,
                )
        self._acct_auth_done.set()

    # =========================================================================
    # Internal: symbol cache
    # =========================================================================

    def _populate_symbol_cache(self, timeout_s: float = DEFAULT_REQUEST_TIMEOUT_S) -> None:
        """Fetch the full symbol list and populate name↔id caches.

        Must be called with ``self._symbol_cache_lock`` held.
        """
        req = _api_msgs.ProtoOASymbolsListReq()
        req.ctidTraderAccountId = self.credentials.account_id

        resp = self.send_and_wait(req, timeout_s=timeout_s)
        if resp is None or hasattr(resp, "errorCode"):
            logger.warning("[cTrader] Symbol list fetch failed — cache empty")
            return

        count = 0
        if hasattr(resp, "symbol"):
            for sym in resp.symbol:
                sid = getattr(sym, "symbolId", None)
                sname = getattr(sym, "symbolName", None)
                if sid is not None and sname:
                    self._symbol_map[sname] = int(sid)
                    self._symbol_names[int(sid)] = sname
                    count += 1

        logger.debug("[cTrader] Symbol cache populated: %d symbols", count)

    def _resolve_name(
        self,
        name: str,
        timeout_s: float = DEFAULT_REQUEST_TIMEOUT_S,
        _retry: bool = True,
    ) -> Optional[int]:
        """Internal: resolve *name* using cache + alias + suffix strategies."""
        # 1. Exact match
        with self._symbol_cache_lock:
            if name in self._symbol_map:
                return self._symbol_map[name]

        # 2. Alias override
        alias = self._symbol_aliases.get(name)
        if alias:
            with self._symbol_cache_lock:
                if alias in self._symbol_map:
                    return self._symbol_map[alias]

        # 3. Suffix stripped (e.g. "XAUUSD+" → "XAUUSD")
        stripped = name.rstrip("+").rstrip("-C")
        if stripped != name:
            with self._symbol_cache_lock:
                if stripped in self._symbol_map:
                    return self._symbol_map[stripped]

        # 4. Case-insensitive prefix match
        name_upper = name.upper()
        with self._symbol_cache_lock:
            for sym_name, sym_id in self._symbol_map.items():
                if sym_name.upper().startswith(name_upper):
                    logger.debug("[cTrader] Fuzzy match: %s → %s (id=%d)", name, sym_name, sym_id)
                    return sym_id

        # 5. Retry after cache refresh (once)
        if _retry:
            logger.debug("[cTrader] Symbol %r not found — refreshing cache", name)
            with self._symbol_cache_lock:
                self._symbol_map.clear()
                self._symbol_names.clear()
                self._populate_symbol_cache(timeout_s=timeout_s)
            return self._resolve_name(name, timeout_s=timeout_s, _retry=False)

        return None

    def _fetch_symbol_digits(
        self,
        symbol_id: int,
        timeout_s: float = DEFAULT_REQUEST_TIMEOUT_S,
    ) -> Optional[int]:
        """Fetch the digits (price precision) for *symbol_id* via the API."""
        req = _api_msgs.ProtoOASymbolByIdReq()
        req.ctidTraderAccountId = self.credentials.account_id
        req.symbolId.append(symbol_id)

        resp = self.send_and_wait(req, timeout_s=timeout_s)
        if resp is None or hasattr(resp, "errorCode"):
            return None

        if hasattr(resp, "symbol") and resp.symbol:
            sym = resp.symbol[0]
            # Prefer the 'digits' field; fall back to 'pipPosition' + 1
            digits = getattr(sym, "digits", None)
            if digits is not None and digits > 0:
                return int(digits)
            pip_pos = getattr(sym, "pipPosition", None)
            if pip_pos is not None:
                return int(pip_pos) + 1

        return None


class HotStandbyCTraderConnector:
    """
    Two-connector wrapper with atomic active/standby failover and persisted state.

    Enabled via ``CTRADER_HOT_STANDBY=1`` in ``build_connector``.
    """

    def __init__(
        self,
        primary: CTraderConnector,
        standby: CTraderConnector,
        *,
        environment: str,
    ) -> None:
        self._connectors: List[CTraderConnector] = [primary, standby]
        self._active_idx = 0
        self._failover_count = 0
        self._generation = 0
        self._last_failover_utc = ""
        self._lock = threading.Lock()
        self._handler_registry: List[Tuple[str, Callable[[Any], None]]] = []
        self._state_path = _hot_standby_state_path(environment)
        self._restore_state()

    @property
    def credentials(self) -> CTraderCredentials:
        return self._connectors[self._active_idx].credentials

    @property
    def selected_endpoint(self) -> str:
        return self._connectors[self._active_idx].selected_endpoint

    @property
    def request_timeout_count(self) -> int:
        return self._connectors[0].request_timeout_count + self._connectors[1].request_timeout_count

    @property
    def failover_count(self) -> int:
        return int(self._failover_count)

    @property
    def failover_generation(self) -> int:
        return int(self._generation)

    @property
    def last_failover_utc(self) -> str:
        return self._last_failover_utc

    @property
    def standby_connected(self) -> bool:
        return self._standby().is_connected()

    @property
    def health_status(self) -> str:
        active_up = self._active().is_connected()
        standby_up = self._standby().is_connected()
        if active_up and standby_up:
            return "UP"
        if active_up and not standby_up:
            return "DEGRADED"
        return "DOWN"

    def _restore_state(self) -> None:
        try:
            if not self._state_path.exists():
                return
            payload = json.loads(self._state_path.read_text(encoding="utf-8"))
            idx = int(payload.get("active_idx", 0))
            self._active_idx = 0 if idx not in {0, 1} else idx
            self._failover_count = int(payload.get("failover_count", 0))
            self._generation = int(payload.get("generation", 0))
            self._last_failover_utc = str(payload.get("last_failover_utc", "") or "")
        except Exception:
            logger.debug("[cTrader] Could not restore hot-standby state", exc_info=True)

    def _persist_state(self) -> None:
        try:
            payload = {
                "active_idx": int(self._active_idx),
                "failover_count": int(self._failover_count),
                "generation": int(self._generation),
                "last_failover_utc": self._last_failover_utc,
                "active_endpoint": self._connectors[self._active_idx].selected_endpoint,
                "standby_endpoint": self._connectors[1 - self._active_idx].selected_endpoint,
                "updated_utc": datetime.now(timezone.utc).isoformat(),
            }
            _atomic_write_json(self._state_path, payload)
        except Exception:
            logger.debug("[cTrader] Could not persist hot-standby state", exc_info=True)

    def _active(self) -> CTraderConnector:
        return self._connectors[self._active_idx]

    def _standby(self) -> CTraderConnector:
        return self._connectors[1 - self._active_idx]

    def start(self, timeout_s: float = DEFAULT_CONNECT_TIMEOUT_S) -> bool:
        ok_active = self._active().start(timeout_s=timeout_s)
        standby_retries = max(
            int(_connector_env_value("CTRADER_HOT_STANDBY_START_RETRIES") or "2"), 0
        )
        standby_delay_s = max(
            float(_connector_env_value("CTRADER_HOT_STANDBY_START_RETRY_DELAY_S") or "2.0"),
            0.0,
        )
        ok_standby = False
        attempts = standby_retries + 1
        for attempt in range(1, attempts + 1):
            ok_standby = self._standby().start(timeout_s=timeout_s)
            if ok_standby:
                break
            if attempt < attempts:
                logger.warning(
                    "[cTrader] Standby start attempt %d/%d failed; retrying in %.1fs",
                    attempt,
                    attempts,
                    standby_delay_s,
                )
                try:
                    self._standby().stop()
                except Exception:
                    pass
                if standby_delay_s > 0:
                    time.sleep(standby_delay_s)
        if not ok_active:
            return False
        if not ok_standby:
            logger.warning("[cTrader] Hot standby failed to start; running single-active mode")
        for name, handler in self._handler_registry:
            try:
                self._active().add_push_handler(name, handler)
                if ok_standby:
                    self._standby().add_push_handler(name, handler)
            except Exception:
                pass
        self._persist_state()
        return True

    def stop(self) -> None:
        for c in self._connectors:
            try:
                c.stop()
            except Exception:
                pass
        self._persist_state()

    def is_connected(self) -> bool:
        return self._active().is_connected()

    def _atomic_failover(self, reason: str = "unknown") -> bool:
        with self._lock:
            old_idx = self._active_idx
            new_idx = 1 - old_idx
            new_conn = self._connectors[new_idx]
            if not new_conn.is_connected():
                if not new_conn.start(timeout_s=DEFAULT_CONNECT_TIMEOUT_S):
                    logger.warning("[cTrader] Hot-standby failover blocked: standby not connected")
                    return False
                for name, handler in self._handler_registry:
                    try:
                        new_conn.add_push_handler(name, handler)
                    except Exception:
                        pass
            self._active_idx = new_idx
            self._failover_count += 1
            self._generation += 1
            self._last_failover_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
            logger.warning(
                "[cTrader] Hot-standby failover: %s -> %s (reason=%s, generation=%d)",
                self._connectors[old_idx].selected_endpoint or f"connector-{old_idx}",
                self._connectors[new_idx].selected_endpoint or f"connector-{new_idx}",
                reason,
                self._generation,
            )
            self._persist_state()
            return True

    def add_push_handler(self, message_name: str, handler: Callable[[Any], None]) -> None:
        self._handler_registry.append((message_name, handler))
        self._active().add_push_handler(message_name, handler)
        self._standby().add_push_handler(message_name, handler)

    def remove_push_handler(self, message_name: str, handler: Callable[[Any], None]) -> None:
        self._handler_registry = [
            (n, h) for (n, h) in self._handler_registry if not (n == message_name and h == handler)
        ]
        self._active().remove_push_handler(message_name, handler)
        self._standby().remove_push_handler(message_name, handler)

    def send_and_wait(self, message: Any, timeout_s: Optional[float] = None) -> Optional[Any]:
        resp = self._active().send_and_wait(message, timeout_s=timeout_s)
        if resp is not None:
            return resp
        if not self._active().is_connected() or self._active().request_timeout_count > 0:
            if self._atomic_failover(reason="request_failure"):
                return self._active().send_and_wait(message, timeout_s=timeout_s)
        return resp

    def send_heartbeat(self) -> None:
        self._active().send_heartbeat()
        try:
            self._standby().send_heartbeat()
        except Exception:
            pass

    def resolve_symbols(
        self, names: List[str], timeout_s: float = DEFAULT_REQUEST_TIMEOUT_S
    ) -> Dict[str, int]:
        return self._active().resolve_symbols(names, timeout_s=timeout_s)

    def find_symbol_id(
        self, name: str, timeout_s: float = DEFAULT_REQUEST_TIMEOUT_S
    ) -> Optional[int]:
        sid = self._active().find_symbol_id(name, timeout_s=timeout_s)
        if sid is None and self._atomic_failover(reason="resolve_symbol"):
            sid = self._active().find_symbol_id(name, timeout_s=timeout_s)
        return sid

    def symbol_name_for_id(self, symbol_id: int) -> str:
        return self._active().symbol_name_for_id(symbol_id)

    def get_digits(self, symbol_id: int, timeout_s: float = DEFAULT_REQUEST_TIMEOUT_S) -> int:
        return self._active().get_digits(symbol_id, timeout_s=timeout_s)

    def get_account_snapshot(self, timeout_s: float = DEFAULT_REQUEST_TIMEOUT_S) -> Dict[str, Any]:
        try:
            return self._active().get_account_snapshot(timeout_s=timeout_s)
        except Exception:
            if self._atomic_failover(reason="account_snapshot"):
                return self._active().get_account_snapshot(timeout_s=timeout_s)
            raise


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _parse_env_file(path: Path) -> Dict[str, str]:
    """Parse a simple ``KEY=VALUE`` env file, ignoring comments and blanks."""
    result: Dict[str, str] = {}
    try:
        with open(path, "r") as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" not in line:
                    continue
                key, _, val = line.partition("=")
                key = key.strip()
                val = val.strip().strip('"').strip("'")
                # Strip ENC:: prefix (encrypted credential marker)
                if val.startswith("ENC::"):
                    val = val[5:]
                if key:
                    result[key] = val
    except OSError:
        pass
    return result


def _infer_digits_from_name(symbol: str) -> int:
    """Heuristic fallback: infer price digits from the symbol name.

    Based on typical broker conventions:
    - BTC/ETH/crypto pairs  → 2
    - Indices (NAS/US/GER)  → 2
    - Metals (XAU/XAG)      → 2
    - FX majors             → 5
    - JPY pairs             → 3
    """
    s = symbol.upper()
    if any(x in s for x in ("BTC", "ETH", "LTC", "XRP", "ADA", "SOL")):
        return 2
    if any(x in s for x in ("NAS", "US30", "US500", "SP500", "GER", "UK1", "JP2", "HK5")):
        return 2
    if any(x in s for x in ("XAU", "XAG", "GOLD", "SILVER")):
        return 2
    if "JPY" in s:
        return 3
    return 5


def build_connector(
    credentials: Optional[CTraderCredentials] = None,
    timeout_s: float = DEFAULT_CONNECT_TIMEOUT_S,
    symbol_aliases: Optional[Dict[str, str]] = None,
) -> Any:
    """Convenience factory: create and start a :class:`CTraderConnector`.

    If *credentials* is ``None``, credentials are loaded from environment
    variables / ``.env`` files via :meth:`CTraderCredentials.from_env`.

    Parameters
    ----------
    credentials:
        Optional explicit credentials.  ``None`` → read from env.
    timeout_s:
        Connection + auth timeout.
    symbol_aliases:
        Optional symbol alias overrides.

    Returns
    -------
    CTraderConnector
        A connected, authenticated connector ready for use.

    Raises
    ------
    RuntimeError
        If connection or authentication fails.
    """
    if credentials is None:
        credentials = CTraderCredentials.from_env()

    use_hot_standby = _connector_env_bool("CTRADER_HOT_STANDBY", False)
    if use_hot_standby:
        primary = CTraderConnector(
            credentials=credentials,
            symbol_aliases=symbol_aliases,
            request_timeout_s=timeout_s,
        )
        standby = CTraderConnector(
            credentials=credentials,
            symbol_aliases=symbol_aliases,
            request_timeout_s=timeout_s,
        )
        wrapper = HotStandbyCTraderConnector(
            primary,
            standby,
            environment=credentials.environment,
        )
        ok = wrapper.start(timeout_s=timeout_s)
        if not ok:
            raise RuntimeError(
                f"HotStandbyCTraderConnector failed to connect to "
                f"{credentials.environment} account {credentials.account_id}"
            )
        return wrapper

    connector = CTraderConnector(
        credentials=credentials,
        symbol_aliases=symbol_aliases,
        request_timeout_s=timeout_s,
    )
    ok = connector.start(timeout_s=timeout_s)
    if not ok:
        raise RuntimeError(
            f"CTraderConnector failed to connect to "
            f"{credentials.environment} account {credentials.account_id}"
        )
    return connector
