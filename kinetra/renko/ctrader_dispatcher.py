"""
cTrader Live Dispatcher
=======================

Sprint 6 — Concrete cTrader implementations of the broker-neutral interfaces
defined in :mod:`kinetra.renko.live_trader`.

Provides
--------
- :class:`CTraderBarProvider`
    Implements :class:`~kinetra.renko.live_trader.BarProvider` using the
    cTrader Open API **native live trendbar subscription**
    (``ProtoOASubscribeLiveTrendbarReq``).  When a completed M1 trendbar
    arrives inside a ``ProtoOASpotEvent``, it is decoded from delta
    encoding and pushed to all registered subscriber callbacks.

    cTrader is the **primary research and live-trading feed** (§29.4) because:
    - Tighter spreads than MetaAPI/MT5.
    - Cleaner UTC alignment (no broker-timezone offset artefacts).
    - Native server-side Renko-compatible bar delivery (ProtoOATrendbar)
      in the SpotEvent stream — no M1 reconstruction needed on our side.
    - Persistent TCP+TLS socket; no cloud proxy latency.

- :class:`CTraderOrderDispatcher`
    Implements :class:`~kinetra.renko.live_trader.OrderDispatcher` by
    submitting ``ProtoOANewOrderReq`` (MARKET type with hard stop-loss)
    and ``ProtoOAClosePositionReq`` via the shared
    :class:`~kinetra.connectors.ctrader_connector.CTraderConnector`.
    Execution events are received via the ``ProtoOAExecutionEvent`` push
    handler on the same connection.

Why cTrader first (MT5 / MetaAPI compatibility note)
------------------------------------------------------
MT5 has no native Renko brick API.  MetaAPI wraps MT5 and therefore also
has no native Renko.  In both cases Renko bricks must be constructed
client-side from M1 closes using ``build_renko()``.

cTrader's ``ProtoOATrendbar`` IS a completed OHLCV bar (M1, M5, …) —
NOT a Renko brick in the charting sense.  The "native Renko" that cTrader
advertises in its charting UI is a server-side aggregation of these bars,
but the Open API only exposes the underlying OHLCV bars, not the Renko
bricks themselves.

Therefore our construction path is identical for all brokers:
    broker bar feed (M1 close) → build_renko() → Layer 1 flip filter

The cTrader advantage over MetaAPI/MT5 is:
    1. Lower latency (TCP vs WebSocket via cloud).
    2. Cleaner data (tighter spreads, better UTC alignment).
    3. No cloud-relay dependency.

Spread estimation
-----------------
``ProtoOASpotEvent`` carries ``bid`` and ``ask`` as integer fields scaled
by ``10^digits``.  Both the connector's digit cache and the raw integer
values are used to convert to a floating-point spread in price units
(same units as the brick size and stop).  ``get_spread_pts()`` returns
this live value with a 1.0-point fallback if not yet observed.

Order-fill tracking (two-phase)
--------------------------------
``CTraderOrderDispatcher.open_position()`` posts a ``ProtoOANewOrderReq``
and blocks (up to ``fill_timeout_s``) for the corresponding fill event.
An ``_ExecutionWaiter`` is registered on the push handler **before** the
request is sent (so no event is ever lost to a race).

The waiter uses a two-phase matching strategy:

1. **ORDER_ACCEPTED** (``executionType == 1``) — matched by ``clientOrderId``
   / ``label``.  The broker-assigned ``orderId`` is stored; the waiter does
   **not** fire yet.
2. **ORDER_FILLED / POSITION_OPENED** (``executionType`` in
   ``_SUCCESS_EXECUTION_TYPES``) — matched by ``clientOrderId``, ``label``,
   **or** the broker ``orderId`` learned in phase 1.  Only then does the
   waiter fire.

Events that carry no identifiers are always rejected to prevent false-positive
captures from unrelated concurrent execution events (e.g. stop-loss hits on
other open positions).

Hard rules (§28 / §29 AGENT_RULES_MASTER.md)
----------------------------------------------
- ❌ Never import ``ctrader_open_api`` or ``twisted`` outside
  ``kinetra/connectors/ctrader_connector.py``.  This module only imports
  the connector.
- ❌ Never apply sizing, risk, or signal logic here — pure plumbing.
- ❌ Never open a second CTraderConnector for orders if one already exists
  for bar data — share the single authenticated session.
- ✅ Always guard the import with ``try/except ImportError``.
- ✅ Always unsubscribe from market data in ``stop()``.
- ✅ Always record broker_source = "ctrader" in every trade log entry.

Usage::

    from kinetra.connectors.ctrader_connector import CTraderCredentials
    from kinetra.renko.ctrader_dispatcher import (
        CTraderBarProvider,
        CTraderOrderDispatcher,
        build_ctrader_session,
    )
    from kinetra.renko.live_trader import RenkoLiveTrader, LiveTraderConfig, PERGate

    dispatcher, bar_provider = build_ctrader_session()

    config = LiveTraderConfig(
        symbols=["XAUUSD", "NAS100"],
        gate=PERGate.SIMULATED,
        target_risk_usd=50.0,
        dispatcher=dispatcher,
    )
    trader = RenkoLiveTrader(config, bar_provider=bar_provider)
    trader.start()

See Also
--------
- ``kinetra/connectors/ctrader_connector.py``  — shared Twisted transport
- ``kinetra/renko/live_trader.py``             — OrderDispatcher / BarProvider ABCs
- ``scripts/ctrader/download_ctrader_history.py`` — historical download using the same connector
- ``AGENT_RULES_MASTER.md §28 / §29``          — multi-broker + Renko architecture rules
"""

from __future__ import annotations

import logging
import math
import os
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Set

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional connector import — guard so that broker-blind modules stay importable
# ---------------------------------------------------------------------------

try:
    from kinetra.connectors.ctrader_connector import (
        CTRADER_M1_PERIOD,
        CTraderConnector,
        CTraderCredentials,
        build_connector,
    )

    _CONNECTOR_AVAILABLE = True
except ImportError:
    _CONNECTOR_AVAILABLE = False
    CTraderConnector = None  # type: ignore[assignment,misc]
    CTraderCredentials = None  # type: ignore[assignment,misc]
    build_connector = None  # type: ignore[assignment]
    CTRADER_M1_PERIOD = 1

# ---------------------------------------------------------------------------
# cTrader protobuf message types — accessed via the connector's _api_msgs.
# We never import ctrader_open_api directly here; instead we use the
# connector's already-imported module references exposed via class attrs.
# ---------------------------------------------------------------------------

#: ProtoOAOrderType.MARKET value (integer constant — avoids direct enum import).
_ORDER_TYPE_MARKET: int = 1

#: ProtoOATradeSide values.
_TRADE_SIDE_BUY: int = 1
_TRADE_SIDE_SELL: int = 2

#: ProtoOATimeInForce.FILL_OR_KILL — default for market orders.
_TIF_FOK: int = 1

#: ProtoOATrendbarPeriod.M1
_PERIOD_M1: int = 1

# ProtoOAExecutionType enum values (from openapi-proto-messages/OpenApiModelMessages.proto)
#   ORDER_ACCEPTED       = 2
#   ORDER_FILLED         = 3
#   ORDER_REPLACED       = 4
#   ORDER_CANCELLED      = 5
#   ORDER_EXPIRED        = 6
#   ORDER_REJECTED       = 7
#   ORDER_CANCEL_REJECTED= 8
#   SWAP                 = 9
#   DEPOSIT_WITHDRAW     = 10
#   ORDER_PARTIAL_FILL   = 11

#: Execution types that indicate a complete or partial fill (open or close).
_FILL_EXECUTION_TYPES: frozenset[int] = frozenset(
    {
        3,  # ORDER_FILLED
        11,  # ORDER_PARTIAL_FILL
    }
)

#: Execution types that fire the _ExecutionWaiter for an open order.
_SUCCESS_EXECUTION_TYPES: frozenset[int] = frozenset(
    {
        3,  # ORDER_FILLED
        11,  # ORDER_PARTIAL_FILL
    }
)

#: Execution types that indicate a close / position exit.
_CLOSE_EXECUTION_TYPES: frozenset[int] = frozenset(
    {
        3,  # ORDER_FILLED (closing order)
        11,  # ORDER_PARTIAL_FILL (partial close)
    }
)

#: Seconds to wait for a fill notification after sending a market order.
_FILL_WAIT_S: float = float(os.getenv("CTRADER_FILL_WAIT_S", "30.0"))

#: Seconds between spot-subscription retries on startup.
_SUBSCRIBE_RETRY_INTERVAL_S: float = max(
    float(os.getenv("CTRADER_SUBSCRIBE_RETRY_INTERVAL_S", "3.0")),
    0.2,
)

#: Number of subscription retry attempts per symbol before giving up.
_SUBSCRIBE_MAX_RETRIES: int = max(
    int(os.getenv("CTRADER_SUBSCRIBE_MAX_RETRIES", "12")),
    1,
)


def _extract_response_error(resp: Any) -> Optional[str]:
    """Return normalized error text for a broker response, or None."""
    if resp is None:
        return None
    code = getattr(resp, "errorCode", None)
    desc = str(getattr(resp, "description", "") or "").strip()
    if code is None:
        # Some explicit error payload types may not carry errorCode.
        name = type(resp).__name__.lower()
        if "error" in name and desc:
            return desc
        return None
    if isinstance(code, (int, float)):
        c = int(code)
        return f"{c}: {desc}" if c != 0 else None
    cstr = str(code).strip()
    if not cstr:
        return None
    if cstr in {"0", "NONE", "NO_ERROR", "OK", "NULL"}:
        return None
    return f"{cstr}: {desc}"


def _select_price_scale(
    *,
    low_raw: int,
    delta_open: int,
    delta_high: int,
    delta_close: int,
    digits: int,
    pip_position: int,
) -> float:
    """Pick a robust raw-price scale for cTrader trendbar decoding."""
    scales = {
        float(10 ** max(digits, 0)),
        float(10 ** max(pip_position + 1, 0)),
        float(10 ** max(digits + 1, 0)),
        float(10 ** max(digits + 2, 0)),
        100000.0,
    }
    best_scale = 100000.0
    best_score = float("inf")
    for scale in scales:
        l = low_raw / scale
        o = (low_raw + delta_open) / scale
        h = (low_raw + delta_high) / scale
        c = (low_raw + delta_close) / scale
        if not (o > 0 and h > 0 and l > 0 and c > 0):
            continue
        if h < l:
            continue
        rel_range = (h - l) / max(abs(c), 1e-12)
        score = rel_range
        if c < 1e-6 or c > 1e6:
            score += 1e3
        if score < best_score:
            best_score = score
            best_scale = scale
    return best_scale


def _select_spot_scale(*, raw_diff: int, digits: int, pip_position: int) -> float:
    """Pick a robust raw-price scale for cTrader bid/ask spot decoding."""
    if raw_diff <= 0:
        return float(10 ** max(digits, 0))
    candidates = {
        float(10 ** max(digits, 0)),
        float(10 ** max(pip_position + 1, 0)),
        float(10 ** max(digits + 1, 0)),
        float(10 ** max(digits + 2, 0)),
        100000.0,
    }
    spread_candidates: List[float] = []
    for scale in candidates:
        spread = raw_diff / scale
        if math.isfinite(spread) and spread > 0:
            spread_candidates.append(float(spread))
    if not spread_candidates:
        return float(10 ** max(digits, 0))
    # Conservative: choose smallest positive spread candidate.
    chosen_spread = min(spread_candidates)
    for scale in sorted(candidates):
        spread = raw_diff / scale
        if abs(spread - chosen_spread) <= 1e-12:
            return float(scale)
    return 100000.0


# ---------------------------------------------------------------------------
# _ExecutionWaiter — captures the fill event for a specific order
# ---------------------------------------------------------------------------


class _ExecutionWaiter:
    """Thread-safe one-shot waiter for a ``ProtoOAExecutionEvent``.

    Registered as a push handler on ``"ProtoOAExecutionEvent"`` immediately
    before sending the order request.  The handler fires on the Twisted
    reactor thread; the placing thread blocks on :meth:`wait` until the
    matching execution event arrives or the timeout expires.

    Matching strategy (two-phase)
    -----------------------------
    1. **ORDER_ACCEPTED** (``executionType == 1``): matched by ``clientOrderId``
       or ``label``.  The broker-assigned ``orderId`` is extracted and stored so
       that the subsequent ORDER_FILLED event can be matched even if cTrader
       omits ``clientOrderId`` on the fill notification.  The waiter does **not**
       fire on ORDER_ACCEPTED — it continues waiting.
    2. **ORDER_FILLED / POSITION_OPENED** (types in ``_SUCCESS_EXECUTION_TYPES``):
       matched by ``clientOrderId``, ``label``, **or** the stored broker
       ``orderId`` learned in phase 1.  Only then does the waiter fire.
    3. Events with **no identifiers at all** are always rejected to prevent
       false-positive captures from unrelated concurrent execution events
       (e.g. stop-loss hits on other positions).

    Parameters
    ----------
    client_order_id:
        The ``clientOrderId`` string we set in ``ProtoOANewOrderReq``.
    timeout_s:
        How long :meth:`wait` will block before returning ``None``.
    """

    #: executionType value for ORDER_ACCEPTED (pre-fill acknowledgement).
    _TYPE_ORDER_ACCEPTED: int = 2

    def __init__(self, client_order_id: str, timeout_s: float = _FILL_WAIT_S) -> None:
        self._coid = client_order_id
        self._timeout_s = timeout_s
        self._event = threading.Event()
        self._payload: Optional[Any] = None
        # Broker-assigned orderId learned from the ORDER_ACCEPTED phase.
        self._broker_order_id: Optional[str] = None
        self._id_lock = threading.Lock()

    def handle(self, payload: Any) -> None:
        """Push-handler callback — called on the reactor thread."""
        exec_type = int(getattr(payload, "executionType", -1))

        order = getattr(payload, "order", None)
        label = ""
        event_client_oid = ""
        event_broker_oid = ""
        if order is not None:
            trade_data = getattr(order, "tradeData", None)
            if trade_data is not None:
                label = str(getattr(trade_data, "label", "") or "")
            event_client_oid = str(getattr(order, "clientOrderId", "") or "")
            event_broker_oid = str(getattr(order, "orderId", "") or "")

        # --- identity check -------------------------------------------------
        # Reject events that carry no identifiers — they cannot be attributed
        # to our order and accepting them would cause false-positive captures.
        with self._id_lock:
            known_broker_oid = self._broker_order_id

        matched_by_client = self._coid and self._coid in {label, event_client_oid}
        matched_by_broker = bool(
            known_broker_oid and event_broker_oid and known_broker_oid == event_broker_oid
        )

        if not (label or event_client_oid or event_broker_oid):
            return  # No identifiers — cannot be our order

        if not (matched_by_client or matched_by_broker):
            return  # Identifiers present but none match ours

        # --- phase 1: ORDER_ACCEPTED ----------------------------------------
        # Learn the broker orderId for phase-2 matching, then keep waiting.
        if exec_type == self._TYPE_ORDER_ACCEPTED:
            if event_broker_oid:
                with self._id_lock:
                    if self._broker_order_id is None:
                        self._broker_order_id = event_broker_oid
            return  # Do not fire yet — wait for the actual fill

        # --- phase 2: fill event --------------------------------------------
        # Only fire on recognised fill execution types.
        if exec_type not in _SUCCESS_EXECUTION_TYPES:
            return  # Skip ORDER_REJECTED, ORDER_EXPIRED, etc.

        self._payload = payload
        self._event.set()

    def wait(self) -> Optional[Any]:
        """Block until the matching execution event arrives or timeout.

        Returns
        -------
        Any or None
            The ``ProtoOAExecutionEvent`` payload, or ``None`` on timeout.
        """
        self._event.wait(timeout=self._timeout_s)
        return self._payload


# ---------------------------------------------------------------------------
# _SpotCache — thread-safe bid/ask + digit cache updated by SpotEvent stream
# ---------------------------------------------------------------------------


@dataclass
class _SpotCache:
    """Per-symbol live bid/ask cache fed by ``ProtoOASpotEvent`` messages.

    Prices arrive as raw integers scaled by ``10^digits``.  We store both
    the raw integers and the floating-point spread so that
    ``get_spread()`` never needs to compute divisions on the hot path.

    Attributes
    ----------
    bid:
        Latest bid price (float).
    ask:
        Latest ask price (float).
    spread:
        ``ask - bid`` (float, in price units matching brick size).
    digits:
        cTrader price precision for this symbol.
    updated_at:
        UTC timestamp of the last update.
    """

    bid: float = 0.0
    ask: float = 0.0
    spread: float = 0.0
    digits: int = 5
    updated_at: Optional[datetime] = None


class _SpotCacheRegistry:
    """Thread-safe registry of :class:`_SpotCache` per symbol-id."""

    def __init__(self) -> None:
        self._cache: Dict[int, _SpotCache] = {}
        self._lock = threading.Lock()

    def update(self, symbol_id: int, bid_raw: int, ask_raw: int, digits: int) -> None:
        """Update the cache entry for *symbol_id*."""
        safe_digits = int(digits) if isinstance(digits, int) else int(digits or 0)
        if safe_digits <= 0 or safe_digits > 10:
            with self._lock:
                cached = self._cache.get(symbol_id)
                safe_digits = cached.digits if cached is not None else 5
        safe_digits = max(1, min(safe_digits, 10))
        scale = 10**safe_digits
        bid = bid_raw / scale
        ask = ask_raw / scale
        with self._lock:
            entry = self._cache.setdefault(symbol_id, _SpotCache(digits=safe_digits))
            entry.bid = bid
            entry.ask = ask
            entry.spread = max(ask - bid, 0.0)
            entry.digits = safe_digits
            entry.updated_at = datetime.now(tz=timezone.utc)

    def get_spread(self, symbol_id: int) -> Optional[float]:
        """Return the current spread for *symbol_id*, or None if unseen."""
        with self._lock:
            entry = self._cache.get(symbol_id)
            return entry.spread if entry is not None else None

    def get_digits(self, symbol_id: int) -> int:
        """Return cached digits for *symbol_id*, defaulting to 5."""
        with self._lock:
            entry = self._cache.get(symbol_id)
            d = entry.digits if entry is not None else 5
            return max(1, min(int(d), 10))

    def get_mid(self, symbol_id: int) -> Optional[float]:
        """Return current mid price for *symbol_id*, or None if unavailable."""
        with self._lock:
            entry = self._cache.get(symbol_id)
            if entry is None:
                return None
            if entry.bid <= 0 or entry.ask <= 0 or entry.ask < entry.bid:
                return None
            return (entry.bid + entry.ask) / 2.0


# ---------------------------------------------------------------------------
# CTraderBarProvider
# ---------------------------------------------------------------------------


class CTraderBarProvider:
    """Real-time M1 bar provider using cTrader native live trendbar stream.

    Implements the :class:`~kinetra.renko.live_trader.BarProvider` interface
    (duck-typed — no explicit import to keep the dependency one-directional).

    How it works
    ------------
    1. On :meth:`start`, the connector is connected (if not already) and a
       ``ProtoOASubscribeLiveTrendbarReq`` (period=M1) + a
       ``ProtoOASubscribeSpotsReq`` are sent for each subscribed symbol.

    2. The connector's ``_on_message`` callback fires ``"ProtoOASpotEvent"``
       push handlers on every price update.  Each ``SpotEvent`` MAY contain
       a ``trendbar`` sub-message carrying the just-completed M1 bar in
       delta-encoded OHLCV format.

    3. We decode the delta encoding (same logic as ``download_ctrader_history.py``),
       scale the prices by ``10^digits``, and call all registered subscriber
       callbacks with the completed bar.

    4. The ``bid`` and ``ask`` fields in every ``SpotEvent`` are used to
       maintain a live spread cache for :meth:`get_spread_pts`.

    Delta encoding
    --------------
    cTrader encodes trendbar prices as::

        low   = trendbar.low / (10^digits)
        open  = low + trendbar.deltaOpen / (10^digits)
        high  = low + trendbar.deltaHigh / (10^digits)
        close = low + trendbar.deltaClose / (10^digits)

    The ``utcTimestampInMinutes`` field gives the bar open time in UTC
    minutes since epoch (multiply by 60 to get Unix seconds).

    Parameters
    ----------
    connector:
        A started and authenticated :class:`CTraderConnector`.
    """

    def __init__(self, connector: "CTraderConnector") -> None:
        if not _CONNECTOR_AVAILABLE:
            raise ImportError(
                "kinetra.connectors.ctrader_connector is required. "
                "Ensure ctrader-open-api is installed: pip install ctrader-open-api"
            )
        self._connector = connector
        self._spot_cache = _SpotCacheRegistry()

        # symbol name → list of bar callbacks
        self._bar_callbacks: Dict[str, List[Callable[..., None]]] = {}
        # symbol name → cTrader symbolId (resolved on subscribe)
        self._symbol_ids: Dict[str, int] = {}
        # reverse map: symbolId → symbol name
        self._id_to_name: Dict[int, str] = {}
        # symbolId -> last emitted M1 epoch-minute (dedupe repeated trendbars)
        self._last_bar_minute: Dict[int, int] = {}
        # symbolId -> last emitted close (sanity checks for scale drift)
        self._last_bar_close: Dict[int, float] = {}
        # symbolId -> preferred price scale (stabilize decode across bars)
        self._price_scale: Dict[int, float] = {}

        self._subscribed_ids: Set[int] = set()
        self._lock = threading.Lock()
        self._started: bool = False
        self._stopped: bool = False
        self._watchdog_stop = threading.Event()
        self._watchdog_thread: Optional[threading.Thread] = None
        self._last_failover_generation: int = int(
            getattr(self._connector, "failover_generation", 0)
        )
        self._last_connected_up: bool = bool(self._connector.is_connected())

        # Pending subscriptions registered before start()
        self._pending_symbols: List[tuple[str, Callable[..., None]]] = []

    # ── BarProvider interface ────────────────────────────────────────────────

    def subscribe(self, symbol: str, callback: Callable[..., None]) -> None:
        """Register *callback* for completed M1 bars on *symbol*.

        May be called before or after :meth:`start`.

        Parameters
        ----------
        symbol:
            Canonical symbol name (e.g. ``"XAUUSD"``).
        callback:
            Called with keyword arguments ``(symbol, open_, high, low,
            close, volume, timestamp)`` for each completed M1 bar.
        """
        symbol = str(symbol).strip().upper()
        with self._lock:
            self._bar_callbacks.setdefault(symbol, []).append(callback)

        if self._started and not self._stopped:
            self._ensure_subscribed(symbol)
        else:
            self._pending_symbols.append((symbol, callback))

    def start(self) -> None:
        """Register push handlers and subscribe all pending symbols.

        The connector must already be started (connected + authenticated)
        before calling this method.

        Raises
        ------
        RuntimeError
            If the connector is not connected.
        """
        if not self._connector.is_connected():
            raise RuntimeError("CTraderConnector must be started before CTraderBarProvider.start()")

        # Register the SpotEvent handler once
        self._connector.add_push_handler("ProtoOASpotEvent", self._on_spot_event)

        self._started = True
        self._stopped = False
        self._watchdog_stop.clear()
        self._last_failover_generation = int(getattr(self._connector, "failover_generation", 0))
        self._last_connected_up = bool(self._connector.is_connected())
        logger.info("[cTrader] BarProvider starting …")

        # Materialise all pending subscriptions
        seen_symbols: Set[str] = set()
        for symbol, _ in self._pending_symbols:
            if symbol not in seen_symbols:
                self._ensure_subscribed(symbol)
                seen_symbols.add(symbol)
        self._pending_symbols.clear()
        self._start_watchdog()

        logger.info(
            "[cTrader] BarProvider ready — subscribed to %d symbols", len(self._subscribed_ids)
        )

    def stop(self) -> None:
        """Unsubscribe from all live trendbar + spot streams."""
        if self._stopped:
            return
        self._stopped = True
        self._watchdog_stop.set()
        if self._watchdog_thread is not None:
            self._watchdog_thread.join(timeout=5.0)
        self._watchdog_thread = None
        logger.info("[cTrader] BarProvider stopping …")

        self._connector.remove_push_handler("ProtoOASpotEvent", self._on_spot_event)

        for symbol_id in list(self._subscribed_ids):
            self._unsubscribe_symbol(symbol_id)
        with self._lock:
            self._last_bar_minute.clear()
            self._last_bar_close.clear()
            self._price_scale.clear()

        logger.info("[cTrader] BarProvider stopped.")

    def _start_watchdog(self) -> None:
        if self._watchdog_thread is not None and self._watchdog_thread.is_alive():
            return
        self._watchdog_thread = threading.Thread(
            target=self._subscription_watchdog,
            name="ctrader_barprovider_watchdog",
            daemon=True,
        )
        self._watchdog_thread.start()

    def _subscription_watchdog(self) -> None:
        """Keep live subscriptions healthy across reconnect/failover transitions."""
        interval_s = max(float(os.getenv("CTRADER_SUB_WATCHDOG_INTERVAL_S", "5.0")), 1.0)
        while not self._watchdog_stop.wait(timeout=interval_s):
            if not self._started or self._stopped:
                continue
            connected_up = bool(self._connector.is_connected())
            generation = int(getattr(self._connector, "failover_generation", 0))
            if not connected_up:
                self._last_connected_up = False
                continue

            must_resubscribe = False
            reason = ""
            if not self._last_connected_up:
                must_resubscribe = True
                reason = "reconnected"
            elif generation != self._last_failover_generation:
                must_resubscribe = True
                reason = f"failover_generation {self._last_failover_generation}->{generation}"

            if must_resubscribe:
                with self._lock:
                    self._subscribed_ids.clear()
                    self._last_bar_minute.clear()
                logger.warning("[cTrader] Refreshing trendbar/spot subscriptions (%s)", reason)

            self._last_connected_up = True
            self._last_failover_generation = generation

            symbols = list(self._bar_callbacks.keys())
            for symbol in symbols:
                try:
                    self._ensure_subscribed(symbol)
                except Exception:
                    logger.exception(
                        "[cTrader] Watchdog failed ensuring subscription for %s",
                        symbol,
                    )

    # ── Spread access (used by CTraderOrderDispatcher) ───────────────────────

    def get_spread_pts(self, symbol: str) -> float:
        """Return the current bid-ask spread in price units (float).

        Falls back to 1.0 if no SpotEvent has been received yet for
        *symbol*.  After the first quote arrives this is always fresh
        because SpotEvents are emitted on every tick.

        Parameters
        ----------
        symbol:
            Canonical symbol name.

        Returns
        -------
        float
            ``ask - bid`` in the same price units as the brick size.
        """
        symbol = str(symbol).strip().upper()
        sid = self._symbol_ids.get(symbol)
        if sid is None:
            return 1.0
        spread = self._spot_cache.get_spread(sid)
        return spread if spread is not None else 1.0

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _ensure_subscribed(self, symbol: str) -> None:
        """Resolve symbol → id and subscribe if not already subscribed."""
        symbol = str(symbol).strip().upper()
        sid = self._connector.find_symbol_id(symbol)
        if sid is None:
            logger.warning("[cTrader] Cannot resolve symbol %r — skipping subscription", symbol)
            return

        with self._lock:
            self._symbol_ids[symbol] = sid
            self._id_to_name[sid] = symbol

        if sid in self._subscribed_ids:
            return

        digits = self._connector.get_digits(sid)
        self._spot_cache.update(sid, 0, 0, digits)  # prime the cache entry

        for attempt in range(1, _SUBSCRIBE_MAX_RETRIES + 1):
            # cTrader requires spot subscription before trendbars.
            ok_spots = self._subscribe_spots(sid)
            # Small settle delay to avoid race where trendbar request arrives
            # before spot subscription is fully active server-side.
            if ok_spots:
                time.sleep(0.2)
            ok_trendbars = self._subscribe_live_trendbars(sid)
            if ok_trendbars and ok_spots:
                with self._lock:
                    self._subscribed_ids.add(sid)
                logger.info(
                    "[cTrader] Subscribed to M1 trendbars + spots for %s (id=%d)", symbol, sid
                )
                return
            logger.warning(
                "[cTrader] Subscription attempt %d/%d failed for %s",
                attempt,
                _SUBSCRIBE_MAX_RETRIES,
                symbol,
            )
            time.sleep(_SUBSCRIBE_RETRY_INTERVAL_S)

        logger.error(
            "[cTrader] Failed to subscribe to %s after %d attempts", symbol, _SUBSCRIBE_MAX_RETRIES
        )

    def _subscribe_live_trendbars(self, symbol_id: int) -> bool:
        """Send ``ProtoOASubscribeLiveTrendbarReq`` for M1 on *symbol_id*."""
        # Access the api_msgs module through the connector's private reference.
        # This is the ONLY way to construct cTrader Protobuf messages without
        # importing ctrader_open_api directly (which would violate §28).
        try:
            api_msgs = _get_api_msgs()
            req = api_msgs.ProtoOASubscribeLiveTrendbarReq()
            req.ctidTraderAccountId = self._connector.credentials.account_id
            req.symbolId = symbol_id
            req.period = _PERIOD_M1

            resp = self._connector.send_and_wait(req)
            if resp is None:
                return False
            if hasattr(resp, "errorCode"):
                code = getattr(resp, "errorCode", "?")
                desc = getattr(resp, "description", "")
                if str(code).upper() == "ALREADY_SUBSCRIBED":
                    logger.info(
                        "[cTrader] SubscribeLiveTrendbar already active for id=%d",
                        symbol_id,
                    )
                    return True
                logger.warning(
                    "[cTrader] SubscribeLiveTrendbar error for id=%d: %s — %s",
                    symbol_id,
                    code,
                    desc,
                )
                return False
            return True
        except Exception:
            logger.exception("[cTrader] _subscribe_live_trendbars raised for id=%d", symbol_id)
            return False

    def _subscribe_spots(self, symbol_id: int) -> bool:
        """Send ``ProtoOASubscribeSpotsReq`` for *symbol_id*.

        Spots provide the live bid/ask used to estimate spread and also
        deliver the ``trendbar`` sub-message when a bar completes.
        """
        try:
            api_msgs = _get_api_msgs()
            req = api_msgs.ProtoOASubscribeSpotsReq()
            req.ctidTraderAccountId = self._connector.credentials.account_id
            req.symbolId.append(symbol_id)

            resp = self._connector.send_and_wait(req)
            if resp is None:
                return False
            if hasattr(resp, "errorCode"):
                code = getattr(resp, "errorCode", "?")
                desc = getattr(resp, "description", "")
                if str(code).upper() == "ALREADY_SUBSCRIBED":
                    logger.info("[cTrader] SubscribeSpots already active for id=%d", symbol_id)
                    return True
                logger.warning(
                    "[cTrader] SubscribeSpots error for id=%d: %s — %s",
                    symbol_id,
                    code,
                    desc,
                )
                return False
            return True
        except Exception:
            logger.exception("[cTrader] _subscribe_spots raised for id=%d", symbol_id)
            return False

    def _unsubscribe_symbol(self, symbol_id: int) -> None:
        """Send unsubscribe requests for live trendbars and spots."""
        try:
            api_msgs = _get_api_msgs()
            account_id = self._connector.credentials.account_id

            unsub_tb = api_msgs.ProtoOAUnsubscribeLiveTrendbarReq()
            unsub_tb.ctidTraderAccountId = account_id
            unsub_tb.symbolId = symbol_id
            unsub_tb.period = _PERIOD_M1
            self._connector.send_and_wait(unsub_tb)

            unsub_spots = api_msgs.ProtoOAUnsubscribeSpotsReq()
            unsub_spots.ctidTraderAccountId = account_id
            unsub_spots.symbolId.append(symbol_id)
            self._connector.send_and_wait(unsub_spots)

        except Exception:
            logger.debug("[cTrader] _unsubscribe_symbol raised for id=%d", symbol_id, exc_info=True)

    def _on_spot_event(self, payload: Any) -> None:
        """Push handler — called on the Twisted reactor thread.

        Extracts the completed M1 trendbar (if present) and the live
        bid/ask spread from a ``ProtoOASpotEvent``.

        This method MUST NOT block — it is called synchronously on the
        reactor thread.
        """
        symbol_id = getattr(payload, "symbolId", None)
        if symbol_id is None:
            return

        # Update spread cache
        spot_mid_event: Optional[float] = None
        bid_raw = getattr(payload, "bid", 0)
        ask_raw = getattr(payload, "ask", 0)
        digits = int(self._connector.get_digits(symbol_id) or 0)
        if digits <= 0 or digits > 10:
            digits = self._spot_cache.get_digits(symbol_id)
        pip_position = max(int(digits) - 1, 0)
        if bid_raw > 0 and ask_raw > 0:
            raw_diff = int(ask_raw) - int(bid_raw)
            spot_scale = _select_spot_scale(
                raw_diff=raw_diff,
                digits=int(digits),
                pip_position=pip_position,
            )
            spot_mid_event = ((int(bid_raw) + int(ask_raw)) / 2.0) / max(spot_scale, 1.0)
            spot_digits = int(round(math.log10(max(spot_scale, 1.0))))
            spot_digits = max(1, min(spot_digits, 10))
            self._spot_cache.update(symbol_id, int(bid_raw), int(ask_raw), spot_digits)

        # Extract completed trendbar if present
        trendbar = getattr(payload, "trendbar", None)
        # The trendbar field is a repeated field; check length > 0
        if trendbar is None or len(trendbar) == 0:
            return

        symbol_name = self._id_to_name.get(symbol_id)
        if symbol_name is None:
            return

        callbacks = self._bar_callbacks.get(symbol_name, [])
        if not callbacks:
            return

        for tb in trendbar:
            # Only process M1 bars
            period = getattr(tb, "period", _PERIOD_M1)
            if period != _PERIOD_M1:
                continue

            # Decode delta encoding
            low_raw = getattr(tb, "low", 0)
            delta_open = getattr(tb, "deltaOpen", 0)
            delta_high = getattr(tb, "deltaHigh", 0)
            delta_close = getattr(tb, "deltaClose", 0)
            volume = getattr(tb, "volume", 0)
            inferred_scale = _select_price_scale(
                low_raw=int(low_raw),
                delta_open=int(delta_open),
                delta_high=int(delta_high),
                delta_close=int(delta_close),
                digits=int(digits),
                pip_position=pip_position,
            )
            fixed_scale = float(10 ** max(int(digits), 1))

            # Bar open time: utcTimestampInMinutes × 60 → Unix seconds
            ts_min = getattr(tb, "utcTimestampInMinutes", 0)
            if ts_min == 0:
                # Fallback: some SDK versions use timestamp (ms)
                ts_ms = getattr(tb, "timestamp", 0)
                ts = (
                    datetime.fromtimestamp(ts_ms / 1000.0, tz=timezone.utc)
                    if ts_ms > 0
                    else datetime.now(tz=timezone.utc)
                )
                ts_min_key = int(ts.timestamp() // 60)
            else:
                ts = datetime.fromtimestamp(ts_min * 60, tz=timezone.utc)
                ts_min_key = int(ts_min)

            # Some brokers re-emit the same completed trendbar on successive spot events.
            # Emit each M1 bar at most once per symbol.
            with self._lock:
                last_min = self._last_bar_minute.get(symbol_id)
                if last_min is not None and ts_min_key <= last_min:
                    continue
                self._last_bar_minute[symbol_id] = ts_min_key

            if low_raw == 0:
                continue

            scale_candidates: List[float] = []
            with self._lock:
                preferred_scale = self._price_scale.get(symbol_id)
                prev_close = self._last_bar_close.get(symbol_id)
            spot_mid = (
                spot_mid_event
                if spot_mid_event is not None
                else self._spot_cache.get_mid(symbol_id)
            )
            for s in (preferred_scale, fixed_scale, inferred_scale, 100000.0):
                if s is None:
                    continue
                sf = float(s)
                if sf <= 0:
                    continue
                if all(abs(sf - x) > 1e-9 for x in scale_candidates):
                    scale_candidates.append(sf)

            chosen: Optional[tuple[float, float, float, float, float]] = None
            best_score = float("inf")
            for scale in scale_candidates:
                o = (low_raw + delta_open) / scale
                h = (low_raw + delta_high) / scale
                l = low_raw / scale
                c = (low_raw + delta_close) / scale
                if not (o > 0 and h > 0 and l > 0 and c > 0):
                    continue
                if h < l:
                    continue
                jump_score = abs(c - prev_close) if prev_close is not None else 0.0
                spot_score = abs(c - spot_mid) if spot_mid is not None else 0.0
                if prev_close is None:
                    # First usable bar: prefer scale closest to live spot-mid.
                    if spot_mid is not None and abs(c - spot_mid) < best_score:
                        best_score = abs(c - spot_mid)
                        chosen = (scale, o, h, l, c)
                    elif spot_mid is None and abs(scale - fixed_scale) < 1e-9:
                        chosen = (scale, o, h, l, c)
                    continue
                # Strongly prefer continuity with previous close, secondarily spot mid.
                score = jump_score + (0.4 * spot_score)
                if score < best_score:
                    best_score = score
                    chosen = (scale, o, h, l, c)

            if chosen is None:
                continue
            scale, o, h, l, c = chosen

            if prev_close is not None:
                # Guardrail: reject anomalous M1 close jumps due to decode/scale mismatch.
                max_jump_abs = max(50.0, abs(prev_close) * 0.20)
                if abs(c - prev_close) > max_jump_abs:
                    logger.error(
                        "[cTrader] Dropping anomalous trendbar for %s (id=%d): close %.5f -> %.5f (jump=%.5f, scale=%g)",
                        symbol_name,
                        symbol_id,
                        prev_close,
                        c,
                        abs(c - prev_close),
                        scale,
                    )
                    continue
            if spot_mid is not None:
                # Spot anchor guardrail: decoded bar close must stay near live mid.
                max_spot_dev = max(25.0, abs(spot_mid) * 0.25)
                if abs(c - spot_mid) > max_spot_dev:
                    logger.error(
                        "[cTrader] Dropping off-anchor trendbar for %s (id=%d): close %.5f vs spot_mid %.5f (dev=%.5f, scale=%g)",
                        symbol_name,
                        symbol_id,
                        c,
                        spot_mid,
                        abs(c - spot_mid),
                        scale,
                    )
                    continue

            with self._lock:
                self._price_scale[symbol_id] = float(scale)
                self._last_bar_close[symbol_id] = float(c)

            for cb in callbacks:
                try:
                    cb(
                        symbol=symbol_name,
                        open_=o,
                        high=h,
                        low=l,
                        close=c,
                        volume=float(volume),
                        timestamp=ts,
                    )
                except Exception:
                    logger.exception("[cTrader] Bar callback raised for %s", symbol_name)


# ---------------------------------------------------------------------------
# CTraderOrderDispatcher
# ---------------------------------------------------------------------------


class CTraderOrderDispatcher:
    """Order dispatcher that submits and closes real trades via cTrader Open API.

    Implements the :class:`~kinetra.renko.live_trader.OrderDispatcher`
    interface (duck-typed — avoids a circular import).

    Order flow
    ----------
    ``open_position()``::

        1. Generate a unique ``clientOrderId`` string.
        2. Register an :class:`_ExecutionWaiter` for ``"ProtoOAExecutionEvent"``
           push messages (before sending — no race condition).
        3. Send ``ProtoOANewOrderReq`` (MARKET type, with stop-loss, no TP).
        4. Block on the waiter for up to ``fill_timeout_s`` seconds.
        5. Parse the ``ProtoOAExecutionEvent`` to extract ``positionId``
           (used as ``order_id`` in :class:`~kinetra.renko.live_trader.LiveTrade`).
        6. Unregister the waiter.

    ``close_position()``::

        1. Send ``ProtoOAClosePositionReq`` with the ``positionId``.
        2. Return success/failure based on whether an error response arrived.
           (Close fill events are also tracked via ExecutionEvent but we
           use a simpler fire-and-check approach for close orders.)

    Volume units
    ------------
    cTrader uses ``volume`` in units of 1/100 of a lot (i.e. 0.01 lot = 100
    volume units).  ``open_position()`` converts the ``lots`` argument:
    ``volume = round(lots * 100) * 100`` (cTrader's minimum unit is 100).

    Parameters
    ----------
    connector:
        A started and authenticated :class:`CTraderConnector`.
    bar_provider:
        The :class:`CTraderBarProvider` sharing the same connector.
        Used for live spread estimation via :meth:`get_spread_pts`.
    fill_timeout_s:
        Seconds to wait for a ``ProtoOAExecutionEvent`` after sending a
        market order.  Defaults to ``_FILL_WAIT_S`` (from env
        ``CTRADER_FILL_WAIT_S``, default 30s).
    """

    def __init__(
        self,
        connector: "CTraderConnector",
        bar_provider: CTraderBarProvider,
        fill_timeout_s: float = _FILL_WAIT_S,
    ) -> None:
        if not _CONNECTOR_AVAILABLE:
            raise ImportError(
                "kinetra.connectors.ctrader_connector is required. "
                "Ensure ctrader-open-api is installed: pip install ctrader-open-api"
            )
        self._connector = connector
        self._bar_provider = bar_provider
        self._fill_timeout_s = fill_timeout_s
        self._order_counter: int = 0
        self._counter_lock = threading.Lock()
        logger.info(
            "[cTrader] OrderDispatcher configured: fill_timeout_s=%.1f equity_timeout_s=%s",
            float(self._fill_timeout_s),
            os.getenv("CTRADER_EQUITY_TIMEOUT_S", "15.0"),
        )

    # ── OrderDispatcher interface ────────────────────────────────────────────

    def open_position(
        self,
        symbol: str,
        direction: Any,  # TradeDirection — not imported to avoid circular dep
        lots: float,
        price: float,
        stop_price: float,
        comment: str = "",
    ) -> Any:  # OrderResult
        """Submit a market order with a hard stop-loss.

        Parameters
        ----------
        symbol:
            Broker symbol name (e.g. ``"XAUUSD"``).
        direction:
            :class:`~kinetra.renko.live_trader.TradeDirection` enum value.
            ``TradeDirection.LONG`` → BUY, ``TradeDirection.SHORT`` → SELL.
        lots:
            Volume in standard lots (e.g. ``0.01``).
        price:
            Expected fill price (informational — market orders fill at market).
        stop_price:
            Hard stop-loss price. For market orders we send
            ``relativeStopLoss`` (broker-compatible on Pepperstone cTrader),
            derived from ``abs(price - stop_price)``.
        comment:
            Optional order comment / label (max 31 chars for cTrader).

        Returns
        -------
        OrderResult
            ``.success=True`` and ``.order_id`` = positionId on fill.
            ``.success=False`` with ``.error`` on rejection or timeout.
        """
        from kinetra.renko.live_trader import OrderResult

        if not self._connector.is_connected():
            return OrderResult(success=False, error="cTrader connector not connected")

        symbol_id = self._bar_provider._symbol_ids.get(symbol)
        if symbol_id is None:
            # Try resolving on the fly
            symbol_id = self._connector.find_symbol_id(symbol)
            if symbol_id is None:
                return OrderResult(
                    success=False,
                    error=f"Cannot resolve symbol {symbol!r} to a cTrader symbolId",
                )
            with self._bar_provider._lock:
                self._bar_provider._symbol_ids[symbol] = symbol_id
                self._bar_provider._id_to_name[symbol_id] = symbol

        digits = self._connector.get_digits(symbol_id)
        is_long = str(direction).endswith("LONG")
        trade_side = _TRADE_SIDE_BUY if is_long else _TRADE_SIDE_SELL

        # cTrader volume: lots × 100 × 100 = lots × 10 000
        # (1 lot = 100 units; 1 unit = 100 cTrader volume units)
        volume = max(int(round(lots * 100)) * 100, 100)

        # Absolute stop-loss in cTrader integer format (price × 10^digits).
        scale = 10**digits
        stop_loss_int = int(round(stop_price * scale))
        # Relative stop-loss distance for market orders. cTrader expects this
        # in 1/100000 price units.
        relative_stop = int(round(abs(float(price) - float(stop_price)) * 100000.0))

        client_order_id = self._next_client_order_id()
        label = (comment[:31] or client_order_id)[:31]

        logger.info(
            "[cTrader] OPEN %s %s %.2f lots (vol=%d) @ ~%.5f  stop=%.5f  label=%s",
            symbol,
            "BUY" if is_long else "SELL",
            lots,
            volume,
            price,
            stop_price,
            label,
        )

        # Register the execution waiter BEFORE sending the order so we
        # cannot miss a fill event that arrives before the RPC returns.
        waiter = _ExecutionWaiter(client_order_id, timeout_s=self._fill_timeout_s)
        self._connector.add_push_handler("ProtoOAExecutionEvent", waiter.handle)

        try:
            api_msgs = _get_api_msgs()
            req = api_msgs.ProtoOANewOrderReq()
            req.ctidTraderAccountId = self._connector.credentials.account_id
            req.symbolId = symbol_id
            req.orderType = _ORDER_TYPE_MARKET
            req.tradeSide = trade_side
            req.volume = volume
            if relative_stop > 0:
                req.relativeStopLoss = relative_stop
            else:
                req.stopLoss = stop_loss_int
            req.label = label
            req.clientOrderId = client_order_id
            if comment:
                req.comment = comment[:31]

            # cTrader does not send a synchronous RPC response to
            # ProtoOANewOrderReq.  The fill is delivered as a push
            # ProtoOAExecutionEvent (no clientMsgId), which the waiter
            # captures.  We use a short 3-second window only to catch
            # immediate ProtoOAErrorRes rejections (e.g. invalid symbol,
            # insufficient margin).  A None return simply means no
            # immediate error — the fill will arrive via the push handler.
            resp = self._connector.send_and_wait(req, timeout_s=3.0)

            # Check for immediate error response (e.g. invalid symbol).
            err = _extract_response_error(resp)
            if err:
                # Fallback: some brokers/accounts may still require absolute
                # stopLoss even on market requests.
                if "INVALID_REQUEST" in err.upper() and "SL/TP in absolute values" in err:
                    req2 = api_msgs.ProtoOANewOrderReq()
                    req2.ctidTraderAccountId = self._connector.credentials.account_id
                    req2.symbolId = symbol_id
                    req2.orderType = _ORDER_TYPE_MARKET
                    req2.tradeSide = trade_side
                    req2.volume = volume
                    req2.stopLoss = stop_loss_int
                    req2.label = label
                    req2.clientOrderId = client_order_id
                    if comment:
                        req2.comment = comment[:31]
                    resp2 = self._connector.send_and_wait(req2, timeout_s=3.0)
                    err2 = _extract_response_error(resp2)
                    if err2:
                        return OrderResult(success=False, error=err2)
                else:
                    return OrderResult(success=False, error=err)

            # Block until the execution event arrives
            exec_event = waiter.wait()
        finally:
            self._connector.remove_push_handler("ProtoOAExecutionEvent", waiter.handle)

        if exec_event is None:
            reconciled = self._reconcile_position_after_timeout(
                symbol=symbol,
                symbol_id=int(symbol_id),
                expected_price=float(price),
                expected_lots=float(lots),
            )
            if reconciled is not None:
                logger.warning(
                    "[cTrader] Fill event timeout but position exists (reconciled): symbol=%s positionId=%s",
                    symbol,
                    reconciled,
                )
                return OrderResult(
                    success=True,
                    order_id=str(reconciled),
                    filled_price=float(price),
                    filled_lots=float(lots),
                    raw={
                        "reconciled_after_timeout": True,
                        "position_id": str(reconciled),
                    },
                )
            logger.error(
                "[cTrader] No fill event received within %.0fs for %s",
                self._fill_timeout_s,
                symbol,
            )
            return OrderResult(
                success=False,
                error=f"Fill timeout after {self._fill_timeout_s:.0f}s",
            )

        # Parse the execution event
        return self._parse_execution_event(exec_event, price, lots, expected_open=True)

    def close_position(
        self,
        symbol: str,
        order_id: str,
        price: float,
        lots: float,
        comment: str = "",
    ) -> Any:  # OrderResult
        """Close an open position by positionId.

        Parameters
        ----------
        symbol:
            Broker symbol (informational — cTrader closes by positionId).
        order_id:
            The ``positionId`` returned by :meth:`open_position`.
        price:
            Expected fill price (for logging).
        lots:
            Volume to close (whole position if lots ≥ open volume).
        comment:
            Optional comment.

        Returns
        -------
        OrderResult
        """
        from kinetra.renko.live_trader import OrderResult

        if not self._connector.is_connected():
            return OrderResult(success=False, error="cTrader connector not connected")

        try:
            position_id = int(order_id)
        except (TypeError, ValueError):
            return OrderResult(success=False, error=f"Invalid positionId: {order_id!r}")

        # cTrader volume in integer units
        volume = max(int(round(lots * 100)) * 100, 100)

        # Generate a clientOrderId for tracking the close execution event
        client_order_id = self._next_client_order_id()

        logger.info(
            "[cTrader] CLOSE %s positionId=%s vol=%d @ ~%.5f  clientOrderId=%s  %s",
            symbol,
            order_id,
            volume,
            price,
            client_order_id,
            comment,
        )

        # Register execution waiter BEFORE sending the request (race-condition safety)
        waiter = _ExecutionWaiter(client_order_id, timeout_s=self._fill_timeout_s)
        self._connector.add_push_handler("ProtoOAExecutionEvent", waiter.handle)

        try:
            api_msgs = _get_api_msgs()
            req = api_msgs.ProtoOAClosePositionReq()
            req.ctidTraderAccountId = self._connector.credentials.account_id
            req.positionId = position_id
            req.volume = volume
            # Some broker feeds include clientOrderId in the execution event for closes
            req.clientOrderId = client_order_id

            # cTrader does not send a synchronous RPC response to
            # ProtoOAClosePositionReq.  The fill is delivered as a push
            # ProtoOAExecutionEvent.  Use a short timeout only to catch
            # immediate ProtoOAErrorRes rejections.
            resp = self._connector.send_and_wait(req, timeout_s=3.0)

            # Check for immediate error response
            err = _extract_response_error(resp)
            if err:
                logger.error("[cTrader] Close immediate error: %s", err)
                return OrderResult(success=False, error=err)

            # Block until the execution event arrives
            exec_event = waiter.wait()
        except Exception as exc:
            logger.exception("[cTrader] close_position raised for positionId=%s", order_id)
            return OrderResult(success=False, error=str(exc))
        finally:
            self._connector.remove_push_handler("ProtoOAExecutionEvent", waiter.handle)

        if exec_event is None:
            logger.warning(
                "[cTrader] No fill event received within %.0fs for close positionId=%s",
                self._fill_timeout_s,
                order_id,
            )
            # Return success anyway — cTrader close requests are fire-and-forget
            # and the position may have closed without an execution event
            return OrderResult(
                success=True,
                order_id=order_id,
                filled_price=price,
                filled_lots=lots,
                raw={"execution_event_timeout": True},
            )

        # Parse the execution event
        return self._parse_execution_event(exec_event, price, lots, expected_open=False)

    def get_spread_pts(self, symbol: str) -> float:
        """Return the current bid-ask spread in price units.

        Delegates to the :class:`CTraderBarProvider`'s live spread cache.
        Falls back to 1.0 if no quote has been received yet.

        Parameters
        ----------
        symbol:
            Canonical symbol name.

        Returns
        -------
        float
            ``ask - bid`` in the same price units as the brick size.
        """
        return self._bar_provider.get_spread_pts(symbol)

    def get_equity(self) -> Optional[float]:
        """Poll current account balance from cTrader.

        Returns the broker-reported balance (float) so the engine can use
        real equity for lot sizing.  Returns ``None`` on any error so the
        caller can safely fall back to its internal simulated equity.
        """
        try:
            timeout_s = max(float(os.getenv("CTRADER_EQUITY_TIMEOUT_S", "15.0")), 5.0)
            snapshot = self._connector.get_account_snapshot(timeout_s=timeout_s)
            return float(snapshot["balance"])
        except Exception as exc:
            logger.warning("[cTrader] get_equity failed: %s", exc)
            return None

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _next_client_order_id(self) -> str:
        """Generate a unique client order ID string."""
        with self._counter_lock:
            self._order_counter += 1
            return f"RENKO-{self._order_counter:08d}"

    def _reconcile_position_after_timeout(
        self,
        symbol: str,
        symbol_id: int,
        expected_price: float,
        expected_lots: float,
    ) -> Optional[str]:
        """Best-effort fallback: inspect open positions when fill event times out."""
        try:
            api_msgs = _get_api_msgs()
            req = api_msgs.ProtoOAReconcileReq()
            req.ctidTraderAccountId = self._connector.credentials.account_id
            rec = self._connector.send_and_wait(req, timeout_s=max(self._fill_timeout_s, 10.0))
            if rec is None or hasattr(rec, "errorCode"):
                return None

            digits = int(self._connector.get_digits(symbol_id) or 2)
            scale = float(10 ** max(digits, 0))
            best_id: Optional[str] = None
            best_score = float("inf")

            for pos in getattr(rec, "position", []):
                sid = int(getattr(pos, "symbolId", 0) or 0)
                if sid != symbol_id:
                    continue
                pid = str(getattr(pos, "positionId", "") or "")
                if not pid:
                    continue

                trade_data = getattr(pos, "tradeData", None)
                raw_open = int(getattr(trade_data, "openPrice", 0) or 0) if trade_data else 0
                open_price = (raw_open / scale) if raw_open > 0 else float(expected_price)
                volume_units = int(getattr(pos, "volume", 0) or 0)
                lots = float(volume_units) / 10000.0 if volume_units > 0 else float(expected_lots)

                score = abs(open_price - float(expected_price)) + (
                    0.2 * abs(lots - float(expected_lots))
                )
                if score < best_score:
                    best_score = score
                    best_id = pid
            return best_id
        except Exception:
            logger.exception(
                "[cTrader] reconcile fallback failed after fill timeout for %s",
                symbol,
            )
            return None

    def _parse_execution_event(
        self,
        payload: Any,
        filled_price: float,
        filled_lots: float,
        *,
        expected_open: bool = True,
    ) -> Any:
        """Convert a ``ProtoOAExecutionEvent`` to an :class:`OrderResult`.

        Parameters
        ----------
        payload:
            The decoded ``ProtoOAExecutionEvent`` Protobuf message.
        filled_price:
            Fallback price if the event does not carry ``executionPrice``.
        filled_lots:
            Fallback lot size.
        expected_open:
            ``True`` if we expect a position-open execution type;
            ``False`` for a close.

        Returns
        -------
        OrderResult
        """
        from kinetra.renko.live_trader import OrderResult

        exec_type = getattr(payload, "executionType", -1)
        error_code = getattr(payload, "errorCode", None)

        if error_code:
            desc = getattr(payload, "description", "")
            logger.error("[cTrader] Execution rejected: %s — %s", error_code, desc)
            return OrderResult(success=False, error=f"{error_code}: {desc}")

        # Extract positionId and executionPrice from the nested order/position
        position_id: Optional[str] = None
        actual_price = filled_price
        client_order_id = ""
        order_id = ""

        position = getattr(payload, "position", None)
        if position is not None:
            position_id = str(getattr(position, "positionId", ""))
            trade_data = getattr(position, "tradeData", None)
            if trade_data is not None:
                open_price = getattr(trade_data, "openPrice", 0)
                if open_price > 0:
                    # openPrice in position.tradeData is stored as raw int
                    # scaled by 10^digits — we use the fallback float here
                    # since we don't have easy access to digits at this point.
                    # The filled_price passed in is the price at signal time,
                    # which is close enough for logging and PnL tracking.
                    actual_price = filled_price

        order = getattr(payload, "order", None)
        if order is not None and position_id is None:
            position_id = str(getattr(order, "positionId", ""))
            exec_price = getattr(order, "executionPrice", 0.0)
            if exec_price > 0:
                actual_price = float(exec_price)
        if order is not None:
            client_order_id = str(getattr(order, "clientOrderId", "") or "")
            order_id = str(getattr(order, "orderId", "") or "")

        if not position_id:
            position_id = f"UNKNOWN-{int(time.time())}"
            logger.warning("[cTrader] Could not extract positionId from ExecutionEvent")

        logger.info(
            "[cTrader] Execution confirmed: type=%d positionId=%s price=%.5f",
            exec_type,
            position_id,
            actual_price,
        )

        return OrderResult(
            success=True,
            order_id=position_id,
            filled_price=actual_price,
            filled_lots=filled_lots,
            raw={
                "execution_type": int(exec_type),
                "position_id": position_id,
                "client_order_id": client_order_id,
                "broker_order_id": order_id,
            },
        )


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


class ConnectionWatchdog:
    """Daemon thread that keeps the cTrader session alive and detects drops.

    Responsibilities
    ----------------
    1. **Client-initiated heartbeat** — sends ``ProtoHeartbeatEvent`` every
       ``heartbeat_s`` seconds (default 25 s) to prevent the broker from
       closing the idle TCP connection (typical broker idle timeout: ~60 s).
    2. **Disconnect detection** — polls ``connector.is_connected()`` every
       ``check_s`` seconds.  If the check fails ``max_missed`` times in a row
       the watchdog sets *stop_event* so the engine returns cleanly, and sets
       ``self.lost = True`` so the caller can trigger a reconnect.

    Usage
    -----
    ::

        watchdog = ConnectionWatchdog(connector, stop_event)
        watchdog.start()
        try:
            engine.run(bar_provider, dispatcher, stop_event=stop_event)
        finally:
            watchdog.stop()

        if watchdog.lost:
            # attempt reconnect / restart
            ...

    Parameters
    ----------
    connector:
        An authenticated :class:`~kinetra.connectors.ctrader_connector.CTraderConnector`.
    stop_event:
        ``threading.Event`` shared with the engine — set this to stop the
        trading loop cleanly.
    heartbeat_s:
        Seconds between client-initiated heartbeat messages.
    check_s:
        Seconds between ``is_connected()`` polls.
    max_missed:
        Consecutive failed checks before declaring the connection lost.
    """

    def __init__(
        self,
        connector: Any,
        stop_event: threading.Event,
        heartbeat_s: float = 25.0,
        check_s: float = 10.0,
        max_missed: int = 3,
    ) -> None:
        self._connector = connector
        self._stop_event = stop_event
        self._heartbeat_s = heartbeat_s
        self._check_s = check_s
        self._max_missed = max_missed
        self._own_stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True, name="ctrader-watchdog")
        self.lost: bool = False
        self._missed: int = 0
        self._last_hb: float = 0.0

    def start(self) -> None:
        self._last_hb = time.monotonic()
        self._thread.start()
        logger.debug(
            "[watchdog] Started — heartbeat=%.0fs check=%.0fs max_missed=%d",
            self._heartbeat_s,
            self._check_s,
            self._max_missed,
        )

    def stop(self) -> None:
        self._own_stop.set()
        self._thread.join(timeout=self._check_s + 2)

    def _run(self) -> None:
        while not self._own_stop.wait(self._check_s):
            now = time.monotonic()

            # ── Heartbeat ────────────────────────────────────────────────────
            if now - self._last_hb >= self._heartbeat_s:
                try:
                    self._connector.send_heartbeat()
                    logger.debug("[watchdog] Heartbeat sent")
                except Exception as exc:
                    logger.warning("[watchdog] Heartbeat error: %s", exc)
                self._last_hb = now

            # ── Connection check ──────────────────────────────────────────────
            try:
                ok = self._connector.is_connected()
            except Exception:
                ok = False

            if ok:
                if self._missed:
                    logger.info("[watchdog] Connection restored")
                    self._missed = 0
            else:
                self._missed += 1
                logger.warning(
                    "[watchdog] Connection check failed (%d/%d)",
                    self._missed,
                    self._max_missed,
                )
                if self._missed >= self._max_missed:
                    logger.error("[watchdog] Connection lost — signalling engine stop")
                    self.lost = True
                    self._stop_event.set()
                    return


def _get_api_msgs() -> Any:
    """Return the ``OpenApiMessages_pb2`` module via the connector's import.

    This is the canonical way to construct cTrader Protobuf messages from
    within ``kinetra/renko/`` without importing ``ctrader_open_api`` directly
    (which would violate the §28 broker-isolation rule).

    The connector already imported the module at ``kinetra/connectors/
    ctrader_connector.py``; we access it via the module-level reference
    that the connector exposes through its private ``_api_msgs`` attribute.
    """
    try:
        # Access the already-imported module via the connector's module globals
        import kinetra.connectors.ctrader_connector as _mod

        return _mod._api_msgs
    except (ImportError, AttributeError) as exc:
        raise ImportError(
            "ctrader_open_api is required for cTrader live trading. "
            "Install it with: pip install ctrader-open-api"
        ) from exc


def build_ctrader_session(
    credentials: Optional["CTraderCredentials"] = None,
    connect_timeout_s: float = 30.0,
    fill_timeout_s: float = _FILL_WAIT_S,
) -> "tuple[CTraderOrderDispatcher, CTraderBarProvider]":
    """Convenience factory: connect and return a ready-to-use dispatcher pair.

    Parameters
    ----------
    credentials:
        Optional explicit :class:`~kinetra.connectors.ctrader_connector.CTraderCredentials`.
        If ``None``, credentials are loaded from environment variables /
        ``.env.openapi`` via :meth:`~kinetra.connectors.ctrader_connector.CTraderCredentials.from_env`.
    connect_timeout_s:
        Seconds to wait for connection + app + account authentication.
    fill_timeout_s:
        Seconds to wait for fill execution events in
        :meth:`CTraderOrderDispatcher.open_position`.

    Returns
    -------
    (CTraderOrderDispatcher, CTraderBarProvider)
        Both share the same :class:`~kinetra.connectors.ctrader_connector.CTraderConnector`
        session.  Call ``bar_provider.start()`` to begin receiving live bars.

    Raises
    ------
    ImportError
        If ``ctrader_open_api`` / ``ctrader_connector`` is not available.
    RuntimeError
        If the connector fails to connect within *connect_timeout_s*.
    ValueError
        If credentials cannot be resolved from any source.
    """
    if not _CONNECTOR_AVAILABLE:
        raise ImportError(
            "kinetra.connectors.ctrader_connector is required. "
            "Ensure ctrader-open-api is installed: pip install ctrader-open-api"
        )

    connector = build_connector(
        credentials=credentials,
        timeout_s=connect_timeout_s,
    )

    bar_provider = CTraderBarProvider(connector)
    dispatcher = CTraderOrderDispatcher(
        connector=connector,
        bar_provider=bar_provider,
        fill_timeout_s=fill_timeout_s,
    )

    return dispatcher, bar_provider
