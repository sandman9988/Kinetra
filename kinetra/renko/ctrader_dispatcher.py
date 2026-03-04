"""
cTrader Live Dispatcher - NATIVE REDESIGN
==========================================

COMPLETE REWRITE - Position-centric design to fix execution issues.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Set, Tuple, Union

if TYPE_CHECKING:
    from kinetra.renko.live_trader import OrderDispatcher

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Constants from cTrader Open API proto
# -----------------------------------------------------------------------------

_EXEC_TYPE_ORDER_ACCEPTED = 2
_EXEC_TYPE_ORDER_FILLED = 3
_EXEC_TYPE_ORDER_PARTIAL_FILL = 11
_EXEC_TYPE_ORDER_REJECTED = 7
_EXEC_TYPE_ORDER_CANCELLED = 5
_EXEC_TYPE_ORDER_EXPIRED = 6

_FILL_EXEC_TYPES: Set[int] = {_EXEC_TYPE_ORDER_FILLED, _EXEC_TYPE_ORDER_PARTIAL_FILL}

_ORDER_TYPE_MARKET = 1
_TRADE_SIDE_BUY = 1
_TRADE_SIDE_SELL = 2

# -----------------------------------------------------------------------------
# Operation Tracking (Simple, reliable)
# -----------------------------------------------------------------------------


@dataclass
class PendingOperation:
    """Tracks a single operation awaiting execution event."""

    operation_id: str
    operation_type: str  # "open" or "close"
    symbol: str
    client_order_id: Optional[str] = None
    position_id: Optional[str] = None
    event: threading.Event = field(default_factory=threading.Event)
    result_payload: Optional[Any] = None
    success: bool = False
    error: Optional[str] = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class OperationRegistry:
    """Thread-safe registry of pending operations with multiple lookup indexes."""

    def __init__(self):
        self._ops: Dict[str, PendingOperation] = {}  # operation_id -> op
        self._by_client_id: Dict[str, str] = {}  # client_order_id -> operation_id
        self._by_position_id: Dict[str, str] = {}  # position_id -> operation_id
        self._lock = threading.Lock()

    def register(self, op: PendingOperation) -> None:
        """Register a new pending operation."""
        with self._lock:
            self._ops[op.operation_id] = op
            if op.client_order_id:
                self._by_client_id[op.client_order_id] = op.operation_id
            if op.position_id:
                self._by_position_id[op.position_id] = op.operation_id
            logger.debug(
                "[REGISTRY] Registered %s | op=%s clientOid=%s posId=%s",
                op.operation_type,
                op.operation_id,
                op.client_order_id or "-",
                op.position_id or "-",
            )

    def find(
        self,
        operation_id: Optional[str] = None,
        client_order_id: Optional[str] = None,
        position_id: Optional[str] = None,
    ) -> Optional[PendingOperation]:
        """Find operation by any identifier."""
        with self._lock:
            if operation_id and operation_id in self._ops:
                return self._ops[operation_id]
            if client_order_id and client_order_id in self._by_client_id:
                return self._ops[self._by_client_id[client_order_id]]
            if position_id and position_id in self._by_position_id:
                return self._ops[self._by_position_id[position_id]]
            return None

    def update_ids(
        self,
        op: PendingOperation,
        client_order_id: Optional[str] = None,
        position_id: Optional[str] = None,
    ) -> None:
        """Update operation with newly learned identifiers."""
        with self._lock:
            if client_order_id and client_order_id not in self._by_client_id:
                op.client_order_id = client_order_id
                self._by_client_id[client_order_id] = op.operation_id
            if position_id and position_id not in self._by_position_id:
                op.position_id = position_id
                self._by_position_id[position_id] = op.operation_id

    def complete(
        self,
        op: PendingOperation,
        success: bool,
        payload: Optional[Any] = None,
        error: Optional[str] = None,
    ) -> None:
        """Mark operation complete and signal waiter."""
        op.success = success
        op.result_payload = payload
        op.error = error
        op.event.set()
        # Clean up after delay
        threading.Timer(10.0, self._cleanup, args=[op.operation_id]).start()

    def _cleanup(self, operation_id: str) -> None:
        """Remove completed operation."""
        with self._lock:
            if operation_id not in self._ops:
                return
            op = self._ops[operation_id]
            del self._ops[operation_id]
            if op.client_order_id and op.client_order_id in self._by_client_id:
                if self._by_client_id[op.client_order_id] == operation_id:
                    del self._by_client_id[op.client_order_id]
            if op.position_id and op.position_id in self._by_position_id:
                if self._by_position_id[op.position_id] == operation_id:
                    del self._by_position_id[op.position_id]


# -----------------------------------------------------------------------------
# Execution Event Handler
# -----------------------------------------------------------------------------


class ExecutionEventHandler:
    """Handles ProtoOAExecutionEvent and routes to pending operations."""

    def __init__(self, registry: OperationRegistry):
        self._registry = registry
        self._shutdown = False

    def __call__(self, payload: Any) -> None:
        """Called by connector on reactor thread."""
        if self._shutdown:
            return
        try:
            self._handle(payload)
        except Exception as exc:
            logger.exception("[EXEC] Error: %s", exc)

    def _handle(self, payload: Any) -> None:
        exec_type = int(getattr(payload, "executionType", -1))

        # Extract identifiers from all possible sources
        ids = self._extract_ids(payload)

        logger.debug(
            "[EXEC] Event type=%s clientOid=%s orderId=%s positionId=%s",
            exec_type,
            ids.get("client_order_id") or "-",
            ids.get("order_id") or "-",
            ids.get("position_id") or "-",
        )

        # Find matching operation
        op = self._registry.find(
            client_order_id=ids.get("client_order_id"),
            position_id=ids.get("position_id"),
        )

        if op is None:
            logger.debug("[EXEC] No matching operation")
            return

        # Update with any new IDs we learned
        self._registry.update_ids(
            op,
            client_order_id=ids.get("client_order_id"),
            position_id=ids.get("position_id"),
        )

        # ORDER_ACCEPTED: Learn IDs but don't complete
        if exec_type == _EXEC_TYPE_ORDER_ACCEPTED:
            logger.debug("[EXEC] ORDER_ACCEPTED for op=%s", op.operation_id)
            return

        # Rejection
        if exec_type == _EXEC_TYPE_ORDER_REJECTED:
            logger.warning("[EXEC] ORDER_REJECTED for op=%s", op.operation_id)
            self._registry.complete(op, success=False, error="Order rejected")
            return

        # Fill
        if exec_type in _FILL_EXEC_TYPES:
            logger.info(
                "[EXEC] FILL op=%s type=%s positionId=%s",
                op.operation_id,
                op.operation_type,
                ids.get("position_id"),
            )
            self._registry.complete(op, success=True, payload=payload)
            return

    def _extract_ids(self, payload: Any) -> Dict[str, Optional[str]]:
        """Extract all identifiers from execution event."""
        result: Dict[str, Optional[str]] = {
            "client_order_id": None,
            "order_id": None,
            "position_id": None,
        }

        order = getattr(payload, "order", None)
        if order is not None:
            result["client_order_id"] = str(getattr(order, "clientOrderId", "") or "")
            result["order_id"] = str(getattr(order, "orderId", "") or "")
            result["position_id"] = str(getattr(order, "positionId", "") or "")

            if not result["client_order_id"]:
                trade_data = getattr(order, "tradeData", None)
                if trade_data:
                    result["client_order_id"] = str(getattr(trade_data, "label", "") or "")

        position = getattr(payload, "position", None)
        if position is not None:
            pos_id = str(getattr(position, "positionId", "") or "")
            if pos_id:
                result["position_id"] = pos_id

        deal = getattr(payload, "deal", None)
        if deal is not None:
            deal_pos_id = str(getattr(deal, "positionId", "") or "")
            if deal_pos_id:
                result["position_id"] = deal_pos_id
            deal_order_id = str(getattr(deal, "orderId", "") or "")
            if deal_order_id:
                result["order_id"] = deal_order_id

        return result


# -----------------------------------------------------------------------------
# Imports and Setup
# -----------------------------------------------------------------------------

try:
    from ctrader_open_api.messages.OpenApiMessages_pb2 import (
        ProtoOAAmendPositionSLTPReq,
        ProtoOAClosePositionReq,
        ProtoOAExecutionEvent,
        ProtoOANewOrderReq,
        ProtoOASubscribeDepthQuotesReq,
        ProtoOASubscribeLiveTrendbarReq,
        ProtoOASubscribeSpotsReq,
        ProtoOAUnsubscribeDepthQuotesReq,
        ProtoOAUnsubscribeLiveTrendbarReq,
        ProtoOAUnsubscribeSpotsReq,
    )
    from ctrader_open_api.messages.OpenApiModelMessages_pb2 import (
        ProtoOATrendbar,
        ProtoOATrendbarPeriod,
    )
    from twisted.internet import reactor

    _CTRADER_AVAILABLE = True
except ImportError:
    _CTRADER_AVAILABLE = False
    logger.warning("ctrader_open_api not available - CTraderDispatcher will not function")


def _get_api_msgs():
    """Lazy import to avoid loading protobuf at module import time."""
    from ctrader_open_api.messages import OpenApiMessages_pb2 as api_msgs

    return api_msgs


def _get_model_msgs():
    """Lazy import for model messages."""
    from ctrader_open_api.messages import OpenApiModelMessages_pb2 as model_msgs

    return model_msgs


# -----------------------------------------------------------------------------
# Bar Provider - Build M1 from Ticks
# -----------------------------------------------------------------------------


@dataclass
class _SpotCache:
    bid: float = 0.0
    ask: float = 0.0
    spread: float = 0.0
    digits: int = 5
    updated_at: Optional[datetime] = None


class RenkoBrickBuilder:
    """
    Build Renko bricks directly from tick prices.

    Handles:
    - Gap filling (multiple bricks from single tick)
    - Proper reversal logic (2× brick size for color change)
    - Continuation logic (1× brick size for same color)
    """

    def __init__(self, brick_size: float, symbol_id: int = 0, digits: int = 5):
        self.brick_size = float(brick_size)
        self.symbol_id = symbol_id
        self.digits = digits
        self.last_brick_close: Optional[float] = None
        self.direction: Optional[int] = None  # 1 for Up, -1 for Down
        self.bricks: List[Dict[str, Any]] = []
        self._lock = threading.Lock()

    def process_tick(self, price: float, timestamp: datetime) -> List[Dict[str, Any]]:
        """Process a tick and return any new bricks formed."""
        with self._lock:
            # Initialize first brick
            if self.last_brick_close is None:
                self.last_brick_close = price
                return []

            new_bricks = []

            while True:
                # Reversal thresholds require 2x brick size
                up_reversal = self.last_brick_close + (2 * self.brick_size)
                down_reversal = self.last_brick_close - (2 * self.brick_size)

                # Continuation thresholds require 1x brick size
                up_continuation = self.last_brick_close + self.brick_size
                down_continuation = self.last_brick_close - self.brick_size

                if self.direction is None:
                    # First trend establishment
                    if price >= up_continuation:
                        self._create_brick(up_continuation, 1, timestamp, new_bricks)
                    elif price <= down_continuation:
                        self._create_brick(down_continuation, -1, timestamp, new_bricks)
                    else:
                        break

                elif self.direction == 1:  # Currently UP
                    if price >= up_continuation:
                        self._create_brick(up_continuation, 1, timestamp, new_bricks)
                    elif price <= down_reversal:
                        # Reversal: new brick opens at last close, closes 1 brick down
                        self._create_brick(
                            self.last_brick_close - self.brick_size, -1, timestamp, new_bricks
                        )
                    else:
                        break

                elif self.direction == -1:  # Currently DOWN
                    if price <= down_continuation:
                        self._create_brick(down_continuation, -1, timestamp, new_bricks)
                    elif price >= up_reversal:
                        # Reversal: new brick opens at last close, closes 1 brick up
                        self._create_brick(
                            self.last_brick_close + self.brick_size, 1, timestamp, new_bricks
                        )
                    else:
                        break

            return new_bricks

    def _create_brick(
        self,
        close_price: float,
        direction: int,
        timestamp: datetime,
        list_to_append: List[Dict[str, Any]],
    ) -> None:
        """Create a new brick and update state."""
        brick = {
            "open": self.last_brick_close,
            "close": close_price,
            "direction": direction,
            "timestamp": timestamp,
            "symbol_id": self.symbol_id,
        }
        self.last_brick_close = close_price
        self.direction = direction
        list_to_append.append(brick)
        self.bricks.append(brick)


class _SpotCacheRegistry:
    def __init__(self) -> None:
        self._cache: Dict[int, _SpotCache] = {}
        self._lock = threading.Lock()

    def update(self, symbol_id: int, bid_raw: int, ask_raw: int, digits: int) -> None:
        safe_digits = max(1, min(int(digits), 10))
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
        with self._lock:
            entry = self._cache.get(symbol_id)
            return entry.spread if entry is not None else None

    def get_digits(self, symbol_id: int) -> int:
        with self._lock:
            entry = self._cache.get(symbol_id)
            return max(1, min(int(entry.digits if entry else 5), 10))

    def get_mid(self, symbol_id: int) -> Optional[float]:
        with self._lock:
            entry = self._cache.get(symbol_id)
            if entry is None or entry.bid <= 0 or entry.ask <= 0:
                return None
            return (entry.bid + entry.ask) / 2.0


class CTraderBarProvider:
    """Real-time M1 bar provider using cTrader spot events to build bars."""

    def __init__(self, connector: Any) -> None:
        if not _CTRADER_AVAILABLE:
            raise ImportError("ctrader-open-api is required")
        self._connector = connector
        self._spot_cache = _SpotCacheRegistry()
        self._bar_callbacks: Dict[str, List[Callable[..., None]]] = {}
        self._symbol_ids: Dict[str, int] = {}
        self._id_to_name: Dict[int, str] = {}
        self._last_bar_minute: Dict[int, int] = {}
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
        self._pending_symbols: List[tuple[str, Callable[..., None]]] = []

        # M1 bar accumulation from ticks
        self._current_bar: Dict[int, Dict[str, Any]] = {}  # symbol_id -> OHLCV

        # Direct tick callbacks — receive every spot price without M1 delay.
        # The Renko engine uses this path instead of waiting for minute boundaries.
        self._tick_callbacks: Dict[str, List[Callable[..., None]]] = {}
        self._ticks_fired: int = 0  # total ticks delivered to tick callbacks

    def subscribe(self, symbol: str, callback: Callable[..., None]) -> None:
        if self._started:
            self._do_subscribe(symbol, callback)
        else:
            self._pending_symbols.append((symbol, callback))
        if symbol not in self._bar_callbacks:
            self._bar_callbacks[symbol] = []
        if callback not in self._bar_callbacks[symbol]:
            self._bar_callbacks[symbol].append(callback)

    def subscribe_ticks(self, symbol: str, callback: Callable[..., None]) -> None:
        """Subscribe to raw spot-tick prices for direct Renko construction.

        Calls ``callback(symbol, price, timestamp)`` on every spot tick,
        bypassing the 60-second M1 bar accumulation window.  This matches
        the reference ``ProRenkoSystem.process_tick()`` design: Renko bricks
        are formed the instant a price threshold is crossed, not at the next
        minute boundary.

        The underlying ``ProtoOASubscribeSpotsReq`` subscription is shared
        with the M1 accumulation path — no extra network traffic is added.
        """
        if symbol not in self._tick_callbacks:
            self._tick_callbacks[symbol] = []
        if callback not in self._tick_callbacks[symbol]:
            self._tick_callbacks[symbol].append(callback)
        # Ensure the spot subscription is active.  _do_subscribe() is
        # idempotent: if the symbol_id is already registered it returns early.
        if self._started:
            with self._lock:
                already = symbol in self._symbol_ids
            if not already:
                self._do_subscribe(symbol, lambda **_: None)
        else:
            # Piggyback on the pending list so start() picks it up.
            if not any(s == symbol for s, _ in self._pending_symbols):
                self._pending_symbols.append((symbol, lambda **_: None))

    @property
    def ticks_fired(self) -> int:
        """Total ticks delivered to tick callbacks (diagnostic counter)."""
        return self._ticks_fired

    def _do_subscribe(self, symbol: str, callback: Callable[..., None]) -> None:
        try:
            symbol_id = self._connector.find_symbol_id(symbol)
        except Exception as exc:
            logger.error("[BarProvider] Cannot resolve %s: %s", symbol, exc)
            return
        if symbol_id is None:
            logger.error("[BarProvider] Symbol not found: %s", symbol)
            return
        with self._lock:
            self._symbol_ids[symbol] = symbol_id
            self._id_to_name[symbol_id] = symbol
        digits = self._connector.get_digits(symbol_id)
        self._subscribe_spot(symbol_id)
        with self._lock:
            self._subscribed_ids.add(symbol_id)
        logger.info("[BarProvider] Subscribed %s (id=%d)", symbol, symbol_id)

    def _subscribe_spot(self, symbol_id: int) -> None:
        api_msgs = _get_api_msgs()
        req = api_msgs.ProtoOASubscribeSpotsReq()
        req.ctidTraderAccountId = self._connector.credentials.account_id
        req.symbolId.append(symbol_id)
        try:
            resp = self._connector.send_and_wait(req, timeout_s=5.0)
            if resp is None:
                # None means the request timed out — spot events will NOT arrive.
                logger.error(
                    "[BarProvider] Spot subscribe timed out for symbol_id=%s"
                    " — no ticks will be received",
                    symbol_id,
                )
                return
            err = getattr(resp, "errorCode", None)
            if err:
                logger.error("[BarProvider] Spot subscribe error: %s", err)
            else:
                logger.info("[BarProvider] Spot subscription confirmed symbol_id=%s", symbol_id)
        except Exception as exc:
            logger.error("[BarProvider] Spot subscribe failed: %s", exc)

    def start(self) -> None:
        if self._started:
            return
        if hasattr(self._connector, "start"):
            if not self._connector.is_connected():
                self._connector.start()
        else:
            logger.warning(
                "[BarProvider] Connector has no start method, assuming already connected"
            )
        for symbol, cb in self._pending_symbols:
            self._do_subscribe(symbol, cb)
        self._pending_symbols.clear()
        logger.info("[BarProvider] Registering ProtoOASpotEvent handler...")
        self._connector.add_push_handler("ProtoOASpotEvent", self._on_spot_event)
        self._started = True
        self._watchdog_stop.clear()
        self._watchdog_thread = threading.Thread(target=self._watchdog, daemon=True)
        self._watchdog_thread.start()
        logger.info("[BarProvider] Started and listening for spot events")

    def stop(self) -> None:
        if self._stopped:
            return
        self._stopped = True
        self._watchdog_stop.set()
        if self._watchdog_thread:
            self._watchdog_thread.join(timeout=2.0)
        try:
            self._connector.remove_push_handler("ProtoOASpotEvent", self._on_spot_event)
        except Exception:
            pass
        self._unsubscribe_all()
        logger.info("[BarProvider] Stopped")

    def _unsubscribe_all(self) -> None:
        if not self._subscribed_ids:
            return
        api_msgs = _get_api_msgs()
        for symbol_id in list(self._subscribed_ids):
            # Spots only — we never subscribe to live trendbars.
            try:
                req = api_msgs.ProtoOAUnsubscribeSpotsReq()
                req.ctidTraderAccountId = self._connector.credentials.account_id
                req.symbolId.append(symbol_id)
                self._connector.send_and_wait(req, timeout_s=3.0)
            except Exception:
                pass

    def get_spread_pts(self, symbol: str) -> float:
        with self._lock:
            symbol_id = self._symbol_ids.get(symbol)
        if symbol_id is None:
            return 1.0
        spread = self._spot_cache.get_spread(symbol_id)
        return spread if spread is not None else 1.0

    def _on_spot_event(self, payload: Any) -> None:
        """Process spot event and build M1 bars from ticks."""
        try:
            symbol_id = int(getattr(payload, "symbolId", 0))
            bid_raw = int(getattr(payload, "bid", 0))
            ask_raw = int(getattr(payload, "ask", 0))
            timestamp = int(getattr(payload, "timestamp", 0))  # Unix timestamp in ms

            if symbol_id == 0 or (bid_raw == 0 and ask_raw == 0):
                return

            digits = self._spot_cache.get_digits(symbol_id)
            self._spot_cache.update(symbol_id, bid_raw, ask_raw, digits)

            # Calculate mid price
            scale = 10**digits
            if bid_raw > 0 and ask_raw > 0:
                mid = (bid_raw + ask_raw) / (2 * scale)
            elif bid_raw > 0:
                mid = bid_raw / scale
            elif ask_raw > 0:
                mid = ask_raw / scale
            else:
                return

            # Build M1 bar from tick
            self._process_tick(symbol_id, mid, timestamp)

        except Exception as exc:
            logger.error("[BarProvider] Spot event error: %s", exc)

    def _process_tick(self, symbol_id: int, price: float, timestamp_ms: int) -> None:
        """Fire tick callbacks or accumulate ticks into M1 bars (mutually exclusive)."""
        symbol = self._id_to_name.get(symbol_id, "UNKNOWN")
        tick_cbs = self._tick_callbacks.get(symbol, [])
        if tick_cbs:
            tick_ts = (
                datetime.utcfromtimestamp(timestamp_ms / 1000.0).replace(tzinfo=timezone.utc)
                if timestamp_ms > 0
                else datetime.now(timezone.utc)
            )
            self._ticks_fired += 1
            for cb in tick_cbs:
                try:
                    cb(symbol=symbol, price=price, timestamp=tick_ts)
                except Exception as exc:
                    logger.error("[BarProvider] Tick callback error: %s", exc)
            return  # skip M1 accumulation when tick subscribers are registered

        # ── M1 accumulation (used when no tick subscribers) ──────────────────
        # Get current minute (UTC)
        if timestamp_ms > 0:
            current_minute = timestamp_ms // 60000
        else:
            current_minute = int(datetime.now(timezone.utc).timestamp()) // 60

        with self._lock:
            if symbol_id not in self._current_bar:
                # First tick - initialize
                self._current_bar[symbol_id] = {
                    "open": price,
                    "high": price,
                    "low": price,
                    "close": price,
                    "volume": 1,
                    "minute": current_minute,
                }
                return

            bar = self._current_bar[symbol_id]

            if current_minute != bar["minute"]:
                # Minute changed - emit completed bar
                self._emit_m1_bar(symbol_id, bar)
                # Start new bar
                self._current_bar[symbol_id] = {
                    "open": price,
                    "high": price,
                    "low": price,
                    "close": price,
                    "volume": 1,
                    "minute": current_minute,
                }
            else:
                # Same minute - update OHLC
                bar["high"] = max(bar["high"], price)
                bar["low"] = min(bar["low"], price)
                bar["close"] = price
                bar["volume"] += 1

    def _emit_m1_bar(self, symbol_id: int, bar: Dict[str, Any]) -> None:
        """Emit completed M1 bar to subscribers."""
        with self._lock:
            symbol = self._id_to_name.get(symbol_id, "UNKNOWN")
            last_emitted = self._last_bar_minute.get(symbol_id, 0)

            if bar["minute"] <= last_emitted:
                return  # Already emitted

            self._last_bar_minute[symbol_id] = bar["minute"]

        timestamp = datetime.utcfromtimestamp(bar["minute"] * 60).replace(tzinfo=timezone.utc)

        callbacks = self._bar_callbacks.get(symbol, [])
        for cb in callbacks:
            try:
                cb(
                    symbol=symbol,
                    close=bar["close"],
                    timestamp=timestamp,
                    open=bar["open"],
                    high=bar["high"],
                    low=bar["low"],
                )
            except Exception as exc:
                logger.error("[BarProvider] Callback error: %s", exc)

        logger.info(
            "[BarProvider] M1 BAR | %s | %s | O=%.2f H=%.2f L=%.2f C=%.2f V=%d",
            symbol,
            timestamp.strftime("%H:%M"),
            bar["open"],
            bar["high"],
            bar["low"],
            bar["close"],
            bar["volume"],
        )

    def _watchdog(self) -> None:
        while not self._watchdog_stop.is_set():
            self._watchdog_stop.wait(5.0)
            if self._watchdog_stop.is_set():
                break
            try:
                current_gen = int(getattr(self._connector, "failover_generation", 0))
                current_up = bool(self._connector.is_connected())
                if current_gen != self._last_failover_generation or (
                    current_up and not self._last_connected_up
                ):
                    logger.info("[BarProvider] Failover detected, resubscribing...")
                    with self._lock:
                        symbols = list(self._bar_callbacks.keys())
                    for sym in symbols:
                        cbs = self._bar_callbacks.get(sym, [])
                        for cb in cbs:
                            self._do_subscribe(sym, cb)
                    self._last_failover_generation = current_gen
                    self._last_connected_up = current_up
            except Exception as exc:
                logger.error("[BarProvider] Watchdog error: %s", exc)


# -----------------------------------------------------------------------------
# Order Dispatcher - COMPLETE REWRITE
# -----------------------------------------------------------------------------


@dataclass
class CTraderOrderResult:
    """Result from order operation."""

    success: bool
    order_id: Optional[str] = None  # This is the positionId for cTrader
    filled_price: Optional[float] = None
    filled_lots: Optional[float] = None
    error: Optional[str] = None
    raw: Dict[str, Any] = field(default_factory=dict)


class CTraderOrderDispatcher:
    """
    cTrader Order Dispatcher - NATIVE POSITION-CENTRIC DESIGN.
    """

    def __init__(
        self, connector: Any, bar_provider: CTraderBarProvider, fill_timeout_s: float = 5.0
    ):
        self._connector = connector
        self._bar_provider = bar_provider
        self._fill_timeout_s = fill_timeout_s

        # New native execution handling
        self._registry = OperationRegistry()
        self._exec_handler = ExecutionEventHandler(self._registry)
        self._connector.add_push_handler("ProtoOAExecutionEvent", self._exec_handler)
        self._connector.add_push_handler("ProtoOADepthEvent", self._on_depth_event)

        # L2 depth cache for OBI execution buffer
        self._depth_lock = threading.Lock()
        # symbol_id -> quote_id -> (side, price, size)
        self._depth_books: Dict[int, Dict[int, Tuple[str, float, float]]] = {}
        self._depth_last_update_ts: Dict[int, float] = {}
        self._depth_subscribed_ids: Set[int] = set()

        # Metrics
        self._client_counter = 0
        self._counter_lock = threading.Lock()
        self._orders_submitted = 0
        self._orders_filled = 0
        self._orders_failed = 0
        self._consecutive_failures = 0
        self._circuit_breaker = False
        self._metrics_lock = threading.Lock()

        logger.info("[DISPATCHER] Native position-centric design initialized")

    def _ensure_depth_subscription(self, symbol_id: int) -> None:
        with self._depth_lock:
            if symbol_id in self._depth_subscribed_ids:
                return
        try:
            api_msgs = _get_api_msgs()
            req = api_msgs.ProtoOASubscribeDepthQuotesReq()
            req.ctidTraderAccountId = self._connector.credentials.account_id
            req.symbolId.append(symbol_id)
            resp = self._connector.send_and_wait(req, timeout_s=5.0)
            err = getattr(resp, "errorCode", None)
            if err:
                logger.warning("[DEPTH] subscribe error symbol_id=%s err=%s", symbol_id, err)
                return
            with self._depth_lock:
                self._depth_subscribed_ids.add(symbol_id)
        except Exception as exc:
            logger.warning("[DEPTH] subscribe failed symbol_id=%s err=%s", symbol_id, exc)

    def _on_depth_event(self, payload: Any) -> None:
        try:
            symbol_id = int(getattr(payload, "symbolId", 0))
            if symbol_id <= 0:
                return
            digits = int(self._connector.get_digits(symbol_id))
            scale = float(10 ** max(1, min(digits, 10)))
            with self._depth_lock:
                book = self._depth_books.setdefault(symbol_id, {})
                for q in getattr(payload, "newQuotes", []):
                    qid = int(getattr(q, "id", 0))
                    size = float(getattr(q, "size", 0.0))
                    bid_raw = int(getattr(q, "bid", 0))
                    ask_raw = int(getattr(q, "ask", 0))
                    if qid <= 0 or size <= 0.0:
                        continue
                    if bid_raw > 0 and ask_raw <= 0:
                        side = "bid"
                        px = float(bid_raw / scale)
                    elif ask_raw > 0:
                        side = "ask"
                        px = float(ask_raw / scale)
                    else:
                        continue
                    book[qid] = (side, px, size)
                for del_id in getattr(payload, "deletedQuotes", []):
                    try:
                        book.pop(int(del_id), None)
                    except Exception:
                        continue
                self._depth_last_update_ts[symbol_id] = time.time()
        except Exception as exc:
            logger.debug("[DEPTH] event parse failed: %s", exc)

    def get_order_book_imbalance(self, symbol: str, levels: int = 5) -> Optional[float]:
        """Return weighted top-of-book imbalance in [-1, 1]."""
        try:
            symbol_id = self._bar_provider._symbol_ids.get(symbol)
            if symbol_id is None:
                symbol_id = self._connector.find_symbol_id(symbol)
            if symbol_id is None:
                return None
            self._ensure_depth_subscription(int(symbol_id))

            use_levels = max(int(levels), 1)
            with self._depth_lock:
                quotes = list(self._depth_books.get(int(symbol_id), {}).values())
            if not quotes:
                return None

            bids = sorted(
                [(float(px), float(sz)) for side, px, sz in quotes if side == "bid" and sz > 0.0],
                key=lambda x: x[0],
                reverse=True,
            )
            asks = sorted(
                [(float(px), float(sz)) for side, px, sz in quotes if side == "ask" and sz > 0.0],
                key=lambda x: x[0],
            )
            if not bids or not asks:
                return None

            def _weighted_sum(levels_list: List[Tuple[float, float]]) -> float:
                total = 0.0
                for i, (_px, sz) in enumerate(levels_list[:use_levels]):
                    w = 1.0 / float(i + 1)
                    total += float(sz) * w
                return total

            wb = _weighted_sum(bids)
            wa = _weighted_sum(asks)
            denom = wb + wa
            if denom <= 0.0:
                return None
            return float(max(-1.0, min(1.0, (wb - wa) / denom)))
        except Exception as exc:
            logger.debug("[DEPTH] imbalance failed for %s: %s", symbol, exc)
            return None

    def _next_client_id(self) -> str:
        with self._counter_lock:
            self._client_counter += 1
            return f"kt-{self._client_counter:06d}-{int(time.time() * 1000) % 10000}"

    def open_position(
        self,
        symbol: str,
        direction: Any,
        lots: float,
        price: float,
        stop_price: float,
        comment: str = "",
    ) -> CTraderOrderResult:
        """Open a position and return positionId on success."""
        from kinetra.renko.live_trader import TradeDirection

        # Check circuit breaker
        with self._metrics_lock:
            if self._circuit_breaker:
                return CTraderOrderResult(
                    success=False, error=f"Circuit breaker: {self._consecutive_failures} failures"
                )

        is_long = direction == TradeDirection.LONG
        trade_side = _TRADE_SIDE_BUY if is_long else _TRADE_SIDE_SELL
        client_id = self._next_client_id()
        label = (comment[:31] or client_id)[:31]

        logger.info(
            "[OPEN] %s %s %.3f lots @ ~%.2f | clientId=%s",
            symbol,
            "BUY" if is_long else "SELL",
            lots,
            price,
            client_id,
        )

        try:
            # Get symbol
            symbol_id = self._bar_provider._symbol_ids.get(symbol)
            if symbol_id is None:
                symbol_id = self._connector.find_symbol_id(symbol)
            if symbol_id is None:
                return CTraderOrderResult(success=False, error=f"Symbol not found: {symbol}")

            digits = self._connector.get_digits(symbol_id)
            scale = 10**digits
            volume = max(int(round(lots * 100)) * 100, 100)
            stop_loss_int = int(round(stop_price * scale))
            relative_stop = int(round(abs(float(price) - float(stop_price)) * 100000.0))

            # Build request
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
            req.clientOrderId = client_id
            if comment:
                req.comment = comment[:31]

            # Register pending operation
            op = PendingOperation(
                operation_id=client_id,
                operation_type="open",
                symbol=symbol,
                client_order_id=client_id,
            )
            self._registry.register(op)

            # Send
            with self._metrics_lock:
                self._orders_submitted += 1

            resp = self._connector.send_and_wait(req, timeout_s=3.0)
            err = self._extract_error(resp)
            if err:
                self._registry.complete(op, success=False, error=err)
                return CTraderOrderResult(success=False, error=err, raw={"client_id": client_id})

            # Wait for fill
            if not op.event.wait(timeout=self._fill_timeout_s):
                logger.warning("[OPEN] Timeout, reconciling | clientId=%s", client_id)
                position_id = self._reconcile_position_by_label(symbol_id, label)
                if position_id:
                    logger.info("[OPEN] Reconciled positionId=%s", position_id)
                    with self._metrics_lock:
                        self._orders_filled += 1
                        self._consecutive_failures = 0
                    return CTraderOrderResult(
                        success=True,
                        order_id=position_id,
                        raw={"reconciled": True, "position_id": position_id},
                    )
                else:
                    with self._metrics_lock:
                        self._orders_failed += 1
                        self._consecutive_failures += 1
                        if self._consecutive_failures >= 5:
                            self._circuit_breaker = True
                    return CTraderOrderResult(
                        success=False,
                        error="Fill timeout and reconciliation failed",
                        raw={"client_id": client_id},
                    )

            # Success - extract positionId
            position_id = self._extract_position_id_from_op(op)
            if position_id:
                with self._metrics_lock:
                    self._orders_filled += 1
                    self._consecutive_failures = 0
                return CTraderOrderResult(
                    success=True,
                    order_id=position_id,
                    filled_price=price,
                    filled_lots=lots,
                    raw={"position_id": position_id, "client_id": client_id},
                )
            else:
                return CTraderOrderResult(
                    success=False,
                    error="Fill confirmed but positionId not found",
                    raw={"client_id": client_id},
                )

        except Exception as exc:
            logger.exception("[OPEN] Exception: %s", exc)
            return CTraderOrderResult(success=False, error=str(exc))

    def close_position(
        self, symbol: str, order_id: str, price: float, lots: float, comment: str = ""
    ) -> CTraderOrderResult:
        """Close a position by positionId."""
        position_id = order_id  # Clarify: order_id is positionId

        logger.info("[CLOSE] %s positionId=%s lots=%.3f", symbol, position_id, lots)

        try:
            pid_int = int(position_id)
        except (TypeError, ValueError):
            return CTraderOrderResult(success=False, error=f"Invalid positionId: {position_id}")

        try:
            volume = max(int(round(lots * 100)) * 100, 100)

            # Register pending operation - KEY: use position_id as the key!
            op = PendingOperation(
                operation_id=position_id,  # Use position_id as operation key
                operation_type="close",
                symbol=symbol,
                position_id=position_id,
            )
            self._registry.register(op)

            # Build and send close request
            api_msgs = _get_api_msgs()
            req = api_msgs.ProtoOAClosePositionReq()
            req.ctidTraderAccountId = self._connector.credentials.account_id
            req.positionId = pid_int
            req.volume = volume

            resp = self._connector.send_and_wait(req, timeout_s=3.0)
            err = self._extract_error(resp)
            if err:
                self._registry.complete(op, success=False, error=err)
                return CTraderOrderResult(success=False, error=err)

            # Wait for execution event
            if not op.event.wait(timeout=self._fill_timeout_s):
                logger.warning(
                    "[CLOSE] Timeout, checking position status | positionId=%s", position_id
                )

                # Verify via position list
                is_open = self.is_position_open(position_id)
                if is_open is False:
                    logger.info(
                        "[CLOSE] Confirmed closed via reconciliation | positionId=%s", position_id
                    )
                    return CTraderOrderResult(
                        success=True, order_id=position_id, raw={"reconciled": True}
                    )
                elif is_open is True:
                    logger.error("[CLOSE] Position still open | positionId=%s", position_id)
                    return CTraderOrderResult(
                        success=False,
                        error="Close failed - position still open",
                        raw={"position_id": position_id},
                    )
                else:
                    logger.error("[CLOSE] Status unknown | positionId=%s", position_id)
                    return CTraderOrderResult(
                        success=False,
                        error="Close status unknown - reconcile failed",
                        raw={"position_id": position_id},
                    )

            # Success
            logger.info("[CLOSE] Confirmed via execution event | positionId=%s", position_id)
            return CTraderOrderResult(
                success=True, order_id=position_id, raw={"event_received": True}
            )

        except Exception as exc:
            logger.exception("[CLOSE] Exception: %s", exc)
            return CTraderOrderResult(success=False, error=str(exc))

    def is_position_open(self, order_id: str) -> Optional[bool]:
        """Check if position is still open via API."""
        try:
            pid_int = int(order_id)
            api_msgs = _get_api_msgs()
            req = api_msgs.ProtoOAPositionListReq()
            req.ctidTraderAccountId = self._connector.credentials.account_id

            resp = self._connector.send_and_wait(req, timeout_s=5.0)
            err = getattr(resp, "errorCode", None)
            if err:
                logger.error("[RECONCILE] PositionList error: %s", err)
                return None

            for pos in getattr(resp, "position", []):
                if int(getattr(pos, "positionId", 0)) == pid_int:
                    return True
            return False
        except Exception as exc:
            logger.exception("[RECONCILE] Error: %s", exc)
            return None

    def _reconcile_position_by_label(self, symbol_id: int, label: str) -> Optional[str]:
        """Try to find position by label after timeout."""
        try:
            time.sleep(0.5)  # Brief delay for position to be created
            api_msgs = _get_api_msgs()
            req = api_msgs.ProtoOAPositionListReq()
            req.ctidTraderAccountId = self._connector.credentials.account_id

            resp = self._connector.send_and_wait(req, timeout_s=5.0)
            err = getattr(resp, "errorCode", None)
            if err:
                logger.error("[RECONCILE] PositionList error: %s", err)
                return None

            for pos in getattr(resp, "position", []):
                trade_data = getattr(pos, "tradeData", None)
                if trade_data:
                    pos_label = str(getattr(trade_data, "label", "") or "")
                    if pos_label == label:
                        position_id = str(getattr(pos, "positionId", "") or "")
                        if position_id:
                            return position_id
            return None
        except Exception as exc:
            logger.exception("[RECONCILE] Error: %s", exc)
            return None

    def _extract_position_id_from_op(self, op: PendingOperation) -> Optional[str]:
        """Extract positionId from completed operation."""
        if op.position_id:
            return op.position_id

        if op.result_payload:
            position = getattr(op.result_payload, "position", None)
            if position:
                pos_id = str(getattr(position, "positionId", "") or "")
                if pos_id:
                    return pos_id

            order = getattr(op.result_payload, "order", None)
            if order:
                pos_id = str(getattr(order, "positionId", "") or "")
                if pos_id:
                    return pos_id

        return None

    def _extract_error(self, resp: Any) -> Optional[str]:
        """Extract error from response."""
        if resp is None:
            return None
        err = getattr(resp, "errorCode", None)
        if err:
            return str(err)
        return None

    def get_fill_metrics(self) -> Dict[str, Any]:
        """Get current fill metrics."""
        with self._metrics_lock:
            return {
                "submitted": self._orders_submitted,
                "filled": self._orders_filled,
                "failed": self._orders_failed,
                "consecutive_failures": self._consecutive_failures,
                "circuit_breaker": self._circuit_breaker,
            }

    def reset_circuit_breaker(self) -> None:
        """Reset circuit breaker after manual intervention."""
        with self._metrics_lock:
            self._circuit_breaker = False
            self._consecutive_failures = 0
            logger.info("[DISPATCHER] Circuit breaker reset")

    def close(self) -> None:
        """Cleanup dispatcher handlers/subscriptions."""
        try:
            self._connector.remove_push_handler("ProtoOAExecutionEvent", self._exec_handler)
        except Exception:
            pass
        try:
            self._connector.remove_push_handler("ProtoOADepthEvent", self._on_depth_event)
        except Exception:
            pass
        api_msgs = _get_api_msgs()
        with self._depth_lock:
            ids = list(self._depth_subscribed_ids)
            self._depth_subscribed_ids.clear()
        for symbol_id in ids:
            try:
                req = api_msgs.ProtoOAUnsubscribeDepthQuotesReq()
                req.ctidTraderAccountId = self._connector.credentials.account_id
                req.symbolId.append(int(symbol_id))
                self._connector.send_and_wait(req, timeout_s=3.0)
            except Exception:
                continue


# -----------------------------------------------------------------------------
# Convenience function
# -----------------------------------------------------------------------------


def build_ctrader_session(credentials: Any) -> Tuple[CTraderBarProvider, CTraderOrderDispatcher]:
    """Build connected bar provider and order dispatcher."""
    from kinetra.connectors.ctrader_connector import CTraderConnector

    connector = CTraderConnector(credentials)
    connector.start()

    bar_provider = CTraderBarProvider(connector)
    dispatcher = CTraderOrderDispatcher(connector, bar_provider)

    return bar_provider, dispatcher
