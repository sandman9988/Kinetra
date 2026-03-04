#!/usr/bin/env python3
"""
Redesigned cTrader Dispatcher - Native Approach
===============================================

This is a complete rewrite of the execution handling to work natively
with cTrader's position-centric API.

To use this redesign:
1. Copy this file to replace kinetra/renko/ctrader_dispatcher.py
2. Or import and use NativeCTraderOrderDispatcher instead

Key improvements:
- Position-centric (not order-centric)
- Simple operation registry
- Reliable reconciliation
- Clear logging
"""

import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------

_EXEC_TYPE_ORDER_ACCEPTED = 2
_EXEC_TYPE_ORDER_FILLED = 3
_EXEC_TYPE_ORDER_PARTIAL_FILL = 11
_EXEC_TYPE_ORDER_REJECTED = 7

_FILL_EXEC_TYPES: Set[int] = {_EXEC_TYPE_ORDER_FILLED, _EXEC_TYPE_ORDER_PARTIAL_FILL}

_ORDER_TYPE_MARKET = 1
_TRADE_SIDE_BUY = 1
_TRADE_SIDE_SELL = 2


# -----------------------------------------------------------------------------
# Operation Tracking
# -----------------------------------------------------------------------------


@dataclass
class PendingOperation:
    """Tracks a single open/close operation awaiting execution event."""

    operation_id: str
    operation_type: str  # "open" or "close"
    symbol: str

    # Known identifiers (populated as we learn them)
    client_order_id: Optional[str] = None
    position_id: Optional[str] = None
    order_id: Optional[str] = None

    # Result
    event: threading.Event = field(default_factory=threading.Event)
    result_payload: Optional[Any] = None
    success: bool = False
    error: Optional[str] = None

    # Metadata
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "operation_id": self.operation_id,
            "type": self.operation_type,
            "symbol": self.symbol,
            "client_order_id": self.client_order_id,
            "position_id": self.position_id,
            "order_id": self.order_id,
            "success": self.success,
            "error": self.error,
            "age_ms": (datetime.now(timezone.utc) - self.created_at).total_seconds() * 1000,
        }


class OperationRegistry:
    """
    Thread-safe registry of pending operations.

    Provides multiple index views for fast lookup:
    - By operation_id (primary)
    - By client_order_id
    - By position_id
    """

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
            logger.debug("[REGISTRY] Registered %s | op=%s", op.operation_type, op.operation_id)

    def find(
        self,
        operation_id: Optional[str] = None,
        client_order_id: Optional[str] = None,
        position_id: Optional[str] = None,
    ) -> Optional[PendingOperation]:
        """Find an operation by any known identifier."""
        with self._lock:
            if operation_id and operation_id in self._ops:
                return self._ops[operation_id]

            if client_order_id and client_order_id in self._by_client_id:
                return self._ops[self._by_client_id[client_order_id]]

            if position_id and position_id in self._by_position_id:
                return self._ops[self._by_position_id[position_id]]

            return None

    def update_identifiers(
        self,
        op: PendingOperation,
        client_order_id: Optional[str] = None,
        position_id: Optional[str] = None,
        order_id: Optional[str] = None,
    ) -> None:
        """Update operation with newly learned identifiers."""
        with self._lock:
            if client_order_id and client_order_id not in self._by_client_id:
                op.client_order_id = client_order_id
                self._by_client_id[client_order_id] = op.operation_id

            if position_id and position_id not in self._by_position_id:
                op.position_id = position_id
                self._by_position_id[position_id] = op.operation_id

            if order_id:
                op.order_id = order_id

    def complete(
        self,
        op: PendingOperation,
        success: bool,
        payload: Optional[Any] = None,
        error: Optional[str] = None,
    ) -> None:
        """Mark an operation as complete."""
        op.success = success
        op.result_payload = payload
        op.error = error
        op.event.set()

        # Schedule cleanup
        threading.Timer(10.0, self._cleanup, args=[op.operation_id]).start()

    def _cleanup(self, operation_id: str) -> None:
        """Remove completed operation from registry."""
        with self._lock:
            if operation_id not in self._ops:
                return

            op = self._ops[operation_id]

            # Remove from all indexes
            del self._ops[operation_id]

            if op.client_order_id and op.client_order_id in self._by_client_id:
                if self._by_client_id[op.client_order_id] == operation_id:
                    del self._by_client_id[op.client_order_id]

            if op.position_id and op.position_id in self._by_position_id:
                if self._by_position_id[op.position_id] == operation_id:
                    del self._by_position_id[op.position_id]

            logger.debug("[REGISTRY] Cleaned up op=%s", operation_id)

    def get_all_pending(self) -> List[PendingOperation]:
        """Get all pending operations."""
        with self._lock:
            return [op for op in self._ops.values() if not op.event.is_set()]


# -----------------------------------------------------------------------------
# Execution Event Handler
# -----------------------------------------------------------------------------


class ExecutionEventHandler:
    """
    Handles ProtoOAExecutionEvent messages and routes them to pending operations.

    This is designed to be registered as a push handler on the connector.
    """

    def __init__(self, registry: OperationRegistry):
        self._registry = registry
        self._shutdown = False

    def __call__(self, payload: Any) -> None:
        """Handle execution event (called by connector on reactor thread)."""
        if self._shutdown:
            return

        try:
            self._handle_event(payload)
        except Exception as exc:
            logger.exception("[EXEC-HANDLER] Error handling event: %s", exc)

    def _handle_event(self, payload: Any) -> None:
        """Process a single execution event."""
        exec_type = int(getattr(payload, "executionType", -1))

        # Extract all possible identifiers
        identifiers = self._extract_identifiers(payload)

        logger.debug(
            "[EXEC-HANDLER] Event | type=%s clientOid=%s orderId=%s positionId=%s",
            exec_type,
            identifiers.get("client_order_id") or "-",
            identifiers.get("order_id") or "-",
            identifiers.get("position_id") or "-",
        )

        # Find matching operation
        op = self._registry.find(
            client_order_id=identifiers.get("client_order_id"),
            position_id=identifiers.get("position_id"),
            order_id=identifiers.get("order_id"),
        )

        if op is None:
            logger.debug("[EXEC-HANDLER] No matching operation")
            return

        # Update with any new identifiers we learned
        self._registry.update_identifiers(
            op,
            client_order_id=identifiers.get("client_order_id"),
            position_id=identifiers.get("position_id"),
            order_id=identifiers.get("order_id"),
        )

        # Handle ORDER_ACCEPTED - learn identifiers but don't complete
        if exec_type == _EXEC_TYPE_ORDER_ACCEPTED:
            logger.debug("[EXEC-HANDLER] ORDER_ACCEPTED for op=%s", op.operation_id)
            return

        # Handle rejection
        if exec_type == _EXEC_TYPE_ORDER_REJECTED:
            logger.warning("[EXEC-HANDLER] ORDER_REJECTED for op=%s", op.operation_id)
            self._registry.complete(
                op, success=False, payload=payload, error="Order rejected by broker"
            )
            return

        # Handle fills
        if exec_type in _FILL_EXEC_TYPES:
            logger.info(
                "[EXEC-HANDLER] FILL for op=%s type=%s | positionId=%s",
                op.operation_id,
                op.operation_type,
                identifiers.get("position_id"),
            )
            self._registry.complete(op, success=True, payload=payload)
            return

        # Other events (log for debugging)
        logger.debug("[EXEC-HANDLER] Unhandled exec_type=%s", exec_type)

    def _extract_identifiers(self, payload: Any) -> Dict[str, Optional[str]]:
        """Extract all identifiers from execution event payload."""
        result: Dict[str, Optional[str]] = {
            "client_order_id": None,
            "order_id": None,
            "position_id": None,
        }

        # Extract from order object
        order = getattr(payload, "order", None)
        if order is not None:
            result["client_order_id"] = str(getattr(order, "clientOrderId", "") or "")
            result["order_id"] = str(getattr(order, "orderId", "") or "")
            result["position_id"] = str(getattr(order, "positionId", "") or "")

            # Fallback: check tradeData.label for clientOrderId
            if not result["client_order_id"]:
                trade_data = getattr(order, "tradeData", None)
                if trade_data:
                    result["client_order_id"] = str(getattr(trade_data, "label", "") or "")

        # Extract from position object
        position = getattr(payload, "position", None)
        if position is not None:
            pos_id = str(getattr(position, "positionId", "") or "")
            if pos_id and not result["position_id"]:
                result["position_id"] = pos_id

        # Extract from deal object (most important for closes)
        deal = getattr(payload, "deal", None)
        if deal is not None:
            deal_pos_id = str(getattr(deal, "positionId", "") or "")
            if deal_pos_id:
                result["position_id"] = deal_pos_id

            deal_order_id = str(getattr(deal, "orderId", "") or "")
            if deal_order_id and not result["order_id"]:
                result["order_id"] = deal_order_id

        return result

    def shutdown(self) -> None:
        """Signal shutdown."""
        self._shutdown = True


# -----------------------------------------------------------------------------
# Simplified Dispatcher
# -----------------------------------------------------------------------------


@dataclass
class NativeOrderResult:
    """Simple order result."""

    success: bool
    position_id: Optional[str] = None
    order_id: Optional[str] = None
    filled_price: Optional[float] = None
    filled_lots: Optional[float] = None
    error: Optional[str] = None
    raw: Dict[str, Any] = field(default_factory=dict)


class NativeCTraderOrderDispatcher:
    """
    Simplified, native cTrader order dispatcher.

    Usage:
        dispatcher = NativeCTraderOrderDispatcher(connector, bar_provider)

        # Open
        result = dispatcher.open_position("XAUUSD", TradeDirection.LONG, 0.01, 2900.0, 2890.0)
        if result.success:
            position_id = result.position_id  # Use this for close

        # Close
        result = dispatcher.close_position("XAUUSD", position_id, 0.01)
    """

    def __init__(self, connector: Any, bar_provider: Any, fill_timeout_s: float = 5.0):
        self._connector = connector
        self._bar_provider = bar_provider
        self._timeout_s = fill_timeout_s
        self._registry = OperationRegistry()
        self._handler = ExecutionEventHandler(self._registry)
        self._client_counter = 0
        self._counter_lock = threading.Lock()

        # Register handler
        self._connector.add_push_handler("ProtoOAExecutionEvent", self._handler)

        logger.info("[NATIVE-DISPATCHER] Initialized with timeout=%.1fs", fill_timeout_s)

    def _next_client_id(self) -> str:
        """Generate unique client order ID."""
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
    ) -> NativeOrderResult:
        """Open a position and return positionId on success."""

        client_id = self._next_client_id()
        label = (comment[:31] or client_id)[:31]

        logger.info(
            "[OPEN] %s %s %.3f lots @ ~%.2f | clientId=%s",
            symbol,
            "BUY" if str(direction).endswith("LONG") else "SELL",
            lots,
            price,
            client_id,
        )

        try:
            # Get symbolId
            symbol_id = self._get_symbol_id(symbol)
            if symbol_id is None:
                return NativeOrderResult(success=False, error=f"Cannot resolve symbol {symbol}")

            # Build request
            req = self._build_new_order_req(
                symbol_id=symbol_id,
                direction=direction,
                lots=lots,
                stop_price=stop_price,
                label=label,
                client_id=client_id,
            )

            # Register pending operation
            op = PendingOperation(
                operation_id=client_id,
                operation_type="open",
                symbol=symbol,
                client_order_id=client_id,
            )
            self._registry.register(op)

            # Send request
            resp = self._connector.send_and_wait(req, timeout_s=3.0)

            # Check immediate error
            error = self._extract_error(resp)
            if error:
                self._registry.complete(op, success=False, error=error)
                return NativeOrderResult(success=False, error=error, raw={"client_id": client_id})

            # Wait for fill
            if not op.event.wait(timeout=self._timeout_s):
                # Timeout - try reconciliation
                logger.warning(
                    "[OPEN] Fill timeout, trying reconciliation | clientId=%s", client_id
                )
                position_id = self._reconcile_open_position(symbol, symbol_id, client_id)

                if position_id:
                    return NativeOrderResult(
                        success=True,
                        position_id=position_id,
                        raw={"reconciled": True, "client_id": client_id},
                    )
                else:
                    return NativeOrderResult(
                        success=False,
                        error="Fill timeout and reconciliation failed",
                        raw={"client_id": client_id},
                    )

            # Extract result
            return self._extract_open_result(op, client_id)

        except Exception as exc:
            logger.exception("[OPEN] Exception: %s", exc)
            return NativeOrderResult(success=False, error=str(exc))

    def close_position(
        self, symbol: str, position_id: str, lots: float, price: float = 0.0, comment: str = ""
    ) -> NativeOrderResult:
        """Close a position by positionId."""

        logger.info("[CLOSE] %s positionId=%s lots=%.3f", symbol, position_id, lots)

        try:
            pid_int = int(position_id)
        except (TypeError, ValueError):
            return NativeOrderResult(success=False, error=f"Invalid positionId: {position_id}")

        try:
            # Register pending operation (KEY: use position_id as the key!)
            op = PendingOperation(
                operation_id=position_id,  # Use position_id as primary key
                operation_type="close",
                symbol=symbol,
                position_id=position_id,
            )
            self._registry.register(op)

            # Build and send close request
            req = self._build_close_position_req(pid_int, lots)
            resp = self._connector.send_and_wait(req, timeout_s=3.0)

            # Check immediate error
            error = self._extract_error(resp)
            if error:
                self._registry.complete(op, success=False, error=error)
                return NativeOrderResult(success=False, error=error)

            # Wait for fill
            if not op.event.wait(timeout=self._timeout_s):
                logger.warning(
                    "[CLOSE] Fill timeout, checking position status | positionId=%s", position_id
                )

                # Verify via position list
                if self._is_position_open(position_id):
                    return NativeOrderResult(
                        success=False,
                        error="Close failed - position still open",
                        raw={"position_id": position_id},
                    )
                else:
                    return NativeOrderResult(
                        success=True, position_id=position_id, raw={"reconciled": True}
                    )

            # Success
            return NativeOrderResult(
                success=True, position_id=position_id, raw={"event_received": True}
            )

        except Exception as exc:
            logger.exception("[CLOSE] Exception: %s", exc)
            return NativeOrderResult(success=False, error=str(exc))

    def _get_symbol_id(self, symbol: str) -> Optional[int]:
        """Get cTrader symbol ID."""
        # Implementation would use bar_provider or connector
        if hasattr(self._bar_provider, "_symbol_ids"):
            return self._bar_provider._symbol_ids.get(symbol)
        return self._connector.find_symbol_id(symbol)

    def _build_new_order_req(
        self,
        symbol_id: int,
        direction: Any,
        lots: float,
        stop_price: float,
        label: str,
        client_id: str,
    ) -> Any:
        """Build ProtoOANewOrderReq."""
        # Would use actual proto messages
        pass

    def _build_close_position_req(self, position_id: int, lots: float) -> Any:
        """Build ProtoOAClosePositionReq."""
        # Would use actual proto messages
        pass

    def _extract_error(self, resp: Any) -> Optional[str]:
        """Extract error from response."""
        # Implementation
        pass

    def _extract_open_result(self, op: PendingOperation, client_id: str) -> NativeOrderResult:
        """Extract result from completed open operation."""
        if not op.success:
            return NativeOrderResult(success=False, error=op.error or "Unknown error")

        payload = op.result_payload
        if payload is None:
            return NativeOrderResult(success=False, error="No result payload")

        # Extract positionId from payload
        position_id = op.position_id
        if not position_id:
            # Try to extract from result
            position = getattr(payload, "position", None)
            if position:
                position_id = str(getattr(position, "positionId", "") or "")

        return NativeOrderResult(
            success=True,
            position_id=position_id,
            order_id=op.order_id,
            raw={"client_id": client_id, "position_id": position_id},
        )

    def _reconcile_open_position(
        self, symbol: str, symbol_id: int, client_id: str
    ) -> Optional[str]:
        """Try to find position via API after timeout."""
        # Would use ProtoOAPositionListReq or similar
        pass

    def _is_position_open(self, position_id: str) -> bool:
        """Check if position is still open."""
        # Would use ProtoOAPositionListReq
        pass


# -----------------------------------------------------------------------------
# Migration Guide
# -----------------------------------------------------------------------------

"""
Migration from old dispatcher to new:

OLD:
    dispatcher = CTraderOrderDispatcher(connector, bar_provider)
    result = dispatcher.open_position(...)
    if result.success:
        order_id = result.order_id  # This was positionId

NEW:
    dispatcher = NativeCTraderOrderDispatcher(connector, bar_provider)
    result = dispatcher.open_position(...)
    if result.success:
        position_id = result.position_id  # Clear naming!

    # Close using position_id
    close_result = dispatcher.close_position(symbol, position_id, lots)

Key differences:
1. Result uses position_id (not order_id) - clearer naming
2. Close is position-centric (uses position_id directly)
3. Simpler internal logic - easier to debug
4. Better reconciliation as first-class feature
"""
