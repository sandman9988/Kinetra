"""
Native cTrader Execution Manager
================================

Redesigned to work naturally with cTrader's Open API.

Key Insights from cTrader Architecture:
---------------------------------------
1. cTrader is position-centric, not order-centric
2. ProtoOAClosePositionReq uses positionId, not orderId
3. Execution events have Deal objects with positionId
4. clientOrderId is optional and may not be present on close events

Design Principles:
------------------
1. **Position ID is the primary key** - Everything revolves around positionId
2. **Simple matching** - Match by positionId from Deal or Position objects
3. **Reconciliation as fallback** - Poll position list if event matching fails
4. **No two-phase complexity** - Single-phase wait for fill events

Execution Flow:
---------------
open_position():
    1. Send ProtoOANewOrderReq with clientOrderId
    2. Wait for ExecutionEvent with matching clientOrderId
    3. Extract positionId from event
    4. Return positionId as order_id

close_position():
    1. Send ProtoOAClosePositionReq with positionId
    2. Wait for ExecutionEvent with matching positionId (from Deal)
    3. Confirm position is closed via reconciliation
    4. Return success/failure

Matching Strategy:
------------------
For OPEN events:
- Primary: Match by clientOrderId (from order.label or order.clientOrderId)
- Fallback: Match by positionId if we already know it (reconciliation)

For CLOSE events:
- Primary: Match by positionId from Deal object
- Note: clientOrderId is often NOT present on close execution events
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Set, Tuple

if TYPE_CHECKING:
    from kinetra.renko.live_trader import OrderDispatcher

logger = logging.getLogger(__name__)

# Execution type constants from cTrader Open API proto
_EXEC_TYPE_ORDER_ACCEPTED = 2
_EXEC_TYPE_ORDER_FILLED = 3
_EXEC_TYPE_ORDER_PARTIAL_FILL = 11

_SUCCESS_EXEC_TYPES: Set[int] = {_EXEC_TYPE_ORDER_FILLED, _EXEC_TYPE_ORDER_PARTIAL_FILL}


@dataclass
class _PendingOperation:
    """Represents an operation waiting for execution event confirmation."""

    operation_id: str  # Our internal tracking ID (clientOrderId for opens)
    position_id: Optional[str]  # cTrader position ID (known after open fill)
    operation_type: str  # "open" or "close"
    event: threading.Event
    result_payload: Optional[Any] = None
    created_at: datetime = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now(timezone.utc)


class NativeExecutionManager:
    """
    Native cTrader execution event manager.

    Handles matching of execution events to pending operations using
    cTrader-native identifiers (positionId, clientOrderId).

    This class is designed to be registered as a push handler for
    ProtoOAExecutionEvent messages.
    """

    def __init__(self, timeout_s: float = 5.0):
        self._timeout_s = timeout_s
        self._pending: Dict[str, _PendingOperation] = {}  # operation_id -> pending
        self._position_to_op: Dict[str, str] = {}  # position_id -> operation_id
        self._lock = threading.Lock()
        self._shutdown = False

    def register_open(self, client_order_id: str) -> _PendingOperation:
        """Register a new pending open operation."""
        with self._lock:
            op = _PendingOperation(
                operation_id=client_order_id,
                position_id=None,
                operation_type="open",
                event=threading.Event(),
            )
            self._pending[client_order_id] = op
            logger.debug("[EXEC-MGR] Registered OPEN | clientOrderId=%s", client_order_id)
            return op

    def register_close(
        self, position_id: str, client_order_id: Optional[str] = None
    ) -> _PendingOperation:
        """Register a new pending close operation."""
        with self._lock:
            # Use position_id as primary key for closes
            op = _PendingOperation(
                operation_id=position_id,  # Use positionId as the key
                position_id=position_id,
                operation_type="close",
                event=threading.Event(),
            )
            self._pending[position_id] = op
            if client_order_id:
                # Also index by clientOrderId if provided (for fallback matching)
                self._position_to_op[client_order_id] = position_id
            logger.debug(
                "[EXEC-MGR] Registered CLOSE | positionId=%s clientOrderId=%s",
                position_id,
                client_order_id or "n/a",
            )
            return op

    def handle_execution_event(self, payload: Any) -> None:
        """
        Process ProtoOAExecutionEvent.

        Called by the connector's push handler on the reactor thread.
        """
        if self._shutdown:
            return

        exec_type = int(getattr(payload, "executionType", -1))

        # Extract identifiers from various sources
        order = getattr(payload, "order", None)
        position = getattr(payload, "position", None)
        deal = getattr(payload, "deal", None)

        client_oid = ""
        position_id = ""

        # Try to get identifiers from order
        if order is not None:
            client_oid = str(getattr(order, "clientOrderId", "") or "")
            if not client_oid:
                # Fallback to label
                trade_data = getattr(order, "tradeData", None)
                if trade_data:
                    client_oid = str(getattr(trade_data, "label", "") or "")
            position_id = str(getattr(order, "positionId", "") or "")

        # Try to get positionId from position object
        if position is not None and not position_id:
            position_id = str(getattr(position, "positionId", "") or "")

        # Try to get positionId from deal object (most reliable for closes)
        if deal is not None:
            deal_position_id = str(getattr(deal, "positionId", "") or "")
            if deal_position_id:
                position_id = deal_position_id

        # Skip if no identifiers
        if not client_oid and not position_id:
            logger.debug("[EXEC-MGR] Event ignored - no identifiers | exec_type=%s", exec_type)
            return

        with self._lock:
            matched_op: Optional[_PendingOperation] = None

            # Try to match by clientOrderId (for opens)
            if client_oid and client_oid in self._pending:
                matched_op = self._pending[client_oid]

            # Try to match by positionId (for closes)
            elif position_id and position_id in self._pending:
                matched_op = self._pending[position_id]

            # Check if client_oid maps to a position-based operation
            elif client_oid and client_oid in self._position_to_op:
                mapped_pos_id = self._position_to_op[client_oid]
                if mapped_pos_id in self._pending:
                    matched_op = self._pending[mapped_pos_id]

            if matched_op is None:
                logger.debug(
                    "[EXEC-MGR] Event not matched | exec_type=%s clientOid=%s positionId=%s pending=%s",
                    exec_type,
                    client_oid,
                    position_id,
                    list(self._pending.keys()),
                )
                return

            # Handle ORDER_ACCEPTED - just update positionId if we learn it
            if exec_type == _EXEC_TYPE_ORDER_ACCEPTED:
                if position_id and not matched_op.position_id:
                    matched_op.position_id = position_id
                    self._position_to_op[position_id] = matched_op.operation_id
                    logger.debug(
                        "[EXEC-MGR] Learned positionId from ORDER_ACCEPTED | op=%s positionId=%s",
                        matched_op.operation_id,
                        position_id,
                    )
                return  # Don't signal yet - wait for actual fill

            # Handle fill events
            if exec_type in _SUCCESS_EXEC_TYPES:
                matched_op.result_payload = payload
                matched_op.event.set()

                # Update positionId if we learned it
                if position_id and not matched_op.position_id:
                    matched_op.position_id = position_id

                logger.info(
                    "[EXEC-MGR] Fill confirmed | op=%s type=%s positionId=%s exec_type=%s",
                    matched_op.operation_id,
                    matched_op.operation_type,
                    position_id or "n/a",
                    exec_type,
                )

                # Clean up completed operation after a delay
                # (keep it briefly for any late events)
                threading.Timer(
                    5.0, self._cleanup_operation, args=[matched_op.operation_id]
                ).start()

    def _cleanup_operation(self, operation_id: str) -> None:
        """Remove completed operation from tracking."""
        with self._lock:
            if operation_id in self._pending:
                op = self._pending[operation_id]
                del self._pending[operation_id]
                # Clean up position mapping too
                if op.position_id and op.position_id in self._position_to_op:
                    del self._position_to_op[op.position_id]
                logger.debug("[EXEC-MGR] Cleaned up operation | op=%s", operation_id)

    def wait_for_result(
        self, operation_id: str, timeout_s: Optional[float] = None
    ) -> Optional[Any]:
        """Wait for an operation to complete."""
        with self._lock:
            op = self._pending.get(operation_id)
            if op is None:
                return None

        timeout = timeout_s or self._timeout_s
        success = op.event.wait(timeout=timeout)

        if success:
            return op.result_payload
        return None  # Timeout

    def cancel_operation(self, operation_id: str) -> bool:
        """Cancel a pending operation (e.g., on error)."""
        with self._lock:
            if operation_id in self._pending:
                del self._pending[operation_id]
                return True
            return False

    def shutdown(self) -> None:
        """Signal all pending operations to abort."""
        self._shutdown = True
        with self._lock:
            for op in self._pending.values():
                op.event.set()  # Signal to unblock waiters


class NativeCTraderDispatcher:
    """
    Simplified cTrader dispatcher using native position-centric design.

    This replaces the complex two-phase waiter with a simpler model:
    - ExecutionManager handles all event matching
    - Position ID is the primary key
    - Reconciliation is used as a reliable fallback
    """

    def __init__(self, connector, bar_provider, fill_timeout_s: float = 5.0):
        self._connector = connector
        self._bar_provider = bar_provider
        self._fill_timeout_s = fill_timeout_s
        self._exec_mgr = NativeExecutionManager(timeout_s=fill_timeout_s)
        self._client_order_counter = 0
        self._counter_lock = threading.Lock()

        # Register execution manager as push handler
        self._connector.add_push_handler(
            "ProtoOAExecutionEvent", self._exec_mgr.handle_execution_event
        )

    def _next_client_order_id(self) -> str:
        """Generate unique client order ID."""
        with self._counter_lock:
            self._client_order_counter += 1
            return f"kinetra-{self._client_order_counter:06d}-{int(time.time())}"

    def open_position(
        self,
        symbol: str,
        direction: Any,
        lots: float,
        price: float,
        stop_price: float,
        comment: str = "",
    ) -> Any:
        """
        Open a position and return positionId on success.
        """
        from kinetra.renko.live_trader import OrderResult

        # Implementation would go here...
        # 1. Get symbolId
        # 2. Build ProtoOANewOrderReq
        # 3. Register with execution manager
        # 4. Send request
        # 5. Wait for fill event
        # 6. Extract positionId from result
        pass

    def close_position(
        self, symbol: str, position_id: str, price: float, lots: float, comment: str = ""
    ) -> Any:
        """
        Close a position by positionId.

        Key insight: ProtoOAClosePositionReq sends a Deal, not an Order.
        The Deal has the positionId, so we match by that.
        """
        from kinetra.renko.live_trader import OrderResult

        # Implementation would go here...
        # 1. Register close with execution manager (key=position_id)
        # 2. Send ProtoOAClosePositionReq
        # 3. Wait for execution event
        # 4. Verify via reconciliation
        pass

    def is_position_open(self, position_id: str) -> Optional[bool]:
        """Check if a position is still open via API call."""
        # Implementation using ProtoOAPositionListReq or ProtoOADealListByPositionIdReq
        pass
