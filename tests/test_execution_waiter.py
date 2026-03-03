"""
Tests for _ExecutionWaiter (kinetra.renko.ctrader_dispatcher)
=============================================================

_ExecutionWaiter uses no broker imports — only threading — so every case can
be exercised with plain namespace objects.  No mocking framework required.

Coverage targets
----------------
- Normal fill path  (ORDER_FILLED arrives directly, matched by clientOrderId)
- Two-phase path    (ORDER_ACCEPTED → learns broker orderId → ORDER_FILLED)
- Broker-id fallback (ORDER_FILLED carries only broker orderId, not clientOrderId)
- Label matching    (clientOrderId absent but label present)
- False-positive rejection: events with no identifiers
- Wrong-id rejection: identifiers present but belong to a different order
- Non-fill exec types are skipped (ORDER_REJECTED, ORDER_EXPIRED, etc.)
- Timeout          (no matching event → wait() returns None)
- Thread safety    (handler called from a background thread)
"""

from __future__ import annotations

import threading
import time
from types import SimpleNamespace
from typing import Any, Optional

import pytest

# The class is module-private but importable for tests.
from kinetra.renko.ctrader_dispatcher import _ExecutionWaiter, _SUCCESS_EXECUTION_TYPES


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_CLIENT_OID = "RENKO-00000001"
_BROKER_OID = "9876543"

# cTrader executionType constants — from OpenApiModelMessages.proto
_TYPE_ORDER_ACCEPTED = 2
_TYPE_ORDER_FILLED = 3
_TYPE_ORDER_PARTIAL_FILL = 11
_TYPE_ORDER_REJECTED = 7
_TYPE_ORDER_EXPIRED = 6


def _make_payload(
    exec_type: int,
    client_order_id: str = "",
    broker_order_id: str = "",
    label: str = "",
    error_code: Optional[str] = None,
) -> Any:
    """Build a minimal ProtoOAExecutionEvent-like namespace."""
    trade_data = SimpleNamespace(label=label) if label else None
    order = SimpleNamespace(
        clientOrderId=client_order_id,
        orderId=broker_order_id,
        tradeData=trade_data,
    )
    ns = SimpleNamespace(executionType=exec_type, order=order)
    if error_code is not None:
        ns.errorCode = error_code
    return ns


def _make_payload_no_order(exec_type: int) -> Any:
    """Payload with no ``order`` subfield at all."""
    return SimpleNamespace(executionType=exec_type)


def _fire_after(waiter: _ExecutionWaiter, payload: Any, delay_s: float = 0.02) -> None:
    """Call waiter.handle(payload) from a background thread after *delay_s*."""

    def _run() -> None:
        time.sleep(delay_s)
        waiter.handle(payload)

    t = threading.Thread(target=_run, daemon=True)
    t.start()


# ---------------------------------------------------------------------------
# Normal fill path
# ---------------------------------------------------------------------------


def test_normal_fill_matched_by_client_order_id() -> None:
    """ORDER_FILLED with matching clientOrderId fires the waiter."""
    waiter = _ExecutionWaiter(_CLIENT_OID, timeout_s=1.0)
    payload = _make_payload(_TYPE_ORDER_FILLED, client_order_id=_CLIENT_OID)
    _fire_after(waiter, payload)

    result = waiter.wait()
    assert result is payload


def test_fill_matched_by_label() -> None:
    """ORDER_FILLED with matching label (clientOrderId absent) fires the waiter."""
    waiter = _ExecutionWaiter(_CLIENT_OID, timeout_s=1.0)
    payload = _make_payload(_TYPE_ORDER_FILLED, label=_CLIENT_OID)
    _fire_after(waiter, payload)

    result = waiter.wait()
    assert result is payload


def test_partial_fill_fires_waiter() -> None:
    """ORDER_PARTIAL_FILL (type 11) is in _SUCCESS_EXECUTION_TYPES and must fire."""
    assert _TYPE_ORDER_PARTIAL_FILL in _SUCCESS_EXECUTION_TYPES
    waiter = _ExecutionWaiter(_CLIENT_OID, timeout_s=1.0)
    payload = _make_payload(_TYPE_ORDER_PARTIAL_FILL, client_order_id=_CLIENT_OID)
    _fire_after(waiter, payload)

    result = waiter.wait()
    assert result is payload


# ---------------------------------------------------------------------------
# Two-phase path (ORDER_ACCEPTED → ORDER_FILLED)
# ---------------------------------------------------------------------------


def test_two_phase_order_accepted_then_filled() -> None:
    """ORDER_ACCEPTED must not fire; subsequent ORDER_FILLED must fire."""
    waiter = _ExecutionWaiter(_CLIENT_OID, timeout_s=1.0)

    accepted = _make_payload(
        _TYPE_ORDER_ACCEPTED,
        client_order_id=_CLIENT_OID,
        broker_order_id=_BROKER_OID,
    )
    filled = _make_payload(
        _TYPE_ORDER_FILLED,
        client_order_id=_CLIENT_OID,
        broker_order_id=_BROKER_OID,
    )

    def _sequence() -> None:
        time.sleep(0.01)
        waiter.handle(accepted)
        time.sleep(0.01)
        waiter.handle(filled)

    threading.Thread(target=_sequence, daemon=True).start()

    result = waiter.wait()
    assert result is filled  # Must be the fill, not the accept


def test_two_phase_broker_id_fallback_on_fill() -> None:
    """If ORDER_FILLED omits clientOrderId, broker orderId learned from
    ORDER_ACCEPTED must still match."""
    waiter = _ExecutionWaiter(_CLIENT_OID, timeout_s=1.0)

    # Phase 1: accept carries both ids
    accepted = _make_payload(
        _TYPE_ORDER_ACCEPTED,
        client_order_id=_CLIENT_OID,
        broker_order_id=_BROKER_OID,
    )
    # Phase 2: fill carries only broker orderId (clientOrderId stripped by broker)
    filled = _make_payload(
        _TYPE_ORDER_FILLED,
        client_order_id="",
        broker_order_id=_BROKER_OID,
    )

    def _sequence() -> None:
        time.sleep(0.01)
        waiter.handle(accepted)
        time.sleep(0.01)
        waiter.handle(filled)

    threading.Thread(target=_sequence, daemon=True).start()

    result = waiter.wait()
    assert result is filled


def test_broker_id_learned_before_fill_arrives() -> None:
    """Broker orderId stored from ORDER_ACCEPTED is available immediately for
    matching even if the fill arrives within microseconds."""
    waiter = _ExecutionWaiter(_CLIENT_OID, timeout_s=1.0)

    accepted = _make_payload(
        _TYPE_ORDER_ACCEPTED,
        client_order_id=_CLIENT_OID,
        broker_order_id=_BROKER_OID,
    )
    waiter.handle(accepted)  # synchronous — sets _broker_order_id

    # Internal state should now know the broker orderId
    assert waiter._broker_order_id == _BROKER_OID

    filled = _make_payload(
        _TYPE_ORDER_FILLED,
        client_order_id="",
        broker_order_id=_BROKER_OID,
    )
    waiter.handle(filled)

    result = waiter.wait()
    assert result is filled


# ---------------------------------------------------------------------------
# Rejection — no identifiers
# ---------------------------------------------------------------------------


def test_no_identifiers_rejected() -> None:
    """Events with no label, clientOrderId, or orderId must never fire."""
    waiter = _ExecutionWaiter(_CLIENT_OID, timeout_s=0.05)
    # Payload with order field but all identifier strings are empty
    payload = _make_payload(_TYPE_ORDER_FILLED, client_order_id="", broker_order_id="", label="")
    waiter.handle(payload)

    result = waiter.wait()
    assert result is None  # Timed out — correctly rejected


def test_no_order_field_rejected() -> None:
    """Payload without any ``order`` subfield must be rejected."""
    waiter = _ExecutionWaiter(_CLIENT_OID, timeout_s=0.05)
    payload = _make_payload_no_order(_TYPE_ORDER_FILLED)
    waiter.handle(payload)

    result = waiter.wait()
    assert result is None


# ---------------------------------------------------------------------------
# Rejection — wrong identifiers
# ---------------------------------------------------------------------------


def test_wrong_client_order_id_rejected() -> None:
    """Events with a different clientOrderId must not fire our waiter."""
    waiter = _ExecutionWaiter(_CLIENT_OID, timeout_s=0.05)
    wrong = _make_payload(_TYPE_ORDER_FILLED, client_order_id="RENKO-99999999")
    waiter.handle(wrong)

    result = waiter.wait()
    assert result is None


def test_wrong_broker_order_id_rejected() -> None:
    """After learning our broker orderId, a fill for a different orderId is rejected."""
    waiter = _ExecutionWaiter(_CLIENT_OID, timeout_s=0.05)

    accepted = _make_payload(
        _TYPE_ORDER_ACCEPTED,
        client_order_id=_CLIENT_OID,
        broker_order_id=_BROKER_OID,
    )
    waiter.handle(accepted)

    other_fill = _make_payload(
        _TYPE_ORDER_FILLED,
        client_order_id="",
        broker_order_id="1111111",  # different order
    )
    waiter.handle(other_fill)

    result = waiter.wait()
    assert result is None


# ---------------------------------------------------------------------------
# Non-fill execution types are skipped
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("exec_type", [_TYPE_ORDER_REJECTED, _TYPE_ORDER_EXPIRED, 4, 5, 9, 10])
def test_non_fill_exec_types_do_not_fire(exec_type: int) -> None:
    """Types not in _SUCCESS_EXECUTION_TYPES must not fire the waiter."""
    assert exec_type not in _SUCCESS_EXECUTION_TYPES, (
        f"Type {exec_type} is in _SUCCESS_EXECUTION_TYPES — update this test"
    )
    waiter = _ExecutionWaiter(_CLIENT_OID, timeout_s=0.05)
    payload = _make_payload(exec_type, client_order_id=_CLIENT_OID)
    waiter.handle(payload)

    result = waiter.wait()
    assert result is None


def test_order_accepted_alone_times_out() -> None:
    """ORDER_ACCEPTED matched by clientOrderId must NOT fire the waiter by itself."""
    waiter = _ExecutionWaiter(_CLIENT_OID, timeout_s=0.05)
    payload = _make_payload(
        _TYPE_ORDER_ACCEPTED,
        client_order_id=_CLIENT_OID,
        broker_order_id=_BROKER_OID,
    )
    waiter.handle(payload)

    result = waiter.wait()
    assert result is None


# ---------------------------------------------------------------------------
# Timeout
# ---------------------------------------------------------------------------


def test_timeout_returns_none() -> None:
    """If no matching event arrives, wait() returns None after timeout."""
    waiter = _ExecutionWaiter(_CLIENT_OID, timeout_s=0.05)
    result = waiter.wait()
    assert result is None


# ---------------------------------------------------------------------------
# Concurrent false-positive guard
# ---------------------------------------------------------------------------


def test_unrelated_events_before_ours_do_not_fire() -> None:
    """Multiple unrelated events must all be rejected; our fill still fires."""
    waiter = _ExecutionWaiter(_CLIENT_OID, timeout_s=1.0)

    def _sequence() -> None:
        time.sleep(0.01)
        # Three unrelated fills
        for i in range(3):
            waiter.handle(
                _make_payload(_TYPE_ORDER_FILLED, client_order_id=f"OTHER-{i:08d}")
            )
        # Correct fill arrives last
        time.sleep(0.01)
        waiter.handle(_make_payload(_TYPE_ORDER_FILLED, client_order_id=_CLIENT_OID))

    threading.Thread(target=_sequence, daemon=True).start()

    result = waiter.wait()
    assert result is not None
    assert result.order.clientOrderId == _CLIENT_OID


# ---------------------------------------------------------------------------
# Thread safety
# ---------------------------------------------------------------------------


def test_handle_from_background_thread_safe() -> None:
    """Handler called concurrently from many threads must not corrupt state."""
    waiter = _ExecutionWaiter(_CLIENT_OID, timeout_s=1.0)
    correct = _make_payload(_TYPE_ORDER_FILLED, client_order_id=_CLIENT_OID)

    errors: list[Exception] = []

    def _spam_wrong() -> None:
        for _ in range(50):
            try:
                waiter.handle(_make_payload(_TYPE_ORDER_FILLED, client_order_id="NOISE"))
            except Exception as exc:
                errors.append(exc)

    def _send_correct() -> None:
        time.sleep(0.02)
        waiter.handle(correct)

    threads = [threading.Thread(target=_spam_wrong, daemon=True) for _ in range(4)]
    threads.append(threading.Thread(target=_send_correct, daemon=True))
    for t in threads:
        t.start()

    result = waiter.wait()

    assert not errors, f"Exceptions in handler threads: {errors}"
    assert result is correct
