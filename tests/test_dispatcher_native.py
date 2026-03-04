#!/usr/bin/env python3
"""
Quick test of the native cTrader dispatcher.
Tests the execution flow without full trading engine.
"""

import os
import sys
import threading
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# Test imports
try:
    from kinetra.renko.ctrader_dispatcher import (
        CTraderBarProvider,
        CTraderOrderDispatcher,
        CTraderOrderResult,
        ExecutionEventHandler,
        OperationRegistry,
        PendingOperation,
    )

    print("✓ Imports successful")
except ImportError as e:
    print(f"✗ Import failed: {e}")
    sys.exit(1)

# Test registry
try:
    registry = OperationRegistry()
    op = PendingOperation(
        operation_id="test-123",
        operation_type="open",
        symbol="XAUUSD",
        client_order_id="kt-000001-1234",
    )
    registry.register(op)

    # Test find by client_order_id
    found = registry.find(client_order_id="kt-000001-1234")
    assert found is not None
    assert found.operation_id == "test-123"

    # Test update IDs
    registry.update_ids(op, position_id="12345")
    found2 = registry.find(position_id="12345")
    assert found2 is not None

    # Test complete
    registry.complete(op, success=True)
    assert op.success is True
    assert op.event.is_set()

    print("✓ OperationRegistry works")
except Exception as e:
    print(f"✗ Registry test failed: {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)

# Test execution handler
try:
    registry2 = OperationRegistry()
    handler = ExecutionEventHandler(registry2)

    # Mock payload
    class MockOrder:
        clientOrderId = "kt-000002-5678"
        orderId = "98765"
        positionId = "55555"

    class MockPayload:
        executionType = 3  # ORDER_FILLED
        order = MockOrder()
        position = None
        deal = None

    op2 = PendingOperation(
        operation_id="kt-000002-5678",
        operation_type="open",
        symbol="XAUUSD",
        client_order_id="kt-000002-5678",
    )
    registry2.register(op2)

    # Call handler
    handler(MockPayload())

    # Check if operation was completed
    if op2.event.wait(timeout=1.0):
        print("✓ ExecutionEventHandler works")
    else:
        print("✗ ExecutionEventHandler did not complete operation")
        sys.exit(1)
except Exception as e:
    print(f"✗ Handler test failed: {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 60)
print("All tests passed!")
print("=" * 60)
print("\nThe native dispatcher structure is working.")
print("Next: Test with live connection (requires credentials)")
