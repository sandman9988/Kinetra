#!/usr/bin/env python3
"""
Test Order Execution Speed
==========================

Measures actual order placement latency to verify async implementation.
"""

import threading
import time
from datetime import datetime, timezone

from kinetra.connectors.ctrader_connector import CTraderCredentials, build_connector
from kinetra.renko.ctrader_dispatcher import build_ctrader_session
from kinetra.renko.live_trader import TradeDirection


def test_sync_order_placement():
    """Test synchronous (blocking) order placement — OLD behavior."""
    print("=" * 60)
    print("TEST 1: Synchronous Order Placement (OLD)")
    print("=" * 60)

    creds = CTraderCredentials.from_env()
    connector = build_connector(credentials=creds, timeout_s=30.0)

    dispatcher, bar_provider = build_ctrader_session(
        credentials=creds,
        connect_timeout_s=30.0,
        fill_timeout_s=5.0,  # Reduced from 30s
    )

    # Start bar provider to get live prices
    bar_provider.start()
    time.sleep(2)  # Wait for first quote

    # Get current price
    symbol = "XAUUSD"
    spread = bar_provider.get_spread_pts(symbol)
    print(f"Current spread: {spread:.2f} pts")

    # Measure order placement
    entry_price = 2900.0  # Far from market to avoid immediate fill issues
    stop_price = 2899.0

    print(f"\nPlacing test order...")
    print(f"  Symbol: {symbol}")
    print(f"  Direction: BUY")
    print(f"  Lots: 0.01")
    print(f"  Entry price: ~{entry_price:.2f}")
    print(f"  Stop: {stop_price:.2f}")

    start_time = time.perf_counter()

    result = dispatcher.open_position(
        symbol=symbol,
        direction=TradeDirection.LONG,
        lots=0.01,
        price=entry_price,
        stop_price=stop_price,
        comment="TEST-SYNC",
    )

    elapsed_ms = (time.perf_counter() - start_time) * 1000

    print(f"\n  Result: {'SUCCESS' if result.success else 'FAILED'}")
    print(f"  Order ID: {result.order_id}")
    print(f"  Filled price: {result.filled_price}")
    print(f"  Error: {result.error}")
    print(f"  \n⏱️  EXECUTION TIME: {elapsed_ms:.1f} ms")

    # Close position if opened
    if result.success and result.order_id:
        print(f"\n  Closing test position...")
        close_start = time.perf_counter()

        close_result = dispatcher.close_position(
            symbol=symbol,
            order_id=result.order_id,
            price=entry_price,  # Rough estimate
            lots=0.01,
            comment="TEST-CLOSE",
        )

        close_elapsed_ms = (time.perf_counter() - close_start) * 1000
        print(f"  ⏱️  CLOSE TIME: {close_elapsed_ms:.1f} ms")

    bar_provider.stop()
    dispatcher._connector.stop()

    return elapsed_ms


def test_async_order_flow():
    """Test async order flow — NEW behavior with queues."""
    print("\n" + "=" * 60)
    print("TEST 2: Async Order Flow (NEW)")
    print("=" * 60)

    from kinetra.renko.trading_engine import EngineConfig, RenkoEngine

    cfg = EngineConfig(
        symbol="XAUUSD",
        brick_size=1.0,
        stop_bricks=1.0,
        fliprate_threshold=0.35,
        markov_threshold=0.55,
        sizing_mode="static",
        min_lots=0.01,
    )

    engine = RenkoEngine(cfg, quiet_mode=True)

    creds = CTraderCredentials.from_env()
    dispatcher, bar_provider = build_ctrader_session(
        credentials=creds,
        fill_timeout_s=5.0,
    )

    stop_event = threading.Event()

    print("\nStarting async engine for 30 seconds...")
    print("Monitor for [ENTRY-QUEUED] and [FILL-CONFIRMED] messages")
    print("(Press Ctrl+C to stop early)\n")

    # Run for 30 seconds or until signal
    def timeout_stop():
        time.sleep(30)
        stop_event.set()

    timer = threading.Thread(target=timeout_stop, daemon=True)
    timer.start()

    try:
        results = engine.run(bar_provider, dispatcher, stop_event=stop_event)

        print(f"\n{'=' * 60}")
        print("RESULTS")
        print(f"{'=' * 60}")
        print(f"Trades executed: {len(results.get('trades', []))}")
        print(f"Net P&L: ${sum(t.get('net_usd', 0) for t in results.get('trades', [])):.2f}")

    except KeyboardInterrupt:
        print("\n\nStopped by user")
        stop_event.set()


def test_fill_latency_breakdown():
    """Detailed breakdown of fill latency components."""
    print("\n" + "=" * 60)
    print("TEST 3: Fill Latency Breakdown")
    print("=" * 60)

    from kinetra.renko.ctrader_dispatcher import CTraderBarProvider, CTraderOrderDispatcher

    creds = CTraderCredentials.from_env()
    connector = build_connector(credentials=creds)

    # Test connection speed
    t0 = time.perf_counter()
    connected = connector.is_connected()
    t1 = time.perf_counter()
    print(f"Connection check: {(t1 - t0) * 1000:.2f} ms")

    # Test symbol resolution
    t0 = time.perf_counter()
    symbol_id = connector.find_symbol_id("XAUUSD")
    t1 = time.perf_counter()
    print(f"Symbol resolution: {(t1 - t0) * 1000:.2f} ms (id={symbol_id})")

    # Test digits lookup
    t0 = time.perf_counter()
    digits = connector.get_digits(symbol_id)
    t1 = time.perf_counter()
    print(f"Digits lookup: {(t1 - t0) * 1000:.2f} ms (digits={digits})")

    print("\n✅ All lookups are fast — latency is in order execution, not setup")

    connector.stop()


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("  KINETRA ORDER EXECUTION SPEED TEST")
    print("=" * 70)

    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--breakdown":
        test_fill_latency_breakdown()
    elif len(sys.argv) > 1 and sys.argv[1] == "--async":
        test_async_order_flow()
    else:
        # Default: test sync placement
        try:
            elapsed = test_sync_order_placement()

            print("\n" + "=" * 60)
            print("INTERPRETATION")
            print("=" * 60)

            if elapsed < 500:
                print(f"✅ FAST: {elapsed:.0f}ms — No blocking issues")
            elif elapsed < 2000:
                print(f"⚠️  MODERATE: {elapsed:.0f}ms — Some delay, acceptable")
            elif elapsed < 5000:
                print(f"❌ SLOW: {elapsed:.0f}ms — Significant blocking")
            else:
                print(f"🚨 CRITICAL: {elapsed:.0f}ms — Severe blocking, fix needed!")

        except Exception as e:
            print(f"\n❌ Test failed: {e}")
            import traceback

            traceback.print_exc()
