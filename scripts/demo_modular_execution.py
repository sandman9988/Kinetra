#!/usr/bin/env python3
"""
Demo: Modular Order Execution - Same Agent Code for Backtest and Live

Shows how dependency injection allows SAME agent code to run in
both backtest and live trading contexts.

Key insight: Agent doesn't know if it's in backtest or live!
OrderExecutor interface abstracts the difference.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np

from kinetra.order_executor import create_executor, OrderExecutor, OrderResult
from kinetra.market_microstructure import SymbolSpec, AssetClass


class SimpleAgent:
    """
    Example trading agent that works in BOTH backtest and live.

    Agent only knows about OrderExecutor interface,
    not whether it's BacktestExecutor or LiveExecutor!
    """

    def __init__(self, executor: OrderExecutor):
        """
        Initialize agent.

        Args:
            executor: OrderExecutor (backtest or live - agent doesn't care!)
        """
        self.executor = executor
        self.position_open = False

    def decide_and_execute(self) -> OrderResult:
        """
        Make trading decision and execute order.

        SAME code runs in backtest AND live!
        """
        # Get current market state
        current_price = self.executor.get_current_price()

        # Simple strategy: Random entry
        if not self.position_open and np.random.rand() > 0.95:
            # Open long position with safe SL/TP
            sl, tp = self.executor.validator.get_safe_sl_tp(
                price=current_price,
                direction=1,  # Long
                sl_distance_pips=20,
                tp_distance_pips=40,
            )

            result = self.executor.execute_order(
                action='open_long',
                volume=1.0,
                sl=sl,
                tp=tp,
                comment="SimpleAgent entry"
            )

            if result.success:
                self.position_open = True
                print(f"[Agent] Opened long @ {result.fill_price:.5f}, SL={result.actual_sl:.5f}")
            else:
                print(f"[Agent] Order rejected: {result.error_message}")

            return result

        # Close position randomly
        elif self.position_open and np.random.rand() > 0.9:
            result = self.executor.execute_order(
                action='close',
                comment="SimpleAgent exit"
            )

            if result.success:
                self.position_open = False
                print(f"[Agent] Closed @ {result.fill_price:.5f}")

            return result

        return OrderResult(success=False, error_message="No action")


def demo_backtest_mode():
    """Demo: Agent running in backtest mode."""
    print("="*70)
    print("DEMO 1: BACKTEST MODE")
    print("="*70)

    # Create synthetic data
    data = pd.DataFrame({
        'open': np.random.randn(100).cumsum() + 1.08500,
        'high': np.random.randn(100).cumsum() + 1.08510,
        'low': np.random.randn(100).cumsum() + 1.08490,
        'close': np.random.randn(100).cumsum() + 1.08500,
        'spread': np.random.randint(10, 20, 100),  # Dynamic spread!
    }, index=pd.date_range('2024-01-01', periods=100, freq='H'))

    # Create spec with realistic constraints
    spec = SymbolSpec(
        symbol="EURUSD",
        asset_class=AssetClass.FOREX,
        digits=5,
        trade_stops_level=15,  # Minimum 15 points
        trade_freeze_level=10,
    )

    # Create backtest executor
    executor = create_executor(
        spec=spec,
        mode='backtest',
        data=data
    )

    # Create agent (uses executor interface)
    agent = SimpleAgent(executor)

    print(f"\nRunning backtest with {len(data)} bars...")
    print(f"Spec: {spec.symbol}, Stops Level: {spec.trade_stops_level} points\n")

    # Run agent for 10 bars
    for i in range(10):
        result = agent.decide_and_execute()
        executor.step_forward()  # Backtest-specific method

    print("\n✓ Backtest complete")


def demo_live_mode():
    """Demo: Agent running in live mode (skeleton)."""
    print("\n" + "="*70)
    print("DEMO 2: LIVE MODE (Skeleton)")
    print("="*70)

    # Create spec (SAME as backtest!)
    spec = SymbolSpec(
        symbol="EURUSD",
        asset_class=AssetClass.FOREX,
        digits=5,
        trade_stops_level=15,
        trade_freeze_level=10,
    )

    print(f"\nIn live mode, you would:")
    print(f"1. Connect to MT5 via MetaApi SDK")
    print(f"2. Create LiveExecutor with MT5 connection")
    print(f"3. Create agent with LiveExecutor")
    print(f"4. Agent code is IDENTICAL to backtest!")

    print("\nExample code:")
    print("""
    # Connect to MT5
    from metaapi_cloud_sdk import MetaApi
    api = MetaApi(token='your-token')
    account = await api.metatrader_account_api.get_account('account-id')
    connection = account.get_streaming_connection()
    await connection.connect()

    # Create live executor (SAME interface as backtest!)
    executor = create_executor(
        spec=spec,
        mode='live',
        mt5_connection=connection
    )

    # Create agent (SAME code as backtest!)
    agent = SimpleAgent(executor)

    # Run agent (SAME code as backtest!)
    while True:
        result = agent.decide_and_execute()
        await asyncio.sleep(60)  # Run every minute
    """)

    print("\n✓ Live mode skeleton shown")


def demo_constraint_validation():
    """Demo: Constraint validation prevents invalid orders."""
    print("\n" + "="*70)
    print("DEMO 3: CONSTRAINT VALIDATION (Backtest & Live)")
    print("="*70)

    # Create data
    data = pd.DataFrame({
        'close': [1.08500] * 10,
        'spread': [15] * 10,
    }, index=pd.date_range('2024-01-01', periods=10, freq='H'))

    # Create spec with strict constraints
    spec = SymbolSpec(
        symbol="EURUSD",
        asset_class=AssetClass.FOREX,
        digits=5,
        trade_stops_level=20,  # STRICT: 20 points minimum
        trade_freeze_level=10,
    )

    executor = create_executor(spec=spec, mode='backtest', data=data)

    print(f"\nSpec constraints:")
    print(f"  Minimum stops level: {spec.trade_stops_level} points ({spec.trade_stops_level * spec.point:.5f})")

    # Test 1: Invalid SL (too close)
    print(f"\nTest 1: Invalid SL (too close)")
    current_price = 1.08500
    invalid_sl = 1.08485  # Only 15 points (< 20 minimum)

    result = executor.execute_order(
        action='open_long',
        sl=invalid_sl
    )

    if result.success:
        print(f"  ✗ Order accepted (BUG!)")
    else:
        print(f"  ✓ Order rejected: {result.error_message}")

    # Test 2: Valid SL (meets minimum)
    print(f"\nTest 2: Valid SL (meets minimum)")
    valid_sl = 1.08480  # 20 points (exactly minimum)

    result = executor.execute_order(
        action='open_long',
        sl=valid_sl
    )

    if result.success:
        print(f"  ✓ Order accepted, SL @ {result.actual_sl:.5f}")
    else:
        print(f"  ✗ Order rejected: {result.error_message}")

    # Test 3: Auto-adjust SL
    print(f"\nTest 3: Auto-adjust SL (validator fixes it)")

    # Recreate executor with auto_adjust=True
    from kinetra.order_validator import OrderValidator
    validator = OrderValidator(spec, auto_adjust_stops=True)

    executor_auto = create_executor(spec=spec, mode='backtest', data=data)
    executor_auto.validator = validator

    result = executor_auto.execute_order(
        action='open_long',
        sl=invalid_sl  # Too close, but validator will fix
    )

    if result.success:
        print(f"  ✓ Order accepted with AUTO-ADJUSTED SL")
        print(f"    Desired: {invalid_sl:.5f}")
        print(f"    Actual:  {result.actual_sl:.5f}")
    else:
        print(f"  ✗ Order rejected: {result.error_message}")

    print(f"\n✓ Constraint validation works!")


def main():
    """Run all demos."""
    print("\n" + "="*70)
    print("MODULAR ORDER EXECUTION DEMO")
    print("Same Agent Code → Backtest OR Live")
    print("="*70)

    demo_backtest_mode()
    demo_live_mode()
    demo_constraint_validation()

    print("\n" + "="*70)
    print("KEY TAKEAWAYS")
    print("="*70)

    print("""
1. AGENT IS CONTEXT-AGNOSTIC
   - Agent only knows OrderExecutor interface
   - Doesn't care if backtest or live
   - SAME code runs in both contexts

2. SHARED VALIDATION
   - OrderValidator used in BOTH backtest and live
   - Prevents sim-to-real gap
   - If it passes validation in backtest, it will in live

3. MODULAR ARCHITECTURE
   - Easy to switch between backtest and live
   - Easy to add new executors (paper trading, etc.)
   - Dependency injection pattern

4. REALISTIC BACKTESTING
   - Dynamic spread (from candle data)
   - Freeze zones (from SymbolSpec)
   - Stop validation (from SymbolSpec)
   - Slippage simulation

→ If agent works in realistic backtest, it WILL work in live!
    """)


if __name__ == "__main__":
    main()
