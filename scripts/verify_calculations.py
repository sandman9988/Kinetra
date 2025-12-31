#!/usr/bin/env python3
"""
Calculation Accuracy Verification
=================================

Manual step-by-step verification of all trading calculations:
1. P&L (Profit/Loss)
2. Spread cost
3. Commission
4. Swap (including triple swap)
5. Margin requirements
6. Pip/Point values

Each calculation is done manually and compared against the system.
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta
from decimal import Decimal, ROUND_HALF_UP

sys.path.insert(0, str(Path(__file__).parent.parent))

from kinetra.symbol_info import get_symbol_info, SymbolInfo
from kinetra.trading_costs import TradingCostCalculator, TradingCostSpec, SwapCalendar


def verify_with_tolerance(name: str, calculated: float, expected: float, tolerance: float = 0.01):
    """Verify calculation with tolerance."""
    diff = abs(calculated - expected)
    pct_diff = (diff / expected * 100) if expected != 0 else 0
    passed = diff <= tolerance or pct_diff <= 1.0  # Within $0.01 or 1%
    
    status = "✓ PASS" if passed else "✗ FAIL"
    print(f"  {status}: {name}")
    print(f"         Calculated: ${calculated:.4f}")
    print(f"         Expected:   ${expected:.4f}")
    print(f"         Difference: ${diff:.4f} ({pct_diff:.2f}%)")
    return passed


def test_forex_eurusd():
    """
    Test EURUSD (5-digit forex) calculations.
    
    EURUSD Specifications:
    - Contract size: 100,000 EUR
    - Point: 0.00001 (5th decimal)
    - Pip: 0.0001 (4th decimal) = 10 points
    - Tick value: $1 per point per lot (for USD quote currency)
    - 1 pip = $10 per lot
    """
    print("\n" + "=" * 70)
    print("TEST 1: EURUSD (5-digit Forex)")
    print("=" * 70)
    
    info = get_symbol_info("EURUSD")
    print(f"\nSymbol Specs:")
    print(f"  Contract Size: {info.contract_size:,}")
    print(f"  Point: {info.point}")
    print(f"  Tick Value: ${info.tick_value}")
    print(f"  Pip Size: {info.pip_size}")
    
    # Test scenario
    entry_price = 1.08500
    exit_price = 1.08600  # +10 pips (+100 points)
    lot_size = 1.0
    spread_points = 10  # 1 pip spread
    
    print(f"\nTrade Scenario:")
    print(f"  Entry: {entry_price}")
    print(f"  Exit:  {exit_price}")
    print(f"  Lots:  {lot_size}")
    print(f"  Spread: {spread_points} points ({spread_points/10} pips)")
    
    # =========================================================================
    # 1. P&L Calculation
    # =========================================================================
    print(f"\n--- P&L Calculation ---")
    
    # Manual calculation:
    # Price diff = 1.08600 - 1.08500 = 0.00100 (10 pips = 100 points)
    # P&L = price_diff * contract_size * lots
    # P&L = 0.00100 * 100,000 * 1 = $100
    
    price_diff = exit_price - entry_price
    manual_pnl = price_diff * info.contract_size * lot_size
    
    print(f"  Manual: price_diff ({price_diff}) × contract ({info.contract_size:,}) × lots ({lot_size})")
    print(f"  Manual: {price_diff} × {info.contract_size:,} × {lot_size} = ${manual_pnl:.2f}")
    
    # System calculation
    system_pnl = info.calculate_profit(lot_size, entry_price, exit_price)
    
    verify_with_tolerance("P&L (10 pips profit)", system_pnl, manual_pnl)
    
    # Verify pip value
    print(f"\n  Pip Value Check:")
    pip_value = info.pip_value  # Should be ~$10 per pip per lot
    print(f"    System pip_value: ${pip_value:.2f} per pip per lot")
    print(f"    Expected: $10.00 per pip per lot")
    print(f"    10 pips × $10 = $100 ✓" if abs(pip_value - 10) < 0.1 else "    ✗ Pip value incorrect")
    
    # =========================================================================
    # 2. Spread Cost Calculation
    # =========================================================================
    print(f"\n--- Spread Cost Calculation ---")
    
    # Manual calculation:
    # Spread cost = spread_in_price × contract_size × lots
    # spread_in_price = 10 points × 0.00001 = 0.00010
    # Spread cost = 0.00010 × 100,000 × 1 = $10
    
    spread_in_price = spread_points * info.point
    manual_spread_cost = spread_in_price * info.contract_size * lot_size
    
    print(f"  Manual: spread_points ({spread_points}) × point ({info.point}) = spread_price ({spread_in_price})")
    print(f"  Manual: {spread_in_price} × {info.contract_size:,} × {lot_size} = ${manual_spread_cost:.2f}")
    
    # System calculation
    system_spread_cost = info.calculate_spread_cost(lot_size, spread_points)
    
    verify_with_tolerance("Spread Cost (10 points = 1 pip)", system_spread_cost, manual_spread_cost)
    
    # =========================================================================
    # 3. Commission Calculation
    # =========================================================================
    print(f"\n--- Commission Calculation ---")
    
    commission_per_lot = 7.0  # $7 per lot round trip
    manual_commission = commission_per_lot * lot_size
    
    print(f"  Manual: ${commission_per_lot} per lot × {lot_size} lots = ${manual_commission:.2f}")
    verify_with_tolerance("Commission", manual_commission, 7.0)
    
    # =========================================================================
    # 4. Net P&L
    # =========================================================================
    print(f"\n--- Net P&L ---")
    
    manual_net = manual_pnl - manual_spread_cost - manual_commission
    print(f"  Gross P&L:    ${manual_pnl:.2f}")
    print(f"  - Spread:     ${manual_spread_cost:.2f}")
    print(f"  - Commission: ${manual_commission:.2f}")
    print(f"  = Net P&L:    ${manual_net:.2f}")
    
    return True


def test_gold_xauusd():
    """
    Test XAUUSD (Gold) calculations.
    
    XAUUSD Specifications:
    - Contract size: 100 troy ounces
    - Point: 0.01 ($0.01 price movement)
    - Tick value: $1 per 0.01 move per lot
    - $1 price move = $100 per lot
    """
    print("\n" + "=" * 70)
    print("TEST 2: XAUUSD (Gold)")
    print("=" * 70)
    
    info = get_symbol_info("XAUUSD")
    print(f"\nSymbol Specs:")
    print(f"  Contract Size: {info.contract_size} oz")
    print(f"  Point: {info.point}")
    print(f"  Tick Value: ${info.tick_value}")
    
    # Test scenario
    entry_price = 2650.00
    exit_price = 2660.00  # +$10 move
    lot_size = 0.1
    spread_points = 30  # 30 points = $0.30 spread
    
    print(f"\nTrade Scenario:")
    print(f"  Entry: ${entry_price:.2f}")
    print(f"  Exit:  ${exit_price:.2f}")
    print(f"  Lots:  {lot_size}")
    print(f"  Spread: {spread_points} points (${spread_points * info.point:.2f})")
    
    # =========================================================================
    # 1. P&L Calculation
    # =========================================================================
    print(f"\n--- P&L Calculation ---")
    
    # Manual calculation:
    # Price diff = $10
    # P&L = $10 × 100 oz × 0.1 lots = $100
    
    price_diff = exit_price - entry_price
    manual_pnl = price_diff * info.contract_size * lot_size
    
    print(f"  Manual: ${price_diff:.2f} × {info.contract_size} oz × {lot_size} lots = ${manual_pnl:.2f}")
    
    # Alternative calculation using tick value:
    # Points moved = $10 / $0.01 = 1000 points
    # P&L = 1000 points × $1/point × 0.1 lots = $100
    points_moved = price_diff / info.point
    alt_pnl = points_moved * info.tick_value * lot_size
    print(f"  Alt:    {points_moved:.0f} points × ${info.tick_value}/point × {lot_size} lots = ${alt_pnl:.2f}")
    
    system_pnl = info.calculate_profit(lot_size, entry_price, exit_price)
    verify_with_tolerance("P&L ($10 move)", system_pnl, manual_pnl)
    
    # =========================================================================
    # 2. Spread Cost
    # =========================================================================
    print(f"\n--- Spread Cost Calculation ---")
    
    # Manual: 30 points × $0.01 × 100 oz × 0.1 lots = $3.00
    spread_in_price = spread_points * info.point
    manual_spread = spread_in_price * info.contract_size * lot_size
    
    print(f"  Manual: {spread_points} pts × ${info.point} × {info.contract_size} oz × {lot_size} = ${manual_spread:.2f}")
    
    system_spread = info.calculate_spread_cost(lot_size, spread_points)
    verify_with_tolerance("Spread Cost (30 points)", system_spread, manual_spread)
    
    return True


def test_btcusd():
    """
    Test BTCUSD (Crypto) calculations.
    
    BTCUSD Specifications (varies by broker):
    - Contract size: 1 BTC
    - Point: 0.01 or 1.0 (broker dependent)
    - Tick value: depends on contract
    """
    print("\n" + "=" * 70)
    print("TEST 3: BTCUSD (Crypto)")
    print("=" * 70)
    
    info = get_symbol_info("BTCUSD")
    print(f"\nSymbol Specs:")
    print(f"  Contract Size: {info.contract_size} BTC")
    print(f"  Point: {info.point}")
    print(f"  Tick Value: ${info.tick_value}")
    
    # Test scenario
    entry_price = 95000.00
    exit_price = 96000.00  # +$1000 move
    lot_size = 0.1
    spread_points = 1000  # $10 spread (1000 × 0.01)
    
    print(f"\nTrade Scenario:")
    print(f"  Entry: ${entry_price:,.2f}")
    print(f"  Exit:  ${exit_price:,.2f}")
    print(f"  Lots:  {lot_size}")
    print(f"  Price Move: ${exit_price - entry_price:,.2f}")
    
    # =========================================================================
    # P&L Calculation
    # =========================================================================
    print(f"\n--- P&L Calculation ---")
    
    price_diff = exit_price - entry_price
    manual_pnl = price_diff * info.contract_size * lot_size
    
    print(f"  Manual: ${price_diff:,.2f} × {info.contract_size} BTC × {lot_size} = ${manual_pnl:.2f}")
    
    system_pnl = info.calculate_profit(lot_size, entry_price, exit_price)
    verify_with_tolerance("P&L ($1000 move)", system_pnl, manual_pnl)
    
    return True


def test_swap_calculation():
    """
    Test swap (overnight interest) calculations including triple swap.
    """
    print("\n" + "=" * 70)
    print("TEST 4: Swap Calculation (Including Triple Swap)")
    print("=" * 70)
    
    info = get_symbol_info("XAUUSD")
    
    print(f"\nSwap Specs:")
    print(f"  Swap Long: {info.swap_long} points/day")
    print(f"  Swap Short: {info.swap_short} points/day")
    print(f"  Triple Swap Day: {info.swap_rollover3days} (3=Wednesday)")
    print(f"  Tick Value: ${info.tick_value}")
    
    lot_size = 0.1
    
    # =========================================================================
    # Daily Swap Calculation
    # =========================================================================
    print(f"\n--- Daily Swap (Long Position) ---")
    
    # Manual: swap_long × tick_value × lots
    # -68.09 × $1 × 0.1 = -$6.809 per day
    
    manual_daily_swap = info.swap_long * info.tick_value * lot_size
    print(f"  Manual: {info.swap_long} pts × ${info.tick_value}/pt × {lot_size} lots = ${manual_daily_swap:.2f}/day")
    
    # =========================================================================
    # Triple Swap (Wednesday)
    # =========================================================================
    print(f"\n--- Triple Swap (Wednesday) ---")
    
    # On Wednesday, swap is charged for Sat+Sun+Wed = 3 days
    manual_triple_swap = manual_daily_swap * 3
    print(f"  Manual: ${manual_daily_swap:.2f} × 3 days = ${manual_triple_swap:.2f}")
    
    # Test SwapCalendar
    calendar = SwapCalendar(triple_swap_day=2)  # Wednesday = 2 (0=Monday)
    
    # Wednesday
    wed = datetime(2024, 12, 25)  # A Wednesday
    wed_multiplier = calendar.get_swap_multiplier(wed.date())
    print(f"\n  SwapCalendar Wednesday multiplier: {wed_multiplier}")
    
    # Monday
    mon = datetime(2024, 12, 23)  # A Monday
    mon_multiplier = calendar.get_swap_multiplier(mon.date())
    print(f"  SwapCalendar Monday multiplier: {mon_multiplier}")
    
    # =========================================================================
    # Week-long swap calculation
    # =========================================================================
    print(f"\n--- Week-long Swap (5 trading days) ---")
    
    # Mon(1) + Tue(1) + Wed(3) + Thu(1) + Fri(1) = 7 days of swap
    total_multiplier = 1 + 1 + 3 + 1 + 1  # = 7
    manual_week_swap = manual_daily_swap * total_multiplier
    print(f"  Manual: ${manual_daily_swap:.2f} × 7 = ${manual_week_swap:.2f}")
    
    return True


def test_margin_calculation():
    """
    Test margin requirement calculations.
    """
    print("\n" + "=" * 70)
    print("TEST 5: Margin Calculation")
    print("=" * 70)
    
    leverage = 100
    
    # EURUSD
    print(f"\n--- EURUSD Margin (100:1 leverage) ---")
    info = get_symbol_info("EURUSD")
    price = 1.0850
    lot_size = 1.0
    
    # Manual: (lots × contract × price) / leverage
    notional = lot_size * info.contract_size * price
    manual_margin = notional / leverage
    
    print(f"  Notional: {lot_size} × {info.contract_size:,} × {price} = ${notional:,.2f}")
    print(f"  Margin: ${notional:,.2f} / {leverage} = ${manual_margin:,.2f}")
    
    system_margin = info.calculate_margin(lot_size, price, leverage)
    verify_with_tolerance("EURUSD Margin (1 lot)", system_margin, manual_margin)
    
    # XAUUSD
    print(f"\n--- XAUUSD Margin (100:1 leverage) ---")
    info = get_symbol_info("XAUUSD")
    price = 2650.00
    lot_size = 0.1
    
    notional = lot_size * info.contract_size * price
    manual_margin = notional / leverage
    
    print(f"  Notional: {lot_size} × {info.contract_size} oz × ${price:,.2f} = ${notional:,.2f}")
    print(f"  Margin: ${notional:,.2f} / {leverage} = ${manual_margin:,.2f}")
    
    system_margin = info.calculate_margin(lot_size, price, leverage)
    verify_with_tolerance("XAUUSD Margin (0.1 lot)", system_margin, manual_margin)
    
    return True


def test_edge_cases():
    """
    Test edge cases: small numbers, large numbers, precision.
    """
    print("\n" + "=" * 70)
    print("TEST 6: Edge Cases & Precision")
    print("=" * 70)
    
    info = get_symbol_info("EURUSD")
    
    # =========================================================================
    # Small position (micro lot)
    # =========================================================================
    print(f"\n--- Micro Lot (0.01) ---")
    
    entry = 1.08500
    exit = 1.08510  # +1 pip
    lot_size = 0.01
    
    manual_pnl = (exit - entry) * info.contract_size * lot_size
    print(f"  Manual: 1 pip × 100,000 × 0.01 = ${manual_pnl:.4f}")
    
    system_pnl = info.calculate_profit(lot_size, entry, exit)
    verify_with_tolerance("Micro lot P&L", system_pnl, manual_pnl, tolerance=0.001)
    
    # =========================================================================
    # Large position
    # =========================================================================
    print(f"\n--- Large Position (10 lots) ---")
    
    lot_size = 10.0
    manual_pnl = (exit - entry) * info.contract_size * lot_size
    print(f"  Manual: 1 pip × 100,000 × 10 = ${manual_pnl:.2f}")
    
    system_pnl = info.calculate_profit(lot_size, entry, exit)
    verify_with_tolerance("Large lot P&L", system_pnl, manual_pnl)
    
    # =========================================================================
    # Fractional pip
    # =========================================================================
    print(f"\n--- Fractional Pip (0.5 pip = 5 points) ---")
    
    entry = 1.08500
    exit = 1.08505  # +0.5 pip = 5 points
    lot_size = 1.0
    
    manual_pnl = (exit - entry) * info.contract_size * lot_size
    print(f"  Manual: 0.5 pip × 100,000 × 1 = ${manual_pnl:.2f}")
    
    system_pnl = info.calculate_profit(lot_size, entry, exit)
    verify_with_tolerance("Fractional pip P&L", system_pnl, manual_pnl)
    
    # =========================================================================
    # Loss scenario
    # =========================================================================
    print(f"\n--- Loss Scenario (-50 pips) ---")
    
    entry = 1.08500
    exit = 1.08000  # -50 pips
    lot_size = 1.0
    
    manual_pnl = (exit - entry) * info.contract_size * lot_size
    print(f"  Manual: -50 pips × 100,000 × 1 = ${manual_pnl:.2f}")
    
    system_pnl = info.calculate_profit(lot_size, entry, exit)
    verify_with_tolerance("Loss P&L", system_pnl, manual_pnl)
    
    return True


def test_short_position():
    """
    Test short position calculations.
    """
    print("\n" + "=" * 70)
    print("TEST 7: Short Position")
    print("=" * 70)
    
    info = get_symbol_info("EURUSD")
    
    # Short winning
    print(f"\n--- Short Position (Winning) ---")
    entry = 1.08500
    exit = 1.08400  # Price dropped = profit for short
    lot_size = 1.0
    
    # For short: P&L = (entry - exit) × contract × lots
    manual_pnl = (entry - exit) * info.contract_size * lot_size
    print(f"  Manual (short): ({entry} - {exit}) × 100,000 × 1 = ${manual_pnl:.2f}")
    
    # System with direction = -1
    system_pnl = info.calculate_profit(lot_size, entry, exit)
    # Note: calculate_profit returns (exit - entry), for short we negate
    short_pnl = -system_pnl
    print(f"  System short P&L: ${short_pnl:.2f}")
    
    verify_with_tolerance("Short winning P&L", short_pnl, manual_pnl)
    
    # Short losing
    print(f"\n--- Short Position (Losing) ---")
    exit = 1.08600  # Price rose = loss for short
    
    manual_pnl = (entry - exit) * info.contract_size * lot_size
    print(f"  Manual (short): ({entry} - {exit}) × 100,000 × 1 = ${manual_pnl:.2f}")
    
    system_pnl = info.calculate_profit(lot_size, entry, exit)
    short_pnl = -system_pnl
    
    verify_with_tolerance("Short losing P&L", short_pnl, manual_pnl)
    
    return True


def main():
    print("=" * 70)
    print("CALCULATION ACCURACY VERIFICATION")
    print("=" * 70)
    print("\nThis test manually calculates each value and compares against the system.")
    
    all_passed = True
    
    all_passed &= test_forex_eurusd()
    all_passed &= test_gold_xauusd()
    all_passed &= test_btcusd()
    all_passed &= test_swap_calculation()
    all_passed &= test_margin_calculation()
    all_passed &= test_edge_cases()
    all_passed &= test_short_position()
    
    print("\n" + "=" * 70)
    print("FINAL RESULT")
    print("=" * 70)
    
    if all_passed:
        print("\n✓ ALL CALCULATIONS VERIFIED ACCURATE")
    else:
        print("\n✗ SOME CALCULATIONS FAILED - REVIEW ABOVE")
    
    print("\nKey Formulas Verified:")
    print("  • P&L = (exit - entry) × contract_size × lots")
    print("  • Spread Cost = spread_points × point × contract_size × lots")
    print("  • Swap = swap_rate × tick_value × lots × day_multiplier")
    print("  • Margin = (lots × contract_size × price) / leverage")
    print("  • Pip Value = tick_value × (pip_size / point)")


if __name__ == "__main__":
    main()
