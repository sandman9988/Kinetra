#!/usr/bin/env python3
"""
Test MT5 Freeze Zones and Stops Levels Implementation

Verifies:
1. SymbolSpec has new fields (trade_stops_level, trade_freeze_level, etc.)
2. Validation methods work correctly
3. JSON serialization includes new fields
4. Data loader can read specs with new fields
"""

import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from kinetra.market_microstructure import SymbolSpec, AssetClass


def test_new_fields():
    """Test that SymbolSpec has new freeze zone fields."""
    print("="*80)
    print("TEST 1: New Fields in SymbolSpec")
    print("="*80)

    spec = SymbolSpec(
        symbol="EURUSD",
        asset_class=AssetClass.FOREX,
        digits=5,
        trade_stops_level=15,  # 15 points minimum
        trade_freeze_level=10,  # 10 points freeze zone
        trade_mode="FULL",
        filling_mode="IOC",
        order_mode="MARKET_LIMIT",
        order_gtc_mode="GTC"
    )

    print(f"\nCreated EURUSD spec:")
    print(f"  Symbol: {spec.symbol}")
    print(f"  Digits: {spec.digits}")
    print(f"  Point: {spec.point}")
    print(f"  Trade Stops Level: {spec.trade_stops_level} points")
    print(f"  Trade Freeze Level: {spec.trade_freeze_level} points")
    print(f"  Trade Mode: {spec.trade_mode}")
    print(f"  Filling Mode: {spec.filling_mode}")
    print(f"  Order Mode: {spec.order_mode}")
    print(f"  Order GTC Mode: {spec.order_gtc_mode}")

    # Verify fields exist
    assert hasattr(spec, 'trade_stops_level'), "Missing trade_stops_level field"
    assert hasattr(spec, 'trade_freeze_level'), "Missing trade_freeze_level field"
    assert hasattr(spec, 'trade_mode'), "Missing trade_mode field"
    assert hasattr(spec, 'filling_mode'), "Missing filling_mode field"
    assert hasattr(spec, 'order_mode'), "Missing order_mode field"
    assert hasattr(spec, 'order_gtc_mode'), "Missing order_gtc_mode field"

    print("\n✅ TEST PASSED: All new fields present\n")
    return True


def test_stop_distance_validation():
    """Test stop distance validation method."""
    print("="*80)
    print("TEST 2: Stop Distance Validation")
    print("="*80)

    # Create spec with 15-point minimum stop distance
    spec = SymbolSpec(
        symbol="EURUSD",
        asset_class=AssetClass.FOREX,
        digits=5,
        trade_stops_level=15  # 15 points = 0.00015
    )

    current_price = 1.08500

    # Test cases
    test_cases = [
        (1.08490, False, "10 points - TOO CLOSE"),
        (1.08485, True, "15 points - EXACTLY at minimum"),
        (1.08480, True, "20 points - SAFE"),
        (1.08520, True, "20 points above - SAFE"),
    ]

    print(f"\nCurrent price: {current_price}")
    print(f"Minimum stop distance: {spec.trade_stops_level} points ({spec.trade_stops_level * spec.point:.5f})\n")

    for stop_price, should_pass, description in test_cases:
        is_valid, error_msg = spec.validate_stop_distance(current_price, stop_price)

        distance_points = int(abs(stop_price - current_price) / spec.point)
        status = "✓ VALID" if is_valid else "✗ INVALID"

        print(f"  SL @ {stop_price:.5f} ({distance_points} points): {status} - {description}")

        if error_msg:
            print(f"    Error: {error_msg}")

        # Verify expected result
        if should_pass and not is_valid:
            print(f"    ❌ TEST FAILED: Expected VALID but got INVALID")
            return False
        elif not should_pass and is_valid:
            print(f"    ❌ TEST FAILED: Expected INVALID but got VALID")
            return False

    print("\n✅ TEST PASSED: Stop distance validation working correctly\n")
    return True


def test_safe_stop_distance():
    """Test safe stop distance calculation."""
    print("="*80)
    print("TEST 3: Safe Stop Distance Calculation")
    print("="*80)

    spec = SymbolSpec(
        symbol="EURUSD",
        asset_class=AssetClass.FOREX,
        digits=5,
        trade_stops_level=15
    )

    # Default safety multiplier (1.5x)
    safe_distance = spec.get_safe_stop_distance()
    expected_distance = 15 * spec.point * 1.5

    print(f"\nMinimum stop distance: {spec.trade_stops_level} points ({spec.trade_stops_level * spec.point:.5f})")
    print(f"Safe stop distance (1.5x): {safe_distance:.5f}")
    print(f"Expected: {expected_distance:.5f}")

    assert abs(safe_distance - expected_distance) < 1e-9, "Safe distance calculation incorrect"

    # Custom safety multiplier (2.0x)
    safe_distance_2x = spec.get_safe_stop_distance(safety_multiplier=2.0)
    expected_2x = 15 * spec.point * 2.0

    print(f"\nSafe stop distance (2.0x): {safe_distance_2x:.5f}")
    print(f"Expected: {expected_2x:.5f}")

    assert abs(safe_distance_2x - expected_2x) < 1e-9, "Safe distance 2x calculation incorrect"

    print("\n✅ TEST PASSED: Safe stop distance calculations correct\n")
    return True


def test_json_serialization():
    """Test that new fields are included in JSON serialization."""
    print("="*80)
    print("TEST 4: JSON Serialization")
    print("="*80)

    from kinetra.mt5_spec_extractor import MT5SpecExtractor

    # Create spec with all new fields
    spec = SymbolSpec(
        symbol="EURUSD",
        asset_class=AssetClass.FOREX,
        digits=5,
        trade_stops_level=15,
        trade_freeze_level=10,
        trade_mode="FULL",
        filling_mode="IOC",
        order_mode="MARKET_LIMIT",
        order_gtc_mode="GTC"
    )

    # Convert to dict (as would be done for JSON export)
    extractor = MT5SpecExtractor()
    spec_dict = extractor._spec_to_dict(spec)

    print("\nSerialized spec (new fields):")
    print(f"  trade_stops_level: {spec_dict['trade_stops_level']}")
    print(f"  trade_freeze_level: {spec_dict['trade_freeze_level']}")
    print(f"  trade_mode: {spec_dict['trade_mode']}")
    print(f"  filling_mode: {spec_dict['filling_mode']}")
    print(f"  order_mode: {spec_dict['order_mode']}")
    print(f"  order_gtc_mode: {spec_dict['order_gtc_mode']}")

    # Verify all fields present
    assert 'trade_stops_level' in spec_dict, "Missing trade_stops_level in JSON"
    assert 'trade_freeze_level' in spec_dict, "Missing trade_freeze_level in JSON"
    assert 'trade_mode' in spec_dict, "Missing trade_mode in JSON"
    assert 'filling_mode' in spec_dict, "Missing filling_mode in JSON"
    assert 'order_mode' in spec_dict, "Missing order_mode in JSON"
    assert 'order_gtc_mode' in spec_dict, "Missing order_gtc_mode in JSON"

    print("\n✅ TEST PASSED: All new fields serialized to JSON\n")
    return True


def test_data_loader_compatibility():
    """Test that UnifiedDataLoader can read specs with new fields."""
    print("="*80)
    print("TEST 5: Data Loader Compatibility")
    print("="*80)

    from kinetra.data_loader import UnifiedDataLoader
    import json
    import tempfile

    # Create temp JSON with new fields
    test_spec = {
        "EURUSD": {
            "symbol": "EURUSD",
            "asset_class": "forex",
            "digits": 5,
            "point": 0.00001,
            "contract_size": 100000,
            "volume_min": 0.01,
            "volume_max": 100.0,
            "volume_step": 0.01,
            "margin_initial_rate_buy": 0.01,
            "margin_initial_rate_sell": 0.01,
            "margin_maintenance_rate_buy": 0.005,
            "margin_maintenance_rate_sell": 0.005,
            "margin_hedge": 0.0,
            "margin_currency": "USD",
            "margin_mode": "FOREX",
            "spread_typical": 1.5,
            "spread_min": 0.8,
            "spread_max": 5.0,
            "commission_per_lot": 0.0,
            "swap_long": -12.16,
            "swap_short": 4.37,
            "swap_type": "points",
            "swap_triple_day": "wednesday",
            "profit_calc_mode": "FOREX",
            "trade_stops_level": 15,
            "trade_freeze_level": 10,
            "trade_mode": "FULL",
            "filling_mode": "IOC",
            "order_mode": "MARKET_LIMIT",
            "order_gtc_mode": "GTC",
            "trading_hours": None,
            "last_updated": "2024-12-30T00:00:00",
            "source": "test"
        }
    }

    # Write to temp file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(test_spec, f, indent=2)
        temp_file = f.name

    try:
        # Load with UnifiedDataLoader
        loader = UnifiedDataLoader(verbose=True, specs_file=temp_file)

        # Verify spec loaded correctly
        assert "EURUSD" in loader.specs_cache, "EURUSD not in specs cache"

        spec = loader.specs_cache["EURUSD"]

        print(f"\nLoaded spec from JSON:")
        print(f"  Symbol: {spec.symbol}")
        print(f"  Trade Stops Level: {spec.trade_stops_level}")
        print(f"  Trade Freeze Level: {spec.trade_freeze_level}")
        print(f"  Trade Mode: {spec.trade_mode}")
        print(f"  Filling Mode: {spec.filling_mode}")

        # Verify values
        assert spec.trade_stops_level == 15, f"Wrong stops level: {spec.trade_stops_level}"
        assert spec.trade_freeze_level == 10, f"Wrong freeze level: {spec.trade_freeze_level}"
        assert spec.trade_mode == "FULL", f"Wrong trade mode: {spec.trade_mode}"
        assert spec.filling_mode == "IOC", f"Wrong filling mode: {spec.filling_mode}"

        print("\n✅ TEST PASSED: Data loader correctly loads new fields\n")
        return True

    finally:
        # Cleanup
        import os
        os.unlink(temp_file)


def main():
    """Run all tests."""
    print("\n" + "="*80)
    print("MT5 FREEZE ZONES & STOPS LEVELS TEST SUITE")
    print("="*80 + "\n")

    tests = [
        ("New Fields in SymbolSpec", test_new_fields),
        ("Stop Distance Validation", test_stop_distance_validation),
        ("Safe Stop Distance Calculation", test_safe_stop_distance),
        ("JSON Serialization", test_json_serialization),
        ("Data Loader Compatibility", test_data_loader_compatibility),
    ]

    results = {}

    for name, test_func in tests:
        try:
            result = test_func()
            results[name] = result
        except Exception as e:
            print(f"\n❌ TEST FAILED: {name}")
            print(f"   Error: {e}")
            import traceback
            traceback.print_exc()
            results[name] = False

    # Summary
    print("="*80)
    print("TEST SUMMARY")
    print("="*80)

    passed = sum(1 for r in results.values() if r)
    total = len(results)

    for name, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"  {status}: {name}")

    print()
    print(f"OVERALL: {passed}/{total} tests passed ({100*passed//total}%)")
    print("="*80)

    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
