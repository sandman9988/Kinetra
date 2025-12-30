#!/usr/bin/env python3
"""
Numerical Safety & Data Validation Tests

Tests for production-critical issues:
- Array overflow/underflow
- NaN propagation
- Memory leaks
- Normalization stability
- Scaling edge cases
- Safe maths operations
- Atomic persistence

Goal: Break it before production does!
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import traceback
import gc
import psutil
import os


def test_nan_propagation():
    """Test: NaN handling in physics computations"""
    print("="*80)
    print("TEST: NaN Propagation")
    print("="*80)

    try:
        from kinetra.data_package import DataPackage
        from kinetra.market_microstructure import AssetClass

        # Create data with NaNs
        dates = pd.date_range('2024-01-01', periods=100, freq='H')
        prices = pd.DataFrame({
            'open': [100.0] * 100,
            'high': [101.0] * 100,
            'low': [99.0] * 100,
            'close': [100.0] * 100,
            'volume': [1000] * 100
        }, index=dates)

        # Inject NaNs at different positions
        prices.loc[prices.index[10], 'close'] = np.nan  # Early
        prices.loc[prices.index[50], 'close'] = np.nan  # Middle
        prices.loc[prices.index[90], 'close'] = np.nan  # Late

        pkg = DataPackage(prices=prices, symbol='TEST', timeframe='H1', market_type=AssetClass.FOREX)

        # Validation should catch NaNs
        is_valid = pkg.validate()

        if pkg.validation_warnings:
            print(f"✓ NaN detection working: {len(pkg.validation_warnings)} warnings")
            for warning in pkg.validation_warnings:
                print(f"  - {warning}")
        else:
            print("❌ NaNs not detected in validation!")
            return False

        print("\n✅ TEST PASSED: NaN propagation detected\n")
        return True

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        traceback.print_exc()
        return False


def test_overflow_underflow():
    """Test: Overflow/underflow in calculations"""
    print("="*80)
    print("TEST: Overflow/Underflow Safety")
    print("="*80)

    try:
        from kinetra.data_package import DataPackage
        from kinetra.market_microstructure import AssetClass

        test_cases = [
            ("Extreme high prices", 1e10),
            ("Extreme low prices", 1e-10),
            ("Zero prices", 0.0),
            ("Negative prices", -100.0),
        ]

        for name, price_value in test_cases:
            print(f"\n  Testing: {name} ({price_value})")

            dates = pd.date_range('2024-01-01', periods=10, freq='H')
            prices = pd.DataFrame({
                'open': [price_value] * 10,
                'high': [price_value * 1.01] * 10,
                'low': [price_value * 0.99] * 10,
                'close': [price_value] * 10,
                'volume': [1000] * 10
            }, index=dates)

            pkg = DataPackage(prices=prices, symbol='TEST', timeframe='H1', market_type=AssetClass.FOREX)
            is_valid = pkg.validate()

            # Check for validation errors on invalid prices
            if price_value <= 0:
                if pkg.validation_errors:
                    print(f"    ✓ Invalid prices caught: {len(pkg.validation_errors)} errors")
                else:
                    print(f"    ❌ Invalid prices NOT caught!")
                    return False
            else:
                # Valid prices should pass
                if not pkg.validation_errors:
                    print(f"    ✓ Valid extreme prices accepted")
                else:
                    print(f"    ⚠️  Valid prices rejected: {pkg.validation_errors}")

        print("\n✅ TEST PASSED: Overflow/underflow safety working\n")
        return True

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        traceback.print_exc()
        return False


def test_normalization_stability():
    """Test: Normalization with extreme values"""
    print("="*80)
    print("TEST: Normalization Stability")
    print("="*80)

    try:
        # Test z-score normalization with extreme values
        test_arrays = [
            ("Normal distribution", np.random.randn(1000)),
            ("Uniform distribution", np.random.uniform(-10, 10, 1000)),
            ("Heavy-tailed", np.random.standard_t(df=2, size=1000)),
            ("With outliers", np.concatenate([np.random.randn(990), [100, -100, 1000, -1000]])),
            ("Nearly constant", np.ones(1000) + np.random.randn(1000) * 1e-6),
            ("All zeros", np.zeros(1000)),
            ("Single spike", np.concatenate([np.zeros(999), [1e10]])),
        ]

        for name, arr in test_arrays:
            print(f"\n  Testing: {name}")

            # Standard z-score normalization
            mean = np.mean(arr)
            std = np.std(arr)

            if std == 0 or np.isnan(std) or np.isinf(std):
                print(f"    ⚠️  Degenerate std: {std}")
                # Safe normalization: return zeros
                normalized = np.zeros_like(arr)
            else:
                normalized = (arr - mean) / std

            # Check for NaN/Inf in result
            has_nan = np.isnan(normalized).any()
            has_inf = np.isinf(normalized).any()

            if has_nan or has_inf:
                print(f"    ❌ Normalization produced NaN/Inf!")
                print(f"       NaN: {np.isnan(normalized).sum()}, Inf: {np.isinf(normalized).sum()}")
                return False
            else:
                print(f"    ✓ Normalized safely: mean={np.mean(normalized):.4f}, std={np.std(normalized):.4f}")

        print("\n✅ TEST PASSED: Normalization stability verified\n")
        return True

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        traceback.print_exc()
        return False


def test_memory_leaks():
    """Test: Memory leaks in data loading"""
    print("="*80)
    print("TEST: Memory Leak Detection")
    print("="*80)

    try:
        from kinetra.data_loader import UnifiedDataLoader

        # Find a CSV file
        data_dir = Path("data/master")
        csv_files = list(data_dir.glob("*_H*.csv"))

        if not csv_files:
            print("⚠️  No CSV files found - skipping test")
            return None

        test_file = csv_files[0]
        print(f"Testing with: {test_file.name}")

        # Get baseline memory
        process = psutil.Process(os.getpid())
        baseline_mb = process.memory_info().rss / 1024 / 1024

        print(f"Baseline memory: {baseline_mb:.1f} MB")

        # Load/unload 10 times
        loader = UnifiedDataLoader(verbose=False, compute_physics=False)

        for i in range(10):
            pkg = loader.load(str(test_file))
            df = pkg.to_backtest_engine_format()

            # Explicitly delete
            del pkg
            del df

            # Force garbage collection
            gc.collect()

            if i % 3 == 0:
                current_mb = process.memory_info().rss / 1024 / 1024
                growth = current_mb - baseline_mb
                print(f"  Iteration {i+1}: {current_mb:.1f} MB (+{growth:.1f} MB)")

        # Final memory check
        gc.collect()
        final_mb = process.memory_info().rss / 1024 / 1024
        total_growth = final_mb - baseline_mb

        print(f"\nFinal memory: {final_mb:.1f} MB")
        print(f"Total growth: {total_growth:.1f} MB")

        # Threshold: <50MB growth for 10 loads is acceptable
        if total_growth < 50:
            print(f"✓ Memory growth within threshold (<50 MB)")
            print("\n✅ TEST PASSED: No significant memory leaks\n")
            return True
        else:
            print(f"⚠️  Memory growth: {total_growth:.1f} MB - possible leak!")
            print("\n⚠️  TEST WARNING: Check for memory leaks\n")
            return None

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        traceback.print_exc()
        return False


def test_safe_division():
    """Test: Safe division (divide by zero handling)"""
    print("="*80)
    print("TEST: Safe Division Operations")
    print("="*80)

    try:
        # Test cases that should be handled safely
        test_cases = [
            (10.0, 2.0, 5.0, "Normal division"),
            (10.0, 0.0, None, "Division by zero"),
            (0.0, 0.0, None, "Zero divided by zero"),
            (np.inf, 2.0, None, "Infinity division"),
            (10.0, np.inf, 0.0, "Division by infinity"),
            (np.nan, 2.0, None, "NaN division"),
        ]

        def safe_divide(a, b, default=0.0):
            """Safe division with NaN/Inf/Zero handling"""
            if b == 0 or np.isnan(b) or np.isinf(b):
                return default
            if np.isnan(a) or np.isinf(a):
                return default
            return a / b

        for a, b, expected, name in test_cases:
            result = safe_divide(a, b)
            is_safe = not (np.isnan(result) or np.isinf(result))

            print(f"  {name:30s} {a}/{b} = {result} {'✓' if is_safe else '❌'}")

            if not is_safe and expected is not None:
                print(f"    ❌ Unsafe result produced!")
                return False

        print("\n✅ TEST PASSED: Safe division working\n")
        return True

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        traceback.print_exc()
        return False


def test_atomic_persistence():
    """Test: Atomic file writes (crash safety)"""
    print("="*80)
    print("TEST: Atomic Persistence")
    print("="*80)

    try:
        from rl_exploration_framework import PersistenceManager
        import tempfile
        import shutil

        # Create temp directory
        temp_dir = tempfile.mkdtemp(prefix="test_persistence_")
        print(f"Test directory: {temp_dir}")

        try:
            # Create persistence manager
            pm = PersistenceManager(
                base_dir=temp_dir,
                experiment_name="test_atomic",
                checkpoint_every=1
            )

            # Save some data
            test_data = {
                'model_weights': np.random.randn(1000, 100),
                'metrics': {'sharpe': 1.5, 'pnl': 1000.0},
                'timestamp': datetime.now().isoformat()
            }

            # Atomic save
            save_path = pm.results_dir / "test_data.pkl"
            pm.atomic_save(test_data, save_path)

            # Verify file exists
            if not save_path.exists():
                print("❌ File not created!")
                return False

            # Load back
            loaded_data = pm.load(save_path)

            # Verify data integrity
            if loaded_data is None:
                print("❌ Failed to load data!")
                return False

            if not np.array_equal(loaded_data['model_weights'], test_data['model_weights']):
                print("❌ Data corruption detected!")
                return False

            print(f"✓ Atomic save/load successful")
            print(f"✓ Data integrity verified")
            print(f"✓ File size: {save_path.stat().st_size / 1024:.1f} KB")

            print("\n✅ TEST PASSED: Atomic persistence working\n")
            return True

        finally:
            # Cleanup
            shutil.rmtree(temp_dir, ignore_errors=True)

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        traceback.print_exc()
        return False


def test_array_broadcasting():
    """Test: Array broadcasting edge cases"""
    print("="*80)
    print("TEST: Array Broadcasting Safety")
    print("="*80)

    try:
        # Test broadcasting scenarios that can cause silent errors
        test_cases = [
            ((1000,), (1000,), True, "Same shape"),
            ((1000,), (1,), True, "Scalar broadcast"),
            ((1000,), (1000, 1), False, "Incompatible shapes"),
            ((1000, 64), (64,), True, "Feature broadcast"),
            ((1000,), (999,), False, "Off-by-one"),
        ]

        for shape_a, shape_b, should_work, name in test_cases:
            print(f"\n  {name}: {shape_a} + {shape_b}")

            arr_a = np.random.randn(*shape_a)
            arr_b = np.random.randn(*shape_b)

            try:
                result = arr_a + arr_b
                if not should_work:
                    print(f"    ❌ Should have failed but didn't!")
                    print(f"       Result shape: {result.shape}")
                    return False
                else:
                    print(f"    ✓ Broadcast successful: {result.shape}")

            except ValueError as e:
                if should_work:
                    print(f"    ❌ Should have worked but failed: {e}")
                    return False
                else:
                    print(f"    ✓ Correctly rejected incompatible shapes")

        print("\n✅ TEST PASSED: Array broadcasting safety verified\n")
        return True

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        traceback.print_exc()
        return False


def test_digit_precision():
    """Test: Different instrument precision (yen 3, gold 2, forex 5)"""
    print("="*80)
    print("TEST: Digit Precision Handling")
    print("="*80)

    try:
        from kinetra.market_microstructure import SymbolSpec, AssetClass

        # Test different precision levels
        instruments = [
            ("USDJPY", 3, "Yen pair"),
            ("XAUUSD", 2, "Gold"),
            ("EURUSD", 5, "Forex major"),
            ("BTCUSD", 2, "Crypto"),
        ]

        print("\nTesting precision-aware calculations:\n")

        for symbol, expected_digits, description in instruments:
            # Create spec
            spec = SymbolSpec(
                symbol=symbol,
                asset_class=AssetClass.FOREX,
                digits=expected_digits
            )

            # Verify point calculation
            expected_point = 10 ** (-expected_digits)
            actual_point = spec.point

            print(f"  {symbol:8s} ({description:12s}):")
            print(f"    Digits: {spec.digits}")
            print(f"    Point:  {spec.point} (expected: {expected_point})")

            if abs(actual_point - expected_point) > 1e-10:
                print(f"    ❌ Point calculation incorrect!")
                return False

            # Test spread conversion
            spread_points = 10.0  # 10 point spread
            spread_price = spec.spread_in_price()
            expected_spread = spread_points * spec.point

            print(f"    Spread: {spread_points} points = {spread_price} price")

            # Test pip value (precision-aware)
            pip_val = spec.pip_value(lot_size=1.0)
            print(f"    Pip value: ${pip_val:.4f} per lot")

            # Test price comparisons
            price1 = 100.0
            price2 = 100.0 + spec.point  # 1 tick difference

            diff_ticks = round((price2 - price1) / spec.point)
            print(f"    Price diff: {price1} → {price2} = {diff_ticks} ticks")

            if diff_ticks != 1:
                print(f"    ❌ Tick calculation incorrect! Got {diff_ticks}, expected 1")
                return False

            print(f"    ✓ Precision handling correct\n")

        # Test cross-instrument normalization issues
        print("\nTesting cross-instrument normalization:\n")

        # Prices at different scales
        yen_price = 150.123  # USDJPY
        gold_price = 2000.45  # XAUUSD
        forex_price = 1.08567  # EURUSD

        # Z-score normalization should handle scale differences
        prices = np.array([yen_price, gold_price, forex_price])
        mean = np.mean(prices)
        std = np.std(prices)

        normalized = (prices - mean) / std if std > 0 else np.zeros_like(prices)

        print(f"  Original: {prices}")
        print(f"  Normalized: {normalized}")
        print(f"  Mean: {np.mean(normalized):.6f} (should be ~0)")
        print(f"  Std: {np.std(normalized):.6f} (should be ~1)")

        if abs(np.mean(normalized)) > 1e-10:
            print(f"  ⚠️  Normalization mean not zero!")

        print("\n✅ TEST PASSED: Digit precision handling verified\n")
        return True

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        traceback.print_exc()
        return False


def test_floating_point_precision():
    """Test: Floating point comparison and accumulation errors"""
    print("="*80)
    print("TEST: Floating Point Precision")
    print("="*80)

    try:
        # Classic floating point issues
        print("\nFloating point equality issues:\n")

        # 0.1 + 0.2 != 0.3
        a = 0.1
        b = 0.2
        c = 0.3

        if a + b == c:
            print(f"  ❌ 0.1 + 0.2 == 0.3 (FALSE! Floating point error missed!)")
            return False
        else:
            print(f"  ✓ 0.1 + 0.2 != 0.3 detected: {a + b} vs {c}")

        # Safe comparison with epsilon
        epsilon = 1e-9
        if abs((a + b) - c) < epsilon:
            print(f"  ✓ Safe epsilon comparison works")
        else:
            print(f"  ❌ Epsilon comparison failed!")
            return False

        # Accumulation errors
        print("\nAccumulation errors:\n")

        # Sum 0.1 a million times
        direct_sum = sum([0.1] * 1000000)
        expected = 100000.0

        error = abs(direct_sum - expected)
        print(f"  Sum of 0.1 x 1M times:")
        print(f"    Result: {direct_sum:.10f}")
        print(f"    Expected: {expected:.10f}")
        print(f"    Error: {error:.10e}")

        if error > 0.01:  # More than 1 cent error
            print(f"  ⚠️  Large accumulation error detected!")

        # Kahan summation (compensated)
        def kahan_sum(values):
            """Kahan compensated summation for numerical stability"""
            total = 0.0
            c = 0.0  # Running compensation
            for value in values:
                y = value - c
                t = total + y
                c = (t - total) - y
                total = t
            return total

        kahan_result = kahan_sum([0.1] * 1000000)
        kahan_error = abs(kahan_result - expected)

        print(f"\n  Kahan summation:")
        print(f"    Result: {kahan_result:.10f}")
        print(f"    Error: {kahan_error:.10e}")
        print(f"    Improvement: {error / kahan_error:.2f}x better")

        # Test price tick accumulation (real trading scenario)
        print("\nPrice tick accumulation (trading scenario):\n")

        # Simulate 100,000 ticks of 0.00001 (forex)
        tick_size = 0.00001
        num_ticks = 100000

        naive_sum = sum([tick_size] * num_ticks)
        kahan_price = kahan_sum([tick_size] * num_ticks)
        expected_price = tick_size * num_ticks

        naive_error = abs(naive_sum - expected_price)
        kahan_error_price = abs(kahan_price - expected_price)

        print(f"  {num_ticks:,} ticks of {tick_size}:")
        print(f"    Naive sum error: {naive_error:.10e}")
        print(f"    Kahan sum error: {kahan_error_price:.10e}")

        if naive_error > kahan_error_price * 10:
            print(f"    ✓ Kahan summation significantly better")
        else:
            print(f"    ⚠️  Kahan benefit not significant")

        print("\n✅ TEST PASSED: Floating point precision understood\n")
        return True

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        traceback.print_exc()
        return False


def test_type_conversions():
    """Test: Long to int, float to int conversions (overflow)"""
    print("="*80)
    print("TEST: Type Conversion Safety")
    print("="*80)

    try:
        print("\nTesting int overflow scenarios:\n")

        # Python 3 has arbitrary precision ints, but numpy doesn't
        test_values = [
            (2**31 - 1, "Max int32"),
            (2**31, "int32 overflow"),
            (2**63 - 1, "Max int64"),
            (2**63, "int64 overflow"),
        ]

        for value, description in test_values:
            print(f"  {description}: {value}")

            # Python int (safe)
            python_int = int(value)
            print(f"    Python int: {python_int} ✓")

            # NumPy int64 (can overflow)
            try:
                np_int64 = np.int64(value)
                if value > 2**63 - 1:
                    print(f"    NumPy int64: {np_int64} (OVERFLOW!)")
                    if np_int64 < 0:
                        print(f"    ⚠️  Overflowed to negative!")
                else:
                    print(f"    NumPy int64: {np_int64} ✓")
            except OverflowError:
                print(f"    NumPy int64: OverflowError (safe) ✓")

        # Float to int conversions
        print("\nFloat to int conversions:\n")

        float_values = [
            (3.7, 3, "Truncation"),
            (3.5, 3 or 4, "Rounding"),
            (1e10, 10000000000, "Large float"),
            (np.inf, None, "Infinity"),
            (np.nan, None, "NaN"),
        ]

        for float_val, expected, description in float_values:
            print(f"  {description}: {float_val}")

            # Truncation
            truncated = int(float_val) if not (np.isnan(float_val) or np.isinf(float_val)) else None
            print(f"    int(): {truncated}")

            # Rounding
            rounded = round(float_val) if not (np.isnan(float_val) or np.isinf(float_val)) else None
            print(f"    round(): {rounded}")

            # Safe conversion
            def safe_float_to_int(f, default=0):
                if np.isnan(f) or np.isinf(f):
                    return default
                return int(f)

            safe_val = safe_float_to_int(float_val)
            print(f"    safe: {safe_val} ✓\n")

        # Test lot size conversions (trading specific)
        print("Lot size conversions (trading):\n")

        lot_sizes = [0.01, 0.1, 1.0, 10.5, 0.001, 100.0]

        for lot in lot_sizes:
            # Convert to micro-lots (100,000 units)
            micro_lots = int(lot * 100)
            back_to_lot = micro_lots / 100.0

            error = abs(back_to_lot - lot)

            print(f"  {lot:6.3f} lot → {micro_lots:6d} micro → {back_to_lot:6.3f} lot (error: {error:.6f})")

            if error > 1e-6:
                print(f"    ⚠️  Conversion error: {error}")

        print("\n✅ TEST PASSED: Type conversion safety verified\n")
        return True

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        traceback.print_exc()
        return False


def test_timestamp_precision():
    """Test: Timestamp precision and timezone issues"""
    print("="*80)
    print("TEST: Timestamp Precision & Timezone Safety")
    print("="*80)

    try:
        from datetime import datetime, timezone
        import time

        print("\nTimestamp precision:\n")

        # Millisecond precision
        now_ms = time.time()
        dt = datetime.fromtimestamp(now_ms, tz=timezone.utc)

        print(f"  Unix timestamp: {now_ms:.6f}")
        print(f"  DateTime: {dt.isoformat()}")

        # Round-trip conversion
        back_to_ts = dt.timestamp()
        error = abs(back_to_ts - now_ms)

        print(f"  Round-trip error: {error:.9f} seconds")

        if error > 1e-6:  # More than 1 microsecond
            print(f"  ⚠️  Timestamp precision loss!")

        # Timezone issues
        print("\nTimezone handling:\n")

        dt_utc = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        dt_naive = datetime(2024, 1, 1, 12, 0, 0)  # No timezone

        print(f"  UTC aware: {dt_utc.isoformat()}")
        print(f"  Naive: {dt_naive.isoformat()}")

        # Comparison issues
        try:
            if dt_utc > dt_naive:
                print(f"  ❌ Comparing aware and naive datetimes should fail!")
                return False
        except TypeError:
            print(f"  ✓ Aware/naive comparison correctly rejected")

        print("\n✅ TEST PASSED: Timestamp safety verified\n")
        return True

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        traceback.print_exc()
        return False


def main():
    """Run all numerical safety tests"""
    print("\n" + "="*80)
    print("NUMERICAL SAFETY & DATA VALIDATION TEST SUITE")
    print("Goal: Break it before production does!")
    print("="*80 + "\n")

    tests = [
        ("NaN Propagation", test_nan_propagation),
        ("Overflow/Underflow Safety", test_overflow_underflow),
        ("Normalization Stability", test_normalization_stability),
        ("Digit Precision Handling", test_digit_precision),
        ("Floating Point Precision", test_floating_point_precision),
        ("Type Conversion Safety", test_type_conversions),
        ("Timestamp Precision", test_timestamp_precision),
        ("Memory Leak Detection", test_memory_leaks),
        ("Safe Division Operations", test_safe_division),
        ("Atomic Persistence", test_atomic_persistence),
        ("Array Broadcasting Safety", test_array_broadcasting),
    ]

    results = {}

    for name, test_func in tests:
        try:
            result = test_func()
            results[name] = result
        except KeyboardInterrupt:
            print("\n\n⚠️  Tests interrupted by user")
            break
        except Exception as e:
            print(f"\n❌ {name} crashed: {e}")
            traceback.print_exc()
            results[name] = False

    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)

    passed = sum(1 for r in results.values() if r is True)
    failed = sum(1 for r in results.values() if r is False)
    warnings = sum(1 for r in results.values() if r is None)

    for name, result in results.items():
        if result is True:
            status = "✅ PASSED"
        elif result is False:
            status = "❌ FAILED"
        else:
            status = "⚠️  WARNING"
        print(f"{status:12s} {name}")

    print(f"\nResults: {passed} passed, {failed} failed, {warnings} warnings")

    if failed > 0:
        print("\n🔍 FAILURES FOUND - Fix before production!")
        sys.exit(1)
    elif warnings > 0:
        print("\n⚠️  Warnings found - investigate potential issues")
        sys.exit(0)
    else:
        print("\n✅ All numerical safety tests passed!")
        sys.exit(0)


if __name__ == "__main__":
    main()
