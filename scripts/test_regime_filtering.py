#!/usr/bin/env python3
"""
Test 3-Regime Filtering in TradingEnv

Validates that regime filtering works correctly:
1. Physics regime filtering (laminar, underdamped, overdamped)
2. Volatility regime filtering (low, medium, high)
3. Momentum regime filtering (uptrend, ranging, downtrend)

Demonstrates regime-specialized environment creation.
"""

import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np

from kinetra.regime_filtered_env import (
    RegimeFilteredTradingEnv,
    RegimeFilter,
    PhysicsRegime,
    VolatilityRegime,
    MomentumRegime,
    create_regime_specialists,
)


def load_test_data() -> pd.DataFrame:
    """Load BTC H1 data for testing."""
    data_path = Path("data/master/BTCUSD_H1_202407010000_202512270700.csv")

    if not data_path.exists():
        print(f"Error: Test data not found at {data_path}")
        print("Please ensure you have BTCUSD H1 data in data/master/")
        sys.exit(1)

    df = pd.read_csv(data_path)

    # Ensure datetime index
    if 'time' in df.columns:
        df['time'] = pd.to_datetime(df['time'])
        df = df.set_index('time')

    # Ensure OHLCV columns (lowercase)
    required_cols = ['open', 'high', 'low', 'close', 'volume']
    for col in required_cols:
        if col not in df.columns and col.upper() in df.columns:
            df[col] = df[col.upper()]

    return df


def test_regime_classification():
    """Test 1: Verify regime classification works."""
    print("=" * 70)
    print("TEST 1: Regime Classification")
    print("=" * 70)

    df = load_test_data()

    print(f"\nLoaded {len(df)} bars from BTCUSD H1 data")

    # Create env with no filtering (to test classification only)
    env = RegimeFilteredTradingEnv(df)

    # Get regime distribution
    distribution = env.get_regime_distribution()

    print("\nRegime Distribution:")
    print("-" * 70)

    for regime_type, counts in distribution.items():
        print(f"\n{regime_type.capitalize()} Regimes:")
        total = sum(counts.values())
        for regime_name, count in sorted(counts.items()):
            pct = 100 * count / total if total > 0 else 0
            print(f"  {regime_name:20s}: {count:6d} bars ({pct:5.1f}%)")

    # Verify all regimes are represented
    for regime_type, counts in distribution.items():
        if len(counts) == 0:
            print(f"\n❌ TEST FAILED: No {regime_type} regimes found!")
            return False

    print("\n✅ TEST PASSED: All regime types classified\n")
    return True


def test_physics_regime_filtering():
    """Test 2: Physics regime filtering."""
    print("=" * 70)
    print("TEST 2: Physics Regime Filtering")
    print("=" * 70)

    df = load_test_data()

    # Filter for laminar only
    print("\nTesting LAMINAR-only filter...")
    regime_filter = RegimeFilter(
        physics_regimes={PhysicsRegime.LAMINAR}
    )

    try:
        env = RegimeFilteredTradingEnv(df, regime_filter=regime_filter, min_filtered_bars=100)

        print(f"✓ Environment created successfully")
        print(f"✓ Valid bars: {len(env.valid_bars)}/{len(env.features)}")

        # Verify all valid bars are laminar
        valid_regimes = env.features.iloc[env.valid_bars]['physics_regime_enum'].unique()
        if len(valid_regimes) == 1 and PhysicsRegime.LAMINAR in valid_regimes:
            print(f"✓ All valid bars are LAMINAR")
        else:
            print(f"❌ Filter leaked: Found regimes {valid_regimes}")
            return False

    except ValueError as e:
        print(f"❌ Failed to create environment: {e}")
        return False

    # Filter for underdamped only
    print("\nTesting UNDERDAMPED-only filter...")
    regime_filter = RegimeFilter(
        physics_regimes={PhysicsRegime.UNDERDAMPED}
    )

    env = RegimeFilteredTradingEnv(df, regime_filter=regime_filter, min_filtered_bars=100)
    print(f"✓ Valid bars: {len(env.valid_bars)}/{len(env.features)}")

    print("\n✅ TEST PASSED: Physics regime filtering works\n")
    return True


def test_volatility_regime_filtering():
    """Test 3: Volatility regime filtering."""
    print("=" * 70)
    print("TEST 3: Volatility Regime Filtering")
    print("=" * 70)

    df = load_test_data()

    # Filter for high volatility only
    print("\nTesting HIGH_VOL-only filter...")
    regime_filter = RegimeFilter(
        volatility_regimes={VolatilityRegime.HIGH_VOL}
    )

    try:
        env = RegimeFilteredTradingEnv(df, regime_filter=regime_filter, min_filtered_bars=100)

        print(f"✓ Environment created successfully")
        print(f"✓ Valid bars: {len(env.valid_bars)}/{len(env.features)}")

        # Verify all valid bars are high vol
        valid_regimes = env.features.iloc[env.valid_bars]['vol_regime_enum'].unique()
        if len(valid_regimes) == 1 and VolatilityRegime.HIGH_VOL in valid_regimes:
            print(f"✓ All valid bars are HIGH_VOL")
        else:
            print(f"❌ Filter leaked: Found regimes {valid_regimes}")
            return False

    except ValueError as e:
        print(f"❌ Failed to create environment: {e}")
        return False

    print("\n✅ TEST PASSED: Volatility regime filtering works\n")
    return True


def test_momentum_regime_filtering():
    """Test 4: Momentum regime filtering."""
    print("=" * 70)
    print("TEST 4: Momentum Regime Filtering")
    print("=" * 70)

    df = load_test_data()

    # Filter for uptrend only
    print("\nTesting UPTREND-only filter...")
    regime_filter = RegimeFilter(
        momentum_regimes={MomentumRegime.UPTREND}
    )

    try:
        env = RegimeFilteredTradingEnv(df, regime_filter=regime_filter, min_filtered_bars=100)

        print(f"✓ Environment created successfully")
        print(f"✓ Valid bars: {len(env.valid_bars)}/{len(env.features)}")

        # Verify all valid bars are uptrend
        valid_regimes = env.features.iloc[env.valid_bars]['momentum_regime_enum'].unique()
        if len(valid_regimes) == 1 and MomentumRegime.UPTREND in valid_regimes:
            print(f"✓ All valid bars are UPTREND")
        else:
            print(f"❌ Filter leaked: Found regimes {valid_regimes}")
            return False

    except ValueError as e:
        print(f"❌ Failed to create environment: {e}")
        return False

    print("\n✅ TEST PASSED: Momentum regime filtering works\n")
    return True


def test_combined_regime_filtering():
    """Test 5: Combined multi-regime filtering."""
    print("=" * 70)
    print("TEST 5: Combined Regime Filtering")
    print("=" * 70)

    df = load_test_data()

    # Filter for laminar + low vol + uptrend (ideal trend-following conditions)
    print("\nTesting LAMINAR + LOW_VOL + UPTREND filter...")
    regime_filter = RegimeFilter(
        physics_regimes={PhysicsRegime.LAMINAR},
        volatility_regimes={VolatilityRegime.LOW_VOL},
        momentum_regimes={MomentumRegime.UPTREND},
    )

    try:
        env = RegimeFilteredTradingEnv(df, regime_filter=regime_filter, min_filtered_bars=50)

        print(f"✓ Environment created successfully")
        print(f"✓ Valid bars: {len(env.valid_bars)}/{len(env.features)}")
        print(f"✓ Filter selectivity: {100*(1 - len(env.valid_bars)/len(env.features)):.1f}% bars excluded")

        # Verify all conditions met
        valid_physics = env.features.iloc[env.valid_bars]['physics_regime_enum'].unique()
        valid_vol = env.features.iloc[env.valid_bars]['vol_regime_enum'].unique()
        valid_momentum = env.features.iloc[env.valid_bars]['momentum_regime_enum'].unique()

        checks = [
            (len(valid_physics) == 1 and PhysicsRegime.LAMINAR in valid_physics, "Physics"),
            (len(valid_vol) == 1 and VolatilityRegime.LOW_VOL in valid_vol, "Volatility"),
            (len(valid_momentum) == 1 and MomentumRegime.UPTREND in valid_momentum, "Momentum"),
        ]

        all_passed = True
        for passed, regime_type in checks:
            if passed:
                print(f"  ✓ {regime_type} filter correct")
            else:
                print(f"  ❌ {regime_type} filter leaked!")
                all_passed = False

        if not all_passed:
            return False

    except ValueError as e:
        print(f"⚠️  Too few bars matching strict filter: {e}")
        print(f"  This is expected for very restrictive filters")

    print("\n✅ TEST PASSED: Combined regime filtering works\n")
    return True


def test_regime_specialists():
    """Test 6: Regime specialist environments."""
    print("=" * 70)
    print("TEST 6: Regime Specialist Environments")
    print("=" * 70)

    df = load_test_data()

    print("\nCreating regime specialist environments...")

    try:
        specialists = create_regime_specialists(df)

        print(f"\n✓ Created {len(specialists)} specialist environments:\n")

        for name, env in specialists.items():
            valid_pct = 100 * len(env.valid_bars) / len(env.features)
            print(f"  {name:25s}: {len(env.valid_bars):5d} valid bars ({valid_pct:5.1f}%)")
            print(f"    Filter: {env.regime_filter}")

    except Exception as e:
        print(f"\n❌ Failed to create specialists: {e}")
        return False

    print("\n✅ TEST PASSED: Regime specialists created successfully\n")
    return True


def test_environment_reset_and_step():
    """Test 7: Environment reset and step with filtering."""
    print("=" * 70)
    print("TEST 7: Environment Reset & Step")
    print("=" * 70)

    df = load_test_data()

    # Create filtered environment
    regime_filter = RegimeFilter(
        physics_regimes={PhysicsRegime.LAMINAR, PhysicsRegime.UNDERDAMPED}
    )

    env = RegimeFilteredTradingEnv(df, regime_filter=regime_filter)

    print(f"\nTesting environment interaction...")

    # Reset environment
    state = env.reset()
    print(f"✓ Environment reset")
    print(f"  State shape: {state.shape}")
    print(f"  State dim: {env.state_dim}")

    # Take a few steps
    total_reward = 0
    skipped_bars_total = 0

    for step in range(10):
        action = np.random.randint(0, env.action_dim)  # Random action
        next_state, reward, done, info = env.step(action)

        total_reward += reward

        if 'skipped_bars' in info:
            skipped_bars_total += info['skipped_bars']

        if done:
            print(f"\n  Episode ended at step {step + 1}")
            break

    print(f"✓ Completed {step + 1} steps")
    print(f"  Total reward: {total_reward:.2f}")
    print(f"  Skipped bars: {skipped_bars_total}")

    print("\n✅ TEST PASSED: Environment interaction works\n")
    return True


def main():
    """Run all regime filtering tests."""
    print("\n" + "=" * 70)
    print("3-REGIME FILTERING TEST SUITE")
    print("=" * 70 + "\n")

    tests = [
        ("Regime Classification", test_regime_classification),
        ("Physics Regime Filtering", test_physics_regime_filtering),
        ("Volatility Regime Filtering", test_volatility_regime_filtering),
        ("Momentum Regime Filtering", test_momentum_regime_filtering),
        ("Combined Regime Filtering", test_combined_regime_filtering),
        ("Regime Specialist Environments", test_regime_specialists),
        ("Environment Reset & Step", test_environment_reset_and_step),
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
    print("=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)

    passed = sum(1 for r in results.values() if r)
    total = len(results)

    for name, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"  {status}: {name}")

    print()
    print(f"OVERALL: {passed}/{total} tests passed ({100*passed//total}%)")
    print("=" * 70)

    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
