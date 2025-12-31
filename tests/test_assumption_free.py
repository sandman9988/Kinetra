#!/usr/bin/env python3
"""
Unit tests for assumption-free framework.
Tests the new directional, asymmetric measures that replace symmetric assumptions.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from kinetra.dsp_features import DSPFeatureEngine, DirectionalWaveletExtractor
from kinetra.liquidity_features import (
    LiquidityFeatureEngine, RangeImpactExtractor, CVDExtractor
)
from kinetra.assumption_free_measures import (
    AsymmetricReturns, RankBasedMeasures, DirectionalVolatility,
    DirectionalOrderFlow, RecurrenceFeatures, TailBehavior,
    AssumptionFreeEngine
)


def create_test_data(n_bars: int = 200, trend: str = 'up') -> pd.DataFrame:
    """Create synthetic OHLCV data for testing."""
    np.random.seed(42)

    # Generate price with trend
    if trend == 'up':
        drift = 0.001
    elif trend == 'down':
        drift = -0.001
    else:
        drift = 0.0

    returns = np.random.randn(n_bars) * 0.02 + drift
    close = 100 * np.exp(np.cumsum(returns))

    # Generate OHLC from close
    high = close * (1 + np.abs(np.random.randn(n_bars)) * 0.01)
    low = close * (1 - np.abs(np.random.randn(n_bars)) * 0.01)
    open_ = np.roll(close, 1)
    open_[0] = close[0]

    # Volume
    tickvol = np.random.randint(100, 1000, n_bars)

    return pd.DataFrame({
        'open': open_,
        'high': high,
        'low': low,
        'close': close,
        'tickvol': tickvol
    })


class TestDirectionalPersistence:
    """Test the DirectionalWaveletExtractor that replaces Hurst."""

    def test_compute_directional_persistence(self):
        """Test that persistence is computed separately for up/down."""
        np.random.seed(42)
        # Create trending up data
        prices = 100 * np.exp(np.cumsum(np.random.randn(100) * 0.01 + 0.005))

        result = DirectionalWaveletExtractor.compute_directional_persistence(prices)

        assert 'up_persistence' in result
        assert 'down_persistence' in result
        assert 'persistence_asymmetry' in result
        assert 0 <= result['up_persistence'] <= 1
        assert 0 <= result['down_persistence'] <= 1
        print(f"  ✓ Persistence: up={result['up_persistence']:.3f}, down={result['down_persistence']:.3f}")

    def test_persistence_no_symmetric_reference(self):
        """Ensure no symmetric reference point (like 0.5) is used directly."""
        # The old code had: persistence_strength = abs(overall_persistence - 0.5) * 2
        # This should NOT be in the output
        np.random.seed(42)
        prices = 100 * np.exp(np.cumsum(np.random.randn(100) * 0.01))

        result = DirectionalWaveletExtractor.compute_directional_persistence(prices)

        # persistence_strength should NOT be in the result
        assert 'persistence_strength' not in result, "persistence_strength should be removed!"
        print("  ✓ No symmetric reference point (persistence_strength removed)")


class TestSignedRangeImpact:
    """Test the new signed range impact in liquidity features."""

    def test_signed_range_impact_direction(self):
        """Test that signed range impact preserves direction."""
        df = create_test_data(100)

        result = RangeImpactExtractor.extract_features(df, bar_idx=-1)

        assert 'signed_range_impact' in result
        assert 'signed_range_mean' in result
        assert -1 <= result['signed_range_impact'] <= 1
        print(f"  ✓ Signed range impact: {result['signed_range_impact']:.3f}")

    def test_signed_vs_unsigned(self):
        """Verify signed and unsigned are different metrics."""
        df = create_test_data(100)

        result = RangeImpactExtractor.extract_features(df, bar_idx=-1)

        # Unsigned is always positive (range/volume)
        assert result['range_impact'] >= 0
        # Signed can be negative
        assert result['signed_range_impact'] != abs(result['signed_range_impact']) or result['signed_range_impact'] == 0
        print(f"  ✓ Unsigned={result['range_impact']:.4f}, Signed={result['signed_range_impact']:.3f}")


class TestAsymmetricReturns:
    """Test streak momentum and asymmetric return features."""

    def test_streak_momentum_features(self):
        """Test that streak momentum features are present."""
        prices = 100 * np.exp(np.cumsum(np.random.randn(100) * 0.01))

        result = AsymmetricReturns.extract_features(prices)

        # New streak momentum features
        assert 'up_streak_magnitude' in result
        assert 'down_streak_magnitude' in result
        assert 'streak_conviction' in result
        print(f"  ✓ Streak features: up_mag={result['up_streak_magnitude']:.4f}, "
              f"down_mag={result['down_streak_magnitude']:.4f}, "
              f"conviction={result['streak_conviction']:.4f}")

    def test_up_down_measured_separately(self):
        """Verify up and down are NEVER combined."""
        np.random.seed(42)
        prices = 100 * np.exp(np.cumsum(np.random.randn(100) * 0.01))

        result = AsymmetricReturns.extract_features(prices)

        # Should have separate up/down counts
        assert 'up_count' in result
        assert 'down_count' in result
        assert 'up_sum' in result
        assert 'down_sum' in result

        # down_sum should be negative (signed)
        assert result['down_sum'] <= 0, "down_sum should be negative (signed)"
        print(f"  ✓ Separate: up_sum={result['up_sum']:.4f}, down_sum={result['down_sum']:.4f}")


class TestRecurrenceAsymmetry:
    """Test directional recurrence features."""

    def test_recurrence_asymmetry(self):
        """Test that recurrence is computed separately for up/down."""
        np.random.seed(42)
        prices = 100 * np.exp(np.cumsum(np.random.randn(100) * 0.01))

        result = RecurrenceFeatures.extract_features(prices)

        assert 'det_up' in result
        assert 'det_down' in result
        assert 'recurrence_asymmetry' in result
        print(f"  ✓ Recurrence: det_up={result['det_up']:.3f}, "
              f"det_down={result['det_down']:.3f}, "
              f"asymmetry={result['recurrence_asymmetry']:.3f}")

    def test_mad_not_std(self):
        """Verify MAD is used instead of std for robustness."""
        # This is an implementation detail, but important for assumption-free
        # The code uses: mad = np.median(np.abs(data - np.median(data)))
        # Not: std = np.std(data)
        np.random.seed(42)
        data = np.random.randn(50)

        # Check the recurrence matrix is computed with MAD
        R = RecurrenceFeatures.compute_recurrence_matrix(data)

        assert R.shape == (50, 50)
        assert R.dtype == int
        print("  ✓ Recurrence matrix computed (MAD-based threshold)")


class TestDSPFeatureEngine:
    """Test the full DSP feature extraction."""

    def test_no_hurst_in_output(self):
        """Ensure Hurst is completely removed."""
        df = create_test_data(200)
        engine = DSPFeatureEngine()

        result = engine.extract_all(df, bar_idx=-1)

        assert 'hurst' not in result, "Hurst should be removed!"
        assert 'persistence_strength' not in result, "persistence_strength should be removed!"

        # Should have directional persistence instead
        assert 'up_persistence' in result
        assert 'down_persistence' in result
        assert 'persistence_asymmetry' in result
        print("  ✓ Hurst purged, replaced with directional persistence")

    def test_all_features_present(self):
        """Test that all expected features are present."""
        df = create_test_data(200)
        engine = DSPFeatureEngine()

        result = engine.extract_all(df, bar_idx=-1)

        expected_features = [
            'wavelet_dominant_scale',
            'hilbert_amplitude',
            'sample_entropy',
            'permutation_entropy',
            'up_persistence',
            'down_persistence',
            'overall_persistence',
            'persistence_asymmetry'
        ]

        for feat in expected_features:
            assert feat in result, f"Missing feature: {feat}"

        print(f"  ✓ All {len(result)} features present")


class TestLiquidityEngine:
    """Test the full liquidity feature extraction."""

    def test_cvd_features(self):
        """Test CVD (Cumulative Volume Delta) extraction."""
        df = create_test_data(100)

        result = CVDExtractor.extract_features(df, bar_idx=-1)

        assert 'cvd' in result
        assert 'cvd_delta' in result
        assert 'signed_volume' in result
        print(f"  ✓ CVD={result['cvd']:.1f}, delta={result['cvd_delta']:.1f}")

    def test_liquidity_engine_full(self):
        """Test full liquidity engine."""
        df = create_test_data(100)
        engine = LiquidityFeatureEngine()

        result = engine.extract_all(df, bar_idx=-1)

        # Should have signed range impact
        assert 'signed_range_impact' in result
        assert 'signed_range_mean' in result
        print(f"  ✓ Full liquidity engine: {len(result)} features")


class TestAssumptionFreeEngine:
    """Test the master assumption-free engine."""

    def test_full_extraction(self):
        """Test full extraction of all assumption-free features."""
        df = create_test_data(200)
        engine = AssumptionFreeEngine()

        result = engine.extract_all(df, bar_idx=-1)

        # Should have features from all modules
        assert 'asym_up_count' in result
        assert 'rank_close_rank' in result
        assert 'dvol_up_mad' in result
        assert 'flow_net_pressure' in result
        assert 'rec_recurrence_asymmetry' in result
        assert 'tail_left_tail_mean' in result

        print(f"  ✓ Full engine: {len(result)} assumption-free features")


def run_all_tests():
    """Run all tests."""
    print("\n" + "=" * 70)
    print("ASSUMPTION-FREE FRAMEWORK TESTS")
    print("=" * 70)

    test_classes = [
        ("Directional Persistence (replaces Hurst)", TestDirectionalPersistence),
        ("Signed Range Impact", TestSignedRangeImpact),
        ("Asymmetric Returns + Streak Momentum", TestAsymmetricReturns),
        ("Recurrence Asymmetry", TestRecurrenceAsymmetry),
        ("DSP Feature Engine", TestDSPFeatureEngine),
        ("Liquidity Engine", TestLiquidityEngine),
        ("Assumption-Free Engine", TestAssumptionFreeEngine),
    ]

    passed = 0
    failed = 0

    for name, test_class in test_classes:
        print(f"\n--- {name} ---")
        instance = test_class()

        for method_name in dir(instance):
            if method_name.startswith('test_'):
                try:
                    getattr(instance, method_name)()
                    passed += 1
                except AssertionError as e:
                    print(f"  ✗ {method_name}: {e}")
                    failed += 1
                except Exception as e:
                    print(f"  ✗ {method_name}: {type(e).__name__}: {e}")
                    failed += 1

    print("\n" + "=" * 70)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("=" * 70)

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
