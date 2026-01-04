"""
Tests for Denoise Filters Module
=================================

Property-based tests for non-linear denoising.
"""
# ruff: noqa: I001
import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from kinetra.denoise_filters import (
    DenoiseMethod,
    denoise_ohlc,
    denoise_prices,
    savgol_denoise,
    median_denoise,
    _detect_dominant_cycle,
)


# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def sample_ohlc_data():
    """Generate sample OHLC data."""
    np.random.seed(42)
    n = 500

    # Simulate price walk with trend
    returns = np.random.normal(0.001, 0.02, n)
    prices = 100 * np.exp(np.cumsum(returns))

    df = pd.DataFrame(
        {
            "time": pd.date_range("2024-01-01", periods=n, freq="H"),
            "open": prices + np.random.normal(0, 0.5, n),
            "high": prices + np.abs(np.random.normal(1, 0.5, n)),
            "low": prices - np.abs(np.random.normal(1, 0.5, n)),
            "close": prices,
            "volume": np.random.randint(1000, 10000, n),
        }
    )

    return df


# ============================================================================
# PROPERTY TESTS
# ============================================================================


@given(
    st.lists(
        st.floats(min_value=1.0, max_value=10000.0, allow_nan=False, allow_infinity=False),
        min_size=100,
        max_size=1000,
    )
)
@settings(deadline=None, max_examples=10)
def test_savgol_always_finite(prices):
    """Property: Savitzky-Golay output is always finite."""
    prices_arr = np.array(prices)
    denoised, window = savgol_denoise(prices_arr)

    assert np.all(np.isfinite(denoised)), "Denoised prices contain NaN or Inf"
    assert len(denoised) == len(prices_arr), "Length mismatch"


@given(
    st.lists(
        st.floats(min_value=1.0, max_value=10000.0, allow_nan=False, allow_infinity=False),
        min_size=100,
        max_size=1000,
    )
)
@settings(deadline=None, max_examples=10)
def test_median_always_finite(prices):
    """Property: Median filter output is always finite."""
    prices_arr = np.array(prices)
    denoised, window = median_denoise(prices_arr)

    assert np.all(np.isfinite(denoised)), "Denoised prices contain NaN or Inf"
    assert len(denoised) == len(prices_arr), "Length mismatch"


@given(
    st.lists(
        st.floats(min_value=1.0, max_value=10000.0, allow_nan=False, allow_infinity=False),
        min_size=100,
        max_size=1000,
    )
)
@settings(deadline=None, max_examples=10)
def test_denoising_reduces_volatility(prices):
    """Property: Denoising should reduce volatility (or keep it same for constant)."""
    prices_arr = np.array(prices)

    # Original volatility
    original_vol = np.std(np.diff(prices_arr))

    # Denoised volatility
    denoised, _ = savgol_denoise(prices_arr)
    denoised_vol = np.std(np.diff(denoised))

    # Denoising should reduce volatility (or keep it near zero for constant prices)
    # Allow for numerical precision edge cases
    if original_vol < 1e-10:  # Essentially constant
        assert denoised_vol < 1e-8, "Denoising added volatility to constant series"
    else:
        assert denoised_vol <= original_vol * 1.2, "Denoising increased volatility significantly"


# ============================================================================
# UNIT TESTS
# ============================================================================


def test_savgol_basic(sample_ohlc_data):
    """Test basic Savitzky-Golay denoising."""
    df = sample_ohlc_data
    prices = df["close"].values

    denoised, window = savgol_denoise(prices)

    # Check output
    assert len(denoised) == len(prices)
    assert np.all(np.isfinite(denoised))
    assert window > 0
    assert window % 2 == 1  # Must be odd


def test_median_basic(sample_ohlc_data):
    """Test basic median filter."""
    df = sample_ohlc_data
    prices = df["close"].values

    denoised, window = median_denoise(prices)

    # Check output
    assert len(denoised) == len(prices)
    assert np.all(np.isfinite(denoised))
    assert window > 0


def test_denoise_prices_function(sample_ohlc_data):
    """Test high-level denoise_prices function."""
    df = sample_ohlc_data

    result = denoise_prices(df, price_col="close", method=DenoiseMethod.SAVGOL)

    # Check result structure
    assert result.denoised_prices is not None
    assert len(result.denoised_prices) == len(df)
    assert result.volatility_reduction_pct >= 0
    assert result.dominant_cycle_bars > 0
    assert result.method_used == "savitzky_golay"


def test_denoise_ohlc_function(sample_ohlc_data):
    """Test denoising all OHLC columns."""
    df = sample_ohlc_data

    df_denoised = denoise_ohlc(df, method=DenoiseMethod.SAVGOL)

    # Check all denoised columns created
    assert "open_denoised" in df_denoised.columns
    assert "high_denoised" in df_denoised.columns
    assert "low_denoised" in df_denoised.columns
    assert "close_denoised" in df_denoised.columns

    # Check all finite
    for col in ["open_denoised", "high_denoised", "low_denoised", "close_denoised"]:
        assert df_denoised[col].notna().all()
        assert np.isfinite(df_denoised[col]).all()


def test_cycle_detection():
    """Test DSP cycle detection."""
    # Create synthetic signal with known cycle
    n = 500
    t = np.arange(n)
    period = 50  # bars

    # Signal with dominant 50-bar cycle + noise
    signal = 100 + 10 * np.sin(2 * np.pi * t / period) + np.random.normal(0, 1, n)

    detected_cycle = _detect_dominant_cycle(signal)

    # Should detect cycle close to 50 (within 20% tolerance)
    assert detected_cycle is not None
    assert 40 <= detected_cycle <= 60, f"Expected ~50, got {detected_cycle}"


def test_noise_removal():
    """Test that noise is actually removed."""
    # Create signal with high-frequency noise
    n = 500
    t = np.arange(n)

    # Smooth trend
    trend = 100 + 0.1 * t

    # High-frequency noise
    noise = 5 * np.sin(2 * np.pi * t / 3)  # 3-bar cycle (high freq)

    signal = trend + noise

    # Denoise
    denoised, _ = savgol_denoise(signal)

    # Denoised should be closer to trend than original
    original_error = np.mean((signal - trend) ** 2)
    denoised_error = np.mean((denoised - trend) ** 2)

    assert denoised_error < original_error, "Denoising did not reduce error vs trend"


def test_edge_preservation():
    """Test that sharp edges are preserved."""
    # Create signal with sharp jump
    n = 500
    signal = np.ones(n) * 100

    # Sharp jump in middle
    signal[250:] = 150

    # Add noise
    signal += np.random.normal(0, 1, n)

    # Denoise
    denoised, _ = savgol_denoise(signal, polyorder=3)

    # Jump should still be present (check around transition)
    jump_preserved = denoised[260] > denoised[240]
    assert jump_preserved, "Sharp edge was over-smoothed"


# ============================================================================
# NUMERICAL STABILITY TESTS
# ============================================================================


def test_no_nan_on_small_input():
    """Test no NaN on minimum viable input."""
    prices = np.array([100.0, 101.0, 102.0, 101.5, 103.0])

    # Should not crash or produce NaN
    denoised, window = savgol_denoise(prices, window_length=3, polyorder=1)

    assert np.all(np.isfinite(denoised))
    assert len(denoised) == len(prices)


def test_constant_input():
    """Test denoising constant prices."""
    prices = np.ones(100) * 100.0

    denoised, _ = savgol_denoise(prices)

    # Should remain constant
    assert np.allclose(denoised, 100.0)


def test_extreme_volatility():
    """Test with extreme price swings."""
    # Create wild swings
    prices = np.array([100, 200, 50, 300, 10, 250] * 20)

    denoised, _ = median_denoise(prices)

    # Should smooth but remain finite
    assert np.all(np.isfinite(denoised))
    assert np.std(denoised) < np.std(prices)  # Should reduce volatility
