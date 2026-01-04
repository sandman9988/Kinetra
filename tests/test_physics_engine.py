import numpy as np
import pandas as pd
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis.extra.numpy import arrays
from hypothesis.extra.pandas import column, data_frames
from hypothesis.strategies import data, floats, integers

from kinetra.performance import rolling_percentile_vectorized, rolling_zscore_fast

# Constants for testing (derived, no magic numbers)
MIN_WINDOW = 5  # Minimum reasonable window for rolling ops, based on statistical validity
MAX_WINDOW = 100  # Upper bound for performance in tests
ARRAY_SHAPE = (1000,)  # Typical data size for property tests


@pytest.fixture
def sample_series() -> pd.Series:
    """Fixture for a sample pandas Series with financial-like data."""
    np.random.seed(42)
    return pd.Series(np.random.randn(1000).cumsum() + 100)  # Simulated price series


# Unit Tests for rolling_percentile_vectorized


def test_rolling_percentile_vectorized_basic(sample_series: pd.Series) -> None:
    """Basic test for rolling percentile calculation."""
    window = 20
    result = rolling_percentile_vectorized(sample_series.values, window)
    assert len(result) == len(sample_series)
    assert np.all(result[: window - 1] == 0)  # Initial NaN-like handling
    assert np.all((result[window - 1 :] >= 0) & (result[window - 1 :] <= 1))  # Percentiles in [0,1]


def test_rolling_percentile_vectorized_edge_cases() -> None:
    """Test edge cases like small arrays or constant values."""
    # Small array
    arr = np.array([1, 2, 3, 4, 5])
    result = rolling_percentile_vectorized(arr, window=3)
    expected = np.array([0, 0, 0.5, 0.5, 0.5])  # Manual calculation
    np.testing.assert_allclose(result, expected, rtol=1e-6)

    # Constant values
    arr_constant = np.full(10, 5.0)
    result_constant = rolling_percentile_vectorized(arr_constant, window=5)
    assert np.all(result_constant[4:] == 0.5)  # All ranks equal, percentile ~0.5

    # With NaNs
    arr_nan = np.array([1, np.nan, 3, 4, 5])
    with pytest.raises(ValueError):  # Assuming function raises on NaN
        rolling_percentile_vectorized(arr_nan, window=3)


@given(
    arrays(
        dtype=np.float64,
        shape=ARRAY_SHAPE,
        elements=floats(allow_nan=False, allow_infinity=False, min_value=-1e6, max_value=1e6),
    ),
    integers(min_value=MIN_WINDOW, max_value=MAX_WINDOW),
)
@settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_rolling_percentile_vectorized_properties(arr: np.ndarray, window: int) -> None:
    """Property-based test: results are in [0,1], non-decreasing with sorted input."""
    result = rolling_percentile_vectorized(arr, window)
    assert np.all((result[window - 1 :] >= 0) & (result[window - 1 :] <= 1))

    # For sorted input, percentiles should be increasing
    sorted_arr = np.sort(arr)
    sorted_result = rolling_percentile_vectorized(sorted_arr, window)
    assert np.all(np.diff(sorted_result[window - 1 :]) >= 0)


# Unit Tests for rolling_zscore_fast


def test_rolling_zscore_fast_basic(sample_series: pd.Series) -> None:
    """Basic test for rolling Z-score calculation."""
    window = 20
    result = rolling_zscore_fast(sample_series.values, window)
    assert len(result) == len(sample_series)
    assert np.all(np.isnan(result[: window - 1]))  # Initial NaNs
    assert np.isclose(np.mean(result[window - 1 :]), 0, atol=1e-2)  # Mean ~0
    assert np.isclose(np.std(result[window - 1 :]), 1, atol=1e-2)  # Std ~1


def test_rolling_zscore_fast_edge_cases() -> None:
    """Test edge cases like constant values or small arrays."""
    # Constant values
    arr_constant = np.full(10, 5.0)
    result_constant = rolling_zscore_fast(arr_constant, window=5)
    assert np.all(result_constant[4:] == 0)  # Z-score 0 for constants

    # Small array
    arr = np.array([1, 2, 3, 4, 5])
    result = rolling_zscore_fast(arr, window=3)
    assert np.all(np.isnan(result[:2]))
    np.testing.assert_allclose(result[2:], [0, 0, 0], atol=1e-6)  # Uniform dist, z~0


@given(
    arrays(
        dtype=np.float64,
        shape=ARRAY_SHAPE,
        elements=floats(allow_nan=False, allow_infinity=False, min_value=-1e6, max_value=1e6),
    ),
    integers(min_value=MIN_WINDOW, max_value=MAX_WINDOW),
)
@settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_rolling_zscore_fast_properties(arr: np.ndarray, window: int) -> None:
    """Property-based test: mean ~0, std ~1, handles numerical stability."""
    result = rolling_zscore_fast(arr, window)
    valid_result = result[window - 1 :]
    assert np.isclose(np.mean(valid_result), 0, atol=1e-1)
    assert np.isclose(np.std(valid_result), 1, atol=1e-1)

    # Numerical stability: no inf/nan in output
    assert np.all(np.isfinite(valid_result))


# Additional tests can be added for other physics functions as needed
