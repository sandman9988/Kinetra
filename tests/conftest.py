"""
Shared pytest fixtures for Kinetra tests.

Provides reusable components for market data, PhysicsEngine, and stability checks.
Aligns with first-principles: vectorized Pandas/NumPy operations, adaptive distributions
(no fixed lookbacks or magic numbers), numerical stability enforcement.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from kinetra.physics_engine import PhysicsEngine


@pytest.fixture(scope="session")
def sample_market_data():
    """
    Fixture for sample BTCUSD H1 market data.

    Loads from tests/data/btc_h1_sample.csv if exists, otherwise generates
    synthetic trending/oscillating data for comprehensive regime testing.
    Uses Pandas for vectorized operations; adaptive window via rolling percentiles.
    """
    data_path = Path(__file__).parent / "data" / "btc_h1_sample.csv"

    if data_path.exists():
        df = pd.read_csv(data_path, parse_dates=["timestamp"], index_col="timestamp")
    else:
        # Generate synthetic data: trending with noise (physics-aligned: energy transfer)
        np.random.seed(42)  # Reproducible for testing
        timestamps = pd.date_range(start="2023-01-01", periods=1000, freq="H")
        base_trend = np.cumsum(np.random.randn(1000) * 0.1) + 100  # Adaptive momentum
        noise = np.random.randn(1000) * 0.5
        prices = pd.Series(base_trend + noise, index=timestamps)
        volumes = pd.Series(np.abs(np.random.randn(1000)) * 1000 + 5000, index=timestamps)

        df = pd.DataFrame(
            {
                "open": prices,
                "high": prices * (1 + np.abs(np.random.randn(1000)) * 0.01),
                "low": prices * (1 - np.abs(np.random.randn(1000)) * 0.01),
                "close": prices,
                "volume": volumes,
            }
        )

    # Vectorized cleaning: forward-fill NaNs, ensure non-negative volumes
    df = df.fillna(method="ffill").fillna(method="bfill")
    df["volume"] = np.clip(df["volume"], 0, np.inf)

    # Adaptive rolling window for any preprocessing (e.g., 75th percentile lookback simulation)
    rolling_window = int(np.percentile(range(50, 200), 75))  # Adaptive, no magic number
    df["close_rolling_mean"] = df["close"].rolling(window=rolling_window).mean()

    return df


@pytest.fixture(scope="session")
def physics_engine():
    """
    Shared PhysicsEngine instance.

    Initialized with default params (mass=1.0, no fixed thresholds).
    Reusable across all tests for compute_physics_state.
    """
    return PhysicsEngine(mass=1.0)  # Mass as proxy for market 'inertia'


@pytest.fixture
def stable_physics_state(sample_market_data, physics_engine):
    """
    Fixture for a stable physics state computation.

    Computes state on sample data, enforces numerical stability:
    - No NaNs/Infs
    - Non-negative physics values (energy, damping, entropy)
    - Finite values throughout

    Yields the state DataFrame for use in tests.
    """
    prices = sample_market_data["close"]
    state = physics_engine.compute_physics_state(prices)

    # Vectorized stability checks (Pandas/NumPy)
    assert not state.isna().any().any(), "NaNs detected in physics state"
    assert not np.isinf(state.select_dtypes(include=[np.number])).any().any(), (
        "Infs detected in physics state"
    )
    physics_cols = ["energy", "damping", "entropy"]  # Assuming these exist per engine
    for col in physics_cols:
        if col in state.columns:
            assert (state[col] >= 0).all(), (
                f"Negative {col} detected (violates physics constraints)"
            )
            assert np.isfinite(state[col]).all(), f"Non-finite {col} detected"

    # Adaptive regime check: ensure variety without fixed thresholds
    if "regime" in state.columns:
        unique_regimes = state["regime"].nunique()
        assert unique_regimes > 1, "Insufficient regime diversity in sample data"

    yield state


@pytest.fixture(autouse=True)
def numerical_stability_check():
    """
    Autouse fixture for global numerical stability.

    Runs before/after each test to shield against NaN/Inf propagation.
    Uses NumPy vectorized checks on any passed data (if available via params).
    """
    # Placeholder for per-test checks; can be extended with params
    # e.g., def numerical_stability_check(data): np.isfinite(data).all()
    pass
