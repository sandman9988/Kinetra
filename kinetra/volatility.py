"""
Volatility Estimators for MAE/MFE Normalization

Primary: Rogers-Satchell (handles drift in trending markets)
Fallback: Yang-Zhang (handles overnight gaps)

NO ATR - it's lagging and doesn't handle drift well.
"""

import numpy as np
import pandas as pd


def rogers_satchell(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Rogers-Satchell volatility estimator.

    Handles drift (trending markets) - ideal for berserker conditions.

    Formula:
    σ² = ln(H/C)*ln(H/O) + ln(L/C)*ln(L/O)

    Properties:
    - Uses OHLC data efficiently
    - Handles trending markets (drift)
    - No overnight component (use yang_zhang if overnight matters)
    - 14.5% variance reduction vs ATR empirically

    Args:
        df: DataFrame with 'open', 'high', 'low', 'close' columns
        period: Rolling window for averaging

    Returns:
        Volatility in price units (multiply by 100/close for %)
    """
    log_hc = np.log(df['high'] / df['close'])
    log_ho = np.log(df['high'] / df['open'])
    log_lc = np.log(df['low'] / df['close'])
    log_lo = np.log(df['low'] / df['open'])

    rs_var = log_hc * log_ho + log_lc * log_lo

    # Clip negative variance (numerical edge cases)
    vol = np.sqrt(rs_var.rolling(period).mean().clip(lower=1e-10))

    # Return in price units
    return vol * df['close']


def yang_zhang(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Yang-Zhang volatility estimator.

    Most efficient OHLC estimator. Handles overnight gaps.
    Use as fallback when overnight volatility matters.

    Formula:
    σ² = σ_overnight² + k*σ_open-close² + (1-k)*σ_RS²

    Where:
    - σ_overnight = var(ln(Open/Close_prev))
    - σ_open-close = var(ln(Close/Open))
    - σ_RS = Rogers-Satchell variance
    - k = 0.34 / (1.34 + (n+1)/(n-1))

    Args:
        df: DataFrame with 'open', 'high', 'low', 'close' columns
        period: Rolling window for averaging

    Returns:
        Volatility in price units
    """
    # Overnight volatility: open vs previous close
    log_oc_prev = np.log(df['open'] / df['close'].shift())
    overnight_var = log_oc_prev.rolling(period).var()

    # Open-to-close volatility
    log_co = np.log(df['close'] / df['open'])
    oc_var = log_co.rolling(period).var()

    # Rogers-Satchell component
    log_hc = np.log(df['high'] / df['close'])
    log_ho = np.log(df['high'] / df['open'])
    log_lc = np.log(df['low'] / df['close'])
    log_lo = np.log(df['low'] / df['open'])
    rs_var = (log_hc * log_ho + log_lc * log_lo).rolling(period).mean()

    # Yang-Zhang weighting
    k = 0.34 / (1.34 + (period + 1) / (period - 1))

    yz_var = overnight_var + k * oc_var + (1 - k) * rs_var

    # Return in price units
    return np.sqrt(yz_var.clip(lower=1e-10)) * df['close']


def garman_klass(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Garman-Klass volatility estimator.

    8x more efficient than close-to-close.

    Formula:
    σ² = 0.5*(ln(H/L))² - (2*ln(2)-1)*(ln(C/O))²

    Args:
        df: DataFrame with 'open', 'high', 'low', 'close' columns
        period: Rolling window for averaging

    Returns:
        Volatility in price units
    """
    log_hl = np.log(df['high'] / df['low'])
    log_co = np.log(df['close'] / df['open'])

    gk_var = 0.5 * (log_hl ** 2) - (2 * np.log(2) - 1) * (log_co ** 2)

    return np.sqrt(gk_var.rolling(period).mean().clip(lower=1e-10)) * df['close']


def parkinson(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Parkinson volatility estimator.

    Uses only high-low range. 5x more efficient than close-to-close.
    Underestimates when there's drift.

    Formula:
    σ² = (1/4*ln(2)) * (ln(H/L))²

    Args:
        df: DataFrame with 'high', 'low' columns
        period: Rolling window for averaging

    Returns:
        Volatility in price units
    """
    log_hl = np.log(df['high'] / df['low'])
    parkinson_var = (1 / (4 * np.log(2))) * (log_hl ** 2)

    return np.sqrt(parkinson_var.rolling(period).mean()) * df['close']


def compute_volatility(
    df: pd.DataFrame,
    period: int = 14,
    method: str = 'rogers_satchell'
) -> pd.Series:
    """
    Compute volatility using specified method.

    Primary: rogers_satchell (handles drift)
    Fallback: yang_zhang (handles overnight)

    Args:
        df: DataFrame with OHLC columns
        period: Rolling window
        method: 'rogers_satchell', 'yang_zhang', 'garman_klass', 'parkinson'

    Returns:
        Volatility series in price units
    """
    methods = {
        'rogers_satchell': rogers_satchell,
        'yang_zhang': yang_zhang,
        'garman_klass': garman_klass,
        'parkinson': parkinson,
    }

    if method not in methods:
        raise ValueError(f"Unknown method: {method}. Use: {list(methods.keys())}")

    return methods[method](df, period)


def normalize_by_volatility(
    value: pd.Series,
    df: pd.DataFrame,
    period: int = 14,
    method: str = 'rogers_satchell'
) -> pd.Series:
    """
    Normalize a value (like MAE/MFE) by volatility.

    Makes metrics regime-invariant across high/low vol periods.

    Args:
        value: Series to normalize (e.g., MFE in price units)
        df: DataFrame with OHLC for volatility computation
        period: Volatility lookback
        method: Volatility estimation method

    Returns:
        Normalized series (in volatility units, e.g., "1.5x vol")
    """
    vol = compute_volatility(df, period, method)
    return value / vol.clip(lower=1e-10)
