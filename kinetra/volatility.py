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
    log_hc = np.log(df["high"] / df["close"])
    log_ho = np.log(df["high"] / df["open"])
    log_lc = np.log(df["low"] / df["close"])
    log_lo = np.log(df["low"] / df["open"])

    rs_var = log_hc * log_ho + log_lc * log_lo

    # Clip negative variance (numerical edge cases)
    vol = np.sqrt(rs_var.rolling(period).mean().clip(lower=1e-10))

    # Return in price units
    return vol * df["close"]


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
    log_oc_prev = np.log(df["open"] / df["close"].shift())
    overnight_var = log_oc_prev.rolling(period).var()

    # Open-to-close volatility
    log_co = np.log(df["close"] / df["open"])
    oc_var = log_co.rolling(period).var()

    # Rogers-Satchell component
    log_hc = np.log(df["high"] / df["close"])
    log_ho = np.log(df["high"] / df["open"])
    log_lc = np.log(df["low"] / df["close"])
    log_lo = np.log(df["low"] / df["open"])
    rs_var = (log_hc * log_ho + log_lc * log_lo).rolling(period).mean()

    # Yang-Zhang weighting
    k = 0.34 / (1.34 + (period + 1) / (period - 1))

    yz_var = overnight_var + k * oc_var + (1 - k) * rs_var

    # Return in price units
    return np.sqrt(yz_var.clip(lower=1e-10)) * df["close"]


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
    log_hl = np.log(df["high"] / df["low"])
    log_co = np.log(df["close"] / df["open"])

    gk_var = 0.5 * (log_hl**2) - (2 * np.log(2) - 1) * (log_co**2)

    return np.sqrt(gk_var.rolling(period).mean().clip(lower=1e-10)) * df["close"]


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
    log_hl = np.log(df["high"] / df["low"])
    parkinson_var = (1 / (4 * np.log(2))) * (log_hl**2)

    return np.sqrt(parkinson_var.rolling(period).mean()) * df["close"]


def compute_volatility(
    df: pd.DataFrame, period: int = 14, method: str = "rogers_satchell"
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
        "rogers_satchell": rogers_satchell,
        "yang_zhang": yang_zhang,
        "garman_klass": garman_klass,
        "parkinson": parkinson,
    }

    if method not in methods:
        raise ValueError(f"Unknown method: {method}. Use: {list(methods.keys())}")

    return methods[method](df, period)


def normalize_by_volatility(
    value: pd.Series, df: pd.DataFrame, period: int = 14, method: str = "rogers_satchell"
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


# =============================================================================
# POTENTIAL ENERGY METRICS
# =============================================================================


def potential_energy(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Calculate Potential Energy - stored energy from range compression.

    Physics Analogy:
    - Compressed spring stores potential energy
    - Compressed price range (squeeze) stores market potential energy
    - When released, converts to kinetic energy (price movement)

    Formula:
    PE = range_compression * volatility_compression

    Where:
    - range_compression = 1 - (bar_range / avg_range)  [0=expanded, 1=compressed]
    - volatility_compression = 1 - (current_vol / avg_vol)  [0=volatile, 1=quiet]

    High PE = market is coiled, ready for explosive move (Bollinger squeeze analog)

    Args:
        df: DataFrame with OHLC columns
        period: Lookback period for averaging

    Returns:
        Potential energy series [0, 1] where 1 = maximum compression
    """
    # Range compression: how tight is current bar vs average?
    bar_range = df["high"] - df["low"]
    avg_range = bar_range.rolling(period).mean()
    range_compression = (1 - bar_range / avg_range).clip(0, 1)

    # Volatility compression: how quiet is current period vs average?
    returns = df["close"].pct_change()
    current_vol = returns.abs()
    avg_vol = current_vol.rolling(period).mean()
    vol_compression = (1 - current_vol / avg_vol.clip(lower=1e-10)).clip(0, 1)

    # Combined PE: geometric mean of compressions
    pe = np.sqrt(range_compression * vol_compression)

    return pe


def forward_energy_release(df: pd.DataFrame, forward_bars: int = 5) -> pd.Series:
    """
    Calculate Forward Energy Release - the actual move that occurred.

    This is the ACTUAL energy that was released after each bar.
    Used to measure how much of the potential energy was captured.

    Formula:
    FER = max(|high[t:t+n] - close[t]|, |close[t] - low[t:t+n]|) / close[t]

    Args:
        df: DataFrame with OHLC columns
        forward_bars: How many bars forward to look for max move

    Returns:
        Forward energy release as percentage of price
    """
    close = df["close"]

    # Forward rolling max high and min low
    forward_high = df["high"].shift(-1).rolling(forward_bars).max().shift(-forward_bars + 1)
    forward_low = df["low"].shift(-1).rolling(forward_bars).min().shift(-forward_bars + 1)

    # Max move in either direction
    up_move = (forward_high - close) / close
    down_move = (close - forward_low) / close

    fer = np.maximum(up_move.abs(), down_move.abs())

    return fer


def energy_efficiency(df: pd.DataFrame, pe_period: int = 14, forward_bars: int = 5) -> pd.DataFrame:
    """
    Calculate Energy Efficiency metrics for each bar.

    Compares Potential Energy (compression) to Actual Release (movement).

    Returns DataFrame with:
    - potential_energy: How compressed/coiled the market was
    - forward_release: How much it actually moved
    - energy_efficiency: forward_release / potential_energy (can be > 1)
    - energy_pct: Percentile rank of potential energy

    Args:
        df: DataFrame with OHLC columns
        pe_period: Lookback for potential energy calculation
        forward_bars: Forward window for measuring release

    Returns:
        DataFrame with energy metrics
    """
    pe = potential_energy(df, pe_period)
    fer = forward_energy_release(df, forward_bars)

    # Efficiency: how much of the potential was realized?
    # Can be > 1 if release exceeds expected potential
    efficiency = fer / pe.clip(lower=0.01)

    # Percentile rank of PE (adaptive threshold)
    pe_pct = pe.rolling(500, min_periods=20).apply(
        lambda x: (x.iloc[-1] > x[:-1]).mean() if len(x) > 1 else 0.5, raw=False
    )

    return pd.DataFrame(
        {
            "potential_energy": pe,
            "forward_release": fer,
            "energy_efficiency": efficiency,
            "energy_pct": pe_pct,
        }
    )


def analyze_energy_distribution(df: pd.DataFrame, pe_period: int = 14) -> dict:
    """
    Analyze Potential Energy distribution across the dataset.

    Similar to spread distribution analysis - gives baseline understanding
    of PE before trading.

    Args:
        df: DataFrame with OHLC columns
        pe_period: Lookback for PE calculation

    Returns:
        Dictionary with distribution statistics
    """
    pe = potential_energy(df, pe_period).dropna()

    percentiles = [10, 25, 50, 75, 90, 95, 99]
    pct_values = {f"p{p}": pe.quantile(p / 100) for p in percentiles}

    return {
        "count": len(pe),
        "mean": pe.mean(),
        "std": pe.std(),
        "min": pe.min(),
        "max": pe.max(),
        **pct_values,
        # Squeeze detection thresholds
        "high_pe_threshold": pe.quantile(0.75),  # Top 25% = high compression
        "extreme_pe_threshold": pe.quantile(0.90),  # Top 10% = extreme compression
    }
