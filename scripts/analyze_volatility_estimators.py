#!/usr/bin/env python3
"""
Compare Volatility Estimators for Energy Normalization

Estimators:
1. ATR (Average True Range) - Simple, lagging
2. Yang-Zhang - Uses OHLC efficiently, less biased
3. Parkinson - Uses High/Low only
4. Rogers-Satchell - Uses OHLC, handles drift
5. Garman-Klass - Uses OHLC, more efficient than close-to-close

Goal: Find best normalizer for MAE/MFE that gives regime-invariance.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from kinetra.mt5_connector import load_csv_data
from kinetra.physics_engine import PhysicsEngine


def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Standard ATR."""
    high_low = df['high'] - df['low']
    high_close = (df['high'] - df['close'].shift()).abs()
    low_close = (df['low'] - df['close'].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def compute_parkinson(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Parkinson volatility estimator.
    Uses high-low range. Efficiency factor 5x over close-to-close.

    σ² = (1/4*ln(2)) * (ln(H/L))²
    """
    log_hl = np.log(df['high'] / df['low'])
    parkinson_var = (1 / (4 * np.log(2))) * (log_hl ** 2)
    return np.sqrt(parkinson_var.rolling(period).mean()) * df['close']


def compute_garman_klass(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Garman-Klass volatility estimator.
    Uses OHLC, more efficient than Parkinson.

    σ² = 0.5*(ln(H/L))² - (2*ln(2)-1)*(ln(C/O))²
    """
    log_hl = np.log(df['high'] / df['low'])
    log_co = np.log(df['close'] / df['open'])

    gk_var = 0.5 * (log_hl ** 2) - (2 * np.log(2) - 1) * (log_co ** 2)
    return np.sqrt(gk_var.rolling(period).mean().clip(lower=0)) * df['close']


def compute_rogers_satchell(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Rogers-Satchell volatility estimator.
    Handles drift (trending markets).

    σ² = ln(H/C)*ln(H/O) + ln(L/C)*ln(L/O)
    """
    log_hc = np.log(df['high'] / df['close'])
    log_ho = np.log(df['high'] / df['open'])
    log_lc = np.log(df['low'] / df['close'])
    log_lo = np.log(df['low'] / df['open'])

    rs_var = log_hc * log_ho + log_lc * log_lo
    return np.sqrt(rs_var.rolling(period).mean().clip(lower=0)) * df['close']


def compute_yang_zhang(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Yang-Zhang volatility estimator.
    Combines overnight and open-to-close volatility.
    Most efficient OHLC estimator.

    σ² = σ_overnight² + k*σ_open-close² + (1-k)*σ_RS²

    Where:
    - σ_overnight = var(ln(Open/Close_prev))
    - σ_open-close = var(ln(Close/Open))
    - σ_RS = Rogers-Satchell variance
    - k = 0.34 / (1.34 + (n+1)/(n-1))
    """
    # Overnight volatility: open vs previous close
    log_oc_prev = np.log(df['open'] / df['close'].shift())
    overnight_var = log_oc_prev.rolling(period).var()

    # Open-to-close volatility
    log_co = np.log(df['close'] / df['open'])
    oc_var = log_co.rolling(period).var()

    # Rogers-Satchell
    log_hc = np.log(df['high'] / df['close'])
    log_ho = np.log(df['high'] / df['open'])
    log_lc = np.log(df['low'] / df['close'])
    log_lo = np.log(df['low'] / df['open'])
    rs_var = (log_hc * log_ho + log_lc * log_lo).rolling(period).mean()

    # Yang-Zhang weighting
    k = 0.34 / (1.34 + (period + 1) / (period - 1))

    yz_var = overnight_var + k * oc_var + (1 - k) * rs_var
    return np.sqrt(yz_var.clip(lower=0)) * df['close']


def test_normalizer(df: pd.DataFrame, signals: pd.Index, direction_col: str, vol_col: str):
    """
    Test a volatility normalizer for MAE/MFE regime-invariance.
    """
    # Split into high/low vol regimes
    vol_pct = df[vol_col].rolling(100).apply(
        lambda x: (x.iloc[-1] > x.iloc[:-1]).mean() if len(x) > 1 else 0.5, raw=False
    ).fillna(0.5)

    high_vol_signals = [s for s in signals if s in df.index and vol_pct.loc[s] > 0.7]
    low_vol_signals = [s for s in signals if s in df.index and vol_pct.loc[s] < 0.3]

    results = {'high_vol': {'raw': [], 'norm': []}, 'low_vol': {'raw': [], 'norm': []}}

    for regime, regime_signals in [('high_vol', high_vol_signals), ('low_vol', low_vol_signals)]:
        for bar in regime_signals:
            idx = df.index.get_loc(bar)
            direction = int(df.loc[bar, direction_col])
            if direction == 0 or idx + 5 >= len(df):
                continue

            entry_price = df.iloc[idx]['close']
            vol = df.iloc[idx][vol_col]

            # 5-bar MFE
            if direction == 1:
                mfe = max((df.iloc[idx + i]['high'] - entry_price) / entry_price * 100 for i in range(1, 6))
            else:
                mfe = max((entry_price - df.iloc[idx + i]['low']) / entry_price * 100 for i in range(1, 6))

            vol_pct_price = vol / entry_price * 100 if entry_price > 0 else 0.01

            results[regime]['raw'].append(mfe)
            results[regime]['norm'].append(mfe / vol_pct_price if vol_pct_price > 0 else mfe)

    return results


def main():
    # Load data
    project_root = Path(__file__).parent.parent
    csv_files = list(project_root.glob("*BTCUSD*.csv"))
    if not csv_files:
        print("No BTCUSD CSV file found")
        return

    data = load_csv_data(str(csv_files[0]))
    print(f"Loaded {len(data)} bars")

    # Compute physics
    engine = PhysicsEngine(lookback=20)
    physics = engine.compute_physics_state(data['close'], data['volume'], include_percentiles=True)

    df = data.copy()
    df['energy_pct'] = physics['energy_pct']
    df['damping_pct'] = physics['damping_pct']
    df['momentum_5'] = df['close'].pct_change(5)
    df['counter_direction'] = -np.sign(df['momentum_5'])

    # Compute all volatility estimators
    print("\nComputing volatility estimators...")
    df['atr'] = compute_atr(df, period=14)
    df['parkinson'] = compute_parkinson(df, period=14)
    df['garman_klass'] = compute_garman_klass(df, period=14)
    df['rogers_satchell'] = compute_rogers_satchell(df, period=14)
    df['yang_zhang'] = compute_yang_zhang(df, period=14)

    # Flow consistency
    df['return_sign'] = np.sign(df['close'].pct_change())
    df['flow_consistency'] = df['return_sign'].rolling(5).apply(
        lambda x: (x == x.iloc[-1]).mean() if len(x) > 0 else 0.5, raw=False
    ).fillna(0.5)

    # Volume percentile
    vol_window = min(500, len(df))
    df['volume_pct'] = df['volume'].rolling(vol_window).apply(
        lambda x: (x.iloc[-1] > x.iloc[:-1]).mean() if len(x) > 1 else 0.5, raw=False
    ).fillna(0.5)

    df = df.dropna()

    # Signals
    berserker = (df['energy_pct'] > 0.75) & (df['damping_pct'] < 0.25)
    berserker_plus = berserker & (df['flow_consistency'] > 0.7) & (df['volume_pct'] > 0.6)

    print(f"Berserker+ signals: {berserker_plus.sum()}")

    print("\n" + "=" * 80)
    print("VOLATILITY ESTIMATOR COMPARISON")
    print("=" * 80)

    estimators = ['atr', 'parkinson', 'garman_klass', 'rogers_satchell', 'yang_zhang']

    print(f"\n{'Estimator':>18} │ {'High Vol CV':>12} │ {'Low Vol CV':>12} │ {'Variance Red.':>14}")
    print("─" * 65)

    best_estimator = None
    best_reduction = -float('inf')

    for est in estimators:
        results = test_normalizer(df, df[berserker_plus].index, 'counter_direction', est)

        # Coefficient of variation for each regime
        if len(results['high_vol']['raw']) < 30 or len(results['low_vol']['raw']) < 10:
            print(f"{est:>18} │ {'<30 signals':>12} │ {'-':>12} │ {'-':>14}")
            continue

        # High vol
        raw_cv_hv = np.std(results['high_vol']['raw']) / np.mean(results['high_vol']['raw']) if np.mean(results['high_vol']['raw']) > 0 else 0
        norm_cv_hv = np.std(results['high_vol']['norm']) / np.mean(results['high_vol']['norm']) if np.mean(results['high_vol']['norm']) > 0 else 0

        # Low vol
        raw_cv_lv = np.std(results['low_vol']['raw']) / np.mean(results['low_vol']['raw']) if np.mean(results['low_vol']['raw']) > 0 else 0
        norm_cv_lv = np.std(results['low_vol']['norm']) / np.mean(results['low_vol']['norm']) if np.mean(results['low_vol']['norm']) > 0 else 0

        # Overall variance reduction
        overall_raw_cv = (raw_cv_hv + raw_cv_lv) / 2
        overall_norm_cv = (norm_cv_hv + norm_cv_lv) / 2
        reduction = (1 - overall_norm_cv / overall_raw_cv) * 100 if overall_raw_cv > 0 else 0

        print(f"{est:>18} │ {norm_cv_hv:>12.3f} │ {norm_cv_lv:>12.3f} │ {reduction:>+13.1f}%")

        if reduction > best_reduction:
            best_reduction = reduction
            best_estimator = est

    print("\n" + "─" * 65)
    print(f"BEST: {best_estimator} with {best_reduction:+.1f}% variance reduction")

    # === DETAILED COMPARISON: ATR vs YANG ZHANG ===
    print("\n" + "=" * 80)
    print("DETAILED COMPARISON: ATR vs YANG-ZHANG")
    print("=" * 80)

    for est in ['atr', 'yang_zhang']:
        results = test_normalizer(df, df[berserker_plus].index, 'counter_direction', est)

        print(f"\n  {est.upper()}:")

        for regime in ['high_vol', 'low_vol']:
            raw = results[regime]['raw']
            norm = results[regime]['norm']

            if len(raw) < 10:
                print(f"    {regime}: <10 signals")
                continue

            print(f"    {regime} ({len(raw)} signals):")
            print(f"      Raw MFE:  mean={np.mean(raw):.4f}%, std={np.std(raw):.4f}%")
            print(f"      Norm MFE: mean={np.mean(norm):.2f}x, std={np.std(norm):.2f}x")

    # === CROSS-REGIME CONSISTENCY ===
    print("\n" + "=" * 80)
    print("CROSS-REGIME CONSISTENCY (Goal: same normalized MFE across regimes)")
    print("=" * 80)

    for est in ['atr', 'yang_zhang']:
        results = test_normalizer(df, df[berserker_plus].index, 'counter_direction', est)

        hv_mean = np.mean(results['high_vol']['norm']) if results['high_vol']['norm'] else 0
        lv_mean = np.mean(results['low_vol']['norm']) if results['low_vol']['norm'] else 0

        # Difference between regimes (lower = more consistent)
        regime_diff = abs(hv_mean - lv_mean) / max(hv_mean, lv_mean) * 100 if max(hv_mean, lv_mean) > 0 else 0

        print(f"\n  {est.upper()}:")
        print(f"    High Vol normalized mean: {hv_mean:.2f}x")
        print(f"    Low Vol normalized mean:  {lv_mean:.2f}x")
        print(f"    Cross-regime difference:  {regime_diff:.1f}%")

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("""
  Volatility Estimator Properties:

  ATR (Average True Range):
    - Simple: TR = max(H-L, |H-C_prev|, |L-C_prev|)
    - Lagging, equally weights all components
    - Doesn't handle drift well

  Parkinson:
    - Uses only High-Low range
    - 5x more efficient than close-to-close
    - Underestimates vol when there's drift

  Garman-Klass:
    - Uses OHLC
    - 8x more efficient than close-to-close
    - Still biased by overnight gaps

  Rogers-Satchell:
    - Uses OHLC
    - Handles drift (trending markets)
    - No overnight component

  Yang-Zhang:
    - Combines overnight + open-to-close + Rogers-Satchell
    - Most efficient OHLC estimator
    - Handles both drift and overnight gaps

  RECOMMENDATION:
    Use YANG-ZHANG for MAE/MFE normalization in ARS formula:
    R_t = (PnL / E_t) + α·(MFE/YZ) - β·(MAE/YZ) - γ·Time
""")


if __name__ == "__main__":
    main()
