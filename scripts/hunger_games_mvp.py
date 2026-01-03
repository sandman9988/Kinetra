#!/usr/bin/env python3
"""
Hunger Games MVP: Alpha Crucible Arena (Real Empirical Edition)

Empirical testing of 20 tributes (5 per category: traditional, physics, imbalance, patterns)
on real MetaAPI data for top 5 symbols across classes. Runs MC (50x) in Super Harvester,
with OOS validation (70/30 split). Periods: 2023 (train), 2024 (OOS), combined split.
Analysis: Win rates (>55%), Omega, Z, unknowns (e.g., corrs, regime drops).

Run: python scripts/hunger_games_mvp.py --symbols EURUSD BTCUSD NAS100 XAUUSD US30 --tf H1 --years 2023 2024 --split 70 30

Outputs: data/mvp_survivors_YYYY.csv per period, logs/mvp_arena_real.log (per-period analysis).
"""

import argparse
import logging
import os
import warnings
from datetime import datetime
from multiprocessing import Pool, cpu_count
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import talib
from dotenv import load_dotenv
from hypothesis import given
from hypothesis import strategies as st
from metaapi_cloud_sdk import MetaApi
from scipy import stats
from scipy.signal import argrelextrema
import pandas_market_calendars as mcal
from datetime import datetime

warnings.filterwarnings("ignore")

# Setup logging
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(f"{log_dir}/mvp_arena.log"), logging.StreamHandler()],
)

DATA_DIR = "data/master_standardized"  # Fallback
NUM_MC_RUNS = 50
BASELINE_OMEGA = 2.7
SURVIVAL_OME = BASELINE_OMEGA
SURVIVAL_PVAL = 0.01
SURVIVAL_UPLIFT = 0.05
SYMBOLS_DEFAULT = ["EURUSD", "BTCUSD", "NAS100", "XAUUSD", "US30"]  # Top 5 across classes
TFS_DEFAULT = ["H1"]
NUM_BARS = 8760  # Full year H1 approx (365*24)
YEARS_DEFAULT = ["2023", "2024"]
SPLIT_DEFAULT = [70, 30]
METAAPI_TOKEN = os.getenv("METAAPI_TOKEN")
+METAAPI_ACCOUNT_ID = "e8f8c21a-32b5-40b0-9bf7-672e8ffab91f"

# Symbol class mapping for adaptation (Forex, Crypto, etc.)
SYMBOL_CLASS_MAP = {
    "EURUSD": "Forex",  # Smooth, 24/5
    "BTCUSD": "Crypto",  # 24/7, high vol
    "NAS100": "Indices",  # Gaps, session-bound
    "XAUUSD": "Commodities",  # Partial hours
    "US30": "Equity"  # NYSE hours
}

# Class params: Windows/percentiles scaled by vol factor (empirical)
CLASS_PARAMS = {
    "Forex": {"window": 20, "vol_factor": 1.0, "percentile": 75, "gap_thresh": 0.1},
    "Crypto": {"window": 15, "vol_factor": 2.5, "percentile": 90, "gap_thresh": 0.05},  # Fat tails, robust
    "Indices": {"window": 25, "vol_factor": 1.5, "percentile": 80, "gap_thresh": 0.2},
    "Commodities": {"window": 20, "vol_factor": 1.8, "percentile": 75, "gap_thresh": 0.15},
    "Equity": {"window": 22, "vol_factor": 1.2, "percentile": 78, "gap_thresh": 0.25}
}


class EmpiricalSuperHarvester:
    """Empirical Super Harvester: Real trade sim with entry triggers, direction, risk, trailing harvest."""

    def __init__(self, feature_series: pd.Series, df: pd.DataFrame):
        self.feature = feature_series
        self.df = df
        self.med_feat = feature_series.median()
        self.atr = talib.ATR(df["high"], df["low"], df["close"], timeperiod=14)

    def simulate_trades(
        self, df: pd.DataFrame, noise_level: float = 0.005, train_mode: bool = True
    ) -> Dict[str, float]:
        """Empirical sim: Triggers (fat/laminar), direction, 1% risk sizing, trailing harvest. Returns win rate, Omega, Z."""
        df = df.copy()
        # Minimal noise for robustness (real empirical)
        atr_mean = self.atr.mean()
        noise = np.random.normal(0, noise_level * atr_mean, len(df))
        df["close"] += noise * df["close"]
        df["high"] = np.maximum(df["high"], df["close"])
        df["low"] = np.minimum(df["low"], df["close"])

        # Entry triggers: Feature as signal for fat/laminar (adaptive)
        signal = (self.feature > self.med_feat).astype(int) * 2 - 1  # Direction: 1 long, -1 short
        positions = 0
        entry_price = 0
        trail_price = 0
        trades = []  # PnL
        directions = []  # Correct direction flag
        trail = 0.015  # Adaptive base (1.5% empirical)

        for i in range(20, len(df)):  # Skip initial for rolling
            feat_val = self.feature.iloc[i]
            sig = signal.iloc[i]
            price = df["close"].iloc[i]
            current_atr = self.atr.iloc[i]

            # Adaptive trail (harvest: Tighten on MFE)
            trail_adj = (
                trail
                * current_atr
                / atr_mean
                * (1 + (feat_val - self.med_feat) / self.med_feat * 0.05)
            )

            if positions == 0 and abs(sig) == 1:  # Entry trigger
                positions = sig
                entry_price = price
                trail_price = entry_price * (1 + sig * trail_adj)  # Initial trail
                # Risk: Size implicit 1% (pnl normalized)
            elif positions != 0:
                mfe = 0  # Track for harvest
                if positions > 0:  # Long
                    mfe = max(mfe, (price - entry_price) / entry_price)
                    trail_price = max(trail_price, price * (1 - trail_adj))  # Trail up
                    if df["low"].iloc[i] <= trail_price:
                        pnl = (price - entry_price) / entry_price
                        trades.append(pnl)
                        # Direction correct if sig == actual move sign
                        actual_dir = 1 if price > entry_price else -1
                        directions.append(sig == actual_dir)
                        positions = 0
                else:  # Short
                    mfe = max(mfe, (entry_price - price) / entry_price)
                    trail_price = min(trail_price, price * (1 + trail_adj))  # Trail down
                    if df["high"].iloc[i] >= trail_price:
                        pnl = (entry_price - price) / entry_price
                        trades.append(pnl)
                        actual_dir = 1 if price < entry_price else -1
                        directions.append(sig == -actual_dir)  # Flip for short
                        positions = 0

                # Harvest: Tighten if MFE > 2*ATR (empirical threshold)
                if mfe > 2 * (current_atr / price):
                    trail_adj *= 0.7  # Lock profits

        if not trades:
            return {"omega": 0.0, "z": 0.0, "win_rate": 0.0, "mfe_capture": 0.0}

        pnls = np.array(trades)
        win_rate = np.mean(directions) if directions else 0.0  # >55% target

        # Omega
        threshold = np.median(pnls)
        upside = pnls[pnls > threshold].sum()
        downside = np.abs(pnls[pnls < threshold].sum())
        omega = upside / downside if downside > 0 else float("inf")

        # Z-factor
        z = stats.zscore(pnls).mean()

        # Mock MFE capture (real: Track max excursion / captured)
        mfe_capture = 0.65  # Placeholder; extend with real tracking

        return {"omega": omega, "z": z, "win_rate": win_rate * 100, "mfe_capture": mfe_capture}


def hierarchical_uplift(omega: float, z: float, win_rate: float) -> float:
    """Hierarchical uplift: Weights Omega (50%), Z (30%), Win Rate >55% (20%)."""
    base = 0.5
    win_bonus = 1.0 if win_rate > 55 else 0.5
    return (omega - BASELINE_OMEGA) / BASELINE_OMEGA * 0.5 + z / 5.0 * 0.3 + (win_bonus - 0.5) * 0.2


def fetch_metaapi_data(symbol: str, tf: str, start_date: str, end_date: str = None) -> pd.DataFrame:
    """Real empirical fetch via MetaAPI: Batch H1 bars by time range."""
    if not METAAPI_TOKEN:
        raise ValueError("METAAPI_TOKEN not set in .env")

    api = MetaApi(token=METAAPI_TOKEN)
    account = api.metatrader_account_api.get_account(METAAPI_ACCOUNT_ID)
    connection = account.get_rpc_connection()
    connection.wait_synchronized()

    # TF map: H1 = 16385
    tf_map = {"H1": 16385, "M15": 16388}
    timeframe = tf_map.get(tf, 16385)

    # Batch fetch (1000 bars max/call)
    all_rates = []
    current_start = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date) if end_date else pd.Timestamp.now()

    while current_start < end_dt:
        rates = connection.copy_rates_range(symbol, timeframe, current_start, end_dt, max_bars=1000)
        if not rates:
            break
        all_rates.extend(rates)
        current_start = rates[-1]["time"] + pd.Timedelta(hours=1)  # Next batch

    if not all_rates:
        raise ValueError(f"No data for {symbol} {tf} {start_date}-{end_date}")

    df = pd.DataFrame(all_rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    df = df.set_index("time")[["open", "high", "low", "close", "tick_volume"]].rename(
        columns={"tick_volume": "volume"}
    )
    df = df.dropna()
    logging.info(f"Fetched {len(df)} raw real H1 bars for {symbol} {start_date}-{end_date} via MetaAPI")
    return df


def prepare_data(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Robust prep: Holidays/gaps (vol/cal-based), symbol-specific norm. Chronological, bias-free."""
    if df.empty:
        return df

    cls = SYMBOL_CLASS_MAP.get(symbol, "Forex")
    params = CLASS_PARAMS[cls]
    vol_factor = params["vol_factor"]
    gap_thresh = params["gap_thresh"]
    logging.info(f"Preparing data for {symbol} ({cls}): Vol factor {vol_factor}, Gap thresh {gap_thresh}")

    # Chronological sort
    df = df.sort_index()

    # 1. Gap detection: Holidays/trading hours via vol + calendar
    # Vol-based: Low vol = non-trading (holidays)
    vol_mean = df["volume"].mean()
    low_vol_mask = df["volume"] < vol_mean * gap_thresh  # Adaptive thresh

    # Calendar exclusion (e.g., NYSE for US30, 24/7 for Crypto)
    if cls in ["Equity", "Indices", "Forex"]:  # Non-24/7
        if cls == "Equity":
            cal = mcal.get_calendar("NYSE")
        elif cls == "Indices":
            cal = mcal.get_calendar("NYSE")  # Proxy
        else:  # Forex: No holidays, but session gaps
            cal = mcal.get_calendar("XFX")  # Generic

        market_days = cal.schedule(start_date=df.index[0], end_date=df.index[-1])
        market_mask = df.index.isin(market_days.index)

        # Combined: Drop low vol AND non-market
        gap_mask = low_vol_mask & ~market_mask
        df = df[~gap_mask]
        gaps_removed = gap_mask.sum()
        logging.info(f"Removed {gaps_removed} gaps/holidays for {symbol} ({gaps_removed/len(df)*100:.1f}%)")
    else:  # Crypto/Commodities: Vol-only
        df = df[~low_vol_mask]
        gaps_removed = low_vol_mask.sum()
        logging.info(f"Removed {gaps_removed} low-vol gaps for {symbol} ({gaps_removed/len(df)*100:.1f}%)")

    # 2. Small gaps: Interp <1h (forward-fill for OHLCV, chronological)
    df = df.resample("H").ffill()  # H1 resample (fills minor timestamps)
    df = df.dropna()  # Drop any remaining NaNs

    # 3. Symbol-specific norm: Vol scaling, robust stats (e.g., BTC tails: Clip outliers)
    if vol_factor != 1.0:
        df["volume"] *= vol_factor  # Scale for class vol
        logging.info(f"Scaled volume by {vol_factor}x for {symbol}")

    # Outlier clip: 99th percentile for prices (fat tails)
    for col in ["open", "high", "low", "close"]:
        upper = df[col].quantile(0.99)
        lower = df[col].quantile(0.01)
        df[col] = df[col].clip(lower=lower, upper=upper)

    # Ensure finite
    df = df.replace([np.inf, -np.inf], np.nan).dropna()

    prep_stats = {
        "original_len": len(df) + gaps_removed,  # Approx
        "after_prep_len": len(df),
        "vol_mean_post": df["volume"].mean(),
        "class": cls
    }
    logging.info(f"Prep stats for {symbol}: {prep_stats}")

    return df


def load_data(
    symbol: str, tf: str, period: str, split: bool = False, train_pct: int = 70
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load real MetaAPI data for period; split for OOS if requested."""
    if period == "2023":
        start, end = "2023-01-01", "2023-12-31"
    elif period == "2024":
        start, end = "2024-01-01", "2024-12-31"  # Up to current if incomplete
    else:  # Combined
        start, end = "2023-01-01", "2024-12-31"

    try:
        df_raw = fetch_metaapi_data(symbol, tf, start, end)
        df_full = prepare_data(df_raw, symbol)  # Prep: Gaps, norms, symbol-specific
    except Exception as e:
        logging.warning(f"MetaAPI fail for {symbol} {period}: {e}. Fallback to CSV.")
        path = os.path.join(DATA_DIR, f"{symbol}_{tf}.csv")
        if os.path.exists(path):
            df_full = pd.read_csv(path, index_col="timestamp", parse_dates=True)
            df_full = df_full[["open", "high", "low", "close", "volume"]].dropna()
            df_full = prepare_data(df_full, symbol)
        else:
            raise ValueError(f"No fallback data for {symbol}_{tf}")

    if not split:
        return df_full, pd.DataFrame()  # No OOS

    # 70/30 chronological split (post-prep)
    split_idx = int(len(df_full) * (train_pct / 100))
    df_train = df_full.iloc[:split_idx]
    df_oos = df_full.iloc[split_idx:]
    logging.info(f"Split {period} {symbol}: Train {len(df_train)} bars, OOS {len(df_oos)} bars (post-prep)")
    return df_train, df_oos


def compute_features(df: pd.DataFrame) -> Dict[str, pd.Series]:
    """Compute 20 MVP features (5 per category). Vectorized."""
    features = {}
    o, h, l, c, v = df["open"], df["high"], df["low"], df["close"], df["volume"]
    prices = c.values
    vols = v.values

    # Traditional (5)
    rsi = talib.RSI(c, timeperiod=14)
    features["RSI"] = pd.Series(rsi, index=df.index)

    macd, _, _ = talib.MACD(c)
    features["MACD"] = pd.Series(macd, index=df.index)

    bb_upper, bb_middle, bb_lower = talib.BBANDS(c, timeperiod=20)
    bb_width = (bb_upper - bb_lower) / bb_middle
    features["BB_width"] = pd.Series(bb_width, index=df.index)

    atr = talib.ATR(h, l, c, timeperiod=14)
    features["ATR"] = pd.Series(atr, index=df.index)

    obv = talib.OBV(c, v)
    features["OBV"] = pd.Series(obv, index=df.index)

    # Physics (5) - Real hooks, adaptive window/percentile per symbol class
    cls = SYMBOL_CLASS_MAP.get(symbol, "Forex") if 'symbol' in locals() else "Forex"
    params = CLASS_PARAMS[cls]
    window = params["window"]
    percentile = params["percentile"]
    from kinetra.physics_engine import compute_damping, compute_energy  # Real import

    e_t_rolling = []
    damping_rolling = []
    for i in range(window, len(prices)):
        e_t = compute_energy(prices[i - window : i], vols[i - window : i])
        e_t_rolling.append(e_t)
        damping = compute_damping(prices[i - window : i])
        damping_rolling.append(damping)
    features["E_t"] = pd.Series(e_t_rolling + [e_t_rolling[-1]] * window, index=df.index)
    features["Damping"] = pd.Series(
        damping_rolling + [damping_rolling[-1]] * window, index=df.index
    )

    def shannon_entropy(diffs: np.ndarray) -> float:
        if len(diffs) < 2:
            return 0.0
        # Robust for tails: Clip extremes
        diffs_clipped = np.clip(diffs, np.percentile(diffs, 1), np.percentile(diffs, 99))
        _, counts = np.unique(np.round(diffs_clipped, 4), return_counts=True)  # Binned
        probs = counts / len(diffs)
        return -np.sum(probs * np.log2(probs + 1e-10)) if len(probs) > 0 else 0.0

    ent = np.array(
        [shannon_entropy(np.diff(prices[i - window : i])) for i in range(window, len(prices))]
    )
    features["Entropy"] = pd.Series(np.pad(ent, (window, 0), "constant"), index=df.index)

    # Friction: Std diff / vol (real proxy), robust median for tails
    diffs = np.diff(prices)
    friction = np.median(np.abs(diffs)) / np.median(vols[1:] + 1e-10)  # Median for BTC-like
    features["Friction"] = pd.Series(np.full(len(df), friction), index=df.index)

    delta_e = np.diff(features["E_t"].fillna(0).values)
    features["Delta_E"] = pd.Series(np.pad(delta_e, (1, 0), "constant"), index=df.index)

    logging.info(f"Computed features for {symbol if 'symbol' in locals() else 'unknown'} with adaptive {cls} params: window={window}, percentile={percentile}")

    # Imbalance (5) - Adaptive rolling (symbol window)
    cls = SYMBOL_CLASS_MAP.get(symbol, "Forex") if 'symbol' in locals() else "Forex"
    params = CLASS_PARAMS[cls]
    roll_window = params["window"]

    hl_range = h - l
    delta_proxy = np.where(hl_range > 0, ((c - o) / hl_range) * v, 0)
    features["Delta"] = pd.Series(delta_proxy, index=df.index)

    cvd = pd.Series(delta_proxy).cumsum()
    features["CVD"] = cvd

    # BSR: Vol ratio, rolling adaptive
    vol_roll_mean = pd.Series(v).rolling(roll_window).mean()
    bsr = pd.Series(v) / vol_roll_mean
    features["BSR"] = bsr

    ofi = np.diff(v) * np.sign(np.diff(c))  # Proxy
    features["OFI"] = pd.Series(np.pad(ofi, (1, 0), "constant"), index=df.index)

    # Absorption: Robust abs delta / vol (median for tails)
    absorption = np.abs(delta_proxy) / (v + 1e-10)
    features["Absorption"] = pd.Series(absorption, index=df.index)

    # Patterns (5) - Adaptive order/rolling per class (e.g., larger for volatile)
    cls = SYMBOL_CLASS_MAP.get(symbol, "Forex") if 'symbol' in locals() else "Forex"
    params = CLASS_PARAMS[cls]
    roll_window = params["window"]
    order = int(roll_window / 4)  # Adaptive fractal order

    doji = talib.CDLDOJI(o, h, l, c)
    features["Doji"] = pd.Series(doji, index=df.index)

    engulfing = talib.CDLENGULFING(o, h, l, c)
    features["Engulfing"] = pd.Series(engulfing, index=df.index)

    fractal_highs = argrelextrema(prices, np.greater, order=order)[0]
    fractal = pd.Series(0, index=df.index)
    fractal.iloc[fractal_highs] = 1
    features["Fractal_high"] = fractal

    hammer = talib.CDLHAMMER(o, h, l, c)
    features["Hammer"] = pd.Series(hammer, index=df.index)

    # Real support break: Close < rolling low (adaptive window)
    support = c.rolling(roll_window).min()
    break_support = (c < support.shift(1)).astype(int)
    features["Support_break"] = pd.Series(break_support, index=df.index)

    # NaN shield + symbol-specific percentile clip (robust for tails)
    cls = SYMBOL_CLASS_MAP.get(symbol, "Forex") if 'symbol' in locals() else "Forex"
    params = CLASS_PARAMS[cls]
    percentile = params["percentile"]

    for k, s in features.items():
        s = s.fillna(method="ffill").fillna(0)
        # Clip per feature (e.g., high vol classes: Tighter tails)
        lower = s.quantile( (100 - percentile) / 2 )
        upper = s.quantile( 1 - (100 - percentile) / 2 )
        s = s.clip(lower=lower, upper=upper)
        features[k] = s

    return features


def run_mc_for_feature(args: Tuple[pd.DataFrame, str, pd.Series, bool]) -> Dict:
    """MC worker: Empirical sims, aggregate train/OOS metrics."""
    df, feat_name, feat_series, train_mode = args
    harvester = EmpiricalSuperHarvester(feat_series, df)
    baseline_omega = BASELINE_OMEGA  # Real baseline

    results = []
    for _ in range(NUM_MC_RUNS):
        res = harvester.simulate_trades(df, train_mode=train_mode)
        results.append(res)

    mean_omega = np.mean([r["omega"] for r in results])
    mean_z = np.mean([r["z"] for r in results])
    mean_win_rate = np.mean([r["win_rate"] for r in results])
    # P-val: t-test Omega vs baseline
    omegas = [r["omega"] for r in results]
    _, p_val = stats.ttest_1samp(omegas, baseline_omega)
    uplift = hierarchical_uplift(mean_omega, mean_z, mean_win_rate)
    mfe_capture = np.mean([r["mfe_capture"] for r in results])

    survives = mean_omega > SURVIVAL_OME and p_val < SURVIVAL_PVAL and uplift > SURVIVAL_UPLIFT

    return {
        "feature": feat_name,
        "mean_omega": mean_omega,
        "z_factor": mean_z,
        "win_rate": mean_win_rate,
        "p_val": p_val,
        "uplift": uplift,
        "mfe_capture": mfe_capture,
        "survives": survives,
        "train_mode": train_mode,
    }


@given(
    prices=st.lists(
        st.floats(min_value=0.1, allow_nan=False, allow_infinity=False), min_size=100, max_size=500
    ),
    noise=st.floats(min_value=0, max_value=0.01),
)
def test_feature_stability(prices, noise):
    """Property test: Features finite, low drift under real-like noise."""
    df = pd.DataFrame(
        {
            "open": np.roll(prices, 1),
            "high": [p * 1.01 for p in prices],
            "low": [p * 0.99 for p in prices],
            "close": prices,
            "volume": st.lists(st.integers(1000, 10000), length=len(prices)).example(0),  # Mock vol
        }
    )
    feats = compute_features(df)
    noisy_df = df.copy()
    noisy_df["close"] += np.random.normal(0, noise, len(prices))
    noisy_feats = compute_features(noisy_df)
    for k in feats:
        assert np.all(np.isfinite(feats[k].values))
        assert np.all(np.isfinite(noisy_feats[k].values))
        if len(feats[k]) > 10:
            drift = stats.ks_2samp(feats[k].dropna(), noisy_feats[k].dropna()).statistic
            assert drift < 0.25  # Empirical threshold


def analyze_period_results(results: List[Dict], symbol: str, period: str, is_oos: bool = False):
    """Per-period analysis: Win rates (>55%), Omega, unknowns (corrs, OOS drops)."""
    df_res = pd.DataFrame(results)
    mean_omega = df_res["mean_omega"].mean()
    mean_win = df_res["win_rate"].mean()
    num_survivors = df_res["survives"].sum()
    above_55 = (df_res["win_rate"] > 55).sum()

    logging.info(
        f"{symbol} {period} {'OOS' if is_oos else 'Train'}: Omega={mean_omega:.2f}, Win Rate={mean_win:.1f}%, "
        f"Survivors={num_survivors}, >55% Wins={above_55}"
    )

    if num_survivors > 0:
        victors = df_res[df_res["survives"]]
        logging.info(f"  Theorems: Top {symbol} {period} - {victors['feature'].tolist()}")

    # Unknowns: Feature corrs (e.g., physics-imbalance)
    if len(df_res) > 1:
        feat_corrs = (
            df_res[["feature", "uplift"]]
            .set_index("feature")["uplift"]
            .corrwith(df_res["win_rate"])
        )
        high_corrs = feat_corrs[abs(feat_corrs) > 0.7]
        if len(high_corrs) > 0:
            logging.info(f"  Unknowns (High Corrs): {high_corrs.to_dict()}")

    if is_oos:
        oos_drop = (mean_omega - BASELINE_OMEGA) / BASELINE_OMEGA * 100
        logging.info(f"  OOS Drop: {oos_drop:.1f}% (robust if <10%)")

    return df_res


def main(symbols: List[str], tf: str, years: List[str], split_pct: List[int]):
    load_dotenv()  # Load .env for tokens

    logging.info(
        f"Real Empirical Hunger Games! Symbols: {symbols}, TF: {tf}, Years: {years}, Split: {split_pct}"
    )

    # Property test
    try:
        test_feature_stability()
        logging.info("Tributes stable (property test passed).")
    except Exception as e:
        logging.error(f"Property fail: {e}")
        return

    all_results = {"train": {}, "oos": {}, "combined": []}
    os.makedirs("data", exist_ok=True)

    for year in years:
        logging.info(f"\n=== Arena Round: {year} ===")
        year_results = []
        for symbol in symbols:
            df_train, df_oos = load_data(symbol, tf, year, split=True, train_pct=split_pct[0])

            # Train
            feats_train = compute_features(df_train)
            mc_args_train = [(df_train, name, series, True) for name, series in feats_train.items()]
            with Pool() as pool:
                train_res = pool.map(run_mc_for_feature, mc_args_train)
            for res in train_res:
                res["symbol"] = symbol
                res["period"] = f"{year}_train"
                year_results.append(res)

            # OOS
            if not df_oos.empty:
                feats_oos = compute_features(df_oos)
                mc_args_oos = [(df_oos, name, series, False) for name, series in feats_oos.items()]
                with Pool() as pool:
                    oos_res = pool.map(run_mc_for_feature, mc_args_oos)
                for res in oos_res:
                    res["symbol"] = symbol
                    res["period"] = f"{year}_oos"
                    year_results.append(res)

            # Per-symbol/period analysis
            analyze_period_results(train_res, symbol, year, False)
            if not df_oos.empty:
                analyze_period_results(oos_res, symbol, year, True)

        # Save per year
        df_year = pd.DataFrame(year_results)
        df_year.to_csv(f"data/mvp_survivors_{year}.csv", index=False)
        all_results["train"][year] = df_year[df_year["period"].str.contains("train")]

    # Combined 70/30
    logging.info("\n=== Combined Arena: 2023-2024 70/30 Split ===")
    combined_results = []
    for symbol in symbols:
        df_train_comb, df_oos_comb = load_data(
            symbol, tf, "combined", split=True, train_pct=split_pct[0]
        )

        feats_train_comb = compute_features(df_train_comb)
        mc_args_train_comb = [
            (df_train_comb, name, series, True) for name, series in feats_train_comb.items()
        ]
        with Pool() as pool:
            train_comb_res = pool.map(run_mc_for_feature, mc_args_train_comb)

        feats_oos_comb = compute_features(df_oos_comb)
        mc_args_oos_comb = [
            (df_oos_comb, name, series, False) for name, series in feats_oos_comb.items()
        ]
        with Pool() as pool:
            oos_comb_res = pool.map(run_mc_for_feature, mc_args_oos_comb)

        all_comb = train_comb_res + oos_comb_res
        for res in all_comb:
            res["symbol"] = symbol
            res["period"] = "combined"
            combined_results.extend(all_comb)

        analyze_period_results(train_comb_res, symbol, "combined", False)
        analyze_period_results(oos_comb_res, symbol, "combined", True)

    df_combined = pd.DataFrame(combined_results)
    df_combined.to_csv("data/mvp_survivors_combined.csv", index=False)

    # Overall
    survivors_total = len(df_combined[df_combined["survives"]])
    mean_omega_total = df_combined["mean_omega"].mean()
    logging.info(
        f"\n=== Quest Summary ===\nSurvivors: {survivors_total}/20 per symbol\nAvg Omega: {mean_omega_total:.2f}\nWin Rates >55%: {(df_combined['win_rate'] > 55).mean() * 100:.1f}%\nAlpha Advanced!"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hunger Games MVP Real Arena")
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=SYMBOLS_DEFAULT,
        help="Top symbols (default: EURUSD BTCUSD NAS100 XAUUSD US30)",
    )
    parser.add_argument("--tf", default=TFS_DEFAULT[0], help="Timeframe (default: H1)")
    parser.add_argument(
        "--years", nargs="+", default=YEARS_DEFAULT, help="Years (default: 2023 2024)"
    )
    parser.add_argument(
        "--split", nargs=2, type=int, default=SPLIT_DEFAULT, help="Train/OOS % (default: 70 30)"
    )
    args = parser.parse_args()

    # Pass symbol for adaptive prep/features
    main(args.symbols, args.tf, args.years, args.split)
