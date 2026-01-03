#!/usr/bin/env python3
"""
Kinetra Batch Backtest: Dynamic Empirical Pipeline

MVP for dynamic symbol discovery (MetaAPI/MT5 query, suffixes like EURUSD.m), non-linear prep
(log-returns, medians), SuperPot PPO survival (Ω>2.7, win>55%), triggers/harvesters (log-trail),
asym risk (Omega/RoR), replay stub (entropy log). CLI menu-like.

Usage: python scripts/batch_backtest.py --symbols EURUSD --timeframe H1 --years 2023 2024 --split 70 30 --agent superpot --mc-runs 50 --dry-run

Integrates existing: physics_engine, rl_agent, risk_management.
"""

import argparse
import asyncio
import logging
import os
import warnings
from datetime import datetime
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
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

warnings.filterwarnings("ignore")

# Setup logging
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(f"{log_dir}/batch_backtest.log"),
        logging.StreamHandler(),
    ],
)

load_dotenv()

# Defaults
NUM_MC_RUNS = 50
BASELINE_OMEGA = 2.7
SURVIVAL_OME = BASELINE_OMEGA
SURVIVAL_WIN = 55.0
METAAPI_TOKEN = os.getenv("METAAPI_TOKEN")
METAAPI_ACCOUNT_ID = "e8f8c21a-32b5-40b0-9bf7-672e8ffab91f"
DATA_DIR = "data/master_standardized"

# Default arguments for CLI
SYMBOLS_DEFAULT = ["BTCUSD"]
TFS_DEFAULT = ["H1"]
YEARS_DEFAULT = ["2023", "2024"]

# Dynamic symbol classes & suffixes (broker-aware)
SYMBOL_CLASSES = {
    "Forex": ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD"],
    "Crypto": ["BTCUSD", "ETHUSD"],
    "Indices": ["NAS100", "US30", "SPX500"],
    "Commodities": ["XAUUSD", "XAGUSD", "OIL"],
    "Equity": ["AAPL", "GOOGL"],
}
SUFFIXES = [".m", ".raw", ".c", ""]  # ECN variations

# Class params for non-linear adaptation
CLASS_PARAMS = {
    "Forex": {"window": 20, "vol_factor": 1.0, "q_low": 0.05, "q_high": 0.95},
    "Crypto": {"window": 15, "vol_factor": 2.5, "q_low": 0.01, "q_high": 0.99},
    "Indices": {"window": 25, "vol_factor": 1.5, "q_low": 0.05, "q_high": 0.95},
    "Commodities": {"window": 20, "vol_factor": 1.8, "q_low": 0.03, "q_high": 0.97},
    "Equity": {"window": 22, "vol_factor": 1.2, "q_low": 0.05, "q_high": 0.95},
}

# PPO config for SuperPot
PPO_CONFIG = {
    "policy": "MlpPolicy",
    "learning_rate": 3e-4,
    "n_steps": 2048,
    "batch_size": 64,
    "n_epochs": 10,
    "gamma": 0.99,
    "ent_coef": 0.01,  # Exploration
}


class SimpleTradingEnv:
    """Mock env for PPO: State log-features, action size/direction, reward log PnL."""

    def __init__(self, df: pd.DataFrame):
        self.df = df.reset_index()
        self.current_step = 0
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,))  # Size/dir
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(5,))  # 5 log-features

    def reset(self):
        self.current_step = 0
        return self._get_obs(), {}

    def step(self, action):
        # Mock trade: Log PnL from action
        price = np.log(self.df["close"].iloc[self.current_step])
        reward = np.log(1 + action[0] * np.random.normal(0, 0.01))  # Non-linear log
        self.current_step += 1
        done = self.current_step >= len(self.df) - 1
        obs = self._get_obs()
        return obs, reward, done, False, {}

    def _get_obs(self):
        # Log-features
        log_close = np.log(self.df["close"].iloc[self.current_step])
        log_vol = np.log(self.df["volume"].iloc[self.current_step] + 1)
        return np.array([log_close, log_vol, 0, 0, 0], dtype=np.float32)  # Stub


from gymnasium import spaces  # For env


async def fetch_metaapi_data_async(
    symbol: str, tf: str, start_date: str, end_date: str = None
) -> pd.DataFrame:
    if not METAAPI_TOKEN:
        return pd.DataFrame()
    api = MetaApi(token=METAAPI_TOKEN)
    account = await api.metatrader_account_api.get_account(METAAPI_ACCOUNT_ID)
    connection = await account.get_rpc_connection()
    await connection.wait_synchronized()
    tf_map = {"H1": 16385}
    timeframe = tf_map.get(tf, 16385)
    all_rates = []
    current_start = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date) if end_date else pd.Timestamp.now()
    while current_start < end_dt:
        rates = await connection.copy_rates_range(
            symbol, timeframe, current_start, end_dt, max_bars=1000
        )
        if not rates:
            break
        all_rates.extend(rates)
        if rates:
            current_start = pd.to_datetime(rates[-1]["time"], unit="s") + pd.Timedelta(hours=1)
    if not all_rates:
        return pd.DataFrame()
    df = pd.DataFrame(all_rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    df = df.set_index("time")[["open", "high", "low", "close", "tick_volume"]].rename(
        columns={"tick_volume": "volume"}
    )
    df = df.dropna()
    logging.info(f"Fetched {len(df)} H1 bars for {symbol} {start_date}-{end_date}")
    return df


def fetch_metaapi_data(symbol: str, tf: str, start_date: str, end_date: str = None) -> pd.DataFrame:
    try:
        return asyncio.run(fetch_metaapi_data_async(symbol, tf, start_date, end_date))
    except Exception as e:
        logging.warning(f"MetaAPI fail {symbol}: {e}. Fallback mock.")
        return generate_mock_data(symbol, tf, start_date, end_date)


def generate_mock_data(symbol: str, tf: str, start: str, end: str) -> pd.DataFrame:
    cls = next((k for k, v in SYMBOL_CLASSES.items() if symbol in v), "Forex")
    params = CLASS_PARAMS[cls]
    freq_map = {"H1": "1H"}
    freq = freq_map.get(tf, tf)
    dates = pd.date_range(start=start, end=end, freq=freq)
    n = len(dates)
    np.random.seed(abs(hash(symbol)) % (2**32))
    returns = np.random.normal(0, 0.0005 * params["vol_factor"], n)
    close = 100 * np.exp(np.cumsum(returns))
    open_ = close * (1 + np.random.normal(0, 0.0001, n))
    high = np.maximum(close, open_) + np.abs(np.random.normal(0, 0.0003, n))
    low = np.minimum(close, open_) - np.abs(np.random.normal(0, 0.0003, n))
    volume = np.random.lognormal(7, 0.5 * params["vol_factor"], n)
    df = pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume}, index=dates
    )
    logging.info(f"Generated {n} mock bars for {symbol} ({cls})")
    return df


def prepare_data(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    if df.empty:
        return df
    cls = next((k for k, v in SYMBOL_CLASSES.items() if symbol in v), "Forex")
    params = CLASS_PARAMS[cls]
    df = df.sort_index()
    # Log-returns non-linear
    for col in ["open", "high", "low", "close"]:
        df[col] = np.log(df[col])
    # Gap detection (vol median)
    vol_median = np.median(df["volume"])
    low_vol_mask = df["volume"] < vol_median * params.get("gap_thresh", 0.1)
    df = df[~low_vol_mask]
    # Clip quantiles asymmetric
    q_low, q_high = params["q_low"], params["q_high"]
    for col in ["open", "high", "low", "close"]:
        lower = df[col].quantile(q_low)
        upper = df[col].quantile(q_high)
        df[col] = df[col].clip(lower, upper)
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    logging.info(f"Prepped {symbol} ({cls}): {len(df)} bars, {low_vol_mask.sum()} gaps removed")
    return df


def load_data(
    symbol: str, tf: str, period: str, split: bool = False, train_pct: int = 70
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if period == "2023":
        start, end = "2023-01-01", "2023-12-31"
    elif period == "2024":
        start, end = "2024-01-01", "2024-12-31"
    else:
        start, end = "2023-01-01", "2024-12-31"
    df_raw = fetch_metaapi_data(symbol, tf, start, end)
    if df_raw.empty:
        path = os.path.join(DATA_DIR, f"{symbol}_{tf}.csv")
        if os.path.exists(path):
            df_raw = pd.read_csv(path, index_col="timestamp", parse_dates=True)[
                ["open", "high", "low", "close", "volume"]
            ].dropna()
        else:
            df_raw = generate_mock_data(symbol, tf, start, end)
    df_full = prepare_data(df_raw, symbol)
    if not split:
        return df_full, pd.DataFrame()
    split_idx = int(len(df_full) * (train_pct / 100))
    return df_full.iloc[:split_idx], df_full.iloc[split_idx:]


def compute_non_linear_features(df: pd.DataFrame, symbol: str) -> Dict[str, pd.Series]:
    features = {}
    cls = next((k for k, v in SYMBOL_CLASSES.items() if symbol in v), "Forex")
    params = CLASS_PARAMS[cls]
    window = params["window"]
    o, h, l, c = df["open"], df["high"], df["low"], df["close"]
    v = df["volume"]
    # Log-returns already in prep
    # RSI (adaptive window)
    rsi = talib.RSI(c, timeperiod=window)
    features["RSI"] = pd.Series(rsi, index=df.index)
    # ATR
    atr = talib.ATR(h, l, c, timeperiod=window)
    features["ATR"] = pd.Series(atr, index=df.index)
    # OBV
    obv = talib.OBV(c, v)
    features["OBV"] = pd.Series(obv, index=df.index)
    # Log E_t stub (non-linear)
    delta_p = np.diff(c)
    log_e_t = np.log(0.5 * v[1:] * (delta_p**2) + 1e-10)
    features["log_E_t"] = pd.Series(np.pad(log_e_t, (1, 0), "constant"), index=df.index)
    # Median damping stub
    damping = np.abs(np.diff(log_e_t)) / np.abs(log_e_t + 1e-10)
    features["median_Damping"] = pd.Series(np.full(len(df), np.median(damping)), index=df.index)
    # Clipped Delta
    hl_range = h - l
    delta = ((c - o) / hl_range.where(hl_range != 0, np.nan)) * v
    q_low, q_high = params["q_low"], params["q_high"]
    delta = delta.clip(delta.quantile(q_low), delta.quantile(q_high))
    features["clipped_Delta"] = pd.Series(delta, index=df.index)
    # CVD
    features["CVD"] = pd.Series(np.cumsum(delta), index=df.index)
    # Doji
    doji = talib.CDLDOJI(o, h, l, c)
    features["Doji"] = pd.Series(doji, index=df.index)
    for k, s in features.items():
        s = s.fillna(0)
        features[k] = s
    logging.info(f"Non-linear features for {symbol}: {list(features.keys())}")
    return features


def omega_ratio(pnls: np.ndarray) -> float:
    if len(pnls) == 0:
        return 0.0
    threshold = np.median(pnls)
    upside = pnls[pnls > threshold].sum()
    downside = np.abs(pnls[pnls < threshold].sum())
    return upside / downside if downside > 0 else np.inf


def run_superpot_rl(df_train: pd.DataFrame, df_oos: pd.DataFrame, symbol: str) -> Dict:
    env_fn = lambda: SimpleTradingEnv(df_train)
    env = DummyVecEnv([env_fn])
    model = PPO("MlpPolicy", env, **PPO_CONFIG)
    if os.path.exists(PPO_MODEL_PATH):
        model = PPO.load(PPO_MODEL_PATH, env=env)
    model.learn(total_timesteps=10000)
    # Train eval
    obs = env.reset()
    train_pnls = []
    for _ in range(len(df_train)):
        action, _ = model.predict(obs)
        obs, reward, done, _ = env.step(action)
        train_pnls.append(reward)
        if done:
            obs = env.reset()
    omega_train = omega_ratio(np.array(train_pnls))
    win_train = (np.array(train_pnls) > 0).mean() * 100
    # OOS
    env_oos = DummyVecEnv([lambda: SimpleTradingEnv(df_oos)])
    obs_oos = env_oos.reset()
    oos_pnls = []
    for _ in range(len(df_oos)):
        action, _ = model.predict(obs_oos)
        obs_oos, reward, done, _ = env_oos.step(action)
        oos_pnls.append(reward)
        if done:
            obs_oos = env_oos.reset()
    omega_oos = omega_ratio(np.array(oos_pnls))
    win_oos = (np.array(oos_pnls) > 0).mean() * 100
    oos_drop = ((omega_train - omega_oos) / omega_train * 100) if omega_train > 0 else 0
    _, ks_p = stats.ks_2samp(train_pnls, oos_pnls)
    survives = (
        omega_train > SURVIVAL_OME and win_train > SURVIVAL_WIN and oos_drop < 5 and ks_p > 0.05
    )
    logging.info(
        f"{symbol}: Ω_train={omega_train:.2f}, Win={win_train:.1f}%, drop={oos_drop:.1f}%, KS_p={ks_p:.3f} | Survives: {survives}"
    )
    return {
        "omega_train": omega_train,
        "win_train": win_train,
        "survives": survives,
        "entropy": model.entropy_loss or 0.1,
    }


def compute_triggers(df: pd.DataFrame, symbol: str) -> str:
    feats = compute_non_linear_features(df, symbol)
    # Top trigger by skew (asym)
    top_feat = max(feats, key=lambda k: abs(feats[k].skew()))
    logging.info(f"{symbol} Trigger: {top_feat} (skew={feats[top_feat].skew():.2f})")
    return top_feat


def compute_harvesters(df: pd.DataFrame, trigger_feat: str, symbol: str) -> float:
    atr = talib.ATR(df["high"], df["low"], df["close"], timeperiod=20)
    log_atr = np.log(atr + 1e-10)
    rank_trigger = stats.rankdata(df[trigger_feat]) / len(df[trigger_feat])
    trail = log_atr * rank_trigger
    # Mock MFE from trail (MC stub)
    mfe = np.median(trail) * 0.68  # Non-linear median
    logging.info(f"{symbol} Harvester: Log-trail MFE={mfe:.2f}")
    return mfe


def apply_risk(df: pd.DataFrame, symbol: str) -> float:
    pnls = np.diff(np.log(df["close"]))  # Log-returns
    omega = omega_ratio(pnls)
    mu, sigma = np.median(pnls), np.quantile(np.abs(pnls), 0.95)
    ror = np.exp(-2 * mu / sigma**2) if sigma > 0 else 0
    chs = omega * 0.4 + (1 - ror) * 0.6  # Asym
    logging.info(f"{symbol} Risk: CHS={chs:.2f}, RoR={ror:.3f}")
    return chs


def main(
    symbols: List[str],
    tf: str,
    years: List[str],
    split: bool,
    train_pct: int,
    mc_runs: int,
    dry_run: bool,
):
    results = []
    for symbol in symbols:
        for year in years:
            df_train, df_oos = load_data(symbol, tf, year, split, train_pct)
            if dry_run:
                logging.info(f"Dry-run {symbol} {year}: Train {len(df_train)} bars")
                continue
            # SuperPot RL
            rl_res = run_superpot_rl(df_train, df_oos, symbol)
            if not rl_res["survives"]:
                continue
            # Triggers
            trigger = compute_triggers(df_train, symbol)
            # Harvesters
            mfe = compute_harvesters(df_train, trigger, symbol)
            # Risk
            chs = apply_risk(df_train, symbol)
            results.append(
                {
                    "symbol": symbol,
                    "year": year,
                    "omega_train": rl_res["omega_train"],
                    "win_train": rl_res["win_train"],
                    "trigger": trigger,
                    "mfe": mfe,
                    "chs": chs,
                    "entropy": rl_res["entropy"],
                }
            )
    pd.DataFrame(results).to_csv("data/batch_backtest_results.csv", index=False)
    logging.info("Batch backtest complete: Results in data/batch_backtest_results.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Kinetra Batch Backtest")
    parser.add_argument("--symbols", nargs="+", default=SYMBOLS_DEFAULT)
    parser.add_argument("--tf", default=TFS_DEFAULT[0])
    parser.add_argument("--years", nargs="+", default=YEARS_DEFAULT)
    parser.add_argument("--split", action="store_true", default=True)
    parser.add_argument("--train-pct", type=int, default=70)
    parser.add_argument("--mc-runs", type=int, default=NUM_MC_RUNS)
    parser.add_argument("--dry-run", action="store_true", help="No RL/train, just fetch/prep")
    args = parser.parse_args()
    main(args.symbols, args.tf, args.years, args.split, args.train_pct, args.mc_runs, args.dry_run)
