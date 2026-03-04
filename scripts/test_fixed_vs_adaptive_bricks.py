"""
Empirical test: Fixed vs Adaptive brick sizes over a 3-month window.

Three scenarios:
  1. Fixed 1.0       — hardcoded brick_size=1.0
  2. Fixed DSP       — DSP run once on full window, brick_size held constant
  3. Monthly adaptive — DSP recalibrated every 30 days on preceding 60 days

Uses existing XAUUSD M1 data. Prints a comparison table.
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import json

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from kinetra.renko.backtest import (
    FilterParams,
    InstrumentBacktestResult,
    LatencyParams,
    SizingMode,
    StopParams,
    VolSizingParams,
    backtest_instrument,
)

LATENCY = LatencyParams(
    entry_latency_ms=250.0,
    exit_latency_ms=250.0,
    entry_jitter_ms=10.0,
    exit_jitter_ms=10.0,
)

SIZING = VolSizingParams(
    initial_equity=1_000.0,
    fixed_lot=0.01,
    lot_ceiling=50.0,
    compounding_capital_per_lot=1_000.0,
)

# ── Data paths ────────────────────────────────────────────────────────────────

_SYM_DIR = ROOT / "data/master_standardized/ctrader/pepperstone/metals/XAUUSD"

M1_PATH = _SYM_DIR / "XAUUSD_M1_202312212301_202603022131.csv"
TICK_DIR = _SYM_DIR / "ticks"

SYMBOL = "XAUUSD"
MONTHS = 3  # test window
CALIB_DAYS = 60  # adaptive calibration lookback

SPEC_PATH = _SYM_DIR / "contract_spec.json"


def friction_floor(y_mult: float = 5.0, friction_mult: float = 4.0) -> float:
    """Compute minimum viable brick size from contract spec friction."""
    spec = json.loads(SPEC_PATH.read_text())
    tick_size = float(spec.get("tick_size", 0.01))
    spread_pts = float(spec.get("spread_typical_pts", 22.0))
    commission_rt = float(spec.get("commission_per_lot", 3.5)) * 2  # round-trip
    contract_size = float(spec.get("contract_size", 100.0))

    spread_price = spread_pts * tick_size          # $0.22
    comm_per_unit = commission_rt / contract_size  # $0.07
    friction_per_unit = comm_per_unit + spread_price

    floor = max(y_mult * spread_price, friction_mult * friction_per_unit)
    return round(floor, 2)


def _find_tick_csv() -> Path | None:
    """Return the most recent tick CSV in TICK_DIR, or None."""
    if not TICK_DIR.exists():
        return None
    files = sorted(TICK_DIR.glob("XAUUSD_ticks_*.csv"))
    return files[-1] if files else None


def load_prices(months: int) -> tuple[pd.Series, str]:
    """Load price series for backtesting.

    Prefers tick bid prices if available; falls back to M1 closes.
    Returns (prices, source_label).
    """
    tick_csv = _find_tick_csv()
    if tick_csv is not None:
        df = pd.read_csv(tick_csv)
        df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
        df = df.dropna(subset=["time", "bid"]).set_index("time").sort_index()
        cutoff = df.index.max() - pd.DateOffset(months=months)
        prices = df.loc[cutoff:, "bid"].dropna()
        label = f"TICK bid  ({tick_csv.name})"
        print(
            f"Loaded {len(prices):,} ticks  "
            f"({prices.index[0].date()} → {prices.index[-1].date()})  [tick mode]"
        )
        return prices, label

    # Fallback: M1 closes
    df = pd.read_csv(M1_PATH, parse_dates=["time"])
    df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
    df = df.dropna(subset=["time", "close"]).set_index("time").sort_index()
    cutoff = df.index.max() - pd.DateOffset(months=months)
    prices = df.loc[cutoff:, "close"].dropna()
    print(
        f"Loaded {len(prices):,} M1 bars  "
        f"({prices.index[0].date()} → {prices.index[-1].date()})  [M1 fallback — download ticks for accuracy]"
    )
    return prices, "M1 close"


# Keep old name for adaptive function compatibility
def load_closes(months: int) -> pd.Series:
    prices, _ = load_prices(months)
    return prices


# ── ATR brick size (replaces VR/DSP) ─────────────────────────────────────────


def atr_brick_size(closes: pd.Series, period: int = 14, divisor: float = 10.0) -> float:
    """ATR(period, daily) / divisor as a simple brick size estimate."""
    daily = closes.resample("1D").last().dropna()
    tr = daily.diff().abs().dropna()
    atr = tr.rolling(period).mean().iloc[-1]
    return round(float(atr) / divisor, 2)


# ── Backtest helper ───────────────────────────────────────────────────────────


def time_scaled_filters(brick_size: float, closes: pd.Series) -> FilterParams:
    """Scale filter windows to ~1 trading day of bricks (consistent time horizon)."""
    from kinetra.renko.brick_engine import build_renko, bricks_per_day as _bpd
    bricks = build_renko(closes, brick_size)
    bpd = _bpd(bricks)
    # Window = 1 day of bricks, clamped to [10, 200]
    window = int(max(10, min(200, round(bpd))))
    return FilterParams(
        fliprate_window=window,
        fliprate_threshold=0.35,
        markov_window=window,
        markov_threshold=0.55,
    )


def run_bt(closes: pd.Series, brick_size: float) -> InstrumentBacktestResult:
    fp = time_scaled_filters(brick_size, closes)
    return backtest_instrument(
        SYMBOL,
        closes,
        brick_size=brick_size,
        filter_params=fp,
        stop_params=StopParams(),
        latency_params=LATENCY,
        sizing_mode=SizingMode.COMPOUNDING,
        vol_sizing_params=SIZING,
    )


# ── Adaptive: stitch monthly segments ────────────────────────────────────────


def run_adaptive(closes: pd.Series, calib_days: int = 60) -> dict:
    """
    Recalibrate brick_size every 30 days using the preceding calib_days.
    Returns aggregate metrics stitched from per-segment backtests.
    """
    start = closes.index[0]
    end = closes.index[-1]
    segment_start = start + pd.Timedelta(days=calib_days)  # first usable date

    all_trades: list = []
    equity = 0.0
    equity_curve: list = []
    brick_sizes_used: list = []

    step = pd.Timedelta(days=30)
    t = segment_start
    while t < end:
        seg_end = min(t + step, end)

        # calibration window: calib_days before segment start
        calib_closes = closes.loc[t - pd.Timedelta(days=calib_days) : t]
        if len(calib_closes) < 500:
            t += step
            continue

        b_size = dsp_brick_size(calib_closes)
        brick_sizes_used.append(b_size)

        seg_closes = closes.loc[t:seg_end]
        if len(seg_closes) < 200:
            t += step
            continue

        res = run_bt(seg_closes, b_size)
        all_trades.extend(res.trades)
        for e in res.equity_curve:
            equity_curve.append(equity + e)
        if res.equity_curve:
            equity += res.equity_curve[-1]

        t += step

    if not all_trades:
        return {"trades": 0, "omega": 0.0, "win_rate": 0.0,
                "net_pnl": 0.0, "max_dd": 0.0,
                "brick_sizes": brick_sizes_used}

    net_pnls = np.array([t.net_usd for t in all_trades])
    wins = (net_pnls > 0).sum()
    win_rate = wins / len(net_pnls)

    # omega: sum(gains) / sum(losses)
    gains = net_pnls[net_pnls > 0].sum()
    losses = abs(net_pnls[net_pnls < 0].sum())
    omega = gains / losses if losses > 0 else float("inf")

    # max drawdown from equity curve
    if equity_curve:
        arr = np.array(equity_curve)
        peak = np.maximum.accumulate(arr)
        dd = (arr - peak).min()
    else:
        dd = 0.0

    return {
        "trades": len(all_trades),
        "omega": round(omega, 3),
        "win_rate": round(win_rate, 4),
        "net_pnl": round(float(net_pnls.sum()), 2),
        "max_dd": round(float(dd), 2),
        "brick_sizes": brick_sizes_used,
    }


# ── Main ──────────────────────────────────────────────────────────────────────


def _fmt(r: InstrumentBacktestResult) -> dict:
    return {
        "trades": len(r.trades),
        "omega": round(r.omega, 3),
        "win_rate": round(r.win_rate, 4),
        "net_pnl": round(sum(t.net_usd for t in r.trades), 2),
        "max_dd": round(r.max_dd_usd, 2),
        "brick_sizes": None,
    }


def main():
    prices, data_source = load_prices(MONTHS)
    trading_days = MONTHS * 21

    floor = friction_floor()
    atr_b = atr_brick_size(prices)
    print(f"\nData source    : {data_source}")
    print(f"Friction floor : ${floor}")
    print(f"ATR/10 estimate: ${atr_b}")

    # Sweep from floor × 1 up to floor × 20, ~12 steps
    multipliers = [1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 7.0, 10.0, 13.0, 17.0, 20.0]
    sweep_sizes = sorted(set([round(floor * m, 2) for m in multipliers]))

    print(f"\n══ BRICK SIZE SWEEP — floor=${floor} ({MONTHS}mo XAUUSD) ═══════════════")
    print(f"  {'Size':>6} {'×floor':>6} {'Win':>7} {'Trades':>7} {'Omega':>7} {'Win%':>7} "
          f"{'Net$':>10} {'MaxDD$':>10} {'Tr/day':>7}")
    print(f"  {'─'*6} {'─'*6} {'─'*7} {'─'*7} {'─'*7} {'─'*7} {'─'*10} {'─'*10} {'─'*7}")

    best_omega, best_size = 0.0, sweep_sizes[0]
    for b in sweep_sizes:
        fp = time_scaled_filters(b, prices)
        r = run_bt(prices, brick_size=b)
        s = _fmt(r)
        tpd = s['trades'] / trading_days
        mult = b / floor
        marker = " ◄ ATR" if abs(b - atr_b) < floor * 0.5 else ""
        if s['omega'] > best_omega and s['trades'] >= 30:
            best_omega = s['omega']
            best_size = b
        print(f"  {b:>6.2f} {mult:>5.1f}× {fp.markov_window:>7} {s['trades']:>7} {s['omega']:>7.3f} "
              f"{s['win_rate']:>6.1%} {s['net_pnl']:>10,.0f} "
              f"{s['max_dd']:>10,.0f} {tpd:>7.1f}{marker}")

    print(f"\n  Best omega: brick_size={best_size} (omega={best_omega:.3f})")
    print(f"  = {best_size/floor:.1f}× friction floor")


if __name__ == "__main__":
    main()
