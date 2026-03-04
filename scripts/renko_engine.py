#!/usr/bin/env python3
"""
Kinetra Renko Brick Trade Engine - Unified Runner
==================================================

CANONICAL REFERENCE: XAUUSD
  - All engine changes validated on XAUUSD first
  - XAUUSD is the empirical test bed
  - Other symbols only added after XAUUSD validation passes

SINGLE ENGINE, SEQUENTIAL VALIDATION:
  1. Download historical M1 data
  2. DSP analysis → find brick size > friction costs
  3. Backtest last 3 months (quick validation)
  4. If good → backtest 3 years with rolling OOS
  5. If pass → paper trading (simulated)
  6. If pass → micro lots live
  7. If pass → scaled lots live

NO NEW CODE AT LATER STAGES - same engine throughout.
All code changes go through full validation chain on XAUUSD first.

Usage:
    python scripts/renko_engine.py XAUUSD --stage all
    python scripts/renko_engine.py XAUUSD --stage backtest --months 3
    python scripts/renko_engine.py XAUUSD --stage paper
    python scripts/renko_engine.py XAUUSD --stage live --size micro
"""

import json
import logging
import os
import re
import subprocess
import sys
import threading
import time
from collections import deque
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import click
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

if TYPE_CHECKING:
    from kinetra.renko.trading_engine import RenkoEngine

from rich import box
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table

from kinetra.preflight_enhanced import (
    EnhancedPreflight,
    PreflightConfig,
    format_preflight_report,
)
from kinetra.renko.brick_engine import bricks_per_day, build_renko
from kinetra.renko.trading_engine import EngineConfig, RenkoEngine

KR = PROJECT_ROOT
console = Console()
LOG = logging.getLogger("kinetra.renko")
_STANDARDIZED_SYMBOLS: set[str] = set()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    handlers=[
        logging.FileHandler(
            KR
            / "outputs"
            / "logs"
            / f"renko_engine_{datetime.now(timezone.utc).strftime('%Y%m%d')}.log"
        ),
        logging.StreamHandler(),
    ],
)


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────


def load_m1_data(symbol: str) -> Optional[pd.Series]:
    """Load M1 data from canonical location."""
    from kinetra.data_utils import load_mt5_csv

    _standardize_symbol_m1_data(symbol, max_depth=6)
    data_dir = _canonical_symbol_dir(symbol)
    best_file = _select_best_m1_file(data_dir, symbol)

    if best_file is None:
        return None

    df = load_mt5_csv(str(best_file))

    # load_mt5_csv returns Title case columns: Close, and sets datetime as index
    close_col = "Close" if "Close" in df.columns else "close"

    # Index should already be DatetimeIndex from load_mt5_csv
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError(f"Index is not DatetimeIndex: {type(df.index)}")

    # Create Series with proper DatetimeIndex (ensure UTC)
    series = pd.Series(df[close_col].values, index=df.index)

    # Ensure UTC timezone
    if series.index.tz is None:
        series.index = series.index.tz_localize("UTC")
    else:
        series.index = series.index.tz_convert("UTC")

    # Sort by index and remove duplicates
    series = series.sort_index()
    series = series[~series.index.duplicated(keep="first")]

    return series


def load_m1_market_data(symbol: str) -> Optional[pd.DataFrame]:
    """Load canonical M1 close+spread data indexed by UTC timestamp."""
    _standardize_symbol_m1_data(symbol, max_depth=6)
    data_dir = _canonical_symbol_dir(symbol)
    best_file = _select_best_m1_file(data_dir, symbol)
    if best_file is None:
        return None
    df = _load_m1_csv_generic(best_file)
    if df is None or df.empty:
        return None
    df = df[["time", "close", "spread"]].copy()
    df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df["spread"] = pd.to_numeric(df["spread"], errors="coerce")
    df = df.dropna(subset=["time", "close"])
    df = df.sort_values("time").drop_duplicates(subset=["time"], keep="last").set_index("time")
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")
    return df


def get_data_path(symbol: str) -> Path:
    """Get canonical data directory for symbol."""
    return _canonical_symbol_dir(symbol)


def _format_number(value: float) -> str:
    """Format large numbers with K/M suffixes for compact display."""
    abs_val = abs(value)
    if abs_val >= 1_000_000:
        return f"{value / 1_000_000:.2f}M"
    elif abs_val >= 1_000:
        return f"{value / 1_000:.2f}K"
    else:
        return f"{value:.2f}"


def _adaptive_panel_width(min_width: int = 88, max_width: int = 160, pad: int = 2) -> int:
    """Choose a panel width based on current terminal width."""
    try:
        usable = max(int(console.size.width) - int(pad), 40)
    except Exception:
        usable = 120
    if usable < min_width:
        return usable
    return min(usable, max_width)


def _run_preflight_checks(
    connector,
    symbol: str,
    min_balance_usd: float = 100.0,
    require_hot_standby: bool = False,
) -> bool:
    """Run comprehensive live-trading safety checks before enabling REAL orders.

    Enhanced preflight includes:
    - DNS resolution validation
    - TCP connection pool health
    - Heartbeat/keep-alive verification
    - Broker session validation
    - Account balance check
    - Margin requirements
    - Symbol resolution
    - Market hours validation
    - Hot standby verification (if required)
    - Connection health service check
    """
    config = PreflightConfig(
        symbol=symbol,
        min_balance_usd=min_balance_usd,
        require_hot_standby=require_hot_standby,
        enable_health_service=True,
    )

    preflight = EnhancedPreflight(connector, config)

    click.echo()
    click.secho("Running enhanced preflight checks...", fg="cyan", bold=True)
    click.echo()

    def _progress(msg: str):
        click.echo(f"  {msg}")

    result = preflight.run_all_checks(progress_callback=_progress)

    click.echo()
    click.echo(format_preflight_report(result, verbose=False))

    # Print summary account info
    try:
        snapshot = connector.get_account_snapshot(timeout_s=5.0)
        click.echo(
            f"\n  Account: {snapshot.get('account_id')}  "
            f"Broker: {snapshot.get('broker_name', 'unknown')}  "
            f"Balance: ${float(snapshot.get('balance', 0.0)):,.2f}"
        )
    except Exception:
        pass

    return result.can_trade


def _load_dsp_profile(symbol: str) -> tuple[Optional[dict], Path]:
    """Load per-symbol DSP profile used by backtest/paper/live modes."""
    dsp_file = get_data_path(symbol) / "dsp_profile.json"
    if not dsp_file.exists():
        return None, dsp_file
    with open(dsp_file) as f:
        return json.load(f), dsp_file


def _compute_and_store_dsp_profile(
    symbol: str, closes: Optional[pd.Series] = None
) -> tuple[Optional[dict], Path]:
    """Compute DSP from M1 data and persist canonical profile."""
    dsp_file = get_data_path(symbol) / "dsp_profile.json"
    if closes is None:
        closes = load_m1_data(symbol)
    if closes is None or len(closes) == 0:
        return None, dsp_file

    from kinetra.renko.dsp import run_dsp

    raw = run_dsp(closes, symbol=symbol, bars_per_hour=60.0)
    if raw is None:
        return None, dsp_file

    # Convert DSPResult dataclass → plain dict (run_dsp returns a dataclass).
    result: dict = asdict(raw) if not isinstance(raw, dict) else dict(raw)
    if "error" in result:
        return None, dsp_file

    # Alias dsp_brick_size → brick_size so downstream dsp.get("brick_size") works.
    if "brick_size" not in result and "dsp_brick_size" in result:
        result["brick_size"] = result["dsp_brick_size"]

    # Ensure cadence key exists for downstream window scaling/diagnostics.
    result.setdefault("bars_per_hour", 60.0)

    dsp_file.parent.mkdir(parents=True, exist_ok=True)
    with open(dsp_file, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, default=str)

    # Keep legacy output artifact for operator visibility.
    legacy_dir = KR / "outputs" / "dsp"
    legacy_dir.mkdir(parents=True, exist_ok=True)
    legacy_file = legacy_dir / f"{symbol}_dsp.json"
    with open(legacy_file, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, default=str)

    return result, dsp_file


def _load_or_build_dsp_profile(
    symbol: str, closes: Optional[pd.Series] = None
) -> tuple[Optional[dict], Path]:
    """Load canonical DSP profile; auto-build it from M1 when missing."""
    dsp, dsp_file = _load_dsp_profile(symbol)
    if dsp is not None:
        return dsp, dsp_file
    return _compute_and_store_dsp_profile(symbol, closes=closes)


def _json_safe(value):
    """Convert nested objects to JSON-safe types."""
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return value


def _choose_unique_path(base_dir: Path, stem: str) -> Path:
    """Return a non-existing path by appending an incrementing suffix."""
    candidate = base_dir / f"{stem}.json"
    if not candidate.exists():
        return candidate
    idx = 1
    while True:
        candidate = base_dir / f"{stem}_{idx}.json"
        if not candidate.exists():
            return candidate
        idx += 1


def _choose_unique_file(base_dir: Path, stem: str, ext: str) -> Path:
    """Return a non-existing file path with requested extension."""
    suffix = ext if ext.startswith(".") else f".{ext}"
    candidate = base_dir / f"{stem}{suffix}"
    if not candidate.exists():
        return candidate
    idx = 1
    while True:
        candidate = base_dir / f"{stem}_{idx}{suffix}"
        if not candidate.exists():
            return candidate
        idx += 1


def _save_equity_plot(
    *,
    symbol: str,
    mode: str,
    trades: list,
    initial_equity: float = 1000.0,
    run_id: str = "",
) -> Optional[Path]:
    """Save equity curve PNG from trade list; returns path when successful."""
    if not trades:
        return None
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return None

    nets = []
    times = []
    for i, t in enumerate(trades):
        if isinstance(t, dict):
            net = float(t.get("net_usd", 0.0))
            ts = t.get("exit_time")
        else:
            net = float(getattr(t, "net_usd", 0.0))
            ts = getattr(t, "exit_time", None)
        nets.append(net)
        times.append(ts if ts is not None else i)

    eq = np.asarray(nets, dtype=np.float64)
    eq = np.cumsum(eq) + float(initial_equity)
    if eq.size == 0:
        return None

    out_dir = KR / "outputs" / "renko_results" / "plots" / symbol.upper()
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = (
        f"{symbol.lower()}_{mode}_equity_{run_id}" if run_id else f"{symbol.lower()}_{mode}_equity"
    )
    out_path = _choose_unique_file(out_dir, stem, ".png")

    # Prefer time axis when timestamps are parseable.
    x = pd.to_datetime(times, utc=True, errors="coerce")
    use_time_axis = bool(getattr(x, "notna", lambda: pd.Series([], dtype=bool))().all())
    if use_time_axis:
        x_axis = x
    else:
        x_axis = np.arange(eq.size)

    fig, ax = plt.subplots(figsize=(10.5, 4.2))
    ax.plot(x_axis, eq, lw=1.5, color="#0b6e4f")
    ax.set_title(f"{symbol.upper()} {mode.upper()} Equity Curve")
    ax.set_ylabel("Equity (USD)")
    ax.grid(alpha=0.25)
    if not use_time_axis:
        ax.set_xlabel("Trade #")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return out_path


def _save_equity_compare_plot(
    *,
    symbol: str,
    mode: str,
    trades_is: list,
    trades_oos: list,
    initial_equity: float = 1000.0,
    run_id: str = "",
) -> Optional[Path]:
    """Save a 2-line IS vs OOS equity curve PNG."""
    if not trades_is and not trades_oos:
        return None
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return None

    def _curve(trades_list: list) -> np.ndarray:
        if not trades_list:
            return np.asarray([float(initial_equity)], dtype=np.float64)
        vals = []
        for t in trades_list:
            if isinstance(t, dict):
                vals.append(float(t.get("net_usd", 0.0)))
            else:
                vals.append(float(getattr(t, "net_usd", 0.0)))
        arr = np.cumsum(np.asarray(vals, dtype=np.float64)) + float(initial_equity)
        return arr if arr.size > 0 else np.asarray([float(initial_equity)], dtype=np.float64)

    eq_is = _curve(trades_is)
    eq_oos = _curve(trades_oos)

    out_dir = KR / "outputs" / "renko_results" / "plots" / symbol.upper()
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = (
        f"{symbol.lower()}_{mode}_equity_compare_{run_id}"
        if run_id
        else f"{symbol.lower()}_{mode}_equity_compare"
    )
    out_path = _choose_unique_file(out_dir, stem, ".png")

    fig, ax = plt.subplots(figsize=(10.5, 4.2))
    ax.plot(np.arange(eq_is.size), eq_is, lw=1.4, color="#1f77b4", label="IS")
    ax.plot(np.arange(eq_oos.size), eq_oos, lw=1.4, color="#d62728", label="OOS")
    ax.set_title(f"{symbol.upper()} {mode.upper()} Equity (IS vs OOS)")
    ax.set_xlabel("Trade #")
    ax.set_ylabel("Equity (USD)")
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return out_path


def _save_leverage_plot(
    *,
    symbol: str,
    mode: str,
    trades: list,
    initial_equity: float,
    usd_per_point: float,
    max_leverage: Optional[float] = None,
    run_id: str = "",
) -> Optional[Path]:
    """Save actual leverage usage plot from closed trades."""
    if not trades:
        return None
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return None

    lev_vals: list[float] = []
    times = []
    equity = float(initial_equity)
    usdpp = max(float(usd_per_point), 0.0)

    for i, t in enumerate(trades):
        if isinstance(t, dict):
            lots = float(t.get("lots", 0.0))
            entry_price = float(t.get("entry_price", 0.0))
            net = float(t.get("net_usd", 0.0))
            ts = t.get("entry_time") or t.get("exit_time")
        else:
            lots = float(getattr(t, "lots", 0.0))
            entry_price = float(getattr(t, "entry_price", 0.0))
            net = float(getattr(t, "net_usd", 0.0))
            ts = getattr(t, "entry_time", None) or getattr(t, "exit_time", None)

        notional_usd = abs(lots) * abs(entry_price) * usdpp
        lev = notional_usd / max(abs(equity), 1e-12)
        lev_vals.append(float(lev))
        times.append(ts if ts is not None else i)
        equity += net

    arr = np.asarray(lev_vals, dtype=np.float64)
    if arr.size == 0:
        return None

    out_dir = KR / "outputs" / "renko_results" / "plots" / symbol.upper()
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = (
        f"{symbol.lower()}_{mode}_leverage_{run_id}"
        if run_id
        else f"{symbol.lower()}_{mode}_leverage"
    )
    out_path = _choose_unique_file(out_dir, stem, ".png")

    x = pd.to_datetime(times, utc=True, errors="coerce")
    use_time_axis = bool(getattr(x, "notna", lambda: pd.Series([], dtype=bool))().all())
    x_axis = x if use_time_axis else np.arange(arr.size)

    fig, ax = plt.subplots(figsize=(10.5, 4.2))
    ax.plot(x_axis, arr, lw=1.4, color="#8b0000", label="Actual leverage")
    if max_leverage is not None and float(max_leverage) > 0:
        ax.axhline(
            y=float(max_leverage),
            color="#1f77b4",
            lw=1.2,
            ls="--",
            label=f"Max leverage 1:{float(max_leverage):.1f}",
        )
    ax.set_title(f"{symbol.upper()} {mode.upper()} Leverage Usage")
    ax.set_ylabel("Leverage (x)")
    if not use_time_axis:
        ax.set_xlabel("Trade #")
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return out_path


def _find_m1_source_file(symbol: str) -> Optional[Path]:
    data_dir = _canonical_symbol_dir(symbol)
    return _select_best_m1_file(data_dir, symbol)


def _canonical_category_for_symbol(symbol: str) -> str:
    from kinetra.canonical_asset_classification import get_asset_class_with_fallback

    cls = get_asset_class_with_fallback(symbol)
    val = str(getattr(cls, "value", "other")).lower()
    mapping = {
        "metal": "metals",
        "metals": "metals",
        "index": "indices",
        "indices": "indices",
        "commodity": "commodities",
        "commodities": "commodities",
        "stock": "shares",
        "shares": "shares",
        "etf": "other",
        "etfs": "other",
    }
    return mapping.get(val, val if val in {"forex", "crypto", "energy"} else "other")


def _canonical_symbol_dir(symbol: str) -> Path:
    category = _canonical_category_for_symbol(symbol)
    return (
        KR / "data" / "master_standardized" / "ctrader" / "pepperstone" / category / symbol.upper()
    )


def _list_symbol_m1_files(symbol: str, max_depth: int = 6) -> list[Path]:
    data_root = KR / "data"
    pattern = re.compile(
        rf"{re.escape(symbol.upper())}_M1_(\d{{12,14}})_(\d{{12,14}})\.csv$",
        re.IGNORECASE,
    )
    out: list[Path] = []
    for p in data_root.rglob("*.csv"):
        try:
            rel = p.relative_to(data_root)
        except Exception:
            continue
        if len(rel.parts) > max_depth:
            continue
        if pattern.search(p.name):
            out.append(p)
    return sorted(out)


def _score_m1_file(path: Path, symbol: str) -> tuple[float, int, float]:
    pattern = re.compile(
        rf"{re.escape(symbol.upper())}_M1_(\d{{12,14}})_(\d{{12,14}})\.csv$",
        re.IGNORECASE,
    )
    span_s = -1.0
    m = pattern.search(path.name)
    if m:
        fmt = "%Y%m%d%H%M%S" if len(m.group(1)) == 14 else "%Y%m%d%H%M"
        try:
            s = datetime.strptime(m.group(1), fmt).replace(tzinfo=timezone.utc)
            e = datetime.strptime(m.group(2), fmt).replace(tzinfo=timezone.utc)
            span_s = max((e - s).total_seconds(), 0.0)
        except Exception:
            span_s = -1.0
    st = path.stat()
    return span_s, int(st.st_size), float(st.st_mtime)


def _load_m1_csv_generic(path: Path) -> Optional[pd.DataFrame]:
    try:
        df = pd.read_csv(path)
    except Exception:
        return None
    if df.empty:
        return None

    cols = {c.lower(): c for c in df.columns}
    time_col = next((cols[k] for k in ("time", "datetime", "date", "timestamp") if k in cols), None)
    close_col = cols.get("close")
    if time_col is None or close_col is None:
        return None

    work = pd.DataFrame()
    work["time"] = pd.to_datetime(df[time_col], utc=True, errors="coerce")
    for col in ("open", "high", "low", "close", "volume", "spread"):
        src = cols.get(col)
        work[col] = pd.to_numeric(df[src], errors="coerce") if src is not None else pd.NA

    work = work.dropna(subset=["time", "close"]).sort_values("time")
    if work.empty:
        return None
    work = work.drop_duplicates(subset=["time"], keep="last")
    return work


def _standardize_symbol_m1_data(symbol: str, max_depth: int = 6) -> None:
    """Recursively discover and merge symbol M1 CSVs into canonical path."""
    sym = symbol.upper()
    if sym in _STANDARDIZED_SYMBOLS:
        return

    files = _list_symbol_m1_files(sym, max_depth=max_depth)
    if not files:
        return

    ranked = sorted(files, key=lambda p: _score_m1_file(p, sym))
    parts: list[pd.DataFrame] = []
    for p in ranked:
        d = _load_m1_csv_generic(p)
        if d is not None and not d.empty:
            parts.append(d)
    if not parts:
        return

    merged = pd.concat(parts, ignore_index=True)
    merged = merged.sort_values("time").drop_duplicates(subset=["time"], keep="last")
    if merged.empty:
        return

    # Fill tiny gaps safely so the canonical dataset is contiguous for M1 consumers.
    for col in ("open", "high", "low", "volume", "spread"):
        if col in merged.columns:
            merged[col] = merged[col].ffill().bfill()
    merged["close"] = merged["close"].ffill().bfill()
    merged["time"] = merged["time"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")

    out_dir = _canonical_symbol_dir(sym)
    out_dir.mkdir(parents=True, exist_ok=True)
    t = pd.to_datetime(merged["time"], utc=True, errors="coerce").dropna()
    if t.empty:
        return
    start_str = t.iloc[0].strftime("%Y%m%d%H%M")
    end_str = t.iloc[-1].strftime("%Y%m%d%H%M")
    out_path = out_dir / f"{sym}_M1_{start_str}_{end_str}.csv"

    # Keep the CSV with most data in canonical folder; overwrite only if improved/new.
    best_existing = _select_best_m1_file(out_dir, sym)
    should_write = True
    if best_existing is not None and best_existing.exists():
        try:
            existing_rows = max(sum(1 for _ in best_existing.open("r", encoding="utf-8")) - 1, 0)
        except Exception:
            existing_rows = 0
        should_write = len(merged) >= existing_rows
    if should_write:
        merged.to_csv(out_path, index=False)
        LOG.info("Standardized %s M1 data -> %s (%d bars)", sym, out_path, len(merged))

    _STANDARDIZED_SYMBOLS.add(sym)


def _persist_execution_audit(
    *,
    symbol: str,
    mode: str,
    results: dict,
    metadata: Optional[dict] = None,
) -> Path:
    """Persist signal/order/trade reconciliation artifact."""
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    out_dir = KR / "outputs" / "renko_results" / "audit" / symbol.upper()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = _choose_unique_path(out_dir, f"{symbol.lower()}_{mode}_audit_{ts}")

    payload = {
        "symbol": symbol.upper(),
        "mode": mode,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "summary": results.get("summary", {}),
        "signals": results.get("signals", []),
        "trades": results.get("trades", []),
        "metadata": metadata or {},
    }
    with open(out_path, "x", encoding="utf-8") as f:
        json.dump(_json_safe(payload), f, indent=2)
    return out_path


def _select_best_m1_file(data_dir: Path, symbol: str) -> Optional[Path]:
    """Pick the best M1 CSV for a symbol by widest encoded time span."""
    files = sorted(data_dir.glob("*_M1_*.csv"))
    if not files:
        return None

    pattern = re.compile(
        rf"{re.escape(symbol.upper())}_M1_(\d{{12,14}})_(\d{{12,14}})\.csv$",
        re.IGNORECASE,
    )
    scored = []
    for p in files:
        span_s = -1.0
        m = pattern.search(p.name)
        if m:
            fmt = "%Y%m%d%H%M%S" if len(m.group(1)) == 14 else "%Y%m%d%H%M"
            try:
                s = datetime.strptime(m.group(1), fmt).replace(tzinfo=timezone.utc)
                e = datetime.strptime(m.group(2), fmt).replace(tzinfo=timezone.utc)
                span_s = max((e - s).total_seconds(), 0.0)
            except Exception:
                span_s = -1.0
        st = p.stat()
        scored.append((span_s, int(st.st_size), float(st.st_mtime), p))

    scored.sort(reverse=True, key=lambda x: (x[0], x[1], x[2]))
    return scored[0][3]


def _ensure_backtest_history(symbol: str, months: int, auto_download: bool) -> bool:
    """Ensure local M1 history can cover requested backtest lookback."""
    sym = symbol.upper()
    closes = load_m1_data(sym)

    def _has_coverage(series: pd.Series, req_months: int) -> bool:
        if series is None or series.empty:
            return False
        end_ts = series.index[-1]
        have_start = series.index[0]
        need_start = end_ts - pd.DateOffset(months=int(req_months))
        return have_start <= need_start

    def _run_download(days: int, mode: str) -> bool:
        cmd = [
            sys.executable,
            str(KR / "scripts" / "ctrader" / "download_ctrader_history.py"),
            "--symbols",
            sym,
            "--days",
            str(max(int(days), 1)),
            "--chunk-days",
            "90",
        ]
        if mode == "backfill":
            cmd.append("--backfill")
        else:
            cmd.append("--resume")
        rc = subprocess.run(cmd, cwd=str(KR)).returncode
        if rc != 0:
            click.secho(f"❌ Auto-{mode} failed (exit={rc})", fg="red")
            return False
        _STANDARDIZED_SYMBOLS.discard(sym)
        _standardize_symbol_m1_data(sym, max_depth=6)
        return True

    if closes is None or closes.empty:
        if not auto_download:
            return False
        days = max(int(months) * 31, 31)
        click.secho(
            f"No local M1 data for {symbol}. Auto-download enabled; fetching ~{days} days.",
            fg="yellow",
        )
        if not _run_download(days=days, mode="resume"):
            return False
        refreshed = load_m1_data(sym)
        if not _has_coverage(refreshed, months):
            click.secho(
                f"❌ Auto-download completed but history still does not cover {months} months.",
                fg="red",
            )
            return False
        return True

    end_ts = closes.index[-1]
    have_start = closes.index[0]
    need_start = end_ts - pd.DateOffset(months=int(months))
    if have_start <= need_start:
        return True

    if not auto_download:
        have_days = int((end_ts - have_start).total_seconds() // 86400)
        need_days = int((end_ts - need_start).total_seconds() // 86400)
        click.secho(
            f"⚠️ Insufficient local history for {months} months: have ~{have_days} days, need ~{need_days} days. "
            "Re-run with --auto-download.",
            fg="yellow",
        )
        return False

    # 1) Forward fill first (resume latest bars) in 90-day API chunk mode.
    click.secho(f"Forward-fill latest bars for {sym} (resume mode)...", fg="yellow")
    if not _run_download(days=90, mode="resume"):
        return False

    # 2) Backfill older history in paginated 90-day windows (go backwards).
    refreshed = load_m1_data(sym)
    max_pages = 36  # up to ~9 years in 90-day pages
    page = 0
    while refreshed is not None and not refreshed.empty and not _has_coverage(refreshed, months):
        end_ts = refreshed.index[-1]
        have_start = refreshed.index[0]
        need_start = end_ts - pd.DateOffset(months=int(months))
        missing_days = max(1, int((have_start - need_start).total_seconds() // 86400) + 2)
        page_days = min(90, missing_days)
        page += 1
        click.secho(
            f"Backfill page {page}/{max_pages}: downloading {page_days} day chunk backwards...",
            fg="yellow",
        )
        if not _run_download(days=page_days, mode="backfill"):
            return False
        refreshed = load_m1_data(sym)
        if page >= max_pages:
            break

    if not _has_coverage(refreshed, months):
        have_days = 0
        if refreshed is not None and not refreshed.empty:
            have_days = int((refreshed.index[-1] - refreshed.index[0]).total_seconds() // 86400)
            need_days = int(
                (
                    refreshed.index[-1] - (refreshed.index[-1] - pd.DateOffset(months=int(months)))
                ).total_seconds()
                // 86400
            )
        else:
            need_days = int(months) * 31
        click.secho(
            f"❌ Backfill completed but history is still short: have ~{have_days} days, need ~{need_days} days.",
            fg="red",
        )
        return False
    return True


def _build_stream_runtime(symbol: str, dsp: dict, lot_ceiling: float, live_orders: bool):
    """Build shared runtime components for paper/dry-run/live execution."""
    from kinetra.renko.ctrader_dispatcher import CTraderBarProvider, CTraderOrderDispatcher
    from kinetra.renko.live_trader import PaperDispatcher

    cfg, spec = _build_engine_config(symbol, dsp, sizing_mode="static", lot_ceiling=lot_ceiling)
    engine = RenkoEngine(cfg, spec=spec if live_orders else None, quiet_mode=live_orders)
    bar_provider = CTraderBarProvider
    dispatcher_cls = CTraderOrderDispatcher if live_orders else PaperDispatcher
    return cfg, spec, engine, bar_provider, dispatcher_cls


def _apply_strategy_overrides(
    cfg: EngineConfig,
    *,
    stop_bricks: Optional[float] = None,
    trailing_mfe_fraction: Optional[float] = None,
    trailing_mfe_after_bricks: Optional[int] = None,
    target_risk: Optional[float] = None,
    brick_size: Optional[float] = None,
    fliprate_window: Optional[int] = None,
    markov_window: Optional[int] = None,
    fliprate_threshold: Optional[float] = None,
    markov_threshold: Optional[float] = None,
    max_leverage: Optional[float] = None,
    spread_sides: Optional[float] = None,
    conservative_fills: Optional[bool] = None,
    entry_slip_bricks: Optional[float] = None,
    exit_slip_bricks: Optional[float] = None,
    stop_worst_case_bricks: Optional[float] = None,
    flip_trade_through_bricks: Optional[float] = None,
    use_pe_gate: Optional[bool] = None,
    pe_window: Optional[int] = None,
    pe_order: Optional[int] = None,
    pe_entry_threshold: Optional[float] = None,
    pe_exit_threshold: Optional[float] = None,
    use_time_decay: Optional[bool] = None,
    stale_brick_factor: Optional[float] = None,
    markov_stale_penalty: Optional[float] = None,
    use_obi_slippage_buffer: Optional[bool] = None,
    obi_levels: Optional[int] = None,
    obi_max_buffer_bricks: Optional[float] = None,
    obi_ema_alpha: Optional[float] = None,
) -> None:
    """Apply CLI optimization overrides consistently across all execution modes."""
    if stop_bricks is not None:
        cfg.stop_bricks = float(stop_bricks)
    if trailing_mfe_fraction is not None:
        cfg.trailing_mfe_fraction = float(trailing_mfe_fraction)
    if trailing_mfe_after_bricks is not None:
        cfg.trailing_mfe_after_bricks = int(trailing_mfe_after_bricks)
    if target_risk is not None:
        cfg.target_risk_usd = float(target_risk)
    if brick_size is not None:
        cfg.brick_size = float(brick_size)
    if fliprate_window is not None:
        cfg.fliprate_window = int(fliprate_window)
    if markov_window is not None:
        cfg.markov_window = int(markov_window)
    if fliprate_threshold is not None:
        cfg.fliprate_threshold = float(fliprate_threshold)
    if markov_threshold is not None:
        cfg.markov_threshold = float(markov_threshold)
    if max_leverage is not None:
        lev = float(max_leverage)
        if lev <= 0:
            raise ValueError("max_leverage must be > 0")
        # Conservative cap: effective leverage cannot exceed broker or user cap.
        cfg.margin_rate = max(float(getattr(cfg, "margin_rate", 0.0)), 1.0 / lev)
    if spread_sides is not None:
        cfg.spread_sides = float(spread_sides)
    if conservative_fills is not None:
        cfg.conservative_fills = bool(conservative_fills)
    if entry_slip_bricks is not None:
        cfg.entry_slip_bricks = float(entry_slip_bricks)
    if exit_slip_bricks is not None:
        cfg.exit_slip_bricks = float(exit_slip_bricks)
    if stop_worst_case_bricks is not None:
        cfg.stop_worst_case_bricks = float(stop_worst_case_bricks)
    if flip_trade_through_bricks is not None:
        cfg.flip_trade_through_bricks = float(flip_trade_through_bricks)
    if use_pe_gate is not None:
        cfg.use_permutation_entropy = bool(use_pe_gate)
    if pe_window is not None:
        cfg.pe_window = int(pe_window)
    if pe_order is not None:
        cfg.pe_order = int(pe_order)
    if pe_entry_threshold is not None:
        cfg.pe_entry_threshold = float(pe_entry_threshold)
    if pe_exit_threshold is not None:
        cfg.pe_exit_threshold = float(pe_exit_threshold)
    if use_time_decay is not None:
        cfg.use_time_decay = bool(use_time_decay)
    if stale_brick_factor is not None:
        cfg.stale_brick_factor = float(stale_brick_factor)
    if markov_stale_penalty is not None:
        cfg.markov_stale_penalty = float(markov_stale_penalty)
    if use_obi_slippage_buffer is not None:
        cfg.use_obi_slippage_buffer = bool(use_obi_slippage_buffer)
    if obi_levels is not None:
        cfg.obi_levels = int(obi_levels)
    if obi_max_buffer_bricks is not None:
        cfg.obi_max_buffer_bricks = float(obi_max_buffer_bricks)
    if obi_ema_alpha is not None:
        cfg.obi_ema_alpha = float(obi_ema_alpha)


def _apply_latency_slippage_to_completed_trades(
    *,
    completed_trades: list,
    brick_size: float,
    usd_per_point: float,
    entry_latency_ms: int = 0,
    exit_latency_ms: int = 0,
    latency_jitter_ms: int = 0,
    latency_seed: int = 42,
) -> dict:
    """Apply latency-induced adverse slippage to already-closed trades."""
    if not completed_trades or (
        entry_latency_ms <= 0 and exit_latency_ms <= 0 and latency_jitter_ms <= 0
    ):
        return {
            "enabled": False,
            "entry_latency_ms": int(entry_latency_ms),
            "exit_latency_ms": int(exit_latency_ms),
            "latency_jitter_ms": int(latency_jitter_ms),
            "latency_seed": int(latency_seed),
            "trades_adjusted": 0,
            "avg_entry_slip": 0.0,
            "avg_exit_slip": 0.0,
        }

    rng = np.random.default_rng(int(latency_seed))
    entry_base = max(float(entry_latency_ms), 0.0) / 60000.0 * float(brick_size)
    exit_base = max(float(exit_latency_ms), 0.0) / 60000.0 * float(brick_size)
    jitter_std = max(float(latency_jitter_ms), 0.0) / 60000.0 * float(brick_size)

    adjusted = 0
    entry_slip_sum = 0.0
    exit_slip_sum = 0.0

    for t in completed_trades:
        if getattr(t, "is_open", False):
            continue
        if getattr(t, "entry_price", None) is None or getattr(t, "exit_price", None) is None:
            continue

        try:
            dir_sign = int(getattr(t, "direction").value)
        except Exception:
            continue
        if dir_sign not in (-1, 1):
            continue

        e_slip = max(0.0, entry_base + float(rng.normal(0.0, jitter_std)))
        x_slip = max(0.0, exit_base + float(rng.normal(0.0, jitter_std)))

        original_entry = float(t.entry_price)
        original_exit = float(t.exit_price)
        new_entry = original_entry + dir_sign * e_slip
        new_exit = original_exit - dir_sign * x_slip

        t.entry_price = new_entry
        t.exit_price = new_exit
        t.gross_pts = (new_exit - new_entry) * dir_sign
        t.gross_usd = t.gross_pts * float(usd_per_point) * float(t.lots)
        t.net_usd = t.gross_usd - float(getattr(t, "friction_usd", 0.0))

        adjusted += 1
        entry_slip_sum += e_slip
        exit_slip_sum += x_slip

    return {
        "enabled": adjusted > 0,
        "entry_latency_ms": int(entry_latency_ms),
        "exit_latency_ms": int(exit_latency_ms),
        "latency_jitter_ms": int(latency_jitter_ms),
        "latency_seed": int(latency_seed),
        "trades_adjusted": int(adjusted),
        "avg_entry_slip": float(entry_slip_sum / adjusted) if adjusted > 0 else 0.0,
        "avg_exit_slip": float(exit_slip_sum / adjusted) if adjusted > 0 else 0.0,
    }


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


class DriftAdaptationController:
    """Bounded online adaptation of safe live parameters under drift."""

    def __init__(self, engine: "RenkoEngine", opts: dict) -> None:
        self.engine = engine
        self.opts = opts
        self.enabled = bool(opts.get("enabled", False))
        self.last_bar = -1
        self.baseline = {
            "target_risk_usd": float(engine.cfg.target_risk_usd),
            "stop_bricks": float(engine.cfg.stop_bricks),
            "fliprate_threshold": float(engine.cfg.fliprate_threshold),
            "markov_threshold": float(engine.cfg.markov_threshold),
        }
        self.adapt_count = 0
        self.last_action = "none"

    def _set_param(self, key: str, new_val: float, reason: str) -> None:
        old_val = float(getattr(self.engine.cfg, key))
        if abs(new_val - old_val) < 1e-12:
            return
        setattr(self.engine.cfg, key, float(new_val))
        self.adapt_count += 1
        self.last_action = f"{key}:{old_val:.4f}->{new_val:.4f} ({reason})"
        LOG.warning("[ADAPT] %s", self.last_action)

    def _rollback(self, reason: str) -> None:
        for key, val in self.baseline.items():
            setattr(self.engine.cfg, key, float(val))
        self.last_action = f"rollback ({reason})"
        LOG.warning("[ADAPT] rollback to baseline: %s", reason)

    def maybe_adapt(self, bars: int) -> None:
        if not self.enabled:
            return
        step = int(self.opts.get("every_bars", 100))
        if bars <= 0 or bars - self.last_bar < step:
            return
        self.last_bar = bars

        summary = self.engine._make_results().get("summary", {})
        n_trades = int(summary.get("n_trades", 0))
        if n_trades < int(self.opts.get("min_trades", 20)):
            return

        omega = float(summary.get("omega", 0.0))
        win_rate = float(summary.get("win_rate", 0.0))
        if omega < float(self.opts.get("rollback_omega", 0.8)) and self.adapt_count > 0:
            self._rollback(f"omega={omega:.3f} below rollback")
            return

        low_omega = float(self.opts.get("low_omega", 1.0))
        high_omega = float(self.opts.get("high_omega", 1.8))
        low_wr = float(self.opts.get("low_wr", 0.45))
        high_wr = float(self.opts.get("high_wr", 0.60))
        step_risk = float(self.opts.get("step_risk", 0.10))
        step_thr = float(self.opts.get("step_thr", 0.01))
        step_stop = float(self.opts.get("step_stop", 0.05))

        if omega < low_omega or win_rate < low_wr:
            # Defensive mode under drift.
            self._set_param(
                "target_risk_usd",
                _clamp(
                    self.engine.cfg.target_risk_usd * (1.0 - step_risk),
                    float(self.opts.get("risk_min", 25.0)),
                    float(self.opts.get("risk_max", 200.0)),
                ),
                f"defensive omega={omega:.3f} wr={win_rate:.3f}",
            )
            self._set_param(
                "stop_bricks",
                _clamp(
                    self.engine.cfg.stop_bricks * (1.0 + step_stop),
                    float(self.opts.get("stop_min", 0.3)),
                    float(self.opts.get("stop_max", 1.5)),
                ),
                "defensive widen stop",
            )
            self._set_param(
                "fliprate_threshold",
                _clamp(
                    self.engine.cfg.fliprate_threshold + step_thr,
                    float(self.opts.get("flip_min", 0.25)),
                    float(self.opts.get("flip_max", 0.45)),
                ),
                "defensive stricter flip gate",
            )
            self._set_param(
                "markov_threshold",
                _clamp(
                    self.engine.cfg.markov_threshold + step_thr,
                    float(self.opts.get("markov_min", 0.50)),
                    float(self.opts.get("markov_max", 0.70)),
                ),
                "defensive stricter markov gate",
            )
            return

        if omega > high_omega and win_rate > high_wr:
            # Opportunity mode in stable favorable regime.
            self._set_param(
                "target_risk_usd",
                _clamp(
                    self.engine.cfg.target_risk_usd * (1.0 + step_risk),
                    float(self.opts.get("risk_min", 25.0)),
                    float(self.opts.get("risk_max", 200.0)),
                ),
                f"offensive omega={omega:.3f} wr={win_rate:.3f}",
            )
            self._set_param(
                "stop_bricks",
                _clamp(
                    self.engine.cfg.stop_bricks * (1.0 - step_stop),
                    float(self.opts.get("stop_min", 0.3)),
                    float(self.opts.get("stop_max", 1.5)),
                ),
                "offensive tighten stop",
            )
            self._set_param(
                "fliprate_threshold",
                _clamp(
                    self.engine.cfg.fliprate_threshold - step_thr,
                    float(self.opts.get("flip_min", 0.25)),
                    float(self.opts.get("flip_max", 0.45)),
                ),
                "offensive looser flip gate",
            )
            self._set_param(
                "markov_threshold",
                _clamp(
                    self.engine.cfg.markov_threshold - step_thr,
                    float(self.opts.get("markov_min", 0.50)),
                    float(self.opts.get("markov_max", 0.70)),
                ),
                "offensive looser markov gate",
            )


def _run_streaming_engine(
    engine, bar_provider, dispatcher, stop_event: threading.Event, stop_msg: str
):
    """Run the shared streaming loop for paper/dry-run/live modes."""
    try:
        return engine.run(
            bar_provider=bar_provider,
            dispatcher=dispatcher,
            stop_event=stop_event,
        )
    except KeyboardInterrupt:
        click.echo()
        click.secho(stop_msg, fg="yellow")
        stop_event.set()
        return engine._make_results()


def _run_execution_preflight_trade(
    connector,
    symbol: str,
    cfg: EngineConfig,
    bar_provider_cls,
    dispatcher_cls,
    lots: float,
    hold_seconds: float = 1.5,
    timeout_s: float = 90.0,  # Increased from 30s - M1 bars arrive once per minute
) -> bool:
    """Place and close a tiny live order to verify execution path."""
    from kinetra.renko.live_trader import TradeDirection

    click.echo()
    click.secho("Running execution preflight (tiny live order)...", fg="yellow", bold=True)

    bar_provider = bar_provider_cls(connector)
    dispatcher = dispatcher_cls(connector, bar_provider)
    latest = {"close": None}
    bar_ready = threading.Event()

    def _on_bar(symbol: str, close: float, timestamp, **_) -> None:
        latest["close"] = float(close)
        bar_ready.set()

    bar_provider.subscribe(symbol, _on_bar)
    try:
        bar_provider.start()
    except Exception as e:
        click.secho(f"❌ Execution preflight failed: bar provider start error ({e})", fg="red")
        return False

    try:
        click.secho(f"  Waiting for live M1 bar (timeout: {timeout_s}s)...", fg="cyan")
        if not bar_ready.wait(timeout=timeout_s):
            click.secho("❌ Execution preflight failed: no live bar received", fg="red")
            return False

        entry_price = float(latest["close"])
        stop_dist = max(cfg.stop_bricks * cfg.brick_size, cfg.tick_size)
        open_result = dispatcher.open_position(
            symbol=symbol,
            direction=TradeDirection.LONG,
            lots=lots,
            price=entry_price,
            stop_price=entry_price - stop_dist,
            comment="preflight_open",
        )
        if not getattr(open_result, "success", False):
            click.secho(
                f"❌ Execution preflight failed: open rejected ({getattr(open_result, 'error', 'unknown')})",
                fg="red",
            )
            return False

        order_id = getattr(open_result, "order_id", "")
        click.secho(f"  Opened tiny preflight position id={order_id}", fg="cyan")
        time.sleep(max(0.0, hold_seconds))

        close_result = dispatcher.close_position(
            symbol=symbol,
            order_id=order_id,
            price=entry_price,
            lots=lots,
            comment="preflight_close",
        )
        if not getattr(close_result, "success", False):
            click.secho(
                f"❌ Execution preflight failed: close rejected ({getattr(close_result, 'error', 'unknown')})",
                fg="red",
            )
            return False

        click.secho("✅ Execution preflight passed (open + close confirmed)", fg="green")
        return True
    finally:
        try:
            bar_provider.stop()
        except Exception:
            pass


def _stats_panel(
    summary: dict,
    symbol: str,
    mode: str,
    engine: Optional["RenkoEngine"] = None,
    trades: list = None,
) -> Panel:
    """Build a rich Panel from engine state (single source of truth) with detailed stats."""
    # ── EXTRACT METRICS ────────────────────────────────────────────────────────
    # Trade counts
    n = summary.get("n_trades", 0)
    w = summary.get("n_winners", 0)
    l = summary.get("n_losers", 0)
    wr = summary.get("win_rate", 0.0)

    # P&L
    net = summary.get("net_usd", 0.0)
    eq = summary.get("final_equity", 0.0)
    gross_profit = summary.get("gross_profit", 0.0)
    gross_loss = summary.get("gross_loss", 0.0)
    avg_trade = summary.get("avg_trade", 0.0)
    avg_w = summary.get("avg_winner", 0.0)
    avg_l = summary.get("avg_loser", 0.0)
    expect = summary.get("expectancy", 0.0)
    pf = summary.get("profit_factor", 0.0)

    # Risk-adjusted metrics
    om = summary.get("omega", 0.0)
    z_factor = summary.get("z_factor", 0.0)
    sharpe = summary.get("sharpe_ratio", 0.0)
    calmar = summary.get("calmar_ratio", 0.0)

    # Streaks
    max_ws = summary.get("max_win_streak", 0)
    max_ls = summary.get("max_loss_streak", 0)

    # Drawdown
    dd = summary.get("max_drawdown_pct", 0.0)
    dd_usd = summary.get("max_drawdown_usd", 0.0)
    dd_mtm = summary.get("max_drawdown_mtm_pct", dd)
    dd_mtm_usd = summary.get("max_drawdown_mtm_usd", dd_usd)
    max_margin_used = summary.get("max_used_margin_usd", 0.0)
    min_margin_level = summary.get("min_margin_level_pct", None)
    margin_rejects = summary.get("margin_rejects", 0)
    total_return = summary.get("total_return_pct", 0.0)

    # MAE/MFE (prefer summary, fall back to calculating from trades)
    mae_usd = summary.get("avg_mae_usd", 0.0)
    mfe_usd = summary.get("avg_mfe_usd", 0.0)
    mfe_ratio = summary.get("mfe_mae_ratio", 0.0)
    if not mae_usd and trades:
        maes = [
            abs(float(getattr(t, "max_adverse_excursion", 0)))
            for t in trades
            if hasattr(t, "max_adverse_excursion")
        ]
        mfes = [
            abs(float(getattr(t, "max_favorable_excursion", 0)))
            for t in trades
            if hasattr(t, "max_favorable_excursion")
        ]
        if maes:
            mae_usd = sum(maes) / len(maes)
        if mfes:
            mfe_usd = sum(mfes) / len(mfes)
        if maes and mfes and mae_usd > 0:
            mfe_ratio = mfe_usd / mae_usd

    # Holding times
    avg_hold = summary.get("avg_holding_hours", 0.0)
    avg_win_hold = summary.get("avg_winner_hours", 0.0)
    avg_lose_hold = summary.get("avg_loser_hours", 0.0)

    # ── BUILD TABLE ────────────────────────────────────────────────────────────
    t = Table(box=box.SIMPLE, show_header=False, padding=(0, 1))
    t.add_column("", style="dim", width=24)
    t.add_column("", justify="right", width=20)
    t.add_column("", style="dim", width=24)
    t.add_column("", justify="right", width=20)

    # Colour coding
    wr_col = "green" if wr >= 0.5 else "red"
    net_col = "green" if net >= 0 else "red"
    om_col = "green" if om >= 1.5 else ("yellow" if om >= 1.0 else "red")
    z_col = "green" if z_factor >= 2.5 else ("yellow" if z_factor >= 1.5 else "red")
    mfe_col = "green" if mfe_ratio >= 1.5 else ("yellow" if mfe_ratio >= 1.0 else "red")

    # ── LIVE POSITION (if available) ───────────────────────────────────────────
    if engine is not None:
        pos_str = "FLAT"
        if engine._in_pos:
            dir_str = "LONG" if engine._pos_dir == 1 else "SHORT"
            pos_str = (
                f"[bold yellow]{dir_str}[/] {engine._entry_lots:.3f} @ ${engine._entry_price:.2f}"
            )

        t.add_row(
            "Position",
            pos_str,
            "Bricks",
            f"{int(getattr(engine, 'bricks_processed', 0))}",
        )

    # ── SECTION: PERFORMANCE ──────────────────────────────────────────────────
    t.add_row(
        "[dim]── PERFORMANCE ──────────[/]",
        "",
        "[dim]── PERFORMANCE ──────────[/]",
        "",
    )
    t.add_row(
        "Trades",
        f"{n}  ({w}W / {l}L)",
        "Win rate",
        f"[{wr_col}]{wr:.1%}[/]",
    )
    t.add_row(
        "Net P&L",
        f"[{net_col}]${_format_number(net)}[/]",
        "Total return",
        f"{total_return:.1f}%",
    )

    # ── SECTION: RISK-ADJUSTED ────────────────────────────────────────────────
    t.add_row(
        "[dim]── RISK-ADJUSTED ────────[/]",
        "",
        "[dim]── RISK-ADJUSTED ────────[/]",
        "",
    )
    t.add_row(
        "Omega",
        f"[{om_col}]{om:.3f}[/]",
        "Z-Factor",
        f"[{z_col}]{z_factor:.2f}[/]",
    )
    t.add_row(
        "Sharpe",
        f"{sharpe:.2f}",
        "Calmar",
        f"{calmar:.1f}",
    )
    t.add_row(
        "Profit factor",
        f"{pf:.2f}",
        "Return/DD",
        f"{abs(total_return / dd):.1f}" if dd != 0 else "N/A",
    )

    # ── SECTION: P&L DETAIL ───────────────────────────────────────────────────
    t.add_row(
        "[dim]── P&L DETAIL ───────────[/]",
        "",
        "[dim]── P&L DETAIL ───────────[/]",
        "",
    )
    t.add_row(
        "Gross profit",
        f"[green]${_format_number(gross_profit)}[/]",
        "Gross loss",
        f"[red]-${_format_number(gross_loss)}[/]",
    )
    t.add_row(
        "Avg winner",
        f"[green]${_format_number(avg_w)}[/]",
        "Avg loser",
        f"[red]-${_format_number(avg_l)}[/]",
    )
    t.add_row(
        "Avg trade",
        f"${_format_number(avg_trade)}",
        "Expectancy",
        f"${_format_number(expect)}",
    )

    # ── SECTION: RISK ─────────────────────────────────────────────────────────
    t.add_row(
        "[dim]── RISK ─────────────────[/]",
        "",
        "[dim]── RISK ─────────────────[/]",
        "",
    )
    t.add_row(
        "Max drawdown",
        f"{dd:.2f}%",
        "Max DD $",
        f"${_format_number(dd_usd)}",
    )
    t.add_row(
        "Max MTM DD",
        f"{dd_mtm:.2f}%",
        "Max MTM DD $",
        f"${_format_number(dd_mtm_usd)}",
    )
    t.add_row(
        "Max win streak",
        f"{max_ws}",
        "Max loss streak",
        f"{max_ls}",
    )
    t.add_row(
        "Max used margin",
        f"${_format_number(float(max_margin_used or 0.0))}",
        "Margin rejects",
        f"{int(margin_rejects or 0)}",
    )
    t.add_row(
        "Min margin level",
        (f"{float(min_margin_level):.1f}%" if min_margin_level is not None else "N/A"),
        "",
        "",
    )

    # ── SECTION: EXECUTION (if data available) ────────────────────────────────
    if mae_usd > 0 or mfe_usd > 0 or avg_hold > 0:
        t.add_row(
            "[dim]── EXECUTION ───────────[/]",
            "",
            "[dim]── EXECUTION ───────────[/]",
            "",
        )
        if mae_usd > 0 or mfe_usd > 0:
            t.add_row(
                "Avg MAE",
                f"[red]${_format_number(mae_usd)}[/]",
                "Avg MFE",
                f"[green]${_format_number(mfe_usd)}[/]",
            )
            t.add_row(
                "MFE/MAE ratio",
                f"[{mfe_col}]{mfe_ratio:.2f}[/]",
                "",
                "",
            )
        if avg_hold > 0:
            t.add_row(
                "Avg hold time",
                f"{avg_hold:.1f}h",
                "Win hold",
                f"{avg_win_hold:.1f}h" if avg_win_hold > 0 else "-",
            )
            t.add_row(
                "Loser hold",
                f"{avg_lose_hold:.1f}h" if avg_lose_hold > 0 else "-",
                "",
                "",
            )

    # ── SECTION: EQUITY ───────────────────────────────────────────────────────
    t.add_row(
        "[dim]── EQUITY ───────────────[/]",
        "",
        "[dim]── EQUITY ───────────────[/]",
        "",
    )
    t.add_row(
        "Final equity",
        f"[bold]${_format_number(eq)}[/]",
        "",
        "",
    )

    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    border = "green" if mode.startswith("live") or mode == "paper" else "cyan"
    return Panel(
        t,
        title=f"[bold]{symbol}[/]  [{mode.upper()}]  {ts}",
        border_style=border,
        width=_adaptive_panel_width(min_width=96, max_width=170),
    )


def _print_stats(summary: dict, symbol: str, mode: str, trades: list = None) -> None:
    """Render stats panel to terminal and log a one-liner to file."""
    console.print(_stats_panel(summary, symbol, mode, trades=trades))
    LOG.info(
        "[%s %s] trades=%d net=%.2f omega=%.3f dd_closed=%.2f%% dd_mtm=%.2f%% equity=%.2f",
        symbol,
        mode,
        summary.get("n_trades", 0),
        summary.get("net_usd", 0.0),
        summary.get("omega", 0.0),
        summary.get("max_drawdown_pct", 0.0),
        summary.get("max_drawdown_mtm_pct", summary.get("max_drawdown_pct", 0.0)),
        summary.get("final_equity", 0.0),
    )


def _print_spread_sanity_samples(
    *,
    symbol: str,
    mode: str,
    trades: list,
    cfg: EngineConfig,
    sample_n: int = 20,
    seed: int = 42,
) -> None:
    """Print a small random sample of trade-level spread sanity diagnostics."""
    if not trades:
        return
    n_total = len(trades)
    n = min(int(sample_n), n_total)
    if n <= 0:
        return

    rng = np.random.default_rng(int(seed))
    idx = np.sort(rng.choice(n_total, size=n, replace=False))
    spread_points = float(getattr(cfg, "spread_ticks", 0.0))
    spread_price = spread_points * float(getattr(cfg, "tick_size", 0.0))
    spread_sides = float(getattr(cfg, "spread_sides", 1.0))
    click.echo()
    click.echo(
        f"[SPREAD SANITY] {symbol} {mode} sample={n}/{n_total} "
        f"spread_points={spread_points:.2f} spread_price={spread_price:.5f} spread_sides={spread_sides:.2f}"
    )
    click.echo("  idx  lots  sp_pts  sp_px   entry_fill  exit_fill  gross_usd  spread_usd  net_usd")
    for i in idx:
        t = trades[int(i)]
        if not isinstance(t, dict):
            continue
        lots = float(t.get("lots", 0.0) or 0.0)
        entry_fill = float(t.get("entry_price", 0.0) or 0.0)
        exit_fill = float(t.get("exit_price", 0.0) or 0.0)
        spread_usd = float(t.get("spread_usd", 0.0) or 0.0)
        net_usd = float(t.get("net_usd", 0.0) or 0.0)
        denom = (
            float(getattr(cfg, "usd_per_tick", 0.0)) * max(spread_sides, 1e-12) * max(lots, 1e-12)
        )
        spread_pts_trade = spread_usd / denom if denom > 0 else float("nan")
        spread_px_trade = spread_pts_trade * float(getattr(cfg, "tick_size", 0.0))
        gross_usd = float(
            t.get(
                "gross_usd",
                net_usd
                + spread_usd
                + float(t.get("commission_usd", 0.0) or 0.0)
                + float(t.get("swap_usd", 0.0) or 0.0),
            )
            or 0.0
        )
        click.echo(
            f"  {int(i):4d}  {lots:4.2f}  {spread_pts_trade:6.2f}  {spread_px_trade:6.4f}  "
            f"{entry_fill:10.5f}  {exit_fill:10.5f}  "
            f"{gross_usd:9.2f}  {spread_usd:10.2f}  {net_usd:8.2f}"
        )


def _dashboard_panel(
    engine: "RenkoEngine",
    symbol: str,
    mode: str,
    bricks: int,
    acct_info: dict,
) -> Panel:
    """Build compact live snapshot panel for operator visibility."""
    summary = engine._make_results().get("summary", {})
    last_eval = getattr(engine, "_last_eval", {}) or {}

    def _fmt_eval_num(v) -> str:
        try:
            f = float(v)
            if pd.isna(f):
                return "n/a"
            return f"{f:.3f}"
        except Exception:
            return "n/a"

    table = Table(box=box.SIMPLE, show_header=False, expand=True)
    table.add_column(style="cyan")
    table.add_column()
    account_type_raw = acct_info.get("account_type", "-")

    def _format_account_type(v) -> str:
        try:
            code = int(v)
        except Exception:
            return str(v)
        labels = {
            0: "Unknown",
            1: "Hedged",
            2: "Netting",
        }
        return f"{labels.get(code, 'Type')} ({code})"

    table.add_row("[dim]Trade Performance[/]", "")
    table.add_row("Mode", mode.upper())
    table.add_row("Symbol", symbol)
    table.add_row("Renko Bricks", str(int(bricks)))
    table.add_row("Dupes dropped", str(int(getattr(engine, "duplicate_bars_dropped", 0))))
    table.add_row("Trades", str(int(summary.get("n_trades", 0))))
    table.add_row("Net P&L", f"${float(summary.get('net_usd', 0.0)):,.2f}")
    table.add_row("Omega", f"{float(summary.get('omega', 0.0)):.3f}")

    table.add_row("[dim]Trade Eval / Decision[/]", "")
    table.add_row(
        "Eval",
        f"{last_eval.get('direction', 'NA')} | flip={last_eval.get('is_flip', False)} | pass={last_eval.get('entry_ok', False)}",
    )
    table.add_row(
        "Warmup",
        f"ready={last_eval.get('warmup_ready', False)} remaining={int(last_eval.get('warmup_remaining', 0) or 0)}",
    )
    warmup_target = int(max(getattr(engine.cfg, "min_warmup_bricks", 2), 1))
    table.add_row(
        "Warmup target",
        f"{warmup_target} bricks (FR={int(engine.cfg.fliprate_window)}, M={int(engine.cfg.markov_window)})",
    )
    table.add_row(
        "Startup skip",
        f"remaining_flips={int(last_eval.get('startup_skip_remaining', 0) or 0)}",
    )
    table.add_row(
        "Eval metrics",
        f"FR={_fmt_eval_num(last_eval.get('fr'))} M={_fmt_eval_num(last_eval.get('markov'))} lots={float(last_eval.get('lots', 0.0)):.3f}",
    )
    table.add_row("Decision", str(last_eval.get("reason", "n/a")))
    last_brick_time = getattr(engine, "last_brick_time", None)
    last_brick_s = (
        pd.Timestamp(last_brick_time).strftime("%H:%M:%S") if last_brick_time is not None else "n/a"
    )
    if last_brick_time is not None:
        try:
            age_s = int(
                (pd.Timestamp.now(tz="UTC") - pd.Timestamp(last_brick_time)).total_seconds()
            )
            last_brick_age = f"{max(age_s, 0)}s"
        except Exception:
            last_brick_age = "n/a"
    else:
        last_brick_age = "n/a"
    _tick_mode = bool(acct_info.get("tick_mode", False))
    _ticks_received = int(acct_info.get("ticks_received", 0))
    table.add_row("[dim]Live Filter Activity[/]", "")
    table.add_row(
        "Data source",
        "[cyan]Tick feed[/]" if _tick_mode else "[dim]M1 bars[/]",
    )
    if _tick_mode:
        table.add_row("Ticks received", str(_ticks_received))
    table.add_row(
        "Ticks seen" if _tick_mode else "Bars seen",
        str(int(getattr(engine, "stream_bars_seen", 0))),
    )
    table.add_row("Flips seen", str(int(getattr(engine, "flips_seen", 0))))
    table.add_row("Filter-ready", str(int(getattr(engine, "filter_ready_bricks", 0))))
    table.add_row("Last brick UTC", last_brick_s)
    table.add_row("Last brick age", last_brick_age)

    table.add_row("[dim]Connection[/]", "")
    status = str(acct_info.get("connection_status", "-")).upper()
    hb_on = bool(acct_info.get("heartbeat_on", False))
    if status == "UP":
        status_fmt = "[green]UP[/]"
        hb_dot = "[green]●[/]" if hb_on else "[green]○[/]"
    elif status == "DEGRADED":
        status_fmt = "[yellow]DEGRADED[/]"
        hb_dot = "[yellow]●[/]" if hb_on else "[yellow]○[/]"
    elif status == "DOWN":
        status_fmt = "[red]DOWN[/]"
        hb_dot = "[red]●[/]" if hb_on else "[red]○[/]"
    else:
        status_fmt = status
        hb_dot = "[yellow]○[/]" if hb_on else "[dim]○[/]"
    table.add_row("Status", status_fmt)
    table.add_row("Heartbeat", hb_dot)
    table.add_row("Endpoint", str(acct_info.get("endpoint", "-")))
    table.add_row("Failovers", str(acct_info.get("failover_count", "-")))
    table.add_row("Generation", str(acct_info.get("failover_generation", "-")))
    table.add_row("Last failover", str(acct_info.get("last_failover_utc", "-")))
    table.add_row("Req timeouts", str(acct_info.get("request_timeouts", "-")))
    table.add_row("Snapshot source", str(acct_info.get("snapshot_source", "-")))

    # Health and Fill Metrics
    table.add_row("[dim]Health & Execution[/]", "")
    health_status = str(acct_info.get("health_status", "UNKNOWN"))
    if health_status == "HEALTHY":
        health_fmt = "[green]HEALTHY[/]"
    elif health_status == "DEGRADED":
        health_fmt = "[yellow]DEGRADED[/]"
    elif health_status in ("UNHEALTHY", "CRITICAL"):
        health_fmt = "[red]{}[/]".format(health_status)
    else:
        health_fmt = health_status
    table.add_row("Health status", health_fmt)
    table.add_row("Health latency", f"{float(acct_info.get('health_latency_ms', 0)):.1f}ms")

    fill_rate = float(acct_info.get("fill_success_rate", 1.0))
    if fill_rate >= 0.95:
        fill_fmt = f"[green]{fill_rate:.1%}[/]"
    elif fill_rate >= 0.80:
        fill_fmt = f"[yellow]{fill_rate:.1%}[/]"
    else:
        fill_fmt = f"[red]{fill_rate:.1%}[/]"
    table.add_row("Fill success", fill_fmt)

    consec_fails = int(acct_info.get("consecutive_fill_failures", 0))
    if consec_fails == 0:
        consec_fmt = "[green]0[/]"
    elif consec_fails < 3:
        consec_fmt = f"[yellow]{consec_fails}[/]"
    else:
        consec_fmt = f"[red]{consec_fails}[/]"
    table.add_row("Consec failures", consec_fmt)

    orders_sub = int(acct_info.get("orders_submitted", 0))
    orders_ok = int(acct_info.get("orders_filled", 0))
    orders_fail = int(acct_info.get("orders_failed", 0))
    if orders_sub > 0:
        table.add_row("Orders", f"{orders_ok}/{orders_sub} OK, {orders_fail} fail")

    table.add_row("[dim]Account / Broker Info[/]", "")
    table.add_row("Balance", f"${float(acct_info.get('balance', 0.0)):,.2f}")
    table.add_row("Account", str(acct_info.get("account_id", "-")))
    table.add_row("Broker", str(acct_info.get("broker", "-")))
    table.add_row("Environment", str(acct_info.get("environment", "-")))
    table.add_row("Account type", _format_account_type(account_type_raw))

    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    return Panel(
        table,
        title=f"[bold]{symbol}[/] LIVE SNAPSHOT  {ts}",
        border_style="red",
        width=_adaptive_panel_width(min_width=96, max_width=170),
    )


def _dashboard_errors_panel(symbol: str, acct_info: dict) -> Panel:
    """Build separate recent-errors panel for live dashboard."""
    recent_errors = list(acct_info.get("recent_errors", []) or [])
    table = Table(box=box.SIMPLE, show_header=False, expand=True)
    table.add_column(style="cyan")
    table.add_column()
    if recent_errors:
        for idx, msg in enumerate(recent_errors, start=1):
            table.add_row(f"E{idx}", str(msg))
    else:
        table.add_row("E1", "none")
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    return Panel(
        table,
        title=f"[bold]{symbol}[/] LIVE ERRORS  {ts}",
        border_style="red",
        width=_adaptive_panel_width(min_width=96, max_width=170),
    )


def _dashboard_renderable(
    engine: "RenkoEngine",
    symbol: str,
    mode: str,
    bricks: int,
    acct_info: dict,
):
    """Render snapshot + recent-errors as stacked panels."""
    return Group(
        _dashboard_panel(engine, symbol, mode, bricks, acct_info),
        _dashboard_errors_panel(symbol, acct_info),
    )


def _print_dashboard(
    engine: "RenkoEngine",
    symbol: str,
    mode: str,
    bricks: int,
    acct_info: dict,
) -> None:
    """Print one snapshot panel (non-live mode)."""
    console.print(_dashboard_renderable(engine, symbol, mode, bricks, acct_info))


def _print_system_spec(
    cfg: EngineConfig,
    spec,
    dsp: dict,
    *,
    entry_latency_ms: Optional[int] = None,
    exit_latency_ms: Optional[int] = None,
    latency_jitter_ms: Optional[int] = None,
    latency_seed: Optional[int] = None,
) -> None:
    """Print complete trading system specification before performance analysis."""
    console.print()

    # Header
    spec_table = Table(box=box.ROUNDED, title="[bold cyan]TRADING SYSTEM SPECIFICATION[/]")
    spec_table.add_column("Parameter", style="cyan", width=30)
    spec_table.add_column("Value", justify="right", width=25)
    spec_table.add_column("Notes", style="dim", width=40)

    # Instrument & Broker
    spec_table.add_row("Symbol", cfg.symbol, "cTrader Pepperstone ECN")
    _sym = spec.symbol.upper()
    if any(x in _sym for x in ("XAU", "XAG", "XPT", "XPD")):
        _cs_unit = f"{spec.contract_size:.0f} oz"
        _cs_note = "Standard precious-metal lot"
    elif any(c.isdigit() for c in _sym):
        _cs_unit = f"{spec.contract_size:.0f} units"
        _cs_note = "Index CFD contract"
    else:
        _cs_unit = f"{spec.contract_size:.0f} units"
        _cs_note = "Standard forex lot"
    spec_table.add_row("Contract Size", _cs_unit, _cs_note)
    spec_table.add_row(
        "Tick Size", f"${spec.tick_size:.2f}", f"Value per tick: ${spec.tick_value_usd:.2f}"
    )

    # Brick & Filters
    spec_table.add_row("Brick Size", f"${cfg.brick_size:.2f}", "Price movement threshold")
    brick_window = cfg.fliprate_window
    spec_table.add_row(
        "Brick Window",
        f"{brick_window} bricks",
        f"Filter lookback period (~{brick_window / 50:.1f} days)",
    )

    # Entry & Exit
    spec_table.add_row("Entry Signal", "Colour flip + filters", "2-brick direction change required")
    spec_table.add_row(
        "FlipRate Gate", f"< {cfg.fliprate_threshold:.0%}", "Reject choppy markets (% flips)"
    )
    spec_table.add_row(
        "Markov Gate", f"> {cfg.markov_threshold:.0%}", "Require direction persistence"
    )
    spec_table.add_row(
        "Stop Loss (SL)",
        f"{cfg.stop_bricks:.1f} brick",
        f"${cfg.stop_bricks * cfg.brick_size:.2f} fixed distance",
    )
    spec_table.add_row("Exit Signal", "Colour change (opposite)", "Exit on first reversal brick")
    if (
        float(getattr(cfg, "trailing_mfe_fraction", 0.0)) > 0
        and int(getattr(cfg, "trailing_mfe_after_bricks", 0)) > 0
    ):
        spec_table.add_row(
            "Trailing Stop",
            f"{100.0 * float(cfg.trailing_mfe_fraction):.0f}% MFE after {int(cfg.trailing_mfe_after_bricks)} bricks",
            "Locks in fraction of max favorable excursion",
        )
    else:
        spec_table.add_row("Trailing Stop", "Off", "Standard fixed stop used")
    if bool(getattr(cfg, "conservative_fills", False)):
        spec_table.add_row(
            "Fill Model",
            "Conservative",
            (
                f"entry={float(getattr(cfg, 'entry_slip_bricks', 0.0)):.2f}b "
                f"exit={float(getattr(cfg, 'exit_slip_bricks', 0.0)):.2f}b "
                f"stop+={float(getattr(cfg, 'stop_worst_case_bricks', 0.0)):.2f}b "
                f"flip+={float(getattr(cfg, 'flip_trade_through_bricks', 0.0)):.2f}b"
            ),
        )
    else:
        spec_table.add_row("Fill Model", "Standard", "No adverse offset beyond friction model")

    # Friction Costs
    spec_table.add_row(
        "Spread",
        f"{cfg.spread_ticks:.1f} ticks",
        f"${cfg.spread_ticks * cfg.usd_per_tick * max(float(getattr(cfg, 'spread_sides', 1.0)), 0.0):.2f} modeled RT",
    )
    spec_table.add_row(
        "Commission", f"${cfg.commission_per_lot:.2f}/lot", "ECN round-trip per standard lot"
    )
    if (
        entry_latency_ms is not None
        or exit_latency_ms is not None
        or latency_jitter_ms is not None
        or latency_seed is not None
    ):
        spec_table.add_row(
            "Latency Model",
            "Backtest-only",
            "Adverse post-fill slippage model",
        )
        spec_table.add_row(
            "Entry Latency",
            f"{int(entry_latency_ms or 0)} ms",
            "Applied as adverse entry offset",
        )
        spec_table.add_row(
            "Exit Latency",
            f"{int(exit_latency_ms or 0)} ms",
            "Applied as adverse exit offset",
        )
        spec_table.add_row(
            "Latency Jitter",
            f"{int(latency_jitter_ms or 0)} ms",
            "Gaussian stddev (adverse-clamped)",
        )
        spec_table.add_row(
            "Latency Seed",
            f"{int(latency_seed or 0)}",
            "Deterministic RNG seed",
        )

    # Swap rates — prefer broker-polled USD; otherwise show raw broker value
    # in its native unit (pips for mode 0, % p.a. for mode 1).
    if spec.swap_long_usd_per_lot_per_eff_day is not None:
        _sl_str = f"${spec.swap_long_usd_per_lot_per_eff_day:.3f}/day"
        _ss_str = f"${spec.swap_short_usd_per_lot_per_eff_day:+.3f}/day"
        _swap_note = "USD per lot per day (broker-polled)"
    elif spec.swap_mode == 1:
        # Percentage annual rate — indices (NAS100, GER40, JPN225, etc.)
        _sl_str = f"{spec.swap_long_points:.2f}% p.a."
        _ss_str = f"{spec.swap_short_points:+.2f}% p.a."
        _swap_note = "Annual rate applied to notional"
    elif spec.swap_long_points or spec.swap_short_points:
        # Pips per day — metals, forex, commodities
        _sl_str = f"{spec.swap_long_points:.2f} pips/day"
        _ss_str = f"{spec.swap_short_points:+.2f} pips/day"
        _swap_note = "Pips per lot per day"
    else:
        _sl_str = None
    if _sl_str:
        spec_table.add_row("Swap Long", _sl_str, _swap_note)
        spec_table.add_row("Swap Short", _ss_str, _swap_note)
        spec_table.add_row(
            "Triple Swap", f"Day {spec.triple_swap_day}", "3× swap charged on this weekday"
        )

    # Position Sizing
    spec_table.add_row("Initial Equity", f"${cfg.initial_equity:,.2f}", "Starting account balance")
    implied_lev = (1.0 / float(cfg.margin_rate)) if float(cfg.margin_rate) > 0 else 0.0
    spec_table.add_row(
        "Max Leverage",
        f"1:{implied_lev:.1f}" if implied_lev > 0 else "N/A",
        "Effective cap after broker + user constraints",
    )
    spec_table.add_row(
        "Margin Rate",
        f"{100.0 * float(cfg.margin_rate):.3f}%",
        f"Implied leverage ~1:{implied_lev:.1f}" if implied_lev > 0 else "N/A",
    )
    spec_table.add_row(
        "Margin Call / Stop-Out",
        f"{float(cfg.margin_call_level_pct):.1f}% / {float(cfg.stop_out_level_pct):.1f}%",
        "Forced close when margin level <= stop-out",
    )
    spec_table.add_row(
        "Risk per Trade", f"${cfg.target_risk_usd:.2f}", "Target USD at risk per position"
    )
    spec_table.add_row(
        "Lot Ceiling", f"{cfg.gate_lot_ceiling:.2f} lots", "Maximum position size cap"
    )
    sizing_note = (
        "Dual scenario (static + compound)" if cfg.symbol.upper() == "XAUUSD" else cfg.sizing_mode
    )
    spec_table.add_row(
        "Sizing Mode (Backtest)",
        sizing_note,
        (
            f"Static (0.01) + Compounding (max {cfg.gate_lot_ceiling:.2f})"
            if cfg.symbol.upper() == "XAUUSD"
            else ""
        ),
    )

    # Trading Hours
    _sym_hours = spec.symbol.upper()
    if any(c.isdigit() for c in _sym_hours) and not any(x in _sym_hours for x in ("XAU", "XAG")):
        _hours_note = "Index CFD session (broker-specific hours)"
    else:
        _hours_note = "No weekend trading (forex/metals session)"
    spec_table.add_row("Trading Hours", "24/5 (Mon-Fri)", _hours_note)
    spec_table.add_row("Week Start", "Monday 00:00 UTC", "New trading week begins")
    spec_table.add_row("Week Close", "Friday 24:00 UTC", "End of week, swap settlement")

    # DSP-derived
    vr_peak = dsp.get("vr_peak_scale", "N/A")
    bars_per_hour = dsp.get("bars_per_hour", "N/A")
    regime = dsp.get("regime", "UNKNOWN")
    spec_table.add_row("VR Peak Scale", f"{vr_peak} bars", "Trend persistence peak from DSP")
    spec_table.add_row("DSP Source Rate", f"{bars_per_hour} bars/hour", "Expected: 60 for M1")
    spec_table.add_row("Regime", str(regime), "Market classification from DSP")

    # Validation gates
    spec_table.add_row("Min Omega", "≥ 1.5", "Statistical significance gate")
    spec_table.add_row("Min Trades", "≥ 30", "Sample size sufficiency")
    spec_table.add_row("Target Win Rate", "> 50%", "More winners than losers")
    spec_table.add_row("Target MFE/MAE", "> 1.5", "Execution quality (capture favorable moves)")

    console.print(spec_table)


# ──────────────────────────────────────────────────────────────────────────────
# STAGE 2: DSP ANALYSIS
# ──────────────────────────────────────────────────────────────────────────────


def stage_dsp(symbol: str) -> bool:
    """Run DSP analysis to find optimal brick size."""
    click.echo(f"\n{'=' * 60}")
    click.secho("STAGE 2: DSP ANALYSIS", fg="cyan", bold=True)
    click.echo(f"{'=' * 60}")

    # Load M1 data
    market_df = load_m1_market_data(symbol)
    if market_df is None or market_df.empty:
        click.secho(f"❌ No M1 data found for {symbol}", fg="red")
        return False
    closes = market_df["close"]
    spread_series = market_df["spread"] if "spread" in market_df.columns else None

    click.echo(f"Loaded {len(closes):,} M1 bars")

    click.echo("Running DSP analysis...")
    result, dsp_file = _compute_and_store_dsp_profile(symbol, closes=closes)
    if result is None:
        click.secho("❌ DSP analysis failed", fg="red")
        return False

    click.secho(f"✅ DSP profile saved to {dsp_file}", fg="green")

    # Print summary
    click.echo()
    click.echo(f"Brick Size: ${result.get('brick_size', 0):.2f}")
    click.echo(f"VR Peak Scale: {result.get('vr_peak_scale', 0)} bars (M1)")
    click.echo(f"DSP Source Rate: {result.get('bars_per_hour', 0)} bars/hour")
    click.echo(f"Regime: {result.get('regime', 'UNKNOWN')}")
    click.echo(f"Bricks/Day: {result.get('bricks_per_day', 0):.1f}")

    return True


def _build_engine_config(
    symbol: str,
    dsp: dict,
    sizing_mode: str = "compounding",
    lot_ceiling: float = 50.0,
) -> tuple:
    """Build a canonical EngineConfig from a DSP profile dict.

    Returns:
        (EngineConfig, InstrumentSpec) tuple

    Calibrates against InstrumentSpec (contract_spec.json) for:
    - tick_size, usd_per_tick (from broker)
    - spread_ticks (median from CSV or spec)
    - commission_per_lot (ECN rate)

    Converts vr_peak_scale (in source-data bars from DSP profile, M1 in this
    runner) to brick-based filter window by building sample bricks and
    measuring empirical brick frequency.
    """
    from kinetra.friction_cost import build_spec_with_csv_spread

    # Load spec and enrich with CSV-measured spread (bid/ask data from bar CSV).
    # Priority chain inside spec: csv_measured → typical → broker → snapshot.
    spec = build_spec_with_csv_spread(symbol)

    LOG.info(
        "Loaded %s spec: tick_size=%.5f, contract_size=%.0f, "
        "spread=%.1f pts (csv=%.1f, typical=%.1f, raw=%.1f), commission=$%.2f/lot",
        symbol,
        spec.tick_size,
        spec.contract_size,
        spec.csv_measured_spread_points or spec.typical_spread_points or spec.spread_points,
        spec.csv_measured_spread_points,
        spec.typical_spread_points,
        spec.spread_points,
        spec.commission_per_lot,
    )

    stop_bricks = 1.0 if symbol.upper() == "XAUUSD" else 0.5
    brick_size = float(dsp.get("brick_size", 1.0))

    # Load M1 data to estimate brick frequency
    closes = load_m1_data(symbol)
    if closes is not None and len(closes) > 1000:
        # Build sample bricks to measure frequency
        bricks = build_renko(closes.tail(min(10000, len(closes))), brick_size)
        if len(bricks) > 10:
            bpd = bricks_per_day(bricks)
            # IMPORTANT: use the actual source cadence from DSP profile.
            # In this runner we compute DSP on M1, so bars_per_hour is 60.
            vr_peak_scale = int(dsp.get("vr_peak_scale", 50))
            source_bph = float(dsp.get("bars_per_hour", 60.0))
            # Backward-compat: older profiles were generated on M1 but left at
            # default bars_per_hour=2.0, which inflates warmup windows ~30x.
            if source_bph <= 2.1:
                LOG.warning(
                    "Legacy DSP profile cadence detected (bars_per_hour=%.3f). "
                    "Normalizing to M1 cadence (60.0 bars/hour). Re-run --stage dsp to refresh profile.",
                    source_bph,
                )
                source_bph = 60.0
            source_bars_per_day = max(1.0, source_bph * 24.0)
            days_in_peak = vr_peak_scale / source_bars_per_day
            # Guardrails: keep filter windows in a practical range.
            window = max(10, min(100, int(round(bpd * days_in_peak))))
        else:
            window = 50  # Fallback
    else:
        window = 50  # Fallback

    # Spread: use spec.spread_usd_per_lot which already resolves through the
    # priority chain: csv_measured_spread_points (set by poll_symbol_specs from
    # the bar CSV's <SPREAD> column) → typical_spread_points → spread_points →
    # bid/ask snapshot.  Convert back to ticks for the engine's arithmetic.
    spread_ticks = (
        spec.spread_usd_per_lot / spec.tick_value_usd
        if spec.tick_value_usd > 0
        else spec.spread_points
    )
    _spread_source = (
        "csv"
        if spec.csv_measured_spread_points > 0
        else "typical"
        if spec.typical_spread_points > 0
        else "spec"
    )
    LOG.info(
        "%s spread: %.1f points ($%.4f/lot) [source: %s]",
        symbol,
        spread_ticks,
        spread_ticks * spec.tick_value_usd,
        _spread_source,
    )

    # Swap: mode 0 (pips) → direct USD/day; mode 1 (% p.a.) → approximate using
    # poll-time mid price (bid/ask average or bid). Negative = cost, positive = credit.
    import math

    _swap_long_usd = spec.swap_long_usd_per_day
    _swap_short_usd = spec.swap_short_usd_per_day
    if math.isnan(_swap_long_usd):
        # Mode 1 (% p.a.): convert using poll-time price as proxy
        _ref_price = (spec.bid + spec.ask) / 2.0 if spec.ask > spec.bid > 0 else spec.bid
        if _ref_price > 0:
            _swap_long_usd = (
                _ref_price
                * spec.contract_size
                * spec.quote_usd_rate
                * (spec.swap_long_points / 100.0)
                / 365.0
            )
            _swap_short_usd = (
                _ref_price
                * spec.contract_size
                * spec.quote_usd_rate
                * (spec.swap_short_points / 100.0)
                / 365.0
            )
        else:
            _swap_long_usd = _swap_short_usd = 0.0

    LOG.info(
        "%s swap: long=$%.4f/day  short=$%.4f/day  triple_day=%d",
        symbol,
        _swap_long_usd,
        _swap_short_usd,
        spec.triple_swap_day,
    )

    return (
        EngineConfig(
            symbol=symbol,
            brick_size=brick_size,
            usd_per_tick=spec.tick_value_usd,  # tick_size × contract_size × quote_usd_rate
            tick_size=spec.tick_size,
            stop_bricks=stop_bricks,
            fliprate_window=window,
            markov_window=window,
            fliprate_threshold=0.35,
            markov_threshold=0.55,
            spread_ticks=spread_ticks,  # csv_measured → typical → spec → snapshot
            commission_per_lot=spec.commission_per_lot * 2,  # per-side × 2 = round-trip
            swap_long_usd_per_day=_swap_long_usd,
            swap_short_usd_per_day=_swap_short_usd,
            triple_swap_day=spec.triple_swap_day,
            margin_rate=float(spec.margin_initial) if float(spec.margin_initial) > 0 else 0.01,
            sizing_mode=sizing_mode,
            gate_lot_ceiling=lot_ceiling,
        ),
        spec,
    )


# ──────────────────────────────────────────────────────────────────────────────
# STAGE 3: BACKTEST
# ──────────────────────────────────────────────────────────────────────────────


def stage_backtest(
    symbol: str,
    months: int = 3,
    min_omega: float = 1.5,
    min_trades: int = 30,
    auto_download: bool = False,
    stop_bricks_override: Optional[float] = None,
    target_risk_override: Optional[float] = None,
    brick_size_override: Optional[float] = None,
    fliprate_window_override: Optional[int] = None,
    markov_window_override: Optional[int] = None,
    fliprate_threshold_override: Optional[float] = None,
    markov_threshold_override: Optional[float] = None,
    max_leverage_override: Optional[float] = None,
    trailing_mfe_fraction_override: Optional[float] = None,
    trailing_mfe_after_bricks_override: Optional[int] = None,
    entry_latency_ms: int = 250,
    exit_latency_ms: int = 250,
    latency_jitter_ms: int = 10,
    latency_seed: int = 42,
    lot_ceiling_override: float = 50.0,
    conservative_fills: bool = True,
    spread_sides: float = 1.0,
    entry_slip_bricks: float = 0.0,
    exit_slip_bricks: float = 0.10,
    stop_worst_case_bricks: float = 0.25,
    flip_trade_through_bricks: float = 0.10,
    use_pe_gate_override: Optional[bool] = None,
    pe_window_override: Optional[int] = None,
    pe_order_override: Optional[int] = None,
    pe_entry_threshold_override: Optional[float] = None,
    pe_exit_threshold_override: Optional[float] = None,
    use_time_decay_override: Optional[bool] = None,
    stale_brick_factor_override: Optional[float] = None,
    markov_stale_penalty_override: Optional[float] = None,
    use_obi_slippage_buffer_override: Optional[bool] = None,
    obi_levels_override: Optional[int] = None,
    obi_max_buffer_bricks_override: Optional[float] = None,
    obi_ema_alpha_override: Optional[float] = None,
) -> bool:
    """Backtest the last N months of data."""
    run_started_utc = datetime.now(timezone.utc)
    run_id = run_started_utc.strftime("%Y%m%dT%H%M%S_%fZ")

    click.echo(f"\n{'=' * 60}")
    click.secho(f"STAGE 3: BACKTEST ({months} months)", fg="cyan", bold=True)
    click.echo(f"{'=' * 60}")

    if not _ensure_backtest_history(symbol, months, auto_download=auto_download):
        click.secho(f"❌ Unable to prepare required M1 history for {symbol}", fg="red")
        return False

    # Load all M1 data (close + spread for dynamic spread costing)
    market_df = load_m1_market_data(symbol)
    if market_df is None or market_df.empty:
        click.secho(f"❌ No M1 data found for {symbol}", fg="red")
        return False
    closes = market_df["close"]
    spread_series = market_df["spread"] if "spread" in market_df.columns else None

    # Filter to last N months
    cutoff = closes.index[-1] - pd.DateOffset(months=months)
    test_closes = closes[closes.index >= cutoff]

    click.echo(f"Testing {len(test_closes):,} bars ({months} months)")
    click.echo(f"Range: {test_closes.index[0]} to {test_closes.index[-1]}")

    # Load DSP profile
    dsp, dsp_file = _load_or_build_dsp_profile(symbol, closes=closes)
    if dsp is None:
        click.secho(f"❌ No DSP profile found at {dsp_file}", fg="red")
        return False

    # Build config for both scenarios
    cfg_sample, spec = _build_engine_config(
        symbol, dsp, sizing_mode="static", lot_ceiling=float(lot_ceiling_override)
    )
    _apply_strategy_overrides(
        cfg_sample,
        stop_bricks=stop_bricks_override,
        trailing_mfe_fraction=trailing_mfe_fraction_override,
        trailing_mfe_after_bricks=trailing_mfe_after_bricks_override,
        target_risk=target_risk_override,
        brick_size=brick_size_override,
        fliprate_window=fliprate_window_override,
        markov_window=markov_window_override,
        fliprate_threshold=fliprate_threshold_override,
        markov_threshold=markov_threshold_override,
        max_leverage=max_leverage_override,
        spread_sides=spread_sides,
        conservative_fills=conservative_fills,
        entry_slip_bricks=entry_slip_bricks,
        exit_slip_bricks=exit_slip_bricks,
        stop_worst_case_bricks=stop_worst_case_bricks,
        flip_trade_through_bricks=flip_trade_through_bricks,
        use_pe_gate=use_pe_gate_override,
        pe_window=pe_window_override,
        pe_order=pe_order_override,
        pe_entry_threshold=pe_entry_threshold_override,
        pe_exit_threshold=pe_exit_threshold_override,
        use_time_decay=use_time_decay_override,
        stale_brick_factor=stale_brick_factor_override,
        markov_stale_penalty=markov_stale_penalty_override,
        use_obi_slippage_buffer=use_obi_slippage_buffer_override,
        obi_levels=obi_levels_override,
        obi_max_buffer_bricks=obi_max_buffer_bricks_override,
        obi_ema_alpha=obi_ema_alpha_override,
    )

    # Print system specification BEFORE backtest
    click.echo()
    _print_system_spec(
        cfg_sample,
        spec,
        dsp,
        entry_latency_ms=entry_latency_ms,
        exit_latency_ms=exit_latency_ms,
        latency_jitter_ms=latency_jitter_ms,
        latency_seed=latency_seed,
    )

    # Run backtest with both sizing scenarios for XAUUSD
    scenarios = ["static", "compounding"] if symbol.upper() == "XAUUSD" else ["risk_based"]
    all_pass = True
    scenario_reports: dict = {}
    original_log_level = LOG.level
    engine_log = logging.getLogger("kinetra.renko.trading_engine")
    original_engine_log_level = engine_log.level
    LOG.setLevel(logging.WARNING)
    engine_log.setLevel(logging.ERROR)
    try:
        for scenario in scenarios:
            click.echo(f"--- {scenario.upper()} SIZING ---")

            # Static uses fixed min lot; non-static scenarios respect user lot cap.
            lot_ceiling = 0.01 if scenario == "static" else float(lot_ceiling_override)

            cfg, _ = _build_engine_config(
                symbol, dsp, sizing_mode=scenario, lot_ceiling=lot_ceiling
            )
            _apply_strategy_overrides(
                cfg,
                stop_bricks=stop_bricks_override,
                trailing_mfe_fraction=trailing_mfe_fraction_override,
                trailing_mfe_after_bricks=trailing_mfe_after_bricks_override,
                target_risk=target_risk_override,
                brick_size=brick_size_override,
                fliprate_window=fliprate_window_override,
                markov_window=markov_window_override,
                fliprate_threshold=fliprate_threshold_override,
                markov_threshold=markov_threshold_override,
                max_leverage=max_leverage_override,
                spread_sides=spread_sides,
                conservative_fills=conservative_fills,
                entry_slip_bricks=entry_slip_bricks,
                exit_slip_bricks=exit_slip_bricks,
                stop_worst_case_bricks=stop_worst_case_bricks,
                flip_trade_through_bricks=flip_trade_through_bricks,
                use_pe_gate=use_pe_gate_override,
                pe_window=pe_window_override,
                pe_order=pe_order_override,
                pe_entry_threshold=pe_entry_threshold_override,
                pe_exit_threshold=pe_exit_threshold_override,
                use_time_decay=use_time_decay_override,
                stale_brick_factor=stale_brick_factor_override,
                markov_stale_penalty=markov_stale_penalty_override,
                use_obi_slippage_buffer=use_obi_slippage_buffer_override,
                obi_levels=obi_levels_override,
                obi_max_buffer_bricks=obi_max_buffer_bricks_override,
                obi_ema_alpha=obi_ema_alpha_override,
            )
            engine = RenkoEngine(cfg)
            engine.set_dynamic_spread_series(spread_series)
            results = engine.backtest(test_closes)
            latency_meta = _apply_latency_slippage_to_completed_trades(
                completed_trades=engine._completed,
                brick_size=float(cfg.brick_size),
                usd_per_point=float(engine._usd_per_point),
                entry_latency_ms=int(entry_latency_ms),
                exit_latency_ms=int(exit_latency_ms),
                latency_jitter_ms=int(latency_jitter_ms),
                latency_seed=int(latency_seed),
            )
            if latency_meta.get("enabled", False):
                results = engine._make_results()
                LOG.warning(
                    "[BACKTEST LATENCY] scenario=%s adjusted=%d entry_ms=%d exit_ms=%d jitter_ms=%d avg_entry_slip=%.5f avg_exit_slip=%.5f",
                    scenario,
                    int(latency_meta.get("trades_adjusted", 0)),
                    int(latency_meta.get("entry_latency_ms", 0)),
                    int(latency_meta.get("exit_latency_ms", 0)),
                    int(latency_meta.get("latency_jitter_ms", 0)),
                    float(latency_meta.get("avg_entry_slip", 0.0)),
                    float(latency_meta.get("avg_exit_slip", 0.0)),
                )

            if "error" in results:
                click.secho(f"❌ Error: {results['error']}", fg="red")
                scenario_reports[scenario] = {
                    "status": "ERROR",
                    "error": str(results["error"]),
                    "lot_ceiling": lot_ceiling,
                    "engine_config": asdict(cfg),
                }
                all_pass = False
                continue

            summary = results.get("summary", {})
            trades = results.get("trades", [])
            audit_path = _persist_execution_audit(
                symbol=symbol,
                mode=f"backtest_{scenario}",
                results=results,
                metadata={
                    "months": int(months),
                    "scenario": scenario,
                    "lot_ceiling": lot_ceiling,
                    "engine_config": asdict(cfg),
                },
            )
            equity_plot = _save_equity_plot(
                symbol=symbol,
                mode=f"backtest_{scenario}",
                trades=trades,
                initial_equity=float(cfg.initial_equity),
                run_id=run_id,
            )
            lev_cap = (1.0 / float(cfg.margin_rate)) if float(cfg.margin_rate) > 0 else None
            leverage_plot = _save_leverage_plot(
                symbol=symbol,
                mode=f"backtest_{scenario}",
                trades=trades,
                initial_equity=float(cfg.initial_equity),
                usd_per_point=float(engine._usd_per_point),
                max_leverage=lev_cap,
                run_id=run_id,
            )

            _print_stats(summary, symbol, f"backtest-{scenario}", trades=trades)
            _print_spread_sanity_samples(
                symbol=symbol,
                mode=f"backtest-{scenario}",
                trades=trades,
                cfg=cfg,
                sample_n=20,
                seed=42,
            )

            n_trades = int(summary.get("n_trades", 0))
            omega = float(summary.get("omega", 0.0))
            passes = n_trades >= min_trades and omega >= min_omega

            scenario_reports[scenario] = {
                "status": "PASS" if passes else "FAIL",
                "pass_gates": bool(passes),
                "lot_ceiling": lot_ceiling,
                "engine_config": asdict(cfg),
                "latency": latency_meta,
                "summary": summary,
                "audit_path": str(audit_path),
                "equity_plot_path": str(equity_plot) if equity_plot is not None else None,
                "leverage_plot_path": str(leverage_plot) if leverage_plot is not None else None,
            }

            if passes:
                click.secho("✅ PASS", fg="green")
            else:
                click.secho(
                    f"❌ FAIL - trades={n_trades} (need {min_trades}), omega={omega:.2f} (need {min_omega})",
                    fg="red",
                )
                all_pass = False
    finally:
        LOG.setLevel(original_log_level)
        engine_log.setLevel(original_engine_log_level)

    m1_source = _find_m1_source_file(symbol)
    m1_source_meta = None
    if m1_source is not None:
        st = m1_source.stat()
        m1_source_meta = {
            "path": str(m1_source),
            "size_bytes": int(st.st_size),
            "modified_utc": datetime.fromtimestamp(st.st_mtime, tz=timezone.utc).isoformat(),
        }

    report = {
        "symbol": symbol,
        "stage": "backtest",
        "run_id": run_id,
        "run_started_utc": run_started_utc.isoformat(),
        "run_finished_utc": datetime.now(timezone.utc).isoformat(),
        "command": " ".join(
            [
                "python",
                "scripts/renko_engine.py",
                symbol,
                "--stage",
                "backtest",
                "--months",
                str(months),
                "--min-omega",
                str(min_omega),
                "--min-trades",
                str(min_trades),
                *(["--auto-download"] if auto_download else []),
                *(
                    ["--stop-bricks", str(stop_bricks_override)]
                    if stop_bricks_override is not None
                    else []
                ),
                *(
                    ["--trailing-mfe-fraction", str(trailing_mfe_fraction_override)]
                    if trailing_mfe_fraction_override is not None
                    else []
                ),
                *(
                    ["--trailing-mfe-after-bricks", str(trailing_mfe_after_bricks_override)]
                    if trailing_mfe_after_bricks_override is not None
                    else []
                ),
                *(
                    ["--target-risk", str(target_risk_override)]
                    if target_risk_override is not None
                    else []
                ),
                *(
                    ["--brick-size", str(brick_size_override)]
                    if brick_size_override is not None
                    else []
                ),
                *(
                    ["--fliprate-window", str(fliprate_window_override)]
                    if fliprate_window_override is not None
                    else []
                ),
                *(
                    ["--markov-window", str(markov_window_override)]
                    if markov_window_override is not None
                    else []
                ),
                *(
                    ["--fliprate-threshold", str(fliprate_threshold_override)]
                    if fliprate_threshold_override is not None
                    else []
                ),
                *(
                    ["--markov-threshold", str(markov_threshold_override)]
                    if markov_threshold_override is not None
                    else []
                ),
                *(
                    ["--max-leverage", str(max_leverage_override)]
                    if max_leverage_override is not None
                    else []
                ),
                "--entry-latency-ms",
                str(int(entry_latency_ms)),
                "--exit-latency-ms",
                str(int(exit_latency_ms)),
                "--latency-jitter-ms",
                str(int(latency_jitter_ms)),
                "--latency-seed",
                str(int(latency_seed)),
            ]
        ),
        "inputs": {
            "months": int(months),
            "min_omega": float(min_omega),
            "min_trades": int(min_trades),
            "auto_download": bool(auto_download),
            "latency": {
                "entry_latency_ms": int(entry_latency_ms),
                "exit_latency_ms": int(exit_latency_ms),
                "latency_jitter_ms": int(latency_jitter_ms),
                "latency_seed": int(latency_seed),
            },
            "execution_model": {
                "conservative_fills": bool(conservative_fills),
                "spread_sides": float(spread_sides),
                "entry_slip_bricks": float(entry_slip_bricks),
                "exit_slip_bricks": float(exit_slip_bricks),
                "stop_worst_case_bricks": float(stop_worst_case_bricks),
                "flip_trade_through_bricks": float(flip_trade_through_bricks),
            },
            "overrides": {
                "stop_bricks": stop_bricks_override,
                "trailing_mfe_fraction": trailing_mfe_fraction_override,
                "trailing_mfe_after_bricks": trailing_mfe_after_bricks_override,
                "target_risk": target_risk_override,
                "brick_size": brick_size_override,
                "fliprate_window": fliprate_window_override,
                "markov_window": markov_window_override,
                "fliprate_threshold": fliprate_threshold_override,
                "markov_threshold": markov_threshold_override,
                "max_leverage": max_leverage_override,
            },
        },
        "data_window": {
            "bars_tested": int(len(test_closes)),
            "range_start_utc": test_closes.index[0].isoformat(),
            "range_end_utc": test_closes.index[-1].isoformat(),
        },
        "data_source": m1_source_meta,
        "dsp_profile": {
            "path": str(dsp_file),
            "values": dsp,
        },
        "instrument_spec": spec.to_dict(),
        "scenario_order": scenarios,
        "scenario_results": scenario_reports,
        "overall_status": "PASS" if all_pass else "FAIL",
    }

    out_dir = KR / "outputs" / "renko_results" / "backtests" / symbol.upper()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_stem = f"{symbol.lower()}_backtest_{months}m_{run_id}"
    out_path = _choose_unique_path(out_dir, out_stem)
    with open(out_path, "x", encoding="utf-8") as f:
        json.dump(_json_safe(report), f, indent=2)
    click.secho(f"Saved backtest report: {out_path}", fg="cyan")

    return all_pass


# ──────────────────────────────────────────────────────────────────────────────
# STAGE 3B: FULL BACKTEST (ROLLING OOS)
# ──────────────────────────────────────────────────────────────────────────────


def stage_full_rolling_oos(
    symbol: str,
    months: int = 24,
    oos_window_months: int = 3,
    wf_is_months: int = 3,
    wf_trials: int = 16,
    wf_seed: int = 42,
    min_omega: float = 1.5,
    min_trades: int = 30,
    auto_download: bool = False,
    stop_bricks_override: Optional[float] = None,
    trailing_mfe_fraction_override: Optional[float] = None,
    trailing_mfe_after_bricks_override: Optional[int] = None,
    target_risk_override: Optional[float] = None,
    brick_size_override: Optional[float] = None,
    fliprate_window_override: Optional[int] = None,
    markov_window_override: Optional[int] = None,
    fliprate_threshold_override: Optional[float] = None,
    markov_threshold_override: Optional[float] = None,
    max_leverage_override: Optional[float] = None,
    entry_latency_ms: int = 250,
    exit_latency_ms: int = 250,
    latency_jitter_ms: int = 10,
    latency_seed: int = 42,
    lot_ceiling_override: float = 50.0,
    conservative_fills: bool = True,
    spread_sides: float = 1.0,
    entry_slip_bricks: float = 0.0,
    exit_slip_bricks: float = 0.10,
    stop_worst_case_bricks: float = 0.25,
    flip_trade_through_bricks: float = 0.10,
    use_pe_gate_override: Optional[bool] = None,
    pe_window_override: Optional[int] = None,
    pe_order_override: Optional[int] = None,
    pe_entry_threshold_override: Optional[float] = None,
    pe_exit_threshold_override: Optional[float] = None,
    use_time_decay_override: Optional[bool] = None,
    stale_brick_factor_override: Optional[float] = None,
    markov_stale_penalty_override: Optional[float] = None,
    use_obi_slippage_buffer_override: Optional[bool] = None,
    obi_levels_override: Optional[int] = None,
    obi_max_buffer_bricks_override: Optional[float] = None,
    obi_ema_alpha_override: Optional[float] = None,
) -> bool:
    """Walk-forward full backtest with IS fit + OOS evaluation per fold."""
    if months <= 0 or oos_window_months <= 0 or wf_is_months <= 0:
        click.secho("❌ months, oos_window_months and wf_is_months must be > 0", fg="red")
        return False

    run_started_utc = datetime.now(timezone.utc)
    run_id = run_started_utc.strftime("%Y%m%dT%H%M%S_%fZ")

    click.echo(f"\n{'=' * 60}")
    click.secho(
        (
            "STAGE 3B: FULL BACKTEST "
            f"(WALK-FORWARD, {months}m total, IS={wf_is_months}m, OOS={oos_window_months}m, trials={wf_trials})"
        ),
        fg="cyan",
        bold=True,
    )
    click.echo(f"{'=' * 60}")

    if not _ensure_backtest_history(symbol, months, auto_download=auto_download):
        click.secho(f"❌ Unable to prepare required M1 history for {symbol}", fg="red")
        return False

    market_df = load_m1_market_data(symbol)
    if market_df is None or market_df.empty:
        click.secho(f"❌ No M1 data found for {symbol}", fg="red")
        return False
    closes = market_df["close"]
    spread_series = market_df["spread"] if "spread" in market_df.columns else None

    cutoff = closes.index[-1] - pd.DateOffset(months=months)
    test_closes = closes[closes.index >= cutoff]
    if test_closes.empty:
        click.secho("❌ Empty test window after history filter", fg="red")
        return False

    click.echo(f"Testing {len(test_closes):,} bars ({months} months total)")
    click.echo(f"Range: {test_closes.index[0]} to {test_closes.index[-1]}")

    dsp, dsp_file = _load_or_build_dsp_profile(symbol, closes=closes)
    if dsp is None:
        click.secho(f"❌ No DSP profile found at {dsp_file}", fg="red")
        return False

    cfg_sample, spec = _build_engine_config(
        symbol, dsp, sizing_mode="static", lot_ceiling=float(lot_ceiling_override)
    )
    _apply_strategy_overrides(
        cfg_sample,
        stop_bricks=stop_bricks_override,
        trailing_mfe_fraction=trailing_mfe_fraction_override,
        trailing_mfe_after_bricks=trailing_mfe_after_bricks_override,
        target_risk=target_risk_override,
        brick_size=brick_size_override,
        fliprate_window=fliprate_window_override,
        markov_window=markov_window_override,
        fliprate_threshold=fliprate_threshold_override,
        markov_threshold=markov_threshold_override,
        max_leverage=max_leverage_override,
        spread_sides=spread_sides,
        conservative_fills=conservative_fills,
        entry_slip_bricks=entry_slip_bricks,
        exit_slip_bricks=exit_slip_bricks,
        stop_worst_case_bricks=stop_worst_case_bricks,
        flip_trade_through_bricks=flip_trade_through_bricks,
        use_pe_gate=use_pe_gate_override,
        pe_window=pe_window_override,
        pe_order=pe_order_override,
        pe_entry_threshold=pe_entry_threshold_override,
        pe_exit_threshold=pe_exit_threshold_override,
        use_time_decay=use_time_decay_override,
        stale_brick_factor=stale_brick_factor_override,
        markov_stale_penalty=markov_stale_penalty_override,
        use_obi_slippage_buffer=use_obi_slippage_buffer_override,
        obi_levels=obi_levels_override,
        obi_max_buffer_bricks=obi_max_buffer_bricks_override,
        obi_ema_alpha=obi_ema_alpha_override,
    )
    click.echo()
    _print_system_spec(
        cfg_sample,
        spec,
        dsp,
        entry_latency_ms=entry_latency_ms,
        exit_latency_ms=exit_latency_ms,
        latency_jitter_ms=latency_jitter_ms,
        latency_seed=latency_seed,
    )

    # Build sequential OOS folds inside [test_start, test_end].
    folds: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    fold_start = pd.Timestamp(test_closes.index[0])
    test_end = pd.Timestamp(test_closes.index[-1])
    while fold_start < test_end:
        fold_end = min(fold_start + pd.DateOffset(months=oos_window_months), test_end)
        if fold_end <= fold_start:
            break
        folds.append((fold_start, fold_end))
        fold_start = fold_end
    if not folds:
        click.secho("❌ Failed to construct OOS folds", fg="red")
        return False

    scenarios = ["static", "compounding"] if symbol.upper() == "XAUUSD" else ["risk_based"]
    scenario_reports: dict = {}
    all_pass = True

    def _summary_score(summary: dict) -> float:
        n_trades = int(summary.get("n_trades", 0))
        omega = float(summary.get("omega", 0.0))
        net_usd = float(summary.get("net_usd", 0.0))
        dd_pct = abs(float(summary.get("max_drawdown_pct", 0.0))) / 100.0
        trade_term = min(n_trades, 250) / 250.0
        net_term = _clamp(net_usd / 3000.0, -2.0, 2.0)
        dd_penalty = max(0.0, dd_pct - 0.12) * 3.5
        low_trade_penalty = max(0, 25 - n_trades) * 0.08
        return float(omega + trade_term + net_term - dd_penalty - low_trade_penalty)

    def _fit_fold_params(base_cfg: EngineConfig, is_series: pd.Series, seed: int) -> dict:
        rng = np.random.default_rng(int(seed))
        if wf_trials <= 0:
            return {
                "brick_size": float(base_cfg.brick_size),
                "stop_bricks": float(base_cfg.stop_bricks),
                "target_risk_usd": float(base_cfg.target_risk_usd),
                "fliprate_threshold": float(base_cfg.fliprate_threshold),
                "markov_threshold": float(base_cfg.markov_threshold),
                "trailing_mfe_fraction": float(base_cfg.trailing_mfe_fraction),
                "trailing_mfe_after_bricks": int(base_cfg.trailing_mfe_after_bricks),
                "score": float("-inf"),
            }

        # Sane local bounds around current config.
        b0 = float(base_cfg.brick_size)
        s0 = float(base_cfg.stop_bricks)
        r0 = float(base_cfg.target_risk_usd)
        f0 = float(base_cfg.fliprate_threshold)
        m0 = float(base_cfg.markov_threshold)
        tf0 = float(base_cfg.trailing_mfe_fraction)
        ta0 = int(base_cfg.trailing_mfe_after_bricks)
        best = None
        best_score = -float("inf")

        for _ in range(int(wf_trials)):
            cand = {
                "brick_size": float(_clamp(rng.uniform(b0 * 0.7, b0 * 1.6), 0.05, 9999.0)),
                "stop_bricks": float(_clamp(rng.uniform(s0 * 0.6, s0 * 1.8), 0.2, 5.0)),
                "target_risk_usd": float(_clamp(rng.uniform(r0 * 0.5, r0 * 1.8), 5.0, 10000.0)),
                "fliprate_threshold": float(_clamp(rng.uniform(f0 - 0.12, f0 + 0.12), 0.15, 0.65)),
                "markov_threshold": float(_clamp(rng.uniform(m0 - 0.12, m0 + 0.12), 0.40, 0.90)),
                "trailing_mfe_fraction": float(
                    _clamp(rng.uniform(max(0.2, tf0 - 0.2), min(0.9, tf0 + 0.2)), 0.2, 0.9)
                ),
                "trailing_mfe_after_bricks": int(
                    _clamp(round(rng.uniform(max(1, ta0 - 2), max(2, ta0 + 3))), 1, 12)
                ),
            }
            cfg_fit = EngineConfig(**asdict(base_cfg))
            cfg_fit.brick_size = cand["brick_size"]
            cfg_fit.stop_bricks = cand["stop_bricks"]
            cfg_fit.target_risk_usd = cand["target_risk_usd"]
            cfg_fit.fliprate_threshold = cand["fliprate_threshold"]
            cfg_fit.markov_threshold = cand["markov_threshold"]
            cfg_fit.trailing_mfe_fraction = cand["trailing_mfe_fraction"]
            cfg_fit.trailing_mfe_after_bricks = cand["trailing_mfe_after_bricks"]

            engine_fit = RenkoEngine(cfg_fit, quiet_mode=True)
            fit_res = engine_fit.backtest(is_series)
            if "error" in fit_res:
                continue
            fit_score = _summary_score(fit_res.get("summary", {}))
            if fit_score > best_score:
                best_score = fit_score
                best = cand

        if best is None:
            best = {
                "brick_size": float(base_cfg.brick_size),
                "stop_bricks": float(base_cfg.stop_bricks),
                "target_risk_usd": float(base_cfg.target_risk_usd),
                "fliprate_threshold": float(base_cfg.fliprate_threshold),
                "markov_threshold": float(base_cfg.markov_threshold),
                "trailing_mfe_fraction": float(base_cfg.trailing_mfe_fraction),
                "trailing_mfe_after_bricks": int(base_cfg.trailing_mfe_after_bricks),
            }
        best["score"] = float(best_score)
        return best

    def _delta_block(is_s: dict, oos_s: dict) -> dict:
        return {
            "omega_delta": float(oos_s.get("omega", 0.0)) - float(is_s.get("omega", 0.0)),
            "net_usd_delta": float(oos_s.get("net_usd", 0.0)) - float(is_s.get("net_usd", 0.0)),
            "win_rate_delta": float(oos_s.get("win_rate", 0.0)) - float(is_s.get("win_rate", 0.0)),
            "n_trades_delta": int(oos_s.get("n_trades", 0)) - int(is_s.get("n_trades", 0)),
            "dd_pct_delta": float(oos_s.get("max_drawdown_pct", 0.0))
            - float(is_s.get("max_drawdown_pct", 0.0)),
        }

    original_log_level = LOG.level
    engine_log = logging.getLogger("kinetra.renko.trading_engine")
    original_engine_log_level = engine_log.level
    LOG.setLevel(logging.WARNING)
    engine_log.setLevel(logging.ERROR)
    try:
        for scenario in scenarios:
            click.echo(f"--- {scenario.upper()} SIZING (rolling OOS) ---")
            lot_ceiling = 0.01 if scenario == "static" else float(lot_ceiling_override)
            cfg, _ = _build_engine_config(
                symbol, dsp, sizing_mode=scenario, lot_ceiling=lot_ceiling
            )
            _apply_strategy_overrides(
                cfg,
                stop_bricks=stop_bricks_override,
                trailing_mfe_fraction=trailing_mfe_fraction_override,
                trailing_mfe_after_bricks=trailing_mfe_after_bricks_override,
                target_risk=target_risk_override,
                brick_size=brick_size_override,
                fliprate_window=fliprate_window_override,
                markov_window=markov_window_override,
                fliprate_threshold=fliprate_threshold_override,
                markov_threshold=markov_threshold_override,
                max_leverage=max_leverage_override,
                spread_sides=spread_sides,
                conservative_fills=conservative_fills,
                entry_slip_bricks=entry_slip_bricks,
                exit_slip_bricks=exit_slip_bricks,
                stop_worst_case_bricks=stop_worst_case_bricks,
                flip_trade_through_bricks=flip_trade_through_bricks,
                use_pe_gate=use_pe_gate_override,
                pe_window=pe_window_override,
                pe_order=pe_order_override,
                pe_entry_threshold=pe_entry_threshold_override,
                pe_exit_threshold=pe_exit_threshold_override,
                use_time_decay=use_time_decay_override,
                stale_brick_factor=stale_brick_factor_override,
                markov_stale_penalty=markov_stale_penalty_override,
                use_obi_slippage_buffer=use_obi_slippage_buffer_override,
                obi_levels=obi_levels_override,
                obi_max_buffer_bricks=obi_max_buffer_bricks_override,
                obi_ema_alpha=obi_ema_alpha_override,
            )

            all_completed_oos = []
            all_completed_is = []
            fold_reports: list[dict] = []
            for i, (fstart, fend) in enumerate(folds, start=1):
                is_start = fstart - pd.DateOffset(months=wf_is_months)
                is_series = closes[(closes.index >= is_start) & (closes.index < fstart)]
                if i < len(folds):
                    oos_series = test_closes[
                        (test_closes.index >= fstart) & (test_closes.index < fend)
                    ]
                else:
                    oos_series = test_closes[
                        (test_closes.index >= fstart) & (test_closes.index <= fend)
                    ]
                if is_series.empty or oos_series.empty:
                    continue

                best = _fit_fold_params(cfg, is_series, seed=wf_seed + i)
                cfg_fold = EngineConfig(**asdict(cfg))
                cfg_fold.brick_size = float(best["brick_size"])
                cfg_fold.stop_bricks = float(best["stop_bricks"])
                cfg_fold.target_risk_usd = float(best["target_risk_usd"])
                cfg_fold.fliprate_threshold = float(best["fliprate_threshold"])
                cfg_fold.markov_threshold = float(best["markov_threshold"])
                cfg_fold.trailing_mfe_fraction = float(best["trailing_mfe_fraction"])
                cfg_fold.trailing_mfe_after_bricks = int(best["trailing_mfe_after_bricks"])

                engine_is = RenkoEngine(cfg_fold, quiet_mode=True)
                engine_is.set_dynamic_spread_series(spread_series)
                is_results = engine_is.backtest(is_series)
                if "error" in is_results:
                    fold_reports.append(
                        {
                            "fold": i,
                            "range_start_utc": fstart.isoformat(),
                            "range_end_utc": fend.isoformat(),
                            "status": "ERROR",
                            "error": f"is_error: {is_results['error']}",
                        }
                    )
                    continue
                is_summary = is_results.get("summary", {})
                all_completed_is.extend(engine_is._completed)

                engine_oos = RenkoEngine(cfg_fold, quiet_mode=True)
                engine_oos.set_dynamic_spread_series(spread_series)
                oos_results = engine_oos.backtest(oos_series)
                latency_meta = _apply_latency_slippage_to_completed_trades(
                    completed_trades=engine_oos._completed,
                    brick_size=float(cfg_fold.brick_size),
                    usd_per_point=float(engine_oos._usd_per_point),
                    entry_latency_ms=int(entry_latency_ms),
                    exit_latency_ms=int(exit_latency_ms),
                    latency_jitter_ms=int(latency_jitter_ms),
                    latency_seed=int(latency_seed),
                )
                if latency_meta.get("enabled", False):
                    oos_results = engine_oos._make_results()

                if "error" in oos_results:
                    fold_reports.append(
                        {
                            "fold": i,
                            "range_start_utc": fstart.isoformat(),
                            "range_end_utc": fend.isoformat(),
                            "status": "ERROR",
                            "error": f"oos_error: {oos_results['error']}",
                        }
                    )
                    continue

                oos_summary = oos_results.get("summary", {})
                all_completed_oos.extend(engine_oos._completed)
                deltas = _delta_block(is_summary, oos_summary)
                fold_reports.append(
                    {
                        "fold": i,
                        "is_start_utc": is_start.isoformat(),
                        "is_end_utc": fstart.isoformat(),
                        "range_start_utc": fstart.isoformat(),
                        "range_end_utc": fend.isoformat(),
                        "fit_score": float(best.get("score", 0.0)),
                        "fitted_params": {
                            "brick_size": cfg_fold.brick_size,
                            "stop_bricks": cfg_fold.stop_bricks,
                            "target_risk_usd": cfg_fold.target_risk_usd,
                            "fliprate_threshold": cfg_fold.fliprate_threshold,
                            "markov_threshold": cfg_fold.markov_threshold,
                            "trailing_mfe_fraction": cfg_fold.trailing_mfe_fraction,
                            "trailing_mfe_after_bricks": cfg_fold.trailing_mfe_after_bricks,
                        },
                        "is_summary": {
                            "n_trades": int(is_summary.get("n_trades", 0)),
                            "omega": float(is_summary.get("omega", 0.0)),
                            "net_usd": float(is_summary.get("net_usd", 0.0)),
                            "win_rate": float(is_summary.get("win_rate", 0.0)),
                            "max_drawdown_pct": float(is_summary.get("max_drawdown_pct", 0.0)),
                        },
                        "oos_summary": {
                            "n_trades": int(oos_summary.get("n_trades", 0)),
                            "omega": float(oos_summary.get("omega", 0.0)),
                            "net_usd": float(oos_summary.get("net_usd", 0.0)),
                            "win_rate": float(oos_summary.get("win_rate", 0.0)),
                            "max_drawdown_pct": float(oos_summary.get("max_drawdown_pct", 0.0)),
                        },
                        "delta_oos_minus_is": deltas,
                        "latency": latency_meta,
                    }
                )

            if not all_completed_oos:
                scenario_reports[scenario] = {
                    "status": "ERROR",
                    "error": "no_trades_across_folds",
                    "lot_ceiling": lot_ceiling,
                    "engine_config": asdict(cfg),
                    "folds": fold_reports,
                }
                all_pass = False
                continue

            agg_is_engine = RenkoEngine(cfg, quiet_mode=True)
            agg_is_engine._completed = all_completed_is
            agg_is_results = agg_is_engine._make_results()
            agg_is_summary = agg_is_results.get("summary", {})

            agg_oos_engine = RenkoEngine(cfg, quiet_mode=True)
            agg_oos_engine._completed = all_completed_oos
            agg_oos_results = agg_oos_engine._make_results()
            agg_oos_summary = agg_oos_results.get("summary", {})
            agg_oos_trades = agg_oos_results.get("trades", [])

            _print_stats(
                agg_is_summary,
                symbol,
                f"full-is-{scenario}",
                trades=agg_is_results.get("trades", []),
            )
            _print_stats(agg_oos_summary, symbol, f"full-oos-{scenario}", trades=agg_oos_trades)
            _print_spread_sanity_samples(
                symbol=symbol,
                mode=f"full-oos-{scenario}",
                trades=agg_oos_trades,
                cfg=cfg,
                sample_n=20,
                seed=42,
            )
            agg_delta = _delta_block(agg_is_summary, agg_oos_summary)
            click.echo("  Δ OOS-IS:")
            click.echo(
                "    omega={:+.3f}  win_rate={:+.2%}  net=${:+,.2f}  trades={:+d}  dd_pct={:+.2f}".format(
                    float(agg_delta["omega_delta"]),
                    float(agg_delta["win_rate_delta"]),
                    float(agg_delta["net_usd_delta"]),
                    int(agg_delta["n_trades_delta"]),
                    float(agg_delta["dd_pct_delta"]),
                )
            )

            n_trades = int(agg_oos_summary.get("n_trades", 0))
            omega = float(agg_oos_summary.get("omega", 0.0))
            passes = n_trades >= min_trades and omega >= min_omega
            if passes:
                click.secho("✅ PASS", fg="green")
            else:
                click.secho(
                    f"❌ FAIL - trades={n_trades} (need {min_trades}), omega={omega:.2f} (need {min_omega})",
                    fg="red",
                )
                all_pass = False

            audit_path = _persist_execution_audit(
                symbol=symbol,
                mode=f"full_oos_{scenario}",
                results=agg_oos_results,
                metadata={
                    "months": int(months),
                    "oos_window_months": int(oos_window_months),
                    "wf_is_months": int(wf_is_months),
                    "wf_trials": int(wf_trials),
                    "wf_seed": int(wf_seed),
                    "scenario": scenario,
                    "folds": fold_reports,
                    "is_summary": agg_is_summary,
                    "lot_ceiling": lot_ceiling,
                    "engine_config": asdict(cfg),
                },
            )
            equity_plot_oos = _save_equity_plot(
                symbol=symbol,
                mode=f"full_oos_{scenario}",
                trades=agg_oos_trades,
                initial_equity=float(cfg.initial_equity),
                run_id=run_id,
            )
            equity_plot_is = _save_equity_plot(
                symbol=symbol,
                mode=f"full_is_{scenario}",
                trades=agg_is_results.get("trades", []),
                initial_equity=float(cfg.initial_equity),
                run_id=run_id,
            )
            equity_plot_compare = _save_equity_compare_plot(
                symbol=symbol,
                mode=f"full_{scenario}",
                trades_is=agg_is_results.get("trades", []),
                trades_oos=agg_oos_trades,
                initial_equity=float(cfg.initial_equity),
                run_id=run_id,
            )
            lev_cap = (1.0 / float(cfg.margin_rate)) if float(cfg.margin_rate) > 0 else None
            leverage_plot_oos = _save_leverage_plot(
                symbol=symbol,
                mode=f"full_oos_{scenario}",
                trades=agg_oos_trades,
                initial_equity=float(cfg.initial_equity),
                usd_per_point=float(agg_oos_engine._usd_per_point),
                max_leverage=lev_cap,
                run_id=run_id,
            )
            leverage_plot_is = _save_leverage_plot(
                symbol=symbol,
                mode=f"full_is_{scenario}",
                trades=agg_is_results.get("trades", []),
                initial_equity=float(cfg.initial_equity),
                usd_per_point=float(agg_is_engine._usd_per_point),
                max_leverage=lev_cap,
                run_id=run_id,
            )

            scenario_reports[scenario] = {
                "status": "PASS" if passes else "FAIL",
                "pass_gates": bool(passes),
                "lot_ceiling": lot_ceiling,
                "engine_config": asdict(cfg),
                "is_summary": agg_is_summary,
                "oos_summary": agg_oos_summary,
                "delta_oos_minus_is": agg_delta,
                "audit_path": str(audit_path),
                "equity_plot_oos_path": str(equity_plot_oos)
                if equity_plot_oos is not None
                else None,
                "equity_plot_is_path": str(equity_plot_is) if equity_plot_is is not None else None,
                "equity_plot_compare_path": (
                    str(equity_plot_compare) if equity_plot_compare is not None else None
                ),
                "leverage_plot_oos_path": (
                    str(leverage_plot_oos) if leverage_plot_oos is not None else None
                ),
                "leverage_plot_is_path": (
                    str(leverage_plot_is) if leverage_plot_is is not None else None
                ),
                "folds": fold_reports,
            }
    finally:
        LOG.setLevel(original_log_level)
        engine_log.setLevel(original_engine_log_level)

    report = {
        "symbol": symbol,
        "stage": "full",
        "mode": "rolling_oos",
        "run_id": run_id,
        "run_started_utc": run_started_utc.isoformat(),
        "run_finished_utc": datetime.now(timezone.utc).isoformat(),
        "command": " ".join(
            [
                "python",
                "scripts/renko_engine.py",
                symbol,
                "--stage",
                "full",
                "--months",
                str(months),
                "--oos-window-months",
                str(oos_window_months),
                "--wf-is-months",
                str(wf_is_months),
                "--wf-trials",
                str(wf_trials),
                "--wf-seed",
                str(wf_seed),
            ]
        ),
        "inputs": {
            "months": int(months),
            "oos_window_months": int(oos_window_months),
            "wf_is_months": int(wf_is_months),
            "wf_trials": int(wf_trials),
            "wf_seed": int(wf_seed),
            "min_omega": float(min_omega),
            "min_trades": int(min_trades),
            "auto_download": bool(auto_download),
            "latency": {
                "entry_latency_ms": int(entry_latency_ms),
                "exit_latency_ms": int(exit_latency_ms),
                "latency_jitter_ms": int(latency_jitter_ms),
                "latency_seed": int(latency_seed),
            },
            "execution_model": {
                "conservative_fills": bool(conservative_fills),
                "spread_sides": float(spread_sides),
                "entry_slip_bricks": float(entry_slip_bricks),
                "exit_slip_bricks": float(exit_slip_bricks),
                "stop_worst_case_bricks": float(stop_worst_case_bricks),
                "flip_trade_through_bricks": float(flip_trade_through_bricks),
            },
            "overrides": {
                "stop_bricks": stop_bricks_override,
                "trailing_mfe_fraction": trailing_mfe_fraction_override,
                "trailing_mfe_after_bricks": trailing_mfe_after_bricks_override,
                "target_risk": target_risk_override,
                "brick_size": brick_size_override,
                "fliprate_window": fliprate_window_override,
                "markov_window": markov_window_override,
                "fliprate_threshold": fliprate_threshold_override,
                "markov_threshold": markov_threshold_override,
                "max_leverage": max_leverage_override,
            },
        },
        "folds": [
            {"fold": i + 1, "range_start_utc": s.isoformat(), "range_end_utc": e.isoformat()}
            for i, (s, e) in enumerate(folds)
        ],
        "scenario_results": scenario_reports,
        "overall_status": "PASS" if all_pass else "FAIL",
    }

    out_dir = KR / "outputs" / "renko_results" / "backtests" / symbol.upper()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_stem = f"{symbol.lower()}_full_oos_{months}m_{run_id}"
    out_path = _choose_unique_path(out_dir, out_stem)
    with open(out_path, "x", encoding="utf-8") as f:
        json.dump(_json_safe(report), f, indent=2)
    click.secho(f"Saved full rolling OOS report: {out_path}", fg="cyan")
    return all_pass


# ──────────────────────────────────────────────────────────────────────────────
# STAGE 4: PAPER TRADING (Live Broker Data, No Real Orders)
# ──────────────────────────────────────────────────────────────────────────────


def stage_paper(
    symbol: str,
    months: int = 3,
    min_omega: float = 1.5,
    min_trades: int = 30,
    stop_bricks_override: Optional[float] = None,
    trailing_mfe_fraction_override: Optional[float] = None,
    trailing_mfe_after_bricks_override: Optional[int] = None,
    target_risk_override: Optional[float] = None,
    brick_size_override: Optional[float] = None,
    fliprate_window_override: Optional[int] = None,
    markov_window_override: Optional[int] = None,
    fliprate_threshold_override: Optional[float] = None,
    markov_threshold_override: Optional[float] = None,
    max_leverage_override: Optional[float] = None,
    use_pe_gate_override: Optional[bool] = None,
    pe_window_override: Optional[int] = None,
    pe_order_override: Optional[int] = None,
    pe_entry_threshold_override: Optional[float] = None,
    pe_exit_threshold_override: Optional[float] = None,
    use_time_decay_override: Optional[bool] = None,
    stale_brick_factor_override: Optional[float] = None,
    markov_stale_penalty_override: Optional[float] = None,
    use_obi_slippage_buffer_override: Optional[bool] = None,
    obi_levels_override: Optional[int] = None,
    obi_max_buffer_bricks_override: Optional[float] = None,
    obi_ema_alpha_override: Optional[float] = None,
    drift_adapt: bool = False,
    drift_opts: Optional[dict] = None,
) -> bool:
    """Paper trading with live broker data - NO REAL ORDERS.

    Connects to cTrader Open API to stream live M1 bars, runs the same
    RenkoEngine strategy logic as backtesting, and displays real-time stats.
    Uses PaperDispatcher to simulate fills without placing real orders.
    """
    click.echo(f"\n{'=' * 60}")
    click.secho("STAGE 4: PAPER TRADING (Live Data)", fg="cyan", bold=True)
    click.echo(f"{'=' * 60}")
    click.secho("⚠️  Paper trading - NO REAL ORDERS will be placed", fg="yellow")

    # Try to import cTrader connector
    try:
        from kinetra.connectors.ctrader_connector import build_connector
    except ImportError as e:
        click.secho(f"❌ cTrader connector not available: {e}", fg="red")
        click.echo("Install with: pip install ctrader-open-api")
        return False

    # Load DSP profile
    dsp, dsp_file = _load_or_build_dsp_profile(symbol)
    if dsp is None:
        click.secho(f"❌ No DSP profile found at {dsp_file}", fg="red")
        click.echo("Run --stage dsp first")
        return False

    # Shared runtime build (same engine path as live, paper dispatcher)
    cfg, spec, engine, bar_provider_cls, dispatcher_cls = _build_stream_runtime(
        symbol, dsp, lot_ceiling=0.01, live_orders=False
    )
    _apply_strategy_overrides(
        cfg,
        stop_bricks=stop_bricks_override,
        trailing_mfe_fraction=trailing_mfe_fraction_override,
        trailing_mfe_after_bricks=trailing_mfe_after_bricks_override,
        target_risk=target_risk_override,
        brick_size=brick_size_override,
        fliprate_window=fliprate_window_override,
        markov_window=markov_window_override,
        fliprate_threshold=fliprate_threshold_override,
        markov_threshold=markov_threshold_override,
        max_leverage=max_leverage_override,
        use_pe_gate=use_pe_gate_override,
        pe_window=pe_window_override,
        pe_order=pe_order_override,
        pe_entry_threshold=pe_entry_threshold_override,
        pe_exit_threshold=pe_exit_threshold_override,
        use_time_decay=use_time_decay_override,
        stale_brick_factor=stale_brick_factor_override,
        markov_stale_penalty=markov_stale_penalty_override,
        use_obi_slippage_buffer=use_obi_slippage_buffer_override,
        obi_levels=obi_levels_override,
        obi_max_buffer_bricks=obi_max_buffer_bricks_override,
        obi_ema_alpha=obi_ema_alpha_override,
    )
    adaptor = DriftAdaptationController(
        engine,
        {"enabled": drift_adapt, **(drift_opts or {})},
    )

    # Print system specification
    click.echo()
    _print_system_spec(cfg, spec, dsp)

    click.echo()
    click.secho("Connecting to cTrader...", fg="yellow")

    # Connect to cTrader
    try:
        connector = build_connector(timeout_s=30.0)
        click.secho("✅ Connected to cTrader", fg="green")
    except Exception as e:
        click.secho(f"❌ Failed to connect to cTrader: {e}", fg="red")
        click.echo("Check your .env.openapi credentials")
        return False

    # Create bar provider and paper dispatcher
    bar_provider = bar_provider_cls(connector)
    paper_dispatcher = dispatcher_cls(spread_pts={symbol: cfg.spread_ticks})

    # Stats tracking
    stats_lock = threading.Lock()
    last_stats_time = [time.time()]
    stats_interval = 60.0  # Print stats every 60 seconds
    trade_count = [0]
    stop_event = threading.Event()

    def print_periodic_stats():
        """Print stats periodically."""
        while not stop_event.is_set():
            time.sleep(stats_interval)
            with stats_lock:
                results = engine._make_results()
                summary = results.get("summary", {})
                trades = results.get("trades", [])
                n_trades = len(trades)
                if n_trades > trade_count[0]:
                    trade_count[0] = n_trades
                    adaptor.maybe_adapt(int(getattr(engine, "bricks_processed", 0)))
                    console.print()
                    _print_stats(summary, symbol, "paper", trades=trades)
                    last_stats_time[0] = time.time()

    # Start periodic stats thread
    stats_thread = threading.Thread(target=print_periodic_stats, daemon=True)
    stats_thread.start()

    click.echo()
    click.secho("Starting paper trading loop...", fg="cyan")
    click.echo("Streaming live M1 bars from cTrader")
    click.echo("Press Ctrl+C to stop and see final results")
    click.echo()

    results: dict = {"summary": {}, "trades": [], "signals": []}
    try:
        results = _run_streaming_engine(
            engine,
            bar_provider,
            paper_dispatcher,
            stop_event=stop_event,
            stop_msg="Stopping paper trading...",
        )
    finally:
        stop_event.set()
        try:
            bar_provider.stop()
        except Exception:
            pass
        try:
            connector.stop()
        except Exception:
            pass

        # Print final stats
        summary = results.get("summary", {})
        trades = results.get("trades", [])

        click.echo()
        click.secho("=" * 60, fg="cyan")
        click.secho("PAPER TRADING RESULTS", fg="cyan", bold=True)
        click.secho("=" * 60, fg="cyan")
        _print_stats(summary, symbol, "paper", trades=trades)
        audit_path = _persist_execution_audit(
            symbol=symbol,
            mode="paper",
            results=results,
            metadata={"stage": "paper", "min_omega": min_omega, "min_trades": min_trades},
        )
        click.secho(f"Saved execution audit: {audit_path}", fg="cyan")

        # Summary
        n_trades = summary.get("n_trades", 0)
        omega = summary.get("omega", 0.0)
        if n_trades >= min_trades and omega >= min_omega:
            click.secho(f"\n✅ PASS - {n_trades} trades, Omega={omega:.3f}", fg="green")
            return True
        else:
            click.secho(
                f"\n⚠️  Insufficient trades or Omega - {n_trades} trades (need {min_trades}), Omega={omega:.3f} (need {min_omega})",
                fg="yellow",
            )
            return False


# ──────────────────────────────────────────────────────────────────────────────
# MAIN CLI
# ──────────────────────────────────────────────────────────────────────────────


def stage_live_real(
    symbol: str = "XAUUSD",
    live_size: str = "micro",
    live_orders: bool = True,
    preflight_test_order: bool = False,
    ack_live: str = "",
    preflight_lots: float = 0.01,
    stop_bricks_override: Optional[float] = None,
    trailing_mfe_fraction_override: Optional[float] = None,
    trailing_mfe_after_bricks_override: Optional[int] = None,
    target_risk_override: Optional[float] = None,
    brick_size_override: Optional[float] = None,
    fliprate_window_override: Optional[int] = None,
    markov_window_override: Optional[int] = None,
    fliprate_threshold_override: Optional[float] = None,
    markov_threshold_override: Optional[float] = None,
    max_leverage_override: Optional[float] = None,
    use_pe_gate_override: Optional[bool] = None,
    pe_window_override: Optional[int] = None,
    pe_order_override: Optional[int] = None,
    pe_entry_threshold_override: Optional[float] = None,
    pe_exit_threshold_override: Optional[float] = None,
    use_time_decay_override: Optional[bool] = None,
    stale_brick_factor_override: Optional[float] = None,
    markov_stale_penalty_override: Optional[float] = None,
    use_obi_slippage_buffer_override: Optional[bool] = None,
    obi_levels_override: Optional[int] = None,
    obi_max_buffer_bricks_override: Optional[float] = None,
    obi_ema_alpha_override: Optional[float] = None,
    drift_adapt: bool = False,
    drift_opts: Optional[dict] = None,
) -> bool:
    """Live trading via cTrader — real orders when live_orders=True, paper when False."""
    import threading
    import time

    from kinetra.connectors.ctrader_connector import build_connector

    click.echo(f"\n{'=' * 60}")
    if live_orders:
        click.secho("STAGE 5: LIVE TRADING (REAL ORDERS)", fg="red", bold=True)
    else:
        click.secho("STAGE 5: DRY-RUN (Live data, paper orders)", fg="cyan", bold=True)
    click.echo(f"{'=' * 60}")

    LOT_CEILINGS = {"micro": 0.01, "small": 0.10, "full": 50.0}
    lot_ceiling = LOT_CEILINGS.get(live_size, 0.01)

    dsp, dsp_file = _load_or_build_dsp_profile(symbol)
    if dsp is None:
        click.secho("❌ No DSP profile found", fg="red")
        return False
    cfg, spec, engine, bar_provider_cls, dispatcher_cls = _build_stream_runtime(
        symbol, dsp, lot_ceiling=lot_ceiling, live_orders=live_orders
    )
    _apply_strategy_overrides(
        cfg,
        stop_bricks=stop_bricks_override,
        trailing_mfe_fraction=trailing_mfe_fraction_override,
        trailing_mfe_after_bricks=trailing_mfe_after_bricks_override,
        target_risk=target_risk_override,
        brick_size=brick_size_override,
        fliprate_window=fliprate_window_override,
        markov_window=markov_window_override,
        fliprate_threshold=fliprate_threshold_override,
        markov_threshold=markov_threshold_override,
        max_leverage=max_leverage_override,
        use_pe_gate=use_pe_gate_override,
        pe_window=pe_window_override,
        pe_order=pe_order_override,
        pe_entry_threshold=pe_entry_threshold_override,
        pe_exit_threshold=pe_exit_threshold_override,
        use_time_decay=use_time_decay_override,
        stale_brick_factor=stale_brick_factor_override,
        markov_stale_penalty=markov_stale_penalty_override,
        use_obi_slippage_buffer=use_obi_slippage_buffer_override,
        obi_levels=obi_levels_override,
        obi_max_buffer_bricks=obi_max_buffer_bricks_override,
        obi_ema_alpha=obi_ema_alpha_override,
    )
    adaptor = DriftAdaptationController(
        engine,
        {"enabled": drift_adapt, **(drift_opts or {})},
    )

    click.echo()
    _print_system_spec(cfg, spec, dsp)

    click.echo()
    _cfg_color = "red" if live_orders else "cyan"
    _cfg_label = "LIVE TRADING CONFIGURATION" if live_orders else "DRY-RUN CONFIGURATION"
    click.secho("=" * 60, fg=_cfg_color)
    click.secho(f"  {_cfg_label}", fg=_cfg_color, bold=True)
    click.secho("=" * 60, fg=_cfg_color)
    click.echo(f"  Symbol:        {symbol}")
    click.echo(f"  Lot Ceiling:   {lot_ceiling:.2f} lots ({live_size})")
    click.echo(f"  Fill timeout:  {os.getenv('CTRADER_FILL_WAIT_S', '30')}s")
    click.echo(f"  Equity timeout:{os.getenv('CTRADER_EQUITY_TIMEOUT_S', '15')}s")
    click.echo(
        f"  Dashboard mode:{os.getenv('KINETRA_LIVE_DASHBOARD_REPLACE', 'auto')} "
        "(auto=single-frame on TTY)"
    )
    click.echo()

    click.secho("Connecting to cTrader...", fg="yellow")
    try:
        connector = build_connector(timeout_s=30.0)
        click.secho("✅ Connected", fg="green")
    except Exception as e:
        click.secho(f"❌ Failed: {e}", fg="red")
        return False

    click.echo()
    click.secho("Running preflight checks...", fg="yellow", bold=True)
    if not _run_preflight_checks(connector, symbol):
        click.secho("❌ Preflight failed - aborting", fg="red")
        try:
            connector.stop()
        except Exception:
            pass
        return False
    click.secho("✅ Preflight passed", fg="green")
    click.echo()

    if live_orders and preflight_test_order:
        if ack_live != "I_UNDERSTAND_LIVE_RISK":
            click.secho(
                '❌ Execution preflight blocked: pass --ack-live "I_UNDERSTAND_LIVE_RISK"',
                fg="red",
            )
            try:
                connector.stop()
            except Exception:
                pass
            return False
        if preflight_lots <= 0:
            click.secho("❌ Execution preflight blocked: --preflight-lots must be > 0", fg="red")
            try:
                connector.stop()
            except Exception:
                pass
            return False
        if not _run_execution_preflight_trade(
            connector=connector,
            symbol=symbol,
            cfg=cfg,
            bar_provider_cls=bar_provider_cls,
            dispatcher_cls=dispatcher_cls,
            lots=preflight_lots,
        ):
            try:
                connector.stop()
            except Exception:
                pass
            return False

    import logging

    logging.getLogger("kinetra.connectors").setLevel(logging.WARNING)
    logging.getLogger("kinetra.dns_hardening").setLevel(logging.WARNING)
    logging.getLogger("kinetra.renko.ctrader_dispatcher").setLevel(logging.WARNING)

    recent_errors = deque(maxlen=5)
    errors_lock = threading.Lock()

    class _DashboardErrorHandler(logging.Handler):
        """Collect recent ERROR log lines for the live dashboard."""

        def emit(self, record: logging.LogRecord) -> None:
            try:
                if record.levelno < logging.ERROR:
                    return
                ts = datetime.fromtimestamp(record.created, timezone.utc).strftime("%H:%M:%S")
                msg = f"{ts} {record.name}: {record.getMessage()}"
                if len(msg) > 120:
                    msg = msg[:117] + "..."
                with errors_lock:
                    recent_errors.appendleft(msg)
            except Exception:
                pass

    dashboard_error_handler = _DashboardErrorHandler()
    dashboard_error_handler.setLevel(logging.ERROR)
    # Attach once at the top of the hierarchy to avoid duplicate records
    # from child loggers propagating upward.
    dashboard_error_loggers = [logging.getLogger("kinetra")]
    for _lg in dashboard_error_loggers:
        _lg.addHandler(dashboard_error_handler)

    bar_provider = bar_provider_cls(connector)
    live_dispatcher = dispatcher_cls(connector, bar_provider)

    stats_lock = threading.Lock()
    bar_count = [0]
    stop_event = threading.Event()
    replace_raw = os.getenv("KINETRA_LIVE_DASHBOARD_REPLACE", "auto").strip().lower()
    if replace_raw in {"0", "false", "no", "off"}:
        replace_panel = False
    elif replace_raw in {"1", "true", "yes", "on"}:
        replace_panel = True
    else:
        # auto: prefer single-frame live refresh on interactive TTY.
        replace_panel = True

    use_live_replace = bool(replace_panel and console.is_terminal and sys.stdout.isatty())
    if replace_panel and not use_live_replace:
        LOG.info(
            "Live dashboard replace mode disabled (stdout is not an interactive TTY); using periodic snapshots"
        )

    def _conn_status() -> str:
        hs = str(getattr(connector, "health_status", "") or "").upper()
        if hs in {"UP", "DEGRADED", "DOWN"}:
            return hs
        return "UP" if connector.is_connected() else "DOWN"

    # Initialize health monitoring service
    from kinetra.monitoring.connection_health import ConnectionHealthService

    health_service = ConnectionHealthService(connector)
    health_service.start_monitoring()

    last_acct_info = {
        "broker": "Pepperstone",
        "account_id": connector.credentials.account_id,
        "account_type": "ECN",
        "environment": connector.credentials.environment,
        "balance": engine._live_equity,
        "connection_status": _conn_status(),
        "endpoint": getattr(connector, "selected_endpoint", "-") or "-",
        "failover_count": int(getattr(connector, "failover_count", 0)),
        "failover_generation": int(getattr(connector, "failover_generation", 0)),
        "last_failover_utc": getattr(connector, "last_failover_utc", "-") or "-",
        "request_timeouts": int(getattr(connector, "request_timeout_count", 0)),
        "snapshot_source": "cached",
        "recent_errors": [],
        "health_status": "UNKNOWN",
        "health_latency_ms": 0.0,
        "fill_success_rate": 1.0,
        "consecutive_fill_failures": 0,
    }
    broker_balance_start: Optional[float] = None
    broker_balance_end: Optional[float] = None
    try:
        init_snapshot = connector.get_account_snapshot(timeout_s=10.0)
        broker_balance = float(init_snapshot.get("balance", engine._live_equity))
        broker_balance_start = float(broker_balance)
        engine._live_equity = broker_balance
        last_acct_info = {
            "broker": init_snapshot.get("broker_name", "Pepperstone"),
            "account_id": init_snapshot.get("account_id", connector.credentials.account_id),
            "account_type": init_snapshot.get("account_type", "ECN"),
            "environment": connector.credentials.environment,
            "balance": broker_balance,
            "connection_status": _conn_status(),
            "endpoint": getattr(connector, "selected_endpoint", "-") or "-",
            "failover_count": int(getattr(connector, "failover_count", 0)),
            "failover_generation": int(getattr(connector, "failover_generation", 0)),
            "last_failover_utc": getattr(connector, "last_failover_utc", "-") or "-",
            "request_timeouts": int(getattr(connector, "request_timeout_count", 0)),
            "snapshot_source": "fresh",
            "recent_errors": [],
            "health_status": "UNKNOWN",
            "health_latency_ms": 0.0,
            "fill_success_rate": 1.0,
            "consecutive_fill_failures": 0,
        }
    except Exception as e:
        LOG.warning("Initial account snapshot unavailable, using fallback equity: %s", e)

    _mode_label = "LIVE" if live_orders else "DRY-RUN"

    def update_dashboard():
        last_bricks = 0
        last_print_ts = 0.0
        last_signature = None
        last_emit_ts = 0.0
        last_health_probe_ts = 0.0
        min_interval_default = "2" if use_live_replace else "120"
        min_interval_s = max(
            float(os.getenv("KINETRA_LIVE_DASHBOARD_MIN_INTERVAL_S", min_interval_default)), 2.0
        )
        min_emit_gap_s = max(
            float(os.getenv("KINETRA_LIVE_DASHBOARD_MIN_EMIT_GAP_S", "1.5")),
            0.5,
        )
        heartbeat_interval_s = max(
            float(os.getenv("KINETRA_LIVE_HEARTBEAT_INTERVAL_S", "2.0")), 1.0
        )
        stale_brick_probe_s = max(
            float(os.getenv("KINETRA_LIVE_STALE_BRICK_PROBE_S", "90.0")),
            10.0,
        )
        health_probe_interval_s = max(
            float(os.getenv("KINETRA_LIVE_HEALTH_PROBE_INTERVAL_S", "15.0")),
            5.0,
        )
        heartbeat_on = False
        last_heartbeat_flip = time.time()
        while not stop_event.is_set():
            time.sleep(2)
            with stats_lock:
                current_bricks = int(getattr(engine, "bricks_processed", 0))
                now = time.time()
                if (now - last_heartbeat_flip) >= heartbeat_interval_s:
                    heartbeat_on = not heartbeat_on
                    last_heartbeat_flip = now
                bricks_advanced = current_bricks > last_bricks
                if bricks_advanced:
                    bar_count[0] = current_bricks
                    last_bricks = current_bricks
                    adaptor.maybe_adapt(current_bricks)
                if bricks_advanced:
                    try:
                        snapshot = connector.get_account_snapshot()
                        acct_info = {
                            "broker": snapshot.get("broker_name", "Pepperstone"),
                            "account_id": snapshot.get(
                                "account_id", connector.credentials.account_id
                            ),
                            "account_type": snapshot.get("account_type", "ECN"),
                            "environment": connector.credentials.environment,
                            "balance": snapshot.get("balance", engine._live_equity),
                            "connection_status": _conn_status(),
                            "endpoint": getattr(connector, "selected_endpoint", "-") or "-",
                            "failover_count": int(getattr(connector, "failover_count", 0)),
                            "failover_generation": int(
                                getattr(connector, "failover_generation", 0)
                            ),
                            "last_failover_utc": getattr(connector, "last_failover_utc", "-")
                            or "-",
                            "request_timeouts": int(getattr(connector, "request_timeout_count", 0)),
                            "snapshot_source": "fresh",
                            "heartbeat_on": heartbeat_on,
                        }
                        last_acct_info.update(acct_info)
                    except Exception:
                        acct_info = dict(last_acct_info)
                        acct_info["connection_status"] = _conn_status()
                        acct_info["endpoint"] = getattr(connector, "selected_endpoint", "-") or "-"
                        acct_info["failover_count"] = int(getattr(connector, "failover_count", 0))
                        acct_info["failover_generation"] = int(
                            getattr(connector, "failover_generation", 0)
                        )
                        acct_info["last_failover_utc"] = (
                            getattr(connector, "last_failover_utc", "-") or "-"
                        )
                        acct_info["request_timeouts"] = int(
                            getattr(connector, "request_timeout_count", 0)
                        )
                        acct_info["snapshot_source"] = "cached"
                        acct_info["heartbeat_on"] = heartbeat_on
                else:
                    acct_info = dict(last_acct_info)
                    acct_info["connection_status"] = _conn_status()
                    acct_info["endpoint"] = getattr(connector, "selected_endpoint", "-") or "-"
                    acct_info["failover_count"] = int(getattr(connector, "failover_count", 0))
                    acct_info["failover_generation"] = int(
                        getattr(connector, "failover_generation", 0)
                    )
                    acct_info["last_failover_utc"] = (
                        getattr(connector, "last_failover_utc", "-") or "-"
                    )
                    acct_info["request_timeouts"] = int(
                        getattr(connector, "request_timeout_count", 0)
                    )
                    acct_info["snapshot_source"] = "cached"
                    acct_info["heartbeat_on"] = heartbeat_on
                    last_brick_time = getattr(engine, "last_brick_time", None)
                    stale_for_s = 0.0
                    if last_brick_time is not None:
                        try:
                            stale_for_s = max(
                                0.0,
                                (
                                    pd.Timestamp.now(tz="UTC") - pd.Timestamp(last_brick_time)
                                ).total_seconds(),
                            )
                        except Exception:
                            stale_for_s = 0.0
                    status = str(acct_info.get("connection_status", "")).upper()
                    should_probe = status == "DOWN" or (
                        stale_for_s >= stale_brick_probe_s and status != "UP"
                    )
                    if should_probe and (now - last_health_probe_ts) >= health_probe_interval_s:
                        try:
                            probe = connector.get_account_snapshot(timeout_s=5.0)
                            acct_info["balance"] = float(
                                probe.get("balance", acct_info.get("balance", engine._live_equity))
                            )
                            acct_info["snapshot_source"] = "health_probe"
                            acct_info["connection_status"] = _conn_status()
                            acct_info["endpoint"] = (
                                getattr(connector, "selected_endpoint", "-") or "-"
                            )
                            last_acct_info.update(acct_info)
                            LOG.info(
                                "Live health probe succeeded (status=%s, stale_brick_s=%.0f)",
                                acct_info["connection_status"],
                                stale_for_s,
                            )
                        except Exception as probe_exc:
                            LOG.warning("Live health probe failed: %s", probe_exc)
                        finally:
                            last_health_probe_ts = now
                try:
                    with errors_lock:
                        err_lines = list(recent_errors)
                    summary = engine._make_results().get("summary", {})
                    last_eval = getattr(engine, "_last_eval", {}) or {}
                    acct_info["recent_errors"] = err_lines

                    # Update health service data
                    try:
                        health_result = health_service.get_health_status()
                        acct_info["health_status"] = health_result.status.name
                        acct_info["health_latency_ms"] = round(health_result.metrics.latency_ms, 1)
                        acct_info["health_packet_loss"] = round(
                            health_result.metrics.packet_loss_rate, 4
                        )
                    except Exception:
                        pass

                    # Update fill metrics from dispatcher
                    try:
                        fill_metrics = live_dispatcher.get_fill_metrics()
                        acct_info["fill_success_rate"] = fill_metrics.get("fill_success_rate", 1.0)
                        acct_info["consecutive_fill_failures"] = fill_metrics.get(
                            "consecutive_failures", 0
                        )
                        acct_info["orders_submitted"] = fill_metrics.get("orders_submitted", 0)
                        acct_info["orders_filled"] = fill_metrics.get("orders_filled", 0)
                        acct_info["orders_failed"] = fill_metrics.get("orders_failed", 0)
                    except Exception:
                        pass

                    # Tick feed diagnostics
                    acct_info["tick_mode"] = hasattr(bar_provider, "subscribe_ticks")
                    acct_info["ticks_received"] = int(
                        getattr(bar_provider, "ticks_fired", 0)
                    )

                    # Build a comprehensive signature that changes only when
                    # meaningful data changes — prevents spam in non-TTY mode.
                    signature = (
                        int(summary.get("n_trades", 0)),
                        float(summary.get("net_usd", 0.0)),
                        float(summary.get("omega", 0.0)),
                        str(last_eval.get("reason", "n/a")),
                        bool(last_eval.get("is_flip", False)),
                        bool(last_eval.get("entry_ok", False)),
                        bool(acct_info.get("heartbeat_on", False)),
                        str(acct_info.get("connection_status", "-")),
                        str(acct_info.get("snapshot_source", "-")),
                        str(acct_info.get("health_status", "-")),
                        float(acct_info.get("fill_success_rate", 1.0)),
                        int(acct_info.get("consecutive_fill_failures", 0)),
                        err_lines[0] if err_lines else "",
                    )

                    # Only emit when signature changes OR minimum interval elapsed
                    should_emit = last_signature is None or signature != last_signature
                    time_since_emit = now - last_emit_ts
                    time_since_print = now - last_print_ts

                    # Force emit if enough time has passed (heartbeat)
                    if time_since_emit >= min_interval_s:
                        should_emit = True

                    # In non-TTY mode, be more conservative to avoid spam
                    if not use_live_replace and not should_emit:
                        # Skip this update — nothing changed and not yet time for heartbeat
                        continue

                    if should_emit and time_since_emit >= min_emit_gap_s:
                        # Update state before rendering to prevent duplicate emissions
                        last_signature = signature
                        last_emit_ts = now
                        last_print_ts = now

                        panel = _dashboard_renderable(
                            engine, symbol, _mode_label, current_bricks, acct_info
                        )
                        if live_ctx is not None:
                            # Rich Live context — single-frame in-place refresh
                            live_ctx.update(panel, refresh=True)
                        else:
                            # Non-TTY / file redirect mode — periodic snapshots only
                            # Clear line and print fresh panel (avoid stacking)
                            if console.is_terminal:
                                console.clear_live()
                            _print_dashboard(engine, symbol, _mode_label, current_bricks, acct_info)
                except Exception as e:
                    LOG.warning("Dashboard update failed: %s", e)

    # Print before starting the live display — click writes bypass Rich's
    # live context and corrupt the in-place refresh if called after start().
    if live_orders:
        click.secho("Starting LIVE trading...", fg="red", bold=True)
        click.echo("REAL orders will be submitted")
    else:
        click.secho("Starting DRY-RUN...", fg="cyan", bold=True)
        click.echo("Paper orders only — no real trades")
    click.echo("Press Ctrl+C to stop")
    click.echo()

    live_ctx = None
    dashboard_thread = None
    # Collect stream handlers that need to be silenced while the Rich live
    # panel is active.  Log output written to stdout/stderr after live_ctx.start()
    # bypasses Rich's in-place refresh and corrupts the display.
    _silenced_handlers: list = []
    if use_live_replace:
        live_ctx = Live(console=console, refresh_per_second=4, transient=False)
        # Silence any StreamHandler (but not FileHandler) before starting the
        # live display so that log lines don't tear the frame.
        _silenced_handlers = [
            h
            for h in logging.root.handlers
            if isinstance(h, logging.StreamHandler)
            and not isinstance(h, logging.FileHandler)
        ]
        for _h in _silenced_handlers:
            _h.setLevel(logging.CRITICAL + 1)
        live_ctx.start()
    dashboard_thread = threading.Thread(target=update_dashboard, daemon=True)
    dashboard_thread.start()

    results: dict = {"summary": {}, "trades": [], "signals": []}
    try:
        results = _run_streaming_engine(
            engine,
            bar_provider,
            live_dispatcher,
            stop_event=stop_event,
            stop_msg="Stopping live trading...",
        )
    finally:
        stop_event.set()

        # Stop health monitoring
        try:
            health_service.stop_monitoring()
        except Exception:
            pass

        if live_ctx is not None:
            try:
                live_ctx.stop()
            except Exception:
                pass
            # Restore stream handler levels now that the live panel is gone.
            for _h in _silenced_handlers:
                _h.setLevel(logging.INFO)
        try:
            final_snapshot = connector.get_account_snapshot(timeout_s=10.0)
            broker_balance_end = float(
                final_snapshot.get("balance", last_acct_info.get("balance", engine._live_equity))
            )
        except Exception:
            try:
                broker_balance_end = float(last_acct_info.get("balance", engine._live_equity))
            except Exception:
                broker_balance_end = None
        click.echo()
        click.secho("Closing connections...", fg="yellow")
        try:
            bar_provider.stop()
            connector.stop()
        except Exception as e:
            click.secho(f"  Warning: {e}", fg="yellow")
        for _lg in dashboard_error_loggers:
            try:
                _lg.removeHandler(dashboard_error_handler)
            except Exception:
                pass

        summary = results.get("summary", {})
        trades = results.get("trades", [])

        click.echo()
        click.secho("=" * 60, fg="cyan")
        click.secho(f"  {_mode_label} RESULTS", fg="cyan", bold=True)
        click.secho("=" * 60, fg="cyan")
        _print_stats(summary, symbol, _mode_label, trades=trades)
        audit_path = _persist_execution_audit(
            symbol=symbol,
            mode="live",
            results=results,
            metadata={"stage": "live", "live_size": live_size, "lot_ceiling": lot_ceiling},
        )
        click.secho(f"Saved execution audit: {audit_path}", fg="cyan")

        n_trades = summary.get("n_trades", 0)
        omega = summary.get("omega", 0.0)
        net_pnl = summary.get("net_usd", 0.0)

        click.echo()
        if n_trades > 0:
            click.secho(f"  Trades:  {n_trades}", fg="cyan")
            click.secho(
                f"  Strategy Net P&L: ${net_pnl:,.2f}",
                fg="green" if net_pnl > 0 else "red",
            )
            if broker_balance_start is not None and broker_balance_end is not None:
                broker_delta = float(broker_balance_end - broker_balance_start)
                click.secho(
                    f"  Broker Balance Δ: ${broker_delta:,.2f} (start=${broker_balance_start:,.2f} -> end=${broker_balance_end:,.2f})",
                    fg="green" if broker_delta > 0 else ("red" if broker_delta < 0 else "yellow"),
                )
            click.secho(f"  Omega:   {omega:.3f}", fg="green" if omega >= 1.5 else "yellow")
        else:
            click.secho("  No trades executed", fg="yellow")

        click.echo()
        click.secho("⚠️  Verify positions in cTrader terminal", fg="yellow")

    return True


def stage_live_dryrun(
    symbol: str,
    live_size: str = "micro",
    stop_bricks_override: Optional[float] = None,
    trailing_mfe_fraction_override: Optional[float] = None,
    trailing_mfe_after_bricks_override: Optional[int] = None,
    target_risk_override: Optional[float] = None,
    brick_size_override: Optional[float] = None,
    fliprate_window_override: Optional[int] = None,
    markov_window_override: Optional[int] = None,
    fliprate_threshold_override: Optional[float] = None,
    markov_threshold_override: Optional[float] = None,
    max_leverage_override: Optional[float] = None,
    use_pe_gate_override: Optional[bool] = None,
    pe_window_override: Optional[int] = None,
    pe_order_override: Optional[int] = None,
    pe_entry_threshold_override: Optional[float] = None,
    pe_exit_threshold_override: Optional[float] = None,
    use_time_decay_override: Optional[bool] = None,
    stale_brick_factor_override: Optional[float] = None,
    markov_stale_penalty_override: Optional[float] = None,
    use_obi_slippage_buffer_override: Optional[bool] = None,
    obi_levels_override: Optional[int] = None,
    obi_max_buffer_bricks_override: Optional[float] = None,
    obi_ema_alpha_override: Optional[float] = None,
    drift_adapt: bool = False,
    drift_opts: Optional[dict] = None,
) -> bool:
    """Dry-run: live data, paper orders — routes through stage_live_real(live_orders=False)."""
    return stage_live_real(
        symbol=symbol,
        live_size=live_size,
        live_orders=False,
        stop_bricks_override=stop_bricks_override,
        trailing_mfe_fraction_override=trailing_mfe_fraction_override,
        trailing_mfe_after_bricks_override=trailing_mfe_after_bricks_override,
        target_risk_override=target_risk_override,
        brick_size_override=brick_size_override,
        fliprate_window_override=fliprate_window_override,
        markov_window_override=markov_window_override,
        fliprate_threshold_override=fliprate_threshold_override,
        markov_threshold_override=markov_threshold_override,
        max_leverage_override=max_leverage_override,
        use_pe_gate_override=use_pe_gate_override,
        pe_window_override=pe_window_override,
        pe_order_override=pe_order_override,
        pe_entry_threshold_override=pe_entry_threshold_override,
        pe_exit_threshold_override=pe_exit_threshold_override,
        use_time_decay_override=use_time_decay_override,
        stale_brick_factor_override=stale_brick_factor_override,
        markov_stale_penalty_override=markov_stale_penalty_override,
        use_obi_slippage_buffer_override=use_obi_slippage_buffer_override,
        obi_levels_override=obi_levels_override,
        obi_max_buffer_bricks_override=obi_max_buffer_bricks_override,
        obi_ema_alpha_override=obi_ema_alpha_override,
        drift_adapt=drift_adapt,
        drift_opts=drift_opts,
    )


@click.command()
@click.argument("symbol", default="XAUUSD")
@click.option("--dry-run", is_flag=True, help="Paper trading with live data")
@click.option(
    "--live-size",
    type=click.Choice(["micro", "small", "full"]),
    default="micro",
    help="Lot size for live trading",
)
@click.option(
    "--stage",
    type=click.Choice(["dsp", "backtest", "full", "paper", "live", "all"]),
    default="backtest",
    help="Which stage to run",
)
@click.option("--months", type=int, default=3, help="Months of data for backtest")
@click.option(
    "--oos-window-months",
    type=int,
    default=3,
    help="OOS fold size in months for --stage full (rolling OOS)",
)
@click.option(
    "--wf-is-months",
    type=int,
    default=3,
    help="Walk-forward IS window size in months for --stage full",
)
@click.option(
    "--wf-trials",
    type=int,
    default=16,
    help="Walk-forward fitting trials per fold for --stage full",
)
@click.option(
    "--wf-seed",
    type=int,
    default=42,
    help="Walk-forward fitting RNG seed for --stage full",
)
@click.option("--min-omega", type=float, default=1.5, help="Minimum Omega ratio to pass")
@click.option("--min-trades", type=int, default=30, help="Minimum trades to pass")
@click.option("--stop-bricks", type=float, default=None, help="Override stop distance in bricks")
@click.option(
    "--trailing-mfe-fraction",
    type=float,
    default=None,
    help="Lock this fraction of MFE in trailing stop (0..1, optional)",
)
@click.option(
    "--trailing-mfe-after-bricks",
    type=int,
    default=None,
    help="Activate trailing stop after this many held bricks (optional)",
)
@click.option("--target-risk", type=float, default=None, help="Override target USD risk per trade")
@click.option(
    "--lot-ceiling",
    type=float,
    default=50.0,
    help="Lot ceiling cap for non-static sizing in backtest/full",
)
@click.option(
    "--max-leverage",
    type=float,
    default=None,
    help="Maximum allowed leverage (e.g., 100 means 1:100). Enforced as margin cap across modes.",
)
@click.option(
    "--brick-size", type=float, default=None, help="Override Renko brick size (price units)"
)
@click.option(
    "--fliprate-window", type=int, default=None, help="Override flip-rate window (bricks)"
)
@click.option("--markov-window", type=int, default=None, help="Override Markov window (bricks)")
@click.option(
    "--fliprate-threshold", type=float, default=None, help="Override flip-rate gate threshold"
)
@click.option("--markov-threshold", type=float, default=None, help="Override Markov gate threshold")
@click.option(
    "--use-pe-gate/--no-use-pe-gate",
    default=None,
    help="Enable/disable permutation-entropy regime gate",
)
@click.option("--pe-window", type=int, default=None, help="Permutation-entropy rolling window")
@click.option("--pe-order", type=int, default=None, help="Permutation-entropy pattern order")
@click.option("--pe-entry-threshold", type=float, default=None, help="PE entry threshold (0..1)")
@click.option(
    "--pe-exit-threshold", type=float, default=None, help="PE emergency exit threshold (0..1)"
)
@click.option(
    "--use-time-decay/--no-use-time-decay",
    default=None,
    help="Enable/disable stale-brick time-decay Markov penalty",
)
@click.option(
    "--stale-brick-factor", type=float, default=None, help="Stale-brick threshold multiplier"
)
@click.option(
    "--markov-stale-penalty",
    type=float,
    default=None,
    help="Maximum Markov penalty for stale bricks",
)
@click.option(
    "--use-obi-slippage-buffer/--no-use-obi-slippage-buffer",
    default=None,
    help="Enable/disable OBI-driven entry slippage buffer",
)
@click.option("--obi-levels", type=int, default=None, help="Order-book depth levels for OBI")
@click.option(
    "--obi-max-buffer-bricks",
    type=float,
    default=None,
    help="Maximum OBI entry buffer in brick units",
)
@click.option(
    "--obi-ema-alpha",
    type=float,
    default=None,
    help="EMA alpha for smoothing OBI signal (0..1)",
)
@click.option("--monday-start", type=str, default="", help="Legacy launcher option (reserved)")
@click.option("--friday-end", type=str, default="", help="Legacy launcher option (reserved)")
@click.option(
    "--entry-latency-ms",
    type=int,
    default=0,
    help="Backtest latency model: entry-side latency in milliseconds",
)
@click.option(
    "--exit-latency-ms",
    type=int,
    default=0,
    help="Backtest latency model: exit-side latency in milliseconds",
)
@click.option(
    "--latency-jitter-ms",
    type=int,
    default=0,
    help="Backtest latency model: Gaussian jitter (stddev) in milliseconds",
)
@click.option("--latency-seed", type=int, default=42, help="Backtest latency model RNG seed")
@click.option(
    "--conservative-fills/--no-conservative-fills",
    default=True,
    help="Backtest/full: enable conservative adverse fill model",
)
@click.option(
    "--spread-sides",
    type=float,
    default=1.0,
    help="Backtest/full spread multiplier on bid-ask width (1.0=one RT spread, 2.0=extra conservative)",
)
@click.option(
    "--entry-slip-bricks",
    type=float,
    default=0.0,
    help="Backtest/full adverse entry offset in bricks",
)
@click.option(
    "--exit-slip-bricks",
    type=float,
    default=0.10,
    help="Backtest/full adverse exit offset in bricks",
)
@click.option(
    "--stop-worst-case-bricks",
    type=float,
    default=0.25,
    help="Backtest/full extra adverse stop fill offset in bricks",
)
@click.option(
    "--flip-trade-through-bricks",
    type=float,
    default=0.10,
    help="Backtest/full extra adverse flip-exit trade-through offset in bricks",
)
@click.option(
    "--auto-download",
    is_flag=True,
    help="Auto-download/backfill M1 history when local coverage is insufficient",
)
@click.option(
    "--preflight-test-order/--no-preflight-test-order",
    default=False,
    help="Place/close tiny real order before live loop",
)
@click.option(
    "--preflight-lots",
    type=float,
    default=0.01,
    help="Lots for execution preflight test order",
)
@click.option(
    "--ack-live",
    type=str,
    default="",
    help='Required when --preflight-test-order. Must equal "I_UNDERSTAND_LIVE_RISK".',
)
@click.option(
    "--drift-adapt/--no-drift-adapt", default=False, help="Enable bounded online drift adaptation"
)
@click.option(
    "--adapt-every-bars", type=int, default=100, help="Adaptation evaluation cadence in bricks"
)
@click.option(
    "--adapt-min-trades", type=int, default=20, help="Minimum trades before adaptation is active"
)
@click.option(
    "--adapt-low-omega", type=float, default=1.0, help="Defensive adaptation trigger omega"
)
@click.option(
    "--adapt-high-omega", type=float, default=1.8, help="Offensive adaptation trigger omega"
)
@click.option(
    "--adapt-low-win-rate", type=float, default=0.45, help="Defensive adaptation trigger win rate"
)
@click.option(
    "--adapt-high-win-rate", type=float, default=0.60, help="Offensive adaptation trigger win rate"
)
@click.option(
    "--adapt-step-risk", type=float, default=0.10, help="Fractional risk step per adaptation"
)
@click.option(
    "--adapt-step-threshold", type=float, default=0.01, help="Threshold step per adaptation"
)
@click.option(
    "--adapt-step-stop", type=float, default=0.05, help="Stop-bricks fractional step per adaptation"
)
@click.option("--adapt-risk-min", type=float, default=25.0, help="Minimum target risk cap")
@click.option("--adapt-risk-max", type=float, default=200.0, help="Maximum target risk cap")
@click.option("--adapt-stop-min", type=float, default=0.30, help="Minimum stop_bricks cap")
@click.option("--adapt-stop-max", type=float, default=1.50, help="Maximum stop_bricks cap")
@click.option(
    "--adapt-fliprate-min", type=float, default=0.25, help="Minimum fliprate threshold cap"
)
@click.option(
    "--adapt-fliprate-max", type=float, default=0.45, help="Maximum fliprate threshold cap"
)
@click.option("--adapt-markov-min", type=float, default=0.50, help="Minimum markov threshold cap")
@click.option("--adapt-markov-max", type=float, default=0.70, help="Maximum markov threshold cap")
@click.option("--adapt-rollback-omega", type=float, default=0.80, help="Rollback trigger omega")
def main(
    symbol: str = "XAUUSD",
    stage: str = "backtest",
    months: int = 3,
    oos_window_months: int = 3,
    wf_is_months: int = 3,
    wf_trials: int = 16,
    wf_seed: int = 42,
    min_omega: float = 1.5,
    min_trades: int = 30,
    dry_run: bool = False,
    live_size: str = "micro",
    stop_bricks: Optional[float] = None,
    trailing_mfe_fraction: Optional[float] = None,
    trailing_mfe_after_bricks: Optional[int] = None,
    target_risk: Optional[float] = None,
    lot_ceiling: float = 50.0,
    max_leverage: Optional[float] = None,
    brick_size: Optional[float] = None,
    fliprate_window: Optional[int] = None,
    markov_window: Optional[int] = None,
    fliprate_threshold: Optional[float] = None,
    markov_threshold: Optional[float] = None,
    use_pe_gate: Optional[bool] = None,
    pe_window: Optional[int] = None,
    pe_order: Optional[int] = None,
    pe_entry_threshold: Optional[float] = None,
    pe_exit_threshold: Optional[float] = None,
    use_time_decay: Optional[bool] = None,
    stale_brick_factor: Optional[float] = None,
    markov_stale_penalty: Optional[float] = None,
    use_obi_slippage_buffer: Optional[bool] = None,
    obi_levels: Optional[int] = None,
    obi_max_buffer_bricks: Optional[float] = None,
    obi_ema_alpha: Optional[float] = None,
    monday_start: str = "",
    friday_end: str = "",
    entry_latency_ms: int = 0,
    exit_latency_ms: int = 0,
    latency_jitter_ms: int = 0,
    latency_seed: int = 42,
    conservative_fills: bool = True,
    spread_sides: float = 1.0,
    entry_slip_bricks: float = 0.0,
    exit_slip_bricks: float = 0.10,
    stop_worst_case_bricks: float = 0.25,
    flip_trade_through_bricks: float = 0.10,
    auto_download: bool = False,
    preflight_test_order: bool = False,
    preflight_lots: float = 0.01,
    ack_live: str = "",
    drift_adapt: bool = False,
    adapt_every_bars: int = 100,
    adapt_min_trades: int = 20,
    adapt_low_omega: float = 1.0,
    adapt_high_omega: float = 1.8,
    adapt_low_win_rate: float = 0.45,
    adapt_high_win_rate: float = 0.60,
    adapt_step_risk: float = 0.10,
    adapt_step_threshold: float = 0.01,
    adapt_step_stop: float = 0.05,
    adapt_risk_min: float = 25.0,
    adapt_risk_max: float = 200.0,
    adapt_stop_min: float = 0.30,
    adapt_stop_max: float = 1.50,
    adapt_fliprate_min: float = 0.25,
    adapt_fliprate_max: float = 0.45,
    adapt_markov_min: float = 0.50,
    adapt_markov_max: float = 0.70,
    adapt_rollback_omega: float = 0.80,
):
    """Kinetra Renko Brick Trade Engine.

    Validate a symbol through the full pipeline:
    DSP → Backtest → Paper → Live
    """
    click.echo(f"{'=' * 70}")
    click.secho(f"Kinetra Renko Engine: {symbol}", fg="cyan", bold=True)
    click.echo(f"{'=' * 70}")
    drift_opts = {
        "every_bars": adapt_every_bars,
        "min_trades": adapt_min_trades,
        "low_omega": adapt_low_omega,
        "high_omega": adapt_high_omega,
        "low_wr": adapt_low_win_rate,
        "high_wr": adapt_high_win_rate,
        "step_risk": adapt_step_risk,
        "step_thr": adapt_step_threshold,
        "step_stop": adapt_step_stop,
        "risk_min": adapt_risk_min,
        "risk_max": adapt_risk_max,
        "stop_min": adapt_stop_min,
        "stop_max": adapt_stop_max,
        "flip_min": adapt_fliprate_min,
        "flip_max": adapt_fliprate_max,
        "markov_min": adapt_markov_min,
        "markov_max": adapt_markov_max,
        "rollback_omega": adapt_rollback_omega,
    }
    if monday_start or friday_end:
        LOG.info(
            "Accepted legacy launcher options: monday_start=%s friday_end=%s",
            monday_start or "-",
            friday_end or "-",
        )
    if lot_ceiling <= 0:
        click.secho("❌ --lot-ceiling must be > 0", fg="red")
        raise SystemExit(1)
    if max_leverage is not None and max_leverage <= 0:
        click.secho("❌ --max-leverage must be > 0", fg="red")
        raise SystemExit(1)
    if spread_sides <= 0:
        click.secho("❌ --spread-sides must be > 0", fg="red")
        raise SystemExit(1)
    if (
        entry_slip_bricks < 0
        or exit_slip_bricks < 0
        or stop_worst_case_bricks < 0
        or flip_trade_through_bricks < 0
    ):
        click.secho("❌ slip/offset brick parameters must be >= 0", fg="red")
        raise SystemExit(1)
    if pe_window is not None and pe_window < 3:
        click.secho("❌ --pe-window must be >= 3", fg="red")
        raise SystemExit(1)
    if pe_order is not None and pe_order < 2:
        click.secho("❌ --pe-order must be >= 2", fg="red")
        raise SystemExit(1)
    if pe_entry_threshold is not None and not (0.0 < pe_entry_threshold < 1.0):
        click.secho("❌ --pe-entry-threshold must be in (0,1)", fg="red")
        raise SystemExit(1)
    if pe_exit_threshold is not None and not (0.0 < pe_exit_threshold < 1.0):
        click.secho("❌ --pe-exit-threshold must be in (0,1)", fg="red")
        raise SystemExit(1)
    if (
        pe_entry_threshold is not None
        and pe_exit_threshold is not None
        and pe_exit_threshold <= pe_entry_threshold
    ):
        click.secho("❌ --pe-exit-threshold must be > --pe-entry-threshold", fg="red")
        raise SystemExit(1)
    if stale_brick_factor is not None and stale_brick_factor <= 0:
        click.secho("❌ --stale-brick-factor must be > 0", fg="red")
        raise SystemExit(1)
    if markov_stale_penalty is not None and markov_stale_penalty < 0:
        click.secho("❌ --markov-stale-penalty must be >= 0", fg="red")
        raise SystemExit(1)
    if obi_levels is not None and obi_levels <= 0:
        click.secho("❌ --obi-levels must be > 0", fg="red")
        raise SystemExit(1)
    if obi_max_buffer_bricks is not None and obi_max_buffer_bricks < 0:
        click.secho("❌ --obi-max-buffer-bricks must be >= 0", fg="red")
        raise SystemExit(1)
    if obi_ema_alpha is not None and not (0.0 <= obi_ema_alpha <= 1.0):
        click.secho("❌ --obi-ema-alpha must be in [0,1]", fg="red")
        raise SystemExit(1)

    if stage in ("dsp", "all"):
        if not stage_dsp(symbol):
            click.secho("❌ DSP stage failed", fg="red")
            raise SystemExit(1)

    if stage in ("backtest", "all"):
        if not stage_backtest(
            symbol,
            months=months,
            min_omega=min_omega,
            min_trades=min_trades,
            auto_download=auto_download,
            stop_bricks_override=stop_bricks,
            trailing_mfe_fraction_override=trailing_mfe_fraction,
            trailing_mfe_after_bricks_override=trailing_mfe_after_bricks,
            target_risk_override=target_risk,
            brick_size_override=brick_size,
            fliprate_window_override=fliprate_window,
            markov_window_override=markov_window,
            fliprate_threshold_override=fliprate_threshold,
            markov_threshold_override=markov_threshold,
            max_leverage_override=max_leverage,
            entry_latency_ms=entry_latency_ms,
            exit_latency_ms=exit_latency_ms,
            latency_jitter_ms=latency_jitter_ms,
            latency_seed=latency_seed,
            lot_ceiling_override=lot_ceiling,
            conservative_fills=conservative_fills,
            spread_sides=spread_sides,
            entry_slip_bricks=entry_slip_bricks,
            exit_slip_bricks=exit_slip_bricks,
            stop_worst_case_bricks=stop_worst_case_bricks,
            flip_trade_through_bricks=flip_trade_through_bricks,
            use_pe_gate_override=use_pe_gate,
            pe_window_override=pe_window,
            pe_order_override=pe_order,
            pe_entry_threshold_override=pe_entry_threshold,
            pe_exit_threshold_override=pe_exit_threshold,
            use_time_decay_override=use_time_decay,
            stale_brick_factor_override=stale_brick_factor,
            markov_stale_penalty_override=markov_stale_penalty,
            use_obi_slippage_buffer_override=use_obi_slippage_buffer,
            obi_levels_override=obi_levels,
            obi_max_buffer_bricks_override=obi_max_buffer_bricks,
            obi_ema_alpha_override=obi_ema_alpha,
        ):
            click.secho("❌ Backtest stage failed", fg="red")
            raise SystemExit(1)

    if stage == "full":
        if not stage_full_rolling_oos(
            symbol,
            months=months,
            oos_window_months=oos_window_months,
            wf_is_months=wf_is_months,
            wf_trials=wf_trials,
            wf_seed=wf_seed,
            min_omega=min_omega,
            min_trades=min_trades,
            auto_download=auto_download,
            stop_bricks_override=stop_bricks,
            trailing_mfe_fraction_override=trailing_mfe_fraction,
            trailing_mfe_after_bricks_override=trailing_mfe_after_bricks,
            target_risk_override=target_risk,
            brick_size_override=brick_size,
            fliprate_window_override=fliprate_window,
            markov_window_override=markov_window,
            fliprate_threshold_override=fliprate_threshold,
            markov_threshold_override=markov_threshold,
            max_leverage_override=max_leverage,
            entry_latency_ms=entry_latency_ms,
            exit_latency_ms=exit_latency_ms,
            latency_jitter_ms=latency_jitter_ms,
            latency_seed=latency_seed,
            lot_ceiling_override=lot_ceiling,
            conservative_fills=conservative_fills,
            spread_sides=spread_sides,
            entry_slip_bricks=entry_slip_bricks,
            exit_slip_bricks=exit_slip_bricks,
            stop_worst_case_bricks=stop_worst_case_bricks,
            flip_trade_through_bricks=flip_trade_through_bricks,
            use_pe_gate_override=use_pe_gate,
            pe_window_override=pe_window,
            pe_order_override=pe_order,
            pe_entry_threshold_override=pe_entry_threshold,
            pe_exit_threshold_override=pe_exit_threshold,
            use_time_decay_override=use_time_decay,
            stale_brick_factor_override=stale_brick_factor,
            markov_stale_penalty_override=markov_stale_penalty,
            use_obi_slippage_buffer_override=use_obi_slippage_buffer,
            obi_levels_override=obi_levels,
            obi_max_buffer_bricks_override=obi_max_buffer_bricks,
            obi_ema_alpha_override=obi_ema_alpha,
        ):
            click.secho("❌ Full rolling OOS stage failed", fg="red")
            raise SystemExit(1)

    if stage in ("paper", "all"):
        if not stage_paper(
            symbol,
            months=months,
            min_omega=min_omega,
            min_trades=min_trades,
            stop_bricks_override=stop_bricks,
            trailing_mfe_fraction_override=trailing_mfe_fraction,
            trailing_mfe_after_bricks_override=trailing_mfe_after_bricks,
            target_risk_override=target_risk,
            brick_size_override=brick_size,
            fliprate_window_override=fliprate_window,
            markov_window_override=markov_window,
            fliprate_threshold_override=fliprate_threshold,
            markov_threshold_override=markov_threshold,
            max_leverage_override=max_leverage,
            use_pe_gate_override=use_pe_gate,
            pe_window_override=pe_window,
            pe_order_override=pe_order,
            pe_entry_threshold_override=pe_entry_threshold,
            pe_exit_threshold_override=pe_exit_threshold,
            use_time_decay_override=use_time_decay,
            stale_brick_factor_override=stale_brick_factor,
            markov_stale_penalty_override=markov_stale_penalty,
            use_obi_slippage_buffer_override=use_obi_slippage_buffer,
            obi_levels_override=obi_levels,
            obi_max_buffer_bricks_override=obi_max_buffer_bricks,
            obi_ema_alpha_override=obi_ema_alpha,
            drift_adapt=drift_adapt,
            drift_opts=drift_opts,
        ):
            click.secho("❌ Paper stage failed", fg="red")
            raise SystemExit(1)

    if stage == "live":
        if dry_run:
            # Dry-run: live data, paper orders
            if not stage_live_dryrun(
                symbol,
                live_size=live_size,
                stop_bricks_override=stop_bricks,
                trailing_mfe_fraction_override=trailing_mfe_fraction,
                trailing_mfe_after_bricks_override=trailing_mfe_after_bricks,
                target_risk_override=target_risk,
                brick_size_override=brick_size,
                fliprate_window_override=fliprate_window,
                markov_window_override=markov_window,
                fliprate_threshold_override=fliprate_threshold,
                markov_threshold_override=markov_threshold,
                max_leverage_override=max_leverage,
                use_pe_gate_override=use_pe_gate,
                pe_window_override=pe_window,
                pe_order_override=pe_order,
                pe_entry_threshold_override=pe_entry_threshold,
                pe_exit_threshold_override=pe_exit_threshold,
                use_time_decay_override=use_time_decay,
                stale_brick_factor_override=stale_brick_factor,
                markov_stale_penalty_override=markov_stale_penalty,
                use_obi_slippage_buffer_override=use_obi_slippage_buffer,
                obi_levels_override=obi_levels,
                obi_max_buffer_bricks_override=obi_max_buffer_bricks,
                obi_ema_alpha_override=obi_ema_alpha,
                drift_adapt=drift_adapt,
                drift_opts=drift_opts,
            ):
                raise SystemExit(1)
        else:
            # REAL live trading
            if not stage_live_real(
                symbol,
                live_size=live_size,
                preflight_test_order=preflight_test_order,
                preflight_lots=preflight_lots,
                ack_live=ack_live,
                stop_bricks_override=stop_bricks,
                trailing_mfe_fraction_override=trailing_mfe_fraction,
                trailing_mfe_after_bricks_override=trailing_mfe_after_bricks,
                target_risk_override=target_risk,
                brick_size_override=brick_size,
                fliprate_window_override=fliprate_window,
                markov_window_override=markov_window,
                fliprate_threshold_override=fliprate_threshold,
                markov_threshold_override=markov_threshold,
                max_leverage_override=max_leverage,
                use_pe_gate_override=use_pe_gate,
                pe_window_override=pe_window,
                pe_order_override=pe_order,
                pe_entry_threshold_override=pe_entry_threshold,
                pe_exit_threshold_override=pe_exit_threshold,
                use_time_decay_override=use_time_decay,
                stale_brick_factor_override=stale_brick_factor,
                markov_stale_penalty_override=markov_stale_penalty,
                use_obi_slippage_buffer_override=use_obi_slippage_buffer,
                obi_levels_override=obi_levels,
                obi_max_buffer_bricks_override=obi_max_buffer_bricks,
                obi_ema_alpha_override=obi_ema_alpha,
                drift_adapt=drift_adapt,
                drift_opts=drift_opts,
            ):
                raise SystemExit(1)

    click.secho("✅ All stages passed!", fg="green")


if __name__ == "__main__":
    main()
