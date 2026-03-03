#!/usr/bin/env python3
"""Download cTrader M1 history + Renko contract specs.

Outputs (per symbol):
1. M1 CSV with canonical columns:
   time,open,high,low,close,volume,spread
2. contract_spec.json with friction/sizing metadata used by Renko pipeline.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from kinetra.canonical_asset_classification import get_asset_class_with_fallback
from kinetra.connectors.ctrader_connector import CTraderConnector, CTraderCredentials
from kinetra.monitoring import emit_event, emit_health, telemetry_span

logger = logging.getLogger("ctrader.download")


def _fmt_eta(seconds: float) -> str:
    s = max(int(seconds), 0)
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    if h > 0:
        return f"{h:02d}:{m:02d}:{sec:02d}"
    return f"{m:02d}:{sec:02d}"


def _parse_env_file(path: Path) -> Dict[str, str]:
    out: Dict[str, str] = {}
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        s = line.strip()
        if not s or s.startswith("#") or "=" not in s:
            continue
        k, _, v = s.partition("=")
        out[k.strip()] = v.strip().strip('"').strip("'")
    return out


def _env_or_file(key: str, env_data: Dict[str, str], default: str = "") -> str:
    return (os.getenv(key) or env_data.get(key) or default).strip()


def _category_for_symbol(symbol: str) -> str:
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


def _is_ecn_commission_symbol(symbol: str) -> bool:
    s = (symbol or "").upper()
    if not s:
        return False
    if s.startswith(("XAU", "XAG", "XPT", "XPD")):
        return True
    return bool(re.fullmatch(r"[A-Z]{6}", s))


def _select_price_scale(
    *,
    low_raw: int,
    delta_open: int,
    delta_high: int,
    delta_close: int,
    digits: int,
    pip_position: int,
) -> float:
    scales = {
        float(10 ** max(digits, 0)),
        float(10 ** max(pip_position + 1, 0)),
        100000.0,
    }
    best_scale = 100000.0
    best_score = float("inf")
    for scale in scales:
        l = low_raw / scale
        o = (low_raw + delta_open) / scale
        h = (low_raw + delta_high) / scale
        c = (low_raw + delta_close) / scale
        if not (o > 0 and h > 0 and l > 0 and c > 0):
            continue
        if h < l:
            continue
        rel_range = (h - l) / max(abs(c), 1e-12)
        score = rel_range
        if c < 1e-6 or c > 1e6:
            score += 1e3
        if score < best_score:
            best_score = score
            best_scale = scale
    return best_scale


def _decode_trendbar(tb: object, digits: int, pip_position: int) -> Optional[Dict[str, float]]:
    low_raw = int(getattr(tb, "low", 0) or 0)
    if low_raw <= 0:
        return None

    delta_open = int(getattr(tb, "deltaOpen", 0) or 0)
    delta_high = int(getattr(tb, "deltaHigh", 0) or 0)
    delta_close = int(getattr(tb, "deltaClose", 0) or 0)
    volume = float(getattr(tb, "volume", 0) or 0.0)
    scale = _select_price_scale(
        low_raw=low_raw,
        delta_open=delta_open,
        delta_high=delta_high,
        delta_close=delta_close,
        digits=digits,
        pip_position=pip_position,
    )

    ts_min = int(getattr(tb, "utcTimestampInMinutes", 0) or 0)
    if ts_min > 0:
        ts = datetime.fromtimestamp(ts_min * 60, tz=timezone.utc)
    else:
        # Fallback for SDK variants where timestamp is ms.
        ts_ms = int(getattr(tb, "timestamp", 0) or 0)
        if ts_ms <= 0:
            return None
        ts = datetime.fromtimestamp(ts_ms / 1000.0, tz=timezone.utc)

    o = (low_raw + delta_open) / scale
    h = (low_raw + delta_high) / scale
    l = low_raw / scale
    c = (low_raw + delta_close) / scale
    if not (o > 0 and h > 0 and l > 0 and c > 0):
        return None

    return {
        "time": ts.isoformat(),
        "open": o,
        "high": h,
        "low": l,
        "close": c,
        "volume": volume,
    }


@dataclass
class SymbolMeta:
    symbol_id: int
    symbol_name: str
    base_asset_id: Optional[int]
    quote_asset_id: Optional[int]
    digits: Optional[int] = None
    pip_position: Optional[int] = None


def _get_symbol_metas(conn: CTraderConnector, symbols: Sequence[str]) -> Dict[str, SymbolMeta]:
    from ctrader_open_api.messages import OpenApiMessages_pb2 as api_msgs

    req = api_msgs.ProtoOASymbolsListReq()
    req.ctidTraderAccountId = conn.credentials.account_id
    req.includeArchivedSymbols = False
    resp = conn.send_and_wait(req, timeout_s=20.0)
    if resp is None or hasattr(resp, "errorCode"):
        raise RuntimeError("Failed to fetch cTrader symbols list")

    lookup: Dict[str, SymbolMeta] = {}
    for sym in getattr(resp, "symbol", []):
        symbol_name = str(getattr(sym, "symbolName", "") or "")
        sid = int(getattr(sym, "symbolId", 0) or 0)
        if not symbol_name or sid <= 0:
            continue
        lookup[symbol_name] = SymbolMeta(
            symbol_id=sid,
            symbol_name=symbol_name,
            base_asset_id=int(getattr(sym, "baseAssetId", 0) or 0) or None,
            quote_asset_id=int(getattr(sym, "quoteAssetId", 0) or 0) or None,
            digits=None,
            pip_position=None,
        )

    out: Dict[str, SymbolMeta] = {}
    for requested in symbols:
        sid = conn.find_symbol_id(requested)
        if sid is None:
            raise RuntimeError(f"Could not resolve symbol: {requested}")
        broker_symbol = conn.symbol_name_for_id(sid)
        meta = lookup.get(broker_symbol)
        if meta is None:
            meta = SymbolMeta(
                symbol_id=sid,
                symbol_name=broker_symbol,
                base_asset_id=None,
                quote_asset_id=None,
                digits=None,
                pip_position=None,
            )
        out[requested] = meta
    return out


def _load_symbol_precision(conn: CTraderConnector, symbol_id: int) -> tuple[int, int]:
    from ctrader_open_api.messages import OpenApiMessages_pb2 as api_msgs

    req = api_msgs.ProtoOASymbolByIdReq()
    req.ctidTraderAccountId = conn.credentials.account_id
    req.symbolId.append(symbol_id)
    resp = conn.send_and_wait(req, timeout_s=20.0)
    if resp is None or hasattr(resp, "errorCode") or not getattr(resp, "symbol", []):
        digits = conn.get_digits(symbol_id)
        return digits, max(digits - 1, 0)
    sym = resp.symbol[0]
    digits = int(getattr(sym, "digits", 0) or 0) or conn.get_digits(symbol_id)
    pip_pos = int(getattr(sym, "pipPosition", max(digits - 1, 0)) or max(digits - 1, 0))
    return digits, pip_pos


def _get_assets_map(conn: CTraderConnector) -> Dict[int, str]:
    from ctrader_open_api.messages import OpenApiMessages_pb2 as api_msgs

    req = api_msgs.ProtoOAAssetListReq()
    req.ctidTraderAccountId = conn.credentials.account_id
    resp = conn.send_and_wait(req, timeout_s=20.0)
    if resp is None or hasattr(resp, "errorCode"):
        return {}
    assets: Dict[int, str] = {}
    for a in getattr(resp, "asset", []):
        aid = int(getattr(a, "assetId", 0) or 0)
        name = str(getattr(a, "name", "") or "")
        if aid > 0 and name:
            assets[aid] = name.upper()
    return assets


def _collect_live_spread_samples(
    conn: CTraderConnector,
    symbol_id: int,
    *,
    tick_size: float,
    digits: int,
    pip_position: int,
    sample_seconds: float,
) -> List[float]:
    from ctrader_open_api.messages import OpenApiMessages_pb2 as api_msgs

    samples: List[float] = []

    def _handler(payload: object) -> None:
        sid = int(getattr(payload, "symbolId", 0) or 0)
        if sid != symbol_id:
            return
        bid_raw = int(getattr(payload, "bid", 0) or 0)
        ask_raw = int(getattr(payload, "ask", 0) or 0)
        if bid_raw <= 0 or ask_raw <= 0:
            return
        raw_diff = ask_raw - bid_raw
        if raw_diff <= 0:
            return

        # cTrader SpotEvent scaling can vary by SDK/feed. Evaluate plausible scales.
        scale_candidates = {
            float(10 ** max(digits, 0)),
            float(10 ** max(pip_position + 1, 0)),
            100000.0,  # common Open API absolute price scale
        }
        pts_candidates: List[float] = []
        for scale in scale_candidates:
            spread_price = raw_diff / scale
            spread_pts = spread_price / max(tick_size, 1e-12)
            if math.isfinite(spread_pts) and spread_pts > 0:
                pts_candidates.append(float(spread_pts))
        if not pts_candidates:
            return

        # Pick the smallest positive candidate (avoids catastrophic overscaling).
        spread_pts = min(pts_candidates)
        samples.append(spread_pts)

    conn.add_push_handler("ProtoOASpotEvent", _handler)
    try:
        req = api_msgs.ProtoOASubscribeSpotsReq()
        req.ctidTraderAccountId = conn.credentials.account_id
        req.symbolId.append(symbol_id)
        req.subscribeToSpotTimestamp = True
        conn.send_and_wait(req, timeout_s=10.0)
        t_end = time.monotonic() + max(sample_seconds, 0.5)
        while time.monotonic() < t_end:
            time.sleep(0.1)
    finally:
        try:
            un = api_msgs.ProtoOAUnsubscribeSpotsReq()
            un.ctidTraderAccountId = conn.credentials.account_id
            un.symbolId.append(symbol_id)
            conn.send_and_wait(un, timeout_s=10.0)
        except Exception:
            logger.debug("spot unsubscribe failed", exc_info=True)
        conn.remove_push_handler("ProtoOASpotEvent", _handler)

    return samples


def _download_m1_history(
    conn: CTraderConnector,
    symbol_id: int,
    *,
    symbol_label: str,
    digits: int,
    pip_position: int,
    days: int,
    chunk_days: int,
    start_override: Optional[datetime] = None,
    end_override: Optional[datetime] = None,
    chunk_retries: int = 2,
    max_consecutive_failures: int = 3,
) -> pd.DataFrame:
    from ctrader_open_api.messages import OpenApiMessages_pb2 as api_msgs

    end = end_override or datetime.now(tz=timezone.utc)
    start = start_override or (end - timedelta(days=max(days, 1)))
    if start >= end:
        return pd.DataFrame(columns=["time", "open", "high", "low", "close", "volume"])
    # SDK enum naming differs across versions; numeric value 1 is canonical M1.
    period = 1
    rows: List[Dict[str, float]] = []

    chunk_span = timedelta(days=max(chunk_days, 1))
    total_seconds = max((end - start).total_seconds(), 1.0)
    total_chunks = max(int(math.ceil(total_seconds / chunk_span.total_seconds())), 1)

    cur = start
    chunk_idx = 0
    t0 = time.perf_counter()
    consecutive_failures = 0
    while cur < end:
        nxt = min(cur + chunk_span, end)
        chunk_idx += 1

        req = api_msgs.ProtoOAGetTrendbarsReq()
        req.ctidTraderAccountId = conn.credentials.account_id
        req.symbolId = symbol_id
        req.period = period
        req.fromTimestamp = int(cur.timestamp() * 1000)
        req.toTimestamp = int(nxt.timestamp() * 1000)
        req.count = 10000

        resp = None
        for attempt in range(max(chunk_retries + 1, 1)):
            resp = conn.send_and_wait(req, timeout_s=30.0)
            if resp is not None and not hasattr(resp, "errorCode"):
                break
            if attempt < chunk_retries:
                time.sleep(0.25)
        if resp is None or hasattr(resp, "errorCode"):
            consecutive_failures += 1
            logger.warning("trendbar request failed for %s → %s", cur.isoformat(), nxt.isoformat())
            if consecutive_failures >= max(max_consecutive_failures, 1):
                logger.warning(
                    "%s: stopping early after %d consecutive chunk failures; partial data will be saved",
                    symbol_label,
                    consecutive_failures,
                )
                break
        else:
            consecutive_failures = 0

            for tb in getattr(resp, "trendbar", []):
                row = _decode_trendbar(tb, digits=digits, pip_position=pip_position)
                if row is not None:
                    rows.append(row)

        elapsed = time.perf_counter() - t0
        progress = chunk_idx / total_chunks
        eta_s = (elapsed / max(progress, 1e-9)) - elapsed
        logger.info(
            "%s: history chunk %d/%d (%.1f%%) elapsed=%s eta=%s bars=%d",
            symbol_label,
            chunk_idx,
            total_chunks,
            progress * 100.0,
            _fmt_eta(elapsed),
            _fmt_eta(eta_s),
            len(rows),
        )

        cur = nxt
        time.sleep(0.05)

    if not rows:
        return pd.DataFrame(columns=["time", "open", "high", "low", "close", "volume"])

    df = pd.DataFrame(rows)
    df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
    df = df.dropna(subset=["time"])
    df = df.sort_values("time").drop_duplicates(subset=["time"], keep="last")
    df["time"] = df["time"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    return df[["time", "open", "high", "low", "close", "volume"]].reset_index(drop=True)


def _build_contract_spec(
    conn: CTraderConnector,
    meta: SymbolMeta,
    *,
    spread_typical_pts: float,
    assets_by_id: Dict[int, str],
) -> Dict[str, object]:
    from ctrader_open_api.messages import OpenApiMessages_pb2 as api_msgs

    req = api_msgs.ProtoOASymbolByIdReq()
    req.ctidTraderAccountId = conn.credentials.account_id
    req.symbolId.append(meta.symbol_id)
    resp = conn.send_and_wait(req, timeout_s=20.0)
    if resp is None or hasattr(resp, "errorCode") or not getattr(resp, "symbol", []):
        raise RuntimeError(f"Failed to fetch symbol spec for {meta.symbol_name}")

    sym = resp.symbol[0]
    digits = int(getattr(sym, "digits", 0) or 0) or conn.get_digits(meta.symbol_id)
    tick_size = 10 ** (-digits)
    lot_size = float(getattr(sym, "lotSize", 0) or 0.0)
    min_volume_units = float(getattr(sym, "minVolume", 0) or 0.0)
    step_volume_units = float(getattr(sym, "stepVolume", 0) or 0.0)
    max_volume_units = float(getattr(sym, "maxVolume", 0) or 0.0)

    volume_min = (min_volume_units / lot_size) if lot_size > 0 else 0.0
    volume_step = (step_volume_units / lot_size) if lot_size > 0 else 0.0
    volume_max = (max_volume_units / lot_size) if lot_size > 0 else 0.0

    quote_asset = assets_by_id.get(meta.quote_asset_id or -1, "")
    usd_per_price_unit = lot_size if quote_asset == "USD" and lot_size > 0 else 0.0

    commission = float(getattr(sym, "commission", 0) or 0.0)
    precise_comm = float(getattr(sym, "preciseTradingCommissionRate", 0) or 0.0)
    min_comm = float(getattr(sym, "minCommission", 0) or 0.0)
    precise_min_comm = float(getattr(sym, "preciseMinCommission", 0) or 0.0)
    swap_long_points = float(getattr(sym, "swapLong", 0.0) or 0.0)
    swap_short_points = float(getattr(sym, "swapShort", 0.0) or 0.0)
    swap_mode = int(getattr(sym, "swapCalculationType", 0) or 0)
    swap_triple_day = int(getattr(sym, "swapRollover3Days", 0) or 0)

    raw_comm_per_lot = float(precise_comm or commission or 0.0)
    default_ecn_comm_per_lot = float(os.getenv("CTRADER_ECN_COMMISSION_PER_LOT", "3.5"))
    is_ecn_symbol = _is_ecn_commission_symbol(meta.symbol_name)
    is_commission_free = raw_comm_per_lot <= 0.0 and not is_ecn_symbol
    commission_inferred = raw_comm_per_lot <= 0.0 and is_ecn_symbol
    resolved_comm_per_lot = default_ecn_comm_per_lot if commission_inferred else raw_comm_per_lot

    return {
        "symbol": meta.symbol_name.upper(),
        "symbol_id": int(meta.symbol_id),
        "base_asset_id": int(meta.base_asset_id or 0),
        "quote_asset_id": int(meta.quote_asset_id or 0),
        "base_asset": assets_by_id.get(meta.base_asset_id or -1, ""),
        "quote_asset": quote_asset,
        "broker_symbol": meta.symbol_name,
        "broker_source": "ctrader",
        "spread_typical_pts": float(spread_typical_pts),
        "spread_points": float(spread_typical_pts),
        "tick_size": float(tick_size),
        "tickSize": float(tick_size),
        "point_value": float(tick_size),
        "commission_per_lot": float(resolved_comm_per_lot),
        "commission": float(commission),
        "min_commission": float(precise_min_comm or min_comm),
        "commission_raw_per_lot": float(raw_comm_per_lot),
        "commission_inferred_ecn_default": bool(commission_inferred),
        "is_commission_free": bool(is_commission_free),
        "commission_default_per_lot": float(default_ecn_comm_per_lot),
        "contract_size": float(lot_size),
        "volume_min": float(volume_min),
        "volume_step": float(volume_step),
        "volume_max": float(volume_max),
        "swap_long_points": float(swap_long_points),
        "swap_short_points": float(swap_short_points),
        "swap_mode": int(swap_mode),
        "swap_triple_day": int(swap_triple_day),
        "account_type": os.getenv("CTRADER_ACCOUNT_TYPE", ""),
        "usd_per_price_unit": float(usd_per_price_unit),
        "broker_friction": {
            "spread_points": float(spread_typical_pts),
            "tick_size": float(tick_size),
            "commission": float(commission),
            "commission_per_lot": float(resolved_comm_per_lot),
            "min_commission": float(precise_min_comm or min_comm),
            "commission_raw_per_lot": float(raw_comm_per_lot),
            "commission_inferred_ecn_default": bool(commission_inferred),
            "is_commission_free": bool(is_commission_free),
            "commission_default_per_lot": float(default_ecn_comm_per_lot),
            "swap_long_points": float(swap_long_points),
            "swap_short_points": float(swap_short_points),
            "swap_mode": int(swap_mode),
            "swap_triple_day": int(swap_triple_day),
        },
        "ctrader_raw": {
            "symbol_id": int(meta.symbol_id),
            "digits": int(digits),
            "pipPosition": int(getattr(sym, "pipPosition", 0) or 0),
            "lotSize": float(lot_size),
            "minVolume": float(min_volume_units),
            "stepVolume": float(step_volume_units),
            "maxVolume": float(max_volume_units),
            "commission": float(commission),
            "preciseTradingCommissionRate": float(precise_comm),
            "minCommission": float(min_comm),
            "preciseMinCommission": float(precise_min_comm),
            "swapLong": float(swap_long_points),
            "swapShort": float(swap_short_points),
            "swapCalculationType": int(swap_mode),
            "swapRollover3Days": int(swap_triple_day),
            "leverageId": int(getattr(sym, "leverageId", 0) or 0),
        },
    }


def _safe_name(s: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in s)


def _account_slug(env_data: Dict[str, str], creds: CTraderCredentials) -> str:
    explicit = _env_or_file("CTRADER_ACCOUNT_SLUG", env_data)
    if explicit:
        return _safe_name(explicit)
    base = f"pepperstone_{creds.environment}_{creds.account_id}"
    return _safe_name(base)


def _parse_symbols(symbols_arg: str) -> List[str]:
    syms = [s.strip().upper() for s in symbols_arg.split(",") if s.strip()]
    if not syms:
        raise ValueError("No symbols provided")
    return syms


def _load_existing_csv_resume(
    sym_dir: Path, symbol: str
) -> tuple[pd.DataFrame, Optional[datetime]]:
    """Load existing CSV and return (df, last_timestamp + 1min) for forward resume."""
    files = sorted(sym_dir.glob(f"{symbol.upper()}_M1_*.csv"))
    if not files:
        return pd.DataFrame(), None
    latest = files[-1]
    try:
        existing = pd.read_csv(latest)
    except Exception:
        return pd.DataFrame(), None
    if "time" not in existing.columns or existing.empty:
        return pd.DataFrame(), None
    t = pd.to_datetime(existing["time"], utc=True, errors="coerce").dropna()
    if t.empty:
        return pd.DataFrame(), None
    last_ts = t.iloc[-1].to_pydatetime() + timedelta(minutes=1)
    logger.info(
        "%s: resume from cached file %s (last=%s)",
        symbol,
        latest.name,
        t.iloc[-1].isoformat(),
    )
    return existing, last_ts


def _load_existing_csv_backfill(
    sym_dir: Path, symbol: str
) -> tuple[pd.DataFrame, Optional[datetime]]:
    """Load existing CSV and return (df, first_timestamp) for backward backfill.

    Returns the first timestamp in the cached data so we can download older data
    before it.
    """
    files = sorted(sym_dir.glob(f"{symbol.upper()}_M1_*.csv"))
    if not files:
        return pd.DataFrame(), None
    latest = files[-1]
    try:
        existing = pd.read_csv(latest)
    except Exception:
        return pd.DataFrame(), None
    if "time" not in existing.columns or existing.empty:
        return pd.DataFrame(), None
    t = pd.to_datetime(existing["time"], utc=True, errors="coerce").dropna()
    if t.empty:
        return pd.DataFrame(), None
    first_ts = t.iloc[0].to_pydatetime()
    logger.info(
        "%s: backfill from cached file %s (first=%s)",
        symbol,
        latest.name,
        t.iloc[0].isoformat(),
    )
    return existing, first_ts


def _archive_old_csvs(sym_dir: Path, symbol: str, *, keep_last: int = 2) -> None:
    """Archive older CSV files, keeping the ones with longest history.

    Files are named: SYMBOL_M1_YYYYMMDDHHMM_YYYYMMDDHHMM.csv
    We keep the file(s) with the longest time span (most bars), not just newest.
    """
    files = list(sym_dir.glob(f"{symbol.upper()}_M1_*.csv"))
    if len(files) <= keep_last:
        return

    # Parse date range from each filename and calculate duration
    def get_file_duration(filepath: Path) -> tuple[timedelta, str]:
        """Extract start/end timestamps from filename and return duration."""
        match = re.search(
            rf"{symbol.upper()}_M1_(\d{{14}})_(\d{{14}})\.csv", filepath.name, re.IGNORECASE
        )
        if match:
            start_str, end_str = match.groups()
            try:
                start = datetime.strptime(start_str, "%Y%m%d%H%M")
                end = datetime.strptime(end_str, "%Y%m%d%H%M")
                duration = end - start
                return (duration, filepath.name)
            except ValueError:
                pass
        # Fallback: use file modification time if parsing fails
        return (timedelta(days=0), filepath.name)

    # Sort by duration (longest first), then by filename for stability
    files_with_duration = [(get_file_duration(f), f) for f in files]
    files_with_duration.sort(key=lambda x: (-x[0][0].total_seconds(), x[0][1]))

    # Keep the files with longest history, archive the rest
    archive_dir = sym_dir / "archive"
    archive_dir.mkdir(parents=True, exist_ok=True)
    to_archive = [f for _, f in files_with_duration[keep_last:]]
    for src in to_archive:
        dst = archive_dir / src.name
        if dst.exists():
            dst.unlink()
        src.rename(dst)


def _spec_fingerprint(spec: Dict[str, object]) -> str:
    payload = json.dumps(spec, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _snapshot_contract_spec(
    sym_dir: Path,
    spec: Dict[str, object],
    *,
    keep_last: int = 20,
) -> tuple[bool, Optional[Path]]:
    spec_path = sym_dir / "contract_spec.json"
    new_fp = _spec_fingerprint(spec)
    old_fp = None
    if spec_path.exists():
        try:
            old_spec = json.loads(spec_path.read_text(encoding="utf-8"))
            old_fp = _spec_fingerprint(old_spec)
        except Exception:
            old_fp = None

    # Always keep latest spec at canonical path.
    spec_path.write_text(json.dumps(spec, indent=2), encoding="utf-8")

    changed = old_fp != new_fp
    if not changed:
        return False, None

    spec_archive_dir = sym_dir / "archive" / "contract_specs"
    spec_archive_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    snap_path = spec_archive_dir / f"contract_spec_{ts}.json"
    snap_path.write_text(json.dumps(spec, indent=2), encoding="utf-8")

    # Keep only latest N snapshots.
    snaps = sorted(spec_archive_dir.glob("contract_spec_*.json"))
    if len(snaps) > keep_last:
        for old in snaps[: len(snaps) - keep_last]:
            old.unlink()

    # Append lightweight change log for quick diff tracking.
    changelog = spec_archive_dir / "changelog.jsonl"
    changelog_entry = {
        "timestamp_utc": ts,
        "old_sha256": old_fp,
        "new_sha256": new_fp,
        "snapshot": snap_path.name,
    }
    with changelog.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(changelog_entry) + "\n")

    return True, snap_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Download cTrader M1 history + contract specs")
    parser.add_argument(
        "--symbols", required=True, help="Comma-separated symbols, e.g. XAUUSD,NAS100"
    )
    parser.add_argument("--days", type=int, default=30, help="History lookback in days")
    parser.add_argument("--chunk-days", type=int, default=7, help="Chunk size for API requests")
    parser.add_argument(
        "--spread-sample-seconds", type=float, default=8.0, help="Live spot sample window"
    )
    parser.add_argument(
        "--default-spread-pts",
        type=float,
        default=float(os.getenv("CTRADER_DEFAULT_SPREAD_PTS", "1.0")),
        help="Fallback spread in points when no live samples are available",
    )
    parser.add_argument(
        "--max-spread-pts",
        type=float,
        default=float(os.getenv("CTRADER_MAX_SPREAD_PTS", "1000")),
        help="Sanity cap for median spread points; above this falls back to default",
    )
    parser.add_argument(
        "--min-spread-samples",
        type=int,
        default=int(os.getenv("CTRADER_MIN_SPREAD_SAMPLES", "20")),
        help="Minimum live spread samples required before using live median",
    )
    parser.add_argument(
        "--data-root",
        default="data/master_standardized",
        help="Root folder where broker/account/category/symbol paths are created",
    )
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        default=True,
        help="Resume from latest local M1 CSV if present (default: on)",
    )
    parser.add_argument(
        "--no-resume",
        dest="resume",
        action="store_false",
        help="Disable resume and redownload the full requested window.",
    )
    parser.add_argument(
        "--backfill",
        action="store_true",
        default=False,
        help="Backfill mode: download older data BEFORE the first cached timestamp. "
        "Useful for extending historical data backwards in time.",
    )
    parser.add_argument("--chunk-retries", type=int, default=2, help="Retries per failed chunk")
    parser.add_argument(
        "--max-consecutive-failures",
        type=int,
        default=3,
        help="Abort symbol download early after N consecutive chunk failures",
    )
    parser.add_argument(
        "--parallel-symbols",
        type=int,
        default=1,
        help="Download symbols in parallel using one worker process per symbol",
    )
    parser.add_argument(
        "--keep-spec-snapshots",
        type=int,
        default=20,
        help="How many changed contract spec snapshots to retain per symbol",
    )
    parser.add_argument(
        "--connect-retries",
        type=int,
        default=3,
        help="Connector start/auth retries before failing",
    )
    parser.add_argument(
        "--connect-retry-sleep",
        type=float,
        default=3.0,
        help="Seconds to sleep between connector start retries",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    env_data = _parse_env_file(PROJECT_ROOT / ".env")
    env_data.update(_parse_env_file(PROJECT_ROOT / ".env.openapi"))
    symbols = _parse_symbols(args.symbols)
    emit_event(
        stream="system",
        component="ctrader_download",
        event_type="run_start",
        status="info",
        payload={
            "symbols": symbols,
            "days": int(args.days),
            "chunk_days": int(args.chunk_days),
            "resume": bool(args.resume),
            "backfill": bool(args.backfill),
            "parallel_symbols": int(args.parallel_symbols),
        },
    )

    if args.parallel_symbols > 1 and len(symbols) > 1:
        logger.info(
            "parallel symbol mode: workers=%d symbols=%d", args.parallel_symbols, len(symbols)
        )
        py = sys.executable
        script = str(Path(__file__).resolve())
        base_cmd = [
            py,
            script,
            "--days",
            str(args.days),
            "--chunk-days",
            str(args.chunk_days),
            "--spread-sample-seconds",
            str(args.spread_sample_seconds),
            "--default-spread-pts",
            str(args.default_spread_pts),
            "--max-spread-pts",
            str(args.max_spread_pts),
            "--min-spread-samples",
            str(args.min_spread_samples),
            "--data-root",
            str(args.data_root),
            "--chunk-retries",
            str(args.chunk_retries),
            "--max-consecutive-failures",
            str(args.max_consecutive_failures),
            "--parallel-symbols",
            "1",
            "--keep-spec-snapshots",
            str(args.keep_spec_snapshots),
            "--connect-retries",
            str(args.connect_retries),
            "--connect-retry-sleep",
            str(args.connect_retry_sleep),
        ]
        if args.verbose:
            base_cmd.append("--verbose")
        if args.backfill:
            base_cmd.append("--backfill")
        elif args.resume:
            base_cmd.append("--resume")
        else:
            base_cmd.append("--no-resume")

        failures = 0
        with ThreadPoolExecutor(max_workers=args.parallel_symbols) as ex:
            fut_map = {}
            for sym in symbols:
                cmd = base_cmd + ["--symbols", sym]
                fut = ex.submit(subprocess.run, cmd, cwd=PROJECT_ROOT)
                fut_map[fut] = sym
            for fut in as_completed(fut_map):
                sym = fut_map[fut]
                try:
                    proc = fut.result()
                    rc = int(getattr(proc, "returncode", 1))
                except Exception:
                    rc = 1
                if rc != 0:
                    failures += 1
                    logger.error("%s: parallel worker failed (exit=%d)", sym, rc)
                else:
                    logger.info("%s: parallel worker complete", sym)
        return 0 if failures == 0 else 2

    creds = CTraderCredentials.from_env()
    conn = CTraderConnector(creds)
    started = False
    total_attempts = max(int(args.connect_retries), 0) + 1
    with telemetry_span(
        stream="system",
        component="ctrader_download",
        operation="connector_startup",
        payload={"attempts": total_attempts, "environment": creds.environment},
    ):
        for attempt in range(1, total_attempts + 1):
            if conn.start(timeout_s=25.0):
                started = True
                emit_health(
                    component="ctrader_download",
                    status="ok",
                    checks={"connector_start": "pass"},
                    metrics={"attempt": attempt},
                    details={"environment": creds.environment},
                )
                break
            logger.warning(
                "cTrader connector start/auth failed (attempt %d/%d)",
                attempt,
                total_attempts,
            )
            emit_event(
                stream="system",
                component="ctrader_download",
                event_type="connector_retry",
                status="warn",
                payload={"attempt": attempt, "attempts": total_attempts},
            )
            if attempt < total_attempts:
                time.sleep(max(float(args.connect_retry_sleep), 0.0))
    if not started:
        logger.error(
            "cTrader connector failed to start/authenticate after %d attempts", total_attempts
        )
        emit_health(
            component="ctrader_download",
            status="critical",
            checks={"connector_start": "fail"},
            details={"attempts": total_attempts},
        )
        return 2

    try:
        assets_by_id = _get_assets_map(conn)
        metas = _get_symbol_metas(conn, symbols)
        acct_slug = _account_slug(env_data, creds)
        data_root = (PROJECT_ROOT / args.data_root).resolve()
        t_total = time.perf_counter()
        total_symbols = len(symbols)

        for idx, requested in enumerate(symbols, start=1):
            t_sym = time.perf_counter()
            meta = metas[requested]
            logger.info(
                "Downloading %s (broker symbol=%s, id=%d)",
                requested,
                meta.symbol_name,
                meta.symbol_id,
            )

            precise_digits, pip_pos = _load_symbol_precision(conn, meta.symbol_id)
            meta.digits = precise_digits
            meta.pip_position = pip_pos

            cat = _category_for_symbol(requested)
            sym_dir = data_root / "ctrader" / acct_slug / cat / requested.upper()
            sym_dir.mkdir(parents=True, exist_ok=True)
            existing_df = pd.DataFrame()
            resume_start = None
            backfill_end = None  # For backfill mode: download up to this timestamp
            existing_rows = 0

            if args.backfill:
                # Backfill: download older data BEFORE the first cached timestamp
                existing_df, backfill_end = _load_existing_csv_backfill(sym_dir, requested)
                existing_rows = len(existing_df)
                if backfill_end is not None:
                    logger.info(
                        "%s: backfill mode - downloading %d days before %s",
                        requested,
                        args.days,
                        backfill_end.isoformat(),
                    )
            elif args.resume:
                # Resume: download newer data AFTER the last cached timestamp
                existing_df, resume_start = _load_existing_csv_resume(sym_dir, requested)
                existing_rows = len(existing_df)

            # Calculate download window based on mode
            if args.backfill and backfill_end is not None:
                # Backfill: download from (first_ts - days) to first_ts
                download_end = backfill_end
                download_start = download_end - timedelta(days=args.days)
            else:
                # Normal or resume mode
                download_start = resume_start
                download_end = None

            df = _download_m1_history(
                conn,
                meta.symbol_id,
                symbol_label=requested,
                digits=precise_digits,
                pip_position=pip_pos,
                days=args.days,
                chunk_days=args.chunk_days,
                start_override=download_start,
                end_override=download_end,
                chunk_retries=args.chunk_retries,
                max_consecutive_failures=args.max_consecutive_failures,
            )
            new_rows = len(df)
            if not existing_df.empty and not df.empty:
                df = pd.concat([existing_df, df], ignore_index=True)
                df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
                df = (
                    df.dropna(subset=["time"])
                    .sort_values("time")
                    .drop_duplicates(subset=["time"], keep="last")
                )
                df["time"] = df["time"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
                df = df[
                    ["time", "open", "high", "low", "close", "volume"]
                    + (["spread"] if "spread" in df.columns else [])
                ]
                logger.info(
                    "%s: merged cached+new rows cached=%d new=%d deduped_total=%d",
                    requested,
                    existing_rows,
                    new_rows,
                    len(df),
                )
            elif df.empty and not existing_df.empty:
                logger.info("%s: no new bars downloaded; keeping cached dataset", requested)
                df = existing_df.copy()
            else:
                logger.info("%s: downloaded rows=%d", requested, len(df))

            if df.empty:
                logger.warning("%s: no bars returned", requested)
                continue

            digits = conn.get_digits(meta.symbol_id)
            tick_size = 10 ** (-digits)
            spread_samples = _collect_live_spread_samples(
                conn,
                meta.symbol_id,
                tick_size=tick_size,
                digits=precise_digits,
                pip_position=pip_pos,
                sample_seconds=args.spread_sample_seconds,
            )
            if len(spread_samples) >= max(args.min_spread_samples, 1):
                spread_pts = float(pd.Series(spread_samples).median())
                spread_source = "live_spot_median"
            else:
                spread_pts = float(args.default_spread_pts)
                spread_source = "fallback_default_insufficient_samples"
            if spread_pts > float(args.max_spread_pts):
                logger.warning(
                    "%s: spread_pts %.4f exceeds max_spread_pts %.2f; using default %.4f",
                    requested,
                    spread_pts,
                    args.max_spread_pts,
                    args.default_spread_pts,
                )
                spread_pts = float(args.default_spread_pts)
                spread_source = "fallback_default_cap"
            df["spread"] = spread_pts

            times = pd.to_datetime(df["time"], utc=True, errors="coerce").dropna()
            start_str = times.iloc[0].strftime("%Y%m%d%H%M")
            end_str = times.iloc[-1].strftime("%Y%m%d%H%M")
            csv_name = f"{requested.upper()}_M1_{start_str}_{end_str}.csv"
            csv_path = sym_dir / csv_name
            df.to_csv(csv_path, index=False)
            _archive_old_csvs(sym_dir, requested, keep_last=2)

            spec = _build_contract_spec(
                conn,
                meta,
                spread_typical_pts=spread_pts,
                assets_by_id=assets_by_id,
            )
            spec["spread_source"] = spread_source
            spec["spread_samples_count"] = int(len(spread_samples))
            changed, snap_path = _snapshot_contract_spec(
                sym_dir,
                spec,
                keep_last=max(args.keep_spec_snapshots, 1),
            )
            if changed and snap_path is not None:
                logger.info("%s: contract spec changed, snapshot saved: %s", requested, snap_path)
            else:
                logger.info("%s: contract spec unchanged", requested)

            logger.info(
                "%s: wrote %s bars to %s (spread_pts=%.4f, source=%s)",
                requested,
                len(df),
                csv_path,
                spread_pts,
                spread_source,
            )
            emit_event(
                stream="system",
                component="ctrader_download",
                event_type="symbol_download_complete",
                status="ok",
                payload={
                    "symbol": requested,
                    "bars": int(len(df)),
                    "spread_pts": float(spread_pts),
                    "spread_source": spread_source,
                    "csv_path": str(csv_path),
                },
            )
            elapsed_total = time.perf_counter() - t_total
            done = idx / max(total_symbols, 1)
            eta_total = (elapsed_total / max(done, 1e-9)) - elapsed_total
            logger.info(
                "overall progress %d/%d (%.1f%%) last_symbol_elapsed=%s total_elapsed=%s total_eta=%s",
                idx,
                total_symbols,
                done * 100.0,
                _fmt_eta(time.perf_counter() - t_sym),
                _fmt_eta(elapsed_total),
                _fmt_eta(eta_total),
            )

    finally:
        conn.stop()

    logger.info("cTrader download complete.")
    emit_health(
        component="ctrader_download",
        status="ok",
        checks={"run_complete": "pass"},
        metrics={"symbols": len(symbols)},
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
