#!/usr/bin/env python3
"""Download historical BID tick data from cTrader.

Output (per symbol):
  {data_root}/ctrader/{slug}/{cat}/{SYMBOL}/ticks/{SYMBOL}_ticks_{from}_{to}.csv
  Columns: time,bid   (time = ISO-8601 UTC, bid = price in instrument units)

Usage:
  python scripts/ctrader/download_ticks.py --symbols XAUUSD --months 3
  python scripts/ctrader/download_ticks.py --symbols XAUUSD --days 7 --verbose
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from kinetra.canonical_asset_classification import get_asset_class_with_fallback
from kinetra.connectors.ctrader_connector import CTraderConnector, CTraderCredentials

logger = logging.getLogger("ctrader.ticks")


# ── Helpers ───────────────────────────────────────────────────────────────────


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


def _safe_name(s: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in s)


def _account_slug(env_data: Dict[str, str], creds: CTraderCredentials) -> str:
    explicit = _env_or_file("CTRADER_ACCOUNT_SLUG", env_data)
    if explicit:
        return _safe_name(explicit)
    base = f"pepperstone_{creds.environment}_{creds.account_id}"
    return _safe_name(base)


def _load_symbol_id_and_digits(conn: CTraderConnector, symbol: str) -> tuple[int, int]:
    """Resolve symbol ID and digit precision."""
    from ctrader_open_api.messages import OpenApiMessages_pb2 as api_msgs

    symbol_id = conn.find_symbol_id(symbol)
    if symbol_id is None:
        raise RuntimeError(f"Could not resolve symbol: {symbol}")

    req = api_msgs.ProtoOASymbolByIdReq()
    req.ctidTraderAccountId = conn.credentials.account_id
    req.symbolId.append(symbol_id)
    resp = conn.send_and_wait(req, timeout_s=20.0)
    if resp is None or hasattr(resp, "errorCode") or not getattr(resp, "symbol", []):
        digits = conn.get_digits(symbol_id)
    else:
        sym = resp.symbol[0]
        digits = int(getattr(sym, "digits", 0) or 0) or conn.get_digits(symbol_id)

    return symbol_id, digits


# ── Core tick downloader ───────────────────────────────────────────────────────


def _decode_tick_batch(tick_data: list, scale: float) -> tuple[List[dict], int]:
    """Decode one ProtoOAGetTickDataRes batch into rows.

    Returns (rows, last_abs_ts_ms).
    Each row: {'time': ISO str, 'bid': float}.
    Delta-encoding: timestamp[0] and tick[0] are absolute; rest are deltas.
    """
    rows: List[dict] = []
    abs_ts_ms = 0
    abs_price_pts = 0
    for td in tick_data:
        abs_ts_ms += int(td.timestamp)
        abs_price_pts += int(td.tick)
        ts = datetime.fromtimestamp(abs_ts_ms / 1000.0, tz=timezone.utc)
        rows.append({"time": ts.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z",
                     "bid": abs_price_pts / scale})
    return rows, abs_ts_ms


def _download_ticks_window(
    conn: CTraderConnector,
    symbol_id: int,
    scale: float,
    from_ms: int,
    to_ms: int,
    *,
    max_hasmore: int = 200,
) -> tuple[List[dict], bool]:
    """Download all ticks in [from_ms, to_ms]. Returns (rows, reached_api_limit).

    Handles hasMore pagination within the window.
    reached_api_limit=True means the API stopped returning data early (retention limit).
    """
    from ctrader_open_api.messages import OpenApiMessages_pb2 as api_msgs

    all_rows: List[dict] = []
    cur_from = from_ms
    reached_limit = False

    for _ in range(max_hasmore):
        req = api_msgs.ProtoOAGetTickDataReq()
        req.ctidTraderAccountId = conn.credentials.account_id
        req.symbolId = symbol_id
        req.type = 1  # BID
        req.fromTimestamp = cur_from
        req.toTimestamp = to_ms

        resp = conn.send_and_wait(req, timeout_s=30.0)
        if resp is None or hasattr(resp, "errorCode"):
            reached_limit = True
            break

        ticks = list(getattr(resp, "tickData", []))
        if not ticks:
            break

        rows, last_ts_ms = _decode_tick_batch(ticks, scale)
        all_rows.extend(rows)

        if not getattr(resp, "hasMore", False):
            break

        # Advance past last received tick
        cur_from = last_ts_ms + 1
        if cur_from >= to_ms:
            break

        time.sleep(0.02)

    return all_rows, reached_limit


def _download_ticks(
    conn: CTraderConnector,
    symbol_id: int,
    digits: int,
    *,
    symbol_label: str,
    start: datetime,
    end: datetime,
    chunk_hours: int = 4,
) -> pd.DataFrame:
    """Download BID ticks for the given window. Returns DataFrame(time, bid)."""
    scale = float(10 ** digits)
    chunk_span = timedelta(hours=max(chunk_hours, 1))

    total_seconds = max((end - start).total_seconds(), 1.0)
    total_chunks = max(int(math.ceil(total_seconds / chunk_span.total_seconds())), 1)

    all_rows: List[dict] = []
    cur = start
    chunk_idx = 0
    t0 = time.perf_counter()
    empty_chunks = 0

    while cur < end:
        nxt = min(cur + chunk_span, end)
        chunk_idx += 1

        rows, limit_hit = _download_ticks_window(
            conn,
            symbol_id,
            scale,
            from_ms=int(cur.timestamp() * 1000),
            to_ms=int(nxt.timestamp() * 1000),
        )

        if rows:
            all_rows.extend(rows)
            empty_chunks = 0
        else:
            empty_chunks += 1
            if empty_chunks >= 3 and chunk_idx > 3:
                logger.warning(
                    "%s: 3 consecutive empty chunks — API retention limit likely reached at %s",
                    symbol_label,
                    cur.date(),
                )
                break

        if limit_hit:
            logger.warning("%s: API error at chunk %d, stopping early", symbol_label, chunk_idx)
            break

        elapsed = time.perf_counter() - t0
        progress = chunk_idx / total_chunks
        eta_s = (elapsed / max(progress, 1e-9)) - elapsed
        logger.info(
            "%s: ticks chunk %d/%d (%.1f%%)  ticks_so_far=%d  elapsed=%s  eta=%s",
            symbol_label,
            chunk_idx,
            total_chunks,
            progress * 100.0,
            len(all_rows),
            _fmt_eta(elapsed),
            _fmt_eta(eta_s),
        )

        cur = nxt
        time.sleep(0.05)

    if not all_rows:
        return pd.DataFrame(columns=["time", "bid"])

    df = pd.DataFrame(all_rows)
    df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
    df = df.dropna(subset=["time"]).sort_values("time").drop_duplicates(subset=["time"], keep="last")
    df["time"] = df["time"].dt.strftime("%Y-%m-%dT%H:%M:%S.%f").str[:-3] + "Z"
    return df[["time", "bid"]].reset_index(drop=True)


# ── Resume logic ──────────────────────────────────────────────────────────────


def _find_existing_tick_files(tick_dir: Path, symbol: str) -> list[Path]:
    return sorted(tick_dir.glob(f"{symbol.upper()}_ticks_*.csv"))


def _last_tick_time(tick_dir: Path, symbol: str) -> Optional[datetime]:
    """Return timestamp of last tick in existing files, or None."""
    files = _find_existing_tick_files(tick_dir, symbol)
    if not files:
        return None
    latest = files[-1]
    try:
        df = pd.read_csv(latest, usecols=["time"])
        t = pd.to_datetime(df["time"], utc=True, errors="coerce").dropna()
        if t.empty:
            return None
        return t.iloc[-1].to_pydatetime()
    except Exception:
        return None


def _merge_and_save(tick_dir: Path, symbol: str, new_df: pd.DataFrame) -> Path:
    """Merge new_df with any existing tick CSVs and save a single merged file."""
    existing_files = _find_existing_tick_files(tick_dir, symbol)
    dfs = []
    for f in existing_files:
        try:
            dfs.append(pd.read_csv(f))
        except Exception:
            pass
    dfs.append(new_df)

    df = pd.concat(dfs, ignore_index=True)
    df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
    df = (
        df.dropna(subset=["time"])
        .sort_values("time")
        .drop_duplicates(subset=["time"], keep="last")
    )
    df["time"] = df["time"].dt.strftime("%Y-%m-%dT%H:%M:%S.%f").str[:-3] + "Z"
    df = df[["time", "bid"]].reset_index(drop=True)

    t_col = pd.to_datetime(df["time"], utc=True)
    start_str = t_col.iloc[0].strftime("%Y%m%d%H%M")
    end_str = t_col.iloc[-1].strftime("%Y%m%d%H%M")
    out_path = tick_dir / f"{symbol.upper()}_ticks_{start_str}_{end_str}.csv"
    df.to_csv(out_path, index=False)

    # Remove old files (keep only the merged one)
    for f in existing_files:
        if f.resolve() != out_path.resolve():
            try:
                f.unlink()
            except Exception:
                pass

    return out_path


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(description="Download cTrader BID tick history")
    parser.add_argument("--symbols", required=True, help="Comma-separated symbols e.g. XAUUSD")
    parser.add_argument("--months", type=int, default=3, help="Lookback in months (default 3)")
    parser.add_argument("--days", type=int, default=0, help="Lookback in days (overrides --months)")
    parser.add_argument("--chunk-hours", type=int, default=4, help="Hours per API chunk (default 4)")
    parser.add_argument("--data-root", default="data/master_standardized")
    parser.add_argument("--no-resume", action="store_true", help="Ignore existing data, redownload")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--connect-retries", type=int, default=3, help="Connector start retries"
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    env_data = _parse_env_file(PROJECT_ROOT / ".env")
    env_data.update(_parse_env_file(PROJECT_ROOT / ".env.openapi"))

    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    if not symbols:
        logger.error("No symbols provided")
        return 1

    creds = CTraderCredentials.from_env()
    conn = CTraderConnector(creds)

    started = False
    for attempt in range(1, args.connect_retries + 2):
        if conn.start(timeout_s=25.0):
            started = True
            break
        logger.warning("Connector start failed (attempt %d/%d)", attempt, args.connect_retries + 1)
        if attempt <= args.connect_retries:
            time.sleep(3.0)

    if not started:
        logger.error("Could not connect to cTrader after %d attempts", args.connect_retries + 1)
        return 2

    acct_slug = _account_slug(env_data, creds)
    data_root = (PROJECT_ROOT / args.data_root).resolve()

    end_dt = datetime.now(tz=timezone.utc)
    if args.days > 0:
        start_dt = end_dt - timedelta(days=args.days)
    else:
        start_dt = end_dt - timedelta(days=args.months * 31)

    try:
        for symbol in symbols:
            logger.info("=== %s ===", symbol)

            symbol_id, digits = _load_symbol_id_and_digits(conn, symbol)
            logger.info("%s: symbol_id=%d digits=%d", symbol, symbol_id, digits)

            cat = _category_for_symbol(symbol)
            tick_dir = data_root / "ctrader" / acct_slug / cat / symbol / "ticks"
            tick_dir.mkdir(parents=True, exist_ok=True)

            dl_start = start_dt
            if not args.no_resume:
                last_ts = _last_tick_time(tick_dir, symbol)
                if last_ts is not None:
                    dl_start = last_ts + timedelta(milliseconds=1)
                    logger.info("%s: resuming from %s", symbol, last_ts.isoformat())

            if dl_start >= end_dt:
                logger.info("%s: already up to date", symbol)
                continue

            logger.info(
                "%s: downloading ticks %s → %s",
                symbol,
                dl_start.strftime("%Y-%m-%d"),
                end_dt.strftime("%Y-%m-%d"),
            )

            df = _download_ticks(
                conn,
                symbol_id,
                digits,
                symbol_label=symbol,
                start=dl_start,
                end=end_dt,
                chunk_hours=args.chunk_hours,
            )

            if df.empty:
                logger.warning("%s: no ticks downloaded", symbol)
                continue

            out_path = _merge_and_save(tick_dir, symbol, df)
            logger.info("%s: saved %d ticks → %s", symbol, len(df), out_path)

    finally:
        try:
            conn.stop()
        except Exception:
            pass

    return 0


if __name__ == "__main__":
    sys.exit(main())
