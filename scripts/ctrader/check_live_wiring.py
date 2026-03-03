#!/usr/bin/env python3
"""Safe cTrader live wiring check (optionally with tiny order roundtrip).

Checks:
1) credentials load
2) connector auth
3) account snapshot
4) symbol resolution
5) cTrader dispatcher + bar provider wiring
6) optional quote/bar observation window
7) optional live order roundtrip (open + close)
"""

from __future__ import annotations

import argparse
import csv
import logging
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from kinetra.connectors.ctrader_connector import CTraderConnector, CTraderCredentials
from kinetra.monitoring import emit_event, emit_health
from kinetra.renko.ctrader_dispatcher import CTraderBarProvider, CTraderOrderDispatcher
from kinetra.renko.live_trader import TradeDirection

LOG = logging.getLogger("ctrader.live_wiring")


@dataclass
class Check:
    name: str
    ok: bool
    detail: str


def _parse_symbols(raw: str) -> List[str]:
    out = [s.strip().upper() for s in raw.split(",") if s.strip()]
    if not out:
        raise ValueError("No symbols specified")
    return out


def _print_results(results: List[Check]) -> None:
    for r in results:
        mark = "PASS" if r.ok else "FAIL"
        print(f"[{mark}] {r.name} - {r.detail}")


def _load_last_csv_close(data_root: Path, symbol: str) -> Optional[float]:
    symbol = symbol.strip().upper()
    if not symbol:
        return None
    try:
        folders = sorted(p for p in data_root.glob(f"*/{symbol}") if p.is_dir())
    except Exception:
        return None
    if not folders:
        return None
    csvs = sorted(folders[-1].glob(f"{symbol}_M1_*.csv"))
    if not csvs:
        return None
    path = csvs[-1]
    try:
        with path.open("r", encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            close_val: Optional[float] = None
            for row in reader:
                raw = row.get("close")
                if raw is None:
                    continue
                try:
                    v = float(raw)
                except Exception:
                    continue
                if math.isfinite(v) and v > 0:
                    close_val = v
            return close_val
    except Exception:
        return None


def _scale_hint(raw: float, ref: float) -> Optional[int]:
    if raw <= 0 or ref <= 0:
        return None
    ratio = raw / ref
    if not math.isfinite(ratio) or ratio <= 0:
        return None
    # Find nearest power-of-ten correction k where raw / 10^k ~= ref.
    k = int(round(math.log10(ratio)))
    if 1 <= abs(k) <= 9:
        return k
    return None


def _evaluate_price_normalization(
    symbols: List[str],
    live_last_close: Dict[str, float],
    live_spreads: Dict[str, float],
    data_root: Path,
) -> Tuple[bool, str]:
    details: List[str] = []
    ok = True
    for sym in symbols:
        ref = _load_last_csv_close(data_root, sym)
        live = live_last_close.get(sym)
        spr = float(live_spreads.get(sym, 0.0) or 0.0)
        if ref is None or live is None or live <= 0:
            details.append(f"{sym}: skipped(ref={ref}, live={live})")
            continue
        ratio = live / ref
        ratio_ok = 0.05 <= ratio <= 20.0
        spread_ok = 0.0 <= spr <= (abs(live) * 0.1)
        if not ratio_ok or not spread_ok:
            ok = False
            hint = _scale_hint(live, ref)
            hint_txt = f", scale_hint=10^{hint}" if hint is not None else ""
            details.append(
                f"{sym}: live={live:.6f} ref={ref:.6f} ratio={ratio:.6f} spread={spr:.6f}{hint_txt}"
            )
        else:
            details.append(
                f"{sym}: live={live:.6f} ref={ref:.6f} ratio={ratio:.6f} spread={spr:.6f}"
            )
    return ok, "; ".join(details)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Safe cTrader live wiring check (optional tiny order roundtrip)"
    )
    ap.add_argument("--symbols", default="XAUUSD,NAS100")
    ap.add_argument("--timeout", type=float, default=20.0)
    ap.add_argument(
        "--observe-seconds",
        type=float,
        default=10.0,
        help="How long to watch quotes/bars after provider start",
    )
    ap.add_argument("--fill-timeout", type=float, default=10.0)
    ap.add_argument("--require-bars", action="store_true")
    ap.add_argument(
        "--data-root",
        default="data/master_standardized/ctrader/pepperstone_demo_45841299",
        help="Used for price normalization sanity against latest local M1 close",
    )
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument(
        "--test-order", action="store_true", help="Send tiny live test order and close it"
    )
    ap.add_argument(
        "--ack-live",
        default="",
        help='Required when --test-order. Must equal "I_UNDERSTAND_LIVE_RISK".',
    )
    ap.add_argument(
        "--test-symbol", default="", help="Order test symbol; defaults to first --symbols"
    )
    ap.add_argument("--test-lots", type=float, default=0.01, help="Tiny lot size for order test")
    ap.add_argument(
        "--test-direction",
        choices=["long", "short"],
        default="long",
        help="Direction for order test",
    )
    ap.add_argument(
        "--test-hold-seconds",
        type=float,
        default=1.5,
        help="Delay between open and close for order test",
    )
    ap.add_argument(
        "--test-stop-distance",
        type=float,
        default=1.0,
        help="Stop distance in price units for order test",
    )
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    symbols = _parse_symbols(args.symbols)
    data_root = (PROJECT_ROOT / args.data_root).resolve()
    results: List[Check] = []
    if args.test_order and args.ack_live != "I_UNDERSTAND_LIVE_RISK":
        results.append(
            Check(
                "live_ack",
                False,
                'missing/invalid ack; pass --ack-live "I_UNDERSTAND_LIVE_RISK"',
            )
        )
        _print_results(results)
        emit_health(component="live_wiring", status="critical", checks={"live_ack": "fail"})
        return 7

    try:
        creds = CTraderCredentials.from_env()
    except Exception as exc:
        results.append(Check("credentials", False, str(exc)))
        _print_results(results)
        emit_health(component="live_wiring", status="critical", checks={"credentials": "fail"})
        return 2

    results.append(
        Check("credentials", True, f"env={creds.environment} account={creds.account_id}")
    )
    emit_event(
        stream="live_trading",
        component="live_wiring",
        event_type="start",
        status="info",
        payload={"symbols": symbols, "environment": creds.environment},
    )

    conn = CTraderConnector(creds)
    provider = None
    started = False
    try:
        started = conn.start(timeout_s=args.timeout)
        results.append(Check("connector_auth", started, f"timeout={args.timeout:.1f}s"))
        if not started:
            _print_results(results)
            emit_health(
                component="live_wiring", status="critical", checks={"connector_auth": "fail"}
            )
            return 3

        snap = conn.get_account_snapshot(timeout_s=min(args.timeout, 10.0))
        results.append(
            Check(
                "account_snapshot",
                True,
                (
                    f"broker={snap.get('broker_name') or 'unknown'} "
                    f"login={snap.get('trader_login')} "
                    f"balance={float(snap.get('balance', 0.0) or 0.0):.2f} "
                    f"used_margin={float(snap.get('used_margin', 0.0) or 0.0):.2f}"
                ),
            )
        )

        unresolved = [
            s for s in symbols if conn.find_symbol_id(s, timeout_s=min(args.timeout, 10.0)) is None
        ]
        results.append(Check("symbol_resolution", len(unresolved) == 0, f"unresolved={unresolved}"))
        if unresolved:
            _print_results(results)
            emit_health(
                component="live_wiring",
                status="critical",
                checks={"symbol_resolution": "fail"},
                details={"unresolved": unresolved},
            )
            return 4

        provider = CTraderBarProvider(conn)
        order_dispatcher = CTraderOrderDispatcher(
            connector=conn, bar_provider=provider, fill_timeout_s=args.fill_timeout
        )
        results.append(Check("dispatcher_wiring", True, f"fill_timeout={args.fill_timeout:.1f}s"))

        bar_counts: Dict[str, int] = {s: 0 for s in symbols}
        last_close: Dict[str, float] = {}

        def _on_bar(**kwargs):
            sym = str(kwargs.get("symbol", ""))
            if sym in bar_counts:
                bar_counts[sym] += 1
                try:
                    last_close[sym] = float(kwargs.get("close"))
                except Exception:
                    pass

        for s in symbols:
            provider.subscribe(s, _on_bar)
        provider.start()
        t0 = time.monotonic()
        spreads: Dict[str, float] = {s: 0.0 for s in symbols}
        while (time.monotonic() - t0) < max(args.observe_seconds, 0.0):
            for s in symbols:
                try:
                    spreads[s] = float(provider.get_spread_pts(s))
                except Exception:
                    spreads[s] = 0.0
            time.sleep(0.2)
        provider.stop()

        any_quote = any(v > 0 for v in spreads.values())
        results.append(Check("quote_observation", any_quote, f"spreads={spreads}"))
        got_all_bars = all(bar_counts[s] > 0 for s in symbols)
        bars_ok = got_all_bars if args.require_bars else True
        results.append(
            Check(
                "bar_observation",
                bars_ok,
                f"counts={bar_counts} last_close={last_close or {}}",
            )
        )
        norm_ok, norm_detail = _evaluate_price_normalization(
            symbols=symbols,
            live_last_close=last_close,
            live_spreads=spreads,
            data_root=data_root,
        )
        results.append(Check("price_normalization", norm_ok, norm_detail))
        if args.test_order:
            sym = args.test_symbol.strip().upper() if args.test_symbol else symbols[0]
            if sym not in symbols:
                symbols.append(sym)
            if conn.find_symbol_id(sym, timeout_s=min(args.timeout, 10.0)) is None:
                results.append(Check("order_roundtrip", False, f"{sym}: unresolved symbol"))
            else:
                px = last_close.get(sym)
                if px is None or not math.isfinite(px) or px <= 0:
                    px = _load_last_csv_close(data_root, sym)
                if px is None or not math.isfinite(px) or px <= 0:
                    results.append(
                        Check("order_roundtrip", False, f"{sym}: no valid price for order test")
                    )
                else:
                    direction = (
                        TradeDirection.LONG
                        if args.test_direction.lower() == "long"
                        else TradeDirection.SHORT
                    )
                    stop_dist = float(max(args.test_stop_distance, 1e-6))
                    stop_price = (
                        float(px - stop_dist)
                        if direction == TradeDirection.LONG
                        else float(px + stop_dist)
                    )
                    lots = float(max(args.test_lots, 0.0))
                    opened = order_dispatcher.open_position(
                        symbol=sym,
                        direction=direction,
                        lots=lots,
                        price=float(px),
                        stop_price=stop_price,
                        comment="wiring_smoke_open",
                    )
                    if not opened.success or not opened.order_id:
                        results.append(
                            Check(
                                "order_roundtrip",
                                False,
                                f"{sym}: open failed (error={opened.error})",
                            )
                        )
                    else:
                        time.sleep(max(args.test_hold_seconds, 0.0))
                        close_px = last_close.get(sym, float(px))
                        closed = order_dispatcher.close_position(
                            symbol=sym,
                            order_id=str(opened.order_id),
                            price=float(close_px),
                            lots=lots,
                            comment="wiring_smoke_close",
                        )
                        ok = bool(closed.success)
                        detail = (
                            f"{sym}: opened={opened.order_id} close_ok={closed.success} "
                            f"open_fill={opened.filled_price} close_fill={closed.filled_price} "
                            f"open_err={opened.error or '-'} close_err={closed.error or '-'}"
                        )
                        results.append(Check("order_roundtrip", ok, detail))

    finally:
        try:
            if provider is not None:
                provider.stop()
        except Exception:
            pass
        if started:
            conn.stop()

    _print_results(results)
    failed_required = [r for r in results if not r.ok and r.name not in {"quote_observation"}]
    if failed_required:
        print("[NO-GO] live wiring check failed")
        emit_health(
            component="live_wiring",
            status="critical",
            checks={r.name: "fail" for r in failed_required},
        )
        return 5

    if args.require_bars and any(r.name == "bar_observation" and not r.ok for r in results):
        print("[NO-GO] live wiring check failed (bar requirement not met)")
        emit_health(
            component="live_wiring",
            status="critical",
            checks={"bar_observation": "fail"},
        )
        return 6

    if args.test_order:
        print("[GO] live wiring + order roundtrip passed")
    else:
        print("[GO] live wiring check passed (safe mode, no orders sent)")
    emit_health(
        component="live_wiring",
        status="ok",
        checks={r.name: ("pass" if r.ok else "warn") for r in results},
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
