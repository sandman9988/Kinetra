#!/usr/bin/env python3
"""Pre-live readiness gate for cTrader + Renko.

Fails closed unless all required checks pass.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from kinetra.connectors.ctrader_connector import CTraderConnector, CTraderCredentials
from kinetra.renko.live_trader import PERGate


@dataclass
class CheckResult:
    name: str
    ok: bool
    detail: str = ""


def _parse_symbols(raw: str) -> List[str]:
    out = [s.strip().upper() for s in raw.split(",") if s.strip()]
    if not out:
        raise ValueError("No symbols specified")
    return out


def _find_latest_m1_csv(symbol_dir: Path, symbol: str) -> Path | None:
    files = sorted(symbol_dir.glob(f"{symbol}_M1_*.csv"))
    return files[-1] if files else None


def _check_csv(csv_path: Path, max_spread_pts: float, min_bars: int) -> CheckResult:
    try:
        df = pd.read_csv(
            csv_path, usecols=["time", "open", "high", "low", "close", "volume", "spread"]
        )
    except Exception as exc:
        return CheckResult("csv_schema", False, f"{csv_path.name}: {exc}")

    if len(df) < min_bars:
        return CheckResult("csv_bars", False, f"{csv_path.name}: {len(df)} bars < min {min_bars}")

    med_spread = float(pd.to_numeric(df["spread"], errors="coerce").dropna().median())
    if not (med_spread > 0):
        return CheckResult("csv_spread", False, f"{csv_path.name}: non-positive spread median")
    if med_spread > max_spread_pts:
        return CheckResult(
            "csv_spread",
            False,
            f"{csv_path.name}: spread median {med_spread:.4f} > max {max_spread_pts:.4f}",
        )

    return CheckResult(
        "csv_ok", True, f"{csv_path.name}: bars={len(df)}, spread_med={med_spread:.4f}"
    )


def _check_contract_spec(spec_path: Path, max_spread_pts: float) -> CheckResult:
    try:
        spec = json.loads(spec_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return CheckResult("contract_spec", False, f"{spec_path}: {exc}")

    required = [
        "spread_typical_pts",
        "tick_size",
        "commission_per_lot",
        "swap_long_points",
        "swap_short_points",
        "contract_size",
        "volume_min",
        "volume_step",
        "volume_max",
        "broker_symbol",
    ]
    missing = [k for k in required if k not in spec]
    if missing:
        return CheckResult("contract_spec_fields", False, f"missing: {missing}")

    spread = float(spec.get("spread_typical_pts", 0.0) or 0.0)
    tick = float(spec.get("tick_size", 0.0) or 0.0)
    commission = float(spec.get("commission_per_lot", 0.0) or 0.0)
    swap_long = float(spec.get("swap_long_points", 0.0) or 0.0)
    swap_short = float(spec.get("swap_short_points", 0.0) or 0.0)
    if spread <= 0 or spread > max_spread_pts:
        return CheckResult("contract_spec_spread", False, f"spread_typical_pts={spread}")
    if tick <= 0:
        return CheckResult("contract_spec_tick", False, f"tick_size={tick}")
    return CheckResult(
        "contract_spec_ok",
        True,
        (
            f"spread={spread:.4f}, tick={tick:g}, "
            f"commission_per_lot={commission:.4f}, "
            f"swap_long={swap_long:.4f}, swap_short={swap_short:.4f}"
        ),
    )


def _print_results(results: Sequence[CheckResult]) -> None:
    for r in results:
        mark = "PASS" if r.ok else "FAIL"
        msg = f"[{mark}] {r.name}"
        if r.detail:
            msg += f" - {r.detail}"
        print(msg)


def main() -> int:
    parser = argparse.ArgumentParser(description="Pre-live readiness gate (cTrader + Renko)")
    parser.add_argument(
        "--symbols", required=True, help="Comma-separated symbols, e.g. XAUUSD,NAS100"
    )
    parser.add_argument(
        "--data-root",
        default="data/master_standardized/ctrader/pepperstone_demo_45841299",
        help="Data root containing <category>/<SYMBOL>/",
    )
    parser.add_argument("--max-spread-pts", type=float, default=1000.0)
    parser.add_argument("--min-bars", type=int, default=10000)
    parser.add_argument("--min-balance", type=float, default=0.0)
    parser.add_argument(
        "--target-gate",
        choices=[g.value for g in PERGate],
        default=PERGate.SIMULATED.value,
        help="Intended gate; LIVE requires explicit ack",
    )
    parser.add_argument(
        "--ack-live",
        default="",
        help='Must equal "I_UNDERSTAND_LIVE_RISK" when --target-gate=full',
    )
    args = parser.parse_args()

    symbols = _parse_symbols(args.symbols)
    data_root = (PROJECT_ROOT / args.data_root).resolve()
    results: List[CheckResult] = []

    # Live gate explicit acknowledgement.
    if args.target_gate == PERGate.FULL.value:
        if args.ack_live != "I_UNDERSTAND_LIVE_RISK":
            results.append(
                CheckResult(
                    "live_ack",
                    False,
                    'missing ack; pass --ack-live "I_UNDERSTAND_LIVE_RISK"',
                )
            )
            _print_results(results)
            return 2
        results.append(CheckResult("live_ack", True, "explicit live-risk acknowledgement present"))

    # Broker connectivity + account snapshot + symbol resolution.
    try:
        creds = CTraderCredentials.from_env()
        conn = CTraderConnector(creds)
    except Exception as exc:
        results.append(CheckResult("credentials", False, str(exc)))
        _print_results(results)
        return 2

    started = False
    try:
        started = conn.start(timeout_s=20.0)
        results.append(
            CheckResult(
                "connector_start", started, f"env={creds.environment} account={creds.account_id}"
            )
        )
        if not started:
            _print_results(results)
            return 2

        snap = conn.get_account_snapshot(timeout_s=10.0)
        bal = float(snap.get("balance", 0.0) or 0.0)
        bal_ok = bal >= float(args.min_balance)
        results.append(
            CheckResult(
                "account_snapshot",
                bal_ok,
                (
                    f"broker={snap.get('broker_name') or 'unknown'} "
                    f"login={snap.get('trader_login')} balance={bal:.2f} "
                    f"used_margin={float(snap.get('used_margin', 0.0) or 0.0):.2f}"
                ),
            )
        )

        unresolved = [s for s in symbols if conn.find_symbol_id(s, timeout_s=10.0) is None]
        results.append(
            CheckResult("symbol_resolution", len(unresolved) == 0, f"unresolved={unresolved}")
        )
    finally:
        if started:
            conn.stop()

    # Data/spec checks.
    for symbol in symbols:
        symbol_dir = next((p for p in data_root.glob(f"*/{symbol}") if p.is_dir()), None)
        if symbol_dir is None:
            results.append(CheckResult(f"data_path:{symbol}", False, f"missing under {data_root}"))
            continue
        results.append(CheckResult(f"data_path:{symbol}", True, str(symbol_dir)))

        csv_path = _find_latest_m1_csv(symbol_dir, symbol)
        if csv_path is None:
            results.append(CheckResult(f"m1_csv:{symbol}", False, "no *_M1_*.csv found"))
        else:
            results.append(
                _check_csv(csv_path, max_spread_pts=args.max_spread_pts, min_bars=args.min_bars)
            )

        spec_path = symbol_dir / "contract_spec.json"
        if not spec_path.exists():
            results.append(
                CheckResult(f"contract_spec:{symbol}", False, "missing contract_spec.json")
            )
        else:
            results.append(_check_contract_spec(spec_path, max_spread_pts=args.max_spread_pts))

    _print_results(results)
    failed = [r for r in results if not r.ok]
    if failed:
        print(f"[BLOCKED] pre-live readiness failed ({len(failed)} checks)")
        return 2

    print("[READY] pre-live readiness checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
