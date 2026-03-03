#!/usr/bin/env python3
"""cTrader connectivity preflight (Pepperstone/Open API).

Checks:
1. Credentials load from .env/.env.openapi
2. DNS hardening candidate expansion
3. TCP reachability probe on endpoint candidates
4. Connector start/auth
5. Symbol resolution (default: XAUUSD)
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from kinetra.connectors.ctrader_connector import CTraderConnector, CTraderCredentials
from kinetra.dns_hardening import expand_endpoint_candidates, select_reachable_endpoint
from kinetra.monitoring import emit_event, emit_health


def _parse_env_file(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        s = line.strip()
        if not s or s.startswith("#") or "=" not in s:
            continue
        key, _, val = s.partition("=")
        out[key.strip()] = val.strip().strip('"').strip("'")
    return out


def _env_or_file(key: str, env_data: dict[str, str], default: str = "") -> str:
    return (os.getenv(key) or env_data.get(key) or default).strip()


def _build_candidates(creds: CTraderCredentials) -> tuple[list[str], int]:
    # Keep this in sync with connector defaults.
    from ctrader_open_api import EndPoints

    primary = (
        EndPoints.PROTOBUF_DEMO_HOST
        if creds.environment.lower() == "demo"
        else EndPoints.PROTOBUF_LIVE_HOST
    )
    env_data = _parse_env_file(PROJECT_ROOT / ".env")
    env_data.update(_parse_env_file(PROJECT_ROOT / ".env.openapi"))
    alt = [
        p.strip() for p in _env_or_file("CTRADER_ALT_ENDPOINTS", env_data).split(",") if p.strip()
    ]
    include_ips = _env_or_file(
        "CTRADER_INCLUDE_RESOLVED_IP_FALLBACKS", env_data, "false"
    ).lower() in {"1", "true", "yes", "on"}
    candidates = expand_endpoint_candidates(
        [primary, *alt],
        include_resolved_ips=include_ips,
        service_name=f"ctrader-{creds.environment}",
    )
    return candidates, EndPoints.PROTOBUF_PORT


def main() -> int:
    parser = argparse.ArgumentParser(description="cTrader Open API preflight")
    parser.add_argument("--symbol", default="XAUUSD", help="Symbol to resolve after auth")
    parser.add_argument("--timeout", type=float, default=20.0, help="Connect/auth timeout seconds")
    parser.add_argument(
        "--probe-timeout", type=float, default=2.0, help="TCP probe timeout seconds"
    )
    args = parser.parse_args()

    try:
        creds = CTraderCredentials.from_env()
    except Exception as exc:
        print(f"[FAIL] Credentials: {exc}")
        return 2

    print(f"[OK] Credentials loaded (env={creds.environment}, account_id={creds.account_id})")
    emit_event(
        stream="system",
        component="ctrader_connect",
        event_type="preflight_start",
        status="info",
        payload={"environment": creds.environment, "account_id": creds.account_id},
    )

    candidates, port = _build_candidates(creds)
    if not candidates:
        print("[FAIL] No DNS-validated endpoint candidates")
        emit_health(
            component="ctrader_connect",
            status="critical",
            checks={"dns_candidates": "fail"},
        )
        return 3
    print("[INFO] Endpoint candidates:")
    for c in candidates:
        print(f"  - {c}:{port}")

    selected = select_reachable_endpoint(
        candidates,
        port=port,
        timeout_s=args.probe_timeout,
        service_name=f"ctrader-{creds.environment}",
    )
    if not selected:
        print(
            "[WARN] No probe-reachable endpoint candidates; continuing with connector "
            "direct-connect fallback"
        )
        emit_health(
            component="ctrader_connect",
            status="warn",
            checks={"endpoint_probe": "fail"},
            details={"candidates": candidates},
        )
    else:
        print(f"[OK] Reachable endpoint selected: {selected}:{port}")

    conn = CTraderConnector(creds)
    try:
        ok = conn.start(timeout_s=args.timeout)
        if not ok:
            print("[FAIL] Connector auth/start failed")
            emit_health(
                component="ctrader_connect",
                status="critical",
                checks={"connector_auth": "fail"},
            )
            return 5
        print("[OK] Connector authenticated")
        snap = conn.get_account_snapshot(timeout_s=min(args.timeout, 10.0))
        print(
            "[OK] Account snapshot: "
            f"broker={snap.get('broker_name') or 'unknown'} "
            f"login={snap.get('trader_login')} "
            f"balance={snap.get('balance'):.2f} "
            f"used_margin={snap.get('used_margin'):.2f}"
        )
        sid = conn.find_symbol_id(args.symbol, timeout_s=min(args.timeout, 10.0))
        if sid is None:
            print(f"[FAIL] Symbol resolve failed: {args.symbol}")
            return 6
        print(f"[OK] Symbol resolved: {args.symbol} -> {sid}")
        print("[PASS] cTrader preflight complete")
        emit_health(
            component="ctrader_connect",
            status="ok",
            checks={
                "dns_candidates": "pass",
                "connector_auth": "pass",
                "symbol_resolution": "pass",
            },
            details={"symbol": args.symbol, "symbol_id": sid},
        )
        return 0
    finally:
        conn.stop()


if __name__ == "__main__":
    raise SystemExit(main())
