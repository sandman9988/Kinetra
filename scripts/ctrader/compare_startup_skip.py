#!/usr/bin/env python3
"""A/B compare startup flip-skip settings for paper gold replay.

Runs two back-to-back paper replays on the same config:
- startup_skip_flips=0
- startup_skip_flips=N (default 2)

Then parses each resulting trade log and prints a compact delta table.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class Metrics:
    log_path: Path
    n_trades: int
    net_usd: float
    gross_profit_usd: float
    gross_loss_abs_usd: float
    win_rate: float
    profit_factor: float
    omega_ratio: float
    max_dd_usd: float
    max_dd_pct: float
    max_consecutive_losses: int
    max_consecutive_wins: int
    cost_share_pct: float


def _run_variant(args: argparse.Namespace, startup_skip_flips: int) -> Path:
    cmd = [
        sys.executable,
        "scripts/ctrader/run_paper_gold.py",
        "--symbol",
        args.symbol,
        "--data-root",
        args.data_root,
        "--brick-size",
        str(args.brick_size),
        "--stop-bricks",
        str(args.stop_bricks),
        "--paper-lots",
        str(args.paper_lots),
        "--drawdown-halt-pct",
        str(args.drawdown_halt_pct),
        "--startup-skip-flips",
        str(startup_skip_flips),
        "--log-level",
        args.log_level,
    ]
    if args.timeout_seconds > 0:
        cmd.extend(["--timeout-seconds", str(args.timeout_seconds)])

    proc = subprocess.run(
        cmd,
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"Variant startup_skip_flips={startup_skip_flips} failed (rc={proc.returncode}).\n"
            f"STDERR:\n{proc.stderr[-2000:]}\nSTDOUT:\n{proc.stdout[-2000:]}"
        )

    marker = "[OK] Trader log:"
    log_path: Path | None = None
    for line in (proc.stdout + "\n" + proc.stderr).splitlines():
        if marker in line:
            tail = line.split(marker, 1)[1].strip()
            if tail:
                log_path = Path(tail)
    if log_path is None or not log_path.exists():
        raise RuntimeError(
            f"Could not locate trader log path for startup_skip_flips={startup_skip_flips}"
        )
    return log_path


def _parse_metrics(log_path: Path, initial_equity: float, omega_threshold: float) -> Metrics:
    pnls: list[float] = []
    frictions: list[float] = []
    max_w = max_l = 0
    cur_w = cur_l = 0

    with log_path.open("r", encoding="utf-8") as fh:
        for raw in fh:
            raw = raw.strip()
            if not raw:
                continue
            try:
                obj: dict[str, Any] = json.loads(raw)
            except Exception:
                continue
            if "net_usd" not in obj or obj.get("is_open") is not False:
                continue
            try:
                p = float(obj.get("net_usd", 0.0))
            except Exception:
                continue
            f = float(obj.get("friction_usd", 0.0) or 0.0)
            pnls.append(p)
            frictions.append(max(f, 0.0))
            if p > 0:
                cur_w += 1
                cur_l = 0
                max_w = max(max_w, cur_w)
            elif p < 0:
                cur_l += 1
                cur_w = 0
                max_l = max(max_l, cur_l)
            else:
                cur_w = 0
                cur_l = 0

    if not pnls:
        raise RuntimeError(f"No closed trades in {log_path}")

    n = len(pnls)
    gp = sum(p for p in pnls if p > 0)
    gl = abs(sum(p for p in pnls if p < 0))
    net = sum(pnls)
    wins = sum(1 for p in pnls if p > 0)
    wr = wins / n
    pf = gp / gl if gl > 1e-12 else (math.inf if gp > 0 else 0.0)

    up = sum(max(p - omega_threshold, 0.0) for p in pnls)
    dn = sum(max(omega_threshold - p, 0.0) for p in pnls)
    omega = up / dn if dn > 1e-12 else (math.inf if up > 0 else 0.0)

    eq = initial_equity
    peak = initial_equity
    max_dd_usd = 0.0
    max_dd_pct = 0.0
    for p in pnls:
        eq += p
        if eq > peak:
            peak = eq
        dd = peak - eq
        max_dd_usd = max(max_dd_usd, dd)
        if peak > 1e-12:
            max_dd_pct = max(max_dd_pct, dd / peak)

    total_friction = sum(frictions)
    gross_abs = sum(abs(p) for p in pnls)
    cost_share = (total_friction / gross_abs) if gross_abs > 1e-12 else 0.0

    return Metrics(
        log_path=log_path,
        n_trades=n,
        net_usd=net,
        gross_profit_usd=gp,
        gross_loss_abs_usd=gl,
        win_rate=wr,
        profit_factor=pf,
        omega_ratio=omega,
        max_dd_usd=max_dd_usd,
        max_dd_pct=max_dd_pct,
        max_consecutive_losses=max_l,
        max_consecutive_wins=max_w,
        cost_share_pct=cost_share * 100.0,
    )


def _fmt(m: Metrics) -> str:
    return (
        f"trades={m.n_trades} net={m.net_usd:.2f} wr={m.win_rate:.3f} "
        f"pf={m.profit_factor:.3f} omega={m.omega_ratio:.3f} "
        f"dd={m.max_dd_usd:.2f} ({m.max_dd_pct * 100:.2f}%) "
        f"Lstreak={m.max_consecutive_losses} cost_share={m.cost_share_pct:.2f}%"
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="A/B compare startup skip flips")
    ap.add_argument("--symbol", default="XAUUSD")
    ap.add_argument(
        "--data-root",
        default="data/master_standardized/ctrader/pepperstone_demo_45841299",
    )
    ap.add_argument("--brick-size", type=float, default=1.0)
    ap.add_argument("--stop-bricks", type=float, default=1.0)
    ap.add_argument("--paper-lots", type=float, default=0.01)
    ap.add_argument("--drawdown-halt-pct", type=float, default=1.0)
    ap.add_argument("--timeout-seconds", type=float, default=0.0)
    ap.add_argument("--omega-threshold", type=float, default=0.0)
    ap.add_argument("--initial-equity", type=float, default=1000.0)
    ap.add_argument("--skip-a", type=int, default=0, help="Variant A startup_skip_flips")
    ap.add_argument("--skip-b", type=int, default=2, help="Variant B startup_skip_flips")
    ap.add_argument(
        "--log-level",
        default="WARNING",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Use WARNING to keep replay fast",
    )
    args = ap.parse_args()

    print(f"[RUN] Variant A startup_skip_flips={args.skip_a}")
    log_a = _run_variant(args, args.skip_a)
    print(f"[RUN] Variant B startup_skip_flips={args.skip_b}")
    log_b = _run_variant(args, args.skip_b)

    m_a = _parse_metrics(log_a, args.initial_equity, args.omega_threshold)
    m_b = _parse_metrics(log_b, args.initial_equity, args.omega_threshold)

    print("")
    print(f"[A skip={args.skip_a}] {m_a.log_path}")
    print(_fmt(m_a))
    print(f"[B skip={args.skip_b}] {m_b.log_path}")
    print(_fmt(m_b))
    print("")
    print("[DELTA B-A]")
    print(f"trades={m_b.n_trades - m_a.n_trades:+d}")
    print(f"net_usd={m_b.net_usd - m_a.net_usd:+.2f}")
    print(f"win_rate={m_b.win_rate - m_a.win_rate:+.3f}")
    print(f"profit_factor={m_b.profit_factor - m_a.profit_factor:+.3f}")
    print(f"omega_ratio={m_b.omega_ratio - m_a.omega_ratio:+.3f}")
    print(f"max_dd_usd={m_b.max_dd_usd - m_a.max_dd_usd:+.2f}")
    print(f"max_dd_pct={(m_b.max_dd_pct - m_a.max_dd_pct) * 100:+.2f}%")
    print(f"max_consecutive_losses={m_b.max_consecutive_losses - m_a.max_consecutive_losses:+d}")
    print(f"cost_share_pct={m_b.cost_share_pct - m_a.cost_share_pct:+.2f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
