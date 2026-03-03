#!/usr/bin/env python3
"""Trading performance/readiness report from live_trades JSONL logs.

Includes:
- cumulative P/L and equity DD (USD + %)
- streaks, win/loss quality metrics
- worst daily/session loss
- exposure concentration (concurrency + symbol concentration)
- friction stress (+25%, +50%) on exact same trades
- regime stability across contiguous blocks
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass
class TradeRec:
    symbol: str
    net_usd: float
    gross_usd: float
    friction_usd: float
    entry_time: datetime | None
    exit_time: datetime | None


def _latest_log(log_dir: Path) -> Path | None:
    logs = sorted(log_dir.glob("live_trades_*.jsonl"))
    return logs[-1] if logs else None


def _streaks(pnls: list[float]) -> tuple[int, int]:
    max_w = max_l = 0
    cur_w = cur_l = 0
    for p in pnls:
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
    return max_w, max_l


def _parse_ts(raw: Any) -> datetime | None:
    if raw is None:
        return None
    s = str(raw).strip()
    if not s:
        return None
    try:
        ts = datetime.fromisoformat(s.replace("Z", "+00:00"))
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        return ts
    except Exception:
        return None


def _load_trades(log_path: Path) -> list[TradeRec]:
    out: list[TradeRec] = []
    with log_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if "net_usd" not in obj or obj.get("is_open") is not False:
                continue
            try:
                net = float(obj.get("net_usd", 0.0))
            except Exception:
                continue
            gross = float(obj.get("gross_usd", net))
            friction = float(obj.get("friction_usd", max(gross - net, 0.0)))
            out.append(
                TradeRec(
                    symbol=str(obj.get("symbol", "UNKNOWN")),
                    net_usd=net,
                    gross_usd=gross,
                    friction_usd=max(friction, 0.0),
                    entry_time=_parse_ts(obj.get("entry_time")),
                    exit_time=_parse_ts(obj.get("exit_time")),
                )
            )
    out.sort(key=lambda t: t.exit_time or datetime.min.replace(tzinfo=timezone.utc))
    return out


def _omega_ratio(returns: list[float], threshold: float = 0.0) -> float:
    up = sum(max(r - threshold, 0.0) for r in returns)
    dn = sum(max(threshold - r, 0.0) for r in returns)
    if dn <= 1e-12:
        return math.inf if up > 0 else 0.0
    return up / dn


def _basic_metrics(
    pnls: list[float], initial_equity: float, omega_threshold: float
) -> dict[str, float]:
    n = len(pnls)
    gp = sum(p for p in pnls if p > 0)
    gl = abs(sum(p for p in pnls if p < 0))
    wins = sum(1 for p in pnls if p > 0)
    losses = sum(1 for p in pnls if p < 0)
    avg_win = (gp / wins) if wins > 0 else 0.0
    avg_loss = (sum(p for p in pnls if p < 0) / losses) if losses > 0 else 0.0
    payoff = (avg_win / abs(avg_loss)) if losses > 0 and abs(avg_loss) > 1e-12 else math.inf
    pf = (gp / gl) if gl > 1e-12 else (math.inf if gp > 0 else 0.0)
    omega = _omega_ratio(pnls, threshold=omega_threshold)
    net = sum(pnls)

    eq = initial_equity
    peak = initial_equity
    max_dd_usd = 0.0
    max_dd_pct = 0.0
    worst_trade = 0.0
    for p in pnls:
        eq += p
        peak = max(peak, eq)
        dd_usd = peak - eq
        dd_pct = (dd_usd / peak) if peak > 1e-12 else 0.0
        max_dd_usd = max(max_dd_usd, dd_usd)
        max_dd_pct = max(max_dd_pct, dd_pct)
        worst_trade = min(worst_trade, p)

    return {
        "n_trades": float(n),
        "wins": float(wins),
        "losses": float(losses),
        "win_rate": (wins / n) if n > 0 else 0.0,
        "net_usd": net,
        "gross_profit_usd": gp,
        "gross_loss_abs_usd": gl,
        "profit_factor": pf,
        "omega_ratio": omega,
        "avg_win_usd": avg_win,
        "avg_loss_usd": avg_loss,
        "payoff_ratio": payoff,
        "max_dd_usd": max_dd_usd,
        "max_dd_pct": max_dd_pct,
        "ending_equity_usd": eq,
        "worst_trade_usd": worst_trade,
    }


def _worst_daily_loss(trades: list[TradeRec]) -> tuple[str, float]:
    by_day: dict[str, float] = defaultdict(float)
    for t in trades:
        if t.exit_time is None:
            continue
        k = t.exit_time.date().isoformat()
        by_day[k] += t.net_usd
    if not by_day:
        return "n/a", 0.0
    day, pnl = min(by_day.items(), key=lambda kv: kv[1])
    return day, pnl


def _max_concurrency(trades: list[TradeRec]) -> int:
    events: list[tuple[datetime, int]] = []
    for t in trades:
        if t.entry_time is None or t.exit_time is None:
            continue
        # close before open at same timestamp
        events.append((t.entry_time, +1))
        events.append((t.exit_time, -1))
    if not events:
        return 0
    events.sort(key=lambda x: (x[0], x[1]))
    cur = 0
    mx = 0
    for _, d in events:
        cur += d
        mx = max(mx, cur)
    return mx


def _symbol_concentration_hhi(trades: list[TradeRec]) -> float:
    c: dict[str, int] = defaultdict(int)
    for t in trades:
        c[t.symbol] += 1
    total = sum(c.values())
    if total <= 0:
        return 0.0
    return sum((v / total) ** 2 for v in c.values())


def _regime_blocks(trades: list[TradeRec], blocks: int) -> list[list[TradeRec]]:
    n = len(trades)
    if n == 0:
        return []
    b = max(1, blocks)
    chunk = max(1, n // b)
    out: list[list[TradeRec]] = []
    i = 0
    while i < n:
        out.append(trades[i : min(i + chunk, n)])
        i += chunk
    if len(out) > b:
        out[b - 1].extend([x for part in out[b:] for x in part])
        out = out[:b]
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Trading readiness report")
    ap.add_argument("--log-dir", default="results/renko/live")
    ap.add_argument("--log-path", default="", help="Optional explicit log JSONL path")
    ap.add_argument("--min-trades", type=int, default=30)
    ap.add_argument("--min-omega", type=float, default=1.5)
    ap.add_argument("--omega-threshold", type=float, default=0.0)
    ap.add_argument("--initial-equity", type=float, default=1000.0)
    ap.add_argument("--regime-blocks", type=int, default=3)
    ap.add_argument("--max-tolerable-dd-pct", type=float, default=20.0)
    ap.add_argument(
        "--summary-compact",
        action="store_true",
        help="Print compact 6-line summary for terminal dashboards",
    )
    args = ap.parse_args()

    log_dir = Path(args.log_dir).resolve()
    log_path = Path(args.log_path).resolve() if args.log_path else _latest_log(log_dir)
    if log_path is None:
        print(f"[INFO] No logs in {log_dir}")
        return 0

    trades = _load_trades(log_path)
    pnls: list[float] = [t.net_usd for t in trades]

    if not pnls:
        print(f"[INFO] No closed trades in {log_path}")
        return 0

    m = _basic_metrics(
        pnls,
        initial_equity=float(args.initial_equity),
        omega_threshold=float(args.omega_threshold),
    )
    max_w, max_l = _streaks(pnls)
    worst_day, worst_day_pnl = _worst_daily_loss(trades)
    max_conc = _max_concurrency(trades)
    hhi = _symbol_concentration_hhi(trades)

    ready = int(m["n_trades"]) >= args.min_trades and m["omega_ratio"] >= args.min_omega

    if args.summary_compact:
        print(f"[LOG] {log_path}")
        print(
            f"[TRADES] n={int(m['n_trades'])} wins={int(m['wins'])} losses={int(m['losses'])} "
            f"win_rate={m['win_rate']:.3f}"
        )
        print(
            f"[P/L] net_usd={m['net_usd']:.2f} gross_profit={m['gross_profit_usd']:.2f} "
            f"gross_loss_abs={m['gross_loss_abs_usd']:.2f}"
        )
        print(f"[QUALITY] omega={m['omega_ratio']:.3f} profit_factor={m['profit_factor']:.3f}")
        print(f"[STREAKS] max_consecutive_wins={max_w} max_consecutive_losses={max_l}")
        print(
            "[READINESS] "
            + ("READY_FOR_MICRO=YES" if ready else "READY_FOR_MICRO=NO")
            + f" (requires trades>={args.min_trades}, omega>={args.min_omega:.2f})"
        )
        return 0

    print(f"[LOG] {log_path}")
    print(
        f"[TRADES] n={int(m['n_trades'])} wins={int(m['wins'])} losses={int(m['losses'])} "
        f"win_rate={m['win_rate']:.3f}"
    )
    print(
        f"[P/L] net_usd={m['net_usd']:.2f} gross_profit={m['gross_profit_usd']:.2f} "
        f"gross_loss_abs={m['gross_loss_abs_usd']:.2f} ending_equity={m['ending_equity_usd']:.2f}"
    )
    print(
        f"[RISK] max_drawdown_usd={m['max_dd_usd']:.2f} max_drawdown_pct={m['max_dd_pct'] * 100:.2f}% "
        f"worst_trade_usd={m['worst_trade_usd']:.2f} worst_day={worst_day}:{worst_day_pnl:.2f}"
    )
    print(
        f"[QUALITY] omega_ratio(th={args.omega_threshold:.2f})={m['omega_ratio']:.3f} "
        f"profit_factor={m['profit_factor']:.3f} avg_win={m['avg_win_usd']:.2f} "
        f"avg_loss={m['avg_loss_usd']:.2f} payoff={m['payoff_ratio']:.3f}"
    )
    print(f"[STREAKS] max_consecutive_wins={max_w} max_consecutive_losses={max_l}")
    print(
        f"[EXPOSURE] max_simultaneous_positions={max_conc} symbol_hhi={hhi:.3f} "
        "(1.0=single-symbol concentration)"
    )

    # Cost sensitivity on exact same gross path with friction stress.
    gross = [t.gross_usd for t in trades]
    friction = [t.friction_usd for t in trades]
    base_net = [g - f for g, f in zip(gross, friction)]
    for shock in (0.25, 0.50):
        stressed = [g - (1.0 + shock) * f for g, f in zip(gross, friction)]
        ms = _basic_metrics(stressed, float(args.initial_equity), float(args.omega_threshold))
        print(
            f"[COST +{int(shock * 100)}%] net={ms['net_usd']:.2f} pf={ms['profit_factor']:.3f} "
            f"dd_usd={ms['max_dd_usd']:.2f} dd_pct={ms['max_dd_pct'] * 100:.2f}% "
            f"delta_net={ms['net_usd'] - sum(base_net):.2f}"
        )

    # Regime stability by contiguous blocks.
    blocks = _regime_blocks(trades, max(1, args.regime_blocks))
    for i, blk in enumerate(blocks, start=1):
        bp = [t.net_usd for t in blk]
        bm = _basic_metrics(bp, float(args.initial_equity), float(args.omega_threshold))
        start = blk[0].exit_time.isoformat() if blk and blk[0].exit_time else "n/a"
        end = blk[-1].exit_time.isoformat() if blk and blk[-1].exit_time else "n/a"
        print(
            f"[REGIME {i}/{len(blocks)}] {start} -> {end} "
            f"n={int(bm['n_trades'])} net={bm['net_usd']:.2f} pf={bm['profit_factor']:.3f} "
            f"dd_usd={bm['max_dd_usd']:.2f} dd_pct={bm['max_dd_pct'] * 100:.2f}%"
        )

    # Practical micro launch constraints from streak profile.
    brake_at = max(8, int(round(max_l * 0.8))) if max_l > 0 else 8
    flat_at = max(12, int(round(max_l * 1.2))) if max_l > 0 else 12
    per_trade_risk_cap_pct = float(args.max_tolerable_dd_pct) / 20.0
    print(
        f"[MICRO RULES] daily_loss_cap=1.0-2.0% | loss_brake_at={brake_at} | "
        f"flat_after={flat_at} | per_trade_risk_cap<={per_trade_risk_cap_pct:.2f}% "
        f"(for 20-loss survival at maxDD={args.max_tolerable_dd_pct:.1f}%)"
    )

    print(
        "[READINESS] "
        + ("TRADING_READY_FOR_MICRO=YES" if ready else "TRADING_READY_FOR_MICRO=NO")
        + f" (requires trades>={args.min_trades}, omega_ratio>={args.min_omega:.2f})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
