#!/usr/bin/env python3
"""A/B paper-trade backtest: single symbol vs dual symbol within SMALL gate confines.

Scenario A — baseline
    Single symbol (default: XAUUSD).

Scenario B — dual
    Two symbols (default: XAUUSD + NAS100) sharing one portfolio with a
    combined drawdown circuit breaker, matching the SMALL PER gate limits.

Why SMALL gate, not MICRO?
    NAS100 volume_min = 0.10 lots (Pepperstone contract spec).
    MICRO gate ceiling = 0.01 lots → cannot trade NAS100.
    SMALL gate ceiling = 0.10 lots → fits both symbols.

Lot sizing in paper mode
    gate=PERGate.SIMULATED, paper_lots=0.01.
    _quantize_lots rounds paper_lots to each symbol's volume_step:
      XAUUSD → 0.01 lots (volume_step=0.01, volume_min=0.01)
      NAS100 → 0.10 lots (volume_step=0.10, volume_min=0.10)
    Absolute P&L figures differ between symbols by design.
    Signal-quality metrics (omega, win_rate, profit_factor) are lot-size-
    independent and are the primary comparison targets.

Combined constraint enforced
    drawdown_halt_pct (default 0.05) stops BOTH symbols if portfolio equity
    drops more than threshold × initial_equity_usd.  This mirrors the SMALL
    gate 5% max-DD rule applied to the shared portfolio.

Usage
    python scripts/ctrader/ab_dual_symbol_backtest.py
    python scripts/ctrader/ab_dual_symbol_backtest.py \\
        --sym-a XAUUSD --brick-a 1.0 \\
        --sym-b NAS100 --brick-b 5.0 \\
        --drawdown-halt-pct 0.05 \\
        --initial-equity 1000
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from kinetra.renko.live_trader import (
    HistoricalBarProvider,
    LiveTraderConfig,
    PaperDispatcher,
    PERGate,
    RenkoLiveTrader,
)

LOG = logging.getLogger("ctrader.ab_dual_backtest")

# ── defaults ─────────────────────────────────────────────────────────────────

_DEFAULT_DATA_ROOT = "data/master_standardized/ctrader/pepperstone_demo_45841299"
_DEFAULT_SYM_A = "XAUUSD"
_DEFAULT_SYM_B = "NAS100"
_DEFAULT_BRICK_A = 1.0
_DEFAULT_BRICK_B = 5.0
# Ordered list of candidate directories that may contain real qualification.json files.
# The script copies whichever it finds first into the temp qual_base.
_DEFAULT_QUAL_SEARCH_DIRS: List[str] = [
    "outputs/renko_output",
    "data/renko_qualified",
]

# Fallback contract fields when contract_spec.json is absent.
_CONTRACT_DEFAULTS: Dict[str, Dict[str, float]] = {
    "XAUUSD": {
        "tick_size": 0.01,
        "contract_size": 100.0,
        "volume_min": 0.01,
        "volume_step": 0.01,
        "volume_max": 100.0,
    },
    "NAS100": {
        "tick_size": 0.1,
        "contract_size": 100.0,
        "volume_min": 0.1,
        "volume_step": 0.1,
        "volume_max": 100.0,
    },
}

_GENERIC_FALLBACK: Dict[str, float] = {
    "tick_size": 0.01,
    "contract_size": 100.0,
    "volume_min": 0.01,
    "volume_step": 0.01,
    "volume_max": 100.0,
}


# ── helpers ──────────────────────────────────────────────────────────────────


def _safe_pos(x: Any, default: float) -> float:
    try:
        v = float(x)
    except Exception:
        return float(default)
    return float(default) if not math.isfinite(v) or v <= 0 else v


def _load_contract(symbol_dir: Path, symbol: str) -> Dict[str, float]:
    spec_path = symbol_dir / "contract_spec.json"
    raw: Dict[str, Any] = {}
    if spec_path.exists():
        try:
            raw = json.loads(spec_path.read_text(encoding="utf-8"))
        except Exception:
            pass
    fallback = _CONTRACT_DEFAULTS.get(symbol.upper(), _GENERIC_FALLBACK)
    return {
        "tick_size": _safe_pos(raw.get("tick_size") or raw.get("tickSize"), fallback["tick_size"]),
        "contract_size": _safe_pos(
            raw.get("contract_size") or raw.get("usd_per_price_unit"), fallback["contract_size"]
        ),
        "volume_min": _safe_pos(raw.get("volume_min"), fallback["volume_min"]),
        "volume_step": _safe_pos(raw.get("volume_step"), fallback["volume_step"]),
        "volume_max": _safe_pos(raw.get("volume_max"), fallback["volume_max"]),
    }


def _resolve_symbol_dir(data_root: Path, symbol: str) -> Path:
    matches = sorted(p for p in data_root.glob(f"*/{symbol}") if p.is_dir())
    if not matches:
        raise FileNotFoundError(f"{symbol}: no folder found under {data_root}")
    with_spec = [m for m in matches if (m / "contract_spec.json").exists()]
    return sorted(with_spec or matches)[-1]


def _load_csv(symbol_dir: Path, symbol: str) -> pd.DataFrame:
    files = sorted(symbol_dir.glob(f"{symbol}_M1_*.csv"))
    if not files:
        raise FileNotFoundError(f"No {symbol}_M1_*.csv in {symbol_dir}")
    path = files[-1]
    df = pd.read_csv(path)
    lower = {c.lower(): c for c in df.columns}
    time_col = next((lower[k] for k in ("time", "datetime", "date") if k in lower), None)
    close_col = lower.get("close")
    if time_col is None or close_col is None:
        raise ValueError(f"CSV must have time+close columns: {path}")
    out = pd.DataFrame(
        {
            "time": pd.to_datetime(df[time_col], utc=True, errors="coerce"),
            "close": pd.to_numeric(df[close_col], errors="coerce"),
        }
    )
    out = out.dropna(subset=["time", "close"]).sort_values("time").reset_index(drop=True)
    LOG.info(
        "[CSV] %s: %d rows  %s → %s", symbol, len(out), out["time"].iloc[0], out["time"].iloc[-1]
    )
    return out


def _trim_overlap(frames: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """Trim all frames to their common overlapping date window."""
    if len(frames) < 2:
        return frames
    start = max(df["time"].iloc[0] for df in frames.values())
    end = min(df["time"].iloc[-1] for df in frames.values())
    if start >= end:
        raise ValueError(f"No overlapping date window: start={start} end={end}")
    trimmed = {
        sym: df[(df["time"] >= start) & (df["time"] <= end)].reset_index(drop=True)
        for sym, df in frames.items()
    }
    for sym, df in trimmed.items():
        LOG.info(
            "[OVERLAP] %s: %d rows  %s → %s", sym, len(df), df["time"].iloc[0], df["time"].iloc[-1]
        )
    return trimmed


def _install_qual(
    *,
    symbol: str,
    brick_size: float,
    contract: Dict[str, float],
    qual_base: Path,
    qual_search_dirs: List[Path],
) -> None:
    """Write qualification.json into qual_base for this symbol.

    ``brick_size`` from the caller is ALWAYS authoritative — it is never
    overridden by any value found in an on-disk qualification file.

    If a real qualification file exists (qualified=True), its filter_params
    (fliprate/markov thresholds and windows) are reused so that the replay
    sees the same signal gates as the live stack.  The brick_size field in
    the written file is patched to the caller-supplied value.

    If no real qualification is found, a synthetic file is written using
    fliprate_threshold=1.0 (gate disabled) and markov_threshold=0.55.
    """
    sym_dir = qual_base / symbol
    sym_dir.mkdir(parents=True, exist_ok=True)

    # Try real qualification files — borrow filter_params only
    for search_dir in qual_search_dirs:
        q_path = search_dir / symbol / "qualification.json"
        if not q_path.exists():
            continue
        try:
            raw = json.loads(q_path.read_text(encoding="utf-8"))
            if not raw.get("qualified", False):
                LOG.debug("[QUAL] %s: real qual at %s is qualified=False, skipping", symbol, q_path)
                continue

            # Patch brick_size to the empirically validated CLI value — never
            # inherit the pipeline's VR-optimised brick_size from disk.
            raw["brick_size"] = float(brick_size)

            fp = raw.get("filter_params", {})
            (sym_dir / "qualification.json").write_text(json.dumps(raw, indent=2), encoding="utf-8")
            sp_src = search_dir / symbol / "session_profile.json"
            sp_dst = sym_dir / "session_profile.json"
            if sp_src.exists():
                shutil.copy2(sp_src, sp_dst)
            else:
                sbm = raw.get("session_break_minutes", 30.0)
                sp_dst.write_text(
                    json.dumps({"session_break_minutes": float(sbm)}, indent=2),
                    encoding="utf-8",
                )
            LOG.info(
                "[QUAL] %s: real filter_params from %s "
                "(brick=%.4f [CLI], fr_thr=%.3f, mk_thr=%.3f)",
                symbol,
                q_path,
                brick_size,
                fp.get("fliprate_threshold", "?"),
                fp.get("markov_threshold", "?"),
            )
            return
        except Exception as exc:
            LOG.warning("[QUAL] %s: failed to read %s: %s", symbol, q_path, exc)

    # Fall back: write a synthetic qualification
    LOG.warning(
        "[QUAL] %s: no real qualification found — writing synthetic "
        "(brick=%.4f, fr_thr=1.0 [gate disabled], mk_thr=0.55)",
        symbol,
        brick_size,
    )
    qual = {
        "symbol": symbol,
        "qualified": True,
        "disqualified": False,
        "disqualification_reason": "",
        "broker_source": "ctrader",
        "cluster": "precious_metals" if "XAU" in symbol else "indices",
        "brick_size": float(brick_size),
        "omega": 9.99,
        "z_factor": 9.99,
        "n_trades": 999,
        "filter_params": {
            "fliprate_window": 50,
            "fliprate_threshold": 1.0,  # gate disabled — no real qual data
            "markov_window": 50,
            "markov_threshold": 0.55,
        },
        "usd_per_point": float(contract["contract_size"]),
        "tick_size": float(contract["tick_size"]),
        "volume_min": float(contract["volume_min"]),
        "volume_step": float(contract["volume_step"]),
        "volume_max": float(contract["volume_max"]),
        "pipeline_version": "ab_paper_synthetic",
        "recalibration_due": False,
    }
    (sym_dir / "qualification.json").write_text(json.dumps(qual, indent=2), encoding="utf-8")
    (sym_dir / "session_profile.json").write_text(
        json.dumps({"session_break_minutes": 30.0}, indent=2), encoding="utf-8"
    )


def _wait_for_replay(provider: HistoricalBarProvider, poll: float = 0.1) -> None:
    while True:
        th = getattr(provider, "_thread", None)
        if th is None or not th.is_alive():
            return
        time.sleep(poll)


def _run_scenario(
    *,
    label: str,
    frames: Dict[str, pd.DataFrame],
    qual_base: Path,
    drawdown_halt_pct: float,
    initial_equity: float,
    monday_open: str,
    friday_close: str,
    loss_brake: int,
    loss_flat: int,
    loss_pause: float,
    startup_skip: int,
    contracts: Dict[str, Dict[str, float]],
    effective_brick_sizes: Dict[str, float],
) -> Dict[str, Any]:
    LOG.info("[%s] Starting scenario with symbols: %s", label, list(frames.keys()))
    provider = HistoricalBarProvider(
        {sym: df[["time", "close"]] for sym, df in frames.items()},
        speed_multiplier=float("inf"),
    )
    dispatcher = PaperDispatcher()

    # Use the smallest volume_step across symbols as lot_step for the config.
    lot_step = min(c["volume_step"] for c in contracts.values())
    min_lots = min(c["volume_min"] for c in contracts.values())

    cfg = LiveTraderConfig(
        symbols=list(frames.keys()),
        gate=PERGate.SIMULATED,
        stop_bricks=1.0,
        qual_dir=qual_base,
        broker_source="ctrader",
        lot_step=lot_step,
        min_lots=min_lots,
        poll_interval_seconds=0.05,
        drawdown_halt_pct=float(drawdown_halt_pct),
        initial_equity_usd=float(initial_equity),
        paper_lots=0.01,
        startup_skip_flips=int(startup_skip),
        monday_open_utc=str(monday_open),
        friday_close_utc=str(friday_close),
        loss_brake_after_consecutive_losses=int(max(loss_brake, 1)),
        loss_flat_after_consecutive_losses=int(max(loss_flat, loss_brake + 1)),
        loss_pause_minutes=float(max(loss_pause, 1.0)),
    )
    trader = RenkoLiveTrader(config=cfg, bar_provider=provider, dispatcher=dispatcher)
    trader.start()
    _wait_for_replay(provider)
    trader.stop()
    summary = trader.session_summary()
    summary["_scenario_label"] = label
    summary["_symbols"] = list(frames.keys())
    summary["_effective_brick_sizes"] = effective_brick_sizes
    LOG.info(
        "[%s] Done: trades=%d pnl=%.2f omega=%.3f halted=%s",
        label,
        summary["n_completed_trades"],
        summary["session_pnl_usd"],
        summary["omega"],
        summary["is_halted"],
    )
    return summary


# ── output ───────────────────────────────────────────────────────────────────


def _fmt(v: Any, fmt: str = ".3f") -> str:
    if isinstance(v, float) and math.isinf(v):
        return "∞"
    if isinstance(v, float) and math.isnan(v):
        return "n/a"
    try:
        return format(v, fmt)
    except Exception:
        return str(v)


def _print_report(a: Dict[str, Any], b: Dict[str, Any]) -> None:
    hr = "─" * 72
    print()
    print("  A/B DUAL-SYMBOL PAPER BACKTEST REPORT")
    print(hr)
    bsz_a = a.get("_effective_brick_sizes", {})
    bsz_b = b.get("_effective_brick_sizes", {})
    print(f"  Scenario A  : {a['_scenario_label']}  symbols={a['_symbols']}")
    for sym, bsz in bsz_a.items():
        print(f"               brick_size[{sym}] = {bsz:.4f}")
    print(f"  Scenario B  : {b['_scenario_label']}  symbols={b['_symbols']}")
    for sym, bsz in bsz_b.items():
        print(f"               brick_size[{sym}] = {bsz:.4f}")
    print(hr)

    # Portfolio-level comparison
    rows = [
        ("n_trades", "n_completed_trades", ".0f"),
        ("session_pnl_usd", "session_pnl_usd", ".2f"),
        ("omega", "omega", ".4f"),
        ("profit_factor", "profit_factor", ".4f"),
        ("win_rate", "win_rate", ".4f"),
        ("gross_profit", "gross_profit_usd", ".2f"),
        ("gross_loss_abs", "gross_loss_abs_usd", ".2f"),
        ("portfolio_dd", "portfolio_drawdown", ".4f"),
        ("max_cons_wins", "max_consecutive_wins", ".0f"),
        ("max_cons_losses", "max_consecutive_losses", ".0f"),
        ("is_halted", "is_halted", ""),
        ("can_advance_gate", "can_advance_gate", ""),
    ]

    print(f"\n  {'Metric':<28} {'A':>14} {'B':>14} {'Δ (B−A)':>12}")
    print(f"  {'─' * 28} {'─' * 14} {'─' * 14} {'─' * 12}")
    for label, key, fmt in rows:
        va = a.get(key)
        vb = b.get(key)
        if isinstance(va, float) and isinstance(vb, float) and fmt:
            delta = vb - va
            print(f"  {label:<28} {_fmt(va, fmt):>14} {_fmt(vb, fmt):>14} {_fmt(delta, fmt):>12}")
        else:
            print(f"  {label:<28} {str(va):>14} {str(vb):>14} {'':>12}")

    # Per-instrument breakdown for Scenario B
    per = b.get("per_instrument", {})
    if per:
        print(f"\n  {'─' * 72}")
        print("  Scenario B — per-instrument breakdown")
        print(f"  {'─' * 72}")
        print(
            f"  {'Symbol':<12} {'Trades':>8} {'Net USD':>12} {'Win Rate':>10} {'Open':>8} {'Cons.L':>8}"
        )
        print(f"  {'─' * 12} {'─' * 8} {'─' * 12} {'─' * 10} {'─' * 8} {'─' * 8}")
        for sym, st in per.items():
            print(
                f"  {sym:<12} {st['n_trades']:>8} {st['net_usd']:>12.2f} {st['win_rate']:>10.4f}"
                f" {str(st['open_position']):>8} {st['consecutive_losses']:>8}"
            )

    # Per-instrument breakdown for Scenario A
    per_a = a.get("per_instrument", {})
    if per_a:
        print(f"\n  {'─' * 72}")
        print("  Scenario A — per-instrument breakdown")
        print(f"  {'─' * 72}")
        print(
            f"  {'Symbol':<12} {'Trades':>8} {'Net USD':>12} {'Win Rate':>10} {'Open':>8} {'Cons.L':>8}"
        )
        print(f"  {'─' * 12} {'─' * 8} {'─' * 12} {'─' * 10} {'─' * 8} {'─' * 8}")
        for sym, st in per_a.items():
            print(
                f"  {sym:<12} {st['n_trades']:>8} {st['net_usd']:>12.2f} {st['win_rate']:>10.4f}"
                f" {str(st['open_position']):>8} {st['consecutive_losses']:>8}"
            )

    # Notes
    print(f"\n  {'─' * 72}")
    print("  Notes")
    print("  ─ paper_lots=0.01 for both scenarios (NAS100 quantises to 0.10 lots")
    print("    via volume_step; absolute P&L not directly comparable across symbols).")
    print("  ─ Signal-quality metrics (omega, win_rate, profit_factor) are")
    print("    lot-size-independent and are the primary A/B comparison targets.")
    print("  ─ drawdown_halt_pct is a shared portfolio circuit breaker (SMALL gate")
    print("    equivalent: 5%). Both symbols halt if the combined portfolio DD fires.")
    print()


# ── CLI ──────────────────────────────────────────────────────────────────────


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="A/B dual-symbol paper backtest")
    p.add_argument("--data-root", default=_DEFAULT_DATA_ROOT)
    p.add_argument("--sym-a", default=_DEFAULT_SYM_A)
    p.add_argument("--sym-b", default=_DEFAULT_SYM_B)
    p.add_argument("--brick-a", type=float, default=_DEFAULT_BRICK_A)
    p.add_argument("--brick-b", type=float, default=_DEFAULT_BRICK_B)
    p.add_argument(
        "--drawdown-halt-pct",
        type=float,
        default=0.05,
        help="Combined portfolio DD halt fraction (default 0.05 = SMALL gate 5%)",
    )
    p.add_argument("--initial-equity", type=float, default=1000.0)
    p.add_argument("--monday-open-utc", default="00:00")
    p.add_argument("--friday-close-utc", default="23:59")
    p.add_argument("--startup-skip-flips", type=int, default=2)
    p.add_argument("--loss-brake-after", type=int, default=8)
    p.add_argument("--loss-flat-after", type=int, default=12)
    p.add_argument("--loss-pause-minutes", type=float, default=120.0)
    p.add_argument("--output-json", default="", help="Optional path to write results JSON")
    p.add_argument(
        "--qual-dirs",
        nargs="+",
        default=_DEFAULT_QUAL_SEARCH_DIRS,
        help="Ordered list of directories to search for real qualification.json "
        "files (first match wins). Falls back to synthetic if none found.",
    )
    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    data_root = (PROJECT_ROOT / args.data_root).resolve()
    sym_a = args.sym_a.strip().upper()
    sym_b = args.sym_b.strip().upper()
    if sym_a == sym_b:
        raise ValueError(f"--sym-a and --sym-b must differ (both={sym_a})")

    # Resolve directories and load data
    dir_a = _resolve_symbol_dir(data_root, sym_a)
    dir_b = _resolve_symbol_dir(data_root, sym_b)
    contract_a = _load_contract(dir_a, sym_a)
    contract_b = _load_contract(dir_b, sym_b)
    df_a = _load_csv(dir_a, sym_a)
    df_b = _load_csv(dir_b, sym_b)

    # Trim both to overlapping window for fair comparison
    trimmed = _trim_overlap({sym_a: df_a, sym_b: df_b})
    df_a_t = trimmed[sym_a]
    df_b_t = trimmed[sym_b]

    qual_search_dirs = [(PROJECT_ROOT / d).resolve() for d in args.qual_dirs]

    # CLI brick sizes are authoritative — never overridden by on-disk values
    brick_sizes = {sym_a: args.brick_a, sym_b: args.brick_b}

    with tempfile.TemporaryDirectory(prefix="kinetra_ab_qual_") as tmp:
        qual_base = Path(tmp)

        # Install qualification profiles (filter_params from disk if available;
        # brick_size always from CLI)
        for sym, bsz, ctr in [
            (sym_a, args.brick_a, contract_a),
            (sym_b, args.brick_b, contract_b),
        ]:
            _install_qual(
                symbol=sym,
                brick_size=bsz,
                contract=ctr,
                qual_base=qual_base,
                qual_search_dirs=qual_search_dirs,
            )

        common_kwargs = dict(
            qual_base=qual_base,
            drawdown_halt_pct=args.drawdown_halt_pct,
            initial_equity=args.initial_equity,
            monday_open=args.monday_open_utc,
            friday_close=args.friday_close_utc,
            loss_brake=args.loss_brake_after,
            loss_flat=args.loss_flat_after,
            loss_pause=args.loss_pause_minutes,
            startup_skip=args.startup_skip_flips,
        )

        # Scenario A — single symbol (trimmed to overlap window)
        summary_a = _run_scenario(
            label=f"A ({sym_a} only)",
            frames={sym_a: df_a_t},
            contracts={sym_a: contract_a},
            effective_brick_sizes={sym_a: brick_sizes[sym_a]},
            **common_kwargs,
        )

        # Scenario B — dual symbols
        summary_b = _run_scenario(
            label=f"B ({sym_a}+{sym_b})",
            frames={sym_a: df_a_t, sym_b: df_b_t},
            contracts={sym_a: contract_a, sym_b: contract_b},
            effective_brick_sizes=brick_sizes,
            **common_kwargs,
        )

    _print_report(summary_a, summary_b)

    if args.output_json:
        out_path = Path(args.output_json)
        out_path.write_text(
            json.dumps({"scenario_a": summary_a, "scenario_b": summary_b}, indent=2, default=str),
            encoding="utf-8",
        )
        print(f"  Results saved: {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
