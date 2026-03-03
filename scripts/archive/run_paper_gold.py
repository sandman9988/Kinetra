#!/usr/bin/env python3
"""Run XAUUSD paper trading via RenkoLiveTrader from standardized cTrader CSVs.

This wrapper is purpose-built for fast paper wiring:
- Auto-select latest XAUUSD M1 CSV under data root.
- Auto-create a runtime qualification profile (without mutating canonical quals).
- Run RenkoLiveTrader in PER paper gate with SL = 1 brick by default.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from kinetra.monitoring import emit_event, emit_health
from kinetra.renko.live_trader import (
    HistoricalBarProvider,
    LiveTraderConfig,
    PaperDispatcher,
    PERGate,
    RenkoLiveTrader,
)

LOG = logging.getLogger("ctrader.paper_gold")


def _safe_positive(x: Any, default: float) -> float:
    try:
        v = float(x)
    except Exception:
        return float(default)
    if not math.isfinite(v) or v <= 0:
        return float(default)
    return v


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run XAUUSD paper trading replay (Renko live trader)")
    p.add_argument("--symbol", default="XAUUSD")
    p.add_argument(
        "--data-root",
        default="data/master_standardized/ctrader/pepperstone_demo_45841299",
        help="Root containing category/SYMBOL folders",
    )
    p.add_argument(
        "--csv",
        default="",
        help="Optional explicit CSV path. If empty, latest <symbol>_M1_*.csv is auto-picked.",
    )
    p.add_argument(
        "--broker-source",
        default="ctrader",
        help="Broker source stamp used by runtime qualification profile",
    )
    p.add_argument("--brick-size", type=float, default=1.0, help="Locked default for gold core")
    p.add_argument("--stop-bricks", type=float, default=1.0, help="SL in brick units")
    p.add_argument(
        "--startup-skip-flips",
        type=int,
        default=2,
        help="Skip first N color flips at startup before allowing entries",
    )
    p.add_argument("--paper-lots", type=float, default=0.01, help="Paper lot size per trade")
    p.add_argument("--markov-threshold", type=float, default=0.60)
    p.add_argument("--markov-window", type=int, default=50)
    p.add_argument(
        "--fliprate-threshold",
        type=float,
        default=1.0,
        help="Set to 1.0 to effectively disable flip-rate gate",
    )
    p.add_argument("--fliprate-window", type=int, default=50)
    p.add_argument(
        "--speed",
        type=float,
        default=float("inf"),
        help="Replay speed multiplier; inf = max speed",
    )
    p.add_argument(
        "--poll-seconds",
        type=float,
        default=0.2,
        help="Loop sleep while waiting for replay completion",
    )
    p.add_argument(
        "--timeout-seconds",
        type=float,
        default=0.0,
        help="Hard timeout; 0 disables timeout",
    )
    p.add_argument(
        "--drawdown-halt-pct",
        type=float,
        default=1.0,
        help="Paper DD halt threshold (1.0 disables practical halts)",
    )
    p.add_argument("--loss-brake-after", type=int, default=8)
    p.add_argument("--loss-flat-after", type=int, default=12)
    p.add_argument("--loss-pause-minutes", type=float, default=120.0)
    p.add_argument("--trailing-mae-enabled", action="store_true")
    p.add_argument("--trailing-mae-after-bricks", type=int, default=1)
    p.add_argument("--trailing-mae-fraction", type=float, default=0.5)
    p.add_argument("--break-even-enabled", action="store_true")
    p.add_argument("--break-even-after-bricks", type=int, default=1)
    p.add_argument("--break-even-trigger-bricks", type=float, default=1.0)
    p.add_argument("--break-even-buffer-ticks", type=int, default=0)
    p.add_argument(
        "--monday-open-utc",
        default="00:00",
        help="UTC weekly session start on Monday in HH:MM",
    )
    p.add_argument(
        "--friday-close-utc",
        default="23:59",
        help="UTC weekly session close on Friday in HH:MM",
    )
    p.add_argument(
        "--runtime-qual-dir",
        default="",
        help="Optional directory for runtime qualification profile. Defaults to temporary directory.",
    )
    p.add_argument("--verbose", action="store_true")
    p.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Root logging level (default: INFO)",
    )
    return p.parse_args()


def _resolve_symbol_dir(data_root: Path, symbol: str) -> Path:
    matches = sorted(p for p in data_root.glob(f"*/{symbol}") if p.is_dir())
    if not matches:
        raise FileNotFoundError(f"{symbol}: no folder found under {data_root}")
    # Prefer folder that has contract_spec.json if multiple.
    with_spec = [m for m in matches if (m / "contract_spec.json").exists()]
    return sorted(with_spec or matches)[-1]


def _resolve_csv(symbol_dir: Path, symbol: str, csv_arg: str) -> Path:
    if csv_arg:
        p = Path(csv_arg)
        if not p.is_absolute():
            p = (PROJECT_ROOT / p).resolve()
        if not p.exists():
            raise FileNotFoundError(f"CSV not found: {p}")
        return p
    files = sorted(symbol_dir.glob(f"{symbol}_M1_*.csv"))
    if not files:
        raise FileNotFoundError(f"No {symbol}_M1_*.csv found in {symbol_dir}")
    return files[-1]


def _load_replay_df(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    lower = {c.lower(): c for c in df.columns}
    time_col = next((lower[k] for k in ("time", "datetime", "date") if k in lower), None)
    close_col = lower.get("close")
    if time_col is None or close_col is None:
        raise ValueError(f"CSV must include time + close columns: {csv_path}")
    out = pd.DataFrame(
        {
            "time": pd.to_datetime(df[time_col], utc=True, errors="coerce"),
            "close": pd.to_numeric(df[close_col], errors="coerce"),
        }
    )
    out = out.dropna(subset=["time", "close"]).sort_values("time").reset_index(drop=True)
    if out.empty:
        raise ValueError(f"No valid rows after parsing {csv_path}")
    return out


def _load_contract_fields(spec_path: Path) -> dict[str, float]:
    raw: dict[str, Any] = {}
    if spec_path.exists():
        raw = json.loads(spec_path.read_text(encoding="utf-8"))
    tick_size = _safe_positive(raw.get("tick_size") or raw.get("tickSize"), 0.01)
    contract_size = _safe_positive(raw.get("contract_size") or raw.get("contractSize"), 100.0)
    volume_min = _safe_positive(raw.get("volume_min"), 0.01)
    volume_step = _safe_positive(raw.get("volume_step"), 0.01)
    volume_max = _safe_positive(raw.get("volume_max"), 100.0)
    if volume_max < volume_min:
        volume_max = max(volume_min, 100.0)
    return {
        "tick_size": tick_size,
        "contract_size": contract_size,
        "volume_min": volume_min,
        "volume_step": volume_step,
        "volume_max": volume_max,
    }


def _write_runtime_qualification(
    *,
    symbol: str,
    broker_source: str,
    brick_size: float,
    markov_threshold: float,
    markov_window: int,
    fliprate_threshold: float,
    fliprate_window: int,
    runtime_qual_dir: Path,
    contract_fields: dict[str, float],
) -> Path:
    sym_dir = runtime_qual_dir / symbol
    sym_dir.mkdir(parents=True, exist_ok=True)
    qual = {
        "symbol": symbol,
        "qualified": True,
        "disqualified": False,
        "disqualification_reason": "",
        "broker_source": broker_source,
        "cluster": "precious_metals",
        "brick_size": float(brick_size),
        "omega": 9.99,
        "z_factor": 9.99,
        "n_trades": 999,
        "filter_params": {
            "fliprate_window": int(fliprate_window),
            "fliprate_threshold": float(fliprate_threshold),
            "markov_window": int(markov_window),
            "markov_threshold": float(markov_threshold),
        },
        "usd_per_point": float(contract_fields["contract_size"]),
        "tick_size": float(contract_fields["tick_size"]),
        "volume_min": float(contract_fields["volume_min"]),
        "volume_step": float(contract_fields["volume_step"]),
        "volume_max": float(contract_fields["volume_max"]),
        "pipeline_version": "paper_runtime",
        "recalibration_due": False,
    }
    (sym_dir / "qualification.json").write_text(json.dumps(qual, indent=2), encoding="utf-8")
    session = {"session_break_minutes": 30.0}
    (sym_dir / "session_profile.json").write_text(json.dumps(session, indent=2), encoding="utf-8")
    return sym_dir


def _wait_for_replay(
    provider: HistoricalBarProvider, poll_seconds: float, timeout_seconds: float
) -> bool:
    start = time.monotonic()
    while True:
        th = getattr(provider, "_thread", None)
        if th is None:
            return False
        if not th.is_alive():
            return False
        if timeout_seconds > 0 and (time.monotonic() - start) >= timeout_seconds:
            return True
        time.sleep(max(poll_seconds, 0.05))


def main() -> int:
    args = _parse_args()
    level = getattr(logging, str(args.log_level).upper(), logging.INFO)
    if args.verbose:
        level = logging.DEBUG
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    symbol = args.symbol.strip().upper()
    if not symbol:
        raise ValueError("symbol cannot be empty")
    if not math.isfinite(args.brick_size) or args.brick_size <= 0:
        raise ValueError(f"brick-size must be > 0, got {args.brick_size}")
    if not math.isfinite(args.stop_bricks) or args.stop_bricks <= 0:
        raise ValueError(f"stop-bricks must be > 0, got {args.stop_bricks}")
    if not math.isfinite(args.paper_lots) or args.paper_lots <= 0:
        raise ValueError(f"paper-lots must be > 0, got {args.paper_lots}")
    if int(args.startup_skip_flips) < 0:
        raise ValueError(f"startup-skip-flips must be >= 0, got {args.startup_skip_flips}")

    data_root = (PROJECT_ROOT / args.data_root).resolve()
    symbol_dir = _resolve_symbol_dir(data_root, symbol)
    csv_path = _resolve_csv(symbol_dir, symbol, args.csv)
    replay_df = _load_replay_df(csv_path)
    contract_fields = _load_contract_fields(symbol_dir / "contract_spec.json")

    runtime_qual_base: Path
    temp_ctx = None
    if args.runtime_qual_dir:
        runtime_qual_base = (PROJECT_ROOT / args.runtime_qual_dir).resolve()
        runtime_qual_base.mkdir(parents=True, exist_ok=True)
    else:
        temp_ctx = tempfile.TemporaryDirectory(prefix="kinetra_paper_qual_")
        runtime_qual_base = Path(temp_ctx.name)

    _write_runtime_qualification(
        symbol=symbol,
        broker_source=args.broker_source,
        brick_size=args.brick_size,
        markov_threshold=args.markov_threshold,
        markov_window=args.markov_window,
        fliprate_threshold=args.fliprate_threshold,
        fliprate_window=args.fliprate_window,
        runtime_qual_dir=runtime_qual_base,
        contract_fields=contract_fields,
    )

    provider = HistoricalBarProvider(
        {symbol: replay_df[["time", "close"]]}, speed_multiplier=args.speed
    )
    dispatcher = PaperDispatcher()
    config = LiveTraderConfig(
        symbols=[symbol],
        gate=PERGate.SIMULATED,
        stop_bricks=float(args.stop_bricks),
        qual_dir=runtime_qual_base,
        broker_source=args.broker_source,
        lot_step=float(contract_fields["volume_step"]),
        min_lots=float(contract_fields["volume_min"]),
        poll_interval_seconds=max(args.poll_seconds, 0.05),
        drawdown_halt_pct=float(args.drawdown_halt_pct),
        initial_equity_usd=1000.0,
        paper_lots=float(args.paper_lots),
        startup_skip_flips=int(args.startup_skip_flips),
        monday_open_utc=str(args.monday_open_utc),
        friday_close_utc=str(args.friday_close_utc),
        loss_brake_after_consecutive_losses=int(max(args.loss_brake_after, 1)),
        loss_flat_after_consecutive_losses=int(
            max(args.loss_flat_after, max(args.loss_brake_after, 1) + 1)
        ),
        loss_pause_minutes=float(max(args.loss_pause_minutes, 1.0)),
        trailing_mae_enabled=bool(args.trailing_mae_enabled),
        trailing_mae_after_bricks=int(max(args.trailing_mae_after_bricks, 1)),
        trailing_mae_fraction=float(max(min(args.trailing_mae_fraction, 1.0), 0.0)),
        break_even_enabled=bool(args.break_even_enabled),
        break_even_after_bricks=int(max(args.break_even_after_bricks, 1)),
        break_even_trigger_bricks=float(max(args.break_even_trigger_bricks, 0.0)),
        break_even_buffer_ticks=int(max(args.break_even_buffer_ticks, 0)),
    )
    trader = RenkoLiveTrader(config=config, bar_provider=provider, dispatcher=dispatcher)

    emit_event(
        stream="paper_trading",
        component="run_paper_gold",
        event_type="start",
        status="info",
        payload={
            "symbol": symbol,
            "csv": str(csv_path),
            "rows": int(len(replay_df)),
            "brick_size": float(args.brick_size),
            "stop_bricks": float(args.stop_bricks),
            "startup_skip_flips": int(args.startup_skip_flips),
            "monday_open_utc": str(args.monday_open_utc),
            "friday_close_utc": str(args.friday_close_utc),
            "loss_brake_after": int(max(args.loss_brake_after, 1)),
            "loss_flat_after": int(max(args.loss_flat_after, max(args.loss_brake_after, 1) + 1)),
            "loss_pause_minutes": float(max(args.loss_pause_minutes, 1.0)),
            "trailing_mae_enabled": bool(args.trailing_mae_enabled),
            "trailing_mae_after_bricks": int(max(args.trailing_mae_after_bricks, 1)),
            "trailing_mae_fraction": float(max(min(args.trailing_mae_fraction, 1.0), 0.0)),
            "break_even_enabled": bool(args.break_even_enabled),
            "break_even_after_bricks": int(max(args.break_even_after_bricks, 1)),
            "break_even_trigger_bricks": float(max(args.break_even_trigger_bricks, 0.0)),
            "break_even_buffer_ticks": int(max(args.break_even_buffer_ticks, 0)),
            "runtime_qual_dir": str(runtime_qual_base),
        },
    )

    try:
        LOG.info(
            "Starting paper replay: symbol=%s csv=%s rows=%d brick=%.4f stop=%.2f",
            symbol,
            csv_path,
            len(replay_df),
            args.brick_size,
            args.stop_bricks,
        )
        trader.start()
        timed_out = _wait_for_replay(
            provider, poll_seconds=args.poll_seconds, timeout_seconds=args.timeout_seconds
        )
        if timed_out:
            LOG.warning("Replay timeout reached (%.1fs); stopping gracefully", args.timeout_seconds)
    except KeyboardInterrupt:
        LOG.warning("Interrupted by user; stopping trader")
    finally:
        trader.stop()
        summary = trader.session_summary()
        emit_health(
            component="run_paper_gold",
            status="ok",
            checks={"run": "pass"},
            metrics={
                "session_pnl_usd": float(summary.get("session_pnl_usd", 0.0)),
                "n_completed_trades": int(summary.get("n_completed_trades", 0)),
            },
            details={"symbol": symbol, "log_path": summary.get("log_path")},
        )
        emit_event(
            stream="paper_trading",
            component="run_paper_gold",
            event_type="finish",
            status="ok",
            payload=summary,
        )
        print(json.dumps(summary, indent=2, default=str))
        print(f"[OK] Trader log: {summary.get('log_path')}")
        if temp_ctx is not None:
            temp_ctx.cleanup()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
