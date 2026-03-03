#!/usr/bin/env python3
"""Run XAUUSD live trading via cTrader Open API (or dry-run shadow mode).

Profile defaults:
- symbol: XAUUSD
- gate: micro (0.01 lot ceiling, always fixed)
- stop: 0.5 brick
- startup skip: 2 flips

Lot sizing (real orders only):
  lots = 0.01 per $1 000 of conservative equity (balance − used_margin),
  capped by the active PER gate ceiling.
  micro gate ceiling = 0.01 (fixed regardless of account size).
  Dry-run always uses paper_lots=0.01.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import signal
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from kinetra.monitoring import emit_event, emit_health
from kinetra.renko.ctrader_dispatcher import CTraderBarProvider, build_ctrader_session
from kinetra.renko.live_trader import (
    BarProvider,
    HistoricalBarProvider,
    LiveTraderConfig,
    PaperDispatcher,
    PERGate,
    RenkoLiveTrader,
)

LOG = logging.getLogger("ctrader.live_gold")


def _safe_positive(x: Any, default: float) -> float:
    try:
        v = float(x)
    except Exception:
        return float(default)
    return float(default) if v <= 0 else v


def _compute_live_lots(
    *,
    balance: float,
    used_margin: float,
    gate: "PERGate",
    lot_step: float,
    min_lots: float,
    brick_size: float,
    usd_per_point: float,
) -> tuple[float, float]:
    """Return (lots, target_risk_usd) using the equity-proportional rule.

    Rule: 0.01 lot per $1 000 of conservative equity, capped by gate ceiling.
    Conservative equity = min(balance, balance - used_margin) = balance - used_margin.
    This is equivalent to free margin, i.e. the lower of equity and balance when
    unrealised losses are present (worst-case at startup).
    """
    conservative_equity = max(balance - used_margin, 0.0)
    raw = math.floor(conservative_equity / 1_000.0) * 0.01
    ceiling = gate.lot_ceiling  # 0.01 micro / 0.10 small / 999 full
    clamped = min(raw, ceiling)
    # Quantise to lot step and enforce minimum
    if lot_step > 0:
        clamped = round(clamped / lot_step) * lot_step
    lots = max(clamped, min_lots)
    target_risk_usd = lots * brick_size * usd_per_point
    return lots, target_risk_usd


class CTraderBarProviderAdapter(BarProvider):
    """Adapter from keyword-style cTrader callbacks to RenkoLiveTrader callback."""

    def __init__(self, inner: CTraderBarProvider) -> None:
        self._inner = inner

    def subscribe(
        self,
        symbol: str,
        callback: Callable[[str, float, Any], None],
    ) -> None:
        def _on_bar(**kwargs: Any) -> None:
            try:
                sym = str(kwargs.get("symbol", symbol))
                close = float(kwargs["close"])
                ts = kwargs["timestamp"]
                callback(sym, close, ts)
            except Exception:
                LOG.exception("Bar adapter callback failed for %s", symbol)

        self._inner.subscribe(symbol, _on_bar)

    def start(self) -> None:
        self._inner.start()

    def stop(self) -> None:
        self._inner.stop()


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run XAUUSD live micro trading (cTrader)")
    p.add_argument("--symbol", default="XAUUSD")
    p.add_argument("--broker-source", default="ctrader")
    p.add_argument(
        "--data-root", default="data/master_standardized/ctrader/pepperstone_demo_45841299"
    )
    p.add_argument("--stop-bricks", type=float, default=0.5)
    p.add_argument("--brick-size", type=float, default=1.0)
    p.add_argument("--startup-skip-flips", type=int, default=2)
    p.add_argument("--target-risk-usd", type=float, default=100.0)
    p.add_argument("--drawdown-halt-pct", type=float, default=0.02)
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
        "--status-interval-seconds",
        type=float,
        default=30.0,
        help="Periodic status print interval while running",
    )
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
    p.add_argument("--runtime-qual-dir", default="")
    p.add_argument("--connect-timeout", type=float, default=30.0)
    p.add_argument("--fill-timeout", type=float, default=10.0)
    p.add_argument(
        "--gate",
        default="micro",
        choices=["micro", "small", "full"],
        help="PER gate for real orders: micro=0.01 lot, small=0.10 lot, full=unlimited",
    )
    p.add_argument(
        "--dry-run", action="store_true", help="Use paper dispatcher with live cTrader bars"
    )
    p.add_argument(
        "--ack-live",
        default="",
        help='Required unless --dry-run. Must equal "I_UNDERSTAND_LIVE_RISK".',
    )
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def _resolve_symbol_dir(data_root: Path, symbol: str) -> Path:
    matches = sorted(p for p in data_root.glob(f"*/{symbol}") if p.is_dir())
    if not matches:
        raise FileNotFoundError(f"{symbol}: no folder found under {data_root}")
    with_spec = [m for m in matches if (m / "contract_spec.json").exists()]
    return sorted(with_spec or matches)[-1]


def _load_contract_fields(spec_path: Path) -> dict[str, float]:
    raw: Dict[str, Any] = {}
    if spec_path.exists():
        raw = json.loads(spec_path.read_text(encoding="utf-8"))
    tick_size = _safe_positive(raw.get("tick_size") or raw.get("tickSize"), 0.01)
    contract_size = _safe_positive(raw.get("contract_size") or raw.get("contractSize"), 100.0)
    volume_min = _safe_positive(raw.get("volume_min"), 0.01)
    volume_step = _safe_positive(raw.get("volume_step"), 0.01)
    volume_max = _safe_positive(raw.get("volume_max"), 100.0)
    if volume_max < volume_min:
        volume_max = volume_min
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
    runtime_qual_dir: Path,
    brick_size: float,
    contract_fields: dict[str, float],
) -> None:
    sym_dir = runtime_qual_dir / symbol
    sym_dir.mkdir(parents=True, exist_ok=True)
    qual = {
        "symbol": symbol,
        "qualified": True,
        "broker_source": broker_source,
        "cluster": "precious_metals",
        "brick_size": float(brick_size),
        "omega": 9.99,
        "z_factor": 9.99,
        "n_trades": 999,
        "filter_params": {
            "fliprate_window": 50,
            "fliprate_threshold": 1.0,
            "markov_window": 50,
            "markov_threshold": 0.60,
        },
        "usd_per_point": float(contract_fields["contract_size"]),
        "tick_size": float(contract_fields["tick_size"]),
        "volume_min": float(contract_fields["volume_min"]),
        "volume_step": float(contract_fields["volume_step"]),
        "volume_max": float(contract_fields["volume_max"]),
        "pipeline_version": "live_runtime",
        "recalibration_due": False,
    }
    (sym_dir / "qualification.json").write_text(json.dumps(qual, indent=2), encoding="utf-8")
    (sym_dir / "session_profile.json").write_text(
        json.dumps({"session_break_minutes": 30.0}, indent=2),
        encoding="utf-8",
    )


def main() -> int:
    args = _parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    symbol = args.symbol.strip().upper()
    if not symbol:
        raise ValueError("symbol cannot be empty")
    if not args.dry_run and args.ack_live != "I_UNDERSTAND_LIVE_RISK":
        raise ValueError('live mode requires --ack-live "I_UNDERSTAND_LIVE_RISK"')

    data_root = (PROJECT_ROOT / args.data_root).resolve()
    symbol_dir = _resolve_symbol_dir(data_root, symbol)
    contract_fields = _load_contract_fields(symbol_dir / "contract_spec.json")

    temp_ctx: Optional[tempfile.TemporaryDirectory[str]] = None
    if args.runtime_qual_dir:
        qual_dir = (PROJECT_ROOT / args.runtime_qual_dir).resolve()
        qual_dir.mkdir(parents=True, exist_ok=True)
    else:
        temp_ctx = tempfile.TemporaryDirectory(prefix="kinetra_live_qual_")
        qual_dir = Path(temp_ctx.name)

    _write_runtime_qualification(
        symbol=symbol,
        broker_source=args.broker_source,
        runtime_qual_dir=qual_dir,
        brick_size=float(args.brick_size),
        contract_fields=contract_fields,
    )

    dispatcher, c_bar_provider = build_ctrader_session(
        credentials=None,
        connect_timeout_s=float(args.connect_timeout),
        fill_timeout_s=float(args.fill_timeout),
    )
    bar_provider = CTraderBarProviderAdapter(c_bar_provider)
    if args.dry_run:
        order_dispatcher = PaperDispatcher()
        gate = PERGate.SIMULATED
    else:
        order_dispatcher = dispatcher
        gate = PERGate(args.gate)

    # Equity-proportional lot sizing for real orders: 0.01 lot per $1 000.
    # Conservative equity = balance − used_margin (lower of the two when
    # unrealised losses exist). Lot is capped by the active gate ceiling.
    # Dry-run always uses fixed paper_lots (0.01); sizing is skipped.
    effective_target_risk_usd = float(args.target_risk_usd)
    computed_lots: Optional[float] = None
    if not args.dry_run:
        try:
            snap = c_bar_provider._connector.get_account_snapshot(timeout_s=10.0)
            acct_balance = float(snap.get("balance", 0.0) or 0.0)
            acct_used_margin = float(snap.get("used_margin", 0.0) or 0.0)
            computed_lots, effective_target_risk_usd = _compute_live_lots(
                balance=acct_balance,
                used_margin=acct_used_margin,
                gate=gate,
                lot_step=float(contract_fields["volume_step"]),
                min_lots=float(contract_fields["volume_min"]),
                brick_size=float(args.brick_size),
                usd_per_point=float(contract_fields["contract_size"]),
            )
            LOG.info(
                "[SIZING] balance=%.2f used_margin=%.2f conservative_equity=%.2f "
                "gate=%s computed_lots=%.4f target_risk_usd=%.4f",
                acct_balance,
                acct_used_margin,
                max(acct_balance - acct_used_margin, 0.0),
                gate.value,
                computed_lots,
                effective_target_risk_usd,
            )
        except Exception:
            LOG.warning(
                "[SIZING] Account snapshot failed; falling back to target_risk_usd=%.2f",
                effective_target_risk_usd,
                exc_info=True,
            )

    cfg = LiveTraderConfig(
        symbols=[symbol],
        gate=gate,
        stop_bricks=float(args.stop_bricks),
        target_risk_usd=effective_target_risk_usd,
        qual_dir=qual_dir,
        broker_source=args.broker_source,
        lot_step=float(contract_fields["volume_step"]),
        min_lots=float(contract_fields["volume_min"]),
        drawdown_halt_pct=float(args.drawdown_halt_pct),
        startup_skip_flips=int(max(args.startup_skip_flips, 0)),
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
        initial_equity_usd=1000.0,
        paper_lots=0.01,
    )
    trader = RenkoLiveTrader(config=cfg, bar_provider=bar_provider, dispatcher=order_dispatcher)

    stop_flag = {"stop": False}

    def _request_stop(signum, frame):  # type: ignore[no-untyped-def]
        LOG.warning("Signal %s received; stopping...", signum)
        stop_flag["stop"] = True

    signal.signal(signal.SIGINT, _request_stop)
    signal.signal(signal.SIGTERM, _request_stop)

    emit_event(
        stream="live_trading",
        component="run_live_gold",
        event_type="start",
        status="info",
        payload={
            "symbol": symbol,
            "mode": "dry_run" if args.dry_run else "live_micro",
            "stop_bricks": float(args.stop_bricks),
            "startup_skip_flips": int(args.startup_skip_flips),
            "drawdown_halt_pct": float(args.drawdown_halt_pct),
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
        },
    )

    try:
        LOG.info(
            (
                "Starting %s trader: symbol=%s gate=%s stop_bricks=%.2f "
                "startup_skip_flips=%d monday_open_utc=%s friday_close_utc=%s "
                "loss_brake=%d loss_flat=%d loss_pause_minutes=%.1f "
                "trailing_mae=%s after_bars=%d frac=%.2f "
                "break_even=%s after_bars=%d trigger_bricks=%.2f buffer_ticks=%d"
            ),
            "dry-run" if args.dry_run else "live",
            symbol,
            gate.value,
            float(args.stop_bricks),
            int(args.startup_skip_flips),
            str(args.monday_open_utc),
            str(args.friday_close_utc),
            int(max(args.loss_brake_after, 1)),
            int(max(args.loss_flat_after, max(args.loss_brake_after, 1) + 1)),
            float(max(args.loss_pause_minutes, 1.0)),
            bool(args.trailing_mae_enabled),
            int(max(args.trailing_mae_after_bricks, 1)),
            float(max(min(args.trailing_mae_fraction, 1.0), 0.0)),
            bool(args.break_even_enabled),
            int(max(args.break_even_after_bricks, 1)),
            float(max(args.break_even_trigger_bricks, 0.0)),
            int(max(args.break_even_buffer_ticks, 0)),
        )
        trader.start()
        status_interval = max(float(args.status_interval_seconds), 5.0)
        last_status = 0.0
        while not stop_flag["stop"]:
            now = time.monotonic()
            if (now - last_status) >= status_interval:
                s = trader.session_summary()
                trades = int(s.get("n_completed_trades", 0))
                pnl = float(s.get("session_pnl_usd", 0.0))
                dd_pct = 100.0 * float(s.get("portfolio_drawdown", 0.0))
                halted = bool(s.get("is_halted", False))
                win_rate = float(s.get("win_rate", 0.0))
                omega = float(s.get("omega", 0.0))
                if args.verbose:
                    LOG.info(
                        "[STATUS] trades=%d pnl=%.2f dd=%.2f%% halted=%s win_rate=%.3f omega=%.3f",
                        trades,
                        pnl,
                        dd_pct,
                        halted,
                        win_rate,
                        omega,
                    )
                else:
                    print("\x1b[2J\x1b[H", end="")
                    print(
                        f"[LIVE STATUS] mode={'dry' if args.dry_run else 'live'} "
                        f"symbol={symbol} gate={gate.value}"
                    )
                    print(
                        f"[STATUS] trades={trades} pnl={pnl:.2f} dd={dd_pct:.2f}% "
                        f"halted={halted} win_rate={win_rate:.3f} omega={omega:.3f}"
                    )
                    print(
                        f"[ENTRY] signals={int(s.get('entry_signals_seen', 0))} "
                        f"opened={int(s.get('entries_opened', 0))}"
                    )
                    blocks = s.get("entry_block_counts", {}) or {}
                    top_blocks = list(blocks.items())[:5]
                    if top_blocks:
                        rendered = " | ".join(f"{k}={int(v)}" for k, v in top_blocks)
                        print(f"[BLOCKERS] {rendered}")
                    print(
                        f"[CONFIG] brick={float(args.brick_size):.2f} stop_bricks={float(args.stop_bricks):.2f} "
                        f"skip_flips={int(args.startup_skip_flips)} trailing_mae={bool(args.trailing_mae_enabled)} "
                        f"after_bricks={int(max(args.trailing_mae_after_bricks, 1))} "
                        f"frac={float(max(min(args.trailing_mae_fraction, 1.0), 0.0)):.2f}"
                    )
                    sys.stdout.flush()
                last_status = now
            time.sleep(1.0)
    finally:
        trader.stop()
        summary = trader.session_summary()
        print(json.dumps(summary, indent=2, default=str))
        print(f"[OK] Trader log: {summary.get('log_path')}")
        emit_health(
            component="run_live_gold",
            status="ok",
            checks={"run": "pass"},
            metrics={
                "session_pnl_usd": float(summary.get("session_pnl_usd", 0.0)),
                "n_completed_trades": int(summary.get("n_completed_trades", 0)),
            },
            details={"mode": "dry_run" if args.dry_run else "live_micro"},
        )
        if temp_ctx is not None:
            temp_ctx.cleanup()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
