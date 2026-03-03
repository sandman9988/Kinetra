#!/usr/bin/env python3
"""One-command launcher for live trading + paper stats split terminals.

Runs:
1) Optional preflight wiring check
2) Live runner (dry-run or real live)
3) Paper readiness stats watcher

If tmux is installed, opens a split session; otherwise prints Zed-ready commands.
"""

from __future__ import annotations

import argparse
import os
import shlex
import shutil
import subprocess
from pathlib import Path


def _q(x: object) -> str:
    return shlex.quote(str(x))


def _bool_flag(v: bool) -> str:
    return "true" if v else "false"


def _parse_args() -> argparse.Namespace:
    root_default = str(Path.home() / "Projects" / "Kinetra")
    p = argparse.ArgumentParser(description="Launch live trading + paper stats in one command")

    # General
    p.add_argument("--root", default=root_default)
    p.add_argument("--session-name", default="kinetra-live-paper")
    p.add_argument("--no-attach", action="store_true", help="Create tmux session but do not attach")
    p.add_argument(
        "--kill-existing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Kill existing tmux session with same name before starting",
    )

    # Live runner options
    p.add_argument("--symbol", default="XAUUSD")
    p.add_argument(
        "--data-root",
        default="data/master_standardized/ctrader/pepperstone_demo_45841299",
    )
    p.add_argument("--broker-source", default="ctrader")
    p.add_argument("--brick-size", type=float, default=1.0)
    p.add_argument("--stop-bricks", type=float, default=0.5)
    p.add_argument("--startup-skip-flips", type=int, default=2)
    p.add_argument("--target-risk-usd", type=float, default=100.0)
    p.add_argument("--drawdown-halt-pct", type=float, default=0.02)
    p.add_argument("--loss-brake-after", type=int, default=8)
    p.add_argument("--loss-flat-after", type=int, default=12)
    p.add_argument("--loss-pause-minutes", type=float, default=120.0)
    p.add_argument(
        "--trailing-mae-enabled",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    p.add_argument("--trailing-mae-after-bricks", type=int, default=1)
    p.add_argument("--trailing-mae-fraction", type=float, default=0.5)
    p.add_argument(
        "--break-even-enabled",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    p.add_argument("--break-even-after-bricks", type=int, default=1)
    p.add_argument("--break-even-trigger-bricks", type=float, default=1.0)
    p.add_argument("--break-even-buffer-ticks", type=int, default=0)
    p.add_argument("--status-interval-seconds", type=float, default=20.0)
    p.add_argument("--monday-open-utc", default="03:00")
    p.add_argument("--friday-close-utc", default="20:55")
    p.add_argument("--runtime-qual-dir", default="")
    p.add_argument("--connect-timeout", type=float, default=30.0)
    p.add_argument("--fill-timeout", type=float, default=10.0)
    p.add_argument(
        "--live-mode",
        choices=["dry", "live"],
        default="dry",
        help="dry = no orders, live = real orders (requires --ack-live)",
    )
    p.add_argument("--ack-live", default="")
    p.add_argument("--verbose-live", action="store_true")

    # Preflight options
    p.add_argument(
        "--run-preflight",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run check_live_wiring.py before starting panes",
    )
    p.add_argument("--preflight-timeout", type=float, default=20.0)
    p.add_argument("--preflight-observe-seconds", type=float, default=10.0)
    p.add_argument("--preflight-require-bars", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--preflight-verbose", action="store_true")

    # Paper stats options
    p.add_argument("--stats-watch-seconds", type=float, default=5.0)
    p.add_argument("--max-health-age-minutes", type=float, default=30.0)
    p.add_argument("--paper-log-dir", default="results/renko/live")
    p.add_argument("--paper-log-path", default="")
    p.add_argument("--min-trades", type=int, default=30)
    p.add_argument("--min-omega", type=float, default=1.5)
    p.add_argument("--omega-threshold", type=float, default=0.0)
    p.add_argument("--initial-equity", type=float, default=1000.0)
    p.add_argument("--regime-blocks", type=int, default=3)
    p.add_argument("--max-tolerable-dd-pct", type=float, default=20.0)

    return p.parse_args()


def _build_live_cmd(args: argparse.Namespace) -> str:
    root = Path(args.root).resolve()
    parts = [
        "python",
        "scripts/ctrader/run_live_gold.py",
        "--symbol",
        args.symbol,
        "--broker-source",
        args.broker_source,
        "--data-root",
        args.data_root,
        "--brick-size",
        str(args.brick_size),
        "--stop-bricks",
        str(args.stop_bricks),
        "--startup-skip-flips",
        str(args.startup_skip_flips),
        "--target-risk-usd",
        str(args.target_risk_usd),
        "--drawdown-halt-pct",
        str(args.drawdown_halt_pct),
        "--loss-brake-after",
        str(args.loss_brake_after),
        "--loss-flat-after",
        str(args.loss_flat_after),
        "--loss-pause-minutes",
        str(args.loss_pause_minutes),
        "--trailing-mae-after-bricks",
        str(max(int(args.trailing_mae_after_bricks), 1)),
        "--trailing-mae-fraction",
        str(max(min(float(args.trailing_mae_fraction), 1.0), 0.0)),
        "--break-even-after-bricks",
        str(max(int(args.break_even_after_bricks), 1)),
        "--break-even-trigger-bricks",
        str(max(float(args.break_even_trigger_bricks), 0.0)),
        "--break-even-buffer-ticks",
        str(max(int(args.break_even_buffer_ticks), 0)),
        "--status-interval-seconds",
        str(args.status_interval_seconds),
        "--monday-open-utc",
        args.monday_open_utc,
        "--friday-close-utc",
        args.friday_close_utc,
        "--connect-timeout",
        str(args.connect_timeout),
        "--fill-timeout",
        str(args.fill_timeout),
    ]
    if args.runtime_qual_dir:
        parts.extend(["--runtime-qual-dir", args.runtime_qual_dir])
    if args.trailing_mae_enabled:
        parts.append("--trailing-mae-enabled")
    if args.break_even_enabled:
        parts.append("--break-even-enabled")
    if args.verbose_live:
        parts.append("--verbose")
    if args.live_mode == "dry":
        parts.append("--dry-run")
    else:
        if args.ack_live != "I_UNDERSTAND_LIVE_RISK":
            raise ValueError('live mode requires --ack-live "I_UNDERSTAND_LIVE_RISK"')
        parts.extend(["--ack-live", args.ack_live])
    run = " ".join(_q(x) for x in parts)
    return f"cd {_q(root)} && source .venv/bin/activate && {run}"


def _build_preflight_cmd(args: argparse.Namespace) -> str:
    root = Path(args.root).resolve()
    parts = [
        "python",
        "scripts/ctrader/check_live_wiring.py",
        "--symbols",
        args.symbol,
        "--timeout",
        str(args.preflight_timeout),
        "--observe-seconds",
        str(args.preflight_observe_seconds),
        "--data-root",
        args.data_root,
    ]
    if args.preflight_require_bars:
        parts.append("--require-bars")
    if args.preflight_verbose:
        parts.append("--verbose")
    run = " ".join(_q(x) for x in parts)
    return f"cd {_q(root)} && source .venv/bin/activate && {run}"


def _build_stats_cmd(args: argparse.Namespace) -> str:
    root = Path(args.root).resolve()
    watch_n = _q(max(args.stats_watch_seconds, 1.0))
    if args.live_mode == "live":
        health_cmd = " ".join(
            _q(x)
            for x in (
                "python",
                "scripts/monitoring/system_health_overview.py",
                "--hours",
                "2",
                "--tail",
                "12",
                "--max-health-age-minutes",
                str(max(float(args.max_health_age_minutes), 0.0)),
            )
        )
        perf_parts = [
            "python",
            "scripts/ctrader/paper_readiness_report.py",
            "--summary-compact",
            "--log-dir",
            args.paper_log_dir,
            "--min-trades",
            str(args.min_trades),
            "--min-omega",
            str(args.min_omega),
            "--omega-threshold",
            str(args.omega_threshold),
            "--initial-equity",
            str(args.initial_equity),
            "--regime-blocks",
            str(args.regime_blocks),
            "--max-tolerable-dd-pct",
            str(args.max_tolerable_dd_pct),
        ]
        if args.paper_log_path:
            perf_parts.extend(["--log-path", args.paper_log_path])
        perf_cmd = " ".join(_q(x) for x in perf_parts)
        loop = (
            "while true; do "
            "clear; "
            "echo '============================================================'; "
            "date -u '+UTC %Y-%m-%d %H:%M:%S'; "
            "echo '[TRADING PERFORMANCE]'; "
            f"{perf_cmd}; "
            "echo; "
            "echo '[SYSTEM HEALTH]'; "
            f"{health_cmd}; "
            "echo; "
            f"sleep {watch_n}; "
            "done"
        )
        watch = f"bash -lc {_q(loop)}"
    else:
        parts = [
            "python",
            "scripts/ctrader/paper_readiness_report.py",
            "--log-dir",
            args.paper_log_dir,
            "--min-trades",
            str(args.min_trades),
            "--min-omega",
            str(args.min_omega),
            "--omega-threshold",
            str(args.omega_threshold),
            "--initial-equity",
            str(args.initial_equity),
            "--regime-blocks",
            str(args.regime_blocks),
            "--max-tolerable-dd-pct",
            str(args.max_tolerable_dd_pct),
        ]
        if args.paper_log_path:
            parts.extend(["--log-path", args.paper_log_path])
        inner = " ".join(_q(x) for x in parts)
        watch = f"watch -n {watch_n} {_q(inner)}"
    return f"cd {_q(root)} && source .venv/bin/activate && {watch}"


def _run_bash(command: str) -> None:
    subprocess.run(["bash", "-lc", command], check=True)


def main() -> int:
    args = _parse_args()
    live_cmd = _build_live_cmd(args)
    stats_cmd = _build_stats_cmd(args)

    if args.run_preflight:
        preflight_cmd = _build_preflight_cmd(args)
        print("[INFO] Running preflight wiring check...")
        _run_bash(preflight_cmd)

    tmux = shutil.which("tmux")
    if not tmux:
        print("[INFO] tmux is not installed.")
        print("[INFO] Use Zed split terminal and run:\n")
        print("Pane 1 (live trading):")
        print(live_cmd)
        print("\nPane 2 (paper stats):")
        print(stats_cmd)
        return 0

    session = args.session_name
    if args.kill_existing:
        has_existing = subprocess.run(
            [tmux, "has-session", "-t", session],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if has_existing.returncode == 0:
            subprocess.run([tmux, "kill-session", "-t", session], check=False)

    subprocess.run(
        [tmux, "new-session", "-d", "-s", session, f"bash -lc {shlex.quote(live_cmd)}"], check=True
    )
    subprocess.run([tmux, "set-option", "-t", session, "-g", "mouse", "on"], check=False)
    subprocess.run(
        [tmux, "set-option", "-t", session, "-g", "history-limit", "200000"], check=False
    )
    subprocess.run([tmux, "set-window-option", "-t", session, "-g", "mode-keys", "vi"], check=False)
    subprocess.run(
        [
            tmux,
            "bind-key",
            "-T",
            "root",
            "WheelUpPane",
            "if-shell",
            "-F",
            "#{pane_in_mode}",
            "send-keys -M",
            "copy-mode -e",
        ],
        check=False,
    )
    subprocess.run(
        [tmux, "bind-key", "-T", "copy-mode-vi", "WheelUpPane", "send-keys", "-X", "scroll-up"],
        check=False,
    )
    subprocess.run(
        [tmux, "bind-key", "-T", "copy-mode-vi", "WheelDownPane", "send-keys", "-X", "scroll-down"],
        check=False,
    )
    subprocess.run(
        [tmux, "split-window", "-h", "-t", session, f"bash -lc {shlex.quote(stats_cmd)}"],
        check=True,
    )
    subprocess.run([tmux, "select-layout", "-t", session, "even-horizontal"], check=False)

    if not args.no_attach:
        if os.environ.get("TMUX"):
            subprocess.run([tmux, "switch-client", "-t", session], check=True)
        else:
            subprocess.run([tmux, "attach-session", "-t", session], check=True)
    else:
        print(f"[OK] tmux session created: {session}")
        print(f"[INFO] Attach with: tmux attach-session -t {session}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
