#!/usr/bin/env python3
"""Run cTrader research cycle: optional download -> readiness -> staged OOS."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _run(cmd: list[str]) -> int:
    print(f"[RUN] {' '.join(cmd)}")
    proc = subprocess.run(cmd, cwd=PROJECT_ROOT)
    if proc.returncode != 0:
        print(f"[FAIL] exit={proc.returncode}: {' '.join(cmd)}")
    return proc.returncode


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Research cycle: optional cTrader download + readiness + staged rolling OOS"
    )
    parser.add_argument("--symbols", default="XAUUSD,NAS100")
    parser.add_argument("--days", type=int, default=365)
    parser.add_argument("--chunk-days", type=int, default=7)
    parser.add_argument("--spread-sample-seconds", type=float, default=8.0)
    parser.add_argument(
        "--data-root",
        default="data/master_standardized/ctrader/pepperstone_demo_45841299",
    )
    parser.add_argument("--results-dir", default="results/renko")
    parser.add_argument("--report-name", default="staged_oos_report_long.json")
    parser.add_argument("--rolling-window-bars", type=int, default=900)
    parser.add_argument("--rolling-step-bars", type=int, default=225)
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--skip-readiness", action="store_true")
    parser.add_argument("--include-all-staged", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    py = sys.executable

    if not args.skip_download:
        cmd = [
            py,
            "scripts/ctrader/download_ctrader_history.py",
            "--symbols",
            args.symbols,
            "--days",
            str(args.days),
            "--chunk-days",
            str(args.chunk_days),
            "--spread-sample-seconds",
            str(args.spread_sample_seconds),
        ]
        if args.verbose:
            cmd.append("--verbose")
        rc = _run(cmd)
        if rc != 0:
            return rc

    if not args.skip_readiness:
        cmd = [
            py,
            "scripts/ctrader/pre_live_readiness.py",
            "--symbols",
            args.symbols,
            "--data-root",
            args.data_root,
            "--max-spread-pts",
            "1000",
            "--min-bars",
            "10000",
        ]
        rc = _run(cmd)
        if rc != 0:
            return rc

    cmd = [
        py,
        "scripts/ctrader/generate_renko_engine.py",
        "--symbols",
        args.symbols,
        "--mode",
        "staged",
        "--data-root",
        args.data_root,
        "--results-dir",
        args.results_dir,
        "--staged-report-name",
        args.report_name,
        "--rolling-window-bars",
        str(args.rolling_window_bars),
        "--rolling-step-bars",
        str(args.rolling_step_bars),
        "--limited-min-trades",
        "0",
        "--limited-min-omega",
        "-1",
        "--limited-min-z",
        "-999",
        "--oos-min-omega",
        "-1",
        "--oos-min-z",
        "-999",
        "--oos-min-trades",
        "0",
    ]
    if args.include_all_staged:
        cmd.append("--staged-include-all-symbols")
    if args.force:
        cmd.append("--force")
    if args.verbose:
        cmd.append("--verbose")
    return _run(cmd)


if __name__ == "__main__":
    raise SystemExit(main())
