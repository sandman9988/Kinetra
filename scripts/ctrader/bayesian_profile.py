#!/usr/bin/env python3
"""Bayesian parameter search for launcher profile defaults.

Optimizes strategy parameters only (no latency variables) and emits shell-style
KEY=VALUE output for launch.sh consumption.
"""

from __future__ import annotations

import argparse
import importlib.util
import logging
import math
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _load_renko_engine_module(repo_root: Path):
    script_path = repo_root / "scripts" / "renko_engine.py"
    spec = importlib.util.spec_from_file_location("kinetra_renko_engine_script", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module from {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _gaussian_pdf(x: float, mu: float, sigma: float) -> float:
    s = max(float(sigma), 1e-9)
    z = (float(x) - float(mu)) / s
    return math.exp(-0.5 * z * z) / (s * math.sqrt(2.0 * math.pi))


def main() -> int:
    parser = argparse.ArgumentParser(description="Bayesian profile tuner for Kinetra launcher")
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--months", type=int, default=3)
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--base-brick", type=float, required=True)
    parser.add_argument("--base-stop", type=float, required=True)
    parser.add_argument("--base-risk", type=float, required=True)
    parser.add_argument("--base-flip", type=float, required=True)
    parser.add_argument("--base-markov", type=float, required=True)
    parser.add_argument("--base-trail-frac", type=float, required=True)
    parser.add_argument("--base-trail-after", type=int, required=True)
    parser.add_argument("--base-pe-entry", type=float, required=True)
    parser.add_argument("--base-pe-exit", type=float, required=True)
    parser.add_argument("--base-stale-factor", type=float, required=True)
    parser.add_argument("--base-stale-penalty", type=float, required=True)
    parser.add_argument("--format", choices=("shell",), default="shell")
    args = parser.parse_args()

    if args.months <= 0:
        raise SystemExit("--months must be > 0")
    if args.trials <= 0:
        raise SystemExit("--trials must be > 0")

    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    renko = _load_renko_engine_module(repo_root)
    logging.getLogger("kinetra.renko").setLevel(logging.WARNING)
    logging.getLogger("kinetra.renko.trading_engine").setLevel(logging.ERROR)

    closes = renko.load_m1_data(args.symbol)
    if closes is None or closes.empty:
        raise SystemExit(f"No M1 data found for {args.symbol}")

    cutoff = closes.index[-1] - pd.DateOffset(months=int(args.months))
    test_closes = closes[closes.index >= cutoff]
    if test_closes.empty:
        raise SystemExit(f"No test window data for {args.symbol} and {args.months} months")

    dsp, _ = renko._load_or_build_dsp_profile(args.symbol, closes=closes)
    if dsp is None:
        raise SystemExit(f"DSP profile unavailable for {args.symbol}")

    base_cfg, _ = renko._build_engine_config(args.symbol, dsp, sizing_mode="static")
    # Start from launcher baseline values, not raw engine defaults.
    base_cfg.brick_size = float(args.base_brick)
    base_cfg.stop_bricks = float(args.base_stop)
    base_cfg.target_risk_usd = float(args.base_risk)
    base_cfg.fliprate_threshold = float(args.base_flip)
    base_cfg.markov_threshold = float(args.base_markov)
    base_cfg.trailing_mfe_fraction = float(args.base_trail_frac)
    base_cfg.trailing_mfe_after_bricks = int(args.base_trail_after)
    base_cfg.pe_entry_threshold = float(args.base_pe_entry)
    base_cfg.pe_exit_threshold = float(args.base_pe_exit)
    base_cfg.stale_brick_factor = float(args.base_stale_factor)
    base_cfg.markov_stale_penalty = float(args.base_stale_penalty)

    brick_low = max(0.1, float(args.base_brick) * 0.65)
    brick_high = max(brick_low + 1e-6, float(args.base_brick) * 1.75)
    stop_low = max(0.2, float(args.base_stop) * 0.50)
    stop_high = max(stop_low + 1e-6, float(args.base_stop) * 2.00)
    risk_low = max(10.0, float(args.base_risk) * 0.35)
    risk_high = max(risk_low + 1e-6, float(args.base_risk) * 1.90)

    def objective(params: dict[str, float]) -> float:
        cfg = replace(
            base_cfg,
            brick_size=float(params["brick_size"]),
            stop_bricks=float(params["stop_bricks"]),
            target_risk_usd=float(params["target_risk"]),
            fliprate_threshold=float(params["fliprate_threshold"]),
            markov_threshold=float(params["markov_threshold"]),
            trailing_mfe_fraction=float(params["trailing_mfe_fraction"]),
            trailing_mfe_after_bricks=int(round(params["trailing_mfe_after_bricks"])),
            pe_entry_threshold=float(params["pe_entry_threshold"]),
            pe_exit_threshold=float(params["pe_exit_threshold"]),
            stale_brick_factor=float(params["stale_brick_factor"]),
            markov_stale_penalty=float(params["markov_stale_penalty"]),
        )
        if cfg.pe_exit_threshold <= cfg.pe_entry_threshold:
            return -1e6
        engine = renko.RenkoEngine(cfg, quiet_mode=True)
        res = engine.backtest(test_closes)
        if "error" in res:
            return -1e6
        summary: dict[str, Any] = res.get("summary", {}) or {}
        n_trades = int(summary.get("n_trades", 0))
        omega = float(summary.get("omega", 0.0))
        net_usd = float(summary.get("net_usd", 0.0))
        dd_pct = float(summary.get("max_drawdown_pct", 1.0))

        # Favor robust, tradable configs:
        #  - high omega
        #  - enough trades
        #  - controlled drawdown
        #  - positive net
        trade_term = min(n_trades, 200) / 200.0
        dd_penalty = max(0.0, dd_pct - 0.12) * 3.0
        net_term = _clamp(net_usd / 2000.0, -1.5, 1.5)
        low_trade_penalty = max(0, 25 - n_trades) * 0.08

        score = omega + trade_term + net_term - dd_penalty - low_trade_penalty
        if not math.isfinite(score):
            return -1e6
        return float(score)

    bounds: dict[str, tuple[float, float]] = {
        "brick_size": (brick_low, brick_high),
        "stop_bricks": (stop_low, stop_high),
        "target_risk": (risk_low, risk_high),
        "fliprate_threshold": (0.20, 0.50),
        "markov_threshold": (0.45, 0.80),
        "trailing_mfe_fraction": (0.20, 0.80),
        "trailing_mfe_after_bricks": (1.0, 6.0),
        "pe_entry_threshold": (0.45, 0.90),
        "pe_exit_threshold": (0.65, 0.98),
        "stale_brick_factor": (1.25, 4.00),
        "markov_stale_penalty": (0.05, 0.35),
    }
    rng = np.random.default_rng(int(args.seed))
    trials: list[tuple[dict[str, float], float]] = []
    n_startup = min(8, max(4, int(args.trials // 3)))
    n_candidates = 28

    def sample_uniform() -> dict[str, float]:
        out: dict[str, float] = {}
        for k, (lo, hi) in bounds.items():
            out[k] = float(rng.uniform(lo, hi))
        out["pe_exit_threshold"] = float(
            _clamp(
                max(out["pe_exit_threshold"], out["pe_entry_threshold"] + 0.05),
                bounds["pe_exit_threshold"][0],
                bounds["pe_exit_threshold"][1],
            )
        )
        return out

    def propose_tpe() -> dict[str, float]:
        ranked = sorted(trials, key=lambda t: t[1], reverse=True)
        n_good = max(3, int(math.ceil(len(ranked) * 0.30)))
        good = ranked[:n_good]
        bad = ranked[n_good:] if len(ranked) > n_good else ranked[-n_good:]

        best_params = None
        best_acq = -float("inf")
        for _ in range(n_candidates):
            cand: dict[str, float] = {}
            for name, (lo, hi) in bounds.items():
                good_vals = np.array([p[name] for p, _ in good], dtype=float)
                bad_vals = np.array([p[name] for p, _ in bad], dtype=float)
                span = hi - lo
                g_mu = float(good_vals.mean())
                b_mu = float(bad_vals.mean())
                g_sd = max(float(good_vals.std(ddof=0)), span * 0.06, 1e-6)
                b_sd = max(float(bad_vals.std(ddof=0)), span * 0.10, 1e-6)
                sample = float(rng.normal(g_mu, g_sd))
                cand[name] = float(_clamp(sample, lo, hi))

                # small exploration chance
                if rng.random() < 0.08:
                    cand[name] = float(rng.uniform(lo, hi))

                # Estimate l(x)/g(x) style acquisition via independent KDEs.
                px_good = _gaussian_pdf(cand[name], g_mu, g_sd)
                px_bad = _gaussian_pdf(cand[name], b_mu, b_sd)
                if name == "brick_size":
                    acq = math.log((px_good + 1e-12) / (px_bad + 1e-12))
                else:
                    acq += math.log((px_good + 1e-12) / (px_bad + 1e-12))

            if acq > best_acq:
                best_acq = acq
                best_params = cand
        assert best_params is not None
        best_params["pe_exit_threshold"] = float(
            _clamp(
                max(best_params["pe_exit_threshold"], best_params["pe_entry_threshold"] + 0.05),
                bounds["pe_exit_threshold"][0],
                bounds["pe_exit_threshold"][1],
            )
        )
        return best_params

    started = time.time()
    n_trials = int(args.trials)
    for i in range(n_trials):
        params = sample_uniform() if i < n_startup or len(trials) < 6 else propose_tpe()
        score = objective(params)
        trials.append((params, score))
        done = i + 1
        elapsed = max(time.time() - started, 1e-6)
        avg_per_trial = elapsed / float(done)
        eta = max(0.0, avg_per_trial * float(n_trials - done))
        mins = int(eta // 60)
        secs = int(eta % 60)
        print(
            f"\r[Bayesian] trial {done}/{n_trials}  elapsed={elapsed:.1f}s  eta={mins:02d}:{secs:02d}",
            end="",
            file=sys.stderr,
            flush=True,
        )
    print("", file=sys.stderr, flush=True)

    best_params, best_score = max(trials, key=lambda t: t[1])
    p = best_params
    p["pe_exit_threshold"] = float(
        _clamp(
            max(float(p["pe_exit_threshold"]), float(p["pe_entry_threshold"]) + 0.05),
            bounds["pe_exit_threshold"][0],
            bounds["pe_exit_threshold"][1],
        )
    )
    best_cfg = replace(
        base_cfg,
        brick_size=float(p["brick_size"]),
        stop_bricks=float(p["stop_bricks"]),
        target_risk_usd=float(p["target_risk"]),
        fliprate_threshold=float(p["fliprate_threshold"]),
        markov_threshold=float(p["markov_threshold"]),
        trailing_mfe_fraction=float(p["trailing_mfe_fraction"]),
        trailing_mfe_after_bricks=int(round(p["trailing_mfe_after_bricks"])),
        pe_entry_threshold=float(p["pe_entry_threshold"]),
        pe_exit_threshold=float(p["pe_exit_threshold"]),
        stale_brick_factor=float(p["stale_brick_factor"]),
        markov_stale_penalty=float(p["markov_stale_penalty"]),
    )
    best_engine = renko.RenkoEngine(best_cfg, quiet_mode=True)
    best_res = best_engine.backtest(test_closes)
    best_summary: dict[str, Any] = best_res.get("summary", {}) or {}
    best_omega = float(best_summary.get("omega", 0.0))
    best_trades = int(best_summary.get("n_trades", 0))

    # Gate defaults for post-optimization qualification checks.
    min_omega = _clamp(best_omega * 0.85, 1.1, 2.2)
    min_trades = int(_clamp(max(20, best_trades * 0.60), 20, 80))

    print(f"BRICK_SIZE={best_cfg.brick_size:.6f}")
    print(f"STOP_BRICKS={best_cfg.stop_bricks:.6f}")
    print(f"TARGET_RISK={best_cfg.target_risk_usd:.6f}")
    print(f"FLIPRATE_THRESHOLD={best_cfg.fliprate_threshold:.6f}")
    print(f"MARKOV_THRESHOLD={best_cfg.markov_threshold:.6f}")
    print(f"TRAILING_MFE_FRACTION={best_cfg.trailing_mfe_fraction:.6f}")
    print(f"TRAILING_MFE_AFTER_BRICKS={int(best_cfg.trailing_mfe_after_bricks):d}")
    print(f"PE_ENTRY_THRESHOLD={best_cfg.pe_entry_threshold:.6f}")
    print(f"PE_EXIT_THRESHOLD={best_cfg.pe_exit_threshold:.6f}")
    print(f"STALE_BRICK_FACTOR={best_cfg.stale_brick_factor:.6f}")
    print(f"MARKOV_STALE_PENALTY={best_cfg.markov_stale_penalty:.6f}")
    print(f"MIN_OMEGA={min_omega:.6f}")
    print(f"MIN_TRADES={min_trades:d}")
    print(f"BAYES_BEST_SCORE={float(best_score):.6f}")
    print(f"BAYES_BEST_OMEGA={best_omega:.6f}")
    print(f"BAYES_BEST_TRADES={best_trades:d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
