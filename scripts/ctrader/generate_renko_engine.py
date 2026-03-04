#!/usr/bin/env python3
"""Generate Renko engine artifacts from cTrader standardized data.

Reads latest per-symbol M1 CSV + contract_spec.json and runs:
- qualification-only (default), or
- full portfolio pipeline.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from kinetra.monitoring import emit_event, emit_health, telemetry_span
from kinetra.renko.backtest import (
    FilterParams,
    StopParams,
    backtest_instrument,
    walk_forward_instrument,
)
from kinetra.renko.brick_engine import bricks_per_day, build_renko
from kinetra.renko.dsp import run_dsp, scaled_filter_params
from kinetra.renko.orchestrator import run_full_pipeline, run_qualification_only
from kinetra.renko.session import detect_session_break

logger = logging.getLogger("ctrader.renko_engine")


def _is_ecn_commission_symbol(symbol: str) -> bool:
    s = (symbol or "").upper()
    if not s:
        return False
    if s.startswith(("XAU", "XAG", "XPT", "XPD")):
        return True
    return bool(re.fullmatch(r"[A-Z]{6}", s))


def _parse_symbols(raw: str) -> List[str]:
    out = [s.strip().upper() for s in raw.split(",") if s.strip()]
    if not out:
        raise ValueError("No symbols specified")
    return out


def _resolve_symbol_dir(data_root: Path, symbol: str) -> Path:
    matches = [p for p in data_root.glob(f"*/{symbol}") if p.is_dir()]
    if not matches:
        raise FileNotFoundError(f"{symbol}: no directory found under {data_root}")
    if len(matches) > 1:
        # Prefer the directory with a contract_spec if multiple categories/accounts exist.
        with_spec = [m for m in matches if (m / "contract_spec.json").exists()]
        if with_spec:
            matches = with_spec
    return sorted(matches)[-1]


def _latest_m1_csv(symbol_dir: Path, symbol: str) -> Path:
    files = sorted(symbol_dir.glob(f"{symbol}_M1_*.csv"))
    if not files:
        raise FileNotFoundError(f"{symbol}: no *_M1_*.csv in {symbol_dir}")
    return files[-1]


def _load_contract_spec(spec_path: Path) -> Tuple[float, float, float, float, float]:
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    spread = float(
        spec.get("spread_typical_pts") or spec.get("spread_points") or spec.get("spread") or 1.0
    )
    tick = float(spec.get("tick_size") or spec.get("tickSize") or spec.get("point_value") or 0.0001)
    raw_commission = spec.get("commission_per_lot")
    if raw_commission is None:
        raw_commission = spec.get("commission")
    commission = float(raw_commission or 0.0)
    is_commission_free = bool(spec.get("is_commission_free", False))
    if commission <= 0.0 and not is_commission_free:
        symbol = str(spec.get("symbol") or spec_path.parent.name).upper()
        if _is_ecn_commission_symbol(symbol):
            commission = float(spec.get("commission_default_per_lot") or 3.5)
    swap_long = float(spec.get("swap_long_points") or 0.0)
    swap_short = float(spec.get("swap_short_points") or 0.0)
    if spread <= 0:
        spread = 1.0
    if tick <= 0:
        tick = 0.0001
    return spread, tick, commission, swap_long, swap_short


def _normalize_ohlcv(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    col_lower = {c.lower(): c for c in df.columns}
    rename = {}
    for canon in ("time", "open", "high", "low", "close", "volume", "spread"):
        if canon in df.columns:
            continue
        if canon in col_lower:
            rename[col_lower[canon]] = canon
    if rename:
        df = df.rename(columns=rename)

    if "volume" not in df.columns:
        for alias in ("tick_volume", "tickvolume", "vol"):
            if alias in col_lower:
                df = df.rename(columns={col_lower[alias]: "volume"})
                break

    required = ["time", "open", "high", "low", "close", "volume"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{symbol}: missing required columns {missing}")

    df = df[
        ["time", "open", "high", "low", "close", "volume"]
        + (["spread"] if "spread" in df.columns else [])
    ]
    df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
    df = df.dropna(subset=["time"])
    for c in ("open", "high", "low", "close", "volume"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    if "spread" in df.columns:
        df["spread"] = pd.to_numeric(df["spread"], errors="coerce")
    df = df.dropna(subset=["open", "high", "low", "close", "volume"])
    df = df.sort_values("time").reset_index(drop=True)
    return df


def _load_inputs(
    data_root: Path,
    symbols: List[str],
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, Tuple[float, float, float, float, float]]]:
    m1_data: Dict[str, pd.DataFrame] = {}
    spread_specs: Dict[str, Tuple[float, float, float, float, float]] = {}

    for symbol in symbols:
        sym_dir = _resolve_symbol_dir(data_root, symbol)
        csv_path = _latest_m1_csv(sym_dir, symbol)
        spec_path = sym_dir / "contract_spec.json"
        if not spec_path.exists():
            raise FileNotFoundError(f"{symbol}: missing contract_spec.json in {sym_dir}")

        df = pd.read_csv(csv_path)
        df = _normalize_ohlcv(df, symbol)
        if df.empty:
            raise ValueError(f"{symbol}: M1 dataframe empty after normalization: {csv_path}")

        spread_pts, tick_size, commission_per_lot, swap_long, swap_short = _load_contract_spec(
            spec_path
        )
        m1_data[symbol] = df
        spread_specs[symbol] = (
            spread_pts,
            tick_size,
            commission_per_lot,
            swap_long,
            swap_short,
        )
        logger.info(
            "%s: loaded %d bars from %s (spread_pts=%.4f tick=%.6f comm_per_lot=%.4f swapL=%.4f swapS=%.4f)",
            symbol,
            len(df),
            csv_path.name,
            spread_pts,
            tick_size,
            commission_per_lot,
            swap_long,
            swap_short,
        )

    return m1_data, spread_specs


def _m30_closes_from_m1(df_m1: pd.DataFrame) -> pd.Series:
    idx = pd.to_datetime(df_m1["time"], utc=True, errors="coerce")
    ser = pd.Series(pd.to_numeric(df_m1["close"], errors="coerce").values, index=idx).dropna()
    ser = ser[~ser.index.isna()].sort_index()
    return ser.resample("30min").last().dropna()


def _limited_precheck(
    symbol: str,
    m1_df: pd.DataFrame,
    *,
    limited_m1_bars: int,
    min_trades: int,
    min_omega: float,
    min_z: float,
    oos_min_omega: float,
    oos_min_z: float,
    oos_min_trades: int,
) -> Tuple[bool, Dict[str, object]]:
    m1_slice = m1_df.tail(max(limited_m1_bars, 1)).copy()
    if len(m1_slice) < 2000:
        return False, {"reason": f"insufficient M1 bars for precheck: {len(m1_slice)}"}

    m30_closes = _m30_closes_from_m1(m1_slice)
    if len(m30_closes) < 200:
        return False, {"reason": f"insufficient M30 bars for precheck: {len(m30_closes)}"}

    try:
        session = detect_session_break(m1_slice, symbol=symbol, broker_source="ctrader")
        session_break_minutes = float(session.session_break_minutes)
    except Exception as exc:
        logger.debug("%s: precheck session detect failed (%s), using default 30m", symbol, exc)
        session_break_minutes = 30.0

    dsp = run_dsp(m30_closes.values, symbol=symbol)
    brick_size = float(max(dsp.dsp_brick_size, 1e-10))
    try:
        temp_bricks = build_renko(
            m30_closes,
            brick_size=brick_size,
            session_break_minutes=session_break_minutes,
        )
        bpd = float(max(bricks_per_day(temp_bricks), 1.0))
    except Exception:
        bpd = 5.0
    filter_params = scaled_filter_params(bpd)
    stop_params = StopParams()

    bt = backtest_instrument(
        symbol=symbol,
        closes=m30_closes,
        brick_size=brick_size,
        filter_params=filter_params,
        stop_params=stop_params,
        session_break_minutes=session_break_minutes,
    )
    wf = walk_forward_instrument(
        symbol=symbol,
        closes=m30_closes,
        brick_size=brick_size,
        filter_params=filter_params,
        stop_params=stop_params,
        oos_min_omega=oos_min_omega,
        oos_min_z=oos_min_z,
        oos_min_trades=oos_min_trades,
        session_break_minutes=session_break_minutes,
    )

    passed = (
        len(bt.trades) >= min_trades
        and bt.omega >= min_omega
        and bt.z_factor >= min_z
        and (wf is None or wf.oos_passed)
    )
    details: Dict[str, object] = {
        "n_trades": int(len(bt.trades)),
        "omega": float(bt.omega),
        "z_factor": float(bt.z_factor),
        "brick_size": brick_size,
        "session_break_minutes": session_break_minutes,
        "m30_bars": int(len(m30_closes)),
        "oos_passed": bool(wf.oos_passed) if wf is not None else None,
        "oos_omega": float(wf.oos_omega) if wf is not None else None,
        "oos_z": float(wf.oos_z) if wf is not None else None,
        "filter_params": {
            "fliprate_window": int(filter_params.fliprate_window),
            "fliprate_threshold": float(filter_params.fliprate_threshold),
            "markov_window": int(filter_params.markov_window),
            "markov_threshold": float(filter_params.markov_threshold),
        },
    }
    return passed, details


def _rolling_oos_summary(
    symbol: str,
    m1_df: pd.DataFrame,
    *,
    brick_size: float,
    filter_params: FilterParams,
    stop_params: Optional[StopParams],
    session_break_minutes: float,
    window_bars: int,
    step_bars: int,
    oos_min_omega: float,
    oos_min_z: float,
    oos_min_trades: int,
) -> Dict[str, object]:
    m30_closes = _m30_closes_from_m1(m1_df)
    n = len(m30_closes)
    if n < max(window_bars, 120):
        return {"symbol": symbol, "n_windows": 0, "pass_rate": 0.0, "windows": []}

    windows: List[Dict[str, object]] = []
    win = max(window_bars, 100)
    step = max(step_bars, 10)
    for start in range(0, n - win + 1, step):
        seg = m30_closes.iloc[start : start + win]
        wf = walk_forward_instrument(
            symbol=symbol,
            closes=seg,
            brick_size=brick_size,
            filter_params=filter_params,
            stop_params=stop_params,
            oos_min_omega=oos_min_omega,
            oos_min_z=oos_min_z,
            oos_min_trades=oos_min_trades,
            session_break_minutes=session_break_minutes,
        )
        if wf is None:
            continue
        windows.append(
            {
                "start": str(seg.index[0]),
                "end": str(seg.index[-1]),
                "is_trades": int(wf.is_trades),
                "is_omega": float(wf.is_omega),
                "is_z": float(wf.is_z),
                "oos_trades": int(wf.oos_trades),
                "oos_omega": float(wf.oos_omega),
                "oos_z": float(wf.oos_z),
                "oos_passed": bool(wf.oos_passed),
            }
        )

    n_windows = len(windows)
    n_pass = sum(1 for w in windows if w["oos_passed"])
    pass_rate = (n_pass / n_windows) if n_windows > 0 else 0.0
    return {
        "symbol": symbol,
        "n_windows": n_windows,
        "n_passed": n_pass,
        "pass_rate": pass_rate,
        "windows": windows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate Renko engine artifacts from cTrader data"
    )
    parser.add_argument(
        "--symbols", required=True, help="Comma-separated symbols, e.g. XAUUSD,NAS100"
    )
    parser.add_argument(
        "--data-root",
        default="data/master_standardized/ctrader/pepperstone_demo_45841299",
        help="Root containing category/symbol folders",
    )
    parser.add_argument(
        "--mode",
        choices=["qualification", "full", "staged"],
        default="qualification",
        help=(
            "qualification: build registry only; "
            "full: run portfolio pipeline; "
            "staged: limited precheck -> full pipeline (passed symbols) + rolling OOS report"
        ),
    )
    parser.add_argument("--broker-source", default="ctrader")
    parser.add_argument("--output-dir", default="data/renko_qualified")
    parser.add_argument("--results-dir", default="results/renko")
    parser.add_argument("--n-workers", type=int, default=1)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--run-mc", action="store_true")
    parser.add_argument("--mc-runs", type=int, default=500)
    parser.add_argument("--limited-m1-bars", type=int, default=30000)
    parser.add_argument("--limited-min-trades", type=int, default=5)
    parser.add_argument("--limited-min-omega", type=float, default=1.0)
    parser.add_argument("--limited-min-z", type=float, default=0.0)
    parser.add_argument("--rolling-window-bars", type=int, default=900)
    parser.add_argument("--rolling-step-bars", type=int, default=225)
    parser.add_argument("--oos-min-omega", type=float, default=1.0)
    parser.add_argument("--oos-min-z", type=float, default=0.0)
    parser.add_argument("--oos-min-trades", type=int, default=3)
    parser.add_argument(
        "--staged-include-all-symbols",
        action="store_true",
        help="In staged mode, include all requested symbols in full pipeline even if limited precheck fails.",
    )
    parser.add_argument(
        "--staged-report-name",
        default="staged_oos_report.json",
        help="Output filename for staged-mode OOS report under --results-dir.",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    symbols = _parse_symbols(args.symbols)
    data_root = (PROJECT_ROOT / args.data_root).resolve()
    output_dir = (PROJECT_ROOT / args.output_dir).resolve()
    results_dir = (PROJECT_ROOT / args.results_dir).resolve()
    emit_event(
        stream="system",
        component="renko_engine",
        event_type="run_start",
        status="info",
        payload={
            "mode": args.mode,
            "symbols": symbols,
            "data_root": str(data_root),
        },
    )

    with telemetry_span(
        stream="system",
        component="renko_engine",
        operation="load_inputs",
        payload={"symbols": symbols},
    ):
        m1_data, spread_specs = _load_inputs(data_root, symbols)

    if args.mode == "qualification":
        registry = run_qualification_only(
            m1_data=m1_data,
            spread_specs=spread_specs,
            output_dir=output_dir,
            broker_source=args.broker_source,
            force=args.force,
            n_workers=max(args.n_workers, 1),
        )
        total_results = len(registry.all_results())
        print(
            "[OK] qualification complete: "
            f"qualified={registry.qualified_count}/{total_results} "
            f"results written under {output_dir}"
        )
        emit_health(
            component="renko_engine",
            status="ok",
            checks={"qualification": "pass"},
            metrics={
                "qualified": int(registry.qualified_count),
                "total": int(total_results),
            },
        )
        return 0

    if args.mode == "staged":
        passed_symbols: List[str] = []
        staged_meta: Dict[str, Dict[str, object]] = {}

        for symbol in symbols:
            passed, details = _limited_precheck(
                symbol,
                m1_data[symbol],
                limited_m1_bars=args.limited_m1_bars,
                min_trades=args.limited_min_trades,
                min_omega=args.limited_min_omega,
                min_z=args.limited_min_z,
                oos_min_omega=args.oos_min_omega,
                oos_min_z=args.oos_min_z,
                oos_min_trades=args.oos_min_trades,
            )
            staged_meta[symbol] = details
            status = "PASS" if passed else "FAIL"
            print(
                f"[{status}] limited precheck {symbol}: trades={details.get('n_trades')} "
                f"omega={details.get('omega')} z={details.get('z_factor')} "
                f"oos_passed={details.get('oos_passed')}"
            )
            if passed or args.staged_include_all_symbols:
                passed_symbols.append(symbol)
            emit_event(
                stream="system",
                component="renko_engine",
                event_type="limited_precheck",
                status="ok" if passed else "warn",
                payload={"symbol": symbol, **details},
            )

        if not passed_symbols:
            print("[FAIL] staged mode: no symbols passed limited precheck; skipping full pipeline")
            emit_health(
                component="renko_engine",
                status="critical",
                checks={"staged_precheck": "fail"},
                details={"symbols_requested": symbols},
            )
            return 2

        staged_m1 = {s: m1_data[s] for s in passed_symbols}
        staged_specs = {s: spread_specs[s] for s in passed_symbols}
        result = run_full_pipeline(
            m1_data=staged_m1,
            spread_specs=staged_specs,
            output_dir=output_dir,
            results_dir=results_dir,
            broker_source=args.broker_source,
            force=args.force,
            n_workers=max(args.n_workers, 1),
            run_mc=args.run_mc,
            mc_runs=max(args.mc_runs, 1),
        )

        rolling_reports: Dict[str, Dict[str, object]] = {}
        for symbol in passed_symbols:
            meta = staged_meta[symbol]
            fp_raw = meta.get("filter_params") or {}
            fp = FilterParams(
                fliprate_window=int(fp_raw.get("fliprate_window", 20)),
                fliprate_threshold=float(fp_raw.get("fliprate_threshold", 0.35)),
                markov_window=int(fp_raw.get("markov_window", 20)),
                markov_threshold=float(fp_raw.get("markov_threshold", 0.55)),
            )
            rolling = _rolling_oos_summary(
                symbol=symbol,
                m1_df=m1_data[symbol],
                brick_size=float(meta.get("brick_size") or 0.0),
                filter_params=fp,
                stop_params=StopParams(),
                session_break_minutes=float(meta.get("session_break_minutes") or 30.0),
                window_bars=args.rolling_window_bars,
                step_bars=args.rolling_step_bars,
                oos_min_omega=args.oos_min_omega,
                oos_min_z=args.oos_min_z,
                oos_min_trades=args.oos_min_trades,
            )
            rolling_reports[symbol] = rolling
            print(
                f"[OOS] {symbol}: windows={rolling['n_windows']} "
                f"pass_rate={rolling['pass_rate']:.2%} ({rolling.get('n_passed', 0)}/{rolling['n_windows']})"
            )

        out = {
            "mode": "staged",
            "symbols_requested": symbols,
            "symbols_passed_limited": passed_symbols,
            "limited_precheck": staged_meta,
            "rolling_oos": rolling_reports,
            "pipeline": {
                "qualified": int(result.n_qualified),
                "total": int(result.n_instruments),
                "omega": float(result.portfolio_omega),
                "z": float(result.portfolio_z),
                "deployment_ready": bool(result.deployment_ready),
            },
        }
        results_dir.mkdir(parents=True, exist_ok=True)
        staged_path = results_dir / args.staged_report_name
        staged_path.write_text(json.dumps(out, indent=2), encoding="utf-8")

        print(
            "[OK] staged pipeline complete: "
            f"limited_pass={len(passed_symbols)}/{len(symbols)} "
            f"qualified={result.n_qualified}/{result.n_instruments} "
            f"deployment_ready={result.deployment_ready} "
            f"staged_report={staged_path}"
        )
        emit_health(
            component="renko_engine",
            status="ok" if result.deployment_ready else "warn",
            checks={"staged_pipeline": "pass"},
            metrics={
                "limited_pass": len(passed_symbols),
                "requested": len(symbols),
                "qualified": int(result.n_qualified),
                "total": int(result.n_instruments),
                "deployment_ready": bool(result.deployment_ready),
            },
            details={"staged_report": str(staged_path)},
        )
        return 0

    result = run_full_pipeline(
        m1_data=m1_data,
        spread_specs=spread_specs,
        output_dir=output_dir,
        results_dir=results_dir,
        broker_source=args.broker_source,
        force=args.force,
        n_workers=max(args.n_workers, 1),
        run_mc=args.run_mc,
        mc_runs=max(args.mc_runs, 1),
    )
    print(
        "[OK] full pipeline complete: "
        f"qualified={result.n_qualified}/{result.n_instruments} "
        f"omega={result.portfolio_omega:.3f} z={result.portfolio_z:.3f} "
        f"deployment_ready={result.deployment_ready} "
        f"result_file={results_dir / 'portfolio_result.json'}"
    )
    emit_health(
        component="renko_engine",
        status="ok" if result.deployment_ready else "warn",
        checks={"full_pipeline": "pass"},
        metrics={
            "qualified": int(result.n_qualified),
            "total": int(result.n_instruments),
            "omega": float(result.portfolio_omega),
            "z": float(result.portfolio_z),
            "deployment_ready": bool(result.deployment_ready),
        },
        details={"result_file": str(results_dir / "portfolio_result.json")},
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
