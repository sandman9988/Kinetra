#!/usr/bin/env python3
"""
Parallel Performance Benchmark

Tests the parallel backtest runner with multiple instruments.
Benchmarks CPU and GPU performance.
"""

import time
from pathlib import Path

import pandas as pd

from kinetra.parallel import (
    ParallelBacktestRunner,
    get_system_resources,
    parallel_map,
    print_parallel_info,
)
from kinetra.symbol_spec import SymbolSpec


def load_mt5_data(filepath: Path) -> pd.DataFrame:
    """Load MT5 format CSV data."""
    df = pd.read_csv(filepath, sep="\t")
    df["time"] = pd.to_datetime(df["<DATE>"] + " " + df["<TIME>"])
    df = df.set_index("time")
    df = df.rename(
        columns={
            "<OPEN>": "open",
            "<HIGH>": "high",
            "<LOW>": "low",
            "<CLOSE>": "close",
            "<TICKVOL>": "tick_volume",
            "<VOL>": "volume",
            "<SPREAD>": "spread",
        }
    )
    df = df.drop(columns=["<DATE>", "<TIME>"], errors="ignore")
    return df


def create_symbol_spec(symbol: str) -> SymbolSpec:
    """Create appropriate symbol spec based on symbol name."""
    if "BTC" in symbol or "ETH" in symbol:
        return SymbolSpec(
            symbol=symbol,
            contract_size=1.0,
            tick_size=0.01,
            tick_value=0.01,
            digits=2,
            margin_initial=0.10,
            spread_points=50.0,
        )
    elif "XAU" in symbol or "GOLD" in symbol:
        return SymbolSpec(
            symbol=symbol,
            contract_size=100.0,
            tick_size=0.01,
            tick_value=1.0,
            digits=2,
            margin_initial=0.01,
            spread_points=30.0,
        )
    else:
        return SymbolSpec(
            symbol=symbol,
            contract_size=1.0,
            tick_size=1.0,
            tick_value=1.0,
            digits=0,
            margin_initial=0.05,
            spread_points=5.0,
        )


def simple_signal(symbol: str, row: pd.Series, physics: pd.DataFrame, bar_idx: int) -> int:
    """Simple momentum signal for testing."""
    if bar_idx < 20 or physics is None or bar_idx >= len(physics):
        return 0

    phys = physics.iloc[bar_idx]
    energy_pct = phys.get("energy_pct", 0.5)

    if energy_pct > 0.7:
        return 1  # Long
    elif energy_pct < 0.3:
        return -1  # Short
    return 0


def main():
    print("=" * 70)
    print("PARALLEL PERFORMANCE BENCHMARK")
    print("=" * 70)
    print()

    # System info
    print_parallel_info()
    print()

    # Load all available data
    data_dir = Path("data/master")
    h1_files = list(data_dir.glob("*_H1_*.csv"))

    print(f"Loading {len(h1_files)} instruments...")

    data_dict = {}
    symbol_specs = {}

    for filepath in h1_files:
        symbol = filepath.name.split("_")[0]
        try:
            df = load_mt5_data(filepath)
            data_dict[symbol] = df
            symbol_specs[symbol] = create_symbol_spec(symbol)
            print(f"  {symbol}: {len(df):,} bars")
        except Exception as e:
            print(f"  {symbol}: Error - {e}")

    total_bars = sum(len(df) for df in data_dict.values())
    print(f"\nTotal: {len(data_dict)} instruments, {total_bars:,} bars")
    print()

    # Benchmark 1: Sequential vs Parallel (1 worker vs N workers)
    print("=" * 70)
    print("BENCHMARK 1: Sequential vs Parallel")
    print("=" * 70)

    # Sequential (1 worker)
    print("\n[1 Worker - Sequential]")
    runner_seq = ParallelBacktestRunner(n_workers=1, use_gpu=False, verbose=False)
    start = time.time()
    result_seq = runner_seq.run_multi_instrument(
        data_dict=data_dict,
        symbol_specs=symbol_specs,
        signal_func=simple_signal,
        precompute_physics=False,
    )
    time_seq = time.time() - start
    print(f"  Time: {time_seq:.2f}s")
    print(f"  Trades: {result_seq['total_trades']}")
    print(f"  PnL: ${result_seq['total_net_pnl']:,.2f}")

    # Parallel (30 workers)
    print("\n[30 Workers - Parallel]")
    runner_par = ParallelBacktestRunner(n_workers=30, use_gpu=False, verbose=False)
    start = time.time()
    result_par = runner_par.run_multi_instrument(
        data_dict=data_dict,
        symbol_specs=symbol_specs,
        signal_func=simple_signal,
        precompute_physics=False,
    )
    time_par = time.time() - start
    print(f"  Time: {time_par:.2f}s")
    print(f"  Trades: {result_par['total_trades']}")
    print(f"  PnL: ${result_par['total_net_pnl']:,.2f}")

    speedup = time_seq / time_par if time_par > 0 else 0
    print(f"\n  Speedup: {speedup:.1f}x faster with 30 workers")

    # Benchmark 2: With GPU physics pre-computation
    print("\n" + "=" * 70)
    print("BENCHMARK 2: CPU vs GPU Physics")
    print("=" * 70)

    resources = get_system_resources()
    if resources.gpu_available:
        print("\n[CPU Physics]")
        runner_cpu = ParallelBacktestRunner(n_workers=30, use_gpu=False, verbose=False)
        start = time.time()
        result_cpu = runner_cpu.run_multi_instrument(
            data_dict=data_dict,
            symbol_specs=symbol_specs,
            signal_func=simple_signal,
            precompute_physics=True,
        )
        time_cpu = time.time() - start
        print(f"  Time: {time_cpu:.2f}s")

        print("\n[GPU Physics]")
        runner_gpu = ParallelBacktestRunner(n_workers=30, use_gpu=True, verbose=False)
        start = time.time()
        result_gpu = runner_gpu.run_multi_instrument(
            data_dict=data_dict,
            symbol_specs=symbol_specs,
            signal_func=simple_signal,
            precompute_physics=True,
        )
        time_gpu = time.time() - start
        print(f"  Time: {time_gpu:.2f}s")

        gpu_speedup = time_cpu / time_gpu if time_gpu > 0 else 0
        print(f"\n  GPU Speedup: {gpu_speedup:.1f}x faster")
    else:
        print("  GPU not available, skipping GPU benchmark")

    # Benchmark 3: Scaling test
    print("\n" + "=" * 70)
    print("BENCHMARK 3: Worker Scaling")
    print("=" * 70)

    worker_counts = [1, 2, 4, 8, 16, 24, 30]
    results = []

    for n_workers in worker_counts:
        runner = ParallelBacktestRunner(n_workers=n_workers, use_gpu=False, verbose=False)
        start = time.time()
        result = runner.run_multi_instrument(
            data_dict=data_dict,
            symbol_specs=symbol_specs,
            signal_func=simple_signal,
            precompute_physics=False,
        )
        elapsed = time.time() - start
        results.append((n_workers, elapsed))
        print(f"  {n_workers:2d} workers: {elapsed:.2f}s ({total_bars / elapsed:,.0f} bars/sec)")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Instruments tested: {len(data_dict)}")
    print(f"Total bars processed: {total_bars:,}")
    print(f"Best time: {min(r[1] for r in results):.2f}s")
    print(f"Throughput: {total_bars / min(r[1] for r in results):,.0f} bars/second")
    print(f"Sequential speedup: {speedup:.1f}x")
    if resources.gpu_available:
        print(f"GPU physics speedup: {gpu_speedup:.1f}x")
    print("=" * 70)


if __name__ == "__main__":
    main()
