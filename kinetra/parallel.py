"""
Parallel processing utilities for backtesting.

Provides CPU multiprocessing and GPU acceleration for:
- Multi-instrument backtests (parallel per instrument)
- Physics calculations (vectorized GPU)
- Walk-forward optimization (parallel windows)

System: AMD Ryzen 9 5950X (32 cores), 128GB RAM, AMD RX 7600 (8GB VRAM)
"""

from __future__ import annotations

import gc
import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import psutil

# Try to import torch for GPU acceleration
try:
    import torch

    TORCH_AVAILABLE = True
    CUDA_AVAILABLE = torch.cuda.is_available()
    # Check for ROCm (AMD GPU)
    ROCM_AVAILABLE = CUDA_AVAILABLE and "rocm" in torch.__config__.show().lower()
except ImportError:
    TORCH_AVAILABLE = False
    CUDA_AVAILABLE = False
    ROCM_AVAILABLE = False


@dataclass
class SystemResources:
    """System resource information."""

    cpu_cores: int
    cpu_threads: int
    ram_total_gb: float
    ram_available_gb: float
    gpu_available: bool
    gpu_name: str
    gpu_vram_gb: float

    @classmethod
    def detect(cls) -> "SystemResources":
        """Detect system resources."""
        mem = psutil.virtual_memory()
        cpu_count = mp.cpu_count()

        gpu_name = ""
        gpu_vram = 0.0
        gpu_available = False

        if TORCH_AVAILABLE and CUDA_AVAILABLE:
            import torch

            gpu_available = True
            gpu_name = torch.cuda.get_device_name(0)
            gpu_vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)

        return cls(
            cpu_cores=cpu_count // 2,  # Physical cores
            cpu_threads=cpu_count,
            ram_total_gb=mem.total / (1024**3),
            ram_available_gb=mem.available / (1024**3),
            gpu_available=gpu_available,
            gpu_name=gpu_name,
            gpu_vram_gb=gpu_vram,
        )


@dataclass
class ParallelConfig:
    """Configuration for parallel processing."""

    # CPU settings
    n_workers: int = -1  # -1 = all cores
    use_threading: bool = False  # Use threads instead of processes (for I/O bound)

    # GPU settings
    use_gpu: bool = True  # Use GPU if available
    gpu_device: int = 0  # GPU device index
    gpu_batch_size: int = 10000  # Batch size for GPU operations

    # Memory settings
    max_ram_usage_pct: float = 0.80  # Use up to 80% of available RAM
    chunk_size_mb: int = 1000  # Process data in chunks of this size

    def __post_init__(self):
        if self.n_workers == -1:
            self.n_workers = mp.cpu_count()


def get_system_resources() -> SystemResources:
    """Get current system resources."""
    return SystemResources.detect()


def get_optimal_workers(leave_cores: int = 2) -> int:
    """Get optimal number of workers based on system."""
    cpu_count = mp.cpu_count()
    # Leave some cores for system/other tasks
    return max(1, cpu_count - leave_cores)


def get_max_memory_per_worker(n_workers: int, max_usage_pct: float = 0.80) -> int:
    """Calculate max memory per worker in bytes."""
    mem = psutil.virtual_memory()
    available = mem.available * max_usage_pct
    return int(available / n_workers)


def get_optimal_batch_size(
    item_size_bytes: int,
    n_items: int,
    max_memory_pct: float = 0.50,
) -> int:
    """
    Calculate optimal batch size based on available memory.

    Args:
        item_size_bytes: Size of each item in bytes
        n_items: Total number of items
        max_memory_pct: Max percentage of available RAM to use

    Returns:
        Optimal batch size
    """
    mem = psutil.virtual_memory()
    available = mem.available * max_memory_pct
    max_items = int(available / item_size_bytes) if item_size_bytes > 0 else n_items
    return min(max_items, n_items)


def get_device() -> str:
    """Get the best available device (cuda/rocm or cpu)."""
    if TORCH_AVAILABLE and CUDA_AVAILABLE:
        return "cuda"
    return "cpu"


# =============================================================================
# CPU Parallel Processing
# =============================================================================


def parallel_map(
    func: Callable,
    items: List[Any],
    n_workers: int = -1,
    use_threading: bool = False,
    show_progress: bool = False,
) -> List[Any]:
    """
    Apply function to items in parallel.

    Args:
        func: Function to apply to each item
        items: List of items to process
        n_workers: Number of workers (-1 = all cores)
        use_threading: Use threads instead of processes
        show_progress: Show progress bar

    Returns:
        List of results in same order as items
    """
    if n_workers == -1:
        n_workers = get_optimal_workers()

    if n_workers == 1 or len(items) == 1:
        # Sequential execution for single item or worker
        return [func(item) for item in items]

    executor_class = ThreadPoolExecutor if use_threading else ProcessPoolExecutor

    results = [None] * len(items)
    with executor_class(max_workers=n_workers) as executor:
        # Submit all tasks with their indices
        future_to_idx = {executor.submit(func, item): i for i, item in enumerate(items)}

        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                results[idx] = future.result()
            except Exception as e:
                print(f"Error processing item {idx}: {e}")
                results[idx] = None

    return results


def parallel_backtest_instruments(
    engine_factory: Callable,
    data_dict: Dict[str, pd.DataFrame],
    symbol_specs: Dict[str, Any],
    signal_func: Callable,
    n_workers: int = -1,
) -> Dict[str, Any]:
    """
    Run backtests for multiple instruments in parallel.

    Each instrument gets its own engine instance and runs independently.
    Results are aggregated at the end.

    Args:
        engine_factory: Function that creates a BacktestEngine instance
        data_dict: Dict of symbol -> DataFrame
        symbol_specs: Dict of symbol -> SymbolSpec
        signal_func: Signal generation function
        n_workers: Number of parallel workers

    Returns:
        Dict with aggregated results
    """

    def run_single_instrument(args: Tuple[str, pd.DataFrame, Any]) -> Tuple[str, Any]:
        symbol, data, spec = args
        engine = engine_factory()

        # Wrap signal func to include symbol
        def wrapped_signal(row, physics, bar_idx):
            return signal_func(symbol, row, physics, bar_idx)

        result = engine.run_backtest(data=data, symbol_spec=spec, signal_func=wrapped_signal)
        return symbol, result

    # Prepare items for parallel processing
    items = [(sym, data_dict[sym], symbol_specs[sym]) for sym in data_dict.keys()]

    # Run in parallel
    results = parallel_map(run_single_instrument, items, n_workers=n_workers)

    # Aggregate results
    instrument_results = {sym: result for sym, result in results if result is not None}

    return {
        "instrument_results": instrument_results,
        "total_trades": sum(r.total_trades for r in instrument_results.values()),
        "total_net_pnl": sum(r.total_net_pnl for r in instrument_results.values()),
    }


# =============================================================================
# GPU Accelerated Physics Calculations
# =============================================================================


class GPUPhysicsEngine:
    """GPU-accelerated physics calculations using PyTorch."""

    def __init__(self, device: str = "auto", batch_size: int = 10000):
        if device == "auto":
            self.device = get_device()
        else:
            self.device = device

        self.batch_size = batch_size
        self._initialized = False

        if self.device == "cuda" and TORCH_AVAILABLE:
            self._init_gpu()

    def _init_gpu(self):
        """Initialize GPU resources."""
        # Set optimal settings for AMD ROCm
        if ROCM_AVAILABLE:
            # Disable features that don't work well with ROCm
            torch.backends.cudnn.benchmark = True
        self._initialized = True

    def compute_physics_batch(
        self,
        close_prices: np.ndarray,
        lookback: int = 20,
    ) -> Dict[str, np.ndarray]:
        """
        Compute physics metrics for a batch of price series.

        Args:
            close_prices: 2D array of shape (n_series, n_bars)
            lookback: Lookback period for calculations

        Returns:
            Dict with physics metrics arrays
        """
        if not TORCH_AVAILABLE or self.device == "cpu":
            return self._compute_physics_cpu(close_prices, lookback)

        return self._compute_physics_gpu(close_prices, lookback)

    def _compute_physics_gpu(
        self,
        close_prices: np.ndarray,
        lookback: int,
    ) -> Dict[str, np.ndarray]:
        """GPU-accelerated physics computation."""
        # Convert to torch tensor on GPU
        prices = torch.from_numpy(close_prices.astype(np.float32)).to(self.device)

        n_series, n_bars = prices.shape

        # Compute returns
        returns = torch.zeros_like(prices)
        returns[:, 1:] = (prices[:, 1:] - prices[:, :-1]) / prices[:, :-1]

        # Rolling statistics using unfold
        # Energy: rolling std of returns (volatility)
        energy = torch.zeros_like(prices)
        for i in range(lookback, n_bars):
            window = returns[:, i - lookback : i]
            energy[:, i] = window.std(dim=1)

        # Damping: rolling autocorrelation (mean reversion tendency)
        damping = torch.zeros_like(prices)
        for i in range(lookback + 1, n_bars):
            window = returns[:, i - lookback : i]
            window_lag = returns[:, i - lookback - 1 : i - 1]
            # Simplified autocorrelation
            mean_w = window.mean(dim=1, keepdim=True)
            mean_l = window_lag.mean(dim=1, keepdim=True)
            cov = ((window - mean_w) * (window_lag - mean_l)).mean(dim=1)
            var_w = ((window - mean_w) ** 2).mean(dim=1)
            var_l = ((window_lag - mean_l) ** 2).mean(dim=1)
            damping[:, i] = cov / (torch.sqrt(var_w * var_l) + 1e-10)

        # Entropy: distribution of returns (using simplified histogram approach)
        entropy = torch.zeros_like(prices)
        for i in range(lookback, n_bars):
            window = returns[:, i - lookback : i]
            # Use variance as proxy for entropy
            entropy[:, i] = torch.log(window.var(dim=1) + 1e-10)

        # Convert to percentiles
        def to_percentile(arr):
            """Convert to rolling percentile."""
            result = torch.zeros_like(arr)
            for i in range(lookback * 2, n_bars):
                window = arr[:, i - lookback * 2 : i]
                current = arr[:, i : i + 1]
                result[:, i] = (window < current).float().mean(dim=1)
            return result

        energy_pct = to_percentile(energy)
        damping_pct = to_percentile(damping.abs())
        entropy_pct = to_percentile(entropy)

        # Move back to CPU and convert to numpy
        return {
            "energy": energy.cpu().numpy(),
            "damping": damping.cpu().numpy(),
            "entropy": entropy.cpu().numpy(),
            "energy_pct": energy_pct.cpu().numpy(),
            "damping_pct": damping_pct.cpu().numpy(),
            "entropy_pct": entropy_pct.cpu().numpy(),
        }

    def _compute_physics_cpu(
        self,
        close_prices: np.ndarray,
        lookback: int,
    ) -> Dict[str, np.ndarray]:
        """CPU fallback for physics computation."""
        n_series, n_bars = close_prices.shape

        # Compute returns
        returns = np.zeros_like(close_prices)
        returns[:, 1:] = (close_prices[:, 1:] - close_prices[:, :-1]) / close_prices[:, :-1]

        # Rolling statistics
        energy = np.zeros_like(close_prices)
        damping = np.zeros_like(close_prices)
        entropy = np.zeros_like(close_prices)

        for i in range(lookback, n_bars):
            window = returns[:, i - lookback : i]
            energy[:, i] = np.std(window, axis=1)
            entropy[:, i] = np.log(np.var(window, axis=1) + 1e-10)

        # Simple percentile conversion
        def to_percentile(arr):
            result = np.zeros_like(arr)
            for i in range(lookback * 2, n_bars):
                window = arr[:, i - lookback * 2 : i]
                current = arr[:, i : i + 1]
                result[:, i] = (window < current).mean(axis=1)
            return result

        return {
            "energy": energy,
            "damping": damping,
            "entropy": entropy,
            "energy_pct": to_percentile(energy),
            "damping_pct": to_percentile(np.abs(damping)),
            "entropy_pct": to_percentile(entropy),
        }


def compute_physics_parallel(
    data_dict: Dict[str, pd.DataFrame],
    lookback: int = 20,
    use_gpu: bool = True,
    n_workers: int = -1,
) -> Dict[str, pd.DataFrame]:
    """
    Compute physics for multiple instruments in parallel.

    Uses GPU if available, otherwise falls back to parallel CPU.

    Args:
        data_dict: Dict of symbol -> DataFrame with 'close' column
        lookback: Lookback period
        use_gpu: Whether to use GPU acceleration
        n_workers: Number of CPU workers if not using GPU

    Returns:
        Dict of symbol -> DataFrame with physics columns
    """
    symbols = list(data_dict.keys())

    if use_gpu and TORCH_AVAILABLE and CUDA_AVAILABLE:
        # Batch all instruments together for GPU
        gpu_engine = GPUPhysicsEngine()

        # Stack all close prices
        max_len = max(len(df) for df in data_dict.values())
        close_batch = np.zeros((len(symbols), max_len))

        for i, sym in enumerate(symbols):
            df = data_dict[sym]
            close_batch[i, : len(df)] = df["close"].values

        # Compute on GPU
        physics_batch = gpu_engine.compute_physics_batch(close_batch, lookback)

        # Split back to per-instrument DataFrames
        results = {}
        for i, sym in enumerate(symbols):
            df_len = len(data_dict[sym])
            physics_df = pd.DataFrame(
                {col: arr[i, :df_len] for col, arr in physics_batch.items()},
                index=data_dict[sym].index,
            )
            results[sym] = physics_df

        return results

    else:
        # CPU parallel fallback - use threading to avoid pickle issues
        from .physics_engine import PhysicsEngine

        results = {}
        for sym in symbols:
            df = data_dict[sym]
            engine = PhysicsEngine(lookback=lookback)
            physics = engine.compute_physics_state(df["close"])
            results[sym] = physics

        return results


# =============================================================================
# Parallel Walk-Forward Optimization
# =============================================================================


def parallel_walk_forward(
    engine_factory: Callable,
    data: pd.DataFrame,
    symbol_spec: Any,
    signal_func: Callable,
    is_bars: int = 5000,
    oos_bars: int = 1000,
    n_windows: int = 10,
    n_workers: int = -1,
) -> List[Any]:
    """
    Run walk-forward optimization windows in parallel.

    Args:
        engine_factory: Function to create engine instances
        data: Full dataset
        symbol_spec: Symbol specification
        signal_func: Signal generation function
        is_bars: In-sample bars per window
        oos_bars: Out-of-sample bars per window
        n_windows: Number of walk-forward windows
        n_workers: Number of parallel workers

    Returns:
        List of results per window
    """
    total_bars = len(data)
    step_size = (total_bars - is_bars - oos_bars) // (n_windows - 1) if n_windows > 1 else 0

    def run_window(window_idx: int) -> Dict[str, Any]:
        start_idx = window_idx * step_size
        is_end = start_idx + is_bars
        oos_end = is_end + oos_bars

        if oos_end > total_bars:
            return None

        is_data = data.iloc[start_idx:is_end]
        oos_data = data.iloc[is_end:oos_end]

        engine = engine_factory()

        # Run in-sample (training)
        is_result = engine.run_backtest(
            data=is_data, symbol_spec=symbol_spec, signal_func=signal_func
        )

        # Run out-of-sample (validation)
        oos_result = engine.run_backtest(
            data=oos_data, symbol_spec=symbol_spec, signal_func=signal_func
        )

        return {
            "window": window_idx,
            "is_start": is_data.index[0],
            "is_end": is_data.index[-1],
            "oos_start": oos_data.index[0],
            "oos_end": oos_data.index[-1],
            "is_result": is_result,
            "oos_result": oos_result,
            "is_sharpe": is_result.sharpe_ratio,
            "oos_sharpe": oos_result.sharpe_ratio,
        }

    # Run windows in parallel
    results = parallel_map(run_window, list(range(n_windows)), n_workers=n_workers)

    return [r for r in results if r is not None]


# =============================================================================
# Utility Functions
# =============================================================================


def benchmark_parallel(
    func: Callable,
    items: List[Any],
    n_workers_list: List[int] = None,
) -> Dict[int, float]:
    """
    Benchmark parallel execution with different worker counts.

    Args:
        func: Function to benchmark
        items: Items to process
        n_workers_list: List of worker counts to test

    Returns:
        Dict of n_workers -> execution time in seconds
    """
    import time

    if n_workers_list is None:
        max_workers = get_optimal_workers()
        n_workers_list = [1, 2, 4, 8, max_workers]
        n_workers_list = [n for n in n_workers_list if n <= max_workers]

    results = {}
    for n_workers in n_workers_list:
        start = time.time()
        parallel_map(func, items, n_workers=n_workers)
        elapsed = time.time() - start
        results[n_workers] = elapsed
        print(f"  {n_workers} workers: {elapsed:.2f}s")

    return results


def print_parallel_info():
    """Print information about parallel processing capabilities."""
    res = get_system_resources()
    print("=" * 60)
    print("PARALLEL PROCESSING CAPABILITIES")
    print("=" * 60)
    print(f"CPU Cores:        {res.cpu_cores} physical ({res.cpu_threads} threads)")
    print(f"Optimal Workers:  {get_optimal_workers()}")
    print(f"RAM Total:        {res.ram_total_gb:.1f} GB")
    print(f"RAM Available:    {res.ram_available_gb:.1f} GB")
    print(f"PyTorch:          {'Available' if TORCH_AVAILABLE else 'Not installed'}")
    print(f"CUDA/ROCm:        {'Available' if CUDA_AVAILABLE else 'Not available'}")
    if res.gpu_available:
        print(f"GPU Device:       {res.gpu_name}")
        print(f"GPU VRAM:         {res.gpu_vram_gb:.1f} GB")
        print(f"ROCm (AMD):       {'Yes' if ROCM_AVAILABLE else 'No'}")
    print("=" * 60)


# =============================================================================
# High-Performance Parallel Backtest Runner
# =============================================================================


class ParallelBacktestRunner:
    """
    High-performance parallel backtest runner.

    Maximizes CPU and RAM usage for large-scale backtesting:
    - 32 CPU cores for parallel instrument processing
    - 128GB RAM for in-memory data caching
    - GPU acceleration for physics calculations

    Usage:
        runner = ParallelBacktestRunner(n_workers=30, max_ram_pct=0.80)
        results = runner.run_multi_instrument(
            data_dict={'BTCUSD': btc_df, 'XAUUSD': gold_df, ...},
            symbol_specs=specs,
            signal_func=my_signal,
        )
    """

    def __init__(
        self,
        n_workers: int = -1,
        max_ram_pct: float = 0.80,
        use_gpu: bool = True,
        verbose: bool = True,
    ):
        """
        Initialize parallel runner.

        Args:
            n_workers: Number of parallel workers (-1 = auto, uses all but 2 cores)
            max_ram_pct: Maximum RAM usage as percentage (0.80 = 80%)
            use_gpu: Use GPU for physics calculations if available
            verbose: Print progress information
        """
        self.resources = get_system_resources()
        self.n_workers = n_workers if n_workers > 0 else get_optimal_workers()
        self.max_ram_pct = max_ram_pct
        self.use_gpu = use_gpu and self.resources.gpu_available
        self.verbose = verbose

        # Calculate memory budget
        self.max_ram_bytes = int(self.resources.ram_available_gb * 1024**3 * max_ram_pct)
        self.ram_per_worker = self.max_ram_bytes // self.n_workers

        if self.verbose:
            print("ParallelBacktestRunner initialized:")
            print(f"  Workers: {self.n_workers}")
            print(f"  RAM budget: {self.max_ram_bytes / 1024**3:.1f} GB")
            print(f"  RAM per worker: {self.ram_per_worker / 1024**3:.2f} GB")
            print(f"  GPU: {self.resources.gpu_name if self.use_gpu else 'Disabled'}")

    def run_multi_instrument(
        self,
        data_dict: Dict[str, pd.DataFrame],
        symbol_specs: Dict[str, Any],
        signal_func: Callable[[str, pd.Series, pd.DataFrame, int], int],
        initial_capital: float = 10000.0,
        precompute_physics: bool = True,
    ) -> Dict[str, Any]:
        """
        Run parallel backtest across multiple instruments.

        Each instrument runs in a separate process, sharing no state.
        Results are aggregated at the end.

        Args:
            data_dict: Dict of symbol -> DataFrame with OHLCV
            symbol_specs: Dict of symbol -> SymbolSpec
            signal_func: Function(symbol, row, physics, bar_idx) -> signal
            initial_capital: Starting capital per instrument
            precompute_physics: Pre-compute physics on GPU before parallel run

        Returns:
            Dict with per-instrument results and aggregate metrics
        """
        import time

        start_time = time.time()

        symbols = list(data_dict.keys())
        n_instruments = len(symbols)

        if self.verbose:
            total_bars = sum(len(df) for df in data_dict.values())
            print("\nRunning parallel backtest:")
            print(f"  Instruments: {n_instruments}")
            print(f"  Total bars: {total_bars:,}")

        # Step 1: Pre-compute physics on GPU (if enabled)
        physics_dict = None
        if precompute_physics:
            if self.verbose:
                print("  Pre-computing physics (GPU)...", end=" ", flush=True)
            physics_dict = compute_physics_parallel(
                data_dict,
                use_gpu=self.use_gpu,
                n_workers=self.n_workers,
            )
            if self.verbose:
                print("Done")

        # Step 2: Run backtests in parallel
        if self.verbose:
            print(
                f"  Running {n_instruments} backtests on {self.n_workers} workers...",
                end=" ",
                flush=True,
            )

        # Prepare work items
        work_items = [
            (
                sym,
                data_dict[sym],
                symbol_specs[sym],
                physics_dict[sym] if physics_dict else None,
                signal_func,
                initial_capital,
            )
            for sym in symbols
        ]

        # Run in parallel
        results = parallel_map(
            _run_single_backtest,
            work_items,
            n_workers=self.n_workers,
        )

        if self.verbose:
            print("Done")

        # Step 3: Aggregate results
        instrument_results = {}
        all_trades = []
        total_pnl = 0.0

        for sym, result in zip(symbols, results):
            if result is not None:
                instrument_results[sym] = result
                all_trades.extend(result.trades)
                total_pnl += result.total_net_pnl

        elapsed = time.time() - start_time

        if self.verbose:
            print(f"\nCompleted in {elapsed:.2f}s")
            print(f"  Total trades: {len(all_trades)}")
            print(f"  Total PnL: ${total_pnl:,.2f}")

        return {
            "instrument_results": instrument_results,
            "trades": all_trades,
            "total_net_pnl": total_pnl,
            "total_trades": len(all_trades),
            "elapsed_seconds": elapsed,
            "instruments_per_second": n_instruments / elapsed,
        }

    def run_parameter_sweep(
        self,
        data: pd.DataFrame,
        symbol_spec: Any,
        signal_func_factory: Callable[[Dict], Callable],
        param_grid: Dict[str, List[Any]],
        initial_capital: float = 10000.0,
    ) -> List[Dict[str, Any]]:
        """
        Run parallel parameter sweep optimization.

        Args:
            data: Price data DataFrame
            symbol_spec: Symbol specification
            signal_func_factory: Function that takes params dict and returns signal_func
            param_grid: Dict of param_name -> list of values to test
            initial_capital: Starting capital

        Returns:
            List of results for each parameter combination
        """
        import itertools
        import time

        # Generate all parameter combinations
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        combinations = list(itertools.product(*param_values))

        if self.verbose:
            print(f"Parameter sweep: {len(combinations)} combinations")
            print(f"  Parameters: {param_names}")
            print(f"  Using {self.n_workers} workers")

        start_time = time.time()

        # Prepare work items
        work_items = []
        for combo in combinations:
            params = dict(zip(param_names, combo))
            signal_func = signal_func_factory(params)
            work_items.append(
                (
                    "SWEEP",
                    data,
                    symbol_spec,
                    None,  # physics computed per-run
                    signal_func,
                    initial_capital,
                    params,
                )
            )

        # Run in parallel
        results = parallel_map(
            _run_single_backtest_with_params,
            work_items,
            n_workers=self.n_workers,
        )

        elapsed = time.time() - start_time

        if self.verbose:
            print(f"Completed {len(combinations)} runs in {elapsed:.2f}s")
            print(f"  {len(combinations) / elapsed:.1f} runs/second")

        # Sort by performance
        valid_results = [r for r in results if r is not None]
        valid_results.sort(key=lambda x: x.get("sharpe", 0), reverse=True)

        return valid_results

    def run_walk_forward(
        self,
        data: pd.DataFrame,
        symbol_spec: Any,
        signal_func: Callable,
        is_bars: int = 5000,
        oos_bars: int = 1000,
        n_windows: int = None,
        initial_capital: float = 10000.0,
    ) -> List[Dict[str, Any]]:
        """
        Run parallel walk-forward analysis.

        Args:
            data: Full price data
            symbol_spec: Symbol specification
            signal_func: Signal generation function
            is_bars: In-sample bars per window
            oos_bars: Out-of-sample bars per window
            n_windows: Number of windows (auto-calculated if None)
            initial_capital: Starting capital

        Returns:
            List of window results with IS and OOS metrics
        """
        return parallel_walk_forward(
            engine_factory=lambda: _create_backtest_engine(initial_capital),
            data=data,
            symbol_spec=symbol_spec,
            signal_func=signal_func,
            is_bars=is_bars,
            oos_bars=oos_bars,
            n_windows=n_windows or self.n_workers,
            n_workers=self.n_workers,
        )


def _create_backtest_engine(initial_capital: float):
    """Create a BacktestEngine instance (for use in worker processes)."""
    from .backtest_engine import BacktestEngine

    return BacktestEngine(initial_capital=initial_capital)


def _run_single_backtest(args: Tuple) -> Any:
    """Run a single backtest (worker function)."""
    symbol, data, spec, physics, signal_func, initial_capital = args

    from .backtest_engine import BacktestEngine

    engine = BacktestEngine(initial_capital=initial_capital)

    # Wrap signal func to include symbol
    def wrapped_signal(row, phys, bar_idx):
        return signal_func(symbol, row, phys, bar_idx)

    try:
        result = engine.run_backtest(
            data=data,
            symbol_spec=spec,
            signal_func=wrapped_signal,
        )
        return result
    except Exception as e:
        print(f"Error in {symbol}: {e}")
        return None


def _run_single_backtest_with_params(args: Tuple) -> Dict[str, Any]:
    """Run a single backtest with parameters (for parameter sweep)."""
    symbol, data, spec, physics, signal_func, initial_capital, params = args

    from .backtest_engine import BacktestEngine

    engine = BacktestEngine(initial_capital=initial_capital)

    try:
        result = engine.run_backtest(
            data=data,
            symbol_spec=spec,
            signal_func=signal_func,
        )
        return {
            "params": params,
            "total_trades": result.total_trades,
            "win_rate": result.win_rate,
            "total_net_pnl": result.total_net_pnl,
            "sharpe": result.sharpe_ratio,
            "sortino": result.sortino_ratio,
            "max_drawdown_pct": result.max_drawdown_pct,
        }
    except Exception as e:
        return {"params": params, "error": str(e)}
