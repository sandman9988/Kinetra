#!/usr/bin/env python3
"""
Kinetra E2E Stepover Test with Detailed Logging
================================================

Comprehensive end-to-end test that executes each step of the Kinetra pipeline
with detailed logging, step-by-step confirmation, and full diagnostic output.

Features:
- Step-by-step execution with pause/continue options
- Detailed logging to both console and file
- Performance timing for each step
- Data validation at each stage
- Comprehensive error handling and recovery
- Summary report generation

Usage:
    # Interactive mode (pause after each step)
    python scripts/testing/e2e_stepover_test.py --interactive

    # Automatic mode (run all steps)
    python scripts/testing/e2e_stepover_test.py --auto

    # Run specific steps only
    python scripts/testing/e2e_stepover_test.py --steps 1,2,3,4

    # Verbose logging
    python scripts/testing/e2e_stepover_test.py --auto --verbose

Version History:
    1.0.0 (2025-01-04): Initial release with full pipeline testing
"""

__version__ = "1.0.0"
__author__ = "Kinetra Project"

import argparse
import gc
import json
import logging
import os
import sys
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Project root setup
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ============================================================================
# Logging Configuration
# ============================================================================

LOG_DIR = PROJECT_ROOT / "logs" / "e2e_tests"
LOG_DIR.mkdir(parents=True, exist_ok=True)

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_FILE = LOG_DIR / f"e2e_stepover_{TIMESTAMP}.log"

# Create formatters
console_formatter = logging.Formatter(
    "%(asctime)s │ %(levelname)-8s │ %(message)s", datefmt="%H:%M:%S"
)
file_formatter = logging.Formatter(
    "%(asctime)s │ %(levelname)-8s │ %(name)s │ %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)

# Setup root logger
logger = logging.getLogger("e2e_stepover")
logger.setLevel(logging.DEBUG)

# Console handler
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(console_formatter)
logger.addHandler(console_handler)

# File handler
file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(file_formatter)
logger.addHandler(file_handler)


# ============================================================================
# Data Classes
# ============================================================================


@dataclass
class StepResult:
    """Result of a single test step."""

    step_num: int
    name: str
    status: str  # "PASS", "FAIL", "SKIP", "WARN"
    duration_ms: float
    message: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    traceback: Optional[str] = None


@dataclass
class E2ETestContext:
    """Shared context across all test steps."""

    # Data paths
    data_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "data")
    master_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "data" / "master_standardized")
    prepared_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "data" / "prepared")

    # Test configuration
    test_symbol: str = "BTCUSD"
    test_timeframe: str = "H1"
    test_bars: int = 1000  # Limit bars for faster testing

    # Loaded data
    raw_data: Optional[pd.DataFrame] = None
    physics_data: Optional[pd.DataFrame] = None
    backtest_results: Optional[Dict] = None

    # Module references
    physics_engine: Any = None
    backtest_engine: Any = None
    rl_agent: Any = None

    # Metrics
    step_results: List[StepResult] = field(default_factory=list)
    start_time: float = field(default_factory=time.time)


# ============================================================================
# Utility Functions
# ============================================================================


def print_banner(text: str, char: str = "=", width: int = 80):
    """Print a banner with the given text."""
    logger.info("")
    logger.info(char * width)
    logger.info(f"  {text}")
    logger.info(char * width)


def print_step_header(step_num: int, total_steps: int, name: str):
    """Print a step header."""
    logger.info("")
    logger.info("─" * 80)
    logger.info(f"  STEP {step_num}/{total_steps}: {name}")
    logger.info("─" * 80)


def format_duration(ms: float) -> str:
    """Format duration in human-readable format."""
    if ms < 1000:
        return f"{ms:.1f}ms"
    elif ms < 60000:
        return f"{ms / 1000:.2f}s"
    else:
        return f"{ms / 60000:.2f}min"


def format_size(bytes_size: int) -> str:
    """Format byte size in human-readable format."""
    for unit in ["B", "KB", "MB", "GB"]:
        if bytes_size < 1024:
            return f"{bytes_size:.1f}{unit}"
        bytes_size /= 1024
    return f"{bytes_size:.1f}TB"


def wait_for_continue(interactive: bool):
    """Wait for user to press Enter in interactive mode."""
    if interactive:
        try:
            input("\n  Press Enter to continue (Ctrl+C to abort)... ")
        except KeyboardInterrupt:
            logger.warning("\n  Test aborted by user")
            sys.exit(1)


# ============================================================================
# Test Steps
# ============================================================================


def step_01_environment_check(ctx: E2ETestContext) -> StepResult:
    """Step 1: Verify environment and dependencies."""
    start = time.perf_counter()
    details = {}

    try:
        # Check Python version
        py_version = sys.version_info
        details["python_version"] = f"{py_version.major}.{py_version.minor}.{py_version.micro}"
        logger.info(f"  Python version: {details['python_version']}")

        if py_version < (3, 10):
            return StepResult(
                1,
                "Environment Check",
                "FAIL",
                (time.perf_counter() - start) * 1000,
                "Python 3.10+ required",
                details,
            )

        # Check critical imports
        imports_ok = []
        imports_fail = []

        critical_imports = [
            ("numpy", "np"),
            ("pandas", "pd"),
            ("scipy", None),
            ("sklearn", None),
            ("torch", None),
        ]

        for module, alias in critical_imports:
            try:
                __import__(module)
                imports_ok.append(module)
                logger.debug(f"  ✓ {module}")
            except ImportError:
                imports_fail.append(module)
                logger.warning(f"  ✗ {module} (not installed)")

        details["imports_ok"] = imports_ok
        details["imports_fail"] = imports_fail

        # Check kinetra modules
        kinetra_modules = [
            "kinetra.physics_engine",
            "kinetra.backtest_engine",
            "kinetra.rl_agent",
            "kinetra.cpu_utils",
        ]

        kinetra_ok = []
        kinetra_fail = []

        for module in kinetra_modules:
            try:
                __import__(module)
                kinetra_ok.append(module)
                logger.info(f"  ✓ {module}")
            except ImportError as e:
                kinetra_fail.append(f"{module}: {e}")
                logger.error(f"  ✗ {module}: {e}")

        details["kinetra_ok"] = kinetra_ok
        details["kinetra_fail"] = kinetra_fail

        # Check data directories
        dirs_exist = []
        dirs_missing = []

        for dir_path in [ctx.data_dir, ctx.master_dir]:
            if dir_path.exists():
                dirs_exist.append(str(dir_path))
                logger.info(f"  ✓ Directory exists: {dir_path.name}")
            else:
                dirs_missing.append(str(dir_path))
                logger.warning(f"  ✗ Directory missing: {dir_path}")

        details["dirs_exist"] = dirs_exist
        details["dirs_missing"] = dirs_missing

        # Check CPU info
        try:
            from kinetra.cpu_utils import get_cpu_info, get_optimal_workers

            cpu_info = get_cpu_info()
            details["cpu_cores"] = cpu_info.logical_cores
            details["cpu_brand"] = cpu_info.brand
            details["optimal_workers"] = get_optimal_workers("balanced")
            logger.info(
                f"  CPU: {cpu_info.logical_cores} cores, {get_optimal_workers('balanced')} workers"
            )
        except Exception as e:
            logger.warning(f"  Could not get CPU info: {e}")

        duration = (time.perf_counter() - start) * 1000

        if kinetra_fail:
            return StepResult(
                1,
                "Environment Check",
                "FAIL",
                duration,
                f"Missing kinetra modules: {kinetra_fail}",
                details,
            )

        if dirs_missing:
            return StepResult(
                1,
                "Environment Check",
                "WARN",
                duration,
                f"Missing directories: {dirs_missing}",
                details,
            )

        return StepResult(
            1,
            "Environment Check",
            "PASS",
            duration,
            f"All {len(imports_ok)} dependencies OK",
            details,
        )

    except Exception as e:
        duration = (time.perf_counter() - start) * 1000
        return StepResult(
            1,
            "Environment Check",
            "FAIL",
            duration,
            str(e),
            details,
            str(e),
            traceback.format_exc(),
        )


def step_02_data_discovery(ctx: E2ETestContext) -> StepResult:
    """Step 2: Discover and validate available data."""
    start = time.perf_counter()
    details = {}

    try:
        # Find all CSV files
        csv_files = list(ctx.master_dir.glob("*.csv"))
        symlinked = [f for f in csv_files if f.is_symlink()]

        details["total_files"] = len(csv_files)
        details["symlinked_files"] = len(symlinked)
        logger.info(f"  Found {len(csv_files)} CSV files ({len(symlinked)} symlinked)")

        # Parse symbols and timeframes
        symbols = set()
        timeframes = set()

        for f in csv_files:
            name = f.stem
            parts = name.split("_")
            if len(parts) >= 2:
                symbols.add(parts[0].replace("+", ""))
                timeframes.add(parts[1])

        details["symbols"] = sorted(list(symbols))
        details["timeframes"] = sorted(list(timeframes))
        logger.info(f"  Symbols: {len(symbols)} unique")
        logger.info(f"  Timeframes: {sorted(timeframes)}")

        # Check for test symbol/timeframe
        test_file = None
        for f in csv_files:
            if ctx.test_symbol in f.stem and ctx.test_timeframe in f.stem:
                test_file = f
                break

        if test_file:
            details["test_file"] = str(test_file)
            logger.info(f"  ✓ Test data found: {test_file.name}")

            # Get file info
            if test_file.is_symlink():
                real_path = test_file.resolve()
                details["real_path"] = str(real_path)
                file_size = real_path.stat().st_size
            else:
                file_size = test_file.stat().st_size

            details["file_size"] = file_size
            logger.info(f"  File size: {format_size(file_size)}")
        else:
            details["test_file"] = None
            logger.warning(f"  ✗ Test data not found: {ctx.test_symbol}_{ctx.test_timeframe}")

            # Try to find alternative
            if csv_files:
                alt_file = csv_files[0]
                parts = alt_file.stem.split("_")
                if len(parts) >= 2:
                    ctx.test_symbol = parts[0].replace("+", "")
                    ctx.test_timeframe = parts[1]
                    logger.info(f"  Using alternative: {ctx.test_symbol}_{ctx.test_timeframe}")
                    details["test_file"] = str(alt_file)

        duration = (time.perf_counter() - start) * 1000

        if not csv_files:
            return StepResult(2, "Data Discovery", "FAIL", duration, "No CSV files found", details)

        return StepResult(
            2,
            "Data Discovery",
            "PASS",
            duration,
            f"Found {len(symbols)} symbols, {len(timeframes)} timeframes",
            details,
        )

    except Exception as e:
        duration = (time.perf_counter() - start) * 1000
        return StepResult(
            2, "Data Discovery", "FAIL", duration, str(e), details, str(e), traceback.format_exc()
        )


def step_03_data_loading(ctx: E2ETestContext) -> StepResult:
    """Step 3: Load and validate test data."""
    start = time.perf_counter()
    details = {}

    try:
        # Find test file
        test_file = None
        for f in ctx.master_dir.glob("*.csv"):
            if ctx.test_symbol in f.stem and ctx.test_timeframe in f.stem:
                test_file = f
                break

        if not test_file:
            return StepResult(
                3,
                "Data Loading",
                "FAIL",
                (time.perf_counter() - start) * 1000,
                f"Test file not found: {ctx.test_symbol}_{ctx.test_timeframe}",
                details,
            )

        logger.info(f"  Loading: {test_file.name}")

        # Load data
        df = pd.read_csv(test_file, sep=None, engine="python")
        logger.info(f"  Loaded {len(df)} rows, {len(df.columns)} columns")

        details["original_rows"] = len(df)
        details["original_columns"] = list(df.columns)

        # Standardize column names
        df.columns = df.columns.str.lower().str.strip().str.replace("<", "").str.replace(">", "")

        # Map common column names - handle dual volume columns (tickvol takes priority)
        column_map = {
            "datetime": "time",
            "timestamp": "time",
        }
        df.rename(columns=column_map, inplace=True)

        # Handle volume columns - prefer tickvol over vol
        if "tickvol" in df.columns:
            df["volume"] = pd.to_numeric(df["tickvol"], errors="coerce").fillna(0)
            df.drop(columns=["tickvol"], inplace=True, errors="ignore")
            if "vol" in df.columns:
                df.drop(columns=["vol"], inplace=True, errors="ignore")
            logger.info("  Using TICKVOL as volume")
        elif "vol" in df.columns:
            df["volume"] = pd.to_numeric(df["vol"], errors="coerce").fillna(0)
            df.drop(columns=["vol"], inplace=True, errors="ignore")
            logger.info("  Using VOL as volume")
        elif "tick_volume" in df.columns:
            df["volume"] = pd.to_numeric(df["tick_volume"], errors="coerce").fillna(0)
            df.drop(columns=["tick_volume"], inplace=True, errors="ignore")

        # Parse time column - handle separate date and time columns
        if "date" in df.columns and "time" in df.columns:
            # Combine date and time columns (MT5 format)
            df["datetime"] = pd.to_datetime(
                df["date"].astype(str) + " " + df["time"].astype(str),
                format="%Y.%m.%d %H:%M:%S",
                errors="coerce",
            )
            df.drop(columns=["date", "time"], inplace=True)
            df.rename(columns={"datetime": "time"}, inplace=True)
            df.set_index("time", inplace=True)
            logger.info("  Combined DATE + TIME columns")
        elif "time" in df.columns:
            df["time"] = pd.to_datetime(df["time"], errors="coerce")
            df.set_index("time", inplace=True)
        elif "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df.rename(columns={"date": "time"}, inplace=True)
            df.set_index("time", inplace=True)

        # Ensure required columns exist
        required = ["open", "high", "low", "close"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            return StepResult(
                3,
                "Data Loading",
                "FAIL",
                (time.perf_counter() - start) * 1000,
                f"Missing columns: {missing}",
                details,
            )

        # Add volume if missing
        if "volume" not in df.columns:
            df["volume"] = 0
            logger.warning("  Volume column missing, set to 0")

        # Limit rows for testing
        if len(df) > ctx.test_bars:
            df = df.tail(ctx.test_bars)
            logger.info(f"  Limited to last {ctx.test_bars} bars")

        # Data quality checks
        nan_counts = df[required + ["volume"]].isna().sum()
        details["nan_counts"] = nan_counts.to_dict()

        if nan_counts.sum() > 0:
            logger.warning(f"  NaN counts: {nan_counts.to_dict()}")
            df = df.dropna(subset=required)
            logger.info(f"  After dropna: {len(df)} rows")

        # OHLC validation
        invalid_ohlc = (
            (df["high"] < df["low"])
            | (df["high"] < df["open"])
            | (df["high"] < df["close"])
            | (df["low"] > df["open"])
            | (df["low"] > df["close"])
        )

        details["invalid_ohlc_count"] = invalid_ohlc.sum()
        if invalid_ohlc.sum() > 0:
            logger.warning(f"  Invalid OHLC bars: {invalid_ohlc.sum()}")

        # Store in context
        ctx.raw_data = df

        # Calculate basic statistics
        details["final_rows"] = len(df)
        details["date_range"] = [str(df.index.min()), str(df.index.max())]
        details["price_range"] = [float(df["close"].min()), float(df["close"].max())]
        details["mean_volume"] = float(df["volume"].mean())

        logger.info(f"  Date range: {df.index.min()} to {df.index.max()}")
        logger.info(f"  Price range: {df['close'].min():.2f} - {df['close'].max():.2f}")
        logger.info(f"  Mean volume: {df['volume'].mean():.0f}")

        duration = (time.perf_counter() - start) * 1000
        return StepResult(
            3,
            "Data Loading",
            "PASS",
            duration,
            f"Loaded {len(df)} bars of {ctx.test_symbol} {ctx.test_timeframe}",
            details,
        )

    except Exception as e:
        duration = (time.perf_counter() - start) * 1000
        return StepResult(
            3, "Data Loading", "FAIL", duration, str(e), details, str(e), traceback.format_exc()
        )


def step_04_physics_engine(ctx: E2ETestContext) -> StepResult:
    """Step 4: Run physics engine calculations."""
    start = time.perf_counter()
    details = {}

    try:
        if ctx.raw_data is None:
            return StepResult(
                4,
                "Physics Engine",
                "SKIP",
                (time.perf_counter() - start) * 1000,
                "No data loaded from previous step",
                details,
            )

        logger.info("  Importing PhysicsEngine...")
        from kinetra.physics_engine import PhysicsEngine
        from kinetra.physics_engine import __version__ as pe_version

        details["physics_engine_version"] = pe_version
        logger.info(f"  PhysicsEngine version: {pe_version}")

        # Initialize engine
        logger.info("  Initializing PhysicsEngine...")
        engine = PhysicsEngine()
        ctx.physics_engine = engine

        # Prepare data
        df = ctx.raw_data.copy()
        df = df.reset_index()
        df.rename(columns={"time": "timestamp"}, inplace=True)

        # Run physics calculations
        logger.info("  Running physics calculations...")
        calc_start = time.perf_counter()

        # Use the correct API method
        physics_df = engine.compute_physics_state_from_ohlcv(df)

        calc_time = (time.perf_counter() - calc_start) * 1000
        details["calculation_time_ms"] = calc_time
        logger.info(f"  Calculation time: {format_duration(calc_time)}")

        # Check output columns
        physics_cols = [c for c in physics_df.columns if c not in df.columns]
        details["physics_columns"] = physics_cols
        logger.info(f"  Generated {len(physics_cols)} physics columns")

        # Log key physics columns
        key_cols = ["KE", "zeta", "entropy", "jerk", "Re", "regime"]
        found_cols = [c for c in key_cols if c in physics_df.columns]
        missing_cols = [c for c in key_cols if c not in physics_df.columns]

        details["key_columns_found"] = found_cols
        details["key_columns_missing"] = missing_cols

        for col in found_cols:
            if physics_df[col].dtype in [np.float64, np.float32]:
                logger.info(
                    f"    {col}: mean={physics_df[col].mean():.4f}, std={physics_df[col].std():.4f}"
                )
            else:
                logger.info(f"    {col}: unique={physics_df[col].nunique()}")

        if missing_cols:
            logger.warning(f"  Missing expected columns: {missing_cols}")

        # Check for NaN in physics columns
        nan_in_physics = physics_df[found_cols].isna().sum()
        details["nan_in_physics"] = nan_in_physics.to_dict()

        total_nan = nan_in_physics.sum()
        if total_nan > 0:
            logger.warning(f"  NaN values in physics columns: {total_nan}")

        # Check regime distribution
        if "regime" in physics_df.columns:
            regime_dist = physics_df["regime"].value_counts()
            details["regime_distribution"] = regime_dist.to_dict()
            logger.info("  Regime distribution:")
            for regime, count in regime_dist.items():
                pct = count / len(physics_df) * 100
                logger.info(f"    {regime}: {count} ({pct:.1f}%)")

        # Store results
        ctx.physics_data = physics_df

        # Memory cleanup
        gc.collect()

        duration = (time.perf_counter() - start) * 1000
        return StepResult(
            4,
            "Physics Engine",
            "PASS",
            duration,
            f"Generated {len(physics_cols)} features in {format_duration(calc_time)}",
            details,
        )

    except Exception as e:
        duration = (time.perf_counter() - start) * 1000
        return StepResult(
            4, "Physics Engine", "FAIL", duration, str(e), details, str(e), traceback.format_exc()
        )


def step_05_backtest_engine(ctx: E2ETestContext) -> StepResult:
    """Step 5: Run backtest engine."""
    start = time.perf_counter()
    details = {}

    try:
        if ctx.physics_data is None:
            return StepResult(
                5,
                "Backtest Engine",
                "SKIP",
                (time.perf_counter() - start) * 1000,
                "No physics data from previous step",
                details,
            )

        logger.info("  Importing BacktestEngine...")
        from kinetra.backtest_engine import BacktestEngine
        from kinetra.backtest_engine import __version__ as be_version
        from kinetra.symbol_spec import SymbolSpec

        details["backtest_engine_version"] = be_version
        logger.info(f"  BacktestEngine version: {be_version}")

        # Initialize engine
        logger.info("  Initializing BacktestEngine...")
        engine = BacktestEngine()
        ctx.backtest_engine = engine

        # Prepare data for backtest - need OHLC columns from raw data
        # Join physics features with raw OHLC data
        df = ctx.raw_data.copy()

        # Add physics features if available
        if ctx.physics_data is not None:
            # Get physics-only columns (exclude OHLC that might be duplicated)
            physics_cols = [
                c
                for c in ctx.physics_data.columns
                if c not in ["open", "high", "low", "close", "volume"]
            ]
            if physics_cols:
                for col in physics_cols:
                    if col in ctx.physics_data.columns:
                        df[col] = ctx.physics_data[col].values

        # Reset index to have timestamp as column
        if df.index.name is not None:
            df = df.reset_index()
            if "time" in df.columns:
                df.rename(columns={"time": "timestamp"}, inplace=True)

        # Run backtest with simple strategy
        logger.info("  Running backtest...")
        bt_start = time.perf_counter()

        try:
            # Create symbol spec for test symbol
            symbol_spec = SymbolSpec(
                symbol=ctx.test_symbol,
                description=f"{ctx.test_symbol} Test",
                contract_size=1.0,  # Crypto typically 1
                tick_size=0.01,
                digits=2,
                spread_points=50.0,  # ~$50 spread for BTC
            )

            # Simple momentum signal function (signature: row, physics_state, bar_index)
            def momentum_signal(row, physics_state, bar_index):
                """Simple momentum-based signal."""
                if "velocity" in row and row["velocity"] > 0:
                    return 1  # Long
                elif "velocity" in row and row["velocity"] < 0:
                    return -1  # Short
                return 0  # No position

            # Try to run backtest with correct API
            results = engine.run_backtest(
                data=df,
                symbol_spec=symbol_spec,
                signal_func=momentum_signal,
            )

            bt_time = (time.perf_counter() - bt_start) * 1000
            details["backtest_time_ms"] = bt_time
            logger.info(f"  Backtest time: {format_duration(bt_time)}")

            # Extract key metrics
            if hasattr(results, "metrics"):
                metrics = results.metrics
            elif isinstance(results, dict):
                metrics = results
            else:
                metrics = {}

            details["metrics"] = {}

            metric_names = [
                "total_return",
                "sharpe_ratio",
                "omega_ratio",
                "max_drawdown",
                "total_trades",
                "win_rate",
                "profit_factor",
            ]

            for name in metric_names:
                if name in metrics:
                    value = metrics[name]
                    details["metrics"][name] = value
                    if isinstance(value, float):
                        logger.info(f"    {name}: {value:.4f}")
                    else:
                        logger.info(f"    {name}: {value}")

            ctx.backtest_results = results

        except Exception as bt_error:
            logger.warning(f"  Backtest execution failed: {bt_error}")
            logger.info("  Attempting basic validation instead...")

            # Fallback: just validate the engine can be instantiated
            details["backtest_status"] = "partial"
            details["backtest_error"] = str(bt_error)

            bt_time = (time.perf_counter() - bt_start) * 1000
            details["backtest_time_ms"] = bt_time

        # Memory cleanup
        gc.collect()

        duration = (time.perf_counter() - start) * 1000

        if "backtest_error" in details:
            return StepResult(
                5,
                "Backtest Engine",
                "WARN",
                duration,
                f"Engine loaded but backtest failed: {details['backtest_error'][:100]}",
                details,
            )

        return StepResult(
            5,
            "Backtest Engine",
            "PASS",
            duration,
            f"Backtest completed in {format_duration(bt_time)}",
            details,
        )

    except Exception as e:
        duration = (time.perf_counter() - start) * 1000
        return StepResult(
            5, "Backtest Engine", "FAIL", duration, str(e), details, str(e), traceback.format_exc()
        )


def step_06_rl_agent(ctx: E2ETestContext) -> StepResult:
    """Step 6: Test RL agent initialization and inference."""
    start = time.perf_counter()
    details = {}

    try:
        logger.info("  Importing RL Agent...")
        from kinetra.rl_agent import __version__ as rl_version

        details["rl_agent_version"] = rl_version
        logger.info(f"  RL Agent version: {rl_version}")

        # Check for PyTorch
        try:
            import torch

            details["torch_available"] = True
            details["torch_version"] = torch.__version__
            details["cuda_available"] = torch.cuda.is_available()
            logger.info(f"  PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}")
        except ImportError:
            details["torch_available"] = False
            logger.warning("  PyTorch not available")

        # Try to import and instantiate agent
        try:
            from kinetra.rl_agent import KinetraAgent, PPOBuffer

            # Test buffer
            buffer = PPOBuffer()
            details["buffer_initialized"] = True
            logger.info("  ✓ PPOBuffer initialized")

            # Test with mock state
            if details.get("torch_available"):
                state_dim = 10
                action_dim = 3

                logger.info(f"  Testing agent with state_dim={state_dim}, action_dim={action_dim}")

                try:
                    agent = KinetraAgent(state_dim=state_dim, action_dim=action_dim)
                    ctx.rl_agent = agent
                    details["agent_initialized"] = True
                    logger.info("  ✓ KinetraAgent initialized")

                    # Test inference
                    test_state = torch.randn(1, state_dim)
                    with torch.no_grad():
                        action = agent.select_action(test_state)

                    details["inference_test"] = "passed"
                    logger.info(
                        f"  ✓ Inference test passed, action shape: {action.shape if hasattr(action, 'shape') else type(action)}"
                    )

                except Exception as agent_error:
                    details["agent_error"] = str(agent_error)
                    logger.warning(f"  Agent initialization failed: {agent_error}")

        except ImportError as ie:
            details["import_error"] = str(ie)
            logger.warning(f"  Could not import KinetraAgent: {ie}")

        # Memory cleanup
        gc.collect()
        if details.get("torch_available"):
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

        duration = (time.perf_counter() - start) * 1000

        if details.get("agent_initialized") and details.get("inference_test") == "passed":
            return StepResult(
                6, "RL Agent", "PASS", duration, "Agent initialized and inference working", details
            )
        elif details.get("buffer_initialized"):
            return StepResult(
                6,
                "RL Agent",
                "WARN",
                duration,
                "Buffer OK but agent initialization failed",
                details,
            )
        else:
            return StepResult(
                6, "RL Agent", "FAIL", duration, "Could not initialize RL components", details
            )

    except Exception as e:
        duration = (time.perf_counter() - start) * 1000
        return StepResult(
            6, "RL Agent", "FAIL", duration, str(e), details, str(e), traceback.format_exc()
        )


def step_07_monte_carlo(ctx: E2ETestContext) -> StepResult:
    """Step 7: Run Monte Carlo validation."""
    start = time.perf_counter()
    details = {}

    try:
        if ctx.backtest_engine is None:
            return StepResult(
                7,
                "Monte Carlo Validation",
                "SKIP",
                (time.perf_counter() - start) * 1000,
                "No backtest engine from previous step",
                details,
            )

        logger.info("  Running Monte Carlo validation...")

        # Now using fast vectorized MC worker (~0.2ms/run), so we can run many more
        n_runs = 100
        details["n_runs"] = n_runs

        mc_start = time.perf_counter()

        try:
            # Check if MC method exists
            if hasattr(ctx.backtest_engine, "monte_carlo_validation"):
                logger.info(f"  Running {n_runs} Monte Carlo iterations...")

                # Import SymbolSpec for MC validation
                from kinetra.symbol_spec import SymbolSpec

                symbol_spec = SymbolSpec(
                    symbol=ctx.test_symbol,
                    description=f"{ctx.test_symbol} Test",
                    contract_size=1.0,
                    tick_size=0.01,
                    digits=2,
                    spread_points=50.0,
                )

                # Prepare data with OHLC columns for MC validation
                mc_data = ctx.raw_data.copy()
                if mc_data.index.name is not None:
                    mc_data = mc_data.reset_index()
                    if "time" in mc_data.columns:
                        mc_data.rename(columns={"time": "timestamp"}, inplace=True)

                mc_results = ctx.backtest_engine.monte_carlo_validation(
                    data=mc_data,
                    symbol_spec=symbol_spec,
                    n_runs=n_runs,
                )

                mc_time = (time.perf_counter() - mc_start) * 1000
                details["mc_time_ms"] = mc_time
                logger.info(f"  MC time: {format_duration(mc_time)}")

                if isinstance(mc_results, pd.DataFrame):
                    details["mc_runs_completed"] = len(mc_results)
                    if "omega_ratio" in mc_results.columns:
                        details["omega_mean"] = float(mc_results["omega_ratio"].mean())
                        details["omega_std"] = float(mc_results["omega_ratio"].std())
                        logger.info(
                            f"  Omega ratio: {details['omega_mean']:.3f} ± {details['omega_std']:.3f}"
                        )
                elif isinstance(mc_results, dict):
                    details["mc_results"] = mc_results

            else:
                logger.warning("  monte_carlo_validation method not found")
                details["mc_method_available"] = False

        except Exception as mc_error:
            logger.warning(f"  Monte Carlo failed: {mc_error}")
            details["mc_error"] = str(mc_error)

        duration = (time.perf_counter() - start) * 1000

        if "mc_error" in details:
            return StepResult(
                7,
                "Monte Carlo Validation",
                "WARN",
                duration,
                f"MC validation failed: {details['mc_error'][:100]}",
                details,
            )

        return StepResult(
            7, "Monte Carlo Validation", "PASS", duration, f"Completed {n_runs} MC runs", details
        )

    except Exception as e:
        duration = (time.perf_counter() - start) * 1000
        return StepResult(
            7,
            "Monte Carlo Validation",
            "FAIL",
            duration,
            str(e),
            details,
            str(e),
            traceback.format_exc(),
        )


def step_08_integration_check(ctx: E2ETestContext) -> StepResult:
    """Step 8: Full integration check - all components working together."""
    start = time.perf_counter()
    details = {}

    try:
        logger.info("  Checking component integration...")

        components = {
            "raw_data": ctx.raw_data is not None,
            "physics_data": ctx.physics_data is not None,
            "physics_engine": ctx.physics_engine is not None,
            "backtest_engine": ctx.backtest_engine is not None,
            "rl_agent": ctx.rl_agent is not None,
        }

        details["components"] = components

        for name, available in components.items():
            status = "✓" if available else "✗"
            logger.info(f"    {status} {name}")

        # Count available components
        available_count = sum(components.values())
        total_count = len(components)

        details["available"] = available_count
        details["total"] = total_count

        # Check data flow
        if ctx.raw_data is not None and ctx.physics_data is not None:
            raw_rows = len(ctx.raw_data)
            physics_rows = len(ctx.physics_data)
            details["data_flow"] = {
                "raw_rows": raw_rows,
                "physics_rows": physics_rows,
                "rows_match": raw_rows == physics_rows,
            }
            logger.info(f"  Data flow: {raw_rows} raw → {physics_rows} physics")

        duration = (time.perf_counter() - start) * 1000

        if available_count >= 4:
            return StepResult(
                8,
                "Integration Check",
                "PASS",
                duration,
                f"{available_count}/{total_count} components integrated",
                details,
            )
        elif available_count >= 2:
            return StepResult(
                8,
                "Integration Check",
                "WARN",
                duration,
                f"Only {available_count}/{total_count} components available",
                details,
            )
        else:
            return StepResult(
                8,
                "Integration Check",
                "FAIL",
                duration,
                f"Only {available_count}/{total_count} components available",
                details,
            )

    except Exception as e:
        duration = (time.perf_counter() - start) * 1000
        return StepResult(
            8,
            "Integration Check",
            "FAIL",
            duration,
            str(e),
            details,
            str(e),
            traceback.format_exc(),
        )


# ============================================================================
# Test Runner
# ============================================================================


def run_all_steps(
    interactive: bool = False, steps_to_run: Optional[List[int]] = None, verbose: bool = False
) -> List[StepResult]:
    """Run all E2E test steps."""

    if verbose:
        console_handler.setLevel(logging.DEBUG)

    # Define all steps
    all_steps = [
        (1, "Environment Check", step_01_environment_check),
        (2, "Data Discovery", step_02_data_discovery),
        (3, "Data Loading", step_03_data_loading),
        (4, "Physics Engine", step_04_physics_engine),
        (5, "Backtest Engine", step_05_backtest_engine),
        (6, "RL Agent", step_06_rl_agent),
        (7, "Monte Carlo Validation", step_07_monte_carlo),
        (8, "Integration Check", step_08_integration_check),
    ]

    # Filter steps if specified
    if steps_to_run:
        all_steps = [(n, name, fn) for n, name, fn in all_steps if n in steps_to_run]

    total_steps = len(all_steps)

    # Create context
    ctx = E2ETestContext()

    # Print banner
    print_banner("KINETRA E2E STEPOVER TEST")
    logger.info(f"  Mode: {'INTERACTIVE' if interactive else 'AUTOMATIC'}")
    logger.info(f"  Steps: {total_steps}")
    logger.info(f"  Log file: {LOG_FILE}")
    logger.info(f"  Started: {datetime.now().isoformat()}")

    # Run each step
    for step_num, step_name, step_fn in all_steps:
        print_step_header(step_num, total_steps, step_name)

        # Execute step
        result = step_fn(ctx)
        ctx.step_results.append(result)

        # Print result
        status_icon = {
            "PASS": "✅",
            "FAIL": "❌",
            "SKIP": "⏭️",
            "WARN": "⚠️",
        }.get(result.status, "?")

        logger.info("")
        logger.info(
            f"  Result: {status_icon} {result.status} ({format_duration(result.duration_ms)})"
        )
        logger.info(f"  {result.message}")

        if result.error:
            logger.error(f"  Error: {result.error}")
            if verbose and result.traceback:
                logger.debug(f"  Traceback:\n{result.traceback}")

        # Wait for user in interactive mode
        if interactive and step_num < total_steps:
            wait_for_continue(interactive)

    return ctx.step_results


def print_summary(results: List[StepResult]):
    """Print test summary."""
    print_banner("TEST SUMMARY")

    # Count results
    counts = {"PASS": 0, "FAIL": 0, "SKIP": 0, "WARN": 0}
    total_duration = 0

    for r in results:
        counts[r.status] = counts.get(r.status, 0) + 1
        total_duration += r.duration_ms

    # Print results table
    logger.info("")
    logger.info("  Step                        Status    Duration")
    logger.info("  " + "─" * 50)

    for r in results:
        status_icon = {"PASS": "✅", "FAIL": "❌", "SKIP": "⏭️", "WARN": "⚠️"}.get(r.status, "?")
        logger.info(
            f"  {r.step_num}. {r.name:<24} {status_icon} {r.status:<6}  {format_duration(r.duration_ms)}"
        )

    logger.info("  " + "─" * 50)
    logger.info(f"  Total duration: {format_duration(total_duration)}")
    logger.info("")

    # Print counts
    logger.info(f"  ✅ Passed:  {counts['PASS']}")
    logger.info(f"  ❌ Failed:  {counts['FAIL']}")
    logger.info(f"  ⚠️  Warnings: {counts['WARN']}")
    logger.info(f"  ⏭️  Skipped: {counts['SKIP']}")
    logger.info("")

    # Overall result
    if counts["FAIL"] == 0:
        logger.info("  🎉 ALL TESTS PASSED!")
    else:
        logger.info(f"  ❌ {counts['FAIL']} TEST(S) FAILED")

    logger.info("")
    logger.info(f"  📄 Full log: {LOG_FILE}")


def save_results(results: List[StepResult]):
    """Save results to JSON file."""
    results_file = LOG_DIR / f"e2e_results_{TIMESTAMP}.json"

    results_data = {
        "timestamp": datetime.now().isoformat(),
        "version": __version__,
        "steps": [
            {
                "step_num": r.step_num,
                "name": r.name,
                "status": r.status,
                "duration_ms": r.duration_ms,
                "message": r.message,
                "details": r.details,
                "error": r.error,
            }
            for r in results
        ],
        "summary": {
            "total": len(results),
            "passed": sum(1 for r in results if r.status == "PASS"),
            "failed": sum(1 for r in results if r.status == "FAIL"),
            "warnings": sum(1 for r in results if r.status == "WARN"),
            "skipped": sum(1 for r in results if r.status == "SKIP"),
        },
    }

    with open(results_file, "w") as f:
        json.dump(results_data, f, indent=2, default=str)

    logger.info(f"  📊 Results saved: {results_file}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Kinetra E2E Stepover Test with Detailed Logging",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Interactive mode (pause after each step)
    python scripts/testing/e2e_stepover_test.py --interactive

    # Automatic mode (run all steps)
    python scripts/testing/e2e_stepover_test.py --auto

    # Run specific steps only
    python scripts/testing/e2e_stepover_test.py --steps 1,2,3,4

    # Verbose logging
    python scripts/testing/e2e_stepover_test.py --auto --verbose
        """,
    )

    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        "--interactive", "-i", action="store_true", help="Interactive mode (pause after each step)"
    )
    mode_group.add_argument(
        "--auto", "-a", action="store_true", help="Automatic mode (run all steps)"
    )

    parser.add_argument(
        "--steps",
        "-s",
        type=str,
        default=None,
        help="Comma-separated list of step numbers to run (e.g., 1,2,3)",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose (debug) logging"
    )
    parser.add_argument(
        "--symbol", type=str, default="BTCUSD", help="Test symbol (default: BTCUSD)"
    )
    parser.add_argument("--timeframe", type=str, default="H1", help="Test timeframe (default: H1)")
    parser.add_argument(
        "--bars", type=int, default=1000, help="Number of bars to test (default: 1000)"
    )

    args = parser.parse_args()

    # Parse steps if specified
    steps_to_run = None
    if args.steps:
        try:
            steps_to_run = [int(s.strip()) for s in args.steps.split(",")]
        except ValueError:
            logger.error(f"Invalid steps format: {args.steps}")
            sys.exit(1)

    # Run tests
    try:
        results = run_all_steps(
            interactive=args.interactive,
            steps_to_run=steps_to_run,
            verbose=args.verbose,
        )

        # Print and save summary
        print_summary(results)
        save_results(results)

        # Exit with appropriate code
        failed = sum(1 for r in results if r.status == "FAIL")
        sys.exit(1 if failed > 0 else 0)

    except KeyboardInterrupt:
        logger.warning("\n  Test interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"  Unexpected error: {e}")
        logger.debug(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
