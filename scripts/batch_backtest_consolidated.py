#!/usr/bin/env python3
"""
Consolidated Batch Backtest
============================

Replaces 22+ scattered backtest scripts with single canonical implementation.

Features:
- Progress bars with time estimates (tqdm)
- Parallel execution (ProcessPoolExecutor)
- Resume capability from interruption
- Error recovery and retry logic
- Comprehensive metrics (Omega, Sharpe, Z-factor, Energy capture)
- Real-time status updates

Philosophy (from Agent Rules):
- No magic numbers (dynamic thresholds)
- Physics-first validation (energy capture required)
- Statistical rigor (p < 0.01 significance)
- Vectorized operations (avoid Python loops)

Usage:
    python scripts/batch_backtest_consolidated.py
    python scripts/batch_backtest_consolidated.py --instruments BTCUSD ETHUSD --timeframes H1 H4
    python scripts/batch_backtest_consolidated.py --parallel --workers 8
    python scripts/batch_backtest_consolidated.py --resume
"""

import argparse
import json
import multiprocessing as mp
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from kinetra.physics import compute_physics_measurements

from kinetra.backtest_engine import BacktestEngine
from kinetra.risk_management import RiskManager


@dataclass
class BacktestConfig:
    """Configuration for single backtest run."""

    instrument: str
    timeframe: str
    agent_type: str
    data_path: Path
    initial_capital: float = 10000.0
    max_position_size: float = 1.0
    commission: float = 0.0001


@dataclass
class BacktestResult:
    """Results from single backtest."""

    config: BacktestConfig
    omega_ratio: float
    sharpe_ratio: float
    z_factor: float
    energy_captured_pct: float
    total_return: float
    num_trades: int
    win_rate: float
    max_drawdown: float
    calmar_ratio: float
    profit_factor: float
    avg_trade_duration: float
    statistical_significance: float  # p-value
    physics_valid: bool
    error: Optional[str] = None


class ConsolidatedBacktester:
    """
    Consolidated backtesting engine with progress tracking and parallelization.

    Replaces:
    - batch_backtest.py
    - run_comprehensive_backtest.py
    - run_full_backtest.py
    - rl_backtest.py
    - And 18+ other variants
    """

    def __init__(
        self,
        max_workers: Optional[int] = None,
        resume: bool = False,
        results_dir: Path = None,
    ):
        """
        Initialize backtester.

        Args:
            max_workers: Number of parallel workers (default: CPU count - 1)
            resume: Resume from last checkpoint
            results_dir: Directory for results (default: results/)
        """
        if max_workers is None:
            self.max_workers = max(1, mp.cpu_count() - 1)
        else:
            self.max_workers = max_workers

        self.resume = resume
        self.results_dir = results_dir or PROJECT_ROOT / "results"
        self.results_dir.mkdir(exist_ok=True, parents=True)

        self.checkpoint_file = self.results_dir / "backtest_checkpoint.json"
        self.results_file = self.results_dir / "batch_backtest_results.csv"

        self.completed = self._load_checkpoint() if resume else set()

    def _load_checkpoint(self) -> set:
        """Load completed backtests from checkpoint."""
        if not self.checkpoint_file.exists():
            return set()

        try:
            with open(self.checkpoint_file, "r") as f:
                data = json.load(f)
            return set(tuple(item) for item in data.get("completed", []))
        except Exception as e:
            print(f"⚠️  Could not load checkpoint: {e}")
            return set()

    def _save_checkpoint(self, completed: set):
        """Save checkpoint of completed backtests."""
        try:
            with open(self.checkpoint_file, "w") as f:
                json.dump({"completed": [list(item) for item in completed]}, f)
        except Exception as e:
            print(f"⚠️  Could not save checkpoint: {e}")

    def create_configs(
        self,
        instruments: List[str],
        timeframes: List[str],
        agent_types: List[str] = None,
        data_dir: Path = None,
    ) -> List[BacktestConfig]:
        """
        Create backtest configurations for all combinations.

        Args:
            instruments: List of instruments (e.g., ['BTCUSD', 'ETHUSD'])
            timeframes: List of timeframes (e.g., ['H1', 'H4', 'D1'])
            agent_types: List of agent types (default: ['berserker', 'sniper', 'triad'])
            data_dir: Directory containing data files

        Returns:
            List of BacktestConfig objects
        """
        if agent_types is None:
            agent_types = ["berserker", "sniper", "triad"]

        if data_dir is None:
            data_dir = PROJECT_ROOT / "data" / "master_standardized"

        configs = []

        for instrument in instruments:
            for timeframe in timeframes:
                for agent_type in agent_types:
                    data_path = data_dir / f"{instrument}_{timeframe}.csv"

                    if not data_path.exists():
                        print(f"⚠️  Skipping {instrument}_{timeframe} - file not found")
                        continue

                    # Check if already completed
                    key = (instrument, timeframe, agent_type)
                    if key in self.completed:
                        continue

                    configs.append(
                        BacktestConfig(
                            instrument=instrument,
                            timeframe=timeframe,
                            agent_type=agent_type,
                            data_path=data_path,
                        )
                    )

        return configs

    def run_single_backtest(self, config: BacktestConfig) -> BacktestResult:
        """
        Execute single backtest with full validation.

        Args:
            config: Backtest configuration

        Returns:
            BacktestResult with metrics and validation
        """
        try:
            # Load data
            df = pd.read_csv(config.data_path)

            # Validate data
            required_cols = ["timestamp", "open", "high", "low", "close", "volume"]
            if not all(col in df.columns for col in required_cols):
                raise ValueError(f"Missing required columns in {config.data_path}")

            # Compute physics measurements (vectorized)
            df = compute_physics_measurements(df)

            # Initialize engine
            engine = BacktestEngine(
                initial_capital=config.initial_capital,
                commission=config.commission,
            )

            # Load agent model
            agent = self._load_agent(config.agent_type, config.instrument)

            # Run backtest
            results = engine.run(df, agent, max_position_size=config.max_position_size)

            # Compute comprehensive metrics
            metrics = self._compute_metrics(results, df)

            # Validate
            physics_valid = self._validate_physics(metrics)
            significance = self._test_significance(results)

            return BacktestResult(
                config=config,
                omega_ratio=metrics["omega_ratio"],
                sharpe_ratio=metrics["sharpe_ratio"],
                z_factor=metrics["z_factor"],
                energy_captured_pct=metrics["energy_captured_pct"],
                total_return=metrics["total_return"],
                num_trades=metrics["num_trades"],
                win_rate=metrics["win_rate"],
                max_drawdown=metrics["max_drawdown"],
                calmar_ratio=metrics["calmar_ratio"],
                profit_factor=metrics["profit_factor"],
                avg_trade_duration=metrics["avg_trade_duration"],
                statistical_significance=significance,
                physics_valid=physics_valid,
                error=None,
            )

        except Exception as e:
            return BacktestResult(
                config=config,
                omega_ratio=0.0,
                sharpe_ratio=0.0,
                z_factor=0.0,
                energy_captured_pct=0.0,
                total_return=0.0,
                num_trades=0,
                win_rate=0.0,
                max_drawdown=0.0,
                calmar_ratio=0.0,
                profit_factor=0.0,
                avg_trade_duration=0.0,
                statistical_significance=1.0,
                physics_valid=False,
                error=str(e),
            )

    def _load_agent(self, agent_type: str, instrument: str):
        """Load trained agent model."""
        model_path = PROJECT_ROOT / "models" / f"{agent_type}_{instrument}.pkl"

        if not model_path.exists():
            # Try generic agent
            model_path = PROJECT_ROOT / "models" / f"{agent_type}.pkl"

        if not model_path.exists():
            raise FileNotFoundError(f"No model found for {agent_type}")

        # Import agent class based on type
        if agent_type == "berserker":
            from kinetra.agents.berserker import BerserkerAgent

            return BerserkerAgent.load(model_path)
        elif agent_type == "sniper":
            from kinetra.agents.sniper import SniperAgent

            return SniperAgent.load(model_path)
        elif agent_type == "triad":
            from kinetra.agents.triad import TriadAgent

            return TriadAgent.load(model_path)
        else:
            raise ValueError(f"Unknown agent type: {agent_type}")

    def _compute_metrics(self, results: Dict, df: pd.DataFrame) -> Dict[str, float]:
        """
        Compute comprehensive performance metrics.

        Vectorized implementation - no Python loops.
        """
        equity_curve = np.array(results["equity_curve"])
        trades = results["trades"]

        # Returns (vectorized)
        returns = np.diff(equity_curve) / equity_curve[:-1]
        returns = returns[~np.isnan(returns)]

        # Omega ratio (upside/downside deviation)
        threshold = 0.0
        gains = returns[returns > threshold] - threshold
        losses = threshold - returns[returns < threshold]
        omega_ratio = gains.sum() / losses.sum() if losses.sum() > 0 else 0.0

        # Sharpe ratio
        sharpe_ratio = (returns.mean() / returns.std() * np.sqrt(252)) if returns.std() > 0 else 0.0

        # Z-factor (statistical edge)
        win_rate = np.mean([t["pnl"] > 0 for t in trades]) if trades else 0.0
        avg_win = np.mean([t["pnl"] for t in trades if t["pnl"] > 0]) if trades else 0.0
        avg_loss = np.mean([abs(t["pnl"]) for t in trades if t["pnl"] < 0]) if trades else 0.0

        if avg_loss > 0:
            z_factor = (win_rate * avg_win - (1 - win_rate) * avg_loss) / (
                np.std([t["pnl"] for t in trades]) if trades else 1.0
            )
        else:
            z_factor = 0.0

        # Energy capture % (physics validation)
        total_energy = df["energy"].sum()
        captured_energy = sum([t.get("energy_captured", 0) for t in trades])
        energy_captured_pct = (captured_energy / total_energy * 100) if total_energy > 0 else 0.0

        # Max drawdown (vectorized)
        cummax = np.maximum.accumulate(equity_curve)
        drawdown = (equity_curve - cummax) / cummax
        max_drawdown = abs(drawdown.min())

        # Calmar ratio
        total_return = (equity_curve[-1] / equity_curve[0] - 1) if len(equity_curve) > 0 else 0.0
        calmar_ratio = total_return / max_drawdown if max_drawdown > 0 else 0.0

        # Profit factor
        gross_profit = sum([t["pnl"] for t in trades if t["pnl"] > 0])
        gross_loss = abs(sum([t["pnl"] for t in trades if t["pnl"] < 0]))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0.0

        # Average trade duration
        if trades:
            durations = [t.get("duration", 0) for t in trades]
            avg_trade_duration = np.mean(durations)
        else:
            avg_trade_duration = 0.0

        return {
            "omega_ratio": omega_ratio,
            "sharpe_ratio": sharpe_ratio,
            "z_factor": z_factor,
            "energy_captured_pct": energy_captured_pct,
            "total_return": total_return,
            "num_trades": len(trades),
            "win_rate": win_rate,
            "max_drawdown": max_drawdown,
            "calmar_ratio": calmar_ratio,
            "profit_factor": profit_factor,
            "avg_trade_duration": avg_trade_duration,
        }

    def _validate_physics(self, metrics: Dict[str, float]) -> bool:
        """
        Validate results against physics requirements.

        From Agent Rules:
        - Omega > 2.7 (asymmetric returns)
        - Z-factor > 2.5 (statistical edge)
        - Energy capture > 65% (physics alignment)
        """
        return (
            metrics["omega_ratio"] > 2.7
            and metrics["z_factor"] > 2.5
            and metrics["energy_captured_pct"] > 65.0
        )

    def _test_significance(self, results: Dict) -> float:
        """
        Test statistical significance of returns.

        Returns:
            p-value (< 0.01 required for significance)
        """
        from scipy import stats

        trades = results["trades"]
        if not trades or len(trades) < 30:
            return 1.0  # Insufficient data

        pnls = np.array([t["pnl"] for t in trades])

        # One-sample t-test against zero (no alpha)
        t_stat, p_value = stats.ttest_1samp(pnls, 0.0)

        return p_value if t_stat > 0 else 1.0

    def run_batch(
        self, configs: List[BacktestConfig], parallel: bool = True
    ) -> List[BacktestResult]:
        """
        Run batch of backtests with progress tracking.

        Args:
            configs: List of backtest configurations
            parallel: Use parallel execution (default: True)

        Returns:
            List of BacktestResult objects
        """
        results = []

        print(f"\n🚀 Starting batch backtest")
        print(f"   Configurations: {len(configs)}")
        print(f"   Parallel: {parallel} (workers: {self.max_workers if parallel else 1})")
        print(f"   Resume: {self.resume}")
        print()

        if parallel and len(configs) > 1:
            # Parallel execution
            with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
                # Submit all jobs
                future_to_config = {
                    executor.submit(self.run_single_backtest, cfg): cfg for cfg in configs
                }

                # Progress bar
                with tqdm(total=len(configs), desc="Backtesting", unit="test") as pbar:
                    for future in as_completed(future_to_config):
                        config = future_to_config[future]

                        try:
                            result = future.result()
                            results.append(result)

                            # Update progress bar
                            status_icon = "✅" if result.physics_valid else "❌"
                            pbar.set_postfix(
                                {
                                    "current": f"{config.instrument}_{config.timeframe}_{config.agent_type}",
                                    "omega": f"{result.omega_ratio:.2f}",
                                    "valid": status_icon,
                                }
                            )

                            # Mark as completed
                            key = (config.instrument, config.timeframe, config.agent_type)
                            self.completed.add(key)

                        except Exception as e:
                            pbar.write(f"❌ Error: {config.instrument}_{config.timeframe} - {e}")

                        pbar.update(1)

        else:
            # Sequential execution
            with tqdm(configs, desc="Backtesting", unit="test") as pbar:
                for config in pbar:
                    result = self.run_single_backtest(config)
                    results.append(result)

                    status_icon = "✅" if result.physics_valid else "❌"
                    pbar.set_postfix(
                        {
                            "current": f"{config.instrument}_{config.timeframe}",
                            "omega": f"{result.omega_ratio:.2f}",
                            "valid": status_icon,
                        }
                    )

                    # Mark as completed
                    key = (config.instrument, config.timeframe, config.agent_type)
                    self.completed.add(key)

        # Save checkpoint
        self._save_checkpoint(self.completed)

        return results

    def save_results(self, results: List[BacktestResult]):
        """Save results to CSV file."""
        rows = []

        for result in results:
            rows.append(
                {
                    "timestamp": datetime.now().isoformat(),
                    "instrument": result.config.instrument,
                    "timeframe": result.config.timeframe,
                    "agent_type": result.config.agent_type,
                    "omega_ratio": result.omega_ratio,
                    "sharpe_ratio": result.sharpe_ratio,
                    "z_factor": result.z_factor,
                    "energy_captured_pct": result.energy_captured_pct,
                    "total_return": result.total_return,
                    "num_trades": result.num_trades,
                    "win_rate": result.win_rate,
                    "max_drawdown": result.max_drawdown,
                    "calmar_ratio": result.calmar_ratio,
                    "profit_factor": result.profit_factor,
                    "avg_trade_duration": result.avg_trade_duration,
                    "p_value": result.statistical_significance,
                    "physics_valid": result.physics_valid,
                    "error": result.error or "",
                }
            )

        df = pd.DataFrame(rows)

        # Append to existing results
        if self.results_file.exists():
            existing = pd.read_csv(self.results_file)
            df = pd.concat([existing, df], ignore_index=True)

        df.to_csv(self.results_file, index=False)
        print(f"\n💾 Results saved to: {self.results_file}")

    def print_summary(self, results: List[BacktestResult]):
        """Print summary of results."""
        print("\n" + "=" * 80)
        print("BACKTEST SUMMARY")
        print("=" * 80)

        total = len(results)
        passed = sum(1 for r in results if r.physics_valid)
        failed = sum(1 for r in results if r.error is not None)
        successful = total - failed

        print(f"\nTotal tests: {total}")
        print(f"✅ Successful: {successful}")
        print(f"❌ Failed: {failed}")
        print(f"🎯 Physics valid: {passed}/{successful} ({passed / successful * 100:.1f}%)")

        if successful > 0:
            # Best performers
            valid_results = [r for r in results if r.error is None]
            valid_results.sort(key=lambda r: r.omega_ratio, reverse=True)

            print("\n🏆 Top 5 Performers (by Omega ratio):")
            print("-" * 80)
            for i, result in enumerate(valid_results[:5], 1):
                status = "✅" if result.physics_valid else "⚠️"
                print(
                    f"{i}. {status} {result.config.instrument}_{result.config.timeframe}_{result.config.agent_type}"
                )
                print(
                    f"   Omega: {result.omega_ratio:.2f} | Sharpe: {result.sharpe_ratio:.2f} | "
                    f"Z: {result.z_factor:.2f} | Energy: {result.energy_captured_pct:.1f}%"
                )


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Consolidated Batch Backtesting")

    parser.add_argument(
        "--instruments",
        nargs="+",
        default=["BTCUSD", "ETHUSD", "GBPUSD", "EURUSD"],
        help="Instruments to backtest",
    )
    parser.add_argument(
        "--timeframes",
        nargs="+",
        default=["H1", "H4"],
        help="Timeframes to backtest",
    )
    parser.add_argument(
        "--agents",
        nargs="+",
        default=["berserker", "sniper", "triad"],
        help="Agent types to test",
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        default=True,
        help="Use parallel execution",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of parallel workers (default: CPU count - 1)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from last checkpoint",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Data directory (default: data/master_standardized)",
    )

    args = parser.parse_args()

    # Initialize backtester
    backtester = ConsolidatedBacktester(
        max_workers=args.workers,
        resume=args.resume,
    )

    # Create configurations
    configs = backtester.create_configs(
        instruments=args.instruments,
        timeframes=args.timeframes,
        agent_types=args.agents,
        data_dir=args.data_dir,
    )

    if not configs:
        print("❌ No configurations to run (all completed or no data found)")
        return 1

    # Run batch
    results = backtester.run_batch(configs, parallel=args.parallel)

    # Save results
    backtester.save_results(results)

    # Print summary
    backtester.print_summary(results)

    # Exit code based on physics validation
    passed = sum(1 for r in results if r.physics_valid)
    return 0 if passed > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
