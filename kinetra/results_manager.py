"""
Results Manager - Experiment Tracking and Housekeeping

Organizes backtest runs with:
- Structured directory layout (by date/symbol/timeframe)
- JSON + Parquet storage for results and trades
- Automatic cleanup of old runs
- Run comparison and aggregation
"""

import json
import os
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .backtest_engine import BacktestResult, Trade, TradeDirection


class ResultsManager:
    """
    Manages backtest results storage and retrieval.

    Directory structure:
    results/
    ├── 2025-12-28/
    │   ├── BTCUSD/
    │   │   ├── H1/
    │   │   │   ├── run_001/
    │   │   │   │   ├── config.json
    │   │   │   │   ├── summary.json
    │   │   │   │   ├── trades.parquet
    │   │   │   │   ├── equity_curve.parquet
    │   │   │   │   └── plots/
    │   │   │   │       ├── equity_curve.png
    │   │   │   │       ├── trade_distribution.png
    │   │   │   │       └── drawdown.png
    │   │   │   └── run_002/
    │   │   └── M30/
    │   └── EURUSD/
    └── 2025-12-27/
    """

    def __init__(self, base_dir: str = "results"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def create_run(
        self, symbol: str, timeframe: str, config: Dict, date: Optional[datetime] = None
    ) -> Path:
        """
        Create a new run directory and return its path.

        Args:
            symbol: Trading symbol
            timeframe: Timeframe (H1, M30, etc.)
            config: Run configuration dict
            date: Date for organizing (defaults to today)

        Returns:
            Path to the new run directory
        """
        date = date or datetime.now()
        date_str = date.strftime("%Y-%m-%d")

        # Create directory structure
        run_dir = self.base_dir / date_str / symbol / timeframe
        run_dir.mkdir(parents=True, exist_ok=True)

        # Find next run number
        existing_runs = [d for d in run_dir.iterdir() if d.is_dir() and d.name.startswith("run_")]
        run_num = len(existing_runs) + 1

        run_path = run_dir / f"run_{run_num:03d}"
        run_path.mkdir(exist_ok=True)

        # Create plots subdirectory
        (run_path / "plots").mkdir(exist_ok=True)

        # Save config
        config_with_meta = {
            **config,
            "_meta": {
                "symbol": symbol,
                "timeframe": timeframe,
                "run_id": f"run_{run_num:03d}",
                "created_at": datetime.now().isoformat(),
                "date": date_str,
            },
        }

        with open(run_path / "config.json", "w") as f:
            json.dump(config_with_meta, f, indent=2, default=str)

        return run_path

    def save_result(
        self,
        run_path: Path,
        result: BacktestResult,
        save_trades: bool = True,
        save_equity: bool = True,
    ):
        """
        Save backtest result to run directory.

        Args:
            run_path: Path to run directory
            result: BacktestResult object
            save_trades: Whether to save individual trades
            save_equity: Whether to save equity curve
        """
        # Save summary
        summary = result.to_dict()
        summary["_saved_at"] = datetime.now().isoformat()

        with open(run_path / "summary.json", "w") as f:
            json.dump(summary, f, indent=2, default=str)

        # Save trades as parquet - Vectorized with list comprehension
        if save_trades and result.trades:
            trades_data = [
                {
                    "trade_id": t.trade_id,
                    "symbol": t.symbol,
                    "direction": t.direction.value,
                    "lots": t.lots,
                    "entry_time": t.entry_time,
                    "entry_price": t.entry_price,
                    "exit_time": t.exit_time,
                    "exit_price": t.exit_price,
                    "spread_cost": t.spread_cost,
                    "commission": t.commission,
                    "slippage": t.slippage,
                    "swap_cost": t.swap_cost,
                    "gross_pnl": t.gross_pnl,
                    "net_pnl": t.net_pnl,
                    "energy_at_entry": t.energy_at_entry,
                    "regime_at_entry": t.regime_at_entry,
                    "mfe": t.mfe,
                    "mae": t.mae,
                }
                for t in result.trades
            ]

            trades_df = pd.DataFrame(trades_data)
            trades_df.to_parquet(run_path / "trades.parquet", index=False)

        # Save equity curve
        if save_equity and result.equity_curve is not None:
            equity_df = pd.DataFrame(
                {"bar": range(len(result.equity_curve)), "equity": result.equity_curve.values}
            )
            equity_df.to_parquet(run_path / "equity_curve.parquet", index=False)

    def load_result(self, run_path: Path) -> Dict:
        """Load result from run directory."""
        summary_path = run_path / "summary.json"
        if not summary_path.exists():
            raise FileNotFoundError(f"No summary found in {run_path}")

        with open(summary_path) as f:
            summary = json.load(f)

        # Load trades if available
        trades_path = run_path / "trades.parquet"
        if trades_path.exists():
            summary["trades_df"] = pd.read_parquet(trades_path)

        # Load equity curve if available
        equity_path = run_path / "equity_curve.parquet"
        if equity_path.exists():
            summary["equity_df"] = pd.read_parquet(equity_path)

        return summary

    def list_runs(
        self,
        symbol: Optional[str] = None,
        timeframe: Optional[str] = None,
        date: Optional[str] = None,
        last_n_days: int = 7,
    ) -> List[Dict]:
        """
        List available runs with optional filtering.

        Args:
            symbol: Filter by symbol
            timeframe: Filter by timeframe
            date: Filter by specific date (YYYY-MM-DD)
            last_n_days: Only show runs from last N days

        Returns:
            List of run info dicts
        """
        runs = []

        # Determine date range
        if date:
            dates = [date]
        else:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=last_n_days)
            dates = [
                (start_date + timedelta(days=i)).strftime("%Y-%m-%d")
                for i in range(last_n_days + 1)
            ]

        for date_str in dates:
            date_path = self.base_dir / date_str
            if not date_path.exists():
                continue

            for symbol_dir in date_path.iterdir():
                if not symbol_dir.is_dir():
                    continue
                if symbol and symbol_dir.name != symbol:
                    continue

                for tf_dir in symbol_dir.iterdir():
                    if not tf_dir.is_dir():
                        continue
                    if timeframe and tf_dir.name != timeframe:
                        continue

                    for run_dir in tf_dir.iterdir():
                        if not run_dir.is_dir() or not run_dir.name.startswith("run_"):
                            continue

                        config_path = run_dir / "config.json"
                        summary_path = run_dir / "summary.json"

                        run_info = {
                            "path": str(run_dir),
                            "date": date_str,
                            "symbol": symbol_dir.name,
                            "timeframe": tf_dir.name,
                            "run_id": run_dir.name,
                        }

                        if summary_path.exists():
                            with open(summary_path) as f:
                                summary = json.load(f)
                            run_info["total_trades"] = summary.get("total_trades", 0)
                            run_info["net_pnl"] = summary.get("total_net_pnl", 0)
                            run_info["sharpe"] = summary.get("sharpe_ratio", 0)

                        runs.append(run_info)

        return sorted(runs, key=lambda x: (x["date"], x["symbol"], x["run_id"]), reverse=True)

    def compare_runs(self, run_paths: List[Path]) -> pd.DataFrame:
        """
        Compare multiple runs side by side.

        Args:
            run_paths: List of run directory paths

        Returns:
            DataFrame with comparison
        """
        comparisons = []

        for path in run_paths:
            path = Path(path)
            try:
                result = self.load_result(path)
                config_path = path / "config.json"

                with open(config_path) as f:
                    config = json.load(f)

                comparisons.append(
                    {
                        "run_id": config.get("_meta", {}).get("run_id", path.name),
                        "symbol": config.get("_meta", {}).get("symbol", ""),
                        "timeframe": config.get("_meta", {}).get("timeframe", ""),
                        "trades": result.get("total_trades", 0),
                        "win_rate": result.get("win_rate", 0),
                        "gross_pnl": result.get("total_gross_pnl", 0),
                        "total_costs": result.get("total_costs", 0),
                        "net_pnl": result.get("total_net_pnl", 0),
                        "max_dd": result.get("max_drawdown", 0),
                        "sharpe": result.get("sharpe_ratio", 0),
                        "omega": result.get("omega_ratio", 0),
                    }
                )
            except Exception as e:
                print(f"Error loading {path}: {e}")

        return pd.DataFrame(comparisons)

    def aggregate_monte_carlo(self, run_paths: List[Path]) -> Dict:
        """
        Aggregate results from Monte Carlo runs.

        Args:
            run_paths: List of MC run paths

        Returns:
            Aggregated statistics
        """
        metrics = {
            "net_pnl": [],
            "sharpe_ratio": [],
            "omega_ratio": [],
            "max_drawdown": [],
            "win_rate": [],
        }

        for path in run_paths:
            try:
                result = self.load_result(Path(path))
                for key in metrics:
                    if key in result:
                        metrics[key].append(result[key])
            except Exception:
                continue

        aggregated = {}
        for key, values in metrics.items():
            if values:
                arr = np.array(values)
                aggregated[key] = {
                    "mean": float(np.mean(arr)),
                    "std": float(np.std(arr)),
                    "min": float(np.min(arr)),
                    "max": float(np.max(arr)),
                    "median": float(np.median(arr)),
                    "p5": float(np.percentile(arr, 5)),
                    "p95": float(np.percentile(arr, 95)),
                }

        return aggregated

    def cleanup_old_runs(self, days_to_keep: int = 30, dry_run: bool = True) -> List[str]:
        """
        Remove runs older than specified days.

        Args:
            days_to_keep: Keep runs from last N days
            dry_run: If True, only list what would be deleted

        Returns:
            List of deleted (or would-be-deleted) paths
        """
        cutoff = datetime.now() - timedelta(days=days_to_keep)
        to_delete = []

        for date_dir in self.base_dir.iterdir():
            if not date_dir.is_dir():
                continue

            try:
                dir_date = datetime.strptime(date_dir.name, "%Y-%m-%d")
                if dir_date < cutoff:
                    to_delete.append(str(date_dir))
                    if not dry_run:
                        shutil.rmtree(date_dir)
            except ValueError:
                continue

        return to_delete

    def get_disk_usage(self) -> Dict:
        """Get disk usage statistics for results directory."""
        total_size = 0
        file_count = 0
        run_count = 0

        for root, dirs, files in os.walk(self.base_dir):
            for f in files:
                file_path = Path(root) / f
                total_size += file_path.stat().st_size
                file_count += 1

            if Path(root).name.startswith("run_"):
                run_count += 1

        return {
            "total_size_mb": total_size / (1024 * 1024),
            "total_files": file_count,
            "total_runs": run_count,
            "path": str(self.base_dir),
        }


def format_run_summary(result: Dict) -> str:
    """Format a run result for console output."""
    lines = [
        "=" * 60,
        f"BACKTEST SUMMARY",
        "=" * 60,
        "",
        f"Trades: {result.get('total_trades', 0)} "
        f"(W: {result.get('winning_trades', 0)} / L: {result.get('losing_trades', 0)})",
        f"Win Rate: {result.get('win_rate', 0):.1%}",
        "",
        "P&L Breakdown:",
        f"  Gross P&L:    ${result.get('total_gross_pnl', 0):>12,.2f}",
        f"  Total Costs:  ${result.get('total_costs', 0):>12,.2f}",
    ]

    costs = result.get("cost_breakdown", {})
    if costs:
        lines.extend(
            [
                f"    - Spread:     ${costs.get('spread', 0):>10,.2f}",
                f"    - Commission: ${costs.get('commission', 0):>10,.2f}",
                f"    - Slippage:   ${costs.get('slippage', 0):>10,.2f}",
                f"    - Swap:       ${costs.get('swap', 0):>10,.2f}",
            ]
        )

    lines.extend(
        [
            f"  Net P&L:      ${result.get('total_net_pnl', 0):>12,.2f}",
            "",
            "Risk Metrics:",
            f"  Max Drawdown: ${result.get('max_drawdown', 0):>10,.2f} "
            f"({result.get('max_drawdown_pct', 0):.1f}%)",
            f"  Sharpe Ratio: {result.get('sharpe_ratio', 0):>10.2f}",
            f"  Omega Ratio:  {result.get('omega_ratio', 0):>10.2f}",
            f"  Z-Factor:     {result.get('z_factor', 0):>10.2f}",
            "",
            "Physics Metrics:",
            f"  Energy Captured: {result.get('energy_captured_pct', 0):.1%}",
            "=" * 60,
        ]
    )

    return "\n".join(lines)
