#!/usr/bin/env python3
"""
Multi-Dataset Testing Harness
==============================

Runs assumption-free research across all 87 datasets.
Discovers regimes, identifies patterns, and produces comprehensive analysis.

Usage:
    python scripts/research/multi_dataset_harness.py [--parallel N] [--output DIR]
"""

import argparse
import json
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

warnings.filterwarnings("ignore")

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd

from kinetra.config import MAX_WORKERS
from kinetra.dsp_features import DSPFeatureEngine, extract_dsp_features
from kinetra.liquidity_features import LiquidityFeatureEngine, extract_liquidity_features
from kinetra.regime_discovery import (
    CrossAssetRegimeAnalyzer,
    RegimeDiscoveryEngine,
    RegimeDiscoveryResult,
    discover_regimes,
)


@dataclass
class DatasetInfo:
    """Metadata about a dataset."""

    path: Path
    symbol: str
    timeframe: str
    asset_class: str
    start_date: str
    end_date: str
    bar_count: int


@dataclass
class DatasetResult:
    """Results from analyzing a single dataset."""

    dataset_key: str
    asset_class: str
    timeframe: str
    bar_count: int
    n_regimes: int
    regime_labels: List[str]
    aic: float
    bic: float
    fat_candle_count: int
    fat_candle_pct: float
    mean_return: float
    volatility: float
    up_persistence: float
    down_persistence: float
    persistence_asymmetry: float
    dominant_scale: int
    processing_time: float
    error: Optional[str] = None


class DatasetDiscovery:
    """Discovers and catalogs all available datasets."""

    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.datasets: List[DatasetInfo] = []

    def discover(self) -> List[DatasetInfo]:
        """Find all CSV datasets in master directory."""
        master_dir = self.data_dir / "master"

        if not master_dir.exists():
            print(f"[ERROR] Master data directory not found: {master_dir}")
            return []

        # Vectorized: nested list comprehension instead of nested loops
        datasets = [
            info
            for asset_class_dir in master_dir.iterdir()
            if asset_class_dir.is_dir()
            for csv_file in asset_class_dir.glob("*.csv")
            if (info := self._parse_filename(csv_file, asset_class_dir.name)) is not None
        ]

        self.datasets = sorted(datasets, key=lambda x: (x.asset_class, x.symbol, x.timeframe))
        return self.datasets

    def _parse_filename(self, path: Path, asset_class: str) -> Optional[DatasetInfo]:
        """Parse dataset info from filename."""
        # Format: SYMBOL_TIMEFRAME_STARTDATE_ENDDATE.csv
        name = path.stem
        parts = name.split("_")

        if len(parts) < 4:
            return None

        symbol = parts[0]
        timeframe = parts[1]

        # Quick row count
        try:
            with open(path, "r") as f:
                bar_count = sum(1 for _ in f) - 1  # Subtract header
        except Exception:
            bar_count = 0

        return DatasetInfo(
            path=path,
            symbol=symbol,
            timeframe=timeframe,
            asset_class=asset_class,
            start_date=parts[2] if len(parts) > 2 else "",
            end_date=parts[3] if len(parts) > 3 else "",
            bar_count=bar_count,
        )

    def get_by_class(self, asset_class: str) -> List[DatasetInfo]:
        """Get datasets for a specific asset class."""
        return [d for d in self.datasets if d.asset_class == asset_class]

    def get_by_timeframe(self, timeframe: str) -> List[DatasetInfo]:
        """Get datasets for a specific timeframe."""
        return [d for d in self.datasets if d.timeframe == timeframe]


def load_dataset(info: DatasetInfo) -> pd.DataFrame:
    """Load a dataset from CSV."""
    df = pd.read_csv(info.path, sep="\t")

    # Standardize column names
    df.columns = [c.strip("<>").lower() for c in df.columns]

    # Parse datetime
    if "date" in df.columns and "time" in df.columns:
        df["datetime"] = pd.to_datetime(df["date"] + " " + df["time"])
    elif "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"])

    # Ensure required columns
    required = ["open", "high", "low", "close"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    return df


def analyze_single_dataset(info: DatasetInfo) -> DatasetResult:
    """
    Analyze a single dataset.

    This is the core analysis function run on each dataset.
    """
    start_time = datetime.now()
    dataset_key = f"{info.symbol}_{info.timeframe}"

    try:
        # Load data
        df = load_dataset(info)

        if len(df) < 200:
            return DatasetResult(
                dataset_key=dataset_key,
                asset_class=info.asset_class,
                timeframe=info.timeframe,
                bar_count=len(df),
                n_regimes=0,
                regime_labels=[],
                aic=0.0,
                bic=0.0,
                fat_candle_count=0,
                fat_candle_pct=0.0,
                mean_return=0.0,
                volatility=0.0,
                up_persistence=0.5,
                down_persistence=0.5,
                persistence_asymmetry=0.0,
                dominant_scale=0,
                processing_time=0.0,
                error="Insufficient data (< 200 bars)",
            )

        # Compute log returns
        log_returns = np.log(df["close"] / df["close"].shift(1)).dropna().values

        # Basic statistics
        mean_return = np.mean(log_returns)
        volatility = np.std(log_returns)

        # Fat candle detection (range > 3x median)
        ranges = (df["high"] - df["low"]).values
        median_range = np.median(ranges)
        fat_candles = np.sum(ranges > 3 * median_range)
        fat_pct = fat_candles / len(ranges) * 100

        # Extract DSP features for sample
        dsp_engine = DSPFeatureEngine()
        sample_dsp = dsp_engine.extract_all(df, bar_idx=len(df) - 1)
        up_persistence = sample_dsp.get("up_persistence", 0.5)
        down_persistence = sample_dsp.get("down_persistence", 0.5)
        persistence_asymmetry = sample_dsp.get("persistence_asymmetry", 0.0)
        dominant_scale = sample_dsp.get("wavelet_dominant_scale", 0)

        # Regime discovery
        regime_engine = RegimeDiscoveryEngine(min_regimes=2, max_regimes=8)
        regime_result = regime_engine.fit(df, start_bar=100)

        # Extract regime labels
        regime_labels = [p.label for p in regime_result.regime_profiles]

        processing_time = (datetime.now() - start_time).total_seconds()

        return DatasetResult(
            dataset_key=dataset_key,
            asset_class=info.asset_class,
            timeframe=info.timeframe,
            bar_count=len(df),
            n_regimes=regime_result.n_regimes,
            regime_labels=regime_labels,
            aic=regime_result.aic,
            bic=regime_result.bic,
            fat_candle_count=int(fat_candles),
            fat_candle_pct=fat_pct,
            mean_return=mean_return,
            volatility=volatility,
            up_persistence=up_persistence,
            down_persistence=down_persistence,
            persistence_asymmetry=persistence_asymmetry,
            dominant_scale=dominant_scale,
            processing_time=processing_time,
            error=None,
        )

    except Exception as e:
        processing_time = (datetime.now() - start_time).total_seconds()
        return DatasetResult(
            dataset_key=dataset_key,
            asset_class=info.asset_class,
            timeframe=info.timeframe,
            bar_count=info.bar_count,
            n_regimes=0,
            regime_labels=[],
            aic=0.0,
            bic=0.0,
            fat_candle_count=0,
            fat_candle_pct=0.0,
            mean_return=0.0,
            volatility=0.0,
            up_persistence=0.5,
            down_persistence=0.5,
            persistence_asymmetry=0.0,
            dominant_scale=0,
            processing_time=processing_time,
            error=str(e),
        )


class TestHarness:
    """
    Orchestrates testing across all datasets.
    """

    def __init__(self, data_dir: Path, output_dir: Path, n_workers: int = None):
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.n_workers = n_workers or min(MAX_WORKERS, 8)
        self.discovery = DatasetDiscovery(data_dir)
        self.results: List[DatasetResult] = []

    def run(self, parallel: bool = True) -> List[DatasetResult]:
        """
        Run analysis across all datasets.
        """
        # Discover datasets
        datasets = self.discovery.discover()
        print(
            f"\n[HARNESS] Found {len(datasets)} datasets across {len(set(d.asset_class for d in datasets))} asset classes"
        )

        # Print breakdown
        by_class = {}
        for d in datasets:
            by_class[d.asset_class] = by_class.get(d.asset_class, 0) + 1
        for cls, count in sorted(by_class.items()):
            print(f"  {cls}: {count}")

        print(f"\n[HARNESS] Starting analysis with {self.n_workers} workers...")
        start_time = datetime.now()

        if parallel and self.n_workers > 1:
            self.results = self._run_parallel(datasets)
        else:
            self.results = self._run_sequential(datasets)

        elapsed = (datetime.now() - start_time).total_seconds()
        print(f"\n[HARNESS] Completed in {elapsed:.1f}s ({len(self.results)} datasets)")

        # Save results
        self._save_results()

        return self.results

    def _run_parallel(self, datasets: List[DatasetInfo]) -> List[DatasetResult]:
        """Run analysis in parallel."""
        results = []

        with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
            futures = {executor.submit(analyze_single_dataset, info): info for info in datasets}

            for i, future in enumerate(as_completed(futures)):
                info = futures[future]
                try:
                    result = future.result(timeout=300)  # 5 min timeout
                    results.append(result)

                    status = "OK" if result.error is None else f"WARN: {result.error[:30]}"
                    print(
                        f"  [{i + 1}/{len(datasets)}] {result.dataset_key}: {result.n_regimes} regimes ({status})"
                    )

                except Exception as e:
                    print(
                        f"  [{i + 1}/{len(datasets)}] {info.symbol}_{info.timeframe}: ERROR - {e}"
                    )

        return results

    def _run_sequential(self, datasets: List[DatasetInfo]) -> List[DatasetResult]:
        """Run analysis sequentially (for debugging)."""
        results = []

        for i, info in enumerate(datasets):
            result = analyze_single_dataset(info)
            results.append(result)

            status = "OK" if result.error is None else f"WARN: {result.error[:30]}"
            print(
                f"  [{i + 1}/{len(datasets)}] {result.dataset_key}: {result.n_regimes} regimes ({status})"
            )

        return results

    def _save_results(self):
        """Save results to JSON and CSV."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save as JSON
        json_path = self.output_dir / f"harness_results_{timestamp}.json"
        with open(json_path, "w") as f:
            json.dump([asdict(r) for r in self.results], f, indent=2, default=str)
        print(f"[HARNESS] Results saved to {json_path}")

        # Save as CSV for easy analysis
        csv_path = self.output_dir / f"harness_results_{timestamp}.csv"
        df = pd.DataFrame([asdict(r) for r in self.results])
        df.to_csv(csv_path, index=False)
        print(f"[HARNESS] Results saved to {csv_path}")

    def generate_summary(self) -> Dict:
        """
        Generate summary statistics from results.
        """
        if not self.results:
            return {}

        successful = [r for r in self.results if r.error is None]

        summary = {
            "total_datasets": len(self.results),
            "successful": len(successful),
            "failed": len(self.results) - len(successful),
            "by_asset_class": {},
            "by_timeframe": {},
            "regime_statistics": {},
            "feature_insights": {},
        }

        # By asset class
        for cls in set(r.asset_class for r in successful):
            class_results = [r for r in successful if r.asset_class == cls]
            summary["by_asset_class"][cls] = {
                "count": len(class_results),
                "avg_regimes": np.mean([r.n_regimes for r in class_results]),
                "avg_volatility": np.mean([r.volatility for r in class_results]),
                "avg_up_persistence": np.mean([r.up_persistence for r in class_results]),
                "avg_down_persistence": np.mean([r.down_persistence for r in class_results]),
                "avg_persistence_asymmetry": np.mean(
                    [r.persistence_asymmetry for r in class_results]
                ),
                "avg_fat_candle_pct": np.mean([r.fat_candle_pct for r in class_results]),
            }

        # By timeframe
        for tf in set(r.timeframe for r in successful):
            tf_results = [r for r in successful if r.timeframe == tf]
            summary["by_timeframe"][tf] = {
                "count": len(tf_results),
                "avg_regimes": np.mean([r.n_regimes for r in tf_results]),
                "avg_volatility": np.mean([r.volatility for r in tf_results]),
            }

        # Regime label frequency
        all_labels = []
        for r in successful:
            all_labels.extend(r.regime_labels)

        label_counts = {}
        for label in all_labels:
            label_counts[label] = label_counts.get(label, 0) + 1

        summary["regime_statistics"]["label_frequency"] = dict(
            sorted(label_counts.items(), key=lambda x: -x[1])
        )
        summary["regime_statistics"]["avg_regimes_per_dataset"] = np.mean(
            [r.n_regimes for r in successful]
        )

        # Feature insights - Persistence distribution (replaces Hurst)
        # Persistence > 0.5 = trending, < 0.5 = mean-reverting
        summary["feature_insights"]["persistence_distribution"] = {
            "trending_pct": np.mean(
                [r.up_persistence > 0.55 or r.down_persistence > 0.55 for r in successful]
            )
            * 100,
            "mean_reverting_pct": np.mean(
                [r.up_persistence < 0.45 and r.down_persistence < 0.45 for r in successful]
            )
            * 100,
            "random_walk_pct": np.mean(
                [
                    0.45 <= r.up_persistence <= 0.55 and 0.45 <= r.down_persistence <= 0.55
                    for r in successful
                ]
            )
            * 100,
            "asymmetric_pct": np.mean([abs(r.persistence_asymmetry) > 0.1 for r in successful])
            * 100,
        }

        return summary


def main():
    parser = argparse.ArgumentParser(description="Multi-dataset testing harness")
    parser.add_argument("--parallel", type=int, default=4, help="Number of parallel workers")
    parser.add_argument("--output", type=str, default="data/research", help="Output directory")
    parser.add_argument("--sequential", action="store_true", help="Run sequentially (debug mode)")

    args = parser.parse_args()

    data_dir = project_root / "data"
    output_dir = project_root / args.output

    harness = TestHarness(data_dir, output_dir, n_workers=args.parallel)

    # Run analysis
    results = harness.run(parallel=not args.sequential)

    # Generate and print summary
    summary = harness.generate_summary()

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print(f"\nTotal: {summary.get('total_datasets', 0)} datasets")
    print(f"Successful: {summary.get('successful', 0)}")
    print(f"Failed: {summary.get('failed', 0)}")

    print("\n--- By Asset Class ---")
    for cls, stats in summary.get("by_asset_class", {}).items():
        print(
            f"  {cls}: {stats['count']} datasets, "
            f"avg {stats['avg_regimes']:.1f} regimes, "
            f"up_persist {stats['avg_up_persistence']:.3f}, "
            f"down_persist {stats['avg_down_persistence']:.3f}"
        )

    print("\n--- By Timeframe ---")
    for tf, stats in summary.get("by_timeframe", {}).items():
        print(f"  {tf}: {stats['count']} datasets, avg {stats['avg_regimes']:.1f} regimes")

    print("\n--- Regime Labels (most common) ---")
    labels = summary.get("regime_statistics", {}).get("label_frequency", {})
    for label, count in list(labels.items())[:10]:
        print(f"  {label}: {count} occurrences")

    print("\n--- Persistence Distribution ---")
    persist_dist = summary.get("feature_insights", {}).get("persistence_distribution", {})
    print(f"  Trending (persist > 0.55): {persist_dist.get('trending_pct', 0):.1f}%")
    print(f"  Mean-reverting (persist < 0.45): {persist_dist.get('mean_reverting_pct', 0):.1f}%")
    print(f"  Random walk: {persist_dist.get('random_walk_pct', 0):.1f}%")
    print(f"  Asymmetric (|asym| > 0.1): {persist_dist.get('asymmetric_pct', 0):.1f}%")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
