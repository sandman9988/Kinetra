#!/usr/bin/env python3
"""
Fat Candle Forensics - What Triggers Big Moves?

RESEARCH QUESTION:
What conditions precede large market movements ("fat candles")?
Are these conditions consistent across asset classes?
Can we predict the next fat candle?

APPROACH:
1. Find all fat candles (>3 ATR moves)
2. Look back 5-50 bars before each
3. Identify which measurements were elevated
4. Test if patterns are consistent
5. Build predictive models
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


@dataclass
class FatCandlePattern:
    """Pattern observed before fat candles."""

    asset_class: str
    measurements_elevated: List[str]
    frequency: int  # How often this pattern occurs
    success_rate: float  # How often it leads to fat candle


class FatCandleForensics:
    """
    Analyze what triggers large market movements.

    Goal: Empirically discover preconditions for big moves.
    """

    def __init__(self, atr_threshold: float = 3.0, lookback_bars: int = 20):
        """
        Args:
            atr_threshold: Fat candle = range > threshold * ATR
            lookback_bars: How far back to look for preconditions
        """
        self.atr_threshold = atr_threshold
        self.lookback_bars = lookback_bars

    def find_fat_candles(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Identify all fat candles in the dataset.

        Returns DataFrame with fat candle locations and properties.
        """
        # Calculate ATR
        bar_range = df["high"] - df["low"]
        atr_20 = bar_range.rolling(20).mean()

        # Find fat candles
        fat_mask = bar_range > (self.atr_threshold * atr_20)

        # Build detailed fat candle DataFrame
        fat_indices = np.where(fat_mask)[0]
        fat_candles = []

        for idx in fat_indices:
            if idx < self.lookback_bars:  # Need lookback history
                continue

            fat_candles.append(
                {
                    "index": idx,
                    "timestamp": df.index[idx],
                    "open": df["open"].iloc[idx],
                    "high": df["high"].iloc[idx],
                    "low": df["low"].iloc[idx],
                    "close": df["close"].iloc[idx],
                    "range": bar_range.iloc[idx],
                    "atr": atr_20.iloc[idx],
                    "range_atr_ratio": (bar_range.iloc[idx] / atr_20.iloc[idx]),
                    "direction": 1 if df["close"].iloc[idx] > df["open"].iloc[idx] else -1,
                    "returns": (df["close"].iloc[idx] / df["close"].iloc[idx - 1]) - 1,
                }
            )

        return pd.DataFrame(fat_candles)

    def analyze_preconditions(
        self, df: pd.DataFrame, physics: pd.DataFrame, fat_candles: pd.DataFrame
    ) -> Dict:
        """
        Analyze what was happening before each fat candle.

        Returns dictionary of precondition statistics.
        """
        preconditions = defaultdict(list)

        # For each fat candle
        for _, fat_candle in fat_candles.iterrows():
            idx = fat_candle["index"]

            # Look back N bars
            lookback_slice = slice(idx - self.lookback_bars, idx)

            # Check raw price action
            recent_returns = df["close"].pct_change().iloc[lookback_slice]
            preconditions["returns_mean_before"].append(recent_returns.mean())
            preconditions["returns_std_before"].append(recent_returns.std())
            preconditions["returns_sum_before"].append(recent_returns.sum())  # Cumulative move

            # Check volume behavior
            recent_volume = df["volume"].iloc[lookback_slice]
            volume_ma = df["volume"].rolling(50).mean().iloc[idx]
            preconditions["volume_ratio_before"].append(
                recent_volume.mean() / volume_ma if volume_ma > 0 else 1.0
            )

            # Check physics measurements
            if "energy" in physics.columns:
                recent_energy = physics["energy"].iloc[lookback_slice]
                preconditions["energy_mean_before"].append(recent_energy.mean())
                preconditions["energy_max_before"].append(recent_energy.max())
                preconditions["energy_increase_before"].append(
                    recent_energy.iloc[-1] - recent_energy.iloc[0]
                )

            if "damping" in physics.columns:
                recent_damping = physics["damping"].iloc[lookback_slice]
                preconditions["damping_mean_before"].append(recent_damping.mean())
                preconditions["damping_min_before"].append(recent_damping.min())

            if "reynolds" in physics.columns:
                recent_reynolds = physics["reynolds"].iloc[lookback_slice]
                preconditions["reynolds_mean_before"].append(recent_reynolds.mean())
                preconditions["reynolds_max_before"].append(recent_reynolds.max())

            if "entropy" in physics.columns:
                recent_entropy = physics["entropy"].iloc[lookback_slice]
                preconditions["entropy_mean_before"].append(recent_entropy.mean())

        # Convert to DataFrame for easy analysis
        preconditions_df = pd.DataFrame(preconditions)

        # Compute summary statistics
        summary = {
            "n_fat_candles": len(fat_candles),
            "preconditions_stats": preconditions_df.describe().to_dict(),
        }

        return summary

    def compare_to_normal_bars(
        self,
        df: pd.DataFrame,
        physics: pd.DataFrame,
        fat_candles: pd.DataFrame,
        n_normal_samples: int = 1000,
    ) -> pd.DataFrame:
        """
        Compare preconditions before fat candles vs normal bars.

        HYPOTHESIS: If our measurements are useful, they should be
        significantly different before fat candles.
        """
        from scipy import stats

        # Vectorized preconditions computation
        indices = fat_candles["index"].values
        valid_mask = indices >= self.lookback_bars
        valid_indices = indices[valid_mask]
        n_valid = len(valid_indices)

        # Precompute series once
        returns_std_series = df["close"].pct_change().rolling(self.lookback_bars).std()
        volume_series = df["volume"]
        volume_ma_series = volume_series.rolling(50).mean()

        # Preallocate arrays
        fat_data = {
            "returns_std": np.full(n_valid, np.nan),
            "volume_ratio": np.full(n_valid, np.nan),
            "energy_mean": np.full(n_valid, np.nan),
            "damping_mean": np.full(n_valid, np.nan),
            "reynolds_mean": np.full(n_valid, np.nan),
        }

        # Compute for valid indices
        for i, idx in enumerate(valid_indices):
            lookback_slice = slice(idx - self.lookback_bars, idx)

            fat_data["returns_std"][i] = returns_std_series.iloc[idx]
            vol_ma = volume_ma_series.iloc[idx]
            if vol_ma > 0:
                fat_data["volume_ratio"][i] = volume_series.iloc[lookback_slice].mean() / vol_ma

            if "energy" in physics.columns:
                fat_data["energy_mean"][i] = physics["energy"].iloc[lookback_slice].mean()
            if "damping" in physics.columns:
                fat_data["damping_mean"][i] = physics["damping"].iloc[lookback_slice].mean()
            if "reynolds" in physics.columns:
                fat_data["reynolds_mean"][i] = physics["reynolds"].iloc[lookback_slice].mean()

        fat_df = pd.DataFrame(fat_data)

        # Sample normal bars (not fat candles)
        normal_indices = np.random.choice(
            range(self.lookback_bars, len(df) - 1),
            size=min(n_normal_samples, len(df) - self.lookback_bars),
            replace=False,
        )

        normal_preconditions = []

        for idx in normal_indices:
            # Skip if this is a fat candle
            bar_range = df["high"].iloc[idx] - df["low"].iloc[idx]
            atr = (df["high"] - df["low"]).rolling(20).mean().iloc[idx]
            if bar_range > (self.atr_threshold * atr):
                continue

            lookback_slice = slice(idx - self.lookback_bars, idx)

            normal_preconditions.append(
                {
                    "returns_std": df["close"].pct_change().iloc[lookback_slice].std(),
                    "volume_ratio": df["volume"].iloc[lookback_slice].mean()
                    / df["volume"].rolling(50).mean().iloc[idx],
                    "energy_mean": physics["energy"].iloc[lookback_slice].mean()
                    if "energy" in physics.columns
                    else np.nan,
                    "damping_mean": physics["damping"].iloc[lookback_slice].mean()
                    if "damping" in physics.columns
                    else np.nan,
                    "reynolds_mean": physics["reynolds"].iloc[lookback_slice].mean()
                    if "reynolds" in physics.columns
                    else np.nan,
                }
            )

        normal_df = pd.DataFrame(normal_preconditions)

        # Statistical comparison
        comparison = []

        for col in fat_df.columns:
            fat_values = fat_df[col].dropna()
            normal_values = normal_df[col].dropna()

            if len(fat_values) < 10 or len(normal_values) < 10:
                continue

            # T-test
            t_stat, p_value = stats.ttest_ind(fat_values, normal_values)

            # Effect size (Cohen's d)
            cohens_d = (fat_values.mean() - normal_values.mean()) / np.sqrt(
                (fat_values.std() ** 2 + normal_values.std() ** 2) / 2
            )

            comparison.append(
                {
                    "measurement": col,
                    "fat_candle_mean": fat_values.mean(),
                    "normal_mean": normal_values.mean(),
                    "difference": fat_values.mean() - normal_values.mean(),
                    "cohens_d": cohens_d,
                    "t_statistic": t_stat,
                    "p_value": p_value,
                    "significant": p_value < 0.05,
                }
            )

        comparison_df = pd.DataFrame(comparison)
        comparison_df = comparison_df.sort_values("cohens_d", ascending=False, key=abs)

        return comparison_df

    def find_cross_class_patterns(self, results_by_class: Dict[str, List[Dict]]) -> pd.DataFrame:
        """
        Find precondition patterns that are consistent across asset classes.

        HYPOTHESIS: If a pattern appears in all classes, it's fundamental.
        If it only appears in one class, it's class-specific.
        """
        patterns = []

        for asset_class, class_results in results_by_class.items():
            for result in class_results:
                patterns.append(
                    {
                        "asset_class": asset_class,
                        "instrument": result["instrument"],
                        **result["preconditions"],
                    }
                )

        df = pd.DataFrame(patterns)

        # Group by asset class and compute mean
        class_summary = df.groupby("asset_class").mean()

        print("\n📊 Cross-Class Fat Candle Preconditions:")
        print(class_summary)

        return class_summary


def main():
    """Run fat candle forensics on all instruments."""
    print("=" * 80)
    print("FAT CANDLE FORENSICS - What Triggers Big Moves?")
    print("=" * 80)

    from kinetra.data_loader import UnifiedDataLoader

    data_dir = Path("data/prepared/train")

    if not data_dir.exists():
        print(f"\n❌ Data directory not found: {data_dir}")
        return

    forensics = FatCandleForensics(atr_threshold=3.0, lookback_bars=20)

    # Analyze instruments from different classes
    class_samples = {
        "forex": ["EURUSD+", "GBPUSD+", "USDJPY+"],
        "crypto": ["BTCUSD", "ETHUSD"],
        "indices": ["NAS100", "SPX500"],
        "metals": ["XAUUSD+", "XAGUSD"],
    }

    results_by_class = defaultdict(list)

    for asset_class, instruments in class_samples.items():
        print(f"\n{'=' * 80}")
        print(f"Analyzing {asset_class.upper()}")
        print("=" * 80)

        for instrument in instruments:
            # Find instrument file
            files = list(data_dir.glob(f"{instrument}_H1_*.csv"))
            if not files:
                print(f"  ⏭️  Skipping {instrument} (not found)")
                continue

            csv_file = files[0]
            print(f"\n🔍 {instrument} H1")

            # Load data
            loader = UnifiedDataLoader(verbose=False, compute_physics=True)
            pkg = loader.load(str(csv_file))
            df = pkg.to_backtest_engine_format()
            physics = pkg.physics_state

            # Find fat candles
            fat_candles = forensics.find_fat_candles(df)
            print(f"  Fat candles found: {len(fat_candles)}")

            if len(fat_candles) < 5:
                print(f"  ⏭️  Skipping (too few fat candles)")
                continue

            # Analyze preconditions
            preconditions = forensics.analyze_preconditions(df, physics, fat_candles)

            # Compare to normal bars
            comparison = forensics.compare_to_normal_bars(df, physics, fat_candles)

            print("\n  📈 Measurements significantly different before fat candles:")
            significant = comparison[comparison["significant"]]
            # Vectorized string formatting
            if len(significant) > 0:
                measurements = significant["measurement"].values
                cohens_d = significant["cohens_d"].values
                p_values = significant["p_value"].values
                for meas, d, p in zip(measurements, cohens_d, p_values):
                    print(f"    {meas:20s}: Cohen's d = {d:6.2f}, p = {p:.4f}")

            results_by_class[asset_class].append(
                {
                    "instrument": instrument,
                    "n_fat_candles": len(fat_candles),
                    "preconditions": preconditions["preconditions_stats"],
                    "significant_measurements": significant["measurement"].tolist(),
                }
            )

    # Cross-class comparison
    print("\n" + "=" * 80)
    print("CROSS-CLASS PATTERN ANALYSIS")
    print("=" * 80)

    # Find common preconditions across all classes
    all_significant = defaultdict(int)
    for class_results in results_by_class.values():
        for result in class_results:
            for measurement in result["significant_measurements"]:
                all_significant[measurement] += 1

    print("\nMeasurements significant across multiple classes:")
    for measurement, count in sorted(all_significant.items(), key=lambda x: -x[1]):
        print(f"  {measurement:30s}: {count} instruments")

    # Save results
    output_dir = Path("research_output/fat_candles")
    output_dir.mkdir(parents=True, exist_ok=True)

    import json

    with open(output_dir / "forensics_results.json", "w") as f:
        json.dump(dict(results_by_class), f, indent=2, default=str)

    print(f"\n💾 Saved: {output_dir / 'forensics_results.json'}")
    print("\n✅ Fat candle forensics complete!")


if __name__ == "__main__":
    main()
