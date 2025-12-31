#!/usr/bin/env python3
"""
Measurement Toolkit - First Principles Research

ASSUMPTION: We know nothing about what drives markets.

This toolkit provides:
1. Raw observable measurements (price, volume, time)
2. Derived measurements (returns, volatility, etc.)
3. Physics-inspired measurements (energy, damping, etc.)
4. Statistical validation (correlation, predictive power)

Purpose: Discover empirically what actually matters.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
from dataclasses import dataclass
from collections import defaultdict


@dataclass
class MeasurementResult:
    """Results from measuring an instrument."""
    instrument: str
    timeframe: str
    asset_class: str

    # Raw observations
    raw_measurements: pd.DataFrame

    # Derived measurements
    derived_measurements: pd.DataFrame

    # Physics measurements
    physics_measurements: pd.DataFrame

    # Statistical properties
    statistics: Dict

    # Fat candle analysis
    fat_candles: pd.DataFrame


class MeasurementToolkit:
    """
    First-principles measurement toolkit.

    Philosophy: Measure everything, assume nothing.
    """

    def __init__(self, verbose: bool = True):
        self.verbose = verbose

    def measure_instrument(
        self,
        df: pd.DataFrame,
        instrument: str,
        timeframe: str,
        asset_class: str = "unknown"
    ) -> MeasurementResult:
        """
        Measure everything observable about an instrument.

        Args:
            df: OHLCV DataFrame
            instrument: Symbol name
            timeframe: M15, H1, etc.
            asset_class: forex, crypto, indices, metals

        Returns:
            Complete measurement suite
        """
        if self.verbose:
            print(f"\n📏 Measuring {instrument} {timeframe} ({asset_class})")
            print(f"   Bars: {len(df):,}")

        # 1. Raw observations (what we can directly observe)
        raw = self._measure_raw_observables(df)

        # 2. Derived measurements (computed from raw)
        derived = self._measure_derived(df, raw)

        # 3. Physics measurements (our current hypotheses)
        physics = self._measure_physics(df)

        # 4. Statistical properties
        stats = self._compute_statistics(raw, derived, physics)

        # 5. Fat candle analysis (big moves)
        fat_candles = self._analyze_fat_candles(df, derived)

        if self.verbose:
            print(f"   Raw measurements: {len(raw.columns)}")
            print(f"   Derived measurements: {len(derived.columns)}")
            print(f"   Physics measurements: {len(physics.columns)}")
            print(f"   Fat candles found: {len(fat_candles)}")

        return MeasurementResult(
            instrument=instrument,
            timeframe=timeframe,
            asset_class=asset_class,
            raw_measurements=raw,
            derived_measurements=derived,
            physics_measurements=physics,
            statistics=stats,
            fat_candles=fat_candles,
        )

    def _measure_raw_observables(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Raw observations - what we can directly see.

        These are the ONLY things we truly know.
        """
        raw = pd.DataFrame(index=df.index)

        # Price observations
        raw['open'] = df['open']
        raw['high'] = df['high']
        raw['low'] = df['low']
        raw['close'] = df['close']
        raw['volume'] = df['volume']

        # Range observations
        raw['bar_range'] = df['high'] - df['low']
        raw['bar_body'] = (df['close'] - df['open']).abs()
        raw['upper_wick'] = df['high'] - df[['open', 'close']].max(axis=1)
        raw['lower_wick'] = df[['open', 'close']].min(axis=1) - df['low']

        # Direction observation
        raw['bar_direction'] = np.sign(df['close'] - df['open'])

        # Time observations (if datetime index)
        if isinstance(df.index, pd.DatetimeIndex):
            raw['hour'] = df.index.hour
            raw['day_of_week'] = df.index.dayofweek
            raw['day_of_month'] = df.index.day
            raw['month'] = df.index.month

        return raw

    def _measure_derived(self, df: pd.DataFrame, raw: pd.DataFrame) -> pd.DataFrame:
        """
        Derived measurements - computed from raw observations.

        These are transformations, not new information.
        """
        derived = pd.DataFrame(index=df.index)

        # Returns (basic)
        derived['returns'] = df['close'].pct_change()
        derived['log_returns'] = np.log(df['close'] / df['close'].shift(1))

        # Returns (OHLC variations)
        derived['returns_hl'] = (df['high'] / df['low']) - 1  # High-low range
        derived['returns_oc'] = (df['close'] / df['open']) - 1  # Open-close

        # Volatility proxies (multiple methods)
        for window in [5, 10, 20, 50]:
            derived[f'volatility_{window}'] = derived['returns'].rolling(window).std()
            derived[f'volatility_hl_{window}'] = derived['returns_hl'].rolling(window).std()

        # Volume behavior
        derived['volume_change'] = df['volume'].pct_change()
        for window in [5, 10, 20]:
            derived[f'volume_ma_{window}'] = df['volume'].rolling(window).mean()
            derived[f'volume_std_{window}'] = df['volume'].rolling(window).std()

        # Range behavior
        for window in [5, 10, 20]:
            derived[f'atr_{window}'] = raw['bar_range'].rolling(window).mean()
            derived[f'range_std_{window}'] = raw['bar_range'].rolling(window).std()

        # Price levels (technical)
        for window in [10, 20, 50, 100, 200]:
            derived[f'sma_{window}'] = df['close'].rolling(window).mean()
            derived[f'ema_{window}'] = df['close'].ewm(span=window).mean()

        # Momentum (rate of change)
        for window in [5, 10, 20]:
            derived[f'roc_{window}'] = df['close'].pct_change(window)

        # Autocorrelation (memory)
        for lag in [1, 5, 10, 20]:
            derived[f'autocorr_{lag}'] = derived['returns'].rolling(50).apply(
                lambda x: x.autocorr(lag=lag) if len(x) > lag else np.nan,
                raw=False
            )

        return derived

    def _measure_physics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Physics-inspired measurements.

        HYPOTHESIS: Markets behave like physical systems.
        VALIDATION NEEDED: Do these actually predict anything?
        """
        from kinetra.physics_engine import PhysicsEngine

        # Compute full physics state
        engine = PhysicsEngine()
        physics = engine.compute_physics_state_from_ohlcv(df)

        # Return all physics measurements
        return physics

    def _compute_statistics(
        self,
        raw: pd.DataFrame,
        derived: pd.DataFrame,
        physics: pd.DataFrame
    ) -> Dict:
        """
        Statistical properties of measurements.

        Helps understand what each measurement captures.
        """
        stats = {}

        # Basic statistics
        stats['n_bars'] = len(raw)
        stats['timespan_days'] = (raw.index[-1] - raw.index[0]).total_seconds() / 86400

        # Price statistics
        stats['price_mean'] = raw['close'].mean()
        stats['price_std'] = raw['close'].std()
        stats['price_min'] = raw['close'].min()
        stats['price_max'] = raw['close'].max()

        # Returns statistics
        stats['returns_mean'] = derived['returns'].mean()
        stats['returns_std'] = derived['returns'].std()
        stats['returns_skew'] = derived['returns'].skew()
        stats['returns_kurtosis'] = derived['returns'].kurtosis()

        # Volume statistics
        stats['volume_mean'] = raw['volume'].mean()
        stats['volume_std'] = raw['volume'].std()

        # Range statistics
        stats['range_mean'] = raw['bar_range'].mean()
        stats['range_std'] = raw['bar_range'].std()

        # Physics statistics (test if they vary)
        if 'energy' in physics.columns:
            stats['energy_mean'] = physics['energy'].mean()
            stats['energy_std'] = physics['energy'].std()

        if 'damping' in physics.columns:
            stats['damping_mean'] = physics['damping'].mean()
            stats['damping_std'] = physics['damping'].std()

        if 'reynolds' in physics.columns:
            stats['reynolds_mean'] = physics['reynolds'].mean()
            stats['reynolds_std'] = physics['reynolds'].std()

        return stats

    def _analyze_fat_candles(
        self,
        df: pd.DataFrame,
        derived: pd.DataFrame,
        threshold_atr: float = 3.0
    ) -> pd.DataFrame:
        """
        Find and analyze "fat candles" (big moves).

        Question: What triggers these? Are triggers consistent?
        """
        # Calculate ATR
        atr = derived['atr_20']

        # Find fat candles (range > threshold * ATR)
        bar_range = df['high'] - df['low']
        fat_mask = bar_range > (threshold_atr * atr)

        fat_candles = pd.DataFrame({
            'timestamp': df.index[fat_mask],
            'open': df['open'][fat_mask],
            'high': df['high'][fat_mask],
            'low': df['low'][fat_mask],
            'close': df['close'][fat_mask],
            'range': bar_range[fat_mask],
            'atr': atr[fat_mask],
            'range_atr_ratio': (bar_range / atr)[fat_mask],
            'direction': np.sign(df['close'] - df['open'])[fat_mask],
        })

        return fat_candles

    def compare_measurements_across_classes(
        self,
        results: List[MeasurementResult]
    ) -> pd.DataFrame:
        """
        Compare measurements across asset classes.

        Question: Do forex and crypto have fundamentally different dynamics?
        """
        comparison = []

        for result in results:
            row = {
                'instrument': result.instrument,
                'timeframe': result.timeframe,
                'asset_class': result.asset_class,
            }

            # Add key statistics
            row.update(result.statistics)

            comparison.append(row)

        df = pd.DataFrame(comparison)

        if self.verbose:
            print("\n📊 Cross-Class Comparison:")
            print("\nReturns Statistics by Class:")
            print(df.groupby('asset_class')['returns_mean', 'returns_std'].mean())

            print("\nVolatility by Class:")
            print(df.groupby('asset_class')['returns_std'].describe())

        return df

    def test_predictive_power(
        self,
        result: MeasurementResult,
        lookahead_bars: int = 1
    ) -> pd.DataFrame:
        """
        Test which measurements actually predict future moves.

        CRITICAL: This is how we validate our hypotheses.
        """
        from scipy.stats import pearsonr, spearmanr

        # Future returns (what we want to predict)
        future_returns = result.derived_measurements['returns'].shift(-lookahead_bars)

        predictive_power = []

        # Test all measurements
        all_measurements = pd.concat([
            result.raw_measurements,
            result.derived_measurements,
            result.physics_measurements,
        ], axis=1)

        for col in all_measurements.columns:
            if col == 'returns':  # Skip target variable
                continue

            # Drop NaN for correlation
            valid_mask = ~(all_measurements[col].isna() | future_returns.isna())

            if valid_mask.sum() < 100:  # Need enough data
                continue

            x = all_measurements[col][valid_mask]
            y = future_returns[valid_mask]

            # Compute correlations
            try:
                pearson_corr, pearson_p = pearsonr(x, y)
                spearman_corr, spearman_p = spearmanr(x, y)

                predictive_power.append({
                    'measurement': col,
                    'pearson_corr': pearson_corr,
                    'pearson_pvalue': pearson_p,
                    'spearman_corr': spearman_corr,
                    'spearman_pvalue': spearman_p,
                    'abs_pearson': abs(pearson_corr),
                    'abs_spearman': abs(spearman_corr),
                    'significant': pearson_p < 0.05,
                })
            except Exception:
                continue

        df = pd.DataFrame(predictive_power)
        df = df.sort_values('abs_pearson', ascending=False)

        if self.verbose:
            print(f"\n🔮 Predictive Power (lookahead={lookahead_bars} bars):")
            print("\nTop 10 measurements (by absolute correlation):")
            print(df.head(10)[['measurement', 'pearson_corr', 'pearson_pvalue']])

            # Count how many physics measurements are in top 20
            top_20 = df.head(20)
            physics_cols = [col for col in top_20['measurement'] if any(
                x in col for x in ['energy', 'damping', 'reynolds', 'entropy', 'potential']
            )]
            print(f"\nPhysics measurements in top 20: {len(physics_cols)}/20")
            if physics_cols:
                print("  -", ", ".join(physics_cols))

        return df


def main():
    """Run measurement toolkit on all instruments."""
    print("=" * 80)
    print("MEASUREMENT TOOLKIT - First Principles Research")
    print("=" * 80)

    # Load data
    from kinetra.data_loader import UnifiedDataLoader

    data_dir = Path("data/prepared/train")

    if not data_dir.exists():
        print(f"\n❌ Data directory not found: {data_dir}")
        print("   Run: python scripts/prepare_data.py")
        return

    # Initialize toolkit
    toolkit = MeasurementToolkit(verbose=True)

    # Measure all instruments
    results = []

    csv_files = list(data_dir.glob("*.csv"))[:10]  # Start with first 10

    print(f"\n🔬 Measuring {len(csv_files)} instruments...")

    for csv_file in csv_files:
        # Parse filename
        parts = csv_file.stem.split('_')
        instrument = parts[0]
        timeframe = parts[1]

        # Classify asset class
        if any(x in instrument.upper() for x in ['BTC', 'ETH', 'XRP']):
            asset_class = 'crypto'
        elif any(x in instrument.upper() for x in ['XAU', 'XAG', 'XPT']):
            asset_class = 'metals'
        elif any(x in instrument.upper() for x in ['NAS', 'SPX', 'DJ', 'DAX', 'FTSE']):
            asset_class = 'indices'
        elif len(instrument.replace('+', '').replace('-', '')) == 6:
            asset_class = 'forex'
        else:
            asset_class = 'unknown'

        # Load data
        loader = UnifiedDataLoader(verbose=False)
        pkg = loader.load(str(csv_file))
        df = pkg.to_backtest_engine_format()

        # Measure
        result = toolkit.measure_instrument(df, instrument, timeframe, asset_class)
        results.append(result)

        # Test predictive power for first instrument
        if len(results) == 1:
            print("\n" + "=" * 80)
            print("PREDICTIVE POWER ANALYSIS")
            print("=" * 80)
            toolkit.test_predictive_power(result, lookahead_bars=1)

    # Cross-class comparison
    print("\n" + "=" * 80)
    print("CROSS-CLASS COMPARISON")
    print("=" * 80)
    comparison = toolkit.compare_measurements_across_classes(results)

    # Save results
    output_dir = Path("research_output/measurements")
    output_dir.mkdir(parents=True, exist_ok=True)

    comparison.to_csv(output_dir / "cross_class_comparison.csv", index=False)
    print(f"\n💾 Saved: {output_dir / 'cross_class_comparison.csv'}")

    print("\n✅ Measurement analysis complete!")


if __name__ == "__main__":
    main()
