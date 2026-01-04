#!/usr/bin/env python3
"""
Noise Quality Test Suite
=========================

Compare denoising performance across instrument groups.

Tests:
1. Signal-to-Noise Ratio (SNR) improvement
2. Information preservation (no oversmoothing)
3. Regime transition preservation
4. Cross-instrument performance
5. Method comparison (Savitzky-Golay vs Median vs Wavelet)

Version: 1.0.0
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import pytest
from scipy.signal import welch  # type: ignore[import-untyped]

from kinetra.denoise_filters import DenoiseMethod, denoise_ohlc

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Asset classification (from canonical rules)
ASSET_GROUPS = {
    "crypto": ["BTCUSD", "ETHUSD", "BNBUSD", "ADAUSD", "SOLUSD"],
    "forex_major": ["EURUSD", "GBPUSD", "USDJPY", "USDCHF", "AUDUSD", "NZDUSD", "USDCAD"],
    "forex_minor": ["EURJPY", "GBPJPY", "EURGBP", "AUDNZD", "EURAUD"],
    "metals": ["XAUUSD", "XAGUSD", "XPTUSD", "XPDUSD"],
    "indices": ["US500", "US100", "US30", "DE40", "UK100", "JP225"],
    "commodities": ["USOIL", "UKOIL", "NATGAS"],
}


class NoiseQualityMetrics:
    """Calculate noise quality metrics for denoised data."""

    @staticmethod
    def signal_to_noise_ratio(original: np.ndarray, denoised: np.ndarray) -> float:
        """
        Calculate SNR improvement.

        SNR = 10 * log10(signal_power / noise_power)

        Higher is better - means more noise removed while preserving signal.
        """
        # Noise = original - denoised
        noise = original - denoised

        # Power spectral density
        signal_power = np.var(denoised)
        noise_power = np.var(noise)

        if noise_power == 0:
            return float('inf')

        snr = 10 * np.log10(signal_power / noise_power)
        return float(snr)

    @staticmethod
    def mean_squared_error(original: np.ndarray, denoised: np.ndarray) -> float:
        """
        MSE between original and denoised.

        Lower is better - but TOO low means oversmoothing.
        """
        return float(np.mean((original - denoised) ** 2))

    @staticmethod
    def spectral_preservation(original: np.ndarray, denoised: np.ndarray) -> float:
        """
        Measure how well low-frequency components are preserved.

        Uses Welch's method to compute power spectral density.

        Returns:
            Correlation between original and denoised spectra (0-1)
            Higher is better - means trend structure preserved
        """
        # Compute power spectral density
        freqs_orig, psd_orig = welch(original, nperseg=min(256, len(original) // 4))
        freqs_den, psd_den = welch(denoised, nperseg=min(256, len(denoised) // 4))

        # Correlation of low-frequency components (< 0.1 normalized frequency)
        low_freq_mask = freqs_orig < 0.1
        if not np.any(low_freq_mask):
            return 1.0

        corr = np.corrcoef(psd_orig[low_freq_mask], psd_den[low_freq_mask])[0, 1]
        return float(corr)

    @staticmethod
    def regime_preservation(original: np.ndarray, denoised: np.ndarray, threshold: float = 2.0) -> float:
        """
        Check if regime changes (large moves) are preserved.

        Regime changes = returns > threshold * std

        Returns:
            Fraction of regime changes preserved (0-1)
        """
        # Detect regime changes in original
        original_returns = np.diff(original)
        std_orig = np.std(original_returns)

        regime_changes_orig = np.abs(original_returns) > (threshold * std_orig)

        if not np.any(regime_changes_orig):
            return 1.0  # No regime changes to preserve

        # Check if denoised preserves these
        denoised_returns = np.diff(denoised)
        std_den = np.std(denoised_returns)

        regime_changes_den = np.abs(denoised_returns) > (threshold * std_den)

        # Fraction of original regime changes still present
        preserved = np.logical_and(regime_changes_orig, regime_changes_den).sum()
        total = regime_changes_orig.sum()

        return float(preserved / total if total > 0 else 1.0)

    @staticmethod
    def smoothness_metric(denoised: np.ndarray) -> float:
        """
        Measure smoothness via second derivative (jerk).

        Lower is smoother.
        """
        jerk = np.diff(denoised, n=2)
        return float(np.std(jerk))


class TestNoiseQuality:
    """Test denoising quality across instruments and methods."""

    @pytest.fixture
    def sample_data_dir(self) -> Path:
        """Get sample data directory."""
        data_dir = Path("data/prepared")
        if not data_dir.exists():
            data_dir = Path("data/master")
        return data_dir

    def load_instrument_data(self, instrument: str, data_dir: Path) -> Optional[pd.DataFrame]:
        """Load instrument data if available."""
        # Try H1 first, then D1
        for timeframe in ["H1", "D1"]:
            filepath = data_dir / f"{instrument}_{timeframe}.csv"
            if filepath.exists():
                try:
                    df = pd.read_csv(filepath, parse_dates=["timestamp"])
                    if len(df) >= 1000:  # Need sufficient data
                        return df
                except Exception as e:
                    logger.warning(f"Failed to load {filepath}: {e}")
        return None

    def test_snr_improvement_by_asset_group(self, sample_data_dir: Path) -> None:
        """Test SNR improvement across asset groups."""
        results: Dict[str, List[float]] = {}

        for group_name, instruments in ASSET_GROUPS.items():
            snr_values = []

            for instrument in instruments:
                df = self.load_instrument_data(instrument, sample_data_dir)
                if df is None:
                    continue

                # Test on close prices
                original = df["close"].values

                # Apply Savitzky-Golay (default method)
                denoised = denoise_ohlc(
                    df[["open", "high", "low", "close", "volume"]].values,
                    method=DenoiseMethod.SAVGOL
                )[:, 3]  # Close column

                # Calculate SNR
                snr = NoiseQualityMetrics.signal_to_noise_ratio(original, denoised)
                snr_values.append(snr)

                logger.info(f"{instrument}: SNR improvement = {snr:.2f} dB")

            if snr_values:
                results[group_name] = snr_values
                avg_snr = np.mean(snr_values)
                logger.info(f"{group_name} average SNR: {avg_snr:.2f} dB")

                # Assert reasonable SNR improvement
                assert avg_snr > 5.0, f"{group_name} SNR too low: {avg_snr:.2f} dB"

        # Compare groups
        if len(results) > 1:
            logger.info("\n=== SNR Comparison Across Asset Groups ===")
            for group in sorted(results.keys(), key=lambda g: float(np.mean(results[g])), reverse=True):
                logger.info(f"{group:15s}: {np.mean(results[group]):6.2f} dB (±{np.std(results[group]):5.2f})")

    def test_method_comparison(self, sample_data_dir: Path) -> None:
        """Compare denoising methods on representative instruments."""
        methods = [DenoiseMethod.SAVGOL, DenoiseMethod.MEDIAN, DenoiseMethod.WAVELET]

        # Test on one instrument from each group
        test_instruments = {
            "BTCUSD": "crypto",
            "EURUSD": "forex_major",
            "XAUUSD": "metals"
        }

        comparison_results: Dict[str, Dict[str, Dict[str, float]]] = {}

        for instrument, group in test_instruments.items():
            df = self.load_instrument_data(instrument, sample_data_dir)
            if df is None:
                logger.warning(f"Skipping {instrument} - no data")
                continue

            original = df["close"].values
            comparison_results[instrument] = {}

            for method in methods:
                denoised = denoise_ohlc(
                    df[["open", "high", "low", "close", "volume"]].values,
                    method=method
                )[:, 3]

                metrics = {
                    "snr": NoiseQualityMetrics.signal_to_noise_ratio(original, denoised),
                    "mse": NoiseQualityMetrics.mean_squared_error(original, denoised),
                    "spectral": NoiseQualityMetrics.spectral_preservation(original, denoised),
                    "regime": NoiseQualityMetrics.regime_preservation(original, denoised),
                    "smoothness": NoiseQualityMetrics.smoothness_metric(denoised),
                }

                comparison_results[instrument][method.value] = metrics

                logger.info(f"\n{instrument} - {method.value}:")
                logger.info(f"  SNR:      {metrics['snr']:.2f} dB")
                logger.info(f"  MSE:      {metrics['mse']:.6f}")
                logger.info(f"  Spectral: {metrics['spectral']:.3f}")
                logger.info(f"  Regime:   {metrics['regime']:.1%}")
                logger.info(f"  Smooth:   {metrics['smoothness']:.6f}")

        # Print summary
        if comparison_results:
            logger.info("\n=== Method Comparison Summary ===")
            for instrument in comparison_results:
                logger.info(f"\n{instrument}:")
                for method in methods:
                    method_key = method.value
                    if method_key in comparison_results[instrument]:
                        m = comparison_results[instrument][method_key]
                        logger.info(f"  {method_key:10s}: SNR={m['snr']:6.2f} dB, "
                                  f"Spectral={m['spectral']:.3f}, "
                                  f"Regime={m['regime']:.1%}")

    def test_regime_preservation_critical(self, sample_data_dir: Path) -> None:
        """
        CRITICAL: Ensure denoising doesn't destroy regime transitions.

        This is crucial for Kinetra's physics-based approach.
        """
        # Test on volatile instruments where regime changes are common
        volatile_instruments = ["BTCUSD", "ETHUSD", "XAUUSD"]

        for instrument in volatile_instruments:
            df = self.load_instrument_data(instrument, sample_data_dir)
            if df is None:
                continue

            original = df["close"].values

            for method in [DenoiseMethod.SAVGOL, DenoiseMethod.MEDIAN]:
                denoised = denoise_ohlc(
                    df[["open", "high", "low", "close", "volume"]].values,
                    method=method
                )[:, 3]

                regime_preserved = NoiseQualityMetrics.regime_preservation(original, denoised)

                logger.info(f"{instrument} - {method.value}: "
                          f"Regime preservation = {regime_preserved:.1%}")

                # CRITICAL ASSERTION: Must preserve >80% of regime changes
                assert regime_preserved > 0.80, (
                    f"{instrument} with {method.value} loses too many regime changes: "
                    f"{regime_preserved:.1%} < 80%"
                )

    def test_oversmoothing_detection(self, sample_data_dir: Path) -> None:
        """
        Detect oversmoothing (too aggressive denoising).

        Oversmoothing = spectral preservation < 0.95 OR regime preservation < 0.80
        """
        test_instruments = ["EURUSD", "BTCUSD", "XAUUSD"]

        for instrument in test_instruments:
            df = self.load_instrument_data(instrument, sample_data_dir)
            if df is None:
                continue

            original = df["close"].values

            for method in [DenoiseMethod.SAVGOL, DenoiseMethod.MEDIAN]:
                denoised = denoise_ohlc(
                    df[["open", "high", "low", "close", "volume"]].values,
                    method=method
                )[:, 3]

                spectral = NoiseQualityMetrics.spectral_preservation(original, denoised)
                regime = NoiseQualityMetrics.regime_preservation(original, denoised)

                logger.info(f"{instrument} - {method.value}: "
                          f"Spectral={spectral:.3f}, Regime={regime:.1%}")

                # Check for oversmoothing
                if spectral < 0.95:
                    logger.warning(f"⚠️  {instrument}/{method.value}: "
                                 f"Low spectral preservation ({spectral:.3f})")

                if regime < 0.80:
                    logger.warning(f"⚠️  {instrument}/{method.value}: "
                                 f"Low regime preservation ({regime:.1%})")

                # Should not be BOTH low
                assert not (spectral < 0.95 and regime < 0.80), (
                    f"{instrument}/{method.value} is oversmoothing: "
                    f"spectral={spectral:.3f}, regime={regime:.1%}"
                )

    def test_noise_characteristics_by_group(self, sample_data_dir: Path) -> None:
        """
        Analyze raw noise characteristics across asset groups.

        Helps identify which groups benefit most from denoising.
        """
        noise_stats: Dict[str, List[Dict[str, float]]] = {}

        for group_name, instruments in ASSET_GROUPS.items():
            group_stats = []

            for instrument in instruments[:3]:  # Limit to 3 per group for speed
                df = self.load_instrument_data(instrument, sample_data_dir)
                if df is None:
                    continue

                # Calculate noise characteristics
                returns = df["close"].pct_change().dropna()
                volatility = returns.std()

                # High-frequency noise (2nd derivative)
                jerk = np.diff(df["close"].values, n=2)
                jerk_std = np.std(jerk)

                # Autocorrelation at lag 1 (noise should have low autocorr)
                autocorr = returns.autocorr(lag=1)

                stats_dict = {
                    "volatility": volatility,
                    "jerk_std": jerk_std,
                    "autocorr": autocorr,
                }
                group_stats.append(stats_dict)

                logger.info(f"{instrument}: vol={volatility:.6f}, "
                          f"jerk={jerk_std:.6f}, autocorr={autocorr:.3f}")

            if group_stats:
                noise_stats[group_name] = group_stats

                # Group averages
                avg_vol = np.mean([s["volatility"] for s in group_stats])
                avg_jerk = np.mean([s["jerk_std"] for s in group_stats])
                avg_autocorr = np.mean([s["autocorr"] for s in group_stats])

                logger.info(f"{group_name} averages: "
                          f"vol={avg_vol:.6f}, jerk={avg_jerk:.6f}, autocorr={avg_autocorr:.3f}")

        # Identify noisiest groups
        if noise_stats:
            logger.info("\n=== Noisiest Asset Groups (by jerk) ===")
            sorted_groups = sorted(
                noise_stats.items(),
                key=lambda x: float(np.mean([s["jerk_std"] for s in x[1]])),
                reverse=True
            )
            for group_name, stats_list in sorted_groups:
                avg_jerk = np.mean([s["jerk_std"] for s in stats_list])
                logger.info(f"{group_name:15s}: {avg_jerk:.6f}")


def generate_noise_quality_report(data_dir: Path = Path("data/prepared")) -> Dict[str, Any]:
    """
    Generate comprehensive noise quality report.

    Returns:
        Dictionary with all metrics across instruments and methods
    """
    report: Dict[str, Any] = {
        "asset_groups": {},
        "method_comparison": {},
        "recommendations": []
    }

    metrics = NoiseQualityMetrics()

    # Test each asset group
    for group_name, instruments in ASSET_GROUPS.items():
        group_results = []

        for instrument in instruments:
            # Try to load data
            for timeframe in ["H1", "D1"]:
                filepath = data_dir / f"{instrument}_{timeframe}.csv"
                if not filepath.exists():
                    continue

                try:
                    df = pd.read_csv(filepath, parse_dates=["timestamp"])
                    if len(df) < 1000:
                        continue

                    original = df["close"].values

                    # Test all methods
                    for method in [DenoiseMethod.SAVGOL, DenoiseMethod.MEDIAN, DenoiseMethod.WAVELET]:
                        denoised = denoise_ohlc(
                            df[["open", "high", "low", "close", "volume"]].values,
                            method=method
                        )[:, 3]

                        result = {
                            "instrument": instrument,
                            "timeframe": timeframe,
                            "method": method.value,
                            "snr": metrics.signal_to_noise_ratio(original, denoised),
                            "spectral": metrics.spectral_preservation(original, denoised),
                            "regime": metrics.regime_preservation(original, denoised),
                            "smoothness": metrics.smoothness_metric(denoised),
                        }
                        group_results.append(result)

                    break  # Only process first available timeframe

                except Exception as e:
                    logger.warning(f"Error processing {filepath}: {e}")
                    continue

        report["asset_groups"][group_name] = group_results

    # Recommendations based on analysis
    for group_name, results in report["asset_groups"].items():
        if not results:
            continue

        # Find best method for this group
        method_scores: Dict[str, List[float]] = {}
        for result in results:
            method_name = str(result["method"])
            score = float(result["snr"] + result["spectral"] * 10 + result["regime"] * 10)
            method_scores[method_name] = method_scores.get(method_name, []) + [score]

        best_method = max(method_scores.items(), key=lambda x: np.mean(x[1]))[0]
        report["recommendations"].append({
            "asset_group": group_name,
            "best_method": best_method,
            "avg_score": np.mean(method_scores[best_method])
        })

    return report


if __name__ == "__main__":
    # Run noise quality analysis
    print("=" * 80)
    print("NOISE QUALITY ANALYSIS")
    print("=" * 80)

    report = generate_noise_quality_report()

    print("\n=== RECOMMENDATIONS BY ASSET GROUP ===")
    for rec in report["recommendations"]:
        print(f"{rec['asset_group']:15s}: {rec['best_method']:10s} (score: {rec['avg_score']:.2f})")

    print("\n✅ Run with pytest for comprehensive validation:")
    print("   pytest tests/test_noise_quality.py -v -s")
