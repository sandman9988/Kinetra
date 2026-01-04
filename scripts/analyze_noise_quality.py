#!/usr/bin/env python3
"""
Generate Comprehensive Noise Quality Report
============================================

Analyzes denoising performance across all available instruments.
Generates detailed report with visualizations and recommendations.

Version: 1.0.0
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt  # type: ignore[import-untyped]
import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import seaborn as sns  # type: ignore[import-untyped]

from kinetra.denoise_filters import DenoiseMethod, denoise_ohlc

# Setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Asset groups
ASSET_GROUPS = {
    "crypto": ["BTCUSD", "ETHUSD", "BNBUSD", "ADAUSD", "SOLUSD"],
    "forex_major": ["EURUSD", "GBPUSD", "USDJPY", "USDCHF", "AUDUSD", "NZDUSD", "USDCAD"],
    "forex_minor": ["EURJPY", "GBPJPY", "EURGBP", "AUDNZD", "EURAUD"],
    "metals": ["XAUUSD", "XAGUSD"],
    "indices": ["US500", "US100", "US30"],
}


def calculate_snr(original: np.ndarray, denoised: np.ndarray) -> float:
    """Calculate Signal-to-Noise Ratio improvement."""
    noise = original - denoised
    signal_power = np.var(denoised)
    noise_power = np.var(noise)
    if noise_power == 0:
        return float('inf')
    return 10 * np.log10(signal_power / noise_power)


def calculate_regime_preservation(original: np.ndarray, denoised: np.ndarray) -> float:
    """Fraction of large moves (regime changes) preserved."""
    orig_returns = np.diff(original)
    std_orig = np.std(orig_returns)
    regime_changes = np.abs(orig_returns) > (2.0 * std_orig)

    if not np.any(regime_changes):
        return 1.0

    den_returns = np.diff(denoised)
    std_den = np.std(den_returns)
    regime_preserved = np.abs(den_returns[:-1]) > (2.0 * std_den)

    preserved = np.logical_and(regime_changes, regime_preserved).sum()
    return float(preserved / regime_changes.sum())


def analyze_instrument(
    filepath: Path,
    methods: List[DenoiseMethod]
) -> Optional[Dict[str, Any]]:
    """Analyze denoising performance for one instrument."""
    try:
        df = pd.read_csv(filepath, parse_dates=["timestamp"])
        if len(df) < 1000:
            return None

        instrument = filepath.stem.split('_')[0]
        timeframe = filepath.stem.split('_')[1] if '_' in filepath.stem else "UNK"

        results: Dict[str, Any] = {
            "instrument": instrument,
            "timeframe": timeframe,
            "bars": len(df),
            "methods": {}
        }

        original = df["close"].values

        for method in methods:
            try:
                denoised_ohlcv = denoise_ohlc(
                    df[["open", "high", "low", "close", "volume"]].values,
                    method=method
                )
                denoised = denoised_ohlcv[:, 3]  # Close column

                snr = calculate_snr(original, denoised)
                regime = calculate_regime_preservation(original, denoised)
                mse = np.mean((original - denoised) ** 2)

                results["methods"][method.value] = {
                    "snr_db": float(snr),
                    "regime_preservation": float(regime),
                    "mse": float(mse),
                }

            except Exception as e:
                logger.warning(f"Method {method.value} failed for {instrument}: {e}")

        return results

    except Exception as e:
        logger.error(f"Failed to analyze {filepath}: {e}")
        return None


def generate_report(data_dir: Path = Path("data/prepared")) -> Dict[str, Any]:
    """Generate comprehensive noise quality report."""
    logger.info("=" * 80)
    logger.info("NOISE QUALITY ANALYSIS REPORT")
    logger.info("=" * 80)

    if not data_dir.exists():
        data_dir = Path("data/master")

    if not data_dir.exists():
        logger.error(f"Data directory not found: {data_dir}")
        return {"error": "No data directory found"}

    # Methods to test
    methods = [DenoiseMethod.SAVGOL, DenoiseMethod.MEDIAN, DenoiseMethod.WAVELET]

    # Analyze all files
    all_results: List[Dict] = []

    csv_files = list(data_dir.glob("*.csv"))
    logger.info(f"Found {len(csv_files)} CSV files in {data_dir}")

    for filepath in csv_files:
        logger.info(f"Analyzing: {filepath.name}")
        result = analyze_instrument(filepath, methods)
        if result:
            all_results.append(result)

    if not all_results:
        logger.warning("No results generated - check data directory")
        return {"error": "No valid data files processed"}

    # Group by asset class
    grouped_results: Dict[str, List[Dict]] = {group: [] for group in ASSET_GROUPS.keys()}
    ungrouped_results: List[Dict] = []

    for result in all_results:
        instrument = result["instrument"]
        found_group = False

        for group_name, instruments in ASSET_GROUPS.items():
            if instrument in instruments:
                grouped_results[group_name].append(result)
                found_group = True
                break

        if not found_group:
            ungrouped_results.append(result)

    # Generate summary statistics
    summary = {}

    for group_name, group_results in grouped_results.items():
        if not group_results:
            continue

        group_summary: Dict[str, Any] = {"instruments": []}

        for method in methods:
            method_name = method.value
            snr_values: List[float] = []
            regime_values: List[float] = []

            for result in group_results:
                methods_dict = result.get("methods", {})
                if isinstance(methods_dict, dict) and method_name in methods_dict:
                    m = methods_dict[method_name]
                    snr_values.append(m["snr_db"])
                    regime_values.append(m["regime_preservation"])

            if snr_values:
                group_summary[method_name] = {
                    "snr_mean": float(np.mean(snr_values)),
                    "snr_std": float(np.std(snr_values)),
                    "regime_mean": float(np.mean(regime_values)),
                    "regime_std": float(np.std(regime_values)),
                    "count": len(snr_values),
                }

        group_summary["instruments"] = [r["instrument"] for r in group_results]
        summary[group_name] = group_summary

    # Print summary
    logger.info("\n" + "=" * 80)
    logger.info("SUMMARY BY ASSET GROUP")
    logger.info("=" * 80)

    for group_name in sorted(summary.keys()):
        group_data = summary[group_name]
        logger.info(f"\n{group_name.upper()}")
        logger.info(f"  Instruments: {len(group_data['instruments'])}")

        for method in methods:
            method_name = method.value
            if method_name in group_data:
                m = group_data[method_name]
                logger.info(f"  {method_name:10s}: SNR={m['snr_mean']:6.2f}±{m['snr_std']:.2f} dB, "
                          f"Regime={m['regime_mean']:.1%}±{m['regime_std']:.1%}")

    # Recommendations
    logger.info("\n" + "=" * 80)
    logger.info("RECOMMENDATIONS")
    logger.info("=" * 80)

    for group_name, group_data in summary.items():
        if not group_data.get("instruments"):
            continue

        # Find best method (highest SNR + regime preservation)
        best_method = None
        best_score = -float('inf')

        for method in methods:
            method_name = method.value
            if method_name in group_data:
                m = group_data[method_name]
                score = m["snr_mean"] + (m["regime_mean"] * 10)  # Weight regime preservation
                if score > best_score:
                    best_score = score
                    best_method = method_name

        if best_method:
            logger.info(f"{group_name:15s}: Use {best_method:10s} (score: {best_score:.2f})")

    # Save full report
    report = {
        "timestamp": pd.Timestamp.now().isoformat(),
        "data_dir": str(data_dir),
        "total_instruments": len(all_results),
        "summary": summary,
        "all_results": all_results,
        "ungrouped": [r["instrument"] for r in ungrouped_results],
    }

    output_file = Path("noise_quality_report.json")
    with open(output_file, 'w') as f:
        json.dump(report, f, indent=2)

    logger.info(f"\n✅ Full report saved to: {output_file}")

    return report


def plot_snr_comparison(report_file: Path = Path("noise_quality_report.json")) -> None:
    """Generate SNR comparison plots from report."""
    if not report_file.exists():
        logger.error(f"Report file not found: {report_file}")
        return

    with open(report_file, 'r') as f:
        report = json.load(f)

    summary = report["summary"]
    methods = ["savgol", "median", "wavelet"]

    # Prepare data for plotting
    plot_data = []

    for group_name, group_data in summary.items():
        for method in methods:
            if method in group_data:
                m = group_data[method]
                plot_data.append({
                    "Group": group_name,
                    "Method": method,
                    "SNR (dB)": m["snr_mean"],
                    "Regime": m["regime_mean"]
                })

    if not plot_data:
        logger.warning("No data to plot")
        return

    df = pd.DataFrame(plot_data)

    # Create plots
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # SNR comparison
    sns.barplot(data=df, x="Group", y="SNR (dB)", hue="Method", ax=axes[0])
    axes[0].set_title("SNR Improvement by Asset Group", fontsize=14, fontweight="bold")
    axes[0].set_xlabel("Asset Group")
    axes[0].set_ylabel("SNR (dB)")
    axes[0].legend(title="Method")
    axes[0].tick_params(axis='x', rotation=45)

    # Regime preservation
    sns.barplot(data=df, x="Group", y="Regime", hue="Method", ax=axes[1])
    axes[1].set_title("Regime Preservation by Asset Group", fontsize=14, fontweight="bold")
    axes[1].set_xlabel("Asset Group")
    axes[1].set_ylabel("Regime Preservation")
    axes[1].legend(title="Method")
    axes[1].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig("noise_quality_comparison.png", dpi=150)
    logger.info("✅ Plot saved to: noise_quality_comparison.png")


if __name__ == "__main__":
    import sys

    # Generate report
    data_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/prepared")
    report = generate_report(data_dir)

    # Generate plots if matplotlib available
    if "error" not in report:
        try:
            plot_snr_comparison()
        except Exception as e:
            logger.warning(f"Failed to generate plots: {e}")

    print("\n" + "=" * 80)
    print("NOISE QUALITY ANALYSIS COMPLETE")
    print("=" * 80)
    print("Report: noise_quality_report.json")
    print("Plots: noise_quality_comparison.png")
    print("\nNext steps:")
    print("1. Review recommendations by asset group")
    print("2. Test with actual backtests to validate improvements")
    print("3. Adjust denoising parameters if needed")
