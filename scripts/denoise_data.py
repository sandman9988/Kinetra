#!/usr/bin/env python3
"""
Denoise Financial Data Script
==============================

Apply non-linear denoising filters to market data.

Physics-first approach:
- NO linear filters (MA/EMA) - they destroy non-linear dynamics
- Preserves sharp trends, regime changes, critical moves
- Removes high-frequency noise only
- Vectorized operations (NO Python loops)

Usage:
    # Denoise all files in data/prepared/ using Savitzky-Golay
    python scripts/denoise_data.py

    # Denoise specific files with custom method
    python scripts/denoise_data.py --method median --input data/master/BTCUSD_H1.csv

    # Process directory
    python scripts/denoise_data.py --input-dir data/prepared/ --method savgol

__version__ = "1.0.0"
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd  # type: ignore[import-untyped]
from tqdm import tqdm  # type: ignore[import-untyped]

# Project root - must be set before kinetra imports
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from kinetra.denoise_filters import DenoiseMethod, denoise_ohlc  # noqa: E402

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def denoise_file(filepath: Path, method: DenoiseMethod, output_dir: Path) -> bool:
    """
    Denoise a single CSV file.

    Args:
        filepath: Input CSV file
        method: Denoising method
        output_dir: Output directory

    Returns:
        True if successful
    """
    try:
        logger.info(f"Processing: {filepath.name}")

        # Load data
        df = pd.read_csv(filepath)

        # Normalize column names
        df.columns = df.columns.str.lower().str.strip()

        # Check required columns
        required = ["close"]
        if not all(col in df.columns for col in required):
            logger.warning(f"Skipping {filepath.name} - missing required columns")
            return False

        # Denoise all OHLC columns
        df_denoised = denoise_ohlc(df, method=method)

        # Save denoised data
        output_file = output_dir / f"{filepath.stem}_denoised.csv"
        df_denoised.to_csv(output_file, index=False)

        logger.info(f"✅ Saved: {output_file.name}")

        # Print metrics
        if "close_denoised" in df_denoised.columns:
            original_vol = df["close"].pct_change().std()
            denoised_vol = df_denoised["close_denoised"].pct_change().std()
            reduction = (1 - denoised_vol / original_vol) * 100

            logger.info(f"   Volatility reduction: {reduction:.1f}%")

        return True

    except Exception as e:
        logger.error(f"Failed to process {filepath.name}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Denoise financial data")
    parser.add_argument(
        "--method",
        choices=["savgol", "median", "lowess", "wavelet"],
        default="savgol",
        help="Denoising method (default: savgol)",
    )
    parser.add_argument(
        "--input",
        type=str,
        help="Input CSV file (if not specified, processes all in input-dir)",
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default="data/prepared",
        help="Input directory (default: data/prepared)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/prepared/denoised",
        help="Output directory (default: data/prepared/denoised)",
    )

    args = parser.parse_args()

    # Map string to enum
    method_map = {
        "savgol": DenoiseMethod.SAVGOL,
        "median": DenoiseMethod.MEDIAN,
        "lowess": DenoiseMethod.LOWESS,
        "wavelet": DenoiseMethod.WAVELET,
    }
    method = method_map[args.method]

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"🔬 Denoising method: {args.method.upper()}")
    logger.info(f"📁 Output directory: {output_dir}")

    # Process files
    if args.input:
        # Single file
        filepath = Path(args.input)
        if not filepath.exists():
            logger.error(f"File not found: {filepath}")
            return 1

        success = denoise_file(filepath, method, output_dir)
        return 0 if success else 1

    else:
        # Process directory
        input_dir = Path(args.input_dir)
        if not input_dir.exists():
            logger.error(f"Directory not found: {input_dir}")
            return 1

        csv_files = list(input_dir.glob("*.csv"))
        # Exclude already denoised files
        csv_files = [f for f in csv_files if "_denoised" not in f.name]

        if not csv_files:
            logger.warning(f"No CSV files found in {input_dir}")
            return 1

        logger.info(f"Found {len(csv_files)} files to process")

        success_count = 0
        for filepath in tqdm(csv_files, desc="Denoising files"):
            if denoise_file(filepath, method, output_dir):
                success_count += 1

        logger.info(f"\n✅ Successfully processed {success_count}/{len(csv_files)} files")
        return 0 if success_count == len(csv_files) else 1


if __name__ == "__main__":
    sys.exit(main())
