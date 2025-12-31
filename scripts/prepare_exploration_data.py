"""
Prepare Downloaded Data for Exploration Training
==================================================

Takes raw downloaded data and prepares it for multi-instrument RL exploration:
1. Organize by asset class (forex/crypto/commodities)
2. Generate physics features (energy, entropy, damping)
3. Create standardized training datasets
4. Save metadata for each asset

Usage:
    python scripts/prepare_exploration_data.py
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import List, Dict
import json

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from kinetra.physics_engine import PhysicsEngine
from kinetra.data_manager import DataManager


def detect_asset_class(symbol: str) -> str:
    """Detect asset class from symbol name."""
    symbol_clean = symbol.replace('+', '').replace('-', '').upper()

    # Forex pairs (6 characters, all letters)
    if len(symbol_clean) == 6 and symbol_clean.isalpha():
        return "forex"

    # Crypto
    if 'BTC' in symbol_clean or 'ETH' in symbol_clean or 'CRYPTO' in symbol_clean:
        return "crypto"

    # Metals
    if symbol_clean.startswith(('XAU', 'XAG', 'GOLD', 'SILVER', 'XPTUSD', 'XPDUSD')):
        return "metals"

    # Commodities
    if any(x in symbol_clean for x in ['COPPER', 'OIL', 'WTI', 'BRENT', 'GAS']):
        return "commodities"

    # Indices
    if any(x in symbol_clean for x in ['SPX', 'NAS', 'DOW', 'DAX', 'FTSE', 'NIKKEI']):
        return "indices"

    return "unknown"


def prepare_physics_features(data: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """
    Generate physics-based features for exploration.

    Args:
        data: OHLCV DataFrame
        symbol: Asset symbol

    Returns:
        DataFrame with physics features added
    """
    print(f"  Generating physics features for {symbol}...")

    # Initialize physics engine
    engine = PhysicsEngine(lookback=20)

    # Compute physics state
    physics = engine.compute_physics_state(data['close'], include_percentiles=True)

    # Add to dataframe
    result = data.copy()
    result['energy'] = physics['energy']
    result['entropy'] = physics['entropy']
    result['damping'] = physics['damping']
    result['regime'] = physics['regime']
    result['regime_confidence'] = physics['regime_confidence']

    # Add percentile-based features
    result['energy_percentile'] = physics['energy_percentile']
    result['entropy_percentile'] = physics['entropy_percentile']

    # Add price-based features
    result['returns'] = data['close'].pct_change()
    result['log_returns'] = np.log(data['close'] / data['close'].shift(1))
    result['volatility'] = result['returns'].rolling(20).std()

    # Add volume features if available
    if 'volume' in data.columns:
        result['volume_change'] = data['volume'].pct_change()
        result['volume_ma'] = data['volume'].rolling(20).mean()

    # Drop NaN rows
    result = result.dropna()

    print(f"    Generated {len(result)} bars with physics features")

    return result


def prepare_all_data():
    """Prepare all downloaded data for exploration training."""

    print("\n" + "="*80)
    print(" "*20 + "PREPARE DATA FOR EXPLORATION")
    print("="*80)

    # Find all CSV files in data/master
    data_dir = project_root / "data" / "master"
    csv_files = list(data_dir.glob("*.csv"))

    if not csv_files:
        print("\n❌ No CSV files found in data/master/")
        print("   Run data download script first")
        return

    print(f"\n✅ Found {len(csv_files)} data files")

    # Group by asset class
    assets_by_class = {}

    for csv_file in csv_files:
        # Parse filename: SYMBOL_TIMEFRAME_START_END.csv
        filename = csv_file.stem
        parts = filename.split('_')

        if len(parts) < 2:
            print(f"⚠️  Skipping malformed filename: {filename}")
            continue

        symbol = parts[0]
        timeframe = parts[1]

        asset_class = detect_asset_class(symbol)

        if asset_class not in assets_by_class:
            assets_by_class[asset_class] = []

        assets_by_class[asset_class].append({
            'symbol': symbol,
            'timeframe': timeframe,
            'file': csv_file,
            'asset_class': asset_class
        })

    # Display summary
    print(f"\nAssets by class:")
    for asset_class, assets in assets_by_class.items():
        symbols = set(a['symbol'] for a in assets)
        print(f"  {asset_class:12s}: {len(symbols)} symbols, {len(assets)} files")

    # Prepare training data
    print(f"\n" + "="*80)
    print("GENERATING PHYSICS FEATURES")
    print("="*80)

    prepared_data = []

    for asset_class, assets in assets_by_class.items():
        print(f"\n[{asset_class.upper()}]")

        for asset in assets:
            try:
                # Load data
                df = pd.read_csv(asset['file'])

                # Check required columns
                required_cols = ['time', 'open', 'high', 'low', 'close']
                if not all(col in df.columns for col in required_cols):
                    print(f"  ⚠️  Skipping {asset['symbol']} {asset['timeframe']}: missing columns")
                    continue

                # Convert time to datetime
                df['time'] = pd.to_datetime(df['time'])
                df = df.set_index('time').sort_index()

                # Generate physics features
                df_with_physics = prepare_physics_features(df, f"{asset['symbol']}_{asset['timeframe']}")

                # Save prepared data
                output_dir = project_root / "data" / "prepared" / asset_class
                output_dir.mkdir(parents=True, exist_ok=True)

                output_file = output_dir / f"{asset['symbol']}_{asset['timeframe']}_physics.parquet"
                df_with_physics.to_parquet(output_file)

                # Save metadata
                metadata = {
                    'symbol': asset['symbol'],
                    'timeframe': asset['timeframe'],
                    'asset_class': asset_class,
                    'bars': len(df_with_physics),
                    'start_date': str(df_with_physics.index[0]),
                    'end_date': str(df_with_physics.index[-1]),
                    'features': list(df_with_physics.columns),
                    'prepared_at': str(pd.Timestamp.now())
                }

                metadata_file = output_dir / f"{asset['symbol']}_{asset['timeframe']}_metadata.json"
                with open(metadata_file, 'w') as f:
                    json.dump(metadata, f, indent=2)

                prepared_data.append(metadata)

                print(f"  ✅ {asset['symbol']} {asset['timeframe']}: {len(df_with_physics)} bars prepared")

            except Exception as e:
                print(f"  ❌ Error preparing {asset['symbol']} {asset['timeframe']}: {e}")
                continue

    # Save global manifest
    print(f"\n" + "="*80)
    print("SAVING MANIFEST")
    print("="*80)

    manifest = {
        'prepared_at': str(pd.Timestamp.now()),
        'total_datasets': len(prepared_data),
        'asset_classes': {
            asset_class: len([d for d in prepared_data if d['asset_class'] == asset_class])
            for asset_class in assets_by_class.keys()
        },
        'datasets': prepared_data
    }

    manifest_file = project_root / "data" / "prepared" / "manifest.json"
    with open(manifest_file, 'w') as f:
        json.dump(manifest, f, indent=2)

    print(f"\n✅ Manifest saved: {manifest_file}")
    print(f"\nSummary:")
    print(f"  Total datasets prepared: {len(prepared_data)}")
    for asset_class, count in manifest['asset_classes'].items():
        print(f"  {asset_class:12s}: {count} datasets")

    print(f"\n" + "="*80)
    print("✅ DATA PREPARATION COMPLETE")
    print("="*80)

    print(f"\nNext steps:")
    print(f"  1. Review prepared data in: data/prepared/")
    print(f"  2. Run exploration training: python scripts/run_exploration_batch.py")
    print(f"  3. Monitor results in: results/")


if __name__ == '__main__':
    prepare_all_data()
