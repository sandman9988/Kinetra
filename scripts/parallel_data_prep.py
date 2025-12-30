"""
Parallel Data Preparation Pipeline
===================================
32-thread parallel processing for:
- Holiday tagging (using market_calendar)
- Physics feature computation (60+ measurements)
- Parquet output with full metadata

Uses ThreadPoolExecutor for I/O and ProcessPoolExecutor for CPU.
"""

import sys
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import multiprocessing as mp

import pandas as pd
import numpy as np

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from kinetra.physics_engine import PhysicsEngine
from kinetra.market_calendar import get_calendar_for_symbol

# Use all 32 threads
MAX_WORKERS = 32


class PrepProgress:
    """Thread-safe progress tracker."""

    def __init__(self, total: int):
        self.total = total
        self.completed = 0
        self.failed = 0
        self.start_time = datetime.now()
        import threading
        self._lock = threading.Lock()

    def complete(self, success: bool = True):
        with self._lock:
            if success:
                self.completed += 1
            else:
                self.failed += 1

    def status(self) -> str:
        elapsed = (datetime.now() - self.start_time).total_seconds()
        rate = self.completed / elapsed if elapsed > 0 else 0
        pct = (self.completed / self.total * 100) if self.total > 0 else 0
        return f"[{self.completed}/{self.total}] {pct:.0f}% | {rate:.1f} files/sec | ❌{self.failed}"


def load_csv(filepath: Path) -> pd.DataFrame:
    """Load MetaTrader CSV format."""
    df = pd.read_csv(filepath, sep='\t')
    # Normalize column names
    df.columns = [c.lower().replace('<', '').replace('>', '') for c in df.columns]
    # Create datetime index
    df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'])
    df = df.set_index('datetime').sort_index()
    return df


def add_holiday_tags(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Tag bars with holiday/session info."""
    calendar = get_calendar_for_symbol(symbol)

    # Is holiday
    df['is_holiday'] = df.index.map(lambda dt: calendar.is_holiday(dt)).astype(int)

    # Is trading time (expected to be open)
    df['is_trading_time'] = df.index.map(lambda dt: calendar.is_trading_time(dt)).astype(int)

    # Day of week (0=Mon, 6=Sun)
    df['day_of_week'] = df.index.dayofweek

    # Hour of day
    df['hour'] = df.index.hour

    # Session tags (Asian/European/US)
    def get_session(hour: int) -> int:
        if 0 <= hour < 8:
            return 0  # Asian
        elif 8 <= hour < 16:
            return 1  # European
        else:
            return 2  # US
    df['session'] = df['hour'].map(get_session)

    # Weekend flag
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)

    # Month
    df['month'] = df.index.month

    # Quarter
    df['quarter'] = df.index.quarter

    return df


def compute_physics_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute all physics measurements using PhysicsEngine."""
    engine = PhysicsEngine(
        vel_window=1,
        damping_window=64,
        entropy_window=64,
        re_slow=24,
        re_fast=6,
        pe_window=72,
        pct_window=min(500, len(df) // 2),  # Adaptive
        n_clusters=3,
    )

    # Compute physics state
    physics_df = engine.compute_physics_state(
        prices=df['close'],
        volume=df.get('tickvol'),
        high=df.get('high'),
        low=df.get('low'),
        open_price=df.get('open'),
        include_percentiles=True,
        include_kinematics=True,
        include_flow=True,
    )

    # Merge physics features
    for col in physics_df.columns:
        df[col] = physics_df[col].values

    return df


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived/engineered features."""

    # Price-based
    df['range'] = df['high'] - df['low']
    df['body'] = abs(df['close'] - df['open'])
    df['body_pct'] = df['body'] / df['range'].replace(0, np.nan)
    df['upper_wick'] = df['high'] - df[['open', 'close']].max(axis=1)
    df['lower_wick'] = df[['open', 'close']].min(axis=1) - df['low']

    # Direction
    df['direction'] = np.sign(df['close'] - df['open'])

    # Gap from previous close
    df['gap'] = df['open'] - df['close'].shift(1)
    df['gap_pct'] = df['gap'] / df['close'].shift(1)

    # Rolling stats (multiple windows)
    for window in [6, 12, 24, 48]:
        df[f'vol_ma_{window}'] = df['tickvol'].rolling(window).mean()
        df[f'range_ma_{window}'] = df['range'].rolling(window).mean()
        df[f'close_ma_{window}'] = df['close'].rolling(window).mean()
        df[f'close_std_{window}'] = df['close'].rolling(window).std()

        # Price vs MA
        df[f'close_vs_ma_{window}'] = (df['close'] - df[f'close_ma_{window}']) / df[f'close_std_{window}'].replace(0, np.nan)

    # ATR-like volatility
    tr = pd.concat([
        df['high'] - df['low'],
        abs(df['high'] - df['close'].shift(1)),
        abs(df['low'] - df['close'].shift(1))
    ], axis=1).max(axis=1)
    df['tr'] = tr
    df['atr_14'] = tr.rolling(14).mean()
    df['atr_pct'] = df['atr_14'] / df['close']

    # Consecutive direction
    df['consec_up'] = (df['direction'] == 1).astype(int).groupby((df['direction'] != 1).cumsum()).cumsum()
    df['consec_down'] = (df['direction'] == -1).astype(int).groupby((df['direction'] != -1).cumsum()).cumsum()

    # Volume spikes
    df['vol_spike'] = (df['tickvol'] > df['tickvol'].rolling(24).mean() * 2).astype(int)

    # Big move detection
    df['big_move'] = (df['range'] > df['atr_14'] * 1.5).astype(int)

    return df


def process_single_file(args) -> dict:
    """Process a single CSV file - runs in thread/process pool."""
    filepath, output_dir, asset_class = args

    try:
        # Parse filename for metadata
        stem = filepath.stem
        parts = stem.split('_')
        symbol = parts[0]
        timeframe = parts[1] if len(parts) > 1 else 'H1'

        # Load data
        df = load_csv(filepath)

        if len(df) < 100:
            return {'status': 'skipped', 'file': str(filepath), 'reason': 'too few bars'}

        # Add metadata columns
        df['symbol'] = symbol
        df['timeframe'] = timeframe
        df['asset_class'] = asset_class

        # Add holiday tags
        df = add_holiday_tags(df, symbol)

        # Compute physics features
        df = compute_physics_features(df)

        # Add derived features
        df = add_derived_features(df)

        # Output path
        output_path = output_dir / asset_class / f"{stem}.parquet"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Write parquet (columnar, compressed)
        df.to_parquet(output_path, compression='zstd')

        feature_count = len([c for c in df.columns if c not in ['symbol', 'timeframe', 'asset_class', 'date', 'time']])

        return {
            'status': 'success',
            'file': str(filepath),
            'output': str(output_path),
            'bars': len(df),
            'features': feature_count,
        }

    except Exception as e:
        return {'status': 'failed', 'file': str(filepath), 'error': str(e)}


def prep_all_data(data_dir: Path = None, output_dir: Path = None):
    """Main entry point - parallel prep all downloaded data."""

    if data_dir is None:
        data_dir = project_root / "data" / "master"
    if output_dir is None:
        output_dir = project_root / "data" / "prepared"

    print("\n" + "="*70)
    print("PARALLEL DATA PREPARATION (32 threads)")
    print("="*70)

    # Find all CSV files
    all_files = []
    for asset_dir in data_dir.iterdir():
        if asset_dir.is_dir() and asset_dir.name not in ['.', '..']:
            asset_class = asset_dir.name
            for csv_file in asset_dir.glob("*.csv"):
                all_files.append((csv_file, output_dir, asset_class))

    print(f"  Found: {len(all_files)} CSV files")
    print(f"  Output: {output_dir}")
    print(f"  Workers: {MAX_WORKERS}")

    if not all_files:
        print("  ❌ No files to process!")
        return

    # Progress tracker
    progress = PrepProgress(len(all_files))

    print(f"\n[1] Processing files...")

    results = []

    # Use ProcessPoolExecutor for CPU-bound physics computation
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_single_file, args): args for args in all_files}

        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            progress.complete(result['status'] == 'success')

            # Heartbeat
            print(f"\r  💓 {progress.status()}", end="", flush=True)

    print()

    # Summary
    print("\n" + "="*70)
    print("PREPARATION SUMMARY")
    print("="*70)

    success = [r for r in results if r['status'] == 'success']
    failed = [r for r in results if r['status'] == 'failed']
    skipped = [r for r in results if r['status'] == 'skipped']

    print(f"  ✅ Success: {len(success)}")
    print(f"  ❌ Failed: {len(failed)}")
    print(f"  ⏭️  Skipped: {len(skipped)}")

    if success:
        total_bars = sum(r['bars'] for r in success)
        avg_features = sum(r['features'] for r in success) / len(success)
        print(f"  📊 Total bars: {total_bars:,}")
        print(f"  📐 Avg features per file: {avg_features:.0f}")

    if failed:
        print("\n  Failed files:")
        for r in failed[:5]:
            print(f"    - {Path(r['file']).name}: {r.get('error', '?')}")
        if len(failed) > 5:
            print(f"    ... and {len(failed) - 5} more")

    # Write manifest
    manifest = {
        'prepared_at': datetime.now().isoformat(),
        'total_files': len(results),
        'success': len(success),
        'failed': len(failed),
        'skipped': len(skipped),
        'files': success,
    }

    import json
    manifest_path = output_dir / "prep_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)

    print(f"\n  📄 Manifest: {manifest_path}")
    print("\n✅ DATA READY FOR PHYSICS EXPLORATION!")
    print(f"   Run: python rl_exploration_framework.py --data-dir {output_dir}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Parallel data preparation")
    parser.add_argument('--data-dir', type=Path, default=None)
    parser.add_argument('--output-dir', type=Path, default=None)
    args = parser.parse_args()

    prep_all_data(args.data_dir, args.output_dir)
