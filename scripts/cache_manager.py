#!/usr/bin/env python3
"""
DATA CACHE MANAGER
==================

Once data is prepared, save it - never prepare again!

Usage:
    # First time: prepare and cache
    python cache_manager.py prepare
    
    # Later: just load from cache (instant!)
    python cache_manager.py load
"""

import sys
import pickle
import hashlib
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

CACHE_DIR = Path("/workspace/data/cache")


def get_cache_path(name: str) -> Path:
    """Get cache file path."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / f"{name}.pkl"


def save_to_cache(data: dict, name: str):
    """Save prepared data to cache."""
    cache_path = get_cache_path(name)
    
    with open(cache_path, 'wb') as f:
        pickle.dump({
            'data': data,
            'timestamp': datetime.now().isoformat(),
            'version': '1.0',
        }, f)
    
    size_mb = cache_path.stat().st_size / (1024 * 1024)
    print(f"✓ Cached {len(data)} items to {cache_path.name} ({size_mb:.1f} MB)")


def load_from_cache(name: str) -> dict:
    """Load prepared data from cache."""
    cache_path = get_cache_path(name)
    
    if not cache_path.exists():
        raise FileNotFoundError(f"Cache not found: {cache_path}")
    
    with open(cache_path, 'rb') as f:
        cached = pickle.load(f)
    
    print(f"✓ Loaded {len(cached['data'])} items from cache (saved: {cached['timestamp']})")
    return cached['data']


def cache_exists(name: str) -> bool:
    """Check if cache exists."""
    return get_cache_path(name).exists()


def load_csv_direct(filepath: Path) -> pd.DataFrame:
    """Load MT5 CSV directly."""
    df = pd.read_csv(filepath, sep='\t')
    df.columns = [c.strip('<>').lower() for c in df.columns]
    df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'])
    df.set_index('datetime', inplace=True)
    df = df.rename(columns={'tickvol': 'volume'})
    return df[['open', 'high', 'low', 'close', 'volume', 'spread']]


def prepare_all_data(data_dir: Path) -> dict:
    """Prepare all CSV files in directory."""
    datasets = {}
    
    csv_files = list(data_dir.glob("*.csv"))
    print(f"Found {len(csv_files)} CSV files")
    
    for i, filepath in enumerate(csv_files):
        try:
            name = filepath.stem
            df = load_csv_direct(filepath)
            datasets[name] = {
                'data': df,
                'symbol': name.split('_')[0],
                'timeframe': name.split('_')[1] if '_' in name else 'H1',
                'bars': len(df),
            }
            print(f"  [{i+1}/{len(csv_files)}] {name}: {len(df)} bars")
        except Exception as e:
            print(f"  ⚠ {filepath.name}: {e}")
    
    return datasets


def main():
    if len(sys.argv) < 2:
        print("Usage: python cache_manager.py [prepare|load|status]")
        return
    
    command = sys.argv[1]
    data_dir = Path("/workspace/data/runs/berserker_run3/data")
    cache_name = "all_instruments"
    
    if command == "prepare":
        print("=" * 60)
        print("PREPARING AND CACHING ALL DATA")
        print("=" * 60)
        
        datasets = prepare_all_data(data_dir)
        save_to_cache(datasets, cache_name)
        
        print(f"\n✅ Done! Next time just run: python cache_manager.py load")
        
    elif command == "load":
        print("=" * 60)
        print("LOADING FROM CACHE")
        print("=" * 60)
        
        if cache_exists(cache_name):
            datasets = load_from_cache(cache_name)
            print(f"\nAvailable: {list(datasets.keys())[:5]}... ({len(datasets)} total)")
        else:
            print("⚠ No cache found. Run 'prepare' first.")
            
    elif command == "status":
        cache_path = get_cache_path(cache_name)
        if cache_path.exists():
            size_mb = cache_path.stat().st_size / (1024 * 1024)
            print(f"✓ Cache exists: {cache_path} ({size_mb:.1f} MB)")
        else:
            print("✗ No cache")
    
    else:
        print(f"Unknown command: {command}")


if __name__ == "__main__":
    main()
