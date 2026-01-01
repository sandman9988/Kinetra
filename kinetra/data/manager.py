"""
Core Data Manager
=================

Main data management interface.
Consolidated from data_manager.py, data_management.py, unified_data_manager.py.

Features:
- Broker/account/asset class organization
- Raw data immutability (append-only)
- Training data generation
- Integration with all submodules
"""

from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from .atomic_ops import AtomicFileWriter
from .cache import CacheManager
from .download import DownloadManager
from .integrity import IntegrityChecker
from .test_isolation import TestRunManager


class DataManager:
    """
    Core data management system.
    
    Directory Structure:
        data/
        ├── raw/                # Raw broker data (immutable, append-only)
        │   └── {broker}/
        │       └── {account}/
        │           ├── forex/
        │           ├── metals/
        │           └── indices/
        ├── training/           # Standardized training data
        │   ├── forex/
        │   └── metals/
        ├── cache/              # Feature cache
        └── test_runs/          # Isolated test runs
    
    Philosophy:
    - Raw data is immutable (source of truth)
    - Training data is regenerated fresh
    - All operations are atomic
    """
    
    def __init__(self, base_dir: Optional[Path] = None):
        """
        Initialize data manager.
        
        Args:
            base_dir: Base directory for data (default: ./data)
        """
        if base_dir is None:
            base_dir = Path(__file__).parent.parent.parent / "data"
            
        self.base_dir = Path(base_dir)
        self.raw_dir = self.base_dir / "raw"
        self.training_dir = self.base_dir / "training"
        self.cache_dir = self.base_dir / "cache"
        self.test_runs_dir = self.base_dir / "test_runs"
        
        # Create directories
        for d in [self.raw_dir, self.training_dir, self.cache_dir, self.test_runs_dir]:
            d.mkdir(parents=True, exist_ok=True)
            
        # Initialize submodules
        self.download_manager = DownloadManager(self.raw_dir)
        self.integrity_checker = IntegrityChecker()
        self.cache_manager = CacheManager(self.cache_dir)
        self.test_run_manager = TestRunManager(self.test_runs_dir)
        
    def get_asset_class(self, symbol: str) -> str:
        """Determine asset class from symbol."""
        symbol = symbol.upper()
        
        # Forex pairs
        currencies = ['USD', 'EUR', 'GBP', 'JPY', 'CHF', 'CAD', 'AUD', 'NZD']
        for c1 in currencies:
            for c2 in currencies:
                if c1 != c2 and f"{c1}{c2}" in symbol:
                    return 'forex'
                    
        # Metals
        if any(m in symbol for m in ['XAU', 'XAG', 'GOLD', 'SILVER']):
            return 'metals'
            
        # Crypto
        if any(c in symbol for c in ['BTC', 'ETH', 'CRYPTO']):
            return 'crypto'
            
        # Indices
        if any(i in symbol for i in ['US30', 'US500', 'NAS', 'DAX', 'FTSE']):
            return 'indices'
            
        return 'unknown'
        
    def list_available_data(self) -> Dict[str, List[str]]:
        """List all available raw data by asset class."""
        data = {}
        
        for asset_dir in self.raw_dir.rglob('*'):
            if asset_dir.is_dir() and asset_dir.parent.name in ['forex', 'metals', 'indices', 'crypto']:
                asset_class = asset_dir.parent.name
                if asset_class not in data:
                    data[asset_class] = []
                    
                # Find CSV files
                for csv_file in asset_dir.glob('*.csv'):
                    symbol = csv_file.stem.split('_')[0]
                    if symbol not in data[asset_class]:
                        data[asset_class].append(symbol)
                        
        return data
        
    def prepare_training_data(
        self,
        symbols: List[str],
        timeframe: str = 'H1',
        force_regenerate: bool = False
    ) -> Dict[str, Path]:
        """
        Prepare standardized training data.
        
        Args:
            symbols: List of symbols
            timeframe: Timeframe
            force_regenerate: Force regeneration even if exists
            
        Returns:
            Dict mapping symbol to training data path
        """
        result = {}
        
        for symbol in symbols:
            asset_class = self.get_asset_class(symbol)
            output_dir = self.training_dir / asset_class
            output_dir.mkdir(parents=True, exist_ok=True)
            
            output_path = output_dir / f"{symbol}_{timeframe}_standardized.parquet"
            
            if not force_regenerate and output_path.exists():
                result[symbol] = output_path
                continue
                
            # TODO: Load raw data, standardize, save
            # This is a stub - full implementation would:
            # 1. Load raw data
            # 2. Standardize format
            # 3. Calculate normalization stats
            # 4. Save as parquet
            
            result[symbol] = output_path
            
        return result
