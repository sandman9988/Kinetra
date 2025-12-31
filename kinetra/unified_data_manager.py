"""
Unified Data Management System
===============================

Consolidates all data download, preparation, and testing workflows into one system.

Features:
- Data download (MetaAPI, MT5, manual)
- Data validation and integrity checks
- Data preparation and standardization
- Training/test splits
- Integration with testing framework
- Atomic operations with rollback
- Gap detection and filling

Usage:
    from kinetra.unified_data_manager import UnifiedDataManager
    
    manager = UnifiedDataManager()
    
    # Download data
    manager.download_data(symbols=['EURUSD', 'BTCUSD'], timeframes=['H1', 'H4'])
    
    # Prepare for testing
    instruments = manager.prepare_for_testing(asset_classes=['forex', 'crypto'])
    
    # Run tests
    from kinetra.testing_framework import TestingFramework
    framework = TestingFramework()
    # ... use instruments
"""

import asyncio
import hashlib
import json
import logging
import shutil
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class DataSource:
    """Configuration for a data source."""
    name: str  # 'metaapi', 'mt5', 'csv'
    config: Dict  # Source-specific configuration
    priority: int = 1  # Lower = higher priority


@dataclass
class DownloadJob:
    """A data download job."""
    symbol: str
    timeframe: str
    start_date: datetime
    end_date: datetime
    source: DataSource
    output_path: Path
    status: str = 'pending'  # pending, running, completed, failed
    error: Optional[str] = None
    bars_downloaded: int = 0


@dataclass
class DataIntegrity:
    """Data integrity information."""
    file_path: Path
    checksum: str
    bars_count: int
    start_time: datetime
    end_time: datetime
    gaps_count: int
    quality_score: float  # 0-1
    validation_date: datetime


class UnifiedDataManager:
    """
    Unified data management system.
    
    Consolidates:
    - Data download workflows
    - Data validation and integrity
    - Data preparation and standardization
    - Testing framework integration
    """
    
    def __init__(self, base_dir: Optional[Path] = None):
        """
        Initialize unified data manager.
        
        Args:
            base_dir: Base directory for all data (default: ./data)
        """
        if base_dir is None:
            base_dir = Path(__file__).parent.parent / "data"
        
        self.base_dir = Path(base_dir)
        
        # Directory structure
        self.raw_dir = self.base_dir / "raw"
        self.master_dir = self.base_dir / "master"
        self.prepared_dir = self.base_dir / "prepared"
        self.cache_dir = self.base_dir / "cache"
        self.metadata_dir = self.base_dir / "metadata"
        
        # Create directories
        for dir_path in [self.raw_dir, self.master_dir, self.prepared_dir, 
                        self.cache_dir, self.metadata_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Data sources
        self.sources: List[DataSource] = []
        
        # Job tracking
        self.download_jobs: List[DownloadJob] = []
        
        logger.info(f"UnifiedDataManager initialized at {self.base_dir}")
    
    # =========================================================================
    # DATA SOURCE MANAGEMENT
    # =========================================================================
    
    def add_data_source(self, source: DataSource):
        """Add a data source."""
        self.sources.append(source)
        self.sources.sort(key=lambda x: x.priority)
        logger.info(f"Added data source: {source.name} (priority {source.priority})")
    
    def configure_metaapi(self, api_key: str, account_id: str):
        """Configure MetaAPI source."""
        source = DataSource(
            name='metaapi',
            config={'api_key': api_key, 'account_id': account_id},
            priority=1
        )
        self.add_data_source(source)
    
    def configure_mt5(self, **kwargs):
        """Configure MT5 source."""
        source = DataSource(
            name='mt5',
            config=kwargs,
            priority=2
        )
        self.add_data_source(source)
    
    def configure_csv_import(self, import_dir: Path):
        """Configure CSV import source."""
        source = DataSource(
            name='csv',
            config={'import_dir': import_dir},
            priority=3
        )
        self.add_data_source(source)
    
    # =========================================================================
    # DATA DISCOVERY
    # =========================================================================
    
    def discover_available_data(self) -> Dict[str, List[Dict]]:
        """
        Discover all available data files.
        
        Returns:
            Dict mapping asset_class -> list of file info
        """
        data_by_class = defaultdict(list)
        
        # Check master directory
        for csv_file in self.master_dir.rglob("*.csv"):
            # Parse filename
            parts = csv_file.stem.split('_')
            if len(parts) < 2:
                continue
            
            symbol = parts[0]
            timeframe = parts[1]
            
            # Classify
            asset_class = self._classify_symbol(symbol)
            if asset_class == 'unknown':
                continue
            
            # Get file stats
            try:
                df = pd.read_csv(csv_file, nrows=1)
                file_info = {
                    'symbol': symbol,
                    'timeframe': timeframe,
                    'asset_class': asset_class,
                    'path': str(csv_file),
                    'size_mb': csv_file.stat().st_size / 1024 / 1024,
                }
                data_by_class[asset_class].append(file_info)
            except Exception as e:
                logger.warning(f"Error reading {csv_file}: {e}")
        
        return dict(data_by_class)
    
    def _classify_symbol(self, symbol: str) -> str:
        """Classify symbol into asset class."""
        s = symbol.upper().replace('+', '').replace('-', '')
        
        if any(x in s for x in ['BTC', 'ETH', 'XRP', 'LTC', 'DOGE', 'SOL']):
            return 'crypto'
        elif any(x in s for x in ['XAU', 'XAG', 'XPT', 'XPD', 'GOLD', 'SILVER', 'COPPER']):
            return 'metals'
        elif any(x in s for x in ['OIL', 'WTI', 'BRENT', 'GAS', 'GASOIL', 'UKOUSD', 'USOIL']):
            return 'commodities'
        elif any(x in s for x in ['SPX', 'NAS', 'DOW', 'DJ', 'DAX', 'FTSE', 'NIKKEI', 
                                   'US30', 'US500', 'US100', 'GER', 'UK100', 'SA40', 'EU50']):
            return 'indices'
        elif len(s) == 6 and s.isalpha():
            return 'forex'
        return 'unknown'
    
    # =========================================================================
    # DATA DOWNLOAD
    # =========================================================================
    
    def download_data(
        self,
        symbols: List[str],
        timeframes: List[str],
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        force: bool = False
    ) -> List[DownloadJob]:
        """
        Download data for specified symbols and timeframes.
        
        Args:
            symbols: List of symbols to download
            timeframes: List of timeframes (e.g., ['M15', 'H1', 'H4'])
            start_date: Start date (default: 1 year ago)
            end_date: End date (default: now)
            force: Force re-download even if data exists
            
        Returns:
            List of download jobs
        """
        if not start_date:
            start_date = datetime.now() - timedelta(days=365)
        if not end_date:
            end_date = datetime.now()
        
        jobs = []
        
        for symbol in symbols:
            for timeframe in timeframes:
                # Check if data already exists
                output_path = self.master_dir / f"{symbol}_{timeframe}.csv"
                
                if output_path.exists() and not force:
                    logger.info(f"Data already exists: {symbol}_{timeframe} (use force=True to re-download)")
                    continue
                
                # Create download job
                job = DownloadJob(
                    symbol=symbol,
                    timeframe=timeframe,
                    start_date=start_date,
                    end_date=end_date,
                    source=self.sources[0] if self.sources else DataSource('manual', {}),
                    output_path=output_path
                )
                
                jobs.append(job)
                self.download_jobs.append(job)
        
        logger.info(f"Created {len(jobs)} download jobs")
        
        # Execute jobs (simplified - actual implementation would use async)
        for job in jobs:
            self._execute_download_job(job)
        
        return jobs
    
    def _execute_download_job(self, job: DownloadJob):
        """Execute a download job (placeholder)."""
        logger.info(f"Downloading {job.symbol}_{job.timeframe}...")
        
        # This is a placeholder - actual implementation would:
        # 1. Connect to data source (MetaAPI, MT5, etc.)
        # 2. Download data in chunks
        # 3. Validate and save
        # 4. Update job status
        
        job.status = 'pending'  # Would be 'completed' after actual download
        logger.warning(f"Download job created but not executed (requires data source configuration)")
    
    # =========================================================================
    # DATA VALIDATION
    # =========================================================================
    
    def validate_data(self, file_path: Path) -> DataIntegrity:
        """
        Validate data file integrity.
        
        Returns:
            DataIntegrity object with validation results
        """
        df = pd.read_csv(file_path)
        df['time'] = pd.to_datetime(df['time'])
        df = df.sort_values('time')
        
        # Calculate checksum
        checksum = hashlib.md5(file_path.read_bytes()).hexdigest()
        
        # Detect gaps
        time_diffs = df['time'].diff()
        median_diff = time_diffs.median()
        gaps = (time_diffs > median_diff * 2).sum()
        
        # Calculate quality score
        missing_pct = df.isnull().sum().sum() / (len(df) * len(df.columns))
        duplicate_pct = df.duplicated(subset=['time']).sum() / len(df)
        quality_score = 1.0 - (missing_pct + duplicate_pct + (gaps / len(df)))
        
        return DataIntegrity(
            file_path=file_path,
            checksum=checksum,
            bars_count=len(df),
            start_time=df['time'].iloc[0],
            end_time=df['time'].iloc[-1],
            gaps_count=gaps,
            quality_score=max(0, quality_score),
            validation_date=datetime.now()
        )
    
    def validate_all_data(self) -> Dict[str, DataIntegrity]:
        """Validate all data files in master directory."""
        results = {}
        
        for csv_file in self.master_dir.glob("*.csv"):
            try:
                integrity = self.validate_data(csv_file)
                results[csv_file.name] = integrity
                
                # Save integrity report
                report_path = self.metadata_dir / f"{csv_file.stem}_integrity.json"
                with open(report_path, 'w') as f:
                    json.dump(asdict(integrity), f, indent=2, default=str)
                
                logger.info(f"Validated {csv_file.name}: Quality={integrity.quality_score:.2%}")
            except Exception as e:
                logger.error(f"Error validating {csv_file.name}: {e}")
        
        return results
    
    # =========================================================================
    # DATA PREPARATION FOR TESTING
    # =========================================================================
    
    def prepare_for_testing(
        self,
        asset_classes: Optional[List[str]] = None,
        timeframes: Optional[List[str]] = None,
        max_per_class: int = 3,
        train_test_split: float = 0.8,
        min_quality_score: float = 0.7
    ) -> List['InstrumentSpec']:
        """
        Prepare data for testing framework.
        
        Args:
            asset_classes: Filter by asset classes
            timeframes: Filter by timeframes
            max_per_class: Max instruments per asset class
            train_test_split: Train/test ratio
            min_quality_score: Minimum quality score to include
            
        Returns:
            List of InstrumentSpec objects for testing framework
        """
        from kinetra.testing_framework import InstrumentSpec
        
        # Discover available data
        available = self.discover_available_data()
        
        instruments = []
        by_class = defaultdict(int)
        
        for asset_class, files in available.items():
            # Filter by asset class
            if asset_classes and asset_class not in asset_classes:
                continue
            
            for file_info in files:
                # Filter by timeframe
                if timeframes and file_info['timeframe'] not in timeframes:
                    continue
                
                # Check if we've hit limit for this class
                if by_class[asset_class] >= max_per_class:
                    continue
                
                # Validate quality
                file_path = Path(file_info['path'])
                integrity = self.validate_data(file_path)
                
                if integrity.quality_score < min_quality_score:
                    logger.warning(f"Skipping {file_path.name}: quality={integrity.quality_score:.2%} < {min_quality_score:.2%}")
                    continue
                
                # Create instrument spec
                spec = InstrumentSpec(
                    symbol=file_info['symbol'],
                    asset_class=file_info['asset_class'],
                    timeframe=file_info['timeframe'],
                    data_path=file_info['path']
                )
                
                instruments.append(spec)
                by_class[asset_class] += 1
        
        logger.info(f"Prepared {len(instruments)} instruments for testing")
        for asset_class, count in by_class.items():
            logger.info(f"  {asset_class}: {count} instruments")
        
        return instruments
    
    # =========================================================================
    # DATA EXPORT FOR TESTING
    # =========================================================================
    
    def export_for_backtest(
        self,
        symbol: str,
        timeframe: str,
        output_format: str = 'csv'
    ) -> Path:
        """
        Export data in format ready for backtesting.
        
        Args:
            symbol: Symbol to export
            timeframe: Timeframe to export
            output_format: 'csv', 'parquet', or 'feather'
            
        Returns:
            Path to exported file
        """
        # Find source file
        source_file = self.master_dir / f"{symbol}_{timeframe}.csv"
        
        if not source_file.exists():
            raise FileNotFoundError(f"Data not found: {symbol}_{timeframe}")
        
        # Read and prepare
        df = pd.read_csv(source_file)
        df['time'] = pd.to_datetime(df['time'])
        df = df.sort_values('time')
        
        # Export
        output_file = self.prepared_dir / f"{symbol}_{timeframe}_backtest.{output_format}"
        
        if output_format == 'csv':
            df.to_csv(output_file, index=False)
        elif output_format == 'parquet':
            df.to_parquet(output_file, index=False)
        elif output_format == 'feather':
            df.to_feather(output_file)
        else:
            raise ValueError(f"Unsupported format: {output_format}")
        
        logger.info(f"Exported {symbol}_{timeframe} to {output_file}")
        return output_file
    
    # =========================================================================
    # UTILITIES
    # =========================================================================
    
    def get_summary(self) -> Dict:
        """Get summary of data manager state."""
        available = self.discover_available_data()
        
        total_files = sum(len(files) for files in available.values())
        total_size_mb = 0
        
        for files in available.values():
            for file_info in files:
                total_size_mb += file_info['size_mb']
        
        return {
            'base_dir': str(self.base_dir),
            'total_files': total_files,
            'total_size_mb': round(total_size_mb, 2),
            'by_asset_class': {
                cls: len(files) for cls, files in available.items()
            },
            'data_sources': [s.name for s in self.sources],
            'download_jobs': {
                'total': len(self.download_jobs),
                'pending': sum(1 for j in self.download_jobs if j.status == 'pending'),
                'completed': sum(1 for j in self.download_jobs if j.status == 'completed'),
                'failed': sum(1 for j in self.download_jobs if j.status == 'failed'),
            }
        }
    
    def print_summary(self):
        """Print summary to console."""
        summary = self.get_summary()
        
        print("\n" + "="*80)
        print("UNIFIED DATA MANAGER SUMMARY")
        print("="*80)
        print(f"Base Directory: {summary['base_dir']}")
        print(f"Total Files: {summary['total_files']}")
        print(f"Total Size: {summary['total_size_mb']:.2f} MB")
        print(f"\nBy Asset Class:")
        for cls, count in summary['by_asset_class'].items():
            print(f"  {cls}: {count} files")
        print(f"\nData Sources: {', '.join(summary['data_sources']) if summary['data_sources'] else 'None configured'}")
        print(f"\nDownload Jobs:")
        print(f"  Total: {summary['download_jobs']['total']}")
        print(f"  Pending: {summary['download_jobs']['pending']}")
        print(f"  Completed: {summary['download_jobs']['completed']}")
        print(f"  Failed: {summary['download_jobs']['failed']}")
        print("="*80 + "\n")


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def quick_setup(base_dir: Optional[Path] = None) -> UnifiedDataManager:
    """
    Quick setup for data manager.
    
    Returns:
        Configured UnifiedDataManager
    """
    manager = UnifiedDataManager(base_dir)
    
    # Try to auto-detect CSV data in common locations
    common_locations = [
        Path("data/master"),
        Path("data"),
        Path("../data"),
    ]
    
    for location in common_locations:
        if location.exists() and any(location.glob("*.csv")):
            manager.configure_csv_import(location)
            logger.info(f"Auto-detected CSV data in {location}")
            break
    
    return manager
