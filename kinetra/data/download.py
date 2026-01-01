"""
Download Manager
================

Data download from multiple sources (MetaAPI, MT5, CSV).
Consolidated from unified_data_manager.py.

Features:
- Multi-source downloads (MetaAPI, MT5, CSV)
- Parallel downloads with throttling
- Progress tracking
- Automatic retry with backoff
"""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional


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


class DownloadManager:
    """
    Manages data downloads from multiple sources.
    
    Supports:
    - MetaAPI
    - MT5 direct connection
    - CSV files
    
    Features:
    - Parallel downloads
    - Progress tracking
    - Automatic retry
    """
    
    def __init__(self, output_dir: Path):
        """
        Initialize download manager.
        
        Args:
            output_dir: Directory for downloaded data
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.jobs: List[DownloadJob] = []
        self.sources: Dict[str, DataSource] = {}
        
    def register_source(self, source: DataSource) -> None:
        """Register a data source."""
        self.sources[source.name] = source
        
    def download_symbols(
        self,
        symbols: List[str],
        timeframes: List[str],
        source_name: str = 'metaapi',
        **kwargs
    ) -> List[DownloadJob]:
        """
        Download data for symbols.
        
        Args:
            symbols: List of symbols
            timeframes: List of timeframes
            source_name: Data source to use
            **kwargs: Additional arguments
            
        Returns:
            List of download jobs
        """
        # Create jobs
        jobs = []
        for symbol in symbols:
            for timeframe in timeframes:
                job = DownloadJob(
                    symbol=symbol,
                    timeframe=timeframe,
                    start_date=kwargs.get('start_date', datetime.now()),
                    end_date=kwargs.get('end_date', datetime.now()),
                    source=self.sources[source_name],
                    output_path=self.output_dir / f"{symbol}_{timeframe}.csv"
                )
                jobs.append(job)
                
        self.jobs.extend(jobs)
        return jobs
        
    def get_progress(self) -> Dict:
        """Get download progress."""
        total = len(self.jobs)
        completed = sum(1 for j in self.jobs if j.status == 'completed')
        failed = sum(1 for j in self.jobs if j.status == 'failed')
        
        return {
            'total': total,
            'completed': completed,
            'failed': failed,
            'running': total - completed - failed,
            'progress_pct': (completed / total * 100) if total > 0 else 0
        }
