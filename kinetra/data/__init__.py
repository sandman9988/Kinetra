"""
Kinetra Data Management Package
================================

Unified data management system consolidating:
- data_manager.py: Atomic operations, broker/account organization, integrity checks
- data_management.py: Scientific data handling, test run isolation, caching
- unified_data_manager.py: Download workflows, testing framework integration

Main Components:
- DataManager: Core data management with atomic operations
- DownloadManager: Data download from multiple sources (MetaAPI, MT5, CSV)
- IntegrityChecker: Data validation, gap detection, quality scoring
- CacheManager: Feature caching and deduplication
- TestRunManager: Isolated test run data management

Usage:
    from kinetra.data import DataManager, DownloadManager
    
    # Core data management
    dm = DataManager()
    dm.prepare_training_data(['EURUSD', 'BTCUSD'])
    
    # Download new data
    downloader = DownloadManager()
    downloader.download_symbols(['GBPUSD'], timeframes=['H1', 'H4'])
"""

from .manager import DataManager
from .download import DownloadManager
from .integrity import IntegrityChecker
from .cache import CacheManager
from .test_isolation import TestRunManager

__all__ = [
    'DataManager',
    'DownloadManager',
    'IntegrityChecker',
    'CacheManager',
    'TestRunManager',
]
