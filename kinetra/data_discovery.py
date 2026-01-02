"""
Dynamic Data Discovery for Kinetra
===================================

Automatically discover and select data files based on:
- Asset class (crypto, forex, indices, metals, commodities)
- Timeframe (M15, M30, H1, H4, D1)
- Selection strategy (all, top N, random sample)

NO HARDCODED PATHS - discovers from data directory structure.

Usage:
    # Discover all crypto H1 data
    discovery = DataDiscovery()
    files = discovery.find(asset_class='crypto', timeframe='H1')

    # Get top 5 forex instruments
    files = discovery.find(asset_class='forex', timeframe='H4', limit=5)

    # Get all available data
    files = discovery.find_all()

    # Get random sample
    files = discovery.find(asset_class='indices', sample=3)
"""

from pathlib import Path
from typing import List, Optional, Dict, Set, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
import re
import random


@dataclass
class DataFile:
    """Metadata for a discovered data file."""
    path: Path
    symbol: str
    timeframe: str
    asset_class: str
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    split: str = "unknown"  # train, test, master, prepared

    @property
    def filename(self) -> str:
        return self.path.name

    @property
    def size_mb(self) -> float:
        return self.path.stat().st_size / 1024 / 1024

    def __repr__(self) -> str:
        return f"DataFile({self.asset_class}/{self.symbol}_{self.timeframe})"


class DataDiscovery:
    """
    Discover and filter data files dynamically.

    No hardcoded paths - scans data directory structure.
    """

    # Asset class mapping (directory name -> class name)
    ASSET_CLASSES = {
        'crypto': ['crypto', 'cryptocurrency'],
        'forex': ['forex', 'fx'],
        'indices': ['indices', 'index', 'stocks'],
        'metals': ['metals', 'precious'],
        'commodities': ['commodities', 'energy', 'commodity'],
    }

    # Valid timeframes
    TIMEFRAMES = ['M1', 'M5', 'M15', 'M30', 'H1', 'H4', 'D1', 'W1', 'MN1']

    def __init__(self, data_root: str = "data"):
        """
        Initialize data discovery.

        Args:
            data_root: Root data directory (default: "data")
        """
        self.data_root = Path(data_root)
        self._cache: Dict[str, List[DataFile]] = {}
        self._asset_class_cache: Dict[str, Set[str]] = defaultdict(set)

    def discover_all(self, refresh: bool = False) -> List[DataFile]:
        """
        Discover all available data files.

        Args:
            refresh: Force refresh cache

        Returns:
            List of discovered data files
        """
        if not refresh and "all" in self._cache:
            return self._cache["all"]

        discovered = []

        # Search in common data directories
        search_paths = [
            self.data_root / "master",
            self.data_root / "prepared" / "train",
            self.data_root / "prepared" / "test",
            self.data_root / "prepared",
        ]

        for base_path in search_paths:
            if not base_path.exists():
                continue

            # Determine split type
            split = "unknown"
            if "master" in str(base_path):
                split = "master"
            elif "train" in str(base_path):
                split = "train"
            elif "test" in str(base_path):
                split = "test"
            elif "prepared" in str(base_path):
                split = "prepared"

            # Scan for CSV files
            for csv_file in base_path.rglob("*.csv"):
                try:
                    metadata = self._parse_file(csv_file, split)
                    if metadata:
                        discovered.append(metadata)
                        # Cache asset class info
                        self._asset_class_cache[metadata.asset_class].add(metadata.symbol)
                except Exception as e:
                    # Skip files we can't parse
                    continue

        self._cache["all"] = discovered
        return discovered

    def find(
        self,
        asset_class: Optional[str] = None,
        timeframe: Optional[str] = None,
        symbol: Optional[str] = None,
        split: Optional[str] = None,
        limit: Optional[int] = None,
        sample: Optional[int] = None,
        sort_by: str = "symbol"
    ) -> List[DataFile]:
        """
        Find data files matching criteria.

        Args:
            asset_class: Filter by asset class (crypto, forex, indices, metals, commodities)
            timeframe: Filter by timeframe (M15, M30, H1, H4, D1, etc)
            symbol: Filter by symbol (exact match or regex)
            split: Filter by split (train, test, master, prepared)
            limit: Limit to top N results (after sorting)
            sample: Random sample of N results (mutually exclusive with limit)
            sort_by: Sort by field (symbol, timeframe, size, path)

        Returns:
            List of matching data files

        Examples:
            # All crypto H1 data
            find(asset_class='crypto', timeframe='H1')

            # Top 5 forex instruments
            find(asset_class='forex', limit=5)

            # Random 3 indices
            find(asset_class='indices', sample=3)

            # All BTC data
            find(symbol='BTC.*')
        """
        # Discover all files if not cached
        all_files = self.discover_all()

        # Apply filters
        filtered = all_files

        if asset_class:
            filtered = [f for f in filtered if f.asset_class == asset_class.lower()]

        if timeframe:
            filtered = [f for f in filtered if f.timeframe == timeframe.upper()]

        if symbol:
            # Support regex matching
            pattern = re.compile(symbol, re.IGNORECASE)
            filtered = [f for f in filtered if pattern.search(f.symbol)]

        if split:
            filtered = [f for f in filtered if f.split == split.lower()]

        # Sort results
        if sort_by == "symbol":
            filtered = sorted(filtered, key=lambda f: f.symbol)
        elif sort_by == "timeframe":
            filtered = sorted(filtered, key=lambda f: f.timeframe)
        elif sort_by == "size":
            filtered = sorted(filtered, key=lambda f: f.size_mb, reverse=True)
        elif sort_by == "path":
            filtered = sorted(filtered, key=lambda f: str(f.path))

        # Apply limit or sample
        if sample is not None:
            filtered = random.sample(filtered, min(sample, len(filtered)))
        elif limit is not None:
            filtered = filtered[:limit]

        return filtered

    def get_asset_classes(self) -> List[str]:
        """Get list of available asset classes."""
        self.discover_all()
        return sorted(self._asset_class_cache.keys())

    def get_symbols(self, asset_class: Optional[str] = None) -> List[str]:
        """
        Get list of available symbols.

        Args:
            asset_class: Filter by asset class

        Returns:
            Sorted list of symbols
        """
        self.discover_all()

        if asset_class:
            return sorted(self._asset_class_cache.get(asset_class.lower(), set()))
        else:
            all_symbols = set()
            for symbols in self._asset_class_cache.values():
                all_symbols.update(symbols)
            return sorted(all_symbols)

    def get_timeframes(self, asset_class: Optional[str] = None) -> List[str]:
        """
        Get list of available timeframes.

        Args:
            asset_class: Filter by asset class

        Returns:
            Sorted list of timeframes
        """
        files = self.find(asset_class=asset_class) if asset_class else self.discover_all()
        timeframes = set(f.timeframe for f in files)

        # Sort by timeframe hierarchy
        timeframe_order = {tf: i for i, tf in enumerate(self.TIMEFRAMES)}
        return sorted(timeframes, key=lambda tf: timeframe_order.get(tf, 999))

    def get_top_symbols(
        self,
        asset_class: str,
        n: int = 5,
        by: str = "name"
    ) -> List[str]:
        """
        Get top N symbols for an asset class.

        Args:
            asset_class: Asset class to filter
            n: Number of symbols to return
            by: Sorting criteria ('name', 'files', 'data_size')

        Returns:
            List of top N symbols
        """
        files = self.find(asset_class=asset_class)

        if by == "name":
            # Alphabetically first
            symbols = sorted(set(f.symbol for f in files))[:n]
        elif by == "files":
            # Most files available
            symbol_counts = defaultdict(int)
            for f in files:
                symbol_counts[f.symbol] += 1
            symbols = [s for s, _ in sorted(symbol_counts.items(), key=lambda x: -x[1])][:n]
        elif by == "data_size":
            # Largest total data
            symbol_sizes = defaultdict(float)
            for f in files:
                symbol_sizes[f.symbol] += f.size_mb
            symbols = [s for s, _ in sorted(symbol_sizes.items(), key=lambda x: -x[1])][:n]
        else:
            symbols = sorted(set(f.symbol for f in files))[:n]

        return symbols

    def get_preparation_status(
        self,
        asset_class: Optional[str] = None,
        timeframe: Optional[str] = None,
        symbol: Optional[str] = None
    ) -> Dict:
        """
        Check preparation status of data files.

        Returns dict with:
        - master_files: Available master data
        - train_files: Prepared training data
        - test_files: Prepared test data
        - missing_preparation: Master files without train/test splits

        Args:
            asset_class: Filter by asset class
            timeframe: Filter by timeframe
            symbol: Filter by symbol

        Returns:
            Dictionary with preparation status
        """
        # Get master files
        master_files = self.find(
            asset_class=asset_class,
            timeframe=timeframe,
            symbol=symbol,
            split='master'
        )

        # Get prepared files
        train_files = self.find(
            asset_class=asset_class,
            timeframe=timeframe,
            symbol=symbol,
            split='train'
        )

        test_files = self.find(
            asset_class=asset_class,
            timeframe=timeframe,
            symbol=symbol,
            split='test'
        )

        # Find master files without preparation
        master_keys = set(f"{f.symbol}_{f.timeframe}" for f in master_files)
        train_keys = set(f"{f.symbol}_{f.timeframe}" for f in train_files)
        test_keys = set(f"{f.symbol}_{f.timeframe}" for f in test_files)

        prepared_keys = train_keys & test_keys
        missing_keys = master_keys - prepared_keys

        missing_preparation = [f for f in master_files if f"{f.symbol}_{f.timeframe}" in missing_keys]

        return {
            "master_files": master_files,
            "train_files": train_files,
            "test_files": test_files,
            "missing_preparation": missing_preparation,
            "preparation_complete": len(missing_preparation) == 0,
            "preparation_percentage": (len(prepared_keys) / len(master_keys) * 100) if master_keys else 0
        }

    def needs_preparation(
        self,
        asset_class: Optional[str] = None,
        timeframe: Optional[str] = None
    ) -> List[DataFile]:
        """
        Get list of master files that need preparation (train/test split).

        Args:
            asset_class: Filter by asset class
            timeframe: Filter by timeframe

        Returns:
            List of master files without train/test splits
        """
        status = self.get_preparation_status(asset_class, timeframe)
        return status["missing_preparation"]

    def summary(self) -> Dict:
        """
        Get summary statistics of available data.

        Returns:
            Dictionary with summary stats including preparation status
        """
        all_files = self.discover_all()

        stats = {
            "total_files": len(all_files),
            "total_size_mb": sum(f.size_mb for f in all_files),
            "asset_classes": {},
            "timeframes": {},
            "splits": {},
            "preparation_status": {}
        }

        # Count by asset class
        for asset_class in self.get_asset_classes():
            files = self.find(asset_class=asset_class)
            prep_status = self.get_preparation_status(asset_class=asset_class)

            stats["asset_classes"][asset_class] = {
                "files": len(files),
                "symbols": len(set(f.symbol for f in files)),
                "size_mb": sum(f.size_mb for f in files),
                "prepared": prep_status["preparation_complete"],
                "preparation_%": prep_status["preparation_percentage"]
            }

        # Count by timeframe
        for timeframe in self.get_timeframes():
            files = self.find(timeframe=timeframe)
            stats["timeframes"][timeframe] = len(files)

        # Count by split
        for split in ["master", "train", "test", "prepared"]:
            files = self.find(split=split)
            stats["splits"][split] = len(files)

        # Overall preparation status
        prep_status = self.get_preparation_status()
        stats["preparation_status"] = {
            "master": len(prep_status["master_files"]),
            "train": len(prep_status["train_files"]),
            "test": len(prep_status["test_files"]),
            "missing": len(prep_status["missing_preparation"]),
            "percentage": prep_status["preparation_percentage"]
        }

        return stats

    def _parse_file(self, filepath: Path, split: str) -> Optional[DataFile]:
        """
        Parse filename to extract metadata.

        Supports formats:
        - SYMBOL_TIMEFRAME_STARTDATE_ENDDATE.csv
        - SYMBOL_TIMEFRAME.csv

        Args:
            filepath: Path to data file
            split: Data split (train, test, master, prepared)

        Returns:
            DataFile metadata or None if can't parse
        """
        filename = filepath.name

        # Detect asset class from directory structure
        asset_class = self._detect_asset_class(filepath)

        # Parse filename patterns
        # Pattern 1: SYMBOL_TIMEFRAME_STARTDATE_ENDDATE.csv
        pattern1 = r'^([A-Z0-9+\-_]+?)_([MHDWmhdw][0-9]+)_(\d{12})_(\d{12})\.csv$'
        match = re.match(pattern1, filename)

        if match:
            symbol, timeframe, start_date, end_date = match.groups()
            return DataFile(
                path=filepath,
                symbol=symbol,
                timeframe=timeframe.upper(),
                asset_class=asset_class,
                start_date=start_date,
                end_date=end_date,
                split=split
            )

        # Pattern 2: SYMBOL_TIMEFRAME.csv
        pattern2 = r'^([A-Z0-9+\-_]+?)_([MHDWmhdw][0-9]+)\.csv$'
        match = re.match(pattern2, filename)

        if match:
            symbol, timeframe = match.groups()
            return DataFile(
                path=filepath,
                symbol=symbol,
                timeframe=timeframe.upper(),
                asset_class=asset_class,
                split=split
            )

        return None

    def _detect_asset_class(self, filepath: Path) -> str:
        """
        Detect asset class from directory path.

        Args:
            filepath: Path to data file

        Returns:
            Asset class name
        """
        path_str = str(filepath).lower()

        for asset_class, variants in self.ASSET_CLASSES.items():
            for variant in variants:
                if f"/{variant}/" in path_str or f"/{variant}" in path_str:
                    return asset_class

        return "unknown"


# Convenience functions
def find_data(
    asset_class: Optional[str] = None,
    timeframe: Optional[str] = None,
    limit: Optional[int] = None,
    **kwargs
) -> List[DataFile]:
    """
    Convenience function to find data files.

    Args:
        asset_class: Filter by asset class
        timeframe: Filter by timeframe
        limit: Limit results to top N
        **kwargs: Additional filter arguments

    Returns:
        List of matching data files
    """
    discovery = DataDiscovery()
    return discovery.find(
        asset_class=asset_class,
        timeframe=timeframe,
        limit=limit,
        **kwargs
    )


def get_top_instruments(asset_class: str, n: int = 5) -> List[str]:
    """
    Get top N instruments for an asset class.

    Args:
        asset_class: Asset class to query
        n: Number of instruments to return

    Returns:
        List of top N symbols
    """
    discovery = DataDiscovery()
    return discovery.get_top_symbols(asset_class, n)


def get_all_data() -> List[DataFile]:
    """Get all available data files."""
    discovery = DataDiscovery()
    return discovery.discover_all()


def print_data_summary():
    """Print summary of available data."""
    discovery = DataDiscovery()
    stats = discovery.summary()

    print(f"\n{'='*60}")
    print("DATA SUMMARY")
    print(f"{'='*60}")
    print(f"Total Files: {stats['total_files']}")
    print(f"Total Size: {stats['total_size_mb']:.1f} MB")

    print(f"\nAsset Classes:")
    for asset_class, info in stats['asset_classes'].items():
        print(f"  {asset_class:12s}: {info['files']:3d} files, {info['symbols']:2d} symbols, {info['size_mb']:.1f} MB")

    print(f"\nTimeframes:")
    for timeframe, count in stats['timeframes'].items():
        print(f"  {timeframe}: {count} files")

    print(f"\nData Splits:")
    for split, count in stats['splits'].items():
        if count > 0:
            print(f"  {split:10s}: {count} files")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    # Demo usage
    print_data_summary()

    print("\nExamples:")
    print("-" * 60)

    # Example 1: All crypto H1 data
    files = find_data(asset_class='crypto', timeframe='H1')
    print(f"\n1. All crypto H1 data: {len(files)} files")
    for f in files[:3]:
        print(f"   {f}")

    # Example 2: Top 5 forex instruments
    top_forex = get_top_instruments('forex', n=5)
    print(f"\n2. Top 5 forex instruments: {top_forex}")

    # Example 3: Random sample
    sample = find_data(asset_class='indices', sample=3)
    print(f"\n3. Random 3 indices: {[f.symbol for f in sample]}")
