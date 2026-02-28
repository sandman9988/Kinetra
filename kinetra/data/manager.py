"""
Core Data Manager
=================

Main data management interface.
Consolidated from data_manager.py, data_management.py, unified_data_manager.py.

Features:
- Broker/account/asset class organization
- Raw data immutability (append-only)
- Training data generation
- Training run lifecycle management (create_run / get_run / list_runs)
- Integration with all submodules
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from kinetra.config import DATA_DIR

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
            base_dir = DATA_DIR

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
        currencies = ["USD", "EUR", "GBP", "JPY", "CHF", "CAD", "AUD", "NZD"]
        for c1 in currencies:
            for c2 in currencies:
                if c1 != c2 and f"{c1}{c2}" in symbol:
                    return "forex"

        # Metals
        if any(m in symbol for m in ["XAU", "XAG", "GOLD", "SILVER"]):
            return "metals"

        # Crypto
        if any(c in symbol for c in ["BTC", "ETH", "CRYPTO"]):
            return "crypto"

        # Indices
        if any(i in symbol for i in ["US30", "US500", "NAS", "DAX", "FTSE"]):
            return "indices"

        return "unknown"

    def list_available_data(self) -> Dict[str, List[str]]:
        """List all available raw data by asset class."""
        data = {}

        for asset_dir in self.raw_dir.rglob("*"):
            if asset_dir.is_dir() and asset_dir.parent.name in [
                "forex",
                "metals",
                "indices",
                "crypto",
            ]:
                asset_class = asset_dir.parent.name
                if asset_class not in data:
                    data[asset_class] = []

                # Find CSV files
                for csv_file in asset_dir.glob("*.csv"):
                    symbol = csv_file.stem.split("_")[0]
                    if symbol not in data[asset_class]:
                        data[asset_class].append(symbol)

        return data

    def prepare_training_data(
        self, symbols: List[str], timeframe: str = "H1", force_regenerate: bool = False
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

    # ------------------------------------------------------------------
    # Training-run lifecycle  (DRY-10 Phase B)
    # ------------------------------------------------------------------
    #
    # These methods replace the phantom ``create_run`` / ``get_run`` /
    # ``list_runs`` calls that training scripts (train_berserker.py,
    # train_sniper.py, …) previously made against the old
    # ``kinetra.data_manager.DataManager`` class, where the methods did
    # not actually exist.  Runs are stored under
    # ``<base_dir>/runs/<name>/``  which is one of the directories
    # scanned by ``kinetra.model_manifest.discover_model_files``.

    def create_run(
        self,
        strategy: str,
        name: Optional[str] = None,
    ) -> Path:
        """Create a new named training run directory and return its path.

        The directory is created immediately.  A ``run_meta.json`` file
        is written inside so that :meth:`list_runs` can report it.

        Args:
            strategy: Strategy name (e.g. ``"berserker"``, ``"sniper"``).
            name: Optional explicit run name.  When omitted an
                auto-generated name ``{strategy}_{YYYYMMDD_HHMMSS}``
                is used.

        Returns:
            Path to the newly-created run directory.
        """
        if name is None:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            name = f"{strategy}_{ts}"

        run_dir = self.base_dir / "runs" / name
        run_dir.mkdir(parents=True, exist_ok=True)

        # Subdirectories expected by training scripts
        for sub in ("data", "models", "logs", "checkpoints"):
            (run_dir / sub).mkdir(exist_ok=True)

        # Metadata sidecar
        meta: Dict[str, Any] = {
            "name": name,
            "strategy": strategy,
            "created_at": datetime.now().isoformat(),
        }
        with open(run_dir / "run_meta.json", "w") as fh:
            json.dump(meta, fh, indent=2)

        return run_dir

    def get_run(self, name: str) -> Optional[Path]:
        """Return the path to an existing run, or ``None`` if not found.

        Args:
            name: Run name (directory name under ``<base_dir>/runs/``).

        Returns:
            Path to the run directory, or ``None`` when it does not exist.
        """
        run_dir = self.base_dir / "runs" / name
        return run_dir if run_dir.is_dir() else None

    def list_runs(self, strategy: Optional[str] = None) -> List[Dict[str, Any]]:
        """List all training runs, newest first.

        Args:
            strategy: Optional filter — only return runs whose metadata
                ``strategy`` field matches this value.

        Returns:
            List of dicts, each with at least a ``"name"`` key.  Additional
            keys come from the ``run_meta.json`` sidecar when present.
        """
        runs_root = self.base_dir / "runs"
        if not runs_root.exists():
            return []

        runs: List[Dict[str, Any]] = []
        for run_dir in sorted(runs_root.iterdir(), reverse=True):
            if not run_dir.is_dir():
                continue

            meta_path = run_dir / "run_meta.json"
            if meta_path.exists():
                try:
                    with open(meta_path) as fh:
                        meta = json.load(fh)
                except Exception:
                    meta = {"name": run_dir.name}
            else:
                meta = {"name": run_dir.name}

            if strategy is not None and meta.get("strategy") != strategy:
                continue

            runs.append(meta)

        return runs
