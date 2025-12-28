"""
Data Management System for Kinetra

Handles:
- Master data folder structure
- Run-specific data copies
- Data preparation for training
"""

import os
import shutil
from pathlib import Path
from datetime import datetime
from typing import List, Optional
import json


class DataManager:
    """Manages data folders for training runs."""

    def __init__(self, project_root: Optional[Path] = None):
        self.project_root = project_root or Path(__file__).parent.parent
        self.data_root = self.project_root / "data"
        self.master_dir = self.data_root / "master"
        self.runs_dir = self.data_root / "runs"

        # Create directories
        self.master_dir.mkdir(parents=True, exist_ok=True)
        self.runs_dir.mkdir(parents=True, exist_ok=True)

    def get_master_files(self) -> List[Path]:
        """Get all CSV files in master folder."""
        return sorted(self.master_dir.glob("*.csv"))

    def import_to_master(self, source_files: List[Path], clean: bool = False) -> int:
        """Import files to master data folder.

        Args:
            source_files: List of CSV file paths to import
            clean: If True, delete existing master data first

        Returns:
            Number of files imported
        """
        if clean:
            for f in self.master_dir.glob("*.csv"):
                f.unlink()

        count = 0
        for src in source_files:
            if src.exists() and src.suffix == ".csv":
                dst = self.master_dir / src.name
                shutil.copy2(src, dst)
                count += 1
                print(f"  Imported: {src.name}")

        return count

    def create_run(self, name: Optional[str] = None, strategy: str = "berserker") -> Path:
        """Create a new run folder with copy of master data.

        Args:
            name: Run name (e.g., 'run1'). Auto-generated if None.
            strategy: Strategy type for folder organization

        Returns:
            Path to the new run folder
        """
        # Generate run name if not provided
        if name is None:
            existing = list(self.runs_dir.glob(f"{strategy}_run*"))
            run_num = len(existing) + 1
            name = f"{strategy}_run{run_num}"

        run_dir = self.runs_dir / name
        run_data = run_dir / "data"
        run_models = run_dir / "models"
        run_logs = run_dir / "logs"

        # Create run structure
        run_data.mkdir(parents=True, exist_ok=True)
        run_models.mkdir(parents=True, exist_ok=True)
        run_logs.mkdir(parents=True, exist_ok=True)

        # Copy master data
        for f in self.get_master_files():
            shutil.copy2(f, run_data / f.name)

        # Create run metadata
        metadata = {
            "name": name,
            "strategy": strategy,
            "created": datetime.now().isoformat(),
            "master_files": [f.name for f in self.get_master_files()],
            "status": "created",
        }
        with open(run_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"Created run: {name}")
        print(f"  Data: {len(metadata['master_files'])} files")
        print(f"  Path: {run_dir}")

        return run_dir

    def get_run(self, name: str) -> Optional[Path]:
        """Get path to an existing run."""
        run_dir = self.runs_dir / name
        if run_dir.exists():
            return run_dir
        return None

    def list_runs(self) -> List[dict]:
        """List all runs with their metadata."""
        runs = []
        for run_dir in sorted(self.runs_dir.iterdir()):
            if run_dir.is_dir():
                meta_file = run_dir / "metadata.json"
                if meta_file.exists():
                    with open(meta_file) as f:
                        meta = json.load(f)
                    runs.append(meta)
        return runs

    def clean_old_data(self):
        """Remove CSV files from project root (old location)."""
        project_csvs = list(self.project_root.glob("*.csv"))
        for f in project_csvs:
            # Don't delete if in data folder
            if "data" not in str(f):
                print(f"Removing: {f.name}")
                f.unlink()


def setup_data_structure():
    """Interactive setup of data structure."""
    dm = DataManager()

    print("=" * 60)
    print("KINETRA DATA STRUCTURE SETUP")
    print("=" * 60)

    # Check for existing CSVs in project root
    project_root = Path(__file__).parent.parent
    old_csvs = list(project_root.glob("*.csv"))

    if old_csvs:
        print(f"\nFound {len(old_csvs)} CSV files in project root:")
        for f in old_csvs:
            size_mb = f.stat().st_size / 1024 / 1024
            print(f"  - {f.name} ({size_mb:.1f} MB)")

        print("\nImporting to master data folder...")
        dm.import_to_master(old_csvs, clean=True)

        print("\nCleaning up old files...")
        dm.clean_old_data()

    # Show master data
    master_files = dm.get_master_files()
    print(f"\nMaster data folder: {dm.master_dir}")
    print(f"Files: {len(master_files)}")
    for f in master_files:
        print(f"  - {f.name}")

    # Create initial run
    print("\nCreating initial run...")
    run_dir = dm.create_run(name="berserker_run1", strategy="berserker")

    print("\n" + "=" * 60)
    print("DATA STRUCTURE READY")
    print("=" * 60)
    print(f"""
Folder structure:
  data/
    master/          <- All training data (source of truth)
    runs/
      berserker_run1/
        data/        <- Copy of master for this run
        models/      <- Model checkpoints
        logs/        <- Training logs

To create a new run:
  from kinetra.data_manager import DataManager
  dm = DataManager()
  run_dir = dm.create_run("my_run")
""")

    return dm


if __name__ == "__main__":
    setup_data_structure()
