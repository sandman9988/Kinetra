"""
Test Run Isolation
==================

Isolated test run data management for reproducibility.
Extracted from data_management.py.

Features:
- Immutable data snapshots per test run
- Complete reproducibility
- Isolated results and cache
"""

import json
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class TestRunMetadata:
    """Metadata for a test run."""
    run_id: str
    timestamp: str
    test_suite: str
    instruments: List[str]
    config: Dict
    data_snapshot_path: str  # Path to immutable data snapshot
    results_path: str
    cache_path: str
    status: str = "running"  # running, completed, failed
    
    def to_dict(self) -> dict:
        return asdict(self)


class TestRunManager:
    """
    Manages isolated test runs with immutable data snapshots.
    
    Ensures complete reproducibility:
    - Each test run has isolated data snapshot
    - Results are isolated
    - Cache is per-run
    """
    
    def __init__(self, test_runs_dir: Path):
        """
        Initialize test run manager.
        
        Args:
            test_runs_dir: Directory for test runs
        """
        self.test_runs_dir = Path(test_runs_dir)
        self.test_runs_dir.mkdir(parents=True, exist_ok=True)
        
    def create_run(
        self,
        test_suite: str,
        instruments: List[str],
        config: Dict
    ) -> TestRunMetadata:
        """
        Create a new isolated test run.
        
        Args:
            test_suite: Name of test suite
            instruments: List of instruments
            config: Test configuration
            
        Returns:
            TestRunMetadata for the new run
        """
        # Generate run ID
        timestamp = datetime.now()
        run_id = f"{test_suite}_{timestamp.strftime('%Y%m%d_%H%M%S')}"
        
        # Create run directory
        run_dir = self.test_runs_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        data_dir = run_dir / 'data'
        results_dir = run_dir / 'results'
        cache_dir = run_dir / 'cache'
        
        data_dir.mkdir(exist_ok=True)
        results_dir.mkdir(exist_ok=True)
        cache_dir.mkdir(exist_ok=True)
        
        # Create metadata
        metadata = TestRunMetadata(
            run_id=run_id,
            timestamp=timestamp.isoformat(),
            test_suite=test_suite,
            instruments=instruments,
            config=config,
            data_snapshot_path=str(data_dir),
            results_path=str(results_dir),
            cache_path=str(cache_dir),
            status='running'
        )
        
        # Save metadata
        metadata_path = run_dir / 'metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata.to_dict(), f, indent=2)
            
        return metadata
        
    def load_run(self, run_id: str) -> Optional[TestRunMetadata]:
        """Load test run metadata."""
        run_dir = self.test_runs_dir / run_id
        metadata_path = run_dir / 'metadata.json'
        
        if not metadata_path.exists():
            return None
            
        with open(metadata_path, 'r') as f:
            data = json.load(f)
            return TestRunMetadata(**data)
            
    def update_status(self, run_id: str, status: str) -> None:
        """Update test run status."""
        metadata = self.load_run(run_id)
        if metadata:
            metadata.status = status
            
            run_dir = self.test_runs_dir / run_id
            metadata_path = run_dir / 'metadata.json'
            
            with open(metadata_path, 'w') as f:
                json.dump(metadata.to_dict(), f, indent=2)
                
    def list_runs(self, test_suite: Optional[str] = None) -> List[TestRunMetadata]:
        """List all test runs, optionally filtered by suite."""
        runs = []
        
        for run_dir in self.test_runs_dir.iterdir():
            if run_dir.is_dir():
                metadata = self.load_run(run_dir.name)
                if metadata:
                    if test_suite is None or metadata.test_suite == test_suite:
                        runs.append(metadata)
                        
        # Sort by timestamp (newest first)
        runs.sort(key=lambda r: r.timestamp, reverse=True)
        return runs
