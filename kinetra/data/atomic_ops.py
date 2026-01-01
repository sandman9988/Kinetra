"""
Atomic File Operations
======================

Atomic file writing operations to prevent data corruption.
Extracted from data_management.py.

Philosophy:
- All writes are atomic (temp file + rename)
- No partial writes visible to readers
- Automatic cleanup on failure
"""

import shutil
import tempfile
from pathlib import Path
from typing import Callable, Optional


class AtomicFileWriter:
    """
    Atomic file writer - writes to temp file, then moves atomically.
    
    Guarantees:
    - Either complete file or no file (no partial writes)
    - Original file preserved until write succeeds
    - Automatic cleanup on error
    
    Usage:
        with AtomicFileWriter('/path/to/file.csv') as f:
            df.to_csv(f, index=False)
    """
    
    def __init__(self, target_path: Path):
        """
        Initialize atomic writer.
        
        Args:
            target_path: Final destination path
        """
        self.target_path = Path(target_path)
        self.temp_path: Optional[Path] = None
        self.temp_fd = None
        
    def __enter__(self):
        """Create temp file in same directory as target."""
        # Create temp file in same dir (ensures same filesystem)
        self.target_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.temp_fd, temp_name = tempfile.mkstemp(
            suffix='.tmp',
            prefix=f'.{self.target_path.stem}_',
            dir=self.target_path.parent
        )
        self.temp_path = Path(temp_name)
        
        return self.temp_path
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Move temp file to target or cleanup on error."""
        try:
            if exc_type is None:
                # Success - atomic rename
                shutil.move(str(self.temp_path), str(self.target_path))
            else:
                # Error - cleanup temp file
                if self.temp_path and self.temp_path.exists():
                    self.temp_path.unlink()
        finally:
            if self.temp_fd is not None:
                import os
                try:
                    os.close(self.temp_fd)
                except OSError:
                    pass


def atomic_write(path: Path, write_func: Callable[[Path], None]) -> None:
    """
    Write file atomically using a write function.
    
    Args:
        path: Target file path
        write_func: Function that writes to path
        
    Example:
        def write_csv(p):
            df.to_csv(p, index=False)
        
        atomic_write(Path('data.csv'), write_csv)
    """
    with AtomicFileWriter(path) as temp_path:
        write_func(temp_path)
