"""
File Safety Utilities
=====================

Atomic saves, file integrity verification, and backup management:
- Atomic file writes (no partial writes)
- Checksums and integrity verification
- Automatic backups with rotation
- Recovery from corrupted files
"""

import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union


@dataclass
class FileChecksum:
    """File checksum information."""

    path: str
    algorithm: str
    checksum: str
    size: int
    modified: float


@dataclass
class BackupInfo:
    """Backup file information."""

    original_path: str
    backup_path: str
    timestamp: datetime
    size: int
    checksum: str


class AtomicWriter:
    """
    Atomic file writer - ensures files are never partially written.

    Uses write-to-temp-then-rename pattern for atomic operations.

    Usage:
        with AtomicWriter("config.json") as f:
            json.dump(data, f)
        # File is only updated if write succeeds completely
    """

    def __init__(
        self,
        path: Union[str, Path],
        mode: str = "w",
        encoding: str = "utf-8",
        backup: bool = True,
    ):
        self.path = Path(path)
        self.mode = mode
        self.encoding = encoding if "b" not in mode else None
        self.backup = backup
        self._temp_file = None
        self._file = None

    def __enter__(self):
        # Create temp file in same directory for atomic rename
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, temp_path = tempfile.mkstemp(
            dir=self.path.parent, prefix=f".{self.path.name}.", suffix=".tmp"
        )
        self._temp_path = Path(temp_path)

        # Open with appropriate mode
        if self.encoding:
            self._file = open(fd, self.mode, encoding=self.encoding)
        else:
            self._file = os.fdopen(fd, self.mode)

        return self._file

    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            self._file.close()

            if exc_type is None:
                # Success - backup original if exists, then atomic rename
                if self.backup and self.path.exists():
                    backup_path = self.path.with_suffix(self.path.suffix + ".bak")
                    shutil.copy2(self.path, backup_path)

                # Atomic rename (works on POSIX, mostly atomic on Windows)
                self._temp_path.replace(self.path)
            else:
                # Error occurred - remove temp file
                self._temp_path.unlink(missing_ok=True)
        except Exception:
            # Clean up temp file on any error
            self._temp_path.unlink(missing_ok=True)
            raise

        return False  # Don't suppress exceptions


def atomic_write(
    path: Union[str, Path],
    content: Union[str, bytes],
    backup: bool = True,
) -> bool:
    """
    Atomically write content to file.

    Args:
        path: File path
        content: Content to write (str or bytes)
        backup: Create backup of existing file

    Returns:
        True if successful
    """
    mode = "wb" if isinstance(content, bytes) else "w"
    encoding = None if isinstance(content, bytes) else "utf-8"

    try:
        with AtomicWriter(path, mode=mode, encoding=encoding, backup=backup) as f:
            f.write(content)
        return True
    except Exception:
        return False


def atomic_json_write(
    path: Union[str, Path],
    data: dict,
    indent: int = 2,
    backup: bool = True,
) -> bool:
    """Atomically write JSON data to file."""
    try:
        with AtomicWriter(path, backup=backup) as f:
            json.dump(data, f, indent=indent, default=str)
        return True
    except Exception:
        return False


class IntegrityChecker:
    """
    File integrity verification using checksums.

    Usage:
        checker = IntegrityChecker()
        checksum = checker.compute_checksum("data.csv")
        is_valid = checker.verify_checksum("data.csv", checksum)
    """

    ALGORITHMS = {"md5", "sha1", "sha256", "sha512"}

    def __init__(self, algorithm: str = "sha256"):
        if algorithm not in self.ALGORITHMS:
            raise ValueError(f"Algorithm must be one of {self.ALGORITHMS}")
        self.algorithm = algorithm

    def compute_checksum(
        self, path: Union[str, Path], chunk_size: int = 65536
    ) -> FileChecksum:
        """Compute checksum of file."""
        path = Path(path)
        hasher = hashlib.new(self.algorithm)

        with open(path, "rb") as f:
            while chunk := f.read(chunk_size):
                hasher.update(chunk)

        stat = path.stat()
        return FileChecksum(
            path=str(path),
            algorithm=self.algorithm,
            checksum=hasher.hexdigest(),
            size=stat.st_size,
            modified=stat.st_mtime,
        )

    def verify_checksum(
        self, path: Union[str, Path], expected: Union[str, FileChecksum]
    ) -> bool:
        """Verify file matches expected checksum."""
        if isinstance(expected, FileChecksum):
            expected = expected.checksum

        current = self.compute_checksum(path)
        return current.checksum == expected

    def create_manifest(
        self, directory: Union[str, Path], patterns: List[str] = None
    ) -> Dict[str, FileChecksum]:
        """Create checksum manifest for directory."""
        directory = Path(directory)
        patterns = patterns or ["*"]
        manifest = {}

        for pattern in patterns:
            for path in directory.rglob(pattern):
                if path.is_file():
                    try:
                        manifest[str(path.relative_to(directory))] = (
                            self.compute_checksum(path)
                        )
                    except (IOError, PermissionError):
                        pass

        return manifest

    def verify_manifest(
        self, directory: Union[str, Path], manifest: Dict[str, FileChecksum]
    ) -> Dict[str, str]:
        """
        Verify directory against manifest.

        Returns:
            Dict of path -> status ('ok', 'modified', 'missing', 'new')
        """
        directory = Path(directory)
        results = {}

        # Check files in manifest
        for rel_path, expected in manifest.items():
            full_path = directory / rel_path
            if not full_path.exists():
                results[rel_path] = "missing"
            elif self.verify_checksum(full_path, expected):
                results[rel_path] = "ok"
            else:
                results[rel_path] = "modified"

        # Check for new files
        for path in directory.rglob("*"):
            if path.is_file():
                rel_path = str(path.relative_to(directory))
                if rel_path not in manifest:
                    results[rel_path] = "new"

        return results

    def save_manifest(
        self, manifest: Dict[str, FileChecksum], path: Union[str, Path]
    ) -> bool:
        """Save manifest to JSON file."""
        data = {
            rel_path: {
                "checksum": cs.checksum,
                "algorithm": cs.algorithm,
                "size": cs.size,
                "modified": cs.modified,
            }
            for rel_path, cs in manifest.items()
        }
        return atomic_json_write(path, data)

    def load_manifest(self, path: Union[str, Path]) -> Dict[str, FileChecksum]:
        """Load manifest from JSON file."""
        with open(path) as f:
            data = json.load(f)

        return {
            rel_path: FileChecksum(
                path=rel_path,
                algorithm=info["algorithm"],
                checksum=info["checksum"],
                size=info["size"],
                modified=info["modified"],
            )
            for rel_path, info in data.items()
        }


class BackupManager:
    """
    Automatic backup management with rotation.

    Usage:
        backup = BackupManager("backups/")
        backup.create("important_data.csv")
        backup.restore("important_data.csv")  # Restore latest
    """

    def __init__(
        self,
        backup_dir: Union[str, Path] = "backups",
        max_backups: int = 10,
        compress: bool = False,
    ):
        self.backup_dir = Path(backup_dir)
        self.max_backups = max_backups
        self.compress = compress
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        self._checker = IntegrityChecker()

    def _get_backup_name(self, original: Path) -> str:
        """Generate backup filename with timestamp."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        suffix = ".gz" if self.compress else ""
        return f"{original.name}.{timestamp}{suffix}"

    def _get_backup_path(self, original: Path) -> Path:
        """Get full backup path."""
        # Create subdirectory structure matching original
        rel_dir = original.parent.name if original.parent != Path(".") else ""
        backup_subdir = self.backup_dir / rel_dir
        backup_subdir.mkdir(parents=True, exist_ok=True)
        return backup_subdir / self._get_backup_name(original)

    def create(self, path: Union[str, Path]) -> Optional[BackupInfo]:
        """
        Create backup of file.

        Args:
            path: File to backup

        Returns:
            BackupInfo or None if failed
        """
        path = Path(path)
        if not path.exists():
            return None

        backup_path = self._get_backup_path(path)

        try:
            if self.compress:
                import gzip

                with open(path, "rb") as f_in:
                    with gzip.open(backup_path, "wb") as f_out:
                        shutil.copyfileobj(f_in, f_out)
            else:
                shutil.copy2(path, backup_path)

            # Compute checksum
            checksum = self._checker.compute_checksum(path)

            # Rotate old backups
            self._rotate_backups(path)

            return BackupInfo(
                original_path=str(path),
                backup_path=str(backup_path),
                timestamp=datetime.now(),
                size=path.stat().st_size,
                checksum=checksum.checksum,
            )
        except Exception:
            return None

    def _rotate_backups(self, original: Path):
        """Remove old backups exceeding max_backups."""
        # Find all backups for this file
        pattern = f"{original.name}.*"
        backups = sorted(
            self.backup_dir.rglob(pattern), key=lambda p: p.stat().st_mtime, reverse=True
        )

        # Remove excess backups
        for old_backup in backups[self.max_backups :]:
            old_backup.unlink(missing_ok=True)

    def list_backups(self, path: Union[str, Path]) -> List[BackupInfo]:
        """List all backups for a file."""
        path = Path(path)
        pattern = f"{path.name}.*"
        backups = []

        for backup_path in sorted(
            self.backup_dir.rglob(pattern), key=lambda p: p.stat().st_mtime, reverse=True
        ):
            try:
                stat = backup_path.stat()
                backups.append(
                    BackupInfo(
                        original_path=str(path),
                        backup_path=str(backup_path),
                        timestamp=datetime.fromtimestamp(stat.st_mtime),
                        size=stat.st_size,
                        checksum="",  # Don't compute unless needed
                    )
                )
            except (IOError, PermissionError):
                pass

        return backups

    def restore(
        self,
        path: Union[str, Path],
        backup_index: int = 0,
        verify: bool = True,
    ) -> Tuple[bool, str]:
        """
        Restore file from backup.

        Args:
            path: Original file path
            backup_index: Which backup to restore (0 = latest)
            verify: Verify integrity after restore

        Returns:
            (success, message)
        """
        backups = self.list_backups(path)
        if not backups:
            return False, "No backups found"

        if backup_index >= len(backups):
            return False, f"Backup index {backup_index} not found"

        backup = backups[backup_index]
        backup_path = Path(backup.backup_path)
        target_path = Path(path)

        try:
            # Create backup of current file before restoring
            if target_path.exists():
                self.create(target_path)

            # Restore
            if self.compress or backup_path.suffix == ".gz":
                import gzip

                with gzip.open(backup_path, "rb") as f_in:
                    with open(target_path, "wb") as f_out:
                        shutil.copyfileobj(f_in, f_out)
            else:
                shutil.copy2(backup_path, target_path)

            msg = f"Restored from {backup_path.name}"

            # Verify if requested and checksum available
            if verify and backup.checksum:
                if self._checker.verify_checksum(target_path, backup.checksum):
                    msg += " (verified)"
                else:
                    return False, "Restored but verification failed"

            return True, msg
        except Exception as e:
            return False, f"Restore failed: {e}"

    def get_stats(self) -> Dict[str, any]:
        """Get backup statistics."""
        total_size = 0
        total_files = 0
        oldest = None
        newest = None

        for path in self.backup_dir.rglob("*"):
            if path.is_file():
                stat = path.stat()
                total_size += stat.st_size
                total_files += 1

                mtime = datetime.fromtimestamp(stat.st_mtime)
                if oldest is None or mtime < oldest:
                    oldest = mtime
                if newest is None or mtime > newest:
                    newest = mtime

        return {
            "backup_dir": str(self.backup_dir),
            "total_files": total_files,
            "total_size_mb": total_size / (1024 * 1024),
            "oldest_backup": oldest,
            "newest_backup": newest,
            "max_backups_per_file": self.max_backups,
            "compression": self.compress,
        }


class SafeFileHandler:
    """
    High-level safe file operations combining atomic writes,
    integrity checks, and backups.

    Usage:
        handler = SafeFileHandler()
        handler.safe_write("config.json", data)
        data = handler.safe_read("config.json")
    """

    def __init__(
        self,
        backup_dir: str = "backups",
        auto_backup: bool = True,
        verify_writes: bool = True,
    ):
        self.backup_manager = BackupManager(backup_dir)
        self.integrity_checker = IntegrityChecker()
        self.auto_backup = auto_backup
        self.verify_writes = verify_writes

    def safe_write(
        self,
        path: Union[str, Path],
        content: Union[str, bytes, dict],
        backup: bool = None,
    ) -> Tuple[bool, str]:
        """
        Safely write content to file.

        - Creates backup before overwriting
        - Uses atomic write
        - Verifies write integrity

        Args:
            path: File path
            content: Content to write (str, bytes, or dict for JSON)
            backup: Override auto_backup setting

        Returns:
            (success, message)
        """
        path = Path(path)
        do_backup = backup if backup is not None else self.auto_backup

        # Create backup if file exists
        if do_backup and path.exists():
            backup_info = self.backup_manager.create(path)
            if not backup_info:
                return False, "Failed to create backup"

        # Determine content type and write
        if isinstance(content, dict):
            success = atomic_json_write(path, content, backup=False)
        else:
            success = atomic_write(path, content, backup=False)

        if not success:
            return False, "Atomic write failed"

        # Verify write
        if self.verify_writes:
            try:
                if isinstance(content, dict):
                    with open(path) as f:
                        written = json.load(f)
                    if written != content:
                        return False, "Write verification failed (content mismatch)"
                else:
                    with open(path, "rb" if isinstance(content, bytes) else "r") as f:
                        written = f.read()
                    if written != content:
                        return False, "Write verification failed (content mismatch)"
            except Exception as e:
                return False, f"Write verification failed: {e}"

        return True, "Write successful"

    def safe_read(
        self,
        path: Union[str, Path],
        expected_checksum: str = None,
        as_json: bool = False,
    ) -> Tuple[Optional[Union[str, bytes, dict]], str]:
        """
        Safely read file with optional integrity verification.

        Args:
            path: File path
            expected_checksum: Verify against this checksum
            as_json: Parse as JSON

        Returns:
            (content or None, message)
        """
        path = Path(path)

        if not path.exists():
            return None, "File not found"

        # Verify checksum if provided
        if expected_checksum:
            if not self.integrity_checker.verify_checksum(path, expected_checksum):
                return None, "Checksum verification failed"

        try:
            if as_json:
                with open(path) as f:
                    return json.load(f), "Read successful"
            else:
                with open(path, "rb") as f:
                    return f.read(), "Read successful"
        except Exception as e:
            return None, f"Read failed: {e}"

    def safe_copy(
        self,
        src: Union[str, Path],
        dst: Union[str, Path],
        verify: bool = True,
    ) -> Tuple[bool, str]:
        """
        Safely copy file with integrity verification.

        Args:
            src: Source path
            dst: Destination path
            verify: Verify copy integrity

        Returns:
            (success, message)
        """
        src, dst = Path(src), Path(dst)

        if not src.exists():
            return False, "Source file not found"

        # Compute source checksum
        src_checksum = self.integrity_checker.compute_checksum(src)

        # Read and write atomically
        with open(src, "rb") as f:
            content = f.read()

        if not atomic_write(dst, content, backup=self.auto_backup):
            return False, "Copy failed"

        # Verify
        if verify:
            if not self.integrity_checker.verify_checksum(dst, src_checksum):
                dst.unlink(missing_ok=True)
                return False, "Copy verification failed"

        return True, "Copy successful"


# Convenience functions
def safe_write(path: str, content: Union[str, bytes, dict]) -> bool:
    """Convenience function for safe file write."""
    handler = SafeFileHandler(auto_backup=True)
    success, _ = handler.safe_write(path, content)
    return success


def safe_read(path: str, as_json: bool = False) -> Optional[Union[str, bytes, dict]]:
    """Convenience function for safe file read."""
    handler = SafeFileHandler()
    content, _ = handler.safe_read(path, as_json=as_json)
    return content


def create_backup(path: str, backup_dir: str = "backups") -> Optional[str]:
    """Create backup of file, return backup path."""
    manager = BackupManager(backup_dir)
    info = manager.create(path)
    return info.backup_path if info else None


def verify_integrity(path: str, expected_checksum: str = None) -> bool:
    """Verify file integrity."""
    checker = IntegrityChecker()
    if expected_checksum:
        return checker.verify_checksum(path, expected_checksum)
    else:
        # Just check file is readable and compute checksum
        try:
            checker.compute_checksum(path)
            return True
        except Exception:
            return False
