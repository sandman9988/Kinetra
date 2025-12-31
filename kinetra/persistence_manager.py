"""
Atomic Persistence Manager
===========================

Provides crash-safe atomic file operations with automated backups.

CRITICAL SAFETY FEATURES:
1. Atomic saves (write to temp, rename - never corrupts existing file)
2. Automated timestamped backups before overwrites
3. Backup rotation (keeps last N backups)
4. Integrity validation (checksums)
5. Crash recovery (restores from backup if save failed)

Usage:
    from kinetra.persistence_manager import PersistenceManager

    pm = PersistenceManager(backup_dir="data/backups", max_backups=10)

    # Atomic save with automatic backup
    pm.atomic_save("data/master/crypto/BTCUSD_H1_20240101_20241231.csv", dataframe)

    # Restore from latest backup
    pm.restore_latest("data/master/crypto/BTCUSD_H1_20240101_20241231.csv")
"""

import os
import shutil
import hashlib
import tempfile
from pathlib import Path
from datetime import datetime
from typing import Union, Optional, Callable
import json


class PersistenceManager:
    """Crash-safe persistence with atomic saves and automated backups."""

    def __init__(
        self,
        backup_dir: Union[str, Path] = "data/backups",
        max_backups: int = 10,
        verify_checksums: bool = True
    ):
        """
        Initialize persistence manager.

        Args:
            backup_dir: Directory for backup storage
            max_backups: Maximum backups to keep per file (oldest deleted)
            verify_checksums: Whether to verify file integrity with checksums
        """
        self.backup_dir = Path(backup_dir)
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        self.max_backups = max_backups
        self.verify_checksums = verify_checksums

        # Backup manifest tracks all backups
        self.manifest_file = self.backup_dir / "backup_manifest.json"
        self.manifest = self._load_manifest()

    def _load_manifest(self) -> dict:
        """Load backup manifest."""
        if self.manifest_file.exists():
            try:
                with open(self.manifest_file, 'r') as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}

    def _save_manifest(self):
        """Save backup manifest atomically."""
        temp_file = self.manifest_file.with_suffix('.tmp')
        with open(temp_file, 'w') as f:
            json.dump(self.manifest, f, indent=2)
        temp_file.replace(self.manifest_file)

    def _compute_checksum(self, filepath: Path) -> str:
        """Compute SHA256 checksum of file."""
        sha256 = hashlib.sha256()
        with open(filepath, 'rb') as f:
            for chunk in iter(lambda: f.read(65536), b''):
                sha256.update(chunk)
        return sha256.hexdigest()

    def _create_backup(self, filepath: Path) -> Optional[Path]:
        """
        Create timestamped backup of file.

        Returns backup path, or None if file doesn't exist.
        """
        if not filepath.exists():
            return None

        # Generate backup filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"{filepath.stem}_{timestamp}{filepath.suffix}"
        backup_path = self.backup_dir / filepath.parent.name / backup_name
        backup_path.parent.mkdir(parents=True, exist_ok=True)

        # Copy file to backup
        shutil.copy2(filepath, backup_path)

        # Compute checksum if enabled
        checksum = self._compute_checksum(backup_path) if self.verify_checksums else None

        # Update manifest
        file_key = str(filepath)
        if file_key not in self.manifest:
            self.manifest[file_key] = []

        self.manifest[file_key].append({
            'backup_path': str(backup_path),
            'timestamp': timestamp,
            'checksum': checksum,
            'size_bytes': backup_path.stat().st_size
        })

        # Rotate old backups
        self._rotate_backups(file_key)
        self._save_manifest()

        return backup_path

    def _rotate_backups(self, file_key: str):
        """Delete oldest backups if exceeding max_backups."""
        backups = self.manifest.get(file_key, [])

        if len(backups) > self.max_backups:
            # Sort by timestamp, delete oldest
            backups_sorted = sorted(backups, key=lambda x: x['timestamp'])
            to_delete = backups_sorted[:len(backups) - self.max_backups]

            for backup in to_delete:
                backup_path = Path(backup['backup_path'])
                if backup_path.exists():
                    backup_path.unlink()
                backups.remove(backup)

            self.manifest[file_key] = backups

    def atomic_save(
        self,
        filepath: Union[str, Path],
        content: Union[str, bytes, object],
        writer: Optional[Callable] = None
    ) -> bool:
        """
        Atomically save content to file with automatic backup.

        Args:
            filepath: Destination file path
            content: Content to save (str, bytes, or object if writer provided)
            writer: Optional function(file_path, content) to write custom formats
                   (e.g., pandas DataFrame.to_csv)

        Returns:
            True if save successful, False otherwise

        Example:
            # Save DataFrame
            pm.atomic_save("data.csv", df, writer=lambda p, c: c.to_csv(p, index=False))

            # Save string
            pm.atomic_save("config.txt", "some text")

            # Save bytes
            pm.atomic_save("data.bin", b"binary data")
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        try:
            # Step 1: Create backup of existing file
            if filepath.exists():
                backup_path = self._create_backup(filepath)
                print(f"✅ Backup created: {backup_path}")

            # Step 2: Write to temporary file
            with tempfile.NamedTemporaryFile(
                mode='wb' if isinstance(content, bytes) else 'w',
                dir=filepath.parent,
                delete=False,
                suffix='.tmp'
            ) as tmp_file:
                temp_path = Path(tmp_file.name)

                if writer:
                    # Use custom writer (close file first)
                    tmp_file.close()
                    writer(temp_path, content)
                elif isinstance(content, bytes):
                    tmp_file.write(content)
                else:
                    tmp_file.write(str(content))

            # Step 3: Verify temp file exists and has content
            if not temp_path.exists() or temp_path.stat().st_size == 0:
                raise IOError(f"Temporary file write failed: {temp_path}")

            # Step 4: Atomic rename (replaces destination)
            temp_path.replace(filepath)

            print(f"✅ Atomic save successful: {filepath}")
            return True

        except Exception as e:
            print(f"❌ Atomic save failed: {e}")

            # Attempt to restore from backup
            if filepath.exists():
                print("⚠️  File may be corrupted, attempting restore from backup...")
                restored = self.restore_latest(filepath)
                if restored:
                    print(f"✅ Restored from backup: {filepath}")
                else:
                    print(f"❌ Could not restore from backup")

            # Clean up temp file if it exists
            if temp_path.exists():
                temp_path.unlink()

            return False

    def restore_latest(self, filepath: Union[str, Path]) -> bool:
        """
        Restore file from latest backup.

        Returns True if restore successful, False otherwise.
        """
        filepath = Path(filepath)
        file_key = str(filepath)

        backups = self.manifest.get(file_key, [])
        if not backups:
            print(f"❌ No backups found for {filepath}")
            return False

        # Get latest backup
        latest_backup = sorted(backups, key=lambda x: x['timestamp'])[-1]
        backup_path = Path(latest_backup['backup_path'])

        if not backup_path.exists():
            print(f"❌ Backup file missing: {backup_path}")
            return False

        try:
            # Verify checksum if enabled
            if self.verify_checksums and latest_backup['checksum']:
                current_checksum = self._compute_checksum(backup_path)
                if current_checksum != latest_backup['checksum']:
                    print(f"⚠️  Backup checksum mismatch! File may be corrupted.")
                    return False

            # Copy backup to destination
            shutil.copy2(backup_path, filepath)
            print(f"✅ Restored from backup: {backup_path} → {filepath}")
            return True

        except Exception as e:
            print(f"❌ Restore failed: {e}")
            return False

    def list_backups(self, filepath: Union[str, Path]) -> list:
        """List all backups for a file."""
        file_key = str(Path(filepath))
        return self.manifest.get(file_key, [])

    def cleanup_old_backups(self, days_old: int = 30):
        """Delete backups older than specified days."""
        cutoff = datetime.now().timestamp() - (days_old * 86400)

        for file_key, backups in list(self.manifest.items()):
            for backup in list(backups):
                backup_time = datetime.strptime(backup['timestamp'], "%Y%m%d_%H%M%S").timestamp()

                if backup_time < cutoff:
                    backup_path = Path(backup['backup_path'])
                    if backup_path.exists():
                        backup_path.unlink()
                    backups.remove(backup)
                    print(f"🗑️  Deleted old backup: {backup_path}")

            # Remove file key if no backups left
            if not backups:
                del self.manifest[file_key]

        self._save_manifest()


# Global persistence manager instance
_persistence_manager = None

def get_persistence_manager(**kwargs) -> PersistenceManager:
    """Get global PersistenceManager instance (singleton)."""
    global _persistence_manager
    if _persistence_manager is None:
        _persistence_manager = PersistenceManager(**kwargs)
    return _persistence_manager
