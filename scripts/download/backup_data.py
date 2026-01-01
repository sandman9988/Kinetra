#!/usr/bin/env python3
"""
Automated Data Backup System
==============================

Backs up all data/master CSV files with atomic safety.

Usage:
    # Backup all master data
    python scripts/backup_data.py

    # Restore from latest backup
    python scripts/backup_data.py --restore

    # Cleanup old backups (>30 days)
    python scripts/backup_data.py --cleanup --days 30
"""

import sys
from pathlib import Path
import argparse

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from kinetra.persistence_manager import get_persistence_manager


def backup_all_data(master_dir: Path):
    """Backup all CSV files in master directory."""
    pm = get_persistence_manager(backup_dir="data/backups", max_backups=10)

    csv_files = sorted(master_dir.glob("*.csv"))

    if not csv_files:
        print(f"⚠️  No CSV files found in {master_dir}")
        return

    print(f"\n📦 Backing up {len(csv_files)} files from {master_dir}...")

    success_count = 0
    for csv_file in csv_files:
        try:
            # Read and immediately write back (creates backup)
            with open(csv_file, 'rb') as f:
                content = f.read()

            if pm.atomic_save(csv_file, content):
                success_count += 1
                print(f"  ✅ {csv_file.name}")
        except Exception as e:
            print(f"  ❌ {csv_file.name}: {e}")

    print(f"\n✅ Backed up {success_count}/{len(csv_files)} files")
    print(f"📁 Backups stored in: data/backups/")


def restore_all_data(master_dir: Path):
    """Restore all files from latest backups."""
    pm = get_persistence_manager(backup_dir="data/backups", max_backups=10)

    # Find all files that have backups
    files_to_restore = []
    for file_key in pm.manifest.keys():
        filepath = Path(file_key)
        if filepath.parent == master_dir:
            files_to_restore.append(filepath)

    if not files_to_restore:
        print("⚠️  No backup files found")
        return

    print(f"\n🔄 Restoring {len(files_to_restore)} files...")

    success_count = 0
    for filepath in files_to_restore:
        if pm.restore_latest(filepath):
            success_count += 1

    print(f"\n✅ Restored {success_count}/{len(files_to_restore)} files")


def cleanup_old_backups(days_old: int):
    """Delete backups older than specified days."""
    pm = get_persistence_manager(backup_dir="data/backups", max_backups=10)

    print(f"\n🗑️  Cleaning up backups older than {days_old} days...")
    pm.cleanup_old_backups(days_old=days_old)
    print("✅ Cleanup complete")


def main():
    parser = argparse.ArgumentParser(description="Automated data backup system")
    parser.add_argument("--restore", action="store_true", help="Restore from latest backups")
    parser.add_argument("--cleanup", action="store_true", help="Delete old backups")
    parser.add_argument("--days", type=int, default=30, help="Days threshold for cleanup (default: 30)")
    parser.add_argument("--dir", type=str, default="data/master", help="Data directory (default: data/master)")

    args = parser.parse_args()

    master_dir = Path(args.dir)

    if args.restore:
        restore_all_data(master_dir)
    elif args.cleanup:
        cleanup_old_backups(args.days)
    else:
        # Default: backup all data
        backup_all_data(master_dir)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Backup interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
