#!/usr/bin/env python3
"""
Simple Data Backup Wrapper for Pre-Commit Hook
===============================================

Lightweight wrapper around the full backup system for use in pre-commit hooks.

Usage:
    python scripts/backup_data.py [--quiet]
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

try:
    from scripts.download.backup_data import backup_all_data

    def main():
        quiet = "--quiet" in sys.argv
        master_dir = project_root / "data" / "master"

        if not master_dir.exists():
            if not quiet:
                print("ℹ️  No data/master directory found, skipping backup")
            return 0

        try:
            if not quiet:
                print("💾 Creating data backup before commit...")
            backup_all_data(master_dir)
            if not quiet:
                print("✅ Backup complete")
            return 0
        except Exception as e:
            if not quiet:
                print(f"⚠️  Backup failed: {e}")
            # Non-fatal for pre-commit hook
            return 0

    if __name__ == "__main__":
        sys.exit(main())

except ImportError:
    # If backup system not available, just exit cleanly
    if "--quiet" not in sys.argv:
        print("⚠️  Backup system not available, skipping")
    sys.exit(0)
