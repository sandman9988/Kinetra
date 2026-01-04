#!/usr/bin/env python3
"""
Clean Up METAAPI Environment Variables from .bashrc
====================================================

This script removes METAAPI_TOKEN and METAAPI_ACCOUNT_ID environment
variable exports from ~/.bashrc to prevent them from overriding the
.env file values.

Features:
- Backs up .bashrc before making changes
- Removes all lines that export METAAPI_TOKEN or METAAPI_ACCOUNT_ID
- Preserves all other content
- Shows what was removed
- Safe and reversible

Usage:
    python scripts/cleanup_bashrc_metaapi.py

Safety:
    - Creates backup at ~/.bashrc.backup.TIMESTAMP
    - Only removes METAAPI-related exports
    - Can be run multiple times safely (idempotent)
"""

import re
import shutil
import sys
from datetime import datetime
from pathlib import Path


def print_header(text: str):
    """Print section header."""
    print(f"\n{'=' * 80}")
    print(f"  {text}")
    print(f"{'=' * 80}\n")


def backup_file(filepath: Path) -> Path:
    """
    Create a timestamped backup of the file.

    Args:
        filepath: Path to file to backup

    Returns:
        Path to backup file
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = filepath.parent / f"{filepath.name}.backup.{timestamp}"

    shutil.copy2(filepath, backup_path)
    return backup_path


def clean_bashrc(bashrc_path: Path, dry_run: bool = False) -> tuple:
    """
    Remove METAAPI environment variable exports from .bashrc.

    Args:
        bashrc_path: Path to .bashrc file
        dry_run: If True, don't actually modify the file

    Returns:
        Tuple of (lines_removed, removed_content_list)
    """
    if not bashrc_path.exists():
        return 0, []

    # Read current content
    with open(bashrc_path, "r") as f:
        lines = f.readlines()

    # Pattern to match METAAPI exports
    # Matches: export METAAPI_TOKEN=... or export METAAPI_ACCOUNT_ID=...
    metaapi_pattern = re.compile(
        r"^\s*export\s+(METAAPI_TOKEN|METAAPI_ACCOUNT_ID)\s*=.*$", re.IGNORECASE
    )

    # Filter out METAAPI lines
    removed_lines = []
    cleaned_lines = []

    for line in lines:
        if metaapi_pattern.match(line):
            removed_lines.append(line.rstrip())
        else:
            cleaned_lines.append(line)

    # Write back if not dry run
    if not dry_run and removed_lines:
        with open(bashrc_path, "w") as f:
            f.writelines(cleaned_lines)

    return len(removed_lines), removed_lines


def main():
    """Main entry point."""
    print_header("Clean Up METAAPI Environment Variables from .bashrc")

    # Get .bashrc path
    home = Path.home()
    bashrc_path = home / ".bashrc"

    if not bashrc_path.exists():
        print(f"❌ .bashrc file not found at: {bashrc_path}")
        print("\n💡 This script is designed for bash shells.")
        print("   If you use zsh, edit ~/.zshrc instead")
        print("   If you use fish, edit ~/.config/fish/config.fish instead")
        return 1

    print(f"📄 Found .bashrc at: {bashrc_path}")
    print(f"   File size: {bashrc_path.stat().st_size} bytes")

    # Dry run first to show what would be removed
    print("\n🔍 Scanning for METAAPI environment variables...")
    num_lines, removed_lines = clean_bashrc(bashrc_path, dry_run=True)

    if num_lines == 0:
        print("\n✅ No METAAPI environment variables found in .bashrc")
        print("   Nothing to clean up!")
        return 0

    # Show what will be removed
    print(f"\n⚠️  Found {num_lines} line(s) to remove:")
    print("─" * 80)
    for i, line in enumerate(removed_lines, 1):
        print(f"  {i}. {line}")
    print("─" * 80)

    # Ask for confirmation
    print("\n❓ These lines will be removed from your .bashrc")
    print("   A backup will be created first.")
    print()

    response = input("Continue? (y/N): ").strip().lower()

    if response != "y":
        print("\n❌ Cancelled - no changes made")
        return 0

    # Create backup
    print("\n📦 Creating backup...")
    try:
        backup_path = backup_file(bashrc_path)
        print(f"✅ Backup created: {backup_path}")
    except Exception as e:
        print(f"❌ Failed to create backup: {e}")
        return 1

    # Clean the file
    print("\n🧹 Cleaning .bashrc...")
    try:
        num_removed, _ = clean_bashrc(bashrc_path, dry_run=False)
        print(f"✅ Removed {num_removed} line(s)")
    except Exception as e:
        print(f"❌ Failed to clean .bashrc: {e}")
        print(f"\n💡 Restore from backup: cp {backup_path} {bashrc_path}")
        return 1

    # Success message
    print_header("Cleanup Complete!")

    print("✅ Successfully removed METAAPI environment variables from .bashrc")
    print(f"\n📋 Summary:")
    print(f"   Lines removed: {num_removed}")
    print(f"   Backup location: {backup_path}")

    print("\n⚠️  IMPORTANT: Reload your shell to apply changes:")
    print("   source ~/.bashrc")
    print("\nOr close and reopen your terminal.")

    print("\n💡 From now on:")
    print("   - METAAPI credentials will be read from .env file only")
    print("   - Use Menu 1 → 1 to configure credentials")
    print("   - .env file is gitignored (safer than environment variables)")

    print("\n🔄 To undo this change:")
    print(f"   cp {backup_path} ~/.bashrc")
    print("   source ~/.bashrc")

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\n❌ Cancelled by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
