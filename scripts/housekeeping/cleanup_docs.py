#!/usr/bin/env python3
"""
Documentation Housekeeping Script
==================================

Cleans up excessive markdown files, moving temporary/duplicate docs to archive.

Keeps only essential documentation:
- README.md (main project docs)
- AGENT_RULES_MASTER.md (canonical rules)
- INSTALL.md (setup guide)
- VERSION.md (version tracking)
- .github/ files (CI/CD, copilot instructions)
- docs/ directory (organized documentation)

Moves to archive/:
- Status reports (SESSION_*, STATUS_*, COMMIT_*, etc.)
- Action plans (ACTION_*, NEXT_STEPS_*, etc.)
- Temporary summaries (*_SUMMARY.md, *_COMPLETE.md)
- Audit reports (*_AUDIT.md, *_REPORT.md)
- Implementation docs (*_IMPLEMENTATION*.md)
- Workflow/planning docs (WORKFLOW_*, PLAN_*, GUIDE_*)

Usage:
    python scripts/housekeeping/cleanup_docs.py --dry-run
    python scripts/housekeeping/cleanup_docs.py --execute

__version__ = "1.0.0"
"""

import argparse
import shutil
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

# Essential files to keep in root
KEEP_IN_ROOT = {
    "README.md",
    "AGENT_RULES_MASTER.md",
    "INSTALL.md",
    "VERSION.md",
    "Makefile",
    "pyproject.toml",
    "requirements.txt",
    "Dockerfile",
    "Dockerfile.rocm",
    ".gitignore",
    ".env.example",
}

# Patterns for files to archive (temporary/duplicate/status)
ARCHIVE_PATTERNS = [
    # Status reports
    ("SESSION_", "session-reports"),
    ("STATUS_", "status-reports"),
    ("COMMIT_", "status-reports"),
    ("DEPLOYMENT_", "status-reports"),
    ("FINAL_", "status-reports"),
    ("PRODUCTION_", "status-reports"),
    ("MORNING_", "status-reports"),
    
    # Action plans
    ("ACTION_", "action-plans"),
    ("NEXT_STEPS", "action-plans"),
    ("IMMEDIATE_", "action-plans"),
    
    # Summaries & completions
    ("_SUMMARY", "summaries"),
    ("_COMPLETE", "summaries"),
    
    # Audits & reports
    ("_AUDIT", "audits"),
    ("_REPORT", "audits"),
    ("_FINDINGS", "audits"),
    
    # Implementation docs
    ("_IMPLEMENTATION", "implementation-notes"),
    ("_EXECUTION", "implementation-notes"),
    
    # Plans & guides
    ("_PLAN", "planning"),
    ("_GUIDE", "planning"),
    ("WORKFLOW_", "planning"),
    ("BRANCH_", "planning"),
    
    # Testing artifacts
    ("EXHAUSTIVE_TESTING_", "testing-artifacts"),
    ("TEST_RESULTS", "testing-artifacts"),
    ("TESTING_", "testing-artifacts"),
    
    # Consolidation artifacts
    ("CONSOLIDATION_", "consolidation"),
    ("VECTORIZATION_", "consolidation"),
    
    # Quick refs (move to docs/)
    ("QUICK_", "quick-refs"),
    ("QUICKREF", "quick-refs"),
    ("QUICKSTART", "quick-refs"),
    
    # Integration/menu
    ("MENU_", "menu-system"),
    ("INTEGRATION_", "integration"),
    ("DATA_", "data-management"),
    
    # Exploration/features
    ("EXPLORATION_", "features"),
    ("FEATURE_", "features"),
    ("ENHANCEMENT_", "features"),
    
    # Meta-documentation
    ("DENOISE_DRL_", "denoise-drl"),
    ("METAAPI_", "metaapi"),
    ("RULES_", "rules"),
]


def categorize_file(filename: str) -> Tuple[bool, str]:
    """
    Determine if file should be archived and which subdirectory.
    
    Returns:
        (should_archive, archive_subdir)
    """
    # Keep essential files
    if filename in KEEP_IN_ROOT:
        return False, ""
    
    # Check patterns
    for pattern, subdir in ARCHIVE_PATTERNS:
        if pattern in filename.upper():
            return True, subdir
    
    # Default: unknown .md files should be reviewed
    if filename.endswith('.md'):
        return True, "misc"
    
    return False, ""


def find_archivable_files(root_dir: Path) -> List[Tuple[Path, str]]:
    """Find all files that should be archived."""
    archivable = []
    
    for filepath in root_dir.glob("*.md"):
        if filepath.name in KEEP_IN_ROOT:
            continue
        
        should_archive, subdir = categorize_file(filepath.name)
        if should_archive:
            archivable.append((filepath, subdir))
    
    # Also check for specific non-MD files
    for pattern in ["*.txt", "*.sh", "*.log", "*.json"]:
        for filepath in root_dir.glob(pattern):
            if filepath.name in KEEP_IN_ROOT:
                continue
            
            # Archive temp files
            if any(x in filepath.name.upper() for x in 
                   ["SESSION", "STATUS", "COMMIT", "DIAGNOSTIC", "LINT_REPORT",
                    "VECTORIZATION", "FIX_BATCH", "EXHAUSTIVE_TEST"]):
                archivable.append((filepath, "temp-files"))
    
    return archivable


def create_archive_manifest(archive_dir: Path, archived_files: List[Tuple[Path, str]]):
    """Create manifest of archived files."""
    manifest_path = archive_dir / "ARCHIVE_MANIFEST.md"
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Group by subdirectory
    by_subdir = {}
    for filepath, subdir in archived_files:
        by_subdir.setdefault(subdir, []).append(filepath)
    
    content = f"""# Archive Manifest
**Generated:** {timestamp}

This directory contains temporary, duplicate, and historical documentation
that has been archived from the project root for cleanliness.

## Essential Documentation Location

All active documentation is now in:
- `/README.md` - Main project documentation
- `/AGENT_RULES_MASTER.md` - Canonical rules (single source of truth)
- `/INSTALL.md` - Setup and installation guide
- `/VERSION.md` - Version tracking
- `/docs/` - Organized documentation by category

## Archived Files

Total archived: {len(archived_files)} files

"""
    
    for subdir in sorted(by_subdir.keys()):
        files = sorted(by_subdir[subdir], key=lambda p: p.name)
        content += f"\n### {subdir}/ ({len(files)} files)\n\n"
        for filepath in files:
            content += f"- `{filepath.name}`\n"
    
    content += """
## Restoration

To restore a file:
```bash
# Find the file
find archive/ -name "FILENAME"

# Copy back to root
cp archive/path/to/FILENAME ./
```

## Cleanup Policy

Files in archive/ are retained for historical reference but are not
actively maintained. They may be permanently deleted in future cleanup cycles.
"""
    
    manifest_path.write_text(content)
    print(f"✅ Created manifest: {manifest_path}")


def archive_files(files: List[Tuple[Path, str]], archive_base: Path, dry_run: bool = False):
    """Move files to archive directory."""
    if dry_run:
        print("\n🔍 DRY RUN - Files that would be archived:\n")
    else:
        print("\n📦 Archiving files...\n")
    
    archived = []
    
    for filepath, subdir in sorted(files, key=lambda x: (x[1], x[0].name)):
        archive_dir = archive_base / subdir
        dest = archive_dir / filepath.name
        
        if dry_run:
            print(f"  Would move: {filepath.name} → archive/{subdir}/")
        else:
            archive_dir.mkdir(parents=True, exist_ok=True)
            
            # Check if destination exists
            if dest.exists():
                # Rename with timestamp
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                dest = archive_dir / f"{filepath.stem}_{timestamp}{filepath.suffix}"
            
            shutil.move(str(filepath), str(dest))
            print(f"  ✅ {filepath.name} → archive/{subdir}/")
            archived.append((filepath, subdir))
    
    return archived


def main():
    parser = argparse.ArgumentParser(description="Clean up documentation files")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without making changes"
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Execute the cleanup (required to make changes)"
    )
    
    args = parser.parse_args()
    
    if not args.dry_run and not args.execute:
        print("❌ Must specify --dry-run or --execute")
        print("   Use --dry-run to preview changes first")
        return 1
    
    # Setup paths
    project_root = Path(__file__).parent.parent.parent
    archive_base = project_root / "archive" / "documentation"
    
    print("=" * 80)
    print("KINETRA DOCUMENTATION HOUSEKEEPING")
    print("=" * 80)
    print(f"\nProject root: {project_root}")
    print(f"Archive location: {archive_base}")
    
    # Find files to archive
    files_to_archive = find_archivable_files(project_root)
    
    print(f"\nFound {len(files_to_archive)} files to archive")
    
    if not files_to_archive:
        print("\n✅ No files need archiving - directory is clean!")
        return 0
    
    # Archive files
    archived = archive_files(files_to_archive, archive_base, dry_run=args.dry_run)
    
    if args.execute and archived:
        # Create manifest
        create_archive_manifest(archive_base, archived)
        
        print(f"\n✅ Successfully archived {len(archived)} files to archive/documentation/")
        print("\nEssential documentation remains in:")
        print("  - README.md")
        print("  - AGENT_RULES_MASTER.md")
        print("  - INSTALL.md")
        print("  - VERSION.md")
        print("  - docs/")
    elif args.dry_run:
        print(f"\n🔍 DRY RUN COMPLETE - {len(files_to_archive)} files would be archived")
        print("   Run with --execute to perform the cleanup")
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
