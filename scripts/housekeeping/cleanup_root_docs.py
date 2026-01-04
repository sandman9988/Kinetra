#!/usr/bin/env python3
"""
Cleanup Root Documentation
Consolidates scattered markdown files into organized structure.

Version: 1.0.0
"""

import shutil
from datetime import datetime
from pathlib import Path


def main():
    """Execute root documentation cleanup."""
    root = Path("/home/renierdejager/Kinetra")
    archive_dir = root / "archive" / "session_reports" / f"cleanup_{datetime.now().strftime('%Y%m%d')}"
    archive_dir.mkdir(parents=True, exist_ok=True)

    # Files to archive (session-specific status reports)
    to_archive = [
        "ACTIONS_COMPLETE_SUMMARY.txt",
        "CI_MONITORING.md",
        "CI_TEST_RESULTS.md",
        "COMMIT_READY.md",
        "CONSOLIDATION_EXECUTE.sh",
        "CONSOLIDATION_EXECUTION_PLAN.md",
        "CONSOLIDATION_PLAN.md",
        "DATA_DISCOVERY_INTEGRATION.md",
        "DATA_SCRIPT_CONSOLIDATION.md",
        "DEPLOYMENT_SUCCESS.md",
        "DEPLOYMENT_SUMMARY_2026-01-03.md",
        "ENHANCEMENT_SUMMARY.md",
        "EXHAUSTIVE_TESTING_ACTION_PLAN.md",
        "EXHAUSTIVE_TESTING_IMPLEMENTATION_SUMMARY.md",
        "EXHAUSTIVE_TESTING_PATCH_SUMMARY.md",
        "EXHAUSTIVE_TESTING_QUICKREF.md",
        "EXHAUSTIVE_TESTING_QUICKSTART.md",
        "EXHAUSTIVE_TESTING_VALIDATION.md",
        "FEATURE_COMPLETION_SUMMARY.md",
        "FINAL_STATUS.md",
        "IMMEDIATE_ACTIONS_COMPLETE.md",
        "LINTER_FIX_PLAN.md",
        "MENU_AUDIT_FINDINGS.md",
        "MENU_IMPROVEMENTS_PLAN.md",
        "MENU_UX_INTEGRATION_GUIDE.md",
        "MORNING_READINESS.md",
        "MORNING_TESTING_GUIDE.md",
        "NEXT_STEPS.md",
        "NEXT_STEPS_SUMMARY.md",
        "PRODUCTION_READY_SUMMARY.md",
        "PROJECT_AUDIT_REPORT.md",
        "PROJECT_CONSOLIDATION_MASTER_PLAN.md",
        "REMOVE_PLACEHOLDERS_NOW.txt",
        "RULES_ENFORCEMENT_REPORT.md",
        "SCRIPT_CONSOLIDATION_ANALYSIS.md",
        "SESSION_COMPLETION_STATUS.md",
        "SESSION_DATA_ARCHITECTURE_COMPLETE.md",
        "SESSION_DELIVERABLES_INDEX.md",
        "SESSION_E2E_TESTING_COMPLETE.md",
        "SESSION_SUMMARY.txt",
        "STATUS_REPORT_ALL_ACTIONS.md",
        "TEST_RESULTS_AND_FIXES.md",
        "TEST_RESULTS_COMPLETE.md",
        "VECTORIZATION_ACTION_PLAN.md",
        "VECTORIZATION_FIXES_SUMMARY.md",
        "VERSIONING_SUMMARY.md",
        "WORKFLOW_AUDIT.md",
        "WORKFLOW_DATA_PATHS.md",
    ]

    # Archive files
    archived_count = 0
    for filename in to_archive:
        source = root / filename
        if source.exists():
            dest = archive_dir / filename
            shutil.move(str(source), str(dest))
            archived_count += 1
            print(f"✓ Archived: {filename}")

    # Create archive manifest
    manifest = archive_dir / "MANIFEST.md"
    with open(manifest, 'w') as f:
        f.write(f"""# Archive Manifest - {datetime.now().strftime('%Y-%m-%d')}

**Files Archived**: {archived_count}  
**Reason**: Root directory cleanup - moved session-specific status reports

## What Was Archived

Session status reports, completion summaries, and temporary planning documents
that served their purpose and are no longer needed in the root directory.

## What Remains in Root

- `README.md` - Project overview
- `AGENT_RULES_MASTER.md` - Canonical rules (never move)
- `ACTION_ITEMS.md` - Current action items
- `DOCS_INDEX.md` - Documentation index
- `EXPLORATION_LAB_DESIGN.md` - Core design doc
- `QUICK_REFERENCE.md` - Developer quick ref
- `QUICK_START_WORKFLOW.md` - Getting started guide
- `VERSION.md` - Version tracking
- Various guides (VECTORIZATION, TESTING, INSTALL, etc.)

## Files Archived

""")
        for filename in sorted(to_archive):
            f.write(f"- {filename}\n")

    print(f"\n✅ Cleanup complete: {archived_count} files archived to {archive_dir.relative_to(root)}")
    print(f"📋 Manifest created: {manifest.relative_to(root)}")


if __name__ == "__main__":
    main()
