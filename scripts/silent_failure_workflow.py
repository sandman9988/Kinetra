#!/usr/bin/env python3
"""
Silent Failure Workflow Orchestrator for Kinetra

This script orchestrates the complete workflow for detecting, analyzing, and fixing
silent failures in the codebase.

Workflow:
1. Detect failures (run all tests and collect failures)
2. Analyze failures (categorize and prioritize)
3. Fix failures (apply automated fixes)
4. Validate fixes (ensure nothing broke)
5. Report results (generate comprehensive report)

Usage:
    # Run complete workflow
    python scripts/silent_failure_workflow.py
    
    # Run in dry-run mode (no actual fixes)
    python scripts/silent_failure_workflow.py --dry-run
    
    # Run specific phases
    python scripts/silent_failure_workflow.py --detect-only
    python scripts/silent_failure_workflow.py --fix-only
    
    # Quick mode (faster, less comprehensive)
    python scripts/silent_failure_workflow.py --quick
"""

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from kinetra.silent_failure_logger import SilentFailureLogger


class WorkflowOrchestrator:
    """Orchestrates the complete silent failure workflow."""
    
    def __init__(self, dry_run: bool = False, quick_mode: bool = False):
        """
        Initialize workflow orchestrator.
        
        Args:
            dry_run: If True, don't apply actual fixes
            quick_mode: If True, run faster but less comprehensive detection
        """
        self.dry_run = dry_run
        self.quick_mode = quick_mode
        self.logger = SilentFailureLogger.get_instance()
        
        # Workflow results
        self.results = {
            "started_at": datetime.now().isoformat(),
            "phases_completed": [],
            "phases_failed": [],
            "total_failures_detected": 0,
            "total_failures_fixed": 0,
            "validation_passed": False,
        }
    
    def print_header(self, title: str):
        """Print a formatted header."""
        print(f"\n{'='*70}")
        print(f"{title:^70}")
        print(f"{'='*70}\n")
    
    def run_phase(self, phase_name: str, command: list, success_message: str) -> bool:
        """
        Run a workflow phase.
        
        Args:
            phase_name: Name of the phase
            command: Command to execute
            success_message: Message to print on success
        
        Returns:
            True if phase succeeded
        """
        self.print_header(f"Phase: {phase_name}")
        
        try:
            print(f"Running: {' '.join(command)}\n")
            result = subprocess.run(
                command,
                cwd=project_root,
                capture_output=False,
                text=True,
            )
            
            if result.returncode == 0:
                print(f"\n✓ {success_message}")
                self.results["phases_completed"].append(phase_name)
                return True
            else:
                print(f"\n✗ {phase_name} failed with code {result.returncode}")
                self.results["phases_failed"].append(phase_name)
                return False
                
        except Exception as e:
            print(f"\n✗ {phase_name} failed with error: {e}")
            self.results["phases_failed"].append(phase_name)
            return False
    
    def detect_failures(self) -> bool:
        """
        Phase 1: Detect failures.
        
        Returns:
            True if detection succeeded
        """
        mode = "quick" if self.quick_mode else "comprehensive"
        
        command = [
            sys.executable,
            str(project_root / "scripts" / "detect_silent_failures.py"),
            "--mode", mode,
        ]
        
        return self.run_phase(
            "Failure Detection",
            command,
            f"Failure detection completed ({mode} mode)"
        )
    
    def analyze_failures(self) -> bool:
        """
        Phase 2: Analyze failures.
        
        Returns:
            True if analysis succeeded
        """
        command = [
            sys.executable,
            str(project_root / "scripts" / "fix_silent_failures.py"),
            "--analyze",
        ]
        
        success = self.run_phase(
            "Failure Analysis",
            command,
            "Failure analysis completed"
        )
        
        if success:
            # Update results with detection stats
            stats = self.logger.get_statistics()
            # Safely get the value with fallback
            self.results["total_failures_detected"] = stats.get("total_unique_failures", 0)
        
        return success
    
    def fix_failures(self) -> bool:
        """
        Phase 3: Fix failures.
        
        Returns:
            True if fixes applied successfully
        """
        command = [
            sys.executable,
            str(project_root / "scripts" / "fix_silent_failures.py"),
            "--fix",
            "--auto",
        ]
        
        if self.dry_run:
            command.append("--dry-run")
        
        return self.run_phase(
            "Automated Fixing",
            command,
            "Fixes applied successfully" if not self.dry_run else "Fixes analyzed (dry run)"
        )
    
    def validate_fixes(self) -> bool:
        """
        Phase 4: Validate fixes.
        
        Returns:
            True if validation passed
        """
        if self.dry_run:
            print("\n⚠ Skipping validation in dry-run mode")
            return True
        
        command = [
            sys.executable,
            str(project_root / "scripts" / "fix_silent_failures.py"),
            "--validate",
        ]
        
        success = self.run_phase(
            "Fix Validation",
            command,
            "Validation passed"
        )
        
        self.results["validation_passed"] = success
        return success
    
    def generate_report(self):
        """Phase 5: Generate comprehensive report."""
        self.print_header("Final Report")
        
        # Export detailed report
        report_path = self.logger.export_report()
        
        # Print summary
        print(f"Workflow: {'DRY RUN' if self.dry_run else 'EXECUTED'}")
        print(f"Mode: {'Quick' if self.quick_mode else 'Comprehensive'}")
        print(f"Started: {self.results['started_at']}")
        print(f"Completed: {datetime.now().isoformat()}")
        
        print(f"\nPhases Completed: {len(self.results['phases_completed'])}")
        for phase in self.results['phases_completed']:
            print(f"  ✓ {phase}")
        
        if self.results['phases_failed']:
            print(f"\nPhases Failed: {len(self.results['phases_failed'])}")
            for phase in self.results['phases_failed']:
                print(f"  ✗ {phase}")
        
        print(f"\nFailures Detected: {self.results['total_failures_detected']}")
        print(f"Validation: {'PASSED' if self.results['validation_passed'] else 'FAILED'}")
        
        print(f"\n📊 Detailed report: {report_path}")
        
        # Get statistics
        stats = self.logger.get_statistics()
        
        print(f"\n{'='*70}")
        print("Failure Statistics")
        print(f"{'='*70}")
        
        print("\nBy Category:")
        for category, count in sorted(stats["by_category"].items(), key=lambda x: -x[1]):
            print(f"  {category:25} {count:>5}")
        
        print("\nBy Severity:")
        for severity, count in sorted(stats["by_severity"].items(), key=lambda x: -x[1]):
            print(f"  {severity:25} {count:>5}")
        
        if stats["top_failures"]:
            print(f"\nTop {min(5, len(stats['top_failures']))} Most Common Failures:")
            for i, failure in enumerate(stats["top_failures"][:5], 1):
                record = failure["record"]
                print(f"\n{i}. {record['exception_type']} (occurred {failure['count']} times)")
                print(f"   Location: {record['file_path']}:{record['line_number']}")
                print(f"   Function: {record['function_name']}")
                print(f"   Category: {record['category']}")
                print(f"   Severity: {record['severity']}")
                print(f"   Message: {record['exception_message'][:80]}...")
        
        # Next steps
        print(f"\n{'='*70}")
        print("Next Steps")
        print(f"{'='*70}")
        
        if self.dry_run:
            print("\n1. Review the analysis above")
            print("2. Run without --dry-run to apply fixes:")
            print("   python scripts/silent_failure_workflow.py")
        elif not self.results['validation_passed']:
            print("\n⚠ Validation failed!")
            print("1. Review the changes in data/backups/failure_fixes/")
            print("2. Manually verify the fixes")
            print("3. Run tests to ensure nothing broke")
        else:
            print("\n✓ All phases completed successfully!")
            print("1. Review the changes")
            print("2. Run tests: pytest tests/")
            print("3. Commit changes if tests pass")
    
    def run_workflow(
        self,
        detect_only: bool = False,
        analyze_only: bool = False,
        fix_only: bool = False,
    ):
        """
        Run the complete workflow.
        
        Args:
            detect_only: Only run detection phase
            analyze_only: Only run analysis phase
            fix_only: Only run fixing phase
        """
        self.print_header("Silent Failure Workflow")
        
        print(f"Mode: {'DRY RUN' if self.dry_run else 'LIVE'}")
        print(f"Speed: {'Quick' if self.quick_mode else 'Comprehensive'}")
        
        if detect_only:
            print("\nRunning: Detection only")
            self.detect_failures()
        elif analyze_only:
            print("\nRunning: Analysis only")
            self.analyze_failures()
        elif fix_only:
            print("\nRunning: Fixing only")
            self.fix_failures()
            self.validate_fixes()
        else:
            # Run complete workflow
            print("\nRunning: Complete workflow")
            print("  1. Detect failures")
            print("  2. Analyze failures")
            print("  3. Fix failures")
            print("  4. Validate fixes")
            print("  5. Generate report")
            
            # Phase 1: Detect
            if not self.detect_failures():
                print("\n⚠ Detection failed, stopping workflow")
                self.generate_report()
                return
            
            # Phase 2: Analyze
            if not self.analyze_failures():
                print("\n⚠ Analysis failed, stopping workflow")
                self.generate_report()
                return
            
            # Phase 3: Fix
            if not self.fix_failures():
                print("\n⚠ Fixing failed, stopping workflow")
                self.generate_report()
                return
            
            # Phase 4: Validate (skip in dry-run)
            if not self.dry_run:
                if not self.validate_fixes():
                    print("\n⚠ Validation failed!")
        
        # Phase 5: Report
        self.generate_report()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run silent failure detection and fixing workflow"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run in dry-run mode (no actual fixes applied)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run in quick mode (faster, less comprehensive)",
    )
    parser.add_argument(
        "--detect-only",
        action="store_true",
        help="Only run failure detection",
    )
    parser.add_argument(
        "--analyze-only",
        action="store_true",
        help="Only run failure analysis",
    )
    parser.add_argument(
        "--fix-only",
        action="store_true",
        help="Only run fixing (assumes detection already run)",
    )
    
    args = parser.parse_args()
    
    # Initialize orchestrator
    orchestrator = WorkflowOrchestrator(
        dry_run=args.dry_run,
        quick_mode=args.quick,
    )
    
    # Run workflow
    try:
        orchestrator.run_workflow(
            detect_only=args.detect_only,
            analyze_only=args.analyze_only,
            fix_only=args.fix_only,
        )
    except KeyboardInterrupt:
        print("\n\n⚠ Workflow interrupted by user")
        orchestrator.generate_report()
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Workflow failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
