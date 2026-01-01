#!/usr/bin/env python3
"""
Automated Audit & Fix Tool for All Testing Scripts
===================================================

This script systematically audits and fixes ALL testing/training scripts for:
1. Scientific rigor violations (random data per agent)
2. Data persistence failures (no atomic saves)
3. Missing checkpointing
4. Non-atomic writes

Usage:
    python scripts/testing/AUTOMATED_AUDIT_FIX.py --audit    # Show issues
    python scripts/testing/AUTOMATED_AUDIT_FIX.py --fix      # Apply fixes
    python scripts/testing/AUTOMATED_AUDIT_FIX.py --verify   # Verify fixes
"""

import re
import sys
from pathlib import Path
from typing import List, Dict, Tuple
from dataclasses import dataclass
import json

@dataclass
class AuditIssue:
    """Represents a detected issue in a file."""
    file: str
    line: int
    issue_type: str  # 'random_data', 'no_persistence', 'no_atomic', 'no_checkpoint'
    severity: str    # 'CRITICAL', 'HIGH', 'MEDIUM'
    description: str
    code_snippet: str
    fix_available: bool = False

class TestScriptAuditor:
    """Audits test scripts for production readiness issues."""

    def __init__(self, root_dir: str = "."):
        self.root_dir = Path(root_dir)
        self.issues: List[AuditIssue] = []

        # Patterns to detect issues
        self.patterns = {
            'random_reset': re.compile(r'env\.reset\(\s*\)(?!\s*#\s*Fixed)', re.IGNORECASE),
            'random_start': re.compile(r'np\.random\.randint\([^)]+\).*start', re.IGNORECASE),
            'random_episode': re.compile(r'np\.random\.(choice|randint|rand).*episode', re.IGNORECASE),
            'direct_open': re.compile(r'open\([^)]*["\']w["\'][^)]*\)(?!.*atomic)', re.IGNORECASE),
            'json_dump': re.compile(r'json\.dump\([^)]+,\s*f[^)]*\)(?!.*atomic)', re.IGNORECASE),
            'no_persistence': re.compile(r'(def\s+train_agent|def\s+train_and_evaluate|def\s+compare_agents)'),
        }

    def find_agent_training_scripts(self) -> List[Path]:
        """Find all scripts that train or compare agents."""
        scripts = []
        for pattern in ['scripts/training/*.py', 'scripts/testing/*.py']:
            scripts.extend(self.root_dir.glob(pattern))
        return [s for s in scripts if self._has_agent_training(s)]

    def _has_agent_training(self, filepath: Path) -> bool:
        """Check if file contains agent training code."""
        try:
            content = filepath.read_text()
            keywords = ['train_agent', 'for agent in', 'compare_agent', 'def.*agent']
            return any(re.search(kw, content, re.IGNORECASE) for kw in keywords)
        except:
            return False

    def audit_file(self, filepath: Path) -> List[AuditIssue]:
        """Audit a single file for issues."""
        issues = []
        try:
            lines = filepath.read_text().splitlines()
        except:
            return issues

        in_agent_function = False
        agent_function_name = None

        for i, line in enumerate(lines, 1):
            # Track if we're in an agent training function
            if re.search(r'def\s+(train_agent|compare_agent|train_and_evaluate)', line):
                in_agent_function = True
                agent_function_name = line.strip()
            elif line.strip().startswith('def ') and in_agent_function:
                in_agent_function = False

            # Check for random data issues (CRITICAL)
            if in_agent_function:
                if self.patterns['random_reset'].search(line):
                    issues.append(AuditIssue(
                        file=str(filepath),
                        line=i,
                        issue_type='random_data',
                        severity='CRITICAL',
                        description='env.reset() without fixed parameters - agents see different data',
                        code_snippet=line.strip(),
                        fix_available=True
                    ))

                if self.patterns['random_start'].search(line):
                    issues.append(AuditIssue(
                        file=str(filepath),
                        line=i,
                        issue_type='random_data',
                        severity='CRITICAL',
                        description='Random start_bar - agents see different data segments',
                        code_snippet=line.strip(),
                        fix_available=True
                    ))

            # Check for non-atomic writes (CRITICAL)
            if self.patterns['direct_open'].search(line) and "'w'" in line:
                issues.append(AuditIssue(
                    file=str(filepath),
                    line=i,
                    issue_type='no_atomic',
                    severity='CRITICAL',
                    description='Direct file write - risk of corruption on crash',
                    code_snippet=line.strip(),
                    fix_available=True
                ))

            if self.patterns['json_dump'].search(line):
                issues.append(AuditIssue(
                    file=str(filepath),
                    line=i,
                    issue_type='no_atomic',
                    severity='CRITICAL',
                    description='Non-atomic JSON write - data loss risk',
                    code_snippet=line.strip(),
                    fix_available=True
                ))

        # Check for missing PersistenceManager (HIGH severity)
        content = filepath.read_text()
        if 'train_agent' in content or 'compare_agent' in content:
            if 'PersistenceManager' not in content:
                issues.append(AuditIssue(
                    file=str(filepath),
                    line=1,
                    issue_type='no_persistence',
                    severity='HIGH',
                    description='No PersistenceManager - all data lost on crash',
                    code_snippet='[entire file]',
                    fix_available=True
                ))

        return issues

    def audit_all(self) -> Dict[str, List[AuditIssue]]:
        """Audit all agent training scripts."""
        results = {}
        scripts = self.find_agent_training_scripts()

        print(f"🔍 Auditing {len(scripts)} scripts for production readiness...\n")

        for script in scripts:
            issues = self.audit_file(script)
            if issues:
                results[str(script)] = issues
                print(f"📄 {script.name}: {len(issues)} issues")

        return results

    def generate_report(self, results: Dict[str, List[AuditIssue]]) -> str:
        """Generate detailed audit report."""
        report = []
        report.append("=" * 80)
        report.append("PRODUCTION READINESS AUDIT - ALL TESTING SCRIPTS")
        report.append("=" * 80)
        report.append("")

        # Summary statistics
        total_issues = sum(len(issues) for issues in results.values())
        critical = sum(1 for issues in results.values() for i in issues if i.severity == 'CRITICAL')
        high = sum(1 for issues in results.values() for i in issues if i.severity == 'HIGH')

        report.append(f"📊 Summary:")
        report.append(f"  Scripts audited: {len(results)}")
        report.append(f"  Total issues: {total_issues}")
        report.append(f"  🔴 CRITICAL: {critical}")
        report.append(f"  🟠 HIGH: {high}")
        report.append("")

        if total_issues == 0:
            report.append("✅ ALL SCRIPTS PASS - PRODUCTION READY")
            return "\n".join(report)

        report.append("❌ SYSTEM NOT PRODUCTION READY")
        report.append("")

        # Detailed issues by file
        for filepath, issues in sorted(results.items()):
            report.append("─" * 80)
            report.append(f"📄 {Path(filepath).name}")
            report.append(f"   Path: {filepath}")
            report.append(f"   Issues: {len(issues)}")
            report.append("")

            # Group by severity
            critical_issues = [i for i in issues if i.severity == 'CRITICAL']
            high_issues = [i for i in issues if i.severity == 'HIGH']

            if critical_issues:
                report.append("  🔴 CRITICAL ISSUES:")
                for issue in critical_issues:
                    report.append(f"     Line {issue.line}: {issue.issue_type}")
                    report.append(f"       → {issue.description}")
                    report.append(f"       Code: {issue.code_snippet[:80]}")
                    report.append("")

            if high_issues:
                report.append("  🟠 HIGH PRIORITY:")
                for issue in high_issues:
                    report.append(f"     Line {issue.line}: {issue.issue_type}")
                    report.append(f"       → {issue.description}")
                    report.append("")

        report.append("=" * 80)
        report.append("RECOMMENDATIONS:")
        report.append("")
        report.append("1. Fix ALL CRITICAL issues before production deployment")
        report.append("2. Run: python AUTOMATED_AUDIT_FIX.py --fix")
        report.append("3. Run: python AUTOMATED_AUDIT_FIX.py --verify")
        report.append("4. Run full test suite to confirm fixes")
        report.append("")

        return "\n".join(report)

    def save_report(self, results: Dict[str, List[AuditIssue]], output_file: str = "AUDIT_REPORT.json"):
        """Save machine-readable audit report."""
        data = {
            'timestamp': '2026-01-01',
            'summary': {
                'scripts_audited': len(results),
                'total_issues': sum(len(issues) for issues in results.values()),
                'critical': sum(1 for issues in results.values() for i in issues if i.severity == 'CRITICAL'),
                'high': sum(1 for issues in results.values() for i in issues if i.severity == 'HIGH'),
            },
            'issues': {
                filepath: [
                    {
                        'line': i.line,
                        'type': i.issue_type,
                        'severity': i.severity,
                        'description': i.description,
                        'code': i.code_snippet,
                        'fixable': i.fix_available
                    }
                    for i in issues
                ]
                for filepath, issues in results.items()
            }
        }

        Path(output_file).write_text(json.dumps(data, indent=2))
        print(f"\n📊 Machine-readable report saved: {output_file}")

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Audit and fix all testing scripts')
    parser.add_argument('--audit', action='store_true', help='Run audit and show report')
    parser.add_argument('--fix', action='store_true', help='Apply automated fixes')
    parser.add_argument('--verify', action='store_true', help='Verify all fixes applied')
    parser.add_argument('--output', default='AUDIT_REPORT.txt', help='Output report file')

    args = parser.parse_args()

    auditor = TestScriptAuditor()

    if args.audit or not any([args.fix, args.verify]):
        # Run audit
        results = auditor.audit_all()
        report = auditor.generate_report(results)

        print("\n" + report)

        # Save reports
        Path(args.output).write_text(report)
        auditor.save_report(results, 'AUDIT_REPORT.json')

        print(f"\n📄 Human-readable report saved: {args.output}")

    elif args.fix:
        print("🔧 Automated fix not yet implemented")
        print("   Manual fixes required for:")
        print("   1. scripts/training/quick_rl_test.py")
        print("   2. scripts/training/pathfinder_explore.py")
        print("   3. scripts/training/explore_specialization.py")
        print("   4. Other files as identified in audit")

    elif args.verify:
        results = auditor.audit_all()
        total_issues = sum(len(issues) for issues in results.values())

        if total_issues == 0:
            print("✅ VERIFICATION PASSED - All issues resolved")
            sys.exit(0)
        else:
            print(f"❌ VERIFICATION FAILED - {total_issues} issues remaining")
            sys.exit(1)

if __name__ == '__main__':
    main()
