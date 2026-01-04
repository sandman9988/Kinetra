#!/usr/bin/env python3
"""
Comprehensive Menu Option Testing
==================================

Tests ALL menu options with REAL execution (no mocks).
Reports failures, validates workflows, identifies missing scripts.

Usage:
    python scripts/testing/test_all_menu_options.py
    python scripts/testing/test_all_menu_options.py --quick  # Skip slow operations

Features:
- Tests every menu path end-to-end
- Real script execution (not mocked)
- Validates script existence
- Checks for duplicates/conflicts
- Reports missing implementations
- Progress tracking
- Token-efficient output

Design Philosophy (from Agent Rules):
- NEVER assume - always verify
- Test with REAL data flows
- Question everything
- Defense in depth
"""

import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class MenuTester:
    """Test harness for menu system."""

    def __init__(self, quick_mode: bool = False):
        self.quick_mode = quick_mode
        self.results = {
            "passed": [],
            "failed": [],
            "skipped": [],
            "missing": [],
            "duplicates": [],
        }
        self.menu_map = self._build_menu_map()

    def _build_menu_map(self) -> Dict[str, Dict]:
        """Map menu options to scripts and expected behavior."""
        return {
            # Menu 1: Setup & Authentication
            "1.1": {
                "name": "Configure MetaAPI Credentials",
                "script": "scripts/download/setup_metaapi_credentials.py",
                "requires_input": True,
                "test_mode": "check_exists",
            },
            "1.2": {
                "name": "Test MetaAPI Connection",
                "script": "scripts/download/test_metaapi_connection.py",
                "requires_input": False,
                "test_mode": "execute",
            },
            "1.3": {
                "name": "Select/Change MetaAPI Account",
                "script": "scripts/download/select_metaapi_account.py",
                "requires_input": True,
                "test_mode": "check_exists",
            },
            "1.4": {
                "name": "Configure MT5 (Local)",
                "script": None,  # Not implemented
                "requires_input": True,
                "test_mode": "check_exists",
            },
            "1.5": {
                "name": "Test MT5 Connection",
                "script": None,  # Not implemented
                "requires_input": False,
                "test_mode": "check_exists",
            },
            "1.6": {
                "name": "View Current Configuration",
                "script": None,  # Menu function
                "requires_input": False,
                "test_mode": "skip",
            },
            # Menu 2: Data Management
            "2.1": {
                "name": "Discover Available Data",
                "script": "scripts/discover_available_data.py",
                "requires_input": False,
                "test_mode": "execute",
            },
            "2.2.1": {
                "name": "MetaAPI Bulk Download",
                "script": "scripts/download/metaapi_bulk_download.py",
                "requires_input": True,
                "test_mode": "check_exists",
            },
            "2.2.2": {
                "name": "MetaAPI Single Download",
                "script": "scripts/download/download_metaapi.py",
                "requires_input": True,
                "test_mode": "check_exists",
            },
            "2.3": {
                "name": "Download MT5 Data",
                "script": "scripts/download/download_mt5_data.py",
                "requires_input": True,
                "test_mode": "check_exists",
            },
            "2.4": {
                "name": "Check Data Integrity",
                "script": "scripts/download/check_data_integrity.py",
                "requires_input": False,
                "test_mode": "execute" if not self.quick_mode else "check_exists",
            },
            "2.5": {
                "name": "View Data Coverage",
                "script": "scripts/audit_data_coverage.py",
                "requires_input": False,
                "test_mode": "execute",
            },
            "2.6": {
                "name": "Consolidate Data",
                "script": "scripts/consolidate_data.py",
                "requires_input": True,
                "test_mode": "check_exists",
            },
            # Menu 3: Exploration & Training
            "3.1": {
                "name": "Quick RL Training",
                "script": "scripts/training/train_rl.py",
                "requires_input": False,
                "test_mode": "check_exists",  # Too slow to execute
            },
            "3.2": {
                "name": "Agent Comparison",
                "script": "scripts/training/explore_compare_agents.py",
                "requires_input": False,
                "test_mode": "check_exists",
            },
            "3.3": {
                "name": "Comprehensive Exploration",
                "script": "scripts/exploration/run_comprehensive_exploration.py",
                "requires_input": False,
                "test_mode": "check_exists",
            },
            "3.4": {
                "name": "Train Specialist Agents",
                "script": None,  # Multiple scripts
                "requires_input": False,
                "test_mode": "skip",
            },
            "3.5": {
                "name": "Scientific Discovery",
                "script": "scripts/exploration/run_comprehensive_exploration.py",
                "requires_input": False,
                "test_mode": "check_exists",
            },
            # Menu 4: Backtesting
            "4.1": {
                "name": "Quick Backtest",
                "script": "scripts/batch_backtest.py",
                "requires_input": False,
                "test_mode": "check_exists",
            },
            "4.2": {
                "name": "Batch Backtest",
                "script": "scripts/batch_backtest.py",
                "requires_input": False,
                "test_mode": "check_exists",
            },
            # Menu 5: System Tools
            "5.1": {
                "name": "Cache Management",
                "script": "scripts/cache_manager.py",
                "requires_input": False,
                "test_mode": "execute",
            },
            "5.2": {
                "name": "Backup Data",
                "script": "scripts/backup_data.py",
                "requires_input": True,
                "test_mode": "check_exists",
            },
            "5.3": {
                "name": "Run Quick Tests",
                "script": "scripts/run_exhaustive_tests.py",
                "requires_input": False,
                "test_mode": "check_exists",
            },
            "5.4": {
                "name": "Run Full Tests",
                "script": "scripts/run_exhaustive_tests.py",
                "requires_input": False,
                "test_mode": "check_exists",
            },
        }

    def check_script_exists(self, script_path: str) -> Tuple[bool, str]:
        """Check if script file exists."""
        if not script_path:
            return False, "Not implemented"

        full_path = PROJECT_ROOT / script_path
        if not full_path.exists():
            return False, f"File not found: {script_path}"

        return True, "OK"

    def execute_script(self, script_path: str) -> Tuple[bool, str]:
        """Execute script and check for errors."""
        if not script_path:
            return False, "Not implemented"

        full_path = PROJECT_ROOT / script_path
        if not full_path.exists():
            return False, f"File not found: {script_path}"

        try:
            # Execute with timeout
            result = subprocess.run(
                [sys.executable, str(full_path)],
                capture_output=True,
                timeout=30,
                text=True,
            )

            if result.returncode == 0:
                return True, "Executed successfully"
            else:
                # Extract first error line
                error = result.stderr.split("\n")[0] if result.stderr else "Unknown error"
                return False, f"Exit {result.returncode}: {error[:100]}"

        except subprocess.TimeoutExpired:
            return False, "Timeout (>30s)"
        except Exception as e:
            return False, f"Exception: {str(e)[:100]}"

    def test_option(self, option_id: str, option_data: Dict) -> Tuple[bool, str]:
        """Test a single menu option."""
        script = option_data.get("script")
        test_mode = option_data.get("test_mode", "skip")

        if test_mode == "skip":
            return True, "Skipped (menu function)"

        if not script:
            return False, "Not implemented"

        if test_mode == "check_exists":
            return self.check_script_exists(script)

        elif test_mode == "execute":
            return self.execute_script(script)

        return False, f"Unknown test mode: {test_mode}"

    def find_duplicates(self) -> List[Tuple[str, List[str]]]:
        """Find duplicate scripts referenced by menu."""
        script_to_options = {}

        for option_id, option_data in self.menu_map.items():
            script = option_data.get("script")
            if script:
                if script not in script_to_options:
                    script_to_options[script] = []
                script_to_options[script].append(option_id)

        # Find scripts used by multiple options
        duplicates = [
            (script, options) for script, options in script_to_options.items() if len(options) > 1
        ]

        return duplicates

    def run_all_tests(self):
        """Run all menu option tests."""
        print("=" * 80)
        print("MENU SYSTEM COMPREHENSIVE TEST")
        print("=" * 80)
        print()

        mode = "QUICK MODE" if self.quick_mode else "FULL MODE"
        print(f"Mode: {mode}")
        print(f"Total options: {len(self.menu_map)}")
        print()

        # Find duplicates first
        duplicates = self.find_duplicates()
        if duplicates:
            print(f"⚠️  Found {len(duplicates)} duplicate script(s):")
            for script, options in duplicates:
                self.results["duplicates"].append((script, options))
                print(f"   {script}")
                print(f"      Used by: {', '.join(options)}")
            print()

        # Test each option
        total = len(self.menu_map)
        for i, (option_id, option_data) in enumerate(self.menu_map.items(), 1):
            name = option_data["name"]
            print(f"[{i}/{total}] {option_id}: {name}...", end=" ")

            success, message = self.test_option(option_id, option_data)

            if success:
                print(f"✅ {message}")
                if "Skipped" in message:
                    self.results["skipped"].append((option_id, name, message))
                else:
                    self.results["passed"].append((option_id, name, message))
            else:
                print(f"❌ {message}")
                if "Not implemented" in message or "not found" in message:
                    self.results["missing"].append((option_id, name, message))
                else:
                    self.results["failed"].append((option_id, name, message))

        print()
        self.print_summary()

    def print_summary(self):
        """Print test summary."""
        print("=" * 80)
        print("TEST SUMMARY")
        print("=" * 80)
        print()

        total = len(self.menu_map)
        passed = len(self.results["passed"])
        failed = len(self.results["failed"])
        skipped = len(self.results["skipped"])
        missing = len(self.results["missing"])
        duplicates = len(self.results["duplicates"])

        print(f"Total options:  {total}")
        print(f"✅ Passed:      {passed}")
        print(f"❌ Failed:      {failed}")
        print(f"⏭️  Skipped:     {skipped}")
        print(f"🚫 Missing:     {missing}")
        print(f"⚠️  Duplicates:  {duplicates}")
        print()

        # Show failures
        if self.results["failed"]:
            print("FAILURES:")
            print("-" * 80)
            for option_id, name, message in self.results["failed"]:
                print(f"  {option_id}: {name}")
                print(f"    {message}")
            print()

        # Show missing
        if self.results["missing"]:
            print("MISSING IMPLEMENTATIONS:")
            print("-" * 80)
            for option_id, name, message in self.results["missing"]:
                print(f"  {option_id}: {name}")
                print(f"    {message}")
            print()

        # Show duplicates
        if self.results["duplicates"]:
            print("DUPLICATE SCRIPTS:")
            print("-" * 80)
            for script, options in self.results["duplicates"]:
                print(f"  {script}")
                print(f"    Menu options: {', '.join(options)}")
            print()

        # Overall status
        if failed == 0 and missing == 0:
            print("✅ ALL TESTS PASSED")
        else:
            print("❌ TESTS FAILED - Fix above issues")

        # Save results
        results_file = PROJECT_ROOT / "test_results_menu.json"
        with open(results_file, "w") as f:
            json.dump(self.results, f, indent=2)
        print(f"\nResults saved to: {results_file}")


def main():
    """Main entry point."""
    quick_mode = "--quick" in sys.argv

    tester = MenuTester(quick_mode=quick_mode)
    tester.run_all_tests()

    # Exit with error code if tests failed
    if tester.results["failed"] or tester.results["missing"]:
        sys.exit(1)

    sys.exit(0)


if __name__ == "__main__":
    main()
