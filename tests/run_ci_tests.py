#!/usr/bin/env python3
"""
Kinetra CI Test Suite
=====================

Comprehensive CI/CD-style testing suite that validates:
1. Data integrity and availability
2. System components and dependencies
3. Menu navigation and workflows
4. Performance with real data loads
5. Integration between components
6. Error handling and recovery

Usage:
    python tests/run_ci_tests.py                # Standard CI tests
    python tests/run_ci_tests.py --smoke        # Quick smoke tests only
    python tests/run_ci_tests.py --full         # Full test suite
    python tests/run_ci_tests.py --data-only    # Data validation only
"""

import sys
import subprocess
import time
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple, Optional

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))


class CITestRunner:
    """CI/CD test runner for comprehensive system validation."""

    def __init__(self):
        self.results = []
        self.start_time = datetime.now()
        self.data_stats = {}

    def log(self, message: str, level: str = "INFO"):
        """Log test output."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        symbols = {"INFO": "ℹ️", "PASS": "✅", "FAIL": "❌", "WARN": "⚠️", "RUN": "🔄"}
        symbol = symbols.get(level, "•")
        print(f"[{timestamp}] {symbol} {message}")

    def run_command(self, cmd: List[str], timeout: int = 60, name: str = "") -> Tuple[bool, str, float]:
        """Run a command and return success, output, duration."""
        start = time.time()
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout
            )
            duration = time.time() - start
            success = result.returncode == 0
            output = (result.stdout or "") + (result.stderr or "")
            return success, output, duration
        except subprocess.TimeoutExpired:
            duration = time.time() - start
            return False, f"Timeout after {timeout}s", duration
        except Exception as e:
            duration = time.time() - start
            return False, str(e), duration

    # =========================================================================
    # PHASE 1: DATA VALIDATION
    # =========================================================================

    def test_data_availability(self) -> bool:
        """Test that required data files exist."""
        self.log("Testing data availability...", "RUN")

        checks = [
            ("data/master", "Master data directory"),
            ("data/prepared", "Prepared data directory"),
            ("data/master/crypto", "Crypto data"),
            ("data/master/forex", "Forex data"),
            ("data/master/indices", "Indices data"),
        ]

        all_passed = True
        for path, description in checks:
            if Path(path).exists():
                # Count files
                csv_files = list(Path(path).rglob("*.csv"))
                self.log(f"  ✓ {description}: {len(csv_files)} files", "INFO")
                self.data_stats[description] = len(csv_files)
            else:
                self.log(f"  ✗ {description}: NOT FOUND", "FAIL")
                all_passed = False

        return all_passed

    def test_data_integrity(self) -> bool:
        """Test data file integrity."""
        self.log("Testing data integrity...", "RUN")

        # Sample a few files and check format
        import csv
        test_files = []
        for pattern in ["data/master/*/*.csv", "data/prepared/*/*.csv"]:
            test_files.extend(list(Path().glob(pattern))[:3])

        passed = 0
        failed = 0

        for file_path in test_files:
            try:
                with open(file_path, 'r') as f:
                    reader = csv.reader(f, delimiter='\t')
                    header = next(reader)
                    first_row = next(reader)

                    # Check has expected columns
                    expected_cols = ['<DATE>', '<TIME>', '<OPEN>', '<HIGH>', '<LOW>', '<CLOSE>']
                    if all(col in header for col in expected_cols):
                        passed += 1
                    else:
                        self.log(f"  ✗ Invalid format: {file_path.name}", "WARN")
                        failed += 1
            except Exception as e:
                self.log(f"  ✗ Error reading {file_path.name}: {e}", "FAIL")
                failed += 1

        self.log(f"  Checked {passed + failed} files: {passed} valid, {failed} invalid", "INFO")
        return failed == 0

    def test_data_size_load(self) -> bool:
        """Test loading larger data files for performance."""
        self.log("Testing data load performance...", "RUN")

        # Find largest files
        import os
        files = []
        for root, dirs, filenames in os.walk("data/master"):
            for filename in filenames:
                if filename.endswith('.csv'):
                    path = os.path.join(root, filename)
                    size = os.path.getsize(path)
                    files.append((path, size))

        # Test loading top 3 largest files
        files.sort(key=lambda x: x[1], reverse=True)
        largest = files[:3]

        for file_path, size in largest:
            size_mb = size / 1024 / 1024
            start = time.time()
            try:
                with open(file_path, 'r') as f:
                    lines = sum(1 for _ in f)
                duration = time.time() - start
                rate = size_mb / duration if duration > 0 else 0
                self.log(f"  ✓ Loaded {Path(file_path).name} ({size_mb:.1f}MB, {lines:,} lines) in {duration:.2f}s ({rate:.1f} MB/s)", "INFO")
            except Exception as e:
                self.log(f"  ✗ Failed to load {Path(file_path).name}: {e}", "FAIL")
                return False

        return True

    # =========================================================================
    # PHASE 2: SMOKE TESTS
    # =========================================================================

    def test_imports(self) -> bool:
        """Test that core modules can be imported."""
        self.log("Testing imports...", "RUN")

        modules = [
            "kinetra_menu",
            "kinetra.workflow_manager",
            "kinetra.environment",
            "kinetra.agents.ppo_agent",
            "e2e_testing_framework",
        ]

        failed = []
        for module in modules:
            try:
                __import__(module)
                self.log(f"  ✓ {module}", "INFO")
            except Exception as e:
                self.log(f"  ✗ {module}: {e}", "FAIL")
                failed.append(module)

        return len(failed) == 0

    def test_dependencies(self) -> bool:
        """Test that required dependencies are installed."""
        self.log("Testing dependencies...", "RUN")

        deps = [
            "numpy",
            "pandas",
            "torch",
            "gymnasium",
            "stable_baselines3",
            "tqdm",
            "pytest"
        ]

        failed = []
        for dep in deps:
            try:
                __import__(dep)
                self.log(f"  ✓ {dep}", "INFO")
            except Exception as e:
                self.log(f"  ✗ {dep}: NOT INSTALLED", "FAIL")
                failed.append(dep)

        return len(failed) == 0

    # =========================================================================
    # PHASE 3: MENU & NAVIGATION TESTS
    # =========================================================================

    def test_menu_system(self) -> bool:
        """Run menu system tests."""
        self.log("Running menu system tests...", "RUN")
        success, output, duration = self.run_command(
            [sys.executable, "tests/test_menu_system.py"],
            timeout=60,
            name="Menu System"
        )

        if success:
            self.log(f"  Menu system tests passed ({duration:.1f}s)", "PASS")
        else:
            self.log(f"  Menu system tests failed", "FAIL")
            self.log(f"  Output: {output[:200]}", "INFO")

        return success

    def test_menu_workflow(self) -> bool:
        """Run menu workflow tests."""
        self.log("Running menu workflow tests...", "RUN")
        success, output, duration = self.run_command(
            [sys.executable, "tests/test_menu_workflow.py"],
            timeout=120,
            name="Menu Workflow"
        )

        if success:
            self.log(f"  Menu workflow tests passed ({duration:.1f}s)", "PASS")
        else:
            self.log(f"  Menu workflow tests failed", "FAIL")
            self.log(f"  Output: {output[:200]}", "INFO")

        return success

    # =========================================================================
    # PHASE 4: STRESS & PERFORMANCE TESTS
    # =========================================================================

    def test_stress_light(self) -> bool:
        """Run light stress tests."""
        self.log("Running light stress tests...", "RUN")
        success, output, duration = self.run_command(
            [sys.executable, "tests/test_system_stress.py", "--light"],
            timeout=180,
            name="Stress Light"
        )

        if success:
            self.log(f"  Stress tests passed ({duration:.1f}s)", "PASS")
        else:
            self.log(f"  Stress tests failed", "FAIL")

        return success

    def test_performance_profiling(self) -> bool:
        """Run performance profiling."""
        self.log("Running performance profiling...", "RUN")
        success, output, duration = self.run_command(
            [sys.executable, "tests/test_performance_profiling.py", "--quick"],
            timeout=180,
            name="Performance Profiling"
        )

        if success:
            self.log(f"  Performance profiling passed ({duration:.1f}s)", "PASS")
        else:
            self.log(f"  Performance profiling failed", "FAIL")

        return success

    # =========================================================================
    # PHASE 5: INTEGRATION TESTS
    # =========================================================================

    def test_integration(self) -> bool:
        """Run integration tests."""
        self.log("Running integration tests...", "RUN")
        success, output, duration = self.run_command(
            [sys.executable, "tests/test_integration.py"],
            timeout=120,
            name="Integration"
        )

        if success:
            self.log(f"  Integration tests passed ({duration:.1f}s)", "PASS")
        else:
            self.log(f"  Integration tests failed", "FAIL")

        return success

    # =========================================================================
    # TEST SUITES
    # =========================================================================

    def run_smoke_tests(self) -> Dict[str, bool]:
        """Run quick smoke tests."""
        self.log("\n" + "="*80, "INFO")
        self.log("SMOKE TESTS", "INFO")
        self.log("="*80, "INFO")

        tests = {
            "imports": self.test_imports,
            "dependencies": self.test_dependencies,
            "data_availability": self.test_data_availability,
        }

        results = {}
        for name, test_func in tests.items():
            results[name] = test_func()
            self.results.append((name, results[name]))

        return results

    def run_data_validation(self) -> Dict[str, bool]:
        """Run data validation tests."""
        self.log("\n" + "="*80, "INFO")
        self.log("DATA VALIDATION", "INFO")
        self.log("="*80, "INFO")

        tests = {
            "data_availability": self.test_data_availability,
            "data_integrity": self.test_data_integrity,
            "data_load_performance": self.test_data_size_load,
        }

        results = {}
        for name, test_func in tests.items():
            results[name] = test_func()
            self.results.append((name, results[name]))

        return results

    def run_standard_ci(self) -> Dict[str, bool]:
        """Run standard CI test suite."""
        results = {}

        # Phase 1: Smoke tests
        smoke_results = self.run_smoke_tests()
        results.update(smoke_results)

        # Phase 2: Data validation
        self.log("\n" + "="*80, "INFO")
        self.log("DATA VALIDATION", "INFO")
        self.log("="*80, "INFO")

        data_tests = {
            "data_integrity": self.test_data_integrity,
            "data_load_performance": self.test_data_size_load,
        }

        for name, test_func in data_tests.items():
            results[name] = test_func()
            self.results.append((name, results[name]))

        # Phase 3: Menu tests
        self.log("\n" + "="*80, "INFO")
        self.log("MENU & NAVIGATION TESTS", "INFO")
        self.log("="*80, "INFO")

        menu_tests = {
            "menu_system": self.test_menu_system,
            "menu_workflow": self.test_menu_workflow,
        }

        for name, test_func in menu_tests.items():
            results[name] = test_func()
            self.results.append((name, results[name]))

        # Phase 4: Stress tests
        self.log("\n" + "="*80, "INFO")
        self.log("STRESS TESTS", "INFO")
        self.log("="*80, "INFO")

        stress_result = self.test_stress_light()
        results["stress_light"] = stress_result
        self.results.append(("stress_light", stress_result))

        return results

    def run_full_ci(self) -> Dict[str, bool]:
        """Run full CI test suite."""
        # Run standard CI first
        results = self.run_standard_ci()

        # Add performance profiling
        self.log("\n" + "="*80, "INFO")
        self.log("PERFORMANCE PROFILING", "INFO")
        self.log("="*80, "INFO")

        perf_result = self.test_performance_profiling()
        results["performance"] = perf_result
        self.results.append(("performance", perf_result))

        # Add integration tests
        self.log("\n" + "="*80, "INFO")
        self.log("INTEGRATION TESTS", "INFO")
        self.log("="*80, "INFO")

        integration_result = self.test_integration()
        results["integration"] = integration_result
        self.results.append(("integration", integration_result))

        return results

    def print_summary(self):
        """Print CI test summary."""
        duration = (datetime.now() - self.start_time).total_seconds()

        self.log("\n" + "="*80, "INFO")
        self.log("CI TEST SUMMARY", "INFO")
        self.log("="*80, "INFO")

        passed = sum(1 for _, success in self.results if success)
        failed = len(self.results) - passed

        self.log(f"\nDuration: {duration:.1f}s", "INFO")
        self.log(f"Total Tests: {len(self.results)}", "INFO")
        self.log(f"Passed: {passed}", "PASS")
        self.log(f"Failed: {failed}", "FAIL" if failed > 0 else "INFO")
        self.log(f"Success Rate: {(passed/len(self.results)*100):.1f}%", "INFO")

        # Data stats
        if self.data_stats:
            self.log("\nData Statistics:", "INFO")
            for key, value in self.data_stats.items():
                self.log(f"  {key}: {value}", "INFO")

        # Detailed results
        self.log("\n" + "-"*80, "INFO")
        self.log("Detailed Results:", "INFO")
        self.log("-"*80, "INFO")

        for test_name, success in self.results:
            level = "PASS" if success else "FAIL"
            self.log(f"{test_name}", level)

        self.log("\n" + "="*80, "INFO")

        if failed == 0:
            self.log("✅ ALL CI TESTS PASSED", "PASS")
            return 0
        else:
            self.log(f"❌ {failed} CI TEST(S) FAILED", "FAIL")
            return 1


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Kinetra CI Test Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        '--smoke',
        action='store_true',
        help="Run only smoke tests (fast, ~30s)"
    )

    parser.add_argument(
        '--data-only',
        action='store_true',
        help="Run only data validation tests"
    )

    parser.add_argument(
        '--full',
        action='store_true',
        help="Run full CI suite including performance and integration"
    )

    args = parser.parse_args()

    runner = CITestRunner()

    try:
        if args.smoke:
            runner.run_smoke_tests()
        elif args.data_only:
            runner.run_data_validation()
        elif args.full:
            runner.run_full_ci()
        else:
            runner.run_standard_ci()

        return runner.print_summary()

    except KeyboardInterrupt:
        runner.log("\n⚠️  CI tests interrupted by user", "WARN")
        return runner.print_summary()
    except Exception as e:
        runner.log(f"\n❌ CI test runner failed: {e}", "FAIL")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
