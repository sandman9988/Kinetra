#!/usr/bin/env python3
"""
Exhaustive Test Runner for Kinetra
===================================

Comprehensive test orchestration script for local and CI environments.

Usage:
    # Fast CI mode (subset of combinations)
    python scripts/run_exhaustive_tests.py --ci-mode

    # Full exhaustive mode (all combinations)
    python scripts/run_exhaustive_tests.py --full

    # Specific test types
    python scripts/run_exhaustive_tests.py --test-type unit
    python scripts/run_exhaustive_tests.py --test-type integration
    python scripts/run_exhaustive_tests.py --test-type monte_carlo
    python scripts/run_exhaustive_tests.py --test-type walk_forward

    # All test types
    python scripts/run_exhaustive_tests.py --all

    # Parallel execution
    python scripts/run_exhaustive_tests.py --full --parallel 4

    # With coverage
    python scripts/run_exhaustive_tests.py --ci-mode --coverage

Philosophy:
- CI mode: Fast feedback for PRs (5-10 minutes)
- Full mode: Exhaustive validation for releases (1-2 hours)
- All modes: Rigorous statistical validation (p < 0.01)
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Project root
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class Colors:
    """ANSI color codes for terminal output."""

    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


def print_header(message: str) -> None:
    """Print formatted header."""
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'=' * 70}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{message.center(70)}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'=' * 70}{Colors.ENDC}\n")


def print_success(message: str) -> None:
    """Print success message."""
    print(f"{Colors.OKGREEN}✅ {message}{Colors.ENDC}")


def print_error(message: str) -> None:
    """Print error message."""
    print(f"{Colors.FAIL}❌ {message}{Colors.ENDC}")


def print_warning(message: str) -> None:
    """Print warning message."""
    print(f"{Colors.WARNING}⚠️  {message}{Colors.ENDC}")


def print_info(message: str) -> None:
    """Print info message."""
    print(f"{Colors.OKCYAN}ℹ️  {message}{Colors.ENDC}")


def run_command(
    cmd: List[str], env: Optional[Dict[str, str]] = None, timeout: Optional[int] = None
) -> Tuple[int, str, str]:
    """
    Run shell command and capture output.

    Args:
        cmd: Command and arguments
        env: Environment variables
        timeout: Timeout in seconds

    Returns:
        Tuple of (return_code, stdout, stderr)
    """
    env_dict = os.environ.copy()
    if env:
        env_dict.update(env)

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            env=env_dict,
            timeout=timeout,
            cwd=PROJECT_ROOT,
        )
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return -1, "", f"Command timed out after {timeout}s"
    except Exception as e:
        return -1, "", str(e)


def check_dependencies() -> bool:
    """Check that all required dependencies are installed."""
    print_info("Checking dependencies...")

    required = ["pytest", "numpy", "pandas", "torch"]
    missing = []

    for package in required:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)

    if missing:
        print_error(f"Missing dependencies: {', '.join(missing)}")
        print_info("Install with: pip install -r requirements.txt")
        return False

    print_success("All dependencies installed")
    return True


def verify_agent_factory() -> bool:
    """Verify AgentFactory can create all agent types."""
    print_info("Verifying AgentFactory...")

    cmd = [sys.executable, "-m", "kinetra.agent_factory"]
    returncode, stdout, stderr = run_command(cmd, timeout=30)

    if returncode != 0:
        print_error("AgentFactory verification failed")
        print(stderr)
        return False

    if "All agent factory tests passed" in stdout:
        print_success("AgentFactory verified - all 6 agents working")
        return True

    print_warning("AgentFactory verification incomplete")
    return False


def run_pytest(
    test_path: str,
    ci_mode: bool = False,
    parallel: Optional[int] = None,
    coverage: bool = False,
    verbose: bool = True,
    timeout: int = 3600,
) -> Tuple[bool, Dict[str, any]]:
    """
    Run pytest with specified configuration.

    Args:
        test_path: Path to test file or directory
        ci_mode: Enable CI mode (subset testing)
        parallel: Number of parallel workers
        coverage: Enable coverage reporting
        verbose: Verbose output
        timeout: Test timeout in seconds

    Returns:
        Tuple of (success, results_dict)
    """
    cmd = [sys.executable, "-m", "pytest", test_path]

    if verbose:
        cmd.append("-v")

    cmd.append("--tb=short")

    if parallel:
        cmd.extend(["-n", str(parallel)])

    if coverage:
        cmd.extend(["--cov=kinetra", "--cov-report=term-missing", "--cov-report=html"])

    env = {"KINETRA_CI_MODE": "1" if ci_mode else "0"}

    print_info(f"Running: {' '.join(cmd)}")
    print_info(f"CI Mode: {ci_mode}, Parallel: {parallel or 'auto'}")

    start_time = time.time()
    returncode, stdout, stderr = run_command(cmd, env=env, timeout=timeout)
    elapsed = time.time() - start_time

    results = {
        "success": returncode == 0,
        "elapsed_time": elapsed,
        "stdout": stdout,
        "stderr": stderr,
    }

    # Parse pytest output for stats
    if "passed" in stdout:
        import re

        match = re.search(r"(\d+) passed", stdout)
        if match:
            results["passed"] = int(match.group(1))

        match = re.search(r"(\d+) failed", stdout)
        if match:
            results["failed"] = int(match.group(1))

    return returncode == 0, results


def run_test_suite(args: argparse.Namespace) -> bool:
    """
    Run the complete test suite.

    Args:
        args: Parsed command line arguments

    Returns:
        True if all tests passed
    """
    all_passed = True
    results = {}

    # Determine test types to run
    if args.all:
        test_types = ["unit", "integration", "monte_carlo", "walk_forward"]
    elif args.test_type:
        test_types = [args.test_type]
    else:
        test_types = ["unit"]  # Default

    # Run each test type
    for test_type in test_types:
        print_header(f"{test_type.upper()} TESTS")

        test_path = (
            f"tests/test_exhaustive_combinations.py::TestExhaustiveCombinations"
            f"::test_all_combos[{test_type}]"
        )

        # Adjust timeout based on mode and test type
        if args.ci_mode:
            timeout = 600 if test_type == "monte_carlo" else 300
        else:
            timeout = 3600 if test_type == "monte_carlo" else 1800

        success, result = run_pytest(
            test_path=test_path,
            ci_mode=args.ci_mode,
            parallel=args.parallel,
            coverage=args.coverage and test_type == test_types[-1],  # Coverage on last test
            timeout=timeout,
        )

        results[test_type] = result

        if success:
            elapsed_min = result["elapsed_time"] / 60
            print_success(f"{test_type} tests passed in {elapsed_min:.1f} minutes")
        else:
            print_error(f"{test_type} tests failed")
            all_passed = False

            if args.stop_on_fail:
                print_warning("Stopping due to --stop-on-fail")
                break

    return all_passed, results


def run_agent_tests() -> bool:
    """Run agent-specific tests."""
    print_header("AGENT TESTS")

    tests = [
        "tests/test_exhaustive_combinations.py::test_all_agents",
        "tests/test_exhaustive_combinations.py::test_all_regimes",
    ]

    for test in tests:
        success, _ = run_pytest(test, ci_mode=True, timeout=120)
        if not success:
            print_error(f"Failed: {test}")
            return False

    print_success("All agent tests passed")
    return True


def generate_summary_report(results: Dict[str, Dict], output_path: Path) -> None:
    """Generate summary report of test results."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    summary = {
        "timestamp": timestamp,
        "results": {},
        "total_elapsed": 0,
        "total_passed": 0,
        "total_failed": 0,
    }

    for test_type, result in results.items():
        summary["results"][test_type] = {
            "success": result.get("success", False),
            "elapsed_time": result.get("elapsed_time", 0),
            "passed": result.get("passed", 0),
            "failed": result.get("failed", 0),
        }
        summary["total_elapsed"] += result.get("elapsed_time", 0)
        summary["total_passed"] += result.get("passed", 0)
        summary["total_failed"] += result.get("failed", 0)

    # Write JSON report
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)

    print_info(f"Summary report saved to: {output_path}")

    # Print summary
    print_header("TEST SUMMARY")
    print(f"Timestamp: {timestamp}")
    print(f"Total elapsed: {summary['total_elapsed'] / 60:.1f} minutes")
    print(f"Total passed: {summary['total_passed']}")
    print(f"Total failed: {summary['total_failed']}")

    for test_type, stats in summary["results"].items():
        status = "✅" if stats["success"] else "❌"
        print(f"  {status} {test_type:15} - {stats['elapsed_time'] / 60:.1f} min")


def generate_dashboard_report(output_path: Path) -> bool:
    """Generate interactive dashboard report from test results."""
    print_info("Generating dashboard report...")

    try:
        from kinetra.test_dashboard import TestDashboard

        dashboard = TestDashboard()
        dashboard.generate_static_report(str(output_path))

        print_success(f"Dashboard report saved to: {output_path}")
        return True
    except ImportError as e:
        print_warning(f"Dashboard dependencies not available: {e}")
        print_info("Install with: pip install plotly dash dash-bootstrap-components")
        return False
    except Exception as e:
        print_error(f"Failed to generate dashboard: {e}")
        return False


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Exhaustive test runner for Kinetra",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Fast CI mode for PR testing
  python scripts/run_exhaustive_tests.py --ci-mode

  # Full exhaustive testing for release
  python scripts/run_exhaustive_tests.py --full --all

  # Specific test type with parallelization
  python scripts/run_exhaustive_tests.py --ci-mode --test-type unit --parallel 4

  # With coverage reporting
  python scripts/run_exhaustive_tests.py --ci-mode --coverage
        """,
    )

    # Mode selection
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        "--ci-mode",
        action="store_true",
        help="CI mode - fast subset testing (5-10 minutes)",
    )
    mode_group.add_argument(
        "--full",
        action="store_true",
        help="Full exhaustive mode - all combinations (1-2 hours)",
    )

    # Test selection
    parser.add_argument(
        "--test-type",
        choices=["unit", "integration", "monte_carlo", "walk_forward"],
        help="Specific test type to run",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all test types (unit, integration, monte_carlo, walk_forward)",
    )

    # Execution options
    parser.add_argument(
        "--parallel",
        type=int,
        metavar="N",
        help="Number of parallel workers (default: auto)",
    )
    parser.add_argument(
        "--coverage",
        action="store_true",
        help="Generate coverage report",
    )
    parser.add_argument(
        "--stop-on-fail",
        action="store_true",
        help="Stop on first test failure",
    )
    parser.add_argument(
        "--skip-verify",
        action="store_true",
        help="Skip pre-flight verification",
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=Path("test_results"),
        help="Directory for test reports (default: test_results)",
    )
    parser.add_argument(
        "--generate-dashboard",
        action="store_true",
        help="Generate interactive dashboard report (requires plotly/dash)",
    )

    args = parser.parse_args()

    # Print banner
    print_header("KINETRA EXHAUSTIVE TEST RUNNER")

    mode = "CI MODE (subset)" if args.ci_mode else "FULL MODE (exhaustive)"
    print_info(f"Test mode: {mode}")

    # Pre-flight checks
    if not args.skip_verify:
        print_header("PRE-FLIGHT VERIFICATION")

        if not check_dependencies():
            return 1

        if not verify_agent_factory():
            print_warning("AgentFactory verification incomplete, continuing anyway...")

    # Run tests
    start_time = time.time()

    # Run agent tests first (quick validation)
    if not run_agent_tests():
        print_error("Agent tests failed - aborting")
        return 1

    # Run main test suite
    all_passed, results = run_test_suite(args)

    total_elapsed = time.time() - start_time

    # Generate report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = args.report_dir / f"test_report_{timestamp}.json"
    generate_summary_report(results, report_path)

    # Generate dashboard if requested
    if args.generate_dashboard:
        dashboard_path = args.report_dir / f"test_dashboard_{timestamp}.html"
        generate_dashboard_report(dashboard_path)

    # Final status
    print_header("FINAL RESULTS")
    if all_passed:
        print_success(f"All tests passed in {total_elapsed / 60:.1f} minutes")
        print_success("Ready for deployment! 🚀")
        return 0
    else:
        print_error(f"Some tests failed after {total_elapsed / 60:.1f} minutes")
        print_info("Check logs and fix issues before deployment")
        return 1


if __name__ == "__main__":
    sys.exit(main())
