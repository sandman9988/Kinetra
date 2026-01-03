#!/usr/bin/env python3
"""
KINETRA UNIFIED MENU
====================
Single comprehensive menu for all Kinetra workflows.
No duplicates. All scripts integrated.

Philosophy: First principles, no assumptions, statistical rigor (p < 0.01)
"""

import os
import subprocess
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))


def print_header(title: str):
    """Print formatted header."""
    print("\n" + "=" * 80)
    print(f" {title}")
    print("=" * 80)


def get_choice(prompt: str, valid_choices: list) -> str:
    """Get user choice with validation."""
    while True:
        choice = input(f"\n{prompt}: ").strip()
        if choice in valid_choices:
            return choice
        print(f"❌ Invalid choice. Please select from: {', '.join(valid_choices)}")


def run_script(script_path: str, args: list = None) -> bool:
    """Run a script and return success status."""
    args = args or []
    cmd = [sys.executable, script_path] + args
    result = subprocess.run(cmd, cwd=PROJECT_ROOT)
    return result.returncode == 0


# =============================================================================
# 1. CREDENTIALS & SETUP
# =============================================================================


def credentials_menu():
    """Credential management menu."""
    print_header("CREDENTIALS & SETUP")

    # Check credential status
    try:
        sys.path.insert(0, str(PROJECT_ROOT / "scripts/utils"))
        from secure_token_helper import get_metaapi_token, list_configured_accounts

        token = get_metaapi_token()
        accounts = list_configured_accounts()

        print("\n📋 CREDENTIAL STATUS:")
        if token:
            print(f"  ✅ Token: Configured")
        else:
            print(f"  ⚠️  Token: NOT configured")

        if accounts:
            print(f"  ✅ Accounts: {len(accounts)} configured")
            for label, acc_id in list(accounts.items())[:3]:
                print(f"     • {label}: {acc_id[:8]}...")
        else:
            print(f"  ⚠️  Accounts: NOT configured")
    except Exception as e:
        print(f"\n⚠️  Could not check credentials: {e}")

    print("""
Options:
  1. Setup MetaAPI Credentials (Token + Accounts)
  2. Add Additional Account
  3. View All Configured Accounts
  0. Back
    """)

    choice = get_choice("Select", ["0", "1", "2", "3"])

    if choice == "1":
        run_script("scripts/setup_metaapi_credentials.py")
    elif choice == "2":
        run_script("scripts/setup_metaapi_credentials.py", ["--add-account"])
    elif choice == "3":
        run_script("scripts/utils/secure_token_helper.py")


# =============================================================================
# 2. DATA MANAGEMENT
# =============================================================================


def data_menu():
    """Data management menu."""
    print_header("DATA MANAGEMENT")

    print("""
Options:
  1. Download Data (Interactive)
  2. Download Data (CLI - Specific Instrument)
  3. Consolidate Data (Create Master Standardized)
  4. Audit Data Coverage
  5. Check Data Integrity
  6. Backup Data
  0. Back
    """)

    choice = get_choice("Select", ["0", "1", "2", "3", "4", "5", "6"])

    if choice == "1":
        run_script("scripts/download/download_interactive.py")
    elif choice == "2":
        print(
            "\nExample: python scripts/mt5_metaapi_sync.py --init --symbol BTCUSD --timeframe H1 --years 2"
        )
        symbol = input("Symbol (e.g., BTCUSD): ").strip()
        timeframe = input("Timeframe (M15/M30/H1/H4/D1): ").strip()
        years = input("Years of data (1-5): ").strip()
        if symbol and timeframe and years:
            run_script(
                "scripts/mt5_metaapi_sync.py",
                ["--init", "--symbol", symbol, "--timeframe", timeframe, "--years", years],
            )
    elif choice == "3":
        print("\nConsolidating data...")
        run_script("scripts/consolidate_data.py", ["--symlink"])
    elif choice == "4":
        run_script("scripts/audit_data_coverage.py", ["--show-gaps"])
    elif choice == "5":
        run_script("scripts/download/check_data_integrity.py")
    elif choice == "6":
        run_script("scripts/download/backup_data.py")


# =============================================================================
# 3. TESTING
# =============================================================================


def testing_menu():
    """Testing menu."""
    print_header("TESTING")

    print("""
Options:
  1. Run Exhaustive Tests (Full Suite + Dashboard)
  2. Run Exhaustive Tests (CI Mode - Fast)
  3. Run Unit Tests Only
  4. Run Integration Tests
  5. Run Monte Carlo Validation
  6. Run Walk-Forward Tests
  0. Back
    """)

    choice = get_choice("Select", ["0", "1", "2", "3", "4", "5", "6"])

    if choice == "1":
        run_script("scripts/run_exhaustive_tests.py", ["--full", "--generate-dashboard"])
    elif choice == "2":
        run_script("scripts/run_exhaustive_tests.py", ["--ci-mode", "--generate-dashboard"])
    elif choice == "3":
        subprocess.run(
            [sys.executable, "-m", "pytest", "tests/", "-v", "-k", "not integration"],
            cwd=PROJECT_ROOT,
        )
    elif choice == "4":
        subprocess.run(
            [sys.executable, "-m", "pytest", "tests/", "-v", "-k", "integration"], cwd=PROJECT_ROOT
        )
    elif choice == "5":
        run_script("scripts/run_exhaustive_tests.py", ["--full", "--test-type", "monte_carlo"])
    elif choice == "6":
        run_script("scripts/run_exhaustive_tests.py", ["--full", "--test-type", "walk_forward"])


# =============================================================================
# 4. BACKTESTING
# =============================================================================


def backtest_menu():
    """Backtesting menu."""
    print_header("BACKTESTING")

    print("""
Options:
  1. Batch Backtest (Multiple Instruments/Timeframes)
  2. Single Instrument Backtest
  3. Agent Comparison Backtest
  4. Physics Validation Backtest
  0. Back
    """)

    choice = get_choice("Select", ["0", "1", "2", "3", "4"])

    if choice == "1":
        print("\nExample: python scripts/batch_backtest.py --instrument BTCUSD --timeframe H1")
        instrument = input("Instrument (or leave blank for all): ").strip()
        timeframe = input("Timeframe (or leave blank for all): ").strip()
        args = []
        if instrument:
            args.extend(["--instrument", instrument])
        if timeframe:
            args.extend(["--timeframe", timeframe])
        run_script("scripts/batch_backtest.py", args)
    elif choice == "2":
        instrument = input("Instrument: ").strip()
        timeframe = input("Timeframe: ").strip()
        if instrument and timeframe:
            run_script(
                "scripts/batch_backtest.py", ["--instrument", instrument, "--timeframe", timeframe]
            )
    elif choice == "3":
        run_script("scripts/testing/scripts/explore_compare_agents.py")
    elif choice == "4":
        run_script("scripts/testing/run_physics_backtest.py")


# =============================================================================
# 5. WORKFLOWS (INTEGRATED MULTI-STEP)
# =============================================================================


def workflows_menu():
    """Pre-built workflows combining multiple steps."""
    print_header("INTEGRATED WORKFLOWS")

    print("""
Complete end-to-end workflows:

  1. First-Time Setup
     → Setup credentials → Download sample data → Consolidate → Run quick test

  2. Download → Consolidate → Test
     → Interactive download → Consolidate → Full exhaustive tests

  3. Quick Validation
     → Check data coverage → Run CI tests → Show dashboard

  4. Full Research Cycle
     → Download all missing data → Consolidate → Monte Carlo validation

  0. Back
    """)

    choice = get_choice("Select", ["0", "1", "2", "3", "4"])

    if choice == "1":
        workflow_first_time_setup()
    elif choice == "2":
        workflow_download_consolidate_test()
    elif choice == "3":
        workflow_quick_validation()
    elif choice == "4":
        workflow_full_research()


def workflow_first_time_setup():
    """First-time setup workflow."""
    print_header("FIRST-TIME SETUP WORKFLOW")

    print("\n📋 Step 1/4: Setup credentials...")
    if not run_script("scripts/setup_metaapi_credentials.py"):
        print("❌ Credential setup failed")
        return

    print("\n📥 Step 2/4: Download sample data (BTCUSD H1, 2 years)...")
    if not run_script(
        "scripts/mt5_metaapi_sync.py",
        ["--init", "--symbol", "BTCUSD", "--timeframe", "H1", "--years", "2"],
    ):
        print("⚠️  Download had issues, continuing...")

    print("\n📊 Step 3/4: Consolidate data...")
    if not run_script("scripts/consolidate_data.py", ["--symlink"]):
        print("⚠️  Consolidation had issues, continuing...")

    print("\n🧪 Step 4/4: Run quick test...")
    subprocess.run(
        [sys.executable, "-m", "pytest", "tests/test_physics.py", "-v"], cwd=PROJECT_ROOT
    )

    print("\n✅ First-time setup complete!")


def workflow_download_consolidate_test():
    """Download → Consolidate → Test workflow."""
    print_header("DOWNLOAD → CONSOLIDATE → TEST WORKFLOW")

    print("\n📥 Step 1/3: Interactive download...")
    if not run_script("scripts/download/download_interactive.py"):
        print("❌ Download failed")
        return

    print("\n📊 Step 2/3: Consolidate data...")
    if not run_script("scripts/consolidate_data.py", ["--symlink"]):
        print("❌ Consolidation failed")
        return

    print("\n🧪 Step 3/3: Run exhaustive tests...")
    run_script("scripts/run_exhaustive_tests.py", ["--full", "--generate-dashboard"])

    print("\n✅ Workflow complete! Check test_results/ for dashboard.")


def workflow_quick_validation():
    """Quick validation workflow."""
    print_header("QUICK VALIDATION WORKFLOW")

    print("\n📊 Step 1/2: Check data coverage...")
    run_script("scripts/audit_data_coverage.py", ["--show-gaps"])

    print("\n🧪 Step 2/2: Run CI tests...")
    run_script("scripts/run_exhaustive_tests.py", ["--ci-mode", "--generate-dashboard"])

    print("\n✅ Quick validation complete!")


def workflow_full_research():
    """Full research workflow."""
    print_header("FULL RESEARCH CYCLE")

    print("\n📊 Step 1/4: Audit current coverage...")
    run_script("scripts/audit_data_coverage.py", ["--show-gaps"])

    proceed = input("\nDownload all missing high-priority data? (y/N): ").strip().lower()
    if proceed == "y":
        print("\n📥 Step 2/4: Downloading missing data...")
        # Download priority combos
        priority_combos = [
            ("BTCUSD", "D1", "3"),
            ("EURUSD", "H1", "2"),
            ("EURUSD", "H4", "2"),
            ("EURUSD", "D1", "3"),
        ]
        for symbol, tf, years in priority_combos:
            print(f"\n  Downloading {symbol} {tf}...")
            run_script(
                "scripts/mt5_metaapi_sync.py",
                ["--init", "--symbol", symbol, "--timeframe", tf, "--years", years],
            )

    print("\n📊 Step 3/4: Consolidate data...")
    run_script("scripts/consolidate_data.py", ["--symlink"])

    print("\n🧪 Step 4/4: Run Monte Carlo validation...")
    run_script(
        "scripts/run_exhaustive_tests.py",
        ["--full", "--test-type", "monte_carlo", "--generate-dashboard"],
    )

    print("\n✅ Full research cycle complete!")


# =============================================================================
# 6. UTILITIES
# =============================================================================


def utilities_menu():
    """Utilities and diagnostics."""
    print_header("UTILITIES")

    print("""
Options:
  1. Check GPU/ROCm Status
  2. View System Status
  3. Check Recent Test Results
  4. Export Metrics (Grafana)
  5. Validate Theorems
  0. Back
    """)

    choice = get_choice("Select", ["0", "1", "2", "3", "4", "5"])

    if choice == "1":
        run_script("scripts/setup/check_gpu.py")
    elif choice == "2":
        print("\n📊 System Status:")
        run_script("scripts/audit_data_coverage.py", ["--show-gaps"])
    elif choice == "3":
        results_dir = PROJECT_ROOT / "test_results"
        if results_dir.exists():
            print(f"\n📂 Recent test results in: {results_dir}")
            subprocess.run(["ls", "-lht", str(results_dir)], cwd=PROJECT_ROOT)
        else:
            print("\n⚠️  No test results found")
    elif choice == "4":
        run_script("scripts/testing/test_grafana_export.py")
    elif choice == "5":
        run_script("scripts/testing/validate_theorems.py")


# =============================================================================
# MAIN MENU
# =============================================================================


def main_menu():
    """Main menu loop."""
    while True:
        print_header("KINETRA - UNIFIED MENU")

        print("""
Philosophy: First principles, no assumptions, statistical rigor (p < 0.01)

Main Sections:
  1. Credentials & Setup
  2. Data Management
  3. Testing
  4. Backtesting
  5. Workflows (Complete End-to-End)
  6. Utilities

  0. Exit

Navigation: Type 0 to exit | Ctrl+C to interrupt
        """)

        choice = get_choice("Select", ["0", "1", "2", "3", "4", "5", "6"])

        if choice == "0":
            print("\n👋 Goodbye!")
            break
        elif choice == "1":
            credentials_menu()
        elif choice == "2":
            data_menu()
        elif choice == "3":
            testing_menu()
        elif choice == "4":
            backtest_menu()
        elif choice == "5":
            workflows_menu()
        elif choice == "6":
            utilities_menu()


if __name__ == "__main__":
    try:
        main_menu()
    except KeyboardInterrupt:
        print("\n\n👋 Interrupted. Goodbye!")
        sys.exit(0)
