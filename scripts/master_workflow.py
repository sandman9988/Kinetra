#!/usr/bin/env python3
"""
Kinetra Master Workflow
=======================

Complete end-to-end workflow in one process:
1. Authentication & account selection
2. Download data (with options)
3. Check and fill missing data
4. Check data integrity
5. Prepare data (train/test split)
6. Run exploration (agent comparison)

Usage:
    python scripts/master_workflow.py
"""

import os
import sys
import subprocess
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))


def print_header(text: str):
    """Print section header."""
    print("\n" + "=" * 80)
    print(f"  {text}")
    print("=" * 80)


def check_credentials() -> bool:
    """Check if MetaAPI credentials are set."""
    token = os.environ.get('METAAPI_TOKEN')
    account_id = os.environ.get('METAAPI_ACCOUNT_ID')

    # Check for placeholder values
    placeholder_patterns = ['your-token-here', 'your-account-id-here', 'placeholder', 'example']

    if token:
        # Check if it's a placeholder
        if any(placeholder in token.lower() for placeholder in placeholder_patterns):
            print(f"\n⚠️  Found placeholder METAAPI_TOKEN (ignoring it)")
            print("   Will prompt you for the real token...")
        else:
            print(f"\n✅ Found API token: {token[:20]}...")
    else:
        print("\nℹ️  No METAAPI_TOKEN set (will prompt interactively)")

    if account_id:
        # Check if it's a placeholder
        if any(placeholder in account_id.lower() for placeholder in placeholder_patterns):
            print(f"⚠️  Found placeholder METAAPI_ACCOUNT_ID (ignoring it)")
            print("   Will list your accounts...")
        else:
            print(f"✅ Found account ID: {account_id}")
    else:
        print("ℹ️  No METAAPI_ACCOUNT_ID set (will select interactively)")

    return True


def run_step(step_name: str, script_path: str, required: bool = True, allow_exit: bool = True) -> bool:
    """Run a workflow step."""
    print_header(step_name)

    if not Path(script_path).exists():
        print(f"\n❌ Script not found: {script_path}")
        return False

    try:
        result = subprocess.run([sys.executable, script_path])

        if result.returncode != 0:
            if required:
                print(f"\n❌ {step_name} failed (exit code {result.returncode})")
                return False
            else:
                print(f"\n⚠️  {step_name} completed with warnings")

        print(f"\n✅ {step_name} complete")

        # Exit offramp
        if allow_exit:
            print("\nOptions:")
            print("  1. Continue to next step")
            print("  2. Exit workflow")

            choice = input("\nSelect [1-2]: ").strip()
            if choice == '2':
                print("\n👋 Exiting workflow")
                return False

        return True

    except KeyboardInterrupt:
        print(f"\n\n⚠️  {step_name} interrupted by user")
        return False
    except Exception as e:
        print(f"\n❌ {step_name} error: {e}")
        return False


def main():
    """Run complete workflow."""
    print_header("KINETRA MASTER WORKFLOW")

    print("""
Complete end-to-end workflow:

1. Authentication & Account Selection
2. Download Data (interactive)
3. Check & Fill Missing Data
4. Check Data Integrity
5. Prepare Data (train/test split)
6. Explore & Compare Agents

THE MARKET TELLS US, WE DON'T ASSUME!
""")

    response = input("\nProceed with complete workflow? [1=Yes, 2=No]: ").strip()
    if response != '1':
        print("\n👋 Workflow cancelled")
        return

    # Check credentials first
    print_header("STEP 1: AUTHENTICATION")

    if not check_credentials():
        print("\n⚠️  Set credentials and run again")
        return

    # Ask which steps to run
    print_header("WORKFLOW OPTIONS")

    print("""
Which steps do you want to run?

  1. Full workflow (recommended for first run)
  2. Skip download (use existing data)
  3. Download only (stop after downloading)
  4. Custom (choose each step)
""")

    workflow_choice = input("\nSelect workflow [1-4]: ").strip()

    # Determine steps to run
    run_download = True
    run_fill = True
    run_integrity = True
    run_prepare = True
    run_explore = True

    if workflow_choice == '2':
        run_download = False
        print("\n✅ Will skip download, use existing data")

    elif workflow_choice == '3':
        run_fill = False
        run_integrity = False
        run_prepare = False
        run_explore = False
        print("\n✅ Will download only")

    elif workflow_choice == '4':
        run_download = input("\n  Download data? [1=Yes, 2=No]: ").strip() == '1'
        if run_download:
            run_fill = input("  Fill missing data? [1=Yes, 2=No]: ").strip() == '1'
        run_integrity = input("  Check integrity? [1=Yes, 2=No]: ").strip() == '1'
        run_prepare = input("  Prepare data? [1=Yes, 2=No]: ").strip() == '1'
        run_explore = input("  Run exploration? [1=Yes, 2=No]: ").strip() == '1'

    # Run workflow steps
    print_header("STARTING WORKFLOW")

    # Step 2: Download
    if run_download:
        if not run_step(
            "STEP 2: DOWNLOAD DATA",
            "scripts/download_interactive.py",
            required=True,
            allow_exit=True
        ):
            return
    else:
        print_header("STEP 2: DOWNLOAD DATA")
        print("\n⏭️  Skipped (using existing data)")

    # Step 3: Fill missing
    if run_fill:
        # This step is optional - can continue even if it fails
        if not run_step(
            "STEP 3: CHECK & FILL MISSING DATA",
            "scripts/check_and_fill_data.py",
            required=False,
            allow_exit=True
        ):
            return
    else:
        print_header("STEP 3: CHECK & FILL MISSING DATA")
        print("\n⏭️  Skipped")

    # Step 4: Integrity check
    if run_integrity:
        if not run_step(
            "STEP 4: CHECK DATA INTEGRITY",
            "scripts/check_data_integrity.py",
            required=False,
            allow_exit=True
        ):
            return
    else:
        print_header("STEP 4: CHECK DATA INTEGRITY")
        print("\n⏭️  Skipped")

    # Step 5: Prepare data
    if run_prepare:
        if not run_step(
            "STEP 5: PREPARE DATA",
            "scripts/prepare_data.py",
            required=True,
            allow_exit=True
        ):
            return
    else:
        print_header("STEP 5: PREPARE DATA")
        print("\n⏭️  Skipped")

    # Step 6: Exploration
    if run_explore:
        print_header("STEP 6: EXPLORATION")

        print("""
Exploration options:

  1. Universal Agent Baseline (LinearQ only)
  2. Compare All Agents (LinearQ vs PPO vs SAC vs TD3)
  3. Open Testing Menu (choose manually)
  4. Skip exploration
""")

        explore_choice = input("\nSelect exploration [1-4]: ").strip()

        if explore_choice == '1':
            run_step(
                "UNIVERSAL AGENT BASELINE",
                "scripts/explore_universal.py",
                required=False,
                allow_exit=False  # Don't need exit prompt for last step
            )

        elif explore_choice == '2':
            run_step(
                "COMPARE AGENTS (LinearQ vs PPO vs SAC vs TD3)",
                "scripts/explore_compare_agents.py",
                required=False,
                allow_exit=False  # Don't need exit prompt for last step
            )

        elif explore_choice == '3':
            run_step(
                "TESTING MENU",
                "scripts/test_menu.py",
                required=False,
                allow_exit=False  # Don't need exit prompt for last step
            )

        else:
            print("\n⏭️  Skipped exploration")

    else:
        print_header("STEP 6: EXPLORATION")
        print("\n⏭️  Skipped")

    # Final summary
    print_header("WORKFLOW COMPLETE!")

    print("""
✅ All steps completed successfully!

Next steps:
  • Review results in results/exploration/
  • Run additional exploration: python scripts/test_menu.py
  • Check prepared data: ls data/prepared/train/

THE MARKET HAS TOLD US - NOW WE KNOW!
""")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Workflow interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n❌ Workflow error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
