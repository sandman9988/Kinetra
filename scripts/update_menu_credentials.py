#!/usr/bin/env python3
"""
Update kinetra_menu.py to integrate credential setup workflow.

This script patches the authentication menu to:
1. Add credential setup option
2. Display credential status
3. Streamline the download → consolidate → test flow
4. Check credentials before downloading
"""

import sys
from pathlib import Path

# Read the menu file
menu_file = Path(__file__).parent.parent / "kinetra_menu.py"
content = menu_file.read_text()

# Find the insertion point (before show_authentication_menu)
setup_function = '''
def setup_metaapi_credentials_menu(wf_manager: WorkflowManager) -> bool:
    """Run the credential setup script."""
    print_submenu_header("Setup MetaAPI Credentials")

    print("\\n🔐 Launching credential setup wizard...")
    print("This will guide you through:")
    print("  • Setting your MetaAPI token")
    print("  • Selecting and labeling broker accounts")
    print("  • Saving credentials securely to .env\\n")

    try:
        import subprocess
        result = subprocess.run(
            [sys.executable, "scripts/setup_metaapi_credentials.py"], check=False
        )
        return result.returncode == 0
    except Exception as e:
        print(f"\\n❌ Error running setup: {e}")
        return False


'''

# Insert the new function before show_authentication_menu
if "def setup_metaapi_credentials_menu" not in content:
    content = content.replace(
        "def show_authentication_menu(", setup_function + "def show_authentication_menu("
    )

# Update show_authentication_menu to add credential status display
old_auth_menu = '''    print("""
Available options:
  1. Select MetaAPI Account
  2. Test Connection
  0. Back to Main Menu
    """)

    choice = get_input("Select option", ["0", "1", "2"])'''

new_auth_menu = '''    # Check credential status
    try:
        from scripts.utils.secure_token_helper import get_metaapi_token, list_configured_accounts

        token = get_metaapi_token()
        accounts = list_configured_accounts()

        print("\\n" + "-" * 80)
        print("CREDENTIAL STATUS")
        print("-" * 80)

        if token:
            print(f"✅ MetaAPI Token: Configured")
        else:
            print(f"⚠️  MetaAPI Token: Not configured")

        if accounts:
            print(f"✅ Accounts Configured: {len(accounts)}")
            for label, account_id in list(accounts.items())[:3]:
                print(f"   • {label}: {account_id[:8]}...")
            if len(accounts) > 3:
                print(f"   ... and {len(accounts) - 3} more")
        else:
            print(f"⚠️  No accounts configured")
        print("-" * 80)
    except Exception as e:
        print(f"\\n⚠️  Could not check credentials: {e}")

    print("""
Available options:
  1. Setup Credentials (Configure Token & Accounts)
  2. Download Data → Consolidate → Test (Full Workflow)
  3. Test Connection
  0. Back to Main Menu

💡 First time? Start with option 1 to setup credentials
    """)

    choice = get_input("Select option", ["0", "1", "2", "3"])'''

content = content.replace(old_auth_menu, new_auth_menu)

# Update the choice handlers
old_handlers = """    result = False
    try:
        if choice == "0":
            result = False
        elif choice == "1":
            if context:
                context.last_action = "Select MetaAPI Account"
            result = select_metaapi_account(wf_manager)
        elif choice == "2":
            if context:
                context.last_action = "Test Connection"
            result = test_connection(wf_manager)"""

new_handlers = """    result = False
    try:
        if choice == "0":
            result = False
        elif choice == "1":
            if context:
                context.last_action = "Setup Credentials"
            result = setup_metaapi_credentials_menu(wf_manager)
            if result:
                print("\\n✅ Credentials configured!")
                input("\\n📊 Press Enter to continue...")
        elif choice == "2":
            if context:
                context.last_action = "Download Data Workflow"
            result = select_metaapi_account(wf_manager)
        elif choice == "3":
            if context:
                context.last_action = "Test Connection"
            result = test_connection(wf_manager)"""

content = content.replace(old_handlers, new_handlers)

# Update select_metaapi_account to check credentials first and streamline flow
old_select_start = '''def select_metaapi_account(wf_manager: WorkflowManager) -> bool:
    """Select and authenticate with MetaAPI account."""
    print_submenu_header("Select MetaAPI Account")

    print("\\n📋 Launching account selection...")

    try:
        import subprocess

        result = subprocess.run(
            [sys.executable, "scripts/download/select_metaapi_account.py"], stderr=subprocess.STDOUT
        )

        if result.returncode == 0:
            print("\\n✅ Account selected successfully")

            # Prompt to download data
            print("\\n" + "=" * 80)
            print("  NEXT STEP: Download Market Data")
            print("=" * 80)
            print("\\nWould you like to download data now?")
            print("  [1] Yes - Launch interactive data download")
            print("  [2] No - Return to main menu")

            choice = get_input("\\nSelect option", ["1", "2"], default="1")

            if choice == "1":
                print("\\n📥 Launching interactive download...")
                result = subprocess.run(
                    [sys.executable, "scripts/download/download_interactive.py"], check=False
                )
                if result.returncode == 0:'''

new_select_start = '''def select_metaapi_account(wf_manager: WorkflowManager) -> bool:
    """Download data, consolidate, and optionally run tests - full workflow."""
    print_submenu_header("Download Data → Consolidate → Test")

    # Check if credentials are configured
    try:
        from scripts.utils.secure_token_helper import get_metaapi_token, list_configured_accounts

        token = get_metaapi_token()
        accounts = list_configured_accounts()

        if not token or not accounts:
            print("\\n⚠️  Credentials not configured!")
            print("\\nWould you like to setup credentials now?")
            print("  [1] Yes - Run credential setup")
            print("  [2] No - Return to menu")

            choice = get_input("\\nSelect option", ["1", "2"], default="1")

            if choice == "1":
                if setup_metaapi_credentials_menu(wf_manager):
                    print("\\n✅ Credentials configured! Continuing with download...\\n")
                else:
                    print("\\n❌ Credential setup failed or cancelled")
                    return False
            else:
                return False
    except Exception as e:
        print(f"\\n⚠️  Could not verify credentials: {e}")

    print("\\n📥 Launching interactive download...")

    try:
        import subprocess

        download_result = subprocess.run(
            [sys.executable, "scripts/download/download_interactive.py"], check=False
        )

        if download_result.returncode == 0:'''

content = content.replace(old_select_start, new_select_start)

# Simplify the rest of select_metaapi_account (remove nested account selection prompts)
old_nested = """                    print("\\n✅ Download complete!")

                    # Prompt for data preparation
                    print("\\n" + "=" * 80)
                    print("  NEXT STEP: Prepare Data for Testing")
                    print("=" * 80)
                    print("\\nWould you like to consolidate and prepare the data?")
                    print("  [1] Yes - Run data consolidation")
                    print("  [2] No - I'll do it later")

                    prep_choice = get_input("\\nSelect option", ["1", "2"], default="1")

                    if prep_choice == "1":"""

new_nested = """            print("\\n✅ Download complete!")

            # Prompt for data preparation
            print("\\n" + "=" * 80)
            print("  NEXT STEP: Prepare Data for Testing")
            print("=" * 80)
            print("\\nWould you like to consolidate and prepare the data?")
            print("  [1] Yes - Run data consolidation")
            print("  [2] No - I'll do it later")

            prep_choice = get_input("\\nSelect option", ["1", "2"], default="1")

            if prep_choice == "1":"""

content = content.replace(old_nested, new_nested)

# Fix the ending of select_metaapi_account
old_ending = """                        else:
                            print("\\n⚠️ Data consolidation completed with warnings")
                    else:
                        print("\\n✅ You can consolidate data later from Data Management menu")
                else:
                    print(f"\\n⚠️ Download completed with warnings (exit code {result.returncode})")
            else:
                print("\\n✅ You can download data later from Data Management menu")

            input("\\n📊 Press Enter to return to menu...")
            return True
        else:
            print("\\n❌ Account selection failed")
            return False

    except Exception as e:
        print(f"\\n❌ Error selecting account: {e}")
        return False"""

new_ending = """                else:
                    print("\\n⚠️ Data consolidation completed with warnings")
            else:
                print("\\n✅ You can consolidate data later from Data Management menu")
        else:
            print(f"\\n⚠️ Download completed with warnings (exit code {download_result.returncode})")

        input("\\n📊 Press Enter to return to menu...")
        return download_result.returncode == 0

    except Exception as e:
        print(f"\\n❌ Error in workflow: {e}")
        return False"""

content = content.replace(old_ending, new_ending)

# Write the updated file
menu_file.write_text(content)

print("✅ Successfully updated kinetra_menu.py")
print("\nChanges made:")
print("  • Added setup_metaapi_credentials_menu() function")
print("  • Added credential status display in authentication menu")
print("  • Updated menu options to include credential setup")
print("  • Streamlined download → consolidate → test workflow")
print("  • Added credential check before downloading data")
print("\nYou can now run: python kinetra_menu.py")
