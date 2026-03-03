#!/usr/bin/env python3
"""
MetaAPI Credentials Setup
==========================

Interactive script to configure MetaAPI credentials securely.

Features:
- Tests existing credentials before prompting for changes
- Allows updating account ID without re-entering token
- Validates credentials with API call before saving
- Saves to .env file (atomic write)
- Adds to .gitignore automatically
- Supports optional encryption

Usage:
    python scripts/download/setup_metaapi_credentials.py

Output:
    - .env file with METAAPI_TOKEN and METAAPI_ACCOUNT_ID
    - Validates credentials work before saving
"""

import asyncio
import getpass
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Try to import metaapi
try:
    from metaapi_cloud_sdk import MetaApi

    METAAPI_AVAILABLE = True
except ImportError:
    METAAPI_AVAILABLE = False


def print_header(text: str):
    """Print section header."""
    print(f"\n{'=' * 80}")
    print(f"  {text}")
    print(f"{'=' * 80}\n")


def print_token_type_info():
    """Print information about different token types."""
    print("\n" + "─" * 80)
    print("⚠️  IMPORTANT: MetaAPI Token Types")
    print("─" * 80)
    print("""
There are TWO types of MetaAPI tokens:

1. 🔑 API Access Token (Recommended for Kinetra)
   - Can list/manage all accounts
   - Get from: https://app.metaapi.cloud/api-access/generate-token
   - Use this for: Data downloads, account discovery, full system access

2. 🔐 Account Access Token
   - Limited to ONE specific account
   - Get from individual account settings
   - Use this for: Live trading with a single account only

For Kinetra setup, you need an API Access Token (#1).
""")
    print("─" * 80)


def get_existing_credentials() -> tuple:
    """
    Load existing credentials from .env if available.

    Returns:
        Tuple of (token, account_id) or (None, None)
    """
    env_file = PROJECT_ROOT / ".env"

    if not env_file.exists():
        return None, None

    token = None
    account_id = None

    try:
        with open(env_file, "r") as f:
            for line in f:
                line = line.strip()
                if line.startswith("METAAPI_TOKEN="):
                    token = line.split("=", 1)[1].strip().strip('"').strip("'")
                    # Remove ENC:: prefix if present (old encryption format)
                    if token.startswith("ENC::"):
                        token = token[5:]
                elif line.startswith("METAAPI_ACCOUNT_ID="):
                    account_id = line.split("=", 1)[1].strip().strip('"').strip("'")
                    if account_id.startswith("ENC::"):
                        account_id = account_id[5:]
    except Exception as e:
        print(f"⚠️  Warning: Could not read .env: {e}")

    return token, account_id


async def test_credentials(token: str, account_id: str = None, silent: bool = False) -> tuple:
    """
    Test credentials by making API call.

    Args:
        token: MetaAPI token
        account_id: MetaAPI account ID (optional)
        silent: If True, suppress output

    Returns:
        Tuple of (is_valid, accounts_list, error_message)
    """
    if not METAAPI_AVAILABLE:
        if not silent:
            print("⚠️  Cannot validate - metaapi-cloud-sdk not installed")
            print("   Install with: pip install metaapi-cloud-sdk")
        return False, [], "metaapi-cloud-sdk not installed"

    try:
        api = MetaApi(token=token)

        # Get accounts
        try:
            accounts = (
                await api.metatrader_account_api.get_accounts_with_infinite_scroll_pagination()
            )
        except Exception as e:
            error_msg = str(e)

            # Check if this is an account token vs API token issue
            if (
                "account access token" in error_msg.lower()
                or "api access token" in error_msg.lower()
            ):
                if not silent:
                    print("❌ Wrong token type detected!")
                    print_token_type_info()
                    print("\n💡 Your token appears to be an Account Access Token.")
                    print("   For Kinetra, you need an API Access Token instead.")
                    print(
                        "\n   Get one here: https://app.metaapi.cloud/api-access/generate-token\n"
                    )
                return False, [], "Account token provided, need API token"
            else:
                raise  # Re-raise if it's a different error

        if not accounts:
            if not silent:
                print("❌ No accounts found for this token")
            return False, [], "No accounts found"

        if not silent:
            print(f"\n✅ Token valid! Found {len(accounts)} account(s):")
            for acc in accounts:
                status_icon = "✅" if acc.state == "DEPLOYED" else "⚠️"
                print(f"  {status_icon} {acc.name} ({acc.id}) - {acc.type} - {acc.state}")

        # If account_id provided, verify it exists
        if account_id:
            account_ids = [acc.id for acc in accounts]
            if account_id in account_ids:
                if not silent:
                    print(f"\n✅ Account ID '{account_id}' verified")
                return True, accounts, None
            else:
                if not silent:
                    print(f"\n❌ Account ID '{account_id}' not found")
                    print(f"   Available IDs: {', '.join(account_ids)}")
                return False, accounts, f"Account ID '{account_id}' not found"

        return True, accounts, None

    except Exception as e:
        error_msg = str(e)
        if not silent:
            print(f"❌ Validation failed: {error_msg}")
        return False, [], error_msg


def select_account_from_list(accounts) -> str:
    """
    Interactive account selection from list.

    Args:
        accounts: List of MetaAPI account objects

    Returns:
        Selected account ID or None
    """
    if not accounts:
        return None

    print("\n📋 Available Accounts:")
    print("─" * 80)

    for i, acc in enumerate(accounts, 1):
        status_icon = "✅" if acc.state == "DEPLOYED" else "⚠️"
        print(f"  {i}. {status_icon} {acc.name}")
        print(f"     ID: {acc.id}")
        print(f"     Type: {acc.type} | State: {acc.state}")
        print()

    while True:
        try:
            choice = input(f"Select account (1-{len(accounts)}) or 0 to skip: ").strip()

            if choice == "0":
                return None

            idx = int(choice) - 1
            if 0 <= idx < len(accounts):
                return accounts[idx].id
            else:
                print(f"❌ Invalid choice. Enter 1-{len(accounts)} or 0 to skip.")
        except ValueError:
            print("❌ Invalid input. Enter a number.")
        except KeyboardInterrupt:
            print("\n\n❌ Cancelled")
            return None


def save_credentials_to_env(token: str, account_id: str = None):
    """
    Save credentials to .env file.

    Args:
        token: MetaAPI token
        account_id: MetaAPI account ID (optional)
    """
    env_file = PROJECT_ROOT / ".env"

    # Read existing .env if it exists
    env_lines = {}
    if env_file.exists():
        try:
            with open(env_file, "r") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        key, value = line.split("=", 1)
                        env_lines[key] = value
        except Exception as e:
            print(f"⚠️  Warning: Could not read existing .env: {e}")

    # Update credentials
    env_lines["METAAPI_TOKEN"] = token
    if account_id:
        env_lines["METAAPI_ACCOUNT_ID"] = account_id
    elif "METAAPI_ACCOUNT_ID" in env_lines and not account_id:
        # Keep existing account ID if none provided
        pass

    # Write to file
    try:
        with open(env_file, "w") as f:
            f.write("# Kinetra MetaAPI Credentials\n")
            f.write("# Auto-generated - DO NOT COMMIT TO GIT\n")
            f.write("# Get your token from: https://app.metaapi.cloud/token\n\n")

            for key, value in env_lines.items():
                f.write(f"{key}={value}\n")

        print(f"\n✅ Credentials saved to: {env_file}")
        print(f"   File size: {env_file.stat().st_size} bytes")

    except Exception as e:
        print(f"\n❌ Failed to save credentials: {e}")
        raise

    # Add to .gitignore
    gitignore = PROJECT_ROOT / ".gitignore"
    if gitignore.exists():
        content = gitignore.read_text()
        if ".env" not in content:
            try:
                with open(gitignore, "a") as f:
                    f.write("\n# Environment variables (DO NOT COMMIT)\n")
                    f.write(".env\n")
                print("✅ Added .env to .gitignore")
            except Exception as e:
                print(f"⚠️  Warning: Could not update .gitignore: {e}")
    else:
        print("⚠️  Warning: .gitignore not found - make sure .env is not committed!")


def main():
    """Main setup flow."""
    print_header("MetaAPI Credentials Setup")

    print("""
This script will help you configure MetaAPI credentials for Kinetra.

You'll need:
1. MetaAPI API Access Token (from https://app.metaapi.cloud/api-access/generate-token)
2. MetaAPI Account ID (selected from your available accounts)

Your credentials will be stored in .env file (local only, not committed).
""")

    print_token_type_info()

    # Check if credentials already exist
    existing_token, existing_account_id = get_existing_credentials()

    token = existing_token
    account_id = existing_account_id

    if existing_token:
        print("📋 Found existing credentials:")
        print(f"   Token: {existing_token[:10]}...{existing_token[-10:]}")
        if existing_account_id:
            print(f"   Account ID: {existing_account_id}")

        # Test existing credentials
        print("\n🔍 Testing existing credentials...")
        is_valid, accounts, error = asyncio.run(
            test_credentials(existing_token, existing_account_id, silent=False)
        )

        if is_valid:
            print("\n✅ Existing credentials are valid!")

            # Give options
            print("\nWhat would you like to do?")
            print("  1. Keep current credentials (no changes)")
            print("  2. Update/select Account ID only")
            print("  3. Update Token (replace everything)")
            print("  0. Cancel")

            choice = input("\nSelect option (0-3): ").strip()

            if choice == "0":
                print("\n👋 No changes made. Exiting.")
                return
            elif choice == "1":
                print("\n✅ Keeping existing credentials.")
                return
            elif choice == "2":
                # Update account ID only
                print("\n" + "─" * 80)
                print("UPDATE ACCOUNT ID")
                print("─" * 80)

                if accounts:
                    new_account_id = select_account_from_list(accounts)
                    if new_account_id:
                        account_id = new_account_id
                        save_credentials_to_env(token, account_id)
                        print("\n✅ Account ID updated successfully!")
                    else:
                        print("\n❌ No account selected. No changes made.")
                else:
                    print("\n❌ No accounts available to select from.")
                return
            elif choice == "3":
                # Continue to token update below
                print("\n⚠️  Updating token (existing token will be replaced)...")
                token = None  # Force re-entry
            else:
                print("\n❌ Invalid choice. Exiting.")
                return
        else:
            print(f"\n❌ Existing credentials are INVALID: {error}")
            print("   You need to update your credentials.")

            update = input("\nUpdate credentials now? (Y/n): ").strip().lower()
            if update == "n":
                print("\n👋 Exiting. Run again when ready to update.")
                return

    # Get new token (only if we don't have a valid one)
    if not token:
        print("\n" + "─" * 80)
        print("STEP 1: MetaAPI API Access Token")
        print("─" * 80)
        print("\n⚠️  Make sure to use an API Access Token (not Account Token)!")
        print("Get your API token from: https://app.metaapi.cloud/api-access/generate-token")
        print("(Your input will be hidden for security)")

        try:
            token = getpass.getpass("\nEnter MetaAPI API Access Token: ").strip()
        except KeyboardInterrupt:
            print("\n\n❌ Setup cancelled")
            return

        if not token:
            print("\n❌ Token cannot be empty")
            return

        # Validate new token
        print("\n🔍 Validating token...")

        is_valid, accounts, error = asyncio.run(test_credentials(token, None, silent=False))

        if not is_valid:
            print(f"\n❌ Token validation failed: {error}")

            if "account token provided" in error.lower():
                print("\n💡 TIP: You need an API Access Token, not an Account Access Token.")
                print("   Go to: https://app.metaapi.cloud/api-access/generate-token")

            retry = input("\nTry again? (y/N): ").strip().lower()
            if retry == "y":
                main()
            return

    # Get account ID if we don't have one or user wants to select
    if not account_id:
        print("\n" + "─" * 80)
        print("STEP 2: MetaAPI Account ID")
        print("─" * 80)

        # Get accounts list if we don't have it yet
        if "accounts" not in locals() or not accounts:
            is_valid, accounts, error = asyncio.run(test_credentials(token, None, silent=True))

        if accounts:
            print("\nSelect an account from the list below:")
            account_id = select_account_from_list(accounts)

            if not account_id:
                print("\n⚠️  No account selected.")
                save_anyway = input("Save token without account ID? (y/N): ").strip().lower()
                if save_anyway != "y":
                    print("\n❌ Setup cancelled")
                    return
        else:
            print("\n⚠️  No accounts found. You can add an account ID manually later.")

    # Final validation if we have both
    if token and account_id:
        print("\n🔍 Final validation...")
        is_valid, _, error = asyncio.run(test_credentials(token, account_id, silent=False))

        if not is_valid:
            print(f"\n❌ Validation failed: {error}")
            save_anyway = input("\nSave anyway? (y/N): ").strip().lower()
            if save_anyway != "y":
                print("\n❌ Setup cancelled")
                return

    # Save credentials
    print("\n" + "─" * 80)
    print("SAVING CREDENTIALS")
    print("─" * 80)

    save_credentials_to_env(token, account_id if account_id else None)

    # Final summary
    print_header("Setup Complete!")

    print("✅ Credentials configured successfully")
    print("\nNext steps:")
    print("  1. Download data (Menu 2 → 2)")
    print("  2. Run discovery (Menu 2 → 1)")
    print("  3. Start backtesting (Menu 4 → 1)")

    print("\n⚠️  SECURITY REMINDER:")
    print("  - Never commit .env to git")
    print("  - Never share your token")
    print("  - Rotate token if compromised")

    print(f"\n📄 Configuration saved to: {env_file}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Setup cancelled by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
