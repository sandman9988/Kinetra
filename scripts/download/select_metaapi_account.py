"""
Interactive MetaAPI Account Selector
=====================================

Simple, clean interface to:
1. List all MetaAPI broker accounts
2. Use arrow keys to select account
3. Hit Enter to connect

Usage:
    python scripts/select_metaapi_account.py

Environment Variables:
    METAAPI_TOKEN - Your MetaAPI API token (required)
"""

import asyncio
import os
import sys
from pathlib import Path
from typing import List, Optional

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Try to import metaapi
try:
    from metaapi_cloud_sdk import MetaApi

    METAAPI_AVAILABLE = True
except ImportError:
    METAAPI_AVAILABLE = False
    print("❌ metaapi-cloud-sdk package not installed")
    print("   Install with: pip install metaapi-cloud-sdk")
    sys.exit(1)

# Try to import inquirer for interactive menu
try:
    import inquirer

    INQUIRER_AVAILABLE = True
except ImportError:
    INQUIRER_AVAILABLE = False
    print("⚠️  inquirer package not installed - using simple numbered selection")
    print("   For arrow key navigation, install: pip install inquirer")


def get_api_token() -> Optional[str]:
    """
    Get MetaAPI token from environment or secure prompt.

    Returns:
        API token string, or None if not available
    """
    # Try environment first
    token = os.getenv("METAAPI_TOKEN")
    if token:
        return token

    # Try loading from .env file
    env_file = Path(__file__).parent.parent.parent / ".env"
    if env_file.exists():
        try:
            with open(env_file, "r") as f:
                for line in f:
                    if line.startswith("METAAPI_TOKEN="):
                        token = line.split("=", 1)[1].strip().strip('"').strip("'")
                        if token:
                            return token
        except Exception as e:
            print(f"⚠️  Warning: Could not read .env file: {e}")

    # Prompt user
    print("\n⚠️  METAAPI_TOKEN not found in environment or .env file")
    print("   Get your token from: https://app.metaapi.cloud/token")
    print()

    try:
        import getpass

        token = getpass.getpass("Enter your MetaAPI token (input hidden): ").strip()
        if token:
            # Offer to save
            save = input("\nSave token to .env file for future use? (y/N): ").strip().lower()
            if save == "y":
                try:
                    with open(env_file, "a") as f:
                        f.write(f"\n# MetaAPI Token (added by select_metaapi_account.py)\n")
                        f.write(f"METAAPI_TOKEN={token}\n")
                    print(f"✅ Token saved to {env_file}")
                    print("   Make sure .env is in .gitignore!")
                except Exception as e:
                    print(f"⚠️  Could not save token: {e}")
            return token
    except KeyboardInterrupt:
        print("\n❌ Cancelled")
        return None

    return None


async def list_and_select_account(api_token: str) -> Optional[dict]:
    """
    List all MetaAPI accounts and allow interactive selection.

    Returns:
        dict with 'id', 'name', 'login', 'server' keys, or None if cancelled
    """
    print("\n" + "=" * 80)
    print(" " * 25 + "METAAPI ACCOUNT SELECTOR")
    print("=" * 80)

    # Initialize MetaAPI
    print("\n[Connecting to MetaAPI...]")
    try:
        api = MetaApi(api_token)
    except Exception as e:
        print(f"❌ Failed to initialize MetaAPI: {e}")
        return None

    # Fetch accounts
    print("[Fetching your accounts...]")
    try:
        accounts = await api.metatrader_account_api.get_accounts_with_infinite_scroll_pagination()
    except Exception as e:
        print(f"❌ Failed to fetch accounts: {e}")
        return None

    if not accounts or len(accounts) == 0:
        print("\n❌ No accounts found")
        print("   Add an MT5 account at: https://app.metaapi.cloud/accounts")
        return None

    print(f"\n✅ Found {len(accounts)} account(s)\n")

    # Prepare account choices
    account_choices = []
    account_map = {}

    for i, acc in enumerate(accounts):
        # Build display name
        display_name = f"{acc.name} (Login: {acc.login}, Server: {acc.server})"
        account_choices.append(display_name)
        account_map[display_name] = {
            "id": acc.id,
            "name": acc.name,
            "login": acc.login,
            "server": acc.server,
            "type": acc.type if hasattr(acc, "type") else "N/A",
            "state": acc.state if hasattr(acc, "state") else "N/A",
        }

    # Interactive selection
    if INQUIRER_AVAILABLE and len(accounts) > 1:
        # Arrow key selection
        print("Use ↑/↓ arrow keys to navigate, Enter to select:\n")
        questions = [
            inquirer.List(
                "account",
                message="Select your MetaAPI account",
                choices=account_choices,
            ),
        ]

        try:
            answers = inquirer.prompt(questions)
            if answers is None:  # User cancelled (Ctrl+C)
                print("\n❌ Selection cancelled")
                return None

            selected_display = answers["account"]
            selected_account = account_map[selected_display]

        except KeyboardInterrupt:
            print("\n❌ Selection cancelled")
            return None

    else:
        # Simple numbered selection (fallback)
        print("Available accounts:\n")
        for i, choice in enumerate(account_choices, 1):
            print(f"  [{i}] {choice}")

        if len(accounts) == 1:
            print(f"\n✅ Auto-selecting the only account")
            selected_account = account_map[account_choices[0]]
        else:
            # Ask user to pick by number
            while True:
                try:
                    selection = input(f"\nSelect account [1-{len(accounts)}]: ").strip()
                    idx = int(selection) - 1
                    if 0 <= idx < len(accounts):
                        selected_account = account_map[account_choices[idx]]
                        break
                    else:
                        print(f"❌ Invalid selection. Choose 1-{len(accounts)}")
                except ValueError:
                    print(f"❌ Invalid input. Enter a number 1-{len(accounts)}")
                except KeyboardInterrupt:
                    print("\n❌ Selection cancelled")
                    return None

    # Display selection
    print("\n" + "=" * 80)
    print("✅ SELECTED ACCOUNT")
    print("=" * 80)
    print(f"  Account ID:  {selected_account['id']}")
    print(f"  Name:        {selected_account['name']}")
    print(f"  Login:       {selected_account['login']}")
    print(f"  Server:      {selected_account['server']}")
    print(f"  Type:        {selected_account['type']}")
    print(f"  State:       {selected_account['state']}")
    print("=" * 80)

    return selected_account


async def main():
    """Main entry point."""
    # Get API token securely
    api_token = get_api_token()

    if not api_token:
        print("❌ ERROR: No API token available")
        print("   Set METAAPI_TOKEN environment variable or add to .env file")
        print("   Get your token from: https://app.metaapi.cloud/token")
        return

    # Select account
    selected = await list_and_select_account(api_token)

    if selected:
        print(f"\n✅ Success! Use this account ID in your scripts:")
        print(f'\n   ACCOUNT_ID = "{selected["id"]}"')
        print(f"\nNext steps:")
        print(f"  1. Update test_mt5_vantage_full.py with this ACCOUNT_ID")
        print(f"  2. Run the backtest to get real broker specs from MetaAPI")
    else:
        print(f"\n❌ No account selected")


if __name__ == "__main__":
    asyncio.run(main())
