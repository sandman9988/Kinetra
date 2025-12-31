"""
Interactive MetaAPI Account Selector
=====================================

Simple, clean interface to:
1. List all MetaAPI broker accounts
2. Use arrow keys to select account
3. Hit Enter to connect

Usage:
    python scripts/select_metaapi_account.py
"""

import sys
from pathlib import Path
from typing import List, Optional
import asyncio

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


async def list_and_select_account(api_token: str) -> Optional[dict]:
    """
    List all MetaAPI accounts and allow interactive selection.

    Returns:
        dict with 'id', 'name', 'login', 'server' keys, or None if cancelled
    """
    print("\n" + "="*80)
    print(" "*25 + "METAAPI ACCOUNT SELECTOR")
    print("="*80)

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
            'id': acc.id,
            'name': acc.name,
            'login': acc.login,
            'server': acc.server,
            'type': acc.type if hasattr(acc, 'type') else 'N/A',
            'state': acc.state if hasattr(acc, 'state') else 'N/A',
        }

    # Interactive selection
    if INQUIRER_AVAILABLE and len(accounts) > 1:
        # Arrow key selection
        print("Use ↑/↓ arrow keys to navigate, Enter to select:\n")
        questions = [
            inquirer.List(
                'account',
                message="Select your MetaAPI account",
                choices=account_choices,
            ),
        ]

        try:
            answers = inquirer.prompt(questions)
            if answers is None:  # User cancelled (Ctrl+C)
                print("\n❌ Selection cancelled")
                return None

            selected_display = answers['account']
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
    print("\n" + "="*80)
    print("✅ SELECTED ACCOUNT")
    print("="*80)
    print(f"  Account ID:  {selected_account['id']}")
    print(f"  Name:        {selected_account['name']}")
    print(f"  Login:       {selected_account['login']}")
    print(f"  Server:      {selected_account['server']}")
    print(f"  Type:        {selected_account['type']}")
    print(f"  State:       {selected_account['state']}")
    print("="*80)

    return selected_account


async def main():
    """Main entry point."""
    API_TOKEN = "eyJhbGciOiJSUzUxMiIsInR5cCI6IkpXVCJ9.eyJfaWQiOiJjMTdhODAwNThhOWE3OWE0NDNkZjBlOGM1NDZjZjlmMSIsImFjY2Vzc1J1bGVzIjpbeyJpZCI6InRyYWRpbmctYWNjb3VudC1tYW5hZ2VtZW50LWFwaSIsIm1ldGhvZHMiOlsidHJhZGluZy1hY2NvdW50LW1hbmFnZW1lbnQtYXBpOnJlc3Q6cHVibGljOio6KiJdLCJyb2xlcyI6WyJyZWFkZXIiLCJ3cml0ZXIiXSwicmVzb3VyY2VzIjpbIio6JFVTRVJfSUQkOioiXX0seyJpZCI6Im1ldGFhcGktcmVzdC1hcGkiLCJtZXRob2RzIjpbIm1ldGFhcGktYXBpOnJlc3Q6cHVibGljOio6KiJdLCJyb2xlcyI6WyJyZWFkZXIiLCJ3cml0ZXIiXSwicmVzb3VyY2VzIjpbIio6JFVTRVJfSUQkOioiXX0seyJpZCI6Im1ldGFhcGktcnBjLWFwaSIsIm1ldGhvZHMiOlsibWV0YWFwaS1hcGk6d3M6cHVibGljOio6KiJdLCJyb2xlcyI6WyJyZWFkZXIiLCJ3cml0ZXIiXSwicmVzb3VyY2VzIjpbIio6JFVTRVJfSUQkOioiXX0seyJpZCI6Im1ldGFhcGktcmVhbC10aW1lLXN0cmVhbWluZy1hcGkiLCJtZXRob2RzIjpbIm1ldGFhcGktYXBpOndzOnB1YmxpYzoqOioiXSwicm9sZXMiOlsicmVhZGVyIiwid3JpdGVyIl0sInJlc291cmNlcyI6WyIqOiRVU0VSX0lEJDoqIl19LHsiaWQiOiJtZXRhc3RhdHMtYXBpIiwibWV0aG9kcyI6WyJtZXRhc3RhdHMtYXBpOnJlc3Q6cHVibGljOio6KiJdLCJyb2xlcyI6WyJyZWFkZXIiLCJ3cml0ZXIiXSwicmVzb3VyY2VzIjpbIio6JFVTRVJfSUQkOioiXX0seyJpZCI6InJpc2stbWFuYWdlbWVudC1hcGkiLCJtZXRob2RzIjpbInJpc2stbWFuYWdlbWVudC1hcGk6cmVzdDpwdWJsaWM6KjoqIl0sInJvbGVzIjpbInJlYWRlciIsIndyaXRlciJdLCJyZXNvdXJjZXMiOlsiKjokVVNFUl9JRCQ6KiJdfSx7ImlkIjoiY29weWZhY3RvcnktYXBpIiwibWV0aG9kcyI6WyJjb3B5ZmFjdG9yeS1hcGk6cmVzdDpwdWJsaWM6KjoqIl0sInJvbGVzIjpbInJlYWRlciIsIndyaXRlciJdLCJyZXNvdXJjZXMiOlsiKjokVVNFUl9JRCQ6KiJdfSx7ImlkIjoibXQtbWFuYWdlci1hcGkiLCJtZXRob2RzIjpbIm10LW1hbmFnZXItYXBpOnJlc3Q6ZGVhbGluZzoqOioiLCJtdC1tYW5hZ2VyLWFwaTpyZXN0OnB1YmxpYzoqOioiXSwicm9sZXMiOlsicmVhZGVyIiwid3JpdGVyIl0sInJlc291cmNlcyI6WyIqOiRVU0VSX0lEJDoqIl19LHsiaWQiOiJiaWxsaW5nLWFwaSIsIm1ldGhvZHMiOlsiYmlsbGluZy1hcGk6cmVzdDpwdWJsaWM6KjoqIl0sInJvbGVzIjpbInJlYWRlciJdLCJyZXNvdXJjZXMiOlsiKjokVVNFUl9JRCQ6KiJdfV0sImlnbm9yZVJhdGVMaW1pdHMiOmZhbHNlLCJ0b2tlbklkIjoiMjAyMTAyMTMiLCJpbXBlcnNvbmF0ZWQiOmZhbHNlLCJyZWFsVXNlcklkIjoiYzE3YTgwMDU4YTlhNzlhNDQzZGYwZThjNTQ2Y2Y5ZjEiLCJpYXQiOjE3NjcxMjE0MDYsImV4cCI6MTc3NDg5NzQwNn0.oHB_5iSe2nm_lWbIRTvFKDy1sXkq1xMXaROHvoJjgXAa2n8OkjJuj6bYbqAO4F_xHrEEKykjEgWN6Vfm7tMG9AU1o-XoHf3ayUMro90NLq-kUTcMBoE6GkAPugQenj3-1oySJ7nlnZdH-luqSRhpcnHob91O4kO670gzKUbWO2jCTpr_9d8ZzRHHqTQbukW-JdNQ53C0KSR7RGg50MBGr55IlyDmMsstqznsmCms7vDkbtoxRfUWssMOZ-4eKA-wtJJz47jQUAnJEDwGFWwoKweIhK_WnjJgFfJoOP7S_7rLBr6elkhQbzd5xENGJqmNj1I0CdiiQpNuDX5sLCt2PLvQ-Owll3LdBDpRGlb-rWJR4gaAPY3nZzCaMakfjRmZtQsCN9FGvOphG0b0IQAD3sQKZ2FzO08IenPWiZS90s4mP88vmafnC-lybMWWXKT8CnQu4YSgFsY-v74lJ_xGi6Ye-4nwzECrkGam9WceD5cGnk8bDchH-4WN68LAjPnKg0XxABd1AYnops89qcmzupoiM34BfaigMLYin5Ea81YgvGcSEwF8UQ070SDdGL2NptuznhMA2iCJoGwF0FN-uKA-jBQvPcyUEDUTjl3cbV9JECry7uAk_HeQKPzF2l0KQBOqENAytnNyWYwaq9lY3XsH7d5ZG35jFzeFCCdrokA"

    if API_TOKEN is None:
        print("❌ ERROR: API_TOKEN not set")
        print("   Edit this script and add your token")
        return

    # Select account
    selected = await list_and_select_account(API_TOKEN)

    if selected:
        print(f"\n✅ Success! Use this account ID in your scripts:")
        print(f"\n   ACCOUNT_ID = \"{selected['id']}\"")
        print(f"\nNext steps:")
        print(f"  1. Update test_mt5_vantage_full.py with this ACCOUNT_ID")
        print(f"  2. Run the backtest to get real broker specs from MetaAPI")
    else:
        print(f"\n❌ No account selected")


if __name__ == '__main__':
    asyncio.run(main())
