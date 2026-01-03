"""
Secure Token Helper
===================

Read-only utility for accessing MetaAPI credentials from environment.
All credential setup should be done via scripts/setup_metaapi_credentials.py

Usage:
    from scripts.utils.secure_token_helper import get_metaapi_token, require_token

    # Option 1: Get token or None
    token = get_metaapi_token()
    if not token:
        print("Run: python scripts/setup_metaapi_credentials.py")
        sys.exit(1)

    # Option 2: Get token or exit automatically
    token = require_token()  # Exits if not found
"""

import os
import sys
from pathlib import Path
from typing import Optional


def get_metaapi_token() -> Optional[str]:
    """
    Get MetaAPI token from environment or .env file.

    Checks in order:
    1. METAAPI_TOKEN environment variable
    2. .env file in project root

    Returns:
        API token string, or None if not available
    """
    # Try environment first
    token = os.getenv("METAAPI_TOKEN")
    if token:
        return token

    # Try loading from .env file
    project_root = Path(__file__).parent.parent.parent
    env_file = project_root / ".env"

    if env_file.exists():
        try:
            with open(env_file, "r") as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("METAAPI_TOKEN="):
                        token = line.split("=", 1)[1].strip().strip('"').strip("'")
                        if token:
                            return token
        except Exception as e:
            print(f"⚠️  Warning: Could not read .env file: {e}", file=sys.stderr)

    return None


def get_metaapi_account_id(account_label: Optional[str] = None) -> Optional[str]:
    """
    Get MetaAPI account ID from environment or .env file.

    Args:
        account_label: Optional account label (e.g., 'VANTAGE', 'DEMO')
                      If None, returns default METAAPI_ACCOUNT_ID

    Returns:
        Account ID string, or None if not available
    """
    # Determine which env var to look for
    if account_label:
        env_var = f"METAAPI_ACCOUNT_{account_label.upper()}"
    else:
        env_var = "METAAPI_ACCOUNT_ID"

    # Try environment first
    account_id = os.getenv(env_var)
    if account_id:
        return account_id

    # Try loading from .env file
    project_root = Path(__file__).parent.parent.parent
    env_file = project_root / ".env"

    if env_file.exists():
        try:
            with open(env_file, "r") as f:
                for line in f:
                    line = line.strip()
                    if line.startswith(f"{env_var}="):
                        account_id = line.split("=", 1)[1].strip().strip('"').strip("'")
                        if account_id:
                            return account_id
        except Exception as e:
            print(f"⚠️  Warning: Could not read .env file: {e}", file=sys.stderr)

    return None


def require_token() -> str:
    """
    Get token or exit if not available.

    Returns:
        Token string (guaranteed non-None)

    Raises:
        SystemExit if token not available
    """
    token = get_metaapi_token()
    if not token:
        print("\n❌ ERROR: No MetaAPI token found", file=sys.stderr)
        print("   Run the setup script to configure credentials:", file=sys.stderr)
        print("   python scripts/setup_metaapi_credentials.py", file=sys.stderr)
        print()
        print("   Or manually set METAAPI_TOKEN environment variable", file=sys.stderr)
        print("   Get your token from: https://app.metaapi.cloud/token", file=sys.stderr)
        sys.exit(1)
    return token


def require_account_id(account_label: Optional[str] = None) -> str:
    """
    Get account ID or exit if not available.

    Args:
        account_label: Optional account label (e.g., 'VANTAGE', 'DEMO')

    Returns:
        Account ID string (guaranteed non-None)

    Raises:
        SystemExit if account ID not available
    """
    account_id = get_metaapi_account_id(account_label)
    if not account_id:
        if account_label:
            var_name = f"METAAPI_ACCOUNT_{account_label.upper()}"
            print(f"\n❌ ERROR: No account ID found for '{account_label}'", file=sys.stderr)
            print(f"   Environment variable {var_name} not set", file=sys.stderr)
        else:
            print("\n❌ ERROR: No default MetaAPI account ID found", file=sys.stderr)
            print("   Environment variable METAAPI_ACCOUNT_ID not set", file=sys.stderr)

        print()
        print("   Run the setup script to configure credentials:", file=sys.stderr)
        print("   python scripts/setup_metaapi_credentials.py", file=sys.stderr)
        sys.exit(1)
    return account_id


def list_configured_accounts() -> dict:
    """
    List all configured MetaAPI accounts.

    Returns:
        Dictionary mapping account labels to account IDs
    """
    accounts = {}

    # Get default account
    default_id = get_metaapi_account_id()
    if default_id:
        accounts["DEFAULT"] = default_id

    # Try loading all METAAPI_ACCOUNT_* vars from .env
    project_root = Path(__file__).parent.parent.parent
    env_file = project_root / ".env"

    if env_file.exists():
        try:
            with open(env_file, "r") as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("METAAPI_ACCOUNT_") and "=" in line:
                        key, value = line.split("=", 1)
                        key = key.strip()
                        value = value.strip().strip('"').strip("'")

                        if key != "METAAPI_ACCOUNT_ID" and value:
                            # Extract label from METAAPI_ACCOUNT_LABEL
                            label = key.replace("METAAPI_ACCOUNT_", "")
                            accounts[label] = value
        except Exception:
            pass

    return accounts


if __name__ == "__main__":
    print("=" * 80)
    print("Secure Token Helper - Status Check")
    print("=" * 80)

    token = get_metaapi_token()
    if token:
        print(f"✅ Token found: {token[:20]}...{token[-10:]}")
    else:
        print("❌ No token found")
        print("   Run: python scripts/setup_metaapi_credentials.py")

    print()
    accounts = list_configured_accounts()
    if accounts:
        print(f"✅ Found {len(accounts)} configured account(s):")
        for label, account_id in accounts.items():
            print(f"   • {label}: {account_id}")
    else:
        print("❌ No accounts configured")
        print("   Run: python scripts/setup_metaapi_credentials.py")
