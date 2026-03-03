#!/usr/bin/env python3
"""
Test MetaAPI Connection
========================

Simple script to test if configured MetaAPI credentials are working.

Features:
- Tests connection to configured account only
- Does NOT require API Access Token (works with Account tokens too)
- Shows account status, deployment state, and connection health
- Validates credentials from .env file

Usage:
    python scripts/download/test_metaapi_connection.py

Environment Variables (from .env):
    METAAPI_TOKEN - Your MetaAPI token (API or Account token)
    METAAPI_ACCOUNT_ID - Your MetaAPI account ID
"""

import asyncio
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


def get_credentials() -> tuple:
    """
    Load credentials from .env file.

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
                    # Remove ENC:: prefix if present
                    if token.startswith("ENC::"):
                        token = token[5:]
                elif line.startswith("METAAPI_ACCOUNT_ID="):
                    account_id = line.split("=", 1)[1].strip().strip('"').strip("'")
                    if account_id.startswith("ENC::"):
                        account_id = account_id[5:]
    except Exception as e:
        print(f"⚠️  Warning: Could not read .env: {e}")

    # NOTE: Environment variables are NOT checked here
    # Only .env file is used to avoid override issues
    # If you need to use environment variables, set them in .env file

    return token, account_id


async def test_account_connection(token: str, account_id: str) -> bool:
    """
    Test connection to specific MetaAPI account.

    Args:
        token: MetaAPI token (API or Account token)
        account_id: MetaAPI account ID

    Returns:
        True if connection successful, False otherwise
    """
    try:
        print("🔄 Initializing MetaAPI...")
        api = MetaApi(token=token)

        print(f"🔄 Connecting to account: {account_id}...")
        account = await api.metatrader_account_api.get_account(account_id)

        # Display account info
        print("\n" + "─" * 80)
        print("✅ ACCOUNT FOUND")
        print("─" * 80)
        print(f"  Account ID:     {account.id}")
        print(f"  Name:           {account.name}")
        print(f"  Type:           {account.type}")
        print(f"  State:          {account.state}")

        if hasattr(account, "login"):
            print(f"  Login:          {account.login}")
        if hasattr(account, "server"):
            print(f"  Server:         {account.server}")
        if hasattr(account, "platform"):
            print(f"  Platform:       {account.platform}")
        if hasattr(account, "region"):
            print(f"  Region:         {account.region}")

        print("─" * 80)

        # Check deployment state
        if account.state == "DEPLOYED":
            print("\n✅ Account is DEPLOYED and ready to use")

            # Try to get a streaming connection (most robust test)
            print("\n🔄 Testing streaming connection...")
            try:
                connection = account.get_streaming_connection()
                await connection.connect()

                # Wait a moment for connection to establish
                await asyncio.sleep(2)

                # Check if connected
                if connection.health_monitor.health_status.get("connected", False):
                    print("✅ Streaming connection established successfully!")
                    print(f"   Health status: {connection.health_monitor.health_status}")
                else:
                    print("⚠️  Connection established but health status unclear")
                    print(f"   Health status: {connection.health_monitor.health_status}")

                # Close connection
                await connection.close()
                print("✅ Connection closed cleanly")

            except Exception as e:
                print(f"⚠️  Streaming connection test failed: {e}")
                print("   (Account exists but streaming may not be available)")

        elif account.state == "DEPLOYING":
            print("\n⚠️  Account is currently DEPLOYING")
            print("   Please wait a few minutes and try again")

        elif account.state == "UNDEPLOYED":
            print("\n⚠️  Account is UNDEPLOYED")
            print("   Deploy the account at: https://app.metaapi.cloud/accounts")

        else:
            print(f"\n⚠️  Account state: {account.state}")
            print("   Check account status at: https://app.metaapi.cloud/accounts")

        return True

    except Exception as e:
        error_msg = str(e)
        print(f"\n❌ Connection test failed: {error_msg}")

        # Provide helpful troubleshooting
        if "not found" in error_msg.lower() or "does not exist" in error_msg.lower():
            print("\n💡 Troubleshooting:")
            print("   - Verify the Account ID is correct")
            print("   - Check that the account exists at: https://app.metaapi.cloud/accounts")
            print("   - Re-run credential setup (Menu 1 → 1)")

        elif "authentication" in error_msg.lower() or "unauthorized" in error_msg.lower():
            print("\n💡 Troubleshooting:")
            print("   - Verify your token is correct")
            print("   - Token may have expired - generate a new one")
            print("   - Re-run credential setup (Menu 1 → 1)")

        elif "account access token" in error_msg.lower() or "api access token" in error_msg.lower():
            print("\n💡 Troubleshooting:")
            print("   - This test works with both API and Account tokens")
            print("   - However, your token and account ID may be mismatched")
            print("   - Re-run credential setup (Menu 1 → 1)")

        else:
            print("\n💡 Troubleshooting:")
            print("   - Check your internet connection")
            print("   - Verify MetaAPI service status")
            print("   - Re-run credential setup (Menu 1 → 1)")

        return False


def main():
    """Main entry point."""
    print_header("MetaAPI Connection Test")

    # Check if metaapi SDK is available
    if not METAAPI_AVAILABLE:
        print("❌ metaapi-cloud-sdk not installed")
        print("\nInstall with:")
        print("   pip install metaapi-cloud-sdk")
        print("\nOr:")
        print("   pip install -e .[dev]")
        sys.exit(1)

    # Load credentials
    print("📋 Loading credentials from .env file...")
    token, account_id = get_credentials()

    if not token:
        print("\n❌ METAAPI_TOKEN not found")
        print("\n💡 Run credential setup first:")
        print("   Menu 1 → Configure MetaAPI Credentials")
        print("\nOr manually add to .env file:")
        print("   METAAPI_TOKEN=your_token_here")
        sys.exit(1)

    if not account_id:
        print("\n❌ METAAPI_ACCOUNT_ID not found")
        print("\n💡 Run credential setup first:")
        print("   Menu 1 → Configure MetaAPI Credentials")
        print("\nOr manually add to .env file:")
        print("   METAAPI_ACCOUNT_ID=your_account_id_here")
        sys.exit(1)

    # Display what we're testing
    print("\n✅ Found credentials:")
    print(f"   Token:      {token[:10]}...{token[-10:] if len(token) > 20 else ''}")
    print(f"   Account ID: {account_id}")

    # Run test
    print("\n" + "═" * 80)
    print("  TESTING CONNECTION")
    print("═" * 80)

    success = asyncio.run(test_account_connection(token, account_id))

    # Final summary
    print("\n" + "═" * 80)
    if success:
        print("  ✅ CONNECTION TEST PASSED")
        print("═" * 80)
        print("\n✅ Your MetaAPI credentials are working correctly!")
        print("\nNext steps:")
        print("  1. Download data (Menu 2 → Download Data)")
        print("  2. Run discovery (Menu 2 → Discover Available Data)")
        print("  3. Start training (Menu 3 → Quick RL Training)")
    else:
        print("  ❌ CONNECTION TEST FAILED")
        print("═" * 80)
        print("\n❌ Could not connect to MetaAPI account")
        print("\nRecommended actions:")
        print("  1. Re-run credential setup (Menu 1 → 1)")
        print("  2. Verify account at: https://app.metaapi.cloud/accounts")
        print("  3. Check MetaAPI service status")

    return 0 if success else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\n⚠️  Test cancelled by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
