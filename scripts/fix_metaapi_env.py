#!/usr/bin/env python3
"""
Fix MetaAPI Environment Variable Override Issue
================================================

This script diagnoses and fixes the issue where placeholder MetaAPI
environment variables override the real credentials in .env file.

Problem:
    - METAAPI_TOKEN and METAAPI_ACCOUNT_ID are set in shell environment
    - These placeholder values (e.g., "your-token-here") override .env
    - Scripts fail because they use placeholder instead of real credentials

Solution:
    1. Detect if environment variables are set with placeholder values
    2. Show the user how to unset them for current session
    3. Check ~/.bashrc, ~/.zshrc, ~/.profile for persistent exports
    4. Offer to clean up shell config files
    5. Verify .env file has real credentials

Usage:
    python scripts/fix_metaapi_env.py

Features:
    - Safe: Only shows commands, doesn't modify environment directly
    - Informative: Explains what's wrong and why
    - Comprehensive: Checks all common shell config files
    - Validates: Ensures .env has real credentials
"""

import os
import re
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# Placeholder patterns that indicate fake credentials
PLACEHOLDER_PATTERNS = [
    "your-token-here",
    "your-account-id-here",
    "placeholder",
    "example",
    "xxx",
    "yyy",
    "test",
]


def print_header(text: str):
    """Print section header."""
    print(f"\n{'=' * 80}")
    print(f"  {text}")
    print(f"{'=' * 80}\n")


def print_section(text: str):
    """Print subsection."""
    print(f"\n{'─' * 80}")
    print(f"  {text}")
    print(f"{'─' * 80}")


def is_placeholder_value(value: str) -> bool:
    """
    Check if a value looks like a placeholder.

    Args:
        value: String to check

    Returns:
        True if value appears to be a placeholder
    """
    if not value:
        return True

    value_lower = value.lower()

    # Check against known placeholder patterns
    for pattern in PLACEHOLDER_PATTERNS:
        if pattern in value_lower:
            return True

    # Very short values are suspicious
    if len(value) < 10:
        return True

    return False


def check_environment_variables() -> tuple:
    """
    Check if METAAPI environment variables are set.

    Returns:
        Tuple of (token, account_id, has_placeholders)
    """
    token = os.environ.get("METAAPI_TOKEN", "")
    account_id = os.environ.get("METAAPI_ACCOUNT_ID", "")

    has_placeholders = False
    if token and is_placeholder_value(token):
        has_placeholders = True
    if account_id and is_placeholder_value(account_id):
        has_placeholders = True

    return token, account_id, has_placeholders


def check_env_file() -> tuple:
    """
    Check if .env file has METAAPI credentials.

    Returns:
        Tuple of (token, account_id, exists)
    """
    env_file = PROJECT_ROOT / ".env"

    if not env_file.exists():
        return "", "", False

    token = ""
    account_id = ""

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
        return "", "", False

    return token, account_id, True


def scan_shell_config_file(filepath: Path) -> list:
    """
    Scan a shell config file for METAAPI exports.

    Args:
        filepath: Path to config file

    Returns:
        List of (line_number, line_content) tuples
    """
    if not filepath.exists():
        return []

    matches = []
    metaapi_pattern = re.compile(
        r"^\s*export\s+(METAAPI_TOKEN|METAAPI_ACCOUNT_ID)\s*=.*$", re.IGNORECASE
    )

    try:
        with open(filepath, "r") as f:
            for i, line in enumerate(f, 1):
                if metaapi_pattern.match(line):
                    matches.append((i, line.rstrip()))
    except Exception as e:
        print(f"⚠️  Could not read {filepath}: {e}")

    return matches


def get_shell_config_files() -> dict:
    """
    Get all shell config files to check.

    Returns:
        Dict of {name: path} for existing config files
    """
    home = Path.home()
    candidates = {
        ".bashrc": home / ".bashrc",
        ".bash_profile": home / ".bash_profile",
        ".profile": home / ".profile",
        ".zshrc": home / ".zshrc",
        ".zprofile": home / ".zprofile",
    }

    return {name: path for name, path in candidates.items() if path.exists()}


def generate_unset_commands() -> str:
    """Generate shell commands to unset environment variables."""
    return """# For current terminal session:
unset METAAPI_TOKEN
unset METAAPI_ACCOUNT_ID

# Or use the helper script:
source scripts/unset_metaapi_env.sh
"""


def main():
    """Main entry point."""
    print_header("MetaAPI Environment Variable Diagnostic & Fix")

    # Step 1: Check environment variables
    print_section("Step 1: Check Current Environment Variables")

    env_token, env_account_id, has_env_placeholders = check_environment_variables()

    if env_token or env_account_id:
        print("✋ Found METAAPI environment variables in current shell:\n")
        if env_token:
            is_placeholder = is_placeholder_value(env_token)
            status = "⚠️  PLACEHOLDER" if is_placeholder else "✅ Real value"
            token_display = f"{env_token[:15]}..." if len(env_token) > 15 else env_token
            print(f"  METAAPI_TOKEN:      {token_display}")
            print(f"                      {status}\n")

        if env_account_id:
            is_placeholder = is_placeholder_value(env_account_id)
            status = "⚠️  PLACEHOLDER" if is_placeholder else "✅ Real value"
            print(f"  METAAPI_ACCOUNT_ID: {env_account_id}")
            print(f"                      {status}\n")

        if has_env_placeholders:
            print("❌ PROBLEM DETECTED: Placeholder environment variables are set!")
            print("   These will override your .env file credentials.\n")
    else:
        print("✅ No METAAPI environment variables set in current shell")
        print("   This is GOOD - credentials will come from .env file\n")

    # Step 2: Check .env file
    print_section("Step 2: Check .env File Credentials")

    env_file_token, env_file_account_id, env_file_exists = check_env_file()

    if not env_file_exists:
        print("❌ .env file not found")
        print("\n💡 Create .env file and configure credentials:")
        print("   Menu 1 → Configure MetaAPI Credentials")
    elif not env_file_token or not env_file_account_id:
        print("⚠️  .env file exists but missing credentials")
        print("\n💡 Configure credentials:")
        print("   Menu 1 → Configure MetaAPI Credentials")
    else:
        # Check if .env has placeholders
        env_file_has_placeholders = is_placeholder_value(env_file_token) or is_placeholder_value(
            env_file_account_id
        )

        if env_file_has_placeholders:
            print("⚠️  .env file has placeholder credentials\n")
            if env_file_token and is_placeholder_value(env_file_token):
                token_display = (
                    f"{env_file_token[:20]}..." if len(env_file_token) > 20 else env_file_token
                )
                print(f"  METAAPI_TOKEN:      {token_display} (PLACEHOLDER)")
            if env_file_account_id and is_placeholder_value(env_file_account_id):
                print(f"  METAAPI_ACCOUNT_ID: {env_file_account_id} (PLACEHOLDER)")

            print("\n💡 Configure real credentials:")
            print("   Menu 1 → Configure MetaAPI Credentials")
        else:
            print("✅ .env file has real credentials\n")
            token_display = (
                f"{env_file_token[:10]}...{env_file_token[-10:]}"
                if len(env_file_token) > 20
                else env_file_token
            )
            print(f"  METAAPI_TOKEN:      {token_display}")
            print(f"  METAAPI_ACCOUNT_ID: {env_file_account_id}")

    # Step 3: Check shell config files
    print_section("Step 3: Check Shell Config Files")

    shell_configs = get_shell_config_files()
    found_in_configs = {}

    for name, path in shell_configs.items():
        matches = scan_shell_config_file(path)
        if matches:
            found_in_configs[name] = (path, matches)

    if found_in_configs:
        print("⚠️  Found METAAPI exports in shell config files:\n")

        for name, (path, matches) in found_in_configs.items():
            print(f"  📄 {name} ({path}):")
            for line_num, line_content in matches:
                print(f"     Line {line_num}: {line_content}")
            print()

        print("💡 These exports will set environment variables every time you start a shell.")
        print("   Use the cleanup script to remove them:")
        print("   python scripts/cleanup_bashrc_metaapi.py")
    else:
        print("✅ No METAAPI exports found in shell config files")
        print("   Checked:", ", ".join(shell_configs.keys()) if shell_configs else "None")

    # Step 4: Summary and recommendations
    print_header("Summary & Recommendations")

    issues = []
    actions = []

    # Check for environment variable issues
    if env_token or env_account_id:
        if has_env_placeholders:
            issues.append("❌ Placeholder environment variables are overriding .env")
            actions.append("1. Unset environment variables in current terminal:")
            actions.append("   " + generate_unset_commands().replace("\n", "\n   ").strip())

    # Check for shell config issues
    if found_in_configs:
        issues.append(f"⚠️  METAAPI exports found in {len(found_in_configs)} config file(s)")
        actions.append("2. Remove exports from shell config files:")
        actions.append("   python scripts/cleanup_bashrc_metaapi.py")

    # Check for .env issues
    if not env_file_exists or not env_file_token or not env_file_account_id:
        issues.append("❌ .env file missing or incomplete")
        actions.append("3. Configure credentials:")
        actions.append("   Menu 1 → Configure MetaAPI Credentials")
    elif is_placeholder_value(env_file_token) or is_placeholder_value(env_file_account_id):
        issues.append("⚠️  .env file has placeholder credentials")
        actions.append("3. Update credentials in .env:")
        actions.append("   Menu 1 → Configure MetaAPI Credentials")

    # Print summary
    if issues:
        print("🔍 Issues Found:\n")
        for issue in issues:
            print(f"  {issue}")

        print("\n" + "=" * 80)
        print("  📋 ACTION PLAN")
        print("=" * 80 + "\n")

        for action in actions:
            print(action)

        print("\n4. Verify the fix:")
        print("   python scripts/download/test_metaapi_connection.py")

        print("\n5. If still having issues:")
        print("   - Close terminal and open a new one")
        print("   - Run this diagnostic again")
        print("   - Check for other shell init files (.bash_login, etc.)")

    else:
        print("✅ Everything looks good!")
        print("\nYour MetaAPI configuration is correct:")
        print("  ✓ No environment variable overrides")
        print("  ✓ .env file has credentials")
        print("  ✓ No exports in shell config files")

        print("\n💡 Next step: Test the connection")
        print("   python scripts/download/test_metaapi_connection.py")

    # Step 5: Technical explanation
    print_section("Why This Matters")

    print("""
Environment variables take precedence over .env file values. When you have:

  Shell Environment:  METAAPI_TOKEN=your-token-here
  .env file:          METAAPI_TOKEN=real_token_abc123...

Python's os.environ will return "your-token-here" (the placeholder),
causing all API calls to fail.

The fix is to:
  1. Unset the environment variables (temporary fix for current shell)
  2. Remove exports from shell config files (permanent fix)
  3. Restart your terminal (to get a clean environment)

After that, all scripts will correctly read from .env file.
""")

    return 0 if not issues else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\n⚠️  Cancelled by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
