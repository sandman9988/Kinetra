#!/usr/bin/env python3
"""
Kinetra Master Workflow
=======================

Complete end-to-end workflow in one process:
1. Authentication & account selection
2. Download data (with options)
2.5. Auto-convert MT5 format to standard format
3. Check and fill missing data
4. Check data integrity
5. Prepare data (train/test split)
6. Run exploration (agent comparison)

Features:
- Atomic credential storage with file tampering detection
- Automatic backups and recovery
- Performance measurement and logging
- Comprehensive failure handling
- Step-by-step progress tracking with resume capability

Usage:
    python scripts/master_workflow.py
"""

import os
import sys
import subprocess
import getpass
import time
from pathlib import Path

from cryptography.fernet import Fernet

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from kinetra.workflow_manager import WorkflowManager


def _get_encryption_key() -> bytes:
    """
    Get the encryption key for securing credentials.

    The key is expected in the KINETRA_SECRET_KEY environment variable.
    It must be a valid Fernet key (base64-encoded 32-byte key).
    """
    key = os.environ.get("KINETRA_SECRET_KEY")
    if not key:
        raise RuntimeError(
            "Encryption key not set. Please define KINETRA_SECRET_KEY with a Fernet key."
        )
    return key.encode("utf-8")


def encrypt_value(value: str) -> str:
    """
    Encrypt a sensitive value using Fernet and mark it as encrypted.
    """
    if not value:
        return value
    f = Fernet(_get_encryption_key())
    token = f.encrypt(value.encode("utf-8"))
    # Prefix to signal that the value is encrypted
    return "ENC::" + token.decode("utf-8")


def print_header(text: str):
    """Print section header."""
    print("\n" + "=" * 80)
    print(f"  {text}")
    print("=" * 80)


def save_credentials_to_env(token: str, account_id: str = None, wf_manager: WorkflowManager = None):
    """
    Save credentials to .env file with atomic safety and integrity checks.
    
    Args:
        token: MetaAPI token
        account_id: MetaAPI account ID (optional)
        wf_manager: WorkflowManager instance for atomic operations
    """
    # Use script's parent directory, not cwd (in case user runs from subdirectory)
    script_dir = Path(__file__).parent.parent
    env_file = script_dir / '.env'

    if wf_manager:
        wf_manager.logger.info(f"📝 Saving credentials to: {env_file.absolute()}")
    else:
        print(f"\n📝 Saving to: {env_file.absolute()}")

    # Read existing .env if it exists (with integrity check)
    env_lines = {}
    if env_file.exists():
        try:
            # Verify file integrity before reading
            if wf_manager and not wf_manager.verify_file_integrity(env_file):
                if wf_manager:
                    wf_manager.logger.warning("⚠️  .env file integrity check failed, will recreate")
                else:
                    print("⚠️  .env file integrity check failed, will recreate")
            else:
                with open(env_file, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith('#') and '=' in line:
                            key, value = line.split('=', 1)
                            env_lines[key] = value
                if wf_manager:
                    wf_manager.logger.info(f"   Loaded {len(env_lines)} existing credentials")
                else:
                    print(f"   Loaded {len(env_lines)} existing credentials")
        except Exception as e:
            if wf_manager:
                wf_manager.logger.error(f"⚠️  Could not read existing .env: {e}")
            else:
                print(f"⚠️  Could not read existing .env: {e}")

    # Update credentials (encrypt sensitive values before storing)
    if token:
        env_lines['METAAPI_TOKEN'] = encrypt_value(token)
    if account_id:
        env_lines['METAAPI_ACCOUNT_ID'] = encrypt_value(account_id)

    # Prepare content
    content = "# Kinetra MetaAPI Credentials\n"
    content += "# Auto-generated - do not commit to git\n\n"
    for key, value in env_lines.items():
        content += f"{key}={value}\n"

    # Use atomic write if WorkflowManager available
    if wf_manager:
        success = wf_manager.atomic_write(env_file, content)
        if success:
            wf_manager.logger.info(f"✅ Credentials saved securely with backup and checksum")
            wf_manager.logger.info(f"   Saved {len(env_lines)} credentials")
        else:
            wf_manager.logger.error(f"❌ Failed to save credentials")
            raise RuntimeError("Failed to save credentials")
    else:
        # Fallback to direct write (less safe)
        try:
            with open(env_file, 'w') as f:
                f.write(content)

            if env_file.exists():
                size = env_file.stat().st_size
                print(f"✅ Credentials saved to {env_file}")
                print(f"   File size: {size} bytes")
                print(f"   Saved {len(env_lines)} credentials")
            else:
                print(f"❌ Failed to create {env_file}")

        except Exception as e:
            print(f"❌ Failed to save credentials: {e}")
            import traceback
            traceback.print_exc()
            raise

    # Add to .gitignore if not already there
    gitignore = script_dir / '.gitignore'
    if gitignore.exists():
        content = gitignore.read_text()
        if '.env' not in content:
            with open(gitignore, 'a') as f:
                f.write("\n# Environment variables\n.env\n")


def check_credentials(wf_manager: WorkflowManager = None, auto_restore: bool = False) -> bool:
    """
    Check if MetaAPI credentials are set and prompt if needed.
    
    Args:
        wf_manager: WorkflowManager instance for logging and atomic operations
        auto_restore: If True, automatically restore from backup without prompting
        
    Returns:
        True if credentials are available
    """
    logger = wf_manager.logger if wf_manager else None
    
    # Check for placeholder values
    placeholder_patterns = ['your-token-here', 'your-account-id-here', 'placeholder', 'example']

    # Try loading from .env file first (use script's parent directory)
    script_dir = Path(__file__).parent.parent
    env_file = script_dir / '.env'

    if logger:
        logger.info(f"🔍 Looking for credentials in: {env_file.absolute()}")
    else:
        print(f"\n🔍 Looking for credentials in: {env_file.absolute()}")

    if env_file.exists():
        # Verify file integrity
        if wf_manager:
            if not wf_manager.verify_file_integrity(env_file):
                logger.warning("⚠️  .env file integrity check failed - may have been tampered with")
                
                if auto_restore:
                    # Automatically restore without prompting
                    if wf_manager.restore_from_backup(env_file):
                        logger.info("✅ Automatically restored .env from backup")
                    else:
                        logger.warning("⚠️  No backup available, will prompt for credentials")
                else:
                    # Prompt user for restore decision
                    while True:
                        restore = input("\nAttempt to restore from backup? [1=Yes, 2=No]: ").strip()
                        if restore in ['1', '2']:
                            break
                        print("⚠️  Invalid input. Please enter 1 or 2.")
                    
                    if restore == '1':
                        if wf_manager.restore_from_backup(env_file):
                            logger.info("✅ Restored .env from backup")
                        else:
                            logger.warning("⚠️  No backup available, will prompt for credentials")
        
        if logger:
            logger.info(f"✅ Found .env file ({env_file.stat().st_size} bytes)")
        else:
            print(f"✅ Found .env file ({env_file.stat().st_size} bytes)")
        
        try:
            with open(env_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        key, value = line.split('=', 1)
                        if key == 'METAAPI_TOKEN' and key not in os.environ:
                            os.environ[key] = value
                            if logger:
                                logger.info(f"   Loaded METAAPI_TOKEN")
                            else:
                                print(f"   Loaded METAAPI_TOKEN")
                        elif key == 'METAAPI_ACCOUNT_ID' and key not in os.environ:
                            os.environ[key] = value
                            if logger:
                                logger.info(f"   Loaded METAAPI_ACCOUNT_ID")
                            else:
                                print(f"   Loaded METAAPI_ACCOUNT_ID")
        except Exception as e:
            if logger:
                logger.error(f"⚠️  Could not read .env file: {e}")
            else:
                print(f"⚠️  Could not read .env file: {e}")
    else:
        if logger:
            logger.info(f"ℹ️  No .env file found (will prompt for credentials)")
        else:
            print(f"ℹ️  No .env file found (will prompt for credentials)")

    token = os.environ.get('METAAPI_TOKEN')
    account_id = os.environ.get('METAAPI_ACCOUNT_ID')

    # Check token
    has_valid_token = False
    if token and not any(placeholder in token.lower() for placeholder in placeholder_patterns):
        msg = f"✅ Found valid API token: {token[:8]}***"
        if logger:
            logger.info(msg)
        else:
            print(f"\n{msg}")
        has_valid_token = True
    else:
        if token:
            msg = "⚠️  Found placeholder METAAPI_TOKEN (ignoring it)"
            if logger:
                logger.warning(msg)
            else:
                print(f"\n{msg}")
        else:
            msg = "ℹ️  No METAAPI_TOKEN set"
            if logger:
                logger.info(msg)
            else:
                print(f"\n{msg}")

        # Prompt for token NOW (hidden input)
        print("\n📋 MetaAPI Token Required")
        print("Get your token from: https://app.metaapi.cloud/")
        token = getpass.getpass("\nEnter your MetaAPI token (hidden): ").strip()

        if not token:
            msg = "❌ No token provided - cannot proceed"
            if logger:
                logger.error(msg)
            else:
                print(f"\n{msg}")
            return False

        # Save to environment for this session
        os.environ['METAAPI_TOKEN'] = token
        has_valid_token = True

        # Ask to save persistently
        save_choice = input("\n💾 Save credentials to .env file? [1=Yes, 2=No]: ").strip()
        if save_choice == '1':
            # Also prompt for account_id to save both together
            print("\n📋 MetaAPI Account ID (optional, can skip if using multiple accounts)")
            print("Get this from: https://app.metaapi.cloud/")
            print("(UUID format: e8f8c21a-32b5-40b0-9bf7-672e8ffab91f)")
            account_id_input = getpass.getpass("\nEnter your MetaAPI account ID (hidden, or press Enter to skip): ").strip()

            if account_id_input:
                os.environ['METAAPI_ACCOUNT_ID'] = account_id_input
                account_id = account_id_input
                save_credentials_to_env(token, account_id, wf_manager)
            else:
                save_credentials_to_env(token, account_id=None, wf_manager=wf_manager)

        msg = "✅ Token set for workflow"
        if logger:
            logger.info(msg)
        else:
            print(msg)

    # Check account ID (after potential prompting above)
    has_valid_account = False
    if not account_id:
        account_id = os.environ.get('METAAPI_ACCOUNT_ID')

    if account_id and not any(placeholder in account_id.lower() for placeholder in placeholder_patterns):
        msg = f"✅ Found valid account ID: {account_id[:8]}***"
        if logger:
            logger.info(msg)
        else:
            print(msg)
        has_valid_account = True
    else:
        if account_id:
            msg = "⚠️  Found placeholder METAAPI_ACCOUNT_ID (ignoring it)"
            if logger:
                logger.warning(msg)
            else:
                print(msg)
        else:
            msg = "ℹ️  No METAAPI_ACCOUNT_ID set"
            if logger:
                logger.info(msg)
            else:
                print(msg)

        msg = "ℹ️  Account will be selected interactively during download"
        if logger:
            logger.info(msg)
        else:
            print(msg)

    return True


def run_step(
    wf_manager: WorkflowManager,
    step_name: str,
    script_path: str,
    required: bool = True,
    allow_exit: bool = True
) -> bool:
    """
    Run a workflow step with comprehensive error handling and performance tracking.
    
    Args:
        wf_manager: WorkflowManager instance
        step_name: Display name of the step
        script_path: Path to script to execute
        required: Whether this is a critical step
        allow_exit: Whether to prompt for exit after step
        
    Returns:
        True if successful and should continue, False otherwise
    """
    def execute_script():
        """Inner function to execute the script."""
        if not Path(script_path).exists():
            raise FileNotFoundError(f"Script not found: {script_path}")

        # Pass through stdin/stdout/stderr so interactive scripts work
        result = subprocess.run(
            [sys.executable, script_path],
            stdin=sys.stdin,
            stdout=sys.stdout,
            stderr=sys.stderr
        )

        if result.returncode != 0:
            raise RuntimeError(f"Script exited with code {result.returncode}")
        
        return True
    
    # Execute step with retry logic
    success, result = wf_manager.execute_step(
        step_name,
        execute_script,
        critical=required,
        max_retries=3 if required else 1
    )
    
    if not success:
        return False
    
    # Exit offramp
    if allow_exit:
        print("\nOptions:")
        print("  1. Continue to next step")
        print("  2. Exit workflow")
        print("  3. Save progress and exit (resume later)")

        while True:
            choice = input("\nSelect [1-3]: ").strip()
            if choice in ['1', '2', '3']:
                break
            print("⚠️  Invalid input. Please enter 1, 2, or 3.")
        
        if choice == '2':
            wf_manager.logger.info("👋 User chose to exit workflow")
            return False
        elif choice == '3':
            wf_manager.logger.info("💾 Saving progress and exiting")
            wf_manager.complete_workflow(status="paused")
            return False
    
    return True


def main():
    """Run complete workflow with comprehensive management."""
    # Initialize workflow manager
    wf_manager = WorkflowManager(
        log_dir="logs/workflow",
        backup_dir="data/backups/workflow",
        enable_backups=True,
        enable_checksums=True,
        max_retries=3
    )
    
    # Start workflow
    wf_manager.start_workflow(
        "master_workflow",
        metadata={
            "version": "2.0",
            "features": [
                "atomic_storage",
                "file_tampering_checks",
                "auto_backups",
                "performance_tracking",
                "failure_recovery"
            ]
        }
    )
    
    try:
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
            wf_manager.logger.info("👋 Workflow cancelled by user")
            wf_manager.complete_workflow(status="cancelled")
            return

        # Check credentials first
        def auth_step():
            """Authentication step."""
            print_header("STEP 1: AUTHENTICATION")
            if not check_credentials(wf_manager):
                raise RuntimeError("Authentication failed")
            return True
        
        success, _ = wf_manager.execute_step(
            "Authentication & Credential Verification",
            auth_step,
            critical=True
        )
        
        if not success:
            wf_manager.complete_workflow(status="failed")
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
            wf_manager.logger.info("✅ Will skip download, use existing data")

        elif workflow_choice == '3':
            run_fill = False
            run_integrity = False
            run_prepare = False
            run_explore = False
            wf_manager.logger.info("✅ Will download only")

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
                wf_manager,
                "STEP 2: DOWNLOAD DATA",
                "scripts/download_interactive.py",
                required=True,
                allow_exit=True
            ):
                wf_manager.complete_workflow(status="interrupted")
                return
        else:
            print_header("STEP 2: DOWNLOAD DATA")
            wf_manager.logger.info("⏭️  Skipped (using existing data)")

        # Step 2.5: Convert MT5 format (automatic, no user input)
        print_header("STEP 2.5: CONVERT MT5 FORMAT")

        convert_script = Path("scripts/convert_mt5_format.py")
        if convert_script.exists():
            def convert_step():
                """MT5 format conversion."""
                wf_manager.logger.info("🔄 Auto-converting MT5 format files to standard format...")
                wf_manager.logger.info("   (Combining <DATE>+<TIME> → time, renaming columns)")

                result = subprocess.run(
                    [sys.executable, str(convert_script)],
                    stdin=sys.stdin,
                    stdout=sys.stdout,
                    stderr=sys.stderr
                )

                if result.returncode != 0:
                    raise RuntimeError(f"Conversion failed with code {result.returncode}")
                return True
            
            wf_manager.execute_step(
                "MT5 Format Conversion",
                convert_step,
                critical=False  # Non-critical step
            )
        else:
            wf_manager.logger.info("⏭️  Converter not found (files may already be in correct format)")

        # Step 3: Fill missing
        if run_fill:
            if not run_step(
                wf_manager,
                "STEP 3: CHECK & FILL MISSING DATA",
                "scripts/check_and_fill_data.py",
                required=False,
                allow_exit=True
            ):
                wf_manager.complete_workflow(status="interrupted")
                return
        else:
            print_header("STEP 3: CHECK & FILL MISSING DATA")
            wf_manager.logger.info("⏭️  Skipped")

        # Step 4: Integrity check
        if run_integrity:
            if not run_step(
                wf_manager,
                "STEP 4: CHECK DATA INTEGRITY",
                "scripts/check_data_integrity.py",
                required=False,
                allow_exit=True
            ):
                wf_manager.complete_workflow(status="interrupted")
                return
        else:
            print_header("STEP 4: CHECK DATA INTEGRITY")
            wf_manager.logger.info("⏭️  Skipped")

        # Step 5: Prepare data
        if run_prepare:
            if not run_step(
                wf_manager,
                "STEP 5: PREPARE DATA",
                "scripts/prepare_data.py",
                required=True,
                allow_exit=True
            ):
                wf_manager.complete_workflow(status="interrupted")
                return
        else:
            print_header("STEP 5: PREPARE DATA")
            wf_manager.logger.info("⏭️  Skipped")

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
                    wf_manager,
                    "UNIVERSAL AGENT BASELINE",
                    "scripts/explore_universal.py",
                    required=False,
                    allow_exit=False  # Don't need exit prompt for last step
                )

            elif explore_choice == '2':
                run_step(
                    wf_manager,
                    "COMPARE AGENTS (LinearQ vs PPO vs SAC vs TD3)",
                    "scripts/explore_compare_agents.py",
                    required=False,
                    allow_exit=False  # Don't need exit prompt for last step
                )

            elif explore_choice == '3':
                run_step(
                    wf_manager,
                    "TESTING MENU",
                    "scripts/test_menu.py",
                    required=False,
                    allow_exit=False  # Don't need exit prompt for last step
                )

            else:
                wf_manager.logger.info("⏭️  Skipped exploration")

        else:
            print_header("STEP 6: EXPLORATION")
            wf_manager.logger.info("⏭️  Skipped")

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

        # Complete workflow
        wf_manager.complete_workflow(status="completed")
        
    except KeyboardInterrupt:
        wf_manager.logger.warning("\n⚠️  Workflow interrupted by user")
        wf_manager.complete_workflow(status="interrupted")
        sys.exit(0)
    except Exception as e:
        wf_manager.logger.error(f"\n❌ Workflow error: {e}")
        import traceback
        traceback.print_exc()
        wf_manager.complete_workflow(status="failed")
        sys.exit(1)


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
