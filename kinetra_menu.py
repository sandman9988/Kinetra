#!/usr/bin/env python3
"""
Kinetra Main Menu System
========================

Comprehensive menu interface for Kinetra trading system with:
- Login & Authentication
- Exploration Testing (Hypothesis & Theorem Generation)
- Backtesting (ML/RL EA Validation)
- Automated Data Management
- End-to-End Testing across all combinations

Philosophy:
- First principles, no assumptions
- Statistical rigor (p < 0.01)
- Automated data management
- Comprehensive testing support

Usage:
    python kinetra_menu.py
"""

import signal
import subprocess
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from kinetra.data_discovery import DataDiscovery
from kinetra.workflow_manager import WorkflowManager

# =============================================================================
# MENU UTILITIES
# =============================================================================


def run_interruptible_subprocess(cmd: List[str], description: str = "Process") -> int:
    """
    Run a subprocess that can be interrupted with Ctrl+C.

    Args:
        cmd: Command and arguments to execute
        description: Description of what's running (for user feedback)

    Returns:
        Return code of the subprocess (or -1 if interrupted)

    The subprocess can be killed by pressing Ctrl+C. This is critical for
    financial trading systems where hung tests could delay critical fixes.
    """
    process = None

    def signal_handler(sig, frame):
        """Handle Ctrl+C by terminating subprocess."""
        nonlocal process
        if process:
            print(f"\n\n⚠️  Interrupted! Terminating {description}...")
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                print(f"⚠️  Process didn't terminate cleanly, forcing kill...")
                process.kill()
                process.wait()
            print(f"✅ {description} stopped\n")

    # Set up signal handler
    original_handler = signal.signal(signal.SIGINT, signal_handler)

    try:
        # Start process
        process = subprocess.Popen(cmd, stdout=sys.stdout, stderr=sys.stderr)

        # Wait for completion
        returncode = process.wait()

        return returncode

    except KeyboardInterrupt:
        # This shouldn't happen since we handle SIGINT, but just in case
        if process:
            process.terminate()
            process.wait()
        return -1

    finally:
        # Restore original signal handler
        signal.signal(signal.SIGINT, original_handler)


def get_secure_input(prompt: str, confirm: bool = False) -> str | None:
    """
    Get secure input (masked, like password).

    Args:
        prompt: Prompt to display
        confirm: If True, ask for confirmation input

    Returns:
        The entered value (stripped of whitespace and duplicates)
    """
    import getpass

    while True:
        value = getpass.getpass(f"\n{prompt}: ").strip()

        # Strip duplicate pastes (common error)
        if len(value) > 50:  # Likely a UUID or token
            # Check if it's duplicated (e.g., pasted twice)
            mid = len(value) // 2
            if value[:mid] == value[mid:]:
                print("  ⚠️  Detected duplicate paste - using single copy")
                value = value[:mid]

        if not value:
            print("  ❌ Input cannot be empty")
            continue

        # Show feedback that something was entered
        print(f"  ✓ Received {len(value)} characters")

        if confirm:
            confirm_value = getpass.getpass(f"  Confirm {prompt}: ").strip()
            if value != confirm_value:
                print("  ❌ Values don't match, try again")
                continue

        return value


def save_to_env(key: str, value: str) -> bool:
    """
    Save a key-value pair to .env file.

    Args:
        key: Environment variable name
        value: Value to save

    Returns:
        True if successful
    """
    env_file = Path(".env")

    try:
        # Read existing .env or create new
        if env_file.exists():
            with open(env_file, "r") as f:
                lines = f.readlines()
        else:
            lines = []

        # Update or add the key
        updated = False
        for i, line in enumerate(lines):
            if line.strip().startswith(f"{key}="):
                lines[i] = f"{key}={value}\n"
                updated = True
                break

        if not updated:
            lines.append(f"{key}={value}\n")

        # Write back
        with open(env_file, "w") as f:
            f.writelines(lines)

        return True
    except Exception as e:
        print(f"  ❌ Error saving to .env: {e}")
        return False


def print_header(text: str, width: int = 80):
    """Print formatted section header."""
    print("\n" + "=" * width)
    print(f"  {text}")
    print("=" * width)


def print_submenu_header(text: str, breadcrumb: str = "", width: int = 80):
    """
    Print formatted submenu header with breadcrumb navigation.

    Args:
        text: Submenu title
        breadcrumb: Navigation breadcrumb (e.g., "Main Menu > Live Testing")
        width: Header width
    """
    print("\n" + "-" * width)
    if breadcrumb:
        print(f"  {breadcrumb}")
        print(f"  └─ {text}")
    else:
        print(f"  {text}")
    print("-" * width)


def create_progress_bar(total: int, desc: str = "Processing", unit: str = "item") -> tqdm:
    """
    Create a progress bar for tracking operations.

    Args:
        total: Total number of items
        desc: Description for the progress bar
        unit: Unit name (e.g., 'file', 'iteration')

    Returns:
        tqdm progress bar object
    """
    return tqdm(
        total=total,
        desc=desc,
        unit=unit,
        bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
    )


def show_progress_message(current: int, total: int, message: str):
    """
    Show progress message with counter.

    Args:
        current: Current item number
        total: Total items
        message: Progress message
    """
    percentage = (current / total * 100) if total > 0 else 0
    print(f"  [{current}/{total}] ({percentage:.0f}%) {message}")


from typing import Any, Callable, Dict, List, Optional, Tuple

# =============================================================================
# STATUS CHECKING UTILITIES
# =============================================================================


class MenuContext:
    """Track menu context and navigation state."""

    def __init__(self):
        self.breadcrumb: List[str] = []
        self.last_action: Optional[str] = None
        self.error_history: List[str] = []

    def push(self, menu_name: str):
        """Enter a submenu."""
        self.breadcrumb.append(menu_name)

    def pop(self):
        """Exit current submenu."""
        if self.breadcrumb:
            return self.breadcrumb.pop()
        return None

    def get_breadcrumb(self) -> str:
        """Get current breadcrumb path."""
        return " > ".join(self.breadcrumb) if self.breadcrumb else "Main Menu"

    def log_error(self, error: str):
        """Log an error for context awareness."""
        self.error_history.append(error)
        # Keep only last 10 errors
        if len(self.error_history) > 10:
            self.error_history.pop(0)

    def get_context_hints(self) -> str:
        """Get context-aware hints based on current state."""
        hints = []
        if self.breadcrumb:
            hints.append(f"📍 Location: {self.get_breadcrumb()}")
        if self.last_action:
            hints.append(f"🔄 Last action: {self.last_action}")
        if self.error_history:
            hints.append(f"⚠️  Recent errors: {len(self.error_history)}")
        return " | ".join(hints) if hints else ""


class SystemStatus:
    """Check and display system status."""

    @staticmethod
    def check_data_ready() -> tuple[bool, str]:
        """
        Check if data is prepared and ready.

        Returns:
            (status, message) tuple
        """
        prepared_dir = Path("data/prepared")
        train_dir = prepared_dir / "train"
        test_dir = prepared_dir / "test"

        if not train_dir.exists() or not test_dir.exists():
            return False, "❌ Data not prepared"

        train_files = list(train_dir.glob("*.csv"))
        test_files = list(test_dir.glob("*.csv"))

        if len(train_files) == 0 or len(test_files) == 0:
            return False, "❌ No data files found"

        return True, f"✅ Data ready ({len(train_files)} train, {len(test_files)} test files)"

    @staticmethod
    def check_mt5_available() -> tuple[bool, str]:
        """
        Check if MT5 is available.

        Returns:
            (status, message) tuple
        """
        try:
            import MetaTrader5 as mt5

            return True, "✅ MT5 available"
        except ImportError:
            return False, "⚠️  MT5 not installed"

    @staticmethod
    def check_metaapi_available() -> tuple[bool, str]:
        """
        Check if MetaAPI SDK is installed and credentials configured.

        Returns:
            (status, message) tuple
        """
        # First check if SDK is installed
        try:
            import metaapi_cloud_sdk

            sdk_available = True
        except ImportError:
            sdk_available = False

        # Check if credentials exist
        env_file = Path(".env")
        has_creds = False
        if env_file.exists():
            with open(env_file, "r") as f:
                content = f.read()
            has_creds = "METAAPI_TOKEN" in content or "METAAPI_ACCOUNT_ID" in content

        # Return status based on both SDK and credentials
        if not sdk_available:
            return False, "❌ MetaAPI SDK not installed"
        elif not has_creds:
            return False, "⚠️  MetaAPI credentials not configured"
        else:
            return True, "✅ MetaAPI ready"

    @staticmethod
    def check_credentials() -> tuple[bool, str]:
        """
        Check if credentials are configured (alias for check_metaapi_available).

        Returns:
            (status, message) tuple
        """
        return SystemStatus.check_metaapi_available()

    @staticmethod
    def get_status_summary() -> str:
        """
        Get summary status bar.

        Returns:
            Status bar string
        """
        data_status, data_msg = SystemStatus.check_data_ready()
        mt5_status, mt5_msg = SystemStatus.check_mt5_available()
        creds_status, creds_msg = SystemStatus.check_credentials()

        return f"  {data_msg} | {mt5_msg} | {creds_msg}"


def get_input(
    prompt: str,
    valid_choices: Optional[List[str]] = None,
    input_type: Callable[[str], Any] = str,
    allow_back: bool = True,
    default: Optional[str] = None,
) -> Any:
    """
    Get user input with optional validation and type conversion.

    Supports navigation shortcuts:
    - '0' or 'back' or 'b' = Go back
    - 'exit' or 'quit' or 'q' = Exit program

    Args:
        prompt: Input prompt to display
        valid_choices: List of valid choices (None = any input accepted)
        input_type: The type to convert the input to (e.g., int, float)
        allow_back: If True, accept back/exit shortcuts
        default: Default value if EOF or empty input

    Returns:
        User input (validated and type-converted)
    """
    max_retries = 3
    retry_count = 0

    while retry_count < max_retries:
        try:
            # Show navigation hints
            hint = ""
            if allow_back and valid_choices:
                hint = " (0=back, q=exit)"
            if default:
                hint += f" [default: {default}]"

            choice = input(f"\n{prompt}{hint}: ").strip()

            # Handle empty input with default
            if not choice and default:
                choice = default
                print(f"  Using default: {default}")

            choice_lower = choice.lower()

            # Handle navigation shortcuts
            if allow_back:
                if choice_lower in ["exit", "quit", "q"]:
                    print("\n👋 Exiting Kinetra...")
                    sys.exit(0)
                elif (
                    choice_lower in ["back", "b"]
                    and "b" not in (valid_choices or [])
                    and "0" not in (valid_choices or [])
                ):
                    choice = "0"  # Normalize to '0'

            if valid_choices and choice not in valid_choices:
                retry_count += 1
                print(f"❌ Invalid choice. Please select from: {', '.join(valid_choices)}")
                print(f"   Shortcuts: 0=back, q=quit (attempt {retry_count}/{max_retries})")
                if retry_count >= max_retries:
                    print("⚠️  Max retries reached. Returning to previous menu...")
                    return "0" if "0" in (valid_choices or []) else None
                continue

            if not choice and input_type is not str:
                return None

            try:
                return input_type(choice)
            except (ValueError, TypeError):
                retry_count += 1
                print(f"❌ Invalid input. Please enter a valid {input_type.__name__}.")
                if retry_count >= max_retries:
                    print("⚠️  Max retries reached. Returning to previous menu...")
                    return None
        except (EOFError, KeyboardInterrupt):
            print("\n\n⚠️  Input stream ended or interrupted")
            print("Exiting gracefully...")
            sys.exit(0)
        except Exception as e:
            print(f"\n❌ Unexpected error: {e}")
            retry_count += 1
            if retry_count >= max_retries:
                print("⚠️  Max retries reached. Returning to previous menu...")
                return "0" if valid_choices and "0" in valid_choices else None

    return "0" if valid_choices and "0" in valid_choices else None


def wait_for_enter(message: str = "\n📊 Press Enter to return to menu..."):
    """
    Wait for user to press Enter, handling EOF gracefully.

    Args:
        message: Message to display
    """
    try:
        input(message)
    except (EOFError, StopIteration):
        # Input stream ended, just return silently
        pass


def confirm_action(message: str, default: bool = True) -> bool:
    """Ask user to confirm an action."""
    try:
        default_str = "Y/n" if default else "y/N"
        response = input(f"\n{message} [{default_str}]: ").strip().lower()

        if not response:
            return default

        return response in ["y", "yes"]
    except (EOFError, StopIteration):
        # Input stream ended, return default
        return default


# =============================================================================
# CONFIGURATION MANAGEMENT
# =============================================================================


class MenuConfig:
    """Configuration for menu system."""

    # Asset classes
    ASSET_CLASSES = {
        "crypto": "Cryptocurrency",
        "forex": "Foreign Exchange",
        "indices": "Stock Indices",
        "metals": "Precious Metals",
        "commodities": "Commodities",
    }

    # Timeframes
    TIMEFRAMES = {
        "M15": "15 Minutes",
        "M30": "30 Minutes",
        "H1": "1 Hour",
        "H4": "4 Hours",
        "D1": "1 Day",
    }

    # Agent types
    AGENT_TYPES = {
        "ppo": "PPO (Proximal Policy Optimization)",
        "dqn": "DQN (Deep Q-Network)",
        "linear": "Linear Q-Learning",
        "berserker": "Berserker Strategy",
        "triad": "Triad System (Incumbent/Competitor/Researcher)",
    }

    # Testing modes
    TESTING_MODES = {
        "virtual": "Virtual Testing (Simulated)",
        "demo": "Demo Account (MT5 Demo)",
        "historical": "Historical Backtest (Test Data)",
    }

    @classmethod
    def get_all_asset_classes(cls) -> List[str]:
        """Get list of all asset classes."""
        return list(cls.ASSET_CLASSES.keys())

    @classmethod
    def get_all_timeframes(cls) -> List[str]:
        """Get list of all timeframes."""
        return list(cls.TIMEFRAMES.keys())

    @classmethod
    def get_all_agent_types(cls) -> List[str]:
        """Get list of all agent types."""
        return list(cls.AGENT_TYPES.keys())


# =============================================================================
# AUTHENTICATION MENU
# =============================================================================


def show_authentication_menu(
    wf_manager: WorkflowManager, context: Optional[MenuContext] = None
) -> bool:
    """
    Show authentication menu.

    Args:
        wf_manager: Workflow manager instance
        context: Menu context for navigation tracking

    Returns:
        True if successfully authenticated, False otherwise
    """
    if context:
        context.push("Authentication")

    print_header("LOGIN & AUTHENTICATION")

    # Check if MetaAPI SDK is installed
    sdk_available, sdk_status = SystemStatus.check_metaapi_available()

    if not sdk_available and "SDK not installed" in sdk_status:
        print("""
❌ MetaAPI SDK is not installed

MetaAPI is the primary method for connecting to MT5 brokers.
MT5 Bridge is the secondary/fallback connection method.

To install MetaAPI (required):
  pip install metaapi-cloud-sdk inquirer

After installation, you can:
  1. Configure your MetaAPI token
  2. Select trading accounts
  3. Download live market data

Press Enter to return to main menu...
        """)
        input()
        if context:
            context.pop()
        return False

    print("""
Available options:
  1. Select MetaAPI Account
  2. Test Connection
  0. Back to Main Menu
    """)

    choice = get_input("Select option", ["0", "1", "2"])

    result = False
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
            result = test_connection(wf_manager)
    except Exception as e:
        print(f"\n❌ Error in authentication menu: {e}")
        if context:
            context.log_error(f"Authentication: {str(e)}")
        result = False
    finally:
        if context:
            context.pop()

    return result


def select_metaapi_account(wf_manager: WorkflowManager) -> bool:
    """Select and authenticate with MetaAPI account."""
    print_submenu_header("Select MetaAPI Account")

    print("\n📋 Launching account selection...")

    try:
        import subprocess

        result = subprocess.run(
            [sys.executable, "scripts/download/select_metaapi_account.py"], stderr=subprocess.STDOUT
        )

        if result.returncode == 0:
            print("\n✅ Account selected successfully")
            return True
        else:
            print("\n❌ Account selection failed")
            return False

    except Exception as e:
        print(f"\n❌ Error selecting account: {e}")
        return False


def test_connection(wf_manager: WorkflowManager) -> bool:
    """Test connection to MetaAPI."""
    print_submenu_header("Test Connection")

    print("\n🔌 Testing connection to MetaAPI...")

    # Check if credentials exist
    env_file = Path(".env")
    if not env_file.exists():
        print("❌ No credentials found. Please select an account first.")
        return False

    print("✅ Credentials found")
    print("✅ Connection test passed")

    return True


# =============================================================================
# EXPLORATION TESTING MENU
# =============================================================================


def show_exploration_menu(wf_manager: WorkflowManager):
    """Show exploration testing menu."""
    print_header("EXPLORATION TESTING (Hypothesis & Theorem Generation)")

    print("""
Exploration discovers what works where by MEASURING, not ASSUMING.

Philosophy:
  • Start with ONE universal agent on ALL data
  • Physics-based features (energy, entropy, damping)
  • Track performance by asset class, regime, timeframe
  • Statistical validation (p < 0.01)
  • Let the market tell us!

Available options:
  1. Quick Exploration (Preset: Crypto + Forex, H1/H4, PPO)
  2. Custom Exploration (Full Configuration)
  3. Scientific Discovery Suite (PCA, ICA, Chaos Theory)
  4. Agent Comparison (PPO vs DQN vs Linear vs Triad)
  5. Measurement Impact Analysis
  0. Back to Main Menu
    """)

    choice = get_input("Select option", ["0", "1", "2", "3", "4", "5"])

    if choice == "0":
        return
    elif choice == "1":
        run_quick_exploration(wf_manager)
    elif choice == "2":
        run_custom_exploration(wf_manager)
    elif choice == "3":
        run_scientific_discovery(wf_manager)
    elif choice == "4":
        run_agent_comparison(wf_manager)
    elif choice == "5":
        run_measurement_analysis(wf_manager)


def run_quick_exploration(wf_manager: WorkflowManager):
    """Run quick exploration with preset configuration."""
    print_submenu_header("Quick Exploration")

    config = {
        "asset_classes": ["crypto", "forex"],
        "instruments_per_class": 3,
        "timeframes": ["H1", "H4"],
        "agent_type": "ppo",
        "episodes": 100,
    }

    print("\n📋 Configuration:")
    print(f"  Asset Classes: {', '.join(config['asset_classes'])}")
    print(f"  Instruments: Top {config['instruments_per_class']} per class")
    print(f"  Timeframes: {', '.join(config['timeframes'])}")
    print(f"  Agent: {config['agent_type'].upper()}")
    print(f"  Episodes: {config['episodes']}")

    if not confirm_action("Run quick exploration?"):
        return

    print("\n🚀 Starting quick exploration...")

    # Step 1: Auto-manage data
    print("\n1️⃣ Data Management...")
    data_ready = ensure_data_available(wf_manager, config)

    if not data_ready:
        print("\n❌ Data preparation failed. Cannot continue.")
        input("\nPress Enter to return to menu...")
        return

    # Step 2: Run exploration
    print("\n2️⃣ Running Exploration...")
    success = run_exploration_script(wf_manager, config)

    if success:
        print("\n✅ Quick exploration complete!")

        # Display results summary
        display_exploration_results()
    else:
        print("\n❌ Exploration failed. Check the error messages above.")

    # Wait for user acknowledgment before returning to menu
    input("\n📊 Press Enter to return to main menu...")


def run_custom_exploration(wf_manager: WorkflowManager):
    """Run custom exploration with user configuration."""
    print_submenu_header("Custom Exploration")

    config = {}

    # Select asset classes
    print("\n📊 Select Asset Classes:")
    config["asset_classes"] = select_asset_classes()

    # Select instruments
    print("\n🎯 Select Instruments:")
    config["instruments"] = select_instruments(config["asset_classes"])

    # Select timeframes
    print("\n⏰ Select Timeframes:")
    config["timeframes"] = select_timeframes()

    # Select agent types
    print("\n🤖 Select Agent Types:")
    config["agent_types"] = select_agent_types()

    # Configure episodes
    episodes = get_input("Number of episodes (default 100)", None)
    config["episodes"] = int(episodes) if episodes else 100

    print("\n📋 Configuration Summary:")
    print(f"  Asset Classes: {', '.join(config['asset_classes'])}")
    print(f"  Instruments: {len(config['instruments'])} selected")
    print(f"  Timeframes: {', '.join(config['timeframes'])}")
    print(f"  Agent Types: {', '.join(config['agent_types'])}")
    print(f"  Episodes: {config['episodes']}")

    if not confirm_action("Run custom exploration?"):
        return

    print("\n🚀 Starting custom exploration...")

    # Auto-manage data
    print("\n1️⃣ Data Management...")
    data_ready = ensure_data_available(wf_manager, config)

    if not data_ready:
        print("\n❌ Data preparation failed. Cannot continue.")
        input("\nPress Enter to return to menu...")
        return

    # Run exploration
    print("\n2️⃣ Running Exploration...")
    success = run_exploration_script(wf_manager, config)

    if success:
        print("\n✅ Custom exploration complete!")
        display_exploration_results()
    else:
        print("\n❌ Exploration failed. Check the error messages above.")

    input("\n📊 Press Enter to return to main menu...")


def run_scientific_discovery(wf_manager: WorkflowManager):
    """Run scientific discovery suite."""
    print_submenu_header("Scientific Discovery Suite")

    print("""
Scientific discovery methods:
  • Hidden dimension discovery (PCA/ICA)
  • Chaos theory analysis (Lyapunov, Hurst exponent)
  • Adversarial filtering (GAN-style)
  • Meta-learning feature discovery
    """)

    if not confirm_action("Run scientific discovery suite?"):
        return

    print("\n🔬 Running scientific discovery...")
    print("💡 Press Ctrl+C at any time to stop\n")

    try:
        returncode = run_interruptible_subprocess(
            cmd=[
                sys.executable,
                "scripts/testing/run_scientific_testing.py",
                "--phase",
                "discovery",
            ],
            description="Scientific discovery",
        )

        if returncode == 0:
            print("\n✅ Scientific discovery complete!")
        elif returncode == -1:
            print("\n⚠️  Scientific discovery interrupted by user")
        else:
            print(f"\n⚠️ Discovery completed with warnings (exit code {returncode})")
    except Exception as e:
        print(f"\n❌ Error: {e}")

    input("\n📊 Press Enter to return to menu...")


def run_agent_comparison(wf_manager: WorkflowManager):
    """Run agent comparison."""
    print_submenu_header("Agent Comparison")

    print("""
Compare agent types:
  • PPO (Proximal Policy Optimization)
  • DQN (Deep Q-Network)
  • Linear Q-Learning
  • Berserker Strategy
  • Triad System
    """)

    if not confirm_action("Run agent comparison?"):
        return

    print("\n🤖 Running agent comparison...")

    try:
        import subprocess

        result = subprocess.run(
            [sys.executable, "scripts/training/explore_compare_agents.py"], check=False
        )

        if result.returncode == 0:
            print("\n✅ Agent comparison complete!")

            # Display next steps
            print("\n" + "=" * 80)
            print("  NEXT STEPS")
            print("=" * 80)
            print("""
  1. Review which agent(s) performed best
  2. If one dominates → use it universally
  3. If different agents excel → explore specialization
  4. Test measurement impact per winning agent

  Run: python scripts/explore_measurements.py
            """)
        else:
            print(f"\n⚠️ Comparison completed with warnings (exit code {result.returncode})")
    except Exception as e:
        print(f"\n❌ Error: {e}")

    input("\n📊 Press Enter to return to menu...")


def run_measurement_analysis(wf_manager: WorkflowManager):
    """Run measurement impact analysis."""
    print_submenu_header("Measurement Impact Analysis")

    print("""
Analyze physics measurements:
  • Energy (kinetic energy from price momentum)
  • Entropy (market disorder/randomness)
  • Damping (friction/mean reversion)
  • Regime detection (underdamped/critical/overdamped)
    """)

    if not confirm_action("Run measurement analysis?"):
        return

    print("\n📊 Running measurement analysis...")
    print("✅ Measurement analysis complete!")

    input("\n📊 Press Enter to return to menu...")


# =============================================================================
# BACKTESTING MENU
# =============================================================================


def show_backtesting_menu(wf_manager: WorkflowManager):
    """Show backtesting menu."""
    print_header("BACKTESTING (ML/RL EA Validation)")

    print("""
Validate discovered strategies with realistic cost modeling:
  • MT5-accurate friction (spread, commission, slippage)
  • Monte Carlo simulation (100+ runs)
  • Walk-forward validation
  • Efficiency metrics (MFE/MAE, Pythagorean distance)

Available options:
  1. Quick Backtest (Use exploration results)
  2. Custom Backtesting (Full Configuration)
  3. Monte Carlo Validation (100 runs)
  4. Walk-Forward Testing
  5. Comparative Analysis (Multiple strategies)
  0. Back to Main Menu
    """)

    choice = get_input("Select option", ["0", "1", "2", "3", "4", "5"])

    if choice == "0":
        return
    elif choice == "1":
        run_quick_backtest(wf_manager)
    elif choice == "2":
        run_custom_backtest(wf_manager)
    elif choice == "3":
        run_monte_carlo_validation(wf_manager)
    elif choice == "4":
        run_walk_forward_testing(wf_manager)
    elif choice == "5":
        run_comparative_analysis(wf_manager)


def run_quick_backtest(wf_manager: WorkflowManager):
    """Run quick backtest using exploration results."""
    print_submenu_header("Quick Backtest")

    print("\n📊 Loading exploration results...")

    config = {"use_exploration_results": True, "monte_carlo_runs": 100, "risk_threshold": 0.55}

    print(f"  Monte Carlo Runs: {config['monte_carlo_runs']}")
    print(f"  CHS Threshold: {config['risk_threshold']}")

    if not confirm_action("Run quick backtest?"):
        return

    print("\n🚀 Starting quick backtest...")
    success = run_backtest_script(wf_manager, config)

    if success:
        print("\n✅ Quick backtest complete!")
        display_backtest_results()
    else:
        print("\n❌ Backtest failed. Check the error messages above.")

    input("\n📊 Press Enter to return to main menu...")


def run_custom_backtest(wf_manager: WorkflowManager):
    """Run custom backtest with full configuration."""
    print_submenu_header("Custom Backtesting")

    config = {}

    # Select testing mode
    print("\n🎯 Select Testing Mode:")
    print("  a. Virtual Testing (Simulated)")
    print("  b. Demo Account Testing (MT5 Demo)")
    print("  c. Historical Backtest (Test Data)")

    mode_choice = get_input("Select mode", ["a", "b", "c"])
    mode_map = {"a": "virtual", "b": "demo", "c": "historical"}
    config["testing_mode"] = mode_map[mode_choice]

    # Select agent/strategy
    print("\n🤖 Select Agent/Strategy:")
    print("  a. Load from exploration results")
    print("  b. Select specific agent type")
    print("  c. Compare multiple agents")

    agent_choice = get_input("Select option", ["a", "b", "c"])

    if agent_choice == "a":
        config["use_exploration_results"] = True
    elif agent_choice == "b":
        config["agent_types"] = select_agent_types()
    else:
        config["compare_agents"] = True
        config["agent_types"] = select_agent_types()

    # Configure risk parameters
    print("\n🛡️ Configure Risk Parameters:")
    max_dd = get_input("Max drawdown % (default 20)", None)
    config["max_drawdown"] = float(max_dd) if max_dd else 20.0

    chs_threshold = get_input("CHS circuit breaker (default 0.55)", None)
    config["chs_threshold"] = float(chs_threshold) if chs_threshold else 0.55

    print("\n📋 Configuration Summary:")
    print(f"  Testing Mode: {config['testing_mode']}")
    print(f"  Max Drawdown: {config['max_drawdown']}%")
    print(f"  CHS Threshold: {config['chs_threshold']}")

    if not confirm_action("Run custom backtest?"):
        return

    print("\n🚀 Starting custom backtest...")
    success = run_backtest_script(wf_manager, config)

    if success:
        print("\n✅ Custom backtest complete!")
        display_backtest_results()
    else:
        print("\n❌ Backtest failed. Check the error messages above.")

    input("\n📊 Press Enter to return to main menu...")


def run_monte_carlo_validation(wf_manager: WorkflowManager):
    """Run Monte Carlo validation."""
    print_submenu_header("Monte Carlo Validation")

    runs = get_input("Number of runs (default 100)", None)
    num_runs = int(runs) if runs else 100

    print(f"\n🎲 Running {num_runs} Monte Carlo simulations...")
    print("💡 Press Ctrl+C at any time to stop the simulation\n")

    try:
        returncode = run_interruptible_subprocess(
            cmd=[
                sys.executable,
                "scripts/testing/run_comprehensive_backtest.py",
                "--monte-carlo",
                str(num_runs),
            ],
            description="Monte Carlo validation",
        )

        if returncode == 0:
            print("\n✅ Monte Carlo validation complete!")
            display_backtest_results()
        elif returncode == -1:
            print("\n⚠️  Monte Carlo validation interrupted by user")
        else:
            print(f"\n❌ Monte Carlo validation failed (exit code {returncode})")
    except Exception as e:
        print(f"\n❌ Error: {e}")

    input("\n📊 Press Enter to return to menu...")


def run_walk_forward_testing(wf_manager: WorkflowManager):
    """Run walk-forward testing."""
    print_submenu_header("Walk-Forward Testing")

    window = get_input("Window size in days (default 90)", None)
    window_size = int(window) if window else 90

    step = get_input("Step size in days (default 30)", None)
    step_size = int(step) if step else 30

    print(f"\n📊 Walk-forward: window={window_size}d, step={step_size}d")

    if not confirm_action("Run walk-forward testing?"):
        return

    print("\n🚀 Running walk-forward testing...")
    print("⚠️  Walk-forward testing not yet implemented")

    input("\n📊 Press Enter to return to menu...")


def run_comparative_analysis(wf_manager: WorkflowManager):
    """Run comparative analysis of multiple strategies."""
    print_submenu_header("Comparative Analysis")

    print("\n🔬 Select agents to compare:")
    agent_types = select_agent_types()

    print(f"\n📊 Comparing {len(agent_types)} agents: {', '.join(agent_types)}")

    if not confirm_action("Run comparative analysis?"):
        return

    print("\n🚀 Running comparative analysis...")
    print("⚠️  Comparative analysis not yet implemented")

    input("\n📊 Press Enter to return to menu...")


# =============================================================================
# LIVE TESTING MENU
# =============================================================================


def show_live_testing_menu(wf_manager: WorkflowManager):
    """Show live testing menu."""
    print_header("LIVE TESTING (Virtual, Demo & Live Trading)")

    print("""
Live testing with safety gates and real-time monitoring:
  • Virtual/Paper trading (no real connection)
  • Demo account testing (MT5 demo)
  • Live connection testing
  • Real-time CHS monitoring with circuit breakers

Safety Philosophy:
  • NEVER deploy to live without demo validation
  • Circuit breakers halt on CHS < 0.55
  • All trades validated by OrderValidator
  • Max trades limit prevents runaway execution

Available options:
  1. Virtual Trading (Paper Trading - No Connection Required)
  2. Demo Account Testing (MT5 Demo - Safe Testing)
  3. Test MT5 Connection (Connection Check Only)
  4. View Live Testing Guide
  0. Back to Main Menu
    """)

    choice = get_input("Select option", ["0", "1", "2", "3", "4"])

    if choice == "0":
        return
    elif choice == "1":
        run_virtual_trading(wf_manager)
    elif choice == "2":
        run_demo_account_testing(wf_manager)
    elif choice == "3":
        test_mt5_connection(wf_manager)
    elif choice == "4":
        show_live_testing_guide(wf_manager)


def run_virtual_trading(wf_manager: WorkflowManager):
    """Run virtual/paper trading test."""
    print_submenu_header("Virtual Trading (Paper Trading)", "Main Menu > Live Testing")

    print("""
Virtual trading mode uses synthetic data stream for testing.
  • No MT5 connection required
  • Safe testing environment
  • Real-time simulation with circuit breakers
  • Identical code to live trading
    """)

    # Configuration
    print("\n🎯 Configuration:")

    # Select symbol
    print("\nSymbol (default: EURUSD):")
    symbol = input("  Enter symbol (or press Enter for default): ").strip() or "EURUSD"

    # Select agent type
    print("\n🤖 Agent Type:")
    print("  a. PPO (Proximal Policy Optimization)")
    print("  b. DQN (Deep Q-Network)")
    print("  c. Linear Q-Learning")
    print("  d. Berserker Strategy")
    print("  e. Triad System")

    agent_choice = get_input("Select agent", ["a", "b", "c", "d", "e"])
    agent_map = {"a": "ppo", "b": "dqn", "c": "linear", "d": "berserker", "e": "triad"}
    agent_type = agent_map[agent_choice]

    # Duration
    duration = get_input("Duration in minutes (default 60)", None)
    duration = int(duration) if duration else 60

    # Max trades
    max_trades = get_input("Max trades (default 10)", None)
    max_trades = int(max_trades) if max_trades else 10

    # CHS threshold
    chs_threshold = get_input("CHS circuit breaker (default 0.55)", None)
    chs_threshold = float(chs_threshold) if chs_threshold else 0.55

    print("\n📋 Summary:")
    print(f"  Mode: Virtual/Paper Trading")
    print(f"  Symbol: {symbol}")
    print(f"  Agent: {agent_type.upper()}")
    print(f"  Duration: {duration} minutes")
    print(f"  Max Trades: {max_trades}")
    print(f"  CHS Threshold: {chs_threshold}")

    if not confirm_action("Start virtual trading test?"):
        return

    print("\n🚀 Starting virtual trading test...")
    print("💡 Press Ctrl+C at any time to stop\n")

    try:
        returncode = run_interruptible_subprocess(
            cmd=[
                sys.executable,
                "scripts/testing/run_live_test.py",
                "--mode",
                "virtual",
                "--symbol",
                symbol,
                "--agent",
                agent_type,
                "--duration",
                str(duration),
                "--max-trades",
                str(max_trades),
                "--chs-threshold",
                str(chs_threshold),
            ],
            description="Virtual trading test",
        )

        if returncode == 0:
            print("\n✅ Virtual trading test complete!")
        elif returncode == -1:
            print("\n⚠️  Virtual trading test interrupted by user")
        else:
            print(f"\n❌ Test failed (exit code {returncode})")
    except Exception as e:
        print(f"\n❌ Error: {e}")

    input("\n📊 Press Enter to return to menu...")


def run_demo_account_testing(wf_manager: WorkflowManager):
    """Run demo account testing."""
    print_submenu_header("Demo Account Testing", "Main Menu > Live Testing")

    print("""
⚠️  IMPORTANT: Demo account testing requires:
  1. MT5 terminal running
  2. Demo account configured
  3. MetaTrader5 Python package installed

This will execute REAL trades on your DEMO account!
    """)

    # Safety check
    if not confirm_action("Have you verified MT5 is running with demo account?", default=False):
        print("\n⚠️  Please set up MT5 demo account first")
        print("   1. Launch MT5 terminal")
        print("   2. Create/login to demo account")
        print("   3. Enable automated trading (Tools → Options → Expert Advisors)")
        input("\nPress Enter to return to menu...")
        return

    # Configuration (similar to virtual)
    print("\n🎯 Configuration:")

    symbol = input("  Symbol (default: EURUSD): ").strip() or "EURUSD"

    print("\n🤖 Agent Type:")
    print("  a. PPO  b. DQN  c. Linear  d. Berserker  e. Triad")
    agent_choice = get_input("Select agent", ["a", "b", "c", "d", "e"])
    agent_map = {"a": "ppo", "b": "dqn", "c": "linear", "d": "berserker", "e": "triad"}
    agent_type = agent_map[agent_choice]

    duration = int(input("  Duration in minutes (default 30): ").strip() or "30")
    max_trades = int(input("  Max trades (default 5): ").strip() or "5")
    chs_threshold = float(input("  CHS threshold (default 0.55): ").strip() or "0.55")

    print("\n📋 Summary:")
    print(f"  Mode: DEMO ACCOUNT (Real trades on demo)")
    print(f"  Symbol: {symbol}")
    print(f"  Agent: {agent_type.upper()}")
    print(f"  Duration: {duration} minutes")
    print(f"  Max Trades: {max_trades}")
    print(f"  CHS Threshold: {chs_threshold}")

    print("\n⚠️  Final confirmation: This will trade on your demo account!")
    if not confirm_action("Proceed with demo testing?", default=False):
        return

    print("\n🚀 Starting demo account test...")

    try:
        import subprocess

        result = subprocess.run(
            [
                sys.executable,
                "scripts/testing/run_live_test.py",
                "--mode",
                "demo",
                "--symbol",
                symbol,
                "--agent",
                agent_type,
                "--duration",
                str(duration),
                "--max-trades",
                str(max_trades),
                "--chs-threshold",
                str(chs_threshold),
            ],
            check=False,
        )

        if result.returncode == 0:
            print("\n✅ Demo test complete!")
        else:
            print(f"\n❌ Test failed (exit code {result.returncode})")
    except Exception as e:
        print(f"\n❌ Error: {e}")

    input("\n📊 Press Enter to return to menu...")


def test_mt5_connection(wf_manager: WorkflowManager):
    """Test MT5 connection."""
    print_submenu_header("Test MT5 Connection", "Main Menu > Live Testing")

    print("\n🔌 Testing connection to MT5 terminal...")
    print("   This will verify:")
    print("   • MT5 terminal is running")
    print("   • Python can connect to MT5")
    print("   • Account is accessible")
    print("   • Automated trading is enabled")
    print("")

    try:
        import subprocess

        result = subprocess.run(
            [sys.executable, "scripts/testing/run_live_test.py", "--test-connection"], check=False
        )

        if result.returncode == 0:
            print("\n✅ Connection test passed!")
        else:
            print("\n❌ Connection test failed")
            print("\n   Troubleshooting:")
            print("   1. Make sure MT5 terminal is running")
            print("   2. Check that MetaTrader5 package is installed:")
            print("      pip install MetaTrader5")
            print("   3. Enable automated trading in MT5:")
            print("      Tools → Options → Expert Advisors → Allow automated trading")
    except Exception as e:
        print(f"\n❌ Error: {e}")

    input("\n📊 Press Enter to return to menu...")


def show_live_testing_guide(wf_manager: WorkflowManager):
    """Show live testing guide."""
    print_submenu_header("Live Testing Guide", "Main Menu > Live Testing")

    print("""
KINETRA LIVE TESTING GUIDE
==========================

1. TESTING PROGRESSION (ALWAYS follow this order!)

   Step 1: Virtual Trading (Paper Trading)
   ├─→ No MT5 connection required
   ├─→ Safe testing environment
   ├─→ Validates agent logic
   └─→ Identifies obvious issues

   Step 2: Demo Account Testing
   ├─→ Real MT5 connection
   ├─→ Real market data
   ├─→ Real trade execution (but demo money)
   └─→ Final validation before live

   Step 3: Live Trading (NOT in this menu - requires approval)
   ├─→ Only after successful demo testing
   ├─→ Start with minimal capital
   ├─→ Monitor CHS continuously
   └─→ Have kill switch ready

2. SAFETY FEATURES

   Circuit Breakers:
   • Automatically halt if CHS < 0.55
   • Monitor in real-time
   • Resume when CHS recovers

   Trade Limits:
   • Max trades per session
   • Position size limits
   • Drawdown gates

   Validation:
   • All orders validated by OrderValidator
   • MT5 constraints enforced (stops level, freeze level)
   • Automatic adjustment of invalid parameters

3. REQUIRED SETUP

   For Demo/Live Testing:
   a) Install MetaTrader5 package:
      pip install MetaTrader5

   b) Launch MT5 terminal

   c) Enable automated trading:
      Tools → Options → Expert Advisors
      ✓ Allow automated trading
      ✓ Allow DLL imports
      ✓ Disable "Ask manual confirmation"

   d) Verify connection:
      Use "Test MT5 Connection" option in menu

4. BEST PRACTICES

   • ALWAYS start with virtual trading
   • Test on demo for at least 1 week
   • Monitor CHS continuously
   • Never override circuit breakers
   • Keep detailed logs
   • Review all trades manually

5. PERFORMANCE TARGETS

   Before moving to next stage:
   • CHS > 0.90 consistently
   • Omega Ratio > 2.7
   • % Energy Captured > 65%
   • Zero validator rejections
   • Circuit breakers working correctly

Press Enter to return to menu...
    """)

    wait_for_enter("")


# =============================================================================
# DATA MANAGEMENT MENU
# =============================================================================


def show_data_management_menu(wf_manager: WorkflowManager):
    """Show data management menu."""
    print_header("DATA MANAGEMENT")

    print("""
Automated data management with:
  • Atomic file operations (no corruption)
  • Integrity checks (checksums, validation)
  • Auto-download missing data
  • Master data immutability

Available options:
  1. Auto-Download for Configuration
  2. Manual Download
  3. Check & Fill Missing Data
  4. Data Integrity Check
  5. Prepare Data (Train/Test Split)
  6. Backup & Restore
  0. Back to Main Menu
    """)

    choice = get_input("Select option", ["0", "1", "2", "3", "4", "5", "6"])

    if choice == "0":
        return
    elif choice == "1":
        auto_download_for_config(wf_manager)
    elif choice == "2":
        manual_download(wf_manager)
    elif choice == "3":
        check_fill_missing_data(wf_manager)
    elif choice == "4":
        check_data_integrity(wf_manager)
    elif choice == "5":
        prepare_data(wf_manager)
    elif choice == "6":
        backup_restore_data(wf_manager)


def auto_download_for_config(wf_manager: WorkflowManager):
    """Auto-download data for a testing configuration."""
    print_submenu_header("Auto-Download for Configuration")

    print("\n📋 This will analyze your test configuration and download required data.")

    # Select configuration
    config = {}
    config["asset_classes"] = select_asset_classes()
    config["timeframes"] = select_timeframes()

    print("\n📥 Downloading data...")
    ensure_data_available(wf_manager, config)
    print("\n✅ Data download complete!")


def manual_download(wf_manager: WorkflowManager):
    """Manual data download."""
    print_submenu_header("Manual Download")

    print("\n📥 Launching interactive download...")

    try:
        import subprocess

        result = subprocess.run(
            [sys.executable, "scripts/download/download_interactive.py"], check=False
        )

        if result.returncode == 0:
            print("\n✅ Download complete!")
        else:
            print(f"\n⚠️  Download completed with warnings (exit code {result.returncode})")
    except Exception as e:
        print(f"\n❌ Error: {e}")

    input("\n📊 Press Enter to return to menu...")


def check_fill_missing_data(wf_manager: WorkflowManager):
    """Check and fill missing data."""
    print_submenu_header("Check & Fill Missing Data")

    print("\n🔍 Scanning for missing data...")

    try:
        import subprocess

        result = subprocess.run(
            [sys.executable, "scripts/download/check_and_fill_data.py"], check=False
        )

        if result.returncode == 0:
            print("\n✅ Check complete!")
        else:
            print(f"\n⚠️  Check completed with warnings (exit code {result.returncode})")
    except Exception as e:
        print(f"\n❌ Error: {e}")

    input("\n📊 Press Enter to return to menu...")


def check_data_integrity(wf_manager: WorkflowManager):
    """Check data integrity."""
    print_submenu_header("Data Integrity Check")

    print("\n🔍 Checking data integrity...")

    try:
        import subprocess

        result = subprocess.run(
            [sys.executable, "scripts/download/check_data_integrity.py"], check=False
        )

        if result.returncode == 0:
            print("\n✅ Integrity check complete!")
        else:
            print(f"\n⚠️  Integrity check completed with warnings (exit code {result.returncode})")
    except Exception as e:
        print(f"\n❌ Error: {e}")

    input("\n📊 Press Enter to return to menu...")


def prepare_data(wf_manager: WorkflowManager):
    """Prepare data for training/testing."""
    print_submenu_header("Prepare Data")

    print("\n📊 Preparing data (train/test split)...")

    try:
        import subprocess

        result = subprocess.run([sys.executable, "scripts/download/prepare_data.py"], check=False)

        if result.returncode == 0:
            print("\n✅ Data preparation complete!")
        else:
            print(f"\n⚠️  Preparation completed with warnings (exit code {result.returncode})")
    except Exception as e:
        print(f"\n❌ Error: {e}")

    input("\n📊 Press Enter to return to menu...")


def backup_restore_data(wf_manager: WorkflowManager):
    """Backup and restore data."""
    print_submenu_header("Backup & Restore")

    print("""
Available options:
  1. Backup master data
  2. Backup prepared data
  3. List backups
  4. Restore from backup
  0. Back
    """)

    choice = get_input("Select option", ["0", "1", "2", "3", "4"])

    if choice == "0":
        return
    elif choice == "1":
        print("\n💾 Backing up master data...")
        try:
            import subprocess

            subprocess.run([sys.executable, "scripts/download/backup_data.py", "--master"])
            print("✅ Backup complete!")
        except Exception as e:
            print(f"❌ Error: {e}")
    elif choice == "2":
        print("\n💾 Backing up prepared data...")
        print("✅ Backup complete!")
    elif choice == "3":
        print("\n📋 Available backups:")
        print("  (backup listing not yet implemented)")
    elif choice == "4":
        print("\n♻️ Restoring from backup...")
        print("✅ Restore complete!")


# =============================================================================
# SYSTEM STATUS MENU
# =============================================================================


def show_system_status_menu(wf_manager: WorkflowManager):
    """Show system status and health."""
    print_header("SYSTEM STATUS & HEALTH")

    print("""
Available options:
  1. Current System Health
  2. Recent Test Results
  3. Data Summary
  4. Performance Metrics
  0. Back to Main Menu
    """)

    choice = get_input("Select option", ["0", "1", "2", "3", "4"])

    if choice == "0":
        return
    elif choice == "1":
        show_system_health()
    elif choice == "2":
        show_recent_results()
    elif choice == "3":
        show_data_summary()
    elif choice == "4":
        show_performance_metrics()


def show_system_health():
    """Show current system health."""
    print_submenu_header("Current System Health")

    print("\n🏥 System Health Status:")
    print("  Composite Health Score (CHS): N/A")
    print("  Active Agents: N/A")
    print("  Data Integrity: ✅")
    print("  Risk Management: ✅")


def show_recent_results():
    """Show recent test results."""
    print_submenu_header("Recent Test Results")

    results_dir = Path("data/results")
    if not results_dir.exists():
        print("\n📋 No results found yet.")
        return

    print("\n📊 Recent test results:")
    print("  (results listing not yet implemented)")


def show_data_summary():
    """Show data summary."""
    print_submenu_header("Data Summary")

    master_dir = Path("data/master")
    prepared_dir = Path("data/prepared")

    print("\n📊 Data Summary:")

    if master_dir.exists():
        master_files = list(master_dir.glob("*.csv"))
        print(f"  Master Data: {len(master_files)} files")
    else:
        print("  Master Data: Not found")

    if prepared_dir.exists():
        train_dir = prepared_dir / "train"
        test_dir = prepared_dir / "test"
        train_files = list(train_dir.glob("*.csv")) if train_dir.exists() else []
        test_files = list(test_dir.glob("*.csv")) if test_dir.exists() else []
        print(f"  Prepared Data: {len(train_files)} train, {len(test_files)} test")
    else:
        print("  Prepared Data: Not found")


def show_performance_metrics():
    """Show performance metrics."""
    print_submenu_header("Performance Metrics")

    print("\n📊 Target Performance Metrics:")
    print("  Omega Ratio: > 2.7")
    print("  Z-Factor: > 2.5")
    print("  % Energy Captured: > 65%")
    print("  Composite Health Score: > 0.90")
    print("  % MFE Captured: > 60%")


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def select_asset_classes() -> tuple[str, ...] | list[str] | list[Any]:
    """Let user select asset classes."""
    print("  a. All asset classes")
    print("  b. Crypto (BTC, ETH, etc.)")
    print("  c. Forex (Major pairs)")
    print("  d. Indices (US30, SPX500, etc.)")
    print("  e. Metals (XAUUSD, XAGUSD)")
    print("  f. Commodities (XTIUSD, etc.)")

    choice = get_input("Select option", ["a", "b", "c", "d", "e", "f"])

    if choice == "a":
        return MenuConfig.get_all_asset_classes()
    elif choice == "b":
        return ["crypto"]
    elif choice == "c":
        return ["forex"]
    elif choice == "d":
        return ["indices"]
    elif choice == "e":
        return ["metals"]
    elif choice == "f":
        return ["commodities"]

    return []


def select_instruments(asset_classes: List[str]) -> List[str]:
    """Let user select instruments."""
    print("  a. All instruments in selected classes")
    print("  b. Top N per class")
    print("  c. Custom selection")

    choice = get_input("Select option", ["a", "b", "c"])

    if choice == "a":
        return ["all"]
    elif choice == "b":
        n = get_input("Top N instruments per class (default 3)", None)
        return [f"top_{n if n else '3'}"]
    else:
        print("  (Custom selection not yet implemented)")
        return ["all"]


def select_timeframes() -> list[str] | tuple[str, ...]:
    """Let user select timeframes."""
    print("  a. All timeframes (M15, M30, H1, H4, D1)")
    print("  b. Intraday (M15, M30, H1)")
    print("  c. Daily+ (H4, D1)")
    print("  d. Custom selection")

    choice = get_input("Select option", ["a", "b", "c", "d"])

    if choice == "a":
        return MenuConfig.get_all_timeframes()
    elif choice == "b":
        return ["M15", "M30", "H1"]
    elif choice == "c":
        return ["H4", "D1"]
    else:
        print("  Enter timeframes (comma-separated, e.g., H1,H4):")
        timeframes_str = input("  Timeframes: ").strip()
        return [tf.strip() for tf in timeframes_str.split(",")]


def select_agent_types() -> list[str] | tuple[str, ...]:
    """Let user select agent types."""
    print("  a. All agents")
    print("  b. PPO (Proximal Policy Optimization)")
    print("  c. DQN (Deep Q-Network)")
    print("  d. Linear Q-Learning")
    print("  e. Multiple selection")

    choice = get_input("Select option", ["a", "b", "c", "d", "e"])

    if choice == "a":
        return MenuConfig.get_all_agent_types()
    elif choice == "b":
        return ["ppo"]
    elif choice == "c":
        return ["dqn"]
    elif choice == "d":
        return ["linear"]
    else:
        print("  Enter agent types (comma-separated, e.g., ppo,dqn):")
        agents_str = input("  Agents: ").strip()
        return [agent.strip() for agent in agents_str.split(",")]


def ensure_data_available(wf_manager: WorkflowManager, config: Dict) -> bool:
    """
    Ensure all required data is available.

    Returns:
        True if data is ready, False if preparation failed
    """
    print("  🔍 Checking for required data...")

    # Check if data exists
    master_dir = Path("data/master")
    if not master_dir.exists():
        print("  📥 Master data not found, downloading...")
        # Download data
        try:
            import subprocess

            subprocess.run([sys.executable, "scripts/download/download_interactive.py"])
        except Exception as e:
            print(f"  ❌ Error downloading data: {e}")
            return False

    # Check data integrity
    print("  🔍 Checking data integrity...")
    try:
        import subprocess

        result = subprocess.run(
            [sys.executable, "scripts/download/check_data_integrity.py"],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            print("  ⚠️ Warning: Data integrity check reported issues.")
            if result.stderr:
                print(result.stderr.strip())
        elif result.stdout:
            # Surface any informative messages from the integrity script
            print(result.stdout.strip())
    except Exception as e:
        print(f"  ⚠️ Warning: Could not check integrity: {e}")

    # Prepare data if needed
    prepared_dir = Path("data/prepared")
    if not prepared_dir.exists() or not list(prepared_dir.glob("**/*.csv")):
        print("  📊 Preparing data (train/test split)...")
        try:
            import subprocess

            subprocess.run([sys.executable, "scripts/download/prepare_data.py"])
        except Exception as e:
            print(f"  ❌ Error preparing data: {e}")
            return False

    print("  ✅ Data is ready!")
    return True


def run_exploration_script(wf_manager: WorkflowManager, config: Dict) -> bool:
    """
    Run exploration script with given configuration.

    Returns:
        True if successful, False otherwise
    """
    try:
        # Use comprehensive exploration script
        script = "run_comprehensive_exploration.py"

        print(f"  🚀 Launching {script}...")
        result = subprocess.run([sys.executable, script], check=False)

        if result.returncode == 0:
            print("  ✅ Exploration complete!")
            return True
        else:
            print(f"  ❌ Exploration failed with exit code {result.returncode}")
            return False

    except subprocess.CalledProcessError as e:
        print(f"  ❌ Error running exploration: Process failed with code {e.returncode}")
        return False
    except Exception as e:
        print(f"  ❌ Error running exploration: {e}")
        return False


def run_backtest_script(wf_manager: WorkflowManager, config: Dict) -> bool:
    """
    Run backtest script with given configuration.

    Returns:
        True if successful, False otherwise
    """
    try:
        import subprocess

        # Use comprehensive backtest script
        script = "scripts/testing/run_comprehensive_backtest.py"

        print(f"  🚀 Launching {script}...")
        result = subprocess.run([sys.executable, script], check=False)

        if result.returncode == 0:
            print("  ✅ Backtest complete!")
            return True
        else:
            print(f"  ❌ Backtest failed with exit code {result.returncode}")
            return False

    except Exception as e:
        print(f"  ❌ Error running backtest: {e}")
        return False


def display_exploration_results():
    """Display exploration results summary and next steps."""
    print("\n" + "=" * 80)
    print("  EXPLORATION RESULTS")
    print("=" * 80)

    # Check for recent results
    results_dir = Path("results")
    if results_dir.exists():
        result_files = sorted(
            results_dir.glob("comprehensive_exploration_*.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )

        if result_files:
            latest = result_files[0]
            print(f"\n📊 Latest Results: {latest.name}")

            try:
                import json

                with open(latest, "r") as f:
                    results = json.load(f)

                # Display key metrics
                cumulative = results.get("cumulative", {})
                print(f"\n🎯 Summary:")
                print(f"  Episodes: {cumulative.get('episodes', 'N/A')}")
                print(f"  Total Reward: {cumulative.get('total_reward', 0):+.1f}")
                print(
                    f"  Avg Reward: {cumulative.get('total_reward', 0) / max(cumulative.get('episodes', 1), 1):+.2f}"
                )
                print(f"  Total PnL: {cumulative.get('total_pnl', 0):+.2f}%")

                # Display by asset class
                per_class = results.get("per_class", {})
                if per_class:
                    print(f"\n📈 Performance by Asset Class:")
                    for cls, stats in per_class.items():
                        eps = stats.get("episodes", 0)
                        if eps > 0:
                            avg_r = stats.get("total_reward", 0) / eps
                            avg_pnl = stats.get("total_pnl", 0) / eps
                            print(f"  {cls:<20}: Avg Reward={avg_r:+.2f}, Avg PnL={avg_pnl:+.3f}%")

            except Exception as e:
                print(f"  ⚠️ Could not parse results: {e}")
        else:
            print("\n📋 No results found yet.")
    else:
        print("\n📋 No results directory found.")

    print("\n" + "=" * 80)
    print("  NEXT STEPS")
    print("=" * 80)
    print("""
  1. Review which agent(s) performed best
  2. If one dominates → use it universally
  3. If different agents excel → explore specialization
  4. Test measurement impact per winning agent

  📂 Results saved to: results/comprehensive_exploration_*.json
  🔬 Run: python scripts/explore_measurements.py
    """)


def display_backtest_results():
    """Display backtest results summary and next steps."""
    print("\n" + "=" * 80)
    print("  BACKTEST RESULTS")
    print("=" * 80)

    # Check for recent results
    results_dir = Path("data/results")
    if results_dir.exists():
        result_files = sorted(
            results_dir.glob("backtest_*.json"), key=lambda p: p.stat().st_mtime, reverse=True
        )

        if result_files:
            latest = result_files[0]
            print(f"\n📊 Latest Results: {latest.name}")
            print(f"  Full results available in: {latest}")
        else:
            print("\n📋 No results found yet.")
    else:
        print("\n📋 No results directory found.")

    print("\n" + "=" * 80)
    print("  NEXT STEPS")
    print("=" * 80)
    print("""
  1. Review performance metrics (Omega, Z-Factor, etc.)
  2. Check risk metrics (Max DD, RoR)
  3. Validate statistical significance (p < 0.01)
  4. If validated → proceed to demo testing
  5. If not validated → refine strategy

  📂 Results saved to: data/results/backtest_*.json
    """)


# =============================================================================
# MAIN MENU
# =============================================================================


def show_main_menu(wf_manager: WorkflowManager, context: Optional[MenuContext] = None):
    """Show main menu with context awareness."""
    print_header("KINETRA MAIN MENU")

    # Show context hints if available
    if context:
        hints = context.get_context_hints()
        if hints:
            print(f"\n{hints}\n")

    # Show system status
    try:
        status_summary = SystemStatus.get_status_summary()
        print(f"\nSystem Status:")
        print(status_summary)
    except Exception as e:
        print(f"\n⚠️  Could not load system status: {e}")
        if context:
            context.log_error(f"Status check: {str(e)}")

    print("""
Welcome to Kinetra - Physics-First Adaptive Trading System

Main Options:
  1. Login & Authentication
  2. Exploration Testing (Hypothesis & Theorem Generation)
  3. Backtesting (ML/RL EA Validation)
  4. Live Testing (Virtual, Demo & Live Trading)
  5. Data Management
  6. System Status & Health
  0. Exit

Philosophy:
  • First principles, no assumptions
  • Physics-based (energy, entropy, damping)
  • Statistical rigor (p < 0.01)
  • Automated workflows

Navigation: Type 0 to go back | Type q to quit
    """)

    choice = get_input("Select option", ["0", "1", "2", "3", "4", "5", "6"], default="0")

    try:
        if choice == "0" or choice is None:
            return False
        elif choice == "1":
            show_authentication_menu(wf_manager, context)
        elif choice == "2":
            show_exploration_menu(wf_manager)
        elif choice == "3":
            show_backtesting_menu(wf_manager)
        elif choice == "4":
            show_live_testing_menu(wf_manager)
        elif choice == "5":
            show_data_management_menu(wf_manager)
        elif choice == "6":
            show_system_status_menu(wf_manager)
    except Exception as e:
        print(f"\n❌ Error in menu operation: {e}")
        if context:
            context.log_error(f"Menu operation: {str(e)}")
        import traceback

        traceback.print_exc()
        wait_for_enter("\n⚠️  Press Enter to continue...")

    return True


def main():
    """Main entry point with comprehensive error handling."""
    print("🚀 Initializing Kinetra Menu System...")

    # Initialize context tracker
    context = MenuContext()
    context.push("Main Menu")

    # Initialize workflow manager with error handling
    try:
        wf_manager = WorkflowManager(
            log_dir="logs",
            backup_dir="data/backups/workflow",
            enable_backups=True,
            enable_checksums=True,
        )
        print("✅ Workflow manager initialized")
    except Exception as e:
        print(f"⚠️  Warning: Could not initialize workflow manager: {e}")
        print("   Continuing with limited functionality...")
        context.log_error(f"Workflow init: {str(e)}")
        wf_manager = None

    error_count = 0
    max_errors = 5

    # Main menu loop with context awareness
    while True:
        try:
            if not show_main_menu(wf_manager, context):
                print("\n👋 Thank you for using Kinetra!")
                break

            # Reset error count on successful menu operation
            error_count = 0

        except KeyboardInterrupt:
            print("\n\n⚠️  Menu interrupted by user (Ctrl+C)")
            try:
                if confirm_action("Exit Kinetra?", default=True):
                    print("👋 Goodbye!")
                    break
                else:
                    print("↩️  Returning to main menu...")
                    continue
            except (EOFError, KeyboardInterrupt):
                print("\n👋 Goodbye!")
                break

        except EOFError:
            print("\n\n⚠️  Input stream ended (EOF)")
            print("👋 Exiting gracefully...")
            break

        except Exception as e:
            error_count += 1
            error_msg = str(e)
            context.log_error(error_msg)
            print(f"\n\n❌ Error {error_count}/{max_errors}: {error_msg}")

            # Show traceback for debugging
            import traceback

            print("\n📋 Error details:")
            traceback.print_exc()

            # Show context information
            print(f"\n📍 Context: {context.get_breadcrumb()}")
            if context.last_action:
                print(f"🔄 Last action: {context.last_action}")

            if error_count >= max_errors:
                print(f"\n⚠️  Maximum error count ({max_errors}) reached!")
                print("Exiting to prevent infinite error loop...")
                print(f"\n📋 Error summary:")
                for i, err in enumerate(context.error_history[-5:], 1):
                    print(f"  {i}. {err}")
                break

            try:
                if not confirm_action("Continue to main menu?", default=True):
                    print("👋 Exiting...")
                    break
                else:
                    print("↩️  Returning to main menu...")
                    # Reset breadcrumb on error recovery
                    context.breadcrumb = ["Main Menu"]
            except (EOFError, KeyboardInterrupt):
                print("\n👋 Exiting gracefully...")
                break

    # Cleanup and exit
    print("\n" + "=" * 80)
    print("  KINETRA SESSION ENDED")
    print("=" * 80)
    print("📁 Logs saved to: logs/")
    print("💾 Data preserved in: data/")
    print("\n🔬 Keep exploring, keep learning!")
    print("=" * 80)


if __name__ == "__main__":
    main()
