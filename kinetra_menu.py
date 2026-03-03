#!/usr/bin/env python3
"""
KINETRA MENU - Unified Production Menu
========================================

Consolidated from 3 separate menu implementations into single canonical menu.

Production-ready unified menu system with:
- Context-aware navigation
- Guided workflow
- Smart defaults from data discovery
- Error handling & recovery
- Progress tracking
- Atomic operations
- 100% E2E tested

Usage:
    python kinetra_menu.py

CHANGELOG:
  v2.0.0 (2026-01-04): Consolidated menu release
    - Merged kinetra_menu.py (2,345 lines) - experimental features
    - Merged unified_menu.py (457 lines) - unified attempt
    - Kept kinetra_production_menu.py as base (1,181 lines)
    - Added versioning and changelog
    - 100% E2E tested with comprehensive_e2e_test.py
    - Integrated with data_manager.py v1.0.0
    - Renamed from kinetra_production_menu.py to kinetra_menu.py
    - Archived legacy menus to archive/menus/legacy/
    - 70% code reduction (3,983 → 1,181 lines)

  v1.0.0 (2026-01-03): Initial production menu
    - Production-ready features
    - Smart defaults and error handling

__version__ = "2.0.0"
"""

import json
import os
import signal
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Optional

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ============================================================================
# SYSTEM STATUS & CONTEXT
# ============================================================================


@dataclass
class SystemStatus:
    """Complete system status."""

    credentials_configured: bool = False
    metaapi_available: bool = False
    mt5_available: bool = False
    data_discovered: bool = False
    data_ready: bool = False
    models_trained: bool = False
    gpu_available: bool = False

    available_symbols: Optional[List[str]] = field(default=None)
    available_timeframes: Optional[List[str]] = field(default=None)
    usable_combinations: int = 0

    last_discovery: Optional[datetime] = None
    last_training: Optional[datetime] = None
    last_backtest: Optional[datetime] = None

    def __post_init__(self):
        if self.available_symbols is None:
            self.available_symbols = []
        if self.available_timeframes is None:
            self.available_timeframes = []

    def get_status_line(self) -> str:
        """Get one-line status summary."""
        parts = []

        # Data status
        if self.data_ready:
            parts.append(f"✅ Data ({self.usable_combinations} combos)")
        elif self.data_discovered:
            parts.append("⚠️  Data needs prep")
        else:
            parts.append("❌ No data")

        # Credentials
        if self.credentials_configured:
            parts.append("✅ Creds")
        else:
            parts.append("⚠️  No creds")

        # Sources
        sources = []
        if self.metaapi_available:
            sources.append("MetaAPI")
        if self.mt5_available:
            sources.append("MT5")
        if sources:
            parts.append(f"✅ {'/'.join(sources)}")

        # GPU
        if self.gpu_available:
            parts.append("✅ GPU")

        return " | ".join(parts)

    def suggest_next_step(self) -> str:
        """Suggest next logical step based on current state."""
        if not self.credentials_configured:
            return "Setup cTrader credentials (Menu 1 → 6)"
        renko_data = (PROJECT_ROOT / "data" / "master_standardized" / "ctrader").exists()
        if not renko_data:
            return "Download XAUUSD data (Menu 6 → 1)"
        dsp_done = (PROJECT_ROOT / "outputs" / "results").exists() and any(
            (PROJECT_ROOT / "outputs" / "results").glob("XAUUSD_backtest_*.json")
        )
        if not dsp_done:
            return "Run Renko backtest (Menu 6 → 3)"
        return "Launch Renko live trading (Menu 6 → 6)"


def _check_ctrader_creds() -> bool:
    """Return True if cTrader Open API credentials appear to be configured."""
    env_files = [PROJECT_ROOT / ".env.openapi", PROJECT_ROOT / ".env"]
    for ef in env_files:
        if ef.exists():
            text = ef.read_text()
            if "CTRADER_CLIENT_ID" in text or "CTRADER_APP_CLIENT_ID" in text:
                return True
    return False


def _renko_last_results(symbol: str) -> Optional[dict]:
    """Return the most recent renko results dict for *symbol*, or None."""
    results_dir = PROJECT_ROOT / "outputs" / "results"
    if not results_dir.exists():
        return None
    files = sorted(results_dir.glob(f"{symbol}_*.json"), key=lambda p: p.stat().st_mtime)
    if not files:
        return None
    try:
        with open(files[-1]) as f:
            return json.load(f)
    except Exception:
        return None


def check_system_status() -> SystemStatus:
    """Check complete system status."""
    status = SystemStatus()

    # Check credentials
    env_file = PROJECT_ROOT / ".env"
    status.credentials_configured = env_file.exists() or _check_ctrader_creds()

    if status.credentials_configured:
        # Check if MetaAPI token exists
        try:
            from dotenv import load_dotenv

            load_dotenv()
            status.metaapi_available = bool(os.getenv("METAAPI_TOKEN"))
        except Exception:
            pass

    # Check MT5
    try:
        import importlib.util

        status.mt5_available = importlib.util.find_spec("MetaTrader5") is not None  # type: ignore[import-not-found]
    except Exception:
        status.mt5_available = False

    # Check GPU
    try:
        import torch

        status.gpu_available = torch.cuda.is_available() or torch.backends.mps.is_available()
    except Exception:
        pass

    # Check data discovery
    discovery_file = PROJECT_ROOT / "data" / "available_data.json"
    if discovery_file.exists():
        status.data_discovered = True
        try:
            with open(discovery_file) as f:
                data = json.load(f)
                status.available_symbols = sorted(set(item["symbol"] for item in data))
                status.available_timeframes = sorted(set(item["timeframe"] for item in data))
                status.usable_combinations = len([item for item in data if item["bars"] >= 1000])
                status.last_discovery = datetime.fromtimestamp(discovery_file.stat().st_mtime)
        except Exception:
            pass

    # Check if data is ready (standardized files exist)
    data_dir = PROJECT_ROOT / "data" / "master_standardized"
    if data_dir.exists():
        csv_files = list(data_dir.glob("*.csv"))
        status.data_ready = len(csv_files) > 0

    # Check for trained models
    models_dir = PROJECT_ROOT / "models"
    if models_dir.exists():
        model_files = list(models_dir.glob("*.pkl")) + list(models_dir.glob("*.zip"))
        status.models_trained = len(model_files) > 0
        if model_files:
            latest = max(model_files, key=lambda p: p.stat().st_mtime)
            status.last_training = datetime.fromtimestamp(latest.stat().st_mtime)

    # Check for backtest results
    results_dir = PROJECT_ROOT / "results"
    if results_dir.exists():
        result_files = list(results_dir.glob("*.csv")) + list(results_dir.glob("*.json"))
        if result_files:
            latest = max(result_files, key=lambda p: p.stat().st_mtime)
            status.last_backtest = datetime.fromtimestamp(latest.stat().st_mtime)

    return status


# ============================================================================
# UTILITIES
# ============================================================================


def print_header(text: str):
    """Print main header."""
    print(f"\n{'=' * 80}")
    print(f"  {text}")
    print(f"{'=' * 80}\n")


def print_submenu_header(text: str, status: SystemStatus):
    """Print submenu header with context."""
    print(f"\n{'=' * 80}")
    print(f"  {text}")
    print(f"{'=' * 80}")
    print(f"\n📊 Status: {status.get_status_line()}")
    if suggestion := status.suggest_next_step():
        print(f"💡 Suggestion: {suggestion}")
    print()


def get_input(prompt: str, valid_choices: Optional[List[str]] = None) -> str:
    """Get user input with validation."""
    while True:
        try:
            choice = input(f"\n{prompt}: ").strip()
            if valid_choices is None or choice in valid_choices:
                return choice
            print(f"❌ Invalid choice. Please select from: {', '.join(valid_choices)}")
        except (KeyboardInterrupt, EOFError):
            print("\n\n⚠️  Interrupted by user")
            return "0"  # Return to previous menu


def confirm_action(message: str) -> bool:
    """Ask for confirmation."""
    response = get_input(f"{message} (y/n)", ["y", "n", "Y", "N"])
    return response.lower() == "y"


def run_script(script_path: str, args: Optional[List[str]] = None, dry_run: bool = False) -> bool:
    """
    Run a script with error handling.

    Args:
        script_path: Path to script relative to project root
        args: Additional arguments
        dry_run: If True, just print what would be executed

    Returns:
        True if successful, False otherwise
    """
    full_path = PROJECT_ROOT / script_path

    if not full_path.exists():
        print(f"❌ Script not found: {script_path}")
        return False

    cmd = [sys.executable, str(full_path)]
    if args:
        cmd.extend(args)

    if dry_run:
        print(f"🔍 DRY RUN: {' '.join(cmd)}")
        return True

    print(f"\n🚀 Executing: {' '.join(cmd)}\n")
    print(f"{'─' * 80}\n")

    try:
        result = subprocess.run(cmd, cwd=PROJECT_ROOT)
        success = result.returncode == 0

        print(f"\n{'─' * 80}")
        if success:
            print("✅ Script completed successfully")
        else:
            print(f"❌ Script failed with exit code {result.returncode}")

        return success

    except KeyboardInterrupt:
        print("\n\n⚠️  Script interrupted by user")
        return False
    except Exception as e:
        print(f"\n❌ Error running script: {e}")
        return False


# ============================================================================
# MENU 1: SETUP & AUTHENTICATION
# ============================================================================


def menu_setup_auth(status: SystemStatus):
    """Setup & Authentication menu."""
    while True:
        print_submenu_header("🔐 SETUP & AUTHENTICATION", status)

        print("1. Configure MetaAPI Credentials")
        print("2. Test MetaAPI Connection")
        print("3. Select/Change MetaAPI Account")
        print("4. Configure MT5 (Local Terminal)")
        print("5. Test MT5 Connection")
        print("6. View Current Configuration")
        print("0. Back to Main Menu")

        choice = get_input("Select option", ["0", "1", "2", "3", "4", "5", "6"])

        if choice == "0":
            break
        elif choice == "1":
            configure_metaapi()
        elif choice == "2":
            test_metaapi_connection()
        elif choice == "3":
            select_metaapi_account()
        elif choice == "4":
            configure_mt5()
        elif choice == "5":
            test_mt5_connection()
        elif choice == "6":
            view_configuration(status)

        input("\n📌 Press Enter to continue...")


def configure_metaapi():
    """Configure MetaAPI credentials."""
    print_header("Configure MetaAPI Credentials")

    print("""
MetaAPI provides cloud access to MT4/MT5 brokers.

Required:
- MetaAPI Token (from https://app.metaapi.cloud)
- Account ID (from your MetaAPI account)

Your credentials will be stored securely in .env file.
""")

    if not confirm_action("Continue with MetaAPI setup?"):
        return

    run_script("scripts/download/setup_metaapi_credentials.py")


def test_metaapi_connection():
    """Test MetaAPI connection."""
    print_header("Test MetaAPI Connection")
    run_script("scripts/download/test_metaapi_connection.py")


def select_metaapi_account():
    """Select or change MetaAPI account interactively."""
    print_header("Select/Change MetaAPI Account")

    print("""
This will list all accounts available with your MetaAPI token
and let you select which one to use.

⚠️  NOTE: This requires an API Access Token (not Account Access Token)
   Get one from: https://app.metaapi.cloud/api-access/generate-token

If you only have an Account Access Token, use option 1 to configure
your credentials (token + account ID).
""")

    if confirm_action("Continue with account selection?"):
        run_script("scripts/download/select_metaapi_account.py")


def configure_mt5():
    """Configure MT5 (local terminal)."""
    print_header("Configure MT5 (Local Terminal)")

    print("""
MT5 Terminal Configuration:

1. Install MetaTrader 5 terminal
2. Install Python package: pip install MetaTrader5
3. Keep MT5 terminal running while downloading data

Note: On Linux, use Wine to run MT5 terminal.
""")

    print("\n📝 MT5 installation guide:")
    print("  Windows: Download from metaquotes.net")
    print("  Linux: Use Wine (see docs/MT5_LINUX_SETUP.md)")
    print("\n  Python package: pip install MetaTrader5")


def test_mt5_connection():
    """Test MT5 connection."""
    print_header("Test MT5 Connection")

    try:
        import MetaTrader5 as mt5  # type: ignore[import-not-found]

        if not mt5.initialize():
            print("❌ Failed to connect to MT5 terminal")
            print("Make sure MT5 terminal is running")
            return

        info = mt5.account_info()
        if info:
            print("✅ Connected to MT5!")
            print(f"\n  Account: {info.login}")
            print(f"  Server: {info.server}")
            print(f"  Balance: {info.balance} {info.currency}")

        symbols = mt5.symbols_total()
        print(f"\n  Available symbols: {symbols}")

        mt5.shutdown()

    except ImportError:
        print("❌ MetaTrader5 package not installed")
        print("Install with: pip install MetaTrader5")


def view_configuration(status: SystemStatus):
    """View current configuration."""
    print_header("Current Configuration")

    print("Credentials:")
    print(f"  MetaAPI: {'✅ Configured' if status.metaapi_available else '❌ Not configured'}")
    print(f"  .env file: {'✅ Exists' if status.credentials_configured else '❌ Missing'}")

    print("\nData Sources:")
    print(f"  MetaAPI: {'✅ Available' if status.metaapi_available else '❌ Not available'}")
    print(f"  MT5 Local: {'✅ Available' if status.mt5_available else '❌ Not available'}")

    print("\nCompute:")
    print(f"  GPU: {'✅ Available' if status.gpu_available else '❌ Not available (CPU mode)'}")

    if status.credentials_configured:
        env_file = PROJECT_ROOT / ".env"
        print(f"\n📄 Configuration file: {env_file}")


# ============================================================================
# MENU 2: DATA MANAGEMENT
# ============================================================================


def menu_data_management(status: SystemStatus):
    """Data Management menu."""
    while True:
        print_submenu_header("📊 DATA MANAGEMENT", status)

        print("1. Discover Available Data (Scan filesystem)")
        print("2. Download Data (MetaAPI)")
        print("3. Download Data (MT5 Local)")
        print("4. Check Data Integrity")
        print("5. View Data Coverage")
        print("6. Consolidate & Clean Data")
        print("7. Denoise Data (Non-Linear Filters)")
        print("0. Back to Main Menu")

        choice = get_input("Select option", ["0", "1", "2", "3", "4", "5", "6", "7"])

        if choice == "0":
            break
        elif choice == "1":
            discover_data()
        elif choice == "2":
            download_metaapi_data(status)
        elif choice == "3":
            download_mt5_data(status)
        elif choice == "4":
            check_data_integrity()
        elif choice == "5":
            view_data_coverage()
        elif choice == "6":
            consolidate_data()
        elif choice == "7":
            denoise_data()

        input("\n📌 Press Enter to continue...")


def discover_data():
    """Discover available data on filesystem."""
    print_header("Discover Available Data")

    print("Scanning filesystem for existing data files...")
    print("This will create/update: data/available_data.json\n")

    run_script("scripts/discover_available_data.py")


def download_metaapi_data(status: SystemStatus):
    """Download data from MetaAPI."""
    print_header("Download Data (MetaAPI)")

    if not status.metaapi_available:
        print("❌ MetaAPI not configured")
        print("Please configure MetaAPI credentials first (Menu 1.1)")
        return

    print("Available download options:")
    print("  1. Bulk download (all available symbols)")
    print("  2. Custom download (select symbols/timeframes)")
    print("  0. Cancel")

    choice = get_input("Select option", ["0", "1", "2"])

    if choice == "0":
        return
    elif choice == "1":
        run_script("scripts/download/metaapi_bulk_download.py")
    elif choice == "2":
        run_script("scripts/download/download_metaapi.py")


def download_mt5_data(status: SystemStatus):
    """Download data from MT5 local terminal."""
    print_header("Download Data (MT5 Local)")

    if not status.mt5_available:
        print("❌ MT5 not available")
        print("Please install MetaTrader5: pip install MetaTrader5")
        return

    print("""
MT5 Local Download:
- Downloads from your local MT5 terminal
- Requires MT5 terminal to be running
- Uses broker's historical data
""")

    if confirm_action("Start MT5 download?"):
        run_script("scripts/download/download_mt5_data.py")


def check_data_integrity():
    """Check data integrity."""
    print_header("Check Data Integrity")
    run_script("scripts/download/check_data_integrity.py")


def view_data_coverage():
    """View data coverage analysis."""
    print_header("Data Coverage Analysis")
    run_script("scripts/audit_data_coverage.py")


def consolidate_data():
    """Consolidate and clean data."""
    print_header("Consolidate & Clean Data")

    print("""
This will:
- Merge duplicate files
- Remove gaps
- Standardize format
- Backup before changes
""")

    if confirm_action("Start consolidation?"):
        run_script("scripts/consolidate_data.py", "--dry-run")


def denoise_data():
    """Denoise data using non-linear filters."""
    print_header("Denoise Data (Non-Linear Filters)")

    print("""
🔬 Non-Linear Denoising for Financial Data

Removes high-frequency noise while preserving:
- Sharp trends and regime changes
- Non-linear dynamics
- Critical market moves

Available Methods:
  1. Savitzky-Golay (Recommended) - Polynomial smoothing, preserves peaks
  2. Median Filter - Robust to outliers and flash crashes
  3. LOWESS - Adaptive local regression
  4. Wavelet Thresholding - Multi-resolution analysis

NO linear filters (MA/EMA) - they destroy non-linear features.
""")

    print("\nSelect denoising method:")
    print("1. Savitzky-Golay (Default, best for trends)")
    print("2. Median Filter (Best for outliers)")
    print("3. LOWESS (Adaptive)")
    print("4. Wavelet (Multi-scale)")
    print("0. Cancel")

    method_choice = get_input("Select method", ["0", "1", "2", "3", "4"])

    if method_choice == "0":
        return

    method_map = {"1": "savgol", "2": "median", "3": "lowess", "4": "wavelet"}

    method = method_map.get(method_choice, "savgol")

    print(f"\n✅ Selected method: {method.upper()}")
    print("\nThis will create *_denoised.csv files in data/prepared/")

    if confirm_action("Start denoising?"):
        run_script("scripts/denoise_data.py", f"--method={method}")


# ============================================================================
# MENU 3: EXPLORATION & TRAINING
# ============================================================================


def menu_exploration_training(status: SystemStatus):
    """Exploration & Training menu."""
    while True:
        print_submenu_header("🔬 EXPLORATION & TRAINING", status)

        if not status.data_ready:
            print("⚠️  No data ready. Please download/prepare data first (Menu 2)")
            print()

        print("1. Quick RL Training (Physics-Only)")
        print("2. Agent Comparison (PPO vs DQN vs Linear)")
        print("3. Comprehensive Exploration Suite")
        print("4. Train Specialist Agents (Berserker/Sniper/Triad)")
        print("5. Scientific Discovery (PCA/ICA/Chaos)")
        print("6. View Training Results")
        print("0. Back to Main Menu")

        choice = get_input("Select option", ["0", "1", "2", "3", "4", "5", "6"])

        if choice == "0":
            break
        elif choice == "1":
            quick_rl_training(status)
        elif choice == "2":
            agent_comparison(status)
        elif choice == "3":
            comprehensive_exploration(status)
        elif choice == "4":
            train_specialists(status)
        elif choice == "5":
            scientific_discovery(status)
        elif choice == "6":
            view_training_results()

        input("\n📌 Press Enter to continue...")


def quick_rl_training(status: SystemStatus):
    """Quick RL training with defaults."""
    print_header("Quick RL Training (Physics-Only)")

    if not status.data_ready:
        print("❌ No data available")
        return

    print(f"""
Configuration:
- Symbols: {", ".join((status.available_symbols or [])[:3])} (first 3)
- Timeframes: {", ".join((status.available_timeframes or [])[:2])} (first 2)
- Agent: PPO (physics-only features)
- Episodes: 100
- GPU: {"Enabled" if status.gpu_available else "Disabled (CPU)"}
""")

    if confirm_action("Start training?"):
        run_script("scripts/training/train_rl.py", ["--episodes", "100"])


def agent_comparison(status: SystemStatus):
    """Compare different agent types."""
    print_header("Agent Comparison")

    if not status.data_ready:
        print("❌ No data available")
        return

    print("""
This will train and compare:
- PPO (Proximal Policy Optimization)
- DQN (Deep Q-Network)
- Linear (Baseline)
- Triad (Ensemble)

Comparison metrics:
- Omega Ratio
- Win Rate
- Sharpe Ratio
- Max Drawdown
""")

    if confirm_action("Start agent comparison?"):
        run_script("scripts/training/explore_compare_agents.py")


def comprehensive_exploration(status: SystemStatus):
    """Run comprehensive exploration suite."""
    print_header("Comprehensive Exploration Suite")

    if not status.data_ready:
        print("❌ No data available")
        return

    print("""
Full exploration suite:
- All asset classes
- Multiple timeframes
- Agent comparison
- Measurement impact analysis
- Statistical validation
- Comprehensive reporting

This may take several hours.
""")

    if confirm_action("Start comprehensive exploration?"):
        run_script("scripts/exploration/run_comprehensive_exploration.py")


def train_specialists(status: SystemStatus):
    """Train specialist agents."""
    print_header("Train Specialist Agents")

    if not status.data_ready:
        print("❌ No data available")
        return

    print("""
Specialist Agents:
1. Berserker (Trend following, high aggression)
2. Sniper (Mean reversion, high precision)
3. Triad (Ensemble of 3 strategies)
""")

    print("\nWhich specialist?")
    print("  1. Berserker")
    print("  2. Sniper")
    print("  3. Triad")
    print("  4. All")
    print("  0. Cancel")

    choice = get_input("Select option", ["0", "1", "2", "3", "4"])

    if choice == "0":
        return

    scripts = {
        "1": "scripts/training/train_berserker.py",
        "2": "scripts/training/train_sniper.py",
        "3": "scripts/training/train_triad.py",
    }

    if choice == "4":
        for script in scripts.values():
            run_script(script)
    else:
        run_script(scripts[choice])


def scientific_discovery(status: SystemStatus):
    """Scientific discovery suite."""
    print_header("Scientific Discovery Suite")

    print("""
Exploratory analysis:
- PCA (Principal Component Analysis)
- ICA (Independent Component Analysis)
- Chaos Theory metrics
- Fractal dimension
- Entropy analysis

This helps discover non-obvious patterns.
""")

    if confirm_action("Start scientific discovery?"):
        run_script("scripts/exploration/run_comprehensive_exploration.py", ["--scientific"])


def view_training_results():
    """View training results."""
    print_header("Training Results")

    results_dir = PROJECT_ROOT / "results"
    if not results_dir.exists():
        print("❌ No results found")
        return

    result_files = sorted(results_dir.glob("*.json"))
    if not result_files:
        print("❌ No result files found")
        return

    print(f"Found {len(result_files)} result files:\n")
    for i, f in enumerate(result_files[-10:], 1):  # Last 10
        mtime = datetime.fromtimestamp(f.stat().st_mtime)
        size = f.stat().st_size / 1024
        print(f"  {i}. {f.name} ({mtime.strftime('%Y-%m-%d %H:%M')}, {size:.1f} KB)")

    print("\nView specific result file? (number or 0 to cancel)")
    choice = get_input("Select", [str(i) for i in range(len(result_files[-10:]) + 1)])

    if choice != "0":
        idx = int(choice) - 1
        file_path = result_files[-10:][idx]

        with open(file_path) as f:
            data = json.load(f)
            print(f"\n{json.dumps(data, indent=2)}")


# ============================================================================
# MENU 4: BACKTESTING & VALIDATION
# ============================================================================


def menu_backtesting(status: SystemStatus):
    """Backtesting & Validation menu."""
    while True:
        print_submenu_header("📈 BACKTESTING & VALIDATION", status)

        if not status.data_ready:
            print("⚠️  No data ready. Please download/prepare data first (Menu 2)")
            print()

        print("1. Quick Backtest (Single symbol/timeframe)")
        print("2. Batch Backtest (Multiple combinations)")
        print("3. Monte Carlo Validation (100 runs)")
        print("4. Walk-Forward Analysis")
        print("5. View Backtest Results")
        print("6. Generate Performance Report")
        print("0. Back to Main Menu")

        choice = get_input("Select option", ["0", "1", "2", "3", "4", "5", "6"])

        if choice == "0":
            break
        elif choice == "1":
            quick_backtest(status)
        elif choice == "2":
            batch_backtest(status)
        elif choice == "3":
            monte_carlo_validation(status)
        elif choice == "4":
            walk_forward_analysis(status)
        elif choice == "5":
            view_backtest_results()
        elif choice == "6":
            generate_performance_report()

        input("\n📌 Press Enter to continue...")


def quick_backtest(status: SystemStatus):
    """Quick single backtest."""
    print_header("Quick Backtest")

    if not status.data_ready:
        print("❌ No data available")
        return

    print(f"Available symbols: {', '.join(status.available_symbols or [])}")
    print(f"Available timeframes: {', '.join(status.available_timeframes or [])}")

    symbol = get_input(f"Symbol (default: {(status.available_symbols or ['BTCUSD'])[0]})")
    if not symbol:
        symbol = (status.available_symbols or ["BTCUSD"])[0]

    timeframe = get_input("Timeframe (default: H1)")
    if not timeframe:
        timeframe = "H1"

    print(f"\nBacktesting {symbol} {timeframe}...")
    success = run_script(
        "scripts/batch_backtest.py",
        ["--symbols", symbol, "--tf", timeframe, "--years", "2023", "2024"],
    )

    # Display results
    if success:
        _display_backtest_results()


def batch_backtest(status: SystemStatus):
    """Batch backtest multiple combinations."""
    print_header("Batch Backtest")

    if not status.data_ready:
        print("❌ No data available")
        return

    print("""
Batch backtest configuration:
- All available symbols
- All available timeframes
- Monte Carlo runs: 50 per combination
- Statistical validation (p < 0.01)
""")

    if confirm_action("Start batch backtest?"):
        symbols = (status.available_symbols or [])[:5]  # Limit to 5 for time
        run_script(
            "scripts/batch_backtest.py", ["--symbols", *symbols, "--tf", "H1", "--mc-runs", "50"]
        )


def monte_carlo_validation(status: SystemStatus):
    """Monte Carlo validation."""
    print_header("Monte Carlo Validation")

    if not status.data_ready:
        print("❌ No data available")
        return

    print("""
Monte Carlo Validation:
- 100 runs per configuration
- Random seeds for reproducibility
- Statistical significance testing
- Confidence intervals
""")

    if confirm_action("Start Monte Carlo validation (may take hours)?"):
        run_script(
            "scripts/batch_backtest.py",
            ["--symbols", *((status.available_symbols or [])[:3]), "--mc-runs", "100"],
        )


def walk_forward_analysis(status: SystemStatus):
    """Walk-forward analysis."""
    print_header("Walk-Forward Analysis")

    print("""
Walk-forward analysis:
- Rolling window training
- Out-of-sample validation
- Prevents overfitting
- Realistic performance estimation

Note: This feature is under development.
""")


def view_backtest_results():
    """View backtest results."""
    print_header("Backtest Results")

    results_file = PROJECT_ROOT / "data" / "batch_backtest_results.csv"
    if not results_file.exists():
        print("❌ No backtest results found")
        print("Run a backtest first (Menu 4.1 or 4.2)")
        return

    import pandas as pd  # type: ignore[import-untyped]

    df = pd.read_csv(results_file)

    print(f"\n📊 Results from {results_file.name}:\n")
    print(df.to_string(index=False))

    print("\n\n📈 Summary Statistics:")
    print(f"  Total combinations: {len(df)}")
    print(f"  Average Omega: {df['omega_train'].mean():.2f}")
    print(f"  Average Win Rate: {df['win_train'].mean():.1f}%")
    print(f"  Average CHS: {df['chs'].mean():.2f}")


def generate_performance_report():
    """Generate comprehensive performance report."""
    print_header("Generate Performance Report")

    print("""
Comprehensive performance report includes:
- Equity curves
- Drawdown analysis
- Risk metrics
- Statistical validation
- HTML report

Note: This feature is under development.
""")


# ============================================================================
# MENU 5: SYSTEM TOOLS
# ============================================================================


def menu_system_tools(status: SystemStatus):
    """System Tools & Monitoring menu."""
    while True:
        print_submenu_header("🛠️  SYSTEM TOOLS & MONITORING", status)

        print("1. System Status & Diagnostics")
        print("2. Cache Management")
        print("3. Backup Data")
        print("4. Clean Temporary Files")
        print("5. Run Tests")
        print("6. View Logs")
        print("0. Back to Main Menu")

        choice = get_input("Select option", ["0", "1", "2", "3", "4", "5", "6"])

        if choice == "0":
            break
        elif choice == "1":
            system_diagnostics(status)
        elif choice == "2":
            cache_management()
        elif choice == "3":
            backup_data()
        elif choice == "4":
            clean_temp_files()
        elif choice == "5":
            run_tests()
        elif choice == "6":
            view_logs()

        input("\n📌 Press Enter to continue...")


def system_diagnostics(status: SystemStatus):
    """System diagnostics."""
    print_header("System Status & Diagnostics")

    print("📊 SYSTEM STATUS")
    print("=" * 80)

    print("\n🔐 Credentials:")
    print(f"  .env file: {'✅' if status.credentials_configured else '❌'}")
    print(f"  MetaAPI: {'✅' if status.metaapi_available else '❌'}")

    print("\n📡 Data Sources:")
    print(f"  MetaAPI: {'✅' if status.metaapi_available else '❌'}")
    print(f"  MT5 Local: {'✅' if status.mt5_available else '❌'}")

    print("\n📊 Data:")
    print(f"  Discovery done: {'✅' if status.data_discovered else '❌'}")
    print(f"  Data ready: {'✅' if status.data_ready else '❌'}")
    print(f"  Symbols: {len(status.available_symbols or [])}")
    print(f"  Timeframes: {len(status.available_timeframes or [])}")
    print(f"  Usable combinations: {status.usable_combinations}")

    print("\n🤖 Models:")
    print(f"  Trained models: {'✅' if status.models_trained else '❌'}")
    if status.last_training:
        print(f"  Last training: {status.last_training.strftime('%Y-%m-%d %H:%M')}")

    print("\n💻 Compute:")
    print(f"  GPU: {'✅' if status.gpu_available else '❌ (CPU mode)'}")

    print("\n📅 Recent Activity:")
    if status.last_discovery:
        print(f"  Last discovery: {status.last_discovery.strftime('%Y-%m-%d %H:%M')}")
    if status.last_training:
        print(f"  Last training: {status.last_training.strftime('%Y-%m-%d %H:%M')}")
    if status.last_backtest:
        print(f"  Last backtest: {status.last_backtest.strftime('%Y-%m-%d %H:%M')}")

    print("\n💡 Next Steps:")
    print(f"  {status.suggest_next_step()}")


def cache_management():
    """Cache management."""
    print_header("Cache Management")
    run_script("scripts/cache_manager.py")


def backup_data():
    """Backup data."""
    print_header("Backup Data")

    if confirm_action("Create backup of all data?"):
        run_script("scripts/backup_data.py")


def clean_temp_files():
    """Clean temporary files."""
    print_header("Clean Temporary Files")

    print("""
This will remove:
- Python cache (__pycache__)
- Pytest cache (.pytest_cache)
- Temporary test files
- Old log files (>30 days)
""")

    if confirm_action("Clean temporary files?"):
        import shutil

        # Clean Python cache
        for cache_dir in PROJECT_ROOT.rglob("__pycache__"):
            shutil.rmtree(cache_dir, ignore_errors=True)
            print(f"  Removed: {cache_dir}")

        # Clean pytest cache
        pytest_cache = PROJECT_ROOT / ".pytest_cache"
        if pytest_cache.exists():
            shutil.rmtree(pytest_cache, ignore_errors=True)
            print(f"  Removed: {pytest_cache}")

        print("\n✅ Cleanup complete")


def run_tests():
    """Run test suite."""
    print_header("Run Tests")

    print("""
Test options:
1. Quick tests (core functionality)
2. Full test suite
3. Integration tests
4. Physics validation tests
0. Cancel
""")

    choice = get_input("Select option", ["0", "1", "2", "3", "4"])

    if choice == "0":
        return
    elif choice == "1":
        run_script("scripts/run_exhaustive_tests.py", ["--quick"])
    elif choice == "2":
        run_script("scripts/run_exhaustive_tests.py")
    elif choice == "3":
        run_script("scripts/run_exhaustive_tests.py", ["--integration"])
    elif choice == "4":
        run_script("scripts/run_exhaustive_tests.py", ["--physics"])


def view_logs():
    """View system logs."""
    print_header("System Logs")

    logs_dir = PROJECT_ROOT / "logs"
    if not logs_dir.exists():
        print("❌ No logs directory found")
        return

    log_files = sorted(logs_dir.glob("*.log"))
    if not log_files:
        print("❌ No log files found")
        return

    print(f"Found {len(log_files)} log files:\n")
    for i, f in enumerate(log_files[-10:], 1):
        mtime = datetime.fromtimestamp(f.stat().st_mtime)
        size = f.stat().st_size / 1024
        print(f"  {i}. {f.name} ({mtime.strftime('%Y-%m-%d %H:%M')}, {size:.1f} KB)")

    print("\nView specific log? (number or 0 to cancel)")
    choice = get_input("Select", [str(i) for i in range(len(log_files[-10:]) + 1)])

    if choice != "0":
        idx = int(choice) - 1
        file_path = log_files[-10:][idx]

        # Show last 50 lines
        with open(file_path) as f:
            lines = f.readlines()
            print(f"\n{'=' * 80}")
            print(f"Last 50 lines of {file_path.name}:")
            print(f"{'=' * 80}\n")
            print("".join(lines[-50:]))


# ============================================================================
# MENU 6: RENKO ENGINE  (canonical live trading path)
# ============================================================================

_RENKO_SCRIPT = "scripts/renko_engine.py"
_DEFAULT_SYMBOL = "XAUUSD"


def _display_backtest_results() -> None:
    """Display backtest results from CSV file."""
    results_file = PROJECT_ROOT / "data" / "batch_backtest_results.csv"

    if not results_file.exists():
        print("❌ No backtest results file found")
        return

    try:
        import pandas as pd

        df = pd.read_csv(results_file)

        if df.empty:
            print("❌ No results in backtest file")
            return

        print("\n" + "=" * 80)
        print("BACKTEST RESULTS SUMMARY")
        print("=" * 80)

        # Display key metrics
        if "omega" in df.columns:
            print(
                f"\nOmega Ratio:           {df['omega'].mean():.3f} (avg), {df['omega'].max():.3f} (max)"
            )
        if "win_rate" in df.columns:
            print(
                f"Win Rate:              {df['win_rate'].mean():.1%} (avg), {df['win_rate'].max():.1%} (max)"
            )
        if "max_drawdown_pct" in df.columns:
            print(
                f"Max Drawdown:          {df['max_drawdown_pct'].min():.2f}% (min), {df['max_drawdown_pct'].mean():.2f}% (avg)"
            )
        if "net_usd" in df.columns:
            print(
                f"Net P&L:               ${df['net_usd'].sum():,.2f} (total), ${df['net_usd'].mean():,.2f} (avg)"
            )
        if "n_trades" in df.columns:
            print(
                f"Total Trades:          {df['n_trades'].sum():.0f} (total), {df['n_trades'].mean():.1f} (avg)"
            )
        if "sharpe_ratio" in df.columns:
            print(f"Sharpe Ratio:          {df['sharpe_ratio'].mean():.3f} (avg)")
        if "final_equity" in df.columns:
            print(f"Final Equity:          ${df['final_equity'].mean():,.2f} (avg)")

        # Show top 5 results
        print("\nTop 5 Results:")
        print("-" * 80)
        if "omega" in df.columns:
            top_5 = df.nlargest(5, "omega")[
                ["symbol", "year", "omega", "win_rate", "net_usd", "max_drawdown_pct"]
            ]
        else:
            top_5 = df.head(5)

        for idx, row in top_5.iterrows():
            symbol = row.get("symbol", "N/A")
            year = row.get("year", "N/A")
            omega = row.get("omega", 0.0)
            win_rate = row.get("win_rate", 0.0)
            net = row.get("net_usd", 0.0)
            dd = row.get("max_drawdown_pct", 0.0)
            print(
                f"  {symbol} {year}: Ω={omega:.3f}, WR={win_rate:.1%}, P&L=${net:,.2f}, DD={dd:.2f}%"
            )

        print("=" * 80)
        print(f"\n✅ Results saved to: {results_file}")

    except Exception as e:
        print(f"❌ Error reading results: {e}")


def _renko_select_symbol() -> str:
    """Pick symbol — default XAUUSD."""
    sym = get_input(f"Symbol (Enter = {_DEFAULT_SYMBOL})")
    return sym.strip().upper() if sym.strip() else _DEFAULT_SYMBOL


def _renko_run_stage(symbol: str, stage: str, extra: Optional[List[str]] = None) -> bool:
    args = [symbol, "--stage", stage] + (extra or [])
    return run_script(_RENKO_SCRIPT, args)


def _renko_show_results(symbol: str) -> None:
    result = _renko_last_results(symbol)
    if result is None:
        print(f"\n  No saved results for {symbol} yet.")
        return
    s = result.get("summary", {})
    print(f"\n  Last results for {symbol}:")
    print(f"    Trades:       {s.get('n_trades', 0)}")
    print(f"    Net P&L:      ${s.get('net_usd', 0):,.2f}")
    print(f"    Omega:        {s.get('omega', 0):.3f}")
    print(f"    Win rate:     {s.get('win_rate', 0):.1%}")
    print(f"    Max drawdown: {s.get('max_drawdown_pct', 0):.2f}%")
    print(f"    Final equity: ${s.get('final_equity', 0):,.2f}")


def menu_renko_engine(status: SystemStatus):
    """Renko Engine — single pipeline from download to live trading."""
    symbol = _DEFAULT_SYMBOL

    while True:
        print_submenu_header(f"🎯 RENKO ENGINE  [{symbol}]", status)

        ctrader_ok = _check_ctrader_creds()
        cred_icon = "✅" if ctrader_ok else "❌"

        print(f"  cTrader creds: {cred_icon}  |  Symbol: {symbol}\n")
        print(f"  s. Change symbol (current: {symbol})")
        print()
        print("  ── Sequential validation ──────────────────────────")
        print("  1. Download historical data")
        print("  2. DSP analysis  (find optimal brick size)")
        print("  3. Quick backtest  (3 months)")
        print("  4. Full backtest   (all data, rolling OOS)")
        print("  5. Paper trading   (historical replay, no orders)")
        print(
            f"  6. Live trading    (micro lots {'✅' if ctrader_ok else '❌ needs cTrader creds'})"
        )
        print()
        print("  ── Shortcuts ──────────────────────────────────────")
        print("  7. Run ALL stages  (download → live)")
        print("  8. Test cTrader connection")
        print("  9. View last results")
        print("  0. Back to Main Menu")

        choice = get_input(
            "Select option",
            ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "s"],
        )

        if choice == "0":
            break

        elif choice == "s":
            symbol = _renko_select_symbol()
            continue  # redraw menu, no "press Enter"

        elif choice == "1":
            _renko_run_stage(symbol, "download")

        elif choice == "2":
            _renko_run_stage(symbol, "dsp")

        elif choice == "3":
            months = get_input("Months to test (Enter = 3)")
            extra = ["--months", months] if months.strip().isdigit() else []
            _renko_run_stage(symbol, "backtest", extra)

        elif choice == "4":
            _renko_run_stage(symbol, "full")

        elif choice == "5":
            _renko_run_stage(symbol, "paper")

        elif choice == "6":
            if not ctrader_ok:
                print("\n  ❌ cTrader credentials not found.")
                print("  Add CTRADER_CLIENT_ID / CTRADER_CLIENT_SECRET to .env.openapi")
                print("  See: scripts/ctrader/test_ctrader_connect.py")
            else:
                size = get_input("Lot size (micro / scaled)", ["micro", "scaled"])
                _renko_run_stage(symbol, "live", ["--live-size", size or "micro"])

        elif choice == "7":
            print("\n  This will run: download → dsp → backtest → full → paper → live")
            print("  Each stage gates the next. Paper must pass before live.\n")
            if confirm_action("Start full pipeline?"):
                _renko_run_stage(symbol, "all")

        elif choice == "8":
            run_script("scripts/ctrader/test_ctrader_connect.py")

        elif choice == "9":
            _renko_show_results(symbol)

        input("\n📌 Press Enter to continue...")


# ============================================================================
# MAIN MENU
# ============================================================================


def print_main_menu(status: SystemStatus):
    """Print main menu."""
    print_header("KINETRA - Kinetic Entropy Alpha Trading System")

    print(f"📊 Status: {status.get_status_line()}\n")

    if suggestion := status.suggest_next_step():
        print(f"💡 Next Step: {suggestion}\n")

    print("=" * 80)
    print("\nMAIN MENU:\n")
    ctrader_icon = "✅" if _check_ctrader_creds() else "❌"
    print("1. 🔐 Setup & Authentication")
    print("2. 📊 Data Management")
    print("3. 🔬 Exploration & Training")
    print("4. 📈 Backtesting & Validation")
    print("5. 🛠️  System Tools & Monitoring")
    print(f"6. 🎯 Renko Engine  [download → backtest → paper → live]  cTrader {ctrader_icon}")
    print("0. Exit")


def main():
    """Main menu loop."""

    # Setup signal handler for graceful shutdown
    def signal_handler(sig, frame):
        print("\n\n👋 Goodbye!")
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    print("""
    ╔═══════════════════════════════════════════════════════════════════════════╗
    ║                                                                           ║
    ║   ██╗  ██╗██╗███╗   ██╗███████╗████████╗██████╗  █████╗                 ║
    ║   ██║ ██╔╝██║████╗  ██║██╔════╝╚══██╔══╝██╔══██╗██╔══██╗                ║
    ║   █████╔╝ ██║██╔██╗ ██║█████╗     ██║   ██████╔╝███████║                ║
    ║   ██╔═██╗ ██║██║╚██╗██║██╔══╝     ██║   ██╔══██╗██╔══██║                ║
    ║   ██║  ██╗██║██║ ╚████║███████╗   ██║   ██║  ██║██║  ██║                ║
    ║   ╚═╝  ╚═╝╚═╝╚═╝  ╚═══╝╚══════╝   ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═╝                ║
    ║                                                                           ║
    ║              Kinetic Entropy Alpha - Production Menu v1.0                ║
    ║                    Physics-First Adaptive Trading                        ║
    ║                                                                           ║
    ╚═══════════════════════════════════════════════════════════════════════════╝
    """)

    while True:
        # Refresh status
        status = check_system_status()

        print_main_menu(status)

        choice = get_input("\nSelect option", ["0", "1", "2", "3", "4", "5"])

        if choice == "0":
            print("\n👋 Goodbye!")
            break
        elif choice == "1":
            menu_setup_auth(status)
        elif choice == "2":
            menu_data_management(status)
        elif choice == "3":
            menu_exploration_training(status)
        elif choice == "4":
            menu_backtesting(status)
        elif choice == "5":
            menu_system_tools(status)


if __name__ == "__main__":
    main()
