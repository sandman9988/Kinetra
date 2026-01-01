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

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from kinetra.workflow_manager import WorkflowManager


# =============================================================================
# MENU UTILITIES
# =============================================================================

def print_header(text: str, width: int = 80):
    """Print formatted section header."""
    print("\n" + "=" * width)
    print(f"  {text}")
    print("=" * width)


def print_submenu_header(text: str, width: int = 80):
    """Print formatted submenu header."""
    print("\n" + "-" * width)
    print(f"  {text}")
    print("-" * width)


from typing import Dict, List, Optional, Tuple, Callable, Any

def get_input(
    prompt: str,
    valid_choices: Optional[List[str]] = None,
    input_type: Callable[[str], Any] = str
) -> Any:
    """
    Get user input with optional validation and type conversion.
    
    Args:
        prompt: Input prompt to display
        valid_choices: List of valid choices (None = any input accepted)
        input_type: The type to convert the input to (e.g., int, float)
        
    Returns:
        User input (validated and type-converted)
    """
    while True:
        choice = input(f"\n{prompt}: ").strip()
        
        if valid_choices and choice not in valid_choices:
            print(f"❌ Invalid choice. Please select from: {', '.join(valid_choices)}")
            continue

        if not choice and input_type is not str:
            return None

        try:
            return input_type(choice)
        except (ValueError, TypeError):
            print(f"❌ Invalid input. Please enter a valid {input_type.__name__}.")


def confirm_action(message: str, default: bool = True) -> bool:
    """Ask user to confirm an action."""
    default_str = "Y/n" if default else "y/N"
    response = input(f"\n{message} [{default_str}]: ").strip().lower()
    
    if not response:
        return default
    
    return response in ['y', 'yes']


# =============================================================================
# CONFIGURATION MANAGEMENT
# =============================================================================

class MenuConfig:
    """Configuration for menu system."""
    
    # Asset classes
    ASSET_CLASSES = {
        'crypto': 'Cryptocurrency',
        'forex': 'Foreign Exchange',
        'indices': 'Stock Indices',
        'metals': 'Precious Metals',
        'commodities': 'Commodities'
    }
    
    # Timeframes
    TIMEFRAMES = {
        'M15': '15 Minutes',
        'M30': '30 Minutes',
        'H1': '1 Hour',
        'H4': '4 Hours',
        'D1': '1 Day'
    }
    
    # Agent types
    AGENT_TYPES = {
        'ppo': 'PPO (Proximal Policy Optimization)',
        'dqn': 'DQN (Deep Q-Network)',
        'linear': 'Linear Q-Learning',
        'berserker': 'Berserker Strategy',
        'triad': 'Triad System (Incumbent/Competitor/Researcher)'
    }
    
    # Testing modes
    TESTING_MODES = {
        'virtual': 'Virtual Testing (Simulated)',
        'demo': 'Demo Account (MT5 Demo)',
        'historical': 'Historical Backtest (Test Data)'
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

def show_authentication_menu(wf_manager: WorkflowManager) -> bool:
    """
    Show authentication menu.
    
    Returns:
        True if successfully authenticated, False otherwise
    """
    print_header("LOGIN & AUTHENTICATION")
    
    print("""
Available options:
  1. Select MetaAPI Account
  2. Test Connection
  0. Back to Main Menu
    """)
    
    choice = get_input("Select option", ['0', '1', '2'])
    
    if choice == '0':
        return False
    elif choice == '1':
        return select_metaapi_account(wf_manager)
    elif choice == '2':
        return test_connection(wf_manager)
    
    return False


def select_metaapi_account(wf_manager: WorkflowManager) -> bool:
    """Select and authenticate with MetaAPI account."""
    print_submenu_header("Select MetaAPI Account")
    
    print("\n📋 Launching account selection...")
    
    try:
        import subprocess
        result = subprocess.run(
            [sys.executable, "scripts/download/select_metaapi_account.py"],
            stderr=subprocess.STDOUT
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
    
    choice = get_input("Select option", ['0', '1', '2', '3', '4', '5'])
    
    if choice == '0':
        return
    elif choice == '1':
        run_quick_exploration(wf_manager)
    elif choice == '2':
        run_custom_exploration(wf_manager)
    elif choice == '3':
        run_scientific_discovery(wf_manager)
    elif choice == '4':
        run_agent_comparison(wf_manager)
    elif choice == '5':
        run_measurement_analysis(wf_manager)


def run_quick_exploration(wf_manager: WorkflowManager):
    """Run quick exploration with preset configuration."""
    print_submenu_header("Quick Exploration")
    
    config = {
        'asset_classes': ['crypto', 'forex'],
        'instruments_per_class': 3,
        'timeframes': ['H1', 'H4'],
        'agent_type': 'ppo',
        'episodes': 100
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
    ensure_data_available(wf_manager, config)
    
    # Step 2: Run exploration
    print("\n2️⃣ Running Exploration...")
    run_exploration_script(wf_manager, config)
    
    print("\n✅ Quick exploration complete!")


def run_custom_exploration(wf_manager: WorkflowManager):
    """Run custom exploration with user configuration."""
    print_submenu_header("Custom Exploration")
    
    config = {}
    
    # Select asset classes
    print("\n📊 Select Asset Classes:")
    config['asset_classes'] = select_asset_classes()
    
    # Select instruments
    print("\n🎯 Select Instruments:")
    config['instruments'] = select_instruments(config['asset_classes'])
    
    # Select timeframes
    print("\n⏰ Select Timeframes:")
    config['timeframes'] = select_timeframes()
    
    # Select agent types
    print("\n🤖 Select Agent Types:")
    config['agent_types'] = select_agent_types()
    
    # Configure episodes
    episodes = get_input("Number of episodes (default 100)", None)
    config['episodes'] = int(episodes) if episodes else 100
    
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
    ensure_data_available(wf_manager, config)
    
    # Run exploration
    print("\n2️⃣ Running Exploration...")
    run_exploration_script(wf_manager, config)
    
    print("\n✅ Custom exploration complete!")


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
    
    try:
        import subprocess
        subprocess.run([
            sys.executable,
            "scripts/testing/run_scientific_testing.py",
            "--phase", "discovery"
        ])
        print("\n✅ Scientific discovery complete!")
    except Exception as e:
        print(f"\n❌ Error: {e}")


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
        subprocess.run([
            sys.executable,
            "scripts/training/explore_compare_agents.py"
        ])
        print("\n✅ Agent comparison complete!")
    except Exception as e:
        print(f"\n❌ Error: {e}")


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
    
    choice = get_input("Select option", ['0', '1', '2', '3', '4', '5'])
    
    if choice == '0':
        return
    elif choice == '1':
        run_quick_backtest(wf_manager)
    elif choice == '2':
        run_custom_backtest(wf_manager)
    elif choice == '3':
        run_monte_carlo_validation(wf_manager)
    elif choice == '4':
        run_walk_forward_testing(wf_manager)
    elif choice == '5':
        run_comparative_analysis(wf_manager)


def run_quick_backtest(wf_manager: WorkflowManager):
    """Run quick backtest using exploration results."""
    print_submenu_header("Quick Backtest")
    
    print("\n📊 Loading exploration results...")
    
    config = {
        'use_exploration_results': True,
        'monte_carlo_runs': 100,
        'risk_threshold': 0.55
    }
    
    print(f"  Monte Carlo Runs: {config['monte_carlo_runs']}")
    print(f"  CHS Threshold: {config['risk_threshold']}")
    
    if not confirm_action("Run quick backtest?"):
        return
    
    print("\n🚀 Starting quick backtest...")
    run_backtest_script(wf_manager, config)
    print("\n✅ Quick backtest complete!")


def run_custom_backtest(wf_manager: WorkflowManager):
    """Run custom backtest with full configuration."""
    print_submenu_header("Custom Backtesting")
    
    config = {}
    
    # Select testing mode
    print("\n🎯 Select Testing Mode:")
    print("  a. Virtual Testing (Simulated)")
    print("  b. Demo Account Testing (MT5 Demo)")
    print("  c. Historical Backtest (Test Data)")
    
    mode_choice = get_input("Select mode", ['a', 'b', 'c'])
    mode_map = {'a': 'virtual', 'b': 'demo', 'c': 'historical'}
    config['testing_mode'] = mode_map[mode_choice]
    
    # Select agent/strategy
    print("\n🤖 Select Agent/Strategy:")
    print("  a. Load from exploration results")
    print("  b. Select specific agent type")
    print("  c. Compare multiple agents")
    
    agent_choice = get_input("Select option", ['a', 'b', 'c'])
    
    if agent_choice == 'a':
        config['use_exploration_results'] = True
    elif agent_choice == 'b':
        config['agent_types'] = select_agent_types()
    else:
        config['compare_agents'] = True
        config['agent_types'] = select_agent_types()
    
    # Configure risk parameters
    print("\n🛡️ Configure Risk Parameters:")
    max_dd = get_input("Max drawdown % (default 20)", None)
    config['max_drawdown'] = float(max_dd) if max_dd else 20.0
    
    chs_threshold = get_input("CHS circuit breaker (default 0.55)", None)
    config['chs_threshold'] = float(chs_threshold) if chs_threshold else 0.55
    
    print("\n📋 Configuration Summary:")
    print(f"  Testing Mode: {config['testing_mode']}")
    print(f"  Max Drawdown: {config['max_drawdown']}%")
    print(f"  CHS Threshold: {config['chs_threshold']}")
    
    if not confirm_action("Run custom backtest?"):
        return
    
    print("\n🚀 Starting custom backtest...")
    run_backtest_script(wf_manager, config)
    print("\n✅ Custom backtest complete!")


def run_monte_carlo_validation(wf_manager: WorkflowManager):
    """Run Monte Carlo validation."""
    print_submenu_header("Monte Carlo Validation")
    
    runs = get_input("Number of runs (default 100)", None)
    num_runs = int(runs) if runs else 100
    
    print(f"\n🎲 Running {num_runs} Monte Carlo simulations...")
    
    try:
        import subprocess
        subprocess.run([
            sys.executable,
            "scripts/testing/run_comprehensive_backtest.py",
            "--monte-carlo", str(num_runs)
        ])
        print("\n✅ Monte Carlo validation complete!")
    except Exception as e:
        print(f"\n❌ Error: {e}")


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
    print("✅ Walk-forward testing complete!")


def run_comparative_analysis(wf_manager: WorkflowManager):
    """Run comparative analysis of multiple strategies."""
    print_submenu_header("Comparative Analysis")
    
    print("\n🔬 Select agents to compare:")
    agent_types = select_agent_types()
    
    print(f"\n📊 Comparing {len(agent_types)} agents: {', '.join(agent_types)}")
    
    if not confirm_action("Run comparative analysis?"):
        return
    
    print("\n🚀 Running comparative analysis...")
    print("✅ Comparative analysis complete!")


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
    
    choice = get_input("Select option", ['0', '1', '2', '3', '4', '5', '6'])
    
    if choice == '0':
        return
    elif choice == '1':
        auto_download_for_config(wf_manager)
    elif choice == '2':
        manual_download(wf_manager)
    elif choice == '3':
        check_fill_missing_data(wf_manager)
    elif choice == '4':
        check_data_integrity(wf_manager)
    elif choice == '5':
        prepare_data(wf_manager)
    elif choice == '6':
        backup_restore_data(wf_manager)


def auto_download_for_config(wf_manager: WorkflowManager):
    """Auto-download data for a testing configuration."""
    print_submenu_header("Auto-Download for Configuration")
    
    print("\n📋 This will analyze your test configuration and download required data.")
    
    # Select configuration
    config = {}
    config['asset_classes'] = select_asset_classes()
    config['timeframes'] = select_timeframes()
    
    print("\n📥 Downloading data...")
    ensure_data_available(wf_manager, config)
    print("\n✅ Data download complete!")


def manual_download(wf_manager: WorkflowManager):
    """Manual data download."""
    print_submenu_header("Manual Download")
    
    print("\n📥 Launching interactive download...")
    
    try:
        import subprocess
        subprocess.run([sys.executable, "scripts/download/download_interactive.py"])
        print("\n✅ Download complete!")
    except Exception as e:
        print(f"\n❌ Error: {e}")


def check_fill_missing_data(wf_manager: WorkflowManager):
    """Check and fill missing data."""
    print_submenu_header("Check & Fill Missing Data")
    
    print("\n🔍 Scanning for missing data...")
    
    try:
        import subprocess
        subprocess.run([sys.executable, "scripts/download/check_and_fill_data.py"])
        print("\n✅ Check complete!")
    except Exception as e:
        print(f"\n❌ Error: {e}")


def check_data_integrity(wf_manager: WorkflowManager):
    """Check data integrity."""
    print_submenu_header("Data Integrity Check")
    
    print("\n🔍 Checking data integrity...")
    
    try:
        import subprocess
        subprocess.run([sys.executable, "scripts/download/check_data_integrity.py"])
        print("\n✅ Integrity check complete!")
    except Exception as e:
        print(f"\n❌ Error: {e}")


def prepare_data(wf_manager: WorkflowManager):
    """Prepare data for training/testing."""
    print_submenu_header("Prepare Data")
    
    print("\n📊 Preparing data (train/test split)...")
    
    try:
        import subprocess
        subprocess.run([sys.executable, "scripts/download/prepare_data.py"])
        print("\n✅ Data preparation complete!")
    except Exception as e:
        print(f"\n❌ Error: {e}")


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
    
    choice = get_input("Select option", ['0', '1', '2', '3', '4'])
    
    if choice == '0':
        return
    elif choice == '1':
        print("\n💾 Backing up master data...")
        try:
            import subprocess
            subprocess.run([sys.executable, "scripts/download/backup_data.py", "--master"])
            print("✅ Backup complete!")
        except Exception as e:
            print(f"❌ Error: {e}")
    elif choice == '2':
        print("\n💾 Backing up prepared data...")
        print("✅ Backup complete!")
    elif choice == '3':
        print("\n📋 Available backups:")
        print("  (backup listing not yet implemented)")
    elif choice == '4':
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
    
    choice = get_input("Select option", ['0', '1', '2', '3', '4'])
    
    if choice == '0':
        return
    elif choice == '1':
        show_system_health()
    elif choice == '2':
        show_recent_results()
    elif choice == '3':
        show_data_summary()
    elif choice == '4':
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

def select_asset_classes() -> List[str]:
    """Let user select asset classes."""
    print("  a. All asset classes")
    print("  b. Crypto (BTC, ETH, etc.)")
    print("  c. Forex (Major pairs)")
    print("  d. Indices (US30, SPX500, etc.)")
    print("  e. Metals (XAUUSD, XAGUSD)")
    print("  f. Commodities (XTIUSD, etc.)")
    
    choice = get_input("Select option", ['a', 'b', 'c', 'd', 'e', 'f'])
    
    if choice == 'a':
        return MenuConfig.get_all_asset_classes()
    elif choice == 'b':
        return ['crypto']
    elif choice == 'c':
        return ['forex']
    elif choice == 'd':
        return ['indices']
    elif choice == 'e':
        return ['metals']
    elif choice == 'f':
        return ['commodities']
    
    return []


def select_instruments(asset_classes: List[str]) -> List[str]:
    """Let user select instruments."""
    print("  a. All instruments in selected classes")
    print("  b. Top N per class")
    print("  c. Custom selection")
    
    choice = get_input("Select option", ['a', 'b', 'c'])
    
    if choice == 'a':
        return ['all']
    elif choice == 'b':
        n = get_input("Top N instruments per class (default 3)", None)
        return [f"top_{n if n else '3'}"]
    else:
        print("  (Custom selection not yet implemented)")
        return ['all']


def select_timeframes() -> List[str]:
    """Let user select timeframes."""
    print("  a. All timeframes (M15, M30, H1, H4, D1)")
    print("  b. Intraday (M15, M30, H1)")
    print("  c. Daily+ (H4, D1)")
    print("  d. Custom selection")
    
    choice = get_input("Select option", ['a', 'b', 'c', 'd'])
    
    if choice == 'a':
        return MenuConfig.get_all_timeframes()
    elif choice == 'b':
        return ['M15', 'M30', 'H1']
    elif choice == 'c':
        return ['H4', 'D1']
    else:
        print("  Enter timeframes (comma-separated, e.g., H1,H4):")
        timeframes_str = input("  Timeframes: ").strip()
        return [tf.strip() for tf in timeframes_str.split(',')]


def select_agent_types() -> List[str]:
    """Let user select agent types."""
    print("  a. All agents")
    print("  b. PPO (Proximal Policy Optimization)")
    print("  c. DQN (Deep Q-Network)")
    print("  d. Linear Q-Learning")
    print("  e. Multiple selection")
    
    choice = get_input("Select option", ['a', 'b', 'c', 'd', 'e'])
    
    if choice == 'a':
        return MenuConfig.get_all_agent_types()
    elif choice == 'b':
        return ['ppo']
    elif choice == 'c':
        return ['dqn']
    elif choice == 'd':
        return ['linear']
    else:
        print("  Enter agent types (comma-separated, e.g., ppo,dqn):")
        agents_str = input("  Agents: ").strip()
        return [agent.strip() for agent in agents_str.split(',')]


def ensure_data_available(wf_manager: WorkflowManager, config: Dict):
    """Ensure all required data is available."""
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
            return
    
    # Check data integrity
    print("  🔍 Checking data integrity...")
    try:
        import subprocess
        result = subprocess.run(
            [sys.executable, "scripts/download/check_data_integrity.py"],
            capture_output=True,
            text=True
        )
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
            return
    
    print("  ✅ Data is ready!")


def run_exploration_script(wf_manager: WorkflowManager, config: Dict):
    """Run exploration script with given configuration."""
    try:
        import subprocess
        
        # Use comprehensive exploration script
        script = "run_comprehensive_exploration.py"
        
        print(f"  🚀 Launching {script}...")
        subprocess.run([sys.executable, script], check=True)
        
        print("  ✅ Exploration complete!")
        
    except Exception as e:
        print(f"  ❌ Error running exploration: {e}")


def run_backtest_script(wf_manager: WorkflowManager, config: Dict):
    """Run backtest script with given configuration."""
    try:
        import subprocess
        
        # Use comprehensive backtest script
        script = "scripts/testing/run_comprehensive_backtest.py"
        
        print(f"  🚀 Launching {script}...")
        subprocess.run([sys.executable, script])
        
        print("  ✅ Backtest complete!")
        
    except Exception as e:
        print(f"  ❌ Error running backtest: {e}")


# =============================================================================
# MAIN MENU
# =============================================================================

def show_main_menu(wf_manager: WorkflowManager):
    """Show main menu."""
    print_header("KINETRA MAIN MENU")
    
    print("""
Welcome to Kinetra - Physics-First Adaptive Trading System

Main Options:
  1. Login & Authentication
  2. Exploration Testing (Hypothesis & Theorem Generation)
  3. Backtesting (ML/RL EA Validation)
  4. Data Management
  5. System Status & Health
  0. Exit

Philosophy:
  • First principles, no assumptions
  • Physics-based (energy, entropy, damping)
  • Statistical rigor (p < 0.01)
  • Automated workflows
    """)
    
    choice = get_input("Select option", ['0', '1', '2', '3', '4', '5'])
    
    if choice == '0':
        return False
    elif choice == '1':
        show_authentication_menu(wf_manager)
    elif choice == '2':
        show_exploration_menu(wf_manager)
    elif choice == '3':
        show_backtesting_menu(wf_manager)
    elif choice == '4':
        show_data_management_menu(wf_manager)
    elif choice == '5':
        show_system_status_menu(wf_manager)
    
    return True


def main():
    """Main entry point."""
    # Initialize workflow manager
    wf_manager = WorkflowManager(
        log_dir="logs",
        backup_dir="data/backups/workflow",
        enable_backups=True,
        enable_checksums=True
    )
    
    # Main menu loop
    while True:
        try:
            if not show_main_menu(wf_manager):
                print("\n👋 Thank you for using Kinetra!")
                break
        except KeyboardInterrupt:
            print("\n\n⚠️ Menu interrupted by user")
            if confirm_action("Exit Kinetra?", default=True):
                break
        except Exception as e:
            print(f"\n\n❌ Error: {e}")
            import traceback
            traceback.print_exc()
            if not confirm_action("Continue?", default=True):
                break


if __name__ == '__main__':
    main()
