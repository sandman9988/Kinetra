#!/usr/bin/env python3
"""
Testing Menu
============

Main testing interface offering:
1. Exploration - First principles discovery
2. Optimization - Replay learning
3. Backtesting - Validate strategies

Usage:
    python scripts/test_menu.py
"""

import sys
import subprocess
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))


def print_header(text: str):
    """Print section header."""
    print("\n" + "=" * 80)
    print(f"  {text}")
    print("=" * 80)


def check_data_prepared() -> bool:
    """Check if data has been prepared."""
    prepared_dir = Path("data/prepared")
    train_dir = prepared_dir / "train"
    test_dir = prepared_dir / "test"

    if not train_dir.exists() or not test_dir.exists():
        return False

    train_files = list(train_dir.glob("*.csv"))
    test_files = list(test_dir.glob("*.csv"))

    return len(train_files) > 0 and len(test_files) > 0


def show_exploration_menu():
    """Show exploration testing options."""
    print_header("EXPLORATION - FIRST PRINCIPLES")

    print("""
Exploration discovers what works where by MEASURING, not ASSUMING.

Starting point:
  • ONE universal agent on ALL data
  • Physics-based features (energy, entropy, damping, regime)
  • Track performance by:
    - Asset class (forex, crypto, indices, metals, commodities)
    - Regime (overdamped, underdamped, laminar, breakout)
    - Timeframe (M15, M30, H1, H4)
    - Volatility (low, medium, high)

What we discover:
  1. Which measurements impact which symbol classes
  2. Which measurement stacking gives what result per class/symbol
  3. Which agent policy should be used per class
  4. Risk management parameters

THE MARKET TELLS US, WE DON'T ASSUME!

Available exploration tests:
  1. Universal Agent Baseline - Train ONE agent on ALL instruments
  2. Measurement Impact - What features matter where
  3. Stacking Analysis - Feature combinations per class
  4. Policy Discovery - What agent type per class
  5. Risk Management - What risk params per class/regime
  6. Full Exploration - Run all above
""")

    choice = input("Select exploration test [1-6, or 0 to go back]: ").strip()

    if choice == '1':
        subprocess.run([sys.executable, "scripts/explore_universal.py"])
    elif choice == '2':
        subprocess.run([sys.executable, "scripts/explore_measurements.py"])
    elif choice == '3':
        subprocess.run([sys.executable, "scripts/explore_stacking.py"])
    elif choice == '4':
        subprocess.run([sys.executable, "scripts/explore_policies.py"])
    elif choice == '5':
        subprocess.run([sys.executable, "scripts/explore_risk.py"])
    elif choice == '6':
        subprocess.run([sys.executable, "scripts/explore_full.py"])


def show_optimization_menu():
    """Show optimization testing options."""
    print_header("OPTIMIZATION - REPLAY LEARNING")

    print("""
Optimization uses replay learning to improve agent performance.

Based on exploration results, we:
  • Replay successful episodes
  • Learn from failures
  • Optimize parameters per discovered specialization
  • Fine-tune risk management

Available optimization tests:
  1. Experience Replay - Learn from best episodes
  2. Parameter Tuning - Optimize hyperparameters
  3. Risk Optimization - Fine-tune risk params
  4. Full Optimization - Run all above
""")

    choice = input("Select optimization test [1-4, or 0 to go back]: ").strip()

    if choice == '1':
        subprocess.run([sys.executable, "scripts/optimize_replay.py"])
    elif choice == '2':
        subprocess.run([sys.executable, "scripts/optimize_params.py"])
    elif choice == '3':
        subprocess.run([sys.executable, "scripts/optimize_risk.py"])
    elif choice == '4':
        subprocess.run([sys.executable, "scripts/optimize_full.py"])


def show_backtest_menu():
    """Show backtesting options."""
    print_header("BACKTESTING - VALIDATE STRATEGIES")

    print("""
Backtesting validates discovered strategies on held-out test data.

Uses:
  • MT5-accurate friction modeling
  • Realistic slippage and spread
  • Proper margin requirements
  • Currency conversion
  • Swap/rollover costs

Available backtest modes:
  1. Universal Agent - Test baseline on test set
  2. Specialized Agents - Test discovered specialists
  3. Compare Strategies - Universal vs Specialists
  4. Risk Analysis - Drawdown, CVaR, Sharpe
  5. Full Backtest - Complete validation
""")

    choice = input("Select backtest mode [1-5, or 0 to go back]: ").strip()

    if choice == '1':
        subprocess.run([sys.executable, "scripts/backtest_universal.py"])
    elif choice == '2':
        subprocess.run([sys.executable, "scripts/backtest_specialists.py"])
    elif choice == '3':
        subprocess.run([sys.executable, "scripts/backtest_compare.py"])
    elif choice == '4':
        subprocess.run([sys.executable, "scripts/backtest_risk.py"])
    elif choice == '5':
        subprocess.run([sys.executable, "scripts/backtest_full.py"])


def main():
    """Main testing menu."""
    print_header("KINETRA TESTING MENU")

    # Check if data is prepared
    if not check_data_prepared():
        print("\n❌ Data not prepared!")
        print("\nYou must prepare data first:")
        print("  1. python scripts/download_interactive.py    # Download data")
        print("  2. python scripts/check_data_integrity.py   # Check integrity")
        print("  3. python scripts/prepare_data.py           # Prepare data")
        print("\nThen run this menu again.")
        return

    print("\n✅ Data is prepared and ready")

    while True:
        print_header("SELECT TESTING APPROACH")

        print("""
1. EXPLORATION - First Principles Discovery
   • Start with universal agent
   • Measure what works where
   • Discover specialization needs
   • Let the market tell us

2. OPTIMIZATION - Replay Learning
   • Replay successful episodes
   • Optimize parameters
   • Fine-tune risk management
   • Based on exploration results

3. BACKTESTING - Validate Strategies
   • Test on held-out data
   • MT5-accurate friction
   • Risk analysis
   • Compare approaches

0. Exit
""")

        choice = input("Select testing approach [0-3]: ").strip()

        if choice == '0':
            print("\n👋 Goodbye!")
            break
        elif choice == '1':
            show_exploration_menu()
        elif choice == '2':
            show_optimization_menu()
        elif choice == '3':
            show_backtest_menu()
        else:
            print("\n❌ Invalid choice")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Menu interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
