#!/usr/bin/env python3
"""
Example: How to Use the Testing Framework Programmatically

This script demonstrates how to use the Kinetra Testing Framework
in your own code to run custom experiments.
"""

import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from kinetra.testing_framework import (
    TestingFramework,
    TestConfiguration,
    InstrumentSpec,
    StatisticalValidator,
    EfficiencyMetrics,
)


def example_minimal():
    """Minimal example: Test control vs physics on a few instruments."""
    print("\n" + "="*80)
    print("EXAMPLE 1: Minimal Test")
    print("="*80 + "\n")
    
    # Create framework
    framework = TestingFramework(output_dir="test_results/example_minimal")
    
    # Define instruments manually
    instruments = [
        InstrumentSpec(
            symbol="BTCUSD",
            asset_class="crypto",
            timeframe="H1",
            data_path="data/master/BTCUSD_H1.csv"
        ),
        InstrumentSpec(
            symbol="EURUSD",
            asset_class="forex",
            timeframe="H1",
            data_path="data/master/EURUSD_H1.csv"
        ),
    ]
    
    # Create control test
    control_test = TestConfiguration(
        name="control_baseline",
        description="Baseline using standard indicators",
        instruments=instruments,
        agent_type="control",
        agent_config={
            "indicators": ["SMA_20", "RSI_14"],
        },
        episodes=10,  # Quick test
        use_gpu=False,
    )
    
    # Create physics test
    physics_test = TestConfiguration(
        name="physics_energy",
        description="Physics-based using energy and damping",
        instruments=instruments,
        agent_type="physics",
        agent_config={
            "features": ["energy", "damping", "entropy"],
            "adaptive": True,
        },
        episodes=10,
        use_gpu=True,
    )
    
    # Add tests
    framework.add_test(control_test)
    framework.add_test(physics_test)
    
    # Run
    results = framework.run_all_tests()
    
    # Save and report
    framework.save_results()
    framework.generate_report()
    
    # Compare
    comparison = framework.compare_tests(
        ["control_baseline", "physics_energy"],
        metric="sharpe_ratio"
    )
    
    print(f"\nWinner: {comparison.winner}")
    print(f"Statistical significance: {comparison.statistical_significance}")
    print(f"Effect sizes: {comparison.effect_sizes}")


def example_comprehensive():
    """Comprehensive example: Multiple test suites with discovery."""
    print("\n" + "="*80)
    print("EXAMPLE 2: Comprehensive Test with Discovery")
    print("="*80 + "\n")
    
    # Import suite creators
    from scripts.unified_test_framework import (
        discover_instruments,
        create_control_suite,
        create_physics_suite,
        create_hidden_dimension_suite,
        create_chaos_theory_suite,
    )
    
    # Discover instruments
    instruments = discover_instruments(
        asset_classes=["crypto", "forex"],
        timeframes=["H1"],
        max_per_class=2
    )
    
    if not instruments:
        print("No instruments found - skipping")
        return
    
    print(f"Found {len(instruments)} instruments")
    
    # Create framework
    framework = TestingFramework(output_dir="test_results/example_comprehensive")
    
    # Add core tests
    framework.add_test(create_control_suite(instruments))
    framework.add_test(create_physics_suite(instruments))
    
    # Add discovery tests
    framework.add_test(create_hidden_dimension_suite(instruments))
    framework.add_test(create_chaos_theory_suite(instruments))
    
    # Reduce episodes for example
    for test in framework.tests:
        test.episodes = 20
    
    # Run
    print("\nRunning tests...")
    results = framework.run_all_tests()
    
    # Save
    framework.save_results()
    
    # Generate report
    framework.generate_report()
    
    # Filter for significant results only
    significant = StatisticalValidator.filter_significant_results(
        results,
        metric_name='sharpe_ratio',
        alpha=0.05
    )
    
    print(f"\nTotal results: {len(results)}")
    print(f"Statistically significant: {len(significant)}")


def example_custom_metrics():
    """Example: Custom efficiency metrics analysis."""
    print("\n" + "="*80)
    print("EXAMPLE 3: Custom Efficiency Metrics")
    print("="*80 + "\n")
    
    # Simulate some trade history
    trade_history = [
        {"mfe": 100, "mae": -20, "pnl": 80},
        {"mfe": 150, "mae": -30, "pnl": 120},
        {"mfe": 80, "mae": -50, "pnl": -30},
        {"mfe": 200, "mae": -15, "pnl": 180},
    ]
    
    # Calculate efficiency
    mfe_captured, mae_ratio = EfficiencyMetrics.calculate_mfe_mae(trade_history)
    
    print(f"MFE Captured: {mfe_captured:.2f}%")
    print(f"MAE Ratio: {mae_ratio:.3f}")
    
    # Simulate equity curve
    import numpy as np
    equity_curve = np.array([100, 110, 105, 130, 125, 150, 145, 170])
    
    pyth_efficiency = EfficiencyMetrics.calculate_pythagorean_efficiency(equity_curve)
    
    print(f"Pythagorean Efficiency: {pyth_efficiency:.3f}")
    print("\nInterpretation:")
    print("- MFE Captured > 60% is good (we're capturing the potential)")
    print("- MAE Ratio < 0.5 is good (losses are small relative to wins)")
    print("- Pythagorean Efficiency closer to 1.0 is better (straight path to profit)")


def example_statistical_validation():
    """Example: Statistical validation and significance testing."""
    print("\n" + "="*80)
    print("EXAMPLE 4: Statistical Validation")
    print("="*80 + "\n")
    
    import numpy as np
    
    # Simulate results from two strategies
    strategy_a_returns = np.random.normal(0.05, 0.1, 50)  # Mean 5%, std 10%
    strategy_b_returns = np.random.normal(0.03, 0.1, 50)  # Mean 3%, std 10%
    
    # Test significance
    validator = StatisticalValidator()
    
    is_sig, p_value = validator.test_significance(
        strategy_a_returns,
        strategy_b_returns,
        alpha=0.05,
        correction='bonferroni',
        n_tests=1
    )
    
    print(f"Strategy A mean: {np.mean(strategy_a_returns):.4f}")
    print(f"Strategy B mean: {np.mean(strategy_b_returns):.4f}")
    print(f"P-value: {p_value:.4f}")
    print(f"Statistically significant: {is_sig}")
    
    # Calculate effect size
    effect_size = validator.calculate_effect_size(strategy_a_returns, strategy_b_returns)
    
    print(f"Cohen's d effect size: {effect_size:.3f}")
    
    if effect_size < 0.2:
        print("  → Small effect")
    elif effect_size < 0.5:
        print("  → Medium effect")
    else:
        print("  → Large effect")


def main():
    """Run all examples."""
    print("\n" + "="*80)
    print("KINETRA TESTING FRAMEWORK - EXAMPLES")
    print("="*80)
    
    try:
        example_minimal()
    except Exception as e:
        print(f"Example 1 error (expected if no data): {e}")
    
    try:
        example_comprehensive()
    except Exception as e:
        print(f"Example 2 error (expected if no data): {e}")
    
    example_custom_metrics()
    
    example_statistical_validation()
    
    print("\n" + "="*80)
    print("EXAMPLES COMPLETE")
    print("="*80 + "\n")
    
    print("Next steps:")
    print("1. Ensure you have data in data/master/")
    print("2. Run: python scripts/unified_test_framework.py --quick")
    print("3. Explore: python scripts/unified_test_framework.py --suite hidden")
    print("4. Full run: python scripts/unified_test_framework.py --extreme")


if __name__ == "__main__":
    main()
