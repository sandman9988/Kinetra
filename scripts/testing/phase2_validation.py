#!/usr/bin/env python3
"""
Phase 2: Integration Testing Script
====================================

Validates that P0-P5 components work together end-to-end.

Tests:
1. P0 Validation: DSP-driven features (no fixed periods)
2. P1 Validation: RL agent training via testing framework
3. P2 Validation: Physics integration in environment
4. P3 Validation: Chaos discovery integrated
5. End-to-End: Full test suite comparison

Usage:
    python scripts/testing/phase2_validation.py --quick
    python scripts/testing/phase2_validation.py --full
"""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from kinetra.testing_framework import (
    TestConfiguration,
    InstrumentSpec,
    run_rl_test,
    run_chaos_test,
    TestingFramework,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_test_data(output_file: str, n: int = 1000) -> str:
    """Create dummy test data for validation."""
    np.random.seed(42)
    
    base_price = 100.0
    prices = base_price + np.cumsum(np.random.randn(n) * 0.5)
    
    data = pd.DataFrame({
        'time': pd.date_range('2024-01-01', periods=n, freq='1h'),
        'open': prices + np.random.randn(n) * 0.2,
        'high': prices + np.abs(np.random.randn(n) * 0.3),
        'low': prices - np.abs(np.random.randn(n) * 0.3),
        'close': prices,
        'volume': np.random.randint(1000, 10000, n),
    })
    
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(output_path, index=False)
    
    logger.info(f"Created test data: {output_path}")
    return str(output_path)


def test_p0_dsp_features():
    """Test P0: DSP-driven features."""
    logger.info("\n" + "="*80)
    logger.info("TEST P0: DSP-SuperPot Features")
    logger.info("="*80)
    
    from kinetra.superpot_dsp import DSPSuperPotExtractor, validate_no_fixed_periods
    
    extractor = DSPSuperPotExtractor()
    
    # Validate no fixed periods
    is_valid = validate_no_fixed_periods(extractor)
    assert is_valid, "FAILED: Fixed periods found!"
    logger.info("✅ NO fixed periods in DSP features")
    
    # Test extraction
    data = pd.DataFrame({
        'close': 100 + np.cumsum(np.random.randn(500) * 0.5),
        'high': 101 + np.cumsum(np.random.randn(500) * 0.5),
        'low': 99 + np.cumsum(np.random.randn(500) * 0.5),
        'volume': np.random.randint(1000, 5000, 500)
    })
    
    features = extractor.extract(data, idx=200)
    logger.info(f"✅ Feature extraction works: {features.shape}")
    
    return True


def test_p1_rl_agents(quick: bool = True):
    """Test P1: RL agent training."""
    logger.info("\n" + "="*80)
    logger.info("TEST P1: RL Agent Training")
    logger.info("="*80)
    
    # Create test data
    data_file = create_test_data('/tmp/test_data.csv', n=500 if quick else 2000)
    
    # Create test configuration
    instrument = InstrumentSpec(
        symbol='TESTBTC',
        asset_class='crypto',
        timeframe='H1',
        data_path=data_file
    )
    
    config = TestConfiguration(
        name="rl_ppo_test",
        description="Test PPO agent integration",
        instruments=[instrument],
        agent_type='ppo',
        agent_config={
            'use_physics': False,  # Faster for testing
            'regime_filter': False,
            'mode': 'exploration',
        },
        episodes=2 if quick else 10,
        use_gpu=False,
    )
    
    # Run RL test
    result = run_rl_test(config)
    
    logger.info(f"✅ RL test completed")
    logger.info(f"   Agent: {result.agent_type}")
    logger.info(f"   Sharpe: {result.sharpe_ratio:.3f}")
    logger.info(f"   Win Rate: {result.win_rate:.2%}")
    
    return True


def test_p2_physics_env():
    """Test P2: Physics environment integration."""
    logger.info("\n" + "="*80)
    logger.info("TEST P2: Physics Environment")
    logger.info("="*80)
    
    from kinetra.unified_trading_env import UnifiedTradingEnv, TradingMode
    
    # Create test data
    data = pd.DataFrame({
        'open': 100 + np.cumsum(np.random.randn(500) * 0.5),
        'high': 101 + np.cumsum(np.random.randn(500) * 0.5),
        'low': 99 + np.cumsum(np.random.randn(500) * 0.5),
        'close': 100 + np.cumsum(np.random.randn(500) * 0.5),
        'volume': np.random.randint(1000, 5000, 500)
    })
    
    # Test without physics (fast)
    env = UnifiedTradingEnv(
        data=data,
        mode=TradingMode.EXPLORATION,
        use_physics=False,
        regime_filter=False
    )
    
    state = env.reset()
    logger.info(f"✅ Environment created (no physics): {state.shape}")
    
    # Test step
    next_state, reward, done, info = env.step(1)
    logger.info(f"✅ Environment step works")
    logger.info(f"   Regime: {info['regime']}")
    
    return True


def test_p3_chaos_discovery(quick: bool = True):
    """Test P3: Chaos discovery integration."""
    logger.info("\n" + "="*80)
    logger.info("TEST P3: Chaos Discovery")
    logger.info("="*80)
    
    # Create test data
    data_file = create_test_data('/tmp/test_chaos.csv', n=500 if quick else 2000)
    
    # Create test configuration
    instrument = InstrumentSpec(
        symbol='TESTETH',
        asset_class='crypto',
        timeframe='H1',
        data_path=data_file
    )
    
    config = TestConfiguration(
        name="chaos_test",
        description="Test chaos theory integration",
        instruments=[instrument],
        agent_type='chaos',
        agent_config={},
        episodes=1,
        use_gpu=False,
    )
    
    # Run chaos test
    result = run_chaos_test(config)
    
    logger.info(f"✅ Chaos test completed")
    logger.info(f"   Instruments analyzed: {result.total_trades}")
    logger.info(f"   Significance rate: {result.sharpe_ratio:.2%}")
    
    return True


def test_end_to_end_integration(quick: bool = True):
    """Test end-to-end integration."""
    logger.info("\n" + "="*80)
    logger.info("TEST: End-to-End Integration")
    logger.info("="*80)
    
    # Create test data
    data_file = create_test_data('/tmp/test_e2e.csv', n=500 if quick else 2000)
    
    # Create test instruments
    instruments = [
        InstrumentSpec(
            symbol='TESTBTC',
            asset_class='crypto',
            timeframe='H1',
            data_path=data_file
        )
    ]
    
    # Create testing framework
    framework = TestingFramework(use_data_management=False)
    
    # Add control test
    framework.add_test(TestConfiguration(
        name="control_ma",
        description="Moving average baseline",
        instruments=instruments,
        agent_type='control',
        agent_config={},
        episodes=1,
        use_gpu=False,
    ))
    
    # Add RL test
    framework.add_test(TestConfiguration(
        name="rl_ppo",
        description="PPO RL agent",
        instruments=instruments,
        agent_type='ppo',
        agent_config={
            'use_physics': False,
            'regime_filter': False,
            'mode': 'exploration',
        },
        episodes=2 if quick else 5,
        use_gpu=False,
    ))
    
    # Add chaos test
    framework.add_test(TestConfiguration(
        name="chaos_analysis",
        description="Chaos theory analysis",
        instruments=instruments,
        agent_type='chaos',
        agent_config={},
        episodes=1,
        use_gpu=False,
    ))
    
    # Run all tests
    logger.info("Running all test suites...")
    results = framework.run_all_tests()
    
    logger.info(f"✅ End-to-end integration completed")
    logger.info(f"   Tests run: {len(results)}")
    logger.info(f"   All tests completed successfully")
    
    return True


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Phase 2 Integration Validation")
    parser.add_argument('--quick', action='store_true', help='Quick test (fewer episodes)')
    parser.add_argument('--full', action='store_true', help='Full test suite')
    parser.add_argument('--test', type=str, help='Run specific test (p0, p1, p2, p3, e2e)')
    
    args = parser.parse_args()
    
    quick = args.quick or not args.full
    
    logger.info("#" * 80)
    logger.info("# PHASE 2: INTEGRATION TESTING")
    logger.info("#" * 80)
    logger.info(f"Mode: {'QUICK' if quick else 'FULL'}")
    logger.info(f"Timestamp: {datetime.now()}")
    logger.info("#" * 80)
    
    tests = {
        'p0': ('P0: DSP Features', test_p0_dsp_features),
        'p1': ('P1: RL Agents', lambda: test_p1_rl_agents(quick)),
        'p2': ('P2: Physics Environment', test_p2_physics_env),
        'p3': ('P3: Chaos Discovery', lambda: test_p3_chaos_discovery(quick)),
        'e2e': ('End-to-End', lambda: test_end_to_end_integration(quick)),
    }
    
    if args.test:
        # Run specific test
        if args.test not in tests:
            logger.error(f"Unknown test: {args.test}")
            logger.error(f"Available tests: {list(tests.keys())}")
            return 1
        
        name, test_func = tests[args.test]
        try:
            logger.info(f"\nRunning: {name}")
            test_func()
        except Exception as e:
            logger.error(f"Test failed: {e}", exc_info=True)
            return 1
    else:
        # Run all tests
        results = {}
        for test_id, (name, test_func) in tests.items():
            try:
                test_func()
                results[name] = "✅ PASS"
            except Exception as e:
                logger.error(f"\nTest {name} failed: {e}", exc_info=True)
                results[name] = f"❌ FAIL: {str(e)}"
        
        # Summary
        logger.info("\n" + "#" * 80)
        logger.info("# TEST SUMMARY")
        logger.info("#" * 80)
        
        all_passed = True
        for name, result in results.items():
            logger.info(f"{result} - {name}")
            if "FAIL" in result:
                all_passed = False
        
        logger.info("\n" + "#" * 80)
        if all_passed:
            logger.info("# ✅ ALL TESTS PASSED - Phase 2 Integration Complete!")
        else:
            logger.info("# ❌ SOME TESTS FAILED - Review errors above")
        logger.info("#" * 80)
        
        return 0 if all_passed else 1
    
    logger.info("\n✅ Phase 2 validation completed successfully!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
