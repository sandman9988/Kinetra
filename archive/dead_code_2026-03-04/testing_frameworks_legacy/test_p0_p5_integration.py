#!/usr/bin/env python3
"""
Integration Test for P0-P5 Components
======================================

Tests that all integrated components work together:
- P0: DSP-SuperPot feature extraction
- P1: Agent Factory
- P2: Unified Trading Environment
- P3: Discovery Methods (Chaos)
- P4: Results Analyzer
- P5: Unified Training CLI (tested via imports)

Run this to validate Phase 1 integration is complete.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_p0_dsp_superpot():
    """Test P0: DSP-SuperPot feature extraction."""
    print("\n" + "="*80)
    print("P0: DSP-SuperPot Feature Extraction")
    print("="*80)
    
    from kinetra.superpot_dsp import DSPSuperPotExtractor, validate_no_fixed_periods
    
    # Create extractor
    extractor = DSPSuperPotExtractor()
    print(f"✅ Created DSP SuperPot Extractor")
    print(f"   Features: {extractor.n_features}")
    print(f"   Sample features: {extractor.feature_names[:5]}")
    
    # Validate no fixed periods
    is_valid = validate_no_fixed_periods(extractor)
    assert is_valid, "FAILED: Fixed periods found in feature names!"
    print(f"✅ Validation: NO fixed periods in feature names")
    
    # Test extraction
    data = pd.DataFrame({
        'close': 100 + np.cumsum(np.random.randn(500) * 0.5),
        'high': 101 + np.cumsum(np.random.randn(500) * 0.5),
        'low': 99 + np.cumsum(np.random.randn(500) * 0.5),
        'volume': np.random.randint(1000, 5000, 500)
    })
    
    features = extractor.extract(data, idx=200)
    print(f"✅ Feature extraction works")
    print(f"   Feature vector shape: {features.shape}")
    print(f"   Sample values: {features[:5]}")
    
    return True


def test_p1_agent_factory():
    """Test P1: Agent Factory."""
    print("\n" + "="*80)
    print("P1: Agent Factory")
    print("="*80)
    
    from kinetra.agent_factory import AgentFactory
    
    # List available agents
    agents = AgentFactory.list_available_agents()
    print(f"✅ Available agents: {agents}")
    
    # Create PPO agent
    ppo_agent = AgentFactory.create('ppo', state_dim=32, action_dim=4)
    print(f"✅ Created PPO agent: {type(ppo_agent).__name__}")
    
    # Create DQN agent
    dqn_agent = AgentFactory.create('dqn', state_dim=32, action_dim=4)
    print(f"✅ Created DQN agent: {type(dqn_agent).__name__}")
    
    return True


def test_p2_unified_env():
    """Test P2: Unified Trading Environment."""
    print("\n" + "="*80)
    print("P2: Unified Trading Environment")
    print("="*80)
    
    from kinetra.unified_trading_env import UnifiedTradingEnv, TradingMode
    
    # Create dummy data
    data = pd.DataFrame({
        'open': 100 + np.cumsum(np.random.randn(500) * 0.5),
        'high': 101 + np.cumsum(np.random.randn(500) * 0.5),
        'low': 99 + np.cumsum(np.random.randn(500) * 0.5),
        'close': 100 + np.cumsum(np.random.randn(500) * 0.5),
        'volume': np.random.randint(1000, 5000, 500)
    })
    
    # Create environment without physics
    env = UnifiedTradingEnv(
        data=data,
        mode=TradingMode.EXPLORATION,
        use_physics=False,
        regime_filter=False
    )
    print(f"✅ Created environment (no physics)")
    print(f"   Observation dim: {env.observation_dim}")
    
    # Test reset
    state = env.reset()
    print(f"✅ Environment reset works")
    print(f"   State shape: {state.shape}")
    
    # Test step
    next_state, reward, done, info = env.step(1)  # BUY action
    print(f"✅ Environment step works")
    print(f"   Reward: {reward:.2f}")
    print(f"   Regime: {info['regime']}")
    
    return True


def test_p3_discovery_methods():
    """Test P3: Discovery Methods."""
    print("\n" + "="*80)
    print("P3: Discovery Methods (Chaos Theory)")
    print("="*80)
    
    from kinetra.discovery_methods import ChaosTheoryDiscovery
    
    # Create analyzer
    analyzer = ChaosTheoryDiscovery()
    print(f"✅ Created ChaosTheoryDiscovery analyzer")
    
    # Create test data
    data = pd.DataFrame({
        'close': 100 + np.cumsum(np.random.randn(500) * 0.5)
    })
    
    # Run discovery
    result = analyzer.discover(data, config={})
    print(f"✅ Discovery completed")
    print(f"   Patterns found: {len(result.discovered_patterns)}")
    print(f"   Statistical significance: {result.statistical_significance}")
    print(f"   p-value: {result.p_value:.4f}")
    
    # Check patterns
    for pattern in result.discovered_patterns:
        print(f"   - {pattern['metric']}: {pattern['value']:.4f}")
    
    return True


def test_p4_results_analyzer():
    """Test P4: Results Analyzer."""
    print("\n" + "="*80)
    print("P4: Results Analyzer")
    print("="*80)
    
    from kinetra.results_analyzer import ResultsAnalyzer
    
    # Create analyzer
    analyzer = ResultsAnalyzer(results_dir="test_results_tmp")
    print(f"✅ Created ResultsAnalyzer")
    print(f"   Results dir: {analyzer.results_dir}")
    
    # Test comparison (with dummy data)
    comparison = analyzer.compare_suites(['control', 'physics', 'rl'])
    print(f"✅ Suite comparison works")
    print(f"   Suites compared: {len(comparison)}")
    
    # Test winner identification
    winner = analyzer.identify_winner(comparison)
    print(f"✅ Winner identification works")
    print(f"   {winner['message']}")
    
    # Clean up
    import shutil
    if analyzer.results_dir.exists():
        shutil.rmtree(analyzer.results_dir)
    
    return True


def test_p5_training_cli():
    """Test P5: Unified Training CLI (imports only)."""
    print("\n" + "="*80)
    print("P5: Unified Training CLI")
    print("="*80)
    
    # Test imports
    import scripts.train as train_module
    print(f"✅ train.py imports successfully")
    
    # Check key functions exist
    assert hasattr(train_module, 'parse_args'), "Missing parse_args"
    assert hasattr(train_module, 'train_agent'), "Missing train_agent"
    assert hasattr(train_module, 'load_data'), "Missing load_data"
    print(f"✅ All required functions present")
    
    return True


def run_all_tests():
    """Run all integration tests."""
    print("\n" + "#"*80)
    print("# P0-P5 INTEGRATION TEST SUITE")
    print("#"*80)
    
    tests = [
        ("P0: DSP-SuperPot", test_p0_dsp_superpot),
        ("P1: Agent Factory", test_p1_agent_factory),
        ("P2: Unified Trading Environment", test_p2_unified_env),
        ("P3: Discovery Methods", test_p3_discovery_methods),
        ("P4: Results Analyzer", test_p4_results_analyzer),
        ("P5: Unified Training CLI", test_p5_training_cli),
    ]
    
    results = {}
    for name, test_func in tests:
        try:
            success = test_func()
            results[name] = "✅ PASS"
        except Exception as e:
            print(f"\n❌ FAILED: {e}")
            import traceback
            traceback.print_exc()
            results[name] = f"❌ FAIL: {str(e)}"
    
    # Summary
    print("\n" + "#"*80)
    print("# TEST SUMMARY")
    print("#"*80)
    
    all_passed = True
    for name, result in results.items():
        print(f"{result} - {name}")
        if "FAIL" in result:
            all_passed = False
    
    print("\n" + "#"*80)
    if all_passed:
        print("# ✅ ALL TESTS PASSED - Phase 1 Integration Complete!")
    else:
        print("# ❌ SOME TESTS FAILED - Review errors above")
    print("#"*80 + "\n")
    
    return all_passed


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
