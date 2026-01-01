#!/usr/bin/env python3
"""
Test Kinetra Menu System
========================

Basic validation tests for menu system and E2E framework.
"""

import sys
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_menu_imports():
    """Test that menu system can be imported."""
    print("Testing menu imports...")
    
    try:
        import kinetra_menu
        print("✅ Menu imports successfully")
        return True
    except Exception as e:
        print(f"❌ Menu import failed: {e}")
        return False


def test_e2e_imports():
    """Test that E2E framework can be imported."""
    print("\nTesting E2E framework imports...")
    
    try:
        import e2e_testing_framework
        print("✅ E2E framework imports successfully")
        return True
    except Exception as e:
        print(f"❌ E2E framework import failed: {e}")
        return False


def test_menu_config():
    """Test menu configuration."""
    print("\nTesting menu configuration...")
    
    try:
        from kinetra_menu import MenuConfig
        
        # Test asset classes
        asset_classes = MenuConfig.get_all_asset_classes()
        assert len(asset_classes) > 0, "No asset classes defined"
        print(f"  Asset classes: {', '.join(asset_classes)}")
        
        # Test timeframes
        timeframes = MenuConfig.get_all_timeframes()
        assert len(timeframes) > 0, "No timeframes defined"
        print(f"  Timeframes: {', '.join(timeframes)}")
        
        # Test agent types
        agent_types = MenuConfig.get_all_agent_types()
        assert len(agent_types) > 0, "No agent types defined"
        print(f"  Agent types: {', '.join(agent_types)}")
        
        print("✅ Menu configuration valid")
        return True
        
    except Exception as e:
        print(f"❌ Menu configuration failed: {e}")
        return False


def test_e2e_presets():
    """Test E2E preset configurations."""
    print("\nTesting E2E presets...")
    
    try:
        from e2e_testing_framework import E2EPresets
        
        # Test quick validation preset
        quick = E2EPresets.quick_validation()
        assert quick.name == "quick_validation"
        assert len(quick.asset_classes) > 0
        assert len(quick.timeframes) > 0
        print(f"  Quick validation: {len(quick.asset_classes)} asset classes, {len(quick.timeframes)} timeframes")
        
        # Test asset class preset
        crypto_test = E2EPresets.asset_class_test('crypto')
        assert crypto_test.name == "crypto_test"
        assert 'crypto' in crypto_test.asset_classes
        print(f"  Crypto test: {crypto_test.description}")
        
        # Test agent type preset
        ppo_test = E2EPresets.agent_type_test('ppo')
        assert ppo_test.name == "ppo_agent_test"
        assert 'ppo' in ppo_test.agent_types
        print(f"  PPO test: {ppo_test.description}")
        
        # Test full system test
        full_test = E2EPresets.full_system_test()
        assert full_test.name == "full_system_test"
        assert len(full_test.asset_classes) >= 5
        print(f"  Full test: {len(full_test.asset_classes)} asset classes")
        
        print("✅ E2E presets valid")
        return True
        
    except Exception as e:
        print(f"❌ E2E presets failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_instrument_registry():
    """Test instrument registry."""
    print("\nTesting instrument registry...")
    
    try:
        from e2e_testing_framework import InstrumentRegistry
        
        # Test crypto instruments
        crypto = InstrumentRegistry.get_instruments('crypto')
        assert len(crypto) > 0, "No crypto instruments"
        print(f"  Crypto instruments: {len(crypto)}")
        
        # Test forex instruments
        forex = InstrumentRegistry.get_instruments('forex')
        assert len(forex) > 0, "No forex instruments"
        print(f"  Forex instruments: {len(forex)}")
        
        # Test all instruments
        all_instruments = InstrumentRegistry.get_all_instruments()
        assert len(all_instruments) > 0, "No instruments"
        print(f"  Total instruments: {len(all_instruments)}")
        
        # Test top instruments
        top_3 = InstrumentRegistry.get_top_instruments('crypto', 3)
        assert len(top_3) == 3, "Top 3 failed"
        print(f"  Top 3 crypto: {', '.join(top_3)}")
        
        print("✅ Instrument registry valid")
        return True
        
    except Exception as e:
        print(f"❌ Instrument registry failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_e2e_test_matrix():
    """Test E2E test matrix generation."""
    print("\nTesting E2E test matrix generation...")
    
    try:
        from e2e_testing_framework import E2ETestRunner, E2EPresets
        
        # Create quick validation test
        config = E2EPresets.quick_validation()
        runner = E2ETestRunner(config)
        
        # Generate test matrix
        test_matrix = runner.generate_test_matrix()
        
        assert len(test_matrix) > 0, "Empty test matrix"
        print(f"  Test matrix size: {len(test_matrix)} combinations")
        
        # Check test structure
        first_test = test_matrix[0]
        assert 'asset_class' in first_test
        assert 'instrument' in first_test
        assert 'timeframe' in first_test
        assert 'agent_type' in first_test
        assert 'test_id' in first_test
        print(f"  First test: {first_test['test_id']}")
        
        # Estimate duration
        duration_hours, duration_str = runner.estimate_duration()
        print(f"  Estimated duration: {duration_str}")
        
        print("✅ Test matrix generation valid")
        return True
        
    except Exception as e:
        print(f"❌ Test matrix generation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("=" * 80)
    print("  Kinetra Menu System Tests")
    print("=" * 80)
    
    results = []
    
    # Run tests
    results.append(("Menu Imports", test_menu_imports()))
    results.append(("E2E Imports", test_e2e_imports()))
    results.append(("Menu Config", test_menu_config()))
    results.append(("E2E Presets", test_e2e_presets()))
    results.append(("Instrument Registry", test_instrument_registry()))
    results.append(("Test Matrix Generation", test_e2e_test_matrix()))
    
    # Summary
    print("\n" + "=" * 80)
    print("  Test Summary")
    print("=" * 80)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {test_name}")
    
    print("\n" + "-" * 80)
    print(f"Results: {passed}/{total} tests passed")
    print("=" * 80)
    
    return 0 if passed == total else 1


if __name__ == '__main__':
    sys.exit(main())
