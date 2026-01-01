#!/usr/bin/env python3
"""
Test Live Testing Menu Integration
===================================

Verifies that the live testing menu is properly integrated
and all components are accessible.
"""

import sys
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_menu_imports():
    """Test that menu can be imported."""
    try:
        import kinetra_menu
        print("✅ Menu imports successfully")
        return True
    except Exception as e:
        print(f"❌ Menu import failed: {e}")
        return False


def test_live_script_imports():
    """Test that live testing script can be imported."""
    try:
        # Don't actually import (missing dependencies in test env)
        # Just check file exists and is valid Python
        script_path = Path("scripts/testing/run_live_test.py")
        if not script_path.exists():
            print("❌ Live testing script not found")
            return False
        
        # Check it's valid Python
        import ast
        with open(script_path, 'r') as f:
            code = f.read()
        ast.parse(code)
        
        print("✅ Live testing script is valid")
        return True
    except Exception as e:
        print(f"❌ Live testing script validation failed: {e}")
        return False


def test_menu_structure():
    """Test menu structure and functions."""
    try:
        import kinetra_menu
        
        # Check key functions exist
        required_functions = [
            'show_main_menu',
            'show_live_testing_menu',
            'run_virtual_trading',
            'run_demo_account_testing',
            'test_mt5_connection',
            'show_live_testing_guide',
        ]
        
        for func_name in required_functions:
            if not hasattr(kinetra_menu, func_name):
                print(f"❌ Missing function: {func_name}")
                return False
        
        print(f"✅ All {len(required_functions)} required functions present")
        
        # Check SystemStatus class
        if not hasattr(kinetra_menu, 'SystemStatus'):
            print("❌ Missing SystemStatus class")
            return False
        
        print("✅ SystemStatus class present")
        return True
        
    except Exception as e:
        print(f"❌ Menu structure test failed: {e}")
        return False


def test_status_checks():
    """Test status checking functions."""
    try:
        import kinetra_menu
        
        status = kinetra_menu.SystemStatus()
        
        # Test status checks (should not crash)
        data_status, data_msg = status.check_data_ready()
        mt5_status, mt5_msg = status.check_mt5_available()
        creds_status, creds_msg = status.check_credentials()
        
        print(f"✅ Status checks work:")
        print(f"   Data: {data_msg}")
        print(f"   MT5: {mt5_msg}")
        print(f"   Credentials: {creds_msg}")
        
        # Test summary
        summary = status.get_status_summary()
        print(f"✅ Status summary: {summary}")
        
        return True
        
    except Exception as e:
        print(f"❌ Status checks failed: {e}")
        return False


def test_documentation():
    """Test that documentation exists."""
    try:
        doc_path = Path("docs/LIVE_TESTING_GUIDE.md")
        if not doc_path.exists():
            print("❌ Live testing guide not found")
            return False
        
        # Check it's not empty
        with open(doc_path, 'r') as f:
            content = f.read()
        
        if len(content) < 1000:
            print("❌ Live testing guide seems incomplete")
            return False
        
        print(f"✅ Live testing guide exists ({len(content)} bytes)")
        
        # Check key sections
        required_sections = [
            "Testing Modes",
            "Safety Features",
            "Testing Progression",
            "Command Line Options",
        ]
        
        for section in required_sections:
            if section not in content:
                print(f"❌ Missing section: {section}")
                return False
        
        print(f"✅ All {len(required_sections)} required sections present")
        return True
        
    except Exception as e:
        print(f"❌ Documentation test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("="*70)
    print("  LIVE TESTING MENU - INTEGRATION TEST")
    print("="*70)
    print()
    
    tests = [
        ("Menu Imports", test_menu_imports),
        ("Live Script", test_live_script_imports),
        ("Menu Structure", test_menu_structure),
        ("Status Checks", test_status_checks),
        ("Documentation", test_documentation),
    ]
    
    results = []
    for name, test_func in tests:
        print(f"\nTesting: {name}")
        print("-"*70)
        result = test_func()
        results.append(result)
        print()
    
    # Summary
    print("="*70)
    print("  TEST SUMMARY")
    print("="*70)
    passed = sum(results)
    total = len(results)
    
    for (name, _), result in zip(tests, results):
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {name}")
    
    print()
    print(f"Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n✅ All tests passed! Live testing integration is ready.")
        return 0
    else:
        print(f"\n❌ {total - passed} test(s) failed.")
        return 1


if __name__ == '__main__':
    sys.exit(main())
