#!/usr/bin/env python3
"""
Kinetra Comprehensive Test Runner
=================================

Master test runner that executes all test suites:
- Basic menu system tests
- Complete workflow tests
- System stress tests (light/standard/heavy)
- Performance profiling and bottleneck analysis

Usage:
    python tests/run_all_tests.py
    python tests/run_all_tests.py --quick
    python tests/run_all_tests.py --full
    python tests/run_all_tests.py --with-profiling
"""

import sys
import subprocess
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))


def run_test(name: str, command: List[str]) -> Tuple[bool, str]:
    """Run a single test suite."""
    print(f"\n{'='*80}")
    print(f"Running: {name}")
    print(f"{'='*80}\n")

    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )

        success = result.returncode == 0
        output = (result.stdout or "") + (("\n" + result.stderr) if result.stderr else "")

        if success:
            print(f"✅ {name} - PASSED")
        else:
            print(f"❌ {name} - FAILED")
            print(f"Error output: {output[:500]}")  # Print first 500 chars

        return success, output

    except subprocess.TimeoutExpired:
        print(f"⏱️  {name} - TIMEOUT")
        return False, "Test timed out after 5 minutes"
    except Exception as e:
        print(f"❌ {name} - ERROR: {e}")
        return False, str(e)


class TestRunner:
    """Comprehensive test runner for Kinetra."""
    
    def __init__(self):
        self.results = []
        self.start_time = datetime.now()

    def run_basic_tests(self) -> Dict[str, bool]:
        """Run basic test suite."""
        print("\n" + "="*80)
        print("PHASE 1: Basic Tests")
        print("="*80)
        
        results = {}
        
        # Menu system tests
        test_path = Path(__file__).parent / "test_menu_system.py"
        success, output = run_test(
            "Menu System Tests",
            [sys.executable, str(test_path)]
        )
        results['menu_system'] = success
        self.results.append(('Menu System Tests', success))
        
        return results
    
    def run_workflow_tests(self) -> Dict[str, bool]:
        """Run workflow test suite."""
        print("\n" + "="*80)
        print("PHASE 2: Workflow Tests")
        print("="*80)
        
        results = {}
        
        # Comprehensive workflow tests
        test_path = Path(__file__).parent / "test_menu_workflow.py"
        success, output = run_test(
            "Menu Workflow Tests",
            [sys.executable, str(test_path)]
        )
        results['menu_workflow'] = success
        self.results.append(('Menu Workflow Tests', success))
        
        return results
    
    def run_stress_tests(self, level: str = 'standard') -> Dict[str, bool]:
        """Run stress test suite."""
        print("\n" + "="*80)
        print(f"PHASE 3: Stress Tests ({level})")
        print("="*80)
        
        results = {}
        
        # Determine stress test arguments
        if level == 'light':
            args = ['--light']
        elif level == 'heavy':
            args = ['--heavy']
        else:
            args = []
        
        # System stress tests
        test_path = Path(__file__).parent / "test_system_stress.py"
        success, output = run_test(
            f"System Stress Test ({level})",
            [sys.executable, str(test_path)] + args
        )
        results[f'stress_{level}'] = success
        self.results.append((f'System Stress Test ({level})', success))
        
        return results
    
    def run_performance_profiling(self) -> Dict[str, bool]:
        """Run performance profiling."""
        print("\n" + "="*80)
        print("PHASE 4: Performance Profiling")
        print("="*80)
        
        results = {}
        
        # Performance profiling with save
        test_path = Path(__file__).parent / "test_performance_profiling.py"
        success, output = run_test(
            "Performance Profiling",
            [sys.executable, str(test_path), "--full", "--save"]
        )
        results['performance_profiling'] = success
        self.results.append(('Performance Profiling', success))
        
        return results
    
    def print_summary(self):
        """Print test summary."""
        duration = (datetime.now() - self.start_time).total_seconds()
        
        print("\n" + "="*80)
        print("TEST SUMMARY")
        print("="*80)
        
        passed = sum(1 for _, success in self.results if success)
        failed = len(self.results) - passed
        
        print(f"\nDuration: {duration:.2f}s")
        print(f"Total Tests: {len(self.results)}")
        print(f"Passed: {passed}")
        print(f"Failed: {failed}")
        print(f"Success Rate: {(passed/len(self.results)*100):.1f}%")
        
        print("\n" + "-"*80)
        print("Detailed Results:")
        print("-"*80)
        
        for test_name, success in self.results:
            status = "✅ PASS" if success else "❌ FAIL"
            print(f"{status}: {test_name}")
        
        print("\n" + "="*80)
        
        if failed == 0:
            print("✅ ALL TESTS PASSED")
        else:
            print(f"❌ {failed} TEST(S) FAILED")
        
        print("="*80)
        
        return failed == 0


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Kinetra Comprehensive Test Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--quick',
        action='store_true',
        help="Run only basic and workflow tests (skip stress and profiling)"
    )
    
    parser.add_argument(
        '--full',
        action='store_true',
        help="Run all tests including heavy stress test"
    )
    
    parser.add_argument(
        '--with-profiling',
        action='store_true',
        help="Include performance profiling (adds ~30s)"
    )
    
    parser.add_argument(
        '--stress-level',
        choices=['light', 'standard', 'heavy'],
        default='standard',
        help="Stress test level (default: standard)"
    )
    
    args = parser.parse_args()
    
    runner = TestRunner()
    
    try:
        # Phase 1: Basic tests (always run)
        runner.run_basic_tests()
        
        # Phase 2: Workflow tests (always run)
        runner.run_workflow_tests()
        
        # Phase 3: Stress tests (skip if quick)
        if not args.quick:
            if args.full:
                # Run all stress levels for full test
                runner.run_stress_tests('light')
                runner.run_stress_tests('standard')
                runner.run_stress_tests('heavy')
            else:
                # Run specified stress level
                runner.run_stress_tests(args.stress_level)
        
        # Phase 4: Performance profiling (optional)
        if args.with_profiling or args.full:
            runner.run_performance_profiling()
        
        # Print summary
        all_passed = runner.print_summary()
        
        return 0 if all_passed else 1
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Tests interrupted by user")
        runner.print_summary()
        return 1
    except Exception as e:
        print(f"\n\n❌ Test runner failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
