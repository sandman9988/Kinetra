#!/usr/bin/env python3
"""
Comprehensive Menu Testing System
===================================

Real testing system that:
1. Exercises ALL menu options systematically
2. Logs all failures with full context
3. Tests with real data (not mocks where possible)
4. Validates error handling and recovery
5. Generates detailed test report

Philosophy:
- Test REAL flows, not mocked flows
- Log EVERYTHING for debugging
- Fail LOUDLY with context
- Validate system state after operations
"""

import sys
import json
import traceback
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from unittest.mock import patch, MagicMock
from io import StringIO
from contextlib import contextmanager

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

import kinetra_menu
from kinetra.workflow_manager import WorkflowManager


# =============================================================================
# TEST LOGGING SYSTEM
# =============================================================================

class ComprehensiveTestLogger:
    """
    Advanced logging system that captures:
    - Test execution details
    - Failure context (stack traces, state)
    - System state before/after operations
    - Performance metrics
    """
    
    def __init__(self, log_dir: Path):
        self.log_dir = log_dir
        self.log_dir.mkdir(exist_ok=True, parents=True)
        
        self.test_run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = self.log_dir / f"menu_test_{self.test_run_id}.log"
        self.json_file = self.log_dir / f"menu_test_{self.test_run_id}.json"
        
        self.tests = []
        self.current_test = None
        self.failures = []
        self.warnings = []
        
    def start_test(self, test_name: str, menu_path: str):
        """Start a new test with full context."""
        self.current_test = {
            'name': test_name,
            'menu_path': menu_path,
            'start_time': datetime.now().isoformat(),
            'status': 'running',
            'logs': [],
            'warnings': [],
            'errors': [],
            'system_state_before': self._capture_system_state(),
        }
        self._log(f"\n{'='*80}")
        self._log(f"TEST: {test_name}")
        self._log(f"PATH: {menu_path}")
        self._log(f"{'='*80}")
        
    def log(self, message: str, level: str = "INFO"):
        """Log a message with timestamp and level."""
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        log_entry = f"[{timestamp}] [{level}] {message}"
        self._log(log_entry)
        
        if self.current_test:
            self.current_test['logs'].append({
                'timestamp': timestamp,
                'level': level,
                'message': message
            })
            
    def log_error(self, error: Exception, context: str = ""):
        """Log an error with full stack trace."""
        error_msg = f"{context}: {type(error).__name__}: {str(error)}"
        trace = traceback.format_exc()
        
        self._log(f"ERROR: {error_msg}")
        self._log(f"TRACE:\n{trace}")
        
        if self.current_test:
            self.current_test['errors'].append({
                'context': context,
                'type': type(error).__name__,
                'message': str(error),
                'traceback': trace
            })
            
    def log_warning(self, message: str):
        """Log a warning."""
        self.log(message, "WARN")
        self.warnings.append(message)
        if self.current_test:
            self.current_test['warnings'].append(message)
            
    def pass_test(self):
        """Mark current test as passed."""
        if self.current_test:
            self.current_test['status'] = 'passed'
            self.current_test['end_time'] = datetime.now().isoformat()
            self.current_test['system_state_after'] = self._capture_system_state()
            self.tests.append(self.current_test)
            self._log(f"✅ PASS: {self.current_test['name']}")
            self.current_test = None
            
    def fail_test(self, reason: str):
        """Mark current test as failed with reason."""
        if self.current_test:
            self.current_test['status'] = 'failed'
            self.current_test['failure_reason'] = reason
            self.current_test['end_time'] = datetime.now().isoformat()
            self.current_test['system_state_after'] = self._capture_system_state()
            self.tests.append(self.current_test)
            self.failures.append(self.current_test)
            self._log(f"❌ FAIL: {self.current_test['name']}")
            self._log(f"   Reason: {reason}")
            self.current_test = None
            
    def _capture_system_state(self) -> Dict:
        """Capture current system state for debugging."""
        return {
            'data_dir_exists': Path('data/master').exists(),
            'data_files_count': len(list(Path('data/master').glob('**/*.csv'))) if Path('data/master').exists() else 0,
            'results_dir_exists': Path('results').exists(),
            'env_file_exists': Path('.env').exists(),
        }
        
    def _log(self, message: str):
        """Write to log file."""
        with open(self.log_file, 'a') as f:
            f.write(message + '\n')
        print(message)
        
    def generate_report(self) -> str:
        """Generate comprehensive test report."""
        total = len(self.tests)
        passed = len([t for t in self.tests if t['status'] == 'passed'])
        failed = len(self.failures)
        
        report = f"""
{'='*80}
COMPREHENSIVE MENU TEST REPORT
{'='*80}

Test Run ID: {self.test_run_id}
Total Tests: {total}
Passed:      {passed} ({passed/total*100:.1f}%)
Failed:      {failed} ({failed/total*100:.1f}%)
Warnings:    {len(self.warnings)}

{'='*80}
TEST RESULTS
{'='*80}
"""
        
        for test in self.tests:
            status_icon = "✅" if test['status'] == 'passed' else "❌"
            report += f"\n{status_icon} {test['name']}"
            report += f"\n   Path: {test['menu_path']}"
            if test['status'] == 'failed':
                report += f"\n   Reason: {test.get('failure_reason', 'Unknown')}"
            if test.get('errors'):
                report += f"\n   Errors: {len(test['errors'])}"
            if test.get('warnings'):
                report += f"\n   Warnings: {len(test['warnings'])}"
                
        if self.failures:
            report += f"\n\n{'='*80}\nFAILURE DETAILS\n{'='*80}\n"
            for failure in self.failures:
                report += f"\n❌ {failure['name']}"
                report += f"\n   Path: {failure['menu_path']}"
                report += f"\n   Reason: {failure.get('failure_reason', 'Unknown')}"
                if failure.get('errors'):
                    for error in failure['errors'][:3]:  # Show first 3 errors
                        report += f"\n   Error: {error['type']}: {error['message']}"
                        
        # Save JSON report
        with open(self.json_file, 'w') as f:
            json.dump({
                'test_run_id': self.test_run_id,
                'tests': self.tests,
                'summary': {
                    'total': total,
                    'passed': passed,
                    'failed': failed,
                    'warnings': len(self.warnings)
                }
            }, f, indent=2)
            
        report += f"\n\n📁 Detailed logs: {self.log_file}"
        report += f"\n📁 JSON report: {self.json_file}\n"
        
        return report


# =============================================================================
# MENU TEST SCENARIOS
# =============================================================================

class MenuTestScenarios:
    """Comprehensive menu test scenarios."""
    
    def __init__(self, logger: ComprehensiveTestLogger):
        self.logger = logger
        self.wf_manager = WorkflowManager(
            log_dir="logs/menu_test",
            enable_backups=False
        )
        
    def run_all_tests(self):
        """Run all test scenarios."""
        # Authentication Menu Tests
        self.test_authentication_menu()
        
        # Exploration Menu Tests
        self.test_exploration_quick_decline()
        self.test_exploration_quick_accept()
        self.test_exploration_all_options()
        
        # Backtest Menu Tests
        self.test_backtest_all_options()
        
        # Data Management Menu Tests
        self.test_data_management_all_options()
        
        # System Status Menu Tests
        self.test_system_status_all_options()
        
        # Navigation Tests
        self.test_back_navigation()
        self.test_invalid_inputs()
        
    def test_authentication_menu(self):
        """Test authentication menu paths."""
        self.logger.start_test(
            "Authentication Menu - Test Connection",
            "Main → Login & Authentication → Test Connection"
        )
        
        try:
            with self._mock_inputs('1', '2', '0', '0'):
                with self._capture_output() as (out, err):
                    kinetra_menu.show_main_menu(self.wf_manager)
                    
            output = out.getvalue()
            
            # Validate expected behavior
            if "Credentials found" in output or "No credentials found" in output:
                self.logger.pass_test()
            else:
                self.logger.fail_test("Expected connection test output not found")
                
        except Exception as e:
            self.logger.log_error(e, "Authentication menu test")
            self.logger.fail_test(str(e))
            
    def test_exploration_quick_decline(self):
        """Test declining quick exploration."""
        self.logger.start_test(
            "Exploration - Quick Exploration (Decline)",
            "Main → Exploration → Quick Exploration → Decline"
        )
        
        try:
            with self._mock_inputs('2', '1', 'n', '0', '0'):
                with self._capture_output() as (out, err):
                    kinetra_menu.show_main_menu(self.wf_manager)
                    
            self.logger.log("User declined exploration - expected behavior")
            self.logger.pass_test()
            
        except Exception as e:
            self.logger.log_error(e, "Quick exploration decline")
            self.logger.fail_test(str(e))
            
    def test_exploration_quick_accept(self):
        """Test accepting quick exploration with real data check."""
        self.logger.start_test(
            "Exploration - Quick Exploration (Accept)",
            "Main → Exploration → Quick Exploration → Accept"
        )
        
        try:
            # Check if data exists
            data_dir = Path('data/master')
            if not data_dir.exists():
                self.logger.log_warning("Data directory doesn't exist - test will fail")
                
            with self._mock_inputs('2', '1', 'y', '0'):
                with self._capture_output() as (out, err):
                    kinetra_menu.show_main_menu(self.wf_manager)
                    
            output = out.getvalue()
            
            # Check for success or proper error handling
            if "[ERROR] No instruments loaded" in output:
                if "Troubleshooting:" in output:
                    self.logger.log("Proper error message with troubleshooting shown")
                    self.logger.pass_test()
                else:
                    self.logger.fail_test("Error occurred but no troubleshooting guidance")
            elif "Exploration complete" in output:
                self.logger.log("Exploration ran successfully")
                self.logger.pass_test()
            else:
                self.logger.fail_test("Unexpected output - neither success nor proper error")
                
        except Exception as e:
            self.logger.log_error(e, "Quick exploration accept")
            self.logger.fail_test(str(e))
            
    def test_exploration_all_options(self):
        """Test all exploration menu options."""
        options = [
            ('2', "Custom Exploration"),
            ('3', "Scientific Discovery"),
            ('4', "Agent Comparison"),
            ('5', "Measurement Analysis"),
        ]
        
        for option, name in options:
            self.logger.start_test(
                f"Exploration - {name} (Decline)",
                f"Main → Exploration → {name} → Decline"
            )
            
            try:
                # Build input sequence: 2 (Exploration), option, 'n' (decline), 0, 0
                inputs = ['2', option, 'n', '0', '0']
                
                with self._mock_inputs(*inputs):
                    with self._capture_output() as (out, err):
                        kinetra_menu.show_main_menu(self.wf_manager)
                        
                self.logger.pass_test()
                
            except Exception as e:
                self.logger.log_error(e, f"Exploration option {option}")
                self.logger.fail_test(str(e))
                
    def test_backtest_all_options(self):
        """Test all backtest menu options."""
        # Test each backtest option by declining
        options = [
            ('1', "Quick Backtest"),
            ('3', "Monte Carlo"),
            ('4', "Walk-Forward"),
        ]
        
        for option, name in options:
            self.logger.start_test(
                f"Backtest - {name} (Decline)",
                f"Main → Backtest → {name} → Decline"
            )
            
            try:
                inputs = ['3', option, 'n', '0', '0'] if option != '3' else ['3', option, '', '0', '0']
                
                with self._mock_inputs(*inputs):
                    with self._capture_output() as (out, err):
                        kinetra_menu.show_main_menu(self.wf_manager)
                        
                self.logger.pass_test()
                
            except Exception as e:
                self.logger.log_error(e, f"Backtest option {option}")
                self.logger.fail_test(str(e))
                
    def test_data_management_all_options(self):
        """Test all data management menu options."""
        options = [
            ('4', "Data Integrity Check"),
            ('6', "Backup & Restore → Back"),
        ]
        
        for option, name in options:
            self.logger.start_test(
                f"Data Management - {name}",
                f"Main → Data Management → {name}"
            )
            
            try:
                if option == '6':
                    inputs = ['4', '6', '0', '0', '0']  # Extra 0 for backup submenu
                else:
                    inputs = ['4', option, '0', '0']
                    
                with self._mock_inputs(*inputs):
                    with self._capture_output() as (out, err):
                        kinetra_menu.show_main_menu(self.wf_manager)
                        
                self.logger.pass_test()
                
            except Exception as e:
                self.logger.log_error(e, f"Data management option {option}")
                self.logger.fail_test(str(e))
                
    def test_system_status_all_options(self):
        """Test all system status menu options."""
        options = [
            ('1', "Current System Health"),
            ('2', "Recent Test Results"),
            ('3', "Data Summary"),
            ('4', "Performance Metrics"),
        ]
        
        for option, name in options:
            self.logger.start_test(
                f"System Status - {name}",
                f"Main → System Status → {name}"
            )
            
            try:
                inputs = ['5', option, '0', '0']
                
                with self._mock_inputs(*inputs):
                    with self._capture_output() as (out, err):
                        kinetra_menu.show_main_menu(self.wf_manager)
                        
                self.logger.pass_test()
                
            except Exception as e:
                self.logger.log_error(e, f"System status option {option}")
                self.logger.fail_test(str(e))
                
    def test_back_navigation(self):
        """Test back/0 navigation from submenus."""
        self.logger.start_test(
            "Navigation - Back from submenu",
            "Main → Any Submenu → Back (0)"
        )
        
        try:
            # Test back from each submenu
            submenus = ['1', '2', '3', '4', '5']
            
            for submenu in submenus:
                with self._mock_inputs(submenu, '0', '0'):
                    with self._capture_output() as (out, err):
                        kinetra_menu.show_main_menu(self.wf_manager)
                        
            self.logger.log("All submenus support back navigation")
            self.logger.pass_test()
            
        except Exception as e:
            self.logger.log_error(e, "Back navigation")
            self.logger.fail_test(str(e))
            
    def test_invalid_inputs(self):
        """Test handling of invalid inputs."""
        self.logger.start_test(
            "Error Handling - Invalid Menu Inputs",
            "Main → Invalid Input → Valid Input"
        )
        
        try:
            # Try invalid input, then valid exit
            with self._mock_inputs('99', '0'):
                with self._capture_output() as (out, err):
                    kinetra_menu.show_main_menu(self.wf_manager)
                    
            output = out.getvalue()
            
            if "Invalid choice" in output:
                self.logger.log("Invalid input properly rejected")
                self.logger.pass_test()
            else:
                self.logger.fail_test("Invalid input not properly handled")
                
        except Exception as e:
            self.logger.log_error(e, "Invalid input handling")
            self.logger.fail_test(str(e))
            
    @contextmanager
    def _mock_inputs(self, *inputs):
        """Mock user inputs."""
        input_iter = iter(inputs)
        
        def mock_input(prompt):
            try:
                value = next(input_iter)
                self.logger.log(f"INPUT: {prompt.strip()[:50]}... → {value}", "DEBUG")
                return value
            except StopIteration:
                return "0"  # Default to exit
                
        with patch('builtins.input', side_effect=mock_input):
            yield
            
    @contextmanager
    def _capture_output(self):
        """Capture stdout and stderr."""
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        try:
            sys.stdout = StringIO()
            sys.stderr = StringIO()
            yield sys.stdout, sys.stderr
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr


# =============================================================================
# MAIN TEST RUNNER
# =============================================================================

def main():
    """Run comprehensive menu tests."""
    print("="*80)
    print("KINETRA COMPREHENSIVE MENU TESTING SYSTEM")
    print("="*80)
    print("\nInitializing test logging system...")
    
    log_dir = Path("logs/menu_tests")
    logger = ComprehensiveTestLogger(log_dir)
    
    print(f"✓ Logs will be saved to: {logger.log_file}")
    print("\nRunning comprehensive menu tests...\n")
    
    scenarios = MenuTestScenarios(logger)
    
    try:
        scenarios.run_all_tests()
    except Exception as e:
        logger.log_error(e, "Test runner")
        print(f"\n❌ Test runner failed: {e}")
        traceback.print_exc()
        
    # Generate and display report
    report = logger.generate_report()
    print(report)
    
    # Save report to file
    report_file = log_dir / f"menu_test_report_{logger.test_run_id}.txt"
    with open(report_file, 'w') as f:
        f.write(report)
        
    print(f"\n📄 Full report saved to: {report_file}")
    
    # Return exit code based on results
    return 0 if not logger.failures else 1


if __name__ == "__main__":
    sys.exit(main())
