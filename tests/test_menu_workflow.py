#!/usr/bin/env python3
"""
Test Kinetra Menu System - Complete Workflow Test
==================================================

Comprehensive test that cycles through every menu option to validate
full functionality and proper error handling.

This test automates the entire menu navigation by mocking user inputs
and verifying that each menu path executes correctly.
"""

import sys
import io
import logging
from pathlib import Path
from unittest.mock import patch, MagicMock, call
from contextlib import contextmanager

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

import kinetra_menu
from kinetra.workflow_manager import WorkflowManager


# =============================================================================
# TEST UTILITIES
# =============================================================================

@contextmanager
def mock_input_sequence(*inputs):
    """
    Context manager to mock a sequence of user inputs.
    
    Args:
        *inputs: Sequence of input values to return
    """
    input_iter = iter(inputs)
    
    def mock_input(prompt):
        try:
            value = next(input_iter)
            print(f"[MOCK INPUT] {prompt}: {value}")
            return value
        except StopIteration:
            print(f"[MOCK INPUT] {prompt}: <EOF>")
            return ""
    
    with patch('builtins.input', side_effect=mock_input):
        yield


@contextmanager
def capture_output():
    """Capture stdout and stderr."""
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    try:
        sys.stdout = io.StringIO()
        sys.stderr = io.StringIO()
        yield sys.stdout, sys.stderr
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr


class WorkflowTestLogger:
    """Logger for workflow tests."""
    
    def __init__(self):
        self.tests_run = []
        self.tests_passed = []
        self.tests_failed = []
        self.current_test = None
    
    def start_test(self, test_name: str):
        """Start a new test."""
        self.current_test = test_name
        print(f"\n{'='*80}")
        print(f"Testing: {test_name}")
        print(f"{'='*80}")
    
    def log_action(self, action: str):
        """Log an action."""
        print(f"  → {action}")
    
    def pass_test(self):
        """Mark current test as passed."""
        if self.current_test:
            self.tests_passed.append(self.current_test)
            self.tests_run.append((self.current_test, True))
            print(f"  ✅ PASS: {self.current_test}")
    
    def fail_test(self, error: str):
        """Mark current test as failed."""
        if self.current_test:
            self.tests_failed.append(self.current_test)
            self.tests_run.append((self.current_test, False))
            print(f"  ❌ FAIL: {self.current_test}")
            print(f"  Error: {error}")
    
    def print_summary(self):
        """Print test summary."""
        print(f"\n{'='*80}")
        print(f"Test Summary")
        print(f"{'='*80}")
        print(f"Total Tests: {len(self.tests_run)}")
        print(f"Passed: {len(self.tests_passed)}")
        print(f"Failed: {len(self.tests_failed)}")
        print(f"{'='*80}")
        
        for test_name, passed in self.tests_run:
            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"{status}: {test_name}")


# =============================================================================
# MENU WORKFLOW TESTS
# =============================================================================

def test_main_menu_navigation(logger: WorkflowTestLogger):
    """Test main menu displays and accepts valid input."""
    logger.start_test("Main Menu Navigation")
    
    try:
        wf_manager = WorkflowManager(
            log_dir="logs/test",
            enable_backups=False
        )
        
        # Test main menu with exit
        with mock_input_sequence('0'):
            with capture_output():
                result = kinetra_menu.show_main_menu(wf_manager)
        
        logger.log_action("Main menu accepts '0' to exit")
        assert result == False, "Main menu should return False on exit"
        
        logger.pass_test()
        return True
        
    except Exception as e:
        logger.fail_test(str(e))
        return False


def test_authentication_menu_paths(logger: WorkflowTestLogger):
    """Test all authentication menu paths."""
    logger.start_test("Authentication Menu Paths")
    
    try:
        wf_manager = WorkflowManager(
            log_dir="logs/test",
            enable_backups=False
        )
        
        # Test option 1: Select MetaAPI Account (then back)
        logger.log_action("Testing: Login & Authentication → Select MetaAPI Account")
        with mock_input_sequence('1', '1', '0'):
            with patch('subprocess.run') as mock_subprocess:
                mock_subprocess.return_value = MagicMock(returncode=0)
                with capture_output():
                    kinetra_menu.show_main_menu(wf_manager)
        
        # Test option 2: Test Connection
        logger.log_action("Testing: Login & Authentication → Test Connection")
        with mock_input_sequence('1', '2', '0'):
            with capture_output():
                kinetra_menu.show_main_menu(wf_manager)
        
        # Test back to main menu
        logger.log_action("Testing: Login & Authentication → Back")
        with mock_input_sequence('1', '0', '0'):
            with capture_output():
                kinetra_menu.show_main_menu(wf_manager)
        
        logger.pass_test()
        return True
        
    except Exception as e:
        logger.fail_test(str(e))
        return False


def test_exploration_menu_paths(logger: WorkflowTestLogger):
    """Test all exploration testing menu paths."""
    logger.start_test("Exploration Menu Paths")
    
    try:
        wf_manager = WorkflowManager(
            log_dir="logs/test",
            enable_backups=False
        )
        
        # Test option 1: Quick Exploration (decline)
        logger.log_action("Testing: Exploration → Quick Exploration (decline)")
        with mock_input_sequence('2', '1', 'n', '0', '0'):
            with capture_output():
                kinetra_menu.show_main_menu(wf_manager)
        
        # Test option 2: Custom Exploration (minimal input)
        logger.log_action("Testing: Exploration → Custom Exploration (decline)")
        with mock_input_sequence('2', '2', 'b', 'a', 'a', 'b', '', 'n', '0', '0'):
            with capture_output():
                kinetra_menu.show_main_menu(wf_manager)
        
        # Test option 3: Scientific Discovery (decline)
        logger.log_action("Testing: Exploration → Scientific Discovery Suite")
        with mock_input_sequence('2', '3', 'n', '0', '0'):
            with capture_output():
                kinetra_menu.show_main_menu(wf_manager)
        
        # Test option 4: Agent Comparison (decline)
        logger.log_action("Testing: Exploration → Agent Comparison")
        with mock_input_sequence('2', '4', 'n', '0', '0'):
            with capture_output():
                kinetra_menu.show_main_menu(wf_manager)
        
        # Test option 5: Measurement Analysis (decline)
        logger.log_action("Testing: Exploration → Measurement Analysis")
        with mock_input_sequence('2', '5', 'n', '0', '0'):
            with capture_output():
                kinetra_menu.show_main_menu(wf_manager)
        
        # Test back to main menu
        logger.log_action("Testing: Exploration → Back")
        with mock_input_sequence('2', '0', '0'):
            with capture_output():
                kinetra_menu.show_main_menu(wf_manager)
        
        logger.pass_test()
        return True
        
    except Exception as e:
        logger.fail_test(str(e))
        return False


def test_backtesting_menu_paths(logger: WorkflowTestLogger):
    """Test all backtesting menu paths."""
    logger.start_test("Backtesting Menu Paths")
    
    try:
        wf_manager = WorkflowManager(
            log_dir="logs/test",
            enable_backups=False
        )
        
        # Test option 1: Quick Backtest (decline)
        logger.log_action("Testing: Backtesting → Quick Backtest")
        with mock_input_sequence('3', '1', 'n', '0', '0'):
            with capture_output():
                kinetra_menu.show_main_menu(wf_manager)
        
        # Test option 2: Custom Backtesting (minimal input, decline)
        logger.log_action("Testing: Backtesting → Custom Backtesting")
        with mock_input_sequence('3', '2', 'a', 'a', '', '', 'n', '0', '0'):
            with capture_output():
                kinetra_menu.show_main_menu(wf_manager)
        
        # Test option 3: Monte Carlo Validation (decline by default)
        logger.log_action("Testing: Backtesting → Monte Carlo Validation")
        with mock_input_sequence('3', '3', '', '0', '0'):
            with patch('subprocess.run') as mock_subprocess:
                with capture_output():
                    kinetra_menu.show_main_menu(wf_manager)
        
        # Test option 4: Walk-Forward Testing (decline)
        logger.log_action("Testing: Backtesting → Walk-Forward Testing")
        with mock_input_sequence('3', '4', '', '', 'n', '0', '0'):
            with capture_output():
                kinetra_menu.show_main_menu(wf_manager)
        
        # Test option 5: Comparative Analysis (decline)
        logger.log_action("Testing: Backtesting → Comparative Analysis")
        with mock_input_sequence('3', '5', 'b', 'n', '0', '0'):
            with capture_output():
                kinetra_menu.show_main_menu(wf_manager)
        
        # Test back to main menu
        logger.log_action("Testing: Backtesting → Back")
        with mock_input_sequence('3', '0', '0'):
            with capture_output():
                kinetra_menu.show_main_menu(wf_manager)
        
        logger.pass_test()
        return True
        
    except Exception as e:
        logger.fail_test(str(e))
        return False


def test_data_management_menu_paths(logger: WorkflowTestLogger):
    """Test all data management menu paths."""
    logger.start_test("Data Management Menu Paths")
    
    try:
        wf_manager = WorkflowManager(
            log_dir="logs/test",
            enable_backups=False
        )
        
        # Test option 1: Auto-Download for Configuration
        logger.log_action("Testing: Data Management → Auto-Download")
        with mock_input_sequence('4', '1', 'b', 'a', '0', '0'):
            with capture_output():
                kinetra_menu.show_main_menu(wf_manager)
        
        # Test option 2: Manual Download
        logger.log_action("Testing: Data Management → Manual Download")
        with mock_input_sequence('4', '2', '0', '0'):
            with patch('subprocess.run') as mock_subprocess:
                with capture_output():
                    kinetra_menu.show_main_menu(wf_manager)
        
        # Test option 3: Check & Fill Missing Data
        logger.log_action("Testing: Data Management → Check & Fill Missing Data")
        with mock_input_sequence('4', '3', '0', '0'):
            with patch('subprocess.run') as mock_subprocess:
                with capture_output():
                    kinetra_menu.show_main_menu(wf_manager)
        
        # Test option 4: Data Integrity Check
        logger.log_action("Testing: Data Management → Data Integrity Check")
        with mock_input_sequence('4', '4', '0', '0'):
            with patch('subprocess.run') as mock_subprocess:
                with capture_output():
                    kinetra_menu.show_main_menu(wf_manager)
        
        # Test option 5: Prepare Data
        logger.log_action("Testing: Data Management → Prepare Data")
        with mock_input_sequence('4', '5', '0', '0'):
            with patch('subprocess.run') as mock_subprocess:
                with capture_output():
                    kinetra_menu.show_main_menu(wf_manager)
        
        # Test option 6: Backup & Restore → all sub-options
        logger.log_action("Testing: Data Management → Backup & Restore → Backup master")
        with mock_input_sequence('4', '6', '1', '0', '0', '0'):
            with patch('subprocess.run') as mock_subprocess:
                with capture_output():
                    kinetra_menu.show_main_menu(wf_manager)
        
        logger.log_action("Testing: Data Management → Backup & Restore → Backup prepared")
        with mock_input_sequence('4', '6', '2', '0', '0', '0'):
            with capture_output():
                kinetra_menu.show_main_menu(wf_manager)
        
        logger.log_action("Testing: Data Management → Backup & Restore → List backups")
        with mock_input_sequence('4', '6', '3', '0', '0', '0'):
            with capture_output():
                kinetra_menu.show_main_menu(wf_manager)
        
        logger.log_action("Testing: Data Management → Backup & Restore → Restore")
        with mock_input_sequence('4', '6', '4', '0', '0', '0'):
            with capture_output():
                kinetra_menu.show_main_menu(wf_manager)
        
        logger.log_action("Testing: Data Management → Backup & Restore → Back")
        with mock_input_sequence('4', '6', '0', '0', '0'):
            with capture_output():
                kinetra_menu.show_main_menu(wf_manager)
        
        # Test back to main menu
        logger.log_action("Testing: Data Management → Back")
        with mock_input_sequence('4', '0', '0'):
            with capture_output():
                kinetra_menu.show_main_menu(wf_manager)
        
        logger.pass_test()
        return True
        
    except Exception as e:
        logger.fail_test(str(e))
        return False


def test_system_status_menu_paths(logger: WorkflowTestLogger):
    """Test all system status menu paths."""
    logger.start_test("System Status Menu Paths")
    
    try:
        wf_manager = WorkflowManager(
            log_dir="logs/test",
            enable_backups=False
        )
        
        # Test option 1: Current System Health
        logger.log_action("Testing: System Status → Current System Health")
        with mock_input_sequence('5', '1', '0', '0'):
            with capture_output():
                kinetra_menu.show_main_menu(wf_manager)
        
        # Test option 2: Recent Test Results
        logger.log_action("Testing: System Status → Recent Test Results")
        with mock_input_sequence('5', '2', '0', '0'):
            with capture_output():
                kinetra_menu.show_main_menu(wf_manager)
        
        # Test option 3: Data Summary
        logger.log_action("Testing: System Status → Data Summary")
        with mock_input_sequence('5', '3', '0', '0'):
            with capture_output():
                kinetra_menu.show_main_menu(wf_manager)
        
        # Test option 4: Performance Metrics
        logger.log_action("Testing: System Status → Performance Metrics")
        with mock_input_sequence('5', '4', '0', '0'):
            with capture_output():
                kinetra_menu.show_main_menu(wf_manager)
        
        # Test back to main menu
        logger.log_action("Testing: System Status → Back")
        with mock_input_sequence('5', '0', '0'):
            with capture_output():
                kinetra_menu.show_main_menu(wf_manager)
        
        logger.pass_test()
        return True
        
    except Exception as e:
        logger.fail_test(str(e))
        return False


def test_input_validation(logger: WorkflowTestLogger):
    """Test input validation and error handling."""
    logger.start_test("Input Validation and Error Handling")
    
    try:
        wf_manager = WorkflowManager(
            log_dir="logs/test",
            enable_backups=False
        )
        
        # Test invalid main menu choice (should retry)
        logger.log_action("Testing: Invalid input → retry → valid input")
        with mock_input_sequence('99', 'invalid', '0'):
            with capture_output():
                kinetra_menu.show_main_menu(wf_manager)
        
        logger.log_action("Input validation handles invalid choices correctly")
        
        # Test get_input utility function directly
        logger.log_action("Testing: get_input with valid choices")
        with mock_input_sequence('a'):
            result = kinetra_menu.get_input("Test prompt", ['a', 'b', 'c'])
            assert result == 'a', "get_input should return valid choice"
        
        logger.log_action("Testing: get_input with invalid then valid choice")
        with mock_input_sequence('z', 'b'):
            result = kinetra_menu.get_input("Test prompt", ['a', 'b', 'c'])
            assert result == 'b', "get_input should retry on invalid input"
        
        logger.pass_test()
        return True
        
    except Exception as e:
        logger.fail_test(str(e))
        return False


def test_menu_config_methods(logger: WorkflowTestLogger):
    """Test MenuConfig utility methods."""
    logger.start_test("MenuConfig Utility Methods")
    
    try:
        from kinetra_menu import MenuConfig
        
        # Test get_all_asset_classes
        logger.log_action("Testing: MenuConfig.get_all_asset_classes()")
        asset_classes = MenuConfig.get_all_asset_classes()
        assert isinstance(asset_classes, list), "Should return list"
        assert len(asset_classes) > 0, "Should have asset classes"
        assert 'crypto' in asset_classes, "Should include crypto"
        
        # Test get_all_timeframes
        logger.log_action("Testing: MenuConfig.get_all_timeframes()")
        timeframes = MenuConfig.get_all_timeframes()
        assert isinstance(timeframes, list), "Should return list"
        assert len(timeframes) > 0, "Should have timeframes"
        assert 'H1' in timeframes, "Should include H1"
        
        # Test get_all_agent_types
        logger.log_action("Testing: MenuConfig.get_all_agent_types()")
        agent_types = MenuConfig.get_all_agent_types()
        assert isinstance(agent_types, list), "Should return list"
        assert len(agent_types) > 0, "Should have agent types"
        assert 'ppo' in agent_types, "Should include ppo"
        
        logger.pass_test()
        return True
        
    except Exception as e:
        logger.fail_test(str(e))
        return False


def test_helper_functions(logger: WorkflowTestLogger):
    """Test menu helper functions."""
    logger.start_test("Menu Helper Functions")
    
    try:
        # Test select_asset_classes
        logger.log_action("Testing: select_asset_classes()")
        with mock_input_sequence('a'):
            result = kinetra_menu.select_asset_classes()
            assert isinstance(result, list), "Should return list"
            assert len(result) >= 5, "Option 'a' should select all asset classes"
        
        with mock_input_sequence('b'):
            result = kinetra_menu.select_asset_classes()
            assert result == ['crypto'], "Option 'b' should select crypto"
        
        # Test select_timeframes
        logger.log_action("Testing: select_timeframes()")
        with mock_input_sequence('a'):
            result = kinetra_menu.select_timeframes()
            assert isinstance(result, list), "Should return list"
            assert len(result) >= 5, "Option 'a' should select all timeframes"
        
        with mock_input_sequence('b'):
            result = kinetra_menu.select_timeframes()
            assert 'M15' in result, "Option 'b' should include intraday timeframes"
        
        # Test select_agent_types
        logger.log_action("Testing: select_agent_types()")
        with mock_input_sequence('a'):
            result = kinetra_menu.select_agent_types()
            assert isinstance(result, list), "Should return list"
            assert len(result) >= 3, "Option 'a' should select all agents"
        
        with mock_input_sequence('b'):
            result = kinetra_menu.select_agent_types()
            assert result == ['ppo'], "Option 'b' should select PPO"
        
        # Test select_instruments
        logger.log_action("Testing: select_instruments()")
        with mock_input_sequence('a'):
            result = kinetra_menu.select_instruments(['crypto'])
            assert result == ['all'], "Option 'a' should select all"
        
        with mock_input_sequence('b', '5'):
            result = kinetra_menu.select_instruments(['crypto'])
            assert result == ['top_5'], "Option 'b' should select top N"
        
        logger.pass_test()
        return True
        
    except Exception as e:
        logger.fail_test(str(e))
        return False


def test_confirm_action(logger: WorkflowTestLogger):
    """Test confirm_action utility function."""
    logger.start_test("Confirm Action Function")
    
    try:
        # Test default True
        logger.log_action("Testing: confirm_action with default=True")
        with mock_input_sequence(''):
            result = kinetra_menu.confirm_action("Test?", default=True)
            assert result == True, "Empty input should return default"
        
        with mock_input_sequence('y'):
            result = kinetra_menu.confirm_action("Test?", default=True)
            assert result == True, "'y' should return True"
        
        with mock_input_sequence('n'):
            result = kinetra_menu.confirm_action("Test?", default=True)
            assert result == False, "'n' should return False"
        
        # Test default False
        logger.log_action("Testing: confirm_action with default=False")
        with mock_input_sequence(''):
            result = kinetra_menu.confirm_action("Test?", default=False)
            assert result == False, "Empty input should return default"
        
        with mock_input_sequence('yes'):
            result = kinetra_menu.confirm_action("Test?", default=False)
            assert result == True, "'yes' should return True"
        
        logger.pass_test()
        return True
        
    except Exception as e:
        logger.fail_test(str(e))
        return False


# =============================================================================
# MAIN TEST RUNNER
# =============================================================================

def main():
    """Run all workflow tests."""
    print("="*80)
    print("  Kinetra Menu System - Complete Workflow Test")
    print("="*80)
    print("\nThis test systematically cycles through every menu option")
    print("to validate full functionality and proper error handling.\n")
    
    logger = WorkflowTestLogger()
    
    # Run all tests
    test_main_menu_navigation(logger)
    test_authentication_menu_paths(logger)
    test_exploration_menu_paths(logger)
    test_backtesting_menu_paths(logger)
    test_data_management_menu_paths(logger)
    test_system_status_menu_paths(logger)
    test_input_validation(logger)
    test_menu_config_methods(logger)
    test_helper_functions(logger)
    test_confirm_action(logger)
    
    # Print summary
    logger.print_summary()
    
    # Return exit code
    return 0 if len(logger.tests_failed) == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
