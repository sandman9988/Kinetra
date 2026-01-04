#!/usr/bin/env python3
"""
Continuous Menu Exerciser
==========================

Exercises the Kinetra menu system continuously by:
- Navigating through all menu options automatically
- Testing each function path
- Logging all errors encountered
- Collecting statistics on menu behavior
- Identifying missing scripts and broken paths

Usage:
    python scripts/testing/exercise_menu_continuous.py [--iterations N] [--log-file PATH]
"""

import argparse
import json
import logging
import sys
import time
import traceback
from datetime import datetime
from io import StringIO
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from unittest.mock import patch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class MenuExerciser:
    """Exercises menu system continuously and logs errors."""
    
    def __init__(self, log_file: str = "logs/menu_exercise.log", verbose: bool = True):
        """
        Initialize menu exerciser.
        
        Args:
            log_file: Path to log file
            verbose: Whether to print verbose output
        """
        self.log_file = Path(log_file)
        self.verbose = verbose
        
        # Create log directory
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self.setup_logging()
        
        # Statistics
        self.stats = {
            'start_time': datetime.now(),
            'end_time': None,
            'iterations': 0,
            'menu_paths_tested': 0,
            'errors_found': [],
            'missing_scripts': [],
            'successful_paths': [],
            'context_issues': [],
            'navigation_errors': []
        }
        
        # Menu paths to test (menu option sequences)
        self.test_paths = self.generate_test_paths()
        
        self.logger.info("="*80)
        self.logger.info("Menu Exerciser Initialized")
        self.logger.info("="*80)
        self.logger.info(f"Log file: {log_file}")
        self.logger.info(f"Test paths: {len(self.test_paths)}")
        self.logger.info("="*80)
    
    def setup_logging(self):
        """Setup logging configuration."""
        self.logger = logging.getLogger('MenuExerciser')
        self.logger.setLevel(logging.DEBUG)
        self.logger.handlers = []
        
        # File handler
        file_handler = logging.FileHandler(self.log_file)
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        self.logger.addHandler(file_handler)
        
        # Console handler
        if self.verbose:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(logging.INFO)
            console_formatter = logging.Formatter(
                '%(asctime)s - %(levelname)s - %(message)s',
                datefmt='%H:%M:%S'
            )
            console_handler.setFormatter(console_formatter)
            self.logger.addHandler(console_handler)
    
    def generate_test_paths(self) -> List[Tuple[str, List[str]]]:
        """
        Generate list of menu paths to test.
        
        Returns:
            List of (description, input_sequence) tuples
        """
        paths = [
            # Main menu -> Exit
            ("Main Menu Exit", ['0']),
            
            # Authentication menu paths
            ("Authentication Menu", ['1', '0']),
            ("Test Connection", ['1', '2', '0']),
            
            # Exploration menu paths
            ("Exploration Menu", ['2', '0']),
            ("Quick Exploration (cancel)", ['2', '1', 'n', '0']),
            ("Custom Exploration Menu", ['2', '2', '0']),
            ("Scientific Discovery Menu", ['2', '3', 'n', '0']),
            ("Agent Comparison Menu", ['2', '4', 'n', '0']),
            ("Measurement Analysis Menu", ['2', '5', 'n', '0']),
            
            # Backtesting menu paths
            ("Backtesting Menu", ['3', '0']),
            ("Quick Backtest (cancel)", ['3', '1', 'n', '0']),
            ("Custom Backtest Menu", ['3', '2', '0']),
            ("Monte Carlo Menu", ['3', '3', '0']),
            ("Walk Forward Menu", ['3', '4', 'n', '0']),
            ("Comparative Analysis Menu", ['3', '5', 'n', '0']),
            
            # Live Testing menu paths
            ("Live Testing Menu", ['4', '0']),
            ("Virtual Trading Menu", ['4', '1', '0']),
            ("Demo Account Menu", ['4', '2', 'n', '0']),
            ("MT5 Connection Test Menu", ['4', '3', '0']),
            ("Live Testing Guide", ['4', '4', '0']),
            
            # Data Management menu paths
            ("Data Management Menu", ['5', '0']),
            ("Auto Download Menu", ['5', '1', '0']),
            ("Manual Download Menu", ['5', '2', '0']),
            ("Check Missing Data Menu", ['5', '3', '0']),
            ("Data Integrity Menu", ['5', '4', '0']),
            ("Prepare Data Menu", ['5', '5', '0']),
            ("Backup Restore Menu", ['5', '6', '0']),
            
            # System Status menu paths
            ("System Status Menu", ['6', '0']),
            ("System Health", ['6', '1', '0']),
            ("Recent Results", ['6', '2', '0']),
            ("Data Summary", ['6', '3', '0']),
            ("Performance Metrics", ['6', '4', '0']),
        ]
        
        return paths
    
    def exercise_menu_path(self, description: str, inputs: List[str]) -> Tuple[bool, Optional[str]]:
        """
        Exercise a single menu path.
        
        Args:
            description: Description of the menu path
            inputs: List of inputs to provide
            
        Returns:
            (success, error_message) tuple
        """
        self.logger.info(f"\n{'='*80}")
        self.logger.info(f"Testing: {description}")
        self.logger.info(f"Inputs: {inputs}")
        self.logger.info(f"{'='*80}")
        
        try:
            # Prepare input sequence
            input_sequence = '\n'.join(inputs) + '\n'
            
            # Capture output
            output_buffer = StringIO()
            
            # Mock input and capture output
            with patch('builtins.input', side_effect=inputs):
                with patch('sys.stdout', output_buffer):
                    with patch('sys.stderr', output_buffer):
                        try:
                            # Import menu (fresh each time)
                            if 'kinetra_menu' in sys.modules:
                                del sys.modules['kinetra_menu']
                            
                            import kinetra_menu
                            
                            # Create workflow manager
                            from kinetra.workflow_manager import WorkflowManager
                            wf_manager = WorkflowManager(
                                log_dir="logs",
                                backup_dir="data/backups/workflow",
                                enable_backups=False,
                                enable_checksums=False
                            )
                            
                            # Test specific menu based on first input
                            if inputs[0] == '1':
                                # Authentication menu
                                kinetra_menu.show_authentication_menu(wf_manager)
                            elif inputs[0] == '2':
                                # Exploration menu
                                kinetra_menu.show_exploration_menu(wf_manager)
                            elif inputs[0] == '3':
                                # Backtesting menu
                                kinetra_menu.show_backtesting_menu(wf_manager)
                            elif inputs[0] == '4':
                                # Live testing menu
                                kinetra_menu.show_live_testing_menu(wf_manager)
                            elif inputs[0] == '5':
                                # Data management menu
                                kinetra_menu.show_data_management_menu(wf_manager)
                            elif inputs[0] == '6':
                                # System status menu
                                kinetra_menu.show_system_status_menu(wf_manager)
                            
                        except EOFError:
                            # Expected when we run out of inputs
                            pass
                        except SystemExit as e:
                            # Menu might exit - that's ok if exit code is 0
                            if e.code != 0:
                                raise
            
            # Get output
            output = output_buffer.getvalue()
            
            # Check for common error patterns
            error_patterns = [
                'Traceback',
                'Error:',
                'Exception:',
                'Failed',
                'not found',
                'missing',
                'ImportError',
                'ModuleNotFoundError',
                'FileNotFoundError',
                'AttributeError',
                'TypeError',
                'ValueError'
            ]
            
            found_errors = []
            for pattern in error_patterns:
                if pattern in output:
                    found_errors.append(pattern)
            
            if found_errors:
                error_msg = f"Found error patterns: {found_errors}"
                self.logger.warning(f"⚠️  {error_msg}")
                self.logger.debug(f"Output:\n{output}")
                return False, error_msg
            
            self.logger.info(f"✅ {description} - SUCCESS")
            return True, None
            
        except Exception as e:
            error_msg = f"{type(e).__name__}: {str(e)}"
            self.logger.error(f"❌ {description} - FAILED: {error_msg}")
            self.logger.debug(traceback.format_exc())
            return False, error_msg
    
    def run_iteration(self, iteration: int) -> Dict:
        """
        Run one complete iteration of menu testing.
        
        Args:
            iteration: Iteration number
            
        Returns:
            Dictionary with iteration results
        """
        self.logger.info(f"\n{'#'*80}")
        self.logger.info(f"# ITERATION {iteration}")
        self.logger.info(f"{'#'*80}\n")
        
        results = {
            'iteration': iteration,
            'timestamp': datetime.now().isoformat(),
            'paths_tested': 0,
            'successes': 0,
            'failures': 0,
            'errors': []
        }
        
        # Test each path
        for description, inputs in self.test_paths:
            success, error = self.exercise_menu_path(description, inputs)
            
            results['paths_tested'] += 1
            self.stats['menu_paths_tested'] += 1
            
            if success:
                results['successes'] += 1
                if description not in self.stats['successful_paths']:
                    self.stats['successful_paths'].append(description)
            else:
                results['failures'] += 1
                error_record = {
                    'path': description,
                    'inputs': inputs,
                    'error': error,
                    'iteration': iteration,
                    'timestamp': datetime.now().isoformat()
                }
                results['errors'].append(error_record)
                
                # Add to global error list if not already present
                error_key = f"{description}:{error}"
                if error_key not in [f"{e['path']}:{e['error']}" for e in self.stats['errors_found']]:
                    self.stats['errors_found'].append(error_record)
            
            # Small delay between tests
            time.sleep(0.1)
        
        # Print iteration summary
        self.logger.info(f"\n{'='*80}")
        self.logger.info(f"Iteration {iteration} Summary:")
        self.logger.info(f"  Paths tested: {results['paths_tested']}")
        self.logger.info(f"  Successes: {results['successes']}")
        self.logger.info(f"  Failures: {results['failures']}")
        if results['failures'] > 0:
            self.logger.info(f"  Success rate: {(results['successes']/results['paths_tested'])*100:.1f}%")
        self.logger.info(f"{'='*80}\n")
        
        return results
    
    def print_final_report(self):
        """Print final testing report."""
        self.stats['end_time'] = datetime.now()
        duration = (self.stats['end_time'] - self.stats['start_time']).total_seconds()
        
        self.logger.info(f"\n{'#'*80}")
        self.logger.info(f"# FINAL REPORT")
        self.logger.info(f"{'#'*80}\n")
        
        self.logger.info(f"Duration: {duration:.1f}s")
        self.logger.info(f"Iterations: {self.stats['iterations']}")
        self.logger.info(f"Total paths tested: {self.stats['menu_paths_tested']}")
        self.logger.info(f"Successful paths: {len(self.stats['successful_paths'])}")
        self.logger.info(f"Unique errors: {len(self.stats['errors_found'])}")
        
        if self.stats['errors_found']:
            self.logger.info(f"\n{'='*80}")
            self.logger.info(f"ERRORS FOUND ({len(self.stats['errors_found'])} unique)")
            self.logger.info(f"{'='*80}")
            
            for i, error in enumerate(self.stats['errors_found'], 1):
                self.logger.info(f"\n{i}. {error['path']}")
                self.logger.info(f"   Inputs: {error['inputs']}")
                self.logger.info(f"   Error: {error['error']}")
                self.logger.info(f"   First seen: Iteration {error['iteration']}")
        
        # Save report to JSON
        self.save_report()
        
        self.logger.info(f"\n{'#'*80}")
        self.logger.info(f"Report saved to: {self.log_file.parent / 'menu_exercise_report.json'}")
        self.logger.info(f"{'#'*80}\n")
    
    def save_report(self):
        """Save testing report to JSON file."""
        try:
            report_file = self.log_file.parent / 'menu_exercise_report.json'
            
            # Convert datetime to string
            report = self.stats.copy()
            report['start_time'] = report['start_time'].isoformat()
            if report['end_time']:
                report['end_time'] = report['end_time'].isoformat()
            
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2)
            
            self.logger.debug(f"Report saved to {report_file}")
            
        except Exception as e:
            self.logger.warning(f"Failed to save report: {e}")
    
    def run(self, max_iterations: Optional[int] = None):
        """
        Run continuous menu exerciser.
        
        Args:
            max_iterations: Maximum iterations to run (None = run once)
        """
        self.logger.info("Starting menu exerciser...")
        
        try:
            iteration = 1
            while True:
                self.stats['iterations'] = iteration
                self.run_iteration(iteration)
                
                # Check if we should stop
                if max_iterations and iteration >= max_iterations:
                    break
                
                iteration += 1
                
        except KeyboardInterrupt:
            self.logger.info("\n⚠️  Interrupted by user (Ctrl+C)")
        except Exception as e:
            self.logger.error(f"Unexpected error: {e}")
            self.logger.debug(traceback.format_exc())
        finally:
            self.print_final_report()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Exercise Kinetra menu system continuously"
    )
    parser.add_argument(
        '--iterations',
        type=int,
        default=1,
        help='Number of iterations to run (default: 1)'
    )
    parser.add_argument(
        '--log-file',
        type=str,
        default='logs/menu_exercise.log',
        help='Path to log file'
    )
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress console output'
    )
    
    args = parser.parse_args()
    
    # Create exerciser
    exerciser = MenuExerciser(
        log_file=args.log_file,
        verbose=not args.quiet
    )
    
    # Run exerciser
    exerciser.run(max_iterations=args.iterations)


if __name__ == '__main__':
    main()
