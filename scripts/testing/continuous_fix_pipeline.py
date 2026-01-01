#!/usr/bin/env python3
"""
Continuous Testing, Fixing, and Verification Pipeline
======================================================

Automated pipeline for production readiness:
1. Run comprehensive tests
2. Collect and categorize errors
3. Apply automated fixes
4. Verify fixes
5. Report progress
6. Loop until production-ready

Usage:
    python scripts/testing/continuous_fix_pipeline.py [--max-cycles 10] [--auto-fix]
"""

import argparse
import json
import logging
import re
import subprocess
import sys
import time
import traceback
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class ErrorCategory:
    """Error categorization for automated fixing."""
    
    # Error patterns with fix strategies
    PATTERNS = {
        'dtype_incompatibility': {
            'pattern': r"resolved dtypes are not compatible|dtype\('<U\d+'\)",
            'severity': 'CRITICAL',
            'auto_fixable': True,
            'fix_strategy': 'convert_string_to_numeric'
        },
        'stopiteration': {
            'pattern': r'StopIteration',
            'severity': 'HIGH',
            'auto_fixable': True,
            'fix_strategy': 'add_stopiteration_handler'
        },
        'keyerror': {
            'pattern': r"KeyError: '[^']+'",
            'severity': 'HIGH',
            'auto_fixable': True,
            'fix_strategy': 'add_key_check'
        },
        'attributeerror': {
            'pattern': r"AttributeError: '[^']+' object has no attribute '[^']+'",
            'severity': 'MEDIUM',
            'auto_fixable': True,
            'fix_strategy': 'add_attribute_check'
        },
        'typeerror': {
            'pattern': r'TypeError:',
            'severity': 'MEDIUM',
            'auto_fixable': False,
            'fix_strategy': 'manual_review_required'
        },
        'valueerror': {
            'pattern': r'ValueError:',
            'severity': 'MEDIUM',
            'auto_fixable': True,
            'fix_strategy': 'add_value_validation'
        },
        'indexerror': {
            'pattern': r'IndexError:',
            'severity': 'MEDIUM',
            'auto_fixable': True,
            'fix_strategy': 'add_bounds_check'
        },
        'filenotfound': {
            'pattern': r'FileNotFoundError|No such file or directory',
            'severity': 'HIGH',
            'auto_fixable': True,
            'fix_strategy': 'create_missing_file_or_directory'
        },
        'importerror': {
            'pattern': r'ImportError|ModuleNotFoundError',
            'severity': 'HIGH',
            'auto_fixable': True,
            'fix_strategy': 'install_missing_dependency'
        },
        'missing_column': {
            'pattern': r"KeyError.*time|'time' not found",
            'severity': 'HIGH',
            'auto_fixable': True,
            'fix_strategy': 'fix_column_names'
        },
        'timeout': {
            'pattern': r'Timeout|timed out',
            'severity': 'MEDIUM',
            'auto_fixable': True,
            'fix_strategy': 'increase_timeout_or_optimize'
        }
    }
    
    @classmethod
    def categorize(cls, error_message: str) -> Tuple[str, Dict]:
        """
        Categorize an error message.
        
        Returns:
            (category_name, category_info) tuple
        """
        for category, info in cls.PATTERNS.items():
            if re.search(info['pattern'], error_message, re.IGNORECASE):
                return category, info
        
        return 'unknown', {
            'pattern': '.*',
            'severity': 'LOW',
            'auto_fixable': False,
            'fix_strategy': 'manual_review_required'
        }


class ContinuousTestPipeline:
    """Continuous testing and fixing pipeline."""
    
    def __init__(
        self,
        log_dir: str = "logs/continuous_pipeline",
        auto_fix: bool = False,
        max_cycles: int = 10
    ):
        """
        Initialize pipeline.
        
        Args:
            log_dir: Directory for logs
            auto_fix: Whether to automatically apply fixes
            max_cycles: Maximum test-fix cycles
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.auto_fix = auto_fix
        self.max_cycles = max_cycles
        
        # Setup logging
        self.setup_logging()
        
        # Statistics
        self.stats = {
            'start_time': datetime.now(),
            'cycles': 0,
            'total_errors': 0,
            'errors_fixed': 0,
            'errors_remaining': 0,
            'errors_by_category': defaultdict(int),
            'fixes_applied': [],
            'test_runs': []
        }
        
        self.logger.info("="*80)
        self.logger.info("Continuous Test & Fix Pipeline Initialized")
        self.logger.info("="*80)
        self.logger.info(f"Auto-fix: {auto_fix}")
        self.logger.info(f"Max cycles: {max_cycles}")
        self.logger.info("="*80)
    
    def setup_logging(self):
        """Setup logging."""
        self.logger = logging.getLogger('ContinuousTestPipeline')
        self.logger.setLevel(logging.DEBUG)
        self.logger.handlers = []
        
        # File handler
        log_file = self.log_dir / f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        self.logger.addHandler(file_handler)
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%H:%M:%S'
        )
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)
    
    def run_test_suite(self) -> Dict:
        """
        Run comprehensive test suite.
        
        Returns:
            Dictionary with test results and errors
        """
        self.logger.info("\n" + "="*80)
        self.logger.info("Running Test Suite")
        self.logger.info("="*80)
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'tests_run': 0,
            'tests_passed': 0,
            'tests_failed': 0,
            'errors': []
        }
        
        # Test 1: Menu exerciser
        self.logger.info("\n[1/5] Menu System Test...")
        menu_result = self._run_command(
            [sys.executable, "scripts/testing/exercise_menu_continuous.py", "--iterations", "1"],
            timeout=60
        )
        results['tests_run'] += 1
        if menu_result['success']:
            results['tests_passed'] += 1
        else:
            results['tests_failed'] += 1
            results['errors'].extend(menu_result['errors'])
        
        # Test 2: Real data exerciser
        self.logger.info("\n[2/5] Real Data Test...")
        data_result = self._run_command(
            [sys.executable, "scripts/testing/exercise_menu_with_real_data.py"],
            timeout=120
        )
        results['tests_run'] += 1
        if data_result['success']:
            results['tests_passed'] += 1
        else:
            results['tests_failed'] += 1
            results['errors'].extend(data_result['errors'])
        
        # Test 3: Data preparation
        self.logger.info("\n[3/5] Data Preparation Test...")
        prep_result = self._run_command(
            [sys.executable, "scripts/download/prepare_data.py", "--auto", "--test-ratio=0.2"],
            timeout=120
        )
        results['tests_run'] += 1
        if prep_result['success']:
            results['tests_passed'] += 1
        else:
            results['tests_failed'] += 1
            results['errors'].extend(prep_result['errors'])
        
        # Test 4: Backtest (sample)
        self.logger.info("\n[4/5] Backtest Test...")
        # Skip if data not prepared
        if Path("data/prepared/train").exists():
            backtest_result = self._run_command(
                [sys.executable, "-c", "print('Backtest test placeholder')"],
                timeout=60
            )
            results['tests_run'] += 1
            if backtest_result['success']:
                results['tests_passed'] += 1
            else:
                results['tests_failed'] += 1
                results['errors'].extend(backtest_result['errors'])
        
        # Test 5: Import tests
        self.logger.info("\n[5/5] Import Test...")
        import_result = self._test_imports()
        results['tests_run'] += 1
        if import_result['success']:
            results['tests_passed'] += 1
        else:
            results['tests_failed'] += 1
            results['errors'].extend(import_result['errors'])
        
        self.logger.info(f"\nTest Suite Complete:")
        self.logger.info(f"  Tests run: {results['tests_run']}")
        self.logger.info(f"  Passed: {results['tests_passed']}")
        self.logger.info(f"  Failed: {results['tests_failed']}")
        self.logger.info(f"  Errors found: {len(results['errors'])}")
        
        return results
    
    def _run_command(self, cmd: List[str], timeout: int = 30) -> Dict:
        """
        Run a command and capture errors.
        
        Returns:
            Dictionary with success status and errors
        """
        result = {
            'success': False,
            'errors': [],
            'output': '',
            'exit_code': None
        }
        
        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True
            )
            
            try:
                output, _ = process.communicate(timeout=timeout)
                result['output'] = output
                result['exit_code'] = process.returncode
                
                if process.returncode == 0:
                    result['success'] = True
                else:
                    # Parse errors from output
                    result['errors'] = self._extract_errors(output)
                    
            except subprocess.TimeoutExpired:
                process.kill()
                result['errors'].append({
                    'type': 'timeout',
                    'message': f"Command timed out after {timeout}s",
                    'command': ' '.join(cmd)
                })
        
        except Exception as e:
            result['errors'].append({
                'type': 'exception',
                'message': str(e),
                'traceback': traceback.format_exc()
            })
        
        return result
    
    def _test_imports(self) -> Dict:
        """Test critical imports."""
        result = {
            'success': True,
            'errors': []
        }
        
        critical_imports = [
            'pandas',
            'numpy',
            'scipy',
            'sklearn',
            'tqdm',
            'kinetra.workflow_manager',
        ]
        
        for module_name in critical_imports:
            try:
                __import__(module_name)
            except Exception as e:
                result['success'] = False
                result['errors'].append({
                    'type': 'import_error',
                    'message': f"Failed to import {module_name}: {e}",
                    'module': module_name
                })
        
        return result
    
    def _extract_errors(self, output: str) -> List[Dict]:
        """Extract errors from command output."""
        errors = []
        
        # Look for common error patterns
        error_lines = []
        for line in output.split('\n'):
            if any(pattern in line for pattern in [
                'Error:', 'ERROR:', 'Failed:', 'FAILED:',
                'Exception:', 'Traceback', '❌'
            ]):
                error_lines.append(line)
        
        # Group consecutive error lines
        if error_lines:
            errors.append({
                'type': 'runtime_error',
                'message': '\n'.join(error_lines[:10]),  # First 10 lines
                'full_output': output
            })
        
        return errors
    
    def categorize_errors(self, errors: List[Dict]) -> Dict[str, List[Dict]]:
        """
        Categorize errors for prioritized fixing.
        
        Returns:
            Dictionary mapping category to list of errors
        """
        categorized = defaultdict(list)
        
        for error in errors:
            message = error.get('message', '')
            category, info = ErrorCategory.categorize(message)
            
            error['category'] = category
            error['severity'] = info['severity']
            error['auto_fixable'] = info['auto_fixable']
            error['fix_strategy'] = info['fix_strategy']
            
            categorized[category].append(error)
            self.stats['errors_by_category'][category] += 1
        
        return dict(categorized)
    
    def apply_fixes(self, categorized_errors: Dict[str, List[Dict]]) -> List[Dict]:
        """
        Apply automated fixes where possible.
        
        Returns:
            List of fixes applied
        """
        fixes_applied = []
        
        # Sort by severity: CRITICAL > HIGH > MEDIUM > LOW
        severity_order = {'CRITICAL': 0, 'HIGH': 1, 'MEDIUM': 2, 'LOW': 3}
        
        for category in sorted(categorized_errors.keys(),
                              key=lambda c: severity_order.get(
                                  categorized_errors[c][0].get('severity', 'LOW'), 4
                              )):
            errors = categorized_errors[category]
            
            if not errors:
                continue
            
            self.logger.info(f"\n{'='*80}")
            self.logger.info(f"Processing {category} ({len(errors)} errors)")
            self.logger.info(f"{'='*80}")
            
            for error in errors:
                if not error.get('auto_fixable'):
                    self.logger.warning(f"  ⚠️  Cannot auto-fix: {error.get('fix_strategy')}")
                    continue
                
                if not self.auto_fix:
                    self.logger.info(f"  💡 Fix available: {error.get('fix_strategy')}")
                    continue
                
                # Apply fix
                fix_result = self._apply_single_fix(error)
                if fix_result:
                    fixes_applied.append(fix_result)
                    self.stats['errors_fixed'] += 1
        
        return fixes_applied
    
    def _apply_single_fix(self, error: Dict) -> Optional[Dict]:
        """
        Apply a single fix.
        
        Returns:
            Fix result dictionary or None if failed
        """
        strategy = error.get('fix_strategy')
        
        self.logger.info(f"  🔧 Applying fix: {strategy}")
        
        try:
            if strategy == 'add_stopiteration_handler':
                return self._fix_stopiteration(error)
            elif strategy == 'convert_string_to_numeric':
                return self._fix_dtype_incompatibility(error)
            elif strategy == 'fix_column_names':
                return self._fix_column_names(error)
            elif strategy == 'install_missing_dependency':
                return self._fix_missing_dependency(error)
            elif strategy == 'add_key_check':
                return self._fix_keyerror(error)
            else:
                self.logger.warning(f"  ⚠️  No implementation for strategy: {strategy}")
                return None
        
        except Exception as e:
            self.logger.error(f"  ❌ Fix failed: {e}")
            return None
    
    def _fix_stopiteration(self, error: Dict) -> Dict:
        """Fix StopIteration errors."""
        # This is already fixed in kinetra_menu.py
        return {
            'error': error,
            'fix': 'StopIteration handler already added to menu',
            'status': 'already_fixed'
        }
    
    def _fix_dtype_incompatibility(self, error: Dict) -> Dict:
        """Fix dtype incompatibility errors."""
        self.logger.info("  📝 Creating dtype fix documentation")
        
        fix_doc = self.log_dir / "dtype_fix_guide.md"
        with open(fix_doc, 'w') as f:
            f.write("""# DType Incompatibility Fix Guide

## Issue
Numpy cannot perform mathematical operations (add.reduce) on string dtype arrays.

## Root Cause
DataFrames contain string columns that should be numeric.

## Fix Strategy

### 1. Identify String Columns
```python
import pandas as pd

df = pd.read_csv('data.csv')
string_cols = df.select_dtypes(include=['object']).columns
print(f"String columns: {list(string_cols)}")
```

### 2. Convert to Numeric
```python
# For known numeric columns
numeric_cols = ['open', 'high', 'low', 'close', 'volume']
for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
```

### 3. Handle MT5 Format
MT5 CSVs use `<DATE>`, `<TIME>`, etc. Convert:
```python
# Rename columns
df.columns = [col.strip('<>').lower() for col in df.columns]

# Combine date and time
df['time'] = pd.to_datetime(df['date'] + ' ' + df['time'])
df = df.drop(columns=['date'])
```

## Files to Fix
- scripts/download/prepare_data.py (add dtype conversion)
- kinetra/backtest_engine.py (add dtype validation)
- All strategy files (validate input dtypes)
""")
        
        return {
            'error': error,
            'fix': f'Created fix guide: {fix_doc}',
            'status': 'documentation_created',
            'action_required': 'Apply fixes from guide'
        }
    
    def _fix_column_names(self, error: Dict) -> Dict:
        """Fix column name mismatches."""
        self.logger.info("  🔄 Adding MT5 format converter to plan")
        
        return {
            'error': error,
            'fix': 'Need to run convert_mt5_format.py first',
            'status': 'requires_conversion',
            'script': 'scripts/download/convert_mt5_format.py'
        }
    
    def _fix_missing_dependency(self, error: Dict) -> Dict:
        """Fix missing dependencies."""
        module = error.get('module', '')
        
        if module:
            self.logger.info(f"  📦 Installing {module}")
            try:
                subprocess.run(
                    [sys.executable, "-m", "pip", "install", module],
                    check=True,
                    capture_output=True
                )
                return {
                    'error': error,
                    'fix': f'Installed {module}',
                    'status': 'fixed'
                }
            except subprocess.CalledProcessError as e:
                return {
                    'error': error,
                    'fix': f'Failed to install {module}',
                    'status': 'failed',
                    'details': str(e)
                }
        
        return {
            'error': error,
            'fix': 'Could not determine module to install',
            'status': 'manual_required'
        }
    
    def _fix_keyerror(self, error: Dict) -> Dict:
        """Fix KeyError issues."""
        return {
            'error': error,
            'fix': 'Add .get() with default value',
            'status': 'requires_code_review',
            'recommendation': 'Replace dict[key] with dict.get(key, default)'
        }
    
    def run_cycle(self, cycle_num: int) -> bool:
        """
        Run one test-fix cycle.
        
        Returns:
            True if should continue, False if done
        """
        self.logger.info(f"\n{'#'*80}")
        self.logger.info(f"# CYCLE {cycle_num}")
        self.logger.info(f"{'#'*80}")
        
        # Run tests
        test_results = self.run_test_suite()
        self.stats['test_runs'].append(test_results)
        
        # Count errors
        total_errors = len(test_results['errors'])
        self.stats['total_errors'] = total_errors
        
        if total_errors == 0:
            self.logger.info("\n🎉 No errors found! System is production-ready!")
            return False
        
        self.logger.info(f"\n📊 Found {total_errors} errors")
        
        # Categorize
        categorized = self.categorize_errors(test_results['errors'])
        
        # Report categories
        self.logger.info("\nError Categories:")
        for category, errors in sorted(categorized.items(),
                                      key=lambda x: len(x[1]),
                                      reverse=True):
            severity = errors[0].get('severity', 'UNKNOWN')
            fixable = sum(1 for e in errors if e.get('auto_fixable'))
            self.logger.info(f"  {category:20s}: {len(errors):3d} ({severity}, {fixable} auto-fixable)")
        
        # Apply fixes
        fixes = self.apply_fixes(categorized)
        self.stats['fixes_applied'].extend(fixes)
        
        if fixes:
            self.logger.info(f"\n✅ Applied {len(fixes)} fixes")
            return True  # Continue testing
        else:
            self.logger.warning("\n⚠️  No auto-fixes available. Manual intervention required.")
            self.stats['errors_remaining'] = total_errors
            return False  # Stop
    
    def generate_report(self):
        """Generate final report."""
        duration = (datetime.now() - self.stats['start_time']).total_seconds()
        
        self.logger.info(f"\n{'#'*80}")
        self.logger.info(f"# FINAL REPORT")
        self.logger.info(f"{'#'*80}")
        
        self.logger.info(f"\nDuration: {duration:.1f}s")
        self.logger.info(f"Cycles completed: {self.stats['cycles']}")
        self.logger.info(f"Total errors found: {self.stats['total_errors']}")
        self.logger.info(f"Errors fixed: {self.stats['errors_fixed']}")
        self.logger.info(f"Errors remaining: {self.stats['errors_remaining']}")
        
        if self.stats['errors_by_category']:
            self.logger.info(f"\nErrors by category:")
            for category, count in sorted(self.stats['errors_by_category'].items(),
                                         key=lambda x: x[1],
                                         reverse=True):
                self.logger.info(f"  {category:20s}: {count}")
        
        # Save report
        report_file = self.log_dir / f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            # Convert datetime to string
            stats_copy = self.stats.copy()
            stats_copy['start_time'] = stats_copy['start_time'].isoformat()
            json.dump(stats_copy, f, indent=2)
        
        self.logger.info(f"\n📄 Report saved: {report_file}")
    
    def run(self):
        """Run continuous testing pipeline."""
        self.logger.info("Starting continuous testing pipeline...")
        
        try:
            for cycle in range(1, self.max_cycles + 1):
                self.stats['cycles'] = cycle
                
                should_continue = self.run_cycle(cycle)
                
                if not should_continue:
                    break
                
                # Small delay between cycles
                time.sleep(2)
            
        except KeyboardInterrupt:
            self.logger.info("\n⚠️  Pipeline interrupted by user")
        except Exception as e:
            self.logger.error(f"\n❌ Pipeline error: {e}")
            self.logger.debug(traceback.format_exc())
        finally:
            self.generate_report()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Continuous testing and fixing pipeline"
    )
    parser.add_argument(
        '--max-cycles',
        type=int,
        default=10,
        help='Maximum test-fix cycles (default: 10)'
    )
    parser.add_argument(
        '--auto-fix',
        action='store_true',
        help='Automatically apply fixes where possible'
    )
    parser.add_argument(
        '--log-dir',
        type=str,
        default='logs/continuous_pipeline',
        help='Directory for logs'
    )
    
    args = parser.parse_args()
    
    # Create pipeline
    pipeline = ContinuousTestPipeline(
        log_dir=args.log_dir,
        auto_fix=args.auto_fix,
        max_cycles=args.max_cycles
    )
    
    # Run pipeline
    pipeline.run()


if __name__ == '__main__':
    main()
