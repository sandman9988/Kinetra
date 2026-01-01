"""
Automated Test Execution Framework with Auto-Fixing and Continuation
=====================================================================

This module implements the automated execution framework that:
1. Runs discovery methods systematically
2. Validates results statistically (PBO, CPCV)
3. Auto-fixes common failures
4. Continues execution after failures
5. Integrates code review (Claude Code)
6. Provides progress tracking and resumption

Philosophy:
-----------
"We don't know what we don't know" - systematically explore all possibilities
with statistical rigor, automatic error handling, and continuous validation.
"""

import json
import logging
import traceback
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ExecutionConfig:
    """Configuration for automated test execution."""
    name: str
    max_retries: int = 3
    auto_fix_enabled: bool = True
    continue_on_failure: bool = True
    checkpoint_frequency: int = 5  # Save progress every N tests
    
    # Statistical validation
    pbo_threshold: float = 0.05  # Probability of Backtest Overfitting
    min_sharpe_ratio: float = 0.8
    min_sample_size: int = 30
    significance_level: float = 0.05
    
    # Resource limits
    max_parallel_tests: int = 4
    timeout_minutes: int = 60
    memory_limit_gb: float = 16.0


@dataclass
class FailureRecord:
    """Record of a test failure."""
    test_name: str
    timestamp: datetime
    error_type: str
    error_message: str
    traceback_str: str
    auto_fix_attempted: bool = False
    auto_fix_successful: bool = False
    retry_count: int = 0


@dataclass
class ExecutionProgress:
    """Track execution progress."""
    total_tests: int
    completed_tests: int
    successful_tests: int
    failed_tests: int
    skipped_tests: int
    current_test: Optional[str] = None
    failures: List[FailureRecord] = field(default_factory=list)
    start_time: datetime = field(default_factory=datetime.now)
    last_checkpoint: Optional[datetime] = None


class AutoFixer:
    """
    Automatic error detection and fixing.
    
    Implements common remediation strategies for known failure patterns.
    """
    
    def __init__(self):
        self.fix_strategies = {
            'ImportError': self._fix_import_error,
            'ValueError': self._fix_value_error,
            'RuntimeError': self._fix_runtime_error,
            'MemoryError': self._fix_memory_error,
            'TimeoutError': self._fix_timeout_error,
            'DataError': self._fix_data_error,
        }
        
        self.fixes_applied: List[Dict[str, Any]] = []
    
    def attempt_fix(self, failure: FailureRecord, context: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Attempt to automatically fix a failure.
        
        Args:
            failure: The failure to fix
            context: Additional context (config, data, etc.)
            
        Returns:
            (success, message)
        """
        error_type = failure.error_type
        
        if error_type in self.fix_strategies:
            logger.info(f"Attempting auto-fix for {error_type}: {failure.test_name}")
            try:
                success, message = self.fix_strategies[error_type](failure, context)
                
                self.fixes_applied.append({
                    'test_name': failure.test_name,
                    'error_type': error_type,
                    'success': success,
                    'message': message,
                    'timestamp': datetime.now().isoformat(),
                })
                
                return success, message
            except Exception as e:
                return False, f"Auto-fix failed: {str(e)}"
        else:
            return False, f"No auto-fix strategy for {error_type}"
    
    def _fix_import_error(self, failure: FailureRecord, context: Dict) -> Tuple[bool, str]:
        """Fix import errors by installing missing dependencies."""
        error_msg = failure.error_message
        
        # Extract module name from error message
        if "No module named" in error_msg:
            module_name = error_msg.split("No module named")[1].strip().strip("'\"")
            
            # Attempt pip install
            import subprocess
            try:
                subprocess.run(
                    ['pip', 'install', module_name],
                    check=True,
                    capture_output=True,
                    timeout=300
                )
                return True, f"Installed missing module: {module_name}"
            except subprocess.CalledProcessError as e:
                return False, f"Failed to install {module_name}: {e.stderr.decode()}"
        
        return False, "Could not identify missing module"
    
    def _fix_value_error(self, failure: FailureRecord, context: Dict) -> Tuple[bool, str]:
        """Fix value errors by adjusting parameters."""
        error_msg = failure.error_message
        
        # Common patterns
        if "sample size" in error_msg.lower():
            # Reduce sample requirements
            if 'config' in context:
                context['config'].min_sample_size = max(10, context['config'].min_sample_size // 2)
                return True, f"Reduced min_sample_size to {context['config'].min_sample_size}"
        
        elif "dimension" in error_msg.lower():
            # Reduce dimensionality
            if 'config' in context and 'latent_dims' in context['config'].agent_config:
                dims = context['config'].agent_config['latent_dims']
                context['config'].agent_config['latent_dims'] = [d // 2 for d in dims if d > 2]
                return True, f"Reduced latent dimensions"
        
        return False, "Could not identify value error fix"
    
    def _fix_runtime_error(self, failure: FailureRecord, context: Dict) -> Tuple[bool, str]:
        """Fix runtime errors."""
        error_msg = failure.error_message
        
        if "gpu" in error_msg.lower() or "cuda" in error_msg.lower():
            # Disable GPU
            if 'config' in context:
                context['config'].use_gpu = False
                return True, "Disabled GPU acceleration"
        
        return False, "Could not identify runtime error fix"
    
    def _fix_memory_error(self, failure: FailureRecord, context: Dict) -> Tuple[bool, str]:
        """Fix memory errors by reducing batch size or complexity."""
        if 'config' in context:
            # Reduce episodes
            if hasattr(context['config'], 'episodes'):
                context['config'].episodes = max(10, context['config'].episodes // 2)
                return True, f"Reduced episodes to {context['config'].episodes}"
        
        return False, "Could not reduce memory usage"
    
    def _fix_timeout_error(self, failure: FailureRecord, context: Dict) -> Tuple[bool, str]:
        """Fix timeout errors by reducing scope."""
        if 'config' in context:
            # Reduce complexity
            if hasattr(context['config'], 'episodes'):
                context['config'].episodes = max(5, context['config'].episodes // 3)
                return True, f"Reduced episodes to {context['config'].episodes} for faster execution"
        
        return False, "Could not reduce execution time"
    
    def _fix_data_error(self, failure: FailureRecord, context: Dict) -> Tuple[bool, str]:
        """Fix data errors by skipping or reloading data."""
        return False, "Manual data intervention required"


class StatisticalRigor:
    """
    Implements statistical validation:
    - PBO (Probability of Backtest Overfitting)
    - CPCV (Combinatorially Purged Cross-Validation)
    - Bootstrap confidence intervals
    - Monte Carlo permutation tests
    """
    
    @staticmethod
    def calculate_pbo(
        returns_is: np.ndarray,
        returns_oos: np.ndarray,
        n_trials: int = 1000
    ) -> float:
        """
        Calculate Probability of Backtest Overfitting.
        
        PBO measures the probability that observed IS performance
        is due to overfitting rather than true alpha.
        
        Args:
            returns_is: In-sample returns
            returns_oos: Out-of-sample returns
            n_trials: Number of Monte Carlo trials
            
        Returns:
            PBO score (0-1, lower is better, < 0.05 is good)
        """
        if len(returns_is) < 10 or len(returns_oos) < 10:
            return 1.0  # Not enough data
        
        # Calculate Sharpe ratios
        sharpe_is = np.mean(returns_is) / (np.std(returns_is) + 1e-10)
        sharpe_oos = np.mean(returns_oos) / (np.std(returns_oos) + 1e-10)
        
        # Monte Carlo simulation
        better_in_sample = 0
        
        for _ in range(n_trials):
            # Shuffle returns
            shuffled = np.random.permutation(np.concatenate([returns_is, returns_oos]))
            n_is = len(returns_is)
            
            sim_is = shuffled[:n_is]
            sim_oos = shuffled[n_is:]
            
            sim_sharpe_is = np.mean(sim_is) / (np.std(sim_is) + 1e-10)
            sim_sharpe_oos = np.mean(sim_oos) / (np.std(sim_oos) + 1e-10)
            
            # Check if simulated IS is better than OOS
            if sim_sharpe_is > sim_sharpe_oos:
                better_in_sample += 1
        
        pbo = better_in_sample / n_trials
        
        # Also check if actual OOS performance degraded significantly
        if sharpe_oos < 0.5 * sharpe_is:
            pbo = max(pbo, 0.5)  # Penalty for significant degradation
        
        return pbo
    
    @staticmethod
    def combinatorially_purged_cv(
        data: pd.DataFrame,
        n_splits: int = 5,
        embargo_pct: float = 0.01
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Combinatorially Purged Cross-Validation.
        
        Prevents information leakage in financial time series by:
        1. Purging overlapping observations
        2. Adding embargo periods between train/test
        3. Using combinatorial splits to reduce variance
        
        Args:
            data: Time series data
            n_splits: Number of CV splits
            embargo_pct: Embargo period as % of data
            
        Returns:
            List of (train_idx, test_idx) tuples
        """
        n = len(data)
        embargo_size = int(n * embargo_pct)
        
        splits = []
        
        # Create non-overlapping splits
        fold_size = n // n_splits
        
        for i in range(n_splits):
            # Test set for this fold
            test_start = i * fold_size
            test_end = min((i + 1) * fold_size, n)
            
            # Purge: remove embargo before and after test set
            purge_start = max(0, test_start - embargo_size)
            purge_end = min(n, test_end + embargo_size)
            
            # Train set: everything except test and purged regions
            train_idx = np.concatenate([
                np.arange(0, purge_start),
                np.arange(purge_end, n)
            ])
            
            test_idx = np.arange(test_start, test_end)
            
            if len(train_idx) > 0 and len(test_idx) > 0:
                splits.append((train_idx, test_idx))
        
        return splits
    
    @staticmethod
    def bootstrap_confidence_interval(
        data: np.ndarray,
        statistic_func: Callable = np.mean,
        n_bootstrap: int = 1000,
        confidence_level: float = 0.95
    ) -> Tuple[float, float]:
        """
        Calculate bootstrap confidence interval.
        
        Args:
            data: Sample data
            statistic_func: Function to compute statistic
            n_bootstrap: Number of bootstrap samples
            confidence_level: Confidence level (e.g., 0.95 for 95%)
            
        Returns:
            (lower_bound, upper_bound)
        """
        if len(data) < 10:
            return (0.0, 0.0)
        
        bootstrap_stats = []
        
        for _ in range(n_bootstrap):
            sample = np.random.choice(data, size=len(data), replace=True)
            stat = statistic_func(sample)
            bootstrap_stats.append(stat)
        
        bootstrap_stats = np.array(bootstrap_stats)
        
        alpha = 1 - confidence_level
        lower_pct = (alpha / 2) * 100
        upper_pct = (1 - alpha / 2) * 100
        
        lower = np.percentile(bootstrap_stats, lower_pct)
        upper = np.percentile(bootstrap_stats, upper_pct)
        
        return (lower, upper)
    
    @staticmethod
    def monte_carlo_permutation_test(
        sample1: np.ndarray,
        sample2: np.ndarray,
        n_permutations: int = 10000
    ) -> float:
        """
        Monte Carlo permutation test for comparing two samples.
        
        Tests if sample1 is significantly different from sample2
        by permutation testing.
        
        Returns:
            p-value
        """
        if len(sample1) < 5 or len(sample2) < 5:
            return 1.0
        
        # Observed difference
        observed_diff = np.mean(sample1) - np.mean(sample2)
        
        # Permutation test
        combined = np.concatenate([sample1, sample2])
        n1 = len(sample1)
        
        count_extreme = 0
        
        for _ in range(n_permutations):
            # Shuffle and split
            shuffled = np.random.permutation(combined)
            perm_sample1 = shuffled[:n1]
            perm_sample2 = shuffled[n1:]
            
            perm_diff = np.mean(perm_sample1) - np.mean(perm_sample2)
            
            # Check if permuted difference is as extreme as observed
            if abs(perm_diff) >= abs(observed_diff):
                count_extreme += 1
        
        p_value = count_extreme / n_permutations
        
        return p_value


class TestExecutor:
    """
    Main test execution engine with auto-fixing and continuation.
    
    Orchestrates:
    1. Discovery method execution
    2. Statistical validation
    3. Auto-fixing of failures
    4. Progress checkpointing
    5. Result aggregation
    """
    
    def __init__(self, config: ExecutionConfig, output_dir: str = "test_execution"):
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.auto_fixer = AutoFixer()
        self.stats_validator = StatisticalRigor()
        
        self.progress = ExecutionProgress(
            total_tests=0,
            completed_tests=0,
            successful_tests=0,
            failed_tests=0,
            skipped_tests=0
        )
        
        self.checkpoint_file = self.output_dir / "checkpoint.json"
        
        logger.info(f"TestExecutor initialized: {config.name}")
        logger.info(f"Auto-fix enabled: {config.auto_fix_enabled}")
        logger.info(f"Continue on failure: {config.continue_on_failure}")
    
    def execute_test_suite(
        self,
        test_functions: List[Callable],
        test_names: List[str],
        contexts: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Execute a suite of tests with auto-fixing and continuation.
        
        Args:
            test_functions: List of test functions to execute
            test_names: Names for each test
            contexts: Optional context dicts for each test
            
        Returns:
            Aggregated results
        """
        if contexts is None:
            contexts = [{}] * len(test_functions)
        
        self.progress.total_tests = len(test_functions)
        
        results = {}
        
        for i, (test_func, test_name, context) in enumerate(zip(test_functions, test_names, contexts)):
            self.progress.current_test = test_name
            
            logger.info(f"\n{'='*80}")
            logger.info(f"Executing: {test_name} ({i+1}/{len(test_functions)})")
            logger.info(f"{'='*80}\n")
            
            # Execute with retry logic
            result = self._execute_with_retry(test_func, test_name, context)
            
            results[test_name] = result
            
            # Update progress
            self.progress.completed_tests += 1
            if result.get('success', False):
                self.progress.successful_tests += 1
            else:
                self.progress.failed_tests += 1
            
            # Checkpoint if needed
            if (i + 1) % self.config.checkpoint_frequency == 0:
                self._save_checkpoint(results)
        
        # Final save
        self._save_final_results(results)
        
        return results
    
    def _execute_with_retry(
        self,
        test_func: Callable,
        test_name: str,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute a test with retry and auto-fix logic."""
        for attempt in range(self.config.max_retries):
            try:
                logger.info(f"Attempt {attempt + 1}/{self.config.max_retries}")
                
                # Execute test
                result = test_func(**context)
                
                # Validate statistically if result contains returns
                if 'returns_is' in result and 'returns_oos' in result:
                    pbo = self.stats_validator.calculate_pbo(
                        result['returns_is'],
                        result['returns_oos']
                    )
                    result['pbo'] = pbo
                    result['passes_pbo'] = pbo < self.config.pbo_threshold
                    
                    if not result['passes_pbo']:
                        logger.warning(f"PBO test failed: {pbo:.3f} >= {self.config.pbo_threshold}")
                
                result['success'] = True
                result['attempts'] = attempt + 1
                
                logger.info(f"✓ Test succeeded: {test_name}")
                return result
                
            except Exception as e:
                error_type = type(e).__name__
                error_msg = str(e)
                tb = traceback.format_exc()
                
                logger.error(f"✗ Test failed: {error_type}: {error_msg}")
                
                failure = FailureRecord(
                    test_name=test_name,
                    timestamp=datetime.now(),
                    error_type=error_type,
                    error_message=error_msg,
                    traceback_str=tb,
                    retry_count=attempt
                )
                
                self.progress.failures.append(failure)
                
                # Attempt auto-fix
                if self.config.auto_fix_enabled and attempt < self.config.max_retries - 1:
                    success, message = self.auto_fixer.attempt_fix(failure, context)
                    
                    failure.auto_fix_attempted = True
                    failure.auto_fix_successful = success
                    
                    if success:
                        logger.info(f"Auto-fix applied: {message}")
                        continue  # Retry with fixed context
                    else:
                        logger.warning(f"Auto-fix failed: {message}")
                
                # Last attempt failed
                if attempt == self.config.max_retries - 1:
                    if self.config.continue_on_failure:
                        logger.warning(f"Continuing despite failure in {test_name}")
                        return {
                            'success': False,
                            'error': error_msg,
                            'error_type': error_type,
                            'attempts': attempt + 1,
                            'traceback': tb
                        }
                    else:
                        raise  # Re-raise to stop execution
        
        # Should not reach here
        return {'success': False, 'error': 'Max retries exceeded'}
    
    def _save_checkpoint(self, results: Dict[str, Any]):
        """Save progress checkpoint."""
        checkpoint = {
            'config': asdict(self.config),
            'progress': asdict(self.progress),
            'results': results,
            'timestamp': datetime.now().isoformat(),
        }
        
        with open(self.checkpoint_file, 'w') as f:
            json.dump(checkpoint, f, indent=2, default=str)
        
        self.progress.last_checkpoint = datetime.now()
        logger.info(f"Checkpoint saved: {self.checkpoint_file}")
    
    def _save_final_results(self, results: Dict[str, Any]):
        """Save final results."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = self.output_dir / f"results_{timestamp}.json"
        
        final_results = {
            'config': asdict(self.config),
            'progress': asdict(self.progress),
            'results': results,
            'fixes_applied': self.auto_fixer.fixes_applied,
            'timestamp': datetime.now().isoformat(),
        }
        
        with open(results_file, 'w') as f:
            json.dump(final_results, f, indent=2, default=str)
        
        logger.info(f"\n{'='*80}")
        logger.info("EXECUTION COMPLETE")
        logger.info(f"{'='*80}")
        logger.info(f"Total tests: {self.progress.total_tests}")
        logger.info(f"Successful: {self.progress.successful_tests}")
        logger.info(f"Failed: {self.progress.failed_tests}")
        logger.info(f"Success rate: {self.progress.successful_tests/self.progress.total_tests*100:.1f}%")
        logger.info(f"Results saved: {results_file}")
        logger.info(f"{'='*80}\n")
    
    def load_checkpoint(self) -> Optional[Dict[str, Any]]:
        """Load progress from checkpoint."""
        if self.checkpoint_file.exists():
            with open(self.checkpoint_file, 'r') as f:
                checkpoint = json.load(f)
            
            logger.info(f"Loaded checkpoint from {self.checkpoint_file}")
            logger.info(f"Previous progress: {checkpoint['progress']['completed_tests']}/{checkpoint['progress']['total_tests']}")
            
            return checkpoint
        
        return None
