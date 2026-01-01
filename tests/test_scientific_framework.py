"""
Unit tests for scientific testing framework components.
"""

import unittest
import numpy as np
import pandas as pd
from datetime import datetime

from kinetra.test_executor import (
    TestExecutor,
    ExecutionConfig,
    AutoFixer,
    StatisticalRigor,
    FailureRecord
)
from kinetra.discovery_methods import (
    DiscoveryMethodRunner,
    HiddenDimensionDiscovery,
    ChaosTheoryDiscovery,
    AdversarialDiscovery,
    MetaLearningDiscovery
)


class TestAutoFixer(unittest.TestCase):
    """Test automatic error fixing."""
    
    def setUp(self):
        self.fixer = AutoFixer()
    
    def test_import_error_detection(self):
        """Test detection of import errors."""
        failure = FailureRecord(
            test_name="test1",
            timestamp=datetime.now(),
            error_type="ImportError",
            error_message="No module named 'nonexistent_module'",
            traceback_str="...",
        )
        
        success, message = self.fixer.attempt_fix(failure, {})
        # Should attempt to fix (might fail if module doesn't exist)
        self.assertIn("module", message.lower())
    
    def test_value_error_fix(self):
        """Test fixing value errors."""
        config = ExecutionConfig(name="test", min_sample_size=100)
        
        failure = FailureRecord(
            test_name="test1",
            timestamp=datetime.now(),
            error_type="ValueError",
            error_message="sample size too small",
            traceback_str="...",
        )
        
        success, message = self.fixer.attempt_fix(failure, {'config': config})
        # Should reduce sample size
        self.assertTrue(config.min_sample_size < 100 or not success)


class TestStatisticalRigor(unittest.TestCase):
    """Test statistical validation methods."""
    
    def setUp(self):
        self.validator = StatisticalRigor()
    
    def test_pbo_calculation(self):
        """Test PBO calculation."""
        # Create synthetic data
        np.random.seed(42)
        
        # Good strategy: IS and OOS similar
        returns_is = np.random.normal(0.001, 0.02, 100)
        returns_oos = np.random.normal(0.001, 0.02, 100)
        
        pbo = self.validator.calculate_pbo(returns_is, returns_oos, n_trials=100)
        
        # PBO should be relatively low for good strategy
        self.assertIsInstance(pbo, float)
        self.assertGreaterEqual(pbo, 0.0)
        self.assertLessEqual(pbo, 1.0)
    
    def test_pbo_overfitting_detection(self):
        """Test PBO detects overfitting."""
        np.random.seed(42)
        
        # Overfit strategy: great IS, poor OOS
        returns_is = np.random.normal(0.01, 0.02, 100)  # High returns
        returns_oos = np.random.normal(-0.005, 0.02, 100)  # Negative returns
        
        pbo = self.validator.calculate_pbo(returns_is, returns_oos, n_trials=100)
        
        # PBO should be high for overfit strategy
        self.assertGreater(pbo, 0.3)
    
    def test_cpcv_splits(self):
        """Test CPCV split generation."""
        data = pd.DataFrame({
            'close': np.random.randn(1000)
        })
        
        splits = self.validator.combinatorially_purged_cv(data, n_splits=5)
        
        # Should have 5 splits
        self.assertEqual(len(splits), 5)
        
        # Each split should have train and test
        for train_idx, test_idx in splits:
            self.assertGreater(len(train_idx), 0)
            self.assertGreater(len(test_idx), 0)
            
            # No overlap
            self.assertEqual(len(set(train_idx) & set(test_idx)), 0)
    
    def test_bootstrap_confidence_interval(self):
        """Test bootstrap CI calculation."""
        np.random.seed(42)
        data = np.random.normal(10, 2, 100)
        
        lower, upper = self.validator.bootstrap_confidence_interval(
            data,
            statistic_func=np.mean,
            n_bootstrap=1000,
            confidence_level=0.95
        )
        
        # CI should contain true mean (10)
        self.assertLess(lower, 10)
        self.assertGreater(upper, 10)
        
        # CI should be reasonable width
        self.assertLess(upper - lower, 2)
    
    def test_monte_carlo_permutation(self):
        """Test Monte Carlo permutation test."""
        np.random.seed(42)
        
        # Two samples from same distribution
        sample1 = np.random.normal(0, 1, 100)
        sample2 = np.random.normal(0, 1, 100)
        
        p_value = self.validator.monte_carlo_permutation_test(
            sample1, sample2, n_permutations=1000
        )
        
        # Should not be significant
        self.assertGreater(p_value, 0.05)
        
        # Two samples from different distributions
        sample1 = np.random.normal(0, 1, 100)
        sample2 = np.random.normal(2, 1, 100)
        
        p_value = self.validator.monte_carlo_permutation_test(
            sample1, sample2, n_permutations=1000
        )
        
        # Should be significant
        self.assertLess(p_value, 0.05)


class TestDiscoveryMethods(unittest.TestCase):
    """Test discovery method implementations."""
    
    def setUp(self):
        # Create synthetic data
        np.random.seed(42)
        n = 1000
        
        # Price data with trend
        prices = 100 + np.cumsum(np.random.randn(n) * 0.01)
        
        self.data = pd.DataFrame({
            'close': prices,
            'open': prices * 0.99,
            'high': prices * 1.01,
            'low': prices * 0.98,
            'volume': np.random.randint(1000, 10000, n),
            'feature1': np.random.randn(n),
            'feature2': np.random.randn(n),
            'feature3': np.random.randn(n),
        })
    
    def test_hidden_dimension_discovery(self):
        """Test hidden dimension discovery."""
        method = HiddenDimensionDiscovery()
        
        config = {
            'methods': ['pca'],
            'latent_dims': [2, 3]
        }
        
        result = method.discover(self.data, config)
        
        self.assertEqual(result.method_name, "HiddenDimension")
        self.assertGreater(len(result.discovered_patterns), 0)
        self.assertIsInstance(result.p_value, float)
    
    def test_chaos_theory_discovery(self):
        """Test chaos theory analysis."""
        method = ChaosTheoryDiscovery()
        
        result = method.discover(self.data, {})
        
        self.assertEqual(result.method_name, "ChaosTheory")
        self.assertGreater(len(result.discovered_patterns), 0)
        
        # Should have Lyapunov, Hurst, entropy
        metrics = [p['metric'] for p in result.discovered_patterns]
        self.assertIn('lyapunov_exponent', metrics)
        self.assertIn('hurst_exponent', metrics)
        self.assertIn('approximate_entropy', metrics)
    
    def test_adversarial_discovery(self):
        """Test adversarial pattern discovery."""
        method = AdversarialDiscovery()
        
        result = method.discover(self.data, {})
        
        self.assertEqual(result.method_name, "Adversarial")
        # May or may not find patterns depending on data
        self.assertIsInstance(result.p_value, float)
    
    def test_meta_learning_discovery(self):
        """Test meta-learning discovery."""
        method = MetaLearningDiscovery()
        
        result = method.discover(self.data, {})
        
        self.assertEqual(result.method_name, "MetaLearning")
        self.assertGreater(len(result.discovered_patterns), 0)
    
    def test_discovery_runner(self):
        """Test discovery method runner."""
        runner = DiscoveryMethodRunner()
        
        results = runner.run_all_discoveries(
            self.data,
            methods=['hidden_dimensions', 'chaos_theory'],
            config={
                'hidden_dimensions': {'methods': ['pca'], 'latent_dims': [2]},
                'chaos_theory': {}
            }
        )
        
        self.assertIn('hidden_dimensions', results)
        self.assertIn('chaos_theory', results)
        
        self.assertEqual(results['hidden_dimensions'].method_name, "HiddenDimension")
        self.assertEqual(results['chaos_theory'].method_name, "ChaosTheory")


class TestTestExecutor(unittest.TestCase):
    """Test test executor."""
    
    def test_executor_initialization(self):
        """Test executor initialization."""
        config = ExecutionConfig(
            name="test_execution",
            max_retries=2,
            auto_fix_enabled=True
        )
        
        executor = TestExecutor(config)
        
        self.assertEqual(executor.config.name, "test_execution")
        self.assertEqual(executor.config.max_retries, 2)
        self.assertTrue(executor.config.auto_fix_enabled)
    
    def test_simple_execution(self):
        """Test simple test execution."""
        config = ExecutionConfig(name="test", max_retries=1)
        executor = TestExecutor(config)
        
        # Define simple test functions
        def test1():
            return {'success': True, 'value': 42}
        
        def test2():
            return {'success': True, 'value': 100}
        
        results = executor.execute_test_suite(
            test_functions=[test1, test2],
            test_names=['test1', 'test2']
        )
        
        self.assertEqual(len(results), 2)
        self.assertTrue(results['test1']['success'])
        self.assertTrue(results['test2']['success'])
    
    def test_retry_on_failure(self):
        """Test retry logic."""
        config = ExecutionConfig(
            name="test",
            max_retries=3,
            auto_fix_enabled=False,
            continue_on_failure=True
        )
        executor = TestExecutor(config)
        
        attempt_count = [0]
        
        def failing_test():
            attempt_count[0] += 1
            if attempt_count[0] < 3:
                raise ValueError("Test failure")
            return {'success': True}
        
        results = executor.execute_test_suite(
            test_functions=[failing_test],
            test_names=['failing_test']
        )
        
        # Should have retried
        self.assertGreaterEqual(attempt_count[0], 2)


if __name__ == '__main__':
    unittest.main()
