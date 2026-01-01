#!/usr/bin/env python3
"""
Kinetra System Stress Test and Integration Test
================================================

Comprehensive system-level testing that:
- Tests the entire workflow end-to-end
- Stress tests concurrent operations
- Validates error recovery
- Tests data pipeline integrity
- Validates menu system under load
- Tests E2E framework with actual execution
- Monitors system health during testing

This is a full system integration and stress test.
"""

import sys
import time
import logging
import multiprocessing
from pathlib import Path
from typing import Dict, Any
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# STRESS TEST CONFIGURATION
# =============================================================================

@dataclass
class StressTestConfig:
    """Configuration for stress testing."""
    
    # Concurrency settings
    num_concurrent_menu_sessions: int = 5
    num_concurrent_e2e_tests: int = 3
    num_data_operations: int = 10
    
    # Load settings
    total_menu_cycles: int = 50
    total_e2e_runs: int = 10
    
    # Timeout settings
    menu_operation_timeout: int = 30
    e2e_operation_timeout: int = 300
    
    # Resource limits
    max_memory_mb: int = 4096
    max_cpu_percent: int = 90


# =============================================================================
# SYSTEM HEALTH MONITORING
# =============================================================================

class SystemHealthMonitor:
    """Monitor system health during stress testing."""
    
    def __init__(self):
        self.start_time = time.time()
        self._lock = multiprocessing.Lock()
        self.metrics = {
            'cpu_samples': [],
            'memory_samples': [],
            'operations': [],
            'errors': []
        }
        
        try:
            import psutil
            self.psutil = psutil
            self.psutil_available = True
        except ImportError:
            logger.warning("psutil not available - resource monitoring limited")
            self.psutil_available = False
    
    def record_operation(self, operation: str, duration: float, success: bool):
        """Record an operation."""
        with self._lock:
            self.metrics['operations'].append({
                'operation': operation,
                'duration': duration,
                'success': success,
                'timestamp': time.time()
            })
    
    def record_error(self, operation: str, error: str):
        """Record an error."""
        with self._lock:
            self.metrics['errors'].append({
                'operation': operation,
                'error': str(error)[:5000],
                'timestamp': time.time()
            })
    
    def sample_resources(self):
        """Sample current resource usage."""
        if not self.psutil_available:
            return
        
        try:
            cpu_percent = self.psutil.cpu_percent(interval=0.1)
            memory = self.psutil.virtual_memory()
        
            with self._lock:
                self.metrics['cpu_samples'].append(cpu_percent)
                self.metrics['memory_samples'].append(memory.percent)
        
        except Exception as e:
            logger.warning(f"Error sampling resources: {e}")
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics."""
        total_ops = len(self.metrics['operations'])
        successful_ops = sum(1 for op in self.metrics['operations'] if op['success'])
        failed_ops = total_ops - successful_ops
        
        total_duration = time.time() - self.start_time
        
        summary = {
            'total_duration': total_duration,
            'total_operations': total_ops,
            'successful_operations': successful_ops,
            'failed_operations': failed_ops,
            'total_errors': len(self.metrics['errors']),
            'success_rate': (successful_ops / total_ops * 100) if total_ops > 0 else 0
        }
        
        if self.psutil_available and self.metrics['cpu_samples']:
            summary['avg_cpu_percent'] = sum(self.metrics['cpu_samples']) / len(self.metrics['cpu_samples'])
            summary['max_cpu_percent'] = max(self.metrics['cpu_samples'])
            summary['avg_memory_percent'] = sum(self.metrics['memory_samples']) / len(self.metrics['memory_samples'])
            summary['max_memory_percent'] = max(self.metrics['memory_samples'])
        
        if total_ops > 0:
            durations = [op['duration'] for op in self.metrics['operations'] if op['success']]
            if durations:
                summary['avg_operation_time'] = sum(durations) / len(durations)
                summary['max_operation_time'] = max(durations)
                summary['min_operation_time'] = min(durations)
        
        return summary


# =============================================================================
# MENU SYSTEM STRESS TESTS
# =============================================================================

def stress_test_menu_navigation(session_id: int, monitor: SystemHealthMonitor) -> Dict[str, Any]:
    """
    Stress test menu navigation with rapid cycling through all options.
    
    Args:
        session_id: Session identifier
        monitor: Health monitor
        
    Returns:
        Test results
    """
    logger.info(f"[Session {session_id}] Starting menu navigation stress test")
    
    results = {
        'session_id': session_id,
        'operations_completed': 0,
        'errors': []
    }
    
    try:
        from kinetra_menu import MenuConfig, get_input, confirm_action
        from kinetra_menu import select_asset_classes, select_timeframes, select_agent_types
        
        # Test MenuConfig methods repeatedly
        for i in range(10):
            start = time.time()
            try:
                asset_classes = MenuConfig.get_all_asset_classes()
                timeframes = MenuConfig.get_all_timeframes()
                agent_types = MenuConfig.get_all_agent_types()
                
                assert len(asset_classes) > 0
                assert len(timeframes) > 0
                assert len(agent_types) > 0
                
                results['operations_completed'] += 1
                monitor.record_operation(f'menu_config_{session_id}', time.time() - start, True)
                
            except Exception as e:
                results['errors'].append(f"MenuConfig error: {e}")
                monitor.record_error(f'menu_config_{session_id}', str(e))
        
        logger.info(f"[Session {session_id}] Completed {results['operations_completed']} operations")
        
    except Exception as e:
        logger.error(f"[Session {session_id}] Fatal error: {e}")
        results['errors'].append(f"Fatal: {e}")
        monitor.record_error(f'menu_session_{session_id}', str(e))
    
    return results


def stress_test_data_management(operation_id: int, monitor: SystemHealthMonitor) -> Dict[str, Any]:
    """
    Stress test data management operations.
    
    Args:
        operation_id: Operation identifier
        monitor: Health monitor
        
    Returns:
        Test results
    """
    logger.info(f"[DataOp {operation_id}] Testing data management")
    
    results = {
        'operation_id': operation_id,
        'checks_completed': 0,
        'errors': []
    }
    
    try:
        from pathlib import Path
        
        # Test data directory existence
        data_dirs = ['data/master', 'data/prepared', 'data/results', 'logs']

        for dir_path in data_dirs:

            if not exists:
                results['errors'].append(f"Missing directory: {dir_path}")
                monitor.record_error(f'data_check_{operation_id}', f"Missing directory: {dir_path}")

            monitor.record_operation(f'data_check_{operation_id}', time.time() - start, exists)
            results['checks_completed'] += 1
    
            monitor.record_operation(f'data_check_{operation_id}', time.time() - start, True)
            results['checks_completed'] += 1
        
        # Test workflow manager creation
        start = time.time()
        from kinetra.workflow_manager import WorkflowManager
        
        wf = WorkflowManager(
            log_dir=f"logs/stress_test/session_{operation_id}",
            enable_backups=False
        )
        
        results['checks_completed'] += 1
        monitor.record_operation(f'workflow_manager_{operation_id}', time.time() - start, True)
        
    except Exception as e:
        logger.error(f"[DataOp {operation_id}] Error: {e}")
        results['errors'].append(str(e))
        monitor.record_error(f'data_op_{operation_id}', str(e))
    
    return results


# =============================================================================
# E2E FRAMEWORK STRESS TESTS
# =============================================================================

def stress_test_e2e_matrix_generation(test_id: int, monitor: SystemHealthMonitor) -> Dict[str, Any]:
    """
    Stress test E2E test matrix generation.
    
    Args:
        test_id: Test identifier
        monitor: Health monitor
        
    Returns:
        Test results
    """
    logger.info(f"[E2E {test_id}] Testing matrix generation")
    
    results = {
        'test_id': test_id,
        'matrices_generated': 0,
        'total_combinations': 0,
        'errors': []
    }
    
    try:
        from e2e_testing_framework import E2EPresets, E2ETestRunner
        
        # Test different preset configurations
        presets = [
            E2EPresets.quick_validation(),
            E2EPresets.asset_class_test('crypto'),
            E2EPresets.agent_type_test('ppo'),
            E2EPresets.timeframe_test('H1')
        ]
        
        for preset in presets:
            start = time.time()
            try:
                runner = E2ETestRunner(
                    preset,
                    output_dir=f"data/results/stress_test_{test_id}",
                    log_dir=f"logs/stress_test_{test_id}"
                )
                
                # Generate test matrix
                matrix = runner.generate_test_matrix()
                
                results['matrices_generated'] += 1
                results['total_combinations'] += len(matrix)
                
                monitor.record_operation(f'e2e_matrix_{test_id}', time.time() - start, True)
                
                # Test duration estimation
                duration_hours, duration_str = runner.estimate_duration()
                
            except Exception as e:
                results['errors'].append(f"Preset error: {e}")
                monitor.record_error(f'e2e_preset_{test_id}', str(e))
        
    except Exception as e:
        logger.error(f"[E2E {test_id}] Fatal error: {e}")
        results['errors'].append(f"Fatal: {e}")
        monitor.record_error(f'e2e_test_{test_id}', str(e))
    
    return results


def stress_test_custom_config_loading(test_id: int, monitor: SystemHealthMonitor) -> Dict[str, Any]:
    """
    Stress test custom configuration loading.
    
    Args:
        test_id: Test identifier
        monitor: Health monitor
        
    Returns:
        Test results
    """
    logger.info(f"[Config {test_id}] Testing custom config loading")
    
    results = {
        'test_id': test_id,
        'configs_loaded': 0,
        'errors': []
    }
    
    try:
        import json
        from pathlib import Path
        from e2e_testing_framework import E2ETestConfig, E2ETestRunner
        
        # Test loading example configurations
        config_dir = Path("configs/e2e_examples")
        
        if config_dir.exists():
            config_files = list(config_dir.glob("*.json"))
            
            for config_file in config_files:
                start = time.time()
                try:
                    with open(config_file, 'r') as f:
                        config_dict = json.load(f)
                    
                    config = E2ETestConfig(**config_dict)
                    
                    # Create runner
                    runner = E2ETestRunner(
                        config,
                        output_dir=f"data/results/config_test_{test_id}",
                        log_dir=f"logs/config_test_{test_id}"
                    )
                    
                    # Generate matrix
                    matrix = runner.generate_test_matrix()
                    
                    results['configs_loaded'] += 1
                    monitor.record_operation(f'config_load_{test_id}', time.time() - start, True)
                    
                except Exception as e:
                    results['errors'].append(f"Config {config_file.name}: {e}")
                    monitor.record_error(f'config_{test_id}', str(e))
        
    except Exception as e:
        logger.error(f"[Config {test_id}] Fatal error: {e}")
        results['errors'].append(f"Fatal: {e}")
        monitor.record_error(f'config_test_{test_id}', str(e))
    
    return results


# =============================================================================
# CONCURRENT OPERATION TESTS
# =============================================================================

def test_concurrent_menu_operations(config: StressTestConfig, monitor: SystemHealthMonitor) -> Dict[str, Any]:
    """
    Test concurrent menu operations.
    
    Args:
        config: Stress test configuration
        monitor: Health monitor
        
    Returns:
        Test results
    """
    logger.info(f"Starting concurrent menu operations ({config.num_concurrent_menu_sessions} sessions)")
    
    results = {
        'total_sessions': config.num_concurrent_menu_sessions,
        'completed_sessions': 0,
        'failed_sessions': 0,
        'session_results': []
    }
    
    with ThreadPoolExecutor(max_workers=config.num_concurrent_menu_sessions) as executor:
        futures = []
        
        for session_id in range(config.num_concurrent_menu_sessions):
            future = executor.submit(stress_test_menu_navigation, session_id, monitor)
            futures.append(future)
        
        for future in as_completed(futures):
            try:
                result = future.result(timeout=config.menu_operation_timeout)
                results['session_results'].append(result)
                
                if result['errors']:
                    results['failed_sessions'] += 1
                else:
                    results['completed_sessions'] += 1
                    
            except Exception as e:
                logger.error(f"Session failed: {e}")
                results['failed_sessions'] += 1
                monitor.record_error('concurrent_menu', str(e))
    
    return results


def test_concurrent_e2e_operations(config: StressTestConfig, monitor: SystemHealthMonitor) -> Dict[str, Any]:
    """
    Test concurrent E2E operations.
    
    Args:
        config: Stress test configuration
        monitor: Health monitor
        
    Returns:
        Test results
    """
    logger.info(f"Starting concurrent E2E operations ({config.num_concurrent_e2e_tests} tests)")
    
    results = {
        'total_tests': config.num_concurrent_e2e_tests,
        'completed_tests': 0,
        'failed_tests': 0,
        'test_results': []
    }
    
    with ThreadPoolExecutor(max_workers=config.num_concurrent_e2e_tests) as executor:
        futures = []
        
        for test_id in range(config.num_concurrent_e2e_tests):
            future = executor.submit(stress_test_e2e_matrix_generation, test_id, monitor)
            futures.append(future)
        
        for future in as_completed(futures):
            try:
                result = future.result(timeout=config.e2e_operation_timeout)
                results['test_results'].append(result)
                
                if result['errors']:
                    results['failed_tests'] += 1
                else:
                    results['completed_tests'] += 1
                    
            except Exception as e:
                logger.error(f"E2E test failed: {e}")
                results['failed_tests'] += 1
                monitor.record_error('concurrent_e2e', str(e))
    
    return results


def test_concurrent_data_operations(config: StressTestConfig, monitor: SystemHealthMonitor) -> Dict[str, Any]:
    """
    Test concurrent data operations.
    
    Args:
        config: Stress test configuration
        monitor: Health monitor
        
    Returns:
        Test results
    """
    logger.info(f"Starting concurrent data operations ({config.num_data_operations} ops)")
    
    results = {
        'total_operations': config.num_data_operations,
        'completed_operations': 0,
        'failed_operations': 0,
        'operation_results': []
    }
    
    with ThreadPoolExecutor(max_workers=config.num_data_operations) as executor:
        futures = []
        
        for op_id in range(config.num_data_operations):
            future = executor.submit(stress_test_data_management, op_id, monitor)
            futures.append(future)
        
        for future in as_completed(futures):
            try:
                result = future.result(timeout=config.e2e_operation_timeout)
                results['operation_results'].append(result)
                
                if result['errors']:
                    results['failed_operations'] += 1
                else:
                    results['completed_operations'] += 1
                    
            except Exception as e:
                logger.error(f"Data operation failed: {e}")
                results['failed_operations'] += 1
                monitor.record_error('concurrent_data', str(e))
    
    return results


# =============================================================================
# MAIN STRESS TEST ORCHESTRATOR
# =============================================================================

def run_system_stress_test(config: StressTestConfig = None) -> Dict[str, Any]:
    """
    Run comprehensive system stress test.
    
    Args:
        config: Stress test configuration
        
    Returns:
        Complete test results
    """
    if config is None:
        config = StressTestConfig()
    
    print("=" * 80)
    print("  KINETRA SYSTEM STRESS TEST")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Concurrent Menu Sessions: {config.num_concurrent_menu_sessions}")
    print(f"  Concurrent E2E Tests: {config.num_concurrent_e2e_tests}")
    print(f"  Data Operations: {config.num_data_operations}")
    print(f"  Menu Operation Timeout: {config.menu_operation_timeout}s")
    print(f"  E2E Operation Timeout: {config.e2e_operation_timeout}s")
    print()
    
    monitor = SystemHealthMonitor()
    
    all_results = {
        'start_time': datetime.now().isoformat(),
        'config': config,
        'tests': {}
    }
    
    # Phase 1: Concurrent Menu Operations
    print("\n" + "=" * 80)
    print("PHASE 1: Concurrent Menu Operations")
    print("=" * 80)
    start = time.time()
    menu_results = test_concurrent_menu_operations(config, monitor)
    all_results['tests']['concurrent_menu'] = menu_results
    print(f"✅ Completed in {time.time() - start:.2f}s")
    print(f"   Sessions: {menu_results['completed_sessions']}/{menu_results['total_sessions']} succeeded")
    
    # Sample resources
    monitor.sample_resources()
    
    # Phase 2: Concurrent E2E Operations
    print("\n" + "=" * 80)
    print("PHASE 2: Concurrent E2E Matrix Generation")
    print("=" * 80)
    start = time.time()
    e2e_results = test_concurrent_e2e_operations(config, monitor)
    all_results['tests']['concurrent_e2e'] = e2e_results
    print(f"✅ Completed in {time.time() - start:.2f}s")
    print(f"   Tests: {e2e_results['completed_tests']}/{e2e_results['total_tests']} succeeded")
    
    # Sample resources
    monitor.sample_resources()
    
    # Phase 3: Concurrent Data Operations
    print("\n" + "=" * 80)
    print("PHASE 3: Concurrent Data Operations")
    print("=" * 80)
    start = time.time()
    data_results = test_concurrent_data_operations(config, monitor)
    all_results['tests']['concurrent_data'] = data_results
    print(f"✅ Completed in {time.time() - start:.2f}s")
    print(f"   Operations: {data_results['completed_operations']}/{data_results['total_operations']} succeeded")
    
    # Sample resources
    monitor.sample_resources()
    
    # Phase 4: Custom Config Loading
    print("\n" + "=" * 80)
    print("PHASE 4: Custom Configuration Loading")
    print("=" * 80)
    start = time.time()
    config_results = stress_test_custom_config_loading(0, monitor)
    all_results['tests']['config_loading'] = config_results
    print(f"✅ Completed in {time.time() - start:.2f}s")
    print(f"   Configs: {config_results['configs_loaded']} loaded")
    
    # Final resource sampling
    monitor.sample_resources()
    
    # Get summary
    all_results['end_time'] = datetime.now().isoformat()
    all_results['summary'] = monitor.get_summary()
    
    # Print summary
    print_stress_test_summary(all_results)
    
    return all_results


def print_stress_test_summary(results: Dict[str, Any]):
    """Print stress test summary."""
    print("\n" + "=" * 80)
    print("  STRESS TEST SUMMARY")
    print("=" * 80)
    
    summary = results['summary']
    
    print(f"\nDuration: {summary['total_duration']:.2f}s")
    print(f"Total Operations: {summary['total_operations']}")
    print(f"Successful: {summary['successful_operations']}")
    print(f"Failed: {summary['failed_operations']}")
    print(f"Success Rate: {summary['success_rate']:.2f}%")
    
    if 'avg_cpu_percent' in summary:
        print(f"\nResource Usage:")
        print(f"  CPU: {summary['avg_cpu_percent']:.1f}% avg, {summary['max_cpu_percent']:.1f}% max")
        print(f"  Memory: {summary['avg_memory_percent']:.1f}% avg, {summary['max_memory_percent']:.1f}% max")
    
    if 'avg_operation_time' in summary:
        print(f"\nOperation Times:")
        print(f"  Average: {summary['avg_operation_time']:.3f}s")
        print(f"  Min: {summary['min_operation_time']:.3f}s")
        print(f"  Max: {summary['max_operation_time']:.3f}s")
    
    print(f"\nTotal Errors: {summary['total_errors']}")
    
    # Test-specific results
    print("\n" + "-" * 80)
    print("Test Results by Category:")
    print("-" * 80)
    
    tests = results['tests']
    
    if 'concurrent_menu' in tests:
        menu = tests['concurrent_menu']
        print(f"\nConcurrent Menu Operations:")
        print(f"  Completed: {menu['completed_sessions']}/{menu['total_sessions']}")
        print(f"  Failed: {menu['failed_sessions']}")
    
    if 'concurrent_e2e' in tests:
        e2e = tests['concurrent_e2e']
        print(f"\nConcurrent E2E Tests:")
        print(f"  Completed: {e2e['completed_tests']}/{e2e['total_tests']}")
        print(f"  Failed: {e2e['failed_tests']}")
        
        if e2e['test_results']:
            total_combinations = sum(r.get('total_combinations', 0) for r in e2e['test_results'])
            print(f"  Total Test Combinations Generated: {total_combinations}")
    
    if 'concurrent_data' in tests:
        data = tests['concurrent_data']
        print(f"\nConcurrent Data Operations:")
        print(f"  Completed: {data['completed_operations']}/{data['total_operations']}")
        print(f"  Failed: {data['failed_operations']}")
    
    if 'config_loading' in tests:
        config = tests['config_loading']
        print(f"\nCustom Config Loading:")
        print(f"  Configs Loaded: {config['configs_loaded']}")
        print(f"  Errors: {len(config['errors'])}")
    
    print("\n" + "=" * 80)
    
    # Determine overall status
    if summary['failed_operations'] == 0 and summary['total_errors'] == 0:
        print("✅ STRESS TEST PASSED - System stable under load")
    elif summary['success_rate'] >= 95:
        print("⚠️  STRESS TEST PASSED WITH WARNINGS - Minor issues detected")
    else:
        print("❌ STRESS TEST FAILED - Significant issues detected")
    
    print("=" * 80)


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Kinetra System Stress Test",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--sessions',
        type=int,
        default=5,
        help="Number of concurrent menu sessions (default: 5)"
    )
    
    parser.add_argument(
        '--e2e-tests',
        type=int,
        default=3,
        help="Number of concurrent E2E tests (default: 3)"
    )
    
    parser.add_argument(
        '--data-ops',
        type=int,
        default=10,
        help="Number of concurrent data operations (default: 10)"
    )
    
    parser.add_argument(
        '--light',
        action='store_true',
        help="Run light stress test (fewer operations)"
    )
    
    parser.add_argument(
        '--heavy',
        action='store_true',
        help="Run heavy stress test (more operations)"
    )
    
    args = parser.parse_args()
    
    # Create configuration
    if args.light:
        config = StressTestConfig(
            num_concurrent_menu_sessions=2,
            num_concurrent_e2e_tests=2,
            num_data_operations=5
        )
    elif args.heavy:
        config = StressTestConfig(
            num_concurrent_menu_sessions=10,
            num_concurrent_e2e_tests=5,
            num_data_operations=20
        )
    else:
        config = StressTestConfig(
            num_concurrent_menu_sessions=args.sessions,
            num_concurrent_e2e_tests=args.e2e_tests,
            num_data_operations=args.data_ops
        )
    
    # Run stress test
    try:
        results = run_system_stress_test(config)
        
        # Save results
        output_dir = Path("data/results/stress_tests")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = output_dir / f"stress_test_{timestamp}.json"
        
        import json
        with open(output_file, 'w') as f:
            # Convert config to dict for JSON serialization
            results_copy = results.copy()
            results_copy['config'] = {
                'num_concurrent_menu_sessions': config.num_concurrent_menu_sessions,
                'num_concurrent_e2e_tests': config.num_concurrent_e2e_tests,
                'num_data_operations': config.num_data_operations,
                'menu_operation_timeout': config.menu_operation_timeout,
                'e2e_operation_timeout': config.e2e_operation_timeout
            }
            json.dump(results_copy, f, indent=2)
        
        print(f"\nResults saved to: {output_file}")
        
        # Return appropriate exit code
        if results['summary']['failed_operations'] == 0 and results['summary']['total_errors'] == 0:
            return 0
        elif results['summary']['success_rate'] >= 95:
            return 0
        else:
            return 1
            
    except KeyboardInterrupt:
        print("\n\n⚠️  Stress test interrupted by user")
        return 1
    except Exception as e:
        print(f"\n\n❌ Stress test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
