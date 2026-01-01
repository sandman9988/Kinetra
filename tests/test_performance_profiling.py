#!/usr/bin/env python3
"""
Kinetra Performance Profiling and Bottleneck Analysis
=====================================================

Comprehensive performance analysis tool that:
- Profiles menu system operations
- Profiles E2E framework operations
- Identifies bottlenecks and slow paths
- Measures memory allocation patterns
- Analyzes I/O operations
- Provides optimization recommendations

Usage:
    python tests/test_performance_profiling.py --full
    python tests/test_performance_profiling.py --menu-only
    python tests/test_performance_profiling.py --e2e-only
"""

import sys
import time
import cProfile
import pstats
import io
import tracemalloc
from pathlib import Path
from typing import Dict, List, Any, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime
import logging

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# PERFORMANCE METRICS
# =============================================================================

@dataclass
class PerformanceMetrics:
    """Performance metrics for an operation."""
    operation_name: str
    execution_time: float
    cpu_time: float
    memory_allocated: int = 0
    memory_peak: int = 0
    call_count: int = 0
    bottlenecks: List[str] = field(default_factory=list)
    optimization_suggestions: List[str] = field(default_factory=list)


@dataclass
class ProfilingReport:
    """Complete profiling report."""
    timestamp: str
    total_time: float
    operations: List[PerformanceMetrics] = field(default_factory=list)
    hotspots: Dict[str, float] = field(default_factory=dict)
    memory_profile: Dict[str, Any] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)


# =============================================================================
# PROFILING UTILITIES
# =============================================================================

class PerformanceProfiler:
    """Performance profiler with bottleneck detection."""
    
    def __init__(self):
        self.metrics: List[PerformanceMetrics] = []
        self.profiler = cProfile.Profile()
        self.memory_tracking = False
        
    def profile_operation(
        self,
        operation_name: str,
        operation_func: Callable,
        *args,
        **kwargs
    ) -> Tuple[Any, PerformanceMetrics]:
        """
        Profile a single operation.
        
        Args:
            operation_name: Name of the operation
            operation_func: Function to profile
            *args: Arguments for the function
            **kwargs: Keyword arguments for the function
            
        Returns:
            Tuple of (result, metrics)
        """
        logger.info(f"Profiling: {operation_name}")
        
        # Start memory tracking
        tracemalloc.start()
        
        # Profile execution
        start_time = time.time()
        self.profiler.enable()
        
        try:
            result = operation_func(*args, **kwargs)
            success = True
        except Exception as e:
            logger.error(f"Error in {operation_name}: {e}")
            result = None
            success = False
        
        self.profiler.disable()
        execution_time = time.time() - start_time
        
        # Get memory stats
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        # Get CPU time from profiler
        stats_stream = io.StringIO()
        stats = pstats.Stats(self.profiler, stream=stats_stream)
        stats.sort_stats('cumulative')

        # Extract CPU time (total time recorded by profiler for this run)
        cpu_time = stats.total_tt

        # Create metrics
        metrics = PerformanceMetrics(
            operation_name=operation_name,
            execution_time=execution_time,
            cpu_time=cpu_time,
            memory_allocated=current,
            memory_peak=peak,
            call_count=stats.total_calls
        )
        
        # Analyze for bottlenecks
        self._analyze_bottlenecks(metrics, stats)
        
        self.metrics.append(metrics)
        
        return result, metrics
    
    def _analyze_bottlenecks(self, metrics: PerformanceMetrics, stats: pstats.Stats):
        """Analyze profiling stats for bottlenecks."""
        
        # Get top time-consuming functions
        stats_list = sorted(
            [(key, value) for key, value in stats.stats.items()],
            key=lambda x: x[1][3],  # Sort by cumulative time
            reverse=True
        )[:10]
        
        # Identify bottlenecks
        for func_key, func_stats in stats_list:
            filename, line, func_name = func_key
            cumtime = func_stats[3]
            
            # Check if this function is a bottleneck (>10% of total time)
            if cumtime > metrics.cpu_time * 0.1:
                bottleneck = f"{func_name} in {Path(filename).name}:{line} ({cumtime:.3f}s)"
                metrics.bottlenecks.append(bottleneck)
        
        # Generate optimization suggestions
        self._generate_suggestions(metrics)
    
    def _generate_suggestions(self, metrics: PerformanceMetrics):
        """Generate optimization suggestions."""
        
        # Memory-based suggestions
        if metrics.memory_peak > 100 * 1024 * 1024:  # > 100MB
            metrics.optimization_suggestions.append(
                "High memory usage detected - consider streaming or pagination"
            )
        
        # CPU-based suggestions
        if metrics.cpu_time > metrics.execution_time * 0.8:
            metrics.optimization_suggestions.append(
                "CPU-bound operation - consider caching or parallel processing"
            )
        
        # I/O-based suggestions
        if metrics.execution_time > metrics.cpu_time * 2:
            metrics.optimization_suggestions.append(
                "I/O-bound operation - consider async I/O or connection pooling"
            )
        
        # Call count suggestions
        if metrics.call_count > 10000:
            metrics.optimization_suggestions.append(
                "High function call count - consider reducing abstraction layers"
            )
    
    def generate_report(self) -> ProfilingReport:
        """Generate comprehensive profiling report."""
        
        total_time = sum(m.execution_time for m in self.metrics)
        
        # Identify hotspots
        hotspots = {}
        for metric in self.metrics:
            if metric.execution_time > total_time * 0.05:  # > 5% of total time
                hotspots[metric.operation_name] = metric.execution_time
        
        # Memory profile summary
        memory_profile = {
            'total_allocated': sum(m.memory_allocated for m in self.metrics),
            'peak_usage': max((m.memory_peak for m in self.metrics), default=0),
            'avg_per_operation': sum(m.memory_allocated for m in self.metrics) / len(self.metrics) if self.metrics else 0
        }
        
        # Generate overall recommendations
        recommendations = self._generate_overall_recommendations()
        
        return ProfilingReport(
            timestamp=datetime.now().isoformat(),
            total_time=total_time,
            operations=self.metrics,
            hotspots=hotspots,
            memory_profile=memory_profile,
            recommendations=recommendations
        )
    
    def _generate_overall_recommendations(self) -> List[str]:
        """Generate overall optimization recommendations."""
        recommendations = []
        
        # Check for repeated operations
        operation_counts = {}
        for metric in self.metrics:
            base_name = metric.operation_name.split('_')[0]
            operation_counts[base_name] = operation_counts.get(base_name, 0) + 1
        
        for op, count in operation_counts.items():
            if count > 5:
                recommendations.append(
                    f"Operation '{op}' repeated {count} times - consider batching or caching"
                )
        
        # Check total memory usage
        total_memory = sum(m.memory_peak for m in self.metrics)
        if total_memory > 500 * 1024 * 1024:  # > 500MB
            recommendations.append(
                "High total memory usage - consider implementing memory pooling"
            )
        
        # Check for slow operations
        slow_ops = [m for m in self.metrics if m.execution_time > 1.0]
        if slow_ops:
            recommendations.append(
                f"{len(slow_ops)} operations taking >1s - review for optimization"
            )
        
        return recommendations


# =============================================================================
# MENU SYSTEM PROFILING
# =============================================================================

def profile_menu_config_operations(profiler: PerformanceProfiler):
    """Profile menu configuration operations."""
    logger.info("Profiling menu configuration operations...")
    
    from kinetra_menu import MenuConfig
    
    # Profile getting asset classes
    profiler.profile_operation(
        "menu_get_asset_classes",
        MenuConfig.get_all_asset_classes
    )
    
    # Profile getting timeframes
    profiler.profile_operation(
        "menu_get_timeframes",
        MenuConfig.get_all_timeframes
    )
    
    # Profile getting agent types
    profiler.profile_operation(
        "menu_get_agent_types",
        MenuConfig.get_all_agent_types
    )
    
    # Repeated calls to test caching opportunities
    for i in range(10):
        profiler.profile_operation(
            f"menu_get_asset_classes_repeat_{i}",
            MenuConfig.get_all_asset_classes
        )


def profile_menu_helper_functions(profiler: PerformanceProfiler):
    """Profile menu helper functions."""
    logger.info("Profiling menu helper functions...")
    
    from kinetra_menu import select_asset_classes, select_timeframes, select_agent_types
    from unittest.mock import patch
    
    # Mock input for automated testing
    def mock_input_factory(response):
        def mock_input(prompt):
            return response
        return mock_input
    
    # Profile asset class selection
    with patch('builtins.input', mock_input_factory('a')):
        profiler.profile_operation(
            "menu_select_asset_classes",
            select_asset_classes
        )
    
    # Profile timeframe selection
    with patch('builtins.input', mock_input_factory('a')):
        profiler.profile_operation(
            "menu_select_timeframes",
            select_timeframes
        )
    
    # Profile agent type selection
    with patch('builtins.input', mock_input_factory('a')):
        profiler.profile_operation(
            "menu_select_agent_types",
            select_agent_types
        )


def profile_workflow_manager_creation(profiler: PerformanceProfiler):
    """Profile workflow manager creation."""
    logger.info("Profiling workflow manager creation...")
    
    from kinetra.workflow_manager import WorkflowManager
    
    # Profile single creation
    profiler.profile_operation(
        "workflow_manager_create",
        WorkflowManager,
        log_dir="logs/profile_test",
        enable_backups=False
    )
    
    # Profile multiple creations to check overhead
    for i in range(5):
        profiler.profile_operation(
            f"workflow_manager_create_repeat_{i}",
            WorkflowManager,
            log_dir=f"logs/profile_test_{i}",
            enable_backups=False
        )


# =============================================================================
# E2E FRAMEWORK PROFILING
# =============================================================================

def profile_e2e_presets(profiler: PerformanceProfiler):
    """Profile E2E preset generation."""
    logger.info("Profiling E2E preset generation...")
    
    from e2e_testing_framework import E2EPresets
    
    # Profile each preset
    profiler.profile_operation(
        "e2e_preset_quick_validation",
        E2EPresets.quick_validation
    )
    
    profiler.profile_operation(
        "e2e_preset_full_system",
        E2EPresets.full_system_test
    )
    
    profiler.profile_operation(
        "e2e_preset_asset_class",
        E2EPresets.asset_class_test,
        'crypto'
    )
    
    profiler.profile_operation(
        "e2e_preset_agent_type",
        E2EPresets.agent_type_test,
        'ppo'
    )


def profile_e2e_test_matrix_generation(profiler: PerformanceProfiler):
    """Profile E2E test matrix generation."""
    logger.info("Profiling E2E test matrix generation...")
    
    from e2e_testing_framework import E2EPresets, E2ETestRunner
    
    # Profile matrix generation for different sizes
    configs = [
        ("quick", E2EPresets.quick_validation()),
        ("medium", E2EPresets.asset_class_test('crypto')),
        ("large", E2EPresets.agent_type_test('ppo'))
    ]
    
    for config_name, config in configs:
        def generate_matrix():
            runner = E2ETestRunner(
                config,
                output_dir=f"data/results/profile_{config_name}",
                log_dir=f"logs/profile_{config_name}"
            )
            return runner.generate_test_matrix()
        
        profiler.profile_operation(
            f"e2e_matrix_generation_{config_name}",
            generate_matrix
        )


def profile_instrument_registry(profiler: PerformanceProfiler):
    """Profile instrument registry operations."""
    logger.info("Profiling instrument registry operations...")
    
    from e2e_testing_framework import InstrumentRegistry
    
    # Profile getting instruments by asset class
    for asset_class in ['crypto', 'forex', 'indices', 'metals', 'commodities']:
        profiler.profile_operation(
            f"instrument_registry_get_{asset_class}",
            InstrumentRegistry.get_instruments,
            asset_class
        )
    
    # Profile getting all instruments
    profiler.profile_operation(
        "instrument_registry_get_all",
        InstrumentRegistry.get_all_instruments
    )
    
    # Profile getting top instruments
    profiler.profile_operation(
        "instrument_registry_get_top_3",
        InstrumentRegistry.get_top_instruments,
        'crypto',
        3
    )


def profile_config_loading(profiler: PerformanceProfiler):
    """Profile configuration loading from JSON."""
    logger.info("Profiling configuration loading...")
    
    import json
    from e2e_testing_framework import E2ETestConfig
    from pathlib import Path
    
    config_dir = Path("configs/e2e_examples")
    
    if config_dir.exists():
        for config_file in config_dir.glob("*.json"):
            def load_config():
                with open(config_file, 'r') as f:
                    config_dict = json.load(f)
                return E2ETestConfig(**config_dict)
            
            profiler.profile_operation(
                f"config_load_{config_file.stem}",
                load_config
            )


# =============================================================================
# FILE I/O PROFILING
# =============================================================================

def profile_file_operations(profiler: PerformanceProfiler):
    """Profile file I/O operations."""
    logger.info("Profiling file I/O operations...")
    
    from pathlib import Path
    
    # Profile directory existence checks
    dirs = ['data/master', 'data/prepared', 'logs', 'configs']
    for dir_path in dirs:
        profiler.profile_operation(
            f"file_check_exists_{dir_path.replace('/', '_')}",
            lambda p=dir_path: Path(p).exists()
        )
    
    # Profile directory creation
    test_dir = Path("logs/profile_test_io")
    
    def create_and_cleanup():
        test_dir.mkdir(parents=True, exist_ok=True)
        test_dir.rmdir()
    
    profiler.profile_operation(
        "file_create_directory",
        create_and_cleanup
    )


# =============================================================================
# MAIN PROFILING ORCHESTRATOR
# =============================================================================

def run_performance_profiling(
    profile_menu: bool = True,
    profile_e2e: bool = True,
    profile_io: bool = True
) -> ProfilingReport:
    """
    Run comprehensive performance profiling.
    
    Args:
        profile_menu: Profile menu system
        profile_e2e: Profile E2E framework
        profile_io: Profile file I/O
        
    Returns:
        Profiling report
    """
    print("=" * 80)
    print("  KINETRA PERFORMANCE PROFILING & BOTTLENECK ANALYSIS")
    print("=" * 80)
    print()
    
    profiler = PerformanceProfiler()
    
    # Menu system profiling
    if profile_menu:
        print("\n" + "=" * 80)
        print("PROFILING: Menu System")
        print("=" * 80)
        
        profile_menu_config_operations(profiler)
        profile_menu_helper_functions(profiler)
        profile_workflow_manager_creation(profiler)
    
    # E2E framework profiling
    if profile_e2e:
        print("\n" + "=" * 80)
        print("PROFILING: E2E Framework")
        print("=" * 80)
        
        profile_e2e_presets(profiler)
        profile_e2e_test_matrix_generation(profiler)
        profile_instrument_registry(profiler)
        profile_config_loading(profiler)
    
    # File I/O profiling
    if profile_io:
        print("\n" + "=" * 80)
        print("PROFILING: File I/O Operations")
        print("=" * 80)
        
        profile_file_operations(profiler)
    
    # Generate report
    report = profiler.generate_report()
    
    # Print report
    print_profiling_report(report)
    
    return report


def print_profiling_report(report: ProfilingReport):
    """Print comprehensive profiling report."""
    print("\n" + "=" * 80)
    print("  PERFORMANCE PROFILING REPORT")
    print("=" * 80)
    
    print(f"\nTimestamp: {report.timestamp}")
    print(f"Total Profiling Time: {report.total_time:.3f}s")
    print(f"Operations Profiled: {len(report.operations)}")
    
    # Performance summary
    print("\n" + "-" * 80)
    print("PERFORMANCE SUMMARY")
    print("-" * 80)
    
    # Sort operations by execution time
    sorted_ops = sorted(report.operations, key=lambda x: x.execution_time, reverse=True)
    
    print("\nTop 10 Slowest Operations:")
    print(f"{'Operation':<50} {'Time (ms)':<12} {'Memory (KB)':<12} {'Calls':<10}")
    print("-" * 84)
    
    for op in sorted_ops[:10]:
        print(f"{op.operation_name:<50} {op.execution_time*1000:>10.2f}ms "
              f"{op.memory_peak/1024:>10.1f}KB {op.call_count:>9,}")
    
    # Hotspots
    if report.hotspots:
        print("\n" + "-" * 80)
        print("PERFORMANCE HOTSPOTS (>5% of total time)")
        print("-" * 80)
        
        for op_name, time_spent in sorted(report.hotspots.items(), key=lambda x: x[1], reverse=True):
            percentage = (time_spent / report.total_time) * 100
            print(f"  {op_name:<50} {time_spent*1000:>10.2f}ms ({percentage:>5.1f}%)")
    
    # Memory profile
    print("\n" + "-" * 80)
    print("MEMORY PROFILE")
    print("-" * 80)
    
    print(f"Total Allocated: {report.memory_profile['total_allocated']/1024/1024:.2f} MB")
    print(f"Peak Usage: {report.memory_profile['peak_usage']/1024/1024:.2f} MB")
    print(f"Avg per Operation: {report.memory_profile['avg_per_operation']/1024:.2f} KB")
    
    # Bottlenecks
    print("\n" + "-" * 80)
    print("BOTTLENECKS DETECTED")
    print("-" * 80)
    
    bottleneck_count = 0
    for op in report.operations:
        if op.bottlenecks:
            print(f"\n{op.operation_name}:")
            for bottleneck in op.bottlenecks:
                print(f"  ⚠️  {bottleneck}")
                bottleneck_count += 1
    
    if bottleneck_count == 0:
        print("  ✅ No significant bottlenecks detected")
    
    # Optimization suggestions
    print("\n" + "=" * 80)
    print("OPTIMIZATION RECOMMENDATIONS")
    print("=" * 80)
    
    # Per-operation suggestions
    print("\nOperation-Specific Suggestions:")
    suggestion_count = 0
    for op in report.operations:
        if op.optimization_suggestions:
            print(f"\n{op.operation_name}:")
            for suggestion in op.optimization_suggestions:
                print(f"  💡 {suggestion}")
                suggestion_count += 1
    
    if suggestion_count == 0:
        print("  ✅ No operation-specific optimizations needed")
    
    # Overall recommendations
    if report.recommendations:
        print("\nOverall Recommendations:")
        for i, recommendation in enumerate(report.recommendations, 1):
            print(f"  {i}. {recommendation}")
    else:
        print("\n  ✅ System is well-optimized")
    
    # Performance score
    print("\n" + "=" * 80)
    print("PERFORMANCE SCORE")
    print("=" * 80)
    
    score = calculate_performance_score(report)
    print(f"\nOverall Performance Score: {score}/100")
    
    if score >= 90:
        print("  ✅ EXCELLENT - System is highly optimized")
    elif score >= 75:
        print("  ✅ GOOD - Minor optimizations possible")
    elif score >= 60:
        print("  ⚠️  FAIR - Several optimization opportunities")
    else:
        print("  ❌ POOR - Significant optimization needed")
    
    print("\n" + "=" * 80)


def calculate_performance_score(report: ProfilingReport) -> int:
    """Calculate overall performance score (0-100)."""
    score = 100
    
    # Deduct for slow operations
    slow_ops = sum(1 for op in report.operations if op.execution_time > 0.1)
    score -= min(slow_ops * 5, 30)
    
    # Deduct for high memory usage
    if report.memory_profile['peak_usage'] > 100 * 1024 * 1024:  # > 100MB
        score -= 20
    elif report.memory_profile['peak_usage'] > 50 * 1024 * 1024:  # > 50MB
        score -= 10
    
    # Deduct for bottlenecks
    bottleneck_count = sum(len(op.bottlenecks) for op in report.operations)
    score -= min(bottleneck_count * 5, 30)
    
    # Deduct for recommendations
    score -= min(len(report.recommendations) * 3, 20)
    
    return max(score, 0)


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Kinetra Performance Profiling and Bottleneck Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--full',
        action='store_true',
        help="Profile everything (menu, E2E, I/O)"
    )
    
    parser.add_argument(
        '--menu-only',
        action='store_true',
        help="Profile menu system only"
    )
    
    parser.add_argument(
        '--e2e-only',
        action='store_true',
        help="Profile E2E framework only"
    )
    
    parser.add_argument(
        '--io-only',
        action='store_true',
        help="Profile file I/O only"
    )
    
    parser.add_argument(
        '--save',
        action='store_true',
        help="Save profiling report to JSON"
    )
    
    args = parser.parse_args()
    
    # Determine what to profile
    if args.menu_only:
        profile_menu, profile_e2e, profile_io = True, False, False
    elif args.e2e_only:
        profile_menu, profile_e2e, profile_io = False, True, False
    elif args.io_only:
        profile_menu, profile_e2e, profile_io = False, False, True
    else:  # full or default
        profile_menu, profile_e2e, profile_io = True, True, True
    
    # Run profiling
    try:
        report = run_performance_profiling(profile_menu, profile_e2e, profile_io)
        
        # Save report if requested
        if args.save:
            output_dir = Path("data/results/profiling")
            output_dir.mkdir(parents=True, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = output_dir / f"profiling_report_{timestamp}.json"
            
            import json
            
            # Convert report to dict for JSON serialization
            report_dict = {
                'timestamp': report.timestamp,
                'total_time': report.total_time,
                'operations': [
                    {
                        'name': op.operation_name,
                        'execution_time': op.execution_time,
                        'cpu_time': op.cpu_time,
                        'memory_allocated': op.memory_allocated,
                        'memory_peak': op.memory_peak,
                        'call_count': op.call_count,
                        'bottlenecks': op.bottlenecks,
                        'optimization_suggestions': op.optimization_suggestions
                    }
                    for op in report.operations
                ],
                'hotspots': report.hotspots,
                'memory_profile': report.memory_profile,
                'recommendations': report.recommendations
            }
            
            with open(output_file, 'w') as f:
                json.dump(report_dict, f, indent=2)
            
            print(f"\n📊 Report saved to: {output_file}")
        
        return 0
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Profiling interrupted by user")
        return 1
    except Exception as e:
        print(f"\n\n❌ Profiling failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
