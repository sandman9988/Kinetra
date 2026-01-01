"""
Parallelization Utilities with Auto-Scaling
============================================

Smart parallelization that:
- Auto-detects optimal worker count
- Monitors resource usage
- Scales dynamically based on load
- Handles failures gracefully
"""

import multiprocessing as mp
import threading
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, TypeVar

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False


T = TypeVar('T')


@dataclass
class ResourceUsage:
    """Current resource usage snapshot."""
    cpu_percent: float
    memory_percent: float
    memory_available_gb: float
    cpu_count: int
    load_avg: tuple


@dataclass
class ParallelConfig:
    """Configuration for parallel execution."""
    max_workers: int = -1  # -1 = auto
    min_workers: int = 1
    use_processes: bool = True  # False = use threads
    memory_limit_pct: float = 0.80
    cpu_limit_pct: float = 0.90
    scale_up_threshold: float = 0.50  # Scale up if CPU < 50%
    scale_down_threshold: float = 0.95  # Scale down if CPU > 95%
    check_interval: float = 1.0  # Seconds between resource checks


@dataclass
class TaskResult:
    """Result of a parallel task."""
    task_id: int
    success: bool
    result: Any = None
    error: Optional[str] = None
    duration: float = 0.0


def get_resource_usage() -> ResourceUsage:
    """Get current resource usage."""
    if PSUTIL_AVAILABLE:
        mem = psutil.virtual_memory()
        return ResourceUsage(
            cpu_percent=psutil.cpu_percent(interval=0.1),
            memory_percent=mem.percent,
            memory_available_gb=mem.available / (1024**3),
            cpu_count=mp.cpu_count(),
            load_avg=psutil.getloadavg() if hasattr(psutil, 'getloadavg') else (0, 0, 0)
        )
    else:
        return ResourceUsage(
            cpu_percent=50.0,
            memory_percent=50.0,
            memory_available_gb=8.0,
            cpu_count=mp.cpu_count(),
            load_avg=(0, 0, 0)
        )


def get_optimal_config(task_type: str = "cpu") -> ParallelConfig:
    """
    Get optimal parallel configuration based on system resources.

    Args:
        task_type: 'cpu' for CPU-bound, 'io' for I/O-bound, 'memory' for memory-heavy

    Returns:
        ParallelConfig with optimal settings
    """
    usage = get_resource_usage()
    cpu_count = usage.cpu_count

    if task_type == "cpu":
        # CPU-bound: leave cores for system
        max_workers = max(1, cpu_count - 2)
        use_processes = True
        memory_limit = 0.80

    elif task_type == "io":
        # I/O-bound: can use more workers than cores
        max_workers = cpu_count * 2
        use_processes = False  # Threads for I/O
        memory_limit = 0.90

    elif task_type == "memory":
        # Memory-heavy: limit workers based on available memory
        mem_per_worker_gb = 2.0  # Estimate
        max_by_memory = int(usage.memory_available_gb / mem_per_worker_gb)
        max_workers = min(cpu_count, max_by_memory)
        use_processes = True
        memory_limit = 0.70

    else:
        max_workers = max(1, cpu_count - 2)
        use_processes = True
        memory_limit = 0.80

    return ParallelConfig(
        max_workers=max_workers,
        use_processes=use_processes,
        memory_limit_pct=memory_limit
    )


class AutoScaler:
    """
    Dynamic worker scaling based on resource usage.

    Monitors CPU/memory and adjusts worker count in real-time.
    """

    def __init__(self, config: ParallelConfig = None):
        self.config = config or get_optimal_config()
        self.current_workers = self.config.max_workers
        self._running = False
        self._monitor_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

    def start_monitoring(self):
        """Start resource monitoring in background."""
        if self._running:
            return

        self._running = True
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()

    def stop_monitoring(self):
        """Stop resource monitoring."""
        self._running = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=2.0)

    def _monitor_loop(self):
        """Background monitoring loop."""
        while self._running:
            try:
                usage = get_resource_usage()
                self._adjust_workers(usage)
            except Exception:
                pass
            time.sleep(self.config.check_interval)

    def _adjust_workers(self, usage: ResourceUsage):
        """Adjust worker count based on resource usage."""
        with self._lock:
            # Check memory first
            if usage.memory_percent > self.config.memory_limit_pct * 100:
                # Scale down due to memory pressure
                new_workers = max(
                    self.config.min_workers,
                    self.current_workers - 1
                )
            elif usage.cpu_percent < self.config.scale_up_threshold * 100:
                # Scale up - system has headroom
                new_workers = min(
                    self.config.max_workers,
                    self.current_workers + 1
                )
            elif usage.cpu_percent > self.config.scale_down_threshold * 100:
                # Scale down - system is overloaded
                new_workers = max(
                    self.config.min_workers,
                    self.current_workers - 1
                )
            else:
                new_workers = self.current_workers

            if new_workers != self.current_workers:
                self.current_workers = new_workers

    def get_worker_count(self) -> int:
        """Get current recommended worker count."""
        with self._lock:
            return self.current_workers


class ParallelExecutor:
    """
    High-level parallel execution with auto-scaling and error handling.

    Usage:
        executor = ParallelExecutor()
        results = executor.map(process_func, items)
    """

    def __init__(
        self,
        config: ParallelConfig = None,
        auto_scale: bool = True,
        verbose: bool = False
    ):
        self.config = config or get_optimal_config()
        self.auto_scale = auto_scale
        self.verbose = verbose
        self.scaler = AutoScaler(self.config) if auto_scale else None
        self._results: List[TaskResult] = []

    def map(
        self,
        func: Callable[[T], Any],
        items: List[T],
        timeout: float = None,
        on_error: str = "continue"  # 'continue', 'stop', 'raise'
    ) -> List[TaskResult]:
        """
        Apply function to items in parallel.

        Args:
            func: Function to apply
            items: Items to process
            timeout: Timeout per task in seconds
            on_error: Error handling ('continue', 'stop', 'raise')

        Returns:
            List of TaskResult objects
        """
        if not items:
            return []

        # Start auto-scaling if enabled
        if self.scaler:
            self.scaler.start_monitoring()

        try:
            results = self._execute_parallel(func, items, timeout, on_error)
        finally:
            if self.scaler:
                self.scaler.stop_monitoring()

        return results

    def _execute_parallel(
        self,
        func: Callable,
        items: List,
        timeout: float,
        on_error: str
    ) -> list[None]:
        """Execute tasks in parallel."""
        n_workers = self.scaler.get_worker_count() if self.scaler else self.config.max_workers

        if self.verbose:
            print(f"Starting parallel execution with {n_workers} workers")

        ExecutorClass = ProcessPoolExecutor if self.config.use_processes else ThreadPoolExecutor

        results = [None] * len(items)
        completed = 0
        failed = 0

        with ExecutorClass(max_workers=n_workers) as executor:
            # Submit all tasks
            future_to_idx = {}
            for i, item in enumerate(items):
                future = executor.submit(self._run_task, func, item, i)
                future_to_idx[future] = i

            # Collect results
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]

                try:
                    result = future.result(timeout=timeout)
                    results[idx] = result

                    if result.success:
                        completed += 1
                    else:
                        failed += 1
                        if on_error == "raise":
                            raise RuntimeError(f"Task {idx} failed: {result.error}")
                        elif on_error == "stop":
                            break

                except Exception as e:
                    results[idx] = TaskResult(
                        task_id=idx,
                        success=False,
                        error=str(e)
                    )
                    failed += 1

                    if on_error == "raise":
                        raise
                    elif on_error == "stop":
                        break

                if self.verbose and (completed + failed) % 10 == 0:
                    print(f"Progress: {completed + failed}/{len(items)} (failed: {failed})")

        if self.verbose:
            print(f"Completed: {completed}, Failed: {failed}")

        return [r for r in results if r is not None]

    @staticmethod
    def _run_task(func: Callable, item: Any, task_id: int) -> TaskResult:
        """Run a single task with error handling."""
        start = time.perf_counter()
        try:
            result = func(item)
            return TaskResult(
                task_id=task_id,
                success=True,
                result=result,
                duration=time.perf_counter() - start
            )
        except Exception as e:
            return TaskResult(
                task_id=task_id,
                success=False,
                error=str(e),
                duration=time.perf_counter() - start
            )

    def get_stats(self) -> Dict[str, Any]:
        """Get execution statistics."""
        if not self._results:
            return {}

        successful = [r for r in self._results if r.success]
        failed = [r for r in self._results if not r.success]
        durations = [r.duration for r in self._results]

        return {
            "total": len(self._results),
            "successful": len(successful),
            "failed": len(failed),
            "success_rate": len(successful) / len(self._results) if self._results else 0,
            "total_duration": sum(durations),
            "avg_duration": sum(durations) / len(durations) if durations else 0,
            "max_duration": max(durations) if durations else 0,
            "min_duration": min(durations) if durations else 0,
        }


def print_parallel_info():
    """Print information about parallel processing capabilities."""
    usage = get_resource_usage()

    print("=" * 60)
    print("PARALLEL PROCESSING CONFIGURATION")
    print("=" * 60)
    print(f"CPU Cores:         {usage.cpu_count}")
    print(f"CPU Usage:         {usage.cpu_percent:.1f}%")
    print(f"Memory Usage:      {usage.memory_percent:.1f}%")
    print(f"Memory Available:  {usage.memory_available_gb:.1f} GB")
    print(f"Load Average:      {usage.load_avg}")
    print("-" * 60)
    print(f"Recommended Workers (CPU-bound):    {get_optimal_config('cpu').max_workers}")
    print(f"Recommended Workers (I/O-bound):    {get_optimal_config('io').max_workers}")
    print(f"Recommended Workers (Memory-heavy): {get_optimal_config('memory').max_workers}")
    print("=" * 60)
