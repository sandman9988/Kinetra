"""
Real-Time Folder Monitoring
===========================

Monitor folders for:
- Performance issues
- Failures
- File changes
- Resource usage
"""

import hashlib
import json
import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False


class EventType(Enum):
    """Types of monitored events."""
    FILE_CREATED = "file_created"
    FILE_MODIFIED = "file_modified"
    FILE_DELETED = "file_deleted"
    PERFORMANCE_ALERT = "performance_alert"
    FAILURE_DETECTED = "failure_detected"
    RESOURCE_WARNING = "resource_warning"


@dataclass
class MonitorEvent:
    """A monitored event."""
    event_type: EventType
    timestamp: datetime
    path: str
    message: str
    severity: str = "info"  # info, warning, error, critical
    metadata: Dict = field(default_factory=dict)


@dataclass
class PerformanceMetrics:
    """System performance metrics."""
    cpu_percent: float
    memory_percent: float
    memory_available_gb: float
    disk_usage_percent: float
    disk_free_gb: float
    process_count: int
    timestamp: datetime = field(default_factory=datetime.now)


class PerformanceTracker:
    """
    Track and analyze performance metrics over time.

    Usage:
        tracker = PerformanceTracker()
        tracker.start()
        # ... later ...
        stats = tracker.get_stats()
    """

    def __init__(self, history_size: int = 1000):
        self.history: deque = deque(maxlen=history_size)
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

        # Alert thresholds
        self.cpu_threshold = 90.0
        self.memory_threshold = 85.0
        self.disk_threshold = 90.0

        # Alert callbacks
        self.on_alert: Optional[Callable[[str, float, float], None]] = None

    def get_current_metrics(self) -> PerformanceMetrics:
        """Get current system performance metrics."""
        if PSUTIL_AVAILABLE:
            mem = psutil.virtual_memory()
            disk = psutil.disk_usage('/')

            return PerformanceMetrics(
                cpu_percent=psutil.cpu_percent(interval=0.1),
                memory_percent=mem.percent,
                memory_available_gb=mem.available / (1024**3),
                disk_usage_percent=disk.percent,
                disk_free_gb=disk.free / (1024**3),
                process_count=len(psutil.pids())
            )
        else:
            return PerformanceMetrics(
                cpu_percent=0,
                memory_percent=0,
                memory_available_gb=0,
                disk_usage_percent=0,
                disk_free_gb=0,
                process_count=0
            )

    def record(self) -> PerformanceMetrics:
        """Record current metrics."""
        metrics = self.get_current_metrics()

        with self._lock:
            self.history.append(metrics)

        # Check alerts
        self._check_alerts(metrics)

        return metrics

    def _check_alerts(self, metrics: PerformanceMetrics):
        """Check for performance alerts."""
        if not self.on_alert:
            return

        if metrics.cpu_percent > self.cpu_threshold:
            self.on_alert("CPU", metrics.cpu_percent, self.cpu_threshold)

        if metrics.memory_percent > self.memory_threshold:
            self.on_alert("Memory", metrics.memory_percent, self.memory_threshold)

        if metrics.disk_usage_percent > self.disk_threshold:
            self.on_alert("Disk", metrics.disk_usage_percent, self.disk_threshold)

    def start(self, interval: float = 5.0):
        """Start background monitoring."""
        if self._running:
            return

        self._running = True
        self._thread = threading.Thread(
            target=self._monitor_loop,
            args=(interval,),
            daemon=True
        )
        self._thread.start()

    def stop(self):
        """Stop background monitoring."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)

    def _monitor_loop(self, interval: float):
        """Background monitoring loop."""
        while self._running:
            try:
                self.record()
            except Exception:
                pass
            time.sleep(interval)

    def get_stats(self) -> Dict[str, any]:
        """Get aggregated statistics."""
        with self._lock:
            if not self.history:
                return {}

            history_list = list(self.history)

        cpu_values = [m.cpu_percent for m in history_list]
        mem_values = [m.memory_percent for m in history_list]

        return {
            "samples": len(history_list),
            "duration_seconds": (history_list[-1].timestamp - history_list[0].timestamp).total_seconds() if len(history_list) > 1 else 0,
            "cpu": {
                "current": cpu_values[-1] if cpu_values else 0,
                "avg": sum(cpu_values) / len(cpu_values) if cpu_values else 0,
                "max": max(cpu_values) if cpu_values else 0,
                "min": min(cpu_values) if cpu_values else 0,
            },
            "memory": {
                "current": mem_values[-1] if mem_values else 0,
                "avg": sum(mem_values) / len(mem_values) if mem_values else 0,
                "max": max(mem_values) if mem_values else 0,
                "min": min(mem_values) if mem_values else 0,
            },
            "disk_free_gb": history_list[-1].disk_free_gb if history_list else 0,
        }


class FolderMonitor:
    """
    Monitor folders for changes, failures, and performance issues.

    Usage:
        monitor = FolderMonitor(["/path/to/watch"])
        monitor.on_event = my_callback
        monitor.start()
    """

    def __init__(
        self,
        watch_paths: List[str] = None,
        patterns: List[str] = None,
        ignore_patterns: List[str] = None
    ):
        self.watch_paths = [Path(p) for p in (watch_paths or ["."])]
        self.patterns = patterns or ["*.py", "*.csv", "*.log"]
        self.ignore_patterns = ignore_patterns or ["__pycache__", ".git", "*.pyc"]

        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._file_states: Dict[str, Tuple[float, str]] = {}  # path -> (mtime, hash)
        self._events: deque = deque(maxlen=1000)

        # Callbacks
        self.on_event: Optional[Callable[[MonitorEvent], None]] = None
        self.on_failure: Optional[Callable[[str, str], None]] = None

        # Performance tracker
        self.performance = PerformanceTracker()

        # Logger
        self.logger = logging.getLogger("FolderMonitor")

    def _should_watch(self, path: Path) -> bool:
        """Check if path matches watch patterns."""
        path_str = str(path)

        # Check ignore patterns
        for pattern in self.ignore_patterns:
            if pattern in path_str:
                return False

        # Check file patterns
        for pattern in self.patterns:
            if path.match(pattern):
                return True

        return False

    def _get_file_hash(self, path: Path, sample_size: int = 1024) -> str:
        """Get quick hash of file (first bytes + size)."""
        try:
            stat = path.stat()
            with open(path, 'rb') as f:
                sample = f.read(sample_size)
            return hashlib.md5(sample + str(stat.st_size).encode()).hexdigest()
        except (IOError, PermissionError):
            return ""

    def _scan_files(self) -> Dict[str, Tuple[float, str]]:
        """Scan all watched files."""
        states = {}

        for watch_path in self.watch_paths:
            if not watch_path.exists():
                continue

            if watch_path.is_file():
                if self._should_watch(watch_path):
                    mtime = watch_path.stat().st_mtime
                    file_hash = self._get_file_hash(watch_path)
                    states[str(watch_path)] = (mtime, file_hash)
            else:
                for pattern in self.patterns:
                    for path in watch_path.rglob(pattern.lstrip('*')):
                        if self._should_watch(path):
                            try:
                                mtime = path.stat().st_mtime
                                file_hash = self._get_file_hash(path)
                                states[str(path)] = (mtime, file_hash)
                            except (IOError, PermissionError):
                                pass

        return states

    def _check_changes(self) -> List[MonitorEvent]:
        """Check for file changes."""
        events = []
        current_states = self._scan_files()

        # Check for new and modified files
        for path, (mtime, file_hash) in current_states.items():
            if path not in self._file_states:
                events.append(MonitorEvent(
                    event_type=EventType.FILE_CREATED,
                    timestamp=datetime.now(),
                    path=path,
                    message=f"File created: {path}"
                ))
            else:
                old_mtime, old_hash = self._file_states[path]
                if mtime != old_mtime or file_hash != old_hash:
                    events.append(MonitorEvent(
                        event_type=EventType.FILE_MODIFIED,
                        timestamp=datetime.now(),
                        path=path,
                        message=f"File modified: {path}",
                        metadata={"old_mtime": old_mtime, "new_mtime": mtime}
                    ))

        # Check for deleted files
        for path in self._file_states:
            if path not in current_states:
                events.append(MonitorEvent(
                    event_type=EventType.FILE_DELETED,
                    timestamp=datetime.now(),
                    path=path,
                    message=f"File deleted: {path}"
                ))

        self._file_states = current_states
        return events

    def _check_log_failures(self) -> List[MonitorEvent]:
        """Check log files for failures/errors."""
        events = []
        error_patterns = [
            "ERROR", "FAILED", "Exception", "Traceback",
            "CRITICAL", "FATAL", "OutOfMemory"
        ]

        for watch_path in self.watch_paths:
            for log_file in watch_path.rglob("*.log"):
                try:
                    # Only check recently modified logs
                    if time.time() - log_file.stat().st_mtime > 60:
                        continue

                    with open(log_file, 'r', errors='ignore') as f:
                        # Read last 100 lines
                        lines = f.readlines()[-100:]

                    for line in lines:
                        for pattern in error_patterns:
                            if pattern in line:
                                events.append(MonitorEvent(
                                    event_type=EventType.FAILURE_DETECTED,
                                    timestamp=datetime.now(),
                                    path=str(log_file),
                                    message=line.strip()[:200],
                                    severity="error"
                                ))
                                break
                except (IOError, PermissionError):
                    pass

        return events

    def start(self, interval: float = 2.0, monitor_performance: bool = True):
        """Start folder monitoring."""
        if self._running:
            return

        # Initial scan
        self._file_states = self._scan_files()

        self._running = True
        self._thread = threading.Thread(
            target=self._monitor_loop,
            args=(interval,),
            daemon=True
        )
        self._thread.start()

        if monitor_performance:
            self.performance.on_alert = self._on_performance_alert
            self.performance.start()

        self.logger.info(f"Started monitoring {len(self.watch_paths)} paths")

    def stop(self):
        """Stop folder monitoring."""
        self._running = False
        self.performance.stop()
        if self._thread:
            self._thread.join(timeout=2.0)
        self.logger.info("Stopped monitoring")

    def _monitor_loop(self, interval: float):
        """Background monitoring loop."""
        failure_check_counter = 0

        while self._running:
            try:
                # Check file changes
                events = self._check_changes()

                # Periodically check for failures in logs
                failure_check_counter += 1
                if failure_check_counter >= 5:  # Every 5 iterations
                    events.extend(self._check_log_failures())
                    failure_check_counter = 0

                # Process events
                for event in events:
                    self._events.append(event)

                    if self.on_event:
                        self.on_event(event)

                    if event.event_type == EventType.FAILURE_DETECTED and self.on_failure:
                        self.on_failure(event.path, event.message)

                    self.logger.info(f"[{event.event_type.value}] {event.message}")

            except Exception as e:
                self.logger.error(f"Monitor error: {e}")

            time.sleep(interval)

    def _on_performance_alert(self, metric: str, value: float, threshold: float):
        """Handle performance alerts."""
        event = MonitorEvent(
            event_type=EventType.PERFORMANCE_ALERT,
            timestamp=datetime.now(),
            path="system",
            message=f"{metric} usage at {value:.1f}% (threshold: {threshold:.1f}%)",
            severity="warning" if value < threshold + 5 else "error",
            metadata={"metric": metric, "value": value, "threshold": threshold}
        )

        self._events.append(event)

        if self.on_event:
            self.on_event(event)

        self.logger.warning(event.message)

    def get_recent_events(self, count: int = 50) -> List[MonitorEvent]:
        """Get recent events."""
        return list(self._events)[-count:]

    def get_status(self) -> Dict[str, any]:
        """Get monitor status."""
        return {
            "running": self._running,
            "watched_paths": [str(p) for p in self.watch_paths],
            "files_tracked": len(self._file_states),
            "events_recorded": len(self._events),
            "performance": self.performance.get_stats()
        }


def start_monitoring(
    paths: List[str] = None,
    on_event: Callable[[MonitorEvent], None] = None,
    interval: float = 2.0
) -> FolderMonitor:
    """
    Convenience function to start folder monitoring.

    Args:
        paths: Paths to watch
        on_event: Callback for events
        interval: Check interval in seconds

    Returns:
        FolderMonitor instance
    """
    monitor = FolderMonitor(paths)
    monitor.on_event = on_event
    monitor.start(interval=interval)
    return monitor


def create_monitor_daemon(config_file: str = "monitor_config.json"):
    """Create a monitoring daemon script."""
    config = {
        "watch_paths": ["."],
        "patterns": ["*.py", "*.csv", "*.log"],
        "ignore_patterns": ["__pycache__", ".git", "*.pyc", ".venv"],
        "interval": 2.0,
        "log_file": "monitor.log",
        "performance_alerts": {
            "cpu_threshold": 90,
            "memory_threshold": 85,
            "disk_threshold": 90
        }
    }

    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)

    return config_file


def print_monitor_status(monitor: FolderMonitor):
    """Print monitoring status."""
    status = monitor.get_status()
    perf = status.get('performance', {})

    print("=" * 60)
    print("FOLDER MONITOR STATUS")
    print("=" * 60)
    print(f"Running:         {'✅ Yes' if status['running'] else '❌ No'}")
    print(f"Files tracked:   {status['files_tracked']}")
    print(f"Events recorded: {status['events_recorded']}")
    print("")
    print("Watched paths:")
    for path in status['watched_paths']:
        print(f"  - {path}")
    print("")
    print("Performance:")
    if perf:
        cpu = perf.get('cpu', {})
        mem = perf.get('memory', {})
        print(f"  CPU:    {cpu.get('current', 0):.1f}% (avg: {cpu.get('avg', 0):.1f}%)")
        print(f"  Memory: {mem.get('current', 0):.1f}% (avg: {mem.get('avg', 0):.1f}%)")
        print(f"  Disk:   {perf.get('disk_free_gb', 0):.1f} GB free")
    print("=" * 60)
