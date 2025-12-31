"""
System-Wide Silent Failure Logger for Kinetra

Provides centralized logging of silent failures for automatic analysis and AI agent corrections.
Captures errors that are caught but not properly handled, enabling:
- Pattern detection for recurring issues
- Automatic analysis by AI agents
- Debugging and diagnostics
- Health monitoring and alerting

Usage:
    from kinetra.silent_failure_logger import log_failure, SilentFailureLogger
    
    # Simple usage
    try:
        risky_operation()
    except Exception as e:
        log_failure(e, context={"operation": "risky_operation"})
    
    # With decorator
    @log_failures()
    def my_function():
        pass
    
    # Access global logger
    logger = SilentFailureLogger.get_instance()
    failures = logger.get_failures(category="import_error")
"""

import json
import logging
import os
import sys
import traceback
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from threading import Lock
from typing import Any, Callable, Dict, List, Optional, Set


class FailureCategory(Enum):
    """Categories of failures for classification."""
    IMPORT_ERROR = "import_error"
    TYPE_ERROR = "type_error"
    VALUE_ERROR = "value_error"
    ATTRIBUTE_ERROR = "attribute_error"
    KEY_ERROR = "key_error"
    INDEX_ERROR = "index_error"
    FILE_ERROR = "file_error"
    NETWORK_ERROR = "network_error"
    TIMEOUT_ERROR = "timeout_error"
    PERMISSION_ERROR = "permission_error"
    DATA_ERROR = "data_error"
    CALCULATION_ERROR = "calculation_error"
    VALIDATION_ERROR = "validation_error"
    UNKNOWN = "unknown"


class FailureSeverity(Enum):
    """Severity levels for failures."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class FailureRecord:
    """Record of a single silent failure."""
    
    # Core information
    timestamp: str
    category: str
    severity: str
    exception_type: str
    exception_message: str
    
    # Location information
    file_path: str
    function_name: str
    line_number: int
    
    # Context
    context: Dict[str, Any] = field(default_factory=dict)
    stack_trace: str = ""
    
    # Metadata
    failure_id: str = ""
    module: str = ""
    tags: List[str] = field(default_factory=list)
    
    # Analysis fields
    count: int = 1
    first_seen: Optional[str] = None
    last_seen: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2, default=str)


class SilentFailureLogger:
    """
    Centralized logger for silent failures.
    
    Singleton pattern ensures single instance across the application.
    Thread-safe for concurrent logging.
    """
    
    _instance = None
    _lock = Lock()
    
    def __init__(
        self,
        log_dir: Optional[Path] = None,
        max_records: int = 10000,
        enable_console: bool = False,
        enable_file: bool = True,
        enable_aggregation: bool = True,
    ):
        """
        Initialize the silent failure logger.
        
        Args:
            log_dir: Directory for log files (default: data/logs/silent_failures)
            max_records: Maximum number of records to keep in memory
            enable_console: Whether to log to console
            enable_file: Whether to log to file
            enable_aggregation: Whether to aggregate similar failures
        """
        self.log_dir = log_dir or Path(os.getenv(
            "KINETRA_FAILURE_LOG_DIR",
            "data/logs/silent_failures"
        ))
        self.max_records = max_records
        self.enable_console = enable_console
        self.enable_file = enable_file
        self.enable_aggregation = enable_aggregation
        
        # Create log directory
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # In-memory storage
        self.failures: List[FailureRecord] = []
        self.failure_counts: Dict[str, int] = defaultdict(int)
        self.failure_signatures: Dict[str, FailureRecord] = {}
        
        # Lock for thread safety
        self._write_lock = Lock()
        
        # Initialize logging
        self._setup_logging()
        
        # Session info
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_start = datetime.now()
        
    @classmethod
    def get_instance(cls) -> "SilentFailureLogger":
        """Get singleton instance."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance
    
    @classmethod
    def reset_instance(cls):
        """Reset singleton instance (mainly for testing)."""
        with cls._lock:
            cls._instance = None
    
    def _setup_logging(self):
        """Set up Python logging integration."""
        self.logger = logging.getLogger("kinetra.silent_failures")
        self.logger.setLevel(logging.WARNING)
        
        # Avoid duplicate handlers
        if not self.logger.handlers:
            if self.enable_console:
                console_handler = logging.StreamHandler(sys.stdout)
                console_handler.setLevel(logging.WARNING)
                formatter = logging.Formatter(
                    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
                )
                console_handler.setFormatter(formatter)
                self.logger.addHandler(console_handler)
    
    def _categorize_exception(self, exc: Exception) -> FailureCategory:
        """Automatically categorize an exception."""
        exc_type = type(exc).__name__
        
        category_map = {
            "ImportError": FailureCategory.IMPORT_ERROR,
            "ModuleNotFoundError": FailureCategory.IMPORT_ERROR,
            "TypeError": FailureCategory.TYPE_ERROR,
            "ValueError": FailureCategory.VALUE_ERROR,
            "AttributeError": FailureCategory.ATTRIBUTE_ERROR,
            "KeyError": FailureCategory.KEY_ERROR,
            "IndexError": FailureCategory.INDEX_ERROR,
            "FileNotFoundError": FailureCategory.FILE_ERROR,
            "IOError": FailureCategory.FILE_ERROR,
            "PermissionError": FailureCategory.PERMISSION_ERROR,
            "OSError": FailureCategory.FILE_ERROR,
            "ConnectionError": FailureCategory.NETWORK_ERROR,
            "TimeoutError": FailureCategory.TIMEOUT_ERROR,
            "asyncio.TimeoutError": FailureCategory.TIMEOUT_ERROR,
            "OverflowError": FailureCategory.CALCULATION_ERROR,
            "ZeroDivisionError": FailureCategory.CALCULATION_ERROR,
            "FloatingPointError": FailureCategory.CALCULATION_ERROR,
        }
        
        return category_map.get(exc_type, FailureCategory.UNKNOWN)
    
    def _determine_severity(
        self,
        exc: Exception,
        category: FailureCategory,
        context: Dict[str, Any]
    ) -> FailureSeverity:
        """Determine the severity of a failure."""
        # Critical: System-breaking errors
        if isinstance(exc, (SystemExit, KeyboardInterrupt)):
            return FailureSeverity.CRITICAL
        
        # High: Import errors, permission errors
        if category in (FailureCategory.IMPORT_ERROR, FailureCategory.PERMISSION_ERROR):
            return FailureSeverity.HIGH
        
        # Medium: Type errors, attribute errors
        if category in (
            FailureCategory.TYPE_ERROR,
            FailureCategory.ATTRIBUTE_ERROR,
            FailureCategory.CALCULATION_ERROR
        ):
            return FailureSeverity.MEDIUM
        
        # Check context for severity hints
        if context.get("severity"):
            try:
                return FailureSeverity(context["severity"])
            except (ValueError, KeyError):
                pass
        
        # Default to low
        return FailureSeverity.LOW
    
    def _get_caller_info(self, skip_frames: int = 2) -> Dict[str, Any]:
        """
        Get information about the caller.
        
        Args:
            skip_frames: Number of frames to skip (default: 2 for log_failure call)
        
        Returns:
            Dict with file_path, function_name, line_number, module
        """
        try:
            frame = sys._getframe(skip_frames)
            return {
                "file_path": frame.f_code.co_filename,
                "function_name": frame.f_code.co_name,
                "line_number": frame.f_lineno,
                "module": frame.f_globals.get("__name__", "unknown"),
            }
        except Exception:
            return {
                "file_path": "unknown",
                "function_name": "unknown",
                "line_number": 0,
                "module": "unknown",
            }
    
    def _generate_failure_signature(self, record: FailureRecord) -> str:
        """
        Generate a unique signature for a failure for deduplication.
        
        Uses exception type, file, function, and line number.
        """
        return f"{record.exception_type}:{record.file_path}:{record.function_name}:{record.line_number}"
    
    def log(
        self,
        exception: Exception,
        context: Optional[Dict[str, Any]] = None,
        category: Optional[FailureCategory] = None,
        severity: Optional[FailureSeverity] = None,
        tags: Optional[List[str]] = None,
        skip_frames: int = 2,
    ) -> FailureRecord:
        """
        Log a silent failure.
        
        Args:
            exception: The exception that was caught
            context: Additional context information
            category: Override automatic categorization
            severity: Override automatic severity
            tags: Custom tags for filtering
            skip_frames: Number of stack frames to skip for caller info
        
        Returns:
            FailureRecord that was logged
        """
        # Get caller information
        caller_info = self._get_caller_info(skip_frames=skip_frames)
        
        # Categorize and determine severity
        auto_category = category or self._categorize_exception(exception)
        auto_severity = severity or self._determine_severity(
            exception, auto_category, context or {}
        )
        
        # Create failure record
        timestamp = datetime.now().isoformat()
        record = FailureRecord(
            timestamp=timestamp,
            category=auto_category.value,
            severity=auto_severity.value,
            exception_type=type(exception).__name__,
            exception_message=str(exception),
            file_path=caller_info["file_path"],
            function_name=caller_info["function_name"],
            line_number=caller_info["line_number"],
            module=caller_info["module"],
            context=context or {},
            stack_trace=traceback.format_exc(),
            tags=tags or [],
            first_seen=timestamp,
            last_seen=timestamp,
        )
        
        # Generate signature
        signature = self._generate_failure_signature(record)
        record.failure_id = signature
        
        # Handle aggregation
        with self._write_lock:
            if self.enable_aggregation and signature in self.failure_signatures:
                # Update existing record
                existing = self.failure_signatures[signature]
                existing.count += 1
                existing.last_seen = timestamp
                self.failure_counts[signature] += 1
            else:
                # New failure
                self.failure_signatures[signature] = record
                self.failures.append(record)
                self.failure_counts[signature] = 1
                
                # Enforce max records
                if len(self.failures) > self.max_records:
                    removed = self.failures.pop(0)
                    removed_sig = self._generate_failure_signature(removed)
                    if removed_sig in self.failure_signatures:
                        del self.failure_signatures[removed_sig]
                    if removed_sig in self.failure_counts:
                        del self.failure_counts[removed_sig]
            
            # Write to file
            if self.enable_file:
                self._write_to_file(record)
            
            # Log to console if enabled
            if self.enable_console:
                self.logger.warning(
                    f"Silent failure: {record.exception_type} in "
                    f"{record.function_name} ({record.file_path}:{record.line_number}): "
                    f"{record.exception_message}"
                )
        
        return record
    
    def _write_to_file(self, record: FailureRecord):
        """Write failure record to JSON file."""
        try:
            # Daily log file
            date_str = datetime.now().strftime("%Y%m%d")
            log_file = self.log_dir / f"failures_{date_str}.jsonl"
            
            # Append to JSONL file (one JSON object per line)
            with open(log_file, 'a') as f:
                f.write(record.to_json() + '\n')
        except Exception as e:
            # Avoid infinite loop - just print to stderr with context
            print(
                f"Failed to write failure log to {log_file}: {e}\n"
                f"Original failure: {record.exception_type} - {record.exception_message}",
                file=sys.stderr
            )
    
    def get_failures(
        self,
        category: Optional[str] = None,
        severity: Optional[str] = None,
        module: Optional[str] = None,
        tags: Optional[List[str]] = None,
        limit: Optional[int] = None,
    ) -> List[FailureRecord]:
        """
        Get failures with optional filtering.
        
        Args:
            category: Filter by category
            severity: Filter by severity
            module: Filter by module
            tags: Filter by tags (any match)
            limit: Maximum number of results
        
        Returns:
            List of matching FailureRecords
        """
        results = self.failures.copy()
        
        # Apply filters
        if category:
            results = [r for r in results if r.category == category]
        if severity:
            results = [r for r in results if r.severity == severity]
        if module:
            results = [r for r in results if r.module == module]
        if tags:
            results = [r for r in results if any(t in r.tags for t in tags)]
        
        # Apply limit
        if limit:
            results = results[-limit:]
        
        return results
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about logged failures."""
        total = len(self.failures)
        
        # Count by category
        by_category = defaultdict(int)
        by_severity = defaultdict(int)
        by_module = defaultdict(int)
        
        for record in self.failures:
            by_category[record.category] += record.count
            by_severity[record.severity] += record.count
            by_module[record.module] += record.count
        
        # Top failures
        top_failures = sorted(
            self.failure_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )[:10]
        
        return {
            "total_unique_failures": total,
            "total_failure_count": sum(self.failure_counts.values()),
            "session_id": self.session_id,
            "session_start": self.session_start.isoformat(),
            "session_duration_seconds": (datetime.now() - self.session_start).total_seconds(),
            "by_category": dict(by_category),
            "by_severity": dict(by_severity),
            "by_module": dict(by_module),
            "top_failures": [
                {
                    "signature": sig,
                    "count": count,
                    "record": self.failure_signatures[sig].to_dict()
                }
                for sig, count in top_failures
            ],
        }
    
    def export_report(self, output_file: Optional[Path] = None) -> Path:
        """
        Export a comprehensive failure report.
        
        Args:
            output_file: Output file path (default: auto-generated)
        
        Returns:
            Path to the exported report
        """
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = self.log_dir / f"failure_report_{timestamp}.json"
        
        report = {
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "session_id": self.session_id,
                "session_start": self.session_start.isoformat(),
            },
            "statistics": self.get_statistics(),
            "failures": [r.to_dict() for r in self.failures],
        }
        
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        return output_file
    
    def clear(self):
        """Clear all logged failures."""
        with self._write_lock:
            self.failures.clear()
            self.failure_counts.clear()
            self.failure_signatures.clear()


# Global convenience function
def log_failure(
    exception: Exception,
    context: Optional[Dict[str, Any]] = None,
    category: Optional[FailureCategory] = None,
    severity: Optional[FailureSeverity] = None,
    tags: Optional[List[str]] = None,
) -> FailureRecord:
    """
    Convenience function to log a failure using the global logger.
    
    Args:
        exception: The exception that was caught
        context: Additional context information
        category: Override automatic categorization
        severity: Override automatic severity
        tags: Custom tags for filtering
    
    Returns:
        FailureRecord that was logged
    """
    logger = SilentFailureLogger.get_instance()
    return logger.log(
        exception,
        context=context,
        category=category,
        severity=severity,
        tags=tags,
        skip_frames=3,  # Skip this function frame
    )


def log_failures(
    context: Optional[Dict[str, Any]] = None,
    category: Optional[FailureCategory] = None,
    severity: Optional[FailureSeverity] = None,
    tags: Optional[List[str]] = None,
    reraise: bool = False,
):
    """
    Decorator to automatically log failures in a function.
    
    Args:
        context: Additional context to include
        category: Override automatic categorization
        severity: Override automatic severity
        tags: Custom tags for filtering
        reraise: Whether to re-raise the exception after logging
    
    Example:
        @log_failures(context={"operation": "data_loading"})
        def load_data():
            risky_operation()
    """
    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # Add function context
                func_context = context.copy() if context else {}
                func_context.update({
                    "function": func.__name__,
                    "module": func.__module__,
                    "args_count": len(args),
                    "kwargs_keys": list(kwargs.keys()),
                })
                
                # Log the failure
                log_failure(
                    e,
                    context=func_context,
                    category=category,
                    severity=severity,
                    tags=tags,
                )
                
                # Re-raise if requested
                if reraise:
                    raise
        
        return wrapper
    return decorator


def get_failure_logger() -> SilentFailureLogger:
    """Get the global failure logger instance."""
    return SilentFailureLogger.get_instance()
