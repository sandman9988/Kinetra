"""
Workflow Manager for Master Workflow
======================================

Provides comprehensive workflow management with:
- Atomic credential storage with file tampering detection
- Automatic backups and recovery
- Performance measurement and logging
- Step-by-step progress tracking
- Failure handling and retry logic

Usage:
    from kinetra.workflow_manager import WorkflowManager
    
    wf = WorkflowManager()
    wf.start_workflow("master_workflow")
    wf.execute_step("download_data", download_fn, critical=True)
    wf.complete_workflow()
"""

import os
import sys
import json
import time
import hashlib
import logging
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Callable, Any, Tuple
from dataclasses import dataclass, field, asdict
from contextlib import contextmanager


# Instance pool for WorkflowManager
_workflow_manager_pool: Dict[str, "WorkflowManager"] = {}


def _get_pool_key(
    log_dir: str,
    backup_dir: str,
    state_dir: str,
    enable_backups: bool,
    enable_checksums: bool,
    max_retries: int,
    retry_delay: float
) -> str:
    """Generate a unique key for WorkflowManager instance pooling."""
    return f"{log_dir}:{backup_dir}:{state_dir}:{enable_backups}:{enable_checksums}:{max_retries}:{retry_delay}"


@dataclass
class WorkflowStep:
    """Represents a single workflow step."""
    name: str
    status: str = "pending"  # pending, running, completed, failed, skipped
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    duration: Optional[float] = None
    error: Optional[str] = None
    retry_count: int = 0
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class WorkflowState:
    """Complete workflow state for persistence."""
    workflow_name: str
    workflow_id: str
    start_time: float
    end_time: Optional[float] = None
    total_duration: Optional[float] = None
    status: str = "running"  # running, completed, failed, interrupted
    steps: List[WorkflowStep] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'workflow_name': self.workflow_name,
            'workflow_id': self.workflow_id,
            'start_time': self.start_time,
            'end_time': self.end_time,
            'total_duration': self.total_duration,
            'status': self.status,
            'steps': [s.to_dict() for s in self.steps],
            'metadata': self.metadata
        }


class WorkflowManager:
    """
    Comprehensive workflow manager with atomic operations,
    backups, logging, and performance tracking.
    """
    
    def __init__(
        self,
        log_dir: str = "logs",
        backup_dir: str = "data/backups/workflow",
        state_dir: str = ".workflow_state",
        enable_backups: bool = True,
        enable_checksums: bool = True,
        max_retries: int = 3,
        retry_delay: float = 2.0
    ):
        """
        Initialize workflow manager.
        
        Args:
            log_dir: Directory for workflow logs
            backup_dir: Directory for backups
            state_dir: Directory for workflow state
            enable_backups: Enable automatic backups
            enable_checksums: Enable file integrity checks
            max_retries: Maximum retry attempts for failed steps
            retry_delay: Initial delay between retries (seconds)
        """
        self.log_dir = Path(log_dir)
        self.backup_dir = Path(backup_dir)
        self.state_dir = Path(state_dir)
        self.enable_backups = enable_backups
        self.enable_checksums = enable_checksums
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
        # Create directories
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize logging
        self._setup_logging()
        
        # Current workflow state
        self.current_workflow: Optional[WorkflowState] = None
        self.current_step: Optional[WorkflowStep] = None
        
        # Performance metrics
        self.step_timings: Dict[str, List[float]] = {}
    
    def _setup_logging(self):
        """Setup structured logging."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = self.log_dir / f"workflow_{timestamp}.log"
        
        # Configure logger
        self.logger = logging.getLogger("WorkflowManager")
        self.logger.setLevel(logging.DEBUG)
        
        # File handler (detailed)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        
        # Console handler (important only)
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter('%(message)s')
        console_handler.setFormatter(console_formatter)
        
        # Add handlers
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
        self.logger.info(f"Workflow logging initialized: {log_file}")
    
    def compute_checksum(self, file_path: Path) -> str:
        """
        Compute SHA256 checksum of a file.
        
        Args:
            file_path: Path to file
            
        Returns:
            Checksum string
        """
        sha256 = hashlib.sha256()
        try:
            with open(file_path, 'rb') as f:
                while chunk := f.read(8192):
                    sha256.update(chunk)
            return sha256.hexdigest()
        except Exception as e:
            self.logger.error(f"Failed to compute checksum for {file_path}: {e}")
            return ""
    
    def verify_file_integrity(self, file_path: Path, expected_checksum: Optional[str] = None) -> bool:
        """
        Verify file integrity using checksum.
        
        Args:
            file_path: Path to file
            expected_checksum: Expected checksum (if None, just compute and store)
            
        Returns:
            True if file is valid
        """
        if not self.enable_checksums:
            return True
        
        if not file_path.exists():
            self.logger.warning(f"File does not exist: {file_path}")
            return False
        
        current_checksum = self.compute_checksum(file_path)
        
        if expected_checksum:
            is_valid = current_checksum == expected_checksum
            if not is_valid:
                self.logger.error(
                    f"File integrity check FAILED for {file_path}\n"
                    f"  Expected: {expected_checksum}\n"
                    f"  Got:      {current_checksum}"
                )
            return is_valid
        else:
            # Store checksum for future verification
            # Use hash of full path to avoid collisions
            import hashlib
            path_hash = hashlib.md5(str(file_path.absolute()).encode()).hexdigest()[:8]
            checksum_file = self.state_dir / f"{file_path.name}.{path_hash}.checksum"
            try:
                with open(checksum_file, 'w') as f:
                    f.write(current_checksum)
                self.logger.debug(f"Stored checksum for {file_path}: {current_checksum[:16]}...")
            except Exception as e:
                self.logger.warning(f"Could not store checksum: {e}")
            return True
    
    def atomic_write(self, file_path: Path, content: str) -> bool:
        """
        Atomically write content to file with backup.
        
        Args:
            file_path: Target file path
            content: Content to write
            
        Returns:
            True if successful
        """
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Backup existing file
        if self.enable_backups and file_path.exists():
            self.backup_file(file_path)
        
        # Write to temporary file
        temp_file = file_path.parent / f".{file_path.name}.tmp"
        
        try:
            # Write content
            with open(temp_file, 'w') as f:
                f.write(content)
            
            # Compute checksum
            checksum = self.compute_checksum(temp_file)
            
            # Atomic rename
            temp_file.replace(file_path)
            
            # Store checksum (use hash of full path to avoid collisions)
            if self.enable_checksums:
                import hashlib
                path_hash = hashlib.md5(str(file_path.absolute()).encode()).hexdigest()[:8]
                checksum_file = self.state_dir / f"{file_path.name}.{path_hash}.checksum"
                with open(checksum_file, 'w') as f:
                    f.write(checksum)
            
            self.logger.info(f"✅ Atomically wrote to {file_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to write {file_path}: {e}")
            if temp_file.exists():
                temp_file.unlink()
            return False
    
    def backup_file(self, file_path: Path) -> Optional[Path]:
        """
        Create timestamped backup of file.
        
        Args:
            file_path: File to backup
            
        Returns:
            Path to backup file, or None if failed
        """
        if not file_path.exists():
            return None
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"{file_path.name}.{timestamp}.bak"
        backup_path = self.backup_dir / backup_name
        
        try:
            shutil.copy2(file_path, backup_path)
            self.logger.info(f"📦 Backed up {file_path.name} → {backup_name}")
            return backup_path
        except Exception as e:
            self.logger.error(f"❌ Backup failed for {file_path}: {e}")
            return None
    
    def restore_from_backup(self, file_path: Path, backup_path: Optional[Path] = None) -> bool:
        """
        Restore file from backup.
        
        Args:
            file_path: Target file path
            backup_path: Specific backup to restore (or latest if None)
            
        Returns:
            True if successful
        """
        if backup_path is None:
            # Find latest backup
            pattern = f"{file_path.name}.*.bak"
            backups = sorted(self.backup_dir.glob(pattern), reverse=True)
            if not backups:
                self.logger.error(f"No backups found for {file_path}")
                return False
            backup_path = backups[0]
        
        try:
            shutil.copy2(backup_path, file_path)
            self.logger.info(f"🔄 Restored {file_path.name} from {backup_path.name}")
            return True
        except Exception as e:
            self.logger.error(f"❌ Restore failed: {e}")
            return False
    
    def start_workflow(self, workflow_name: str, metadata: Optional[Dict] = None):
        """
        Start a new workflow.
        
        Args:
            workflow_name: Name of the workflow
            metadata: Optional metadata dictionary
        """
        workflow_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        self.current_workflow = WorkflowState(
            workflow_name=workflow_name,
            workflow_id=workflow_id,
            start_time=time.time(),
            metadata=metadata or {}
        )
        
        self.logger.info("=" * 80)
        self.logger.info(f"🚀 WORKFLOW STARTED: {workflow_name}")
        self.logger.info(f"📋 Workflow ID: {workflow_id}")
        self.logger.info("=" * 80)
        
        # Save initial state
        self._save_workflow_state()
    
    def execute_step(
        self,
        step_name: str,
        step_func: Callable,
        *args,
        critical: bool = True,
        max_retries: Optional[int] = None,
        **kwargs
    ) -> Tuple[bool, Any]:
        """
        Execute a workflow step with retry logic and performance tracking.
        
        Args:
            step_name: Name of the step
            step_func: Function to execute
            *args: Positional arguments for step_func
            critical: If True, workflow stops on failure
            max_retries: Override default max retries
            **kwargs: Keyword arguments for step_func
            
        Returns:
            Tuple of (success, result)
        """
        if self.current_workflow is None:
            raise RuntimeError("No active workflow. Call start_workflow() first.")
        
        # Create step
        step = WorkflowStep(name=step_name)
        self.current_workflow.steps.append(step)
        self.current_step = step
        
        max_attempts = max_retries if max_retries is not None else self.max_retries
        retry_delay = self.retry_delay
        
        self.logger.info("")
        self.logger.info("─" * 80)
        self.logger.info(f"▶️  STEP: {step_name}")
        self.logger.info("─" * 80)
        
        step.status = "running"
        step.start_time = time.time()
        self._save_workflow_state()
        
        result = None
        success = False
        
        for attempt in range(1, max_attempts + 1):
            try:
                if attempt > 1:
                    self.logger.warning(f"🔄 Retry attempt {attempt}/{max_attempts}")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                
                # Execute step
                result = step_func(*args, **kwargs)
                success = True
                
                step.status = "completed"
                step.end_time = time.time()
                step.duration = step.end_time - step.start_time
                
                # Track timing
                if step_name not in self.step_timings:
                    self.step_timings[step_name] = []
                self.step_timings[step_name].append(step.duration)
                
                self.logger.info(f"✅ {step_name} completed in {step.duration:.2f}s")
                self._save_workflow_state()
                break
                
            except KeyboardInterrupt:
                step.status = "interrupted"
                step.error = "User interrupted"
                self.logger.warning(f"⚠️  {step_name} interrupted by user")
                self._save_workflow_state()
                raise
                
            except Exception as e:
                step.retry_count = attempt
                error_msg = str(e)
                
                if attempt >= max_attempts:
                    step.status = "failed"
                    step.error = error_msg
                    step.end_time = time.time()
                    step.duration = step.end_time - step.start_time
                    
                    self.logger.error(f"❌ {step_name} failed after {attempt} attempts")
                    self.logger.error(f"   Error: {error_msg}")
                    
                    if critical:
                        self.logger.error(f"   This is a CRITICAL step - workflow cannot continue")
                        self._save_workflow_state()
                        return False, None
                    else:
                        self.logger.warning(f"   Non-critical step - workflow continues")
                        self._save_workflow_state()
                        return False, None
                else:
                    self.logger.warning(f"⚠️  {step_name} attempt {attempt} failed: {error_msg}")
        
        self.current_step = None
        return success, result
    
    def complete_workflow(self, status: str = "completed"):
        """
        Complete the workflow and generate summary.
        
        Args:
            status: Final workflow status (completed, failed, interrupted)
        """
        if self.current_workflow is None:
            return
        
        self.current_workflow.status = status
        self.current_workflow.end_time = time.time()
        self.current_workflow.total_duration = (
            self.current_workflow.end_time - self.current_workflow.start_time
        )
        
        self._save_workflow_state()
        self._print_summary()
        
        self.logger.info("=" * 80)
        self.logger.info(f"🏁 WORKFLOW {status.upper()}")
        self.logger.info("=" * 80)
    
    def _save_workflow_state(self):
        """Save current workflow state to disk."""
        if self.current_workflow is None:
            return
        
        state_file = self.state_dir / f"{self.current_workflow.workflow_id}.json"
        
        try:
            state_json = json.dumps(self.current_workflow.to_dict(), indent=2)
            self.atomic_write(state_file, state_json)
        except Exception as e:
            self.logger.error(f"Failed to save workflow state: {e}")
    
    def _print_summary(self):
        """Print workflow execution summary."""
        if self.current_workflow is None:
            return
        
        wf = self.current_workflow
        
        self.logger.info("")
        self.logger.info("=" * 80)
        self.logger.info("WORKFLOW SUMMARY")
        self.logger.info("=" * 80)
        self.logger.info(f"Workflow: {wf.workflow_name}")
        self.logger.info(f"Status: {wf.status}")
        self.logger.info(f"Total Duration: {wf.total_duration:.2f}s")
        self.logger.info("")
        
        # Step breakdown
        self.logger.info("Steps:")
        for step in wf.steps:
            status_emoji = {
                'completed': '✅',
                'failed': '❌',
                'skipped': '⏭️',
                'interrupted': '⚠️'
            }.get(step.status, '❓')
            
            duration_str = f"{step.duration:.2f}s" if step.duration else "N/A"
            retry_str = f" (retries: {step.retry_count})" if step.retry_count > 0 else ""
            
            self.logger.info(f"  {status_emoji} {step.name}: {duration_str}{retry_str}")
            
            if step.error:
                self.logger.info(f"     Error: {step.error}")
        
        # Performance insights
        if self.step_timings:
            self.logger.info("")
            self.logger.info("Performance Insights:")
            total_measured = sum(sum(times) for times in self.step_timings.values())
            for step_name, times in sorted(self.step_timings.items(), key=lambda x: -sum(x[1])):
                avg_time = sum(times) / len(times)
                total_time = sum(times)
                pct = (total_time / total_measured * 100) if total_measured > 0 else 0
                self.logger.info(f"  {step_name}: {avg_time:.2f}s avg, {total_time:.2f}s total ({pct:.1f}%)")
    
    @contextmanager
    def timed_operation(self, operation_name: str):
        """
        Context manager for timing operations.
        
        Usage:
            with wf.timed_operation("data_processing"):
                process_data()
        """
        self.logger.debug(f"⏱️  Starting: {operation_name}")
        start = time.time()
        try:
            yield
        finally:
            duration = time.time() - start
            self.logger.debug(f"⏱️  Completed: {operation_name} in {duration:.2f}s")
