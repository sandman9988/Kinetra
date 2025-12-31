#!/usr/bin/env python3
"""
Tests for Workflow Manager
===========================

Tests all workflow management features:
- Atomic file operations
- Checksum verification
- Backup and restore
- Workflow state management
- Step execution with retries
- Performance tracking
- Error handling
"""

import os
import sys
import time
import tempfile
import shutil
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

# Direct import to avoid heavy dependencies
import importlib.util
spec = importlib.util.spec_from_file_location(
    "workflow_manager",
    Path(__file__).parent.parent / "kinetra" / "workflow_manager.py"
)
workflow_manager = importlib.util.module_from_spec(spec)
spec.loader.exec_module(workflow_manager)

WorkflowManager = workflow_manager.WorkflowManager
WorkflowStep = workflow_manager.WorkflowStep
WorkflowState = workflow_manager.WorkflowState


def test_workflow_manager_initialization():
    """Test 1: WorkflowManager initializes correctly."""
    print("\n" + "=" * 80)
    print("TEST 1: WorkflowManager Initialization")
    print("=" * 80)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        wf = WorkflowManager(
            log_dir=f"{tmpdir}/logs",
            backup_dir=f"{tmpdir}/backups",
            state_dir=f"{tmpdir}/state"
        )
        
        # Check directories created
        assert Path(f"{tmpdir}/logs").exists(), "Log directory not created"
        assert Path(f"{tmpdir}/backups").exists(), "Backup directory not created"
        assert Path(f"{tmpdir}/state").exists(), "State directory not created"
        
        # Check logger initialized
        assert wf.logger is not None, "Logger not initialized"
        
        print("✅ WorkflowManager initialized successfully")
        print(f"   Log dir: {wf.log_dir}")
        print(f"   Backup dir: {wf.backup_dir}")
        print(f"   State dir: {wf.state_dir}")


def test_atomic_write():
    """Test 2: Atomic write operations."""
    print("\n" + "=" * 80)
    print("TEST 2: Atomic Write Operations")
    print("=" * 80)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        wf = WorkflowManager(
            log_dir=f"{tmpdir}/logs",
            backup_dir=f"{tmpdir}/backups",
            state_dir=f"{tmpdir}/state"
        )
        
        test_file = Path(tmpdir) / "test.txt"
        content = "Hello, Atomic World!"
        
        # Write content
        success = wf.atomic_write(test_file, content)
        assert success, "Atomic write failed"
        assert test_file.exists(), "File not created"
        
        # Verify content
        with open(test_file, 'r') as f:
            read_content = f.read()
        assert read_content == content, "Content mismatch"
        
        # Verify checksum stored (check for any checksum file for this filename)
        checksum_files = list(wf.state_dir.glob(f"{test_file.name}.*.checksum"))
        assert len(checksum_files) > 0, "Checksum not stored"
        
        print("✅ Atomic write successful")
        print(f"   File: {test_file}")
        print(f"   Size: {test_file.stat().st_size} bytes")
        print(f"   Checksum: {checksum_files[0].read_text()[:16]}...")


def test_checksum_verification():
    """Test 3: File integrity checks."""
    print("\n" + "=" * 80)
    print("TEST 3: Checksum Verification")
    print("=" * 80)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        wf = WorkflowManager(
            log_dir=f"{tmpdir}/logs",
            backup_dir=f"{tmpdir}/backups",
            state_dir=f"{tmpdir}/state",
            enable_checksums=True
        )
        
        test_file = Path(tmpdir) / "test.txt"
        content = "Test content for checksum"
        
        # Write and get checksum
        wf.atomic_write(test_file, content)
        checksum1 = wf.compute_checksum(test_file)
        
        # Verify integrity (should pass)
        is_valid = wf.verify_file_integrity(test_file, checksum1)
        assert is_valid, "Valid file failed integrity check"
        
        # Tamper with file
        with open(test_file, 'w') as f:
            f.write("Tampered content")
        
        # Verify integrity (should fail)
        is_valid = wf.verify_file_integrity(test_file, checksum1)
        assert not is_valid, "Tampered file passed integrity check"
        
        print("✅ Checksum verification working")
        print(f"   Original checksum: {checksum1[:16]}...")
        print(f"   Detected tampering correctly")


def test_backup_and_restore():
    """Test 4: Backup and restore functionality."""
    print("\n" + "=" * 80)
    print("TEST 4: Backup and Restore")
    print("=" * 80)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        wf = WorkflowManager(
            log_dir=f"{tmpdir}/logs",
            backup_dir=f"{tmpdir}/backups",
            state_dir=f"{tmpdir}/state",
            enable_backups=True
        )
        
        test_file = Path(tmpdir) / "test.txt"
        original_content = "Original content"
        modified_content = "Modified content"
        
        # Create original file
        wf.atomic_write(test_file, original_content)
        
        # Modify file (should create backup)
        wf.atomic_write(test_file, modified_content)
        
        # Check backup exists
        backups = list(wf.backup_dir.glob("test.txt.*.bak"))
        assert len(backups) > 0, "No backup created"
        
        # Verify backup content
        with open(backups[0], 'r') as f:
            backup_content = f.read()
        assert backup_content == original_content, "Backup content mismatch"
        
        # Restore from backup
        success = wf.restore_from_backup(test_file)
        assert success, "Restore failed"
        
        # Verify restored content
        with open(test_file, 'r') as f:
            restored_content = f.read()
        assert restored_content == original_content, "Restored content mismatch"
        
        print("✅ Backup and restore working")
        print(f"   Backups created: {len(backups)}")
        print(f"   Successfully restored original content")


def test_workflow_state_persistence():
    """Test 5: Workflow state saving and loading."""
    print("\n" + "=" * 80)
    print("TEST 5: Workflow State Persistence")
    print("=" * 80)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        wf = WorkflowManager(
            log_dir=f"{tmpdir}/logs",
            backup_dir=f"{tmpdir}/backups",
            state_dir=f"{tmpdir}/state"
        )
        
        # Start workflow
        wf.start_workflow("test_workflow", metadata={"version": "1.0"})
        
        # Check state file created
        state_files = list(wf.state_dir.glob("*.json"))
        assert len(state_files) > 0, "No state file created"
        
        # Verify state content
        import json
        with open(state_files[0], 'r') as f:
            state_data = json.load(f)
        
        assert state_data['workflow_name'] == "test_workflow", "Workflow name mismatch"
        assert state_data['metadata']['version'] == "1.0", "Metadata mismatch"
        assert state_data['status'] == "running", "Status should be running"
        
        print("✅ Workflow state persistence working")
        print(f"   State file: {state_files[0].name}")
        print(f"   Workflow: {state_data['workflow_name']}")
        print(f"   Status: {state_data['status']}")


def test_step_execution():
    """Test 6: Step execution with timing."""
    print("\n" + "=" * 80)
    print("TEST 6: Step Execution")
    print("=" * 80)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        wf = WorkflowManager(
            log_dir=f"{tmpdir}/logs",
            backup_dir=f"{tmpdir}/backups",
            state_dir=f"{tmpdir}/state"
        )
        
        wf.start_workflow("test_workflow")
        
        # Execute successful step
        def successful_step():
            time.sleep(0.1)  # Simulate work
            return "Success"
        
        success, result = wf.execute_step(
            "test_step",
            successful_step,
            critical=True
        )
        
        assert success, "Step should succeed"
        assert result == "Success", "Result mismatch"
        assert len(wf.current_workflow.steps) == 1, "Step not recorded"
        
        # Check step details
        step = wf.current_workflow.steps[0]
        assert step.name == "test_step", "Step name mismatch"
        assert step.status == "completed", "Step status should be completed"
        assert step.duration > 0, "Duration not recorded"
        
        print("✅ Step execution working")
        print(f"   Step: {step.name}")
        print(f"   Status: {step.status}")
        print(f"   Duration: {step.duration:.3f}s")


def test_retry_logic():
    """Test 7: Retry logic with exponential backoff."""
    print("\n" + "=" * 80)
    print("TEST 7: Retry Logic")
    print("=" * 80)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        wf = WorkflowManager(
            log_dir=f"{tmpdir}/logs",
            backup_dir=f"{tmpdir}/backups",
            state_dir=f"{tmpdir}/state",
            max_retries=3,
            retry_delay=0.1
        )
        
        wf.start_workflow("test_workflow")
        
        # Create step that fails twice then succeeds
        attempt_count = [0]
        
        def flaky_step():
            attempt_count[0] += 1
            if attempt_count[0] < 3:
                raise RuntimeError(f"Attempt {attempt_count[0]} failed")
            return "Success on attempt 3"
        
        success, result = wf.execute_step(
            "flaky_step",
            flaky_step,
            critical=False,
            max_retries=3
        )
        
        assert success, "Step should eventually succeed"
        assert attempt_count[0] == 3, f"Expected 3 attempts, got {attempt_count[0]}"
        
        # Check retry count recorded
        step = wf.current_workflow.steps[0]
        assert step.retry_count == 2, f"Expected 2 retries, got {step.retry_count}"
        
        print("✅ Retry logic working")
        print(f"   Attempts: {attempt_count[0]}")
        print(f"   Retries recorded: {step.retry_count}")


def test_error_handling():
    """Test 8: Error handling for critical steps."""
    print("\n" + "=" * 80)
    print("TEST 8: Error Handling")
    print("=" * 80)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        wf = WorkflowManager(
            log_dir=f"{tmpdir}/logs",
            backup_dir=f"{tmpdir}/backups",
            state_dir=f"{tmpdir}/state",
            max_retries=2
        )
        
        wf.start_workflow("test_workflow")
        
        # Create step that always fails
        def failing_step():
            raise RuntimeError("This step always fails")
        
        # Test critical step (should return False)
        success, result = wf.execute_step(
            "critical_failing_step",
            failing_step,
            critical=True,
            max_retries=2
        )
        
        assert not success, "Critical step should fail"
        
        # Check error recorded
        step = wf.current_workflow.steps[0]
        assert step.status == "failed", "Status should be failed"
        assert step.error is not None, "Error should be recorded"
        assert "always fails" in step.error, "Error message not captured"
        
        print("✅ Error handling working")
        print(f"   Status: {step.status}")
        print(f"   Error: {step.error}")


def test_performance_tracking():
    """Test 9: Performance measurement and reporting."""
    print("\n" + "=" * 80)
    print("TEST 9: Performance Tracking")
    print("=" * 80)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        wf = WorkflowManager(
            log_dir=f"{tmpdir}/logs",
            backup_dir=f"{tmpdir}/backups",
            state_dir=f"{tmpdir}/state"
        )
        
        wf.start_workflow("test_workflow")
        
        # Execute multiple steps with different durations
        def fast_step():
            time.sleep(0.05)
            return "Fast"
        
        def slow_step():
            time.sleep(0.15)
            return "Slow"
        
        wf.execute_step("fast_step_1", fast_step)
        wf.execute_step("slow_step", slow_step)
        wf.execute_step("fast_step_2", fast_step)
        
        # Check timing tracked
        assert "fast_step_1" in wf.step_timings, "Fast step 1 not tracked"
        assert "slow_step" in wf.step_timings, "Slow step not tracked"
        assert "fast_step_2" in wf.step_timings, "Fast step 2 not tracked"
        
        # Complete workflow
        wf.complete_workflow(status="completed")
        
        # Verify total duration
        assert wf.current_workflow.total_duration > 0, "Total duration not recorded"
        
        print("✅ Performance tracking working")
        print(f"   Total duration: {wf.current_workflow.total_duration:.3f}s")
        for step_name, times in wf.step_timings.items():
            print(f"   {step_name}: {sum(times):.3f}s")


def test_timed_operation_context():
    """Test 10: Timed operation context manager."""
    print("\n" + "=" * 80)
    print("TEST 10: Timed Operation Context")
    print("=" * 80)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        wf = WorkflowManager(
            log_dir=f"{tmpdir}/logs",
            backup_dir=f"{tmpdir}/backups",
            state_dir=f"{tmpdir}/state"
        )
        
        # Use timed operation context
        with wf.timed_operation("test_operation"):
            time.sleep(0.1)
        
        # Check logs contain timing info (we can't easily verify without parsing logs,
        # but we can at least ensure it doesn't crash)
        print("✅ Timed operation context working")


def run_all_tests():
    """Run all workflow manager tests."""
    print("\n" + "=" * 80)
    print("WORKFLOW MANAGER TEST SUITE")
    print("=" * 80)
    
    tests = [
        test_workflow_manager_initialization,
        test_atomic_write,
        test_checksum_verification,
        test_backup_and_restore,
        test_workflow_state_persistence,
        test_step_execution,
        test_retry_logic,
        test_error_handling,
        test_performance_tracking,
        test_timed_operation_context,
    ]
    
    passed = 0
    failed = 0
    
    for test_func in tests:
        try:
            test_func()
            passed += 1
        except AssertionError as e:
            print(f"\n❌ {test_func.__name__} FAILED: {e}")
            failed += 1
        except Exception as e:
            print(f"\n❌ {test_func.__name__} ERROR: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print(f"Total tests: {len(tests)}")
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")
    print("=" * 80)
    
    return failed == 0


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
