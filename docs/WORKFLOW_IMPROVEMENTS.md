# Master Workflow Improvements Documentation

## Overview

The master workflow (`scripts/master_workflow.py`) has been enhanced with comprehensive management features for improved reliability, security, and user experience.

## New Features

### 1. Workflow Manager (`kinetra/workflow_manager.py`)

A comprehensive workflow management system that provides:

#### Atomic File Operations
- **Write-to-temp-then-rename pattern**: Ensures crash-safe file operations
- **Automatic backups**: Creates timestamped backups before modifying files
- **Checksum verification**: SHA256 checksums detect file tampering
- **Rollback capability**: Restore from backups on corruption detection

#### Logging & Monitoring
- **Structured logging**: Logs to both file and console with timestamps
- **Performance tracking**: Measures and reports step execution times
- **Workflow state persistence**: Saves state as JSON for resume capability
- **Debug mode**: Verbose output for troubleshooting

#### Error Handling & Retries
- **Exponential backoff**: Automatic retry with increasing delays
- **Critical vs. non-critical steps**: Graceful degradation for optional steps
- **Error context**: Detailed error messages with actionable suggestions
- **KeyboardInterrupt handling**: Clean shutdown on user interruption

### 2. Enhanced Master Workflow

#### Secure Credential Storage
```python
# Before: Plain text write with basic error handling
with open('.env', 'w') as f:
    f.write(content)

# After: Atomic write with backup and checksum
wf_manager.atomic_write(env_file, content)
# - Creates timestamped backup of existing file
# - Writes to temp file first
# - Computes SHA256 checksum
# - Atomic rename to final location
# - Verifies integrity on next read
```

#### File Tampering Detection
```python
# Verify file integrity before reading
if not wf_manager.verify_file_integrity(env_file):
    # File has been tampered with!
    # Offer to restore from backup
    if wf_manager.restore_from_backup(env_file):
        # Successfully restored
    else:
        # No backup available, prompt for credentials
```

#### Step Execution with Retries
```python
# Execute step with automatic retry logic
success, result = wf_manager.execute_step(
    step_name="Download Data",
    step_func=download_function,
    critical=True,        # Workflow stops if this fails
    max_retries=3         # Try up to 3 times
)
```

#### Progress Tracking & Resume
```python
# Workflow state saved after each step
wf_manager.start_workflow("master_workflow")
wf_manager.execute_step("step1", func1)
wf_manager.execute_step("step2", func2)
# State saved to: .workflow_state/{workflow_id}.json

# Can resume interrupted workflow (future enhancement)
```

## Usage Examples

### Basic Workflow Execution

```bash
# Run the enhanced master workflow
python3 scripts/master_workflow.py
```

The workflow will:
1. Initialize WorkflowManager with logging
2. Check credentials (with tampering detection)
3. Execute each step with retry logic
4. Track performance metrics
5. Save state after each step
6. Generate comprehensive summary

### Working with Workflow Manager Directly

```python
from kinetra.workflow_manager import WorkflowManager

# Initialize
wf = WorkflowManager(
    log_dir="logs/workflow",
    backup_dir="data/backups/workflow",
    enable_backups=True,
    enable_checksums=True,
    max_retries=3
)

# Start workflow
wf.start_workflow("my_workflow", metadata={"version": "1.0"})

# Execute steps
def my_step():
    # Do work
    return "success"

success, result = wf.execute_step(
    "My Step",
    my_step,
    critical=True
)

# Complete workflow
wf.complete_workflow(status="completed")
```

### Atomic File Operations

```python
# Atomic write with backup and checksum
wf.atomic_write(
    Path(".env"),
    "METAAPI_TOKEN=abc123\nMETAAPI_ACCOUNT_ID=xyz789"
)
# Creates: .env
# Creates: data/backups/workflow/.env.20231231_123456.bak
# Creates: .workflow_state/.env.checksum

# Verify integrity
is_valid = wf.verify_file_integrity(Path(".env"))

# Restore from backup if needed
if not is_valid:
    wf.restore_from_backup(Path(".env"))
```

## File Locations

### Logs
- Location: `logs/workflow/workflow_YYYYMMDD_HHMMSS.log`
- Format: Structured text with timestamps
- Retention: Manual cleanup (consider implementing auto-cleanup)

### Backups
- Location: `data/backups/workflow/{filename}.{timestamp}.bak`
- Naming: `{original_name}.YYYYMMDD_HHMMSS.bak`
- Retention: All backups kept (consider implementing rotation)

### Workflow State
- Location: `.workflow_state/{workflow_id}.json`
- Format: JSON with workflow metadata and step details
- Content: Workflow name, status, steps, timing, errors

### Checksums
- Location: `.workflow_state/{filename}.checksum`
- Format: Plain text SHA256 hash
- Purpose: Detect file tampering between runs

## Security Considerations

### Credential Protection
1. **.env files are gitignored**: Automatically added to `.gitignore`
2. **Atomic writes prevent partial exposure**: Temp files in same directory
3. **Backups are in separate directory**: Not exposed to web servers
4. **Checksums detect tampering**: Alert user if file modified externally

### File Permissions
- Workflow creates files with default umask
- Consider setting restrictive permissions for sensitive files:
  ```python
  env_file.chmod(0o600)  # Owner read/write only
  ```

### Backup Security
- Backups contain sensitive data (credentials)
- Ensure backup directory has appropriate permissions
- Consider encrypting backups for production use

## Performance Tracking

### Step Timing
Each step execution is timed automatically:
```
Step: Download Data
  Duration: 45.23s
  Retries: 0
  Status: completed
```

### Performance Summary
At workflow completion:
```
Performance Insights:
  Download Data: 45.23s avg, 45.23s total (65.2%)
  Prepare Data: 18.12s avg, 18.12s total (26.1%)
  Check Integrity: 6.04s avg, 6.04s total (8.7%)
Total Duration: 69.39s
```

### Optimization Opportunities
The performance report helps identify:
- Slowest steps (optimization targets)
- Time distribution across workflow
- Steps that may benefit from parallelization

## Error Handling

### Retry Logic
```
⚠️  Step attempt 1 failed: Connection timeout
🔄 Retry attempt 2/3
⚠️  Step attempt 2 failed: Connection timeout
🔄 Retry attempt 3/3
✅ Step completed on attempt 3
```

### Critical vs. Non-Critical Steps
- **Critical steps**: Workflow stops on failure
- **Non-critical steps**: Warning logged, workflow continues
- **Configurable per step**: `critical=True/False`

### Error Context
Errors include:
- Step name and attempt number
- Error message and type
- Execution context
- Retry count
- Suggestions for resolution

## Testing

### Test Suite
Location: `tests/test_workflow_manager.py`

Tests cover:
1. ✅ WorkflowManager initialization
2. ✅ Atomic write operations
3. ✅ Checksum verification
4. ✅ Backup and restore
5. ✅ Workflow state persistence
6. ✅ Step execution
7. ✅ Retry logic
8. ✅ Error handling
9. ✅ Performance tracking
10. ✅ Timed operations

Run tests:
```bash
python3 tests/test_workflow_manager.py
```

Expected output:
```
Total tests: 10
✅ Passed: 10
❌ Failed: 0
```

## Future Enhancements

### Planned Features (Speed Optimization)
1. **Parallel execution**: Run independent steps concurrently
2. **Data caching**: Avoid redundant computations
3. **Resume capability**: Continue from last successful step
4. **Progress estimation**: Predict remaining time based on history

### Potential Improvements
1. **Backup rotation**: Auto-cleanup old backups (keep last N)
2. **Log rotation**: Auto-cleanup old logs
3. **Encrypted backups**: For production deployments
4. **Remote state storage**: Share state across machines
5. **Workflow visualization**: Generate flowcharts from state
6. **Performance profiling**: Detailed breakdown within steps

## Troubleshooting

### Issue: File integrity check failed
```
⚠️  .env file integrity check failed
```
**Solution**: Workflow will offer to restore from backup. Choose option 1.

### Issue: No backup available
```
⚠️  No backup available
```
**Solution**: Workflow will prompt for credentials again.

### Issue: Step keeps failing
```
❌ Step failed after 3 attempts
```
**Solutions**:
1. Check error message for specific issue
2. Review logs in `logs/workflow/`
3. Verify network connectivity
4. Check file permissions

### Issue: Workflow interrupted
```
⚠️  Workflow interrupted by user
```
**Solution**: Workflow state saved. Re-run to start fresh (resume not yet implemented).

## Migration Guide

### Updating Existing Workflows

Old workflow code:
```python
def main():
    check_credentials()
    run_step("Download", "download.py")
    run_step("Process", "process.py")
```

New workflow code:
```python
def main():
    wf = WorkflowManager()
    wf.start_workflow("my_workflow")
    
    # Credentials with tampering detection
    check_credentials(wf)
    
    # Steps with retry logic
    run_step(wf, "Download", "download.py", critical=True)
    run_step(wf, "Process", "process.py", critical=False)
    
    wf.complete_workflow(status="completed")
```

Benefits:
- Automatic backups before credential changes
- File tampering detection
- Retry logic with exponential backoff
- Performance tracking
- State persistence
- Better error messages

## API Reference

### WorkflowManager

#### Constructor
```python
WorkflowManager(
    log_dir: str = "logs",
    backup_dir: str = "data/backups/workflow",
    state_dir: str = ".workflow_state",
    enable_backups: bool = True,
    enable_checksums: bool = True,
    max_retries: int = 3,
    retry_delay: float = 2.0
)
```

#### Methods

**atomic_write(file_path, content)**
- Atomically write content to file
- Creates backup if file exists
- Computes and stores checksum
- Returns: `bool` (success)

**verify_file_integrity(file_path, expected_checksum=None)**
- Verify file hasn't been tampered with
- Returns: `bool` (is valid)

**backup_file(file_path)**
- Create timestamped backup
- Returns: `Path` to backup or `None`

**restore_from_backup(file_path, backup_path=None)**
- Restore file from backup
- Returns: `bool` (success)

**start_workflow(workflow_name, metadata=None)**
- Initialize new workflow
- Creates workflow state

**execute_step(step_name, step_func, *args, critical=True, max_retries=None, **kwargs)**
- Execute step with retry logic
- Track timing and performance
- Returns: `(bool, Any)` (success, result)

**complete_workflow(status="completed")**
- Finalize workflow
- Generate performance summary
- Save final state

**timed_operation(operation_name)**
- Context manager for timing
- Usage: `with wf.timed_operation("name"): ...`

## Best Practices

1. **Always use atomic_write for sensitive files**
   - Especially for .env, config files, state files

2. **Set appropriate criticality for steps**
   - Critical: Authentication, data download, core processing
   - Non-critical: Reporting, optional optimizations

3. **Enable checksums in production**
   - Detect unauthorized file modifications

4. **Enable backups for important operations**
   - Can roll back on errors

5. **Review performance summaries**
   - Identify optimization opportunities
   - Monitor for performance degradation

6. **Monitor log files**
   - Check for warnings and errors
   - Track workflow execution patterns

7. **Test workflow changes**
   - Run test suite after modifications
   - Verify error handling works correctly

8. **Keep backups secure**
   - Restrict directory permissions
   - Consider encryption for sensitive data

## Conclusion

The enhanced master workflow provides enterprise-grade reliability with:
- ✅ Atomic operations (crash-safe)
- ✅ File tampering detection
- ✅ Automatic backups
- ✅ Comprehensive logging
- ✅ Performance tracking
- ✅ Retry logic
- ✅ State persistence
- ✅ Full test coverage

This makes the workflow robust, maintainable, and production-ready.
