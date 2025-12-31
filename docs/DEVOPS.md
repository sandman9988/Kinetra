# Kinetra DevOps Module

Comprehensive DevOps utilities for development, deployment, and monitoring.

## Overview

The DevOps module (`kinetra.devops`) provides:

| Feature | Module | Description |
|---------|--------|-------------|
| **Deduplication** | `dedup.py` | Find and remove duplicate data/code |
| **Parallelization** | `parallel.py` | Auto-scaling parallel execution |
| **GPU Management** | `gpu.py` | Auto-detect and configure GPUs |
| **Git Sync** | `git_sync.py` | Automatic local/remote sync |
| **Security** | `security.py` | Scan for vulnerabilities |
| **Environment** | `env_manager.py` | Python path/env management |
| **Monitoring** | `monitor.py` | Real-time folder monitoring |

## Quick Start

### DevOps Manager Script

The master script handles all operations:

```bash
# Show all available options
python scripts/devops_manager.py --help

# Run all checks
python scripts/devops_manager.py --all

# Individual operations
python scripts/devops_manager.py --dedup      # Find duplicates
python scripts/devops_manager.py --security   # Security scan
python scripts/devops_manager.py --env        # Setup environment
python scripts/devops_manager.py --gpu        # Configure GPU
python scripts/devops_manager.py --git        # Git sync status
python scripts/devops_manager.py --parallel   # Show parallelization config
python scripts/devops_manager.py --monitor    # Start monitoring
python scripts/devops_manager.py --fix        # Auto-fix issues
```

### Environment Setup

For permanent environment configuration:

```bash
# Load environment (add to ~/.bashrc for permanent setup)
source env_setup.sh

# Install dependencies too
source env_setup.sh --install
```

## Module Details

### 1. Deduplication (`kinetra.devops.dedup`)

Find and remove duplicate files and code.

```python
from kinetra.devops import DataDeduplicator, CodeDeduplicator, find_duplicates

# Find duplicate data files
data_dedup = DataDeduplicator("data/")
duplicates = data_dedup.scan_directory()
print(data_dedup.get_duplicate_report())

# Remove duplicates (dry_run=True to preview)
deleted = data_dedup.remove_duplicates(keep="first", dry_run=False)

# Find duplicate code
code_dedup = CodeDeduplicator("kinetra/")
similar = code_dedup.find_similar_functions()
print(code_dedup.get_report())

# Convenience function
results = find_duplicates(".", check_data=True, check_code=True)
```

### 2. Parallelization (`kinetra.devops.parallel`)

Auto-scaling parallel execution with resource monitoring.

```python
from kinetra.devops import ParallelExecutor, get_optimal_config, AutoScaler

# Get optimal configuration
config = get_optimal_config("cpu")  # 'cpu', 'io', or 'memory'
print(f"Recommended workers: {config.max_workers}")

# Execute in parallel with auto-scaling
executor = ParallelExecutor(auto_scale=True, verbose=True)
results = executor.map(
    func=process_item,
    items=data_list,
    timeout=30.0,
    on_error="continue"  # 'continue', 'stop', or 'raise'
)

# Check statistics
stats = executor.get_stats()
print(f"Success rate: {stats['success_rate']:.1%}")
```

### 3. GPU Management (`kinetra.devops.gpu`)

Auto-detect and configure GPU acceleration.

```python
from kinetra.devops import GPUManager, auto_detect_gpu, get_gpu_info

# Print GPU information
print(get_gpu_info())

# Auto-detect and configure
manager = GPUManager()
gpu = manager.auto_detect()
print(f"GPU Type: {gpu.gpu_type.value}")
print(f"Name: {gpu.name}")

# Configure PyTorch
device = manager.configure_pytorch()  # Returns 'cuda:0', 'mps', or 'cpu'

# Set optimal environment variables
manager.set_optimal_environment()
```

### 4. Git Sync (`kinetra.devops.git_sync`)

Automatic local/remote synchronization.

```python
from kinetra.devops import GitSync, check_sync_status, auto_sync

# Check status
print(check_sync_status("."))

# Manual sync operations
sync = GitSync(".")
status = sync.check_status()
print(f"Branch: {status.branch}")
print(f"Status: {status.sync_status.value}")
print(f"Ahead: {status.ahead_count}, Behind: {status.behind_count}")

# Auto-sync (pull with rebase, optionally push)
success, msg = sync.auto_sync(push_changes=False)

# Start background auto-sync (pulls every 5 minutes)
sync.start_auto_sync(interval_seconds=300)
```

### 5. Security Scanning (`kinetra.devops.security`)

Scan codebase for vulnerabilities.

```python
from kinetra.devops import SecurityScanner, scan_codebase, check_secrets

# Full security scan
print(scan_codebase("."))

# Check only for hardcoded secrets
secrets = check_secrets(".")
for issue in secrets:
    print(f"[{issue.severity.value}] {issue.file}:{issue.line} - {issue.message}")

# Programmatic access
scanner = SecurityScanner(".")
issues = scanner.scan()

# Check dependencies
dep_issues = scanner.check_dependencies()
```

#### Detected Issues:
- Hardcoded secrets (API keys, passwords, tokens)
- AWS credentials
- Database URLs with passwords
- Private keys
- Insecure code patterns (eval, exec, pickle, SQL injection)
- Weak cryptography
- Debug mode enabled
- Unpinned dependencies

### 6. Environment Management (`kinetra.devops.env_manager`)

Permanently resolve Python path and environment issues.

```python
from kinetra.devops import EnvManager, setup_environment, verify_environment

# Quick setup
success, msg = setup_environment()

# Verify environment
result = verify_environment()
print(f"Valid: {result['valid']}")
print(f"Python: {result['python_version']}")
for issue in result['issues']:
    print(f"  ❌ {issue}")

# Full management
manager = EnvManager(".")
manager.setup_environment(create_venv=True)
manager.install_requirements()
print(manager.get_report())
```

#### Generated Files:
- `env_setup.sh` - Bash environment setup
- `.pythonstartup` - Python startup file
- `.envrc` - direnv configuration

### 7. Real-Time Monitoring (`kinetra.devops.monitor`)

Monitor folders for changes, failures, and performance issues.

```python
from kinetra.devops import FolderMonitor, start_monitoring, PerformanceTracker

# Quick start monitoring
def on_event(event):
    print(f"[{event.event_type.value}] {event.message}")

monitor = start_monitoring(
    paths=["kinetra/", "scripts/"],
    on_event=on_event,
    interval=2.0
)

# Get status
status = monitor.get_status()
print(f"Files tracked: {status['files_tracked']}")
print(f"Events: {status['events_recorded']}")

# Stop when done
monitor.stop()

# Performance tracking
tracker = PerformanceTracker()
tracker.start()
# ... later ...
stats = tracker.get_stats()
print(f"CPU avg: {stats['cpu']['avg']:.1f}%")
```

#### Monitored Events:
- File created/modified/deleted
- Performance alerts (CPU, memory, disk)
- Failure detection in log files

### Running as a Daemon

Install as a systemd service:

```bash
# Copy service file
sudo cp scripts/kinetra-monitor.service /etc/systemd/system/

# Create log directory
sudo mkdir -p /var/log/kinetra

# Enable and start
sudo systemctl daemon-reload
sudo systemctl enable kinetra-monitor
sudo systemctl start kinetra-monitor

# Check status
sudo systemctl status kinetra-monitor
journalctl -u kinetra-monitor -f
```

## Convenience Aliases

After sourcing `env_setup.sh`:

```bash
kinetra-devops     # DevOps manager
kinetra-test       # Run tests
kinetra-lint       # Run linter
kinetra-security   # Security scan
kinetra-monitor    # Start monitoring
```

## Best Practices

1. **Run security scan before commits**:
   ```bash
   python scripts/devops_manager.py --security
   ```

2. **Check git sync status regularly**:
   ```bash
   python scripts/devops_manager.py --git
   ```

3. **Set up monitoring for production**:
   ```bash
   # Install as service
   sudo systemctl enable kinetra-monitor
   ```

4. **Source environment in new terminals**:
   ```bash
   echo 'source /workspace/env_setup.sh' >> ~/.bashrc
   ```

5. **Use parallel execution for data processing**:
   ```python
   executor = ParallelExecutor(auto_scale=True)
   results = executor.map(process_func, large_dataset)
   ```
