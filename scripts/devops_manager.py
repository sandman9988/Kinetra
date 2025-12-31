#!/usr/bin/env python3
"""
Kinetra DevOps Manager
======================

Master script for all DevOps operations:
- Deduplication
- Parallelization config
- GPU setup
- Git sync
- Security scanning
- Environment setup
- Real-time monitoring
- File backup and integrity

Usage:
    python scripts/devops_manager.py --all           # Run all checks
    python scripts/devops_manager.py --dedup        # Find duplicates
    python scripts/devops_manager.py --security     # Security scan
    python scripts/devops_manager.py --env          # Setup environment
    python scripts/devops_manager.py --gpu          # Configure GPU
    python scripts/devops_manager.py --git          # Git sync status
    python scripts/devops_manager.py --monitor      # Start monitoring
    python scripts/devops_manager.py --backup       # Backup critical files
    python scripts/devops_manager.py --integrity    # Verify file integrity
    python scripts/devops_manager.py --fix          # Auto-fix issues
"""

import argparse
import os
import sys
from pathlib import Path

# Add workspace to path
workspace = Path(__file__).parent.parent
sys.path.insert(0, str(workspace))

from kinetra.devops import (
    BackupManager,
    CodeDeduplicator,
    # Deduplication
    DataDeduplicator,
    # Environment
    EnvManager,
    GitSync,
    GPUManager,
    IntegrityChecker,
    # Security
    SecurityScanner,
    check_sync_status,
    get_gpu_info,
    get_optimal_config,
    scan_codebase,
    # Monitoring
    start_monitoring,
    verify_environment,
)


def print_header(title: str):
    """Print section header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def run_dedup(remove: bool = False):
    """Run deduplication analysis."""
    print_header("DEDUPLICATION ANALYSIS")

    # Data files
    print("\n[1/2] Scanning for duplicate data files...")
    data_dedup = DataDeduplicator(str(workspace / "data"))
    print(data_dedup.get_duplicate_report())

    # Code duplication
    print("\n[2/2] Scanning for duplicate code...")
    code_dedup = CodeDeduplicator(str(workspace / "kinetra"))
    print(code_dedup.get_report())

    if remove:
        print("\n[Auto-Remove] Removing duplicate data files...")
        deleted = data_dedup.remove_duplicates(keep="first", dry_run=False)
        print(f"Deleted {len(deleted)} duplicate files")


def run_security():
    """Run security scan."""
    print_header("SECURITY SCAN")
    print(scan_codebase(str(workspace)))


def run_env_setup():
    """Setup environment."""
    print_header("ENVIRONMENT SETUP")

    manager = EnvManager(str(workspace))

    # Setup
    print("\n[1/3] Setting up environment...")
    success, msg = manager.setup_environment(create_venv=False)  # Don't create venv in existing project
    print(f"  {'✅' if success else '❌'} {msg}")

    # Verify
    print("\n[2/3] Verifying environment...")
    verification = manager.verify_environment()
    for key, value in verification.items():
        if key != 'issues':
            print(f"  {key}: {value}")

    if verification['issues']:
        print("\n  Issues found:")
        for issue in verification['issues']:
            print(f"    ❌ {issue}")
    else:
        print("\n  ✅ No issues found")

    # Report
    print("\n[3/3] Environment Report:")
    print(manager.get_report())


def run_gpu_setup():
    """Setup GPU."""
    print_header("GPU CONFIGURATION")

    print(get_gpu_info())

    # Configure
    manager = GPUManager()
    manager.auto_detect()

    print("\nConfiguring optimal GPU settings...")
    device = manager.configure_pytorch()
    manager.set_optimal_environment()

    print(f"\n✅ GPU configured. PyTorch device: {device}")

    # Print environment variables set
    print("\nEnvironment variables set:")
    for key in ['HSA_OVERRIDE_GFX_VERSION', 'PYTORCH_HIP_ALLOC_CONF',
                'GPU_MAX_HW_QUEUES', 'HIP_VISIBLE_DEVICES',
                'CUDA_DEVICE_ORDER']:
        value = os.environ.get(key)
        if value:
            print(f"  {key}={value}")


def run_git_sync():
    """Run git sync check."""
    print_header("GIT SYNC STATUS")
    print(check_sync_status(str(workspace)))


def run_parallel_config():
    """Show parallelization configuration."""
    print_header("PARALLELIZATION CONFIGURATION")

    print("\nOptimal configurations by task type:")
    for task_type in ['cpu', 'io', 'memory']:
        config = get_optimal_config(task_type)
        print(f"\n  {task_type.upper()}-bound tasks:")
        print(f"    Max workers: {config.max_workers}")
        print(f"    Use processes: {config.use_processes}")
        print(f"    Memory limit: {config.memory_limit_pct*100:.0f}%")


def run_monitoring():
    """Start real-time monitoring."""
    print_header("REAL-TIME MONITORING")

    print("\nStarting folder monitor...")
    print("Press Ctrl+C to stop\n")

    def on_event(event):
        print(f"[{event.timestamp.strftime('%H:%M:%S')}] [{event.severity.upper()}] {event.message}")

    monitor = start_monitoring(
        paths=[str(workspace / "kinetra"), str(workspace / "scripts")],
        on_event=on_event,
        interval=2.0
    )

    try:
        while True:
            import time
            time.sleep(10)
            status = monitor.get_status()
            perf = status.get('performance', {})
            cpu = perf.get('cpu', {}).get('current', 0)
            mem = perf.get('memory', {}).get('current', 0)
            print(f"\n[Status] Files: {status['files_tracked']} | CPU: {cpu:.1f}% | Memory: {mem:.1f}%")
    except KeyboardInterrupt:
        print("\nStopping monitor...")
        monitor.stop()


def run_backup():
    """Backup critical files."""
    print_header("FILE BACKUP")

    backup_dir = workspace / "backups"
    manager = BackupManager(str(backup_dir), max_backups=10, compress=False)

    # Critical files to backup
    critical_files = [
        "pyproject.toml",
        "requirements.txt",
        ".env",
        "kinetra/__init__.py",
    ]

    print(f"\nBackup directory: {backup_dir}")
    print("\nBacking up critical files...")

    for file_path in critical_files:
        full_path = workspace / file_path
        if full_path.exists():
            info = manager.create(full_path)
            if info:
                print(f"  ✅ {file_path} -> {Path(info.backup_path).name}")
            else:
                print(f"  ❌ {file_path} - backup failed")
        else:
            print(f"  ⏭️  {file_path} - not found")

    # Show stats
    stats = manager.get_stats()
    print("\nBackup Statistics:")
    print(f"  Total backups: {stats['total_files']}")
    print(f"  Total size: {stats['total_size_mb']:.2f} MB")
    if stats['oldest_backup']:
        print(f"  Oldest: {stats['oldest_backup'].strftime('%Y-%m-%d %H:%M')}")
    if stats['newest_backup']:
        print(f"  Newest: {stats['newest_backup'].strftime('%Y-%m-%d %H:%M')}")


def run_integrity():
    """Verify file integrity."""
    print_header("FILE INTEGRITY CHECK")

    checker = IntegrityChecker(algorithm="sha256")
    manifest_path = workspace / ".file_manifest.json"

    # Critical directories to check
    check_dirs = ["kinetra", "scripts"]

    if manifest_path.exists():
        print("\nLoading existing manifest...")
        manifest = checker.load_manifest(manifest_path)
        print(f"  Manifest has {len(manifest)} files")

        print("\nVerifying integrity...")
        results = checker.verify_manifest(workspace, manifest)

        ok_count = sum(1 for s in results.values() if s == "ok")
        modified_count = sum(1 for s in results.values() if s == "modified")
        missing_count = sum(1 for s in results.values() if s == "missing")
        new_count = sum(1 for s in results.values() if s == "new")

        print(f"\n  ✅ OK: {ok_count}")
        print(f"  📝 Modified: {modified_count}")
        print(f"  ❌ Missing: {missing_count}")
        print(f"  🆕 New: {new_count}")

        if modified_count > 0:
            print("\n  Modified files:")
            for path, status in results.items():
                if status == "modified":
                    print(f"    - {path}")

        if missing_count > 0:
            print("\n  Missing files:")
            for path, status in results.items():
                if status == "missing":
                    print(f"    - {path}")

    else:
        print("\nNo manifest found. Creating new manifest...")

    # Create/update manifest
    print("\nCreating integrity manifest...")
    manifest = {}
    for check_dir in check_dirs:
        dir_path = workspace / check_dir
        if dir_path.exists():
            dir_manifest = checker.create_manifest(dir_path, patterns=["*.py"])
            # Prefix with directory name
            for rel_path, checksum in dir_manifest.items():
                manifest[f"{check_dir}/{rel_path}"] = checksum

    checker.save_manifest(manifest, manifest_path)
    print(f"  Saved manifest with {len(manifest)} files to {manifest_path.name}")


def run_all():
    """Run all checks."""
    run_env_setup()
    run_gpu_setup()
    run_parallel_config()
    run_git_sync()
    run_security()
    run_dedup(remove=False)
    run_integrity()


def run_fix():
    """Auto-fix detected issues."""
    print_header("AUTO-FIX MODE")

    # 1. Setup environment
    print("\n[1/4] Fixing environment...")
    manager = EnvManager(str(workspace))
    success, msg = manager.setup_environment(create_venv=False)
    print(f"  {'✅' if success else '❌'} {msg}")

    # 2. Configure GPU
    print("\n[2/4] Configuring GPU...")
    gpu_manager = GPUManager()
    gpu_manager.auto_detect()
    gpu_manager.set_optimal_environment()
    device = gpu_manager.configure_pytorch()
    print(f"  ✅ GPU device: {device}")

    # 3. Git sync
    print("\n[3/4] Syncing with remote...")
    git = GitSync(str(workspace))
    success, msg = git.auto_sync(push_changes=False)  # Pull only, don't push
    print(f"  {'✅' if success else '❌'} {msg}")

    # 4. Report remaining issues
    print("\n[4/4] Remaining issues:")

    # Check security
    scanner = SecurityScanner(str(workspace))
    issues = scanner.scan()
    critical_high = [i for i in issues if i.severity.value in ('critical', 'high')]
    if critical_high:
        print(f"  ⚠️  {len(critical_high)} security issues need manual review")
    else:
        print("  ✅ No critical security issues")

    # Check environment
    verification = verify_environment()
    if verification['issues']:
        print(f"  ⚠️  {len(verification['issues'])} environment issues:")
        for issue in verification['issues']:
            print(f"      - {issue}")
    else:
        print("  ✅ Environment verified")


def main():
    parser = argparse.ArgumentParser(
        description="Kinetra DevOps Manager",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('--all', action='store_true', help='Run all checks')
    parser.add_argument('--dedup', action='store_true', help='Find duplicates')
    parser.add_argument('--dedup-remove', action='store_true', help='Remove duplicate data files')
    parser.add_argument('--security', action='store_true', help='Security scan')
    parser.add_argument('--env', action='store_true', help='Setup environment')
    parser.add_argument('--gpu', action='store_true', help='Configure GPU')
    parser.add_argument('--git', action='store_true', help='Git sync status')
    parser.add_argument('--parallel', action='store_true', help='Show parallelization config')
    parser.add_argument('--monitor', action='store_true', help='Start real-time monitoring')
    parser.add_argument('--backup', action='store_true', help='Backup critical files')
    parser.add_argument('--integrity', action='store_true', help='Verify file integrity')
    parser.add_argument('--fix', action='store_true', help='Auto-fix issues')

    args = parser.parse_args()

    # If no arguments, show help
    if not any(vars(args).values()):
        parser.print_help()
        return

    # Run requested operations
    if args.all:
        run_all()
    else:
        if args.dedup:
            run_dedup(remove=False)
        if args.dedup_remove:
            run_dedup(remove=True)
        if args.security:
            run_security()
        if args.env:
            run_env_setup()
        if args.gpu:
            run_gpu_setup()
        if args.git:
            run_git_sync()
        if args.parallel:
            run_parallel_config()
        if args.monitor:
            run_monitoring()
        if args.backup:
            run_backup()
        if args.integrity:
            run_integrity()
        if args.fix:
            run_fix()


if __name__ == "__main__":
    main()
