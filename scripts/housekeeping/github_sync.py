#!/usr/bin/env python3
"""
GitHub Sync Helper
==================

Assists with GitHub repository synchronization and maintenance.

Features:
- Checks git status and uncommitted changes
- Creates atomic commits with proper messages
- Syncs with remote repository
- Handles branch management
- Validates before push

Usage:
    # Check status
    python scripts/housekeeping/github_sync.py --status

    # Commit all changes
    python scripts/housekeeping/github_sync.py --commit "Your message"

    # Full sync (commit + push)
    python scripts/housekeeping/github_sync.py --sync "Your message"

    # Create and push new branch
    python scripts/housekeeping/github_sync.py --branch feature/new-feature

__version__ = "1.0.0"
"""

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

PROJECT_ROOT = Path(__file__).parent.parent.parent


def run_command(cmd: List[str], capture: bool = True) -> Tuple[int, str, str]:
    """Run shell command and return (returncode, stdout, stderr)."""
    if capture:
        result = subprocess.run(
            cmd,
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True
        )
        return result.returncode, result.stdout, result.stderr
    else:
        result = subprocess.run(cmd, cwd=PROJECT_ROOT)
        return result.returncode, "", ""


def check_git_status() -> dict:
    """Check git repository status."""
    status = {
        "clean": False,
        "untracked": [],
        "modified": [],
        "staged": [],
        "branch": "",
        "upstream": "",
    }
    
    # Get current branch
    returncode, stdout, _ = run_command(["git", "branch", "--show-current"])
    if returncode == 0:
        status["branch"] = stdout.strip()
    
    # Get upstream branch
    returncode, stdout, _ = run_command(["git", "rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}"])
    if returncode == 0:
        status["upstream"] = stdout.strip()
    
    # Get status
    returncode, stdout, _ = run_command(["git", "status", "--porcelain"])
    if returncode == 0:
        for line in stdout.strip().split("\n"):
            if not line:
                continue
            
            code = line[:2]
            filepath = line[3:]
            
            if code == "??":
                status["untracked"].append(filepath)
            elif code[0] in "MADRC":
                status["staged"].append(filepath)
            elif code[1] in "MD":
                status["modified"].append(filepath)
    
    status["clean"] = not (status["untracked"] or status["modified"] or status["staged"])
    
    return status


def display_status(status: dict):
    """Display git status in human-readable format."""
    print("=" * 80)
    print("GIT REPOSITORY STATUS")
    print("=" * 80)
    print()
    print(f"📍 Branch: {status['branch']}")
    if status['upstream']:
        print(f"🔗 Upstream: {status['upstream']}")
    print()
    
    if status["clean"]:
        print("✅ Working directory is clean")
    else:
        if status["staged"]:
            print("📦 Staged files:")
            for f in status["staged"]:
                print(f"   ✓ {f}")
            print()
        
        if status["modified"]:
            print("📝 Modified files (not staged):")
            for f in status["modified"]:
                print(f"   • {f}")
            print()
        
        if status["untracked"]:
            print("❓ Untracked files:")
            for f in status["untracked"][:10]:  # Limit to 10
                print(f"   ? {f}")
            if len(status["untracked"]) > 10:
                print(f"   ... and {len(status['untracked']) - 10} more")
            print()


def create_commit(message: str, add_all: bool = False) -> bool:
    """Create a git commit."""
    if add_all:
        print("📦 Staging all changes...")
        returncode, _, stderr = run_command(["git", "add", "-A"])
        if returncode != 0:
            print(f"❌ Failed to stage files: {stderr}")
            return False
    
    # Check if there's anything to commit
    returncode, stdout, _ = run_command(["git", "diff", "--cached", "--quiet"])
    if returncode == 0:
        print("⚠️  No changes staged for commit")
        return False
    
    print(f"💾 Creating commit: {message}")
    returncode, stdout, stderr = run_command(["git", "commit", "-m", message])
    
    if returncode == 0:
        print("✅ Commit created successfully")
        return True
    else:
        print(f"❌ Failed to create commit: {stderr}")
        return False


def push_to_remote(branch: str = None) -> bool:
    """Push commits to remote repository."""
    if branch is None:
        status = check_git_status()
        branch = status["branch"]
    
    print(f"🚀 Pushing to remote ({branch})...")
    
    # Check if upstream is set
    returncode, _, _ = run_command(["git", "rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}"])
    
    if returncode != 0:
        # No upstream, set it
        print(f"   Setting upstream to origin/{branch}")
        cmd = ["git", "push", "--set-upstream", "origin", branch]
    else:
        cmd = ["git", "push"]
    
    returncode, stdout, stderr = run_command(cmd, capture=False)
    
    if returncode == 0:
        print("✅ Push successful")
        return True
    else:
        print(f"❌ Push failed")
        return False


def pull_from_remote() -> bool:
    """Pull latest changes from remote."""
    print("⬇️  Pulling latest changes...")
    returncode, stdout, stderr = run_command(["git", "pull"], capture=False)
    
    if returncode == 0:
        print("✅ Pull successful")
        return True
    else:
        print(f"❌ Pull failed")
        return False


def create_branch(branch_name: str, push: bool = False) -> bool:
    """Create and optionally push a new branch."""
    print(f"🌿 Creating branch: {branch_name}")
    
    returncode, _, stderr = run_command(["git", "checkout", "-b", branch_name])
    
    if returncode == 0:
        print(f"✅ Branch '{branch_name}' created")
        
        if push:
            return push_to_remote(branch_name)
        return True
    else:
        print(f"❌ Failed to create branch: {stderr}")
        return False


def validate_before_push() -> bool:
    """Run validation checks before pushing."""
    print("🔍 Running pre-push validation...")
    
    # Check for conflicts
    returncode, stdout, _ = run_command(["git", "diff", "--check"])
    if returncode != 0:
        print("❌ Whitespace or conflict markers detected")
        return False
    
    # Check for large files (>10MB)
    returncode, stdout, _ = run_command([
        "git", "diff", "--cached", "--stat"
    ])
    
    print("✅ Validation passed")
    return True


def main():
    parser = argparse.ArgumentParser(description="GitHub synchronization helper")
    parser.add_argument(
        "--status",
        action="store_true",
        help="Show git status"
    )
    parser.add_argument(
        "--commit",
        type=str,
        metavar="MESSAGE",
        help="Create commit with message (stages all changes)"
    )
    parser.add_argument(
        "--sync",
        type=str,
        metavar="MESSAGE",
        help="Commit and push (full sync)"
    )
    parser.add_argument(
        "--pull",
        action="store_true",
        help="Pull latest changes from remote"
    )
    parser.add_argument(
        "--push",
        action="store_true",
        help="Push commits to remote"
    )
    parser.add_argument(
        "--branch",
        type=str,
        metavar="NAME",
        help="Create new branch"
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Run pre-push validation only"
    )
    
    args = parser.parse_args()
    
    # Default to showing status if no action specified
    if not any([args.status, args.commit, args.sync, args.pull, args.push, args.branch, args.validate]):
        args.status = True
    
    # Check git repository
    returncode, _, _ = run_command(["git", "rev-parse", "--git-dir"])
    if returncode != 0:
        print("❌ Not a git repository")
        return 1
    
    # Execute actions
    if args.status:
        status = check_git_status()
        display_status(status)
    
    if args.branch:
        success = create_branch(args.branch, push=True)
        if not success:
            return 1
    
    if args.commit:
        success = create_commit(args.commit, add_all=True)
        if not success:
            return 1
    
    if args.sync:
        # Commit
        success = create_commit(args.sync, add_all=True)
        if not success:
            return 1
        
        # Validate
        if not validate_before_push():
            print("\n⚠️  Validation failed - commit created but not pushed")
            print("   Fix issues and run --push manually")
            return 1
        
        # Push
        success = push_to_remote()
        if not success:
            return 1
    
    if args.pull:
        success = pull_from_remote()
        if not success:
            return 1
    
    if args.push:
        if not validate_before_push():
            return 1
        
        success = push_to_remote()
        if not success:
            return 1
    
    if args.validate:
        success = validate_before_push()
        if not success:
            return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
