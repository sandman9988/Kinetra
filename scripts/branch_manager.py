#!/usr/bin/env python3
"""
Branch Management Helper Script

Provides utilities for managing local and remote Git branches,
ensuring proper synchronization between local development,
local main, and cloud/remote main branches.

Usage:
    python scripts/branch_manager.py --setup     # Initial setup
    python scripts/branch_manager.py --sync      # Sync with remote
    python scripts/branch_manager.py --status    # Show branch status
    python scripts/branch_manager.py --cleanup   # Clean merged branches
    python scripts/branch_manager.py --list      # List all branches
"""

import argparse
import subprocess
import sys
from typing import List, Tuple, Optional
from pathlib import Path


class BranchManager:
    """Manages Git branch operations for Kinetra repository."""
    
    def __init__(self, repo_path: str = "."):
        self.repo_path = Path(repo_path)
        
    def run_git(self, *args: str, check: bool = True) -> Tuple[int, str, str]:
        """Run a git command and return (returncode, stdout, stderr)."""
        cmd = ["git"] + list(args)
        try:
            result = subprocess.run(
                cmd,
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                check=check
            )
            return result.returncode, result.stdout.strip(), result.stderr.strip()
        except subprocess.CalledProcessError as e:
            return e.returncode, e.stdout.strip() if e.stdout else "", e.stderr.strip() if e.stderr else ""
    
    def get_current_branch(self) -> Optional[str]:
        """Get the name of the current branch."""
        code, stdout, _ = self.run_git("branch", "--show-current", check=False)
        return stdout if code == 0 else None
    
    def branch_exists(self, branch: str, remote: bool = False) -> bool:
        """Check if a branch exists locally or remotely."""
        if remote:
            code, _, _ = self.run_git("ls-remote", "--exit-code", "--heads", "origin", branch, check=False)
        else:
            code, _, _ = self.run_git("show-ref", "--verify", "--quiet", f"refs/heads/{branch}", check=False)
        return code == 0
    
    def setup_local_main(self) -> bool:
        """Set up local main branch tracking remote main."""
        print("🔧 Setting up local main branch...")
        
        # Check if remote main exists
        if not self.branch_exists("main", remote=True):
            print("❌ Remote main branch does not exist!")
            return False
        
        # Fetch latest from remote
        print("📥 Fetching from remote...")
        code, _, stderr = self.run_git("fetch", "origin", "main", check=False)
        if code != 0:
            print(f"❌ Failed to fetch: {stderr}")
            return False
        
        # Check if local main exists
        if self.branch_exists("main"):
            print("✅ Local main branch already exists")
            # Update it to track origin/main
            code, _, stderr = self.run_git("branch", "--set-upstream-to=origin/main", "main", check=False)
            if code != 0:
                print(f"⚠️  Warning: Could not set upstream: {stderr}")
        else:
            # Create local main from origin/main
            print("📌 Creating local main branch from origin/main...")
            code, _, stderr = self.run_git("checkout", "-b", "main", "origin/main", check=False)
            if code != 0:
                print(f"❌ Failed to create local main: {stderr}")
                return False
        
        print("✅ Local main branch is now tracking origin/main")
        return True
    
    def sync_with_remote(self) -> bool:
        """Sync current branch with remote."""
        current_branch = self.get_current_branch()
        if not current_branch:
            print("❌ Not on any branch (detached HEAD?)")
            return False
        
        print(f"🔄 Syncing branch: {current_branch}")
        
        # Fetch all remotes
        print("📥 Fetching from origin...")
        code, _, stderr = self.run_git("fetch", "origin", check=False)
        if code != 0:
            print(f"❌ Failed to fetch: {stderr}")
            return False
        
        # Check if remote branch exists
        if not self.branch_exists(current_branch, remote=True):
            print(f"⚠️  Remote branch origin/{current_branch} does not exist")
            print("💡 Push your branch with: git push -u origin {current_branch}")
            return True
        
        # Get status
        code, stdout, _ = self.run_git("rev-list", "--left-right", "--count", 
                                       f"HEAD...origin/{current_branch}", check=False)
        if code == 0 and stdout:
            ahead, behind = stdout.split()
            
            if ahead == "0" and behind == "0":
                print("✅ Branch is up to date with remote")
            elif behind != "0" and ahead == "0":
                print(f"📥 Pulling {behind} commit(s) from remote...")
                code, _, stderr = self.run_git("pull", "--rebase", check=False)
                if code == 0:
                    print("✅ Successfully synced with remote")
                else:
                    print(f"❌ Failed to pull: {stderr}")
                    return False
            elif ahead != "0" and behind == "0":
                print(f"📤 Your branch is {ahead} commit(s) ahead of remote")
                print("💡 Push with: git push origin {current_branch}")
            else:
                print(f"⚠️  Branch has diverged (ahead: {ahead}, behind: {behind})")
                print("💡 Consider rebasing: git pull --rebase")
        
        return True
    
    def show_status(self) -> None:
        """Show comprehensive branch status."""
        print("=" * 60)
        print("📊 Kinetra Branch Status")
        print("=" * 60)
        
        # Current branch
        current = self.get_current_branch()
        print(f"\n📍 Current Branch: {current or 'DETACHED HEAD'}")
        
        # Local branches
        print("\n🏠 Local Branches:")
        code, stdout, _ = self.run_git("branch", "-vv", check=False)
        if code == 0:
            for line in stdout.split('\n'):
                print(f"  {line}")
        
        # Check main branch setup
        print("\n🎯 Main Branch Status:")
        if self.branch_exists("main"):
            code, stdout, _ = self.run_git("rev-list", "--left-right", "--count", 
                                          "main...origin/main", check=False)
            if code == 0 and stdout:
                ahead, behind = stdout.split()
                if ahead == "0" and behind == "0":
                    print("  ✅ Local main is in sync with origin/main")
                elif behind != "0":
                    print(f"  ⚠️  Local main is {behind} commit(s) behind origin/main")
                elif ahead != "0":
                    print(f"  ⚠️  Local main is {ahead} commit(s) ahead of origin/main")
            else:
                print("  ⚠️  Could not compare with origin/main")
        else:
            print("  ❌ Local main branch does not exist")
            print("  💡 Run: python scripts/branch_manager.py --setup")
        
        # Uncommitted changes
        print("\n📝 Working Directory:")
        code, stdout, _ = self.run_git("status", "--short", check=False)
        if code == 0:
            if stdout:
                print("  ⚠️  You have uncommitted changes:")
                for line in stdout.split('\n')[:5]:  # Show first 5 changes
                    print(f"     {line}")
                if len(stdout.split('\n')) > 5:
                    print(f"     ... and {len(stdout.split('\n')) - 5} more")
            else:
                print("  ✅ Working directory clean")
        
        print("\n" + "=" * 60)
    
    def list_branches(self, remote: bool = False, all_branches: bool = False) -> None:
        """List branches."""
        if all_branches:
            print("📋 All Branches (local and remote):")
            code, stdout, _ = self.run_git("branch", "-a", "-vv", check=False)
        elif remote:
            print("☁️  Remote Branches:")
            code, stdout, _ = self.run_git("branch", "-r", "-vv", check=False)
        else:
            print("🏠 Local Branches:")
            code, stdout, _ = self.run_git("branch", "-vv", check=False)
        
        if code == 0:
            for line in stdout.split('\n'):
                print(f"  {line}")
    
    def cleanup_merged_branches(self, dry_run: bool = True) -> None:
        """Clean up branches that have been merged to main."""
        print("🧹 Cleaning up merged branches...")
        
        # Ensure main exists
        if not self.branch_exists("main"):
            print("❌ Local main branch does not exist. Run --setup first.")
            return
        
        # Get list of merged branches
        code, stdout, _ = self.run_git("branch", "--merged", "main", check=False)
        if code != 0:
            print("❌ Failed to get merged branches")
            return
        
        merged_branches = []
        for line in stdout.split('\n'):
            branch = line.strip().replace('* ', '')
            # Skip main and current branch
            if branch and branch != 'main' and not line.startswith('*'):
                merged_branches.append(branch)
        
        if not merged_branches:
            print("✅ No merged branches to clean up")
            return
        
        print(f"\n📋 Found {len(merged_branches)} merged branch(es):")
        for branch in merged_branches:
            print(f"  - {branch}")
        
        if dry_run:
            print("\n💡 This is a dry run. Use --cleanup --confirm to actually delete.")
            return
        
        # Delete branches
        for branch in merged_branches:
            code, _, stderr = self.run_git("branch", "-d", branch, check=False)
            if code == 0:
                print(f"  ✅ Deleted: {branch}")
            else:
                print(f"  ❌ Failed to delete {branch}: {stderr}")


def main():
    parser = argparse.ArgumentParser(
        description="Kinetra Branch Management Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Initial setup - create local main tracking remote
  python scripts/branch_manager.py --setup
  
  # Check branch status
  python scripts/branch_manager.py --status
  
  # Sync current branch with remote
  python scripts/branch_manager.py --sync
  
  # List all branches
  python scripts/branch_manager.py --list --all
  
  # Clean up merged branches (dry run)
  python scripts/branch_manager.py --cleanup
  
  # Actually delete merged branches
  python scripts/branch_manager.py --cleanup --confirm
        """
    )
    
    parser.add_argument("--setup", action="store_true",
                       help="Set up local main branch tracking origin/main")
    parser.add_argument("--sync", action="store_true",
                       help="Sync current branch with remote")
    parser.add_argument("--status", action="store_true",
                       help="Show comprehensive branch status")
    parser.add_argument("--list", action="store_true",
                       help="List branches")
    parser.add_argument("--cleanup", action="store_true",
                       help="Clean up merged branches")
    parser.add_argument("--all", action="store_true",
                       help="Show all branches (local and remote)")
    parser.add_argument("--remote", action="store_true",
                       help="Show only remote branches")
    parser.add_argument("--confirm", action="store_true",
                       help="Confirm cleanup (actually delete branches)")
    
    args = parser.parse_args()
    
    # If no arguments, show status
    if not any([args.setup, args.sync, args.status, args.list, args.cleanup]):
        args.status = True
    
    manager = BranchManager()
    
    try:
        if args.setup:
            success = manager.setup_local_main()
            sys.exit(0 if success else 1)
        
        if args.sync:
            success = manager.sync_with_remote()
            sys.exit(0 if success else 1)
        
        if args.status:
            manager.show_status()
        
        if args.list:
            manager.list_branches(remote=args.remote, all_branches=args.all)
        
        if args.cleanup:
            manager.cleanup_merged_branches(dry_run=not args.confirm)
    
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
