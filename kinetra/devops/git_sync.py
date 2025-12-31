"""
Git Auto-Sync Utilities
=======================

Automatically sync local and remote git repositories:
- Check sync status
- Auto-pull/push
- Conflict detection
- Branch management
"""

import subprocess
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional, Tuple


class SyncStatus(Enum):
    """Git sync status."""
    SYNCED = "synced"
    AHEAD = "ahead"  # Local has commits not in remote
    BEHIND = "behind"  # Remote has commits not in local
    DIVERGED = "diverged"  # Both have unique commits
    CONFLICT = "conflict"  # Merge conflict
    NO_REMOTE = "no_remote"
    ERROR = "error"


@dataclass
class GitStatus:
    """Current git repository status."""
    branch: str
    sync_status: SyncStatus
    ahead_count: int
    behind_count: int
    uncommitted_changes: int
    untracked_files: int
    stashed_changes: int
    remote: str
    last_commit: str
    last_sync: Optional[datetime] = None


class GitSync:
    """
    Git synchronization manager.

    Usage:
        sync = GitSync("/path/to/repo")
        status = sync.check_status()
        sync.auto_sync()
    """

    def __init__(self, repo_path: str = "."):
        self.repo_path = Path(repo_path).resolve()
        self._auto_sync_thread: Optional[threading.Thread] = None
        self._running = False

    def _run_git(self, *args, check: bool = False) -> Tuple[int, str, str]:
        """Run a git command and return (returncode, stdout, stderr)."""
        try:
            result = subprocess.run(
                ['git'] + list(args),
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                timeout=30
            )
            return result.returncode, result.stdout.strip(), result.stderr.strip()
        except subprocess.SubprocessError as e:
            return -1, "", str(e)

    def check_status(self) -> GitStatus:
        """Check current git status."""
        # Get current branch
        code, branch, _ = self._run_git('rev-parse', '--abbrev-ref', 'HEAD')
        if code != 0:
            return GitStatus(
                branch="unknown",
                sync_status=SyncStatus.ERROR,
                ahead_count=0, behind_count=0,
                uncommitted_changes=0, untracked_files=0,
                stashed_changes=0, remote="", last_commit=""
            )

        # Fetch latest (without merging)
        self._run_git('fetch', '--quiet')

        # Get remote tracking branch
        code, remote, _ = self._run_git('rev-parse', '--abbrev-ref', f'{branch}@{{upstream}}')
        if code != 0:
            sync_status = SyncStatus.NO_REMOTE
            remote = ""
            ahead_count = behind_count = 0
        else:
            # Count ahead/behind
            code, counts, _ = self._run_git('rev-list', '--left-right', '--count', f'{branch}...{remote}')
            if code == 0:
                parts = counts.split()
                ahead_count = int(parts[0]) if len(parts) > 0 else 0
                behind_count = int(parts[1]) if len(parts) > 1 else 0
            else:
                ahead_count = behind_count = 0

            # Determine sync status
            if ahead_count == 0 and behind_count == 0:
                sync_status = SyncStatus.SYNCED
            elif ahead_count > 0 and behind_count == 0:
                sync_status = SyncStatus.AHEAD
            elif ahead_count == 0 and behind_count > 0:
                sync_status = SyncStatus.BEHIND
            else:
                sync_status = SyncStatus.DIVERGED

        # Count uncommitted changes
        code, status_output, _ = self._run_git('status', '--porcelain')
        lines = status_output.split('\n') if status_output else []
        uncommitted = len([line for line in lines if line and not line.startswith('?')])
        untracked = len([line for line in lines if line.startswith('?')])

        # Count stashed changes
        code, stash_output, _ = self._run_git('stash', 'list')
        stashed = len(stash_output.split('\n')) if stash_output else 0

        # Get last commit
        code, last_commit, _ = self._run_git('log', '-1', '--format=%h %s')

        return GitStatus(
            branch=branch,
            sync_status=sync_status,
            ahead_count=ahead_count,
            behind_count=behind_count,
            uncommitted_changes=uncommitted,
            untracked_files=untracked,
            stashed_changes=stashed,
            remote=remote,
            last_commit=last_commit,
            last_sync=datetime.now()
        )

    def pull(self, rebase: bool = True) -> Tuple[bool, str]:
        """Pull changes from remote."""
        args = ['pull']
        if rebase:
            args.append('--rebase')

        code, stdout, stderr = self._run_git(*args)

        if code == 0:
            return True, stdout or "Already up to date"
        else:
            return False, stderr or "Pull failed"

    def push(self, force: bool = False) -> Tuple[bool, str]:
        """Push changes to remote."""
        args = ['push']
        if force:
            args.append('--force-with-lease')

        code, stdout, stderr = self._run_git(*args)

        if code == 0:
            return True, stdout or "Push successful"
        else:
            return False, stderr or "Push failed"

    def stash(self) -> Tuple[bool, str]:
        """Stash current changes."""
        code, stdout, stderr = self._run_git('stash', 'push', '-m', f'auto-stash-{datetime.now().isoformat()}')
        return code == 0, stdout or stderr

    def stash_pop(self) -> Tuple[bool, str]:
        """Pop the last stash."""
        code, stdout, stderr = self._run_git('stash', 'pop')
        return code == 0, stdout or stderr

    def auto_sync(self, push_changes: bool = True) -> Tuple[bool, str]:
        """
        Automatically sync with remote.

        Steps:
        1. Fetch latest
        2. Stash local changes if any
        3. Pull with rebase
        4. Pop stash
        5. Push if ahead

        Args:
            push_changes: Whether to push local changes to remote

        Returns:
            (success, message)
        """
        status = self.check_status()
        messages = []

        # Handle uncommitted changes
        had_changes = status.uncommitted_changes > 0
        if had_changes:
            success, msg = self.stash()
            if not success:
                return False, f"Failed to stash changes: {msg}"
            messages.append("Stashed local changes")

        try:
            # Pull if behind
            if status.sync_status in (SyncStatus.BEHIND, SyncStatus.DIVERGED):
                success, msg = self.pull(rebase=True)
                if not success:
                    if had_changes:
                        self.stash_pop()
                    return False, f"Failed to pull: {msg}"
                messages.append(f"Pulled changes: {msg}")

            # Push if ahead
            if push_changes and status.sync_status in (SyncStatus.AHEAD, SyncStatus.DIVERGED):
                success, msg = self.push()
                if not success:
                    messages.append(f"Warning: Push failed: {msg}")
                else:
                    messages.append("Pushed local changes")

        finally:
            # Restore stashed changes
            if had_changes:
                success, msg = self.stash_pop()
                if success:
                    messages.append("Restored local changes")
                else:
                    messages.append(f"Warning: Failed to restore stash: {msg}")

        return True, "; ".join(messages) if messages else "Already synced"

    def start_auto_sync(self, interval_seconds: int = 300):
        """Start background auto-sync."""
        if self._running:
            return

        self._running = True
        self._auto_sync_thread = threading.Thread(
            target=self._auto_sync_loop,
            args=(interval_seconds,),
            daemon=True
        )
        self._auto_sync_thread.start()

    def stop_auto_sync(self):
        """Stop background auto-sync."""
        self._running = False
        if self._auto_sync_thread:
            self._auto_sync_thread.join(timeout=5)

    def _auto_sync_loop(self, interval: int):
        """Background auto-sync loop."""
        while self._running:
            try:
                success, msg = self.auto_sync(push_changes=False)  # Only pull by default
                if not success:
                    print(f"[GitSync] Warning: {msg}")
            except Exception as e:
                print(f"[GitSync] Error: {e}")

            time.sleep(interval)


def auto_sync(repo_path: str = ".", push: bool = False) -> Tuple[bool, str]:
    """Convenience function to auto-sync a repository."""
    sync = GitSync(repo_path)
    return sync.auto_sync(push_changes=push)


def check_sync_status(repo_path: str = ".") -> str:
    """Get formatted sync status."""
    sync = GitSync(repo_path)
    status = sync.check_status()

    lines = ["=" * 60, "GIT SYNC STATUS", "=" * 60]
    lines.append(f"Repository:     {sync.repo_path}")
    lines.append(f"Branch:         {status.branch}")
    lines.append(f"Remote:         {status.remote or 'None'}")
    lines.append(f"Status:         {status.sync_status.value}")

    if status.ahead_count > 0:
        lines.append(f"Commits ahead:  {status.ahead_count}")
    if status.behind_count > 0:
        lines.append(f"Commits behind: {status.behind_count}")

    lines.append(f"Uncommitted:    {status.uncommitted_changes} files")
    lines.append(f"Untracked:      {status.untracked_files} files")
    lines.append(f"Stashed:        {status.stashed_changes} entries")
    lines.append(f"Last commit:    {status.last_commit}")
    lines.append("=" * 60)

    # Status indicator
    if status.sync_status == SyncStatus.SYNCED:
        lines.append("✅ Repository is synchronized")
    elif status.sync_status == SyncStatus.AHEAD:
        lines.append("📤 Local changes ready to push")
    elif status.sync_status == SyncStatus.BEHIND:
        lines.append("📥 Updates available from remote")
    elif status.sync_status == SyncStatus.DIVERGED:
        lines.append("⚠️ Local and remote have diverged - rebase needed")
    elif status.sync_status == SyncStatus.NO_REMOTE:
        lines.append("⚠️ No remote tracking branch configured")
    else:
        lines.append("❌ Error checking status")

    return "\n".join(lines)
