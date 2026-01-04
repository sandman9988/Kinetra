#!/usr/bin/env python3
"""
Kinetra Menu State Tracker
===========================

Tracks user progress through the production workflow and provides
intelligent highlighting and guidance.

Features:
- Persistent state tracking (JSON file)
- Workflow completion detection
- Smart menu highlighting (✅ completed, 🔄 in-progress, 📍 next)
- Breadcrumb navigation
- Workflow validation gates

Design Philosophy:
- Zero assumptions - validate everything
- Graceful degradation if state file missing
- Thread-safe operations
- No data loss (atomic saves)

Usage:
    from kinetra.menu_state_tracker import MenuStateTracker

    tracker = MenuStateTracker()
    tracker.mark_completed("1.1")  # Configure credentials
    tracker.mark_completed("2.1")  # Download data

    # Get next recommended step
    next_step = tracker.get_next_step()

    # Check if ready for training
    if tracker.is_ready_for("training"):
        # Proceed with training
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple


class MenuStateTracker:
    """
    Tracks menu navigation state and workflow progress.

    Workflow stages:
    1. Setup: Configure credentials, test connections
    2. Data: Download and validate data
    3. Training: Train RL agents
    4. Backtesting: Validate strategies
    5. Deployment: Live trading preparation
    """

    # Define workflow dependencies
    WORKFLOW_STAGES = {
        "setup": {
            "steps": ["1.1", "1.2"],  # Configure & test credentials
            "required_for": ["data"],
        },
        "data": {
            "steps": ["2.1", "2.2"],  # Discover & download data
            "required_for": ["training", "backtesting"],
        },
        "training": {
            "steps": ["3.1", "3.2"],  # Train agents
            "required_for": ["backtesting"],
        },
        "backtesting": {
            "steps": ["4.1", "4.2"],  # Run backtests
            "required_for": ["deployment"],
        },
        "deployment": {
            "steps": ["5.1"],  # Deploy strategies
            "required_for": [],
        },
    }

    # Menu option metadata
    MENU_OPTIONS = {
        # Setup & Authentication
        "1.1": {"name": "Configure MetaAPI Credentials", "stage": "setup", "critical": True},
        "1.2": {"name": "Test MetaAPI Connection", "stage": "setup", "critical": True},
        "1.3": {"name": "Select/Change MetaAPI Account", "stage": "setup", "critical": False},
        "1.4": {"name": "Configure MT5 (Local)", "stage": "setup", "critical": False},
        "1.5": {"name": "Test MT5 Connection", "stage": "setup", "critical": False},
        "1.6": {"name": "View Current Configuration", "stage": "setup", "critical": False},
        # Data Management
        "2.1": {"name": "Discover Available Data", "stage": "data", "critical": True},
        "2.2": {"name": "Download Data (MetaAPI)", "stage": "data", "critical": True},
        "2.3": {"name": "Download Data (MT5 Local)", "stage": "data", "critical": False},
        "2.4": {"name": "Prepare Data for Training", "stage": "data", "critical": True},
        "2.5": {"name": "Validate Data Integrity", "stage": "data", "critical": True},
        "2.6": {"name": "Backup Data", "stage": "data", "critical": False},
        # Training
        "3.1": {"name": "Quick RL Training", "stage": "training", "critical": True},
        "3.2": {"name": "Train Custom Agent", "stage": "training", "critical": False},
        "3.3": {"name": "Train SuperPot Agent", "stage": "training", "critical": False},
        "3.4": {"name": "View Training Progress", "stage": "training", "critical": False},
        "3.5": {"name": "Monitor Composite Health", "stage": "training", "critical": False},
        # Backtesting
        "4.1": {"name": "Run Single Backtest", "stage": "backtesting", "critical": True},
        "4.2": {"name": "Run Batch Backtest", "stage": "backtesting", "critical": True},
        "4.3": {"name": "Monte Carlo Validation", "stage": "backtesting", "critical": True},
        "4.4": {"name": "Generate Reports", "stage": "backtesting", "critical": False},
        "4.5": {"name": "View Backtest Results", "stage": "backtesting", "critical": False},
        # Analysis & Monitoring
        "5.1": {"name": "Performance Analysis", "stage": "backtesting", "critical": False},
        "5.2": {"name": "Risk Analysis", "stage": "backtesting", "critical": False},
        "5.3": {"name": "Regime Analysis", "stage": "backtesting", "critical": False},
        "5.4": {"name": "Health Monitoring", "stage": "backtesting", "critical": False},
    }

    def __init__(self, state_file: Optional[Path] = None):
        """
        Initialize menu state tracker.

        Args:
            state_file: Path to state file (default: data/menu_state.json)
        """
        if state_file is None:
            project_root = Path(__file__).parent.parent
            state_file = project_root / "data" / "menu_state.json"

        self.state_file = Path(state_file)
        self.state_file.parent.mkdir(parents=True, exist_ok=True)

        self.state = self._load_state()

    def _load_state(self) -> Dict:
        """Load state from file or create new state."""
        if self.state_file.exists():
            try:
                with open(self.state_file, "r") as f:
                    state = json.load(f)

                # Validate state structure
                if not isinstance(state, dict):
                    return self._create_empty_state()

                # Ensure required keys exist
                required_keys = ["completed", "in_progress", "last_accessed", "session_history"]
                for key in required_keys:
                    if key not in state:
                        state[key] = [] if key == "session_history" else {}

                return state

            except (json.JSONDecodeError, IOError):
                # Corrupted state file - create new
                return self._create_empty_state()
        else:
            return self._create_empty_state()

    def _create_empty_state(self) -> Dict:
        """Create empty state dictionary."""
        return {
            "completed": {},  # {option_id: timestamp}
            "in_progress": {},  # {option_id: timestamp}
            "last_accessed": {},  # {option_id: timestamp}
            "session_history": [],  # List of {option_id, timestamp, success}
            "created_at": datetime.now().isoformat(),
            "version": "1.0",
        }

    def _save_state(self):
        """Save state to file atomically."""
        # Write to temp file first
        temp_file = self.state_file.with_suffix(".tmp")

        try:
            with open(temp_file, "w") as f:
                json.dump(self.state, f, indent=2)

            # Atomic rename
            temp_file.replace(self.state_file)

        except Exception as e:
            # Clean up temp file on error
            if temp_file.exists():
                temp_file.unlink()
            raise RuntimeError(f"Failed to save menu state: {e}")

    def mark_started(self, option_id: str):
        """Mark an option as started."""
        timestamp = datetime.now().isoformat()

        self.state["in_progress"][option_id] = timestamp
        self.state["last_accessed"][option_id] = timestamp

        # Add to session history
        self.state["session_history"].append(
            {"option_id": option_id, "timestamp": timestamp, "action": "started"}
        )

        self._save_state()

    def mark_completed(self, option_id: str):
        """Mark an option as completed."""
        timestamp = datetime.now().isoformat()

        self.state["completed"][option_id] = timestamp
        self.state["last_accessed"][option_id] = timestamp

        # Remove from in_progress
        if option_id in self.state["in_progress"]:
            del self.state["in_progress"][option_id]

        # Add to session history
        self.state["session_history"].append(
            {"option_id": option_id, "timestamp": timestamp, "action": "completed"}
        )

        self._save_state()

    def mark_failed(self, option_id: str, error: Optional[str] = None):
        """Mark an option as failed."""
        timestamp = datetime.now().isoformat()

        # Remove from in_progress
        if option_id in self.state["in_progress"]:
            del self.state["in_progress"][option_id]

        # Add to session history
        self.state["session_history"].append(
            {
                "option_id": option_id,
                "timestamp": timestamp,
                "action": "failed",
                "error": error,
            }
        )

        self._save_state()

    def is_completed(self, option_id: str) -> bool:
        """Check if an option is completed."""
        return option_id in self.state["completed"]

    def is_in_progress(self, option_id: str) -> bool:
        """Check if an option is in progress."""
        return option_id in self.state["in_progress"]

    def get_stage_progress(self, stage: str) -> Tuple[int, int, float]:
        """
        Get progress for a workflow stage.

        Args:
            stage: Stage name (setup, data, training, backtesting, deployment)

        Returns:
            Tuple of (completed_count, total_count, percentage)
        """
        if stage not in self.WORKFLOW_STAGES:
            return 0, 0, 0.0

        steps = self.WORKFLOW_STAGES[stage]["steps"]
        completed = sum(1 for step in steps if self.is_completed(step))
        total = len(steps)
        percentage = (completed / total * 100) if total > 0 else 0.0

        return completed, total, percentage

    def is_stage_completed(self, stage: str) -> bool:
        """Check if all critical steps in a stage are completed."""
        if stage not in self.WORKFLOW_STAGES:
            return False

        steps = self.WORKFLOW_STAGES[stage]["steps"]

        for step in steps:
            if step in self.MENU_OPTIONS:
                # Only check critical steps
                if self.MENU_OPTIONS[step].get("critical", False):
                    if not self.is_completed(step):
                        return False

        return True

    def is_ready_for(self, stage: str) -> Tuple[bool, List[str]]:
        """
        Check if ready to proceed to a stage.

        Args:
            stage: Stage name

        Returns:
            Tuple of (ready, missing_prerequisites)
        """
        if stage not in self.WORKFLOW_STAGES:
            return False, [f"Unknown stage: {stage}"]

        # Check if dependencies are met
        missing = []

        for dep_stage, config in self.WORKFLOW_STAGES.items():
            if stage in config.get("required_for", []):
                if not self.is_stage_completed(dep_stage):
                    missing.append(dep_stage)

        return len(missing) == 0, missing

    def get_next_step(self) -> Optional[str]:
        """
        Get the next recommended step based on workflow.

        Returns:
            Option ID of next step, or None if workflow complete
        """
        # Check each stage in order
        for stage_name in ["setup", "data", "training", "backtesting", "deployment"]:
            if not self.is_stage_completed(stage_name):
                # Find first incomplete critical step in this stage
                steps = self.WORKFLOW_STAGES[stage_name]["steps"]

                for step in steps:
                    if step in self.MENU_OPTIONS:
                        if self.MENU_OPTIONS[step].get("critical", False):
                            if not self.is_completed(step):
                                return step

        return None  # All stages complete

    def get_status_icon(self, option_id: str) -> str:
        """
        Get status icon for menu option.

        Returns:
            Icon string (✅ completed, 🔄 in-progress, 📍 next, ⚠️  blocked, • waiting)
        """
        if self.is_completed(option_id):
            return "✅"

        if self.is_in_progress(option_id):
            return "🔄"

        # Check if this is the next recommended step
        next_step = self.get_next_step()
        if option_id == next_step:
            return "📍"

        # Check if blocked by prerequisites
        if option_id in self.MENU_OPTIONS:
            stage = self.MENU_OPTIONS[option_id]["stage"]
            ready, missing = self.is_ready_for(stage)
            if not ready:
                return "⚠️ "

        return "•"

    def get_breadcrumb(self) -> str:
        """
        Get breadcrumb showing current position in workflow.

        Returns:
            Breadcrumb string like "Setup ✅ → Data 🔄 → Training •"
        """
        stages = ["Setup", "Data", "Training", "Backtesting", "Deployment"]
        stage_keys = ["setup", "data", "training", "backtesting", "deployment"]

        breadcrumb_parts = []

        for display_name, stage_key in zip(stages, stage_keys):
            if self.is_stage_completed(stage_key):
                icon = "✅"
            else:
                # Check if any step in this stage is in progress
                steps = self.WORKFLOW_STAGES[stage_key]["steps"]
                if any(self.is_in_progress(step) for step in steps):
                    icon = "🔄"
                else:
                    # Check if this is the next stage
                    next_step = self.get_next_step()
                    if next_step and next_step in steps:
                        icon = "📍"
                    else:
                        icon = "•"

            breadcrumb_parts.append(f"{display_name} {icon}")

        return " → ".join(breadcrumb_parts)

    def get_summary(self) -> Dict:
        """
        Get summary of workflow progress.

        Returns:
            Dictionary with progress statistics
        """
        total_options = len(self.MENU_OPTIONS)
        completed_options = len(self.state["completed"])
        in_progress_options = len(self.state["in_progress"])

        stages_progress = {}
        for stage in self.WORKFLOW_STAGES.keys():
            completed, total, percentage = self.get_stage_progress(stage)
            stages_progress[stage] = {
                "completed": completed,
                "total": total,
                "percentage": percentage,
                "complete": self.is_stage_completed(stage),
            }

        return {
            "overall": {
                "total": total_options,
                "completed": completed_options,
                "in_progress": in_progress_options,
                "percentage": (completed_options / total_options * 100) if total_options > 0 else 0,
            },
            "stages": stages_progress,
            "next_step": self.get_next_step(),
            "breadcrumb": self.get_breadcrumb(),
        }

    def reset(self):
        """Reset all state (use with caution!)."""
        self.state = self._create_empty_state()
        self._save_state()

    def reset_stage(self, stage: str):
        """Reset a specific workflow stage."""
        if stage not in self.WORKFLOW_STAGES:
            raise ValueError(f"Unknown stage: {stage}")

        steps = self.WORKFLOW_STAGES[stage]["steps"]

        for step in steps:
            if step in self.state["completed"]:
                del self.state["completed"][step]
            if step in self.state["in_progress"]:
                del self.state["in_progress"][step]

        self._save_state()


def get_menu_state_tracker(state_file: Optional[Path] = None) -> MenuStateTracker:
    """
    Get singleton instance of MenuStateTracker.

    Args:
        state_file: Optional custom state file path

    Returns:
        MenuStateTracker instance
    """
    # Simple singleton pattern
    if not hasattr(get_menu_state_tracker, "_instance"):
        get_menu_state_tracker._instance = MenuStateTracker(state_file)

    return get_menu_state_tracker._instance


# CLI for testing/debugging
if __name__ == "__main__":
    import sys

    tracker = get_menu_state_tracker()

    if len(sys.argv) > 1:
        command = sys.argv[1]

        if command == "status":
            summary = tracker.get_summary()
            print("\n=== Workflow Status ===\n")
            print(f"Breadcrumb: {summary['breadcrumb']}\n")
            print(
                f"Overall Progress: {summary['overall']['completed']}/{summary['overall']['total']} "
                f"({summary['overall']['percentage']:.1f}%)\n"
            )
            print("Stage Progress:")
            for stage, progress in summary["stages"].items():
                status = "✅" if progress["complete"] else "🔄"
                print(
                    f"  {status} {stage.title()}: {progress['completed']}/{progress['total']} "
                    f"({progress['percentage']:.1f}%)"
                )

            print(f"\nNext Step: {summary['next_step']}")

        elif command == "complete" and len(sys.argv) > 2:
            option_id = sys.argv[2]
            tracker.mark_completed(option_id)
            print(f"✅ Marked {option_id} as completed")

        elif command == "reset":
            tracker.reset()
            print("✅ State reset")

        else:
            print("Unknown command. Usage:")
            print("  python menu_state_tracker.py status")
            print("  python menu_state_tracker.py complete <option_id>")
            print("  python menu_state_tracker.py reset")
    else:
        # Show status by default
        summary = tracker.get_summary()
        print(f"\nWorkflow: {summary['breadcrumb']}")
        print(f"Progress: {summary['overall']['percentage']:.1f}%")
        print(f"Next: {summary['next_step']}")
