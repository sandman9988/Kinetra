#!/usr/bin/env python3
"""
Enhanced Menu UX Module
=======================

Visual enhancements for Kinetra menu system:
- Progress bars with tqdm
- Countdown timers
- Visual feedback for user actions
- Highlighted menu selections
- Confirmation messages
- Abort/escape options
- Status indicators

Usage:
    from kinetra.menu_ux import (
        show_progress,
        countdown,
        confirm_with_visual,
        show_token_saved,
        MenuHighlighter
    )
"""

import getpass
import sys
import time
from pathlib import Path
from typing import Any, Callable, List, Optional

from tqdm import tqdm

# =============================================================================
# VISUAL CONSTANTS & THEMES
# =============================================================================


class Colors:
    """ANSI color codes for terminal output."""

    # Basic colors
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    UNDERLINE = "\033[4m"

    # Foreground colors
    BLACK = "\033[30m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"

    # Bright foreground colors
    BRIGHT_BLACK = "\033[90m"
    BRIGHT_RED = "\033[91m"
    BRIGHT_GREEN = "\033[92m"
    BRIGHT_YELLOW = "\033[93m"
    BRIGHT_BLUE = "\033[94m"
    BRIGHT_MAGENTA = "\033[95m"
    BRIGHT_CYAN = "\033[96m"
    BRIGHT_WHITE = "\033[97m"

    # Background colors
    BG_BLACK = "\033[40m"
    BG_RED = "\033[41m"
    BG_GREEN = "\033[42m"
    BG_YELLOW = "\033[43m"
    BG_BLUE = "\033[44m"
    BG_MAGENTA = "\033[45m"
    BG_CYAN = "\033[46m"
    BG_WHITE = "\033[47m"


class Icons:
    """Unicode icons for visual feedback."""

    # Status
    SUCCESS = "✅"
    ERROR = "❌"
    WARNING = "⚠️ "
    INFO = "ℹ️ "

    # Actions
    DOWNLOAD = "📥"
    UPLOAD = "📤"
    SAVE = "💾"
    FOLDER = "📁"
    FILE = "📄"

    # Progress
    ROCKET = "🚀"
    CHECK = "✓"
    CROSS = "✗"
    ARROW = "→"
    HOURGLASS = "⏳"

    # Menu
    MENU = "📋"
    SETTINGS = "⚙️ "
    BACK = "◄"
    NEXT = "►"

    # Data
    CHART = "📊"
    DATABASE = "🗄️ "
    LOCK = "🔒"
    KEY = "🔑"


# =============================================================================
# PROGRESS BARS
# =============================================================================


def show_progress(
    iterable,
    description: str = "Processing",
    unit: str = "items",
    total: Optional[int] = None,
    leave: bool = True,
    colour: str = "green",
) -> tqdm:
    """
    Show progress bar with tqdm.

    Args:
        iterable: Items to iterate over
        description: Progress bar description
        unit: Unit name (e.g., "items", "files", "bars")
        total: Total items (auto-detected if None)
        leave: Keep progress bar after completion
        colour: Progress bar color

    Returns:
        tqdm iterator

    Example:
        for item in show_progress(items, "Downloading", unit="files"):
            download(item)
    """
    return tqdm(
        iterable,
        desc=description,
        unit=unit,
        total=total,
        leave=leave,
        colour=colour,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
    )


def progress_bar(
    total: int, description: str = "Progress", unit: str = "items", colour: str = "green"
) -> tqdm:
    """
    Create manual progress bar (update with bar.update(n)).

    Args:
        total: Total number of items
        description: Progress bar description
        unit: Unit name
        colour: Bar color

    Returns:
        tqdm progress bar object

    Example:
        bar = progress_bar(100, "Downloading")
        for i in range(100):
            # Do work
            bar.update(1)
        bar.close()
    """
    return tqdm(
        total=total,
        desc=description,
        unit=unit,
        colour=colour,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
    )


# =============================================================================
# COUNTDOWN TIMERS
# =============================================================================


def countdown(seconds: int, message: str = "Starting in", can_skip: bool = True) -> bool:
    """
    Show countdown with option to skip.

    Args:
        seconds: Countdown duration
        message: Message to display
        can_skip: Allow user to press Enter to skip

    Returns:
        True if completed, False if skipped

    Example:
        countdown(5, "Test starting in", can_skip=True)
    """
    import select

    skip_msg = " (Press Enter to skip)" if can_skip else ""

    for i in range(seconds, 0, -1):
        print(f"\r{Icons.HOURGLASS} {message} {i} seconds{skip_msg}...", end="", flush=True)

        # Check for Enter key (Unix-like systems)
        if can_skip and sys.stdin.isatty():
            # Wait 1 second or until input
            ready, _, _ = select.select([sys.stdin], [], [], 1)
            if ready:
                sys.stdin.readline()  # Consume the input
                print(f"\r{Icons.CHECK} Skipped!                                    ")
                return False
        else:
            time.sleep(1)

    print(f"\r{Icons.ROCKET} Starting now!                                    ")
    return True


def simple_countdown(seconds: int, message: str = "Starting in") -> None:
    """
    Simple countdown without skip option (cross-platform).

    Args:
        seconds: Countdown duration
        message: Message to display
    """
    for i in range(seconds, 0, -1):
        print(f"\r{Icons.HOURGLASS} {message} {i} seconds...", end="", flush=True)
        time.sleep(1)
    print(f"\r{Icons.ROCKET} Starting now!                ")


# =============================================================================
# VISUAL FEEDBACK
# =============================================================================


def show_token_saved(token: str, account_id: str = None, save_location: Path = None):
    """
    Show visual confirmation that credentials were saved.

    Args:
        token: API token (will be masked)
        account_id: Account ID (will be masked)
        save_location: Where credentials were saved
    """
    print("\n" + "═" * 80)
    print(f"{Icons.SUCCESS} {Colors.BOLD}{Colors.GREEN}CREDENTIALS SAVED{Colors.RESET}")
    print("═" * 80)

    # Show masked token
    if token:
        masked_token = (
            token[:8] + "*" * (len(token) - 16) + token[-8:]
            if len(token) > 16
            else token[:4] + "***"
        )
        print(f"\n{Icons.KEY} Token:      {masked_token}")

    # Show masked account ID
    if account_id:
        masked_id = (
            account_id[:8] + "***" + account_id[-4:]
            if len(account_id) > 12
            else account_id[:4] + "***"
        )
        print(f"{Icons.LOCK} Account ID: {masked_id}")

    # Show save location
    if save_location:
        print(f"{Icons.SAVE} Saved to:   {save_location}")

    print("\n" + "─" * 80)
    print(f"{Colors.GREEN}Your credentials are now ready to use!{Colors.RESET}")
    print("─" * 80 + "\n")


def show_token_pasted():
    """Show confirmation that token was pasted."""
    print(f"\n{Icons.SUCCESS} {Colors.GREEN}Token pasted successfully!{Colors.RESET}")


def show_success(message: str, details: str = None):
    """
    Show success message with optional details.

    Args:
        message: Main success message
        details: Optional details
    """
    print(f"\n{Icons.SUCCESS} {Colors.BOLD}{Colors.GREEN}{message}{Colors.RESET}")
    if details:
        print(f"   {Colors.DIM}{details}{Colors.RESET}")


def show_error(message: str, details: str = None):
    """
    Show error message with optional details.

    Args:
        message: Main error message
        details: Optional error details
    """
    print(f"\n{Icons.ERROR} {Colors.BOLD}{Colors.RED}{message}{Colors.RESET}")
    if details:
        print(f"   {Colors.DIM}{details}{Colors.RESET}")


def show_warning(message: str, details: str = None):
    """
    Show warning message with optional details.

    Args:
        message: Main warning message
        details: Optional warning details
    """
    print(f"\n{Icons.WARNING}{Colors.BOLD}{Colors.YELLOW}{message}{Colors.RESET}")
    if details:
        print(f"   {Colors.DIM}{details}{Colors.RESET}")


def show_info(message: str, details: str = None):
    """
    Show info message with optional details.

    Args:
        message: Main info message
        details: Optional info details
    """
    print(f"\n{Icons.INFO}{Colors.BOLD}{Colors.CYAN}{message}{Colors.RESET}")
    if details:
        print(f"   {Colors.DIM}{details}{Colors.RESET}")


# =============================================================================
# MENU HIGHLIGHTING
# =============================================================================


class MenuHighlighter:
    """Enhanced menu with highlighted selections."""

    def __init__(self, title: str, options: List[str], allow_back: bool = True):
        """
        Initialize menu.

        Args:
            title: Menu title
            options: List of menu options
            allow_back: Show "Back" option (0)
        """
        self.title = title
        self.options = options
        self.allow_back = allow_back
        self.current_selection = None

    def display(self, highlight_choice: str = None):
        """
        Display menu with optional highlighted choice.

        Args:
            highlight_choice: Choice to highlight (e.g., "1", "2")
        """
        # Header
        print("\n" + "═" * 80)
        print(f"  {Colors.BOLD}{Colors.CYAN}{self.title}{Colors.RESET}")
        print("═" * 80)
        print()

        # Options
        for i, option in enumerate(self.options, 1):
            choice_str = str(i)

            # Highlight current selection
            if choice_str == highlight_choice:
                print(
                    f"  {Colors.BG_CYAN}{Colors.BLACK} [{choice_str}] {option} {Colors.RESET} {Icons.ARROW}"
                )
                self.current_selection = choice_str
            else:
                print(f"  [{choice_str}] {option}")

        # Back option
        if self.allow_back:
            if "0" == highlight_choice:
                print(
                    f"\n  {Colors.BG_CYAN}{Colors.BLACK} [0] Back to Previous Menu {Colors.RESET} {Icons.BACK}"
                )
            else:
                print(f"\n  [0] Back to Previous Menu")

        print()

    def get_choice(
        self,
        prompt: str = "Select option",
        valid_choices: List[str] = None,
        allow_quit: bool = True,
    ) -> Optional[str]:
        """
        Get user choice with validation and visual feedback.

        Args:
            prompt: Input prompt
            valid_choices: List of valid choices (auto-detected if None)
            allow_quit: Allow 'q' to quit

        Returns:
            User's choice or None if quit
        """
        if valid_choices is None:
            valid_choices = [str(i) for i in range(1, len(self.options) + 1)]
            if self.allow_back:
                valid_choices.append("0")

        while True:
            # Show prompt
            prompt_text = f"{prompt}"
            if allow_quit:
                prompt_text += f" (q=quit, 0=back)"
            prompt_text += ": "

            try:
                choice = input(prompt_text).strip().lower()

                # Handle quit
                if allow_quit and choice == "q":
                    print(f"\n{Icons.INFO} Exiting...")
                    return None

                # Validate choice
                if choice in valid_choices:
                    # Re-display menu with highlight
                    self.display(highlight_choice=choice)
                    return choice
                else:
                    show_error("Invalid choice", f"Please select from: {', '.join(valid_choices)}")

            except (KeyboardInterrupt, EOFError):
                print(f"\n\n{Icons.INFO} Interrupted by user")
                return None


# =============================================================================
# CONFIRMATION DIALOGS
# =============================================================================


def confirm_with_visual(
    message: str, default: bool = False, show_consequences: bool = False, consequences: str = None
) -> bool:
    """
    Ask for confirmation with visual feedback.

    Args:
        message: Confirmation message
        default: Default choice (True=Yes, False=No)
        show_consequences: Show warning about consequences
        consequences: What will happen if confirmed

    Returns:
        True if confirmed, False otherwise

    Example:
        if confirm_with_visual("Delete all data?", default=False,
                               show_consequences=True,
                               consequences="All files will be permanently deleted"):
            delete_data()
    """
    # Show warning if needed
    if show_consequences and consequences:
        print(f"\n{Icons.WARNING}{Colors.YELLOW}WARNING:{Colors.RESET}")
        print(f"   {consequences}")

    # Build prompt
    if default:
        choices = f"[{Colors.BOLD}Y{Colors.RESET}/n]"
    else:
        choices = f"[y/{Colors.BOLD}N{Colors.RESET}]"

    prompt = f"\n{message} {choices}: "

    try:
        response = input(prompt).strip().lower()

        # Handle response
        if response == "":
            confirmed = default
        elif response in ["y", "yes"]:
            confirmed = True
        elif response in ["n", "no"]:
            confirmed = False
        else:
            print(f"{Icons.ERROR} Invalid response, defaulting to {'Yes' if default else 'No'}")
            confirmed = default

        # Visual feedback
        if confirmed:
            print(f"{Icons.CHECK} {Colors.GREEN}Confirmed{Colors.RESET}")
        else:
            print(f"{Icons.CROSS} {Colors.RED}Cancelled{Colors.RESET}")

        return confirmed

    except (KeyboardInterrupt, EOFError):
        print(f"\n{Icons.CROSS} {Colors.RED}Cancelled{Colors.RESET}")
        return False


def confirm_action(message: str, default: str = "n") -> bool:
    """
    Simple yes/no confirmation.

    Args:
        message: Confirmation message
        default: Default choice ('y' or 'n')

    Returns:
        True if confirmed, False otherwise
    """
    if default.lower() == "y":
        prompt = f"{message} [Y/n]: "
        default_bool = True
    else:
        prompt = f"{message} [y/N]: "
        default_bool = False

    try:
        response = input(prompt).strip().lower()

        if response == "":
            return default_bool
        elif response in ["y", "yes"]:
            return True
        else:
            return False
    except (KeyboardInterrupt, EOFError):
        return False


# =============================================================================
# ABORT/ESCAPE OPTIONS
# =============================================================================


def show_abort_options():
    """Show available abort options."""
    print(f"\n{Colors.DIM}Press Ctrl+C to abort at any time{Colors.RESET}")


def prompt_with_abort(
    prompt: str, allow_back: bool = True, allow_skip: bool = False
) -> Optional[str]:
    """
    Get input with abort/back/skip options.

    Args:
        prompt: Input prompt
        allow_back: Allow '0' to go back
        allow_skip: Allow 's' to skip

    Returns:
        User input or special values: None (abort), '0' (back), 'skip' (skip)
    """
    # Build prompt suffix
    options = []
    if allow_back:
        options.append("0=back")
    if allow_skip:
        options.append("s=skip")
    options.append("Ctrl+C=abort")

    full_prompt = f"{prompt} ({', '.join(options)}): "

    try:
        value = input(full_prompt).strip()

        # Handle special values
        if allow_back and value == "0":
            print(f"{Icons.BACK} Going back...")
            return "0"

        if allow_skip and value.lower() == "s":
            print(f"{Icons.NEXT} Skipping...")
            return "skip"

        return value

    except (KeyboardInterrupt, EOFError):
        print(f"\n{Icons.CROSS} Aborted by user")
        return None


# =============================================================================
# SPINNER/LOADING ANIMATIONS
# =============================================================================


def show_spinner(message: str, duration: float = 2.0):
    """
    Show a simple spinner animation.

    Args:
        message: Message to display
        duration: How long to spin (seconds)
    """
    frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
    end_time = time.time() + duration
    i = 0

    while time.time() < end_time:
        print(f"\r{frames[i % len(frames)]} {message}...", end="", flush=True)
        time.sleep(0.1)
        i += 1

    print(f"\r{Icons.CHECK} {message}... Done!    ")


# =============================================================================
# SECURE INPUT
# =============================================================================


def get_secure_input_with_feedback(
    prompt: str, confirm: bool = False, show_pasted: bool = True
) -> Optional[str]:
    """
    Get secure input (password/token) with visual feedback.

    Args:
        prompt: Input prompt
        confirm: Require confirmation
        show_pasted: Show confirmation when pasted

    Returns:
        User input or None if cancelled
    """
    try:
        # First input
        value = getpass.getpass(f"\n{Icons.KEY} {prompt} (hidden): ")

        if not value:
            show_error("No input provided")
            return None

        # Show pasted confirmation
        if show_pasted and len(value) > 10:
            show_token_pasted()

        # Confirm if requested
        if confirm:
            confirm_value = getpass.getpass(f"{Icons.KEY} Confirm {prompt} (hidden): ")

            if value != confirm_value:
                show_error("Values do not match")
                return None

            show_success("Confirmed!")

        return value

    except (KeyboardInterrupt, EOFError):
        print(f"\n{Icons.CROSS} Cancelled")
        return None


# =============================================================================
# STATUS INDICATORS
# =============================================================================


class StatusIndicator:
    """Real-time status indicator."""

    def __init__(self, total_steps: int, description: str = "Progress"):
        """
        Initialize status indicator.

        Args:
            total_steps: Total number of steps
            description: Process description
        """
        self.total_steps = total_steps
        self.current_step = 0
        self.description = description
        self.status_messages = []

    def update(self, step: int = None, message: str = None):
        """
        Update status.

        Args:
            step: Current step number (or increment by 1 if None)
            message: Status message
        """
        if step is not None:
            self.current_step = step
        else:
            self.current_step += 1

        # Build status line
        progress = int((self.current_step / self.total_steps) * 100)
        bar_length = 30
        filled = int((progress / 100) * bar_length)
        bar = "█" * filled + "░" * (bar_length - filled)

        status = f"\r{Icons.ROCKET} {self.description}: [{bar}] {progress}%"

        if message:
            status += f" - {message}"
            self.status_messages.append(message)

        print(status, end="", flush=True)

        # New line if complete
        if self.current_step >= self.total_steps:
            print()

    def complete(self, message: str = "Complete!"):
        """Mark as complete."""
        self.current_step = self.total_steps
        self.update(message=message)
        print(f"\n{Icons.SUCCESS} {Colors.GREEN}{message}{Colors.RESET}\n")


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    print("Kinetra Menu UX Examples\n")

    # Example 1: Progress bar
    print("1. Progress Bar:")
    items = range(100)
    for item in show_progress(items, "Processing", unit="items"):
        time.sleep(0.01)

    # Example 2: Countdown
    print("\n2. Countdown:")
    countdown(3, "Starting test in", can_skip=True)

    # Example 3: Visual feedback
    print("\n3. Visual Feedback:")
    show_token_saved(
        token="eyJhbGciOiJSUzUxMiIsInR5cCI6IkpXVCJ9.long_token_here",
        account_id="a1b2c3d4-e5f6-7890-abcd-ef1234567890",
        save_location=Path(".env"),
    )

    # Example 4: Menu with highlighting
    print("\n4. Menu with Highlighting:")
    menu = MenuHighlighter("Main Menu", ["Download Data", "Run Tests", "View Results"])
    menu.display()

    # Example 5: Confirmation
    print("\n5. Confirmation:")
    confirmed = confirm_with_visual("Proceed with download?", default=True, show_consequences=False)
    print(f"User confirmed: {confirmed}")

    # Example 6: Status indicator
    print("\n6. Status Indicator:")
    status = StatusIndicator(5, "Running Tests")
    for i in range(1, 6):
        status.update(message=f"Test {i}/5")
        time.sleep(0.5)
    status.complete("All tests passed!")

    print("\n✅ Examples complete!")
