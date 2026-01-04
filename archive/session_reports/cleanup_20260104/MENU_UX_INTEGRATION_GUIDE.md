# Menu UX Integration Guide

**Date**: 2026-01-03  
**Module**: `kinetra/menu_ux.py`  
**Status**: ✅ Ready for Integration

---

## Overview

The new `menu_ux.py` module provides enhanced visual feedback for the Kinetra menu system:

- ✅ **Progress bars** with tqdm
- ✅ **Countdown timers** with skip option
- ✅ **Visual feedback** for credentials, downloads, errors
- ✅ **Highlighted menu selections** with color
- ✅ **Abort/escape options** (Ctrl+C, back, skip)
- ✅ **Status indicators** with real-time progress
- ✅ **Secure input** with visual confirmation

---

## Quick Start

### Import the Module

```python
from kinetra.menu_ux import (
    # Progress
    show_progress,
    progress_bar,
    countdown,
    
    # Visual feedback
    show_token_saved,
    show_success,
    show_error,
    show_warning,
    show_info,
    
    # Menu
    MenuHighlighter,
    
    # Confirmation
    confirm_with_visual,
    confirm_action,
    
    # Input
    get_secure_input_with_feedback,
    prompt_with_abort,
    
    # Status
    StatusIndicator,
    
    # Icons & Colors
    Icons,
    Colors,
)
```

---

## Integration Examples

### 1. Enhanced Token Input & Confirmation

**BEFORE** (in `kinetra_menu.py`):
```python
def select_metaapi_account(wf_manager: WorkflowManager) -> bool:
    print("\n📋 Launching account selection...")
    
    result = subprocess.run(
        [sys.executable, "scripts/download/select_metaapi_account.py"]
    )
    
    if result.returncode == 0:
        print("\n✅ Account selected successfully")
        return True
```

**AFTER** (with UX enhancements):
```python
from kinetra.menu_ux import (
    show_info, show_success, show_token_saved,
    countdown, confirm_with_visual
)

def select_metaapi_account(wf_manager: WorkflowManager) -> bool:
    show_info("Launching account selection...", "This will open the MetaAPI selector")
    
    # Optional countdown before heavy operation
    countdown(3, "Starting in", can_skip=True)
    
    result = subprocess.run(
        [sys.executable, "scripts/download/select_metaapi_account.py"]
    )
    
    if result.returncode == 0:
        # Visual confirmation
        show_success("Account selected successfully!", 
                    "Your credentials are now configured")
        
        # Ask for next step with visual confirmation
        if confirm_with_visual("Download data now?", default=True):
            download_data(wf_manager)
        
        return True
```

---

### 2. Progress Bars for Downloads

**BEFORE**:
```python
def download_data(symbols):
    for symbol in symbols:
        print(f"Downloading {symbol}...")
        download(symbol)
```

**AFTER**:
```python
from kinetra.menu_ux import show_progress

def download_data(symbols):
    for symbol in show_progress(symbols, "Downloading", unit="symbols"):
        download(symbol)
    
    show_success(f"Downloaded {len(symbols)} symbols!")
```

**With manual progress bar**:
```python
from kinetra.menu_ux import progress_bar, show_success

def download_data(symbols):
    bar = progress_bar(len(symbols), "Downloading", unit="symbols")
    
    for symbol in symbols:
        # Download work here
        result = download(symbol)
        bar.update(1)
    
    bar.close()
    show_success(f"Downloaded {len(symbols)} symbols!")
```

---

### 3. Highlighted Menu Selections

**BEFORE**:
```python
def show_data_management_menu():
    print("\n=== DATA MANAGEMENT ===\n")
    print("  1. Download Data")
    print("  2. Consolidate Data")
    print("  3. Run Tests")
    print("  0. Back")
    
    choice = input("Select option: ")
```

**AFTER**:
```python
from kinetra.menu_ux import MenuHighlighter

def show_data_management_menu():
    menu = MenuHighlighter(
        title="DATA MANAGEMENT",
        options=[
            "Download Data",
            "Consolidate Data", 
            "Run Tests"
        ],
        allow_back=True
    )
    
    menu.display()
    choice = menu.get_choice(
        prompt="Select option",
        allow_quit=True
    )
    
    if choice is None:
        return  # User quit
    elif choice == "0":
        return  # User went back
    elif choice == "1":
        download_data()
    # ... etc
```

---

### 4. Secure Token Input with Visual Feedback

**BEFORE**:
```python
import getpass

token = getpass.getpass("Enter MetaAPI token: ")
print("Token received")
```

**AFTER**:
```python
from kinetra.menu_ux import get_secure_input_with_feedback, show_token_saved

token = get_secure_input_with_feedback(
    prompt="Enter MetaAPI token",
    confirm=False,
    show_pasted=True  # Shows "Token pasted!" when user pastes
)

if token:
    # Save and show visual confirmation
    save_to_env(token)
    show_token_saved(
        token=token,
        save_location=Path(".env")
    )
```

**Output**:
```
🔑 Enter MetaAPI token (hidden): **********************

✅ Token pasted successfully!

════════════════════════════════════════════════════════════════════════════════
✅ CREDENTIALS SAVED
════════════════════════════════════════════════════════════════════════════════

🔑 Token:      eyJhbGci***XXXXX***jrokA
💾 Saved to:   .env

────────────────────────────────────────────────────────────────────────────────
Your credentials are now ready to use!
────────────────────────────────────────────────────────────────────────────────
```

---

### 5. Countdown with Skip Option

**BEFORE**:
```python
print("Starting download in 5 seconds...")
time.sleep(5)
start_download()
```

**AFTER**:
```python
from kinetra.menu_ux import countdown, show_info

show_info("Preparing to download data...")

# User can press Enter to skip countdown
if countdown(5, "Download starting in", can_skip=True):
    show_info("Countdown complete, starting download...")
else:
    show_info("Skipped countdown, starting immediately...")

start_download()
```

**Output**:
```
⏳ Download starting in 3 seconds (Press Enter to skip)...
# User presses Enter
✓ Skipped!
```

---

### 6. Status Indicator for Multi-Step Operations

**BEFORE**:
```python
print("Step 1: Downloading...")
download()
print("Step 2: Validating...")
validate()
print("Step 3: Consolidating...")
consolidate()
print("Complete!")
```

**AFTER**:
```python
from kinetra.menu_ux import StatusIndicator

status = StatusIndicator(total_steps=3, description="Data Pipeline")

status.update(step=1, message="Downloading data...")
download()

status.update(step=2, message="Validating data...")
validate()

status.update(step=3, message="Consolidating data...")
consolidate()

status.complete("Data pipeline complete!")
```

**Output**:
```
🚀 Data Pipeline: [██████████░░░░░░░░░░░░░░░░░░] 33% - Downloading data...
🚀 Data Pipeline: [████████████████████░░░░░░░░] 66% - Validating data...
🚀 Data Pipeline: [████████████████████████████] 100% - Consolidating data...

✅ Data pipeline complete!
```

---

### 7. Confirmation with Consequences Warning

**BEFORE**:
```python
confirm = input("Delete all data? [y/N]: ")
if confirm.lower() == 'y':
    delete_data()
```

**AFTER**:
```python
from kinetra.menu_ux import confirm_with_visual

if confirm_with_visual(
    "Delete all data?",
    default=False,
    show_consequences=True,
    consequences="All downloaded files will be permanently deleted. This cannot be undone."
):
    delete_data()
else:
    show_info("Delete cancelled")
```

**Output**:
```
⚠️ WARNING:
   All downloaded files will be permanently deleted. This cannot be undone.

Delete all data? [y/N]: y
✓ Confirmed
```

---

### 8. Input with Abort Options

**BEFORE**:
```python
symbol = input("Enter symbol: ")
```

**AFTER**:
```python
from kinetra.menu_ux import prompt_with_abort

symbol = prompt_with_abort(
    prompt="Enter symbol",
    allow_back=True,
    allow_skip=True
)

if symbol is None:
    # User aborted (Ctrl+C)
    return
elif symbol == "0":
    # User went back
    return
elif symbol == "skip":
    # User skipped
    symbol = "EURUSD"  # Default
```

---

## Complete Integration Example: Enhanced Account Selection

Here's a complete before/after showing all improvements together:

### BEFORE
```python
def select_metaapi_account(wf_manager: WorkflowManager) -> bool:
    print("\n📋 Launching account selection...")
    
    result = subprocess.run(
        [sys.executable, "scripts/download/select_metaapi_account.py"]
    )
    
    if result.returncode == 0:
        print("\n✅ Account selected successfully")
        choice = input("\nDownload data now? [1=Yes, 2=No]: ")
        if choice == "1":
            result = subprocess.run(
                [sys.executable, "scripts/download/download_interactive.py"]
            )
        return True
```

### AFTER
```python
from kinetra.menu_ux import (
    show_info, show_success, show_error,
    countdown, confirm_with_visual,
    MenuHighlighter, StatusIndicator
)

def select_metaapi_account(wf_manager: WorkflowManager) -> bool:
    # Step 1: Show what's about to happen
    show_info(
        "Launching account selection...",
        "You'll be prompted for your MetaAPI credentials"
    )
    
    # Step 2: Optional countdown (user can skip)
    countdown(3, "Starting in", can_skip=True)
    
    # Step 3: Status indicator
    status = StatusIndicator(total_steps=4, description="Setup Workflow")
    
    # Account selection
    status.update(step=1, message="Selecting account...")
    result = subprocess.run(
        [sys.executable, "scripts/download/select_metaapi_account.py"]
    )
    
    if result.returncode != 0:
        show_error("Account selection failed", "Please check your credentials")
        return False
    
    show_success("Account selected successfully!")
    
    # Step 4: Ask about download with visual confirmation
    status.update(step=2, message="Ready for data download...")
    
    if confirm_with_visual(
        "Download data now?",
        default=True,
        show_consequences=False
    ):
        # Download with progress
        status.update(step=3, message="Downloading data...")
        result = subprocess.run(
            [sys.executable, "scripts/download/download_interactive.py"]
        )
        
        if result.returncode == 0:
            show_success("Download complete!")
        else:
            show_warning("Download completed with warnings")
    else:
        show_info("You can download data later from Data Management menu")
    
    status.complete("Setup workflow complete!")
    return True
```

---

## Color & Icon Reference

### Available Colors
```python
from kinetra.menu_ux import Colors

# Basic
Colors.RESET
Colors.BOLD
Colors.DIM
Colors.UNDERLINE

# Foreground colors
Colors.RED, Colors.GREEN, Colors.YELLOW, Colors.BLUE
Colors.CYAN, Colors.MAGENTA, Colors.WHITE, Colors.BLACK

# Bright variants
Colors.BRIGHT_RED, Colors.BRIGHT_GREEN, etc.

# Background colors
Colors.BG_RED, Colors.BG_GREEN, etc.
```

### Available Icons
```python
from kinetra.menu_ux import Icons

# Status
Icons.SUCCESS  # ✅
Icons.ERROR    # ❌
Icons.WARNING  # ⚠️
Icons.INFO     # ℹ️

# Actions
Icons.DOWNLOAD # 📥
Icons.SAVE     # 💾
Icons.FOLDER   # 📁
Icons.FILE     # 📄

# Progress
Icons.ROCKET   # 🚀
Icons.CHECK    # ✓
Icons.CROSS    # ✗
Icons.HOURGLASS # ⏳

# Menu
Icons.MENU     # 📋
Icons.BACK     # ◄
Icons.NEXT     # ►

# Data
Icons.CHART    # 📊
Icons.KEY      # 🔑
Icons.LOCK     # 🔒
```

---

## Testing the Module

Run the module directly to see all examples:

```bash
python -m kinetra.menu_ux
```

**Output**:
```
Kinetra Menu UX Examples

1. Progress Bar:
Processing: 100%|████████████████████| 100/100 [00:01<00:00, 99.50items/s]

2. Countdown:
⏳ Starting test in 3 seconds (Press Enter to skip)...
🚀 Starting now!

3. Visual Feedback:
════════════════════════════════════════════════════════════════════════════════
✅ CREDENTIALS SAVED
════════════════════════════════════════════════════════════════════════════════

🔑 Token:      eyJhbGci***long_token***here
🔒 Account ID: a1b2c3d4***7890
💾 Saved to:   .env

────────────────────────────────────────────────────────────────────────────────
Your credentials are now ready to use!
────────────────────────────────────────────────────────────────────────────────

4. Menu with Highlighting:
════════════════════════════════════════════════════════════════════════════════
  Main Menu
════════════════════════════════════════════════════════════════════════════════

  [1] Download Data
  [2] Run Tests
  [3] View Results

  [0] Back to Previous Menu

5. Confirmation:
Proceed with download? [Y/n]: y
✓ Confirmed

6. Status Indicator:
🚀 Running Tests: [████████████████████████████] 100% - Test 5/5

✅ All tests passed!

✅ Examples complete!
```

---

## Implementation Checklist

### Phase 1: Core Menu Enhancement (Priority 1)
- [ ] Add progress bars to `download_interactive.py`
- [ ] Add visual token confirmation to account selection
- [ ] Add highlighted menu to main menu (`show_main_menu`)
- [ ] Add countdown before long operations

### Phase 2: Workflow Prompts (Priority 2)
- [ ] Add status indicator to full workflow (login → download → test)
- [ ] Add abort options to all input prompts
- [ ] Add visual confirmations after each major step
- [ ] Add warning confirmations for destructive operations

### Phase 3: Data Operations (Priority 3)
- [ ] Add progress bar to `consolidate_data.py`
- [ ] Add status indicator to `run_exhaustive_tests.py`
- [ ] Add visual feedback to data integrity checks
- [ ] Add countdown/skip for automated operations

---

## Best Practices

### DO:
✅ Use `show_success()`, `show_error()`, `show_warning()` for all feedback  
✅ Add progress bars to loops > 10 iterations  
✅ Use `confirm_with_visual()` for destructive operations  
✅ Use `StatusIndicator` for multi-step workflows  
✅ Add countdown with skip option before heavy operations  
✅ Use `MenuHighlighter` for all menu displays  

### DON'T:
❌ Use bare `print()` for status messages  
❌ Use bare `input()` for confirmations  
❌ Show progress bars for < 5 items  
❌ Skip visual confirmation for credential saves  
❌ Use blocking operations without abort options  

---

## Migration Path

### Step 1: Import Module
Add to top of `kinetra_menu.py`:
```python
from kinetra.menu_ux import *
```

### Step 2: Replace Basic Prints
Replace:
```python
print("✅ Success")
print("❌ Error")
print("⚠️ Warning")
```

With:
```python
show_success("Success")
show_error("Error")
show_warning("Warning")
```

### Step 3: Add Progress Bars
Replace:
```python
for item in items:
    process(item)
```

With:
```python
for item in show_progress(items, "Processing"):
    process(item)
```

### Step 4: Enhance Menus
Replace menu displays with `MenuHighlighter`

### Step 5: Add Visual Confirmations
Replace credential saves with `show_token_saved()`

---

## Troubleshooting

### Progress bars not showing?
- Check if `tqdm` is installed: `pip install tqdm`
- Ensure stdout is not redirected
- Try `leave=True` parameter

### Colors not displaying?
- Some terminals don't support ANSI colors
- Colors auto-disabled in non-TTY environments
- Use `Colors.RESET` to ensure cleanup

### Countdown skip not working?
- Unix/Linux only feature (uses `select.select()`)
- Fallback: Use `simple_countdown()` for cross-platform
- Windows: Use simple countdown without skip

---

## Summary

The `menu_ux.py` module provides a complete toolkit for enhancing the Kinetra menu system with minimal code changes. Key benefits:

- **Better UX**: Visual feedback, progress indicators, countdowns
- **Safer**: Abort options, confirmations, visual warnings
- **Clearer**: Highlighted selections, status indicators
- **Professional**: Consistent styling, icons, colors

**Start with**: Add visual feedback to credential saving (highest impact, easiest integration)

**Next**: Add progress bars to downloads (immediate user value)

**Then**: Migrate all menus to `MenuHighlighter` (consistent experience)

---

**Status**: 🟢 Ready for Integration  
**Module**: `kinetra/menu_ux.py` (779 lines)  
**Dependencies**: `tqdm` (already in requirements.txt)  
**Backward Compatible**: ✅ Yes (can be integrated incrementally)