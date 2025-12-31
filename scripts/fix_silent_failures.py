#!/usr/bin/env python3
"""
Automated Silent Failure Fixer for Kinetra

This script analyzes detected silent failures and provides automated fixes.
It can suggest fixes, apply them, and validate the results.

Usage:
    # Analyze failures and suggest fixes
    python scripts/fix_silent_failures.py --analyze
    
    # Apply fixes automatically (with backup)
    python scripts/fix_silent_failures.py --fix --auto
    
    # Apply specific fix by ID
    python scripts/fix_silent_failures.py --fix --id FAILURE_ID
    
    # Dry run (show what would be fixed)
    python scripts/fix_silent_failures.py --fix --dry-run
"""

import argparse
import json
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from kinetra.silent_failure_logger import (
    SilentFailureLogger,
    FailureRecord,
    FailureCategory,
)


class FailureFixer:
    """Automated failure fixing system."""
    
    def __init__(self, dry_run: bool = False):
        """
        Initialize failure fixer.
        
        Args:
            dry_run: If True, only show what would be fixed without applying
        """
        self.dry_run = dry_run
        self.logger = SilentFailureLogger.get_instance()
        self.fixes_applied = []
        self.fixes_failed = []
        
        # Backup directory
        self.backup_dir = project_root / "data" / "backups" / "failure_fixes"
        self.backup_dir.mkdir(parents=True, exist_ok=True)
    
    def analyze_failures(self) -> Dict[str, List[FailureRecord]]:
        """
        Analyze failures and group by fix strategy.
        
        Returns:
            Dictionary of fix strategies to failure records
        """
        print("\n🔍 Analyzing failures...")
        
        strategies = {
            "add_len_method": [],
            "add_missing_import": [],
            "fix_type_error": [],
            "add_error_handling": [],
            "fix_attribute_error": [],
            "manual_review": [],
        }
        
        for failure in self.logger.failures:
            strategy = self._determine_fix_strategy(failure)
            strategies[strategy].append(failure)
        
        # Print analysis
        print(f"\n{'='*60}")
        print("Fix Strategy Analysis")
        print(f"{'='*60}")
        
        for strategy, failures in strategies.items():
            if failures:
                print(f"\n{strategy}: {len(failures)} failures")
                for failure in failures[:3]:  # Show first 3
                    print(f"  - {failure.file_path}:{failure.line_number}")
                    print(f"    {failure.exception_type}: {failure.exception_message[:60]}...")
                
                if len(failures) > 3:
                    print(f"  ... and {len(failures) - 3} more")
        
        return strategies
    
    def _determine_fix_strategy(self, failure: FailureRecord) -> str:
        """
        Determine the appropriate fix strategy for a failure.
        
        Args:
            failure: Failure record
        
        Returns:
            Strategy name
        """
        # InstrumentData len() issue
        if (failure.exception_type == "TypeError" and 
            "has no len()" in failure.exception_message):
            return "add_len_method"
        
        # Import errors
        if failure.category == FailureCategory.IMPORT_ERROR.value:
            return "add_missing_import"
        
        # Type errors
        if failure.category == FailureCategory.TYPE_ERROR.value:
            return "fix_type_error"
        
        # Attribute errors
        if failure.category == FailureCategory.ATTRIBUTE_ERROR.value:
            return "fix_attribute_error"
        
        # Default to manual review
        return "manual_review"
    
    def fix_instrument_data_len(self, failure: FailureRecord) -> bool:
        """
        Fix InstrumentData missing __len__ method.
        
        Args:
            failure: Failure record
        
        Returns:
            True if fix was successful
        """
        print(f"\n🔧 Fixing InstrumentData len() issue...")
        
        # Find the InstrumentData class definition
        # This is likely in exploration_integration.py or similar
        possible_files = [
            project_root / "kinetra" / "exploration_integration.py",
            project_root / "kinetra" / "data_package.py",
            project_root / "kinetra" / "measurements.py",
        ]
        
        for file_path in possible_files:
            if not file_path.exists():
                continue
            
            with open(file_path, 'r') as f:
                content = f.read()
            
            # Look for InstrumentData class
            if "class InstrumentData" in content:
                print(f"  Found InstrumentData in {file_path.name}")
                
                # Check if __len__ already exists
                if "__len__" in content:
                    print("  ✓ __len__ method already exists")
                    return True
                
                # Find the class and add __len__ method
                fixed_content = self._add_len_method_to_class(content, "InstrumentData")
                
                if fixed_content != content:
                    if not self.dry_run:
                        # Backup original
                        self._backup_file(file_path)
                        
                        # Write fixed content
                        with open(file_path, 'w') as f:
                            f.write(fixed_content)
                        
                        print(f"  ✓ Added __len__ method to InstrumentData")
                        return True
                    else:
                        print(f"  [DRY RUN] Would add __len__ method to InstrumentData")
                        return True
        
        print("  ✗ Could not find InstrumentData class")
        return False
    
    def _add_len_method_to_class(self, content: str, class_name: str) -> str:
        """
        Add __len__ method to a dataclass.
        
        Args:
            content: File content
            class_name: Name of the class
        
        Returns:
            Modified content
        """
        # Find the class definition
        class_pattern = rf"(class {class_name}[^:]*:.*?)(\n(?=class|\Z))"
        
        match = re.search(class_pattern, content, re.DOTALL)
        if not match:
            return content
        
        class_body = match.group(1)
        
        # Define __len__ method as separate lines for proper formatting
        len_method_lines = [
            "",
            "    def __len__(self) -> int:",
            '        """Return the number of data points."""',
            "        # Try common attribute names for data storage",
            "        for attr in ['data', 'prices', 'bars', 'values', 'items']:",
            "            if hasattr(self, attr):",
            "                value = getattr(self, attr)",
            "                if hasattr(value, '__len__'):",
            "                    return len(value)",
            "        # Default to 0 if no suitable attribute found",
            "        return 0",
            "",
        ]
        len_method = "\n".join(len_method_lines)
        
        # Insert before the next class or end of file
        # Find good insertion point (after last field/method)
        lines = class_body.split('\n')
        insertion_point = len(lines) - 1
        
        # Find last method or field
        for i in range(len(lines) - 1, -1, -1):
            line = lines[i].strip()
            if line and not line.startswith('#'):
                insertion_point = i + 1
                break
        
        # Insert the method
        lines.insert(insertion_point, len_method)
        new_class_body = '\n'.join(lines)
        
        # Replace in content
        new_content = content.replace(class_body, new_class_body)
        return new_content
    
    def fix_explore_interactive_len_call(self) -> bool:
        """Fix the len(data) call in explore_interactive.py."""
        print(f"\n🔧 Fixing explore_interactive.py len() call...")
        
        script_path = project_root / "scripts" / "explore_interactive.py"
        if not script_path.exists():
            print(f"  ✗ Script not found: {script_path}")
            return False
        
        with open(script_path, 'r') as f:
            content = f.read()
        
        # Find the problematic line (around line 293)
        # Replace: print(f"  {key}: {len(data)} bars")
        # With: print(f"  {key}: {len(data.data) if hasattr(data, 'data') else len(data.prices) if hasattr(data, 'prices') else 'N/A'} bars")
        
        patterns = [
            (r'print\(f"  \{key\}: \{len\(data\)\} bars"\)', 
             r'print(f"  {key}: {len(data.data) if hasattr(data, \'data\') else len(data.prices) if hasattr(data, \'prices\') else \'N/A\'} bars")'),
            (r'len\(data\)', 
             r'(len(data.data) if hasattr(data, "data") else len(data.prices) if hasattr(data, "prices") else 0)'),
        ]
        
        modified = False
        for pattern, replacement in patterns:
            if re.search(pattern, content):
                new_content = re.sub(pattern, replacement, content)
                if new_content != content:
                    content = new_content
                    modified = True
                    print(f"  ✓ Fixed len(data) call")
        
        if modified:
            if not self.dry_run:
                # Backup original
                self._backup_file(script_path)
                
                # Write fixed content
                with open(script_path, 'w') as f:
                    f.write(content)
                
                print(f"  ✓ Applied fix to {script_path.name}")
                return True
            else:
                print(f"  [DRY RUN] Would fix {script_path.name}")
                return True
        
        print("  ✗ Could not find pattern to fix")
        return False
    
    def _backup_file(self, file_path: Path):
        """
        Create a backup of a file before modifying.
        
        Args:
            file_path: Path to file to backup
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"{file_path.name}.{timestamp}.bak"
        backup_path = self.backup_dir / backup_name
        
        shutil.copy2(file_path, backup_path)
        print(f"  📦 Backed up to: {backup_path}")
    
    def apply_fixes(self, strategies: Optional[Dict[str, List[FailureRecord]]] = None):
        """
        Apply automated fixes.
        
        Args:
            strategies: Fix strategies (if None, will analyze first)
        """
        if strategies is None:
            strategies = self.analyze_failures()
        
        print(f"\n{'='*60}")
        print("Applying Fixes" + (" [DRY RUN]" if self.dry_run else ""))
        print(f"{'='*60}\n")
        
        # Fix InstrumentData len() issue
        if strategies["add_len_method"] or strategies["fix_type_error"]:
            print("\n1. Fixing InstrumentData len() issue...")
            
            # Fix the class definition
            success = self.fix_instrument_data_len(
                strategies["add_len_method"][0] if strategies["add_len_method"] 
                else strategies["fix_type_error"][0]
            )
            
            if success:
                self.fixes_applied.append("instrument_data_len_method")
            else:
                self.fixes_failed.append("instrument_data_len_method")
            
            # Fix the script call
            success = self.fix_explore_interactive_len_call()
            
            if success:
                self.fixes_applied.append("explore_interactive_len_call")
            else:
                self.fixes_failed.append("explore_interactive_len_call")
        
        # Print summary
        self.print_fix_summary()
    
    def print_fix_summary(self):
        """Print summary of applied fixes."""
        print(f"\n{'='*60}")
        print("Fix Summary")
        print(f"{'='*60}")
        print(f"Fixes applied:  {len(self.fixes_applied)}")
        for fix in self.fixes_applied:
            print(f"  ✓ {fix}")
        
        print(f"\nFixes failed:   {len(self.fixes_failed)}")
        for fix in self.fixes_failed:
            print(f"  ✗ {fix}")
        
        if self.fixes_applied and not self.dry_run:
            print(f"\n📦 Backups saved to: {self.backup_dir}")
    
    def validate_fixes(self) -> bool:
        """
        Validate that fixes didn't break anything.
        
        Returns:
            True if validation passed
        """
        print(f"\n{'='*60}")
        print("Validating Fixes")
        print(f"{'='*60}\n")
        
        # Try to import modules that were fixed
        validation_passed = True
        
        try:
            # Test InstrumentData fix
            print("Testing InstrumentData fix...")
            from kinetra.exploration_integration import InstrumentData
            
            # Check if __len__ exists
            if hasattr(InstrumentData, '__len__'):
                print("  ✓ __len__ method exists")
            else:
                print("  ✗ __len__ method missing")
                validation_passed = False
            
        except Exception as e:
            print(f"  ✗ Validation failed: {e}")
            validation_passed = False
        
        return validation_passed


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Fix silent failures in Kinetra codebase"
    )
    parser.add_argument(
        "--analyze",
        action="store_true",
        help="Analyze failures and suggest fixes",
    )
    parser.add_argument(
        "--fix",
        action="store_true",
        help="Apply automated fixes",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be fixed without applying",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate applied fixes",
    )
    parser.add_argument(
        "--auto",
        action="store_true",
        help="Automatically apply all fixes without confirmation",
    )
    
    args = parser.parse_args()
    
    # Initialize fixer
    fixer = FailureFixer(dry_run=args.dry_run)
    
    # Analyze failures
    if args.analyze or args.fix:
        strategies = fixer.analyze_failures()
        
        if args.fix:
            if not args.auto and not args.dry_run:
                response = input("\nApply fixes? (y/N): ")
                if response.lower() != 'y':
                    print("Aborted.")
                    return
            
            fixer.apply_fixes(strategies)
            
            if args.validate and not args.dry_run:
                validation_passed = fixer.validate_fixes()
                if not validation_passed:
                    print("\n⚠ Validation failed! Review changes before committing.")
                    sys.exit(1)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
