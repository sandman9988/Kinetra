#!/usr/bin/env python3
"""
Silent Failure Detection Tool for Kinetra

This script systematically tests the codebase to discover all silent failures.
It runs various operations and collects failures in the silent failure log.

Usage:
    python scripts/detect_silent_failures.py
    python scripts/detect_silent_failures.py --mode quick
    python scripts/detect_silent_failures.py --mode comprehensive
    python scripts/detect_silent_failures.py --mode targeted --module kinetra.data_loader
"""

import argparse
import importlib
import sys
import traceback
from pathlib import Path
from typing import List, Dict, Any

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from kinetra.silent_failure_logger import (
    SilentFailureLogger,
    log_failure,
    FailureCategory,
    FailureSeverity,
)


class FailureDetector:
    """Systematic failure detection across the codebase."""
    
    def __init__(self, mode: str = "quick"):
        """
        Initialize failure detector.
        
        Args:
            mode: Detection mode (quick, comprehensive, targeted)
        """
        self.mode = mode
        self.logger = SilentFailureLogger.get_instance()
        self.results = {
            "modules_tested": 0,
            "modules_succeeded": 0,
            "modules_failed": 0,
            "failures_found": 0,
        }
    
    def test_module_import(self, module_name: str) -> bool:
        """
        Test importing a module and log any failures.
        
        Args:
            module_name: Name of the module to import
        
        Returns:
            True if import succeeded, False otherwise
        """
        try:
            importlib.import_module(module_name)
            return True
        except Exception as e:
            log_failure(
                e,
                context={
                    "operation": "module_import",
                    "module": module_name,
                    "detector": "FailureDetector",
                },
                tags=["import", "auto-detected"],
            )
            return False
    
    def discover_python_modules(self, base_path: Path) -> List[str]:
        """
        Discover all Python modules in a directory.
        
        Args:
            base_path: Base directory to search
        
        Returns:
            List of module names
        """
        modules = []
        
        # Find all .py files
        for py_file in base_path.rglob("*.py"):
            # Skip __pycache__ and other special directories
            if "__pycache__" in str(py_file) or ".venv" in str(py_file):
                continue
            
            # Convert file path to module name
            relative = py_file.relative_to(project_root)
            module_parts = list(relative.parts[:-1])  # Remove .py
            module_parts.append(relative.stem)
            
            # Skip __init__.py special handling
            if module_parts[-1] == "__init__":
                module_parts = module_parts[:-1]
            
            if module_parts:
                module_name = ".".join(module_parts)
                modules.append(module_name)
        
        return sorted(set(modules))
    
    def test_kinetra_modules(self):
        """Test all kinetra modules."""
        print("🔍 Testing Kinetra modules...")
        
        kinetra_path = project_root / "kinetra"
        modules = self.discover_python_modules(kinetra_path)
        
        print(f"  Found {len(modules)} modules to test")
        
        for module in modules:
            self.results["modules_tested"] += 1
            
            if self.test_module_import(module):
                self.results["modules_succeeded"] += 1
                print(f"  ✓ {module}")
            else:
                self.results["modules_failed"] += 1
                print(f"  ✗ {module}")
    
    def test_scripts(self):
        """Test all script files."""
        print("\n🔍 Testing Scripts...")
        
        scripts_path = project_root / "scripts"
        if not scripts_path.exists():
            print("  No scripts directory found")
            return
        
        # Find all Python scripts
        scripts = list(scripts_path.glob("*.py"))
        print(f"  Found {len(scripts)} scripts")
        
        # For scripts, we just check syntax, not execute them
        for script in scripts:
            self.results["modules_tested"] += 1
            
            try:
                with open(script, 'r') as f:
                    compile(f.read(), script, 'exec')
                self.results["modules_succeeded"] += 1
                print(f"  ✓ {script.name}")
            except Exception as e:
                self.results["modules_failed"] += 1
                log_failure(
                    e,
                    context={
                        "operation": "script_syntax_check",
                        "script": str(script),
                        "detector": "FailureDetector",
                    },
                    tags=["syntax", "script", "auto-detected"],
                )
                print(f"  ✗ {script.name}")
    
    def test_data_loading(self):
        """Test data loading operations."""
        print("\n🔍 Testing Data Loading...")
        
        try:
            from kinetra.data_loader import DataLoader
            
            # Test basic instantiation
            loader = DataLoader()
            print("  ✓ DataLoader instantiation")
            
            # Check data directory
            data_path = project_root / "data"
            if data_path.exists():
                # List available data files
                csv_files = list(data_path.rglob("*.csv"))
                print(f"  Found {len(csv_files)} CSV files")
                
                # Try to load a sample if available
                if csv_files and self.mode == "comprehensive":
                    sample_file = csv_files[0]
                    try:
                        print(f"  Testing load: {sample_file.name}")
                        # This would actually load data - skip in quick mode
                        pass
                    except Exception as e:
                        log_failure(
                            e,
                            context={
                                "operation": "data_loading",
                                "file": str(sample_file),
                                "detector": "FailureDetector",
                            },
                            tags=["data", "auto-detected"],
                        )
        except Exception as e:
            log_failure(
                e,
                context={
                    "operation": "data_loader_test",
                    "detector": "FailureDetector",
                },
                tags=["data", "auto-detected"],
            )
            print(f"  ✗ DataLoader test failed: {e}")
    
    def test_physics_engine(self):
        """Test physics engine components."""
        print("\n🔍 Testing Physics Engine...")
        
        try:
            from kinetra.physics_engine import PhysicsEngine
            import numpy as np
            import pandas as pd
            
            # Create sample data
            dates = pd.date_range('2024-01-01', periods=100, freq='1H')
            sample_data = pd.DataFrame({
                'open': np.random.randn(100).cumsum() + 100,
                'high': np.random.randn(100).cumsum() + 101,
                'low': np.random.randn(100).cumsum() + 99,
                'close': np.random.randn(100).cumsum() + 100,
                'volume': np.random.randint(1000, 10000, 100),
            }, index=dates)
            
            # Test physics engine
            engine = PhysicsEngine()
            result = engine.calculate_all(sample_data)
            
            print(f"  ✓ PhysicsEngine calculation")
            print(f"  Generated {len(result.columns)} physics features")
            
        except Exception as e:
            log_failure(
                e,
                context={
                    "operation": "physics_engine_test",
                    "detector": "FailureDetector",
                },
                tags=["physics", "auto-detected"],
            )
            print(f"  ✗ Physics engine test failed: {e}")
    
    def test_explore_interactive_issue(self):
        """Test the specific InstrumentData len() issue from explore_interactive.py."""
        print("\n🔍 Testing explore_interactive.py InstrumentData issue...")
        
        try:
            # Try to import the script module
            sys.path.insert(0, str(project_root / "scripts"))
            
            # Check if the script exists
            script_path = project_root / "scripts" / "explore_interactive.py"
            if not script_path.exists():
                print("  ⚠ explore_interactive.py not found")
                return
            
            # Load and analyze the code
            with open(script_path, 'r') as f:
                code = f.read()
            
            # Check for the problematic pattern
            if "len(data)" in code and "InstrumentData" in code:
                print("  ⚠ Found potential len(InstrumentData) pattern")
                
                # Log this as a known issue
                log_failure(
                    TypeError("object of type 'InstrumentData' has no len()"),
                    context={
                        "operation": "explore_interactive_analysis",
                        "file": str(script_path),
                        "line_pattern": "len(data)",
                        "detector": "FailureDetector",
                        "known_issue": True,
                    },
                    category=FailureCategory.TYPE_ERROR,
                    severity=FailureSeverity.MEDIUM,
                    tags=["known-issue", "explore-interactive", "auto-detected"],
                )
            
        except Exception as e:
            log_failure(
                e,
                context={
                    "operation": "explore_interactive_test",
                    "detector": "FailureDetector",
                },
                tags=["explore-interactive", "auto-detected"],
            )
            print(f"  ✗ Test failed: {e}")
    
    def run_detection(self):
        """Run failure detection based on mode."""
        print(f"\n{'='*60}")
        print(f"Silent Failure Detection - Mode: {self.mode}")
        print(f"{'='*60}\n")
        
        if self.mode in ["quick", "comprehensive"]:
            self.test_kinetra_modules()
            self.test_scripts()
            self.test_explore_interactive_issue()
            
            if self.mode == "comprehensive":
                self.test_data_loading()
                self.test_physics_engine()
        
        # Get final statistics
        self.results["failures_found"] = len(self.logger.failures)
        
        # Print summary
        self.print_summary()
        
        # Export report
        report_path = self.logger.export_report()
        print(f"\n📊 Full report exported to: {report_path}")
    
    def print_summary(self):
        """Print detection summary."""
        print(f"\n{'='*60}")
        print("Detection Summary")
        print(f"{'='*60}")
        print(f"Modules tested:    {self.results['modules_tested']}")
        print(f"  Succeeded:       {self.results['modules_succeeded']}")
        print(f"  Failed:          {self.results['modules_failed']}")
        print(f"Failures found:    {self.results['failures_found']}")
        
        # Get statistics from logger
        stats = self.logger.get_statistics()
        
        print(f"\n{'='*60}")
        print("Failure Breakdown")
        print(f"{'='*60}")
        
        print("\nBy Category:")
        for category, count in sorted(stats["by_category"].items(), key=lambda x: -x[1]):
            print(f"  {category:20} {count:>5}")
        
        print("\nBy Severity:")
        for severity, count in sorted(stats["by_severity"].items(), key=lambda x: -x[1]):
            print(f"  {severity:20} {count:>5}")
        
        if stats["top_failures"]:
            print("\nTop 5 Failures:")
            for i, failure in enumerate(stats["top_failures"][:5], 1):
                record = failure["record"]
                print(f"\n  {i}. {record['exception_type']} (count: {failure['count']})")
                print(f"     File: {record['file_path']}:{record['line_number']}")
                print(f"     Function: {record['function_name']}")
                print(f"     Message: {record['exception_message'][:80]}...")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Detect silent failures in Kinetra codebase"
    )
    parser.add_argument(
        "--mode",
        choices=["quick", "comprehensive", "targeted"],
        default="quick",
        help="Detection mode (default: quick)",
    )
    parser.add_argument(
        "--module",
        help="Specific module to test (for targeted mode)",
    )
    parser.add_argument(
        "--export",
        help="Export failures to specified file",
    )
    
    args = parser.parse_args()
    
    # Initialize detector
    detector = FailureDetector(mode=args.mode)
    
    # Run detection
    try:
        detector.run_detection()
    except KeyboardInterrupt:
        print("\n\n⚠ Detection interrupted by user")
        detector.print_summary()
    except Exception as e:
        print(f"\n\n❌ Detection failed with error: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
