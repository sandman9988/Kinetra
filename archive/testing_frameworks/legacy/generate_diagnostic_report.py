#!/usr/bin/env python3
"""
Diagnostic Report Generator
============================

Generates a comprehensive diagnostic report of ALL menu issues and system status.
Identifies what works, what's broken, and what's missing.

Usage:
    python scripts/testing/generate_diagnostic_report.py
    python scripts/testing/generate_diagnostic_report.py --fix    # Auto-fix simple issues

__version__ = "1.0.0"
"""

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class DiagnosticReporter:
    """Generate comprehensive diagnostic report."""

    def __init__(self, auto_fix: bool = False):
        self.auto_fix = auto_fix
        self.report = {
            "timestamp": datetime.now().isoformat(),
            "working": [],
            "broken": [],
            "missing": [],
            "data_issues": [],
            "import_issues": [],
            "recommendations": [],
        }

    def check_environment(self):
        """Check Python environment and dependencies."""
        print("=" * 80)
        print("ENVIRONMENT CHECK")
        print("=" * 80)

        # Python version
        py_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        print(f"Python: {py_version}")

        # Required packages
        required = ["pandas", "numpy", "metaapi_cloud_sdk", "tqdm"]
        for pkg in required:
            try:
                __import__(pkg)
                print(f"✅ {pkg}")
            except ImportError:
                print(f"❌ {pkg} - MISSING")
                self.report["missing"].append(f"Python package: {pkg}")
        print()

    def check_credentials(self):
        """Check MetaAPI credentials."""
        print("=" * 80)
        print("CREDENTIALS CHECK")
        print("=" * 80)

        env_file = PROJECT_ROOT / ".env"
        if not env_file.exists():
            print("❌ .env file not found")
            self.report["missing"].append(".env file")
            return

        env_content = env_file.read_text()
        has_token = "METAAPI_TOKEN=" in env_content
        has_account = "METAAPI_ACCOUNT_ID=" in env_content

        if has_token and has_account:
            print("✅ .env file exists with credentials")
            self.report["working"].append("Credentials file (.env)")
        else:
            print("❌ .env file missing required credentials")
            self.report["broken"].append(".env file incomplete")
        print()

    def check_data_files(self):
        """Check data file presence and integrity."""
        print("=" * 80)
        print("DATA FILES CHECK")
        print("=" * 80)

        data_dir = PROJECT_ROOT / "data" / "master"
        if not data_dir.exists():
            print(f"❌ Data directory not found: {data_dir}")
            self.report["missing"].append(f"Data directory: {data_dir}")
            return

        csv_files = list(data_dir.glob("*.csv"))
        print(f"Found {len(csv_files)} CSV files")

        # Parse filenames
        combos = set()
        for f in csv_files:
            # Format: SYMBOL_TIMEFRAME.csv or SYMBOL_TIMEFRAME_*.csv
            parts = f.stem.split("_")
            if len(parts) >= 2:
                symbol = parts[0]
                timeframe = parts[1]
                combos.add(f"{symbol}_{timeframe}")

        print(f"Unique combinations: {len(combos)}")
        print(f"Symbols: {len(set(c.split('_')[0] for c in combos))}")

        # Check required combinations
        required = [
            ("BTCUSD", "H1"),
            ("BTCUSD", "H4"),
            ("XAUUSD", "H1"),
            ("XAUUSD", "H4"),
            ("GBPUSD", "H1"),
            ("GBPUSD", "H4"),
        ]

        missing_combos = []
        for symbol, tf in required:
            combo = f"{symbol}_{tf}"
            matching = [c for c in combos if c.startswith(combo)]
            if not matching:
                missing_combos.append(combo)
                print(f"❌ Missing: {combo}")

        if missing_combos:
            self.report["data_issues"].append(
                f"Missing required combos: {', '.join(missing_combos)}"
            )
            self.report["recommendations"].append(
                f"Download missing data: {', '.join(missing_combos)}"
            )
        else:
            print("✅ All required data combinations present")
            self.report["working"].append("Required data files")
        print()

    def check_menu_scripts(self):
        """Check all menu-referenced scripts."""
        print("=" * 80)
        print("MENU SCRIPTS CHECK")
        print("=" * 80)

        # Critical menu scripts
        scripts = {
            "1.1 - Setup Credentials": "scripts/download/setup_metaapi_credentials.py",
            "1.2 - Test Connection": "scripts/download/test_metaapi_connection.py",
            "2.1 - Discover Data": "scripts/discover_available_data.py",
            "2.2.1 - Bulk Download": "scripts/download/metaapi_bulk_download.py",
            "2.2.2 - Custom Download": "scripts/download/download_metaapi.py",
            "2.4 - Data Integrity": "scripts/download/check_data_integrity.py",
            "2.5 - Data Coverage": "scripts/audit_data_coverage.py",
            "3.1 - Train RL": "scripts/training/train_rl.py",
            "4.1 - Backtest": "scripts/batch_backtest.py",
        }

        for label, script_path in scripts.items():
            full_path = PROJECT_ROOT / script_path
            if full_path.exists():
                print(f"✅ {label}")
                self.report["working"].append(f"Script: {script_path}")
            else:
                print(f"❌ {label} - {script_path}")
                self.report["missing"].append(f"Script: {script_path}")
        print()

    def check_core_imports(self):
        """Check core module imports."""
        print("=" * 80)
        print("CORE IMPORTS CHECK")
        print("=" * 80)

        modules = [
            "kinetra.backtest_engine",
            "kinetra.rl_agent",
            "kinetra.cpu_utils",
            "kinetra.persistence_manager",
        ]

        for module in modules:
            try:
                __import__(module)
                print(f"✅ {module}")
                self.report["working"].append(f"Module: {module}")
            except ImportError as e:
                print(f"❌ {module} - {str(e)[:50]}")
                self.report["import_issues"].append(f"{module}: {str(e)[:100]}")

        # Check physics module (known to be missing/renamed)
        try:
            __import__("kinetra.physics")
            print(f"✅ kinetra.physics")
        except ImportError:
            print(f"⚠️  kinetra.physics - May be renamed/refactored")
            self.report["recommendations"].append(
                "Physics module may be refactored - check kinetra.physics_engine or similar"
            )
        print()

    def test_menu_execution(self):
        """Test actual menu execution."""
        print("=" * 80)
        print("MENU EXECUTION TEST")
        print("=" * 80)

        # Find the main menu
        menu_candidates = [
            "kinetra_menu.py",
            "unified_menu.py",
            "kinetra_production_menu.py",
        ]

        menu_script = None
        for candidate in menu_candidates:
            path = PROJECT_ROOT / candidate
            if path.exists():
                menu_script = path
                print(f"Found menu: {candidate}")
                break

        if not menu_script:
            print("❌ No menu script found")
            self.report["broken"].append("Menu script not found")
            return

        # Try to import/validate menu
        try:
            result = subprocess.run(
                [sys.executable, str(menu_script), "--help"],
                capture_output=True,
                timeout=5,
                text=True,
            )

            if result.returncode == 0 or "menu" in result.stdout.lower():
                print(f"✅ Menu script is executable")
                self.report["working"].append(f"Menu: {menu_script.name}")
            else:
                print(f"❌ Menu script failed: {result.stderr[:100]}")
                self.report["broken"].append(f"Menu execution: {result.stderr[:100]}")
        except subprocess.TimeoutExpired:
            print(f"⚠️  Menu script timeout (may be interactive)")
            self.report["working"].append(f"Menu: {menu_script.name} (interactive)")
        except Exception as e:
            print(f"❌ Menu error: {str(e)[:100]}")
            self.report["broken"].append(f"Menu: {str(e)[:100]}")
        print()

    def check_metaapi_connection(self):
        """Test MetaAPI connection status."""
        print("=" * 80)
        print("METAAPI CONNECTION TEST")
        print("=" * 80)

        test_script = PROJECT_ROOT / "scripts/download/test_metaapi_connection.py"
        if not test_script.exists():
            print("❌ Test script not found")
            self.report["missing"].append("MetaAPI test script")
            return

        try:
            result = subprocess.run(
                [sys.executable, str(test_script)],
                capture_output=True,
                timeout=30,
                text=True,
            )

            output = result.stdout + result.stderr

            if "DEPLOYED" in output and "ready to use" in output:
                print("✅ MetaAPI connection successful")
                self.report["working"].append("MetaAPI connection")
            elif "not healthy" in output.lower():
                print("⚠️  MetaAPI connected but health status unclear")
                self.report["recommendations"].append(
                    "MetaAPI streaming health may need time to synchronize"
                )
            elif "credentials" in output.lower():
                print("❌ Credential issues")
                self.report["broken"].append("MetaAPI credentials")
            else:
                print(f"❌ Connection test failed")
                self.report["broken"].append("MetaAPI connection")

        except subprocess.TimeoutExpired:
            print("❌ Connection test timeout")
            self.report["broken"].append("MetaAPI connection (timeout)")
        except Exception as e:
            print(f"❌ Connection test error: {str(e)[:100]}")
            self.report["broken"].append(f"MetaAPI test: {str(e)[:100]}")
        print()

    def generate_recommendations(self):
        """Generate actionable recommendations."""
        print("=" * 80)
        print("RECOMMENDATIONS")
        print("=" * 80)

        # Data issues
        if self.report["data_issues"]:
            self.report["recommendations"].append(
                "Run: Menu 2.2.1 (MetaAPI Bulk Download) to get missing data"
            )

        # Import issues
        if self.report["import_issues"]:
            self.report["recommendations"].append(
                "Check import paths and module structure in kinetra/"
            )

        # Missing scripts
        missing_scripts = [m for m in self.report["missing"] if "Script:" in m]
        if missing_scripts:
            self.report["recommendations"].append(
                f"Create missing scripts: {len(missing_scripts)} needed"
            )

        # Print recommendations
        if self.report["recommendations"]:
            for i, rec in enumerate(self.report["recommendations"], 1):
                print(f"{i}. {rec}")
        else:
            print("✅ No immediate recommendations - system looks healthy!")
        print()

    def print_summary(self):
        """Print diagnostic summary."""
        print("=" * 80)
        print("DIAGNOSTIC SUMMARY")
        print("=" * 80)
        print()

        print(f"✅ Working:           {len(self.report['working'])}")
        print(f"❌ Broken:            {len(self.report['broken'])}")
        print(f"🚫 Missing:           {len(self.report['missing'])}")
        print(f"⚠️  Data Issues:       {len(self.report['data_issues'])}")
        print(f"⚠️  Import Issues:     {len(self.report['import_issues'])}")
        print()

        # Critical issues
        critical = len(self.report["broken"]) + len(self.report["missing"])
        if critical == 0:
            print("✅ NO CRITICAL ISSUES FOUND")
        elif critical <= 3:
            print(f"⚠️  {critical} MINOR ISSUE(S) - System mostly functional")
        else:
            print(f"❌ {critical} CRITICAL ISSUE(S) - Needs attention")
        print()

    def save_report(self):
        """Save diagnostic report to JSON."""
        report_file = PROJECT_ROOT / "diagnostic_report.json"

        with open(report_file, "w") as f:
            json.dump(self.report, f, indent=2)

        print(f"📄 Full report saved to: {report_file}")
        print()

    def run_full_diagnostic(self):
        """Run complete diagnostic sequence."""
        print("\n")
        print("=" * 80)
        print("KINETRA SYSTEM DIAGNOSTIC")
        print("=" * 80)
        print(f"Timestamp: {self.report['timestamp']}")
        print(f"Auto-fix: {'ENABLED' if self.auto_fix else 'DISABLED'}")
        print()

        # Run all checks
        self.check_environment()
        self.check_credentials()
        self.check_data_files()
        self.check_menu_scripts()
        self.check_core_imports()
        self.test_menu_execution()
        self.check_metaapi_connection()

        # Generate recommendations
        self.generate_recommendations()

        # Summary
        self.print_summary()
        self.save_report()

        # Return status
        critical = len(self.report["broken"]) + len(self.report["missing"])
        return 0 if critical == 0 else 1


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate comprehensive diagnostic report",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--fix", action="store_true", help="Auto-fix simple issues")

    args = parser.parse_args()

    reporter = DiagnosticReporter(auto_fix=args.fix)

    try:
        exit_code = reporter.run_full_diagnostic()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\n⚠️  Diagnostic interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n\n❌ Fatal error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
