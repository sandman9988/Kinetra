#!/usr/bin/env python3
"""
Kinetra Rules Linter - Validates code against canonical rules in AGENT_RULES_MASTER.md

This script enforces non-negotiable constraints from the canonical rulebook:
- No traditional TA indicators (RSI, MACD, BB, ATR, ADX, etc.)
- No magic numbers in thresholds
- No hardcoded API keys/credentials
- Use of PersistenceManager.atomic_save() for data operations
- Proper RNG seeding in backtest code
- No live order placement code
- Vectorization over Python loops where possible

Exit code 0: All checks passed
Exit code 1: Rule violations found
"""

import argparse
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple


class RuleViolation:
    """Represents a rule violation found in code."""

    def __init__(
        self, filepath: Path, line_num: int, rule: str, message: str, severity: str = "ERROR"
    ):
        self.filepath = filepath
        self.line_num = line_num
        self.rule = rule
        self.message = message
        self.severity = severity

    def __str__(self) -> str:
        return f"{self.filepath}:{self.line_num}: {self.severity}: [{self.rule}] {self.message}"


class RulesLinter:
    """Lints Kinetra codebase against canonical rules."""

    # Traditional TA indicators that are BANNED
    BANNED_TA_INDICATORS = [
        r"\bRSI\b",
        r"\brsi\b",
        r"\bMACD\b",
        r"\bmacd\b",
        r"\bBollinger\b",
        r"\bbollinger\b",
        r"\bBB\b",
        r"\bATR\b",
        r"\batr\b",
        r"\bADX\b",
        r"\badx\b",
        r"\bStochastic\b",
        r"\bstochastic\b",
        r"\bCCI\b",
        r"\bcci\b",
        r"\bWilliams\b",
        r"\bwilliams\b",
        r"\bSMA\b",
        r"\bEMA\b",  # Simple/Exponential Moving Average
    ]

    # Patterns for magic numbers (static thresholds)
    MAGIC_NUMBER_PATTERNS = [
        (r"if\s+\w+\s*[><=]+\s*0\.\d+", "Static threshold comparison"),
        (r"if\s+\w+\s*[><=]+\s*[12]\.\d+", "Static threshold comparison"),
        (r"\.rolling\(\s*\d+\s*\)", "Fixed rolling window (should be DSP-derived)"),
        (r"window\s*=\s*\d+(?!\s*#\s*DSP)", "Hardcoded window size (should be DSP-derived)"),
        (r"period\s*=\s*\d+(?!\s*#\s*DSP)", "Hardcoded period (should be DSP-derived)"),
    ]

    # Security patterns
    SECURITY_PATTERNS = [
        (
            r'(METAAPI_TOKEN|API_KEY|SECRET|PASSWORD)\s*=\s*["\'][^"\']+["\']',
            "Hardcoded credential",
        ),
        (r"(?i)place.*order(?!.*#.*test)", "Live order placement (only backtest/paper allowed)"),
        (r"\.execute_order\((?!.*test)", "Direct order execution (use paper trading)"),
    ]

    # Data safety patterns
    DATA_SAFETY_PATTERNS = [
        (r"\.to_csv\((?!.*atomic)", "Direct CSV write (use PersistenceManager.atomic_save)"),
        (r'with\s+open\([^)]*,\s*["\']w["\'](?!.*temp)', "Direct file write (use atomic_save)"),
        (r"os\.remove\((?!.*backup)", "File deletion without backup"),
    ]

    # RNG/Determinism patterns
    RNG_PATTERNS = [
        (r"random\.(choice|randint|uniform|sample)(?!.*seed)", "Random call without seeding"),
        (
            r"np\.random\.(choice|randint|uniform|rand|randn)(?!.*seed)",
            "NumPy random without seeding",
        ),
    ]

    # Anti-patterns (loops that should be vectorized)
    VECTORIZATION_WARNINGS = [
        (r"for\s+i\s+in\s+range\(len\(", "Python loop over range(len()) - consider vectorization"),
        (r"for\s+\w+\s+in\s+df\.iterrows\(\)", "DataFrame.iterrows() - consider vectorization"),
        (r"\.apply\(lambda.*for.*in", "apply() with loop inside - consider vectorization"),
    ]

    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.violations: List[RuleViolation] = []

    def lint_file(self, filepath: Path) -> None:
        """Lint a single Python file."""
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                lines = f.readlines()
        except Exception as e:
            print(f"WARNING: Could not read {filepath}: {e}", file=sys.stderr)
            return

        for line_num, line in enumerate(lines, start=1):
            self._check_ta_indicators(filepath, line_num, line)
            self._check_magic_numbers(filepath, line_num, line)
            self._check_security(filepath, line_num, line)
            self._check_data_safety(filepath, line_num, line)
            self._check_rng_seeding(filepath, line_num, line)
            self._check_vectorization(filepath, line_num, line)

    def _check_ta_indicators(self, filepath: Path, line_num: int, line: str) -> None:
        """Check for banned traditional TA indicators."""
        # Skip comments and docstrings
        if line.strip().startswith("#") or '"""' in line or "'''" in line:
            return

        for pattern in self.BANNED_TA_INDICATORS:
            if re.search(pattern, line):
                self.violations.append(
                    RuleViolation(
                        filepath,
                        line_num,
                        "NO_TA_INDICATORS",
                        f"Banned TA indicator detected: {pattern}. Use physics-based features only.",
                        severity="ERROR",
                    )
                )

    def _check_magic_numbers(self, filepath: Path, line_num: int, line: str) -> None:
        """Check for magic numbers and static thresholds."""
        # Skip test files and configuration
        if "test_" in str(filepath) or "config" in str(filepath):
            return

        # Skip lines with explicit justification comments
        if "# magic number ok" in line.lower() or "# static threshold ok" in line.lower():
            return

        for pattern, description in self.MAGIC_NUMBER_PATTERNS:
            if re.search(pattern, line, re.IGNORECASE):
                self.violations.append(
                    RuleViolation(
                        filepath,
                        line_num,
                        "NO_MAGIC_NUMBERS",
                        f"{description}. Use rolling percentiles or DSP-derived values.",
                        severity="WARNING",
                    )
                )

    def _check_security(self, filepath: Path, line_num: int, line: str) -> None:
        """Check for security violations."""
        for pattern, description in self.SECURITY_PATTERNS:
            if re.search(pattern, line, re.IGNORECASE):
                self.violations.append(
                    RuleViolation(
                        filepath,
                        line_num,
                        "SECURITY_VIOLATION",
                        f"{description}. This is a hard prohibition.",
                        severity="ERROR",
                    )
                )

    def _check_data_safety(self, filepath: Path, line_num: int, line: str) -> None:
        """Check for unsafe data operations."""
        # Skip test files
        if "test_" in str(filepath):
            return

        for pattern, description in self.DATA_SAFETY_PATTERNS:
            if re.search(pattern, line):
                self.violations.append(
                    RuleViolation(
                        filepath,
                        line_num,
                        "DATA_SAFETY",
                        f"{description}. Use PersistenceManager for data operations.",
                        severity="ERROR",
                    )
                )

    def _check_rng_seeding(self, filepath: Path, line_num: int, line: str) -> None:
        """Check for unseeded random number generation in backtest code."""
        # Only check backtest-related files
        if "backtest" not in str(filepath).lower() and "test_" not in str(filepath):
            return

        for pattern, description in self.RNG_PATTERNS:
            if re.search(pattern, line):
                self.violations.append(
                    RuleViolation(
                        filepath,
                        line_num,
                        "DETERMINISM",
                        f"{description}. Backtests must be deterministic.",
                        severity="ERROR",
                    )
                )

    def _check_vectorization(self, filepath: Path, line_num: int, line: str) -> None:
        """Check for code that should be vectorized."""
        # Skip test files
        if "test_" in str(filepath):
            return

        # Skip if explicitly marked as unavoidable
        if "vectorization unavoidable" in line.lower() or "loop required" in line.lower():
            return

        for pattern, description in self.VECTORIZATION_WARNINGS:
            if re.search(pattern, line):
                self.violations.append(
                    RuleViolation(
                        filepath,
                        line_num,
                        "VECTORIZATION",
                        f"{description}. Prefer NumPy/Pandas vectorized operations.",
                        severity="WARNING",
                    )
                )

    def lint_project(self, paths: List[Path] = None) -> None:
        """Lint the entire project or specific paths."""
        if paths is None:
            # Default: lint kinetra/ directory
            paths = [self.project_root / "kinetra"]

        python_files = []
        for path in paths:
            if path.is_file() and path.suffix == ".py":
                python_files.append(path)
            elif path.is_dir():
                python_files.extend(path.rglob("*.py"))

        for filepath in python_files:
            # Skip __pycache__ and .venv
            if "__pycache__" in str(filepath) or ".venv" in str(filepath):
                continue
            self.lint_file(filepath)

    def check_canonical_references(self) -> None:
        """Check that instruction files reference the canonical rulebook."""
        instruction_files = [
            self.project_root / ".claude" / "instructions.md",
            self.project_root / ".github" / "copilot-instructions.md",
            self.project_root / ".claude" / "type_checking_guidelines.md",
        ]

        master_rulebook = "AGENT_RULES_MASTER.md"

        for filepath in instruction_files:
            if not filepath.exists():
                continue

            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    content = f.read()

                if master_rulebook not in content:
                    self.violations.append(
                        RuleViolation(
                            filepath,
                            1,
                            "CANONICAL_REFERENCE",
                            f"Instruction file does not reference {master_rulebook}",
                            severity="WARNING",
                        )
                    )
            except Exception as e:
                print(f"WARNING: Could not read {filepath}: {e}", file=sys.stderr)

    def print_report(self) -> None:
        """Print a summary report of violations."""
        if not self.violations:
            print("✅ No rule violations found!")
            return

        # Group by severity
        errors = [v for v in self.violations if v.severity == "ERROR"]
        warnings = [v for v in self.violations if v.severity == "WARNING"]

        if errors:
            print(f"\n❌ ERRORS ({len(errors)}):")
            for violation in errors:
                print(f"  {violation}")

        if warnings:
            print(f"\n⚠️  WARNINGS ({len(warnings)}):")
            for violation in warnings:
                print(f"  {violation}")

        print(f"\nTotal: {len(errors)} errors, {len(warnings)} warnings")

        if errors:
            print("\n❌ ERRORS found - please fix before committing")
        elif warnings:
            print("\n⚠️  Warnings found - consider addressing these")

    def has_errors(self) -> bool:
        """Return True if any ERROR-level violations found."""
        return any(v.severity == "ERROR" for v in self.violations)


def main():
    parser = argparse.ArgumentParser(
        description="Lint Kinetra code against canonical rules",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Lint entire project
  python scripts/lint_rules.py

  # Lint specific files
  python scripts/lint_rules.py kinetra/physics_engine.py

  # Lint specific directory
  python scripts/lint_rules.py kinetra/rl/

  # Check canonical references only
  python scripts/lint_rules.py --check-references
        """,
    )
    parser.add_argument(
        "paths", nargs="*", help="Specific files or directories to lint (default: kinetra/)"
    )
    parser.add_argument(
        "--check-references",
        action="store_true",
        help="Only check that instruction files reference the canonical rulebook",
    )
    parser.add_argument(
        "--warnings-as-errors", action="store_true", help="Treat warnings as errors (exit code 1)"
    )

    args = parser.parse_args()

    # Find project root (contains AGENT_RULES_MASTER.md)
    current = Path(__file__).resolve()
    project_root = None
    for parent in [current] + list(current.parents):
        if (parent / "AGENT_RULES_MASTER.md").exists():
            project_root = parent
            break

    if project_root is None:
        print(
            "ERROR: Could not find project root (AGENT_RULES_MASTER.md not found)", file=sys.stderr
        )
        sys.exit(1)

    linter = RulesLinter(project_root)

    # Check canonical references
    if args.check_references:
        linter.check_canonical_references()
    else:
        # Lint code
        if args.paths:
            paths = [Path(p) for p in args.paths]
        else:
            paths = None
        linter.lint_project(paths)

        # Also check canonical references
        linter.check_canonical_references()

    # Print report
    linter.print_report()

    # Exit with appropriate code
    if linter.has_errors():
        sys.exit(1)
    elif args.warnings_as_errors and linter.violations:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
