#!/usr/bin/env python3
"""
Vectorization Linter for Kinetra Project

Automatically detects Python code patterns that violate the vectorization rule:
"Explicit Python loops are the last resort. Prefer NumPy/Pandas vectorized ops."

Usage:
    python scripts/vectorization_linter.py                    # Scan entire project
    python scripts/vectorization_linter.py path/to/file.py    # Scan specific file
    python scripts/vectorization_linter.py --fix              # Auto-fix simple cases
"""

import argparse
import ast
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple


@dataclass
class VectorizationViolation:
    """Represents a detected vectorization violation."""

    file_path: str
    line_number: int
    severity: str  # 'high', 'medium', 'low'
    violation_type: str
    code_snippet: str
    suggestion: str

    def __str__(self):
        severity_emoji = {"high": "🔴", "medium": "🟡", "low": "🟢"}
        return (
            f"{severity_emoji[self.severity]} {self.file_path}:{self.line_number}\n"
            f"  Type: {self.violation_type}\n"
            f"  Code: {self.code_snippet.strip()}\n"
            f"  💡 {self.suggestion}\n"
        )


class VectorizationLinter(ast.NodeVisitor):
    """AST-based linter for detecting vectorization violations."""

    def __init__(self, file_path: str, source_code: str):
        self.file_path = file_path
        self.source_lines = source_code.splitlines()
        self.violations: List[VectorizationViolation] = []
        self.current_function = None
        self.in_loop = False

    def visit_FunctionDef(self, node: ast.FunctionDef):
        """Track current function for context."""
        old_function = self.current_function
        self.current_function = node.name
        self.generic_visit(node)
        self.current_function = old_function

    def visit_For(self, node: ast.For):
        """Detect for loops that could be vectorized."""
        old_in_loop = self.in_loop
        self.in_loop = True

        # Get the loop code
        code_snippet = self._get_code_snippet(node.lineno, node.end_lineno or node.lineno)

        # Check for iterrows() - HIGH SEVERITY
        if self._is_iterrows_loop(node):
            self.violations.append(
                VectorizationViolation(
                    file_path=self.file_path,
                    line_number=node.lineno,
                    severity="high",
                    violation_type="DataFrame.iterrows()",
                    code_snippet=code_snippet,
                    suggestion="Use vectorized operations: df['col'].apply(), df.assign(), or direct column operations",
                )
            )

        # Check for range(len()) with iloc - HIGH SEVERITY
        elif self._is_range_len_iloc_loop(node):
            self.violations.append(
                VectorizationViolation(
                    file_path=self.file_path,
                    line_number=node.lineno,
                    severity="high",
                    violation_type="range(len()) with .iloc[]",
                    code_snippet=code_snippet,
                    suggestion="Use .values or direct column operations: arr = df['col'].values",
                )
            )

        # Check for list append in loop - MEDIUM SEVERITY
        elif self._has_append_in_loop(node):
            self.violations.append(
                VectorizationViolation(
                    file_path=self.file_path,
                    line_number=node.lineno,
                    severity="medium",
                    violation_type="list.append() in loop",
                    code_snippet=code_snippet,
                    suggestion="Use list comprehension or pre-allocate NumPy array",
                )
            )

        # Check for DataFrame append in loop - HIGH SEVERITY
        elif self._has_dataframe_append_in_loop(node):
            self.violations.append(
                VectorizationViolation(
                    file_path=self.file_path,
                    line_number=node.lineno,
                    severity="high",
                    violation_type="DataFrame.append() in loop",
                    code_snippet=code_snippet,
                    suggestion="Collect data in list, then pd.DataFrame(data) once at the end",
                )
            )

        self.generic_visit(node)
        self.in_loop = old_in_loop

    def _is_iterrows_loop(self, node: ast.For) -> bool:
        """Check if loop uses DataFrame.iterrows()."""
        if isinstance(node.iter, ast.Call):
            if isinstance(node.iter.func, ast.Attribute):
                return node.iter.func.attr == "iterrows"
        return False

    def _is_range_len_iloc_loop(self, node: ast.For) -> bool:
        """Check for range(len(df)) with df.iloc[i] pattern."""
        # Check if iterator is range(len(...))
        if isinstance(node.iter, ast.Call):
            if isinstance(node.iter.func, ast.Name) and node.iter.func.id == "range":
                if len(node.iter.args) > 0:
                    arg = node.iter.args[0]
                    if isinstance(arg, ast.Call):
                        if isinstance(arg.func, ast.Name) and arg.func.id == "len":
                            # Now check if body contains .iloc[i] or similar
                            return self._contains_iloc_access(node.body)
        return False

    def _contains_iloc_access(self, body: List[ast.stmt]) -> bool:
        """Check if loop body contains .iloc[] access."""
        for stmt in ast.walk(ast.Module(body=body)):
            if isinstance(stmt, ast.Subscript):
                if isinstance(stmt.value, ast.Attribute):
                    if stmt.value.attr == "iloc":
                        return True
        return False

    def _has_append_in_loop(self, node: ast.For) -> bool:
        """Check if loop contains list.append() calls."""
        for stmt in ast.walk(ast.Module(body=node.body)):
            if isinstance(stmt, ast.Call):
                if isinstance(stmt.func, ast.Attribute):
                    if stmt.func.attr == "append":
                        # Check it's not a DataFrame append
                        if not self._is_dataframe_var(stmt.func.value):
                            return True
        return False

    def _has_dataframe_append_in_loop(self, node: ast.For) -> bool:
        """Check if loop contains DataFrame.append() calls."""
        for stmt in ast.walk(ast.Module(body=node.body)):
            if isinstance(stmt, ast.Call):
                if isinstance(stmt.func, ast.Attribute):
                    if stmt.func.attr == "append":
                        if self._is_dataframe_var(stmt.func.value):
                            return True
        return False

    def _is_dataframe_var(self, node: ast.expr) -> bool:
        """Heuristic to check if variable might be a DataFrame."""
        if isinstance(node, ast.Name):
            name = node.id.lower()
            return "df" in name or "data" in name or "frame" in name
        return False

    def _get_code_snippet(self, start_line: int, end_line: int, max_lines: int = 3) -> str:
        """Extract code snippet from source."""
        try:
            lines = self.source_lines[start_line - 1 : min(start_line - 1 + max_lines, end_line)]
            return "\n".join(lines)
        except IndexError:
            return "<code unavailable>"


def scan_file(file_path: str) -> List[VectorizationViolation]:
    """Scan a single Python file for vectorization violations."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            source_code = f.read()

        # Parse AST
        tree = ast.parse(source_code, filename=file_path)

        # Run linter
        linter = VectorizationLinter(file_path, source_code)
        linter.visit(tree)

        return linter.violations

    except SyntaxError as e:
        print(f"⚠️  Syntax error in {file_path}: {e}", file=sys.stderr)
        return []
    except Exception as e:
        print(f"⚠️  Error scanning {file_path}: {e}", file=sys.stderr)
        return []


def scan_directory(
    root_dir: str, exclude_patterns: List[str] = None
) -> List[VectorizationViolation]:
    """Recursively scan directory for Python files."""
    if exclude_patterns is None:
        exclude_patterns = [
            "__pycache__",
            ".git",
            "venv",
            "env",
            ".pytest_cache",
            "node_modules",
            "archive",
        ]

    all_violations = []

    for root, dirs, files in os.walk(root_dir):
        # Filter out excluded directories
        dirs[:] = [d for d in dirs if not any(pattern in d for pattern in exclude_patterns)]

        for file in files:
            if file.endswith(".py"):
                file_path = os.path.join(root, file)
                violations = scan_file(file_path)
                all_violations.extend(violations)

    return all_violations


def print_summary(violations: List[VectorizationViolation]):
    """Print summary statistics."""
    if not violations:
        print("✅ No vectorization violations found!")
        return

    # Count by severity
    high = sum(1 for v in violations if v.severity == "high")
    medium = sum(1 for v in violations if v.severity == "medium")
    low = sum(1 for v in violations if v.severity == "low")

    print("\n" + "=" * 80)
    print("VECTORIZATION VIOLATIONS SUMMARY")
    print("=" * 80)
    print(f"🔴 High Priority:   {high:3d} violations")
    print(f"🟡 Medium Priority: {medium:3d} violations")
    print(f"🟢 Low Priority:    {low:3d} violations")
    print(f"   Total:           {len(violations):3d} violations")
    print("=" * 80)

    # Group by file
    by_file = {}
    for v in violations:
        if v.file_path not in by_file:
            by_file[v.file_path] = []
        by_file[v.file_path].append(v)

    print(f"\nFiles with violations: {len(by_file)}")
    print("\nTop 10 files by violation count:")
    sorted_files = sorted(by_file.items(), key=lambda x: len(x[1]), reverse=True)[:10]
    for file_path, file_violations in sorted_files:
        rel_path = os.path.relpath(file_path)
        print(f"  {len(file_violations):3d} - {rel_path}")


def main():
    parser = argparse.ArgumentParser(description="Lint Python code for vectorization violations")
    parser.add_argument(
        "path",
        nargs="?",
        default=".",
        help="File or directory to scan (default: current directory)",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Show all violations with details"
    )
    parser.add_argument(
        "--severity", choices=["high", "medium", "low"], help="Filter by severity level"
    )
    parser.add_argument(
        "--summary-only", "-s", action="store_true", help="Show only summary statistics"
    )
    parser.add_argument("--output", "-o", help="Write results to file")

    args = parser.parse_args()

    # Determine if path is file or directory
    path = Path(args.path)

    if path.is_file():
        violations = scan_file(str(path))
    elif path.is_dir():
        violations = scan_directory(str(path))
    else:
        print(f"Error: {path} is not a valid file or directory", file=sys.stderr)
        sys.exit(1)

    # Filter by severity if specified
    if args.severity:
        violations = [v for v in violations if v.severity == args.severity]

    # Sort by severity (high -> medium -> low) then by file
    severity_order = {"high": 0, "medium": 1, "low": 2}
    violations.sort(key=lambda v: (severity_order[v.severity], v.file_path, v.line_number))

    # Output
    output_lines = []

    if not args.summary_only and (args.verbose or len(violations) <= 50):
        output_lines.append("\n" + "=" * 80)
        output_lines.append("DETAILED VIOLATIONS")
        output_lines.append("=" * 80 + "\n")
        for v in violations:
            output_lines.append(str(v))

    # Always show summary
    import io

    summary_buffer = io.StringIO()
    original_stdout = sys.stdout
    sys.stdout = summary_buffer
    print_summary(violations)
    sys.stdout = original_stdout
    summary_text = summary_buffer.getvalue()
    output_lines.append(summary_text)

    # Write to file or stdout
    output_text = "\n".join(output_lines)

    if args.output:
        with open(args.output, "w") as f:
            f.write(output_text)
        print(f"Results written to {args.output}")
    else:
        print(output_text)

    # Exit with error code if violations found
    sys.exit(1 if violations else 0)


if __name__ == "__main__":
    main()
