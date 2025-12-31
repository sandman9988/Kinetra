"""
Deduplication Utilities
=======================

Find and remove duplicate:
- Data files (CSV, Parquet)
- Code blocks
- Similar functions
"""

import difflib
import hashlib
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple


@dataclass
class DuplicateFile:
    """Represents a duplicate file."""
    path: Path
    hash: str
    size: int
    group_id: int


@dataclass
class DuplicateCode:
    """Represents duplicate code block."""
    file1: Path
    file2: Path
    line_start1: int
    line_end1: int
    line_start2: int
    line_end2: int
    similarity: float
    code_snippet: str


class DataDeduplicator:
    """
    Find and manage duplicate data files.

    Uses content hashing for exact duplicates and
    fuzzy matching for near-duplicates.
    """

    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.file_hashes: Dict[str, List[Path]] = defaultdict(list)

    def compute_hash(self, filepath: Path, chunk_size: int = 8192) -> str:
        """Compute SHA256 hash of file content."""
        sha256 = hashlib.sha256()
        try:
            with open(filepath, 'rb') as f:
                while chunk := f.read(chunk_size):
                    sha256.update(chunk)
            return sha256.hexdigest()
        except (IOError, PermissionError):
            return ""

    def scan_directory(self, patterns: List[str] = None) -> Dict[str, List[Path]]:
        """
        Scan directory for duplicate files.

        Args:
            patterns: File patterns to scan (default: ['*.csv', '*.parquet'])

        Returns:
            Dict mapping hash -> list of duplicate file paths
        """
        if patterns is None:
            patterns = ['*.csv', '*.parquet', '*.json']

        self.file_hashes.clear()

        for pattern in patterns:
            for filepath in self.data_dir.rglob(pattern):
                if filepath.is_file():
                    file_hash = self.compute_hash(filepath)
                    if file_hash:
                        self.file_hashes[file_hash].append(filepath)

        # Filter to only duplicates
        duplicates = {
            h: paths for h, paths in self.file_hashes.items()
            if len(paths) > 1
        }

        return duplicates

    def get_duplicate_report(self) -> str:
        """Generate human-readable duplicate report."""
        duplicates = self.scan_directory()

        if not duplicates:
            return "No duplicate files found."

        lines = ["=" * 60, "DUPLICATE DATA FILES REPORT", "=" * 60, ""]

        total_wasted = 0
        for i, (file_hash, paths) in enumerate(duplicates.items(), 1):
            size = paths[0].stat().st_size
            wasted = size * (len(paths) - 1)
            total_wasted += wasted

            lines.append(f"Group {i} (Hash: {file_hash[:16]}...)")
            lines.append(f"  Size: {size:,} bytes ({len(paths)} copies)")
            lines.append(f"  Wasted space: {wasted:,} bytes")
            for path in paths:
                lines.append(f"    - {path}")
            lines.append("")

        lines.append("-" * 60)
        lines.append(f"Total duplicate groups: {len(duplicates)}")
        lines.append(f"Total wasted space: {total_wasted:,} bytes ({total_wasted/1024/1024:.2f} MB)")

        return "\n".join(lines)

    def remove_duplicates(self, keep: str = "first", dry_run: bool = True) -> List[Path]:
        """
        Remove duplicate files.

        Args:
            keep: Which copy to keep ('first', 'last', 'shortest_path')
            dry_run: If True, only report what would be deleted

        Returns:
            List of deleted (or would-be-deleted) file paths
        """
        duplicates = self.scan_directory()
        to_delete = []

        for paths in duplicates.values():
            sorted_paths = sorted(paths)

            if keep == "first":
                to_delete.extend(sorted_paths[1:])
            elif keep == "last":
                to_delete.extend(sorted_paths[:-1])
            elif keep == "shortest_path":
                sorted_by_len = sorted(paths, key=lambda p: len(str(p)))
                to_delete.extend(sorted_by_len[1:])

        if not dry_run:
            for path in to_delete:
                try:
                    path.unlink()
                except (IOError, PermissionError) as e:
                    print(f"Error deleting {path}: {e}")

        return to_delete


class CodeDeduplicator:
    """
    Find duplicate code blocks across Python files.

    Uses AST analysis and fuzzy matching to find:
    - Exact duplicate functions
    - Similar code blocks
    - Copy-pasted code with minor modifications
    """

    def __init__(self, source_dir: str = "."):
        self.source_dir = Path(source_dir)
        self.min_lines = 5  # Minimum lines to consider as duplicate
        self.similarity_threshold = 0.85

    def normalize_code(self, code: str) -> str:
        """Normalize code for comparison (remove comments, whitespace)."""
        # Remove comments
        code = re.sub(r'#.*$', '', code, flags=re.MULTILINE)
        # Remove docstrings
        code = re.sub(r'"""[\s\S]*?"""', '', code)
        code = re.sub(r"'''[\s\S]*?'''", '', code)
        # Normalize whitespace
        code = '\n'.join(line.strip() for line in code.split('\n') if line.strip())
        return code

    def extract_functions(self, filepath: Path) -> List[Tuple[str, int, int, str]]:
        """Extract function definitions from file."""
        functions = []
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
        except (IOError, UnicodeDecodeError):
            return functions

        # Simple regex-based function extraction
        pattern = r'^(def\s+\w+.*?)(?=\n(?:def\s|\nclass\s|\Z))'

        for match in re.finditer(pattern, content, re.MULTILINE | re.DOTALL):
            func_code = match.group(1)
            start_line = content[:match.start()].count('\n') + 1
            end_line = start_line + func_code.count('\n')

            # Extract function name
            name_match = re.match(r'def\s+(\w+)', func_code)
            func_name = name_match.group(1) if name_match else "unknown"

            functions.append((func_name, start_line, end_line, func_code))

        return functions

    def find_similar_functions(self, files: List[Path] = None) -> List[DuplicateCode]:
        """Find similar functions across files."""
        if files is None:
            files = list(self.source_dir.rglob("*.py"))

        all_functions = []
        for filepath in files:
            for name, start, end, code in self.extract_functions(filepath):
                normalized = self.normalize_code(code)
                if normalized.count('\n') >= self.min_lines:
                    all_functions.append((filepath, name, start, end, code, normalized))

        duplicates = []
        seen = set()

        for i, (file1, name1, start1, end1, code1, norm1) in enumerate(all_functions):
            for file2, name2, start2, end2, code2, norm2 in all_functions[i+1:]:
                # Skip if same file and overlapping
                if file1 == file2 and not (end1 < start2 or end2 < start1):
                    continue

                # Check similarity
                similarity = difflib.SequenceMatcher(None, norm1, norm2).ratio()

                if similarity >= self.similarity_threshold:
                    key = (str(file1), start1, str(file2), start2)
                    if key not in seen:
                        seen.add(key)
                        duplicates.append(DuplicateCode(
                            file1=file1,
                            file2=file2,
                            line_start1=start1,
                            line_end1=end1,
                            line_start2=start2,
                            line_end2=end2,
                            similarity=similarity,
                            code_snippet=code1[:200] + "..." if len(code1) > 200 else code1
                        ))

        return sorted(duplicates, key=lambda d: -d.similarity)

    def get_report(self) -> str:
        """Generate code duplication report."""
        duplicates = self.find_similar_functions()

        if not duplicates:
            return "No significant code duplication found."

        lines = ["=" * 60, "CODE DUPLICATION REPORT", "=" * 60, ""]

        for i, dup in enumerate(duplicates[:20], 1):  # Top 20
            lines.append(f"{i}. Similarity: {dup.similarity:.1%}")
            lines.append(f"   File 1: {dup.file1}:{dup.line_start1}-{dup.line_end1}")
            lines.append(f"   File 2: {dup.file2}:{dup.line_start2}-{dup.line_end2}")
            lines.append(f"   Snippet: {dup.code_snippet[:100]}...")
            lines.append("")

        if len(duplicates) > 20:
            lines.append(f"... and {len(duplicates) - 20} more duplicates")

        return "\n".join(lines)


def find_duplicates(
    path: str = ".",
    check_data: bool = True,
    check_code: bool = True
) -> Dict[str, any]:
    """
    Convenience function to find all duplicates.

    Args:
        path: Root path to scan
        check_data: Check for duplicate data files
        check_code: Check for duplicate code

    Returns:
        Dict with 'data' and 'code' duplication results
    """
    results = {}

    if check_data:
        data_dir = Path(path) / "data" if (Path(path) / "data").exists() else Path(path)
        dedup = DataDeduplicator(str(data_dir))
        results['data'] = {
            'duplicates': dedup.scan_directory(),
            'report': dedup.get_duplicate_report()
        }

    if check_code:
        code_dedup = CodeDeduplicator(path)
        duplicates = code_dedup.find_similar_functions()
        results['code'] = {
            'duplicates': duplicates,
            'report': code_dedup.get_report()
        }

    return results
