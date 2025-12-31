"""
Security Scanning and Best Practices
=====================================

Scan codebase for:
- Hardcoded secrets/credentials
- Security vulnerabilities
- Dependency issues
- Best practice violations
"""

import ast
import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, List


class Severity(Enum):
    """Security issue severity levels."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


@dataclass
class SecurityIssue:
    """Represents a security issue found in code."""
    severity: Severity
    category: str
    message: str
    file: str
    line: int
    code_snippet: str
    recommendation: str


# Patterns for secret detection
SECRET_PATTERNS = [
    # API Keys
    (r'(?i)(api[_-]?key|apikey)\s*[=:]\s*["\']([a-zA-Z0-9_\-]{20,})["\']', "API Key", Severity.CRITICAL),
    (r'(?i)(secret[_-]?key|secretkey)\s*[=:]\s*["\']([a-zA-Z0-9_\-]{20,})["\']', "Secret Key", Severity.CRITICAL),

    # AWS
    (r'AKIA[0-9A-Z]{16}', "AWS Access Key ID", Severity.CRITICAL),
    (r'(?i)(aws[_-]?secret|aws_secret_access_key)\s*[=:]\s*["\']([a-zA-Z0-9+/=]{40})["\']', "AWS Secret Key", Severity.CRITICAL),

    # Database URLs
    (r'(?i)(postgres|mysql|mongodb|redis)://[^:\s]+:[^@\s]+@', "Database URL with Password", Severity.CRITICAL),

    # Private Keys
    (r'-----BEGIN (RSA |EC |OPENSSH )?PRIVATE KEY-----', "Private Key", Severity.CRITICAL),

    # Generic passwords
    (r'(?i)(password|passwd|pwd)\s*[=:]\s*["\']([^"\']{8,})["\']', "Hardcoded Password", Severity.HIGH),

    # Tokens
    (r'(?i)(token|bearer|jwt)\s*[=:]\s*["\']([a-zA-Z0-9_\-\.]{20,})["\']', "Token", Severity.HIGH),

    # OAuth
    (r'(?i)(client[_-]?secret|oauth[_-]?secret)\s*[=:]\s*["\']([a-zA-Z0-9_\-]{20,})["\']', "OAuth Secret", Severity.HIGH),
]

# Insecure code patterns
INSECURE_PATTERNS = [
    # Pickle (unsafe deserialization)
    (r'pickle\.loads?\(', "Pickle Deserialization", Severity.HIGH,
     "Pickle can execute arbitrary code. Use JSON or safer alternatives."),

    # Eval/exec
    (r'\beval\s*\(', "Use of eval()", Severity.HIGH,
     "eval() can execute arbitrary code. Use ast.literal_eval() for data parsing."),
    (r'\bexec\s*\(', "Use of exec()", Severity.HIGH,
     "exec() can execute arbitrary code. Avoid dynamic code execution."),

    # Shell injection
    (r'os\.system\s*\(', "os.system() usage", Severity.MEDIUM,
     "Use subprocess.run() with shell=False for safer command execution."),
    (r'subprocess\.[^(]+\([^)]*shell\s*=\s*True', "Shell=True in subprocess", Severity.MEDIUM,
     "shell=True can lead to shell injection. Use shell=False when possible."),

    # SQL Injection
    (r'cursor\.execute\s*\([^)]*%s[^)]*%\s*\(', "Potential SQL Injection", Severity.HIGH,
     "Use parameterized queries with placeholders."),
    (r'cursor\.execute\s*\([^)]*\.format\s*\(', "Potential SQL Injection", Severity.HIGH,
     "Use parameterized queries instead of string formatting."),

    # Weak crypto
    (r'hashlib\.(md5|sha1)\(', "Weak Hash Algorithm", Severity.MEDIUM,
     "Use SHA-256 or stronger for security-sensitive hashing."),
    (r'random\.(random|randint|choice)\(', "Insecure Random", Severity.LOW,
     "Use secrets module for security-sensitive random values."),

    # Debug mode
    (r'(?i)(debug|DEBUG)\s*=\s*True', "Debug Mode Enabled", Severity.LOW,
     "Ensure debug mode is disabled in production."),

    # Hardcoded IPs
    (r'(?<!\d)(?:25[0-5]|2[0-4]\d|[01]?\d?\d)(?:\.(?:25[0-5]|2[0-4]\d|[01]?\d?\d)){3}(?!\d)',
     "Hardcoded IP Address", Severity.INFO,
     "Consider using configuration for IP addresses."),
]


class SecurityScanner:
    """
    Scan codebase for security issues.

    Usage:
        scanner = SecurityScanner(".")
        issues = scanner.scan()
        print(scanner.get_report())
    """

    def __init__(self, source_dir: str = "."):
        self.source_dir = Path(source_dir)
        self.issues: List[SecurityIssue] = []
        self.files_scanned = 0
        self.excluded_dirs = {'.git', '__pycache__', 'node_modules', '.venv', 'venv', 'env'}
        self.excluded_files = {'.pyc', '.pyo', '.so', '.dll', '.exe'}

    def scan(self, file_patterns: List[str] = None) -> List[SecurityIssue]:
        """
        Scan codebase for security issues.

        Args:
            file_patterns: Patterns to scan (default: ['*.py'])

        Returns:
            List of SecurityIssue objects
        """
        if file_patterns is None:
            file_patterns = ['*.py', '*.yaml', '*.yml', '*.json', '*.env', '*.ini', '*.cfg']

        self.issues = []
        self.files_scanned = 0

        for pattern in file_patterns:
            for filepath in self.source_dir.rglob(pattern):
                if self._should_skip(filepath):
                    continue

                self.files_scanned += 1
                self._scan_file(filepath)

        # Sort by severity
        severity_order = {
            Severity.CRITICAL: 0,
            Severity.HIGH: 1,
            Severity.MEDIUM: 2,
            Severity.LOW: 3,
            Severity.INFO: 4
        }
        self.issues.sort(key=lambda i: severity_order[i.severity])

        return self.issues

    def _should_skip(self, filepath: Path) -> bool:
        """Check if file should be skipped."""
        # Skip excluded directories
        for part in filepath.parts:
            if part in self.excluded_dirs:
                return True

        # Skip excluded file types
        if filepath.suffix in self.excluded_files:
            return True

        return False

    def _scan_file(self, filepath: Path):
        """Scan a single file for security issues."""
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                lines = content.split('\n')
        except (IOError, PermissionError):
            return

        # Scan for secrets
        for pattern, category, severity in SECRET_PATTERNS:
            for match in re.finditer(pattern, content):
                line_num = content[:match.start()].count('\n') + 1
                self.issues.append(SecurityIssue(
                    severity=severity,
                    category="Secret Detection",
                    message=f"Possible {category} found",
                    file=str(filepath),
                    line=line_num,
                    code_snippet=self._get_line(lines, line_num),
                    recommendation=f"Remove {category.lower()} and use environment variables or secret manager"
                ))

        # Scan for insecure patterns
        for pattern, category, severity, recommendation in INSECURE_PATTERNS:
            for match in re.finditer(pattern, content):
                line_num = content[:match.start()].count('\n') + 1
                self.issues.append(SecurityIssue(
                    severity=severity,
                    category="Insecure Code",
                    message=category,
                    file=str(filepath),
                    line=line_num,
                    code_snippet=self._get_line(lines, line_num),
                    recommendation=recommendation
                ))

        # Python-specific checks
        if filepath.suffix == '.py':
            self._scan_python_file(filepath, content, lines)

    def _scan_python_file(self, filepath: Path, content: str, lines: List[str]):
        """Additional Python-specific security checks."""
        try:
            tree = ast.parse(content)
        except SyntaxError:
            return

        for node in ast.walk(tree):
            # Check for dangerous imports
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name in ('pickle', 'cPickle'):
                        self.issues.append(SecurityIssue(
                            severity=Severity.MEDIUM,
                            category="Dangerous Import",
                            message=f"Import of {alias.name} module",
                            file=str(filepath),
                            line=node.lineno,
                            code_snippet=self._get_line(lines, node.lineno),
                            recommendation="Be cautious with pickle - it can deserialize malicious payloads"
                        ))

            # Check for assert statements (not executed with -O flag)
            if isinstance(node, ast.Assert):
                # Only flag if it looks like security validation
                assert_code = self._get_line(lines, node.lineno)
                if any(word in assert_code.lower() for word in ['auth', 'permission', 'access', 'valid']):
                    self.issues.append(SecurityIssue(
                        severity=Severity.MEDIUM,
                        category="Security Validation",
                        message="Assert used for security check",
                        file=str(filepath),
                        line=node.lineno,
                        code_snippet=assert_code,
                        recommendation="Use explicit if/raise for security checks (asserts are disabled with -O)"
                    ))

    def _get_line(self, lines: List[str], line_num: int) -> str:
        """Get a line from the file (1-indexed)."""
        if 0 < line_num <= len(lines):
            return lines[line_num - 1].strip()[:100]
        return ""

    def check_dependencies(self) -> List[SecurityIssue]:
        """Check for known vulnerable dependencies."""
        issues = []

        # Check requirements.txt
        req_file = self.source_dir / "requirements.txt"
        if req_file.exists():
            try:
                with open(req_file) as f:
                    for line_num, line in enumerate(f, 1):
                        line = line.strip()
                        if line and not line.startswith('#'):
                            # Check for unpinned versions
                            if '==' not in line and '>=' not in line:
                                issues.append(SecurityIssue(
                                    severity=Severity.LOW,
                                    category="Dependency",
                                    message="Unpinned dependency version",
                                    file="requirements.txt",
                                    line=line_num,
                                    code_snippet=line,
                                    recommendation="Pin dependency versions for reproducibility"
                                ))
            except IOError:
                pass

        return issues

    def get_report(self) -> str:
        """Generate security scan report."""
        lines = ["=" * 70, "SECURITY SCAN REPORT", "=" * 70]
        lines.append(f"Scanned files: {self.files_scanned}")
        lines.append(f"Issues found: {len(self.issues)}")
        lines.append("")

        # Summary by severity
        severity_counts = {}
        for issue in self.issues:
            severity_counts[issue.severity] = severity_counts.get(issue.severity, 0) + 1

        lines.append("Summary by Severity:")
        for severity in Severity:
            count = severity_counts.get(severity, 0)
            if count > 0:
                emoji = {"critical": "🔴", "high": "🟠", "medium": "🟡", "low": "🔵", "info": "⚪"}
                lines.append(f"  {emoji.get(severity.value, '')} {severity.value.upper()}: {count}")

        lines.append("")
        lines.append("-" * 70)

        # Detailed issues
        for i, issue in enumerate(self.issues, 1):
            lines.append(f"\n{i}. [{issue.severity.value.upper()}] {issue.message}")
            lines.append(f"   Category: {issue.category}")
            lines.append(f"   Location: {issue.file}:{issue.line}")
            lines.append(f"   Code: {issue.code_snippet}")
            lines.append(f"   Fix: {issue.recommendation}")

        if not self.issues:
            lines.append("\n✅ No security issues found!")

        lines.append("\n" + "=" * 70)

        return "\n".join(lines)


def scan_codebase(path: str = ".") -> str:
    """Convenience function to scan codebase and return report."""
    scanner = SecurityScanner(path)
    scanner.scan()
    scanner.issues.extend(scanner.check_dependencies())
    return scanner.get_report()


def check_secrets(path: str = ".") -> List[SecurityIssue]:
    """Check only for hardcoded secrets."""
    scanner = SecurityScanner(path)
    scanner.scan()
    return [i for i in scanner.issues if i.category == "Secret Detection"]


def quick_security_check() -> Dict[str, any]:
    """Quick security check returning structured results."""
    scanner = SecurityScanner(".")
    issues = scanner.scan()
    issues.extend(scanner.check_dependencies())

    return {
        "files_scanned": scanner.files_scanned,
        "total_issues": len(issues),
        "critical": len([i for i in issues if i.severity == Severity.CRITICAL]),
        "high": len([i for i in issues if i.severity == Severity.HIGH]),
        "medium": len([i for i in issues if i.severity == Severity.MEDIUM]),
        "low": len([i for i in issues if i.severity == Severity.LOW]),
        "info": len([i for i in issues if i.severity == Severity.INFO]),
        "issues": [
            {
                "severity": i.severity.value,
                "message": i.message,
                "file": i.file,
                "line": i.line,
            }
            for i in issues[:20]  # Top 20
        ]
    }
