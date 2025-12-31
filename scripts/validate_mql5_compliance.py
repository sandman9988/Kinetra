#!/usr/bin/env python3
"""
MQL5/MetaAPI Compliance Validator

Validates that all MT5 and MetaAPI integrations comply with standard library
conventions and best practices.

Usage:
    python scripts/validate_mql5_compliance.py
    python scripts/validate_mql5_compliance.py --fix  # Auto-fix issues
"""

import os
import re
import sys
import json
import asyncio
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
from enum import Enum

sys.path.insert(0, str(Path(__file__).parent.parent))


class Severity(Enum):
    """Issue severity levels."""
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


@dataclass
class ComplianceIssue:
    """Represents a compliance issue."""
    file: str
    line: int
    severity: Severity
    code: str
    message: str
    suggestion: str = ""


class MQL5ComplianceValidator:
    """Validates MQL5/MetaAPI compliance."""
    
    # Standard MQL5 timeframe strings
    VALID_TIMEFRAMES = {
        "M1", "M2", "M3", "M4", "M5", "M6", "M10", "M12", "M15", "M20", "M30",
        "H1", "H2", "H3", "H4", "H6", "H8", "H12", "D1", "W1", "MN1"
    }
    
    # MetaAPI timeframe strings
    METAAPI_TIMEFRAMES = {
        "1m", "2m", "3m", "4m", "5m", "6m", "10m", "12m", "15m", "20m", "30m",
        "1h", "2h", "3h", "4h", "6h", "8h", "12h", "1d", "1w", "1mn"
    }
    
    # Standard MQL5 symbol properties (SYMBOL_*)
    MQL5_SYMBOL_PROPERTIES = {
        "SYMBOL_BID", "SYMBOL_ASK", "SYMBOL_LAST", "SYMBOL_VOLUME",
        "SYMBOL_POINT", "SYMBOL_DIGITS", "SYMBOL_SPREAD",
        "SYMBOL_TRADE_CONTRACT_SIZE", "SYMBOL_TRADE_TICK_SIZE",
        "SYMBOL_TRADE_TICK_VALUE", "SYMBOL_TRADE_STOPS_LEVEL",
        "SYMBOL_TRADE_FREEZE_LEVEL", "SYMBOL_SWAP_LONG", "SYMBOL_SWAP_SHORT",
        "SYMBOL_VOLUME_MIN", "SYMBOL_VOLUME_MAX", "SYMBOL_VOLUME_STEP",
        "SYMBOL_MARGIN_INITIAL", "SYMBOL_MARGIN_MAINTENANCE",
    }
    
    # MetaAPI property names (camelCase)
    METAAPI_PROPERTIES = {
        "bid", "ask", "last", "volume", "point", "digits", "spread",
        "contractSize", "tickSize", "tickValue", "stopsLevel", "freezeLevel",
        "swapLong", "swapShort", "minVolume", "maxVolume", "volumeStep",
        "marginInitial", "marginMaintenance", "tradeMode", "pipSize"
    }
    
    # Order types
    MQL5_ORDER_TYPES = {
        "ORDER_TYPE_BUY", "ORDER_TYPE_SELL",
        "ORDER_TYPE_BUY_LIMIT", "ORDER_TYPE_SELL_LIMIT",
        "ORDER_TYPE_BUY_STOP", "ORDER_TYPE_SELL_STOP",
        "ORDER_TYPE_BUY_STOP_LIMIT", "ORDER_TYPE_SELL_STOP_LIMIT"
    }
    
    def __init__(self, workspace: Path = None):
        self.workspace = workspace or Path(__file__).parent.parent
        self.issues: List[ComplianceIssue] = []
    
    def validate_all(self) -> List[ComplianceIssue]:
        """Run all validation checks."""
        self.issues = []
        
        # Validate Python files
        python_files = list(self.workspace.glob("**/*.py"))
        python_files = [f for f in python_files if "venv" not in str(f) and ".git" not in str(f)]
        
        for filepath in python_files:
            self._validate_python_file(filepath)
        
        # Validate MQL5 files
        mql5_files = list(self.workspace.glob("**/*.mq5"))
        for filepath in mql5_files:
            self._validate_mql5_file(filepath)
        
        # Validate config files
        self._validate_config_files()
        
        return self.issues
    
    def _validate_python_file(self, filepath: Path):
        """Validate a Python file for MetaAPI/MT5 compliance."""
        try:
            content = filepath.read_text()
        except Exception:
            return
        
        lines = content.split('\n')
        
        # Skip non-relevant files
        if 'mt5' not in filepath.name.lower() and \
           'metaapi' not in filepath.name.lower() and \
           'MT5' not in content and \
           'MetaApi' not in content:
            return
        
        for i, line in enumerate(lines, 1):
            # Check for deprecated asyncio patterns
            if 'asyncio.get_event_loop()' in line and 'loop.run_until_complete' in content:
                self.issues.append(ComplianceIssue(
                    file=str(filepath.relative_to(self.workspace)),
                    line=i,
                    severity=Severity.WARNING,
                    code="ASYNC001",
                    message="Deprecated asyncio pattern: get_event_loop() with run_until_complete",
                    suggestion="Use asyncio.run() for Python 3.7+"
                ))
            
            # Check for hardcoded timeframe strings that should be constants
            tf_match = re.search(r'["\']([MmHhDdWw][0-9]+|M[Nn]1)["\']', line)
            if tf_match:
                tf = tf_match.group(1).upper()
                if tf not in self.VALID_TIMEFRAMES and tf.lower() not in self.METAAPI_TIMEFRAMES:
                    self.issues.append(ComplianceIssue(
                        file=str(filepath.relative_to(self.workspace)),
                        line=i,
                        severity=Severity.WARNING,
                        code="TF001",
                        message=f"Non-standard timeframe string: {tf_match.group(1)}",
                        suggestion=f"Use standard MQL5 timeframes: {', '.join(sorted(self.VALID_TIMEFRAMES))}"
                    ))
            
            # Check for MetaAPI rate limit handling
            if 'await' in line and ('get_historical' in line or 'get_candles' in line):
                # Check if rate limiting is implemented nearby
                context_start = max(0, i - 10)
                context = '\n'.join(lines[context_start:i+5])
                if 'sleep' not in context and 'semaphore' not in context.lower():
                    self.issues.append(ComplianceIssue(
                        file=str(filepath.relative_to(self.workspace)),
                        line=i,
                        severity=Severity.INFO,
                        code="RATE001",
                        message="MetaAPI historical data call without visible rate limiting",
                        suggestion="Consider adding asyncio.sleep() or semaphore for rate limiting"
                    ))
            
            # Check for proper error handling with MetaAPI
            if 'await' in line and 'metaapi' in line.lower():
                context_start = max(0, i - 5)
                context = '\n'.join(lines[context_start:i+5])
                if 'try' not in context and 'except' not in context:
                    self.issues.append(ComplianceIssue(
                        file=str(filepath.relative_to(self.workspace)),
                        line=i,
                        severity=Severity.INFO,
                        code="ERR001",
                        message="MetaAPI async call without try/except",
                        suggestion="Wrap MetaAPI calls in try/except for robust error handling"
                    ))
            
            # Check for proper volume validation
            if 'volume' in line.lower() and ('=' in line or '<' in line or '>' in line):
                if 'volume_min' not in content.lower() and 'minVolume' not in content:
                    self.issues.append(ComplianceIssue(
                        file=str(filepath.relative_to(self.workspace)),
                        line=i,
                        severity=Severity.INFO,
                        code="VOL001",
                        message="Volume handling without min/max validation",
                        suggestion="Validate volume against SYMBOL_VOLUME_MIN/MAX"
                    ))
            
            # Check for stops level validation in order placement
            if ('sl' in line.lower() or 'tp' in line.lower() or 'stop' in line.lower()):
                if 'trade' in line.lower() or 'order' in line.lower():
                    if 'stops_level' not in content.lower() and 'stopsLevel' not in content:
                        self.issues.append(ComplianceIssue(
                            file=str(filepath.relative_to(self.workspace)),
                            line=i,
                            severity=Severity.WARNING,
                            code="STOP001",
                            message="SL/TP handling without STOPS_LEVEL validation",
                            suggestion="Validate SL/TP distance against SYMBOL_TRADE_STOPS_LEVEL"
                        ))
            
            # Check for proper swap calculation
            if 'swap' in line.lower() and ('*' in line or 'calculate' in line.lower()):
                if 'swap_mode' not in content.lower() and 'swapMode' not in content:
                    self.issues.append(ComplianceIssue(
                        file=str(filepath.relative_to(self.workspace)),
                        line=i,
                        severity=Severity.WARNING,
                        code="SWAP001",
                        message="Swap calculation without SWAP_MODE check",
                        suggestion="Check SYMBOL_SWAP_MODE for proper swap calculation method"
                    ))
            
            # Check for triple swap day handling
            if 'swap' in line.lower() and 'day' in line.lower():
                if 'wednesday' not in content.lower() and 'rollover3day' not in content.lower():
                    self.issues.append(ComplianceIssue(
                        file=str(filepath.relative_to(self.workspace)),
                        line=i,
                        severity=Severity.INFO,
                        code="SWAP002",
                        message="Swap day handling without triple swap day check",
                        suggestion="Handle SYMBOL_SWAP_ROLLOVER3DAYS for weekend swap"
                    ))
    
    def _validate_mql5_file(self, filepath: Path):
        """Validate an MQL5 file for standard library compliance."""
        try:
            content = filepath.read_text()
        except Exception:
            return
        
        lines = content.split('\n')
        
        for i, line in enumerate(lines, 1):
            # Check for #property strict
            if i == 1 and '#property strict' not in content[:500]:
                self.issues.append(ComplianceIssue(
                    file=str(filepath.relative_to(self.workspace)),
                    line=1,
                    severity=Severity.WARNING,
                    code="MQL001",
                    message="Missing #property strict directive",
                    suggestion="Add '#property strict' for better error checking"
                ))
            
            # Check for GetLastError() usage
            if 'FileOpen' in line or 'OrderSend' in line or 'CopyRates' in line:
                context_end = min(len(lines), i + 10)
                context = '\n'.join(lines[i:context_end])
                if 'GetLastError' not in context and 'INVALID_HANDLE' not in context:
                    self.issues.append(ComplianceIssue(
                        file=str(filepath.relative_to(self.workspace)),
                        line=i,
                        severity=Severity.WARNING,
                        code="MQL002",
                        message="MQL5 function without error checking",
                        suggestion="Check GetLastError() or return value after critical operations"
                    ))
            
            # Check for magic number usage
            if 'magic' in line.lower() and '=' in line:
                if re.search(r'magic\s*[=<>!]=?\s*0\b', line.lower()):
                    self.issues.append(ComplianceIssue(
                        file=str(filepath.relative_to(self.workspace)),
                        line=i,
                        severity=Severity.INFO,
                        code="MQL003",
                        message="Magic number set to 0",
                        suggestion="Use a unique non-zero magic number to identify your EA's trades"
                    ))
            
            # Check for ArraySetAsSeries
            if 'CopyRates' in line and 'rates' in line.lower():
                context_start = max(0, i - 10)
                context = '\n'.join(lines[context_start:i])
                if 'ArraySetAsSeries' not in context:
                    self.issues.append(ComplianceIssue(
                        file=str(filepath.relative_to(self.workspace)),
                        line=i,
                        severity=Severity.INFO,
                        code="MQL004",
                        message="CopyRates without ArraySetAsSeries",
                        suggestion="Call ArraySetAsSeries(rates, true) before CopyRates for MT5 compatibility"
                    ))
    
    def _validate_config_files(self):
        """Validate configuration files for proper settings."""
        env_example = self.workspace / ".env.example"
        if env_example.exists():
            content = env_example.read_text()
            
            # Check for MetaAPI token placeholder
            if 'METAAPI_TOKEN' not in content:
                self.issues.append(ComplianceIssue(
                    file=".env.example",
                    line=1,
                    severity=Severity.INFO,
                    code="CFG001",
                    message="Missing METAAPI_TOKEN in .env.example",
                    suggestion="Add METAAPI_TOKEN=your-token-here"
                ))
            
            if 'METAAPI_ACCOUNT_ID' not in content:
                self.issues.append(ComplianceIssue(
                    file=".env.example",
                    line=1,
                    severity=Severity.INFO,
                    code="CFG002",
                    message="Missing METAAPI_ACCOUNT_ID in .env.example",
                    suggestion="Add METAAPI_ACCOUNT_ID=your-account-id"
                ))
    
    def print_report(self):
        """Print compliance report."""
        if not self.issues:
            print("✓ All MQL5/MetaAPI compliance checks passed!")
            return
        
        # Group by severity
        errors = [i for i in self.issues if i.severity == Severity.ERROR]
        warnings = [i for i in self.issues if i.severity == Severity.WARNING]
        infos = [i for i in self.issues if i.severity == Severity.INFO]
        
        print("=" * 70)
        print("MQL5/MetaAPI COMPLIANCE REPORT")
        print("=" * 70)
        print(f"Errors: {len(errors)} | Warnings: {len(warnings)} | Info: {len(infos)}")
        print("-" * 70)
        
        # Print by file
        issues_by_file: Dict[str, List[ComplianceIssue]] = {}
        for issue in self.issues:
            if issue.file not in issues_by_file:
                issues_by_file[issue.file] = []
            issues_by_file[issue.file].append(issue)
        
        for filepath, issues in sorted(issues_by_file.items()):
            print(f"\n{filepath}:")
            for issue in sorted(issues, key=lambda x: x.line):
                icon = "✗" if issue.severity == Severity.ERROR else \
                       "⚠" if issue.severity == Severity.WARNING else "ℹ"
                print(f"  {icon} Line {issue.line}: [{issue.code}] {issue.message}")
                if issue.suggestion:
                    print(f"    → {issue.suggestion}")
        
        print("\n" + "=" * 70)
        print(f"Total: {len(self.issues)} issues found")
        print("=" * 70)
    
    def generate_json_report(self) -> str:
        """Generate JSON report for CI/CD integration."""
        report = {
            "summary": {
                "total": len(self.issues),
                "errors": len([i for i in self.issues if i.severity == Severity.ERROR]),
                "warnings": len([i for i in self.issues if i.severity == Severity.WARNING]),
                "info": len([i for i in self.issues if i.severity == Severity.INFO]),
            },
            "issues": [
                {
                    "file": i.file,
                    "line": i.line,
                    "severity": i.severity.value,
                    "code": i.code,
                    "message": i.message,
                    "suggestion": i.suggestion,
                }
                for i in self.issues
            ]
        }
        return json.dumps(report, indent=2)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Validate MQL5/MetaAPI compliance")
    parser.add_argument('--json', action='store_true', help='Output as JSON')
    parser.add_argument('--strict', action='store_true', help='Fail on warnings')
    args = parser.parse_args()
    
    validator = MQL5ComplianceValidator()
    validator.validate_all()
    
    if args.json:
        print(validator.generate_json_report())
    else:
        validator.print_report()
    
    # Exit code
    errors = len([i for i in validator.issues if i.severity == Severity.ERROR])
    warnings = len([i for i in validator.issues if i.severity == Severity.WARNING])
    
    if errors > 0:
        sys.exit(1)
    elif args.strict and warnings > 0:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
