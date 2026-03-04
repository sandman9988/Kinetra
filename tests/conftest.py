"""
PyTest Configuration and Shared Fixtures
========================================

Shared fixtures for all tests in the test suite.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is in path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pytest

# ═══════════════════════════════════════════════════════════════════════════════
# Project Structure Check
# ═══════════════════════════════════════════════════════════════════════════════


def pytest_configure(config):
    """Configure pytest."""
    # Add project root to path if not already there
    project_root = str(Path(__file__).parent.parent)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)


# ═══════════════════════════════════════════════════════════════════════════════
# Session Fixtures
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.fixture(scope="session")
def test_data_dir() -> Path:
    """Return path to test data directory."""
    data_dir = Path(__file__).parent / "data"
    data_dir.mkdir(exist_ok=True)
    return data_dir


@pytest.fixture(scope="session")
def temp_output_dir() -> Path:
    """Return path to temporary output directory for tests."""
    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(exist_ok=True)
    return output_dir


# ═══════════════════════════════════════════════════════════════════════════════
# Cleanup
# ═══════════════════════════════════════════════════════════════════════════════


def pytest_sessionfinish(session, exitstatus):
    """Clean up after test session."""
    # Optional: clean up temp files
    pass
