"""
Pytest configuration for Kinetra testing.

Adds command-line options for test data selection.
"""


def pytest_addoption(parser):
    """Add command-line options for test configuration."""
    parser.addoption(
        "--symbol",
        action="store",
        default=None,
        help="Symbol to test (e.g., EURUSD, BTCUSD). If not specified, uses any available data."
    )
    parser.addoption(
        "--timeframe",
        action="store",
        default="H1",
        help="Timeframe to test (default: H1)"
    )
