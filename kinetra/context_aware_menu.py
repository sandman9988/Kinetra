#!/usr/bin/env python3
"""
Context-Aware Menu Wrapper
===========================

Adds context awareness to menu options based on:
- Data availability
- Dependencies installed
- Previous workflow state
- System capabilities

This wraps around menu functions to provide intelligent filtering
and helpful messages.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from kinetra.config import PROJECT_ROOT


@dataclass
class MenuContext:
    """Context information for menu system."""

    data_prepared: bool = False
    data_available: bool = False
    mt5_installed: bool = False
    credentials_configured: bool = False
    gpu_available: bool = False
    has_trained_models: bool = False
    last_exploration_results: Optional[Path] = None
    last_backtest_results: Optional[Path] = None

    def __str__(self) -> str:
        """Get status summary."""
        parts = []

        if self.data_prepared:
            parts.append("✅ Data ready")
        elif self.data_available:
            parts.append("⚠️  Data needs preparation")
        else:
            parts.append("❌ No data")

        if self.mt5_installed:
            parts.append("✅ MT5")
        else:
            parts.append("⚠️  No MT5")

        if self.credentials_configured:
            parts.append("✅ Credentials")
        else:
            parts.append("⚠️  No credentials")

        if self.gpu_available:
            parts.append("✅ GPU")

        return " | ".join(parts)


def check_context() -> MenuContext:
    """
    Check current system context.

    Returns:
        MenuContext with current state

    NOTE — Renko-only architecture:
        The ``data_prepared`` flag is intentionally always False in the Renko
        pipeline.  The old "Prepare Data" step (physics feature extraction into
        data/prepared_standardized/) has no equivalent in the Renko pipeline.
        Renko operates directly on raw M1 → M30 aggregated OHLCV data; there is
        no separate feature-preparation stage.

        ``data/prepared_standardized/`` contains stale physics-era H4 artefacts
        from the pre-Renko architecture.  Reading those files here and returning
        ``data_prepared=True`` would confuse any context-aware gate that still
        uses this helper.  The canonical Renko pipeline gates live in
        ``kinetra_menu.check_system_status()`` — use ``SystemStatus`` instead.
    """
    ctx = MenuContext()

    # Check data — M1 presence in master_standardized/ is the Renko prerequisite.
    # data_prepared is always False: the Renko pipeline has no prepare-data step.
    # (data/prepared_standardized/ contains stale physics-era H4 files and must
    # not be treated as a valid pipeline artefact.)
    ctx.data_prepared = False

    master_dir = PROJECT_ROOT / "data" / "master_standardized"
    ctx.data_available = master_dir.exists() and len(list(master_dir.rglob("*.csv"))) > 0

    # Check MT5
    try:
        import MetaTrader5  # noqa: F401

        ctx.mt5_installed = True
    except ImportError:
        ctx.mt5_installed = False

    # Check credentials
    ctx.credentials_configured = (PROJECT_ROOT / ".env").exists()

    # Check GPU
    try:
        import torch

        ctx.gpu_available = torch.cuda.is_available() or torch.backends.mps.is_available()
    except ImportError:
        ctx.gpu_available = False

    # Check for previous results
    results_dir = PROJECT_ROOT / "results"
    if results_dir.exists():
        exploration_results = sorted(results_dir.glob("comprehensive_exploration_*.json"))
        if exploration_results:
            ctx.last_exploration_results = exploration_results[-1]

        backtest_results = sorted(results_dir.glob("backtest_*.json"))
        if backtest_results:
            ctx.last_backtest_results = backtest_results[-1]

    # Check for trained models
    models_dir = PROJECT_ROOT / "models"
    ctx.has_trained_models = models_dir.exists() and len(list(models_dir.rglob("*.pkl"))) > 0

    return ctx


def get_available_options(menu_type: str, context: MenuContext) -> List[Dict[str, any]]:
    """
    Get available menu options based on context.

    Args:
        menu_type: Type of menu (exploration, backtesting, live, etc.)
        context: Current system context

    Returns:
        List of available options with metadata
    """
    options = []

    if menu_type == "exploration":
        # Quick exploration always available if data exists
        if context.data_prepared:
            options.append(
                {"id": "1", "title": "Quick Exploration", "available": True, "reason": None}
            )
        else:
            options.append(
                {
                    "id": "1",
                    "title": "Quick Exploration",
                    "available": False,
                    "reason": "Data not prepared. Go to Data Management → Prepare Data",
                }
            )

        # Custom exploration
        if context.data_prepared:
            options.append(
                {"id": "2", "title": "Custom Exploration", "available": True, "reason": None}
            )
        else:
            options.append(
                {
                    "id": "2",
                    "title": "Custom Exploration",
                    "available": False,
                    "reason": "Data not prepared",
                }
            )

        # Scientific discovery
        options.append(
            {
                "id": "3",
                "title": "Scientific Discovery Suite",
                "available": True,  # Can run with synthetic data
                "reason": None,
            }
        )

        # Agent comparison
        if context.data_prepared:
            options.append(
                {"id": "4", "title": "Agent Comparison", "available": True, "reason": None}
            )
        else:
            options.append(
                {
                    "id": "4",
                    "title": "Agent Comparison",
                    "available": False,
                    "reason": "Data not prepared",
                }
            )

    elif menu_type == "backtesting":
        # Quick backtest
        if context.last_exploration_results:
            options.append(
                {
                    "id": "1",
                    "title": "Quick Backtest (Use exploration results)",
                    "available": True,
                    "reason": None,
                }
            )
        else:
            options.append(
                {
                    "id": "1",
                    "title": "Quick Backtest",
                    "available": False,
                    "reason": "No exploration results found. Run exploration first.",
                }
            )

        # Custom backtest
        if context.data_prepared:
            options.append(
                {"id": "2", "title": "Custom Backtesting", "available": True, "reason": None}
            )
        else:
            options.append(
                {
                    "id": "2",
                    "title": "Custom Backtesting",
                    "available": False,
                    "reason": "Data not prepared",
                }
            )

        # Monte Carlo
        if context.data_prepared:
            options.append(
                {
                    "id": "3",
                    "title": "Monte Carlo Validation (100 runs)",
                    "available": True,
                    "reason": None,
                }
            )
        else:
            options.append(
                {
                    "id": "3",
                    "title": "Monte Carlo Validation",
                    "available": False,
                    "reason": "Data not prepared",
                }
            )

    elif menu_type == "live":
        # Virtual trading always available
        options.append(
            {
                "id": "1",
                "title": "Virtual Trading (Paper Trading)",
                "available": True,
                "reason": None,
            }
        )

        # Demo account
        if context.mt5_installed:
            options.append(
                {"id": "2", "title": "Demo Account Testing", "available": True, "reason": None}
            )
        else:
            options.append(
                {
                    "id": "2",
                    "title": "Demo Account Testing",
                    "available": False,
                    "reason": "MT5 not installed. Install MetaTrader5: pip install MetaTrader5",
                }
            )

        # MT5 connection test
        options.append(
            {
                "id": "3",
                "title": "Test MT5 Connection",
                "available": context.mt5_installed,
                "reason": None if context.mt5_installed else "MT5 not installed",
            }
        )

    return options


def print_context_aware_menu(title: str, menu_type: str, context: MenuContext):
    """
    Print menu with context-aware availability.

    Args:
        title: Menu title
        menu_type: Type of menu
        context: System context
    """
    print(f"\n{'=' * 80}")
    print(f"  {title}")
    print(f"{'=' * 80}\n")

    print(f"📊 Context: {context}\n")

    options = get_available_options(menu_type, context)

    print("Available options:")
    for opt in options:
        status = "✅" if opt["available"] else "❌"
        print(f"  {opt['id']}. {status} {opt['title']}")
        if not opt["available"] and opt["reason"]:
            print(f"     ↳ {opt['reason']}")

    print("  0. Back to Main Menu")
    print()


if __name__ == "__main__":
    # Demo
    ctx = check_context()
    print("\nCurrent System Context:")
    print(f"  {ctx}")
    print()

    print("\n" + "=" * 80)
    print("  CONTEXT-AWARE MENU DEMO")
    print("=" * 80)

    print_context_aware_menu("EXPLORATION TESTING", "exploration", ctx)
    print_context_aware_menu("BACKTESTING", "backtesting", ctx)
    print_context_aware_menu("LIVE TESTING", "live", ctx)
