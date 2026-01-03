#!/usr/bin/env python3
"""
Hyperparameter Optimization (HPO) Runner
=========================================

Command-line interface for running hyperparameter optimization
on Kinetra agents using Optuna.

Usage:
    # Optimize single agent/instrument/timeframe
    python scripts/run_hpo.py --agent ppo --instrument BTCUSD --timeframe H1 --trials 100

    # Optimize multiple configurations (sweep)
    python scripts/run_hpo.py --sweep --agents ppo dqn --instruments BTCUSD EURUSD --timeframes H1 H4

    # Use specific data file
    python scripts/run_hpo.py --agent ppo --data data/BTCUSD_H1.csv --trials 50

    # Distributed optimization with database storage
    python scripts/run_hpo.py --agent ppo --instrument BTCUSD --timeframe H1 \
        --storage sqlite:///hpo_studies.db --n-jobs 4

    # Continue previous study
    python scripts/run_hpo.py --agent ppo --instrument BTCUSD --timeframe H1 \
        --storage sqlite:///hpo_studies.db --study-name ppo_btcusd_h1 --trials 50
"""

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from kinetra.hpo_optimizer import HPOOptimizer, run_hpo_sweep

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Hyperparameter optimization for Kinetra agents",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single optimization
  %(prog)s --agent ppo --instrument BTCUSD --timeframe H1 --trials 100

  # Sweep across multiple configurations
  %(prog)s --sweep --agents ppo dqn linear_q --instruments BTCUSD EURUSD \\
           --timeframes H1 H4 --trials 50

  # Distributed optimization
  %(prog)s --agent ppo --instrument BTCUSD --timeframe H1 \\
           --storage sqlite:///hpo.db --n-jobs 4

  # GPU acceleration
  %(prog)s --agent ppo --instrument BTCUSD --timeframe H1 --use-gpu

Metrics:
  - omega: Omega ratio (default) - asymmetric returns
  - sharpe: Sharpe ratio - risk-adjusted returns
  - z_factor: Z-factor - statistical edge significance
  - energy_pct: % Energy captured - physics alignment
        """,
    )

    # Mode selection
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        "--agent",
        type=str,
        choices=["ppo", "dqn", "linear_q", "a3c", "sac", "td3"],
        help="Agent type for single optimization",
    )
    mode_group.add_argument(
        "--sweep",
        action="store_true",
        help="Run HPO sweep across multiple configurations",
    )

    # Configuration
    parser.add_argument(
        "--instrument",
        type=str,
        help="Instrument symbol (e.g., BTCUSD, EURUSD)",
    )
    parser.add_argument(
        "--timeframe",
        type=str,
        choices=["M15", "M30", "H1", "H4", "D1"],
        help="Timeframe",
    )
    parser.add_argument(
        "--data",
        type=Path,
        help="Path to data CSV file (overrides instrument/timeframe auto-load)",
    )

    # Sweep mode options
    parser.add_argument(
        "--agents",
        nargs="+",
        choices=["ppo", "dqn", "linear_q", "a3c", "sac", "td3"],
        help="Agent types for sweep mode",
    )
    parser.add_argument(
        "--instruments",
        nargs="+",
        help="Instruments for sweep mode (e.g., BTCUSD EURUSD XAUUSD)",
    )
    parser.add_argument(
        "--timeframes",
        nargs="+",
        choices=["M15", "M30", "H1", "H4", "D1"],
        help="Timeframes for sweep mode",
    )

    # Optimization settings
    parser.add_argument(
        "--trials",
        type=int,
        default=100,
        help="Number of optimization trials (default: 100)",
    )
    parser.add_argument(
        "--monte-carlo",
        type=int,
        default=10,
        help="Monte Carlo runs per trial for robustness (default: 10)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        help="Timeout in seconds (optional)",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="omega",
        choices=["omega", "sharpe", "z_factor", "energy_pct"],
        help="Primary metric to optimize (default: omega)",
    )

    # Optuna settings
    parser.add_argument(
        "--study-name",
        type=str,
        help="Study name for persistence (auto-generated if not provided)",
    )
    parser.add_argument(
        "--storage",
        type=str,
        help="Optuna storage URL (e.g., sqlite:///hpo_studies.db)",
    )
    parser.add_argument(
        "--pruner",
        type=str,
        default="median",
        choices=["median", "successive_halving", "none"],
        help="Pruning strategy (default: median)",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="Number of parallel jobs (default: 1, -1 for all cores)",
    )

    # Output settings
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("hpo_results"),
        help="Output directory for results (default: hpo_results/)",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress bar",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress non-critical logging",
    )

    # GPU settings
    parser.add_argument(
        "--use-gpu",
        action="store_true",
        help="Enable GPU acceleration (requires CUDA/ROCm)",
    )

    return parser.parse_args()


def validate_args(args):
    """Validate argument combinations."""
    if args.sweep:
        # Sweep mode requires lists
        if not args.agents:
            raise ValueError("--sweep mode requires --agents")
        if not args.instruments:
            raise ValueError("--sweep mode requires --instruments")
        if not args.timeframes:
            raise ValueError("--sweep mode requires --timeframes")
    else:
        # Single mode requires instrument/timeframe or data file
        if args.data is None:
            if not args.instrument:
                raise ValueError("--agent mode requires --instrument or --data")
            if not args.timeframe:
                raise ValueError("--agent mode requires --timeframe or --data")


def run_single_optimization(args):
    """Run single HPO optimization."""
    logger.info("=" * 80)
    logger.info(f"Single HPO: {args.agent}")
    if args.data:
        logger.info(f"Data: {args.data}")
    else:
        logger.info(f"Instrument: {args.instrument} {args.timeframe}")
    logger.info(f"Trials: {args.trials}, Monte Carlo: {args.monte_carlo}")
    logger.info(f"Metric: {args.metric}")
    logger.info("=" * 80)

    # Create optimizer
    optimizer = HPOOptimizer(
        agent_type=args.agent,
        instrument=args.instrument or "CUSTOM",
        timeframe=args.timeframe or "CUSTOM",
        data_path=args.data,
        n_trials=args.trials,
        n_monte_carlo=args.monte_carlo,
        timeout=args.timeout,
        study_name=args.study_name,
        storage=args.storage,
        pruner=args.pruner,
        n_jobs=args.n_jobs,
        show_progress=not args.no_progress,
        metric=args.metric,
    )

    # Run optimization
    best_params = optimizer.optimize()

    # Save results
    output_subdir = args.output_dir / f"{args.agent}_{args.instrument}_{args.timeframe}"
    optimizer.save_results(output_subdir)

    # Print summary
    logger.info("\n" + "=" * 80)
    logger.info("OPTIMIZATION COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Best {args.metric}: {optimizer.best_value:.4f}")
    logger.info(f"Best parameters:")
    for param, value in best_params.items():
        logger.info(f"  {param:20s} = {value}")

    # Print top 5 parameter sets
    logger.info("\nTop 5 parameter sets:")
    top_5 = optimizer.get_top_n_params(n=5)
    for i, result in enumerate(top_5, 1):
        logger.info(f"\n{i}. Trial {result['trial_number']}: {result['value']:.4f}")
        for param, value in result["params"].items():
            logger.info(f"   {param:20s} = {value}")

    logger.info(f"\nResults saved to: {output_subdir}")
    logger.info("=" * 80 + "\n")

    return best_params


def run_sweep_optimization(args):
    """Run HPO sweep across multiple configurations."""
    logger.info("=" * 80)
    logger.info("HPO SWEEP MODE")
    logger.info(f"Agents: {args.agents}")
    logger.info(f"Instruments: {args.instruments}")
    logger.info(f"Timeframes: {args.timeframes}")
    logger.info(f"Trials per config: {args.trials}")
    total = len(args.agents) * len(args.instruments) * len(args.timeframes)
    logger.info(f"Total configurations: {total}")
    logger.info("=" * 80)

    # Run sweep
    results = run_hpo_sweep(
        agent_types=args.agents,
        instruments=args.instruments,
        timeframes=args.timeframes,
        n_trials=args.trials,
        output_dir=args.output_dir,
    )

    # Print summary
    logger.info("\n" + "=" * 80)
    logger.info("SWEEP COMPLETE")
    logger.info("=" * 80)

    successful = [k for k, v in results.items() if "error" not in v]
    failed = [k for k, v in results.items() if "error" in v]

    logger.info(f"Successful: {len(successful)}/{total}")
    logger.info(f"Failed: {len(failed)}/{total}")

    if successful:
        logger.info("\nBest results:")
        # Sort by best_value
        sorted_results = sorted(
            [(k, v) for k, v in results.items() if "error" not in v],
            key=lambda x: x[1].get("best_value", 0),
            reverse=True,
        )

        for i, (key, result) in enumerate(sorted_results[:10], 1):
            logger.info(f"{i:2d}. {key:30s} = {result.get('best_value', 0):.4f}")

    if failed:
        logger.info("\nFailed configurations:")
        for key in failed:
            logger.info(f"  - {key}: {results[key]['error']}")

    logger.info(f"\nResults saved to: {args.output_dir}")
    logger.info("=" * 80 + "\n")

    return results


def main():
    """Main entry point."""
    args = parse_args()

    # Setup logging
    if args.quiet:
        logging.getLogger("kinetra").setLevel(logging.WARNING)
        logging.getLogger("optuna").setLevel(logging.ERROR)

    # Validate arguments
    try:
        validate_args(args)
    except ValueError as e:
        logger.error(f"Argument validation error: {e}")
        sys.exit(1)

    # GPU setup
    if args.use_gpu:
        try:
            import torch

            if torch.cuda.is_available():
                logger.info(f"GPU acceleration enabled: {torch.cuda.get_device_name(0)}")
            else:
                logger.warning("GPU requested but CUDA not available, using CPU")
        except ImportError:
            logger.warning("GPU requested but PyTorch not installed, using CPU")

    # Run optimization
    try:
        if args.sweep:
            results = run_sweep_optimization(args)
        else:
            results = run_single_optimization(args)

        logger.info("✅ HPO completed successfully!")
        return 0

    except KeyboardInterrupt:
        logger.warning("\n⚠️  HPO interrupted by user")
        return 130

    except Exception as e:
        logger.error(f"\n❌ HPO failed: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
