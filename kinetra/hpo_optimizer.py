"""
Hyperparameter Optimization (HPO) for Kinetra Agents
=====================================================

Integrates Optuna for automated hyperparameter tuning across all agent types.
Uses physics-informed objective function (Omega ratio, Z-factor, energy capture).

Key Features:
- Multi-objective optimization (Omega, Sharpe, drawdown)
- Agent-specific search spaces
- Pruning for early stopping of unpromising trials
- Distributed optimization support
- Integration with BacktestEngine for validation

Usage:
    from kinetra.hpo_optimizer import HPOOptimizer

    optimizer = HPOOptimizer(
        agent_type='ppo',
        instrument='BTCUSD',
        timeframe='H1',
        n_trials=100
    )

    best_params = optimizer.optimize()
    print(f"Best Omega: {optimizer.best_value}")
"""

import logging
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import optuna
import pandas as pd
from optuna.pruners import MedianPruner, SuccessiveHalvingPruner
from optuna.samplers import TPESampler
from optuna.visualization import (
    plot_optimization_history,
    plot_parallel_coordinate,
    plot_param_importances,
)

from kinetra.agent_factory import AgentFactory
from kinetra.backtest_engine import BacktestEngine

# Suppress Optuna logging by default
optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore", category=optuna.exceptions.ExperimentalWarning)

logger = logging.getLogger(__name__)


class HPOOptimizer:
    """
    Hyperparameter optimizer for Kinetra agents using Optuna.

    Optimizes for multi-objective function combining:
    - Omega ratio (asymmetric returns)
    - Z-factor (statistical edge)
    - % Energy captured (physics alignment)
    - Sharpe ratio (risk-adjusted returns)
    - Maximum drawdown (risk control)
    """

    # Agent-specific hyperparameter search spaces
    SEARCH_SPACES = {
        "ppo": {
            "learning_rate": ("log_uniform", 1e-5, 1e-2),
            "gamma": ("uniform", 0.90, 0.999),
            "gae_lambda": ("uniform", 0.90, 0.99),
            "clip_epsilon": ("uniform", 0.1, 0.3),
            "value_loss_coef": ("uniform", 0.3, 0.7),
            "entropy_coef": ("log_uniform", 1e-4, 1e-2),
            "n_epochs": ("int", 3, 15),
            "batch_size": ("categorical", [32, 64, 128, 256]),
            "hidden_dim": ("categorical", [64, 128, 256, 512]),
        },
        "dqn": {
            "learning_rate": ("log_uniform", 1e-5, 1e-2),
            "gamma": ("uniform", 0.90, 0.999),
            "epsilon_start": ("uniform", 0.9, 1.0),
            "epsilon_end": ("uniform", 0.01, 0.1),
            "epsilon_decay": ("log_uniform", 0.99, 0.9999),
            "buffer_size": ("categorical", [10000, 50000, 100000]),
            "batch_size": ("categorical", [32, 64, 128, 256]),
            "hidden_dim": ("categorical", [64, 128, 256, 512]),
            "target_update_freq": ("int", 100, 1000),
        },
        "linear_q": {
            "learning_rate": ("log_uniform", 1e-4, 1e-1),
            "gamma": ("uniform", 0.90, 0.999),
            "epsilon_start": ("uniform", 0.9, 1.0),
            "epsilon_end": ("uniform", 0.01, 0.1),
            "epsilon_decay": ("log_uniform", 0.99, 0.9999),
        },
        "a3c": {
            "learning_rate": ("log_uniform", 1e-5, 1e-2),
            "gamma": ("uniform", 0.90, 0.999),
            "entropy_coef": ("log_uniform", 1e-4, 1e-2),
            "value_loss_coef": ("uniform", 0.3, 0.7),
            "hidden_dim": ("categorical", [64, 128, 256, 512]),
            "n_workers": ("int", 2, 8),
        },
        "sac": {
            "learning_rate": ("log_uniform", 1e-5, 1e-2),
            "gamma": ("uniform", 0.90, 0.999),
            "tau": ("log_uniform", 1e-4, 1e-2),
            "alpha": ("log_uniform", 1e-3, 0.5),
            "buffer_size": ("categorical", [10000, 50000, 100000]),
            "batch_size": ("categorical", [64, 128, 256, 512]),
            "hidden_dim": ("categorical", [128, 256, 512]),
        },
        "td3": {
            "learning_rate": ("log_uniform", 1e-5, 1e-2),
            "gamma": ("uniform", 0.90, 0.999),
            "tau": ("log_uniform", 1e-4, 1e-2),
            "policy_noise": ("uniform", 0.1, 0.3),
            "noise_clip": ("uniform", 0.3, 0.7),
            "policy_delay": ("int", 2, 4),
            "buffer_size": ("categorical", [10000, 50000, 100000]),
            "batch_size": ("categorical", [64, 128, 256, 512]),
            "hidden_dim": ("categorical", [128, 256, 512]),
        },
    }

    def __init__(
        self,
        agent_type: str,
        instrument: str,
        timeframe: str,
        data: Optional[pd.DataFrame] = None,
        data_path: Optional[Path] = None,
        n_trials: int = 100,
        n_monte_carlo: int = 10,
        timeout: Optional[int] = None,
        study_name: Optional[str] = None,
        storage: Optional[str] = None,
        pruner: str = "median",
        n_jobs: int = 1,
        show_progress: bool = True,
        direction: str = "maximize",
        metric: str = "omega",
    ):
        """
        Initialize HPO optimizer.

        Args:
            agent_type: Agent type (ppo, dqn, linear_q, a3c, sac, td3)
            instrument: Instrument symbol (e.g., BTCUSD)
            timeframe: Timeframe (e.g., H1, H4, D1)
            data: Optional pre-loaded DataFrame
            data_path: Optional path to data CSV
            n_trials: Number of optimization trials
            n_monte_carlo: Monte Carlo runs per trial for robustness
            timeout: Optional timeout in seconds
            study_name: Optional study name for persistence
            storage: Optional Optuna storage URL (e.g., sqlite:///optuna.db)
            pruner: Pruning strategy (median, successive_halving, none)
            n_jobs: Number of parallel jobs (-1 for all cores)
            show_progress: Show progress bar
            direction: Optimization direction (maximize or minimize)
            metric: Primary metric to optimize (omega, sharpe, z_factor, energy_pct)
        """
        self.agent_type = agent_type.lower()
        self.instrument = instrument
        self.timeframe = timeframe
        self.n_trials = n_trials
        self.n_monte_carlo = n_monte_carlo
        self.timeout = timeout
        self.n_jobs = n_jobs
        self.show_progress = show_progress
        self.direction = direction
        self.metric = metric

        # Validate agent type
        if self.agent_type not in self.SEARCH_SPACES:
            raise ValueError(
                f"Unknown agent type: {agent_type}. Available: {list(self.SEARCH_SPACES.keys())}"
            )

        # Load data
        if data is not None:
            self.data = data
        elif data_path is not None:
            self.data = pd.read_csv(data_path, parse_dates=["time"])
        else:
            # Try to load from standard location
            std_path = Path(f"data/master_standardized/{instrument}_{timeframe}.csv")
            if std_path.exists():
                self.data = pd.read_csv(std_path, parse_dates=["time"])
            else:
                raise ValueError(
                    f"No data provided and {std_path} not found. "
                    "Provide data or data_path parameter."
                )

        logger.info(f"Loaded {len(self.data)} rows for {instrument} {timeframe}")

        # Setup pruner
        if pruner == "median":
            self.pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=10)
        elif pruner == "successive_halving":
            self.pruner = SuccessiveHalvingPruner()
        else:
            self.pruner = None

        # Create Optuna study
        self.study_name = study_name or f"hpo_{agent_type}_{instrument}_{timeframe}"
        self.study = optuna.create_study(
            study_name=self.study_name,
            storage=storage,
            sampler=TPESampler(seed=42),
            pruner=self.pruner,
            direction=direction,
            load_if_exists=True,
        )

        self.best_params = None
        self.best_value = None
        self.best_trial = None

    def _suggest_hyperparameters(self, trial: optuna.Trial) -> Dict[str, Any]:
        """
        Suggest hyperparameters for the trial based on agent type.

        Args:
            trial: Optuna trial object

        Returns:
            Dictionary of suggested hyperparameters
        """
        params = {}
        search_space = self.SEARCH_SPACES[self.agent_type]

        for param_name, param_config in search_space.items():
            param_type = param_config[0]

            if param_type == "uniform":
                params[param_name] = trial.suggest_float(
                    param_name, param_config[1], param_config[2]
                )
            elif param_type == "log_uniform":
                params[param_name] = trial.suggest_float(
                    param_name, param_config[1], param_config[2], log=True
                )
            elif param_type == "int":
                params[param_name] = trial.suggest_int(param_name, param_config[1], param_config[2])
            elif param_type == "categorical":
                params[param_name] = trial.suggest_categorical(param_name, param_config[1])

        return params

    def _objective(self, trial: optuna.Trial) -> float:
        """
        Objective function for Optuna optimization.

        Runs Monte Carlo backtests with suggested hyperparameters
        and returns the mean objective metric.

        Args:
            trial: Optuna trial object

        Returns:
            Objective value (higher is better for maximize direction)
        """
        # Suggest hyperparameters
        hyperparams = self._suggest_hyperparameters(trial)

        # Run Monte Carlo backtests for robustness
        metrics_list = []

        for mc_run in range(self.n_monte_carlo):
            try:
                # Create agent with suggested hyperparameters
                agent = AgentFactory.create_agent(agent_type=self.agent_type, **hyperparams)

                # Create backtest engine
                engine = BacktestEngine(
                    data=self.data.copy(),
                    initial_balance=10000,
                    risk_per_trade=0.02,
                    seed=42 + mc_run,  # Different seed per MC run
                )

                # Run backtest
                trades, metrics = engine.run(agent=agent, verbose=False)

                if metrics and len(trades) > 10:  # Minimum 10 trades for validity
                    metrics_list.append(metrics)

                # Prune unpromising trials early
                if mc_run >= 3 and len(metrics_list) > 0:
                    # Report intermediate value for pruning
                    intermediate_value = np.mean([m.get(self.metric, 0) for m in metrics_list])
                    trial.report(intermediate_value, mc_run)

                    if trial.should_prune():
                        raise optuna.TrialPruned()

            except Exception as e:
                logger.warning(f"Trial {trial.number} MC run {mc_run} failed: {e}")
                continue

        if not metrics_list:
            # All runs failed - return worst possible value
            return -np.inf if self.direction == "maximize" else np.inf

        # Compute statistics across Monte Carlo runs
        omega_values = [m.get("omega_ratio", 0) for m in metrics_list]
        sharpe_values = [m.get("sharpe_ratio", 0) for m in metrics_list]
        z_values = [m.get("z_factor", 0) for m in metrics_list]
        energy_pct = [m.get("pct_energy_captured", 0) for m in metrics_list]
        drawdown_values = [m.get("max_drawdown", 0) for m in metrics_list]

        # Primary metric
        if self.metric == "omega":
            primary = np.mean(omega_values)
        elif self.metric == "sharpe":
            primary = np.mean(sharpe_values)
        elif self.metric == "z_factor":
            primary = np.mean(z_values)
        elif self.metric == "energy_pct":
            primary = np.mean(energy_pct)
        else:
            primary = np.mean(omega_values)  # Default to Omega

        # Log all metrics for analysis
        trial.set_user_attr("omega_mean", np.mean(omega_values))
        trial.set_user_attr("omega_std", np.std(omega_values))
        trial.set_user_attr("sharpe_mean", np.mean(sharpe_values))
        trial.set_user_attr("z_factor_mean", np.mean(z_values))
        trial.set_user_attr("energy_pct_mean", np.mean(energy_pct))
        trial.set_user_attr("drawdown_mean", np.mean(drawdown_values))
        trial.set_user_attr("n_valid_runs", len(metrics_list))

        # Composite objective with penalty for high variance and drawdown
        variance_penalty = np.std(omega_values) * 0.1
        drawdown_penalty = abs(np.mean(drawdown_values)) * 0.05

        objective_value = primary - variance_penalty - drawdown_penalty

        return objective_value

    def optimize(self) -> Dict[str, Any]:
        """
        Run hyperparameter optimization.

        Returns:
            Dictionary of best hyperparameters found
        """
        logger.info(f"Starting HPO for {self.agent_type} on {self.instrument} {self.timeframe}")
        logger.info(f"Trials: {self.n_trials}, Monte Carlo: {self.n_monte_carlo}")

        # Run optimization
        self.study.optimize(
            self._objective,
            n_trials=self.n_trials,
            timeout=self.timeout,
            n_jobs=self.n_jobs,
            show_progress_bar=self.show_progress,
        )

        # Store best results
        self.best_trial = self.study.best_trial
        self.best_params = self.study.best_params
        self.best_value = self.study.best_value

        logger.info(f"Optimization complete!")
        logger.info(f"Best {self.metric}: {self.best_value:.4f}")
        logger.info(f"Best params: {self.best_params}")

        # Log additional metrics
        if hasattr(self.best_trial, "user_attrs"):
            logger.info(f"  Omega: {self.best_trial.user_attrs.get('omega_mean', 0):.3f}")
            logger.info(f"  Sharpe: {self.best_trial.user_attrs.get('sharpe_mean', 0):.3f}")
            logger.info(f"  Z-factor: {self.best_trial.user_attrs.get('z_factor_mean', 0):.3f}")
            logger.info(f"  Energy %: {self.best_trial.user_attrs.get('energy_pct_mean', 0):.1f}%")

        return self.best_params

    def get_best_agent(self) -> Any:
        """
        Create an agent instance with the best hyperparameters.

        Returns:
            Agent instance configured with optimized hyperparameters
        """
        if self.best_params is None:
            raise ValueError("No optimization run yet. Call optimize() first.")

        return AgentFactory.create_agent(agent_type=self.agent_type, **self.best_params)

    def save_results(self, output_dir: Path):
        """
        Save optimization results and visualizations.

        Args:
            output_dir: Directory to save results
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save best parameters as JSON
        import json

        params_file = output_dir / f"{self.study_name}_best_params.json"
        with open(params_file, "w") as f:
            json.dump(
                {
                    "agent_type": self.agent_type,
                    "instrument": self.instrument,
                    "timeframe": self.timeframe,
                    "best_value": self.best_value,
                    "best_params": self.best_params,
                    "n_trials": len(self.study.trials),
                    "timestamp": datetime.now().isoformat(),
                },
                f,
                indent=2,
            )

        logger.info(f"Saved best params to {params_file}")

        # Save study results as CSV
        df = self.study.trials_dataframe()
        csv_file = output_dir / f"{self.study_name}_trials.csv"
        df.to_csv(csv_file, index=False)
        logger.info(f"Saved trial history to {csv_file}")

        # Generate visualizations
        try:
            # Optimization history
            fig = plot_optimization_history(self.study)
            fig.write_html(output_dir / f"{self.study_name}_history.html")

            # Parameter importances
            fig = plot_param_importances(self.study)
            fig.write_html(output_dir / f"{self.study_name}_importances.html")

            # Parallel coordinate plot
            fig = plot_parallel_coordinate(self.study)
            fig.write_html(output_dir / f"{self.study_name}_parallel.html")

            logger.info(f"Saved visualizations to {output_dir}")

        except Exception as e:
            logger.warning(f"Could not generate visualizations: {e}")

    def get_top_n_params(self, n: int = 5) -> List[Dict[str, Any]]:
        """
        Get top N parameter sets.

        Args:
            n: Number of top parameter sets to return

        Returns:
            List of dictionaries containing top N parameter sets
        """
        sorted_trials = sorted(
            self.study.trials,
            key=lambda t: t.value if t.value is not None else -np.inf,
            reverse=True,
        )

        return [
            {
                "trial_number": trial.number,
                "value": trial.value,
                "params": trial.params,
                "user_attrs": trial.user_attrs,
            }
            for trial in sorted_trials[:n]
        ]


def run_hpo_sweep(
    agent_types: List[str],
    instruments: List[str],
    timeframes: List[str],
    n_trials: int = 50,
    output_dir: Path = Path("hpo_results"),
) -> Dict[str, Dict[str, Any]]:
    """
    Run HPO sweep across multiple agents, instruments, and timeframes.

    Args:
        agent_types: List of agent types to optimize
        instruments: List of instruments
        timeframes: List of timeframes
        n_trials: Trials per configuration
        output_dir: Output directory for results

    Returns:
        Dictionary mapping (agent, instrument, timeframe) -> best params
    """
    results = {}
    total = len(agent_types) * len(instruments) * len(timeframes)
    current = 0

    for agent_type in agent_types:
        for instrument in instruments:
            for timeframe in timeframes:
                current += 1
                logger.info(f"\n{'=' * 80}")
                logger.info(f"HPO {current}/{total}: {agent_type} - {instrument} {timeframe}")
                logger.info(f"{'=' * 80}\n")

                key = f"{agent_type}_{instrument}_{timeframe}"

                try:
                    optimizer = HPOOptimizer(
                        agent_type=agent_type,
                        instrument=instrument,
                        timeframe=timeframe,
                        n_trials=n_trials,
                        n_monte_carlo=5,  # Reduced for sweep
                    )

                    best_params = optimizer.optimize()
                    optimizer.save_results(output_dir / key)

                    results[key] = {
                        "agent_type": agent_type,
                        "instrument": instrument,
                        "timeframe": timeframe,
                        "best_params": best_params,
                        "best_value": optimizer.best_value,
                    }

                except Exception as e:
                    logger.error(f"HPO failed for {key}: {e}")
                    results[key] = {"error": str(e)}

    # Save consolidated results
    import json

    summary_file = output_dir / "hpo_sweep_summary.json"
    with open(summary_file, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"\n{'=' * 80}")
    logger.info(f"HPO Sweep Complete! Results: {summary_file}")
    logger.info(f"{'=' * 80}\n")

    return results
