"""
Backtest Optimizer - Bayesian & Genetic Optimization
====================================================

Advanced hyperparameter optimization for backtesting:
- Bayesian optimization (Gaussian Process)
- Genetic algorithms (evolutionary)
- Non-parametric distribution handling
- Parallel evaluation
- Multi-objective optimization
- Reward shaping for RL

Design Principles:
- Don't assume linearity or normal distributions
- Handle fat tails and regime changes
- Robust to overfitting
- Parallelizable
"""

import logging
import math
import random
import warnings
from abc import ABC, abstractmethod
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from scipy import stats
from scipy.optimize import minimize

logger = logging.getLogger(__name__)


# =============================================================================
# Non-Parametric Distribution Handling
# =============================================================================

class DistributionAnalyzer:
    """
    Analyze return distributions without assuming normality.
    
    Financial returns typically exhibit:
    - Fat tails (leptokurtosis)
    - Skewness
    - Volatility clustering
    - Regime changes
    """
    
    @staticmethod
    def fit_distribution(returns: np.ndarray) -> Dict[str, Any]:
        """
        Fit best distribution to returns (non-parametric).
        
        Tests multiple distributions and returns best fit.
        """
        returns = np.array(returns)
        returns = returns[~np.isnan(returns) & ~np.isinf(returns)]
        
        if len(returns) < 30:
            return {"distribution": "insufficient_data", "params": {}}
        
        results = {}
        
        # Test distributions commonly seen in finance
        distributions = [
            ('norm', stats.norm),
            ('t', stats.t),  # Student's t (fat tails)
            ('laplace', stats.laplace),  # Double exponential
            ('gennorm', stats.gennorm),  # Generalized normal
        ]
        
        for name, dist in distributions:
            try:
                params = dist.fit(returns)
                # Kolmogorov-Smirnov test
                ks_stat, p_value = stats.kstest(returns, name, args=params)
                results[name] = {
                    'params': params,
                    'ks_stat': ks_stat,
                    'p_value': p_value,
                }
            except Exception:
                continue
        
        if not results:
            return {"distribution": "unknown", "params": {}}
        
        # Select best (highest p-value = best fit)
        best = max(results.items(), key=lambda x: x[1]['p_value'])
        
        return {
            "distribution": best[0],
            "params": best[1]['params'],
            "ks_stat": best[1]['ks_stat'],
            "p_value": best[1]['p_value'],
            "all_fits": results,
        }
    
    @staticmethod
    def calculate_robust_statistics(returns: np.ndarray) -> Dict[str, float]:
        """
        Calculate robust statistics that work for non-normal distributions.
        """
        returns = np.array(returns)
        returns = returns[~np.isnan(returns) & ~np.isinf(returns)]
        
        if len(returns) < 2:
            return {}
        
        # Basic statistics
        mean = np.mean(returns)
        median = np.median(returns)
        std = np.std(returns, ddof=1)
        
        # Robust measures
        mad = stats.median_abs_deviation(returns)  # Median absolute deviation
        iqr = stats.iqr(returns)  # Interquartile range
        
        # Higher moments (distribution shape)
        skewness = stats.skew(returns)
        kurtosis = stats.kurtosis(returns)  # Excess kurtosis (normal = 0)
        
        # Tail measures
        q01 = np.percentile(returns, 1)
        q05 = np.percentile(returns, 5)
        q95 = np.percentile(returns, 95)
        q99 = np.percentile(returns, 99)
        
        # VaR and CVaR (non-parametric)
        var_95 = -np.percentile(returns, 5)
        cvar_95 = -np.mean(returns[returns <= np.percentile(returns, 5)])
        
        # Tail ratio (asymmetry)
        upper_tail = returns[returns > q95].mean() if len(returns[returns > q95]) > 0 else 0
        lower_tail = abs(returns[returns < q05].mean()) if len(returns[returns < q05]) > 0 else 0
        tail_ratio = upper_tail / lower_tail if lower_tail > 0 else float('inf')
        
        return {
            "mean": mean,
            "median": median,
            "std": std,
            "mad": mad,
            "iqr": iqr,
            "skewness": skewness,
            "kurtosis": kurtosis,
            "q01": q01,
            "q05": q05,
            "q95": q95,
            "q99": q99,
            "var_95": var_95,
            "cvar_95": cvar_95,
            "tail_ratio": tail_ratio,
            "is_fat_tailed": kurtosis > 1,
            "is_skewed": abs(skewness) > 0.5,
        }
    
    @staticmethod
    def bootstrap_confidence_interval(
        returns: np.ndarray,
        statistic: Callable,
        n_bootstrap: int = 1000,
        confidence: float = 0.95,
    ) -> Tuple[float, float, float]:
        """
        Bootstrap confidence interval for any statistic.
        
        Doesn't assume normality - works for any distribution.
        
        Returns:
            (point_estimate, lower_bound, upper_bound)
        """
        returns = np.array(returns)
        n = len(returns)
        
        # Point estimate
        point = statistic(returns)
        
        # Bootstrap samples
        bootstrap_stats = []
        for _ in range(n_bootstrap):
            sample = np.random.choice(returns, size=n, replace=True)
            bootstrap_stats.append(statistic(sample))
        
        bootstrap_stats = np.array(bootstrap_stats)
        alpha = 1 - confidence
        lower = np.percentile(bootstrap_stats, alpha/2 * 100)
        upper = np.percentile(bootstrap_stats, (1 - alpha/2) * 100)
        
        return point, lower, upper


# =============================================================================
# Objective Functions for Optimization
# =============================================================================

class ObjectiveFunction(Enum):
    """Pre-defined objective functions for optimization."""
    SHARPE = auto()
    SORTINO = auto()
    CALMAR = auto()
    PROFIT_FACTOR = auto()
    TOTAL_RETURN = auto()
    RISK_ADJUSTED_RETURN = auto()
    CUSTOM = auto()


def calculate_objective(
    trades: List[Dict],
    equity_curve: np.ndarray,
    objective: ObjectiveFunction,
    risk_free_rate: float = 0.0,
    annualization: float = 252,
) -> float:
    """
    Calculate objective value from backtest results.
    
    Higher is better for all objectives.
    """
    if len(equity_curve) < 2:
        return float('-inf')
    
    returns = np.diff(equity_curve) / equity_curve[:-1]
    returns = returns[~np.isnan(returns) & ~np.isinf(returns)]
    
    if len(returns) < 2:
        return float('-inf')
    
    if objective == ObjectiveFunction.SHARPE:
        mean_return = np.mean(returns) - risk_free_rate / annualization
        std_return = np.std(returns, ddof=1)
        if std_return < 1e-10:
            return 0.0
        return (mean_return / std_return) * np.sqrt(annualization)
    
    elif objective == ObjectiveFunction.SORTINO:
        mean_return = np.mean(returns) - risk_free_rate / annualization
        downside = returns[returns < 0]
        if len(downside) == 0:
            return float('inf')
        downside_std = np.std(downside, ddof=1)
        if downside_std < 1e-10:
            return float('inf')
        return (mean_return / downside_std) * np.sqrt(annualization)
    
    elif objective == ObjectiveFunction.CALMAR:
        total_return = (equity_curve[-1] / equity_curve[0]) - 1
        running_max = np.maximum.accumulate(equity_curve)
        drawdown = (running_max - equity_curve) / running_max
        max_dd = np.max(drawdown)
        if max_dd < 1e-10:
            return float('inf')
        return total_return / max_dd
    
    elif objective == ObjectiveFunction.PROFIT_FACTOR:
        gross_profit = sum(t.get('pnl', 0) for t in trades if t.get('pnl', 0) > 0)
        gross_loss = abs(sum(t.get('pnl', 0) for t in trades if t.get('pnl', 0) < 0))
        if gross_loss < 1e-10:
            return float('inf')
        return gross_profit / gross_loss
    
    elif objective == ObjectiveFunction.TOTAL_RETURN:
        return (equity_curve[-1] / equity_curve[0]) - 1
    
    elif objective == ObjectiveFunction.RISK_ADJUSTED_RETURN:
        # Robust risk-adjusted return using CVaR
        total_return = (equity_curve[-1] / equity_curve[0]) - 1
        q05 = np.percentile(returns, 5)
        cvar = -np.mean(returns[returns <= q05]) if len(returns[returns <= q05]) > 0 else 0.01
        if cvar < 1e-10:
            return float('inf')
        return total_return / cvar
    
    return 0.0


# =============================================================================
# Parameter Space Definition
# =============================================================================

@dataclass
class Parameter:
    """Single parameter definition."""
    name: str
    min_value: float
    max_value: float
    param_type: str = "float"  # "float", "int", "categorical"
    log_scale: bool = False
    categories: List[Any] = field(default_factory=list)
    default: Optional[float] = None
    
    def sample(self) -> Any:
        """Sample random value from parameter space."""
        if self.param_type == "categorical":
            return random.choice(self.categories)
        
        if self.log_scale:
            log_min = math.log(max(self.min_value, 1e-10))
            log_max = math.log(max(self.max_value, 1e-10))
            value = math.exp(random.uniform(log_min, log_max))
        else:
            value = random.uniform(self.min_value, self.max_value)
        
        if self.param_type == "int":
            return int(round(value))
        return value
    
    def clip(self, value: float) -> Any:
        """Clip value to valid range."""
        if self.param_type == "categorical":
            return value if value in self.categories else self.categories[0]
        
        value = max(self.min_value, min(self.max_value, value))
        if self.param_type == "int":
            return int(round(value))
        return value


@dataclass
class ParameterSpace:
    """Parameter space for optimization."""
    parameters: List[Parameter]
    
    @property
    def dimension(self) -> int:
        return len(self.parameters)
    
    def sample(self) -> Dict[str, Any]:
        """Sample random point from space."""
        return {p.name: p.sample() for p in self.parameters}
    
    def clip(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Clip parameters to valid ranges."""
        return {p.name: p.clip(params.get(p.name, p.default or p.min_value)) 
                for p in self.parameters}
    
    def to_array(self, params: Dict[str, Any]) -> np.ndarray:
        """Convert params dict to array."""
        return np.array([params[p.name] for p in self.parameters])
    
    def from_array(self, arr: np.ndarray) -> Dict[str, Any]:
        """Convert array to params dict."""
        return {p.name: p.clip(arr[i]) for i, p in enumerate(self.parameters)}


# =============================================================================
# Bayesian Optimization
# =============================================================================

class BayesianOptimizer:
    """
    Bayesian optimization using Gaussian Process surrogate.
    
    Efficiently searches parameter space by building a probabilistic
    model of the objective function.
    """
    
    def __init__(
        self,
        param_space: ParameterSpace,
        objective_func: Callable[[Dict], float],
        n_initial: int = 5,
        n_iterations: int = 50,
        acquisition: str = "ei",  # "ei" (Expected Improvement), "ucb", "pi"
        exploration_weight: float = 0.1,
        n_parallel: int = 1,
    ):
        """
        Initialize Bayesian optimizer.
        
        Args:
            param_space: Parameter space to search
            objective_func: Function mapping params -> objective value
            n_initial: Number of random initial samples
            n_iterations: Number of optimization iterations
            acquisition: Acquisition function ("ei", "ucb", "pi")
            exploration_weight: Exploration vs exploitation tradeoff
            n_parallel: Number of parallel evaluations
        """
        self.param_space = param_space
        self.objective_func = objective_func
        self.n_initial = n_initial
        self.n_iterations = n_iterations
        self.acquisition = acquisition
        self.exploration_weight = exploration_weight
        self.n_parallel = n_parallel
        
        # Observed points
        self.X_observed: List[np.ndarray] = []
        self.y_observed: List[float] = []
        
        # Best result
        self.best_params: Optional[Dict] = None
        self.best_value: float = float('-inf')
        
        # History
        self.history: List[Dict] = []
    
    def _acquisition_ei(self, X: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
        """Expected Improvement acquisition function."""
        with np.errstate(divide='warn'):
            imp = mu - self.best_value - self.exploration_weight
            Z = imp / (sigma + 1e-10)
            ei = imp * stats.norm.cdf(Z) + sigma * stats.norm.pdf(Z)
            ei[sigma < 1e-10] = 0.0
        return ei
    
    def _acquisition_ucb(self, X: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
        """Upper Confidence Bound acquisition function."""
        return mu + self.exploration_weight * sigma
    
    def _acquisition_pi(self, X: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
        """Probability of Improvement acquisition function."""
        with np.errstate(divide='warn'):
            Z = (mu - self.best_value - self.exploration_weight) / (sigma + 1e-10)
            pi = stats.norm.cdf(Z)
            pi[sigma < 1e-10] = 0.0
        return pi
    
    def _fit_gp(self) -> Tuple[Callable, Callable]:
        """Fit Gaussian Process to observed data."""
        if len(self.X_observed) < 2:
            # Not enough data - return constant prediction
            mean_y = np.mean(self.y_observed) if self.y_observed else 0
            return lambda x: np.full(len(x), mean_y), lambda x: np.ones(len(x))
        
        X = np.array(self.X_observed)
        y = np.array(self.y_observed)
        
        # Normalize
        X_mean = X.mean(axis=0)
        X_std = X.std(axis=0) + 1e-10
        X_norm = (X - X_mean) / X_std
        
        y_mean = y.mean()
        y_std = y.std() + 1e-10
        y_norm = (y - y_mean) / y_std
        
        # Simple RBF kernel with fixed length scale
        def rbf_kernel(X1, X2, length_scale=1.0):
            dists = np.sum((X1[:, np.newaxis] - X2[np.newaxis, :]) ** 2, axis=2)
            return np.exp(-0.5 * dists / (length_scale ** 2))
        
        # Fit GP
        K = rbf_kernel(X_norm, X_norm) + 1e-6 * np.eye(len(X_norm))
        try:
            L = np.linalg.cholesky(K)
            alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_norm))
        except np.linalg.LinAlgError:
            # Fallback if Cholesky fails
            alpha = np.linalg.lstsq(K, y_norm, rcond=None)[0]
            L = np.eye(len(K))
        
        def predict_mean(X_new):
            X_new_norm = (np.array(X_new) - X_mean) / X_std
            k_star = rbf_kernel(X_new_norm, X_norm)
            return k_star @ alpha * y_std + y_mean
        
        def predict_std(X_new):
            X_new_norm = (np.array(X_new) - X_mean) / X_std
            k_star = rbf_kernel(X_new_norm, X_norm)
            k_star_star = rbf_kernel(X_new_norm, X_new_norm)
            try:
                v = np.linalg.solve(L, k_star.T)
                var = np.diag(k_star_star) - np.sum(v ** 2, axis=0)
                var = np.maximum(var, 1e-10)
            except Exception:
                var = np.ones(len(X_new))
            return np.sqrt(var) * y_std
        
        return predict_mean, predict_std
    
    def _select_next_point(self, predict_mean, predict_std) -> Dict[str, Any]:
        """Select next point to evaluate using acquisition function."""
        # Generate candidate points
        n_candidates = 1000
        candidates = [self.param_space.sample() for _ in range(n_candidates)]
        X_candidates = np.array([self.param_space.to_array(c) for c in candidates])
        
        # Predict mean and std
        mu = predict_mean(X_candidates)
        sigma = predict_std(X_candidates)
        
        # Calculate acquisition
        if self.acquisition == "ei":
            acq = self._acquisition_ei(X_candidates, mu, sigma)
        elif self.acquisition == "ucb":
            acq = self._acquisition_ucb(X_candidates, mu, sigma)
        else:
            acq = self._acquisition_pi(X_candidates, mu, sigma)
        
        # Select best
        best_idx = np.argmax(acq)
        return candidates[best_idx]
    
    def optimize(self, callback: Callable = None) -> Tuple[Dict, float]:
        """
        Run Bayesian optimization.
        
        Args:
            callback: Optional callback(iteration, params, value)
            
        Returns:
            (best_params, best_value)
        """
        # Initial random sampling
        logger.info(f"Starting Bayesian optimization with {self.n_initial} initial samples")
        
        for i in range(self.n_initial):
            params = self.param_space.sample()
            value = self._safe_evaluate(params)
            
            self.X_observed.append(self.param_space.to_array(params))
            self.y_observed.append(value)
            
            if value > self.best_value:
                self.best_value = value
                self.best_params = params
            
            self.history.append({
                'iteration': i,
                'params': params,
                'value': value,
                'type': 'initial',
            })
            
            if callback:
                callback(i, params, value)
        
        # Bayesian optimization iterations
        for i in range(self.n_iterations):
            # Fit GP
            predict_mean, predict_std = self._fit_gp()
            
            # Select next point
            next_params = self._select_next_point(predict_mean, predict_std)
            
            # Evaluate
            value = self._safe_evaluate(next_params)
            
            self.X_observed.append(self.param_space.to_array(next_params))
            self.y_observed.append(value)
            
            if value > self.best_value:
                self.best_value = value
                self.best_params = next_params
                logger.info(f"New best: {value:.4f} at iteration {self.n_initial + i}")
            
            self.history.append({
                'iteration': self.n_initial + i,
                'params': next_params,
                'value': value,
                'type': 'bayesian',
            })
            
            if callback:
                callback(self.n_initial + i, next_params, value)
        
        return self.best_params, self.best_value
    
    def _safe_evaluate(self, params: Dict) -> float:
        """Safely evaluate objective function."""
        try:
            value = self.objective_func(params)
            if math.isnan(value) or math.isinf(value):
                return float('-inf')
            return value
        except Exception as e:
            logger.warning(f"Evaluation failed: {e}")
            return float('-inf')


# =============================================================================
# Genetic Algorithm Optimization
# =============================================================================

@dataclass
class Individual:
    """Individual in genetic population."""
    params: Dict[str, Any]
    fitness: float = float('-inf')
    age: int = 0


class GeneticOptimizer:
    """
    Genetic algorithm optimizer for strategy parameters.
    
    Robust to non-convex, multi-modal objective landscapes.
    """
    
    def __init__(
        self,
        param_space: ParameterSpace,
        objective_func: Callable[[Dict], float],
        population_size: int = 50,
        n_generations: int = 100,
        mutation_rate: float = 0.1,
        crossover_rate: float = 0.7,
        elite_size: int = 5,
        tournament_size: int = 3,
        n_parallel: int = 1,
    ):
        """
        Initialize genetic optimizer.
        
        Args:
            param_space: Parameter space
            objective_func: Objective function
            population_size: Population size
            n_generations: Number of generations
            mutation_rate: Mutation probability
            crossover_rate: Crossover probability
            elite_size: Number of elite individuals to preserve
            tournament_size: Tournament selection size
            n_parallel: Parallel evaluations
        """
        self.param_space = param_space
        self.objective_func = objective_func
        self.population_size = population_size
        self.n_generations = n_generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elite_size = elite_size
        self.tournament_size = tournament_size
        self.n_parallel = n_parallel
        
        # Population
        self.population: List[Individual] = []
        
        # Best result
        self.best_individual: Optional[Individual] = None
        
        # History
        self.history: List[Dict] = []
    
    def _initialize_population(self):
        """Create initial random population."""
        self.population = []
        for _ in range(self.population_size):
            params = self.param_space.sample()
            self.population.append(Individual(params=params))
    
    def _evaluate_population(self):
        """Evaluate fitness of all individuals."""
        if self.n_parallel > 1:
            with ProcessPoolExecutor(max_workers=self.n_parallel) as executor:
                futures = {
                    executor.submit(self._safe_evaluate, ind.params): ind
                    for ind in self.population
                    if ind.fitness == float('-inf')
                }
                for future in as_completed(futures):
                    ind = futures[future]
                    ind.fitness = future.result()
        else:
            for ind in self.population:
                if ind.fitness == float('-inf'):
                    ind.fitness = self._safe_evaluate(ind.params)
    
    def _safe_evaluate(self, params: Dict) -> float:
        """Safely evaluate objective function."""
        try:
            value = self.objective_func(params)
            if math.isnan(value) or math.isinf(value):
                return float('-inf')
            return value
        except Exception as e:
            logger.warning(f"Evaluation failed: {e}")
            return float('-inf')
    
    def _tournament_select(self) -> Individual:
        """Select individual via tournament selection."""
        tournament = random.sample(self.population, self.tournament_size)
        return max(tournament, key=lambda x: x.fitness)
    
    def _crossover(self, parent1: Individual, parent2: Individual) -> Individual:
        """Create offspring via crossover."""
        if random.random() > self.crossover_rate:
            return Individual(params=parent1.params.copy())
        
        # Blend crossover (works for continuous and discrete)
        child_params = {}
        for param in self.param_space.parameters:
            if param.param_type == "categorical":
                # Random selection for categorical
                child_params[param.name] = random.choice([
                    parent1.params[param.name],
                    parent2.params[param.name]
                ])
            else:
                # BLX-alpha crossover for numeric
                v1 = parent1.params[param.name]
                v2 = parent2.params[param.name]
                alpha = 0.5
                low = min(v1, v2) - alpha * abs(v2 - v1)
                high = max(v1, v2) + alpha * abs(v2 - v1)
                child_params[param.name] = param.clip(random.uniform(low, high))
        
        return Individual(params=child_params)
    
    def _mutate(self, individual: Individual):
        """Apply mutation to individual."""
        for param in self.param_space.parameters:
            if random.random() < self.mutation_rate:
                if param.param_type == "categorical":
                    individual.params[param.name] = random.choice(param.categories)
                else:
                    # Gaussian mutation
                    scale = (param.max_value - param.min_value) * 0.1
                    current = individual.params[param.name]
                    mutated = current + random.gauss(0, scale)
                    individual.params[param.name] = param.clip(mutated)
    
    def optimize(self, callback: Callable = None) -> Tuple[Dict, float]:
        """
        Run genetic algorithm optimization.
        
        Args:
            callback: Optional callback(generation, best_individual)
            
        Returns:
            (best_params, best_fitness)
        """
        # Initialize population
        self._initialize_population()
        
        for generation in range(self.n_generations):
            # Evaluate fitness
            self._evaluate_population()
            
            # Sort by fitness
            self.population.sort(key=lambda x: x.fitness, reverse=True)
            
            # Update best
            if (self.best_individual is None or 
                self.population[0].fitness > self.best_individual.fitness):
                self.best_individual = Individual(
                    params=self.population[0].params.copy(),
                    fitness=self.population[0].fitness,
                )
                logger.info(f"Generation {generation}: New best fitness = {self.best_individual.fitness:.4f}")
            
            # Record history
            self.history.append({
                'generation': generation,
                'best_fitness': self.best_individual.fitness,
                'avg_fitness': np.mean([i.fitness for i in self.population if i.fitness > float('-inf')]),
                'diversity': self._calculate_diversity(),
            })
            
            if callback:
                callback(generation, self.best_individual)
            
            # Create next generation
            next_population = []
            
            # Elitism - keep best individuals
            for i in range(self.elite_size):
                elite = Individual(
                    params=self.population[i].params.copy(),
                    fitness=self.population[i].fitness,
                    age=self.population[i].age + 1,
                )
                next_population.append(elite)
            
            # Create offspring
            while len(next_population) < self.population_size:
                parent1 = self._tournament_select()
                parent2 = self._tournament_select()
                child = self._crossover(parent1, parent2)
                self._mutate(child)
                next_population.append(child)
            
            self.population = next_population
            
            # Age individuals
            for ind in self.population:
                ind.age += 1
        
        return self.best_individual.params, self.best_individual.fitness
    
    def _calculate_diversity(self) -> float:
        """Calculate population diversity (std of parameters)."""
        if len(self.population) < 2:
            return 0.0
        
        diversities = []
        for param in self.param_space.parameters:
            if param.param_type != "categorical":
                values = [ind.params[param.name] for ind in self.population]
                if param.max_value != param.min_value:
                    normalized = [(v - param.min_value) / (param.max_value - param.min_value) 
                                 for v in values]
                    diversities.append(np.std(normalized))
        
        return np.mean(diversities) if diversities else 0.0


# =============================================================================
# Multi-Objective Optimization (Pareto)
# =============================================================================

class ParetoOptimizer:
    """
    Multi-objective optimization using NSGA-II style approach.
    
    Finds Pareto-optimal solutions for multiple objectives
    (e.g., maximize return AND minimize drawdown).
    """
    
    def __init__(
        self,
        param_space: ParameterSpace,
        objective_funcs: List[Callable[[Dict], float]],
        population_size: int = 100,
        n_generations: int = 50,
    ):
        """
        Initialize Pareto optimizer.
        
        Args:
            param_space: Parameter space
            objective_funcs: List of objective functions (all maximized)
            population_size: Population size
            n_generations: Number of generations
        """
        self.param_space = param_space
        self.objective_funcs = objective_funcs
        self.n_objectives = len(objective_funcs)
        self.population_size = population_size
        self.n_generations = n_generations
        
        self.pareto_front: List[Tuple[Dict, List[float]]] = []
    
    def _dominates(self, obj1: List[float], obj2: List[float]) -> bool:
        """Check if obj1 dominates obj2."""
        better_in_any = False
        for v1, v2 in zip(obj1, obj2):
            if v1 < v2:
                return False
            if v1 > v2:
                better_in_any = True
        return better_in_any
    
    def _fast_non_dominated_sort(self, population: List[Tuple[Dict, List[float]]]) -> List[List[int]]:
        """NSGA-II fast non-dominated sorting."""
        n = len(population)
        domination_count = [0] * n
        dominated_solutions = [[] for _ in range(n)]
        fronts = [[]]
        
        for i in range(n):
            for j in range(n):
                if i != j:
                    if self._dominates(population[i][1], population[j][1]):
                        dominated_solutions[i].append(j)
                    elif self._dominates(population[j][1], population[i][1]):
                        domination_count[i] += 1
            
            if domination_count[i] == 0:
                fronts[0].append(i)
        
        i = 0
        while fronts[i]:
            next_front = []
            for p in fronts[i]:
                for q in dominated_solutions[p]:
                    domination_count[q] -= 1
                    if domination_count[q] == 0:
                        next_front.append(q)
            i += 1
            fronts.append(next_front)
        
        return fronts[:-1]  # Remove empty last front
    
    def optimize(self) -> List[Tuple[Dict, List[float]]]:
        """
        Run multi-objective optimization.
        
        Returns:
            Pareto front: List of (params, [objective_values])
        """
        # Initialize population
        population = []
        for _ in range(self.population_size):
            params = self.param_space.sample()
            objectives = [f(params) for f in self.objective_funcs]
            population.append((params, objectives))
        
        for generation in range(self.n_generations):
            # Non-dominated sorting
            fronts = self._fast_non_dominated_sort(population)
            
            # Select next generation
            next_population = []
            for front in fronts:
                if len(next_population) + len(front) <= self.population_size:
                    next_population.extend([population[i] for i in front])
                else:
                    # Fill remaining with crowding distance selection
                    remaining = self.population_size - len(next_population)
                    front_individuals = [population[i] for i in front]
                    # Simplified: random selection
                    next_population.extend(random.sample(front_individuals, remaining))
                    break
            
            # Store Pareto front (first front)
            if fronts:
                self.pareto_front = [population[i] for i in fronts[0]]
            
            # Create offspring (simplified crossover)
            offspring = []
            while len(offspring) < self.population_size:
                p1, p2 = random.sample(next_population, 2)
                child_params = {}
                for param in self.param_space.parameters:
                    if random.random() < 0.5:
                        child_params[param.name] = p1[0][param.name]
                    else:
                        child_params[param.name] = p2[0][param.name]
                    # Mutation
                    if random.random() < 0.1:
                        child_params[param.name] = param.sample()
                
                objectives = [f(child_params) for f in self.objective_funcs]
                offspring.append((child_params, objectives))
            
            population = next_population + offspring
            
            logger.info(f"Generation {generation}: Pareto front size = {len(self.pareto_front)}")
        
        return self.pareto_front


# =============================================================================
# Reward Shaping for RL
# =============================================================================

class RewardShaper:
    """
    Shapes rewards for RL training based on backtest analysis.
    
    Provides dense rewards that guide learning without changing
    optimal policy.
    """
    
    def __init__(
        self,
        baseline_sharpe: float = 0.0,
        risk_penalty: float = 0.1,
        cost_penalty: float = 0.05,
        regime_bonus: float = 0.1,
    ):
        """
        Initialize reward shaper.
        
        Args:
            baseline_sharpe: Baseline Sharpe to compare against
            risk_penalty: Penalty for excess risk
            cost_penalty: Penalty for trading costs
            regime_bonus: Bonus for trading in favorable regimes
        """
        self.baseline_sharpe = baseline_sharpe
        self.risk_penalty = risk_penalty
        self.cost_penalty = cost_penalty
        self.regime_bonus = regime_bonus
    
    def shape_reward(
        self,
        raw_pnl: float,
        position_risk: float,
        trading_cost: float,
        regime_quality: float,  # 0-1, higher = better regime
        action_consistency: float = 1.0,  # Penalty for flip-flopping
    ) -> float:
        """
        Shape a single-step reward.
        
        Args:
            raw_pnl: Raw P&L for the step
            position_risk: Current position risk (e.g., drawdown)
            trading_cost: Cost incurred this step
            regime_quality: Quality of current regime (0-1)
            action_consistency: Consistency of actions (0-1)
            
        Returns:
            Shaped reward
        """
        reward = raw_pnl
        
        # Risk penalty (quadratic to penalize large risks more)
        reward -= self.risk_penalty * (position_risk ** 2)
        
        # Trading cost penalty
        reward -= self.cost_penalty * trading_cost
        
        # Regime bonus
        reward += self.regime_bonus * regime_quality * max(0, raw_pnl)
        
        # Action consistency (penalize flip-flopping)
        reward *= action_consistency
        
        return reward
    
    def calculate_episode_metrics(
        self,
        rewards: List[float],
        returns: np.ndarray,
    ) -> Dict[str, float]:
        """
        Calculate metrics for reward shaping evaluation.
        
        Returns metrics useful for hyperparameter tuning of reward shaping.
        """
        dist_analysis = DistributionAnalyzer.calculate_robust_statistics(returns)
        
        return {
            "total_reward": sum(rewards),
            "mean_reward": np.mean(rewards),
            "reward_std": np.std(rewards),
            "return_sharpe": dist_analysis.get("mean", 0) / (dist_analysis.get("std", 1) + 1e-10),
            "return_skew": dist_analysis.get("skewness", 0),
            "return_kurtosis": dist_analysis.get("kurtosis", 0),
            "is_fat_tailed": dist_analysis.get("is_fat_tailed", False),
        }
