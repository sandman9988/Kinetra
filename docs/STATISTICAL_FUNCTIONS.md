# Statistical Functions - MT5 to Kinetra Mapping

## Overview

MT5 provides a comprehensive statistical library with 5 functions per distribution. Kinetra uses Python's `scipy.stats` which provides equivalent functionality.

## MT5 Function Pattern

For each distribution `X`, MT5 provides:

1. **Probability Density**: `MathProbabilityDensityX()` - PDF
2. **Cumulative Distribution**: `MathCumulativeDistributionX()` - CDF
3. **Quantiles**: `MathQuantileX()` - Inverse CDF
4. **Random Generation**: `MathRandomX()` - RNG
5. **Theoretical Moments**: `MathMomentsX()` - Mean, variance, skewness, kurtosis

## Distribution Mappings

### Normal Distribution

| MT5 Function | SciPy Equivalent | Kinetra Usage |
|--------------|------------------|---------------|
| `MathProbabilityDensityNormal(x, mu, sigma)` | `scipy.stats.norm.pdf(x, mu, sigma)` | Return distribution analysis |
| `MathCumulativeDistributionNormal(x, mu, sigma)` | `scipy.stats.norm.cdf(x, mu, sigma)` | VaR calculation |
| `MathQuantileNormal(p, mu, sigma)` | `scipy.stats.norm.ppf(p, mu, sigma)` | Confidence intervals |
| `MathRandomNormal(mu, sigma, count)` | `np.random.normal(mu, sigma, count)` | Monte Carlo simulation |
| `MathMomentsNormal(mu, sigma)` | `scipy.stats.norm.stats(mu, sigma, moments='mvsk')` | Distribution validation |

**Kinetra Implementation**:
```python
from scipy import stats
import numpy as np

class StatisticalRiskMetrics:
    """Statistical risk calculations using SciPy (MT5 equivalent)."""

    @staticmethod
    def calculate_var_normal(returns: np.ndarray, confidence: float = 0.95) -> float:
        """
        Value at Risk (VaR) using normal distribution.

        MT5 equivalent: MathQuantileNormal(1 - confidence, mu, sigma)
        """
        mu = np.mean(returns)
        sigma = np.std(returns)
        return stats.norm.ppf(1 - confidence, mu, sigma)

    @staticmethod
    def calculate_cvar_normal(returns: np.ndarray, confidence: float = 0.95) -> float:
        """
        Conditional Value at Risk (CVaR) using normal distribution.

        MT5 equivalent: Integral of tail beyond VaR
        """
        mu = np.mean(returns)
        sigma = np.std(returns)
        var_threshold = stats.norm.ppf(1 - confidence, mu, sigma)

        # CVaR = expected value of returns below VaR
        return mu - sigma * stats.norm.pdf(stats.norm.ppf(1 - confidence)) / (1 - confidence)
```

### Student's t-Distribution (Fat Tails)

| MT5 Function | SciPy Equivalent | Kinetra Usage |
|--------------|------------------|---------------|
| `MathProbabilityDensityT(x, nu)` | `scipy.stats.t.pdf(x, nu)` | Fat-tailed return analysis |
| `MathCumulativeDistributionT(x, nu)` | `scipy.stats.t.cdf(x, nu)` | VaR with fat tails |
| `MathQuantileT(p, nu)` | `scipy.stats.t.ppf(p, nu)` | Conservative risk estimates |
| `MathRandomT(nu, count)` | `np.random.standard_t(nu, count)` | Stress testing |
| `MathMomentsT(nu)` | `scipy.stats.t.stats(nu, moments='mvsk')` | Distribution fitting |

**Kinetra Implementation**:
```python
@staticmethod
def calculate_var_t_distribution(returns: np.ndarray, confidence: float = 0.95) -> float:
    """
    VaR using Student's t-distribution (better for fat tails).

    MT5 equivalent: MathQuantileT(1 - confidence, nu)
    """
    # Estimate degrees of freedom
    from scipy.stats import t
    params = t.fit(returns)
    df, loc, scale = params

    return t.ppf(1 - confidence, df, loc, scale)
```

### Log-Normal Distribution (Prices)

| MT5 Function | SciPy Equivalent | Kinetra Usage |
|--------------|------------------|---------------|
| `MathProbabilityDensityLognormal(x, mu, sigma)` | `scipy.stats.lognorm.pdf(x, sigma, scale=exp(mu))` | Price distribution |
| `MathCumulativeDistributionLognormal(x, mu, sigma)` | `scipy.stats.lognorm.cdf(x, sigma, scale=exp(mu))` | Price probabilities |
| `MathQuantileLognormal(p, mu, sigma)` | `scipy.stats.lognorm.ppf(p, sigma, scale=exp(mu))` | Price targets |
| `MathRandomLognormal(mu, sigma, count)` | `np.random.lognormal(mu, sigma, count)` | Price path simulation |
| `MathMomentsLognormal(mu, sigma)` | `scipy.stats.lognorm.stats(sigma, scale=exp(mu))` | Price statistics |

**Kinetra Implementation**:
```python
@staticmethod
def simulate_price_paths(initial_price: float, mu: float, sigma: float,
                        steps: int, paths: int) -> np.ndarray:
    """
    Simulate geometric Brownian motion price paths.

    MT5 equivalent: MathRandomLognormal() for multiplicative changes
    """
    dt = 1.0 / steps

    # Generate random shocks
    shocks = np.random.normal(mu * dt, sigma * np.sqrt(dt), (paths, steps))

    # Compute price paths
    log_returns = np.cumsum(shocks, axis=1)
    price_paths = initial_price * np.exp(log_returns)

    return price_paths
```

### Exponential Distribution (Time-to-Event)

| MT5 Function | SciPy Equivalent | Kinetra Usage |
|--------------|------------------|---------------|
| `MathProbabilityDensityExponential(x, mu)` | `scipy.stats.expon.pdf(x, scale=mu)` | Trade arrival modeling |
| `MathCumulativeDistributionExponential(x, mu)` | `scipy.stats.expon.cdf(x, scale=mu)` | Time-to-drawdown |
| `MathQuantileExponential(p, mu)` | `scipy.stats.expon.ppf(p, scale=mu)` | Expected wait time |
| `MathRandomExponential(mu, count)` | `np.random.exponential(mu, count)` | Event simulation |
| `MathMomentsExponential(mu)` | `scipy.stats.expon.stats(scale=mu)` | Timing statistics |

**Kinetra Implementation**:
```python
@staticmethod
def estimate_time_to_drawdown(drawdown_history: np.ndarray,
                              threshold: float) -> float:
    """
    Estimate time until next drawdown exceeds threshold.

    MT5 equivalent: MathQuantileExponential() for failure time
    """
    # Find intervals between threshold crossings
    crossings = np.where(drawdown_history > threshold)[0]
    if len(crossings) < 2:
        return np.inf

    intervals = np.diff(crossings)
    mean_interval = np.mean(intervals)

    # Exponential distribution: 63.2% probability within mean_interval
    return mean_interval
```

### Weibull Distribution (Extreme Events)

| MT5 Function | SciPy Equivalent | Kinetra Usage |
|--------------|------------------|---------------|
| `MathProbabilityDensityWeibull(x, a, b)` | `scipy.stats.weibull_min.pdf(x, a, scale=b)` | Extreme loss modeling |
| `MathCumulativeDistributionWeibull(x, a, b)` | `scipy.stats.weibull_min.cdf(x, a, scale=b)` | Tail risk |
| `MathQuantileWeibull(p, a, b)` | `scipy.stats.weibull_min.ppf(p, a, scale=b)` | Worst-case scenarios |
| `MathRandomWeibull(a, b, count)` | `scipy.stats.weibull_min.rvs(a, scale=b, size=count)` | Stress testing |
| `MathMomentsWeibull(a, b)` | `scipy.stats.weibull_min.stats(a, scale=b)` | Tail statistics |

**Kinetra Implementation**:
```python
@staticmethod
def model_extreme_losses(losses: np.ndarray, percentile: float = 0.99) -> dict:
    """
    Model extreme losses using Weibull distribution.

    MT5 equivalent: MathQuantileWeibull() for tail risk
    """
    from scipy.stats import weibull_min

    # Fit Weibull to losses
    params = weibull_min.fit(losses)
    shape, loc, scale = params

    # Calculate extreme loss estimate
    extreme_loss = weibull_min.ppf(percentile, shape, loc, scale)

    return {
        'shape': shape,
        'scale': scale,
        'extreme_loss_99pct': extreme_loss,
        'expected_loss': weibull_min.mean(shape, loc, scale),
    }
```

## Integration with Portfolio Health Monitor

### Current Implementation

**kinetra/portfolio_health.py** already uses some statistical measures:

```python
# CVaR calculation (uses normal approximation)
def calculate_cvar(returns: np.ndarray, confidence: float = 0.95) -> float:
    """Conditional Value at Risk."""
    var_threshold = np.percentile(returns, (1 - confidence) * 100)
    return returns[returns <= var_threshold].mean()
```

### Enhanced Statistical Risk Metrics

```python
class EnhancedRiskMetrics:
    """Enhanced risk metrics using full statistical distributions."""

    @staticmethod
    def calculate_parametric_var(returns: np.ndarray,
                                 confidence: float = 0.95,
                                 distribution: str = 'normal') -> float:
        """
        Parametric VaR using specified distribution.

        Args:
            returns: Historical returns
            confidence: Confidence level (0.95 = 95%)
            distribution: 'normal', 't', 'lognormal'

        Returns:
            Value at Risk

        MT5 equivalent: MathQuantileX(1 - confidence, params)
        """
        if distribution == 'normal':
            mu, sigma = np.mean(returns), np.std(returns)
            return stats.norm.ppf(1 - confidence, mu, sigma)

        elif distribution == 't':
            # Fit t-distribution (better for fat tails)
            params = stats.t.fit(returns)
            df, loc, scale = params
            return stats.t.ppf(1 - confidence, df, loc, scale)

        elif distribution == 'lognormal':
            # For price changes (multiplicative)
            shape, loc, scale = stats.lognorm.fit(returns)
            return stats.lognorm.ppf(1 - confidence, shape, loc, scale)

        else:
            raise ValueError(f"Unknown distribution: {distribution}")

    @staticmethod
    def calculate_cornish_fisher_var(returns: np.ndarray,
                                     confidence: float = 0.95) -> float:
        """
        Cornish-Fisher VaR (adjusts for skewness and kurtosis).

        Better than normal VaR when returns are non-normal.

        MT5 equivalent: Uses MathMomentsNormal() + adjustments
        """
        mu = np.mean(returns)
        sigma = np.std(returns)
        skew = stats.skew(returns)
        kurt = stats.kurtosis(returns)

        # Standard normal quantile
        z = stats.norm.ppf(1 - confidence)

        # Cornish-Fisher expansion
        z_cf = z + (z**2 - 1) * skew / 6
        z_cf += (z**3 - 3*z) * kurt / 24
        z_cf -= (2*z**3 - 5*z) * skew**2 / 36

        return mu + sigma * z_cf

    @staticmethod
    def backtestest_var(returns: np.ndarray, var: float) -> dict:
        """
        Backtest VaR using Kupiec test.

        Tests if VaR violations occur at expected frequency.

        MT5 equivalent: MathCumulativeDistributionBinomial()
        """
        violations = np.sum(returns < var)
        total = len(returns)

        # Expected violations at 95% confidence = 5%
        expected_rate = 0.05
        expected_violations = total * expected_rate

        # Kupiec likelihood ratio test
        actual_rate = violations / total
        if actual_rate > 0 and actual_rate < 1:
            lr = -2 * np.log((1 - expected_rate)**(total - violations) * expected_rate**violations /
                            (1 - actual_rate)**(total - violations) * actual_rate**violations)
        else:
            lr = np.inf

        # Chi-squared distribution with 1 degree of freedom
        p_value = 1 - stats.chi2.cdf(lr, df=1)

        return {
            'violations': violations,
            'expected_violations': expected_violations,
            'violation_rate': actual_rate,
            'lr_statistic': lr,
            'p_value': p_value,
            'rejected': p_value < 0.05,  # Reject if VaR is miscalibrated
        }
```

## Monte Carlo Simulation

### Price Path Generation

```python
class MonteCarloSimulator:
    """Monte Carlo simulation using MT5-equivalent distributions."""

    @staticmethod
    def simulate_geometric_brownian_motion(S0: float, mu: float, sigma: float,
                                          T: float, steps: int, paths: int) -> np.ndarray:
        """
        Simulate GBM price paths.

        MT5 equivalent: MathRandomNormal() for Brownian increments

        Args:
            S0: Initial price
            mu: Drift (annual return)
            sigma: Volatility (annual)
            T: Time horizon (years)
            steps: Number of time steps
            paths: Number of paths to simulate

        Returns:
            Array of shape (paths, steps) with price paths
        """
        dt = T / steps

        # Generate Brownian motion increments
        # MT5: MathRandomNormal(0, sqrt(dt), paths * steps)
        dW = np.random.normal(0, np.sqrt(dt), (paths, steps))

        # Compute log returns
        log_returns = (mu - 0.5 * sigma**2) * dt + sigma * dW

        # Convert to prices
        log_prices = np.cumsum(log_returns, axis=1)
        prices = S0 * np.exp(log_prices)

        return prices

    @staticmethod
    def simulate_with_jumps(S0: float, mu: float, sigma: float,
                           jump_intensity: float, jump_mean: float, jump_std: float,
                           T: float, steps: int, paths: int) -> np.ndarray:
        """
        Simulate jump-diffusion process (Merton model).

        MT5 equivalent:
        - MathRandomNormal() for diffusion
        - MathRandomPoisson() for jump timing
        - MathRandomNormal() for jump size
        """
        dt = T / steps

        # Diffusion component
        dW = np.random.normal(0, np.sqrt(dt), (paths, steps))
        diffusion = (mu - 0.5 * sigma**2) * dt + sigma * dW

        # Jump component
        # MT5: MathRandomPoisson(jump_intensity * dt, paths * steps)
        jumps = np.random.poisson(jump_intensity * dt, (paths, steps))

        # MT5: MathRandomNormal(jump_mean, jump_std, total_jumps)
        jump_sizes = np.random.normal(jump_mean, jump_std, (paths, steps))
        jump_component = jumps * jump_sizes

        # Combined log returns
        log_returns = diffusion + jump_component

        # Convert to prices
        log_prices = np.cumsum(log_returns, axis=1)
        prices = S0 * np.exp(log_prices)

        return prices
```

## Statistical Arbitrage

### Pairs Trading Statistics

```python
class PairsTradingStatistics:
    """Statistical tests for pairs trading."""

    @staticmethod
    def calculate_spread_zscore(price1: np.ndarray, price2: np.ndarray,
                               lookback: int = 20) -> np.ndarray:
        """
        Calculate z-score of price spread.

        MT5 equivalent: MathQuantileNormal() for standardization
        """
        # Calculate spread
        spread = price1 - price2

        # Rolling mean and std
        rolling_mean = pd.Series(spread).rolling(lookback).mean()
        rolling_std = pd.Series(spread).rolling(lookback).std()

        # Z-score (number of standard deviations from mean)
        zscore = (spread - rolling_mean) / rolling_std

        return zscore.values

    @staticmethod
    def test_cointegration(price1: np.ndarray, price2: np.ndarray) -> dict:
        """
        Test for cointegration using Engle-Granger method.

        MT5 equivalent: Uses MathCumulativeDistributionT() for p-values
        """
        from statsmodels.tsa.stattools import coint

        # Perform cointegration test
        score, pvalue, crit_values = coint(price1, price2)

        return {
            'test_statistic': score,
            'p_value': pvalue,
            'critical_values': {
                '1%': crit_values[0],
                '5%': crit_values[1],
                '10%': crit_values[2],
            },
            'is_cointegrated': pvalue < 0.05,
        }
```

## Distribution Fitting and Testing

### Goodness-of-Fit Tests

```python
class DistributionFitting:
    """Fit and test distributions."""

    @staticmethod
    def fit_best_distribution(data: np.ndarray) -> dict:
        """
        Find best-fitting distribution from common candidates.

        MT5 equivalent: Uses MathProbabilityDensityX() for likelihood
        """
        distributions = [
            ('normal', stats.norm),
            ('t', stats.t),
            ('lognormal', stats.lognorm),
            ('exponential', stats.expon),
            ('weibull', stats.weibull_min),
        ]

        best_fit = None
        best_aic = np.inf

        for name, dist in distributions:
            try:
                # Fit distribution
                params = dist.fit(data)

                # Calculate log-likelihood
                log_likelihood = np.sum(dist.logpdf(data, *params))

                # AIC = 2k - 2ln(L)
                k = len(params)
                aic = 2 * k - 2 * log_likelihood

                if aic < best_aic:
                    best_aic = aic
                    best_fit = {
                        'distribution': name,
                        'params': params,
                        'aic': aic,
                        'log_likelihood': log_likelihood,
                    }
            except:
                continue

        return best_fit

    @staticmethod
    def kolmogorov_smirnov_test(data: np.ndarray, distribution: str = 'normal') -> dict:
        """
        Kolmogorov-Smirnov goodness-of-fit test.

        MT5 equivalent: Compares MathCumulativeDistributionX() to empirical CDF
        """
        if distribution == 'normal':
            mu, sigma = np.mean(data), np.std(data)
            statistic, pvalue = stats.kstest(data, 'norm', args=(mu, sigma))

        elif distribution == 't':
            params = stats.t.fit(data)
            statistic, pvalue = stats.kstest(data, 't', args=params)

        else:
            raise ValueError(f"Unknown distribution: {distribution}")

        return {
            'statistic': statistic,
            'p_value': pvalue,
            'is_normal': pvalue > 0.05,  # Null hypothesis: data is from distribution
        }
```

## Usage in Kinetra

### Integration with PortfolioHealthMonitor

```python
# Example: Enhanced risk calculation
from kinetra.portfolio_health import PortfolioHealthMonitor

class EnhancedPortfolioHealthMonitor(PortfolioHealthMonitor):
    """Portfolio health with enhanced statistical risk metrics."""

    def calculate_downside_risk_enhanced(self, returns: np.ndarray) -> dict:
        """
        Enhanced downside risk using multiple distributions.

        Uses MT5-equivalent statistical functions.
        """
        # VaR using different distributions
        var_normal = stats.norm.ppf(0.05, np.mean(returns), np.std(returns))
        var_t = self._calculate_var_t_distribution(returns, 0.95)
        var_cf = self._calculate_cornish_fisher_var(returns, 0.95)

        # CVaR (expected shortfall)
        cvar_95 = returns[returns <= var_normal].mean()
        cvar_99 = returns[returns <= stats.norm.ppf(0.01, np.mean(returns), np.std(returns))].mean()

        # Extreme loss modeling (Weibull)
        extreme_model = self._model_extreme_losses(returns[returns < 0])

        return {
            'var_95_normal': var_normal,
            'var_95_t': var_t,
            'var_95_cornish_fisher': var_cf,
            'cvar_95': cvar_95,
            'cvar_99': cvar_99,
            'extreme_loss_99pct': extreme_model['extreme_loss_99pct'],
        }
```

## Summary

| MT5 Category | Python/SciPy | Kinetra Usage |
|--------------|--------------|---------------|
| **Normal** | `scipy.stats.norm` | VaR, CVaR, return analysis |
| **Student's t** | `scipy.stats.t` | Fat-tailed risk, conservative estimates |
| **Log-normal** | `scipy.stats.lognorm` | Price modeling, multiplicative returns |
| **Exponential** | `scipy.stats.expon` | Time-to-event, drawdown duration |
| **Weibull** | `scipy.stats.weibull_min` | Extreme losses, tail risk |
| **Binomial** | `scipy.stats.binom` | VaR backtesting, discrete outcomes |
| **Poisson** | `scipy.stats.poisson` | Trade arrival, jump models |
| **Chi-squared** | `scipy.stats.chi2` | Hypothesis testing, correlation tests |
| **F-distribution** | `scipy.stats.f` | Variance ratio tests |

All MT5 statistical functions have direct Python equivalents, providing the same capabilities for risk modeling, distribution fitting, and Monte Carlo simulation!

---

## ALGLIB Package Mapping (MT5 to Python)

MT5 uses ALGLIB for numerical analysis. Below is the complete mapping to Python libraries.

### Data Analysis (dataanalysis.mqh)

| ALGLIB Class | Python Equivalent | Kinetra Status |
|--------------|-------------------|----------------|
| `CBdSS` - Error functions | `scipy.special.erf`, `scipy.special.erfc` | ✅ Available |
| `CDForest` - Decision tree forests | `sklearn.ensemble.RandomForestClassifier` | ✅ Can integrate |
| `CKMeans` - K-means clustering | `sklearn.cluster.KMeans` | ⏳ Planned for regime clustering |
| `CLDA` - Linear discriminant analysis | `sklearn.discriminant_analysis.LinearDiscriminantAnalysis` | ⏳ For signal classification |
| `CLinReg` - Linear regression | `sklearn.linear_model.LinearRegression` | ✅ Available |
| `CMLPBase` - Neural networks (MLP) | `torch.nn.Sequential`, `sklearn.neural_network.MLPRegressor` | ✅ KinetraAgent uses PyTorch |
| `CLogit` - Logit regression | `sklearn.linear_model.LogisticRegression` | ✅ Available |
| `CMarkovCPD` - Markov chains | `hmmlearn.hmm.GaussianHMM` | ✅ Already using in regime detection |
| `CMLPTrain` - MLP training | `torch.optim.Adam`, PPO in stable-baselines3 | ✅ Implemented |
| `CMLPE` - Neural network ensembles | Custom ensemble implementation | ⏳ Can add |
| `CPCAnalysis` - PCA | `sklearn.decomposition.PCA` | ✅ Available |

**Kinetra Usage**:
```python
# K-means for regime clustering (similar to CKMeans)
from sklearn.cluster import KMeans

def cluster_market_regimes(features: np.ndarray, n_clusters: int = 3) -> np.ndarray:
    """
    Cluster market states using K-means.
    
    ALGLIB equivalent: CKMeans
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(features)
    return labels
```

### Optimization (optimization.mqh)

| ALGLIB Class | Python Equivalent | Kinetra Status |
|--------------|-------------------|----------------|
| `CMinCG` - Conjugate gradient | `scipy.optimize.minimize(method='CG')` | ✅ Available |
| `CMinBLEIC` - Linear constraints | `scipy.optimize.minimize(method='SLSQP')` | ✅ For risk constraints |
| `CMinLBFGS` - L-BFGS | `scipy.optimize.minimize(method='L-BFGS-B')` | ✅ Available |
| `CMinQP` - Quadratic programming | `cvxpy.Problem` or `scipy.optimize.quadratic_assignment` | ✅ For portfolio optimization |
| `CMinLM` - Levenberg-Marquardt | `scipy.optimize.least_squares(method='lm')` | ✅ Available |

**Kinetra Usage**:
```python
from scipy.optimize import minimize

def optimize_portfolio_weights(returns: np.ndarray, 
                               risk_aversion: float = 1.0) -> np.ndarray:
    """
    Optimize portfolio weights using mean-variance optimization.
    
    ALGLIB equivalent: CMinQP (quadratic programming)
    """
    n_assets = returns.shape[1]
    
    # Objective: maximize return - risk_aversion * variance
    def objective(weights):
        portfolio_return = np.sum(weights * returns.mean(axis=0))
        portfolio_var = np.dot(weights, np.dot(np.cov(returns.T), weights))
        return -(portfolio_return - risk_aversion * portfolio_var)
    
    # Constraints: weights sum to 1
    constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
    bounds = [(0, 1) for _ in range(n_assets)]
    
    result = minimize(objective, 
                     x0=np.ones(n_assets) / n_assets,
                     method='SLSQP',
                     bounds=bounds,
                     constraints=constraints)
    
    return result.x
```

### Linear Algebra (linalg.mqh)

| ALGLIB Class | Python Equivalent | Kinetra Status |
|--------------|-------------------|----------------|
| `COrtFac` - QR/LQ decomposition | `np.linalg.qr` | ✅ Available |
| `CEigenVDetect` - Eigenvalues/vectors | `np.linalg.eig`, `scipy.linalg.eigh` | ✅ For PCA, correlation analysis |
| `CMatGen` - Random matrices | `np.random.randn()` | ✅ Available |
| `CTrFac` - LU/Cholesky decomposition | `scipy.linalg.lu`, `scipy.linalg.cholesky` | ✅ Available |
| `CRCond` - Condition number | `np.linalg.cond` | ✅ For matrix stability |
| `CMatInv` - Matrix inversion | `np.linalg.inv` | ✅ Available |
| `CBdSingValueDecompose` - SVD | `np.linalg.svd` | ✅ For dimensionality reduction |
| `CMatDet` - Determinant | `np.linalg.det` | ✅ Available |
| `CSchur` - Schur decomposition | `scipy.linalg.schur` | ✅ Available |

**Kinetra Usage**:
```python
# PCA using eigenvalue decomposition (CEigenVDetect)
def perform_pca(data: np.ndarray, n_components: int = 3) -> np.ndarray:
    """
    Principal Component Analysis using eigenvalue decomposition.
    
    ALGLIB equivalent: CEigenVDetect + CPCAnalysis
    """
    # Center data
    data_centered = data - data.mean(axis=0)
    
    # Covariance matrix
    cov_matrix = np.cov(data_centered.T)
    
    # Eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    
    # Sort by eigenvalue (descending)
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # Select top components
    components = eigenvectors[:, :n_components]
    
    # Transform data
    transformed = np.dot(data_centered, components)
    
    return transformed, eigenvalues[:n_components]
```

### Interpolation (interpolation.mqh)

| ALGLIB Class | Python Equivalent | Kinetra Status |
|--------------|-------------------|----------------|
| `CIDWInt` - Inverse distance weighting | `scipy.interpolate.Rbf` | ✅ Available |
| `CRatInt` - Rational interpolation | Custom implementation | ⏳ Can add |
| `CPolInt` - Polynomial interpolation | `np.polyfit`, `scipy.interpolate.BarycentricInterpolator` | ✅ Available |
| `CSpline1D` - 1D spline | `scipy.interpolate.UnivariateSpline` | ✅ For smoothing |
| `CLSFit` - Least squares fit | `scipy.optimize.curve_fit` | ✅ Available |
| `CPSpline` - Parametric spline | `scipy.interpolate.splprep`, `splev` | ✅ Available |
| `CSpline2D` - 2D spline | `scipy.interpolate.RectBivariateSpline` | ✅ Available |

### Fast Transforms (fasttransforms.mqh)

| ALGLIB Class | Python Equivalent | Kinetra Status |
|--------------|-------------------|----------------|
| `CFastFourierTransform` - FFT | `np.fft.fft`, `scipy.fft.fft` | ✅ For frequency analysis |
| `CConv` - Convolution | `np.convolve`, `scipy.signal.convolve` | ✅ For signal processing |
| `CCorr` - Cross-correlation | `np.correlate`, `scipy.signal.correlate` | ✅ For signal similarity |
| `CFastHartleyTransform` - FHT | Custom or `scipy.fft` variants | ⏳ Rarely needed |

**Kinetra Usage**:
```python
from scipy.fft import fft, fftfreq

def detect_market_cycles(prices: np.ndarray, sampling_rate: float = 1.0) -> dict:
    """
    Detect dominant market cycles using FFT.
    
    ALGLIB equivalent: CFastFourierTransform
    """
    # Detrend
    prices_detrended = prices - np.mean(prices)
    
    # FFT
    fft_values = fft(prices_detrended)
    freqs = fftfreq(len(prices), 1/sampling_rate)
    
    # Power spectrum
    power = np.abs(fft_values) ** 2
    
    # Find dominant frequencies (positive only)
    positive_freqs = freqs[:len(freqs)//2]
    positive_power = power[:len(power)//2]
    
    # Top 3 cycles
    top_indices = np.argsort(positive_power)[-3:][::-1]
    dominant_cycles = []
    
    for idx in top_indices:
        if positive_freqs[idx] > 0:
            period = 1 / positive_freqs[idx]
            dominant_cycles.append({
                'period': period,
                'frequency': positive_freqs[idx],
                'power': positive_power[idx],
            })
    
    return {'cycles': dominant_cycles}
```

### Integration (integration.mqh)

| ALGLIB Class | Python Equivalent | Kinetra Status |
|--------------|-------------------|----------------|
| `CGaussQ` - Gaussian quadrature | `scipy.integrate.quadrature` | ✅ Available |
| `CGaussKronrodQ` - Gauss-Kronrod | `scipy.integrate.quad` (uses adaptive quadrature) | ✅ Available |
| `CAutoGK` - Adaptive integrator | `scipy.integrate.quad` | ✅ For option pricing |

### Solvers (solvers.mqh)

| ALGLIB Class | Python Equivalent | Kinetra Status |
|--------------|-------------------|----------------|
| `CDenseSolver` - Linear system solver | `np.linalg.solve` | ✅ Available |
| `CNlEq` - Nonlinear equations | `scipy.optimize.fsolve` | ✅ For equilibrium finding |

### Statistics (statistics.mqh)

| ALGLIB Class | Python Equivalent | Kinetra Status |
|--------------|-------------------|----------------|
| `CBaseStat` - Basic statistics | `np.mean`, `np.std`, `scipy.stats.describe` | ✅ Implemented |
| `CCorrTests` - Correlation tests | `scipy.stats.pearsonr`, `scipy.stats.spearmanr` | ✅ Available |
| `CJarqueBera` - Jarque-Bera test | `scipy.stats.jarque_bera` | ✅ For normality testing |
| `CMannWhitneyU` - Mann-Whitney U | `scipy.stats.mannwhitneyu` | ✅ Non-parametric test |
| `CSignTest` - Sign test | `scipy.stats.binom_test` | ✅ Available |
| `CStudentTests` - t-tests | `scipy.stats.ttest_ind`, `ttest_rel` | ✅ For mean comparison |
| `CVarianceTests` - F-test, χ² | `scipy.stats.f`, `scipy.stats.chi2` | ✅ For variance tests |
| `CWilcoxonSignedRank` - Wilcoxon | `scipy.stats.wilcoxon` | ✅ Non-parametric paired test |

**Kinetra Usage**:
```python
from scipy import stats

def test_strategy_significance(returns_strategy: np.ndarray,
                               returns_benchmark: np.ndarray) -> dict:
    """
    Test if strategy returns are significantly different from benchmark.
    
    ALGLIB equivalent: CStudentTests (t-test)
    """
    # Paired t-test
    t_stat, p_value = stats.ttest_rel(returns_strategy, returns_benchmark)
    
    # Mann-Whitney U (non-parametric alternative)
    u_stat, u_pvalue = stats.mannwhitneyu(returns_strategy, returns_benchmark)
    
    # Jarque-Bera (test normality assumption)
    jb_stat, jb_pvalue = stats.jarque_bera(returns_strategy - returns_benchmark)
    
    return {
        't_test': {
            'statistic': t_stat,
            'p_value': p_value,
            'significant': p_value < 0.05,
        },
        'mann_whitney': {
            'statistic': u_stat,
            'p_value': u_pvalue,
            'significant': u_pvalue < 0.05,
        },
        'jarque_bera': {
            'statistic': jb_stat,
            'p_value': jb_pvalue,
            'is_normal': jb_pvalue > 0.05,
        },
    }
```

### Special Functions (specialfunctions.mqh)

| ALGLIB Class | Python Equivalent | Kinetra Status |
|--------------|-------------------|----------------|
| `CGammaFunc` - Gamma function | `scipy.special.gamma` | ✅ Available |
| `CIncGammaF` - Incomplete gamma | `scipy.special.gammainc` | ✅ For CDFs |
| `CBetaF` - Beta function | `scipy.special.beta` | ✅ Available |
| `CIncBetaF` - Incomplete beta | `scipy.special.betainc` | ✅ For beta distribution |
| `CPsiF` - Psi (digamma) | `scipy.special.digamma` | ✅ Available |
| `CAiryF` - Airy function | `scipy.special.airy` | ✅ Available |
| `CBessel` - Bessel functions | `scipy.special.jn`, `yn` | ✅ For oscillatory processes |
| `CJacobianElliptic` - Jacobian elliptic | `scipy.special.ellipj` | ✅ Available |
| `CDawson` - Dawson integral | `scipy.special.dawsn` | ✅ Available |
| `CTrigIntegrals` - Trig integrals | `scipy.special.sici`, `shichi` | ✅ Available |
| `CElliptic` - Elliptic integrals | `scipy.special.ellipk`, `ellipe` | ✅ Available |
| `CExpIntegrals` - Exponential integrals | `scipy.special.expi` | ✅ Available |
| `CFresnel` - Fresnel integrals | `scipy.special.fresnel` | ✅ Available |
| `CHermite` - Hermite polynomials | `scipy.special.hermite` | ✅ Available |
| `CChebyshev` - Chebyshev polynomials | `scipy.special.chebyt`, `chebyu` | ✅ For approximation |
| `CLaguerre` - Laguerre polynomials | `scipy.special.laguerre` | ✅ Available |
| `CLegendre` - Legendre polynomials | `scipy.special.legendre` | ✅ Available |

### Differential Equations (diffequations.mqh)

| ALGLIB Class | Python Equivalent | Kinetra Status |
|--------------|-------------------|----------------|
| `CODESolver` - ODE solver | `scipy.integrate.odeint`, `solve_ivp` | ✅ For dynamic systems |

**Kinetra Usage**:
```python
from scipy.integrate import odeint

def solve_market_dynamics(initial_state: np.ndarray,
                          time_points: np.ndarray,
                          params: dict) -> np.ndarray:
    """
    Solve market dynamics using ODE.
    
    ALGLIB equivalent: CODESolver
    
    Example: Mean-reverting price model
    dP/dt = -theta * (P - mu) + sigma * dW
    """
    def model(state, t):
        P = state[0]
        theta = params['mean_reversion']
        mu = params['equilibrium']
        
        # Deterministic part (stochastic term handled separately)
        dP_dt = -theta * (P - mu)
        
        return [dP_dt]
    
    solution = odeint(model, initial_state, time_points)
    
    return solution
```

## Python Libraries Summary

Kinetra uses modern Python equivalents for all ALGLIB functionality:

| Category | MT5 (ALGLIB) | Python Libraries |
|----------|--------------|------------------|
| **Numerical Arrays** | `matrix.mqh` | NumPy |
| **Statistics** | `statistics.mqh` | SciPy Stats |
| **Optimization** | `optimization.mqh` | SciPy Optimize, CVXPY |
| **Linear Algebra** | `linalg.mqh` | NumPy, SciPy LinAlg |
| **Machine Learning** | `dataanalysis.mqh` | Scikit-learn, PyTorch |
| **Signal Processing** | `fasttransforms.mqh` | SciPy FFT, SciPy Signal |
| **Interpolation** | `interpolation.mqh` | SciPy Interpolate |
| **Integration** | `integration.mqh` | SciPy Integrate |
| **Special Functions** | `specialfunctions.mqh` | SciPy Special |
| **Differential Equations** | `diffequations.mqh` | SciPy Integrate |
| **Random Numbers** | `alglib.mqh` | NumPy Random, SciPy Stats |

**All ALGLIB functions have Python equivalents!** Python often has more advanced versions with better performance and more features.

## Advantages of Python Stack

1. **More Advanced**: Python libraries are often more sophisticated (e.g., scikit-learn vs ALGLIB ML)
2. **Better Performance**: NumPy/SciPy are highly optimized (BLAS/LAPACK backends)
3. **Larger Ecosystem**: More libraries, more community support
4. **Active Development**: Constant updates and improvements
5. **GPU Support**: PyTorch, CuPy for GPU acceleration

Kinetra is well-positioned with its Python stack! ✅
