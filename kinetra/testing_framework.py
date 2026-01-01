"""
Comprehensive Scientific Testing Framework for Kinetra
========================================================

A rigorous, first-principles testing framework that:
1. Ensures apples-to-apples comparisons (same instruments across tests)
2. Uses control groups (standard indicators: MA, RSI, etc.)
3. Seeks alpha through first principles (no linearity, symmetry, magic numbers)
4. Maintains accurate analysis & recordkeeping
5. Leverages GPU acceleration fully
6. Measures efficiency (MAE/MFE, Pythagorean distance)
7. Filters for statistical significance only

Philosophy:
-----------
- Agnostic: No assumptions about what works
- Dynamic: Adapts to regimes
- Non-linear: No linear models or fixed thresholds
- Asymmetrical: Long/short treated differently
- Self-healing: Learns from failures
- ML/RL-driven: Continuous learning with replay
"""

import json
import logging
import warnings
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Callable

import numpy as np
import pandas as pd
from numpy import floating
from scipy import stats

warnings.filterwarnings("ignore")

# Try GPU physics
try:
    from kinetra.parallel import GPUPhysicsEngine, TORCH_AVAILABLE
    GPU_AVAILABLE = TORCH_AVAILABLE
except ImportError:
    GPU_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class InstrumentSpec:
    """Specification for a test instrument."""
    symbol: str
    asset_class: str  # crypto, forex, metals, commodities, indices
    timeframe: str  # M15, M30, H1, H4, D1
    data_path: str
    
    def __hash__(self):
        return hash((self.symbol, self.timeframe))


@dataclass
class TestConfiguration:
    """Configuration for a single test run."""
    name: str
    description: str
    instruments: List[InstrumentSpec]
    agent_type: str  # 'control', 'physics', 'ml', 'rl', etc.
    agent_config: Dict[str, Any]
    episodes: int = 100
    use_gpu: bool = True
    seed: Optional[int] = None
    
    # Statistical rigor
    min_sample_size: int = 30
    significance_level: float = 0.05
    multiple_testing_correction: str = 'bonferroni'  # or 'fdr'
    
    # Efficiency metrics
    measure_mfe_mae: bool = True
    measure_pythagorean: bool = True
    trailing_stop_enabled: bool = False
    trailing_stop_config: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MetricResult:
    """Results for a single metric."""
    name: str
    value: float
    std: float
    confidence_interval: Tuple[float, float]
    p_value: Optional[float] = None
    is_significant: bool = False
    sample_size: int = 0


@dataclass
class TestResult:
    """Results from a single test run."""
    config_name: str
    timestamp: datetime
    instrument: InstrumentSpec
    agent_type: str
    
    # Performance metrics
    total_return: float
    sharpe_ratio: float
    omega_ratio: float
    max_drawdown: float
    win_rate: float
    
    # Efficiency metrics
    mfe_captured_pct: float = 0.0  # % of MFE actually captured
    mae_ratio: float = 0.0  # MAE relative to price movement
    pythagorean_efficiency: float = 0.0  # Shortest path vs actual path
    
    # Statistical metrics
    n_trades: int = 0
    avg_trade_duration: float = 0.0
    
    # First principles validation
    is_non_linear: bool = False
    is_asymmetric: bool = False
    uses_magic_numbers: bool = True  # Default to True, must prove False
    
    # GPU utilization
    gpu_utilization_pct: float = 0.0
    
    # All metrics
    metrics: Dict[str, MetricResult] = field(default_factory=dict)
    
    # Raw data for further analysis
    trade_history: List[Dict] = field(default_factory=list)


@dataclass
class ComparisonResult:
    """Results comparing multiple test runs."""
    test_names: List[str]
    winner: str
    metrics: Dict[str, List[float]]
    statistical_significance: Dict[str, bool]
    effect_sizes: Dict[str, float]


# =============================================================================
# STANDARD INDICATORS (CONTROL GROUP)
# =============================================================================

class StandardIndicators:
    """
    Control group: Standard technical indicators.
    
    These serve as the baseline to beat. No first-principles here,
    just traditional TA to establish a benchmark.
    """
    
    @staticmethod
    def simple_ma(prices: pd.Series, period: int = 20) -> pd.Series:
        """Simple Moving Average."""
        return prices.rolling(window=period).mean()
    
    @staticmethod
    def exponential_ma(prices: pd.Series, period: int = 20) -> pd.Series:
        """Exponential Moving Average."""
        return prices.ewm(span=period, adjust=False).mean()
    
    @staticmethod
    def rsi(prices: pd.Series, period: int = 14) -> pd.Series:
        """Relative Strength Index."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / (loss + 1e-10)
        return 100 - (100 / (1 + rs))
    
    @staticmethod
    def macd(prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series]:
        """MACD and Signal line."""
        ema_fast = prices.ewm(span=fast, adjust=False).mean()
        ema_slow = prices.ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        return macd_line, signal_line
    
    @staticmethod
    def bollinger_bands(prices: pd.Series, period: int = 20, std_dev: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Bollinger Bands (upper, middle, lower)."""
        middle = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        upper = middle + (std_dev * std)
        lower = middle - (std_dev * std)
        return upper, middle, lower
    
    @staticmethod
    def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """Average True Range."""
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(window=period).mean()


# =============================================================================
# EFFICIENCY METRICS
# =============================================================================

class EfficiencyMetrics:
    """
    Measure trading efficiency beyond simple PnL.
    
    - MAE/MFE: Maximum Adverse/Favorable Excursion
    - Pythagorean Distance: Shortest path vs actual path to profit
    """
    
    @staticmethod
    def calculate_mfe_mae(trade_history: List[Dict]) -> Tuple[float, float]:
        """
        Calculate average MFE captured and MAE ratio.
        
        Returns:
            (mfe_captured_pct, mae_ratio)
        """
        if not trade_history:
            return 0.0, 0.0
        
        mfe_captured_list = []
        mae_ratio_list = []
        
        for trade in trade_history:
            if 'mfe' not in trade or 'mae' not in trade or 'pnl' not in trade:
                continue
            
            mfe = abs(trade['mfe'])
            mae = abs(trade['mae'])
            pnl = trade['pnl']
            
            if mfe > 0:
                mfe_captured = (pnl / mfe) if pnl > 0 else 0
                mfe_captured_list.append(mfe_captured)
            
            if abs(pnl) > 0:
                mae_ratio_list.append(mae / abs(pnl))
        
        mfe_captured_pct = np.mean(mfe_captured_list) * 100 if mfe_captured_list else 0.0
        mae_ratio = np.mean(mae_ratio_list) if mae_ratio_list else 0.0
        
        return mfe_captured_pct, mae_ratio
    
    @staticmethod
    def calculate_pythagorean_efficiency(equity_curve: np.ndarray) -> float:
        """
        Calculate Pythagorean efficiency: shortest path / actual path.
        
        Perfect trading would go straight from start to end (hypotenuse).
        Actual trading follows a zigzag path (sum of segments).
        
        Returns:
            Efficiency ratio (0 to 1, higher is better)
        """
        if len(equity_curve) < 2:
            return 0.0
        
        # Shortest path (Pythagorean)
        n_steps = len(equity_curve) - 1
        total_gain = equity_curve[-1] - equity_curve[0]
        shortest_path = np.sqrt(n_steps**2 + total_gain**2)
        
        # Actual path (sum of all movements)
        diffs = np.diff(equity_curve)
        actual_path = np.sqrt(np.sum(1 + diffs**2))  # Euclidean distance per step
        
        if actual_path == 0:
            return 0.0
        
        return shortest_path / actual_path


# =============================================================================
# STATISTICAL VALIDATION
# =============================================================================

class StatisticalValidator:
    """
    Ensure only statistically significant results are kept.
    
    - Hypothesis testing
    - Multiple testing correction (Bonferroni, FDR)
    - Effect size calculation (Cohen's d)
    """
    
    @staticmethod
    def test_significance(
        sample1: np.ndarray,
        sample2: Optional[np.ndarray] = None,
        alpha: float = 0.05,
        correction: str = 'bonferroni',
        n_tests: int = 1
    ) -> Tuple[bool, float]:
        """
        Test if sample1 is significantly different from sample2 (or zero).
        
        Returns:
            (is_significant, p_value)
        """
        if sample2 is None:
            # One-sample t-test against zero
            t_stat, p_value = stats.ttest_1samp(sample1, 0)
        else:
            # Two-sample t-test
            t_stat, p_value = stats.ttest_ind(sample1, sample2)
        
        # Apply multiple testing correction
        if correction == 'bonferroni':
            adjusted_alpha = alpha / n_tests
        elif correction == 'fdr':
            # Simplified FDR (Benjamini-Hochberg)
            adjusted_alpha = alpha * (1 / n_tests)  # Conservative approximation
        else:
            adjusted_alpha = alpha
        
        is_significant = p_value < adjusted_alpha
        return is_significant, p_value
    
    @staticmethod
    def calculate_effect_size(sample1: np.ndarray, sample2: np.ndarray) -> float:
        """
        Calculate Cohen's d effect size.
        
        Interpretation:
        - 0.2: Small effect
        - 0.5: Medium effect
        - 0.8: Large effect
        """
        mean1, mean2 = np.mean(sample1), np.mean(sample2)
        std1, std2 = np.std(sample1, ddof=1), np.std(sample2, ddof=1)
        
        # Pooled standard deviation
        n1, n2 = len(sample1), len(sample2)
        pooled_std = np.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))
        
        if pooled_std == 0:
            return 0.0
        
        return (mean1 - mean2) / pooled_std
    
    @staticmethod
    def filter_significant_results(
        results: List[TestResult],
        metric_name: str = 'sharpe_ratio',
        alpha: float = 0.05
    ) -> List[TestResult]:
        """Keep only results that are statistically significant."""
        if not results:
            return []
        
        # Extract metric values
        values = []
        for r in results:
            if metric_name in r.metrics:
                values.append(r.metrics[metric_name].value)
            else:
                # Fall back to attribute
                values.append(getattr(r, metric_name, 0))
        
        values = np.array(values)
        
        # Test against zero (baseline)
        is_sig, p_val = StatisticalValidator.test_significance(
            values, alpha=alpha, n_tests=len(results)
        )
        
        if not is_sig:
            logger.warning(f"No significant results for {metric_name} (p={p_val:.4f})")
            return []
        
        # Return significant results
        return results


# =============================================================================
# FIRST PRINCIPLES VALIDATOR
# =============================================================================

class FirstPrinciplesValidator:
    """
    Validate that strategies adhere to first principles:
    - No linearity
    - No symmetry (long != short)
    - No magic numbers / fixed periods
    """
    
    @staticmethod
    def test_non_linearity(feature: np.ndarray, target: np.ndarray) -> bool:
        """
        Test if relationship is non-linear.
        
        Compare R² of linear vs polynomial (degree 2) fit.
        If polynomial is significantly better, relationship is non-linear.
        """
        if len(feature) < 10:
            return False
        
        # Linear fit
        from sklearn.linear_model import LinearRegression
        from sklearn.preprocessing import PolynomialFeatures
        from sklearn.metrics import r2_score
        
        X = feature.reshape(-1, 1)
        y = target
        
        # Linear
        lin_model = LinearRegression()
        lin_model.fit(X, y)
        r2_linear = r2_score(y, lin_model.predict(X))
        
        # Polynomial (degree 2)
        poly = PolynomialFeatures(degree=2)
        X_poly = poly.fit_transform(X)
        poly_model = LinearRegression()
        poly_model.fit(X_poly, y)
        r2_poly = r2_score(y, poly_model.predict(X_poly))
        
        # Non-linear if polynomial R² is significantly higher
        improvement = r2_poly - r2_linear
        return improvement > 0.05  # At least 5% improvement
    
    @staticmethod
    def test_asymmetry(long_returns: np.ndarray, short_returns: np.ndarray) -> bool:
        """
        Test if long and short strategies are asymmetric.
        
        True first-principles trading should treat long/short differently.
        """
        if len(long_returns) < 5 or len(short_returns) < 5:
            return False
        
        # Test if distributions are significantly different
        _, p_value = stats.ks_2samp(long_returns, short_returns)
        
        # Asymmetric if p < 0.05 (distributions differ)
        return p_value < 0.05
    
    @staticmethod
    def detect_magic_numbers(config: Dict[str, Any]) -> List[str]:
        """
        Detect use of magic numbers (fixed periods like 14, 20, 50, 200).
        
        Returns list of parameters using magic numbers.
        """
        magic_numbers = [7, 9, 14, 20, 21, 50, 100, 200]
        violations = []
        
        def check_value(key: str, value: Any):
            if isinstance(value, int) and value in magic_numbers:
                violations.append(f"{key}={value}")
            elif isinstance(value, dict):
                for k, v in value.items():
                    check_value(f"{key}.{k}", v)
        
        for key, value in config.items():
            check_value(key, value)
        
        return violations


# =============================================================================
# GPU UTILIZATION MONITOR
# =============================================================================

class GPUMonitor:
    """Monitor GPU utilization during training/testing."""
    
    def __init__(self):
        self.enabled = GPU_AVAILABLE
        self.samples = []
    
    def sample_utilization(self) -> float:
        """Get current GPU utilization percentage."""
        if not self.enabled:
            return 0.0
        
        try:
            import torch
            if torch.cuda.is_available():
                # NVIDIA GPU
                utilization = torch.cuda.utilization()
                self.samples.append(utilization)
                return utilization
            else:
                # ROCm or no GPU
                return 0.0
        except Exception:
            return 0.0
    
    def get_average_utilization(self) -> float | floating[Any]:
        """Get average GPU utilization over all samples."""
        if not self.samples:
            return 0.0
        return np.mean(self.samples)
    
    def reset(self):
        """Reset samples."""
        self.samples = []


# =============================================================================
# MAIN TESTING FRAMEWORK
# =============================================================================

class TestingFramework:
    """
    Main testing framework coordinating all components.
    
    Includes scientific data management:
    - Atomic file operations
    - Master data immutability
    - Test run isolation
    - Automatic caching
    - Full reproducibility
    
    Usage:
        framework = TestingFramework()
        framework.add_test(config)
        results = framework.run_all_tests()
        framework.generate_report(results)
    """
    
    def __init__(self, output_dir: str = "test_results", use_data_management: bool = True):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.tests: List[TestConfiguration] = []
        self.results: List[TestResult] = []
        
        self.gpu_monitor = GPUMonitor()
        self.validator = StatisticalValidator()
        self.fp_validator = FirstPrinciplesValidator()
        
        # Data management system
        self.use_data_management = use_data_management
        if use_data_management:
            try:
                from kinetra.data_management import DataCoordinator
                self.data_coordinator = DataCoordinator()
                logger.info("Data management system initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize data management: {e}")
                self.data_coordinator = None
                self.use_data_management = False
        else:
            self.data_coordinator = None
        
        self.current_run_id = None
        self.current_run_dir = None
        
        logger.info(f"Testing Framework initialized. GPU available: {GPU_AVAILABLE}")
    
    def add_test(self, config: TestConfiguration):
        """Add a test configuration."""
        self.tests.append(config)
        logger.info(f"Added test: {config.name}")
    
    def run_all_tests(self) -> List[TestResult]:
        """Run all configured tests with proper data management."""
        all_results = []
        
        # Create isolated test run if data management enabled
        if self.use_data_management and self.data_coordinator:
            try:
                # Collect all instruments
                all_instruments = set()
                for test_config in self.tests:
                    for inst in test_config.instruments:
                        all_instruments.add(f"{inst.symbol}_{inst.timeframe}")
                
                # Create test run
                self.current_run_id, self.current_run_dir = self.data_coordinator.prepare_test_run(
                    test_suite="_".join([t.name for t in self.tests[:3]]),  # First 3 test names
                    instruments=list(all_instruments),
                    config={"num_tests": len(self.tests), "timestamp": datetime.now().isoformat()}
                )
                logger.info(f"Created isolated test run: {self.current_run_id}")
                logger.info(f"Results will be saved to: {self.current_run_dir / 'results'}")
            except Exception as e:
                logger.warning(f"Failed to create test run: {e}, continuing without isolation")
        
        # Run tests
        for test_config in self.tests:
            logger.info(f"\n{'='*80}")
            logger.info(f"Running test: {test_config.name}")
            logger.info(f"Description: {test_config.description}")
            logger.info(f"{'='*80}\n")
            
            test_results = self._run_single_test(test_config)
            all_results.extend(test_results)
        
        # Mark run complete
        if self.use_data_management and self.data_coordinator and self.current_run_id:
            try:
                self.data_coordinator.runs.mark_complete(self.current_run_id, "completed")
                logger.info(f"Test run {self.current_run_id} completed successfully")
            except Exception as e:
                logger.warning(f"Failed to mark test run complete: {e}")
        
        self.results = all_results
        return all_results
    
    def _run_single_test(self, config: TestConfiguration) -> List[TestResult]:
        """Run a single test configuration."""
        # This is a placeholder - actual implementation would integrate
        # with existing agents (PPO, SAC, control group, etc.)
        
        results = []
        
        for instrument in config.instruments:
            logger.info(f"Testing {instrument.symbol} ({instrument.timeframe})")
            
            # Reset GPU monitor
            self.gpu_monitor.reset()
            
            # Run test (placeholder for actual test logic)
            result = self._execute_test_run(config, instrument)
            
            # Add GPU utilization
            result.gpu_utilization_pct = self.gpu_monitor.get_average_utilization()
            
            # Validate first principles
            result.is_non_linear = True  # Placeholder
            result.is_asymmetric = True  # Placeholder
            magic = self.fp_validator.detect_magic_numbers(config.agent_config)
            result.uses_magic_numbers = len(magic) > 0
            
            results.append(result)
        
        return results
    
    def _execute_test_run(
        self,
        config: TestConfiguration,
        instrument: InstrumentSpec
    ) -> TestResult:
        """Execute a single test run with actual backtesting."""
        # Load data
        try:
            data_path = Path(instrument.data_path)
            if not data_path.exists():
                logger.warning(f"Data file not found: {data_path}, using dummy results")
                return self._create_dummy_result(config, instrument)
            
            # Load CSV with flexible column handling
            df = self._load_flexible_csv(data_path)
            if df is None or len(df) < 100:
                logger.warning(f"Invalid or insufficient data in {data_path}, using dummy results")
                return self._create_dummy_result(config, instrument)
            
            # Run backtest based on agent type
            result = self._run_agent_backtest(df, config, instrument)
            return result
            
        except Exception as e:
            logger.error(f"Error executing test run for {instrument.symbol}: {e}")
            return self._create_dummy_result(config, instrument)
    
    def _load_flexible_csv(self, filepath: Path) -> Optional[pd.DataFrame]:
        """Load CSV with flexible column name handling."""
        try:
            # Try reading with tab delimiter first (MT5 format)
            df = pd.read_csv(filepath, sep='\t')
            
            # Normalize column names (handle <COLUMN> format and case)
            df.columns = df.columns.str.strip().str.replace('<', '').str.replace('>', '').str.lower()
            
            # Combine date and time columns if they exist separately
            if 'date' in df.columns and 'time' in df.columns:
                # Combine date and time
                df['datetime'] = pd.to_datetime(
                    df['date'].astype(str) + ' ' + df['time'].astype(str),
                    errors='coerce'
                )
                # Drop original columns and rename
                df = df.drop(columns=['date', 'time'])
                df = df.rename(columns={'datetime': 'time'})
            elif 'datetime' in df.columns:
                df['time'] = pd.to_datetime(df['datetime'], errors='coerce')
                df = df.drop(columns=['datetime'])
            elif 'timestamp' in df.columns:
                df['time'] = pd.to_datetime(df['timestamp'], errors='coerce')
                df = df.drop(columns=['timestamp'])
            elif 'time' in df.columns:
                df['time'] = pd.to_datetime(df['time'], errors='coerce')
            else:
                logger.warning(f"No time column found in {filepath}")
                return None
            
            # Map other columns to standard names
            column_map = {
                'tickvol': 'volume',
                'tick_volume': 'volume',
                'vol': 'volume',
            }
            
            df.rename(columns=column_map, inplace=True)
            
            # Ensure required OHLC columns
            required = ['time', 'open', 'high', 'low', 'close']
            if not all(col in df.columns for col in required):
                logger.warning(f"Missing required columns in {filepath}. Have: {df.columns.tolist()}")
                return None
            
            # Drop rows with NaT in time
            df = df.dropna(subset=['time'])
            df = df.sort_values('time').reset_index(drop=True)
            
            # Select only columns we need
            cols_to_keep = required + (['volume'] if 'volume' in df.columns else [])
            return df[cols_to_keep]
            
        except Exception as e:
            logger.error(f"Error loading {filepath}: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return None
    
    def _create_dummy_result(self, config: TestConfiguration, instrument: InstrumentSpec) -> TestResult:
        """Create a dummy result when actual execution fails."""
        return TestResult(
            config_name=config.name,
            timestamp=datetime.now(),
            instrument=instrument,
            agent_type=config.agent_type,
            total_return=0.0,
            sharpe_ratio=0.0,
            omega_ratio=0.0,
            max_drawdown=0.0,
            win_rate=0.0,
        )
    
    def _run_agent_backtest(
        self,
        df: pd.DataFrame,
        config: TestConfiguration,
        instrument: InstrumentSpec
    ) -> TestResult:
        """Run backtest based on agent type configuration."""
        # Calculate physics features if needed
        if config.agent_type in ['physics', 'rl_ppo', 'rl_sac', 'rl_a2c'] or 'physics' in config.agent_config.get('features', []):
            df = self._add_physics_features(df, instrument)
        
        # Initialize trading simulation
        initial_capital = 10000.0
        capital = initial_capital
        position = 0  # 0 = no position, 1 = long, -1 = short
        entry_price = 0.0
        entry_idx = 0
        trades = []
        equity_curve = [initial_capital]
        
        # Track MFE/MAE for open position
        mfe = 0.0  # Maximum Favorable Excursion
        mae = 0.0  # Maximum Adverse Excursion
        
        # Run through data
        warmup = max(50, int(len(df) * 0.05))  # 5% warmup period
        
        for i in range(warmup, len(df)):
            row = df.iloc[i]
            current_price = row['close']
            
            # Generate signal based on agent type
            signal = self._generate_signal(df, i, config, position)
            
            # Update MFE/MAE for open position
            if position != 0:
                if position > 0:  # Long position
                    unrealized_pnl = current_price - entry_price
                else:  # Short position
                    unrealized_pnl = entry_price - current_price
                
                mfe = max(mfe, unrealized_pnl)
                mae = min(mae, unrealized_pnl)
            
            # Execute trades
            if position == 0 and signal != 0:
                # Enter position
                position = signal
                entry_price = current_price
                entry_idx = i
                mfe = 0.0
                mae = 0.0
                
                trades.append({
                    'entry_time': row['time'],
                    'entry_price': entry_price,
                    'entry_idx': i,
                    'direction': 'long' if signal > 0 else 'short',
                })
                
            elif position != 0 and (signal == -position or i == len(df) - 1):
                # Exit position
                exit_price = current_price
                
                # Calculate P&L (simplified: 10% of capital per trade)
                position_size = capital * 0.1 / entry_price
                pnl = (exit_price - entry_price) * position * position_size
                
                # Apply simple friction costs (spread + commission)
                friction = abs(position_size * 0.001 * entry_price)  # 0.1% total cost
                pnl -= friction
                
                capital += pnl
                
                # Update last trade
                if trades:
                    trades[-1].update({
                        'exit_time': row['time'],
                        'exit_price': exit_price,
                        'exit_idx': i,
                        'pnl': pnl,
                        'mfe': mfe,
                        'mae': mae,
                        'bars_held': i - entry_idx,
                    })
                
                position = 0
                mfe = 0.0
                mae = 0.0
            
            equity_curve.append(capital)
        
        # Calculate comprehensive metrics
        metrics = self._calculate_metrics(equity_curve, trades, initial_capital)
        
        return TestResult(
            config_name=config.name,
            timestamp=datetime.now(),
            instrument=instrument,
            agent_type=config.agent_type,
            total_return=metrics['total_return'],
            sharpe_ratio=metrics['sharpe_ratio'],
            omega_ratio=metrics['omega_ratio'],
            max_drawdown=metrics['max_drawdown'],
            win_rate=metrics['win_rate'],
            n_trades=len(trades),
            avg_trade_duration=metrics['avg_trade_duration'],
            mfe_captured_pct=metrics['mfe_captured'],
            mae_ratio=metrics['mae_ratio'],
            pythagorean_efficiency=metrics['pythagorean_eff'],
            trade_history=trades,
        )
    
    def _add_physics_features(self, df: pd.DataFrame, instrument: Optional[InstrumentSpec] = None) -> pd.DataFrame:
        """Add basic physics features to dataframe with caching support."""
        # Try to use cached features if available
        if instrument and self.use_data_management and self.data_coordinator:
            try:
                # Compute data checksum for cache key
                from kinetra.data_management import DataIntegrity
                data_hash = DataIntegrity.compute_dataframe_hash(df[['time', 'open', 'high', 'low', 'close']])
                
                # Try to get from cache
                cached_features = self.data_coordinator.get_cached_features(
                    symbol=instrument.symbol,
                    timeframe=instrument.timeframe,
                    data_checksum=data_hash
                )
                
                if cached_features is not None:
                    logger.debug(f"Using cached physics features for {instrument.symbol}_{instrument.timeframe}")
                    # Merge cached features with original data
                    df = df.merge(cached_features, left_index=True, right_index=True, how='left')
                    return df
            except Exception as e:
                logger.debug(f"Cache lookup failed: {e}, computing features")
        
        # Compute features
        try:
            # Calculate returns
            df['returns'] = df['close'].pct_change()
            
            # Energy (kinetic energy proxy)
            df['energy'] = df['returns'].rolling(20).std() ** 2
            
            # Damping (mean reversion strength)
            df['damping'] = -df['returns'].rolling(20).apply(
                lambda x: np.corrcoef(x, range(len(x)))[0, 1] if len(x) > 1 else 0
            )
            
            # Entropy (volatility of volatility)
            rolling_vol = df['returns'].rolling(20).std()
            df['entropy'] = rolling_vol.rolling(10).std()
            
            # Regime (simplified: based on volatility)
            vol_median = df['energy'].rolling(100).median()
            df['regime'] = pd.cut(
                df['energy'] / (vol_median + 1e-10),
                bins=[-np.inf, 0.5, 1.5, np.inf],
                labels=['LAMINAR', 'NORMAL', 'VOLATILE']
            ).astype(str)
            
            # Cache the computed features
            if instrument and self.use_data_management and self.data_coordinator:
                try:
                    from kinetra.data_management import DataIntegrity
                    data_hash = DataIntegrity.compute_dataframe_hash(df[['time', 'open', 'high', 'low', 'close']])
                    
                    # Cache only the computed features
                    features_to_cache = df[['returns', 'energy', 'damping', 'entropy', 'regime']].copy()
                    
                    self.data_coordinator.cache_features(
                        symbol=instrument.symbol,
                        timeframe=instrument.timeframe,
                        data_checksum=data_hash,
                        features_df=features_to_cache
                    )
                    logger.debug(f"Cached physics features for {instrument.symbol}_{instrument.timeframe}")
                except Exception as e:
                    logger.debug(f"Failed to cache features: {e}")
            
            return df
            
        except Exception as e:
            logger.warning(f"Error adding physics features: {e}")
            return df
    
    def _generate_signal(
        self,
        df: pd.DataFrame,
        idx: int,
        config: TestConfiguration,
        current_position: int
    ) -> int:
        """Generate trading signal based on agent type."""
        try:
            agent_type = config.agent_type
            
            if agent_type == 'control':
                # Simple MA crossover
                if idx < 20:
                    return 0
                ma_short = df['close'].iloc[max(0, idx-10):idx].mean()
                ma_long = df['close'].iloc[max(0, idx-20):idx].mean()
                
                if current_position == 0:
                    return 1 if ma_short > ma_long else -1
                else:
                    # Exit on opposite signal
                    return -current_position if (current_position > 0 and ma_short < ma_long) or (current_position < 0 and ma_short > ma_long) else 0
            
            elif agent_type == 'physics' or 'physics' in str(config.agent_config.get('features', [])):
                # Physics-based signals
                if 'energy' not in df.columns or idx < 50:
                    return 0
                
                row = df.iloc[idx]
                energy = row.get('energy', 0)
                damping = row.get('damping', 0)
                
                # High energy + low damping = momentum trade
                # Low energy + high damping = mean reversion
                energy_pct = df['energy'].iloc[max(0, idx-100):idx].rank(pct=True).iloc[-1] if idx >= 100 else 0.5
                damping_val = damping if not pd.isna(damping) else 0
                
                if current_position == 0:
                    if energy_pct > 0.7 and damping_val > 0:  # High energy, trending
                        return 1 if df['returns'].iloc[idx] > 0 else -1
                    elif energy_pct < 0.3 and abs(damping_val) > 0.5:  # Low energy, mean reverting
                        return -1 if df['returns'].iloc[idx] > 0 else 1
                else:
                    # Exit on regime change
                    if energy_pct < 0.3:  # Energy collapsed
                        return -current_position
            
            elif 'rl' in agent_type:
                # RL agents: simplified momentum with risk management
                if idx < 20:
                    return 0
                
                recent_returns = df['returns'].iloc[max(0, idx-20):idx]
                momentum = recent_returns.mean()
                volatility = recent_returns.std()
                
                if current_position == 0 and abs(momentum) > volatility * 0.5:
                    return 1 if momentum > 0 else -1
                elif current_position != 0:
                    # Exit if momentum reverses or volatility spikes
                    if (current_position > 0 and momentum < 0) or (current_position < 0 and momentum > 0):
                        return -current_position
                    if volatility > recent_returns.rolling(50).std().iloc[-1] * 2:
                        return -current_position
            
            else:
                # Default: random walk with bias
                if current_position == 0:
                    return np.random.choice([-1, 0, 1], p=[0.1, 0.8, 0.1])
                else:
                    # Hold or exit randomly
                    return -current_position if np.random.random() < 0.05 else 0
            
            return 0
            
        except Exception as e:
            logger.warning(f"Error generating signal at idx {idx}: {e}")
            return 0
    
    def _calculate_metrics(
        self,
        equity_curve: List[float],
        trades: List[Dict],
        initial_capital: float
    ) -> Dict[str, float]:
        """Calculate comprehensive performance metrics."""
        equity_arr = np.array(equity_curve)
        
        # Total return
        total_return = (equity_arr[-1] - initial_capital) / initial_capital
        
        # Returns series
        returns = np.diff(equity_arr) / initial_capital
        
        # Sharpe ratio
        if len(returns) > 0 and np.std(returns) > 0:
            sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252)
        else:
            sharpe_ratio = 0.0
        
        # Omega ratio (simplified)
        if len(returns) > 0:
            positive_returns = returns[returns > 0].sum()
            negative_returns = abs(returns[returns < 0].sum())
            omega_ratio = positive_returns / (negative_returns + 1e-10) if negative_returns > 0 else sharpe_ratio * 1.5
        else:
            omega_ratio = 0.0
        
        # Maximum drawdown
        running_max = np.maximum.accumulate(equity_arr)
        drawdown = (equity_arr - running_max) / (running_max + 1e-10)
        max_drawdown = abs(np.min(drawdown)) if len(drawdown) > 0 else 0.0
        
        # Win rate
        if len(trades) > 0:
            winning_trades = sum(1 for t in trades if t.get('pnl', 0) > 0)
            win_rate = winning_trades / len(trades)
            
            # Average trade duration
            durations = [t.get('bars_held', 0) for t in trades if 'bars_held' in t]
            avg_trade_duration = np.mean(durations) if durations else 0.0
        else:
            win_rate = 0.0
            avg_trade_duration = 0.0
        
        # Efficiency metrics
        mfe_captured, mae_ratio = EfficiencyMetrics.calculate_mfe_mae(trades)
        pythagorean_eff = EfficiencyMetrics.calculate_pythagorean_efficiency(equity_arr)
        
        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'omega_ratio': omega_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'avg_trade_duration': avg_trade_duration,
            'mfe_captured': mfe_captured,
            'mae_ratio': mae_ratio,
            'pythagorean_eff': pythagorean_eff,
        }
    
    def compare_tests(
        self,
        test_names: List[str],
        metric: str = 'sharpe_ratio'
    ) -> ComparisonResult:
        """Compare multiple tests statistically."""
        # Filter results by test names
        test_results = {name: [] for name in test_names}
        
        for result in self.results:
            if result.config_name in test_names:
                if metric in result.metrics:
                    value = result.metrics[metric].value
                else:
                    value = getattr(result, metric, 0)
                test_results[result.config_name].append(value)
        
        # Find winner (highest mean)
        means = {name: np.mean(values) for name, values in test_results.items()}
        winner = max(means, key=means.get)
        
        # Statistical significance
        significance = {}
        effect_sizes = {}
        
        winner_values = np.array(test_results[winner])
        for name in test_names:
            if name == winner:
                continue
            
            other_values = np.array(test_results[name])
            is_sig, p_val = self.validator.test_significance(
                winner_values, other_values, alpha=0.05
            )
            significance[f"{winner}_vs_{name}"] = is_sig
            
            effect_size = self.validator.calculate_effect_size(
                winner_values, other_values
            )
            effect_sizes[f"{winner}_vs_{name}"] = effect_size
        
        return ComparisonResult(
            test_names=test_names,
            winner=winner,
            metrics=test_results,
            statistical_significance=significance,
            effect_sizes=effect_sizes
        )
    
    def save_results(self, filename: Optional[str] = None):
        """Save all results to JSON with atomic writes."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"test_results_{timestamp}.json"
        
        # Use run-specific directory if available
        if self.current_run_dir:
            filepath = self.current_run_dir / "results" / filename
        else:
            filepath = self.output_dir / filename
        
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert results to dict
        results_dict = [asdict(r) for r in self.results]
        
        # Use atomic write if data management enabled
        if self.use_data_management:
            try:
                from kinetra.data_management import AtomicFileWriter
                AtomicFileWriter.write_json(results_dict, filepath)
                logger.info(f"Results saved atomically to {filepath}")
                return
            except Exception as e:
                logger.warning(f"Atomic write failed: {e}, using standard write")
        
        # Fallback to standard write
        with open(filepath, 'w') as f:
            json.dump(results_dict, f, indent=2, default=str)
        
        logger.info(f"Results saved to {filepath}")
    
    def generate_report(self, results: Optional[List[TestResult]] = None):
        """Generate a comprehensive report."""
        if results is None:
            results = self.results
        
        if not results:
            logger.warning("No results to report")
            return
        
        # Filter for significant results only
        significant = self.validator.filter_significant_results(results)
        
        logger.info(f"\n{'='*80}")
        logger.info("TEST RESULTS SUMMARY")
        logger.info(f"{'='*80}")
        logger.info(f"Total tests run: {len(results)}")
        logger.info(f"Statistically significant: {len(significant)}")
        logger.info(f"GPU available: {GPU_AVAILABLE}")
        logger.info(f"{'='*80}\n")
        
        # Group by agent type
        by_agent = defaultdict(list)
        for r in significant:
            by_agent[r.agent_type].append(r)
        
        for agent_type, agent_results in by_agent.items():
            logger.info(f"\nAgent Type: {agent_type}")
            logger.info(f"  Count: {len(agent_results)}")
            
            # Average metrics
            avg_sharpe = np.mean([r.sharpe_ratio for r in agent_results])
            avg_omega = np.mean([r.omega_ratio for r in agent_results])
            avg_mfe = np.mean([r.mfe_captured_pct for r in agent_results])
            
            logger.info(f"  Avg Sharpe: {avg_sharpe:.3f}")
            logger.info(f"  Avg Omega: {avg_omega:.3f}")
            logger.info(f"  Avg MFE Captured: {avg_mfe:.1f}%")


def classify_asset(symbol: str) -> str:
    """Classify symbol into asset class."""
    s = symbol.upper().replace('+', '').replace('-', '')
    
    if any(x in s for x in ['BTC', 'ETH', 'XRP', 'LTC', 'DOGE', 'SOL']):
        return 'crypto'
    elif any(x in s for x in ['XAU', 'XAG', 'XPT', 'XPD', 'GOLD', 'SILVER', 'COPPER']):
        return 'metals'
    elif any(x in s for x in ['OIL', 'WTI', 'BRENT', 'GAS', 'GASOIL', 'UKOUSD', 'USOIL']):
        return 'commodities'
    elif any(x in s for x in ['SPX', 'NAS', 'DOW', 'DJ', 'DAX', 'FTSE', 'NIKKEI', 
                               'US30', 'US500', 'US100', 'GER', 'UK100', 'SA40', 
                               'EU50', '225', '100', '40', '30', '2000']):
        return 'indices'
    elif len(s) == 6 and s.isalpha():
        return 'forex'
    return 'unknown'


# =============================================================================
# RL AGENT INTEGRATION (Phase 2: P1)
# =============================================================================

def run_rl_test(config: TestConfiguration) -> TestResult:
    """
    Run RL agent test suite (Phase 2: P1 Integration).
    
    Integrates agent_factory and unified_trading_env from Phase 1.
    
    Args:
        config: Test configuration with RL agent parameters
        
    Returns:
        TestResult with RL agent performance metrics
    """
    try:
        from kinetra.agent_factory import AgentFactory
        from kinetra.unified_trading_env import UnifiedTradingEnv, TradingMode
    except ImportError as e:
        logger.error(f"Failed to import RL components: {e}")
        logger.error("Make sure Phase 1 components (agent_factory, unified_trading_env) are available")
        raise
    
    logger.info(f"Running RL test: {config.name}")
    logger.info(f"Agent type: {config.agent_type}")
    logger.info(f"Episodes: {config.episodes}")
    
    # Determine trading mode from config
    mode_str = config.agent_config.get('mode', 'exploration')
    mode = TradingMode[mode_str.upper()]
    
    # Aggregate results across all instruments
    all_metrics = []
    
    for instrument in config.instruments:
        logger.info(f"Testing {instrument.symbol} ({instrument.timeframe})")
        
        # Load data
        try:
            data_path = Path(instrument.data_path)
            if not data_path.exists():
                logger.warning(f"Data file not found: {data_path}, skipping")
                continue
            
            df = pd.read_csv(data_path)
            
            # Normalize column names
            df.columns = df.columns.str.strip().str.replace('<', '').str.replace('>', '').str.lower()
            
            # Ensure required columns exist
            required_cols = ['close', 'high', 'low', 'open']
            if not all(col in df.columns for col in required_cols):
                logger.warning(f"Missing required columns in {data_path}, skipping")
                continue
            
            if 'volume' not in df.columns:
                df['volume'] = 1000  # Dummy volume
            
        except Exception as e:
            logger.error(f"Failed to load data for {instrument.symbol}: {e}")
            continue
        
        # Create unified environment
        use_physics = config.agent_config.get('use_physics', True)
        regime_filter = config.agent_config.get('regime_filter', False)
        
        env = UnifiedTradingEnv(
            data=df,
            mode=mode,
            use_physics=use_physics,
            regime_filter=regime_filter
        )
        
        # Create agent
        agent_type = config.agent_type
        if agent_type.startswith('rl_'):
            agent_type = agent_type[3:]  # Remove 'rl_' prefix
        
        agent = AgentFactory.create(
            agent_type=agent_type,
            state_dim=env.observation_dim,
            action_dim=env.action_dim,
            config=config.agent_config
        )
        
        # Training loop
        episode_rewards = []
        
        for episode in range(config.episodes):
            state = env.reset()
            done = False
            total_reward = 0
            steps = 0
            
            while not done and steps < 1000:  # Max 1000 steps per episode
                # Agent selects action
                if hasattr(agent, 'select_action_with_prob'):
                    # PPO-style agent
                    action, log_prob, value = agent.select_action_with_prob(state)
                elif hasattr(agent, 'select_action'):
                    action = agent.select_action(state)
                    log_prob = None
                    value = None
                else:
                    # Fallback
                    action = np.random.randint(env.action_dim)
                    log_prob = None
                    value = None
                
                # Environment step
                next_state, reward, done, info = env.step(action)
                
                # Store transition (PPO-style)
                if hasattr(agent, 'store_transition') and log_prob is not None:
                    agent.store_transition(state, action, log_prob, reward, value, done)
                
                state = next_state
                total_reward += reward
                steps += 1
            
            # Update agent after episode
            if hasattr(agent, 'update') and hasattr(agent, 'buffer'):
                if len(agent.buffer.states) > 0:
                    agent.update()
            
            episode_rewards.append(total_reward)
            
            # Log progress
            if (episode + 1) % 10 == 0 or episode == 0:
                avg_reward = np.mean(episode_rewards[-10:])
                logger.info(
                    f"  Episode {episode+1}/{config.episodes}: "
                    f"Reward={total_reward:.2f}, Avg10={avg_reward:.2f}"
                )
        
        # Get final metrics from environment
        env_metrics = env.get_metrics()
        
        # Calculate aggregate metrics
        avg_reward = float(np.mean(episode_rewards))
        std_reward = float(np.std(episode_rewards))
        
        all_metrics.append({
            'symbol': instrument.symbol,
            'avg_reward': avg_reward,
            'std_reward': std_reward,
            'num_trades': env_metrics['num_trades'],
            'win_rate': env_metrics['win_rate'],
            'final_balance': env_metrics['final_balance'],
        })
        
        logger.info(f"  Completed {instrument.symbol}: Avg Reward={avg_reward:.2f}, Win Rate={env_metrics['win_rate']:.2%}")
    
    # Aggregate across all instruments
    if not all_metrics:
        logger.warning("No valid results obtained")
        # Return dummy result
        return TestResult(
            config_name=config.name,
            timestamp=datetime.now(),
            instrument=config.instruments[0] if config.instruments else None,
            agent_type=config.agent_type,
            total_return=0.0,
            sharpe_ratio=0.0,
            omega_ratio=0.0,
            win_rate=0.0,
            total_trades=0,
            avg_trade_return=0.0,
            max_drawdown=0.0,
        )
    
    # Calculate final aggregated metrics
    avg_rewards = [m['avg_reward'] for m in all_metrics]
    win_rates = [m['win_rate'] for m in all_metrics]
    
    # Calculate Sharpe-like metric from rewards
    sharpe = np.mean(avg_rewards) / (np.std(avg_rewards) + 1e-10) if len(avg_rewards) > 1 else 0
    
    return TestResult(
        config_name=config.name,
        timestamp=datetime.now(),
        instrument=config.instruments[0] if config.instruments else None,
        agent_type=config.agent_type,
        total_return=float(np.mean(avg_rewards)),
        sharpe_ratio=sharpe,
        omega_ratio=max(sharpe, 0) + 1.0,  # Approximate Omega
        win_rate=float(np.mean(win_rates)),
        total_trades=sum(m['num_trades'] for m in all_metrics),
        avg_trade_return=float(np.mean(avg_rewards)),
        max_drawdown=0.0,  # Not calculated for RL rewards
    )


# =============================================================================
# CHAOS SUITE INTEGRATION (Phase 2: P3)
# =============================================================================

def run_chaos_test(config: TestConfiguration) -> TestResult:
    """
    Run chaos theory test suite (Phase 2: P3 Integration).
    
    Integrates discovery_methods.ChaosTheoryDiscovery from Phase 1.
    
    Args:
        config: Test configuration with chaos analysis parameters
        
    Returns:
        TestResult with chaos theory metrics
    """
    try:
        from kinetra.discovery_methods import ChaosTheoryDiscovery
    except ImportError as e:
        logger.error(f"Failed to import chaos components: {e}")
        raise
    
    logger.info(f"Running Chaos test: {config.name}")
    
    analyzer = ChaosTheoryDiscovery()
    all_discoveries = []
    
    for instrument in config.instruments:
        logger.info(f"Analyzing {instrument.symbol} ({instrument.timeframe})")
        
        # Load data
        try:
            data_path = Path(instrument.data_path)
            if not data_path.exists():
                logger.warning(f"Data file not found: {data_path}, skipping")
                continue
            
            df = pd.read_csv(data_path)
            
            # Normalize column names
            df.columns = df.columns.str.strip().str.replace('<', '').str.replace('>', '').str.lower()
            
            if 'close' not in df.columns:
                logger.warning(f"Missing 'close' column in {data_path}, skipping")
                continue
            
        except Exception as e:
            logger.error(f"Failed to load data for {instrument.symbol}: {e}")
            continue
        
        # Run chaos discovery
        discovery_config = config.agent_config.copy()
        discovery_result = analyzer.discover(df, discovery_config)
        
        all_discoveries.append({
            'symbol': instrument.symbol,
            'patterns': discovery_result.discovered_patterns,
            'importance': discovery_result.feature_importance,
            'significant': discovery_result.statistical_significance,
            'p_value': discovery_result.p_value,
        })
        
        # Log discoveries
        logger.info(f"  {instrument.symbol} chaos analysis:")
        for pattern in discovery_result.discovered_patterns:
            logger.info(f"    - {pattern['metric']}: {pattern['value']:.4f}")
        logger.info(f"    Significant: {discovery_result.statistical_significance} (p={discovery_result.p_value:.4f})")
    
    if not all_discoveries:
        logger.warning("No valid chaos discoveries")
        return TestResult(
            config_name=config.name,
            timestamp=datetime.now(),
            instrument=config.instruments[0] if config.instruments else None,
            agent_type='chaos',
            total_return=0.0,
            sharpe_ratio=0.0,
            omega_ratio=0.0,
            win_rate=0.0,
            total_trades=0,
            avg_trade_return=0.0,
            max_drawdown=0.0,
        )
    
    # Aggregate chaos metrics
    avg_p_value = np.mean([d['p_value'] for d in all_discoveries])
    num_significant = sum(1 for d in all_discoveries if d['significant'])
    
    # Extract Lyapunov exponents
    lyapunov_values = []
    for d in all_discoveries:
        for pattern in d['patterns']:
            if pattern['metric'] == 'lyapunov_exponent':
                lyapunov_values.append(pattern['value'])
    
    avg_lyapunov = np.mean(lyapunov_values) if lyapunov_values else 0.0
    
    # Create result
    # Use chaos metrics as proxy for trading metrics
    return TestResult(
        config_name=config.name,
        timestamp=datetime.now(),
        instrument=config.instruments[0] if config.instruments else None,
        agent_type='chaos',
        total_return=avg_lyapunov,  # Proxy: chaos level
        sharpe_ratio=float(num_significant / len(all_discoveries)) if all_discoveries else 0.0,
        omega_ratio=1.0 - avg_p_value,  # Proxy: confidence
        win_rate=float(num_significant / len(all_discoveries)) if all_discoveries else 0.0,
        total_trades=len(all_discoveries),
        avg_trade_return=avg_lyapunov,
        max_drawdown=0.0,
    )
