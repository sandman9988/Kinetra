"""
Physics-Based Backtester using Backtesting.py

Empirically tests thermodynamics/physics/kinematics trading strategies:
- Energy-based momentum trading (kinetic energy)
- Damping-based mean reversion (friction/resistance)
- Entropy-based volatility trading (disorder/uncertainty)
- Regime-adaptive multi-physics strategies
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, Any, List, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
from backtesting import Backtest, Strategy
from backtesting.lib import crossover

from .config import MAX_WORKERS
from .physics_engine import PhysicsEngine, RegimeType

# Try GPU physics (ROCm/CUDA)
try:
    from .parallel import GPUPhysicsEngine, TORCH_AVAILABLE
    GPU_AVAILABLE = TORCH_AVAILABLE
except ImportError:
    GPU_AVAILABLE = False


# =============================================================================
# PHYSICS INDICATOR FUNCTIONS (for use as Backtesting.py indicators)
# Backtesting.py passes numpy arrays, so we convert to pandas for calculations
# =============================================================================

def compute_kinetic_energy(close, mass: float = 1.0):
    """
    Calculate kinetic energy from price momentum.
    E_t = 0.5 * m * (ΔP_t / Δt)²
    """
    close = pd.Series(close)
    velocity = close.diff()
    energy = 0.5 * mass * velocity ** 2
    return energy.fillna(0.0).values


def compute_damping(close, lookback: int = 20):
    """
    Calculate damping coefficient (market friction).
    ζ = rolling_std(returns) / rolling_mean(|returns|)
    """
    close = pd.Series(close)
    returns = close.pct_change()
    volatility = returns.rolling(lookback).std()
    mean_abs_return = returns.abs().rolling(lookback).mean()
    damping = volatility / (mean_abs_return + 1e-10)
    result = damping.fillna(damping.mean()).clip(lower=0.0)
    return result.values


def compute_entropy(close, lookback: int = 20, bins: int = 10):
    """
    Calculate Shannon entropy of price distribution.
    H = -Σ p_i * log(p_i)
    """
    close = pd.Series(close)
    returns = close.pct_change()

    def rolling_entropy(window):
        window = np.array(window)
        window = window[~np.isnan(window)]
        if len(window) < 5:
            return 0.0
        counts, _ = np.histogram(window, bins=bins)
        total = counts.sum()
        if total == 0:
            return 0.0
        probs = counts / total
        probs = probs[probs > 0]
        return -np.sum(probs * np.log(probs)) if len(probs) > 0 else 0.0

    entropy = returns.rolling(lookback).apply(rolling_entropy, raw=True)
    return entropy.fillna(0.0).clip(lower=0.0).values


def compute_momentum(close, lookback: int = 14):
    """Simple momentum: current price / price n periods ago."""
    close = pd.Series(close)
    result = close / close.shift(lookback)
    return result.fillna(1.0).values


def compute_velocity(close):
    """Price velocity (rate of change)."""
    close = pd.Series(close)
    return close.diff().fillna(0.0).values


def compute_acceleration(close):
    """Price acceleration (second derivative)."""
    close = pd.Series(close)
    return close.diff().diff().fillna(0.0).values


def compute_atr(high, low, close, period: int = 14):
    """Average True Range for volatility normalization."""
    high = pd.Series(high)
    low = pd.Series(low)
    close = pd.Series(close)
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    result = tr.rolling(period).mean()
    return result.fillna(result.mean()).values


def compute_ewm(data, span: int):
    """Exponential weighted moving average."""
    s = pd.Series(data)
    return s.ewm(span=span).mean().values


def compute_sma(data, period: int):
    """Simple moving average."""
    s = pd.Series(data)
    return s.rolling(period).mean().fillna(s.mean()).values


def compute_rolling_std(data, period: int):
    """Rolling standard deviation."""
    s = pd.Series(data)
    return s.rolling(period).std().fillna(0.0).values


def compute_zscore(data, lookback: int):
    """Z-score of price."""
    s = pd.Series(data)
    mean = s.rolling(lookback).mean()
    std = s.rolling(lookback).std() + 1e-10
    zscore = (s - mean) / std
    return zscore.fillna(0.0).values


def compute_pct_change(data, periods: int = 1):
    """Percentage change."""
    s = pd.Series(data)
    return s.pct_change(periods).fillna(0.0).values


# =============================================================================
# BASE PHYSICS STRATEGY
# =============================================================================

class BasePhysicsStrategy(Strategy):
    """
    Base class for all physics-based trading strategies.

    Provides common physics calculations and risk management.
    """
    # Parameters (can be optimized)
    lookback = 20
    mass = 1.0
    energy_threshold_pct = 75  # Percentile for energy threshold
    damping_low_pct = 25      # Percentile for low damping
    damping_high_pct = 75     # Percentile for high damping
    risk_per_trade = 0.02     # 2% risk per trade

    def init(self):
        """Initialize indicators."""
        # Core physics metrics
        self.energy = self.I(compute_kinetic_energy, self.data.Close, self.mass)
        self.damping = self.I(compute_damping, self.data.Close, self.lookback)
        self.entropy = self.I(compute_entropy, self.data.Close, self.lookback)

        # Kinematics
        self.velocity = self.I(compute_velocity, self.data.Close)
        self.acceleration = self.I(compute_acceleration, self.data.Close)
        self.momentum = self.I(compute_momentum, self.data.Close, self.lookback)

        # Volatility for position sizing
        if hasattr(self.data, 'High') and hasattr(self.data, 'Low'):
            self.atr = self.I(compute_atr, self.data.High, self.data.Low, self.data.Close, 14)
        else:
            self.atr = self.I(lambda c: c.rolling(14).std(), self.data.Close)

    def get_regime(self) -> str:
        """
        Classify current market regime using dynamic percentiles.
        NO FIXED THRESHOLDS - adapts to market conditions.
        """
        if len(self.energy) < self.lookback:
            return "critical"

        # Calculate dynamic thresholds
        energy_history = self.energy[-100:] if len(self.energy) > 100 else self.energy
        damping_history = self.damping[-100:] if len(self.damping) > 100 else self.damping

        energy_75 = np.percentile(energy_history, self.energy_threshold_pct)
        damping_25 = np.percentile(damping_history, self.damping_low_pct)
        damping_75 = np.percentile(damping_history, self.damping_high_pct)

        current_energy = self.energy[-1]
        current_damping = self.damping[-1]

        if current_energy > energy_75 and current_damping < damping_25:
            return "underdamped"  # High energy, low friction -> trending
        elif damping_25 <= current_damping <= damping_75:
            return "critical"     # Balanced -> transitional
        else:
            return "overdamped"   # High friction -> ranging

    def calculate_position_size(self) -> float:
        """
        Calculate position size based on ATR and risk parameters.
        Uses physics-aware position sizing.
        """
        if self.atr[-1] <= 0 or np.isnan(self.atr[-1]):
            return 0.1  # Minimum position

        # Risk-adjusted position size
        equity = self.equity
        risk_amount = equity * self.risk_per_trade

        # Position size = risk / ATR (normalized)
        size = risk_amount / (self.atr[-1] * 100)

        # Clamp to reasonable bounds
        return np.clip(size, 0.01, 1.0)


# =============================================================================
# ENERGY-BASED MOMENTUM STRATEGY
# =============================================================================

class EnergyMomentumStrategy(BasePhysicsStrategy):
    """
    Kinetic Energy Momentum Strategy.

    Physics Principle: E = 0.5 * m * v²

    Trading Logic:
    - High kinetic energy + positive velocity -> BUY (trending up)
    - High kinetic energy + negative velocity -> SELL (trending down)
    - Low energy -> CLOSE (no momentum)

    Empirical Hypothesis: Markets with high kinetic energy tend to continue
    in their direction (momentum persistence).
    """
    energy_multiplier = 1.5  # Energy threshold multiplier

    def init(self):
        super().init()
        # Smoothed energy for signal generation
        self.smooth_energy = self.I(
            compute_ewm, self.energy, max(2, self.lookback // 2)
        )

    def next(self):
        if len(self.energy) < self.lookback:
            return

        # Dynamic energy threshold
        energy_threshold = np.percentile(
            self.energy[-100:] if len(self.energy) > 100 else self.energy,
            self.energy_threshold_pct
        )

        current_energy = self.smooth_energy[-1]
        velocity = self.velocity[-1]

        # High energy momentum trading
        if current_energy > energy_threshold * self.energy_multiplier:
            if velocity > 0:
                if not self.position.is_long:
                    self.position.close()
                    self.buy(size=self.calculate_position_size())
            elif velocity < 0:
                if not self.position.is_short:
                    self.position.close()
                    self.sell(size=self.calculate_position_size())
        elif current_energy < energy_threshold * 0.5:
            # Low energy - close position
            self.position.close()


# =============================================================================
# DAMPING-BASED MEAN REVERSION STRATEGY
# =============================================================================

class DampingReversionStrategy(BasePhysicsStrategy):
    """
    Damping Coefficient Mean Reversion Strategy.

    Physics Principle: ζ = friction / (2 * √(k * m))

    Trading Logic:
    - High damping (overdamped) -> expect mean reversion
    - Buy when price drops in high damping regime
    - Sell when price rises in high damping regime

    Empirical Hypothesis: Overdamped markets resist large moves
    and tend to revert to equilibrium.
    """
    reversion_threshold = 2.0  # StdDev threshold for reversion signal

    def init(self):
        super().init()
        # Z-score of price for mean reversion signals
        self.price_zscore = self.I(
            compute_zscore, self.data.Close, self.lookback
        )

    def next(self):
        if len(self.damping) < self.lookback:
            return

        regime = self.get_regime()
        zscore = self.price_zscore[-1]

        # Only trade mean reversion in overdamped (high friction) regimes
        if regime == "overdamped":
            if zscore < -self.reversion_threshold:
                # Oversold in high-friction market -> BUY
                if not self.position.is_long:
                    self.position.close()
                    self.buy(size=self.calculate_position_size())
            elif zscore > self.reversion_threshold:
                # Overbought in high-friction market -> SELL
                if not self.position.is_short:
                    self.position.close()
                    self.sell(size=self.calculate_position_size())
            elif abs(zscore) < 0.5:
                # Price near mean, exit
                self.position.close()
        else:
            # Don't trade mean reversion in trending markets
            if regime == "underdamped":
                self.position.close()


# =============================================================================
# ENTROPY-BASED VOLATILITY STRATEGY
# =============================================================================

class EntropyVolatilityStrategy(BasePhysicsStrategy):
    """
    Shannon Entropy Volatility Strategy.

    Physics Principle: H = -Σ p_i * log(p_i)

    Trading Logic:
    - Low entropy -> ordered market, trend-follow
    - High entropy -> disordered market, reduce exposure
    - Entropy spike -> volatility explosion imminent, prepare for breakout

    Empirical Hypothesis: Entropy is a leading indicator of volatility.
    Low entropy periods precede large moves.
    """
    entropy_spike_threshold = 1.5  # Multiplier for entropy spike detection

    def init(self):
        super().init()
        # Entropy rate of change
        self.entropy_roc = self.I(
            compute_pct_change, self.entropy, 5
        )
        # Smoothed entropy
        self.smooth_entropy = self.I(
            compute_sma, self.entropy, 5
        )

    def next(self):
        if len(self.entropy) < self.lookback:
            return

        # Dynamic entropy thresholds
        entropy_history = self.entropy[-100:] if len(self.entropy) > 100 else self.entropy
        entropy_mean = np.mean(entropy_history)
        entropy_std = np.std(entropy_history)

        current_entropy = self.smooth_entropy[-1]
        entropy_roc = self.entropy_roc[-1] if not np.isnan(self.entropy_roc[-1]) else 0

        # Low entropy = ordered market, trend-follow
        if current_entropy < entropy_mean - entropy_std:
            velocity = self.velocity[-1]
            if velocity > 0:
                if not self.position.is_long:
                    self.position.close()
                    self.buy(size=self.calculate_position_size())
            elif velocity < 0:
                if not self.position.is_short:
                    self.position.close()
                    self.sell(size=self.calculate_position_size())

        # High entropy = disordered, reduce exposure
        elif current_entropy > entropy_mean + entropy_std * self.entropy_spike_threshold:
            self.position.close()

        # Entropy spike (rapid increase) = prepare for breakout
        elif entropy_roc > 0.2:  # 20% increase in entropy
            # Reduce position size, wait for direction
            if self.position:
                self.position.close()


# =============================================================================
# ACCELERATION-BASED TREND CHANGE STRATEGY
# =============================================================================

class AccelerationTrendStrategy(BasePhysicsStrategy):
    """
    Kinematics-Based Acceleration Strategy.

    Physics Principle: a = dv/dt = d²x/dt²

    Trading Logic:
    - Positive acceleration + positive velocity -> momentum building, BUY
    - Negative acceleration + positive velocity -> momentum fading, prepare to SELL
    - Acceleration sign change -> trend reversal signal

    Empirical Hypothesis: Acceleration predicts trend changes before velocity does.
    """
    acceleration_smoothing = 3

    def init(self):
        super().init()
        # Smoothed acceleration
        self.smooth_accel = self.I(
            compute_sma, self.acceleration, self.acceleration_smoothing
        )
        # Acceleration sign - computed from smoothed
        self.accel_sign = self.I(
            lambda a: np.sign(compute_sma(a, self.acceleration_smoothing)),
            self.acceleration
        )

    def next(self):
        if len(self.acceleration) < self.lookback:
            return

        velocity = self.velocity[-1]
        accel = self.smooth_accel[-1]

        if np.isnan(accel):
            return

        # Momentum building: velocity and acceleration same direction
        if velocity > 0 and accel > 0:
            if not self.position.is_long:
                self.position.close()
                self.buy(size=self.calculate_position_size())
        elif velocity < 0 and accel < 0:
            if not self.position.is_short:
                self.position.close()
                self.sell(size=self.calculate_position_size())

        # Momentum fading: velocity positive but decelerating
        elif velocity > 0 and accel < 0:
            if self.position.is_long:
                self.position.close()
        elif velocity < 0 and accel > 0:
            if self.position.is_short:
                self.position.close()


# =============================================================================
# MULTI-PHYSICS REGIME ADAPTIVE STRATEGY
# =============================================================================

class MultiPhysicsStrategy(BasePhysicsStrategy):
    """
    Combined Multi-Physics Regime-Adaptive Strategy.

    Integrates all physics principles:
    - Energy for momentum detection
    - Damping for mean reversion signals
    - Entropy for volatility prediction
    - Acceleration for trend change detection

    Uses regime classification to select appropriate sub-strategy.
    """
    # Regime-specific parameters
    underdamped_energy_mult = 1.2
    overdamped_reversion_thresh = 1.5

    def init(self):
        super().init()
        # Z-score for mean reversion
        self.price_zscore = self.I(
            compute_zscore, self.data.Close, self.lookback
        )
        # Smoothed metrics
        self.smooth_energy = self.I(
            compute_ewm, self.energy, max(2, self.lookback // 2)
        )
        self.smooth_accel = self.I(
            compute_sma, self.acceleration, 3
        )

    def next(self):
        if len(self.energy) < self.lookback:
            return

        regime = self.get_regime()

        if regime == "underdamped":
            self._trade_momentum()
        elif regime == "overdamped":
            self._trade_reversion()
        else:  # critical
            self._trade_breakout()

    def _trade_momentum(self):
        """Trade momentum in underdamped (trending) regime."""
        energy_threshold = np.percentile(
            self.energy[-100:] if len(self.energy) > 100 else self.energy,
            self.energy_threshold_pct
        )

        if self.smooth_energy[-1] > energy_threshold * self.underdamped_energy_mult:
            velocity = self.velocity[-1]
            accel = self.smooth_accel[-1] if not np.isnan(self.smooth_accel[-1]) else 0

            # Momentum with acceleration confirmation
            if velocity > 0 and accel >= 0:
                if not self.position.is_long:
                    self.position.close()
                    self.buy(size=self.calculate_position_size())
            elif velocity < 0 and accel <= 0:
                if not self.position.is_short:
                    self.position.close()
                    self.sell(size=self.calculate_position_size())

    def _trade_reversion(self):
        """Trade mean reversion in overdamped (ranging) regime."""
        zscore = self.price_zscore[-1]

        if zscore < -self.overdamped_reversion_thresh:
            if not self.position.is_long:
                self.position.close()
                self.buy(size=self.calculate_position_size())
        elif zscore > self.overdamped_reversion_thresh:
            if not self.position.is_short:
                self.position.close()
                self.sell(size=self.calculate_position_size())
        elif abs(zscore) < 0.3:
            self.position.close()

    def _trade_breakout(self):
        """Trade breakouts in critical (transitional) regime."""
        # Wait for strong directional signal
        energy_threshold = np.percentile(
            self.energy[-100:] if len(self.energy) > 100 else self.energy,
            90  # Higher threshold for breakout
        )

        if self.smooth_energy[-1] > energy_threshold:
            velocity = self.velocity[-1]
            if velocity > 0:
                if not self.position.is_long:
                    self.position.close()
                    self.buy(size=self.calculate_position_size() * 0.5)  # Smaller size
            elif velocity < 0:
                if not self.position.is_short:
                    self.position.close()
                    self.sell(size=self.calculate_position_size() * 0.5)
        else:
            # Low energy in critical regime - stay flat
            self.position.close()


# =============================================================================
# THERMODYNAMIC EQUILIBRIUM STRATEGY
# =============================================================================

class ThermodynamicEquilibriumStrategy(BasePhysicsStrategy):
    """
    Thermodynamic Equilibrium Strategy.

    Physics Principle: Systems tend toward equilibrium (maximum entropy).

    Trading Logic:
    - Calculate "temperature" = energy / entropy
    - High temperature = energy building faster than disorder -> trend
    - Low temperature = disorder dominates -> consolidation
    - Temperature changes predict regime shifts

    Empirical Hypothesis: Market "temperature" is a leading indicator
    of volatility expansion/contraction.
    """
    temp_lookback = 10

    def init(self):
        super().init()
        # Market temperature: ratio of energy to entropy
        self.temperature = self.I(
            lambda e, h: e / (h + 1e-6),
            self.energy, self.entropy
        )
        # Temperature rate of change
        self.temp_roc = self.I(
            compute_pct_change, self.temperature, self.temp_lookback
        )

    def next(self):
        if len(self.temperature) < self.lookback:
            return

        temp = self.temperature[-1]
        temp_roc = self.temp_roc[-1] if not np.isnan(self.temp_roc[-1]) else 0
        velocity = self.velocity[-1]

        # Dynamic temperature thresholds
        temp_history = self.temperature[-100:] if len(self.temperature) > 100 else self.temperature
        temp_history = [t for t in temp_history if not np.isnan(t) and not np.isinf(t)]
        if len(temp_history) < 10:
            return

        temp_75 = np.percentile(temp_history, 75)
        temp_25 = np.percentile(temp_history, 25)

        # High and rising temperature - energy building
        if temp > temp_75 and temp_roc > 0:
            if velocity > 0:
                if not self.position.is_long:
                    self.position.close()
                    self.buy(size=self.calculate_position_size())
            elif velocity < 0:
                if not self.position.is_short:
                    self.position.close()
                    self.sell(size=self.calculate_position_size())

        # Temperature dropping - energy dissipating
        elif temp_roc < -0.1:  # 10% temperature drop
            self.position.close()

        # Low temperature - consolidation, stay flat
        elif temp < temp_25:
            self.position.close()


# =============================================================================
# STRATEGY REGISTRY
# =============================================================================

STRATEGY_REGISTRY = {
    "energy_momentum": EnergyMomentumStrategy,
    "damping_reversion": DampingReversionStrategy,
    "entropy_volatility": EntropyVolatilityStrategy,
    "acceleration_trend": AccelerationTrendStrategy,
    "multi_physics": MultiPhysicsStrategy,
    "thermodynamic": ThermodynamicEquilibriumStrategy,
}


def get_strategy(name: str) -> type:
    """Get strategy class by name."""
    if name not in STRATEGY_REGISTRY:
        raise ValueError(f"Unknown strategy: {name}. Available: {list(STRATEGY_REGISTRY.keys())}")
    return STRATEGY_REGISTRY[name]


def list_strategies() -> List[str]:
    """List all available strategies."""
    return list(STRATEGY_REGISTRY.keys())


# =============================================================================
# BACKTEST RUNNER
# =============================================================================

class PhysicsBacktestRunner:
    """
    Runner for physics-based backtests.

    Provides:
    - Single strategy backtesting
    - Strategy comparison
    - Parameter optimization
    - Monte Carlo validation
    """

    def __init__(
        self,
        cash: float = 100000.0,
        commission: float = 0.001,
        margin: float = 1.0,
        trade_on_close: bool = True,
        exclusive_orders: bool = True
    ):
        """
        Initialize backtest runner.

        Args:
            cash: Starting capital
            commission: Commission per trade (0.1% default)
            margin: Margin requirement (1.0 = no leverage)
            trade_on_close: Execute trades at close price
            exclusive_orders: One order at a time
        """
        self.cash = cash
        self.commission = commission
        self.margin = margin
        self.trade_on_close = trade_on_close
        self.exclusive_orders = exclusive_orders

    def run(
        self,
        data: pd.DataFrame,
        strategy: str = "multi_physics",
        **strategy_params
    ) -> Dict[str, Any]:
        """
        Run a single backtest.

        Args:
            data: OHLCV DataFrame with columns: Open, High, Low, Close, Volume
            strategy: Strategy name from registry
            **strategy_params: Parameters to pass to strategy

        Returns:
            Dict with backtest results and statistics
        """
        strategy_class = get_strategy(strategy)

        # Apply custom parameters
        for key, value in strategy_params.items():
            setattr(strategy_class, key, value)

        bt = Backtest(
            data,
            strategy_class,
            cash=self.cash,
            commission=self.commission,
            margin=self.margin,
            trade_on_close=self.trade_on_close,
            exclusive_orders=self.exclusive_orders
        )

        stats = bt.run()

        return self._process_results(stats, strategy)

    def compare_strategies(
        self,
        data: pd.DataFrame,
        strategies: Optional[List[str]] = None,
        parallel: bool = False
    ) -> pd.DataFrame:
        """
        Compare multiple strategies on the same data.

        Args:
            data: OHLCV DataFrame
            strategies: List of strategy names (all if None)
            parallel: If True, use parallel execution (may have pickling issues in some environments)

        Returns:
            DataFrame comparing strategy performance
        """
        if strategies is None:
            strategies = list_strategies()

        def run_strategy(strategy_name: str) -> Optional[Dict]:
            """Run single strategy for parallel execution."""
            try:
                result = self.run(data, strategy=strategy_name)
                result["strategy"] = strategy_name
                return result
            except Exception as e:
                import traceback
                error_msg = f"Error running {strategy_name}: {e}\n{traceback.format_exc()}"
                print(error_msg)
                # Return a dict with error info instead of None to help debugging
                return {
                    "strategy": strategy_name,
                    "error": str(e),
                    "Return [%]": 0.0,
                    "Max. Drawdown [%]": 0.0,
                    "# Trades": 0,
                }

        results = []
        n_workers = min(mp.cpu_count(), len(strategies), MAX_WORKERS)

        # Use parallel execution only if explicitly enabled and there are enough strategies
        if parallel and n_workers > 1 and len(strategies) >= 3:
            # Parallel strategy comparison
            with ProcessPoolExecutor(max_workers=n_workers) as executor:
                futures = {executor.submit(run_strategy, name): name for name in strategies}
                for future in as_completed(futures):
                    result = future.result()
                    if result is not None:
                        results.append(result)
        else:
            # Sequential execution (safer, especially for tests)
            for strategy_name in strategies:
                result = run_strategy(strategy_name)
                if result is not None:
                    results.append(result)

        return pd.DataFrame(results)

    def optimize(
        self,
        data: pd.DataFrame,
        strategy: str,
        param_ranges: Dict[str, range],
        maximize: str = "Return [%]"
    ) -> Tuple[Dict, Any]:
        """
        Optimize strategy parameters.

        Args:
            data: OHLCV DataFrame
            strategy: Strategy name
            param_ranges: Dict of parameter names to ranges
            maximize: Metric to maximize

        Returns:
            Tuple of (best_params, stats)
        """
        strategy_class = get_strategy(strategy)

        bt = Backtest(
            data,
            strategy_class,
            cash=self.cash,
            commission=self.commission,
            margin=self.margin,
            trade_on_close=self.trade_on_close,
            exclusive_orders=self.exclusive_orders
        )

        stats = bt.optimize(
            **param_ranges,
            maximize=maximize,
            return_heatmap=False
        )

        # Extract optimal parameters
        best_params = {}
        for param in param_ranges.keys():
            if hasattr(stats, '_strategy') and hasattr(stats._strategy, param):
                best_params[param] = getattr(stats._strategy, param)

        return best_params, stats

    def monte_carlo(
        self,
        data: pd.DataFrame,
        strategy: str,
        n_runs: int = 100,
        sample_pct: float = 0.8,
        **strategy_params
    ) -> pd.DataFrame:
        """
        Run Monte Carlo validation with random subsampling.

        Args:
            data: OHLCV DataFrame
            strategy: Strategy name
            n_runs: Number of Monte Carlo runs
            sample_pct: Percentage of data to use per run
            **strategy_params: Strategy parameters

        Returns:
            DataFrame with results from all runs
        """
        results = []
        n_samples = int(len(data) * sample_pct)

        for i in range(n_runs):
            # Random start point
            max_start = len(data) - n_samples
            if max_start <= 0:
                start = 0
            else:
                start = np.random.randint(0, max_start)

            sample_data = data.iloc[start:start + n_samples].copy()

            if len(sample_data) < 50:  # Minimum data requirement
                continue

            try:
                result = self.run(sample_data, strategy=strategy, **strategy_params)
                result["run"] = i
                result["start_idx"] = start
                results.append(result)
            except Exception as e:
                print(f"Monte Carlo run {i} failed: {e}")

        return pd.DataFrame(results)

    def _process_results(self, stats, strategy_name: str) -> Dict[str, Any]:
        """Process backtesting.py stats into our format."""
        return {
            "strategy": strategy_name,
            "return_pct": float(stats["Return [%]"]),
            "buy_hold_return_pct": float(stats["Buy & Hold Return [%]"]),
            "sharpe_ratio": float(stats["Sharpe Ratio"]) if not np.isnan(stats["Sharpe Ratio"]) else 0.0,
            "sortino_ratio": float(stats["Sortino Ratio"]) if not np.isnan(stats["Sortino Ratio"]) else 0.0,
            "max_drawdown_pct": float(stats["Max. Drawdown [%]"]),
            "win_rate": float(stats["Win Rate [%]"]) if not np.isnan(stats["Win Rate [%]"]) else 0.0,
            "profit_factor": float(stats["Profit Factor"]) if not np.isnan(stats["Profit Factor"]) else 0.0,
            "total_trades": int(stats["# Trades"]),
            "avg_trade_pct": float(stats["Avg. Trade [%]"]) if not np.isnan(stats["Avg. Trade [%]"]) else 0.0,
            "exposure_pct": float(stats["Exposure Time [%]"]),
            "final_equity": float(stats["Equity Final [$]"]),
            "sqn": float(stats["SQN"]) if not np.isnan(stats["SQN"]) else 0.0,
        }


# =============================================================================
# PHYSICS METRICS CALCULATOR
# =============================================================================

def calculate_physics_metrics(data: pd.DataFrame, results: Dict) -> Dict[str, float]:
    """
    Calculate physics-specific performance metrics.

    Args:
        data: OHLCV DataFrame
        results: Backtest results dict

    Returns:
        Dict with physics metrics
    """
    engine = PhysicsEngine()
    physics_state = engine.compute_physics_state(data['Close'])

    # Total available energy in the data
    total_energy = physics_state['energy'].sum()

    # Energy captured (simplified - assume returns are proportional to energy)
    if results['return_pct'] > 0:
        energy_captured_pct = min(100.0, (results['return_pct'] / 100) *
                                   (total_energy / (total_energy / len(data) * 100)))
    else:
        energy_captured_pct = 0.0

    # Regime distribution
    regime_counts = physics_state['regime'].value_counts()
    regime_dist = regime_counts / len(physics_state) * 100

    # Average entropy during trades (simplified)
    avg_entropy = physics_state['entropy'].mean()

    # Average damping
    avg_damping = physics_state['damping'].mean()

    return {
        "total_energy": float(total_energy),
        "energy_captured_pct": float(energy_captured_pct),
        "avg_entropy": float(avg_entropy),
        "avg_damping": float(avg_damping),
        "regime_underdamped_pct": float(regime_dist.get('underdamped', 0)),
        "regime_critical_pct": float(regime_dist.get('critical', 0)),
        "regime_overdamped_pct": float(regime_dist.get('overdamped', 0)),
        "regime_laminar_pct": float(regime_dist.get('laminar', 0)),
        "regime_breakout_pct": float(regime_dist.get('breakout', 0)),
    }
