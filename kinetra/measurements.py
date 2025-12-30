"""
Comprehensive Measurement Framework for Multi-Asset Exploration
================================================================

PHILOSOPHY: We don't know what we don't know.

Don't assume:
- "Trending" means the same thing for crypto vs metals vs indices
- RSI/MACD are the right momentum measures for all classes
- Relationships are stable - they may INVERT during volatility

APPROACH:
1. Measure EVERYTHING we can compute
2. Track correlations between ALL measurements
3. Let the RL agent discover what matters per class
4. Explore inverse relationships during high-energy regimes

PHYSICS INSIGHT:
During turbulent flow (high Reynolds), relationships flip.
High volatility = massive energy release potential.
What works in laminar flow fails in turbulent.
Reynolds should be INVERSE to ROC during instability.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
import numpy as np
import pandas as pd
from collections import defaultdict


# =============================================================================
# MEASUREMENT CATEGORIES
# =============================================================================

class MeasurementCategory(Enum):
    """Categories of measurements - for organization, not assumption."""
    VOLATILITY = "volatility"
    MOMENTUM = "momentum"
    MEAN_REVERSION = "mean_reversion"
    MICROSTRUCTURE = "microstructure"
    PHYSICS = "physics"
    ENERGY = "energy"  # Potential energy, release indicators
    FLOW = "flow"  # Reynolds-like flow dynamics
    CROSS_ASSET = "cross_asset"
    TIME = "time"  # Session, day-of-week, etc.


@dataclass
class Measurement:
    """A single measurement with metadata."""
    name: str
    category: MeasurementCategory
    value: float
    z_score: float = 0.0  # Normalized value
    percentile: float = 0.5  # Where in distribution
    is_extreme: bool = False  # > 2 std or < -2 std


@dataclass
class MeasurementSet:
    """Complete set of measurements for one bar."""
    timestamp: pd.Timestamp
    measurements: Dict[str, Measurement] = field(default_factory=dict)

    def to_array(self) -> np.ndarray:
        """Convert to feature array for RL."""
        return np.array([m.z_score for m in self.measurements.values()])

    def get_names(self) -> List[str]:
        """Get measurement names in order."""
        return list(self.measurements.keys())


# =============================================================================
# VOLATILITY MEASUREMENTS (Multiple Estimators)
# =============================================================================

class VolatilityMeasures:
    """
    Multiple volatility estimators - each captures different dynamics.

    Don't assume ATR is best. Yang-Zhang handles overnight gaps.
    Parkinson uses high-low range. Rogers-Satchell handles drift.
    """

    @staticmethod
    def atr(high: np.ndarray, low: np.ndarray, close: np.ndarray,
            period: int = 14) -> np.ndarray:
        """Average True Range - classic but ignores overnight."""
        tr = np.maximum(
            high - low,
            np.maximum(
                np.abs(high - np.roll(close, 1)),
                np.abs(low - np.roll(close, 1))
            )
        )
        tr[0] = high[0] - low[0]

        atr = np.zeros_like(tr)
        atr[period-1] = np.mean(tr[:period])
        for i in range(period, len(tr)):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
        return atr

    @staticmethod
    def yang_zhang(open_: np.ndarray, high: np.ndarray, low: np.ndarray,
                   close: np.ndarray, period: int = 20) -> np.ndarray:
        """
        Yang-Zhang volatility - handles overnight gaps + intraday.
        Better for assets with gaps (indices, some forex).
        """
        n = len(close)
        vol = np.zeros(n)

        for i in range(period, n):
            h = high[i-period:i]
            l = low[i-period:i]
            o = open_[i-period:i]
            c = close[i-period:i]
            c_prev = close[i-period-1:i-1] if i > period else c

            # Overnight variance
            log_oc = np.log(o / (c_prev + 1e-10) + 1e-10)
            overnight_var = np.var(log_oc)

            # Open-to-close variance
            log_co = np.log(c / (o + 1e-10) + 1e-10)
            open_close_var = np.var(log_co)

            # Rogers-Satchell component
            log_hc = np.log(h / (c + 1e-10) + 1e-10)
            log_lc = np.log(l / (c + 1e-10) + 1e-10)
            log_ho = np.log(h / (o + 1e-10) + 1e-10)
            log_lo = np.log(l / (o + 1e-10) + 1e-10)
            rs_var = np.mean(log_ho * log_hc + log_lo * log_lc)

            k = 0.34 / (1.34 + (period + 1) / (period - 1))
            vol[i] = np.sqrt(max(0, overnight_var + k * open_close_var + (1 - k) * rs_var))

        return vol

    @staticmethod
    def parkinson(high: np.ndarray, low: np.ndarray, period: int = 20) -> np.ndarray:
        """
        Parkinson volatility - uses high-low range.
        More efficient than close-to-close for continuous markets.
        """
        n = len(high)
        vol = np.zeros(n)

        log_hl = np.log(high / (low + 1e-10) + 1e-10)
        factor = 1.0 / (4.0 * np.log(2.0))

        for i in range(period, n):
            vol[i] = np.sqrt(factor * np.mean(log_hl[i-period:i] ** 2))

        return vol

    @staticmethod
    def rogers_satchell(open_: np.ndarray, high: np.ndarray, low: np.ndarray,
                        close: np.ndarray, period: int = 20) -> np.ndarray:
        """
        Rogers-Satchell - drift independent.
        Good for trending assets (metals, crypto) where there's persistent drift.
        """
        n = len(close)
        vol = np.zeros(n)

        for i in range(period, n):
            h = high[i-period:i]
            l = low[i-period:i]
            o = open_[i-period:i]
            c = close[i-period:i]

            log_ho = np.log(h / (o + 1e-10) + 1e-10)
            log_hc = np.log(h / (c + 1e-10) + 1e-10)
            log_lo = np.log(l / (o + 1e-10) + 1e-10)
            log_lc = np.log(l / (c + 1e-10) + 1e-10)

            rs = log_ho * log_hc + log_lo * log_lc
            vol[i] = np.sqrt(max(0, np.mean(rs)))

        return vol

    @staticmethod
    def garman_klass(open_: np.ndarray, high: np.ndarray, low: np.ndarray,
                     close: np.ndarray, period: int = 20) -> np.ndarray:
        """
        Garman-Klass volatility - efficient estimator.
        Combines open-close and high-low information.
        """
        n = len(close)
        vol = np.zeros(n)

        for i in range(period, n):
            h = high[i-period:i]
            l = low[i-period:i]
            o = open_[i-period:i]
            c = close[i-period:i]

            log_hl = np.log(h / (l + 1e-10) + 1e-10)
            log_co = np.log(c / (o + 1e-10) + 1e-10)

            gk = 0.5 * log_hl**2 - (2*np.log(2) - 1) * log_co**2
            vol[i] = np.sqrt(max(0, np.mean(gk)))

        return vol

    @staticmethod
    def volatility_of_volatility(vol_series: np.ndarray, period: int = 20) -> np.ndarray:
        """
        Vol-of-vol: How unstable is volatility itself?
        High vol-of-vol = regime uncertainty = potential inversions.
        """
        n = len(vol_series)
        vov = np.zeros(n)

        for i in range(period, n):
            window = vol_series[i-period:i]
            if np.mean(window) > 0:
                vov[i] = np.std(window) / np.mean(window)  # CV of volatility

        return vov


# =============================================================================
# MOMENTUM MEASUREMENTS (Multiple Approaches)
# =============================================================================

class MomentumMeasures:
    """
    Multiple momentum measures - "trending" means different things.

    - ROC: Pure price change
    - RSI: Bounded oscillator
    - MACD: Trend-following
    - ADX: Trend strength (not direction)
    - Aroon: Time since high/low
    - Momentum divergence: Price vs momentum disagreement
    """

    @staticmethod
    def roc(close: np.ndarray, periods: List[int] = [5, 10, 20, 50]) -> Dict[str, np.ndarray]:
        """Rate of Change at multiple timeframes."""
        result = {}
        for p in periods:
            roc = np.zeros_like(close)
            roc[p:] = (close[p:] - close[:-p]) / (close[:-p] + 1e-10) * 100
            result[f'roc_{p}'] = roc
        return result

    @staticmethod
    def rsi(close: np.ndarray, period: int = 14) -> np.ndarray:
        """Relative Strength Index."""
        delta = np.diff(close, prepend=close[0])
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)

        avg_gain = np.zeros_like(close)
        avg_loss = np.zeros_like(close)

        avg_gain[period] = np.mean(gain[1:period+1])
        avg_loss[period] = np.mean(loss[1:period+1])

        for i in range(period + 1, len(close)):
            avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i]) / period
            avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i]) / period

        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
        return rsi

    @staticmethod
    def macd(close: np.ndarray, fast: int = 12, slow: int = 26,
             signal: int = 9) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """MACD with histogram."""
        def ema(data, period):
            result = np.zeros_like(data)
            result[0] = data[0]
            alpha = 2 / (period + 1)
            for i in range(1, len(data)):
                result[i] = alpha * data[i] + (1 - alpha) * result[i-1]
            return result

        ema_fast = ema(close, fast)
        ema_slow = ema(close, slow)
        macd_line = ema_fast - ema_slow
        signal_line = ema(macd_line, signal)
        histogram = macd_line - signal_line

        return macd_line, signal_line, histogram

    @staticmethod
    def adx(high: np.ndarray, low: np.ndarray, close: np.ndarray,
            period: int = 14) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Average Directional Index - trend strength."""
        n = len(close)

        # True Range
        tr = np.maximum(
            high - low,
            np.maximum(
                np.abs(high - np.roll(close, 1)),
                np.abs(low - np.roll(close, 1))
            )
        )
        tr[0] = high[0] - low[0]

        # Directional Movement
        up_move = high - np.roll(high, 1)
        down_move = np.roll(low, 1) - low
        up_move[0] = 0
        down_move[0] = 0

        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)

        # Smoothed averages
        def smooth(data, period):
            result = np.zeros_like(data)
            result[period-1] = np.sum(data[:period])
            for i in range(period, len(data)):
                result[i] = result[i-1] - result[i-1]/period + data[i]
            return result

        atr = smooth(tr, period)
        plus_di = 100 * smooth(plus_dm, period) / (atr + 1e-10)
        minus_di = 100 * smooth(minus_dm, period) / (atr + 1e-10)

        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
        adx = smooth(dx, period) / period

        return adx, plus_di, minus_di

    @staticmethod
    def aroon(high: np.ndarray, low: np.ndarray, period: int = 25) -> Tuple[np.ndarray, np.ndarray]:
        """Aroon - time since high/low."""
        n = len(high)
        aroon_up = np.zeros(n)
        aroon_down = np.zeros(n)

        for i in range(period, n):
            high_idx = np.argmax(high[i-period:i+1])
            low_idx = np.argmin(low[i-period:i+1])
            aroon_up[i] = 100 * high_idx / period
            aroon_down[i] = 100 * low_idx / period

        return aroon_up, aroon_down

    @staticmethod
    def momentum_divergence(close: np.ndarray, rsi: np.ndarray,
                            period: int = 20) -> np.ndarray:
        """
        Divergence between price and momentum.
        Price making new highs but RSI not = bearish divergence.
        """
        n = len(close)
        divergence = np.zeros(n)

        for i in range(period, n):
            price_change = (close[i] - close[i-period]) / (close[i-period] + 1e-10)
            rsi_change = rsi[i] - rsi[i-period]

            # Normalize both to [-1, 1]
            price_norm = np.tanh(price_change * 10)
            rsi_norm = rsi_change / 50  # RSI is 0-100, so change is -100 to +100

            # Divergence = disagreement
            divergence[i] = price_norm - rsi_norm

        return divergence


# =============================================================================
# MEAN REVERSION MEASUREMENTS
# =============================================================================

class MeanReversionMeasures:
    """
    Mean reversion indicators.

    CRITICAL: What "mean reversion" looks like differs by asset class.
    Forex may mean-revert on H1, but crypto may not.
    """

    @staticmethod
    def bollinger_percent_b(close: np.ndarray, period: int = 20,
                            std_mult: float = 2.0) -> np.ndarray:
        """Bollinger %B - where price is within bands."""
        n = len(close)
        percent_b = np.zeros(n)

        for i in range(period, n):
            window = close[i-period:i+1]
            ma = np.mean(window)
            std = np.std(window)

            upper = ma + std_mult * std
            lower = ma - std_mult * std

            if upper != lower:
                percent_b[i] = (close[i] - lower) / (upper - lower)

        return percent_b

    @staticmethod
    def z_score(close: np.ndarray, period: int = 20) -> np.ndarray:
        """Z-score from rolling mean."""
        n = len(close)
        z = np.zeros(n)

        for i in range(period, n):
            window = close[i-period:i]
            mean = np.mean(window)
            std = np.std(window)
            if std > 0:
                z[i] = (close[i] - mean) / std

        return z

    @staticmethod
    def hurst_exponent(close: np.ndarray, max_lag: int = 20) -> np.ndarray:
        """
        Hurst exponent - measure of persistence/anti-persistence.
        H < 0.5: Mean-reverting
        H = 0.5: Random walk
        H > 0.5: Trending

        This is THE key measurement for MR vs trending classification.
        """
        n = len(close)
        hurst = np.zeros(n)

        for i in range(max_lag * 2, n):
            window = close[i-max_lag*2:i]

            lags = range(2, max_lag)
            rs_values = []

            for lag in lags:
                # Calculate R/S statistic
                diffs = np.diff(window)
                mean_diff = np.mean(diffs)
                centered = diffs - mean_diff

                cumsum = np.cumsum(centered)
                r = np.max(cumsum) - np.min(cumsum)
                s = np.std(diffs)

                if s > 0:
                    rs_values.append((lag, r / s))

            if len(rs_values) > 2:
                lags_log = np.log([x[0] for x in rs_values])
                rs_log = np.log([x[1] for x in rs_values])

                # Linear regression to get Hurst exponent
                slope, _ = np.polyfit(lags_log, rs_log, 1)
                hurst[i] = slope

        return hurst

    @staticmethod
    def distance_from_vwap(close: np.ndarray, volume: np.ndarray,
                           period: int = 20) -> np.ndarray:
        """Distance from VWAP - institutional reference point."""
        n = len(close)
        dist = np.zeros(n)

        for i in range(period, n):
            c = close[i-period:i+1]
            v = volume[i-period:i+1]

            vwap = np.sum(c * v) / (np.sum(v) + 1e-10)
            dist[i] = (close[i] - vwap) / (vwap + 1e-10) * 100

        return dist


# =============================================================================
# ENERGY & FLOW MEASUREMENTS (Physics-Inspired)
# =============================================================================

class EnergyFlowMeasures:
    """
    Physics-inspired measurements for energy and flow dynamics.

    KEY INSIGHT: During high volatility (turbulent flow), relationships INVERT.
    Reynolds number should be INVERSE to ROC during instability.
    """

    @staticmethod
    def kinetic_energy(close: np.ndarray, period: int = 20) -> np.ndarray:
        """
        Kinetic energy proxy: (velocity)^2
        Velocity = rate of change
        """
        velocity = np.diff(close, prepend=close[0]) / (close + 1e-10)

        n = len(close)
        ke = np.zeros(n)

        for i in range(period, n):
            ke[i] = np.mean(velocity[i-period:i] ** 2)

        return ke

    @staticmethod
    def potential_energy(close: np.ndarray, period: int = 50) -> np.ndarray:
        """
        Potential energy proxy: Distance from equilibrium (long-term mean).
        Further from mean = more potential energy = reversion potential.
        """
        n = len(close)
        pe = np.zeros(n)

        for i in range(period, n):
            mean = np.mean(close[i-period:i])
            pe[i] = ((close[i] - mean) / (mean + 1e-10)) ** 2

        return pe

    @staticmethod
    def energy_release_rate(close: np.ndarray, vol: np.ndarray,
                            period: int = 10) -> np.ndarray:
        """
        Energy release rate: How fast is stored energy being released?
        High value = explosive move potential exhausting.
        """
        n = len(close)
        err = np.zeros(n)

        for i in range(period, n):
            # Energy = vol * price_change
            energy = vol[i-period:i] * np.abs(np.diff(close[i-period:i+1]))
            err[i] = np.mean(np.diff(energy))

        return err

    @staticmethod
    def reynolds_number_proxy(close: np.ndarray, vol: np.ndarray,
                              period: int = 20) -> np.ndarray:
        """
        Reynolds number proxy: Inertia / Viscosity.

        High Reynolds = turbulent flow = unpredictable.
        Low Reynolds = laminar flow = predictable.

        Inertia ~ momentum (price change * "mass")
        Viscosity ~ resistance to change (inverse volatility)
        """
        n = len(close)
        reynolds = np.zeros(n)

        for i in range(period, n):
            # Inertia: magnitude of price movement
            price_change = np.abs(close[i] - close[i-period]) / (close[i-period] + 1e-10)

            # Viscosity: inverse of volatility (low vol = high viscosity)
            avg_vol = np.mean(vol[i-period:i])
            viscosity = 1.0 / (avg_vol + 1e-10)

            reynolds[i] = price_change / (viscosity + 1e-10)

        return reynolds

    @staticmethod
    def reynolds_roc_inverse(reynolds: np.ndarray, roc: np.ndarray) -> np.ndarray:
        """
        KEY INSIGHT: Reynolds should be INVERSE to ROC during instability.

        When this relationship breaks (both high or both low together),
        it signals regime transition or opportunity.

        Returns: Correlation between Reynolds and ROC (should be negative in stable regimes)
        """
        n = len(reynolds)
        inverse_measure = np.zeros(n)
        period = 20

        for i in range(period, n):
            re = reynolds[i-period:i]
            ro = roc[i-period:i]

            # Normalize both
            re_norm = (re - np.mean(re)) / (np.std(re) + 1e-10)
            ro_norm = (ro - np.mean(ro)) / (np.std(ro) + 1e-10)

            # Correlation (should be negative in stable flow)
            corr = np.corrcoef(re_norm, ro_norm)[0, 1] if len(re) > 2 else 0
            inverse_measure[i] = corr

        return inverse_measure

    @staticmethod
    def flow_regime(reynolds: np.ndarray) -> np.ndarray:
        """
        Classify flow regime from Reynolds number.

        Returns:
        0 = Laminar (predictable)
        1 = Transitional (unstable)
        2 = Turbulent (chaotic)
        """
        # Percentile-based thresholds (learned from data)
        n = len(reynolds)
        regime = np.zeros(n)
        period = 100

        for i in range(period, n):
            window = reynolds[max(0, i-period):i]
            p33 = np.percentile(window, 33)
            p66 = np.percentile(window, 66)

            if reynolds[i] < p33:
                regime[i] = 0  # Laminar
            elif reynolds[i] < p66:
                regime[i] = 1  # Transitional
            else:
                regime[i] = 2  # Turbulent

        return regime

    @staticmethod
    def entropy_rate(close: np.ndarray, bins: int = 10, period: int = 50) -> np.ndarray:
        """
        Entropy rate: How unpredictable is the next move?
        High entropy = high uncertainty = potential for surprise.
        """
        returns = np.diff(close, prepend=close[0]) / (close + 1e-10)

        n = len(close)
        entropy = np.zeros(n)

        for i in range(period, n):
            window = returns[i-period:i]

            # Discretize returns into bins
            hist, _ = np.histogram(window, bins=bins)
            probs = hist / (np.sum(hist) + 1e-10)

            # Shannon entropy
            probs = probs[probs > 0]
            entropy[i] = -np.sum(probs * np.log2(probs))

        return entropy


# =============================================================================
# MICROSTRUCTURE MEASUREMENTS
# =============================================================================

class MicrostructureMeasures:
    """
    Market microstructure measurements.

    Spread, volume, tick activity - execution quality indicators.
    """

    @staticmethod
    def spread_ratio(spread: np.ndarray, period: int = 100) -> np.ndarray:
        """Spread relative to rolling minimum (from SpreadGate)."""
        n = len(spread)
        ratio = np.ones(n)

        for i in range(period, n):
            rolling_min = np.min(spread[i-period:i])
            if rolling_min > 0:
                ratio[i] = spread[i] / rolling_min

        return ratio

    @staticmethod
    def volume_ratio(volume: np.ndarray, period: int = 50) -> np.ndarray:
        """Volume relative to rolling mean."""
        n = len(volume)
        ratio = np.ones(n)

        for i in range(period, n):
            rolling_mean = np.mean(volume[i-period:i])
            if rolling_mean > 0:
                ratio[i] = volume[i] / rolling_mean

        return ratio

    @staticmethod
    def volume_price_trend(close: np.ndarray, volume: np.ndarray) -> np.ndarray:
        """Volume-price trend: Cumulative volume-weighted price change."""
        returns = np.diff(close, prepend=close[0]) / (close + 1e-10)
        vpt = np.cumsum(returns * volume)
        return vpt

    @staticmethod
    def tick_intensity(volume: np.ndarray, period: int = 20) -> np.ndarray:
        """
        Tick intensity: Rate of change of volume.
        Sudden volume spike = information event.
        """
        n = len(volume)
        intensity = np.zeros(n)

        for i in range(period, n):
            if np.mean(volume[i-period:i-1]) > 0:
                intensity[i] = volume[i] / np.mean(volume[i-period:i-1])

        return intensity

    @staticmethod
    def liquidity_score(spread: np.ndarray, volume: np.ndarray,
                        period: int = 50) -> np.ndarray:
        """
        Composite liquidity score.
        High volume + low spread = good liquidity.
        """
        spread_ratio = MicrostructureMeasures.spread_ratio(spread, period)
        volume_ratio = MicrostructureMeasures.volume_ratio(volume, period)

        # High volume ratio good, high spread ratio bad
        liquidity = volume_ratio / (spread_ratio + 1e-10)
        return liquidity


# =============================================================================
# TIME-BASED MEASUREMENTS
# =============================================================================

class TimeMeasures:
    """
    Time-based features: Session, day-of-week, hour.

    Different asset classes have different time patterns.
    """

    @staticmethod
    def extract_time_features(timestamps: pd.DatetimeIndex) -> Dict[str, np.ndarray]:
        """Extract all time-based features."""
        return {
            'hour': np.array([t.hour for t in timestamps]),
            'day_of_week': np.array([t.dayofweek for t in timestamps]),
            'day_of_month': np.array([t.day for t in timestamps]),
            'week_of_year': np.array([t.isocalendar()[1] for t in timestamps]),
            'month': np.array([t.month for t in timestamps]),
            'is_month_end': np.array([t.is_month_end for t in timestamps]).astype(float),
            'is_quarter_end': np.array([t.is_quarter_end for t in timestamps]).astype(float),
        }

    @staticmethod
    def session_indicator(hour: np.ndarray) -> Dict[str, np.ndarray]:
        """Trading session indicators (UTC)."""
        return {
            'is_asian': ((hour >= 0) & (hour < 8)).astype(float),
            'is_london': ((hour >= 7) & (hour < 16)).astype(float),
            'is_newyork': ((hour >= 13) & (hour < 22)).astype(float),
            'is_overlap_london_ny': ((hour >= 13) & (hour < 16)).astype(float),
        }


# =============================================================================
# COMPREHENSIVE MEASUREMENT ENGINE
# =============================================================================

class MeasurementEngine:
    """
    Comprehensive measurement engine.

    Computes ALL measurements for exploration.
    Tracks correlations between measurements.
    Discovers what matters per asset class.
    """

    def __init__(self):
        self.measurement_history: Dict[str, List[np.ndarray]] = defaultdict(list)
        self.correlation_matrix: Optional[np.ndarray] = None
        self.measurement_names: List[str] = []

    def compute_all(self,
                    open_: np.ndarray,
                    high: np.ndarray,
                    low: np.ndarray,
                    close: np.ndarray,
                    volume: np.ndarray,
                    spread: np.ndarray,
                    timestamps: Optional[pd.DatetimeIndex] = None) -> Dict[str, np.ndarray]:
        """
        Compute ALL measurements.

        Returns dict of measurement_name -> array.
        """
        measurements = {}

        # === VOLATILITY ===
        measurements['vol_atr'] = VolatilityMeasures.atr(high, low, close)
        measurements['vol_yang_zhang'] = VolatilityMeasures.yang_zhang(open_, high, low, close)
        measurements['vol_parkinson'] = VolatilityMeasures.parkinson(high, low)
        measurements['vol_rogers_satchell'] = VolatilityMeasures.rogers_satchell(open_, high, low, close)
        measurements['vol_garman_klass'] = VolatilityMeasures.garman_klass(open_, high, low, close)
        measurements['vol_of_vol'] = VolatilityMeasures.volatility_of_volatility(measurements['vol_yang_zhang'])

        # === MOMENTUM ===
        roc_dict = MomentumMeasures.roc(close)
        measurements.update(roc_dict)

        measurements['rsi_14'] = MomentumMeasures.rsi(close, 14)
        measurements['rsi_7'] = MomentumMeasures.rsi(close, 7)
        measurements['rsi_21'] = MomentumMeasures.rsi(close, 21)

        macd, macd_signal, macd_hist = MomentumMeasures.macd(close)
        measurements['macd'] = macd
        measurements['macd_signal'] = macd_signal
        measurements['macd_histogram'] = macd_hist

        adx, plus_di, minus_di = MomentumMeasures.adx(high, low, close)
        measurements['adx'] = adx
        measurements['plus_di'] = plus_di
        measurements['minus_di'] = minus_di

        aroon_up, aroon_down = MomentumMeasures.aroon(high, low)
        measurements['aroon_up'] = aroon_up
        measurements['aroon_down'] = aroon_down
        measurements['aroon_oscillator'] = aroon_up - aroon_down

        measurements['momentum_divergence'] = MomentumMeasures.momentum_divergence(
            close, measurements['rsi_14']
        )

        # === MEAN REVERSION ===
        measurements['bollinger_pct_b'] = MeanReversionMeasures.bollinger_percent_b(close)
        measurements['zscore_20'] = MeanReversionMeasures.z_score(close, 20)
        measurements['zscore_50'] = MeanReversionMeasures.z_score(close, 50)
        measurements['hurst'] = MeanReversionMeasures.hurst_exponent(close)
        measurements['dist_from_vwap'] = MeanReversionMeasures.distance_from_vwap(close, volume)

        # === ENERGY & FLOW (Physics) ===
        measurements['kinetic_energy'] = EnergyFlowMeasures.kinetic_energy(close)
        measurements['potential_energy'] = EnergyFlowMeasures.potential_energy(close)
        measurements['energy_release_rate'] = EnergyFlowMeasures.energy_release_rate(
            close, measurements['vol_yang_zhang']
        )
        measurements['reynolds'] = EnergyFlowMeasures.reynolds_number_proxy(
            close, measurements['vol_yang_zhang']
        )
        measurements['reynolds_roc_inverse'] = EnergyFlowMeasures.reynolds_roc_inverse(
            measurements['reynolds'], measurements['roc_10']
        )
        measurements['flow_regime'] = EnergyFlowMeasures.flow_regime(measurements['reynolds'])
        measurements['entropy_rate'] = EnergyFlowMeasures.entropy_rate(close)

        # === MICROSTRUCTURE ===
        measurements['spread_ratio'] = MicrostructureMeasures.spread_ratio(spread)
        measurements['volume_ratio'] = MicrostructureMeasures.volume_ratio(volume)
        measurements['volume_price_trend'] = MicrostructureMeasures.volume_price_trend(close, volume)
        measurements['tick_intensity'] = MicrostructureMeasures.tick_intensity(volume)
        measurements['liquidity_score'] = MicrostructureMeasures.liquidity_score(spread, volume)

        # === TIME (if timestamps available) ===
        if timestamps is not None:
            time_features = TimeMeasures.extract_time_features(timestamps)
            measurements.update({f'time_{k}': v for k, v in time_features.items()})

            session_features = TimeMeasures.session_indicator(time_features['hour'])
            measurements.update({f'session_{k}': v for k, v in session_features.items()})

        # === CROSS-MEASUREMENTS (Ratios & Interactions) ===
        # Volatility ratios
        measurements['vol_ratio_yz_atr'] = measurements['vol_yang_zhang'] / (measurements['vol_atr'] + 1e-10)
        measurements['vol_ratio_pk_atr'] = measurements['vol_parkinson'] / (measurements['vol_atr'] + 1e-10)

        # Momentum-volatility interaction
        measurements['momentum_vol_ratio'] = np.abs(measurements['roc_10']) / (measurements['vol_yang_zhang'] + 1e-10)

        # Energy-momentum interaction
        measurements['energy_momentum'] = measurements['kinetic_energy'] * np.sign(measurements['roc_10'])

        # Hurst-based regime indicator
        measurements['is_mean_reverting'] = (measurements['hurst'] < 0.45).astype(float)
        measurements['is_trending'] = (measurements['hurst'] > 0.55).astype(float)
        measurements['is_random_walk'] = ((measurements['hurst'] >= 0.45) & (measurements['hurst'] <= 0.55)).astype(float)

        self.measurement_names = list(measurements.keys())
        return measurements

    def normalize_measurements(self, measurements: Dict[str, np.ndarray],
                               lookback: int = 200) -> Dict[str, np.ndarray]:
        """
        Normalize all measurements to z-scores for RL.

        Uses rolling normalization to avoid lookahead.
        """
        normalized = {}

        for name, values in measurements.items():
            n = len(values)
            z = np.zeros(n)

            for i in range(lookback, n):
                window = values[max(0, i-lookback):i]
                mean = np.mean(window)
                std = np.std(window)
                if std > 0:
                    z[i] = (values[i] - mean) / std
                else:
                    z[i] = 0

            normalized[f'{name}_z'] = z

        return normalized

    def compute_correlation_matrix(self, measurements: Dict[str, np.ndarray],
                                   start_idx: int = 200) -> pd.DataFrame:
        """
        Compute correlation matrix between all measurements.

        This reveals relationships - including inverses during volatility.
        """
        # Stack all measurements into matrix
        names = list(measurements.keys())
        data = np.column_stack([measurements[n][start_idx:] for n in names])

        # Remove NaN/inf
        valid_mask = np.all(np.isfinite(data), axis=1)
        data = data[valid_mask]

        # Compute correlation
        corr = np.corrcoef(data.T)

        self.correlation_matrix = corr
        return pd.DataFrame(corr, index=names, columns=names)

    def find_inverse_relationships(self, corr_df: pd.DataFrame,
                                   threshold: float = -0.5) -> List[Tuple[str, str, float]]:
        """
        Find strongly inverse relationships.

        These are candidates for regime-dependent strategies.
        """
        inverses = []

        names = corr_df.columns
        for i, name1 in enumerate(names):
            for j, name2 in enumerate(names):
                if i < j:  # Upper triangle only
                    corr = corr_df.iloc[i, j]
                    if corr < threshold:
                        inverses.append((name1, name2, corr))

        return sorted(inverses, key=lambda x: x[2])

    def get_feature_array(self, measurements: Dict[str, np.ndarray],
                          idx: int) -> np.ndarray:
        """Get feature array for a single bar."""
        return np.array([measurements[name][idx] for name in self.measurement_names])


# =============================================================================
# CORRELATION EXPLORER
# =============================================================================

class CorrelationExplorer:
    """
    Explore correlations between measurements and outcomes.

    KEY: What correlates with GOOD trades (high PnL, low MAE)?
    This may differ by asset class.
    """

    def __init__(self):
        self.trade_measurements: List[Dict[str, float]] = []
        self.trade_outcomes: List[Dict[str, float]] = []

    def record_trade(self, entry_measurements: Dict[str, float],
                     exit_measurements: Dict[str, float],
                     pnl: float, mae: float, mfe: float, bars_held: int):
        """Record a trade with its measurements and outcome."""
        # Store entry measurements
        entry_record = {f'entry_{k}': v for k, v in entry_measurements.items()}
        exit_record = {f'exit_{k}': v for k, v in exit_measurements.items()}

        self.trade_measurements.append({**entry_record, **exit_record})
        self.trade_outcomes.append({
            'pnl': pnl,
            'mae': mae,
            'mfe': mfe,
            'bars_held': bars_held,
            'edge_ratio': mfe / (abs(mae) + abs(mfe) + 1e-10),
            'profit_factor': mfe / (abs(mae) + 1e-10) if mae < 0 else float('inf'),
        })

    def analyze_correlations(self) -> pd.DataFrame:
        """
        Analyze which entry measurements correlate with good outcomes.

        Returns DataFrame with correlations.
        """
        if len(self.trade_measurements) < 30:
            return pd.DataFrame()

        # Convert to DataFrames
        meas_df = pd.DataFrame(self.trade_measurements)
        out_df = pd.DataFrame(self.trade_outcomes)

        # Compute correlations between entry features and outcomes
        correlations = {}

        for meas_col in meas_df.columns:
            if meas_col.startswith('entry_'):
                correlations[meas_col] = {}
                for out_col in out_df.columns:
                    corr = meas_df[meas_col].corr(out_df[out_col])
                    correlations[meas_col][out_col] = corr

        return pd.DataFrame(correlations).T

    def find_predictive_features(self, min_corr: float = 0.2) -> Dict[str, List[str]]:
        """
        Find which features predict which outcomes.

        Returns dict: outcome -> list of predictive features.
        """
        corr_df = self.analyze_correlations()
        if corr_df.empty:
            return {}

        predictive = defaultdict(list)

        for outcome in corr_df.columns:
            for feature in corr_df.index:
                corr = abs(corr_df.loc[feature, outcome])
                if corr >= min_corr:
                    predictive[outcome].append((feature, corr_df.loc[feature, outcome]))

        # Sort by absolute correlation
        for outcome in predictive:
            predictive[outcome] = sorted(predictive[outcome], key=lambda x: abs(x[1]), reverse=True)

        return dict(predictive)


# Export
__all__ = [
    'MeasurementCategory',
    'Measurement',
    'MeasurementSet',
    'VolatilityMeasures',
    'MomentumMeasures',
    'MeanReversionMeasures',
    'EnergyFlowMeasures',
    'MicrostructureMeasures',
    'TimeMeasures',
    'MeasurementEngine',
    'CorrelationExplorer',
]
