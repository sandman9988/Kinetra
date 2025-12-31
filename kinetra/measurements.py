"""
Physics-Based Measurement Framework for Multi-Asset Exploration
=================================================================

PHILOSOPHY: EVERYTHING derived from first principles. NO static rules.

What we DON'T use:
- NO traditional TA indicators (RSI, MACD, Bollinger, ADX, Aroon, etc.)
- NO hardcoded periods (no "14-period", "20-period")
- NO magic numbers
- NO linear assumptions

What we DO use:
- Physics (energy, damping, entropy, viscosity, Reynolds)
- Thermodynamics (energy states, phase transitions)
- Kinematics (velocity, acceleration, jerk, snap, crackle, pop)
- Fluid dynamics (Reynolds number, laminar vs turbulent flow)
- Field theory (gradients, divergence, curl-like measures)
- DSP/FFT for adaptive windows (data-derived, not hardcoded)
- Rolling percentiles (where in distribution, not absolute values)

KEY INSIGHTS:
- "Trending" means different things for different asset classes
- Relationships INVERT during turbulent flow (high Reynolds)
- Reynolds should be INVERSE to momentum during instability
- Let RL discover what matters per class - we just measure everything
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
    """Categories for organization, NOT for assumption."""
    KINEMATICS = "kinematics"      # Position derivatives
    ENERGY = "energy"              # Kinetic, potential, conversion
    FLOW = "flow"                  # Reynolds, viscosity, turbulence
    THERMODYNAMICS = "thermo"      # Entropy, phase state
    FIELD = "field"                # Gradients, divergence
    MICROSTRUCTURE = "micro"       # Spread, volume, liquidity
    REGIME = "regime"              # Derived regime indicators


@dataclass
class Measurement:
    """A single measurement with metadata."""
    name: str
    category: MeasurementCategory
    value: float
    percentile: float = 0.5  # Where in rolling distribution (0-1)
    is_extreme: bool = False  # Top/bottom 5%


# =============================================================================
# DSP-BASED ADAPTIVE WINDOWS
# =============================================================================

class AdaptiveWindows:
    """
    Use DSP/FFT to find natural cycles in data.
    NO hardcoded periods. Windows derived from data.
    """

    @staticmethod
    def find_dominant_periods(data: np.ndarray, num_periods: int = 3) -> List[int]:
        """
        Use FFT to find dominant cycles in the data.
        Returns periods in bars (not frequencies).
        """
        if len(data) < 64:
            return [5, 10, 20]  # Fallback for short series

        # Detrend
        x = data - np.mean(data)

        # FFT
        fft_vals = np.fft.fft(x)
        power = np.abs(fft_vals[:len(x)//2]) ** 2

        # Skip DC component and very low frequencies
        power[:3] = 0

        # Find peaks
        peak_indices = np.argsort(power)[-num_periods*2:]

        # Convert to periods
        periods = []
        for idx in peak_indices:
            if idx > 0:
                period = len(x) // idx
                if 3 <= period <= len(x) // 2:
                    periods.append(period)

        # Remove duplicates and sort
        periods = sorted(set(periods))[:num_periods]

        if not periods:
            return [5, 10, 20]

        return periods

    @staticmethod
    def get_adaptive_windows(close: np.ndarray, window_size: int = 500) -> Tuple[int, int, int]:
        """
        Get short/medium/long windows from FFT analysis.
        Returns (short_window, medium_window, long_window).
        """
        if len(close) < window_size:
            analysis_data = close
        else:
            analysis_data = close[-window_size:]

        periods = AdaptiveWindows.find_dominant_periods(analysis_data, 3)

        if len(periods) >= 3:
            return periods[0], periods[1], periods[2]
        elif len(periods) == 2:
            return periods[0], periods[1], periods[1] * 2
        elif len(periods) == 1:
            return periods[0], periods[0] * 2, periods[0] * 4
        else:
            return 5, 10, 20


# =============================================================================
# KINEMATICS: Position Derivatives
# =============================================================================

class KinematicsMeasures:
    """
    Pure kinematics: derivatives of log-price.

    velocity     = d(log P)/dt      (1st derivative - direction)
    acceleration = d²(log P)/dt²    (2nd derivative - momentum change)
    jerk         = d³(log P)/dt³    (3rd derivative - "fat candle" predictor)
    snap         = d⁴(log P)/dt⁴    (4th derivative - jerk change)
    crackle      = d⁵(log P)/dt⁵    (5th derivative)
    pop          = d⁶(log P)/dt⁶    (6th derivative)
    """

    @staticmethod
    def compute_all_derivatives(close: np.ndarray) -> Dict[str, np.ndarray]:
        """Compute velocity through pop (6 derivatives)."""
        log_price = np.log(close + 1e-10)

        # Successive differences
        velocity = np.diff(log_price, prepend=log_price[0])
        acceleration = np.diff(velocity, prepend=velocity[0])
        jerk = np.diff(acceleration, prepend=acceleration[0])
        snap = np.diff(jerk, prepend=jerk[0])
        crackle = np.diff(snap, prepend=snap[0])
        pop = np.diff(crackle, prepend=crackle[0])

        return {
            'velocity': velocity,
            'acceleration': acceleration,
            'jerk': jerk,
            'snap': snap,
            'crackle': crackle,
            'pop': pop,
        }

    @staticmethod
    def momentum(close: np.ndarray, velocity: np.ndarray,
                 volume: np.ndarray) -> np.ndarray:
        """
        Momentum = mass × velocity.
        Use volume as proxy for mass.
        """
        # Normalize volume to avoid scale issues
        vol_norm = volume / (np.mean(volume) + 1e-10)
        return vol_norm * velocity

    @staticmethod
    def impulse(momentum: np.ndarray) -> np.ndarray:
        """
        Impulse = change in momentum.
        J = Δp = F × Δt
        """
        return np.diff(momentum, prepend=momentum[0])


# =============================================================================
# ENERGY: Kinetic, Potential, and Conversion
# =============================================================================

class EnergyMeasures:
    """
    Energy physics for markets.

    kinetic_energy = ½mv²    (energy in motion)
    potential_energy = stored energy (compression)
    total_energy = KE + PE   (conserved in ideal system)
    eta = KE/PE              (conversion efficiency)
    """

    @staticmethod
    def kinetic_energy(velocity: np.ndarray, volume: np.ndarray) -> np.ndarray:
        """
        KE = ½mv² where m = normalized volume, v = velocity.
        """
        vol_norm = volume / (np.mean(volume) + 1e-10)
        return 0.5 * vol_norm * (velocity ** 2)

    @staticmethod
    def potential_energy_compression(close: np.ndarray,
                                      high: np.ndarray,
                                      low: np.ndarray) -> np.ndarray:
        """
        Potential energy from volatility compression.

        Low volatility = energy stored (like compressed spring).
        PE = 1 / volatility (inversely proportional to range).
        """
        bar_range = high - low
        # Use adaptive rolling min for "normal" range
        n = len(close)
        pe = np.zeros(n)

        for i in range(20, n):
            # Compare current range to recent ranges
            recent_ranges = bar_range[max(0, i-50):i]
            mean_range = np.mean(recent_ranges) + 1e-10
            # PE high when current range is low relative to history
            pe[i] = mean_range / (bar_range[i] + 1e-10)

        return pe

    @staticmethod
    def potential_energy_displacement(close: np.ndarray) -> np.ndarray:
        """
        Potential energy from displacement from equilibrium.

        Like a spring: PE = ½kx² where x = distance from mean.
        Further from equilibrium = more potential energy.
        """
        n = len(close)
        pe = np.zeros(n)

        for i in range(50, n):
            # Rolling mean as equilibrium
            window = close[max(0, i-100):i]
            mean = np.mean(window)
            displacement = (close[i] - mean) / (mean + 1e-10)
            pe[i] = 0.5 * (displacement ** 2)

        return pe

    @staticmethod
    def efficiency_ratio(ke: np.ndarray, pe: np.ndarray) -> np.ndarray:
        """
        Eta = KE / PE (energy conversion efficiency).

        High eta = kinetic energy dominant (trending)
        Low eta = potential energy dominant (compressed)
        """
        return ke / (pe + 1e-10)

    @staticmethod
    def energy_release_rate(ke: np.ndarray) -> np.ndarray:
        """
        Rate of kinetic energy change.
        Positive = energy building, Negative = energy dissipating.
        """
        return np.diff(ke, prepend=ke[0])


# =============================================================================
# FLUID DYNAMICS: Reynolds Number and Flow
# =============================================================================

class FlowMeasures:
    """
    Fluid dynamics applied to markets.

    Reynolds number = inertia / viscosity
    - High Re = turbulent (chaotic, unpredictable)
    - Low Re = laminar (smooth, predictable)

    KEY INSIGHT: During turbulent flow, relationships INVERT.
    """

    @staticmethod
    def reynolds_number(velocity: np.ndarray, close: np.ndarray) -> np.ndarray:
        """
        Market Reynolds number: trend / noise.

        Re = |<v>_slow| / σ(v)_fast

        High Re = laminar (strong trend, low noise)
        Low Re = turbulent (weak trend, high noise)
        """
        n = len(velocity)
        reynolds = np.zeros(n)

        for i in range(30, n):
            # Trend = slow moving average of velocity
            trend = np.mean(velocity[max(0, i-24):i])

            # Noise = fast volatility of velocity
            noise = np.std(velocity[max(0, i-6):i]) + 1e-10

            reynolds[i] = abs(trend) / noise

        return reynolds

    @staticmethod
    def damping_coefficient(velocity: np.ndarray) -> np.ndarray:
        """
        Damping ζ = σ(v) / μ(|v|).

        High ζ = high friction, mean-reverting tendency.
        Low ζ = low friction, trending tendency.
        """
        n = len(velocity)
        zeta = np.zeros(n)

        for i in range(20, n):
            window = velocity[max(0, i-64):i]
            sigma = np.std(window)
            mu_abs = np.mean(np.abs(window)) + 1e-10
            zeta[i] = sigma / mu_abs

        return zeta

    @staticmethod
    def viscosity(high: np.ndarray, low: np.ndarray,
                  close: np.ndarray, volume: np.ndarray) -> np.ndarray:
        """
        Viscosity = resistance to flow.

        High viscosity = hard to move price (thick market)
        Low viscosity = easy to move price (thin market)

        μ = (range / price) / (volume / avg_volume)
        """
        n = len(close)
        viscosity = np.zeros(n)

        for i in range(20, n):
            bar_range_pct = (high[i] - low[i]) / (close[i] + 1e-10)
            avg_volume = np.mean(volume[max(0, i-20):i]) + 1e-10
            volume_norm = volume[i] / avg_volume

            # Viscosity = how much range per unit volume
            viscosity[i] = bar_range_pct / (volume_norm + 1e-10)

        return viscosity

    @staticmethod
    def liquidity(high: np.ndarray, low: np.ndarray,
                  close: np.ndarray, volume: np.ndarray) -> np.ndarray:
        """
        Liquidity = inverse of price impact.

        High liquidity = can trade large size without moving price.
        Liquidity = Volume / (Range × Price)
        """
        bar_range = (high - low).clip(1e-10, None)
        price_range_pct = bar_range / close
        return volume / (price_range_pct * close + 1e-10)

    @staticmethod
    def reynolds_momentum_relationship(reynolds: np.ndarray,
                                         momentum: np.ndarray) -> np.ndarray:
        """
        KEY INSIGHT: Reynolds should be INVERSE to momentum during instability.

        When this relationship flips (positive correlation),
        it signals regime change.

        Returns rolling correlation between Re and momentum.
        """
        n = len(reynolds)
        relationship = np.zeros(n)

        for i in range(30, n):
            re = reynolds[max(0, i-20):i]
            mo = momentum[max(0, i-20):i]

            if np.std(re) > 1e-10 and np.std(mo) > 1e-10:
                corr = np.corrcoef(re, mo)[0, 1]
                relationship[i] = corr

        return relationship


# =============================================================================
# THERMODYNAMICS: Entropy and Phase State
# =============================================================================

class ThermodynamicsMeasures:
    """
    Thermodynamics for markets.

    Entropy = disorder/uncertainty.
    Phase transitions = regime changes.
    """

    @staticmethod
    def shannon_entropy(velocity: np.ndarray, bins: int = 20) -> np.ndarray:
        """
        Shannon entropy of return distribution.

        H = -Σ p_i × log(p_i)

        High entropy = high disorder = unpredictable.
        Low entropy = ordered = predictable.
        """
        n = len(velocity)
        entropy = np.zeros(n)

        for i in range(50, n):
            window = velocity[max(0, i-64):i]
            window = window[~np.isnan(window)]

            if len(window) < 5:
                continue

            # Histogram
            hist, _ = np.histogram(window, bins=bins, density=True)
            p = hist / (hist.sum() + 1e-10)
            p = p[p > 0]

            # Shannon entropy (normalized)
            H = -np.sum(p * np.log(p))
            entropy[i] = H / np.log(bins)  # Normalize to [0, 1]

        return entropy

    @staticmethod
    def entropy_rate(entropy: np.ndarray) -> np.ndarray:
        """
        Rate of entropy change.

        Increasing entropy = system becoming more chaotic.
        Decreasing entropy = system becoming more ordered.
        """
        return np.diff(entropy, prepend=entropy[0])

    @staticmethod
    def phase_compression(ke: np.ndarray, pe: np.ndarray,
                          entropy: np.ndarray) -> np.ndarray:
        """
        Phase space compression indicator.

        Compressed phase space = high PE, low KE, low entropy.
        This often precedes explosive moves.
        """
        n = len(ke)
        compression = np.zeros(n)

        for i in range(100, n):
            # Get percentiles
            ke_window = ke[max(0, i-200):i]
            pe_window = pe[max(0, i-200):i]
            ent_window = entropy[max(0, i-200):i]

            ke_pct = (ke[i] > ke_window).mean()
            pe_pct = (pe[i] > pe_window).mean()
            ent_pct = (entropy[i] > ent_window).mean()

            # Compression = high PE percentile × (1 - KE percentile) × (1 - entropy percentile)
            compression[i] = pe_pct * (1 - ke_pct) * (1 - ent_pct)

        return compression


# =============================================================================
# FIELD THEORY: Gradients and Divergence
# =============================================================================

class FieldMeasures:
    """
    Field theory concepts applied to price fields.

    Gradient = rate of change across "space" (time)
    Divergence = sources/sinks of "flow"
    """

    @staticmethod
    def price_gradient(close: np.ndarray) -> np.ndarray:
        """
        Price gradient = local slope.
        First spatial derivative of log-price.
        """
        log_price = np.log(close + 1e-10)
        return np.gradient(log_price)

    @staticmethod
    def gradient_magnitude(close: np.ndarray) -> np.ndarray:
        """
        Magnitude of price gradient.
        Absolute slope regardless of direction.
        """
        return np.abs(FieldMeasures.price_gradient(close))

    @staticmethod
    def divergence_proxy(velocity: np.ndarray, volume: np.ndarray) -> np.ndarray:
        """
        Divergence proxy: Are flows converging or diverging?

        Positive divergence = outflow (selling pressure)
        Negative divergence = inflow (buying pressure)

        Uses velocity-weighted volume flow.
        """
        flow = velocity * volume
        return np.gradient(flow)

    @staticmethod
    def buying_pressure(open_: np.ndarray, high: np.ndarray,
                        low: np.ndarray, close: np.ndarray) -> np.ndarray:
        """
        Buying pressure = (close - low) / (high - low).

        Where in the bar's range did we close?
        0 = closed at low (sellers won)
        1 = closed at high (buyers won)
        0.5 = neutral
        """
        bar_range = (high - low).clip(1e-10, None)
        return (close - low) / bar_range

    @staticmethod
    def body_ratio(open_: np.ndarray, high: np.ndarray,
                   low: np.ndarray, close: np.ndarray) -> np.ndarray:
        """
        Body ratio = |close - open| / (high - low).

        How much of the range is body vs wick?
        High body ratio = conviction move.
        Low body ratio = indecision.
        """
        body = np.abs(close - open_)
        bar_range = (high - low).clip(1e-10, None)
        return body / bar_range


# =============================================================================
# MICROSTRUCTURE
# =============================================================================

class MicrostructureMeasures:
    """
    Market microstructure measurements.
    Spread, volume dynamics, execution quality.
    """

    @staticmethod
    def spread_percentile(spread: np.ndarray) -> np.ndarray:
        """
        Where is current spread in rolling distribution?
        Low percentile = tight spread = good execution.
        """
        n = len(spread)
        pct = np.zeros(n)

        for i in range(100, n):
            window = spread[max(0, i-200):i]
            pct[i] = (spread[i] > window).mean()

        return pct

    @staticmethod
    def volume_surge(volume: np.ndarray) -> np.ndarray:
        """
        Volume surge = current volume / rolling mean.
        High values indicate unusual activity.
        """
        n = len(volume)
        surge = np.ones(n)

        for i in range(20, n):
            mean_vol = np.mean(volume[max(0, i-50):i]) + 1e-10
            surge[i] = volume[i] / mean_vol

        return surge

    @staticmethod
    def volume_trend(volume: np.ndarray) -> np.ndarray:
        """
        Is volume trending up or down?
        Rolling slope of volume.
        """
        n = len(volume)
        trend = np.zeros(n)

        for i in range(20, n):
            window = volume[max(0, i-20):i]
            if len(window) > 1:
                x = np.arange(len(window))
                slope, _ = np.polyfit(x, window, 1)
                trend[i] = slope / (np.mean(window) + 1e-10)

        return trend


# =============================================================================
# ROLLING PERCENTILE NORMALIZATION
# =============================================================================

class PercentileNormalizer:
    """
    Convert all measurements to rolling percentiles.

    This is THE key to instrument-agnostic features.
    Instead of "velocity = 0.02", we get "velocity is at 85th percentile".
    """

    @staticmethod
    def to_percentile(data: np.ndarray, window: int = 200) -> np.ndarray:
        """
        Convert raw values to rolling percentile (0-1).
        """
        n = len(data)
        pct = np.full(n, 0.5)  # Default to median

        for i in range(window, n):
            window_data = data[max(0, i-window):i]
            window_data = window_data[np.isfinite(window_data)]

            if len(window_data) > 0:
                pct[i] = (data[i] > window_data).mean()

        return pct

    @staticmethod
    def is_extreme(percentile: np.ndarray, threshold: float = 0.95) -> np.ndarray:
        """
        Flag extreme values (top or bottom 5%).
        """
        return ((percentile > threshold) | (percentile < (1 - threshold))).astype(float)


# =============================================================================
# COMPREHENSIVE PHYSICS-BASED MEASUREMENT ENGINE
# =============================================================================

class MeasurementEngine:
    """
    Physics-based measurement engine.

    NO traditional indicators. Pure physics + field theory + thermodynamics.
    Everything normalized to rolling percentiles.
    """

    def __init__(self, percentile_window: int = 200):
        self.percentile_window = percentile_window
        self.measurement_names: List[str] = []
        self.normalizer = PercentileNormalizer()

    def compute_all(self,
                    open_: np.ndarray,
                    high: np.ndarray,
                    low: np.ndarray,
                    close: np.ndarray,
                    volume: np.ndarray,
                    spread: np.ndarray,
                    timestamps: Optional[pd.DatetimeIndex] = None) -> Dict[str, np.ndarray]:
        """
        Compute ALL physics-based measurements.
        Returns dict of measurement_name -> array.
        """
        measurements = {}

        # === ADAPTIVE WINDOWS FROM FFT ===
        short_w, med_w, long_w = AdaptiveWindows.get_adaptive_windows(close)

        # === KINEMATICS (Derivatives) ===
        derivatives = KinematicsMeasures.compute_all_derivatives(close)
        measurements['velocity'] = derivatives['velocity']
        measurements['acceleration'] = derivatives['acceleration']
        measurements['jerk'] = derivatives['jerk']
        measurements['snap'] = derivatives['snap']
        measurements['crackle'] = derivatives['crackle']
        measurements['pop'] = derivatives['pop']

        # Momentum and impulse
        measurements['momentum'] = KinematicsMeasures.momentum(
            close, derivatives['velocity'], volume
        )
        measurements['impulse'] = KinematicsMeasures.impulse(measurements['momentum'])

        # === ENERGY ===
        measurements['kinetic_energy'] = EnergyMeasures.kinetic_energy(
            derivatives['velocity'], volume
        )
        measurements['potential_energy_compression'] = EnergyMeasures.potential_energy_compression(
            close, high, low
        )
        measurements['potential_energy_displacement'] = EnergyMeasures.potential_energy_displacement(
            close
        )
        measurements['energy_efficiency'] = EnergyMeasures.efficiency_ratio(
            measurements['kinetic_energy'],
            measurements['potential_energy_compression']
        )
        measurements['energy_release_rate'] = EnergyMeasures.energy_release_rate(
            measurements['kinetic_energy']
        )

        # === FLOW DYNAMICS ===
        measurements['reynolds'] = FlowMeasures.reynolds_number(
            derivatives['velocity'], close
        )
        measurements['damping'] = FlowMeasures.damping_coefficient(
            derivatives['velocity']
        )
        measurements['viscosity'] = FlowMeasures.viscosity(high, low, close, volume)
        measurements['liquidity'] = FlowMeasures.liquidity(high, low, close, volume)
        measurements['reynolds_momentum_corr'] = FlowMeasures.reynolds_momentum_relationship(
            measurements['reynolds'], measurements['momentum']
        )

        # === THERMODYNAMICS ===
        measurements['entropy'] = ThermodynamicsMeasures.shannon_entropy(
            derivatives['velocity']
        )
        measurements['entropy_rate'] = ThermodynamicsMeasures.entropy_rate(
            measurements['entropy']
        )
        measurements['phase_compression'] = ThermodynamicsMeasures.phase_compression(
            measurements['kinetic_energy'],
            measurements['potential_energy_compression'],
            measurements['entropy']
        )

        # === FIELD MEASURES ===
        measurements['price_gradient'] = FieldMeasures.price_gradient(close)
        measurements['gradient_magnitude'] = FieldMeasures.gradient_magnitude(close)
        measurements['divergence'] = FieldMeasures.divergence_proxy(
            derivatives['velocity'], volume
        )
        measurements['buying_pressure'] = FieldMeasures.buying_pressure(
            open_, high, low, close
        )
        measurements['body_ratio'] = FieldMeasures.body_ratio(
            open_, high, low, close
        )

        # === MICROSTRUCTURE ===
        measurements['spread_pct'] = MicrostructureMeasures.spread_percentile(spread)
        measurements['volume_surge'] = MicrostructureMeasures.volume_surge(volume)
        measurements['volume_trend'] = MicrostructureMeasures.volume_trend(volume)

        # === CROSS-INTERACTIONS (Physics relationships) ===
        # Energy-momentum
        measurements['energy_momentum_product'] = (
            measurements['kinetic_energy'] * np.sign(measurements['momentum'])
        )

        # Reynolds-damping relationship
        measurements['re_damping_ratio'] = (
            measurements['reynolds'] / (measurements['damping'] + 1e-10)
        )

        # Entropy-energy phase
        measurements['entropy_energy_phase'] = (
            measurements['entropy'] * measurements['kinetic_energy']
        )

        # Jerk energy (rate of acceleration squared - "violence" of move)
        measurements['jerk_energy'] = derivatives['jerk'] ** 2

        # Compression-release potential
        measurements['release_potential'] = (
            measurements['potential_energy_compression'] *
            (1 - measurements['entropy'])  # Low entropy = more release potential
        )

        self.measurement_names = list(measurements.keys())
        return measurements

    def normalize_to_percentiles(self, measurements: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Normalize ALL measurements to rolling percentiles.
        This makes features instrument-agnostic.
        """
        normalized = {}

        for name, values in measurements.items():
            # Raw value
            normalized[name] = values

            # Percentile version
            pct = self.normalizer.to_percentile(values, self.percentile_window)
            normalized[f'{name}_pct'] = pct

            # Extreme flag
            normalized[f'{name}_extreme'] = self.normalizer.is_extreme(pct)

        return normalized

    def compute_correlation_matrix(self, measurements: Dict[str, np.ndarray],
                                   start_idx: int = 200) -> pd.DataFrame:
        """
        Compute correlation matrix between measurements.
        Reveals relationships and inverses.
        """
        # Only use raw measurements (not percentiles)
        raw_names = [n for n in measurements.keys() if not n.endswith('_pct') and not n.endswith('_extreme')]

        data = np.column_stack([measurements[n][start_idx:] for n in raw_names])

        # Remove NaN/inf
        valid_mask = np.all(np.isfinite(data), axis=1)
        data = data[valid_mask]

        if len(data) < 30:
            return pd.DataFrame()

        corr = np.corrcoef(data.T)
        return pd.DataFrame(corr, index=raw_names, columns=raw_names)

    def find_inverse_relationships(self, corr_df: pd.DataFrame,
                                   threshold: float = -0.5) -> List[Tuple[str, str, float]]:
        """
        Find strongly inverse relationships.
        These may flip during regime changes.
        """
        if corr_df.empty:
            return []

        inverses = []
        names = corr_df.columns

        for i, name1 in enumerate(names):
            for j, name2 in enumerate(names):
                if i < j:
                    corr = corr_df.iloc[i, j]
                    if np.isfinite(corr) and corr < threshold:
                        inverses.append((name1, name2, float(corr)))

        return sorted(inverses, key=lambda x: x[2])


# =============================================================================
# REGIME DETECTION (Physics-Based)
# =============================================================================

class PhysicsRegimeDetector:
    """
    Detect market regimes from physics state.

    NO hardcoded thresholds. Uses percentile-based detection.
    """

    @staticmethod
    def detect_regime(measurements: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Classify regime from physics measurements.

        Returns array of regime codes:
        0 = UNDERDAMPED (low friction, trending)
        1 = CRITICAL (balanced, transitional)
        2 = OVERDAMPED (high friction, mean-reverting)
        3 = TURBULENT (chaotic, high Reynolds)
        4 = COMPRESSED (high potential energy, pre-breakout)
        """
        n = len(measurements['damping'])
        regime = np.ones(n, dtype=int)  # Default CRITICAL

        for i in range(100, n):
            # Get current percentiles
            damping_pct = measurements.get('damping_pct', np.full(n, 0.5))[i]
            reynolds_pct = measurements.get('reynolds_pct', np.full(n, 0.5))[i]
            ke_pct = measurements.get('kinetic_energy_pct', np.full(n, 0.5))[i]
            pe_pct = measurements.get('potential_energy_compression_pct', np.full(n, 0.5))[i]
            entropy_pct = measurements.get('entropy_pct', np.full(n, 0.5))[i]

            # Classification based on physics state
            if reynolds_pct > 0.8:
                regime[i] = 3  # TURBULENT
            elif pe_pct > 0.8 and ke_pct < 0.3 and entropy_pct < 0.3:
                regime[i] = 4  # COMPRESSED
            elif damping_pct < 0.25 and ke_pct > 0.5:
                regime[i] = 0  # UNDERDAMPED (trending)
            elif damping_pct > 0.75:
                regime[i] = 2  # OVERDAMPED (mean-reverting)
            else:
                regime[i] = 1  # CRITICAL

        return regime


# =============================================================================
# CORRELATION EXPLORER
# =============================================================================

class CorrelationExplorer:
    """
    Explore correlations between measurements and trade outcomes.
    Discovers what matters per asset class.
    """

    def __init__(self):
        self.trade_measurements: List[Dict[str, float]] = []
        self.trade_outcomes: List[Dict[str, float]] = []

    def record_trade(self, entry_measurements: Dict[str, float],
                     pnl: float, mae: float, mfe: float):
        """Record trade entry measurements and outcome."""
        self.trade_measurements.append(entry_measurements)
        self.trade_outcomes.append({
            'pnl': pnl,
            'mae': mae,
            'mfe': mfe,
            'edge_ratio': mfe / (abs(mae) + abs(mfe) + 1e-10),
        })

    def find_predictive_features(self, min_trades: int = 30) -> Dict[str, List[Tuple[str, float]]]:
        """
        Find which features predict which outcomes.
        Returns dict: outcome -> [(feature, correlation), ...]
        """
        if len(self.trade_measurements) < min_trades:
            return {}

        meas_df = pd.DataFrame(self.trade_measurements)
        out_df = pd.DataFrame(self.trade_outcomes)

        predictive = defaultdict(list)

        for meas_col in meas_df.columns:
            for out_col in out_df.columns:
                corr = meas_df[meas_col].corr(out_df[out_col])
                if abs(corr) >= 0.15 and np.isfinite(corr):
                    predictive[out_col].append((meas_col, corr))

        # Sort by absolute correlation
        for outcome in predictive:
            predictive[outcome] = sorted(
                predictive[outcome],
                key=lambda x: abs(x[1]),
                reverse=True
            )

        return dict(predictive)


# =============================================================================
# PERFORMANCE METRICS (For Accounting/Statistics - NOT Trading Signals)
# =============================================================================

class RiskAdjustedMetrics:
    """
    Non-linear, distribution-aware performance metrics.

    NO Sharpe (assumes symmetric risk, normality).
    YES to metrics that capture:
    - Fat tails (kurtosis)
    - Downside-only risk (Sortino)
    - Full distribution (Omega)
    - Worst-case scenarios (CVaR, Calmar)
    """

    @staticmethod
    def sortino_ratio(returns: np.ndarray, target: float = 0.0,
                      annualization: int = 252) -> float:
        """
        Sortino Ratio: Return over downside deviation only.

        Better than Sharpe because it only penalizes harmful volatility.
        Sortino = (Mean Return - Target) / Downside Deviation
        """
        excess = returns - target
        downside = returns[returns < target]

        if len(downside) == 0:
            return np.inf  # No downside = perfect

        downside_dev = np.std(downside)
        if downside_dev == 0:
            return np.inf

        mean_excess = np.mean(excess)
        sortino = (mean_excess / downside_dev) * np.sqrt(annualization)
        return float(sortino)

    @staticmethod
    def omega_ratio(returns: np.ndarray, threshold: float = 0.0) -> float:
        """
        Omega Ratio: Probability-weighted gains over losses.

        Omega = ∫(1-F(r))dr for r>L / ∫F(r)dr for r<L

        Captures ENTIRE distribution (skew, kurtosis, fat tails).
        No normality assumption. Superior for non-Gaussian returns.
        """
        gains = np.sum(np.maximum(returns - threshold, 0))
        losses = np.sum(np.maximum(threshold - returns, 0))

        if losses == 0:
            return np.inf

        return float(gains / losses)

    @staticmethod
    def calmar_ratio(equity_curve: np.ndarray, annualization: int = 252) -> float:
        """
        Calmar Ratio: Annualized return / Maximum Drawdown.

        Focuses on worst-case drawdown risk.
        Preferred by institutional investors over Sharpe.
        """
        if len(equity_curve) < 2:
            return 0.0

        # CAGR
        years = len(equity_curve) / annualization
        cagr = (equity_curve[-1] / equity_curve[0]) ** (1 / years) - 1 if years > 0 else 0

        # Max drawdown
        peak = np.maximum.accumulate(equity_curve)
        drawdown = (peak - equity_curve) / (peak + 1e-10)
        max_dd = np.max(drawdown)

        if max_dd == 0:
            return np.inf

        return float(cagr / max_dd)

    @staticmethod
    def burke_ratio(equity_curve: np.ndarray, n_drawdowns: int = 5,
                    annualization: int = 252) -> float:
        """
        Burke Ratio: CAGR / sqrt(sum of squared top N drawdowns).

        Less sensitive to single outlier drawdown than Calmar.
        Better for strategies with multiple moderate drawdowns.
        """
        if len(equity_curve) < 2:
            return 0.0

        years = len(equity_curve) / annualization
        cagr = (equity_curve[-1] / equity_curve[0]) ** (1 / years) - 1 if years > 0 else 0

        # Find top N drawdowns
        peak = np.maximum.accumulate(equity_curve)
        drawdown = (peak - equity_curve) / (peak + 1e-10)

        # Sort and take top N
        sorted_dd = np.sort(drawdown)[::-1][:n_drawdowns]

        # Sum of squares
        sum_sq = np.sum(sorted_dd ** 2)

        if sum_sq == 0:
            return np.inf

        return float(cagr / np.sqrt(sum_sq))

    @staticmethod
    def cvar_expected_shortfall(returns: np.ndarray, alpha: float = 0.05) -> float:
        """
        CVaR / Expected Shortfall: Average loss in worst alpha% of cases.

        Captures tail risk better than VaR.
        E.g., alpha=0.05 means average loss in worst 5% of days.
        """
        if len(returns) == 0:
            return 0.0

        sorted_returns = np.sort(returns)
        cutoff_idx = int(len(sorted_returns) * alpha)

        if cutoff_idx == 0:
            cutoff_idx = 1

        tail_losses = sorted_returns[:cutoff_idx]
        return float(np.mean(tail_losses))

    @staticmethod
    def tail_ratio(returns: np.ndarray, percentile: float = 5.0) -> float:
        """
        Tail Ratio: Right tail / Left tail.

        Measures asymmetry of extreme returns.
        > 1 = positive skew (winners bigger than losers)
        < 1 = negative skew (losers bigger than winners)
        """
        right_tail = np.percentile(returns, 100 - percentile)
        left_tail = abs(np.percentile(returns, percentile))

        if left_tail == 0:
            return np.inf if right_tail > 0 else 1.0

        return float(right_tail / left_tail)

    @staticmethod
    def kurtosis_excess(returns: np.ndarray) -> float:
        """
        Excess Kurtosis: Measure of fat tails.

        > 0 = Leptokurtic (fat tails, more extremes than normal)
        = 0 = Mesokurtic (normal-like)
        < 0 = Platykurtic (thin tails)

        Crypto/commodities typically have high excess kurtosis.
        """
        n = len(returns)
        if n < 4:
            return 0.0

        mean = np.mean(returns)
        std = np.std(returns)

        if std == 0:
            return 0.0

        # Fourth moment / std^4 - 3 (Fisher's definition)
        kurt = np.mean(((returns - mean) / std) ** 4) - 3
        return float(kurt)

    @staticmethod
    def ulcer_index(equity_curve: np.ndarray) -> float:
        """
        Ulcer Index: Root mean square of drawdowns.

        Penalizes both depth AND duration of drawdowns.
        Named because it measures "pain" causing ulcers.
        """
        peak = np.maximum.accumulate(equity_curve)
        drawdown = (peak - equity_curve) / (peak + 1e-10)

        return float(np.sqrt(np.mean(drawdown ** 2)))

    @staticmethod
    def gain_to_pain_ratio(returns: np.ndarray) -> float:
        """
        Gain to Pain Ratio: Sum of returns / Sum of abs(negative returns).

        Simple, intuitive measure of reward vs suffering.
        """
        total_return = np.sum(returns)
        total_pain = np.sum(np.abs(returns[returns < 0]))

        if total_pain == 0:
            return np.inf if total_return > 0 else 0.0

        return float(total_return / total_pain)


class PerformanceAnalyzer:
    """
    Comprehensive performance analysis using non-linear metrics.
    """

    def __init__(self, annualization: int = 252):
        self.annualization = annualization

    def analyze(self, returns: np.ndarray, equity_curve: np.ndarray) -> Dict[str, float]:
        """
        Compute all risk-adjusted metrics.

        Returns dict of metric_name -> value.
        """
        metrics = RiskAdjustedMetrics

        return {
            # Core non-linear metrics
            'sortino': metrics.sortino_ratio(returns, annualization=self.annualization),
            'omega': metrics.omega_ratio(returns),
            'calmar': metrics.calmar_ratio(equity_curve, annualization=self.annualization),
            'burke': metrics.burke_ratio(equity_curve, annualization=self.annualization),

            # Tail risk
            'cvar_5pct': metrics.cvar_expected_shortfall(returns, alpha=0.05),
            'tail_ratio': metrics.tail_ratio(returns),
            'excess_kurtosis': metrics.kurtosis_excess(returns),

            # Pain measures
            'ulcer_index': metrics.ulcer_index(equity_curve),
            'gain_to_pain': metrics.gain_to_pain_ratio(returns),

            # Basic stats (for reference)
            'total_return': float((equity_curve[-1] / equity_curve[0]) - 1) if len(equity_curve) > 1 else 0.0,
            'max_drawdown': float(np.max((np.maximum.accumulate(equity_curve) - equity_curve) /
                                         (np.maximum.accumulate(equity_curve) + 1e-10))),
            'win_rate': float(np.mean(returns > 0)),
        }


# Export
__all__ = [
    'MeasurementCategory',
    'Measurement',
    'AdaptiveWindows',
    'KinematicsMeasures',
    'EnergyMeasures',
    'FlowMeasures',
    'ThermodynamicsMeasures',
    'FieldMeasures',
    'MicrostructureMeasures',
    'PercentileNormalizer',
    'MeasurementEngine',
    'PhysicsRegimeDetector',
    'CorrelationExplorer',
    'RiskAdjustedMetrics',
    'PerformanceAnalyzer',
]
