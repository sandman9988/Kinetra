"""
Composite Stacking Framework for Multi-Asset Exploration
=========================================================

PHYSICS-ONLY SIGNAL GENERATION. NO TRADITIONAL INDICATORS.

Philosophy: Let the agent discover what works from PHYSICS state.

We DON'T use:
- RSI, MACD, ADX, Aroon, Bollinger Bands, VWAP
- Any traditional technical indicators
- Hardcoded periods or thresholds

We DO use:
- Kinematics: velocity, acceleration, jerk, snap, crackle, pop
- Energy: kinetic, potential, efficiency, release rate
- Flow: Reynolds, damping, viscosity, liquidity
- Thermodynamics: entropy, entropy rate, phase compression
- Field: gradients, divergence, buying pressure

NO ASSUMPTIONS about what "trending" or "mean-reverting" means.
Let the physics state and RL discover the patterns.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Callable
from enum import Enum
import numpy as np
import pandas as pd
from collections import defaultdict


# =============================================================================
# SIGNAL TYPES (Physics-Based, Discovered Not Assumed)
# =============================================================================

class SignalType(Enum):
    """
    Signal types based on physics categories.
    Agent discovers which physics features produce which patterns.
    """
    KINEMATICS = "kinematics"          # Motion derivatives
    ENERGY = "energy"                   # Energy state and transitions
    FLOW = "flow"                       # Fluid dynamics regime
    THERMODYNAMICS = "thermodynamics"   # Entropy and order
    MICROSTRUCTURE = "microstructure"   # Execution quality
    COMPOSITE = "composite"


@dataclass
class Signal:
    """A single signal with metadata about its source."""
    name: str
    signal_type: SignalType
    value: float  # -1 to +1 (short to long)
    confidence: float  # 0 to 1
    source_measurements: List[str] = field(default_factory=list)


# =============================================================================
# PHYSICS-BASED SIGNAL GENERATORS
# =============================================================================

class SignalGenerator:
    """
    Base class for physics-based signal generators.

    In exploration mode, we generate signals from physics measurements
    but DON'T assume they're correct. The agent learns which to trust.
    """

    def __init__(self, name: str, signal_type: SignalType):
        self.name = name
        self.signal_type = signal_type
        self.performance_history: List[float] = []

    def generate(self, measurements: Dict[str, float]) -> Signal:
        """Generate signal from measurements. Override in subclasses."""
        raise NotImplementedError

    def record_outcome(self, signal_value: float, actual_pnl: float):
        """Record how well the signal predicted the outcome."""
        agreement = signal_value * np.sign(actual_pnl)
        self.performance_history.append(agreement)

    def get_reliability(self, window: int = 100) -> float:
        """Get recent reliability of this signal."""
        if len(self.performance_history) < window:
            return 0.5  # Neutral
        recent = self.performance_history[-window:]
        return (np.mean(recent) + 1) / 2  # Scale to 0-1


class KinematicsSignalGenerator(SignalGenerator):
    """
    Signal from motion derivatives (velocity through pop).

    Uses 6 derivatives of log-price:
    - velocity (1st) - direction
    - acceleration (2nd) - momentum change
    - jerk (3rd) - "fat candle" predictor
    - snap (4th) - jerk change
    - crackle (5th), pop (6th) - higher order dynamics
    """

    def __init__(self):
        super().__init__("kinematics", SignalType.KINEMATICS)

    def generate(self, measurements: Dict[str, float]) -> Signal:
        components = []
        sources = []

        # Velocity (direction)
        velocity = measurements.get('velocity_pct', 0.5)
        components.append(velocity - 0.5)  # Center at 0
        sources.append('velocity')

        # Acceleration (momentum change)
        accel = measurements.get('acceleration_pct', 0.5)
        components.append(accel - 0.5)
        sources.append('acceleration')

        # Jerk (best fat candle predictor)
        jerk = measurements.get('jerk_pct', 0.5)
        components.append((jerk - 0.5) * 1.5)  # Weight jerk higher
        sources.append('jerk')

        # Momentum (volume-weighted velocity)
        momentum = measurements.get('momentum_pct', 0.5)
        components.append(momentum - 0.5)
        sources.append('momentum')

        # Impulse (change in momentum)
        impulse = measurements.get('impulse_pct', 0.5)
        components.append(impulse - 0.5)
        sources.append('impulse')

        if not components:
            return Signal(self.name, self.signal_type, 0, 0, [])

        # Weighted average (jerk gets more weight)
        signal_value = np.tanh(np.sum(components) / len(components) * 2)

        # Confidence: how aligned are the derivatives?
        confidence = 1.0 - np.std(components)

        return Signal(self.name, self.signal_type, signal_value, confidence, sources)


class EnergySignalGenerator(SignalGenerator):
    """
    Signal from energy state and transitions.

    Physics principle: Energy must go somewhere.
    - High KE = motion (trending)
    - High PE = compression (coiled spring)
    - Energy release rate = imminent move
    """

    def __init__(self):
        super().__init__("energy", SignalType.ENERGY)

    def generate(self, measurements: Dict[str, float]) -> Signal:
        components = []
        sources = []

        # Kinetic energy percentile
        ke = measurements.get('kinetic_energy_pct', 0.5)

        # Potential energy (compression) percentile
        # (measurement currently unused in signal computation)

        # Energy efficiency ratio
        efficiency = measurements.get('energy_efficiency_pct', 0.5)

        # Energy release rate
        release_rate = measurements.get('energy_release_rate_pct', 0.5)

        # Phase compression (high PE, low KE, low entropy)
        phase_comp = measurements.get('phase_compression_pct', 0.5)

        # Release potential (compression × low entropy)
        release_potential = measurements.get('release_potential_pct', 0.5)

        # Signal logic:
        # High KE with positive release rate = momentum continuing
        # High PE with rising release rate = breakout imminent
        # High phase compression = coiled spring

        ke_signal = (ke - 0.5) * 2
        release_signal = (release_rate - 0.5) * 2
        compression_signal = (phase_comp - 0.5) * 1.5  # Compression matters

        components.extend([ke_signal, release_signal, compression_signal])
        sources.extend(['kinetic_energy', 'energy_release_rate', 'phase_compression'])

        if not components:
            return Signal(self.name, self.signal_type, 0, 0, [])

        # Combine energy signals
        # High energy state (KE or compression) with positive release = bullish
        signal_value = np.tanh(np.mean(components))

        # Confidence based on energy magnitude
        confidence = min(1.0, abs(ke - 0.5) + abs(release_rate - 0.5))

        return Signal(self.name, self.signal_type, signal_value, confidence, sources)


class FlowRegimeSignalGenerator(SignalGenerator):
    """
    Signal from fluid dynamics regime.

    KEY PHYSICS INSIGHT: Reynolds number determines flow regime.
    - High Re = turbulent (chaotic, relationships may invert)
    - Low Re = laminar (smooth, predictable)

    Damping coefficient indicates friction/mean-reversion tendency.
    """

    def __init__(self):
        super().__init__("flow_regime", SignalType.FLOW)

    def generate(self, measurements: Dict[str, float]) -> Signal:
        # Reynolds number percentile
        reynolds = measurements.get('reynolds_pct', 0.5)

        # Damping coefficient
        damping = measurements.get('damping_pct', 0.5)

        # Viscosity (resistance to flow)
        viscosity = measurements.get('viscosity_pct', 0.5)

        # Liquidity
        liquidity = measurements.get('liquidity_pct', 0.5)

        # Reynolds-momentum correlation (should be inverse in stable regime)
        re_mom_corr = measurements.get('reynolds_momentum_corr_pct', 0.5)

        # Re-damping ratio
        re_damping = measurements.get('re_damping_ratio_pct', 0.5)

        # Regime classification:
        # High Reynolds (>0.8) = turbulent → be cautious
        # Low damping (<0.3) = trending → follow momentum
        # High damping (>0.7) = mean-reverting → fade moves

        sources = ['reynolds', 'damping', 'viscosity', 'liquidity']

        # Regime quality signal (positive = good for trading)
        if reynolds < 0.3:
            # Laminar regime - stable, predictable
            regime_quality = 0.3
        elif reynolds > 0.8:
            # Turbulent - chaotic, avoid
            regime_quality = -0.5
        else:
            # Transitional
            regime_quality = 0.0

        # Flow direction from Re-momentum relationship
        # If relationship is breaking (re_mom_corr > 0.6), regime is shifting
        if re_mom_corr > 0.7:
            regime_shift = 0.5  # Warning signal
        else:
            regime_shift = 0.0

        signal_value = regime_quality - regime_shift

        # Confidence based on how clear the regime is
        confidence = 1.0 - abs(reynolds - 0.5)

        return Signal(self.name, self.signal_type, signal_value, confidence, sources)


class ThermodynamicsSignalGenerator(SignalGenerator):
    """
    Signal from entropy and phase state.

    Physics principle:
    - Low entropy = ordered = predictable
    - High entropy = disordered = random
    - Rising entropy rate = system becoming chaotic
    - Falling entropy rate = system becoming ordered
    """

    def __init__(self):
        super().__init__("thermodynamics", SignalType.THERMODYNAMICS)

    def generate(self, measurements: Dict[str, float]) -> Signal:
        # Entropy percentile
        entropy = measurements.get('entropy_pct', 0.5)

        # Entropy rate (change in entropy)
        entropy_rate = measurements.get('entropy_rate_pct', 0.5)

        # Phase compression (ordered energy state)
        phase_comp = measurements.get('phase_compression_pct', 0.5)

        # Entropy-energy phase interaction
        entropy_energy = measurements.get('entropy_energy_phase_pct', 0.5)

        sources = ['entropy', 'entropy_rate', 'phase_compression']

        # Signal logic:
        # Low entropy + falling entropy rate = system ordering → trend forming
        # High entropy + rising entropy rate = increasing chaos → avoid
        # High phase compression = ordered energy state → potential breakout

        order_signal = (0.5 - entropy) * 2  # Low entropy = positive
        trend_forming = (0.5 - entropy_rate) * 1.5  # Falling rate = positive
        compression_signal = (phase_comp - 0.5) * 2

        signal_value = np.tanh((order_signal + trend_forming + compression_signal) / 3)

        # Confidence: low entropy = more predictable = higher confidence
        confidence = 1.0 - entropy

        return Signal(self.name, self.signal_type, signal_value, confidence, sources)


class MicrostructureSignalGenerator(SignalGenerator):
    """
    Signal from market microstructure.

    About WHEN to trade, not direction.
    - Wide spread = bad execution
    - Low volume = low liquidity
    - Volume surge = potential news/action
    """

    def __init__(self):
        super().__init__("microstructure", SignalType.MICROSTRUCTURE)

    def generate(self, measurements: Dict[str, float]) -> Signal:
        # Spread percentile (high = wide spread = bad)
        spread = measurements.get('spread_pct_pct', 0.5)

        # Volume surge (high = unusual activity)
        volume_surge = measurements.get('volume_surge_pct', 0.5)

        # Volume trend (rising = increasing interest)
        volume_trend = measurements.get('volume_trend_pct', 0.5)

        # Liquidity from flow measures
        liquidity = measurements.get('liquidity_pct', 0.5)

        sources = ['spread_pct', 'volume_surge', 'volume_trend', 'liquidity']

        # Execution quality:
        # Good: low spread, high liquidity, moderate volume
        # Bad: high spread, low liquidity, extreme volume (could be news)

        spread_quality = (0.5 - spread)  # Low spread = positive
        liquidity_quality = (liquidity - 0.5)

        # Volume surge is mixed - could be opportunity or danger
        # Moderate surge is good, extreme surge is warning
        if volume_surge > 0.9:
            volume_signal = -0.3  # Extreme = caution
        elif volume_surge > 0.6:
            volume_signal = 0.2  # Good activity
        else:
            volume_signal = 0.0  # Low activity

        # This is about execution quality, not direction
        signal_value = np.tanh(spread_quality + liquidity_quality + volume_signal)

        # Confidence based on liquidity
        confidence = min(1.0, liquidity)

        return Signal(self.name, self.signal_type, signal_value, confidence, sources)


# =============================================================================
# COMPOSITE STACKER
# =============================================================================

class CompositeStacker:
    """
    Stacks multiple physics-based signal generators and learns weights per asset class.

    CRITICAL: No hardcoded weights. Agent learns what works from physics state.
    """

    def __init__(self, asset_class: str):
        self.asset_class = asset_class

        # Physics-based signal generators (NO traditional indicators)
        self.generators: Dict[str, SignalGenerator] = {
            'kinematics': KinematicsSignalGenerator(),
            'energy': EnergySignalGenerator(),
            'flow_regime': FlowRegimeSignalGenerator(),
            'thermodynamics': ThermodynamicsSignalGenerator(),
            'microstructure': MicrostructureSignalGenerator(),
        }

        # Learned weights (start uniform)
        self.weights: Dict[str, float] = {name: 1.0 for name in self.generators}

        # Performance tracking per generator
        self.generator_performance: Dict[str, List[Tuple[float, float]]] = defaultdict(list)

    def generate_composite(self, measurements: Dict[str, float]) -> Dict[str, Signal]:
        """Generate all physics-based signals."""
        signals = {}
        for name, generator in self.generators.items():
            signals[name] = generator.generate(measurements)
        return signals

    def get_composite_signal(self, signals: Dict[str, Signal]) -> Tuple[float, float]:
        """
        Combine signals into single composite.

        Returns (signal_value, confidence).
        """
        total_signal = 0.0
        total_weight = 0.0

        for name, signal in signals.items():
            weight = self.weights[name] * signal.confidence
            total_signal += signal.value * weight
            total_weight += weight

        if total_weight > 0:
            composite_value = total_signal / total_weight
            composite_confidence = min(1.0, total_weight / len(signals))
        else:
            composite_value = 0.0
            composite_confidence = 0.0

        return composite_value, composite_confidence

    def record_outcome(self, signals: Dict[str, Signal], actual_pnl: float):
        """Record outcome and update reliability estimates."""
        for name, signal in signals.items():
            self.generators[name].record_outcome(signal.value, actual_pnl)
            self.generator_performance[name].append((signal.value, actual_pnl))

    def update_weights(self, lookback: int = 100):
        """
        Update weights based on recent performance.

        Generators that predict well get higher weight.
        """
        for name, generator in self.generators.items():
            reliability = generator.get_reliability(lookback)
            # Weight is reliability squared (penalize unreliable more)
            self.weights[name] = reliability ** 2

        # Normalize
        total = sum(self.weights.values())
        if total > 0:
            for name in self.weights:
                self.weights[name] /= total

    def get_feature_vector(self, signals: Dict[str, Signal]) -> np.ndarray:
        """
        Convert signals to feature vector for RL agent.

        Agent receives:
        - Each signal's value
        - Each signal's confidence
        - Each generator's reliability
        - The composite signal
        """
        features = []

        for name in sorted(self.generators.keys()):
            signal = signals[name]
            reliability = self.generators[name].get_reliability()

            features.extend([
                signal.value,
                signal.confidence,
                reliability,
            ])

        # Add composite
        composite_value, composite_confidence = self.get_composite_signal(signals)
        features.extend([composite_value, composite_confidence])

        return np.array(features)

    def get_feature_names(self) -> List[str]:
        """Get names of features in the vector."""
        names = []
        for name in sorted(self.generators.keys()):
            names.extend([
                f'{name}_signal',
                f'{name}_confidence',
                f'{name}_reliability',
            ])
        names.extend(['composite_signal', 'composite_confidence'])
        return names


# =============================================================================
# CLASS-SPECIFIC DISCOVERY ENGINE
# =============================================================================

class ClassDiscoveryEngine:
    """
    Discovers what physics patterns work per asset class.

    Tracks:
    - Which physics measurements correlate with good outcomes per class
    - Which signal generators are reliable per class
    - How relationships change during high-energy regimes
    """

    def __init__(self):
        self.class_stackers: Dict[str, CompositeStacker] = {}
        self.class_discoveries: Dict[str, Dict] = defaultdict(dict)

    def get_stacker(self, asset_class: str) -> CompositeStacker:
        """Get or create stacker for asset class."""
        if asset_class not in self.class_stackers:
            self.class_stackers[asset_class] = CompositeStacker(asset_class)
        return self.class_stackers[asset_class]

    def record_discovery(self, asset_class: str, discovery_type: str,
                         discovery: Dict):
        """Record a discovery about an asset class."""
        if discovery_type not in self.class_discoveries[asset_class]:
            self.class_discoveries[asset_class][discovery_type] = []
        self.class_discoveries[asset_class][discovery_type].append(discovery)

    def get_class_profile(self, asset_class: str) -> Dict:
        """
        Get learned profile for asset class.

        Returns what we've discovered about physics patterns for this class.
        """
        if asset_class not in self.class_stackers:
            return {'status': 'not_enough_data'}

        stacker = self.class_stackers[asset_class]

        profile = {
            'asset_class': asset_class,
            'signal_weights': stacker.weights.copy(),
            'generator_reliability': {
                name: gen.get_reliability()
                for name, gen in stacker.generators.items()
            },
            'discoveries': self.class_discoveries.get(asset_class, {}),
        }

        # Identify dominant physics signal type for this class
        best_generator = max(stacker.weights.items(), key=lambda x: x[1])
        profile['dominant_signal'] = best_generator[0]
        profile['dominant_weight'] = best_generator[1]

        return profile

    def compare_classes(self) -> pd.DataFrame:
        """
        Compare what physics patterns work across classes.

        This is THE key output - shows that physics regimes differ by class.
        """
        rows = []

        for asset_class, stacker in self.class_stackers.items():
            row = {'asset_class': asset_class}

            for name, weight in stacker.weights.items():
                row[f'{name}_weight'] = weight

            for name, generator in stacker.generators.items():
                row[f'{name}_reliability'] = generator.get_reliability()

            rows.append(row)

        return pd.DataFrame(rows)


# =============================================================================
# INVERSE RELATIONSHIP TRACKER
# =============================================================================

class InverseRelationshipTracker:
    """
    Tracks when physics relationships INVERT during high-energy regimes.

    This is the physics insight: turbulent flow changes everything.
    Relationships that work in laminar regime may invert in turbulent.
    """

    def __init__(self):
        # Track measurement pairs and their correlation over time
        self.correlation_history: Dict[Tuple[str, str], List[Tuple[float, float]]] = defaultdict(list)
        # (energy_level, correlation)

    def record_correlation(self, meas1: str, meas2: str,
                           correlation: float, energy_level: float):
        """Record correlation between two physics measurements at given energy level."""
        key = (min(meas1, meas2), max(meas1, meas2))
        self.correlation_history[key].append((energy_level, correlation))

    def find_inversions(self, min_samples: int = 50) -> List[Dict]:
        """
        Find physics measurement pairs whose relationship inverts with energy.

        Returns list of discoveries.
        """
        inversions = []

        for (meas1, meas2), history in self.correlation_history.items():
            if len(history) < min_samples:
                continue

            # Split by energy level
            energy_levels = [h[0] for h in history]
            correlations = [h[1] for h in history]

            median_energy = np.median(energy_levels)

            low_energy_corr = [c for e, c in history if e < median_energy]
            high_energy_corr = [c for e, c in history if e >= median_energy]

            if len(low_energy_corr) < 10 or len(high_energy_corr) < 10:
                continue

            mean_low = np.mean(low_energy_corr)
            mean_high = np.mean(high_energy_corr)

            # Check for inversion (sign change or significant shift)
            if np.sign(mean_low) != np.sign(mean_high):
                inversions.append({
                    'measurement_1': meas1,
                    'measurement_2': meas2,
                    'low_vol_correlation': mean_low,
                    'high_vol_correlation': mean_high,
                    'inversion_magnitude': abs(mean_low - mean_high),
                    'type': 'sign_inversion',
                })
            elif abs(mean_low - mean_high) > 0.5:
                inversions.append({
                    'measurement_1': meas1,
                    'measurement_2': meas2,
                    'low_vol_correlation': mean_low,
                    'high_vol_correlation': mean_high,
                    'inversion_magnitude': abs(mean_low - mean_high),
                    'type': 'magnitude_shift',
                })

        return sorted(inversions, key=lambda x: x['inversion_magnitude'], reverse=True)


# =============================================================================
# MAIN EXPLORATION INTERFACE
# =============================================================================

class ExplorationEngine:
    """
    Main interface for physics-based exploration.

    Combines:
    - Physics measurements (NO traditional indicators)
    - CompositeStacker (combine physics signals)
    - ClassDiscoveryEngine (learn physics patterns per class)
    - InverseRelationshipTracker (find inversions)
    """

    def __init__(self):
        # Import MeasurementEngine - handle both package and direct loading
        try:
            from .measurements import MeasurementEngine
        except ImportError:
            import importlib.util
            from pathlib import Path
            spec = importlib.util.spec_from_file_location(
                'measurements', Path(__file__).parent / 'measurements.py')
            _meas = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(_meas)
            MeasurementEngine = _meas.MeasurementEngine

        self.measurement_engine = MeasurementEngine()
        self.discovery_engine = ClassDiscoveryEngine()
        self.inverse_tracker = InverseRelationshipTracker()

    def process_bar(self, asset_class: str, measurements: Dict[str, float],
                    volatility_level: float) -> Dict:
        """
        Process a single bar using physics measurements.

        Returns signals and features for RL.
        """
        stacker = self.discovery_engine.get_stacker(asset_class)

        # Generate physics-based signals
        signals = stacker.generate_composite(measurements)

        # Get composite
        composite_value, composite_confidence = stacker.get_composite_signal(signals)

        # Get feature vector for RL
        signal_features = stacker.get_feature_vector(signals)

        return {
            'signals': signals,
            'composite_value': composite_value,
            'composite_confidence': composite_confidence,
            'signal_features': signal_features,
            'asset_class': asset_class,
        }

    def record_trade_outcome(self, asset_class: str, signals: Dict[str, Signal],
                             pnl: float, measurements: Dict[str, float],
                             volatility_level: float):
        """Record trade outcome for physics pattern learning."""
        stacker = self.discovery_engine.get_stacker(asset_class)

        # Record outcome for each signal generator
        stacker.record_outcome(signals, pnl)

        # Update weights periodically
        if len(stacker.generator_performance.get('kinematics', [])) % 50 == 0:
            stacker.update_weights()

        # Track inverse relationships between physics measures
        # These are the key pairs that may invert during high-energy regimes
        pairs = [
            ('reynolds_pct', 'momentum_pct'),       # Re vs momentum
            ('kinetic_energy_pct', 'entropy_pct'),  # KE vs entropy
            ('damping_pct', 'velocity_pct'),        # Damping vs velocity
            ('phase_compression_pct', 'jerk_pct'),  # Compression vs jerk
        ]

        for m1, m2 in pairs:
            if m1 in measurements and m2 in measurements:
                corr = measurements[m1] * measurements[m2]  # Simple product as proxy
                self.inverse_tracker.record_correlation(m1, m2, corr, volatility_level)

    def get_discoveries(self) -> Dict:
        """Get all physics pattern discoveries so far."""
        return {
            'class_profiles': {
                cls: self.discovery_engine.get_class_profile(cls)
                for cls in self.discovery_engine.class_stackers
            },
            'class_comparison': self.discovery_engine.compare_classes().to_dict(),
            'inverse_relationships': self.inverse_tracker.find_inversions(),
        }


# Export
__all__ = [
    'SignalType',
    'Signal',
    'SignalGenerator',
    'KinematicsSignalGenerator',
    'EnergySignalGenerator',
    'FlowRegimeSignalGenerator',
    'ThermodynamicsSignalGenerator',
    'MicrostructureSignalGenerator',
    'CompositeStacker',
    'ClassDiscoveryEngine',
    'InverseRelationshipTracker',
    'ExplorationEngine',
]
