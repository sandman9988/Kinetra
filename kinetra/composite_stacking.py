"""
Composite Stacking Framework for Multi-Asset Exploration
=========================================================

PHILOSOPHY: Let the agent discover what works.

Instead of hardcoding "RSI < 30 = buy signal", we:
1. Compute ALL measurements
2. Let agent learn which combinations matter
3. Track what works per asset class
4. Discover inverse relationships during volatility

NO ASSUMPTIONS about what "trending" or "mean-reverting" means.
Let the Hurst exponent and correlations tell us.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Callable
from enum import Enum
import numpy as np
import pandas as pd
from collections import defaultdict


# =============================================================================
# SIGNAL TYPES (Discovered, Not Assumed)
# =============================================================================

class SignalType(Enum):
    """
    Signal types - but we DON'T assume which measurement produces which.
    The agent discovers this.
    """
    MOMENTUM = "momentum"
    MEAN_REVERSION = "mean_reversion"
    VOLATILITY_BREAKOUT = "volatility_breakout"
    FLOW_REGIME = "flow_regime"
    MICROSTRUCTURE = "microstructure"
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
# BASE SIGNAL GENERATORS (Exploration Mode)
# =============================================================================

class SignalGenerator:
    """
    Base class for signal generators.

    In exploration mode, we generate signals from measurements
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
        # Correlation between signal and outcome
        agreement = signal_value * np.sign(actual_pnl)
        self.performance_history.append(agreement)

    def get_reliability(self, window: int = 100) -> float:
        """Get recent reliability of this signal."""
        if len(self.performance_history) < window:
            return 0.5  # Neutral
        recent = self.performance_history[-window:]
        return (np.mean(recent) + 1) / 2  # Scale to 0-1


class MomentumSignalGenerator(SignalGenerator):
    """
    Momentum signal - but let agent discover which measurements matter.

    We provide multiple momentum-related signals, agent weights them.
    """

    def __init__(self):
        super().__init__("momentum", SignalType.MOMENTUM)

    def generate(self, measurements: Dict[str, float]) -> Signal:
        # Multiple momentum indicators - let agent learn weights
        components = []

        # ROC signals
        for period in [5, 10, 20, 50]:
            key = f'roc_{period}_z'
            if key in measurements:
                components.append(('roc', measurements[key]))

        # RSI signal (centered at 50)
        for period in [7, 14, 21]:
            key = f'rsi_{period}_z'
            if key in measurements:
                components.append(('rsi', measurements[key]))

        # MACD histogram
        if 'macd_histogram_z' in measurements:
            components.append(('macd', measurements['macd_histogram_z']))

        # ADX (trend strength, not direction)
        if 'adx_z' in measurements and 'plus_di_z' in measurements and 'minus_di_z' in measurements:
            adx = measurements['adx_z']
            direction = np.sign(measurements['plus_di_z'] - measurements['minus_di_z'])
            components.append(('adx', adx * direction))

        # Aroon oscillator
        if 'aroon_oscillator_z' in measurements:
            components.append(('aroon', measurements['aroon_oscillator_z']))

        if not components:
            return Signal(self.name, self.signal_type, 0, 0, [])

        # Simple average for now - agent learns to weight
        values = [c[1] for c in components]
        sources = [c[0] for c in components]

        signal_value = np.tanh(np.mean(values))  # Bound to [-1, 1]
        confidence = min(1.0, np.std(values))  # Low std = high agreement

        return Signal(self.name, self.signal_type, signal_value, confidence, sources)


class MeanReversionSignalGenerator(SignalGenerator):
    """
    Mean reversion signal - BUT only if Hurst < 0.5.

    Key insight: MR signals are WRONG in trending markets.
    """

    def __init__(self):
        super().__init__("mean_reversion", SignalType.MEAN_REVERSION)

    def generate(self, measurements: Dict[str, float]) -> Signal:
        # Check Hurst first - is MR even valid?
        hurst = measurements.get('hurst_z', 0)
        is_mr_regime = measurements.get('is_mean_reverting', 0)

        if is_mr_regime < 0.5 and hurst > 0:
            # NOT a MR regime - signal should be weak
            regime_weight = 0.3
        else:
            regime_weight = 1.0

        components = []

        # Bollinger %B (0 = lower band, 1 = upper band)
        if 'bollinger_pct_b_z' in measurements:
            # Invert: high %B = overbought = short signal
            components.append(('bb', -measurements['bollinger_pct_b_z']))

        # Z-score
        for period in [20, 50]:
            key = f'zscore_{period}_z'
            if key in measurements:
                # Invert: high z-score = overbought = short signal
                components.append(('zscore', -measurements[key]))

        # Distance from VWAP
        if 'dist_from_vwap_z' in measurements:
            components.append(('vwap', -measurements['dist_from_vwap_z']))

        # RSI extremes (but inverted for MR)
        if 'rsi_14_z' in measurements:
            rsi = measurements['rsi_14_z']
            # Strong RSI = fade it in MR regime
            components.append(('rsi_mr', -rsi))

        if not components:
            return Signal(self.name, self.signal_type, 0, 0, [])

        values = [c[1] for c in components]
        sources = [c[0] for c in components]

        signal_value = np.tanh(np.mean(values)) * regime_weight
        confidence = (1 - abs(hurst)) * regime_weight  # More confident when Hurst near 0

        return Signal(self.name, self.signal_type, signal_value, confidence, sources)


class VolatilityBreakoutSignalGenerator(SignalGenerator):
    """
    Volatility breakout signal.

    High vol + direction = breakout.
    But relationships may INVERT during extreme volatility.
    """

    def __init__(self):
        super().__init__("vol_breakout", SignalType.VOLATILITY_BREAKOUT)

    def generate(self, measurements: Dict[str, float]) -> Signal:
        components = []

        # Vol-of-vol (regime uncertainty)
        vov = measurements.get('vol_of_vol_z', 0)

        # Kinetic energy (price velocity squared)
        ke = measurements.get('kinetic_energy_z', 0)

        # Energy release rate
        err = measurements.get('energy_release_rate_z', 0)

        # Direction from momentum
        roc = measurements.get('roc_10_z', 0)

        # Breakout signal: high energy + clear direction
        energy_level = np.sqrt(ke**2 + err**2) if ke > 0 or err > 0 else 0
        direction = np.sign(roc)

        # BUT: If vol-of-vol is extreme, relationships may invert
        # This is the physics insight
        if abs(vov) > 2:
            # Extreme uncertainty - reduce confidence, maybe invert
            inversion_factor = -0.5  # Partial inversion
        else:
            inversion_factor = 1.0

        signal_value = np.tanh(energy_level * direction * inversion_factor)
        confidence = min(1.0, energy_level / 2)

        sources = ['kinetic_energy', 'energy_release_rate', 'roc']
        return Signal(self.name, self.signal_type, signal_value, confidence, sources)


class FlowRegimeSignalGenerator(SignalGenerator):
    """
    Flow regime signal based on Reynolds number.

    KEY INSIGHT: Reynolds should be INVERSE to ROC during instability.
    When this breaks, it's a regime shift signal.
    """

    def __init__(self):
        super().__init__("flow_regime", SignalType.FLOW_REGIME)

    def generate(self, measurements: Dict[str, float]) -> Signal:
        # Reynolds number
        reynolds = measurements.get('reynolds_z', 0)

        # Reynolds-ROC inverse relationship
        inverse_corr = measurements.get('reynolds_roc_inverse', 0)

        # Flow regime (0=laminar, 1=transitional, 2=turbulent)
        flow = measurements.get('flow_regime', 1)

        # Entropy rate
        entropy = measurements.get('entropy_rate_z', 0)

        # The signal is about REGIME, not direction
        # Laminar + low entropy = stable, trend-following works
        # Turbulent + high entropy = unstable, be cautious

        if flow == 0:  # Laminar
            regime_quality = 1.0  # Good for trading
        elif flow == 2:  # Turbulent
            regime_quality = -0.5  # Be cautious
        else:  # Transitional
            regime_quality = 0.0  # Neutral

        # The inverse correlation breaking is THE key signal
        # Normal: Reynolds and ROC should be negatively correlated
        # If they become positively correlated, regime is shifting
        if inverse_corr > 0.3:  # Should be negative, but it's positive
            regime_shift = 1.0  # Warning: relationships inverting
        else:
            regime_shift = 0.0

        # This signal is about "should we trade" not "which direction"
        signal_value = regime_quality - regime_shift

        # Confidence based on how extreme the readings are
        confidence = 1.0 - abs(regime_shift)

        sources = ['reynolds', 'reynolds_roc_inverse', 'flow_regime', 'entropy_rate']
        return Signal(self.name, self.signal_type, signal_value, confidence, sources)


class MicrostructureSignalGenerator(SignalGenerator):
    """
    Microstructure signal - execution quality.

    High spread + low volume = bad execution.
    But this is about WHEN to trade, not direction.
    """

    def __init__(self):
        super().__init__("microstructure", SignalType.MICROSTRUCTURE)

    def generate(self, measurements: Dict[str, float]) -> Signal:
        # Spread ratio (high = bad)
        spread = measurements.get('spread_ratio_z', 0)

        # Volume ratio (low = bad)
        volume = measurements.get('volume_ratio_z', 0)

        # Liquidity score (high = good)
        liquidity = measurements.get('liquidity_score_z', 0)

        # Tick intensity (sudden spike = news)
        tick = measurements.get('tick_intensity_z', 0)

        # Composite execution quality
        # High liquidity, low spread, normal tick = good
        execution_quality = liquidity - spread - abs(tick - 1)

        # This is NOT a directional signal
        # It's "is now a good time to execute"
        signal_value = np.tanh(execution_quality)
        confidence = min(1.0, abs(liquidity))

        sources = ['spread_ratio', 'volume_ratio', 'liquidity_score', 'tick_intensity']
        return Signal(self.name, self.signal_type, signal_value, confidence, sources)


# =============================================================================
# COMPOSITE STACKER
# =============================================================================

class CompositeStacker:
    """
    Stacks multiple signal generators and learns weights per asset class.

    CRITICAL: No hardcoded weights. Agent learns what works.
    """

    def __init__(self, asset_class: str):
        self.asset_class = asset_class

        # Signal generators
        self.generators: Dict[str, SignalGenerator] = {
            'momentum': MomentumSignalGenerator(),
            'mean_reversion': MeanReversionSignalGenerator(),
            'vol_breakout': VolatilityBreakoutSignalGenerator(),
            'flow_regime': FlowRegimeSignalGenerator(),
            'microstructure': MicrostructureSignalGenerator(),
        }

        # Learned weights (start uniform)
        self.weights: Dict[str, float] = {name: 1.0 for name in self.generators}

        # Performance tracking per generator
        self.generator_performance: Dict[str, List[Tuple[float, float]]] = defaultdict(list)

    def generate_composite(self, measurements: Dict[str, float]) -> Dict[str, Signal]:
        """Generate all signals."""
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
    Discovers what works per asset class.

    Tracks:
    - Which measurements correlate with good outcomes per class
    - Which signal generators are reliable per class
    - How relationships change during volatility
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

        Returns what we've discovered about this class.
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

        # Identify dominant signal type for this class
        best_generator = max(stacker.weights.items(), key=lambda x: x[1])
        profile['dominant_signal'] = best_generator[0]
        profile['dominant_weight'] = best_generator[1]

        return profile

    def compare_classes(self) -> pd.DataFrame:
        """
        Compare what works across classes.

        This is THE key output - shows that "trending" means different things.
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
    Tracks when relationships INVERT during volatility.

    This is the physics insight: turbulent flow changes everything.
    """

    def __init__(self):
        # Track measurement pairs and their correlation over time
        self.correlation_history: Dict[Tuple[str, str], List[Tuple[float, float]]] = defaultdict(list)
        # vol_level, correlation

    def record_correlation(self, meas1: str, meas2: str,
                           correlation: float, volatility_level: float):
        """Record correlation between two measurements at given vol level."""
        key = (min(meas1, meas2), max(meas1, meas2))
        self.correlation_history[key].append((volatility_level, correlation))

    def find_inversions(self, min_samples: int = 50) -> List[Dict]:
        """
        Find measurement pairs whose relationship inverts with volatility.

        Returns list of discoveries.
        """
        inversions = []

        for (meas1, meas2), history in self.correlation_history.items():
            if len(history) < min_samples:
                continue

            # Split by volatility
            vol_levels = [h[0] for h in history]
            correlations = [h[1] for h in history]

            median_vol = np.median(vol_levels)

            low_vol_corr = [c for v, c in history if v < median_vol]
            high_vol_corr = [c for v, c in history if v >= median_vol]

            if len(low_vol_corr) < 10 or len(high_vol_corr) < 10:
                continue

            mean_low = np.mean(low_vol_corr)
            mean_high = np.mean(high_vol_corr)

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
    Main interface for exploring measurements and composites.

    Combines:
    - MeasurementEngine (compute all measurements)
    - CompositeStacker (combine signals)
    - ClassDiscoveryEngine (learn per class)
    - InverseRelationshipTracker (find inversions)
    """

    def __init__(self):
        from .measurements import MeasurementEngine
        self.measurement_engine = MeasurementEngine()
        self.discovery_engine = ClassDiscoveryEngine()
        self.inverse_tracker = InverseRelationshipTracker()

    def process_bar(self, asset_class: str, measurements: Dict[str, float],
                    volatility_level: float) -> Dict:
        """
        Process a single bar for an instrument.

        Returns signals and features for RL.
        """
        stacker = self.discovery_engine.get_stacker(asset_class)

        # Generate signals
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
        """Record trade outcome for learning."""
        stacker = self.discovery_engine.get_stacker(asset_class)

        # Record outcome for each signal generator
        stacker.record_outcome(signals, pnl)

        # Update weights periodically
        if len(stacker.generator_performance['momentum']) % 50 == 0:
            stacker.update_weights()

        # Track inverse relationships
        # Compare a few key measurement pairs
        pairs = [
            ('reynolds_z', 'roc_10_z'),
            ('vol_yang_zhang_z', 'rsi_14_z'),
            ('hurst_z', 'adx_z'),
            ('entropy_rate_z', 'kinetic_energy_z'),
        ]

        for m1, m2 in pairs:
            if m1 in measurements and m2 in measurements:
                corr = measurements[m1] * measurements[m2]  # Simple product as proxy
                self.inverse_tracker.record_correlation(m1, m2, corr, volatility_level)

    def get_discoveries(self) -> Dict:
        """Get all discoveries so far."""
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
    'MomentumSignalGenerator',
    'MeanReversionSignalGenerator',
    'VolatilityBreakoutSignalGenerator',
    'FlowRegimeSignalGenerator',
    'MicrostructureSignalGenerator',
    'CompositeStacker',
    'ClassDiscoveryEngine',
    'InverseRelationshipTracker',
    'ExplorationEngine',
]
