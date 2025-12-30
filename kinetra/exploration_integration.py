"""
Exploration Integration Module
==============================

Wires up the PHYSICS-BASED measurement + composite stacking
into the RL exploration framework.

NO TRADITIONAL INDICATORS. Pure physics measurements only.

This is the bridge between:
- Raw OHLCV data
- MeasurementEngine (physics measurements)
- CompositeStacker (signal generation)
- RL Agent (learning what matters per asset class)

Physics Measurements:
- Kinematics: velocity, acceleration, jerk, snap, crackle, pop
- Energy: kinetic, potential (compression + displacement), efficiency
- Flow: Reynolds, damping, viscosity, liquidity
- Thermodynamics: entropy, entropy rate, phase compression
- Field: gradient, divergence, buying pressure, body ratio
- Microstructure: spread percentile, volume surge, volume trend
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
from dataclasses import dataclass, field

# Import our measurement modules - handle both package and direct loading
import importlib.util
from pathlib import Path

def _load_sibling_module(name: str):
    """Load a sibling module from the same directory."""
    spec = importlib.util.spec_from_file_location(
        name, Path(__file__).parent / f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

try:
    from .measurements import (
        MeasurementEngine,
        KinematicsMeasures,
        EnergyMeasures,
        FlowMeasures,
        ThermodynamicsMeasures,
        FieldMeasures,
        MicrostructureMeasures,
        PercentileNormalizer,
        PhysicsRegimeDetector,
        CorrelationExplorer,
    )
except ImportError:
    _meas = _load_sibling_module('measurements')
    MeasurementEngine = _meas.MeasurementEngine
    KinematicsMeasures = _meas.KinematicsMeasures
    EnergyMeasures = _meas.EnergyMeasures
    FlowMeasures = _meas.FlowMeasures
    ThermodynamicsMeasures = _meas.ThermodynamicsMeasures
    FieldMeasures = _meas.FieldMeasures
    MicrostructureMeasures = _meas.MicrostructureMeasures
    PercentileNormalizer = _meas.PercentileNormalizer
    PhysicsRegimeDetector = _meas.PhysicsRegimeDetector
    CorrelationExplorer = _meas.CorrelationExplorer

try:
    from .composite_stacking import (
        ExplorationEngine,
        CompositeStacker,
        ClassDiscoveryEngine,
        InverseRelationshipTracker,
    )
except ImportError:
    _comp = _load_sibling_module('composite_stacking')
    ExplorationEngine = _comp.ExplorationEngine
    CompositeStacker = _comp.CompositeStacker
    ClassDiscoveryEngine = _comp.ClassDiscoveryEngine
    InverseRelationshipTracker = _comp.InverseRelationshipTracker

try:
    from .multi_agent_design import (
        AssetClass,
        get_asset_class,
        get_instrument_profile,
        INSTRUMENT_PROFILES,
    )
except ImportError:
    _multi = _load_sibling_module('multi_agent_design')
    AssetClass = _multi.AssetClass
    get_asset_class = _multi.get_asset_class
    get_instrument_profile = _multi.get_instrument_profile
    INSTRUMENT_PROFILES = _multi.INSTRUMENT_PROFILES


@dataclass
class InstrumentMeasurements:
    """
    Pre-computed measurements for an instrument.

    Computed once per instrument load, used for each bar during exploration.
    """
    instrument_key: str
    asset_class: str

    # Raw measurements (full arrays)
    raw_measurements: Dict[str, np.ndarray] = field(default_factory=dict)

    # Normalized measurements (z-scores)
    normalized_measurements: Dict[str, np.ndarray] = field(default_factory=dict)

    # Measurement names in order
    measurement_names: List[str] = field(default_factory=list)

    # Correlation matrix (computed after all measurements)
    correlation_matrix: Optional[pd.DataFrame] = None

    # Discovered inverse relationships
    inverse_relationships: List[Dict] = field(default_factory=list)


class ExplorationDataLoader:
    """
    Enhanced data loader that computes all measurements.

    Replaces simple OHLCV loading with comprehensive measurement computation.
    """

    def __init__(self, data_dir: str = "data/master"):
        self.data_dir = Path(data_dir)
        self.measurement_engine = MeasurementEngine()
        self.instruments: Dict[str, InstrumentMeasurements] = {}

    def load_instrument(self, csv_file: Path) -> Optional[InstrumentMeasurements]:
        """
        Load instrument and compute ALL measurements.
        """
        try:
            # Parse instrument info from filename
            parts = csv_file.stem.split('_')
            if len(parts) < 2:
                return None

            instrument = parts[0]
            timeframe = parts[1]
            instrument_key = f"{instrument}_{timeframe}"

            # Get asset class
            asset_class = get_asset_class(instrument)
            if isinstance(asset_class, AssetClass):
                asset_class_str = asset_class.value
            else:
                asset_class_str = str(asset_class)

            # Load raw data (MT5 tab-separated format)
            df = pd.read_csv(csv_file, sep='\t')
            df.columns = [c.lower().replace('<', '').replace('>', '') for c in df.columns]

            # Parse datetime
            if 'date' in df.columns and 'time' in df.columns:
                date_str = df['date'].astype(str).str.replace('.', '-', regex=False)
                df['datetime'] = pd.to_datetime(date_str + ' ' + df['time'].astype(str))
            else:
                return None

            df = df.sort_values('datetime').reset_index(drop=True)

            # Extract arrays
            open_ = df['open'].values.astype(float)
            high = df['high'].values.astype(float)
            low = df['low'].values.astype(float)
            close = df['close'].values.astype(float)

            # Volume (tickvol in MT5)
            volume = df.get('tickvol', df.get('vol', pd.Series(np.ones(len(df))))).values.astype(float)

            # Spread
            spread = df.get('spread', pd.Series(np.ones(len(df)))).values.astype(float)

            # Timestamps
            timestamps = pd.DatetimeIndex(df['datetime'])

            # Compute ALL physics measurements
            raw_measurements = self.measurement_engine.compute_all(
                open_=open_,
                high=high,
                low=low,
                close=close,
                volume=volume,
                spread=spread,
                timestamps=timestamps,
            )

            # Normalize to rolling percentiles (instrument-agnostic)
            normalized_measurements = self.measurement_engine.normalize_to_percentiles(raw_measurements)

            # Merge raw and normalized (percentile versions)
            all_measurements = normalized_measurements  # Already contains both raw and _pct versions

            # Create instrument measurements object
            inst_meas = InstrumentMeasurements(
                instrument_key=instrument_key,
                asset_class=asset_class_str,
                raw_measurements=raw_measurements,
                normalized_measurements=normalized_measurements,
                measurement_names=list(all_measurements.keys()),
            )

            # Compute correlation matrix (for discovering relationships)
            try:
                inst_meas.correlation_matrix = self.measurement_engine.compute_correlation_matrix(
                    raw_measurements, start_idx=200
                )

                # Find inverse relationships
                inst_meas.inverse_relationships = self.measurement_engine.find_inverse_relationships(
                    inst_meas.correlation_matrix
                )
            except Exception:
                pass  # Correlation computation may fail with limited data

            self.instruments[instrument_key] = inst_meas
            return inst_meas

        except Exception as e:
            print(f"  [WARN] Failed to load {csv_file.name}: {e}")
            return None

    def load_all(self, verbose: bool = True) -> int:
        """Load all instruments from data directory."""
        csv_files = list(self.data_dir.glob("*.csv"))

        if verbose:
            print(f"\n[LOADING] {len(csv_files)} files from {self.data_dir}")

        loaded = 0
        for csv_file in csv_files:
            result = self.load_instrument(csv_file)
            if result:
                loaded += 1
                if verbose:
                    n_meas = len(result.measurement_names)
                    print(f"  [OK] {result.instrument_key} ({result.asset_class}): {n_meas} measurements")

        if verbose:
            print(f"\n[LOADED] {loaded} instruments with comprehensive measurements")

        return loaded


class ExplorationFeatureExtractor:
    """
    Physics-based feature extractor for RL agent.

    NO TRADITIONAL INDICATORS. Only physics measurements:
    - Kinematics (velocity → pop)
    - Energy (kinetic, potential, efficiency)
    - Flow dynamics (Reynolds, damping, viscosity)
    - Thermodynamics (entropy, phase compression)
    - Field theory (gradient, divergence, buying pressure)
    """

    def __init__(self):
        self.exploration_engine = ExplorationEngine()
        self.correlation_explorer = CorrelationExplorer()

        # Feature names (built dynamically)
        self._feature_names: Optional[List[str]] = None

    def get_features(self,
                     inst_meas: InstrumentMeasurements,
                     bar_idx: int) -> np.ndarray:
        """
        Get physics-based feature vector for a single bar.

        All features are rolling percentiles (0-1) for instrument-agnostic learning.
        """
        features = []

        # === PHYSICS MEASUREMENTS (percentile versions, 0-1 range) ===
        key_measurements = [
            # Kinematics (6 derivatives)
            'velocity_pct', 'acceleration_pct', 'jerk_pct',
            'snap_pct', 'crackle_pct', 'pop_pct',
            'momentum_pct', 'impulse_pct',

            # Energy
            'kinetic_energy_pct', 'potential_energy_compression_pct',
            'potential_energy_displacement_pct', 'energy_efficiency_pct',
            'energy_release_rate_pct',

            # Flow dynamics
            'reynolds_pct', 'damping_pct', 'viscosity_pct', 'liquidity_pct',
            'reynolds_momentum_corr_pct',

            # Thermodynamics
            'entropy_pct', 'entropy_rate_pct', 'phase_compression_pct',

            # Field theory
            'price_gradient_pct', 'gradient_magnitude_pct',
            'divergence_pct', 'buying_pressure_pct', 'body_ratio_pct',

            # Microstructure
            'spread_pct_pct', 'volume_surge_pct', 'volume_trend_pct',

            # Cross-interactions
            'energy_momentum_product_pct', 're_damping_ratio_pct',
            'entropy_energy_phase_pct', 'jerk_energy_pct', 'release_potential_pct',
        ]

        for name in key_measurements:
            if name in inst_meas.normalized_measurements:
                value = inst_meas.normalized_measurements[name][bar_idx]
                features.append(value if np.isfinite(value) else 0.5)  # Default to median
            else:
                features.append(0.5)  # Default to median percentile

        # === EXTREME FLAGS (binary features) ===
        extreme_measurements = [
            'velocity_extreme', 'jerk_extreme', 'kinetic_energy_extreme',
            'reynolds_extreme', 'entropy_extreme', 'phase_compression_extreme',
        ]

        for name in extreme_measurements:
            if name in inst_meas.normalized_measurements:
                value = inst_meas.normalized_measurements[name][bar_idx]
                features.append(value if np.isfinite(value) else 0.0)
            else:
                features.append(0.0)

        # === COMPOSITE SIGNALS (from exploration engine) ===
        bar_measurements = {}
        for name in inst_meas.normalized_measurements:
            val = inst_meas.normalized_measurements[name][bar_idx]
            if np.isfinite(val):
                bar_measurements[name] = val

        # Generate composite signals
        result = self.exploration_engine.process_bar(
            asset_class=inst_meas.asset_class,
            measurements=bar_measurements,
            volatility_level=bar_measurements.get('kinetic_energy_pct', 0.5),
        )

        # Add signal features
        signal_features = result['signal_features']
        features.extend(signal_features.tolist())

        return np.array(features, dtype=np.float32)

    def get_feature_names(self) -> List[str]:
        """Get names of all physics features."""
        if self._feature_names is not None:
            return self._feature_names

        names = [
            # Kinematics
            'velocity_pct', 'acceleration_pct', 'jerk_pct',
            'snap_pct', 'crackle_pct', 'pop_pct',
            'momentum_pct', 'impulse_pct',

            # Energy
            'kinetic_energy_pct', 'pe_compression_pct',
            'pe_displacement_pct', 'energy_efficiency_pct',
            'energy_release_rate_pct',

            # Flow dynamics
            'reynolds_pct', 'damping_pct', 'viscosity_pct', 'liquidity_pct',
            're_momentum_corr_pct',

            # Thermodynamics
            'entropy_pct', 'entropy_rate_pct', 'phase_compression_pct',

            # Field theory
            'price_gradient_pct', 'gradient_mag_pct',
            'divergence_pct', 'buying_pressure_pct', 'body_ratio_pct',

            # Microstructure
            'spread_pct', 'volume_surge_pct', 'volume_trend_pct',

            # Cross-interactions
            'energy_momentum_pct', 're_damping_ratio_pct',
            'entropy_energy_pct', 'jerk_energy_pct', 'release_potential_pct',

            # Extreme flags
            'velocity_extreme', 'jerk_extreme', 'ke_extreme',
            'reynolds_extreme', 'entropy_extreme', 'compression_extreme',
        ]

        # Add signal feature names from composite stacker
        stacker = CompositeStacker("temp")
        names.extend(stacker.get_feature_names())

        self._feature_names = names
        return names

    def record_trade(self,
                     inst_meas: InstrumentMeasurements,
                     entry_idx: int,
                     exit_idx: int,
                     pnl: float,
                     mae: float,
                     mfe: float,
                     bars_held: int):
        """Record a trade for physics-based discovery learning."""
        # Get entry measurements (physics only)
        entry_meas = {}
        for name in inst_meas.normalized_measurements:
            if entry_idx < len(inst_meas.normalized_measurements[name]):
                val = inst_meas.normalized_measurements[name][entry_idx]
                if np.isfinite(val):
                    entry_meas[name] = val

        # Get exit measurements
        exit_meas = {}
        for name in inst_meas.normalized_measurements:
            if exit_idx < len(inst_meas.normalized_measurements[name]):
                val = inst_meas.normalized_measurements[name][exit_idx]
                if np.isfinite(val):
                    exit_meas[name] = val

        # Record for correlation analysis (discovers which physics features predict outcomes)
        self.correlation_explorer.record_trade(
            entry_measurements=entry_meas,
            pnl=pnl,
            mae=mae,
            mfe=mfe,
        )

        # Record for composite signal learning
        stacker = self.exploration_engine.discovery_engine.get_stacker(inst_meas.asset_class)
        signals = stacker.generate_composite(entry_meas)

        self.exploration_engine.record_trade_outcome(
            asset_class=inst_meas.asset_class,
            signals=signals,
            pnl=pnl,
            measurements=entry_meas,
            volatility_level=entry_meas.get('kinetic_energy_pct', 0.5),  # Use physics measure
        )

    def get_discoveries(self) -> Dict:
        """Get all discoveries from exploration."""
        discoveries = self.exploration_engine.get_discoveries()

        # Add correlation discoveries
        predictive = self.correlation_explorer.find_predictive_features()
        discoveries['predictive_features'] = predictive

        return discoveries


# =============================================================================
# FACTORY FUNCTION FOR INTEGRATION
# =============================================================================

def create_exploration_components(data_dir: str = "data/master_standardized"):
    """
    Factory function to create all exploration components.

    Returns:
        loader: ExplorationDataLoader with all instruments loaded
        extractor: ExplorationFeatureExtractor for RL features
        engine: ExplorationEngine for discoveries
    """
    loader = ExplorationDataLoader(data_dir)
    loader.load_all(verbose=True)

    extractor = ExplorationFeatureExtractor()

    return loader, extractor, extractor.exploration_engine


# =============================================================================
# DISCOVERY REPORTER
# =============================================================================

class DiscoveryReporter:
    """
    Reports discoveries from exploration.

    Outputs what we've learned about:
    - Which measurements matter per class
    - Inverse relationships during volatility
    - Signal generator reliability
    """

    def __init__(self, exploration_engine: ExplorationEngine):
        self.engine = exploration_engine

    def generate_report(self) -> str:
        """Generate human-readable discovery report."""
        discoveries = self.engine.get_discoveries()

        lines = [
            "=" * 70,
            "  EXPLORATION DISCOVERIES",
            "=" * 70,
            "",
        ]

        # Class profiles
        lines.append("CLASS-SPECIFIC DISCOVERIES:")
        lines.append("-" * 50)

        for cls, profile in discoveries.get('class_profiles', {}).items():
            lines.append(f"\n  {cls}:")
            lines.append(f"    Dominant signal: {profile.get('dominant_signal', 'unknown')}")
            lines.append(f"    Dominant weight: {profile.get('dominant_weight', 0):.3f}")

            lines.append("    Generator reliability:")
            for gen, rel in profile.get('generator_reliability', {}).items():
                lines.append(f"      {gen}: {rel:.3f}")

        # Inverse relationships
        inversions = discoveries.get('inverse_relationships', [])
        if inversions:
            lines.append("\n" + "-" * 50)
            lines.append("INVERSE RELATIONSHIPS DURING HIGH VOLATILITY:")
            lines.append("-" * 50)

            for inv in inversions[:10]:  # Top 10
                lines.append(
                    f"  {inv['measurement_1']} vs {inv['measurement_2']}: "
                    f"low_vol={inv['low_vol_correlation']:.3f}, "
                    f"high_vol={inv['high_vol_correlation']:.3f} "
                    f"({inv['type']})"
                )

        lines.append("\n" + "=" * 70)

        return "\n".join(lines)

    def save_report(self, filepath: str = "results/discovery_report.txt"):
        """Save report to file."""
        report = self.generate_report()

        Path(filepath).parent.mkdir(exist_ok=True)
        with open(filepath, 'w') as f:
            f.write(report)

        print(f"[SAVED] Discovery report: {filepath}")


# Export
__all__ = [
    'InstrumentMeasurements',
    'ExplorationDataLoader',
    'ExplorationFeatureExtractor',
    'create_exploration_components',
    'DiscoveryReporter',
]
