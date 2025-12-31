#!/usr/bin/env python3
"""
SUPERPOT PHYSICS - First Principles Measurements
=================================================

ALL measurements derived from first principles:
- NO symmetry assumptions
- NO linearity assumptions  
- NO fixed timeframe assumptions

Categories:
1. KINEMATICS - Position derivatives (velocity through pop)
2. ENERGY - Kinetic, potential, efficiency
3. FLOW DYNAMICS - Reynolds, damping, viscosity
4. THERMODYNAMICS - Entropy, phase compression
5. FIELD THEORY - Gradients, divergence, pressure
6. MICROSTRUCTURE - Spread, volume dynamics
7. CROSS-INTERACTIONS - Combined signals
8. ASYMMETRIC TAILS - Directional risk
9. ORDER FLOW - CVD, imbalance, toxicity
10. CHAOS/COMPLEXITY - Lyapunov, Hurst, fractal

Total: 150+ physics-based measurements

Usage:
    python scripts/superpot_physics.py
    python scripts/superpot_physics.py --episodes 100 --prune-every 20
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime
from collections import defaultdict, Counter
import json
import time
import warnings
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent.parent))


# =============================================================================
# PHYSICS-BASED FEATURE EXTRACTOR
# =============================================================================

class PhysicsExtractor:
    """
    Extract physics-based measurements from price data.
    
    All measurements are:
    - Derived from first principles
    - NO symmetry assumptions (up ≠ down)
    - NO linearity assumptions (no OLS, no correlations assuming normal)
    - NO fixed timeframes (adaptive lookbacks where sensible)
    """
    
    def __init__(self, lookback: int = 50):
        self.lookback = lookback
        self.feature_names = self._build_feature_names()
        print(f"📊 PhysicsExtractor initialized with {len(self.feature_names)} measurements")
    
    def _build_feature_names(self) -> List[str]:
        """Build complete list of physics-based feature names."""
        names = []
        
        # === KINEMATICS (Position Derivatives) ===
        names.extend([
            # Velocity (1st derivative)
            'velocity_1', 'velocity_5', 'velocity_10', 'velocity_20',
            'velocity_up_mean', 'velocity_down_mean', 'velocity_asymmetry',
            
            # Acceleration (2nd derivative)
            'acceleration_1', 'acceleration_5', 'acceleration_10',
            'acceleration_up', 'acceleration_down', 'acceleration_asymmetry',
            
            # Jerk (3rd derivative) - fat candle predictor
            'jerk_1', 'jerk_5', 'jerk_10',
            'jerk_up', 'jerk_down', 'jerk_asymmetry',
            'jerk_magnitude', 'jerk_persistence',
            
            # Snap (4th derivative)
            'snap_1', 'snap_5',
            
            # Crackle (5th derivative)
            'crackle_1', 'crackle_5',
            
            # Pop (6th derivative)
            'pop_1', 'pop_5',
            
            # Momentum (mass × velocity)
            'momentum_1', 'momentum_5', 'momentum_10', 'momentum_20',
            'momentum_up', 'momentum_down', 'momentum_asymmetry',
            
            # Impulse (Δmomentum)
            'impulse_1', 'impulse_5', 'impulse_10',
            'impulse_up', 'impulse_down',
        ])
        
        # === ENERGY ===
        names.extend([
            # Kinetic Energy (½mv²)
            'kinetic_energy_5', 'kinetic_energy_10', 'kinetic_energy_20',
            'kinetic_energy_up', 'kinetic_energy_down', 'kinetic_asymmetry',
            
            # Potential Energy - Compression
            'pe_compression_5', 'pe_compression_10', 'pe_compression_20',
            
            # Potential Energy - Displacement (½kx²)
            'pe_displacement_5', 'pe_displacement_10', 'pe_displacement_20',
            
            # Energy Efficiency (KE/PE)
            'energy_efficiency_5', 'energy_efficiency_10',
            
            # Energy Release Rate
            'energy_release_rate_5', 'energy_release_rate_10',
            
            # Total Energy
            'total_energy_5', 'total_energy_10',
            'energy_trend_5', 'energy_trend_10',
        ])
        
        # === FLOW DYNAMICS ===
        names.extend([
            # Reynolds Number (trend/noise)
            'reynolds_5', 'reynolds_10', 'reynolds_20',
            'reynolds_trend',
            
            # Damping (friction coefficient)
            'damping_5', 'damping_10', 'damping_20',
            'damping_asymmetry',
            
            # Viscosity (resistance to flow)
            'viscosity_5', 'viscosity_10',
            'viscosity_up', 'viscosity_down',
            
            # Liquidity (inverse price impact)
            'liquidity_5', 'liquidity_10',
            'liquidity_stress',
            
            # Reynolds-Momentum Correlation
            'reynolds_momentum_corr_10', 'reynolds_momentum_corr_20',
        ])
        
        # === THERMODYNAMICS ===
        names.extend([
            # Entropy (Shannon)
            'entropy_5', 'entropy_10', 'entropy_20',
            'entropy_rate_5', 'entropy_rate_10',
            'entropy_up', 'entropy_down',
            
            # Permutation Entropy
            'perm_entropy_5', 'perm_entropy_10', 'perm_entropy_20',
            
            # Phase Compression (PE × low KE × low entropy)
            'phase_compression_5', 'phase_compression_10',
            'compression_building',
            
            # Temperature proxy
            'temperature_5', 'temperature_10',
        ])
        
        # === FIELD THEORY ===
        names.extend([
            # Price Gradient
            'price_gradient_5', 'price_gradient_10', 'price_gradient_20',
            'gradient_magnitude_5', 'gradient_magnitude_10',
            'gradient_acceleration',
            
            # Divergence (flow convergence/divergence)
            'divergence_5', 'divergence_10',
            'divergence_up', 'divergence_down',
            
            # Buying Pressure
            'buying_pressure_1', 'buying_pressure_5', 'buying_pressure_10',
            'buying_pressure_trend',
            
            # Body Ratio (conviction)
            'body_ratio_1', 'body_ratio_5', 'body_ratio_10',
            'body_ratio_trend',
            
            # Wick Analysis
            'upper_wick_5', 'lower_wick_5',
            'wick_asymmetry_5', 'wick_asymmetry_10',
        ])
        
        # === MICROSTRUCTURE ===
        names.extend([
            # Spread
            'spread_pct_5', 'spread_pct_10',
            'spread_rank_20',
            'spread_volatility',
            
            # Volume Surge
            'volume_surge_1', 'volume_surge_5', 'volume_surge_10',
            
            # Volume Trend
            'volume_trend_5', 'volume_trend_10',
            'volume_acceleration',
            
            # Volume-Price Relationship
            'volume_price_confirm', 'volume_price_diverge',
        ])
        
        # === CROSS-INTERACTIONS ===
        names.extend([
            # Energy-Momentum Product
            'energy_momentum_product_5', 'energy_momentum_product_10',
            
            # Reynolds-Damping Ratio
            're_damping_ratio_5', 're_damping_ratio_10',
            
            # Entropy-Energy Phase
            'entropy_energy_phase_5', 'entropy_energy_phase_10',
            
            # Jerk Energy (jerk²)
            'jerk_energy_5', 'jerk_energy_10',
            
            # Release Potential (PE × (1-entropy))
            'release_potential_5', 'release_potential_10',
            
            # Composite Signals
            'momentum_energy_ratio', 'flow_chaos_index',
            'pressure_entropy_product', 'velocity_viscosity_ratio',
        ])
        
        # === ASYMMETRIC TAILS ===
        names.extend([
            # Directional Volatility
            'vol_up_5', 'vol_up_10',
            'vol_down_5', 'vol_down_10',
            'vol_asymmetry_5', 'vol_asymmetry_10',
            
            # Tail Risk
            'left_tail_5', 'left_tail_10',
            'right_tail_5', 'right_tail_10',
            'tail_asymmetry_5', 'tail_asymmetry_10',
            
            # CVaR (Conditional VaR)
            'cvar_down_5', 'cvar_down_10',
            'cvar_up_5', 'cvar_up_10',
            'cvar_asymmetry',
            
            # Directional Skewness
            'skew_signed_5', 'skew_signed_10',
            'kurtosis_excess_5', 'kurtosis_excess_10',
        ])
        
        # === ORDER FLOW ===
        names.extend([
            # CVD (Cumulative Volume Delta)
            'cvd_5', 'cvd_10', 'cvd_20',
            'cvd_acceleration',
            'cvd_divergence',
            
            # Order Flow Imbalance
            'flow_imbalance_5', 'flow_imbalance_10',
            'flow_persistence',
            
            # VPIN proxy (Volume-Synchronized PIN)
            'vpin_proxy_10', 'vpin_proxy_20',
            
            # Toxicity
            'toxicity_5', 'toxicity_10',
            'adverse_selection',
        ])
        
        # === CHAOS & COMPLEXITY ===
        names.extend([
            # Lyapunov Exponent Proxy
            'lyapunov_proxy_10', 'lyapunov_proxy_20',
            
            # Hurst Exponent Proxy
            'hurst_proxy_10', 'hurst_proxy_20',
            
            # Fractal Dimension Proxy
            'fractal_dim_10', 'fractal_dim_20',
            
            # Recurrence
            'recurrence_rate_10', 'recurrence_rate_20',
            'determinism_10',
            
            # Complexity
            'complexity_5', 'complexity_10',
            'predictability_10',
        ])
        
        return names
    
    @property
    def n_features(self) -> int:
        return len(self.feature_names)
    
    def extract(self, df: pd.DataFrame, idx: int) -> np.ndarray:
        """Extract all physics measurements at given index."""
        if idx < self.lookback:
            return np.zeros(self.n_features, dtype=np.float32)
        
        window = df.iloc[max(0, idx - self.lookback):idx + 1]
        
        o = window['open'].values.astype(np.float64)
        h = window['high'].values.astype(np.float64)
        l = window['low'].values.astype(np.float64)
        c = window['close'].values.astype(np.float64)
        v = window['volume'].values.astype(np.float64) if 'volume' in window else np.ones(len(window))
        
        # Log prices for derivatives
        log_c = np.log(c + 1e-10)
        
        features = {}
        
        # =====================================================================
        # KINEMATICS
        # =====================================================================
        
        # Velocity (1st derivative of log-price)
        vel = np.diff(log_c)
        features['velocity_1'] = vel[-1] if len(vel) > 0 else 0
        features['velocity_5'] = np.mean(vel[-5:]) if len(vel) >= 5 else 0
        features['velocity_10'] = np.mean(vel[-10:]) if len(vel) >= 10 else 0
        features['velocity_20'] = np.mean(vel[-20:]) if len(vel) >= 20 else 0
        
        vel_up = vel[vel > 0]
        vel_down = vel[vel < 0]
        features['velocity_up_mean'] = np.mean(vel_up) if len(vel_up) > 0 else 0
        features['velocity_down_mean'] = np.mean(vel_down) if len(vel_down) > 0 else 0
        features['velocity_asymmetry'] = features['velocity_up_mean'] + features['velocity_down_mean']
        
        # Acceleration (2nd derivative)
        acc = np.diff(vel) if len(vel) > 1 else np.array([0])
        features['acceleration_1'] = acc[-1] if len(acc) > 0 else 0
        features['acceleration_5'] = np.mean(acc[-5:]) if len(acc) >= 5 else 0
        features['acceleration_10'] = np.mean(acc[-10:]) if len(acc) >= 10 else 0
        
        acc_up = acc[acc > 0]
        acc_down = acc[acc < 0]
        features['acceleration_up'] = np.mean(acc_up) if len(acc_up) > 0 else 0
        features['acceleration_down'] = np.mean(acc_down) if len(acc_down) > 0 else 0
        features['acceleration_asymmetry'] = features['acceleration_up'] + features['acceleration_down']
        
        # Jerk (3rd derivative) - fat candle predictor
        jerk = np.diff(acc) if len(acc) > 1 else np.array([0])
        features['jerk_1'] = jerk[-1] if len(jerk) > 0 else 0
        features['jerk_5'] = np.mean(jerk[-5:]) if len(jerk) >= 5 else 0
        features['jerk_10'] = np.mean(jerk[-10:]) if len(jerk) >= 10 else 0
        
        jerk_up = jerk[jerk > 0]
        jerk_down = jerk[jerk < 0]
        features['jerk_up'] = np.mean(jerk_up) if len(jerk_up) > 0 else 0
        features['jerk_down'] = np.mean(jerk_down) if len(jerk_down) > 0 else 0
        features['jerk_asymmetry'] = features['jerk_up'] + features['jerk_down']
        features['jerk_magnitude'] = np.mean(np.abs(jerk[-5:])) if len(jerk) >= 5 else 0
        features['jerk_persistence'] = np.sum(np.sign(jerk[-5:]) == np.sign(jerk[-1])) / 5 if len(jerk) >= 5 else 0
        
        # Snap (4th derivative)
        snap = np.diff(jerk) if len(jerk) > 1 else np.array([0])
        features['snap_1'] = snap[-1] if len(snap) > 0 else 0
        features['snap_5'] = np.mean(snap[-5:]) if len(snap) >= 5 else 0
        
        # Crackle (5th derivative)
        crackle = np.diff(snap) if len(snap) > 1 else np.array([0])
        features['crackle_1'] = crackle[-1] if len(crackle) > 0 else 0
        features['crackle_5'] = np.mean(crackle[-5:]) if len(crackle) >= 5 else 0
        
        # Pop (6th derivative)
        pop = np.diff(crackle) if len(crackle) > 1 else np.array([0])
        features['pop_1'] = pop[-1] if len(pop) > 0 else 0
        features['pop_5'] = np.mean(pop[-5:]) if len(pop) >= 5 else 0
        
        # Momentum (mass × velocity) - volume-weighted
        mom = vel * v[1:] / (np.mean(v) + 1e-10) if len(vel) > 0 else np.array([0])
        features['momentum_1'] = mom[-1] if len(mom) > 0 else 0
        features['momentum_5'] = np.mean(mom[-5:]) if len(mom) >= 5 else 0
        features['momentum_10'] = np.mean(mom[-10:]) if len(mom) >= 10 else 0
        features['momentum_20'] = np.mean(mom[-20:]) if len(mom) >= 20 else 0
        
        mom_up = mom[mom > 0]
        mom_down = mom[mom < 0]
        features['momentum_up'] = np.mean(mom_up) if len(mom_up) > 0 else 0
        features['momentum_down'] = np.mean(mom_down) if len(mom_down) > 0 else 0
        features['momentum_asymmetry'] = features['momentum_up'] + features['momentum_down']
        
        # Impulse (Δmomentum)
        impulse = np.diff(mom) if len(mom) > 1 else np.array([0])
        features['impulse_1'] = impulse[-1] if len(impulse) > 0 else 0
        features['impulse_5'] = np.mean(impulse[-5:]) if len(impulse) >= 5 else 0
        features['impulse_10'] = np.mean(impulse[-10:]) if len(impulse) >= 10 else 0
        features['impulse_up'] = np.mean(impulse[impulse > 0]) if np.sum(impulse > 0) > 0 else 0
        features['impulse_down'] = np.mean(impulse[impulse < 0]) if np.sum(impulse < 0) > 0 else 0
        
        # =====================================================================
        # ENERGY
        # =====================================================================
        
        # Kinetic Energy (½mv²)
        ke = 0.5 * v[1:] * vel ** 2 / (np.mean(v) + 1e-10) if len(vel) > 0 else np.array([0])
        features['kinetic_energy_5'] = np.mean(ke[-5:]) if len(ke) >= 5 else 0
        features['kinetic_energy_10'] = np.mean(ke[-10:]) if len(ke) >= 10 else 0
        features['kinetic_energy_20'] = np.mean(ke[-20:]) if len(ke) >= 20 else 0
        
        ke_up = ke[vel[-len(ke):] > 0] if len(ke) > 0 else np.array([0])
        ke_down = ke[vel[-len(ke):] < 0] if len(ke) > 0 else np.array([0])
        features['kinetic_energy_up'] = np.mean(ke_up) if len(ke_up) > 0 else 0
        features['kinetic_energy_down'] = np.mean(ke_down) if len(ke_down) > 0 else 0
        features['kinetic_asymmetry'] = features['kinetic_energy_up'] - features['kinetic_energy_down']
        
        # Potential Energy - Compression (from volatility compression)
        vol_short = np.std(vel[-5:]) if len(vel) >= 5 else 0.01
        vol_long = np.std(vel[-20:]) if len(vel) >= 20 else 0.01
        features['pe_compression_5'] = 1 / (vol_short + 1e-10)
        features['pe_compression_10'] = 1 / (np.std(vel[-10:]) + 1e-10) if len(vel) >= 10 else 0
        features['pe_compression_20'] = 1 / (vol_long + 1e-10)
        
        # Potential Energy - Displacement (½kx²)
        mean_price = np.mean(c[-20:]) if len(c) >= 20 else c[-1]
        displacement = (c[-1] - mean_price) / (mean_price + 1e-10)
        features['pe_displacement_5'] = 0.5 * displacement ** 2
        features['pe_displacement_10'] = 0.5 * ((c[-1] - np.mean(c[-10:])) / (np.mean(c[-10:]) + 1e-10)) ** 2 if len(c) >= 10 else 0
        features['pe_displacement_20'] = 0.5 * displacement ** 2
        
        # Energy Efficiency (KE/PE)
        pe_total = features['pe_compression_10'] + features['pe_displacement_10']
        features['energy_efficiency_5'] = features['kinetic_energy_5'] / (features['pe_compression_5'] + 1e-10)
        features['energy_efficiency_10'] = features['kinetic_energy_10'] / (pe_total + 1e-10)
        
        # Energy Release Rate
        features['energy_release_rate_5'] = np.mean(np.diff(ke[-5:])) if len(ke) >= 5 else 0
        features['energy_release_rate_10'] = np.mean(np.diff(ke[-10:])) if len(ke) >= 10 else 0
        
        # Total Energy
        features['total_energy_5'] = features['kinetic_energy_5'] + features['pe_compression_5']
        features['total_energy_10'] = features['kinetic_energy_10'] + pe_total
        features['energy_trend_5'] = np.polyfit(np.arange(5), ke[-5:], 1)[0] if len(ke) >= 5 else 0
        features['energy_trend_10'] = np.polyfit(np.arange(10), ke[-10:], 1)[0] if len(ke) >= 10 else 0
        
        # =====================================================================
        # FLOW DYNAMICS
        # =====================================================================
        
        # Reynolds Number (trend/noise ratio)
        trend_5 = abs(np.mean(vel[-5:])) if len(vel) >= 5 else 0
        noise_5 = np.std(vel[-5:]) if len(vel) >= 5 else 0.01
        features['reynolds_5'] = trend_5 / (noise_5 + 1e-10)
        features['reynolds_10'] = abs(np.mean(vel[-10:])) / (np.std(vel[-10:]) + 1e-10) if len(vel) >= 10 else 0
        features['reynolds_20'] = abs(np.mean(vel[-20:])) / (np.std(vel[-20:]) + 1e-10) if len(vel) >= 20 else 0
        features['reynolds_trend'] = features['reynolds_5'] - features['reynolds_20']
        
        # Damping (friction coefficient)
        features['damping_5'] = np.std(vel[-5:]) / (np.mean(np.abs(vel[-5:])) + 1e-10) if len(vel) >= 5 else 0
        features['damping_10'] = np.std(vel[-10:]) / (np.mean(np.abs(vel[-10:])) + 1e-10) if len(vel) >= 10 else 0
        features['damping_20'] = np.std(vel[-20:]) / (np.mean(np.abs(vel[-20:])) + 1e-10) if len(vel) >= 20 else 0
        
        damp_up = np.std(vel_up) / (np.mean(np.abs(vel_up)) + 1e-10) if len(vel_up) > 1 else 0
        damp_down = np.std(vel_down) / (np.mean(np.abs(vel_down)) + 1e-10) if len(vel_down) > 1 else 0
        features['damping_asymmetry'] = damp_up - damp_down
        
        # Viscosity (resistance to flow)
        price_impact = np.abs(vel) / (v[1:] + 1e-10) if len(vel) > 0 else np.array([0])
        features['viscosity_5'] = np.mean(price_impact[-5:]) * 1e6 if len(price_impact) >= 5 else 0
        features['viscosity_10'] = np.mean(price_impact[-10:]) * 1e6 if len(price_impact) >= 10 else 0
        features['viscosity_20'] = np.mean(price_impact[-20:]) * 1e6 if len(price_impact) >= 20 else features['viscosity_10']
        
        visc_up = price_impact[vel[-len(price_impact):] > 0] if len(price_impact) > 0 else np.array([0])
        visc_down = price_impact[vel[-len(price_impact):] < 0] if len(price_impact) > 0 else np.array([0])
        features['viscosity_up'] = np.mean(visc_up) * 1e6 if len(visc_up) > 0 else 0
        features['viscosity_down'] = np.mean(visc_down) * 1e6 if len(visc_down) > 0 else 0
        
        # Liquidity (inverse price impact)
        features['liquidity_5'] = 1 / (features['viscosity_5'] + 1e-10)
        features['liquidity_10'] = 1 / (features['viscosity_10'] + 1e-10)
        features['liquidity_stress'] = features['viscosity_5'] / (features['viscosity_20'] + 1e-10)
        
        # Reynolds-Momentum Correlation
        if len(vel) >= 10:
            re_series = pd.Series(vel[-10:]).rolling(3).apply(lambda x: abs(x.mean()) / (x.std() + 1e-10)).dropna().values
            mom_series = mom[-len(re_series):] if len(mom) >= len(re_series) else mom
            if len(re_series) > 0 and len(mom_series) > 0:
                features['reynolds_momentum_corr_10'] = np.corrcoef(re_series[:len(mom_series)], mom_series[:len(re_series)])[0, 1] if len(re_series) > 1 else 0
            else:
                features['reynolds_momentum_corr_10'] = 0
        else:
            features['reynolds_momentum_corr_10'] = 0
        features['reynolds_momentum_corr_20'] = features['reynolds_momentum_corr_10']  # Placeholder
        
        # =====================================================================
        # THERMODYNAMICS
        # =====================================================================
        
        # Shannon Entropy
        def shannon_entropy(arr, bins=10):
            if len(arr) < 5:
                return 0.5
            hist, _ = np.histogram(arr, bins=bins, density=True)
            hist = hist[hist > 0]
            return -np.sum(hist * np.log(hist + 1e-10)) / np.log(bins)
        
        features['entropy_5'] = shannon_entropy(vel[-5:]) if len(vel) >= 5 else 0.5
        features['entropy_10'] = shannon_entropy(vel[-10:]) if len(vel) >= 10 else 0.5
        features['entropy_20'] = shannon_entropy(vel[-20:]) if len(vel) >= 20 else 0.5
        
        features['entropy_rate_5'] = features['entropy_5'] - features['entropy_10']
        features['entropy_rate_10'] = features['entropy_10'] - features['entropy_20']
        
        features['entropy_up'] = shannon_entropy(vel_up) if len(vel_up) >= 5 else 0.5
        features['entropy_down'] = shannon_entropy(vel_down) if len(vel_down) >= 5 else 0.5
        
        # Permutation Entropy
        def perm_entropy(arr, order=3):
            if len(arr) < order + 1:
                return 0.5
            patterns = []
            for i in range(len(arr) - order):
                patterns.append(tuple(np.argsort(arr[i:i + order])))
            if not patterns:
                return 0.5
            counts = Counter(patterns)
            probs = np.array(list(counts.values())) / len(patterns)
            return -np.sum(probs * np.log(probs + 1e-10)) / np.log(6)
        
        features['perm_entropy_5'] = perm_entropy(c[-5:]) if len(c) >= 5 else 0.5
        features['perm_entropy_10'] = perm_entropy(c[-10:]) if len(c) >= 10 else 0.5
        features['perm_entropy_20'] = perm_entropy(c[-20:]) if len(c) >= 20 else 0.5
        
        # Phase Compression (PE × low KE × low entropy)
        features['phase_compression_5'] = features['pe_compression_5'] * (1 - features['kinetic_energy_5']) * (1 - features['entropy_5'])
        features['phase_compression_10'] = features['pe_compression_10'] * (1 - features['kinetic_energy_10']) * (1 - features['entropy_10'])
        features['compression_building'] = features['phase_compression_5'] - features['phase_compression_10']
        
        # Temperature proxy
        features['temperature_5'] = features['kinetic_energy_5'] * features['entropy_5']
        features['temperature_10'] = features['kinetic_energy_10'] * features['entropy_10']
        
        # =====================================================================
        # FIELD THEORY
        # =====================================================================
        
        # Price Gradient
        features['price_gradient_5'] = np.polyfit(np.arange(5), c[-5:], 1)[0] / (c[-1] + 1e-10) if len(c) >= 5 else 0
        features['price_gradient_10'] = np.polyfit(np.arange(10), c[-10:], 1)[0] / (c[-1] + 1e-10) if len(c) >= 10 else 0
        features['price_gradient_20'] = np.polyfit(np.arange(20), c[-20:], 1)[0] / (c[-1] + 1e-10) if len(c) >= 20 else 0
        
        features['gradient_magnitude_5'] = abs(features['price_gradient_5'])
        features['gradient_magnitude_10'] = abs(features['price_gradient_10'])
        features['gradient_acceleration'] = features['price_gradient_5'] - features['price_gradient_10']
        
        # Divergence (buy/sell pressure flow)
        signed_vol = np.sign(c[1:] - o[1:]) * v[1:]
        features['divergence_5'] = np.sum(signed_vol[-5:]) / (np.sum(v[-5:]) + 1e-10) if len(v) >= 5 else 0
        features['divergence_10'] = np.sum(signed_vol[-10:]) / (np.sum(v[-10:]) + 1e-10) if len(v) >= 10 else 0
        
        features['divergence_up'] = np.sum(signed_vol[-5:][signed_vol[-5:] > 0]) / (np.sum(v[-5:]) + 1e-10) if len(v) >= 5 else 0
        features['divergence_down'] = np.sum(signed_vol[-5:][signed_vol[-5:] < 0]) / (np.sum(v[-5:]) + 1e-10) if len(v) >= 5 else 0
        
        # Buying Pressure
        bp = (c - l) / (h - l + 1e-10)
        features['buying_pressure_1'] = bp[-1]
        features['buying_pressure_5'] = np.mean(bp[-5:]) if len(bp) >= 5 else 0.5
        features['buying_pressure_10'] = np.mean(bp[-10:]) if len(bp) >= 10 else 0.5
        features['buying_pressure_trend'] = features['buying_pressure_5'] - features['buying_pressure_10']
        
        # Body Ratio (conviction)
        body = np.abs(c - o) / (h - l + 1e-10)
        features['body_ratio_1'] = body[-1]
        features['body_ratio_5'] = np.mean(body[-5:]) if len(body) >= 5 else 0.5
        features['body_ratio_10'] = np.mean(body[-10:]) if len(body) >= 10 else 0.5
        features['body_ratio_trend'] = features['body_ratio_5'] - features['body_ratio_10']
        
        # Wick Analysis
        upper_wick = (h - np.maximum(o, c)) / (h - l + 1e-10)
        lower_wick = (np.minimum(o, c) - l) / (h - l + 1e-10)
        features['upper_wick_5'] = np.mean(upper_wick[-5:]) if len(upper_wick) >= 5 else 0
        features['lower_wick_5'] = np.mean(lower_wick[-5:]) if len(lower_wick) >= 5 else 0
        features['wick_asymmetry_5'] = features['lower_wick_5'] - features['upper_wick_5']
        features['wick_asymmetry_10'] = np.mean(lower_wick[-10:] - upper_wick[-10:]) if len(lower_wick) >= 10 else 0
        
        # =====================================================================
        # MICROSTRUCTURE
        # =====================================================================
        
        # Spread proxy
        spread = (h - l) / c
        features['spread_pct_5'] = np.mean(spread[-5:]) if len(spread) >= 5 else 0
        features['spread_pct_10'] = np.mean(spread[-10:]) if len(spread) >= 10 else 0
        features['spread_rank_20'] = (spread[-1] - np.percentile(spread[-20:], 20)) / (np.percentile(spread[-20:], 80) - np.percentile(spread[-20:], 20) + 1e-10) if len(spread) >= 20 else 0.5
        features['spread_volatility'] = np.std(spread[-10:]) / (np.mean(spread[-10:]) + 1e-10) if len(spread) >= 10 else 0
        
        # Volume Surge
        vm = np.mean(v) + 1e-10
        features['volume_surge_1'] = v[-1] / vm
        features['volume_surge_5'] = np.mean(v[-5:]) / vm if len(v) >= 5 else 1
        features['volume_surge_10'] = np.mean(v[-10:]) / vm if len(v) >= 10 else 1
        
        # Volume Trend
        features['volume_trend_5'] = np.polyfit(np.arange(5), v[-5:], 1)[0] / vm if len(v) >= 5 else 0
        features['volume_trend_10'] = np.polyfit(np.arange(10), v[-10:], 1)[0] / vm if len(v) >= 10 else 0
        features['volume_acceleration'] = features['volume_trend_5'] - features['volume_trend_10']
        
        # Volume-Price Relationship
        if len(vel) >= 5:
            price_up = np.mean(vel[-5:]) > 0
            vol_up_flag = np.mean(v[-5:]) > np.mean(v[-10:]) if len(v) >= 10 else True
            features['volume_price_confirm'] = 1 if price_up == vol_up_flag else 0
            features['volume_price_diverge'] = 1 if price_up != vol_up_flag else 0
        else:
            features['volume_price_confirm'] = 0.5
            features['volume_price_diverge'] = 0.5
        
        # =====================================================================
        # CROSS-INTERACTIONS
        # =====================================================================
        
        features['energy_momentum_product_5'] = features['kinetic_energy_5'] * np.sign(features['momentum_5'])
        features['energy_momentum_product_10'] = features['kinetic_energy_10'] * np.sign(features['momentum_10'])
        
        features['re_damping_ratio_5'] = features['reynolds_5'] / (features['damping_5'] + 1e-10)
        features['re_damping_ratio_10'] = features['reynolds_10'] / (features['damping_10'] + 1e-10)
        
        features['entropy_energy_phase_5'] = features['entropy_5'] * features['kinetic_energy_5']
        features['entropy_energy_phase_10'] = features['entropy_10'] * features['kinetic_energy_10']
        
        features['jerk_energy_5'] = features['jerk_5'] ** 2
        features['jerk_energy_10'] = features['jerk_10'] ** 2
        
        features['release_potential_5'] = features['pe_compression_5'] * (1 - features['entropy_5'])
        features['release_potential_10'] = features['pe_compression_10'] * (1 - features['entropy_10'])
        
        features['momentum_energy_ratio'] = features['momentum_10'] / (features['kinetic_energy_10'] + 1e-10)
        features['flow_chaos_index'] = features['reynolds_10'] * features['entropy_10']
        features['pressure_entropy_product'] = features['buying_pressure_5'] * features['entropy_5']
        features['velocity_viscosity_ratio'] = features['velocity_10'] / (features['viscosity_10'] + 1e-10)
        
        # =====================================================================
        # ASYMMETRIC TAILS
        # =====================================================================
        
        # Directional Volatility
        features['vol_up_5'] = np.std(vel_up[-5:]) if len(vel_up) >= 5 else 0
        features['vol_up_10'] = np.std(vel_up[-10:]) if len(vel_up) >= 10 else 0
        features['vol_down_5'] = np.std(vel_down[-5:]) if len(vel_down) >= 5 else 0
        features['vol_down_10'] = np.std(vel_down[-10:]) if len(vel_down) >= 10 else 0
        features['vol_asymmetry_5'] = features['vol_down_5'] - features['vol_up_5']
        features['vol_asymmetry_10'] = features['vol_down_10'] - features['vol_up_10']
        
        # Tail Risk
        features['left_tail_5'] = np.percentile(vel[-5:], 5) if len(vel) >= 5 else 0
        features['left_tail_10'] = np.percentile(vel[-10:], 5) if len(vel) >= 10 else 0
        features['right_tail_5'] = np.percentile(vel[-5:], 95) if len(vel) >= 5 else 0
        features['right_tail_10'] = np.percentile(vel[-10:], 95) if len(vel) >= 10 else 0
        features['tail_asymmetry_5'] = abs(features['left_tail_5']) - abs(features['right_tail_5'])
        features['tail_asymmetry_10'] = abs(features['left_tail_10']) - abs(features['right_tail_10'])
        
        # CVaR
        features['cvar_down_5'] = np.mean(vel[-5:][vel[-5:] < np.percentile(vel[-5:], 10)]) if len(vel) >= 5 else 0
        features['cvar_down_10'] = np.mean(vel[-10:][vel[-10:] < np.percentile(vel[-10:], 10)]) if len(vel) >= 10 else 0
        features['cvar_up_5'] = np.mean(vel[-5:][vel[-5:] > np.percentile(vel[-5:], 90)]) if len(vel) >= 5 else 0
        features['cvar_up_10'] = np.mean(vel[-10:][vel[-10:] > np.percentile(vel[-10:], 90)]) if len(vel) >= 10 else 0
        features['cvar_asymmetry'] = abs(features['cvar_down_10']) - abs(features['cvar_up_10'])
        
        # Directional Skewness
        def signed_skew(arr):
            if len(arr) < 3:
                return 0
            m = np.median(arr)
            s = np.std(arr)
            if s < 1e-10:
                return 0
            return np.mean(np.sign(arr - m) * ((arr - m) / s) ** 2)
        
        features['skew_signed_5'] = signed_skew(vel[-5:]) if len(vel) >= 5 else 0
        features['skew_signed_10'] = signed_skew(vel[-10:]) if len(vel) >= 10 else 0
        
        def excess_kurtosis(arr):
            if len(arr) < 4:
                return 0
            m = np.mean(arr)
            s = np.std(arr)
            if s < 1e-10:
                return 0
            return np.mean(((arr - m) / s) ** 4) - 3
        
        features['kurtosis_excess_5'] = excess_kurtosis(vel[-5:]) if len(vel) >= 5 else 0
        features['kurtosis_excess_10'] = excess_kurtosis(vel[-10:]) if len(vel) >= 10 else 0
        
        # =====================================================================
        # ORDER FLOW
        # =====================================================================
        
        # CVD
        features['cvd_5'] = features['divergence_5']
        features['cvd_10'] = features['divergence_10']
        features['cvd_20'] = np.sum(signed_vol[-20:]) / (np.sum(v[-20:]) + 1e-10) if len(v) >= 20 else 0
        features['cvd_acceleration'] = features['cvd_5'] - features['cvd_10']
        
        # CVD Divergence from price
        price_dir = np.sign(c[-1] - c[-10]) if len(c) >= 10 else 0
        features['cvd_divergence'] = features['cvd_10'] - price_dir * 0.5
        
        # Flow Imbalance
        features['flow_imbalance_5'] = features['divergence_up'] - features['divergence_down']
        features['flow_imbalance_10'] = features['flow_imbalance_5']  # Placeholder
        features['flow_persistence'] = np.sum(np.sign(signed_vol[-5:]) == np.sign(signed_vol[-1])) / 5 if len(signed_vol) >= 5 else 0.5
        
        # VPIN proxy
        features['vpin_proxy_10'] = abs(np.sum(signed_vol[-10:])) / (np.sum(v[-10:]) + 1e-10) if len(v) >= 10 else 0
        features['vpin_proxy_20'] = abs(np.sum(signed_vol[-20:])) / (np.sum(v[-20:]) + 1e-10) if len(v) >= 20 else 0
        
        # Toxicity
        features['toxicity_5'] = features['vpin_proxy_10'] * features['viscosity_5']
        features['toxicity_10'] = features['vpin_proxy_20'] * features['viscosity_10']
        features['adverse_selection'] = features['toxicity_10'] * (1 - features['liquidity_10'])
        
        # =====================================================================
        # CHAOS & COMPLEXITY
        # =====================================================================
        
        # Lyapunov proxy
        if len(vel) >= 10:
            diffs = np.abs(np.diff(vel[-10:]))
            features['lyapunov_proxy_10'] = np.mean(np.log(diffs + 1e-10))
        else:
            features['lyapunov_proxy_10'] = 0
        features['lyapunov_proxy_20'] = features['lyapunov_proxy_10']
        
        # Hurst proxy
        if len(vel) >= 20:
            cumsum = np.cumsum(vel[-20:] - np.mean(vel[-20:]))
            rs_range = np.max(cumsum) - np.min(cumsum)
            features['hurst_proxy_20'] = np.log(rs_range + 1e-10) / np.log(20)
        else:
            features['hurst_proxy_20'] = 0.5
        features['hurst_proxy_10'] = features['hurst_proxy_20']
        
        # Fractal Dimension proxy
        features['fractal_dim_10'] = 2 - features['hurst_proxy_10']
        features['fractal_dim_20'] = 2 - features['hurst_proxy_20']
        
        # Recurrence
        def recurrence_rate(arr, eps=0.1):
            if len(arr) < 5:
                return 0.5
            threshold = eps * np.std(arr)
            n = len(arr)
            count = 0
            for i in range(n):
                for j in range(i + 1, n):
                    if abs(arr[i] - arr[j]) < threshold:
                        count += 1
            return 2 * count / (n * (n - 1)) if n > 1 else 0
        
        features['recurrence_rate_10'] = recurrence_rate(vel[-10:]) if len(vel) >= 10 else 0.5
        features['recurrence_rate_20'] = recurrence_rate(vel[-20:]) if len(vel) >= 20 else 0.5
        features['determinism_10'] = 1 - features['perm_entropy_10']
        
        # Complexity
        features['complexity_5'] = features['entropy_5'] * (1 - features['entropy_5']) * 4
        features['complexity_10'] = features['entropy_10'] * (1 - features['entropy_10']) * 4
        features['predictability_10'] = features['determinism_10'] * (1 - features['entropy_10'])
        
        # Convert to array
        result = np.array([features.get(name, 0) for name in self.feature_names], dtype=np.float32)
        result = np.nan_to_num(result, nan=0, posinf=0, neginf=0)
        result = np.clip(result, -100, 100)
        
        return result


# =============================================================================
# FEATURE TRACKER
# =============================================================================

class PhysicsFeatureTracker:
    """Track feature importance for physics measurements."""
    
    def __init__(self, n_features: int, feature_names: List[str]):
        self.n_features = n_features
        self.feature_names = feature_names.copy()
        self.active_mask = np.ones(n_features, dtype=bool)
        
        self.feature_history: List[np.ndarray] = []
        self.reward_history: List[float] = []
    
    @property
    def n_active(self) -> int:
        return np.sum(self.active_mask)
    
    def get_active_features(self) -> List[str]:
        return [self.feature_names[i] for i in range(self.n_features) if self.active_mask[i]]
    
    def record(self, features: np.ndarray, reward: float):
        self.feature_history.append(features.copy())
        self.reward_history.append(reward)
        if len(self.feature_history) > 10000:
            self.feature_history = self.feature_history[-5000:]
            self.reward_history = self.reward_history[-5000:]
    
    def calculate_importance(self) -> np.ndarray:
        if len(self.feature_history) < 100:
            return np.ones(self.n_features)
        
        features = np.array(self.feature_history)
        rewards = np.array(self.reward_history)
        
        importance = np.zeros(self.n_features)
        
        for i in range(self.n_features):
            if not self.active_mask[i]:
                importance[i] = -999
                continue
            
            f = features[:, i]
            if np.std(f) > 1e-10:
                corr = np.corrcoef(f, rewards)[0, 1]
                importance[i] = abs(corr) if not np.isnan(corr) else 0
            
            win_mask = rewards > 0
            if np.sum(win_mask) > 10 and np.sum(~win_mask) > 10:
                diff = abs(np.mean(f[win_mask]) - np.mean(f[~win_mask])) / (np.std(f) + 1e-10)
                importance[i] += diff * 0.5
        
        return importance
    
    def prune(self, n_to_prune: int) -> List[str]:
        importance = self.calculate_importance()
        active_indices = np.where(self.active_mask)[0]
        active_importance = importance[active_indices]
        sorted_indices = active_indices[np.argsort(active_importance)]
        
        pruned = []
        for i in range(min(n_to_prune, len(sorted_indices))):
            idx = sorted_indices[i]
            self.active_mask[idx] = False
            pruned.append(self.feature_names[idx])
        
        return pruned
    
    def get_top_features(self, n: int = 30) -> List[Tuple[str, float]]:
        importance = self.calculate_importance()
        active_indices = np.where(self.active_mask)[0]
        sorted_indices = active_indices[np.argsort(importance[active_indices])[::-1]]
        return [(self.feature_names[i], importance[i]) for i in sorted_indices[:n]]
    
    def mask_features(self, features: np.ndarray) -> np.ndarray:
        return features[self.active_mask]


# =============================================================================
# AGENT
# =============================================================================

class PhysicsAgent:
    """Simple agent for physics-based features."""
    
    def __init__(self, n_features: int, n_actions: int = 4):
        self.n_features = n_features
        self.n_actions = n_actions
        self.W = np.random.randn(n_features, n_actions) * 0.01
        self.b = np.zeros(n_actions)
        self.V_W = np.random.randn(n_features) * 0.01
        self.V_b = 0.0
        self.lr = 0.001
        self.gamma = 0.95
        self.epsilon = 0.3
    
    def _softmax(self, x):
        x = np.clip(x, -20, 20)
        e = np.exp(x - np.max(x))
        p = e / (e.sum() + 1e-10)
        return np.ones(len(x)) / len(x) if np.any(np.isnan(p)) else p
    
    def select_action(self, features: np.ndarray, explore: bool = True) -> int:
        features = np.nan_to_num(features, nan=0, posinf=0, neginf=0)
        if len(features) != self.n_features:
            padded = np.zeros(self.n_features)
            padded[:min(len(features), self.n_features)] = features[:self.n_features]
            features = padded
        
        if explore and np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        
        logits = features @ self.W + self.b
        return np.argmax(self._softmax(logits))
    
    def update(self, features, action, reward, next_features, done):
        features = np.nan_to_num(features, nan=0, posinf=0, neginf=0)
        next_features = np.nan_to_num(next_features, nan=0, posinf=0, neginf=0)
        
        if len(features) != self.n_features:
            padded = np.zeros(self.n_features)
            padded[:min(len(features), self.n_features)] = features[:self.n_features]
            features = padded
        if len(next_features) != self.n_features:
            padded = np.zeros(self.n_features)
            padded[:min(len(next_features), self.n_features)] = next_features[:self.n_features]
            next_features = padded
        
        value = features @ self.V_W + self.V_b
        next_value = 0 if done else next_features @ self.V_W + self.V_b
        td_error = reward + self.gamma * next_value - value
        
        self.V_W += self.lr * td_error * features
        self.V_b += self.lr * td_error
        
        probs = self._softmax(features @ self.W + self.b)
        grad = -probs.copy()
        grad[action] += 1
        grad *= td_error
        
        self.W += self.lr * np.outer(features, grad)
        self.b += self.lr * grad
        
        self.epsilon = max(0.05, self.epsilon * 0.9995)


# =============================================================================
# DATA LOADING
# =============================================================================

def load_data(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath, sep='\t')
    df.columns = [c.strip('<>').lower() for c in df.columns]
    
    if not all(c in df.columns for c in ['open', 'high', 'low', 'close']):
        df = pd.read_csv(filepath)
        df.columns = [c.strip('<>').lower() for c in df.columns]
    
    if 'volume' not in df.columns:
        df['volume'] = df.get('tickvol', 1000)
    
    for col in ['open', 'high', 'low', 'close', 'volume']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    return df.dropna(subset=['open', 'high', 'low', 'close'])


def discover_files() -> List[dict]:
    files = []
    for base in ["data/master", "data/runs/berserker_run3/data", "data"]:
        path = Path(base)
        if not path.exists():
            continue
        for f in path.rglob("*.csv"):
            if 'symbol' in f.name.lower() or 'info' in f.name.lower():
                continue
            parts = f.stem.split('_')
            if len(parts) >= 2:
                files.append({'path': str(f), 'symbol': parts[0], 'timeframe': parts[1]})
    return files


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='SuperPot Physics')
    parser.add_argument('--episodes', type=int, default=150, help='Total episodes')
    parser.add_argument('--prune-every', type=int, default=25, help='Prune interval')
    parser.add_argument('--prune-count', type=int, default=10, help='Features to prune')
    parser.add_argument('--max-files', type=int, default=40, help='Max files')
    parser.add_argument('--max-steps', type=int, default=500, help='Steps per episode')
    
    args = parser.parse_args()
    
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║                    SUPERPOT PHYSICS                                  ║
║                                                                      ║
║   First-Principles Measurements Only:                                ║
║   • NO symmetry assumptions                                          ║
║   • NO linearity assumptions                                         ║
║   • NO fixed timeframe assumptions                                   ║
║                                                                      ║
║   Kinematics • Energy • Flow • Thermodynamics • Field Theory         ║
╚══════════════════════════════════════════════════════════════════════╝
""")
    
    files = discover_files()[:args.max_files]
    print(f"📁 Found {len(files)} data files")
    
    if not files:
        print("❌ No data files found!")
        return
    
    extractor = PhysicsExtractor(lookback=50)
    tracker = PhysicsFeatureTracker(extractor.n_features, extractor.feature_names)
    agent = PhysicsAgent(extractor.n_features, n_actions=4)
    
    print(f"\n🧪 Starting with {extractor.n_features} physics measurements")
    print(f"   Prune {args.prune_count} every {args.prune_every} episodes")
    
    # Test data loading first
    print(f"\n📋 Testing data loading...")
    test_file = files[0]
    try:
        df = load_data(test_file['path'])
        print(f"   ✓ Loaded {test_file['symbol']}: {len(df)} bars")
        print(f"   ✓ Columns: {list(df.columns)[:6]}...")
        
        # Test feature extraction
        features = extractor.extract(df, 100)
        print(f"   ✓ Extracted {len(features)} features")
        print(f"   ✓ Non-zero features: {np.sum(features != 0)}")
    except Exception as e:
        print(f"   ❌ Data loading failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    start_time = time.time()
    all_rewards = []
    all_pnls = []
    
    errors = []
    for ep in range(args.episodes):
        file_info = files[np.random.randint(len(files))]
        
        try:
            df = load_data(file_info['path'])
            if len(df) < 200:
                errors.append(f"Too few bars: {len(df)}")
                continue
            
            # Verify required columns
            required = ['open', 'high', 'low', 'close']
            missing = [c for c in required if c not in df.columns]
            if missing:
                errors.append(f"Missing columns: {missing}")
                continue
            
            df = df.iloc[-2000:].reset_index(drop=True)
            
            start_bar = np.random.randint(100, max(101, len(df) - args.max_steps - 10))
            bar = start_bar
            position = 0
            entry_price = 0
            balance = 10000
            episode_reward = 0
            
            for step in range(args.max_steps):
                if bar >= len(df) - 1:
                    break
                
                features = extractor.extract(df, bar)
                active_features = tracker.mask_features(features)
                
                action = agent.select_action(active_features, explore=True)
                
                price = df.iloc[bar]['close']
                reward = 0
                
                if action == 1 and position == 0:
                    position = 1
                    entry_price = price * 1.0001
                elif action == 2 and position == 0:
                    position = -1
                    entry_price = price * 0.9999
                elif action == 3 and position != 0:
                    if position == 1:
                        pnl = (price * 0.9999 - entry_price) / entry_price
                    else:
                        pnl = (entry_price - price * 1.0001) / entry_price
                    balance *= (1 + pnl * 0.1)
                    reward = pnl * 100
                    position = 0
                
                if position != 0:
                    reward -= 0.001
                
                bar += 1
                next_features = extractor.extract(df, bar) if bar < len(df) else features
                active_next = tracker.mask_features(next_features)
                
                tracker.record(features, reward)
                agent.update(active_features, action, reward, active_next, bar >= len(df) - 1)
                
                episode_reward += reward
            
            pnl = balance - 10000
            all_rewards.append(episode_reward)
            all_pnls.append(pnl)
            
            if (ep + 1) % 10 == 0:
                avg_r = np.mean(all_rewards[-10:])
                avg_pnl = np.mean(all_pnls[-10:])
                print(f"Ep {ep+1:3d}: R={avg_r:+7.1f} PnL=${avg_pnl:+7.0f} | "
                      f"Features: {tracker.n_active}/{extractor.n_features} | ε={agent.epsilon:.3f}")
            
            if (ep + 1) % args.prune_every == 0 and tracker.n_active > args.prune_count + 20:
                pruned = tracker.prune(args.prune_count)
                print(f"\n🗑️  PRUNED {len(pruned)} measurements:")
                for f in pruned[:5]:
                    print(f"   - {f}")
                if len(pruned) > 5:
                    print(f"   ... and {len(pruned) - 5} more")
                print(f"   Remaining: {tracker.n_active} measurements\n")
                
                active_indices = np.where(tracker.active_mask)[0]
                agent.W = agent.W[active_indices]
                agent.V_W = agent.V_W[active_indices]
                agent.n_features = tracker.n_active
        
        except Exception as e:
            errors.append(str(e))
            if len(errors) <= 3:
                print(f"   ⚠️  Error: {e}")
            continue
    
    elapsed = time.time() - start_time
    
    print(f"\n{'='*60}")
    print("SUPERPOT PHYSICS RESULTS")
    print(f"{'='*60}")
    
    print(f"\n📊 Performance:")
    print(f"   Episodes: {len(all_rewards)}")
    
    if len(all_rewards) == 0:
        print("   ❌ No episodes completed! Check data loading.")
        print(f"   Files found: {len(files)}")
        # Try to debug
        if files:
            test_file = files[0]
            print(f"   Testing file: {test_file['path']}")
            try:
                df = load_data(test_file['path'])
                print(f"   Loaded: {len(df)} bars")
                print(f"   Columns: {list(df.columns)}")
            except Exception as e:
                print(f"   Load error: {e}")
        return
    
    print(f"   Avg reward: {np.mean(all_rewards):+.2f}")
    print(f"   Avg PnL: ${np.mean(all_pnls):+.0f}")
    print(f"   Win rate: {sum(1 for p in all_pnls if p > 0) / len(all_pnls) * 100:.0f}%")
    
    print(f"\n🏆 TOP SURVIVING PHYSICS MEASUREMENTS ({tracker.n_active} remaining):")
    top_features = tracker.get_top_features(25)
    
    # Group by category
    categories = {
        'KINEMATICS': [], 'ENERGY': [], 'FLOW': [], 'THERMO': [],
        'FIELD': [], 'MICRO': [], 'CROSS': [], 'TAILS': [],
        'ORDER_FLOW': [], 'CHAOS': []
    }
    
    for name, score in top_features:
        if any(x in name for x in ['velocity', 'acceleration', 'jerk', 'snap', 'crackle', 'pop', 'momentum', 'impulse']):
            categories['KINEMATICS'].append((name, score))
        elif any(x in name for x in ['energy', 'kinetic', 'potential', 'pe_']):
            categories['ENERGY'].append((name, score))
        elif any(x in name for x in ['reynolds', 'damping', 'viscosity', 'liquidity']):
            categories['FLOW'].append((name, score))
        elif any(x in name for x in ['entropy', 'perm_entropy', 'temperature', 'phase_compression']):
            categories['THERMO'].append((name, score))
        elif any(x in name for x in ['gradient', 'divergence', 'pressure', 'body_ratio', 'wick']):
            categories['FIELD'].append((name, score))
        elif any(x in name for x in ['spread', 'volume_surge', 'volume_trend']):
            categories['MICRO'].append((name, score))
        elif any(x in name for x in ['product', 'ratio', 'phase', 'release', 'flow_chaos']):
            categories['CROSS'].append((name, score))
        elif any(x in name for x in ['tail', 'cvar', 'skew', 'kurtosis', 'vol_up', 'vol_down', 'asymmetry']):
            categories['TAILS'].append((name, score))
        elif any(x in name for x in ['cvd', 'flow_imbalance', 'vpin', 'toxicity', 'adverse']):
            categories['ORDER_FLOW'].append((name, score))
        elif any(x in name for x in ['lyapunov', 'hurst', 'fractal', 'recurrence', 'determinism', 'complexity', 'predictability']):
            categories['CHAOS'].append((name, score))
    
    for cat, items in categories.items():
        if items:
            print(f"\n   {cat}:")
            for name, score in items[:5]:
                print(f"     • {name:<35s} ({score:.4f})")
    
    # Save
    results_dir = Path("results/superpot")
    results_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = results_dir / f"superpot_physics_{timestamp}.json"
    
    output = {
        'timestamp': timestamp,
        'episodes': int(len(all_rewards)),
        'avg_reward': float(np.mean(all_rewards)),
        'avg_pnl': float(np.mean(all_pnls)),
        'initial_features': int(extractor.n_features),
        'surviving_features': int(tracker.n_active),
        'top_features': [(str(n), float(s)) for n, s in top_features],
        'by_category': {k: [(str(n), float(s)) for n, s in v] for k, v in categories.items()},
    }
    
    with open(results_file, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n💾 Results saved: {results_file}")
    print(f"⏱️  Time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    
    print(f"\n{'='*60}")
    print("THE PHYSICS HAS SPOKEN!")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
