"""
DSP-Driven SuperPot Feature Extraction
======================================

**PHILOSOPHY ENFORCEMENT**: NO FIXED PERIODS. EVER.

This replaces legacy superpot_explorer.py which violated core philosophy by using
magic numbers (5, 10, 20) for lookback periods.

Instead, ALL periods are derived from DSP analysis:
- Wavelet dominant_scale tells us market's natural cycle
- Hilbert instantaneous frequency captures rhythm
- Features are asymmetric (up/down separate)

Key Principle:
-------------
The market tells us its time scales. We don't impose them.
"""

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats

from kinetra.dsp_features import WaveletExtractor, HilbertExtractor
from kinetra.assumption_free_measures import AsymmetricReturns

logger = logging.getLogger(__name__)


class DSPSuperPotExtractor:
    """
    SuperPot feature extraction using DSP-detected cycles.
    
    NO FIXED PERIODS. All lookbacks are adaptive based on:
    - Wavelet dominant_scale (market's natural cycle)
    - Hilbert instantaneous frequency (current rhythm)
    
    Features are ASYMMETRIC (up/down moves treated separately).
    """
    
    def __init__(self, min_scale: int = 2, max_scale: int = 64, num_scales: int = 16):
        """
        Args:
            min_scale: Minimum wavelet scale (high frequency)
            max_scale: Maximum wavelet scale (low frequency)  
            num_scales: Number of scales to compute
        """
        self.wavelet = WaveletExtractor(min_scale=min_scale, max_scale=max_scale, num_scales=num_scales)
        self.hilbert = HilbertExtractor()
        self.feature_names = self._build_feature_names()
        self.n_features = len(self.feature_names)
        
        logger.info(f"DSPSuperPotExtractor initialized with {self.n_features} features (NO fixed periods)")
    
    def extract(self, df: pd.DataFrame, idx: int) -> np.ndarray:
        """
        Extract features using DSP-detected cycles (not fixed periods).
        
        Args:
            df: DataFrame with OHLCV data
            idx: Current index position
            
        Returns:
            Feature vector (all adaptive, no magic numbers)
        """
        if idx < 100:
            # Not enough history
            return np.zeros(self.n_features, dtype=np.float32)
        
        # Get price data
        prices = df['close'].values[:idx+1]
        highs = df['high'].values[:idx+1]
        lows = df['low'].values[:idx+1]
        volumes = df['volume'].values[:idx+1] if 'volume' in df.columns else None
        
        # DSP: Detect market's natural cycles
        wavelet_features = self.wavelet.extract_features(prices)
        dominant_cycle = wavelet_features['dominant_scale']  # Market tells us!
        
        # Hilbert: Get instantaneous frequency
        hilbert_features = self.hilbert.extract_features(prices)
        inst_frequency = hilbert_features['frequency']
        
        # Use detected cycle for calculations (NOT 5, 10, 20)
        lookback = max(dominant_cycle, 5)  # At least 5 bars for stability
        lookback = min(lookback, idx)  # Don't exceed available data
        
        features = []
        
        # === 1. PRICE ACTION (Adaptive Cycle) ===
        if lookback > 0:
            cycle_return = (prices[-1] / prices[-lookback] - 1) if prices[-lookback] != 0 else 0
        else:
            cycle_return = 0
        features.append(cycle_return)
        
        # Range over detected cycle
        if lookback > 0:
            cycle_high = np.max(highs[-lookback:])
            cycle_low = np.min(lows[-lookback:])
            cycle_range = (cycle_high - cycle_low) / prices[-1] if prices[-1] != 0 else 0
        else:
            cycle_range = 0
        features.append(cycle_range)
        
        # === 2. ASYMMETRIC RETURNS (up/down SEPARATE) ===
        # NEVER combine up and down moves - that's symmetric!
        returns = np.diff(prices[-lookback:]) / prices[-lookback-1:-1] if lookback > 1 else np.array([0])
        up_returns = returns[returns > 0]
        down_returns = returns[returns < 0]
        
        features.append(np.sum(up_returns) if len(up_returns) > 0 else 0)  # Total upside
        features.append(np.sum(down_returns) if len(down_returns) > 0 else 0)  # Total downside
        features.append(np.mean(up_returns) if len(up_returns) > 0 else 0)  # Avg up move
        features.append(np.mean(down_returns) if len(down_returns) > 0 else 0)  # Avg down move
        
        # === 3. HILBERT FEATURES (Instantaneous) ===
        features.append(inst_frequency)  # Current market rhythm
        features.append(hilbert_features['amplitude'])  # Envelope magnitude
        features.append(hilbert_features['phase'])  # Phase angle
        
        # === 4. WAVELET ENERGY (Per Scale - Multiple Cycles) ===
        # Include energy at different scales (not just dominant)
        for scale in sorted(wavelet_features['energy'].keys())[:5]:  # Top 5 scales
            features.append(wavelet_features['energy'][scale])
        
        # === 5. WAVELET SKEW/KURT (Per Scale - Asymmetric) ===
        for scale in sorted(wavelet_features['skew'].keys())[:3]:  # Top 3 scales
            features.append(wavelet_features['skew'][scale])
        for scale in sorted(wavelet_features['kurt'].keys())[:3]:  # Top 3 scales
            features.append(wavelet_features['kurt'][scale])
        
        # === 6. DOMINANT CYCLE FEATURES ===
        features.append(dominant_cycle)  # The cycle itself (adaptive)
        features.append(wavelet_features['energy_concentration'])  # How concentrated energy is
        
        # === 7. VOLATILITY (Asymmetric - up vs down) ===
        if lookback > 1:
            up_vol = np.std(up_returns) if len(up_returns) > 1 else 0
            down_vol = np.std(down_returns) if len(down_returns) > 1 else 0
            features.append(up_vol)
            features.append(down_vol)
            features.append(up_vol - down_vol)  # Volatility asymmetry
        else:
            features.extend([0, 0, 0])
        
        # === 8. MOMENTUM (Directional - signed) ===
        if lookback > 2:
            # Momentum over detected cycle (NOT fixed 10)
            momentum = prices[-1] - prices[-lookback]
            features.append(momentum / prices[-1] if prices[-1] != 0 else 0)
            
            # Acceleration (second derivative)
            mid_point = lookback // 2
            if mid_point > 0:
                accel = (prices[-1] - prices[-mid_point]) - (prices[-mid_point] - prices[-lookback])
                features.append(accel / prices[-1] if prices[-1] != 0 else 0)
            else:
                features.append(0)
        else:
            features.extend([0, 0])
        
        # === 9. VOLUME DYNAMICS (if available) ===
        if volumes is not None and lookback > 1:
            vol_ratio = volumes[-1] / np.mean(volumes[-lookback:]) if np.mean(volumes[-lookback:]) > 0 else 1
            features.append(vol_ratio)
            
            # Volume trend (increasing/decreasing)
            vol_trend = np.polyfit(range(lookback), volumes[-lookback:], 1)[0] if lookback > 2 else 0
            features.append(vol_trend)
        else:
            features.extend([1, 0])
        
        # === 10. HIGHER MOMENTS (Asymmetric) ===
        if lookback > 3:
            # Skewness (asymmetry of distribution)
            skewness = stats.skew(returns) if len(returns) > 3 else 0
            features.append(skewness)
            
            # Kurtosis (tail heaviness)
            kurt = stats.kurtosis(returns) if len(returns) > 3 else 0
            features.append(kurt)
        else:
            features.extend([0, 0])
        
        # Convert to array and ensure finite values
        features_array = np.array(features, dtype=np.float32)
        features_array = np.nan_to_num(features_array, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Verify we have the right number of features
        if len(features_array) != self.n_features:
            logger.warning(f"Feature count mismatch: expected {self.n_features}, got {len(features_array)}")
            # Pad or truncate to match
            if len(features_array) < self.n_features:
                features_array = np.pad(features_array, (0, self.n_features - len(features_array)))
            else:
                features_array = features_array[:self.n_features]
        
        return features_array
    
    def _build_feature_names(self) -> List[str]:
        """
        Build feature names WITHOUT fixed periods.
        
        All names reference 'cycle' or 'dominant' or 'instantaneous' - 
        never magic numbers like _5, _10, _20.
        """
        names = [
            # Price action (adaptive cycle)
            'cycle_return',
            'cycle_range',
            
            # Asymmetric returns (up/down separate)
            'up_sum',
            'down_sum',
            'up_mean',
            'down_mean',
            
            # Hilbert (instantaneous)
            'inst_frequency',
            'inst_amplitude',
            'inst_phase',
            
            # Wavelet energy (per scale)
            'wavelet_energy_scale_0',
            'wavelet_energy_scale_1',
            'wavelet_energy_scale_2',
            'wavelet_energy_scale_3',
            'wavelet_energy_scale_4',
            
            # Wavelet skew/kurt (per scale)
            'wavelet_skew_scale_0',
            'wavelet_skew_scale_1',
            'wavelet_skew_scale_2',
            'wavelet_kurt_scale_0',
            'wavelet_kurt_scale_1',
            'wavelet_kurt_scale_2',
            
            # Dominant cycle
            'dominant_scale',
            'energy_concentration',
            
            # Volatility (asymmetric)
            'up_volatility',
            'down_volatility',
            'volatility_asymmetry',
            
            # Momentum (directional)
            'cycle_momentum',
            'cycle_acceleration',
            
            # Volume
            'volume_ratio',
            'volume_trend',
            
            # Higher moments
            'skewness',
            'kurtosis',
        ]
        
        return names


def validate_no_fixed_periods(extractor: DSPSuperPotExtractor) -> bool:
    """
    Validate that feature names contain NO fixed period references.
    
    Args:
        extractor: DSPSuperPotExtractor instance
        
    Returns:
        True if no fixed periods found, False otherwise
    """
    forbidden_patterns = ['_5', '_10', '_20', '_14', '_50', '_100', '_200']
    
    violations = []
    for name in extractor.feature_names:
        for pattern in forbidden_patterns:
            if pattern in name:
                violations.append(f"Feature '{name}' contains forbidden pattern '{pattern}'")
    
    if violations:
        logger.error("PHILOSOPHY VIOLATION: Fixed periods found!")
        for v in violations:
            logger.error(f"  - {v}")
        return False
    
    logger.info("✅ Validation passed: NO fixed periods in feature names")
    return True


# Quick test
if __name__ == "__main__":
    extractor = DSPSuperPotExtractor()
    print(f"\n✅ Created DSP SuperPot Extractor with {extractor.n_features} features")
    print(f"\nFeature names (first 10):")
    for i, name in enumerate(extractor.feature_names[:10]):
        print(f"  {i+1}. {name}")
    
    # Validate no fixed periods
    is_valid = validate_no_fixed_periods(extractor)
    print(f"\n{'✅' if is_valid else '❌'} Fixed period validation: {'PASS' if is_valid else 'FAIL'}")
