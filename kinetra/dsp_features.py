"""
DSP-Driven Feature Extraction (Assumption-Free)
================================================

NO fixed periods. NO linear assumptions. NO symmetry assumptions.
All scales derived from data using wavelets and adaptive methods.

Features:
- Continuous Wavelet Transform (CWT) energy/skew/kurt per scale
- Hilbert transform for instantaneous amplitude/frequency
- Sample entropy for complexity measurement
- Permutation entropy for order-pattern detection
- Hurst exponent for persistence/anti-persistence
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from scipy import signal
from scipy.stats import skew, kurtosis
from dataclasses import dataclass
import pywt  # PyWavelets for CWT (replaces deprecated scipy.signal.cwt)


@dataclass
class DSPFeatures:
    """Container for DSP-extracted features."""
    wavelet_energy: Dict[int, float]  # Scale -> energy
    wavelet_skew: Dict[int, float]    # Scale -> skewness
    wavelet_kurt: Dict[int, float]    # Scale -> kurtosis
    hilbert_amplitude: float
    hilbert_frequency: float
    sample_entropy: float
    permutation_entropy: float
    hurst_exponent: float
    dominant_scale: int
    energy_concentration: float  # How concentrated vs dispersed


class WaveletExtractor:
    """
    Continuous Wavelet Transform using scipy (Ricker/Mexican hat).
    No fixed periods - scales are data-adaptive.
    """

    def __init__(self, min_scale: int = 2, max_scale: int = 64, num_scales: int = 16):
        """
        Args:
            min_scale: Minimum wavelet scale (high frequency)
            max_scale: Maximum wavelet scale (low frequency)
            num_scales: Number of scales to compute
        """
        self.scales = np.logspace(
            np.log10(min_scale),
            np.log10(max_scale),
            num_scales
        ).astype(int)
        self.scales = np.unique(self.scales)  # Remove duplicates

    def compute_cwt(self, data: np.ndarray) -> np.ndarray:
        """
        Compute CWT using Mexican hat wavelet via PyWavelets.
        Returns: 2D array [scales x time], scales array
        """
        if len(data) < self.scales[-1] * 2:
            # Data too short, use smaller scales
            max_scale = len(data) // 4
            scales = np.arange(2, max(3, max_scale))
        else:
            scales = self.scales

        # Use PyWavelets for CWT (scipy.signal.cwt is deprecated)
        # 'mexh' = Mexican hat wavelet (same as Ricker)
        cwt_matrix, _ = pywt.cwt(data, scales, 'mexh')
        return cwt_matrix, scales

    def extract_features(self, data: np.ndarray, lookback: int = 50) -> Dict:
        """
        Extract wavelet features from recent data.

        Args:
            data: Full time series
            lookback: How many recent bars to analyze

        Returns:
            Dictionary of wavelet features per scale
        """
        if len(data) < lookback:
            lookback = len(data)

        recent = data[-lookback:]
        cwt_matrix, scales = self.compute_cwt(recent)

        features = {
            'energy': {},
            'skew': {},
            'kurt': {},
            'dominant_scale': 0,
            'energy_concentration': 0.0
        }

        total_energy = 0
        max_energy = 0
        max_scale = scales[0]

        for i, scale in enumerate(scales):
            coeffs = cwt_matrix[i, :]
            energy = np.sum(coeffs ** 2)
            features['energy'][int(scale)] = energy
            features['skew'][int(scale)] = float(skew(coeffs))
            features['kurt'][int(scale)] = float(kurtosis(coeffs))

            total_energy += energy
            if energy > max_energy:
                max_energy = energy
                max_scale = scale

        features['dominant_scale'] = int(max_scale)

        # Energy concentration: how much in dominant vs spread
        if total_energy > 0:
            features['energy_concentration'] = max_energy / total_energy

        return features


class HilbertExtractor:
    """
    Hilbert transform for instantaneous amplitude and frequency.
    Captures non-stationary dynamics without window assumptions.
    """

    @staticmethod
    def compute_analytic_signal(data: np.ndarray) -> np.ndarray:
        """Compute analytic signal via Hilbert transform."""
        return signal.hilbert(data)

    @staticmethod
    def extract_features(data: np.ndarray, lookback: int = 20) -> Dict:
        """
        Extract instantaneous amplitude and frequency.

        Returns:
            - amplitude: Envelope magnitude
            - frequency: Instantaneous frequency (derivative of phase)
            - phase: Current phase angle
        """
        if len(data) < 10:
            return {'amplitude': 0.0, 'frequency': 0.0, 'phase': 0.0}

        recent = data[-lookback:] if len(data) >= lookback else data

        # Detrend for cleaner analysis
        detrended = recent - np.mean(recent)

        analytic = signal.hilbert(detrended)
        amplitude = np.abs(analytic)
        phase = np.unwrap(np.angle(analytic))

        # Instantaneous frequency = d(phase)/dt
        if len(phase) > 1:
            inst_freq = np.diff(phase) / (2 * np.pi)
            current_freq = np.mean(inst_freq[-5:]) if len(inst_freq) >= 5 else np.mean(inst_freq)
        else:
            current_freq = 0.0

        return {
            'amplitude': float(amplitude[-1]),
            'frequency': float(abs(current_freq)),
            'phase': float(phase[-1] % (2 * np.pi)),
            'amplitude_mean': float(np.mean(amplitude)),
            'amplitude_std': float(np.std(amplitude)),
            'frequency_variability': float(np.std(inst_freq)) if len(phase) > 1 else 0.0
        }


class EntropyExtractor:
    """
    Entropy-based complexity measures.
    Captures disorder/predictability without assumptions.
    """

    @staticmethod
    def sample_entropy(data: np.ndarray, m: int = 2, r: float = 0.2) -> float:
        """
        Sample Entropy - measures complexity/predictability.
        Lower = more regular, Higher = more complex/random.

        Args:
            data: Time series
            m: Embedding dimension
            r: Tolerance (as fraction of std)
        """
        n = len(data)
        if n < m + 2:
            return 0.0

        # Normalize
        std = np.std(data)
        if std == 0:
            return 0.0

        tolerance = r * std

        def count_matches(template_length):
            count = 0
            templates = np.array([data[i:i+template_length] for i in range(n - template_length)])
            for i in range(len(templates)):
                for j in range(i + 1, len(templates)):
                    if np.max(np.abs(templates[i] - templates[j])) < tolerance:
                        count += 1
            return count

        # Count for m and m+1
        a = count_matches(m)
        b = count_matches(m + 1)

        if a == 0 or b == 0:
            return 0.0

        return -np.log(b / a)

    @staticmethod
    def permutation_entropy(data: np.ndarray, order: int = 3, delay: int = 1) -> float:
        """
        Permutation Entropy - order-pattern based complexity.
        Robust to noise, captures structural complexity.

        Args:
            data: Time series
            order: Permutation order (pattern length)
            delay: Time delay between points
        """
        n = len(data)
        if n < order * delay:
            return 0.0

        # Extract ordinal patterns
        from itertools import permutations
        import math

        patterns = {}
        n_patterns = 0

        for i in range(n - (order - 1) * delay):
            # Extract pattern
            indices = [i + j * delay for j in range(order)]
            values = data[indices]
            # Get rank pattern
            pattern = tuple(np.argsort(np.argsort(values)))
            patterns[pattern] = patterns.get(pattern, 0) + 1
            n_patterns += 1

        if n_patterns == 0:
            return 0.0

        # Calculate entropy
        entropy = 0.0
        for count in patterns.values():
            p = count / n_patterns
            if p > 0:
                entropy -= p * np.log2(p)

        # Normalize by maximum possible entropy
        max_entropy = np.log2(math.factorial(order))
        if max_entropy > 0:
            entropy /= max_entropy

        return entropy

    @staticmethod
    def extract_features(data: np.ndarray, lookback: int = 100) -> Dict:
        """Extract all entropy features."""
        if len(data) < lookback:
            lookback = len(data)

        recent = data[-lookback:]

        return {
            'sample_entropy': EntropyExtractor.sample_entropy(recent),
            'permutation_entropy': EntropyExtractor.permutation_entropy(recent),
            'permutation_entropy_short': EntropyExtractor.permutation_entropy(recent[-30:] if len(recent) >= 30 else recent)
        }


class DirectionalWaveletExtractor:
    """
    ASYMMETRIC wavelet analysis.
    Separates positive and negative coefficients - NEVER squares them together.

    REPLACES Hurst exponent which has fatal flaws:
    - Linear regression assumption
    - Drifts to 0.5 on short timeframes
    - Assumes stationarity
    """

    @staticmethod
    def extract_directional_features(cwt_coeffs: np.ndarray, scale: int) -> Dict:
        """
        Extract features from wavelet coefficients ASYMMETRICALLY.
        Positive and negative coefficients analyzed separately.
        """
        positive = cwt_coeffs[cwt_coeffs > 0]
        negative = cwt_coeffs[cwt_coeffs < 0]

        features = {
            # Counts (non-parametric)
            'pos_count': len(positive),
            'neg_count': len(negative),
            'pos_ratio': len(positive) / max(len(cwt_coeffs), 1),

            # Sums (directional energy, NOT squared)
            'pos_sum': np.sum(positive),
            'neg_sum': np.sum(negative),  # Negative number
            'net_energy': np.sum(positive) + np.sum(negative),

            # Medians (robust, separate)
            'pos_median': np.median(positive) if len(positive) > 0 else 0.0,
            'neg_median': np.median(negative) if len(negative) > 0 else 0.0,

            # Max extremes (separate)
            'pos_max': np.max(positive) if len(positive) > 0 else 0.0,
            'neg_min': np.min(negative) if len(negative) > 0 else 0.0,
        }

        return features

    @staticmethod
    def compute_directional_persistence(data: np.ndarray, lookback: int = 100) -> Dict:
        """
        Measure directional persistence without Hurst's flaws.

        Uses: Ratio of consecutive same-sign moves to opposite-sign moves.
        This is non-linear, non-parametric, and asymmetric.
        """
        if len(data) < lookback:
            lookback = len(data)

        recent = data[-lookback:]
        returns = np.diff(recent)

        if len(returns) < 10:
            return {
                'up_persistence': 0.5,
                'down_persistence': 0.5,
                'overall_persistence': 0.5
            }

        # Count consecutive same-direction moves
        up_continues = 0  # Up followed by up
        up_reverses = 0   # Up followed by down
        down_continues = 0  # Down followed by down
        down_reverses = 0   # Down followed by up

        for i in range(len(returns) - 1):
            curr_up = returns[i] > 0
            next_up = returns[i + 1] > 0

            if curr_up:
                if next_up:
                    up_continues += 1
                else:
                    up_reverses += 1
            else:
                if not next_up:
                    down_continues += 1
                else:
                    down_reverses += 1

        # Persistence ratios (separate for up/down)
        up_persist = up_continues / max(up_continues + up_reverses, 1)
        down_persist = down_continues / max(down_continues + down_reverses, 1)

        total_continues = up_continues + down_continues
        total_reverses = up_reverses + down_reverses
        overall = total_continues / max(total_continues + total_reverses, 1)

        return {
            'up_persistence': up_persist,
            'down_persistence': down_persist,
            'overall_persistence': overall,
            'persistence_asymmetry': up_persist - down_persist,  # Positive = up trends stronger
        }


class DSPFeatureEngine:
    """
    Master engine combining all DSP feature extractors.
    Produces assumption-free, scale-adaptive features.
    """

    def __init__(self):
        self.wavelet = WaveletExtractor()
        self.hilbert = HilbertExtractor()
        self.entropy = EntropyExtractor()
        self.directional = DirectionalWaveletExtractor()

    def extract_all(self, prices: pd.DataFrame, bar_idx: int = -1) -> Dict:
        """
        Extract all DSP features at given bar.

        Args:
            prices: DataFrame with OHLCV data
            bar_idx: Bar index to analyze (default: latest)

        Returns:
            Dictionary of all DSP features
        """
        if bar_idx < 0:
            bar_idx = len(prices) + bar_idx

        # Get log returns for analysis
        close = prices['close'].values[:bar_idx + 1]
        if len(close) < 10:
            return self._empty_features()

        log_returns = np.diff(np.log(close))
        if len(log_returns) < 5:
            return self._empty_features()

        # Also analyze range for volatility structure
        high = prices['high'].values[:bar_idx + 1]
        low = prices['low'].values[:bar_idx + 1]
        true_range = high - low

        features = {}

        # Wavelet features on returns
        wavelet_ret = self.wavelet.extract_features(log_returns)
        for scale, energy in wavelet_ret['energy'].items():
            features[f'wavelet_energy_s{scale}'] = energy
            features[f'wavelet_skew_s{scale}'] = wavelet_ret['skew'].get(scale, 0.0)
            features[f'wavelet_kurt_s{scale}'] = wavelet_ret['kurt'].get(scale, 0.0)
        features['wavelet_dominant_scale'] = wavelet_ret['dominant_scale']
        features['wavelet_energy_concentration'] = wavelet_ret['energy_concentration']

        # Wavelet on range (volatility structure)
        wavelet_range = self.wavelet.extract_features(true_range)
        features['range_dominant_scale'] = wavelet_range['dominant_scale']
        features['range_energy_concentration'] = wavelet_range['energy_concentration']

        # Hilbert features
        hilbert_ret = self.hilbert.extract_features(log_returns)
        features['hilbert_amplitude'] = hilbert_ret['amplitude']
        features['hilbert_frequency'] = hilbert_ret['frequency']
        features['hilbert_phase'] = hilbert_ret['phase']
        features['hilbert_amp_mean'] = hilbert_ret['amplitude_mean']
        features['hilbert_amp_std'] = hilbert_ret['amplitude_std']
        features['hilbert_freq_variability'] = hilbert_ret['frequency_variability']

        # Entropy features
        entropy_feats = self.entropy.extract_features(log_returns)
        features['sample_entropy'] = entropy_feats['sample_entropy']
        features['permutation_entropy'] = entropy_feats['permutation_entropy']
        features['perm_entropy_short'] = entropy_feats['permutation_entropy_short']

        # Directional persistence (REPLACES Hurst - no linear/stationarity assumptions)
        persist_feats = self.directional.compute_directional_persistence(close)
        features['up_persistence'] = persist_feats['up_persistence']
        features['down_persistence'] = persist_feats['down_persistence']
        features['overall_persistence'] = persist_feats['overall_persistence']
        features['persistence_asymmetry'] = persist_feats['persistence_asymmetry']
        # NOTE: Removed persistence_strength - it reintroduced symmetric reference point (0.5)
        # Use raw up/down persistence and asymmetry instead

        return features

    def _empty_features(self) -> Dict:
        """Return empty features for insufficient data."""
        return {
            'wavelet_dominant_scale': 0,
            'wavelet_energy_concentration': 0.0,
            'range_dominant_scale': 0,
            'range_energy_concentration': 0.0,
            'hilbert_amplitude': 0.0,
            'hilbert_frequency': 0.0,
            'hilbert_phase': 0.0,
            'hilbert_amp_mean': 0.0,
            'hilbert_amp_std': 0.0,
            'hilbert_freq_variability': 0.0,
            'sample_entropy': 0.0,
            'permutation_entropy': 0.0,
            'perm_entropy_short': 0.0,
            # Directional persistence (replaces Hurst)
            'up_persistence': 0.5,
            'down_persistence': 0.5,
            'overall_persistence': 0.5,
            'persistence_asymmetry': 0.0
        }


# Convenience function
def extract_dsp_features(prices: pd.DataFrame, bar_idx: int = -1) -> Dict:
    """Quick extraction of all DSP features."""
    engine = DSPFeatureEngine()
    return engine.extract_all(prices, bar_idx)
