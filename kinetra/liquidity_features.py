"""
Liquidity Proxy Features (Asymmetric, Non-Linear)
==================================================

Order flow and liquidity proxies derived from OHLCV data.
These capture asymmetric buying/selling pressure and liquidity stress
without requiring Level 2 order book data.

Key Features:
- CVD (Cumulative Volume Delta) - buy/sell imbalance
- Amihud illiquidity - price impact per volume
- Signed volume imbalance - directional pressure
- Range/volume impact - liquidity thinning
- VPIN-like toxicity proxy
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class LiquidityFeatures:
    """Container for liquidity-related features."""
    cvd: float                    # Cumulative volume delta
    cvd_delta: float              # Recent CVD change
    amihud: float                 # Illiquidity measure
    signed_volume: float          # Current bar's directional volume
    range_impact: float           # Price move per unit volume
    volume_imbalance: float       # Normalized buy-sell imbalance
    is_liquidity_thin: bool       # Liquidity stress flag


class CVDExtractor:
    """
    Cumulative Volume Delta (CVD) - Order Flow Imbalance.

    Estimates buy vs sell volume from OHLCV:
    - Close near high = buyers dominant
    - Close near low = sellers dominant

    This is asymmetric by design - captures directional pressure.
    """

    @staticmethod
    def compute_bar_delta(open_: float, high: float, low: float,
                          close: float, volume: float) -> float:
        """
        Compute signed volume delta for a single bar.

        Uses close location value (CLV) to estimate buy/sell ratio.
        CLV = (close - low) / (high - low) maps to [-1, +1] range.
        """
        range_ = high - low
        if range_ == 0:
            return 0.0

        # Close location: 0 = at low, 1 = at high
        clv = (close - low) / range_

        # Convert to [-1, +1]: -1 = all selling, +1 = all buying
        direction = 2 * clv - 1

        return direction * volume

    @staticmethod
    def compute_cvd(prices: pd.DataFrame) -> np.ndarray:
        """
        Compute cumulative volume delta for entire series.

        Returns:
            Array of CVD values (running sum of deltas)
        """
        deltas = np.zeros(len(prices))

        for i in range(len(prices)):
            deltas[i] = CVDExtractor.compute_bar_delta(
                prices['open'].iloc[i],
                prices['high'].iloc[i],
                prices['low'].iloc[i],
                prices['close'].iloc[i],
                prices['tickvol'].iloc[i] if 'tickvol' in prices.columns else 1.0
            )

        return np.cumsum(deltas)

    @staticmethod
    def extract_features(prices: pd.DataFrame, bar_idx: int = -1,
                         lookback: int = 50) -> Dict:
        """
        Extract CVD-related features at given bar.

        Returns:
            cvd: Current cumulative level
            cvd_delta: Change over lookback period
            cvd_acceleration: Rate of change of delta
            cvd_divergence: Price vs CVD divergence
        """
        if bar_idx < 0:
            bar_idx = len(prices) + bar_idx

        if bar_idx < 10:
            return {
                'cvd': 0.0,
                'cvd_delta': 0.0,
                'cvd_acceleration': 0.0,
                'cvd_divergence': 0.0,
                'signed_volume': 0.0
            }

        cvd = CVDExtractor.compute_cvd(prices.iloc[:bar_idx + 1])

        current_cvd = cvd[-1]
        lookback_start = max(0, len(cvd) - lookback)

        # CVD delta (recent change)
        cvd_delta = current_cvd - cvd[lookback_start]

        # CVD acceleration (delta of delta)
        if len(cvd) > lookback:
            mid_point = lookback_start + lookback // 2
            first_half_delta = cvd[mid_point] - cvd[lookback_start]
            second_half_delta = current_cvd - cvd[mid_point]
            cvd_acceleration = second_half_delta - first_half_delta
        else:
            cvd_acceleration = 0.0

        # Price-CVD divergence (bearish divergence = price up, CVD down)
        price_change = prices['close'].iloc[bar_idx] - prices['close'].iloc[lookback_start]
        if abs(cvd_delta) > 0 and abs(price_change) > 0:
            # Normalize to same scale
            cvd_norm = cvd_delta / (abs(cvd_delta) + 1e-10)
            price_norm = price_change / (abs(price_change) + 1e-10)
            divergence = price_norm - cvd_norm  # Positive = bearish divergence
        else:
            divergence = 0.0

        # Current bar's signed volume
        signed_vol = CVDExtractor.compute_bar_delta(
            prices['open'].iloc[bar_idx],
            prices['high'].iloc[bar_idx],
            prices['low'].iloc[bar_idx],
            prices['close'].iloc[bar_idx],
            prices['tickvol'].iloc[bar_idx] if 'tickvol' in prices.columns else 1.0
        )

        return {
            'cvd': current_cvd,
            'cvd_delta': cvd_delta,
            'cvd_acceleration': cvd_acceleration,
            'cvd_divergence': divergence,
            'signed_volume': signed_vol
        }


class AmihudExtractor:
    """
    Amihud Illiquidity Measure.

    Illiquidity = |return| / volume

    High values = low liquidity (big price impact per trade)
    Low values = high liquidity (absorbs volume easily)

    Spikes often precede volatile moves (liquidity thinning).
    """

    @staticmethod
    def compute_amihud(returns: np.ndarray, volumes: np.ndarray) -> np.ndarray:
        """
        Compute Amihud illiquidity for each bar.

        Args:
            returns: Log returns array
            volumes: Volume array

        Returns:
            Array of illiquidity values
        """
        # Avoid division by zero
        safe_volumes = np.maximum(volumes, 1)
        return np.abs(returns) / safe_volumes

    @staticmethod
    def extract_features(prices: pd.DataFrame, bar_idx: int = -1,
                         lookback: int = 50) -> Dict:
        """
        Extract Amihud-related features.

        Returns:
            amihud: Current bar illiquidity
            amihud_mean: Rolling average
            amihud_percentile: Current vs recent distribution
            is_liquidity_thin: Above 90th percentile
        """
        if bar_idx < 0:
            bar_idx = len(prices) + bar_idx

        if bar_idx < 10:
            return {
                'amihud': 0.0,
                'amihud_mean': 0.0,
                'amihud_std': 0.0,
                'amihud_percentile': 0.5,
                'is_liquidity_thin': False
            }

        # Compute log returns
        close = prices['close'].values[:bar_idx + 1]
        log_returns = np.diff(np.log(close))

        # Get volumes
        if 'tickvol' in prices.columns:
            volumes = prices['tickvol'].values[1:bar_idx + 1]
        else:
            volumes = np.ones(len(log_returns))

        amihud = AmihudExtractor.compute_amihud(log_returns, volumes)

        current = amihud[-1] if len(amihud) > 0 else 0.0
        recent = amihud[-lookback:] if len(amihud) >= lookback else amihud

        mean_amihud = np.mean(recent) if len(recent) > 0 else 0.0
        std_amihud = np.std(recent) if len(recent) > 1 else 0.0

        # Percentile in recent distribution
        if len(recent) > 0:
            percentile = np.sum(recent <= current) / len(recent)
        else:
            percentile = 0.5

        return {
            'amihud': current,
            'amihud_mean': mean_amihud,
            'amihud_std': std_amihud,
            'amihud_percentile': percentile,
            'is_liquidity_thin': percentile > 0.9
        }


class RangeImpactExtractor:
    """
    Range/Volume Impact - How much price moves per unit volume.

    High impact = thin liquidity, explosive potential
    Low impact = deep liquidity, absorbing flow
    """

    @staticmethod
    def compute_range_impact(ranges: np.ndarray, volumes: np.ndarray) -> np.ndarray:
        """Compute range/volume impact ratio."""
        safe_volumes = np.maximum(volumes, 1)
        return ranges / safe_volumes

    @staticmethod
    def extract_features(prices: pd.DataFrame, bar_idx: int = -1,
                         lookback: int = 50) -> Dict:
        """
        Extract range impact features.
        """
        if bar_idx < 0:
            bar_idx = len(prices) + bar_idx

        if bar_idx < 10:
            return {
                'range_impact': 0.0,
                'range_impact_percentile': 0.5,
                'normalized_range': 0.0
            }

        ranges = (prices['high'] - prices['low']).values[:bar_idx + 1]

        if 'tickvol' in prices.columns:
            volumes = prices['tickvol'].values[:bar_idx + 1]
        else:
            volumes = np.ones(len(ranges))

        impacts = RangeImpactExtractor.compute_range_impact(ranges, volumes)

        current = impacts[-1]
        recent = impacts[-lookback:] if len(impacts) >= lookback else impacts

        # Percentile
        percentile = np.sum(recent <= current) / len(recent) if len(recent) > 0 else 0.5

        # Normalized range (vs median)
        median_range = np.median(recent) if len(recent) > 0 else 1.0
        normalized = ranges[-1] / median_range if median_range > 0 else 1.0

        return {
            'range_impact': current,
            'range_impact_percentile': percentile,
            'normalized_range': normalized,
            'is_fat_candle': normalized > 3.0  # 3x median range
        }


class VolumeImbalanceExtractor:
    """
    Volume Imbalance - Normalized buy/sell pressure.

    Different from CVD in that it's normalized per bar,
    making it comparable across different volume regimes.
    """

    @staticmethod
    def compute_imbalance(prices: pd.DataFrame) -> np.ndarray:
        """
        Compute normalized volume imbalance for each bar.

        Returns values in [-1, +1]:
        -1 = 100% selling pressure
        +1 = 100% buying pressure
        """
        imbalance = np.zeros(len(prices))

        for i in range(len(prices)):
            high = prices['high'].iloc[i]
            low = prices['low'].iloc[i]
            close = prices['close'].iloc[i]

            range_ = high - low
            if range_ == 0:
                imbalance[i] = 0.0
            else:
                # CLV as imbalance proxy
                clv = (close - low) / range_
                imbalance[i] = 2 * clv - 1

        return imbalance

    @staticmethod
    def extract_features(prices: pd.DataFrame, bar_idx: int = -1,
                         lookback: int = 20) -> Dict:
        """
        Extract volume imbalance features.
        """
        if bar_idx < 0:
            bar_idx = len(prices) + bar_idx

        if bar_idx < 5:
            return {
                'volume_imbalance': 0.0,
                'imbalance_streak': 0,
                'imbalance_skew': 0.0
            }

        imbalance = VolumeImbalanceExtractor.compute_imbalance(
            prices.iloc[:bar_idx + 1]
        )

        current = imbalance[-1]
        recent = imbalance[-lookback:] if len(imbalance) >= lookback else imbalance

        # Streak: consecutive same-sign imbalance
        streak = 0
        sign = np.sign(current)
        for val in reversed(imbalance):
            if np.sign(val) == sign:
                streak += 1
            else:
                break
        streak = int(streak * sign)  # Negative for selling streak

        # Skewness of recent imbalance
        from scipy.stats import skew
        skewness = skew(recent) if len(recent) > 3 else 0.0

        return {
            'volume_imbalance': current,
            'imbalance_streak': streak,
            'imbalance_skew': skewness,
            'imbalance_mean': np.mean(recent)
        }


class LiquidityFeatureEngine:
    """
    Master engine combining all liquidity feature extractors.
    """

    def __init__(self):
        self.cvd = CVDExtractor()
        self.amihud = AmihudExtractor()
        self.range_impact = RangeImpactExtractor()
        self.volume_imbalance = VolumeImbalanceExtractor()

    def extract_all(self, prices: pd.DataFrame, bar_idx: int = -1) -> Dict:
        """
        Extract all liquidity features at given bar.
        """
        features = {}

        # CVD features
        cvd_feats = self.cvd.extract_features(prices, bar_idx)
        features.update({f'cvd_{k}' if not k.startswith('cvd') else k: v
                        for k, v in cvd_feats.items()})

        # Amihud features
        amihud_feats = self.amihud.extract_features(prices, bar_idx)
        features.update(amihud_feats)

        # Range impact features
        impact_feats = self.range_impact.extract_features(prices, bar_idx)
        features.update(impact_feats)

        # Volume imbalance features
        imbal_feats = self.volume_imbalance.extract_features(prices, bar_idx)
        features.update(imbal_feats)

        return features


# Convenience function
def extract_liquidity_features(prices: pd.DataFrame, bar_idx: int = -1) -> Dict:
    """Quick extraction of all liquidity features."""
    engine = LiquidityFeatureEngine()
    return engine.extract_all(prices, bar_idx)
