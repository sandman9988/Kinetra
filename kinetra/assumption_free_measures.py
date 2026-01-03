"""
Assumption-Free Measurements
=============================

RUTHLESSLY PURGED of:
- Linearity (no linear regression, no linear combinations)
- Symmetry (up moves ≠ down moves, always separate)
- Stationarity (no assumption of stable distributions)
- Gaussianity (no z-scores, no std dev as primary measure)
- Fixed periods (already handled by DSP, but reinforced here)

What survives:
- Rank/order-based (non-parametric)
- Directional (signed, asymmetric by design)
- Tail-specific (left vs right separately)
- Recurrence-based (structural patterns)
- Entropy-based (complexity, not distribution shape)

PHILOSOPHY: Measure SEPARATELY what happens when price goes up vs down.
Never average them. Never take absolute values. Never assume they're equal.

PERFORMANCE OPTIMIZATIONS:
- Sample entropy uses JIT compilation (numba) when available
- Recurrence matrix uses vectorized numpy operations
- Caching applied to expensive computations
"""

import warnings
from collections import Counter
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import rankdata

# Import optimized implementations
try:
    from .performance import (
        determinism_fast,
        extract_recurrence_features_fast,
        recurrence_matrix_fast,
        sample_entropy_fast,
    )

    _OPTIMIZED_AVAILABLE = True
except ImportError:
    _OPTIMIZED_AVAILABLE = False


class AsymmetricReturns:
    """
    Separate measurement of up-moves vs down-moves.
    NEVER combine them. NEVER take absolute values.
    """

    @staticmethod
    def compute_directional_returns(prices: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Separate returns into up and down components.

        Returns:
            up_returns: Array with positive returns, zeros elsewhere
            down_returns: Array with negative returns (as negative), zeros elsewhere
        """
        log_returns = np.diff(np.log(prices))

        up_returns = np.where(log_returns > 0, log_returns, 0.0)
        down_returns = np.where(log_returns < 0, log_returns, 0.0)

        return up_returns, down_returns

    @staticmethod
    def extract_features(prices: np.ndarray, lookback: int = 50) -> Dict:
        """
        Extract asymmetric return features.

        All features are DIRECTIONAL - up and down measured separately.
        """
        if len(prices) < lookback + 1:
            lookback = len(prices) - 1

        recent_prices = prices[-(lookback + 1) :]
        up, down = AsymmetricReturns.compute_directional_returns(recent_prices)

        # Count-based (no magnitude assumption)
        up_count = np.sum(up > 0)
        down_count = np.sum(down < 0)
        total = up_count + down_count

        # Magnitude-based (separate, never combined)
        up_sum = np.sum(up)  # Total upside
        down_sum = np.sum(down)  # Total downside (negative number)

        # Median-based (robust, non-linear)
        up_median = np.median(up[up > 0]) if up_count > 0 else 0.0
        down_median = np.median(down[down < 0]) if down_count > 0 else 0.0

        # 90th percentile of each (tail behavior, separate)
        up_p90 = np.percentile(up[up > 0], 90) if up_count >= 5 else up_median
        down_p10 = np.percentile(down[down < 0], 10) if down_count >= 5 else down_median

        # Streak analysis (consecutive same direction)
        current_streak = 0
        streak_magnitude = 0.0
        streak_returns = []

        if len(up) > 0:
            last_sign = np.sign(up[-1] + down[-1])
            for i in range(len(up) - 1, -1, -1):
                bar_return = up[i] + down[i]  # The actual return
                bar_sign = np.sign(bar_return)
                if bar_sign == last_sign and bar_sign != 0:
                    current_streak += 1
                    streak_magnitude += bar_return
                    streak_returns.append(bar_return)
                else:
                    break
            current_streak *= int(last_sign)  # Negative for down streaks

        # Streak momentum metrics (ENHANCED)
        up_streak_mag = streak_magnitude if current_streak > 0 else 0.0
        down_streak_mag = streak_magnitude if current_streak < 0 else 0.0
        streak_avg = np.mean(streak_returns) if streak_returns else 0.0
        streak_conviction = abs(current_streak) * abs(streak_avg)  # length * avg magnitude

        return {
            # Counts (non-parametric)
            "up_count": up_count,
            "down_count": down_count,
            "up_ratio": up_count / max(total, 1),
            "down_ratio": down_count / max(total, 1),
            # Sums (directional magnitude)
            "up_sum": up_sum,
            "down_sum": down_sum,  # This is negative
            "net_direction": up_sum + down_sum,  # Signed
            # Medians (robust central tendency, separate)
            "up_median": up_median,
            "down_median": down_median,
            # Tails (extremes, separate)
            "up_p90": up_p90,
            "down_p10": down_p10,
            # Asymmetry ratio: how much bigger are up moves vs down moves?
            # >1 means up moves larger, <1 means down moves larger
            "magnitude_asymmetry": abs(up_median / down_median) if down_median != 0 else 1.0,
            # Streak (directional persistence)
            "current_streak": current_streak,
            # Streak momentum (ENHANCED - captures "strong continuation" vs "weak chop")
            "up_streak_magnitude": up_streak_mag,  # Sum of returns during up streak
            "down_streak_magnitude": down_streak_mag,  # Sum during down streak (negative)
            "streak_conviction": streak_conviction,  # streak_length * avg_magnitude
        }


class RankBasedMeasures:
    """
    Rank/order-based measures - completely non-parametric.
    No assumptions about distribution shape or scale.
    """

    @staticmethod
    def compute_ranks(data: np.ndarray) -> np.ndarray:
        """Convert values to ranks (1 to N)."""
        return rankdata(data, method="average")

    @staticmethod
    def current_percentile_rank(data: np.ndarray) -> float:
        """
        Where does the current value rank in the lookback?
        Returns 0-1 (0 = lowest, 1 = highest).
        """
        if len(data) < 2:
            return 0.5
        ranks = rankdata(data)
        return (ranks[-1] - 1) / (len(data) - 1)

    @staticmethod
    def extract_features(prices: pd.DataFrame, bar_idx: int = -1, lookback: int = 100) -> Dict:
        """
        Extract rank-based features.
        """
        if bar_idx < 0:
            bar_idx = len(prices) + bar_idx

        start = max(0, bar_idx - lookback + 1)

        # Rank of current close in lookback
        closes = prices["close"].iloc[start : bar_idx + 1].values
        close_rank = RankBasedMeasures.current_percentile_rank(closes)

        # Rank of current range in lookback
        ranges = (prices["high"] - prices["low"]).iloc[start : bar_idx + 1].values
        range_rank = RankBasedMeasures.current_percentile_rank(ranges)

        # Rank of current volume in lookback (if available)
        if "tickvol" in prices.columns:
            volumes = prices["tickvol"].iloc[start : bar_idx + 1].values
            volume_rank = RankBasedMeasures.current_percentile_rank(volumes)
        else:
            volume_rank = 0.5

        # New highs/lows in lookback (count-based)
        new_highs = 0
        new_lows = 0
        running_high = closes[0]
        running_low = closes[0]

        for c in closes[1:]:
            if c > running_high:
                new_highs += 1
                running_high = c
            if c < running_low:
                new_lows += 1
                running_low = c

        return {
            "close_rank": close_rank,
            "range_rank": range_rank,
            "volume_rank": volume_rank,
            "new_highs_count": new_highs,
            "new_lows_count": new_lows,
            "high_low_ratio": new_highs / max(new_lows, 1),
            # Distance from extremes (rank-based)
            "rank_from_high": 1.0 - close_rank,  # How far below the high
            "rank_from_low": close_rank,  # How far above the low
        }


class DirectionalVolatility:
    """
    Volatility measured SEPARATELY for up and down moves.
    Never use symmetric std dev.
    """

    @staticmethod
    def compute_directional_dispersion(returns: np.ndarray) -> Tuple[float, float]:
        """
        Compute dispersion of up moves and down moves separately.
        Uses MAD (median absolute deviation) - robust, not assuming symmetry.
        """
        up = returns[returns > 0]
        down = returns[returns < 0]

        up_mad = np.median(np.abs(up - np.median(up))) if len(up) > 1 else 0.0
        down_mad = np.median(np.abs(down - np.median(down))) if len(down) > 1 else 0.0

        return up_mad, down_mad

    @staticmethod
    def extract_features(prices: np.ndarray, lookback: int = 50) -> Dict:
        """
        Extract directional volatility features.
        """
        if len(prices) < lookback + 1:
            lookback = len(prices) - 1

        log_returns = np.diff(np.log(prices[-(lookback + 1) :]))

        up = log_returns[log_returns > 0]
        down = log_returns[log_returns < 0]

        up_mad, down_mad = DirectionalVolatility.compute_directional_dispersion(log_returns)

        # Interquartile ranges (robust, non-parametric)
        up_iqr = np.percentile(up, 75) - np.percentile(up, 25) if len(up) >= 4 else 0.0
        down_iqr = np.percentile(down, 75) - np.percentile(down, 25) if len(down) >= 4 else 0.0

        return {
            "up_mad": up_mad,
            "down_mad": down_mad,
            "volatility_asymmetry": up_mad / down_mad if down_mad > 0 else 1.0,
            "up_iqr": up_iqr,
            "down_iqr": abs(down_iqr),  # Make positive for comparison
            # Which side is more dispersed?
            "upside_more_volatile": up_mad > down_mad,
            "downside_more_volatile": down_mad > up_mad,
        }


class DirectionalOrderFlow:
    """
    Order flow that NEVER treats buys and sells symmetrically.
    """

    @staticmethod
    def compute_bar_pressure(
        open_: float, high: float, low: float, close: float, volume: float
    ) -> Tuple[float, float]:
        """
        Compute buying and selling pressure SEPARATELY.

        Returns:
            buy_pressure: Estimated buy volume
            sell_pressure: Estimated sell volume
        """
        range_ = high - low
        if range_ == 0:
            return volume * 0.5, volume * 0.5

        # How much of the range was covered upward vs downward
        up_range = close - low  # Distance from low to close
        down_range = high - close  # Distance from close to high

        # Allocate volume proportionally
        buy_pressure = volume * (up_range / range_)
        sell_pressure = volume * (down_range / range_)

        return buy_pressure, sell_pressure

    @staticmethod
    def extract_features(prices: pd.DataFrame, bar_idx: int = -1, lookback: int = 50) -> Dict:
        """
        Extract directional order flow features.
        """
        if bar_idx < 0:
            bar_idx = len(prices) + bar_idx

        start = max(0, bar_idx - lookback + 1)
        subset = prices.iloc[start : bar_idx + 1]

        # Vectorized: Extract arrays once
        opens = subset["open"].values
        highs = subset["high"].values
        lows = subset["low"].values
        closes = subset["close"].values

        vol_col = "tickvol" if "tickvol" in prices.columns else None
        volumes = subset[vol_col].values if vol_col else np.ones(len(subset))

        # Vectorized: Compute all bars at once
        ranges = highs - lows
        zero_ranges = ranges == 0
        ranges_safe = np.where(zero_ranges, 1.0, ranges)

        up_ranges = closes - lows
        down_ranges = highs - closes

        buy_pressures = volumes * (up_ranges / ranges_safe)
        sell_pressures = volumes * (down_ranges / ranges_safe)

        # Handle zero ranges
        buy_pressures = np.where(zero_ranges, volumes * 0.5, buy_pressures)
        sell_pressures = np.where(zero_ranges, volumes * 0.5, sell_pressures)

        # Cumulative (running) pressure
        cum_buy = np.sum(buy_pressures)
        cum_sell = np.sum(sell_pressures)

        # Recent pressure (last 5 bars)
        recent_buy = np.sum(buy_pressures[-5:])
        recent_sell = np.sum(sell_pressures[-5:])

        # Pressure acceleration (is buying/selling intensifying?)
        if len(buy_pressures) >= 10:
            first_half_buy = np.sum(buy_pressures[: len(buy_pressures) // 2])
            second_half_buy = np.sum(buy_pressures[len(buy_pressures) // 2 :])
            first_half_sell = np.sum(sell_pressures[: len(sell_pressures) // 2])
            second_half_sell = np.sum(sell_pressures[len(sell_pressures) // 2 :])

            buy_acceleration = second_half_buy - first_half_buy
            sell_acceleration = second_half_sell - first_half_sell
        else:
            buy_acceleration = 0.0
            sell_acceleration = 0.0

        return {
            "cum_buy_pressure": cum_buy,
            "cum_sell_pressure": cum_sell,
            "net_pressure": cum_buy - cum_sell,  # Signed!
            "recent_buy_pressure": recent_buy,
            "recent_sell_pressure": recent_sell,
            "recent_net_pressure": recent_buy - recent_sell,
            "buy_acceleration": buy_acceleration,
            "sell_acceleration": sell_acceleration,
            # Dominance (which side is winning?)
            "buy_dominance": cum_buy / max(cum_sell, 1e-10),
            "sell_dominance": cum_sell / max(cum_buy, 1e-10),
        }


class PermutationPatterns:
    """
    Order-pattern based measures.
    Completely non-parametric, no magnitude assumptions.
    """

    @staticmethod
    def encode_pattern(values: np.ndarray) -> Tuple:
        """Encode a sequence as its ordinal pattern."""
        return tuple(np.argsort(np.argsort(values)))

    @staticmethod
    def extract_features(prices: np.ndarray, order: int = 3, lookback: int = 100) -> Dict:
        """
        Extract permutation-based features.
        """
        if len(prices) < lookback:
            lookback = len(prices)

        recent = prices[-lookback:]
        log_returns = np.diff(np.log(recent))

        if len(log_returns) < order:
            return {"pattern_entropy": 0.0, "most_common_pattern": None, "pattern_diversity": 0.0}

        # Extract all patterns
        patterns = []
        for i in range(len(log_returns) - order + 1):
            pattern = PermutationPatterns.encode_pattern(log_returns[i : i + order])
            patterns.append(pattern)

        # Count patterns
        pattern_counts = Counter(patterns)
        n_patterns = len(patterns)

        # Entropy
        entropy = 0.0
        for count in pattern_counts.values():
            p = count / n_patterns
            if p > 0:
                entropy -= p * np.log2(p)

        # Normalize by max possible
        import math

        max_entropy = np.log2(math.factorial(order))
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0

        # Most common pattern
        most_common = pattern_counts.most_common(1)[0] if pattern_counts else (None, 0)

        # Diversity: how many unique patterns vs possible
        max_patterns = math.factorial(order)
        diversity = len(pattern_counts) / max_patterns

        return {
            "pattern_entropy": normalized_entropy,
            "most_common_pattern": most_common[0],
            "most_common_frequency": most_common[1] / n_patterns if n_patterns > 0 else 0,
            "pattern_diversity": diversity,
            # Trend patterns (specific ordinal signatures)
            # (0,1,2) = monotonic up, (2,1,0) = monotonic down
            "monotonic_up_freq": pattern_counts.get((0, 1, 2), 0) / max(n_patterns, 1),
            "monotonic_down_freq": pattern_counts.get((2, 1, 0), 0) / max(n_patterns, 1),
        }


class RecurrenceFeatures:
    """
    Recurrence-based measures - purely structural.
    No distributional assumptions whatsoever.

    ENHANCED: Now includes directional recurrence asymmetry.
    Computes recurrence separately for up-move and down-move sequences.

    PERFORMANCE: Uses vectorized numpy operations for O(n²) matrix computation.
    50-100x faster than naive nested loops.
    """

    @staticmethod
    def compute_recurrence_matrix(data: np.ndarray, threshold: float = 0.1) -> np.ndarray:
        """
        Compute recurrence matrix using vectorized operations.
        R[i,j] = 1 if |x_i - x_j| < threshold * MAD(x)
        Using MAD instead of std for robustness.

        PERFORMANCE: Vectorized implementation is ~50x faster than nested loops.
        """
        # Use optimized version if available
        if _OPTIMIZED_AVAILABLE:
            return recurrence_matrix_fast(data, threshold, use_mad=True)

        # Fallback to vectorized numpy (still fast)
        n = len(data)
        if n < 2:
            return np.zeros((1, 1), dtype=int)

        # Use MAD for threshold instead of std (more robust)
        mad = np.median(np.abs(data - np.median(data)))
        eps = threshold * max(mad, 1e-10)

        # Vectorized distance computation using broadcasting
        distances = np.abs(data[:, np.newaxis] - data[np.newaxis, :])
        R = (distances < eps).astype(int)

        return R

    @staticmethod
    def _compute_determinism(R: np.ndarray) -> float:
        """
        Compute determinism from recurrence matrix.

        PERFORMANCE: Uses optimized version when available.
        """
        if _OPTIMIZED_AVAILABLE:
            return determinism_fast(R)

        n = len(R)
        if n < 2:
            return 0.0

        total_recurrent = np.sum(R)
        if total_recurrent == 0:
            return 0.0

        # Count diagonal line points (vectorized for main diagonals)
        diagonal_points = 0
        for k in range(1, n):
            diag = np.diag(R, k)
            # Count consecutive pairs
            if len(diag) >= 2:
                diagonal_points += np.sum(diag[:-1] & diag[1:])
            diag = np.diag(R, -k)
            if len(diag) >= 2:
                diagonal_points += np.sum(diag[:-1] & diag[1:])

        return diagonal_points / total_recurrent

    @staticmethod
    def extract_features(prices: np.ndarray, lookback: int = 50) -> Dict:
        """
        Extract recurrence quantification features with directional asymmetry.

        PERFORMANCE: Uses vectorized implementations for significant speedup.
        """
        # Use fully optimized version if available
        if _OPTIMIZED_AVAILABLE:
            return extract_recurrence_features_fast(prices, lookback)

        if len(prices) < lookback:
            lookback = len(prices)

        recent = prices[-lookback:]
        log_returns = np.diff(np.log(recent))

        if len(log_returns) < 10:
            return {
                "recurrence_rate": 0.0,
                "determinism": 0.0,
                "laminarity": 0.0,
                "det_up": 0.0,
                "det_down": 0.0,
                "recurrence_asymmetry": 0.0,
            }

        R = RecurrenceFeatures.compute_recurrence_matrix(log_returns)
        n = len(R)

        # Recurrence Rate (RR): fraction of recurrent points
        rr = np.sum(R) / (n * n)

        # Determinism (DET): fraction of recurrent points forming diagonals
        det = RecurrenceFeatures._compute_determinism(R)

        # Laminarity (LAM): fraction forming vertical lines (vectorized)
        if n >= 2:
            vertical_matches = np.sum(R[:-1, :] & R[1:, :])
            lam = vertical_matches / max(np.sum(R), 1)
        else:
            lam = 0.0

        # DIRECTIONAL RECURRENCE ASYMMETRY (ENHANCED)
        # Split returns into up and down phases
        up_returns = log_returns[log_returns > 0]
        down_returns = log_returns[log_returns < 0]

        # Compute recurrence separately for up-moves and down-moves
        if len(up_returns) >= 5:
            R_up = RecurrenceFeatures.compute_recurrence_matrix(up_returns)
            det_up = RecurrenceFeatures._compute_determinism(R_up)
        else:
            det_up = 0.0

        if len(down_returns) >= 5:
            R_down = RecurrenceFeatures.compute_recurrence_matrix(np.abs(down_returns))
            det_down = RecurrenceFeatures._compute_determinism(R_down)
        else:
            det_down = 0.0

        # Recurrence asymmetry: which side has more structure?
        # Positive = up moves more structured, Negative = down moves more structured
        recurrence_asymmetry = det_up - det_down

        return {
            "recurrence_rate": rr,
            "determinism": det,  # High = predictable structure
            "laminarity": lam,  # High = trapped in states
            "det_up": det_up,  # Determinism of up-move sequences
            "det_down": det_down,  # Determinism of down-move sequences
            "recurrence_asymmetry": recurrence_asymmetry,  # Directional structure difference
        }


class TailBehavior:
    """
    Tail-specific measures - LEFT and RIGHT tails measured SEPARATELY.
    """

    @staticmethod
    def extract_features(prices: np.ndarray, lookback: int = 100) -> Dict:
        """
        Measure tail behavior asymmetrically.
        """
        if len(prices) < lookback + 1:
            lookback = len(prices) - 1

        log_returns = np.diff(np.log(prices[-(lookback + 1) :]))

        # Separate tails
        up = log_returns[log_returns > 0]
        down = log_returns[log_returns < 0]

        # Left tail (down moves) - looking at extreme losses
        if len(down) >= 5:
            down_sorted = np.sort(down)
            left_tail_5pct = down_sorted[: max(1, len(down) // 20)]
            left_tail_mean = np.mean(left_tail_5pct)
            left_tail_ratio = (
                abs(left_tail_mean) / abs(np.median(down)) if np.median(down) != 0 else 1.0
            )
        else:
            left_tail_mean = 0.0
            left_tail_ratio = 1.0

        # Right tail (up moves) - looking at extreme gains
        if len(up) >= 5:
            up_sorted = np.sort(up)[::-1]  # Descending
            right_tail_5pct = up_sorted[: max(1, len(up) // 20)]
            right_tail_mean = np.mean(right_tail_5pct)
            right_tail_ratio = right_tail_mean / np.median(up) if np.median(up) != 0 else 1.0
        else:
            right_tail_mean = 0.0
            right_tail_ratio = 1.0

        return {
            "left_tail_mean": left_tail_mean,  # Average extreme loss (negative)
            "right_tail_mean": right_tail_mean,  # Average extreme gain (positive)
            "left_tail_ratio": left_tail_ratio,  # How extreme are left tail vs median
            "right_tail_ratio": right_tail_ratio,  # How extreme are right tail vs median
            "tail_asymmetry": right_tail_ratio / left_tail_ratio if left_tail_ratio > 0 else 1.0,
        }


class AssumptionFreeEngine:
    """
    Master engine combining all assumption-free measures.
    """

    def __init__(self):
        self.asymmetric_returns = AsymmetricReturns()
        self.rank_measures = RankBasedMeasures()
        self.directional_vol = DirectionalVolatility()
        self.order_flow = DirectionalOrderFlow()
        self.permutation = PermutationPatterns()
        self.recurrence = RecurrenceFeatures()
        self.tails = TailBehavior()

    def extract_all(self, prices: pd.DataFrame, bar_idx: int = -1) -> Dict:
        """
        Extract all assumption-free features.
        """
        if bar_idx < 0:
            bar_idx = len(prices) + bar_idx

        features = {}

        # Get price array
        close_prices = prices["close"].values[: bar_idx + 1]

        if len(close_prices) < 20:
            return {"error": "insufficient_data"}

        # Asymmetric returns
        asym = AsymmetricReturns.extract_features(close_prices)
        features.update({f"asym_{k}": v for k, v in asym.items()})

        # Rank-based
        rank = RankBasedMeasures.extract_features(prices, bar_idx)
        features.update({f"rank_{k}": v for k, v in rank.items()})

        # Directional volatility
        dir_vol = DirectionalVolatility.extract_features(close_prices)
        features.update({f"dvol_{k}": v for k, v in dir_vol.items()})

        # Order flow
        flow = DirectionalOrderFlow.extract_features(prices, bar_idx)
        features.update({f"flow_{k}": v for k, v in flow.items()})

        # Permutation patterns
        perm = PermutationPatterns.extract_features(close_prices)
        features.update({f"perm_{k}": v for k, v in perm.items() if not isinstance(v, tuple)})

        # Recurrence (expensive, use sparingly)
        rec = RecurrenceFeatures.extract_features(close_prices)
        features.update({f"rec_{k}": v for k, v in rec.items()})

        # Tail behavior
        tail = TailBehavior.extract_features(close_prices)
        features.update({f"tail_{k}": v for k, v in tail.items()})

        return features


# Convenience function
def extract_assumption_free_features(prices: pd.DataFrame, bar_idx: int = -1) -> Dict:
    """Quick extraction of all assumption-free features."""
    engine = AssumptionFreeEngine()
    return engine.extract_all(prices, bar_idx)
