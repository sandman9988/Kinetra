"""
Trend Definition Discovery Module
=================================

CORE INSIGHT: "Trend" means different things for different asset classes.

We DON'T assume RSI/MACD/ADX define trends. Instead:
1. Track ALL measurements
2. Correlate with trend-following trade outcomes PER CLASS
3. Discover which measurements ACTUALLY define "trending" for each class

Example discoveries we might find:
- Forex: Hurst + session_overlap + z_score defines trend
- Crypto: entropy_rate + kinetic_energy + volume_ratio defines trend
- Metals: reynolds + potential_energy + roc_50 defines trend
- Indices: flow_regime + adx + time_hour defines trend
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set
from collections import defaultdict
import numpy as np
import pandas as pd


@dataclass
class TrendTradeOutcome:
    """Record of a trend-following trade attempt."""
    asset_class: str
    instrument: str

    # Entry measurements (ALL of them)
    entry_measurements: Dict[str, float] = field(default_factory=dict)

    # Trade outcome
    pnl: float = 0.0
    mae: float = 0.0
    mfe: float = 0.0
    bars_held: int = 0

    # Was this a "trend following" trade? (held > N bars with direction)
    is_trend_trade: bool = False
    trend_direction: int = 0  # 1 = long trend, -1 = short trend

    # Did the trend continue? (key for learning)
    trend_continued: bool = False  # MFE > 2*MAE = trend worked


class TrendDefinitionLearner:
    """
    Learns what "trend" means for each asset class.

    Approach:
    1. Record ALL measurements at trade entry
    2. Track which trades were successful trend-following
    3. Find measurements that PREDICT trend success per class
    4. Build class-specific trend scores
    """

    def __init__(self, min_trades_for_learning: int = 30):
        self.min_trades = min_trades_for_learning

        # Store all trade outcomes per class
        self.trades: Dict[str, List[TrendTradeOutcome]] = defaultdict(list)

        # Learned trend predictors per class
        # {class: {measurement_name: correlation_with_success}}
        self.trend_predictors: Dict[str, Dict[str, float]] = defaultdict(dict)

        # Top N measurements that define trend per class
        self.trend_definitions: Dict[str, List[Tuple[str, float]]] = {}

        # Measurement importance ranking per class
        self.measurement_rankings: Dict[str, pd.DataFrame] = {}

    def record_trade(
        self,
        asset_class: str,
        instrument: str,
        entry_measurements: Dict[str, float],
        pnl: float,
        mae: float,
        mfe: float,
        bars_held: int,
        direction: int,  # 1=long, -1=short
    ):
        """Record a trade for trend learning."""

        # Determine if this was a trend-following trade
        # Trend trade = held for reasonable duration with clear direction
        is_trend_trade = bars_held >= 5

        # Trend success = MFE significantly > MAE (trend continued in our favor)
        trend_continued = (mfe > abs(mae) * 1.5) and (pnl > 0)

        outcome = TrendTradeOutcome(
            asset_class=asset_class,
            instrument=instrument,
            entry_measurements=entry_measurements.copy(),
            pnl=pnl,
            mae=mae,
            mfe=mfe,
            bars_held=bars_held,
            is_trend_trade=is_trend_trade,
            trend_direction=direction,
            trend_continued=trend_continued,
        )

        self.trades[asset_class].append(outcome)

        # Trigger learning if we have enough data
        if len(self.trades[asset_class]) % 50 == 0:
            self._learn_trend_definition(asset_class)

    def _learn_trend_definition(self, asset_class: str):
        """
        Learn what measurements predict trend success for this class.

        This is THE key learning: which measurements → successful trends?
        """
        trades = self.trades[asset_class]

        if len(trades) < self.min_trades:
            return

        # Only analyze trend trades
        trend_trades = [t for t in trades if t.is_trend_trade]

        if len(trend_trades) < self.min_trades // 2:
            return

        # Get all measurement names
        all_measurements: Set[str] = set()
        for t in trend_trades:
            all_measurements.update(t.entry_measurements.keys())

        # For each measurement, compute correlation with trend success
        correlations = {}

        for meas_name in all_measurements:
            values = []
            successes = []

            for t in trend_trades:
                if meas_name in t.entry_measurements:
                    val = t.entry_measurements[meas_name]
                    if np.isfinite(val):
                        values.append(val)
                        successes.append(1.0 if t.trend_continued else 0.0)

            if len(values) >= 10:
                # Correlation between measurement and success
                corr = np.corrcoef(values, successes)[0, 1]
                if np.isfinite(corr):
                    correlations[meas_name] = corr

        self.trend_predictors[asset_class] = correlations

        # Rank measurements by absolute correlation
        ranked = sorted(
            correlations.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )

        # Store top predictors as "trend definition" for this class
        self.trend_definitions[asset_class] = ranked[:15]  # Top 15

        # Create detailed ranking DataFrame
        self.measurement_rankings[asset_class] = pd.DataFrame([
            {
                'measurement': name,
                'correlation': corr,
                'abs_correlation': abs(corr),
                'direction': 'positive' if corr > 0 else 'negative',
            }
            for name, corr in ranked
        ])

    def get_trend_score(
        self,
        asset_class: str,
        measurements: Dict[str, float],
        top_n: int = 10
    ) -> float:
        """
        Compute trend score using LEARNED definition for this class.

        Returns score from -1 to +1:
        - Positive = measurements suggest trend-following will work
        - Negative = measurements suggest trend-following will fail
        """
        if asset_class not in self.trend_definitions:
            return 0.0  # Not enough data yet

        definition = self.trend_definitions[asset_class][:top_n]

        if not definition:
            return 0.0

        score = 0.0
        weight_sum = 0.0

        for meas_name, correlation in definition:
            if meas_name in measurements:
                val = measurements[meas_name]
                if np.isfinite(val):
                    # Weight by correlation strength
                    weight = abs(correlation)
                    # Score contribution: measurement value * correlation direction
                    score += val * np.sign(correlation) * weight
                    weight_sum += weight

        if weight_sum > 0:
            score = np.tanh(score / weight_sum)  # Normalize to [-1, 1]

        return score

    def get_trend_definition(self, asset_class: str) -> Dict:
        """
        Get the learned trend definition for an asset class.

        Returns what measurements define "trending" for this class.
        """
        if asset_class not in self.trend_definitions:
            return {'status': 'not_enough_data', 'trades_recorded': len(self.trades.get(asset_class, []))}

        definition = self.trend_definitions[asset_class]

        return {
            'asset_class': asset_class,
            'trades_analyzed': len([t for t in self.trades[asset_class] if t.is_trend_trade]),
            'top_predictors': [
                {
                    'measurement': name,
                    'correlation': corr,
                    'interpretation': self._interpret_predictor(name, corr)
                }
                for name, corr in definition[:10]
            ],
            'trend_score_formula': self._describe_formula(definition[:5]),
        }

    def _interpret_predictor(self, measurement: str, correlation: float) -> str:
        """Human-readable interpretation of a predictor."""
        direction = "HIGH" if correlation > 0 else "LOW"
        strength = "strongly" if abs(correlation) > 0.3 else "moderately"

        return f"{direction} {measurement} {strength} predicts trend success"

    def _describe_formula(self, predictors: List[Tuple[str, float]]) -> str:
        """Describe the trend score formula in human terms."""
        parts = []
        for name, corr in predictors:
            sign = "+" if corr > 0 else "-"
            weight = abs(corr)
            parts.append(f"{sign}{weight:.2f}*{name}")

        return "trend_score = tanh(" + " ".join(parts) + ")"

    def compare_trend_definitions(self) -> pd.DataFrame:
        """
        Compare what "trend" means across asset classes.

        This is THE key output - shows that trend ≠ trend.
        """
        rows = []

        for asset_class in self.trend_definitions:
            definition = self.trend_definitions[asset_class]

            if not definition:
                continue

            row = {
                'asset_class': asset_class,
                'n_trades': len([t for t in self.trades[asset_class] if t.is_trend_trade]),
            }

            # Top 5 predictors for each class
            for i, (name, corr) in enumerate(definition[:5]):
                row[f'predictor_{i+1}'] = name
                row[f'corr_{i+1}'] = corr

            rows.append(row)

        return pd.DataFrame(rows)

    def get_discovery_report(self) -> str:
        """Generate human-readable discovery report."""
        lines = [
            "=" * 70,
            "  TREND DEFINITION DISCOVERY",
            "  What 'trending' actually means per asset class",
            "=" * 70,
        ]

        for asset_class in sorted(self.trend_definitions.keys()):
            definition = self.get_trend_definition(asset_class)

            lines.append(f"\n{asset_class}:")
            lines.append(f"  Trades analyzed: {definition['trades_analyzed']}")
            lines.append(f"  Formula: {definition['trend_score_formula']}")
            lines.append(f"  Top predictors:")

            for pred in definition['top_predictors'][:5]:
                bar_len = int(abs(pred['correlation']) * 20)
                bar = "█" * bar_len
                sign = "+" if pred['correlation'] > 0 else "-"
                lines.append(f"    {sign}{abs(pred['correlation']):.3f} {pred['measurement']:<30} {bar}")

        # Comparison
        if len(self.trend_definitions) > 1:
            lines.append("\n" + "-" * 70)
            lines.append("CROSS-CLASS COMPARISON:")
            lines.append("-" * 70)

            # Find measurements unique to each class
            all_top_5: Dict[str, Set[str]] = {}
            for cls, defn in self.trend_definitions.items():
                all_top_5[cls] = set(name for name, _ in defn[:5])

            for cls in all_top_5:
                unique = all_top_5[cls].copy()
                for other_cls, other_meas in all_top_5.items():
                    if other_cls != cls:
                        unique -= other_meas

                if unique:
                    lines.append(f"  {cls} unique trend predictors: {', '.join(unique)}")

        return "\n".join(lines)


class MeanReversionDefinitionLearner(TrendDefinitionLearner):
    """
    Same approach but for mean reversion.

    Learns what measurements predict MR success per class.
    MR success = price reverted after extreme reading.
    """

    def record_trade(
        self,
        asset_class: str,
        instrument: str,
        entry_measurements: Dict[str, float],
        pnl: float,
        mae: float,
        mfe: float,
        bars_held: int,
        direction: int,
    ):
        """Record a trade for MR learning."""

        # MR trade = short hold after extreme reading
        is_mr_trade = bars_held <= 10

        # MR success = quick profit with low MAE (clean reversion)
        mr_success = (pnl > 0) and (abs(mae) < mfe * 0.5) and (bars_held <= 15)

        outcome = TrendTradeOutcome(
            asset_class=asset_class,
            instrument=instrument,
            entry_measurements=entry_measurements.copy(),
            pnl=pnl,
            mae=mae,
            mfe=mfe,
            bars_held=bars_held,
            is_trend_trade=is_mr_trade,  # Reusing field
            trend_direction=direction,
            trend_continued=mr_success,  # Reusing field
        )

        self.trades[asset_class].append(outcome)

        if len(self.trades[asset_class]) % 50 == 0:
            self._learn_trend_definition(asset_class)  # Reuse learning logic

    def get_mr_score(
        self,
        asset_class: str,
        measurements: Dict[str, float],
        top_n: int = 10
    ) -> float:
        """Compute MR score using learned definition."""
        return self.get_trend_score(asset_class, measurements, top_n)

    def get_discovery_report(self) -> str:
        """Generate MR discovery report."""
        lines = [
            "=" * 70,
            "  MEAN REVERSION DEFINITION DISCOVERY",
            "  What 'mean reverting' actually means per asset class",
            "=" * 70,
        ]

        for asset_class in sorted(self.trend_definitions.keys()):
            definition = self.get_trend_definition(asset_class)

            lines.append(f"\n{asset_class}:")
            lines.append(f"  Trades analyzed: {definition['trades_analyzed']}")
            lines.append(f"  Top MR predictors:")

            for pred in definition.get('top_predictors', [])[:5]:
                bar_len = int(abs(pred['correlation']) * 20)
                bar = "█" * bar_len
                sign = "+" if pred['correlation'] > 0 else "-"
                lines.append(f"    {sign}{abs(pred['correlation']):.3f} {pred['measurement']:<30} {bar}")

        return "\n".join(lines)


class UnifiedStrategyLearner:
    """
    Unified learner that discovers:
    1. What defines "trend" per class
    2. What defines "mean reversion" per class
    3. When to use which strategy per class

    The agent doesn't choose between hardcoded strategies -
    it learns what strategies MEAN for each asset class.
    """

    def __init__(self):
        self.trend_learner = TrendDefinitionLearner()
        self.mr_learner = MeanReversionDefinitionLearner()

        # Track when each strategy works per class
        self.strategy_performance: Dict[str, Dict[str, List[float]]] = defaultdict(
            lambda: {'trend': [], 'mr': []}
        )

    def record_trade(
        self,
        asset_class: str,
        instrument: str,
        entry_measurements: Dict[str, float],
        pnl: float,
        mae: float,
        mfe: float,
        bars_held: int,
        direction: int,
    ):
        """Record trade for both learners."""
        # Record for trend learning
        self.trend_learner.record_trade(
            asset_class, instrument, entry_measurements,
            pnl, mae, mfe, bars_held, direction
        )

        # Record for MR learning
        self.mr_learner.record_trade(
            asset_class, instrument, entry_measurements,
            pnl, mae, mfe, bars_held, direction
        )

        # Track strategy performance
        if bars_held >= 10:  # Trend-like trade
            self.strategy_performance[asset_class]['trend'].append(pnl)
        else:  # MR-like trade
            self.strategy_performance[asset_class]['mr'].append(pnl)

    def get_strategy_scores(
        self,
        asset_class: str,
        measurements: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Get both trend and MR scores for current measurements.

        Agent can use these to decide which approach to take.
        """
        return {
            'trend_score': self.trend_learner.get_trend_score(asset_class, measurements),
            'mr_score': self.mr_learner.get_mr_score(asset_class, measurements),
            'trend_reliability': self._get_strategy_reliability(asset_class, 'trend'),
            'mr_reliability': self._get_strategy_reliability(asset_class, 'mr'),
        }

    def _get_strategy_reliability(self, asset_class: str, strategy: str) -> float:
        """Get historical reliability of a strategy for this class."""
        trades = self.strategy_performance[asset_class][strategy]
        if len(trades) < 10:
            return 0.5  # Neutral

        # Reliability = win rate
        wins = sum(1 for pnl in trades if pnl > 0)
        return wins / len(trades)

    def get_class_strategy_profile(self, asset_class: str) -> Dict:
        """Get complete strategy profile for a class."""
        trend_def = self.trend_learner.get_trend_definition(asset_class)
        mr_def = self.mr_learner.get_trend_definition(asset_class)

        trend_trades = self.strategy_performance[asset_class]['trend']
        mr_trades = self.strategy_performance[asset_class]['mr']

        return {
            'asset_class': asset_class,
            'trend': {
                'definition': trend_def,
                'trades': len(trend_trades),
                'win_rate': sum(1 for p in trend_trades if p > 0) / len(trend_trades) if trend_trades else 0,
                'avg_pnl': np.mean(trend_trades) if trend_trades else 0,
            },
            'mean_reversion': {
                'definition': mr_def,
                'trades': len(mr_trades),
                'win_rate': sum(1 for p in mr_trades if p > 0) / len(mr_trades) if mr_trades else 0,
                'avg_pnl': np.mean(mr_trades) if mr_trades else 0,
            },
            'recommended_strategy': 'trend' if (
                self._get_strategy_reliability(asset_class, 'trend') >
                self._get_strategy_reliability(asset_class, 'mr')
            ) else 'mr',
        }

    def get_full_discovery_report(self) -> str:
        """Generate comprehensive discovery report."""
        lines = [
            "=" * 80,
            "  UNIFIED STRATEGY DISCOVERY REPORT",
            "  What works for each asset class - learned from data",
            "=" * 80,
        ]

        # Per-class profiles
        all_classes = set(self.trend_learner.trend_definitions.keys()) | \
                      set(self.mr_learner.trend_definitions.keys())

        for asset_class in sorted(all_classes):
            profile = self.get_class_strategy_profile(asset_class)

            lines.append(f"\n{'─' * 60}")
            lines.append(f"  {asset_class}")
            lines.append(f"{'─' * 60}")

            # Trend info
            trend = profile['trend']
            lines.append(f"\n  TREND-FOLLOWING:")
            lines.append(f"    Trades: {trend['trades']}, Win Rate: {trend['win_rate']:.1%}, Avg PnL: {trend['avg_pnl']:.3f}%")
            if trend['definition'].get('top_predictors'):
                lines.append(f"    Definition (top 3):")
                for pred in trend['definition']['top_predictors'][:3]:
                    lines.append(f"      {pred['correlation']:+.3f} {pred['measurement']}")

            # MR info
            mr = profile['mean_reversion']
            lines.append(f"\n  MEAN REVERSION:")
            lines.append(f"    Trades: {mr['trades']}, Win Rate: {mr['win_rate']:.1%}, Avg PnL: {mr['avg_pnl']:.3f}%")
            if mr['definition'].get('top_predictors'):
                lines.append(f"    Definition (top 3):")
                for pred in mr['definition']['top_predictors'][:3]:
                    lines.append(f"      {pred['correlation']:+.3f} {pred['measurement']}")

            lines.append(f"\n  → RECOMMENDED: {profile['recommended_strategy'].upper()}")

        # Cross-class comparison
        lines.append(f"\n{'=' * 80}")
        lines.append("  CROSS-CLASS COMPARISON: What 'trend' means")
        lines.append("=" * 80)

        comp_df = self.trend_learner.compare_trend_definitions()
        if not comp_df.empty:
            for _, row in comp_df.iterrows():
                cls = row['asset_class']
                predictors = [row.get(f'predictor_{i}', 'N/A') for i in range(1, 4)]
                lines.append(f"  {cls:<20}: {', '.join(str(p) for p in predictors if p != 'N/A')}")

        return "\n".join(lines)


# Export
__all__ = [
    'TrendTradeOutcome',
    'TrendDefinitionLearner',
    'MeanReversionDefinitionLearner',
    'UnifiedStrategyLearner',
]
