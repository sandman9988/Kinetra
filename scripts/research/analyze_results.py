#!/usr/bin/env python3
"""
Results Analysis & Reporting Framework
========================================

Analyzes multi-dataset harness results to answer key research questions:
1. Do regimes differ by asset class?
2. What features predict fat candles?
3. Are there universal patterns vs class-specific?
4. How do timeframes affect regime structure?

Usage:
    python scripts/research/analyze_results.py --input data/research/harness_results_*.json
"""

import sys
from pathlib import Path
import json
import argparse
from typing import Dict, List, Tuple
from dataclasses import dataclass

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class ResearchQuestion:
    """A research question with empirical answer."""
    question: str
    hypothesis: str
    result: str
    evidence: Dict
    confidence: float  # 0-1


class ResultsAnalyzer:
    """
    Comprehensive analysis of multi-dataset research results.
    """

    def __init__(self, results_path: Path):
        self.results_path = results_path
        self.results = self._load_results()
        self.df = pd.DataFrame(self.results)

    def _load_results(self) -> List[Dict]:
        """Load results from JSON."""
        with open(self.results_path, 'r') as f:
            return json.load(f)

    def analyze_all(self) -> Dict:
        """
        Run all analyses and return comprehensive report.
        """
        report = {
            'metadata': {
                'source': str(self.results_path),
                'total_datasets': len(self.results),
                'successful': len([r for r in self.results if r.get('error') is None])
            },
            'questions': [],
            'cross_asset_comparison': {},
            'timeframe_analysis': {},
            'regime_patterns': {},
            'feature_importance': {},
            'recommendations': []
        }

        # Filter to successful results only
        self.df_success = self.df[self.df['error'].isna()].copy()

        if len(self.df_success) < 5:
            report['error'] = "Insufficient successful results for analysis"
            return report

        # Run each analysis
        report['questions'].append(self._question_regimes_by_class())
        report['questions'].append(self._question_fat_candle_predictors())
        report['questions'].append(self._question_universal_patterns())
        report['questions'].append(self._question_timeframe_effects())
        report['questions'].append(self._question_persistence_distribution())

        report['cross_asset_comparison'] = self._cross_asset_comparison()
        report['timeframe_analysis'] = self._timeframe_analysis()
        report['regime_patterns'] = self._regime_pattern_analysis()
        report['feature_importance'] = self._feature_importance()
        report['recommendations'] = self._generate_recommendations()

        return report

    def _question_regimes_by_class(self) -> ResearchQuestion:
        """
        Q1: Do different asset classes have different regime structures?
        """
        # Test if n_regimes differs significantly by class
        classes = self.df_success['asset_class'].unique()

        regime_counts = {}
        for cls in classes:
            regime_counts[cls] = self.df_success[
                self.df_success['asset_class'] == cls
            ]['n_regimes'].values

        # ANOVA test
        if len(classes) >= 2:
            groups = [regime_counts[cls] for cls in classes if len(regime_counts[cls]) > 1]
            if len(groups) >= 2:
                f_stat, p_value = stats.f_oneway(*groups)
            else:
                f_stat, p_value = 0, 1.0
        else:
            f_stat, p_value = 0, 1.0

        significant = p_value < 0.05

        result = (
            "YES - Asset classes have significantly different regime structures"
            if significant else
            "NO - Regime counts are similar across asset classes"
        )

        return ResearchQuestion(
            question="Do different asset classes have different regime structures?",
            hypothesis="Crypto should have more regimes (higher volatility/complexity)",
            result=result,
            evidence={
                'regime_counts_by_class': {
                    cls: {
                        'mean': float(np.mean(vals)),
                        'std': float(np.std(vals)),
                        'count': len(vals)
                    }
                    for cls, vals in regime_counts.items()
                },
                'anova_f_stat': float(f_stat),
                'anova_p_value': float(p_value),
                'significant': significant
            },
            confidence=1.0 - p_value if significant else p_value
        )

    def _question_fat_candle_predictors(self) -> ResearchQuestion:
        """
        Q2: What features predict fat candle frequency?
        """
        # Correlation between features and fat candle percentage
        correlations = {}

        numeric_cols = ['volatility', 'up_persistence', 'down_persistence',
                        'persistence_asymmetry', 'n_regimes', 'dominant_scale']
        target = 'fat_candle_pct'

        for col in numeric_cols:
            if col in self.df_success.columns:
                valid = self.df_success[[col, target]].dropna()
                if len(valid) > 5:
                    r, p = stats.pearsonr(valid[col], valid[target])
                    correlations[col] = {'r': float(r), 'p': float(p)}

        # Find strongest predictor
        strongest = max(correlations.items(), key=lambda x: abs(x[1]['r'])) if correlations else ('none', {'r': 0})

        return ResearchQuestion(
            question="What features predict fat candle (explosive move) frequency?",
            hypothesis="Higher volatility and low persistence (mean-reverting) should predict more fat candles",
            result=f"Strongest predictor: {strongest[0]} (r={strongest[1]['r']:.3f})",
            evidence={
                'correlations': correlations,
                'strongest_predictor': strongest[0],
                'strongest_r': strongest[1]['r']
            },
            confidence=abs(strongest[1]['r'])
        )

    def _question_universal_patterns(self) -> ResearchQuestion:
        """
        Q3: Are there universal regime patterns across all assets?
        """
        # Check which regime labels appear in >50% of successful datasets
        all_labels = []
        for labels in self.df_success['regime_labels']:
            if isinstance(labels, list):
                all_labels.extend(labels)

        label_counts = {}
        for label in all_labels:
            label_counts[label] = label_counts.get(label, 0) + 1

        n_datasets = len(self.df_success)
        universal = [l for l, c in label_counts.items() if c >= n_datasets * 0.5]
        class_specific = [l for l, c in label_counts.items() if c < n_datasets * 0.3]

        return ResearchQuestion(
            question="Are there universal regime patterns vs class-specific ones?",
            hypothesis="Some regimes (calm, explosive) should be universal; others class-specific",
            result=f"{len(universal)} universal patterns, {len(class_specific)} rare/class-specific",
            evidence={
                'universal_patterns': universal,
                'class_specific': class_specific,
                'label_frequency': dict(sorted(label_counts.items(), key=lambda x: -x[1]))
            },
            confidence=len(universal) / max(len(label_counts), 1)
        )

    def _question_timeframe_effects(self) -> ResearchQuestion:
        """
        Q4: How do timeframes affect regime structure?
        """
        timeframes = self.df_success['timeframe'].unique()

        tf_stats = {}
        for tf in timeframes:
            tf_data = self.df_success[self.df_success['timeframe'] == tf]
            tf_stats[tf] = {
                'count': len(tf_data),
                'avg_regimes': float(tf_data['n_regimes'].mean()),
                'avg_up_persistence': float(tf_data['up_persistence'].mean()) if 'up_persistence' in tf_data else 0.5,
                'avg_down_persistence': float(tf_data['down_persistence'].mean()) if 'down_persistence' in tf_data else 0.5,
                'avg_volatility': float(tf_data['volatility'].mean())
            }

        # Test trend: do higher timeframes have fewer regimes?
        tf_order = {'M15': 15, 'M30': 30, 'H1': 60, 'H4': 240}
        ordered = sorted(
            [(tf, tf_stats[tf]['avg_regimes']) for tf in timeframes if tf in tf_order],
            key=lambda x: tf_order.get(x[0], 999)
        )

        if len(ordered) >= 2:
            tf_minutes = [tf_order[x[0]] for x in ordered]
            regimes = [x[1] for x in ordered]
            r, p = stats.spearmanr(tf_minutes, regimes)
        else:
            r, p = 0, 1.0

        return ResearchQuestion(
            question="How do timeframes affect regime structure?",
            hypothesis="Higher timeframes should have fewer, more stable regimes",
            result=f"Correlation between timeframe and regime count: r={r:.3f} (p={p:.3f})",
            evidence={
                'timeframe_stats': tf_stats,
                'correlation': float(r),
                'p_value': float(p)
            },
            confidence=abs(r)
        )

    def _question_persistence_distribution(self) -> ResearchQuestion:
        """
        Q5: What is the distribution of directional persistence across markets?
        Uses up/down persistence separately - no symmetric assumptions.
        """
        up_vals = self.df_success['up_persistence'].dropna() if 'up_persistence' in self.df_success else pd.Series([0.5])
        down_vals = self.df_success['down_persistence'].dropna() if 'down_persistence' in self.df_success else pd.Series([0.5])
        asym_vals = self.df_success['persistence_asymmetry'].dropna() if 'persistence_asymmetry' in self.df_success else pd.Series([0.0])

        # Trending = persistence > 0.55, Mean-reverting = persistence < 0.45
        up_trending = (up_vals > 0.55).mean()
        down_trending = (down_vals > 0.55).mean()
        up_mean_rev = (up_vals < 0.45).mean()
        down_mean_rev = (down_vals < 0.45).mean()
        asymmetric = (abs(asym_vals) > 0.1).mean()

        # By asset class
        persistence_by_class = {}
        for cls in self.df_success['asset_class'].unique():
            cls_data = self.df_success[self.df_success['asset_class'] == cls]
            up_p = cls_data['up_persistence'] if 'up_persistence' in cls_data else pd.Series([0.5])
            down_p = cls_data['down_persistence'] if 'down_persistence' in cls_data else pd.Series([0.5])
            asym_p = cls_data['persistence_asymmetry'] if 'persistence_asymmetry' in cls_data else pd.Series([0.0])
            persistence_by_class[cls] = {
                'mean_up_persistence': float(up_p.mean()),
                'mean_down_persistence': float(down_p.mean()),
                'mean_asymmetry': float(asym_p.mean()),
                'up_trending_pct': float((up_p > 0.55).mean() * 100),
                'down_trending_pct': float((down_p > 0.55).mean() * 100)
            }

        return ResearchQuestion(
            question="What is the distribution of directional persistence across markets?",
            hypothesis="Markets should show asymmetric persistence (up ≠ down)",
            result=f"Up trending: {up_trending*100:.1f}%, Down trending: {down_trending*100:.1f}%, Asymmetric: {asymmetric*100:.1f}%",
            evidence={
                'overall': {
                    'up_trending_pct': float(up_trending * 100),
                    'down_trending_pct': float(down_trending * 100),
                    'up_mean_reverting_pct': float(up_mean_rev * 100),
                    'down_mean_reverting_pct': float(down_mean_rev * 100),
                    'asymmetric_pct': float(asymmetric * 100),
                    'mean_up_persistence': float(up_vals.mean()),
                    'mean_down_persistence': float(down_vals.mean()),
                    'mean_asymmetry': float(asym_vals.mean())
                },
                'by_class': persistence_by_class
            },
            confidence=0.9  # Directional persistence is assumption-free
        )

    def _cross_asset_comparison(self) -> Dict:
        """Compare key metrics across asset classes."""
        comparison = {}

        for cls in self.df_success['asset_class'].unique():
            cls_data = self.df_success[self.df_success['asset_class'] == cls]

            comparison[cls] = {
                'n_datasets': len(cls_data),
                'avg_regimes': float(cls_data['n_regimes'].mean()),
                'std_regimes': float(cls_data['n_regimes'].std()),
                'avg_volatility': float(cls_data['volatility'].mean()),
                'avg_up_persistence': float(cls_data['up_persistence'].mean()) if 'up_persistence' in cls_data else 0.5,
                'avg_down_persistence': float(cls_data['down_persistence'].mean()) if 'down_persistence' in cls_data else 0.5,
                'avg_persistence_asymmetry': float(cls_data['persistence_asymmetry'].mean()) if 'persistence_asymmetry' in cls_data else 0.0,
                'avg_fat_candle_pct': float(cls_data['fat_candle_pct'].mean()),
                'most_common_labels': self._get_top_labels(cls_data, n=3)
            }

        return comparison

    def _get_top_labels(self, df: pd.DataFrame, n: int = 3) -> List[str]:
        """Get most common regime labels from a subset."""
        all_labels = []
        for labels in df['regime_labels']:
            if isinstance(labels, list):
                all_labels.extend(labels)

        label_counts = {}
        for label in all_labels:
            label_counts[label] = label_counts.get(label, 0) + 1

        return [l for l, _ in sorted(label_counts.items(), key=lambda x: -x[1])[:n]]

    def _timeframe_analysis(self) -> Dict:
        """Analyze patterns by timeframe."""
        analysis = {}

        for tf in self.df_success['timeframe'].unique():
            tf_data = self.df_success[self.df_success['timeframe'] == tf]

            analysis[tf] = {
                'n_datasets': len(tf_data),
                'avg_regimes': float(tf_data['n_regimes'].mean()),
                'avg_dominant_scale': float(tf_data['dominant_scale'].mean()),
                'avg_up_persistence': float(tf_data['up_persistence'].mean()) if 'up_persistence' in tf_data else 0.5,
                'avg_down_persistence': float(tf_data['down_persistence'].mean()) if 'down_persistence' in tf_data else 0.5
            }

        return analysis

    def _regime_pattern_analysis(self) -> Dict:
        """Analyze regime patterns and transitions."""
        # Collect all regime labels
        all_labels = []
        for labels in self.df_success['regime_labels']:
            if isinstance(labels, list):
                all_labels.extend(labels)

        # Frequency
        label_freq = {}
        for label in all_labels:
            label_freq[label] = label_freq.get(label, 0) + 1

        # Which labels co-occur?
        co_occurrence = {}
        for labels in self.df_success['regime_labels']:
            if isinstance(labels, list) and len(labels) >= 2:
                for i, l1 in enumerate(labels):
                    for l2 in labels[i+1:]:
                        key = tuple(sorted([l1, l2]))
                        co_occurrence[key] = co_occurrence.get(key, 0) + 1

        return {
            'label_frequency': dict(sorted(label_freq.items(), key=lambda x: -x[1])),
            'top_co_occurrences': dict(
                sorted(co_occurrence.items(), key=lambda x: -x[1])[:10]
            )
        }

    def _feature_importance(self) -> Dict:
        """Estimate which features are most important for regime differentiation."""
        # Based on correlation with key outcomes
        features = ['volatility', 'up_persistence', 'down_persistence',
                    'persistence_asymmetry', 'dominant_scale']
        outcomes = ['n_regimes', 'fat_candle_pct']

        importance = {}

        for feature in features:
            if feature not in self.df_success.columns:
                continue

            importance[feature] = {}
            for outcome in outcomes:
                if outcome not in self.df_success.columns:
                    continue

                valid = self.df_success[[feature, outcome]].dropna()
                if len(valid) > 5:
                    r, p = stats.pearsonr(valid[feature], valid[outcome])
                    importance[feature][outcome] = {
                        'correlation': float(r),
                        'p_value': float(p),
                        'significant': p < 0.05
                    }

        return importance

    def _generate_recommendations(self) -> List[str]:
        """Generate actionable recommendations based on analysis."""
        recommendations = []

        # Based on persistence distribution
        up_persist = self.df_success['up_persistence'].dropna() if 'up_persistence' in self.df_success else pd.Series([0.5])
        down_persist = self.df_success['down_persistence'].dropna() if 'down_persistence' in self.df_success else pd.Series([0.5])
        asym_vals = self.df_success['persistence_asymmetry'].dropna() if 'persistence_asymmetry' in self.df_success else pd.Series([0.0])

        if up_persist.mean() > 0.55:
            recommendations.append(
                "Up moves show persistence - consider momentum-based entries on upside"
            )
        if down_persist.mean() > 0.55:
            recommendations.append(
                "Down moves show persistence - consider momentum-based entries on downside"
            )
        if abs(asym_vals.mean()) > 0.1:
            recommendations.append(
                f"Asymmetric persistence detected (avg={asym_vals.mean():.3f}) - use directional-specific models"
            )

        # Based on regime counts
        avg_regimes = self.df_success['n_regimes'].mean()
        if avg_regimes > 5:
            recommendations.append(
                f"Average {avg_regimes:.1f} regimes discovered - use regime conditioning in RL"
            )

        # Based on class differences in persistence
        if 'up_persistence' in self.df_success:
            class_up = self.df_success.groupby('asset_class')['up_persistence'].mean()
            class_down = self.df_success.groupby('asset_class')['down_persistence'].mean()
            if class_up.max() - class_up.min() > 0.1 or class_down.max() - class_down.min() > 0.1:
                recommendations.append(
                    "Significant persistence differences across classes - use class-specific models"
                )

        # Fat candle insights
        if self.df_success['fat_candle_pct'].mean() > 3:
            recommendations.append(
                "High fat candle frequency - focus on transition precursors for early detection"
            )

        return recommendations

    def print_report(self, report: Dict):
        """Print formatted report to console."""
        print("\n" + "=" * 80)
        print("MULTI-DATASET RESEARCH ANALYSIS REPORT")
        print("=" * 80)

        print(f"\nData Source: {report['metadata']['source']}")
        print(f"Total Datasets: {report['metadata']['total_datasets']}")
        print(f"Successful: {report['metadata']['successful']}")

        print("\n" + "-" * 80)
        print("RESEARCH QUESTIONS & ANSWERS")
        print("-" * 80)

        for q in report.get('questions', []):
            print(f"\nQ: {q.question}")
            print(f"Hypothesis: {q.hypothesis}")
            print(f"Result: {q.result}")
            print(f"Confidence: {q.confidence:.2f}")

        print("\n" + "-" * 80)
        print("CROSS-ASSET COMPARISON")
        print("-" * 80)

        for cls, stats in report.get('cross_asset_comparison', {}).items():
            print(f"\n{cls.upper()}:")
            print(f"  Datasets: {stats['n_datasets']}")
            print(f"  Avg Regimes: {stats['avg_regimes']:.1f} ± {stats['std_regimes']:.1f}")
            print(f"  Avg Volatility: {stats['avg_volatility']:.6f}")
            print(f"  Up Persistence: {stats.get('avg_up_persistence', 0.5):.3f}")
            print(f"  Down Persistence: {stats.get('avg_down_persistence', 0.5):.3f}")
            print(f"  Persistence Asymmetry: {stats.get('avg_persistence_asymmetry', 0.0):.3f}")
            print(f"  Fat Candles: {stats['avg_fat_candle_pct']:.1f}%")
            print(f"  Top Labels: {', '.join(stats['most_common_labels'])}")

        print("\n" + "-" * 80)
        print("RECOMMENDATIONS")
        print("-" * 80)

        for i, rec in enumerate(report.get('recommendations', []), 1):
            print(f"  {i}. {rec}")

        print("\n" + "=" * 80)

    def save_report(self, report: Dict, output_path: Path):
        """Save report to JSON."""
        # Convert dataclass objects to dicts
        def convert(obj):
            if hasattr(obj, '__dict__'):
                return {k: convert(v) for k, v in obj.__dict__.items()}
            elif isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert(i) for i in obj]
            elif isinstance(obj, tuple):
                return str(obj)
            elif isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            else:
                return obj

        with open(output_path, 'w') as f:
            json.dump(convert(report), f, indent=2, default=str)

        print(f"\n[REPORT] Saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Analyze multi-dataset research results')
    parser.add_argument('--input', type=str, required=True, help='Path to harness results JSON')
    parser.add_argument('--output', type=str, help='Output path for report JSON')

    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"[ERROR] Input file not found: {input_path}")
        sys.exit(1)

    analyzer = ResultsAnalyzer(input_path)
    report = analyzer.analyze_all()

    # Print to console
    analyzer.print_report(report)

    # Save if output specified
    if args.output:
        output_path = Path(args.output)
        analyzer.save_report(report, output_path)


if __name__ == "__main__":
    main()
