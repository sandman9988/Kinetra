#!/usr/bin/env python3
"""
Unified Testing Framework - Consolidates All Test Scripts
===========================================================

Integrates:
1. explore_specialization.py - Agent specialization strategies
2. train_triad.py - Triad system training
3. superpot_by_class.py - Asset class testing
4. superpot_complete.py - Complete exploration
5. Additional: Stacking, control groups, efficiency metrics

Usage:
    # Run all tests
    python scripts/unified_test_framework.py --full
    
    # Quick test
    python scripts/unified_test_framework.py --quick
    
    # Specific test suite
    python scripts/unified_test_framework.py --suite control
    python scripts/unified_test_framework.py --suite physics
    python scripts/unified_test_framework.py --suite specialization
    
    # Compare multiple suites
    python scripts/unified_test_framework.py --compare control physics ml
    
    # Custom configuration
    python scripts/unified_test_framework.py --config my_test_config.yaml
"""

import argparse
import json
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from kinetra.testing_framework import (
    TestingFramework,
    TestConfiguration,
    InstrumentSpec,
    StandardIndicators,
    EfficiencyMetrics,
    StatisticalValidator,
    FirstPrinciplesValidator,
    classify_asset,
)


# =============================================================================
# DATA DISCOVERY
# =============================================================================

def discover_instruments(
    data_dirs: Optional[List[str]] = None,
    asset_classes: Optional[List[str]] = None,
    timeframes: Optional[List[str]] = None,
    max_per_class: int = 3
) -> List[InstrumentSpec]:
    """
    Discover available instruments for testing.
    
    Ensures "apples to apples" comparison by selecting same instruments
    across different tests.
    """
    if data_dirs is None:
        data_dirs = [
            "data/master",
            "data/runs/berserker_run3/data",
            "data",
        ]
    
    instruments = []
    seen = set()
    by_class = {}
    
    for data_dir in data_dirs:
        path = Path(data_dir)
        if not path.exists():
            continue
        
        for csv_file in path.rglob("*.csv"):
            # Skip non-data files
            if 'symbol' in csv_file.name.lower() or 'info' in csv_file.name.lower():
                continue
            
            # Parse filename
            name = csv_file.stem
            parts = name.split('_')
            
            if len(parts) < 2:
                continue
            
            symbol = parts[0]
            timeframe = parts[1]
            
            # Filter by timeframe if specified
            if timeframes and timeframe not in timeframes:
                continue
            
            # Classify asset
            asset_class = classify_asset(symbol)
            
            # Filter by asset class if specified
            if asset_classes and asset_class not in asset_classes:
                continue
            
            if asset_class == 'unknown':
                continue
            
            # Create spec
            key = (symbol, timeframe)
            if key in seen:
                continue
            
            spec = InstrumentSpec(
                symbol=symbol,
                asset_class=asset_class,
                timeframe=timeframe,
                data_path=str(csv_file)
            )
            
            # Group by class
            if asset_class not in by_class:
                by_class[asset_class] = []
            
            if len(by_class[asset_class]) < max_per_class:
                by_class[asset_class].append(spec)
                instruments.append(spec)
                seen.add(key)
    
    return instruments


# =============================================================================
# TEST SUITE DEFINITIONS
# =============================================================================

def create_control_suite(instruments: List[InstrumentSpec]) -> TestConfiguration:
    """
    Control group: Standard technical indicators.
    
    Baseline to beat. Uses traditional TA: MA, RSI, MACD, Bollinger Bands.
    """
    return TestConfiguration(
        name="control_standard_indicators",
        description="Control group using standard technical indicators (MA, RSI, MACD, BB, ATR)",
        instruments=instruments,
        agent_type="control",
        agent_config={
            "indicators": ["SMA_20", "EMA_20", "RSI_14", "MACD", "BB_20_2", "ATR_14"],
            "entry_rules": "MA crossover + RSI confirmation",
            "exit_rules": "Bollinger Band touch or opposite signal",
            # Note: Uses magic numbers (14, 20) - this is intentional for control
        },
        episodes=100,
        use_gpu=False,  # Control doesn't need GPU
    )


def create_physics_suite(instruments: List[InstrumentSpec]) -> TestConfiguration:
    """
    Physics-based suite: Energy, damping, entropy.
    
    First principles approach - no magic numbers, adaptive thresholds.
    """
    return TestConfiguration(
        name="physics_first_principles",
        description="Physics-based trading using energy, damping, entropy (no magic numbers)",
        instruments=instruments,
        agent_type="physics",
        agent_config={
            "features": ["energy", "damping", "entropy", "reynolds", "regime"],
            "thresholds": "rolling_percentiles",  # No fixed thresholds
            "entry_logic": "high_energy_low_damping",
            "exit_logic": "entropy_spike_or_regime_change",
            "adaptive": True,
            "use_gpu": True,
        },
        episodes=100,
        use_gpu=True,
    )


def create_rl_suite(instruments: List[InstrumentSpec], agent_type: str = "PPO") -> TestConfiguration:
    """
    Reinforcement Learning suite: PPO, SAC, A2C.
    """
    return TestConfiguration(
        name=f"rl_{agent_type.lower()}",
        description=f"Reinforcement Learning using {agent_type} with physics features",
        instruments=instruments,
        agent_type=f"rl_{agent_type}",
        agent_config={
            "algorithm": agent_type,
            "features": "all_physics_64dim",
            "reward_shaping": "adaptive",
            "network_architecture": [256, 256],
            "learning_rate": 3e-4,
            "use_gpu": True,
        },
        episodes=200,  # More episodes for RL
        use_gpu=True,
    )


def create_specialization_suite(
    instruments: List[InstrumentSpec],
    specialization_type: str = "asset_class"
) -> TestConfiguration:
    """
    Specialization suite: Test different specialization strategies.
    
    Types:
    - asset_class: Separate agents per market type
    - regime: Separate agents per physics regime
    - timeframe: Separate agents per timeframe
    """
    return TestConfiguration(
        name=f"specialization_{specialization_type}",
        description=f"Agent specialization by {specialization_type}",
        instruments=instruments,
        agent_type="specialized",
        agent_config={
            "specialization": specialization_type,
            "base_algorithm": "PPO",
            "shared_features": True,
            "use_gpu": True,
        },
        episodes=150,
        use_gpu=True,
    )


def create_stacking_suite(instruments: List[InstrumentSpec]) -> TestConfiguration:
    """
    Stacking suite: Combine multiple models for marginal gains.
    """
    return TestConfiguration(
        name="stacking_ensemble",
        description="Ensemble stacking of multiple agents (physics + RL + control)",
        instruments=instruments,
        agent_type="stacking",
        agent_config={
            "base_models": ["physics", "PPO", "SAC"],
            "meta_learner": "gradient_boosting",
            "weighting": "adaptive",
            "use_gpu": True,
        },
        episodes=100,
        use_gpu=True,
    )


def create_triad_suite(
    instruments: List[InstrumentSpec],
    role: str = "trader"
) -> TestConfiguration:
    """
    Triad suite: Incumbent, Competitor, Researcher.
    """
    return TestConfiguration(
        name=f"triad_{role}",
        description=f"Triad system with role: {role}",
        instruments=instruments,
        agent_type="triad",
        agent_config={
            "role": role,
            "incumbent": "PPO",
            "competitor": "A2C",
            "researcher": "SAC",
            "competition_frequency": 10,
            "use_gpu": True,
        },
        episodes=150,
        use_gpu=True,
    )


def create_hidden_dimension_suite(instruments: List[InstrumentSpec]) -> TestConfiguration:
    """
    Hidden Dimension Discovery: "We don't know what we don't know"
    
    Uses dimensionality reduction, autoencoders, and PCA to discover
    hidden patterns and features we can't see directly.
    """
    return TestConfiguration(
        name="hidden_dimension_discovery",
        description="Discover hidden dimensions and latent features using autoencoders, PCA, t-SNE",
        instruments=instruments,
        agent_type="hidden_discovery",
        agent_config={
            "methods": ["autoencoder", "pca", "tsne", "umap", "ica"],
            "latent_dims": [8, 16, 32],  # Try different compression levels
            "reconstruction_loss": "mse",
            "feature_extraction": "learned",
            "use_gpu": True,
        },
        episodes=200,
        use_gpu=True,
    )


def create_meta_learning_suite(instruments: List[InstrumentSpec]) -> TestConfiguration:
    """
    Meta-Learning: Learn to learn what matters.
    
    Train a meta-learner to discover which features, combinations,
    and strategies work best across different contexts.
    """
    return TestConfiguration(
        name="meta_learning",
        description="Meta-learning to discover optimal feature combinations and strategies",
        instruments=instruments,
        agent_type="meta_learning",
        agent_config={
            "algorithm": "MAML",  # Model-Agnostic Meta-Learning
            "inner_loop_steps": 5,
            "outer_loop_lr": 1e-3,
            "feature_selection": "automatic",
            "strategy_discovery": True,
            "use_gpu": True,
        },
        episodes=250,
        use_gpu=True,
    )


def create_cross_regime_suite(instruments: List[InstrumentSpec]) -> TestConfiguration:
    """
    Cross-Regime Analysis: Test across all regime transitions.
    
    Studies behavior during regime changes (laminar->underdamped,
    underdamped->overdamped, etc.) where most alpha lives.
    """
    return TestConfiguration(
        name="cross_regime_transitions",
        description="Study behavior during regime transitions to find hidden alpha",
        instruments=instruments,
        agent_type="cross_regime",
        agent_config={
            "regime_pairs": [
                ("laminar", "underdamped"),
                ("underdamped", "overdamped"),
                ("overdamped", "laminar"),
                ("critical", "underdamped"),
            ],
            "transition_window": "adaptive",
            "pre_transition_features": True,
            "use_gpu": True,
        },
        episodes=150,
        use_gpu=True,
    )


def create_cross_asset_suite(instruments: List[InstrumentSpec]) -> TestConfiguration:
    """
    Cross-Asset Transfer Learning: Learn from correlations we can't see.
    
    Train on one asset class, test on another. Discover universal patterns
    that transcend individual markets.
    """
    return TestConfiguration(
        name="cross_asset_transfer",
        description="Transfer learning across asset classes to find universal patterns",
        instruments=instruments,
        agent_type="cross_asset",
        agent_config={
            "transfer_pairs": [
                ("crypto", "forex"),
                ("forex", "metals"),
                ("metals", "indices"),
                ("indices", "commodities"),
            ],
            "shared_representation": True,
            "domain_adaptation": "adversarial",
            "use_gpu": True,
        },
        episodes=200,
        use_gpu=True,
    )


def create_multi_timeframe_fusion_suite(instruments: List[InstrumentSpec]) -> TestConfiguration:
    """
    Multi-Timeframe Fusion: Combine signals across timeframes.
    
    What if the alpha is in the RELATIONSHIP between timeframes,
    not in any single one?
    """
    return TestConfiguration(
        name="multi_timeframe_fusion",
        description="Fuse signals across multiple timeframes to discover temporal patterns",
        instruments=instruments,
        agent_type="mtf_fusion",
        agent_config={
            "timeframes": ["M15", "M30", "H1", "H4", "D1"],
            "fusion_method": "attention",  # Learn what timeframe to focus on
            "temporal_convolution": True,
            "fractal_analysis": True,
            "use_gpu": True,
        },
        episodes=200,
        use_gpu=True,
    )


def create_emergent_behavior_suite(instruments: List[InstrumentSpec]) -> TestConfiguration:
    """
    Emergent Behavior Detection: Find patterns that emerge from complexity.
    
    Use swarm intelligence, genetic algorithms, and evolutionary strategies
    to discover behaviors we wouldn't think to program.
    """
    return TestConfiguration(
        name="emergent_behavior",
        description="Evolutionary strategies to discover emergent trading behaviors",
        instruments=instruments,
        agent_type="emergent",
        agent_config={
            "algorithm": "ES",  # Evolution Strategies
            "population_size": 100,
            "mutation_rate": 0.1,
            "crossover": "uniform",
            "fitness": "sharpe_weighted_omega",
            "allow_mutations": ["features", "thresholds", "logic"],
            "use_gpu": True,
        },
        episodes=300,
        use_gpu=True,
    )


def create_adversarial_discovery_suite(instruments: List[InstrumentSpec]) -> TestConfiguration:
    """
    Adversarial Discovery: GAN-style approach to find market weaknesses.
    
    Generator tries to find profitable patterns, Discriminator tries to
    prove they're random. What survives is real alpha.
    """
    return TestConfiguration(
        name="adversarial_alpha_discovery",
        description="GAN-style adversarial learning to filter noise and find real alpha",
        instruments=instruments,
        agent_type="adversarial",
        agent_config={
            "generator": "RL_agent",
            "discriminator": "statistical_validator",
            "adversarial_loss": "wasserstein",
            "noise_filtering": True,
            "p_value_threshold": 0.01,
            "use_gpu": True,
        },
        episodes=250,
        use_gpu=True,
    )


def create_quantum_inspired_suite(instruments: List[InstrumentSpec]) -> TestConfiguration:
    """
    Quantum-Inspired Exploration: Superposition of strategies.
    
    Instead of choosing ONE strategy, maintain superposition of ALL
    possible strategies and collapse to the best one based on observation.
    """
    return TestConfiguration(
        name="quantum_inspired_superposition",
        description="Quantum-inspired superposition of multiple strategies simultaneously",
        instruments=instruments,
        agent_type="quantum_inspired",
        agent_config={
            "strategy_superposition": True,
            "num_strategies": 20,
            "collapse_function": "bayesian_update",
            "entanglement": "correlation_based",
            "measurement": "continuous",
            "use_gpu": True,
        },
        episodes=200,
        use_gpu=True,
    )


def create_chaos_theory_suite(instruments: List[InstrumentSpec]) -> TestConfiguration:
    """
    Chaos Theory Analysis: Find order in apparent randomness.
    
    Lyapunov exponents, strange attractors, fractal dimensions.
    Markets aren't random - they're chaotic. Different thing entirely.
    """
    return TestConfiguration(
        name="chaos_theory_analysis",
        description="Chaos theory to find deterministic patterns in apparent randomness",
        instruments=instruments,
        agent_type="chaos",
        agent_config={
            "lyapunov_horizon": "adaptive",
            "attractor_reconstruction": True,
            "fractal_dimension": "correlation",
            "recurrence_plots": True,
            "sensitivity_analysis": True,
            "use_gpu": True,
        },
        episodes=150,
        use_gpu=True,
    )


def create_information_theory_suite(instruments: List[InstrumentSpec]) -> TestConfiguration:
    """
    Information Theory: Mutual information, transfer entropy.
    
    What if alpha is in the INFORMATION FLOW between instruments,
    not in price movements themselves?
    """
    return TestConfiguration(
        name="information_theory",
        description="Information theory to measure information flow and dependencies",
        instruments=instruments,
        agent_type="info_theory",
        agent_config={
            "metrics": [
                "mutual_information",
                "transfer_entropy",
                "causality_detection",
                "granger_causality",
            ],
            "lag_analysis": True,
            "conditional_entropy": True,
            "use_gpu": True,
        },
        episodes=150,
        use_gpu=True,
    )


def create_combinatorial_explosion_suite(
    instruments: List[InstrumentSpec],
    max_feature_combinations: int = 1000
) -> TestConfiguration:
    """
    Combinatorial Explosion: Test MASSIVE combinations.
    
    We don't know which feature COMBINATION matters. Test them all.
    Use GPU to make this tractable.
    """
    return TestConfiguration(
        name="combinatorial_explosion",
        description=f"Test up to {max_feature_combinations} feature combinations systematically",
        instruments=instruments,
        agent_type="combinatorial",
        agent_config={
            "feature_pool": "all_64_dimensions",
            "combination_sizes": [2, 3, 4, 5],  # Test pairs, triplets, etc.
            "max_combinations": max_feature_combinations,
            "pruning": "statistical_significance",
            "parallel_evaluation": True,
            "use_gpu": True,
        },
        episodes=100,  # Per combination
        use_gpu=True,
    )


def create_deep_ensemble_suite(instruments: List[InstrumentSpec]) -> TestConfiguration:
    """
    Deep Ensemble: Stack everything we've learned.
    
    Combine ALL approaches: physics, RL, ML, chaos, information theory.
    The wisdom of crowds applied to trading strategies.
    """
    return TestConfiguration(
        name="deep_ensemble_all",
        description="Deep ensemble combining ALL discovered strategies and approaches",
        instruments=instruments,
        agent_type="deep_ensemble",
        agent_config={
            "base_models": [
                "physics", "PPO", "SAC", "A2C",
                "chaos_theory", "information_theory",
                "hidden_discovery", "meta_learning",
                "cross_regime", "emergent",
            ],
            "ensemble_method": "adaptive_weighting",
            "meta_learner": "gradient_boosting",
            "dynamic_rebalancing": True,
            "confidence_gating": True,
            "use_gpu": True,
        },
        episodes=200,
        use_gpu=True,
    )


# =============================================================================
# VISUALIZATION
# =============================================================================

class ResultVisualizer:
    """Generate beautiful plots and analysis."""
    
    def __init__(self, output_dir: str = "test_results/plots"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['font.size'] = 10
    
    def plot_comparison(
        self,
        results: List[Dict],
        metric: str = 'sharpe_ratio',
        title: Optional[str] = None
    ):
        """Plot comparison across test suites."""
        # Group by test name
        by_test = {}
        for r in results:
            name = r['config_name']
            if name not in by_test:
                by_test[name] = []
            by_test[name].append(r.get(metric, 0))
        
        # Create box plot
        fig, ax = plt.subplots()
        
        data = [values for values in by_test.values()]
        labels = list(by_test.keys())
        
        bp = ax.boxplot(data, labels=labels, patch_artist=True)
        
        # Color boxes
        colors = sns.color_palette("husl", len(labels))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
        
        ax.set_ylabel(metric.replace('_', ' ').title())
        ax.set_xlabel('Test Suite')
        
        if title is None:
            title = f'{metric.replace("_", " ").title()} Comparison'
        ax.set_title(title)
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        filename = f'comparison_{metric}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
        filepath = self.output_dir / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved plot: {filepath}")
    
    def plot_efficiency_metrics(self, results: List[Dict]):
        """Plot MFE/MAE and Pythagorean efficiency."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # MFE Captured
        by_test = {}
        for r in results:
            name = r['config_name']
            if name not in by_test:
                by_test[name] = {'mfe': [], 'mae': [], 'pyth': []}
            by_test[name]['mfe'].append(r.get('mfe_captured_pct', 0))
            by_test[name]['mae'].append(r.get('mae_ratio', 0))
            by_test[name]['pyth'].append(r.get('pythagorean_efficiency', 0))
        
        # MFE bar chart
        names = list(by_test.keys())
        mfe_means = [np.mean(by_test[n]['mfe']) for n in names]
        
        colors = sns.color_palette("husl", len(names))
        ax1.bar(range(len(names)), mfe_means, color=colors)
        ax1.set_xticks(range(len(names)))
        ax1.set_xticklabels(names, rotation=45, ha='right')
        ax1.set_ylabel('MFE Captured (%)')
        ax1.set_title('Maximum Favorable Excursion Captured')
        ax1.axhline(y=60, color='r', linestyle='--', label='Target (60%)')
        ax1.legend()
        
        # Pythagorean efficiency
        pyth_means = [np.mean(by_test[n]['pyth']) for n in names]
        
        ax2.bar(range(len(names)), pyth_means, color=colors)
        ax2.set_xticks(range(len(names)))
        ax2.set_xticklabels(names, rotation=45, ha='right')
        ax2.set_ylabel('Efficiency Ratio')
        ax2.set_title('Pythagorean Efficiency (Shortest Path / Actual Path)')
        
        plt.tight_layout()
        
        filename = f'efficiency_metrics_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
        filepath = self.output_dir / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved plot: {filepath}")
    
    def plot_gpu_utilization(self, results: List[Dict]):
        """Plot GPU utilization across tests."""
        by_test = {}
        for r in results:
            name = r['config_name']
            if name not in by_test:
                by_test[name] = []
            by_test[name].append(r.get('gpu_utilization_pct', 0))
        
        fig, ax = plt.subplots()
        
        names = list(by_test.keys())
        util_means = [np.mean(by_test[n]) for n in names]
        
        colors = sns.color_palette("husl", len(names))
        bars = ax.bar(range(len(names)), util_means, color=colors)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}%',
                   ha='center', va='bottom')
        
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=45, ha='right')
        ax.set_ylabel('GPU Utilization (%)')
        ax.set_title('Average GPU Utilization by Test Suite')
        ax.set_ylim(0, 100)
        ax.axhline(y=80, color='g', linestyle='--', label='Target (80%+)')
        ax.legend()
        
        plt.tight_layout()
        
        filename = f'gpu_utilization_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
        filepath = self.output_dir / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved plot: {filepath}")
    
    def plot_first_principles_validation(self, results: List[Dict]):
        """Plot first principles adherence."""
        by_test = {}
        for r in results:
            name = r['config_name']
            if name not in by_test:
                by_test[name] = {'non_linear': 0, 'asymmetric': 0, 'no_magic': 0, 'total': 0}
            
            by_test[name]['non_linear'] += int(r.get('is_non_linear', False))
            by_test[name]['asymmetric'] += int(r.get('is_asymmetric', False))
            by_test[name]['no_magic'] += int(not r.get('uses_magic_numbers', True))
            by_test[name]['total'] += 1
        
        # Calculate percentages
        names = list(by_test.keys())
        non_linear_pct = [(by_test[n]['non_linear'] / by_test[n]['total']) * 100 for n in names]
        asymmetric_pct = [(by_test[n]['asymmetric'] / by_test[n]['total']) * 100 for n in names]
        no_magic_pct = [(by_test[n]['no_magic'] / by_test[n]['total']) * 100 for n in names]
        
        # Stacked bar chart
        fig, ax = plt.subplots()
        
        x = np.arange(len(names))
        width = 0.25
        
        ax.bar(x - width, non_linear_pct, width, label='Non-linear', color='#ff9999')
        ax.bar(x, asymmetric_pct, width, label='Asymmetric', color='#66b3ff')
        ax.bar(x + width, no_magic_pct, width, label='No Magic Numbers', color='#99ff99')
        
        ax.set_ylabel('Adherence (%)')
        ax.set_title('First Principles Validation')
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=45, ha='right')
        ax.legend()
        ax.set_ylim(0, 100)
        
        plt.tight_layout()
        
        filename = f'first_principles_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
        filepath = self.output_dir / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved plot: {filepath}")
    
    def generate_all_plots(self, results: List[Dict]):
        """Generate all plots."""
        print("\nGenerating plots...")
        
        self.plot_comparison(results, 'sharpe_ratio')
        self.plot_comparison(results, 'omega_ratio')
        self.plot_comparison(results, 'max_drawdown', 'Maximum Drawdown Comparison')
        self.plot_efficiency_metrics(results)
        self.plot_gpu_utilization(results)
        self.plot_first_principles_validation(results)
        
        print("All plots generated!")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Unified Testing Framework for Kinetra',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run all tests
    python scripts/unified_test_framework.py --full
    
    # Quick test (fewer episodes)
    python scripts/unified_test_framework.py --quick
    
    # Specific suite
    python scripts/unified_test_framework.py --suite control
    
    # Compare suites
    python scripts/unified_test_framework.py --compare control physics rl
    
    # Custom instruments
    python scripts/unified_test_framework.py --asset-classes crypto forex --timeframes H1 H4
        """
    )
    
    parser.add_argument('--full', action='store_true',
                       help='Run full test suite (all combinations)')
    parser.add_argument('--extreme', action='store_true',
                       help='EXTREME mode: Run ALL possible combinations (hidden dimensions, meta-learning, chaos, etc.)')
    parser.add_argument('--quick', action='store_true',
                       help='Quick test (fewer episodes, limited instruments)')
    parser.add_argument('--suite', type=str, 
                       choices=['control', 'physics', 'rl', 'specialization', 'stacking', 'triad',
                               'hidden', 'meta', 'cross_regime', 'cross_asset', 'mtf', 'emergent',
                               'adversarial', 'quantum', 'chaos', 'info_theory', 'combinatorial', 'deep_ensemble'],
                       help='Run specific test suite')
    parser.add_argument('--compare', nargs='+',
                       help='Compare multiple suites')
    parser.add_argument('--asset-classes', nargs='+',
                       choices=['crypto', 'forex', 'metals', 'commodities', 'indices'],
                       help='Filter by asset classes')
    parser.add_argument('--timeframes', nargs='+',
                       help='Filter by timeframes (e.g., M15 H1 H4)')
    parser.add_argument('--max-instruments', type=int, default=3,
                       help='Max instruments per asset class (default: 3)')
    parser.add_argument('--output-dir', type=str, default='test_results',
                       help='Output directory for results')
    parser.add_argument('--no-plots', action='store_true',
                       help='Skip plot generation')
    parser.add_argument('--episodes', type=int,
                       help='Override number of episodes')
    
    args = parser.parse_args()
    
    # Discover instruments
    print("Discovering instruments...")
    instruments = discover_instruments(
        asset_classes=args.asset_classes,
        timeframes=args.timeframes,
        max_per_class=args.max_instruments if not args.quick else 1
    )
    
    if not instruments:
        print("ERROR: No instruments found!")
        return
    
    print(f"Found {len(instruments)} instruments:")
    by_class = {}
    for inst in instruments:
        if inst.asset_class not in by_class:
            by_class[inst.asset_class] = []
        by_class[inst.asset_class].append(f"{inst.symbol}_{inst.timeframe}")
    
    for cls, syms in by_class.items():
        print(f"  {cls}: {', '.join(syms)}")
    
    # Initialize framework
    framework = TestingFramework(output_dir=args.output_dir)
    
    # Add tests based on arguments
    if args.extreme:
        print("\n" + "="*80)
        print("EXTREME MODE: Testing ALL combinations - 'We don't know what we don't know'")
        print("="*80 + "\n")
        
        # Core baselines
        framework.add_test(create_control_suite(instruments))
        framework.add_test(create_physics_suite(instruments))
        
        # All RL variants
        for agent in ["PPO", "SAC", "A2C"]:
            framework.add_test(create_rl_suite(instruments, agent))
        
        # All specializations
        for spec_type in ["asset_class", "regime", "timeframe"]:
            framework.add_test(create_specialization_suite(instruments, spec_type))
        
        # Stacking and ensembles
        framework.add_test(create_stacking_suite(instruments))
        framework.add_test(create_deep_ensemble_suite(instruments))
        
        # All triad roles
        for role in ["trader", "risk_manager", "portfolio_manager"]:
            framework.add_test(create_triad_suite(instruments, role))
        
        # Discovery methods - "What we don't know"
        framework.add_test(create_hidden_dimension_suite(instruments))
        framework.add_test(create_meta_learning_suite(instruments))
        framework.add_test(create_emergent_behavior_suite(instruments))
        framework.add_test(create_adversarial_discovery_suite(instruments))
        
        # Cross-domain analysis - "What we can't see"
        framework.add_test(create_cross_regime_suite(instruments))
        framework.add_test(create_cross_asset_suite(instruments))
        framework.add_test(create_multi_timeframe_fusion_suite(instruments))
        
        # Advanced theoretical approaches
        framework.add_test(create_quantum_inspired_suite(instruments))
        framework.add_test(create_chaos_theory_suite(instruments))
        framework.add_test(create_information_theory_suite(instruments))
        
        # Combinatorial explosion
        framework.add_test(create_combinatorial_explosion_suite(instruments, max_feature_combinations=1000))
        
        print(f"Total test configurations: {len(framework.tests)}")
        print("This will take significant time and GPU resources!")
        
    elif args.full:
        print("\nRunning FULL test suite...")
        framework.add_test(create_control_suite(instruments))
        framework.add_test(create_physics_suite(instruments))
        framework.add_test(create_rl_suite(instruments, "PPO"))
        framework.add_test(create_rl_suite(instruments, "SAC"))
        framework.add_test(create_rl_suite(instruments, "A2C"))
        framework.add_test(create_specialization_suite(instruments, "asset_class"))
        framework.add_test(create_specialization_suite(instruments, "regime"))
        framework.add_test(create_specialization_suite(instruments, "timeframe"))
        framework.add_test(create_stacking_suite(instruments))
        framework.add_test(create_triad_suite(instruments, "trader"))
        framework.add_test(create_triad_suite(instruments, "risk_manager"))
        framework.add_test(create_triad_suite(instruments, "portfolio_manager"))
    
    elif args.quick:
        print("\nRunning QUICK test suite...")
        # Reduce episodes
        config = create_control_suite(instruments[:2])  # Only 2 instruments
        config.episodes = 10
        framework.add_test(config)
        
        config = create_physics_suite(instruments[:2])
        config.episodes = 10
        framework.add_test(config)
    
    elif args.suite:
        print(f"\nRunning {args.suite} suite...")
        suite_map = {
            'control': lambda: create_control_suite(instruments),
            'physics': lambda: create_physics_suite(instruments),
            'rl': lambda: create_rl_suite(instruments, "PPO"),
            'specialization': lambda: create_specialization_suite(instruments, "asset_class"),
            'stacking': lambda: create_stacking_suite(instruments),
            'triad': lambda: create_triad_suite(instruments, "trader"),
            'hidden': lambda: create_hidden_dimension_suite(instruments),
            'meta': lambda: create_meta_learning_suite(instruments),
            'cross_regime': lambda: create_cross_regime_suite(instruments),
            'cross_asset': lambda: create_cross_asset_suite(instruments),
            'mtf': lambda: create_multi_timeframe_fusion_suite(instruments),
            'emergent': lambda: create_emergent_behavior_suite(instruments),
            'adversarial': lambda: create_adversarial_discovery_suite(instruments),
            'quantum': lambda: create_quantum_inspired_suite(instruments),
            'chaos': lambda: create_chaos_theory_suite(instruments),
            'info_theory': lambda: create_information_theory_suite(instruments),
            'combinatorial': lambda: create_combinatorial_explosion_suite(instruments),
            'deep_ensemble': lambda: create_deep_ensemble_suite(instruments),
        }
        
        if args.suite in suite_map:
            framework.add_test(suite_map[args.suite]())
        else:
            print(f"Unknown suite: {args.suite}")
            return
    
    elif args.compare:
        print(f"\nComparing suites: {', '.join(args.compare)}...")
        for suite_name in args.compare:
            if suite_name == 'control':
                framework.add_test(create_control_suite(instruments))
            elif suite_name == 'physics':
                framework.add_test(create_physics_suite(instruments))
            elif suite_name == 'rl':
                framework.add_test(create_rl_suite(instruments, "PPO"))
            elif suite_name == 'ml':
                framework.add_test(create_rl_suite(instruments, "SAC"))
            elif suite_name == 'hidden':
                framework.add_test(create_hidden_dimension_suite(instruments))
            elif suite_name == 'chaos':
                framework.add_test(create_chaos_theory_suite(instruments))
            elif suite_name == 'quantum':
                framework.add_test(create_quantum_inspired_suite(instruments))
            else:
                print(f"Warning: Unknown suite '{suite_name}' in comparison")
    
    else:
        print("No test specified. Use --full, --quick, --suite, or --compare")
        parser.print_help()
        return
    
    # Override episodes if specified
    if args.episodes:
        for test in framework.tests:
            test.episodes = args.episodes
    
    # Run tests
    print(f"\n{'='*80}")
    print("STARTING TEST RUNS")
    print(f"{'='*80}\n")
    
    results = framework.run_all_tests()
    
    # Save results
    framework.save_results()
    
    # Generate report
    framework.generate_report(results)
    
    # Generate plots
    if not args.no_plots and results:
        visualizer = ResultVisualizer(output_dir=f"{args.output_dir}/plots")
        
        # Convert results to dicts for plotting
        results_dicts = []
        for r in results:
            from dataclasses import asdict
            results_dicts.append(asdict(r))
        
        visualizer.generate_all_plots(results_dicts)
    
    print(f"\n{'='*80}")
    print("ALL TESTS COMPLETE")
    print(f"Results saved to: {args.output_dir}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
