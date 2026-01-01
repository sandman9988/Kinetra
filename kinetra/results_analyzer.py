"""
Results Analyzer
================

Analyze and visualize test results from the testing framework.

Features:
- Load and parse test results from JSON
- Statistical comparison across suites
- Visualization plots (Sharpe, Omega, win rate, p-values)
- Winner identification with significance testing

Philosophy: Let data decide, not opinions.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

logger = logging.getLogger(__name__)

# Set plot style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 10)


class ResultsAnalyzer:
    """
    Analyze and visualize test results.
    
    Loads results from JSON files, performs statistical tests,
    and generates comparison plots.
    """
    
    def __init__(self, results_dir: str = "test_results"):
        """
        Args:
            results_dir: Directory containing test result JSON files
        """
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"ResultsAnalyzer initialized with results_dir={results_dir}")
    
    def load_latest_results(self, pattern: str = "test_*.json") -> List[Dict]:
        """
        Load most recent test results.
        
        Args:
            pattern: File pattern to match
            
        Returns:
            List of result dictionaries
        """
        result_files = sorted(self.results_dir.glob(pattern))
        
        if not result_files:
            logger.warning(f"No test results found matching '{pattern}'")
            return []
        
        all_results = []
        for result_file in result_files:
            try:
                with open(result_file) as f:
                    data = json.load(f)
                    all_results.append(data)
                logger.info(f"Loaded results from {result_file.name}")
            except Exception as e:
                logger.error(f"Failed to load {result_file}: {e}")
        
        return all_results
    
    def load_suite_results(self, suite_name: str) -> pd.DataFrame:
        """
        Load results for a specific test suite.
        
        Args:
            suite_name: Name of the test suite
            
        Returns:
            DataFrame with suite results
        """
        pattern = f"*{suite_name}*.json"
        results = self.load_latest_results(pattern)
        
        if not results:
            logger.warning(f"No results found for suite '{suite_name}'")
            return pd.DataFrame()
        
        # Convert to DataFrame
        rows = []
        for result in results:
            if isinstance(result, dict) and 'metrics' in result:
                row = result['metrics'].copy()
                row['suite'] = suite_name
                row['timestamp'] = result.get('timestamp', '')
                rows.append(row)
        
        return pd.DataFrame(rows)
    
    def compare_suites(self, suite_names: List[str]) -> pd.DataFrame:
        """
        Compare performance across test suites.
        
        Args:
            suite_names: List of suite names to compare
            
        Returns:
            DataFrame with comparison metrics
        """
        logger.info(f"Comparing suites: {suite_names}")
        
        results = []
        for suite in suite_names:
            suite_results = self.load_suite_results(suite)
            
            if suite_results.empty:
                logger.warning(f"No results for suite '{suite}', using placeholder")
                # Add placeholder data
                metrics = {
                    'suite': suite,
                    'sharpe_mean': 0.0,
                    'sharpe_std': 0.0,
                    'omega_mean': 0.0,
                    'omega_std': 0.0,
                    'win_rate': 0.0,
                    'p_value': 1.0,
                    'n_samples': 0
                }
            else:
                # Calculate aggregate metrics
                sharpe_values = suite_results.get('sharpe', pd.Series([0]))
                omega_values = suite_results.get('omega', pd.Series([0]))
                win_rate_values = suite_results.get('win_rate', pd.Series([0]))
                
                metrics = {
                    'suite': suite,
                    'sharpe_mean': sharpe_values.mean(),
                    'sharpe_std': sharpe_values.std(),
                    'omega_mean': omega_values.mean(),
                    'omega_std': omega_values.std(),
                    'win_rate': win_rate_values.mean(),
                    'p_value': self._calculate_significance(sharpe_values),
                    'n_samples': len(suite_results)
                }
            
            results.append(metrics)
            logger.info(f"  {suite}: Sharpe={metrics['sharpe_mean']:.3f}, p={metrics['p_value']:.4f}")
        
        return pd.DataFrame(results)
    
    def _calculate_significance(self, values: pd.Series) -> float:
        """
        Test if results are statistically significant.
        
        Uses one-sample t-test against zero (no edge).
        
        Args:
            values: Series of metric values (e.g., Sharpe ratios)
            
        Returns:
            p-value
        """
        if len(values) < 2:
            return 1.0
        
        # One-sample t-test: is mean significantly different from 0?
        t_stat, p_value = stats.ttest_1samp(values, 0)
        
        return float(p_value)
    
    def plot_comparison(
        self,
        comparison_df: pd.DataFrame,
        output_path: Optional[str] = None
    ) -> str:
        """
        Plot suite comparison.
        
        Args:
            comparison_df: DataFrame from compare_suites()
            output_path: Path to save plot (default: test_results/comparison.png)
            
        Returns:
            Path to saved plot
        """
        if output_path is None:
            output_path = str(self.results_dir / "comparison.png")
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Sharpe ratio comparison
        axes[0, 0].bar(comparison_df['suite'], comparison_df['sharpe_mean'], color='steelblue')
        axes[0, 0].errorbar(
            range(len(comparison_df)),
            comparison_df['sharpe_mean'],
            yerr=comparison_df['sharpe_std'],
            fmt='none',
            color='black',
            capsize=5
        )
        axes[0, 0].set_title('Sharpe Ratio by Suite', fontsize=14, fontweight='bold')
        axes[0, 0].set_ylabel('Sharpe Ratio')
        axes[0, 0].axhline(y=0, color='red', linestyle='--', alpha=0.3, label='No Edge')
        axes[0, 0].legend()
        axes[0, 0].grid(axis='y', alpha=0.3)
        
        # Omega ratio comparison
        axes[0, 1].bar(comparison_df['suite'], comparison_df['omega_mean'], color='darkorange')
        axes[0, 1].errorbar(
            range(len(comparison_df)),
            comparison_df['omega_mean'],
            yerr=comparison_df['omega_std'],
            fmt='none',
            color='black',
            capsize=5
        )
        axes[0, 1].set_title('Omega Ratio by Suite', fontsize=14, fontweight='bold')
        axes[0, 1].set_ylabel('Omega Ratio')
        axes[0, 1].axhline(y=1, color='red', linestyle='--', alpha=0.3, label='No Edge')
        axes[0, 1].legend()
        axes[0, 1].grid(axis='y', alpha=0.3)
        
        # Win rate
        axes[1, 0].bar(comparison_df['suite'], comparison_df['win_rate'], color='seagreen')
        axes[1, 0].set_title('Win Rate by Suite', fontsize=14, fontweight='bold')
        axes[1, 0].set_ylabel('Win Rate')
        axes[1, 0].axhline(y=0.5, color='red', linestyle='--', alpha=0.3, label='Random')
        axes[1, 0].set_ylim([0, 1])
        axes[1, 0].legend()
        axes[1, 0].grid(axis='y', alpha=0.3)
        
        # Statistical significance
        neg_log_p = -np.log10(comparison_df['p_value'].clip(lower=1e-10))
        bars = axes[1, 1].bar(comparison_df['suite'], neg_log_p, color='purple')
        
        # Color bars by significance
        for i, (bar, p_val) in enumerate(zip(bars, comparison_df['p_value'])):
            if p_val < 0.01:
                bar.set_color('darkgreen')
            elif p_val < 0.05:
                bar.set_color('orange')
            else:
                bar.set_color('gray')
        
        axes[1, 1].axhline(
            y=-np.log10(0.01),
            color='darkgreen',
            linestyle='--',
            label='p=0.01 (highly significant)',
            linewidth=2
        )
        axes[1, 1].axhline(
            y=-np.log10(0.05),
            color='orange',
            linestyle='--',
            label='p=0.05 (significant)',
            linewidth=2
        )
        axes[1, 1].set_title('Statistical Significance', fontsize=14, fontweight='bold')
        axes[1, 1].set_ylabel('-log10(p-value)')
        axes[1, 1].legend()
        axes[1, 1].grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"✅ Comparison plot saved to {output_path}")
        return output_path
    
    def identify_winner(self, comparison_df: pd.DataFrame) -> Dict:
        """
        Identify the winning suite based on Sharpe ratio and significance.
        
        Args:
            comparison_df: DataFrame from compare_suites()
            
        Returns:
            Dictionary with winner info
        """
        # Filter to significant results (p < 0.05)
        significant = comparison_df[comparison_df['p_value'] < 0.05]
        
        if significant.empty:
            logger.warning("No statistically significant results found")
            return {
                'winner': 'None',
                'sharpe': 0.0,
                'p_value': 1.0,
                'message': 'No suite achieved statistical significance (p < 0.05)'
            }
        
        # Among significant, find best Sharpe
        winner_idx = significant['sharpe_mean'].idxmax()
        winner = significant.loc[winner_idx]
        
        return {
            'winner': winner['suite'],
            'sharpe': winner['sharpe_mean'],
            'omega': winner['omega_mean'],
            'win_rate': winner['win_rate'],
            'p_value': winner['p_value'],
            'n_samples': winner['n_samples'],
            'message': f"Winner: {winner['suite']} with Sharpe {winner['sharpe_mean']:.3f} (p={winner['p_value']:.4f})"
        }
    
    def generate_report(
        self,
        suite_names: List[str],
        output_file: Optional[str] = None
    ) -> str:
        """
        Generate complete comparison report.
        
        Args:
            suite_names: Suites to compare
            output_file: Path to save report (default: test_results/report.txt)
            
        Returns:
            Path to saved report
        """
        if output_file is None:
            output_file = str(self.results_dir / "report.txt")
        
        # Compare suites
        comparison = self.compare_suites(suite_names)
        
        # Identify winner
        winner_info = self.identify_winner(comparison)
        
        # Generate plot
        plot_path = self.plot_comparison(comparison)
        
        # Write report
        with open(output_file, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("KINETRA TEST SUITE COMPARISON REPORT\n")
            f.write("=" * 80 + "\n\n")
            
            f.write("COMPARISON SUMMARY\n")
            f.write("-" * 80 + "\n")
            f.write(comparison.to_string(index=False))
            f.write("\n\n")
            
            f.write("WINNER\n")
            f.write("-" * 80 + "\n")
            f.write(winner_info['message'] + "\n")
            f.write(f"  Sharpe Ratio: {winner_info['sharpe']:.3f}\n")
            f.write(f"  Omega Ratio: {winner_info['omega']:.3f}\n")
            f.write(f"  Win Rate: {winner_info['win_rate']:.2%}\n")
            f.write(f"  P-value: {winner_info['p_value']:.4f}\n")
            f.write(f"  Sample Size: {winner_info['n_samples']}\n")
            f.write("\n")
            
            f.write("VISUALIZATION\n")
            f.write("-" * 80 + "\n")
            f.write(f"Comparison plot: {plot_path}\n")
            f.write("\n")
            
            f.write("=" * 80 + "\n")
        
        logger.info(f"✅ Report saved to {output_file}")
        return output_file


# Quick test
if __name__ == "__main__":
    print("\n=== Results Analyzer Test ===\n")
    
    # Create analyzer
    analyzer = ResultsAnalyzer(results_dir="test_results")
    print(f"Analyzer created with results_dir: {analyzer.results_dir}")
    
    # Create dummy results for testing
    dummy_results = [
        {
            'suite': 'control',
            'metrics': {'sharpe': 0.5, 'omega': 1.2, 'win_rate': 0.48}
        },
        {
            'suite': 'physics',
            'metrics': {'sharpe': 1.2, 'omega': 2.1, 'win_rate': 0.58}
        },
        {
            'suite': 'rl',
            'metrics': {'sharpe': 1.5, 'omega': 2.5, 'win_rate': 0.62}
        }
    ]
    
    # Compare suites
    comparison = analyzer.compare_suites(['control', 'physics', 'rl'])
    print("\nComparison:")
    print(comparison.to_string(index=False))
    
    # Identify winner
    winner = analyzer.identify_winner(comparison)
    print(f"\n{winner['message']}")
    
    print("\n✅ Results Analyzer test completed!")
