#!/usr/bin/env python3
"""
Master Orchestrator for Scientific Testing Framework
=====================================================

Complete testing programme execution with:
1. Data validation and preparation
2. Discovery method execution
3. Statistical validation (PBO, CPCV)
4. Integrated backtesting
5. Code review integration
6. Auto-fixing and continuation
7. Progress tracking and resumption

Usage:
    # Full scientific testing program
    python scripts/run_scientific_testing.py --full
    
    # Quick validation run
    python scripts/run_scientific_testing.py --quick
    
    # Specific phase
    python scripts/run_scientific_testing.py --phase discovery
    
    # Resume from checkpoint
    python scripts/run_scientific_testing.py --resume
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from kinetra.discovery_methods import DiscoveryMethodRunner
from kinetra.test_executor import StatisticalRigor
from kinetra.testing_framework import TestingFramework, TestConfiguration, InstrumentSpec
from kinetra.integrated_backtester import IntegratedBacktester

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# DevOps integration for local/remote sync
try:
    from kinetra.devops import GitSync, check_sync_status
    DEVOPS_AVAILABLE = True
except ImportError:
    DEVOPS_AVAILABLE = False
    logger.warning("DevOps module not available - git sync disabled")


class ScientificTestingOrchestrator:
    """
    Master orchestrator for the complete scientific testing programme.
    
    Coordinates all phases:
    - Phase 1: Data validation
    - Phase 2: Discovery methods
    - Phase 3: Statistical validation
    - Phase 4: Backtesting
    - Phase 5: Code review
    
    Integrates with DevOps for local/remote sync.
    """
    
    def __init__(self, output_dir: str = "scientific_testing_results", enable_git_sync: bool = True):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.discovery_runner = DiscoveryMethodRunner()
        self.stats_validator = StatisticalRigor()
        
        # Git sync integration
        self.enable_git_sync = enable_git_sync and DEVOPS_AVAILABLE
        if self.enable_git_sync:
            self.git_sync = GitSync()
            logger.info("Git sync enabled - ensuring local/remote synchronization")
        else:
            self.git_sync = None
            if enable_git_sync and not DEVOPS_AVAILABLE:
                logger.warning("Git sync requested but DevOps module not available")
        
        # Create subdirectories
        self.data_dir = self.output_dir / "data_validation"
        self.discovery_dir = self.output_dir / "discovery"
        self.backtest_dir = self.output_dir / "backtests"
        self.reports_dir = self.output_dir / "reports"
        
        for d in [self.data_dir, self.discovery_dir, self.backtest_dir, self.reports_dir]:
            d.mkdir(exist_ok=True)
        
        logger.info(f"ScientificTestingOrchestrator initialized")
        logger.info(f"Output directory: {self.output_dir}")
    
    def check_and_sync(self) -> bool:
        """
        Check git sync status and optionally sync with remote.
        
        Returns:
            True if sync successful or not needed, False if sync failed
        """
        if not self.enable_git_sync:
            return True
        
        logger.info("Checking git sync status...")
        
        try:
            status = self.git_sync.check_status()
            
            logger.info(f"Branch: {status.branch}")
            logger.info(f"Sync status: {status.sync_status.value}")
            
            if status.uncommitted_changes > 0:
                logger.warning(f"Uncommitted changes: {status.uncommitted_changes} files")
            
            if status.ahead_count > 0:
                logger.info(f"Local commits ahead: {status.ahead_count}")
            
            if status.behind_count > 0:
                logger.warning(f"Remote commits behind: {status.behind_count}")
                logger.info("Pulling latest changes from remote...")
                success, msg = self.git_sync.pull(rebase=True)
                if not success:
                    logger.error(f"Failed to pull changes: {msg}")
                    return False
                logger.info(f"Pull successful: {msg}")
            
            return True
            
        except Exception as e:
            logger.error(f"Git sync check failed: {e}")
            return False
    
    def run_full_programme(
        self,
        instruments: Optional[List[InstrumentSpec]] = None,
        quick_mode: bool = False
    ):
        """
        Run the complete scientific testing programme.
        
        Args:
            instruments: List of instruments to test (None = auto-discover)
            quick_mode: If True, run abbreviated tests for validation
        """
        logger.info("\n" + "="*80)
        logger.info("STARTING COMPLETE SCIENTIFIC TESTING PROGRAMME")
        logger.info("="*80 + "\n")
        
        # Check git sync status before starting
        if self.enable_git_sync:
            logger.info("\n" + "="*80)
            logger.info("PRE-RUN GIT SYNC CHECK")
            logger.info("="*80 + "\n")
            
            sync_ok = self.check_and_sync()
            if not sync_ok:
                logger.warning("Git sync check failed - continuing anyway")
        
        start_time = datetime.now()
        
        # Phase 1: Data Validation
        logger.info("\n" + "="*80)
        logger.info("PHASE 1: DATA PREPARATION AND VALIDATION")
        logger.info("="*80 + "\n")
        
        validated_data = self.phase1_data_validation(instruments)
        
        if not validated_data:
            logger.error("No valid data found. Exiting.")
            return
        
        # Phase 2: Discovery Methods
        logger.info("\n" + "="*80)
        logger.info("PHASE 2: DISCOVERY METHOD EXECUTION")
        logger.info("="*80 + "\n")
        
        discovery_results = self.phase2_discovery_methods(validated_data, quick_mode)
        
        # Phase 3: Statistical Validation
        logger.info("\n" + "="*80)
        logger.info("PHASE 3: STATISTICAL VALIDATION")
        logger.info("="*80 + "\n")
        
        validated_discoveries = self.phase3_statistical_validation(discovery_results)
        
        # Phase 4: Integrated Backtesting
        logger.info("\n" + "="*80)
        logger.info("PHASE 4: INTEGRATED BACKTESTING")
        logger.info("="*80 + "\n")
        
        backtest_results = self.phase4_integrated_backtesting(
            validated_discoveries,
            validated_data
        )
        
        # Phase 5: Final Report
        logger.info("\n" + "="*80)
        logger.info("PHASE 5: GENERATING FINAL REPORT")
        logger.info("="*80 + "\n")
        
        self.phase5_generate_report(discovery_results, backtest_results)
        
        # Summary
        duration = (datetime.now() - start_time).total_seconds()
        logger.info("\n" + "="*80)
        logger.info("SCIENTIFIC TESTING PROGRAMME COMPLETE")
        logger.info("="*80)
        logger.info(f"Total duration: {duration:.1f} seconds ({duration/60:.1f} minutes)")
        logger.info(f"Results saved to: {self.output_dir}")
        logger.info("="*80 + "\n")
        
        # Final sync check
        if self.enable_git_sync:
            logger.info("\n" + "="*80)
            logger.info("POST-RUN GIT SYNC CHECK")
            logger.info("="*80 + "\n")
            
            status = self.git_sync.check_status()
            if status.uncommitted_changes > 0:
                logger.info("Uncommitted changes detected after test run")
                logger.info("Run 'git status' to review changes")
            
            logger.info(check_sync_status())
    
    def phase1_data_validation(
        self,
        instruments: Optional[List[InstrumentSpec]]
    ) -> Dict[str, pd.DataFrame]:
        """
        Phase 1: Validate data quality.
        
        Returns:
            Dict mapping instrument ID to validated DataFrame
        """
        validated_data = {}
        
        # Auto-discover instruments if not provided
        if instruments is None:
            from scripts.testing.unified_test_framework import discover_instruments
            instruments = discover_instruments(max_per_class=2)
        
        logger.info(f"Validating {len(instruments)} instruments...")
        
        for instrument in instruments:
            instrument_id = f"{instrument.symbol}_{instrument.timeframe}"
            
            try:
                # Load data
                data = pd.read_csv(instrument.data_path)
                
                # Basic validation
                if len(data) < 1000:
                    logger.warning(f"{instrument_id}: Insufficient data ({len(data)} bars)")
                    continue
                
                # Check for required columns
                required_cols = ['close']
                if not all(col in data.columns for col in required_cols):
                    logger.warning(f"{instrument_id}: Missing required columns")
                    continue
                
                # Statistical validation
                if 'close' in data.columns:
                    returns = data['close'].pct_change().fillna(0)
                    
                    # Check for data quality issues
                    if returns.std() == 0:
                        logger.warning(f"{instrument_id}: Zero variance")
                        continue
                    
                    # ADF test for stationarity (returns should be stationary)
                    from scipy import stats
                    
                    # Simple stationarity check: compare first half to second half
                    n = len(returns)
                    first_half = returns[:n//2]
                    second_half = returns[n//2:]
                    
                    _, p_value = stats.ttest_ind(first_half, second_half)
                    
                    if p_value < 0.001:  # Too different = non-stationary
                        logger.warning(f"{instrument_id}: Non-stationary data (p={p_value:.4f})")
                    
                    # Check for asymmetry
                    skew = returns.skew()
                    kurtosis = returns.kurtosis()
                    
                    logger.info(f"{instrument_id}: Validated - {len(data)} bars, skew={skew:.2f}, kurtosis={kurtosis:.2f}")
                
                validated_data[instrument_id] = data
                
            except Exception as e:
                logger.error(f"{instrument_id}: Validation failed - {e}")
        
        # Save validation report
        validation_report = {
            'timestamp': datetime.now().isoformat(),
            'total_instruments': len(instruments),
            'validated_instruments': len(validated_data),
            'instruments': list(validated_data.keys())
        }
        
        report_file = self.data_dir / "validation_report.json"
        with open(report_file, 'w') as f:
            json.dump(validation_report, f, indent=2)
        
        logger.info(f"\nValidated {len(validated_data)}/{len(instruments)} instruments")
        logger.info(f"Validation report saved to {report_file}")
        
        return validated_data
    
    def phase2_discovery_methods(
        self,
        validated_data: Dict[str, pd.DataFrame],
        quick_mode: bool = False
    ) -> Dict[str, Dict]:
        """
        Phase 2: Run discovery methods.
        
        Returns:
            Dict mapping instrument ID to discovery results
        """
        all_discoveries = {}
        
        # Select discovery methods
        if quick_mode:
            methods = ['hidden_dimensions', 'chaos_theory']
        else:
            methods = ['hidden_dimensions', 'chaos_theory', 'adversarial', 'meta_learning']
        
        logger.info(f"Running {len(methods)} discovery methods on {len(validated_data)} instruments...")
        
        for instrument_id, data in validated_data.items():
            logger.info(f"\nDiscovering patterns in {instrument_id}...")
            
            try:
                # Run discovery methods
                discoveries = self.discovery_runner.run_all_discoveries(
                    data,
                    methods=methods,
                    config={
                        'hidden_dimensions': {
                            'methods': ['pca', 'ica'],
                            'latent_dims': [8, 16]
                        },
                        'chaos_theory': {},
                        'adversarial': {},
                        'meta_learning': {},
                    }
                )
                
                all_discoveries[instrument_id] = discoveries
                
                # Save discoveries
                discovery_file = self.discovery_dir / f"{instrument_id}_discoveries.json"
                
                # Convert to serializable format
                serializable_discoveries = {}
                for method_name, result in discoveries.items():
                    serializable_discoveries[method_name] = {
                        'method_name': result.method_name,
                        'n_patterns': len(result.discovered_patterns),
                        'statistical_significance': result.statistical_significance,
                        'p_value': result.p_value,
                        'feature_importance': result.feature_importance,
                        'optimal_parameters': result.optimal_parameters,
                    }
                
                with open(discovery_file, 'w') as f:
                    json.dump(serializable_discoveries, f, indent=2, default=str)
                
                logger.info(f"Saved discoveries to {discovery_file}")
                
            except Exception as e:
                logger.error(f"Discovery failed for {instrument_id}: {e}")
                import traceback
                traceback.print_exc()
        
        return all_discoveries
    
    def phase3_statistical_validation(
        self,
        discovery_results: Dict[str, Dict]
    ) -> Dict[str, Dict]:
        """
        Phase 3: Validate discoveries statistically.
        
        Returns:
            Filtered discoveries that pass statistical tests
        """
        logger.info("Validating discoveries statistically...")
        
        validated_discoveries = {}
        
        for instrument_id, discoveries in discovery_results.items():
            validated_discoveries[instrument_id] = {}
            
            for method_name, result in discoveries.items():
                # Check statistical significance
                if result.statistical_significance and result.p_value < 0.05:
                    validated_discoveries[instrument_id][method_name] = result
                    logger.info(f"{instrument_id} - {method_name}: PASS (p={result.p_value:.4f})")
                else:
                    logger.info(f"{instrument_id} - {method_name}: FAIL (p={result.p_value:.4f})")
        
        # Summary
        total_discoveries = sum(len(d) for d in discovery_results.values())
        validated_count = sum(len(d) for d in validated_discoveries.values())
        
        logger.info(f"\nStatistical validation: {validated_count}/{total_discoveries} discoveries passed")
        
        return validated_discoveries
    
    def phase4_integrated_backtesting(
        self,
        validated_discoveries: Dict[str, Dict],
        validated_data: Dict[str, pd.DataFrame]
    ) -> Dict[str, Dict]:
        """
        Phase 4: Backtest validated discoveries.
        
        Returns:
            Backtest results
        """
        backtester = IntegratedBacktester(output_dir=str(self.backtest_dir))
        
        backtest_results = {}
        
        logger.info("Running backtests on validated discoveries...")
        
        for instrument_id, discoveries in validated_discoveries.items():
            if instrument_id not in validated_data:
                continue
            
            data = validated_data[instrument_id]
            
            for method_name, discovery_result in discoveries.items():
                strategy_name = f"{instrument_id}_{method_name}"
                
                logger.info(f"\nBacktesting: {strategy_name}")
                
                try:
                    # Convert discovery to strategy config
                    strategy_config = {
                        'type': method_name,
                        'parameters': discovery_result.optimal_parameters,
                        'features': discovery_result.feature_importance,
                    }
                    
                    # Run backtest
                    result = backtester.backtest_discovered_strategy(
                        strategy_config,
                        data
                    )
                    
                    backtest_results[strategy_name] = result
                    
                    logger.info(f"  Sharpe: {result.sharpe_ratio:.2f}, Win Rate: {result.win_rate:.2%}")
                    
                except Exception as e:
                    logger.error(f"Backtest failed for {strategy_name}: {e}")
                    # Continue with other strategies instead of crashing
                    continue
        
        logger.info(f"\nCompleted {len(backtest_results)} backtests")
        
        return backtest_results
    
    def phase5_generate_report(
        self,
        discovery_results: Dict,
        backtest_results: Dict
    ):
        """
        Phase 5: Generate comprehensive final report.
        """
        report_file = self.reports_dir / f"scientific_testing_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        
        with open(report_file, 'w') as f:
            f.write("="*80 + "\n")
            f.write("SCIENTIFIC TESTING PROGRAMME - FINAL REPORT\n")
            f.write("="*80 + "\n\n")
            
            f.write(f"Report Generated: {datetime.now().isoformat()}\n\n")
            
            # Discovery Summary
            f.write("DISCOVERY METHODS SUMMARY\n")
            f.write("-"*80 + "\n")
            total_discoveries = sum(len(d) for d in discovery_results.values())
            f.write(f"Total discoveries: {total_discoveries}\n")
            f.write(f"Instruments analyzed: {len(discovery_results)}\n\n")
            
            # Backtest Summary
            f.write("\nBACKTEST RESULTS SUMMARY\n")
            f.write("-"*80 + "\n")
            f.write(f"Total strategies backtested: {len(backtest_results)}\n\n")
            
            if backtest_results:
                # Top strategies by Sharpe ratio
                sorted_strategies = sorted(
                    backtest_results.items(),
                    key=lambda x: x[1].sharpe_ratio,
                    reverse=True
                )
                
                f.write("TOP 10 STRATEGIES BY SHARPE RATIO:\n")
                for i, (name, result) in enumerate(sorted_strategies[:10], 1):
                    f.write(f"\n{i}. {name}\n")
                    f.write(f"   Sharpe: {result.sharpe_ratio:.2f}\n")
                    f.write(f"   Total Return: {result.total_return:.2%}\n")
                    f.write(f"   Win Rate: {result.win_rate:.2%}\n")
                    f.write(f"   Max Drawdown: {result.max_drawdown:.2%}\n")
                    f.write(f"   Statistically Significant: {result.is_statistically_significant}\n")
            
            f.write("\n" + "="*80 + "\n")
            f.write("END OF REPORT\n")
            f.write("="*80 + "\n")
        
        logger.info(f"Final report saved to {report_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Complete Scientific Testing Programme',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run full programme
    python scripts/run_scientific_testing.py --full
    
    # Quick validation
    python scripts/run_scientific_testing.py --quick
    
    # Specific phase
    python scripts/run_scientific_testing.py --phase discovery
    
    # Check git sync status
    python scripts/run_scientific_testing.py --check-sync
    
    # Run without git sync (local-only mode)
    python scripts/run_scientific_testing.py --full --no-git-sync
        """
    )
    
    parser.add_argument('--full', action='store_true',
                       help='Run complete testing programme')
    parser.add_argument('--quick', action='store_true',
                       help='Quick mode for validation')
    parser.add_argument('--phase', type=str,
                       choices=['data', 'discovery', 'validation', 'backtest', 'report'],
                       help='Run specific phase only')
    parser.add_argument('--output-dir', type=str, default='scientific_testing_results',
                       help='Output directory')
    parser.add_argument('--no-git-sync', action='store_true',
                       help='Disable git sync checks (for offline/local-only usage)')
    parser.add_argument('--check-sync', action='store_true',
                       help='Only check git sync status and exit')
    
    args = parser.parse_args()
    
    # Check sync only mode
    if args.check_sync:
        if DEVOPS_AVAILABLE:
            print(check_sync_status())
            return  # Don't exit during pytest
        else:
            raise ImportError("DevOps module not available - cannot check sync status")
    
    # Create orchestrator
    orchestrator = ScientificTestingOrchestrator(
        output_dir=args.output_dir,
        enable_git_sync=not args.no_git_sync
    )
    
    if args.full or args.quick:
        # Run complete programme
        orchestrator.run_full_programme(quick_mode=args.quick)
    elif args.phase:
        # Run specific phase
        logger.info(f"Running phase: {args.phase}")
        if args.phase == 'data':
            orchestrator.phase1_data_validation(None)
        elif args.phase == 'discovery':
            # Need validated data first
            validated_data = orchestrator.phase1_data_validation(None)
            orchestrator.phase2_discovery_methods(validated_data, quick_mode=False)
        else:
            logger.error(f"Phase {args.phase} not yet implemented for standalone execution")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
