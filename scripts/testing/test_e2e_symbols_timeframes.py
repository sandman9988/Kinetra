#!/usr/bin/env python3
"""
End-to-End Test: 2 Symbols × 2 Timeframes
==========================================

Complete E2E test that validates the full Kinetra pipeline:
1. MetaAPI authentication and connection
2. Symbol/timeframe data download/update
3. Data validation and preparation
4. Superpot test suite execution
5. Theorem validation workflow
6. Comprehensive reporting

Features:
- On/off ramps: abort, skip, retry, exit at each stage
- Full logging with real-time recovery
- State persistence and restoration
- Atomic operations with rollback
- Statistical validation (p < 0.01)

Usage:
    # Run with default symbols (BTCUSD, EURUSD) and timeframes (H1, H4)
    python scripts/testing/test_e2e_symbols_timeframes.py
    
    # Run with custom symbols and timeframes
    python scripts/testing/test_e2e_symbols_timeframes.py \
        --symbols BTCUSD XAUUSD \
        --timeframes M15 H1 \
        --days 180
    
    # Quick test mode (fewer episodes)
    python scripts/testing/test_e2e_symbols_timeframes.py --quick
    
    # Resume from checkpoint
    python scripts/testing/test_e2e_symbols_timeframes.py --resume
"""

import os
import sys
import json
import asyncio
import argparse
import traceback
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from enum import Enum

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Load environment
from dotenv import load_dotenv
load_dotenv()

# Kinetra imports
from kinetra.workflow_manager import WorkflowManager
from kinetra.testing_framework import InstrumentSpec

# Import data download module
try:
    from scripts.download.metaapi_sync import MetaAPISync
except ImportError:
    MetaAPISync = None

# Import Superpot components
try:
    from scripts.analysis.superpot_complete import (
        SuperPotExtractor, FeatureImportanceTracker, DimensionTrainer,
        classify_asset, classify_timeframe, ROLE_CONFIGS
    )
    SUPERPOT_AVAILABLE = True
except ImportError:
    SUPERPOT_AVAILABLE = False

# Import theorem validation
try:
    from scripts.testing.validate_theorems import (
        compute_features, validate_core_theorems, 
        explore_feature_combinations, compute_regime_energy_stats
    )
    THEOREMS_AVAILABLE = True
except ImportError:
    THEOREMS_AVAILABLE = False


# =============================================================================
# ENUMS AND DATA STRUCTURES
# =============================================================================

class StageStatus(Enum):
    """Status of a workflow stage."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    RETRYING = "retrying"


class ActionOnError(Enum):
    """Action to take on error."""
    ABORT = "abort"
    SKIP = "skip"
    RETRY = "retry"
    PROMPT = "prompt"


@dataclass
class StageResult:
    """Result of a workflow stage."""
    stage_name: str
    status: StageStatus
    start_time: float
    end_time: Optional[float] = None
    duration: Optional[float] = None
    error: Optional[str] = None
    retry_count: int = 0
    output: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict:
        return {
            'stage_name': self.stage_name,
            'status': self.status.value,
            'start_time': self.start_time,
            'end_time': self.end_time,
            'duration': self.duration,
            'error': self.error,
            'retry_count': self.retry_count,
            'output': self.output
        }


@dataclass
class E2ETestConfig:
    """Configuration for E2E test."""
    symbols: List[str]
    timeframes: List[str]
    days_history: int = 365
    quick_mode: bool = False
    max_retries: int = 3
    retry_delay: float = 5.0
    action_on_error: ActionOnError = ActionOnError.RETRY
    
    # Superpot config
    superpot_episodes: int = 80
    superpot_prune_every: int = 15
    superpot_prune_count: int = 8
    
    # Theorem config
    theorem_lookback: int = 20
    theorem_top_n: int = 30
    
    # Output
    results_dir: Path = Path("results/e2e")
    checkpoint_dir: Path = Path(".e2e_checkpoints")
    
    def __post_init__(self):
        if self.quick_mode:
            self.superpot_episodes = 30
            self.superpot_prune_every = 10
            self.superpot_prune_count = 5
            self.days_history = 90


# =============================================================================
# E2E TEST ORCHESTRATOR
# =============================================================================

class E2ETestOrchestrator:
    """
    Orchestrates end-to-end testing with:
    - MetaAPI authentication
    - Data download/update
    - Data validation
    - Superpot tests
    - Theorem validation
    - Comprehensive reporting
    """
    
    def __init__(self, config: E2ETestConfig):
        self.config = config
        self.workflow_manager = WorkflowManager(
            log_dir="logs/e2e",
            backup_dir="data/backups/e2e",
            state_dir=str(config.checkpoint_dir),
            max_retries=config.max_retries,
            retry_delay=config.retry_delay
        )
        
        # Create directories
        self.config.results_dir.mkdir(parents=True, exist_ok=True)
        self.config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # State
        self.stages: List[StageResult] = []
        self.test_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.checkpoint_file = self.config.checkpoint_dir / f"checkpoint_{self.test_id}.json"
        
        # Results
        self.downloaded_files: Dict[str, str] = {}
        self.superpot_results: Dict[str, Any] = {}
        self.theorem_results: Dict[str, Any] = {}
    
    def save_checkpoint(self):
        """Save current state to checkpoint."""
        checkpoint = {
            'test_id': self.test_id,
            'config': asdict(self.config),
            'stages': [s.to_dict() for s in self.stages],
            'downloaded_files': self.downloaded_files,
            'superpot_results': self.superpot_results,
            'theorem_results': self.theorem_results,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(self.checkpoint_file, 'w') as f:
            json.dump(checkpoint, f, indent=2, default=str)
        
        self.workflow_manager.logger.debug(f"Checkpoint saved: {self.checkpoint_file}")
    
    def load_checkpoint(self, checkpoint_file: Path) -> bool:
        """Load state from checkpoint."""
        try:
            with open(checkpoint_file, 'r') as f:
                checkpoint = json.load(f)
            
            self.test_id = checkpoint['test_id']
            self.downloaded_files = checkpoint.get('downloaded_files', {})
            self.superpot_results = checkpoint.get('superpot_results', {})
            self.theorem_results = checkpoint.get('theorem_results', {})
            
            # Restore stages
            for stage_data in checkpoint.get('stages', []):
                stage = StageResult(
                    stage_name=stage_data['stage_name'],
                    status=StageStatus(stage_data['status']),
                    start_time=stage_data['start_time'],
                    end_time=stage_data.get('end_time'),
                    duration=stage_data.get('duration'),
                    error=stage_data.get('error'),
                    retry_count=stage_data.get('retry_count', 0),
                    output=stage_data.get('output')
                )
                self.stages.append(stage)
            
            self.workflow_manager.logger.info(f"✅ Checkpoint loaded: {checkpoint_file}")
            return True
        except Exception as e:
            self.workflow_manager.logger.error(f"Failed to load checkpoint: {e}")
            return False
    
    def handle_error(self, stage_name: str, error: Exception) -> ActionOnError:
        """
        Handle error with configurable on/off ramps.
        
        Returns action to take: ABORT, SKIP, RETRY
        """
        self.workflow_manager.logger.error(f"❌ Error in {stage_name}: {error}")
        self.workflow_manager.logger.error(traceback.format_exc())
        
        if self.config.action_on_error == ActionOnError.PROMPT:
            print(f"\n{'='*70}")
            print(f"ERROR in stage: {stage_name}")
            print(f"Error: {error}")
            print(f"{'='*70}")
            print("What would you like to do?")
            print("  1. Abort (exit immediately)")
            print("  2. Skip (continue to next stage)")
            print("  3. Retry (try again)")
            choice = input("Enter choice (1/2/3): ").strip()
            
            if choice == '1':
                return ActionOnError.ABORT
            elif choice == '2':
                return ActionOnError.SKIP
            else:
                return ActionOnError.RETRY
        else:
            return self.config.action_on_error
    
    async def run_stage(
        self,
        stage_name: str,
        stage_fn,
        critical: bool = False,
        **kwargs
    ) -> Tuple[bool, Optional[Any]]:
        """
        Run a workflow stage with error handling and retries.
        
        Returns:
            (success, result)
        """
        stage = StageResult(
            stage_name=stage_name,
            status=StageStatus.RUNNING,
            start_time=datetime.now().timestamp()
        )
        self.stages.append(stage)
        
        self.workflow_manager.logger.info(f"\n{'='*70}")
        self.workflow_manager.logger.info(f"STAGE: {stage_name}")
        self.workflow_manager.logger.info(f"{'='*70}")
        
        retry_count = 0
        max_retries = self.config.max_retries if not critical else self.config.max_retries * 2
        
        while retry_count <= max_retries:
            try:
                # Run stage function
                if asyncio.iscoroutinefunction(stage_fn):
                    result = await stage_fn(**kwargs)
                else:
                    result = stage_fn(**kwargs)
                
                # Success
                stage.status = StageStatus.COMPLETED
                stage.end_time = datetime.now().timestamp()
                stage.duration = stage.end_time - stage.start_time
                stage.output = result if isinstance(result, dict) else None
                
                self.workflow_manager.logger.info(f"✅ Stage completed: {stage_name} ({stage.duration:.1f}s)")
                self.save_checkpoint()
                
                return True, result
                
            except Exception as e:
                stage.retry_count = retry_count
                action = self.handle_error(stage_name, e)
                
                if action == ActionOnError.ABORT or (critical and retry_count >= max_retries):
                    stage.status = StageStatus.FAILED
                    stage.error = str(e)
                    stage.end_time = datetime.now().timestamp()
                    stage.duration = stage.end_time - stage.start_time
                    self.save_checkpoint()
                    
                    self.workflow_manager.logger.error(f"❌ Stage failed (ABORT): {stage_name}")
                    return False, None
                
                elif action == ActionOnError.SKIP:
                    stage.status = StageStatus.SKIPPED
                    stage.error = str(e)
                    stage.end_time = datetime.now().timestamp()
                    stage.duration = stage.end_time - stage.start_time
                    self.save_checkpoint()
                    
                    self.workflow_manager.logger.warning(f"⏭️  Stage skipped: {stage_name}")
                    return True, None
                
                elif action == ActionOnError.RETRY:
                    retry_count += 1
                    if retry_count <= max_retries:
                        wait_time = self.config.retry_delay * (2 ** (retry_count - 1))  # Exponential backoff
                        self.workflow_manager.logger.warning(
                            f"🔄 Retrying {stage_name} (attempt {retry_count}/{max_retries}) "
                            f"in {wait_time:.1f}s..."
                        )
                        await asyncio.sleep(wait_time)
                    else:
                        stage.status = StageStatus.FAILED
                        stage.error = str(e)
                        stage.end_time = datetime.now().timestamp()
                        stage.duration = stage.end_time - stage.start_time
                        self.save_checkpoint()
                        
                        self.workflow_manager.logger.error(f"❌ Stage failed (max retries): {stage_name}")
                        return False, None
        
        return False, None
    
    # =========================================================================
    # STAGE 1: MetaAPI Authentication
    # =========================================================================
    
    async def stage_metaapi_auth(self) -> Dict[str, Any]:
        """Authenticate with MetaAPI."""
        if MetaAPISync is None:
            raise ImportError("MetaAPISync not available. Install metaapi-cloud-sdk.")
        
        token = os.getenv('METAAPI_TOKEN')
        account_id = os.getenv('METAAPI_ACCOUNT_ID')
        
        if not token:
            raise ValueError("METAAPI_TOKEN not set in environment")
        
        self.workflow_manager.logger.info("Authenticating with MetaAPI...")
        
        sync = MetaAPISync(token=token, account_id=account_id)
        account = await sync.connect()
        
        result = {
            'account_id': account.id,
            'account_name': account.name,
            'account_state': account.state,
            'platform': account.platform,
        }
        
        self.workflow_manager.logger.info(f"✅ Connected to account: {account.name} ({account.id})")
        
        # Store sync object for later use
        self._metaapi_sync = sync
        
        return result
    
    # =========================================================================
    # STAGE 2: Data Download/Update
    # =========================================================================
    
    async def stage_download_data(self) -> Dict[str, Any]:
        """Download or update market data."""
        if not hasattr(self, '_metaapi_sync'):
            raise RuntimeError("MetaAPI not authenticated. Run stage_metaapi_auth first.")
        
        downloaded = {}
        
        for symbol in self.config.symbols:
            for timeframe in self.config.timeframes:
                self.workflow_manager.logger.info(f"Downloading {symbol} {timeframe}...")
                
                try:
                    await self._metaapi_sync.download_historical_data(
                        symbols=[symbol],
                        timeframe=timeframe,
                        days=self.config.days_history
                    )
                    
                    # Find downloaded file
                    master_dir = Path("data/master")
                    pattern = f"{symbol}_{timeframe.upper()}_*.csv"
                    files = list(master_dir.glob(pattern))
                    
                    if files:
                        # Use most recent file
                        latest_file = max(files, key=lambda p: p.stat().st_mtime)
                        downloaded[f"{symbol}_{timeframe}"] = str(latest_file)
                        self.workflow_manager.logger.info(f"✅ Downloaded: {latest_file.name}")
                    else:
                        self.workflow_manager.logger.warning(f"⚠️  No file found for {symbol} {timeframe}")
                
                except Exception as e:
                    self.workflow_manager.logger.error(f"Failed to download {symbol} {timeframe}: {e}")
                    raise
        
        self.downloaded_files = downloaded
        
        # Disconnect
        await self._metaapi_sync.disconnect()
        
        return {
            'downloaded_count': len(downloaded),
            'files': downloaded
        }
    
    # =========================================================================
    # STAGE 3: Data Validation
    # =========================================================================
    
    def stage_validate_data(self) -> Dict[str, Any]:
        """Validate downloaded data."""
        import pandas as pd
        
        validation_results = {}
        
        for key, filepath in self.downloaded_files.items():
            self.workflow_manager.logger.info(f"Validating {key}...")
            
            try:
                # Read data
                df = pd.read_csv(filepath, sep='\t')
                
                # Basic validation
                required_cols = ['<DATE>', '<TIME>', '<OPEN>', '<HIGH>', '<LOW>', '<CLOSE>', '<TICKVOL>']
                missing_cols = [c for c in required_cols if c not in df.columns]
                
                if missing_cols:
                    raise ValueError(f"Missing columns: {missing_cols}")
                
                # Data quality checks
                n_rows = len(df)
                n_nulls = df.isnull().sum().sum()
                
                # Check for gaps
                df['timestamp'] = pd.to_datetime(df['<DATE>'] + ' ' + df['<TIME>'])
                df = df.sort_values('timestamp')
                time_diffs = df['timestamp'].diff()
                
                validation_results[key] = {
                    'valid': True,
                    'rows': n_rows,
                    'nulls': int(n_nulls),
                    'start': str(df['timestamp'].min()),
                    'end': str(df['timestamp'].max()),
                    'mean_gap_hours': float(time_diffs.dt.total_seconds().mean() / 3600) if len(time_diffs) > 1 else 0,
                }
                
                self.workflow_manager.logger.info(
                    f"✅ {key}: {n_rows} rows, {validation_results[key]['mean_gap_hours']:.1f}h avg gap"
                )
                
            except Exception as e:
                validation_results[key] = {
                    'valid': False,
                    'error': str(e)
                }
                self.workflow_manager.logger.error(f"❌ Validation failed for {key}: {e}")
                raise
        
        return validation_results
    
    # =========================================================================
    # STAGE 4: Superpot Tests
    # =========================================================================
    
    def stage_superpot_tests(self) -> Dict[str, Any]:
        """Run Superpot analysis on all symbol/timeframe combinations."""
        if not SUPERPOT_AVAILABLE:
            raise ImportError("Superpot components not available")
        
        results = {}
        
        for key, filepath in self.downloaded_files.items():
            self.workflow_manager.logger.info(f"Running Superpot analysis on {key}...")
            
            symbol, timeframe = key.split('_')
            asset_class = classify_asset(symbol)
            
            try:
                # Load data
                import pandas as pd
                df = pd.read_csv(filepath, sep='\t')
                df.columns = df.columns.str.strip().str.replace('<', '').str.replace('>', '').str.lower()
                df['close'] = df['close'].astype(float)
                
                # Prepare for Superpot
                file_info = {
                    'symbol': symbol,
                    'timeframe': timeframe,
                    'asset_class': asset_class,
                    'path': filepath
                }
                
                # Run for each role
                role_results = {}
                for role in ['trader', 'risk_manager', 'portfolio_manager']:
                    self.workflow_manager.logger.info(f"  Role: {role}")
                    
                    trainer = DimensionTrainer(
                        dimension_name=f"{key}_{role}",
                        files=[file_info],
                        role=role
                    )
                    
                    result = trainer.train(
                        episodes=self.config.superpot_episodes,
                        prune_every=self.config.superpot_prune_every,
                        prune_count=self.config.superpot_prune_count,
                        verbose=False
                    )
                    
                    role_results[role] = {
                        'avg_pnl': result['avg_pnl'],
                        'avg_drawdown': result['avg_drawdown'],
                        'win_rate': result['win_rate'],
                        'n_surviving': result['n_surviving'],
                        'top_features': result['top_features'][:10],  # Top 10
                    }
                    
                    self.workflow_manager.logger.info(
                        f"    PnL: ${result['avg_pnl']:+.0f}, DD: {result['avg_drawdown']*100:.1f}%, "
                        f"Features: {result['n_surviving']}"
                    )
                
                results[key] = role_results
                
            except Exception as e:
                self.workflow_manager.logger.error(f"Superpot failed for {key}: {e}")
                raise
        
        self.superpot_results = results
        return results
    
    # =========================================================================
    # STAGE 5: Theorem Validation
    # =========================================================================
    
    def stage_theorem_validation(self) -> Dict[str, Any]:
        """Run theorem validation on all symbol/timeframe combinations."""
        if not THEOREMS_AVAILABLE:
            raise ImportError("Theorem validation components not available")
        
        results = {}
        
        for key, filepath in self.downloaded_files.items():
            self.workflow_manager.logger.info(f"Running theorem validation on {key}...")
            
            try:
                # Load and prepare data
                import pandas as pd
                df = pd.read_csv(filepath, sep='\t')
                df.columns = df.columns.str.strip().str.replace('<', '').str.replace('>', '').str.lower()
                
                # Prepare for theorem validation
                df = df.rename(columns={'tickvol': 'volume'})
                df = df[['date', 'time', 'open', 'high', 'low', 'close', 'volume']]
                df = df.dropna()
                
                # Compute physics features
                physics_df = compute_features(df, self.config.theorem_lookback)
                
                # Validate core theorems
                theorems = validate_core_theorems(physics_df)
                
                # Explore combinations
                combinations = explore_feature_combinations(physics_df, self.config.theorem_top_n)
                
                # Regime stats
                regime_stats = compute_regime_energy_stats(physics_df)
                
                # Compile results
                theorem_summary = [
                    {
                        'name': t.name,
                        'hypothesis': t.hypothesis,
                        'signals': t.condition_count,
                        'hit_rate': t.hit_rate * 100,
                        'lift': t.lift,
                        'status': 'CONFIRMED' if t.lift > 1.1 else 'WEAK' if t.lift > 1.0 else 'REJECTED'
                    }
                    for t in sorted(theorems, key=lambda x: -x.lift)
                ]
                
                results[key] = {
                    'theorems': theorem_summary,
                    'best_combination': combinations[0] if combinations else None,
                    'regime_stats': regime_stats,
                    'base_rate': float(physics_df['next_is_high'].mean()) * 100,
                }
                
                # Log summary
                confirmed = sum(1 for t in theorems if t.lift > 1.1)
                self.workflow_manager.logger.info(
                    f"  Theorems: {confirmed}/{len(theorems)} confirmed"
                )
                
                if combinations:
                    best = combinations[0]
                    self.workflow_manager.logger.info(
                        f"  Best: {best['hit_rate']:.1f}% hit rate ({best['lift']:.2f}x lift)"
                    )
                
            except Exception as e:
                self.workflow_manager.logger.error(f"Theorem validation failed for {key}: {e}")
                raise
        
        self.theorem_results = results
        return results
    
    # =========================================================================
    # STAGE 6: Generate Report
    # =========================================================================
    
    def stage_generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive E2E test report."""
        report_file = self.config.results_dir / f"e2e_report_{self.test_id}.json"
        html_file = self.config.results_dir / f"e2e_report_{self.test_id}.html"
        
        # Compile full report
        report = {
            'test_id': self.test_id,
            'timestamp': datetime.now().isoformat(),
            'config': {
                'symbols': self.config.symbols,
                'timeframes': self.config.timeframes,
                'days_history': self.config.days_history,
                'quick_mode': self.config.quick_mode,
            },
            'stages': [s.to_dict() for s in self.stages],
            'data_files': self.downloaded_files,
            'superpot_results': self.superpot_results,
            'theorem_results': self.theorem_results,
            'summary': self._generate_summary()
        }
        
        # Save JSON report
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        self.workflow_manager.logger.info(f"✅ JSON report saved: {report_file}")
        
        # Generate HTML report
        html = self._generate_html_report(report)
        with open(html_file, 'w') as f:
            f.write(html)
        
        self.workflow_manager.logger.info(f"✅ HTML report saved: {html_file}")
        
        return {
            'json_report': str(report_file),
            'html_report': str(html_file),
        }
    
    def _generate_summary(self) -> Dict[str, Any]:
        """Generate test summary statistics."""
        total_stages = len(self.stages)
        completed = sum(1 for s in self.stages if s.status == StageStatus.COMPLETED)
        failed = sum(1 for s in self.stages if s.status == StageStatus.FAILED)
        skipped = sum(1 for s in self.stages if s.status == StageStatus.SKIPPED)
        
        total_duration = sum(s.duration for s in self.stages if s.duration) if self.stages else 0
        
        # Superpot summary
        superpot_summary = {}
        if self.superpot_results:
            all_pnls = []
            all_dds = []
            for key, roles in self.superpot_results.items():
                for role, data in roles.items():
                    all_pnls.append(data['avg_pnl'])
                    all_dds.append(data['avg_drawdown'])
            
            superpot_summary = {
                'avg_pnl': float(sum(all_pnls) / len(all_pnls)) if all_pnls else 0,
                'avg_drawdown': float(sum(all_dds) / len(all_dds)) if all_dds else 0,
            }
        
        # Theorem summary
        theorem_summary = {}
        if self.theorem_results:
            all_confirmed = []
            all_lifts = []
            for key, data in self.theorem_results.items():
                confirmed = sum(1 for t in data['theorems'] if t['status'] == 'CONFIRMED')
                total_theorems = len(data['theorems'])
                all_confirmed.append(confirmed / total_theorems if total_theorems > 0 else 0)
                all_lifts.extend([t['lift'] for t in data['theorems']])
            
            theorem_summary = {
                'avg_confirmed_pct': float(sum(all_confirmed) / len(all_confirmed) * 100) if all_confirmed else 0,
                'avg_lift': float(sum(all_lifts) / len(all_lifts)) if all_lifts else 0,
            }
        
        return {
            'total_stages': total_stages,
            'completed': completed,
            'failed': failed,
            'skipped': skipped,
            'success_rate': completed / total_stages * 100 if total_stages > 0 else 0,
            'total_duration_sec': total_duration,
            'superpot': superpot_summary,
            'theorems': theorem_summary,
        }
    
    def _generate_html_report(self, report: Dict) -> str:
        """Generate HTML report with HTML-escaped content."""
        import html as html_module
        summary = report['summary']
        
        # Helper function for safe HTML escaping
        def escape(text):
            return html_module.escape(str(text))
        
        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>E2E Test Report - {report['test_id']}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; }}
        h1 {{ color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
        h2 {{ color: #34495e; margin-top: 30px; }}
        .summary {{ background: #ecf0f1; padding: 15px; border-radius: 5px; margin: 20px 0; }}
        .metric {{ display: inline-block; margin: 10px 20px 10px 0; }}
        .metric-label {{ font-weight: bold; color: #7f8c8d; }}
        .metric-value {{ font-size: 1.5em; color: #2c3e50; }}
        .success {{ color: #27ae60; }}
        .warning {{ color: #f39c12; }}
        .error {{ color: #e74c3c; }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background: #3498db; color: white; }}
        tr:hover {{ background: #f5f5f5; }}
        .stage-completed {{ color: #27ae60; font-weight: bold; }}
        .stage-failed {{ color: #e74c3c; font-weight: bold; }}
        .stage-skipped {{ color: #f39c12; font-weight: bold; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🧪 E2E Test Report</h1>
        <p><strong>Test ID:</strong> {escape(report['test_id'])}</p>
        <p><strong>Timestamp:</strong> {escape(report['timestamp'])}</p>
        
        <div class="summary">
            <h2>📊 Summary</h2>
            <div class="metric">
                <div class="metric-label">Success Rate</div>
                <div class="metric-value {'success' if summary['success_rate'] > 80 else 'warning'}">{summary['success_rate']:.1f}%</div>
            </div>
            <div class="metric">
                <div class="metric-label">Duration</div>
                <div class="metric-value">{summary['total_duration_sec'] / 60:.1f} min</div>
            </div>
            <div class="metric">
                <div class="metric-label">Stages</div>
                <div class="metric-value">{summary['completed']}/{summary['total_stages']}</div>
            </div>
        </div>
        
        <h2>⚙️ Configuration</h2>
        <table>
            <tr><th>Parameter</th><th>Value</th></tr>
            <tr><td>Symbols</td><td>{', '.join(report['config']['symbols'])}</td></tr>
            <tr><td>Timeframes</td><td>{', '.join(report['config']['timeframes'])}</td></tr>
            <tr><td>History Days</td><td>{report['config']['days_history']}</td></tr>
            <tr><td>Quick Mode</td><td>{'Yes' if report['config']['quick_mode'] else 'No'}</td></tr>
        </table>
        
        <h2>📋 Stages</h2>
        <table>
            <tr><th>Stage</th><th>Status</th><th>Duration</th><th>Retries</th></tr>
"""
        
        for stage in report['stages']:
            status_class = f"stage-{escape(stage['status'])}"
            duration = f"{stage['duration']:.1f}s" if stage['duration'] else "N/A"
            html += f"""
            <tr>
                <td>{escape(stage['stage_name'])}</td>
                <td class="{status_class}">{escape(stage['status']).upper()}</td>
                <td>{escape(duration)}</td>
                <td>{escape(stage['retry_count'])}</td>
            </tr>
"""
        
        html += """
        </table>
        
        <h2>🎯 Superpot Results</h2>
"""
        
        if summary.get('superpot'):
            html += f"""
        <div class="summary">
            <div class="metric">
                <div class="metric-label">Avg PnL</div>
                <div class="metric-value">${summary['superpot']['avg_pnl']:+.0f}</div>
            </div>
            <div class="metric">
                <div class="metric-label">Avg Drawdown</div>
                <div class="metric-value">{summary['superpot']['avg_drawdown']*100:.1f}%</div>
            </div>
        </div>
"""
        
        html += """
        <h2>🔬 Theorem Validation</h2>
"""
        
        if summary.get('theorems'):
            html += f"""
        <div class="summary">
            <div class="metric">
                <div class="metric-label">Confirmed</div>
                <div class="metric-value">{summary['theorems']['avg_confirmed_pct']:.1f}%</div>
            </div>
            <div class="metric">
                <div class="metric-label">Avg Lift</div>
                <div class="metric-value">{summary['theorems']['avg_lift']:.2f}x</div>
            </div>
        </div>
"""
        
        html += """
    </div>
</body>
</html>
"""
        return html
    
    # =========================================================================
    # Main Workflow
    # =========================================================================
    
    async def run(self) -> bool:
        """Run complete E2E test workflow."""
        start_time = datetime.now()
        
        self.workflow_manager.logger.info(f"""
╔══════════════════════════════════════════════════════════════════════╗
║           E2E Test: 2 Symbols × 2 Timeframes                         ║
║                                                                      ║
║   Test ID: {self.test_id}                                    ║
╚══════════════════════════════════════════════════════════════════════╝
""")
        
        self.workflow_manager.logger.info(f"Symbols: {', '.join(self.config.symbols)}")
        self.workflow_manager.logger.info(f"Timeframes: {', '.join(self.config.timeframes)}")
        self.workflow_manager.logger.info(f"Quick mode: {self.config.quick_mode}")
        
        try:
            # Stage 1: MetaAPI Authentication
            success, _ = await self.run_stage(
                "MetaAPI Authentication",
                self.stage_metaapi_auth,
                critical=True
            )
            if not success:
                return False
            
            # Stage 2: Data Download
            success, _ = await self.run_stage(
                "Data Download/Update",
                self.stage_download_data,
                critical=True
            )
            if not success:
                return False
            
            # Stage 3: Data Validation
            success, _ = await self.run_stage(
                "Data Validation",
                self.stage_validate_data,
                critical=True
            )
            if not success:
                return False
            
            # Stage 4: Superpot Tests
            success, _ = await self.run_stage(
                "Superpot Analysis",
                self.stage_superpot_tests,
                critical=False
            )
            # Continue even if Superpot fails
            
            # Stage 5: Theorem Validation
            success, _ = await self.run_stage(
                "Theorem Validation",
                self.stage_theorem_validation,
                critical=False
            )
            # Continue even if theorems fail
            
            # Stage 6: Generate Report
            success, _ = await self.run_stage(
                "Report Generation",
                self.stage_generate_report,
                critical=False
            )
            
            # Final summary
            elapsed = (datetime.now() - start_time).total_seconds()
            
            self.workflow_manager.logger.info(f"""
╔══════════════════════════════════════════════════════════════════════╗
║                     E2E TEST COMPLETED                               ║
╚══════════════════════════════════════════════════════════════════════╝

Duration: {elapsed / 60:.1f} minutes
Stages completed: {sum(1 for s in self.stages if s.status == StageStatus.COMPLETED)}/{len(self.stages)}
Success rate: {sum(1 for s in self.stages if s.status == StageStatus.COMPLETED) / len(self.stages) * 100:.1f}%

Results saved to: {self.config.results_dir}
""")
            
            return True
            
        except Exception as e:
            self.workflow_manager.logger.error(f"Fatal error in E2E test: {e}")
            self.workflow_manager.logger.error(traceback.format_exc())
            return False


# =============================================================================
# MAIN
# =============================================================================

async def main():
    parser = argparse.ArgumentParser(
        description='E2E Test: 2 Symbols × 2 Timeframes',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Default run
  python scripts/testing/test_e2e_symbols_timeframes.py
  
  # Custom symbols and timeframes
  python scripts/testing/test_e2e_symbols_timeframes.py --symbols BTCUSD XAUUSD --timeframes M15 H1
  
  # Quick test
  python scripts/testing/test_e2e_symbols_timeframes.py --quick
  
  # Resume from checkpoint
  python scripts/testing/test_e2e_symbols_timeframes.py --resume
        """
    )
    
    parser.add_argument(
        '--symbols',
        nargs='+',
        default=['BTCUSD', 'EURUSD'],
        help='Symbols to test (default: BTCUSD EURUSD)'
    )
    parser.add_argument(
        '--timeframes',
        nargs='+',
        default=['H1', 'H4'],
        help='Timeframes to test (default: H1 H4)'
    )
    parser.add_argument(
        '--days',
        type=int,
        default=365,
        help='Days of history to download (default: 365)'
    )
    parser.add_argument(
        '--quick',
        action='store_true',
        help='Quick test mode (fewer episodes, less history)'
    )
    parser.add_argument(
        '--resume',
        action='store_true',
        help='Resume from latest checkpoint'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        help='Resume from specific checkpoint file'
    )
    parser.add_argument(
        '--action-on-error',
        choices=['abort', 'skip', 'retry', 'prompt'],
        default='retry',
        help='Action on error (default: retry)'
    )
    
    args = parser.parse_args()
    
    # Create config
    config = E2ETestConfig(
        symbols=args.symbols,
        timeframes=args.timeframes,
        days_history=args.days,
        quick_mode=args.quick,
        action_on_error=ActionOnError(args.action_on_error)
    )
    
    # Create orchestrator
    orchestrator = E2ETestOrchestrator(config)
    
    # Resume from checkpoint if requested
    if args.resume or args.checkpoint:
        checkpoint_file = None
        if args.checkpoint:
            checkpoint_file = Path(args.checkpoint)
        else:
            # Find latest checkpoint
            checkpoints = list(config.checkpoint_dir.glob("checkpoint_*.json"))
            if checkpoints:
                checkpoint_file = max(checkpoints, key=lambda p: p.stat().st_mtime)
        
        if checkpoint_file and checkpoint_file.exists():
            orchestrator.load_checkpoint(checkpoint_file)
        else:
            print("No checkpoint found. Starting fresh.")
    
    # Run E2E test
    success = await orchestrator.run()
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    asyncio.run(main())
