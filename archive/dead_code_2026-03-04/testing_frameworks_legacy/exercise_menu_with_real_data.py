#!/usr/bin/env python3
"""
Real Data Menu Exerciser
=========================

Exercises the Kinetra menu system with real data to identify:
- Performance bottlenecks
- Data handling issues
- Memory problems
- Slow operations

Uses actual data from data/master/ directory (87 CSV files, 117MB).

Usage:
    python scripts/testing/exercise_menu_with_real_data.py [--verbose] [--profile]
"""

import argparse
import cProfile
import io
import json
import logging
import pstats
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class RealDataExerciser:
    """Exercise menu with real data to find bottlenecks."""
    
    def __init__(self, log_file: str = "logs/real_data_exercise.log", profile: bool = False):
        """
        Initialize real data exerciser.
        
        Args:
            log_file: Path to log file
            profile: Whether to enable profiling
        """
        self.log_file = Path(log_file)
        self.profile = profile
        self.profiler = None
        
        if self.profile:
            self.profiler = cProfile.Profile()
        
        # Create log directory
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self.setup_logging()
        
        # Statistics
        self.stats = {
            'start_time': datetime.now(),
            'end_time': None,
            'data_files_found': 0,
            'data_files_loaded': 0,
            'total_rows_processed': 0,
            'operations': [],
            'errors': [],
            'bottlenecks': []
        }
        
        self.logger.info("="*80)
        self.logger.info("Real Data Menu Exerciser Initialized")
        self.logger.info("="*80)
        self.logger.info(f"Log file: {log_file}")
        self.logger.info(f"Profiling: {profile}")
        self.logger.info("="*80)
    
    def setup_logging(self):
        """Setup logging configuration."""
        self.logger = logging.getLogger('RealDataExerciser')
        self.logger.setLevel(logging.DEBUG)
        self.logger.handlers = []
        
        # File handler
        file_handler = logging.FileHandler(self.log_file)
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        self.logger.addHandler(file_handler)
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%H:%M:%S'
        )
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)
    
    def find_data_files(self) -> List[Path]:
        """
        Find all CSV data files.
        
        Returns:
            List of Path objects to CSV files
        """
        self.logger.info("Scanning for data files...")
        
        data_dir = Path("data/master")
        if not data_dir.exists():
            self.logger.warning(f"Data directory not found: {data_dir}")
            return []
        
        csv_files = list(data_dir.rglob("*.csv"))
        self.stats['data_files_found'] = len(csv_files)
        
        self.logger.info(f"Found {len(csv_files)} CSV files")
        
        # Group by asset class
        by_class = {}
        for file in csv_files:
            asset_class = file.parent.name
            by_class.setdefault(asset_class, []).append(file)
        
        for asset_class, files in sorted(by_class.items()):
            self.logger.info(f"  {asset_class}: {len(files)} files")
        
        return csv_files
    
    def test_data_loading(self, files: List[Path]) -> Dict:
        """
        Test data loading performance.
        
        Args:
            files: List of CSV files to load
            
        Returns:
            Dictionary with test results
        """
        self.logger.info("\n" + "="*80)
        self.logger.info("Testing Data Loading Performance")
        self.logger.info("="*80)
        
        results = {
            'operation': 'data_loading',
            'files_tested': len(files),
            'successful': 0,
            'failed': 0,
            'total_time': 0,
            'total_rows': 0,
            'total_size_mb': 0,
            'slowest_files': [],
            'errors': []
        }
        
        start_time = time.time()
        
        # Test loading each file with progress bar
        with tqdm(total=len(files), desc="Loading data files", unit="file") as pbar:
            for file in files:
                file_start = time.time()
                
                try:
                    # Get file size
                    file_size_mb = file.stat().st_size / (1024 * 1024)
                    results['total_size_mb'] += file_size_mb
                    
                    # Load CSV
                    df = pd.read_csv(file)
                    rows = len(df)
                    results['total_rows'] += rows
                    results['successful'] += 1
                    self.stats['data_files_loaded'] += 1
                    self.stats['total_rows_processed'] += rows
                    
                    file_time = time.time() - file_start
                    
                    # Track slow files (>1 second)
                    if file_time > 1.0:
                        results['slowest_files'].append({
                            'file': str(file),
                            'time': file_time,
                            'rows': rows,
                            'size_mb': file_size_mb
                        })
                    
                    pbar.set_postfix({
                        'rows': f"{rows:,}",
                        'time': f"{file_time:.2f}s"
                    })
                    
                except Exception as e:
                    results['failed'] += 1
                    error_record = {
                        'file': str(file),
                        'error': str(e)
                    }
                    results['errors'].append(error_record)
                    self.stats['errors'].append(error_record)
                    self.logger.error(f"Failed to load {file.name}: {e}")
                
                pbar.update(1)
        
        results['total_time'] = time.time() - start_time
        
        # Summary
        self.logger.info(f"\nData Loading Summary:")
        self.logger.info(f"  Files loaded: {results['successful']}/{len(files)}")
        self.logger.info(f"  Total rows: {results['total_rows']:,}")
        self.logger.info(f"  Total size: {results['total_size_mb']:.1f} MB")
        self.logger.info(f"  Total time: {results['total_time']:.2f}s")
        self.logger.info(f"  Avg speed: {results['total_size_mb']/results['total_time']:.1f} MB/s")
        
        if results['slowest_files']:
            self.logger.info(f"\n  Slowest files (>{1.0}s):")
            for item in sorted(results['slowest_files'], key=lambda x: x['time'], reverse=True)[:5]:
                self.logger.info(f"    {Path(item['file']).name}: {item['time']:.2f}s ({item['rows']:,} rows)")
                
                # Flag as bottleneck if >2s
                if item['time'] > 2.0:
                    self.stats['bottlenecks'].append({
                        'operation': 'data_loading',
                        'file': item['file'],
                        'time': item['time'],
                        'issue': f"Slow file load: {item['time']:.2f}s for {item['rows']:,} rows"
                    })
        
        return results
    
    def test_data_preparation(self) -> Dict:
        """
        Test data preparation (train/test split).
        
        Returns:
            Dictionary with test results
        """
        self.logger.info("\n" + "="*80)
        self.logger.info("Testing Data Preparation")
        self.logger.info("="*80)
        
        results = {
            'operation': 'data_preparation',
            'success': False,
            'time': 0,
            'error': None
        }
        
        start_time = time.time()
        
        try:
            import subprocess
            
            self.logger.info("Running prepare_data.py...")
            
            process = subprocess.Popen(
                [sys.executable, "scripts/download/prepare_data.py"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Monitor output with timeout
            timeout = 60  # 60 seconds
            try:
                stdout, stderr = process.communicate(timeout=timeout)
                
                if process.returncode == 0:
                    results['success'] = True
                    self.logger.info("✅ Data preparation successful")
                else:
                    results['error'] = f"Exit code {process.returncode}"
                    self.logger.error(f"❌ Data preparation failed: {stderr}")
                    
            except subprocess.TimeoutExpired:
                process.kill()
                results['error'] = f"Timeout after {timeout}s"
                self.logger.error(f"❌ Data preparation timed out")
                
                self.stats['bottlenecks'].append({
                    'operation': 'data_preparation',
                    'time': timeout,
                    'issue': f"Operation timeout after {timeout}s"
                })
            
        except Exception as e:
            results['error'] = str(e)
            self.logger.error(f"❌ Error in data preparation: {e}")
            self.logger.debug(traceback.format_exc())
        
        results['time'] = time.time() - start_time
        
        self.logger.info(f"  Time: {results['time']:.2f}s")
        
        return results
    
    def test_data_integrity_check(self) -> Dict:
        """
        Test data integrity checking.
        
        Returns:
            Dictionary with test results
        """
        self.logger.info("\n" + "="*80)
        self.logger.info("Testing Data Integrity Check")
        self.logger.info("="*80)
        
        results = {
            'operation': 'data_integrity_check',
            'success': False,
            'time': 0,
            'error': None
        }
        
        start_time = time.time()
        
        try:
            import subprocess
            
            self.logger.info("Running check_data_integrity.py...")
            
            process = subprocess.Popen(
                [sys.executable, "scripts/download/check_data_integrity.py"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Monitor with timeout
            timeout = 30
            try:
                stdout, stderr = process.communicate(timeout=timeout)
                
                if process.returncode == 0:
                    results['success'] = True
                    self.logger.info("✅ Integrity check successful")
                else:
                    results['error'] = f"Exit code {process.returncode}"
                    self.logger.error(f"❌ Integrity check failed")
                    
            except subprocess.TimeoutExpired:
                process.kill()
                results['error'] = f"Timeout after {timeout}s"
                self.logger.error(f"❌ Integrity check timed out")
                
                self.stats['bottlenecks'].append({
                    'operation': 'data_integrity_check',
                    'time': timeout,
                    'issue': f"Operation timeout after {timeout}s"
                })
            
        except Exception as e:
            results['error'] = str(e)
            self.logger.error(f"❌ Error in integrity check: {e}")
        
        results['time'] = time.time() - start_time
        self.logger.info(f"  Time: {results['time']:.2f}s")
        
        return results
    
    def test_menu_import(self) -> Dict:
        """
        Test menu import time.
        
        Returns:
            Dictionary with test results
        """
        self.logger.info("\n" + "="*80)
        self.logger.info("Testing Menu Import Time")
        self.logger.info("="*80)
        
        results = {
            'operation': 'menu_import',
            'success': False,
            'time': 0,
            'error': None
        }
        
        start_time = time.time()
        
        try:
            # Remove cached import
            if 'kinetra_menu' in sys.modules:
                del sys.modules['kinetra_menu']
            
            # Import menu
            import kinetra_menu
            
            results['success'] = True
            results['time'] = time.time() - start_time
            
            self.logger.info(f"✅ Menu import successful: {results['time']:.3f}s")
            
            if results['time'] > 1.0:
                self.stats['bottlenecks'].append({
                    'operation': 'menu_import',
                    'time': results['time'],
                    'issue': f"Slow menu import: {results['time']:.3f}s"
                })
            
        except Exception as e:
            results['error'] = str(e)
            results['time'] = time.time() - start_time
            self.logger.error(f"❌ Menu import failed: {e}")
            self.logger.debug(traceback.format_exc())
        
        return results
    
    def print_final_report(self):
        """Print final testing report."""
        self.stats['end_time'] = datetime.now()
        duration = (self.stats['end_time'] - self.stats['start_time']).total_seconds()
        
        self.logger.info(f"\n{'#'*80}")
        self.logger.info(f"# FINAL REPORT - REAL DATA EXERCISE")
        self.logger.info(f"{'#'*80}\n")
        
        self.logger.info(f"Duration: {duration:.1f}s")
        self.logger.info(f"Data files found: {self.stats['data_files_found']}")
        self.logger.info(f"Data files loaded: {self.stats['data_files_loaded']}")
        self.logger.info(f"Total rows processed: {self.stats['total_rows_processed']:,}")
        self.logger.info(f"Operations tested: {len(self.stats['operations'])}")
        
        if self.stats['errors']:
            self.logger.info(f"\n{'='*80}")
            self.logger.info(f"ERRORS ({len(self.stats['errors'])})")
            self.logger.info(f"{'='*80}")
            for error in self.stats['errors']:
                self.logger.info(f"  {error}")
        
        if self.stats['bottlenecks']:
            self.logger.info(f"\n{'='*80}")
            self.logger.info(f"BOTTLENECKS IDENTIFIED ({len(self.stats['bottlenecks'])})")
            self.logger.info(f"{'='*80}")
            for bottleneck in self.stats['bottlenecks']:
                self.logger.info(f"  {bottleneck['operation']}: {bottleneck['issue']}")
        
        # Save report
        self.save_report()
        
        # Print profiling stats if enabled
        if self.profile and self.profiler:
            self.print_profiling_stats()
        
        self.logger.info(f"\n{'#'*80}")
        self.logger.info(f"Report saved to: {self.log_file.parent / 'real_data_exercise_report.json'}")
        self.logger.info(f"{'#'*80}\n")
    
    def save_report(self):
        """Save testing report to JSON file."""
        try:
            report_file = self.log_file.parent / 'real_data_exercise_report.json'
            
            # Convert datetime to string
            report = self.stats.copy()
            report['start_time'] = report['start_time'].isoformat()
            if report['end_time']:
                report['end_time'] = report['end_time'].isoformat()
            
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2)
            
            self.logger.debug(f"Report saved to {report_file}")
            
        except Exception as e:
            self.logger.warning(f"Failed to save report: {e}")
    
    def print_profiling_stats(self):
        """Print profiling statistics."""
        self.logger.info(f"\n{'='*80}")
        self.logger.info(f"PROFILING STATISTICS")
        self.logger.info(f"{'='*80}\n")
        
        # Stop profiler
        self.profiler.disable()
        
        # Print stats
        s = io.StringIO()
        ps = pstats.Stats(self.profiler, stream=s)
        ps.strip_dirs()
        ps.sort_stats('cumulative')
        ps.print_stats(20)  # Top 20 functions
        
        self.logger.info(s.getvalue())
    
    def run(self):
        """Run comprehensive real data exercise."""
        self.logger.info("Starting real data exercise...")
        
        if self.profile:
            self.profiler.enable()
        
        try:
            # Find all data files
            files = self.find_data_files()
            
            if not files:
                self.logger.warning("No data files found - cannot exercise menu with real data")
                return
            
            # Test 1: Data loading performance
            result = self.test_data_loading(files)
            self.stats['operations'].append(result)
            
            # Test 2: Menu import time
            result = self.test_menu_import()
            self.stats['operations'].append(result)
            
            # Test 3: Data integrity check
            result = self.test_data_integrity_check()
            self.stats['operations'].append(result)
            
            # Test 4: Data preparation
            result = self.test_data_preparation()
            self.stats['operations'].append(result)
            
        except KeyboardInterrupt:
            self.logger.info("\n⚠️  Interrupted by user (Ctrl+C)")
        except Exception as e:
            self.logger.error(f"Unexpected error: {e}")
            self.logger.debug(traceback.format_exc())
        finally:
            self.print_final_report()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Exercise Kinetra menu with real data"
    )
    parser.add_argument(
        '--log-file',
        type=str,
        default='logs/real_data_exercise.log',
        help='Path to log file'
    )
    parser.add_argument(
        '--profile',
        action='store_true',
        help='Enable profiling'
    )
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress console output'
    )
    
    args = parser.parse_args()
    
    # Create exerciser
    exerciser = RealDataExerciser(
        log_file=args.log_file,
        profile=args.profile
    )
    
    # Run exerciser
    exerciser.run()


if __name__ == '__main__':
    main()
