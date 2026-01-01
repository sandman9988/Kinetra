#!/usr/bin/env python3
"""
Script Classification and Integration Analysis
==============================================

Analyzes all 121 unused scripts and categorizes them into:
1. INTEGRATE - Should be added to menu system
2. STANDALONE - Document as standalone tools  
3. DEPRECATE - Legacy/obsolete, can be removed
4. TEST - Unit/integration tests (don't add to menu)

Usage:
    python scripts/classify_unused_scripts.py
"""

import sys
from pathlib import Path
from typing import Dict, List, Tuple

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))


# =============================================================================
# CLASSIFICATION RULES
# =============================================================================

CLASSIFICATIONS = {
    # =========================================================================
    # INTEGRATE - Add to menu system
    # =========================================================================
    'INTEGRATE': {
        # Analysis scripts - valuable for post-exploration analysis
        'scripts/analysis/quick_results.py': {
            'menu': 'Exploration Testing',
            'action': 'View Recent Results',
            'reason': 'Quick results viewer - useful after exploration runs',
            'priority': 'HIGH'
        },
        'scripts/analysis/analyze_energy.py': {
            'menu': 'Exploration Testing → Measurement Analysis',
            'action': 'Analyze Energy Metrics',
            'reason': 'Core physics analysis - aligns with first principles',
            'priority': 'HIGH'
        },
        'scripts/analysis/analyze_asymmetric_rewards.py': {
            'menu': 'Exploration Testing → Measurement Analysis',
            'action': 'Analyze Asymmetric Rewards',
            'reason': 'Reward shaping analysis - critical for strategy evaluation',
            'priority': 'MEDIUM'
        },
        
        # Training scripts
        'scripts/training/train_rl.py': {
            'menu': 'Exploration Testing',
            'action': 'Train RL Agent (Advanced)',
            'reason': 'Main RL training script - advanced users',
            'priority': 'MEDIUM'
        },
        'scripts/training/monitor_training.py': {
            'menu': 'System Status & Health',
            'action': 'Monitor Training Progress',
            'reason': 'Real-time training monitoring',
            'priority': 'MEDIUM'
        },
        
        # Testing scripts - integration/E2E
        'scripts/testing/batch_backtest.py': {
            'menu': 'Backtesting',
            'action': 'Batch Backtest (Multiple Strategies)',
            'reason': 'Run multiple backtests in batch',
            'priority': 'MEDIUM'
        },
        'scripts/testing/run_full_backtest.py': {
            'menu': 'Backtesting',
            'action': 'Full Backtest (All Instruments)',
            'reason': 'Comprehensive backtest across all data',
            'priority': 'HIGH'
        },
        
        # Download scripts
        'scripts/download/metaapi_sync.py': {
            'menu': 'Data Management',
            'action': 'Sync with MetaAPI (Continuous)',
            'reason': 'Keep data synchronized with broker',
            'priority': 'MEDIUM'
        },
        
        # Utility scripts
        'scripts/dashboard.py': {
            'menu': 'System Status & Health',
            'action': 'Launch Dashboard',
            'reason': 'Visual dashboard for system monitoring',
            'priority': 'HIGH'
        },
        'scripts/benchmark_performance.py': {
            'menu': 'System Status & Health',
            'action': 'Benchmark Performance',
            'reason': 'System performance benchmarking',
            'priority': 'LOW'
        },
    },
    
    # =========================================================================
    # STANDALONE - Document as standalone tools (don't integrate to menu)
    # =========================================================================
    'STANDALONE': {
        # SuperPot scripts - research tools
        'scripts/analysis/superpot_complete.py': 'SuperPot: Complete exploration across all instruments',
        'scripts/analysis/superpot_by_class.py': 'SuperPot: Asset class-specific feature discovery',
        'scripts/analysis/superpot_explorer.py': 'SuperPot: Interactive feature explorer',
        'scripts/analysis/superpot_physics.py': 'SuperPot: Physics-focused feature discovery',
        'scripts/analysis/superpot_empirical.py': 'SuperPot: Empirical feature testing',
        'scripts/analysis/superpot_dsp_driven.py': 'SuperPot: DSP-driven feature extraction',
        
        # Deep analysis scripts - researcher tools
        'scripts/analysis/pathfinder_deep_dive.py': 'Deep analysis: Pathfinder strategy exploration',
        'scripts/analysis/analyze_trade_management.py': 'Analysis: Trade management strategies',
        'scripts/analysis/analyze_triggers.py': 'Analysis: Entry/exit trigger effectiveness',
        'scripts/analysis/debug_csv.py': 'Debug tool: CSV data inspection',
        
        # Training exploration - researcher tools
        'scripts/training/explore_universal.py': 'Research: Universal agent exploration',
        'scripts/training/explore_specialization.py': 'Research: Agent specialization study',
        'scripts/training/explorer_standalone.py': 'Research: Standalone exploration framework',
        'scripts/training/pathfinder_explore.py': 'Research: Pathfinder strategy research',
        
        # Advanced training - for researchers
        'scripts/training/train_berserker.py': 'Training: Berserker strategy agent',
        'scripts/training/train_triad.py': 'Training: Triad system (3-agent ensemble)',
        'scripts/training/train_sniper.py': 'Training: Sniper strategy agent',
        
        # Workflow orchestration - devops tools
        'scripts/master_workflow.py': 'Workflow: Master orchestration script',
        'scripts/silent_failure_workflow.py': 'Workflow: Silent failure detection pipeline',
        'scripts/devops_manager.py': 'DevOps: System management tool',
        'scripts/branch_manager.py': 'DevOps: Git branch management',
        'scripts/cache_manager.py': 'DevOps: Cache management utility',
        
        # Data preparation - for data engineers
        'scripts/download/parallel_data_prep.py': 'Data: Parallel data preparation (advanced)',
        'scripts/download/prepare_exploration_data.py': 'Data: Exploration-specific preparation',
        'scripts/download/standardize_data_cutoff.py': 'Data: Standardize to common cutoff date',
    },
    
    # =========================================================================
    # TEST - Unit/integration tests (belong in tests/, not menu)
    # =========================================================================
    'TEST': {
        # All test_* scripts
        'scripts/testing/test_*.py': 'Unit/integration tests - should remain as tests',
        # Specific test scripts
        'scripts/testing/test_menu.py': 'Menu system tests',
        'scripts/testing/test_backtest_trend.py': 'Backtest trend validation',
        'scripts/testing/test_physics.py': 'Physics engine unit tests',
        'scripts/testing/validate_theorems.py': 'Theorem validation tests',
        'scripts/testing/verify_calculations.py': 'Calculation verification tests',
        'scripts/testing/unified_test_framework.py': 'Unified testing framework',
        
        # Validation scripts
        'scripts/testing/validate_btc_h1_layer1.py': 'BTC H1 layer 1 validation',
        'scripts/testing/validate_mql5_compliance.py': 'MQL5 compliance validation',
        'scripts/testing/validate_thesis.py': 'Thesis validation tests',
        
        # Demo/example scripts
        'scripts/testing/demo_backtest_improvements.py': 'Demo: Backtest improvements',
        'scripts/testing/example_testing_framework.py': 'Example: Testing framework usage',
    },
    
    # =========================================================================
    # DEPRECATE - Legacy/obsolete scripts
    # =========================================================================
    'DEPRECATE': {
        # Duplicate/redundant download scripts
        'scripts/download/download_mt5_data.py': 'Legacy: Replaced by download_interactive.py',
        'scripts/download/download_metaapi.py': 'Legacy: Replaced by download_interactive.py',
        'scripts/download/download_market_data.py': 'Legacy: Replaced by download_interactive.py',
        'scripts/download/convert_mt5_format.py': 'Obsolete: MT5 format conversion (no longer needed)',
        'scripts/download/load_all_symbols.py': 'Obsolete: Replaced by download_interactive.py symbol selection',
        'scripts/download/metaapi_bulk_download.py': 'Legacy: Replaced by download_interactive.py',
        
        # Redundant analysis
        'scripts/analysis/analyze_energy_both.py': 'Duplicate: Same as analyze_energy.py',
        'scripts/analysis/analyze_direction.py': 'Obsolete: Functionality moved to measurements.py',
        'scripts/analysis/analyze_directional_tension.py': 'Obsolete: Integrated into physics engine',
        
        # Redundant testing
        'scripts/testing/rl_backtest.py': 'Duplicate: Use run_comprehensive_backtest.py',
        'scripts/testing/run_physics_backtest.py': 'Duplicate: Use run_comprehensive_backtest.py',
        'scripts/testing/run_exploration_backtest.py': 'Duplicate: Use run_comprehensive_backtest.py',
        'scripts/testing/integrate_realistic_backtest.py': 'Obsolete: Functionality in backtest engine',
        
        # Redundant training
        'scripts/training/quick_rl_test.py': 'Obsolete: Use explore_compare_agents.py',
        'scripts/training/train_fast_multi.py': 'Duplicate: Use train_rl.py with --parallel flag',
        'scripts/training/demo_continual_learning.py': 'Demo: Not production-ready',
        
        # Legacy utilities
        'scripts/detect_silent_failures.py': 'Legacy: Functionality now in testing framework',
        'scripts/fix_silent_failures.py': 'Legacy: Manual tool no longer needed',
        'scripts/demo_modular_execution.py': 'Demo: Example script only',
        'scripts/run_local.py': 'Obsolete: Use kinetra_menu.py instead',
        'scripts/train.py': 'Legacy: Use scripts/training/train_rl.py',
        
        # Setup scripts
        'scripts/setup/check_gpu.py': 'Utility: GPU check (rarely needed, can run manually)',
    },
    
    # =========================================================================
    # INVESTIGATE - Need more context to classify
    # =========================================================================
    'INVESTIGATE': {
        'scripts/run_predictor.py': 'Unknown: Need to review purpose',
        'scripts/monitor_daemon.py': 'Unknown: Daemon monitoring - check if still used',
        'scripts/download/extract_mt5_specs.py': 'Unknown: MT5 spec extraction - still relevant?',
        'scripts/download/fetch_broker_spec_from_metaapi.py': 'Unknown: Broker spec fetching - still used?',
    },
}


# =============================================================================
# ANALYSIS FUNCTIONS
# =============================================================================

def generate_integration_plan() -> str:
    """Generate detailed integration plan for scripts to add to menu."""
    report = """
================================================================================
SCRIPT INTEGRATION PLAN
================================================================================

Scripts recommended for integration into menu system, organized by priority:

"""
    
    # Group by priority
    high_priority = []
    medium_priority = []
    low_priority = []
    
    for script, details in CLASSIFICATIONS['INTEGRATE'].items():
        if details['priority'] == 'HIGH':
            high_priority.append((script, details))
        elif details['priority'] == 'MEDIUM':
            medium_priority.append((script, details))
        else:
            low_priority.append((script, details))
    
    report += "HIGH PRIORITY (Implement First):\n" + "="*80 + "\n\n"
    for script, details in high_priority:
        report += f"📌 {script}\n"
        report += f"   Menu Location: {details['menu']}\n"
        report += f"   Action Name: {details['action']}\n"
        report += f"   Reason: {details['reason']}\n\n"
    
    report += "\nMEDIUM PRIORITY (Implement Second):\n" + "="*80 + "\n\n"
    for script, details in medium_priority:
        report += f"📋 {script}\n"
        report += f"   Menu Location: {details['menu']}\n"
        report += f"   Action Name: {details['action']}\n"
        report += f"   Reason: {details['reason']}\n\n"
    
    report += "\nLOW PRIORITY (Nice to Have):\n" + "="*80 + "\n\n"
    for script, details in low_priority:
        report += f"📌 {script}\n"
        report += f"   Menu Location: {details['menu']}\n"
        report += f"   Action Name: {details['action']}\n"
        report += f"   Reason: {details['reason']}\n\n"
    
    report += f"\nTotal scripts to integrate: {len(CLASSIFICATIONS['INTEGRATE'])}\n"
    return report


def generate_standalone_documentation() -> str:
    """Generate documentation for standalone tools."""
    report = """
================================================================================
STANDALONE TOOLS (Do Not Add to Menu)
================================================================================

These scripts are valuable research/utility tools but should NOT be added to
the menu. They should be documented for advanced users and researchers.

"""
    
    # Group by category
    categories = {
        'SuperPot Research': [],
        'Analysis Tools': [],
        'Training Research': [],
        'DevOps Tools': [],
        'Data Engineering': [],
    }
    
    for script, description in CLASSIFICATIONS['STANDALONE'].items():
        if 'superpot' in script.lower():
            categories['SuperPot Research'].append((script, description))
        elif 'analysis' in script:
            categories['Analysis Tools'].append((script, description))
        elif 'training' in script:
            categories['Training Research'].append((script, description))
        elif any(x in script for x in ['devops', 'branch', 'cache', 'workflow']):
            categories['DevOps Tools'].append((script, description))
        else:
            categories['Data Engineering'].append((script, description))
    
    for category, scripts in categories.items():
        if scripts:
            report += f"\n{category}:\n" + "-"*80 + "\n\n"
            for script, description in scripts:
                report += f"  {script}\n"
                report += f"  → {description}\n\n"
    
    report += f"\nTotal standalone tools: {len(CLASSIFICATIONS['STANDALONE'])}\n"
    report += "\nRecommendation: Create docs/STANDALONE_TOOLS.md documenting these\n"
    return report


def generate_deprecation_list() -> str:
    """Generate list of scripts to deprecate/remove."""
    report = """
================================================================================
SCRIPTS TO DEPRECATE/REMOVE
================================================================================

These scripts are legacy/obsolete and can be safely removed or archived.

"""
    
    # Group by category
    categories = {
        'Duplicate Download Scripts': [],
        'Duplicate Analysis Scripts': [],
        'Duplicate Testing Scripts': [],
        'Duplicate Training Scripts': [],
        'Legacy Utilities': [],
    }
    
    for script, reason in CLASSIFICATIONS['DEPRECATE'].items():
        if 'download' in script:
            categories['Duplicate Download Scripts'].append((script, reason))
        elif 'analysis' in script:
            categories['Duplicate Analysis Scripts'].append((script, reason))
        elif 'testing' in script:
            categories['Duplicate Testing Scripts'].append((script, reason))
        elif 'training' in script:
            categories['Duplicate Training Scripts'].append((script, reason))
        else:
            categories['Legacy Utilities'].append((script, reason))
    
    for category, scripts in categories.items():
        if scripts:
            report += f"\n{category}:\n" + "-"*80 + "\n\n"
            for script, reason in scripts:
                report += f"  ❌ {script}\n"
                report += f"     {reason}\n\n"
    
    report += f"\nTotal scripts to deprecate: {len(CLASSIFICATIONS['DEPRECATE'])}\n"
    report += "\nRecommendation:\n"
    report += "  1. Move to scripts/deprecated/ directory\n"
    report += "  2. Document in DEPRECATION_LOG.md with reason\n"
    report += "  3. Remove after 1 release cycle\n"
    return report


def generate_summary() -> str:
    """Generate overall summary."""
    total_classified = (
        len(CLASSIFICATIONS['INTEGRATE']) +
        len(CLASSIFICATIONS['STANDALONE']) +
        len(CLASSIFICATIONS['TEST']) +
        len(CLASSIFICATIONS['DEPRECATE']) +
        len(CLASSIFICATIONS['INVESTIGATE'])
    )
    
    report = f"""
================================================================================
CLASSIFICATION SUMMARY
================================================================================

Total Scripts Analyzed: {total_classified}

Classification Breakdown:
  ✅ INTEGRATE (Add to menu):        {len(CLASSIFICATIONS['INTEGRATE'])}
  📚 STANDALONE (Document only):      {len(CLASSIFICATIONS['STANDALONE'])}
  🧪 TEST (Keep as tests):            {len(CLASSIFICATIONS['TEST'])}
  ❌ DEPRECATE (Remove/archive):      {len(CLASSIFICATIONS['DEPRECATE'])}
  🔍 INVESTIGATE (Need review):       {len(CLASSIFICATIONS['INVESTIGATE'])}

Recommended Actions:
  1. Integrate {len(CLASSIFICATIONS['INTEGRATE'])} scripts into menu (high priority items first)
  2. Create docs/STANDALONE_TOOLS.md for {len(CLASSIFICATIONS['STANDALONE'])} research tools
  3. Move {len(CLASSIFICATIONS['DEPRECATE'])} scripts to deprecated/ folder
  4. Review {len(CLASSIFICATIONS['INVESTIGATE'])} scripts with team
  5. Keep {len(CLASSIFICATIONS['TEST'])} test scripts in tests/ (not in menu)

Impact on Coverage:
  Current:  9 of 130 scripts used (6.9%)
  After:    9 + {len(CLASSIFICATIONS['INTEGRATE'])} = {9 + len(CLASSIFICATIONS['INTEGRATE'])} scripts
  After deprecation: {9 + len(CLASSIFICATIONS['INTEGRATE'])} of {130 - len(CLASSIFICATIONS['DEPRECATE'])} scripts
  New Coverage: ~{(9 + len(CLASSIFICATIONS['INTEGRATE'])) / (130 - len(CLASSIFICATIONS['DEPRECATE'])) * 100:.1f}%
"""
    return report


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Generate full analysis report."""
    print("="*80)
    print("KINETRA SCRIPT CLASSIFICATION ANALYSIS")
    print("="*80)
    
    # Generate reports
    summary = generate_summary()
    integration_plan = generate_integration_plan()
    standalone_docs = generate_standalone_documentation()
    deprecation_list = generate_deprecation_list()
    
    # Print to console
    print(summary)
    print("\n" + integration_plan)
    print("\n" + standalone_docs)
    print("\n" + deprecation_list)
    
    # Investigate list
    if CLASSIFICATIONS['INVESTIGATE']:
        print("\n" + "="*80)
        print("SCRIPTS REQUIRING INVESTIGATION")
        print("="*80 + "\n")
        for script, note in CLASSIFICATIONS['INVESTIGATE'].items():
            print(f"🔍 {script}")
            print(f"   {note}\n")
    
    # Save to file
    output_dir = Path("docs")
    output_dir.mkdir(exist_ok=True)
    
    report_file = output_dir / "SCRIPT_CLASSIFICATION_ANALYSIS.md"
    with open(report_file, 'w') as f:
        f.write("# Kinetra Script Classification Analysis\n\n")
        f.write("Generated from audit of 121 unused scripts.\n\n")
        f.write(summary)
        f.write("\n\n")
        f.write(integration_plan)
        f.write("\n\n")
        f.write(standalone_docs)
        f.write("\n\n")
        f.write(deprecation_list)
        
        if CLASSIFICATIONS['INVESTIGATE']:
            f.write("\n\n## Scripts Requiring Investigation\n\n")
            for script, note in CLASSIFICATIONS['INVESTIGATE'].items():
                f.write(f"- **{script}**: {note}\n")
    
    print(f"\n📄 Full report saved to: {report_file}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
