# Kinetra Script Classification Analysis

Generated from audit of 121 unused scripts.


================================================================================
CLASSIFICATION SUMMARY
================================================================================

Total Scripts Analyzed: 73

Classification Breakdown:
  ✅ INTEGRATE (Add to menu):        10
  📚 STANDALONE (Document only):      25
  🧪 TEST (Keep as tests):            12
  ❌ DEPRECATE (Remove/archive):      22
  🔍 INVESTIGATE (Need review):       4

Recommended Actions:
  1. Integrate 10 scripts into menu (high priority items first)
  2. Create docs/STANDALONE_TOOLS.md for 25 research tools
  3. Move 22 scripts to deprecated/ folder
  4. Review 4 scripts with team
  5. Keep 12 test scripts in tests/ (not in menu)

Impact on Coverage:
  Current:  9 of 130 scripts used (6.9%)
  After:    9 + 10 = 19 scripts
  After deprecation: 19 of 108 scripts
  New Coverage: ~17.6%



================================================================================
SCRIPT INTEGRATION PLAN
================================================================================

Scripts recommended for integration into menu system, organized by priority:

HIGH PRIORITY (Implement First):
================================================================================

📌 scripts/analysis/quick_results.py
   Menu Location: Exploration Testing
   Action Name: View Recent Results
   Reason: Quick results viewer - useful after exploration runs

📌 scripts/analysis/analyze_energy.py
   Menu Location: Exploration Testing → Measurement Analysis
   Action Name: Analyze Energy Metrics
   Reason: Core physics analysis - aligns with first principles

📌 scripts/testing/run_full_backtest.py
   Menu Location: Backtesting
   Action Name: Full Backtest (All Instruments)
   Reason: Comprehensive backtest across all data

📌 scripts/dashboard.py
   Menu Location: System Status & Health
   Action Name: Launch Dashboard
   Reason: Visual dashboard for system monitoring


MEDIUM PRIORITY (Implement Second):
================================================================================

📋 scripts/analysis/analyze_asymmetric_rewards.py
   Menu Location: Exploration Testing → Measurement Analysis
   Action Name: Analyze Asymmetric Rewards
   Reason: Reward shaping analysis - critical for strategy evaluation

📋 scripts/training/train_rl.py
   Menu Location: Exploration Testing
   Action Name: Train RL Agent (Advanced)
   Reason: Main RL training script - advanced users

📋 scripts/training/monitor_training.py
   Menu Location: System Status & Health
   Action Name: Monitor Training Progress
   Reason: Real-time training monitoring

📋 scripts/testing/batch_backtest.py
   Menu Location: Backtesting
   Action Name: Batch Backtest (Multiple Strategies)
   Reason: Run multiple backtests in batch

📋 scripts/download/metaapi_sync.py
   Menu Location: Data Management
   Action Name: Sync with MetaAPI (Continuous)
   Reason: Keep data synchronized with broker


LOW PRIORITY (Nice to Have):
================================================================================

📌 scripts/benchmark_performance.py
   Menu Location: System Status & Health
   Action Name: Benchmark Performance
   Reason: System performance benchmarking


Total scripts to integrate: 10



================================================================================
STANDALONE TOOLS (Do Not Add to Menu)
================================================================================

These scripts are valuable research/utility tools but should NOT be added to
the menu. They should be documented for advanced users and researchers.


SuperPot Research:
--------------------------------------------------------------------------------

  scripts/analysis/superpot_complete.py
  → SuperPot: Complete exploration across all instruments

  scripts/analysis/superpot_by_class.py
  → SuperPot: Asset class-specific feature discovery

  scripts/analysis/superpot_explorer.py
  → SuperPot: Interactive feature explorer

  scripts/analysis/superpot_physics.py
  → SuperPot: Physics-focused feature discovery

  scripts/analysis/superpot_empirical.py
  → SuperPot: Empirical feature testing

  scripts/analysis/superpot_dsp_driven.py
  → SuperPot: DSP-driven feature extraction


Analysis Tools:
--------------------------------------------------------------------------------

  scripts/analysis/pathfinder_deep_dive.py
  → Deep analysis: Pathfinder strategy exploration

  scripts/analysis/analyze_trade_management.py
  → Analysis: Trade management strategies

  scripts/analysis/analyze_triggers.py
  → Analysis: Entry/exit trigger effectiveness

  scripts/analysis/debug_csv.py
  → Debug tool: CSV data inspection


Training Research:
--------------------------------------------------------------------------------

  scripts/training/explore_universal.py
  → Research: Universal agent exploration

  scripts/training/explore_specialization.py
  → Research: Agent specialization study

  scripts/training/explorer_standalone.py
  → Research: Standalone exploration framework

  scripts/training/pathfinder_explore.py
  → Research: Pathfinder strategy research

  scripts/training/train_berserker.py
  → Training: Berserker strategy agent

  scripts/training/train_triad.py
  → Training: Triad system (3-agent ensemble)

  scripts/training/train_sniper.py
  → Training: Sniper strategy agent


DevOps Tools:
--------------------------------------------------------------------------------

  scripts/master_workflow.py
  → Workflow: Master orchestration script

  scripts/silent_failure_workflow.py
  → Workflow: Silent failure detection pipeline

  scripts/devops_manager.py
  → DevOps: System management tool

  scripts/branch_manager.py
  → DevOps: Git branch management

  scripts/cache_manager.py
  → DevOps: Cache management utility


Data Engineering:
--------------------------------------------------------------------------------

  scripts/download/parallel_data_prep.py
  → Data: Parallel data preparation (advanced)

  scripts/download/prepare_exploration_data.py
  → Data: Exploration-specific preparation

  scripts/download/standardize_data_cutoff.py
  → Data: Standardize to common cutoff date


Total standalone tools: 25

Recommendation: Create docs/STANDALONE_TOOLS.md documenting these



================================================================================
SCRIPTS TO DEPRECATE/REMOVE
================================================================================

These scripts are legacy/obsolete and can be safely removed or archived.


Duplicate Download Scripts:
--------------------------------------------------------------------------------

  ❌ scripts/download/download_mt5_data.py
     Legacy: Replaced by download_interactive.py

  ❌ scripts/download/download_metaapi.py
     Legacy: Replaced by download_interactive.py

  ❌ scripts/download/download_market_data.py
     Legacy: Replaced by download_interactive.py

  ❌ scripts/download/convert_mt5_format.py
     Obsolete: MT5 format conversion (no longer needed)

  ❌ scripts/download/load_all_symbols.py
     Obsolete: Replaced by download_interactive.py symbol selection

  ❌ scripts/download/metaapi_bulk_download.py
     Legacy: Replaced by download_interactive.py


Duplicate Analysis Scripts:
--------------------------------------------------------------------------------

  ❌ scripts/analysis/analyze_energy_both.py
     Duplicate: Same as analyze_energy.py

  ❌ scripts/analysis/analyze_direction.py
     Obsolete: Functionality moved to measurements.py

  ❌ scripts/analysis/analyze_directional_tension.py
     Obsolete: Integrated into physics engine


Duplicate Testing Scripts:
--------------------------------------------------------------------------------

  ❌ scripts/testing/rl_backtest.py
     Duplicate: Use run_comprehensive_backtest.py

  ❌ scripts/testing/run_physics_backtest.py
     Duplicate: Use run_comprehensive_backtest.py

  ❌ scripts/testing/run_exploration_backtest.py
     Duplicate: Use run_comprehensive_backtest.py

  ❌ scripts/testing/integrate_realistic_backtest.py
     Obsolete: Functionality in backtest engine


Duplicate Training Scripts:
--------------------------------------------------------------------------------

  ❌ scripts/training/quick_rl_test.py
     Obsolete: Use explore_compare_agents.py

  ❌ scripts/training/train_fast_multi.py
     Duplicate: Use train_rl.py with --parallel flag

  ❌ scripts/training/demo_continual_learning.py
     Demo: Not production-ready


Legacy Utilities:
--------------------------------------------------------------------------------

  ❌ scripts/detect_silent_failures.py
     Legacy: Functionality now in testing framework

  ❌ scripts/fix_silent_failures.py
     Legacy: Manual tool no longer needed

  ❌ scripts/demo_modular_execution.py
     Demo: Example script only

  ❌ scripts/run_local.py
     Obsolete: Use kinetra_menu.py instead

  ❌ scripts/train.py
     Legacy: Use scripts/training/train_rl.py

  ❌ scripts/setup/check_gpu.py
     Utility: GPU check (rarely needed, can run manually)


Total scripts to deprecate: 22

Recommendation:
  1. Move to scripts/deprecated/ directory
  2. Document in DEPRECATION_LOG.md with reason
  3. Remove after 1 release cycle


## Scripts Requiring Investigation

- **scripts/run_predictor.py**: Unknown: Need to review purpose
- **scripts/monitor_daemon.py**: Unknown: Daemon monitoring - check if still used
- **scripts/download/extract_mt5_specs.py**: Unknown: MT5 spec extraction - still relevant?
- **scripts/download/fetch_broker_spec_from_metaapi.py**: Unknown: Broker spec fetching - still used?
