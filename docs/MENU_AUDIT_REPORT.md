
================================================================================
MENU SYSTEM AUDIT REPORT
================================================================================

📊 SUMMARY
================================================================================

Total Menu Functions:    24
Total Script Calls:      12
Available Scripts:       171
Scripts Called by Menu:  8
Unused Scripts:          163
Missing Scripts:         0

Coverage: 4.7%

================================================================================
MENU STRUCTURE
================================================================================

Menu Functions (12):
  • show_backtesting_menu: Show backtesting menu.
  • show_data_management_menu: Show data management menu.
  • show_data_summary: Show data summary.
  • show_exploration_menu: Show exploration testing menu.
  • show_live_testing_guide: Show live testing guide.
  • show_live_testing_menu: Show live testing menu.
  • show_main_menu: Show main menu with context awareness.
  • show_performance_metrics: Show performance metrics.
  • show_progress_message: Show progress message with counter.
  • show_recent_results: Show recent test results.
  • show_system_health: Show current system health.
  • show_system_status_menu: Show system status and health.

Action Functions (12):
  • run_agent_comparison: Run agent comparison.
  • run_comparative_analysis: Run comparative analysis of multiple strategies.
  • run_custom_backtest: Run custom backtest with full configuration.
  • run_custom_exploration: Run custom exploration with user configuration.
  • run_demo_account_testing: Run demo account testing.
  • run_measurement_analysis: Run measurement impact analysis.
  • run_monte_carlo_validation: Run Monte Carlo validation.
  • run_quick_backtest: Run quick backtest using exploration results.
  • run_quick_exploration: Run quick exploration with preset configuration.
  • run_scientific_discovery: Run scientific discovery suite.
  • run_virtual_trading: Run virtual/paper trading test.
  • run_walk_forward_testing: Run walk-forward testing.

================================================================================
SCRIPT CALLS (Menu → Scripts)
================================================================================

  [✓] run_interruptible_subprocess → scripts/download/select_metaapi_account.py
  [✓] run_interruptible_subprocess → scripts/training/explore_compare_agents.py
  [✓] run_interruptible_subprocess → scripts/testing/run_live_test.py
  [✓] run_interruptible_subprocess → scripts/testing/run_live_test.py
  [✓] run_interruptible_subprocess → scripts/download/download_interactive.py
  [✓] run_interruptible_subprocess → scripts/download/check_and_fill_data.py
  [✓] run_interruptible_subprocess → scripts/download/check_data_integrity.py
  [✓] run_interruptible_subprocess → scripts/download/prepare_data.py
  [✓] run_interruptible_subprocess → scripts/download/backup_data.py
  [✓] run_interruptible_subprocess → scripts/download/download_interactive.py
  [✓] run_interruptible_subprocess → scripts/download/check_data_integrity.py
  [✓] run_interruptible_subprocess → scripts/download/prepare_data.py

================================================================================
COVERAGE BY CATEGORY
================================================================================

analysis/:
  Total:  25
  Called: 0 (0%)
  Unused: 25
  ✗ Unused: analyze_asymmetric_rewards, analyze_berserker_context, analyze_direction, analyze_directional_tension, analyze_energy, analyze_energy_both, analyze_energy_capture, analyze_extended_physics, analyze_fat_candles, analyze_flow_entropy, analyze_per_bar_energy, analyze_reversal, analyze_reynolds_continuation, analyze_trade_management, analyze_triggers, analyze_volatility_estimators, debug_csv, pathfinder_deep_dive, quick_results, superpot_by_class, superpot_complete, superpot_dsp_driven, superpot_empirical, superpot_explorer, superpot_physics

download/:
  Total:  18
  Called: 6 (33%)
  Unused: 12
  ✓ Used: backup_data, check_and_fill_data, check_data_integrity, download_interactive, prepare_data, select_metaapi_account
  ✗ Unused: convert_mt5_format, download_market_data, download_metaapi, download_mt5_data, extract_mt5_specs, fetch_broker_spec_from_metaapi, load_all_symbols, metaapi_bulk_download, metaapi_sync, parallel_data_prep, prepare_exploration_data, standardize_data_cutoff

exploration/:
  Total:  6
  Called: 0 (0%)
  Unused: 6
  ✗ Unused: rl_exploration_framework, rl_exploration_framework_agents, run_comprehensive_exploration, run_exploration_heartbeat, specialist_agents, tripleganger_system

scripts/:
  Total:  42
  Called: 0 (0%)
  Unused: 42
  ✗ Unused: audit_data_coverage, audit_menu_system, backtest_compare, backtest_full, backtest_risk, backtest_specialists, backtest_universal, backup_data, batch_backtest, benchmark_performance, branch_manager, cache_manager, classify_unused_scripts, consolidate_data, dashboard, demo_modular_execution, detect_silent_failures, devops_manager, explore_compare_agents, explore_full, explore_measurements, explore_policies, explore_risk, explore_stacking, explore_universal, fix_silent_failures, hunger_games_mvp, kientra_alpha_pipeline, lint_rules, master_workflow, monitor_daemon, optimize_full, optimize_params, optimize_replay, optimize_risk, run_exhaustive_tests, run_hpo, run_local, run_predictor, silent_failure_workflow, train, vectorization_linter

servers/:
  Total:  1
  Called: 0 (0%)
  Unused: 1
  ✗ Unused: mt5_bridge_server

setup/:
  Total:  1
  Called: 0 (0%)
  Unused: 1
  ✗ Unused: check_gpu

testing/:
  Total:  60
  Called: 1 (2%)
  Unused: 59
  ✓ Used: run_live_test
  ✗ Unused: AUTOMATED_AUDIT_FIX, batch_backtest, conftest, continuous_fix_pipeline, continuous_menu_test, demo_backtest_improvements, example_testing_framework, exercise_menu_continuous, exercise_menu_with_real_data, integrate_realistic_backtest, multi_tf_test, phase2_validation, rl_backtest, run_comprehensive_backtest, run_exploration_backtest, run_full_backtest, run_physics_backtest, run_scientific_testing, test_backtest_numerical_validation, test_backtest_trend, test_berserker_strategy, test_doppelganger_triad, test_e2e_symbols_timeframes, test_end_to_end, test_energy_recovery_hypotheses, test_experience_replay, test_exploration_strategies, test_framework_integration, test_freeze_zones, test_friction_costs, test_grafana_export, test_infrastructure_modules, test_marginal_gains, test_menu, test_metaapi_auth, test_mt5_authentication, test_mt5_friction, test_mt5_logger, test_mt5_vantage_full, test_multi_instrument, test_numerical_safety, test_p0_p5_integration, test_parallel_performance, test_performance_module, test_physics_demo, test_portfolio_health, test_real_data_backtest, test_regime_filtering, test_sac, test_strategies, test_trade_lifecycle, test_trade_lifecycle_real_data, test_transaction_log, unified_test_framework, validate_btc_h1_layer1, validate_mql5_compliance, validate_theorems, validate_thesis, verify_calculations

training/:
  Total:  18
  Called: 1 (6%)
  Unused: 17
  ✓ Used: explore_compare_agents
  ✗ Unused: demo_continual_learning, explore_interactive, explore_specialization, explore_universal, explorer_standalone, monitor_training, pathfinder_explore, quick_rl_test, run_exploration_batch, train_berserker, train_fast_multi, train_rl, train_rl_gpu, train_rl_physics, train_sniper, train_triad, train_with_metrics

================================================================================
⚠️  DEADWEIGHT (Unused Scripts)
================================================================================

These scripts exist but are not called by the menu:

  • scripts/analysis/analyze_asymmetric_rewards.py
  • scripts/analysis/analyze_berserker_context.py
  • scripts/analysis/analyze_direction.py
  • scripts/analysis/analyze_directional_tension.py
  • scripts/analysis/analyze_energy.py
  • scripts/analysis/analyze_energy_both.py
  • scripts/analysis/analyze_energy_capture.py
  • scripts/analysis/analyze_extended_physics.py
  • scripts/analysis/analyze_fat_candles.py
  • scripts/analysis/analyze_flow_entropy.py
  • scripts/analysis/analyze_per_bar_energy.py
  • scripts/analysis/analyze_reversal.py
  • scripts/analysis/analyze_reynolds_continuation.py
  • scripts/analysis/analyze_trade_management.py
  • scripts/analysis/analyze_triggers.py
  • scripts/analysis/analyze_volatility_estimators.py
  • scripts/analysis/debug_csv.py
  • scripts/analysis/pathfinder_deep_dive.py
  • scripts/analysis/quick_results.py
  • scripts/analysis/superpot_by_class.py
  • scripts/analysis/superpot_complete.py
  • scripts/analysis/superpot_dsp_driven.py
  • scripts/analysis/superpot_empirical.py
  • scripts/analysis/superpot_explorer.py
  • scripts/analysis/superpot_physics.py
  • scripts/audit_data_coverage.py
  • scripts/audit_menu_system.py
  • scripts/backup_data.py
  • scripts/batch_backtest.py
  • scripts/benchmark_performance.py
  • scripts/branch_manager.py
  • scripts/cache_manager.py
  • scripts/classify_unused_scripts.py
  • scripts/consolidate_data.py
  • scripts/dashboard.py
  • scripts/demo_modular_execution.py
  • scripts/detect_silent_failures.py
  • scripts/devops_manager.py
  • scripts/download/convert_mt5_format.py
  • scripts/download/download_market_data.py
  • scripts/download/download_metaapi.py
  • scripts/download/download_mt5_data.py
  • scripts/download/extract_mt5_specs.py
  • scripts/download/fetch_broker_spec_from_metaapi.py
  • scripts/download/load_all_symbols.py
  • scripts/download/metaapi_bulk_download.py
  • scripts/download/metaapi_sync.py
  • scripts/download/parallel_data_prep.py
  • scripts/download/prepare_exploration_data.py
  • scripts/download/standardize_data_cutoff.py
  • scripts/exploration/rl_exploration_framework.py
  • scripts/exploration/rl_exploration_framework_agents.py
  • scripts/exploration/run_comprehensive_exploration.py
  • scripts/exploration/run_exploration_heartbeat.py
  • scripts/exploration/specialist_agents.py
  • scripts/exploration/tripleganger_system.py
  • scripts/fix_silent_failures.py
  • scripts/hunger_games_mvp.py
  • scripts/kientra_alpha_pipeline.py
  • scripts/lint_rules.py
  • scripts/master_workflow.py
  • scripts/monitor_daemon.py
  • scripts/run_exhaustive_tests.py
  • scripts/run_hpo.py
  • scripts/run_local.py
  • scripts/run_predictor.py
  • scripts/servers/mt5_bridge_server.py
  • scripts/setup/check_gpu.py
  • scripts/silent_failure_workflow.py
  • scripts/testing/AUTOMATED_AUDIT_FIX.py
  • scripts/testing/batch_backtest.py
  • scripts/testing/conftest.py
  • scripts/testing/continuous_fix_pipeline.py
  • scripts/testing/continuous_menu_test.py
  • scripts/testing/demo_backtest_improvements.py
  • scripts/testing/example_testing_framework.py
  • scripts/testing/exercise_menu_continuous.py
  • scripts/testing/exercise_menu_with_real_data.py
  • scripts/testing/integrate_realistic_backtest.py
  • scripts/testing/multi_tf_test.py
  • scripts/testing/phase2_validation.py
  • scripts/testing/rl_backtest.py
  • scripts/testing/run_comprehensive_backtest.py
  • scripts/testing/run_exploration_backtest.py
  • scripts/testing/run_full_backtest.py
  • scripts/testing/run_physics_backtest.py
  • scripts/testing/run_scientific_testing.py
  • scripts/testing/scripts/backtest_compare.py
  • scripts/testing/scripts/backtest_full.py
  • scripts/testing/scripts/backtest_risk.py
  • scripts/testing/scripts/backtest_specialists.py
  • scripts/testing/scripts/backtest_universal.py
  • scripts/testing/scripts/explore_compare_agents.py
  • scripts/testing/scripts/explore_full.py
  • scripts/testing/scripts/explore_measurements.py
  • scripts/testing/scripts/explore_policies.py
  • scripts/testing/scripts/explore_risk.py
  • scripts/testing/scripts/explore_stacking.py
  • scripts/testing/scripts/explore_universal.py
  • scripts/testing/scripts/optimize_full.py
  • scripts/testing/scripts/optimize_params.py
  • scripts/testing/scripts/optimize_replay.py
  • scripts/testing/scripts/optimize_risk.py
  • scripts/testing/test_backtest_numerical_validation.py
  • scripts/testing/test_backtest_trend.py
  • scripts/testing/test_berserker_strategy.py
  • scripts/testing/test_doppelganger_triad.py
  • scripts/testing/test_e2e_symbols_timeframes.py
  • scripts/testing/test_end_to_end.py
  • scripts/testing/test_energy_recovery_hypotheses.py
  • scripts/testing/test_experience_replay.py
  • scripts/testing/test_exploration_strategies.py
  • scripts/testing/test_framework_integration.py
  • scripts/testing/test_freeze_zones.py
  • scripts/testing/test_friction_costs.py
  • scripts/testing/test_grafana_export.py
  • scripts/testing/test_infrastructure_modules.py
  • scripts/testing/test_marginal_gains.py
  • scripts/testing/test_menu.py
  • scripts/testing/test_metaapi_auth.py
  • scripts/testing/test_mt5_authentication.py
  • scripts/testing/test_mt5_friction.py
  • scripts/testing/test_mt5_logger.py
  • scripts/testing/test_mt5_vantage_full.py
  • scripts/testing/test_multi_instrument.py
  • scripts/testing/test_numerical_safety.py
  • scripts/testing/test_p0_p5_integration.py
  • scripts/testing/test_parallel_performance.py
  • scripts/testing/test_performance_module.py
  • scripts/testing/test_physics_demo.py
  • scripts/testing/test_portfolio_health.py
  • scripts/testing/test_real_data_backtest.py
  • scripts/testing/test_regime_filtering.py
  • scripts/testing/test_sac.py
  • scripts/testing/test_strategies.py
  • scripts/testing/test_trade_lifecycle.py
  • scripts/testing/test_trade_lifecycle_real_data.py
  • scripts/testing/test_transaction_log.py
  • scripts/testing/unified_test_framework.py
  • scripts/testing/validate_btc_h1_layer1.py
  • scripts/testing/validate_mql5_compliance.py
  • scripts/testing/validate_theorems.py
  • scripts/testing/validate_thesis.py
  • scripts/testing/verify_calculations.py
  • scripts/train.py
  • scripts/training/demo_continual_learning.py
  • scripts/training/explore_interactive.py
  • scripts/training/explore_specialization.py
  • scripts/training/explore_universal.py
  • scripts/training/explorer_standalone.py
  • scripts/training/monitor_training.py
  • scripts/training/pathfinder_explore.py
  • scripts/training/quick_rl_test.py
  • scripts/training/run_exploration_batch.py
  • scripts/training/train_berserker.py
  • scripts/training/train_fast_multi.py
  • scripts/training/train_rl.py
  • scripts/training/train_rl_gpu.py
  • scripts/training/train_rl_physics.py
  • scripts/training/train_sniper.py
  • scripts/training/train_triad.py
  • scripts/training/train_with_metrics.py
  • scripts/vectorization_linter.py

Recommendation: Review if these should be:
  1. Integrated into menu system
  2. Documented as standalone tools
  3. Removed as obsolete

================================================================================
💡 RECOMMENDATIONS
================================================================================

⚠️  HIGH DEADWEIGHT: Consider cleanup
📊 LOW COVERAGE: Many scripts not integrated




╔════════════════════════════════════════════════════════════════════════════╗
║                        KINETRA MENU SYSTEM MAP                             ║
╚════════════════════════════════════════════════════════════════════════════╝

MAIN MENU
│
├─[1] Login & Authentication
│   ├─ Select MetaAPI Account → scripts/download/select_metaapi_account.py
│   └─ Test Connection (inline)
│
├─[2] Exploration Testing
│   ├─ Quick Exploration → run_comprehensive_exploration.py
│   ├─ Custom Exploration → run_comprehensive_exploration.py
│   ├─ Scientific Discovery → scripts/testing/run_scientific_testing.py
│   ├─ Agent Comparison → scripts/training/explore_compare_agents.py
│   └─ Measurement Analysis (NOT IMPLEMENTED)
│
├─[3] Backtesting
│   ├─ Quick Backtest → scripts/testing/run_comprehensive_backtest.py
│   ├─ Custom Backtesting → scripts/testing/run_comprehensive_backtest.py
│   ├─ Monte Carlo → scripts/testing/run_comprehensive_backtest.py
│   ├─ Walk-Forward (NOT IMPLEMENTED)
│   └─ Comparative Analysis (NOT IMPLEMENTED)
│
├─[4] Data Management
│   ├─ Auto-Download (inline)
│   ├─ Manual Download → scripts/download/download_interactive.py
│   ├─ Check & Fill → scripts/download/check_and_fill_data.py
│   ├─ Data Integrity → scripts/download/check_data_integrity.py
│   ├─ Prepare Data → scripts/download/prepare_data.py
│   └─ Backup & Restore → scripts/download/backup_data.py
│
└─[5] System Status & Health
    ├─ Current Health (inline)
    ├─ Recent Results (inline)
    ├─ Data Summary (inline)
    └─ Performance Metrics (inline)

Legend:
  → Script call
  (inline) = Implemented directly in menu
  (NOT IMPLEMENTED) = Placeholder, shows warning
