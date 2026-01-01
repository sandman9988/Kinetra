#!/bin/bash
# Script to organize scripts into subdirectories

# Download scripts
for f in download_*.py metaapi_*.py check_and_fill_data.py backup_data.py parallel_data_prep.py prepare_data.py prepare_exploration_data.py standardize_data_cutoff.py check_data_integrity.py load_all_symbols.py convert_mt5_format.py; do
  [ -f "$f" ] && git mv "$f" download/ 2>/dev/null || true
done

# Analysis scripts
for f in analyze_*.py pathfinder_deep_dive.py superpot_physics.py; do
  [ -f "$f" ] && git mv "$f" analysis/ 2>/dev/null || true
done

# Training scripts
for f in train_*.py explore_*.py explorer_standalone.py run_exploration_heartbeat.py monitor_training.py; do
  [ -f "$f" ] && git mv "$f" training/ 2>/dev/null || true
done

# Setup scripts
for f in setup_*.sh setup_*.ps1 setup_*.bat; do
  [ -f "$f" ] && git mv "$f" setup/ 2>/dev/null || true
done

# Testing/backtest scripts  
for f in test_*.py run_*backtest*.py batch_backtest.py demo_backtest*.py integrate_realistic_backtest.py rl_backtest.py unified_test_framework.py run_scientific_testing.py run_comprehensive_*.py example_testing_framework.py; do
  [ -f "$f" ] && git mv "$f" testing/ 2>/dev/null || true
done

echo "Script organization complete!"
